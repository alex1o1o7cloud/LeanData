import Mathlib

namespace NUMINAMATH_CALUDE_puppy_and_food_cost_l3860_386070

/-- Calculates the total cost of a puppy and food for a given number of weeks -/
def totalCost (puppyCost : ℚ) (foodPerDay : ℚ) (daysSupply : ℕ) (cupPerBag : ℚ) (bagCost : ℚ) : ℚ :=
  let totalDays : ℕ := daysSupply
  let totalFood : ℚ := (totalDays : ℚ) * foodPerDay
  let bagsNeeded : ℚ := totalFood / cupPerBag
  let foodCost : ℚ := bagsNeeded * bagCost
  puppyCost + foodCost

/-- Theorem stating that the total cost of a puppy and food for 3 weeks is $14 -/
theorem puppy_and_food_cost :
  totalCost 10 (1/3) 21 (7/2) 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_puppy_and_food_cost_l3860_386070


namespace NUMINAMATH_CALUDE_correct_answer_l3860_386052

theorem correct_answer (x : ℝ) (h : 3 * x = 90) : x - 30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l3860_386052


namespace NUMINAMATH_CALUDE_bubble_gum_cost_l3860_386034

theorem bubble_gum_cost (total_cost : ℕ) (total_pieces : ℕ) (cost_per_piece : ℕ) : 
  total_cost = 2448 → 
  total_pieces = 136 → 
  total_cost = total_pieces * cost_per_piece → 
  cost_per_piece = 18 := by
  sorry

end NUMINAMATH_CALUDE_bubble_gum_cost_l3860_386034


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3860_386084

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let a : ℚ := 1
  let b : ℚ := -2/3
  let r : ℕ := 6
  (Nat.choose n r) * a^(n-r) * b^r = 1792 := by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3860_386084


namespace NUMINAMATH_CALUDE_highest_score_is_143_l3860_386029

/-- Represents a batsman's performance in a cricket tournament --/
structure BatsmanPerformance where
  totalInnings : ℕ
  averageRuns : ℚ
  highestScore : ℕ
  lowestScore : ℕ
  centuryCount : ℕ

/-- Theorem stating the highest score of the batsman given the conditions --/
theorem highest_score_is_143 (b : BatsmanPerformance) : 
  b.totalInnings = 46 ∧
  b.averageRuns = 58 ∧
  b.highestScore - b.lowestScore = 150 ∧
  (b.totalInnings * b.averageRuns - b.highestScore - b.lowestScore) / (b.totalInnings - 2) = b.averageRuns ∧
  b.centuryCount = 5 ∧
  ∀ score, score ≠ b.highestScore → score < 100 →
  b.highestScore = 143 := by
  sorry


end NUMINAMATH_CALUDE_highest_score_is_143_l3860_386029


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3860_386041

def M : Set ℝ := {x | x^2 - 3*x = 0}
def N : Set ℝ := {x | x > -1}

theorem intersection_of_M_and_N : M ∩ N = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3860_386041


namespace NUMINAMATH_CALUDE_zhang_san_correct_probability_l3860_386093

theorem zhang_san_correct_probability :
  let total_questions : ℕ := 4
  let questions_with_ideas : ℕ := 3
  let questions_unclear : ℕ := 1
  let prob_correct_with_idea : ℚ := 3/4
  let prob_correct_when_unclear : ℚ := 1/4
  let prob_selecting_question_with_idea : ℚ := questions_with_ideas / total_questions
  let prob_selecting_question_unclear : ℚ := questions_unclear / total_questions

  prob_selecting_question_with_idea * prob_correct_with_idea +
  prob_selecting_question_unclear * prob_correct_when_unclear = 5/8 :=
by sorry

end NUMINAMATH_CALUDE_zhang_san_correct_probability_l3860_386093


namespace NUMINAMATH_CALUDE_quadratic_transformation_impossibility_l3860_386050

/-- Represents a quadratic trinomial ax^2 + bx + c -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic trinomial -/
def discriminant (f : QuadraticTrinomial) : ℝ :=
  f.b^2 - 4*f.a*f.c

/-- Represents the allowed operations on quadratic trinomials -/
inductive QuadraticOperation
  | op1 : QuadraticOperation  -- f(x) → x^2 f(1 + 1/x)
  | op2 : QuadraticOperation  -- f(x) → (x-1)^2 f(1/(x-1))

/-- Applies a quadratic operation to a quadratic trinomial -/
def applyOperation (f : QuadraticTrinomial) (op : QuadraticOperation) : QuadraticTrinomial :=
  match op with
  | QuadraticOperation.op1 => QuadraticTrinomial.mk f.a (2*f.a + f.b) (f.a + f.b + f.c)
  | QuadraticOperation.op2 => QuadraticTrinomial.mk f.c (f.b - 2*f.c) (f.a - f.b + f.c)

/-- Theorem stating that it's impossible to transform x^2 + 4x + 3 into x^2 + 10x + 9
    using only the allowed operations -/
theorem quadratic_transformation_impossibility :
  ∀ (ops : List QuadraticOperation),
  let f := QuadraticTrinomial.mk 1 4 3
  let g := QuadraticTrinomial.mk 1 10 9
  let result := ops.foldl applyOperation f
  result ≠ g := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_impossibility_l3860_386050


namespace NUMINAMATH_CALUDE_min_integral_abs_exp_minus_a_l3860_386002

theorem min_integral_abs_exp_minus_a :
  let f (a : ℝ) := ∫ x in (0 : ℝ)..1, |Real.exp (-x) - a|
  ∃ m : ℝ, (∀ a : ℝ, f a ≥ m) ∧ (∃ a : ℝ, f a = m) ∧ m = 1 - 2 * Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_min_integral_abs_exp_minus_a_l3860_386002


namespace NUMINAMATH_CALUDE_deepak_age_l3860_386026

/-- Given the ratio of Arun's age to Deepak's age and Arun's future age, 
    prove Deepak's current age -/
theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l3860_386026


namespace NUMINAMATH_CALUDE_percentage_problem_l3860_386030

/-- Given that 15% of 40 is greater than y% of 16 by 2, prove that y = 25 -/
theorem percentage_problem (y : ℝ) : 
  (0.15 * 40 = y / 100 * 16 + 2) → y = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3860_386030


namespace NUMINAMATH_CALUDE_double_mean_value_function_range_l3860_386020

/-- A function f is a double mean value function on [a,b] if there exist
    two distinct points x₁ and x₂ in (a,b) such that
    f''(x₁) = f''(x₂) = (f(b) - f(a)) / (b - a) -/
def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
  (deriv^[2] f) x₁ = (f b - f a) / (b - a) ∧
  (deriv^[2] f) x₂ = (f b - f a) / (b - a)

/-- The main theorem -/
theorem double_mean_value_function_range :
  ∀ m : ℝ, is_double_mean_value_function (fun x ↦ x^3 - 6/5 * x^2) 0 m →
  3/5 < m ∧ m ≤ 6/5 := by sorry

end NUMINAMATH_CALUDE_double_mean_value_function_range_l3860_386020


namespace NUMINAMATH_CALUDE_product_of_r_values_l3860_386027

theorem product_of_r_values : ∃ (r₁ r₂ : ℝ), 
  (∀ x : ℝ, x ≠ 0 → (1 / (3 * x) = (r₁ - x) / 8 ↔ 1 / (3 * x) = (r₂ - x) / 8)) ∧ 
  (∀ r : ℝ, (∃! x : ℝ, x ≠ 0 ∧ 1 / (3 * x) = (r - x) / 8) → (r = r₁ ∨ r = r₂)) ∧
  r₁ * r₂ = -32/3 :=
sorry

end NUMINAMATH_CALUDE_product_of_r_values_l3860_386027


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3860_386061

def proposition (x : ℕ) : Prop := (1/2:ℝ)^x ≤ 1/2

theorem negation_of_proposition :
  (¬ ∀ (x : ℕ), x > 0 → proposition x) ↔ (∃ (x : ℕ), x > 0 ∧ (1/2:ℝ)^x > 1/2) :=
sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3860_386061


namespace NUMINAMATH_CALUDE_f_lower_bound_and_equality_l3860_386044

def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem f_lower_bound_and_equality (a b : ℝ) 
  (h : (1 / (2 * a)) + (2 / b) = 1) :
  (∀ x, f x a b ≥ 9/2) ∧
  (∃ x, f x a b = 9/2 → a = 3/2 ∧ b = 3) := by
sorry

end NUMINAMATH_CALUDE_f_lower_bound_and_equality_l3860_386044


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3860_386040

/-- A hyperbola in the Cartesian coordinate system -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x^2 / a^2) - (y^2 / b^2) = 1 -/
  equation : ℝ → ℝ → Prop

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  /-- The equation of the parabola -/
  equation : ℝ → ℝ → Prop

/-- The focus of a parabola -/
def parabola_focus (p : Parabola) : ℝ × ℝ := sorry

/-- The right focus of a hyperbola -/
def hyperbola_right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Theorem: Given a hyperbola C with its center at the origin, passing through (1, 0),
    and its right focus coinciding with the focus of y^2 = 8x, 
    the standard equation of C is x^2 - y^2/3 = 1 -/
theorem hyperbola_equation 
  (C : Hyperbola)
  (center_origin : C.equation 0 0)
  (passes_through_1_0 : C.equation 1 0)
  (p : Parabola)
  (p_eq : p.equation = fun x y ↦ y^2 = 8*x)
  (focus_coincide : hyperbola_right_focus C = parabola_focus p) :
  C.equation = fun x y ↦ x^2 - y^2/3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3860_386040


namespace NUMINAMATH_CALUDE_seven_digit_number_product_l3860_386059

theorem seven_digit_number_product : ∃ (x y : ℕ), 
  (1000000 ≤ x ∧ x < 10000000) ∧ 
  (1000000 ≤ y ∧ y < 10000000) ∧ 
  (10^7 * x + y = 3 * x * y) ∧ 
  (x = 1666667 ∧ y = 3333334) := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_number_product_l3860_386059


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l3860_386066

theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a k h : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l3860_386066


namespace NUMINAMATH_CALUDE_parabola_vertex_l3860_386036

/-- Given a quadratic function f(x) = -x^2 + cx + d where c and d are real numbers,
    and the solution to f(x) ≤ 0 is (-∞, -4] ∪ [6, ∞),
    prove that the vertex of the parabola is (1, 25). -/
theorem parabola_vertex (c d : ℝ) :
  (∀ x, -x^2 + c*x + d ≤ 0 ↔ x ≤ -4 ∨ x ≥ 6) →
  ∃ (vertex : ℝ × ℝ), vertex = (1, 25) ∧
    ∀ x, -x^2 + c*x + d ≤ -(x - vertex.1)^2 + vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3860_386036


namespace NUMINAMATH_CALUDE_min_value_ab_l3860_386079

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + 9 * b + 7) :
  a * b ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l3860_386079


namespace NUMINAMATH_CALUDE_absolute_value_inequality_implies_m_equals_negative_four_l3860_386056

theorem absolute_value_inequality_implies_m_equals_negative_four (m : ℝ) :
  (∀ x : ℝ, |2*x - m| ≤ |3*x + 6|) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_implies_m_equals_negative_four_l3860_386056


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3860_386080

-- Define the sets P and Q
def P : Set ℝ := {x | Real.log x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the open interval (1, 2)
def open_interval_one_two : Set ℝ := {x | 1 < x ∧ x < 2}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = open_interval_one_two := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3860_386080


namespace NUMINAMATH_CALUDE_state_tax_calculation_l3860_386068

/-- Calculate the state tax for a partial-year resident -/
theorem state_tax_calculation 
  (months_resident : ℕ) 
  (taxable_income : ℝ) 
  (tax_rate : ℝ) : 
  months_resident = 9 → 
  taxable_income = 42500 → 
  tax_rate = 0.04 → 
  (months_resident : ℝ) / 12 * taxable_income * tax_rate = 1275 := by
  sorry

end NUMINAMATH_CALUDE_state_tax_calculation_l3860_386068


namespace NUMINAMATH_CALUDE_cubic_real_root_l3860_386005

/-- Given a cubic polynomial ax³ + 3x² + bx - 125 = 0 where a and b are real numbers,
    if -3 - 4i is a root of this polynomial, then 5 is the real root of the polynomial. -/
theorem cubic_real_root (a b : ℝ) :
  (∃ (z : ℂ), z = -3 - 4*I ∧ a * z^3 + 3 * z^2 + b * z - 125 = 0) →
  (∃ (x : ℝ), x = 5 ∧ a * x^3 + 3 * x^2 + b * x - 125 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_real_root_l3860_386005


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3860_386033

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3860_386033


namespace NUMINAMATH_CALUDE_cube_surface_area_l3860_386007

/-- The surface area of a cube with side length 20 cm is 2400 square centimeters. -/
theorem cube_surface_area : 
  let side_length : ℝ := 20
  6 * side_length ^ 2 = 2400 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3860_386007


namespace NUMINAMATH_CALUDE_p_money_theorem_l3860_386031

theorem p_money_theorem (p q r : ℚ) : 
  (p = (1/6 * p + 1/6 * p) + 32) → p = 48 := by
  sorry

end NUMINAMATH_CALUDE_p_money_theorem_l3860_386031


namespace NUMINAMATH_CALUDE_first_purchase_quantities_second_purchase_max_profit_new_selling_price_B_l3860_386013

-- Definitions based on the problem conditions
def purchase_price_A : ℝ := 30
def purchase_price_B : ℝ := 25
def selling_price_A : ℝ := 45
def selling_price_B : ℝ := 37
def total_keychains : ℕ := 30
def total_cost : ℝ := 850
def second_purchase_total : ℕ := 80
def second_purchase_max_cost : ℝ := 2200
def original_daily_sales_B : ℕ := 4
def price_reduction_effect : ℝ := 2

-- Part 1
theorem first_purchase_quantities (x y : ℕ) :
  purchase_price_A * x + purchase_price_B * y = total_cost ∧
  x + y = total_keychains →
  x = 20 ∧ y = 10 := by sorry

-- Part 2
theorem second_purchase_max_profit (m : ℕ) :
  m ≤ 40 →
  ∃ (w : ℝ), w = 3 * m + 960 ∧
  w ≤ 1080 ∧
  (m = 40 → w = 1080) := by sorry

-- Part 3
theorem new_selling_price_B (a : ℝ) :
  (a - purchase_price_B) * (78 - 2 * a) = 90 →
  a = 30 ∨ a = 34 := by sorry

end NUMINAMATH_CALUDE_first_purchase_quantities_second_purchase_max_profit_new_selling_price_B_l3860_386013


namespace NUMINAMATH_CALUDE_ball_pricing_theorem_l3860_386090

/-- Represents the price and quantity of basketballs and volleyballs -/
structure BallPrices where
  basketball_price : ℕ
  volleyball_price : ℕ
  basketball_quantity : ℕ
  volleyball_quantity : ℕ

/-- Conditions of the ball purchasing problem -/
def ball_conditions (prices : BallPrices) : Prop :=
  prices.basketball_quantity + prices.volleyball_quantity = 20 ∧
  2 * prices.basketball_price + 3 * prices.volleyball_price = 190 ∧
  3 * prices.basketball_price = 5 * prices.volleyball_price

/-- Cost calculation for a given quantity of basketballs and volleyballs -/
def total_cost (prices : BallPrices) (b_qty : ℕ) (v_qty : ℕ) : ℕ :=
  b_qty * prices.basketball_price + v_qty * prices.volleyball_price

/-- Theorem stating the correct prices and most cost-effective plan -/
theorem ball_pricing_theorem (prices : BallPrices) :
  ball_conditions prices →
  prices.basketball_price = 50 ∧
  prices.volleyball_price = 30 ∧
  (∀ b v, b + v = 20 → b ≥ 8 → total_cost prices b v ≤ 800 →
    total_cost prices 8 12 ≤ total_cost prices b v) :=
sorry

end NUMINAMATH_CALUDE_ball_pricing_theorem_l3860_386090


namespace NUMINAMATH_CALUDE_andrew_work_days_l3860_386096

/-- Given that Andrew worked 2.5 hours each day and a total of 7.5 hours,
    prove that he spent 3 days working on the report. -/
theorem andrew_work_days (hours_per_day : ℝ) (total_hours : ℝ) 
    (h1 : hours_per_day = 2.5)
    (h2 : total_hours = 7.5) :
    total_hours / hours_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_andrew_work_days_l3860_386096


namespace NUMINAMATH_CALUDE_max_glass_height_l3860_386060

/-- The maximum height of a truncated cone-shaped glass that can roll around a circular table without reaching the edge -/
theorem max_glass_height (table_diameter : Real) (glass_bottom_diameter : Real) (glass_top_diameter : Real)
  (h_table : table_diameter = 160)
  (h_glass_bottom : glass_bottom_diameter = 5)
  (h_glass_top : glass_top_diameter = 6.5) :
  ∃ (max_height : Real), 
    (∀ (h : Real), h > 0 ∧ h < max_height → 
      ∃ (x y : Real), x^2 + y^2 < (table_diameter/2)^2 ∧ 
        ((h * glass_bottom_diameter/2) / (glass_top_diameter/2 - glass_bottom_diameter/2))^2 + h^2 = 
        ((y - x) * (glass_top_diameter/2 - glass_bottom_diameter/2) / h)^2) ∧
    max_height < (3/13) * Real.sqrt 6389.4375 := by
  sorry

end NUMINAMATH_CALUDE_max_glass_height_l3860_386060


namespace NUMINAMATH_CALUDE_fraction_irreducible_l3860_386049

theorem fraction_irreducible (n : ℤ) : Int.gcd (3 * n + 10) (4 * n + 13) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l3860_386049


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3860_386092

theorem simplify_sqrt_expression :
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3860_386092


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l3860_386028

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem even_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) (h_dec : decreasing_on_nonneg f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 := by sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l3860_386028


namespace NUMINAMATH_CALUDE_book_cost_price_l3860_386055

def cost_price : ℝ → Prop := λ c => 
  (c * 1.1 + 90 = c * 1.15) ∧ 
  (c > 0)

theorem book_cost_price : ∃ c, cost_price c ∧ c = 1800 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3860_386055


namespace NUMINAMATH_CALUDE_mateo_deducted_salary_l3860_386086

theorem mateo_deducted_salary (weekly_salary : ℝ) (work_days : ℕ) (absent_days : ℕ) : 
  weekly_salary = 791 ∧ work_days = 5 ∧ absent_days = 4 →
  weekly_salary - (weekly_salary / work_days * absent_days) = 158.20 := by
  sorry

end NUMINAMATH_CALUDE_mateo_deducted_salary_l3860_386086


namespace NUMINAMATH_CALUDE_smallest_square_area_l3860_386072

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a square given its side length -/
def square_area (side : ℕ) : ℕ := side * side

/-- Checks if two rectangles can fit side by side within a given length -/
def can_fit_side_by_side (r1 r2 : Rectangle) (length : ℕ) : Prop :=
  r1.width + r2.width ≤ length ∨ r1.height + r2.height ≤ length

/-- Theorem: The smallest possible area of a square containing a 2×3 rectangle and a 4×5 rectangle
    without overlapping and with parallel sides is 49 square units -/
theorem smallest_square_area : ∃ (side : ℕ),
  let r1 : Rectangle := ⟨2, 3⟩
  let r2 : Rectangle := ⟨4, 5⟩
  (∀ (s : ℕ), s < side → ¬(can_fit_side_by_side r1 r2 s)) ∧
  (can_fit_side_by_side r1 r2 side) ∧
  (square_area side = 49) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_area_l3860_386072


namespace NUMINAMATH_CALUDE_min_n_for_constant_term_l3860_386038

theorem min_n_for_constant_term (n : ℕ) : 
  (∃ k : ℕ, (2 * n = 3 * k) ∧ (k ≤ n)) ↔ n ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_constant_term_l3860_386038


namespace NUMINAMATH_CALUDE_profit_margin_relation_l3860_386057

theorem profit_margin_relation (S C : ℝ) (n : ℝ) (h1 : S > 0) (h2 : C > 0) (h3 : n > 0) : 
  ((1 / 3 : ℝ) * S = (1 / n : ℝ) * C) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_profit_margin_relation_l3860_386057


namespace NUMINAMATH_CALUDE_additional_charge_correct_l3860_386003

/-- The charge for each additional 1/5 of a mile in a taxi ride -/
def additional_charge : ℝ := 0.40

/-- The initial charge for the first 1/5 of a mile -/
def initial_charge : ℝ := 2.50

/-- The total distance of the ride in miles -/
def total_distance : ℝ := 8

/-- The total charge for the ride -/
def total_charge : ℝ := 18.10

/-- Theorem stating that the additional charge is correct given the conditions -/
theorem additional_charge_correct :
  initial_charge + (total_distance - 1/5) / (1/5) * additional_charge = total_charge := by
  sorry

end NUMINAMATH_CALUDE_additional_charge_correct_l3860_386003


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3860_386064

/-- Given a function f : ℝ → ℝ with f(0) = 1 and f'(x) > f(x) for all x,
    the set of x where f(x) > e^x is (0, +∞) -/
theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h0 : f 0 = 1) (h1 : ∀ x, deriv f x > f x) :
    {x : ℝ | f x > Real.exp x} = Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3860_386064


namespace NUMINAMATH_CALUDE_girls_in_school_l3860_386045

/-- The number of girls in a school after new students join -/
def total_girls (initial_girls new_girls : ℕ) : ℕ :=
  initial_girls + new_girls

/-- Theorem stating that the total number of girls after new students joined is 1414 -/
theorem girls_in_school (initial_girls new_girls : ℕ) 
  (h1 : initial_girls = 732) 
  (h2 : new_girls = 682) : 
  total_girls initial_girls new_girls = 1414 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_school_l3860_386045


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l3860_386071

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ),
    (1 / x + 1 / (2 * y) = (x^2 + 3 * y^2) * (3 * x^2 + y^2)) ∧
    (1 / x - 1 / (2 * y) = 2 * (y^4 - x^4)) ∧
    (x = (3^(1/5) + 1) / 2) ∧
    (y = (3^(1/5) - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l3860_386071


namespace NUMINAMATH_CALUDE_remainder_theorem_l3860_386004

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 50 * k - 49) :
  (n^2 + 4*n + 5) % 50 = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3860_386004


namespace NUMINAMATH_CALUDE_jacob_has_six_marshmallows_l3860_386054

/-- Calculates the number of marshmallows Jacob currently has -/
def jacobs_marshmallows (graham_crackers : ℕ) (more_marshmallows_needed : ℕ) : ℕ :=
  (graham_crackers / 2) - more_marshmallows_needed

/-- Proves that Jacob has 6 marshmallows given the problem conditions -/
theorem jacob_has_six_marshmallows :
  jacobs_marshmallows 48 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jacob_has_six_marshmallows_l3860_386054


namespace NUMINAMATH_CALUDE_unique_x_divisible_by_15_l3860_386019

def is_valid_x (x : ℕ) : Prop :=
  x < 10 ∧ (∃ n : ℕ, x * 1000 + 200 + x * 10 + 3 = 15 * n)

theorem unique_x_divisible_by_15 : ∃! x : ℕ, is_valid_x x :=
  sorry

end NUMINAMATH_CALUDE_unique_x_divisible_by_15_l3860_386019


namespace NUMINAMATH_CALUDE_max_value_constraint_l3860_386074

theorem max_value_constraint (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  ∃ (M : ℝ), M = Real.sqrt 298 ∧ (8 * x + 3 * y + 15 * z ≤ M) ∧
  ∃ (x₀ y₀ z₀ : ℝ), 9 * x₀^2 + 4 * y₀^2 + 25 * z₀^2 = 1 ∧ 8 * x₀ + 3 * y₀ + 15 * z₀ = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3860_386074


namespace NUMINAMATH_CALUDE_interval_intersection_l3860_386023

theorem interval_intersection (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ 1/2 < x ∧ x < 3/5 := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l3860_386023


namespace NUMINAMATH_CALUDE_oliver_water_usage_l3860_386000

/-- Calculates the weekly water usage for Oliver's baths given the specified conditions. -/
def weekly_water_usage (bucket_capacity : ℕ) (fill_count : ℕ) (remove_count : ℕ) (days_per_week : ℕ) : ℕ :=
  (fill_count * bucket_capacity - remove_count * bucket_capacity) * days_per_week

/-- Theorem stating that Oliver's weekly water usage is 9240 ounces under the given conditions. -/
theorem oliver_water_usage :
  weekly_water_usage 120 14 3 7 = 9240 := by
  sorry

#eval weekly_water_usage 120 14 3 7

end NUMINAMATH_CALUDE_oliver_water_usage_l3860_386000


namespace NUMINAMATH_CALUDE_square_sum_value_l3860_386035

theorem square_sum_value (x y : ℝ) (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3860_386035


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l3860_386015

theorem cheryl_material_usage (bought_type1 bought_type2 leftover : ℚ) :
  bought_type1 = 5/9 →
  bought_type2 = 1/3 →
  leftover = 8/24 →
  bought_type1 + bought_type2 - leftover = 5/9 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l3860_386015


namespace NUMINAMATH_CALUDE_cubic_function_nonnegative_implies_parameter_bound_l3860_386012

theorem cubic_function_nonnegative_implies_parameter_bound 
  (f : ℝ → ℝ) (a : ℝ) 
  (h_def : ∀ x, f x = a * x^3 - 3 * x + 1)
  (h_nonneg : ∀ x ∈ Set.Icc 0 1, f x ≥ 0) :
  a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_nonnegative_implies_parameter_bound_l3860_386012


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_125_l3860_386089

theorem greatest_prime_factor_of_125 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 125 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 125 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_125_l3860_386089


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l3860_386017

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Two lines are different -/
def Line.different (l1 l2 : Line) : Prop := sorry

/-- A line is in a plane -/
def Line.inPlane (l : Line) (p : Plane) : Prop := sorry

/-- A line is outside a plane -/
def Line.outsidePlane (l : Line) (p : Plane) : Prop := sorry

/-- A line is perpendicular to another line -/
def Line.perpendicular (l1 l2 : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def Line.perpendicularToPlane (l : Line) (p : Plane) : Prop := sorry

theorem perpendicular_necessary_not_sufficient
  (α : Plane) (a b l : Line)
  (h1 : a.inPlane α)
  (h2 : b.inPlane α)
  (h3 : l.outsidePlane α)
  (h4 : a.different b) :
  (l.perpendicularToPlane α → (l.perpendicular a ∧ l.perpendicular b)) ∧
  ¬((l.perpendicular a ∧ l.perpendicular b) → l.perpendicularToPlane α) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l3860_386017


namespace NUMINAMATH_CALUDE_probability_of_pair_after_removal_l3860_386021

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : List ℕ)
  (counts : List ℕ)

/-- The probability of selecting a pair from the remaining deck -/
def probability_of_pair (d : Deck) : ℚ :=
  83 / 1035

theorem probability_of_pair_after_removal (d : Deck) : 
  d.total = 50 ∧ 
  d.numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ 
  d.counts = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5] →
  let remaining_deck := {
    total := d.total - 4,
    numbers := d.numbers,
    counts := d.counts.map (fun c => if c = 5 then 3 else 5)
  }
  probability_of_pair remaining_deck = 83 / 1035 := by
sorry

#eval 83 + 1035  -- Should output 1118

end NUMINAMATH_CALUDE_probability_of_pair_after_removal_l3860_386021


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l3860_386047

theorem fractional_equation_solution_range (m x : ℝ) : 
  (m / (2 * x - 1) + 3 = 0) → 
  (x > 0) → 
  (m < 3 ∧ m ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l3860_386047


namespace NUMINAMATH_CALUDE_tom_batteries_total_l3860_386095

/-- The total number of batteries Tom used is 19, given the number of batteries used for each category. -/
theorem tom_batteries_total (flashlight_batteries : ℕ) (toy_batteries : ℕ) (controller_batteries : ℕ)
  (h1 : flashlight_batteries = 2)
  (h2 : toy_batteries = 15)
  (h3 : controller_batteries = 2) :
  flashlight_batteries + toy_batteries + controller_batteries = 19 := by
  sorry

end NUMINAMATH_CALUDE_tom_batteries_total_l3860_386095


namespace NUMINAMATH_CALUDE_hilton_lost_marbles_l3860_386075

/-- Proves the number of marbles Hilton lost given the initial and final conditions -/
theorem hilton_lost_marbles (initial : ℕ) (found : ℕ) (final : ℕ) : 
  initial = 26 → found = 6 → final = 42 → 
  ∃ (lost : ℕ), lost = 10 ∧ final = initial + found - lost + 2 * lost := by
  sorry

end NUMINAMATH_CALUDE_hilton_lost_marbles_l3860_386075


namespace NUMINAMATH_CALUDE_annual_profit_calculation_l3860_386082

theorem annual_profit_calculation (second_half_profit first_half_profit total_profit : ℕ) :
  second_half_profit = 442500 →
  first_half_profit = second_half_profit + 2750000 →
  total_profit = first_half_profit + second_half_profit →
  total_profit = 3635000 := by
  sorry

end NUMINAMATH_CALUDE_annual_profit_calculation_l3860_386082


namespace NUMINAMATH_CALUDE_crayons_per_box_l3860_386037

theorem crayons_per_box (total_crayons : Float) (total_boxes : Float) 
  (h1 : total_crayons = 7.0)
  (h2 : total_boxes = 1.4) :
  total_crayons / total_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_box_l3860_386037


namespace NUMINAMATH_CALUDE_number_of_dogs_l3860_386065

theorem number_of_dogs (total_legs : ℕ) (num_humans : ℕ) (human_legs : ℕ) (dog_legs : ℕ)
  (h1 : total_legs = 24)
  (h2 : num_humans = 2)
  (h3 : human_legs = 2)
  (h4 : dog_legs = 4) :
  (total_legs - num_humans * human_legs) / dog_legs = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_dogs_l3860_386065


namespace NUMINAMATH_CALUDE_not_div_sum_if_div_sum_squares_l3860_386085

theorem not_div_sum_if_div_sum_squares (a b : ℤ) : 
  7 ∣ (a^2 + b^2 + 1) → ¬(7 ∣ (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_not_div_sum_if_div_sum_squares_l3860_386085


namespace NUMINAMATH_CALUDE_find_m_l3860_386022

-- Define the universal set U
def U : Set Nat := {1, 2, 3}

-- Define set A
def A (m : Nat) : Set Nat := {1, m}

-- Define the complement of A in U
def complementA : Set Nat := {2}

-- Theorem to prove
theorem find_m : ∃ m : Nat, m ∈ U ∧ A m ∪ complementA = U := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3860_386022


namespace NUMINAMATH_CALUDE_product_minus_difference_l3860_386051

theorem product_minus_difference (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 6) (h4 : x / y = 6) : x * y - (x - y) = 6 / 49 := by
  sorry

end NUMINAMATH_CALUDE_product_minus_difference_l3860_386051


namespace NUMINAMATH_CALUDE_liquid_depth_inverted_cone_l3860_386048

/-- Represents a right circular cone. -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the liquid in the cone. -/
structure Liquid where
  depthPointDown : ℝ
  depthPointUp : ℝ

/-- Theorem stating the relationship between cone dimensions, liquid depth, and the expression m - n∛p. -/
theorem liquid_depth_inverted_cone (c : Cone) (l : Liquid) 
  (h_height : c.height = 12)
  (h_radius : c.baseRadius = 5)
  (h_depth_down : l.depthPointDown = 9)
  (h_p_cube_free : ∀ (q : ℕ), q > 1 → ¬(q ^ 3 ∣ 37)) :
  ∃ (m n : ℕ), m = 12 ∧ n = 3 ∧ l.depthPointUp = m - n * (37 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_liquid_depth_inverted_cone_l3860_386048


namespace NUMINAMATH_CALUDE_volume_equality_l3860_386011

/-- The volume of the solid obtained by rotating the region bounded by x² = 4y, x² = -4y, x = 4, and x = -4 about the y-axis -/
def V₁ : ℝ := sorry

/-- The volume of the solid obtained by rotating the region defined by x² + y² ≤ 16, x² + (y-2)² ≥ 4, and x² + (y+2)² ≥ 4 about the y-axis -/
def V₂ : ℝ := sorry

/-- Theorem stating that V₁ equals V₂ -/
theorem volume_equality : V₁ = V₂ := by sorry

end NUMINAMATH_CALUDE_volume_equality_l3860_386011


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3860_386099

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3^p.1}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2^(-p.1)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(0, 1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3860_386099


namespace NUMINAMATH_CALUDE_fruits_eaten_over_two_meals_l3860_386069

/-- Calculates the total number of fruits eaten over two meals given specific conditions --/
theorem fruits_eaten_over_two_meals : 
  let apples_last_night : ℕ := 3
  let bananas_last_night : ℕ := 1
  let oranges_last_night : ℕ := 4
  let strawberries_last_night : ℕ := 2
  
  let apples_today : ℕ := apples_last_night + 4
  let bananas_today : ℕ := bananas_last_night * 10
  let oranges_today : ℕ := apples_today * 2
  let strawberries_today : ℕ := (oranges_last_night + apples_last_night) * 3
  
  let total_fruits : ℕ := 
    (apples_last_night + apples_today) +
    (bananas_last_night + bananas_today) +
    (oranges_last_night + oranges_today) +
    (strawberries_last_night + strawberries_today)
  
  total_fruits = 62 := by sorry

end NUMINAMATH_CALUDE_fruits_eaten_over_two_meals_l3860_386069


namespace NUMINAMATH_CALUDE_divisibility_proof_l3860_386032

theorem divisibility_proof (a : ℤ) (n : ℕ) : 
  ∃ k : ℤ, (a + 1)^(2*n + 1) + a^(n + 2) = k * (a^2 + a + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l3860_386032


namespace NUMINAMATH_CALUDE_painting_selections_l3860_386010

/-- The number of traditional Chinese paintings -/
def traditional_paintings : Nat := 5

/-- The number of oil paintings -/
def oil_paintings : Nat := 2

/-- The number of watercolor paintings -/
def watercolor_paintings : Nat := 7

/-- The number of ways to choose one painting from each category -/
def one_from_each : Nat := traditional_paintings * oil_paintings * watercolor_paintings

/-- The number of ways to choose two paintings of different types -/
def two_different_types : Nat := 
  traditional_paintings * oil_paintings + 
  traditional_paintings * watercolor_paintings + 
  oil_paintings * watercolor_paintings

theorem painting_selections :
  one_from_each = 70 ∧ two_different_types = 59 := by sorry

end NUMINAMATH_CALUDE_painting_selections_l3860_386010


namespace NUMINAMATH_CALUDE_animal_count_animal_group_count_l3860_386062

theorem animal_count (total_horses : ℕ) (cow_cow_diff : ℕ) : ℕ :=
  let total_animals := 2 * (total_horses + cow_cow_diff)
  total_animals

theorem animal_group_count : animal_count 75 10 = 170 := by
  sorry

end NUMINAMATH_CALUDE_animal_count_animal_group_count_l3860_386062


namespace NUMINAMATH_CALUDE_smores_theorem_l3860_386091

def smores_problem (graham_crackers : ℕ) (marshmallows : ℕ) : ℕ :=
  let smores_from_graham := graham_crackers / 2
  smores_from_graham - marshmallows

theorem smores_theorem (graham_crackers marshmallows : ℕ) :
  graham_crackers = 48 →
  marshmallows = 6 →
  smores_problem graham_crackers marshmallows = 18 :=
by sorry

end NUMINAMATH_CALUDE_smores_theorem_l3860_386091


namespace NUMINAMATH_CALUDE_zookeeper_fish_count_l3860_386058

theorem zookeeper_fish_count (penguins_fed : ℕ) (total_penguins : ℕ) (penguins_to_feed : ℕ) :
  penguins_fed = 19 →
  total_penguins = 36 →
  penguins_to_feed = 17 →
  penguins_fed + penguins_to_feed = total_penguins :=
by sorry

end NUMINAMATH_CALUDE_zookeeper_fish_count_l3860_386058


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3860_386098

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = 2^p - 1 ∧ Prime n

theorem largest_mersenne_prime_under_500 : 
  (∀ n : ℕ, n < 500 → is_mersenne_prime n → n ≤ 127) ∧ 
  is_mersenne_prime 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3860_386098


namespace NUMINAMATH_CALUDE_triangle_properties_l3860_386094

-- Define the triangle
def triangle_ABC (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_properties :
  ∀ (a b c : ℝ),
  triangle_ABC a b c →
  a = 2 →
  c = 3 →
  Real.cos (Real.arccos (1/4)) = 1/4 →
  b = Real.sqrt 10 ∧
  Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = (3 * Real.sqrt 6) / 8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3860_386094


namespace NUMINAMATH_CALUDE_direct_proportion_percentage_change_l3860_386008

theorem direct_proportion_percentage_change 
  (x y : ℝ) (q : ℝ) (c : ℝ) (hx : x > 0) (hy : y > 0) (hq : q > 0) (hc : c > 0) 
  (h_prop : y = c * x) :
  let x' := x * (1 - q / 100)
  let y' := c * x'
  (y' - y) / y * 100 = q := by
sorry

end NUMINAMATH_CALUDE_direct_proportion_percentage_change_l3860_386008


namespace NUMINAMATH_CALUDE_function_comparison_l3860_386083

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_decreasing : ∀ x y, x < y → y < 0 → f x > f y)

-- Define the theorem
theorem function_comparison (x₁ x₂ : ℝ) 
  (h₁ : x₁ < 0) (h₂ : x₁ + x₂ > 0) : f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_function_comparison_l3860_386083


namespace NUMINAMATH_CALUDE_final_price_is_25_92_l3860_386077

/-- The final price observed by the buyer online given the commission rate,
    product cost, and desired profit rate. -/
def final_price (commission_rate : ℝ) (product_cost : ℝ) (profit_rate : ℝ) : ℝ :=
  let profit := product_cost * profit_rate
  let distributor_price := product_cost + profit
  let commission := distributor_price * commission_rate
  distributor_price + commission

/-- Theorem stating that the final price is $25.92 given the specified conditions -/
theorem final_price_is_25_92 :
  final_price 0.2 18 0.2 = 25.92 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_25_92_l3860_386077


namespace NUMINAMATH_CALUDE_unique_consecutive_set_sum_18_l3860_386097

/-- A set of consecutive positive integers -/
def ConsecutiveSet (a n : ℕ) : Set ℕ := {x | ∃ k, 0 ≤ k ∧ k < n ∧ x = a + k}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSetSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Main theorem: There is exactly one set of consecutive positive integers with sum 18 -/
theorem unique_consecutive_set_sum_18 :
  ∃! p : ℕ × ℕ, p.2 ≥ 2 ∧ ConsecutiveSetSum p.1 p.2 = 18 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_set_sum_18_l3860_386097


namespace NUMINAMATH_CALUDE_y_finishing_time_l3860_386024

/-- The number of days it takes y to finish the remaining work after x has worked for 8 days -/
def days_for_y_to_finish (x_total_days y_total_days x_worked_days : ℕ) : ℕ :=
  (y_total_days * (x_total_days - x_worked_days)) / x_total_days

theorem y_finishing_time 
  (x_total_days : ℕ) 
  (y_total_days : ℕ) 
  (x_worked_days : ℕ) 
  (h1 : x_total_days = 40)
  (h2 : y_total_days = 40)
  (h3 : x_worked_days = 8) :
  days_for_y_to_finish x_total_days y_total_days x_worked_days = 32 := by
sorry

#eval days_for_y_to_finish 40 40 8

end NUMINAMATH_CALUDE_y_finishing_time_l3860_386024


namespace NUMINAMATH_CALUDE_reflection_theorem_l3860_386006

noncomputable def C₁ (x : ℝ) : ℝ := Real.arccos (-x)

theorem reflection_theorem (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) :
  ∃ y, C₁ y = x ∧ y = -Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_reflection_theorem_l3860_386006


namespace NUMINAMATH_CALUDE_five_by_five_uncoverable_l3860_386067

/-- Represents a game board -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (black_squares : Nat)
  (white_squares : Nat)

/-- Represents a domino placement on the board -/
def DominoPlacement := List (Nat × Nat)

/-- Check if a board can be covered by dominoes -/
def can_be_covered (b : Board) (p : DominoPlacement) : Prop :=
  (b.rows * b.cols = 2 * p.length) ∧ 
  (b.black_squares = b.white_squares)

/-- The 5x5 board with specific color pattern -/
def board_5x5 : Board :=
  { rows := 5
  , cols := 5
  , black_squares := 9   -- central 3x3 section
  , white_squares := 16  -- border
  }

/-- Theorem stating that the 5x5 board cannot be covered -/
theorem five_by_five_uncoverable : 
  ∀ p : DominoPlacement, ¬(can_be_covered board_5x5 p) :=
sorry

end NUMINAMATH_CALUDE_five_by_five_uncoverable_l3860_386067


namespace NUMINAMATH_CALUDE_range_of_a_l3860_386016

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 - (a + 1)*x + 1 > 0

def Q (a : ℝ) : Prop := ∀ x : ℝ, |x - 1| ≥ a + 2

-- State the theorem
theorem range_of_a (a : ℝ) : (¬(P a ∨ Q a)) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3860_386016


namespace NUMINAMATH_CALUDE_first_season_episodes_l3860_386001

/-- The number of seasons in the TV show -/
def num_seasons : ℕ := 5

/-- The cost per episode for the first season in dollars -/
def first_season_cost : ℕ := 100000

/-- The cost per episode for seasons after the first in dollars -/
def other_season_cost : ℕ := 2 * first_season_cost

/-- The increase factor for the number of episodes in each season after the first -/
def episode_increase_factor : ℚ := 3/2

/-- The number of episodes in the last season -/
def last_season_episodes : ℕ := 24

/-- The total cost to produce all episodes in dollars -/
def total_cost : ℕ := 16800000

/-- Calculate the total cost of all seasons given the number of episodes in the first season -/
def calculate_total_cost (first_season_episodes : ℕ) : ℚ :=
  let first_season := first_season_cost * first_season_episodes
  let second_season := other_season_cost * (episode_increase_factor * first_season_episodes)
  let third_season := other_season_cost * (episode_increase_factor^2 * first_season_episodes)
  let fourth_season := other_season_cost * (episode_increase_factor^3 * first_season_episodes)
  let fifth_season := other_season_cost * last_season_episodes
  first_season + second_season + third_season + fourth_season + fifth_season

/-- Theorem stating that the number of episodes in the first season is 8 -/
theorem first_season_episodes : ∃ (x : ℕ), x = 8 ∧ calculate_total_cost x = total_cost := by
  sorry

end NUMINAMATH_CALUDE_first_season_episodes_l3860_386001


namespace NUMINAMATH_CALUDE_smallest_x_plus_y_l3860_386078

theorem smallest_x_plus_y : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ x < y ∧
  (100 : ℚ) + (x : ℚ) / y = 2 * ((100 : ℚ) * x / y) ∧
  ∀ (a b : ℕ), a > 0 → b > 0 → a < b → 
    (100 : ℚ) + (a : ℚ) / b = 2 * ((100 : ℚ) * a / b) →
    x + y ≤ a + b ∧
  x + y = 299 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_plus_y_l3860_386078


namespace NUMINAMATH_CALUDE_intersection_S_T_l3860_386076

def S : Set ℝ := {x | |x| < 5}
def T : Set ℝ := {x | (x+7)*(x-3) < 0}

theorem intersection_S_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l3860_386076


namespace NUMINAMATH_CALUDE_solution_volume_l3860_386073

/-- Given two solutions, one of 6 litres and another of V litres, 
    if 20% of the first solution is mixed with 60% of the second solution,
    and the resulting mixture is 36% of the total volume,
    then V equals 4 litres. -/
theorem solution_volume (V : ℝ) : 
  (0.2 * 6 + 0.6 * V) / (6 + V) = 0.36 → V = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_volume_l3860_386073


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3860_386088

theorem quadratic_inequality_solution_range (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 10*x + c < 0) ↔ (c < 25) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3860_386088


namespace NUMINAMATH_CALUDE_quadratic_root_equivalence_l3860_386039

theorem quadratic_root_equivalence (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ Real.sqrt 12 = k * Real.sqrt 3) ∧ 
  (∃ m : ℝ, m > 0 ∧ 5 * Real.sqrt (a + 1) = m * Real.sqrt 3) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_equivalence_l3860_386039


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3860_386014

theorem quadratic_inequality_solution_set (x : ℝ) :
  x^2 - 2*x - 3 > 0 ↔ x < -1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3860_386014


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3860_386043

theorem min_value_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 9) : 
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 
  x/y + y/z + z/x + y/x + z/y + x/z = 9 → 
  (a/b + b/c + c/a)^2 + (b/a + c/b + a/c)^2 ≤ (x/y + y/z + z/x)^2 + (y/x + z/y + x/z)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3860_386043


namespace NUMINAMATH_CALUDE_min_value_and_range_l3860_386018

-- Define the function f(x, y, a) = 2xy - x - y - a(x^2 + y^2)
def f (x y a : ℝ) : ℝ := 2 * x * y - x - y - a * (x^2 + y^2)

theorem min_value_and_range {x y a : ℝ} (hx : x > 0) (hy : y > 0) (hf : f x y a = 0) :
  -- Part 1: When a = 0, minimum value of 2x + 4y and corresponding x, y
  (a = 0 → 2 * x + 4 * y ≥ 3 + 2 * Real.sqrt 2 ∧
    (2 * x + 4 * y = 3 + 2 * Real.sqrt 2 ↔ x = (1 + Real.sqrt 2) / 2 ∧ y = (2 + Real.sqrt 2) / 4)) ∧
  -- Part 2: When a = 1/2, range of x + y
  (a = 1/2 → x + y ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_range_l3860_386018


namespace NUMINAMATH_CALUDE_first_term_of_a_10_l3860_386009

def first_term (n : ℕ) : ℕ :=
  1 + 2 * (List.range n).sum

theorem first_term_of_a_10 : first_term 10 = 91 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_a_10_l3860_386009


namespace NUMINAMATH_CALUDE_even_sum_difference_l3860_386063

def sum_even_range (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

theorem even_sum_difference : sum_even_range 62 110 - sum_even_range 42 90 = 500 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_difference_l3860_386063


namespace NUMINAMATH_CALUDE_max_red_beads_l3860_386025

/-- Represents a string of beads with red, blue, and green colors. -/
structure BeadString where
  total_beads : ℕ
  red_beads : ℕ
  blue_beads : ℕ
  green_beads : ℕ
  sum_constraint : total_beads = red_beads + blue_beads + green_beads
  green_constraint : ∀ n : ℕ, n + 6 ≤ total_beads → ∃ i, n ≤ i ∧ i < n + 6 ∧ green_beads > 0
  blue_constraint : ∀ n : ℕ, n + 11 ≤ total_beads → ∃ i, n ≤ i ∧ i < n + 11 ∧ blue_beads > 0

/-- The maximum number of red beads in a string of 150 beads with given constraints is 112. -/
theorem max_red_beads :
  ∀ bs : BeadString, bs.total_beads = 150 → bs.red_beads ≤ 112 :=
by sorry

end NUMINAMATH_CALUDE_max_red_beads_l3860_386025


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3860_386053

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) : 
  (∀ angle : ℝ, angle = 156 ∧ (180 * (n - 2) : ℝ) = n * angle) → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3860_386053


namespace NUMINAMATH_CALUDE_number_operation_result_l3860_386087

theorem number_operation_result : 
  let n : ℚ := 55
  (n / 5 + 10) = 21 := by sorry

end NUMINAMATH_CALUDE_number_operation_result_l3860_386087


namespace NUMINAMATH_CALUDE_felix_brother_lifting_capacity_l3860_386046

/-- Given information about Felix and his brother's weights and lifting capacities,
    prove how much Felix's brother can lift off the ground. -/
theorem felix_brother_lifting_capacity
  (felix_lift_ratio : ℝ)
  (felix_lift_weight : ℝ)
  (brother_weight_ratio : ℝ)
  (brother_lift_ratio : ℝ)
  (h1 : felix_lift_ratio = 1.5)
  (h2 : felix_lift_weight = 150)
  (h3 : brother_weight_ratio = 2)
  (h4 : brother_lift_ratio = 3) :
  felix_lift_weight * brother_weight_ratio * brother_lift_ratio / felix_lift_ratio = 600 :=
by sorry

end NUMINAMATH_CALUDE_felix_brother_lifting_capacity_l3860_386046


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3860_386081

theorem sine_cosine_inequality (x : Real) (h : Real.sin x + Real.cos x ≤ 0) :
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3860_386081


namespace NUMINAMATH_CALUDE_modulo_problem_l3860_386042

theorem modulo_problem (m : ℕ) : 
  (65 * 76 * 87 ≡ m [ZMOD 25]) → 
  (0 ≤ m ∧ m < 25) → 
  m = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulo_problem_l3860_386042
