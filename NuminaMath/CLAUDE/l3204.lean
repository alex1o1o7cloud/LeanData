import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_implies_m_eq_4_l3204_320467

def S : Finset ℕ := {1, 2, 3, 4}

def A (m : ℕ) : Finset ℕ := S.filter (λ x => x^2 - 5*x + m = 0)

theorem complement_A_implies_m_eq_4 :
  (S \ A m) = {2, 3} → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_complement_A_implies_m_eq_4_l3204_320467


namespace NUMINAMATH_CALUDE_log_problem_l3204_320407

theorem log_problem (y : ℝ) (p : ℝ) 
  (h1 : Real.log 5 / Real.log 9 = y)
  (h2 : Real.log 125 / Real.log 3 = p * y) : 
  p = 6 := by sorry

end NUMINAMATH_CALUDE_log_problem_l3204_320407


namespace NUMINAMATH_CALUDE_complex_expression_proof_l3204_320436

theorem complex_expression_proof :
  let A : ℂ := 5 - 2*I
  let M : ℂ := -3 + 2*I
  let S : ℂ := 2*I
  let P : ℂ := 3
  2 * (A - M + S - P) = 10 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_proof_l3204_320436


namespace NUMINAMATH_CALUDE_walmart_sales_theorem_walmart_december_sales_l3204_320461

/-- Calculates the total sales amount for Wal-Mart in December -/
theorem walmart_sales_theorem (thermometer_price : ℕ) (hot_water_bottle_price : ℕ) 
  (thermometer_to_bottle_ratio : ℕ) (hot_water_bottles_sold : ℕ) : ℕ :=
  let thermometers_sold := thermometer_to_bottle_ratio * hot_water_bottles_sold
  let thermometer_sales := thermometers_sold * thermometer_price
  let hot_water_bottle_sales := hot_water_bottles_sold * hot_water_bottle_price
  thermometer_sales + hot_water_bottle_sales

/-- Proves that the total sales amount for Wal-Mart in December is $1200 -/
theorem walmart_december_sales : 
  walmart_sales_theorem 2 6 7 60 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_walmart_sales_theorem_walmart_december_sales_l3204_320461


namespace NUMINAMATH_CALUDE_base4_calculation_l3204_320473

/-- Represents a number in base 4 --/
def Base4 : Type := ℕ

/-- Converts a base 4 number to its decimal representation --/
def to_decimal (n : Base4) : ℕ := sorry

/-- Converts a decimal number to its base 4 representation --/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Multiplication operation for base 4 numbers --/
def mul_base4 (a b : Base4) : Base4 := 
  to_base4 (to_decimal a * to_decimal b)

/-- Division operation for base 4 numbers --/
def div_base4 (a b : Base4) : Base4 := 
  to_base4 (to_decimal a / to_decimal b)

theorem base4_calculation : 
  mul_base4 (div_base4 (to_base4 210) (to_base4 3)) (to_base4 21) = to_base4 1102 := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l3204_320473


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3204_320492

/-- The equation of the tangent line to y = 2x - x³ at (1, 1) is x + y - 2 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = 2*x - x^3) → -- The curve equation
  ((1 : ℝ) = 1 → (2*(1 : ℝ) - (1 : ℝ)^3) = 1) → -- The point (1, 1) lies on the curve
  (x + y - 2 = 0) ↔ -- The tangent line equation
  (∃ (m : ℝ), y - 1 = m * (x - 1) ∧ 
              m = (2 - 3*(1 : ℝ)^2)) -- Slope of the tangent line at x = 1
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3204_320492


namespace NUMINAMATH_CALUDE_number_equation_solution_l3204_320456

theorem number_equation_solution :
  ∀ x : ℝ, 35 + 3 * x^2 = 89 → x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3204_320456


namespace NUMINAMATH_CALUDE_sale_final_prices_correct_l3204_320429

/-- Calculates the final price of an item after a series of percentage discounts and flat discounts --/
def finalPrice (originalPrice : ℝ) (percentDiscounts : List ℝ) (flatDiscounts : List ℝ) : ℝ :=
  let applyPercentDiscount (price : ℝ) (discount : ℝ) := price * (1 - discount)
  let applyFlatDiscount (price : ℝ) (discount : ℝ) := price - discount
  let priceAfterPercentDiscounts := percentDiscounts.foldl applyPercentDiscount originalPrice
  flatDiscounts.foldl applyFlatDiscount priceAfterPercentDiscounts

/-- Proves that the final prices of the electronic item and clothing item are correct after the 4-day sale --/
theorem sale_final_prices_correct (electronicOriginalPrice clothingOriginalPrice : ℝ) 
  (h1 : electronicOriginalPrice = 480)
  (h2 : clothingOriginalPrice = 260) : 
  let electronicFinalPrice := finalPrice electronicOriginalPrice [0.1, 0.14, 0.12, 0.08] []
  let clothingFinalPrice := finalPrice clothingOriginalPrice [0.1, 0.12, 0.05] [20]
  (electronicFinalPrice = 300.78 ∧ clothingFinalPrice = 176.62) := by
  sorry


end NUMINAMATH_CALUDE_sale_final_prices_correct_l3204_320429


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_range_l3204_320400

theorem quadratic_roots_imply_a_range (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ 
   x^2 + 2*(a-1)*x + 2*a + 6 = 0 ∧
   y^2 + 2*(a-1)*y + 2*a + 6 = 0) →
  a < -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_range_l3204_320400


namespace NUMINAMATH_CALUDE_positive_integer_solutions_inequality_l3204_320424

theorem positive_integer_solutions_inequality (x : ℕ+) :
  2 * (x.val - 1) < 7 - x.val ↔ x = 1 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_inequality_l3204_320424


namespace NUMINAMATH_CALUDE_recycle_388_cans_l3204_320485

/-- Recursively calculate the number of new cans produced from recycling -/
def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 6 then 0
  else
    let new_cans := (2 * initial_cans) / 6
    new_cans + recycle_cans new_cans

/-- The total number of new cans produced from 388 initial cans -/
def total_new_cans : ℕ := recycle_cans 388

/-- Theorem stating that 193 new cans are produced from 388 initial cans -/
theorem recycle_388_cans : total_new_cans = 193 := by
  sorry

end NUMINAMATH_CALUDE_recycle_388_cans_l3204_320485


namespace NUMINAMATH_CALUDE_two_digit_condition_l3204_320420

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property that we want to prove
def satisfiesCondition (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n = 6 * sumOfDigits (n + 7)

-- Statement of the theorem
theorem two_digit_condition :
  ∀ n : ℕ, satisfiesCondition n ↔ (n = 24 ∨ n = 78) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_condition_l3204_320420


namespace NUMINAMATH_CALUDE_sum_of_y_coefficients_correct_expressions_equal_l3204_320472

/-- The sum of coefficients of terms containing y in (5x+3y+2)(2x+5y+3) -/
def sum_of_y_coefficients : ℤ := 65

/-- The original expression -/
def original_expression (x y : ℚ) : ℚ := (5*x + 3*y + 2) * (2*x + 5*y + 3)

/-- Expanded form of the original expression -/
def expanded_expression (x y : ℚ) : ℚ := 
  10*x^2 + 31*x*y + 19*x + 15*y^2 + 19*y + 6

/-- Theorem stating that the sum of coefficients of terms containing y 
    in the expanded expression is equal to sum_of_y_coefficients -/
theorem sum_of_y_coefficients_correct : 
  (31 : ℤ) + 15 + 19 = sum_of_y_coefficients := by sorry

/-- Theorem stating that the original expression and expanded expression are equal -/
theorem expressions_equal (x y : ℚ) : 
  original_expression x y = expanded_expression x y := by sorry

end NUMINAMATH_CALUDE_sum_of_y_coefficients_correct_expressions_equal_l3204_320472


namespace NUMINAMATH_CALUDE_midpoint_triangle_perimeter_l3204_320457

/-- Given a triangle with perimeter p, the perimeter of the triangle formed by 
    connecting the midpoints of its sides is p/2. -/
theorem midpoint_triangle_perimeter (p : ℝ) (h : p > 0) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = p ∧ 
  (a/2 + b/2 + c/2) = p/2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_triangle_perimeter_l3204_320457


namespace NUMINAMATH_CALUDE_joes_lift_l3204_320434

theorem joes_lift (first_lift second_lift : ℕ) 
  (h1 : first_lift + second_lift = 1800)
  (h2 : 2 * first_lift = second_lift + 300) : 
  first_lift = 700 := by
sorry

end NUMINAMATH_CALUDE_joes_lift_l3204_320434


namespace NUMINAMATH_CALUDE_slope_of_CD_is_one_l3204_320479

/-- Given a line l passing through the origin O and intersecting y = e^(x-1) at two different points A and B,
    and lines parallel to y-axis drawn through A and B intersecting y = ln x at C and D respectively,
    prove that the slope of line CD is 1. -/
theorem slope_of_CD_is_one (k : ℝ) (hk : k > 0) : ∃ x₁ x₂ : ℝ, 
  x₁ ≠ x₂ ∧ 
  k * x₁ = Real.exp (x₁ - 1) ∧ 
  k * x₂ = Real.exp (x₂ - 1) ∧ 
  (Real.log (k * x₁) - Real.log (k * x₂)) / (k * x₁ - k * x₂) = 1 := by
  sorry


end NUMINAMATH_CALUDE_slope_of_CD_is_one_l3204_320479


namespace NUMINAMATH_CALUDE_S_not_algorithmically_solvable_l3204_320421

-- Define a type for expressions
inductive Expression
  | finite : Nat → Expression  -- Represents finite sums
  | infinite : Expression      -- Represents infinite sums

-- Define what it means for an expression to be algorithmically solvable
def is_algorithmically_solvable (e : Expression) : Prop :=
  match e with
  | Expression.finite _ => True
  | Expression.infinite => False

-- Define the infinite sum S = 1 + 2 + 3 + ...
def S : Expression := Expression.infinite

-- Theorem statement
theorem S_not_algorithmically_solvable :
  ¬(is_algorithmically_solvable S) :=
sorry

end NUMINAMATH_CALUDE_S_not_algorithmically_solvable_l3204_320421


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l3204_320491

theorem set_equality_implies_sum (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = ({0, a^2, a+b} : Set ℝ) → a^2022 + b^2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l3204_320491


namespace NUMINAMATH_CALUDE_sin_squared_sum_l3204_320490

theorem sin_squared_sum (α β : ℝ) : 
  Real.sin (α + β) ^ 2 = Real.cos α ^ 2 + Real.cos β ^ 2 - 2 * Real.cos α * Real.cos β * Real.cos (α + β) := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_l3204_320490


namespace NUMINAMATH_CALUDE_absolute_value_minus_self_nonnegative_l3204_320427

theorem absolute_value_minus_self_nonnegative (a : ℝ) : |a| - a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_minus_self_nonnegative_l3204_320427


namespace NUMINAMATH_CALUDE_unique_promotion_solution_l3204_320410

/-- Represents the promotional offer for pencils -/
structure PencilPromotion where
  base : ℕ  -- The number of pencils Pete's mom gave money for
  bonus : ℕ -- The additional pencils Pete could buy due to the promotion

/-- Defines the specific promotion where Pete buys 12 more pencils -/
def specificPromotion : PencilPromotion := { base := 49, bonus := 12 }

/-- Theorem stating that the specific promotion is the only one satisfying the conditions -/
theorem unique_promotion_solution : 
  ∀ (p : PencilPromotion), p.bonus = 12 → p.base = 49 := by
  sorry

#check unique_promotion_solution

end NUMINAMATH_CALUDE_unique_promotion_solution_l3204_320410


namespace NUMINAMATH_CALUDE_yard_sale_books_theorem_l3204_320469

/-- Represents Melanie's book collection --/
structure BookCollection where
  initial_books : ℕ
  current_books : ℕ
  magazines : ℕ

/-- Calculates the number of books bought at the yard sale --/
def books_bought (collection : BookCollection) : ℕ :=
  collection.current_books - collection.initial_books

/-- Theorem stating that the number of books bought at the yard sale
    is the difference between current and initial book counts --/
theorem yard_sale_books_theorem (collection : BookCollection)
    (h1 : collection.initial_books = 83)
    (h2 : collection.current_books = 167)
    (h3 : collection.magazines = 57) :
    books_bought collection = 84 := by
  sorry

end NUMINAMATH_CALUDE_yard_sale_books_theorem_l3204_320469


namespace NUMINAMATH_CALUDE_evenProductProbabilityFor6And4_l3204_320425

/-- Represents a spinner with n equal segments numbered from 1 to n -/
structure Spinner :=
  (n : ℕ)

/-- The probability of getting an even product when spinning two spinners -/
def evenProductProbability (spinnerA spinnerB : Spinner) : ℚ :=
  sorry

/-- Theorem stating that the probability of getting an even product
    when spinning a 6-segment spinner and a 4-segment spinner is 1/2 -/
theorem evenProductProbabilityFor6And4 :
  evenProductProbability (Spinner.mk 6) (Spinner.mk 4) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_evenProductProbabilityFor6And4_l3204_320425


namespace NUMINAMATH_CALUDE_james_overtime_multiple_l3204_320488

/-- Harry's pay rate for the first 24 hours -/
def harry_base_rate (x : ℝ) : ℝ := x

/-- Harry's pay rate for additional hours -/
def harry_overtime_rate (x : ℝ) : ℝ := 1.5 * x

/-- James's pay rate for the first 40 hours -/
def james_base_rate (x : ℝ) : ℝ := x

/-- James's pay rate for additional hours -/
def james_overtime_rate (x : ℝ) (m : ℝ) : ℝ := m * x

/-- Total hours worked by James -/
def james_total_hours : ℝ := 41

/-- Theorem stating the multiple of x dollars for James's overtime -/
theorem james_overtime_multiple (x : ℝ) (m : ℝ) :
  (harry_base_rate x * 24 + harry_overtime_rate x * (james_total_hours - 24) =
   james_base_rate x * 40 + james_overtime_rate x m * (james_total_hours - 40)) →
  m = 9.5 := by
  sorry

#check james_overtime_multiple

end NUMINAMATH_CALUDE_james_overtime_multiple_l3204_320488


namespace NUMINAMATH_CALUDE_parabola_reflection_l3204_320498

/-- A parabola is a function of the form y = a(x - h)^2 + k, where (h, k) is the vertex -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Reflection of a parabola along the y-axis -/
def reflect_y_axis (p : Parabola) : Parabola :=
  { a := p.a, h := -p.h, k := p.k }

theorem parabola_reflection :
  let original := Parabola.mk 2 1 (-4)
  let reflected := reflect_y_axis original
  reflected = Parabola.mk 2 (-1) (-4) := by sorry

end NUMINAMATH_CALUDE_parabola_reflection_l3204_320498


namespace NUMINAMATH_CALUDE_eighth_term_value_l3204_320477

/-- An arithmetic sequence with 30 terms, first term 5, and last term 86 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  let d := (86 - 5) / 29
  5 + (n - 1) * d

theorem eighth_term_value :
  arithmetic_sequence 8 = 592 / 29 :=
by sorry

end NUMINAMATH_CALUDE_eighth_term_value_l3204_320477


namespace NUMINAMATH_CALUDE_line_through_points_l3204_320459

def point_on_line (x y x1 y1 x2 y2 : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

theorem line_through_points : 
  point_on_line 4 8 8 16 2 4 ∧
  point_on_line 6 12 8 16 2 4 ∧
  point_on_line 10 20 8 16 2 4 ∧
  ¬ point_on_line 5 11 8 16 2 4 ∧
  ¬ point_on_line 3 7 8 16 2 4 :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l3204_320459


namespace NUMINAMATH_CALUDE_x_value_proof_l3204_320430

theorem x_value_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 4) (h2 : y^2 / z = 9) (h3 : z^2 / x = 16) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3204_320430


namespace NUMINAMATH_CALUDE_complex_product_minus_p_l3204_320417

theorem complex_product_minus_p :
  let P : ℂ := 7 + 3 * Complex.I
  let Q : ℂ := 2 * Complex.I
  let R : ℂ := 7 - 3 * Complex.I
  (P * Q * R) - P = 113 * Complex.I - 7 := by sorry

end NUMINAMATH_CALUDE_complex_product_minus_p_l3204_320417


namespace NUMINAMATH_CALUDE_train_passing_time_l3204_320483

/-- The time it takes for a train to pass a person moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) :
  train_length = 240 →
  train_speed = 100 * (5/18) →
  person_speed = 8 * (5/18) →
  (train_length / (train_speed + person_speed)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3204_320483


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l3204_320412

def f (x : ℝ) : ℝ := x^3

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l3204_320412


namespace NUMINAMATH_CALUDE_max_threes_in_selection_l3204_320453

/-- Represents the count of each card type in the selection -/
structure CardSelection :=
  (threes : ℕ)
  (fours : ℕ)
  (fives : ℕ)

/-- The problem constraints -/
def isValidSelection (s : CardSelection) : Prop :=
  s.threes + s.fours + s.fives = 8 ∧
  3 * s.threes + 4 * s.fours + 5 * s.fives = 27 ∧
  s.threes ≤ 10 ∧ s.fours ≤ 10 ∧ s.fives ≤ 10

/-- The theorem statement -/
theorem max_threes_in_selection :
  ∃ (s : CardSelection), isValidSelection s ∧
    (∀ (t : CardSelection), isValidSelection t → t.threes ≤ s.threes) ∧
    s.threes = 6 :=
sorry

end NUMINAMATH_CALUDE_max_threes_in_selection_l3204_320453


namespace NUMINAMATH_CALUDE_high_school_harriers_loss_percentage_l3204_320439

theorem high_school_harriers_loss_percentage
  (total_games : ℝ)
  (games_won : ℝ)
  (games_lost : ℝ)
  (games_tied : ℝ)
  (h1 : games_won / games_lost = 5 / 3)
  (h2 : games_tied = 0.2 * total_games)
  (h3 : total_games = games_won + games_lost + games_tied) :
  games_lost / total_games = 0.3 := by
sorry

end NUMINAMATH_CALUDE_high_school_harriers_loss_percentage_l3204_320439


namespace NUMINAMATH_CALUDE_find_B_l3204_320402

theorem find_B (A C B : ℤ) (h1 : A = 634) (h2 : A = C + 593) (h3 : B = C + 482) : B = 523 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l3204_320402


namespace NUMINAMATH_CALUDE_circle_alignment_exists_l3204_320431

/-- Represents a circle with a circumference of 100 cm -/
structure Circle :=
  (circumference : ℝ)
  (h_circumference : circumference = 100)

/-- Represents a set of marked points on a circle -/
structure MarkedPoints :=
  (circle : Circle)
  (num_points : ℕ)

/-- Represents a set of arcs on a circle -/
structure Arcs :=
  (circle : Circle)
  (total_length : ℝ)
  (h_length : total_length < 1)

/-- Represents an alignment of two circles -/
def Alignment := ℝ

/-- Checks if a marked point coincides with an arc for a given alignment -/
def coincides (mp : MarkedPoints) (a : Arcs) (alignment : Alignment) : Prop :=
  sorry

theorem circle_alignment_exists (c1 c2 : Circle) 
  (mp : MarkedPoints) (a : Arcs) 
  (h_mp : mp.circle = c1) (h_a : a.circle = c2) 
  (h_num_points : mp.num_points = 100) :
  ∃ (alignment : Alignment), ∀ (p : ℕ) (h_p : p < mp.num_points), 
    ¬ coincides mp a alignment :=
sorry

end NUMINAMATH_CALUDE_circle_alignment_exists_l3204_320431


namespace NUMINAMATH_CALUDE_books_sold_on_wednesday_l3204_320458

theorem books_sold_on_wednesday 
  (initial_stock : ℕ) 
  (sold_monday : ℕ) 
  (sold_tuesday : ℕ) 
  (sold_thursday : ℕ) 
  (sold_friday : ℕ) 
  (unsold : ℕ) 
  (h1 : initial_stock = 800)
  (h2 : sold_monday = 60)
  (h3 : sold_tuesday = 10)
  (h4 : sold_thursday = 44)
  (h5 : sold_friday = 66)
  (h6 : unsold = 600) :
  initial_stock - unsold - (sold_monday + sold_tuesday + sold_thursday + sold_friday) = 20 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_on_wednesday_l3204_320458


namespace NUMINAMATH_CALUDE_circle_parameter_value_l3204_320446

theorem circle_parameter_value (θ : Real) : 
  0 ≤ θ ∧ θ < 2 * Real.pi →
  4 * Real.cos θ = -2 →
  4 * Real.sin θ = 2 * Real.sqrt 3 →
  θ = 2 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_circle_parameter_value_l3204_320446


namespace NUMINAMATH_CALUDE_power_27_mod_13_l3204_320411

theorem power_27_mod_13 : 27^482 ≡ 1 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_power_27_mod_13_l3204_320411


namespace NUMINAMATH_CALUDE_distance_to_nearest_town_l3204_320452

theorem distance_to_nearest_town (d : ℝ) :
  (¬ (d ≥ 6)) ∧ (¬ (d ≤ 5)) ∧ (¬ (d ≤ 4)) → 5 < d ∧ d < 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_town_l3204_320452


namespace NUMINAMATH_CALUDE_inequality_proof_l3204_320465

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.exp (-3))
  (hb : b = Real.log 1.02)
  (hc : c = Real.sin 0.04) : 
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3204_320465


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3204_320470

theorem z_in_first_quadrant (z : ℂ) : 
  (z - 2*I) * (1 + I) = Complex.abs (1 - Real.sqrt 3 * I) → 
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3204_320470


namespace NUMINAMATH_CALUDE_eighteen_picks_required_l3204_320408

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  white : Nat
  black : Nat
  yellow : Nat

/-- The minimum number of picks required to guarantee at least one ball of each color -/
def minPicksRequired (counts : BallCounts) : Nat :=
  counts.white + counts.black + 1

/-- Theorem stating that for the given ball counts, 18 picks are required -/
theorem eighteen_picks_required (counts : BallCounts) 
  (h_white : counts.white = 8)
  (h_black : counts.black = 9)
  (h_yellow : counts.yellow = 7) : 
  minPicksRequired counts = 18 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_picks_required_l3204_320408


namespace NUMINAMATH_CALUDE_markese_earnings_l3204_320409

/-- Proves that Markese earned 16 dollars given the conditions of the problem -/
theorem markese_earnings (E : ℕ) 
  (h1 : E - 5 + E = 37) : 
  E - 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_markese_earnings_l3204_320409


namespace NUMINAMATH_CALUDE_total_beakers_l3204_320432

theorem total_beakers (copper_beakers : ℕ) (drops_per_test : ℕ) (total_drops : ℕ) (non_copper_tested : ℕ) :
  copper_beakers = 8 →
  drops_per_test = 3 →
  total_drops = 45 →
  non_copper_tested = 7 →
  copper_beakers + non_copper_tested = total_drops / drops_per_test :=
by sorry

end NUMINAMATH_CALUDE_total_beakers_l3204_320432


namespace NUMINAMATH_CALUDE_expression_evaluation_l3204_320474

theorem expression_evaluation : 8 - 6 / (4 - 2) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3204_320474


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l3204_320471

/-- The number of ways to choose 4 socks from 6 socks (where one is blue and the rest are different colors), 
    such that at least one chosen sock is blue. -/
def choose_socks (total_socks : ℕ) (blue_socks : ℕ) (choose : ℕ) : ℕ :=
  Nat.choose (total_socks - blue_socks) (choose - 1)

/-- Theorem stating that there are 10 ways to choose 4 socks from 6 socks, 
    where at least one is blue. -/
theorem sock_selection_theorem :
  choose_socks 6 1 4 = 10 := by
  sorry

#eval choose_socks 6 1 4

end NUMINAMATH_CALUDE_sock_selection_theorem_l3204_320471


namespace NUMINAMATH_CALUDE_school_dance_relationship_l3204_320463

theorem school_dance_relationship (b g : ℕ) : 
  (b > 0) →  -- There is at least one boy
  (g ≥ 7) →  -- There are at least 7 girls (for the first boy)
  (∀ i : ℕ, i > 0 ∧ i ≤ b → (7 + i - 1) ≤ g) →  -- Each boy can dance with his required number of girls
  (7 + b - 1 = g) →  -- The last boy dances with all girls
  b = g - 6 := by
sorry

end NUMINAMATH_CALUDE_school_dance_relationship_l3204_320463


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_six_mod_seventeen_l3204_320447

theorem least_five_digit_congruent_to_six_mod_seventeen :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧  -- Five-digit positive integer
    (n % 17 = 6) ∧              -- Congruent to 6 (mod 17)
    (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m % 17 = 6) → n ≤ m) ∧  -- Least such number
    n = 10002                   -- The number is 10,002
  := by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_six_mod_seventeen_l3204_320447


namespace NUMINAMATH_CALUDE_midpoint_chain_l3204_320404

/-- Given points A, B, C, D, E, F on a line segment AB, where:
    C is the midpoint of AB,
    D is the midpoint of AC,
    E is the midpoint of AD,
    F is the midpoint of AE,
    and AF = 3,
    prove that AB = 48. -/
theorem midpoint_chain (A B C D E F : ℝ) 
  (hC : C = (A + B) / 2) 
  (hD : D = (A + C) / 2)
  (hE : E = (A + D) / 2)
  (hF : F = (A + E) / 2)
  (hAF : F - A = 3) : 
  B - A = 48 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_chain_l3204_320404


namespace NUMINAMATH_CALUDE_simplify_expression_l3204_320468

theorem simplify_expression :
  let x : ℝ := Real.sqrt 2
  let y : ℝ := Real.sqrt 3
  (x + 1) ^ (y - 1) / (x - 1) ^ (y + 1) = 3 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3204_320468


namespace NUMINAMATH_CALUDE_range_of_t_l3204_320401

-- Define set A
def A : Set ℝ := {x | (1/4 : ℝ) ≤ 2^x ∧ 2^x ≤ (1/2 : ℝ)}

-- Define set B (parameterized by t)
def B (t : ℝ) : Set ℝ := {x | x^2 - 2*t*x + 1 ≤ 0}

-- Theorem statement
theorem range_of_t (t : ℝ) : 
  (A ∩ B t = A) ↔ t ∈ Set.Iic (-5/4 : ℝ) := by sorry


end NUMINAMATH_CALUDE_range_of_t_l3204_320401


namespace NUMINAMATH_CALUDE_vehicle_distance_time_l3204_320494

/-- Proves that two vehicles traveling in opposite directions for 4 hours
    will be 384 miles apart, given their respective speeds -/
theorem vehicle_distance_time (slower_speed faster_speed : ℝ) 
    (h1 : slower_speed = 44)
    (h2 : faster_speed = slower_speed + 8)
    (distance : ℝ) (h3 : distance = 384) : 
    (slower_speed + faster_speed) * 4 = distance := by
  sorry

end NUMINAMATH_CALUDE_vehicle_distance_time_l3204_320494


namespace NUMINAMATH_CALUDE_smiley_face_tulips_l3204_320451

/-- Calculates the total number of tulips needed for a smiley face design. -/
def total_tulips : ℕ :=
  let red_eyes : ℕ := 8 * 2
  let purple_eyebrows : ℕ := 5 * 2
  let red_nose : ℕ := 12
  let red_smile : ℕ := 18
  let yellow_background : ℕ := 9 * red_smile
  red_eyes + purple_eyebrows + red_nose + red_smile + yellow_background

/-- Theorem stating that the total number of tulips needed is 218. -/
theorem smiley_face_tulips : total_tulips = 218 := by
  sorry

end NUMINAMATH_CALUDE_smiley_face_tulips_l3204_320451


namespace NUMINAMATH_CALUDE_betty_age_l3204_320441

/-- Given the relationships between Albert, Mary, and Betty's ages, prove Betty's age -/
theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 14) :
  betty = 7 := by
  sorry

end NUMINAMATH_CALUDE_betty_age_l3204_320441


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3204_320423

def U : Set Nat := {x | x > 0 ∧ x < 9}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 4, 5, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3204_320423


namespace NUMINAMATH_CALUDE_odd_numbers_theorem_l3204_320416

theorem odd_numbers_theorem (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_numbers_theorem_l3204_320416


namespace NUMINAMATH_CALUDE_only_point_A_in_region_l3204_320428

def plane_region (x y : ℝ) : Prop := x + y - 1 < 0

def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (2, 4)
def point_C : ℝ × ℝ := (-1, 4)
def point_D : ℝ × ℝ := (1, 8)

theorem only_point_A_in_region :
  plane_region point_A.1 point_A.2 ∧
  ¬plane_region point_B.1 point_B.2 ∧
  ¬plane_region point_C.1 point_C.2 ∧
  ¬plane_region point_D.1 point_D.2 := by
  sorry

end NUMINAMATH_CALUDE_only_point_A_in_region_l3204_320428


namespace NUMINAMATH_CALUDE_estimate_proportion_approx_5_7_percent_l3204_320442

/-- Represents the survey data and population information -/
structure SurveyData where
  total_households : ℕ
  ordinary_households : ℕ
  high_income_households : ℕ
  sample_ordinary : ℕ
  sample_high_income : ℕ
  total_with_3plus_housing : ℕ
  ordinary_with_3plus_housing : ℕ
  high_income_with_3plus_housing : ℕ

/-- Calculates the estimated proportion of households with 3+ housing sets -/
def estimate_proportion (data : SurveyData) : ℝ :=
  sorry

/-- Theorem stating that the estimated proportion is approximately 5.7% -/
theorem estimate_proportion_approx_5_7_percent (data : SurveyData)
  (h1 : data.total_households = 100000)
  (h2 : data.ordinary_households = 99000)
  (h3 : data.high_income_households = 1000)
  (h4 : data.sample_ordinary = 990)
  (h5 : data.sample_high_income = 100)
  (h6 : data.total_with_3plus_housing = 120)
  (h7 : data.ordinary_with_3plus_housing = 50)
  (h8 : data.high_income_with_3plus_housing = 70) :
  ∃ ε > 0, |estimate_proportion data - 0.057| < ε :=
sorry

end NUMINAMATH_CALUDE_estimate_proportion_approx_5_7_percent_l3204_320442


namespace NUMINAMATH_CALUDE_max_product_logarithms_l3204_320444

/-- Given a, b, c > 1 satisfying the given equations, the maximum value of lg a · lg c is 16/3 -/
theorem max_product_logarithms (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq1 : Real.log a / Real.log 10 + Real.log c / Real.log b = 3)
  (eq2 : Real.log b / Real.log 10 + Real.log c / Real.log a = 4) :
  (Real.log a / Real.log 10) * (Real.log c / Real.log 10) ≤ 16/3 :=
by sorry

end NUMINAMATH_CALUDE_max_product_logarithms_l3204_320444


namespace NUMINAMATH_CALUDE_danny_thrice_jane_age_l3204_320433

/-- Proves that Danny was thrice as old as Jane 19 years ago -/
theorem danny_thrice_jane_age (danny_age : ℕ) (jane_age : ℕ) 
  (h1 : danny_age = 40) (h2 : jane_age = 26) : 
  ∃ x : ℕ, x = 19 ∧ (danny_age - x) = 3 * (jane_age - x) :=
by sorry

end NUMINAMATH_CALUDE_danny_thrice_jane_age_l3204_320433


namespace NUMINAMATH_CALUDE_justine_colored_sheets_l3204_320418

/-- Given a total number of sheets and binders, calculate the number of sheets Justine colored. -/
def sheets_colored (total_sheets : ℕ) (num_binders : ℕ) : ℕ :=
  let sheets_per_binder := total_sheets / num_binders
  (2 * sheets_per_binder) / 3

/-- Prove that Justine colored 356 sheets given the problem conditions. -/
theorem justine_colored_sheets :
  sheets_colored 3750 7 = 356 := by
  sorry

end NUMINAMATH_CALUDE_justine_colored_sheets_l3204_320418


namespace NUMINAMATH_CALUDE_reverse_digits_problem_l3204_320422

/-- Given a two-digit number, returns the number formed by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The problem statement -/
theorem reverse_digits_problem : ∃ (v : ℕ), 57 + v = reverse_digits 57 :=
  sorry

end NUMINAMATH_CALUDE_reverse_digits_problem_l3204_320422


namespace NUMINAMATH_CALUDE_ticket_price_values_l3204_320419

theorem ticket_price_values (x : ℕ) : 
  (∃ (a b c : ℕ), x * a = 72 ∧ x * b = 90 ∧ x * c = 45) ↔ 
  (x = 1 ∨ x = 3 ∨ x = 9) := by
sorry

end NUMINAMATH_CALUDE_ticket_price_values_l3204_320419


namespace NUMINAMATH_CALUDE_remote_sensing_primary_for_sea_level_info_l3204_320464

/-- Represents different technologies used in geographic information systems -/
inductive GISTechnology
  | RemoteSensing
  | GPS
  | GIS
  | DigitalEarth

/-- Represents the capability of a technology to acquire sea level rise information -/
def can_acquire_sea_level_info (tech : GISTechnology) : Prop :=
  match tech with
  | GISTechnology.RemoteSensing => true
  | _ => false

/-- Theorem stating that Remote Sensing is the primary technology for acquiring sea level rise information -/
theorem remote_sensing_primary_for_sea_level_info :
  ∀ (tech : GISTechnology),
    can_acquire_sea_level_info tech → tech = GISTechnology.RemoteSensing :=
by
  sorry


end NUMINAMATH_CALUDE_remote_sensing_primary_for_sea_level_info_l3204_320464


namespace NUMINAMATH_CALUDE_anya_lost_games_correct_l3204_320443

/-- Represents a girl in the table tennis game --/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- The total number of games played --/
def total_games : ℕ := 19

/-- The number of games each girl played --/
def games_played (g : Girl) : ℕ :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- A game is represented by its number and the two girls who played --/
structure Game where
  number : ℕ
  player1 : Girl
  player2 : Girl

/-- The set of all games played --/
def all_games : Set Game := sorry

/-- The set of games where Anya lost --/
def anya_lost_games : Set ℕ := {4, 8, 12, 16}

/-- The main theorem to prove --/
theorem anya_lost_games_correct :
  ∀ (g : Game), g ∈ all_games → 
    (g.player1 = Girl.Anya ∨ g.player2 = Girl.Anya) → 
    g.number ∈ anya_lost_games :=
  sorry

end NUMINAMATH_CALUDE_anya_lost_games_correct_l3204_320443


namespace NUMINAMATH_CALUDE_medium_revenue_is_24_l3204_320475

/-- Represents the revenue from Tonya's lemonade stand -/
structure LemonadeRevenue where
  total : ℕ
  small : ℕ
  large_cups : ℕ
  small_price : ℕ
  medium_price : ℕ
  large_price : ℕ

/-- Calculates the revenue from medium lemonades -/
def medium_revenue (r : LemonadeRevenue) : ℕ :=
  r.total - r.small - (r.large_cups * r.large_price)

/-- Theorem: The revenue from medium lemonades is 24 -/
theorem medium_revenue_is_24 (r : LemonadeRevenue) 
  (h1 : r.total = 50)
  (h2 : r.small = 11)
  (h3 : r.large_cups = 5)
  (h4 : r.small_price = 1)
  (h5 : r.medium_price = 2)
  (h6 : r.large_price = 3) :
  medium_revenue r = 24 := by
  sorry

end NUMINAMATH_CALUDE_medium_revenue_is_24_l3204_320475


namespace NUMINAMATH_CALUDE_remaining_budget_l3204_320438

/-- Proves that given a weekly food budget of $80, after purchasing a $12 bucket of fried chicken
    and 5 pounds of beef at $3 per pound, the remaining budget is $53. -/
theorem remaining_budget (weekly_budget : ℕ) (chicken_cost : ℕ) (beef_price : ℕ) (beef_amount : ℕ) :
  weekly_budget = 80 →
  chicken_cost = 12 →
  beef_price = 3 →
  beef_amount = 5 →
  weekly_budget - (chicken_cost + beef_price * beef_amount) = 53 := by
  sorry

end NUMINAMATH_CALUDE_remaining_budget_l3204_320438


namespace NUMINAMATH_CALUDE_circle_equation_l3204_320487

/-- The standard equation of a circle with center (-2, 1) passing through (0, 1) -/
theorem circle_equation :
  let center : ℝ × ℝ := (-2, 1)
  let point_on_circle : ℝ × ℝ := (0, 1)
  ∀ (x y : ℝ),
    (x + 2)^2 + (y - 1)^2 = 4 ↔
    (x - center.1)^2 + (y - center.2)^2 = (point_on_circle.1 - center.1)^2 + (point_on_circle.2 - center.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3204_320487


namespace NUMINAMATH_CALUDE_proposition_implication_l3204_320406

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l3204_320406


namespace NUMINAMATH_CALUDE_reggies_lost_games_l3204_320495

/-- Given the conditions of Reggie's marble game, prove the number of games he lost. -/
theorem reggies_lost_games 
  (total_games : ℕ) 
  (initial_marbles : ℕ) 
  (bet_per_game : ℕ) 
  (final_marbles : ℕ) 
  (h1 : total_games = 9)
  (h2 : initial_marbles = 100)
  (h3 : bet_per_game = 10)
  (h4 : final_marbles = 90) :
  (initial_marbles - final_marbles) / bet_per_game = 1 :=
by sorry

end NUMINAMATH_CALUDE_reggies_lost_games_l3204_320495


namespace NUMINAMATH_CALUDE_persimmons_in_box_l3204_320486

/-- Given a box containing apples and persimmons, prove the number of persimmons. -/
theorem persimmons_in_box (apples : ℕ) (persimmons : ℕ) : apples = 3 → persimmons = 2 → persimmons = 2 := by
  sorry

end NUMINAMATH_CALUDE_persimmons_in_box_l3204_320486


namespace NUMINAMATH_CALUDE_worth_of_cloth_is_8540_l3204_320481

/-- Commission rates and sales data for an agent --/
structure SalesData where
  cloth_rate : ℝ
  electronics_rate_low : ℝ
  electronics_rate_high : ℝ
  electronics_threshold : ℝ
  stationery_rate_low : ℝ
  stationery_rate_high : ℝ
  stationery_threshold : ℕ
  total_commission : ℝ
  electronics_sales : ℝ
  stationery_units : ℕ

/-- Calculate the worth of cloth sold given sales data --/
def worth_of_cloth_sold (data : SalesData) : ℝ :=
  sorry

/-- Theorem stating that the worth of cloth sold is 8540 given the specific sales data --/
theorem worth_of_cloth_is_8540 (data : SalesData) 
  (h1 : data.cloth_rate = 0.025)
  (h2 : data.electronics_rate_low = 0.035)
  (h3 : data.electronics_rate_high = 0.045)
  (h4 : data.electronics_threshold = 3000)
  (h5 : data.stationery_rate_low = 10)
  (h6 : data.stationery_rate_high = 15)
  (h7 : data.stationery_threshold = 5)
  (h8 : data.total_commission = 418)
  (h9 : data.electronics_sales = 3100)
  (h10 : data.stationery_units = 8) :
  worth_of_cloth_sold data = 8540 := by
  sorry

end NUMINAMATH_CALUDE_worth_of_cloth_is_8540_l3204_320481


namespace NUMINAMATH_CALUDE_preimage_of_three_one_l3204_320462

/-- A mapping from A to B -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (2, 1) is the preimage of (3, 1) under f -/
theorem preimage_of_three_one :
  f (2, 1) = (3, 1) ∧ ∀ p : ℝ × ℝ, f p = (3, 1) → p = (2, 1) :=
by sorry

end NUMINAMATH_CALUDE_preimage_of_three_one_l3204_320462


namespace NUMINAMATH_CALUDE_constant_in_toll_formula_l3204_320415

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ :=
  constant + 0.50 * (x - 2 : ℝ)

/-- The number of axles on an 18-wheel truck -/
def axles_18_wheel_truck : ℕ := 9

/-- The toll for an 18-wheel truck -/
def toll_18_wheel_truck : ℝ := 5

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll_formula constant axles_18_wheel_truck = toll_18_wheel_truck ∧
    constant = 1.50 := by sorry

end NUMINAMATH_CALUDE_constant_in_toll_formula_l3204_320415


namespace NUMINAMATH_CALUDE_solution_pairs_l3204_320499

theorem solution_pairs (x y : ℕ+) : 
  let d := Nat.gcd x y
  x * y * d = x + y + d^2 ↔ (x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l3204_320499


namespace NUMINAMATH_CALUDE_tile_count_theorem_l3204_320455

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledRectangle where
  length : ℕ
  width : ℕ
  diagonal_tiles : ℕ

/-- The condition that one side is twice the length of the other. -/
def double_side (rect : TiledRectangle) : Prop :=
  rect.length = 2 * rect.width

/-- The number of tiles on the diagonals. -/
def diagonal_count (rect : TiledRectangle) : ℕ :=
  rect.diagonal_tiles

/-- The total number of tiles covering the floor. -/
def total_tiles (rect : TiledRectangle) : ℕ :=
  rect.length * rect.width

/-- The main theorem stating the problem. -/
theorem tile_count_theorem (rect : TiledRectangle) :
  double_side rect → diagonal_count rect = 49 → total_tiles rect = 50 := by
  sorry


end NUMINAMATH_CALUDE_tile_count_theorem_l3204_320455


namespace NUMINAMATH_CALUDE_sine_sum_equality_l3204_320496

theorem sine_sum_equality : 
  Real.sin (45 * π / 180) * Real.sin (105 * π / 180) + 
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sine_sum_equality_l3204_320496


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l3204_320440

theorem continued_fraction_evaluation :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (3 + Real.sqrt 69) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l3204_320440


namespace NUMINAMATH_CALUDE_number_of_persons_working_prove_number_of_persons_working_l3204_320484

/-- The number of days it takes for some persons to finish the job -/
def group_days : ℕ := 8

/-- The number of days it takes for the first person to finish the job -/
def first_person_days : ℕ := 24

/-- The number of days it takes for the second person to finish the job -/
def second_person_days : ℕ := 12

/-- The work rate of a person is the fraction of the job they can complete in one day -/
def work_rate (days : ℕ) : ℚ := 1 / days

/-- The theorem stating that the number of persons working on the job is 2 -/
theorem number_of_persons_working : ℕ :=
  2

/-- Proof that the number of persons working on the job is 2 -/
theorem prove_number_of_persons_working :
  work_rate group_days = work_rate first_person_days + work_rate second_person_days →
  number_of_persons_working = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_persons_working_prove_number_of_persons_working_l3204_320484


namespace NUMINAMATH_CALUDE_wendy_facebook_pictures_l3204_320480

theorem wendy_facebook_pictures (total_albums : ℕ) (pics_in_first_album : ℕ) 
  (pics_per_other_album : ℕ) (other_albums : ℕ) :
  total_albums = other_albums + 1 →
  pics_in_first_album = 44 →
  pics_per_other_album = 7 →
  other_albums = 5 →
  pics_in_first_album + other_albums * pics_per_other_album = 79 := by
  sorry

end NUMINAMATH_CALUDE_wendy_facebook_pictures_l3204_320480


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3204_320476

/-- A line passing through a point and parallel to another line -/
def parallel_line (p : ℝ × ℝ) (m : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 = m * q.1 + (p.2 - m * p.1)}

theorem parallel_line_equation :
  let p : ℝ × ℝ := (0, 7)
  let m : ℝ := -4
  parallel_line p m = {q : ℝ × ℝ | q.2 = -4 * q.1 + 7} := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3204_320476


namespace NUMINAMATH_CALUDE_sequence_equality_l3204_320478

theorem sequence_equality (a b : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = 2 * b n - a n)
  (h2 : ∀ n : ℕ, b (n + 1) = 2 * a n - b n)
  (h3 : ∀ n : ℕ, a n > 0) :
  a 1 = b 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l3204_320478


namespace NUMINAMATH_CALUDE_third_derivative_y_l3204_320413

noncomputable def y (x : ℝ) : ℝ := (1 / x) * Real.sin (2 * x)

theorem third_derivative_y (x : ℝ) (hx : x ≠ 0) :
  (deriv^[3] y) x = 
    ((-6 / x^4 + 12 / x^2) * Real.sin (2 * x) + 
     (12 / x^3 - 8 / x) * Real.cos (2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_third_derivative_y_l3204_320413


namespace NUMINAMATH_CALUDE_prob_at_least_four_girls_value_l3204_320466

def num_children : ℕ := 7
def prob_girl : ℚ := 3/5
def prob_boy : ℚ := 2/5

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def prob_at_least_four_girls : ℚ :=
  (binomial num_children 4 : ℚ) * (prob_girl ^ 4) * (prob_boy ^ 3) +
  (binomial num_children 5 : ℚ) * (prob_girl ^ 5) * (prob_boy ^ 2) +
  (binomial num_children 6 : ℚ) * (prob_girl ^ 6) * (prob_boy ^ 1) +
  (binomial num_children 7 : ℚ) * (prob_girl ^ 7) * (prob_boy ^ 0)

theorem prob_at_least_four_girls_value : prob_at_least_four_girls = 35325/78125 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_four_girls_value_l3204_320466


namespace NUMINAMATH_CALUDE_disprove_square_implies_greater_l3204_320482

theorem disprove_square_implies_greater : ∃ a b : ℝ, a^2 > b^2 ∧ a ≤ b :=
  let a := -3
  let b := 2
  have h1 : a^2 > b^2 := by sorry
  have h2 : a ≤ b := by sorry
  ⟨a, b, h1, h2⟩

#check disprove_square_implies_greater

end NUMINAMATH_CALUDE_disprove_square_implies_greater_l3204_320482


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l3204_320493

/-- Given three points on an inverse proportion function, prove their y-coordinates' relationship -/
theorem inverse_proportion_y_relationship
  (k : ℝ) (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ)
  (h_k : k < 0)
  (h_x : x₁ < x₂ ∧ x₂ < 0 ∧ 0 < x₃)
  (h_y₁ : y₁ = k / x₁)
  (h_y₂ : y₂ = k / x₂)
  (h_y₃ : y₃ = k / x₃) :
  y₂ > y₁ ∧ y₁ > y₃ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l3204_320493


namespace NUMINAMATH_CALUDE_permutations_of_six_l3204_320414

theorem permutations_of_six (n : Nat) : n = 6 → Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_l3204_320414


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l3204_320445

theorem empty_solution_set_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l3204_320445


namespace NUMINAMATH_CALUDE_x_value_l3204_320403

theorem x_value : ∃ x : ℝ, x ≠ 0 ∧ x = 3 * (1/x * (-x)) + 5 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3204_320403


namespace NUMINAMATH_CALUDE_michelle_crayons_l3204_320448

theorem michelle_crayons (num_boxes : ℕ) (crayons_per_box : ℕ) (h1 : num_boxes = 7) (h2 : crayons_per_box = 5) :
  num_boxes * crayons_per_box = 35 := by
  sorry

end NUMINAMATH_CALUDE_michelle_crayons_l3204_320448


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l3204_320426

/-- Prove that if an ellipse and a hyperbola are tangent, then the parameter m of the hyperbola is 8 -/
theorem tangent_ellipse_hyperbola (x y m : ℝ) :
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 1) →  -- Ellipse and hyperbola equations
  (∀ x y, x^2 + 9*y^2 = 9 → x^2 - m*(y+3)^2 ≤ 1) →  -- Tangency condition
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 1) →  -- Point of tangency exists
  m = 8 :=
by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l3204_320426


namespace NUMINAMATH_CALUDE_dress_trim_cuff_length_l3204_320435

/-- Proves that the length of each cuff is 50 cm given the dress trimming conditions --/
theorem dress_trim_cuff_length :
  let hem_length : ℝ := 300
  let waist_length : ℝ := hem_length / 3
  let neck_ruffles : ℕ := 5
  let ruffle_length : ℝ := 20
  let lace_cost_per_meter : ℝ := 6
  let total_spent : ℝ := 36
  let total_lace_length : ℝ := total_spent / lace_cost_per_meter * 100
  let hem_waist_neck_length : ℝ := hem_length + waist_length + (neck_ruffles : ℝ) * ruffle_length
  let cuff_total_length : ℝ := total_lace_length - hem_waist_neck_length
  cuff_total_length / 2 = 50 := by sorry

end NUMINAMATH_CALUDE_dress_trim_cuff_length_l3204_320435


namespace NUMINAMATH_CALUDE_other_communities_count_l3204_320405

/-- Given a school with 300 boys, calculate the number of boys belonging to other communities -/
theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 300 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 54 :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l3204_320405


namespace NUMINAMATH_CALUDE_six_graduates_distribution_l3204_320460

/-- The number of ways to distribute n graduates among 2 employers, 
    with each employer receiving at least k graduates -/
def distribution_schemes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 graduates among 2 employers, 
    with each employer receiving at least 2 graduates, is 50 -/
theorem six_graduates_distribution : distribution_schemes 6 2 = 50 := by sorry

end NUMINAMATH_CALUDE_six_graduates_distribution_l3204_320460


namespace NUMINAMATH_CALUDE_chicken_coops_count_l3204_320454

theorem chicken_coops_count (chickens_per_coop : ℕ) (total_chickens : ℕ) 
  (h1 : chickens_per_coop = 60) 
  (h2 : total_chickens = 540) : 
  total_chickens / chickens_per_coop = 9 := by
  sorry

end NUMINAMATH_CALUDE_chicken_coops_count_l3204_320454


namespace NUMINAMATH_CALUDE_complex_division_l3204_320497

theorem complex_division (i : ℂ) (h : i^2 = -1) : 
  1 / (1 + i)^2 = -1/2 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_l3204_320497


namespace NUMINAMATH_CALUDE_road_building_time_l3204_320489

/-- Given that 60 workers can build a road in 5 days, prove that 40 workers
    working at the same rate will take 7.5 days to build the same road. -/
theorem road_building_time (workers_initial : ℕ) (days_initial : ℝ)
    (workers_new : ℕ) (days_new : ℝ) : 
    workers_initial = 60 → days_initial = 5 → workers_new = 40 → 
    (workers_initial : ℝ) * days_initial = workers_new * days_new →
    days_new = 7.5 := by
  sorry

#check road_building_time

end NUMINAMATH_CALUDE_road_building_time_l3204_320489


namespace NUMINAMATH_CALUDE_regular_pyramid_lateral_area_l3204_320449

/-- Theorem: The lateral surface area of a regular pyramid equals the area of the base
    divided by the cosine of the dihedral angle between a lateral face and the base. -/
theorem regular_pyramid_lateral_area 
  (n : ℕ) -- number of sides in the base
  (S : ℝ) -- area of one lateral face
  (A : ℝ) -- area of the base
  (α : ℝ) -- dihedral angle between a lateral face and the base
  (h1 : n > 0) -- the pyramid has at least 3 sides
  (h2 : S > 0) -- lateral face area is positive
  (h3 : A > 0) -- base area is positive
  (h4 : 0 < α ∧ α < π / 2) -- dihedral angle is between 0 and π/2
  : n * S = A / Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_regular_pyramid_lateral_area_l3204_320449


namespace NUMINAMATH_CALUDE_workshop_workers_count_l3204_320450

/-- Given a workshop with workers including technicians, prove the total number of workers. -/
theorem workshop_workers_count 
  (total_avg : ℝ)  -- Average salary of all workers
  (tech_count : ℕ) -- Number of technicians
  (tech_avg : ℝ)   -- Average salary of technicians
  (rest_avg : ℝ)   -- Average salary of the rest of the workers
  (h1 : total_avg = 850)
  (h2 : tech_count = 7)
  (h3 : tech_avg = 1000)
  (h4 : rest_avg = 780) :
  ∃ (total_workers : ℕ), total_workers = 22 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l3204_320450


namespace NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l3204_320437

/-- An arithmetic progression where the sum of the first twenty terms
    is six times the sum of the first ten terms -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_condition : (20 * a + 190 * d) = 6 * (10 * a + 45 * d)

/-- The ratio of the first term to the common difference is 2 -/
theorem ratio_first_term_to_common_difference
  (ap : ArithmeticProgression) : ap.a / ap.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l3204_320437
