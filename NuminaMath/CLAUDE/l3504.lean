import Mathlib

namespace NUMINAMATH_CALUDE_total_gold_spent_l3504_350474

-- Define the quantities and prices
def gary_grams : ℝ := 30
def gary_price : ℝ := 15
def anna_grams : ℝ := 50
def anna_price : ℝ := 20
def lisa_grams : ℝ := 40
def lisa_price : ℝ := 15
def john_grams : ℝ := 60
def john_price : ℝ := 18

-- Define conversion rates
def euro_to_dollar : ℝ := 1.1
def pound_to_dollar : ℝ := 1.3

-- Define the total spent function
def total_spent : ℝ := 
  gary_grams * gary_price + 
  anna_grams * anna_price + 
  lisa_grams * lisa_price * euro_to_dollar + 
  john_grams * john_price * pound_to_dollar

-- Theorem statement
theorem total_gold_spent : total_spent = 3514 := by
  sorry

end NUMINAMATH_CALUDE_total_gold_spent_l3504_350474


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_expansion_l3504_350422

theorem perfect_square_trinomial_expansion (x : ℝ) : 
  let a : ℝ := x
  let b : ℝ := (1 : ℝ) / 2
  2 * a * b = x := by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_expansion_l3504_350422


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l3504_350448

theorem two_digit_reverse_sum (x y n : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ a b : ℕ, x = 10 * a + b ∧ y = 10 * b + a ∧ a ≠ 0 ∧ b ≠ 0) →  -- y is reverse of x
  (∃ a b : ℕ, x = 10 * a + b ∧ a + b = 8) →  -- sum of digits of x is 8
  x^2 - y^2 = n^2 →  -- x^2 - y^2 = n^2
  x + y + n = 144 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l3504_350448


namespace NUMINAMATH_CALUDE_marble_ratio_l3504_350478

/-- Represents the number of marbles each person has -/
structure Marbles where
  you : ℕ
  brother : ℕ
  friend : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : Marbles) : Prop :=
  m.you = 16 ∧
  m.you + m.brother + m.friend = 63 ∧
  m.you - 2 = 2 * (m.brother + 2) ∧
  ∃ k : ℕ, m.friend = k * m.you

/-- The theorem to prove -/
theorem marble_ratio (m : Marbles) (h : marble_problem m) :
  m.friend * 8 = m.you * 21 := by
  sorry


end NUMINAMATH_CALUDE_marble_ratio_l3504_350478


namespace NUMINAMATH_CALUDE_bella_roses_l3504_350441

def dozen : ℕ := 12

def roses_from_parents : ℕ := 2 * dozen

def number_of_friends : ℕ := 10

def roses_per_friend : ℕ := 2

def total_roses : ℕ := roses_from_parents + number_of_friends * roses_per_friend

theorem bella_roses : total_roses = 44 := by sorry

end NUMINAMATH_CALUDE_bella_roses_l3504_350441


namespace NUMINAMATH_CALUDE_unique_tangent_circle_l3504_350469

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Theorem: There exists exactly one circle of radius 4 that is tangent to two circles of radius 2
    which are tangent to each other, at their point of tangency -/
theorem unique_tangent_circle (c1 c2 : Circle) : 
  c1.radius = 2 → 
  c2.radius = 2 → 
  are_tangent c1 c2 → 
  ∃! c : Circle, c.radius = 4 ∧ are_tangent c c1 ∧ are_tangent c c2 :=
sorry

end NUMINAMATH_CALUDE_unique_tangent_circle_l3504_350469


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3504_350418

/-- A quadratic equation in terms of x is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation (x-1)(x-2)=0 --/
def f (x : ℝ) : ℝ := (x - 1) * (x - 2)

/-- Theorem stating that f is a quadratic equation --/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l3504_350418


namespace NUMINAMATH_CALUDE_tea_price_calculation_l3504_350409

theorem tea_price_calculation (coffee_customers : ℕ) (tea_customers : ℕ) (coffee_price : ℚ) (total_revenue : ℚ) :
  coffee_customers = 7 →
  tea_customers = 8 →
  coffee_price = 5 →
  total_revenue = 67 →
  ∃ tea_price : ℚ, tea_price = 4 ∧ coffee_customers * coffee_price + tea_customers * tea_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_tea_price_calculation_l3504_350409


namespace NUMINAMATH_CALUDE_x_values_l3504_350496

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- State the theorem
theorem x_values (x : ℝ) (h : A x ∩ B x = B x) : x = 0 ∨ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l3504_350496


namespace NUMINAMATH_CALUDE_nth_monomial_formula_l3504_350411

/-- A sequence of monomials is defined as follows:
    For n = 1: 2x
    For n = 2: -4x^2
    For n = 3: 6x^3
    For n = 4: -8x^4
    For n = 5: 10x^5
    ...
    This function represents the coefficient of the nth monomial in the sequence. -/
def monomial_coefficient (n : ℕ) : ℤ :=
  (-1)^(n+1) * (2*n)

/-- This function represents the exponent of x in the nth monomial of the sequence. -/
def monomial_exponent (n : ℕ) : ℕ := n

/-- This theorem states that the nth monomial in the sequence
    can be expressed as (-1)^(n+1) * 2n * x^n for any positive integer n. -/
theorem nth_monomial_formula (n : ℕ) (h : n > 0) :
  monomial_coefficient n = (-1)^(n+1) * (2*n) ∧ monomial_exponent n = n :=
sorry

end NUMINAMATH_CALUDE_nth_monomial_formula_l3504_350411


namespace NUMINAMATH_CALUDE_four_heads_before_three_tails_l3504_350424

/-- The probability of getting heads or tails in a fair coin flip -/
def p_head : ℚ := 1/2
def p_tail : ℚ := 1/2

/-- The probability of encountering 4 heads before 3 tails in repeated fair coin flips -/
noncomputable def q : ℚ := sorry

/-- Theorem stating that q is equal to 28/47 -/
theorem four_heads_before_three_tails : q = 28/47 := by sorry

end NUMINAMATH_CALUDE_four_heads_before_three_tails_l3504_350424


namespace NUMINAMATH_CALUDE_pasture_perimeter_difference_l3504_350477

/-- Calculates the perimeter of a pasture given the number of stakes and the interval between stakes -/
def pasture_perimeter (stakes : ℕ) (interval : ℕ) : ℕ := stakes * interval

/-- The difference between the perimeters of two pastures -/
theorem pasture_perimeter_difference : 
  pasture_perimeter 82 20 - pasture_perimeter 96 10 = 680 := by
  sorry

end NUMINAMATH_CALUDE_pasture_perimeter_difference_l3504_350477


namespace NUMINAMATH_CALUDE_intersection_line_equation_l3504_350442

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 13 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (P Q : ℝ × ℝ),
  circle1 P.1 P.2 ∧ circle1 Q.1 Q.2 ∧
  circle2 P.1 P.2 ∧ circle2 Q.1 Q.2 ∧
  P ≠ Q →
  line P.1 P.2 ∧ line Q.1 Q.2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l3504_350442


namespace NUMINAMATH_CALUDE_solve_equation_l3504_350431

theorem solve_equation : ∃ x : ℚ, (40 / 60 : ℚ) = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3504_350431


namespace NUMINAMATH_CALUDE_smallest_percent_increase_between_3_and_4_l3504_350405

def question_values : List ℕ := [100, 300, 600, 900, 1500, 2400]

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def consecutive_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  List.zip l (List.tail l)

theorem smallest_percent_increase_between_3_and_4 :
  let pairs := consecutive_pairs question_values
  let increases := pairs.map (fun (a, b) => percent_increase a b)
  increases.argmin id = some 2 := by sorry

end NUMINAMATH_CALUDE_smallest_percent_increase_between_3_and_4_l3504_350405


namespace NUMINAMATH_CALUDE_calculate_expression_l3504_350421

theorem calculate_expression : (-2)^2 - (1/8 - 3/4 + 1/2) * (-24) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3504_350421


namespace NUMINAMATH_CALUDE_cambridge_population_l3504_350468

theorem cambridge_population : ∃ (p : ℕ), p > 0 ∧ (
  ∀ (w a : ℚ),
  w > 0 ∧ a > 0 ∧
  w + a = 12 * p ∧
  w / 6 + a / 8 = 12 →
  p = 7
) := by
  sorry

end NUMINAMATH_CALUDE_cambridge_population_l3504_350468


namespace NUMINAMATH_CALUDE_f_properties_l3504_350465

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

theorem f_properties :
  ∃ (T : ℝ) (max_val : ℝ) (k : ℤ → ℝ),
    (∀ x, f (x + T) = f x) ∧ 
    (∀ y, y > 0 → (∀ x, f (x + y) = f x) → y ≥ T) ∧
    (T = 2 * Real.pi) ∧
    (∀ x, f x ≤ max_val) ∧
    (max_val = 2) ∧
    (∀ n, f (k n) = max_val) ∧
    (∀ n, k n = 2 * n * Real.pi + Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3504_350465


namespace NUMINAMATH_CALUDE_unique_arrangements_of_zeros_and_ones_l3504_350482

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def permutations (n : ℕ) : ℕ := factorial n

def combinations (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem unique_arrangements_of_zeros_and_ones : 
  let total_digits : ℕ := 8
  let zeros : ℕ := 4
  let ones : ℕ := 4
  permutations total_digits / (permutations zeros * permutations ones) = 70 := by
  sorry

end NUMINAMATH_CALUDE_unique_arrangements_of_zeros_and_ones_l3504_350482


namespace NUMINAMATH_CALUDE_max_area_rectangular_play_area_l3504_350427

/-- 
Given a rectangular area with perimeter P (excluding one side) and length l and width w,
prove that the maximum area A is achieved when l = P/2 and w = P/6, resulting in A = (P^2)/48.
-/
theorem max_area_rectangular_play_area (P : ℝ) (h : P > 0) :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l + 2*w = P ∧
  ∀ (l' w' : ℝ), l' > 0 → w' > 0 → l' + 2*w' = P →
  l * w ≥ l' * w' ∧
  l = P/2 ∧ w = P/6 ∧ l * w = (P^2)/48 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_play_area_l3504_350427


namespace NUMINAMATH_CALUDE_smallest_value_u_cube_plus_v_cube_l3504_350452

theorem smallest_value_u_cube_plus_v_cube (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 2)
  (h2 : Complex.abs (u^2 + v^2) = 17) :
  Complex.abs (u^3 + v^3) = 47 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_u_cube_plus_v_cube_l3504_350452


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3504_350467

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3504_350467


namespace NUMINAMATH_CALUDE_trail_mix_weight_l3504_350400

theorem trail_mix_weight (peanuts chocolate_chips raisins : ℝ) 
  (h1 : peanuts = 0.17)
  (h2 : chocolate_chips = 0.17)
  (h3 : raisins = 0.08) :
  peanuts + chocolate_chips + raisins = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l3504_350400


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3504_350488

def y : ℕ := 2^3^4^5^6^7^8^9

theorem smallest_multiplier_for_perfect_square (k : ℕ) : 
  k > 0 ∧ 
  ∃ m : ℕ, k * y = m^2 ∧ 
  ∀ l : ℕ, l > 0 ∧ l < k → ¬∃ n : ℕ, l * y = n^2 
  ↔ 
  k = 10 := by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3504_350488


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l3504_350425

def A : Set ℝ := {-3, -1, 1, 3}
def B : Set ℝ := {x : ℝ | x^2 + 2*x - 3 = 0}

theorem intersection_complement_equal : A ∩ (Set.univ \ B) = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l3504_350425


namespace NUMINAMATH_CALUDE_company_average_salary_associates_avg_salary_l3504_350417

theorem company_average_salary 
  (num_managers : ℕ) 
  (num_associates : ℕ) 
  (avg_salary_managers : ℚ) 
  (avg_salary_company : ℚ) : ℚ :=
  let total_employees := num_managers + num_associates
  let total_salary := avg_salary_company * total_employees
  let managers_salary := avg_salary_managers * num_managers
  let associates_salary := total_salary - managers_salary
  associates_salary / num_associates

theorem associates_avg_salary 
  (h1 : company_average_salary 15 75 90000 40000 = 30000) : 
  company_average_salary 15 75 90000 40000 = 30000 := by
  sorry

end NUMINAMATH_CALUDE_company_average_salary_associates_avg_salary_l3504_350417


namespace NUMINAMATH_CALUDE_min_value_of_f_min_value_is_zero_l3504_350459

-- Define the linear function
def f (x : ℝ) : ℝ := -x + 3

-- Define the domain
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem min_value_of_f :
  ∀ x : ℝ, domain x → ∀ y : ℝ, domain y → f y ≥ f 3 := by
  sorry

-- The minimum value is f(3) = 0
theorem min_value_is_zero : f 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_min_value_is_zero_l3504_350459


namespace NUMINAMATH_CALUDE_shortest_side_is_eight_l3504_350473

/-- Represents a rectangular solid with sides in geometric progression -/
structure GeometricSolid where
  b : ℝ
  s : ℝ
  volume : ℝ
  surface_area : ℝ
  volume_eq : volume = b^3 / s
  surface_area_eq : surface_area = 2 * (b^2 / s + b^2 * s + b^2)

/-- The shortest side length of a geometric solid with given properties is 8 -/
theorem shortest_side_is_eight (solid : GeometricSolid)
  (h_volume : solid.volume = 512)
  (h_surface_area : solid.surface_area = 384) :
  min (solid.b / solid.s) (min solid.b (solid.b * solid.s)) = 8 := by
  sorry

#check shortest_side_is_eight

end NUMINAMATH_CALUDE_shortest_side_is_eight_l3504_350473


namespace NUMINAMATH_CALUDE_solution_set_l3504_350429

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom increasing_f : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0
axiom odd_shifted : ∀ x : ℝ, f (x + 1) = -f (-(x + 1))

-- State the theorem
theorem solution_set (x : ℝ) : f (1 - x) > 0 ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l3504_350429


namespace NUMINAMATH_CALUDE_flat_transaction_l3504_350470

theorem flat_transaction (x y : ℝ) : 
  0.14 * x - 0.14 * y = 1.96 ↔ 
  ∃ (gain loss : ℝ), 
    gain = 0.14 * x ∧ 
    loss = 0.14 * y ∧ 
    gain - loss = 1.96 :=
sorry

end NUMINAMATH_CALUDE_flat_transaction_l3504_350470


namespace NUMINAMATH_CALUDE_add_decimals_l3504_350414

theorem add_decimals : 5.47 + 4.26 = 9.73 := by
  sorry

end NUMINAMATH_CALUDE_add_decimals_l3504_350414


namespace NUMINAMATH_CALUDE_negation_truth_values_l3504_350490

theorem negation_truth_values :
  (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ∧
  (¬ ∀ x y : ℝ, Real.sqrt ((x - 1)^2) + (y + 1)^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_truth_values_l3504_350490


namespace NUMINAMATH_CALUDE_paper_tearing_impossibility_l3504_350485

theorem paper_tearing_impossibility : ¬ ∃ (n : ℕ), 1 + 2 * n = 100 := by
  sorry

end NUMINAMATH_CALUDE_paper_tearing_impossibility_l3504_350485


namespace NUMINAMATH_CALUDE_factor_expression_l3504_350401

theorem factor_expression (x : ℝ) : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3504_350401


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3504_350492

theorem expand_and_simplify (x : ℝ) : 
  (1 + x^2 + x^4) * (1 - x^3 + x^5) = 1 - x^3 + x^2 + x^4 + x^9 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3504_350492


namespace NUMINAMATH_CALUDE_gcd_property_l3504_350410

theorem gcd_property (n : ℕ) :
  (∃ d : ℕ, d = Nat.gcd (7 * n + 5) (5 * n + 4) ∧ (d = 1 ∨ d = 3)) ∧
  (Nat.gcd (7 * n + 5) (5 * n + 4) = 3 ↔ ∃ k : ℕ, n = 3 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_gcd_property_l3504_350410


namespace NUMINAMATH_CALUDE_vector_equality_transitivity_l3504_350475

variable {V : Type*} [AddCommGroup V]

theorem vector_equality_transitivity (a b c : V) :
  a = b → b = c → a = c := by sorry

end NUMINAMATH_CALUDE_vector_equality_transitivity_l3504_350475


namespace NUMINAMATH_CALUDE_f_bounded_iff_alpha_in_unit_interval_l3504_350471

/-- The function f defined on pairs of nonnegative integers -/
noncomputable def f (α : ℝ) : ℕ → ℕ → ℝ
| 0, 0 => 1
| m, 0 => 0
| 0, n => 0
| (m+1), (n+1) => α * f α m (n+1) + (1 - α) * f α m n

/-- The theorem statement -/
theorem f_bounded_iff_alpha_in_unit_interval (α : ℝ) :
  (∀ m n : ℕ, |f α m n| < 1989) ↔ 0 < α ∧ α < 1 := by
  sorry

end NUMINAMATH_CALUDE_f_bounded_iff_alpha_in_unit_interval_l3504_350471


namespace NUMINAMATH_CALUDE_special_function_property_l3504_350458

/-- A function satisfying the given property for all real numbers -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ c d : ℝ, c^2 * g d = d^2 * g c

theorem special_function_property (g : ℝ → ℝ) 
  (h1 : SatisfiesProperty g) (h2 : g 4 ≠ 0) : 
  (g 7 - g 3) / g 4 = 2.5 := by sorry

end NUMINAMATH_CALUDE_special_function_property_l3504_350458


namespace NUMINAMATH_CALUDE_food_drive_cans_l3504_350440

theorem food_drive_cans (rachel jaydon mark : ℕ) : 
  jaydon = 2 * rachel + 5 →
  mark = 4 * jaydon →
  rachel + jaydon + mark = 135 →
  mark = 100 := by
sorry

end NUMINAMATH_CALUDE_food_drive_cans_l3504_350440


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l3504_350444

-- Define the parameters of the problem
def total_time : ℝ := 6
def up_time : ℝ := 4
def down_time : ℝ := 2
def avg_speed_total : ℝ := 2

-- Define the theorem
theorem hill_climbing_speed :
  let total_distance := avg_speed_total * total_time
  let distance_one_way := total_distance / 2
  let avg_speed_up := distance_one_way / up_time
  avg_speed_up = 1.5 := by sorry

end NUMINAMATH_CALUDE_hill_climbing_speed_l3504_350444


namespace NUMINAMATH_CALUDE_abcd_product_magnitude_l3504_350435

theorem abcd_product_magnitude (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_eq : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/d ∧ c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 := by
sorry

end NUMINAMATH_CALUDE_abcd_product_magnitude_l3504_350435


namespace NUMINAMATH_CALUDE_total_watermelon_slices_l3504_350434

-- Define the number of watermelons and slices for each person
def danny_watermelons : ℕ := 3
def danny_slices_per_melon : ℕ := 10

def sister_watermelons : ℕ := 1
def sister_slices_per_melon : ℕ := 15

def cousin_watermelons : ℕ := 2
def cousin_slices_per_melon : ℕ := 8

def aunt_watermelons : ℕ := 4
def aunt_slices_per_melon : ℕ := 12

def grandfather_watermelons : ℕ := 1
def grandfather_slices_per_melon : ℕ := 6

-- Define the total number of slices
def total_slices : ℕ := 
  danny_watermelons * danny_slices_per_melon +
  sister_watermelons * sister_slices_per_melon +
  cousin_watermelons * cousin_slices_per_melon +
  aunt_watermelons * aunt_slices_per_melon +
  grandfather_watermelons * grandfather_slices_per_melon

-- Theorem statement
theorem total_watermelon_slices : total_slices = 115 := by
  sorry

end NUMINAMATH_CALUDE_total_watermelon_slices_l3504_350434


namespace NUMINAMATH_CALUDE_custom_mul_unique_identity_l3504_350498

/-- Custom multiplication operation -/
def custom_mul (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c * x * y

theorem custom_mul_unique_identity
  (a b c : ℝ)
  (h1 : custom_mul a b c 1 2 = 3)
  (h2 : custom_mul a b c 2 3 = 4)
  (h3 : ∃ (m : ℝ), m ≠ 0 ∧ ∀ (x : ℝ), custom_mul a b c x m = x) :
  ∃ (m : ℝ), m = 4 ∧ m ≠ 0 ∧ ∀ (x : ℝ), custom_mul a b c x m = x :=
by sorry

end NUMINAMATH_CALUDE_custom_mul_unique_identity_l3504_350498


namespace NUMINAMATH_CALUDE_regina_farm_sale_price_l3504_350480

/-- Calculates the total sale price of animals on Regina's farm -/
def total_sale_price (num_cows : ℕ) (cow_price pig_price goat_price chicken_price rabbit_price : ℕ) : ℕ :=
  let num_pigs := 4 * num_cows
  let num_goats := num_pigs / 2
  let num_chickens := 2 * num_cows
  let num_rabbits := 30
  num_cows * cow_price + num_pigs * pig_price + num_goats * goat_price + 
  num_chickens * chicken_price + num_rabbits * rabbit_price

/-- Theorem stating that the total sale price of all animals on Regina's farm is $74,750 -/
theorem regina_farm_sale_price :
  total_sale_price 20 800 400 600 50 25 = 74750 := by
  sorry

end NUMINAMATH_CALUDE_regina_farm_sale_price_l3504_350480


namespace NUMINAMATH_CALUDE_sum_and_count_equals_1271_l3504_350439

/-- The sum of integers from 50 to 70, inclusive -/
def x : ℕ := (List.range 21).map (· + 50) |>.sum

/-- The number of even integers from 50 to 70, inclusive -/
def y : ℕ := (List.range 21).map (· + 50) |>.filter (· % 2 = 0) |>.length

/-- The theorem stating that x + y equals 1271 -/
theorem sum_and_count_equals_1271 : x + y = 1271 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_equals_1271_l3504_350439


namespace NUMINAMATH_CALUDE_inverse_proportion_point_relation_l3504_350491

theorem inverse_proportion_point_relation :
  ∀ (y₁ y₂ y₃ : ℝ),
  y₁ = 3 / (-5) →
  y₂ = 3 / (-3) →
  y₃ = 3 / 2 →
  y₂ < y₁ ∧ y₁ < y₃ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_relation_l3504_350491


namespace NUMINAMATH_CALUDE_existence_of_m_l3504_350403

theorem existence_of_m (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h_cd : c * d = 1) :
  ∃ m : ℕ, (0 < m) ∧ (a * b ≤ m^2) ∧ (m^2 ≤ (a + c) * (b + d)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l3504_350403


namespace NUMINAMATH_CALUDE_function_equation_solution_l3504_350445

theorem function_equation_solution (c : ℝ) (hc : c > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, x > 0 → f x > 0) :
  (∀ x y, x > 0 → y > 0 → f ((c + 1) * x + f y) = f (x + 2 * y) + 2 * c * x) →
  (∀ x, x > 0 → f x = 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3504_350445


namespace NUMINAMATH_CALUDE_square_division_perimeter_l3504_350430

/-- Given a square with perimeter 160 units, when divided into two congruent rectangles
    horizontally and one of those rectangles is further divided into two congruent rectangles
    vertically, the perimeter of one of the smaller rectangles is 80 units. -/
theorem square_division_perimeter :
  ∀ (s : ℝ),
  s > 0 →
  4 * s = 160 →
  let horizontal_rectangle_width := s
  let horizontal_rectangle_height := s / 2
  let vertical_rectangle_width := s / 2
  let vertical_rectangle_height := s / 2
  2 * (vertical_rectangle_width + vertical_rectangle_height) = 80 :=
by
  sorry

#check square_division_perimeter

end NUMINAMATH_CALUDE_square_division_perimeter_l3504_350430


namespace NUMINAMATH_CALUDE_jacob_jogging_distance_l3504_350464

/-- Calculates the total distance jogged given a constant speed and total jogging time -/
def total_distance_jogged (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that jogging at 4 miles per hour for 3 hours results in a total distance of 12 miles -/
theorem jacob_jogging_distance :
  let speed : ℝ := 4
  let time : ℝ := 3
  total_distance_jogged speed time = 12 := by
  sorry

end NUMINAMATH_CALUDE_jacob_jogging_distance_l3504_350464


namespace NUMINAMATH_CALUDE_sqrt_seven_inequality_l3504_350466

theorem sqrt_seven_inequality (m n : ℕ+) (h : (m : ℝ) / n < Real.sqrt 7) :
  7 - (m : ℝ)^2 / n^2 ≥ 3 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_inequality_l3504_350466


namespace NUMINAMATH_CALUDE_inequality_proof_l3504_350404

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 * (y*z + z*x + x*y)^2 ≤ 3*(y^2 + y*z + z^2)*(z^2 + z*x + x^2)*(x^2 + x*y + y^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3504_350404


namespace NUMINAMATH_CALUDE_ratio_q_p_l3504_350402

def total_cards : ℕ := 60
def num_range : ℕ := 12
def cards_per_number : ℕ := 5
def drawn_cards : ℕ := 5

def p : ℚ := num_range / (total_cards.choose drawn_cards)

def q : ℚ := (num_range * (num_range - 1) * cards_per_number.choose 4 * cards_per_number) / (total_cards.choose drawn_cards)

theorem ratio_q_p : q / p = 275 := by
  sorry

end NUMINAMATH_CALUDE_ratio_q_p_l3504_350402


namespace NUMINAMATH_CALUDE_prize_winning_beverage_probabilities_l3504_350494

/-- The probability of success for each independent event -/
def p : ℚ := 1 / 6

/-- The probability of failure for each independent event -/
def q : ℚ := 1 - p

theorem prize_winning_beverage_probabilities :
  let prob_all_fail := q ^ 3
  let prob_at_least_two_fail := 1 - (3 * p^2 * q + p^3)
  (prob_all_fail = 125 / 216) ∧ (prob_at_least_two_fail = 25 / 27) := by
  sorry

end NUMINAMATH_CALUDE_prize_winning_beverage_probabilities_l3504_350494


namespace NUMINAMATH_CALUDE_triangle_has_two_acute_angles_l3504_350493

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define the property that the sum of angles in a triangle is 180°
def validTriangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Define an acute angle
def isAcute (angle : Real) : Prop := angle < 90

-- Theorem statement
theorem triangle_has_two_acute_angles (t : Triangle) (h : validTriangle t) :
  ∃ (a b : Real), (a = t.angle1 ∨ a = t.angle2 ∨ a = t.angle3) ∧
                  (b = t.angle1 ∨ b = t.angle2 ∨ b = t.angle3) ∧
                  (a ≠ b) ∧
                  isAcute a ∧ isAcute b :=
sorry

end NUMINAMATH_CALUDE_triangle_has_two_acute_angles_l3504_350493


namespace NUMINAMATH_CALUDE_power_of_product_cube_l3504_350447

theorem power_of_product_cube (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_cube_l3504_350447


namespace NUMINAMATH_CALUDE_investment_average_rate_l3504_350472

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) (x : ℝ) :
  total = 6000 →
  rate1 = 0.03 →
  rate2 = 0.07 →
  rate1 * (total - x) = rate2 * x →
  (rate1 * (total - x) + rate2 * x) / total = 0.042 :=
by sorry

end NUMINAMATH_CALUDE_investment_average_rate_l3504_350472


namespace NUMINAMATH_CALUDE_chessboard_polygon_tasteful_tiling_l3504_350433

-- Define a chessboard polygon
def ChessboardPolygon : Type := sorry

-- Define a domino
def Domino : Type := sorry

-- Define a tiling
def Tiling (p : ChessboardPolygon) : Type := sorry

-- Define a tasteful tiling
def TastefulTiling (p : ChessboardPolygon) : Type := sorry

-- Define the property of being tileable by dominoes
def IsTileable (p : ChessboardPolygon) : Prop := sorry

-- Theorem statement
theorem chessboard_polygon_tasteful_tiling 
  (p : ChessboardPolygon) (h : IsTileable p) :
  (∃ t : TastefulTiling p, True) ∧ 
  (∀ t1 t2 : TastefulTiling p, t1 = t2) :=
sorry

end NUMINAMATH_CALUDE_chessboard_polygon_tasteful_tiling_l3504_350433


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l3504_350460

theorem min_distance_to_origin (x y : ℝ) : 
  8 * x + 15 * y = 120 → x ≥ 0 → y ≥ 0 → 
  ∀ x' y' : ℝ, 8 * x' + 15 * y' = 120 → x' ≥ 0 → y' ≥ 0 → 
  Real.sqrt (x^2 + y^2) ≤ Real.sqrt (x'^2 + y'^2) → 
  Real.sqrt (x^2 + y^2) = 120 / 17 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l3504_350460


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l3504_350476

/-- Represents the seating arrangement problem --/
structure SeatingArrangement where
  front_seats : Nat
  back_seats : Nat
  people : Nat
  blocked_front : Nat

/-- Calculates the number of valid seating arrangements --/
def count_arrangements (s : SeatingArrangement) : Nat :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem --/
theorem seating_arrangement_count :
  let s : SeatingArrangement := {
    front_seats := 11,
    back_seats := 12,
    people := 2,
    blocked_front := 3
  }
  count_arrangements s = 346 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l3504_350476


namespace NUMINAMATH_CALUDE_snail_race_l3504_350415

/-- The race problem with three snails -/
theorem snail_race (speed_1 : ℝ) (time_3 : ℝ) : 
  speed_1 = 2 →  -- First snail's speed
  time_3 = 2 →   -- Time taken by the third snail
  (∃ (speed_2 speed_3 distance time_1 : ℝ), 
    speed_2 = 2 * speed_1 ∧             -- Second snail's speed
    speed_3 = 5 * speed_2 ∧             -- Third snail's speed
    distance = speed_3 * time_3 ∧       -- Total distance
    time_1 * speed_1 = distance ∧       -- First snail's time
    time_1 = 20) :=                     -- First snail took 20 minutes
by sorry

end NUMINAMATH_CALUDE_snail_race_l3504_350415


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3504_350416

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4}
  A ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3504_350416


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3504_350453

/-- Represents a rhombus with given diagonal and area -/
structure Rhombus where
  diagonal : ℝ
  area : ℝ

/-- Calculates the length of the side of a rhombus -/
def side_length (r : Rhombus) : ℝ :=
  sorry

theorem rhombus_side_length (r : Rhombus) (h1 : r.diagonal = 30) (h2 : r.area = 600) :
  side_length r = 25 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3504_350453


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3504_350454

theorem no_positive_integer_solutions (k n : ℕ+) (h : n > 2) :
  ¬∃ (x y : ℕ+), x^(n : ℕ) - y^(n : ℕ) = 2^(k : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3504_350454


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l3504_350436

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 5 (1/3) + 1 →
  x^3 - 3*x^2 + 3*x - 6 = 0 ∧ 
  (∃ (a b c : ℤ), x^3 - 3*x^2 + 3*x - 6 = x^3 + a*x^2 + b*x + c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l3504_350436


namespace NUMINAMATH_CALUDE_average_marks_of_passed_candidates_l3504_350495

theorem average_marks_of_passed_candidates 
  (total_candidates : ℕ) 
  (overall_average : ℚ) 
  (failed_average : ℚ) 
  (passed_count : ℕ) 
  (h1 : total_candidates = 120)
  (h2 : overall_average = 35)
  (h3 : failed_average = 15)
  (h4 : passed_count = 100) :
  (total_candidates * overall_average - (total_candidates - passed_count) * failed_average) / passed_count = 39 := by
sorry

end NUMINAMATH_CALUDE_average_marks_of_passed_candidates_l3504_350495


namespace NUMINAMATH_CALUDE_smallest_norwegian_number_l3504_350428

/-- A number is Norwegian if it has three distinct positive divisors whose sum is equal to 2022. -/
def IsNorwegian (n : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₃ > 0 ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃ ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧
    d₁ + d₂ + d₃ = 2022

/-- 1344 is the smallest Norwegian number. -/
theorem smallest_norwegian_number : 
  IsNorwegian 1344 ∧ ∀ m : ℕ, m < 1344 → ¬IsNorwegian m :=
by sorry

end NUMINAMATH_CALUDE_smallest_norwegian_number_l3504_350428


namespace NUMINAMATH_CALUDE_medium_supermarkets_sample_l3504_350407

/-- Represents the number of supermarkets to be sampled -/
def sample_size : ℕ := 200

/-- Represents the number of large supermarkets -/
def large_supermarkets : ℕ := 200

/-- Represents the number of medium supermarkets -/
def medium_supermarkets : ℕ := 400

/-- Represents the number of small supermarkets -/
def small_supermarkets : ℕ := 1400

/-- Represents the total number of supermarkets -/
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets

/-- Theorem stating that the number of medium supermarkets to be sampled is 40 -/
theorem medium_supermarkets_sample :
  (sample_size : ℚ) * medium_supermarkets / total_supermarkets = 40 := by
  sorry

end NUMINAMATH_CALUDE_medium_supermarkets_sample_l3504_350407


namespace NUMINAMATH_CALUDE_complex_numbers_on_circle_l3504_350489

/-- Given non-zero complex numbers a₁, a₂, a₃, a₄, a₅ satisfying certain conditions,
    prove that they lie on the same circle in the complex plane. -/
theorem complex_numbers_on_circle (a₁ a₂ a₃ a₄ a₅ : ℂ) (S : ℝ) 
    (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0)
    (h_ratio : a₂ / a₁ = a₃ / a₂ ∧ a₃ / a₂ = a₄ / a₃ ∧ a₄ / a₃ = a₅ / a₄)
    (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅))
    (h_sum_real : a₁ + a₂ + a₃ + a₄ + a₅ = S)
    (h_S_bound : abs S ≤ 2) :
  ∃ r : ℝ, r > 0 ∧ Complex.abs a₁ = r ∧ Complex.abs a₂ = r ∧ 
    Complex.abs a₃ = r ∧ Complex.abs a₄ = r ∧ Complex.abs a₅ = r :=
by sorry

end NUMINAMATH_CALUDE_complex_numbers_on_circle_l3504_350489


namespace NUMINAMATH_CALUDE_cookie_cost_is_18_l3504_350432

/-- The cost of each cookie Cora buys in April -/
def cookie_cost (cookies_per_day : ℕ) (days_in_april : ℕ) (total_spent : ℕ) : ℚ :=
  total_spent / (cookies_per_day * days_in_april)

/-- Theorem stating that each cookie costs 18 dollars -/
theorem cookie_cost_is_18 :
  cookie_cost 3 30 1620 = 18 := by sorry

end NUMINAMATH_CALUDE_cookie_cost_is_18_l3504_350432


namespace NUMINAMATH_CALUDE_inequality_proof_l3504_350487

theorem inequality_proof (a b c : ℝ) (ha : a = (Real.log 2) / 2) 
  (hb : b = (Real.log Real.pi) / Real.pi) (hc : c = (Real.log 5) / 5) : 
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3504_350487


namespace NUMINAMATH_CALUDE_three_special_lines_l3504_350486

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has intercepts on both axes with equal absolute values -/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ (l.a * t + l.c = 0 ∧ l.b * t + l.c = 0)

/-- The set of lines passing through (1, 2) with equal intercepts -/
def specialLines : Set Line :=
  {l : Line | l.contains 1 2 ∧ l.hasEqualIntercepts}

theorem three_special_lines :
  ∃ (l₁ l₂ l₃ : Line),
    l₁ ∈ specialLines ∧
    l₂ ∈ specialLines ∧
    l₃ ∈ specialLines ∧
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ ∧
    ∀ l : Line, l ∈ specialLines → l = l₁ ∨ l = l₂ ∨ l = l₃ :=
  sorry

end NUMINAMATH_CALUDE_three_special_lines_l3504_350486


namespace NUMINAMATH_CALUDE_polynomial_relation_l3504_350451

theorem polynomial_relation (r : ℝ) : r^3 - 2*r + 1 = 0 → r^6 - 4*r^4 + 4*r^2 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_relation_l3504_350451


namespace NUMINAMATH_CALUDE_notebook_sales_plan_exists_l3504_350484

/-- Represents the sales data for a month -/
structure MonthSales where
  price : ℝ
  sales : ℝ

/-- Represents the notebook sales problem -/
structure NotebookSales where
  initial_inventory : ℕ
  purchase_price : ℝ
  min_sell_price : ℝ
  max_sell_price : ℝ
  july_oct_sales : List MonthSales
  price_sales_relation : ℝ → ℝ

/-- Represents a pricing plan for November and December -/
structure PricingPlan where
  nov_price : ℝ
  nov_sales : ℝ
  dec_price : ℝ
  dec_sales : ℝ

/-- Main theorem statement -/
theorem notebook_sales_plan_exists (problem : NotebookSales) :
  problem.initial_inventory = 550 ∧
  problem.purchase_price = 6 ∧
  problem.min_sell_price = 9 ∧
  problem.max_sell_price = 12 ∧
  problem.july_oct_sales = [⟨9, 115⟩, ⟨10, 100⟩, ⟨11, 85⟩, ⟨12, 70⟩] ∧
  (∀ x, problem.price_sales_relation x = -15 * x + 250) →
  ∃ (plan : PricingPlan),
    -- Remaining inventory after 4 months is 180
    (problem.initial_inventory - (problem.july_oct_sales.map (λ s => s.sales)).sum = 180) ∧
    -- Highest monthly profit in first 4 months is 425, occurring in September
    ((problem.july_oct_sales.map (λ s => (s.price - problem.purchase_price) * s.sales)).maximum = some 425) ∧
    -- Total sales profit for November and December is at least 800
    ((plan.nov_price - problem.purchase_price) * plan.nov_sales +
     (plan.dec_price - problem.purchase_price) * plan.dec_sales ≥ 800) ∧
    -- Pricing plan follows the price-sales relationship
    (problem.price_sales_relation plan.nov_price = plan.nov_sales ∧
     problem.price_sales_relation plan.dec_price = plan.dec_sales) ∧
    -- Prices are within the allowed range
    (plan.nov_price ≥ problem.min_sell_price ∧ plan.nov_price ≤ problem.max_sell_price ∧
     plan.dec_price ≥ problem.min_sell_price ∧ plan.dec_price ≤ problem.max_sell_price) :=
by sorry


end NUMINAMATH_CALUDE_notebook_sales_plan_exists_l3504_350484


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3504_350446

/-- The time it takes to fill a cistern without a leak, given that:
    1. With a leak, it takes T + 2 hours to fill
    2. When full, it takes 24 hours to empty due to the leak -/
theorem cistern_filling_time (T : ℝ) : 
  (∀ (t : ℝ), t > 0 → (1 / T - 1 / (T + 2) = 1 / 24)) → 
  T = 6 := by
sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l3504_350446


namespace NUMINAMATH_CALUDE_number_of_men_l3504_350479

theorem number_of_men (M : ℕ) (W : ℝ) : 
  (W / (M * 40) = W / ((M - 5) * 50)) → M = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_l3504_350479


namespace NUMINAMATH_CALUDE_basketball_probability_l3504_350412

theorem basketball_probability (jack_prob jill_prob sandy_prob : ℚ) 
  (h1 : jack_prob = 1/6)
  (h2 : jill_prob = 1/7)
  (h3 : sandy_prob = 1/8) :
  (1 - jack_prob) * jill_prob * sandy_prob = 5/336 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l3504_350412


namespace NUMINAMATH_CALUDE_min_exponent_sum_l3504_350499

/-- Given a positive integer A with prime factorization A = 2^α * 3^β * 5^γ,
    where α, β, γ are natural numbers, if one-half of A is a perfect square,
    one-third of A is a perfect cube, and one-fifth of A is a perfect fifth power,
    then the minimum value of α + β + γ is 31. -/
theorem min_exponent_sum (A : ℕ+) (α β γ : ℕ)
  (h_factorization : A = 2^α * 3^β * 5^γ)
  (h_half_square : ∃ (k : ℕ), A / 2 = k^2)
  (h_third_cube : ∃ (m : ℕ), A / 3 = m^3)
  (h_fifth_power : ∃ (n : ℕ), A / 5 = n^5) :
  α + β + γ ≥ 31 :=
sorry

end NUMINAMATH_CALUDE_min_exponent_sum_l3504_350499


namespace NUMINAMATH_CALUDE_piano_lesson_cost_l3504_350456

theorem piano_lesson_cost (lesson_duration : Real) (total_hours : Real) (total_cost : Real) :
  lesson_duration = 1.5 →
  total_hours = 18 →
  total_cost = 360 →
  lesson_duration * (total_cost / total_hours) = 30 := by
sorry

end NUMINAMATH_CALUDE_piano_lesson_cost_l3504_350456


namespace NUMINAMATH_CALUDE_large_lemonade_price_l3504_350449

/-- Represents the price and sales data for Tonya's lemonade stand --/
structure LemonadeStand where
  small_price : ℝ
  medium_price : ℝ
  large_price : ℝ
  total_sales : ℝ
  small_sales : ℝ
  medium_sales : ℝ
  large_cups_sold : ℕ

/-- Theorem stating that the price of a large cup of lemonade is $3 --/
theorem large_lemonade_price (stand : LemonadeStand)
  (h1 : stand.small_price = 1)
  (h2 : stand.medium_price = 2)
  (h3 : stand.total_sales = 50)
  (h4 : stand.small_sales = 11)
  (h5 : stand.medium_sales = 24)
  (h6 : stand.large_cups_sold = 5)
  (h7 : stand.total_sales = stand.small_sales + stand.medium_sales + stand.large_price * stand.large_cups_sold) :
  stand.large_price = 3 := by
  sorry


end NUMINAMATH_CALUDE_large_lemonade_price_l3504_350449


namespace NUMINAMATH_CALUDE_population_percentage_l3504_350497

theorem population_percentage (W M : ℝ) (h : M = 1.1111111111111111 * W) : 
  W = 0.9 * M := by
sorry

end NUMINAMATH_CALUDE_population_percentage_l3504_350497


namespace NUMINAMATH_CALUDE_third_trial_point_l3504_350438

/-- The 0.618 method for optimization --/
def golden_ratio : ℝ := 0.618

/-- The lower bound of the initial range --/
def lower_bound : ℝ := 100

/-- The upper bound of the initial range --/
def upper_bound : ℝ := 1100

/-- Calculate the first trial point --/
def x₁ : ℝ := lower_bound + golden_ratio * (upper_bound - lower_bound)

/-- Calculate the second trial point --/
def x₂ : ℝ := lower_bound + (upper_bound - x₁)

/-- Calculate the third trial point --/
def x₃ : ℝ := lower_bound + golden_ratio * (x₂ - lower_bound)

/-- The theorem to be proved --/
theorem third_trial_point : ⌊x₃⌋ = 336 := by sorry

end NUMINAMATH_CALUDE_third_trial_point_l3504_350438


namespace NUMINAMATH_CALUDE_mincheol_midterm_average_l3504_350463

/-- Calculates the average of three exam scores -/
def midterm_average (math_score korean_score english_score : ℕ) : ℚ :=
  (math_score + korean_score + english_score : ℚ) / 3

/-- Theorem: Mincheol's midterm average is 80 points -/
theorem mincheol_midterm_average : 
  midterm_average 70 80 90 = 80 := by
  sorry

end NUMINAMATH_CALUDE_mincheol_midterm_average_l3504_350463


namespace NUMINAMATH_CALUDE_volunteer_arrangement_count_l3504_350483

/-- The number of volunteers --/
def n : ℕ := 6

/-- The number of exhibition areas --/
def m : ℕ := 4

/-- The number of areas that require one person --/
def single_person_areas : ℕ := 2

/-- The number of areas that require two people --/
def double_person_areas : ℕ := 2

/-- The number of specific volunteers that cannot be together --/
def restricted_volunteers : ℕ := 2

/-- The total number of arrangements without restrictions --/
def total_arrangements : ℕ := 180

/-- The number of arrangements where the restricted volunteers are together --/
def restricted_arrangements : ℕ := 24

theorem volunteer_arrangement_count :
  (n = 6) →
  (m = 4) →
  (single_person_areas = 2) →
  (double_person_areas = 2) →
  (restricted_volunteers = 2) →
  (total_arrangements = 180) →
  (restricted_arrangements = 24) →
  (total_arrangements - restricted_arrangements = 156) := by
  sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_count_l3504_350483


namespace NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_five_l3504_350423

theorem smallest_addition_for_multiple_of_five : 
  ∀ n : ℕ, n > 0 ∧ (725 + n) % 5 = 0 → n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_five_l3504_350423


namespace NUMINAMATH_CALUDE_box_production_equations_l3504_350461

/-- Represents the number of iron sheets available -/
def total_sheets : ℕ := 40

/-- Represents the number of box bodies that can be made from one sheet -/
def bodies_per_sheet : ℕ := 15

/-- Represents the number of box bottoms that can be made from one sheet -/
def bottoms_per_sheet : ℕ := 20

/-- Represents the ratio of box bottoms to box bodies in a complete set -/
def bottoms_to_bodies_ratio : ℕ := 2

/-- Theorem stating that the given system of equations correctly represents the problem -/
theorem box_production_equations (x y : ℕ) : 
  (x + y = total_sheets ∧ 
   2 * bodies_per_sheet * x = bottoms_per_sheet * y) ↔ 
  (x + y = total_sheets ∧ 
   bottoms_to_bodies_ratio * (bodies_per_sheet * x) = bottoms_per_sheet * y) :=
sorry

end NUMINAMATH_CALUDE_box_production_equations_l3504_350461


namespace NUMINAMATH_CALUDE_stock_sale_percentage_l3504_350426

/-- Proves that the percentage of stock sold is 100% given the provided conditions -/
theorem stock_sale_percentage
  (cash_realized : ℝ)
  (brokerage_rate : ℝ)
  (cash_after_brokerage : ℝ)
  (h1 : cash_realized = 109.25)
  (h2 : brokerage_rate = 1 / 400)
  (h3 : cash_after_brokerage = 109)
  (h4 : cash_after_brokerage = cash_realized * (1 - brokerage_rate)) :
  cash_realized / (cash_after_brokerage / (1 - brokerage_rate)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_stock_sale_percentage_l3504_350426


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l3504_350437

def A (m : ℝ) : Set ℝ := {-1, 2, 2*m-1}
def B (m : ℝ) : Set ℝ := {2, m^2}

theorem subset_implies_m_equals_one (m : ℝ) :
  B m ⊆ A m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l3504_350437


namespace NUMINAMATH_CALUDE_triangle_sequence_2009_position_l3504_350443

def triangle_sequence (n : ℕ) : ℕ := n

def row_of_term (n : ℕ) : ℕ :=
  (n.sqrt : ℕ) + 1

def position_in_row (n : ℕ) : ℕ :=
  n - (row_of_term n - 1)^2

theorem triangle_sequence_2009_position :
  row_of_term 2009 = 45 ∧ position_in_row 2009 = 73 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sequence_2009_position_l3504_350443


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3504_350450

/-- An ellipse with semi-major axis a, semi-minor axis b, and focal distance c. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_relation : a^2 - b^2 = c^2

/-- The given ellipse with a = 5 and left focus at (-4, 0). -/
def given_ellipse (m : ℝ) : Ellipse :=
  { a := 5
    b := m
    c := 4
    h_positive := by sorry
    h_relation := by sorry }

/-- Theorem stating that m = 3 for the given ellipse. -/
theorem ellipse_m_value :
  ∀ m > 0, (given_ellipse m).b = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3504_350450


namespace NUMINAMATH_CALUDE_smallest_a_value_l3504_350413

-- Define the arithmetic sequence
def is_arithmetic_sequence (a b c : ℕ) : Prop := b - a = c - b

-- Define the function f
def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem smallest_a_value (a b c : ℕ) (r s : ℝ) :
  is_arithmetic_sequence a b c →
  a < b →
  b < c →
  f a b c r = s →
  f a b c s = r →
  r * s = 2017 →
  ∃ (min_a : ℕ), min_a = 1 ∧ ∀ (a' : ℕ), (∃ (b' c' : ℕ) (r' s' : ℝ),
    is_arithmetic_sequence a' b' c' ∧
    a' < b' ∧
    b' < c' ∧
    f a' b' c' r' = s' ∧
    f a' b' c' s' = r' ∧
    r' * s' = 2017) → a' ≥ min_a :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3504_350413


namespace NUMINAMATH_CALUDE_tangent_points_on_line_l3504_350455

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define the line l
def l (x y : ℝ) : Prop := x = -2 ∨ 3*x - 4*y + 18 = 0

-- Define point A
def A : ℝ × ℝ := (-2, 3)

-- Define the chord length condition
def chord_length (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12 -- (2√3)^2 = 12

-- Define the tangent length equality condition
def equal_tangents (x y : ℝ) : Prop :=
  (x + 3)^2 + (y - 1)^2 - 4 = (x - 3)^2 + (y - 4)^2 - 1

-- Main theorem
theorem tangent_points_on_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    l x₁ y₁ ∧ l x₂ y₂ ∧
    equal_tangents x₁ y₁ ∧ equal_tangents x₂ y₂ ∧
    ((x₁ = -2 ∧ y₁ = 7) ∨ (x₁ = -6/11 ∧ y₁ = 45/11)) ∧
    ((x₂ = -2 ∧ y₂ = 7) ∨ (x₂ = -6/11 ∧ y₂ = 45/11)) ∧
    x₁ ≠ x₂ := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_on_line_l3504_350455


namespace NUMINAMATH_CALUDE_banana_arrangements_l3504_350408

def word_length : ℕ := 7

def identical_b_count : ℕ := 2

def distinct_letter_count : ℕ := 5

theorem banana_arrangements :
  (word_length.factorial / identical_b_count.factorial) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3504_350408


namespace NUMINAMATH_CALUDE_count_even_numbers_between_150_and_350_l3504_350481

theorem count_even_numbers_between_150_and_350 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ 150 < n ∧ n < 350) (Finset.range 350)).card = 99 := by
  sorry

end NUMINAMATH_CALUDE_count_even_numbers_between_150_and_350_l3504_350481


namespace NUMINAMATH_CALUDE_tangent_lines_to_cubic_l3504_350419

noncomputable def f (x : ℝ) := x^3

def P : ℝ × ℝ := (1, 1)

theorem tangent_lines_to_cubic (x : ℝ) :
  -- The tangent line at point P(1, 1) is y = 3x - 2
  (HasDerivAt f 3 1 ∧ f 1 = 1) →
  -- There are exactly two tangent lines to the curve that pass through point P(1, 1)
  (∃! (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧
    -- First tangent line: y = 3x - 2
    (m₁ = 3 ∧ P.2 = m₁ * P.1 - 2) ∧
    -- Second tangent line: y = 3/4x + 1/4
    (m₂ = 3/4 ∧ P.2 = m₂ * P.1 + 1/4) ∧
    -- Both lines are tangent to the curve
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
      HasDerivAt f (3 * x₁^2) x₁ ∧
      HasDerivAt f (3 * x₂^2) x₂ ∧
      f x₁ = m₁ * x₁ - 2 ∧
      f x₂ = m₂ * x₂ + 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_cubic_l3504_350419


namespace NUMINAMATH_CALUDE_y_derivative_l3504_350457

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := Real.log (1 / Real.sqrt (1 + x^2))

-- State the theorem
theorem y_derivative (x : ℝ) : 
  deriv y x = -x / (1 + x^2) := by sorry

end NUMINAMATH_CALUDE_y_derivative_l3504_350457


namespace NUMINAMATH_CALUDE_election_win_margin_l3504_350462

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) (winner_percentage : ℚ) 
  (h1 : winner_percentage = 62 / 100)
  (h2 : winner_votes = 806)
  (h3 : winner_votes = (winner_percentage * total_votes).floor) :
  winner_votes - (total_votes - winner_votes) = 312 := by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l3504_350462


namespace NUMINAMATH_CALUDE_profit_difference_l3504_350420

/-- Calculates the difference in profit share between two partners given their investments and total profit -/
theorem profit_difference (mary_investment mike_investment total_profit : ℚ) :
  mary_investment = 650 →
  mike_investment = 350 →
  total_profit = 2999.9999999999995 →
  let total_investment := mary_investment + mike_investment
  let equal_share := (1/3) * total_profit / 2
  let remaining_profit := (2/3) * total_profit
  let mary_ratio := mary_investment / total_investment
  let mike_ratio := mike_investment / total_investment
  let mary_total := equal_share + mary_ratio * remaining_profit
  let mike_total := equal_share + mike_ratio * remaining_profit
  mary_total - mike_total = 600 := by sorry

end NUMINAMATH_CALUDE_profit_difference_l3504_350420


namespace NUMINAMATH_CALUDE_cube_root_unity_product_l3504_350406

theorem cube_root_unity_product (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (1 - ω + ω^2) * (1 + ω - ω^2) = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_unity_product_l3504_350406
