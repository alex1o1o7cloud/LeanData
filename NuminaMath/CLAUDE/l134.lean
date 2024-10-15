import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_dimension_increase_l134_13470

theorem rectangle_dimension_increase (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let L' := 1.3 * L
  let A := L * B
  let A' := 1.885 * A
  ∃ p : ℝ, p > 0 ∧ L' * (B * (1 + p / 100)) = A' ∧ p = 45 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_increase_l134_13470


namespace NUMINAMATH_CALUDE_range_of_m_l134_13485

theorem range_of_m (m : ℝ) : 
  (|m + 3| = m + 3) →
  (|3*m + 9| ≥ 4*m - 3 ↔ -3 ≤ m ∧ m ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l134_13485


namespace NUMINAMATH_CALUDE_system_solution_l134_13456

theorem system_solution (a b c x y z : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (1 / x + 1 / y + 1 / z = 1) ∧ (a * x = b * y) ∧ (b * y = c * z) →
  (x = (a + b + c) / a) ∧ (y = (a + b + c) / b) ∧ (z = (a + b + c) / c) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l134_13456


namespace NUMINAMATH_CALUDE_missing_files_l134_13468

theorem missing_files (total : ℕ) (morning : ℕ) (afternoon : ℕ) : 
  total = 60 → 
  morning = total / 2 → 
  afternoon = 15 → 
  total - (morning + afternoon) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_files_l134_13468


namespace NUMINAMATH_CALUDE_cube_face_sum_l134_13461

theorem cube_face_sum (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ+) : 
  (a₁ * a₂ * a₅ + a₂ * a₃ * a₅ + a₃ * a₄ * a₅ + a₄ * a₁ * a₅ +
   a₁ * a₂ * a₆ + a₂ * a₃ * a₆ + a₃ * a₄ * a₆ + a₄ * a₁ * a₆ = 70) →
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ : ℕ) = 14 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l134_13461


namespace NUMINAMATH_CALUDE_largest_non_sum_30_and_composite_l134_13464

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
def isSum30AndComposite (n : ℕ) : Prop :=
  ∃ k c, 0 < k ∧ isComposite c ∧ n = 30 * k + c

/-- Theorem stating that 210 is the largest positive integer that cannot be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
theorem largest_non_sum_30_and_composite :
  (∀ n : ℕ, 210 < n → isSum30AndComposite n) ∧
  ¬isSum30AndComposite 210 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_30_and_composite_l134_13464


namespace NUMINAMATH_CALUDE_johns_allowance_l134_13407

def weekly_allowance : ℝ → Prop :=
  λ A => 
    let arcade_spent := (3/5) * A
    let remaining_after_arcade := A - arcade_spent
    let toy_store_spent := (1/3) * remaining_after_arcade
    let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
    remaining_after_toy_store = 0.60

theorem johns_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 2.25 := by sorry

end NUMINAMATH_CALUDE_johns_allowance_l134_13407


namespace NUMINAMATH_CALUDE_first_jump_over_2km_l134_13438

def jump_sequence (n : ℕ) : ℕ :=
  2 * 3^(n - 1)

theorem first_jump_over_2km :
  (∀ k < 8, jump_sequence k ≤ 2000) ∧ jump_sequence 8 > 2000 :=
sorry

end NUMINAMATH_CALUDE_first_jump_over_2km_l134_13438


namespace NUMINAMATH_CALUDE_solve_equation_l134_13454

theorem solve_equation : ∃ x : ℝ, (x - 5)^4 = (1/16)⁻¹ ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l134_13454


namespace NUMINAMATH_CALUDE_cricketer_average_after_22nd_inning_l134_13487

/-- Represents a cricketer's performance --/
structure CricketerPerformance where
  innings : ℕ
  scoreLastInning : ℕ
  averageIncrease : ℚ

/-- Calculates the average score after the last inning --/
def averageAfterLastInning (c : CricketerPerformance) : ℚ :=
  let previousAverage := (c.innings - 1 : ℚ) * (c.averageIncrease + (c.scoreLastInning : ℚ) / c.innings)
  (previousAverage + c.scoreLastInning) / c.innings

/-- Theorem stating the cricketer's average after the 22nd inning --/
theorem cricketer_average_after_22nd_inning 
  (c : CricketerPerformance)
  (h1 : c.innings = 22)
  (h2 : c.scoreLastInning = 134)
  (h3 : c.averageIncrease = 7/2) :
  averageAfterLastInning c = 121/2 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_after_22nd_inning_l134_13487


namespace NUMINAMATH_CALUDE_product_difference_equals_one_l134_13460

theorem product_difference_equals_one : (527 : ℤ) * 527 - 526 * 528 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equals_one_l134_13460


namespace NUMINAMATH_CALUDE_min_sum_of_distances_min_sum_of_distances_achievable_l134_13479

theorem min_sum_of_distances (x y z : ℝ) :
  Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2) ≥ Real.sqrt 6 :=
by sorry

theorem min_sum_of_distances_achievable :
  ∃ (x y z : ℝ), Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2) = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_min_sum_of_distances_achievable_l134_13479


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l134_13408

/-- An isosceles triangle with specific side lengths -/
structure IsoscelesTriangle where
  /-- Length of equal sides PQ and PR -/
  side : ℝ
  /-- Length of base QR -/
  base : ℝ
  /-- side is positive -/
  side_pos : side > 0
  /-- base is positive -/
  base_pos : base > 0

/-- The area of an isosceles triangle with side length 13 and base length 10 -/
theorem isosceles_triangle_area (t : IsoscelesTriangle)
  (h_side : t.side = 13)
  (h_base : t.base = 10) :
  let height := Real.sqrt (t.side ^ 2 - (t.base / 2) ^ 2)
  (1 / 2 : ℝ) * t.base * height = 60 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l134_13408


namespace NUMINAMATH_CALUDE_no_linear_term_implies_equal_coefficients_l134_13481

/-- Given a polynomial (x+p)(x-q) with no linear term in x, prove that p = q -/
theorem no_linear_term_implies_equal_coefficients (p q : ℝ) :
  (∀ x : ℝ, ∃ a b : ℝ, (x + p) * (x - q) = a * x^2 + b) → p = q := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_equal_coefficients_l134_13481


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l134_13426

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, (5 * x^2 + 7 = 2 * x - 6) ↔ (x = p + q * I ∨ x = p - q * I)) →
  p + q^2 = 69/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l134_13426


namespace NUMINAMATH_CALUDE_expand_and_simplify_l134_13435

theorem expand_and_simplify (x y : ℝ) : (x - 2*y)^2 - 2*y*(y - 2*x) = x^2 + 2*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l134_13435


namespace NUMINAMATH_CALUDE_lilies_count_l134_13480

/-- The cost of a single chrysanthemum in yuan -/
def chrysanthemum_cost : ℕ := 3

/-- The cost of a single lily in yuan -/
def lily_cost : ℕ := 4

/-- The total amount of money Mom wants to spend in yuan -/
def total_money : ℕ := 100

/-- The number of chrysanthemums Mom wants to buy -/
def chrysanthemums_to_buy : ℕ := 16

/-- The number of lilies that can be bought with the remaining money -/
def lilies_to_buy : ℕ := (total_money - chrysanthemum_cost * chrysanthemums_to_buy) / lily_cost

theorem lilies_count : lilies_to_buy = 13 := by
  sorry

end NUMINAMATH_CALUDE_lilies_count_l134_13480


namespace NUMINAMATH_CALUDE_stick_length_ratio_l134_13471

/-- Proves that the ratio of the second stick to the first stick is 2:1 given the conditions of the problem -/
theorem stick_length_ratio (stick2 : ℝ) 
  (h1 : 3 + stick2 + (stick2 - 1) = 14) : 
  stick2 / 3 = 2 := by sorry

end NUMINAMATH_CALUDE_stick_length_ratio_l134_13471


namespace NUMINAMATH_CALUDE_division_multiplication_order_matters_l134_13445

theorem division_multiplication_order_matters : (32 / 0.25) * 4 ≠ 32 / (0.25 * 4) := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_order_matters_l134_13445


namespace NUMINAMATH_CALUDE_even_function_negative_domain_l134_13431

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_negative_domain
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_positive : ∀ x > 0, f x = x) :
  ∀ x < 0, f x = -x :=
by sorry

end NUMINAMATH_CALUDE_even_function_negative_domain_l134_13431


namespace NUMINAMATH_CALUDE_smallest_multiple_of_nine_l134_13429

theorem smallest_multiple_of_nine (x : ℕ) : x = 18 ↔ 
  (∃ k : ℕ, x = 9 * k) ∧ 
  (x^2 > 200) ∧ 
  (x < Real.sqrt (x^2 - 144) * 5) ∧
  (∀ y : ℕ, y < x → (∃ k : ℕ, y = 9 * k) → 
    (y^2 ≤ 200 ∨ y ≥ Real.sqrt (y^2 - 144) * 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_nine_l134_13429


namespace NUMINAMATH_CALUDE_johns_outfit_cost_l134_13409

theorem johns_outfit_cost (pants_cost : ℝ) (h1 : pants_cost + 1.6 * pants_cost = 130) : pants_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_johns_outfit_cost_l134_13409


namespace NUMINAMATH_CALUDE_tangent_line_parallelism_l134_13499

theorem tangent_line_parallelism (a : ℝ) :
  a > -2 * Real.sqrt 2 →
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ 2 * x₁ + 1 / x₁ = Real.exp x₂ - a :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parallelism_l134_13499


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_l134_13457

/-- A number is prime if it's greater than 1 and its only divisors are 1 and itself -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The expression we're interested in -/
def expression (p q : ℕ) : ℕ :=
  2^2 + p^2 + q^2

theorem prime_sum_of_squares : 
  ∀ p q : ℕ, isPrime p → isPrime q → isPrime (expression p q) → 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_l134_13457


namespace NUMINAMATH_CALUDE_water_transfer_problem_l134_13477

theorem water_transfer_problem (initial_volume : ℝ) (loss_percentage : ℝ) (hemisphere_volume : ℝ) : 
  initial_volume = 10936 →
  loss_percentage = 2.5 →
  hemisphere_volume = 4 →
  ⌈(initial_volume * (1 - loss_percentage / 100)) / hemisphere_volume⌉ = 2666 := by
  sorry

end NUMINAMATH_CALUDE_water_transfer_problem_l134_13477


namespace NUMINAMATH_CALUDE_certain_number_proof_l134_13404

theorem certain_number_proof (x : ℕ+) (h : (55 * x.val) % 7 = 6) : x.val % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l134_13404


namespace NUMINAMATH_CALUDE_no_five_naturals_product_equals_sum_l134_13434

theorem no_five_naturals_product_equals_sum :
  ¬ ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ d * e = a + b + c + d + e := by
sorry

end NUMINAMATH_CALUDE_no_five_naturals_product_equals_sum_l134_13434


namespace NUMINAMATH_CALUDE_tangency_condition_l134_13498

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse x y ∧ hyperbola x y m ∧
  ∀ x' y' : ℝ, ellipse x' y' ∧ hyperbola x' y' m → (x', y') = (x, y)

-- State the theorem
theorem tangency_condition :
  ∀ m : ℝ, are_tangent m ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_tangency_condition_l134_13498


namespace NUMINAMATH_CALUDE_weed_eater_string_cost_is_seven_l134_13406

-- Define the number of lawnmower blades
def num_blades : ℕ := 4

-- Define the cost per blade in dollars
def cost_per_blade : ℕ := 8

-- Define the total spent on supplies in dollars
def total_spent : ℕ := 39

-- Define the cost of the weed eater string
def weed_eater_string_cost : ℕ := total_spent - (num_blades * cost_per_blade)

-- Theorem statement
theorem weed_eater_string_cost_is_seven :
  weed_eater_string_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_weed_eater_string_cost_is_seven_l134_13406


namespace NUMINAMATH_CALUDE_min_handshakes_theorem_l134_13425

/-- Represents the number of people at the conference -/
def num_people : ℕ := 30

/-- Represents the minimum number of handshakes per person -/
def min_handshakes_per_person : ℕ := 3

/-- Calculates the minimum number of handshakes for the given conditions -/
def min_total_handshakes : ℕ :=
  (num_people * min_handshakes_per_person) / 2

/-- Theorem stating that the minimum number of handshakes is 45 -/
theorem min_handshakes_theorem :
  min_total_handshakes = 45 := by sorry

end NUMINAMATH_CALUDE_min_handshakes_theorem_l134_13425


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_product_l134_13416

theorem quadratic_rational_solutions_product : ∃ (d₁ d₂ : ℕ+),
  (∀ (d : ℕ+), (∃ (x : ℚ), 8 * x^2 + 16 * x + d.val = 0) ↔ (d = d₁ ∨ d = d₂)) ∧
  d₁.val * d₂.val = 48 :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_product_l134_13416


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l134_13453

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (total_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  Nat.choose (total_balls + num_boxes - 1) (num_boxes - 1)

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes
    with at least one ball in each box -/
def distribute_balls_no_empty (total_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  distribute_balls (total_balls - num_boxes) num_boxes

theorem seven_balls_three_boxes :
  distribute_balls_no_empty 7 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l134_13453


namespace NUMINAMATH_CALUDE_final_concentration_is_correct_l134_13484

/-- Represents the volume of saline solution in the cup -/
def initial_volume : ℝ := 1

/-- Represents the initial concentration of the saline solution -/
def initial_concentration : ℝ := 0.16

/-- Represents the volume ratio of the large ball -/
def large_ball_ratio : ℝ := 10

/-- Represents the volume ratio of the medium ball -/
def medium_ball_ratio : ℝ := 4

/-- Represents the volume ratio of the small ball -/
def small_ball_ratio : ℝ := 3

/-- Represents the percentage of solution that overflows when the small ball is immersed -/
def overflow_percentage : ℝ := 0.1

/-- Calculates the final concentration of the saline solution after the process -/
def final_concentration : ℝ := sorry

/-- Theorem stating that the final concentration is approximately 10.7% -/
theorem final_concentration_is_correct : 
  ∀ ε > 0, |final_concentration - 0.107| < ε := by sorry

end NUMINAMATH_CALUDE_final_concentration_is_correct_l134_13484


namespace NUMINAMATH_CALUDE_max_area_parallelogram_in_circle_l134_13490

/-- A right-angled parallelogram inscribed in a circle of radius r has maximum area when its sides are r√2 -/
theorem max_area_parallelogram_in_circle (r : ℝ) (h : r > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a * b ≤ x * y) ∧
  (x^2 + y^2 = (2*r)^2) ∧
  x = r * Real.sqrt 2 ∧ y = r * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_area_parallelogram_in_circle_l134_13490


namespace NUMINAMATH_CALUDE_complex_arithmetic_equalities_l134_13467

theorem complex_arithmetic_equalities :
  (16 / (-2)^3 - (-1/2)^3 * (-4) + 2.5 = 0) ∧
  ((-1)^2022 + |(-2)^2 + 4| - (1/2 - 1/4 + 1/8) * (-24) = 10) := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equalities_l134_13467


namespace NUMINAMATH_CALUDE_mina_driving_problem_l134_13437

theorem mina_driving_problem (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_average_speed : ℝ) 
  (h : initial_distance = 20 ∧ initial_speed = 40 ∧ second_speed = 60 ∧ target_average_speed = 55) :
  ∃ additional_distance : ℝ,
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = target_average_speed ∧
    additional_distance = 90 := by
  sorry

end NUMINAMATH_CALUDE_mina_driving_problem_l134_13437


namespace NUMINAMATH_CALUDE_rectangle_length_breadth_difference_l134_13421

/-- Given a rectangular plot with breadth 11 metres and area 21 times its breadth,
    the difference between its length and breadth is 10 metres. -/
theorem rectangle_length_breadth_difference : ℝ → Prop :=
  fun difference =>
    ∀ (length breadth area : ℝ),
      breadth = 11 →
      area = 21 * breadth →
      area = length * breadth →
      difference = length - breadth →
      difference = 10

/-- Proof of the theorem -/
lemma prove_rectangle_length_breadth_difference :
  rectangle_length_breadth_difference 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_breadth_difference_l134_13421


namespace NUMINAMATH_CALUDE_seven_digit_integers_count_l134_13430

/-- The number of different seven-digit integers that can be formed using the digits 1, 2, 2, 3, 3, 3, and 5 -/
def seven_digit_integers : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 3)

/-- Theorem stating that the number of different seven-digit integers
    formed using the digits 1, 2, 2, 3, 3, 3, and 5 is equal to 420 -/
theorem seven_digit_integers_count : seven_digit_integers = 420 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_integers_count_l134_13430


namespace NUMINAMATH_CALUDE_max_value_of_ab_l134_13473

theorem max_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 8) :
  ∃ (m : ℝ), m = 8 ∧ ∀ x y, x > 0 → y > 0 → x + 2*y = 8 → x*y ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_ab_l134_13473


namespace NUMINAMATH_CALUDE_earnings_difference_l134_13462

def car_price : ℕ := 5200
def inspection_cost : ℕ := car_price / 10
def headlight_cost : ℕ := 80
def tire_cost : ℕ := 3 * headlight_cost

def first_offer_earnings : ℕ := car_price - inspection_cost
def second_offer_earnings : ℕ := car_price - (headlight_cost + tire_cost)

theorem earnings_difference : second_offer_earnings - first_offer_earnings = 200 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_l134_13462


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_three_l134_13417

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 n1 : ℝ) (m2 n2 : ℝ) : Prop := m1 * n2 = m2 * n1

/-- The line ax+3y+1=0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 1 = 0

/-- The line 2x+(a+1)y+1=0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := 2 * x + (a + 1) * y + 1 = 0

/-- The main theorem: the lines are parallel if and only if a = -3 -/
theorem lines_parallel_iff_a_eq_neg_three (a : ℝ) : 
  parallel a 3 2 (a + 1) ↔ a = -3 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_three_l134_13417


namespace NUMINAMATH_CALUDE_correspondence_count_l134_13463

-- Define the sets and correspondences
def Triangle : Type := sorry
def Circle : Type := sorry
def RealNumber : Type := ℝ

-- Define the correspondences
def correspondence1 : Triangle → Circle := sorry
def correspondence2 : Triangle → RealNumber := sorry
def correspondence3 : RealNumber → RealNumber := sorry
def correspondence4 : RealNumber → RealNumber := sorry

-- Define what it means to be a mapping
def is_mapping (f : α → β) : Prop := ∀ x : α, ∃! y : β, f x = y

-- Define what it means to be a function
def is_function (f : α → β) : Prop := ∀ x : α, ∃ y : β, f x = y

-- The main theorem
theorem correspondence_count :
  (is_mapping correspondence1 ∧
   is_mapping correspondence2 ∧
   is_mapping correspondence3 ∧
   ¬is_mapping correspondence4) ∧
  (¬is_function correspondence1 ∧
   is_function correspondence2 ∧
   is_function correspondence3 ∧
   ¬is_function correspondence4) :=
sorry

end NUMINAMATH_CALUDE_correspondence_count_l134_13463


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l134_13465

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_1 + a_2 = 10 and a_4 = a_3 + 2,
    prove that a_3 + a_4 = 18. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 2 = 10)
  (h_diff : a 4 = a 3 + 2) :
  a 3 + a 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l134_13465


namespace NUMINAMATH_CALUDE_fixed_cost_satisfies_break_even_equation_l134_13443

/-- The one-time fixed cost for a book publishing project -/
def fixed_cost : ℝ := 35678

/-- The variable cost per book -/
def variable_cost_per_book : ℝ := 11.50

/-- The selling price per book -/
def selling_price_per_book : ℝ := 20.25

/-- The number of books needed to break even -/
def break_even_quantity : ℕ := 4072

/-- Theorem stating that the fixed cost satisfies the break-even equation -/
theorem fixed_cost_satisfies_break_even_equation : 
  fixed_cost + (break_even_quantity : ℝ) * variable_cost_per_book = 
  (break_even_quantity : ℝ) * selling_price_per_book :=
by sorry

end NUMINAMATH_CALUDE_fixed_cost_satisfies_break_even_equation_l134_13443


namespace NUMINAMATH_CALUDE_remainder_three_pow_2040_mod_5_l134_13447

theorem remainder_three_pow_2040_mod_5 : 3^2040 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_pow_2040_mod_5_l134_13447


namespace NUMINAMATH_CALUDE_boat_rental_solutions_l134_13403

theorem boat_rental_solutions :
  ∀ (x y : ℕ),
    12 * x + 5 * y = 99 →
    ((x = 2 ∧ y = 15) ∨ (x = 7 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_boat_rental_solutions_l134_13403


namespace NUMINAMATH_CALUDE_restaurant_bill_change_l134_13476

/-- Calculates the change received after a restaurant bill payment --/
theorem restaurant_bill_change
  (salmon_price truffled_mac_price chicken_katsu_price seafood_pasta_price black_burger_price wine_price : ℝ)
  (discount_rate service_charge_rate additional_tip_rate : ℝ)
  (payment : ℝ)
  (h_salmon : salmon_price = 40)
  (h_truffled_mac : truffled_mac_price = 20)
  (h_chicken_katsu : chicken_katsu_price = 25)
  (h_seafood_pasta : seafood_pasta_price = 30)
  (h_black_burger : black_burger_price = 15)
  (h_wine : wine_price = 50)
  (h_discount : discount_rate = 0.1)
  (h_service : service_charge_rate = 0.12)
  (h_tip : additional_tip_rate = 0.05)
  (h_payment : payment = 300) :
  let food_cost := salmon_price + truffled_mac_price + chicken_katsu_price + seafood_pasta_price + black_burger_price
  let total_cost := food_cost + wine_price
  let service_charge := service_charge_rate * total_cost
  let bill_before_discount := total_cost + service_charge
  let discount := discount_rate * food_cost
  let bill_after_discount := bill_before_discount - discount
  let additional_tip := additional_tip_rate * bill_after_discount
  let final_bill := bill_after_discount + additional_tip
  payment - final_bill = 101.97 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_change_l134_13476


namespace NUMINAMATH_CALUDE_extremum_sum_l134_13415

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_sum (a b : ℝ) : 
  (f a b 1 = 10) ∧ (f' a b 1 = 0) → a + b = -7 :=
by
  sorry

#check extremum_sum

end NUMINAMATH_CALUDE_extremum_sum_l134_13415


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l134_13446

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l134_13446


namespace NUMINAMATH_CALUDE_original_nes_price_l134_13402

/-- Calculates the original NES sale price before tax given trade-in values and final payment details --/
theorem original_nes_price
  (snes_value : ℝ)
  (snes_credit_rate : ℝ)
  (gameboy_value : ℝ)
  (gameboy_credit_rate : ℝ)
  (ps2_value : ℝ)
  (ps2_credit_rate : ℝ)
  (nes_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (cash_paid : ℝ)
  (change_received : ℝ)
  (free_game_value : ℝ)
  (h1 : snes_value = 150)
  (h2 : snes_credit_rate = 0.8)
  (h3 : gameboy_value = 50)
  (h4 : gameboy_credit_rate = 0.75)
  (h5 : ps2_value = 100)
  (h6 : ps2_credit_rate = 0.6)
  (h7 : nes_discount_rate = 0.15)
  (h8 : sales_tax_rate = 0.05)
  (h9 : cash_paid = 80)
  (h10 : change_received = 10)
  (h11 : free_game_value = 30) :
  ∃ (original_price : ℝ), abs (original_price - 289.08) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_original_nes_price_l134_13402


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l134_13420

/-- A function that checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- A function that converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ n : ℕ,
    n > 10 →
    (isPalindrome n 2 ∧ isPalindrome n 4) →
    n ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l134_13420


namespace NUMINAMATH_CALUDE_integer_fraction_pairs_l134_13433

theorem integer_fraction_pairs : 
  ∀ a b : ℕ+, 
    (∃ k l : ℤ, (a.val^2 + b.val : ℤ) = k * (b.val^2 - a.val) ∧ 
                (b.val^2 + a.val : ℤ) = l * (a.val^2 - b.val)) →
    ((a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_pairs_l134_13433


namespace NUMINAMATH_CALUDE_cans_per_bag_l134_13401

theorem cans_per_bag (total_cans : ℕ) (total_bags : ℕ) (h1 : total_cans = 42) (h2 : total_bags = 7) :
  total_cans / total_bags = 6 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l134_13401


namespace NUMINAMATH_CALUDE_cylinder_from_equation_l134_13418

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = d -/
def CylindricalSet (d : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = d}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ d : ℝ, d > 0 ∧ S = CylindricalSet d

/-- Theorem: The set of points satisfying r = d forms a cylinder -/
theorem cylinder_from_equation (d : ℝ) (h : d > 0) : 
  IsCylinder (CylindricalSet d) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_from_equation_l134_13418


namespace NUMINAMATH_CALUDE_monotonicity_f_when_a_is_1_min_a_when_f_has_no_zeros_l134_13432

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

-- Part 1: Monotonicity of f when a = 1
theorem monotonicity_f_when_a_is_1 :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → f 1 x₁ > f 1 x₂ ∧
  ∀ x₃ x₄, 2 ≤ x₃ ∧ x₃ < x₄ → f 1 x₃ < f 1 x₄ := by sorry

-- Part 2: Minimum value of a when f has no zeros in (0, 1/2)
theorem min_a_when_f_has_no_zeros :
  (∀ x, 0 < x ∧ x < 1/2 → f a x ≠ 0) →
  a ≥ 2 - 4 * Real.log 2 := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_f_when_a_is_1_min_a_when_f_has_no_zeros_l134_13432


namespace NUMINAMATH_CALUDE_yoojeongs_marbles_l134_13488

theorem yoojeongs_marbles (marbles_given : ℕ) (marbles_left : ℕ) :
  marbles_given = 8 →
  marbles_left = 24 →
  marbles_given + marbles_left = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_yoojeongs_marbles_l134_13488


namespace NUMINAMATH_CALUDE_sin_cos_sum_identity_l134_13486

theorem sin_cos_sum_identity : 
  Real.sin (70 * π / 180) * Real.sin (10 * π / 180) + 
  Real.cos (10 * π / 180) * Real.cos (70 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_identity_l134_13486


namespace NUMINAMATH_CALUDE_jacks_stamp_collection_value_l134_13475

/-- Given a collection of stamps where all stamps have equal value, 
    calculate the total value of the collection. -/
def stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℕ) : ℕ :=
  total_stamps * (sample_value / sample_stamps)

/-- Prove that Jack's stamp collection is worth 80 dollars -/
theorem jacks_stamp_collection_value :
  stamp_collection_value 20 4 16 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jacks_stamp_collection_value_l134_13475


namespace NUMINAMATH_CALUDE_step_waddle_difference_is_six_l134_13491

/-- The number of steps Gerald takes between consecutive lamp posts -/
def gerald_steps : ℕ := 55

/-- The number of waddles Patricia takes between consecutive lamp posts -/
def patricia_waddles : ℕ := 15

/-- The number of lamp posts -/
def num_posts : ℕ := 31

/-- The total distance between the first and last lamp post in feet -/
def total_distance : ℕ := 3720

/-- Gerald's step length in feet -/
def gerald_step_length : ℚ := total_distance / (gerald_steps * (num_posts - 1))

/-- Patricia's waddle length in feet -/
def patricia_waddle_length : ℚ := total_distance / (patricia_waddles * (num_posts - 1))

/-- The difference between Gerald's step length and Patricia's waddle length -/
def step_waddle_difference : ℚ := patricia_waddle_length - gerald_step_length

theorem step_waddle_difference_is_six :
  step_waddle_difference = 6 := by sorry

end NUMINAMATH_CALUDE_step_waddle_difference_is_six_l134_13491


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l134_13400

theorem arithmetic_sequence_squares (k : ℤ) : ∃ (a : ℕ → ℤ), 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧ 
  (a 1)^2 = 36 + k ∧
  (a 2)^2 = 300 + k ∧
  (a 3)^2 = 596 + k ∧
  k = 925 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l134_13400


namespace NUMINAMATH_CALUDE_proposition_relationship_l134_13424

theorem proposition_relationship :
  (∀ x : ℝ, (x - 3) * (x + 1) > 0 → x^2 - 2*x + 1 > 0) ∧
  (∃ x : ℝ, x^2 - 2*x + 1 > 0 ∧ (x - 3) * (x + 1) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l134_13424


namespace NUMINAMATH_CALUDE_count_less_than_10000_l134_13492

def count_numbers_with_at_most_three_digits (n : ℕ) : ℕ :=
  sorry

theorem count_less_than_10000 : 
  count_numbers_with_at_most_three_digits 10000 = 3231 := by
  sorry

end NUMINAMATH_CALUDE_count_less_than_10000_l134_13492


namespace NUMINAMATH_CALUDE_distance_AB_is_550_l134_13482

/-- The distance between points A and B --/
def distance_AB : ℝ := 550

/-- Xiaodong's speed in meters per minute --/
def speed_Xiaodong : ℝ := 50

/-- Xiaorong's speed in meters per minute --/
def speed_Xiaorong : ℝ := 60

/-- Time taken for Xiaodong and Xiaorong to meet, in minutes --/
def meeting_time : ℝ := 10

/-- Theorem stating that the distance between points A and B is 550 meters --/
theorem distance_AB_is_550 :
  distance_AB = (speed_Xiaodong + speed_Xiaorong) * meeting_time / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_AB_is_550_l134_13482


namespace NUMINAMATH_CALUDE_remainder_after_adding_2025_l134_13451

theorem remainder_after_adding_2025 (n : ℤ) (h : n % 5 = 3) : (n + 2025) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2025_l134_13451


namespace NUMINAMATH_CALUDE_alan_cd_purchase_cost_l134_13478

theorem alan_cd_purchase_cost :
  let avnPrice : ℝ := 12
  let darkPrice : ℝ := 2 * avnPrice
  let darkTotal : ℝ := 2 * darkPrice
  let otherTotal : ℝ := darkTotal + avnPrice
  let ninetyPrice : ℝ := 0.4 * otherTotal
  darkTotal + avnPrice + ninetyPrice = 84 := by
  sorry

end NUMINAMATH_CALUDE_alan_cd_purchase_cost_l134_13478


namespace NUMINAMATH_CALUDE_last_term_formula_l134_13448

def u (n : ℕ) : ℕ := 2 + 5 * ((n - 1) % (3 * ((n - 1).sqrt + 1) - 1))

def f (n : ℕ) : ℕ := (15 * n^2 + 10 * n + 4) / 2

theorem last_term_formula (n : ℕ) : 
  u ((n^2 + n) / 2) = f n :=
sorry

end NUMINAMATH_CALUDE_last_term_formula_l134_13448


namespace NUMINAMATH_CALUDE_part_one_part_two_l134_13459

-- Define the function y
def y (x a : ℝ) : ℝ := 2 * x^2 - (a + 2) * x + a

-- Part 1
theorem part_one : 
  ∀ x : ℝ, y x (-1) > 0 ↔ (x > 1 ∨ x < -1/2) := by sorry

-- Part 2
theorem part_two :
  ∀ a x₁ x₂ : ℝ, 
    (x₁ > 0 ∧ x₂ > 0) →
    (2 * x₁^2 - (a + 2) * x₁ + a = x₁ + 1) →
    (2 * x₂^2 - (a + 2) * x₂ + a = x₂ + 1) →
    (∀ x : ℝ, x > 0 → x₂/x₁ + x₁/x₂ ≥ 6) ∧ 
    (∃ a : ℝ, x₂/x₁ + x₁/x₂ = 6) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l134_13459


namespace NUMINAMATH_CALUDE_two_numbers_subtracted_from_32_l134_13412

theorem two_numbers_subtracted_from_32 : ∃ (A B : ℤ), 
  A ≠ B ∧
  ((32 - A = 23 ∧ 32 - B = 13) ∨ (32 - A = 13 ∧ 32 - B = 23)) ∧
  ¬ (∃ (k : ℤ), |A - B| = 11 * k) ∧
  A = 9 ∧ B = 19 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_subtracted_from_32_l134_13412


namespace NUMINAMATH_CALUDE_dante_sold_coconuts_l134_13428

/-- The number of coconuts Paolo has -/
def paolo_coconuts : ℕ := 14

/-- The number of coconuts Dante has relative to Paolo -/
def dante_multiplier : ℕ := 3

/-- The number of coconuts Dante has left after selling -/
def dante_coconuts_left : ℕ := 32

/-- The number of coconuts Dante sold -/
def dante_sold : ℕ := dante_multiplier * paolo_coconuts - dante_coconuts_left

theorem dante_sold_coconuts : dante_sold = 10 := by
  sorry

end NUMINAMATH_CALUDE_dante_sold_coconuts_l134_13428


namespace NUMINAMATH_CALUDE_divisibility_of_polynomial_l134_13449

theorem divisibility_of_polynomial (x : ℤ) :
  (x^2 + 1) * (x^8 - x^6 + x^4 - x^2 + 1) = x^10 + 1 →
  ∃ k : ℤ, x^2030 + 1 = k * (x^8 - x^6 + x^4 - x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_polynomial_l134_13449


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l134_13497

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  x^6 + 2*x^5 - 3*x^4 + x^3 - 2*x^2 + 5*x - 1 =
  (x - 1) * (x + 2) * (x - 3) * q + (17*x^2 - 52*x + 38) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l134_13497


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l134_13496

/-- A quadratic function f(x) = x^2 + 2x + k has its vertex on the x-axis if and only if k = 1 -/
theorem vertex_on_x_axis (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*y + k ≥ x^2 + 2*x + k) ↔ 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l134_13496


namespace NUMINAMATH_CALUDE_seven_x_minus_three_y_equals_thirteen_l134_13441

theorem seven_x_minus_three_y_equals_thirteen 
  (x y : ℝ) 
  (h1 : 4 * x + y = 8) 
  (h2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 13 := by
sorry

end NUMINAMATH_CALUDE_seven_x_minus_three_y_equals_thirteen_l134_13441


namespace NUMINAMATH_CALUDE_ball_max_height_l134_13405

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 161 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l134_13405


namespace NUMINAMATH_CALUDE_perpendicular_planes_l134_13493

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the theorem
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h_m_perp_α : perpendicular m α) 
  (h_m_para_β : parallel m β) : 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l134_13493


namespace NUMINAMATH_CALUDE_x4_plus_y4_equals_7_l134_13423

theorem x4_plus_y4_equals_7 (x y : ℝ) 
  (hx : x^4 + x^2 = 3) 
  (hy : y^4 - y^2 = 3) : 
  x^4 + y^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x4_plus_y4_equals_7_l134_13423


namespace NUMINAMATH_CALUDE_blueberry_muffin_percentage_l134_13452

/-- Calculates the percentage of blueberry muffins out of the total muffins -/
theorem blueberry_muffin_percentage
  (num_cartons : ℕ)
  (blueberries_per_carton : ℕ)
  (blueberries_per_muffin : ℕ)
  (num_cinnamon_muffins : ℕ)
  (h1 : num_cartons = 3)
  (h2 : blueberries_per_carton = 200)
  (h3 : blueberries_per_muffin = 10)
  (h4 : num_cinnamon_muffins = 60)
  : (((num_cartons * blueberries_per_carton) / blueberries_per_muffin : ℚ) /
     ((num_cartons * blueberries_per_carton) / blueberries_per_muffin + num_cinnamon_muffins)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_muffin_percentage_l134_13452


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l134_13466

-- Define the triangle ABC and related points
variable (A B C D E F : ℝ × ℝ)
variable (α : ℝ)

-- Define the conditions
axiom right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
axiom parallel_line : (D.2 - E.2) / (D.1 - E.1) = (B.2 - A.2) / (B.1 - A.1)
axiom DE_length : Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = 2
axiom BE_length : Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2) = 1
axiom BF_length : Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 1
axiom F_on_hypotenuse : (F.1 - A.1) / (B.1 - A.1) = (F.2 - A.2) / (B.2 - A.2)
axiom angle_FCB : Real.cos α = (F.1 - C.1) / Real.sqrt ((F.1 - C.1)^2 + (F.2 - C.2)^2)

-- Define the theorem
theorem area_of_triangle_ABC :
  (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) =
  (1/2) * (2 * Real.cos (2*α) + 1)^2 * Real.tan (2*α) := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l134_13466


namespace NUMINAMATH_CALUDE_factorial_sum_unique_solution_l134_13440

theorem factorial_sum_unique_solution :
  ∀ w x y z : ℕ+,
  w.val.factorial = x.val.factorial + y.val.factorial + z.val.factorial →
  w = 3 ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_unique_solution_l134_13440


namespace NUMINAMATH_CALUDE_tameka_cracker_sales_l134_13444

/-- Proves that given the conditions in the problem, Tameka sold 30 more boxes on Saturday than on Friday --/
theorem tameka_cracker_sales : ∀ (saturday_sales : ℕ),
  (40 + saturday_sales + saturday_sales / 2 = 145) →
  (saturday_sales = 40 + 30) := by
  sorry

end NUMINAMATH_CALUDE_tameka_cracker_sales_l134_13444


namespace NUMINAMATH_CALUDE_bottles_left_l134_13474

theorem bottles_left (initial : Real) (maria_drank : Real) (sister_drank : Real) 
  (h1 : initial = 45.0)
  (h2 : maria_drank = 14.0)
  (h3 : sister_drank = 8.0) :
  initial - maria_drank - sister_drank = 23.0 := by
  sorry

end NUMINAMATH_CALUDE_bottles_left_l134_13474


namespace NUMINAMATH_CALUDE_softball_team_composition_l134_13414

theorem softball_team_composition :
  ∀ (men women : ℕ),
  men + women = 16 →
  (men : ℚ) / (women : ℚ) = 7 / 9 →
  women - men = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_composition_l134_13414


namespace NUMINAMATH_CALUDE_popsicle_sticks_theorem_l134_13494

def steve_sticks : ℕ := 12

def sid_sticks : ℕ := 2 * steve_sticks

def sam_sticks : ℕ := 3 * sid_sticks

def total_sticks : ℕ := steve_sticks + sid_sticks + sam_sticks

theorem popsicle_sticks_theorem : total_sticks = 108 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_sticks_theorem_l134_13494


namespace NUMINAMATH_CALUDE_even_sum_theorem_l134_13411

theorem even_sum_theorem (n : ℕ) (h1 : Odd n) 
  (h2 : (Finset.sum (Finset.filter Even (Finset.range n)) id) = 95 * 96) : 
  n = 191 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_theorem_l134_13411


namespace NUMINAMATH_CALUDE_digit_sum_in_multiplication_l134_13472

theorem digit_sum_in_multiplication (c d a b : ℕ) : 
  c < 10 → d < 10 → a < 10 → b < 10 →
  (30 + c) * (10 * d + 4) = 100 * a + 10 * b + 8 →
  c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_in_multiplication_l134_13472


namespace NUMINAMATH_CALUDE_election_winner_percentage_l134_13410

theorem election_winner_percentage (winner_votes loser_votes total_votes : ℕ) 
  (h1 : winner_votes = 899)
  (h2 : winner_votes - loser_votes = 348)
  (h3 : total_votes = winner_votes + loser_votes) :
  (winner_votes : ℝ) / (total_votes : ℝ) * 100 = 899 / 1450 * 100 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l134_13410


namespace NUMINAMATH_CALUDE_closest_points_on_hyperbola_l134_13436

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 9

/-- The distance squared function from a point (x, y) to A(0, -3) -/
def distance_squared (x y : ℝ) : ℝ := x^2 + (y + 3)^2

/-- The point A -/
def A : ℝ × ℝ := (0, -3)

/-- Theorem stating that the given points are the closest to A on the hyperbola -/
theorem closest_points_on_hyperbola :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    (x₁ = -3 * Real.sqrt 5 / 2 ∧ y₁ = -3 / 2) ∧
    (x₂ = 3 * Real.sqrt 5 / 2 ∧ y₂ = -3 / 2) ∧
    (∀ (x y : ℝ), hyperbola x y → 
      distance_squared x y ≥ distance_squared x₁ y₁ ∧
      distance_squared x y ≥ distance_squared x₂ y₂) :=
sorry

end NUMINAMATH_CALUDE_closest_points_on_hyperbola_l134_13436


namespace NUMINAMATH_CALUDE_orange_juice_bottles_l134_13427

theorem orange_juice_bottles (orange_price apple_price total_bottles total_cost : ℚ) 
  (h1 : orange_price = 70/100)
  (h2 : apple_price = 60/100)
  (h3 : total_bottles = 70)
  (h4 : total_cost = 4620/100) :
  ∃ (orange_bottles : ℚ), 
    orange_bottles * orange_price + (total_bottles - orange_bottles) * apple_price = total_cost ∧ 
    orange_bottles = 42 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_bottles_l134_13427


namespace NUMINAMATH_CALUDE_sum_of_squares_near_n_l134_13455

theorem sum_of_squares_near_n (n : ℕ) (h : n > 10000) :
  ∃ m : ℕ, ∃ x y : ℕ, 
    m = x^2 + y^2 ∧ 
    0 < m - n ∧
    (m - n : ℝ) < 3 * Real.sqrt (n : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_near_n_l134_13455


namespace NUMINAMATH_CALUDE_find_x1_l134_13413

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h_order : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h_eq1 : (1-x1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^2 + x4^2 = 1/3)
  (h_sum : x1 + x2 + x3 + x4 = 2) : 
  x1 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_find_x1_l134_13413


namespace NUMINAMATH_CALUDE_elsa_lost_marbles_l134_13422

/-- The number of marbles Elsa lost at breakfast -/
def x : ℕ := sorry

/-- Elsa's initial number of marbles -/
def initial_marbles : ℕ := 40

/-- Number of marbles Elsa gave to Susie -/
def marbles_given_to_susie : ℕ := 5

/-- Number of new marbles Elsa's mom bought -/
def new_marbles : ℕ := 12

/-- Elsa's final number of marbles -/
def final_marbles : ℕ := 54

theorem elsa_lost_marbles : 
  initial_marbles - x - marbles_given_to_susie + new_marbles + 2 * marbles_given_to_susie = final_marbles ∧
  x = 3 := by sorry

end NUMINAMATH_CALUDE_elsa_lost_marbles_l134_13422


namespace NUMINAMATH_CALUDE_solution_equation_1_solution_equation_2_l134_13483

-- Equation 1
theorem solution_equation_1 (x : ℝ) : 2*x - 3*(2*x - 3) = x + 4 ↔ x = 1 := by sorry

-- Equation 2
theorem solution_equation_2 (x : ℝ) : (3*x - 1)/4 - 1 = (5*x - 7)/6 ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation_1_solution_equation_2_l134_13483


namespace NUMINAMATH_CALUDE_equation_solution_sum_l134_13458

theorem equation_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 15 = 25) → 
  (d^2 - 6*d + 15 = 25) → 
  c ≥ d → 
  3*c + 2*d = 15 + Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_sum_l134_13458


namespace NUMINAMATH_CALUDE_f_composition_one_ninth_l134_13469

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 + 1 else Real.log x / Real.log 3

theorem f_composition_one_ninth : f (f (1/9)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_one_ninth_l134_13469


namespace NUMINAMATH_CALUDE_angle_measure_proof_l134_13495

theorem angle_measure_proof : 
  ∀ x : ℝ, 
    (90 - x = (1/7) * x + 26) → 
    x = 56 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l134_13495


namespace NUMINAMATH_CALUDE_special_quadrilateral_integer_perimeter_l134_13439

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  O : ℝ × ℝ
  -- Perpendicular conditions
  ab_perp_bc : (A.1 - B.1) * (B.1 - C.1) + (A.2 - B.2) * (B.2 - C.2) = 0
  bc_perp_cd : (B.1 - C.1) * (C.1 - D.1) + (B.2 - C.2) * (C.2 - D.2) = 0
  -- BC tangent to circle condition
  bc_tangent : (B.1 - O.1) * (C.1 - O.1) + (B.2 - O.2) * (C.2 - O.2) = 0
  -- AD is diameter
  ad_diameter : (A.1 - O.1)^2 + (A.2 - O.2)^2 = (D.1 - O.1)^2 + (D.2 - O.2)^2

/-- Perimeter of the quadrilateral is an integer when AB and CD are integers with AB = 2CD -/
theorem special_quadrilateral_integer_perimeter 
  (q : SpecialQuadrilateral) 
  (ab cd : ℕ) 
  (h_ab : ab = 2 * cd) 
  (h_ab_length : (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2 = ab^2) 
  (h_cd_length : (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 = cd^2) :
  ∃ (n : ℕ), 
    (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2 +
    (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2 +
    (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 +
    (q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_integer_perimeter_l134_13439


namespace NUMINAMATH_CALUDE_blake_initial_milk_l134_13489

/-- The amount of milk needed for one milkshake in ounces -/
def milk_per_milkshake : ℕ := 4

/-- The amount of ice cream needed for one milkshake in ounces -/
def ice_cream_per_milkshake : ℕ := 12

/-- The total amount of ice cream available in ounces -/
def total_ice_cream : ℕ := 192

/-- The amount of milk left over after making milkshakes in ounces -/
def milk_leftover : ℕ := 8

/-- The initial amount of milk Blake had -/
def initial_milk : ℕ := total_ice_cream / ice_cream_per_milkshake * milk_per_milkshake + milk_leftover

theorem blake_initial_milk :
  initial_milk = 72 :=
sorry

end NUMINAMATH_CALUDE_blake_initial_milk_l134_13489


namespace NUMINAMATH_CALUDE_watch_correction_l134_13442

/-- Represents the time difference between two dates in hours -/
def timeDifference (startDate endDate : Nat) : Nat :=
  (endDate - startDate) * 24

/-- Represents the additional hours on the last day -/
def additionalHours (startHour endHour : Nat) : Nat :=
  endHour - startHour

/-- Calculates the total hours elapsed -/
def totalHours (daysDifference additionalHours : Nat) : Nat :=
  daysDifference + additionalHours

/-- Converts daily time loss to hourly time loss -/
def hourlyLoss (dailyLoss : Rat) : Rat :=
  dailyLoss / 24

/-- Calculates the total time loss -/
def totalLoss (hourlyLoss : Rat) (totalHours : Nat) : Rat :=
  hourlyLoss * totalHours

theorem watch_correction (watchLoss : Rat) (startDate endDate startHour endHour : Nat) :
  watchLoss = 3.75 →
  startDate = 15 →
  endDate = 24 →
  startHour = 10 →
  endHour = 16 →
  totalLoss (hourlyLoss watchLoss) (totalHours (timeDifference startDate endDate) (additionalHours startHour endHour)) = 34.6875 := by
  sorry

#check watch_correction

end NUMINAMATH_CALUDE_watch_correction_l134_13442


namespace NUMINAMATH_CALUDE_soccer_balls_added_l134_13450

/-- Given the initial number of soccer balls, the number removed, and the final number of balls,
    prove that the number of soccer balls added is 21. -/
theorem soccer_balls_added 
  (initial : ℕ) 
  (removed : ℕ) 
  (final : ℕ) 
  (h1 : initial = 6) 
  (h2 : removed = 3) 
  (h3 : final = 24) : 
  final - (initial - removed) = 21 := by
  sorry

end NUMINAMATH_CALUDE_soccer_balls_added_l134_13450


namespace NUMINAMATH_CALUDE_pascal_triangle_53_l134_13419

theorem pascal_triangle_53 (p : ℕ) (h_prime : Prime p) (h_p : p = 53) :
  (∃! n : ℕ, ∃ k : ℕ, Nat.choose n k = p) :=
sorry

end NUMINAMATH_CALUDE_pascal_triangle_53_l134_13419
