import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_focus_on_y_axis_range_l1084_108454

/-- Represents the equation (m+1)x^2 + (2-m)y^2 = 1 -/
def hyperbola_equation (m x y : ℝ) : Prop :=
  (m + 1) * x^2 + (2 - m) * y^2 = 1

/-- Condition for the equation to represent a hyperbola with focus on y-axis -/
def is_hyperbola_on_y_axis (m : ℝ) : Prop :=
  m + 1 < 0 ∧ 2 - m > 0

/-- The theorem stating the range of m for which the equation represents
    a hyperbola with focus on the y-axis -/
theorem hyperbola_focus_on_y_axis_range :
  ∀ m : ℝ, is_hyperbola_on_y_axis m ↔ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_on_y_axis_range_l1084_108454


namespace NUMINAMATH_CALUDE_smallest_a_value_l1084_108421

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

end NUMINAMATH_CALUDE_smallest_a_value_l1084_108421


namespace NUMINAMATH_CALUDE_certain_number_problem_l1084_108453

theorem certain_number_problem : ∃ x : ℝ, (0.45 * x = 0.35 * 40 + 13) ∧ (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1084_108453


namespace NUMINAMATH_CALUDE_f_of_i_eq_zero_l1084_108472

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function f
def f (x : ℂ) : ℂ := x^3 - x^2 + x - 1

-- Theorem statement
theorem f_of_i_eq_zero : f i = 0 := by sorry

end NUMINAMATH_CALUDE_f_of_i_eq_zero_l1084_108472


namespace NUMINAMATH_CALUDE_pop_survey_l1084_108470

theorem pop_survey (total : ℕ) (pop_angle : ℕ) (pop_count : ℕ) : 
  total = 472 →
  pop_angle = 251 →
  (pop_count : ℝ) / total * 360 ≥ pop_angle.pred →
  (pop_count : ℝ) / total * 360 < pop_angle.succ →
  pop_count = 329 := by
sorry

end NUMINAMATH_CALUDE_pop_survey_l1084_108470


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l1084_108491

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ (n : ℕ), n = 888 ∧
    has_only_even_digits n ∧
    n < 1000 ∧
    n % 9 = 0 ∧
    ∀ m : ℕ, has_only_even_digits m ∧ m < 1000 ∧ m % 9 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l1084_108491


namespace NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l1084_108435

-- First expression
theorem simplify_expression1 (a b : ℝ) : (a - 2*b) - (2*b - 5*a) = 6*a - 4*b := by sorry

-- Second expression
theorem simplify_expression2 (m n : ℝ) : -m^2*n + (4*m*n^2 - 3*m*n) - 2*(m*n^2 - 3*m^2*n) = 5*m^2*n + 2*m*n^2 - 3*m*n := by sorry

end NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l1084_108435


namespace NUMINAMATH_CALUDE_william_window_wash_time_l1084_108440

/-- The time William spends washing vehicles -/
def william_car_wash (window_time : ℕ) : Prop :=
  let normal_car_time := window_time + 7 + 4 + 9
  let suv_time := 2 * normal_car_time
  let total_time := 2 * normal_car_time + suv_time
  total_time = 96

theorem william_window_wash_time :
  ∃ (w : ℕ), william_car_wash w ∧ w = 4 := by
  sorry

end NUMINAMATH_CALUDE_william_window_wash_time_l1084_108440


namespace NUMINAMATH_CALUDE_successive_discounts_theorem_l1084_108428

/-- The original price of the gadget -/
def original_price : ℝ := 350.00

/-- The first discount rate -/
def first_discount : ℝ := 0.10

/-- The second discount rate -/
def second_discount : ℝ := 0.12

/-- The final sale price as a percentage of the original price -/
def final_sale_percentage : ℝ := 0.792

theorem successive_discounts_theorem :
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  (final_price / original_price) = final_sale_percentage := by sorry

end NUMINAMATH_CALUDE_successive_discounts_theorem_l1084_108428


namespace NUMINAMATH_CALUDE_omega_double_omega_8n_plus_5_omega_2_pow_n_minus_1_l1084_108450

-- Define a function to represent the binary expansion of a non-negative integer
def binaryExpansion (n : ℕ) : List (Fin 2) := sorry

-- Define the ω function
def ω (n : ℕ) : ℕ := (binaryExpansion n).sum

-- Theorem 1
theorem omega_double (n : ℕ) : ω (2 * n) = ω n := by sorry

-- Theorem 2
theorem omega_8n_plus_5 (n : ℕ) : ω (8 * n + 5) = ω (4 * n + 3) := by sorry

-- Theorem 3
theorem omega_2_pow_n_minus_1 (n : ℕ) : ω (2^n - 1) = n := by sorry

end NUMINAMATH_CALUDE_omega_double_omega_8n_plus_5_omega_2_pow_n_minus_1_l1084_108450


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1084_108479

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (3 + m) * (-3) + 4 * 3 - 3 + 3 * m = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1084_108479


namespace NUMINAMATH_CALUDE_dihedral_angle_measure_l1084_108403

/-- Represents a dihedral angle formed by two plane mirrors -/
structure DihedralAngle where
  angle : ℝ

/-- Represents a light ray in the context of the problem -/
structure LightRay where
  perpendicular_to_edge : Bool
  parallel_to_first_mirror : Bool

/-- Represents the reflection pattern of the light ray -/
inductive ReflectionPattern
  | Alternating : ReflectionPattern

/-- Represents the result of the light ray's path -/
inductive PathResult
  | ReturnsAlongSamePath : PathResult

theorem dihedral_angle_measure 
  (d : DihedralAngle) 
  (l : LightRay) 
  (r : ReflectionPattern) 
  (p : PathResult) :
  l.perpendicular_to_edge = true →
  l.parallel_to_first_mirror = true →
  r = ReflectionPattern.Alternating →
  p = PathResult.ReturnsAlongSamePath →
  d.angle = 30 := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_measure_l1084_108403


namespace NUMINAMATH_CALUDE_quadrilateral_with_parallel_sides_and_congruent_diagonals_is_rectangle_l1084_108434

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a quadrilateral
def has_parallel_sides (q : Quadrilateral) : Prop := 
  sorry

def has_congruent_diagonals (q : Quadrilateral) : Prop := 
  sorry

def is_rectangle (q : Quadrilateral) : Prop := 
  sorry

-- Theorem statement
theorem quadrilateral_with_parallel_sides_and_congruent_diagonals_is_rectangle 
  (q : Quadrilateral) : 
  has_parallel_sides q → has_congruent_diagonals q → is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_with_parallel_sides_and_congruent_diagonals_is_rectangle_l1084_108434


namespace NUMINAMATH_CALUDE_danny_chemistry_marks_l1084_108475

theorem danny_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 76) 
  (h2 : mathematics = 65) 
  (h3 : physics = 82) 
  (h4 : biology = 75) 
  (h5 : average = 73) 
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) :
  chemistry = 67 :=
by
  sorry

end NUMINAMATH_CALUDE_danny_chemistry_marks_l1084_108475


namespace NUMINAMATH_CALUDE_pi_half_irrational_l1084_108405

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l1084_108405


namespace NUMINAMATH_CALUDE_investment_problem_l1084_108463

theorem investment_problem (P : ℝ) : 
  let A1 := 1.02 * P - 100
  let A2 := 1.03 * A1 + 200
  let A3 := 1.04 * A2
  let A4 := 1.05 * A3
  let A5 := 1.06 * A4
  A5 = 750 →
  1.19304696 * P + 112.27824 = 750 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l1084_108463


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1084_108468

theorem binomial_expansion_coefficient (n : ℕ) : 
  (9 : ℕ) * (n.choose 2) = 54 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1084_108468


namespace NUMINAMATH_CALUDE_b_power_a_equals_negative_one_l1084_108433

theorem b_power_a_equals_negative_one (a b : ℝ) : 
  (a - 5)^2 + |2*b + 2| = 0 → b^a = -1 := by sorry

end NUMINAMATH_CALUDE_b_power_a_equals_negative_one_l1084_108433


namespace NUMINAMATH_CALUDE_only_C_in_position_I_l1084_108497

-- Define a structure for a rectangle with labeled sides
structure LabeledRectangle where
  top : ℕ
  bottom : ℕ
  left : ℕ
  right : ℕ

-- Define the five rectangles
def rectangle_A : LabeledRectangle := ⟨1, 9, 4, 6⟩
def rectangle_B : LabeledRectangle := ⟨0, 6, 1, 3⟩
def rectangle_C : LabeledRectangle := ⟨8, 2, 3, 5⟩
def rectangle_D : LabeledRectangle := ⟨5, 8, 7, 4⟩
def rectangle_E : LabeledRectangle := ⟨2, 0, 9, 7⟩

-- Define a function to check if a rectangle can be placed in position I
def can_be_placed_in_position_I (r : LabeledRectangle) : Prop :=
  ∃ (r2 r4 : LabeledRectangle), 
    r.right = r2.left ∧ r.bottom = r4.top

-- Theorem stating that only rectangle C can be placed in position I
theorem only_C_in_position_I : 
  can_be_placed_in_position_I rectangle_C ∧
  ¬can_be_placed_in_position_I rectangle_A ∧
  ¬can_be_placed_in_position_I rectangle_B ∧
  ¬can_be_placed_in_position_I rectangle_D ∧
  ¬can_be_placed_in_position_I rectangle_E :=
sorry

end NUMINAMATH_CALUDE_only_C_in_position_I_l1084_108497


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_exist_quadratic_with_two_roots_l1084_108496

/-- A quadratic equation x^2 + bx + c = 0 has two distinct real roots if and only if its discriminant is positive -/
theorem quadratic_two_distinct_roots (b c : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + b*x₁ + c = 0 ∧ x₂^2 + b*x₂ + c = 0 ↔ b^2 - 4*c > 0 := by
  sorry

/-- There exist real values b and c such that the quadratic equation x^2 + bx + c = 0 has two distinct real roots -/
theorem exist_quadratic_with_two_roots : ∃ (b c : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + b*x₁ + c = 0 ∧ x₂^2 + b*x₂ + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_exist_quadratic_with_two_roots_l1084_108496


namespace NUMINAMATH_CALUDE_time_to_see_again_l1084_108477

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a person walking -/
structure Walker where
  initialPosition : Point
  speed : ℝ

/-- The setup of the problem -/
def problemSetup : Prop := ∃ (sam kim : Walker) (tower : Point) (r : ℝ),
  -- Sam's initial position and speed
  sam.initialPosition = ⟨-100, -150⟩ ∧ sam.speed = 4 ∧
  -- Kim's initial position and speed
  kim.initialPosition = ⟨-100, 150⟩ ∧ kim.speed = 2 ∧
  -- Tower's position and radius
  tower = ⟨0, 0⟩ ∧ r = 100 ∧
  -- Initial distance between Sam and Kim
  (sam.initialPosition.x - kim.initialPosition.x)^2 + (sam.initialPosition.y - kim.initialPosition.y)^2 = 300^2

/-- The theorem to be proved -/
theorem time_to_see_again (setup : problemSetup) : 
  ∃ (t : ℝ), t = 240 ∧ 
  (∀ (t' : ℝ), t' < t → ∃ (x y : ℝ), 
    x^2 + y^2 = 100^2 ∧ 
    (y - (-150)) / (x - (-100 + 4 * t')) = (150 - (-150)) / (((-100 + 2 * t') - (-100 + 4 * t'))) ∧
    x * (150 - (-150)) = y * (((-100 + 2 * t') - (-100 + 4 * t'))))
  ∧ 
  (∃ (x y : ℝ), 
    x^2 + y^2 = 100^2 ∧ 
    (y - (-150)) / (x - (-100 + 4 * t)) = (150 - (-150)) / (((-100 + 2 * t) - (-100 + 4 * t))) ∧
    x * (150 - (-150)) = y * (((-100 + 2 * t) - (-100 + 4 * t)))) :=
by
  sorry


end NUMINAMATH_CALUDE_time_to_see_again_l1084_108477


namespace NUMINAMATH_CALUDE_subtraction_problem_l1084_108412

theorem subtraction_problem : 4444444444444 - 2222222222222 - 444444444444 = 1777777777778 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1084_108412


namespace NUMINAMATH_CALUDE_sum_of_xy_l1084_108485

theorem sum_of_xy (x y : ℕ+) (h : x + y + x * y = 54) : x + y = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l1084_108485


namespace NUMINAMATH_CALUDE_passing_marks_l1084_108409

/-- The passing marks problem -/
theorem passing_marks (T P : ℝ) 
  (h1 : 0.40 * T = P - 40)
  (h2 : 0.60 * T = P + 20)
  (h3 : 0.45 * T = P - 10) : 
  P = 160 := by sorry

end NUMINAMATH_CALUDE_passing_marks_l1084_108409


namespace NUMINAMATH_CALUDE_composition_difference_l1084_108462

/-- Given two functions f and g, prove that their composition difference
    equals a specific polynomial. -/
theorem composition_difference (x : ℝ) : 
  let f (x : ℝ) := 3 * x^2 + 4 * x - 5
  let g (x : ℝ) := 2 * x + 1
  (f (g x) - g (f x)) = 6 * x^2 + 12 * x + 11 := by
  sorry

end NUMINAMATH_CALUDE_composition_difference_l1084_108462


namespace NUMINAMATH_CALUDE_cylinder_cone_base_radii_equal_l1084_108466

/-- Given a cylinder and a cone with the same height and base radius, 
    if the ratio of their volumes is 3, then their base radii are equal -/
theorem cylinder_cone_base_radii_equal 
  (h : ℝ) -- height of both cylinder and cone
  (r_cylinder : ℝ) -- radius of cylinder base
  (r_cone : ℝ) -- radius of cone base
  (h_positive : h > 0)
  (r_cylinder_positive : r_cylinder > 0)
  (r_cone_positive : r_cone > 0)
  (same_radius : r_cylinder = r_cone)
  (volume_ratio : π * r_cylinder^2 * h / ((1/3) * π * r_cone^2 * h) = 3) :
  r_cylinder = r_cone :=
sorry

end NUMINAMATH_CALUDE_cylinder_cone_base_radii_equal_l1084_108466


namespace NUMINAMATH_CALUDE_orphanage_flowers_l1084_108478

theorem orphanage_flowers (flower_types : ℕ) (flowers_per_type : ℕ) : 
  flower_types = 4 → flowers_per_type = 40 → flower_types * flowers_per_type = 160 := by
  sorry

end NUMINAMATH_CALUDE_orphanage_flowers_l1084_108478


namespace NUMINAMATH_CALUDE_partnership_investment_l1084_108499

/-- A partnership problem where three partners invest different amounts and receive different shares. -/
theorem partnership_investment (a_investment b_investment c_investment : ℝ)
  (b_share a_share : ℝ) :
  b_investment = 11000 →
  c_investment = 18000 →
  b_share = 2200 →
  a_share = 1400 →
  (b_share / b_investment = a_share / a_investment) →
  a_investment = 7000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_l1084_108499


namespace NUMINAMATH_CALUDE_collins_savings_l1084_108413

/-- The amount earned per aluminum can in dollars -/
def earnings_per_can : ℚ := 25 / 100

/-- The number of cans found at home -/
def cans_at_home : ℕ := 12

/-- The number of cans found at grandparents' house -/
def cans_at_grandparents : ℕ := 3 * cans_at_home

/-- The number of cans given by the neighbor -/
def cans_from_neighbor : ℕ := 46

/-- The number of cans brought by dad from the office -/
def cans_from_dad : ℕ := 250

/-- The total number of cans collected -/
def total_cans : ℕ := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_dad

/-- The total earnings from recycling all cans -/
def total_earnings : ℚ := earnings_per_can * total_cans

/-- The amount Collin needs to put into savings -/
def savings_amount : ℚ := total_earnings / 2

/-- Theorem stating that the amount Collin needs to put into savings is $43.00 -/
theorem collins_savings : savings_amount = 43 := by sorry

end NUMINAMATH_CALUDE_collins_savings_l1084_108413


namespace NUMINAMATH_CALUDE_linear_function_value_l1084_108441

/-- A function that is linear in both arguments -/
def LinearInBoth (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ y₁ y₂ a b : ℝ, 
    f (a*x₁ + b*x₂) y₁ = a*(f x₁ y₁) + b*(f x₂ y₁) ∧
    f x₁ (a*y₁ + b*y₂) = a*(f x₁ y₁) + b*(f x₁ y₂)

/-- The main theorem -/
theorem linear_function_value (f : ℝ → ℝ → ℝ) 
  (h_linear : LinearInBoth f)
  (h_3_3 : f 3 3 = 1/(3*3))
  (h_3_4 : f 3 4 = 1/(3*4))
  (h_4_3 : f 4 3 = 1/(4*3))
  (h_4_4 : f 4 4 = 1/(4*4)) :
  f 5 5 = 1/36 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_value_l1084_108441


namespace NUMINAMATH_CALUDE_min_m_plus_n_l1084_108427

noncomputable def f (m n x : ℝ) : ℝ := Real.log x - 2 * m * x^2 - n

theorem min_m_plus_n (m n : ℝ) :
  (∀ x > 0, f m n x ≤ -Real.log 2) →
  (∃ x > 0, f m n x = -Real.log 2) →
  m + n ≥ (1/2) * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l1084_108427


namespace NUMINAMATH_CALUDE_complex_coordinates_of_product_l1084_108490

theorem complex_coordinates_of_product : 
  let z : ℂ := (2 - Complex.I) * (1 + Complex.I)
  Complex.re z = 3 ∧ Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_coordinates_of_product_l1084_108490


namespace NUMINAMATH_CALUDE_sock_pair_difference_sock_conditions_l1084_108495

/-- Represents the number of socks of a specific color -/
structure SockCount where
  red : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- Represents the sock collection of Joseph -/
def josephSocks : SockCount where
  red := 6
  blue := 12
  white := 8
  black := 2

theorem sock_pair_difference : 
  let blue_pairs := josephSocks.blue / 2
  let black_pairs := josephSocks.black / 2
  blue_pairs - black_pairs = 5 := by
  sorry

theorem sock_conditions : 
  -- Joseph has more pairs of blue socks than black socks
  josephSocks.blue > josephSocks.black ∧
  -- He has one less pair of red socks than white socks
  josephSocks.red / 2 + 1 = josephSocks.white / 2 ∧
  -- He has twice as many blue socks as red socks
  josephSocks.blue = 2 * josephSocks.red ∧
  -- He has 28 socks in total
  josephSocks.red + josephSocks.blue + josephSocks.white + josephSocks.black = 28 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_difference_sock_conditions_l1084_108495


namespace NUMINAMATH_CALUDE_real_roots_sum_product_l1084_108480

theorem real_roots_sum_product (c d : ℝ) : 
  (c^4 - 6*c + 3 = 0) → 
  (d^4 - 6*d + 3 = 0) → 
  (c ≠ d) →
  (c*d + c + d = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_real_roots_sum_product_l1084_108480


namespace NUMINAMATH_CALUDE_gcd_problem_l1084_108410

theorem gcd_problem (n : ℕ) : 
  75 ≤ n ∧ n ≤ 90 ∧ Nat.gcd n 15 = 5 → n = 80 ∨ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1084_108410


namespace NUMINAMATH_CALUDE_function_domain_l1084_108469

/-- The function y = √(x-1) / (x-2) is defined for x ≥ 1 and x ≠ 2 -/
theorem function_domain (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x - 1) / (x - 2)) ↔ (x ≥ 1 ∧ x ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_function_domain_l1084_108469


namespace NUMINAMATH_CALUDE_largest_absolute_value_l1084_108432

theorem largest_absolute_value : 
  let S : Finset ℤ := {4, -5, 0, -1}
  ∀ x ∈ S, |(-5 : ℤ)| ≥ |x| := by
  sorry

end NUMINAMATH_CALUDE_largest_absolute_value_l1084_108432


namespace NUMINAMATH_CALUDE_smallest_x_prime_factorization_l1084_108459

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3
def is_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^5

def satisfies_conditions (x : ℕ) : Prop :=
  is_square (2 * x) ∧ is_cube (3 * x) ∧ is_fifth_power (5 * x)

theorem smallest_x_prime_factorization :
  ∃ x : ℕ, 
    satisfies_conditions x ∧ 
    (∀ y : ℕ, satisfies_conditions y → x ≤ y) ∧
    x = 2^15 * 3^20 * 5^24 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_prime_factorization_l1084_108459


namespace NUMINAMATH_CALUDE_equation_solution_l1084_108443

theorem equation_solution :
  let S : Set ℂ := {x | (x - 4)^4 + (x - 6)^4 = 16}
  S = {5 + Complex.I * Real.sqrt 7, 5 - Complex.I * Real.sqrt 7, 6, 4} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1084_108443


namespace NUMINAMATH_CALUDE_vaccine_effectiveness_l1084_108458

-- Define the contingency table data
def a : ℕ := 10  -- Injected and Infected
def b : ℕ := 40  -- Injected and Not Infected
def c : ℕ := 20  -- Not Injected and Infected
def d : ℕ := 30  -- Not Injected and Not Infected
def n : ℕ := 100 -- Total number of observations

-- Define the K² formula
def K_squared : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the thresholds
def lower_threshold : ℚ := 3841 / 1000
def upper_threshold : ℚ := 5024 / 1000

-- Theorem statement
theorem vaccine_effectiveness :
  lower_threshold < K_squared ∧ K_squared < upper_threshold :=
sorry

end NUMINAMATH_CALUDE_vaccine_effectiveness_l1084_108458


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l1084_108420

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 33) : 
  (q1 + q2 + q3 + q4 + q5) / 5 = 15.6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l1084_108420


namespace NUMINAMATH_CALUDE_laptop_price_l1084_108446

theorem laptop_price (upfront_percentage : ℚ) (upfront_payment : ℚ) :
  upfront_percentage = 20 / 100 →
  upfront_payment = 240 →
  upfront_percentage * 1200 = upfront_payment :=
by sorry

end NUMINAMATH_CALUDE_laptop_price_l1084_108446


namespace NUMINAMATH_CALUDE_chips_probability_and_count_l1084_108406

def total_bags : ℕ := 9
def bbq_bags : ℕ := 5

def prob_three_bbq : ℚ := 10 / 84

theorem chips_probability_and_count :
  (total_bags = 9) →
  (bbq_bags = 5) →
  (prob_three_bbq = 10 / 84) →
  (Nat.choose bbq_bags 3 * Nat.choose (total_bags - bbq_bags) 0) / Nat.choose total_bags 3 = prob_three_bbq ∧
  total_bags - bbq_bags = 4 := by
  sorry

end NUMINAMATH_CALUDE_chips_probability_and_count_l1084_108406


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l1084_108498

theorem successive_discounts_equivalence :
  let original_price : ℝ := 50
  let first_discount : ℝ := 0.30
  let second_discount : ℝ := 0.15
  let equivalent_discount : ℝ := 0.405

  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)

  final_price = original_price * (1 - equivalent_discount) :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l1084_108498


namespace NUMINAMATH_CALUDE_root_implies_coefficients_l1084_108408

theorem root_implies_coefficients (p q : ℝ) : 
  (2 * (Complex.I * 2 - 3)^2 + p * (Complex.I * 2 - 3) + q = 0) → 
  (p = 12 ∧ q = 26) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_coefficients_l1084_108408


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1084_108467

theorem complex_fraction_equality : ((-1 : ℂ) + 3*I) / (1 + I) = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1084_108467


namespace NUMINAMATH_CALUDE_characterize_g_l1084_108494

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

-- State the theorem
theorem characterize_g :
  ∀ g : ℝ → ℝ, is_valid_g g ↔ (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_characterize_g_l1084_108494


namespace NUMINAMATH_CALUDE_equation_solution_l1084_108407

theorem equation_solution : ∃! x : ℝ, x + (x + 1) + (x + 2) + (x + 3) = 18 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1084_108407


namespace NUMINAMATH_CALUDE_charlie_age_when_jenny_twice_bobby_l1084_108400

theorem charlie_age_when_jenny_twice_bobby (jenny charlie bobby : ℕ) : 
  jenny = charlie + 5 → 
  charlie = bobby + 3 → 
  ∃ x : ℕ, jenny + x = 2 * (bobby + x) ∧ charlie + x = 11 :=
by sorry

end NUMINAMATH_CALUDE_charlie_age_when_jenny_twice_bobby_l1084_108400


namespace NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l1084_108473

/-- The maximum number of parts that three planes can divide space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts that three planes can divide space into is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l1084_108473


namespace NUMINAMATH_CALUDE_prob_limit_theorem_l1084_108445

/-- The probability that every boy chooses a different number than every girl
    when n boys and n girls choose numbers uniformly from {1, 2, 3, 4, 5} -/
def p (n : ℕ) : ℝ := sorry

/-- The limit of the nth root of p_n as n approaches infinity -/
def limit_p : ℝ := sorry

theorem prob_limit_theorem : 
  limit_p = 6 / 25 := by sorry

end NUMINAMATH_CALUDE_prob_limit_theorem_l1084_108445


namespace NUMINAMATH_CALUDE_modulus_of_3_minus_4i_l1084_108488

theorem modulus_of_3_minus_4i : Complex.abs (3 - 4*I) = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_3_minus_4i_l1084_108488


namespace NUMINAMATH_CALUDE_fraction_simplification_l1084_108437

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  (7 * a + 7 * b) / (a^2 - b^2) = 7 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1084_108437


namespace NUMINAMATH_CALUDE_clock_strike_times_l1084_108461

/-- Represents the time taken for a given number of clock strikes -/
def strike_time (n : ℕ) : ℚ :=
  (n - 1) * (10 : ℚ) / 9

/-- The clock takes 10 seconds to strike 10 times at 10:00 o'clock -/
axiom ten_strikes_time : strike_time 10 = 10

/-- The strikes are uniformly spaced -/
axiom uniform_strikes : ∀ (n m : ℕ), n > 0 → m > 0 → 
  strike_time n / (n - 1) = strike_time m / (m - 1)

theorem clock_strike_times :
  strike_time 8 = 70 / 9 ∧ strike_time 15 = 140 / 9 := by
  sorry

end NUMINAMATH_CALUDE_clock_strike_times_l1084_108461


namespace NUMINAMATH_CALUDE_equation_solutions_l1084_108416

/-- The set of solutions to the equation (x^3 + 3x^2√3 + 9x + 3√3) + (x + √3) = 0 -/
def solution_set : Set ℂ :=
  {z : ℂ | z = -Real.sqrt 3 ∨ z = -Real.sqrt 3 + Complex.I ∨ z = -Real.sqrt 3 - Complex.I}

/-- The equation (x^3 + 3x^2√3 + 9x + 3√3) + (x + √3) = 0 -/
def equation (x : ℂ) : Prop :=
  (x^3 + 3*x^2*Real.sqrt 3 + 9*x + 3*Real.sqrt 3) + (x + Real.sqrt 3) = 0

theorem equation_solutions :
  ∀ x : ℂ, equation x ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1084_108416


namespace NUMINAMATH_CALUDE_floor_sqrt_24_squared_l1084_108402

theorem floor_sqrt_24_squared : ⌊Real.sqrt 24⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_24_squared_l1084_108402


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l1084_108423

/-- A function satisfying the given functional equation is either constantly zero or f(x) = x - 1. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (x - 2) * f y + f (y + 2 * f x) = f (x + y * f x)) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l1084_108423


namespace NUMINAMATH_CALUDE_add_decimals_l1084_108422

theorem add_decimals : 5.47 + 4.26 = 9.73 := by
  sorry

end NUMINAMATH_CALUDE_add_decimals_l1084_108422


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_l1084_108449

theorem roots_sum_reciprocal (x₁ x₂ : ℝ) : 
  (5 * x₁^2 - 3 * x₁ - 2 = 0) → 
  (5 * x₂^2 - 3 * x₂ - 2 = 0) → 
  x₁ ≠ x₂ →
  (1 / x₁ + 1 / x₂ = -3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_l1084_108449


namespace NUMINAMATH_CALUDE_students_not_in_biology_l1084_108457

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : biology_percentage = 325 / 1000) :
  total_students - (total_students * biology_percentage).floor = 594 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l1084_108457


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l1084_108484

/-- A parabola is tangent to a line if they intersect at exactly one point. -/
def is_tangent (a : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 + 10 = 2 * x

/-- The value of a for which the parabola y = ax^2 + 10 is tangent to the line y = 2x -/
theorem parabola_tangent_line : 
  ∃ a : ℝ, is_tangent a ∧ a = (1 : ℝ) / 10 :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l1084_108484


namespace NUMINAMATH_CALUDE_monkey_travel_distance_l1084_108455

/-- Represents the speed and time of movement for a monkey --/
structure MonkeyMovement where
  swingingSpeed : ℝ
  runningSpeed : ℝ
  runningTime : ℝ
  swingingTime : ℝ

/-- Calculates the total distance traveled by the monkey --/
def totalDistance (m : MonkeyMovement) : ℝ :=
  m.runningSpeed * m.runningTime + m.swingingSpeed * m.swingingTime

/-- Theorem stating the total distance traveled by the monkey --/
theorem monkey_travel_distance :
  ∀ (m : MonkeyMovement),
  m.swingingSpeed = 10 ∧
  m.runningSpeed = 15 ∧
  m.runningTime = 5 ∧
  m.swingingTime = 10 →
  totalDistance m = 175 := by
  sorry

end NUMINAMATH_CALUDE_monkey_travel_distance_l1084_108455


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l1084_108465

theorem smallest_factor_for_perfect_square (n : ℕ) (h : n = 31360) : 
  (∃ (y : ℕ), y > 0 ∧ ∃ (k : ℕ), n * y = k^2) ∧ 
  (∀ (z : ℕ), z > 0 → z < 623 → ¬∃ (k : ℕ), n * z = k^2) ∧
  (∃ (k : ℕ), n * 623 = k^2) := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l1084_108465


namespace NUMINAMATH_CALUDE_committee_probability_l1084_108417

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

def probability_at_least_two_of_each : ℚ :=
  1 - (Nat.choose num_girls committee_size +
       num_boys * Nat.choose num_girls (committee_size - 1) +
       Nat.choose num_boys committee_size +
       num_girls * Nat.choose num_boys (committee_size - 1)) /
      Nat.choose total_members committee_size

theorem committee_probability :
  probability_at_least_two_of_each = 457215 / 593775 :=
by sorry

end NUMINAMATH_CALUDE_committee_probability_l1084_108417


namespace NUMINAMATH_CALUDE_power_function_domain_and_oddness_l1084_108447

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_domain_and_oddness (a : ℤ) :
  a ∈ ({-1, 1, 3} : Set ℤ) →
  (∀ x : ℝ, ∃ y : ℝ, y = x^a) ∧ is_odd_function (λ x : ℝ ↦ x^a) ↔
  a ∈ ({1, 3} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_power_function_domain_and_oddness_l1084_108447


namespace NUMINAMATH_CALUDE_largest_fraction_l1084_108460

def fraction_set : Set ℚ := {1/2, 1/3, 1/4, 1/5, 1/10}

theorem largest_fraction :
  ∀ x ∈ fraction_set, (1/2 : ℚ) ≥ x :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l1084_108460


namespace NUMINAMATH_CALUDE_points_on_line_relationship_l1084_108489

/-- Given a line y = -3x + b and three points on this line, prove that y₁ > y₂ > y₃ -/
theorem points_on_line_relationship (b : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -3 * (-2) + b)
  (h2 : y₂ = -3 * (-1) + b)
  (h3 : y₃ = -3 * 1 + b) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_relationship_l1084_108489


namespace NUMINAMATH_CALUDE_number_difference_l1084_108439

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21352)
  (b_div_9 : ∃ k, b = 9 * k)
  (relation : 10 * a + 1 = b) : 
  b - a = 17470 := by sorry

end NUMINAMATH_CALUDE_number_difference_l1084_108439


namespace NUMINAMATH_CALUDE_undefined_expression_l1084_108415

theorem undefined_expression (x : ℝ) : 
  (x^2 - 18*x + 81 = 0) ↔ (x = 9) := by
  sorry

#check undefined_expression

end NUMINAMATH_CALUDE_undefined_expression_l1084_108415


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_not_l1084_108448

theorem negation_of_existence_is_forall_not :
  (¬ ∃ x : ℚ, x^2 - 2 = 0) ↔ (∀ x : ℚ, x^2 - 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_not_l1084_108448


namespace NUMINAMATH_CALUDE_inequality_theorem_l1084_108430

theorem inequality_theorem (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_product : a * b * c * d = 1) :
  1 / (1 + a)^2 + 1 / (1 + b)^2 + 1 / (1 + c)^2 + 1 / (1 + d)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1084_108430


namespace NUMINAMATH_CALUDE_smallest_possible_value_l1084_108451

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 13 * a (n - 1) - 2 * n

/-- The sequence is positive -/
def PositiveSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

theorem smallest_possible_value (a : ℕ → ℝ) 
    (h_recurrence : RecurrenceSequence a) 
    (h_positive : PositiveSequence a) :
    (∀ a₁ : ℝ, a 1 ≥ a₁ → a₁ ≥ 13/36) :=
  sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l1084_108451


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1084_108482

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-3) 2 = {x : ℝ | a * x^2 - 5*x + b > 0}) : 
  {x : ℝ | b * x^2 - 5*x + a > 0} = Set.Iic (-1/3) ∪ Set.Ici (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1084_108482


namespace NUMINAMATH_CALUDE_rice_purchase_comparison_l1084_108471

theorem rice_purchase_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (200 / (100 / a + 100 / b)) ≤ ((100 * a + 100 * b) / 200) := by
  sorry

#check rice_purchase_comparison

end NUMINAMATH_CALUDE_rice_purchase_comparison_l1084_108471


namespace NUMINAMATH_CALUDE_jason_newspaper_earnings_l1084_108483

-- Define the initial and final amounts for Jason
def jason_initial : ℕ := 3
def jason_final : ℕ := 63

-- Define Jason's earnings
def jason_earnings : ℕ := jason_final - jason_initial

-- Theorem to prove
theorem jason_newspaper_earnings :
  jason_earnings = 60 := by
  sorry

end NUMINAMATH_CALUDE_jason_newspaper_earnings_l1084_108483


namespace NUMINAMATH_CALUDE_min_value_of_function_l1084_108474

theorem min_value_of_function (x : ℝ) : |3 - x| + |x - 7| ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1084_108474


namespace NUMINAMATH_CALUDE_storage_box_length_l1084_108426

/-- Calculates the length of cubic storage boxes given total volume, cost per box, and total cost -/
theorem storage_box_length (total_volume : ℝ) (cost_per_box : ℝ) (total_cost : ℝ) :
  total_volume = 1080000 ∧ cost_per_box = 0.5 ∧ total_cost = 300 →
  ∃ (length : ℝ), abs (length - (total_volume / (total_cost / cost_per_box))^(1/3)) < 0.1 := by
sorry

#eval (1080000 / (300 / 0.5))^(1/3)

end NUMINAMATH_CALUDE_storage_box_length_l1084_108426


namespace NUMINAMATH_CALUDE_johns_house_wall_planks_l1084_108414

/-- The number of planks needed for a house wall --/
def total_planks (large_planks small_planks : ℕ) : ℕ :=
  large_planks + small_planks

/-- Theorem stating the total number of planks needed for John's house wall --/
theorem johns_house_wall_planks :
  total_planks 37 42 = 79 := by
  sorry

end NUMINAMATH_CALUDE_johns_house_wall_planks_l1084_108414


namespace NUMINAMATH_CALUDE_string_average_length_l1084_108436

theorem string_average_length (s1 s2 s3 : ℝ) 
  (h1 : s1 = 2) (h2 : s2 = 3) (h3 : s3 = 7) : 
  (s1 + s2 + s3) / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_string_average_length_l1084_108436


namespace NUMINAMATH_CALUDE_power_23_mod_25_l1084_108438

theorem power_23_mod_25 : 23^2057 % 25 = 16 := by sorry

end NUMINAMATH_CALUDE_power_23_mod_25_l1084_108438


namespace NUMINAMATH_CALUDE_marys_investment_l1084_108401

theorem marys_investment (mary_investment : ℕ) (mike_investment : ℕ) (total_profit : ℕ) : 
  mike_investment = 400 →
  total_profit = 7500 →
  let equal_share := total_profit / 3 / 2
  let remaining_profit := total_profit - total_profit / 3
  let mary_share := equal_share + remaining_profit * mary_investment / (mary_investment + mike_investment)
  let mike_share := equal_share + remaining_profit * mike_investment / (mary_investment + mike_investment)
  mary_share = mike_share + 1000 →
  mary_investment = 600 :=
by sorry

end NUMINAMATH_CALUDE_marys_investment_l1084_108401


namespace NUMINAMATH_CALUDE_max_fruits_bought_l1084_108452

/-- Represents the cost of each fruit in RM -/
structure FruitCost where
  apple : ℕ
  mango : ℕ
  papaya : ℕ

/-- Represents the number of each fruit bought -/
structure FruitCount where
  apple : ℕ
  mango : ℕ
  papaya : ℕ

/-- Calculates the total cost of fruits bought -/
def totalCost (cost : FruitCost) (count : FruitCount) : ℕ :=
  cost.apple * count.apple + cost.mango * count.mango + cost.papaya * count.papaya

/-- Calculates the total number of fruits bought -/
def totalFruits (count : FruitCount) : ℕ :=
  count.apple + count.mango + count.papaya

/-- Theorem stating the maximum number of fruits that can be bought under given conditions -/
theorem max_fruits_bought (cost : FruitCost) (count : FruitCount) 
    (h_apple_cost : cost.apple = 3)
    (h_mango_cost : cost.mango = 4)
    (h_papaya_cost : cost.papaya = 5)
    (h_at_least_one : count.apple ≥ 1 ∧ count.mango ≥ 1 ∧ count.papaya ≥ 1)
    (h_total_cost : totalCost cost count = 50) :
    totalFruits count ≤ 15 ∧ ∃ (max_count : FruitCount), totalFruits max_count = 15 ∧ totalCost cost max_count = 50 :=
  sorry


end NUMINAMATH_CALUDE_max_fruits_bought_l1084_108452


namespace NUMINAMATH_CALUDE_intersection_point_exists_l1084_108444

/-- Square with side length 6 -/
def square_side : ℝ := 6

/-- Point P on side AB -/
def P : ℝ × ℝ := (3, square_side)

/-- Point D at origin -/
def D : ℝ × ℝ := (0, 0)

/-- Radius of circle centered at P -/
def r_P : ℝ := 3

/-- Radius of circle centered at D -/
def r_D : ℝ := 5

/-- Definition of circle centered at P -/
def circle_P (x y : ℝ) : Prop :=
  (x - P.1)^2 + (y - P.2)^2 = r_P^2

/-- Definition of circle centered at D -/
def circle_D (x y : ℝ) : Prop :=
  x^2 + y^2 = r_D^2

/-- Theorem stating the existence of intersection point Q and its distance from BC -/
theorem intersection_point_exists : ∃ Q : ℝ × ℝ,
  circle_P Q.1 Q.2 ∧ circle_D Q.1 Q.2 ∧ 
  (∃ d : ℝ, d = Q.2 ∧ d ≥ 0 ∧ d ≤ square_side) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_exists_l1084_108444


namespace NUMINAMATH_CALUDE_population_change_l1084_108429

theorem population_change (P : ℝ) : 
  P * 1.12 * 0.88 = 14784 → P = 15000 := by
  sorry

end NUMINAMATH_CALUDE_population_change_l1084_108429


namespace NUMINAMATH_CALUDE_intersection_range_l1084_108431

theorem intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    x₁^2 - y₁^2 = 6 ∧ 
    x₂^2 - y₂^2 = 6 ∧ 
    y₁ = k * x₁ + 2 ∧ 
    y₂ = k * x₂ + 2 ∧ 
    x₁ > 0 ∧ 
    x₂ > 0) → 
  -Real.sqrt 15 / 3 < k ∧ k < -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_range_l1084_108431


namespace NUMINAMATH_CALUDE_factorial_equation_l1084_108404

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem stating the equality of 7! * 11! and 15 * 12! -/
theorem factorial_equation : factorial 7 * factorial 11 = 15 * factorial 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l1084_108404


namespace NUMINAMATH_CALUDE_alcohol_concentration_l1084_108464

theorem alcohol_concentration (original_volume : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  original_volume = 9 →
  added_water = 3 →
  final_concentration = 42.75 →
  (original_volume * (57 / 100)) = ((original_volume + added_water) * (final_concentration / 100)) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_l1084_108464


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l1084_108493

/-- Proves that given a simple interest of 4016.25, an interest rate of 10% per annum,
    and a time period of 5 years, the principal sum is 8032.5. -/
theorem simple_interest_principal_calculation :
  let simple_interest : ℝ := 4016.25
  let rate : ℝ := 10  -- 10% per annum
  let time : ℝ := 5   -- 5 years
  let principal : ℝ := simple_interest * 100 / (rate * time)
  principal = 8032.5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l1084_108493


namespace NUMINAMATH_CALUDE_circle_tangent_to_directrix_l1084_108487

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (2, 0)

-- Define the right directrix of the hyperbola
def right_directrix (x : ℝ) : Prop := x = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Theorem statement
theorem circle_tangent_to_directrix :
  ∀ x y : ℝ,
  hyperbola x y →
  circle_equation x y ↔
    (∃ (cx cy : ℝ), (cx, cy) = right_focus ∧
      ∀ (dx : ℝ), right_directrix dx →
        (x - cx)^2 + (y - cy)^2 = (x - dx)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_directrix_l1084_108487


namespace NUMINAMATH_CALUDE_vacation_speed_problem_l1084_108419

theorem vacation_speed_problem (distance1 distance2 time_diff : ℝ) 
  (h1 : distance1 = 100)
  (h2 : distance2 = 175)
  (h3 : time_diff = 3)
  (h4 : distance2 / speed = distance1 / speed + time_diff)
  (speed : ℝ) :
  speed = 25 := by
sorry

end NUMINAMATH_CALUDE_vacation_speed_problem_l1084_108419


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1084_108476

theorem polynomial_expansion (x : ℝ) : 
  (5 * x + 3) * (7 * x^2 + 2 * x + 4) = 35 * x^3 + 31 * x^2 + 26 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1084_108476


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l1084_108424

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l1084_108424


namespace NUMINAMATH_CALUDE_count_parallelepipeds_l1084_108418

/-- The number of parallelepipeds formed in a rectangular parallelepiped -/
def num_parallelepipeds (m n k : ℕ) : ℚ :=
  (m * n * k * (m + 1) * (n + 1) * (k + 1) : ℚ) / 8

/-- Theorem: The number of parallelepipeds formed in a rectangular parallelepiped
    with dimensions m × n × k, divided into unit cubes, is equal to
    (m * n * k * (m+1) * (n+1) * (k+1)) / 8 -/
theorem count_parallelepipeds (m n k : ℕ) :
  num_parallelepipeds m n k = (m * n * k * (m + 1) * (n + 1) * (k + 1) : ℚ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_count_parallelepipeds_l1084_108418


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1084_108425

/-- Given a family with the following properties:
  * The total number of children is 180
  * Boys are given $3900 to share
  * Each boy receives $52
  Prove that the ratio of boys to girls is 5:7 -/
theorem boys_to_girls_ratio (total_children : ℕ) (boys_money : ℕ) (boy_share : ℕ)
  (h_total : total_children = 180)
  (h_money : boys_money = 3900)
  (h_share : boy_share = 52)
  : ∃ (boys girls : ℕ), boys + girls = total_children ∧ 
    boys * boy_share = boys_money ∧
    boys * 7 = girls * 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1084_108425


namespace NUMINAMATH_CALUDE_largest_non_representable_largest_non_representable_proof_l1084_108486

def is_representable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 5 * a + 8 * b + 12 * c

theorem largest_non_representable : ℕ :=
  19

theorem largest_non_representable_proof :
  (¬ is_representable largest_non_representable) ∧
  (∀ m : ℕ, m > largest_non_representable → is_representable m) :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_largest_non_representable_proof_l1084_108486


namespace NUMINAMATH_CALUDE_cashew_price_l1084_108492

/-- Proves the price of cashew nuts given the conditions of the problem -/
theorem cashew_price (peanut_price : ℕ) (cashew_amount peanut_amount total_amount : ℕ) (total_price : ℕ) :
  peanut_price = 130 →
  cashew_amount = 3 →
  peanut_amount = 2 →
  total_amount = 5 →
  total_price = 178 →
  cashew_amount * (total_price * total_amount - peanut_price * peanut_amount) / cashew_amount = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_cashew_price_l1084_108492


namespace NUMINAMATH_CALUDE_max_value_sin_cos_product_l1084_108442

theorem max_value_sin_cos_product (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 4.5 ∧
  ∃ x y z : ℝ, (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
              (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_product_l1084_108442


namespace NUMINAMATH_CALUDE_smallest_k_for_four_color_rectangle_l1084_108456

/-- Represents a coloring of an n × n board -/
def Coloring (n : ℕ) (k : ℕ) := Fin n → Fin n → Fin k

/-- Predicate that checks if four cells form a rectangle with different colors -/
def hasFourColorRectangle (n : ℕ) (k : ℕ) (c : Coloring n k) : Prop :=
  ∃ (r1 r2 c1 c2 : Fin n), r1 ≠ r2 ∧ c1 ≠ c2 ∧
    c r1 c1 ≠ c r1 c2 ∧ c r1 c1 ≠ c r2 c1 ∧ c r1 c1 ≠ c r2 c2 ∧
    c r1 c2 ≠ c r2 c1 ∧ c r1 c2 ≠ c r2 c2 ∧
    c r2 c1 ≠ c r2 c2

/-- Main theorem stating the smallest k that guarantees a four-color rectangle -/
theorem smallest_k_for_four_color_rectangle (n : ℕ) (h : n ≥ 2) :
  (∀ k : ℕ, k ≥ 2*n → ∀ c : Coloring n k, hasFourColorRectangle n k c) ∧
  (∃ c : Coloring n (2*n - 1), ¬hasFourColorRectangle n (2*n - 1) c) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_four_color_rectangle_l1084_108456


namespace NUMINAMATH_CALUDE_soccer_league_games_l1084_108481

/-- The number of games played in a soccer league where each team plays every other team once -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a soccer league with 10 teams, where each team plays every other team once, 
    the total number of games played is 45 -/
theorem soccer_league_games : num_games 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1084_108481


namespace NUMINAMATH_CALUDE_julio_lost_fish_l1084_108411

/-- Proves that Julio lost 15 fish given the fishing conditions -/
theorem julio_lost_fish (fish_per_hour : ℕ) (fishing_hours : ℕ) (final_fish_count : ℕ) : 
  fish_per_hour = 7 →
  fishing_hours = 9 →
  final_fish_count = 48 →
  fish_per_hour * fishing_hours - final_fish_count = 15 := by
sorry

end NUMINAMATH_CALUDE_julio_lost_fish_l1084_108411
