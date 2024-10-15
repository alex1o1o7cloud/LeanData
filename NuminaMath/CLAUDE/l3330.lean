import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_for_3003_terms_l3330_333067

theorem smallest_n_for_3003_terms : ∃ (N : ℕ), 
  (N = 19) ∧ 
  (∀ k < N, (Nat.choose (k + 1) 5) < 3003) ∧
  (Nat.choose (N + 1) 5 = 3003) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_3003_terms_l3330_333067


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l3330_333004

theorem cubic_expansion_sum (x a₀ a₁ a₂ a₃ : ℝ) 
  (h : x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) : 
  a₁ + a₂ + a₃ = 19 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l3330_333004


namespace NUMINAMATH_CALUDE_circle_placement_theorem_l3330_333002

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with given side length -/
structure Square where
  sideLength : ℝ

/-- Represents a circle with given diameter -/
structure Circle where
  diameter : ℝ

/-- Theorem: In a 20x25 rectangle with 120 unit squares, there exists a point for a circle with diameter 1 -/
theorem circle_placement_theorem (rect : Rectangle) (squares : Finset Square) (circ : Circle) :
  rect.width = 20 ∧ rect.height = 25 ∧
  squares.card = 120 ∧ (∀ s ∈ squares, s.sideLength = 1) ∧
  circ.diameter = 1 →
  ∃ (center : ℝ × ℝ),
    (center.1 ≥ 0 ∧ center.1 ≤ rect.width ∧ center.2 ≥ 0 ∧ center.2 ≤ rect.height) ∧
    (∀ s ∈ squares, ∀ (point : ℝ × ℝ),
      (point.1 - center.1)^2 + (point.2 - center.2)^2 ≤ (circ.diameter / 2)^2 →
      ¬(point.1 ≥ s.sideLength ∧ point.1 ≤ s.sideLength + 1 ∧
        point.2 ≥ s.sideLength ∧ point.2 ≤ s.sideLength + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_placement_theorem_l3330_333002


namespace NUMINAMATH_CALUDE_equality_and_inequality_of_exponents_l3330_333096

theorem equality_and_inequality_of_exponents (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (2 : ℝ)^x = (3 : ℝ)^y ∧ (3 : ℝ)^y = (4 : ℝ)^z) :
  2 * x = 4 * z ∧ 2 * x > 3 * y :=
by sorry

end NUMINAMATH_CALUDE_equality_and_inequality_of_exponents_l3330_333096


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3330_333009

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (totalPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  totalPopulation / sampleSize

/-- Theorem: The systematic sampling interval for 1000 students with a sample size of 50 is 20 -/
theorem systematic_sampling_interval_example :
  systematicSamplingInterval 1000 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3330_333009


namespace NUMINAMATH_CALUDE_expression_evaluation_l3330_333079

theorem expression_evaluation : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3330_333079


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3330_333094

/-- The equation of the latus rectum of the parabola y = -1/4 * x^2 is y = 1 -/
theorem latus_rectum_of_parabola :
  ∀ (x y : ℝ), y = -(1/4) * x^2 → (∃ (x₀ : ℝ), y = 1 ∧ x₀^2 = -4*y) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3330_333094


namespace NUMINAMATH_CALUDE_shaded_rectangle_probability_l3330_333028

theorem shaded_rectangle_probability : 
  let width : ℕ := 2004
  let height : ℕ := 2
  let shaded_pos1 : ℕ := 501
  let shaded_pos2 : ℕ := 1504
  let total_rectangles : ℕ := height * (width.choose 2)
  let shaded_rectangles_per_row : ℕ := 
    shaded_pos1 * (width - shaded_pos1 + 1) + 
    (shaded_pos2 - shaded_pos1) * (width - shaded_pos2 + 1)
  let total_shaded_rectangles : ℕ := height * shaded_rectangles_per_row
  (total_rectangles - total_shaded_rectangles : ℚ) / total_rectangles = 1501 / 4008 :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_rectangle_probability_l3330_333028


namespace NUMINAMATH_CALUDE_jessica_age_when_justin_born_jessica_age_proof_l3330_333030

theorem jessica_age_when_justin_born (justin_current_age : ℕ) (james_jessica_age_diff : ℕ) (james_future_age : ℕ) (years_to_future : ℕ) : ℕ :=
  let james_current_age := james_future_age - years_to_future
  let jessica_current_age := james_current_age - james_jessica_age_diff
  jessica_current_age - justin_current_age

/- Proof that Jessica was 6 years old when Justin was born -/
theorem jessica_age_proof :
  jessica_age_when_justin_born 26 7 44 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_jessica_age_when_justin_born_jessica_age_proof_l3330_333030


namespace NUMINAMATH_CALUDE_divisor_problem_l3330_333006

theorem divisor_problem (original : Nat) (subtracted : Nat) (divisor : Nat) : 
  original = 427398 →
  subtracted = 8 →
  divisor = 10 →
  (original - subtracted) % divisor = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3330_333006


namespace NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l3330_333097

/-- A trapezoid with a 60° angle that has both inscribed and circumscribed circles -/
structure SpecialTrapezoid where
  /-- The measure of one angle of the trapezoid in degrees -/
  angle : ℝ
  /-- The trapezoid has an inscribed circle -/
  has_inscribed_circle : Prop
  /-- The trapezoid has a circumscribed circle -/
  has_circumscribed_circle : Prop
  /-- The angle measure is 60° -/
  angle_is_60 : angle = 60

/-- The ratio of the bases of the special trapezoid -/
def base_ratio (t : SpecialTrapezoid) : ℝ × ℝ :=
  (1, 3)

/-- Theorem: The ratio of the bases of a special trapezoid is 1:3 -/
theorem special_trapezoid_base_ratio (t : SpecialTrapezoid) :
  base_ratio t = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l3330_333097


namespace NUMINAMATH_CALUDE_three_inverse_mod_191_l3330_333060

theorem three_inverse_mod_191 : ∃ x : ℕ, x < 191 ∧ (3 * x) % 191 = 1 ∧ x = 64 := by sorry

end NUMINAMATH_CALUDE_three_inverse_mod_191_l3330_333060


namespace NUMINAMATH_CALUDE_meeting_arrangements_l3330_333053

def num_schools : ℕ := 3
def members_per_school : ℕ := 6
def host_representatives : ℕ := 3
def other_representatives : ℕ := 1

def arrange_meeting : ℕ := 
  num_schools * (members_per_school.choose host_representatives) * 
  ((members_per_school.choose other_representatives) ^ (num_schools - 1))

theorem meeting_arrangements :
  arrange_meeting = 2160 := by
  sorry

end NUMINAMATH_CALUDE_meeting_arrangements_l3330_333053


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3330_333047

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 1 - 3 * Complex.I) : 
  z.im = -2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3330_333047


namespace NUMINAMATH_CALUDE_min_value_of_f_l3330_333038

def f (x : ℝ) : ℝ := x^4 + 2*x^2 - 1

theorem min_value_of_f :
  ∃ (m : ℝ), m = -1 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3330_333038


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l3330_333029

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Define the largest prime factor function
def largest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_sum_of_divisors_450 :
  largest_prime_factor (sum_of_divisors 450) = 31 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l3330_333029


namespace NUMINAMATH_CALUDE_garden_area_l3330_333084

theorem garden_area (total_posts : ℕ) (post_spacing : ℕ) (longer_side_ratio : ℕ) : 
  total_posts = 20 →
  post_spacing = 4 →
  longer_side_ratio = 2 →
  ∃ (short_side long_side : ℕ),
    short_side * long_side = 336 ∧
    short_side * longer_side_ratio = long_side ∧
    short_side * post_spacing = (short_side - 1) * post_spacing ∧
    long_side * post_spacing = (long_side - 1) * post_spacing ∧
    2 * (short_side + long_side) - 4 = total_posts :=
by sorry

end NUMINAMATH_CALUDE_garden_area_l3330_333084


namespace NUMINAMATH_CALUDE_tangent_line_condition_no_positive_max_for_negative_integer_a_l3330_333049

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * Real.exp x) / x + x

theorem tangent_line_condition (a : ℝ) :
  (∃ (m b : ℝ), m * 1 + b = f a 1 ∧ m * (1 - 0) = f a 1 - (-1) ∧ 0 * m + b = -1) →
  a = -1 / Real.exp 1 := by
  sorry

theorem no_positive_max_for_negative_integer_a :
  ∀ a : ℤ, a < 0 →
  ¬∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a x ≥ f a y ∧ f a x > 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_condition_no_positive_max_for_negative_integer_a_l3330_333049


namespace NUMINAMATH_CALUDE_math_competition_prizes_l3330_333075

theorem math_competition_prizes (x y s : ℝ) 
  (h1 : 100 * (x + 3 * y) = s)
  (h2 : 80 * (x + 5 * y) = s) :
  x = 5 * y ∧ s = 160 * x ∧ s = 800 * y := by
  sorry

end NUMINAMATH_CALUDE_math_competition_prizes_l3330_333075


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3330_333088

theorem solve_exponential_equation :
  ∃ x : ℝ, 5^(3*x) = Real.sqrt 125 ∧ x = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3330_333088


namespace NUMINAMATH_CALUDE_december_savings_l3330_333011

def savings_plan (initial_amount : ℕ) (months : ℕ) : ℕ :=
  (initial_amount : ℕ) * (3 ^ (months - 1))

theorem december_savings :
  savings_plan 10 12 = 1771470 := by
  sorry

end NUMINAMATH_CALUDE_december_savings_l3330_333011


namespace NUMINAMATH_CALUDE_automotive_test_distance_l3330_333081

theorem automotive_test_distance (d : ℝ) (h1 : d / 4 + d / 5 + d / 6 = 37) : 3 * d = 180 := by
  sorry

#check automotive_test_distance

end NUMINAMATH_CALUDE_automotive_test_distance_l3330_333081


namespace NUMINAMATH_CALUDE_birth_interval_proof_l3330_333087

/-- Proves that the interval between births is 2 years given the conditions of the problem -/
theorem birth_interval_proof (num_children : ℕ) (youngest_age : ℕ) (total_age : ℕ) :
  num_children = 5 →
  youngest_age = 7 →
  total_age = 55 →
  (∃ interval : ℕ,
    total_age = youngest_age * num_children + interval * (num_children * (num_children - 1)) / 2 ∧
    interval = 2) := by
  sorry

end NUMINAMATH_CALUDE_birth_interval_proof_l3330_333087


namespace NUMINAMATH_CALUDE_license_plate_count_l3330_333045

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 3

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 3

/-- The number of possible positions for the letter block -/
def letter_block_positions : ℕ := 4

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * (num_digits ^ digits_in_plate) * (num_letters ^ letters_in_plate)

theorem license_plate_count : total_license_plates = 70304000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3330_333045


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_edge_distance_l3330_333022

/-- The volume of a cube, given the distance from its space diagonal to a non-intersecting edge. -/
theorem cube_volume_from_diagonal_edge_distance (d : ℝ) (d_pos : 0 < d) : 
  ∃ (V : ℝ), V = 2 * d^3 * Real.sqrt 2 ∧ 
  (∃ (a : ℝ), a > 0 ∧ a = d * Real.sqrt 2 ∧ V = a^3) :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_edge_distance_l3330_333022


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3330_333012

theorem simple_interest_problem (P R T : ℝ) : 
  P = 300 →
  P * (R + 6) / 100 * T = P * R / 100 * T + 90 →
  T = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3330_333012


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3330_333018

theorem arithmetic_calculation : 12 * 11 + 7 * 8 - 5 * 6 + 10 * 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3330_333018


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3330_333013

/-- Given a = -1 and ab = 2, prove that 3(2a²b + ab²) - (3ab² - a²b) evaluates to -14 -/
theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -1) (h2 : a * b = 2) :
  3 * (2 * a^2 * b + a * b^2) - (3 * a * b^2 - a^2 * b) = -14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3330_333013


namespace NUMINAMATH_CALUDE_square_sum_equals_25_l3330_333040

theorem square_sum_equals_25 (x y : ℝ) 
  (h1 : y + 6 = (x - 3)^2) 
  (h2 : x + 6 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_25_l3330_333040


namespace NUMINAMATH_CALUDE_triangle_lines_theorem_l3330_333000

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  medianCM : ℝ → ℝ → ℝ
  altitudeBH : ℝ → ℝ → ℝ

/-- The given triangle satisfies the problem conditions -/
def given_triangle : Triangle where
  A := (5, 1)
  medianCM := λ x y => 2*x - y - 5
  altitudeBH := λ x y => x - 2*y - 5

/-- The equation of line BC -/
def line_BC (t : Triangle) : ℝ → ℝ → ℝ :=
  λ x y => 6*x - 5*y - 9

/-- The equation of the line symmetric to BC with respect to CM -/
def symmetric_line_BC (t : Triangle) : ℝ → ℝ → ℝ :=
  λ x y => 38*x - 9*y - 125

/-- Main theorem proving the equations of lines -/
theorem triangle_lines_theorem (t : Triangle) (h : t = given_triangle) :
  (line_BC t = λ x y => 6*x - 5*y - 9) ∧
  (symmetric_line_BC t = λ x y => 38*x - 9*y - 125) := by
  sorry

end NUMINAMATH_CALUDE_triangle_lines_theorem_l3330_333000


namespace NUMINAMATH_CALUDE_line_growth_limit_l3330_333061

theorem line_growth_limit :
  let initial_length : ℝ := 2
  let growth_series (n : ℕ) : ℝ := (1 / 3^n) * (1 + Real.sqrt 3)
  (initial_length + ∑' n, growth_series n) = (6 + Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_line_growth_limit_l3330_333061


namespace NUMINAMATH_CALUDE_additive_increasing_nonneg_implies_odd_increasing_l3330_333032

def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂

def is_increasing_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ x₂ → x₂ ≥ 0 → f x₁ ≥ f x₂

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ x₂ → f x₁ ≥ f x₂

theorem additive_increasing_nonneg_implies_odd_increasing
  (f : ℝ → ℝ) (h1 : is_additive f) (h2 : is_increasing_nonneg f) :
  is_odd f ∧ is_increasing f :=
sorry

end NUMINAMATH_CALUDE_additive_increasing_nonneg_implies_odd_increasing_l3330_333032


namespace NUMINAMATH_CALUDE_f_derivative_f_initial_condition_range_of_x_l3330_333050

/-- A function f with the given properties -/
def f : ℝ → ℝ :=
  sorry

theorem f_derivative (x : ℝ) : deriv f x = 5 + Real.cos x :=
  sorry

theorem f_initial_condition : f 0 = 0 :=
  sorry

theorem range_of_x (x : ℝ) :
  f (1 - x) + f (1 - x^2) < 0 ↔ x ∈ Set.Iic (-2) ∪ Set.Ioi 1 :=
  sorry

end NUMINAMATH_CALUDE_f_derivative_f_initial_condition_range_of_x_l3330_333050


namespace NUMINAMATH_CALUDE_otimes_four_two_l3330_333064

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem otimes_four_two : otimes 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_two_l3330_333064


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3330_333021

/-- The eccentricity of a hyperbola with equation x²/2 - y² = -1 is √3 -/
theorem hyperbola_eccentricity : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/2 - y^2 = -1
  ∃ e : ℝ, e = Real.sqrt 3 ∧ 
    ∀ x y : ℝ, h x y → 
      e = Real.sqrt (1 + (x^2/2)/(y^2)) := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3330_333021


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l3330_333023

theorem museum_ticket_cost (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_price : ℕ) (teacher_ticket_price : ℕ) : 
  num_students = 12 → 
  num_teachers = 4 → 
  student_ticket_price = 1 → 
  teacher_ticket_price = 3 → 
  num_students * student_ticket_price + num_teachers * teacher_ticket_price = 24 := by
sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l3330_333023


namespace NUMINAMATH_CALUDE_range_of_a_l3330_333090

def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : IsDecreasing f)
  (h_odd : IsOdd f)
  (h_domain : ∀ x, x ∈ Set.Ioo (-1) 1 → f x ∈ Set.univ)
  (h_condition : f (1 - a) + f (1 - 2*a) < 0) :
  0 < a ∧ a < 2/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3330_333090


namespace NUMINAMATH_CALUDE_g_inverse_composition_l3330_333042

def g : Fin 5 → Fin 5
| 1 => 4
| 2 => 3
| 3 => 1
| 4 => 5
| 5 => 2

theorem g_inverse_composition (h : Function.Bijective g) :
  (Function.invFun g (Function.invFun g (Function.invFun g 3))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_composition_l3330_333042


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3330_333089

theorem square_sum_geq_product_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + a*c :=
by sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3330_333089


namespace NUMINAMATH_CALUDE_greatest_non_sum_of_complex_l3330_333041

/-- A natural number is complex if it has at least two different prime divisors. -/
def is_complex (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n

/-- A natural number is representable as the sum of two complex numbers. -/
def is_sum_of_complex (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_complex a ∧ is_complex b ∧ n = a + b

/-- 23 is the greatest natural number that cannot be represented as the sum of two complex numbers. -/
theorem greatest_non_sum_of_complex : ∀ n : ℕ, n > 23 → is_sum_of_complex n ∧ ¬is_sum_of_complex 23 :=
sorry

end NUMINAMATH_CALUDE_greatest_non_sum_of_complex_l3330_333041


namespace NUMINAMATH_CALUDE_two_numbers_product_l3330_333003

theorem two_numbers_product (ε : ℝ) (h : ε > 0) : 
  ∃ x y : ℝ, x + y = 21 ∧ x^2 + y^2 = 527 ∧ |x * y + 43.05| < ε :=
sorry

end NUMINAMATH_CALUDE_two_numbers_product_l3330_333003


namespace NUMINAMATH_CALUDE_total_stickers_l3330_333005

def folders : Nat := 3
def sheets_per_folder : Nat := 10
def stickers_red : Nat := 3
def stickers_green : Nat := 2
def stickers_blue : Nat := 1

theorem total_stickers :
  folders * sheets_per_folder * stickers_red +
  folders * sheets_per_folder * stickers_green +
  folders * sheets_per_folder * stickers_blue = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_l3330_333005


namespace NUMINAMATH_CALUDE_parallel_plane_intersection_theorem_l3330_333016

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields for a plane

/-- A line in 3D space -/
structure Line where
  -- Add necessary fields for a line

/-- Two planes are parallel -/
def parallel_planes (α β : Plane) : Prop :=
  sorry

/-- A plane intersects another plane along a line -/
def plane_intersect (α γ : Plane) (l : Line) : Prop :=
  sorry

/-- Two lines are parallel -/
def parallel_lines (a b : Line) : Prop :=
  sorry

/-- Theorem: If two parallel planes are intersected by a third plane, 
    the lines of intersection are parallel -/
theorem parallel_plane_intersection_theorem 
  (α β γ : Plane) (a b : Line) 
  (h1 : parallel_planes α β) 
  (h2 : plane_intersect α γ a) 
  (h3 : plane_intersect β γ b) : 
  parallel_lines a b :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_intersection_theorem_l3330_333016


namespace NUMINAMATH_CALUDE_twin_brothers_age_l3330_333080

theorem twin_brothers_age :
  ∀ x : ℕ,
  (x + 1) * (x + 1) = x * x + 17 →
  x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_twin_brothers_age_l3330_333080


namespace NUMINAMATH_CALUDE_circle_equation_l3330_333037

theorem circle_equation (x y : ℝ) :
  let center := (3, 4)
  let point := (0, 0)
  let equation := (x - 3)^2 + (y - 4)^2 = 25
  (∀ p, p.1^2 + p.2^2 = (p.1 - center.1)^2 + (p.2 - center.2)^2 → p = point) →
  equation :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3330_333037


namespace NUMINAMATH_CALUDE_train_speed_l3330_333093

/-- The speed of a train passing through a tunnel -/
theorem train_speed (train_length : ℝ) (tunnel_length : ℝ) (pass_time : ℝ) :
  train_length = 100 →
  tunnel_length = 1.7 →
  pass_time = 1.5 / 60 →
  (train_length / 1000 + tunnel_length) / pass_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3330_333093


namespace NUMINAMATH_CALUDE_solution_count_equals_divisors_of_square_l3330_333010

/-- 
Given a positive integer n, count_solutions n returns the number of
ordered pairs (x, y) of positive integers satisfying xy/(x+y) = n
-/
def count_solutions (n : ℕ+) : ℕ :=
  sorry

/--
Given a positive integer n, num_divisors_square n returns the number of
positive divisors of n²
-/
def num_divisors_square (n : ℕ+) : ℕ :=
  sorry

/--
For any positive integer n, the number of ordered pairs (x, y) of
positive integers satisfying xy/(x+y) = n is equal to the number of
positive divisors of n²
-/
theorem solution_count_equals_divisors_of_square (n : ℕ+) :
  count_solutions n = num_divisors_square n :=
by sorry

end NUMINAMATH_CALUDE_solution_count_equals_divisors_of_square_l3330_333010


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l3330_333039

def M : Set ℝ := {x | x^2 + 6*x - 16 = 0}
def N (a : ℝ) : Set ℝ := {x | x*a - 3 = 0}

theorem subset_implies_a_values (h : N a ⊆ M) : a = -3/8 ∨ a = 0 ∨ a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l3330_333039


namespace NUMINAMATH_CALUDE_each_student_gets_seven_squares_l3330_333070

/-- Calculates the number of chocolate squares each student receives -/
def chocolate_squares_per_student (gerald_bars : ℕ) (squares_per_bar : ℕ) (teacher_multiplier : ℕ) (num_students : ℕ) : ℕ :=
  let total_bars := gerald_bars + gerald_bars * teacher_multiplier
  let total_squares := total_bars * squares_per_bar
  total_squares / num_students

/-- Theorem stating that each student gets 7 squares of chocolate -/
theorem each_student_gets_seven_squares :
  chocolate_squares_per_student 7 8 2 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_each_student_gets_seven_squares_l3330_333070


namespace NUMINAMATH_CALUDE_interesting_iff_prime_power_l3330_333066

def is_interesting (n : ℕ) : Prop :=
  n > 1 ∧
  ∀ x y : ℕ, (Nat.gcd x n ≠ 1 ∧ Nat.gcd y n ≠ 1) → Nat.gcd (x + y) n ≠ 1

theorem interesting_iff_prime_power (n : ℕ) :
  is_interesting n ↔ ∃ (p : ℕ) (s : ℕ), Nat.Prime p ∧ s > 0 ∧ n = p^s :=
sorry

end NUMINAMATH_CALUDE_interesting_iff_prime_power_l3330_333066


namespace NUMINAMATH_CALUDE_heptagon_angle_measure_l3330_333098

-- Define the heptagon
structure Heptagon where
  G : ℝ
  E : ℝ
  O : ℝ
  M : ℝ
  T : ℝ
  R : ℝ
  Y : ℝ

-- Define the theorem
theorem heptagon_angle_measure (GEOMETRY : Heptagon) : 
  GEOMETRY.G = GEOMETRY.E ∧ 
  GEOMETRY.G = GEOMETRY.T ∧ 
  GEOMETRY.O + GEOMETRY.Y = 180 ∧
  GEOMETRY.M = GEOMETRY.R ∧
  GEOMETRY.M = 160 →
  GEOMETRY.G = 400 / 3 := by
sorry

end NUMINAMATH_CALUDE_heptagon_angle_measure_l3330_333098


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3330_333054

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 3 ∧ b > 3 ∧
    n = 1 * a + 4 ∧
    n = 2 * b + 3

theorem smallest_dual_base_representation : 
  is_valid_representation 11 ∧ 
  ∀ m : ℕ, m < 11 → ¬(is_valid_representation m) :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3330_333054


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l3330_333074

-- Define a circle with a center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being non-intersecting
def non_intersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 ≥ (c1.radius + c2.radius)^2

-- Define the property of a circle intersecting another circle
def intersects (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 ≤ (c1.radius + c2.radius)^2

-- Theorem statement
theorem circle_intersection_theorem 
  (c1 c2 c3 c4 c5 c6 : Circle) 
  (h1 : c1.radius ≥ 1) 
  (h2 : c2.radius ≥ 1) 
  (h3 : c3.radius ≥ 1) 
  (h4 : c4.radius ≥ 1) 
  (h5 : c5.radius ≥ 1) 
  (h6 : c6.radius ≥ 1) 
  (h_non_intersect : 
    non_intersecting c1 c2 ∧ non_intersecting c1 c3 ∧ non_intersecting c1 c4 ∧ 
    non_intersecting c1 c5 ∧ non_intersecting c1 c6 ∧ non_intersecting c2 c3 ∧ 
    non_intersecting c2 c4 ∧ non_intersecting c2 c5 ∧ non_intersecting c2 c6 ∧ 
    non_intersecting c3 c4 ∧ non_intersecting c3 c5 ∧ non_intersecting c3 c6 ∧ 
    non_intersecting c4 c5 ∧ non_intersecting c4 c6 ∧ non_intersecting c5 c6)
  (c : Circle)
  (h_intersect : 
    intersects c c1 ∧ intersects c c2 ∧ intersects c c3 ∧ 
    intersects c c4 ∧ intersects c c5 ∧ intersects c c6) :
  c.radius ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_circle_intersection_theorem_l3330_333074


namespace NUMINAMATH_CALUDE_solution_set_min_value_m_plus_n_l3330_333073

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x - 2|

-- Part 1: Solution set of f(x) ≥ 3
theorem solution_set (x : ℝ) : f x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2 := by sorry

-- Part 2: Minimum value of m+n
theorem min_value_m_plus_n (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : ∀ x, f x ≥ 1/m + 1/n) : 
  m + n ≥ 8/3 ∧ (m + n = 8/3 ↔ m = n) := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_m_plus_n_l3330_333073


namespace NUMINAMATH_CALUDE_inequality_proof_l3330_333036

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3330_333036


namespace NUMINAMATH_CALUDE_virginia_sweettarts_l3330_333078

/-- The number of Virginia's friends -/
def num_friends : ℕ := 4

/-- The number of Sweettarts each person ate -/
def sweettarts_per_person : ℕ := 3

/-- The initial number of Sweettarts Virginia had -/
def initial_sweettarts : ℕ := num_friends * sweettarts_per_person + sweettarts_per_person

theorem virginia_sweettarts : initial_sweettarts = 15 := by
  sorry

end NUMINAMATH_CALUDE_virginia_sweettarts_l3330_333078


namespace NUMINAMATH_CALUDE_no_natural_square_difference_2014_l3330_333007

theorem no_natural_square_difference_2014 :
  ∀ m n : ℕ, m^2 ≠ n^2 + 2014 := by
sorry

end NUMINAMATH_CALUDE_no_natural_square_difference_2014_l3330_333007


namespace NUMINAMATH_CALUDE_tan_sum_product_22_23_degrees_l3330_333099

theorem tan_sum_product_22_23_degrees :
  Real.tan (22 * π / 180) + Real.tan (23 * π / 180) + Real.tan (22 * π / 180) * Real.tan (23 * π / 180) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_product_22_23_degrees_l3330_333099


namespace NUMINAMATH_CALUDE_molecular_weight_BaCl2_correct_l3330_333031

/-- The molecular weight of BaCl2 in g/mol -/
def molecular_weight_BaCl2 : ℝ := 207

/-- The number of moles given in the problem -/
def given_moles : ℝ := 8

/-- The total weight of the given moles of BaCl2 in grams -/
def total_weight : ℝ := 1656

/-- Theorem stating that the molecular weight of BaCl2 is correct -/
theorem molecular_weight_BaCl2_correct :
  molecular_weight_BaCl2 = total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_BaCl2_correct_l3330_333031


namespace NUMINAMATH_CALUDE_g_sum_zero_l3330_333059

theorem g_sum_zero (f : ℝ → ℝ) : 
  let g := λ x => f x - f (2010 - x)
  ∀ x, g x + g (2010 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_zero_l3330_333059


namespace NUMINAMATH_CALUDE_book_distribution_l3330_333001

theorem book_distribution (x : ℕ) : 
  (∀ (total_books : ℕ), total_books = 9 * x + 7 → 
    (∀ (student : ℕ), student < x → ∃ (books : ℕ), books = 9)) ∧ 
  (∀ (total_books : ℕ), total_books ≤ 11 * x - 1 → 
    (∃ (student : ℕ), student < x ∧ ∀ (books : ℕ), books < 11)) →
  9 * x + 7 < 11 * x := by
sorry

end NUMINAMATH_CALUDE_book_distribution_l3330_333001


namespace NUMINAMATH_CALUDE_nero_speed_l3330_333024

/-- Given a trail that takes Jerome 6 hours to run at 4 MPH, and Nero 3 hours to run,
    prove that Nero's speed is 8 MPH. -/
theorem nero_speed (jerome_time : ℝ) (nero_time : ℝ) (jerome_speed : ℝ) :
  jerome_time = 6 →
  nero_time = 3 →
  jerome_speed = 4 →
  jerome_time * jerome_speed = nero_time * (jerome_time * jerome_speed / nero_time) :=
by sorry

end NUMINAMATH_CALUDE_nero_speed_l3330_333024


namespace NUMINAMATH_CALUDE_simplify_fraction_l3330_333008

theorem simplify_fraction : (160 : ℚ) / 2880 * 40 = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3330_333008


namespace NUMINAMATH_CALUDE_brownies_on_counter_l3330_333091

def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mooney_ate : ℕ := 4
def mother_added : ℕ := 24

theorem brownies_on_counter : 
  initial_brownies - father_ate - mooney_ate + mother_added = 36 := by
  sorry

end NUMINAMATH_CALUDE_brownies_on_counter_l3330_333091


namespace NUMINAMATH_CALUDE_timmy_initial_money_l3330_333083

/-- Represents the properties of oranges and Timmy's situation --/
structure OrangeProblem where
  calories_per_orange : ℕ
  cost_per_orange : ℚ
  calories_needed : ℕ
  money_left : ℚ

/-- Calculates Timmy's initial amount of money --/
def initial_money (p : OrangeProblem) : ℚ :=
  let oranges_needed := p.calories_needed / p.calories_per_orange
  let oranges_cost := oranges_needed * p.cost_per_orange
  oranges_cost + p.money_left

/-- Theorem stating that given the problem conditions, Timmy's initial money was $10.00 --/
theorem timmy_initial_money :
  let p : OrangeProblem := {
    calories_per_orange := 80,
    cost_per_orange := 6/5, -- $1.20 represented as a rational number
    calories_needed := 400,
    money_left := 4
  }
  initial_money p = 10 := by sorry

end NUMINAMATH_CALUDE_timmy_initial_money_l3330_333083


namespace NUMINAMATH_CALUDE_amy_pencils_before_l3330_333058

/-- The number of pencils Amy bought at the school store -/
def pencils_bought : ℕ := 7

/-- The total number of pencils Amy has now -/
def total_pencils : ℕ := 10

/-- The number of pencils Amy had before buying more -/
def pencils_before : ℕ := total_pencils - pencils_bought

theorem amy_pencils_before : pencils_before = 3 := by
  sorry

end NUMINAMATH_CALUDE_amy_pencils_before_l3330_333058


namespace NUMINAMATH_CALUDE_expression_simplification_l3330_333052

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 5) (hz : z ≠ 7) :
  (x - 2) / (6 - z) * (y - 5) / (2 - x) * (z - 7) / (5 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3330_333052


namespace NUMINAMATH_CALUDE_beadshop_profit_ratio_l3330_333077

theorem beadshop_profit_ratio : 
  ∀ (total_profit monday_profit tuesday_profit wednesday_profit : ℝ),
    total_profit = 1200 →
    monday_profit = (1/3) * total_profit →
    wednesday_profit = 500 →
    tuesday_profit = total_profit - monday_profit - wednesday_profit →
    tuesday_profit / total_profit = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_beadshop_profit_ratio_l3330_333077


namespace NUMINAMATH_CALUDE_largest_of_three_numbers_l3330_333019

theorem largest_of_three_numbers (d e f : ℝ) 
  (sum_eq : d + e + f = 3)
  (sum_prod_eq : d * e + d * f + e * f = -14)
  (prod_eq : d * e * f = 21) :
  max d (max e f) = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_numbers_l3330_333019


namespace NUMINAMATH_CALUDE_club_enrollment_l3330_333063

/-- Given a club with the following properties:
  * Total members: 85
  * Members enrolled in coding course: 45
  * Members enrolled in design course: 32
  * Members enrolled in both courses: 18
  Prove that the number of members not enrolled in either course is 26. -/
theorem club_enrollment (total : ℕ) (coding : ℕ) (design : ℕ) (both : ℕ)
  (h_total : total = 85)
  (h_coding : coding = 45)
  (h_design : design = 32)
  (h_both : both = 18) :
  total - (coding + design - both) = 26 := by
  sorry

end NUMINAMATH_CALUDE_club_enrollment_l3330_333063


namespace NUMINAMATH_CALUDE_smallest_candy_count_l3330_333051

theorem smallest_candy_count : 
  ∃ (n : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    (n + 7) % 9 = 0 ∧ 
    (n - 9) % 7 = 0 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0 → false :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l3330_333051


namespace NUMINAMATH_CALUDE_complex_multiplication_l3330_333035

theorem complex_multiplication (z : ℂ) (h : z + 1 = 2 + I) : z * (1 - I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3330_333035


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l3330_333048

theorem wire_length_around_square_field (field_area : Real) (num_rounds : Nat) : 
  field_area = 24336 ∧ num_rounds = 13 → 
  13 * 4 * Real.sqrt field_area = 8112 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l3330_333048


namespace NUMINAMATH_CALUDE_division_problem_l3330_333072

theorem division_problem : (100 : ℚ) / ((6 : ℚ) / 2) = 100 / 3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3330_333072


namespace NUMINAMATH_CALUDE_towel_shrinkage_l3330_333020

/-- If a rectangle's breadth decreases by 10% and its area decreases by 28%, then its length decreases by 20%. -/
theorem towel_shrinkage (L B : ℝ) (L' B' : ℝ) (h1 : B' = 0.9 * B) (h2 : L' * B' = 0.72 * L * B) :
  L' = 0.8 * L := by
  sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l3330_333020


namespace NUMINAMATH_CALUDE_equation_solution_l3330_333068

theorem equation_solution : 
  ∃ x : ℚ, (x^2 + x + 1) / (x + 2) = x + 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3330_333068


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3330_333071

/-- Represents an isosceles triangle with perimeter 16 and one side length 6 -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  perimeter_eq : side1 + side2 + base = 16
  one_side_6 : side1 = 6 ∨ side2 = 6 ∨ base = 6
  isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base

/-- The base of the isosceles triangle is either 4 or 6 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : t.base = 4 ∨ t.base = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3330_333071


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l3330_333043

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.sin (40 * π / 180) =
  (Real.sin (50 * π / 180) * (Real.cos (60 * π / 180) * Real.cos (70 * π / 180) + Real.cos (20 * π / 180) * Real.cos (30 * π / 180))) /
  (Real.sin (40 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l3330_333043


namespace NUMINAMATH_CALUDE_proposal_i_percentage_l3330_333062

def survey_results (P_i P_ii P_iii P_i_and_ii P_i_and_iii P_ii_and_iii P_all : ℝ) : Prop :=
  P_i + P_ii + P_iii - P_i_and_ii - P_i_and_iii - P_ii_and_iii + P_all = 78 ∧
  P_ii = 30 ∧
  P_iii = 20 ∧
  P_all = 5 ∧
  P_i_and_ii + P_i_and_iii + P_ii_and_iii = 32

theorem proposal_i_percentage :
  ∀ P_i P_ii P_iii P_i_and_ii P_i_and_iii P_ii_and_iii P_all : ℝ,
  survey_results P_i P_ii P_iii P_i_and_ii P_i_and_iii P_ii_and_iii P_all →
  P_i = 55 :=
by sorry

end NUMINAMATH_CALUDE_proposal_i_percentage_l3330_333062


namespace NUMINAMATH_CALUDE_merged_class_size_is_41_l3330_333014

/-- Represents a group of students with a specific student's position --/
structure StudentGroup where
  right_rank : Nat
  left_rank : Nat

/-- Calculates the total number of students in a group --/
def group_size (g : StudentGroup) : Nat :=
  g.right_rank - 1 + g.left_rank

/-- Calculates the total number of students in the merged class --/
def merged_class_size (group_a group_b : StudentGroup) : Nat :=
  group_size group_a + group_size group_b

/-- Theorem stating the total number of students in the merged class --/
theorem merged_class_size_is_41 :
  let group_a : StudentGroup := ⟨13, 8⟩
  let group_b : StudentGroup := ⟨10, 12⟩
  merged_class_size group_a group_b = 41 := by
  sorry

#eval merged_class_size ⟨13, 8⟩ ⟨10, 12⟩

end NUMINAMATH_CALUDE_merged_class_size_is_41_l3330_333014


namespace NUMINAMATH_CALUDE_total_carriages_eq_460_l3330_333027

/-- The number of carriages in Euston -/
def euston : ℕ := 130

/-- The number of carriages in Norfolk -/
def norfolk : ℕ := euston - 20

/-- The number of carriages in Norwich -/
def norwich : ℕ := 100

/-- The number of carriages in Flying Scotsman -/
def flying_scotsman : ℕ := norwich + 20

/-- The total number of carriages -/
def total_carriages : ℕ := euston + norfolk + norwich + flying_scotsman

theorem total_carriages_eq_460 : total_carriages = 460 := by
  sorry

end NUMINAMATH_CALUDE_total_carriages_eq_460_l3330_333027


namespace NUMINAMATH_CALUDE_fraction_simplification_l3330_333044

theorem fraction_simplification : (5 * 8) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3330_333044


namespace NUMINAMATH_CALUDE_boat_distance_theorem_l3330_333025

/-- Proves that a boat traveling downstream in 2 hours and upstream in 3 hours,
    with a speed of 5 km/h in still water, covers a distance of 12 km. -/
theorem boat_distance_theorem (boat_speed : ℝ) (downstream_time upstream_time : ℝ) :
  boat_speed = 5 ∧ downstream_time = 2 ∧ upstream_time = 3 →
  ∃ (stream_speed : ℝ),
    (boat_speed + stream_speed) * downstream_time = (boat_speed - stream_speed) * upstream_time ∧
    (boat_speed + stream_speed) * downstream_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_theorem_l3330_333025


namespace NUMINAMATH_CALUDE_maria_gum_count_l3330_333055

/-- The number of gum pieces Maria has after receiving gum from Tommy and Luis -/
def total_gum (initial : ℕ) (from_tommy : ℕ) (from_luis : ℕ) : ℕ :=
  initial + from_tommy + from_luis

/-- Theorem stating that Maria has 61 pieces of gum after receiving gum from Tommy and Luis -/
theorem maria_gum_count : total_gum 25 16 20 = 61 := by
  sorry

end NUMINAMATH_CALUDE_maria_gum_count_l3330_333055


namespace NUMINAMATH_CALUDE_smallest_congruent_integer_l3330_333065

theorem smallest_congruent_integer : ∃ n : ℕ+, 
  (n : ℤ) % 3 = 2 ∧ 
  (n : ℤ) % 4 = 3 ∧ 
  (n : ℤ) % 5 = 4 ∧ 
  (n : ℤ) % 6 = 5 ∧ 
  (∀ m : ℕ+, m < n → 
    (m : ℤ) % 3 ≠ 2 ∨ 
    (m : ℤ) % 4 ≠ 3 ∨ 
    (m : ℤ) % 5 ≠ 4 ∨ 
    (m : ℤ) % 6 ≠ 5) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_congruent_integer_l3330_333065


namespace NUMINAMATH_CALUDE_upgraded_fraction_is_one_ninth_l3330_333095

/-- Represents a satellite with modular units and sensors -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on the satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

/-- Theorem stating the fraction of upgraded sensors on a specific satellite configuration -/
theorem upgraded_fraction_is_one_ninth (s : Satellite) 
  (h1 : s.units = 24)
  (h2 : s.non_upgraded_per_unit = s.total_upgraded / 3) :
  upgraded_fraction s = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_upgraded_fraction_is_one_ninth_l3330_333095


namespace NUMINAMATH_CALUDE_local_max_derivative_range_l3330_333085

/-- Given a function f with derivative f'(x) = a(x + 1)(x - a) and a local maximum at x = a, 
    prove that a is in the open interval (-1, 0) -/
theorem local_max_derivative_range (f : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h₂ : IsLocalMax f a) : 
  a ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_local_max_derivative_range_l3330_333085


namespace NUMINAMATH_CALUDE_worksheets_graded_l3330_333092

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left = 16 →
  total_worksheets - (problems_left / problems_per_worksheet) = 5 := by
sorry

end NUMINAMATH_CALUDE_worksheets_graded_l3330_333092


namespace NUMINAMATH_CALUDE_max_sum_with_constraints_l3330_333086

theorem max_sum_with_constraints (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  a + b ≤ 14/5 ∧ ∃ (a' b' : ℝ), 4 * a' + 3 * b' = 10 ∧ 3 * a' + 6 * b' = 12 ∧ a' + b' = 14/5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_constraints_l3330_333086


namespace NUMINAMATH_CALUDE_teacher_distribution_count_l3330_333015

/-- The number of ways to distribute teachers to classes --/
def distribute_teachers (n_teachers : ℕ) (n_classes : ℕ) : ℕ :=
  n_classes ^ n_teachers

/-- The number of ways to distribute teachers to classes with at least one empty class --/
def distribute_with_empty (n_teachers : ℕ) (n_classes : ℕ) : ℕ :=
  n_classes * (n_classes - 1) ^ n_teachers

/-- The number of valid distributions of teachers to classes --/
def valid_distributions (n_teachers : ℕ) (n_classes : ℕ) : ℕ :=
  distribute_teachers n_teachers n_classes - distribute_with_empty n_teachers n_classes

theorem teacher_distribution_count :
  valid_distributions 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_teacher_distribution_count_l3330_333015


namespace NUMINAMATH_CALUDE_extraneous_roots_imply_k_equals_one_l3330_333026

-- Define the equation
def equation (x k : ℝ) : Prop :=
  (x - 6) / (x - 5) = k / (5 - x)

-- Define the condition for extraneous roots
def has_extraneous_roots (k : ℝ) : Prop :=
  ∃ x, equation x k ∧ x = 5

-- Theorem statement
theorem extraneous_roots_imply_k_equals_one :
  ∀ k : ℝ, has_extraneous_roots k → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_extraneous_roots_imply_k_equals_one_l3330_333026


namespace NUMINAMATH_CALUDE_max_arithmetic_progressions_l3330_333017

/-- A strictly increasing sequence of 101 real numbers -/
def StrictlyIncreasingSeq (a : Fin 101 → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j

/-- Three terms form an arithmetic progression -/
def IsArithmeticProgression (x y z : ℝ) : Prop :=
  y = (x + z) / 2

/-- Count of arithmetic progressions in a sequence -/
def CountArithmeticProgressions (a : Fin 101 → ℝ) : ℕ :=
  (Finset.range 50).sum (fun i => i + 1) +
  (Finset.range 49).sum (fun i => i + 1)

/-- The main theorem -/
theorem max_arithmetic_progressions (a : Fin 101 → ℝ) 
  (h : StrictlyIncreasingSeq a) :
  CountArithmeticProgressions a = 2500 :=
sorry

end NUMINAMATH_CALUDE_max_arithmetic_progressions_l3330_333017


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3330_333069

/-- A monotonic function on ℝ satisfying f(x) · f(y) = f(x + y) is of the form a^x for some a > 0 -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_mono : Monotone f) 
  (h_eq : ∀ x y : ℝ, f x * f y = f (x + y)) :
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = a ^ x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3330_333069


namespace NUMINAMATH_CALUDE_derivative_at_three_l3330_333082

theorem derivative_at_three : 
  let f (x : ℝ) := (x + 3) / (x^2 + 3)
  deriv f 3 = -1/6 := by sorry

end NUMINAMATH_CALUDE_derivative_at_three_l3330_333082


namespace NUMINAMATH_CALUDE_original_weight_correct_l3330_333056

/-- Represents the original weight Tom could lift per hand in kg -/
def original_weight : ℝ := 80

/-- Represents the total weight Tom can hold with both hands after training in kg -/
def total_weight : ℝ := 352

/-- Theorem stating that the original weight satisfies the given conditions -/
theorem original_weight_correct : 
  2 * (2 * original_weight * 1.1) = total_weight := by sorry

end NUMINAMATH_CALUDE_original_weight_correct_l3330_333056


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3330_333033

theorem inequality_solution_set :
  {x : ℝ | (3 / 8 : ℝ) + |x - (1 / 4 : ℝ)| < (7 / 8 : ℝ)} = Set.Ioo (-(1 / 4 : ℝ)) ((3 / 4 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3330_333033


namespace NUMINAMATH_CALUDE_bicycle_trip_time_l3330_333034

theorem bicycle_trip_time (mary_speed john_speed : ℝ) (distance : ℝ) : 
  mary_speed = 12 → 
  john_speed = 9 → 
  distance = 90 → 
  ∃ t : ℝ, t = 6 ∧ mary_speed * t ^ 2 + john_speed * t ^ 2 = distance ^ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_trip_time_l3330_333034


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l3330_333046

/-- Given a segment AB with endpoints A(2, -2) and B(14, 4), extended through B to point C
    such that BC = 1/3 * AB, prove that the coordinates of point C are (18, 6). -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (2, -2) →
  B = (14, 4) →
  C.1 - B.1 = (1/3) * (B.1 - A.1) →
  C.2 - B.2 = (1/3) * (B.2 - A.2) →
  C = (18, 6) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l3330_333046


namespace NUMINAMATH_CALUDE_triangle_altitude_on_rectangle_diagonal_l3330_333076

/-- Given a rectangle with length l and width w, and a triangle constructed on its diagonal
    such that the area of the triangle equals the area of the rectangle,
    the altitude of the triangle drawn to the diagonal is 2lw / √(l^2 + w^2). -/
theorem triangle_altitude_on_rectangle_diagonal 
  (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := (1 / 2) * diagonal * (2 * l * w / diagonal)
  triangle_area = rectangle_area →
  2 * l * w / diagonal = 2 * l * w / Real.sqrt (l^2 + w^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_on_rectangle_diagonal_l3330_333076


namespace NUMINAMATH_CALUDE_min_double_rooms_part1_min_triple_rooms_part2_l3330_333057

/-- Represents a hotel room configuration --/
structure RoomConfig where
  double_rooms : ℕ
  triple_rooms : ℕ

/-- Calculates the total number of students that can be accommodated --/
def total_students (config : RoomConfig) : ℕ :=
  2 * config.double_rooms + 3 * config.triple_rooms

/-- Calculates the total cost of the room configuration --/
def total_cost (config : RoomConfig) (double_price : ℕ) (triple_price : ℕ) : ℕ :=
  config.double_rooms * double_price + config.triple_rooms * triple_price

/-- Theorem for part (1) --/
theorem min_double_rooms_part1 (male_students female_students : ℕ) 
  (h_total : male_students + female_students = 50)
  (h_male : male_students = 27)
  (h_female : female_students = 23)
  (double_price triple_price : ℕ)
  (h_double_price : double_price = 200)
  (h_triple_price : triple_price = 250) :
  ∃ (config : RoomConfig), 
    total_students config ≥ 50 ∧ 
    config.double_rooms = 1 ∧
    (∀ (other_config : RoomConfig), 
      total_students other_config ≥ 50 → 
      total_cost config double_price triple_price ≤ total_cost other_config double_price triple_price) :=
sorry

/-- Theorem for part (2) --/
theorem min_triple_rooms_part2 (male_students female_students : ℕ) 
  (h_total : male_students + female_students = 50)
  (h_male : male_students = 27)
  (h_female : female_students = 23)
  (double_price triple_price : ℕ)
  (h_double_price : double_price = 160)  -- 20% discount applied
  (h_triple_price : triple_price = 250)
  (max_double_rooms : ℕ)
  (h_max_double : max_double_rooms = 15) :
  ∃ (config : RoomConfig), 
    total_students config ≥ 50 ∧ 
    config.double_rooms ≤ max_double_rooms ∧
    config.triple_rooms = 8 ∧
    (∀ (other_config : RoomConfig), 
      total_students other_config ≥ 50 ∧ 
      other_config.double_rooms ≤ max_double_rooms → 
      total_cost config double_price triple_price ≤ total_cost other_config double_price triple_price) :=
sorry

end NUMINAMATH_CALUDE_min_double_rooms_part1_min_triple_rooms_part2_l3330_333057
