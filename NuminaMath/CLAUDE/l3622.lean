import Mathlib

namespace NUMINAMATH_CALUDE_no_m_exists_for_equality_subset_condition_l3622_362218

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Statement 1: There does not exist an m such that P = S(m)
theorem no_m_exists_for_equality : ¬∃ m : ℝ, P = S m := by sorry

-- Statement 2: For all m ≥ 3, P ⊆ S(m)
theorem subset_condition (m : ℝ) (h : m ≥ 3) : P ⊆ S m := by sorry

end NUMINAMATH_CALUDE_no_m_exists_for_equality_subset_condition_l3622_362218


namespace NUMINAMATH_CALUDE_students_walking_home_fraction_l3622_362207

theorem students_walking_home_fraction (total : ℚ) 
  (bus_fraction : ℚ) (auto_fraction : ℚ) (bike_fraction : ℚ) (scooter_fraction : ℚ) :
  bus_fraction = 1/3 →
  auto_fraction = 1/5 →
  bike_fraction = 1/8 →
  scooter_fraction = 1/15 →
  total = 1 →
  total - (bus_fraction + auto_fraction + bike_fraction + scooter_fraction) = 33/120 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_fraction_l3622_362207


namespace NUMINAMATH_CALUDE_triangle_area_l3622_362245

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

theorem triangle_area (a b c A B C : ℝ) : 
  a = 2 * Real.sqrt 3 →
  f A = 2 →
  b + c = 6 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3622_362245


namespace NUMINAMATH_CALUDE_divisibility_condition_l3622_362293

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def form_number (B : ℕ) : ℕ := 5000 + 200 + 10 * B + 6

theorem divisibility_condition (B : ℕ) (h : B ≤ 9) :
  is_divisible_by_3 (form_number B) ↔ (B = 2 ∨ B = 5 ∨ B = 8) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3622_362293


namespace NUMINAMATH_CALUDE_brothers_money_l3622_362271

theorem brothers_money (michael_initial : ℕ) (brother_initial : ℕ) (candy_cost : ℕ) : 
  michael_initial = 42 →
  brother_initial = 17 →
  candy_cost = 3 →
  brother_initial + michael_initial / 2 - candy_cost = 35 :=
by sorry

end NUMINAMATH_CALUDE_brothers_money_l3622_362271


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3622_362284

theorem sqrt_product_simplification : Real.sqrt 12 * Real.sqrt 75 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3622_362284


namespace NUMINAMATH_CALUDE_subtraction_problem_l3622_362290

theorem subtraction_problem : ∃ x : ℕ, x - 56 = 11 ∧ x = 67 := by sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3622_362290


namespace NUMINAMATH_CALUDE_monic_polynomial_property_l3622_362229

def is_monic_polynomial_with_properties (p : ℝ → ℝ) : Prop :=
  (∀ x, ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆, p x = x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) ∧
  (∀ i : Fin 8, p i = i)

theorem monic_polynomial_property (p : ℝ → ℝ) 
  (h : is_monic_polynomial_with_properties p) : p 8 = 40328 := by
  sorry

end NUMINAMATH_CALUDE_monic_polynomial_property_l3622_362229


namespace NUMINAMATH_CALUDE_distance_between_points_l3622_362264

theorem distance_between_points (max_speed : ℝ) (total_time : ℝ) (stream_speed_ab : ℝ) (stream_speed_ba : ℝ) (speed_percentage_ab : ℝ) (speed_percentage_ba : ℝ) (D : ℝ) :
  max_speed = 5 →
  total_time = 5 →
  stream_speed_ab = 1 →
  stream_speed_ba = 2 →
  speed_percentage_ab = 0.9 →
  speed_percentage_ba = 0.8 →
  D / (speed_percentage_ab * max_speed + stream_speed_ab) + D / (speed_percentage_ba * max_speed - stream_speed_ba) = total_time →
  26 * D = 110 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3622_362264


namespace NUMINAMATH_CALUDE_nine_digit_square_impossibility_l3622_362277

theorem nine_digit_square_impossibility (n : ℕ) : 
  (100000000 ≤ n ∧ n < 1000000000) →  -- n is a nine-digit number
  (∃ (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
    n = 100000000 * d1 + 10000000 * d2 + 1000000 * d3 + 100000 * d4 + 
        10000 * d5 + 1000 * d6 + 100 * d7 + 10 * d8 + 5 ∧
    ({d1, d2, d3, d4, d5, d6, d7, d8, 5} : Finset ℕ) = Finset.range 9) →  -- n uses all digits from 1 to 9 and ends in 5
  ¬∃ (m : ℕ), n = m^2 :=  -- n is not a perfect square
by
  sorry

end NUMINAMATH_CALUDE_nine_digit_square_impossibility_l3622_362277


namespace NUMINAMATH_CALUDE_range_of_a_l3622_362286

/-- Proposition p: The real number x satisfies x^2 - 4ax + 3a^2 < 0 -/
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: The real number x satisfies x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0 -/
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

/-- The set of x satisfying proposition p -/
def A (a : ℝ) : Set ℝ := {x | p a x}

/-- The set of x satisfying proposition q -/
def B : Set ℝ := {x | q x}

theorem range_of_a (a : ℝ) :
  a > 0 ∧ 
  (∀ x, ¬(q x) → ¬(p a x)) ∧
  (∃ x, ¬(q x) ∧ p a x) →
  1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3622_362286


namespace NUMINAMATH_CALUDE_prime_product_sum_relation_l3622_362273

theorem prime_product_sum_relation (a b c d : ℕ) :
  (Prime a ∧ Prime b ∧ Prime c ∧ Prime d) →
  (a * b * c * d = 11 * (a + b + c + d)) →
  (a + b + c + d = 20) := by
sorry

end NUMINAMATH_CALUDE_prime_product_sum_relation_l3622_362273


namespace NUMINAMATH_CALUDE_optimal_bicycle_dropoff_l3622_362261

/-- Represents the problem of finding the optimal bicycle drop-off point --/
theorem optimal_bicycle_dropoff
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (biking_speed : ℝ)
  (h_total_distance : total_distance = 30)
  (h_walking_speed : walking_speed = 5)
  (h_biking_speed : biking_speed = 20)
  (h_speeds_positive : 0 < walking_speed ∧ 0 < biking_speed)
  (h_speeds_order : walking_speed < biking_speed) :
  ∃ (x : ℝ),
    x = 5 ∧
    (∀ (y : ℝ),
      0 ≤ y ∧ y ≤ total_distance →
      max
        ((total_distance - y) / biking_speed + y / walking_speed)
        ((total_distance / 2 - y) / walking_speed + y / biking_speed)
      ≥
      max
        ((total_distance - x) / biking_speed + x / walking_speed)
        ((total_distance / 2 - x) / walking_speed + x / biking_speed)) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_bicycle_dropoff_l3622_362261


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3622_362235

/-- Represents a repeating decimal with a single repeating digit -/
def repeating_decimal_single (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with two repeating digits -/
def repeating_decimal_double (n : ℕ) : ℚ := n / 99

theorem sum_of_repeating_decimals : 
  repeating_decimal_single 6 + repeating_decimal_double 45 = 37 / 33 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3622_362235


namespace NUMINAMATH_CALUDE_fred_marbles_l3622_362220

theorem fred_marbles (total : ℕ) (dark_blue : ℕ) (green : ℕ) (red : ℕ) :
  total = 63 →
  dark_blue ≥ total / 3 →
  green = 4 →
  total = dark_blue + green + red →
  red = 38 := by
sorry

end NUMINAMATH_CALUDE_fred_marbles_l3622_362220


namespace NUMINAMATH_CALUDE_circle_radii_sum_l3622_362238

theorem circle_radii_sum (r R : ℝ) : 
  r > 0 → R > 0 →  -- Radii are positive
  R - r = 5 →  -- Distance between centers
  π * R^2 - π * r^2 = 100 * π →  -- Area between circles
  r + R = 20 := by
sorry

end NUMINAMATH_CALUDE_circle_radii_sum_l3622_362238


namespace NUMINAMATH_CALUDE_inequality_solution_l3622_362217

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (x - 2) + (x - 3) / (3 * x) ≥ 2 ↔ 
  (0 < x ∧ x ≤ 5/6) ∨ (2 < x) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3622_362217


namespace NUMINAMATH_CALUDE_number_puzzle_l3622_362248

theorem number_puzzle : 
  ∀ x : ℚ, (x / 7 - x / 11 = 100) → x = 1925 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3622_362248


namespace NUMINAMATH_CALUDE_min_length_MN_l3622_362292

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Theorem: Minimum length of MN in a unit cube -/
theorem min_length_MN (cube : Cube) (M N L : Point3D) : 
  (cube.A.x = 0 ∧ cube.A.y = 0 ∧ cube.A.z = 0) →  -- A is at origin
  (cube.B.x = 1 ∧ cube.B.y = 0 ∧ cube.B.z = 0) →  -- B is at (1,0,0)
  (cube.C.x = 1 ∧ cube.C.y = 1 ∧ cube.C.z = 0) →  -- C is at (1,1,0)
  (cube.D.x = 0 ∧ cube.D.y = 1 ∧ cube.D.z = 0) →  -- D is at (0,1,0)
  (cube.A1.x = 0 ∧ cube.A1.y = 0 ∧ cube.A1.z = 1) →  -- A1 is at (0,0,1)
  (cube.C1.x = 1 ∧ cube.C1.y = 1 ∧ cube.C1.z = 1) →  -- C1 is at (1,1,1)
  (cube.D1.x = 0 ∧ cube.D1.y = 1 ∧ cube.D1.z = 1) →  -- D1 is at (0,1,1)
  (∃ t : ℝ, M.x = t * cube.A1.x ∧ M.y = t * cube.A1.y ∧ M.z = t * cube.A1.z) →  -- M is on ray AA1
  (∃ s : ℝ, N.x = cube.B.x + s * (cube.C.x - cube.B.x) ∧ 
            N.y = cube.B.y + s * (cube.C.y - cube.B.y) ∧ 
            N.z = cube.B.z + s * (cube.C.z - cube.B.z)) →  -- N is on ray BC
  (∃ u : ℝ, L.x = cube.C1.x + u * (cube.D1.x - cube.C1.x) ∧ 
            L.y = cube.C1.y + u * (cube.D1.y - cube.C1.y) ∧ 
            L.z = cube.C1.z + u * (cube.D1.z - cube.C1.z)) →  -- L is on edge C1D1
  (∃ v : ℝ, M.x + v * (N.x - M.x) = L.x ∧ 
            M.y + v * (N.y - M.y) = L.y ∧ 
            M.z + v * (N.z - M.z) = L.z) →  -- MN intersects C1D1 at L
  (∀ M' N' : Point3D, 
    (∃ t' : ℝ, M'.x = t' * cube.A1.x ∧ M'.y = t' * cube.A1.y ∧ M'.z = t' * cube.A1.z) →
    (∃ s' : ℝ, N'.x = cube.B.x + s' * (cube.C.x - cube.B.x) ∧ 
              N'.y = cube.B.y + s' * (cube.C.y - cube.B.y) ∧ 
              N'.z = cube.B.z + s' * (cube.C.z - cube.B.z)) →
    Real.sqrt ((M'.x - N'.x)^2 + (M'.y - N'.y)^2 + (M'.z - N'.z)^2) ≥ 3) →
  Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2 + (M.z - N.z)^2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_length_MN_l3622_362292


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l3622_362212

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) :
  area = 289 → price_per_foot = 57 →
  4 * Real.sqrt area * price_per_foot = 3876 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l3622_362212


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3622_362202

/-- The equation 3x^2 - 9y^2 - 18y = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0),
    ∀ (x y : ℝ), 3 * x^2 - 9 * y^2 - 18 * y = 0 ↔
      ((y + c)^2 / a^2) - (x^2 / b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3622_362202


namespace NUMINAMATH_CALUDE_potatoes_already_cooked_l3622_362227

/-- Given a chef cooking potatoes with the following conditions:
  - The total number of potatoes to cook is 13
  - Each potato takes 6 minutes to cook
  - It will take 48 minutes to cook the remaining potatoes
  This theorem proves that the number of potatoes already cooked is 5. -/
theorem potatoes_already_cooked 
  (total_potatoes : ℕ) 
  (cooking_time_per_potato : ℕ) 
  (remaining_cooking_time : ℕ) 
  (h1 : total_potatoes = 13)
  (h2 : cooking_time_per_potato = 6)
  (h3 : remaining_cooking_time = 48) :
  total_potatoes - (remaining_cooking_time / cooking_time_per_potato) = 5 :=
by sorry

end NUMINAMATH_CALUDE_potatoes_already_cooked_l3622_362227


namespace NUMINAMATH_CALUDE_p_or_not_q_l3622_362244

def p : Prop := ∃ α : ℝ, Real.sin (Real.pi - α) = Real.cos α

def q : Prop := ∀ m : ℝ, m > 0 → 
  (∀ x y : ℝ, x^2/m^2 - y^2/m^2 = 1 → 
    Real.sqrt (1 + (m^2/m^2)) = Real.sqrt 2) ∧
  (∃ n : ℝ, n ≤ 0 ∧ 
    (∀ x y : ℝ, x^2/n^2 - y^2/n^2 = 1 → 
      Real.sqrt (1 + (n^2/n^2)) = Real.sqrt 2))

theorem p_or_not_q : (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_not_q_l3622_362244


namespace NUMINAMATH_CALUDE_red_peppers_weight_l3622_362272

/-- The weight of red peppers bought by Dale's Vegetarian Restaurant -/
def weight_red_peppers : ℝ :=
  5.666666667 - 2.8333333333333335

/-- Theorem stating that the weight of red peppers is the difference between
    the total weight of peppers and the weight of green peppers -/
theorem red_peppers_weight :
  weight_red_peppers = 5.666666667 - 2.8333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_red_peppers_weight_l3622_362272


namespace NUMINAMATH_CALUDE_largest_expression_l3622_362267

def expr_A : ℚ := 3 + 0 + 4 + 8
def expr_B : ℚ := 3 * 0 + 4 + 8
def expr_C : ℚ := 3 + 0 * 4 + 8
def expr_D : ℚ := 3 + 0 + 4 * 8
def expr_E : ℚ := 3 * 0 * 4 * 8
def expr_F : ℚ := (3 + 0 + 4) / 8

theorem largest_expression :
  expr_D = 35 ∧
  expr_D > expr_A ∧
  expr_D > expr_B ∧
  expr_D > expr_C ∧
  expr_D > expr_E ∧
  expr_D > expr_F :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l3622_362267


namespace NUMINAMATH_CALUDE_doctor_assignment_theorem_l3622_362243

/-- Represents the number of doctors -/
def num_doctors : ℕ := 4

/-- Represents the number of companies -/
def num_companies : ℕ := 3

/-- Calculates the total number of valid assignment schemes -/
def total_assignments : ℕ := sorry

/-- Calculates the number of assignments when one doctor is fixed to a company -/
def fixed_doctor_assignments : ℕ := sorry

/-- Calculates the number of assignments when two doctors cannot be in the same company -/
def separated_doctors_assignments : ℕ := sorry

theorem doctor_assignment_theorem :
  (total_assignments = 36) ∧
  (fixed_doctor_assignments = 12) ∧
  (separated_doctors_assignments = 30) := by sorry

end NUMINAMATH_CALUDE_doctor_assignment_theorem_l3622_362243


namespace NUMINAMATH_CALUDE_complex_division_equals_i_l3622_362275

theorem complex_division_equals_i : (2 + Complex.I) / (1 - 2 * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equals_i_l3622_362275


namespace NUMINAMATH_CALUDE_diagram_scale_l3622_362288

/-- Represents the scale of a diagram as a ratio of two natural numbers -/
structure Scale where
  numerator : ℕ
  denominator : ℕ

/-- Converts centimeters to millimeters -/
def cm_to_mm (cm : ℕ) : ℕ := cm * 10

theorem diagram_scale (actual_length_mm : ℕ) (diagram_length_cm : ℕ) :
  actual_length_mm = 4 →
  diagram_length_cm = 8 →
  ∃ (s : Scale), s.numerator = 20 ∧ s.denominator = 1 ∧
    cm_to_mm diagram_length_cm * s.denominator = actual_length_mm * s.numerator :=
by sorry

end NUMINAMATH_CALUDE_diagram_scale_l3622_362288


namespace NUMINAMATH_CALUDE_f_uniqueness_and_fixed_points_l3622_362287

def is_prime (p : ℕ) : Prop := Nat.Prime p

def f_conditions (f : ℕ → ℕ) : Prop :=
  (∀ p, is_prime p → f p = 1) ∧
  (∀ a b, f (a * b) = a * f b + f a * b)

theorem f_uniqueness_and_fixed_points (f : ℕ → ℕ) (h : f_conditions f) :
  (∀ g, f_conditions g → f = g) ∧
  (∀ n, n = f n ↔ ∃ p, is_prime p ∧ n = p^p) :=
sorry

end NUMINAMATH_CALUDE_f_uniqueness_and_fixed_points_l3622_362287


namespace NUMINAMATH_CALUDE_sum_five_consecutive_odd_numbers_l3622_362282

theorem sum_five_consecutive_odd_numbers (n : ℤ) :
  let middle := 2 * n + 1
  let sum := (2 * n - 3) + (2 * n - 1) + (2 * n + 1) + (2 * n + 3) + (2 * n + 5)
  sum = 5 * middle :=
by sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_odd_numbers_l3622_362282


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3622_362206

/-- The line equation y = √3 * x + m -/
def line_equation (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + m

/-- The circle equation x^2 + (y - 3)^2 = 6 -/
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 6

/-- Two points A and B on both the line and the circle -/
def intersection_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  line_equation A.1 A.2 m ∧ circle_equation A.1 A.2 ∧
  line_equation B.1 B.2 m ∧ circle_equation B.1 B.2

/-- The distance between points A and B is 2√2 -/
def distance_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8

theorem line_circle_intersection (m : ℝ) :
  (∃ A B : ℝ × ℝ, intersection_points A B m ∧ distance_condition A B) →
  m = -1 ∨ m = 7 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3622_362206


namespace NUMINAMATH_CALUDE_nearest_integer_to_sum_l3622_362254

theorem nearest_integer_to_sum : ∃ (n : ℤ), n = 3 ∧ 
  ∀ (m : ℤ), abs (m - (2007 / 2999 + 8001 / 5998 + 2001 / 3999 : ℚ)) ≥ 
              abs (n - (2007 / 2999 + 8001 / 5998 + 2001 / 3999 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_sum_l3622_362254


namespace NUMINAMATH_CALUDE_quadratic_form_coefficients_l3622_362200

theorem quadratic_form_coefficients :
  let f : ℝ → ℝ := λ x => 2 * x * (x - 1) - 3 * x
  ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ a = 2 ∧ b = -5 ∧ c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_coefficients_l3622_362200


namespace NUMINAMATH_CALUDE_scale_division_theorem_l3622_362242

/-- Represents the length of an object in feet and inches -/
structure Length where
  feet : ℕ
  inches : ℕ
  h : inches < 12

/-- Converts a Length to total inches -/
def Length.toInches (l : Length) : ℕ := l.feet * 12 + l.inches

/-- The total length of the scale -/
def totalLength : Length := ⟨6, 8, by norm_num⟩

/-- Number of equal parts to divide the scale into -/
def numParts : ℕ := 2

/-- Represents the result of dividing the scale -/
def dividedLength : Length := ⟨3, 4, by norm_num⟩

theorem scale_division_theorem :
  (totalLength.toInches / numParts : ℕ) = dividedLength.toInches := by
  sorry

end NUMINAMATH_CALUDE_scale_division_theorem_l3622_362242


namespace NUMINAMATH_CALUDE_shaded_area_equals_triangle_area_l3622_362268

/-- Given a circle with diameter d and a perpendicular line from the center to the diameter,
    the area of the shaded region formed by a semicircle with radius equal to the radius
    of the original circle, minus the area of the right triangle formed by the diameter
    and radius of the original circle, is equal to the area of the right triangle. -/
theorem shaded_area_equals_triangle_area (d : ℝ) (h : d > 0) : 
  let r := d / 2
  let semicircle_area := π * r^2 / 2
  let triangle_area := d * r / 2
  semicircle_area - (semicircle_area - triangle_area) = triangle_area := by
  sorry

#check shaded_area_equals_triangle_area

end NUMINAMATH_CALUDE_shaded_area_equals_triangle_area_l3622_362268


namespace NUMINAMATH_CALUDE_jacob_peter_age_difference_l3622_362297

/-- Given that Peter's age 10 years ago was one-third of Jacob's age at that time,
    and Peter is currently 16 years old, prove that Jacob's current age is 12 years
    more than Peter's current age. -/
theorem jacob_peter_age_difference :
  ∀ (peter_age_10_years_ago jacob_age_10_years_ago : ℕ),
  peter_age_10_years_ago = jacob_age_10_years_ago / 3 →
  peter_age_10_years_ago + 10 = 16 →
  jacob_age_10_years_ago + 10 - (peter_age_10_years_ago + 10) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_jacob_peter_age_difference_l3622_362297


namespace NUMINAMATH_CALUDE_tamika_drove_farther_l3622_362201

-- Define the given conditions
def tamika_time : ℝ := 8
def tamika_speed : ℝ := 45
def logan_time : ℝ := 5
def logan_speed : ℝ := 55

-- Define the distance calculation function
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

-- Theorem statement
theorem tamika_drove_farther : 
  distance tamika_time tamika_speed - distance logan_time logan_speed = 85 := by
  sorry

end NUMINAMATH_CALUDE_tamika_drove_farther_l3622_362201


namespace NUMINAMATH_CALUDE_two_x_plus_y_equals_five_l3622_362203

theorem two_x_plus_y_equals_five (x y : ℝ) 
  (eq1 : 7 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  2 * x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_two_x_plus_y_equals_five_l3622_362203


namespace NUMINAMATH_CALUDE_decimal_sum_l3622_362249

theorem decimal_sum : 5.467 + 2.349 + 3.785 = 11.751 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l3622_362249


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l3622_362214

theorem triangle_area_in_circle (r : ℝ) : 
  r > 0 → 
  let a := 5 * (10 / 13)
  let b := 12 * (10 / 13)
  let c := 13 * (10 / 13)
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  c = 2 * r → -- diameter of the circle
  (1/2) * a * b = 3000/169 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l3622_362214


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l3622_362209

theorem circle_tangent_to_parabola_directrix (p : ℝ) : 
  p > 0 → 
  (∃ x y : ℝ, x^2 + y^2 - 6*x - 7 = 0 ∧ 
              y^2 = 2*p*x ∧ 
              x = -p) → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l3622_362209


namespace NUMINAMATH_CALUDE_train_speed_l3622_362210

/-- Given a train of length 125 meters crossing a bridge of length 250 meters in 30 seconds,
    its speed is 45 km/hr. -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 125 →
  bridge_length = 250 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3622_362210


namespace NUMINAMATH_CALUDE_martin_improvement_l3622_362204

/-- Represents Martin's cycling performance --/
structure CyclingPerformance where
  laps : ℕ
  time : ℕ

/-- Calculates the time per lap given a cycling performance --/
def timePerLap (performance : CyclingPerformance) : ℚ :=
  performance.time / performance.laps

/-- Martin's initial cycling performance --/
def initialPerformance : CyclingPerformance :=
  { laps := 15, time := 45 }

/-- Martin's improved cycling performance --/
def improvedPerformance : CyclingPerformance :=
  { laps := 18, time := 42 }

/-- Theorem stating the improvement in Martin's per-lap time --/
theorem martin_improvement :
  timePerLap initialPerformance - timePerLap improvedPerformance = 2/3 := by
  sorry

#eval timePerLap initialPerformance - timePerLap improvedPerformance

end NUMINAMATH_CALUDE_martin_improvement_l3622_362204


namespace NUMINAMATH_CALUDE_sam_average_letters_per_day_l3622_362255

/-- The average number of letters Sam wrote per day -/
theorem sam_average_letters_per_day :
  let tuesday_letters : ℕ := 7
  let wednesday_letters : ℕ := 3
  let total_days : ℕ := 2
  let total_letters : ℕ := tuesday_letters + wednesday_letters
  (total_letters : ℚ) / total_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_sam_average_letters_per_day_l3622_362255


namespace NUMINAMATH_CALUDE_percentage_difference_l3622_362285

theorem percentage_difference (x y : ℝ) (h : x = 6 * y) :
  (x - y) / x * 100 = 500 / 6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3622_362285


namespace NUMINAMATH_CALUDE_tan_585_degrees_l3622_362232

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_585_degrees_l3622_362232


namespace NUMINAMATH_CALUDE_claire_cooking_time_l3622_362278

/-- Represents Claire's daily schedule -/
structure DailySchedule where
  total_hours : ℕ
  sleep_hours : ℕ
  clean_hours : ℕ
  craft_hours : ℕ
  tailor_hours : ℕ
  cook_hours : ℕ

/-- Claire's schedule satisfies the given conditions -/
def is_valid_schedule (s : DailySchedule) : Prop :=
  s.total_hours = 24 ∧
  s.sleep_hours = 8 ∧
  s.clean_hours = 4 ∧
  s.craft_hours = 5 ∧
  s.tailor_hours = s.craft_hours ∧
  s.total_hours = s.sleep_hours + s.clean_hours + s.craft_hours + s.tailor_hours + s.cook_hours

theorem claire_cooking_time (s : DailySchedule) (h : is_valid_schedule s) : s.cook_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_claire_cooking_time_l3622_362278


namespace NUMINAMATH_CALUDE_gcd_24_36_54_l3622_362236

theorem gcd_24_36_54 : Nat.gcd 24 (Nat.gcd 36 54) = 6 := by sorry

end NUMINAMATH_CALUDE_gcd_24_36_54_l3622_362236


namespace NUMINAMATH_CALUDE_homework_problems_per_page_l3622_362265

theorem homework_problems_per_page 
  (total_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 101)
  (h2 : finished_problems = 47)
  (h3 : remaining_pages = 6)
  (h4 : remaining_pages > 0)
  : (total_problems - finished_problems) / remaining_pages = 9 := by
  sorry

end NUMINAMATH_CALUDE_homework_problems_per_page_l3622_362265


namespace NUMINAMATH_CALUDE_building_floors_l3622_362289

-- Define the number of floors in each building
def alexie_floors : ℕ := sorry
def baptiste_floors : ℕ := sorry

-- Define the total number of bathrooms and bedrooms
def total_bathrooms : ℕ := 25
def total_bedrooms : ℕ := 18

-- State the theorem
theorem building_floors :
  (3 * alexie_floors + 4 * baptiste_floors = total_bathrooms) ∧
  (2 * alexie_floors + 3 * baptiste_floors = total_bedrooms) →
  alexie_floors = 3 ∧ baptiste_floors = 4 := by
  sorry

end NUMINAMATH_CALUDE_building_floors_l3622_362289


namespace NUMINAMATH_CALUDE_group_average_age_problem_l3622_362211

theorem group_average_age_problem (n : ℕ) : 
  (n * 14 + 32 = 16 * (n + 1)) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_group_average_age_problem_l3622_362211


namespace NUMINAMATH_CALUDE_star_three_five_l3622_362219

-- Define the star operation
def star (c d : ℝ) : ℝ := c^2 - 2*c*d + d^2

-- Theorem statement
theorem star_three_five : star 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_three_five_l3622_362219


namespace NUMINAMATH_CALUDE_vacuum_cleaner_price_difference_l3622_362281

/-- The in-store price of the vacuum cleaner in dollars -/
def in_store_price : ℚ := 150

/-- The cost of each online payment in dollars -/
def online_payment : ℚ := 35

/-- The number of online payments -/
def num_payments : ℕ := 4

/-- The one-time processing fee for online purchase in dollars -/
def processing_fee : ℚ := 12

/-- The difference in cents between online and in-store purchase -/
def price_difference_cents : ℤ := 200

theorem vacuum_cleaner_price_difference :
  (num_payments * online_payment + processing_fee - in_store_price) * 100 = price_difference_cents := by
  sorry

end NUMINAMATH_CALUDE_vacuum_cleaner_price_difference_l3622_362281


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3622_362246

theorem sin_cos_pi_12 : 2 * Real.sin (π / 12) * Real.cos (π / 12) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3622_362246


namespace NUMINAMATH_CALUDE_kendra_initial_money_l3622_362299

def wooden_toy_price : ℕ := 20
def hat_price : ℕ := 10
def wooden_toys_bought : ℕ := 2
def hats_bought : ℕ := 3
def change_received : ℕ := 30

theorem kendra_initial_money :
  wooden_toy_price * wooden_toys_bought + hat_price * hats_bought + change_received = 100 :=
by sorry

end NUMINAMATH_CALUDE_kendra_initial_money_l3622_362299


namespace NUMINAMATH_CALUDE_symmetric_function_domain_l3622_362228

/-- A function with either odd or even symmetry -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∨ (∀ x, f x = -f (-x))

/-- The theorem stating that if a symmetric function is defined on [3-a, 5], then a = -2 -/
theorem symmetric_function_domain (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc (3 - a) 5, f x ≠ 0 → True) →
  SymmetricFunction f →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_domain_l3622_362228


namespace NUMINAMATH_CALUDE_factorial_15_value_l3622_362263

theorem factorial_15_value : Nat.factorial 15 = 1307674368000 := by
  sorry

#eval Nat.factorial 15

end NUMINAMATH_CALUDE_factorial_15_value_l3622_362263


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l3622_362205

open Real

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 - Real.sqrt 3 * cos x * cos (x + π / 2)

theorem f_monotone_increasing :
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 3 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l3622_362205


namespace NUMINAMATH_CALUDE_quadratic_point_value_l3622_362226

/-- Given a quadratic function y = -ax^2 + 2ax + 3 where a > 0,
    if the point P(m, 3) lies on the graph and m ≠ 0, then m = 2. -/
theorem quadratic_point_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 →
  3 = -a * m^2 + 2 * a * m + 3 →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_value_l3622_362226


namespace NUMINAMATH_CALUDE_sams_dimes_l3622_362233

/-- Sam's dimes problem -/
theorem sams_dimes (initial_dimes given_away_dimes : ℕ) 
  (h1 : initial_dimes = 9)
  (h2 : given_away_dimes = 7) :
  initial_dimes - given_away_dimes = 2 := by
  sorry

end NUMINAMATH_CALUDE_sams_dimes_l3622_362233


namespace NUMINAMATH_CALUDE_brother_bought_ten_books_l3622_362266

/-- The number of books Sarah's brother bought -/
def brothers_total_books (sarah_paperbacks sarah_hardbacks : ℕ) : ℕ :=
  (sarah_paperbacks / 3) + (sarah_hardbacks * 2)

/-- Theorem stating that Sarah's brother bought 10 books in total -/
theorem brother_bought_ten_books :
  brothers_total_books 6 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_brother_bought_ten_books_l3622_362266


namespace NUMINAMATH_CALUDE_range_of_a_l3622_362258

theorem range_of_a (a : ℝ) : 
  ((∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
   (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0)) ↔ 
  (a ≤ -2 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3622_362258


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3622_362225

theorem vector_equation_solution (e₁ e₂ : ℝ × ℝ) (x y : ℝ) 
  (h_not_collinear : ¬ ∃ (k : ℝ), e₂ = k • e₁)
  (h_equation : (3*x - 4*y) • e₁ + (2*x - 3*y) • e₂ = 6 • e₁ + 3 • e₂) :
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3622_362225


namespace NUMINAMATH_CALUDE_guaranteed_babysitting_hours_is_eight_l3622_362257

/-- Calculates the number of guaranteed babysitting hours on Saturday given Donna's work schedule and earnings. -/
def guaranteed_babysitting_hours (
  dog_walking_hours : ℕ)
  (dog_walking_rate : ℚ)
  (dog_walking_days : ℕ)
  (card_shop_hours : ℕ)
  (card_shop_rate : ℚ)
  (card_shop_days : ℕ)
  (babysitting_rate : ℚ)
  (total_earnings : ℚ) : ℚ :=
  let dog_walking_earnings := ↑dog_walking_hours * dog_walking_rate * ↑dog_walking_days
  let card_shop_earnings := ↑card_shop_hours * card_shop_rate * ↑card_shop_days
  let other_earnings := dog_walking_earnings + card_shop_earnings
  let babysitting_earnings := total_earnings - other_earnings
  babysitting_earnings / babysitting_rate

theorem guaranteed_babysitting_hours_is_eight :
  guaranteed_babysitting_hours 2 10 5 2 (25/2) 5 10 305 = 8 := by
  sorry

end NUMINAMATH_CALUDE_guaranteed_babysitting_hours_is_eight_l3622_362257


namespace NUMINAMATH_CALUDE_fraction_equality_l3622_362295

theorem fraction_equality (m n s u : ℚ) 
  (h1 : m / n = 5 / 4) 
  (h2 : s / u = 8 / 15) : 
  (5 * m * s - 2 * n * u) / (7 * n * u - 10 * m * s) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3622_362295


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_two_l3622_362230

theorem cube_root_sum_equals_two :
  (Real.rpow (7 + 3 * Real.sqrt 21) (1/3 : ℝ)) + (Real.rpow (7 - 3 * Real.sqrt 21) (1/3 : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_two_l3622_362230


namespace NUMINAMATH_CALUDE_basketball_game_scores_l3622_362280

/-- Represents the quarterly scores of a team -/
structure QuarterlyScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the scores form an increasing geometric sequence -/
def is_increasing_geometric (s : QuarterlyScores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if the scores form an increasing arithmetic sequence -/
def is_increasing_arithmetic (s : QuarterlyScores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a team -/
def total_score (s : QuarterlyScores) : ℕ :=
  s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def first_half_score (s : QuarterlyScores) : ℕ :=
  s.q1 + s.q2

theorem basketball_game_scores :
  ∀ (raiders wildcats : QuarterlyScores),
    is_increasing_geometric raiders →
    is_increasing_arithmetic wildcats →
    raiders.q1 = wildcats.q1 + 1 →
    total_score raiders = total_score wildcats + 2 →
    total_score raiders ≤ 100 →
    total_score wildcats ≤ 100 →
    first_half_score raiders + first_half_score wildcats = 25 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l3622_362280


namespace NUMINAMATH_CALUDE_nested_square_root_value_l3622_362283

theorem nested_square_root_value :
  ∃ y : ℝ, y = Real.sqrt (2 - Real.sqrt (2 + y)) → y = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l3622_362283


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3622_362260

/-- A type representing the balls -/
inductive Ball : Type
| one
| two
| three
| four

/-- A type representing the boxes -/
inductive Box : Type
| one
| two
| three
| four

/-- A function representing the placement of balls into boxes -/
def Placement := Ball → Box

/-- The event "ball number 1 is placed into box number 1" -/
def event1 (p : Placement) : Prop := p Ball.one = Box.one

/-- The event "ball number 1 is placed into box number 2" -/
def event2 (p : Placement) : Prop := p Ball.one = Box.two

/-- The sample space of all possible placements -/
def Ω : Set Placement := {p | ∀ b : Box, ∃! ball : Ball, p ball = b}

theorem events_mutually_exclusive_but_not_opposite :
  (∀ p ∈ Ω, ¬(event1 p ∧ event2 p)) ∧
  ¬(∀ p ∈ Ω, event1 p ↔ ¬event2 p) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3622_362260


namespace NUMINAMATH_CALUDE_power_function_through_point_l3622_362216

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 4 = 2) : 
  f 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3622_362216


namespace NUMINAMATH_CALUDE_sin_C_value_area_when_b_is_6_l3622_362241

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a^2 + t.c^2 - Real.sqrt 3 * t.a * t.c = t.b^2 ∧
  3 * t.a = 2 * t.b

-- Theorem for part (I)
theorem sin_C_value (t : Triangle) (h : satisfies_conditions t) :
  Real.sin t.C = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by
  sorry

-- Theorem for part (II)
theorem area_when_b_is_6 (t : Triangle) (h : satisfies_conditions t) (h_b : t.b = 6) :
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_C_value_area_when_b_is_6_l3622_362241


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l3622_362247

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₃ : ℝ
  angleA : ℝ
  angleC : ℝ

-- Define the theorem
theorem triangle_angle_theorem (t : Triangle) 
  (h : 1 / t.h₃^2 = 1 / t.a^2 + 1 / t.b^2) : 
  t.angleC = 90 ∨ |t.angleA - t.angleC| = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l3622_362247


namespace NUMINAMATH_CALUDE_shaded_fraction_of_square_l3622_362213

theorem shaded_fraction_of_square (total_squares : ℕ) (split_squares : ℕ) (triangle_area_fraction : ℚ) :
  total_squares = 16 →
  split_squares = 4 →
  triangle_area_fraction = 1/2 →
  (split_squares : ℚ) * triangle_area_fraction / total_squares = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_square_l3622_362213


namespace NUMINAMATH_CALUDE_x_sixth_minus_six_x_squared_l3622_362234

theorem x_sixth_minus_six_x_squared (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_minus_six_x_squared_l3622_362234


namespace NUMINAMATH_CALUDE_sphere_only_circular_cross_sections_l3622_362270

-- Define the possible geometric shapes
inductive GeometricShape
  | Cylinder
  | Cone
  | Sphere
  | ConeWithCircularBase

-- Define a function to check if a shape has circular cross-sections for all plane intersections
def hasCircularCrossSections (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => true
  | _ => false

-- Theorem statement
theorem sphere_only_circular_cross_sections :
  ∀ (shape : GeometricShape),
    hasCircularCrossSections shape ↔ shape = GeometricShape.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_circular_cross_sections_l3622_362270


namespace NUMINAMATH_CALUDE_curve_properties_l3622_362237

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

def curveC (a : ℝ) : ParametricCurve :=
  { x := λ t => 1 + 2*t,
    y := λ t => a*t^2 }

def pointOnCurve (p : Point2D) (c : ParametricCurve) : Prop :=
  ∃ t : ℝ, c.x t = p.x ∧ c.y t = p.y

theorem curve_properties (a : ℝ) :
  pointOnCurve ⟨3, 1⟩ (curveC a) →
  (a = 1) ∧
  (∀ x y : ℝ, (x - 1)^2 = 4*y ↔ pointOnCurve ⟨x, y⟩ (curveC a)) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l3622_362237


namespace NUMINAMATH_CALUDE_helen_baked_554_cookies_this_morning_l3622_362269

/-- Given the total number of chocolate chip cookies and the number baked yesterday,
    calculate the number of chocolate chip cookies baked this morning. -/
def cookies_baked_this_morning (total : ℕ) (yesterday : ℕ) : ℕ :=
  total - yesterday

/-- Theorem stating that Helen baked 554 chocolate chip cookies this morning. -/
theorem helen_baked_554_cookies_this_morning :
  cookies_baked_this_morning 1081 527 = 554 := by
  sorry

end NUMINAMATH_CALUDE_helen_baked_554_cookies_this_morning_l3622_362269


namespace NUMINAMATH_CALUDE_one_point_of_contact_condition_l3622_362262

/-- Two equations have exactly one point of contact -/
def has_one_point_of_contact (f g : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = g x

/-- The parabola y = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The line y = 4x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := 4*x + c

/-- The theorem stating the condition for one point of contact -/
theorem one_point_of_contact_condition :
  ∀ c : ℝ, has_one_point_of_contact f (g c) ↔ c = -3 := by sorry

end NUMINAMATH_CALUDE_one_point_of_contact_condition_l3622_362262


namespace NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l3622_362298

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "lies in" relation between a line and a plane
variable (lies_in : Line → Plane → Prop)

-- Define our specific objects
variable (l m n : Line) (α : Plane)

-- State the theorem
theorem perpendicular_sufficient_not_necessary :
  (lies_in m α) → 
  (lies_in n α) → 
  (∀ x y : Line, lies_in x α → lies_in y α → perp_line_plane l α → perp_line_line l x ∧ perp_line_line l y) ∧ 
  (∃ x y : Line, lies_in x α → lies_in y α → perp_line_line l x ∧ perp_line_line l y ∧ ¬perp_line_plane l α) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l3622_362298


namespace NUMINAMATH_CALUDE_red_stamp_price_l3622_362231

theorem red_stamp_price 
  (red_count blue_count yellow_count : ℕ)
  (blue_price yellow_price : ℚ)
  (total_earnings : ℚ) :
  red_count = 20 →
  blue_count = 80 →
  yellow_count = 7 →
  blue_price = 4/5 →
  yellow_price = 2 →
  total_earnings = 100 →
  (red_count : ℚ) * (total_earnings - blue_count * blue_price - yellow_count * yellow_price) / red_count = 11/10 :=
by sorry

end NUMINAMATH_CALUDE_red_stamp_price_l3622_362231


namespace NUMINAMATH_CALUDE_students_studying_both_subjects_difference_l3622_362259

theorem students_studying_both_subjects_difference (total : ℕ) 
  (math_min math_max science_min science_max : ℕ) : 
  total = 2500 →
  math_min = 1875 →
  math_max = 2000 →
  science_min = 875 →
  science_max = 1125 →
  let max_both := math_min + science_min - total
  let min_both := total - math_max - science_max
  max_both - min_both = 625 := by sorry

end NUMINAMATH_CALUDE_students_studying_both_subjects_difference_l3622_362259


namespace NUMINAMATH_CALUDE_pet_ownership_percentage_l3622_362208

theorem pet_ownership_percentage (total_students : ℕ) (both_pets : ℕ)
  (h1 : total_students = 500)
  (h2 : both_pets = 50) :
  (both_pets : ℚ) / total_students * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_percentage_l3622_362208


namespace NUMINAMATH_CALUDE_planting_schemes_count_l3622_362251

/-- The number of seed types -/
def num_seed_types : ℕ := 5

/-- The number of plots -/
def num_plots : ℕ := 4

/-- The number of seed types to be selected -/
def num_selected : ℕ := 4

/-- The number of options for the first plot (pumpkins or pomegranates) -/
def first_plot_options : ℕ := 2

/-- Calculate the number of planting schemes -/
def num_planting_schemes : ℕ :=
  first_plot_options * (Nat.choose (num_seed_types - 1) (num_selected - 1)) * (Nat.factorial (num_plots - 1))

theorem planting_schemes_count : num_planting_schemes = 48 := by sorry

end NUMINAMATH_CALUDE_planting_schemes_count_l3622_362251


namespace NUMINAMATH_CALUDE_expression_simplification_l3622_362252

theorem expression_simplification :
  (((3 + 4 + 5 + 6 + 7) / 3) + ((3 * 6 + 9)^2 / 9)) = 268 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3622_362252


namespace NUMINAMATH_CALUDE_simplify_expression_l3622_362223

theorem simplify_expression (x y : ℝ) : (5 - 4*x) - (2 + 7*x - y) = 3 - 11*x + y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3622_362223


namespace NUMINAMATH_CALUDE_unit_vectors_equal_squared_magnitude_l3622_362221

/-- Two unit vectors have equal squared magnitudes -/
theorem unit_vectors_equal_squared_magnitude
  {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]
  (a b : n) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  ‖a‖^2 = ‖b‖^2 := by sorry

end NUMINAMATH_CALUDE_unit_vectors_equal_squared_magnitude_l3622_362221


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l3622_362279

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 25) 
  (h2 : coffee = 12) 
  (h3 : tea = 10) 
  (h4 : both = 5) : 
  total - (coffee + tea - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l3622_362279


namespace NUMINAMATH_CALUDE_power_function_sum_l3622_362239

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (α β : ℝ), ∀ x, f x = α * x ^ β

-- State the theorem
theorem power_function_sum (a b : ℝ) :
  isPowerFunction (fun x ↦ a * x^(2*a+1) - b + 1) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_sum_l3622_362239


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_p_l3622_362294

/-- A parabola with equation y^2 = 2px and latus rectum line x = -2 has p = 4 -/
theorem parabola_latus_rectum_p (y x p : ℝ) : 
  (y^2 = 2*p*x) →  -- Parabola equation
  (x = -2)      →  -- Latus rectum line equation
  p = 4         :=  -- Conclusion: p equals 4
by sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_p_l3622_362294


namespace NUMINAMATH_CALUDE_triangle_ratio_l3622_362215

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C →
  Real.cos A = -1/4 →
  b / c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3622_362215


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l3622_362274

/-- The number of non-birch trees -/
def non_birch_trees : ℕ := 9

/-- The number of birch trees -/
def birch_trees : ℕ := 3

/-- The total number of trees -/
def total_trees : ℕ := non_birch_trees + birch_trees

/-- The number of slots available for birch trees -/
def available_slots : ℕ := non_birch_trees + 1

/-- The probability of no two birch trees being adjacent when randomly arranged -/
theorem birch_tree_arrangement_probability :
  (Nat.choose available_slots birch_trees : ℚ) / (Nat.choose total_trees birch_trees : ℚ) = 6 / 11 := by
  sorry

#eval Nat.choose available_slots birch_trees + Nat.choose total_trees birch_trees

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l3622_362274


namespace NUMINAMATH_CALUDE_square_area_ratio_l3622_362250

theorem square_area_ratio : 
  ∀ (a b : ℝ), 
  (4 * a = 16 * b) →  -- Perimeter relation
  (a = 2 * b + 5) →   -- Side length relation
  (a^2 / b^2 = 16) := by  -- Area ratio
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3622_362250


namespace NUMINAMATH_CALUDE_equation_solution_l3622_362291

theorem equation_solution : ∃ x : ℚ, (5 * x - 2) / (6 * x - 6) = 3 / 4 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3622_362291


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l3622_362296

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Angles are positive
  a + b = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  a / b = 5 / 4 →  -- Ratio of angles is 5:4
  (a = 50 ∧ b = 40) ∨ (a = 40 ∧ b = 50) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l3622_362296


namespace NUMINAMATH_CALUDE_sequence_2024th_term_l3622_362224

/-- Definition of the sequence term -/
def sequenceTerm (n : ℕ) : ℤ × ℕ := ((-1)^(n+1) * (2*n - 1), n)

/-- The 2024th term of the sequence -/
def term2024 : ℤ × ℕ := sequenceTerm 2024

/-- Theorem stating the 2024th term of the sequence -/
theorem sequence_2024th_term :
  term2024 = (-4047, 2024) := by sorry

end NUMINAMATH_CALUDE_sequence_2024th_term_l3622_362224


namespace NUMINAMATH_CALUDE_train_passing_time_l3622_362240

/-- Prove that a train with given speed and platform crossing time will take 16 seconds to pass a stationary point -/
theorem train_passing_time (train_speed_kmph : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed_kmph = 72 →
  platform_length = 280 →
  platform_crossing_time = 30 →
  (train_speed_kmph * 1000 / 3600) * ((platform_length + train_speed_kmph * 1000 / 3600 * platform_crossing_time) / (train_speed_kmph * 1000 / 3600)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3622_362240


namespace NUMINAMATH_CALUDE_factorization_equality_l3622_362256

theorem factorization_equality (x : ℝ) : 
  (x + 1)^4 + (x + 3)^4 - 272 = 2 * (x^2 + 4*x + 19) * (x + 5) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3622_362256


namespace NUMINAMATH_CALUDE_quartic_equation_sum_l3622_362276

theorem quartic_equation_sum (a b c : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℕ+, 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (∀ x : ℝ, x^4 - 10*x^3 + a*x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
  a + b + c = 109 := by
sorry

end NUMINAMATH_CALUDE_quartic_equation_sum_l3622_362276


namespace NUMINAMATH_CALUDE_geometric_mean_of_sqrt2_plus_minus_one_l3622_362253

theorem geometric_mean_of_sqrt2_plus_minus_one :
  let a := Real.sqrt 2 + 1
  let b := Real.sqrt 2 - 1
  let geometric_mean := Real.sqrt (a * b)
  geometric_mean = 1 ∨ geometric_mean = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_mean_of_sqrt2_plus_minus_one_l3622_362253


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3622_362222

theorem book_pages_calculation (pages_per_day : ℕ) (days_read : ℕ) (fraction_read : ℚ) : 
  pages_per_day = 12 →
  days_read = 15 →
  fraction_read = 3/4 →
  (pages_per_day * days_read : ℚ) / fraction_read = 240 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l3622_362222
