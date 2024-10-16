import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3214_321417

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  ((1/3)^2 + (1/4)^2) / ((1/5)^2 + (1/6)^2) = 21*x / (65*y) →
  Real.sqrt x / Real.sqrt y = 25 * Real.sqrt 65 / (2 * Real.sqrt 1281) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3214_321417


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3214_321452

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (3 - 2 * i * z = 1 + 4 * i * z) ∧ (z = -i / 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3214_321452


namespace NUMINAMATH_CALUDE_bc_over_a_is_zero_l3214_321402

theorem bc_over_a_is_zero (a b c : ℝ) 
  (h1 : a = 2*b + Real.sqrt 2)
  (h2 : a*b + (Real.sqrt 3 / 2)*c^2 + 1/4 = 0) : 
  b*c/a = 0 := by
  sorry

end NUMINAMATH_CALUDE_bc_over_a_is_zero_l3214_321402


namespace NUMINAMATH_CALUDE_max_discarded_apples_l3214_321430

theorem max_discarded_apples (n : ℕ) : ∃ (q : ℕ), n = 7 * q + 6 ∧ 
  ∀ (r : ℕ), r < 6 → ∃ (q' : ℕ), n ≠ 7 * q' + r :=
sorry

end NUMINAMATH_CALUDE_max_discarded_apples_l3214_321430


namespace NUMINAMATH_CALUDE_amount_A_to_B_plus_ratio_l3214_321470

/-- The amount promised for a B+ grade -/
def amount_B_plus : ℝ := 5

/-- The number of courses in Paul's scorecard -/
def num_courses : ℕ := 10

/-- The flat amount received for each A+ grade -/
def amount_A_plus : ℝ := 15

/-- The maximum amount Paul could receive -/
def max_amount : ℝ := 190

/-- The amount promised for an A grade -/
noncomputable def amount_A : ℝ := 
  (max_amount - 2 * amount_A_plus) / (2 * (num_courses - 2))

/-- Theorem stating that the ratio of amount promised for an A to a B+ is 2:1 -/
theorem amount_A_to_B_plus_ratio : 
  amount_A / amount_B_plus = 2 := by sorry

end NUMINAMATH_CALUDE_amount_A_to_B_plus_ratio_l3214_321470


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l3214_321416

-- Define the given constants
def train_length : ℝ := 250
def pole_passing_time : ℝ := 10
def platform_length : ℝ := 1250
def speed_reduction_factor : ℝ := 0.75

-- Define the theorem
theorem train_platform_passing_time :
  let original_speed := train_length / pole_passing_time
  let incline_speed := original_speed * speed_reduction_factor
  let total_distance := train_length + platform_length
  total_distance / incline_speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l3214_321416


namespace NUMINAMATH_CALUDE_rogers_pennies_l3214_321477

/-- The number of pennies Roger collected initially -/
def pennies_collected : ℕ := sorry

/-- The number of nickels Roger collected -/
def nickels : ℕ := 36

/-- The number of dimes Roger collected -/
def dimes : ℕ := 15

/-- The number of coins Roger donated -/
def coins_donated : ℕ := 66

/-- The number of coins Roger had left after donating -/
def coins_left : ℕ := 27

/-- The total number of coins Roger had initially -/
def total_coins : ℕ := coins_donated + coins_left

theorem rogers_pennies :
  pennies_collected = total_coins - (nickels + dimes) :=
by sorry

end NUMINAMATH_CALUDE_rogers_pennies_l3214_321477


namespace NUMINAMATH_CALUDE_conic_is_parabola_l3214_321400

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  |x - 3| = Real.sqrt ((y + 4)^2 + x^2)

-- Define what it means for an equation to describe a parabola
def describes_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x y, f x y ↔ y = a * x^2 + b * x + c

-- Theorem statement
theorem conic_is_parabola : describes_parabola conic_equation := by
  sorry

end NUMINAMATH_CALUDE_conic_is_parabola_l3214_321400


namespace NUMINAMATH_CALUDE_square_sum_divided_l3214_321429

theorem square_sum_divided : (10^2 + 6^2) / 2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_divided_l3214_321429


namespace NUMINAMATH_CALUDE_inequality_proof_l3214_321490

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3214_321490


namespace NUMINAMATH_CALUDE_pinterest_group_pins_l3214_321448

/-- Calculates the number of pins in a Pinterest group after one month -/
def pinsAfterOneMonth (
  groupSize : ℕ
  ) (averageDailyContribution : ℕ
  ) (weeklyDeletionRate : ℕ
  ) (initialPins : ℕ
  ) : ℕ :=
  let daysInMonth : ℕ := 30
  let weeksInMonth : ℕ := 4
  let monthlyContribution := groupSize * averageDailyContribution * daysInMonth
  let monthlyDeletion := groupSize * weeklyDeletionRate * weeksInMonth
  initialPins + monthlyContribution - monthlyDeletion

theorem pinterest_group_pins :
  pinsAfterOneMonth 20 10 5 1000 = 6600 := by
  sorry

end NUMINAMATH_CALUDE_pinterest_group_pins_l3214_321448


namespace NUMINAMATH_CALUDE_unique_prime_sevens_l3214_321446

def A (n : ℕ+) : ℕ := 1 + 7 * (10^n.val - 1) / 9

def B (n : ℕ+) : ℕ := 3 + 7 * (10^n.val - 1) / 9

theorem unique_prime_sevens : 
  ∃! (n : ℕ+), Nat.Prime (A n) ∧ Nat.Prime (B n) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sevens_l3214_321446


namespace NUMINAMATH_CALUDE_hexagons_from_circle_points_l3214_321449

/-- The number of points on the circle -/
def n : ℕ := 15

/-- The number of vertices in a hexagon -/
def k : ℕ := 6

/-- A function to calculate binomial coefficient -/
def binomial_coefficient (n k : ℕ) : ℕ := (Nat.choose n k)

/-- Theorem: The number of distinct convex hexagons formed from 15 points on a circle is 5005 -/
theorem hexagons_from_circle_points : binomial_coefficient n k = 5005 := by
  sorry

#eval binomial_coefficient n k  -- This should output 5005

end NUMINAMATH_CALUDE_hexagons_from_circle_points_l3214_321449


namespace NUMINAMATH_CALUDE_inscribed_triangle_radius_l3214_321432

/-- Given a regular triangle inscribed in a circular segment with the following properties:
    - The arc of the segment has a central angle α
    - One vertex of the triangle coincides with the midpoint of the arc
    - The other two vertices lie on the chord
    - The area of the triangle is S
    Then the radius R of the circle is given by R = (√(S√3)) / (2 sin²(α/4)) -/
theorem inscribed_triangle_radius (S α : ℝ) (h_S : S > 0) (h_α : 0 < α ∧ α < 2 * Real.pi) :
  ∃ R : ℝ, R > 0 ∧ R = (Real.sqrt (S * Real.sqrt 3)) / (2 * (Real.sin (α / 4))^2) :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_radius_l3214_321432


namespace NUMINAMATH_CALUDE_square_of_difference_of_roots_l3214_321458

theorem square_of_difference_of_roots (p q : ℝ) : 
  (2 * p^2 + 7 * p - 30 = 0) → 
  (2 * q^2 + 7 * q - 30 = 0) → 
  (p - q)^2 = 289 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_of_roots_l3214_321458


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3214_321456

theorem complex_fraction_equality : (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3214_321456


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l3214_321498

theorem gcf_lcm_sum (A B : ℕ) : 
  (A = Nat.gcd 12 (Nat.gcd 18 30)) → 
  (B = Nat.lcm 12 (Nat.lcm 18 30)) → 
  2 * A + B = 192 :=
by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l3214_321498


namespace NUMINAMATH_CALUDE_pencils_to_yuna_l3214_321435

/-- The number of pencils in a dozen -/
def pencils_per_dozen : ℕ := 12

/-- The number of dozens Jimin initially had -/
def initial_dozens : ℕ := 3

/-- The number of pencils Jimin gave to his younger brother -/
def pencils_to_brother : ℕ := 8

/-- The number of pencils Jimin has left -/
def pencils_left : ℕ := 17

/-- Proves that the number of pencils Jimin gave to Yuna is 11 -/
theorem pencils_to_yuna :
  initial_dozens * pencils_per_dozen - pencils_to_brother - pencils_left = 11 := by
  sorry

end NUMINAMATH_CALUDE_pencils_to_yuna_l3214_321435


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3214_321406

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (e : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / a^2 - y₁^2 / b^2 = 1 ∧ 
    x₂^2 / a^2 - y₂^2 / b^2 = 1 ∧
    y₁ - y₂ = x₁ - x₂ ∧
    x₁ > c ∧ x₂ > c) →
  (2 * b^2) / a = (2 * Real.sqrt 2 / 3) * b * e^2 →
  e = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3214_321406


namespace NUMINAMATH_CALUDE_expression_equals_sum_l3214_321476

theorem expression_equals_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_sum_l3214_321476


namespace NUMINAMATH_CALUDE_mushroom_collectors_l3214_321493

theorem mushroom_collectors (n : ℕ) : 
  (n^2 + 9*n - 2) % (n + 11) = 0 → n < 11 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collectors_l3214_321493


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3214_321495

theorem hyperbola_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_imaginary_axis : 2 * b = 2) (h_focal_length : 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 3) :
  ∃ k : ℝ, k = b / a ∧ k = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3214_321495


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3214_321451

theorem arithmetic_mean_problem (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = c →
  b * c = 400 →
  d = 28 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3214_321451


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l3214_321421

/-- A six-digit number starting with 1 -/
def SixDigitNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 1

/-- Function to move the first digit of a six-digit number to the last position -/
def MoveFirstToLast (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

/-- The main theorem -/
theorem unique_six_digit_number : 
  ∀ n : ℕ, SixDigitNumber n → (MoveFirstToLast n = 3 * n) → n = 142857 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l3214_321421


namespace NUMINAMATH_CALUDE_valid_lineups_count_l3214_321455

-- Define the total number of players
def total_players : ℕ := 15

-- Define the number of players in a starting lineup
def lineup_size : ℕ := 6

-- Define the number of players who can't play together
def restricted_players : ℕ := 3

-- Define the function to calculate the number of valid lineups
def valid_lineups : ℕ := sorry

-- Theorem statement
theorem valid_lineups_count :
  valid_lineups = 3300 :=
sorry

end NUMINAMATH_CALUDE_valid_lineups_count_l3214_321455


namespace NUMINAMATH_CALUDE_tetrahedron_in_spheres_l3214_321463

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron defined by four points -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere defined by its center and radius -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Checks if a point is inside a sphere -/
def isInSphere (p : Point3D) (s : Sphere) : Prop := sorry

/-- Checks if a point is inside a tetrahedron -/
def isInTetrahedron (p : Point3D) (t : Tetrahedron) : Prop := sorry

/-- Creates a sphere with diameter AB -/
def sphereAB (t : Tetrahedron) : Sphere := sorry

/-- Creates a sphere with diameter AC -/
def sphereAC (t : Tetrahedron) : Sphere := sorry

/-- Creates a sphere with diameter AD -/
def sphereAD (t : Tetrahedron) : Sphere := sorry

/-- The main theorem: every point in the tetrahedron is in at least one of the three spheres -/
theorem tetrahedron_in_spheres (t : Tetrahedron) (p : Point3D) :
  isInTetrahedron p t →
  (isInSphere p (sphereAB t) ∨ isInSphere p (sphereAC t) ∨ isInSphere p (sphereAD t)) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_in_spheres_l3214_321463


namespace NUMINAMATH_CALUDE_climbing_five_floors_l3214_321413

/-- The number of ways to climb a building with a given number of floors and staircases per floor -/
def climbingWays (floors : ℕ) (staircasesPerFloor : ℕ) : ℕ :=
  staircasesPerFloor ^ (floors - 1)

/-- Theorem: In a 5-floor building with 2 staircases per floor, there are 16 ways to go from the first to the fifth floor -/
theorem climbing_five_floors :
  climbingWays 5 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_climbing_five_floors_l3214_321413


namespace NUMINAMATH_CALUDE_multiply_eight_negative_half_l3214_321483

theorem multiply_eight_negative_half : 8 * (-1/2 : ℚ) = -4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_eight_negative_half_l3214_321483


namespace NUMINAMATH_CALUDE_trajectory_and_line_equation_l3214_321422

/-- The trajectory of point P and the equation of line l -/
theorem trajectory_and_line_equation :
  ∀ (P : ℝ × ℝ) (F : ℝ × ℝ) (l : Set (ℝ × ℝ)) (M : ℝ × ℝ),
  F = (3 * Real.sqrt 3, 0) →
  l = {(x, y) | x = 4 * Real.sqrt 3} →
  M = (4, 2) →
  (∀ (x y : ℝ), P = (x, y) →
    Real.sqrt ((x - 3 * Real.sqrt 3)^2 + y^2) / |x - 4 * Real.sqrt 3| = Real.sqrt 3 / 2) →
  (∃ (B C : ℝ × ℝ), B ∈ l ∧ C ∈ l ∧ B ≠ C ∧ M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) →
  (∀ (x y : ℝ), P = (x, y) → x^2 / 36 + y^2 / 9 = 1) ∧
  (∃ (k : ℝ), k = -1/2 ∧ ∀ (x y : ℝ), y - 2 = k * (x - 4) ↔ x + 2*y - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_line_equation_l3214_321422


namespace NUMINAMATH_CALUDE_square_sum_implies_product_l3214_321497

theorem square_sum_implies_product (m : ℝ) 
  (h : (m - 2023)^2 + (2024 - m)^2 = 2025) : 
  (m - 2023) * (2024 - m) = -1012 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_l3214_321497


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3214_321410

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem f_increasing_on_interval :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₁ < 1 → x₂ < 1 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3214_321410


namespace NUMINAMATH_CALUDE_candy_eaten_l3214_321459

theorem candy_eaten (katie_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) :
  katie_candy = 8 →
  sister_candy = 23 →
  remaining_candy = 23 →
  katie_candy + sister_candy - remaining_candy = 8 :=
by sorry

end NUMINAMATH_CALUDE_candy_eaten_l3214_321459


namespace NUMINAMATH_CALUDE_cos_squared_plus_sin_double_l3214_321460

/-- If the terminal side of angle α passes through point P(2, 1) in the Cartesian coordinate system, then cos²α + sin(2α) = 8/5 -/
theorem cos_squared_plus_sin_double (α : Real) :
  (∃ (x y : Real), x = 2 ∧ y = 1 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_plus_sin_double_l3214_321460


namespace NUMINAMATH_CALUDE_square_perimeter_l3214_321439

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (7/3 * s = 42) → (4 * s = 72) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3214_321439


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l3214_321423

theorem smallest_number_divisible (n : ℕ) : n = 34 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 10 = 2 * k ∧ m - 10 = 6 * k ∧ m - 10 = 12 * k ∧ m - 10 = 24 * k)) ∧
  (∃ k : ℕ, n - 10 = 2 * k ∧ n - 10 = 6 * k ∧ n - 10 = 12 * k ∧ n - 10 = 24 * k) ∧
  n > 10 :=
by sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l3214_321423


namespace NUMINAMATH_CALUDE_tan_addition_formula_l3214_321409

theorem tan_addition_formula (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π / 3) = (-6 - 5 * Real.sqrt 3) / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_formula_l3214_321409


namespace NUMINAMATH_CALUDE_k_range_theorem_l3214_321436

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x - 1

-- Define the even function g and odd function h
def g (x : ℝ) : ℝ := x^2 - 1
def h (x : ℝ) : ℝ := x

-- State the theorem
theorem k_range_theorem (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → g (k*x + k/x) < g (x^2 + 1/x^2 + 1)) ↔ 
  (-3/2 < k ∧ k < 3/2) :=
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l3214_321436


namespace NUMINAMATH_CALUDE_punch_machine_settings_l3214_321492

/-- Represents a punching pattern for a 9-field ticket -/
def PunchingPattern := Fin 9 → Bool

/-- Checks if a punching pattern is symmetric when reversed -/
def is_symmetric (p : PunchingPattern) : Prop :=
  ∀ i : Fin 9, p i = p (8 - i)

/-- The total number of possible punching patterns -/
def total_patterns : ℕ := 2^9

/-- The number of symmetric punching patterns -/
def symmetric_patterns : ℕ := 2^6

/-- The number of valid punching patterns (different when reversed) -/
def valid_patterns : ℕ := total_patterns - symmetric_patterns

theorem punch_machine_settings :
  valid_patterns = 448 :=
sorry

end NUMINAMATH_CALUDE_punch_machine_settings_l3214_321492


namespace NUMINAMATH_CALUDE_injective_function_property_l3214_321431

theorem injective_function_property (f : ℕ → ℕ) :
  (∀ m n : ℕ, m > 0 → n > 0 → f (n * f m) ≤ n * m) →
  Function.Injective f →
  ∀ x : ℕ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_injective_function_property_l3214_321431


namespace NUMINAMATH_CALUDE_school_dance_attendance_l3214_321496

theorem school_dance_attendance (P : ℕ) : 
  (P * 10 / 100 = P / 10) →  -- 10% of P are faculty and staff
  (P * 90 / 100 = P * 9 / 10) →  -- 90% of P are students
  ((P * 9 / 10) * 2 / 3 = (P * 9 / 10) - 30) →  -- Two-thirds of students are girls
  ((P * 9 / 10) * 1 / 3 = 30) →  -- One-third of students are boys
  P = 100 := by sorry

end NUMINAMATH_CALUDE_school_dance_attendance_l3214_321496


namespace NUMINAMATH_CALUDE_triangle_minimum_perimeter_l3214_321465

/-- Given a triangle with sides a, b, c, semiperimeter p, and area S,
    the perimeter is minimized when the triangle is equilateral. -/
theorem triangle_minimum_perimeter 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (p : ℝ) (hp : p = (a + b + c) / 2)
  (S : ℝ) (hS : S > 0)
  (harea : S^2 = p * (p - a) * (p - b) * (p - c)) :
  ∃ (min_a min_b min_c : ℝ),
    min_a = min_b ∧ min_b = min_c ∧
    min_a + min_b + min_c ≤ a + b + c ∧
    (min_a + min_b + min_c) / 2 * ((min_a + min_b + min_c) / 2 - min_a) * 
    ((min_a + min_b + min_c) / 2 - min_b) * ((min_a + min_b + min_c) / 2 - min_c) = S^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_minimum_perimeter_l3214_321465


namespace NUMINAMATH_CALUDE_divisibility_by_1947_l3214_321486

theorem divisibility_by_1947 (n : ℕ) : 
  (46 * 2^(n+1) + 296 * 13 * 2^(n+1)) % 1947 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_1947_l3214_321486


namespace NUMINAMATH_CALUDE_inequality_proof_l3214_321488

theorem inequality_proof (x y z : ℝ) (n : ℕ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) 
  (h_n_pos : n > 0) : 
  (x^4 / (y*(1-y^n))) + (y^4 / (z*(1-z^n))) + (z^4 / (x*(1-x^n))) ≥ 3^n / (3^(n-2) - 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3214_321488


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3214_321437

/-- Given a parabola passing through points (-2,0) and (4,0), 
    its axis of symmetry is the line x = 1 -/
theorem parabola_axis_of_symmetry :
  ∀ (f : ℝ → ℝ),
  (f (-2) = 0) →
  (f 4 = 0) →
  (∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x, f (1 + x) = f (1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3214_321437


namespace NUMINAMATH_CALUDE_train_journey_distance_l3214_321461

/-- Calculates the total distance traveled by a train with increasing speed -/
def train_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  hours * (2 * initial_speed + (hours - 1) * speed_increase) / 2

/-- Theorem: A train traveling for 11 hours, with an initial speed of 10 miles/hr
    and increasing its speed by 10 miles/hr each hour, travels a total of 660 miles -/
theorem train_journey_distance :
  train_distance 10 10 11 = 660 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_distance_l3214_321461


namespace NUMINAMATH_CALUDE_min_value_expression_l3214_321442

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (3 * r) / (p + 2 * q) + (3 * p) / (2 * r + q) + (2 * q) / (p + r) ≥ 29 / 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3214_321442


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3214_321480

theorem inequality_equivalence (x : ℝ) : 
  3/20 + |2*x - 5/40| < 9/40 ↔ 1/40 < x ∧ x < 1/10 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3214_321480


namespace NUMINAMATH_CALUDE_fair_eight_sided_die_probability_l3214_321481

theorem fair_eight_sided_die_probability : 
  let outcomes : Fin 8 × Fin 8 → Prop := λ (a, b) => a ≥ b
  let total_outcomes : ℕ := 64
  let favorable_outcomes : ℕ := (Finset.range 8).sum (λ i => 8 - i)
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 16 := by
sorry

end NUMINAMATH_CALUDE_fair_eight_sided_die_probability_l3214_321481


namespace NUMINAMATH_CALUDE_circles_common_points_l3214_321404

/-- Two circles with radii 2 and 3 have common points if and only if 
    the distance between their centers is between 1 and 5 (inclusive). -/
theorem circles_common_points (d : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (x - d)^2 + y^2 = 9) ↔ 1 ≤ d ∧ d ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_circles_common_points_l3214_321404


namespace NUMINAMATH_CALUDE_curve_classification_l3214_321426

-- Define the curve equation
def curve_equation (x y m : ℝ) : Prop :=
  x^2 / (5 - m) + y^2 / (2 - m) = 1

-- Define the condition for m
def m_condition (m : ℝ) : Prop := m < 3

-- Define the result for different ranges of m
def curve_type (m : ℝ) : Prop :=
  (m < 2 → ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, curve_equation x y m ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  (2 < m → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, curve_equation x y m ↔ (x^2 / a^2 - y^2 / b^2 = 1))

-- Theorem statement
theorem curve_classification (m : ℝ) :
  m_condition m → curve_type m :=
sorry

end NUMINAMATH_CALUDE_curve_classification_l3214_321426


namespace NUMINAMATH_CALUDE_target_line_properties_l3214_321428

/-- The equation of line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 6 * y + 4 = 0

/-- The equation of line l₂ -/
def l₂ (x y : ℝ) : Prop := 2 * x + y = 5

/-- The intersection point of l₁ and l₂ -/
def intersection_point : ℝ × ℝ := (2, 1)

/-- The equation of the line we're proving -/
def target_line (x y : ℝ) : Prop := x - 2 * y = 0

/-- Theorem stating that the target_line passes through the intersection point and is perpendicular to l₂ -/
theorem target_line_properties :
  (target_line (intersection_point.1) (intersection_point.2)) ∧
  (∀ x y : ℝ, l₂ x y → ∀ x' y' : ℝ, target_line x' y' →
    (y' - intersection_point.2) * (x - intersection_point.1) = 
    -(x' - intersection_point.1) * (y - intersection_point.2)) :=
sorry

end NUMINAMATH_CALUDE_target_line_properties_l3214_321428


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3214_321445

/-- For any real number m, the line mx + y - 1 + 2m = 0 passes through the point (-2, 1) -/
theorem fixed_point_on_line (m : ℝ) : m * (-2) + 1 - 1 + 2 * m = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3214_321445


namespace NUMINAMATH_CALUDE_distance_traveled_l3214_321462

theorem distance_traveled (initial_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) :
  initial_speed = 10 →
  increased_speed = 15 →
  additional_distance = 15 →
  ∃ (actual_distance : ℝ) (time : ℝ),
    actual_distance = initial_speed * time ∧
    actual_distance + additional_distance = increased_speed * time ∧
    actual_distance = 30 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l3214_321462


namespace NUMINAMATH_CALUDE_power_of_product_l3214_321485

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3214_321485


namespace NUMINAMATH_CALUDE_square_rectangle_area_ratio_l3214_321478

theorem square_rectangle_area_ratio :
  ∀ (square_perimeter : ℝ) (rect_length rect_width : ℝ),
    square_perimeter = 256 →
    rect_length = 32 →
    rect_width = 64 →
    (square_perimeter / 4)^2 / (rect_length * rect_width) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_ratio_l3214_321478


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3214_321494

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that the common difference is 4 if 2S_3 - 3S_2 = 12 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1)))  -- Definition of S_n
  (h2 : 2 * S 3 - 3 * S 2 = 12)  -- Given condition
  : a 2 - a 1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3214_321494


namespace NUMINAMATH_CALUDE_solve_q_l3214_321440

theorem solve_q (n m q : ℚ) 
  (h1 : (7 : ℚ) / 8 = n / 96)
  (h2 : (7 : ℚ) / 8 = (m + n) / 112)
  (h3 : (7 : ℚ) / 8 = (q - m) / 144) : 
  q = 140 := by sorry

end NUMINAMATH_CALUDE_solve_q_l3214_321440


namespace NUMINAMATH_CALUDE_carlas_sunflowers_l3214_321443

/-- The number of sunflowers Carla has -/
def num_sunflowers : ℕ := sorry

/-- The number of dandelions Carla has -/
def num_dandelions : ℕ := 8

/-- The number of seeds per sunflower -/
def seeds_per_sunflower : ℕ := 9

/-- The number of seeds per dandelion -/
def seeds_per_dandelion : ℕ := 12

/-- The percentage of seeds that come from dandelions -/
def dandelion_seed_percentage : ℚ := 64 / 100

theorem carlas_sunflowers : 
  num_sunflowers = 6 ∧
  num_dandelions * seeds_per_dandelion = 
    (dandelion_seed_percentage : ℚ) * 
    (num_sunflowers * seeds_per_sunflower + num_dandelions * seeds_per_dandelion) :=
by sorry

end NUMINAMATH_CALUDE_carlas_sunflowers_l3214_321443


namespace NUMINAMATH_CALUDE_integer_1200_in_column_B_l3214_321418

/-- The column type representing the six columns A, B, C, D, E, F --/
inductive Column
| A | B | C | D | E | F

/-- The function that maps a positive integer to its corresponding column in the zigzag pattern --/
def columnFor (n : ℕ) : Column :=
  match (n - 1) % 10 with
  | 0 => Column.A
  | 1 => Column.B
  | 2 => Column.C
  | 3 => Column.D
  | 4 => Column.E
  | 5 => Column.F
  | 6 => Column.E
  | 7 => Column.D
  | 8 => Column.C
  | _ => Column.B

/-- Theorem stating that the integer 1200 will be placed in column B --/
theorem integer_1200_in_column_B : columnFor 1200 = Column.B := by
  sorry

end NUMINAMATH_CALUDE_integer_1200_in_column_B_l3214_321418


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_6_power_5_l3214_321482

theorem units_digit_of_7_power_6_power_5 : ∃ n : ℕ, 7^(6^5) ≡ 1 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_6_power_5_l3214_321482


namespace NUMINAMATH_CALUDE_vector_operations_l3214_321475

/-- Given vectors a and b in ℝ², prove their sum and dot product -/
theorem vector_operations (a b : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (3, 1)) : 
  (a.1 + b.1, a.2 + b.2) = (4, 3) ∧ a.1 * b.1 + a.2 * b.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l3214_321475


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_no_intersection_l3214_321425

-- Define a structure for a 3D space
structure Space3D where
  -- Add any necessary fields

-- Define a line in 3D space
structure Line where
  -- Add any necessary fields

-- Define a plane in 3D space
structure Plane where
  -- Add any necessary fields

-- Define parallelism between a line and a plane
def parallel (l : Line) (p : Plane) : Prop := sorry

-- Define intersection between two lines
def intersect (l1 l2 : Line) : Prop := sorry

-- Define a function to get a line in a plane
def line_in_plane (p : Plane) : Line := sorry

-- Theorem statement
theorem line_parallel_to_plane_no_intersection 
  (a : Line) (α : Plane) : 
  parallel a α → ∀ l : Line, (∃ p : Plane, l = line_in_plane p) → ¬(intersect a l) := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_no_intersection_l3214_321425


namespace NUMINAMATH_CALUDE_percentage_equality_theorem_l3214_321466

theorem percentage_equality_theorem (x : ℚ) : 
  (30 : ℚ) / 100 * x = (25 : ℚ) / 100 * 40 → x = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_theorem_l3214_321466


namespace NUMINAMATH_CALUDE_freddy_age_l3214_321401

theorem freddy_age (matthew rebecca freddy : ℕ) : 
  matthew + rebecca + freddy = 35 →
  matthew = rebecca + 2 →
  freddy = matthew + 4 →
  ∃ (x : ℕ), matthew = 4 * x ∧ rebecca = 5 * x ∧ freddy = 7 * x →
  freddy = 15 := by
sorry

end NUMINAMATH_CALUDE_freddy_age_l3214_321401


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3214_321427

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / (3 - i)
  Complex.im z = -1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3214_321427


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3214_321433

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define the relationship between ¬p and ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) := by
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3214_321433


namespace NUMINAMATH_CALUDE_total_hair_cut_equals_41_l3214_321441

/-- Represents the hair length of a person before and after a haircut. -/
structure Haircut where
  original : ℕ  -- Original hair length in inches
  cut : ℕ       -- Amount of hair cut off in inches

/-- Calculates the total amount of hair cut off from multiple haircuts. -/
def total_hair_cut (haircuts : List Haircut) : ℕ :=
  haircuts.map (·.cut) |>.sum

/-- Theorem stating that the total hair cut off from Isabella, Damien, and Ella is 41 inches. -/
theorem total_hair_cut_equals_41 : 
  let isabella : Haircut := { original := 18, cut := 9 }
  let damien : Haircut := { original := 24, cut := 12 }
  let ella : Haircut := { original := 30, cut := 20 }
  total_hair_cut [isabella, damien, ella] = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_hair_cut_equals_41_l3214_321441


namespace NUMINAMATH_CALUDE_dog_tail_length_l3214_321469

/-- Represents the length of a dog's body parts and total length --/
structure DogMeasurements where
  body : ℝ
  head : ℝ
  tail : ℝ
  total : ℝ

/-- Theorem stating the tail length of a dog given specific proportions --/
theorem dog_tail_length (d : DogMeasurements) 
  (h1 : d.tail = d.body / 2)
  (h2 : d.head = d.body / 6)
  (h3 : d.total = d.body + d.head + d.tail)
  (h4 : d.total = 30) : 
  d.tail = 9 := by
  sorry

end NUMINAMATH_CALUDE_dog_tail_length_l3214_321469


namespace NUMINAMATH_CALUDE_company_supervisors_l3214_321415

/-- Represents the number of workers per team lead -/
def workers_per_team_lead : ℕ := 10

/-- Represents the number of team leads per supervisor -/
def team_leads_per_supervisor : ℕ := 3

/-- Represents the total number of workers in the company -/
def total_workers : ℕ := 390

/-- Calculates the number of supervisors in the company -/
def calculate_supervisors : ℕ :=
  (total_workers / workers_per_team_lead) / team_leads_per_supervisor

theorem company_supervisors :
  calculate_supervisors = 13 := by sorry

end NUMINAMATH_CALUDE_company_supervisors_l3214_321415


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3214_321424

theorem simplify_complex_fraction (a b : ℝ) 
  (h1 : a + b ≠ 0) 
  (h2 : a - 2*b ≠ 0) 
  (h3 : a^2 - b^2 ≠ 0) 
  (h4 : a^2 - 4*a*b + 4*b^2 ≠ 0) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3214_321424


namespace NUMINAMATH_CALUDE_percentage_subtraction_equivalence_l3214_321453

theorem percentage_subtraction_equivalence (a : ℝ) : 
  a - (0.05 * a) = 0.95 * a := by sorry

end NUMINAMATH_CALUDE_percentage_subtraction_equivalence_l3214_321453


namespace NUMINAMATH_CALUDE_files_remaining_after_deletion_l3214_321403

theorem files_remaining_after_deletion (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 16)
  (h2 : video_files = 48)
  (h3 : deleted_files = 30) :
  music_files + video_files - deleted_files = 34 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_after_deletion_l3214_321403


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3214_321467

theorem tenth_term_of_sequence (a : ℕ → ℚ) :
  (∀ n : ℕ, a n = (-1)^(n+1) * (2*n) / (2*n+1)) →
  a 10 = -20 / 21 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3214_321467


namespace NUMINAMATH_CALUDE_min_balls_drawn_l3214_321479

/-- Given a bag with 10 white balls, 5 black balls, and 4 blue balls,
    the minimum number of balls to be drawn to ensure at least 2 balls of each color is 17. -/
theorem min_balls_drawn (white : Nat) (black : Nat) (blue : Nat)
    (h_white : white = 10)
    (h_black : black = 5)
    (h_blue : blue = 4) :
    (∃ n : Nat, n = 17 ∧
      ∀ m : Nat, m < n →
        ¬(∃ w b l : Nat, w ≥ 2 ∧ b ≥ 2 ∧ l ≥ 2 ∧
          w + b + l = m ∧ w ≤ white ∧ b ≤ black ∧ l ≤ blue)) :=
by sorry

end NUMINAMATH_CALUDE_min_balls_drawn_l3214_321479


namespace NUMINAMATH_CALUDE_no_prime_sum_53_l3214_321472

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- State the theorem
theorem no_prime_sum_53 : ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 53 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_l3214_321472


namespace NUMINAMATH_CALUDE_cow_spots_problem_l3214_321419

theorem cow_spots_problem (left_spots right_spots total_spots additional_spots : ℕ) :
  left_spots = 16 →
  right_spots = 3 * left_spots + additional_spots →
  total_spots = left_spots + right_spots →
  total_spots = 71 →
  additional_spots = 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_spots_problem_l3214_321419


namespace NUMINAMATH_CALUDE_triangle_count_for_2016_30_triangle_count_formula_l3214_321487

/-- Represents the number of non-overlapping triangles in a mesh region --/
def f (m n : ℕ) : ℕ := 2 * m - n - 2

/-- The theorem states that for 2016 points forming a 30-gon convex hull, 
    the number of non-overlapping triangles is 4000 --/
theorem triangle_count_for_2016_30 :
  f 2016 30 = 4000 := by
  sorry

/-- A more general theorem about the formula for f(m, n) --/
theorem triangle_count_formula {m n : ℕ} (h_m : m > 2) (h_n : 3 ≤ n ∧ n ≤ m) :
  f m n = 2 * m - n - 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_for_2016_30_triangle_count_formula_l3214_321487


namespace NUMINAMATH_CALUDE_complex_modulus_l3214_321499

theorem complex_modulus (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3214_321499


namespace NUMINAMATH_CALUDE_product_eleven_cubed_sum_l3214_321471

theorem product_eleven_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 11^3 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 133 := by sorry

end NUMINAMATH_CALUDE_product_eleven_cubed_sum_l3214_321471


namespace NUMINAMATH_CALUDE_function_difference_bound_l3214_321408

theorem function_difference_bound 
  (f : Set.Icc 0 1 → ℝ) 
  (h1 : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h2 : ∀ (x y : Set.Icc 0 1), x ≠ y → |f x - f y| < |x.val - y.val|) :
  ∀ (x y : Set.Icc 0 1), |f x - f y| < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_function_difference_bound_l3214_321408


namespace NUMINAMATH_CALUDE_equation_solution_l3214_321447

theorem equation_solution :
  ∃ x : ℝ, x ≠ -2 ∧ (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3214_321447


namespace NUMINAMATH_CALUDE_certain_number_problem_l3214_321420

theorem certain_number_problem : ∃ x : ℚ, (1206 / 3 : ℚ) = 3 * x ∧ x = 134 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3214_321420


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3214_321454

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 12/7
  let r : ℚ := a₂ / a₁
  r = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3214_321454


namespace NUMINAMATH_CALUDE_abc_inequality_l3214_321468

theorem abc_inequality (a b c x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a^x = b*c) (eq2 : b^y = c*a) (eq3 : c^z = a*b) :
  1/(2+x) + 1/(2+y) + 1/(2+z) ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3214_321468


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3214_321414

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 323) ∧ 
  (has_no_small_prime_factors 323) ∧ 
  (∀ m : ℕ, m < 323 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3214_321414


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3214_321438

/-- Calculates the price of a stock given the income, dividend rate, and investment amount. -/
theorem stock_price_calculation (income : ℝ) (dividend_rate : ℝ) (investment : ℝ) :
  income = 450 →
  dividend_rate = 0.1 →
  investment = 4860 →
  (investment / (income / dividend_rate)) * 100 = 108 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3214_321438


namespace NUMINAMATH_CALUDE_largest_quantity_l3214_321405

def D : ℚ := 2008/2007 + 2008/2009
def E : ℚ := 2008/2009 + 2010/2009
def F : ℚ := 2009/2008 + 2009/2010 - 1/2009

theorem largest_quantity : D > E ∧ D > F := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l3214_321405


namespace NUMINAMATH_CALUDE_surface_area_is_39_l3214_321489

/-- Represents the structure made of unit cubes -/
structure CubeStructure where
  total_cubes : Nat
  pyramid_base : Nat
  extension_height : Nat

/-- Calculates the exposed surface area of the cube structure -/
def exposed_surface_area (s : CubeStructure) : Nat :=
  sorry

/-- The theorem stating that the exposed surface area of the given structure is 39 square meters -/
theorem surface_area_is_39 (s : CubeStructure) 
  (h1 : s.total_cubes = 18)
  (h2 : s.pyramid_base = 3)
  (h3 : s.extension_height = 4) : 
  exposed_surface_area s = 39 :=
sorry

end NUMINAMATH_CALUDE_surface_area_is_39_l3214_321489


namespace NUMINAMATH_CALUDE_max_four_digit_prime_product_l3214_321407

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem max_four_digit_prime_product :
  ∃ (m x y z : Nat),
    isPrime x ∧ isPrime y ∧ isPrime z ∧
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    isPrime (10 * x + y) ∧
    isPrime (10 * z + x) ∧
    m = x * y * (10 * x + y) ∧
    m ≥ 1000 ∧ m < 10000 ∧
    (∀ (m' x' y' z' : Nat),
      isPrime x' ∧ isPrime y' ∧ isPrime z' ∧
      x' < 10 ∧ y' < 10 ∧ z' < 10 ∧
      x' ≠ y' ∧ y' ≠ z' ∧ x' ≠ z' ∧
      isPrime (10 * x' + y') ∧
      isPrime (10 * z' + x') ∧
      m' = x' * y' * (10 * x' + y') ∧
      m' ≥ 1000 ∧ m' < 10000 →
      m' ≤ m) ∧
    m = 1533 :=
by sorry

end NUMINAMATH_CALUDE_max_four_digit_prime_product_l3214_321407


namespace NUMINAMATH_CALUDE_alphametic_equation_impossible_l3214_321434

theorem alphametic_equation_impossible : 
  ¬ ∃ (K O T U Ch E N W Y : ℕ), 
    (K ∈ Finset.range 10) ∧ 
    (O ∈ Finset.range 10) ∧ 
    (T ∈ Finset.range 10) ∧ 
    (U ∈ Finset.range 10) ∧ 
    (Ch ∈ Finset.range 10) ∧ 
    (E ∈ Finset.range 10) ∧ 
    (N ∈ Finset.range 10) ∧ 
    (W ∈ Finset.range 10) ∧ 
    (Y ∈ Finset.range 10) ∧ 
    (K ≠ O) ∧ (K ≠ T) ∧ (K ≠ U) ∧ (K ≠ Ch) ∧ (K ≠ E) ∧ (K ≠ N) ∧ (K ≠ W) ∧ (K ≠ Y) ∧
    (O ≠ T) ∧ (O ≠ U) ∧ (O ≠ Ch) ∧ (O ≠ E) ∧ (O ≠ N) ∧ (O ≠ W) ∧ (O ≠ Y) ∧
    (T ≠ U) ∧ (T ≠ Ch) ∧ (T ≠ E) ∧ (T ≠ N) ∧ (T ≠ W) ∧ (T ≠ Y) ∧
    (U ≠ Ch) ∧ (U ≠ E) ∧ (U ≠ N) ∧ (U ≠ W) ∧ (U ≠ Y) ∧
    (Ch ≠ E) ∧ (Ch ≠ N) ∧ (Ch ≠ W) ∧ (Ch ≠ Y) ∧
    (E ≠ N) ∧ (E ≠ W) ∧ (E ≠ Y) ∧
    (N ≠ W) ∧ (N ≠ Y) ∧
    (W ≠ Y) ∧
    (K * O * T = U * Ch * E * N * W * Y) :=
sorry

end NUMINAMATH_CALUDE_alphametic_equation_impossible_l3214_321434


namespace NUMINAMATH_CALUDE_smallest_d_for_inverse_l3214_321474

def g (x : ℝ) : ℝ := (x - 3)^2 - 8

theorem smallest_d_for_inverse :
  ∀ d : ℝ, (∀ x y, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_for_inverse_l3214_321474


namespace NUMINAMATH_CALUDE_length_BC_l3214_321484

/-- Right triangles ABC and ABD with specific properties -/
structure RightTriangles where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- D is on the x-axis
  h_D_on_x : D.2 = 0
  -- C is directly below A on the x-axis
  h_C_below_A : C.1 = A.1
  -- Distances
  h_AD : dist A D = 26
  h_BD : dist B D = 10
  h_AC : dist A C = 24
  -- ABC is a right triangle
  h_ABC_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- ABD is a right triangle
  h_ABD_right : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0

/-- The length of BC in the given configuration of right triangles -/
theorem length_BC (t : RightTriangles) : dist t.B t.C = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_length_BC_l3214_321484


namespace NUMINAMATH_CALUDE_cow_spots_count_l3214_321412

/-- The number of spots on a cow with given left and right side spot counts. -/
def total_spots (left : ℕ) (right : ℕ) : ℕ := left + right

/-- The number of spots on the right side of the cow, given the number on the left side. -/
def right_spots (left : ℕ) : ℕ := 3 * left + 7

theorem cow_spots_count :
  let left := 16
  let right := right_spots left
  total_spots left right = 71 := by
  sorry

end NUMINAMATH_CALUDE_cow_spots_count_l3214_321412


namespace NUMINAMATH_CALUDE_vector_magnitude_equivalence_l3214_321450

/-- Given non-zero, non-collinear vectors a and b, prove that |a| = |b| if and only if |a + 2b| = |2a + b| -/
theorem vector_magnitude_equivalence {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) (hnc : ¬ Collinear ℝ {0, a, b}) :
  ‖a‖ = ‖b‖ ↔ ‖a + 2 • b‖ = ‖2 • a + b‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_equivalence_l3214_321450


namespace NUMINAMATH_CALUDE_complex_collinearity_l3214_321444

/-- A complex number represented as a point in the plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- Check if three ComplexPoints are collinear -/
def areCollinear (p q r : ComplexPoint) : Prop :=
  ∃ k : ℝ, (r.re - p.re, r.im - p.im) = k • (q.re - p.re, q.im - p.im)

theorem complex_collinearity :
  ∃! b : ℝ, areCollinear
    (ComplexPoint.mk 3 (-5))
    (ComplexPoint.mk 1 (-1))
    (ComplexPoint.mk (-2) b) ∧
  b = 5 := by sorry

end NUMINAMATH_CALUDE_complex_collinearity_l3214_321444


namespace NUMINAMATH_CALUDE_chrysanthemum_arrangement_count_l3214_321464

/-- Represents the number of pots for each color of chrysanthemum -/
structure ChrysanthemumCounts where
  white : Nat
  yellow : Nat
  red : Nat

/-- Calculates the number of arrangements for chrysanthemums with given conditions -/
def chrysanthemumArrangements (counts : ChrysanthemumCounts) : Nat :=
  sorry

/-- The main theorem stating the number of arrangements for the given problem -/
theorem chrysanthemum_arrangement_count :
  let counts : ChrysanthemumCounts := { white := 2, yellow := 2, red := 1 }
  chrysanthemumArrangements counts = 16 := by
  sorry

end NUMINAMATH_CALUDE_chrysanthemum_arrangement_count_l3214_321464


namespace NUMINAMATH_CALUDE_vertex_not_at_minus_two_one_l3214_321457

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The given parabola y = -2(x-2)^2 + 1 --/
def givenParabola : Parabola := { a := -2, h := 2, k := 1 }

/-- The vertex of a parabola --/
def vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

/-- Theorem stating that the vertex of the given parabola is not at (-2,1) --/
theorem vertex_not_at_minus_two_one :
  vertex givenParabola ≠ (-2, 1) := by
  sorry

end NUMINAMATH_CALUDE_vertex_not_at_minus_two_one_l3214_321457


namespace NUMINAMATH_CALUDE_work_completion_proof_l3214_321491

/-- The number of days it takes for person B to complete the work alone -/
def person_b_days : ℝ := 45

/-- The fraction of work completed by both persons in 5 days -/
def work_completed_together : ℝ := 0.2777777777777778

/-- The number of days they work together -/
def days_worked_together : ℝ := 5

/-- The number of days it takes for person A to complete the work alone -/
def person_a_days : ℝ := 30

theorem work_completion_proof :
  (days_worked_together * (1 / person_a_days + 1 / person_b_days) = work_completed_together) →
  person_a_days = 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3214_321491


namespace NUMINAMATH_CALUDE_first_day_of_month_l3214_321473

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

theorem first_day_of_month (d : DayOfWeek) :
  advanceDay d 29 = DayOfWeek.Monday → d = DayOfWeek.Sunday :=
by sorry


end NUMINAMATH_CALUDE_first_day_of_month_l3214_321473


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l3214_321411

-- Define the slopes and y-intercept
def m₁ : ℚ := 8
def m₂ : ℚ := 4
def b : ℚ := 5

-- Define the x-intercepts
def s : ℚ := -b / m₁
def t : ℚ := -b / m₂

-- Theorem statement
theorem x_intercept_ratio :
  s / t = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l3214_321411
