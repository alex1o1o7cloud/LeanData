import Mathlib

namespace NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l3980_398055

theorem smallest_angle_in_ratio_triangle (α β γ : Real) : 
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- Angles are positive
  β = 2 * α ∧ γ = 3 * α →  -- Angle ratio is 1 : 2 : 3
  α + β + γ = π →         -- Sum of angles in a triangle
  α = π / 6 := by
    sorry

end NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l3980_398055


namespace NUMINAMATH_CALUDE_sum_of_s_and_u_l3980_398045

-- Define complex numbers
variable (p q r s t u : ℝ)

-- Define the conditions
def complex_sum_condition (p q r s t u : ℝ) : Prop :=
  Complex.mk (p + r + t) (q + s + u) = Complex.I * (-7)

-- Theorem statement
theorem sum_of_s_and_u 
  (h1 : q = 5)
  (h2 : t = -p - r)
  (h3 : complex_sum_condition p q r s t u) :
  s + u = -12 := by sorry

end NUMINAMATH_CALUDE_sum_of_s_and_u_l3980_398045


namespace NUMINAMATH_CALUDE_common_tangent_sum_l3980_398021

/-- Given two curves f and g with a common tangent at their intersection point (0, m),
    prove that the sum of their coefficients a and b is 1. -/
theorem common_tangent_sum (a b m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * Real.cos x
  let g : ℝ → ℝ := λ x ↦ x^2 + b*x + 1
  let f' : ℝ → ℝ := λ x ↦ -a * Real.sin x
  let g' : ℝ → ℝ := λ x ↦ 2*x + b
  (f 0 = g 0) ∧ (f' 0 = g' 0) → a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l3980_398021


namespace NUMINAMATH_CALUDE_min_degree_g_l3980_398079

/-- Given polynomials f, g, and h satisfying the equation 4f + 5g = h, 
    with deg(f) = 10 and deg(h) = 12, the minimum possible degree of g is 12 -/
theorem min_degree_g (f g h : Polynomial ℝ) 
  (eq : 4 • f + 5 • g = h) 
  (deg_f : Polynomial.degree f = 10)
  (deg_h : Polynomial.degree h = 12) :
  Polynomial.degree g ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_degree_g_l3980_398079


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3980_398096

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 8 * x + 1 = 0

-- Define the root form
def root_form (x m n p : ℝ) : Prop := x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p

-- Theorem statement
theorem quadratic_root_value :
  ∃ (m n p : ℕ+),
    (∀ x : ℝ, quadratic_equation x ↔ root_form x m n p) ∧
    Nat.gcd (Nat.gcd m.val n.val) p.val = 1 ∧
    n = 13 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3980_398096


namespace NUMINAMATH_CALUDE_inequality_proof_l3980_398049

theorem inequality_proof (x : ℝ) (h1 : x > 4/3) (h2 : x ≠ -5) (h3 : x ≠ 4/3) :
  (6*x^2 + 18*x - 60) / ((3*x - 4)*(x + 5)) < 2 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3980_398049


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3980_398037

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Parametric representation of the plane -/
def parametricPlane (s t : ℝ) : Point3D :=
  { x := 2 + 2*s - 3*t
  , y := 1 - 2*s
  , z := 4 + 3*s + 4*t }

/-- The plane equation we want to prove -/
def targetPlane : Plane :=
  { a := 8
  , b := 17
  , c := 6
  , d := -57 }

theorem plane_equation_proof :
  (∀ s t : ℝ, pointOnPlane (parametricPlane s t) targetPlane) ∧
  targetPlane.a > 0 ∧
  Int.gcd (Int.natAbs targetPlane.a) (Int.gcd (Int.natAbs targetPlane.b) (Int.gcd (Int.natAbs targetPlane.c) (Int.natAbs targetPlane.d))) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3980_398037


namespace NUMINAMATH_CALUDE_four_half_planes_theorem_l3980_398009

-- Define a half-plane
def HalfPlane : Type := ℝ × ℝ → Prop

-- Define a set of four half-planes
def FourHalfPlanes : Type := Fin 4 → HalfPlane

-- Define the property of covering the entire plane
def CoversPlane (planes : FourHalfPlanes) : Prop :=
  ∀ (x y : ℝ), ∃ (i : Fin 4), planes i (x, y)

-- Define the property of a subset of three half-planes covering the entire plane
def ThreeCoversPlane (planes : FourHalfPlanes) : Prop :=
  ∃ (i j k : Fin 4) (h : i ≠ j ∧ j ≠ k ∧ i ≠ k),
    ∀ (x y : ℝ), planes i (x, y) ∨ planes j (x, y) ∨ planes k (x, y)

-- The theorem to be proved
theorem four_half_planes_theorem (planes : FourHalfPlanes) :
  CoversPlane planes → ThreeCoversPlane planes :=
by
  sorry

end NUMINAMATH_CALUDE_four_half_planes_theorem_l3980_398009


namespace NUMINAMATH_CALUDE_smallest_cube_ending_368_l3980_398034

theorem smallest_cube_ending_368 : 
  ∃ (n : ℕ), n > 0 ∧ n^3 ≡ 368 [MOD 1000] ∧ ∀ (m : ℕ), m > 0 ∧ m^3 ≡ 368 [MOD 1000] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_368_l3980_398034


namespace NUMINAMATH_CALUDE_triangle_area_l3980_398052

theorem triangle_area (a b c : ℝ) (h1 : c^2 = a^2 + b^2 - 2*a*b + 6) (h2 : 0 < a ∧ 0 < b ∧ 0 < c) : 
  (1/2) * a * b * Real.sin (π/3) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3980_398052


namespace NUMINAMATH_CALUDE_haleys_trees_l3980_398091

theorem haleys_trees (dead : ℕ) (survived : ℕ) : 
  dead = 6 → 
  survived = dead + 1 → 
  dead + survived = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_haleys_trees_l3980_398091


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3980_398065

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (weighted_diff_eq : 2 * y - 3 * x = 10) :
  |y - x| = 12 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3980_398065


namespace NUMINAMATH_CALUDE_journey_time_equation_l3980_398027

theorem journey_time_equation (x : ℝ) (h : x > 0) : 
  let distance : ℝ := 50
  let taxi_speed : ℝ := x + 15
  let bus_speed : ℝ := x
  let taxi_time : ℝ := distance / taxi_speed
  let bus_time : ℝ := distance / bus_speed
  taxi_time = 2/3 * bus_time → distance / taxi_speed = 2/3 * (distance / bus_speed) :=
by sorry

end NUMINAMATH_CALUDE_journey_time_equation_l3980_398027


namespace NUMINAMATH_CALUDE_hexagon_segment_probability_l3980_398060

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of short diagonals in a regular hexagon -/
def num_short_diagonals : ℕ := 6

/-- The number of long diagonals in a regular hexagon -/
def num_long_diagonals : ℕ := 3

/-- The total number of elements in set S -/
def total_elements : ℕ := num_sides + num_short_diagonals + num_long_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := 11 / 35

theorem hexagon_segment_probability :
  (num_sides * (num_sides - 1) + num_short_diagonals * (num_short_diagonals - 1) + num_long_diagonals * (num_long_diagonals - 1)) / (total_elements * (total_elements - 1)) = prob_same_length :=
sorry

end NUMINAMATH_CALUDE_hexagon_segment_probability_l3980_398060


namespace NUMINAMATH_CALUDE_base_9_to_3_conversion_l3980_398047

def to_base_3 (n : ℕ) : ℕ := sorry

def from_base_9 (n : ℕ) : ℕ := sorry

theorem base_9_to_3_conversion :
  to_base_3 (from_base_9 745) = 211112 := by sorry

end NUMINAMATH_CALUDE_base_9_to_3_conversion_l3980_398047


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3980_398022

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 2 * x^2 + 8 * x + 6 = a * (x - h)^2 + k) → a + h + k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3980_398022


namespace NUMINAMATH_CALUDE_unique_four_digit_number_with_reverse_property_l3980_398085

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def reverse_digits (n : ℕ) : ℕ :=
  let d₁ := n / 1000
  let d₂ := (n / 100) % 10
  let d₃ := (n / 10) % 10
  let d₄ := n % 10
  d₄ * 1000 + d₃ * 100 + d₂ * 10 + d₁

theorem unique_four_digit_number_with_reverse_property :
  ∃! n : ℕ, is_four_digit n ∧ n + 7182 = reverse_digits n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_with_reverse_property_l3980_398085


namespace NUMINAMATH_CALUDE_water_tank_theorem_l3980_398039

/-- Represents the water tank problem --/
def WaterTankProblem (maxCapacity initialLossRate initialLossDuration
                      secondaryLossRate secondaryLossDuration
                      refillRate refillDuration : ℕ) : Prop :=
  let initialLoss := initialLossRate * initialLossDuration
  let secondaryLoss := secondaryLossRate * secondaryLossDuration
  let totalLoss := initialLoss + secondaryLoss
  let remainingWater := maxCapacity - totalLoss
  let refillAmount := refillRate * refillDuration
  let finalWaterAmount := remainingWater + refillAmount
  maxCapacity - finalWaterAmount = 140000

/-- The water tank theorem --/
theorem water_tank_theorem : WaterTankProblem 350000 32000 5 10000 10 40000 3 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_theorem_l3980_398039


namespace NUMINAMATH_CALUDE_function_growth_l3980_398088

theorem function_growth (f : ℕ+ → ℝ) 
  (h : ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (h4 : f 4 ≥ 25) :
  ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 := by
sorry

end NUMINAMATH_CALUDE_function_growth_l3980_398088


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3980_398015

/-- 
Given a regular polygon where each exterior angle measures 40°, 
the sum of its interior angles is 1260°.
-/
theorem sum_interior_angles_regular_polygon (n : ℕ) 
  (h_exterior : (360 : ℝ) / n = 40) : 
  (n - 2 : ℝ) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3980_398015


namespace NUMINAMATH_CALUDE_sin_cos_sum_identity_l3980_398093

open Real

theorem sin_cos_sum_identity : 
  sin (15 * π / 180) * cos (75 * π / 180) + cos (15 * π / 180) * sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_identity_l3980_398093


namespace NUMINAMATH_CALUDE_rectangle_area_l3980_398046

/-- A rectangle with diagonal d and length three times its width has area 3d²/10 -/
theorem rectangle_area (d : ℝ) (w : ℝ) (h : w > 0) : 
  w^2 + (3*w)^2 = d^2 → w * (3*w) = (3 * d^2) / 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3980_398046


namespace NUMINAMATH_CALUDE_min_panels_for_intensity_reduction_l3980_398090

/-- Represents the reduction factor of light intensity when passing through a glass panel -/
def reduction_factor : ℝ := 0.9

/-- Calculates the light intensity after passing through a number of panels -/
def intensity_after_panels (a : ℝ) (x : ℕ) : ℝ := a * reduction_factor ^ x

/-- Theorem stating the minimum number of panels required to reduce light intensity to less than 1/11 of original -/
theorem min_panels_for_intensity_reduction (a : ℝ) (h : a > 0) :
  ∃ x : ℕ, (∀ y : ℕ, y < x → intensity_after_panels a y ≥ a / 11) ∧
           intensity_after_panels a x < a / 11 :=
by sorry

end NUMINAMATH_CALUDE_min_panels_for_intensity_reduction_l3980_398090


namespace NUMINAMATH_CALUDE_train_length_calculation_l3980_398008

/-- Calculates the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length_calculation (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 → platform_length = 280 → crossing_time = 26 →
  (train_speed * 1000 / 3600) * crossing_time - platform_length = 240 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3980_398008


namespace NUMINAMATH_CALUDE_percentage_problem_l3980_398074

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x * (x / 100) = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3980_398074


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3980_398016

/-- Proves that in a rhombus with one diagonal of 160 m and an area of 5600 m², 
    the length of the other diagonal is 70 m. -/
theorem rhombus_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 160 → area = 5600 → area = (d1 * d2) / 2 → d2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3980_398016


namespace NUMINAMATH_CALUDE_probability_1AECD_l3980_398092

-- Define the structure of the license plate
structure LicensePlate where
  digit : Fin 10
  vowel1 : Fin 5
  vowel2 : Fin 5
  consonant1 : Fin 21
  consonant2 : Fin 21
  different_consonants : consonant1 ≠ consonant2

-- Define the total number of possible license plates
def total_plates : ℕ := 10 * 5 * 5 * 21 * 20

-- Define the probability of a specific plate
def probability_specific_plate : ℚ := 1 / total_plates

-- Theorem to prove
theorem probability_1AECD :
  probability_specific_plate = 1 / 105000 :=
sorry

end NUMINAMATH_CALUDE_probability_1AECD_l3980_398092


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3980_398051

open Complex

theorem complex_modulus_problem (z : ℂ) (h : (1 - I) * z = 2 * I) : abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3980_398051


namespace NUMINAMATH_CALUDE_opposite_sqrt_81_l3980_398001

theorem opposite_sqrt_81 : -(Real.sqrt (Real.sqrt 81)) = -9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sqrt_81_l3980_398001


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3980_398070

theorem complex_on_imaginary_axis (a : ℝ) :
  let z : ℂ := Complex.mk (a^2 - 2*a) (a^2 - a - 2)
  (Complex.re z = 0) → (a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3980_398070


namespace NUMINAMATH_CALUDE_driving_time_to_school_l3980_398097

theorem driving_time_to_school 
  (total_hours : ℕ) 
  (school_days : ℕ) 
  (drives_both_ways : Bool) : 
  total_hours = 50 → 
  school_days = 75 → 
  drives_both_ways = true → 
  (total_hours * 60) / (school_days * 2) = 20 := by
sorry

end NUMINAMATH_CALUDE_driving_time_to_school_l3980_398097


namespace NUMINAMATH_CALUDE_base6_multiplication_addition_l3980_398048

/-- Converts a base 6 number represented as a list of digits to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6, represented as a list of digits -/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- The main theorem statement -/
theorem base6_multiplication_addition :
  let a := base6ToBase10 [1, 1, 1]  -- 111₆
  let b := 2
  let c := base6ToBase10 [2, 0, 2]  -- 202₆
  base10ToBase6 (a * b + c) = [4, 2, 4] := by
  sorry


end NUMINAMATH_CALUDE_base6_multiplication_addition_l3980_398048


namespace NUMINAMATH_CALUDE_unique_equal_intercept_line_l3980_398040

/-- A line with equal intercepts on both axes passing through (2,3) -/
def EqualInterceptLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ a : ℝ, p.1 + p.2 = a ∧ (2 : ℝ) + (3 : ℝ) = a}

/-- The theorem stating that there is exactly one line with equal intercepts passing through (2,3) -/
theorem unique_equal_intercept_line : 
  ∃! a : ℝ, (2 : ℝ) + (3 : ℝ) = a ∧ EqualInterceptLine = {p : ℝ × ℝ | p.1 + p.2 = a} :=
by sorry

end NUMINAMATH_CALUDE_unique_equal_intercept_line_l3980_398040


namespace NUMINAMATH_CALUDE_triangle_side_length_l3980_398063

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Define the triangle
  A + B + C = Real.pi →  -- Sum of angles in a triangle is π radians
  A = Real.pi / 6 →      -- 30° in radians
  C = 7 * Real.pi / 12 → -- 105° in radians
  b = 8 →                -- Given side length
  -- Law of Sines
  b / Real.sin B = a / Real.sin A →
  -- Conclusion
  a = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3980_398063


namespace NUMINAMATH_CALUDE_coefficient_a7_equals_negative_eight_l3980_398043

/-- Given x^8 = a₀ + a₁(x+1) + a₂(x+1)² + ... + a₈(x+1)⁸, prove that a₇ = -8 -/
theorem coefficient_a7_equals_negative_eight (x : ℝ) (a : Fin 9 → ℝ) :
  x^8 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + 
        a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 + a 8 * (x + 1)^8 →
  a 7 = -8 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a7_equals_negative_eight_l3980_398043


namespace NUMINAMATH_CALUDE_gcd_30_and_number_l3980_398024

theorem gcd_30_and_number (n : ℕ) : 
  70 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 30 n = 10 → n = 70 ∨ n = 80 ∨ n = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_30_and_number_l3980_398024


namespace NUMINAMATH_CALUDE_fraction_simplification_l3980_398028

theorem fraction_simplification (y : ℚ) (h : y = 77) : (7 * y + 77) / 77 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3980_398028


namespace NUMINAMATH_CALUDE_square_minus_double_eq_one_implies_double_square_minus_quadruple_l3980_398086

theorem square_minus_double_eq_one_implies_double_square_minus_quadruple (m : ℝ) :
  m^2 - 2*m = 1 → 2*m^2 - 4*m = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_double_eq_one_implies_double_square_minus_quadruple_l3980_398086


namespace NUMINAMATH_CALUDE_young_bonnet_ratio_l3980_398098

/-- Mrs. Young's bonnet making problem -/
theorem young_bonnet_ratio :
  let monday_bonnets : ℕ := 10
  let thursday_bonnets : ℕ := monday_bonnets + 5
  let friday_bonnets : ℕ := thursday_bonnets - 5
  let total_orphanages : ℕ := 5
  let bonnets_per_orphanage : ℕ := 11
  let total_bonnets : ℕ := total_orphanages * bonnets_per_orphanage
  let tues_wed_bonnets : ℕ := total_bonnets - (monday_bonnets + thursday_bonnets + friday_bonnets)
  tues_wed_bonnets / monday_bonnets = 2 := by
  sorry

end NUMINAMATH_CALUDE_young_bonnet_ratio_l3980_398098


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3980_398076

theorem complex_equation_sum (x y : ℝ) : 
  (x + 2 * Complex.I) * Complex.I = y - Complex.I⁻¹ → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3980_398076


namespace NUMINAMATH_CALUDE_bread_loaves_l3980_398081

theorem bread_loaves (slices_per_loaf : ℕ) (num_friends : ℕ) (slices_per_friend : ℕ) :
  slices_per_loaf = 15 →
  num_friends = 10 →
  slices_per_friend = 6 →
  (num_friends * slices_per_friend) / slices_per_loaf = 4 :=
by sorry

end NUMINAMATH_CALUDE_bread_loaves_l3980_398081


namespace NUMINAMATH_CALUDE_circle_c_equation_l3980_398013

/-- A circle C satisfying the given conditions -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  tangent_y_axis : center.1 = radius
  chord_length : 4 * radius ^ 2 - center.1 ^ 2 = 12
  center_on_line : center.2 = 1/2 * center.1

/-- The equation of the circle C -/
def circle_equation (c : CircleC) (x y : ℝ) : Prop :=
  (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2

/-- Theorem stating that the circle C has the equation (x-2)² + (y-1)² = 4 -/
theorem circle_c_equation (c : CircleC) :
  ∀ x y, circle_equation c x y ↔ (x - 2) ^ 2 + (y - 1) ^ 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_c_equation_l3980_398013


namespace NUMINAMATH_CALUDE_parallel_distinct_iff_a_eq_3_l3980_398061

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop
  line2 : ℝ × ℝ → Prop
  line1_def : ∀ x y, line1 (x, y) ↔ a * x + 2 * y + 3 * a = 0
  line2_def : ∀ x y, line2 (x, y) ↔ 3 * x + (a - 1) * y = a - 7

/-- The lines are parallel -/
def parallel (tl : TwoLines) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, tl.line1 (x, y) ↔ tl.line2 (k * x, k * y)

/-- The lines are distinct -/
def distinct (tl : TwoLines) : Prop :=
  ∃ p, tl.line1 p ∧ ¬tl.line2 p

/-- The main theorem -/
theorem parallel_distinct_iff_a_eq_3 (tl : TwoLines) :
  parallel tl ∧ distinct tl ↔ tl.a = 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_distinct_iff_a_eq_3_l3980_398061


namespace NUMINAMATH_CALUDE_composite_shape_area_l3980_398023

/-- The area of a composite shape consisting of three rectangles --/
def composite_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) : ℕ :=
  rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height

/-- Theorem stating that the area of the given composite shape is 77 square units --/
theorem composite_shape_area : composite_area 10 4 4 7 3 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_area_l3980_398023


namespace NUMINAMATH_CALUDE_samuels_birds_berries_l3980_398066

/-- The number of berries a single bird eats per day -/
def berries_per_day : ℕ := 7

/-- The number of birds Samuel has -/
def samuels_birds : ℕ := 5

/-- The number of days we're considering -/
def days : ℕ := 4

/-- Theorem: Samuel's birds eat 140 berries in 4 days -/
theorem samuels_birds_berries : 
  berries_per_day * samuels_birds * days = 140 := by
  sorry

end NUMINAMATH_CALUDE_samuels_birds_berries_l3980_398066


namespace NUMINAMATH_CALUDE_jessica_shells_count_l3980_398005

def seashell_problem (sally_shells tom_shells total_shells : ℕ) : Prop :=
  ∃ jessica_shells : ℕ, 
    sally_shells + tom_shells + jessica_shells = total_shells

theorem jessica_shells_count (sally_shells tom_shells total_shells : ℕ) 
  (h : seashell_problem sally_shells tom_shells total_shells) :
  ∃ jessica_shells : ℕ, jessica_shells = total_shells - (sally_shells + tom_shells) :=
by
  sorry

#check jessica_shells_count 9 7 21

end NUMINAMATH_CALUDE_jessica_shells_count_l3980_398005


namespace NUMINAMATH_CALUDE_system_solution_l3980_398068

theorem system_solution (a₁ a₂ a₃ a₄ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) :
  ∃! (x₁ x₂ x₃ x₄ : ℝ),
    (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
    (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
    (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
    (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1) ∧
    (x₁ = x₂) ∧ (x₂ = x₃) ∧ (x₃ = x₄) ∧
    (x₁ = 1 / (3 * a₁ - (a₂ + a₃ + a₄))) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3980_398068


namespace NUMINAMATH_CALUDE_inverse_cube_relation_l3980_398019

/-- Given that y varies inversely as the cube of x, prove that x = ∛3 when y = 18, 
    given that y = 2 when x = 3. -/
theorem inverse_cube_relation (x y : ℝ) (k : ℝ) (h1 : y * x^3 = k) 
  (h2 : 2 * 3^3 = k) (h3 : 18 * x^3 = k) : x = (3 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_inverse_cube_relation_l3980_398019


namespace NUMINAMATH_CALUDE_largest_negative_angle_l3980_398067

-- Define the function for angles with the same terminal side as -2002°
def sameTerminalSide (k : ℤ) : ℝ := k * 360 - 2002

-- Theorem statement
theorem largest_negative_angle :
  ∃ (k : ℤ), sameTerminalSide k = -202 ∧
  ∀ (m : ℤ), sameTerminalSide m < 0 → sameTerminalSide m ≤ -202 :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_angle_l3980_398067


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l3980_398077

theorem geometric_sequence_constant (a : ℕ → ℝ) (c : ℝ) :
  a 1 = 2 →
  (∀ n : ℕ, a (n + 1) = a n + c * n) →
  (∃ r : ℝ, r ≠ 1 ∧ a 2 = r * a 1 ∧ a 3 = r * a 2) →
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l3980_398077


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3980_398025

theorem quadratic_equation_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k - 1)*x₁ + k^2 - 1 = 0 ∧
    x₂^2 + (2*k - 1)*x₂ + k^2 - 1 = 0 ∧
    x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3980_398025


namespace NUMINAMATH_CALUDE_janous_inequality_l3980_398050

theorem janous_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), k > 0 ∧ x = 2*k ∧ y = k ∧ z = k) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l3980_398050


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3980_398032

theorem sqrt_expression_equality (t : ℝ) : 
  Real.sqrt (9 * t^4 + 4 * t^2 + 4 * t) = |t| * Real.sqrt ((3 * t^2 + 2 * t) * (3 * t^2 + 2 * t + 2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3980_398032


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l3980_398071

/-- The line equation 4x + 3y - 10 = 0 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y - 10 = 0

/-- The theorem stating the minimum value of m^2 + n^2 for points on the line -/
theorem min_distance_to_origin :
  ∀ m n : ℝ, line_equation m n → ∀ x y : ℝ, line_equation x y → m^2 + n^2 ≤ x^2 + y^2 ∧
  ∃ m₀ n₀ : ℝ, line_equation m₀ n₀ ∧ m₀^2 + n₀^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l3980_398071


namespace NUMINAMATH_CALUDE_simplify_expression_l3980_398036

theorem simplify_expression : (3 + 3 + 5) / 2 - 1 / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3980_398036


namespace NUMINAMATH_CALUDE_circle_intersection_and_common_chord_l3980_398030

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + 45 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 4*x + 3*y - 23 = 0

-- Define the intersection of circles
def circles_intersect (C₁ C₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, C₁ x y ∧ C₂ x y

-- Define the common chord
def common_chord (C₁ C₂ line_eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, (C₁ x y ∧ C₂ x y) → line_eq x y

-- Theorem statement
theorem circle_intersection_and_common_chord :
  (circles_intersect C₁ C₂) ∧
  (common_chord C₁ C₂ line_eq) ∧
  (∃ a b, C₁ a b ∧ C₂ a b ∧ (a - 1)^2 + (b - 3)^2 = 7) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_common_chord_l3980_398030


namespace NUMINAMATH_CALUDE_kimberley_firewood_l3980_398072

def firewood_problem (total houston ela : ℕ) : Prop :=
  total = 35 ∧ houston = 12 ∧ ela = 13

theorem kimberley_firewood (total houston ela : ℕ) 
  (h : firewood_problem total houston ela) : 
  total - (houston + ela) = 10 :=
by sorry

end NUMINAMATH_CALUDE_kimberley_firewood_l3980_398072


namespace NUMINAMATH_CALUDE_largest_package_size_l3980_398011

theorem largest_package_size (alex_markers jordan_markers : ℕ) 
  (h_alex : alex_markers = 56) (h_jordan : jordan_markers = 42) :
  Nat.gcd alex_markers jordan_markers = 14 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l3980_398011


namespace NUMINAMATH_CALUDE_necessary_condition_l3980_398000

theorem necessary_condition (p q : Prop) 
  (h : p → q) : ¬q → ¬p := by sorry

end NUMINAMATH_CALUDE_necessary_condition_l3980_398000


namespace NUMINAMATH_CALUDE_lukas_average_points_l3980_398006

/-- Given a basketball player's total points and number of games, 
    calculate their average points per game. -/
def average_points_per_game (total_points : ℕ) (num_games : ℕ) : ℚ :=
  (total_points : ℚ) / (num_games : ℚ)

/-- Theorem: A player who scores 60 points in 5 games averages 12 points per game. -/
theorem lukas_average_points : 
  average_points_per_game 60 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_lukas_average_points_l3980_398006


namespace NUMINAMATH_CALUDE_fraction_addition_l3980_398029

theorem fraction_addition (c : ℝ) : (6 + 5 * c) / 5 + 3 = (21 + 5 * c) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3980_398029


namespace NUMINAMATH_CALUDE_cubic_km_to_cubic_m_l3980_398042

/-- Proves that 1 cubic kilometer equals 1,000,000,000 cubic meters -/
theorem cubic_km_to_cubic_m : 
  (∀ (km m : ℝ), km = 1 ∧ m = 1000 ∧ km * 1000 = m) → 
  (1 : ℝ)^3 * 1000^3 = 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_km_to_cubic_m_l3980_398042


namespace NUMINAMATH_CALUDE_sum_nine_equals_27_l3980_398010

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  is_arithmetic : ∀ n m : ℕ+, a (n + 1) - a n = a (m + 1) - a m
  on_line : ∀ n : ℕ+, ∃ k b : ℝ, a n = k * n + b ∧ 3 = k * 5 + b

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ+) : ℝ :=
  (n : ℝ) * seq.a n

/-- The main theorem -/
theorem sum_nine_equals_27 (seq : ArithmeticSequence) : sum_n seq 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_nine_equals_27_l3980_398010


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l3980_398078

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem fibonacci_divisibility (k m n s : ℕ) (h : m > 0) (h1 : n > 0) :
  m ∣ fibonacci k → m^n ∣ fibonacci (k * m^(n-1) * s) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l3980_398078


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_l3980_398069

/-- Two points are symmetric about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are equal. -/
def symmetric_about_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_implies_sum (a b : ℝ) :
  symmetric_about_y_axis (a - 2, 3) (1, b + 1) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_l3980_398069


namespace NUMINAMATH_CALUDE_sheila_work_hours_l3980_398095

/-- Represents Sheila's work schedule and earnings -/
structure WorkSchedule where
  hoursPerDayMWF : ℕ
  totalWeeklyEarnings : ℕ
  hourlyRate : ℕ

/-- Calculates the number of hours Sheila works on Tuesday and Thursday -/
def hoursTueThu (schedule : WorkSchedule) : ℕ :=
  let knownDaysHours := 3 * schedule.hoursPerDayMWF
  let knownDaysEarnings := knownDaysHours * schedule.hourlyRate
  let remainingEarnings := schedule.totalWeeklyEarnings - knownDaysEarnings
  remainingEarnings / schedule.hourlyRate

/-- Theorem stating that Sheila works 12 hours on Tuesday and Thursday -/
theorem sheila_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.hoursPerDayMWF = 8)
  (h2 : schedule.totalWeeklyEarnings = 432)
  (h3 : schedule.hourlyRate = 12) : 
  hoursTueThu schedule = 12 := by
  sorry

end NUMINAMATH_CALUDE_sheila_work_hours_l3980_398095


namespace NUMINAMATH_CALUDE_triangle_rigidity_connected_beams_rigidity_l3980_398053

-- Define a structure for a triangle with three sides
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

-- Define a property for a triangle to be rigid
def is_rigid (t : Triangle) : Prop :=
  ∀ (t' : Triangle), t.side1 = t'.side1 ∧ t.side2 = t'.side2 ∧ t.side3 = t'.side3 →
    t = t'

-- Theorem stating that a triangle with fixed side lengths is rigid
theorem triangle_rigidity (t : Triangle) :
  is_rigid t :=
sorry

-- Define a beam as a line segment with fixed length
def Beam := ℝ

-- Define a structure for the connected beams
structure ConnectedBeams :=
  (beam1 : Beam)
  (beam2 : Beam)
  (beam3 : Beam)

-- Function to convert connected beams to a triangle
def beams_to_triangle (b : ConnectedBeams) : Triangle :=
  { side1 := b.beam1,
    side2 := b.beam2,
    side3 := b.beam3 }

-- Theorem stating that connected beams with fixed lengths form a rigid structure
theorem connected_beams_rigidity (b : ConnectedBeams) :
  is_rigid (beams_to_triangle b) :=
sorry

end NUMINAMATH_CALUDE_triangle_rigidity_connected_beams_rigidity_l3980_398053


namespace NUMINAMATH_CALUDE_paula_candies_l3980_398026

theorem paula_candies (x : ℕ) : 
  (x + 4 = 6 * 4) → x = 20 := by
sorry

end NUMINAMATH_CALUDE_paula_candies_l3980_398026


namespace NUMINAMATH_CALUDE_shirley_sold_at_least_20_boxes_l3980_398041

/-- The number of cases Shirley needs to deliver -/
def num_cases : ℕ := 5

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 4

/-- The number of extra boxes, which is unknown but non-negative -/
def extra_boxes : ℕ := sorry

/-- The total number of boxes Shirley sold -/
def total_boxes : ℕ := num_cases * boxes_per_case + extra_boxes

theorem shirley_sold_at_least_20_boxes : total_boxes ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_shirley_sold_at_least_20_boxes_l3980_398041


namespace NUMINAMATH_CALUDE_valid_closed_broken_line_segments_l3980_398035

/-- A closed broken line where each segment intersects exactly once and no three segments share a common point. -/
structure ClosedBrokenLine where
  segments : ℕ
  is_closed : Bool
  each_segment_intersects_once : Bool
  no_three_segments_share_point : Bool

/-- Predicate to check if a ClosedBrokenLine is valid -/
def is_valid_closed_broken_line (line : ClosedBrokenLine) : Prop :=
  line.is_closed ∧ line.each_segment_intersects_once ∧ line.no_three_segments_share_point

/-- Theorem stating that a valid ClosedBrokenLine can have 1996 segments but not 1997 -/
theorem valid_closed_broken_line_segments :
  (∃ (line : ClosedBrokenLine), line.segments = 1996 ∧ is_valid_closed_broken_line line) ∧
  (¬ ∃ (line : ClosedBrokenLine), line.segments = 1997 ∧ is_valid_closed_broken_line line) := by
  sorry

end NUMINAMATH_CALUDE_valid_closed_broken_line_segments_l3980_398035


namespace NUMINAMATH_CALUDE_ordered_pairs_sum_30_l3980_398057

theorem ordered_pairs_sum_30 :
  (Finset.filter (fun p : ℕ × ℕ => p.1 + p.2 = 30 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 31) (Finset.range 31))).card = 29 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_sum_30_l3980_398057


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_prism_side_length_l3980_398064

theorem isosceles_right_triangle_prism_side_length 
  (XY XZ : ℝ) (height volume : ℝ) : 
  XY = XZ →  -- Base triangle is isosceles
  height = 6 →  -- Height of the prism
  volume = 27 →  -- Volume of the prism
  volume = (1/2 * XY * XY) * height →  -- Volume formula for triangular prism
  XY = 3 ∧ XZ = 3  -- Conclusion: side lengths are 3
  :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_prism_side_length_l3980_398064


namespace NUMINAMATH_CALUDE_largest_non_sum_of_5_and_6_l3980_398073

def is_sum_of_5_and_6 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

theorem largest_non_sum_of_5_and_6 :
  (∀ n : ℕ, n > 19 → n ≤ 50 → is_sum_of_5_and_6 n) ∧
  ¬is_sum_of_5_and_6 19 := by
  sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_5_and_6_l3980_398073


namespace NUMINAMATH_CALUDE_mango_community_ratio_l3980_398080

/-- Represents the mango harvest and sales problem. -/
structure MangoHarvest where
  total_kg : ℕ  -- Total kilograms of mangoes harvested
  sold_market_kg : ℕ  -- Kilograms of mangoes sold to the market
  mangoes_per_kg : ℕ  -- Number of mangoes per kilogram
  mangoes_left : ℕ  -- Number of mangoes left after sales

/-- The ratio of mangoes sold to the community to total mangoes harvested is 1/3. -/
theorem mango_community_ratio (h : MangoHarvest) 
  (h_total : h.total_kg = 60)
  (h_market : h.sold_market_kg = 20)
  (h_per_kg : h.mangoes_per_kg = 8)
  (h_left : h.mangoes_left = 160) :
  (h.total_kg * h.mangoes_per_kg - h.sold_market_kg * h.mangoes_per_kg - h.mangoes_left) / 
  (h.total_kg * h.mangoes_per_kg) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_mango_community_ratio_l3980_398080


namespace NUMINAMATH_CALUDE_book_arrangement_problem_l3980_398012

theorem book_arrangement_problem (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 4) :
  Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_problem_l3980_398012


namespace NUMINAMATH_CALUDE_eliza_siblings_l3980_398017

/-- The number of Eliza's siblings -/
def num_siblings : ℕ := 4

/-- The height of the tallest sibling -/
def tallest_sibling_height : ℕ := 70

/-- Eliza's height -/
def eliza_height : ℕ := tallest_sibling_height - 2

/-- The total height of all siblings including Eliza -/
def total_height : ℕ := 330

theorem eliza_siblings :
  (2 * 66 + 60 + tallest_sibling_height + eliza_height = total_height) →
  (num_siblings = 4) := by sorry

end NUMINAMATH_CALUDE_eliza_siblings_l3980_398017


namespace NUMINAMATH_CALUDE_coin_problem_l3980_398099

theorem coin_problem (total_coins : ℕ) (total_value : ℚ) (nickel_value dime_value : ℚ) :
  total_coins = 28 →
  total_value = 260/100 →
  nickel_value = 5/100 →
  dime_value = 10/100 →
  ∃ (nickels dimes : ℕ),
    nickels + dimes = total_coins ∧
    nickels * nickel_value + dimes * dime_value = total_value ∧
    nickels = 4 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l3980_398099


namespace NUMINAMATH_CALUDE_whitewash_cost_l3980_398038

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
theorem whitewash_cost (room_length room_width room_height : ℝ)
                       (door_length door_width : ℝ)
                       (window_length window_width : ℝ)
                       (rate : ℝ) :
  room_length = 25 →
  room_width = 15 →
  room_height = 12 →
  door_length = 6 →
  door_width = 3 →
  window_length = 4 →
  window_width = 3 →
  rate = 5 →
  (2 * (room_length * room_height + room_width * room_height) -
   (door_length * door_width + 3 * window_length * window_width)) * rate = 4530 := by
  sorry

#check whitewash_cost

end NUMINAMATH_CALUDE_whitewash_cost_l3980_398038


namespace NUMINAMATH_CALUDE_ambiguous_dates_count_l3980_398018

/-- The number of months in a year -/
def num_months : ℕ := 12

/-- The maximum day number that can be confused as a month -/
def max_ambiguous_day : ℕ := 12

/-- The number of ambiguous dates in a year -/
def num_ambiguous_dates : ℕ := num_months * max_ambiguous_day - num_months

theorem ambiguous_dates_count :
  num_ambiguous_dates = 132 :=
sorry

end NUMINAMATH_CALUDE_ambiguous_dates_count_l3980_398018


namespace NUMINAMATH_CALUDE_escalator_length_is_200_l3980_398031

/-- The length of an escalator given its speed, a person's walking speed, and the time taken to cover the entire length. -/
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time_taken : ℝ) : ℝ :=
  (escalator_speed + person_speed) * time_taken

/-- Theorem stating that the length of the escalator is 200 feet -/
theorem escalator_length_is_200 :
  escalator_length 15 5 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_is_200_l3980_398031


namespace NUMINAMATH_CALUDE_circle_division_theorem_l3980_398083

/-- A type representing a straight cut through a circle -/
structure Cut where
  -- We don't need to define the internal structure of a cut for this statement

/-- A type representing a circle -/
structure Circle where
  -- We don't need to define the internal structure of a circle for this statement

/-- A function that counts the number of regions created by cuts in a circle -/
def count_regions (circle : Circle) (cuts : List Cut) : ℕ := sorry

/-- Theorem stating that a circle can be divided into 4, 5, 6, and 7 parts using three straight cuts -/
theorem circle_division_theorem (circle : Circle) :
  ∃ (cuts₁ cuts₂ cuts₃ cuts₄ : List Cut),
    (cuts₁.length = 3 ∧ count_regions circle cuts₁ = 4) ∧
    (cuts₂.length = 3 ∧ count_regions circle cuts₂ = 5) ∧
    (cuts₃.length = 3 ∧ count_regions circle cuts₃ = 6) ∧
    (cuts₄.length = 3 ∧ count_regions circle cuts₄ = 7) :=
  sorry

end NUMINAMATH_CALUDE_circle_division_theorem_l3980_398083


namespace NUMINAMATH_CALUDE_equation_solution_l3980_398056

theorem equation_solution (x : ℝ) :
  x > 6 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3980_398056


namespace NUMINAMATH_CALUDE_specific_hyperbola_conjugate_axis_length_l3980_398084

/-- Represents a hyperbola with equation x^2 - y^2/m = 1 -/
structure Hyperbola where
  m : ℝ
  focus : ℝ × ℝ

/-- The length of the conjugate axis of a hyperbola -/
def conjugate_axis_length (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the length of the conjugate axis for a specific hyperbola -/
theorem specific_hyperbola_conjugate_axis_length :
  ∀ (h : Hyperbola), 
  h.m > 0 ∧ h.focus = (-3, 0) → 
  conjugate_axis_length h = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_specific_hyperbola_conjugate_axis_length_l3980_398084


namespace NUMINAMATH_CALUDE_parent_payment_calculation_l3980_398089

/-- Calculates the amount each parent has to pay in different currencies --/
theorem parent_payment_calculation
  (former_salary : ℝ)
  (raise_percentage : ℝ)
  (tax_rate : ℝ)
  (num_kids : ℕ)
  (usd_to_eur : ℝ)
  (usd_to_gbp : ℝ)
  (usd_to_jpy : ℝ)
  (h1 : former_salary = 60000)
  (h2 : raise_percentage = 0.25)
  (h3 : tax_rate = 0.10)
  (h4 : num_kids = 15)
  (h5 : usd_to_eur = 0.85)
  (h6 : usd_to_gbp = 0.75)
  (h7 : usd_to_jpy = 110) :
  let new_salary := former_salary * (1 + raise_percentage)
  let after_tax_salary := new_salary * (1 - tax_rate)
  let amount_per_parent := after_tax_salary / num_kids
  (amount_per_parent / usd_to_eur = 5294.12) ∧
  (amount_per_parent / usd_to_gbp = 6000) ∧
  (amount_per_parent * usd_to_jpy = 495000) :=
by sorry

end NUMINAMATH_CALUDE_parent_payment_calculation_l3980_398089


namespace NUMINAMATH_CALUDE_end_at_multiple_of_4_probability_l3980_398054

/-- Represents the possible moves on the spinner -/
inductive SpinnerMove
| Left2 : SpinnerMove
| Right2 : SpinnerMove
| Right1 : SpinnerMove

/-- The probability of a specific move on the spinner -/
def spinnerProbability (move : SpinnerMove) : ℚ :=
  match move with
  | SpinnerMove.Left2 => 1/4
  | SpinnerMove.Right2 => 1/2
  | SpinnerMove.Right1 => 1/4

/-- The set of cards Jeff can pick from -/
def cardSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

/-- Whether a number is a multiple of 4 -/
def isMultipleOf4 (n : ℕ) : Prop := ∃ k, n = 4 * k

/-- The probability of ending at a multiple of 4 -/
def probEndAtMultipleOf4 : ℚ := 1/32

theorem end_at_multiple_of_4_probability : 
  probEndAtMultipleOf4 = 1/32 :=
sorry

end NUMINAMATH_CALUDE_end_at_multiple_of_4_probability_l3980_398054


namespace NUMINAMATH_CALUDE_alley_width_l3980_398062

/-- The width of an alley given a ladder's two configurations -/
theorem alley_width (L : ℝ) (k h : ℝ) : ∃ w : ℝ,
  (k = L / 2) →
  (h = L * Real.sqrt 3 / 2) →
  (w^2 + (L/2)^2 = L^2) →
  (w^2 + (L * Real.sqrt 3 / 2)^2 = L^2) →
  w = L * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_alley_width_l3980_398062


namespace NUMINAMATH_CALUDE_min_value_theorem_l3980_398094

theorem min_value_theorem (a b : ℝ) (h1 : a * b - 2 * a - b + 1 = 0) (h2 : a > 1) :
  ∀ x y : ℝ, x * y - 2 * x - y + 1 = 0 → x > 1 → (a + 3) * (b + 2) ≤ (x + 3) * (y + 2) ∧
  ∃ a₀ b₀ : ℝ, a₀ * b₀ - 2 * a₀ - b₀ + 1 = 0 ∧ a₀ > 1 ∧ (a₀ + 3) * (b₀ + 2) = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3980_398094


namespace NUMINAMATH_CALUDE_triangle_inscribed_circle_properties_l3980_398059

/-- Given a triangle ABC with semi-perimeter s, inscribed circle center O₁,
    inscribed circle radius r₁, and circumscribed circle radius R -/
theorem triangle_inscribed_circle_properties 
  (A B C O₁ : ℝ × ℝ) (r₁ R s a b c : ℝ) :
  let AO₁ := Real.sqrt ((A.1 - O₁.1)^2 + (A.2 - O₁.2)^2)
  let BO₁ := Real.sqrt ((B.1 - O₁.1)^2 + (B.2 - O₁.2)^2)
  let CO₁ := Real.sqrt ((C.1 - O₁.1)^2 + (C.2 - O₁.2)^2)
  -- Conditions
  s = (a + b + c) / 2 →
  -- Theorem statements
  AO₁^2 = (s / (s - a)) * b * c ∧
  AO₁ * BO₁ * CO₁ = 4 * R * r₁^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inscribed_circle_properties_l3980_398059


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3980_398075

-- Define the conditions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, not_p x → not_q x) ∧ 
  ¬(∀ x, not_q x → not_p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3980_398075


namespace NUMINAMATH_CALUDE_draw_balls_theorem_l3980_398004

/-- The number of ways to draw balls from a bag under specific conditions -/
def draw_balls_count (total_white : ℕ) (total_red : ℕ) (total_black : ℕ) 
                     (draw_count : ℕ) (min_white : ℕ) (max_white : ℕ) 
                     (min_red : ℕ) (max_red : ℕ) (max_black : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to draw balls under given conditions -/
theorem draw_balls_theorem : 
  draw_balls_count 9 5 6 10 3 7 2 5 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_draw_balls_theorem_l3980_398004


namespace NUMINAMATH_CALUDE_product_not_negative_l3980_398082

theorem product_not_negative (x y : ℝ) (n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^(2*n) - y^(2*n) > x) (h2 : y^(2*n) - x^(2*n) > y) :
  x * y > 0 :=
sorry

end NUMINAMATH_CALUDE_product_not_negative_l3980_398082


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3980_398002

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 16/49
  let a₃ : ℚ := 64/343
  (a₂ / a₁ = 4/7) ∧ (a₃ / a₂ = 4/7) → 
  ∃ (r : ℚ), ∀ (n : ℕ), n ≥ 1 → 
    (4/7) * (4/7)^(n-1) = (4/7) * r^(n-1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3980_398002


namespace NUMINAMATH_CALUDE_square_difference_sum_l3980_398003

theorem square_difference_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_sum_l3980_398003


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l3980_398014

theorem smaller_solution_of_quadratic (x : ℝ) :
  x^2 - 14*x + 45 = 0 → ∃ y : ℝ, y^2 - 14*y + 45 = 0 ∧ y ≠ x ∧ (∀ z : ℝ, z^2 - 14*z + 45 = 0 → z = x ∨ z = y) ∧ min x y = 5 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l3980_398014


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_properties_l3980_398058

/-- An ellipse and hyperbola with shared properties -/
structure EllipseHyperbola where
  /-- The distance between the foci -/
  focal_distance : ℝ
  /-- The difference between the major axis of the ellipse and the real axis of the hyperbola -/
  axis_difference : ℝ
  /-- The ratio of eccentricities (ellipse:hyperbola) -/
  eccentricity_ratio : ℝ × ℝ

/-- The equations of the ellipse and hyperbola -/
def curve_equations (eh : EllipseHyperbola) : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) :=
  (λ x y ↦ x^2/49 + y^2/36 = 1, λ x y ↦ x^2/9 - y^2/4 = 1)

/-- The area of the triangle formed by the foci and an intersection point -/
def triangle_area (eh : EllipseHyperbola) : ℝ := 12

/-- Theorem stating the properties of the ellipse and hyperbola -/
theorem ellipse_hyperbola_properties (eh : EllipseHyperbola)
    (h1 : eh.focal_distance = 2 * Real.sqrt 13)
    (h2 : eh.axis_difference = 4)
    (h3 : eh.eccentricity_ratio = (3, 7)) :
    curve_equations eh = (λ x y ↦ x^2/49 + y^2/36 = 1, λ x y ↦ x^2/9 - y^2/4 = 1) ∧
    triangle_area eh = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_properties_l3980_398058


namespace NUMINAMATH_CALUDE_polynomial_equality_l3980_398044

theorem polynomial_equality : 
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 102^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3980_398044


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3980_398087

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 = 2 →                     -- a_1 = 2
  d ≠ 0 →                       -- d ≠ 0
  (a 2) ^ 2 = a 1 * a 5 →       -- a_1, a_2, a_5 form a geometric sequence
  d = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3980_398087


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2023_l3980_398007

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising a number to a power, considering only the units digit -/
def powerMod10 (base : ℕ) (exp : ℕ) : ℕ :=
  (base ^ exp) % 10

theorem units_digit_17_pow_2023 :
  unitsDigit (powerMod10 17 2023) = 3 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2023_l3980_398007


namespace NUMINAMATH_CALUDE_fourth_student_number_l3980_398033

def class_size : ℕ := 54
def sample_size : ℕ := 4

def systematic_sample (start : ℕ) : Fin 4 → ℕ :=
  λ i => (start + i.val * 13) % class_size + 1

theorem fourth_student_number (h : ∃ start : ℕ, 
  (systematic_sample start 1 = 3 ∧ 
   systematic_sample start 2 = 29 ∧ 
   systematic_sample start 3 = 42)) : 
  ∃ start : ℕ, systematic_sample start 0 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_number_l3980_398033


namespace NUMINAMATH_CALUDE_five_hour_charge_l3980_398020

/-- Represents the charge structure and total charge calculation for a psychologist's therapy sessions. -/
structure TherapyCharges where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  moreExpensiveFirst : firstHourCharge = additionalHourCharge + 35
  twoHourTotal : firstHourCharge + additionalHourCharge = 161

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (charges : TherapyCharges) (hours : ℕ) : ℕ :=
  charges.firstHourCharge + (hours - 1) * charges.additionalHourCharge

/-- Theorem stating that the total charge for 5 hours of therapy is $350. -/
theorem five_hour_charge (charges : TherapyCharges) : totalCharge charges 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_five_hour_charge_l3980_398020
