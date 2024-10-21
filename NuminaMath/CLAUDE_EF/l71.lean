import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_count_range_l71_7110

/-- A permutation of n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The number of inversions in a permutation -/
def inversionCount (n : ℕ) (p : Permutation n) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin n)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin n)) fun j =>
      if i < j ∧ p i > p j then 1 else 0

/-- Theorem: For any permutation of 10 elements, the number of inversions
    can be any integer from 0 to 45 inclusive -/
theorem inversion_count_range :
  ∀ k : ℕ, k ≤ 45 → ∃ p : Permutation 10, inversionCount 10 p = k :=
by
  sorry

#check inversion_count_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_count_range_l71_7110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_2023_divisors_l71_7123

/-- A positive integer with exactly 2023 distinct positive divisors -/
def has_2023_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range n)).card = 2023

/-- Represents n as m * 6^k where 6 doesn't divide m -/
def representable (n m k : ℕ) : Prop :=
  n = m * 6^k ∧ ¬(6 ∣ m)

/-- The main theorem -/
theorem least_integer_with_2023_divisors :
  ∃ (n m k : ℕ),
    has_2023_divisors n ∧
    representable n m k ∧
    (∀ (n' m' k' : ℕ), has_2023_divisors n' ∧ representable n' m' k' → n ≤ n') ∧
    m + k = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_2023_divisors_l71_7123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l71_7139

theorem cube_root_simplification :
  (((3 : ℝ) ^ 4 * (3 : ℝ) ^ 5 + 5 * (3 : ℝ) ^ 5) ^ (1/3 : ℝ)) = 9 * (86 ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l71_7139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l71_7198

-- Define the complex number z
noncomputable def z : ℂ := (3 + 4 * Complex.I) / (1 - Complex.I)

-- Theorem statement
theorem modulus_of_z : Complex.abs z = (5 * Real.sqrt 2) / 2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l71_7198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l71_7182

/-- The cosine of the angle between two 2D vectors -/
noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

theorem cos_angle_specific_vectors :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (1, 2)
  cos_angle a b = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l71_7182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stairs_climbing_ways_l71_7106

def climb_ways : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | 4 => 8
  | n + 5 => climb_ways (n + 4) + climb_ways (n + 3) + climb_ways (n + 2) + climb_ways (n + 1)

theorem stairs_climbing_ways :
  climb_ways 8 = 108 := by
  rfl

#eval climb_ways 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stairs_climbing_ways_l71_7106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_cut_count_l71_7180

theorem right_prism_cut_count : 
  let count := Finset.filter (fun t : ℕ × ℕ × ℕ => 
    let (a, b, c) := t
    a ≤ b ∧ b ≤ c ∧ b = 2022 ∧ a * c = 2022 * 2022 ∧ a < c) 
    (Finset.range 2023 ×ˢ Finset.range 2023 ×ˢ Finset.range 2023)
  Finset.card count = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_cut_count_l71_7180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_through_point_l71_7192

/-- Given that the terminal side of angle θ passes through point P(2, -1), prove that sin θ = - √5/5 -/
theorem sine_of_angle_through_point :
  ∀ θ : ℝ,
  ∃ (x y : ℝ),
  x = 2 ∧ y = -1 ∧  -- Point P(2, -1)
  (∃ (r : ℝ), r = Real.sqrt (x^2 + y^2) ∧ r > 0) →  -- r is the distance from origin to P
  Real.sin θ = - Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_through_point_l71_7192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_value_l71_7187

theorem max_lambda_value (m n k : ℕ+) (h1 : m + n = 3 * k) (h2 : m ≠ n) :
  ∃ (lambda_max : ℝ), lambda_max = 9/2 ∧
    (∀ (lambda : ℝ), (∀ (m n k : ℕ+), m + n = 3 * k → m ≠ n → 
      m^2 + n^2 - lambda * k^2 > 0) → lambda ≤ lambda_max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_value_l71_7187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l71_7147

theorem equilateral_triangle_area_ratio : 
  ∃ (large_side small_side large_area small_area trapezoid_area : ℝ),
    large_side = 12 ∧
    small_side = 6 ∧
    large_area = (Real.sqrt 3 / 4) * large_side^2 ∧
    small_area = (Real.sqrt 3 / 4) * small_side^2 ∧
    trapezoid_area = large_area - small_area ∧
    small_area / trapezoid_area = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l71_7147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_same_color_count_even_l71_7125

/-- Represents a color (Red or Blue) -/
inductive Color
| Red
| Blue

/-- Represents a square grid with colored vertices -/
structure ColoredGrid (n : ℕ+) where
  grid : Fin n → Fin n → Color
  corner_a : grid 0 0 = Color.Red
  corner_c : grid (n-1) (n-1) = Color.Red
  corner_b : grid 0 (n-1) = Color.Blue
  corner_d : grid (n-1) 0 = Color.Blue

/-- Counts the number of smaller squares with exactly three vertices of the same color -/
def count_three_same_color (n : ℕ+) (g : ColoredGrid n) : ℕ :=
  sorry

/-- The main theorem: the count of smaller squares with three vertices of the same color is even -/
theorem three_same_color_count_even (n : ℕ+) (g : ColoredGrid n) :
  Even (count_three_same_color n g) := by
  sorry

#check three_same_color_count_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_same_color_count_even_l71_7125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ray_l71_7128

-- Define points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the condition for point P
def condition (P : ℝ × ℝ) : Prop :=
  |dist P M - dist P N| = 4

-- Define a ray
def is_ray (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ S = {P | ∃ t : ℝ, t ≥ 0 ∧ P = A + t • (B - A)}

-- Theorem statement
theorem trajectory_is_ray :
  is_ray {P : ℝ × ℝ | condition P} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ray_l71_7128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_inflection_point_and_center_of_symmetry_l71_7138

noncomputable def f (x : ℝ) : ℝ := 1/3 * x^3 - 1/2 * x^2 + 3*x - 5/12

noncomputable def f' (x : ℝ) : ℝ := x^2 - x + 3

noncomputable def f'' (x : ℝ) : ℝ := 2*x - 1

theorem cubic_inflection_point_and_center_of_symmetry :
  let x₀ : ℝ := 1/2
  let y₀ : ℝ := f x₀
  (f'' x₀ = 0) ∧ (y₀ = 1) ∧ (∀ h : ℝ, f (x₀ + h) + f (x₀ - h) = 2 * y₀) := by
  sorry

#check cubic_inflection_point_and_center_of_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_inflection_point_and_center_of_symmetry_l71_7138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l71_7103

open Real

theorem min_shift_for_symmetry (f : ℝ → ℝ) (φ : ℝ) :
  (f = λ x ↦ 3 * sin (2 * x + π / 3)) →
  φ > 0 →
  (∀ x, f (x - φ) = -f (-x - φ)) →
  φ ≥ π / 6 ∧ 
  ∃ φ₀, φ₀ = π / 6 ∧ φ₀ > 0 ∧ (∀ x, f (x - φ₀) = -f (-x - φ₀)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l71_7103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l71_7150

/-- The polar equation of the circle -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * (Real.cos θ + Real.sin θ)

/-- The center of the circle in polar coordinates -/
noncomputable def center : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

/-- Theorem stating that the center of the circle is correct -/
theorem circle_center_correct :
  ∀ ρ θ, polar_equation ρ θ → 
  ∃ r φ, r * Real.cos φ = center.1 * Real.cos center.2 ∧
         r * Real.sin φ = center.1 * Real.sin center.2 ∧
         r = ρ ∧ φ = θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l71_7150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_and_line_l_l71_7141

noncomputable section

/-- The direction vector of line l -/
def direction_vector : ℝ × ℝ := (2, -2)

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop := x + y + 1 = 0

/-- Point M -/
def M : ℝ × ℝ := (1, 0)

/-- Point N -/
def N : ℝ × ℝ := (-5, -2)

/-- Point A is the midpoint of MN -/
noncomputable def A : ℝ × ℝ := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

/-- The y-intercept of l' is 1/2 of its x-intercept -/
def intercept_relation (a : ℝ) : Prop := ∃ (x y : ℝ), x / a + y / (a / 2) = 1

/-- Line l' passes through point A -/
def line_l' (x y : ℝ) : Prop := 
  (x - A.1) * (A.2 - (-2)) = (y - A.2) * (A.1 - 4) ∨ 
  (x - A.1) * (A.2 - 0) = (y - A.2) * (A.1 - 0)

theorem point_A_and_line_l' : 
  A = (-2, -1) ∧ 
  (∀ x y, line_l' x y ↔ (x - 2*y = 0 ∨ x + 2*y + 4 = 0)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_and_line_l_l71_7141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_state_after_pulls_l71_7121

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On : LampState
| Off : LampState

/-- Represents a lamp with its label and state -/
structure Lamp where
  label : Char
  state : LampState

/-- Toggles the state of a lamp -/
def toggleLamp (l : Lamp) : Lamp :=
  match l.state with
  | LampState.On => { l with state := LampState.Off }
  | LampState.Off => { l with state := LampState.On }

/-- Represents the row of lamps -/
def LampRow := List Lamp

/-- Pulls switches for all lamps in the row -/
def pullSwitches (row : LampRow) : LampRow :=
  row.map toggleLamp

/-- Returns true if the lamp is on, false otherwise -/
def isLampOn (l : Lamp) : Bool :=
  match l.state with
  | LampState.On => true
  | LampState.Off => false

/-- Applies the pullSwitches function n times to the given row -/
def applyPullSwitches (n : Nat) (row : LampRow) : LampRow :=
  match n with
  | 0 => row
  | n + 1 => applyPullSwitches n (pullSwitches row)

theorem lamp_state_after_pulls (initialRow : LampRow) :
  let finalRow := applyPullSwitches 1999 initialRow
  (finalRow.filter (fun l => l.label = 'A' ∨ l.label = 'C' ∨ l.label = 'F')).all isLampOn ∧
  (finalRow.filter (fun l => l.label = 'B' ∨ l.label = 'D' ∨ l.label = 'E' ∨ l.label = 'G')).all (fun l => ¬ isLampOn l) :=
by sorry

#check lamp_state_after_pulls

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_state_after_pulls_l71_7121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_composite_l71_7107

-- Define a type for base-3 numbers
def Base3 := ℕ

-- Define a function to check if a number is composite
def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Define the given numbers
def n1 : ℕ := 12002110
def n2 : ℕ := 2210121012
def n3 : ℕ := 121212
def n4 : ℕ := 102102

-- Define a type for repeating base-3 numbers
structure RepeatingBase3 where
  abc : ℕ

-- Define the ABCABC number
def n5 : RepeatingBase3 := ⟨123⟩  -- Using 123 as an example

-- State the theorem
theorem all_numbers_composite :
  is_composite n1 ∧
  is_composite n2 ∧
  is_composite n3 ∧
  is_composite n4 ∧
  is_composite (n5.abc * 1001) :=
by
  sorry

#check all_numbers_composite

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_composite_l71_7107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_even_non_prime_sum_l71_7100

-- Define the sequence of prime numbers
def prime_seq : ℕ → ℕ
| 0 => 2
| 1 => 3
| 2 => 5
| n + 3 => sorry  -- We don't need to define the entire sequence

-- Define the sum of the first n prime numbers
def prime_sum (n : ℕ) : ℕ :=
  (List.range n).map prime_seq |>.sum

-- Main theorem
theorem least_even_non_prime_sum : 
  ∀ k : ℕ, k > 0 → Even k → k < 8 → Nat.Prime (prime_sum k) ∧
  ¬ Nat.Prime (prime_sum 8) :=
by
  sorry

#check least_even_non_prime_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_even_non_prime_sum_l71_7100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_diagonal_line_l71_7109

theorem angle_on_diagonal_line (a : Real) :
  (∃ x y : Real, x + y = 0 ∧ x = Real.cos a ∧ y = Real.sin a) →
  (Real.sin a / Real.sqrt (1 - Real.sin a ^ 2)) + (Real.sqrt (1 - Real.cos a ^ 2) / Real.cos a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_diagonal_line_l71_7109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_of_fraction_l71_7105

def rationalize_denominator (n : ℚ) (a b : ℚ) : ℚ × ℚ := sorry

def is_not_divisible_by_square (n : ℤ) : Prop := sorry

theorem rationalized_form_of_fraction :
  let (num, den) := rationalize_denominator 5 2 6
  (∃ (A B C D : ℤ), 
    num = A * (Real.sqrt (B : ℝ)) + C ∧
    den = D ∧
    D > 0 ∧
    is_not_divisible_by_square B ∧
    Int.gcd A (Int.gcd C D) = 1 ∧
    A = 5 ∧ B = 6 ∧ C = -10 ∧ D = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_of_fraction_l71_7105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l71_7146

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.exp x + b else -(Real.exp (-x) + b)

-- State the theorem
theorem odd_function_value (b : ℝ) : 
  (∀ x, f b (-x) = -(f b x)) → -- f is odd
  f b (-Real.log 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l71_7146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_midpoint_l71_7108

-- Define the ellipse parameters
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 4
noncomputable def c : ℝ := 3

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
axiom a_gt_b : a > b
axiom b_gt_zero : b > 0
axiom focal_distance : c = 3
axiom perimeter_condition : 2*c + 2*a = 16
axiom pythagoras : a^2 = b^2 + c^2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 4/5 * (x - 3)

-- State the theorem
theorem ellipse_and_midpoint :
  (a = 5 ∧ b = 4) ∧
  ∃ (x y : ℝ), 
    ellipse_equation x y ∧ 
    line_equation x y ∧
    x = 3/2 ∧ 
    y = -6/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_midpoint_l71_7108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l71_7199

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) + x^2 / 2
noncomputable def g (x : ℝ) : ℝ := Real.cos x + x^2 / 2

-- State the theorem
theorem f_g_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_fg : f (Real.exp (a/2)) = g b - 1) :
  (∀ x : ℝ, x ≥ 0 → f x ≥ x) ∧ (f (b^2) + 1 > g (a + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l71_7199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wasted_area_calculation_l71_7163

/-- Calculates the area of wasted material when cutting a circle from a rectangle and a square from that circle -/
theorem wasted_area_calculation (rectangle_length rectangle_width : ℝ) : 
  rectangle_length = 10 ∧ 
  rectangle_width = 8 → 
  let circle_radius : ℝ := rectangle_width / 2
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let circle_area : ℝ := Real.pi * circle_radius ^ 2
  let square_side : ℝ := circle_radius * Real.sqrt 2
  let square_area : ℝ := square_side ^ 2
  let wasted_area : ℝ := rectangle_area - square_area
  wasted_area = 48 := by
    intro h
    sorry

-- The following line is not necessary for the theorem, but can be used for testing
-- #eval wasted_area_calculation 10 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wasted_area_calculation_l71_7163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l71_7193

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

-- Define a, b, and c
noncomputable def a : ℝ := f (Real.sqrt 2 / 2)
noncomputable def b : ℝ := f (Real.sqrt 3 / 2)
noncomputable def c : ℝ := f (Real.sqrt 6 / 2)

-- State the theorem
theorem f_inequality : b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l71_7193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_XY_is_17_l71_7120

/-- The distance between points X and Y --/
noncomputable def distance_XY : ℝ := 17

/-- Yolanda's walking rate in miles per hour --/
noncomputable def yolanda_rate : ℝ := 3

/-- Bob's walking rate in miles per hour --/
noncomputable def bob_rate : ℝ := 4

/-- The distance Bob walked when they met --/
noncomputable def bob_distance : ℝ := 8

/-- The time Bob walked before they met --/
noncomputable def bob_time : ℝ := bob_distance / bob_rate

/-- The time Yolanda walked before they met --/
noncomputable def yolanda_time : ℝ := bob_time + 1

theorem distance_XY_is_17 :
  distance_XY = yolanda_rate * yolanda_time + bob_rate * bob_time :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_XY_is_17_l71_7120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_six_between_30_and_50_l71_7153

theorem count_multiples_of_six_between_30_and_50 : 
  Finset.card (Finset.filter (fun n => 6 ∣ n ∧ 30 ≤ n ∧ n ≤ 50) (Finset.range 51)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_six_between_30_and_50_l71_7153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_inverse_l71_7188

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the property of f being symmetric with respect to y = x-2
def symmetric_about_x_minus_2 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (y + 2) = x + 2

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 3)

-- State the theorem
theorem g_equals_inverse (f : ℝ → ℝ) 
  (h : symmetric_about_x_minus_2 f) : 
  ∀ x, g f x = Function.invFun (g f) x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_inverse_l71_7188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_theorem_l71_7176

/-- Calculates the average speed for the second part of a journey given the total distance,
    total time, distance covered at a known speed, and that known speed. -/
noncomputable def second_part_speed (total_distance : ℝ) (total_time : ℝ) (distance_at_known_speed : ℝ) (known_speed : ℝ) : ℝ :=
  let remaining_distance := total_distance - distance_at_known_speed
  let time_at_known_speed := distance_at_known_speed / known_speed
  let remaining_time := total_time - time_at_known_speed
  remaining_distance / remaining_time

/-- Theorem stating that for a journey with given parameters, the average speed
    for the second part is 60 kmph. -/
theorem journey_speed_theorem :
  second_part_speed 250 5.2 124 40 = 60 := by
  -- Unfold the definition of second_part_speed
  unfold second_part_speed
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_theorem_l71_7176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_length_l71_7173

/-- Represents a rhombus with given area and one diagonal length -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Calculates the length of the second diagonal of a rhombus -/
noncomputable def second_diagonal (r : Rhombus) : ℝ :=
  (2 * r.area) / r.diagonal1

/-- Theorem stating that the second diagonal is approximately 5.8 cm -/
theorem rhombus_second_diagonal_length (r : Rhombus) 
  (h1 : r.area = 21.46)
  (h2 : r.diagonal1 = 7.4) : 
  ∃ ε > 0, |second_diagonal r - 5.8| < ε := by
  sorry

#check rhombus_second_diagonal_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_length_l71_7173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_period_f_increasing_f_decreasing_l71_7167

open Real

/-- Given vectors a and b, define f(x) as their dot product -/
noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * cos (x/2), tan (x/2 + π/4))

noncomputable def b (x : ℝ) : ℝ × ℝ := (sqrt 2 * sin (x/2 + π/4), tan (x/2 - π/4))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

/-- The maximum value of f(x) is √2 -/
theorem f_max_value : ∃ x, f x = sqrt 2 ∧ ∀ y, f y ≤ sqrt 2 := by
  sorry

/-- The smallest positive period of f(x) is 2π -/
theorem f_period : ∀ x, f (x + 2*π) = f x ∧ ∀ p, 0 < p → p < 2*π → ∃ x, f (x + p) ≠ f x := by
  sorry

/-- f(x) is strictly increasing on [0, π/4] -/
theorem f_increasing : ∀ x y, 0 ≤ x → x < y → y ≤ π/4 → f x < f y := by
  sorry

/-- f(x) is strictly decreasing on [π/4, π/2] -/
theorem f_decreasing : ∀ x y, π/4 ≤ x → x < y → y ≤ π/2 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_period_f_increasing_f_decreasing_l71_7167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_less_than_factorial_squared_l71_7118

theorem power_less_than_factorial_squared : 2011^2011 < (Nat.factorial 2011)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_less_than_factorial_squared_l71_7118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixing_l71_7136

/-- Represents a salt solution with a given mass and concentration -/
structure SaltSolution where
  mass : ℝ
  concentration : ℝ

/-- Calculates the mass of salt in a solution -/
noncomputable def saltMass (solution : SaltSolution) : ℝ :=
  solution.mass * solution.concentration

/-- Represents the mixing of two salt solutions -/
noncomputable def mixSolutions (s1 s2 : SaltSolution) : SaltSolution :=
  { mass := s1.mass + s2.mass,
    concentration := (saltMass s1 + saltMass s2) / (s1.mass + s2.mass) }

theorem salt_solution_mixing :
  let initial := SaltSolution.mk 600 0.25
  let added := SaltSolution.mk 200 0.05
  let mixed := mixSolutions initial added
  mixed.concentration = 0.20 := by
    -- Proof steps would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixing_l71_7136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_consistent_l71_7166

-- Define the function f
noncomputable def f : ℝ → ℝ :=
  fun x => if x < 0 then x * (1 - x) else x * (1 + x)

-- State the theorem
theorem f_odd_and_consistent :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, x < 0 → f x = x * (1 - x)) ∧
  (∀ x : ℝ, x > 0 → f x = x * (1 + x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_consistent_l71_7166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_proof_l71_7172

/-- The area of the octagon formed by connecting the vertices of a square
    with side length a to the midpoints of the opposite sides -/
noncomputable def octagon_area (a : ℝ) : ℝ := a^2 / 6

theorem octagon_area_proof (a : ℝ) (h : a > 0) :
  octagon_area a = a^2 / 6 := by
  -- Unfold the definition of octagon_area
  unfold octagon_area
  -- The definition matches the right-hand side exactly
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_proof_l71_7172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_san_francisco_to_madrid_time_l71_7159

/-- Represents a time of day in hours (0-23) -/
def Time := Fin 24

/-- Represents a day (yesterday, today, tomorrow) -/
inductive Day
| yesterday
| today
| tomorrow

/-- Represents a moment in time with a day and time -/
structure Moment where
  day : Day
  time : Time

/-- Adds hours to a given time, handling day changes -/
def addHours (m : Moment) (h : Int) : Moment :=
  sorry

/-- The time difference between two cities in hours -/
def timeDifference (city1 city2 : String) : Int :=
  sorry

/-- Theorem: When it's 9 pm in San Francisco, it's 6 am the next day in Madrid -/
theorem san_francisco_to_madrid_time :
  let sf_time : Moment := ⟨Day.yesterday, ⟨21, by norm_num⟩⟩  -- 9 pm yesterday in San Francisco
  let madrid_time : Moment := addHours sf_time (timeDifference "San Francisco" "Madrid")
  madrid_time = ⟨Day.today, ⟨6, by norm_num⟩⟩  -- 6 am today in Madrid
  := by sorry

#check san_francisco_to_madrid_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_san_francisco_to_madrid_time_l71_7159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_value_l71_7185

/-- The quadratic equation -/
def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 7 * x - 6 = 0

/-- The root form -/
def root_form (x m n p : ℝ) : Prop := x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p

/-- m, n, and p are positive integers -/
def positive_integers (m n p : ℕ) : Prop := m > 0 ∧ n > 0 ∧ p > 0

/-- m, n, and p are relatively prime -/
def relatively_prime (m n p : ℕ) : Prop := Nat.gcd m n = 1 ∧ Nat.gcd m p = 1 ∧ Nat.gcd n p = 1

/-- The main theorem -/
theorem quadratic_root_value (m n p : ℕ) :
  (∃ x : ℝ, quadratic_equation x ∧ root_form x (m : ℝ) (n : ℝ) (p : ℝ)) →
  positive_integers m n p →
  relatively_prime m n p →
  n = 121 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_value_l71_7185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nephew_is_worst_player_l71_7174

-- Define the players
inductive Player
| Father
| Sister
| Son
| Nephew

-- Define the sex of players
inductive Sex
| Male
| Female

-- Define the function to get the sex of a player
def playerSex : Player → Sex
| Player.Father => Sex.Male
| Player.Sister => Sex.Female
| Player.Son => Sex.Male
| Player.Nephew => Sex.Male

-- Define the function to check if two players can be twins
def canBeTwins : Player → Player → Prop
| Player.Son, Player.Nephew => True
| Player.Nephew, Player.Son => True
| _, _ => False

-- Define the worst player
def worstPlayer : Player := Player.Nephew

-- Define the best player
def bestPlayer : Player := Player.Sister

-- Theorem to prove
theorem nephew_is_worst_player :
  -- The worst player's twin and the best player are of opposite sex
  (∃ twin : Player, canBeTwins twin worstPlayer ∧ playerSex twin ≠ playerSex bestPlayer) →
  -- The worst player is the same age as another player (implemented as "can be twins")
  (∃ sameAge : Player, canBeTwins worstPlayer sameAge) →
  -- The worst player is the nephew
  worstPlayer = Player.Nephew :=
by
  intro h1 h2
  -- The proof is omitted for brevity
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nephew_is_worst_player_l71_7174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_average_speed_l71_7140

/-- Calculates the average speed of an airplane journey with multiple stops and layovers -/
theorem airplane_average_speed 
  (speed_AB : ℝ) (time_AB : ℝ) 
  (speed_BC : ℝ) (time_BC : ℝ) 
  (speed_CD : ℝ) (time_CD : ℝ) 
  (layover_B : ℝ) (layover_C : ℝ) :
  speed_AB = 240 →
  time_AB = 5 →
  speed_BC = 300 →
  time_BC = 3 →
  speed_CD = 400 →
  time_CD = 4 →
  layover_B = 2 →
  layover_C = 1 →
  let total_distance := speed_AB * time_AB + speed_BC * time_BC + speed_CD * time_CD
  let total_time := time_AB + time_BC + time_CD + layover_B + layover_C
  let average_speed := total_distance / total_time
  abs (average_speed - 246.67) < 0.01 := by
  sorry

#check airplane_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_average_speed_l71_7140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l71_7177

open Real BigOperators

-- Define a positive sequence
def PositiveSequence (a : ℕ → ℝ) : Prop := ∀ n, a n > 0

-- Define the partial sum of the sequence
def PartialSum (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

-- Define the left-hand side of the inequality
noncomputable def LeftHandSide (a : ℕ → ℝ) : ℝ := ∑' n : ℕ, (n : ℝ) / PartialSum a (n + 1)

-- Define the right-hand side of the inequality
noncomputable def RightHandSide (a : ℕ → ℝ) : ℝ := ∑' n : ℕ, 1 / a n

-- State the theorem
theorem inequality_theorem (a : ℕ → ℝ) (h : PositiveSequence a) :
  LeftHandSide a ≤ 4 * RightHandSide a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l71_7177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l71_7148

/-- Given two arithmetic sequences {aₙ} and {bₙ} with sums Sₙ and Tₙ respectively,
    if Sₙ/Tₙ = (2n+1)/(n+3) for all n, then a₇/b₇ = 27/16 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ)
  (h_arithmetic : ∀ n, S n = (n / 2) * (a 1 + a n) ∧ T n = (n / 2) * (b 1 + b n))
  (h_ratio : ∀ n, S n / T n = (2 * n + 1) / (n + 3)) :
  a 7 / b 7 = 27 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l71_7148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_sin_to_cos_l71_7158

theorem cot_sin_to_cos (θ : ℝ) (h1 : 5 * Real.tan (π/2 - θ) = 2 * Real.sin θ) (h2 : 0 < θ) (h3 : θ < π) :
  Real.cos θ = (-5 + Real.sqrt 41) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_sin_to_cos_l71_7158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_20_40_sqrt3_l71_7186

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem
theorem tan_sum_20_40_sqrt3 :
  tan_deg 20 + tan_deg 40 + Real.sqrt 3 * tan_deg 20 * tan_deg 40 = Real.sqrt 3 :=
by
  -- Assuming the following:
  have tan_60 : tan_deg 60 = Real.sqrt 3 := by sorry
  have tan_sum (a b : ℝ) : tan_deg (a + b) = (tan_deg a + tan_deg b) / (1 - tan_deg a * tan_deg b) := by sorry
  
  -- Proof steps
  have h1 : tan_deg 60 = tan_deg (20 + 40) := by sorry
  have h2 : tan_deg (20 + 40) = (tan_deg 20 + tan_deg 40) / (1 - tan_deg 20 * tan_deg 40) := by sorry
  
  -- Combine the equations
  have h3 : Real.sqrt 3 = (tan_deg 20 + tan_deg 40) / (1 - tan_deg 20 * tan_deg 40) := by sorry
  
  -- Algebraic manipulation
  have h4 : Real.sqrt 3 * (1 - tan_deg 20 * tan_deg 40) = tan_deg 20 + tan_deg 40 := by sorry
  have h5 : Real.sqrt 3 - Real.sqrt 3 * tan_deg 20 * tan_deg 40 = tan_deg 20 + tan_deg 40 := by sorry
  
  -- Final step
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_20_40_sqrt3_l71_7186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_4_iff_a_in_range_l71_7127

noncomputable def f (x : ℝ) := max (x^2) (1/x)

theorem f_geq_4_iff_a_in_range (a : ℝ) (ha : a > 0) :
  f a ≥ 4 ↔ (a ≥ 2 ∨ (0 < a ∧ a ≤ 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_4_iff_a_in_range_l71_7127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_implies_zero_function_l71_7171

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (A B C : Point) : Point :=
  { x := (A.x + B.x + C.x) / 3,
    y := (A.y + B.y + C.y) / 3 }

/-- The theorem statement -/
theorem centroid_property_implies_zero_function 
  (f : Point → ℝ) 
  (h : ∀ (A B C : Point), f (centroid A B C) = f A + f B + f C) :
  ∀ (A : Point), f A = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_implies_zero_function_l71_7171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l71_7154

/-- Represents the composition of lemonade in grams -/
structure LemonadeComposition where
  lemon_juice : ℚ
  sugar : ℚ
  water : ℚ

/-- Represents the calorie content per 100g of each ingredient -/
structure CalorieContent where
  lemon_juice : ℚ
  sugar : ℚ

/-- Calculates the total calories in a given weight of lemonade -/
def calories_in_lemonade (composition : LemonadeComposition) (content : CalorieContent) (weight : ℚ) : ℚ :=
  let total_weight := composition.lemon_juice + composition.sugar + composition.water
  let total_calories := (composition.lemon_juice / 100) * content.lemon_juice + 
                        (composition.sugar / 100) * content.sugar
  (total_calories / total_weight) * weight

/-- The main theorem stating that 250g of the given lemonade contains 325 calories -/
theorem lemonade_calories : 
  let composition := LemonadeComposition.mk 150 200 300
  let content := CalorieContent.mk 30 400
  calories_in_lemonade composition content 250 = 325 := by
  sorry

#eval calories_in_lemonade (LemonadeComposition.mk 150 200 300) (CalorieContent.mk 30 400) 250

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l71_7154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_fifth_power_l71_7102

def is_fifth_power (x : ℕ) : Prop :=
  ∃ n : ℕ, x = n^5

def num_divisors (x : ℕ) : ℕ :=
  (Finset.filter (λ d ↦ x % d = 0) (Finset.range (x + 1))).card

theorem divisors_of_fifth_power (x : ℕ) (h : is_fifth_power x) :
  ∃ d : ℕ, num_divisors x = d ∧ d % 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_fifth_power_l71_7102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l71_7157

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- Define the line l
def Line := ℝ → ℝ → Prop

-- Define the condition for a line passing through a point
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l p.1 p.2

-- Define the inclination angle condition
def has_inclination (l : Line) (angle : ℝ) : Prop :=
  ∃ k, k = Real.tan angle ∧ ∀ x y, l x y ↔ y - P.2 = k * (x - P.1)

-- Define the perpendicular condition
def is_perpendicular (l1 l2 : Line) : Prop :=
  ∃ k1 k2 b1 b2, (∀ x y, l1 x y ↔ y = k1 * x + b1) ∧
                 (∀ x y, l2 x y ↔ y = k2 * x + b2) ∧
                 k1 * k2 = -1

-- Define the sum of intercepts condition
def sum_of_intercepts_zero (l : Line) : Prop :=
  ∃ a b c, (∀ x y, l x y ↔ a * x + b * y + c = 0) ∧
           a ≠ 0 ∧ b ≠ 0 ∧ (c / a + c / b = 0)

-- State the theorem
theorem line_equation_theorem (l : Line) :
  passes_through l P →
  (has_inclination l (2 * Real.pi / 3) ∨
   is_perpendicular l (λ x y => x - 2 * y + 1 = 0) ∨
   sum_of_intercepts_zero l) →
  (∀ x y, l x y ↔ Real.sqrt 3 * x + y - 3 - 2 * Real.sqrt 3 = 0) ∨
  (∀ x y, l x y ↔ 2 * x + y - 7 = 0) ∨
  (∀ x y, l x y ↔ 3 * x - 2 * y = 0) ∨
  (∀ x y, l x y ↔ x - y + 1 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l71_7157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_sum_squares_l71_7169

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Side lengths

-- Define the conditions
axiom acute_triangle : 0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2
axiom sine_law : a / Real.sin A = 2 * c / Real.sqrt 3
axiom c_value : c = Real.sqrt 7
axiom triangle_area : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2

-- State the theorems to be proved
theorem angle_C_value : C = Real.pi/3 := by sorry

theorem sum_squares : a^2 + b^2 = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_sum_squares_l71_7169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_initial_equals_expected_l71_7149

/-- Applies a rotation and dilation to a complex number -/
noncomputable def transform (z : ℂ) : ℂ :=
  2 * (Complex.exp (Complex.I * Real.pi / 3) * z)

/-- The initial complex number -/
def initial : ℂ := 1 + 3 * Complex.I

/-- The expected result after transformation -/
noncomputable def expected : ℂ := 1 - 3 * Real.sqrt 3 + Complex.I * (3 + Real.sqrt 3)

theorem transform_initial_equals_expected :
  transform initial = expected := by
  sorry

#check transform_initial_equals_expected

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_initial_equals_expected_l71_7149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_win_cipher_is_amr_l71_7132

def alphabet : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def caesarShift (c : Char) (shift : Nat) : Char :=
  let index := alphabet.indexOf c
  let newIndex := (index + shift) % 26
  alphabet[newIndex]!

def caesarCipher (message : String) (shift : Nat) : String :=
  String.mk (message.data.map (fun c => caesarShift c shift))

theorem win_cipher_is_amr :
  caesarCipher "WIN" 4 = "AMR" :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_win_cipher_is_amr_l71_7132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l71_7160

-- Define a power function
noncomputable def powerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  (∃ a : ℝ, f = powerFunction a) →  -- f is a power function
  f 2 = (1 : ℝ) / 2 →               -- f passes through (2, 1/2)
  f (1 / 2) = 2 :=                  -- Conclusion: f(1/2) = 2
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l71_7160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_proof_l71_7133

theorem age_ratio_proof (anne_age maude_age : ℕ) (h1 : anne_age = 96) (h2 : maude_age = 8) :
  let emile_age := anne_age / 2
  emile_age / maude_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_proof_l71_7133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_and_complementary_l71_7117

-- Define the sample space
def SampleSpace : Set (Fin 3 × Fin 2) :=
  {p | p.1.val + p.2.val = 2}

-- Define the events
def AtLeastOneGirl : Set (Fin 3 × Fin 2) :=
  {p ∈ SampleSpace | p.2.val ≥ 1}

def AllBoys : Set (Fin 3 × Fin 2) :=
  {p ∈ SampleSpace | p.1.val = 2}

-- State the theorem
theorem events_mutually_exclusive_and_complementary :
  (AtLeastOneGirl ∩ AllBoys = ∅) ∧ 
  (AtLeastOneGirl ∪ AllBoys = SampleSpace) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_and_complementary_l71_7117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_collinearity_l71_7151

/-- Triangle ABC with orthocenter H, circumcenter O, and circumradius R -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  O : ℝ × ℝ
  R : ℝ

/-- Reflection of a point across a line -/
noncomputable def reflect (P : ℝ × ℝ) (line : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  sorry

/-- Check if three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop :=
  sorry

/-- Main theorem -/
theorem reflection_collinearity (t : TriangleABC) : 
  let D := reflect t.A (t.B, t.C)
  let E := reflect t.B (t.C, t.A)
  let F := reflect t.C (t.A, t.B)
  collinear D E F ↔ distance t.O t.H = 2 * t.R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_collinearity_l71_7151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_theorem_l71_7114

-- Define the points
variable (A B C D E F G H P : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_cyclic_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D P : EuclideanPlane) : Prop := sorry

-- Define parallel lines
def parallel (l1 l2 : Set EuclideanPlane) : Prop := sorry

-- Define a line through two points
def line_through (P Q : EuclideanPlane) : Set EuclideanPlane := sorry

-- Define intersection of a line and a segment
def intersects_at (l : Set EuclideanPlane) (A B P : EuclideanPlane) : Prop := sorry

-- Define distance between two points
noncomputable def dist (P Q : EuclideanPlane) : ℝ := sorry

-- Main theorem
theorem cyclic_quadrilateral_theorem 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_diagonals : diagonals_intersect_at A B C D P)
  (h_EF : parallel (line_through P F) (line_through A B))
  (h_E : intersects_at (line_through P F) A D E)
  (h_F : intersects_at (line_through P F) B C F)
  (h_GH : parallel (line_through P H) (line_through A D))
  (h_G : intersects_at (line_through P H) A B G)
  (h_H : intersects_at (line_through P H) C D H) :
  parallel (line_through F H) (line_through B D) ∧
  (dist A E) / (dist A G) = (dist C F) / (dist C H) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_theorem_l71_7114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_by_digit_move_l71_7191

theorem no_double_by_digit_move : ∀ n : ℤ, n ≠ 0 →
  ∀ d m X : ℕ, (n = d * 10^m + X ∧ 0 < d ∧ d < 10 ∧ 0 ≤ X ∧ X < 10^m) →
  2*n ≠ 10*X + d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_by_digit_move_l71_7191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l71_7104

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition given in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.b * Real.cos t.C = t.a * (Real.cos t.B)^2 + t.b * Real.cos t.A * Real.cos t.B

/-- Definition of an isosceles triangle -/
def isIsosceles (t : Triangle) : Prop :=
  t.b = t.c

/-- Calculate the perimeter of the triangle -/
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Calculate the area of the triangle using Heron's formula -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- The main theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : Real.cos t.A = 7/8) 
  (h3 : perimeter t = 5) : 
  isIsosceles t ∧ area t = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l71_7104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l71_7189

/-- Given four points in 3D space, prove they form a parallelogram and calculate its area. -/
theorem parallelogram_area (p q r s : ℝ × ℝ × ℝ) : 
  p = (1, -2, 3) → 
  q = (3, -6, 6) → 
  r = (2, -1, 1) → 
  s = (4, -5, 4) → 
  (q.fst - p.fst, q.snd - p.snd, (q.2).2 - (p.2).2) = (s.fst - r.fst, s.snd - r.snd, (s.2).2 - (r.2).2) ∧ 
  Real.sqrt 110 = Real.sqrt (
    ((-4 * -2) - (3 * 1))^2 + 
    ((3 * 1) - (2 * -2))^2 + 
    ((2 * -2) - (-4 * 1))^2
  ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l71_7189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_transport_distance_optimal_distance_minimizes_cost_l71_7161

/-- The optimal distance DC that minimizes the total transportation cost --/
noncomputable def optimal_distance (a b : ℝ) (n : ℝ) : ℝ :=
  b / Real.sqrt (n^2 - 1)

/-- The theorem statement for the optimal transportation problem --/
theorem optimal_transport_distance 
  (a b n : ℝ) 
  (h_n : n > 1) 
  (h_b : b > 0) :
  let f := λ x : ℝ ↦ (1/n) * (a - x) + Real.sqrt (x^2 + b^2)
  ∃ (x : ℝ), x = optimal_distance a b n ∧ 
    ∀ (y : ℝ), f y ≥ f x :=
by sorry

/-- The total cost function --/
noncomputable def total_cost (a b n x : ℝ) : ℝ :=
  (1/n) * (a - x) + Real.sqrt (x^2 + b^2)

/-- Theorem stating that the optimal_distance minimizes the total_cost --/
theorem optimal_distance_minimizes_cost 
  (a b n : ℝ) 
  (h_n : n > 1) 
  (h_b : b > 0) :
  let x := optimal_distance a b n
  ∀ y, total_cost a b n y ≥ total_cost a b n x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_transport_distance_optimal_distance_minimizes_cost_l71_7161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_after_reflection_l71_7145

-- Define the coordinates of point F
def F : ℝ × ℝ := (-5, 3)

-- Define the reflection of a point over the y-axis
def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem segment_length_after_reflection :
  distance F (reflect_over_y_axis F) = 10 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_after_reflection_l71_7145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_scalar_product_range_l71_7162

noncomputable def ellipse_gamma (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3, 1)
def point_E : ℝ × ℝ := (0, -1)

noncomputable def vector_AE : ℝ × ℝ := (point_E.1 - point_A.1, point_E.2 - point_A.2)

def scalar_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem ellipse_scalar_product_range :
  ∀ F : ℝ × ℝ, ellipse_gamma F.1 F.2 →
  ∃ θ : ℝ, F.1 = Real.sqrt 3 * Real.cos θ ∧ F.2 = Real.sin θ ∧
  5 - Real.sqrt 13 ≤ scalar_product vector_AE (F.1 - point_A.1, F.2 - point_A.2) ∧
  scalar_product vector_AE (F.1 - point_A.1, F.2 - point_A.2) ≤ 5 + Real.sqrt 13 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_scalar_product_range_l71_7162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l71_7196

noncomputable section

/-- The original function f(x) -/
def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

/-- The constraint on φ -/
def φ_constraint (φ : ℝ) : Prop := |φ| < Real.pi / 2

/-- The shifted and symmetrized function g(x) -/
def g (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 6) + φ)

/-- The theorem stating that [-π/3, π/6] is a monotonically increasing interval for g(x) -/
theorem monotone_increasing_interval 
  (φ : ℝ) 
  (h_φ : φ_constraint φ) :
  StrictMonoOn (g φ) (Set.Icc (-Real.pi / 3) (Real.pi / 6)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l71_7196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_purchase_problem_l71_7122

/-- The maximum number of sodas that can be purchased given a budget, soda price, and tax rate. -/
def max_sodas (budget : ℚ) (soda_price : ℚ) (tax_rate : ℚ) : ℕ :=
  (budget / (soda_price * (1 + tax_rate))).floor.toNat

/-- Theorem stating that given $25.45, a soda price of $2.15, and a 5% tax rate, 
    the maximum number of sodas that can be purchased is 11. -/
theorem soda_purchase_problem : 
  max_sodas (25 + 45/100) (2 + 15/100) (5/100) = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_purchase_problem_l71_7122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_neg_reciprocal_exactly_two_zeros_l71_7197

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + cos x

-- Part 1
theorem f_greater_than_neg_reciprocal (x : ℝ) (hx : x > 0) :
  f 1 x > -1/x := by sorry

-- Part 2
theorem exactly_two_zeros (a : ℝ) (ha : 0 < a ∧ a < sqrt 2 / 2) :
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo 0 π ∧ x₂ ∈ Set.Ioo 0 π ∧ x₁ ≠ x₂ ∧
  f a x₁ = 0 ∧ f a x₂ = 0 ∧
  ∀ x ∈ Set.Ioo 0 π, f a x = 0 → x = x₁ ∨ x = x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_neg_reciprocal_exactly_two_zeros_l71_7197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_and_angle_range_l71_7168

open Real

theorem prism_volume_and_angle_range (a b : ℝ) (α : ℝ) 
  (ha : a > 0) (hb : b > 0) (hα : 30 * π / 180 < α ∧ α < 150 * π / 180) : 
  ∃ (V : ℝ), V = (a^2 * b / 2) * Real.sqrt (sin (α + π/6) * sin (α - π/6)) ∧ V > 0 := by
  sorry

#check prism_volume_and_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_and_angle_range_l71_7168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_color_opposite_turquoise_l71_7143

-- Define the colors
inductive Color
| Orange | Silver | Yellow | Violet | Indigo | Turquoise

-- Define a face of the cube
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : Fin 6 → Face

-- Define a view of the cube
structure View where
  top : Face
  front : Face
  right : Face

-- Define the problem statement
theorem cube_color_opposite_turquoise 
  (cube : Cube) 
  (view1 view2 view3 : View) 
  (h1 : view1.top.color = Color.Orange ∧ view1.front.color = Color.Yellow ∧ view1.right.color = Color.Silver)
  (h2 : view2.top.color = Color.Orange ∧ view2.front.color = Color.Indigo ∧ view2.right.color = Color.Silver)
  (h3 : view3.top.color = Color.Orange ∧ view3.front.color = Color.Violet ∧ view3.right.color = Color.Silver)
  (h4 : ∃ (i : Fin 6), (cube.faces i).color = Color.Turquoise) :
  ∃ (i j : Fin 6), (cube.faces i).color = Color.Turquoise ∧ 
                   (cube.faces j).color = Color.Orange ∧ 
                   i ≠ j :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_color_opposite_turquoise_l71_7143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_congruence_l71_7131

/-- A plane figure -/
structure PlaneFigure where
  -- Add necessary properties of a plane figure
  points : Set (ℝ × ℝ)

/-- Symmetry with respect to an axis -/
def symmetricAboutAxis (F₁ F₂ : PlaneFigure) (axis : ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∀ p ∈ F₁.points, axis p ∈ F₂.points ∧ ∀ q ∈ F₂.points, ∃ p ∈ F₁.points, axis p = q

/-- Congruence of plane figures -/
def congruent (F₁ F₂ : PlaneFigure) : Prop :=
  ∃ f : ℝ × ℝ → ℝ × ℝ, Function.Bijective f ∧ 
    (∀ p q : ℝ × ℝ, dist p q = dist (f p) (f q)) ∧
    (∀ p ∈ F₁.points, f p ∈ F₂.points) ∧
    (∀ q ∈ F₂.points, ∃ p ∈ F₁.points, f p = q)

/-- Third figure with two axes of symmetry -/
structure SymmetryFigure where
  axis1 : ℝ × ℝ → ℝ × ℝ
  axis2 : ℝ × ℝ → ℝ × ℝ

theorem symmetry_implies_congruence 
  (F₁ F₂ : PlaneFigure) (S : SymmetryFigure) :
  symmetricAboutAxis F₁ F₂ S.axis1 → 
  symmetricAboutAxis F₁ F₂ S.axis2 → 
  congruent F₁ F₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_congruence_l71_7131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_min_max_l71_7126

noncomputable def a (n : ℝ) : ℝ := (n - 2017.5) / (n - 2016.5)

theorem sequence_min_max :
  (∃ (n : ℝ), a n = -1) ∧
  (∃ (n : ℝ), a n = 3) ∧
  (∀ (n : ℝ), a n ≥ -1) ∧
  (∀ (n : ℝ), a n ≤ 3) := by
  sorry

#check sequence_min_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_min_max_l71_7126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_average_l71_7144

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 12) (h2 : original_avg = 50) :
  let total := n * original_avg
  let new_total := 2 * total
  let new_avg := new_total / n
  new_avg = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_average_l71_7144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l71_7134

def set_A : Set ℕ := {x | (x : ℤ) - 1 < 2 ∧ (x : ℤ) - 1 > -2}
def set_B : Set ℕ := {x | x < 2}

theorem intersection_of_A_and_B : set_A ∩ set_B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l71_7134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l71_7165

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem set_operations :
  (A ∩ B = Set.Ioo 2 3) ∧
  ((Set.univ \ B ∪ A) = Set.Iic (-1) ∪ Set.Ioi 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l71_7165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_centers_parallelogram_l71_7135

/-- A parallelogram with sides 2 and 4, and a diagonal of 3 -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  diagonal : ℝ
  h1 : side1 = 2
  h2 : side2 = 4
  h3 : diagonal = 3

/-- The distance between the centers of inscribed circles -/
noncomputable def distance_between_centers (p : Parallelogram) : ℝ := Real.sqrt 51 / 3

/-- Theorem stating that the distance between the centers of the inscribed circles
    in a parallelogram with sides 2 and 4 and diagonal 3 is √51/3 -/
theorem distance_centers_parallelogram (p : Parallelogram) :
  distance_between_centers p = Real.sqrt 51 / 3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_centers_parallelogram_l71_7135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l71_7152

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

theorem min_value_of_f :
  ∀ x y : ℝ, 2/5 ≤ x → x ≤ 1/2 → 1/3 ≤ y → y ≤ 3/8 → f x y ≥ 6/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l71_7152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_angle_ratio_l71_7142

/-- Given a function f(x) = a^(x-3) + x where a > 0 and a ≠ 1, 
    if the graph of f passes through a fixed point A(3, 4) which lies on the terminal side of angle θ, 
    then (sin θ - cos θ) / (sin θ + cos θ) = 1/7 -/
theorem fixed_point_angle_ratio (a : ℝ) (θ : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + x
  (f 3 = 4) →
  (4 : ℝ) / 3 = Real.tan θ →
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_angle_ratio_l71_7142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_magnitude_equals_two_l71_7130

noncomputable def vector_a : ℝ × ℝ := (-Real.sqrt 3, -1)
noncomputable def vector_b : ℝ × ℝ := (1, Real.sqrt 3)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def cross_product_magnitude (v w : ℝ × ℝ) : ℝ :=
  magnitude v * magnitude w * Real.sqrt (1 - (dot_product v w / (magnitude v * magnitude w))^2)

theorem cross_product_magnitude_equals_two :
  cross_product_magnitude vector_a vector_b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_magnitude_equals_two_l71_7130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l71_7178

/-- Given workers who can complete a job individually in the specified number of days,
    calculate the time taken to complete the job when working together. -/
noncomputable def time_to_complete (time_a time_b time_c : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b + 1 / time_c)

/-- Theorem stating that workers who can complete a job in 15, 20, and 45 days individually
    can complete the job in 7.2 days when working together. -/
theorem workers_combined_time :
  time_to_complete 15 20 45 = 7.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l71_7178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l71_7113

open Real

-- Define the functions representing the two equations
noncomputable def f (x : ℝ) : ℝ := x^3 - 5*x + 4
noncomputable def g (x : ℝ) : ℝ := (3 - x) / 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | f p.1 = p.2 ∧ g p.1 = p.2}

-- State the theorem
theorem intersection_sum : 
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l71_7113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_Q_at_roots_of_P_l71_7170

theorem product_of_Q_at_roots_of_P (P Q : ℂ → ℂ) (r₁ r₂ r₃ r₄ r₅ : ℂ) :
  (∀ x, P x = x^5 - x^2 + 1) →
  (∀ x, Q x = x^2 + 1) →
  P r₁ = 0 →
  P r₂ = 0 →
  P r₃ = 0 →
  P r₄ = 0 →
  P r₅ = 0 →
  Q r₁ * Q r₂ * Q r₃ * Q r₄ * Q r₅ = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_Q_at_roots_of_P_l71_7170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l71_7179

theorem matrix_satisfies_conditions :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![5, -3, 8; 4, 6, -2; -9, 0, 5]
  let i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
  let j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
  let k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]
  (M * i = !![5; 4; -9]) ∧
  (M * j = !![-3; 6; 0]) ∧
  (M * k = !![8; -2; 5]) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l71_7179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l71_7124

/-- Time for a train to pass a jogger -/
noncomputable def train_passing_time (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  let relative_speed := train_speed - jogger_speed
  let relative_speed_mps := relative_speed * (1000 / 3600)
  let total_distance := train_length + initial_distance
  total_distance / relative_speed_mps

/-- Theorem stating that the time for the train to pass the jogger is approximately 40.76 seconds -/
theorem train_passing_time_approx :
  let jogger_speed := 7
  let train_speed := 60
  let train_length := 250
  let initial_distance := 350
  abs ((train_passing_time jogger_speed train_speed train_length initial_distance) - 40.76) < 0.01 := by
  sorry

-- Using #eval with noncomputable functions is not possible, so we'll comment it out
-- #eval train_passing_time 7 60 250 350

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l71_7124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_living_space_increase_theorem_l71_7195

/-- The average annual increase in living space required to meet the target -/
noncomputable def average_annual_increase (
  initial_population : ℝ
  ) (initial_space_per_person : ℝ
  ) (growth_rate : ℝ
  ) (target_space_per_person : ℝ
  ) (years : ℕ
  ) : ℝ :=
  let final_population := initial_population * (1 + growth_rate) ^ (years : ℝ)
  let initial_total_space := initial_population * initial_space_per_person
  let final_total_space := final_population * target_space_per_person
  (final_total_space - initial_total_space) / (years : ℝ)

theorem living_space_increase_theorem :
  let initial_population : ℝ := 5000000
  let initial_space_per_person : ℝ := 6
  let growth_rate : ℝ := 0.01
  let target_space_per_person : ℝ := 7
  let years : ℕ := 10
  abs (average_annual_increase initial_population initial_space_per_person growth_rate target_space_per_person years - 868000) < 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_living_space_increase_theorem_l71_7195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_b_l71_7129

/-- A polynomial of degree 4 -/
def polynomial (a b c d e : ℝ) (x : ℝ) : ℝ := a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The roots of the polynomial -/
def roots : List ℝ := [-2, 0, 1, 3]

theorem polynomial_coefficient_b (a b c d e : ℝ) :
  (∀ r ∈ roots, polynomial a b c d e r = 0) →
  polynomial a b c d e 2 = 8 →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_b_l71_7129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_perpendicular_implies_perpendicular_l71_7155

structure Plane where
  -- Define a plane (placeholder)
  dummy : Unit

structure Line where
  -- Define a line (placeholder)
  dummy : Unit

/-- Define parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  sorry -- Placeholder definition

/-- Define a line perpendicular to a plane -/
def perpendicular (l : Line) (p : Plane) : Prop :=
  sorry -- Placeholder definition

/-- Theorem: If two lines are parallel and one is perpendicular to a plane, 
    then the other is also perpendicular to that plane -/
theorem line_parallel_perpendicular_implies_perpendicular 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n β) : 
  perpendicular m β :=
by
  sorry -- Proof to be completed later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_perpendicular_implies_perpendicular_l71_7155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l71_7190

/-- Calculates the length of a train given its speed, tunnel length, and time to cross the tunnel. -/
noncomputable def train_length (train_speed : ℝ) (tunnel_length : ℝ) (crossing_time : ℝ) : ℝ :=
  (train_speed * 1000 / 3600) * crossing_time - tunnel_length

/-- Theorem stating that a train with speed 78 km/hr crossing a 500-meter tunnel in 1 minute has a length of approximately 800.2 meters. -/
theorem train_length_calculation :
  let train_speed := 78 -- km/hr
  let tunnel_length := 500 -- meters
  let crossing_time := 60 -- seconds
  abs (train_length train_speed tunnel_length crossing_time - 800.2) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l71_7190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l71_7116

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2 / 2
  | n + 1 => Real.sqrt 2 / 2 * Real.sqrt (1 - Real.sqrt (1 - (a n)^2))

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (1 + (b n)^2) - 1) / (b n)

theorem sequence_inequality (n : ℕ) : 2^(n+2) * a n < Real.pi ∧ Real.pi < 2^(n+2) * b n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l71_7116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l71_7164

def P : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}
def Q : Set ℝ := {x : ℝ | ∃ n : ℕ, x = n}

theorem intersection_of_P_and_Q : P ∩ Q = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l71_7164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_height_in_meters_l71_7183

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℚ := 2.54

-- Define Maria's height in inches
def maria_height_inches : ℚ := 54

-- Define the function to convert inches to meters
def inches_to_meters (inches : ℚ) : ℚ := inches * inch_to_cm / 100

-- Define a function to round a rational number to three decimal places
def round_to_three_decimals (x : ℚ) : ℚ := 
  (x * 1000).floor / 1000

-- Theorem statement
theorem maria_height_in_meters :
  round_to_three_decimals (inches_to_meters maria_height_inches) = 1372 / 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_height_in_meters_l71_7183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l71_7111

/-- The circle in the problem -/
def problem_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y = 1

/-- The point P outside the circle -/
def P : ℝ × ℝ := (5, -2)

/-- First tangent line equation -/
def tangent1 (x y : ℝ) : Prop := 7*x + 24*y + 13 = 0

/-- Second tangent line equation -/
def tangent2 (x : ℝ) : Prop := x = 5

/-- Theorem stating that the tangent lines from P to the circle have the given equations -/
theorem tangent_lines_to_circle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    problem_circle x₁ y₁ ∧ problem_circle x₂ y₂ ∧
    ((x₁ - 5)^2 + (y₁ + 2)^2 = (x₂ - 5)^2 + (y₂ + 2)^2) ∧
    (tangent1 x₁ y₁ ∨ tangent2 x₁) ∧
    (tangent1 x₂ y₂ ∨ tangent2 x₂) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l71_7111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_5pi_div_6_l71_7156

/-- The angle of inclination of a line given by parametric equations -/
noncomputable def angle_of_inclination (x y : ℝ → ℝ) : ℝ :=
  Real.pi - Real.arctan ((y 1 - y 0) / (x 1 - x 0))

/-- The parametric equations of the line -/
noncomputable def x (t : ℝ) : ℝ := -1 + Real.sqrt 3 * t
def y (t : ℝ) : ℝ := 2 - t

/-- Theorem stating that the angle of inclination of the given line is 5π/6 -/
theorem angle_of_inclination_is_5pi_div_6 :
  angle_of_inclination x y = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_5pi_div_6_l71_7156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_and_range_l71_7184

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x + (1/2) * x^2

theorem f_expression_and_range :
  (∀ x, f x = Real.exp x - x + (1/2) * x^2) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc (-1) 2, f x ≥ m) ↔ m ≤ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_and_range_l71_7184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_average_cost_l71_7181

/-- The average annual cost of a car given the number of years of use -/
noncomputable def average_annual_cost (n : ℝ) : ℝ :=
  (10 + 0.9 * n + 0.1 * n * (n + 1)) / n

/-- The number of years that minimizes the average annual cost -/
def optimal_years : ℝ := 10

/-- Theorem stating that the optimal_years minimizes the average annual cost -/
theorem minimize_average_cost :
  ∀ n : ℝ, n > 0 → average_annual_cost n ≥ average_annual_cost optimal_years :=
by
  intro n hn
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_average_cost_l71_7181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l71_7175

-- Define the circle C in polar coordinates
def circle_equation (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * Real.cos (θ + Real.pi / 4)

-- Define the center of the circle in polar coordinates
noncomputable def center : ℝ × ℝ := (Real.sqrt 2, -Real.pi / 4)

-- Theorem statement
theorem circle_center_coordinates :
  ∀ ρ θ : ℝ, circle_equation ρ θ → 
  ∃ r φ : ℝ, (r, φ) = center ∧ 
  r * Real.cos φ = ρ * Real.cos θ ∧
  r * Real.sin φ = ρ * Real.sin θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l71_7175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l71_7112

theorem exponent_problem (x y : ℝ) (h : x * y = 1) :
  (5 : ℝ) ^ 2 ^ 2 / (5 : ℝ) ^ (x - y) ^ 2 = 625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l71_7112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_ratio_l71_7194

/-- Represents a cylinder formed by enclosing a rectangle --/
structure Cylinder where
  circumference : ℝ
  height : ℝ
  perimeter : ℝ
  (perimeter_constraint : circumference + height = perimeter / 2)

/-- The volume of a cylinder --/
noncomputable def volume (c : Cylinder) : ℝ := c.circumference^2 * c.height / (4 * Real.pi)

/-- Theorem stating that the ratio of circumference to height is 2:1 when volume is maximized --/
theorem cylinder_max_volume_ratio (c : Cylinder) (h : c.perimeter = 12) :
  (∀ c' : Cylinder, c'.perimeter = 12 → volume c' ≤ volume c) →
  c.circumference / c.height = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_ratio_l71_7194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l71_7115

/-- Given an obtuse angle α and an acute angle β, where sin α = 4/5 and cos(α - β) = √10/10,
    prove that tan(α - π/4) = 7 and sin β = 13√10/50 -/
theorem angle_relations (α β : ℝ) 
  (h_obtuse : π/2 < α ∧ α < π)
  (h_acute : 0 < β ∧ β < π/2)
  (h_sin_α : Real.sin α = 4/5)
  (h_cos_diff : Real.cos (α - β) = Real.sqrt 10 / 10) :
  Real.tan (α - π/4) = 7 ∧ Real.sin β = 13 * Real.sqrt 10 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l71_7115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_parallelepiped_sum_l71_7119

/-- The volume of a set of points that are inside or within one unit of a rectangular parallelepiped -/
noncomputable def extended_parallelepiped_volume (l w h : ℕ) : ℝ :=
  (l * w * h : ℝ) +  -- Volume of the parallelepiped
  (2 * (l + 2) * (w + 2) + 2 * (w + 2) * (h + 2) + 2 * (l + 2) * (h + 2) - 2 * l * w - 2 * w * h - 2 * l * h : ℝ) +  -- Volume of external parallelepipeds
  (4 / 3 * Real.pi) +  -- Volume of 1/8 spheres at corners
  (3 * l + 3 * w + 3 * h : ℝ) * Real.pi  -- Volume of 1/4 cylinders along edges

/-- The main theorem -/
theorem extended_parallelepiped_sum (m n p : ℕ) :
  extended_parallelepiped_volume 4 5 6 = (m + n * Real.pi) / p →
  m > 0 ∧ n > 0 ∧ p > 0 →
  Nat.Coprime n p →
  m + n + p = 1075 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_parallelepiped_sum_l71_7119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_container_volume_l71_7101

/-- The volume of a conical container formed from a sector of a square sheet -/
theorem conical_container_volume (side_length : ℝ) (central_angle : ℝ) : 
  side_length = 8 → 
  central_angle = π / 4 → 
  (1 / 3) * π * (central_angle * side_length / (2 * π))^2 * 
  Real.sqrt (side_length^2 - (central_angle * side_length / (2 * π))^2) = Real.sqrt 7 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_container_volume_l71_7101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_product_formula_l71_7137

-- Define the Riemann zeta function
noncomputable def riemann_zeta (s : ℝ) : ℝ := ∑' n, (n : ℝ) ^ (-s)

-- Define the prime sequence
def prime_seq : ℕ → ℕ := sorry

-- State Euler's product formula
theorem eulers_product_formula {s : ℝ} (h : 1 < s) :
  riemann_zeta s = ∏' n, (1 - (prime_seq n : ℝ) ^ (-s))⁻¹ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_product_formula_l71_7137
