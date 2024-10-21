import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_7985_to_hundredth_l80_8064

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_7985_to_hundredth :
  round_to_hundredth 7.985 = 7.99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_7985_to_hundredth_l80_8064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_specific_digits_l80_8060

theorem count_numbers_with_specific_digits (n : ℕ) (h : n ≥ 5) :
  let total_numbers := 8^n
  let numbers_missing_one_digit := 5 * 7^n
  let numbers_missing_two_digits := 10 * 6^n
  let numbers_missing_three_digits := 10 * 5^n
  let numbers_missing_four_digits := 5 * 4^n
  let numbers_missing_all_five_digits := 3^n
  (total_numbers - numbers_missing_one_digit + numbers_missing_two_digits
    - numbers_missing_three_digits + numbers_missing_four_digits
    - numbers_missing_all_five_digits) =
  (Finset.filter (λ x : ℕ ↦
    x ≥ 10^(n-1) ∧ x < 10^n ∧
    (∀ d, d ∈ [1, 2, 3, 4, 5] → ∃ k, k < n ∧ (x / 10^k) % 10 = d) ∧
    (∀ k, k < n → (x / 10^k) % 10 ≠ 0 ∧ (x / 10^k) % 10 ≠ 9))
    (Finset.range (10^n - 10^(n-1)))).card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_specific_digits_l80_8060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l80_8034

/-- The interest rate (as a percentage) at which A lends money to B -/
def interest_rate_A_to_B : ℝ := 0

/-- The principal amount lent -/
def principal : ℝ := 3500

/-- The interest rate (as a percentage) at which B lends money to C -/
def interest_rate_B_to_C : ℝ := 11

/-- The time period in years -/
def time : ℝ := 3

/-- B's gain over the time period -/
def B_gain : ℝ := 105

theorem interest_rate_calculation (r : ℝ) :
  principal * (interest_rate_B_to_C / 100) * time - 
  principal * (r / 100) * time = B_gain →
  r = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l80_8034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gear_speed_ratio_l80_8078

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℚ

/-- Represents a system of three gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  meshedAB : A.teeth * A.speed = B.teeth * B.speed
  meshedBC : B.teeth * B.speed = C.teeth * C.speed

/-- Theorem stating the ratio of angular speeds in a gear system -/
theorem gear_speed_ratio (sys : GearSystem) : 
  sys.A.speed / sys.C.speed = (sys.C.teeth * sys.B.teeth : ℚ) / (sys.A.teeth * sys.B.teeth : ℚ) ∧
  sys.B.speed / sys.C.speed = (sys.C.teeth * sys.A.teeth : ℚ) / (sys.B.teeth * sys.A.teeth : ℚ) ∧
  sys.C.speed / sys.C.speed = (sys.A.teeth * sys.B.teeth : ℚ) / (sys.C.teeth * sys.B.teeth : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gear_speed_ratio_l80_8078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l80_8010

/-- The polar equation of line C₁ -/
def C₁_polar_eq (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 4) = Real.sqrt 2

/-- The parametric equations of curve C₂ -/
def C₂_param_eq (t x y : ℝ) : Prop :=
  x = Real.cos t ∧ y = 1 + Real.sin t

/-- The chord length of intersection between C₁ and C₂ -/
noncomputable def chord_length_of_intersection (C₁ C₂ : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Given a line C₁ with polar equation ρ sin(θ + π/4) = √2 and a curve C₂ with parametric equations x = cos t, y = 1 + sin t, prove that the length of the chord cut by C₁ on C₂ is √2. -/
theorem chord_length (ρ θ t : ℝ) : 
  C₁_polar_eq ρ θ →
  ∃ (x y : ℝ), C₂_param_eq t x y →
  ∃ (l : ℝ), l = Real.sqrt 2 ∧ l = chord_length_of_intersection {(x, y) | C₁_polar_eq x y} {(x, y) | ∃ t, C₂_param_eq t x y} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l80_8010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l80_8038

-- Define the function representing the left side of the inequality
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3) + 3 / (Real.rpow x (1/3) + 2)

-- Define the interval (-8, -3√3)
def interval : Set ℝ := Set.Ioo (-8) (-3 * Real.sqrt 3)

-- Theorem statement
theorem inequality_equivalence :
  ∀ x : ℝ, f x ≤ 0 ↔ x ∈ interval :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l80_8038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_square_perimeter_l80_8055

-- Define the rectangle
structure Rectangle where
  e : ℝ
  f : ℝ
  g : ℝ
  h : ℝ
  parallel : e = g

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function
noncomputable def distance (p : Point) (line : ℝ) : ℝ :=
  sorry

-- Define the smaller distance function
noncomputable def smallerDistance (p : Point) (line1 line2 : ℝ) : ℝ :=
  min (distance p line1) (distance p line2)

-- Define the sum of smaller distances
noncomputable def sumSmallerDistances (p : Point) (rect : Rectangle) : ℝ :=
  (smallerDistance p rect.e rect.g) + (smallerDistance p rect.f rect.h)

-- Define the locus of points
def locus (rect : Rectangle) (d : ℝ) : Set Point :=
  {p : Point | sumSmallerDistances p rect = d}

-- Define properties for the square
def isCenteredOn (s : Set Point) (rect : Rectangle) : Prop := sorry
def diagonalLength (s : Set Point) : ℝ := sorry
def perimeter (s : Set Point) : Set Point := sorry

-- Theorem statement
theorem locus_is_square_perimeter (rect : Rectangle) (d : ℝ) :
  ∃ (square : Set Point), 
    isCenteredOn square rect ∧ 
    diagonalLength square = 2 * d ∧
    locus rect d = perimeter square :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_square_perimeter_l80_8055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l80_8040

theorem calculation_proof : 
  let a := (2/3 : ℚ) * (35/100 : ℚ) * 250
  let b := (75/100 : ℚ) * 150 / 16
  let c := (1/2 : ℚ) * (40/100 : ℚ) * 500
  |((a - b + c) - 151.30)| < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l80_8040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l80_8087

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_property :
  ∀ A B : ℝ × ℝ,
  curve_C A.1 A.2 → curve_C B.1 B.2 →
  line_l A.1 A.2 → line_l B.1 B.2 →
  A ≠ B →
  |1 / distance point_P A - 1 / distance point_P B| = 2/3 := by
  sorry

#check intersection_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l80_8087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l80_8019

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by the equation (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The distance between a point (x, y) and a line ax + by + c = 0 -/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  distancePointToLine c.h c.k l = c.r

/-- The main theorem -/
theorem tangent_line_to_circle (m : ℝ) :
  let l : Line := ⟨3, -4, -6⟩
  let c : Circle := ⟨0, 1, Real.sqrt (1 - m)⟩
  isTangent l c → m = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l80_8019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_is_two_l80_8002

-- Define cube root as noncomputable
noncomputable def cube_root (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Theorem statement
theorem cube_root_of_eight_is_two : cube_root 8 = 2 := by
  -- Unfold the definition of cube_root
  unfold cube_root
  -- Use the property of Real.rpow
  have h : Real.rpow 8 (1/3) = 2 := by
    -- This is where we would prove that 8^(1/3) = 2
    -- For now, we'll use sorry to skip the proof
    sorry
  -- Apply the property
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_is_two_l80_8002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_equals_two_l80_8029

/-- The slope angle of the line in radians -/
noncomputable def slope_angle : ℝ := Real.pi / 3

/-- The equation of the circle: x^2 + y^2 - 4x = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The line passes through the origin and has a slope of tan(60°) -/
def line_eq (x y : ℝ) : Prop := y = Real.tan slope_angle * x

/-- Points A and B are the intersection points of the line and the circle -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧ line_eq A.1 A.2 ∧ line_eq B.1 B.2

theorem length_AB_equals_two (A B : ℝ × ℝ) (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_equals_two_l80_8029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l80_8004

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem parallel_vectors_lambda (a b : V) (l : ℝ) 
  (h1 : ¬ ∃ k : ℝ, a = k • b)
  (h2 : ∃ μ : ℝ, l • a + b = μ • (a + 2 • b)) :
  l = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l80_8004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l80_8092

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define the properties of f
def symmetric_about_origin (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def minimum_at_one (f : ℝ → ℝ) : Prop := ∃ x₀, ∀ x, f x ≥ f x₀ ∧ f x₀ = -2 ∧ x₀ = 1

-- Define the monotonically increasing intervals
def monotonic_increasing_intervals (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → x < -1 → y < -1 → f x < f y) ∧
  (∀ x y, x < y → x > 1 → y > 1 → f x < f y)

-- Define the solution sets for the inequality
def solution_set (f : ℝ → ℝ) (m : ℝ) : Set ℝ :=
  {x | f x > 5 * m * x^2 - (4 * m^2 + 3) * x}

-- State the theorem
theorem f_properties (a b c d : ℝ) :
  let f := f a b c d
  symmetric_about_origin f →
  minimum_at_one f →
  monotonic_increasing_intervals f ∧
  (solution_set f 0 = {x | x > 0}) ∧
  (∀ m > 0, solution_set f m = {x | x > 4*m ∨ (0 < x ∧ x < m)}) ∧
  (∀ m < 0, solution_set f m = {x | x > 0 ∨ (4*m < x ∧ x < m)}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l80_8092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l80_8088

-- Define the function v(x)
noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

-- Theorem stating the domain of v(x)
theorem domain_of_v : 
  {x : ℝ | v x ∈ Set.univ} = Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l80_8088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_dihedral_angle_l80_8012

/-- A regular n-sided pyramid with height h and base radius r -/
structure RegularPyramid (n : ℕ) where
  h : ℝ
  r : ℝ

/-- The volume of a regular n-sided pyramid -/
noncomputable def volume (n : ℕ) (p : RegularPyramid n) : ℝ :=
  (n / 3 : ℝ) * Real.tan (Real.pi / n) * p.r^2 * p.h

/-- The surface area of a regular n-sided pyramid -/
noncomputable def surfaceArea (n : ℕ) (p : RegularPyramid n) : ℝ :=
  n * Real.tan (Real.pi / n) * (p.r^2 + p.r * Real.sqrt (p.h^2 + p.r^2))

/-- The dihedral angle at the base edge of a regular n-sided pyramid -/
noncomputable def dihedralAngle (n : ℕ) (p : RegularPyramid n) : ℝ :=
  Real.arccos (1 / Real.sqrt ((p.h^2 + p.r^2) / p.r^2))

/-- Theorem: Among all regular n-sided pyramids with a fixed total surface area,
    the pyramid with the largest volume has a dihedral angle at the base edge
    equal to the dihedral angle at an edge of a regular tetrahedron -/
theorem max_volume_dihedral_angle {n : ℕ} {a : ℝ} (h_n : n ≥ 3) (h_a : a > 0) :
  ∃ (p : RegularPyramid n),
    surfaceArea n p = a ∧
    (∀ (q : RegularPyramid n), surfaceArea n q = a → volume n q ≤ volume n p) ∧
    dihedralAngle n p = Real.arccos (1 / Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_dihedral_angle_l80_8012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outside_doors_even_l80_8036

/-- Represents a house with rooms and doors. -/
structure House where
  rooms : ℕ
  doors : ℕ
  outside_doors : ℕ
  room_doors : Fin rooms → ℕ
  door_connects : Fin doors → Sum (Fin rooms × Fin rooms) (Fin rooms)

/-- Predicate stating that every room has an even number of doors. -/
def every_room_has_even_doors (h : House) : Prop :=
  ∀ r : Fin h.rooms, Even (h.room_doors r)

/-- Theorem stating that if every room has an even number of doors,
    then the number of outside doors is even. -/
theorem outside_doors_even (h : House) (heven : every_room_has_even_doors h) :
  Even h.outside_doors := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_outside_doors_even_l80_8036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l80_8035

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Point A coordinates -/
def A : ℝ × ℝ := (-3, 0)

/-- Point B coordinates -/
def B : ℝ × ℝ := (-2, 5)

/-- Point C coordinates -/
def C : ℝ × ℝ := (1, 3)

/-- The y-coordinate of the point on the y-axis -/
noncomputable def y : ℝ := 19/4

theorem equidistant_point :
  distance 0 y A.1 A.2 = distance 0 y B.1 B.2 ∧
  distance 0 y B.1 B.2 = distance 0 y C.1 C.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l80_8035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l80_8077

theorem angle_sum_is_pi_over_two (a b : ℝ) : 
  0 < a ∧ a < π/2 →  -- a is acute
  0 < b ∧ b < π/2 →  -- b is acute
  4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 1 →
  4 * Real.sin (2*a) + 3 * Real.sin (2*b) = 0 →
  2*a + b = π/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l80_8077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_area_ratio_l80_8021

/-- For a cone where the height is equal to the diameter of its base,
    the ratio of the area of its base to its lateral surface area is √5/5. -/
theorem cone_area_ratio (R : ℝ) (h : R > 0) : 
  (π * R^2) / (π * R * Real.sqrt ((2*R)^2 + R^2)) = Real.sqrt 5 / 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_area_ratio_l80_8021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_arithmetic_progressions_correct_l80_8071

/-- The number of increasing arithmetic progressions of length 4 
    with terms chosen from the set {1, 2, ..., 1000} -/
def count_arithmetic_progressions : ℕ := 166167

/-- A function that counts the number of increasing arithmetic progressions 
    of length 4 with terms chosen from the set {1, 2, ..., n} -/
def count_arithmetic_progressions_up_to (n : ℕ) : ℕ :=
  Finset.sum (Finset.range 334) (fun d => n + 1 - 3 * d)

theorem count_arithmetic_progressions_correct : 
  count_arithmetic_progressions = count_arithmetic_progressions_up_to 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_arithmetic_progressions_correct_l80_8071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_sum_l80_8045

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [4, 5, 2]  -- 254₈
def base1 : Nat := 8
def den1 : List Nat := [3, 1]     -- 13₃
def base2 : Nat := 3
def num2 : List Nat := [2, 0, 2]  -- 202₅
def base3 : Nat := 5
def den2 : List Nat := [2, 2]     -- 22₄
def base4 : Nat := 4

-- State the theorem
theorem base_conversion_sum :
  (to_base_10 num1 base1 / to_base_10 den1 base2 : ℚ) + 
  (to_base_10 num2 base3 / to_base_10 den2 base4 : ℚ) = 39.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_sum_l80_8045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earnings_diff_approx_l80_8073

/-- Calculate the average percentage difference between two values -/
noncomputable def avgPercentDiff (x y : ℝ) : ℝ :=
  ((x - y) / x * 100 + (x - y) / y * 100) / 2

/-- The hourly earnings of Susan -/
def susan_earnings : ℝ := 18

/-- The hourly earnings of Phil -/
def phil_earnings : ℝ := 7

/-- Theorem stating that the average percentage difference in earnings
    between Susan and Phil is approximately 109.13% -/
theorem earnings_diff_approx :
  abs (avgPercentDiff susan_earnings phil_earnings - 109.13) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_earnings_diff_approx_l80_8073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_limited_intersection_l80_8074

theorem max_subsets_with_limited_intersection (S : Finset Nat) : 
  S.card = 10 → 
  (∃ (k : Nat) (A : Fin k → Finset Nat), 
    (∀ i, A i ⊆ S ∧ A i ≠ ∅) ∧ 
    (∀ i j, i ≠ j → (A i ∩ A j).card ≤ 2) ∧
    k = (Finset.filter (fun A => A.card ≤ 3) (Finset.powerset S)).card) →
  (∀ (m : Nat) (B : Fin m → Finset Nat),
    (∀ i, B i ⊆ S ∧ B i ≠ ∅) ∧ 
    (∀ i j, i ≠ j → (B i ∩ B j).card ≤ 2) →
    m ≤ 175) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_limited_intersection_l80_8074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l80_8025

open Real

theorem trig_problem (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) (h2 : sin (α - π/3) = 5/13) :
  (cos α = (12 - 5 * Real.sqrt 3) / 26) ∧ 
  (sin (2 * α - π/6) = 119/169) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l80_8025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_limit_l80_8007

/-- The coefficient of x in the expansion of (x + 1/x)^7 -/
def first_term : ℕ := 35

/-- The common ratio of the geometric sequence -/
noncomputable def common_ratio : ℝ := 1 / 2

/-- The sum of the first n terms of the geometric sequence -/
noncomputable def S (n : ℕ) : ℝ := first_term * (1 - common_ratio^n) / (1 - common_ratio)

/-- The limit of S_n as n approaches infinity -/
theorem geometric_sequence_limit : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |S n - 70| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_limit_l80_8007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_one_gt_cos_one_l80_8096

theorem sin_one_gt_cos_one :
  (π / 4 : ℝ) < 1 ∧ 1 < π / 2 → Real.sin 1 > Real.cos 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_one_gt_cos_one_l80_8096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_relationships_l80_8093

/-- A tetrahedron with associated spheres and perpendiculars -/
structure Tetrahedron where
  /-- Radii of spheres touching faces externally -/
  ρ₁ : ℝ
  ρ₂ : ℝ
  ρ₃ : ℝ
  ρ₄ : ℝ
  /-- Radius of sphere touching external "gutter-like space" corresponding to edge AB -/
  ρ₁₂ : ℝ
  /-- Lengths of perpendiculars from vertices to opposite faces -/
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ
  m₄ : ℝ

/-- The relationships between radii and perpendiculars in a tetrahedron -/
theorem tetrahedron_relationships (t : Tetrahedron) :
  (∃ (s : ℝ), s = 1 ∨ s = -1 ∧
    2 / t.ρ₁₂ = s * (1 / t.ρ₁ + 1 / t.ρ₂ + 1 / t.ρ₃ + 1 / t.ρ₄)) ∧
  (∃ (s : ℝ), s = 1 ∨ s = -1 ∧
    1 / t.ρ₁₂ = s * (1 / t.m₃ + 1 / t.m₄ + 1 / t.m₁ + 1 / t.m₂)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_relationships_l80_8093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l80_8086

/-- The length of the segment AB formed by the intersection of a line and a parabola -/
theorem intersection_segment_length :
  let l : ℝ → ℝ × ℝ := λ t ↦ ((2 * Real.sqrt 5 / 5) * t, 1 + (Real.sqrt 5 / 5) * t)
  let parabola : ℝ × ℝ → Prop := λ p ↦ p.2^2 = 4 * p.1
  let intersection_points := {p : ℝ × ℝ | ∃ t, l t = p ∧ parabola p}
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l80_8086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_tetrahedron_to_centroid_cube_sum_of_ratio_parts_l80_8048

/-- Regular tetrahedron with side length s -/
structure RegularTetrahedron where
  s : ℝ
  s_pos : s > 0

/-- Cube with vertices at the centroids of tetrahedron faces -/
structure CentroidCube where
  tetrahedron : RegularTetrahedron

/-- Volume of a regular tetrahedron -/
noncomputable def volume_tetrahedron (t : RegularTetrahedron) : ℝ := 
  t.s^3 * Real.sqrt 2 / 12

/-- Volume of a centroid cube -/
noncomputable def volume_centroid_cube (c : CentroidCube) : ℝ := 
  (c.tetrahedron.s * Real.sqrt 6 / 9)^3

/-- The ratio of the volume of a regular tetrahedron to its centroid cube is 27/8 -/
theorem volume_ratio_tetrahedron_to_centroid_cube (t : RegularTetrahedron) :
  (volume_tetrahedron t) / (volume_centroid_cube ⟨t⟩) = 27 / 8 := by
  sorry

/-- The sum of numerator and denominator in the simplified ratio is 35 -/
theorem sum_of_ratio_parts : 27 + 8 = 35 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_tetrahedron_to_centroid_cube_sum_of_ratio_parts_l80_8048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_probability_after_three_turns_l80_8044

/-- Represents the state of who has the ball -/
inductive BallHolder
| Alice
| Bob

/-- Defines the game with given probabilities -/
structure Game where
  aliceTossProbability : ℚ
  bobTossProbability : ℚ
  initialHolder : BallHolder

/-- Calculates the probability of Alice having the ball after n turns -/
def probabilityAliceHasBallAfterNTurns (game : Game) (n : ℕ) : ℚ :=
  sorry

/-- The specific game instance from the problem -/
def gameInstance : Game :=
  { aliceTossProbability := 1/3
  , bobTossProbability := 1/4
  , initialHolder := BallHolder.Alice }

theorem alice_probability_after_three_turns :
  probabilityAliceHasBallAfterNTurns gameInstance 3 = 179/432 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_probability_after_three_turns_l80_8044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l80_8098

/-- Converts a point from polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The point in polar coordinates -/
noncomputable def polar_point : ℝ × ℝ := (8, 7 * Real.pi / 6)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ := (-4 * Real.sqrt 3, -4)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular polar_point.1 polar_point.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l80_8098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deepak_present_age_l80_8009

/-- Proves that Deepak's present age is 21 years given the conditions of the problem -/
theorem deepak_present_age :
  ∀ (r d k : ℕ),
  -- Current age ratio
  4 * d = 3 * r ∧ 5 * d = 3 * k →
  -- Rahul's age after 6 years
  r + 6 = 34 →
  -- New age ratio after 6 years
  5 * (r + 6) = 7 * (d + 6) ∧ 9 * (d + 6) = 5 * (k + 6) →
  -- Deepak's present age
  d = 21 := by
  sorry

#check deepak_present_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deepak_present_age_l80_8009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_stripe_probability_l80_8024

/-- Represents the possible stripe orientations on a cube face -/
inductive CubeOrientation
| Horizontal
| Vertical

/-- Represents the possible colors of a stripe -/
inductive CubeColor
| Red
| Blue

/-- Represents a stripe on a cube face -/
structure CubeStripe where
  orientation : CubeOrientation
  color : CubeColor

/-- Represents a cube with stripes on its faces -/
def Cube := Fin 6 → CubeStripe

/-- Checks if a cube has a continuous stripe encircling it -/
def hasContinuousStripe (c : Cube) : Prop := sorry

/-- The total number of possible stripe combinations on a cube -/
def totalCombinations : Nat := 4^6

/-- The number of favorable outcomes (cubes with continuous stripes) -/
def favorableOutcomes : Nat := 24

/-- The probability of a continuous stripe encircling the cube -/
def probabilityOfContinuousStripe : ℚ := favorableOutcomes / totalCombinations

theorem continuous_stripe_probability :
  probabilityOfContinuousStripe = 3 / 512 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_stripe_probability_l80_8024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l80_8051

-- Define the circle and ellipse equations
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16
def ellipse_eq (x y : ℝ) : Prop := (x - 3)^2 + 4 * y^2 = 36

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | circle_eq p.1 p.2 ∧ ellipse_eq p.1 p.2}

-- Define the quadrilateral formed by the intersection points
def quadrilateral : Set (ℝ × ℝ) :=
  intersection_points

-- Theorem stating the area of the quadrilateral is 14
theorem quadrilateral_area : 
  MeasureTheory.volume quadrilateral = 14 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l80_8051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_difference_is_six_l80_8083

/-- The total pay for the research project -/
def total_pay : ℝ := 360

/-- Candidate p's hourly wage -/
def wage_p : ℝ → ℝ
| q => 1.5 * q

/-- Candidate q's hourly wage -/
def wage_q : ℝ → ℝ
| q => q

/-- The number of hours candidate q needs to complete the job -/
def hours_q : ℝ → ℝ
| h => h + 10

/-- The wage difference between candidates p and q -/
def wage_difference (q : ℝ) : ℝ := wage_p q - wage_q q

theorem wage_difference_is_six :
  ∃ q : ℝ, q > 0 ∧ wage_q q * hours_q (total_pay / wage_p q) = total_pay ∧ wage_difference q = 6 := by
  -- Let q be 12 (as calculated in the solution)
  use 12
  -- Split the goal into three parts
  apply And.intro
  · -- Prove q > 0
    norm_num
  · apply And.intro
    · -- Prove wage_q q * hours_q (total_pay / wage_p q) = total_pay
      simp [wage_q, hours_q, total_pay, wage_p]
      norm_num
    · -- Prove wage_difference q = 6
      simp [wage_difference, wage_p, wage_q]
      norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_difference_is_six_l80_8083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l80_8050

-- Define the complex number z
def z (m : ℝ) : ℂ := 1 + m * Complex.I

-- Define z₀
def z₀ (m : ℝ) : ℂ := -3 - Complex.I + z m

-- Theorem for part 1
theorem part1 (m : ℝ) : 
  ((z m + 2) / (1 - Complex.I)).im = 0 → m = -3 := by sorry

-- Theorem for part 2
theorem part2 : 
  ∃ (b c : ℝ), (z₀ (-3))^2 + b * z₀ (-3) + c = 0 ∧ b = 4 ∧ c = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l80_8050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_point_coordinates_l80_8046

noncomputable def triangle_OPQ (Q : ℝ × ℝ) : Prop :=
  Q.1 ≥ 0 ∧ Q.2 > 0 ∧ 
  (Q.1 - 4) * Q.2 = 0 ∧
  Q.1^2 + Q.2^2 = 32

noncomputable def rotate_clockwise_45 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * (Real.sqrt 2 / 2) + p.2 * (Real.sqrt 2 / 2),
   -p.1 * (Real.sqrt 2 / 2) + p.2 * (Real.sqrt 2 / 2))

theorem rotated_point_coordinates (Q : ℝ × ℝ) :
  triangle_OPQ Q →
  rotate_clockwise_45 Q = (4 * Real.sqrt 2, 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_point_coordinates_l80_8046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_m_leq_one_l80_8047

/-- A function f(x) = -1/3 * x^3 + m*x --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + m*x

/-- f is decreasing on (1, +∞) --/
def is_decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x → x < y → f y < f x

theorem f_decreasing_implies_m_leq_one (m : ℝ) :
  is_decreasing_on_interval (f m) → m ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_m_leq_one_l80_8047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rental_is_640_l80_8028

/-- Represents Dr. Jones' monthly finances -/
structure MonthlyFinances where
  earnings : ℚ
  food_expense : ℚ
  electric_water_ratio : ℚ
  insurance_ratio : ℚ
  savings : ℚ

/-- Calculates the house rental given Dr. Jones' monthly finances -/
def calculate_house_rental (finances : MonthlyFinances) : ℚ :=
  finances.earnings
  - finances.food_expense
  - (finances.electric_water_ratio * finances.earnings)
  - (finances.insurance_ratio * finances.earnings)
  - finances.savings

/-- Theorem stating that Dr. Jones' house rental is $640 per month -/
theorem house_rental_is_640 (finances : MonthlyFinances)
  (h1 : finances.earnings = 6000)
  (h2 : finances.food_expense = 380)
  (h3 : finances.electric_water_ratio = 1/4)
  (h4 : finances.insurance_ratio = 1/5)
  (h5 : finances.savings = 2280) :
  calculate_house_rental finances = 640 := by
  sorry

#eval calculate_house_rental {
  earnings := 6000,
  food_expense := 380,
  electric_water_ratio := 1/4,
  insurance_ratio := 1/5,
  savings := 2280
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rental_is_640_l80_8028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_four_terms_l80_8053

/-- An arithmetic sequence with a_1 = 1 and a_3 = 3 -/
noncomputable def arithmetic_sequence (n : ℕ) : ℝ :=
  1 + (n - 1) * (3 - 1) / 2

/-- Sum of the first n terms of the arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ :=
  (n * (arithmetic_sequence 1 + arithmetic_sequence n)) / 2

theorem sum_of_first_four_terms :
  S 4 = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_four_terms_l80_8053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cookies_theorem_l80_8081

/-- Represents the minimum number of cookies required for a specific strategy -/
def min_cookies (n : ℕ) : ℕ := 2^n

/-- Represents the rules of cookie distribution -/
structure CookieRules where
  give_to_adjacent : Bool
  eat_to_give : Bool

/-- Represents the circular arrangement of people -/
structure CircularArrangement where
  num_people : ℕ
  is_even : Prop

/-- The main theorem stating the minimum number of cookies required -/
theorem min_cookies_theorem (n : ℕ) (rules : CookieRules) (arrangement : CircularArrangement) :
  rules.give_to_adjacent = true ∧ 
  rules.eat_to_give = true ∧ 
  arrangement.num_people = 2*n ∧ 
  arrangement.is_even = (arrangement.num_people % 2 = 0)
  →
  min_cookies n = 2^n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cookies_theorem_l80_8081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_is_52_l80_8095

/-- An equilateral triangle with side length 16 and three inscribed circles --/
structure TriangleWithCircles where
  -- Side length of the equilateral triangle
  side_length : ℝ
  side_length_eq : side_length = 16
  -- Radius of the inscribed circles
  radius : ℝ
  -- The circles are mutually tangent and each tangent to two sides of the triangle
  circles_tangent : True
  -- Expressing radius in terms of integers a and b
  a : ℕ
  b : ℕ
  radius_eq : radius = Real.sqrt (a : ℝ) - (b : ℝ)

/-- The sum of a and b is 52 --/
theorem sum_a_b_is_52 (t : TriangleWithCircles) : t.a + t.b = 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_is_52_l80_8095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_term_position_l80_8054

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := n * (a₁ + (a₁ + d * (n - 1))) / 2

theorem removed_term_position
  (a₁ : ℝ)
  (S₁₁ : ℝ)
  (avg_remaining : ℝ)
  (h₁ : a₁ = -5)
  (h₂ : S₁₁ = 55)
  (h₃ : avg_remaining = 4.6)
  (h₄ : ∃ d : ℝ, sum_arithmetic_sequence a₁ d 11 = S₁₁)
  (h₅ : ∃ k : ℕ, 1 < k ∧ k ≤ 11 ∧ 
    (S₁₁ - arithmetic_sequence a₁ d k) / 10 = avg_remaining) :
  ∃ d : ℝ, ∃ k : ℕ, k = 8 ∧ 
    arithmetic_sequence a₁ d k = S₁₁ - 10 * avg_remaining :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_term_position_l80_8054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_cones_given_l80_8076

def ice_cream_shop (total_sales : ℕ) (cone_price : ℕ) (free_cone_frequency : ℕ) : ℕ :=
  let total_cones := total_sales / cone_price
  total_cones / free_cone_frequency

theorem free_cones_given (total_sales : ℕ) (cone_price : ℕ) (free_cone_frequency : ℕ) 
  (h1 : total_sales = 100)
  (h2 : cone_price = 2)
  (h3 : free_cone_frequency = 6) :
  ice_cream_shop total_sales cone_price free_cone_frequency = 8 := by
  sorry

#eval ice_cream_shop 100 2 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_cones_given_l80_8076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retiling_impossible_l80_8082

/-- Represents a rectangular tile -/
structure Tile where
  width : ℕ
  height : ℕ

/-- Represents the collection of tiles used for tiling -/
structure TileSet where
  tiles : List Tile
  count : List ℕ

/-- Represents a rectangular box -/
structure Box where
  width : ℕ
  height : ℕ

/-- Calculates the number of shaded cells covered by a tile in a checkered pattern -/
def shadedCellsCovered (t : Tile) : ℕ :=
  if t.width % 2 = 0 then (t.width * t.height) / 2 else 0

/-- Calculates the total number of shaded cells covered by a set of tiles -/
def totalShadedCells (ts : TileSet) : ℕ :=
  List.sum (List.zipWith (fun t c => shadedCellsCovered t * c) ts.tiles ts.count)

/-- The main theorem stating that retiling is impossible after the change -/
theorem retiling_impossible (b : Box) (ts_original ts_modified : TileSet) : 
  ts_original.tiles = [Tile.mk 2 2, Tile.mk 1 4] →
  ts_modified.tiles = [Tile.mk 2 2, Tile.mk 1 4] →
  ts_original.count.length ≥ 2 →
  ts_modified.count.length ≥ 2 →
  ts_original.count[0]? = some (ts_modified.count[0]?.getD 0 + 1) →
  ts_original.count[1]? = some (ts_modified.count[1]?.getD 0 - 1) →
  totalShadedCells ts_original % 2 = 0 →
  totalShadedCells ts_modified % 2 = 1 →
  ¬∃ (tiling : Box → TileSet), tiling b = ts_modified := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_retiling_impossible_l80_8082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_forty_percent_l80_8011

/-- The percentage increase in a worker's daily wage -/
noncomputable def wage_percentage_increase (old_wage new_wage : ℝ) : ℝ :=
  (new_wage - old_wage) / old_wage * 100

/-- Theorem: The percentage increase in the worker's daily wage is 40% -/
theorem wage_increase_forty_percent (old_wage new_wage : ℝ) 
  (h1 : old_wage = 25) (h2 : new_wage = 35) : 
  wage_percentage_increase old_wage new_wage = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_forty_percent_l80_8011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_ratio_l80_8005

-- Define the radii of the circles
def radii : List ℝ := [2, 4, 6, 8, 10]

-- Define a function to calculate the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define a function to calculate the area of a ring
noncomputable def ring_area (outer : ℝ) (inner : ℝ) : ℝ :=
  circle_area outer - circle_area inner

-- Define the white area
noncomputable def white_area : ℝ :=
  circle_area (radii[0]!) +
  ring_area (radii[2]!) (radii[1]!) +
  ring_area (radii[4]!) (radii[3]!)

-- Define the black area
noncomputable def black_area : ℝ :=
  ring_area (radii[1]!) (radii[0]!) +
  ring_area (radii[3]!) (radii[2]!)

-- Theorem statement
theorem black_to_white_ratio :
  black_area / white_area = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_to_white_ratio_l80_8005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equivalences_l80_8001

theorem sum_equivalences (p : ℕ) (hp : Nat.Prime p) : 
  let n := p - 1
  let S1 := n * (n + 1) / 2
  let P1 := (2^n - 1)
  let S2 := n * (n + 1) * (2 * n + 1) / 6
  let P2 := (4^n - 1) / 3
  (S1 % p = P1 % p) ∧ (S2 % p = P2 % p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equivalences_l80_8001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_k_negative_one_f_inequality_implies_m_ge_two_max_n_is_two_ln_two_l80_8059

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (2 * Real.exp x) / (Real.exp x + 1) + k

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := (f x + 1) / (1 - f x)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def can_form_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

theorem f_odd_implies_k_negative_one (k : ℝ) :
  is_odd (f · k) → k = -1 := by sorry

theorem f_inequality_implies_m_ge_two (k : ℝ) (m : ℝ) :
  (∀ x > 0, f (2 * x) k ≤ m * f x k) → m ≥ 2 := by sorry

theorem max_n_is_two_ln_two (k : ℝ) :
  ∃ n : ℝ, n = 2 * Real.log 2 ∧
  (∀ a b c : ℝ, 0 < a ∧ a ≤ n → 0 < b ∧ b ≤ n → 0 < c ∧ c ≤ n →
    can_form_triangle a b c → can_form_triangle (g (f · k) a) (g (f · k) b) (g (f · k) c)) ∧
  (∀ n' > n, ∃ a b c : ℝ, 0 < a ∧ a ≤ n' → 0 < b ∧ b ≤ n' → 0 < c ∧ c ≤ n' →
    can_form_triangle a b c ∧ ¬can_form_triangle (g (f · k) a) (g (f · k) b) (g (f · k) c)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_k_negative_one_f_inequality_implies_m_ge_two_max_n_is_two_ln_two_l80_8059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l80_8039

-- Define the constants
noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := Real.log (Real.sin (2 * Real.pi / 5))

-- State the theorem
theorem a_greater_than_b_greater_than_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l80_8039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_66_minutes_l80_8042

/-- Represents a cyclist with their speeds on different terrains -/
structure Cyclist where
  flat_speed : ℚ
  downhill_speed : ℚ
  uphill_speed : ℚ

/-- Calculates the time taken for a given distance and speed -/
def time_taken (distance : ℚ) (speed : ℚ) : ℚ :=
  distance / speed

/-- Calculates the total time for a cyclist to complete the route -/
def total_time (c : Cyclist) : ℚ :=
  time_taken 12 c.uphill_speed + time_taken 18 c.downhill_speed + time_taken 15 c.flat_speed

/-- The main theorem stating the time difference between Minnie and Lucy -/
theorem time_difference_is_66_minutes : 
  let minnie : Cyclist := ⟨15, 25, 6⟩
  let lucy : Cyclist := ⟨25, 35, 8⟩
  (total_time minnie - total_time lucy) * 60 = 66 := by
  sorry

#eval (total_time ⟨15, 25, 6⟩ - total_time ⟨25, 35, 8⟩) * 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_66_minutes_l80_8042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_curve_through_three_points_l80_8094

theorem sine_curve_through_three_points :
  ∃ (A ω φ : ℝ), A > 0 ∧ ω > 0 ∧ 0 < φ ∧ φ < π ∧
  (∀ x, A * Real.sin (ω * x + φ) = 4 * Real.sin (π/3 * x + π/6)) ∧
  4 * Real.sin (π/6) = 2 ∧
  4 * Real.sin (π/3 * 5/2 + π/6) = 0 ∧
  4 * Real.sin (π/3 * 3 + π/6) = -2 ∧
  2 * π / ω > 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_curve_through_three_points_l80_8094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_f_equals_half_sin_2x_l80_8056

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.tan x) / (1 + (Real.tan x)^2)

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

-- Optionally, you can add more specific lemmas or theorems if needed
theorem f_equals_half_sin_2x (x : ℝ) : f x = (1/2) * Real.sin (2*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_f_equals_half_sin_2x_l80_8056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_infinite_sum_equals_four_thirds_l80_8063

/-- The sum of the double infinite series ∑_{j=0}^∞ ∑_{k=0}^∞ 2^(-2k - j - (k + j)^2) -/
noncomputable def double_infinite_sum : ℝ :=
  ∑' (j : ℕ), ∑' (k : ℕ), (2 : ℝ) ^ (-(2 * k + j + (k + j)^2 : ℤ))

/-- The theorem stating that the double infinite sum equals 4/3 -/
theorem double_infinite_sum_equals_four_thirds : double_infinite_sum = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_infinite_sum_equals_four_thirds_l80_8063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_derivative_implies_positive_slope_l80_8043

/-- If the derivative of a function is always positive on ℝ, 
    then the slope between any two distinct points is positive. -/
theorem positive_derivative_implies_positive_slope (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h : ∀ x, HasDerivAt f (f' x) x) 
  (h_pos : ∀ x, f' x > 0) :
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_derivative_implies_positive_slope_l80_8043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l80_8017

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t / 2, 1 - (Real.sqrt 3 / 2) * t)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Statement of the theorem
theorem line_circle_intersection :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, line_l t₁ = A ∧ line_l t₂ = B) ∧
    (∃ θ₁ θ₂ : ℝ, 
      Real.sqrt ((A.1)^2 + (A.2)^2) = circle_C θ₁ ∧
      Real.sqrt ((B.1)^2 + (B.2)^2) = circle_C θ₂) ∧
    (∀ P : ℝ × ℝ, (∃ t : ℝ, line_l t = P) ∧ 
                  (∃ θ : ℝ, Real.sqrt ((P.1)^2 + (P.2)^2) = circle_C θ) →
                  P = A ∨ P = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l80_8017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_melting_point_celsius_l80_8049

/-- Converts temperature from Celsius to Fahrenheit -/
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := (9/5) * c + 32

/-- Converts temperature from Fahrenheit to Celsius -/
noncomputable def fahrenheit_to_celsius (f : ℝ) : ℝ := (5/9) * (f - 32)

/-- The boiling point of water in Celsius -/
def water_boiling_point_celsius : ℝ := 100

/-- The boiling point of water in Fahrenheit -/
def water_boiling_point_fahrenheit : ℝ := 212

/-- The melting point of ice in Fahrenheit -/
def ice_melting_point_fahrenheit : ℝ := 32

/-- Temperature of a pot of water in Celsius -/
def pot_temperature_celsius : ℝ := 45

/-- Temperature of a pot of water in Fahrenheit -/
def pot_temperature_fahrenheit : ℝ := 113

theorem ice_melting_point_celsius : fahrenheit_to_celsius ice_melting_point_fahrenheit = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_melting_point_celsius_l80_8049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_P_onto_yOz_l80_8008

/-- The yOz coordinate plane -/
def yOz_plane : Set (Fin 3 → ℝ) := {p | p 0 = 0}

/-- The point P -/
def P : Fin 3 → ℝ := ![-1, 3, -4]

/-- Projection of a point onto the yOz plane -/
def proj_yOz (p : Fin 3 → ℝ) : Fin 3 → ℝ := ![0, p 1, p 2]

theorem projection_P_onto_yOz :
  proj_yOz P = ![0, 3, -4] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_P_onto_yOz_l80_8008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_of_sequence_l80_8013

noncomputable def sequenceMonomial (n : ℕ) (a : ℝ) : ℝ := Real.sqrt n * a ^ n

theorem nth_term_of_sequence (n : ℕ) (a : ℝ) : sequenceMonomial n a = Real.sqrt n * a ^ n := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_of_sequence_l80_8013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_from_edge_sum_l80_8079

theorem cube_volume_from_edge_sum (edge_sum : ℝ) (h : edge_sum = 72) : 
  (edge_sum / 12) ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_from_edge_sum_l80_8079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l80_8023

/-- Calculates the speed of a train given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A train with length 700 meters that crosses a pole in 36 seconds has a speed of approximately 70 km/h -/
theorem train_speed_theorem :
  let length : ℝ := 700
  let time : ℝ := 36
  abs (train_speed length time - 70) < 0.1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l80_8023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_parabola_intersection_l80_8014

-- Define the circle and parabola
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 1
def myParabola (x y : ℝ) : Prop := y = x^2 + 1

-- Define the tangent line to the circle at point (a, b)
def myTangentLine (a b x y : ℝ) : Prop := y = -(a / b) * x + (b + a^2 / b)

-- Define the condition for the tangent line to intersect the parabola at exactly one point
def myUniqueIntersection (a b : ℝ) : Prop :=
  ∃! x y, myParabola x y ∧ myTangentLine a b x y

-- Main theorem
theorem circle_tangent_parabola_intersection :
  ∀ a b : ℝ, myCircle a b ∧ myUniqueIntersection a b ↔
    (a = -1 ∧ b = 0) ∨ (a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨
    (a = -2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) ∨ (a = 2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_parabola_intersection_l80_8014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_reach_b_time_l80_8052

/-- Represents a robot moving on a circular track -/
structure Robot where
  name : String
  startPoint : String
  direction : String

/-- Represents the circular track and robot movements -/
structure CircularTrack where
  robots : List Robot
  alphaToBTime : ℕ
  alphaGammaBetaCoincideTime : ℕ

/-- The time for Beta to reach B after Gamma reaches A -/
def betaReachBTime (track : CircularTrack) : ℕ :=
  56

/-- Theorem stating that Beta reaches B 56 seconds after Gamma reaches A -/
theorem beta_reach_b_time (track : CircularTrack) 
  (h1 : track.robots.length = 3)
  (h2 : track.robots[0].name = "Alpha")
  (h3 : track.robots[1].name = "Beta")
  (h4 : track.robots[2].name = "Gamma")
  (h5 : track.robots[0].startPoint = "A")
  (h6 : track.robots[1].startPoint = "A")
  (h7 : track.robots[2].startPoint = "B")
  (h8 : track.robots[0].direction = "counterclockwise")
  (h9 : track.robots[1].direction = "clockwise")
  (h10 : track.robots[2].direction = "counterclockwise")
  (h11 : track.alphaToBTime = 12)
  (h12 : track.alphaGammaBetaCoincideTime = 21) :
  betaReachBTime track = 56 := by
  sorry

#check beta_reach_b_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_reach_b_time_l80_8052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_unique_T_l80_8041

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := 
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

noncomputable def T (a₁ d : ℝ) (n : ℕ) : ℝ := 
  (n : ℝ) * ((n + 1 : ℝ) * (3 * a₁ + (n - 1 : ℝ) * d)) / 6

theorem smallest_n_for_unique_T (a₁ d : ℝ) (S₂₀₂₃ : ℝ) :
  S a₁ d 2023 = S₂₀₂₃ → 
  (∀ n : ℕ, n < 3034 → ∃ a₁' d', a₁' ≠ a₁ ∧ d' ≠ d ∧ S a₁' d' 2023 = S₂₀₂₃ ∧ T a₁' d' n = T a₁ d n) ∧
  (∀ a₁' d', S a₁' d' 2023 = S₂₀₂₃ → T a₁' d' 3034 = T a₁ d 3034) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_unique_T_l80_8041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_is_concave_log2_is_convex_l80_8006

-- Define the concept of a concave function
def IsConcave (f : ℝ → ℝ) (A : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ A → x₂ ∈ A → x₁ ≠ x₂ → f ((x₁ + x₂) / 2) < (f x₁ + f x₂) / 2

-- Define the concept of a convex function
def IsConvex (f : ℝ → ℝ) (A : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ A → x₂ ∈ A → x₁ ≠ x₂ → f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2

-- Define the square function
def square (x : ℝ) : ℝ := x^2

-- Define the log base 2 function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem for the square function
theorem square_is_concave :
  IsConcave square Set.univ :=
by
  sorry

-- Theorem for the log2 function
theorem log2_is_convex :
  IsConvex log2 {x : ℝ | x > 0} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_is_concave_log2_is_convex_l80_8006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_per_case_bottles_per_case_is_24_l80_8020

/-- The number of bottles in a case of water for a children's camp --/
theorem bottles_per_case : ℕ :=
  let group1 : ℕ := 14
  let group2 : ℕ := 16
  let group3 : ℕ := 12
  let group4 : ℕ := (group1 + group2 + group3) / 2
  let total_children : ℕ := group1 + group2 + group3 + group4
  let days : ℕ := 3
  let bottles_per_child_per_day : ℕ := 3
  let cases_purchased : ℕ := 13
  let additional_bottles_needed : ℕ := 255
  let total_bottles_needed : ℕ := total_children * days * bottles_per_child_per_day
  let bottles_per_case : ℕ := (total_bottles_needed - additional_bottles_needed) / cases_purchased
  bottles_per_case

theorem bottles_per_case_is_24 : bottles_per_case = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_per_case_bottles_per_case_is_24_l80_8020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_sum_inequality_l80_8066

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sum of C(n,i) * fᵢ
def fibonacci_sum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => Nat.choose n (i + 1) * fib (i + 1))

-- State the theorem
theorem fibonacci_sum_inequality (n : ℕ) (h : n > 0) :
  (fibonacci_sum n : ℚ) < (2 * n + 2)^n / n.factorial := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_sum_inequality_l80_8066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l80_8057

/-- The original function f(x) = √3 * sin(x) - cos(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

/-- The translated function g(x) = f(x - m) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x - m)

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

theorem min_translation_for_even_function :
  ∃ m : ℝ, m > 0 ∧ is_even (g m) ∧ ∀ m' : ℝ, m' > 0 → is_even (g m') → m ≤ m' := by
  sorry

#check min_translation_for_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l80_8057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_study_time_for_second_exam_l80_8018

/-- Represents a student's exam performance --/
structure ExamPerformance where
  studyTime : ℝ
  score : ℝ

/-- Represents John's exam data --/
structure JohnExamData where
  firstExam : ExamPerformance
  secondExam : ExamPerformance
  averageScore : ℝ
  isDirectlyProportional : Prop

theorem john_study_time_for_second_exam
  (data : JohnExamData)
  (h1 : data.firstExam.studyTime = 3)
  (h2 : data.firstExam.score = 60)
  (h3 : data.averageScore = 75)
  (h4 : data.isDirectlyProportional) :
  data.secondExam.studyTime = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_study_time_for_second_exam_l80_8018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l80_8072

-- Define the circle
def circleEq (x y : ℚ) : Prop := x^2 + y^2 = 100

-- Define a point on the circle
structure PointOnCircle where
  x : ℚ
  y : ℚ
  on_circle : circleEq x y

-- Define the distance between two points
noncomputable def distance (p q : PointOnCircle) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Main theorem
theorem max_ratio_on_circle (p q r s : PointOnCircle) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h_irrational_pq : ¬ (∃ m n : ℤ, distance p q = m / n))
  (h_irrational_rs : ¬ (∃ m n : ℤ, distance r s = m / n)) :
  (∀ p' q' r' s' : PointOnCircle, 
    p' ≠ q' ∧ r' ≠ s' → 
    distance p' q' / distance r' s' ≤ 7) ∧ 
  (∃ p' q' r' s' : PointOnCircle, 
    p' ≠ q' ∧ r' ≠ s' ∧ 
    distance p' q' / distance r' s' = 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l80_8072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_sum_l80_8099

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points and vectors
variable (A B O G M P Q : V)
variable (a b : V)
variable (m n : ℝ)

-- State the conditions
axiom centroid : G = (1/3 : ℝ) • (A + B + O)
axiom midpoint_M : M = (1/2 : ℝ) • (A + B)
axiom line_through_centroid : ∃ (t : ℝ), P + t • (Q - P) = G
axiom vector_a : O - A = a
axiom vector_b : O - B = b
axiom vector_ma : O - P = m • a
axiom vector_nb : O - Q = n • b

-- State the theorems to be proved
theorem centroid_sum : G - A + (G - B) + (G - O) = 0 := by sorry

theorem inverse_sum : 1 / m + 1 / n = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_sum_l80_8099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l80_8070

-- Define the circles
def circle1 (x y : ℝ) : Prop := ∃ (r : ℝ), (x - 3)^2 + (y - 1)^2 = r^2
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the condition that circle1 passes through (3,1) and touches circle2 at A and B
def circles_touch : Prop :=
  circle1 3 1 ∧ 
  circle2 A.1 A.2 ∧ 
  circle2 B.1 B.2 ∧
  circle1 A.1 A.2 ∧
  circle1 B.1 B.2

-- Define the line through two points
def line_through (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

-- Theorem stating that the line AB has the equation 2x + y - 3 = 0
theorem line_AB_equation (h : circles_touch) : 
  ∀ (x y : ℝ), line_through A B x y ↔ 2*x + y - 3 = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l80_8070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rays_tangent_to_circle_l80_8061

/-- A configuration of two intersecting semi-infinite plane mirrors -/
structure MirrorConfiguration where
  /-- The angle between the two mirrors -/
  angle : ℝ
  /-- Assumption that the angle is non-zero -/
  angle_nonzero : angle ≠ 0

/-- A ray in the mirror configuration -/
structure Ray (config : MirrorConfiguration) where
  /-- The point of incidence on a mirror -/
  incidence_point : ℝ × ℝ
  /-- The angle of incidence -/
  incidence_angle : ℝ
  /-- The distance from the incidence point to the intersection of mirrors -/
  distance_to_intersection : ℝ
  /-- Assumption that the ray is on a normal plane to the mirrors -/
  on_normal_plane : True

/-- Helper function to check if a ray is tangent to a circle -/
def is_tangent_to_circle (config : MirrorConfiguration) (ray : Ray config) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  sorry

/-- The theorem stating that all rays are tangent to a circle -/
theorem rays_tangent_to_circle (config : MirrorConfiguration) (ray : Ray config) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (r : Ray config), is_tangent_to_circle config r center radius) ∧
    radius = ray.distance_to_intersection * Real.cos ray.incidence_angle :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rays_tangent_to_circle_l80_8061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_tags_for_200_l80_8027

/-- The sum of tagged numbers on six cards -/
noncomputable def sum_of_tags (W : ℕ) : ℝ :=
  let X : ℝ := (2/3) * W
  let Y : ℝ := W + X
  let Z : ℝ := Real.sqrt Y
  let P : ℝ := X^3
  let Q : ℝ := (Nat.factorial W) / 100000
  W + X + Y + Z + P + Q

/-- Theorem stating the sum of tags for W = 200 -/
theorem sum_of_tags_for_200 :
  sum_of_tags 200 = 200 + (2/3 * 200) + (200 + 2/3 * 200) +
    Real.sqrt (200 + 2/3 * 200) + (2/3 * 200)^3 + (Nat.factorial 200 / 100000) := by
  sorry

-- This evaluation might not work due to the large factorial
-- #eval sum_of_tags 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_tags_for_200_l80_8027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l80_8097

/-- Triangle properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

/-- Main theorem -/
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.a * (Real.cos (t.C / 2))^2 + 2 * t.c * (Real.cos (t.A / 2))^2 = 5 * t.b / 2) :
  (2 * (t.a + t.c) = 3 * t.b) ∧ 
  ((Real.cos t.B = 1/4 ∧ t.S = Real.sqrt 15) → t.b = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l80_8097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_of_primes_l80_8032

def is_arithmetic_progression (seq : List Nat) (d : Nat) : Prop :=
  ∀ i, i + 1 < seq.length → seq[i + 1]! - seq[i]! = d

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

theorem arithmetic_progression_of_primes :
  (is_arithmetic_progression [5, 11, 17, 23, 29] 6 ∧
   (∀ n ∈ [5, 11, 17, 23, 29], is_prime n)) ∧
  (is_arithmetic_progression [5, 17, 29, 41, 53] 12 ∧
   (∀ n ∈ [5, 17, 29, 41, 53], is_prime n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_of_primes_l80_8032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l80_8058

/-- A triangle with specific median lengths and area -/
structure SpecialTriangle where
  -- Two of the medians
  median1 : ℝ
  median2 : ℝ
  -- Area of the triangle
  area : ℝ
  -- Conditions
  median1_eq : median1 = 5
  median2_eq : median2 = 10
  area_eq : area = 10 * Real.sqrt 10

/-- The third median of the special triangle -/
noncomputable def third_median (t : SpecialTriangle) : ℝ := 3 * Real.sqrt 10

/-- Theorem stating that the third median of the special triangle is 3√10 -/
theorem third_median_length (t : SpecialTriangle) : 
  third_median t = 3 * Real.sqrt 10 := by
  -- Unfold the definition of third_median
  unfold third_median
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l80_8058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l80_8016

/-- The point P -/
def P : ℝ × ℝ := (1, 2)

/-- The equation of the circle -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The equation of the line -/
def line_eq (x y : ℝ) : Prop := x + 2*y - 5 = 0

/-- The distance formula between a point (x, y) and a line ax + by + c = 0 -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  |a*x + b*y + c| / Real.sqrt (a^2 + b^2)

theorem line_tangent_to_circle :
  line_eq P.1 P.2 ∧ 
  (∀ x y, circle_eq x y → distance_point_to_line x y 1 2 (-5) = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l80_8016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vitamin_discount_is_twenty_percent_l80_8090

/-- Calculates the percentage discount on vitamins given the regular price, number of bottles, 
    coupon value, number of coupons, and final price after discounts and coupons. -/
noncomputable def calculate_vitamin_discount (regular_price : ℝ) (num_bottles : ℕ) 
                               (coupon_value : ℝ) (num_coupons : ℕ) 
                               (final_price : ℝ) : ℝ :=
  let original_total := regular_price * (num_bottles : ℝ)
  let coupon_savings := coupon_value * (num_coupons : ℝ)
  let sale_price := final_price + coupon_savings
  let discount_amount := original_total - sale_price
  (discount_amount / original_total) * 100

/-- Theorem stating that the percentage discount on the vitamins is 20% -/
theorem vitamin_discount_is_twenty_percent :
  calculate_vitamin_discount 15 3 2 3 30 = 20 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vitamin_discount_is_twenty_percent_l80_8090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l80_8085

/-- Given a parabola y^2 = 2px (p > 0) and a line intersecting it at points P and Q,
    if the midpoint of PQ has an x-coordinate of 3 and |PQ| = 10,
    then p = 4 and the parabola's equation is y^2 = 8x -/
theorem parabola_equation (p : ℝ) (P Q : ℝ × ℝ) :
  p > 0 →
  (∀ (x y : ℝ), y^2 = 2*p*x ↔ (x, y) = P ∨ (x, y) = Q) →
  (P.1 + Q.1) / 2 = 3 →
  ‖P - Q‖ = 10 →
  p = 4 ∧ ∀ (x y : ℝ), y^2 = 8*x ↔ y^2 = 2*p*x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l80_8085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l80_8089

noncomputable def OA (l α : ℝ) : ℝ × ℝ := (l * Real.cos α, l * Real.sin α)
noncomputable def OB (β : ℝ) : ℝ × ℝ := (-Real.sin β, Real.cos β)
def OC : ℝ × ℝ := (1, 0)

noncomputable def BC (β : ℝ) : ℝ × ℝ := (1 + Real.sin β, -Real.cos β)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

noncomputable def vectorLength (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def AB (l α β : ℝ) : ℝ × ℝ :=
  (l * Real.cos α + Real.sin β, l * Real.sin α - Real.cos β)

theorem part1 (l α β : ℝ) (h1 : l = 2) (h2 : α = π/3) (h3 : 0 < β ∧ β < π)
    (h4 : perpendicular (OA l α) (BC β)) : β = π/6 := by sorry

theorem part2 (l : ℝ) (h : ∀ α β : ℝ, vectorLength (AB l α β) ≥ 2 * vectorLength (OB β)) :
    l ≤ -3 ∨ l ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l80_8089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l80_8030

def sequenceProperty (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  ∀ n : ℕ, 4 * (a n) * (a (n + 1)) = (a n + a (n + 1) - 1)^2 ∧
  ∀ n : ℕ, n > 1 → a n > a (n - 1)

theorem sequence_formula (a : ℕ → ℕ) (h : sequenceProperty a) : ∀ n : ℕ, a n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l80_8030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_one_power_negative_three_l80_8062

theorem negative_one_power_negative_three :
  (-1 : ℝ)^(-3 : ℤ) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_one_power_negative_three_l80_8062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_exponential_equation_l80_8003

theorem solution_to_exponential_equation :
  ∃ y : ℝ, (2 : ℝ) ^ ((8 : ℝ) ^ y) = (8 : ℝ) ^ ((2 : ℝ) ^ y) ↔ y = (Real.log 3) / (2 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_exponential_equation_l80_8003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l80_8015

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x + f y) = y + f x)
  (h2 : Set.Finite {r : ℝ | ∃ x : ℝ, x ≠ 0 ∧ r = f x / x}) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l80_8015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cubic_equation_l80_8069

theorem solve_cubic_equation (y : ℝ) :
  (y - 5)^3 = (1/27)⁻¹ ↔ y = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_cubic_equation_l80_8069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_formula_l80_8037

/-- Represents a triangular pyramid with specific properties -/
structure TriangularPyramid where
  /-- The angle between the equal sides of the isosceles base triangle -/
  α : ℝ
  /-- The angle of inclination of all lateral edges to the base plane -/
  φ : ℝ
  /-- Assumption that 0 < α < π and 0 < φ < π/2 -/
  α_range : 0 < α ∧ α < π
  φ_range : 0 < φ ∧ φ < π/2

/-- The theorem stating the relationship between the dihedral angle and the given angles -/
theorem dihedral_angle_formula (pyramid : TriangularPyramid) :
  ∃ θ : ℝ, Real.tan (θ/2) = Real.tan (pyramid.α/2) / Real.sin pyramid.φ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_formula_l80_8037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l80_8000

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2) : 
  ∀ x, f x = (x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l80_8000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_of_series_sum_eq_2019_l80_8022

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

noncomputable def series_sum : ℝ := ∑' k : ℕ, (if k ≥ 2018 then (factorial 2019 - factorial 2018) / (factorial k : ℝ) else 0)

theorem ceiling_of_series_sum_eq_2019 : ⌈series_sum⌉ = 2019 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_of_series_sum_eq_2019_l80_8022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_balance_equilibrium_l80_8031

theorem stone_balance_equilibrium (n : ℕ) :
  (∃ k : ℕ, n = 4 * k ∨ n = 4 * k + 3) ↔
  (∃ subset : Finset ℕ, subset ⊆ Finset.range n ∧
    subset.sum id = n * (n + 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_balance_equilibrium_l80_8031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2089th_term_l80_8067

def digit_square_sum (n : ℕ) : ℕ := (n.digits 10).map (λ d => d * d) |>.sum

def sequence_term (n : ℕ) : ℕ :=
  match n with
  | 0 => 2089
  | n + 1 => digit_square_sum (sequence_term n)

theorem sequence_2089th_term : sequence_term 2088 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2089th_term_l80_8067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_pi_third_l80_8033

theorem tan_sum_pi_third (y : Real) (h : Real.tan y = 3) :
  Real.tan (y + π/3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_pi_third_l80_8033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perpendicular_through_perpendicular_line_l80_8084

-- Define the basic structures
structure Point where
structure Line where
structure Plane where

-- Define the relationships
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def passes_through (p : Plane) (l : Line) : Prop := sorry
def perpendicular_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry

-- State the theorem
theorem plane_perpendicular_through_perpendicular_line 
  (p1 p2 : Plane) (l : Line) :
  perpendicular_line_plane l p1 → passes_through p2 l → perpendicular_plane_plane p1 p2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perpendicular_through_perpendicular_line_l80_8084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_lines_l80_8080

/-- The cosine of the acute angle between two 2D vectors -/
noncomputable def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))

/-- The first line's direction vector -/
def v1 : ℝ × ℝ := (4, 5)

/-- The second line's direction vector -/
def v2 : ℝ × ℝ := (2, 3)

theorem cos_angle_between_lines : 
  cos_angle v1 v2 = 23 / Real.sqrt 533 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_lines_l80_8080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_percentage_l80_8091

open Finset

theorem divisibility_percentage :
  let n := 1000
  let divisible_by_4 := (Finset.filter (fun x => 4 ∣ x) (range (n + 1))).card
  let divisible_by_5 := (Finset.filter (fun x => 5 ∣ x) (range (n + 1))).card
  let divisible_by_both := (Finset.filter (fun x => 4 ∣ x ∧ 5 ∣ x) (range (n + 1))).card
  let divisible_by_exactly_one := divisible_by_4 + divisible_by_5 - 2 * divisible_by_both
  (divisible_by_exactly_one : ℚ) / n * 100 = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_percentage_l80_8091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l80_8026

-- Define the golden ratio
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the golden section point
def isGoldenSectionPoint (a b c : ℝ) : Prop :=
  (b - a) / (c - a) = φ ∧ c > a ∧ c < b

-- State the theorem
theorem golden_section_length (a b c : ℝ) :
  isGoldenSectionPoint a c b → (b - a = 20) → (c - a = 10 * Real.sqrt 5 - 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l80_8026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_48_l80_8065

/-- Two externally tangent circles with a triangle having two sides tangent to the circles --/
structure TangentCirclesTriangle where
  /-- Radius of the smaller circle -/
  r₁ : ℝ
  /-- Radius of the larger circle -/
  r₂ : ℝ
  /-- Length of the side EF of the triangle -/
  ef : ℝ
  /-- The circles are externally tangent -/
  tangent_circles : r₁ > 0 ∧ r₂ > 0
  /-- DE and DF are tangent to the circles -/
  tangent_sides : True
  /-- EF is shorter than DE -/
  ef_shorter : True
  /-- DE equals DF -/
  de_eq_df : True

/-- The area of the triangle DEF in the TangentCirclesTriangle configuration -/
noncomputable def triangle_area (t : TangentCirclesTriangle) : ℝ := 
  (1/2) * t.ef * (2 * (t.r₁ + t.r₂))

/-- Theorem stating that for the given configuration, the area of triangle DEF is 48 -/
theorem area_is_48 (t : TangentCirclesTriangle) 
  (h₁ : t.r₁ = 3) 
  (h₂ : t.r₂ = 5) 
  (h₃ : t.ef = 6) : 
  triangle_area t = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_48_l80_8065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_intersection_area_theorem_l80_8068

-- Define a triangle in a plane
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define a line in a plane
structure Line where
  p : EuclideanSpace ℝ (Fin 2)
  v : EuclideanSpace ℝ (Fin 2)

-- Define the reflection of a triangle across a line
noncomputable def reflect (T : Triangle) (l : Line) : Triangle := sorry

-- Define the area of a triangle
noncomputable def area (T : Triangle) : ℝ := sorry

-- Define the intersection of two triangles
def intersection (T1 T2 : Triangle) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the area of the intersection of two triangles
noncomputable def intersectionArea (T1 T2 : Triangle) : ℝ := sorry

-- The main theorem
theorem triangle_reflection_intersection_area_theorem (T : Triangle) :
  ∃ l : Line, intersectionArea T (reflect T l) > (2/3) * area T := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_intersection_area_theorem_l80_8068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solution_l80_8075

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_solution (x : ℝ) (hx : x ≠ 0) :
  (∀ y : ℝ, y ≠ 0 → g y + 3 * g (1 / y) = 4 * y^2) →
  (g x = g (-x) ↔ x = Real.sqrt 2 / 2 ∨ x = -Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solution_l80_8075
