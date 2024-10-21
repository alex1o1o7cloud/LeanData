import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mix_ratio_depends_on_volumes_l1019_101905

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℝ
  denominator : ℝ
  denominator_pos : denominator > 0

/-- Represents a jar containing an alcohol-water solution -/
structure Jar where
  volume : ℝ
  ratio : Ratio
  volume_pos : volume > 0

/-- The ratio of alcohol to water in a mixture of two jars -/
noncomputable def mixRatio (jar1 jar2 : Jar) : Ratio where
  numerator := (jar1.ratio.numerator / (jar1.ratio.numerator + jar1.ratio.denominator)) * jar1.volume +
                (jar2.ratio.numerator / (jar2.ratio.numerator + jar2.ratio.denominator)) * jar2.volume
  denominator := (jar1.ratio.denominator / (jar1.ratio.numerator + jar1.ratio.denominator)) * jar1.volume +
                 (jar2.ratio.denominator / (jar2.ratio.numerator + jar2.ratio.denominator)) * jar2.volume
  denominator_pos := by
    sorry -- Proof that the denominator is positive

theorem mix_ratio_depends_on_volumes (p q V₁ V₂ : ℝ) 
  (hp : p > 0) (hq : q > 0) (hV₁ : V₁ > 0) (hV₂ : V₂ > 0) :
  ∃ (f : ℝ → ℝ → Ratio), 
    mixRatio { volume := V₁, 
               ratio := { numerator := 2*p, denominator := 3, denominator_pos := by norm_num },
               volume_pos := hV₁ }
             { volume := V₂, 
               ratio := { numerator := 3*q, denominator := 2, denominator_pos := by norm_num },
               volume_pos := hV₂ } = f V₁ V₂ := by
  sorry -- Proof of the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mix_ratio_depends_on_volumes_l1019_101905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l1019_101937

def a : Fin 3 → ℝ := ![3, 2, 1]
def b : Fin 3 → ℝ := ![-1, 4, 2]

def diagonal1 : Fin 3 → ℝ := ![a 0 + b 0, a 1 + b 1, a 2 + b 2]
def diagonal2 : Fin 3 → ℝ := ![b 0 - a 0, b 1 - a 1, b 2 - a 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

noncomputable def cos_angle_between_diagonals : ℝ := 
  dot_product diagonal1 diagonal2 / (magnitude diagonal1 * magnitude diagonal2)

def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ := 
  ![(v 1 * w 2) - (v 2 * w 1), (v 2 * w 0) - (v 0 * w 2), (v 0 * w 1) - (v 1 * w 0)]

noncomputable def parallelogram_area : ℝ := magnitude (cross_product a b)

theorem parallelogram_properties : 
  cos_angle_between_diagonals = 1 / Real.sqrt 21 ∧ parallelogram_area = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l1019_101937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_GHCD_specific_l1019_101928

/-- Representation of a trapezoid ABCD with midpoints G and H -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  altitude : ℝ
  parallel : AB ≠ CD

/-- Calculate the area of quadrilateral GHCD in a trapezoid -/
noncomputable def area_GHCD (t : Trapezoid) : ℝ :=
  ((t.CD + (t.AB + t.CD) / 2) * t.altitude / 2) / 2

/-- Theorem: Area of GHCD in the given trapezoid is 187.5 square units -/
theorem area_GHCD_specific : 
  let t : Trapezoid := { AB := 10, CD := 30, altitude := 15, parallel := by norm_num }
  area_GHCD t = 187.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_GHCD_specific_l1019_101928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1019_101952

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h₁ : d ≠ 0
  h₂ : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ := 
  n * (seq.a 1 + seq.a n) / 2

/-- Theorem: For an arithmetic sequence where a₁, a₃, a₄ form a geometric sequence,
    (S₃ - S₂) / (S₅ - S₃) = 2 -/
theorem arithmetic_geometric_ratio (seq : ArithmeticSequence) 
    (h : (seq.a 3)^2 = seq.a 1 * seq.a 4) : 
    (S seq 3 - S seq 2) / (S seq 5 - S seq 3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1019_101952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1019_101965

noncomputable def f (x : ℝ) := 4 * Real.cos x * Real.sin (x + Real.pi / 6) - 4 * Real.sqrt 3 * Real.sin x * Real.cos x + 1

def domain : Set ℝ := { x | -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 }

noncomputable def axis_of_symmetry (k : ℤ) : ℝ := k * Real.pi / 2 - Real.pi / 6

def increasing_intervals : Set (Set ℝ) := { { x | -Real.pi / 2 ≤ x ∧ x ≤ -Real.pi / 6 }, { x | Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 2 } }

theorem f_properties :
  (∀ x, x ∈ domain → ∃ k : ℤ, f (x + axis_of_symmetry k) = f (axis_of_symmetry k - x)) ∧
  (∀ I ∈ increasing_intervals, ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1019_101965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_addition_puzzle_l1019_101985

/-- Represents a digit in base 6 -/
def Digit6 : Type := { n : ℕ // n < 6 }

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List Digit6 :=
  sorry

/-- Adds two numbers in base 6 -/
def addBase6 (a b : List Digit6) : List Digit6 :=
  sorry

theorem base6_addition_puzzle :
  ∀ (S H E : Digit6),
    S ≠ H → S ≠ E → H ≠ E →
    addBase6 [S, H, E] [H, E] = [H, E, H] →
    toBase6 (S.val + H.val + E.val) = toBase6 12 :=
by
  sorry

#check base6_addition_puzzle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_addition_puzzle_l1019_101985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_pi_half_properties_l1019_101910

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 2)

theorem sin_2x_plus_pi_half_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), ∀ y ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_pi_half_properties_l1019_101910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_mpg_l1019_101942

-- Define the problem parameters
def initial_odometer : ℕ := 56200
def final_odometer : ℕ := 57060
def total_gasoline : ℕ := 32

-- Define the function to calculate average miles per gallon
def average_mpg (initial : ℕ) (final : ℕ) (gasoline : ℕ) : ℚ :=
  (final - initial : ℚ) / gasoline

-- Define the function to round to the nearest tenth
def round_to_tenth (x : ℚ) : ℚ :=
  ⌊(x * 10 + 1/2)⌋ / 10

-- Theorem statement
theorem car_average_mpg :
  round_to_tenth (average_mpg initial_odometer final_odometer total_gasoline) = 269 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_mpg_l1019_101942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_power_l1019_101906

theorem last_digit_power (a n : ℕ) : 
  (a^n) % 10 = ((a % 10)^(n % 4)) % 10 := by
  sorry

#eval Nat.pow (322 % 10) (111569 % 4) % 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_power_l1019_101906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l1019_101953

noncomputable section

open Real

theorem triangle_cosine_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  b^2 = a * c →
  sin A = (sin (B - A) + sin C) / 2 →
  sin A = a / c →
  cos B = (sqrt 5 - 1) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l1019_101953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_groups_for_non_multiple_partition_l1019_101964

theorem min_groups_for_non_multiple_partition : ∃ (n : ℕ), n = 7 ∧ 
  (∀ (partition : Finset (Finset ℕ)),
    (∀ (group : Finset ℕ), group ∈ partition → 
      (∀ x y, x ∈ group → y ∈ group → x ≠ y → (x ∣ y ∨ y ∣ x) → False)) →
    (∀ i, i ∈ Finset.range 100 → ∃ (group : Finset ℕ), group ∈ partition ∧ i + 1 ∈ group) →
    partition.card ≥ n) ∧
  (∃ (partition : Finset (Finset ℕ)),
    (∀ (group : Finset ℕ), group ∈ partition → 
      (∀ x y, x ∈ group → y ∈ group → x ≠ y → (x ∣ y ∨ y ∣ x) → False)) ∧
    (∀ i, i ∈ Finset.range 100 → ∃ (group : Finset ℕ), group ∈ partition ∧ i + 1 ∈ group) ∧
    partition.card = n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_groups_for_non_multiple_partition_l1019_101964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l1019_101931

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x^3 + x^2 + 2 * Real.sqrt x

-- State the theorem
theorem evaluate_g : 2 * g 3 + g 9 = 888 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l1019_101931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_intersecting_line_l1019_101959

/-- Given two points A and B and a line l that intersects segment AB, 
    prove the range of possible slopes for line l. -/
theorem slope_range_for_intersecting_line 
  (A B : ℝ × ℝ) 
  (m : ℝ) 
  (h_A : A = (2, -3)) 
  (h_B : B = (-3, -2)) 
  (h_intersect : (m * A.1 + A.2 - m - 1) * (m * B.1 + B.2 - m - 1) ≤ 0) : 
  let k := -m
  k ≥ 3/4 ∨ k ≤ -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_intersecting_line_l1019_101959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l1019_101951

/-- The function representing the curve -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.log (x + 1)

/-- The derivative of the function -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - 1 / (x + 1)

theorem tangent_parallel_to_x_axis (a : ℝ) :
  f' a 1 = 0 → a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l1019_101951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l1019_101990

noncomputable def points : List (ℝ × ℝ) := [(5, 15), (10, 26), (15, 40), (22, 50), (25, 60), (30, 75)]

noncomputable def isAboveLine (point : ℝ × ℝ) : Bool :=
  point.2 > 2.5 * point.1 + 5

noncomputable def sumXCoordinates (points : List (ℝ × ℝ)) : ℝ :=
  (points.filter isAboveLine).map (·.1) |>.sum

theorem sum_x_coordinates_above_line : sumXCoordinates points = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l1019_101990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1019_101917

theorem problem_statement (x : ℝ) : (25 : ℝ)^(x+1) = 375 + (25 : ℝ)^x → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1019_101917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tan_one_max_value_f_l1019_101966

noncomputable def a (x : ℝ) : ℝ × ℝ := (1/2 * Real.cos x, Real.sqrt 3/2 * Real.sin x)
noncomputable def b : ℝ × ℝ := (Real.sqrt 3, -1)

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

noncomputable def f (x : ℝ) : ℝ := (a x + b).1 * b.1 + (a x + b).2 * b.2

theorem perpendicular_tan_one :
  ∀ x : ℝ, perpendicular (a x) b → Real.tan x = 1 := by sorry

theorem max_value_f :
  ∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧
    f x = 4 + Real.sqrt 6 / 2 ∧
    ∀ y : ℝ, y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tan_one_max_value_f_l1019_101966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_triangle_side_length_l1019_101902

/-- Represents a sequence of equilateral triangles where each subsequent triangle
    is formed by joining the midpoints of the previous triangle's sides. -/
noncomputable def TriangleSequence (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => TriangleSequence a n / 2

/-- The sum of perimeters of all triangles in the sequence. -/
noncomputable def SumOfPerimeters (a : ℝ) : ℝ :=
  3 * a * 2

theorem third_triangle_side_length
  (h1 : TriangleSequence 80 0 = 80)
  (h2 : SumOfPerimeters 80 = 480) :
  TriangleSequence 80 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_triangle_side_length_l1019_101902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_trip_time_l1019_101963

/-- Represents a round trip with different speeds for outbound and return journeys -/
structure RoundTrip where
  outbound_speed : ℝ
  return_speed : ℝ
  total_time : ℝ

/-- Calculates the time taken for the outbound journey in minutes -/
noncomputable def outbound_time (trip : RoundTrip) : ℝ :=
  let distance := (trip.outbound_speed * trip.return_speed * trip.total_time) / (trip.outbound_speed + trip.return_speed)
  (distance / trip.outbound_speed) * 60

theorem cole_trip_time :
  let trip := RoundTrip.mk 60 90 2
  outbound_time trip = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_trip_time_l1019_101963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_min_OM_length_min_OM_length_equality_l1019_101934

-- Define the circle C
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the point Q
noncomputable def Q : ℝ × ℝ := (-2, Real.sqrt 21)

-- Define the tangent line length
def tangent_length : ℝ := 4

-- Theorem for part 1
theorem circle_radius (r : ℝ) (h1 : r > 0) :
  ∃ (D : ℝ × ℝ), D ∈ Circle r ∧ ‖Q - D‖ = tangent_length → r = 3 := by sorry

-- Define a point in the first quadrant
def first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

-- Define the vector OM
def OM (A B : ℝ × ℝ) : ℝ × ℝ := (A.1 + B.1, A.2 + B.2)

-- Theorem for part 2
theorem min_OM_length (r : ℝ) (h1 : r > 0) :
  ∀ (P A B : ℝ × ℝ), 
    P ∈ Circle r → 
    first_quadrant P → 
    A.2 = 0 → 
    B.1 = 0 → 
    (∃ (l : Set (ℝ × ℝ)), A ∈ l ∧ B ∈ l ∧ P ∈ l ∧ (∀ x ∈ l, x ∉ Circle r ∨ x = P)) →
    ‖OM A B‖ ≥ 6 := by sorry

-- Theorem for the equality condition
theorem min_OM_length_equality (r : ℝ) (h1 : r > 0) :
  ∃ (P A B : ℝ × ℝ), 
    P ∈ Circle r ∧
    first_quadrant P ∧
    A.2 = 0 ∧
    B.1 = 0 ∧
    (∃ (l : Set (ℝ × ℝ)), A ∈ l ∧ B ∈ l ∧ P ∈ l ∧ (∀ x ∈ l, x ∉ Circle r ∨ x = P)) ∧
    ‖OM A B‖ = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_min_OM_length_min_OM_length_equality_l1019_101934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_reach_train_approx_50_69_l1019_101901

/-- Calculates the time taken for a man to reach a train while avoiding obstacles -/
noncomputable def time_to_reach_train (train_a_length : ℝ) (train_a_speed : ℝ) (train_b_length : ℝ) (train_b_speed : ℝ) (man_speed : ℝ) (obstacle_time : ℝ) (num_obstacles : ℕ) : ℝ :=
  let train_a_speed_ms := train_a_speed * 1000 / 3600
  let train_b_speed_ms := train_b_speed * 1000 / 3600
  let man_speed_ms := man_speed * 1000 / 3600
  let relative_speed_a := train_a_speed_ms - man_speed_ms
  let relative_speed_b := train_b_speed_ms + man_speed_ms
  let time_a := train_a_length / relative_speed_a
  let time_b := train_b_length / relative_speed_b
  let total_obstacle_time := obstacle_time * (num_obstacles : ℝ)
  time_b + total_obstacle_time

/-- The main theorem stating that the time taken for the man to reach train B is approximately 50.69 seconds -/
theorem time_to_reach_train_approx_50_69 :
  ∃ ε > 0, |time_to_reach_train 420 30 520 40 6 5 2 - 50.69| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_reach_train_approx_50_69_l1019_101901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_reduced_fraction_components_l1019_101927

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem product_of_reduced_fraction_components (a b c : ℕ) :
  a = 0 ∧ b = 1 ∧ c = 8 →
  let f := repeating_decimal_to_fraction a b c
  let n := f.num.natAbs
  let d := f.den
  n * d = 222 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_reduced_fraction_components_l1019_101927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_inside_circle_O_slope_of_chord_AB_l1019_101960

-- Define the circle O
def circle_O (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8

-- Define point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 7

-- Theorem for point P's position relative to circle O
theorem point_P_inside_circle_O : 
  (point_P.1 + 1)^2 + point_P.2^2 < 8 := by sorry

-- Theorem for the slope of chord AB
theorem slope_of_chord_AB :
  ∃ (k : ℝ), (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) ∧
  ∃ (A B : ℝ × ℝ), 
    circle_O A.1 A.2 ∧ 
    circle_O B.1 B.2 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = chord_length^2 ∧
    k = (B.2 - A.2) / (B.1 - A.1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_inside_circle_O_slope_of_chord_AB_l1019_101960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_trig_identity_l1019_101979

theorem complex_trig_identity (θ : ℝ) (h : Real.sin (π + θ) = 1/2) : 
  (Real.cos (3*π + θ) / Real.cos (Real.cos (π - θ) - 1)) + 
  (Real.cos (θ - 2*π) / (Real.sin (θ - 7*π/2) * Real.cos (π - θ) - Real.sin (3*π/2 + θ))) = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_trig_identity_l1019_101979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missed_angle_measure_l1019_101995

/-- The sum of interior angles of a convex polygon with n sides is (n-2) * 180° --/
def sum_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- Thomas' calculation of the sum of interior angles --/
def thomas_sum : ℕ := 3239

/-- The correct sum of interior angles --/
def correct_sum : ℕ := 3240

theorem missed_angle_measure :
  ∃ (n : ℕ), n ≥ 3 ∧ 
  thomas_sum = sum_angles (n - 1) ∧
  correct_sum = sum_angles n ∧
  correct_sum - thomas_sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missed_angle_measure_l1019_101995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isoscelesBaseLength_correct_l1019_101921

/-- IsoscelesTriangle structure to represent isosceles triangles -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  baseAngle : ℝ
  vertexAngle : ℝ

/-- Given an isosceles triangle with base 1 and leg length b, 
    this function returns the base length of another isosceles triangle 
    with legs of length 1 and vertex angle equal to the base angle of the first triangle -/
noncomputable def isoscelesBaseLength (b : ℝ) : ℝ :=
  Real.sqrt (2 - 1 / b)

/-- Theorem stating the correctness of the isoscelesBaseLength function -/
theorem isoscelesBaseLength_correct (b : ℝ) (h : b > 0) :
  let firstTriangle : IsoscelesTriangle := ⟨1, b, 0, 0⟩
  let secondTriangle : IsoscelesTriangle := ⟨isoscelesBaseLength b, 1, 0, 0⟩
  secondTriangle.vertexAngle = firstTriangle.baseAngle ∧
  secondTriangle.base = Real.sqrt (2 - 1 / b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isoscelesBaseLength_correct_l1019_101921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_area_is_half_l1019_101992

/-- A cube with side length 2 -/
def Cube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 2}

/-- The vertex P of the cube -/
def P : Fin 3 → ℝ := λ i ↦ 0

/-- The midpoint Q of a face diagonal from P -/
def Q : Fin 3 → ℝ := λ i ↦ if i.val < 2 then 1 else 0

/-- The quarter point R on an edge opposite of P -/
def R : Fin 3 → ℝ := λ i ↦ if i.val = 2 then 1 else 0

/-- The quarter point S on an edge opposite of P -/
def S : Fin 3 → ℝ := λ i ↦ if i.val = 2 then 1.5 else 0

/-- The plane passing through P, Q, R, and S -/
def Plane : Set (Fin 3 → ℝ) :=
  {p | p 0 = p 1}

/-- The area of the quadrilateral PQRS -/
noncomputable def QuadArea : ℝ := 1/2

theorem quad_area_is_half :
  P ∈ Cube ∧ Q ∈ Cube ∧ R ∈ Cube ∧ S ∈ Cube ∧
  P ∈ Plane ∧ Q ∈ Plane ∧ R ∈ Plane ∧ S ∈ Plane →
  QuadArea = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_area_is_half_l1019_101992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l1019_101950

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
noncomputable def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  { vertices := λ i => ((o.vertices i + o.vertices ((i + 1) % 8)) / 2) }

/-- The area of a regular octagon -/
noncomputable def area (o : RegularOctagon) : ℝ := sorry

/-- The theorem stating that the area of the midpoint octagon is half of the original octagon -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (1 : ℝ) / 2 * area o := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l1019_101950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1019_101943

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hbe : b > Real.exp 1) :
  a^b < b^a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1019_101943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_not_always_equal_major_premise_wrong_l1019_101972

-- Define a rhombus
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j
  diagonals_bisect : True  -- Simplified representation
  diagonals_perpendicular : True  -- Simplified representation

-- Define a square as a special case of rhombus
structure Square extends Rhombus where
  right_angles : True  -- Simplified representation

-- Define a proposition to represent that d1 and d2 are the lengths of the diagonals of r
def AreDiagonals (r : Rhombus) (d1 d2 : ℝ) : Prop := 
  True  -- Simplified representation, replace with actual condition when needed

-- Theorem statement
theorem rhombus_diagonals_not_always_equal :
  ∃ (r : Rhombus), ∃ (d1 d2 : ℝ), d1 ≠ d2 ∧ AreDiagonals r d1 d2 :=
sorry

-- The major premise of the syllogism is wrong
theorem major_premise_wrong :
  ¬(∀ (r : Rhombus), ∃ (d : ℝ), ∀ (d1 d2 : ℝ), AreDiagonals r d1 d2 → d1 = d2 ∧ d1 = d) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_not_always_equal_major_premise_wrong_l1019_101972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_period_sine_with_shift_increasing_l1019_101949

-- Define the cosine squared function
noncomputable def cos_squared (x : ℝ) : ℝ := Real.cos x ^ 2 - 1/2

-- Define the sine function with phase shift
noncomputable def sine_with_shift (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi/3)

-- Theorem for the period of cos_squared
theorem cos_squared_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, cos_squared (x + T) = cos_squared x) ∧
  (∀ S, S > 0 ∧ (∀ x, cos_squared (x + S) = cos_squared x) → T ≤ S) ∧
  T = Real.pi :=
sorry

-- Theorem for the increasing property of sine_with_shift
theorem sine_with_shift_increasing :
  ∀ x y, -Real.pi/12 < x ∧ x < y ∧ y < 5*Real.pi/12 → sine_with_shift x < sine_with_shift y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_period_sine_with_shift_increasing_l1019_101949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_example_l1019_101904

/-- Converts rectangular coordinates to cylindrical coordinates -/
noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, θ, z)

theorem rectangular_to_cylindrical_example :
  let (r, θ, z) := rectangular_to_cylindrical (-3) 4 5
  r = 5 ∧ θ = Real.arctan (-4/3) + Real.pi ∧ z = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_example_l1019_101904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1019_101938

-- Define a straight line in a Cartesian coordinate system
structure Line where
  slope : ℝ
  inclination_angle : ℝ

-- Define the property of being perpendicular to an axis
def perpendicular_to_x_axis (l : Line) : Prop := l.inclination_angle = Real.pi / 2
def perpendicular_to_y_axis (l : Line) : Prop := l.inclination_angle = 0

-- Theorem statement
theorem line_properties :
  ∀ (l : Line),
    (l.inclination_angle ≠ Real.pi / 2 → l.slope = Real.tan l.inclination_angle) ∧
    (perpendicular_to_x_axis l → l.inclination_angle = Real.pi / 2) ∧
    (perpendicular_to_y_axis l → l.inclination_angle = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1019_101938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_set_eq_interval_l1019_101945

-- Define the function f as noncomputable
noncomputable def f (x : Real) : Real := Real.sin x * (Real.cos x - Real.sqrt 3 * Real.sin x)

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f,
  -Real.sqrt 3 ≤ y ∧ y ≤ 1 - Real.sqrt 3 / 2 ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = y) := by
  sorry

-- Define the set representing the range
def range_set : Set Real := { y | ∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = y }

-- State that the range_set is equal to the interval
theorem range_set_eq_interval :
  range_set = Set.Icc (-Real.sqrt 3) (1 - Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_set_eq_interval_l1019_101945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natasha_climb_speed_l1019_101983

/-- Represents a round trip journey with given parameters -/
structure RoundTrip where
  total_time : ℝ
  time_up : ℝ
  time_down : ℝ
  avg_speed_total : ℝ

/-- Calculates the average speed for the upward journey -/
noncomputable def avg_speed_up (trip : RoundTrip) : ℝ :=
  (trip.avg_speed_total * trip.total_time) / (2 * trip.time_up)

/-- Theorem stating the average speed for the upward journey is 1.5 km/h given the specified conditions -/
theorem natasha_climb_speed (trip : RoundTrip) 
    (h1 : trip.total_time = 9)
    (h2 : trip.time_up = 6)
    (h3 : trip.time_down = 3)
    (h4 : trip.avg_speed_total = 2) :
    avg_speed_up trip = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_natasha_climb_speed_l1019_101983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_relation_l1019_101907

theorem log_relation (a b : ℝ) (h1 : a = Real.log 225 / Real.log 8) (h2 : b = Real.log 15 / Real.log 2) : 
  a = 2 * b / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_relation_l1019_101907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_element_in_set_l1019_101908

theorem element_in_set : ∃ M : Set ℕ, 1 ∈ M := by
  use {1, 2, 3}
  simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_element_in_set_l1019_101908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_area_l1019_101941

-- Define the hyperbola C
def hyperbola (b : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2/b^2 = 1 ∧ b > 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

-- Define the asymptotes of the hyperbola
def asymptotes (b : ℝ) (x y : ℝ) : Prop :=
  y = b*x ∨ y = -b*x

-- Define the intersection points
def intersection_points (b : ℝ) (x y : ℝ) : Prop :=
  asymptotes b x y ∧ circle_O x y

-- Define the area of the rectangle formed by intersection points
def rectangle_area (b : ℝ) : Prop :=
  ∃ x y : ℝ, intersection_points b x y ∧
  (4 * x * y = b)

-- The main theorem
theorem hyperbola_intersection_area (b : ℝ) :
  (∀ x y : ℝ, hyperbola b x y) ∧ rectangle_area b → b = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_area_l1019_101941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_problem_solution_l1019_101993

/-- Represents a construction team -/
structure Team where
  daily_area : ℝ → ℝ
  daily_cost : ℝ

/-- The sports center construction problem -/
structure ConstructionProblem where
  team_a : Team
  team_b : Team
  total_days : ℕ
  min_area : ℝ

/-- The specific construction problem instance -/
def problem : ConstructionProblem :=
  { team_a := { daily_area := λ x => x + 300, daily_cost := 3600 }
  , team_b := { daily_area := λ x => x, daily_cost := 2200 }
  , total_days := 22
  , min_area := 15000 }

/-- The condition that Team A and Team B take the same time for their respective areas -/
def equal_time_condition (x : ℝ) : Prop :=
  1800 / (problem.team_a.daily_area x) = 1200 / (problem.team_b.daily_area x)

/-- The theorem stating the solution to the construction problem -/
theorem construction_problem_solution :
  ∃ (x : ℝ),
    equal_time_condition x ∧
    x = 600 ∧
    (∃ (m : ℕ),
      m ≤ problem.total_days ∧
      problem.team_a.daily_area x * m + problem.team_b.daily_area x * (problem.total_days - m) ≥ problem.min_area ∧
      problem.team_a.daily_cost * m + problem.team_b.daily_cost * (problem.total_days - m) = 56800) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_problem_solution_l1019_101993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_two_zeros_condition_l1019_101919

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - 1 + (a * x^2) / Real.exp x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - 1) * f a (x - 1) + x * (1 - Real.log x)

-- Statement for part 1
theorem tangent_line_at_one (a : ℝ) :
  a = 1 → ∃ m b : ℝ, ∀ x : ℝ, m * x + b = (1 + 1 / Real.exp 1) * x - 1 ∧
    HasDerivAt (f 1) m 1 ∧ f 1 1 = m * 1 + b := by sorry

-- Statement for part 2
theorem two_zeros_condition (a : ℝ) :
  (∃ x y : ℝ, 1 ≤ x ∧ x < y ∧ g a x = 0 ∧ g a y = 0 ∧ ∀ z, x < z → z < y → g a z ≠ 0) ↔
  a < 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_two_zeros_condition_l1019_101919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1019_101967

theorem trig_identity (α : ℝ) : 
  Real.tan (4 * α) + 1 / Real.cos (4 * α) = (Real.cos (2 * α) + Real.sin (2 * α)) / (Real.cos (2 * α) - Real.sin (2 * α)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1019_101967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_intersection_points_l1019_101991

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The first graph function -/
def graph1 (x y : ℝ) : Prop :=
  (x - floor x)^2 + y^2 = x - floor x + 1

/-- The second graph function -/
def graph2 (x y : ℝ) : Prop :=
  y = (1/3) * x + 1

/-- The set of intersection points -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | graph1 p.1 p.2 ∧ graph2 p.1 p.2}

/-- The theorem stating the number of intersection points -/
theorem num_intersection_points :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 21 ∧ ∀ p, p ∈ s ↔ p ∈ intersection_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_intersection_points_l1019_101991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l1019_101944

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := Real.exp x

-- Define the tangent line
noncomputable def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := (Real.exp x₀) * (x - x₀) + Real.exp x₀

-- State the theorem
theorem tangent_through_origin :
  ∃ x₀ : ℝ, (tangent_line x₀ 0 = 0) ∧ (∀ x : ℝ, tangent_line x₀ x = Real.exp x₀ * x) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l1019_101944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l1019_101900

/-- Given two functions f and g, and constants A and B, prove that A = B/2 -/
theorem function_composition_equality (A B : ℝ) (h_B : B ≠ 0) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = A * x - 2 * B^2)
  (h_g : ∀ x, g x = B * x^2)
  (h_comp : f (g 2) = 0) :
  A = B / 2 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l1019_101900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l1019_101981

/-- The number of revolutions per minute for a wheel -/
noncomputable def wheelRPM (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let speedCmPerMin := speed * 100000 / 60
  speedCmPerMin / circumference

/-- Theorem stating that a wheel with radius 35 cm on a bus moving at 66 km/h
    has approximately 5000.23 revolutions per minute -/
theorem bus_wheel_rpm :
  ∃ (rpm : ℝ), abs (rpm - wheelRPM 35 66) < 0.01 ∧ abs (rpm - 5000.23) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l1019_101981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1019_101933

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (15*x^4 + 6*x^3 + 5*x^2 + 4*x + 2) / (5*x^4 + 3*x^3 + 9*x^2 + 2*x + 1)

/-- The horizontal asymptote of f(x) -/
def horizontal_asymptote : ℝ := 3

/-- Theorem stating that the horizontal asymptote of f(x) is 3 -/
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - horizontal_asymptote| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1019_101933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1019_101947

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (t p : ℝ)
  (k : ℕ)
  (h_pos : ∀ n, a n > 0)
  (h_first : a 1 = t)
  (h_p_pos : p > 0)
  (h_k_pos : k ≥ 1)
  (h_sum : ∀ n, (Finset.range (k + 1)).sum (λ i ↦ a (n + i)) = 6 * p ^ n)
  (h_geometric : geometric_sequence a) :
  (∃ (q : ℝ), (∀ n, a (n + 1) = q * a n) ∧ q = p) ∧
  (p ≠ 1 → t = 6 * p * (1 - p) / (1 - p ^ (k + 1))) ∧
  (p = 1 → t = 6 / (k + 1 : ℝ)) ∧
  (k = 1 ∧ t = 1 →
    ∃ C : ℝ, ∀ n : ℕ,
      (1 + p) / p * (Finset.range n).sum (λ i ↦ a (i + 1) / p ^ i) - a n / p ^ n - 6 * n = C ∧
      C = -5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1019_101947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_theorem_l1019_101920

open Real

noncomputable section

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the vector addition function
def vectorAdd (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- State the theorem
theorem max_distance_theorem :
  ∃ (max_value : ℝ), max_value = 2 * Real.sqrt 2 + 1 ∧
  ∀ (P : ℝ × ℝ), distance A P = 1 →
  distance O (vectorAdd P B) ≤ max_value := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_theorem_l1019_101920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_real_coeffs_binomial_expansion_l1019_101911

theorem sum_real_coeffs_binomial_expansion (i : ℂ) (h : i * i = -1) :
  let T : ℂ := (λ x : ℂ ↦ (1 + i*x)^2011 + (1 - i*x)^2011) 1
  T = 2^1005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_real_coeffs_binomial_expansion_l1019_101911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expressions_l1019_101909

noncomputable def α : ℝ := Real.arctan 2

theorem trigonometric_expressions :
  (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α + Real.sin α) = 8/5 ∧
  (Real.cos α ^ 2 + Real.sin α * Real.cos α) / (2 * Real.sin α * Real.cos α + Real.sin α ^ 2) = 3/8 ∧
  Real.sin α ^ 2 - Real.sin α * Real.cos α + 2 = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expressions_l1019_101909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_at_one_l1019_101955

noncomputable section

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line
def line (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Define the tangent slope at a point
def tangent_slope (x : ℝ) : ℝ := (1/2) * x

-- Theorem statement
theorem tangent_perpendicular_at_one :
  ∀ (m n : ℝ),
  curve m n →
  (tangent_slope m * (-2) = -1) →
  m = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_at_one_l1019_101955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_l1019_101999

/-- The slant height of a cone with given radius and curved surface area -/
noncomputable def slant_height (radius : ℝ) (curved_surface_area : ℝ) : ℝ :=
  curved_surface_area / (Real.pi * radius)

/-- Theorem: The slant height of a cone with radius 21 m and curved surface area 989.6016858807849 m² is 15 m -/
theorem cone_slant_height :
  let r : ℝ := 21
  let csa : ℝ := 989.6016858807849
  slant_height r csa = 15 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_l1019_101999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ritas_remaining_money_l1019_101997

/-- Calculates the remaining money after Rita's shopping spree --/
noncomputable def remaining_money (initial_amount : ℚ) 
  (dress_price dress_discount : ℚ) (num_dresses : ℕ)
  (pants_price : ℚ) (num_pants : ℕ)
  (jacket_price jacket_discount : ℚ) (num_jackets num_discounted_jackets : ℕ)
  (skirt_price skirt_discount : ℚ) (num_skirts : ℕ)
  (tshirt_price : ℚ) (num_tshirts : ℕ)
  (transportation_cost : ℚ) : ℚ :=
  let dress_total := num_dresses * dress_price * (1 - dress_discount)
  let pants_total := (num_pants - 1) * pants_price
  let jacket_total := (num_jackets - num_discounted_jackets) * jacket_price + 
                      num_discounted_jackets * jacket_price * (1 - jacket_discount)
  let skirt_total := if num_skirts * skirt_price ≥ 30 
                     then num_skirts * skirt_price * (1 - skirt_discount)
                     else num_skirts * skirt_price
  let tshirt_total := (num_tshirts - 1) * tshirt_price
  let total_spent := dress_total + pants_total + jacket_total + skirt_total + tshirt_total + transportation_cost
  initial_amount - total_spent

/-- Theorem stating that Rita's remaining money is $119.2 --/
theorem ritas_remaining_money : 
  remaining_money 400 20 (1/10) 5 15 3 30 (15/100) 4 2 18 (1/5) 2 8 3 5 = 119.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ritas_remaining_money_l1019_101997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_fifth_terms_of_arithmetic_sequences_l1019_101978

/-- Sum of the first n terms of a sequence -/
def sum (s : ℕ → ℚ) (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => s (i + 1))

/-- Given two arithmetic sequences, this theorem proves that if the ratio of the sums
    of their first n terms is (9n+2)/(n+7), then the ratio of their 5th terms is 83/16. -/
theorem ratio_of_fifth_terms_of_arithmetic_sequences
  (a b : ℕ → ℚ)  -- Two sequences of rational numbers
  (h_arithmetic_a : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)  -- a is arithmetic
  (h_arithmetic_b : ∀ n, b (n + 2) - b (n + 1) = b (n + 1) - b n)  -- b is arithmetic
  (h_sum_ratio : ∀ n, sum a n / sum b n = (9 * n + 2) / (n + 7))  -- Ratio of sums
  : a 5 / b 5 = 83 / 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_fifth_terms_of_arithmetic_sequences_l1019_101978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1019_101957

theorem triangle_theorem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  Real.sin B - Real.sin C = Real.sin (A - C) →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  A = π/3 ∧ a + b + c = 2 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1019_101957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l1019_101974

def lambda : Set ℝ := {x : ℝ | ∃ (a b : ℤ), x = a + b * Real.sqrt 3}

theorem a_value (x : ℝ) (a : ℤ) (h1 : x ∈ lambda) (h2 : x = 7 + a * Real.sqrt 3) (h3 : x⁻¹ ∈ lambda) : a = 4 ∨ a = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l1019_101974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1019_101932

/-- The compound interest formula -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- The initial investment rounded to the nearest dollar -/
def initial_investment : ℕ := 45639

/-- The theorem stating that the initial investment grows to the desired amount -/
theorem investment_growth :
  99999 < compound_interest (initial_investment : ℝ) 0.08 2 10 ∧
  compound_interest (initial_investment : ℝ) 0.08 2 10 < 100001 := by
  sorry

#eval initial_investment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1019_101932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1019_101956

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with specific conditions, S₈ = -85 -/
theorem geometric_sequence_sum (a₁ q : ℝ) :
  let S := geometricSum a₁ q
  S 4 = -5 ∧ S 6 = 21 * S 2 → S 8 = -85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1019_101956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_is_ratio_difference_arithmetic_not_necessarily_ratio_difference_special_sequence_not_ratio_difference_fibonacci_like_not_ratio_difference_product_arithmetic_geometric_not_necessarily_ratio_difference_l1019_101922

/-- Definition of a ratio-difference sequence -/
def is_ratio_difference_sequence (a : ℕ → ℝ) : Prop :=
  ∃ t : ℝ, ∀ n : ℕ, a (n + 2) / a (n + 1) - a (n + 1) / a n = t

/-- Definition of a geometric sequence -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem 1: Geometric sequences are always ratio-difference sequences -/
theorem geometric_is_ratio_difference (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  is_ratio_difference_sequence a := by
  sorry

/-- Theorem 2: Arithmetic sequences are not necessarily ratio-difference sequences -/
theorem arithmetic_not_necessarily_ratio_difference :
  ¬∀ a : ℕ → ℝ, is_arithmetic_sequence a → is_ratio_difference_sequence a := by
  sorry

/-- Definition of the sequence a_n = 2^(n-1) / n^2 -/
noncomputable def special_sequence (n : ℕ) : ℝ := 2^(n-1) / (n^2 : ℝ)

/-- Theorem 3: The special sequence is not a ratio-difference sequence -/
theorem special_sequence_not_ratio_difference :
  ¬is_ratio_difference_sequence special_sequence := by
  sorry

/-- Definition of the Fibonacci-like sequence -/
def fibonacci_like : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci_like (n + 1) + fibonacci_like n

/-- Theorem 4: The Fibonacci-like sequence is not a ratio-difference sequence -/
theorem fibonacci_like_not_ratio_difference :
  ¬is_ratio_difference_sequence (λ n => (fibonacci_like n : ℝ)) := by
  sorry

/-- Theorem 5: The product of an arithmetic sequence and a geometric sequence
    is not necessarily a ratio-difference sequence -/
theorem product_arithmetic_geometric_not_necessarily_ratio_difference :
  ¬∀ (a b : ℕ → ℝ), is_arithmetic_sequence a → is_geometric_sequence b →
    is_ratio_difference_sequence (λ n => a n * b n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_is_ratio_difference_arithmetic_not_necessarily_ratio_difference_special_sequence_not_ratio_difference_fibonacci_like_not_ratio_difference_product_arithmetic_geometric_not_necessarily_ratio_difference_l1019_101922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l1019_101973

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus with slant angle π/4
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem parabola_intersection_distance :
  ∀ A B : ℝ × ℝ, intersection_points A B → distance A B = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l1019_101973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_and_diff_l1019_101969

theorem abs_sum_and_diff (a b : ℝ) (ha : |a| = 5) (hb : |b| = 3) :
  (∃ x ∈ ({-8, -2, 2, 8} : Set ℝ), a + b = x) ∧
  (|a + b| = a + b → a - b ∈ ({2, 8} : Set ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_and_diff_l1019_101969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_to_square_l1019_101924

/-- Given a square ABCD with side length s, point P dividing AB in a 1:3 ratio from A,
    and point Q dividing BC in a 1:3 ratio from B, the ratio of the area of triangle APQ
    to the area of square ABCD is √10/32. -/
theorem area_ratio_triangle_to_square (s : ℝ) (h : s > 0) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (s, 0)
  let C : ℝ × ℝ := (s, s)
  let D : ℝ × ℝ := (0, s)
  let P : ℝ × ℝ := (s/4, 0)
  let Q : ℝ × ℝ := (s, s/4)
  let area_triangle := abs ((P.1 - A.1) * (Q.2 - A.2) - (Q.1 - A.1) * (P.2 - A.2)) / 2
  let area_square := s^2
  area_triangle / area_square = Real.sqrt 10 / 32 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_to_square_l1019_101924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_formula_correct_l1019_101926

/-- Represents the annual pension of a retiring employee -/
noncomputable def annual_pension (a b p q : ℝ) : ℝ := (a * q^2 - b * p^2) / (2 * (b * p - a * q))

/-- Proves that the given formula for annual pension satisfies the problem conditions -/
theorem pension_formula_correct (a b p q : ℝ) (h1 : b ≠ a) :
  ∃ (k x : ℝ),
    -- The pension is proportional to the square root of years served
    k > 0 ∧ x > 0 ∧
    -- If served 'a' more years, pension increases by 'p'
    k * Real.sqrt (x + a) = k * Real.sqrt x + p ∧
    -- If served 'b' more years, pension increases by 'q'
    k * Real.sqrt (x + b) = k * Real.sqrt x + q ∧
    -- The original pension equals the formula
    k * Real.sqrt x = annual_pension a b p q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_formula_correct_l1019_101926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l1019_101946

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ := !![5, -3; 2, 1]

theorem inverse_of_A :
  ∃ (B : Matrix (Fin 2) (Fin 2) ℚ), 
    B = !![1/11, 3/11; -2/11, 5/11] ∧ 
    IsUnit (Matrix.det A) ∧
    A * B = 1 ∧ 
    B * A = 1 := by
  -- Construct the inverse matrix B
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![1/11, 3/11; -2/11, 5/11]
  
  -- Prove that B is the inverse of A
  have h1 : B = !![1/11, 3/11; -2/11, 5/11] := rfl
  have h2 : IsUnit (Matrix.det A) := by
    -- Prove that det A is non-zero
    sorry
  have h3 : A * B = 1 := by
    -- Prove that AB = I
    sorry
  have h4 : B * A = 1 := by
    -- Prove that BA = I
    sorry
  
  -- Conclude the proof
  exact ⟨B, h1, h2, h3, h4⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l1019_101946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soldiers_arrival_time_l1019_101940

/-- Represents the transportation problem of soldiers --/
structure TransportProblem where
  totalDistance : ℕ
  numSoldiers : ℕ
  carCapacity : ℕ
  carSpeed : ℕ
  walkingSpeed : ℕ

/-- Calculates the arrival time for all soldiers --/
def arrivalTime (problem : TransportProblem) : ℚ :=
  2 + 36 / 60 -- 2 hours and 36 minutes in decimal hours

/-- Theorem stating that the arrival time for the given problem is 2 hours and 36 minutes --/
theorem soldiers_arrival_time (problem : TransportProblem) 
  (h1 : problem.totalDistance = 20)
  (h2 : problem.numSoldiers = 12)
  (h3 : problem.carCapacity = 4)
  (h4 : problem.carSpeed = 20)
  (h5 : problem.walkingSpeed = 4) :
  arrivalTime problem = 2 + 36 / 60 := by
  sorry

#eval arrivalTime { totalDistance := 20, numSoldiers := 12, carCapacity := 4, carSpeed := 20, walkingSpeed := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soldiers_arrival_time_l1019_101940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_divisible_by_15_l1019_101958

def repeat_2008 (n : ℕ) : ℕ :=
  -- Function to create a number by repeating 2008 n times
  sorry -- Placeholder for the actual implementation

theorem least_divisible_by_15 : 
  (∃ n : ℕ, n > 0 ∧ 15 ∣ repeat_2008 n) ∧ 
  (∀ m : ℕ, m > 0 ∧ 15 ∣ repeat_2008 m → m ≥ 3) ∧
  (15 ∣ repeat_2008 3) :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_divisible_by_15_l1019_101958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_standard_l1019_101994

-- Define the parametric equations as noncomputable
noncomputable def x_of_t (t : ℝ) : ℝ := (2 + 3*t) / (1 + t)
noncomputable def y_of_t (t : ℝ) : ℝ := (1 - 2*t) / (1 + t)

-- State the theorem
theorem parametric_to_standard 
  (x y t : ℝ) 
  (hx : x ≠ 3) 
  (hy : y ≠ -2) 
  (hxt : x = x_of_t t) 
  (hyt : y = y_of_t t) : 
  3 * x + y - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_standard_l1019_101994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_iberic_sets_containing_33_l1019_101929

def IsIberic (X : Set ℕ) : Prop :=
  X ⊆ Finset.range 2017 ∧
  ∀ m n, m ∈ X → n ∈ X → Nat.gcd m n ∈ X

def IsOlympic (X : Set ℕ) : Prop :=
  IsIberic X ∧ ∀ Y, IsIberic Y → X ⊆ Y → X = Y

def MultiplesOf (k : ℕ) (n : ℕ) : Set ℕ :=
  {x | x ≤ n ∧ k ∣ x}

theorem olympic_iberic_sets_containing_33 :
  ∀ X : Set ℕ,
    (IsOlympic X ∧ 33 ∈ X) ↔
    (X = MultiplesOf 3 2016 ∨ X = MultiplesOf 11 2013) :=
by
  sorry

#check olympic_iberic_sets_containing_33

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_iberic_sets_containing_33_l1019_101929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_first_4_terms_l1019_101989

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r^n) / (1 - r)

/-- Theorem: The sum of the first 4 terms of a geometric sequence
    with first term 1 and common ratio 2 is 15 -/
theorem geometric_sum_first_4_terms :
  geometricSum 1 2 4 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_first_4_terms_l1019_101989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_right_angle_possibilities_l1019_101986

-- Define a right angle
def RightAngle (A O B : Point) : Prop := sorry

-- Define a plane
def Plane : Type := sorry

-- Define the projection of an angle onto a plane
def ProjectionOfAngle (A O B : Point) (α : Plane) : Real := sorry

-- Define angle classifications
def IsZeroAngle (θ : Real) : Prop := θ = 0
def IsAcuteAngle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi/2
def IsRightAngle (θ : Real) : Prop := θ = Real.pi/2
def IsObtuseAngle (θ : Real) : Prop := Real.pi/2 < θ ∧ θ < Real.pi
def Is180Angle (θ : Real) : Prop := θ = Real.pi

theorem projection_of_right_angle_possibilities 
  (A O B : Point) (α : Plane) (h : RightAngle A O B) :
  ∃ (θ : Real), 
    (θ = ProjectionOfAngle A O B α) ∧
    (IsZeroAngle θ ∨ IsAcuteAngle θ ∨ IsRightAngle θ ∨ IsObtuseAngle θ ∨ Is180Angle θ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_right_angle_possibilities_l1019_101986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_squares_in_arithmetic_progression_l1019_101998

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- A number is a perfect square if it's equal to some natural number squared. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem infinite_squares_in_arithmetic_progression
  (a : ℕ → ℕ) (d : ℕ) (h_arithmetic : is_arithmetic_progression a d)
  (h_square : ∃ n, is_perfect_square (a n)) :
  ∃ f : ℕ → ℕ, Function.Injective f ∧ ∀ n, is_perfect_square (a (f n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_squares_in_arithmetic_progression_l1019_101998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_J_3_18_12_l1019_101970

/-- Definition of function J for nonzero real numbers a, b, and c -/
noncomputable def J (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : ℝ :=
  a / b + b / c + c / a

/-- Theorem: J(3,18,12) = 17/3 -/
theorem J_3_18_12 : J 3 18 12 (by norm_num) (by norm_num) (by norm_num) = 17 / 3 := by
  -- Unfold the definition of J
  unfold J
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform numerical calculations
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_J_3_18_12_l1019_101970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_initial_percentage_l1019_101916

/-- Represents the water capacity and levels in a reservoir --/
structure Reservoir where
  capacity : ℚ
  initial_level : ℚ
  storm1_deposit : ℚ
  storm2_deposit : ℚ
  storm3_deposit : ℚ
  final_percentage : ℚ

/-- Calculates the initial percentage of fullness in the reservoir --/
def initial_percentage (r : Reservoir) : ℚ :=
  (r.initial_level / r.capacity) * 100

/-- Theorem stating the initial percentage of fullness in the reservoir --/
theorem reservoir_initial_percentage (r : Reservoir)
  (h1 : r.capacity = 200)
  (h2 : r.storm1_deposit = 15)
  (h3 : r.storm2_deposit = 30)
  (h4 : r.storm3_deposit = 75)
  (h5 : r.final_percentage = 80)
  (h6 : r.capacity * (r.final_percentage / 100) = 
        r.initial_level + r.storm1_deposit + r.storm2_deposit + r.storm3_deposit) :
  initial_percentage r = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_initial_percentage_l1019_101916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_sqrt3_plus_minus_sqrt2_l1019_101912

-- Define the two numbers
noncomputable def x : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt 3 - Real.sqrt 2

-- Define arithmetic mean
noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2

-- Define geometric mean
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

-- Theorem statement
theorem mean_of_sqrt3_plus_minus_sqrt2 :
  arithmetic_mean x y = Real.sqrt 3 ∧ 
  (geometric_mean x y = 1 ∨ geometric_mean x y = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_sqrt3_plus_minus_sqrt2_l1019_101912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_implies_m_value_l1019_101935

/-- The function f(x) defined as 2ln(x) + 8/x - m -/
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * Real.log x + 8 / x - m

/-- The theorem stating that if f(x) has a local minimum value of 2, then m = 4ln(2) -/
theorem local_minimum_implies_m_value (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ IsLocalMin (f · m) x ∧ f x m = 2) →
  m = 4 * Real.log 2 := by
  sorry

#check local_minimum_implies_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_implies_m_value_l1019_101935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_speed_is_200_l1019_101971

/-- The distance traveled by the vehicle in kilometers -/
noncomputable def distance : ℝ := 150

/-- The time taken for the journey in hours -/
noncomputable def time : ℝ := 0.75

/-- The speed options provided in km/h -/
def speed_options : List ℝ := [200, 250, 300, 400, 500]

/-- The actual speed of the vehicle in km/h -/
noncomputable def actual_speed : ℝ := distance / time

/-- Function to calculate the absolute difference between two real numbers -/
noncomputable def abs_diff (x y : ℝ) : ℝ := abs (x - y)

theorem closest_speed_is_200 :
  ∀ s ∈ speed_options, abs_diff actual_speed 200 ≤ abs_diff actual_speed s :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_speed_is_200_l1019_101971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_perfect_square_l1019_101918

theorem odd_perfect_square (x y z : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : z * (x * z + 1)^2 = (5 * z + 2 * y) * (2 * z + y)) : 
  ∃ (k : ℕ), z = k^2 ∧ Odd k := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_perfect_square_l1019_101918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_theorem_l1019_101939

/-- Represents the athletes in the race -/
inductive Athlete : Type
  | Andrei : Athlete
  | Boris : Athlete
  | Dmitry : Athlete
  | Yevgeny : Athlete
  | Victor : Athlete
  | Gennady : Athlete

/-- Represents the finishing position of an athlete -/
def Position := Fin 6

/-- Represents the result of the race -/
def RaceResult := Athlete → Position

/-- Checks if there are exactly n athletes between two given positions -/
def exactlyNBetween (result : RaceResult) (a b : Athlete) (n : Nat) : Prop :=
  ∃ (p q : Position), result a = p ∧ result b = q ∧ 
    (p.val > q.val ∧ p.val - q.val - 1 = n) ∨
    (q.val > p.val ∧ q.val - p.val - 1 = n)

/-- Checks if one athlete finished before another -/
def finishedBefore (result : RaceResult) (a b : Athlete) : Prop :=
  (result a).val < (result b).val

/-- The main theorem stating the unique correct finishing order -/
theorem race_result_theorem (result : RaceResult) : 
  (exactlyNBetween result Athlete.Andrei Athlete.Boris 2) ∧ 
  (finishedBefore result Athlete.Dmitry Athlete.Victor) ∧
  (finishedBefore result Athlete.Victor Athlete.Gennady) ∧
  (finishedBefore result Athlete.Yevgeny Athlete.Dmitry) ∧
  (finishedBefore result Athlete.Dmitry Athlete.Boris) →
  (result Athlete.Yevgeny = ⟨0, by norm_num⟩) ∧
  (result Athlete.Dmitry = ⟨1, by norm_num⟩) ∧
  (result Athlete.Victor = ⟨2, by norm_num⟩) ∧
  (result Athlete.Gennady = ⟨3, by norm_num⟩) ∧
  (result Athlete.Boris = ⟨4, by norm_num⟩) ∧
  (result Athlete.Andrei = ⟨5, by norm_num⟩) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_theorem_l1019_101939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_x_cardinality_l1019_101962

def symmetric_difference (x y : Finset ℤ) : Finset ℤ := (x \ y) ∪ (y \ x)

theorem set_x_cardinality 
  (x y : Finset ℤ) 
  (h1 : y.card = 18)
  (h2 : (x ∩ y).card = 6)
  (h3 : (symmetric_difference x y).card = 14) :
  x.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_x_cardinality_l1019_101962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_four_thirds_l1019_101913

/-- A dilation matrix with scale factor k -/
def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- A rotation matrix by angle θ -/
noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

/-- The given result matrix -/
def result_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![15, -20; 20, 15]

theorem tan_theta_is_four_thirds 
  (k : ℝ) 
  (θ : ℝ) 
  (h1 : k > 0)
  (h2 : rotation_matrix θ * dilation_matrix k = result_matrix) : 
  Real.tan θ = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_four_thirds_l1019_101913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l1019_101930

/-- Represents a player in the coin game -/
inductive Player : Type
| Abby : Player
| Bernardo : Player
| Carl : Player
| Debra : Player
| Elina : Player

/-- Represents a ball color in the urn -/
inductive BallColor : Type
| Green : BallColor
| Red : BallColor
| White : BallColor

/-- The number of players in the game -/
def numPlayers : Nat := 5

/-- The number of rounds in the game -/
def numRounds : Nat := 3

/-- The initial number of coins each player has -/
def initialCoins : Nat := 5

/-- The number of coins transferred in each round -/
def coinsTransferred : Nat := 2

/-- The probability of a specific pair of players drawing green and red balls in a single round -/
noncomputable def probFavorableOutcome : ℚ := 1 / 10

/-- Theorem: The probability that all players have exactly 5 coins after 3 rounds is 1/1000 -/
theorem coin_game_probability :
  (probFavorableOutcome ^ numRounds : ℚ) = 1 / 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l1019_101930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1019_101936

/-- A circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- The circle passes through the pole -/
def passes_through_pole (c : PolarCircle) : Prop :=
  ∃ θ : ℝ, c.equation 0 θ

/-- The center of the circle is on the polar axis -/
def center_on_polar_axis (c : PolarCircle) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ θ : ℝ, c.equation r 0 ∨ c.equation r Real.pi

/-- The radius of the circle is 1 -/
def radius_is_one (c : PolarCircle) : Prop :=
  ∃ θ : ℝ, ∀ ρ : ℝ, c.equation ρ θ → (ρ = 1 ∨ ρ = 0)

/-- The main theorem -/
theorem circle_equation (c : PolarCircle) 
  (h1 : passes_through_pole c)
  (h2 : center_on_polar_axis c)
  (h3 : radius_is_one c) :
  ∀ ρ θ : ℝ, c.equation ρ θ ↔ ρ = 2 * Real.cos θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1019_101936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_l1019_101976

noncomputable def floor (x : ℝ) := ⌊x⌋

theorem floor_inequality (x : ℝ) :
  4 * (floor x)^2 - 36 * (floor x) + 45 < 0 ↔ 2 ≤ x ∧ x < 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_l1019_101976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_is_eight_l1019_101968

/-- Represents the fuel efficiency of a car in different environments --/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℝ
  city_miles_per_tankful : ℝ
  highway_city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data --/
noncomputable def city_mpg (car : CarFuelEfficiency) : ℝ :=
  let tank_size := car.highway_miles_per_tankful / (car.highway_city_mpg_difference + car.city_miles_per_tankful / car.highway_miles_per_tankful * car.highway_city_mpg_difference)
  car.city_miles_per_tankful / tank_size

/-- Theorem stating that for the given car data, the city mpg is 8 --/
theorem car_city_mpg_is_eight :
  let car := CarFuelEfficiency.mk 462 336 3
  city_mpg car = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_is_eight_l1019_101968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_formula_l1019_101923

/-- The total area of a rectangle with diagonal d and length three times its width,
    combined with an equilateral triangle attached to its width. -/
noncomputable def total_area (d : ℝ) : ℝ :=
  let w := d / Real.sqrt 10
  let l := 3 * w
  let rectangle_area := w * l
  let triangle_area := Real.sqrt 3 / 4 * w^2
  rectangle_area + triangle_area

/-- Theorem stating that the total area of the combined figure is (12 + √3) / 40 * d^2 -/
theorem total_area_formula (d : ℝ) (h : d > 0) :
  total_area d = (12 + Real.sqrt 3) / 40 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_formula_l1019_101923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_evaluate_expression_l1019_101996

-- Part 1
theorem simplify_expression (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (2 * a^(1/2) * b^(1/3)) * (a^(2/3) * b^(1/2)) / ((1/3) * a^(1/6) * b^(5/6)) = 6 * a :=
by sorry

-- Part 2
theorem evaluate_expression :
  (9/16)^(1/2) + 10^(Real.log 9 / Real.log 10 - 2 * Real.log 2 / Real.log 10) + Real.log (4 * Real.exp 3) - 
  (Real.log 8 / Real.log 9) * (Real.log 33 / Real.log 4) = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_evaluate_expression_l1019_101996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salesman_income_theorem_l1019_101982

/-- Represents the salesman's income over a period of weeks -/
structure SalesmanIncome where
  weeklyIncomes : List ℚ
  deriving Repr

/-- Calculate the average income over a given number of weeks -/
def averageIncome (income : SalesmanIncome) (weeks : ℕ) : ℚ :=
  (income.weeklyIncomes.take weeks).sum / weeks

/-- Calculate the total income over a given number of weeks -/
def totalIncome (income : SalesmanIncome) (weeks : ℕ) : ℚ :=
  (income.weeklyIncomes.take weeks).sum

theorem salesman_income_theorem (income : SalesmanIncome) 
    (h1 : income.weeklyIncomes.length ≥ 7)
    (h2 : averageIncome income 7 = 400)
    (h3 : averageIncome (SalesmanIncome.mk (income.weeklyIncomes.drop 5)) 2 = 365) :
    totalIncome income 5 = 2070 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salesman_income_theorem_l1019_101982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eraser_price_is_correct_l1019_101954

/-- Represents the price of a single pencil -/
noncomputable def pencil_price : ℝ := sorry

/-- Represents the price of a single eraser -/
noncomputable def eraser_price : ℝ := sorry

/-- The price of an eraser is half the price of a pencil -/
axiom eraser_pencil_price_relation : eraser_price = pencil_price / 2

/-- The price of a bundle before discount -/
noncomputable def bundle_price : ℝ := pencil_price + 2 * eraser_price

/-- The number of bundles sold -/
def bundles_sold : ℕ := 20

/-- The discount rate for 20 bundles -/
def discount_rate : ℝ := 0.3

/-- The total earnings after discount -/
def total_earnings : ℝ := 80

/-- The original price before discount -/
noncomputable def original_price : ℝ := total_earnings / (1 - discount_rate)

/-- Theorem stating that the price of one eraser is approximately $1.43 -/
theorem eraser_price_is_correct : ∃ (ε : ℝ), ε > 0 ∧ |eraser_price - 1.43| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eraser_price_is_correct_l1019_101954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rs_value_l1019_101914

theorem rs_value (r s : ℝ) (hr : r > 0) (hs : s > 0)
  (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 3/4) : r * s = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rs_value_l1019_101914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_solution_l1019_101975

theorem exponent_equation_solution : 
  ∀ x : ℝ, (12^3 : ℝ) * (6 : ℝ)^x / 432 = 144 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_solution_l1019_101975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1019_101984

open Real

theorem trigonometric_problem (α β : ℝ) 
  (h1 : Real.cos (α - β/2) = -2*sqrt 7/7)
  (h2 : Real.sin (α/2 - β) = 1/2)
  (h3 : α ∈ Set.Ioo (π/2) π)
  (h4 : β ∈ Set.Ioo 0 (π/2)) : 
  Real.cos ((α + β)/2) = -sqrt 21/14 ∧ 
  Real.tan (α + β) = 5*sqrt 3/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1019_101984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_imply_t_value_l1019_101980

-- Define the function y = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 1 + Real.log x

-- Theorem statement
theorem perpendicular_tangents_imply_t_value : 
  ∀ t : ℝ, t > 0 → 
  (f' 1 * f' t = -1) → 
  t = Real.exp (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_imply_t_value_l1019_101980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_original_price_l1019_101977

theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) : 
  sale_price = 20 →
  discount_percentage = 80 →
  sale_price = (100 - discount_percentage) / 100 * 100 →
  100 = 100 :=
by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_original_price_l1019_101977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1019_101961

open Polynomial

theorem coefficient_x_squared_in_expansion : 
  let expansion := (1 - X : Polynomial ℤ)^6 * (1 + X)^4
  coeff expansion 2 = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l1019_101961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_2021_teams_l1019_101925

/-- Represents a football championship -/
structure Championship where
  num_teams : Nat
  scoring_system : List Nat
  h1 : num_teams > 1
  h2 : scoring_system.length = 3
  h3 : scoring_system[0]! = 0
  h4 : scoring_system[1]! = 1
  h5 : scoring_system[2]! = 3

/-- The minimum score needed to have a chance to play in the final -/
def min_score_for_final (c : Championship) : Nat :=
  sorry

/-- Theorem: In a championship with 2021 teams, the minimum score
    for a team to have a chance to play in the final is 2020 points -/
theorem min_score_2021_teams :
  ∀ c : Championship,
    c.num_teams = 2021 →
    min_score_for_final c = 2020 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_2021_teams_l1019_101925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ki_production_l1019_101903

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction between KOH and NH4I -/
structure Reaction where
  koh_initial : Moles
  nh4i_initial : Moles
  ki_produced : Moles
  koh_to_ki_ratio : ℝ

/-- Theorem stating that given the initial conditions and reaction parameters, 
    the amount of KI produced is 3 moles -/
theorem ki_production (r : Reaction) 
  (h1 : r.koh_initial = (3 : ℝ))
  (h2 : r.nh4i_initial = (3 : ℝ))
  (h3 : r.koh_to_ki_ratio = 1) :
  r.ki_produced = (3 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ki_production_l1019_101903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_is_384_l1019_101915

/-- A function that counts 4-digit numbers beginning with 2 and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := Finset.range 10 \ {2}
  let count_two_twos := 3 * digits.card * digits.card
  let count_other_pairs := 3 * digits.card * digits.card
  count_two_twos + count_other_pairs

/-- Theorem stating that the count of special numbers is 384 -/
theorem count_special_numbers_is_384 : count_special_numbers = 384 := by
  unfold count_special_numbers
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_is_384_l1019_101915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_correct_l1019_101987

/-- A homogeneous truncated prism bounded by coordinate planes, x + y + z = 4, x = 1, and y = 1 -/
structure TruncatedPrism where
  -- Define the bounding planes
  plane1 : (x y z : ℝ) → x + y + z = 4
  plane2 : (x : ℝ) → x = 1
  plane3 : (y : ℝ) → y = 1
  coordPlanes : (x y z : ℝ) → x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

/-- The center of mass of the truncated prism -/
noncomputable def centerOfMass (prism : TruncatedPrism) : ℝ × ℝ × ℝ := (17/36, 17/36, 55/36)

/-- Theorem stating that the center of mass is correct -/
theorem center_of_mass_correct (prism : TruncatedPrism) :
  centerOfMass prism = (17/36, 17/36, 55/36) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_correct_l1019_101987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_factorial_by_sum_count_l1019_101948

theorem divisible_factorial_by_sum_count : 
  (Finset.filter (fun n : ℕ => n ≤ 30 ∧ n.factorial % (n * (n + 1) / 2) = 0) (Finset.range 31)).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_factorial_by_sum_count_l1019_101948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_third_shiny_on_sixth_draw_l1019_101988

def bag_size : ℕ := 7
def shiny_coins : ℕ := 4
def dull_coins : ℕ := 3
def target_draw : ℕ := 6
def target_shiny : ℕ := 3

theorem probability_third_shiny_on_sixth_draw :
  (Nat.choose (target_draw - 1) (target_shiny - 1) * 1 * 1) / Nat.choose bag_size dull_coins = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_third_shiny_on_sixth_draw_l1019_101988
