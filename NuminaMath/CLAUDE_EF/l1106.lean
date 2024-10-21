import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_Pn_l1106_110625

theorem divisibility_of_Pn (n : ℕ) : 
  (899 ∣ (36^n + 24^n - 7^n - 5^n)) ↔ Even n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_Pn_l1106_110625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1106_110675

noncomputable def f (x : ℝ) := 2 * Real.sin (-2 * x + Real.pi / 4)

theorem f_properties :
  let T := Real.pi
  let inc_intervals := [{x | 3 * Real.pi / 8 ≤ x ∧ x ≤ 7 * Real.pi / 8},
                        {x | 11 * Real.pi / 8 ≤ x ∧ x ≤ 15 * Real.pi / 8}]
  let max_val := Real.sqrt 2
  let min_val := -2
  (∀ x, f (x + T) = f x) ∧ 
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), 
    (∃ I ∈ inc_intervals, x ∈ I) ↔ (∀ y ∈ Set.Icc 0 (2 * Real.pi), x < y → f x < f y)) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_val) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ min_val) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = max_val) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = min_val) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1106_110675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_squared_value_l1106_110668

/-- Triangle ABC with circumcenter O and orthocenter H -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  H : ℝ × ℝ

/-- The circumradius of the triangle -/
def circumradius (t : Triangle) : ℝ := 9

/-- The side lengths of the triangle -/
noncomputable def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The sum of squares of side lengths -/
noncomputable def sum_of_squares (t : Triangle) : ℝ :=
  let (a, b, c) := side_lengths t
  a^2 + b^2 + c^2

/-- The squared distance between circumcenter and orthocenter -/
noncomputable def OH_squared (t : Triangle) : ℝ := sorry

theorem oh_squared_value (t : Triangle) :
  sum_of_squares t = 83 → OH_squared t = 646 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_squared_value_l1106_110668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_minus_2alpha_l1106_110607

theorem tan_beta_minus_2alpha (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2) 
  (h2 : Real.tan (α - β) = -2 / 5) : 
  Real.tan (β - 2 * α) = -1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_minus_2alpha_l1106_110607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1106_110677

/-- Geometric sequence with common ratio 2 and a_2 = 1/2 -/
def geometric_sequence (n : ℕ) : ℚ :=
  2^(n - 3)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℚ :=
  (1 - 2^n) / (-3)

theorem geometric_sequence_properties :
  (∀ n : ℕ, geometric_sequence (n + 1) = 2 * geometric_sequence n) ∧
  geometric_sequence 2 = 1/2 ∧
  (∀ n : ℕ, geometric_sequence n = 2^(n - 3)) ∧
  geometric_sum 5 = 31/16 := by
  sorry

#check geometric_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1106_110677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l1106_110615

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance between intersection points of a line passing through the focus 
    and perpendicular to an axis of symmetry -/
noncomputable def intersection_distance (h : Hyperbola) : ℝ :=
  2 * h.b^2 / h.a

theorem hyperbola_eccentricity_sqrt_3 (h : Hyperbola) 
  (h_intersection : intersection_distance h = 2 * (2 * h.a)) :
  eccentricity h = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l1106_110615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1106_110660

-- Define the points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 6)

-- Define the parabola
def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (min_val : ℝ), min_val = 9 ∧
  ∀ (P : ℝ × ℝ), on_parabola P →
    distance P A + distance P B ≥ min_val := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1106_110660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_plane_partition_l1106_110618

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internals of a plane for this problem

/-- Represents the partitioning of space by planes -/
structure SpacePartition where
  planes : Finset Plane3D
  num_regions : ℕ

/-- Predicate to check if planes are non-overlapping -/
def non_overlapping (planes : Finset Plane3D) : Prop :=
  ∀ p1 p2, p1 ∈ planes → p2 ∈ planes → p1 ≠ p2 → -- Definition left abstract
  True -- placeholder, replace with actual condition when implemented

/-- Theorem stating the possible number of regions when space is partitioned by three non-overlapping planes -/
theorem three_plane_partition (sp : SpacePartition) :
  sp.planes.card = 3 ∧ non_overlapping sp.planes →
  sp.num_regions = 4 ∨ sp.num_regions = 6 ∨ sp.num_regions = 7 ∨ sp.num_regions = 8 :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_plane_partition_l1106_110618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_m_range_l1106_110610

-- Define the ellipse equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (9 - m) + y^2 / (2 * m) = 1 ∧ 9 - m > 0 ∧ 2 * m > 0

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / 5 - y^2 / m = 1 ∧ m > 0

-- Define the eccentricity condition
def eccentricity_condition (m : ℝ) : Prop :=
  let e := Real.sqrt ((5 + m) / 5)
  Real.sqrt 6 / 2 < e ∧ e < Real.sqrt 2

-- Main theorem
theorem ellipse_hyperbola_m_range (m : ℝ) 
  (h1 : is_ellipse m) 
  (h2 : is_hyperbola m) 
  (h3 : eccentricity_condition m) : 
  5/2 < m ∧ m < 3 := by
  sorry

#check ellipse_hyperbola_m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_m_range_l1106_110610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_is_half_l1106_110632

/-- A configuration of squares and rectangles forming a large square -/
structure SquareConfiguration where
  /-- The side length of each small square -/
  small_square_side : ℝ
  /-- The side length of the large square -/
  large_square_side : ℝ
  /-- The number of small squares -/
  num_small_squares : ℕ
  /-- The number of rectangles -/
  num_rectangles : ℕ

/-- The ratio of length to width of a rectangle in the configuration -/
noncomputable def rectangle_ratio (config : SquareConfiguration) : ℝ :=
  let rectangle_length := (config.large_square_side - config.small_square_side * config.num_small_squares) / config.num_rectangles
  let rectangle_width := config.large_square_side - config.small_square_side * config.num_small_squares
  rectangle_length / rectangle_width

/-- Theorem stating that the ratio of length to width of a rectangle is 1/2 -/
theorem rectangle_ratio_is_half (config : SquareConfiguration) 
    (h1 : config.small_square_side = 1)
    (h2 : config.large_square_side = 5)
    (h3 : config.num_small_squares = 3)
    (h4 : config.num_rectangles = 2) : 
  rectangle_ratio config = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_is_half_l1106_110632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1106_110673

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = Real.pi ∧    -- Sum of angles in a triangle
  a = 5 ∧                  -- Side a
  b = 4 ∧                  -- Side b
  c = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) ∧  -- Law of cosines
  Real.cos (A - B) = 31/32 →
  -- Conclusions to prove
  Real.cos C = 1/8 ∧
  c = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1106_110673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_239_tan_149_l1106_110628

theorem sin_239_tan_149 (a : ℝ) (h : Real.cos (31 * π / 180) = a) :
  Real.sin (239 * π / 180) * Real.tan (149 * π / 180) = Real.sqrt (1 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_239_tan_149_l1106_110628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_l1106_110654

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - (1/2)^x)

-- State the theorem about the domain and range of f
theorem f_domain_and_range :
  (∀ x : ℝ, f x ≠ 0 → x ≥ 0) ∧
  (∀ y : ℝ, (∃ x : ℝ, f x = y) → 0 ≤ y ∧ y < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_l1106_110654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_1101_equals_13_l1106_110626

/-- Converts a binary digit (0 or 1) to its decimal value -/
def binaryToDecimal (digit : Nat) : Nat :=
  if digit = 0 || digit = 1 then digit else 0

/-- Calculates the decimal value of a binary digit at a given position -/
def binaryDigitValue (digit : Nat) (position : Nat) : Nat :=
  binaryToDecimal digit * 2^position

/-- The binary number 1101 -/
def binaryNumber : List Nat := [1, 1, 0, 1]

theorem binary_1101_equals_13 :
  (List.sum (List.zipWith binaryDigitValue (List.reverse binaryNumber) (List.range 4))) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_1101_equals_13_l1106_110626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_point_zero_one_l1106_110661

/-- The function h(x) -/
noncomputable def h (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4) + 2

/-- Theorem stating the symmetry between f(x) and h(x) about the point (0,1) -/
theorem symmetry_about_point_zero_one :
  ∀ x : ℝ, f x = 2 - h (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_point_zero_one_l1106_110661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lee_family_seating_arrangements_l1106_110606

/-- Represents the Lee family members -/
inductive LeeFamilyMember
  | MrLee
  | MrsLee
  | Child1
  | Child2
  | Child3
deriving Fintype, DecidableEq

/-- Represents a seating arrangement in the Lee family SUV -/
structure SUVSeating where
  driver : LeeFamilyMember
  frontPassenger : LeeFamilyMember
  backSeat1 : LeeFamilyMember
  backSeat2 : LeeFamilyMember
  backSeat3 : LeeFamilyMember
  driver_is_parent : driver = LeeFamilyMember.MrLee ∨ driver = LeeFamilyMember.MrsLee
  all_different : driver ≠ frontPassenger ∧
                  driver ≠ backSeat1 ∧ driver ≠ backSeat2 ∧ driver ≠ backSeat3 ∧
                  frontPassenger ≠ backSeat1 ∧ frontPassenger ≠ backSeat2 ∧ frontPassenger ≠ backSeat3 ∧
                  backSeat1 ≠ backSeat2 ∧ backSeat1 ≠ backSeat3 ∧
                  backSeat2 ≠ backSeat3
deriving DecidableEq

instance : Fintype SUVSeating := by
  sorry -- The implementation of this instance is complex and not necessary for this example

/-- The number of possible seating arrangements for the Lee family in their SUV is 48 -/
theorem lee_family_seating_arrangements :
  Fintype.card SUVSeating = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lee_family_seating_arrangements_l1106_110606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1106_110630

-- Define the equation
def equation (x : ℝ) : Prop := x * (5 * x - 11) = -4

-- Define the solutions
noncomputable def solution1 : ℝ := (11 + Real.sqrt 41) / 10
noncomputable def solution2 : ℝ := (11 - Real.sqrt 41) / 10

-- Define m, n, p
def m : ℕ := 11
def n : ℕ := 41
def p : ℕ := 10

theorem equation_solutions :
  (equation solution1 ∧ equation solution2) ∧
  (Nat.gcd m (Nat.gcd n p) = 1) ∧
  (m + n + p = 62) := by
  sorry

#eval m + n + p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1106_110630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_triangle_perimeter_l1106_110651

/-- The largest perimeter of a triangle with two sides 7 and 9, and the third side an integer -/
theorem largest_triangle_perimeter : 
  ∀ y : ℤ, 
  y > 0 → 
  y + 7 > 9 → 
  y + 9 > 7 → 
  7 + 9 > y → 
  (∀ z : ℤ, z > 0 → z + 7 > 9 → z + 9 > 7 → 7 + 9 > z → y + 7 + 9 ≥ z + 7 + 9) → 
  y + 7 + 9 = 31 := by
  intro y hy1 hy2 hy3 hy4 hmax
  sorry

#check largest_triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_triangle_perimeter_l1106_110651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f4_range_f3_symmetry_f8_symmetry_l1106_110635

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := Real.sin x ^ n + Real.cos x ^ n

theorem f4_range : Set.range (f 4) = Set.Icc (1/2) 1 := by
  sorry

theorem f3_symmetry (x : ℝ) : f 3 (3 * Real.pi / 4 + x) + f 3 (3 * Real.pi / 4 - x) = 0 := by
  sorry

theorem f8_symmetry (x : ℝ) : f 8 (Real.pi / 4 + x) = f 8 (Real.pi / 4 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f4_range_f3_symmetry_f8_symmetry_l1106_110635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1106_110666

/-- Given a triangle ABC with the following properties:
    - D is the midpoint of side BC
    - AD = √10/2
    - 8a * sin(B) = 3√15 * c
    - cos(A) = -1/4
    Prove that the area of triangle ABC is 3√15/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  (D.1 + D.2) / 2 = (b + c) / 2 →  -- D is midpoint of BC
  (a^2 + D.1^2 - (Real.sqrt 10/2)^2) / (2 * a * D.1) = 0 →  -- AD = √10/2 (using cosine rule)
  8 * a * Real.sin B = 3 * Real.sqrt 15 * c →
  Real.cos A = -1/4 →
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 15 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1106_110666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_dihedral_angle_of_planes_l1106_110692

/-- The acute dihedral angle between two planes with given normal vectors -/
theorem acute_dihedral_angle_of_planes (n ν : ℝ × ℝ × ℝ) 
  (hn : n = (1, 0, 1)) (hν : ν = (-1, 1, 0)) : 
  Real.arccos (|(n.1 * ν.1 + n.2.1 * ν.2.1 + n.2.2 * ν.2.2)| / 
    (Real.sqrt (n.1^2 + n.2.1^2 + n.2.2^2) * Real.sqrt (ν.1^2 + ν.2.1^2 + ν.2.2^2))) * (180 / Real.pi) = 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_dihedral_angle_of_planes_l1106_110692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_between_lines_l1106_110681

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the intersection points
def intersectionPoints (k₁ k₂ : Circle) : Set (ℝ × ℝ) :=
  {p | (p.1 - k₁.center.1)^2 + (p.2 - k₁.center.2)^2 = k₁.radius^2 ∧
       (p.1 - k₂.center.1)^2 + (p.2 - k₂.center.2)^2 = k₂.radius^2}

-- Define the angle between two lines
noncomputable def angleBetweenLines (l₁ l₂ : Line) : ℝ :=
  sorry

-- Define point on circle
def pointOnCircle (p : ℝ × ℝ) (k : Circle) : Prop :=
  (p.1 - k.center.1)^2 + (p.2 - k.center.2)^2 = k.radius^2

-- Define point on line
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  (p.1 - l.point1.1) * (l.point2.2 - l.point1.2) = 
  (p.2 - l.point1.2) * (l.point2.1 - l.point1.1)

-- Main theorem
theorem constant_angle_between_lines (k₁ k₂ : Circle) (A B : ℝ × ℝ) 
  (h : {A, B} ⊆ intersectionPoints k₁ k₂) :
  ∀ (c d : Line),
    c.point1 = A → d.point1 = A →
    ∃ (C₁ D₁ C₂ D₂ : ℝ × ℝ),
      pointOnCircle C₁ k₁ ∧ pointOnLine C₁ c ∧ C₁ ≠ A ∧
      pointOnCircle D₁ k₁ ∧ pointOnLine D₁ d ∧ D₁ ≠ A ∧
      pointOnCircle C₂ k₂ ∧ pointOnLine C₂ c ∧ C₂ ≠ A ∧
      pointOnCircle D₂ k₂ ∧ pointOnLine D₂ d ∧ D₂ ≠ A →
    ∃ (θ : ℝ), angleBetweenLines (Line.mk C₁ D₁) (Line.mk C₂ D₂) = θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_between_lines_l1106_110681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1106_110659

theorem trigonometric_inequality (α β x : ℝ) 
  (h1 : Real.tan (α/2) ^ 2 ≤ Real.tan (β/2) ^ 2) 
  (h2 : ∀ k : ℤ, β ≠ k * Real.pi) : 
  (Real.sin (α/2) ^ 2) / (Real.sin (β/2) ^ 2) ≤ 
  (x^2 - 2*x*Real.cos α + 1) / (x^2 - 2*x*Real.cos β + 1) ∧
  (x^2 - 2*x*Real.cos α + 1) / (x^2 - 2*x*Real.cos β + 1) ≤ 
  (Real.cos (α/2) ^ 2) / (Real.cos (β/2) ^ 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1106_110659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_function_minimum_l1106_110679

theorem sec_function_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.cosh (b * x) ≥ 3) ∧ 
  (∃ x, a * Real.cosh (b * x) = 3) →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_function_minimum_l1106_110679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_interval_l1106_110657

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem zeros_in_interval (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 3)
  (h_zero : f 2 = 0) :
  ∃! (s : Finset ℝ), s.card = 2 ∧ (∀ x ∈ s, 0 < x ∧ x < 6 ∧ f x = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_interval_l1106_110657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_problem_l1106_110636

/-- Given two parallel lines l₁ and l₂, prove the value of a and the distance between the lines -/
theorem parallel_lines_problem (a : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ x - y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ x + a * y + 3 = 0
  (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ ∧ l₂ x₂ y₂ → (x₂ - x₁) / (y₂ - y₁) = (x₁ - y₁) / 1) →
  a = -1 ∧
  (let d := |1 - 3| / Real.sqrt (1^2 + (-1)^2)
   d = Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_problem_l1106_110636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1106_110696

noncomputable def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def midpoint_OF₂ : ℝ × ℝ := (Real.sqrt 30 / 6, 0)

noncomputable def point_M : ℝ × ℝ := (-7/3, 0)

theorem ellipse_properties
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (h_midpoint : midpoint_OF₂ = (Real.sqrt 30 / 6, 0))
  (h_AB_F₂B : ∃ (F₂ B : ℝ × ℝ), 
    (a, 0) - (-a, 0) = (5 + 2 * Real.sqrt 6) • (B - F₂))
  (k : ℝ)
  (P Q : ℝ × ℝ)
  (h_PQ : ellipse_C P.1 P.2 a b ∧ ellipse_C Q.1 Q.2 a b)
  (h_line : P.2 = k * (P.1 + 1) ∧ Q.2 = k * (Q.1 + 1)) :
  (a = Real.sqrt 5 ∧ b = Real.sqrt (5/3)) ∧
  ((P - point_M) • (Q - point_M) = 4/9) := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1106_110696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l1106_110649

theorem triangle_angle_theorem (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_equation : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
  B = π/3 ∨ B = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l1106_110649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_minimization_l1106_110650

/-- 
Given a right-angled triangle with height m and hypotenuse c, 
prove that c ≥ 2m, with equality if and only if the triangle is isosceles.
-/
theorem hypotenuse_minimization (m c : ℝ) (h_positive : m > 0) : 
  c ≥ 2 * m ∧ (c = 2 * m ↔ ∃ a : ℝ, a > 0 ∧ c = a * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_minimization_l1106_110650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_eight_l1106_110619

theorem ceiling_sum_equals_eight :
  ⌈(Real.sqrt (16/9 : ℝ))⌉ + ⌈(16/9 : ℝ)⌉ + ⌈((16/9 : ℝ)^2)⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_eight_l1106_110619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_eq_volume_ratio_l1106_110627

/-- A cone with a sphere inscribed in it -/
structure InscribedSphere where
  l : ℝ  -- slant height of the cone
  r : ℝ  -- radius of the cone's base
  R : ℝ  -- radius of the inscribed sphere
  h : ℝ  -- height of the cone
  h_eq : h = Real.sqrt (l^2 - r^2)
  R_eq : R * (l + r) = r * Real.sqrt (l^2 - r^2)

/-- The surface area of the cone -/
noncomputable def cone_surface_area (c : InscribedSphere) : ℝ :=
  Real.pi * c.r^2 + Real.pi * c.r * c.l

/-- The volume of the cone -/
noncomputable def cone_volume (c : InscribedSphere) : ℝ :=
  (1/3) * Real.pi * c.r^2 * c.h

/-- The surface area of the sphere -/
noncomputable def sphere_surface_area (c : InscribedSphere) : ℝ :=
  4 * Real.pi * c.R^2

/-- The volume of the sphere -/
noncomputable def sphere_volume (c : InscribedSphere) : ℝ :=
  (4/3) * Real.pi * c.R^3

/-- The theorem stating that the ratio of surface areas equals the ratio of volumes -/
theorem surface_area_ratio_eq_volume_ratio (c : InscribedSphere) :
  cone_surface_area c / sphere_surface_area c = cone_volume c / sphere_volume c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_eq_volume_ratio_l1106_110627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l1106_110644

theorem tan_one_condition (x : ℝ) :
  (x = π / 4 → Real.tan x = 1) ∧ ¬(Real.tan x = 1 → x = π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_one_condition_l1106_110644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1106_110643

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (3 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : -π/2 < φ ∧ φ < π/2) 
  (h2 : ∀ x, f φ (x - π/4) = f φ (π/4 - x)) :
  φ = -π/4 ∧ 
  (∀ x, f φ (x + π/12) = -f φ (-x + π/12)) ∧
  (∀ x₁ x₂, |f φ x₁ - f φ x₂| = 2 → |x₁ - x₂| ≥ π/3) ∧
  (∃ x₁ x₂, |f φ x₁ - f φ x₂| = 2 ∧ |x₁ - x₂| = π/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1106_110643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l1106_110602

/-- A line in 2D space defined by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Calculate the y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Calculate the x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ :=
  (l.point.2 - l.slope * l.point.1) / l.slope

/-- Calculate the area of a triangle given the coordinates of its vertices -/
noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

theorem area_of_quadrilateral_OBEC : 
  let line1 : Line := { slope := -3, point := (3, 3) }
  let line2 : Line := { slope := 1, point := (3, 3) }
  let O : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, y_intercept line1)
  let C : ℝ × ℝ := (6, 0)
  let E : ℝ × ℝ := (3, 3)
  triangle_area O B E + triangle_area O E C = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l1106_110602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_exchange_700_to_400_l1106_110662

/-- Represents the number of cars of each type -/
structure CarCount where
  zhiguli : ℕ
  volga : ℕ
  mercedes : ℕ

/-- Represents the exchange rates between car types -/
def exchange_rate1 (c c' : CarCount) : Prop :=
  c.zhiguli ≥ 3 ∧
  c'.zhiguli + 3 = c.zhiguli ∧
  c'.volga = c.volga + 1 ∧
  c'.mercedes = c.mercedes + 1

def exchange_rate2 (c c' : CarCount) : Prop :=
  c.volga ≥ 3 ∧
  c'.volga + 3 = c.volga ∧
  c'.zhiguli = c.zhiguli + 2 ∧
  c'.mercedes = c.mercedes + 1

/-- The main theorem stating that it's impossible to exchange 700 Zhiguli for 400 Mercedes -/
theorem no_exchange_700_to_400 : ¬∃ (steps : ℕ) (exchange_sequence : ℕ → CarCount),
  exchange_sequence 0 = ⟨700, 0, 0⟩ ∧
  exchange_sequence steps = ⟨0, 0, 400⟩ ∧
  ∀ (i : ℕ), i < steps →
    (∃ (c' : CarCount), exchange_rate1 (exchange_sequence i) c' ∧ exchange_sequence (i + 1) = c') ∨
    (∃ (c' : CarCount), exchange_rate2 (exchange_sequence i) c' ∧ exchange_sequence (i + 1) = c') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_exchange_700_to_400_l1106_110662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_equivalence_l1106_110608

theorem log_condition_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hna : a ≠ 1) :
  Real.log b / Real.log a > 0 ↔ (a - 1) * (b - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_equivalence_l1106_110608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangents_l1106_110672

/-- A point in the first quadrant on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  first_quadrant : x > 0 ∧ y > 0
  on_parabola : x^2 = 4*y
  y_gt_4 : y > 4

/-- Circle with equation x² + (y-2)² = 4 -/
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

/-- Tangent points from a point to the circle -/
def tangent_points (P : ParabolaPoint) (M N : ℝ × ℝ) : Prop :=
  circle_eq M.1 M.2 ∧ circle_eq N.1 N.2 ∧
  (∃ (k₁ k₂ : ℝ), 
    (M.2 - P.y) = k₁ * (M.1 - P.x) ∧
    (N.2 - P.y) = k₂ * (N.1 - P.x) ∧
    k₁ ≠ k₂)

/-- Intersection points of tangents with x-axis -/
def x_axis_intersections (P : ParabolaPoint) (M N B C : ℝ × ℝ) : Prop :=
  tangent_points P M N ∧
  B.2 = 0 ∧ C.2 = 0 ∧
  (∃ (k₁ k₂ : ℝ),
    (M.2 - P.y) = k₁ * (M.1 - P.x) ∧
    (N.2 - P.y) = k₂ * (N.1 - P.x) ∧
    B.1 = P.x - P.y / k₁ ∧
    C.1 = P.x - P.y / k₂)

theorem parabola_circle_tangents (P : ParabolaPoint) (M N B C : ℝ × ℝ) :
  x_axis_intersections P M N B C →
  (Real.sqrt ((M.1 - P.x)^2 + (M.2 - P.y)^2) = P.y ∧ Real.sqrt ((N.1 - P.x)^2 + (N.2 - P.y)^2) = P.y) ∧
  (P.y = 9 → abs ((C.1 - B.1) * P.y / 2) = 162 / 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangents_l1106_110672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_H2O_approx_l1106_110617

/-- The mass percentage of hydrogen in water (H2O) -/
noncomputable def mass_percentage_H_in_H2O : ℝ :=
  let molar_mass_H : ℝ := 1.01
  let molar_mass_O : ℝ := 16.00
  let H_atoms_in_H2O : ℕ := 2
  let O_atoms_in_H2O : ℕ := 1
  let mass_H_in_H2O : ℝ := H_atoms_in_H2O * molar_mass_H
  let mass_H2O : ℝ := mass_H_in_H2O + O_atoms_in_H2O * molar_mass_O
  (mass_H_in_H2O / mass_H2O) * 100

theorem mass_percentage_H_in_H2O_approx :
  |mass_percentage_H_in_H2O - 11.21| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_H2O_approx_l1106_110617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_tan_arcsin_three_fifths_l1106_110620

theorem cos_tan_arcsin_three_fifths :
  (Real.cos (Real.arcsin (3/5)) = 4/5) ∧ (Real.tan (Real.arcsin (3/5)) = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_tan_arcsin_three_fifths_l1106_110620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_volume_l1106_110613

noncomputable def cube_root_2 : ℝ := 1.2599

noncomputable def volume_fraction (a b c : ℕ) : ℝ :=
  2 * (⌊(a : ℝ) / cube_root_2⌋ * ⌊(b : ℝ) / cube_root_2⌋ * ⌊(c : ℝ) / cube_root_2⌋) / (a * b * c)

theorem rectangular_box_volume (a b c : ℕ) (h1 : 0 < a ∧ a ≤ b ∧ b ≤ c)
  (h2 : volume_fraction a b c = 0.4) :
  a * b * c = 60 ∨ a * b * c = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_volume_l1106_110613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_special_complex_numbers_l1106_110612

noncomputable def complex_midpoint (z₁ z₂ : ℂ) : ℂ := (z₁ + z₂) / 2

theorem midpoint_of_special_complex_numbers :
  complex_midpoint (1 / (1 + Complex.I)) (1 / (1 - Complex.I)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_special_complex_numbers_l1106_110612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l1106_110694

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 1

-- Theorem statement
theorem f_has_unique_zero : ∃! x : ℝ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l1106_110694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_chord_through_point_l1106_110646

/-- A convex figure in a 2D plane -/
structure ConvexFigure where
  -- Add necessary fields and properties for a convex figure
  boundary : Set Point
  interior : Set Point

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A chord of a convex figure -/
structure Chord (F : ConvexFigure) where
  start : Point
  endpoint : Point
  starts_on_boundary : start ∈ F.boundary
  ends_on_boundary : endpoint ∈ F.boundary

/-- Predicate to check if a point is inside a convex figure -/
def is_inside (F : ConvexFigure) (P : Point) : Prop :=
  P ∈ F.interior

/-- Function to calculate distance between two points -/
noncomputable def distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

/-- Predicate to check if a point divides a chord into two equal parts -/
def divides_equally (C : Chord F) (P : Point) : Prop :=
  distance C.start P = distance P C.endpoint

/-- Predicate to check if a point is on a chord -/
def on_chord (C : Chord F) (P : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    P.x = C.start.x + t * (C.endpoint.x - C.start.x) ∧
    P.y = C.start.y + t * (C.endpoint.y - C.start.y)

/-- The main theorem -/
theorem exists_equal_chord_through_point 
  (F : ConvexFigure) (A : Point) (h : is_inside F A) :
  ∃ (C : Chord F), C.start ≠ C.endpoint ∧ on_chord C A ∧ divides_equally C A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_chord_through_point_l1106_110646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_sum_l1106_110641

theorem consecutive_numbers_sum : 
  ∃ n : ℤ, ∃ k ∈ ({n, n+1, n+2, n+3, n+4, n+5} : Set ℤ), 
    (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) - k = 10085 ∧
    2020 ∈ ({n, n+1, n+2, n+3, n+4, n+5} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_sum_l1106_110641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l1106_110689

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 3 then 2 * x^2 - 3 else b * x - 5

theorem continuous_piecewise_function (b : ℝ) :
  Continuous (f b) ↔ b = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l1106_110689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1106_110605

theorem largest_expression :
  let a := Real.sqrt (Real.rpow 7 (1/3) * Real.rpow 8 (1/3))
  let b := Real.sqrt (8 * Real.rpow 7 (1/3))
  let c := Real.sqrt (7 * Real.rpow 8 (1/3))
  let d := Real.rpow (7 * Real.sqrt 8) (1/3)
  let e := Real.rpow (8 * Real.sqrt 7) (1/3)
  b > a ∧ b > c ∧ b > d ∧ b > e := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1106_110605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1106_110614

/-- Parabola with vertex at origin and focus at (0,1) -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_focus : focus = (0, 1)

/-- Line l: x - y - 2 = 0 -/
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

/-- Point on line l -/
structure PointOnL where
  x : ℝ
  y : ℝ
  h_on_l : line_l x y

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Tangent line to parabola at a point -/
def tangent_line (C : Parabola) (p : ℝ × ℝ) : ℝ → ℝ → Prop := sorry

theorem parabola_properties (C : Parabola) (P : PointOnL) :
  /- 1. Equation of parabola C -/
  C.equation = fun x y => x^2 = 4*y ∧
  /- 2. Equation of line AB -/
  (∃ A B : ℝ × ℝ, 
    C.equation A.1 A.2 ∧ 
    C.equation B.1 B.2 ∧ 
    tangent_line C A P.x P.y ∧ 
    tangent_line C B P.x P.y ∧
    (fun x y => P.x * x - 2*y - 2*P.y = 0) = fun x y => 
      (A.2 - B.2) / (A.1 - B.1) * (x - A.1) + A.2 = y) ∧
  /- 3. Minimum value of |AF|·|BF| -/
  (∃ min_value : ℝ, 
    min_value = 9/2 ∧
    ∀ P' : PointOnL, 
      ∃ A' B' : ℝ × ℝ,
        C.equation A'.1 A'.2 ∧ 
        C.equation B'.1 B'.2 ∧ 
        tangent_line C A' P'.x P'.y ∧ 
        tangent_line C B' P'.x P'.y ∧
        distance A' C.focus * distance B' C.focus ≥ min_value) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1106_110614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l1106_110624

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- Definition of the first line: ax - y = 1 -/
def line1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x - y = 1

/-- Definition of the second line: (2 - a)x + ay = -1 -/
def line2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => (2 - a) * x + a * y = -1

/-- The slope of the first line -/
noncomputable def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line -/
noncomputable def slope2 (a : ℝ) : ℝ := -(2 - a) / a

theorem perpendicular_lines_a_equals_one :
  ∀ a : ℝ, a ≠ 0 → perpendicular (slope1 a) (slope2 a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l1106_110624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lakeisha_lawn_mowing_l1106_110616

/-- LaKeisha's lawn mowing problem -/
theorem lakeisha_lawn_mowing (rate : ℚ) (book_cost : ℚ) (mowed_lawns : ℕ) 
  (lawn_length : ℕ) (lawn_width : ℕ) : 
  rate = 1/10 → 
  book_cost = 150 → 
  mowed_lawns = 3 → 
  lawn_length = 20 → 
  lawn_width = 15 → 
  (book_cost - (rate * (mowed_lawns : ℚ) * (lawn_length : ℚ) * (lawn_width : ℚ))) / rate = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lakeisha_lawn_mowing_l1106_110616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1106_110629

noncomputable def f (x : ℝ) := Real.log (x - 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1106_110629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_points_l1106_110601

theorem power_function_through_points (n : ℚ) (m : ℝ) :
  (8 : ℝ)^(n : ℝ) = 4 ∧ (-8 : ℝ)^(n : ℝ) = m → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_points_l1106_110601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_seven_l1106_110648

def divisors_of_196 : List ℕ := [2, 4, 7, 14, 28, 49, 98, 196]

def has_common_factor_greater_than_one (a b : ℕ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ k ∣ a ∧ k ∣ b

def is_valid_arrangement (arr : List ℕ) : Prop :=
  arr.length > 1 ∧
  ∀ i, i < arr.length → has_common_factor_greater_than_one (arr.get! i) (arr.get! ((i+1) % arr.length))

theorem sum_of_adjacent_to_seven (arr : List ℕ) :
  arr.toFinset = divisors_of_196.toFinset →
  is_valid_arrangement arr →
  ∃ i, i < arr.length ∧ arr.get! i = 7 ∧ 
    arr.get! ((i-1) % arr.length) + arr.get! ((i+1) % arr.length) = 63 :=
by sorry

#eval divisors_of_196

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_seven_l1106_110648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_perimeter_l1106_110690

/-- A region with specific properties -/
structure Region where
  side_length : ℚ
  num_sides : ℕ
  area : ℚ

/-- The perimeter of the region -/
noncomputable def perimeter (r : Region) : ℚ := 
  12 + 2 * (r.area + 8) / 12 + 4 + 8 * r.side_length

/-- Theorem: The perimeter of the region with given properties is 30 feet -/
theorem region_perimeter : 
  ∀ (r : Region), r.side_length = 1 ∧ r.num_sides = 12 ∧ r.area = 69 → perimeter r = 30 :=
by
  intro r h
  simp [perimeter]
  norm_num
  ring
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_perimeter_l1106_110690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_18_to_2023_is_correct_l1106_110695

def closest_multiple_of_18_to_2023 : ℤ := 2028

theorem closest_multiple_of_18_to_2023_is_correct :
  ∀ n : ℤ, n ≠ closest_multiple_of_18_to_2023 →
  n % 18 = 0 →
  |n - 2023| ≥ |closest_multiple_of_18_to_2023 - 2023| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_18_to_2023_is_correct_l1106_110695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_theorem_l1106_110600

/-- Represents a grid with rooks -/
structure RookGrid (n : ℕ) where
  size : ℕ := 2 * n
  rooks : ℕ := 2 * n
  is_valid : Prop := 0 < n

/-- Represents a partition of the grid into two connected parts -/
structure GridPartition (n : ℕ) where
  grid : RookGrid n
  is_symmetric : Prop
  is_connected : Prop

/-- The maximum number of rooks that can be in one part of the partition -/
def max_rooks_in_part (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem stating the maximum number of rooks in one part -/
theorem max_rooks_theorem (n : ℕ) (p : GridPartition n) :
  p.grid.is_valid →
  p.is_symmetric →
  p.is_connected →
  max_rooks_in_part n = 2 * n - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_theorem_l1106_110600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_ratio_l1106_110652

/-- The curve equation -/
def curve (x y : ℝ) : Prop :=
  3 * x^2 + 2 * x * y + 4 * y^2 - 13 * x - 24 * y + 48 = 0

/-- The set of all points (x, y) on the curve -/
noncomputable def curve_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve p.1 p.2}

/-- The ratio y/x for a point (x, y) -/
noncomputable def ratio (p : ℝ × ℝ) : ℝ := p.2 / p.1

/-- Theorem: The sum of the maximum and minimum values of y/x on the curve is 5/4 -/
theorem sum_of_max_min_ratio :
  ∃ (max min : ℝ),
    (∀ p ∈ curve_points, ratio p ≤ max) ∧
    (∀ p ∈ curve_points, ratio p ≥ min) ∧
    (∃ p1 p2 : ℝ × ℝ, p1 ∈ curve_points ∧ p2 ∈ curve_points ∧ ratio p1 = max ∧ ratio p2 = min) ∧
    max + min = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_ratio_l1106_110652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_path_length_l1106_110645

-- Define necessary constants and functions
noncomputable def CircleDiameter : Point → Point → Point → Real := sorry
noncomputable def SegmentLength : Point → Point → Real := sorry
noncomputable def OnCircle : Point → Point → Point → Point → Prop := sorry
noncomputable def HighestPoint : Point → Point → Point → Point → Prop := sorry
noncomputable def BrokenLinePath : Point → Point → Point → Real := sorry

theorem broken_line_path_length (O A B C D P : Point) : 
  CircleDiameter O A B = 12 →
  SegmentLength A C = 3 →
  SegmentLength B D = 5 →
  OnCircle P O A B →
  HighestPoint P O A B →
  BrokenLinePath C P D = 3 * Real.sqrt 5 + Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_path_length_l1106_110645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_integers_102_to_200_l1106_110680

def sum_first_n_even_integers (n : ℕ) : ℕ := n * (n + 1)

def sum_even_integers_range (start finish : ℕ) : ℕ :=
  let first_term := if start % 2 = 0 then start else start + 1
  let last_term := if finish % 2 = 0 then finish else finish - 1
  let n := (last_term - first_term) / 2 + 1
  n * (first_term + last_term) / 2

theorem sum_even_integers_102_to_200 :
  sum_first_n_even_integers 50 = 2550 →
  sum_even_integers_range 102 200 = 7550 := by
  sorry

#eval sum_even_integers_range 102 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_integers_102_to_200_l1106_110680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1106_110667

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def line (x y : ℝ) : Prop :=
  y = x + 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt ((a^2 + b^2) / a^2)

theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (intersection : ∃ (A B : ℝ × ℝ), 
    hyperbola a b A.1 A.2 ∧ 
    hyperbola a b B.1 B.2 ∧
    line A.1 A.2 ∧ 
    line B.1 B.2)
  (asymptotes_origin : ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ b * x = a * y ∧ b * x = -a * y)
  (A_first_quadrant : ∃ (x y : ℝ), hyperbola a b x y ∧ line x y ∧ x > 0 ∧ y > 0)
  (OA_OB_relation : ∃ (A B : ℝ × ℝ),
    hyperbola a b A.1 A.2 ∧ 
    hyperbola a b B.1 B.2 ∧
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧
    A.1^2 + A.2^2 = 4 * (B.1^2 + B.2^2)) :
  eccentricity a b = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1106_110667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1106_110655

theorem sin_2alpha_value (α : ℝ) (h : Real.sin (π/4 + α) ^ 2 = 2/3) : 
  Real.sin (2 * α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1106_110655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_real_roots_quadratic_l1106_110638

theorem non_real_roots_quadratic (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ b ∈ Set.Ioo (-8 : ℝ) 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_real_roots_quadratic_l1106_110638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlson_jam_fraction_l1106_110697

/-- The fraction of jam Carlson ate from a jar shared with Malysh -/
theorem carlson_jam_fraction :
  ∀ (n m : ℝ),
  n > 0 → m > 0 →
  let k := 0.6 * n  -- Carlson ate 40% fewer spoons than Malysh
  let p := 2.5 * m  -- Carlson's spoons held 150% more jam
  (k * p) / (n * m + k * p) = 3 / 5 :=
by
  intros n m hn hm
  -- Define k and p
  let k := 0.6 * n
  let p := 2.5 * m
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlson_jam_fraction_l1106_110697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1106_110647

-- Define the function f(x) = x sin(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

-- State the theorem
theorem f_inequality : f (-π/3) > f 1 ∧ f 1 > f (π/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1106_110647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_iff_equal_areas_l1106_110634

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Fin 2 → ℝ)

-- Define the point P
noncomputable def P (q : Quadrilateral) : Fin 2 → ℝ := sorry

-- Define the condition that ABCD is convex
def is_convex (q : Quadrilateral) : Prop := sorry

-- Define the condition that diagonals AC and BD are perpendicular
def diagonals_perpendicular (q : Quadrilateral) : Prop := sorry

-- Define the condition that opposite sides AB and CD are not parallel
def opposite_sides_not_parallel (q : Quadrilateral) : Prop := sorry

-- Define the condition that P is the intersection of perpendicular bisectors of AB and CD
def P_is_intersection_of_bisectors (q : Quadrilateral) : Prop := sorry

-- Define the condition that P lies inside quadrilateral ABCD
def P_inside_quadrilateral (q : Quadrilateral) : Prop := sorry

-- Define the condition that ABCD is cyclic
def is_cyclic (q : Quadrilateral) : Prop := sorry

-- Define the area of a triangle
noncomputable def area_triangle (A B C : Fin 2 → ℝ) : ℝ := sorry

-- Theorem statement
theorem cyclic_iff_equal_areas (q : Quadrilateral) :
  is_convex q →
  diagonals_perpendicular q →
  opposite_sides_not_parallel q →
  P_is_intersection_of_bisectors q →
  P_inside_quadrilateral q →
  (is_cyclic q ↔ area_triangle q.A q.B (P q) = area_triangle q.C q.D (P q)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_iff_equal_areas_l1106_110634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1106_110640

noncomputable def x : ℤ := ⌊5 - Real.sqrt 3⌋
noncomputable def y : ℝ := 5 - Real.sqrt 3 - x

theorem problem_solution : 2 * x^3 - (y^3 + 1 / y^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1106_110640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1106_110653

def z : ℂ := Complex.I * (1 + Complex.I)

theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1106_110653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l1106_110687

theorem perfect_square_sum (n : ℕ) : 
  ∃ k : ℤ, (4 / 9 : ℚ) * (10^(2*n) - 1) + 2 * (8 / 9 : ℚ) * (10^n - 1) + 4 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l1106_110687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l1106_110698

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x + Real.pi / 3) + Real.sqrt 3 * Real.sin (ω * x - Real.pi / 6)

theorem symmetry_center_of_f (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x : ℝ, f ω (x + Real.pi) = f ω x) 
  (h_smallest_period : ∀ T : ℝ, 0 < T → T < Real.pi → ¬(∀ x : ℝ, f ω (x + T) = f ω x)) :
  ∀ x : ℝ, f ω (Real.pi - x) = f ω x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l1106_110698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_workers_count_l1106_110683

/-- Given a workshop with workers, prove that the total number of workers is 22 -/
theorem workshop_workers_count :
  ∀ W : ℕ,
  let T : ℕ := 7;
  let R : ℕ := W - T;
  let avg_all : ℚ := 850;
  let avg_tech : ℚ := 1000;
  let avg_rest : ℚ := 780;
  (W : ℚ) * avg_all = (T : ℚ) * avg_tech + (R : ℚ) * avg_rest →
  W = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_workers_count_l1106_110683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_l1106_110637

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x + 1|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ x^2 - x} = Set.Iic (-1) ∪ Set.Ici 0 := by sorry

-- Part II
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : 2 * m + n = 1) :
  (∀ x, f a x ≤ 1/m + 2/n) → -9 ≤ a ∧ a ≤ 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_l1106_110637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_half_l1106_110688

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then Real.log x / Real.log (1/2)
  else 2 + (4 ^ x)

-- State the theorem
theorem f_composition_half : f (f (1/2)) = -2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_half_l1106_110688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_protective_cover_cost_l1106_110693

/-- The total cost function for the protective cover -/
noncomputable def total_cost (V : ℝ) : ℝ := 1000 * V + 16000 / V - 500

/-- The theorem stating the properties of the protective cover cost -/
theorem protective_cover_cost (V : ℝ) (h : V > 0.5) :
  -- The total cost function
  total_cost V = 1000 * V + 16000 / V - 500 ∧
  -- The minimum total cost
  (∀ W, W > 0.5 → total_cost V ≤ total_cost W) → total_cost V = 7500 ∧
  -- The maximum base area when height is 2 meters and cost doesn't exceed 9500 yuan
  (let S := V / 2; S ≤ 4 ∧ total_cost V ≤ 9500 → S ≤ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_protective_cover_cost_l1106_110693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_time_l1106_110682

/-- Proves that a bus journey of 250 km, partly at 40 kmph for 148 km and partly at 60 kmph
    for the remaining distance, takes 5.4 hours. -/
theorem bus_journey_time (total_distance : ℝ) (speed1 speed2 : ℝ) (distance1 : ℝ) :
  total_distance = 250 →
  speed1 = 40 →
  speed2 = 60 →
  distance1 = 148 →
  (distance1 / speed1 + (total_distance - distance1) / speed2) = 5.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_time_l1106_110682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1106_110671

theorem problem_statement (a b : ℝ) 
  (h1 : (2 : ℝ)^a = 3) 
  (h2 : (8 : ℝ)^b = 1/6) : 
  (a + 3*b + 1)^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1106_110671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_three_l1106_110691

theorem log_sum_equals_three : Real.log 8 + 3 * Real.log 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_three_l1106_110691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_travel_time_l1106_110631

/-- Represents the motion of a pedestrian and a cyclist between two points. -/
structure Motion where
  L : ℝ  -- Distance between points A and B
  vp : ℝ  -- Speed of the pedestrian
  vc : ℝ  -- Speed of the cyclist

/-- The conditions of the problem. -/
def motion_conditions (m : Motion) : Prop :=
  -- After 1 hour, pedestrian is halfway between A and cyclist
  m.vp = (m.L - m.vc) / 2 ∧
  -- After 1.25 hours, they meet
  m.vp * 1.25 + m.vc * 1.25 = m.L ∧
  -- Speeds are positive
  m.vp > 0 ∧ m.vc > 0

/-- The theorem to be proved. -/
theorem pedestrian_travel_time (m : Motion) 
  (h : motion_conditions m) : 
  m.L / m.vp = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_travel_time_l1106_110631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reporter_earnings_per_article_l1106_110623

/-- Represents the payment structure and work conditions of a reporter --/
structure ReporterPayment where
  wordRate : ℚ
  wordsPerMinute : ℚ
  storiesPerPeriod : ℕ
  periodHours : ℕ
  totalEarningsPerHour : ℚ

/-- Calculates the earnings per article for a reporter --/
noncomputable def earningsPerArticle (rp : ReporterPayment) : ℚ :=
  let wordsPerHour := rp.wordsPerMinute * 60
  let wordEarningsPerHour := wordsPerHour * rp.wordRate
  let articleEarningsPerHour := rp.totalEarningsPerHour - wordEarningsPerHour
  let totalArticleEarnings := articleEarningsPerHour * rp.periodHours
  totalArticleEarnings / rp.storiesPerPeriod

/-- Theorem: Given the reporter's payment structure and work conditions,
    the earnings per article is $60 --/
theorem reporter_earnings_per_article :
  let rp : ReporterPayment := {
    wordRate := 1/10,
    wordsPerMinute := 10,
    storiesPerPeriod := 3,
    periodHours := 4,
    totalEarningsPerHour := 105
  }
  earningsPerArticle rp = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reporter_earnings_per_article_l1106_110623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_is_23_l1106_110665

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (4 * x - 23) / 3

-- Theorem statement
theorem h_fixed_point_is_23 :
  ∃! x : ℝ, h x = x ∧ ∀ y : ℝ, h (3 * y + 2) = 4 * y - 5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_is_23_l1106_110665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_l1106_110603

noncomputable def f (x : ℝ) := Real.pi * x + Real.log x / Real.log 2

theorem zero_point_exists : ∃ x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ), f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_l1106_110603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_surface_area_in_tetrahedron_l1106_110622

/-- The edge length of the regular tetrahedron -/
noncomputable def tetrahedron_edge : ℝ := 4

/-- The volume fraction of the tetrahedron filled with water -/
noncomputable def water_volume_fraction : ℝ := 7/8

/-- Predicate asserting that the ball touches all side faces and the water surface -/
def ball_touches_all_side_faces_and_water_surface : Prop := sorry

/-- Predicate asserting that the water volume fraction is correct -/
def water_volume_fraction_is_correct : Prop := sorry

/-- Function to calculate the surface area of the ball -/
noncomputable def surface_area_of_ball : ℝ := sorry

/-- Theorem stating the surface area of the ball -/
theorem ball_surface_area_in_tetrahedron 
  (ball_touches_faces : ball_touches_all_side_faces_and_water_surface)
  (water_level : water_volume_fraction_is_correct) :
  surface_area_of_ball = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_surface_area_in_tetrahedron_l1106_110622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l1106_110684

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (2 * x + Real.pi / 6))^2 - Real.sin (4 * x + Real.pi / 3)

theorem center_of_symmetry :
  ∃ (y : ℝ), (∀ (x : ℝ), f ((-7 * Real.pi / 48) + x) = f ((-7 * Real.pi / 48) - x)) ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l1106_110684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l1106_110663

theorem largest_lambda : 
  (∀ l : ℝ, l > (3/2) → ∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a^2 + b^2 + c^2 + d^2 < a*b + l*b*d + c*d) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + (3/2)*b*d + c*d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l1106_110663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_powers_theorem_l1106_110686

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 4)
noncomputable def y : ℂ := Complex.exp (-2 * Real.pi * Complex.I / 4)

theorem complex_powers_theorem :
  (x^8 + y^8 = 2) ∧ 
  (x^12 + y^12 = 2) ∧ 
  (x^6 + y^6 ≠ 2) ∧ 
  (x^10 + y^10 ≠ 2) ∧ 
  (x^14 + y^14 ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_powers_theorem_l1106_110686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_period_l1106_110604

/-- Compound interest calculation -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * ((1 + r / n) ^ (n * t) - 1)

/-- Rounding to nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem compound_interest_time_period :
  let P : ℝ := 10000
  let r : ℝ := 0.15
  let n : ℝ := 1
  let CI : ℝ := 3886.25
  let t : ℝ := Real.log (1 + CI / P) / (n * Real.log (1 + r / n))
  round_to_nearest t = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_period_l1106_110604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l1106_110639

/-- A rational function with specific properties -/
def rational_function (r s : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, s = λ x ↦ a*x^2 + b*x + c) ∧  -- s is quadratic
  (∃ m k : ℝ, r = λ x ↦ m*x + k) ∧  -- r is linear
  (s (-4) = 0) ∧ (s 1 = 0) ∧  -- roots of s
  (r 0 = 0) ∧ (s 0 ≠ 0) ∧  -- (0,0) is on the graph
  (r (-2) / s (-2) = -2)  -- (-2,-2) is on the graph

/-- The main theorem -/
theorem rational_function_value (r s : ℝ → ℝ) :
  rational_function r s → r 2 / s 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l1106_110639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increases_with_height_l1106_110670

-- Define the data points
def heights : List ℚ := [10, 20, 30, 40, 50, 60, 70]
def times : List ℚ := [423/100, 300/100, 245/100, 213/100, 189/100, 171/100, 159/100]

-- Define a function to calculate average speed
def avgSpeed (h : ℚ) (t : ℚ) : ℚ := h / t

-- Theorem statement
theorem speed_increases_with_height :
  ∀ i j, i < j → i < heights.length → j < heights.length →
    avgSpeed (heights[i]!) (times[i]!) < avgSpeed (heights[j]!) (times[j]!) := by
  intros i j hij hi hj
  sorry

#check speed_increases_with_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increases_with_height_l1106_110670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_solution_sequence_l1106_110669

/-- Represents the state of the boxes -/
def BoxState := Fin 6 → ℕ

/-- The initial state of the boxes -/
def initial_state : BoxState := λ _ ↦ 1

/-- The final desired state of the boxes -/
def final_state : BoxState :=
  λ i ↦ if i = 5 then 2010^(2010^2010) else 0

/-- Type 1 operation: Remove one coin from Bⱼ and add two to Bⱼ₊₁ -/
def type1 (j : Fin 5) (s : BoxState) : BoxState :=
  λ i ↦ if i = j then s j - 1
       else if i = j.succ then s i + 2
       else s i

/-- Type 2 operation: Remove one coin from Bₖ and swap Bₖ₊₁ and Bₖ₊₂ -/
def type2 (k : Fin 4) (s : BoxState) : BoxState :=
  λ i ↦ if i = k then s i - 1
       else if i = k.succ then s (k.succ.succ)
       else if i = k.succ.succ then s (k.succ)
       else s i

/-- A sequence of operations -/
inductive Operation
  | type1 (j : Fin 5)
  | type2 (k : Fin 4)

def apply_operation (op : Operation) (s : BoxState) : BoxState :=
  match op with
  | Operation.type1 j => type1 j s
  | Operation.type2 k => type2 k s

def apply_sequence (seq : List Operation) (s : BoxState) : BoxState :=
  seq.foldl (λ acc op ↦ apply_operation op acc) s

/-- Theorem stating the existence of a sequence transforming initial_state to final_state -/
theorem exists_solution_sequence :
  ∃ (seq : List Operation), apply_sequence seq initial_state = final_state := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_solution_sequence_l1106_110669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_number_l1106_110656

theorem magnitude_of_complex_number : 
  let z : ℂ := 1 - 2*Complex.I 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_number_l1106_110656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l1106_110633

-- Define the circle and its properties
def Circle : Type := Unit

-- Define the number of equal arcs
def num_arcs : ℕ := 18

-- Define the measure of a central angle for one arc
noncomputable def central_angle_measure (c : Circle) : ℝ := 360 / num_arcs

-- Define an inscribed angle
noncomputable def inscribed_angle (c : Circle) (n : ℕ) : ℝ := (n * central_angle_measure c) / 2

-- Define angles x and y
noncomputable def angle_x (c : Circle) : ℝ := inscribed_angle c 3
noncomputable def angle_y (c : Circle) : ℝ := inscribed_angle c 6

-- Theorem statement
theorem sum_of_angles (c : Circle) : angle_x c + angle_y c = 90 := by
  -- Expand definitions
  unfold angle_x angle_y inscribed_angle central_angle_measure
  -- Simplify the expression
  simp [num_arcs]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l1106_110633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_union_equals_original_triangle_result_is_60_l1106_110621

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The area of the union of a triangle and its 180° rotation about its centroid -/
noncomputable def area_union_with_rotation (a b c : ℝ) : ℝ :=
  triangle_area a b c

theorem area_union_equals_original_triangle (a b c : ℝ) 
  (ha : a = 8) (hb : b = 15) (hc : c = 17) :
  area_union_with_rotation a b c = triangle_area a b c := by
  rfl

-- To evaluate the result, we need to use a command that can handle noncomputable functions
theorem result_is_60 : 
  ∃ (x : ℝ), x > 59.9 ∧ x < 60.1 ∧ x = triangle_area 8 15 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_union_equals_original_triangle_result_is_60_l1106_110621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_intercept_problem_l1106_110699

theorem min_value_intercept_problem (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (1 / a + 2 / b = 1) → 4 * a^2 + b^2 ≥ 32 := by
  intro h_line
  -- Proof steps would go here
  sorry

#check min_value_intercept_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_intercept_problem_l1106_110699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_space_diagonals_l1106_110676

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  pentagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - (Q.pentagonal_faces * 5)

/-- Theorem stating that a convex polyhedron Q with specified properties has 315 space diagonals -/
theorem polyhedron_space_diagonals :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 40 ∧
    Q.triangular_faces = 30 ∧
    Q.pentagonal_faces = 10 ∧
    space_diagonals Q = 315 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_space_diagonals_l1106_110676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l1106_110664

noncomputable section

/-- Triangle ABC inscribed in a unit circle -/
structure InscribedTriangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side a
  b : ℝ  -- Side b
  c : ℝ  -- Side c
  h_sum : A + B + C = π
  h_positive : 0 < A ∧ 0 < B ∧ 0 < C
  h_sides : a = 2 * Real.sin A ∧ b = 2 * Real.sin B ∧ c = 2 * Real.sin C
  h_equation : 2 * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 2 * a - b) * Real.sin B

/-- The measure of angle C in the inscribed triangle -/
def angleC (t : InscribedTriangle) : ℝ := t.C

/-- The area of the inscribed triangle -/
def triangleArea (t : InscribedTriangle) : ℝ := 
  1/2 * t.a * t.b * Real.sin t.C

theorem inscribed_triangle_properties (t : InscribedTriangle) : 
  angleC t = π/4 ∧ 
  (∀ s : InscribedTriangle, triangleArea s ≤ (Real.sqrt 2 / 2 + 1 / 2)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l1106_110664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1106_110642

/-- Given a function f and a constant a, proves that f(-3) = 3 when f(3) = 5 -/
theorem function_symmetry (a : ℝ) :
  let f : ℝ → ℝ := λ x => a^2 * x^3 + a * Real.sin x + abs x + 1
  f 3 = 5 → f (-3) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1106_110642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorial_sum_problem_l1106_110674

theorem combinatorial_sum_problem :
  let sum := (1 : ℚ) / (3*2*1*16*15*14*13*12*11*10*9*8*7*6*5*4*3*2*1) + 
             1 / (4*3*2*1*15*14*13*12*11*10*9*8*7*6*5*4*3*2*1) + 
             1 / (5*4*3*2*1*14*13*12*11*10*9*8*7*6*5*4*3*2*1) + 
             1 / (6*5*4*3*2*1*13*12*11*10*9*8*7*6*5*4*3*2*1) + 
             1 / (7*6*5*4*3*2*1*12*11*10*9*8*7*6*5*4*3*2*1) + 
             1 / (8*7*6*5*4*3*2*1*11*10*9*8*7*6*5*4*3*2*1) + 
             1 / (9*8*7*6*5*4*3*2*1*10*9*8*7*6*5*4*3*2*1)
  let M := (sum * (1*18*17*16*15*14*13*12*11*10*9*8*7*6*5*4*3*2*1)).num
  M = 13787 ∧ ⌊M / 100⌋ = 137 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorial_sum_problem_l1106_110674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_l1106_110685

noncomputable def data : List ℝ := [5, 7, 7, 8, 10, 11]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem standard_deviation_of_data : standardDeviation data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_l1106_110685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_estimate_correct_l1106_110609

/-- Represents the foot length and height data for a group of students. -/
structure StudentData where
  n : ℕ -- number of students
  x_sum : ℚ -- sum of foot lengths
  y_sum : ℚ -- sum of heights
  b : ℚ -- slope of the regression line

/-- Calculates the estimated height for a given foot length using linear regression. -/
def estimateHeight (data : StudentData) (foot_length : ℚ) : ℚ :=
  let x_mean : ℚ := data.x_sum / data.n
  let y_mean : ℚ := data.y_sum / data.n
  let a : ℚ := y_mean - data.b * x_mean
  data.b * foot_length + a

/-- Theorem stating that the estimated height for a foot length of 24 cm is 166 cm. -/
theorem height_estimate_correct (data : StudentData) 
    (h_n : data.n = 10)
    (h_x_sum : data.x_sum = 225)
    (h_y_sum : data.y_sum = 1600)
    (h_b : data.b = 4) :
  estimateHeight data 24 = 166 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_estimate_correct_l1106_110609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_lap_time_l1106_110678

/-- Represents the time in minutes for a given number of hours and minutes -/
noncomputable def total_time_in_minutes (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours * 60 + minutes : ℝ)

/-- Represents the time per lap given total time and number of laps -/
noncomputable def time_per_lap (total_time : ℝ) (num_laps : ℕ) : ℝ :=
  total_time / num_laps

theorem park_lap_time :
  let total_time := total_time_in_minutes 1 36
  let num_laps := 5
  time_per_lap total_time num_laps = 19.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_lap_time_l1106_110678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1106_110611

noncomputable def f (x : ℝ) := x^2 - 1

noncomputable def g (x : ℝ) := -Real.sqrt (x + 1)

theorem inverse_function_proof (x : ℝ) (h : x < -1) :
  g (f x) = x ∧ f (g x) = x ∧ ∀ y, y < -1 → g y < -1 := by
  sorry

#check inverse_function_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1106_110611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_unique_T_l1106_110658

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

noncomputable def S (a d : ℝ) (n : ℕ) : ℝ := 
  n * (2 * a + (n - 1) * d) / 2

noncomputable def T (a d : ℝ) (n : ℕ) : ℝ := 
  n * (n + 1) * (3 * a + (n - 1) * d) / 6

theorem smallest_n_for_unique_T (a d : ℝ) : 
  S a d 2023 = 4056 * a + 4053 * d → 
  (∀ m : ℕ, m < 3037 → ∃ (a' d' : ℝ), a' ≠ a ∨ d' ≠ d ∧ S a' d' 2023 = 4056 * a' + 4053 * d' ∧ T a' d' m = T a d m) ∧
  (∀ (a' d' : ℝ), S a' d' 2023 = 4056 * a' + 4053 * d' → T a' d' 3037 = T a d 3037) := by
  sorry

#check smallest_n_for_unique_T

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_unique_T_l1106_110658
