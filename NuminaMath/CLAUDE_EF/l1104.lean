import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_degrees_to_radians_l1104_110497

-- Define the conversion factor from degrees to radians
noncomputable def degToRad : ℝ := Real.pi / 180

-- Theorem statement
theorem sixty_degrees_to_radians :
  60 * degToRad = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_degrees_to_radians_l1104_110497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acute_points_2D_max_acute_points_3D_l1104_110435

-- Define a type for points in 2D and 3D space
structure Point2D where
  x : ℝ
  y : ℝ

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to check if an angle is acute
def isAcuteAngle (a b c : ℝ × ℝ) : Prop := sorry

-- Define a function to check if all angles in a set of points are acute
def allAnglesAcute2D (points : List Point2D) : Prop :=
  ∀ i j k, i ∈ points → j ∈ points → k ∈ points → i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    isAcuteAngle (i.x, i.y) (j.x, j.y) (k.x, k.y)

def allAnglesAcute3D (points : List Point3D) : Prop :=
  ∀ i j k, i ∈ points → j ∈ points → k ∈ points → i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    isAcuteAngle (i.x, i.y) (j.x, j.y) (k.x, k.y)

-- Theorem for 2D case
theorem max_acute_points_2D :
  ∀ (points : List Point2D), allAnglesAcute2D points → points.length ≤ 3 :=
by sorry

-- Theorem for 3D case
theorem max_acute_points_3D :
  ∀ (points : List Point3D), allAnglesAcute3D points → points.length ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acute_points_2D_max_acute_points_3D_l1104_110435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_arithmetic_sequence_in_S_l1104_110418

/-- The set of all positive integers from 1 through 1000 that are not perfect squares -/
def S : Set Nat :=
  {n : Nat | 1 ≤ n ∧ n ≤ 1000 ∧ ¬∃ m : Nat, m * m = n}

/-- A function that checks if a sequence is arithmetic -/
def isArithmeticSequence (seq : List Nat) : Prop :=
  seq.length > 1 ∧ ∃ d : Int, ∀ i : Nat, i + 1 < seq.length →
    (seq.get ⟨i + 1, by sorry⟩ : Int) - (seq.get ⟨i, by sorry⟩ : Int) = d

/-- The theorem stating the length of the longest non-constant arithmetic sequence in S -/
theorem longest_arithmetic_sequence_in_S :
  ∃ seq : List Nat, (∀ n ∈ seq, n ∈ S) ∧
    isArithmeticSequence seq ∧
    seq.length = 333 ∧
    (∀ seq' : List Nat, (∀ n ∈ seq', n ∈ S) →
      isArithmeticSequence seq' →
      seq'.length ≤ seq.length) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_arithmetic_sequence_in_S_l1104_110418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_range_l1104_110409

-- Define the circle and points
def circle_equation (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 9
def origin : ℝ × ℝ := (0, 0)
def point_q : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem circle_point_range (a : ℝ) :
  (a > 5) →
  (∃ (m : ℝ × ℝ), circle_equation a m.1 m.2 ∧ distance origin m = 2 * distance m point_q) →
  (5 < a ∧ a ≤ 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_range_l1104_110409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valentines_theorem_l1104_110466

/-- Represents the number of Valentines Mrs. Franklin had initially -/
noncomputable def initial_valentines : ℝ := 58.5

/-- Represents the number of Valentines Mrs. Franklin had left -/
noncomputable def remaining_valentines : ℝ := 16.25

/-- Represents the number of Valentines given to colleagues -/
noncomputable def valentines_to_colleagues : ℝ := (initial_valentines - remaining_valentines) / 3

/-- Represents the number of Valentines given to students -/
noncomputable def valentines_to_students : ℝ := 2 * valentines_to_colleagues

/-- The total number of Valentines given away -/
noncomputable def total_given_away : ℝ := valentines_to_colleagues + valentines_to_students

theorem valentines_theorem : total_given_away = 42.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valentines_theorem_l1104_110466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_angle_l1104_110465

/-- Line passing through a point with inclination angle α -/
structure Line (α : ℝ) where
  point : ℝ × ℝ

/-- Curve defined by y² = 2x -/
def Curve : Set (ℝ × ℝ) :=
  {p | (p.2)^2 = 2 * p.1}

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_curve_intersection_angle (α : ℝ) :
  let l : Line α := ⟨⟨-2, -4⟩⟩
  ∃ A B : ℝ × ℝ,
    A ∈ Curve ∧ B ∈ Curve ∧
    A ≠ B ∧
    (∃ t1 t2 : ℝ, 
      A = ⟨-2 + t1 * Real.cos α, -4 + t1 * Real.sin α⟩ ∧
      B = ⟨-2 + t2 * Real.cos α, -4 + t2 * Real.sin α⟩) ∧
    distance ⟨-2, -4⟩ A * distance ⟨-2, -4⟩ B = 40 →
  α = π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_angle_l1104_110465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_numbers_l1104_110460

-- Define the complex numbers
def z₁ : ℂ := Complex.mk 1 3
def z₂ : ℂ := Complex.mk 2 (-4)
def z₃ : ℂ := Complex.mk 4 2

-- State the theorem
theorem sum_of_complex_numbers :
  z₁ + z₂ + z₃ = Complex.mk 7 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_numbers_l1104_110460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_is_four_l1104_110440

/-- The number of intersection points between two polar curves -/
noncomputable def intersection_count (f g : ℝ → ℝ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- The first curve: ρ = 11 cos(θ/2) -/
noncomputable def curve1 (θ : ℝ) : ℝ := 11 * Real.cos (θ / 2)

/-- The second curve: ρ = cos(θ/2) -/
noncomputable def curve2 (θ : ℝ) : ℝ := Real.cos (θ / 2)

/-- Theorem stating that the two curves intersect at exactly 4 points -/
theorem intersection_count_is_four :
  intersection_count curve1 curve2 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_is_four_l1104_110440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_square_areas_is_correct_l1104_110479

/-- The sum of areas of all squares formed by repeatedly inscribing squares 
    within quarter-circle arcs in a unit square -/
noncomputable def sumOfSquareAreas : ℝ := (1 + Real.sqrt 3) / 2

/-- The side length of the nth square in the sequence -/
noncomputable def nthSquareSideLength (n : ℕ) : ℝ := ((2 - Real.sqrt 3) ^ n) / 2

/-- The area of the nth square in the sequence -/
noncomputable def nthSquareArea (n : ℕ) : ℝ := (nthSquareSideLength n) ^ 2

theorem sum_of_square_areas_is_correct :
  ∑' n, nthSquareArea n = sumOfSquareAreas := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_square_areas_is_correct_l1104_110479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_line_equation_l1104_110487

/-- Circle with equation x^2 + (y-1)^2 = 4 -/
def myCircle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

/-- Point P -/
noncomputable def P : ℝ × ℝ := (1, 1/2)

/-- Line l passing through point P -/
def line_l (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b ∧ P.2 = m * P.1 + b

/-- Center of the circle -/
noncomputable def C : ℝ × ℝ := (0, 1)

/-- Angle between three points -/
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem: When ∠ACB is minimum, the equation of line l is 4x - 2y - 3 = 0 -/
theorem min_angle_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, line_l m b x y ↔ 4*x - 2*y - 3 = 0) ∧
    (∀ A B : ℝ × ℝ, 
      myCircle A.1 A.2 → myCircle B.1 B.2 → 
      line_l m b A.1 A.2 → line_l m b B.1 B.2 →
      ∀ m' b' : ℝ, (∀ x y : ℝ, line_l m' b' x y → 
        ∃ A' B' : ℝ × ℝ, myCircle A'.1 A'.2 ∧ myCircle B'.1 B'.2 ∧ 
        line_l m' b' A'.1 A'.2 ∧ line_l m' b' B'.1 B'.2) →
      angle C A B ≤ angle C A' B') :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_line_equation_l1104_110487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_points_l1104_110423

-- Define the points
noncomputable def A : ℝ × ℝ := (-2, 1)
noncomputable def B : ℝ × ℝ := (9, 3)
noncomputable def C : ℝ × ℝ := (1, 7)

-- Define the center of the circle
noncomputable def center : ℝ × ℝ := (7/2, 2)

-- Define the radius squared
noncomputable def radius_squared : ℝ := 125/4

-- Theorem statement
theorem circle_equation_through_points :
  (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius_squared ∧
  (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius_squared ∧
  (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius_squared :=
by
  sorry

#check circle_equation_through_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_through_points_l1104_110423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_terms_120_l1104_110437

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (a₁ + arithmetic_sequence a₁ d (n - 1)) / 2

theorem sum_first_10_terms_120 :
  sum_arithmetic_sequence 3 2 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_terms_120_l1104_110437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symposium_partition_l1104_110455

structure Symposium where
  delegates : Finset Nat
  knows : Nat → Nat → Prop

  knows_symmetric : ∀ a b, knows a b → knows b a
  knows_irreflexive : ∀ a, ¬knows a a
  knows_someone : ∀ a, a ∈ delegates → ∃ b, b ∈ delegates ∧ a ≠ b ∧ knows a b
  not_all_know : ∀ a b, a ∈ delegates → b ∈ delegates → a ≠ b → 
    ∃ c, c ∈ delegates ∧ c ≠ a ∧ c ≠ b ∧ (¬knows c a ∨ ¬knows c b)

theorem symposium_partition (s : Symposium) :
  ∃ (g₁ g₂ g₃ : Finset Nat),
    g₁ ∪ g₂ ∪ g₃ = s.delegates ∧
    g₁ ∩ g₂ = ∅ ∧ g₁ ∩ g₃ = ∅ ∧ g₂ ∩ g₃ = ∅ ∧
    (∀ a, a ∈ g₁ → ∃ b, b ∈ g₁ ∧ a ≠ b ∧ s.knows a b) ∧
    (∀ a, a ∈ g₂ → ∃ b, b ∈ g₂ ∧ a ≠ b ∧ s.knows a b) ∧
    (∀ a, a ∈ g₃ → ∃ b, b ∈ g₃ ∧ a ≠ b ∧ s.knows a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symposium_partition_l1104_110455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1104_110450

-- Define the sequence a_n
noncomputable def a (a₀ : ℝ) : ℕ → ℝ
  | 0 => a₀
  | n + 1 => 1 / (2 - a a₀ n)

-- Theorem statement
theorem sequence_formula (a₀ : ℝ) :
  ∀ n : ℕ, n > 0 → a a₀ n = ((n - 1) - (n - 2) * a₀) / (n - (n - 1) * a₀) :=
by
  intro n hn
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1104_110450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_r_2008_l1104_110458

-- Define the polynomial p(x)
def p (x : ℤ) : ℤ := (List.range 2009).foldl (λ acc i => acc + x^i) 0

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^4 + x^3 + 2*x^2 + x + 1

-- Define r(x) as the remainder when p(x) is divided by the divisor
def r (x : ℤ) : ℤ := p x % divisor x

-- State the theorem
theorem remainder_of_r_2008 : (Int.natAbs (r 2008)) % 1000 = 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_r_2008_l1104_110458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_rotation_l1104_110428

/-- Triangle in a 2D plane -/
structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V

/-- Circumcircle of a triangle -/
noncomputable def circumcircle {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) : Set V :=
  sorry

/-- Simson line of a point with respect to a triangle -/
noncomputable def simson_line {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (t : Triangle V) (P : V) : Set V :=
  sorry

/-- Central angle subtended by an arc on a circle -/
noncomputable def central_angle {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (center P1 P2 : V) : ℝ :=
  sorry

/-- Angle between two lines -/
noncomputable def angle_between_lines {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (l1 l2 : Set V) : ℝ :=
  sorry

/-- Main theorem: The angle between Simson lines of two points on the circumcircle
    is half the central angle subtended by the arc between those points -/
theorem simson_line_rotation
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (t : Triangle V) (P1 P2 : V)
  (h1 : P1 ∈ circumcircle t) (h2 : P2 ∈ circumcircle t) :
  angle_between_lines (simson_line t P1) (simson_line t P2) =
  (1/2) * central_angle t.B P1 P2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_rotation_l1104_110428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_divisible_by_101_l1104_110484

def b : ℕ → ℕ
  | 0 => 20  -- Add this case to cover all natural numbers
  | 1 => 20  -- Add this case to cover all natural numbers
  | 2 => 20  -- Add this case to cover all natural numbers
  | 3 => 20  -- Add this case to cover all natural numbers
  | 4 => 20  -- Add this case to cover all natural numbers
  | 5 => 20  -- Add this case to cover all natural numbers
  | 6 => 20  -- Add this case to cover all natural numbers
  | 7 => 20  -- Add this case to cover all natural numbers
  | 8 => 20
  | n + 9 => 100 * b (n + 8) + 2 * (n + 9)

theorem least_n_divisible_by_101 :
  ∀ k : ℕ, k > 8 ∧ k < 15 → ¬(101 ∣ b k) ∧ (101 ∣ b 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_divisible_by_101_l1104_110484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_width_to_perimeter_ratio_l1104_110482

/-- The ratio of width to perimeter for a rectangular room -/
def width_to_perimeter_ratio (length width : ℚ) : ℚ :=
  width / (2 * (length + width))

/-- Theorem: For a room with length 22 feet and width 13 feet, 
    the ratio of width to perimeter is 13:70 -/
theorem room_width_to_perimeter_ratio :
  width_to_perimeter_ratio 22 13 = 13 / 70 := by
  unfold width_to_perimeter_ratio
  norm_num

#eval width_to_perimeter_ratio 22 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_width_to_perimeter_ratio_l1104_110482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1104_110472

-- Define the function f
noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 3^(x - b)

-- State the theorem
theorem range_of_f :
  ∃ b : ℝ, ∀ y : ℝ, (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ f x b = y) ↔ 1 ≤ y ∧ y ≤ 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1104_110472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1104_110401

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n - 2) / (x n + 8)

theorem sequence_convergence :
  ∃ m : ℕ, m ∈ Set.Icc 35 60 ∧
    x m ≤ 3 + 1 / 2^10 ∧
    ∀ k : ℕ, k > 0 ∧ k < m → x k > 3 + 1 / 2^10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1104_110401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_rectangle_circle_l1104_110483

theorem common_area_rectangle_circle : 
  ∀ (rectangle_length rectangle_width circle_radius : ℝ),
    rectangle_length = 10 →
    rectangle_width = 4 →
    circle_radius = 5 →
    (π * circle_radius^2 / 2) + 
    4 * Real.sqrt (circle_radius^2 - (rectangle_width / 2)^2) =
    (25 * π / 2) + 4 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_rectangle_circle_l1104_110483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_isosceles_right_triangle_on_hyperbola_l1104_110486

noncomputable section

/-- The hyperbola equation xy = 1 -/
def OnHyperbola (p : ℝ × ℝ) : Prop := p.1 * p.2 = 1

/-- Check if three points form an isosceles right triangle -/
def IsIsoscelesRightTriangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

/-- Calculate the area of a triangle given three points -/
def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem min_area_isosceles_right_triangle_on_hyperbola :
  ∀ A B C : ℝ × ℝ,
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  OnHyperbola A ∧ OnHyperbola B ∧ OnHyperbola C →
  IsIsoscelesRightTriangle A B C →
  TriangleArea A B C ≥ 3 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_isosceles_right_triangle_on_hyperbola_l1104_110486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_cubic_coefficient_positive_l1104_110412

open Set
open Function

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 5

theorem increasing_cubic_coefficient_positive (a : ℝ) :
  StrictMono (f a) → a ∈ Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_cubic_coefficient_positive_l1104_110412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_characteristics_imply_pyramid_l1104_110406

/-- Characteristics of an investment scheme -/
structure InvestmentScheme where
  highReturns : Bool
  lackOfInfo : Bool
  aggressiveAds : Bool

/-- Predicate to check if a scheme is a financial pyramid -/
def isFinancialPyramid (scheme : InvestmentScheme) : Prop :=
  scheme.highReturns ∧ scheme.lackOfInfo ∧ scheme.aggressiveAds

/-- Theorem stating that if all characteristics are present, it's a financial pyramid -/
theorem all_characteristics_imply_pyramid (scheme : InvestmentScheme) :
    scheme.highReturns ∧ scheme.lackOfInfo ∧ scheme.aggressiveAds →
    isFinancialPyramid scheme := by
  intro h
  exact h

#check all_characteristics_imply_pyramid

/-- Example of a scheme with all characteristics -/
def suspiciousScheme : InvestmentScheme :=
  { highReturns := true
    lackOfInfo := true
    aggressiveAds := true }

/-- Proof that the suspicious scheme is a financial pyramid -/
example : isFinancialPyramid suspiciousScheme := by
  apply all_characteristics_imply_pyramid
  constructor
  · exact rfl
  constructor
  · exact rfl
  · exact rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_characteristics_imply_pyramid_l1104_110406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l1104_110433

def family_size : ℕ := 6
def front_row_size : ℕ := 3
def back_row_size : ℕ := 3
def num_parents : ℕ := 2

def seating_arrangements (total : ℕ) (front : ℕ) (back : ℕ) (parents : ℕ) : ℕ :=
  let driver_choices := parents
  let front_passenger_choices := Nat.choose (total - 1) (front - 1)
  let front_passenger_arrangements := Nat.factorial (front - 1)
  let back_arrangements := Nat.factorial back
  driver_choices * front_passenger_choices * front_passenger_arrangements * back_arrangements

theorem seating_theorem : 
  seating_arrangements family_size front_row_size back_row_size num_parents = 240 := by
  sorry

#eval seating_arrangements family_size front_row_size back_row_size num_parents

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l1104_110433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l1104_110441

theorem equation_roots : 
  let x₁ := ((9 + 3 * Real.sqrt 5) / 6) ^ 2
  let x₂ := ((9 - 3 * Real.sqrt 5) / 6) ^ 2
  ∀ x : ℝ, x > 0 → (3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) = 9) ↔ (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l1104_110441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1104_110485

def is_valid_assignment (r s t u : ℕ) : Prop :=
  Finset.toSet {r, s, t, u} = Finset.toSet {2, 3, 4, 5}

def expression_value (r s t u : ℕ) : ℕ :=
  r * s + u * r + t * r

theorem max_expression_value :
  ∃ (r s t u : ℕ), is_valid_assignment r s t u ∧
    (∀ (r' s' t' u' : ℕ), is_valid_assignment r' s' t' u' →
      expression_value r s t u ≥ expression_value r' s' t' u') ∧
    expression_value r s t u = 45 := by
  sorry

#check max_expression_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1104_110485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_attraction_theorem_l1104_110449

noncomputable def p (x : ℕ) : ℝ := (1/2) * x * (x + 1) * (39 - 2*x)

noncomputable def q (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 6 then 35 - 2*x
  else if 7 ≤ x ∧ x ≤ 12 then 160 / x
  else 0

noncomputable def f (x : ℕ) : ℝ := -3 * x^2 + 40 * x

noncomputable def g (x : ℕ) : ℝ := f x * q x

theorem tourist_attraction_theorem :
  (∀ x : ℕ, 1 ≤ x → x ≤ 12 → f x = p x - p (x-1)) ∧
  (∃ max_month : ℕ, max_month = 5 ∧
    ∀ x : ℕ, 1 ≤ x → x ≤ 12 → g x ≤ g max_month) ∧
  g 5 = 31250000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_attraction_theorem_l1104_110449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_calculation_l1104_110481

/-- Calculates the width of a wall given brick dimensions and wall specifications -/
theorem wall_width_calculation (brick_length brick_width brick_height : ℚ)
                               (wall_length wall_height : ℚ)
                               (num_bricks : ℕ) :
  brick_length = 20/100 →
  brick_width = 10/100 →
  brick_height = 8/100 →
  wall_length = 10 →
  wall_height = 8 →
  num_bricks = 12250 →
  ∃ (wall_width : ℚ), 
    wall_width = (num_bricks : ℚ) * brick_length * brick_width * brick_height / (wall_length * wall_height) ∧
    abs (wall_width - 245/1000) < 1/1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_calculation_l1104_110481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1104_110421

-- Define the set of integers satisfying the inequality
def S : Set ℤ := {x : ℤ | (x - 2)^2 ≤ 4}

-- Theorem stating that the cardinality of S is 5
theorem inequality_solution : Finset.card (Finset.filter (fun x => (x - 2)^2 ≤ 4) (Finset.range 5)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1104_110421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l1104_110411

theorem equilateral_triangle_area_decrease :
  ∀ (s : ℝ),
  s > 0 →
  s^2 * Real.sqrt 3 / 4 = 64 * Real.sqrt 3 →
  ((s - 4)^2 * Real.sqrt 3 / 4 - 64 * Real.sqrt 3) = -28 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l1104_110411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l1104_110442

/-- A quadratic function with vertex (-1, 2) passing through (1, -3) -/
noncomputable def f (x : ℝ) : ℝ := -5/4 * x^2 - 5/2 * x + 3/4

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, 2)

/-- A point that the quadratic function f passes through -/
def point : ℝ × ℝ := (1, -3)

theorem quadratic_properties :
  (f vertex.1 = vertex.2 ∧ 
   (∀ x : ℝ, f x ≥ f vertex.1)) ∧
  f point.1 = point.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l1104_110442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_feed_purchase_full_price_total_l1104_110430

/-- Represents the feed purchase scenario for a farmer --/
structure FeedPurchase where
  total_spent : ℝ
  chicken_feed_percent : ℝ
  chicken_feed_discount : ℝ
  goat_feed_percent : ℝ
  goat_feed_discount : ℝ

/-- Calculates the full price amount for all feed --/
noncomputable def full_price_total (fp : FeedPurchase) : ℝ :=
  let chicken_full_price := (fp.chicken_feed_percent * fp.total_spent) / (1 - fp.chicken_feed_discount)
  let goat_full_price := (fp.goat_feed_percent * fp.total_spent) / (1 - fp.goat_feed_discount)
  let cow_horse_price := fp.total_spent * (1 - fp.chicken_feed_percent - fp.goat_feed_percent)
  chicken_full_price + goat_full_price + cow_horse_price

/-- Theorem stating that the full price total for the given scenario is approximately $146.67 --/
theorem feed_purchase_full_price_total :
  let fp : FeedPurchase := {
    total_spent := 120,
    chicken_feed_percent := 0.3,
    chicken_feed_discount := 0.4,
    goat_feed_percent := 0.2,
    goat_feed_discount := 0.1
  }
  ∃ ε > 0, |full_price_total fp - 146.67| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_feed_purchase_full_price_total_l1104_110430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odds_against_C_l1104_110478

/-- Represents the odds against a runner winning -/
structure Odds where
  against : ℚ
  in_favor : ℚ

/-- Calculates the probability of winning given the odds against -/
def probability_from_odds (o : Odds) : ℚ :=
  o.in_favor / (o.against + o.in_favor)

/-- Represents a three-runner race -/
structure ThreeRunnerRace where
  odds_A : Odds
  odds_B : Odds
  no_ties : Unit  -- Representing the condition that ties are not possible

theorem odds_against_C (race : ThreeRunnerRace) 
  (h_A : race.odds_A = { against := 4, in_favor := 1 })
  (h_B : race.odds_B = { against := 1, in_favor := 2 }) :
  ∃ (odds_C : Odds), odds_C.against = 13 ∧ odds_C.in_favor = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odds_against_C_l1104_110478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_collinearity_and_trajectory_l1104_110415

/-- Given points A, B on y²=x (not origin), and C(1,0), if OA · OB = 0, then A, C, B are collinear,
    and for Q such that AQ = λQB and OQ · AB = 0, Q's trajectory is (x-1/2)² + y² = 1/4 (x≠0) -/
theorem parabola_points_collinearity_and_trajectory :
  ∀ (A B : ℝ × ℝ),
  let O : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (1, 0)
  -- A and B are on the parabola y²=x
  (A.2)^2 = A.1 ∧ (B.2)^2 = B.1 →
  -- A and B are not the origin
  A ≠ O ∧ B ≠ O →
  -- A and B are distinct
  A ≠ B →
  -- OA · OB = 0
  (A.1 * B.1 + A.2 * B.2 = 0) →
  -- A, C, B are collinear
  ∃ (t : ℝ), C = t • A + (1 - t) • B ∧
  -- For any Q such that AQ = λQB and OQ · AB = 0
  ∀ (Q : ℝ × ℝ) (l : ℝ),
  (∃ (μ : ℝ), Q = μ • A + (1 - μ) • B) →
  (Q.1 * (B.1 - A.1) + Q.2 * (B.2 - A.2) = 0) →
  -- Q's trajectory is (x-1/2)² + y² = 1/4 (x≠0)
  (Q.1 ≠ 0 → (Q.1 - 1/2)^2 + (Q.2)^2 = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_collinearity_and_trajectory_l1104_110415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_a_range_l1104_110475

noncomputable def f (x : ℝ) : ℝ := Real.exp x / (x - 1)

theorem f_monotonicity_and_a_range :
  (∀ x y, x > 2 ∧ y > 2 ∧ x < y → f x < f y) ∧
  (∀ x y, ((x < 1 ∧ y < 1) ∨ (1 < x ∧ x < 2 ∧ 1 < y ∧ y < 2)) ∧ x < y → f x > f y) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ≥ 2 → (Real.exp x * (x - 2)) / ((x - 1)^2) ≥ a * (Real.exp x / (x - 1))) → a ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_a_range_l1104_110475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_derivative_l1104_110407

/-- Given a differentiable function f, prove that its derivative at x = 1 is equal to e,
    given that the tangent line to f at (1, f(1)) is y = ex - e. -/
theorem tangent_line_derivative (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (fun x => deriv f 1 * (x - 1) + f 1) = (fun x ↦ ℯ * x - ℯ) → 
  deriv f 1 = ℯ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_derivative_l1104_110407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_equals_seven_l1104_110464

-- Define the expression S as noncomputable
noncomputable def S : ℝ := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
             1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
             1 / (Real.sqrt 12 - 3)

-- Theorem statement
theorem S_equals_seven : S = 7 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_equals_seven_l1104_110464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_production_equation_l1104_110474

/-- Represents the monthly decrease rate of mask production -/
def x : ℝ := sorry

/-- The initial production in June (normalized to 1) -/
def initial_production : ℝ := 1

/-- The production in August (normalized to 0.64) -/
def august_production : ℝ := 0.64

/-- The number of months between June and August -/
def months_elapsed : ℕ := 2

/-- Theorem stating that the equation 100(1-x)² = 64 accurately represents the mask production situation -/
theorem mask_production_equation :
  initial_production * (1 - x) ^ months_elapsed = august_production :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_production_equation_l1104_110474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l1104_110494

/-- A unit cube with an inscribed regular octahedron -/
structure CubeWithOctahedron where
  /-- Vertices of the unit cube -/
  cube_vertices : Fin 8 → ℝ × ℝ × ℝ
  /-- Vertices of the inscribed regular octahedron -/
  octahedron_vertices : Fin 6 → ℝ × ℝ × ℝ
  /-- The cube is a unit cube -/
  is_unit_cube : ∀ i j, i ≠ j → ‖cube_vertices i - cube_vertices j‖ = 1 ∨ ‖cube_vertices i - cube_vertices j‖ = Real.sqrt 3
  /-- The octahedron vertices lie on the cube's main diagonals -/
  vertices_on_diagonals : ∀ v, ∃ i j t, octahedron_vertices v = cube_vertices i + t • (cube_vertices j - cube_vertices i) ∧ 0 < t ∧ t < 1
  /-- The octahedron is regular -/
  is_regular_octahedron : ∀ i j, i ≠ j → ‖octahedron_vertices i - octahedron_vertices j‖ = ‖octahedron_vertices 0 - octahedron_vertices 1‖

/-- The side length of the inscribed regular octahedron is √2/2 -/
theorem octahedron_side_length (c : CubeWithOctahedron) :
  ‖c.octahedron_vertices 0 - c.octahedron_vertices 1‖ = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l1104_110494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_ways_to_remove_digits_l1104_110446

/-- The original number that is repeated -/
def original_number : Nat := 2458710411

/-- The number of times the original number is repeated -/
def repetitions : Nat := 98

/-- The number of digits to be removed -/
def digits_to_remove : Nat := 4

/-- The total number of digits in the initial number -/
def total_digits : Nat := (Nat.digits 10 original_number).length * repetitions

/-- A function that checks if a number is divisible by 6 -/
def is_divisible_by_six (n : Nat) : Prop := n % 6 = 0

/-- A function to represent the resulting number after removal of digits -/
def resulting_number (removed_digits : Finset (Fin total_digits)) : Nat :=
  sorry -- Implementation of this function would go here

/-- The main theorem stating the number of ways to remove digits -/
theorem number_of_ways_to_remove_digits : 
  ∃ (ways : Nat), ways = 97096 ∧ 
  (∃ (removed_digits : Finset (Fin total_digits)), 
    removed_digits.card = digits_to_remove ∧
    is_divisible_by_six (resulting_number removed_digits)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_ways_to_remove_digits_l1104_110446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_mathematician_with_many_descendants_estimate_largest_number_of_descendants_l1104_110400

-- Define the type for mathematicians
def Mathematician : Type := ℕ

-- Define the advisor relation
def is_advisor : Mathematician → Mathematician → Prop := sorry

-- Define the descendant relation
def is_descendant (M M' : Mathematician) : Prop := 
  ∃ (k : ℕ) (sequence : ℕ → Mathematician), 
    sequence 0 = M ∧ 
    sequence k = M' ∧ 
    ∀ i < k, is_advisor (sequence i) (sequence (i + 1))

-- Define the current year
def current_year : ℕ := 2023

-- Define the starting year of records
def start_year : ℕ := 1300

-- Theorem stating the existence of a mathematician with at least 82000 descendants
theorem exists_mathematician_with_many_descendants : 
  ∃ (M : Mathematician), 
    (∃ (descendants : Finset Mathematician), 
      (∀ M' ∈ descendants, is_descendant M M') ∧ 
      descendants.card ≥ 82000) := by sorry

-- Helper theorem to connect the problem statement to the main theorem
theorem estimate_largest_number_of_descendants :
  ∃ (M : Mathematician), 
    (∃ (descendants : Finset Mathematician), 
      (∀ M' ∈ descendants, is_descendant M M') ∧ 
      descendants.card ≥ 82000 ∧ 
      descendants.card ≤ 82620) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_mathematician_with_many_descendants_estimate_largest_number_of_descendants_l1104_110400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_value_l1104_110469

theorem max_integer_value (a b c d : ℕ+) : 
  (a + b + c + d : ℚ) / 4 = 50 →
  (a + b : ℚ) / 2 = 35 →
  c ≤ 129 ∧ d ≤ 129 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_value_l1104_110469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_variety_price_l1104_110447

/-- Represents the price of tea in Rupees per kilogram -/
@[ext] structure TeaPrice where
  price : ℚ

/-- Represents a mixture of teas -/
structure TeaMixture where
  variety1 : TeaPrice
  variety2 : TeaPrice
  variety3 : TeaPrice
  ratio1 : ℚ
  ratio2 : ℚ
  ratio3 : ℚ
  mixedPrice : TeaPrice

/-- Calculate the weighted average price of a tea mixture -/
def weightedAveragePrice (m : TeaMixture) : ℚ :=
  (m.variety1.price * m.ratio1 + m.variety2.price * m.ratio2 + m.variety3.price * m.ratio3) /
  (m.ratio1 + m.ratio2 + m.ratio3)

theorem second_variety_price (m : TeaMixture) 
    (h1 : m.variety1.price = 126)
    (h2 : m.variety3.price = 175.5)
    (h3 : m.ratio1 = 1)
    (h4 : m.ratio2 = 1)
    (h5 : m.ratio3 = 2)
    (h6 : m.mixedPrice.price = 153)
    (h7 : weightedAveragePrice m = m.mixedPrice.price) :
    m.variety2.price = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_variety_price_l1104_110447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_l1104_110438

/-- Given a complex number z such that arg(z + 3) = 3π/4,
    the maximum value of u = 1 / (|z + 6| + |z - 3i|) is √5/15,
    and this maximum is achieved when z = -4 + i. -/
theorem max_value_complex (z : ℂ) (h : Complex.arg (z + 3) = 3 * Real.pi / 4) :
  (∀ w : ℂ, Complex.arg (w + 3) = 3 * Real.pi / 4 →
    1 / (Complex.abs (z + 6) + Complex.abs (z - 3*I)) ≤ 1 / (Complex.abs (w + 6) + Complex.abs (w - 3*I))) →
  1 / (Complex.abs (z + 6) + Complex.abs (z - 3*I)) = Real.sqrt 5 / 15 ∧
  z = -4 + I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_l1104_110438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3867_to_nearest_hundredth_l1104_110496

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_3867_to_nearest_hundredth :
  roundToHundredth 3.867 = 3.87 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3867_to_nearest_hundredth_l1104_110496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_cookies_dozens_l1104_110427

/-- The number of dozens of cookies Danny made -/
def cookies_dozens : ℕ := 4

/-- The number of chips in each cookie -/
def chips_per_cookie : ℕ := 7

/-- The total number of cookies -/
def total_cookies : ℕ := 12 * cookies_dozens

/-- The number of cookies eaten -/
def eaten_cookies : ℕ := total_cookies / 2

/-- The number of cookies left uneaten -/
def uneaten_cookies : ℕ := total_cookies - eaten_cookies

/-- The number of chips left uneaten -/
def uneaten_chips : ℕ := 168

theorem danny_cookies_dozens : 
  cookies_dozens = 4 := by
  -- The proof goes here
  sorry

#eval cookies_dozens

end NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_cookies_dozens_l1104_110427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_path_l1104_110492

theorem shorter_diagonal_path (length width : ℝ) (h1 : length = 2) (h2 : width = 1) :
  let john_path := length + width
  let anne_path := Real.sqrt (length ^ 2 + width ^ 2)
  let percentage_reduction := (john_path - anne_path) / john_path * 100
  ∃ (ε : ℝ), abs (percentage_reduction - 25.47) < ε ∧ ε < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_path_l1104_110492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_in_intervals_l1104_110451

theorem equation_roots_in_intervals (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo 0 1 ∧ x₂ ∈ Set.Ioo 1 2 ∧
   x₁^2 + 2*a*x₁ - a = 0 ∧ x₂^2 + 2*a*x₂ - a = 0) ↔
  a ∈ Set.Ioo (-4/3) (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_in_intervals_l1104_110451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_spot_diameter_l1104_110498

/-- Given a thin lens with focal length F and diameter D, and a point light source
    at distance d > F from the lens, the diameter D' of the light spot on a screen
    placed at the focal point of the lens is D' = F*D/d. -/
theorem light_spot_diameter
  (F D d : ℝ) -- focal length, lens diameter, distance to light source
  (h_pos_F : F > 0)
  (h_pos_D : D > 0)
  (h_pos_d : d > 0)
  (h_d_gt_F : d > F) :
  ∃ D' : ℝ, D' = F * D / d ∧ D' > 0 :=
by
  -- Define D' as F*D/d
  let D' := F * D / d
  
  -- Prove that D' satisfies the equation
  have h_eq : D' = F * D / d := by rfl
  
  -- Prove that D' is positive
  have h_pos : D' > 0 := by
    apply div_pos
    · exact mul_pos h_pos_F h_pos_D
    · exact h_pos_d
  
  -- Conclude the proof
  exact ⟨D', ⟨h_eq, h_pos⟩⟩

#check light_spot_diameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_spot_diameter_l1104_110498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_royalty_calculation_l1104_110490

-- Define the taxation rules
noncomputable def tax_rate (royalty : ℝ) : ℝ :=
  if royalty ≤ 800 then 0
  else if royalty ≤ 4000 then 0.14
  else 0.11

noncomputable def taxable_amount (royalty : ℝ) : ℝ :=
  if royalty ≤ 800 then 0
  else if royalty ≤ 4000 then royalty - 800
  else royalty

-- Define the theorem
theorem royalty_calculation (tax_paid : ℝ) (h1 : tax_paid = 420) : 
  ∃ (royalty : ℝ), tax_rate royalty * taxable_amount royalty = tax_paid ∧ royalty = 3800 := by
  -- The proof is omitted for now
  sorry

#check royalty_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_royalty_calculation_l1104_110490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_l1104_110489

theorem sqrt_sum_equality : Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_l1104_110489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_group_size_l1104_110488

/-- The number of people in the initial group -/
def n : ℕ := 4

/-- The weight increase of the new person compared to the replaced person -/
def weight_increase : ℝ := 129 - 95

/-- The average weight increase for the group -/
def avg_weight_increase : ℝ := 8.5

theorem initial_group_size :
  n = Int.floor (weight_increase / avg_weight_increase) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_group_size_l1104_110488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_l1104_110443

/-- A square in a 2D plane. -/
structure Square where
  /-- The side length of the square. -/
  sideLength : ℝ

/-- A rectangle in a 2D plane. -/
structure Rectangle where
  /-- The length of the rectangle. -/
  length : ℝ
  /-- The width of the rectangle. -/
  width : ℝ

/-- Predicate to check if a square shares a vertex with a rectangle. -/
def shareVertex (s : Square) (r : Rectangle) : Prop :=
  sorry -- Define the sharing vertex condition here

/-- Given a square ABCD and a rectangle BEFG sharing a common vertex B,
    where the side length of ABCD is 6 cm and the length of BEFG is 9 cm,
    prove that the width of BEFG is 4 cm. -/
theorem rectangle_width (ABCD : Square) (BEFG : Rectangle) :
  shareVertex ABCD BEFG →
  ABCD.sideLength = 6 →
  BEFG.length = 9 →
  BEFG.width = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_l1104_110443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_bottle_purchase_l1104_110424

/-- The number of small bottles John purchased -/
def num_small_bottles : ℕ := 747

/-- The number of large bottles John purchased -/
def num_large_bottles : ℕ := 1325

/-- The price of a large bottle in dollars -/
def price_large_bottle : ℚ := 189/100

/-- The price of a small bottle in dollars -/
def price_small_bottle : ℚ := 138/100

/-- The approximate average price paid per bottle in dollars -/
def avg_price : ℚ := 17057/10000

/-- A function to check if two rational numbers are approximately equal within a small epsilon -/
def approx_equal (x y : ℚ) : Prop := abs (x - y) < 1/1000

theorem john_bottle_purchase :
  approx_equal
    ((num_large_bottles * price_large_bottle + num_small_bottles * price_small_bottle) / 
     (num_large_bottles + num_small_bottles))
    avg_price := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_bottle_purchase_l1104_110424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_exponential_logarithm_l1104_110457

/-- The minimum distance between points on y = 2^x and y = log_2(x) -/
theorem min_distance_exponential_logarithm :
  ∃ (min_dist : ℝ),
    min_dist = (1 + Real.log (Real.log 2)) / Real.log 2 * Real.sqrt 2 ∧
    ∀ (a b : ℝ),
      let p := (a, (2 : ℝ) ^ a)
      let q := (b, Real.log b / Real.log 2)
      Real.sqrt ((a - b)^2 + ((2 : ℝ) ^ a - Real.log b / Real.log 2)^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_exponential_logarithm_l1104_110457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_eraser_count_l1104_110448

/-- Represents the contents of a drawer -/
structure Drawer where
  erasers : ℕ
  scissors : ℕ
  sharpeners : ℕ

/-- Represents actions taken on the drawer contents -/
inductive DrawerAction
  | AddErasers (n : ℕ)
  | RemoveScissors (n : ℕ)

/-- Apply an action to the drawer contents -/
def applyAction (d : Drawer) (a : DrawerAction) : Drawer :=
  match a with
  | DrawerAction.AddErasers n => { d with erasers := d.erasers + n }
  | DrawerAction.RemoveScissors n => { d with scissors := d.scissors - n }

/-- The theorem to be proved -/
theorem final_eraser_count
  (initial : Drawer)
  (jason_add : ℕ)
  (jason_remove : ℕ)
  (mia_add : ℕ)
  (h_initial : initial = { erasers := 139, scissors := 118, sharpeners := 57 })
  (h_jason_add : jason_add = 131)
  (h_jason_remove : jason_remove = 25)
  (h_mia_add : mia_add = 84) :
  (applyAction (applyAction (applyAction initial (DrawerAction.AddErasers jason_add))
    (DrawerAction.RemoveScissors jason_remove)) (DrawerAction.AddErasers mia_add)).erasers = 354 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_eraser_count_l1104_110448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l1104_110432

/-- The length of the other parallel side of a trapezium -/
noncomputable def other_side (a b h : ℝ) : ℝ := 2 * (a / h) - b

theorem trapezium_other_side :
  let a : ℝ := 323  -- area
  let b : ℝ := 18   -- one parallel side
  let h : ℝ := 17   -- height
  other_side a b h = 20 := by
  -- Unfold the definition of other_side
  unfold other_side
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l1104_110432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1104_110431

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1/2)

noncomputable def f (x : ℝ) : ℝ := ((a x).1 + (b x).1) * (a x).1 + ((a x).2 + (b x).2) * (a x).2 - 2

theorem triangle_area (A : ℝ) (hA : 0 < A ∧ A < π/2) 
  (ha : Real.sqrt 3 = (Real.sqrt 3)^2 + 1 - 2 * Real.sqrt 3 * Real.cos A) 
  (hf : f A = 1) : 
  1/2 * Real.sqrt 3 * 1 * Real.sin A = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1104_110431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_window_side_length_l1104_110480

/-- Represents the dimensions of a glass pane -/
structure PaneDimensions where
  width : ℝ
  height : ℝ
  ratio : height = (5/2) * width

/-- Represents the dimensions of a window -/
structure WindowDimensions where
  pane : PaneDimensions
  border_width : ℝ
  is_square : ℝ → Prop

/-- Theorem stating the side length of the square window -/
theorem square_window_side_length 
  (w : WindowDimensions) 
  (h_border : w.border_width = 2) 
  (h_square : w.is_square = λ s ↦ 4 * w.pane.width + 5 * w.border_width = s ∧ 
                                 2 * w.pane.height + 3 * w.border_width = s) : 
  ∃ (side : ℝ), w.is_square side ∧ side = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_window_side_length_l1104_110480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1104_110416

/-- The time taken by worker B to complete a work independently -/
noncomputable def time_B : ℝ := sorry

/-- The time taken by worker A to complete the work independently -/
noncomputable def time_A : ℝ := time_B / 2

/-- The time taken by A and B working together to complete the work -/
def time_AB : ℝ := 4

theorem work_completion_time :
  (time_A = time_B / 2) →
  (1 / time_A + 1 / time_B = 1 / time_AB) →
  time_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1104_110416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_participates_in_running_l1104_110456

-- Define the students and events
inductive Student : Type
| A : Student
| B : Student
| C : Student

inductive Event : Type
| Running : Event
| LongJump : Event
| ShotPut : Event

-- Define a function to represent the height order
def HeightOrder : Student → ℕ := sorry

-- Define a function to represent the event participation
def Participates : Student → Event := sorry

-- State the theorem
theorem c_participates_in_running :
  -- Conditions
  (∀ s1 s2 : Student, s1 ≠ s2 → HeightOrder s1 ≠ HeightOrder s2) →
  (∀ e1 e2 : Event, e1 ≠ e2 → ∀ s : Student, Participates s ≠ e1 ∨ Participates s ≠ e2) →
  (HeightOrder Student.A ≠ 3) →
  (∀ s : Student, HeightOrder s = 3 → Participates s ≠ Event.ShotPut) →
  (∀ s : Student, HeightOrder s = 1 → Participates s = Event.LongJump) →
  (HeightOrder Student.B ≠ 1) →
  (Participates Student.B ≠ Event.Running) →
  -- Conclusion
  Participates Student.C = Event.Running :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_participates_in_running_l1104_110456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1104_110444

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus F
def right_focus (a b c : ℝ) : Prop := c^2 = a^2 + b^2

-- Define the asymptote of the hyperbola
def asymptote (a b x y : ℝ) : Prop := y = (b / a) * x

-- Define the circle with center F
def circle_F (b x y c : ℝ) : Prop := (x - c)^2 + y^2 = b^2

-- Define the point M on the hyperbola in the first quadrant
def point_M (a b c : ℝ) : Prop := c^2 / a^2 - (b^2 / a)^2 / b^2 = 1

-- Define MF perpendicular to real axis
def MF_perpendicular (a b : ℝ) : Prop := b^2 / a = b

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_hyperbola : hyperbola a b c (b^2 / a))
  (h_focus : right_focus a b c)
  (h_asymptote : asymptote a b c (b^2 / a))
  (h_circle : circle_F b c (b^2 / a) c)
  (h_point_M : point_M a b c)
  (h_perpendicular : MF_perpendicular a b) :
  c / a = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1104_110444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fuel_usage_l1104_110434

def fuel_this_week : ℝ := 15
def fuel_reduction_percentage : ℝ := 0.2

theorem total_fuel_usage :
  let fuel_last_week := fuel_this_week * (1 - fuel_reduction_percentage)
  fuel_this_week + fuel_last_week = 27 := by
  -- Unfold the definition of fuel_last_week
  simp [fuel_this_week, fuel_reduction_percentage]
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fuel_usage_l1104_110434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frieda_corner_probability_l1104_110499

/-- Represents a position on the 4x4 grid -/
inductive Position
| Corner : Position
| Edge : Position
| Internal : Position

/-- Represents the number of hops, up to 5 -/
def Hops : Type := Fin 6

/-- The probability of reaching a corner from a given position within a certain number of hops -/
def probability_to_corner (start : Position) (hops : Nat) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem frieda_corner_probability :
  probability_to_corner Position.Edge 5 = 93 / 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frieda_corner_probability_l1104_110499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1104_110471

def y : ℕ → ℝ
  | 0 => 200
  | n + 1 => (y n)^2 - y n

theorem series_sum : (∑' n, 1 / (y n + 1)) = 1 / y 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1104_110471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_polygon_area_l1104_110459

/-- A regular dodecahedron with side length 1 -/
structure Dodecahedron where
  side_length : ℝ
  is_regular : Bool
  side_length_eq_one : side_length = 1

/-- A plane that intersects the dodecahedron -/
structure IntersectionPlane where
  parallel_to_opposite_faces : Bool
  divides_into_congruent_solids : Bool

/-- The polygon formed at the intersection -/
def intersection_polygon (d : Dodecahedron) (p : IntersectionPlane) : Set (ℝ × ℝ) := sorry

/-- The area of a polygon -/
noncomputable def polygon_area (p : Set (ℝ × ℝ)) : ℝ := sorry

/-- Main theorem: The area of the intersection polygon is (5 + 5√5) / 4 -/
theorem intersection_polygon_area 
  (d : Dodecahedron) 
  (p : IntersectionPlane) 
  (h1 : d.is_regular = true) 
  (h2 : p.parallel_to_opposite_faces = true) 
  (h3 : p.divides_into_congruent_solids = true) : 
  polygon_area (intersection_polygon d p) = (5 + 5 * Real.sqrt 5) / 4 := by
  sorry

#check intersection_polygon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_polygon_area_l1104_110459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_put_15_oxen_l1104_110439

/-- Calculates the number of oxen C put for grazing given the rental conditions -/
def calculate_c_oxen (total_rent : ℚ) (a_oxen a_months : ℕ) (b_oxen b_months : ℕ) 
  (c_months : ℕ) (c_rent : ℚ) : ℚ :=
  let a_oxen_months := (a_oxen * a_months : ℚ)
  let b_oxen_months := (b_oxen * b_months : ℚ)
  let total_oxen_months := a_oxen_months + b_oxen_months + c_months * (c_rent * (a_oxen_months + b_oxen_months + c_months * (total_rent / c_rent)) / total_rent)
  c_rent * total_oxen_months / total_rent / c_months

/-- Proves that C put 15 oxen for grazing given the specific rental conditions -/
theorem c_put_15_oxen : 
  calculate_c_oxen 175 10 7 12 5 3 45 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_put_15_oxen_l1104_110439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1104_110402

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1104_110402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_every_nat_is_sum_of_two_three_powers_l1104_110491

def is_two_three_power (n : ℕ) : Prop :=
  ∃ (α β : ℕ), n = 2^α * 3^β

def is_valid_sum (s : Finset ℕ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≠ y → (x ∣ y ∨ y ∣ x) → False

theorem every_nat_is_sum_of_two_three_powers :
  ∀ n : ℕ, n > 0 → ∃ s : Finset ℕ, 
    (∀ x, x ∈ s → is_two_three_power x) ∧
    is_valid_sum s ∧
    (s.sum id = n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_every_nat_is_sum_of_two_three_powers_l1104_110491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equals_interval_l1104_110473

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | 0 ≤ x}

-- State that f is monotonically increasing on its domain
axiom f_increasing : ∀ (x y : ℝ), x ∈ domain → y ∈ domain → x < y → f x < f y

-- Define the set of x that satisfies the inequality
def solution_set : Set ℝ := {x : ℝ | f (2*x - 1) < f (1/3)}

-- State the theorem
theorem solution_set_equals_interval :
  solution_set = {x : ℝ | 1/2 ≤ x ∧ x < 2/3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equals_interval_l1104_110473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1104_110422

/-- The ellipse equation -/
def is_ellipse (x y b : ℝ) : Prop := x^2 / 16 + y^2 / b^2 = 1

/-- The foci of the ellipse -/
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁.1 = -F₂.1 ∧ F₁.2 = 0 ∧ F₂.2 = 0

/-- A line passing through F₁ and intersecting the ellipse at A and B -/
def line_intersects (F₁ A B : ℝ × ℝ) (b : ℝ) : Prop :=
  ∃ (t : ℝ), A = F₁ + t • (B - F₁) ∧
  is_ellipse A.1 A.2 b ∧ is_ellipse B.1 B.2 b

/-- The maximum value of |AF₂| + |BF₂| is 10 -/
def max_sum_distances (F₂ A B : ℝ × ℝ) : Prop :=
  ∀ (A' B' : ℝ × ℝ), dist A F₂ + dist B F₂ ≤ 10 ∧
  (dist A' F₂ + dist B' F₂ = 10 → dist A F₂ + dist B F₂ = 10)

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (b : ℝ) : ℝ := Real.sqrt (1 - b^2 / 16)

theorem ellipse_eccentricity (b : ℝ) (F₁ F₂ A B : ℝ × ℝ) :
  foci F₁ F₂ →
  line_intersects F₁ A B b →
  max_sum_distances F₂ A B →
  eccentricity b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1104_110422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_theorem_l1104_110461

/-- The area of overlap between two strips -/
noncomputable def areaOfOverlap (θ : ℝ) : ℝ :=
  2.5 * (1 + Real.sin θ)

/-- The width of the first strip -/
def width1 : ℝ := 2

/-- The width of the second strip -/
def width2 : ℝ := 3

/-- The elevation of the second strip at one end -/
def elevation : ℝ := 1

theorem overlap_area_theorem (θ : ℝ) :
  let w1 := width1
  let w2 := width2
  let h := elevation
  let A := areaOfOverlap θ
  A = (w1 + w2) / 2 * (h + Real.sin θ) :=
by
  sorry

#check overlap_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_theorem_l1104_110461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1104_110404

noncomputable def f (a b x : ℝ) : ℝ := (2 * a * x + b) / (x^2 + 1)

theorem function_properties :
  ∀ a b : ℝ,
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a b x = -f a b (-x)) →
  (f a b (1/2) = 4/5) →
  (a = 1 ∧ b = 0) ∧
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f a b x < f a b y) ∧
  (∀ m : ℝ,
    (∀ x t, x ∈ Set.Icc (-1 : ℝ) 1 → t ∈ Set.Icc (-1 : ℝ) 1 → f a b x ≤ m^2 - 5*m*t - 5) ↔
    (m ≤ -6 ∨ m ≥ 6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1104_110404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l1104_110426

/-- Calculates the number of days required for a group to build a wall, given the number of people and the time taken by a reference group. -/
noncomputable def daysRequired (referencePeople : ℕ) (referenceDays : ℝ) (newPeople : ℕ) : ℝ :=
  (referencePeople : ℝ) * referenceDays / (newPeople : ℝ)

/-- Theorem stating that 30 people will take 11.2 days to build a wall similar to one that 8 people can build in 42 days. -/
theorem wall_building_time :
  let referencePeople := 8
  let referenceDays := 42
  let newPeople := 30
  daysRequired referencePeople referenceDays newPeople = 11.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l1104_110426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_x_from_2_to_4_l1104_110405

open Real MeasureTheory Interval

theorem integral_reciprocal_x_from_2_to_4 :
  ∫ x in (2 : ℝ)..4, 1 / x = log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_x_from_2_to_4_l1104_110405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l1104_110403

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : c^2 = a^2 + b^2

-- Define points A, B, and C
noncomputable def A (h : Hyperbola) : ℝ × ℝ := (-h.c, h.b^2 / h.a)
noncomputable def B (h : Hyperbola) : ℝ × ℝ := (-h.c, -h.b^2 / h.a)
noncomputable def C (h : Hyperbola) : ℝ × ℝ := (0, Real.sqrt ((h.b^4 / h.a^2) - h.c^2))

-- Define eccentricity
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

-- Define the dot product of vectors AC and BC
noncomputable def dot_product_AC_BC (h : Hyperbola) : ℝ :=
  let a := A h
  let b := B h
  let c := C h
  (c.1 - a.1) * (c.1 - b.1) + (c.2 - a.2) * (c.2 - b.2)

-- State the theorem
theorem hyperbola_eccentricity_bound (h : Hyperbola) 
  (h_dot_product : dot_product_AC_BC h = 0) :
  eccentricity h ≥ (Real.sqrt 5 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l1104_110403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_maze_probability_l1104_110445

-- Define the maze structure
structure Maze where
  junctions : Finset String
  paths : Finset (String × String)
  start : String
  target : String

-- Define the probability of choosing a path at a junction
def probability_at_junction (m : Maze) (j : String) : ℚ :=
  1 / (m.paths.filter (fun p => p.1 = j)).card

-- Define the probability of reaching a junction
noncomputable def probability_reach_junction (m : Maze) (j : String) : ℚ :=
  sorry -- This would be calculated based on the maze structure

-- The main theorem
theorem harry_maze_probability (m : Maze) 
  (h1 : m.junctions = {"S", "V", "W", "X", "Y", "Z", "B"})
  (h2 : m.paths = {("S", "V"), ("V", "W"), ("V", "X"), ("V", "Y"), ("W", "X"), ("Y", "X"), ("X", "B")})
  (h3 : m.start = "S")
  (h4 : m.target = "B") :
  probability_reach_junction m "B" = 11 / 18 := by
  sorry

-- Example of creating a maze instance
def example_maze : Maze := {
  junctions := {"S", "V", "W", "X", "Y", "Z", "B"},
  paths := {("S", "V"), ("V", "W"), ("V", "X"), ("V", "Y"), ("W", "X"), ("Y", "X"), ("X", "B")},
  start := "S",
  target := "B"
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_maze_probability_l1104_110445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_prime_factors_l1104_110410

theorem smallest_odd_five_prime_factors : 
  ∀ n : ℕ, n > 0 → Odd n → (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅) →
  n ≥ 15015 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_prime_factors_l1104_110410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_gcd_l1104_110476

theorem existence_of_prime_gcd (S : Finset ℕ) 
  (h1 : S.Nonempty) 
  (h2 : ∀ s ∈ S, s > 1)
  (h3 : ∃ s ∈ S, ∀ n : ℕ, n > 0 → (Nat.gcd s n = 1 ∨ Nat.gcd s n = s)) :
  ∃ s t : ℕ, s ∈ S ∧ t ∈ S ∧ Nat.Prime (Nat.gcd s t) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_gcd_l1104_110476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l1104_110420

/-- Represents a line segment in 3D space -/
structure LineSegment where
  start : ℝ × ℝ × ℝ
  end_ : ℝ × ℝ × ℝ

/-- Calculates the length of a line segment -/
noncomputable def length (seg : LineSegment) : ℝ := sorry

/-- Calculates the volume of the region within a given distance of a line segment -/
noncomputable def regionVolume (seg : LineSegment) (distance : ℝ) : ℝ := sorry

theorem line_segment_length (CD : LineSegment) :
  regionVolume CD 4 = 432 * Real.pi → length CD = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l1104_110420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l1104_110463

-- Define the circle O₁
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*y - 3 = 0

-- Define the fixed point A
def point_A : ℝ × ℝ := (1, -2)

-- Define the line x + 3y = 0
def line_x_3y (x y : ℝ) : Prop := x + 3*y = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Define the theorem
theorem circle_and_line_problem :
  -- Given conditions
  ∃ (O₂_x O₂_y M_x M_y N_x N_y : ℝ),
  -- O₂ is symmetric to A about the line x + 3y = 0
  (line_x_3y ((O₂_x + point_A.1) / 2) ((O₂_y + point_A.2) / 2)) ∧
  -- M and N are on circle O₁
  (circle_O₁ M_x M_y) ∧ (circle_O₁ N_x N_y) ∧
  -- M and N are also on circle O₂
  ((M_x - O₂_x)^2 + (M_y - O₂_y)^2 = (N_x - O₂_x)^2 + (N_y - O₂_y)^2) ∧
  -- |MN| = 2√2
  (distance M_x M_y N_x N_y = 2 * Real.sqrt 2) →
  -- Conclusion 1: Equation of line l
  (∃ (center_x center_y : ℝ),
    circle_O₁ center_x center_y ∧
    center_x + center_y + 1 = 0 ∧
    point_A.1 + point_A.2 + 1 = 0) ∧
  -- Conclusion 2: Equation of circle O₂
  ((O₂_x - 2)^2 + (O₂_y + 1)^2 = 20 ∨ (O₂_x - 2)^2 + (O₂_y + 1)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l1104_110463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_lines_l1104_110495

/-- Line1 is defined as (1, 1, 3) + t(1, -1, 2) -/
def Line1 (t : ℝ) : Fin 3 → ℝ := fun i => match i with
  | 0 => 1 + t
  | 1 => 1 - t
  | 2 => 3 + 2*t

/-- Line2 is defined as (2, 3, 2) + s(-1, 2, 1) -/
def Line2 (s : ℝ) : Fin 3 → ℝ := fun i => match i with
  | 0 => 2 - s
  | 1 => 3 + 2*s
  | 2 => 2 + s

/-- Distance squared between two points in 3D space -/
def distanceSquared (p q : Fin 3 → ℝ) : ℝ :=
  (p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2

/-- The theorem stating the shortest distance between the two lines -/
theorem shortest_distance_between_lines :
  ∃ (t s : ℝ), ∀ (t' s' : ℝ), 
    distanceSquared (Line1 t) (Line2 s) ≤ distanceSquared (Line1 t') (Line2 s') ∧
    distanceSquared (Line1 t) (Line2 s) = 82 := by
  sorry

#check shortest_distance_between_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_lines_l1104_110495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_B_works_l1104_110452

/-- The number of days it takes person A to complete the task alone -/
noncomputable def days_A : ℝ := 20

/-- The number of days it takes person B to complete the task alone -/
noncomputable def days_B : ℝ := 10

/-- The total number of days allowed to complete the task -/
noncomputable def total_days : ℝ := 12

/-- The portion of the task completed in one day by person A -/
noncomputable def rate_A : ℝ := 1 / days_A

/-- The portion of the task completed in one day by person B -/
noncomputable def rate_B : ℝ := 1 / days_B

theorem days_B_works (x : ℝ) : 
  (x * rate_B + (total_days - x) * rate_A = 1) → x = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_B_works_l1104_110452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l1104_110425

/-- A sequence {aₙ} defined by a₁ = 5 and aₙ₊₁ = aₙ + 3 for all n ≥ 1 -/
def a : ℕ → ℝ
  | 0 => 5  -- Add this case to cover n = 0
  | n + 1 => a n + 3

/-- The general formula for the sequence {aₙ} -/
def general_formula (n : ℕ) : ℝ := 3 * n + 2

/-- Theorem stating that the general formula correctly describes the sequence for all n ≥ 1 -/
theorem sequence_formula_correct (n : ℕ) (h : n ≥ 1) : a n = general_formula n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l1104_110425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fdi_rural_andhra_pradesh_l1104_110413

/-- Given the FDI distribution in India, calculate the FDI in rural Andhra Pradesh -/
theorem fdi_rural_andhra_pradesh (total_fdi : ℝ) : 
  let gujarat_fdi := 0.3 * total_fdi
  let gujarat_rural_fdi := 0.2 * gujarat_fdi
  let gujarat_urban_fdi := 0.8 * gujarat_fdi
  let andhra_pradesh_fdi := 0.2 * total_fdi
  let andhra_pradesh_rural_fdi := 0.5 * andhra_pradesh_fdi
  gujarat_urban_fdi = 72 →
  andhra_pradesh_rural_fdi = 30 := by
  intros
  -- Proof steps would go here
  sorry

#check fdi_rural_andhra_pradesh

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fdi_rural_andhra_pradesh_l1104_110413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1104_110467

def point1 : ℝ × ℝ := (3, -2)
def point2 : ℝ × ℝ := (-5, 6)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance point1 point2 = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1104_110467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_methane_combined_moles_l1104_110470

/-- Represents a chemical compound -/
inductive Compound
| Methane
| Oxygen
| Water
| CarbonDioxide

/-- Represents the coefficients in a chemical equation -/
structure Coefficient where
  compound : Compound
  value : ℕ

/-- Represents a chemical equation -/
structure ChemicalEquation where
  reactants : List Coefficient
  products : List Coefficient

def balancedEquation : ChemicalEquation :=
{ reactants := [⟨Compound.Methane, 1⟩, ⟨Compound.Oxygen, 2⟩],
  products := [⟨Compound.CarbonDioxide, 1⟩, ⟨Compound.Water, 2⟩] }

def oxygenUsed : ℕ := 2
def waterProduced : ℕ := 2

theorem methane_combined_moles :
  ∃ (methaneMoles : ℕ), methaneMoles = 1 ∧
  (∃ (c : Coefficient), c ∈ balancedEquation.reactants ∧ c.compound = Compound.Methane ∧ c.value = methaneMoles) ∧
  (∃ (c : Coefficient), c ∈ balancedEquation.reactants ∧ c.compound = Compound.Oxygen ∧ c.value = oxygenUsed) ∧
  (∃ (c : Coefficient), c ∈ balancedEquation.products ∧ c.compound = Compound.Water ∧ c.value = waterProduced) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_methane_combined_moles_l1104_110470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_pentagon_sum_l1104_110477

/-- A labeling of a pentagon is a permutation of the numbers 1 to 10,
    where the first 5 numbers are on the vertices and the last 5 are on the sides. -/
def PentagonLabeling := Fin 10 → Fin 10

/-- The sum of numbers on each side of the pentagon for a given labeling -/
def sideSum (l : PentagonLabeling) : ℕ :=
  let vertices := (List.range 5).map l
  let sides := (List.range 5).map (fun i => l (i + 5))
  (vertices.sum * 2 + sides.sum) / 5

/-- Predicate to check if a labeling results in equal sums on all sides -/
def isValidLabeling (l : PentagonLabeling) : Prop :=
  ∀ i j : Fin 5, 
    l i + l ((i + 1) % 5) + l (i + 5) = 
    l j + l ((j + 1) % 5) + l (j + 5)

/-- The main theorem: The smallest possible sum on each side is 14 -/
theorem smallest_pentagon_sum :
  (∃ l : PentagonLabeling, isValidLabeling l) ∧
  (∀ l : PentagonLabeling, isValidLabeling l → sideSum l ≥ 14) ∧
  (∃ l : PentagonLabeling, isValidLabeling l ∧ sideSum l = 14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_pentagon_sum_l1104_110477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1104_110414

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 25 * Real.sin (4 * x) - 60 * Real.cos (4 * x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1104_110414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_room_dimension_room_dimension_problem_l1104_110454

/-- Represents the dimensions of a room --/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a door or window --/
structure OpeningDimensions where
  width : ℝ
  height : ℝ

/-- Calculate the area to be whitewashed given room dimensions, door, and windows --/
def areaToWhitewash (room : RoomDimensions) (door : OpeningDimensions) (window : OpeningDimensions) (numWindows : ℕ) : ℝ :=
  2 * (room.length + room.width) * room.height - (door.width * door.height + numWindows * window.width * window.height)

/-- The theorem stating the unknown dimension of the room --/
theorem unknown_room_dimension 
  (costPerSqFt : ℝ) 
  (totalCost : ℝ) 
  (door : OpeningDimensions) 
  (window : OpeningDimensions) 
  (numWindows : ℕ) :
  ∃ (x : ℝ), 
    let room := RoomDimensions.mk x 15 12
    totalCost = costPerSqFt * areaToWhitewash room door window numWindows ∧ 
    x = 25 := by
  sorry

/-- The main theorem applying the specific values from the problem --/
theorem room_dimension_problem : 
  ∃ (x : ℝ), 
    let room := RoomDimensions.mk x 15 12
    let door := OpeningDimensions.mk 6 3
    let window := OpeningDimensions.mk 4 3
    9060 = 10 * areaToWhitewash room door window 3 ∧ 
    x = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_room_dimension_room_dimension_problem_l1104_110454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_prime_arithmetic_progression_l1104_110453

def isArithmeticProgression (s : List Nat) (d : Nat) : Prop :=
  ∀ i : Nat, i + 1 < s.length → s.get! (i + 1) = s.get! i + d

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

theorem longest_prime_arithmetic_progression :
  ∀ s : List Nat,
    isArithmeticProgression s 6 →
    (∀ n, n ∈ s → isPrime n) →
    s.length ≤ 5 ∧
    ∃ t : List Nat, t = [5, 11, 17, 23, 29] ∧
      isArithmeticProgression t 6 ∧
      (∀ n, n ∈ t → isPrime n) ∧
      t.length = 5 :=
by sorry

#check longest_prime_arithmetic_progression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_prime_arithmetic_progression_l1104_110453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_k_zero_f_has_minimum_iff_k_gt_one_l1104_110436

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k + x) / (x - 1) * Real.log x

-- Part 1: Monotonicity when k = 0
theorem f_increasing_when_k_zero :
  ∀ x₁ x₂, (0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 ∧ x₁ < x₂) ∨ 
           (1 < x₁ ∧ 1 < x₂ ∧ x₁ < x₂) →
  f 0 x₁ < f 0 x₂ := by
  sorry

-- Part 2: Condition for minimum value
theorem f_has_minimum_iff_k_gt_one :
  ∀ k, (∃ x₀, x₀ > 1 ∧ ∀ x > 1, f k x₀ ≤ f k x) ↔ k > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_k_zero_f_has_minimum_iff_k_gt_one_l1104_110436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_pay_percentage_l1104_110417

/-- Given two employees X and Y with a total weekly pay, calculate the percentage of X's pay compared to Y's. -/
theorem employee_pay_percentage (total_pay y_pay : ℚ) (h1 : total_pay = 560) (h2 : y_pay = 254.55) :
  let x_pay := total_pay - y_pay
  ∃ ε > 0, |((x_pay / y_pay) * 100) - 119.9| < ε := by
  sorry

#eval (560 - 254.55) / 254.55 * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_pay_percentage_l1104_110417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solution_l1104_110408

theorem cubic_root_equation_solution (x : ℝ) :
  (Real.rpow (5 + Real.sqrt x) (1/3) + Real.rpow (5 - Real.sqrt x) (1/3) = 1) → x = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solution_l1104_110408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_from_square_l1104_110493

/-- Given a square with side length a, the area of the regular octagon formed by trimming its corners is 2a^2(√2 - 1) -/
theorem octagon_area_from_square (a : ℝ) (a_pos : a > 0) :
  let square_area := a^2
  let trim_length := a * (1 - 1 / (2 + Real.sqrt 2))
  let trim_area := 4 * (1/2 * trim_length^2)
  let octagon_area := square_area - trim_area
  octagon_area = 2 * a^2 * (Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_from_square_l1104_110493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1104_110419

-- Define the sets A and B
def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := {x | 2 - x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-3) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1104_110419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1104_110468

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

theorem range_of_f :
  (∀ y ∈ Set.range f, y ∈ Set.Icc 1 4) ∧
  (∀ y ∈ Set.Icc 1 4, ∃ x ∈ Set.Icc (-1) 1, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1104_110468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_problem_l1104_110462

theorem alcohol_concentration_problem (vessel1_capacity : ℝ) (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ) (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ) (final_vessel_capacity : ℝ) :
  vessel1_capacity = 2 →
  vessel1_alcohol_percentage = 0.3 →
  vessel2_capacity = 6 →
  vessel2_alcohol_percentage = 0.45 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  (vessel1_capacity * vessel1_alcohol_percentage + vessel2_capacity * vessel2_alcohol_percentage) /
    final_vessel_capacity = 0.33 := by
  intro h1 h2 h3 h4 h5 h6
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_problem_l1104_110462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1104_110429

/-- An ellipse passing through (0, 4) with eccentricity 3/5 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_point : b = 4
  h_ecc : (a^2 - b^2) / a^2 = 9/25

/-- A line passing through (3, 0) with slope 4/5 -/
structure Line where
  m : ℝ
  c : ℝ
  h_point : 3 * m + c = 0
  h_slope : m = 4/5

/-- The midpoint of the line segment intercepted by the ellipse on the line -/
def intersectionMidpoint (e : Ellipse) (l : Line) : ℝ × ℝ := sorry

theorem ellipse_and_line_properties (e : Ellipse) (l : Line) :
  (∀ x y, x^2 / 25 + y^2 / 16 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  intersectionMidpoint e l = (3/2, -6/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1104_110429
