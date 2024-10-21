import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_trail_difference_l1208_120886

/-- Represents a hiking trail with its length and speed --/
structure Trail where
  length : ℝ
  speed : ℝ

/-- Calculates the time to hike a trail --/
noncomputable def hikeTime (trail : Trail) (breakTime : ℝ := 0) : ℝ :=
  trail.length / trail.speed + breakTime

theorem faster_trail_difference (downhillTrail uphillTrail : Trail) 
  (h1 : downhillTrail.length = 20)
  (h2 : downhillTrail.speed = 5)
  (h3 : uphillTrail.length = 12)
  (h4 : uphillTrail.speed = 3)
  : hikeTime uphillTrail 1 - hikeTime downhillTrail = 1 := by
  sorry

#check faster_trail_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_trail_difference_l1208_120886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1208_120871

open Real

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) :
  Real.cos (2 * A - B) + Real.sin (A + B) = 2 →
  AB = 4 →
  -- Triangle ABC exists
  0 < AB ∧ 0 < BC ∧ 0 < AC →
  AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB →
  BC = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1208_120871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_A_l1208_120849

/-- The set S of pairs (a, b) where a ∈ {1, ..., m} and b ∈ {1, ..., n} -/
def S (m n : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ∈ Finset.range m ∧ p.2 ∈ Finset.range n}

/-- The condition that A does not contain a specific pattern -/
def NoPattern (A : Set (ℕ × ℕ)) : Prop :=
  ∀ x₁ x₂ y₁ y₂ y₃ : ℕ,
    x₁ < x₂ → y₁ < y₂ → y₂ < y₃ →
    ¬((x₁, y₁) ∈ A ∧ (x₁, y₂) ∈ A ∧ (x₁, y₃) ∈ A ∧ (x₂, y₂) ∈ A)

theorem max_cardinality_A (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 3) :
  ∃ A : Finset (ℕ × ℕ), A.toSet ⊆ S m n ∧ NoPattern A.toSet ∧ 
    (∀ B : Finset (ℕ × ℕ), B.toSet ⊆ S m n → NoPattern B.toSet → B.card ≤ 2*m + n - 2) ∧
    A.card = 2*m + n - 2 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_A_l1208_120849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_sqrt_l1208_120804

/-- Theorem: The exact value of √((3 - cos²(π/9))(3 - cos²(2π/9))(3 - cos²(4π/9))) is 5√5/4 -/
theorem cosine_product_sqrt : 
  let f (x : Real) := x^3 - (9/2)*x^2 + (27/16)*x - 1/16
  (∀ k ∈ ({1, 2, 4} : Set ℕ), 
    (Real.cos (9 * (k * Real.pi / 9)) = 0) ∧ 
    (f (Real.cos (k * Real.pi / 9))^2 = 0)) →
  Real.sqrt ((3 - Real.cos (Real.pi/9)^2) * (3 - Real.cos (2*Real.pi/9)^2) * (3 - Real.cos (4*Real.pi/9)^2)) = 5 * Real.sqrt 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_sqrt_l1208_120804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_nine_percent_l1208_120881

/-- Calculates the interest rate given the principal, time, and total simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that for the given values, the interest rate is 9% -/
theorem interest_rate_is_nine_percent :
  let principal : ℝ := 8945
  let time : ℝ := 5
  let simple_interest : ℝ := 4025.25
  calculate_interest_rate principal time simple_interest = 9 := by
  -- Unfold the definition of calculate_interest_rate
  unfold calculate_interest_rate
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_nine_percent_l1208_120881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_test_l1208_120889

/-- Check if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The sets of numbers given in the problem --/
def set_A : (ℝ × ℝ × ℝ) := (7, 24, 25)
noncomputable def set_B : (ℝ × ℝ × ℝ) := (Real.sqrt 41, 4, 5)
noncomputable def set_C : (ℝ × ℝ × ℝ) := (5/4, 1, 3/4)
def set_D : (ℝ × ℝ × ℝ) := (40, 50, 60)

theorem right_triangle_test :
  is_right_triangle set_A.1 set_A.2.1 set_A.2.2 ∧
  is_right_triangle set_B.1 set_B.2.1 set_B.2.2 ∧
  is_right_triangle set_C.1 set_C.2.1 set_C.2.2 ∧
  ¬is_right_triangle set_D.1 set_D.2.1 set_D.2.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_test_l1208_120889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_parallelism_l1208_120850

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- Define a membership relation for lines and planes
variable (in_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_parallelism 
  (m n : Line) (α : Plane) 
  (h_outside_m : ¬ in_plane m α) 
  (h_outside_n : ¬ in_plane n α) :
  ((parallel m n ∧ parallel_plane m α) → parallel_plane n α) ∧
  ((parallel m n ∧ parallel_plane n α) → parallel_plane m α) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_parallelism_l1208_120850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_when_a_is_one_zero_point_iff_a_positive_l1208_120899

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * (4 : ℝ) ^ x - (2 : ℝ) ^ x - 1

-- Theorem 1: When a = 1, f(x) has a zero point at x = 0
theorem zero_point_when_a_is_one :
  f 1 0 = 0 :=
by sorry

-- Theorem 2: f(x) has a zero point if and only if a > 0
theorem zero_point_iff_a_positive (a : ℝ) :
  (∃ x : ℝ, f a x = 0) ↔ a > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_when_a_is_one_zero_point_iff_a_positive_l1208_120899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_exceeding_100a1_l1208_120814

/-- An arithmetic sequence with positive first term and common difference,
    where a_2, a_5, and a_9 form a geometric sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : ∀ n, a (n + 1) = a n + d
  h2 : 0 < a 1
  h3 : 0 < d
  h4 : (a 2) * (a 9) = (a 5) * (a 5)

/-- The sum of the first k terms of the arithmetic sequence -/
def sumFirstK (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  (k : ℚ) * (2 * seq.a 1 + (k - 1) * seq.d) / 2

/-- The theorem statement -/
theorem smallest_k_exceeding_100a1 (seq : ArithmeticSequence) :
  (∀ k < 34, sumFirstK seq k ≤ 100 * seq.a 1) ∧
  (sumFirstK seq 34 > 100 * seq.a 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_exceeding_100a1_l1208_120814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_from_unique_squares_cannot_fill_aquarium_l1208_120820

-- Part a
def is_rectangle_possible (squares : List ℕ) : Prop :=
  ∃ (width height : ℕ), width * height = squares.sum

theorem rectangle_from_unique_squares :
  ∃ (squares : List ℕ), (∀ x, x ∈ squares → x > 0) ∧ 
  (∀ x y, x ∈ squares → y ∈ squares → x ≠ y → squares.count x = 1 ∧ squares.count y = 1) ∧
  is_rectangle_possible squares :=
sorry

-- Part b
structure Cube where
  length : ℕ
  width : ℕ
  height : ℕ

def volume (c : Cube) : ℕ :=
  c.length * c.width * c.height

structure Aquarium where
  length : ℕ
  width : ℕ
  height : ℕ

def fits (c : Cube) (a : Aquarium) : Prop :=
  c.length ≤ a.length ∧ c.width ≤ a.width ∧ c.height ≤ a.height

theorem cannot_fill_aquarium (a : Aquarium) :
  ¬∃ (cubes : List Cube), 
    (∀ c, c ∈ cubes → fits c a) ∧
    (∀ c1 c2, c1 ∈ cubes → c2 ∈ cubes → c1 ≠ c2 → volume c1 ≠ volume c2) ∧
    (cubes.map volume).sum = a.length * a.width * a.height :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_from_unique_squares_cannot_fill_aquarium_l1208_120820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_l1208_120870

/-- A convex solid bounded by 12 equilateral triangles and 2 regular hexagons -/
structure ConvexSolid where
  /-- The solid is convex -/
  is_convex : Bool
  /-- The solid is bounded by 12 equilateral triangles -/
  num_triangles : Nat
  /-- The solid is bounded by 2 regular hexagons -/
  num_hexagons : Nat
  /-- Each triangle has a side length of 1 unit -/
  triangle_side_length : ℝ
  /-- The planes of the hexagons are parallel -/
  hexagon_planes_parallel : Bool

/-- The radius of the circumscribed sphere -/
noncomputable def sphere_radius : ℝ := (1/2) * Real.sqrt (3 + Real.sqrt 3)

/-- A point is on the solid -/
def point_on_solid (s : ConvexSolid) (point : ℝ × ℝ × ℝ) : Prop := sorry

/-- Distance between two points in 3D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Theorem stating that a sphere can be circumscribed around the solid with the given radius -/
theorem circumscribed_sphere (s : ConvexSolid) : 
  s.is_convex ∧ 
  s.num_triangles = 12 ∧ 
  s.num_hexagons = 2 ∧ 
  s.triangle_side_length = 1 ∧ 
  s.hexagon_planes_parallel → 
  ∃ (center : ℝ × ℝ × ℝ), ∀ (point : ℝ × ℝ × ℝ), 
    point_on_solid s point → 
    distance center point = sphere_radius :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_l1208_120870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_max_value_one_l1208_120883

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 2 + a * Real.cos x + a

-- State the theorem
theorem exists_a_max_value_one :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 Real.pi, f a x ≤ 1) ∧
           (∃ x ∈ Set.Icc 0 Real.pi, f a x = 1) ∧
           a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_max_value_one_l1208_120883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_large_subset_l1208_120827

theorem divisibility_in_large_subset :
  ∀ (S : Finset ℕ), 
    (∀ n ∈ S, n ≥ 1 ∧ n ≤ 2012) →
    (Finset.card S ≥ 1000) →
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (2 * a) % b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_large_subset_l1208_120827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_composite_l1208_120882

def append_three (n : ℕ) : ℕ := n * 10 + 3

theorem eventually_composite (n : ℕ) : 
  ∃ k : ℕ, ∃ m : ℕ, m > 1 ∧ m < (Nat.iterate append_three k n) ∧ (Nat.iterate append_three k n) % m = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_composite_l1208_120882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divided_by_next_largest_remainder_l1208_120833

def numbers : List Nat := [10, 11, 12, 13]

theorem largest_divided_by_next_largest_remainder :
  (numbers.maximum?.getD 0) % (numbers.filter (· < numbers.maximum?.getD 0)).maximum?.getD 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divided_by_next_largest_remainder_l1208_120833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_md_eq_nd_l1208_120896

-- Define the triangle ABC and other points
variable (A B C D M N : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : ‖B - A‖ < ‖C - A‖)
variable (h2 : angle A B C = 2 * angle C A B)
variable (h3 : D ∈ Seg A C)
variable (h4 : ‖C - D‖ = ‖B - A‖)
variable (h5 : (Line.through B M).parallel (Line.through A C))
variable (h6 : M ∈ angleBisectorExt A B C)
variable (h7 : (Line.through C N).parallel (Line.through A B))

-- State the theorem
theorem md_eq_nd : ‖M - D‖ = ‖N - D‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_md_eq_nd_l1208_120896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_seven_halves_l1208_120860

/-- Configuration of squares and points for the triangle ABC problem -/
structure SquareConfiguration where
  /-- Point A: bottom-right corner of the first square -/
  A : ℝ × ℝ
  /-- Point B: top-left corner of the last square -/
  B : ℝ × ℝ
  /-- Point C: top-right corner of the diagonally adjacent square -/
  C : ℝ × ℝ
  /-- Ensure A is at (0, 0) -/
  A_origin : A = (0, 0)
  /-- Ensure B is at (4, 1) based on four unit squares in a row -/
  B_position : B = (4, 1)
  /-- Ensure C is at (1, 2) based on the diagonal placement -/
  C_position : C = (1, 2)

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem: The area of triangle ABC in the given configuration is 7/2 -/
theorem triangle_area_is_seven_halves (config : SquareConfiguration) :
  triangleArea config.A config.B config.C = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_seven_halves_l1208_120860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_number_probability_guess_probability_l1208_120816

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10

def satisfies_conditions (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  is_prime (tens_digit n) ∧
  is_prime (units_digit n) ∧
  units_digit n < 5 ∧
  n ≥ 40 ∧ n < 90

-- We need to make this function computable
def satisfies_conditions_decidable (n : ℕ) : Bool :=
  n ≥ 10 ∧ n < 100 ∧
  n ≥ 40 ∧ n < 90 ∧
  (tens_digit n = 5 ∨ tens_digit n = 7) ∧
  (units_digit n = 2 ∨ units_digit n = 3)

theorem secret_number_probability :
  (Finset.filter (λ n => satisfies_conditions_decidable n) (Finset.range 100)).card = 4 :=
sorry

theorem guess_probability :
  (1 : ℚ) / (Finset.filter (λ n => satisfies_conditions_decidable n) (Finset.range 100)).card = 1/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_number_probability_guess_probability_l1208_120816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_seating_arrangements_l1208_120858

theorem teacher_seating_arrangements (n m : ℕ) : 
  n = 10 ∧ m = 25 → 
  2 * (Nat.choose (m - n) (n - 1)) = 10010 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_seating_arrangements_l1208_120858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_symmetry_about_pi_l1208_120805

/-- The sine function is symmetric about the line x = π -/
theorem sin_symmetry_about_pi : ∀ x : ℝ, Real.sin x = Real.sin (2 * Real.pi - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_symmetry_about_pi_l1208_120805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l1208_120846

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (2 - 3*t, Real.sqrt 3 * t)

-- Define curve C₁
def curve_C₁ (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define curve C₂
def curve_C₂ : ℝ := Real.pi / 6

-- Define point A (intersection of l and C₂)
def point_A : ℝ × ℝ := 
  let t := (2 - Real.sqrt 3 * Real.tan (Real.pi/6)) / 3
  line_l t

-- Define point B (intersection of C₁ and C₂)
def point_B : ℝ × ℝ := 
  let ρ := curve_C₁ curve_C₂
  (ρ * Real.cos curve_C₂, ρ * Real.sin curve_C₂)

-- Theorem statement
theorem distance_AB : 
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 4 * Real.sqrt 3 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l1208_120846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_power_of_two_with_specific_digits_l1208_120836

/-- Represents the digits of a natural number as a list of natural numbers. -/
def digits : ℕ → List ℕ := sorry

/-- Counts the number of digits in a natural number. -/
def num_digits : ℕ → ℕ := sorry

/-- 
Theorem: For any natural number n, there exists a natural number k such that:
1. All digits of k are either 1 or 2
2. The number of digits of k is equal to n
3. k is divisible by 2^n
-/
theorem multiple_of_power_of_two_with_specific_digits (n : ℕ) :
  ∃ (k : ℕ), (∀ d : ℕ, d ∈ digits k → d = 1 ∨ d = 2) ∧
             (num_digits k = n) ∧
             (k % 2^n = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_power_of_two_with_specific_digits_l1208_120836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l1208_120837

/-- The average age of a cricket team given specific conditions about some players' ages. -/
theorem cricket_team_average_age : ∀ (team_size : ℕ) (captain_age : ℕ) (A : ℚ),
  team_size = 15 →
  captain_age = 28 →
  let wicket_keeper_age := captain_age + 4;
  let star_batsman_age := captain_age - 2;
  let lead_bowler_age := captain_age + 6;
  let remaining_players := team_size - 4;
  (A * ↑team_size) = ↑(captain_age + wicket_keeper_age + star_batsman_age + lead_bowler_age) + 
    (A - 1) * ↑remaining_players →
  A = 109 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l1208_120837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l1208_120828

-- Define IsTriangle as a predicate on Set Point
def IsTriangle (triangle : Set Point) : Prop := sorry

theorem triangle_angle_calculation (triangle1 triangle2 : Set Point) 
  (common_side : Set Point) 
  (angle1_t1 : ℝ) 
  (sum_angles2_3_t1 : ℝ) 
  (angle3_t1 : ℝ) 
  (angle1_t2 : ℝ) 
  (angle2_t2 angle3_t2 : ℝ) :
  IsTriangle triangle1 ∧ 
  IsTriangle triangle2 ∧ 
  common_side ⊆ triangle1 ∧ 
  common_side ⊆ triangle2 ∧ 
  angle1_t1 = 50 ∧ 
  sum_angles2_3_t1 = 120 ∧ 
  angle3_t1 = 50 ∧ 
  angle1_t2 = angle3_t1 ∧ 
  angle2_t2 = angle3_t2 →
  angle2_t2 = 65 ∧ angle3_t2 = 65 := by
  sorry

#check triangle_angle_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l1208_120828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1208_120877

/-- The sum of the series 2 + 6x + 10x^2 + 14x^3 + ... -/
noncomputable def seriesSum (x : ℝ) : ℝ := 2 + (6 * x) / (1 - x)

theorem unique_solution :
  ∃! x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 1 ∧ seriesSum x = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1208_120877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_of_f_l1208_120892

noncomputable def f (x : ℝ) : ℝ := (x^3 + 2*x^2 + 9*x - 18) / (x^2 - 5*x + 6)

theorem vertical_asymptotes_of_f :
  ∃ (a b : ℝ), a ≠ b ∧ 
  (∀ x, x^2 - 5*x + 6 = 0 ↔ (x = a ∨ x = b)) ∧
  (f a = 0/0) ∧ (f b = 0/0) ∧
  (∀ x, x ≠ a → x ≠ b → f x ≠ 0/0) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_of_f_l1208_120892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1208_120829

-- Define constants
noncomputable def train1_length : ℝ := 210
noncomputable def train1_speed : ℝ := 120
noncomputable def train2_length : ℝ := 290.04
noncomputable def crossing_time : ℝ := 9

-- Define conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

-- Theorem statement
theorem train_speed_calculation :
  let total_length := train1_length + train2_length
  let relative_speed := total_length / crossing_time
  let train1_speed_m_s := train1_speed * km_hr_to_m_s
  let train2_speed_m_s := relative_speed - train1_speed_m_s
  let train2_speed_km_hr := train2_speed_m_s / km_hr_to_m_s
  ∃ (ε : ℝ), abs (train2_speed_km_hr - 399.14) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1208_120829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_with_slopes_from_quadratic_l1208_120811

theorem angle_between_lines_with_slopes_from_quadratic : 
  ∃ (k₁ k₂ : ℝ), (6 * k₁^2 + k₁ - 1 = 0) ∧ 
                 (6 * k₂^2 + k₂ - 1 = 0) ∧ 
                 (k₁ ≠ k₂) ∧
                 Real.arctan ((k₂ - k₁) / (1 + k₁ * k₂)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_with_slopes_from_quadratic_l1208_120811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_with_corner_circles_l1208_120834

theorem square_area_with_corner_circles (r : ℝ) (h : r = 7) : 
  (2 * r) ^ 2 = 196 := by
  -- Substitute r with 7
  rw [h]
  -- Simplify the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_with_corner_circles_l1208_120834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1208_120842

theorem trigonometric_inequality : 
  Real.cos (61 * Real.pi / 180) * Real.cos (127 * Real.pi / 180) + Real.cos (29 * Real.pi / 180) * Real.cos (37 * Real.pi / 180) < 
  Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2) ∧
  Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2) < 
  (2 * Real.tan (13 * Real.pi / 180)) / (1 + Real.tan (13 * Real.pi / 180)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1208_120842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1208_120817

noncomputable def f (φ : Real) (x : Real) : Real := Real.sin (2 * x + φ)

noncomputable def g (φ : Real) (x : Real) : Real := f φ (x + Real.pi / 6)

theorem phi_value (φ : Real) 
  (h1 : 0 < φ) (h2 : φ < Real.pi) 
  (h3 : ∀ x, g φ x = g φ (-x)) : 
  φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1208_120817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1208_120859

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 4)
  (h_sum : a 6 + a 8 = 18) :
  (∀ n : ℕ, a n = 2 + n) ∧
  (∀ n : ℕ, (Finset.range n).sum (λ i ↦ 1 / ((i + 1 : ℝ) * a (i + 1))) = 3/4 - (2*n + 3)/(2*(n + 1)*(n + 2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1208_120859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_ten_l1208_120872

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else 2*x + 1

-- Theorem statement
theorem f_equals_ten (x : ℝ) : f x = 10 ↔ x = -3 ∨ x = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_ten_l1208_120872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1208_120879

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  M_on_y_axis : M.1 = 0
  N_on_x_axis : N.2 = 0
  M_midpoint : M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  N_midpoint : N = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

/-- Theorem about the properties of Triangle ABC -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = (-1, 2))
  (h2 : t.B = (4, 3)) :
  t.C = (1, -3) ∧ 
  ∃ (a b : ℝ), a * (t.M.1 - t.N.1) + b * (t.M.2 - t.N.2) = 0 ∧ 
                a = 2 ∧ b = -10 ∧ 
                2 * t.M.1 - 10 * t.M.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1208_120879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficientPaintersDays_eq_three_l1208_120825

/-- The number of work-days needed for two painters to complete a job, where one painter is twice as efficient as the other, given that five painters of standard efficiency can complete the job in 1.8 work-days. -/
noncomputable def efficientPaintersDays : ℝ :=
  let standardRate : ℝ := 1  -- Standard painter's rate
  let efficientRate : ℝ := 2 * standardRate  -- Efficient painter's rate
  let totalWork : ℝ := 5 * standardRate * 1.8  -- Total work to be done
  let combinedRate : ℝ := standardRate + efficientRate  -- Combined rate of the two painters
  totalWork / combinedRate

theorem efficientPaintersDays_eq_three :
  efficientPaintersDays = 3 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficientPaintersDays_eq_three_l1208_120825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_range_l1208_120863

-- Define the equation
noncomputable def f (m x : ℝ) : ℝ := (2 : ℝ)^(2*x) - (m - 1)*(2 : ℝ)^x + 2

-- Define the range of x
def X : Set ℝ := Set.Icc 0 2

-- Define the range of m
noncomputable def M : Set ℝ := Set.union (Set.Ioo 4 (11/2)) {1 + 2*Real.sqrt 2}

-- Statement of the theorem
theorem unique_solution_range :
  ∀ m : ℝ, (∃! x, x ∈ X ∧ f m x = 0) ↔ m ∈ M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_range_l1208_120863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_with_distance_sqrt2_over_2_main_theorem_l1208_120838

noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  (|a * x + b * y + c|) / Real.sqrt (a^2 + b^2)

def line_with_slope_one (m : ℝ) : ℝ → ℝ := λ x ↦ x + m

theorem lines_with_distance_sqrt2_over_2 :
  ∀ m : ℝ, distance_point_to_line 0 0 1 (-1) (-m) = Real.sqrt 2 / 2 ↔ m = 1 ∨ m = -1 :=
by sorry

theorem main_theorem :
  ∀ a b c : ℝ,
    (∃ m : ℝ, (∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = line_with_slope_one m x)) ∧
    distance_point_to_line 0 0 a b c = Real.sqrt 2 / 2 ↔
    (a = 1 ∧ b = -1 ∧ c = 1) ∨ (a = 1 ∧ b = -1 ∧ c = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_with_distance_sqrt2_over_2_main_theorem_l1208_120838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l1208_120857

noncomputable def g (x : ℝ) : ℝ := ⌊2 * x⌋ + (1 / 3)

theorem g_neither_even_nor_odd : 
  ¬(∀ x : ℝ, g x = g (-x)) ∧ ¬(∀ x : ℝ, g x = -g (-x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l1208_120857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_l1208_120853

/-- Given an angle α whose terminal side lies on the line y = -2x, 
    prove that tan α = -2 and cos(2α + 3π/2) = -4/5 -/
theorem angle_on_line (α : ℝ) 
  (h : ∃ (x y : ℝ), y = -2 * x ∧ (∃ (r : ℝ), x = r * Real.cos α ∧ y = r * Real.sin α)) : 
  Real.tan α = -2 ∧ Real.cos (2 * α + 3 * Real.pi / 2) = -4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_l1208_120853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_top_width_l1208_120809

/-- Represents the cross-section of a water channel -/
structure WaterChannel where
  bottomWidth : ℝ
  topWidth : ℝ
  depth : ℝ
  area : ℝ

/-- Calculates the area of a trapezoidal cross-section -/
noncomputable def trapezoidArea (channel : WaterChannel) : ℝ :=
  (channel.bottomWidth + channel.topWidth) * channel.depth / 2

theorem water_channel_top_width 
  (channel : WaterChannel)
  (h1 : channel.bottomWidth = 8)
  (h2 : channel.depth = 70)
  (h3 : channel.area = 770)
  (h4 : trapezoidArea channel = channel.area) :
  channel.topWidth = 14 := by
  sorry

#eval "Lake build successful"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_channel_top_width_l1208_120809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_13_plus_1_parts_l1208_120880

theorem sqrt_13_plus_1_parts : ∃ (a : ℤ) (b : ℝ), 
  (Real.sqrt 13 : ℝ) + 1 = a + b ∧ a = 4 ∧ b = Real.sqrt 13 - 3 ∧ 0 ≤ b ∧ b < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_13_plus_1_parts_l1208_120880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_A_to_B_l1208_120812

/-- The ratio of speeds between two runners A and B -/
noncomputable def speed_ratio (speed_A speed_B : ℝ) : ℝ := speed_A / speed_B

/-- The theorem stating the speed ratio between runners A and B -/
theorem speed_ratio_A_to_B :
  ∀ (speed_A speed_B : ℝ),
  speed_A > 0 →
  speed_B > 0 →
  (200 / speed_A = 120 / speed_B) →
  speed_ratio speed_A speed_B = 5 / 3 :=
by
  intros speed_A speed_B h_A h_B h_eq
  unfold speed_ratio
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check speed_ratio_A_to_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_A_to_B_l1208_120812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_triangle_ratio_bounds_l1208_120810

theorem abc_triangle_ratio_bounds (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  A = 2*B ∧
  a = b * Real.sin A / Real.sin B →
  Real.sqrt 2 < a/b ∧ a/b < Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_triangle_ratio_bounds_l1208_120810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_two_tan_pi_fifth_l1208_120835

theorem tan_alpha_equals_two_tan_pi_fifth (α : ℝ) :
  Real.tan α = 2 * Real.tan (π / 5) →
  (Real.cos (α - 3 * π / 10)) / (Real.sin (α - π / 5)) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_two_tan_pi_fifth_l1208_120835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1208_120841

/-- Geometric sequence with first term 1 and common ratio q -/
def geometricSequence (q : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => q * geometricSequence q n

/-- Theorem: In a geometric sequence with a₁ = 1 and common ratio q ≠ ±1,
    if aₖ = a₂ * a₅, then k = 6 -/
theorem geometric_sequence_property (q : ℝ) (k : ℕ) 
    (hq : q ≠ 1 ∧ q ≠ -1) 
    (h : geometricSequence q k = geometricSequence q 2 * geometricSequence q 5) :
  k = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1208_120841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_five_relatively_prime_l1208_120823

/-- The set S containing integers from 1 to 280 -/
def S : Finset ℕ := Finset.range 280

/-- A function that checks if a list of natural numbers are pairwise relatively prime -/
def are_pairwise_relatively_prime (list : List ℕ) : Prop :=
  ∀ i j, i ≠ j → i < list.length → j < list.length → 
    Nat.gcd (list.getD i 0) (list.getD j 0) = 1

/-- The main theorem statement -/
theorem smallest_n_with_five_relatively_prime : 
  (∀ T : Finset ℕ, T ⊆ S → T.card = 217 → 
    ∃ list : List ℕ, list.toFinset ⊆ T ∧ list.length = 5 ∧ are_pairwise_relatively_prime list) ∧
  (∀ m : ℕ, m < 217 → 
    ∃ T : Finset ℕ, T ⊆ S ∧ T.card = m ∧
      ∀ list : List ℕ, list.toFinset ⊆ T → list.length = 5 → ¬are_pairwise_relatively_prime list) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_five_relatively_prime_l1208_120823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_survive_formula_l1208_120873

/-- The probability that exactly 2 out of 3 seedlings survive, given a survival rate p -/
def prob_two_survive (p : ℝ) : ℝ :=
  Nat.choose 3 2 * p^2 * (1 - p)

/-- Theorem stating that the probability of exactly 2 out of 3 seedlings surviving
    is equal to C₃² * p² * (1-p), where p is the survival rate of a seedling -/
theorem prob_two_survive_formula (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  prob_two_survive p = Nat.choose 3 2 * p^2 * (1 - p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_survive_formula_l1208_120873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_weight_l1208_120866

noncomputable def average_weight_whole_class (num_students_A num_students_B : ℕ) 
                               (avg_weight_A avg_weight_B : ℝ) : ℝ :=
  ((num_students_A : ℝ) * avg_weight_A + (num_students_B : ℝ) * avg_weight_B) / 
  ((num_students_A + num_students_B) : ℝ)

theorem class_average_weight :
  let num_students_A : ℕ := 50
  let num_students_B : ℕ := 50
  let avg_weight_A : ℝ := 60
  let avg_weight_B : ℝ := 80
  average_weight_whole_class num_students_A num_students_B avg_weight_A avg_weight_B = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_weight_l1208_120866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_sqrt_l1208_120856

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem tangent_line_of_sqrt :
  -- The slope of the tangent line at (4, 2) is 1/4
  (deriv f) 4 = 1/4 ∧
  -- The equation of the tangent line is x - 4y + 4 = 0
  ∀ (x y : ℝ), (x - 4*y + 4 = 0) ↔ (y - 2 = (1/4) * (x - 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_sqrt_l1208_120856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_triangular_cross_section_l1208_120887

/-- A cube is a three-dimensional shape with six square faces. --/
structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A triangle is a two-dimensional shape with three straight sides and three angles. --/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side1_pos : side1 > 0
  side2_pos : side2 > 0
  side3_pos : side3 > 0

/-- A cross-section is a 2D shape formed by the intersection of a plane and a 3D object. --/
def CrossSection (shape : Type) := shape → Option Triangle

/-- Theorem: A cube can have a triangular cross-section. --/
theorem cube_triangular_cross_section : ∃ (c : Cube), ∃ (cs : CrossSection Cube), cs c ≠ none := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_triangular_cross_section_l1208_120887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_terms_coprime_l1208_120808

-- Define the sequence
def a : ℕ → ℤ
  | 0 => 100  -- Define for 0 to cover all natural numbers
  | n + 1 => (a n)^2 - (a n) + 1

-- State the theorem
theorem sequence_terms_coprime (n m : ℕ) (h : n ≠ m) : Int.gcd (a n) (a m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_terms_coprime_l1208_120808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1208_120895

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos (3 * x - Real.pi / 3)

-- State the theorem
theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1208_120895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_c_min_price_l1208_120831

/-- Represents a company's product line -/
structure ProductLine where
  total_products : ℕ
  average_price : ℚ
  products_under_1000 : ℕ
  max_price : ℚ

/-- The minimum selling price for any product in the product line -/
noncomputable def min_selling_price (pl : ProductLine) : ℚ :=
  (pl.total_products * pl.average_price - pl.max_price) / (pl.total_products - 1)

/-- Theorem stating the minimum selling price for Company C's product line -/
theorem company_c_min_price :
  let pl : ProductLine := {
    total_products := 25,
    average_price := 1200,
    products_under_1000 := 10,
    max_price := 12000
  }
  min_selling_price pl = 750 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_c_min_price_l1208_120831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l1208_120845

-- Define the power function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 2*m - 2) * x^(m + 1/2*m^2)

-- State the theorem
theorem power_function_increasing (m : ℝ) :
  (∀ x > 0, Monotone (fun x => f m x)) ↔ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l1208_120845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temperature_approximation_l1208_120813

def orlando_temps : List ℝ := [55, 62, 58, 65, 54, 60, 56,
                               70, 74, 71, 77, 64, 68, 72,
                               82, 85, 89, 73, 65, 63, 67,
                               75, 72, 60, 57, 50, 55, 58,
                               69, 67, 70]

def austin_temps : List ℝ := [58, 56, 65, 69, 64, 71, 67,
                              74, 77, 72, 74, 67, 66, 77,
                              88, 82, 79, 76, 69, 60, 67,
                              75, 71, 60, 58, 55, 53, 61,
                              65, 63, 67]

def denver_temps : List ℝ := [40, 48, 50, 60, 52, 56, 70,
                              66, 74, 69, 72, 59, 61, 65,
                              78, 72, 85, 69, 58, 57, 63,
                              72, 68, 56, 60, 50, 49, 53,
                              60, 65, 62]

def all_temps : List ℝ := orlando_temps ++ austin_temps ++ denver_temps

theorem average_temperature_approximation :
  abs ((List.sum all_temps) / (List.length all_temps) - 65.45) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temperature_approximation_l1208_120813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_value_l1208_120855

theorem tan_sum_value (x y : Real) 
  (h1 : Real.sin x + Real.sin y = 85 / 65) 
  (h2 : Real.cos x + Real.cos y = 60 / 65) : 
  Real.tan x + Real.tan y = -408 / 145 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_value_l1208_120855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_time_to_max_distance_l1208_120807

/-- The time it takes for the first runner to complete a lap -/
noncomputable def t₁ : ℝ := 20

/-- The time it takes for the second runner to complete a lap -/
noncomputable def t₂ : ℝ := 28

/-- The relative angular speed of the runners -/
noncomputable def relativeSpeed : ℝ := 1 / t₁ - 1 / t₂

/-- The time it takes for the runners to reach maximum distance -/
noncomputable def maxDistanceTime : ℝ := 1 / (2 * relativeSpeed)

/-- Theorem stating that the shortest time to reach maximum distance is 35 seconds -/
theorem shortest_time_to_max_distance : maxDistanceTime = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_time_to_max_distance_l1208_120807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1208_120800

def has_solutions_with_difference_4 (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ 
    (x^2 - a*x - x + a) * Real.sqrt (x + 5) ≤ 0 ∧
    (y^2 - a*y - y + a) * Real.sqrt (y + 5) ≤ 0 ∧
    |x - y| = 4

theorem solution_set_characterization :
  {a : ℝ | has_solutions_with_difference_4 a} = Set.Ici (-1) ∪ Set.Iic 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1208_120800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_in_triangular_prism_l1208_120806

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  A : Point3D
  B : Point3D
  C : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Calculates the dot product of two vectors -/
def dotProduct (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

/-- Calculates the magnitude of a vector -/
noncomputable def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2 + v.z^2)

/-- Define vector subtraction for Point3D -/
instance : HSub Point3D Point3D Point3D where
  hSub a b := ⟨a.x - b.x, a.y - b.y, a.z - b.z⟩

/-- Theorem about the cosine of the angle between two lines in a triangular prism -/
theorem cosine_angle_in_triangular_prism (prism : TriangularPrism) :
  distance prism.A prism.B = 2 →
  distance prism.A prism.C = 1 →
  distance prism.A prism.A1 = 2 →
  dotProduct (prism.B - prism.A) (prism.C - prism.A) = 0 →
  dotProduct (prism.A1 - prism.A) (prism.B - prism.A) = 0 →
  dotProduct (prism.A1 - prism.A) (prism.C - prism.A) = 0 →
  let AB1 : Point3D := prism.B1 - prism.A
  let A1C : Point3D := prism.C - prism.A1
  abs (dotProduct AB1 A1C) / (magnitude AB1 * magnitude A1C) = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_in_triangular_prism_l1208_120806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_wave_amplitude_l1208_120847

noncomputable def y₁ (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y₂ (t : ℝ) : ℝ := 3 * Real.cos (100 * Real.pi * t + Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y₁ t + y₂ t

theorem combined_wave_amplitude :
  ∃ A : ℝ, ∀ t : ℝ, y t = A * Real.sin (100 * Real.pi * t + Real.pi / 4) ∧ A = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_wave_amplitude_l1208_120847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compatriot_game_every_round_l1208_120848

/-- Represents a chess tournament with the given properties -/
structure ChessTournament where
  /-- The number of participants in the tournament -/
  participants : Nat
  /-- The number of rounds in the tournament -/
  rounds : Nat
  /-- The number of games each participant plays -/
  games_per_participant : Nat
  /-- The minimum number of games each participant plays against compatriots -/
  min_compatriot_games : Nat
  /-- Proof that the number of participants is 10 -/
  participants_count : participants = 10
  /-- Proof that each participant plays against every other participant exactly once -/
  all_play_all : games_per_participant = participants - 1
  /-- Proof that at least half of all games involve compatriots -/
  half_compatriot_games : min_compatriot_games ≥ games_per_participant / 2
  /-- Proof that the number of rounds is equal to the number of games per participant -/
  rounds_count : rounds = games_per_participant

/-- Predicate indicating whether a game in a given round is between compatriots -/
def IsCompatriotGame (t : ChessTournament) (r : Fin t.rounds) (g : Fin (t.participants / 2)) : Prop :=
  sorry

/-- Theorem stating that in every round of the tournament, there is at least one game between compatriots -/
theorem compatriot_game_every_round (t : ChessTournament) :
  ∀ r : Fin t.rounds, ∃ g : Fin (t.participants / 2), IsCompatriotGame t r g :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compatriot_game_every_round_l1208_120848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_b_sin_C_l1208_120888

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 3 ∧ t.b^2 + t.c^2 = 3 + t.b * t.c

-- Theorem for part I
theorem angle_A_measure (t : Triangle) (h : triangle_conditions t) : t.A = π / 3 :=
sorry

-- Theorem for part II
theorem max_b_sin_C (t : Triangle) (h : triangle_conditions t) : 
  ∀ (x : ℝ), t.b * Real.sin t.C ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_b_sin_C_l1208_120888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_product_l1208_120815

-- Define necessary functions
def area : Set (ℝ × ℝ) → ℝ := sorry
def is_inscribed : Set (ℝ × ℝ) → Set (ℝ × ℝ) → Prop := sorry
def vertices : Set (ℝ × ℝ) → Set (ℝ × ℝ) := sorry
def sides : Set (ℝ × ℝ) → Set (Set (ℝ × ℝ)) := sorry
def segment_lengths : Set (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem inscribed_squares_product (a b : ℝ) : 
  (∃ (small_square large_square : Set (ℝ × ℝ)),
    -- Small square has area 4
    (area small_square = 4) ∧
    -- Large square has area 5
    (area large_square = 5) ∧
    -- Small square is inscribed in large square
    (is_inscribed small_square large_square) ∧
    -- Each vertex of small square divides a side of large square
    (∀ v ∈ vertices small_square, ∃ s ∈ sides large_square, v ∈ s) ∧
    -- A vertex divides a side into segments of length a and b
    (∃ v ∈ vertices small_square, ∃ s ∈ sides large_square, 
      segment_lengths s v = (a, b))) →
  a * b = 1/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_product_l1208_120815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l1208_120861

noncomputable section

-- Define the quadratic function f
def f : ℝ → ℝ := λ x ↦ x^2 + 4*x - 2

-- Define g in terms of f
def g : ℝ → ℝ := λ x ↦ if x ≠ 0 then f x / x else 0

theorem quadratic_function_proof :
  -- The graph of f(x) intersects y = -6 at only one point
  (∃! x, f x = -6) ∧
  -- f(0) = -2
  (f 0 = -2) ∧
  -- f(x-2) is an even function
  (∀ x, f (x - 2) = f (-x - 2)) ∧
  -- g(x) = f(x)/x for x ≠ 0
  (∀ x ≠ 0, g x = f x / x) →
  -- Conclusion: f(x) = x^2 + 4x - 2
  ∀ x, f x = x^2 + 4*x - 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l1208_120861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_negative_four_l1208_120854

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 8 * x - 10) / (x^2 - 5 * x + 6)

/-- The horizontal asymptote of g(x) -/
def horizontal_asymptote : ℝ := 3

/-- The point where g(x) crosses its horizontal asymptote -/
def crossing_point : ℝ := -4

/-- Theorem: g(x) crosses its horizontal asymptote at x = -4 -/
theorem g_crosses_asymptote_at_negative_four :
  g crossing_point = horizontal_asymptote := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_negative_four_l1208_120854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowdrift_melting_theorem_l1208_120876

/-- Represents the melting of a snowdrift over time -/
structure Snowdrift where
  initial_height : ℕ
  melting_rate : ℕ → ℕ
  total_melting_time : ℕ

/-- The fraction of snowdrift height melted after half the total melting time -/
def fraction_melted_at_half_time (s : Snowdrift) : ℚ :=
  let half_time := s.total_melting_time / 2
  let height_melted := (List.range half_time).map (fun k => s.melting_rate (k + 1)) |>.sum
  height_melted / s.initial_height

/-- The specific snowdrift described in the problem -/
def problem_snowdrift : Snowdrift where
  initial_height := 468
  melting_rate := fun k => 6 * k
  total_melting_time := 12  -- This is derived from the solution, but could be calculated

theorem snowdrift_melting_theorem :
  fraction_melted_at_half_time problem_snowdrift = 7 / 26 := by
  sorry

#eval fraction_melted_at_half_time problem_snowdrift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowdrift_melting_theorem_l1208_120876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1208_120821

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + b * x + c

noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := x^2 - a * x + b

theorem cubic_function_properties (a b c : ℝ) (ha : a > 0) :
  (∀ x, f' a b x = 0 → x = 0 ∨ x = a) →
  (f a b c 0 = 1) →
  (f' a b 0 = 0) →
  (∃ x y, x ≠ y ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ ∀ z, f a b c z = 0 → z = x ∨ z = y) →
  b = 0 ∧ c = 1 ∧ a = 36 := by
  sorry

#check cubic_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1208_120821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_and_pure_imaginary_is_zero_l1208_120874

/-- A real number that is also a pure imaginary number must be zero. -/
theorem real_and_pure_imaginary_is_zero (a : ℂ) (h1 : a.re = 0) (h2 : ∃ b : ℝ, a = Complex.I * b) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_and_pure_imaginary_is_zero_l1208_120874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_result_independent_of_order_l1208_120839

noncomputable def harmonicMean (a b : ℝ) : ℝ := (a * b) / (a + b)

noncomputable def finalResult (numbers : List ℝ) : ℝ :=
  1 / (numbers.map (λ x => 1 / x)).sum

noncomputable def iteratedHarmonicMean (numbers : List ℝ) (order : List (Nat × Nat)) : ℝ :=
  sorry

theorem final_result_independent_of_order (numbers : List ℝ) (h : numbers.length ≥ 2) :
  ∀ (order : List (Nat × Nat)),
    (∀ (i j : Nat), (i, j) ∈ order → i < numbers.length ∧ j < numbers.length ∧ i ≠ j) →
    (order.length = numbers.length - 1) →
    (iteratedHarmonicMean numbers order = finalResult numbers) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_result_independent_of_order_l1208_120839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_equals_101_l1208_120878

noncomputable def floor (x : ℝ) := Int.floor x

noncomputable def b : ℤ := floor (100 * (11*77 + 12*78 + 13*79 + 14*80) / (11*76 + 12*77 + 13*78 + 14*79))

theorem b_equals_101 : b = 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_equals_101_l1208_120878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l1208_120830

-- Define the hyperbola equation
noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 81 = 1

-- Define the slope of asymptotes
noncomputable def asymptote_slope : ℝ := 3 / 4

-- Theorem statement
theorem hyperbola_asymptote_slope :
  ∀ x y : ℝ, hyperbola_equation x y → 
  ∃ m : ℝ, m = asymptote_slope ∧ 
  (∀ ε > 0, ∃ x₀ > 0, ∀ x > x₀, ∃ y, 
    hyperbola_equation x y ∧ 
    abs (y / x - m) < ε ∧ 
    abs (y / x + m) < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l1208_120830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_240_minus_g_120_eq_zero_l1208_120843

/-- Sum of all even positive divisors of n -/
def sum_even_divisors (n : ℕ+) : ℕ :=
  (Finset.filter (fun d => Even d) (Nat.divisors n)).sum id

/-- g(n) is the quotient of the sum of all even positive divisors of n divided by n -/
def g (n : ℕ+) : ℚ :=
  (sum_even_divisors n : ℚ) / n

/-- Theorem stating that g(240) - g(120) = 0 -/
theorem g_240_minus_g_120_eq_zero : g 240 - g 120 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_240_minus_g_120_eq_zero_l1208_120843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2pi_minus_alpha_l1208_120864

theorem tan_2pi_minus_alpha (α : Real) (h1 : Real.cos α = 5/13) (h2 : π < α ∧ α < 2*π) :
  Real.tan (2*π - α) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2pi_minus_alpha_l1208_120864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parallelepipeds_in_cube_l1208_120898

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the cube -/
def cube : Cuboid := ⟨6, 6, 6⟩

/-- Represents the parallelepiped -/
def small_parallelepiped : Cuboid := ⟨1, 1, 4⟩

/-- Predicate to check if a cuboid's faces are parallel to the cube's faces -/
def is_parallel_to_cube (c : Cuboid) : Prop :=
  c.length ≤ cube.length ∧ c.width ≤ cube.width ∧ c.height ≤ cube.height

/-- The maximum number of parallelepipeds that can fit in the cube -/
def max_parallelepipeds : ℕ := 52

/-- Function to check if a point is within a cuboid -/
def point_in_cuboid (x y z : ℕ) (c : Cuboid) : Prop :=
  x < c.length ∧ y < c.width ∧ z < c.height

/-- Theorem stating that the maximum number of parallelepipeds that can fit in the cube is 52 -/
theorem max_parallelepipeds_in_cube :
  ∀ (n : ℕ), n > max_parallelepipeds →
  ¬ (∃ (arrangement : Fin n → Cuboid),
    (∀ i, arrangement i = small_parallelepiped) ∧
    (∀ i, is_parallel_to_cube (arrangement i)) ∧
    (∀ i j, i ≠ j → ¬ (∃ x y z : ℕ,
      point_in_cuboid x y z cube ∧
      point_in_cuboid x y z (arrangement i) ∧
      point_in_cuboid x y z (arrangement j)))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parallelepipeds_in_cube_l1208_120898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1208_120867

theorem expression_equality : 1^2023 + Real.sqrt 4 - (-Real.sqrt 2) + ((-8 : ℝ) ^ (1/3 : ℝ)) = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1208_120867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_in_sequence_l1208_120893

def sequenceA (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  ∀ n, n ≥ 1 →
    (if ∃ m < n, a m = a n - 2 then a (n + 1) = a n + 3
    else a (n + 1) = a n - 2)

theorem perfect_squares_in_sequence (a : ℕ → ℕ) (h : sequenceA a) :
  ∀ k : ℕ, k > 0 → ∃ n : ℕ, a (n + 1) = k^2 ∧ a (n + 1) = a n + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_in_sequence_l1208_120893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l1208_120875

theorem alpha_range (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) (h3 : Real.sin α > Real.cos α) :
  ∃ k : ℤ, (α ∈ Set.Ioo ((2 * k + 1/2) * Real.pi) ((2 * k + 1) * Real.pi)) ∨
           (α ∈ Set.Ioo ((2 * k + 3/2) * Real.pi) ((2 * k + 2) * Real.pi)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l1208_120875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_2013_correct_max_parking_spaces_l1208_120826

-- Constants
def cars_2010 : ℕ := 144
def cars_2012 : ℕ := 225
def total_investment : ℕ := 250000
def cost_indoor : ℕ := 6000
def cost_outdoor : ℕ := 2000

-- Define the growth rate
noncomputable def growth_rate : ℚ := (cars_2012 : ℚ) / cars_2010 - 1

-- Define the number of cars in 2013
noncomputable def cars_2013 : ℕ := (((cars_2012 : ℚ) * (1 + growth_rate)).floor : ℤ).toNat

-- Define the constraints for parking spaces
def valid_parking_scheme (indoor outdoor : ℕ) : Prop :=
  indoor * cost_indoor + outdoor * cost_outdoor ≤ total_investment ∧
  3 * indoor ≤ outdoor ∧ outdoor ≤ (9 * indoor) / 2

-- Define the set of all valid parking schemes
def valid_schemes : Set (ℕ × ℕ) :=
  {pair | valid_parking_scheme pair.1 pair.2}

-- Theorem for the number of cars in 2013
theorem cars_2013_correct : cars_2013 = 281 := by sorry

-- Theorem for the maximum number of parking spaces
theorem max_parking_spaces :
  valid_schemes = {(17, 74), (18, 71), (19, 68), (20, 65)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_2013_correct_max_parking_spaces_l1208_120826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_l1208_120885

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 2 = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 34 / 5

-- Define the focus of the ellipse/hyperbola
noncomputable def focus : ℝ := Real.sqrt 34

-- Theorem statement
theorem hyperbola_point_distance (a b : ℝ) (M F₂ N O : ℝ × ℝ) :
  hyperbola a b M.1 M.2 →
  ellipse F₂.1 F₂.2 →
  F₂ = (focus, 0) →
  eccentricity = Real.sqrt 34 / 5 →
  dist M F₂ = 18 →
  N = ((M.1 + F₂.1) / 2, (M.2 + F₂.2) / 2) →
  O = (0, 0) →
  dist N O = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_l1208_120885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l1208_120844

theorem min_value_of_function (x : ℝ) : (2 : ℝ)^x + (2 : ℝ)^(2-x) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l1208_120844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_and_sum_l1208_120801

-- Define the functions h and j
def h : ℝ → ℝ := sorry
def j : ℝ → ℝ := sorry

-- State the given conditions
axiom h1 : h 1 = 1
axiom j1 : j 1 = 1
axiom h3 : h 3 = 9
axiom j3 : j 3 = 9
axiom h5 : h 5 = 25
axiom j5 : j 5 = 25
axiom h7 : h 7 = 49
axiom j7 : j 7 = 49

-- Define the theorem
theorem intersection_point_and_sum :
  (h (2 * 7) = 2 * j 7) ∧ (h (2 * 7) = 98) ∧ (7 + 98 = 105) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_and_sum_l1208_120801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_properties_l1208_120868

/-- Represents a cylinder with an inscribed cone. -/
structure CylinderWithCone where
  radius : ℝ
  height : ℝ
  slant_height : ℝ

/-- The ratio of total surface area to lateral surface area is 7:4 -/
def surface_area_ratio (c : CylinderWithCone) : Prop :=
  (2 * c.radius^2 * Real.pi + 2 * c.radius * c.height * Real.pi) / (2 * c.radius * c.height * Real.pi) = 7 / 4

/-- The slant height of the inscribed cone is 30 cm -/
def slant_height_condition (c : CylinderWithCone) : Prop :=
  c.slant_height = 30

/-- The Pythagorean theorem relation between radius, height, and slant height -/
def pythagorean_relation (c : CylinderWithCone) : Prop :=
  c.slant_height^2 = c.radius^2 + c.height^2

/-- The surface area of the cylinder -/
noncomputable def surface_area (c : CylinderWithCone) : ℝ :=
  2 * c.radius^2 * Real.pi + 2 * c.radius * c.height * Real.pi

/-- The volume of the cylinder -/
noncomputable def volume (c : CylinderWithCone) : ℝ :=
  c.radius^2 * c.height * Real.pi

/-- The main theorem stating the surface area and volume of the cylinder -/
theorem cylinder_properties (c : CylinderWithCone) 
    (h1 : surface_area_ratio c)
    (h2 : slant_height_condition c)
    (h3 : pythagorean_relation c) :
    surface_area c = 1512 * Real.pi ∧ volume c = 7776 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_properties_l1208_120868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_length_l1208_120819

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the lengths of sides
def AB : ℝ := 13
def DC : ℝ := 20
def AD : ℝ := 5

-- Define the length of AC
noncomputable def AC : ℝ := Real.sqrt (AB^2 + DC^2)

-- Theorem statement
theorem AC_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.05 ∧ |AC - 24.2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_length_l1208_120819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_sine_function_l1208_120897

theorem min_omega_sine_function (ω φ : ℝ) : 
  ω > 0 → 
  Real.sin (ω * (π / 6) + φ) = 1 → 
  Real.sin (ω * (π / 4) + φ) = 0 → 
  ω ≥ 6 ∧ ∀ ω' > 0, 
    Real.sin (ω' * (π / 6) + φ) = 1 → 
    Real.sin (ω' * (π / 4) + φ) = 0 → 
    ω' ≥ ω :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_sine_function_l1208_120897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equality_l1208_120822

theorem factorial_equality (m n : ℕ) : 3 * 10 * m * n = Nat.factorial 9 ↔ m * n = 6048 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equality_l1208_120822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1208_120852

/-- Calculates the length of a train given its initial speed, acceleration, deceleration due to slope, and time to cross a pole. -/
theorem train_length_calculation (v₀ a d t : Real) :
  v₀ = 60 * (1000 / 3600) →
  a = 4 * (1000 / 3600^2) →
  d = 3.5 * (1000 / 3600^2) →
  t = 3 →
  let net_accel := a - d
  let v := v₀ + net_accel * t
  let L := v₀ * t + (1/2) * net_accel * t^2
  abs (L - 50.17) < 0.01 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1208_120852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_model_height_is_correct_l1208_120862

/-- Represents a water tower with its height and capacity -/
structure WaterTower where
  height : ℝ
  capacity : ℝ

/-- Calculates the height of a scaled model water tower -/
noncomputable def scaled_height (original : WaterTower) (model_capacity : ℝ) : ℝ :=
  original.height * (model_capacity / original.capacity) ^ (1/3)

/-- Theorem stating the correct height of the scaled model -/
theorem scaled_model_height_is_correct (original : WaterTower) 
  (h_height : original.height = 60)
  (h_capacity : original.capacity = 200000)
  (model_capacity : ℝ)
  (h_model_capacity : model_capacity = 0.05) :
  scaled_height original model_capacity = 0.3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval scaled_height ⟨60, 200000⟩ 0.05

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_model_height_is_correct_l1208_120862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l1208_120891

/-- Represents an angle in degrees -/
structure Angle where
  value : ℝ

/-- Returns the length of the side opposite to the given angle in a triangle -/
noncomputable def Angle.opposite_side (angle : Angle) : ℝ := sorry

/-- Given a triangle with angles 45°, 60°, and 75°, where the side opposite
    the 45° angle measures 8 units, the sum of the lengths of the other
    two sides is equal to 8 + 8√2. -/
theorem triangle_side_sum (a b c : Angle) 
    (h_angles : a.value = 45 ∧ b.value = 60 ∧ c.value = 75)
    (h_side : a.opposite_side = 8) :
    b.opposite_side + c.opposite_side = 8 + 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l1208_120891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_probability_theorem_l1208_120851

noncomputable def probability_cos_in_range (a b c d : Real) : Real :=
  (b - a) / (d - c)

theorem cos_probability_theorem :
  let f : Real → Real := fun x ↦ Real.cos (Real.pi * x / 2)
  let interval := Set.Icc (-1 : Real) 1
  let target_range := Set.Icc 0 (1/2 : Real)
  probability_cos_in_range 
    (Real.arccos (1/2) * 2 / Real.pi) 
    1 
    (-1) 
    1 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_probability_theorem_l1208_120851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_reflects_fluctuation_l1208_120803

/-- Standard deviation of a sample -/
def standard_deviation (sample : List ℝ) : ℝ := sorry

/-- Magnitude of fluctuation in a population -/
def magnitude_of_fluctuation (population : List ℝ) : ℝ := sorry

/-- Approximation relation between two real numbers -/
def approximates (x y : ℝ) : Prop := sorry

theorem standard_deviation_reflects_fluctuation 
  (sample : List ℝ) (population : List ℝ) :
  (∀ x, x ∈ sample → x ∈ population) →
  approximates (standard_deviation sample) (magnitude_of_fluctuation population) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_reflects_fluctuation_l1208_120803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_cutting_problem_l1208_120832

/-- Calculates the length of each piece when a rod is cut into equal parts -/
noncomputable def piece_length (rod_length : ℝ) (num_pieces : ℕ) : ℝ :=
  rod_length / (num_pieces : ℝ)

/-- Converts meters to centimeters -/
def meters_to_cm (meters : ℝ) : ℝ :=
  meters * 100

theorem rod_cutting_problem (rod_length : ℝ) (num_pieces : ℕ) 
    (h1 : rod_length = 38.25)
    (h2 : num_pieces = 45) :
    meters_to_cm (piece_length rod_length num_pieces) = 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_cutting_problem_l1208_120832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1208_120894

-- Define the circle C: x^2 + y^2 - 4x = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the line l: kx - 3k - y = 0
def line_l (k x y : ℝ) : Prop := k*x - 3*k - y = 0

-- Theorem statement
theorem line_intersects_circle :
  ∀ (k : ℝ), ∃ (x y : ℝ), circle_C x y ∧ line_l k x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1208_120894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_all_numbers_valid_no_other_valid_numbers_l1208_120890

/-- A function that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A function that checks if a number contains at least one digit 6 -/
def containsSix (n : ℕ) : Prop := ∃ d, d ∈ n.digits 10 ∧ d = 6

/-- The set of numbers that satisfy all conditions -/
def validNumbers : Set ℕ := {n | isThreeDigit n ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ containsSix n}

/-- The list of valid numbers -/
def validNumbersList : List ℕ := [360, 600, 630, 660, 690, 960]

theorem count_valid_numbers : validNumbersList.length = 6 := by
  rfl

theorem all_numbers_valid : ∀ n ∈ validNumbersList, n ∈ validNumbers := by
  sorry

theorem no_other_valid_numbers : ∀ n ∈ validNumbers, n ∈ validNumbersList := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_all_numbers_valid_no_other_valid_numbers_l1208_120890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l1208_120840

-- Define the force function F(x)
noncomputable def F (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 2 then 5
  else if x > 2 then 3 * x + 4
  else 0

-- Define the work function
noncomputable def work (a b : ℝ) : ℝ :=
  ∫ x in a..b, F x

-- Theorem statement
theorem work_calculation :
  work 0 4 = 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l1208_120840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1208_120869

/- Define the ellipse E -/
noncomputable def ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/- Define the eccentricity -/
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

/- Define the perimeter of the quadrilateral -/
noncomputable def quadrilateral_perimeter : ℝ := 4 * Real.sqrt 5

/- Define the line l -/
def line (k m x : ℝ) : ℝ := k * x + m

/- Main theorem -/
theorem ellipse_and_line_intersection :
  ∀ (k m : ℝ),
  (∃ (x₁ x₂ y₁ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧
    y₁ = line k m x₁ ∧ 
    y₂ = line k m x₂ ∧
    (x₁ - 0)^2 + (y₁ - m)^2 = 9 * ((x₂ - 0)^2 + (y₂ - m)^2)) →
  1 < m^2 ∧ m^2 < 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1208_120869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l1208_120824

/-- The function f(x) = x^2 - (1/2)ln(x) + 3/2 -/
noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 3/2

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := 2*x - 1/(2*x)

/-- Theorem stating the range of a for which f(x) is not monotonic in (a-1, a+1) -/
theorem non_monotonic_interval (a : ℝ) :
  (∃ x ∈ Set.Ioo (a - 1) (a + 1), f_deriv x = 0) ↔ a ∈ Set.Icc 1 (3/2) := by
  sorry

#check non_monotonic_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l1208_120824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l1208_120884

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/3 + y^2/4 = 1

/-- Point A -/
def A : ℝ × ℝ := (1, 1)

/-- Point B -/
def B : ℝ × ℝ := (0, -1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The main theorem -/
theorem max_distance_sum :
  ∃ P : ℝ × ℝ, is_on_ellipse P.1 P.2 ∧
    (∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
      distance P A + distance P B ≥ distance Q A + distance Q B) ∧
    distance P A + distance P B = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l1208_120884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_lines_acute_angle_l1208_120865

/-- A type representing a line in a plane -/
structure Line where
  id : ℕ

/-- A type representing a plane with a finite number of lines -/
structure Plane where
  lines : Finset Line

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- Function to calculate the angle between two lines -/
noncomputable def angle_between (l1 l2 : Line) : ℝ := sorry

/-- Theorem stating that given 7 non-parallel lines on a plane, 
    there exist two lines among them such that the angle between them is less than 26° -/
theorem seven_lines_acute_angle (p : Plane) 
  (h1 : p.lines.card = 7)
  (h2 : ∀ l1 l2, l1 ∈ p.lines → l2 ∈ p.lines → l1 ≠ l2 → ¬(are_parallel l1 l2)) :
  ∃ l1 l2, l1 ∈ p.lines ∧ l2 ∈ p.lines ∧ l1 ≠ l2 ∧ angle_between l1 l2 < 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_lines_acute_angle_l1208_120865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_pairs_even_l1208_120818

/-- Represents a seating arrangement around a circular table -/
structure CircularArrangement (α : Type*) where
  elements : List α
  nonempty : elements ≠ []

/-- Counts the number of adjacent pairs with different types in a circular arrangement -/
def countDifferentPairs {α : Type*} [DecidableEq α] [Inhabited α] (arr : CircularArrangement α) : ℕ :=
  let pairs := List.zip (arr.elements ++ [arr.elements.head!]) arr.elements
  pairs.countP (fun (a, b) => a ≠ b)

/-- Theorem: The number of adjacent pairs with different types in a circular arrangement is even -/
theorem different_pairs_even {α : Type*} [DecidableEq α] [Inhabited α] (arr : CircularArrangement α) :
  Even (countDifferentPairs arr) := by
  sorry

#check different_pairs_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_pairs_even_l1208_120818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_points_and_center_on_line_l1208_120802

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := x + y = 2

-- Define the points A and B
def point_A : ℝ × ℝ := (1, -1)
def point_B : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem circle_passes_through_points_and_center_on_line :
  (circle_eq point_A.1 point_A.2) ∧
  (circle_eq point_B.1 point_B.2) ∧
  ∃ (c : ℝ × ℝ), (center_line c.1 c.2) ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - c.1)^2 + (y - c.2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_points_and_center_on_line_l1208_120802
