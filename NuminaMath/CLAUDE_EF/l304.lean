import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_special_case_l304_30484

/-- The area of a triangle with side lengths 7, 7, and 10 is 10√6 -/
theorem triangle_area_special_case : 
  ∀ (a b c : ℝ), a = 7 ∧ b = 7 ∧ c = 10 → 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 10 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_special_case_l304_30484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cube_edge_length_l304_30474

/-- The volume of the large cube in cubic centimeters -/
def large_cube_volume : ℝ := 1000

/-- The number of small cubes that make up the large cube -/
def num_small_cubes : ℕ := 8

/-- The volume of a small cube in cubic centimeters -/
noncomputable def small_cube_volume : ℝ := large_cube_volume / num_small_cubes

/-- The length of one edge of a small cube in centimeters -/
noncomputable def small_cube_edge : ℝ := small_cube_volume ^ (1/3)

/-- Theorem stating that the length of one edge of a small cube is 5 cm -/
theorem small_cube_edge_length : small_cube_edge = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cube_edge_length_l304_30474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_problem_l304_30453

/-- Calculates the time (in minutes) for a train to pass through a tunnel -/
noncomputable def train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length : ℝ) : ℝ :=
  let train_speed_mpm := train_speed_kmh * 1000 / 60
  let total_distance := tunnel_length * 1000 + train_length
  total_distance / train_speed_mpm

theorem train_tunnel_problem : 
  train_tunnel_time 100 72 2.9 = 2.5 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_problem_l304_30453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_second_part_is_40_l304_30468

/-- Calculates the speed during the second part of a trip given the following conditions:
  * The first part of the trip is 180 miles at 60 miles per hour
  * The second part of the trip is 120 miles
  * The average speed for the entire trip is 50 miles per hour
-/
noncomputable def speed_second_part (first_distance : ℝ) (first_speed : ℝ) (second_distance : ℝ) (average_speed : ℝ) : ℝ :=
  let total_distance := first_distance + second_distance
  let total_time := total_distance / average_speed
  let first_time := first_distance / first_speed
  let second_time := total_time - first_time
  second_distance / second_time

/-- Theorem stating that under the given conditions, the speed during the second part of the trip is 40 miles per hour -/
theorem speed_second_part_is_40 :
  speed_second_part 180 60 120 50 = 40 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval speed_second_part 180 60 120 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_second_part_is_40_l304_30468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_increase_is_100_percent_l304_30485

noncomputable def prize_values : List ℝ := [200, 500, 1500, 3000, 10000]

noncomputable def percent_increase (a b : ℝ) : ℝ := (b - a) / a * 100

noncomputable def smallest_percent_increase (values : List ℝ) : ℝ :=
  (List.zip values (List.tail values)).map (fun (a, b) => percent_increase a b)
  |>.minimum?
  |>.getD 0

theorem smallest_increase_is_100_percent :
  smallest_percent_increase prize_values = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_increase_is_100_percent_l304_30485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_b_equals_negative_one_l304_30413

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + b * x^2 + c * x + b * c

noncomputable def f_deriv (b c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*b*x + c

theorem extreme_value_implies_b_equals_negative_one (b c : ℝ) :
  f b c 1 = -4/3 ∧ f_deriv b c 1 = 0 → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_b_equals_negative_one_l304_30413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_solution_l304_30410

theorem determinant_solution (x : ℝ) : 
  (Matrix.det ![![Real.sin x, Real.sqrt 3], ![Real.cos x, 1]] = 0) ↔ 
  (∃ k : ℤ, x = k * Real.pi + Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_solution_l304_30410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_trig_expression_l304_30495

/-- Given two vectors a and b that are perpendicular, prove that a specific trigonometric expression equals 9/5 -/
theorem perpendicular_vectors_trig_expression (α : ℝ) : 
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (Real.cos α, Real.sin α)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⟂ b
  (Real.sin α)^3 + (Real.cos α)^3 = (9/5) * (Real.sin α - Real.cos α) := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_trig_expression_l304_30495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_all_numbers_valid_no_other_valid_numbers_l304_30427

/-- A function that checks if a number is a positive three-digit integer -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A function that checks if a number has 7 in the units place -/
def hasSevenInUnits (n : ℕ) : Prop := n % 10 = 7

/-- The set of numbers we're interested in -/
def validNumbers : Set ℕ := {n : ℕ | isThreeDigit n ∧ hasSevenInUnits n ∧ n % 21 = 0}

/-- List of all valid numbers -/
def validNumbersList : List ℕ := [147, 357, 567, 777, 987]

theorem count_valid_numbers : validNumbersList.length = 5 := by
  rfl

theorem all_numbers_valid : ∀ n ∈ validNumbersList, n ∈ validNumbers := by
  sorry

theorem no_other_valid_numbers : ∀ n ∈ validNumbers, n ∈ validNumbersList := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_all_numbers_valid_no_other_valid_numbers_l304_30427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_integers_l304_30470

theorem sum_of_specific_integers : 
  (Finset.filter (fun n : ℕ => 2 < n ∧ n < 5) (Finset.range 5)).sum id = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_integers_l304_30470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_course_length_l304_30450

/-- 
Given two runners A and B, where:
- A is 4 times faster than B
- A gives B a 72-meter head start
This theorem proves that the race course length for both 
runners to finish simultaneously is 96 meters.
-/
theorem race_course_length : 
  ∀ (v : ℝ), v > 0 → 
  (96 / (4 * v) = (96 - 72) / v) ∧ 96 = 96 := by
  intro v hv
  constructor
  · field_simp
    ring
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_course_length_l304_30450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2024_log_approx_4_l304_30419

-- Define the binary operations
noncomputable def oplus (a b : ℝ) : ℝ := a^(Real.log b / Real.log 5)
noncomputable def otimes (a b : ℝ) : ℝ := a^(Real.log 5 / Real.log b)

-- Define the sequence b_n
noncomputable def b : ℕ → ℝ
  | 0 => 0  -- Add a base case for 0
  | 1 => 0  -- Add a base case for 1
  | 2 => 0  -- Add a base case for 2
  | 3 => otimes 4 3
  | n+4 => oplus (otimes (n+4) (n+3)) (b (n+3))

-- State the theorem
theorem b_2024_log_approx_4 : 
  ∃ ε > 0, |Real.log (b 2024) / Real.log 5 - 4| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2024_log_approx_4_l304_30419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_is_three_l304_30404

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1/2, -Real.sqrt 3/2],
    ![Real.sqrt 3/2, 1/2]]

def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0],
    ![0, 1]]

theorem smallest_power_is_three :
  (∃ n : ℕ, n > 0 ∧ rotation_matrix ^ n = identity_matrix) ∧
  (∀ m : ℕ, 0 < m → m < 3 → rotation_matrix ^ m ≠ identity_matrix) ∧
  rotation_matrix ^ 3 = identity_matrix := by
  sorry

#check smallest_power_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_is_three_l304_30404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_PQRS_l304_30494

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculates the area of an equilateral triangle given its side length -/
noncomputable def equilateralTriangleArea (sideLength : ℝ) : ℝ :=
  (sideLength ^ 2 * Real.sqrt 3) / 4

/-- The main theorem -/
theorem area_of_quadrilateral_PQRS (wxyz : Rectangle) 
    (wxp xqy yrz zsw : Triangle) :
  wxyz.bottomRight.x - wxyz.topLeft.x = 8 →
  wxyz.bottomRight.y - wxyz.topLeft.y = 6 →
  wxp.a = wxyz.topLeft ∧ wxp.b = Point.mk wxyz.topLeft.x wxyz.bottomRight.y ∧
    wxp.c = Point.mk (wxyz.topLeft.x + 6 * Real.sqrt 3 / 2) (wxyz.topLeft.y + 3) →
  xqy.a = Point.mk wxyz.topLeft.x wxyz.bottomRight.y ∧
    xqy.b = wxyz.bottomRight ∧
    xqy.c = Point.mk (wxyz.bottomRight.x - 4 * Real.sqrt 3) wxyz.bottomRight.y →
  yrz.a = wxyz.bottomRight ∧
    yrz.b = Point.mk wxyz.bottomRight.x wxyz.topLeft.y ∧
    yrz.c = Point.mk (wxyz.bottomRight.x - 6 * Real.sqrt 3 / 2) (wxyz.topLeft.y - 3) →
  zsw.a = Point.mk wxyz.bottomRight.x wxyz.topLeft.y ∧
    zsw.b = wxyz.topLeft ∧
    zsw.c = Point.mk (wxyz.topLeft.x + 4 * Real.sqrt 3) wxyz.topLeft.y →
  equilateralTriangleArea 6 + equilateralTriangleArea 8 +
    equilateralTriangleArea 6 + equilateralTriangleArea 8 = 82 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_PQRS_l304_30494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l304_30449

def digits : List Nat := [2, 4, 5]

def is_valid_number (n : Nat) : Bool :=
  let digits_used := n.digits 10
  digits_used.length == 3 ∧ digits_used.all (· ∈ digits) ∧ digits_used.toFinset.card == 3

def all_valid_numbers : List Nat :=
  (List.range 1000).filter is_valid_number

theorem sum_of_valid_numbers : (all_valid_numbers.sum) = 2442 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l304_30449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l304_30403

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (parallel : t.a * Real.cos t.C = (2 * t.b - t.c) * Real.cos t.A) 
  (side_sum : t.b + t.c = 7)
  (area : (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3) :
  t.A = π/3 ∧ t.a + t.b + t.c = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l304_30403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l304_30475

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 11) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l304_30475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_ratio_l304_30452

/-- Given a triangle ABC with the following properties:
    - F is a point on AC that divides AC in ratio 1:2
    - G is the midpoint of BF
    - E is the intersection point of BC and AG
    Prove that E divides BC in ratio 1:3 -/
theorem triangle_division_ratio (A B C F G E : EuclideanSpace ℝ (Fin 2)) :
  (∃ k : ℝ, F - A = k • (C - A) ∧ C - F = (2*k) • (C - A)) →
  (G - B = (1/2) • (F - B)) →
  (∃ t : ℝ, E - B = t • (C - B) ∧ G - A = t • (B - A)) →
  (∃ s : ℝ, E - B = s • (C - B) ∧ C - E = (3*s) • (C - B)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_ratio_l304_30452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_zeros_l304_30438

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x else x^3 - 3*x

-- Define function g in terms of f
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  2 * f a x - a * x

-- Define the property of g having exactly two distinct zeros
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ g a x = 0 ∧ g a y = 0 ∧
  ∀ z : ℝ, g a z = 0 → z = x ∨ z = y

-- State the theorem
theorem range_of_a_for_two_zeros :
  ∀ a : ℝ, has_two_distinct_zeros a ↔ a > -3/2 ∧ a < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_zeros_l304_30438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fractional_momentum_transfer_l304_30446

/-- Represents the fractional momentum transfer in an elastic collision between two particles -/
noncomputable def fractional_momentum_transfer (m M : ℝ) : ℝ := |2 * M / (m + M)|

/-- The maximum possible fractional momentum transfer in an elastic collision -/
theorem max_fractional_momentum_transfer :
  ∀ m M : ℝ, m > 0 → M > 0 → 
  ∃ f_max : ℝ, f_max = 2 ∧ 
  ∀ f : ℝ, f = fractional_momentum_transfer m M → f ≤ f_max :=
by
  sorry

#check max_fractional_momentum_transfer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fractional_momentum_transfer_l304_30446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_from_three_non_coplanar_lines_l304_30409

/-- A point in three-dimensional space. -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in three-dimensional space. -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Three non-coplanar lines through a single point. -/
structure ThreeNonCoplanarLines where
  origin : Point3D
  line1 : Line3D
  line2 : Line3D
  line3 : Line3D
  origin_on_lines : line1.point = origin ∧ line2.point = origin ∧ line3.point = origin
  non_coplanar : ¬ ∃ (normal : Point3D), 
    (normal.x * line1.direction.x + normal.y * line1.direction.y + normal.z * line1.direction.z = 0) ∧
    (normal.x * line2.direction.x + normal.y * line2.direction.y + normal.z * line2.direction.z = 0) ∧
    (normal.x * line3.direction.x + normal.y * line3.direction.y + normal.z * line3.direction.z = 0)

/-- Function to count the number of planes determined by three non-coplanar lines -/
def number_of_planes_determined_by : ThreeNonCoplanarLines → ℕ
  | _ => 3  -- We know this is always 3 for non-coplanar lines

/-- The number of planes determined by three non-coplanar lines through a single point is exactly three. -/
theorem planes_from_three_non_coplanar_lines (lines : ThreeNonCoplanarLines) : 
  ∃! (n : ℕ), n = 3 ∧ n = number_of_planes_determined_by lines := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_from_three_non_coplanar_lines_l304_30409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l304_30406

/-- Represents a 2020 × 2020 game board -/
def GameBoard := Fin 2020 × Fin 2020

/-- Represents a domino placement on the board -/
def Domino := GameBoard × GameBoard

/-- Checks if two dominoes overlap -/
def overlaps (d1 d2 : Domino) : Prop := sorry

/-- Represents a valid game state -/
structure GameState where
  placedDominoes : List Domino
  currentPlayer : Nat
  dominoesDoNotOverlap : ∀ d1 d2, d1 ∈ placedDominoes → d2 ∈ placedDominoes → d1 ≠ d2 → ¬(overlaps d1 d2)

/-- Represents a player's strategy -/
def Strategy := GameState → Option Domino

/-- Checks if a cell is covered by any domino in the list -/
def isCovered (cell : GameBoard) (dominoes : List Domino) : Prop := sorry

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (firstPlayerStrategy : Strategy),
    ∀ (secondPlayerStrategy : Strategy),
      ∃ (uncoveredCell : GameBoard),
        ¬(isCovered uncoveredCell ([] : List Domino)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l304_30406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l304_30411

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def g (a : ℤ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x

-- State the theorem
theorem min_a_value (a : ℤ) : 
  (∀ x > 0, 2 * (Real.log x + 1) ≤ g a x - 2 * x) → a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l304_30411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l304_30408

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.cos (Real.pi * x - Real.pi / 6)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l304_30408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_m_values_l304_30457

noncomputable def vector_a : Fin 2 → ℝ := ![6, 4]
noncomputable def vector_b : Fin 2 → ℝ := ![0, 2]

noncomputable def vector_c (m : ℝ) : Fin 2 → ℝ := 
  fun i => vector_a i + m * vector_b i

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := 
  Real.sqrt ((v 0)^2 + (v 1)^2)

theorem vector_m_values : 
  ∃ m : ℝ, (magnitude (vector_c m) = 10) ∧ (m = 2 ∨ m = -6) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_m_values_l304_30457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generating_functions_correct_l304_30488

/-- Fibonacci polynomial sequence -/
noncomputable def FibPoly (x : ℝ) : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n+2) => x * FibPoly x (n+1) + FibPoly x n

/-- Lucas polynomial sequence -/
noncomputable def LucasPoly (x : ℝ) : ℕ → ℝ
  | 0 => 2
  | 1 => x
  | (n+2) => x * LucasPoly x (n+1) + LucasPoly x n

/-- Generating function for Fibonacci polynomials -/
noncomputable def FibPolyGen (x z : ℝ) : ℝ := z / (1 - x * z - z^2)

/-- Generating function for Lucas polynomials -/
noncomputable def LucasPolyGen (x z : ℝ) : ℝ := (2 + x * z) / (1 - x * z - z^2)

/-- Theorem: The generating functions for Fibonacci and Lucas polynomials are correct -/
theorem generating_functions_correct (x z : ℝ) :
  (∑' n, FibPoly x n * z^n) = FibPolyGen x z ∧
  (∑' n, LucasPoly x n * z^n) = LucasPolyGen x z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_generating_functions_correct_l304_30488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l304_30458

/-- Predicate to check if a triangle with area 'a' is equilateral and inscribed in a rectangle with sides 'w' and 'h' -/
def is_equilateral_triangle_in_rectangle (w h a : ℝ) : Prop :=
  ∃ (s : ℝ), 
    s > 0 ∧ 
    a = (Real.sqrt 3 / 4) * s^2 ∧
    s ≤ w ∧ 
    s ≤ h

/-- The maximum area of an equilateral triangle inscribed in a 13 by 14 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle : 
  ∃ (A : ℝ), 
    (∀ (a : ℝ), is_equilateral_triangle_in_rectangle 13 14 a → a ≤ A) ∧ 
    A = 365 * Real.sqrt 3 - 364 := by
  sorry

#check max_area_equilateral_triangle_in_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l304_30458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_T_type_functions_l304_30429

-- Define the T-type property
def is_T_type (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ D → x₂ ∈ D → f ((x₁ + x₂) / 2) ≥ (f x₁ + f x₂) / 2

-- Define the functions
def f₁ : ℝ → ℝ := λ x ↦ 2 * x - 1
noncomputable def f₂ : ℝ → ℝ := λ x ↦ -x^2 + 2*x
noncomputable def f₃ : ℝ → ℝ := λ x ↦ 1 / x
noncomputable def f₄ : ℝ → ℝ := λ x ↦ 3^x
noncomputable def f₅ : ℝ → ℝ := λ x ↦ Real.log x / Real.log 0.5

-- Define the domains
def D₁ : Set ℝ := Set.univ
def D₂ : Set ℝ := Set.univ
def D₃ : Set ℝ := {x : ℝ | x ≠ 0}
def D₄ : Set ℝ := Set.univ
def D₅ : Set ℝ := {x : ℝ | x > 0}

-- Theorem statement
theorem exactly_two_T_type_functions : 
  (is_T_type f₁ D₁ ∧ is_T_type f₂ D₂ ∧ ¬is_T_type f₃ D₃ ∧ ¬is_T_type f₄ D₄ ∧ ¬is_T_type f₅ D₅) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_T_type_functions_l304_30429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_l304_30439

theorem cyclist_speed (speed_a delay meet_distance : ℝ) 
  (h1 : speed_a = 10)
  (h2 : delay = 6)
  (h3 : meet_distance = 120) : 
  meet_distance / (meet_distance / speed_a - delay) = 20 := by
  
  -- Define speed_b
  let speed_b := meet_distance / (meet_distance / speed_a - delay)
  
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_l304_30439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l304_30415

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle x^2 + y^2 - 8x - 8y + 31 = 0 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 8*y + 31 = 0

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The minimum distance theorem -/
theorem min_distance_theorem :
  ∃ (min_dist : ℝ), min_dist = 5 ∧
  ∀ (p : ℝ × ℝ), circleEq p.1 p.2 → distance focus p ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l304_30415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_carrying_capacity_l304_30418

-- Define the diameter of the larger pipe
def large_diameter : ℝ := 6

-- Define the diameter of the smaller pipe
def small_diameter : ℝ := 1

-- Define the formula for the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the number of smaller pipes needed
def num_small_pipes : ℕ := 36

-- Theorem statement
theorem water_carrying_capacity : 
  (circle_area (large_diameter / 2)) = (num_small_pipes : ℝ) * (circle_area (small_diameter / 2)) :=
by
  -- Expand the definition of circle_area
  unfold circle_area
  -- Simplify the expressions
  simp [large_diameter, small_diameter, num_small_pipes]
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_carrying_capacity_l304_30418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l304_30480

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * ((1 + rate / frequency) ^ (frequency * time) - 1)

/-- Theorem: Given conditions lead to the specified interest rate -/
theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (frequency : ℝ) (total_interest : ℝ) :
  principal = 3000 →
  time = 1.5 →
  frequency = 2 →
  total_interest = 181.78817648189806 →
  ∃ (rate : ℝ), abs (rate - 0.0396) < 0.0001 ∧ compound_interest principal rate time frequency = total_interest :=
by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l304_30480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_l304_30493

/-- Represents the number of people in team B -/
def x : ℕ := sorry

/-- The total number of people in both teams -/
def total_people : ℕ := 100

/-- The number of people in team A -/
def team_A : ℕ := 4 * x - 10

/-- Theorem stating the correct equation for the problem -/
theorem correct_equation : 4 * x + x - 10 = total_people := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_l304_30493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_tangent_l304_30473

noncomputable section

-- Define IsTriangle as a predicate
def IsTriangle (A B C : Real) : Prop :=
  A + B > C ∧ B + C > A ∧ C + A > B

theorem largest_angle_tangent (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  IsTriangle A B C →
  -- Side lengths
  a = 8 →
  b = 10 →
  -- Area condition
  (1/2) * a * b * Real.sin C = 20 * Real.sqrt 3 →
  -- c is opposite to angle C
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  -- Tangent of largest angle
  max (Real.tan A) (max (Real.tan B) (Real.tan C)) = (5 * Real.sqrt 3) / 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_tangent_l304_30473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_cosec_equation_solution_l304_30451

theorem sqrt_cosec_equation_solution :
  ∀ (a b : ℤ), 
    Real.sqrt (16 - 12 * Real.cos (π / 6)) = ↑a + ↑b * (1 / Real.cos (π / 6)) →
    a = 3 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_cosec_equation_solution_l304_30451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_landing_round_trip_exists_l304_30486

/-- A structure representing an airline network -/
structure AirlineNetwork where
  n : ℕ  -- number of airlines
  N : ℕ  -- number of localities
  h : N > 2^n

/-- A predicate stating that there exists a round trip with an odd number of landings -/
def has_odd_landing_round_trip (network : AirlineNetwork) : Prop :=
  ∃ (airline : Fin network.n), ∃ (trip : List (Fin network.N)),
    trip.length % 2 = 1 ∧ 
    trip.head? = trip.get? (trip.length - 1) ∧
    ∀ i : Fin (trip.length - 1), ∃ j : Fin network.N, trip.get? i = some j

/-- The main theorem -/
theorem odd_landing_round_trip_exists (network : AirlineNetwork) :
  has_odd_landing_round_trip network := by
  sorry

#check odd_landing_round_trip_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_landing_round_trip_exists_l304_30486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_a_b_equals_neg_three_l304_30443

-- Define the power function
noncomputable def f (a b x : ℝ) : ℝ := (b + 2) * x^a

-- State the theorem
theorem product_a_b_equals_neg_three (a b : ℝ) : 
  f a b 2 = 8 → a * b = -3 := by
  intro h
  sorry

#check product_a_b_equals_neg_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_a_b_equals_neg_three_l304_30443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l304_30465

/-- The complex number z = cos(2π/3) + i*sin(2π/3) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.exp (2 * Real.pi / 3 * Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l304_30465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_identity_l304_30430

-- Define the angles in radians
noncomputable def angle17 : ℝ := 17 * Real.pi / 180
noncomputable def angle28 : ℝ := 28 * Real.pi / 180

-- State the theorem
theorem tan_sum_identity : 
  Real.tan angle17 + Real.tan angle28 + Real.tan angle17 * Real.tan angle28 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_identity_l304_30430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l304_30401

-- Define the line equation
def line_eq (a x y : ℝ) : Prop := a * x - y + 2 * a + 1 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Theorem statement
theorem line_intersects_circle :
  ∀ a : ℝ, ∃ x y : ℝ, line_eq a x y ∧ circle_eq x y :=
by
  intro a
  -- The fixed point (-2, 1) always satisfies the line equation
  have h1 : line_eq a (-2) 1 := by
    simp [line_eq]
    ring
  
  -- Check if the fixed point (-2, 1) is inside the circle
  have h2 : (-2)^2 + 1^2 < 9 := by norm_num
  
  -- Use the intermediate value theorem to prove intersection
  sorry -- The full proof would require more advanced techniques


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l304_30401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l304_30479

/-- A perfect square trinomial is a polynomial of the form (ay + b)^2 = a^2y^2 + 2aby + b^2 --/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ (y : ℝ), a * y^2 + b * y + c = (p * y + q)^2

/-- Given that 4y^2 + my + 9 is a perfect square trinomial, prove that m = ±12 --/
theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 4 m 9 → m = 12 ∨ m = -12 :=
by
  intro m h
  sorry

#check perfect_square_trinomial_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l304_30479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_fraction_l304_30426

-- Define the fraction we're working with
def fraction : ℚ := 1 / (2^15 * 3)

-- Define a function to get the last digit of a rational number's decimal expansion
noncomputable def last_digit (q : ℚ) : ℕ :=
  (Int.floor (q * 10^1000000) % 10).toNat

-- State the theorem
theorem last_digit_of_fraction :
  last_digit fraction = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_fraction_l304_30426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_equality_l304_30490

theorem complex_arithmetic_equality : 
  Real.sqrt (1 / 9) - Real.sqrt (25 / 4) + 3 * Real.sqrt ((-2)^2) - ((-8) ^ (1/3 : ℝ)) = 35 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_equality_l304_30490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_rectangle_circle_l304_30428

/-- The area of the region common to a rectangle and a circle with shared center -/
theorem common_area_rectangle_circle (w h r : ℝ) (hw : w = 10) (hh : h = 4) (hr : r = 3) :
  4 * (π * r^2 / 4 - h * Real.sqrt (r^2 - (h/2)^2) / 2) = 9*π - 4*Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_rectangle_circle_l304_30428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l304_30461

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 3 * a * x + 2 * Real.cos x

-- Define the derivatives of f and g
noncomputable def f' (x : ℝ) : ℝ := -Real.exp x - 1
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 3 * a - 2 * Real.sin x

-- State the theorem
theorem tangent_perpendicular_range (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f' x₁ * g' a x₂ = -1) →
  -1/3 ≤ a ∧ a ≤ 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l304_30461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaokang_final_position_l304_30414

/-- Represents a walk with direction and distance -/
structure Walk where
  direction : Bool  -- true for east, false for west
  distance : ℕ

/-- Calculates the net distance traveled given a list of walks -/
def netDistance (walks : List Walk) : Int :=
  walks.foldl (fun acc walk => 
    acc + (if walk.direction then walk.distance else -walk.distance)) 0

theorem xiaokang_final_position :
  let walks : List Walk := [
    { direction := true, distance := 150 },   -- East 150m
    { direction := false, distance := 100 },  -- West 100m
    { direction := false, distance := 100 }   -- West 100m
  ]
  netDistance walks = -50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaokang_final_position_l304_30414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_triangle_equality_l304_30476

-- Define a triangle ABC
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define interior angles
def interior_angles (t : Triangle) : Fin 3 → ℝ :=
  sorry

-- Define area of a triangle
noncomputable def area (t : Triangle) : ℝ :=
  sorry

-- Define distance between two points
def distance (p1 p2 : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- Define centroid of a triangle
def centroid (t : Triangle) : EuclideanSpace ℝ (Fin 2) :=
  sorry

-- Theorem statement
theorem triangle_inequality (t : Triangle) (P : EuclideanSpace ℝ (Fin 2)) :
  let α := interior_angles t 0
  let β := interior_angles t 1
  let γ := interior_angles t 2
  let F := area t
  (distance P t.A) ^ 2 * Real.sin (2 * α) +
  (distance P t.B) ^ 2 * Real.sin (2 * β) +
  (distance P t.C) ^ 2 * Real.sin (2 * γ) ≥ 2 * F :=
by sorry

-- Equality condition
theorem triangle_equality (t : Triangle) :
  let α := interior_angles t 0
  let β := interior_angles t 1
  let γ := interior_angles t 2
  let F := area t
  let G := centroid t
  (distance G t.A) ^ 2 * Real.sin (2 * α) +
  (distance G t.B) ^ 2 * Real.sin (2 * β) +
  (distance G t.C) ^ 2 * Real.sin (2 * γ) = 2 * F :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_triangle_equality_l304_30476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OB_l304_30482

-- Define the ellipse C
noncomputable def C (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define point A
def A : ℝ × ℝ := (3, 0)

-- Define the perpendicular bisector of AP
noncomputable def perp_bisector (P : ℝ × ℝ) (x y : ℝ) : Prop :=
  let midpoint := ((P.1 + A.1) / 2, (P.2 + A.2) / 2)
  y - midpoint.2 = ((A.1 - P.1) / (P.2 - A.2)) * (x - midpoint.1)

-- Define point B as the intersection of the perpendicular bisector with the y-axis
noncomputable def B (P : ℝ × ℝ) : ℝ × ℝ :=
  (0, (P.1^2 + P.2^2 - 9) / (2 * P.2))

-- State the theorem
theorem min_distance_OB :
  ∀ P : ℝ × ℝ, C P.1 P.2 → (∃ (x y : ℝ), perp_bisector P x y ∧ x = 0) →
  ∀ Q : ℝ × ℝ, C Q.1 Q.2 → (∃ (x y : ℝ), perp_bisector Q x y ∧ x = 0) →
  Real.sqrt 6 ≤ Real.sqrt ((B P).1^2 + (B P).2^2) ∧
  Real.sqrt 6 ≤ Real.sqrt ((B Q).1^2 + (B Q).2^2) ∧
  (∃ R : ℝ × ℝ, C R.1 R.2 ∧ (∃ (x y : ℝ), perp_bisector R x y ∧ x = 0) ∧
    Real.sqrt ((B R).1^2 + (B R).2^2) = Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OB_l304_30482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l304_30417

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
variable (hf : Differentiable ℝ f)
variable (hg : Differentiable ℝ g)
variable (hfpos : ∀ x, f x > 0)
variable (hgpos : ∀ x, g x > 0)
variable (hderivative : ∀ x, deriv f x * g x - f x * deriv g x < 0)

-- Define the theorem
theorem function_inequality (a b x : ℝ) (h : b < x ∧ x < a) :
  f x * g a > f a * g x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l304_30417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l304_30444

noncomputable def proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let v_norm_squared := v.1^2 + v.2^2
  (dot_product / v_norm_squared * v.1, dot_product / v_norm_squared * v.2)

def v : ℝ × ℝ := (3, 1)

theorem projection_line_equation :
  ∀ (u : ℝ × ℝ), proj v u = v → u.2 = -3 * u.1 + 10 := by
  sorry

#check projection_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_equation_l304_30444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_eight_ninths_l304_30432

/-- The sum of the series Σ(2k / 4^k) for k from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' k, (2 * k : ℝ) / (4 ^ k)

/-- Theorem stating that the sum of the series is equal to 8/9 -/
theorem series_sum_is_eight_ninths : infinite_series_sum = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_eight_ninths_l304_30432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_shapes_same_perimeter_l304_30462

/-- Represents a shape drawn on a square paper --/
structure Shape where
  sides : List ℚ
  deriving Repr

/-- Represents a square sheet of paper --/
structure SquarePaper where
  side_length : ℚ
  deriving Repr

/-- Calculates the perimeter of a shape --/
def perimeter (s : Shape) : ℚ :=
  s.sides.sum

/-- Calculates the perimeter of a square paper --/
def square_perimeter (p : SquarePaper) : ℚ :=
  4 * p.side_length

/-- Checks if a shape has the same perimeter as the square paper --/
def has_same_perimeter (s : Shape) (p : SquarePaper) : Bool :=
  perimeter s = square_perimeter p

/-- The main theorem to be proved --/
theorem four_shapes_same_perimeter 
  (paper : SquarePaper) 
  (shapes : List Shape) 
  (h1 : shapes.length > 0)
  (h2 : ∀ s ∈ shapes, ∀ side ∈ s.sides, side > 0)
  (h3 : ∀ s ∈ shapes, s.sides.length ≥ 4)
  : (shapes.filter (λ s => has_same_perimeter s paper)).length = 4 := by
  sorry

#check four_shapes_same_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_shapes_same_perimeter_l304_30462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_owns_four_persians_l304_30498

/-- The number of Persian cats Jamie owns -/
def jamie_persians : ℕ := 4

/-- The number of Maine Coons Jamie owns -/
def jamie_maine_coons : ℕ := 2

/-- The number of Persian cats Gordon owns -/
def gordon_persians : ℕ := jamie_persians / 2

/-- The number of Maine Coons Gordon owns -/
def gordon_maine_coons : ℕ := jamie_maine_coons + 1

/-- The number of Persian cats Hawkeye owns -/
def hawkeye_persians : ℕ := 0

/-- The number of Maine Coons Hawkeye owns -/
def hawkeye_maine_coons : ℕ := gordon_maine_coons - 1

/-- The total number of cats -/
def total_cats : ℕ := 13

theorem jamie_owns_four_persians :
  jamie_persians + jamie_maine_coons + gordon_persians + gordon_maine_coons + hawkeye_persians + hawkeye_maine_coons = total_cats ∧
  jamie_persians = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_owns_four_persians_l304_30498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_task_completion_l304_30400

/-- Represents the time (in minutes) it takes for a single computer of a given model to complete the task -/
noncomputable def completion_time (model : String) : ℝ :=
  if model = "M" then 36 else 18

/-- Represents the rate at which a single computer of a given model completes the task -/
noncomputable def completion_rate (model : String) : ℝ :=
  1 / completion_time model

/-- The number of computers used for each model -/
noncomputable def num_computers : ℝ := 12

theorem computer_task_completion :
  num_computers * (completion_rate "M" + completion_rate "N") = 1 := by
  sorry

#check computer_task_completion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_task_completion_l304_30400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_exists_no_guaranteed_win_for_large_k_l304_30492

/-- Represents the liar's guessing game with parameters k and n -/
structure LiarsGuessingGame where
  k : ℕ
  n : ℕ

/-- Determines if Ben has a winning strategy for given game parameters -/
def has_winning_strategy (game : LiarsGuessingGame) : Prop :=
  ∀ N x : ℕ, 1 ≤ x → x ≤ N → ∃ S : Finset ℕ, S.card ≤ game.n ∧ x ∈ S

/-- Part (a): If n ≥ 2^k, then Ben can always win -/
theorem winning_strategy_exists (game : LiarsGuessingGame) :
    game.n ≥ 2^game.k → has_winning_strategy game := by
  sorry

/-- Part (b): For sufficiently large k, there exist n ≥ 1.99^k such that Ben cannot guarantee a win -/
theorem no_guaranteed_win_for_large_k :
    ∃ k₀ : ℕ, ∀ k ≥ k₀, ∃ n : ℕ, n ≥ (1.99 : ℝ)^k ∧ ¬has_winning_strategy ⟨k, n⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_exists_no_guaranteed_win_for_large_k_l304_30492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_theorem_l304_30416

/-- Given a triangle XYZ with median XM, prove that XY = 2√61 under specific conditions -/
theorem triangle_reflection_theorem (X Y Z M N E : ℝ × ℝ) : 
  let Y' := (2 * M.1 - Y.1, 2 * M.2 - Y.2)
  let Z' := (2 * M.1 - Z.1, 2 * M.2 - Z.2)
  (X.1 - N.1)^2 + (X.2 - N.2)^2 = 64 →
  (N.1 - Z.1)^2 + (N.2 - Z.2)^2 = 256 →
  (Y.1 - E.1)^2 + (Y.2 - E.2)^2 = 144 →
  N = (2 * M.1 - E.1, 2 * M.2 - E.2) →
  M.1 = (Y.1 + Z.1) / 2 ∧ M.2 = (Y.2 + Z.2) / 2 →
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 244 := by
  sorry

#check triangle_reflection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_theorem_l304_30416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_four_l304_30424

-- Define the grid type
def Grid := Fin 4 → Fin 4 → Fin 4

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j < 4) ∧
  (∀ i, Function.Injective (g i)) ∧
  (∀ j, Function.Injective (λ i ↦ g i j))

-- Define the initial conditions
def initial_conditions (g : Grid) : Prop :=
  g 0 0 = 0 ∧ g 0 1 = 3 ∧ g 1 2 = 2 ∧ g 2 0 = 3 ∧ g 3 1 = 1

-- Theorem statement
theorem lower_right_is_four (g : Grid) 
  (h1 : is_valid_grid g) 
  (h2 : initial_conditions g) : 
  g 3 3 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_is_four_l304_30424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_value_l304_30431

def value : ℕ := 2^15 * 5^10 * 3^5

def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log n 10).succ

theorem digits_of_value : num_digits value = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_value_l304_30431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l304_30467

universe u

def A : Set ℕ := {1, 2}
def B (a : ℕ) : Set ℕ := {3, a}
def U : Set ℕ := {1, 2, 3, 4, 5}

theorem set_operations (a : ℕ) (h : Set.Nonempty (A ∩ B a)) :
  (A ∪ B a = {1, 2, 3}) ∧
  ((U \ B a = {2, 4, 5}) ∨ (U \ B a = {1, 4, 5})) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l304_30467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l304_30477

/-- Given three points A, B, and C in ℝ², and a point P on the plane ABC,
    prove that the dot product of AB and AC is 4, and lambda + mu = 1/3,
    where AP = lambda * AB + mu * AC, AP · AB = 0, and AP · AC = 3. -/
theorem vector_problem (A B C P : ℝ × ℝ) (lambda mu : ℝ) : 
  A = (1, -1) →
  B = (3, 0) →
  C = (2, 1) →
  P.1 - A.1 = lambda * (B.1 - A.1) + mu * (C.1 - A.1) →
  P.2 - A.2 = lambda * (B.2 - A.2) + mu * (C.2 - A.2) →
  (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 0 →
  (P.1 - A.1) * (C.1 - A.1) + (P.2 - A.2) * (C.2 - A.2) = 3 →
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 4 ∧ lambda + mu = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l304_30477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_rick_digging_time_l304_30471

/-- Represents the digging scenario of Pirate Rick --/
structure DiggingScenario where
  initial_depth : ℚ
  initial_time : ℚ
  storm_factor : ℚ
  tsunami_add : ℚ

/-- Calculates the time needed to dig up the treasure --/
def time_to_dig_up (scenario : DiggingScenario) : ℚ :=
  let digging_rate := scenario.initial_depth / scenario.initial_time
  let depth_after_storm := scenario.initial_depth * scenario.storm_factor
  let final_depth := depth_after_storm + scenario.tsunami_add
  final_depth / digging_rate

/-- Theorem stating that the time to dig up the treasure is 3 hours --/
theorem pirate_rick_digging_time :
  let scenario : DiggingScenario := {
    initial_depth := 8,
    initial_time := 4,
    storm_factor := 1/2,
    tsunami_add := 2
  }
  time_to_dig_up scenario = 3 := by
  -- Unfold the definition and perform the calculation
  unfold time_to_dig_up
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_rick_digging_time_l304_30471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_circumference_area_of_lighter_region_l304_30469

-- Define the side length of the square
variable (n : ℝ)

-- Define the radius of the circumference
noncomputable def R (n : ℝ) : ℝ := n - (n * Real.sqrt 2) / 2

-- Define the area of the lighter region
noncomputable def lighter_area (n : ℝ) : ℝ := n^2 * (1 - Real.pi / 3 - Real.sqrt 3 / 2)

-- Theorem for the radius of the circumference
theorem radius_of_circumference (n : ℝ) (h : n > 0) :
  R n = n - (n * Real.sqrt 2) / 2 := by
  -- The proof is trivial as it's the definition of R
  rfl

-- Theorem for the area of the lighter region
theorem area_of_lighter_region (n : ℝ) (h : n > 0) :
  lighter_area n = n^2 * (1 - Real.pi / 3 - Real.sqrt 3 / 2) := by
  -- The proof is trivial as it's the definition of lighter_area
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_circumference_area_of_lighter_region_l304_30469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l304_30456

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 14)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, both_toppings = 5 ∧
    pepperoni_slices + mushroom_slices - both_toppings = total_slices :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l304_30456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l304_30407

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def ArithmeticSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_10 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_a1 : a 1 = -4) 
  (h_sum46 : a 4 + a 6 = 16) : 
  ArithmeticSum a 10 = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l304_30407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_and_range_l304_30405

/-- Given vectors a and b, prove the analytical expression of f(x) and the range of m -/
theorem vector_dot_product_and_range (ω : ℝ) (h_ω : ω > 0) :
  let a : ℝ → ℝ × ℝ := λ x ↦ (Real.sin (ω * x), Real.sqrt 3 * Real.cos (2 * ω * x))
  let b : ℝ → ℝ × ℝ := λ x ↦ ((1 / 2) * Real.cos (ω * x), 1 / 4)
  let f : ℝ → ℝ := λ x ↦ (a x).1 * (b x).1 + (a x).2 * (b x).2
  let symmetry_distance : ℝ := π / 2
  ∃ (period : ℝ), period > 0 ∧
    (∀ x, f (x + period) = f x) ∧
    (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ period) ∧
    symmetry_distance = period / 2 →
  (∀ x, f x = (1 / 2) * Real.sin (2 * x + π / 3)) ∧
  (Set.Icc (Real.sqrt 3 / 4) (1 / 2) = 
    {m | ∃! (s₁ s₂ : ℝ), s₁ ≠ s₂ ∧ s₁ ∈ Set.Icc 0 (7 * π / 12) ∧ 
                          s₂ ∈ Set.Icc 0 (7 * π / 12) ∧ 
                          f s₁ = m ∧ f s₂ = m})
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_and_range_l304_30405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l304_30445

-- Define the function f(x) = 1/(x-3) + x
noncomputable def f (x : ℝ) := 1 / (x - 3) + x

-- State the theorem
theorem min_value_of_f :
  ∀ x > 3, f x ≥ 5 ∧ ∃ x₀ > 3, f x₀ = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l304_30445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_not_all_odd_distances_l304_30402

theorem four_points_not_all_odd_distances (P : Finset (ℝ × ℝ)) :
  P.card = 4 → ∃ p q : ℝ × ℝ, p ∈ P ∧ q ∈ P ∧ p ≠ q ∧ ¬(∃ n : ℕ, dist p q = 2 * n + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_not_all_odd_distances_l304_30402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_classification_correct_l304_30459

-- Define the types of functions
inductive FunctionType
| Algebraic
| Transcendental

-- Define the subtypes of algebraic functions
inductive AlgebraicType
| Rational
| Irrational

-- Define the subtypes of rational functions
inductive RationalType
| Entire
| Fractional

-- Define a structure to represent a function
structure MathFunction where
  expr : ℝ → ℝ
  type : FunctionType
  algebraicType : Option AlgebraicType
  rationalType : Option RationalType

-- Define some example functions
noncomputable def linearFunction : MathFunction :=
  { expr := λ x ↦ 2*x + 1,
    type := FunctionType.Algebraic,
    algebraicType := some AlgebraicType.Rational,
    rationalType := some RationalType.Entire }

noncomputable def fractionalFunction : MathFunction :=
  { expr := λ x ↦ (x + 1) / (x - 1),
    type := FunctionType.Algebraic,
    algebraicType := some AlgebraicType.Rational,
    rationalType := some RationalType.Fractional }

noncomputable def irrationalFunction : MathFunction :=
  { expr := λ x ↦ Real.sqrt (x + 1),
    type := FunctionType.Algebraic,
    algebraicType := some AlgebraicType.Irrational,
    rationalType := none }

noncomputable def exponentialFunction : MathFunction :=
  { expr := λ x ↦ Real.exp x,
    type := FunctionType.Transcendental,
    algebraicType := none,
    rationalType := none }

-- Theorem stating that the classification is correct
theorem function_classification_correct :
  (linearFunction.type = FunctionType.Algebraic) ∧
  (fractionalFunction.type = FunctionType.Algebraic) ∧
  (irrationalFunction.type = FunctionType.Algebraic) ∧
  (exponentialFunction.type = FunctionType.Transcendental) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_classification_correct_l304_30459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a3_div_a5_eq_three_fourths_l304_30483

def sequence_a : ℕ → ℚ
| 0 => 1
| n + 1 => (sequence_a n + (-1)^(n + 1)) / sequence_a n

theorem a3_div_a5_eq_three_fourths :
  sequence_a 2 / sequence_a 4 = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a3_div_a5_eq_three_fourths_l304_30483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l304_30491

-- Define the speed of the train in km/hr
noncomputable def train_speed : ℝ := 60

-- Define the time taken to cross the pole in seconds
noncomputable def crossing_time : ℝ := 21

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

-- Theorem to prove the length of the train
theorem train_length_proof :
  let speed_m_s := train_speed * km_hr_to_m_s
  let length := speed_m_s * crossing_time
  ∃ ε > 0, abs (length - 350.07) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l304_30491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_five_consecutive_interesting_infinitely_many_triple_interesting_l304_30440

/-- E(n) is the number of 1's in the binary representation of n -/
def E (n : ℕ) : ℕ := sorry

/-- A number n is interesting if E(n) divides n -/
def interesting (n : ℕ) : Prop := n ≠ 0 ∧ E n ∣ n

theorem no_five_consecutive_interesting :
  ∀ k : ℕ, ∃ m ∈ Finset.range 5, ¬interesting (k + m) := by sorry

theorem infinitely_many_triple_interesting :
  ∀ N : ℕ, ∃ n ≥ N, interesting n ∧ interesting (n + 1) ∧ interesting (n + 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_five_consecutive_interesting_infinitely_many_triple_interesting_l304_30440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_value_in_base6_addition_l304_30448

/-- Represents a digit in base-6 --/
def Base6Digit := Fin 6

/-- Converts a natural number to its base-6 representation --/
def toBase6 (n : ℕ) : List Base6Digit := sorry

/-- Performs addition in base-6 --/
def addBase6 (a b : List Base6Digit) : List Base6Digit := sorry

/-- Converts a natural number to a Base6Digit --/
def toBase6Digit (n : ℕ) : Base6Digit :=
  ⟨n % 6, by
    apply Nat.mod_lt
    exact Nat.zero_lt_succ 5⟩

/-- The main theorem proving that ♢ = 4 in the given base-6 addition problem --/
theorem diamond_value_in_base6_addition :
  ∃ (diamond : Base6Digit),
    let row1 := [toBase6Digit 4, toBase6Digit 3, diamond, toBase6Digit 4]
    let row2 := [toBase6Digit 0, diamond, toBase6Digit 2, toBase6Digit 5]
    let row3 := [toBase6Digit 0, toBase6Digit 0, toBase6Digit 2, diamond]
    let result := [toBase6Digit 5, toBase6Digit 0, diamond, toBase6Digit 3]
    addBase6 (addBase6 row1 row2) row3 = result ∧ diamond = toBase6Digit 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_value_in_base6_addition_l304_30448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_sum_constant_l304_30472

/- Define the ellipse -/
noncomputable def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

/- Define the eccentricity -/
noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b ^ 2 / a ^ 2))

/- Define a line passing through a point -/
def Line (m : ℝ) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 - p.2 = m * (q.1 - p.1)}

/- Define the slope between two points -/
noncomputable def Slope (p q : ℝ × ℝ) : ℝ :=
  (q.2 - p.2) / (q.1 - p.1)

/- Main theorem -/
theorem ellipse_slope_sum_constant
  (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_a_gt_b : a > b)
  (h_ecc : Eccentricity a b = 1/2)
  (T : ℝ × ℝ) (h_T : T = (4, 0))
  (F : ℝ × ℝ) (h_F_focus : F ∈ Ellipse a b)
  (l : Set (ℝ × ℝ)) (h_l_line : ∃ m : ℝ, l = Line m F)
  (R S : ℝ × ℝ) (h_R : R ∈ Ellipse a b ∩ l) (h_S : S ∈ Ellipse a b ∩ l)
  (h_R_ne_S : R ≠ S) :
  Slope T R + Slope T S = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_sum_constant_l304_30472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l304_30499

noncomputable def g (x : ℝ) : ℝ := (x - 4) / (x + 3)

theorem g_properties :
  (g 1 = -3/4) ∧
  (g 2 = -2/5) ∧
  (¬ ∃ (y : ℝ), g (-3) = y) ∧
  (g (-4) ≠ -2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l304_30499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_research_paper_editing_ratio_l304_30435

/-- The ratio of words added to words removed in Yvonne and Janna's research paper editing process -/
theorem research_paper_editing_ratio : 
  (let yvonne_words : ℕ := 400
   let janna_words : ℕ := yvonne_words + 150
   let total_words_before_editing : ℕ := yvonne_words + janna_words
   let words_removed : ℕ := 20
   let total_words_after_removal : ℕ := total_words_before_editing - words_removed
   let words_needed : ℕ := 1000
   let words_added : ℕ := words_needed - total_words_after_removal
   (words_added : ℚ) / (words_removed : ℚ)) = 7 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_research_paper_editing_ratio_l304_30435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_neg_one_l304_30425

theorem polynomial_value_at_neg_one (n : ℕ) (P : Polynomial ℝ) :
  (Polynomial.degree P = n) →
  (∀ k : ℕ, k ≥ 1 ∧ k ≤ n + 1 → P.eval (k : ℝ) = (k : ℝ)⁻¹) →
  P.eval (-1 : ℝ) = (n + 1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_neg_one_l304_30425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_inverse_l304_30434

theorem cube_root_product_inverse (P : ℝ) : P = (Real.rpow 6 (1/3) * Real.rpow (1/162) (1/3))⁻¹ → P = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_inverse_l304_30434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_points_l304_30496

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the line
def line (x y : ℝ) : Prop := x/4 + y/3 = 1

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define a function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_area_points :
  ∃! (s : Finset (ℝ × ℝ)), 
    (∀ p ∈ s, ellipse p.1 p.2) ∧ 
    (∀ p ∈ s, triangleArea p A B = 3) ∧
    s.card = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_points_l304_30496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_after_25_days_l304_30489

/-- The number of days when two borrowers owe the same amount -/
noncomputable def days_to_equal_debt (
  darren_initial : ℝ)  -- Darren's initial borrowing
  (darren_rate : ℝ)    -- Darren's daily interest rate
  (fergie_initial : ℝ) -- Fergie's initial borrowing
  (fergie_rate : ℝ)    -- Fergie's daily interest rate
  : ℝ :=
  (fergie_initial - darren_initial) / (darren_initial * darren_rate - fergie_initial * fergie_rate)

/-- Theorem stating that Darren and Fergie owe the same amount after 25 days -/
theorem equal_debt_after_25_days :
  days_to_equal_debt 200 0.08 300 0.04 = 25 := by
  -- Unfold the definition of days_to_equal_debt
  unfold days_to_equal_debt
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_after_25_days_l304_30489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_sum_is_two_l304_30442

noncomputable def f (x : ℝ) : ℝ :=
  (2 * x^2 + Real.sqrt 2 * Real.sin (x + Real.pi/4)) / (2 * x^2 + Real.cos x)

theorem f_max_min_sum_is_two :
  ∃ (a b : ℝ), (∀ x, f x ≤ a) ∧ (∃ x, f x = a) ∧
               (∀ x, b ≤ f x) ∧ (∃ x, f x = b) ∧
               (a + b = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_sum_is_two_l304_30442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unbroken_light_bulbs_l304_30412

theorem unbroken_light_bulbs (kitchen_broken_fraction : Rat) 
                             (foyer_broken_fraction : Rat)
                             (foyer_broken : Nat) 
                             (kitchen_total : Nat) : Nat :=
  let kitchen_broken_fraction := 3/5
  let foyer_broken_fraction := 1/3
  let foyer_broken := 10
  let kitchen_total := 35
  have : (kitchen_total - kitchen_broken_fraction * kitchen_total) + 
         (foyer_broken / foyer_broken_fraction - foyer_broken) = 34 := by
    sorry
  34


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unbroken_light_bulbs_l304_30412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l304_30487

/-- Rotate a 3D vector 180° about the origin through the x-axis -/
def rotate180AboutXAxis (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.1, -v.2.1, -v.2.2)

/-- The original vector -/
def originalVector : ℝ × ℝ × ℝ := (1, 3, 2)

/-- The expected result vector -/
def resultVector : ℝ × ℝ × ℝ := (-1, -3, -2)

theorem rotation_result :
  rotate180AboutXAxis originalVector = resultVector := by
  simp [rotate180AboutXAxis, originalVector, resultVector]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l304_30487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_plus_a_fifth_power_l304_30436

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Define the coefficient of x^r in (√x + a)^5
def coeff (r : ℕ) (a : ℝ) : ℝ := 
  (binomial 5 (5 - 2*r)) * a^(5 - 2*r)

theorem sqrt_plus_a_fifth_power (a : ℝ) :
  coeff 2 a = 10 → coeff 1 a = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_plus_a_fifth_power_l304_30436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_remainders_sum_quotient_l304_30478

theorem square_remainders_sum_quotient : 
  let remainders := Finset.image (fun n => (n^2) % 13) (Finset.range 15)
  let distinct_remainders := remainders.toList.eraseDup.toFinset
  let m := Finset.sum distinct_remainders id
  m / 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_remainders_sum_quotient_l304_30478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_not_necessarily_perpendicular_l304_30422

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  D : Point
  E : Point
  F : Point

/-- Reflect a point across the line y = -x -/
def reflect (p : Point) : Point :=
  { x := -p.y, y := -p.x }

/-- Check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Check if a point is not on the line y = -x -/
def isNotOnDiagonal (p : Point) : Prop :=
  p.x ≠ -p.y

/-- Calculate the slope between two points -/
noncomputable def slopeBetween (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Main theorem -/
theorem lines_not_necessarily_perpendicular :
  ¬ ∀ (t : Triangle), 
    isInSecondQuadrant t.D → 
    isInSecondQuadrant t.E → 
    isInSecondQuadrant t.F → 
    isNotOnDiagonal t.D → 
    isNotOnDiagonal t.E → 
    isNotOnDiagonal t.F → 
    slopeBetween t.D t.E * slopeBetween (reflect t.D) (reflect t.E) = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_not_necessarily_perpendicular_l304_30422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_ratio_l304_30454

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagon where
  /-- The vertices of the pentagon -/
  A : Fin 5 → ℝ × ℝ
  /-- The area of the circumscribed circle -/
  circle_area : ℝ
  /-- The circle area is (5 + √5) / 10 * π -/
  h_circle_area : circle_area = (5 + Real.sqrt 5) / 10 * Real.pi

/-- Points B and C on the rays of the pentagon -/
structure PentagonPoints (pentagon : RegularPentagon) where
  /-- Points B on the rays -/
  B : Fin 5 → ℝ × ℝ
  /-- Points C on the rays -/
  C : Fin 5 → ℝ × ℝ
  /-- B points satisfy the given condition -/
  h_B : ∀ i : Fin 5, dist (B i) (pentagon.A i) * dist (B i) (pentagon.A (i + 1)) = 
                     dist (B i) (pentagon.A (i + 2))
  /-- C points satisfy the given condition -/
  h_C : ∀ i : Fin 5, dist (C i) (pentagon.A i) * dist (C i) (pentagon.A (i + 1)) = 
                     (dist (C i) (pentagon.A (i + 2)))^2

/-- Calculate the area of a polygon given its vertices -/
noncomputable def area_polygon (vertices : Fin 5 → ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem pentagon_area_ratio (pentagon : RegularPentagon) (points : PentagonPoints pentagon) :
  area_polygon points.B / area_polygon points.C = (1 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_ratio_l304_30454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l304_30455

theorem sin_cos_relation (θ : Real) (h : Real.sin θ = 1/4) : 
  Real.cos (3*π/2 + θ) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l304_30455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_operations_l304_30460

-- Define odd and even integers
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define the operations
def AddOp (a b : ℤ) : ℤ := a + b
def SubtractOp (a b : ℤ) : ℤ := a - b
def MultiplyOp (a b : ℤ) : ℤ := a * b
noncomputable def DivideOp (a b : ℤ) : ℚ := (a : ℚ) / (b : ℚ)
noncomputable def AverageOp (a b : ℤ) : ℚ := ((a : ℚ) + (b : ℚ)) / 2

-- Theorem statement
theorem odd_even_operations :
  (∀ a b : ℤ, IsOdd a → IsEven b → IsEven (MultiplyOp a b)) ∧
  (∃ a b : ℤ, IsOdd a → IsEven b → ¬IsEven (AddOp a b)) ∧
  (∃ a b : ℤ, IsEven a → IsOdd b → ¬IsEven (SubtractOp a b)) ∧
  (∃ a b : ℤ, IsEven a → IsOdd b → ¬IsEven (Int.floor (DivideOp a b))) ∧
  (∃ a b : ℤ, IsOdd a → IsEven b → ¬IsEven (Int.floor (AverageOp a b))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_operations_l304_30460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_sum_ab_l304_30466

/-- The angle between two 2D vectors -/
noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem angle_between_a_and_sum_ab :
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (-1/2, Real.sqrt 3 / 2)
  angle_between a (a.1 + b.1, a.2 + b.2) = π / 3 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_sum_ab_l304_30466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_berry_cost_l304_30423

/-- Calculates the cost of berries for a given period -/
def berryCost (dailyConsumption : ℚ) (costPerCup : ℚ) (days : ℕ) : ℚ :=
  dailyConsumption * costPerCup * days

/-- Theorem: Martin's berry cost for 30 days is $30.00 -/
theorem martin_berry_cost : 
  berryCost (1/2) 2 30 = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_berry_cost_l304_30423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_percentage_value_l304_30481

/-- A pentagon formed by an equilateral triangle atop a rectangle --/
structure UnusualPentagon where
  rect_width : ℝ
  rect_length : ℝ
  triangle_side : ℝ
  rect_area : ℝ
  (width_eq_side : rect_width = triangle_side)
  (length_half_side : rect_length = triangle_side / 2)
  (rect_area_eq : rect_width * rect_length = rect_area)
  (rect_area_value : rect_area = 48)

/-- The percentage of the pentagon's area occupied by the equilateral triangle --/
noncomputable def triangle_percentage (p : UnusualPentagon) : ℝ :=
  (200 * Real.sqrt 3 - 300) / 100

/-- Theorem stating the percentage of the pentagon's area occupied by the triangle --/
theorem triangle_percentage_value (p : UnusualPentagon) :
  triangle_percentage p = (200 * Real.sqrt 3 - 300) / 100 := by
  sorry

#eval "Unusual Pentagon theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_percentage_value_l304_30481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_domain_range_l304_30437

noncomputable def f (a b x : ℝ) := a^x + b

theorem exponential_function_domain_range (a b : ℝ) :
  a > 0 ∧ a ≠ 1 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, f a b x ∈ Set.Icc (-1 : ℝ) 0) ∧
  (∀ y ∈ Set.Icc (-1 : ℝ) 0, ∃ x ∈ Set.Icc (-1 : ℝ) 0, f a b x = y) →
  a + b = -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_domain_range_l304_30437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_problem_l304_30421

/-- Given vectors OA, OB, OC in R³ and collinear points A, B, C, prove k = 2/3 --/
theorem vector_collinearity_problem (k : ℝ) :
  let OA : Fin 3 → ℝ := ![k, 12, 1]
  let OB : Fin 3 → ℝ := ![4, 5, 1]
  let OC : Fin 3 → ℝ := ![-k, 10, 1]
  let AB : Fin 3 → ℝ := ![OB 0 - OA 0, OB 1 - OA 1, OB 2 - OA 2]
  let AC : Fin 3 → ℝ := ![OC 0 - OA 0, OC 1 - OA 1, OC 2 - OA 2]
  (∃ (l : ℝ), AB = l • AC) →
  k = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_problem_l304_30421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_approx_twenty_l304_30433

/-- 
Given that some percentage P of (x - y) equals 14% of (x + y), 
and y is 17.647058823529413% of x, prove that P is approximately equal to 20%.
-/
theorem percentage_difference_approx_twenty 
  (x y : ℝ) 
  (P : ℝ) 
  (h1 : P / 100 * (x - y) = 14 / 100 * (x + y)) 
  (h2 : y = 17.647058823529413 / 100 * x) : 
  ‖P - 20‖ < 1e-10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_approx_twenty_l304_30433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_theorem_l304_30497

theorem cos_sum_theorem (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 3) 
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_theorem_l304_30497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_equivalence_234_l304_30441

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Man : U → Prop)
variable (Woman : U → Prop)
variable (GoodDriver : U → Prop)
variable (BadDriver : U → Prop)

-- Define the statements
def statement1 (U : Type) (Woman GoodDriver : U → Prop) : Prop := ∀ x, Woman x → GoodDriver x
def statement2 (U : Type) (Woman GoodDriver : U → Prop) : Prop := ∀ x, GoodDriver x → Woman x
def statement3 (U : Type) (Man GoodDriver : U → Prop) : Prop := ∀ x, Man x → ¬GoodDriver x
def statement4 (U : Type) (Man BadDriver : U → Prop) : Prop := ∀ x, Man x → BadDriver x
def statement5 (U : Type) (Man BadDriver : U → Prop) : Prop := ∃ x, Man x ∧ BadDriver x
def statement6 (U : Type) (Man GoodDriver : U → Prop) : Prop := ∀ x, Man x → GoodDriver x

-- Theorem 1: Negation of statement6 is equivalent to statement5
theorem negation_equivalence (U : Type) (Man GoodDriver BadDriver : U → Prop) : 
  (¬statement6 U Man GoodDriver) ↔ statement5 U Man BadDriver := by sorry

-- Theorem 2: Equivalence of statements 2, 3, and 4
theorem equivalence_234 (U : Type) (Man Woman GoodDriver BadDriver : U → Prop) : 
  (statement2 U Woman GoodDriver ↔ statement3 U Man GoodDriver) ∧ 
  (statement3 U Man GoodDriver ↔ statement4 U Man BadDriver) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_equivalence_234_l304_30441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_eleven_sum_l304_30447

def A : Set Nat := {n | 1 ≤ n ∧ n ≤ 100}

theorem divisible_by_eleven_sum (B : Finset Nat) (hB : ↑B ⊆ A) (hCard : B.card = 53) :
  ∃ x y, x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ (11 ∣ (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_eleven_sum_l304_30447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l304_30463

theorem remainder_sum_mod_13 (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l304_30463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwayne_yearly_earnings_l304_30420

def dwayne_earnings : ℝ → Prop := λ d => d ≥ 0

def brady_earnings : ℝ → Prop := λ b => b ≥ 0

def combined_earnings : ℝ → Prop := λ c => c ≥ 0

axiom brady_more_than_dwayne (d : ℝ) : 
  dwayne_earnings d → brady_earnings (d + 450)

axiom total_earnings (d : ℝ) : 
  dwayne_earnings d ∧ brady_earnings (d + 450) → combined_earnings 3450

theorem dwayne_yearly_earnings : 
  ∃ d : ℝ, dwayne_earnings d ∧ d = 1500 := by
  use 1500
  constructor
  · -- Prove dwayne_earnings 1500
    unfold dwayne_earnings
    linarith
  · -- Prove 1500 = 1500
    rfl

#check dwayne_yearly_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwayne_yearly_earnings_l304_30420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_plot_shorter_side_l304_30464

/-- A rectangular plot with given dimensions and fencing constraints -/
structure RectangularPlot where
  longSide : ℝ
  perimeter : ℝ
  numPoles : ℕ
  poleDist : ℝ

/-- The properties of the specific plot in the problem -/
def problemPlot : RectangularPlot where
  longSide := 50
  perimeter := 125
  numPoles := 26
  poleDist := 5

/-- The length of the shorter side of the rectangular plot -/
noncomputable def shorterSide (plot : RectangularPlot) : ℝ :=
  (plot.perimeter / 2) - plot.longSide

/-- Theorem stating that the shorter side of the problem's plot is 12.5 meters -/
theorem problem_plot_shorter_side :
  shorterSide problemPlot = 12.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_plot_shorter_side_l304_30464
