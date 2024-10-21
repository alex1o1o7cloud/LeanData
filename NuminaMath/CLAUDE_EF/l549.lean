import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l549_54941

theorem angle_sum_theorem (θ φ : Real) (h1 : 0 < θ ∧ θ < π / 2) (h2 : 0 < φ ∧ φ < π / 2)
  (h3 : Real.tan θ = 3 / 5) (h4 : Real.sin φ = 2 / Real.sqrt 5) :
  θ + 2 * φ = π - Real.arctan (11 / 27) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l549_54941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_l549_54998

/-- Represents a chessboard with red and blue pieces -/
structure Chessboard :=
  (size : Nat)
  (red_pieces : Finset (Nat × Nat))
  (blue_pieces : Finset (Nat × Nat))

/-- Predicate to check if two positions are in the same row or column -/
def can_see (pos1 pos2 : Nat × Nat) : Bool :=
  pos1.1 = pos2.1 ∨ pos1.2 = pos2.2

/-- Counts the number of pieces of the opposite color that a piece can see -/
def visible_opposite_pieces (board : Chessboard) (pos : Nat × Nat) (is_red : Bool) : Nat :=
  let opposite_pieces := if is_red then board.blue_pieces else board.red_pieces
  (opposite_pieces.filter (fun p => can_see pos p)).card

/-- Predicate to check if the board configuration is valid -/
def valid_board (board : Chessboard) : Prop :=
  board.size = 200 ∧
  (∀ pos ∈ board.red_pieces, visible_opposite_pieces board pos true = 5) ∧
  (∀ pos ∈ board.blue_pieces, visible_opposite_pieces board pos false = 5) ∧
  board.red_pieces ∩ board.blue_pieces = ∅

/-- The main theorem stating the maximum number of pieces -/
theorem max_pieces (board : Chessboard) :
  valid_board board →
  (board.red_pieces.card + board.blue_pieces.card) ≤ 4000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_l549_54998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_zero_removal_false_l549_54953

theorem decimal_zero_removal_false : 
  ¬(∀ (a b : ℚ), ∃ (c : ℚ), (a + b = c ∨ a - b = c) → 
    (∀ (d : ℚ), (d = c ∧ d ≠ ↑(d.num / d.den)) → 
      ∃ (e : ℚ), e ≠ d ∧ e = c ∧ e.den < d.den)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_zero_removal_false_l549_54953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_path_theorem_l549_54926

/-- A circular billiard table -/
structure CircularTable where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A ball on the circular billiard table -/
structure Ball where
  position : ℝ × ℝ
  velocity : ℝ × ℝ

/-- The path of the ball is a sequence of points -/
def BallPath := ℕ → ℝ × ℝ

/-- A point is on the path if it appears in the sequence -/
def OnPath (path : BallPath) (point : ℝ × ℝ) : Prop :=
  ∃ n : ℕ, path n = point

/-- A point appears infinitely often on the path -/
def AppearsInfinitely (path : BallPath) (point : ℝ × ℝ) : Prop :=
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ path n = point

/-- The theorem to be proved -/
theorem ball_path_theorem (table : CircularTable) (ball : Ball) (path : BallPath) 
    (reflects_at_edge : ∀ n : ℕ, (path n).1^2 + (path n).2^2 = table.radius^2 → 
      ∃ v : ℝ × ℝ, path (n+1) = path n + v ∧ path (n+2) = path (n+1) + v)
    (point : ℝ × ℝ) :
    (∃ n₁ n₂ n₃ : ℕ, n₁ < n₂ ∧ n₂ < n₃ ∧ 
      path n₁ = point ∧ path n₂ = point ∧ path n₃ = point) →
    AppearsInfinitely path point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_path_theorem_l549_54926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_four_digit_l549_54939

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem min_ratio_four_digit :
  ∀ x : ℕ, is_four_digit x →
    (x : ℚ) / (digit_sum x : ℚ) ≥ 1099 / (digit_sum 1099) :=
by
  sorry

#check min_ratio_four_digit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_four_digit_l549_54939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l549_54935

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the line
def myLine (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x + Real.sqrt 3

-- State the theorem
theorem tangent_line_to_circle :
  (∃ (x y : ℝ), myLine x y ∧ myCircle x y) ∧  -- The line intersects the circle
  (∀ (x y : ℝ), myLine x y → myCircle x y → 
    ∀ (x' y' : ℝ), x' ≠ x → y' ≠ y → myLine x' y' → ¬ myCircle x' y') ∧  -- The line is tangent to the circle
  myLine 0 (Real.sqrt 3)  -- The line passes through (0, √3)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l549_54935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_value_l549_54906

-- Define the polynomial division operation
noncomputable def poly_divide (p q : ℝ → ℝ) : (ℝ → ℝ) × ℝ :=
  sorry

-- Define x⁶
noncomputable def x_pow_6 : ℝ → ℝ := λ x => x^6

-- Define (x - 1/3)
noncomputable def x_minus_one_third : ℝ → ℝ := λ x => x - 1/3

-- Define (x - 1/4)
noncomputable def x_minus_one_fourth : ℝ → ℝ := λ x => x - 1/4

theorem remainder_value :
  let (q₁, r₁) := poly_divide x_pow_6 x_minus_one_third
  let (q₂, r₂) := poly_divide q₁ x_minus_one_fourth
  r₂ = 1/1024 := by
  sorry

#check remainder_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_value_l549_54906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_A_percentage_l549_54955

/-- Represents the percentage of respondents who liked product A -/
def P_A : ℝ := sorry

/-- Represents the percentage of respondents who liked product B -/
def P_B : ℝ := sorry

/-- Represents the unknown percentage X in the problem -/
def X : ℝ := sorry

/-- The minimum number of people surveyed -/
def min_surveyed : ℕ := 100

theorem product_A_percentage :
  (P_B = X - 20) →
  (P_A + P_B - 23 = 77) →
  (P_A = 100 - X + 20) :=
by
  intro h1 h2
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_A_percentage_l549_54955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l549_54937

-- Problem 1
theorem problem_1 : (Real.pi - 2)^0 - abs 8 + (1/3)^(-2 : ℤ) = 2 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) : 
  (2*x)^3 * (-3*x*y^2) / (-2*x^2*y^2) = 12*x^2 := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (4 - x)^2 - (x - 2)*(x + 3) = -9*x + 22 := by sorry

-- Problem 4
theorem problem_4 : (125 : ℕ)^2 - 124 * 126 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l549_54937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_f_nonnegative_condition_f_nonnegative_x_range_l549_54952

noncomputable def f (a x : ℝ) : ℝ := Real.cos x ^ 2 + a * Real.sin x + a + 1

noncomputable def g (a : ℝ) : ℝ := if a ≥ 0 then 1 else 2 * a + 1

theorem f_minimum_value (a : ℝ) : 
  ∀ x, f a x ≥ g a := by sorry

theorem f_nonnegative_condition :
  (∀ a x, f a x ≥ 0) ↔ a ≥ -1/2 := by sorry

theorem f_nonnegative_x_range (a : ℝ) (h : a ∈ Set.Icc (-2) 0) :
  ∀ k : ℤ, ∀ x ∈ Set.Icc (2 * k * Real.pi - Real.pi) (2 * k * Real.pi), f a x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_f_nonnegative_condition_f_nonnegative_x_range_l549_54952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l549_54982

noncomputable def f (x : ℝ) := Real.sqrt (2 * Real.sin x + 1)

theorem domain_of_f :
  {x : ℝ | ∃ k : ℤ, -π/6 + 2*π*k ≤ x ∧ x ≤ 7*π/6 + 2*π*k} = {x : ℝ | f x ∈ Set.univ} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l549_54982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log400_not_computable_l549_54919

-- Define the given logarithms
noncomputable def log8 : ℝ := Real.log 8
noncomputable def log9 : ℝ := Real.log 9
noncomputable def log7 : ℝ := Real.log 7

-- Define a function that checks if a logarithm can be computed using given values
def can_compute (x : ℝ) : Prop := ∃ (f : ℝ → ℝ → ℝ → ℝ), x = f log8 log9 log7

-- Theorem statement
theorem log400_not_computable :
  can_compute (Real.log 21) ∧
  can_compute (Real.log (9/8)) ∧
  can_compute (Real.log 126) ∧
  can_compute (Real.log 0.875) ∧
  ¬ can_compute (Real.log 400) :=
by
  sorry

#check log400_not_computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log400_not_computable_l549_54919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l549_54944

/-- Helper function to calculate the angle BAC given points A, B, and C -/
noncomputable def angle_BAC (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given a circle with equation x^2 + y^2 = 1 and a point A(1, 0), 
    consider an inscribed triangle ABC where angle BAC = 60°.
    As BC moves on the circle, the locus of its midpoint satisfies 
    x^2 + y^2 = 1/4 with x < 1/4. -/
theorem midpoint_locus (x y : ℝ) : 
  (∃ (B C : ℝ × ℝ), 
    (B.1^2 + B.2^2 = 1) ∧ 
    (C.1^2 + C.2^2 = 1) ∧
    (angle_BAC (1, 0) B C = 60) ∧
    (x = (B.1 + C.1) / 2) ∧
    (y = (B.2 + C.2) / 2)) →
  (x^2 + y^2 = 1/4 ∧ x < 1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l549_54944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l549_54930

noncomputable def y₁ (x : ℝ) : ℝ := 4 * x + 1
noncomputable def y₂ (x : ℝ) : ℝ := x + 2
noncomputable def y₃ (x : ℝ) : ℝ := -2 * x + 4

noncomputable def f (x : ℝ) : ℝ := min (y₁ x) (min (y₂ x) (y₃ x))

theorem max_value_of_f :
  ∃ (M : ℝ), M = 8/3 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l549_54930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l549_54979

theorem function_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x + y) : 
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l549_54979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_points_l549_54964

structure Student where
  name : String
  points : ℚ

def tournament_conditions (students : List Student) : Prop :=
  students.length = 4 ∧
  (∀ s, s ∈ students → s.points ≥ 0 ∧ s.points ≤ 6) ∧
  (∀ s₁ s₂, s₁ ∈ students → s₂ ∈ students → s₁ ≠ s₂ → s₁.points ≠ s₂.points) ∧
  (∃ a d v s, a ∈ students ∧ d ∈ students ∧ v ∈ students ∧ s ∈ students ∧
    a.name = "Andrey" ∧ d.name = "Dima" ∧ v.name = "Vanya" ∧ s.name = "Sasha" ∧
    a.points > d.points ∧ d.points > v.points ∧ v.points > s.points) ∧
  (∃ a s, a ∈ students ∧ s ∈ students ∧ a.name = "Andrey" ∧ s.name = "Sasha" ∧ 
    (a.points - s.points) * 2 = ⌈a.points⌉ - ⌈s.points⌉) ∧
  (students.map (·.points)).sum = 12

theorem chess_tournament_points :
  ∀ students : List Student,
    tournament_conditions students →
    ∃ a d v s, a ∈ students ∧ d ∈ students ∧ v ∈ students ∧ s ∈ students ∧
      a.name = "Andrey" ∧ d.name = "Dima" ∧ v.name = "Vanya" ∧ s.name = "Sasha" ∧
      a.points = 4 ∧ d.points = (7:ℚ)/2 ∧ v.points = (5:ℚ)/2 ∧ s.points = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_points_l549_54964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l549_54902

theorem calculation_proof : 0.25 * (1/2)^(-2 : ℤ) + Real.log 8 / Real.log 10 + 3 * Real.log 5 / Real.log 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l549_54902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_beach_speed_l549_54909

/-- The time (in minutes) before the ice cream melts -/
noncomputable def melt_time : ℚ := 10

/-- The number of blocks to the beach -/
noncomputable def blocks_to_beach : ℚ := 16

/-- The length of each block in miles -/
noncomputable def block_length : ℚ := 1 / 8

/-- The required speed to reach the beach before the ice cream melts -/
noncomputable def required_speed : ℚ := 12

theorem ice_cream_beach_speed :
  required_speed = (blocks_to_beach * block_length) / (melt_time / 60) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_beach_speed_l549_54909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l549_54915

theorem absolute_value_nested_expression : 
  |(|-|(-2 + 3)| - 2| + 2)| = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l549_54915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_round_trip_time_l549_54942

/-- The distance between June's and Julia's houses in miles -/
noncomputable def distance_to_julia : ℝ := 2

/-- The time it takes June to ride to Julia's house in minutes -/
noncomputable def time_to_julia : ℝ := 6

/-- The distance between June's and Bernard's houses in miles -/
noncomputable def distance_to_bernard : ℝ := 5

/-- June's biking speed in miles per minute -/
noncomputable def june_speed : ℝ := distance_to_julia / time_to_julia

/-- The time it takes June to ride to Bernard's house and return in minutes -/
noncomputable def time_to_bernard_round_trip : ℝ := 2 * distance_to_bernard / june_speed

theorem june_round_trip_time :
  time_to_bernard_round_trip = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_round_trip_time_l549_54942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_theorem_l549_54978

noncomputable def point_distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

noncomputable def distance_between_points (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem point_distance_theorem (x y : ℝ) :
  y = 9 →
  distance_between_points x y 1 6 = 11 →
  x > 1 →
  point_distance_from_origin x y = Real.sqrt (194 + 32 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_theorem_l549_54978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l549_54903

/-- A line passing through point (3,4) with y-intercept twice the x-intercept -/
structure SpecialLine where
  a : ℝ
  b : ℝ
  eq : 3 / a + 4 / b = 1
  intercept_relation : b = 2 * a

/-- The area of the triangle formed by the line and the positive x and y axes -/
noncomputable def triangle_area (l : SpecialLine) : ℝ := (1/2) * l.a * l.b

/-- The theorem stating the minimum area of the triangle -/
theorem min_triangle_area (l : SpecialLine) : 
  ∀ ε > 0, triangle_area l ≥ 24 - ε := by
  sorry

#check min_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l549_54903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_reduction_percentage_l549_54921

noncomputable def first_reduction : ℝ := 25 / 100
noncomputable def second_reduction : ℝ := 70 / 100

theorem total_reduction_percentage : 
  let remaining_after_first := 1 - first_reduction
  let remaining_after_second := remaining_after_first * (1 - second_reduction)
  1 - remaining_after_second = 0.775 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_reduction_percentage_l549_54921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_eq_2_g_f_1_plus_f_3_eq_4_f_neg1_eq_f_neg3_not_always_l549_54972

-- Define the real numbers
noncomputable section
open Real

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define g' as the derivative of g
variable (g' : ℝ → ℝ)

-- Axioms based on the given conditions
axiom cond1 : ∀ x, f x + g' x = 2
axiom cond2 : ∀ x, f x - g' (4 - x) = 2
axiom g_even : ∀ x, g (-x) = g x

-- Theorems to prove
theorem f_4_eq_2 : f 4 = 2 := by sorry

theorem g'_2_eq_0 : g' 2 = 0 := by sorry

theorem f_1_plus_f_3_eq_4 : f 1 + f 3 = 4 := by sorry

-- This theorem states that f(-1) = f(-3) may not necessarily hold
theorem f_neg1_eq_f_neg3_not_always : ¬ (∀ f g g' : ℝ → ℝ, (∀ x, f x + g' x = 2) → (∀ x, f x - g' (4 - x) = 2) → (∀ x, g (-x) = g x) → f (-1) = f (-3)) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_eq_2_g_f_1_plus_f_3_eq_4_f_neg1_eq_f_neg3_not_always_l549_54972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_points_distance_l549_54925

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a point is inside a rectangle -/
def isInside (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

theorem rectangle_points_distance (r : Rectangle) (points : Finset Point) :
  r.width = 3 ∧ r.height = 4 ∧ points.card = 6 ∧ (∀ p ∈ points, isInside p r) →
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q ≤ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_points_distance_l549_54925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l549_54999

noncomputable def f (x : ℝ) : ℝ := min (-x^2) (x - 2)

theorem f_properties :
  (f 0 = -2 ∧ f 4 = -16) ∧
  {x : ℝ | f x > -4} = Set.Ioo (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l549_54999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_line_hyperbola_l549_54991

/-- Given two distinct real roots of a trigonometric equation, 
    prove that the line passing through points defined by these roots 
    has no intersection with a specific hyperbola. -/
theorem no_intersection_line_hyperbola (θ a b : ℝ) : 
  a ≠ b →
  a^2 * Real.cos θ + a * Real.sin θ = 0 →
  b^2 * Real.cos θ + b * Real.sin θ = 0 →
  ∀ x y : ℝ, 
    (y = -(Real.tan θ) * x) → 
    (x^2 / (Real.cos θ)^2 - y^2 / (Real.sin θ)^2 ≠ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_line_hyperbola_l549_54991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_coprime_in_sixteen_consecutive_integers_l549_54996

/-- For any set of 16 consecutive integers, there exists an element in the set
    that is coprime to all other elements in the set. -/
theorem exists_coprime_in_sixteen_consecutive_integers :
  ∀ (a : ℤ), ∃ (k : ℕ), k < 16 ∧
  (∀ (j : ℕ), j < 16 → j ≠ k → Int.gcd (a + k) (a + j) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_coprime_in_sixteen_consecutive_integers_l549_54996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l549_54949

theorem a_less_than_b (a b : ℝ) 
  (h1 : (3 : ℝ)^a + (13 : ℝ)^b = (17 : ℝ)^a) 
  (h2 : (5 : ℝ)^a + (7 : ℝ)^b = (11 : ℝ)^b) : 
  a < b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l549_54949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l549_54965

-- Define the hyperbola
structure Hyperbola where
  center : ℝ × ℝ
  eccentricity : ℝ
  vertex : ℝ × ℝ

-- Define the parabola
def parabola (x : ℝ) : ℝ := -8 * x

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  h.center = (0, 0) →
  h.eccentricity = 2 →
  h.vertex = (-2, 0) →
  ∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1 ↔ 
    ∃ t : ℝ, (x, y) = (h.center.1 + t, h.center.2 + h.eccentricity * t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l549_54965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cardinality_l549_54913

/-- The number formed by the last four digits of a natural number -/
def lastFourDigits (m : ℕ) : ℕ := m % 10000

/-- The set of odd natural numbers less than 10000 -/
def oddNaturalsLessThan10000 : Finset ℕ :=
  Finset.filter (fun n => n % 2 = 1) (Finset.range 10000)

/-- The set of numbers n where the last four digits of n^9 is greater than n -/
def greaterSet : Finset ℕ :=
  oddNaturalsLessThan10000.filter (fun n => lastFourDigits (n^9) > n)

/-- The set of numbers n where the last four digits of n^9 is less than n -/
def lesserSet : Finset ℕ :=
  oddNaturalsLessThan10000.filter (fun n => lastFourDigits (n^9) < n)

theorem equal_cardinality : Finset.card greaterSet = Finset.card lesserSet := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cardinality_l549_54913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l549_54933

/-- Represents an ellipse with center at origin and axes along coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope k passing through (4,0) -/
def Line (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * (p.1 - 4)}

theorem ellipse_theorem (C : Ellipse) :
  (∃ (v f₁ f₂ : ℝ × ℝ), 
    v.1^2 / C.a^2 + v.2^2 / C.b^2 = 1 ∧  -- v is on the ellipse
    v.2 = C.b ∧                         -- v is the top vertex
    f₁.1^2 / C.a^2 + f₁.2^2 / C.b^2 = 1 ∧  -- f₁ is on the ellipse
    f₂.1^2 / C.a^2 + f₂.2^2 / C.b^2 = 1 ∧  -- f₂ is on the ellipse
    (v.1 - f₁.1)^2 + (v.2 - f₁.2)^2 = 4 ∧  -- distance v to f₁ is 2
    (v.1 - f₂.1)^2 + (v.2 - f₂.2)^2 = 4 ∧  -- distance v to f₂ is 2
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 4  -- distance f₁ to f₂ is 2
  ) →
  (C.a = 2 ∧ C.b = Real.sqrt 3) ∧  -- standard equation is x²/4 + y²/3 = 1
  (∀ k : ℝ, 
    (∃ A B : ℝ × ℝ, A ∈ Line k ∧ B ∈ Line k ∧ 
      A.1^2 / C.a^2 + A.2^2 / C.b^2 = 1 ∧
      B.1^2 / C.a^2 + B.2^2 / C.b^2 = 1 ∧
      A.1 * B.1 + A.2 * B.2 > 1/2) →
    (-1/2 < k ∧ k < -3*Real.sqrt 3/14) ∨ (3*Real.sqrt 3/14 < k ∧ k < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l549_54933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jasons_music_store_purchase_l549_54956

noncomputable def calculate_total_spent (prices : List Float) (discounts : List Float) (tax_rate : Float) : Float :=
  let discounted_prices := List.zipWith (fun p d => p * (1 - d / 100)) prices discounts
  let total_before_tax := discounted_prices.sum
  let total_with_tax := total_before_tax * (1 + tax_rate / 100)
  total_with_tax

theorem jasons_music_store_purchase :
  let prices := [142.46, 8.89, 7.00, 35.25, 12.15, 14.99, 3.29]
  let discounts := [10.0, 5.0, 15.0, 20.0, 10.0, 25.0, 5.0]
  let tax_rate := 8.0
  (calculate_total_spent prices discounts tax_rate - 211.80).abs < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jasons_music_store_purchase_l549_54956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l549_54900

/-- The area of a triangle with vertices (1,4,5), (3,4,1), and (1,1,1) is √61 -/
theorem triangle_area : ∃ area : ℝ, area = Real.sqrt 61 := by
  let A : Fin 3 → ℝ := ![1, 4, 5]
  let B : Fin 3 → ℝ := ![3, 4, 1]
  let C : Fin 3 → ℝ := ![1, 1, 1]
  let AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]
  let AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]
  let cross_product : Fin 3 → ℝ := ![
    AB 1 * AC 2 - AB 2 * AC 1,
    AB 2 * AC 0 - AB 0 * AC 2,
    AB 0 * AC 1 - AB 1 * AC 0
  ]
  let area : ℝ := (1 / 2) * Real.sqrt ((List.ofFn cross_product).map (· ^ 2)).sum
  exists area
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l549_54900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l549_54912

/-- Piecewise function f(x) -/
noncomputable def f (x k a : ℝ) : ℝ :=
  if x ≥ 0 then x + k * (1 - a^2)
  else x^2 - 4*x + (3 - a)^2

/-- Condition for the existence of a unique x₂ for any non-zero x₁ -/
def unique_pair_exists (k a : ℝ) : Prop :=
  ∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f x₁ k a = f x₂ k a

/-- The range of k satisfies the given conditions -/
theorem k_range (k : ℝ) : 
  (∃ a : ℝ, unique_pair_exists k a) ↔ k ≤ 0 ∨ k ≥ 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l549_54912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_acute_triangle_set_l549_54984

def is_acute_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧ a^2 + b^2 > c^2

def contains_acute_triangle (S : Finset ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_acute_triangle a b c

theorem minimum_acute_triangle_set :
  ∀ S : Finset ℕ, S ⊆ Finset.range 2004 → S.card ≥ 29 → contains_acute_triangle S ∧
  ∀ T : Finset ℕ, T ⊆ Finset.range 2004 → T.card < 29 → ¬contains_acute_triangle T :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_acute_triangle_set_l549_54984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_mixture_pressure_l549_54974

/-- Represents the pressure of a gas mixture after temperature doubling and nitrogen dissociation -/
noncomputable def final_pressure (p : ℝ) : ℝ := (9 / 4) * p

/-- Theorem stating the final pressure of the gas mixture -/
theorem gas_mixture_pressure (p m V R T : ℝ) 
  (h1 : p > 0) (h2 : m > 0) (h3 : V > 0) (h4 : R > 0) (h5 : T > 0) :
  let v : ℝ := m / 28
  let initial_pressure : ℝ := (8 * v * R * T) / V
  let final_pressure : ℝ := (18 * v * R * T) / V
  initial_pressure = p → final_pressure = (9 / 4) * p :=
by sorry

#check gas_mixture_pressure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_mixture_pressure_l549_54974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l549_54989

/-- An arithmetic sequence with common difference d ≠ 0 -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : S seq 8 = S seq 13)
  (h2 : seq.a 15 + seq.a m = 0) :
  m = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l549_54989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_QR_formula_l549_54968

/-- Triangle PQR with median PS to side QR, altitude PT to side PR, area A, and known lengths PR and PS -/
structure Triangle (PR PS A : ℝ) where
  area_positive : A > 0
  side_positive : PR > 0
  median_positive : PS > 0

/-- The length of side QR in triangle PQR -/
noncomputable def length_QR (t : Triangle PR PS A) : ℝ :=
  2 * Real.sqrt (PS^2 - (2 * A / PR)^2)

/-- Theorem: The length of QR in triangle PQR is 2 √(PS² - (2A/PR)²) -/
theorem length_QR_formula (PR PS A : ℝ) (t : Triangle PR PS A) :
  length_QR t = 2 * Real.sqrt (PS^2 - (2 * A / PR)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_QR_formula_l549_54968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l549_54916

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define a circle with center (a, 0) and radius r
def circle_eq (x y a r : ℝ) : Prop := (x - a)^2 + y^2 = r^2

-- Define the condition that the circle passes through three vertices of the ellipse
def passes_through_vertices (a r : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ ellipse x₃ y₃ ∧
    circle_eq x₁ y₁ a r ∧ circle_eq x₂ y₂ a r ∧ circle_eq x₃ y₃ a r

-- Theorem statement
theorem circle_equation :
  ∀ a r : ℝ,
    a > 0 →  -- Center lies on positive semi-axis of x
    passes_through_vertices a r →
    a = 3/4 ∧ r^2 = 25/16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l549_54916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_iff_a_in_range_l549_54976

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x - |x^2 - a*x + 1|

-- Define the property of having exactly two zeros
def has_exactly_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ ∀ z : ℝ, f z = 0 → z = x ∨ z = y

-- State the theorem
theorem f_two_zeros_iff_a_in_range (a : ℝ) :
  has_exactly_two_zeros (f a) ↔ a ∈ {x : ℝ | x < 0 ∨ (0 < x ∧ x < 1) ∨ x > 1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_iff_a_in_range_l549_54976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_tournament_winner_l549_54971

/-- Represents a team in the volleyball tournament -/
structure Team where
  id : ℕ
deriving Repr, DecidableEq

/-- Represents the result of a game between two teams -/
inductive GameResult where
  | Win
  | Loss
deriving Repr, DecidableEq

/-- Represents the tournament -/
structure Tournament where
  teams : Finset Team
  results : Team → Team → GameResult
  total_teams : teams.card = 110
  played_all : ∀ t1 t2 : Team, t1 ≠ t2 → t1 ∈ teams → t2 ∈ teams → 
    (results t1 t2 = GameResult.Win ∧ results t2 t1 = GameResult.Loss) ∨
    (results t1 t2 = GameResult.Loss ∧ results t2 t1 = GameResult.Win)

/-- The main theorem -/
theorem volleyball_tournament_winner (t : Tournament) :
  (∀ s : Finset Team, s ⊆ t.teams → s.card = 55 → 
    ∃ team ∈ s, (s.filter (λ x => t.results team x = GameResult.Loss)).card ≤ 4) →
  ∃ team ∈ t.teams, (t.teams.filter (λ x => t.results team x = GameResult.Loss)).card ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_tournament_winner_l549_54971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l549_54966

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define point A on the line x = -4
def point_A : ℝ × ℝ := (-4, 0)

-- Define tangent point B
def point_B : ℝ × ℝ := (1, 2)

-- Define vectors PE and PF
def vector_PE (P : ℝ × ℝ) (lambda1 : ℝ) : ℝ × ℝ := (lambda1 * (point_A.1 - P.1), lambda1 * (point_A.2 - P.2))
def vector_PF (P : ℝ × ℝ) (lambda2 : ℝ) : ℝ × ℝ := (lambda2 * (point_B.1 - P.1), lambda2 * (point_B.2 - P.2))

-- Define the condition for lambda1 and lambda2
def lambda_condition (lambda1 lambda2 : ℝ) : Prop := 2/lambda1 + 3/lambda2 = 15 ∧ lambda1 > 0 ∧ lambda2 > 0

-- Define the locus equation for point Q
def locus_Q (x y : ℝ) : Prop := y^2 = (8/3) * (x + 1/3)

theorem locus_of_Q (p : ℝ) (P Q : ℝ × ℝ) (lambda1 lambda2 : ℝ) :
  p > 0 →
  parabola p P.1 P.2 →
  vector_PE P lambda1 = vector_PF P lambda2 →
  lambda_condition lambda1 lambda2 →
  locus_Q Q.1 Q.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l549_54966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l549_54946

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_line_intersection_slope 
  (A B : ℝ × ℝ) (k : ℝ) :
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  line k A.1 A.2 →
  line k B.1 B.2 →
  distance A focus = 3 * distance B focus →
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l549_54946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l549_54959

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x + 2) / (x - 4)

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (4 * x + 2) / (x - 3)

-- Theorem statement
theorem inverse_function_ratio :
  ∀ (a b c d : ℝ),
  (∀ x, x ≠ 3 → f_inv x = (a * x + b) / (c * x + d)) →
  a / c = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l549_54959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l549_54975

theorem trigonometric_problem (x : Real) 
  (h1 : -π < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  ((3 * (Real.sin (x/2))^2 - 2 * Real.sin (x/2) * Real.cos (x/2) + (Real.cos (x/2))^2) / (Real.tan x + 1 / Real.tan x) = -108/125) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l549_54975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_neg_alpha_l549_54908

theorem tan_neg_alpha (α : ℝ) (m : ℝ) :
  Real.sin α = 4/5 →
  ((-1 : ℝ), m) ∈ {(x, y) : ℝ × ℝ | x^2 + y^2 = 1 ∧ y / x = Real.tan α} →
  Real.tan (-α) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_neg_alpha_l549_54908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_zeros_l549_54927

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

theorem tangent_line_and_zeros (a m : ℝ) :
  (∀ x, x > 0 → (deriv (f a)) x = 2) →
  (f a) 1 = m + 2 →
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  a = 1 ∧ m = -1 ∧ -1 / Real.exp 1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_zeros_l549_54927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_plus_y_minus_one_l549_54981

/-- The angle of inclination of a line is the angle it makes with the positive x-axis -/
noncomputable def angle_of_inclination (a b c : ℝ) : ℝ :=
  Real.arctan (-a / b)

/-- A line is represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem angle_of_inclination_x_plus_y_minus_one (l : Line) :
  l.a = 1 → l.b = 1 → l.c = -1 →
  0 ≤ angle_of_inclination l.a l.b l.c ∧ 
  angle_of_inclination l.a l.b l.c < π →
  angle_of_inclination l.a l.b l.c = π / 4 * 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_plus_y_minus_one_l549_54981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_dates_2023_l549_54963

def count_valid_dates (year : Nat) : Nat :=
  let yy := year % 100
  let count_for_month (mm : Nat) : Nat :=
    if mm < yy then
      (yy - 1) - mm
    else
      0
  List.sum (List.map (fun m => count_for_month (m + 1)) (List.range 12))

theorem valid_dates_2023 :
  count_valid_dates 2023 = 186 := by
  -- Unfold the definition and simplify
  unfold count_valid_dates
  simp
  -- The rest of the proof would go here
  sorry

#eval count_valid_dates 2023

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_dates_2023_l549_54963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_largest_unshaded_area_l549_54983

-- Define the side length of the square
variable (s : ℝ)

-- Define the areas of the shapes
noncomputable def square_area (s : ℝ) : ℝ := s^2
noncomputable def circle_area (s : ℝ) : ℝ := Real.pi * (s/2)^2
noncomputable def triangle_area (s : ℝ) : ℝ := (s/2)^2 / 2

-- Define the unshaded areas
noncomputable def unshaded_square_area (s : ℝ) : ℝ := square_area s - circle_area s
noncomputable def unshaded_circle_area (s : ℝ) : ℝ := circle_area s - triangle_area s

-- Theorem statement
theorem circle_has_largest_unshaded_area (s : ℝ) (h : s > 0) :
  unshaded_circle_area s > unshaded_square_area s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_largest_unshaded_area_l549_54983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sum_sin_cos_l549_54924

theorem tan_value_from_sum_sin_cos (α : ℝ) (h1 : Real.sin α + Real.cos α = 7/13) (h2 : 0 < α) (h3 : α < Real.pi) :
  Real.tan α = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sum_sin_cos_l549_54924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l549_54992

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_range (ω φ : ℝ) :
  ω > 0 →
  0 < φ → φ < Real.pi / 2 →
  f ω φ 0 = Real.sqrt 2 / 2 →
  (∀ x₁ x₂ : ℝ, Real.pi / 2 < x₁ → x₁ < Real.pi → Real.pi / 2 < x₂ → x₂ < Real.pi → x₁ ≠ x₂ →
    (x₁ - x₂) / (f ω φ x₁ - f ω φ x₂) < 0) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l549_54992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_selling_price_l549_54985

noncomputable def total_cost : ℝ := 450
noncomputable def cost_book1 : ℝ := 262.5
noncomputable def loss_percentage : ℝ := 15
noncomputable def gain_percentage : ℝ := 19

noncomputable def cost_book2 : ℝ := total_cost - cost_book1

noncomputable def selling_price_book1 : ℝ := cost_book1 * (1 - loss_percentage / 100)
noncomputable def selling_price_book2 : ℝ := cost_book2 * (1 + gain_percentage / 100)

noncomputable def total_selling_price : ℝ := selling_price_book1 + selling_price_book2

theorem books_selling_price : total_selling_price = 446.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_selling_price_l549_54985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_theorem_l549_54923

/-- Represents the recipe and calorie information for lemonade --/
structure LemonadeInfo where
  lemon_juice_weight : ℚ
  sugar_weight : ℚ
  water_weight : ℚ
  lemon_juice_calories_per_100g : ℚ
  sugar_calories_per_100g : ℚ

/-- Calculates the number of calories in a given weight of lemonade --/
def calories_in_lemonade (info : LemonadeInfo) (weight : ℚ) : ℚ :=
  let total_weight := info.lemon_juice_weight + info.sugar_weight + info.water_weight
  let total_calories := (info.lemon_juice_weight / 100) * info.lemon_juice_calories_per_100g +
                        (info.sugar_weight / 100) * info.sugar_calories_per_100g
  (total_calories / total_weight) * weight

/-- Theorem stating that 300g of lemonade contains 258 calories --/
theorem lemonade_calories_theorem (info : LemonadeInfo) 
    (h1 : info.lemon_juice_weight = 150)
    (h2 : info.sugar_weight = 150)
    (h3 : info.water_weight = 450)
    (h4 : info.lemon_juice_calories_per_100g = 30)
    (h5 : info.sugar_calories_per_100g = 400) :
    calories_in_lemonade info 300 = 258 := by
  sorry

#eval calories_in_lemonade 
  { lemon_juice_weight := 150
    sugar_weight := 150
    water_weight := 450
    lemon_juice_calories_per_100g := 30
    sugar_calories_per_100g := 400 } 300

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_theorem_l549_54923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_production_revenue_l549_54951

theorem motorcycle_production_revenue (x : ℤ) :
  51 ≤ x ∧ x ≤ 59 → -20 * x^2 + 2200 * x > 60000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_production_revenue_l549_54951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_of_five_l549_54993

noncomputable def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sumOfArithmeticSequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_of_five (a : ℕ → ℝ) 
  (h1 : arithmeticSequence a) 
  (h2 : a 1 + a 5 = 6) : 
  sumOfArithmeticSequence a 5 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_of_five_l549_54993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_belt_size_conversion_l549_54995

def waist_size : ℚ := 42
def inches_per_foot : ℚ := 10
def cm_per_foot : ℚ := 25

theorem belt_size_conversion :
  Int.floor (waist_size / inches_per_foot * cm_per_foot) = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_belt_size_conversion_l549_54995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l549_54914

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4^x - 2^x + 1

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2,
  ∃ y ∈ Set.Icc (3/4 : ℝ) 13,
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (3/4 : ℝ) 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l549_54914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l549_54957

/-- Hyperbola C with equation x²/a² - y² = 1 (a > 0) -/
structure Hyperbola where
  a : ℝ
  a_pos : a > 0

/-- Right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Point A(0, -a) -/
def point_A (h : Hyperbola) : ℝ × ℝ := (0, -h.a)

/-- A point P on the left branch of the hyperbola -/
noncomputable def left_branch_point (h : Hyperbola) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity_range (h : Hyperbola) :
  (∃ (P : ℝ × ℝ), P = left_branch_point h ∧ 
    distance P (point_A h) + distance P (right_focus h) = 7) →
  eccentricity h ≥ Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l549_54957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_twelfth_terms_l549_54987

noncomputable def ArithmeticProgression (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | n => a₁ + (n - 1 : ℝ) * d

noncomputable def SumOfTerms (ap : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (ap 1 + ap n) / 2

theorem sum_of_fourth_and_twelfth_terms 
  (a₁ d : ℝ) 
  (h : SumOfTerms (ArithmeticProgression a₁ d) 15 = 90) : 
  ArithmeticProgression a₁ d 4 + ArithmeticProgression a₁ d 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_twelfth_terms_l549_54987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_lower_bound_l549_54947

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2  -- We define a_0 = 2 to handle the base case
  | n + 1 => sequence_a n * Real.sqrt ((sequence_a n ^ 3 + 2) / (2 * (sequence_a n ^ 3 + 1)))

theorem sequence_a_lower_bound (n : ℕ) (hn : n ≥ 1) : 
  sequence_a n > Real.sqrt (3 / n) := by
  sorry

#check sequence_a_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_lower_bound_l549_54947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_irreducible_polynomials_in_partition_l549_54920

/-- A structure representing a partition of ℕ+ into k subsets -/
structure Partition (k : ℕ) where
  subsets : Fin k → Set ℕ+
  pairwise_disjoint : ∀ i j, i ≠ j → Disjoint (subsets i) (subsets j)
  union_is_naturals : (⋃ i, subsets i) = Set.univ

/-- The theorem statement -/
theorem infinite_irreducible_polynomials_in_partition 
  (k n : ℕ) (h_k : k > 1) (h_n : n > 1) (p : Partition k) : 
  ∃ i : Fin k, ∃ S : Set (Polynomial ℤ), 
    (∀ f ∈ S, Irreducible f ∧ Polynomial.degree f = n ∧ 
      (∀ j l : Fin (n + 1), j ≠ l → f.coeff j ≠ f.coeff l) ∧
      (∀ j : Fin (n + 1), ∃ m : ℕ+, (m : ℤ) = f.coeff j ∧ m ∈ p.subsets i)) ∧
    Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_irreducible_polynomials_in_partition_l549_54920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_cut_square_dimensions_l549_54961

-- Define the square's side length
def square_side : ℝ := 10

-- Define the triangle's legs
def triangle_leg : ℝ := square_side

-- Define the triangle's hypotenuse
noncomputable def triangle_hypotenuse : ℝ := square_side * Real.sqrt 2

-- Theorem statement
theorem diagonal_cut_square_dimensions :
  let square := square_side
  let leg := triangle_leg
  let hypotenuse := triangle_hypotenuse
  square = 10 ∧ leg = 10 ∧ hypotenuse = 10 * Real.sqrt 2 := by
  -- Unfold the definitions
  unfold square_side triangle_leg triangle_hypotenuse
  -- Split the conjunction into separate goals
  apply And.intro
  · -- Prove square = 10
    rfl
  apply And.intro
  · -- Prove leg = 10
    rfl
  · -- Prove hypotenuse = 10 * Real.sqrt 2
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_cut_square_dimensions_l549_54961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l549_54940

noncomputable def plane (x y z : ℝ) : Prop := 5 * x - 3 * y + 7 * z = 40

def given_point : ℝ × ℝ × ℝ := (2, 1, 4)

noncomputable def closest_point : ℝ × ℝ × ℝ := (105/83, 238/83, 603/83)

theorem closest_point_on_plane :
  plane closest_point.1 closest_point.2.1 closest_point.2.2 ∧
  ∀ (p : ℝ × ℝ × ℝ), plane p.1 p.2.1 p.2.2 →
    (closest_point.1 - given_point.1)^2 + 
    (closest_point.2.1 - given_point.2.1)^2 + 
    (closest_point.2.2 - given_point.2.2)^2 ≤
    (p.1 - given_point.1)^2 + 
    (p.2.1 - given_point.2.1)^2 + 
    (p.2.2 - given_point.2.2)^2 :=
by sorry

#check closest_point_on_plane

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l549_54940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_satisfying_condition_l549_54973

def S (m : ℕ) : Set ℕ := {n : ℕ | 3 ≤ n ∧ n ≤ m}

def hasProductTriple (A : Set ℕ) : Prop :=
  ∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c

def satisfiesCondition (m : ℕ) : Prop :=
  ∀ (A B : Set ℕ), (A ∪ B = S m) → (A ∩ B = ∅) →
    hasProductTriple A ∨ hasProductTriple B

theorem smallest_m_satisfying_condition :
  ∀ m : ℕ, m ≥ 3 →
    (satisfiesCondition m ↔ m ≥ 243) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_satisfying_condition_l549_54973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_exists_l549_54970

-- Define the possible interpretations of the operations
inductive Operation
| Add
| SubLR  -- Left minus Right
| SubRL  -- Right minus Left

-- Define the expression type
inductive Expr
| Var : Char → Expr
| Op : Operation → Expr → Expr → Expr

def evaluate (op : Operation) (x y : ℝ) : ℝ :=
  match op with
  | Operation.Add => x + y
  | Operation.SubLR => x - y
  | Operation.SubRL => y - x

def interpretExpr (e : Expr) (opBang opQuery : Operation) (a b : ℝ) : ℝ :=
  match e with
  | Expr.Var c => 
      if c = 'a' then a
      else if c = 'b' then b
      else 0  -- Default case for other variables
  | Expr.Op op left right =>
      let leftVal := interpretExpr left opBang opQuery a b
      let rightVal := interpretExpr right opBang opQuery a b
      match op with
      | Operation.Add => evaluate opBang leftVal rightVal
      | Operation.SubLR => evaluate opQuery leftVal rightVal
      | Operation.SubRL => evaluate opQuery rightVal leftVal

theorem expression_exists : ∃ (e : Expr), 
  ∀ (opBang opQuery : Operation) (a b : ℝ),
  interpretExpr e opBang opQuery a b = 20 * a - 18 * b := by
  sorry

#check expression_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_exists_l549_54970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_below_eight_years_l549_54929

theorem percentage_below_eight_years (total_students : ℕ) (eight_year_olds : ℕ) :
  total_students = 50 →
  eight_year_olds = 24 →
  (total_students - (eight_year_olds + (2 * eight_year_olds) / 3)) * 100 / total_students = 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_below_eight_years_l549_54929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ellipse_intersection_and_area_maximization_l549_54962

/-- Given a line and an ellipse with specified properties, prove conditions about their intersection and area maximization. -/
theorem line_ellipse_intersection_and_area_maximization
  (k a : ℝ) 
  (ha : a > 0)
  (hl : ∀ x y : ℝ, y = k * (x + 1) → ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    A.1^2 + 3 * A.2^2 = a^2 ∧ 
    B.1^2 + 3 * B.2^2 = a^2 ∧
    A.2 = k * (A.1 + 1) ∧ 
    B.2 = k * (B.1 + 1))
  (hc : ∃ C : ℝ × ℝ, C.2 = 0 ∧ C.2 = k * (C.1 + 1))
  (hac : ∀ A B C : ℝ × ℝ, 
    A.1^2 + 3 * A.2^2 = a^2 ∧ 
    B.1^2 + 3 * B.2^2 = a^2 ∧ 
    A.2 = k * (A.1 + 1) ∧ 
    B.2 = k * (B.1 + 1) ∧ 
    C.2 = 0 ∧ C.2 = k * (C.1 + 1) →
    (C.1 - A.1, -A.2) = (2 * (B.1 - C.1), 2 * B.2)) :
  (a^2 > 3 * k^2 / (1 + 3 * k^2)) ∧
  ∃ max_area : ℝ, 
    (∀ A B : ℝ × ℝ, 
      A.1^2 + 3 * A.2^2 = a^2 ∧ 
      B.1^2 + 3 * B.2^2 = a^2 ∧ 
      A.2 = k * (A.1 + 1) ∧ 
      B.2 = k * (B.1 + 1) →
      abs (A.1 * B.2 - A.2 * B.1) / 2 ≤ max_area) ∧
    max_area = Real.sqrt 3 / 2 ∧
    a^2 = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ellipse_intersection_and_area_maximization_l549_54962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_travel_theorem_l549_54945

/-- Calculates the time in minutes for a bird to travel a given distance at a constant speed -/
noncomputable def bird_travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  (distance / speed) * 60

/-- Theorem stating that a bird flying at 8 miles per hour takes 22.5 minutes to travel 3 miles -/
theorem bird_travel_theorem :
  bird_travel_time 3 8 = 22.5 := by
  -- Unfold the definition of bird_travel_time
  unfold bird_travel_time
  -- Simplify the arithmetic
  simp [div_mul_eq_mul_div]
  -- Prove the equality
  norm_num

-- Cannot use #eval with noncomputable functions
/-- Approximate calculation of bird travel time for 3 miles at 8 mph -/
def approx_bird_travel_time : ℚ :=
  (3 : ℚ) / 8 * 60

#eval approx_bird_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_travel_theorem_l549_54945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l549_54997

/-- Represents a parabola y^2 = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- Represents a point (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
noncomputable def focus (parabola : Parabola) : Point :=
  { x := parabola.p / 2, y := 0 }

/-- The directrix of a parabola -/
noncomputable def directrix (parabola : Parabola) : ℝ := -parabola.p / 2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a parabola y^2 = 2px with p > 0, if there exists a point P(6, y) on the parabola
    such that |PF| = 8, where F is the focus, then the distance from F to the directrix is 4. -/
theorem parabola_focus_directrix_distance (parabola : Parabola) 
    (P : Point) (h1 : P.x = 6) (h2 : P.y^2 = 2 * parabola.p * P.x) 
    (h3 : distance P (focus parabola) = 8) : 
    (focus parabola).x - directrix parabola = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l549_54997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_and_inequality_l549_54928

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x * (Real.log x - 1) - x^2

-- Define the property of having exactly two extremum points
def has_two_extrema (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ (x = x₁ ∨ x = x₂))

-- Theorem statement
theorem extrema_and_inequality (a : ℝ) (h : has_two_extrema a) :
  a > 2 * Real.exp 1 ∧
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    (∀ l : ℝ, l ≥ 1 → Real.log x₁ + l * Real.log x₂ > 1 + l) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_and_inequality_l549_54928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_g_value_h_range_l549_54943

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2
noncomputable def g (x : ℝ) : ℝ := 1/2 + Real.sqrt 3 * Real.sin x * Real.cos x
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Theorem 1
theorem symmetry_implies_g_value (a : ℝ) :
  (∀ x, f (a + x) = f (a - x)) → g (2 * a) = 1/2 := by
  sorry

-- Theorem 2
theorem h_range :
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → 1/2 ≤ h x ∧ h x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_g_value_h_range_l549_54943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_decreasing_g_range_iff_t_l549_54905

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

noncomputable def g (x : ℝ) : ℝ :=
  if x > 0 then f x
  else if x = 0 then 5
  else -f (-x)

theorem f_odd (x : ℝ) (h : x ≠ 0) : f x = -f (-x) := by sorry

theorem f_decreasing (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ ≤ 2) :
  f x₁ > f x₂ := by sorry

theorem g_range_iff_t (t : ℝ) :
  (∀ x ∈ Set.Icc (-1) t, g x ≥ 5) ∧ (∃ x ∈ Set.Icc (-1) t, g x = 5) ↔ t ∈ Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_decreasing_g_range_iff_t_l549_54905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equations_coincide_with_curve_l549_54990

noncomputable def x (t : ℝ) : ℝ := Real.tan t
noncomputable def y (t : ℝ) : ℝ := 1 / Real.tan t

-- Theorem statement
theorem parametric_equations_coincide_with_curve :
  ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 →
  (∃ (t : ℝ), x = Real.tan t ∧ y = 1 / Real.tan t) ↔ x * y = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equations_coincide_with_curve_l549_54990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l549_54938

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- The first line: 3x + 4y - 5 = 0 -/
def line1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 5 = 0}

/-- The second line: 3x + 4y + 5 = 0 -/
def line2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 + 5 = 0}

/-- Theorem: The distance between line1 and line2 is 2 -/
theorem distance_between_lines : distance_parallel_lines 3 4 (-5) 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l549_54938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_games_ratio_is_one_tenth_l549_54911

/-- Represents the class of students with their game preferences -/
structure ClassInfo where
  total : ℕ
  basketball : ℕ
  volleyball : ℕ
  neither : ℕ

/-- The ratio of students playing both games to the total number of students -/
def both_games_ratio (c : ClassInfo) : ℚ :=
  let both := c.basketball + c.volleyball - (c.total - c.neither)
  both / c.total

/-- Theorem stating the ratio of students playing both games to the total number of students -/
theorem both_games_ratio_is_one_tenth (c : ClassInfo) 
  (h_total : c.total = 20)
  (h_basketball : c.basketball = c.total / 2)
  (h_volleyball : c.volleyball = c.total * 2 / 5)
  (h_neither : c.neither = 4) :
  both_games_ratio c = 1 / 10 := by
  sorry

#eval both_games_ratio { total := 20, basketball := 10, volleyball := 8, neither := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_games_ratio_is_one_tenth_l549_54911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_in_row_15_l549_54934

def pascal_triangle_row (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (λ k => Nat.choose n k)

theorem fifth_number_in_row_15 :
  (pascal_triangle_row 15).get? 4 = some 1365 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_in_row_15_l549_54934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_g_minimum_value_g_minimum_points_l549_54977

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := sqrt 3 * (sin x)^2 + sin x * cos x

-- Define the function g
def g (x : ℝ) : ℝ := sin (x/2 - π/3) - sqrt 3 / 2

-- Theorem for monotonicity of f
theorem f_monotone_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * π - π/12) (k * π + 5*π/12)) := by
  sorry

-- Theorem for minimum value of g
theorem g_minimum_value :
  ∀ x, g x ≥ -1 := by
  sorry

-- Theorem for x values where g attains its minimum
theorem g_minimum_points (k : ℤ) :
  g (2 * k * π - π/6) = -1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_g_minimum_value_g_minimum_points_l549_54977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_truck_income_l549_54994

/-- Represents the total income of a taco truck during lunch rush -/
def total_income (x y : ℝ) (n : ℕ) : ℝ :=
  4 * y + 3 * x + 2 * n * x

/-- Theorem stating that the total income of the taco truck during lunch rush
    is equal to 4y + 3x + 2nx, given the conditions of the problem -/
theorem taco_truck_income (x y : ℝ) (n : ℕ) :
  total_income x y n = 4 * y + 3 * x + 2 * n * x :=
by
  -- Unfold the definition of total_income
  unfold total_income
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_truck_income_l549_54994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_line_l549_54986

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (skew : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_lines : Line → Line → Prop)
variable (line_not_in_plane : Line → Plane → Prop)
variable (planes_intersect : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel_line : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem intersection_parallel_line 
  (m n l : Line) (α β : Plane) 
  (h1 : skew m n)
  (h2 : perp_line_plane m α)
  (h3 : perp_line_plane n β)
  (h4 : perp_lines l m)
  (h5 : perp_lines l n)
  (h6 : line_not_in_plane l α)
  (h7 : line_not_in_plane l β) :
  planes_intersect α β ∧ 
  ∃ i : Line, line_parallel_line i l ∧ 
          (∀ p : Plane, planes_intersect α β → p = α ∨ p = β → line_in_plane i p) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_line_l549_54986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_is_600cm_l549_54958

/-- Calculates the width of a wall given the dimensions of the wall and bricks, and the number of bricks needed. -/
noncomputable def wall_width (wall_height : ℝ) (wall_depth : ℝ) (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) (num_bricks : ℕ) : ℝ :=
  let brick_volume := brick_length * brick_width * brick_height
  let total_brick_volume := brick_volume * num_bricks
  total_brick_volume / (wall_height * wall_depth)

/-- Theorem stating that the width of the wall is 600 cm given the specified conditions. -/
theorem wall_width_is_600cm :
  let wall_height : ℝ := 800  -- 8 m in cm
  let wall_depth : ℝ := 22.5
  let brick_length : ℝ := 100
  let brick_width : ℝ := 11.25
  let brick_height : ℝ := 6
  let num_bricks : ℕ := 1600
  wall_width wall_height wall_depth brick_length brick_width brick_height num_bricks = 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_is_600cm_l549_54958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_eleven_fifteenths_l549_54907

-- Define the function g as noncomputable
noncomputable def g (a b : ℝ) : ℝ :=
  if a + b ≤ 4 then
    (a * b - 2 * a + 3) / (3 * a)
  else
    (a * b - 3 * b - 1) / (-3 * b)

-- Theorem statement
theorem g_sum_equals_eleven_fifteenths :
  g 3 1 + g 1 5 = 11 / 15 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_eleven_fifteenths_l549_54907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_negative_one_implies_fraction_equals_two_l549_54904

theorem tan_sum_negative_one_implies_fraction_equals_two 
  (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.tan (α + β) = -1) : 
  (Real.cos (β - α) - Real.sin (α + β)) / (Real.cos α * Real.cos β) = 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_negative_one_implies_fraction_equals_two_l549_54904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l549_54967

-- Define the power function as noncomputable
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x^a

-- State the theorem
theorem power_function_decreasing :
  ∃ a : ℝ, (power_function a 2 = 1/2) ∧
  (∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → power_function a x > power_function a y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → power_function a x > power_function a y) :=
by
  -- We claim that a = -1 satisfies the conditions
  use -1
  constructor
  · -- Prove that power_function (-1) 2 = 1/2
    simp [power_function]
    norm_num
  constructor
  · -- Prove decreasing on (-∞, 0)
    intros x y hxy
    simp [power_function]
    sorry -- Complete the proof here
  · -- Prove decreasing on (0, +∞)
    intros x y hxy
    simp [power_function]
    sorry -- Complete the proof here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l549_54967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_max_price_profit_function_quadratic_l549_54922

/-- Represents the daily sales and profit model for a children's toy --/
structure ToyModel where
  cost_price : ℚ
  max_profit_margin : ℚ
  base_price : ℚ
  base_sales : ℚ
  price_step : ℚ
  sales_step : ℚ

/-- Calculate the daily sales volume given a selling price --/
def daily_sales (model : ToyModel) (price : ℚ) : ℚ :=
  model.base_sales - (price - model.base_price) / model.price_step * model.sales_step

/-- Calculate the daily profit given a selling price --/
def daily_profit (model : ToyModel) (price : ℚ) : ℚ :=
  (daily_sales model price) * (price - model.cost_price)

/-- The main theorem stating the maximum profit and its corresponding price --/
theorem max_profit_at_max_price (model : ToyModel) 
  (h_cost : model.cost_price = 30)
  (h_margin : model.max_profit_margin = 1/2)
  (h_base_price : model.base_price = 35)
  (h_base_sales : model.base_sales = 350)
  (h_price_step : model.price_step = 5)
  (h_sales_step : model.sales_step = 50)
  (h_min_price : model.base_price ≤ 45) :
  ∃ (max_price : ℚ),
    max_price = model.cost_price * (1 + model.max_profit_margin) ∧
    daily_profit model max_price = 3750 ∧
    ∀ (price : ℚ), model.base_price ≤ price → daily_profit model price ≤ daily_profit model max_price :=
by sorry

/-- Theorem stating the quadratic form of the daily profit function --/
theorem profit_function_quadratic (model : ToyModel) 
  (h_cost : model.cost_price = 30)
  (h_base_price : model.base_price = 35)
  (h_base_sales : model.base_sales = 350)
  (h_price_step : model.price_step = 5)
  (h_sales_step : model.sales_step = 50) :
  ∀ (price : ℚ), daily_profit model price = -10 * price^2 + 1000 * price - 21000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_max_price_profit_function_quadratic_l549_54922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l549_54918

theorem definite_integral_exp_plus_2x : ∫ x in (Set.Icc 0 1), (Real.exp x + 2 * x) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_plus_2x_l549_54918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_neg_three_eq_three_F_neg_iff_x_in_range_l549_54960

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2 - x else x + 2

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := x * f x

-- Theorem 1: F(-3) = 3
theorem F_neg_three_eq_three : F (-3) = 3 := by sorry

-- Theorem 2: F(x) < 0 iff x ∈ (-2,0) ∪ (2,+∞)
theorem F_neg_iff_x_in_range : 
  ∀ x : ℝ, F x < 0 ↔ (x > -2 ∧ x < 0) ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_neg_three_eq_three_F_neg_iff_x_in_range_l549_54960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_f_60_l549_54950

/-- A function satisfying f(xy) = f(x)/y for all positive real numbers x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → 0 < y → f (x * y) = f x / y

theorem find_f_60 (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 48 = 36) :
  f 60 = 28.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_f_60_l549_54950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_b_not_40_19_l549_54931

/-- Two lines in 3D space --/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Determine if two lines are skew --/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∃ (t u : ℝ), ∀ i : Fin 3, l1.point i + t * l1.direction i ≠ l2.point i + u * l2.direction i

theorem lines_skew_iff_b_not_40_19 (b : ℝ) :
  let l1 : Line3D := ⟨![2, 3, b], ![3, 4, 5]⟩
  let l2 : Line3D := ⟨![5, 3, 1], ![7, 3, 2]⟩
  are_skew l1 l2 ↔ b ≠ 40 / 19 := by
  sorry

#check lines_skew_iff_b_not_40_19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_b_not_40_19_l549_54931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_is_regular_l549_54901

/-- A simple graph with 20 vertices and 100 edges -/
structure MySimpleGraph where
  V : Finset Nat
  E : Finset (Nat × Nat)
  simple : ∀ e ∈ E, e.1 ≠ e.2
  vertex_count : V.card = 20
  edge_count : E.card = 100

/-- The number of ways to choose a pair of non-intersecting edges -/
def nonIntersectingEdgePairs (G : MySimpleGraph) : ℕ := 4050

/-- A graph is regular if all vertices have the same degree -/
def isRegular (G : MySimpleGraph) : Prop :=
  ∃ d : ℕ, ∀ v ∈ G.V, (G.E.filter (λ e => e.1 = v ∨ e.2 = v)).card = d

/-- Main theorem: Given the conditions, the graph is regular -/
theorem graph_is_regular (G : MySimpleGraph) 
  (h : nonIntersectingEdgePairs G = 4050) : isRegular G := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_is_regular_l549_54901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_2F_l549_54948

open Real

-- Define F as a function of x
noncomputable def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

-- Define the substitution function
noncomputable def sub (x : ℝ) : ℝ := (2 * x) / (1 + x^2)

-- State the theorem
theorem G_equals_2F : 
  ∀ x : ℝ, x ≠ 1 → x ≠ -1 → 
  F (sub x) = 2 * F x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_2F_l549_54948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_y_value_l549_54969

theorem largest_y_value (x y : ℝ) (h1 : 4 < x) (h2 : x < 6) (h3 : 6 < y)
  (h4 : ∃ (n : ℕ), y < n) (h5 : ∀ (z : ℝ), 4 < z → z < 6 → 6 < y → (⌊y - z⌋ : ℤ) ≤ 5) :
  y ≤ 10 ∧ ∃ (y' : ℝ), y' = 10 ∧ 4 < x ∧ x < 6 ∧ 6 < y' ∧ (⌊y' - x⌋ : ℤ) ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_y_value_l549_54969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrogen_production_l549_54988

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  NaH : Moles
  H2O : Moles
  NaOH : Moles
  H2 : Moles
  ratio : NaH = H2O ∧ NaH = NaOH ∧ NaH = H2

/-- The theorem stating the number of moles of H₂ produced is equal to the number of moles of NaH -/
theorem hydrogen_production (r : Reaction) (h1 : r.NaH = (2 : ℝ)) (h2 : r.H2O = (36 : ℝ)) : r.H2 = r.NaH := by
  sorry

#check hydrogen_production

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrogen_production_l549_54988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_l549_54954

/-- Given an employee's salary increase and percentage, calculate the new annual salary -/
theorem salary_increase (increase : ℝ) (percent : ℝ) (new_salary : ℝ) : 
  increase = 5000 ∧ 
  percent = 80 ∧ 
  increase = (percent / 100) * (new_salary - increase) →
  new_salary = 11250 := by
  sorry

#check salary_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_l549_54954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_distances_l549_54910

noncomputable section

def P : ℝ × ℝ := (Real.sqrt 10 / 2, 0)

def ellipse (x y : ℝ) : Prop := x^2 + 12 * y^2 = 1

def line_through_P (α : ℝ) (x y : ℝ) : Prop :=
  y = Real.tan α * (x - P.1)

def intersection_points (α : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line_through_P α p.1 p.2}

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_product_of_distances :
  ∃ (α : ℝ), ∀ (M N : ℝ × ℝ),
    M ∈ intersection_points α →
    N ∈ intersection_points α →
    M ≠ N →
    distance P M * distance P N ≥ 19/20 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_distances_l549_54910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_sin_squared_plus_cos_squared_l549_54932

noncomputable def f (x : Real) : Real := Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3/4

theorem max_value_of_f :
  ∃ (M : Real), M = 1 ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ M) ∧
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = M) :=
by
  sorry

-- Note: The following is not part of the theorem, but included to show the given condition
theorem sin_squared_plus_cos_squared (x : Real) : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_sin_squared_plus_cos_squared_l549_54932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparison_l549_54917

/-- Sound pressure level definition -/
noncomputable def sound_pressure_level (p p₀ : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

/-- Theorem for sound pressure comparison -/
theorem sound_pressure_comparison 
  (p₀ p₁ p₂ p₃ : ℝ) 
  (h_p₀_pos : p₀ > 0)
  (h_gasoline : 60 ≤ sound_pressure_level p₁ p₀ ∧ sound_pressure_level p₁ p₀ ≤ 90)
  (h_hybrid : 50 ≤ sound_pressure_level p₂ p₀ ∧ sound_pressure_level p₂ p₀ ≤ 60)
  (h_electric : sound_pressure_level p₃ p₀ = 40) : 
  p₁ ≥ p₂ ∧ p₃ = 100 * p₀ ∧ p₁ ≤ 100 * p₂ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparison_l549_54917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_zero_l549_54980

/-- A triangle with sides 8, 15, and 17 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17

/-- The altitude from B to side AC -/
noncomputable def altitude (t : RightTriangle) : ℝ :=
  (2 * t.a * t.b) / t.c

/-- The ratio of HE to HA in the triangle -/
def ratio : ℝ := 0

/-- Theorem: The ratio HE:HA is 0 for the given triangle -/
theorem ratio_is_zero : ratio = 0 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_is_zero_l549_54980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cone_sphere_volume_l549_54936

theorem ice_cream_cone_sphere_volume (r h : Real) (hr : r > 0) (hh : h > 0) :
  let cone_volume := (1/3) * Real.pi * r^2 * h
  let sphere_radius := (cone_volume * 3 / (4 * Real.pi))^(1/3)
  sphere_radius = r * (3 * h / (4 * r))^(1/3) := by
  sorry

#check ice_cream_cone_sphere_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cone_sphere_volume_l549_54936
