import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_a_value_l733_73365

-- Define the circles
def circle1 (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 36
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the centers and radii of the circles
def center1 (a : ℝ) : ℝ × ℝ := (a, 0)
def center2 : ℝ × ℝ := (0, 2)
def radius1 : ℝ := 6
def radius2 : ℝ := 2

-- Define the condition for internal tangency
def internallyTangent (a : ℝ) : Prop :=
  ((center1 a).1 - center2.1)^2 + ((center1 a).2 - center2.2)^2 = (radius1 - radius2)^2

-- Theorem statement
theorem circles_internally_tangent_a_value :
  ∀ a : ℝ, internallyTangent a → (a = 2 * Real.sqrt 3 ∨ a = -2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_a_value_l733_73365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_general_term_formula_l733_73382

noncomputable def sequence_formula (n : ℕ) : ℝ := (1 - Real.cos (n * Real.pi)) + (n - 1) * (n - 2)

theorem invalid_general_term_formula :
  ¬(sequence_formula 1 = 2 ∧ 
    sequence_formula 2 = 0 ∧ 
    sequence_formula 3 = 2 ∧ 
    sequence_formula 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_general_term_formula_l733_73382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_cookie_count_l733_73380

-- Define the cookie shapes and their properties
structure Rectangle where
  length : ℝ
  width : ℝ

structure Triangle where
  base : ℝ
  height : ℝ

structure Circle where
  radius : ℝ

structure Square where
  side : ℝ

-- Define the friends' cookie data
def art_cookie : Rectangle := { length := 4, width := 3 }
def belle_cookie : Triangle := { base := 6, height := 2 }
def calvin_cookie : Circle := { radius := 2 }
def dana_cookie : Square := { side := 3 }

def art_count : ℕ := 10
def belle_count : ℕ := 20
def calvin_count : ℕ := 15

-- Define the area calculation functions
def rectangle_area (r : Rectangle) : ℝ := r.length * r.width
def triangle_area (t : Triangle) : ℝ := 0.5 * t.base * t.height
noncomputable def circle_area (c : Circle) : ℝ := Real.pi * c.radius * c.radius
def square_area (s : Square) : ℝ := s.side * s.side

-- Define the total dough amount (based on Art's cookies)
def total_dough : ℝ := art_count * rectangle_area art_cookie

-- Define Dana's desired earnings and price per cookie
def dana_desired_earnings : ℝ := 18
def price_per_cookie : ℝ := 0.9

-- Theorem: Dana must make 13 cookies
theorem dana_cookie_count :
  ∃ (n : ℕ), n = 13 ∧
  n * square_area dana_cookie ≤ total_dough ∧
  (n + 1) * square_area dana_cookie > total_dough ∧
  n * price_per_cookie < dana_desired_earnings ∧
  (n + 1) * price_per_cookie > dana_desired_earnings := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_cookie_count_l733_73380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l733_73318

/-- The area of the shaded region formed by four circles of radius 5 units intersecting at the origin -/
noncomputable def shaded_area : ℝ := 50 * (Real.pi - 2)

/-- The radius of each circle -/
def circle_radius : ℝ := 5

/-- Theorem stating the relationship between the shaded area and the circle radius -/
theorem shaded_area_theorem :
  ∀ (r : ℝ), r = circle_radius →
  shaded_area = 8 * (Real.pi * r^2 / 4 - r^2 / 2) :=
by
  intro r h
  simp [shaded_area, circle_radius, h]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l733_73318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charging_piles_growth_equation_l733_73302

/-- Represents the monthly average growth rate of smart charging piles -/
def x : ℝ := sorry

/-- Number of charging piles built in the first month -/
def first_month : ℕ := 301

/-- Number of charging piles built in the third month -/
def third_month : ℕ := 500

/-- Theorem stating the relationship between the number of charging piles
    built in the first and third months, given the monthly average growth rate -/
theorem charging_piles_growth_equation :
  (first_month : ℝ) * (1 + x)^2 = third_month := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charging_piles_growth_equation_l733_73302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_worth_of_specific_die_l733_73319

/-- Represents the probabilities and payoffs of an unfair die roll -/
structure UnfairDie where
  prob_less_than_6 : ℚ
  prob_6 : ℚ
  payoff_less_than_6 : ℚ
  payoff_6 : ℚ

/-- Calculates the expected worth of rolling the unfair die -/
def expectedWorth (d : UnfairDie) : ℚ :=
  d.prob_less_than_6 * d.payoff_less_than_6 + d.prob_6 * d.payoff_6

/-- The specific unfair die from the problem -/
def specificDie : UnfairDie :=
  { prob_less_than_6 := 5/6
  , prob_6 := 1/6
  , payoff_less_than_6 := 5
  , payoff_6 := -30
  }

/-- Theorem stating that the expected worth of rolling the specific unfair die is -5/6 -/
theorem expected_worth_of_specific_die :
  expectedWorth specificDie = -5/6 := by
  unfold expectedWorth specificDie
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_worth_of_specific_die_l733_73319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l733_73371

/-- Jana's walking rate in miles per minute -/
noncomputable def jana_rate : ℚ := 1 / 20

/-- Time Jana walks in minutes -/
def walk_time : ℚ := 15

/-- Function to round a rational number to the nearest tenth -/
def round_to_tenth (x : ℚ) : ℚ := 
  ⌊(x * 10 + 1/2)⌋ / 10

/-- Theorem stating Jana's walking distance rounded to the nearest tenth -/
theorem jana_walking_distance : 
  round_to_tenth (jana_rate * walk_time) = 8/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l733_73371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l733_73313

noncomputable section

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (1/3)x³ + ax² + x -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (1/3) * x^3 + a * x^2 + x

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ :=
  x^2 + 2 * a * x + 1

theorem odd_function_sum (a : ℝ) (h : IsOdd (f a)) : f a 3 + f' a 1 = 14 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l733_73313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unanswered_questions_count_l733_73301

/-- Represents the scoring system for the AHSME competition -/
structure ScoringSystem where
  correct : Int
  wrong : Int
  unanswered : Int

/-- Represents John's performance in the AHSME competition -/
structure Performance where
  correct : Nat
  wrong : Nat
  unanswered : Nat

/-- The new scoring system -/
def newScoring : ScoringSystem :=
  { correct := 5, wrong := 0, unanswered := 2 }

/-- The old scoring system -/
def oldScoring : ScoringSystem :=
  { correct := 4, wrong := -1, unanswered := 0 }

/-- Calculates the score based on the given scoring system and performance -/
def calculateScore (system : ScoringSystem) (perf : Performance) : Int :=
  system.correct * (perf.correct : Int) + system.wrong * (perf.wrong : Int) + system.unanswered * (perf.unanswered : Int)

/-- The main theorem to prove -/
theorem unanswered_questions_count 
  (perf : Performance)
  (h1 : perf.correct + perf.wrong + perf.unanswered = 30)
  (h2 : calculateScore newScoring perf = 93)
  (h3 : calculateScore oldScoring perf + 30 = 84) :
  perf.unanswered = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unanswered_questions_count_l733_73301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_matrix_det_zero_l733_73389

open Matrix Real

noncomputable def cosine_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![cos 2, cos 3, cos 4;
     cos 5, cos 6, cos 7;
     cos 8, cos 9, cos 10]

theorem cosine_matrix_det_zero 
  (h : ∀ n : ℤ, cos (n + 2) = 2 * cos 1 * cos (n + 1) - cos n) : 
  det cosine_matrix = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_matrix_det_zero_l733_73389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundreds_digit_of_factorial_expression_l733_73366

theorem hundreds_digit_of_factorial_expression : 
  ∃ n : ℕ, (Nat.factorial 25 - Nat.factorial 20 + Nat.factorial 10) % 1000 = 800 ∧ 
  (Nat.factorial 25 - Nat.factorial 20 + Nat.factorial 10) / 100 % 10 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundreds_digit_of_factorial_expression_l733_73366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_l733_73322

def operation (a b : ℕ+) : ℚ := (a - b) / (Nat.gcd a b : ℚ)

theorem equivalence (n : ℕ+) :
  (∀ m : ℕ+, m < n → Nat.gcd n (Int.natAbs ((operation n m).num)) = 1) ↔
  (∃ p k : ℕ, Nat.Prime p ∧ n = p^k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_l733_73322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_l733_73381

/-- The point P satisfies the given conditions -/
theorem point_P_satisfies_conditions :
  let P : ℝ × ℝ := (-5, 1)
  (P.1 + 2 * P.2 + 3 = 0) ∧
  (P.2 > P.1 - 1) ∧
  ((|2 * P.1 + P.2 - 6| / Real.sqrt 5 : ℝ) = 3 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_l733_73381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_ratio_l733_73317

/-- Predicate stating that four points form a rectangle -/
def IsRectangle (A B C D : ℝ × ℝ) : Prop := sorry

/-- Predicate stating that a point is the midpoint of two other points -/
def Midpoint (M A B : ℝ × ℝ) : Prop := sorry

/-- Predicate stating that a line is perpendicular to another line at a given point -/
def PerpendicularAt (P Q : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := sorry

/-- Type representing a line in 2D space -/
def Line (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Function to calculate the area of a quadrilateral -/
noncomputable def Area2D (quad : List (ℝ × ℝ)) : ℝ := sorry

/-- Constructor for a quadrilateral from four points -/
def Quadrilateral (A B C D : ℝ × ℝ) : List (ℝ × ℝ) := sorry

/-- Given a rectangle ABCD with midpoints K on BC and L on DA, and M as the intersection of the perpendicular from B to AK with CL, prove that the ratio of areas ABKM:ABCL is 2:3 -/
theorem rectangle_area_ratio (A B C D K L M : ℝ × ℝ) : 
  IsRectangle A B C D →
  Midpoint K B C →
  Midpoint L D A →
  PerpendicularAt B M (Line A K) →
  M ∈ Line C L →
  (Area2D (Quadrilateral A B K M)) / (Area2D (Quadrilateral A B C L)) = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_ratio_l733_73317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_in_A_P_l733_73308

/-- A polynomial of degree 8 -/
def Polynomial8 := Polynomial ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) := {x : ℝ | ∃ c : ℝ, P.eval x = c}

/-- Theorem: If 8 is in A_P for a polynomial P of degree 8, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (h : (8 : ℝ) ∈ A_P P) : 
  ∃ x : ℕ, Odd x ∧ (x : ℝ) ∈ A_P P := by
  sorry

#check min_odd_in_A_P

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_in_A_P_l733_73308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_after_doubling_surface_area_l733_73354

theorem sphere_volume_after_doubling_surface_area :
  ∀ (r : ℝ), 
    (4 * Real.pi * r^2 = 200 * Real.pi) →
    (4 / 3 : ℝ) * Real.pi * (Real.sqrt (2 * 50))^3 = (4000 / 3 : ℝ) * Real.pi :=
by
  intro r h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_after_doubling_surface_area_l733_73354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_l733_73390

/-- The number of days it takes to complete a project when two workers collaborate, 
    with one worker quitting before completion. -/
noncomputable def project_completion_time (a_days : ℝ) (b_days : ℝ) (quit_before : ℝ) : ℝ :=
  let combined_rate := 1 / a_days + 1 / b_days
  (1 + quit_before * (1 / b_days)) / combined_rate

/-- Theorem stating that under the given conditions, the project will be completed in 20 days. -/
theorem project_completion_theorem :
  project_completion_time 20 40 10 = 20 := by
  -- Unfold the definition of project_completion_time
  unfold project_completion_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_l733_73390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_given_segment_sum_l733_73315

-- Define the convex angle
structure ConvexAngle where
  vertex : Point
  side1 : Line
  side2 : Line
  is_convex : Prop  -- Changed from IsConvex to Prop

-- Define the position of point P relative to the angle
inductive PointPosition
  | Inside
  | ExteriorAdjacent
  | OnBoundary
  | AtVertexOrInsideAcute

-- Define the solution existence
inductive SolutionExistence
  | NoSolution
  | OneSolution
  | TwoSolutions

-- Main theorem
theorem line_through_point_with_given_segment_sum 
  (angle : ConvexAngle) 
  (P : Point) 
  (d : ℝ) 
  (pos : PointPosition) :
  ∃ (result : SolutionExistence),
    (pos = PointPosition.Inside → 
      (∃ (a b : ℝ), 
        (d - a - b > 2 * Real.sqrt (a * b) → result = SolutionExistence.TwoSolutions) ∧
        (d - a - b = 2 * Real.sqrt (a * b) → result = SolutionExistence.OneSolution) ∧
        (d - a - b < 2 * Real.sqrt (a * b) → result = SolutionExistence.NoSolution))) ∧
    (pos = PointPosition.ExteriorAdjacent → result = SolutionExistence.OneSolution) ∧
    (pos = PointPosition.OnBoundary → 
      (∃ (dist : ℝ), 
        (d > dist → result = SolutionExistence.OneSolution) ∧
        (d ≤ dist → result = SolutionExistence.NoSolution))) ∧
    (pos = PointPosition.AtVertexOrInsideAcute → result = SolutionExistence.NoSolution) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_given_segment_sum_l733_73315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l733_73303

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  (Real.sin (ω * x))^2 + Real.sqrt 3 * Real.sin (ω * x) * Real.sin (ω * x + Real.pi / 2)

def has_min_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q ∧ q < p → ∃ y, f (y + q) ≠ f y

theorem f_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_period : has_min_period (f ω) (Real.pi / 2)) :
  (∀ k : ℤ, ∀ x : ℝ, 
    (k : ℝ) * Real.pi / 2 - Real.pi / 12 ≤ x ∧ x ≤ (k : ℝ) * Real.pi / 2 + Real.pi / 6 →
    ∀ y : ℝ, x < y → f ω x ≤ f ω y) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 3 → 0 ≤ f ω x ∧ f ω x ≤ 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l733_73303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_exponent_l733_73329

/-- A power function that passes through the point (2, 1/4) -/
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

/-- The condition that the function passes through (2, 1/4) -/
def passes_through (α : ℝ) : Prop := power_function α 2 = 1/4

theorem power_function_exponent :
  ∃ α : ℝ, passes_through α ∧ α = -2 := by
  -- We'll use -2 as our α
  use -2
  constructor
  · -- Prove that the function passes through (2, 1/4)
    simp [passes_through, power_function]
    norm_num
  · -- Prove that α = -2
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_exponent_l733_73329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l733_73321

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log (1/2) else -x^2 - 2*x

def solution_set : Set ℝ :=
  {x | x ≥ 1 ∨ x = 0 ∨ x ≤ -2}

theorem f_inequality_solution :
  ∀ x : ℝ, f x ≤ 0 ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l733_73321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_theorem_l733_73374

/-- The number of arrangements for 5 people with exactly one person between A and B -/
def arrangements_count : ℕ := 36

/-- Total number of people -/
def total_people : ℕ := 5

/-- Number of people between A and B -/
def people_between : ℕ := 1

theorem arrangements_theorem :
  arrangements_count = 
    (Nat.factorial (total_people - 2 - people_between)) * 2 * (total_people - 2) :=
by
  -- The proof goes here
  sorry

#eval arrangements_count
#eval (Nat.factorial (total_people - 2 - people_between)) * 2 * (total_people - 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_theorem_l733_73374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_power_function_decreasing_when_negative_l733_73345

-- Define a power function
noncomputable def power_function (n : ℝ) : ℝ → ℝ := fun x ↦ x^n

-- Theorem 1: The graph of a power function cannot be in the fourth quadrant
theorem power_function_not_in_fourth_quadrant (n : ℝ) :
  ∀ x > 0, power_function n x ≥ 0 := by
  sorry

-- Theorem 2: When n < 0, the power function is decreasing in the first quadrant
theorem power_function_decreasing_when_negative (n : ℝ) (hn : n < 0) :
  ∀ x y, 0 < x → 0 < y → x < y → power_function n y < power_function n x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_power_function_decreasing_when_negative_l733_73345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_configuration_vertex_angle_l733_73304

/-- A cone with vertex A -/
structure Cone where
  vertex : ℝ × ℝ × ℝ
  vertexAngle : ℝ

/-- Configuration of three cones touching externally -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  tangentPlane : Set (ℝ × ℝ × ℝ)
  cone1_touches_cone2 : cone1.vertex = cone2.vertex
  cone1_touches_cone3 : cone1.vertex = cone3.vertex
  cone2_touches_cone3 : cone2.vertex = cone3.vertex
  cones_tangent_to_plane : cone1.vertex ∈ tangentPlane ∧ 
                           cone2.vertex ∈ tangentPlane ∧ 
                           cone3.vertex ∈ tangentPlane
  cones_on_same_side : True  -- Placeholder for the actual condition

/-- The theorem to be proved -/
theorem cone_configuration_vertex_angle 
  (config : ConeConfiguration) 
  (h1 : config.cone1.vertexAngle = config.cone2.vertexAngle) 
  (h2 : config.cone3.vertexAngle = Real.pi / 2) :
  config.cone1.vertexAngle = 2 * Real.arctan (4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_configuration_vertex_angle_l733_73304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_after_19_innings_l733_73305

/-- Represents the average score of a cricketer after a certain number of innings -/
def AverageScore (totalRuns : ℚ) (innings : ℕ) : ℚ :=
  totalRuns / innings

/-- Represents the scenario of a cricketer's score after 19 innings -/
def CricketerScenario (initialAverage : ℚ) : Prop :=
  let totalRunsBefore := initialAverage * 18
  let totalRunsAfter := totalRunsBefore + 99
  let newAverage := AverageScore totalRunsAfter 19
  newAverage = initialAverage + 4

theorem cricketer_average_after_19_innings :
  ∃ initialAverage : ℚ,
    CricketerScenario initialAverage ∧
    AverageScore (initialAverage * 18 + 99) 19 = 27 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_after_19_innings_l733_73305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_correct_l733_73334

/-- The original proposition about right triangles and complementary angles. -/
def original_proposition : String := "In a right triangle, the two acute angles are complementary"

/-- The inverse proposition of the original statement about triangles and complementary angles. -/
def inverse_proposition : String := "If a triangle has two acute angles that are complementary, then the triangle is a right triangle"

/-- Function to swap the hypothesis and conclusion of a proposition. -/
def swap_proposition (p : String) : String := p -- This is a placeholder, as we can't actually swap strings

/-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition. -/
theorem inverse_proposition_is_correct : 
  inverse_proposition = swap_proposition original_proposition := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_correct_l733_73334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_balls_count_l733_73355

/-- The number of ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes -/
def distribute_balls : ℕ := 9

/-- Theorem stating that the number of ways to distribute 6 indistinguishable balls
    into 4 indistinguishable boxes is 9 -/
theorem distribute_balls_count : distribute_balls = 9 := by
  -- Unfold the definition of distribute_balls
  unfold distribute_balls
  -- The proof is complete since we've defined distribute_balls as 9
  rfl

#eval distribute_balls  -- Should output 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_balls_count_l733_73355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvaneft_share_percentage_l733_73391

/-- Represents the share packages in the auction --/
structure SharePackage where
  razneft : ℝ
  dvaneft : ℝ
  trineft : ℝ

/-- Represents the price of shares for each company --/
structure SharePrices where
  razneft : ℝ
  dvaneft : ℝ
  trineft : ℝ

/-- The conditions of the auction --/
def AuctionConditions (sp : SharePackage) (prices : SharePrices) : Prop :=
  -- Total shares of Razneft and Dvaneft equals shares of Trineft
  sp.razneft + sp.dvaneft = sp.trineft
  -- Dvaneft package is 3 times cheaper than Razneft package
  ∧ prices.dvaneft * sp.dvaneft = prices.razneft * sp.razneft / 3
  -- Total cost of Razneft and Dvaneft equals cost of Trineft
  ∧ prices.razneft * sp.razneft + prices.dvaneft * sp.dvaneft = prices.trineft * sp.trineft
  -- Price difference between Razneft and Dvaneft share
  ∧ 10000 ≤ prices.razneft - prices.dvaneft ∧ prices.razneft - prices.dvaneft ≤ 18000
  -- Price range of Trineft share
  ∧ 18000 ≤ prices.trineft ∧ prices.trineft ≤ 42000

/-- The theorem to be proved --/
theorem dvaneft_share_percentage (sp : SharePackage) (prices : SharePrices) 
  (h : AuctionConditions sp prices) : 
  0.15 ≤ sp.dvaneft / (sp.razneft + sp.dvaneft + sp.trineft) 
  ∧ sp.dvaneft / (sp.razneft + sp.dvaneft + sp.trineft) ≤ 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvaneft_share_percentage_l733_73391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l733_73399

/-- Represents a repeating decimal with a repeating part and a non-repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingDenom : ℕ
  nonRepeatingNonneg : 0 ≤ nonRepeating
  repeatingNonneg : 0 ≤ repeating
  repeatingLessThanOne : repeating < 1
  repeatingDenomPositive : 0 < repeatingDenom

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.nonRepeating + d.repeating / (1 - 1 / d.repeatingDenom)

theorem sum_of_repeating_decimals :
  let d1 : RepeatingDecimal := ⟨0, 3/10, 10, by norm_num, by norm_num, by norm_num, by norm_num⟩
  let d2 : RepeatingDecimal := ⟨0, 14/1000, 1000, by norm_num, by norm_num, by norm_num, by norm_num⟩
  let d3 : RepeatingDecimal := ⟨0, 5/10000, 10000, by norm_num, by norm_num, by norm_num, by norm_num⟩
  (d1.toRational + d2.toRational + d3.toRational) = 3478 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l733_73399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l733_73376

/-- The constant term in the expansion of (1+2x^2)(x- 1/x)^8 -/
def constant_term : ℤ := -42

/-- The expression to be expanded -/
noncomputable def expression (x : ℝ) : ℝ := (1 + 2*x^2) * (x - 1/x)^8

theorem constant_term_proof :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = expression x) ∧ 
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  (c = constant_term) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l733_73376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l733_73377

/-- Given hyperbola C with properties as described -/
structure Hyperbola where
  /-- C has the same foci as the ellipse x²/35 + y²/10 = 1 -/
  foci_same_as_ellipse : ℝ → ℝ → Prop
  /-- The asymptotes of C are y = ±(4/3)x -/
  asymptotes : ℝ → ℝ → Prop
  /-- F₁ is the left focus of the hyperbola -/
  F₁ : ℝ × ℝ
  /-- P is a point on the right branch of the hyperbola C -/
  P : ℝ × ℝ
  /-- The midpoint of segment PF₁ lies on the y-axis -/
  midpoint_on_y_axis : Prop

/-- The standard equation of the hyperbola C -/
def standard_equation (C : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2/9 - y^2/16 = 1

/-- The area of triangle PF₁F₂ -/
noncomputable def triangle_area (C : Hyperbola) : ℝ := 80/3

/-- Main theorem stating the properties of the hyperbola C -/
theorem hyperbola_properties (C : Hyperbola) :
  (∀ x y, standard_equation C x y ↔ 
    C.foci_same_as_ellipse x y ∧ C.asymptotes x y) ∧
  triangle_area C = 80/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l733_73377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_satisfies_condition_l733_73361

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop := 3 * x^2 - y^2 = 12

/-- The distance from a point (x, y) to A(4, 0) -/
noncomputable def distToA (x y : ℝ) : ℝ := Real.sqrt ((x - 4)^2 + y^2)

/-- The distance from a point (x, y) to the line x=1 -/
def distToLine (x y : ℝ) : ℝ := abs (x - 1)

/-- Theorem: The trajectory of point M satisfies the given condition -/
theorem trajectory_satisfies_condition :
  ∀ x y : ℝ, trajectory x y ↔ distToA x y = 2 * distToLine x y := by
  sorry

#check trajectory_satisfies_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_satisfies_condition_l733_73361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l733_73378

/-- An isosceles trapezoid with given dimensions has a diagonal of length 2√96. -/
theorem isosceles_trapezoid_diagonal (a b c : ℝ) (h1 : a = 24) (h2 : b = 10) (h3 : c = 12) :
  let d := Real.sqrt ((a - b) ^ 2 / 4 + c ^ 2)
  d = 2 * Real.sqrt 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l733_73378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_time_difference_l733_73369

noncomputable def commute_times (x y : ℝ) : List ℝ := [x, y, 10, 11, 9]

noncomputable def average (lst : List ℝ) : ℝ := (lst.sum) / lst.length

noncomputable def variance (lst : List ℝ) : ℝ :=
  let avg := average lst
  (lst.map (λ t => (t - avg)^2)).sum / lst.length

theorem commute_time_difference (x y : ℝ) 
  (h1 : average (commute_times x y) = 10)
  (h2 : variance (commute_times x y) = 2) :
  |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_time_difference_l733_73369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l733_73312

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def given_condition (t : Triangle) : Prop :=
  (2 * t.a + t.b) / t.c = cos (t.A + t.C) / cos t.C

/-- Helper function to calculate triangle area -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b * sin t.C

/-- Theorem stating the results of the problem -/
theorem triangle_problem (t : Triangle) 
  (h : given_condition t) : 
  t.C = 2 * π / 3 ∧ 
  (t.c = 2 → 
    (∃ (max_area : ℝ), max_area = Real.sqrt 3 / 3 ∧
      ∀ (t' : Triangle), t'.c = 2 → 
        triangle_area t' ≤ max_area ∧
        (triangle_area t' = max_area ↔ t'.a = 2 * Real.sqrt 3 / 3 ∧ t'.b = 2 * Real.sqrt 3 / 3))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l733_73312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_conditional_l733_73316

theorem negation_of_conditional (α : Real) : 
  ¬(α = π/6 → Real.sin α = 1/2) ↔ (α = π/6 ∧ Real.sin α ≠ 1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_conditional_l733_73316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segment_product_sum_eq_diameter_squared_l733_73393

/-- A semicircle with two intersecting chords from the ends of its diameter -/
structure SemicircleWithChords where
  /-- The diameter of the semicircle -/
  d : ℝ
  /-- The first chord -/
  c₁ : ℝ
  /-- The second chord -/
  c₂ : ℝ
  /-- The segment of c₁ adjacent to the diameter -/
  s₁ : ℝ
  /-- The segment of c₂ adjacent to the diameter -/
  s₂ : ℝ
  /-- The diameter is positive -/
  h_d_pos : 0 < d
  /-- The chords are positive -/
  h_c₁_pos : 0 < c₁
  h_c₂_pos : 0 < c₂
  /-- The segments are positive and smaller than their respective chords -/
  h_s₁_pos : 0 < s₁
  h_s₁_lt_c₁ : s₁ < c₁
  h_s₂_pos : 0 < s₂
  h_s₂_lt_c₂ : s₂ < c₂

/-- The theorem to be proved -/
theorem chord_segment_product_sum_eq_diameter_squared (sc : SemicircleWithChords) :
  sc.s₁ * sc.c₁ + sc.s₂ * sc.c₂ = sc.d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segment_product_sum_eq_diameter_squared_l733_73393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yavin_orbit_properties_l733_73349

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ
  apogee : ℝ

/-- Calculate the distance from the star when halfway through the orbit -/
noncomputable def halfwayDistance (orbit : EllipticalOrbit) : ℝ :=
  (orbit.apogee + orbit.perigee) / 2

/-- Calculate the length of the semi-minor axis -/
noncomputable def semiMinorAxis (orbit : EllipticalOrbit) : ℝ :=
  Real.sqrt ((orbit.apogee + orbit.perigee)^2 / 4 - (orbit.apogee - orbit.perigee)^2 / 4)

/-- Theorem about Yavin's orbit -/
theorem yavin_orbit_properties :
  let orbit : EllipticalOrbit := ⟨3, 15⟩
  halfwayDistance orbit = 9 ∧ semiMinorAxis orbit = 3 * Real.sqrt 5 := by
  sorry

#check yavin_orbit_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yavin_orbit_properties_l733_73349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_side_squares_l733_73363

/-- A parallelogram in 2D space -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_parallelogram : (B.1 - A.1, B.2 - A.2) = (C.1 - D.1, C.2 - D.2) ∧ 
                     (C.1 - B.1, C.2 - B.2) = (D.1 - A.1, D.2 - A.2)

/-- Distance between two points in 2D space -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: In any parallelogram, the sum of the squares of the sides 
    is equal to the sum of the squares of the diagonals -/
theorem parallelogram_diagonal_side_squares (p : Parallelogram) :
  (distance p.A p.C)^2 + (distance p.B p.D)^2 = 
  (distance p.A p.B)^2 + (distance p.B p.C)^2 + 
  (distance p.C p.D)^2 + (distance p.D p.A)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_side_squares_l733_73363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_fee_concession_percentage_l733_73332

/-- Proves that the percentage of students not getting a fee waiver who are eligible for half fee concession is 50% -/
theorem half_fee_concession_percentage (total_students : ℕ) 
  (boy_percentage : ℚ) (girl_percentage : ℚ)
  (boy_fee_waiver_percentage : ℚ) (girl_fee_waiver_percentage : ℚ)
  (fee_waiver_count : ℕ) (half_fee_concession_count : ℕ) :
  boy_percentage = 3/5 →
  girl_percentage = 2/5 →
  boy_fee_waiver_percentage = 3/20 →
  girl_fee_waiver_percentage = 3/40 →
  fee_waiver_count = 90 →
  half_fee_concession_count = 330 →
  let total_fee_waiver_percentage := boy_percentage * boy_fee_waiver_percentage + girl_percentage * girl_fee_waiver_percentage
  let non_fee_waiver_percentage := 1 - total_fee_waiver_percentage
  let half_fee_concession_percentage := (half_fee_concession_count : ℚ) / ((non_fee_waiver_percentage * total_students) : ℚ)
  half_fee_concession_percentage = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_fee_concession_percentage_l733_73332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_l733_73370

/-- Converts a base-7 number to decimal --/
def base7ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number --/
def base7Number : List Nat := [2, 1, 0, 0, 2, 0, 1, 1, 2]

/-- The decimal representation of the number --/
def decimalNumber : Nat := base7ToDecimal base7Number

/-- Checks if a number is prime --/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, 1 < m → m < n → ¬(n % m = 0)

theorem largest_prime_divisor :
  (∃ p : Nat, isPrime p ∧ p ∣ decimalNumber ∧
    ∀ q : Nat, isPrime q → q ∣ decimalNumber → q ≤ p) ∧
  (∃ p : Nat, p = 769 ∧ isPrime p ∧ p ∣ decimalNumber ∧
    ∀ q : Nat, isPrime q → q ∣ decimalNumber → q ≤ p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_l733_73370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_0_to_99_l733_73325

def sum_of_digits (n : Nat) : Nat :=
  List.sum (n.digits 10)

def sum_of_digits_range (start finish : Nat) : Nat :=
  List.sum (List.map sum_of_digits (List.range (finish - start + 1)))

theorem sum_of_digits_0_to_99 : sum_of_digits_range 0 99 = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_0_to_99_l733_73325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l733_73358

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line AB
def lineAB (x y : ℝ) : Prop := ∃ (k b : ℝ), y = k*x + b

-- Define the midpoint M
def M : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem line_equation : 
  ∀ (x y : ℝ),
  (∃ (xA yA xB yB : ℝ), 
    parabola xA yA ∧ 
    parabola xB yB ∧
    lineAB xA yA ∧ 
    lineAB xB yB ∧
    lineAB (focus.1) (focus.2) ∧
    M = (((xA + xB)/2), ((yA + yB)/2))) →
  lineAB x y →
  y = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l733_73358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_6_14_in_terms_of_a_and_b_l733_73362

theorem log_6_14_in_terms_of_a_and_b (a b : ℝ) 
  (h1 : Real.log 3 / Real.log 7 = a) 
  (h2 : Real.exp (b * Real.log 7) = 2) : 
  Real.log 14 / Real.log 6 = (b + 1) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_6_14_in_terms_of_a_and_b_l733_73362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_existence_l733_73335

theorem coloring_existence (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 0) :
  ∃ (f : Fin (p - 1) → Fin (2 * n)),
    ∀ (i : ℕ) (h_i : 2 ≤ i ∧ i ≤ n) (S : Finset (Fin (p - 1))),
      S.card = i →
        (∀ (x y : Fin (p - 1)), x ∈ S → y ∈ S → f x = f y) →
          ¬(p ∣ (S.sum (λ x => x.val + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_existence_l733_73335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l733_73343

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def is_acute_triangle (t : Triangle) : Prop := sorry

noncomputable def side_length (t : Triangle) (side : ℕ) : ℝ :=
  match side with
  | 1 => Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  | 2 => Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)
  | _ => Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.1)^2)

def triangle_area (t : Triangle) : ℝ := sorry

def angle_BAC (t : Triangle) : ℝ := sorry

def dot_product (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) 
  (h1 : is_acute_triangle t)
  (h2 : side_length t 3 = 4)
  (h3 : side_length t 2 = 1)
  (h4 : triangle_area t = Real.sqrt 3) :
  angle_BAC t = 60 ∧ dot_product t = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l733_73343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_passed_is_100_l733_73327

/-- Represents the road construction project parameters -/
structure RoadProject where
  totalLength : ℚ
  totalDays : ℚ
  initialWorkers : ℚ
  completedLength : ℚ
  extraWorkers : ℚ

/-- Calculates the number of days passed when progress was realized -/
def daysPassed (project : RoadProject) : ℚ :=
  (project.completedLength * project.totalDays) / 
  (project.totalLength * (1 + project.extraWorkers / project.initialWorkers))

/-- Theorem stating that the number of days passed is 100 for the given project parameters -/
theorem days_passed_is_100 (project : RoadProject) 
  (h1 : project.totalLength = 15)
  (h2 : project.totalDays = 300)
  (h3 : project.initialWorkers = 50)
  (h4 : project.completedLength = 5/2)
  (h5 : project.extraWorkers = 75) :
  daysPassed project = 100 := by
  sorry

/-- Example calculation -/
def exampleProject : RoadProject := {
  totalLength := 15,
  totalDays := 300,
  initialWorkers := 50,
  completedLength := 5/2,
  extraWorkers := 75
}

#eval daysPassed exampleProject

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_passed_is_100_l733_73327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_number_theorem_order_relations_pi_growth_order_l733_73396

-- Define the prime counting function
noncomputable def π : ℝ → ℝ := sorry

-- State the Prime Number Theorem
theorem prime_number_theorem : 
  ∀ ε > 0, ∃ X : ℝ, ∀ x ≥ X, |π x / (x / Real.log x) - 1| < ε := sorry

-- State the order relations
theorem order_relations : 
  ∀ δ > 0, ∃ Y : ℝ, ∀ y ≥ Y, Real.log (Real.log y) < δ * Real.log y ∧ Real.log y < δ * (y / Real.log y) := sorry

-- Theorem to prove
theorem pi_growth_order (ε : ℝ) (hε : ε > 0) :
  ∃ Z : ℝ, ∀ z ≥ Z, 
    |π z - (z / Real.log z)| < ε * (z / Real.log z) ∧
    |π z - Real.log z| > (1 - ε) * (z / Real.log z) ∧
    |π z - Real.log (Real.log z)| > (1 - ε) * (z / Real.log z) := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_number_theorem_order_relations_pi_growth_order_l733_73396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_MON_l733_73337

-- Define the ellipse and circle
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1
def circle' (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the point P on the ellipse in the first quadrant
noncomputable def P : ℝ → ℝ × ℝ
| θ => (4 * Real.cos θ, 3 * Real.sin θ)

-- Define the tangent line from P to the circle
def tangent_line (θ : ℝ) (x y : ℝ) : Prop :=
  x * Real.cos θ + y * Real.sin θ = 9

-- Define the intersection points M and N
noncomputable def M (θ : ℝ) : ℝ := 9 / Real.cos θ
noncomputable def N (θ : ℝ) : ℝ := 3 / Real.sin θ

-- Define the area of triangle MON
noncomputable def area_MON (θ : ℝ) : ℝ := (M θ * N θ) / 2

-- The main theorem
theorem min_area_MON :
  ∃ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi / 2) ∧
  (∀ (φ : ℝ), φ ∈ Set.Ioo 0 (Real.pi / 2) → area_MON θ ≤ area_MON φ) ∧
  area_MON θ = 27 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_MON_l733_73337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l733_73326

-- Define the function f(x) = x - 3 + log₃(x)
noncomputable def f (x : ℝ) : ℝ := x - 3 + (Real.log x) / (Real.log 3)

-- State the theorem
theorem zero_in_interval :
  ∃! x : ℝ, x > 1 ∧ x < 3 ∧ f x = 0 := by
  sorry

-- Additional helper lemmas that might be useful for the proof
lemma f_continuous : Continuous f := by
  sorry

lemma f_monotone : StrictMono f := by
  sorry

lemma f_neg_at_one : f 1 < 0 := by
  sorry

lemma f_pos_at_three : f 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l733_73326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_line_equation_optimal_line_equation_l733_73372

/-- A line passing through point P(4, 6) and intersecting the positive x and y axes -/
structure IntersectingLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 4 / a + 6 / b = 1

/-- The area of the triangle formed by the line and the axes -/
noncomputable def triangleArea (l : IntersectingLine) : ℝ := (1/2) * l.a * l.b

/-- The line that minimizes the triangle area -/
def minAreaLine : IntersectingLine where
  a := 8
  b := 12
  h1 := by norm_num
  h2 := by norm_num
  h3 := by norm_num

theorem min_area_line_equation :
  ∀ (l : IntersectingLine), triangleArea l ≥ triangleArea minAreaLine ∧
  (triangleArea l = triangleArea minAreaLine ↔ l = minAreaLine) :=
sorry

theorem optimal_line_equation (x y : ℝ) :
  (3 * x + 2 * y = 24) ↔
  ∃ (t : ℝ), x = 8 * t ∧ y = 12 * (1 - t) ∧ 0 ≤ t ∧ t ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_line_equation_optimal_line_equation_l733_73372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_extrema_l733_73384

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then (x + 1)^3 * Real.exp (x + 1) - Real.exp 1
  else -((-x + 1)^3 * Real.exp (-x + 1) - Real.exp 1)

-- State the theorem
theorem f_has_two_extrema :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    (∀ x, x ≠ x₁ ∧ x ≠ x₂ → 
      (∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≠ f x))) ∧
  (∀ x₃, (∀ x, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → 
    (∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≠ f x)) → x₃ = x₁ ∨ x₃ = x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_extrema_l733_73384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l733_73351

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then |x - 1| else 2 / x

theorem f_property (a : ℝ) (h : f a = f (a + 1)) : f (-2 * a) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l733_73351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sum_equality_l733_73350

theorem exponent_sum_equality : (8 : ℝ)⁻¹^(0 : ℝ) + (8 : ℝ)^((1 : ℝ)/3)^3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sum_equality_l733_73350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_nonempty_implies_t_less_than_one_l733_73383

theorem intersection_nonempty_implies_t_less_than_one (t : ℝ) : 
  (∅ : Set ℝ) ⊂ ({x : ℝ | x ≤ 1} ∩ {x : ℝ | x > t}) → t < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_nonempty_implies_t_less_than_one_l733_73383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_grazing_area_l733_73357

/-- Represents a circular park with a square monument -/
structure Park where
  parkDiameter : ℝ
  monumentSide : ℝ

/-- Represents a goat tied to the monument -/
structure Goat where
  ropeLength : ℝ

/-- Calculates the grazing area for a goat in the park -/
noncomputable def grazingArea (p : Park) (g : Goat) : ℝ :=
  (Real.pi * g.ropeLength^2) / 4 - (Real.pi * (p.monumentSide / 2)^2) / 4

/-- Theorem stating the grazing area for the given park and goat -/
theorem goat_grazing_area (p : Park) (g : Goat) 
    (h1 : p.parkDiameter = 50)
    (h2 : p.monumentSide = 10)
    (h3 : g.ropeLength = 20) :
  grazingArea p g = 93.75 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_grazing_area_l733_73357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l733_73341

def a : Fin 3 → ℝ := ![1, -2, 1]
def b : Fin 3 → ℝ := ![1, 0, 2]

theorem vector_subtraction :
  (a 0 - b 0 = 0) ∧ (a 1 - b 1 = -2) ∧ (a 2 - b 2 = -1) := by
  simp [a, b]
  norm_num

#check vector_subtraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l733_73341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_divisible_by_2_100_with_only_8_and_9_l733_73306

theorem exists_number_divisible_by_2_100_with_only_8_and_9 :
  ∃ n : ℕ, (2^100 ∣ n) ∧ (∀ d, d ∈ Nat.digits 10 n → d = 8 ∨ d = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_divisible_by_2_100_with_only_8_and_9_l733_73306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l733_73311

/-- The circle defined by x^2 + y^2 + 8x + 24 = 0 -/
def myCircle (x y : ℝ) : Prop :=
  x^2 + y^2 + 8*x + 24 = 0

/-- The parabola defined by y^2 = 8x -/
def myParabola (x y : ℝ) : Prop :=
  y^2 = 8*x

/-- The distance between two points -/
noncomputable def myDistance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem stating the smallest possible distance between a point on the circle and a point on the parabola -/
theorem smallest_distance :
  ∀ (xa ya xb yb : ℝ),
    myCircle xa ya →
    myParabola xb yb →
    myDistance xa ya xb yb ≥ 2 * Real.sqrt 11 - 4 :=
by sorry

#check smallest_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l733_73311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suitable_altitude_is_400_to_1200_l733_73359

/-- The suitable temperature range for the plant in degrees Celsius -/
noncomputable def temp_range : Set ℝ := Set.Icc 16 20

/-- The temperature drop rate in degrees Celsius per 100 meters of altitude -/
def temp_drop_rate : ℝ := 0.5

/-- The temperature at the foot of the mountain in degrees Celsius -/
def base_temp : ℝ := 22

/-- The temperature at a given altitude -/
noncomputable def temp_at_altitude (x : ℝ) : ℝ := base_temp - (x / 100) * temp_drop_rate

/-- The suitable altitude range for planting -/
noncomputable def suitable_altitude_range : Set ℝ := {x : ℝ | temp_at_altitude x ∈ temp_range}

theorem suitable_altitude_is_400_to_1200 : 
  suitable_altitude_range = Set.Icc 400 1200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suitable_altitude_is_400_to_1200_l733_73359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l733_73387

/-- Given a solution with initial volume, initial alcohol percentage, added alcohol, and added water,
    calculate the new alcohol percentage. -/
noncomputable def new_alcohol_percentage (initial_volume : ℝ) (initial_alcohol_percent : ℝ) 
                           (added_alcohol : ℝ) (added_water : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_alcohol_percent / 100
  let total_alcohol := initial_alcohol + added_alcohol
  let total_volume := initial_volume + added_alcohol + added_water
  (total_alcohol / total_volume) * 100

/-- Theorem stating that given a 40-liter solution with 5% alcohol, 
    adding 3.5 liters of alcohol and 6.5 liters of water results in a new solution with 11% alcohol. -/
theorem alcohol_solution_problem : 
  new_alcohol_percentage 40 5 3.5 6.5 = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l733_73387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l733_73394

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x + a)

-- State the theorem
theorem f_properties (a : ℝ) :
  -- f is decreasing on (0, e^(-a-1))
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (-a - 1) → f a x₁ > f a x₂) ∧
  -- f is increasing on (e^(-a-1), +∞)
  (∀ x₁ x₂, Real.exp (-a - 1) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  -- When a ≥ 1, f(x) < ae^x - 1 for all x > 0
  (a ≥ 1 → ∀ x, x > 0 → f a x < a * Real.exp x - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l733_73394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_value_l733_73330

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => -1 / (1 + sequence_a n)

theorem a_2018_value : sequence_a 2017 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_value_l733_73330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powerTower_increasing_and_bounded_l733_73307

noncomputable def powerTower : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.sqrt 2 ^ powerTower n

theorem powerTower_increasing_and_bounded :
  (∀ n : ℕ, powerTower n < powerTower (n + 1)) ∧
  (∀ n : ℕ, powerTower n < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_powerTower_increasing_and_bounded_l733_73307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_squared_l733_73336

-- Define the diameter of the pie
noncomputable def diameter : ℝ := 18

-- Define the number of equal pieces the pie is cut into
def num_pieces : ℕ := 4

-- Define the longest line segment in a piece
noncomputable def longest_segment (d : ℝ) (n : ℕ) : ℝ :=
  d * (Real.sqrt 2) / 2

-- Theorem statement
theorem longest_segment_squared (d : ℝ) (n : ℕ) :
  d = diameter → n = num_pieces →
  (longest_segment d n)^2 = 162 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_squared_l733_73336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_t_range_l733_73397

def f (a x : ℝ) : ℝ := x * abs (x - a) + 2 * x

theorem three_roots_t_range :
  ∃ a ∈ Set.Icc (-3 : ℝ) 3,
    (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
      f a x₁ = t * f a a ∧ f a x₂ = t * f a a ∧ f a x₃ = t * f a a) →
    t ∈ Set.Ioo 1 (25/24) := by
  sorry

#check three_roots_t_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_t_range_l733_73397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l733_73353

/-- Represents the average speed of a cyclist -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Proves that the average speed of a cyclist who travels 50 miles in 8 hours is 6.25 mph -/
theorem cyclist_average_speed :
  let distance : ℝ := 50
  let active_time : ℝ := 8
  average_speed distance active_time = 6.25 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l733_73353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l733_73340

-- Define the ellipse equation
noncomputable def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line equation
noncomputable def distance_to_line (x y : ℝ) : ℝ := |2*x - 3*y + 6| / Real.sqrt 13

-- Statement of the theorem
theorem min_distance_ellipse_to_line :
  ∃ (min_dist : ℝ), min_dist = (6 - Real.sqrt 13) / Real.sqrt 13 ∧
  ∀ (x y : ℝ), is_on_ellipse x y → distance_to_line x y ≥ min_dist :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l733_73340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_numbers_satisfy_triangle_condition_l733_73395

/-- A complex number z satisfies the triangle condition if 0, z, and z^2 form
    the three distinct vertices of an isosceles right triangle where z and z^2
    form the right angle. -/
def satisfies_triangle_condition (z : ℂ) : Prop :=
  z ≠ 0 ∧
  z ≠ z^2 ∧
  Complex.abs z = Complex.abs (z^2 - z) ∧
  (z.re * (z^2 - z).re + z.im * (z^2 - z).im = 0)

/-- There are exactly two complex numbers that satisfy the triangle condition. -/
theorem two_complex_numbers_satisfy_triangle_condition :
  ∃! (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, satisfies_triangle_condition z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_numbers_satisfy_triangle_condition_l733_73395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_is_translation_scaling_transformation_is_homothety_l733_73392

/-- A transformation with the property that the vector between image points is a scalar multiple of the vector between original points. -/
structure ScalingTransformation (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (f : V → V)
  (k : ℝ)
  (property : ∀ (A B : V), f B - f A = k • (B - A))

/-- Prove that a ScalingTransformation with k = 1 is a translation. -/
theorem scaling_transformation_is_translation
  {V : Type*} [AddCommGroup V] [Module ℝ V] (f : ScalingTransformation V)
  (h : f.k = 1) :
  ∃ (v : V), ∀ (x : V), f.f x = x + v :=
by
  sorry

/-- Prove that a ScalingTransformation with k ≠ 1 is a homothety. -/
theorem scaling_transformation_is_homothety
  {V : Type*} [AddCommGroup V] [Module ℝ V] (f : ScalingTransformation V)
  (h : f.k ≠ 1) :
  ∃ (O : V), ∀ (x : V), f.f x - O = f.k • (x - O) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_is_translation_scaling_transformation_is_homothety_l733_73392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l733_73338

theorem vector_angle_problem (α β : Real) 
  (h1 : ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = (2 * Real.sqrt 5 / 5)^2)
  (h2 : 0 < α ∧ α < Real.pi / 2)
  (h3 : -Real.pi / 2 < β ∧ β < 0)
  (h4 : Real.sin β = -5 / 13) : 
  Real.cos (α - β) = 3 / 5 ∧ Real.sin α = 33 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l733_73338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_two_implies_k_equals_one_l733_73346

theorem integral_equals_two_implies_k_equals_one (k : ℝ) : 
  (∫ x in (Set.Icc 0 1), 2*x + k) = 2 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_two_implies_k_equals_one_l733_73346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_proof_l733_73388

/-- The measure of each interior angle of a regular octagon is 135 degrees. -/
def regular_octagon_interior_angle : ℚ :=
  135

/-- A regular octagon has 8 sides. -/
def regular_octagon_sides : ℕ := 8

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℚ :=
  (n - 2) * 180

/-- The measure of each interior angle in a regular polygon is the sum of interior angles divided by the number of sides. -/
def interior_angle_measure (n : ℕ) : ℚ :=
  sum_interior_angles n / n

theorem regular_octagon_interior_angle_proof :
  interior_angle_measure regular_octagon_sides = regular_octagon_interior_angle := by
  sorry

#eval regular_octagon_interior_angle
#eval interior_angle_measure regular_octagon_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_proof_l733_73388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_is_six_point_five_l733_73375

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  width : ℚ
  length : ℚ
  perimeter : ℚ
  total_cost : ℚ
  length_width_relation : length = width + 10
  perimeter_formula : perimeter = 2 * (length + width)
  perimeter_value : perimeter = 180
  total_cost_value : total_cost = 1170

/-- The rate per meter for fencing the rectangular plot -/
def fencing_rate (plot : RectangularPlot) : ℚ :=
  plot.total_cost / plot.perimeter

/-- Theorem stating that the fencing rate is 6.5 for the given conditions -/
theorem fencing_rate_is_six_point_five (plot : RectangularPlot) :
  fencing_rate plot = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_is_six_point_five_l733_73375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_angle_l733_73328

/-- Line passing through a point with inclination angle α -/
structure Line (α : Real) where
  point : Prod Real Real

/-- Curve defined by y² = 2x -/
def Curve : Set (Prod Real Real) :=
  {p : Prod Real Real | p.2^2 = 2 * p.1}

/-- Distance between two points in R² -/
noncomputable def distance (p1 p2 : Prod Real Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_intersection_angle (α : Real) :
  let l : Line α := ⟨(-2, -4)⟩
  let intersections := (Set.inter Curve {p | ∃ t, p.1 = -2 + t * Real.cos α ∧ p.2 = -4 + t * Real.sin α})
  ∀ A B, A ∈ intersections → B ∈ intersections → A ≠ B →
    distance l.point A * distance l.point B = 40 →
      α = π / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_angle_l733_73328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l733_73300

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) + 1 / (x + 1)

-- Define the domain of f(x)
def domain_f : Set ℝ := {x | x ≥ -2 ∧ x ≠ -1}

-- Define f(x-1)
noncomputable def f_shifted (x : ℝ) : ℝ := Real.sqrt (x + 1) + 1 / x

-- Define the domain of f(x-1)
def domain_f_shifted : Set ℝ := {x | x ≥ -1 ∧ x ≠ 0}

theorem function_properties :
  (∀ x ∈ domain_f, f x = Real.sqrt (x + 2) + 1 / (x + 1)) ∧
  (domain_f = Set.Icc (-2) (-1) ∪ Set.Ioi (-1)) ∧
  (f (-2) = -1) ∧
  (∀ x ∈ domain_f_shifted, f (x - 1) = f_shifted x) ∧
  (domain_f_shifted = Set.Icc (-1) 0 ∪ Set.Ioi 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l733_73300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_center_movement_not_necessarily_straight_l733_73324

/-- Represents a circular robot -/
structure CircularRobot where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the movement of a point -/
structure Movement where
  start : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- Predicate to check if a movement is along a straight line -/
def isStraightLineMovement (m : Movement) : Prop :=
  ∃ (a b c : ℝ), a * m.start.1 + b * m.start.2 + c = 0 ∧
                  a * m.endPoint.1 + b * m.endPoint.2 + c = 0

/-- The movement of the robot -/
def robotMovement (initial final : CircularRobot) : Movement :=
  { start := initial.center
    endPoint := final.center }

/-- Predicate to check if all boundary points move along straight lines -/
def allBoundaryPointsStraightLine (initial final : CircularRobot) : Prop :=
  ∀ θ : ℝ, isStraightLineMovement {
    start := (initial.center.1 + initial.radius * Real.cos θ, initial.center.2 + initial.radius * Real.sin θ),
    endPoint := (final.center.1 + final.radius * Real.cos θ, final.center.2 + final.radius * Real.sin θ)
  }

/-- The main theorem -/
theorem robot_center_movement_not_necessarily_straight :
  ¬ (∀ (initial final : CircularRobot),
     allBoundaryPointsStraightLine initial final →
     isStraightLineMovement (robotMovement initial final)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_center_movement_not_necessarily_straight_l733_73324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l733_73367

/-- Parabola C: y^2 = 2x -/
def C (x y : ℝ) : Prop := y^2 = 2*x

/-- Line intersecting C -/
def intersectingLine (x : ℝ) : Prop := x = 2

/-- Point A is in the first quadrant -/
noncomputable def A : ℝ × ℝ := (2, 2)

/-- Point B -/
noncomputable def B : ℝ × ℝ := (2, -2)

/-- Line l passing through N on AB -/
def l (t b : ℝ) (x y : ℝ) : Prop := x = t*y + b

/-- Points C and D where l intersects C -/
def C_point (x₁ y₁ : ℝ) : Prop := C x₁ y₁ ∧ ∃ t b, l t b x₁ y₁
def D_point (x₂ y₂ : ℝ) : Prop := C x₂ y₂ ∧ ∃ t b, l t b x₂ y₂

/-- Slopes of AC and BD -/
noncomputable def k₁ (x₁ y₁ : ℝ) : ℝ := (y₁ - A.2) / (x₁ - A.1)
noncomputable def k₂ (x₂ y₂ : ℝ) : ℝ := (y₂ - B.2) / (x₂ - B.1)

/-- Given condition on slopes -/
def slope_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := 3 / k₁ x₁ y₁ + 1 / k₂ x₂ y₂ = 2

/-- Main theorem -/
theorem parabola_intersection_theorem (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : C_point x₁ y₁) (h₂ : D_point x₂ y₂) (h₃ : slope_condition x₁ y₁ x₂ y₂) :
  y₂ = -3*y₁ ∧ (8 * Real.sqrt 13) / 9 < Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ∧ 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) < 8 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l733_73367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l733_73386

/-- Given a point P(-√3, y) where y > 0 lying on the terminal side of angle α,
    if sin α = (√3/4)y, then cos α = -3/4 -/
theorem cos_alpha_value (y : ℝ) (α : ℝ) (h1 : y > 0) 
    (h2 : Real.sin α = (Real.sqrt 3 / 4) * y) : Real.cos α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l733_73386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l733_73309

-- Define the ellipse C
noncomputable def ellipse_C (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 3

-- Define the major axis
noncomputable def major_axis : ℝ := 2 * Real.sqrt 6

-- Define the moving line
def moving_line (k x : ℝ) : ℝ := k * (x - 2)

-- Theorem statement
theorem ellipse_constant_product :
  ∃ (E : ℝ × ℝ),
    E.2 = 0 ∧
    ∀ (k : ℝ) (A B : ℝ × ℝ),
      k ≠ 0 →
      ellipse_C A.1 A.2 →
      ellipse_C B.1 B.2 →
      A.2 = moving_line k A.1 →
      B.2 = moving_line k B.1 →
      (A.1 - E.1) * (B.1 - E.1) + A.2 * B.2 = -5/9 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l733_73309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_l733_73339

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-1, 3), prove that cos 2α = -4/5 -/
theorem cos_double_angle_special (α : Real) : 
  (Real.cos α = -1 / Real.sqrt 10) → Real.cos (2 * α) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_l733_73339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_interval_l733_73373

theorem alpha_interval (α : Real) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : Real.sin α + Real.cos α = Real.tan α) : 
  π/4 < α ∧ α < π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_interval_l733_73373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l733_73342

def sequenceQ (n : ℕ) : ℚ :=
  (2000000 : ℚ) / (5 ^ n)

theorem last_integer_in_sequence :
  ∃ k : ℕ, sequenceQ k = 128 ∧ ∀ m : ℕ, m > k → ¬(sequenceQ m).isInt :=
by
  -- We'll use 'use 6' because 2000000 / (5^6) = 128
  use 6
  constructor
  · -- Prove sequenceQ 6 = 128
    rw [sequenceQ]
    norm_num
  · -- Prove ∀ m > 6, ¬(sequenceQ m).isInt
    intro m hm
    rw [sequenceQ]
    -- The rest of the proof would go here
    sorry

#eval sequenceQ 6  -- Should output 128

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l733_73342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_roots_real_positive_l733_73331

theorem not_all_roots_real_positive (n : ℕ) (hn : n > 3) 
  (P : Polynomial ℝ) (hP : P = X^n - 5•X^(n-1) + 12•X^(n-2) - 15•X^(n-3) + 
    (Finset.range (n-3)).sum (fun i => (P.coeff i) • X^i)) : 
  ¬(∀ x : ℝ, P.eval x = 0 → x > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_roots_real_positive_l733_73331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_markings_count_l733_73385

/-- The set of markings on a stick divided into thirds and quarters -/
def StickMarkings : Finset ℚ :=
  {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}

/-- The number of markings on a stick divided into thirds and quarters -/
theorem stick_markings_count :
  Finset.card StickMarkings = 7 := by
  rfl

#eval Finset.card StickMarkings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_markings_count_l733_73385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_segments_exist_l733_73344

/-- A regular polygon with 2n vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2 * n) → ℝ × ℝ

/-- A pairing of vertices in a regular polygon -/
def VertexPairing (n : ℕ) := Fin n → Fin (2 * n) × Fin (2 * n)

/-- The length of a segment between two vertices -/
noncomputable def segmentLength (p : RegularPolygon n) (i j : Fin (2 * n)) : ℝ :=
  let (xi, yi) := p.vertices i
  let (xj, yj) := p.vertices j
  Real.sqrt ((xi - xj)^2 + (yi - yj)^2)

/-- Two segments are equal if their lengths are equal -/
def equalSegments (p : RegularPolygon n) (i j k l : Fin (2 * n)) : Prop :=
  segmentLength p i j = segmentLength p k l

theorem equal_segments_exist (n : ℕ) (h : n = 4 * m + 2 ∨ n = 4 * m + 3) 
  (p : RegularPolygon n) (pairing : VertexPairing n) :
  ∃ (i j k l : Fin (2 * n)), i ≠ j ∧ k ≠ l ∧ (i, j) ≠ (k, l) ∧
    (∃ (a b : Fin n), pairing a = (i, j) ∧ pairing b = (k, l)) ∧
    equalSegments p i j k l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_segments_exist_l733_73344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divides_power_plus_one_l733_73347

theorem square_divides_power_plus_one (n : ℕ) : 
  (n^2 ∣ 2^n + 1) ↔ (n = 1 ∨ n = 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divides_power_plus_one_l733_73347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_shirt_price_is_5_l733_73360

/-- The price of a T-shirt in dollars -/
noncomputable def T_shirt_price : ℝ := sorry

/-- The price of pants in dollars -/
def pants_price : ℝ := 4

/-- The price of a skirt in dollars -/
def skirt_price : ℝ := 6

/-- The price of a refurbished T-shirt in dollars -/
noncomputable def refurbished_T_shirt_price : ℝ := T_shirt_price / 2

/-- The total income from selling 2 T-shirts, 1 pair of pants, 4 skirts, and 6 refurbished T-shirts -/
def total_income : ℝ := 53

theorem T_shirt_price_is_5 :
  2 * T_shirt_price + pants_price + 4 * skirt_price + 6 * refurbished_T_shirt_price = total_income →
  T_shirt_price = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_shirt_price_is_5_l733_73360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_solution_l733_73398

/-- A function that determines if a quadratic equation in x and y represents a circle -/
def is_circle (a b c d e f : ℝ) : Prop :=
  a = c ∧ a ≠ 0 ∧ b = 0 ∧ d^2 + e^2 - 4*a*f > 0

/-- The main theorem stating that a = -1 is the only real value for which 
    the given equation represents a circle -/
theorem circle_equation_solution :
  ∃! a : ℝ, is_circle (a^2) (a + 2) 4 8 (5*a) (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_solution_l733_73398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_descending_multiple_of_111_l733_73333

/-- A function that checks if the digits of a natural number are in strictly descending order -/
def digits_descending (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, i < j → j < digits.length → digits[i]! > digits[j]!

/-- Theorem stating that no natural number with digits in strictly descending order is divisible by 111 -/
theorem no_descending_multiple_of_111 :
  ¬ ∃ (n : ℕ), n > 0 ∧ digits_descending n ∧ n % 111 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_descending_multiple_of_111_l733_73333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_equal_l733_73368

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def crossingTime (length : ℝ) (speed : ℝ) : ℝ :=
  length / (speed * 1000 / 3600)

/-- Theorem stating that both trains cross the electric pole in approximately 12 seconds -/
theorem trains_crossing_time_approx_equal (trainA_length trainA_speed trainB_length trainB_speed : ℝ)
  (h1 : trainA_length = 300)
  (h2 : trainA_speed = 90)
  (h3 : trainB_length = 400)
  (h4 : trainB_speed = 120) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |crossingTime trainA_length trainA_speed - 12| < ε ∧
  |crossingTime trainB_length trainB_speed - 12| < ε :=
by
  sorry

#check trains_crossing_time_approx_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_equal_l733_73368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_sqrt3x_minus_y_plus_1_l733_73348

/-- The slope angle of a line with equation √3x - y + 1 = 0 is 60 degrees. -/
theorem slope_angle_sqrt3x_minus_y_plus_1 :
  ∃ α : ℝ, α = 60 * π / 180 ∧
    (∀ x y : ℝ, Real.sqrt 3 * x - y + 1 = 0 → Real.tan α = Real.sqrt 3) :=
by
  -- We'll prove this later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_sqrt3x_minus_y_plus_1_l733_73348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_is_eight_l733_73314

/-- A 10-digit number -/
def TenDigitNumber := Nat

/-- Check if a number is even -/
def is_even (n : Nat) : Prop := ∃ k, n = 2 * k

/-- Sum of digits of a number -/
def sum_of_digits (n : Nat) : Nat := sorry

/-- Get the units digit of a number -/
def units_digit (n : Nat) : Nat := n % 10

/-- Theorem: For a 10-digit even number with sum of digits 89, the units digit is 8 -/
theorem units_digit_is_eight (n : Nat) 
  (h1 : 1000000000 ≤ n ∧ n < 10000000000)  -- 10-digit number
  (h2 : is_even n)                         -- even number
  (h3 : sum_of_digits n = 89)              -- sum of digits is 89
  : units_digit n = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_is_eight_l733_73314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l733_73364

-- Define the force function
def F (x : ℝ) : ℝ := x^2 + 1

-- Define the work function
noncomputable def work (a b : ℝ) : ℝ := ∫ x in a..b, F x

-- Theorem statement
theorem work_calculation :
  work 0 6 = 78 := by
  -- Unfold the definition of work
  unfold work
  -- Evaluate the integral
  simp [F]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l733_73364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maplewood_population_l733_73310

theorem maplewood_population (num_cities : ℕ) (avg_lower avg_upper : ℝ) :
  num_cities = 25 →
  5200 ≤ avg_lower →
  avg_upper ≤ 5700 →
  avg_lower ≤ avg_upper →
  ∃ (total_pop : ℕ), total_pop = 136250 ∧ 
    (avg_lower * (num_cities : ℝ) ≤ (total_pop : ℝ) ∧ 
     (total_pop : ℝ) ≤ avg_upper * (num_cities : ℝ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maplewood_population_l733_73310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_n_eq_9_l733_73356

/-- The number of integer values of n for which 3200 * (2/5)^n is an integer -/
def count_valid_n : ℕ :=
  (Finset.range 9).card

/-- Predicate to check if a given n results in an integer -/
def is_valid (n : ℤ) : Prop :=
  ∃ (k : ℤ), 3200 * (2 : ℚ) ^ n * (5 : ℚ) ^ (-n) = k

theorem count_valid_n_eq_9 :
  count_valid_n = 9 ∧ 
  ∀ (n : ℤ), is_valid n ↔ n ∈ Finset.Ico (-2 : ℤ) 7 := by
  sorry

#check count_valid_n_eq_9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_n_eq_9_l733_73356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contingency_fund_amount_l733_73352

noncomputable def total_donation : ℝ := 240

noncomputable def community_pantry_ratio : ℝ := 1/3
noncomputable def local_crisis_ratio : ℝ := 1/2
noncomputable def livelihood_ratio : ℝ := 1/4

noncomputable def community_pantry : ℝ := total_donation * community_pantry_ratio
noncomputable def local_crisis : ℝ := total_donation * local_crisis_ratio

noncomputable def remaining_after_first_two : ℝ := total_donation - (community_pantry + local_crisis)

noncomputable def livelihood : ℝ := remaining_after_first_two * livelihood_ratio

noncomputable def contingency : ℝ := remaining_after_first_two - livelihood

theorem contingency_fund_amount : contingency = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contingency_fund_amount_l733_73352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l733_73320

noncomputable def f (x : ℝ) := Real.cos (2 * x) + Real.sqrt 3 * Real.sin (2 * x)

theorem f_max_value :
  ∃ (c : ℝ), c ∈ Set.Icc (-π/3) (π/12) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-π/3) (π/12) → f x ≤ f c ∧
  f c = Real.sqrt 3 := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l733_73320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cupcakes_count_l733_73323

def initial_cupcakes : ℕ := 14
def sold_fraction : ℚ := 3/4
def increase_percentage : ℚ := 1/4

def cupcakes_left (initial : ℕ) (sold_frac : ℚ) : ℕ :=
  initial - (Int.floor (↑initial * sold_frac)).toNat

def new_cupcakes (initial : ℕ) (increase : ℚ) : ℕ :=
  (Int.ceil (↑initial * increase)).toNat

theorem final_cupcakes_count :
  cupcakes_left initial_cupcakes sold_fraction + new_cupcakes initial_cupcakes increase_percentage = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cupcakes_count_l733_73323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_nine_last_positive_term_l733_73379

-- Define the arithmetic sequence
noncomputable def arithmeticSequence (n : ℕ) : ℝ := 25 - 3 * ((n : ℝ) - 1)

-- Define the sum of the first n terms
noncomputable def S (n : ℕ) : ℝ := (n : ℝ) * (arithmeticSequence 1 + arithmeticSequence n) / 2

-- Theorem statement
theorem max_sum_at_nine :
  ∀ k : ℕ, k ≠ 9 → S k ≤ S 9 := by
  sorry

-- Additional lemma to show that the 9th term is the last positive term
theorem last_positive_term :
  arithmeticSequence 9 > 0 ∧ arithmeticSequence 10 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_nine_last_positive_term_l733_73379
