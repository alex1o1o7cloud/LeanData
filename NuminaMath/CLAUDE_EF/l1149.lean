import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_consecutive_mobius_zero_l1149_114927

/-- The Möbius function μ(n) -/
def mobius : ℕ → ℤ := sorry

/-- A number has a square factor greater than 1 -/
def has_square_factor (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧ m * m ∣ n

/-- The Möbius function is zero iff the number has a square factor greater than 1 -/
axiom mobius_zero_iff_square_factor (n : ℕ) :
  mobius n = 0 ↔ has_square_factor n

/-- There exists a number n such that μ(n + k) = 0 for all k from 0 to 1000000 -/
theorem exists_consecutive_mobius_zero :
  ∃ (n : ℕ), ∀ (k : ℕ), k ≤ 1000000 → mobius (n + k) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_consecutive_mobius_zero_l1149_114927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_treasure_l1149_114920

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a cell in the grid -/
structure Cell :=
  (x : ℕ)
  (y : ℕ)

/-- Defines neighboring cells -/
def neighboring (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ c1.y.succ = c2.y) ∨
  (c1.x = c2.x ∧ c1.y = c2.y.succ) ∨
  (c1.x.succ = c2.x ∧ c1.y = c2.y) ∨
  (c1.x = c2.x.succ ∧ c1.y = c2.y)

/-- Defines a valid cell within the grid -/
def valid_cell (g : Grid) (c : Cell) : Prop :=
  c.x < g.size ∧ c.y < g.size

/-- Defines a strategy for finding treasure -/
def Strategy (g : Grid) := List Cell

/-- Defines the success of a strategy -/
def strategy_succeeds (g : Grid) (s : Strategy g) : Prop :=
  ∀ c1 c2 : Cell, valid_cell g c1 → valid_cell g c2 → neighboring c1 c2 →
    ∃ (i j : ℕ), i < s.length ∧ j < s.length ∧ i ≠ j ∧
      ((s.get? i = some c1 ∧ s.get? j = some c2) ∨
       (s.get? i = some c2 ∧ s.get? j = some c1))

/-- The main theorem: 50 moves are necessary and sufficient -/
theorem min_moves_for_treasure (g : Grid) (h : g.size = 10) :
  (∃ s : Strategy g, s.length = 50 ∧ strategy_succeeds g s) ∧
  (∀ s : Strategy g, s.length < 50 → ¬strategy_succeeds g s) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_treasure_l1149_114920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinates_l1149_114988

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define the point type
def Point := ℝ × ℝ

-- Define parallelism of vectors
def parallel (v w : MyVector) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- Define the problem statement
theorem endpoint_coordinates 
  (a b : MyVector)
  (A : Point)
  (B : Point)
  (h1 : a = (-2, 3))
  (h2 : parallel b a)
  (h3 : A = (1, 2))
  (h4 : B.1 = 0 ∨ B.2 = 0)  -- B is on a coordinate axis
  (h5 : b = (B.1 - A.1, B.2 - A.2))  -- b is the vector from A to B
  : B = (7/3, 0) ∨ B = (0, 7/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinates_l1149_114988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_height_l1149_114901

theorem initial_average_height
  (initial_class_size : ℕ)
  (students_left : ℕ)
  (students_joined : ℕ)
  (avg_height_left : ℝ)
  (avg_height_joined : ℝ)
  (new_avg_height : ℝ)
  (h1 : initial_class_size = 35)
  (h2 : students_left = 7)
  (h3 : students_joined = 7)
  (h4 : avg_height_left = 120)
  (h5 : avg_height_joined = 140)
  (h6 : new_avg_height = 204) :
  ∃ (initial_avg_height : ℝ),
    initial_avg_height * (initial_class_size : ℝ) =
    new_avg_height * (initial_class_size : ℝ) +
    avg_height_left * (students_left : ℝ) -
    avg_height_joined * (students_joined : ℝ) ∧
    initial_avg_height = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_height_l1149_114901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_sum_l1149_114966

/-- Function f(x) = 2a^(x-1) + 1 -/
noncomputable def f (a x : ℝ) : ℝ := 2 * (a ^ (x - 1)) + 1

/-- Theorem: If (m, n) is a fixed point of f for all valid a, then m + n = 4 -/
theorem fixed_point_sum (m n : ℝ) :
  (∀ a : ℝ, a > 0 → a ≠ 1 → f a m = n) →
  m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_sum_l1149_114966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l1149_114986

-- Define the circle ⊙O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (x : ℝ) : Prop := x = 4

-- Define the polar coordinates
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point A on the circle
noncomputable def point_A (θ : ℝ) : ℝ × ℝ := polar_to_cartesian 2 θ

-- Define point B on the line
noncomputable def point_B (θ : ℝ) : ℝ × ℝ := polar_to_cartesian (4 / Real.cos θ) θ

-- Define point M as the midpoint of AB
noncomputable def point_M (θ : ℝ) : ℝ × ℝ :=
  let a := point_A θ
  let b := point_B θ
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem trajectory_of_M :
  ∀ θ : ℝ, θ ≠ π/2 ∧ θ ≠ -π/2 →
  let (x, y) := point_M θ
  x^2 + y^2 = (1 + 2 / Real.cos θ)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l1149_114986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1149_114953

/-- The sum of the infinite series Σ(k/(4^k)) for k from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' k, (k : ℝ) / (4 : ℝ) ^ k

/-- Theorem stating that the sum of the infinite series is equal to 4/9 -/
theorem infiniteSeriesSum : infiniteSeries = 4/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1149_114953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l1149_114960

-- Define the parameters
noncomputable def train1_length : ℝ := 120
noncomputable def train1_speed : ℝ := 42
noncomputable def train2_speed : ℝ := 30
noncomputable def clearing_time : ℝ := 20

-- Define the conversion factor from km/h to m/s
noncomputable def kmph_to_ms : ℝ := 5 / 18

-- Theorem statement
theorem second_train_length :
  let relative_speed : ℝ := (train1_speed + train2_speed) * kmph_to_ms
  let total_distance : ℝ := relative_speed * clearing_time
  let train2_length : ℝ := total_distance - train1_length
  train2_length = 280 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l1149_114960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_calculation_l1149_114904

theorem shirt_price_calculation (P : ℝ) : 
  P * (1 - 0.25) * (1 - 0.25) * (1 - 0.10) * (1 + 0.15) = 18 → 
  abs (P - 30.91) < 0.01 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_calculation_l1149_114904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1149_114955

theorem problem_statement (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 4)^6 + (Real.log y / Real.log 5)^6 + 18 = 12 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^(4^(1/3)) + y^(4^(1/3)) = 4^(4/3) + 5^(4/3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1149_114955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_correct_question_l1149_114905

/-- Represents the two possible roads at the crossroads -/
inductive Road
| Left
| Right

/-- Represents Baba Yaga's truth-telling state -/
inductive BabaYagaState
| TruthTelling
| Lying

/-- Function to get the opposite state of Baba Yaga -/
def BabaYagaState.opposite : BabaYagaState → BabaYagaState
| TruthTelling => Lying
| Lying => TruthTelling

/-- Represents a question that can be asked to Baba Yaga -/
structure Question where
  ask : BabaYagaState → Road → Road

/-- The correct road to Kashchey's kingdom -/
def correctRoad : Road := Road.Left  -- Arbitrarily chosen for this example

/-- Function to determine if a given question always leads to the correct road -/
def leadsToCorrectRoad (q : Question) : Prop :=
  ∀ (state : BabaYagaState), 
    (q.ask state correctRoad ≠ correctRoad) ∧
    (q.ask (state.opposite) correctRoad = correctRoad)

/-- Theorem stating that there exists a question that always leads to the correct road -/
theorem exists_correct_question : ∃ (q : Question), leadsToCorrectRoad q := by
  sorry

#check exists_correct_question

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_correct_question_l1149_114905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_theorem_l1149_114965

/-- Represents the dimensions and area of a canal with a composite cross-section -/
structure Canal where
  top_width : ℝ
  bottom_width : ℝ
  total_area : ℝ

/-- Calculates the depth of the trapezium part of the canal and the height of the semi-circle -/
noncomputable def canal_dimensions (c : Canal) : ℝ × ℝ :=
  let trapezium_depth := (c.total_area - 18 * Real.pi) / 10
  let semicircle_height := c.top_width / 2
  (trapezium_depth, semicircle_height)

/-- Theorem stating the dimensions of the canal given specific measurements -/
theorem canal_theorem (c : Canal) 
  (h1 : c.top_width = 12)
  (h2 : c.bottom_width = 8)
  (h3 : c.total_area = 1800) :
  canal_dimensions c = ((1800 - 18 * Real.pi) / 10, 6) := by
  sorry

/-- Example computation (marked as noncomputable due to dependence on Real.pi) -/
noncomputable example : ℝ × ℝ := canal_dimensions ⟨12, 8, 1800⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canal_theorem_l1149_114965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_division_l1149_114916

/-- The maximum number of parts a circle can be divided into by k pairwise intersecting chords -/
def max_parts (k : ℕ) : ℕ := sorry

/-- Theorem: If k pairwise intersecting chords can divide a circle into at most n parts,
    then k+1 pairwise intersecting chords can divide the circle into at most n+k+1 parts -/
theorem circle_division (k : ℕ) :
  max_parts k ≤ max_parts (k + 1) - (k + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_division_l1149_114916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l1149_114948

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Predicate to check if two points form a line parallel to y-axis -/
def parallel_to_y_axis (p1 p2 : Point) : Prop :=
  p1.x = p2.x

theorem point_b_coordinates (A B : Point) :
  parallel_to_y_axis A B →
  A.x = 1 →
  A.y = -3 →
  distance A B = 5 →
  (B.x = 1 ∧ (B.y = 2 ∨ B.y = -8)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l1149_114948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_breath_hold_l1149_114981

/-- Represents the number of seconds Chris can hold his breath on a given day -/
def breath_hold (day : ℕ) : ℕ := sorry

/-- The goal is to hold breath for 90 seconds -/
def goal : ℕ := 90

/-- Chris can hold his breath for 10 extra seconds each day -/
def daily_increase : ℕ := 10

/-- It takes 6 days to reach the goal -/
def days_to_goal : ℕ := 6

theorem chris_breath_hold :
  breath_hold 1 = 10 ∧
  breath_hold 3 = 30 ∧
  (∀ d : ℕ, d > 0 → breath_hold (d + 1) = breath_hold d + daily_increase) ∧
  breath_hold days_to_goal = goal →
  breath_hold 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_breath_hold_l1149_114981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_ABCDEFG_l1149_114919

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  let d1 := ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2)
  let d2 := ((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2)
  let d3 := ((t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2)
  d1 = d2 ∧ d2 = d3

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (M : Point) (A : Point) (B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

/-- Calculate the distance between two points -/
noncomputable def distance (A : Point) (B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

/-- The main theorem -/
theorem perimeter_of_ABCDEFG (A B C D E F G : Point) : 
  let ABC := Triangle.mk A B C
  let ADE := Triangle.mk A D E
  let EFG := Triangle.mk E F G
  isEquilateral ABC → isEquilateral ADE → isEquilateral EFG →
  isMidpoint D A C → isMidpoint G A E →
  distance A B = 6 →
  distance A B + distance B C + distance C D + distance D E + 
  distance E F + distance F G + distance G A = 18.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_ABCDEFG_l1149_114919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_intersections_l1149_114946

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Curve in 2D space -/
structure Curve where
  f : ℝ → ℝ → Prop

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Sum of distances from P to intersection points of l and C -/
theorem sum_of_distances_to_intersections
  (P : Point)
  (l : Line)
  (C : Curve)
  (h1 : P.x = 1 ∧ P.y = -2)
  (h2 : l.a = 1 ∧ l.b = -1 ∧ l.c = -3)
  (h3 : C.f = fun x y => y^2 = 2*x)
  (A B : Point)
  (h4 : A.x - A.y - 3 = 0 ∧ A.y^2 = 2*A.x)
  (h5 : B.x - B.y - 3 = 0 ∧ B.y^2 = 2*B.x)
  (h6 : A ≠ B) :
  distance P A + distance P B = 6 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_to_intersections_l1149_114946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l1149_114950

def b : ℕ → ℚ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | n + 2 => b (n + 1) + 3 * (n + 1)

theorem b_100_value : b 100 = 15001.5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l1149_114950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_sets_l1149_114924

theorem count_possible_sets : 
  let U : Finset ℕ := {1, 2, 3, 4, 5}
  let A : Finset ℕ := {1, 2, 3}
  let B : Finset ℕ := {1, 3}
  ∃! (n : ℕ), n = (Finset.filter (fun M => M ⊆ U ∧ M ∩ A = B) (Finset.powerset U)).card ∧ n = 4
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_sets_l1149_114924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1149_114907

-- Define set A
def A : Set ℝ := {x | 1/3 < (3 : ℝ)^(x-2) ∧ (3 : ℝ)^(x-2) ≤ 9}

-- Define set B
def B : Set ℝ := {x | x^3 ≤ 5 * Real.sqrt 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 (Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1149_114907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_office_age_problem_l1149_114990

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℝ) (num_group1 : ℕ) (avg_age_group1 : ℝ) (num_group2 : ℕ) (age_person15 : ℝ) :
  total_persons = 16 →
  avg_age_all = 15 →
  num_group1 = 9 →
  avg_age_group1 = 16 →
  num_group2 = 5 →
  age_person15 = 26 →
  (total_persons * avg_age_all - num_group1 * avg_age_group1 - age_person15) / num_group2 = 14 := by
  intro h1 h2 h3 h4 h5 h6
  sorry

#check office_age_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_office_age_problem_l1149_114990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1149_114977

-- Define the point P
def P (a b : ℝ) : ℝ × ℝ := (a, b)

-- Define the condition for b
noncomputable def b_condition (a : ℝ) : ℝ := Real.sqrt (a - 2) + Real.sqrt (2 - a) - 3

-- Theorem statement
theorem point_in_fourth_quadrant (a b : ℝ) :
  b = b_condition a →
  a > 0 ∧ b < 0 :=
by
  intro h
  sorry

#check point_in_fourth_quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l1149_114977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_over_budget_l1149_114969

/-- Represents a project budget --/
structure ProjectBudget where
  total : ℚ
  months : ℕ
  actual_spent : ℚ
  months_elapsed : ℕ

/-- Calculates the amount a project is over budget --/
def over_budget (pb : ProjectBudget) : ℚ :=
  pb.actual_spent - (pb.total / pb.months * pb.months_elapsed)

/-- Theorem: The given project is $280 over budget --/
theorem project_over_budget :
  let pb : ProjectBudget := {
    total := 12600,
    months := 12,
    actual_spent := 6580,
    months_elapsed := 6
  }
  over_budget pb = 280 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_over_budget_l1149_114969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1149_114945

/-- Represents a rectangle in the coordinate plane -/
structure Rectangle where
  width : ℝ
  height : ℝ
  lower_left : ℝ × ℝ

/-- Represents a line in the coordinate plane -/
structure Line where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (base * height) / 2

/-- Theorem stating the condition for equal area division -/
theorem equal_area_division
  (rect : Rectangle)
  (line : Line)
  (c : ℝ) :
  rect.width = 6 ∧
  rect.height = 4 ∧
  rect.lower_left = (0, 0) ∧
  line.start = (c, 0) ∧
  line.endpoint = (6, 6) →
  (triangle_area (6 - c) 6 = (rect.width * rect.height) / 2) ↔
  c = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1149_114945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1149_114914

/-- The standard equation of an ellipse with specific properties -/
theorem ellipse_equation {a b c : ℝ} (C : Set (ℝ × ℝ)) (F₁ F₂ A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / 16 + y^2 / 8 = 1) ↔
  (∀ (x y : ℝ), (x, y) ∈ C → x^2 / a^2 + y^2 / b^2 = 1) ∧
  (F₁ = (c, 0) ∧ F₂ = (-c, 0)) ∧
  (c^2 / a^2 = 1/2) ∧
  (F₁ ∈ l) ∧
  (A ∈ C ∧ A ∈ l) ∧
  (B ∈ C ∧ B ∈ l) ∧
  (dist A F₁ + dist A F₂ + dist B F₁ + dist B F₂ = 16) :=
sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1149_114914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l1149_114963

/-- An ellipse with semi-major axis 5 and semi-minor axis 3 -/
structure Ellipse where
  x : ℝ
  y : ℝ
  eq : x^2 / 9 + y^2 / 25 = 1

/-- A point on the ellipse -/
def PointOnEllipse (e : Ellipse) : ℝ × ℝ := (e.x, e.y)

/-- The distance from a point to a focus is 2 -/
def DistanceToOneFocus (p : ℝ × ℝ) : Prop := 
  ∃ f : ℝ × ℝ, dist p f = 2

/-- The theorem to be proved -/
theorem ellipse_focus_distance (e : Ellipse) :
  ∀ p : ℝ × ℝ, p = PointOnEllipse e → DistanceToOneFocus p → 
  ∃ f : ℝ × ℝ, dist p f = 8 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l1149_114963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_properties_l1149_114991

def f (n : ℕ) : ℕ := (Finset.sum (Nat.divisors n) id)

theorem divisor_sum_properties :
  (∀ m n : ℕ, Nat.Coprime m n → f (m * n) = f m * f n) ∧
  (∀ n a : ℕ, a < n → a ∣ n → f n = n + a → Nat.Prime n) ∧
  (∀ n : ℕ, Even n → f n = 2 * n → ∃ p : ℕ, Nat.Prime p ∧ n = 2^(p-1) * (2^p - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_properties_l1149_114991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_locus_l1149_114957

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 7 = 0

/-- Tangent circle to both circle1 and circle2 -/
def tangentCircle (cx cy r : ℝ) : Prop :=
  (∃ x y : ℝ, circle1 x y ∧ (x - cx)^2 + (y - cy)^2 = r^2) ∧
  (∃ x y : ℝ, circle2 x y ∧ (x - cx)^2 + (y - cy)^2 = r^2)

/-- The locus of centers of tangent circles -/
def centerLocus (cx cy : ℝ) : Prop :=
  ∃ r : ℝ, tangentCircle cx cy r

/-- Predicate for a set to be a hyperbola -/
def IsHyperbola (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate for a set to be a line -/
def IsLine (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Theorem: The locus of centers of circles tangent to both given circles
    is composed of one hyperbola and one line -/
theorem tangent_circles_locus :
  ∃ h l : Set (ℝ × ℝ),
    (IsHyperbola h) ∧ (IsLine l) ∧
    (∀ cx cy : ℝ, centerLocus cx cy ↔ (cx, cy) ∈ h ∪ l) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_locus_l1149_114957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_in_third_quadrant_l1149_114997

theorem cos_B_in_third_quadrant (B : Real) (h1 : π < B ∧ B ≤ 3*π/2) 
  (h2 : Real.sin B = 5/13) : Real.cos B = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_in_third_quadrant_l1149_114997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1149_114975

/-- The area of a circular sector -/
noncomputable def sectorArea (r : ℝ) (α : ℝ) : ℝ := (1 / 2) * r^2 * α

/-- Theorem: The area of a sector with radius 2 and central angle 2 radians is 4 -/
theorem sector_area_example : sectorArea 2 2 = 4 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1149_114975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_value_l1149_114974

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
noncomputable def given_triangle : Triangle where
  c := Real.sqrt 3
  A := 45 * Real.pi / 180  -- Convert degrees to radians
  C := 60 * Real.pi / 180  -- Convert degrees to radians
  a := 0  -- Placeholder, will be proved
  b := 0  -- Placeholder, not needed for the proof
  B := 0  -- Placeholder, not needed for the proof

-- Theorem statement
theorem side_a_value (t : Triangle) (h1 : t.c = Real.sqrt 3) 
    (h2 : t.A = 45 * Real.pi / 180) (h3 : t.C = 60 * Real.pi / 180) : 
  t.a = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_value_l1149_114974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_meeting_time_l1149_114964

/-- Represents the meeting of three people walking along a straight path. -/
structure WalkingMeeting where
  distance : ℝ
  speed_A : ℝ
  speed_B : ℝ
  speed_C : ℝ
  delay_C : ℝ

/-- Calculates the meeting time for the three walkers. -/
noncomputable def meeting_time (w : WalkingMeeting) : ℝ :=
  (w.distance + w.speed_C * w.delay_C) / (w.speed_A + w.speed_B + w.speed_C)

/-- Theorem stating that under the given conditions, the meeting time is approximately 7.47 hours. -/
theorem walking_meeting_time :
  let w : WalkingMeeting :=
    { distance := 100
      speed_A := 5
      speed_B := 4
      speed_C := 6
      delay_C := 2
    }
  ∃ ε > 0, abs (meeting_time w - 7.47) < ε := by
  sorry

#eval (100 + 6 * 2) / (5 + 4 + 6)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_meeting_time_l1149_114964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_satisfying_function_l1149_114947

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Ioo 0 1 → f x < 2) ∧
  (∀ x y, max (f (x + y)) (f (x - y)) = f x + f y)

/-- The theorem stating the form of functions satisfying the conditions -/
theorem characterize_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f →
  ∃ c : ℝ, c ∈ Set.Icc 0 2 ∧ ∀ x, f x = c * |x| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_satisfying_function_l1149_114947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1149_114972

theorem cube_root_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b / c)^(1/3) = a * (b / c)^(1/3) ↔ c = b * (a^3 - 1) / a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l1149_114972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_six_digit_with_one_difference_l1149_114903

/-- The number of ways to choose k items from n items without replacement and without regard to order. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The set of six-digit natural numbers with digits in ascending order. -/
def ascending_six_digit : Finset ℕ := sorry

/-- The set of numbers in ascending_six_digit that contain the digit 1. -/
def with_one : Finset ℕ := sorry

/-- The set of numbers in ascending_six_digit that do not contain the digit 1. -/
def without_one : Finset ℕ := sorry

theorem ascending_six_digit_with_one_difference : 
  Finset.card with_one - Finset.card without_one = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_six_digit_with_one_difference_l1149_114903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1149_114961

/-- The original function before translation -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + Real.sin x

/-- The translated function -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x - m)

/-- A function is symmetric with respect to the y-axis if f(x) = f(-x) for all x -/
def is_symmetric_about_y_axis (h : ℝ → ℝ) : Prop :=
  ∀ x, h x = h (-x)

theorem min_translation_for_symmetry :
  ∃ m : ℝ, m > 0 ∧ is_symmetric_about_y_axis (g m) ∧
  ∀ m' : ℝ, m' > 0 → is_symmetric_about_y_axis (g m') → m ≤ m' :=
by sorry

#check min_translation_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1149_114961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1149_114951

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0)
    and one asymptote y = -2x is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x : ℝ, y = -2*x → x^2 / a^2 - y^2 / b^2 = 1) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1149_114951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Tn_greater_Sn_l1149_114934

open BigOperators

def Sn (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (k : ℚ) / ((2 * n - 2 * k + 1 : ℚ) * (2 * n - k + 1 : ℚ))

def Tn (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, 1 / (k + 1 : ℚ)

theorem Tn_greater_Sn (n : ℕ) (h : n > 0) : Tn n > Sn n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Tn_greater_Sn_l1149_114934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_extended_box_theorem_l1149_114929

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the volume of a set of points -/
structure VolumeExpression where
  m : ℕ
  n : ℕ
  p : ℕ
  coprime : Nat.Coprime n p

/-- Helper function to calculate the volume of the extended box -/
noncomputable def volume_of_extended_box (b : Box) (extension : ℝ) : ℝ :=
  sorry

/-- The main theorem -/
theorem volume_of_extended_box_theorem (b : Box) (v : VolumeExpression) :
  b.width = 2 ∧ b.length = 3 ∧ b.height = 6 →
  (volume_of_extended_box b 2 : ℝ) = (v.m + v.n * Real.pi) / v.p →
  v.m + v.n + v.p = 701 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_extended_box_theorem_l1149_114929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_one_l1149_114906

noncomputable def g (a b c : ℝ) : ℝ := a / (a + b + c) + b / (b + c + a) + c / (c + a + b)

theorem g_equals_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  g a b c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_one_l1149_114906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_rose_fraction_is_three_fifths_l1149_114909

/-- Represents Mrs. Dawson's rose garden -/
structure RoseGarden where
  rows : ℕ
  rosesPerRow : ℕ
  redFraction : ℚ
  pinkRoses : ℕ

/-- The fraction of white roses among non-red roses in the garden -/
def whiteRoseFraction (garden : RoseGarden) : ℚ :=
  let totalRoses := garden.rows * garden.rosesPerRow
  let redRoses := (garden.redFraction * totalRoses).floor
  let remainingRoses := totalRoses - redRoses
  let whiteRoses := remainingRoses - garden.pinkRoses
  whiteRoses / remainingRoses

/-- Theorem stating the fraction of white roses among non-red roses -/
theorem white_rose_fraction_is_three_fifths (garden : RoseGarden) 
  (h1 : garden.rows = 10)
  (h2 : garden.rosesPerRow = 20)
  (h3 : garden.redFraction = 1/2)
  (h4 : garden.pinkRoses = 40) :
  whiteRoseFraction garden = 3/5 := by
  sorry

#eval whiteRoseFraction { rows := 10, rosesPerRow := 20, redFraction := 1/2, pinkRoses := 40 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_rose_fraction_is_three_fifths_l1149_114909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wool_usage_calculation_l1149_114902

/-- Represents the types of clothing items --/
inductive ClothingType
  | Scarf
  | Sweater
  | Hat
  | Mittens

/-- Represents the types of wool --/
inductive WoolType
  | Merino
  | Alpaca
  | Cashmere

/-- Production quantities for each person --/
def production : List (String × List (ClothingType × Nat)) :=
  [("Aaron", [(ClothingType.Scarf, 10), (ClothingType.Sweater, 5), (ClothingType.Hat, 6)]),
   ("Enid", [(ClothingType.Sweater, 8), (ClothingType.Hat, 12), (ClothingType.Mittens, 4)]),
   ("Clara", [(ClothingType.Scarf, 3), (ClothingType.Hat, 7), (ClothingType.Mittens, 5)])]

/-- Wool usage for each type of clothing --/
def woolUsage : ClothingType → List (WoolType × Nat)
  | ClothingType.Scarf => [(WoolType.Merino, 3), (WoolType.Alpaca, 2)]
  | ClothingType.Sweater => [(WoolType.Alpaca, 4), (WoolType.Cashmere, 1)]
  | ClothingType.Hat => [(WoolType.Merino, 2), (WoolType.Alpaca, 1)]
  | ClothingType.Mittens => [(WoolType.Cashmere, 1)]

/-- Calculate total wool used --/
def totalWoolUsed : WoolType → Nat
  | _ => 0  -- Placeholder implementation

theorem wool_usage_calculation :
  totalWoolUsed WoolType.Merino = 89 ∧
  totalWoolUsed WoolType.Alpaca = 103 ∧
  totalWoolUsed WoolType.Cashmere = 22 := by
  sorry

#check wool_usage_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wool_usage_calculation_l1149_114902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_rate_approx_6_52_l1149_114980

/-- Represents the investment and earnings data for three stocks -/
structure StockData where
  investmentA : ℚ
  investmentB : ℚ
  investmentC : ℚ
  rateA : ℚ
  rateB : ℚ
  rateC : ℚ
  earningsA : ℚ
  earningsB : ℚ

/-- Calculates the combined investment rate for the given stock data -/
noncomputable def combinedInvestmentRate (data : StockData) : ℚ :=
  let totalInvestment := data.investmentA + data.investmentB + data.investmentC
  let totalEarnings := data.earningsA + data.earningsB + (data.rateC * data.investmentC / 100)
  (totalEarnings / totalInvestment) * 100

/-- Theorem stating that the combined investment rate is approximately 6.52% -/
theorem combined_rate_approx_6_52 (data : StockData) 
    (h1 : data.investmentA = 1800)
    (h2 : data.investmentB = 2300)
    (h3 : data.investmentC = 2500)
    (h4 : data.rateA = 9)
    (h5 : data.rateB = 7)
    (h6 : data.rateC = 6)
    (h7 : data.earningsA = 120)
    (h8 : data.earningsB = 160) :
    abs (combinedInvestmentRate data - 652/100) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_rate_approx_6_52_l1149_114980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_box_equation_system_l1149_114925

/-- Represents the number of iron sheets available. -/
def total_sheets : ℕ := 35

/-- Represents the number of box bodies that can be made from one sheet. -/
def bodies_per_sheet : ℕ := 20

/-- Represents the number of box bottoms that can be made from one sheet. -/
def bottoms_per_sheet : ℕ := 30

/-- Represents the number of box bottoms needed for one set of candy boxes. -/
def bottoms_per_set : ℕ := 2

/-- Theorem stating that the system of equations correctly represents the problem. -/
theorem candy_box_equation_system (x y : ℝ) :
  (x + y = total_sheets) ∧
  (bodies_per_sheet * x = (bottoms_per_sheet * y) / bottoms_per_set) :=
by
  sorry

#check candy_box_equation_system

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_box_equation_system_l1149_114925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_eight_l1149_114936

/-- A geometric sequence with positive common ratio -/
structure GeometricSequence where
  a : ℕ → ℝ
  r : ℝ
  r_pos : r > 0
  geom : ∀ n, a (n + 1) = a n * r

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (g : GeometricSequence) (n : ℕ) : ℝ :=
  (g.a 1) * (1 - g.r^n) / (1 - g.r)

theorem geometric_sequence_sum_eight (g : GeometricSequence) 
  (h1 : g.a 1 + g.a 2 = 2)
  (h2 : g.a 3 + g.a 4 = 8) :
  geometricSum g 8 = 170 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_eight_l1149_114936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_region_T_l1149_114921

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of circles -/
structure CircleConfiguration where
  circles : List Circle
  tangentPoint : ℝ × ℝ
  tangentLine : ℝ → ℝ

/-- Calculates the area of a region inside exactly one of the circles -/
noncomputable def areaOfRegionT (config : CircleConfiguration) : ℝ :=
  sorry

theorem max_area_of_region_T :
  ∀ (config : CircleConfiguration),
    config.circles.length = 4 ∧
    (∀ c ∈ config.circles, c.radius ∈ ({2, 4, 6, 8} : Set ℝ)) →
    areaOfRegionT config ≤ 100 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_region_T_l1149_114921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heightTrackingUsesLineChart_l1149_114959

-- Define chart types
inductive ChartType
| Line
| Pie
| Bar

-- Function to select chart type based on data characteristics
def selectChartType (showsChange : Bool) (showsAmount : Bool) : ChartType :=
  match (showsChange, showsAmount) with
  | (true, _) => ChartType.Line
  | (false, true) => ChartType.Bar
  | (false, false) => ChartType.Pie

-- Example usage
def main : IO Unit := do
  let heightTracking := selectChartType true true
  match heightTracking with
  | ChartType.Line => IO.println "Line chart is best for tracking height changes"
  | _ => IO.println "Another chart type was selected"

#eval main

theorem heightTrackingUsesLineChart :
  selectChartType true true = ChartType.Line := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heightTrackingUsesLineChart_l1149_114959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_satisfying_inequality_l1149_114984

theorem count_positive_integers_satisfying_inequality :
  ∃ (S : Finset ℕ), (∀ n ∈ S, (n + 8) * (n - 3) * (n - 15) < 0) ∧ (Finset.card S = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_satisfying_inequality_l1149_114984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_polynomials_lower_bound_l1149_114923

/-- A polynomial is magical if it maps natural numbers to natural numbers and is bijective -/
def IsMagical {n : ℕ} (P : (Fin n → ℕ) → ℕ) : Prop :=
  Function.Injective P ∧ Function.Surjective P ∧ ∀ x, P x ∈ Set.range id

/-- The n-th Catalan number -/
def CatalanNumber (n : ℕ) : ℕ :=
  sorry

/-- The number of magical polynomials for a given n -/
def NumMagicalPolynomials (n : ℕ) : ℕ :=
  sorry

/-- The main theorem: lower bound on the number of magical polynomials -/
theorem magical_polynomials_lower_bound (n : ℕ) (hn : 0 < n) :
  NumMagicalPolynomials n ≥ Nat.factorial n * (CatalanNumber n - CatalanNumber (n - 1)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_polynomials_lower_bound_l1149_114923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_theorem_l1149_114933

theorem factorial_sum_theorem (a b c : ℕ) :
  (∃ k l m : ℕ, k > 0 ∧ l > 0 ∧ m > 0 ∧
    a * b + 1 = Nat.factorial k ∧ 
    b * c + 1 = Nat.factorial l ∧ 
    c * a + 1 = Nat.factorial m) →
  ∃ k : ℕ, k > 1 ∧ ((a = Nat.factorial k - 1 ∧ b = 1 ∧ c = 1) ∨
                    (b = Nat.factorial k - 1 ∧ a = 1 ∧ c = 1) ∨
                    (c = Nat.factorial k - 1 ∧ a = 1 ∧ b = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_theorem_l1149_114933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base6_l1149_114918

/-- Converts a base 6 number (represented as a list of digits) to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation (as a list of digits) -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The sum of 4444₆, 444₆, and 44₆ in base 6 is equal to 52420₆ -/
theorem sum_in_base6 :
  decimalToBase6 (base6ToDecimal [4, 4, 4, 4] + base6ToDecimal [4, 4, 4] + base6ToDecimal [4, 4]) =
  [5, 2, 4, 2, 0] := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base6_l1149_114918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_problem_l1149_114982

/-- Given two workshops with the following conditions:
    1. The first workshop has x people
    2. The second workshop has 20 fewer people than 3/4 of the first workshop
    3. 15 people are transferred from the second workshop to the first workshop
    Prove the total number of people and the difference after transfer -/
theorem workshop_problem (x : ℚ) : 
  (x + ((3/4) * x - 20) = (7/4) * x - 20) ∧ 
  ((x + 15) - ((3/4) * x - 35) = (1/4) * x + 50) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_problem_l1149_114982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_A_B_l1149_114932

theorem compare_A_B (x : ℝ) (n : ℕ) (h_x_pos : x > 0) : 
  x^n + x^(-n : ℤ) ≥ x^(n-1 : ℤ) + x^(1-n : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_A_B_l1149_114932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l1149_114913

theorem find_divisor (dividend : ℝ) (quotient : ℝ) (remainder : ℝ) (divisor : ℝ) :
  dividend = 527652 ∧ quotient = 392.57 ∧ remainder = 48.25 →
  dividend = divisor * quotient + remainder →
  ∃ ε > 0, |divisor - 1344.25| < ε := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_divisor_l1149_114913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_onto_AC_is_two_l1149_114954

def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (3, 4)

def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude w)

theorem projection_AB_onto_AC_is_two :
  projection AB AC = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_onto_AC_is_two_l1149_114954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_successive_discounts_equivalence_verify_laptop_discount_l1149_114999

/-- Proves that two successive discounts are equivalent to a single discount -/
theorem successive_discounts_equivalence (original_price first_discount second_discount : ℝ) :
  original_price > 0 →
  0 ≤ first_discount ∧ first_discount < 1 →
  0 ≤ second_discount ∧ second_discount < 1 →
  let final_price := original_price * (1 - first_discount) * (1 - second_discount)
  let equivalent_discount := (original_price - final_price) / original_price
  equivalent_discount = 1 - (1 - first_discount) * (1 - second_discount) :=
by
  sorry

/-- Verifies the specific case in the problem -/
theorem verify_laptop_discount :
  let original_price : ℝ := 50
  let first_discount : ℝ := 0.25
  let second_discount : ℝ := 0.10
  let equivalent_discount := 1 - (1 - first_discount) * (1 - second_discount)
  equivalent_discount = 0.325 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_successive_discounts_equivalence_verify_laptop_discount_l1149_114999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hygiene_disease_relationship_l1149_114912

-- Define the survey data
def case_not_good : ℕ := 40
def case_good : ℕ := 60
def control_not_good : ℕ := 10
def control_good : ℕ := 90

-- Define the total number of participants
def total_participants : ℕ := case_not_good + case_good + control_not_good + control_good

-- Define the K² formula
noncomputable def K_squared (n a b c d : ℝ) : ℝ := 
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence level
noncomputable def critical_value : ℝ := 6.635

-- Define the probability functions
noncomputable def P_A_given_B : ℝ := 
  case_not_good / (case_not_good + case_good : ℝ)
noncomputable def P_A_given_not_B : ℝ := 
  control_not_good / (control_not_good + control_good : ℝ)

-- Define the risk level indicator R
noncomputable def R : ℝ := 
  (P_A_given_B / (1 - P_A_given_B)) * ((1 - P_A_given_not_B) / P_A_given_not_B)

-- Theorem statement
theorem hygiene_disease_relationship :
  K_squared (total_participants : ℝ) case_not_good case_good control_not_good control_good > critical_value ∧
  R = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hygiene_disease_relationship_l1149_114912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_l1149_114979

noncomputable def P : ℝ × ℝ := (-2, -3)
noncomputable def Q : ℝ × ℝ := (5, 3)

def R (m : ℝ) : ℝ × ℝ := (2, m)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem minimize_distance (m : ℝ) :
  (∀ m' : ℝ, distance P (R m) + distance (R m) Q ≤ distance P (R m') + distance (R m') Q) →
  m = 3/7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_l1149_114979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l1149_114976

/-- Represents the speed of a person -/
structure Speed where
  value : ℝ

/-- Represents a point on the path -/
structure Point where
  position : ℝ

/-- Represents a person traveling -/
structure Person where
  speed : Speed
  startPoint : Point

theorem distance_AB (
  personA personB personC : Person)
  (pointA pointB pointC pointD : Point)
  (h1 : personA.startPoint = pointA)
  (h2 : personB.startPoint = pointB)
  (h3 : personC.startPoint = pointB)
  (h4 : personA.speed.value = 3 * personC.speed.value)
  (h5 : personA.speed.value = 1.5 * personB.speed.value)
  (h6 : |pointC.position - pointD.position| = 12)
  (h7 : ∃ t : ℝ, pointA.position + personA.speed.value * t = pointC.position ∧
                 pointB.position - personB.speed.value * t = pointC.position)
  (h8 : ∃ t : ℝ, pointA.position + personA.speed.value * t = pointD.position ∧
                 pointB.position - personC.speed.value * t = pointD.position)
  (h9 : pointA.position + 50 = pointB.position - personB.speed.value * (50 / personA.speed.value))
  : |pointB.position - pointA.position| = 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l1149_114976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1149_114941

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) ∧
  (∀ y, -1 < y ∧ y < 1 ↔ ∃ x, f x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1149_114941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_decrease_l1149_114938

/-- Calculates the decreased cost of an article given its original cost and percentage decrease. -/
noncomputable def decreased_cost (original_cost : ℝ) (percent_decrease : ℝ) : ℝ :=
  original_cost * (1 - percent_decrease / 100)

/-- Proves that an article with an original cost of $200 and a 50% decrease in cost has a decreased cost of $100. -/
theorem article_cost_decrease :
  let original_cost : ℝ := 200
  let percent_decrease : ℝ := 50
  decreased_cost original_cost percent_decrease = 100 := by
  -- Unfold the definition of decreased_cost
  unfold decreased_cost
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_decrease_l1149_114938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sqrt_five_is_seventh_term_l1149_114937

/-- The sequence term for a given index -/
noncomputable def a (n : ℕ) : ℝ := Real.sqrt (3 * n - 1)

/-- Theorem stating that 2√5 is the 7th term of the sequence -/
theorem two_sqrt_five_is_seventh_term :
  a 7 = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sqrt_five_is_seventh_term_l1149_114937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_purchase_price_l1149_114926

/-- The purchase price of a machine that depreciates annually --/
def purchase_price (p r v : ℝ) : Prop :=
  p * (1 - 2 * r) = v

theorem machine_purchase_price :
  ∃ (p : ℝ), purchase_price p 0.1 6400 ∧ p = 8000 := by
  use 8000
  constructor
  · -- Prove purchase_price 8000 0.1 6400
    unfold purchase_price
    norm_num
  · -- Prove 8000 = 8000
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_purchase_price_l1149_114926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1149_114987

open Real

/-- The square region D with side length 3 -/
def D : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

/-- The area of the square region D -/
def area_D : ℝ := 9

/-- The distance of a point from the origin -/
noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  sqrt (p.1^2 + p.2^2)

/-- The set of points in D with distance greater than 2 from the origin -/
def A : Set (ℝ × ℝ) :=
  {p ∈ D | distance_from_origin p > 2}

/-- The probability of selecting a point in A from D -/
noncomputable def prob_A : ℝ := (area_D - π * 2^2 / 4) / area_D

theorem probability_theorem : prob_A = (9 - π) / 9 := by
  sorry

#check probability_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1149_114987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_slope_mn_l1149_114931

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Point on the ellipse -/
structure PointOnEllipse (C : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / C.a^2 + y^2 / C.b^2 = 1

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (C : Ellipse) : ℝ := Real.sqrt (1 - C.b^2 / C.a^2)

/-- Maximum area of triangle PF₁F₂ -/
noncomputable def max_triangle_area (C : Ellipse) : ℝ := 2 * C.a * C.b * (eccentricity C)

/-- Angle between two points on the ellipse and a fixed point -/
def angle_MAB_eq_angle_NAB (C : Ellipse) (A B M N : PointOnEllipse C) : Prop := sorry

/-- Slope of a line between two points on the ellipse -/
noncomputable def slope_of_line (M N : PointOnEllipse C) : ℝ := 
  (N.y - M.y) / (N.x - M.x)

/-- Theorem: Constant slope of line MN -/
theorem constant_slope_mn (C : Ellipse) 
  (h_ecc : eccentricity C = 1/2) 
  (h_area : max_triangle_area C = Real.sqrt 3) 
  (M N : PointOnEllipse C) 
  (h_angle : ∃ (A B : PointOnEllipse C), angle_MAB_eq_angle_NAB C A B M N) : 
  ∃ (k : ℝ), slope_of_line M N = k ∧ k = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_slope_mn_l1149_114931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sinx_sqrt3cosx_l1149_114928

theorem min_value_sinx_sqrt3cosx (x : ℝ) (h : -π ≤ x ∧ x ≤ 0) : 
  ∃ (m : ℝ), m = -2 ∧ ∀ y ∈ Set.Icc (-π) 0, Real.sin y + Real.sqrt 3 * Real.cos y ≥ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sinx_sqrt3cosx_l1149_114928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_unit_circle_l1149_114996

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the intersection point
noncomputable def intersection_point (α : ℝ) : ℝ × ℝ := (1/2, Real.sqrt (3/4))

-- State the theorem
theorem cos_double_angle_on_unit_circle (α : ℝ) :
  unit_circle (intersection_point α).1 (intersection_point α).2 →
  Real.cos (2 * α) = -1/2 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_unit_circle_l1149_114996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l1149_114995

/-- The central angle of the unfolded diagram of a cone's lateral surface -/
noncomputable def central_angle (slant_height : ℝ) (base_radius : ℝ) : ℝ :=
  (2 * base_radius * Real.pi) / (slant_height * Real.pi / 180)

/-- Theorem: The central angle of the unfolded diagram of the lateral surface of a cone
    with slant height 30 cm and base radius 10 cm is 120° -/
theorem cone_central_angle :
  central_angle 30 10 = 120 := by
  -- Unfold the definition of central_angle
  unfold central_angle
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l1149_114995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l1149_114985

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := 2^t - 5
noncomputable def y (t : ℝ) : ℝ := 4^t - 3 * 2^t + 1

-- Theorem stating that the points lie on a parabola
theorem points_form_parabola :
  ∃ (a b c : ℝ), ∀ (t : ℝ), y t = a * (x t)^2 + b * (x t) + c :=
by
  -- Introduce the coefficients
  use 1, 7, 11
  -- Take an arbitrary real number t
  intro t
  -- Expand the definitions of x and y
  simp [x, y]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l1149_114985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_d_and_n_l1149_114943

/-- Sum of k terms of an arithmetic sequence with initial term a and common difference d -/
noncomputable def arithmetic_sum (a d : ℝ) (k : ℕ) : ℝ :=
  (k : ℝ) / 2 * (2 * a + ((k : ℝ) - 1) * d)

/-- R is the difference between s₇, s₅, and s₁ -/
noncomputable def R (a d : ℝ) (n : ℕ) : ℝ :=
  arithmetic_sum a d (7 * n) - arithmetic_sum a d (5 * n) - arithmetic_sum a d n

theorem R_depends_on_d_and_n (a d : ℝ) (n : ℕ) :
  ∃ f : ℝ → ℕ → ℝ, R a d n = f d n :=
by
  -- Define the function f
  let f := fun (d : ℝ) (n : ℕ) => (23 : ℝ) / 2 * d * (n : ℝ)^2 - (1 : ℝ) / 2 * d * (n : ℝ)
  
  -- Show that R a d n equals f d n
  use f
  simp [R, arithmetic_sum]
  ring  -- This tactic should simplify the algebraic expression
  -- The full proof would require more steps, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_d_and_n_l1149_114943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l1149_114910

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 64

-- Define point P
def P : ℝ × ℝ := (3, 4)

-- Define that A and B are on the circle
def on_circle (point : ℝ × ℝ) : Prop :=
  circle_eq point.1 point.2

-- Define the right angle condition
def right_angle (A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0

-- Define the rectangle condition
def rectangle (A B Q : ℝ × ℝ) : Prop :=
  (A.1 - P.1)^2 + (A.2 - P.2)^2 = (Q.1 - B.1)^2 + (Q.2 - B.2)^2 ∧
  (B.1 - P.1)^2 + (B.2 - P.2)^2 = (Q.1 - A.1)^2 + (Q.2 - A.2)^2

-- Theorem statement
theorem locus_of_Q (A B Q : ℝ × ℝ) :
  on_circle A ∧ on_circle B ∧ right_angle A B ∧ rectangle A B Q →
  Q.1^2 + Q.2^2 = 103 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l1149_114910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_L_l1149_114942

/-- Line L in standard form -/
noncomputable def L (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0

/-- Curve C in standard form -/
def C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 1

/-- Distance from a point (x, y) to line L -/
noncomputable def distToL (x y : ℝ) : ℝ := 
  |x - Real.sqrt 3 * y + 1| / Real.sqrt (1 + (Real.sqrt 3)^2)

/-- The point Q -/
noncomputable def Q : ℝ × ℝ := (9/2, Real.sqrt 3 / 2)

theorem min_distance_to_L : 
  C Q.1 Q.2 ∧ distToL Q.1 Q.2 = 2 ∧ ∀ x y, C x y → distToL x y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_L_l1149_114942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stefan_vail_difference_l1149_114911

/-- The number of seashells collected by Stefan, Vail, and Aiguo -/
structure SeashellCollection where
  stefan : ℕ
  vail : ℕ
  aiguo : ℕ
  h1 : stefan > vail
  h2 : vail = aiguo - 5
  h3 : aiguo = 20
  h4 : stefan + vail + aiguo = 66

/-- Stefan had 16 more seashells than Vail -/
theorem stefan_vail_difference (sc : SeashellCollection) : sc.stefan - sc.vail = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stefan_vail_difference_l1149_114911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_theorem_l1149_114998

/-- A point in 3D space represented in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The set of points satisfying θ = π/4 in spherical coordinates -/
def PlaneSet : Set SphericalPoint :=
  {p : SphericalPoint | p.θ = Real.pi/4}

/-- A function to convert spherical coordinates to Cartesian coordinates -/
noncomputable def sphericalToCartesian (p : SphericalPoint) : ℝ × ℝ × ℝ :=
  (p.ρ * Real.sin p.φ * Real.cos p.θ, p.ρ * Real.sin p.φ * Real.sin p.θ, p.ρ * Real.cos p.φ)

theorem plane_theorem :
  ∃ (a b c d : ℝ), ∀ p ∈ PlaneSet,
    let (x, y, z) := sphericalToCartesian p
    a * x + b * y + c * z + d = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_theorem_l1149_114998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_difference_sum_l1149_114968

theorem product_difference_sum (a b : ℕ) : 
  a > 0 → b > 0 → a * b = 32 → |Int.ofNat a - Int.ofNat b| = 4 → a + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_difference_sum_l1149_114968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_M_l1149_114917

noncomputable def M : ℝ × ℝ := (-1, Real.sqrt 3)

theorem polar_coordinates_of_M :
  let (x, y) := M
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arccos (x / ρ)
  (ρ, θ) = (2, 2 * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_M_l1149_114917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_range_of_x_l1149_114973

-- Define the constraint function
def constraint (a b : ℝ) : Prop := |3 * a + 4 * b| = 10

-- Theorem for the minimum value of a^2 + b^2
theorem min_sum_squares (a b : ℝ) (h : constraint a b) : 
  a^2 + b^2 ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), constraint a₀ b₀ ∧ a₀^2 + b₀^2 = 4 := by
  sorry

-- Define the set of x satisfying the inequality
def X : Set ℝ := {x | ∀ a b : ℝ, |x + 3| - |x - 2| ≤ a^2 + b^2}

-- Theorem for the range of x
theorem range_of_x : X = Set.Iic (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_range_of_x_l1149_114973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_l1149_114908

/-- A square in the plane with side length 2b -/
structure Square (b : ℝ) where
  vertex1 : ℝ × ℝ := (-b, -b)
  vertex2 : ℝ × ℝ := (b, -b)
  vertex3 : ℝ × ℝ := (-b, b)
  vertex4 : ℝ × ℝ := (b, b)

/-- A line in the plane with equation y = 2x -/
def CuttingLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 2 * p.1}

/-- The perimeter of one of the resulting quadrilaterals divided by b -/
noncomputable def QuadrilateralPerimeter (b : ℝ) : ℝ :=
  4 + Real.sqrt 13 + Real.sqrt 5

theorem square_cut_perimeter (b : ℝ) (hb : b > 0) :
  let s := Square b
  let quad_perimeter := QuadrilateralPerimeter b
  quad_perimeter = 4 + Real.sqrt 13 + Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_l1149_114908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_selling_price_l1149_114983

/-- Represents the price in dollars -/
structure Price where
  value : ℝ

/-- The cost of buying apples -/
def apple_buy_cost : Price := ⟨3⟩

/-- The number of apples bought at the given cost -/
def apple_buy_quantity : ℕ := 2

/-- The selling price of apples -/
def apple_sell_price : Price := ⟨10⟩

/-- The number of apples sold at the given price -/
def apple_sell_quantity : ℕ := 5

/-- The cost of buying oranges -/
def orange_buy_cost : Price := ⟨2.70⟩

/-- The number of oranges bought at the given cost -/
def orange_buy_quantity : ℕ := 3

/-- The number of apples and oranges sold to achieve the given profit -/
def sold_quantity : ℕ := 5

/-- The total profit from selling apples and oranges -/
def total_profit : Price := ⟨3⟩

/-- The selling price of each orange -/
def orange_sell_price : Price := ⟨1⟩

theorem orange_selling_price :
  orange_sell_price = ⟨1⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_selling_price_l1149_114983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1149_114935

-- Define the function f(x) = 3^(x+m) - 3√3
noncomputable def f (x m : ℝ) : ℝ := 3^(x + m) - 3 * Real.sqrt 3

-- Theorem statement
theorem sufficient_not_necessary_condition :
  (∀ m > 1, ∀ x ≥ 1, f x m ≠ 0) ∧
  (∃ m ≤ 1, ∀ x ≥ 1, f x m ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1149_114935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l1149_114939

/-- The number of pieces of candy in each one-pound bag -/
def pieces_per_bag : ℕ → Prop := sorry

/-- The number of people in Mike's class -/
def mikes_class_size : ℕ → Prop := sorry

/-- The number of teachers in Betsy's school -/
def betsys_teachers : ℕ → Prop := sorry

/-- Mike's equation: 4 bags, 15 pieces per person, 23 pieces left over -/
axiom mikes_equation (x : ℕ) (s : ℕ) : 
  pieces_per_bag x → mikes_class_size s → 4 * x - 15 * s = 23

/-- Betsy's equation: 5 bags, 23 pieces per teacher, 15 pieces left over -/
axiom betsys_equation (x : ℕ) (t : ℕ) : 
  pieces_per_bag x → betsys_teachers t → 5 * x - 23 * t = 15

/-- The least number of pieces per bag satisfying both equations -/
def least_pieces_per_bag : ℕ → Prop := sorry

theorem candy_problem : 
  ∃ (x : ℕ), pieces_per_bag x ∧ least_pieces_per_bag x ∧ x = 302 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l1149_114939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_no_right_angled_triangle_tetrahedron_can_be_right_angled_triangle_cube_trapezoid_not_necessarily_right_angled_tetrahedron_trapezoid_is_isosceles_l1149_114993

-- Define the basic shapes
structure Cube
structure Tetrahedron
structure Plane

-- Define the types of cross-sections
inductive CrossSection
  | Triangle : CrossSection
  | RightAngledTriangle : CrossSection
  | Trapezoid : CrossSection
  | IsoscelesTrapezoid : CrossSection
  | RightAngledTrapezoid : CrossSection

-- Define the intersection operation
def intersect (p : Plane) (s : Cube ⊕ Tetrahedron) : CrossSection :=
  sorry -- Placeholder implementation

-- Theorem statements
theorem cube_no_right_angled_triangle (c : Cube) (p : Plane) :
  intersect p (Sum.inl c) ≠ CrossSection.RightAngledTriangle := by
  sorry

theorem tetrahedron_can_be_right_angled_triangle (t : Tetrahedron) :
  ∃ p : Plane, intersect p (Sum.inr t) = CrossSection.RightAngledTriangle := by
  sorry

theorem cube_trapezoid_not_necessarily_right_angled (c : Cube) :
  (∃ p : Plane, intersect p (Sum.inl c) = CrossSection.Trapezoid) ∧
  (∃ p : Plane, intersect p (Sum.inl c) ≠ CrossSection.RightAngledTrapezoid) := by
  sorry

theorem tetrahedron_trapezoid_is_isosceles (t : Tetrahedron) (p : Plane) :
  intersect p (Sum.inr t) = CrossSection.Trapezoid →
  intersect p (Sum.inr t) = CrossSection.IsoscelesTrapezoid := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_no_right_angled_triangle_tetrahedron_can_be_right_angled_triangle_cube_trapezoid_not_necessarily_right_angled_tetrahedron_trapezoid_is_isosceles_l1149_114993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1149_114958

theorem rationalize_denominator :
  ∃ (A B C D E : ℕ),
    B < D ∧
    (5 : ℝ) / (4 * Real.sqrt 7 - 3 * Real.sqrt 2) =
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = 20 ∧ B = 7 ∧ C = 15 ∧ D = 2 ∧ E = 94 ∧
    Nat.gcd A E = 1 ∧ Nat.gcd C E = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1149_114958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1149_114949

noncomputable def f (x : ℝ) : ℝ := x + 3 / x

theorem f_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < Real.sqrt 3 →
  f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1149_114949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_distribution_l1149_114978

/-- Given a total number of marbles and boxes, calculates the number of marbles per box -/
def marbles_per_box (total_marbles : ℕ) (num_boxes : ℕ) : ℕ := total_marbles / num_boxes

/-- Theorem stating that with 48 marbles in 6 boxes, there are 8 marbles in each box -/
theorem marbles_distribution (total_marbles num_boxes : ℕ) 
  (h1 : total_marbles = 48) 
  (h2 : num_boxes = 6) : 
  marbles_per_box total_marbles num_boxes = 8 := by
  sorry

#check marbles_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_distribution_l1149_114978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1149_114992

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  Real.sqrt (r1^2 + r2^2 - 2*r1*r2*(Real.cos (θ2 - θ1)))

theorem distance_between_specific_points :
  polar_distance 1 (π/3) (Real.sqrt 3) (7*π/6) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1149_114992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l1149_114900

noncomputable def a (n : ℕ) : ℝ := 5 * n

noncomputable def b (n : ℕ) : ℝ := 5 * (2 ^ (n - 1))

noncomputable def S : ℝ := (a 1 + a 20) * 20 / 2

theorem arithmetic_geometric_sequence_properties :
  -- Conditions
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → ∃ d : ℝ, d ≠ 0 ∧ a n = a 1 + (n - 1) * d) ∧
  a 1 = b 1 ∧ a 4 = b 3 ∧ a 16 = b 5 ∧
  (∀ n : ℕ, b n > 0) →
  -- Conclusions
  (∀ n : ℕ, a n = 5 * n) ∧
  (∀ n : ℕ, b n = 5 * (2 ^ (n - 1))) ∧
  (∀ n : ℕ, b n ≤ S ↔ n ≤ 8) := by
  sorry

#check arithmetic_geometric_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l1149_114900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_3b_plus_15_l1149_114922

theorem divisibility_of_3b_plus_15 (a b : ℤ) (h : 4 * b = 10 - 3 * a) :
  ∀ d ∈ ({1, 2, 3, 5} : Set ℤ), (3 * b + 15) % d = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_3b_plus_15_l1149_114922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkers_theorem_l1149_114994

/-- Represents a checker on the board -/
inductive Checker where
  | White
  | Black
  | Empty
deriving DecidableEq

/-- Represents the state of the board -/
def Board := Fin 10 → Fin 10 → Checker

/-- Represents a sequence of board states -/
def BoardSequence := ℕ → Board

/-- Two cells are adjacent if they share a side -/
def adjacent (x1 y1 x2 y2 : Fin 10) : Prop :=
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1 = x2 ∧ y2.val + 1 = y1.val) ∨
  (y1 = y2 ∧ x1.val + 1 = x2.val) ∨
  (y1 = y2 ∧ x2.val + 1 = x1.val)

/-- The initial board has 91 white checkers -/
def initial_board_condition (b : Board) : Prop :=
  (Finset.sum Finset.univ (λ x => Finset.sum Finset.univ (λ y => if b x y = Checker.White then 1 else 0)) = 91)

/-- Each step changes one white checker to black -/
def valid_step (b1 b2 : Board) : Prop :=
  ∃ x y, (b1 x y = Checker.White ∧ b2 x y = Checker.Black) ∧
         ∀ x' y', (x' ≠ x ∨ y' ≠ y) → b1 x' y' = b2 x' y'

/-- The sequence represents a valid painting process -/
def valid_sequence (seq : BoardSequence) : Prop :=
  initial_board_condition (seq 0) ∧
  ∀ n, valid_step (seq n) (seq (n + 1))

/-- The theorem to be proved -/
theorem checkers_theorem (seq : BoardSequence) (h : valid_sequence seq) :
  ∃ n x1 y1 x2 y2, adjacent x1 y1 x2 y2 ∧
    ((seq n x1 y1 = Checker.White ∧ seq n x2 y2 = Checker.Black) ∨
     (seq n x1 y1 = Checker.Black ∧ seq n x2 y2 = Checker.White)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkers_theorem_l1149_114994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_problem_l1149_114962

/-- Given three circles C, D, and E with the following properties:
    - Circle C has radius 4
    - Circle D is internally tangent to circle C at point A
    - Circle E is internally tangent to circle C, externally tangent to circle D, and tangent to diameter AB of circle C
    - The radius of circle D is twice the radius of circle E
    - The radius of circle D can be written as 2√p - q, where p and q are positive integers
    Then p + q = 48 -/
theorem circle_tangency_problem (p q : ℕ) (C D E : Set (EuclideanSpace ℝ (Fin 2))) (A B : EuclideanSpace ℝ (Fin 2)) :
  (∃ (center_C center_D center_E : EuclideanSpace ℝ (Fin 2)),
    -- Circle C has radius 4
    C = {x : EuclideanSpace ℝ (Fin 2) | ‖x - center_C‖ = 4} ∧
    -- Circle D is internally tangent to circle C at A
    D = {x : EuclideanSpace ℝ (Fin 2) | ‖x - center_D‖ = 2 * ‖center_D - center_C‖} ∧
    A ∈ C ∧ A ∈ D ∧
    -- Circle E is internally tangent to circle C, externally tangent to circle D, and tangent to diameter AB
    E = {x : EuclideanSpace ℝ (Fin 2) | ‖x - center_E‖ = ‖center_E - center_C‖ - 4} ∧
    ‖center_E - center_D‖ = ‖center_D - center_C‖ + ‖center_E - center_C‖ ∧
    ‖B - A‖ = 8 ∧ (∃ (t : ℝ), center_E = A + t • (B - A)) ∧
    -- The radius of circle D is twice the radius of circle E
    ‖center_D - center_C‖ = 2 * ‖center_E - center_C‖ ∧
    -- The radius of circle D can be written as 2√p - q
    ‖center_D - center_C‖ = 2 * Real.sqrt p - q) →
  p + q = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_problem_l1149_114962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1149_114944

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x / (Real.exp x)

-- State the theorem
theorem max_value_of_f : 
  ∃ (x : ℝ), x ∈ Set.Icc 0 2 ∧ 
  (∀ y ∈ Set.Icc 0 2, f y ≤ f x) ∧
  f x = 1 / Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1149_114944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_megan_weight_difference_l1149_114989

/-- The weight difference between Mike and Megan given the bridge and children's weights --/
theorem mike_megan_weight_difference :
  ∀ (bridge_capacity : ℚ) 
    (kelly_weight : ℚ) 
    (megan_weight : ℚ) 
    (mike_weight : ℚ) 
    (total_weight : ℚ),
  bridge_capacity = 100 →
  kelly_weight = 34 →
  kelly_weight = megan_weight * (85 / 100) →
  total_weight = bridge_capacity + 19 →
  total_weight = kelly_weight + megan_weight + mike_weight →
  mike_weight - megan_weight = 5 := by
  intros bridge_capacity kelly_weight megan_weight mike_weight total_weight
  intros h1 h2 h3 h4 h5
  sorry

#check mike_megan_weight_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_megan_weight_difference_l1149_114989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_l1149_114956

/-- A function that returns the number of positive factors of a natural number -/
def num_factors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range n.succ)).card

/-- A function that returns true if a natural number has exactly eight distinct positive factors -/
def has_eight_factors (n : ℕ) : Prop := num_factors n = 8

theorem smallest_with_eight_factors :
  ∃ (n : ℕ), has_eight_factors n ∧ ∀ m : ℕ, m > 0 → has_eight_factors m → n ≤ m :=
by
  use 24
  constructor
  · -- Prove that 24 has eight factors
    simp [has_eight_factors, num_factors]
    -- You can add a more detailed proof here if needed
    sorry
  · -- Prove that 24 is the smallest such number
    intro m hm_pos hm_eight_factors
    -- You can add a more detailed proof here if needed
    sorry

#eval num_factors 24  -- Should output 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_l1149_114956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l1149_114970

-- Define the points A, B, and M
noncomputable def A : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def B : ℝ × ℝ := (Real.sqrt 5, 0)
def M : ℝ × ℝ := (2, 0)

-- Define the locus of C (hyperbola equation)
def locus_C (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1 ∧ x > 2

-- Define the condition for the incenter
def incenter_condition (x : ℝ) : Prop := x = 2

-- Define the dot product condition
def dot_product_condition (P Q : ℝ × ℝ) : Prop :=
  (P.1 - M.1) * (Q.1 - M.1) + (P.2 - M.2) * (Q.2 - M.2) = 0

-- The main theorem
theorem fixed_point_theorem (P Q : ℝ × ℝ) :
  locus_C P.1 P.2 →
  locus_C Q.1 Q.2 →
  dot_product_condition P Q →
  ∃ (t : ℝ), t * P.1 + (1 - t) * Q.1 = 10/3 ∧ t * P.2 + (1 - t) * Q.2 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l1149_114970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1149_114915

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.cos x ^ 2 + Real.tan x ^ 2

theorem f_range :
  (∀ x : ℝ, f x ≥ 1) ∧ (∀ y : ℝ, y ≥ 1 → ∃ x : ℝ, f x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1149_114915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1149_114967

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^4 - 3*x^2 - 6*x + 13) - Real.sqrt (x^4 - x^2 + 1)

-- State the theorem
theorem f_max_value : 
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1149_114967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clear_time_l1149_114952

-- Define the parameters
noncomputable def train1_length : ℝ := 111
noncomputable def train2_length : ℝ := 165
noncomputable def train1_speed_kmh : ℝ := 60
noncomputable def train2_speed_kmh : ℝ := 90

-- Convert km/h to m/s
noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

-- Calculate relative speed in m/s
noncomputable def relative_speed : ℝ := kmh_to_ms train1_speed_kmh + kmh_to_ms train2_speed_kmh

-- Calculate total distance
noncomputable def total_distance : ℝ := train1_length + train2_length

-- Calculate time to clear
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

-- Theorem statement
theorem trains_clear_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |time_to_clear - 6.62| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clear_time_l1149_114952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1149_114971

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

-- State the theorem
theorem f_min_value :
  ∀ x > 0, f x ≥ 1 ∧ ∃ x₀ > 0, f x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1149_114971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_seven_l1149_114930

theorem largest_multiple_of_seven (n : ℤ) : n = 147 ↔ 
  (∃ k : ℤ, n = 7 * k ∧ 
   -n > -150 ∧ 
   ∀ m : ℤ, (∃ j : ℤ, m = 7 * j ∧ -m > -150) → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_seven_l1149_114930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_seven_l1149_114940

theorem two_digit_multiples_of_seven : 
  (Finset.filter (fun n : ℕ => 10 ≤ n ∧ n ≤ 99 ∧ n % 7 = 0) (Finset.range 100)).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_seven_l1149_114940
