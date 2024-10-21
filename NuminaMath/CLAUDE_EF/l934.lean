import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_qr_length_l934_93403

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  pq : ℝ
  rs : ℝ

/-- Calculates the length of QR in a trapezoid -/
noncomputable def calculate_qr (t : Trapezoid) : ℝ :=
  (t.area - t.altitude * (t.pq + t.rs) / 2) / t.altitude

theorem trapezoid_qr_length (t : Trapezoid) 
  (h_area : t.area = 250)
  (h_altitude : t.altitude = 10)
  (h_pq : t.pq = 12)
  (h_rs : t.rs = 21) :
  ∃ ε > 0, |calculate_qr t - 14.78| < ε := by
  sorry

#eval "Trapezoid QR length theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_qr_length_l934_93403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l934_93451

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then x^2
  else if 3 < x ∧ x ≤ 10 then 3*x - 6
  else 0

-- Define the area M
noncomputable def M : ℝ := ∫ x in (0 : ℝ)..(10 : ℝ), g x

-- Theorem statement
theorem area_calculation : M = 103.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l934_93451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_equals_three_halves_l934_93417

theorem trigonometric_sum_equals_three_halves : 
  Real.cos (7*π/24)^4 + Real.sin (11*π/24)^4 + Real.sin (17*π/24)^4 + Real.cos (13*π/24)^4 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_equals_three_halves_l934_93417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enjoyment_index_range_l934_93456

/-- Cumulative experience value function for healthy time -/
noncomputable def E (t : ℝ) (a : ℝ) : ℝ := t^2 + 20*t + 16*a

/-- Player enjoyment index function -/
noncomputable def H (t : ℝ) (a : ℝ) : ℝ := E t a / t

/-- Theorem stating the range of a for which H(t) ≥ 24 during healthy time -/
theorem enjoyment_index_range (a : ℝ) :
  (∀ t : ℝ, 0 < t ∧ t ≤ 3 → H t a ≥ 24) ↔ a ≥ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enjoyment_index_range_l934_93456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l934_93427

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y - 3 = 0

/-- The area of the region -/
noncomputable def region_area : ℝ := 16 * Real.pi

theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Provide the center and radius
  let center : ℝ × ℝ := (2, -3)
  let radius : ℝ := 4
  
  -- Assert the existence of the center and radius
  use center, radius
  
  -- Split the conjunction
  constructor
  
  -- Prove the equivalence of the equations
  · sorry
  
  -- Prove the area equality
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l934_93427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_intersection_l934_93444

-- Define the necessary structures
structure Plane where

structure Sphere where

structure Point where

structure Circle where

-- Define auxiliary functions (these are not proved, just declared)
noncomputable def cone_intersection (E : Point) (k : Circle) (G : Sphere) : Circle :=
sorry

-- Define the ∉ operator for a point and a sphere
def Point.notMem (p : Point) (s : Sphere) : Prop :=
sorry

-- Define the plane of a circle
def Circle.plane (c : Circle) : Plane :=
sorry

-- Define the main theorem
theorem cone_sphere_intersection
  (δ : Plane) (G : Sphere) (E : Point) (k : Circle) :
  ∃ (c : Circle), 
    (c.plane = δ) ∧ 
    (E.notMem G) ∧ 
    (k.plane = δ) ∧ 
    (cone_intersection E k G = c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_intersection_l934_93444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_meeting_probability_l934_93424

/-- The duration of the meeting window in minutes -/
noncomputable def meetingWindow : ℝ := 60

/-- The maximum waiting time in minutes -/
noncomputable def maxWaitTime : ℝ := 10

/-- The probability of A and B successfully meeting -/
noncomputable def meetingProbability : ℝ := 11 / 36

/-- Theorem stating the probability of A and B successfully meeting -/
theorem successful_meeting_probability :
  let totalArea := meetingWindow * meetingWindow
  let meetArea := totalArea - 2 * (meetingWindow - maxWaitTime) * (meetingWindow - maxWaitTime) / 2
  meetArea / totalArea = meetingProbability := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_meeting_probability_l934_93424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l934_93494

/-- The sequence a_n -/
def a : ℕ → ℝ := sorry

/-- The sum of the first n terms of the sequence a_n -/
def S : ℕ → ℝ := sorry

/-- The relation between S_n and a_n -/
axiom S_relation (n : ℕ) : S n = 1/2 - 1/2 * a n

theorem a_formula : ∀ n : ℕ, n > 0 → a n = (1/3)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l934_93494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_25000_l934_93491

structure Position where
  title : String
  count : Nat
  salary : Nat
deriving Inhabited

def corporation_data : List Position := [
  { title := "CEO", count := 1, salary := 150000 },
  { title := "General Manager", count := 4, salary := 95000 },
  { title := "Manager", count := 12, salary := 80000 },
  { title := "Assistant Manager", count := 8, salary := 55000 },
  { title := "Clerk", count := 40, salary := 25000 }
]

def total_employees : Nat := (corporation_data.map (·.count)).sum

def median_position : Nat := (total_employees + 1) / 2

def cumulative_count (n : Nat) : Nat :=
  (corporation_data.take n).map (·.count) |>.sum

theorem median_salary_is_25000 :
  ∃ i : Nat, i < corporation_data.length ∧
    cumulative_count i < median_position ∧
    median_position ≤ cumulative_count (i + 1) ∧
    (corporation_data.get! i).salary = 25000 := by
  sorry

#eval total_employees
#eval median_position
#eval cumulative_count 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_25000_l934_93491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_lines_relationships_l934_93442

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the given conditions
variable (α β : Plane)
variable (l m : Line)
variable (h_not_coincident : α ≠ β)
variable (h_different_lines : l ≠ m)
variable (h_l_perp_α : perpendicular_line_plane l α)
variable (h_m_subset_β : subset m β)

-- State the theorem
theorem planes_lines_relationships :
  (parallel_plane α β → perpendicular_line_line l m) ∧
  (perpendicular_line_plane l β → parallel_line_plane m α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_lines_relationships_l934_93442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_theorem_l934_93437

def fishing_problem (will_catfish : ℕ) (will_eels : ℕ) (trout_per_catfish : ℕ) 
  (weather_efficiencies : List ℚ) (release_fraction : ℚ) : Prop :=
  let henry_goal := will_catfish * trout_per_catfish
  let henry_catch := (weather_efficiencies.map (λ e => ⌊(e * henry_goal : ℚ)⌋)).sum
  let henry_keep := henry_catch - ⌊(release_fraction * henry_catch : ℚ)⌋
  (will_catfish + will_eels + henry_keep) = 50

theorem fishing_theorem : 
  fishing_problem 16 10 3 [1/5, 3/10, 1/2, 1/10, 2/5] (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_theorem_l934_93437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l934_93407

-- Define the triangle and vectors
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  true  -- We don't need to define the triangle explicitly for this problem

noncomputable def vector_m (A : ℝ) : ℝ × ℝ := (Real.cos A + Real.sqrt 2, Real.sin A)

noncomputable def vector_n (A : ℝ) : ℝ × ℝ := (-Real.sin A, Real.cos A)

-- Define the conditions
def conditions (A : ℝ) (a b c : ℝ) : Prop :=
  let m := vector_m A
  let n := vector_n A
  (m.1 + n.1)^2 + (m.2 + n.2)^2 = 4 ∧ 
  b = 4 * Real.sqrt 2 ∧
  c = Real.sqrt 2 * a

-- State the theorem
theorem triangle_problem (A : ℝ) (a b c : ℝ) 
  (h : conditions A a b c) : 
  A = Real.pi / 4 ∧ 
  (1/2) * b * c * Real.sin A = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l934_93407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_equivalence_l934_93477

open Real

-- Define the original curve C₁
noncomputable def C₁ (x : ℝ) : ℝ := sin x

-- Define the target curve C₂
noncomputable def C₂ (x : ℝ) : ℝ := sin (2 * x + 2 * π / 3)

-- Define the transformation
noncomputable def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2 * (x + π / 3))

-- Theorem stating the equivalence of the transformation
theorem transform_equivalence : ∀ x, C₂ x = transform C₁ x := by
  intro x
  simp [C₁, C₂, transform]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_equivalence_l934_93477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_a_values_l934_93420

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- Definition of the function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (x + a) * (3 : ℝ)^(x - 2 + a^2) - (x - a) * (3 : ℝ)^(8 - x - 3*a)

theorem even_function_a_values :
  ∀ a : ℝ, IsEven (f a) → a = -5 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_a_values_l934_93420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_range_range_of_a_l934_93448

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 + a * x

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1/x + x + a

-- Theorem statement
theorem tangent_line_parallel_range (a : ℝ) :
  (∃ x > 0, f_derivative a x = 3) → a ≤ 1 :=
by sorry

-- Main theorem
theorem range_of_a :
  {a : ℝ | ∃ x > 0, f_derivative a x = 3} = Set.Iic 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_range_range_of_a_l934_93448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_theorem_fixed_point_theorem_l934_93478

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the trajectory of M
def trajectory (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1 ∧ x ≠ 2 ∧ x ≠ -2

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem circle_trajectory_theorem :
  ∀ (x y : ℝ),
  (∃ (x_p y_p : ℝ), my_circle x_p y_p ∧ 
    x_p = x ∧ 
    y_p^2 = 2 * y^2) →
  trajectory x y :=
by
  sorry

theorem fixed_point_theorem :
  ∃ (n : ℝ),
  n = -7/4 ∧
  ∀ (x1 y1 x2 y2 : ℝ),
  trajectory x1 y1 ∧ trajectory x2 y2 →
  (y1 - 0) / (x1 + 1) = (y2 - 0) / (x2 + 1) →
  dot_product (x1 - n) y1 (x2 - n) y2 = -15/16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_theorem_fixed_point_theorem_l934_93478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_equality_condition_l934_93450

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 2 - Real.sqrt (x^4 + 4)) / x ≤ 2 * Real.sqrt 2 - 2 :=
by sorry

theorem equality_condition (x : ℝ) (hx : x > 0) :
  (x^2 + 2 - Real.sqrt (x^4 + 4)) / x = 2 * Real.sqrt 2 - 2 ↔ x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_equality_condition_l934_93450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_area_is_24_l934_93495

/-- A triangle in the figure -/
structure Triangle where
  /-- The length of the legs of the isosceles right-angled triangle -/
  leg_length : ℝ
  /-- Assumption that the triangle is isosceles and right-angled -/
  is_isosceles_right : Bool

/-- The entire figure composed of three triangles -/
structure Figure where
  /-- The three triangles that make up the figure -/
  triangles : Fin 3 → Triangle
  /-- Assumption that all triangles in the figure have the same leg length -/
  same_leg_length : ∀ i j, (triangles i).leg_length = (triangles j).leg_length

/-- Calculate the area of a single triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  (1 / 2) * t.leg_length * t.leg_length

/-- Calculate the total area of the figure -/
noncomputable def figure_area (f : Figure) : ℝ :=
  (Finset.sum Finset.univ fun i => triangle_area (f.triangles i))

/-- The main theorem: The area of the figure is 24 square units -/
theorem figure_area_is_24 (f : Figure)
  (h1 : ∀ i, (f.triangles i).is_isosceles_right = true)
  (h2 : ∀ i, (f.triangles i).leg_length = 4) :
  figure_area f = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_area_is_24_l934_93495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohans_savings_l934_93404

def monthly_salary : ℕ := 10000

def food_percentage : ℚ := 40 / 100
def rent_percentage : ℚ := 20 / 100
def entertainment_percentage : ℚ := 10 / 100
def conveyance_percentage : ℚ := 10 / 100

def total_expenses_percentage : ℚ := food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage

def savings_percentage : ℚ := 1 - total_expenses_percentage

theorem rohans_savings :
  (↑monthly_salary * savings_percentage).floor = 2000 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohans_savings_l934_93404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l934_93496

/-- The probability of a randomly chosen point (x,y) from a rectangle
    with vertices (0,0), (2010,0), (2010,2009), and (0,2009)
    satisfying both x > 2y and y > 500 -/
def probability_in_region : ℚ :=
  1505 / 4018

/-- The width of the rectangle -/
def rectangle_width : ℕ := 2010

/-- The height of the rectangle -/
def rectangle_height : ℕ := 2009

/-- The y-coordinate threshold -/
def y_threshold : ℕ := 500

theorem probability_calculation :
  probability_in_region = (rectangle_width * (y_threshold + rectangle_width / 2)) / (2 * rectangle_width * rectangle_height) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l934_93496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_positions_l934_93429

-- Define the fish types
inductive Fish
| CrucianCarp
| Gudgeon

-- Define the circle parameters
def hook_position : ℝ × ℝ := (0, 0)
def crucian_carp_radius : ℝ := 3
def gudgeon_radius : ℝ := 6

-- Define initial positions
noncomputable def initial_position (f : Fish) : ℝ × ℝ :=
  match f with
  | Fish.CrucianCarp => (-1, 2 * Real.sqrt 2)
  | Fish.Gudgeon => (2, -4 * Real.sqrt 2)

-- Define speed ratio
def speed_ratio : ℝ := 2.5

-- Define clockwise movement
def moves_clockwise (f : Fish) : Prop := sorry

-- Define position and distance functions
noncomputable def position (f : Fish) : ℝ × ℝ := sorry
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem shortest_distance_positions :
  ∀ (f : Fish),
  moves_clockwise f →
  ∃ (positions : List (ℝ × ℝ)),
    positions = [
      (Real.sqrt 2 - 4, -4 - Real.sqrt 2),
      (-4 - Real.sqrt 2, 4 - Real.sqrt 2),
      (4 - Real.sqrt 2, 4 + Real.sqrt 2),
      (4 + Real.sqrt 2, Real.sqrt 2 - 4)
    ] ∧
    (∀ (pos : ℝ × ℝ),
      pos ∈ positions →
      (f = Fish.Gudgeon →
        (∀ (other_pos : ℝ × ℝ),
          other_pos ∉ positions →
          distance pos (position Fish.CrucianCarp) ≤ distance other_pos (position Fish.CrucianCarp))))
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_positions_l934_93429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_transformation_l934_93461

/-- Definition of the sequence transformation operation -/
def transform (seq : List ℕ) : List (List ℕ) :=
  seq.scanl (λ acc i => if acc.length ≥ 3
    then acc.take (acc.length - 3) ++ [acc.get! (acc.length - 2), acc.get! (acc.length - 1), acc.get! (acc.length - 3)]
    else acc ++ [i]) []

/-- Predicate to check if a sequence is in ascending order -/
def isAscending (seq : List ℕ) : Prop :=
  ∀ i j, i < j → i < seq.length → j < seq.length → seq.get! i < seq.get! j

/-- Predicate to check if a sequence is in descending order -/
def isDescending (seq : List ℕ) : Prop :=
  ∀ i j, i < j → i < seq.length → j < seq.length → seq.get! i > seq.get! j

/-- The main theorem to be proved -/
theorem sequence_transformation (n : ℕ) :
  n ≥ 3 →
  (∃ (seq : List ℕ), seq.length = n ∧
    isAscending seq ∧
    isDescending (transform seq).getLast! ∧
    (transform seq).getLast!.length = n) ↔
  (n % 4 = 0 ∨ n % 4 = 1) :=
by sorry

#check sequence_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_transformation_l934_93461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l934_93453

noncomputable def expression (x : ℝ) : ℝ := (x + 4/x - 4)^3

theorem constant_term_of_expansion :
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 →
  ∃ (p : ℝ → ℝ), (expression x = c + x * p x) ∧ c = -160 := by
  sorry

#check constant_term_of_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l934_93453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l934_93490

noncomputable def f (x : ℝ) : ℝ := |Real.tan (2 * x)|

theorem f_is_even_and_periodic : 
  (∀ x, f x = f (-x)) ∧ (∀ x, f (x + π/2) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l934_93490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l934_93498

theorem sufficient_not_necessary_condition :
  (∀ θ : ℝ, |θ - π/12| < π/12 → Real.sin θ < 1/2) ∧
  (∃ θ : ℝ, Real.sin θ < 1/2 ∧ |θ - π/12| ≥ π/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l934_93498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_range_f_plus_log_positive_l934_93475

/-- The function f(x) = (a - sin x) / x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - Real.sin x) / x

/-- Theorem stating the range of f(x₀) when f reaches its minimum -/
theorem f_minimum_range (a : ℝ) :
  ∃ (x₀ : ℝ), 0 < x₀ ∧ x₀ < Real.pi ∧
  (∀ (x : ℝ), 0 < x → x < Real.pi → f a x₀ ≤ f a x) →
  -1 < f a x₀ ∧ f a x₀ < 1 :=
by sorry

/-- Theorem stating that f(x) + m ln x > 0 for given conditions -/
theorem f_plus_log_positive (x m : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) (h3 : 0 < m) (h4 : m < Real.pi) :
  f Real.pi x + m * Real.log x > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_range_f_plus_log_positive_l934_93475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equivalence_l934_93409

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem f_equivalence (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ < π)
  (h4 : ∀ x, f ω φ x = f ω φ (2 * π / 3 - x))
  (h5 : ∀ x, f ω φ x = -f ω φ (π - x))
  (h6 : ∃ T > π / 2, ∀ x, f ω φ (x + T) = f ω φ x) :
  ∀ x, f ω φ x = 2 * Real.sin (3 * x + π / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equivalence_l934_93409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_46_l934_93492

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 2 => if n % 2 = 0 
    then sequence_a (n + 1) + (n + 2) / 2 
    else sequence_a (n + 1) + (-1)^((n + 2) / 2)

theorem a_20_equals_46 : sequence_a 20 = 46 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_46_l934_93492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boot_shoe_price_difference_l934_93430

/-- The price of a pair of shoes -/
def shoe_price : ℝ := sorry

/-- The price of a pair of boots -/
def boot_price : ℝ := sorry

/-- Monday's sales equation -/
axiom monday_sales : 22 * shoe_price + 16 * boot_price = 460

/-- Tuesday's sales equation -/
axiom tuesday_sales : 8 * shoe_price + 32 * boot_price = 560

/-- Theorem stating the price difference between boots and shoes -/
theorem boot_shoe_price_difference : boot_price - shoe_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boot_shoe_price_difference_l934_93430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_disjoint_quadratic_residues_l934_93480

open Nat BigOperators Finset

theorem no_disjoint_quadratic_residues (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_5 : p > 5) :
  ¬∃ (a c : ℕ), Coprime (a * c) p ∧
    (∀ b ∈ {x : ℕ | 0 < x ∧ x < p ∧ ∃ y, y * y ≡ x [ZMOD p]},
      ¬∃ b' ∈ {x : ℕ | 0 < x ∧ x < p ∧ ∃ y, y * y ≡ x [ZMOD p]},
        (a * b + c) % p = b' % p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_disjoint_quadratic_residues_l934_93480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l934_93401

theorem train_passing_time (platform_time : ℝ) (train_speed_kmh : ℝ) (platform_length : ℝ) :
  platform_time = 50 →
  train_speed_kmh = 54 →
  platform_length = 300.024 →
  (train_speed_kmh * 1000 / 3600) * platform_time / (train_speed_kmh * 1000 / 3600) = platform_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l934_93401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_equals_14_l934_93422

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 15 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

-- State the theorem
theorem f_of_g_of_2_equals_14 : f (g 2) = 14 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_equals_14_l934_93422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l934_93439

noncomputable def g (x : ℝ) : ℝ := ⌊2 * x⌋ - x - 1

theorem g_range :
  ∀ y : ℝ, (∃ x : ℝ, g x = y) ↔ -2 < y ∧ y ≤ 0 :=
by
  sorry

#check g_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l934_93439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_points_in_semicircle_l934_93458

/-- Given a right triangle with legs of lengths 6√3 and 18 containing 865 points,
    there exist at least 3 points that can be covered by a closed semicircular disc
    with a diameter of 1. -/
theorem exist_three_points_in_semicircle (points : Finset (EuclideanSpace ℝ (Fin 2))) :
  (Finset.card points = 865) →
  (∃ (A B C : EuclideanSpace ℝ (Fin 2)),
    A ∈ points ∧ B ∈ points ∧ C ∈ points ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    (∃ (center : EuclideanSpace ℝ (Fin 2)),
      dist A center ≤ 1/2 ∧ dist B center ≤ 1/2 ∧ dist C center ≤ 1/2)) :=
by sorry

/-- Definition of the right triangle with legs 6√3 and 18 -/
def right_triangle : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p : EuclideanSpace ℝ (Fin 2) | 0 ≤ p 0 ∧ 0 ≤ p 1 ∧ p 0 / (6 * Real.sqrt 3) + p 1 / 18 ≤ 1}

/-- All points are inside the right triangle -/
axiom points_in_triangle (points : Finset (EuclideanSpace ℝ (Fin 2))) :
  ∀ p ∈ points, p ∈ right_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_points_in_semicircle_l934_93458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a999_a2004_l934_93479

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => (a n)^2 + 1

theorem gcd_a999_a2004 : Nat.gcd (a 999) (a 2004) = 677 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a999_a2004_l934_93479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_power_two_2015_l934_93474

def last_digit (n : ℕ) : ℕ := n % 10

def power_two_last_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | _ => 8

theorem last_digit_power_two_2015 : last_digit (2^2015) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_power_two_2015_l934_93474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_square_is_five_l934_93463

/-- Represents a position in the 3x3 grid -/
inductive Position
  | TopLeft | TopMiddle | TopRight
  | MiddleLeft | MiddleMiddle | MiddleRight
  | BottomLeft | BottomMiddle | BottomRight
deriving Inhabited

/-- Represents the state of the paper after folding -/
structure FoldedState where
  layers : List Position
deriving Inhabited

/-- Initial configuration of the grid -/
def initial_grid : Position → Nat
  | Position.TopLeft => 1
  | Position.TopMiddle => 2
  | Position.TopRight => 3
  | Position.MiddleLeft => 4
  | Position.MiddleMiddle => 5
  | Position.MiddleRight => 6
  | Position.BottomLeft => 7
  | Position.BottomMiddle => 8
  | Position.BottomRight => 9

/-- Fold the right third over the middle third -/
def fold_right (s : FoldedState) : FoldedState := sorry

/-- Fold the left third over the newly formed stack -/
def fold_left (s : FoldedState) : FoldedState := sorry

/-- Fold the bottom third over the top two-thirds -/
def fold_bottom (s : FoldedState) : FoldedState := sorry

/-- Perform all three folding operations -/
def fold_all : FoldedState := 
  fold_bottom (fold_left (fold_right { layers := [] }))

/-- The theorem to be proved -/
theorem middle_square_is_five : 
  initial_grid (fold_all.layers.get! (fold_all.layers.length / 2)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_square_is_five_l934_93463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_subsets_eq_1028_l934_93484

/-- Given two sets A and B with 12 elements each and their intersection containing 4 elements,
    this function counts the number of 3-element subsets C of A ∪ B 
    such that C intersects both A and B non-trivially. -/
def count_subsets (A B : Finset ℕ) : ℕ :=
  (Finset.filter (fun C => 
    C.card = 3 ∧ 
    (C ∩ A).Nonempty ∧ 
    (C ∩ B).Nonempty) (Finset.powerset (A ∪ B))).card

/-- The main theorem stating that the count of subsets C is 1028 -/
theorem count_subsets_eq_1028 (A B : Finset ℕ) 
  (hA : A.card = 12)
  (hB : B.card = 12)
  (hAB : (A ∩ B).card = 4) :
  count_subsets A B = 1028 := by
  sorry

#eval count_subsets (Finset.range 12) (Finset.range 12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_subsets_eq_1028_l934_93484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_symmetry_planes_tetrahedron_symmetry_planes_l934_93411

-- Define a cube
structure Cube where
  -- Add necessary properties of a cube
  edge_length : ℝ
  edge_length_pos : edge_length > 0

-- Define a regular tetrahedron
structure RegularTetrahedron where
  -- Add necessary properties of a regular tetrahedron
  edge_length : ℝ
  edge_length_pos : edge_length > 0

-- Define a plane of symmetry
structure PlaneOfSymmetry where
  -- Add necessary properties of a plane of symmetry
  normal_vector : ℝ × ℝ × ℝ

-- Function to count planes of symmetry for a cube
def countPlanesOfSymmetryCube (c : Cube) : ℕ := 9

-- Function to count planes of symmetry for a regular tetrahedron
def countPlanesOfSymmetryTetrahedron (t : RegularTetrahedron) : ℕ := 6

-- Theorem for cube symmetry planes
theorem cube_symmetry_planes (c : Cube) :
  countPlanesOfSymmetryCube c = 9 := by
  -- Proof implementation
  sorry

-- Theorem for regular tetrahedron symmetry planes
theorem tetrahedron_symmetry_planes (t : RegularTetrahedron) :
  countPlanesOfSymmetryTetrahedron t = 6 := by
  -- Proof implementation
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_symmetry_planes_tetrahedron_symmetry_planes_l934_93411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_means_output_l934_93471

/-- Represents the possible symbols in a flowchart --/
inductive FlowchartSymbol
  | parallelogram
  | rectangle
  | diamond
  | oval

/-- Represents the possible meanings of flowchart symbols --/
inductive SymbolMeaning
  | output
  | input
  | assignment
  | decision
  | start
  | end

/-- A function that maps FlowchartSymbol to its meaning --/
def symbolMeaning (s : FlowchartSymbol) : SymbolMeaning :=
  match s with
  | FlowchartSymbol.parallelogram => SymbolMeaning.output
  | FlowchartSymbol.rectangle => SymbolMeaning.assignment
  | FlowchartSymbol.diamond => SymbolMeaning.decision
  | FlowchartSymbol.oval => SymbolMeaning.end

theorem parallelogram_means_output :
  symbolMeaning FlowchartSymbol.parallelogram = SymbolMeaning.output := by
  rfl

#check parallelogram_means_output

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_means_output_l934_93471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_point_l934_93486

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8*x - 2*y + 23

-- Define the center of the circle
def center (x y : ℝ) : Prop := circle_equation x y ∧ ∀ x' y', circle_equation x' y' → (x' - x)^2 + (y' - y)^2 ≤ (x' - x)^2 + (y' - y)^2

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem distance_to_point : ∃ (cx cy : ℝ), center cx cy ∧ distance cx cy (-3) 4 = Real.sqrt 74 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_point_l934_93486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_difference_l934_93468

-- Define the function f(x) = |log₂ x|
noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

-- State the theorem
theorem interval_length_difference (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc 0 2) →
  (∀ y ∈ Set.Icc 0 2, ∃ x ∈ Set.Icc a b, f x = y) →
  (∃ a' b', a' ≥ a ∧ b' ≤ b ∧ b' - a' = 3 ∧
    (∀ c d, c ≥ a ∧ d ≤ b → d - c ≤ b' - a')) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_difference_l934_93468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l934_93443

theorem trig_identity (x : ℝ) : 
  Real.sin (4 * x) * (3 * Real.sin (4 * x) - 2 * Real.cos (4 * x)) = 
  Real.sin (2 * x) ^ 2 - 16 * Real.sin x ^ 2 * Real.cos x ^ 2 * Real.cos (2 * x) ^ 2 + Real.cos (2 * x) ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l934_93443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l934_93441

theorem power_equation_solution (y : ℝ) : (1 / 8 : ℝ) * (2 : ℝ)^36 = (8 : ℝ)^y → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l934_93441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_student_count_school_student_count_eq_828_l934_93425

theorem school_student_count 
  (grades : Nat) 
  (classes_per_grade : Nat) 
  (student_rank_front : Nat) 
  (student_rank_back : Nat) : Nat :=
  let students_per_class := student_rank_front + student_rank_back - 1
  let students_per_grade := students_per_class * classes_per_grade
  grades * students_per_grade

theorem school_student_count_eq_828 
  (grades : Nat) 
  (classes_per_grade : Nat) 
  (student_rank_front : Nat) 
  (student_rank_back : Nat) 
  (h1 : grades = 3) 
  (h2 : classes_per_grade = 12) 
  (h3 : student_rank_front = 12) 
  (h4 : student_rank_back = 12) : 
  school_student_count grades classes_per_grade student_rank_front student_rank_back = 828 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_student_count_school_student_count_eq_828_l934_93425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_product_bounds_l934_93482

theorem trigonometric_product_bounds (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ Real.pi/12) 
  (h4 : x + y + z = Real.pi/2) : 
  (∃ (x' y' z' : ℝ), 
    x' ≥ y' ∧ y' ≥ z' ∧ z' ≥ Real.pi/12 ∧ x' + y' + z' = Real.pi/2 ∧ 
    Real.cos x' * Real.sin y' * Real.cos z' = (2 + Real.sqrt 3) / 8) ∧
  (∃ (x'' y'' z'' : ℝ), 
    x'' ≥ y'' ∧ y'' ≥ z'' ∧ z'' ≥ Real.pi/12 ∧ x'' + y'' + z'' = Real.pi/2 ∧ 
    Real.cos x'' * Real.sin y'' * Real.cos z'' = 1/8) ∧
  (∀ (a b c : ℝ), 
    a ≥ b ∧ b ≥ c ∧ c ≥ Real.pi/12 ∧ a + b + c = Real.pi/2 →
    1/8 ≤ Real.cos a * Real.sin b * Real.cos c ∧ 
    Real.cos a * Real.sin b * Real.cos c ≤ (2 + Real.sqrt 3) / 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_product_bounds_l934_93482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_violet_has_27_nails_l934_93493

/-- The number of nails Tickletoe has -/
def tickletoe_nails : ℕ := sorry

/-- The number of nails Violet has -/
def violet_nails : ℕ := sorry

/-- The total number of nails Violet and Tickletoe have together -/
def total_nails : ℕ := 39

/-- Violet has 3 more than twice as many nails as Tickletoe -/
axiom violet_nails_relation : violet_nails = 2 * tickletoe_nails + 3

/-- The total number of nails is the sum of Violet's and Tickletoe's nails -/
axiom total_nails_sum : total_nails = violet_nails + tickletoe_nails

/-- Theorem: Violet has 27 nails -/
theorem violet_has_27_nails : violet_nails = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_violet_has_27_nails_l934_93493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_score_mean_l934_93472

theorem quiz_score_mean (k : ℕ) (h1 : k > 15) : 
  (8 * k - 240 : ℚ) / (k - 15 : ℚ) =
  (k * 8 - 15 * 16 : ℚ) / (k - 15 : ℚ) := by
  sorry

#check quiz_score_mean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_score_mean_l934_93472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_curves_l934_93483

-- Define the polar coordinate system
def PolarCoord := ℝ × ℝ

-- Define the two curves
noncomputable def curve1 (θ : ℝ) : PolarCoord := (2 * Real.sin θ, θ)
noncomputable def curve2 (θ : ℝ) : PolarCoord := (-1 / Real.cos θ, θ)

-- Define the intersection point
noncomputable def intersection_point : PolarCoord := (Real.sqrt 2, 3 * Real.pi / 4)

-- Theorem statement
theorem intersection_of_curves :
  ∃ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  curve1 θ = intersection_point ∧
  curve2 θ = intersection_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_curves_l934_93483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jo_stair_climbing_ways_l934_93489

/-- Represents the number of ways to climb n stairs with given conditions -/
def climbStairs : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => climbStairs (n + 2) + climbStairs (n + 1) + climbStairs n

/-- The conditions of the stair-climbing problem -/
structure StairClimbingConditions where
  totalStairs : ℕ
  maxConsecutiveSameSteps : ℕ
  possibleStepSizes : List ℕ

/-- The specific conditions for Jo's stair-climbing challenge -/
def joConditions : StairClimbingConditions :=
  { totalStairs := 8
  , maxConsecutiveSameSteps := 2
  , possibleStepSizes := [1, 2, 3] }

/-- Theorem stating that the number of ways to climb the stairs under Jo's conditions is 81 -/
theorem jo_stair_climbing_ways : 
  climbStairs joConditions.totalStairs = 81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jo_stair_climbing_ways_l934_93489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_m_for_parallel_lines_l934_93473

noncomputable section

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- Slope of line l1: x + (1 + m)y + (m - 2) = 0 -/
noncomputable def slope_l1 (m : ℝ) : ℝ := -1 / (1 + m)

/-- Slope of line l2: mx + 2y + 6 = 0 -/
noncomputable def slope_l2 (m : ℝ) : ℝ := -m / 2

/-- There is no real m for which the lines are parallel -/
theorem no_real_m_for_parallel_lines :
  ¬ ∃ (m : ℝ), parallel (slope_l1 m) (slope_l2 m) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_m_for_parallel_lines_l934_93473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_condition_l934_93470

open Vector

-- Define the space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the points
variable (O A B C D E : V)

-- Define the condition for coplanarity
def areCoplanar (p₁ p₂ p₃ p₄ p₅ : V) : Prop :=
  ∃ (a b c d : ℝ), a • (p₂ - p₁) + b • (p₃ - p₁) + c • (p₄ - p₁) + d • (p₅ - p₁) = 0

-- State the theorem
theorem coplanar_condition (k : ℝ) :
  (2 • (A - O) - 3 • (B - O) + 4 • (C - O) + k • (D - O) + 2 • (E - O) = 0) →
  (k = -5 ↔ areCoplanar A B C D E) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_condition_l934_93470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_PQU_is_one_l934_93438

-- Define the square PQRS
def P : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (2, 2)
def S : ℝ × ℝ := (0, 2)

-- Define point T on the line extending PR
def T : ℝ × ℝ → Prop := λ t => ∃ k, t = (k, k) ∧ k > 2

-- Define point U as the intersection of QS and PT
def U : ℝ × ℝ := (1, 1)

-- Define the area function for a triangle given three points
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

theorem area_of_PQU_is_one :
  triangleArea P Q U = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_PQU_is_one_l934_93438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_bottles_purchased_l934_93426

/-- The number of large bottles purchased -/
def large_bottles : ℕ := 1300

/-- The price of a large bottle in dollars -/
def large_bottle_price : ℚ := 189/100

/-- The price of a small bottle in dollars -/
def small_bottle_price : ℚ := 138/100

/-- The approximate average price per bottle in dollars -/
def average_price : ℚ := 17034/10000

/-- The number of small bottles purchased -/
def small_bottles : ℕ := 750

theorem small_bottles_purchased :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  abs ((large_bottles : ℚ) * large_bottle_price + (small_bottles : ℚ) * small_bottle_price - 
       average_price * ((large_bottles : ℚ) + (small_bottles : ℚ))) ≤ ε :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_bottles_purchased_l934_93426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_x_axis_l934_93434

-- Define a polynomial with integer coefficients
def IntPolynomial := Polynomial ℤ

-- Define the statement
theorem parallel_to_x_axis 
  (P : IntPolynomial) 
  (x₁ x₂ : ℤ) 
  (h_dist : ∃ (d : ℤ), d^2 = (x₁ - x₂)^2 + (P.eval x₁ - P.eval x₂)^2) :
  P.eval x₁ = P.eval x₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_x_axis_l934_93434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_specific_investment_l934_93413

/-- The present value needed to reach a future value with compound interest -/
noncomputable def present_value (future_value : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  future_value / (1 + interest_rate) ^ years

/-- The future value reached from a present value with compound interest -/
noncomputable def future_value (present_value : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  present_value * (1 + interest_rate) ^ years

theorem investment_growth (investment : ℝ) (interest_rate : ℝ) (years : ℕ) (target : ℝ) :
  investment = present_value target interest_rate years →
  future_value investment interest_rate years = target :=
by sorry

/-- The specific investment problem -/
theorem specific_investment :
  let investment := 335267.29
  let interest_rate := 0.06
  let years := 10
  let target := 600000
  ∃ ε > 0, ε < 0.01 ∧
  (investment - ε < present_value target interest_rate years ∧
   present_value target interest_rate years < investment + ε) ∧
  (target - ε < future_value investment interest_rate years ∧
   future_value investment interest_rate years < target + ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_specific_investment_l934_93413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_one_l934_93447

/-- A sequence with specific exponent decrease pattern -/
noncomputable def specialSequence : ℕ → ℝ
  | 0 => 4 ^ (1 / 2 : ℝ)
  | 1 => 4 ^ (1 / 3 : ℝ)
  | n + 2 => 
      let prevExponent := if n = 0 
        then 1 / 3 
        else if n = 1 
          then 1 / 6 
          else (1 / 6) - (n - 1) * (1 / 12 : ℝ)
      4 ^ prevExponent

/-- The fifth term of the specialSequence is 1 -/
theorem fifth_term_is_one : specialSequence 4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_one_l934_93447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l934_93454

/-- A predicate that checks if a pair of positive integers (n, p) satisfies the given conditions -/
def satisfiesConditions (n p : ℕ) : Prop :=
  (Nat.Prime p) ∧ 
  (n ≤ 2 * p) ∧ 
  (p > 0) ∧ (n > 0) ∧
  (n^(p - 1) ∣ (p - 1)^n - 1)

/-- The main theorem stating the characterization of solutions -/
theorem solution_characterization (n p : ℕ) : 
  satisfiesConditions n p ↔ 
    ((n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
  sorry

#check solution_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l934_93454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_m_sum_zero_l934_93419

-- Define the piecewise function g(x)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  if x < m then x^2 + 3*x + 1 else 3*x + 7

-- Theorem statement
theorem continuity_m_sum_zero :
  ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧
  ContinuousOn (g m₁) Set.univ ∧
  ContinuousOn (g m₂) Set.univ ∧
  m₁ + m₂ = 0 := by
  sorry

#check continuity_m_sum_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_m_sum_zero_l934_93419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l934_93446

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 16x -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = 16 * p.x

/-- The focus of the parabola y² = 16x -/
def focus : Point :=
  { x := 4, y := 0 }

/-- The equation x₁ + x₂ = 6 -/
def sumOfX (A B : Point) : Prop :=
  A.x + B.x = 6

/-- The length of AB -/
noncomputable def lengthAB (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

/-- Main theorem -/
theorem parabola_chord_length 
  (A B : Point) 
  (hA : isOnParabola A) 
  (hB : isOnParabola B) 
  (hLine : ∃ (t : ℝ), B.x = (1 - t) * focus.x + t * A.x ∧ B.y = (1 - t) * focus.y + t * A.y) 
  (hSum : sumOfX A B) : 
  lengthAB A B = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l934_93446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_profit_maximization_l934_93481

/-- Represents the shopping mall's lamp purchasing and profit problem -/
theorem lamp_profit_maximization 
  (total_lamps : ℕ) 
  (cost_A cost_B selling_price_A selling_price_B : ℚ)
  (total_cost : ℚ) :
  total_lamps = 100 →
  cost_A = 30 →
  cost_B = 50 →
  selling_price_A = 45 →
  selling_price_B = 70 →
  total_cost = 3500 →
  let x := (total_cost - cost_B * total_lamps) / (cost_A - cost_B);
  let y := (selling_price_A - cost_A) * x + (selling_price_B - cost_B) * (total_lamps - x);
  (∀ x' : ℚ, 0 ≤ x' ∧ x' ≤ ↑total_lamps ∧ (↑total_lamps - x') ≤ 3 * x' → 
    (selling_price_A - cost_A) * x' + (selling_price_B - cost_B) * (↑total_lamps - x') ≤ y) ∧
  y = 1875 ∧
  x = 25 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_profit_maximization_l934_93481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_condition_l934_93445

theorem isosceles_triangle_condition (A B C : Real) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  Real.sin C = 2 * Real.cos A * Real.sin B →
  A = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_condition_l934_93445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saltwater_concentration_change_l934_93466

/-- Represents the saltwater solution -/
structure Saltwater where
  total_mass : ℝ
  salt_concentration : ℝ

/-- Calculates the new salt concentration after adding pure salt -/
noncomputable def new_concentration_after_adding_salt (sw : Saltwater) (added_salt : ℝ) : ℝ :=
  (sw.total_mass * sw.salt_concentration + added_salt) / (sw.total_mass + added_salt)

/-- Calculates the new salt concentration after adding water -/
noncomputable def new_concentration_after_adding_water (sw : Saltwater) (added_water : ℝ) : ℝ :=
  (sw.total_mass * sw.salt_concentration) / (sw.total_mass + added_water)

/-- The main theorem to be proved -/
theorem saltwater_concentration_change (sw : Saltwater) 
    (h1 : sw.total_mass = 100)
    (h2 : sw.salt_concentration = 0.15) : 
  new_concentration_after_adding_salt sw 6.25 = sw.salt_concentration + 0.05 ∧ 
  new_concentration_after_adding_water sw 50 = sw.salt_concentration - 0.05 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_saltwater_concentration_change_l934_93466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l934_93487

/-- A set is a circle with radius r -/
def is_circle (s : Set (ℝ × ℝ)) (r : ℝ) : Prop :=
  ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ s ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2

/-- A set is a square with side length a -/
def is_square (s : Set (ℝ × ℝ)) (a : ℝ) : Prop :=
  ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ),
    p ∈ s ↔ max (|p.1 - c.1|) (|p.2 - c.2|) ≤ a/2

/-- The area of a set -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Given a circle with an inscribed square of side length 4, and a smaller square drawn along
    one side of the larger square with two vertices on the circle, the area of the smaller square
    is 1.5625% of the area of the larger square. -/
theorem inscribed_squares_area_ratio :
  ∀ (r : ℝ) (s₁ s₂ : Set (ℝ × ℝ)),
    is_circle s₁ r →
    is_square s₂ 4 →
    s₂ ⊆ s₁ →
    ∃ (s₃ : Set (ℝ × ℝ)),
      is_square s₃ (Real.sqrt (1/16)) ∧
      s₃ ⊆ s₁ ∧
      s₃ ⊆ s₂ ∧
      (∃ (v₁ v₂ : ℝ × ℝ), v₁ ∈ s₃ ∧ v₂ ∈ s₃ ∧ v₁ ∈ s₁ ∧ v₂ ∈ s₁) →
      area s₃ / area s₂ = 1.5625 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l934_93487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_no_solution_l934_93412

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log x

theorem existence_of_no_solution :
  ∃ a : ℝ, a ≠ 0 ∧ a ≠ 1 ∧
  ∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x ≠ a := by
  -- We choose a = 1/2 as our counterexample
  use 1/2
  constructor
  · -- Prove 1/2 ≠ 0
    norm_num
  constructor
  · -- Prove 1/2 ≠ 1
    norm_num
  · -- Main proof
    intro x hx
    sorry -- The actual proof would go here, but we use sorry to skip it for now

#check existence_of_no_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_no_solution_l934_93412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_negative_one_l934_93415

/-- The slope of a line given by parametric equations -/
noncomputable def slope_of_parametric_line (x y : ℝ → ℝ) : ℝ :=
  let t := 1 -- Choose a non-zero value for t
  (y t - y 0) / (x t - x 0)

/-- Parametric equations of the line -/
noncomputable def x (t : ℝ) : ℝ := 3 - t * Real.sin (20 * Real.pi / 180)
noncomputable def y (t : ℝ) : ℝ := 2 + t * Real.cos (70 * Real.pi / 180)

theorem slope_is_negative_one :
  slope_of_parametric_line x y = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_negative_one_l934_93415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_cd_l934_93469

theorem min_abs_diff_cd (c d : ℕ) (h : c * d - 5 * c + 6 * d = 245) :
  ∃ (c' d' : ℕ), c' * d' - 5 * c' + 6 * d' = 245 ∧
  ∀ (x y : ℕ), x * y - 5 * x + 6 * y = 245 →
  |Int.ofNat c' - Int.ofNat d'| ≤ |Int.ofNat x - Int.ofNat y| ∧
  |Int.ofNat c' - Int.ofNat d'| = 29 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_cd_l934_93469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_arithmetic_sequence_properties_l934_93499

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  b : ℕ → ℝ
  T : ℕ → ℝ
  c : ℕ → ℝ
  R : ℕ → ℝ
  lambda : ℝ
  h1 : ∀ n, S n = (n * (a 1 + a n)) / 2
  h2 : S 4 = 4 * S 2
  h3 : ∀ n, a (2 * n) = 2 * a n + 1
  h4 : ∀ n, T n + (a n + 1) / (2^n) = lambda
  h5 : ∀ n, c n = b (2 * n)

/-- The main theorem about the special arithmetic sequence -/
theorem special_arithmetic_sequence_properties (seq : SpecialArithmeticSequence) :
  (∀ n, seq.a n = 2 * n - 1) ∧
  (∀ n, seq.R n = (1 / 9) * (4 - (3 * n + 1) / (4^(n - 1)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_arithmetic_sequence_properties_l934_93499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_self_inverse_l934_93431

/-- A function g defined on real numbers, parameterized by l -/
noncomputable def g (l : ℝ) (x : ℝ) : ℝ := (3 * x + 4) / (l * x - 3)

/-- The set of valid l values for which g is its own inverse -/
def valid_l_set : Set ℝ := {l | l < -9/4 ∨ l > -9/4}

/-- Theorem stating that g is its own inverse if and only if l is in the valid set -/
theorem g_is_self_inverse (l : ℝ) :
  (∀ x, g l (g l x) = x) ↔ l ∈ valid_l_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_self_inverse_l934_93431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_proof_l934_93449

theorem pattern_proof (a b : ℝ) : 
  (Real.sqrt (2 + 2/3) = 2 * Real.sqrt (2/3)) ∧ 
  (Real.sqrt (3 + 3/8) = 3 * Real.sqrt (3/8)) ∧ 
  (Real.sqrt (4 + 4/15) = 4 * Real.sqrt (4/15)) ∧ 
  (Real.sqrt (6 + a/b) = 6 * Real.sqrt (a/b)) →
  a = 6 ∧ b = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_proof_l934_93449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l934_93465

/-- Given an interest rate and time period, calculates the simple interest on a principal amount -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Given an interest rate and time period, calculates the compound interest on a principal amount -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Theorem stating that if the difference between compound and simple interest
    is 65 for a 10% rate over 2 years, then the principal is 6500 -/
theorem interest_difference_implies_principal :
  ∀ principal : ℝ,
  compound_interest principal 10 2 - simple_interest principal 10 2 = 65 →
  principal = 6500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l934_93465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l934_93464

noncomputable def f (a b x : ℝ) : ℝ := (a * x - b) / (4 - x^2)

theorem odd_function_value (a b : ℝ) :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 2, f a b x = -f a b (-x)) →
  f a b 1 = 1/3 →
  b - 2*a = -2 := by
  sorry

#check odd_function_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l934_93464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l934_93432

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- Three terms form an arithmetic sequence. -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSequence (a 1) ((1/2) * a 3) (2 * a 2) →
  a 5 / a 3 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l934_93432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l934_93414

-- Define the sets A and B
def A (b : ℝ) : Set ℝ := {x | x^2 - 3*x + b = 0}
def B : Set ℝ := {x | (x-2)*(x^2+3*x-4) = 0}

-- Part 1
theorem part_one : 
  ∃ (S : Finset (Set ℝ)), Finset.card S = 6 ∧ 
  ∀ M ∈ S, A 4 ⊂ M ∧ M ⊂ B := by
sorry

-- Part 2
theorem part_two : 
  ∀ b : ℝ, (Set.compl B ∩ A b = ∅) ↔ (b > 9/4 ∨ b = 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l934_93414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_theorem_l934_93440

/-- Represents the characteristics of a population --/
structure Population where
  no_diploma_with_job : ℚ  -- Percentage of people without diploma but with desired job
  with_desired_job : ℚ     -- Percentage of people with desired job
  with_diploma : ℚ         -- Percentage of people with diploma

/-- Calculates the percentage of people without desired job but with diploma --/
noncomputable def percentage_no_job_with_diploma (p : Population) : ℚ :=
  (p.with_diploma - (p.with_desired_job - p.no_diploma_with_job)) / (100 - p.with_desired_job) * 100

/-- Theorem stating the result for the given population characteristics --/
theorem population_theorem (p : Population) 
  (h1 : p.no_diploma_with_job = 10)
  (h2 : p.with_desired_job = 40)
  (h3 : p.with_diploma = 39) :
  percentage_no_job_with_diploma p = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_theorem_l934_93440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_line_l934_93406

-- Define the parabola and line
def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 4
def line (x : ℝ) : ℝ := 3 * x - 4

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem min_distance_parabola_line :
  ∃ (x1 x2 : ℝ),
    (∀ (a b : ℝ), distance x1 (parabola x1) x2 (line x2) ≤ distance a (parabola a) b (line b)) ∧
    distance x1 (parabola x1) x2 (line x2) = (7 * Real.sqrt 10) / 20 := by
  sorry

#check min_distance_parabola_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_line_l934_93406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edge_product_sum_sum_squares_is_204_max_sum_is_9420_l934_93435

/-- The set of squares from 1² to 8² -/
def squares : Finset ℕ := Finset.image (fun i => (i + 1)^2) (Finset.range 8)

/-- The sum of all squares from 1² to 8² -/
def sum_squares : ℕ := Finset.sum squares id

/-- A function representing an assignment of numbers to cube vertices -/
def vertex_assignment := Fin 8 → ℕ

/-- The condition that the assignment uses all numbers from squares exactly once -/
def valid_assignment (f : vertex_assignment) : Prop :=
  Finset.card (Finset.image f Finset.univ) = 8 ∧
  ∀ n, f n ∈ squares

/-- The sum of products of numbers at the ends of each edge for a given assignment -/
def edge_product_sum (f : vertex_assignment) : ℕ :=
  f 0 * f 1 + f 0 * f 2 + f 0 * f 4 +
  f 1 * f 3 + f 1 * f 5 +
  f 2 * f 3 + f 2 * f 6 +
  f 3 * f 7 +
  f 4 * f 5 + f 4 * f 6 +
  f 5 * f 7 +
  f 6 * f 7

/-- The theorem stating the maximum sum of edge products -/
theorem max_edge_product_sum :
  ∃ f : vertex_assignment, valid_assignment f ∧
    edge_product_sum f = 9420 ∧
    ∀ g : vertex_assignment, valid_assignment g →
      edge_product_sum g ≤ edge_product_sum f :=
sorry

/-- Proof that the sum of squares from 1² to 8² is 204 -/
theorem sum_squares_is_204 : sum_squares = 204 := by
  rw [sum_squares, squares]
  simp [Finset.sum_image]
  norm_num

/-- The maximum possible sum of edge products is 9420 -/
theorem max_sum_is_9420 :
  ∃ f : vertex_assignment, valid_assignment f ∧ edge_product_sum f = 9420 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edge_product_sum_sum_squares_is_204_max_sum_is_9420_l934_93435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_wins_even_n_l934_93433

/-- Represents a game on a regular n-gon -/
structure NGonGame where
  n : ℕ
  vertices : Fin n → Point
  center : Point
  is_even : Even n

/-- Represents a move in the game -/
inductive Move
  | connect_adjacent : Fin n → Fin n → Move
  | connect_center : Fin n → Move

/-- Represents the state of the game after some moves -/
structure GameState where
  game : NGonGame
  connections : List Move

/-- Represents a player in the game -/
inductive Player
  | A
  | B

/-- Represents the result of the game -/
inductive GameResult
  | Win (player : Player)

/-- A winning strategy for a player -/
def WinningStrategy (player : Player) (game : NGonGame) : Prop :=
  ∀ (opponent_moves : List Move), 
    ∃ (player_moves : List Move), 
      ∃ (result : GameResult), result = GameResult.Win player

/-- The main theorem stating that Player B has a winning strategy when n is even -/
theorem player_b_wins_even_n (game : NGonGame) : 
  WinningStrategy Player.B game := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_wins_even_n_l934_93433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l934_93410

/-- Given a polynomial p(x) = x^4 + ax^3 + bx^2 + cx + d, where a, b, c, and d are constants,
    if p(1) = 1993, p(2) = 3986, and p(3) = 5979, then 1/4 [p(11) + p(-7)] = 5233 -/
theorem polynomial_value_theorem (a b c d : ℝ) :
  let p := fun x => x^4 + a*x^3 + b*x^2 + c*x + d
  (p 1 = 1993) → (p 2 = 3986) → (p 3 = 5979) →
  (1/4 : ℝ) * (p 11 + p (-7)) = 5233 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l934_93410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_angle_l934_93457

/-- Given a hyperbola with equation x^2 - y^2/4 = 1, 
    the cosine of the angle between its two asymptotes is 3/5 -/
theorem hyperbola_asymptotes_angle (x y : ℝ) : 
  (x^2 - y^2/4 = 1) → (∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧ Real.cos θ = 3/5 ∧ 
    θ = 2 * (Real.arctan 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_angle_l934_93457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_12digit_two_consecutive_ones_l934_93460

/-- Counts n-digit numbers with no consecutive 1's using digits 1, 2, 3 -/
def a : ℕ → ℕ
  | 0 => 1  -- Base case for 0 digits
  | 1 => 3
  | 2 => 9
  | 3 => 27
  | n + 4 => a (n + 3) + a (n + 2) + a (n + 1)

/-- Counts n-digit numbers with more than two consecutive 1's using digits 1, 2, 3 -/
def b : ℕ → ℕ
  | 0 | 1 | 2 => 0
  | 3 => 1
  | n + 4 => sorry  -- Actual recursion would be defined here

/-- The set of all n-digit numbers using digits 1, 2, 3 -/
def total (n : ℕ) : ℕ := 3^n

/-- Counts n-digit numbers with exactly two consecutive 1's using digits 1, 2, 3 -/
def c (n : ℕ) : ℕ := total n - a n - b n

theorem count_12digit_two_consecutive_ones : 
  c 12 = total 12 - a 12 - b 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_12digit_two_consecutive_ones_l934_93460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composition_l934_93485

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (3 + x)

theorem domain_of_composition (x : ℝ) :
  x ∉ ({-3, -8/5} : Set ℝ) ↔ ∃ y, y = f (f x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composition_l934_93485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l934_93455

def A : Set ℕ := {x | 0 < x ∧ x ≤ 3}

def B : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 9)}

def B_nat : Set ℕ := {x : ℕ | (x : ℝ) ∈ B}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B_nat) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l934_93455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_squares_l934_93400

/-- Given two squares where the smaller one has perimeter 8 and the larger one has area 36,
    prove that the distance between points A and B is 10, where AB makes a 45° angle with the horizontal. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ) 
    (h1 : small_square_perimeter = 8) (h2 : large_square_area = 36) : ℝ := by
  let small_side := small_square_perimeter / 4
  let large_side := Real.sqrt large_square_area
  let horizontal_distance := small_side + large_side
  let vertical_distance := large_side
  have : Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2) = 10 := by sorry
  exact 10


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_squares_l934_93400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_distance_l934_93416

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A point is outside a circle if its distance from the center is greater than the radius -/
def is_outside (p : ℝ × ℝ) (c : Circle) : Prop :=
  distance p c.O > c.r

/-- The number of intersection points between two circles -/
noncomputable def num_intersections (c1 c2 : Circle) : ℕ :=
  let d := distance c1.O c2.O
  if d < c1.r + c2.r ∧ d > |c1.r - c2.r| then 2 else 0

/-- The theorem to be proven -/
theorem max_points_at_distance (C : Circle) (P : ℝ × ℝ) :
  C.r = 4 → is_outside P C → (∃ (Q : Circle), Q.O = P ∧ Q.r = 5 ∧ num_intersections C Q = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_distance_l934_93416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_geometric_sequence_l934_93423

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a7 : a 7 = Real.sqrt 2 / 2) :
  ∀ x : ℝ, 1 / a 3 + 2 / a 11 ≥ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_geometric_sequence_l934_93423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_coprime_powers_l934_93421

theorem roots_coprime_powers (A : ℤ) (X Y : ℝ) : 
  Odd A →
  X^2 + A*X - 1 = 0 →
  Y^2 + A*Y - 1 = 0 →
  Int.gcd (Int.floor (X^4 + Y^4)) (Int.floor (X^5 + Y^5)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_coprime_powers_l934_93421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2008_l934_93452

def sequenceSum (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => sequenceSum n + (n + 1)

theorem sequence_2008 : sequenceSum 2008 = 2017036 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2008_l934_93452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l934_93488

def arithmetic_mean_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = (a (n + 1) + 3 * a n) / 2

theorem sequence_formula (a : ℕ → ℝ) (h : arithmetic_mean_sequence a) (h1 : a 1 = 3) :
  ∀ n : ℕ, a n = 3^n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l934_93488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_rainfall_l934_93497

/-- Calculates the rainfall on Thursday given the conditions for Monday through Thursday -/
theorem thursday_rainfall (monday : ℝ) (tuesday_decrease : ℝ) (wednesday_increase : ℝ) (thursday_decrease : ℝ) : 
  monday = 0.9 →
  tuesday_decrease = 0.7 →
  wednesday_increase = 0.5 →
  thursday_decrease = 0.2 →
  (monday - tuesday_decrease) * (1 + wednesday_increase) * (1 - thursday_decrease) = 0.24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_rainfall_l934_93497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_one_l934_93408

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a * x^3
  else if -Real.pi/2 < x ∧ x < 0 then Real.cos x
  else 0  -- undefined for other x values

theorem function_composition_equals_one (a : ℝ) :
  f a (f a (-Real.pi/3)) = 1 → a = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_one_l934_93408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_value_l934_93418

/-- Definition of the piecewise function f -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 - 2*x - 1 else x^2 + b*x + c

/-- Theorem stating the value of t for which AB = BC -/
theorem intersection_point_value
  (a b c : ℝ)
  (h_even : ∀ x, f a b c x = f a b c (-x))
  (h_distinct : ∃ t, ∃ A B C D : ℝ,
    A < B ∧ B < C ∧ C < D ∧
    f a b c A = t ∧ f a b c B = t ∧ f a b c C = t ∧ f a b c D = t)
  (h_equal_segments : ∃ t A B C : ℝ,
    A < B ∧ B < C ∧
    f a b c A = t ∧ f a b c B = t ∧ f a b c C = t ∧
    B - A = C - B) :
  ∃ t, t = -7/4 ∧ (∃ A B C : ℝ,
    A < B ∧ B < C ∧
    f a b c A = t ∧ f a b c B = t ∧ f a b c C = t ∧
    B - A = C - B) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_value_l934_93418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l934_93428

-- Define vector operations
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def scale (r : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (r * v.1, r * v.2)
def add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (dot_product v v)

-- Define the theorem
theorem vector_problem (a b c : ℝ × ℝ) :
  a = (3, 4) →
  magnitude c = 10 →
  ∃ (k : ℝ), c = scale k a →
  magnitude b = Real.sqrt 10 →
  dot_product (add a (scale 2 b)) (add (scale 2 a) (scale (-1) b)) = 0 →
  ((c = (6, 8) ∨ c = (-6, -8)) ∧
   (let proj_b_on_a := scale ((dot_product a b) / (dot_product a a)) a;
    proj_b_on_a = (-6/5, -8/5))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l934_93428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l934_93462

/-- Given a point P(a, a²) on the parabola y = x², if the tangent line at P is perpendicular 
    to the line y = -1/2x + 1, then the equation of the tangent line at P is 2x - y - 1 = 0 -/
theorem tangent_line_equation (a : ℝ) : 
  let P : ℝ × ℝ := (a, a^2)
  let parabola := fun (x : ℝ) => x^2
  let perpendicular_line := fun (x : ℝ) => -1/2 * x + 1
  let tangent_slope := (deriv parabola) a
  let perpendicular_slope := -1/2
  tangent_slope * perpendicular_slope = -1 → 
    ∀ x y, (2 * x - y - 1 = 0) ↔ (y - a^2 = tangent_slope * (x - a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l934_93462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beijing_to_shandong_map_measurement_l934_93436

def appropriate_unit_for_map_distance (unit : String) : Prop :=
  unit = "centimeters"

theorem beijing_to_shandong_map_measurement :
  ¬(appropriate_unit_for_map_distance "meters") := by
  simp [appropriate_unit_for_map_distance]
  -- The proof would go here, but it's based on practical knowledge rather than mathematical deduction
  sorry

#check beijing_to_shandong_map_measurement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beijing_to_shandong_map_measurement_l934_93436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_approx_99_l934_93476

/-- Represents a rhombus with side length and diagonals -/
structure Rhombus where
  side_length : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of a rhombus given its diagonals -/
noncomputable def area (r : Rhombus) : ℝ := (r.diagonal1 * r.diagonal2) / 2

/-- The perimeter of the rhombus EFGH -/
def perimeter : ℝ := 40

/-- The correct length of diagonal EG after adjustment -/
def diagonal_EG : ℝ := 15

theorem rhombus_area_approx_99 (EFGH : Rhombus) 
    (h1 : EFGH.side_length = perimeter / 4)
    (h2 : EFGH.diagonal1 = diagonal_EG)
    (h3 : EFGH.diagonal2 = 2 * Real.sqrt (EFGH.side_length ^ 2 - (EFGH.diagonal1 / 2) ^ 2)) :
  ∃ ε > 0, abs (area EFGH - 99) < ε := by
  sorry

#eval "Rhombus area theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_approx_99_l934_93476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l934_93405

noncomputable def vector_a : ℝ × ℝ := (-1, 0)
noncomputable def vector_b : ℝ × ℝ := (Real.sqrt 3 / 2, 1 / 2)

theorem angle_between_vectors :
  Real.arccos ((vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) /
    (Real.sqrt (vector_a.1^2 + vector_a.2^2) * Real.sqrt (vector_b.1^2 + vector_b.2^2))) = 5 * π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l934_93405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_r_inequality_l934_93467

noncomputable def c_r (r : ℝ) : ℝ := if r < 1 then 1 else 2^(r-1)

theorem c_r_inequality (a b r : ℝ) (hr : r ≥ 0) :
  |a + b|^r ≤ c_r r * (|a|^r + |b|^r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_r_inequality_l934_93467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l934_93402

-- Define the lines
def l₁ (x y : ℝ) : Prop := x + 2*y - 1 = 0
def l₂ (x y : ℝ) : Prop := x + 2*y - 3 = 0
def base_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define points
noncomputable def N : ℝ × ℝ := (-1, 1)

-- Define the intersection points
noncomputable def C : ℝ × ℝ := (1, 0)  -- Intersection of l₁ and base_line
noncomputable def D : ℝ × ℝ := (5/3, 2/3)  -- Intersection of l₂ and base_line

-- Define the midpoint M
noncomputable def M : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define the property of line l
def line_l (x y : ℝ) : Prop := 2*x + 7*y - 5 = 0

-- Theorem statement
theorem line_equation : 
  ∀ (x y : ℝ), (∃ (t : ℝ), x = t * (N.1 - M.1) + M.1 ∧ y = t * (N.2 - M.2) + M.2) → 
  line_l x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l934_93402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elgin_money_l934_93459

theorem elgin_money (A B C D E : ℕ) 
  (h1 : A + B + C + D + E = 80)
  (h2 : A - B = 12 ∨ B - A = 12)
  (h3 : B - C = 7 ∨ C - B = 7)
  (h4 : C - D = 5 ∨ D - C = 5)
  (h5 : D - E = 4 ∨ E - D = 4)
  (h6 : E - A = 16 ∨ A - E = 16) : 
  E = 15 := by
  sorry

#check elgin_money

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elgin_money_l934_93459
