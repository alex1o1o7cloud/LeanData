import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l273_27366

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Two lines l₁ and l₂ in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

theorem distance_between_specific_lines :
  let l₁ : Line := ⟨3, 4, -3⟩
  let l₂ : Line := ⟨3, 4, 2⟩
  distance_parallel_lines l₁.A l₁.B l₁.C l₂.C = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l273_27366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inverse_sum_l273_27399

/-- Given a quadratic function f and its inverse f⁻¹, prove that a + c = 0 -/
theorem quadratic_inverse_sum (a b c : ℝ) (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :
  (∀ x, f x = a * x^2 + b * x + c) →
  (∀ x, f_inv x = c * x^2 + b * x + a) →
  (∀ x, f (f_inv x) = x) →
  a + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inverse_sum_l273_27399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_to_plane_are_parallel_l273_27337

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem lines_perpendicular_to_plane_are_parallel (x y : Line) (z : Plane) :
  perpendicular x z → perpendicular y z → parallel x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_to_plane_are_parallel_l273_27337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_truck_tank_radius_l273_27380

/-- Represents a right circular cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder --/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

theorem oil_truck_tank_radius 
  (stationaryTank : Cylinder)
  (oilTruckTank : Cylinder)
  (oilLevelDrop : ℝ)
  (h1 : stationaryTank.radius = 100)
  (h2 : stationaryTank.height = 25)
  (h3 : oilTruckTank.height = 10)
  (h4 : oilLevelDrop = 0.016)
  (h5 : cylinderVolume { radius := stationaryTank.radius, height := oilLevelDrop } = 
        cylinderVolume oilTruckTank) :
  oilTruckTank.radius = 4 := by
  sorry

#check oil_truck_tank_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_truck_tank_radius_l273_27380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_when_a_neg_two_range_when_a_zero_l273_27328

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x - a - 1)

-- Theorem for the domain when a = -2
theorem domain_when_a_neg_two :
  ∀ x : ℝ, f (-2) x ≠ 0 ↔ x ≠ -1 :=
by sorry

-- Theorem for the range when a = 0
theorem range_when_a_zero :
  ∀ y : ℝ, ∃ x : ℝ, f 0 x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_when_a_neg_two_range_when_a_zero_l273_27328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_15th_term_l273_27344

-- Define the sequence terms
noncomputable def term1 (a b : ℝ) : ℝ := Real.log (a^4 * b^9)
noncomputable def term2 (a b : ℝ) : ℝ := Real.log (a^7 * b^15)
noncomputable def term3 (a b : ℝ) : ℝ := Real.log (a^11 * b^20)

-- Define the arithmetic sequence property
def is_arithmetic_sequence (t1 t2 t3 : ℝ) : Prop :=
  t2 - t1 = t3 - t2

-- Define the 15th term
noncomputable def term15 (n : ℝ) (b : ℝ) : ℝ := Real.log (b^n)

-- Theorem statement
theorem arithmetic_sequence_15th_term (a b : ℝ) (n : ℝ) 
  (h1 : a > 0) (h2 : b > 0) :
  is_arithmetic_sequence (term1 a b) (term2 a b) (term3 a b) ∧
  term15 n b = term1 a b + 14 * (term2 a b - term1 a b) →
  n = 139 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_15th_term_l273_27344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l273_27304

noncomputable def empirical_regression (x : ℝ) : ℝ := 8 * x + 11

noncomputable def average_x : ℝ := (2 + 3 + 4 + 5) / 4

def data_points : List (ℝ × ℝ) := [(2, 30), (3, 0), (4, 40), (5, 50)]

theorem find_a_value (a : ℝ) : 
  (empirical_regression average_x = (30 + a + 40 + 50) / 4) → a = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l273_27304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l273_27331

-- Define the train's length in meters
noncomputable def train_length : ℝ := 100

-- Define the train's speed in km/hr
noncomputable def train_speed_kmh : ℝ := 72

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Calculate the train's speed in m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * kmh_to_ms

-- Theorem: The time for the train to cross the electric pole is 5 seconds
theorem train_crossing_time :
  train_length / train_speed_ms = 5 := by
  -- Expand the definitions
  unfold train_length train_speed_ms train_speed_kmh kmh_to_ms
  -- Perform the calculation
  norm_num
  -- The proof is completed automatically
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l273_27331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_process_never_stops_l273_27376

/-- Represents the transformation process on a number -/
def transform (n : ℕ) : ℕ :=
  let a := n / 100
  let b := n % 100
  2 * a + 8 * b

/-- The initial number with 900 ones -/
def initial_number : ℕ := 10^900 - 1

/-- Predicate to check if a number is less than 100 -/
def is_less_than_100 (n : ℕ) : Prop := n < 100

/-- The main theorem stating that the process will not stop -/
theorem process_never_stops :
  ∀ k : ℕ, ¬(is_less_than_100 (Nat.iterate transform k initial_number)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_process_never_stops_l273_27376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_nonnegative_l273_27354

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x < 0}

def B (a : ℝ) : Set ℝ := {-1, -3, a}

theorem a_nonnegative (a : ℝ) (h : (Set.compl A) ∩ B a ≠ ∅) : a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_nonnegative_l273_27354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_rally_gender_ratio_l273_27318

theorem tourist_rally_gender_ratio 
  (a b : ℝ) -- number of participants in first and second trip as real numbers
  (h1 : a > 0) -- ensure first trip has participants
  (h2 : b > 0) -- ensure second trip has participants
  : (0.6 * a + 0.75 * b) ≥ (0.4 * a + 0.25 * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_rally_gender_ratio_l273_27318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_late_to_office_l273_27341

/-- Represents the time in minutes it takes for a man to reach his office when walking at a given pace relative to his usual pace. -/
noncomputable def time_to_office (usual_time_hours : ℝ) (relative_pace : ℝ) : ℝ :=
  (usual_time_hours * 60) / relative_pace

/-- Represents how many minutes late the man arrives when walking at a given pace relative to his usual pace. -/
noncomputable def minutes_late (usual_time_hours : ℝ) (relative_pace : ℝ) : ℝ :=
  time_to_office usual_time_hours relative_pace - (usual_time_hours * 60)

theorem late_to_office (usual_time_hours : ℝ) (h1 : usual_time_hours = 2) :
  minutes_late usual_time_hours (3/4) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_late_to_office_l273_27341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_theorem_l273_27345

/-- Calculates the length of the second train given the speeds of two trains moving in opposite directions, the length of the first train, and the time they take to cross each other. -/
noncomputable def calculate_second_train_length (speed1 speed2 : ℝ) (length1 : ℝ) (cross_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * (5 / 18)
  let total_distance := relative_speed_ms * cross_time
  total_distance - length1 * 1000

/-- The length of the second train is approximately 899.92 meters -/
theorem second_train_length_theorem :
  let speed1 := (210 : ℝ) -- km/hr
  let speed2 := (90 : ℝ) -- km/hr
  let length1 := (1.10 : ℝ) -- km
  let cross_time := (24 : ℝ) -- seconds
  abs (calculate_second_train_length speed1 speed2 length1 cross_time - 899.92) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_theorem_l273_27345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circle_ratio_l273_27379

theorem trapezoid_circle_ratio (α β : Real) (h₁ : 0 < α) (h₂ : α < π)
  (h₃ : 0 < β) (h₄ : β < π) :
  (4 * Real.sin ((α + β) / 2) * Real.cos ((α - β) / 2)) /
  (π * Real.sin α * Real.sin β) =
  4 * Real.sin ((α + β) / 2) * Real.cos ((α - β) / 2) / (π * Real.sin α * Real.sin β) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circle_ratio_l273_27379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_constant_l273_27356

-- Define the curve
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 2 * Real.sqrt 3 * Real.sin θ)

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem sum_of_distances_constant (θ : ℝ) :
  let P := curve θ
  distance P A + distance P B = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_constant_l273_27356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_2_value_l273_27315

noncomputable def f (a : ℝ) (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a^(x + 1) - 2 else g x

theorem f_g_2_value (a : ℝ) (g : ℝ → ℝ) :
  a > 0 ∧ a ≠ 1 ∧
  (∀ x, f a g (-x) = -(f a g x)) →
  f a g (g 2) = Real.sqrt 2 / 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_2_value_l273_27315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_double_angle_from_sine_series_l273_27391

theorem cosine_double_angle_from_sine_series (φ : ℝ) :
  (∑' n, Real.sin φ ^ (2 * n)) = 4 → Real.cos (2 * φ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_double_angle_from_sine_series_l273_27391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_unit_circle_l273_27392

theorem line_tangent_to_unit_circle (α : ℝ) :
  ∃ (x y : ℝ), x * Real.sin α + y * Real.cos α = 1 ∧ x^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_unit_circle_l273_27392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l273_27369

/-- The number of people available for selection -/
def num_people : ℕ := 5

/-- The number of positions to be filled -/
def num_positions : ℕ := 4

/-- The number of arrangements when A is selected -/
def arrangements_with_A : ℕ := 3 * (3 * 2 * 1)

/-- The number of arrangements when A is not selected -/
def arrangements_without_A : ℕ := 4 * 3 * 2 * 1

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := arrangements_with_A + arrangements_without_A

theorem arrangement_count :
  total_arrangements = 42 := by
  unfold total_arrangements arrangements_with_A arrangements_without_A
  norm_num

#eval total_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l273_27369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_needed_for_new_solution_l273_27348

noncomputable section

-- Define the initial amounts of chemicals and water
def chemical_a : ℝ := 0.08
def initial_water : ℝ := 0.04
def chemical_b : ℝ := 0.02

-- Define the target amount of the new solution
def target_amount : ℝ := 0.84

-- Define the total amount of the initial mixture
def initial_total : ℝ := chemical_a + initial_water + chemical_b

-- Define the fraction of water in the initial mixture
def water_fraction : ℝ := initial_water / initial_total

-- Theorem to prove
theorem water_needed_for_new_solution :
  water_fraction * target_amount = 0.24 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_needed_for_new_solution_l273_27348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_are_complementary_l273_27327

/-- Represents the composition of a group --/
structure MyGroup where
  boys : ℕ
  girls : ℕ

/-- Represents an event in the sample space --/
inductive Event
  | AtLeastOneBoy
  | AllGirls

/-- Defines the concept of complementary events --/
def complementary (e1 e2 : Event) : Prop :=
  (e1 = Event.AtLeastOneBoy ∧ e2 = Event.AllGirls) ∨
  (e1 = Event.AllGirls ∧ e2 = Event.AtLeastOneBoy)

/-- The main theorem to be proved --/
theorem events_are_complementary (g : MyGroup) (h1 : g.boys = 3) (h2 : g.girls = 2) :
  complementary Event.AtLeastOneBoy Event.AllGirls := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_are_complementary_l273_27327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_cut_length_no_shorter_cut_l273_27333

/-- An isosceles right triangle with legs of length √2 -/
structure IsoscelesRightTriangle where
  side : ℝ
  is_sqrt_two : side = Real.sqrt 2

/-- A linear cut that divides the area of the triangle into two equal parts -/
structure LinearCut (t : IsoscelesRightTriangle) where
  length : ℝ
  divides_equally : Prop  -- Changed from True to Prop

/-- The shortest linear cut that divides the area of the triangle into two equal parts -/
noncomputable def shortest_cut (t : IsoscelesRightTriangle) : LinearCut t :=
  { length := Real.sqrt (2 * Real.sqrt 2 - 2),
    divides_equally := True }  -- Now True is of type Prop

/-- Theorem stating that the shortest cut has length √(2√2 - 2) -/
theorem shortest_cut_length (t : IsoscelesRightTriangle) :
  (shortest_cut t).length = Real.sqrt (2 * Real.sqrt 2 - 2) :=
by sorry

/-- Theorem stating that no shorter cut exists -/
theorem no_shorter_cut (t : IsoscelesRightTriangle) (c : LinearCut t) :
  c.length ≥ (shortest_cut t).length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_cut_length_no_shorter_cut_l273_27333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_max_l273_27358

open Real

theorem triangle_sine_sum_max (A B C : ℝ) (h : A + B + C = π) :
  let T := Real.sin A + Real.sin B + Real.sin C + Real.sin (π / 3)
  ∃ (max : ℝ), T ≤ max ∧ max = 2 * Real.sqrt 3 ∧
  (T = max ↔ A = π / 3 ∧ B = π / 3 ∧ C = π / 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_max_l273_27358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_a_b_c_l273_27326

-- Define the constants
noncomputable def a : ℝ := Real.exp (Real.log 3 * Real.log 3)
noncomputable def b : ℝ := 3 + 3 * Real.log 3
noncomputable def c : ℝ := (Real.log 3) ^ 3

-- State the theorem
theorem relationship_between_a_b_c : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_a_b_c_l273_27326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_squared_l273_27353

-- Define the trapezoid ABCD
structure Trapezoid (A B C D : ℝ × ℝ) where
  is_trapezoid : True  -- Placeholder for the trapezoid condition
  bc_perp_ab : True    -- Placeholder for BC perpendicular to AB
  bc_perp_cd : True    -- Placeholder for BC perpendicular to CD
  diagonals_perp : True  -- Placeholder for perpendicular diagonals
  ab_length : ‖A - B‖ = Real.sqrt 11
  ad_length : ‖A - D‖ = Real.sqrt 1001

-- Theorem statement
theorem trapezoid_bc_squared (A B C D : ℝ × ℝ) (trap : Trapezoid A B C D) : 
  ‖B - C‖^2 = 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_squared_l273_27353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_chord_length_l273_27359

/-- The circle equation x^2 + y^2 + 4x - 6y + 4 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

/-- The line equation mx - y + 1 = 0 -/
def line_equation (m x y : ℝ) : Prop := m*x - y + 1 = 0

/-- The length of the chord MN -/
noncomputable def chord_length (m : ℝ) : ℝ := sorry

/-- Theorem stating that the chord length is minimal when m = 1 -/
theorem minimal_chord_length :
  ∀ m : ℝ, chord_length m ≥ chord_length 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_chord_length_l273_27359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l273_27390

/-- A quadratic function f(x) = ax^2 + bx + c with a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h1 : QuadraticFunction a b c (-5) = -2.79)
  (h2 : QuadraticFunction a b c 1 = -2.79)
  (h3 : QuadraticFunction a b c 2 = 0) :
  ∀ x, QuadraticFunction a b c x < 0 ↔ -6 < x ∧ x < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l273_27390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_implies_a_equals_one_l273_27322

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem increasing_interval_implies_a_equals_one :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ < f a x₂) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_implies_a_equals_one_l273_27322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mushrooms_count_l273_27305

noncomputable def potatoes : ℝ := 3
noncomputable def carrots : ℝ := 5.5 * potatoes
noncomputable def onions : ℝ := 1.75 * carrots
noncomputable def green_beans : ℝ := (2/3) * onions
noncomputable def bell_peppers : ℝ := 4.5 * green_beans
noncomputable def mushrooms : ℝ := 3.25 * bell_peppers

theorem mushrooms_count : Int.ceil mushrooms = 282 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mushrooms_count_l273_27305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unitPolylineSetIsSquare_sumPolylineSetIsHexagon_diffPolylineSetIsParallelLines_l273_27357

-- Define the polyline distance function
def polylineDistance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define the set of points with polyline distance 1 from origin
def unitPolylineSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| = 1}

-- Define the set of points with sum of polyline distances to (-1, 0) and (1, 0) equal to 4
def sumPolylineSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 + 1| + |p.2| + |p.1 - 1| + |p.2| = 4}

-- Define the set of points with absolute difference of polyline distances to (-1, 0) and (1, 0) equal to 1
def diffPolylineSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |(|p.1 + 1| + |p.2|) - (|p.1 - 1| + |p.2|)| = 1}

-- Theorem statements
theorem unitPolylineSetIsSquare : ∃ (s : Set (ℝ × ℝ)), s = unitPolylineSet := sorry

theorem sumPolylineSetIsHexagon : 
  ∃ (h : Set (ℝ × ℝ)), h = sumPolylineSet := sorry

theorem diffPolylineSetIsParallelLines :
  ∃ (l₁ l₂ : Set (ℝ × ℝ)), diffPolylineSet = l₁ ∪ l₂ := sorry

-- Note: The predicates IsSquare, IsHexagon, IsLine, AreParallel, and Area are removed as they are not defined in the standard library

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unitPolylineSetIsSquare_sumPolylineSetIsHexagon_diffPolylineSetIsParallelLines_l273_27357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l273_27349

def set_A : Set ℝ := {x | ∃ n : ℤ, n ∈ ({-1, 0, 1, 2} : Set ℤ) ∧ x = Real.exp (n : ℝ)}

def set_B : Set ℝ := {x | x ≤ 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {Real.exp (-1), 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l273_27349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l273_27393

noncomputable section

/-- Production cost function -/
def cost (x : ℝ) : ℝ := x^2 / 5 - 48 * x + 8000

/-- Average cost per ton -/
def avg_cost (x : ℝ) : ℝ := cost x / x

/-- Profit function -/
def profit (x : ℝ) : ℝ := 40 * x - cost x

/-- Maximum annual output -/
def max_output : ℝ := 210

theorem production_optimization :
  ∃ (x_min_avg_cost x_max_profit : ℝ),
    x_min_avg_cost > 0 ∧ 
    x_min_avg_cost ≤ max_output ∧
    x_max_profit > 0 ∧ 
    x_max_profit ≤ max_output ∧
    (∀ x, x > 0 → x ≤ max_output → avg_cost x_min_avg_cost ≤ avg_cost x) ∧
    avg_cost x_min_avg_cost = 32 ∧
    x_min_avg_cost = 200 ∧
    (∀ x, x > 0 → x ≤ max_output → profit x ≤ profit x_max_profit) ∧
    profit x_max_profit = 1660 ∧
    x_max_profit = max_output := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l273_27393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_conditions_l273_27335

/-- Represents the properties of a cylindrical water tank -/
structure WaterTank where
  r : ℝ  -- radius
  h : ℝ  -- height
  cost_side : ℝ  -- cost per square meter of side surface
  cost_base : ℝ  -- cost per square meter of base
  total_cost : ℝ  -- total construction cost

/-- The volume of the water tank as a function of radius -/
noncomputable def volume (tank : WaterTank) : ℝ → ℝ :=
  fun r => (Real.pi / 5) * (300 * r - 4 * r^3)

/-- The theorem stating the conditions for maximum volume -/
theorem max_volume_conditions (tank : WaterTank) :
  tank.cost_side = 100 ∧
  tank.cost_base = 160 ∧
  tank.total_cost = 12000 * Real.pi ∧
  tank.r > 0 ∧
  tank.h > 0 ∧
  200 * Real.pi * tank.r * tank.h + 160 * Real.pi * tank.r^2 = tank.total_cost →
  (∃ (r_max h_max : ℝ), r_max = 5 ∧ h_max = 8 ∧
    ∀ r, 0 < r ∧ r < 5 * Real.sqrt 3 →
      volume tank r ≤ volume tank r_max) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_conditions_l273_27335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_locus_characterization_l273_27385

/-- Represents a rhombus with diagonals 2e and 2f -/
structure Rhombus (e f : ℝ) where
  e_nonneg : e ≥ 0
  f_nonneg : f ≥ 0

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The locus of points satisfying the given condition for a rhombus -/
def rhombusLocus (r : Rhombus e f) : Set Point :=
  {p : Point | (p.x - e)^2 + p.y^2 = (p.x^2 + (p.y - f)^2) + ((p.x + e)^2 + p.y^2) + (p.x^2 + (p.y + f)^2)}

/-- The set of two circles centered at A(e, 0) and C(-e, 0) with radius sqrt(e^2 - f^2) -/
def twoCircles (e f : ℝ) : Set Point :=
  {p : Point | (p.x - e)^2 + p.y^2 = e^2 - f^2 ∨ (p.x + e)^2 + p.y^2 = e^2 - f^2}

/-- The four vertices of the rhombus -/
def rhombusVertices (e f : ℝ) : Set Point :=
  {⟨e, 0⟩, ⟨0, f⟩, ⟨-e, 0⟩, ⟨0, -f⟩}

theorem rhombus_locus_characterization (r : Rhombus e f) :
  rhombusLocus r =
    if e > f then twoCircles e f
    else if e = f then rhombusVertices e f
    else ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_locus_characterization_l273_27385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_intersect_A_l273_27367

-- Define R as the real numbers
def R : Type := ℝ

-- Define A and B as subsets of ℝ
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 6}

def B : Set ℝ := {x | x > 5}

-- State the theorem
theorem complement_B_intersect_A :
  (Set.univ \ B) ∩ A = {x : ℝ | 0 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_intersect_A_l273_27367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_l273_27363

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 16 / (x - 1)

-- Define the domain of f
def domain (x : ℝ) : Prop := x > 1

-- Statement of the theorem
theorem min_slope_tangent (x₀ : ℝ) (h : domain x₀) :
  ∃ (k : ℝ), (∀ x, domain x → deriv f x ≥ k) ∧ (deriv f x₀ = k) ∧ (k = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_l273_27363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_l273_27336

-- Define the angle in degrees
noncomputable def angle : ℝ := 2008

-- Define the point P
noncomputable def P : ℝ × ℝ := (Real.cos (angle * Real.pi / 180), Real.sin (angle * Real.pi / 180))

-- Theorem statement
theorem point_in_third_quadrant : P.1 < 0 ∧ P.2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_l273_27336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l273_27382

-- Define f as a noncomputable function from ℝ to ℝ
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(3x+2)
def domain_f_3x_plus_2 : Set ℝ := Set.Ioo 0 1

-- Define the domain of f(2x-1)
def domain_f_2x_minus_1 : Set ℝ := Set.Ioo (3/2) 3

-- Theorem statement
theorem domain_equivalence :
  (∀ x ∈ domain_f_3x_plus_2, ∃ y, f (3*x + 2) = y) →
  (∀ x ∈ domain_f_2x_minus_1, ∃ y, f (2*x - 1) = y) :=
by
  intro h
  intro x hx
  -- The proof is omitted using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l273_27382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_discount_theorem_l273_27373

/-- Calculates the total discount for a shopping trip -/
noncomputable def calculate_total_discount (tshirt_price : ℝ) (backpack_price : ℝ) (cap_price : ℝ) (jeans_price : ℝ) (sneakers_price_usd : ℝ)
                             (tshirt_discount : ℝ) (backpack_discount : ℝ) (cap_discount : ℝ) (jeans_discount : ℝ) (sneakers_discount : ℝ)
                             (exchange_rate : ℝ) : ℝ × ℝ :=
  let tshirt_disc := tshirt_price * tshirt_discount
  let backpack_disc := backpack_price * backpack_discount
  let cap_disc := cap_price * cap_discount
  let jeans_disc := jeans_price * jeans_discount
  let sneakers_disc_usd := sneakers_price_usd * sneakers_discount
  let sneakers_disc_eur := sneakers_disc_usd / exchange_rate
  let total_disc_eur := tshirt_disc + backpack_disc + cap_disc + jeans_disc + sneakers_disc_eur
  let total_disc_usd := total_disc_eur * exchange_rate
  (total_disc_eur, total_disc_usd)

theorem shopping_discount_theorem :
  let result := calculate_total_discount 30 10 5 50 60 0.1 0.2 0.15 0.25 0.3 1.2
  result.1 = 33.25 ∧ result.2 = 39.90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_discount_theorem_l273_27373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_vs_B_l273_27306

/-- The price of Company KW -/
def price_KW : ℝ := sorry

/-- The assets of Company A -/
def assets_A : ℝ := sorry

/-- The assets of Company B -/
def assets_B : ℝ := sorry

/-- The price of Company KW is 30% more than Company A's assets -/
axiom price_vs_A : price_KW = 1.30 * assets_A

/-- The price of Company KW is 78.78787878787878% of the combined assets of A and B -/
axiom price_vs_combined : price_KW = 0.7878787878787878 * (assets_A + assets_B)

/-- The price of Company KW is approximately 100% more than Company B's assets -/
theorem price_vs_B : ∃ (ε : ℝ), abs (price_KW / assets_B - 2) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_vs_B_l273_27306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l273_27381

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

theorem function_properties :
  let C := {(x, y) | y = f x}
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (11 * Real.pi / 6 - x, y) ∈ C) ∧ 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (4 * Real.pi / 3 - x, -y) ∈ C) ∧
  (∀ (x₁ x₂ : ℝ), -Real.pi / 12 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5 * Real.pi / 12 → f x₁ < f x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l273_27381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_prob_difference_l273_27370

-- Define a fair coin
noncomputable def fairCoin : ℝ := 1 / 2

-- Define the number of flips
def numFlips : ℕ := 5

-- Define the binomial probability function
noncomputable def binomialProb (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem fair_coin_prob_difference : 
  |binomialProb numFlips 4 fairCoin - binomialProb numFlips 5 fairCoin| = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_prob_difference_l273_27370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_display_difference_l273_27330

def triangular_display (d : ℕ) : Prop :=
  let row_cans := λ n : ℕ => 19 + d * (n - 7)
  (∀ n ∈ Finset.range 9, row_cans n < row_cans (n + 1)) ∧
  (row_cans 7 = 19) ∧
  (Finset.sum (Finset.range 10) row_cans < 150)

theorem can_display_difference : ∃ d : ℕ, triangular_display d ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_display_difference_l273_27330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_value_l273_27313

def airfare (x : ℕ) : ℚ :=
  if x ≤ 35 then 800 else -10 * x + 1150

def profit (x : ℕ) : ℚ :=
  airfare x * x - 18000

def is_valid_group_size (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 60

theorem max_profit_value :
  ∃ x : ℕ, is_valid_group_size x ∧
    profit x = 15060 ∧
    ∀ y : ℕ, is_valid_group_size y → profit y ≤ profit x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_value_l273_27313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_division_theorem_l273_27375

/-- A convex polygon with n sides divided into m triangles -/
structure ConvexPolygonDivision where
  n : ℕ  -- number of sides of the polygon
  m : ℕ  -- number of triangles
  is_convex : n ≥ 3  -- convex polygon has at least 3 sides
  valid_division : m ≥ 1  -- at least one triangle in the division

/-- Properties of the polygon division -/
def ConvexPolygonDivision.properties (p : ConvexPolygonDivision) : Prop :=
  -- m + n is even
  Even (p.m + p.n) ∧
  -- Number of distinct interior sides
  ∃ k : ℕ, k = (3 * p.m - p.n) / 2 ∧
  -- Number of distinct interior vertices
  ∃ v : ℕ, v = (p.m - p.n + 2) / 2

/-- Main theorem about convex polygon division -/
theorem convex_polygon_division_theorem (p : ConvexPolygonDivision) :
  p.properties := by
  sorry

#check convex_polygon_division_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_division_theorem_l273_27375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_is_two_l273_27347

/-- Represents the average price and relative quantity of eggs sold in a month -/
structure MonthData where
  price : ℝ
  quantity : ℝ

/-- Calculates the average price per dozen eggs over a 3-month period -/
noncomputable def averagePrice (april : MonthData) (may : MonthData) (june : MonthData) : ℝ :=
  let totalRevenue := april.price * april.quantity + may.price * may.quantity + june.price * june.quantity
  let totalQuantity := april.quantity + may.quantity + june.quantity
  totalRevenue / totalQuantity

/-- Theorem stating that under given conditions, the average price is $2.00 -/
theorem average_price_is_two :
  ∀ (x : ℝ),
    let april : MonthData := { price := 1.2, quantity := 2/3 * x }
    let may : MonthData := { price := 1.2, quantity := x }
    let june : MonthData := { price := 3.0, quantity := 4/3 * x }
    averagePrice april may june = 2 := by
  intro x
  simp [averagePrice]
  -- The actual proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_is_two_l273_27347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l273_27398

theorem floor_ceil_fraction_square : ⌊⌈((11 : ℚ) / 5)^2⌉ * ((19 : ℚ) / 3)⌋ = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l273_27398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcade_tickets_correct_l273_27320

def arcade_tickets (whack_a_mole_tickets : ℕ) (candy_cost : ℕ) (candies_bought : ℕ) : ℕ :=
  let total_tickets := candy_cost * candies_bought
  total_tickets - whack_a_mole_tickets

theorem arcade_tickets_correct (whack_a_mole_tickets candy_cost candies_bought : ℕ) :
  arcade_tickets whack_a_mole_tickets candy_cost candies_bought =
  candy_cost * candies_bought - whack_a_mole_tickets := by
  rfl

#eval arcade_tickets 26 9 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcade_tickets_correct_l273_27320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrangular_pyramid_volume_l273_27365

/-- A pyramid with a quadrangular base -/
structure QuadrangularPyramid where
  baseArea : ℝ
  height : ℝ

/-- The volume of a quadrangular pyramid -/
noncomputable def volume (p : QuadrangularPyramid) : ℝ := (1 / 3) * p.baseArea * p.height

/-- Theorem: The volume of a quadrangular pyramid is one-third of the product of its base area and height -/
theorem quadrangular_pyramid_volume (p : QuadrangularPyramid) :
  volume p = (1 / 3) * p.baseArea * p.height := by
  -- Unfold the definition of volume
  unfold volume
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrangular_pyramid_volume_l273_27365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_eq_cos_l273_27361

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => Real.cos
  | (n + 1) => λ x => deriv (f n) x

-- State the theorem
theorem f_2008_eq_cos : f 2008 = f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_eq_cos_l273_27361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_right_triangle_l273_27388

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given conditions -/
def F : Point := ⟨0, 1⟩
def l : Line := ⟨0, 1, 1⟩  -- y = -1
def l₁ (m : ℝ) : Line := ⟨0, 1, m⟩  -- y = -m

/-- Perpendicular foot of P on line l -/
def Q (P : Point) : Point := ⟨P.x, -1⟩

/-- Vector from point A to point B -/
def vec (A B : Point) : Point := ⟨B.x - A.x, B.y - A.y⟩

/-- Dot product of two vectors -/
def dot (v w : Point) : ℝ := v.x * w.x + v.y * w.y

/-- Main theorem -/
theorem trajectory_and_right_triangle (m : ℝ) (hm : m > 2) :
  (∀ P : Point, dot (vec (Q P) P) (vec (Q P) F) = dot (vec F P) (vec F (Q P)) → P.x^2 = 4 * P.y) ∧
  (∃ (a b : ℝ), a ≠ b ∧ 
    let M : Point := ⟨a, -m⟩
    let A : Point := ⟨2 * Real.sqrt m, m⟩
    let B : Point := ⟨-2 * Real.sqrt m, m⟩
    (A.x^2 = 4 * A.y ∧ B.x^2 = 4 * B.y) ∧  -- A and B on trajectory
    (vec M A).x * (vec A B).x + (vec M A).y * (vec A B).y = 0)  -- MAB is right-angled
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_right_triangle_l273_27388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_magnitude_equals_one_l273_27374

def vector_a : Fin 2 → ℝ := ![1, 0]
def vector_b : Fin 2 → ℝ := ![1, 1]

noncomputable def cross_product_magnitude (a b : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((a 0 * b 1 - a 1 * b 0) ^ 2)

theorem cross_product_magnitude_equals_one :
  cross_product_magnitude vector_a vector_b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_magnitude_equals_one_l273_27374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_l273_27301

/-- Given a triangle ABC with specific point conditions, prove that the sum of certain ratios is 1129/345 -/
theorem triangle_ratio_sum (A B C D E F : ℝ × ℝ) : 
  -- D is the midpoint of BC
  D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  -- E divides AC in the ratio 2:3
  E = ((2 * C.1 + 3 * A.1) / 5, (2 * C.2 + 3 * A.2) / 5) →
  -- F is on AD such that AF:FD = 3:2
  F = ((3 * D.1 + 2 * A.1) / 5, (3 * D.2 + 2 * A.2) / 5) →
  -- The sum of EF/FB and BF/FD equals 1129/345
  (dist E F / dist F B) + (dist B F / dist F D) = 1129 / 345 := by
    sorry

/-- Helper function to calculate Euclidean distance between two points -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_l273_27301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_27_point_5_l273_27329

/-- Arithmetic sequence with 15 terms, first term 5, and last term 50 -/
noncomputable def arithmetic_sequence (n : ℕ) : ℝ :=
  let d := (50 - 5) / (15 - 1)
  5 + (n - 1) * d

theorem eighth_term_is_27_point_5 :
  arithmetic_sequence 8 = 27.5 := by
  -- Unfold the definition of arithmetic_sequence
  unfold arithmetic_sequence
  -- Simplify the expression
  simp [Nat.cast_sub, Nat.cast_one]
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_27_point_5_l273_27329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_axis_symmetry_implies_g_value_l273_27386

noncomputable section

-- Define the functions f and g
def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)

def g (φ : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x + φ)

-- State the theorem
theorem same_axis_symmetry_implies_g_value (ω φ : ℝ) 
  (h₁ : ω > 0) 
  (h₂ : 0 < φ ∧ φ < Real.pi) 
  (h₃ : ∀ (k : ℤ), ∃ (m : ℤ), (k * Real.pi + Real.pi / 2) / ω + Real.pi / (3 * ω) = m * Real.pi / 2 - φ / 2) :
  g φ (Real.pi / 3) = -Real.sqrt 3 / 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_axis_symmetry_implies_g_value_l273_27386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_quadratic_coefficients_l273_27339

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

theorem intersection_and_quadratic_coefficients :
  (∃ (a b : ℝ), ∀ x, x ∈ A ∩ B ↔ x^2 + a*x + b < 0) ∧
  A ∩ B = Set.Ioo (-1) 2 ∧
  (∀ x, x ∈ A ∩ B ↔ x^2 - x - 2 < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_quadratic_coefficients_l273_27339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_18pi_l273_27362

/-- A regular heptagon with side length 4 -/
structure RegularHeptagon where
  side_length : ℝ
  is_regular : side_length = 4

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circular arc centered at a vertex of the heptagon -/
structure CircularArc (h : RegularHeptagon) where
  center : Point
  endpoints : Point × Point

/-- The total shaded area formed by the circular arcs -/
noncomputable def total_shaded_area (h : RegularHeptagon) (arcs : List (CircularArc h)) : ℝ :=
  sorry

theorem shaded_area_is_18pi (h : RegularHeptagon) (arcs : List (CircularArc h)) 
  (hcount : arcs.length = 7) :
  total_shaded_area h arcs = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_18pi_l273_27362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_on_line_l273_27342

/-- The complex number z as a function of real m -/
noncomputable def z (m : ℝ) : ℂ := (m * (m - 1)) / (m + 1) + (m^2 + 2*m - 3) * Complex.I

/-- Theorem stating the conditions for z to be a pure imaginary number -/
theorem pure_imaginary (m : ℝ) : z m = (z m).im * Complex.I ↔ m = 0 := by sorry

/-- Theorem stating the conditions for z to be on the line x + y + 3 = 0 -/
theorem on_line (m : ℝ) : (z m).re + (z m).im + 3 = 0 ↔ 
  m = 0 ∨ m = -2 + Real.sqrt 3 ∨ m = -2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_on_line_l273_27342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julias_house_paintable_area_l273_27397

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total paintable area in Julia's house -/
def totalPaintableArea (
  numRooms : ℕ)
  (dimensions : RoomDimensions)
  (unpaintableAreaPerRoom : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableAreaPerRoom := wallArea - unpaintableAreaPerRoom
  (numRooms : ℝ) * paintableAreaPerRoom

/-- Theorem stating that the total paintable area in Julia's house is 1520 square feet -/
theorem julias_house_paintable_area :
  totalPaintableArea 4 ⟨14, 11, 9⟩ 70 = 1520 := by
  sorry

#eval totalPaintableArea 4 ⟨14, 11, 9⟩ 70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julias_house_paintable_area_l273_27397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_functions_satisfy_equation_l273_27371

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 2 * x^2 * y^2

/-- The theorem stating that there are exactly two functions satisfying the equation -/
theorem exactly_two_functions_satisfy_equation :
  ∃! (s : Set (ℝ → ℝ)), 
    (∀ f, f ∈ s → SatisfiesFunctionalEquation f) ∧ 
    (∃ f₁ f₂, f₁ ∈ s ∧ f₂ ∈ s ∧ f₁ ≠ f₂) ∧
    (∀ f, f ∈ s → f = (λ x ↦ x^2 / Real.sqrt 2) ∨ f = (λ x ↦ -x^2 / Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_functions_satisfy_equation_l273_27371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l273_27338

-- Define the constants
noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi
noncomputable def c : ℝ := Real.log 0.5 / Real.log 2

-- State the theorem
theorem a_gt_b_gt_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l273_27338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_solution_set_f_t_range_l273_27355

/-- An odd function f: ℝ → ℝ defined on [-1,1] with f(1) = 1 -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f is defined on [-1,1] -/
axiom f_domain : ∀ x, x ∈ Set.Icc (-1) 1 → f x ≠ 0

/-- f(1) = 1 -/
axiom f_one : f 1 = 1

theorem f_increasing : 
  ∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x < y → f x < f y := by
  sorry

theorem f_solution_set : 
  {x | 0 ≤ x ∧ x < 1/4} = {x | f (x + 1/2) < f (1 - x)} := by
  sorry

theorem f_t_range : 
  {t : ℝ | ∀ x ∈ Set.Icc (-1) 1, ∀ α ∈ Set.Icc (-Real.pi/3) (Real.pi/4), 
    f x ≤ t^2 + t - 1 / (Real.cos α)^2 - 2 * Real.tan α - 1} = 
  Set.Iic (-3) ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_solution_set_f_t_range_l273_27355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_equals_30_l273_27316

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle with width w and length l -/
structure Rectangle where
  w : ℝ
  l : ℝ

/-- Area of a triangle using Heron's formula -/
noncomputable def triangleArea (t : Triangle) : ℝ := 
  let s := (t.a + t.b + t.c) / 2
  (s * (s - t.a) * (s - t.b) * (s - t.c))^(1/2)

/-- Area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.w * r.l

/-- Perimeter of a rectangle -/
def rectanglePerimeter (r : Rectangle) : ℝ := 2 * (r.w + r.l)

theorem rectangle_perimeter_equals_30 (t : Triangle) (r : Rectangle) 
    (h1 : t.a = 9 ∧ t.b = 12 ∧ t.c = 15) 
    (h2 : r.w = 6) 
    (h3 : rectangleArea r = triangleArea t) : 
  rectanglePerimeter r = 30 := by
  sorry

#eval rectanglePerimeter { w := 6, l := 9 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_equals_30_l273_27316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lattice_points_in_T_l273_27396

/-- The region T in the xy-plane -/
def T : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 ≤ 48}

/-- The number of lattice points in T -/
def lattice_points_in_T : ℕ :=
  Finset.card (Finset.filter (fun p => p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 ≤ 48) (Finset.product (Finset.range 49) (Finset.range 49)))

/-- Theorem stating that the number of lattice points in T is 202 -/
theorem count_lattice_points_in_T : lattice_points_in_T = 202 := by
  sorry

#eval lattice_points_in_T

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lattice_points_in_T_l273_27396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_theorem_l273_27334

/-- Represents a ferry crossing a river -/
structure Ferry where
  speed : ℝ

/-- Represents a river crossing scenario -/
structure RiverCrossing where
  ferry1 : Ferry
  ferry2 : Ferry
  first_meeting_distance : ℝ
  second_meeting_distance : ℝ

/-- The width of the river given a river crossing scenario -/
def river_width (rc : RiverCrossing) : ℝ :=
  sorry

theorem river_width_theorem (rc : RiverCrossing) 
  (h1 : rc.ferry1.speed ≠ rc.ferry2.speed)
  (h2 : rc.ferry1.speed > 0)
  (h3 : rc.ferry2.speed > 0)
  (h4 : rc.first_meeting_distance = 720)
  (h5 : rc.second_meeting_distance = 400) :
  river_width rc = 1280 := by
  sorry

#check river_width_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_theorem_l273_27334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_relation_volume_of_PQRS_l273_27368

/-- Represents a pyramid with a rectangular base -/
structure RectangularBasePyramid where
  -- Base dimensions
  base_length : ℝ
  base_width : ℝ
  -- Height (perpendicular to base)
  height : ℝ
  -- Length of edge from apex to a vertex of base
  edge_length : ℝ

/-- Volume of a pyramid -/
noncomputable def pyramid_volume (p : RectangularBasePyramid) : ℝ :=
  (1 / 3) * p.base_length * p.base_width * p.height

/-- Pythagorean theorem for right triangle formed by height, half-diagonal of base, and edge -/
theorem pythagorean_relation (p : RectangularBasePyramid) :
  p.height^2 + (p.base_length^2 / 4 + p.base_width^2 / 4) = p.edge_length^2 := by sorry

/-- Main theorem: Volume of the specific pyramid PQRS -/
theorem volume_of_PQRS :
  ∃ (p : RectangularBasePyramid),
    p.base_length = 10 ∧
    p.base_width = 5 ∧
    p.edge_length = 26 ∧
    pyramid_volume p = (50 * Real.sqrt 651) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_relation_volume_of_PQRS_l273_27368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l273_27309

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  2 * (Real.cos (ω * x / 2))^2 + Real.cos (ω * x + Real.pi / 3) - 1

def is_intersection_point (ω : ℝ) (x : ℝ) : Prop :=
  f ω x = 0

theorem triangle_side_length 
  (ω : ℝ) 
  (A B : ℝ) 
  (h_ω_pos : ω > 0)
  (h_A_intersect : is_intersection_point ω A)
  (h_B_intersect : is_intersection_point ω B)
  (h_AB_adjacent : ∀ x, A < x → x < B → ¬is_intersection_point ω x)
  (h_AB_length : B - A = Real.pi / 2)
  (h_f_A : f ω A = -3/2)
  (c : ℝ)
  (h_c : c = 3)
  (S : ℝ)
  (h_S : S = 3 * Real.sqrt 3)
  (h_acute : 0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2)
  : ∃ a : ℝ, a^2 = 13 ∧ 
    1/2 * a * c * Real.sin A = S ∧
    a^2 = c^2 + 4^2 - 2*c*4*Real.cos A :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l273_27309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sum_l273_27383

-- Define the sequences u, v, and w
noncomputable def u : ℕ → ℝ := sorry
noncomputable def v : ℕ → ℝ := sorry
noncomputable def w : ℕ → ℝ := sorry

-- Define the condition that w_n = u_n + v_n for all n
axiom w_def : ∀ n, w n = u n + v n

-- Define the difference operator
def Δ (x : ℕ → ℝ) (n : ℕ) : ℝ := x (n + 1) - x n

-- State the theorem
theorem difference_of_sum :
  ∀ n, Δ w n = Δ u n + Δ v n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sum_l273_27383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_l273_27323

theorem cos_pi_third : Real.cos (π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_l273_27323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inverse_difference_l273_27351

theorem cosine_inverse_difference (a b : ℝ) :
  a = 4 / 5 →
  b = 1 / 2 →
  Real.cos (Real.arccos a - Real.arcsin b) = (4 * Real.sqrt 3 + 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inverse_difference_l273_27351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l273_27395

/-- The function f(x) = x + x ln x -/
noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

/-- The condition that should hold for all x > 2 -/
def condition (m : ℤ) (x : ℝ) : Prop := (m - 2 : ℝ) * (x - 2) < f x

/-- The theorem stating that 6 is the maximum integer value of m satisfying the condition -/
theorem max_m_value : (∃ m : ℤ, ∀ x > 2, condition m x) ∧ 
  (∀ m : ℤ, (∀ x > 2, condition m x) → m ≤ 6) ∧
  (∀ x > 2, condition 6 x) := by
  sorry

#check max_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l273_27395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l273_27389

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x
def g (a : ℝ) (x : ℝ) : ℝ := (2 - a) * x^3

-- State the theorem
theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (∀ x y : ℝ, x < y → g a x < g a y) ∧
  ¬(∀ x y : ℝ, x < y → g a x < g a y → f a x > f a y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l273_27389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_length_number_above_198_l273_27324

/-- Represents the triangular-shaped sequence -/
def triangularSequence (k : ℕ) (n : ℕ) : ℕ := sorry

/-- The k-th row contains 2k-1 numbers -/
theorem row_length (k : ℕ) : 
  (Finset.range (2 * k - 1)).card = 2 * k - 1 := by sorry

/-- The number directly above 198 in the triangular sequence -/
theorem number_above_198 : 
  ∃ k n, triangularSequence k n = 198 ∧ triangularSequence (k-1) ((n-1)/2) = 170 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_length_number_above_198_l273_27324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_500_l273_27302

theorem closest_perfect_square_to_500 :
  ∀ n : ℤ, n * n ≠ 484 → |500 - 484| ≤ |500 - n * n| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_500_l273_27302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_order_l273_27321

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 - 2*x

-- Define a, b, and c
noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := 3^(1/2)

-- State the theorem
theorem f_order : f b > f a ∧ f a > f c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_order_l273_27321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_center_travel_distance_l273_27319

noncomputable def ball_diameter : ℝ := 6
noncomputable def R₁ : ℝ := 120
noncomputable def R₂ : ℝ := 50
noncomputable def R₃ : ℝ := 90
noncomputable def R₄ : ℝ := 75

noncomputable def center_path_distance : ℝ :=
  Real.pi * ((R₁ - ball_diameter / 2) + (R₂ + ball_diameter / 2) + 
       (R₃ - ball_diameter / 2) + (R₄ + ball_diameter / 2))

theorem ball_center_travel_distance :
  center_path_distance = 335 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_center_travel_distance_l273_27319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_approx_l273_27352

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 21, divided by π --/
noncomputable def coneVolumeDividedByPi : ℝ := by
  -- Define the sector angle in degrees
  let sectorAngle : ℝ := 270
  -- Define the radius of the original circle
  let circleRadius : ℝ := 21
  -- Calculate the base radius of the cone
  let baseRadius : ℝ := circleRadius * (sectorAngle / 360)
  -- Calculate the height of the cone using Pythagoras' theorem
  let coneHeight : ℝ := Real.sqrt (circleRadius ^ 2 - baseRadius ^ 2)
  -- Calculate the volume of the cone divided by π
  exact (1 / 3) * baseRadius ^ 2 * coneHeight

/-- Theorem stating that the volume of the cone divided by π is approximately 1146.32 --/
theorem cone_volume_approx : 
  ∃ ε > 0, |coneVolumeDividedByPi - 1146.32| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_approx_l273_27352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l273_27384

theorem coin_flip_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  6 * p^2 * (1 - p)^2 = (1 : ℝ) / 12 →
  p = (12 - Real.sqrt (96 + 48 * Real.sqrt 2)) / 24 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l273_27384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_markup_l273_27314

/-- The percentage markup on a product's cost price -/
noncomputable def percentage_markup (cost_price selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The percentage markup on the computer table is 10% -/
theorem computer_table_markup :
  let cost_price : ℝ := 7999.999999999999
  let selling_price : ℝ := 8800
  percentage_markup cost_price selling_price = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_markup_l273_27314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curious_not_red_l273_27308

-- Define the universe of spiders
variable (Spider : Type)

-- Define properties of spiders
variable (red : Spider → Prop)
variable (curious : Spider → Prop)
variable (intelligent : Spider → Prop)
variable (can_fly : Spider → Prop)

-- Define Alice's collection of spiders
variable (alice_spiders : Finset Spider)

-- State the given conditions
variable (total_spiders : alice_spiders.card = 17)
variable (red_spiders : (alice_spiders.filter red).card = 7)
variable (curious_spiders : (alice_spiders.filter curious).card = 8)

variable (curious_intelligent : ∀ s ∈ alice_spiders, curious s → intelligent s)
variable (red_cant_fly : ∀ s ∈ alice_spiders, red s → ¬can_fly s)
variable (cant_fly_not_intelligent : ∀ s ∈ alice_spiders, ¬can_fly s → ¬intelligent s)

-- State the theorem to be proved
theorem curious_not_red : ∀ s ∈ alice_spiders, curious s → ¬red s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curious_not_red_l273_27308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percentage_l273_27332

theorem retailer_profit_percentage
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (h1 : markup_percentage = 65)
  (h2 : discount_percentage = 25)
  (h3 : cost_price > 0) :
  (cost_price * (1 + markup_percentage / 100) * (1 - discount_percentage / 100) - cost_price) / cost_price * 100 = 23.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percentage_l273_27332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_remainder_division_by_nine_l273_27325

theorem largest_remainder_division_by_nine :
  ∃ (dividend remainder : ℕ),
    dividend / 9 = 9 ∧
    dividend % 9 = remainder ∧
    remainder = 8 ∧
    dividend = 89 ∧
    ∀ (d r : ℕ), d / 9 = 9 → d % 9 ≤ remainder :=
by
  -- We'll use 89 as the dividend and 8 as the remainder
  let dividend := 89
  let remainder := 8
  
  -- Prove the existence
  use dividend, remainder
  
  -- Prove the conjunction of all conditions
  constructor
  · -- Prove dividend / 9 = 9
    rfl
  · constructor
    · -- Prove dividend % 9 = remainder
      rfl
    · constructor
      · -- Prove remainder = 8
        rfl
      · constructor
        · -- Prove dividend = 89
          rfl
        · -- Prove ∀ (d r : ℕ), d / 9 = 9 → d % 9 ≤ remainder
          intros d r h
          -- This part requires more sophisticated reasoning
          -- For now, we'll use sorry to skip the proof
          sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_remainder_division_by_nine_l273_27325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_rhombus_not_square_l273_27378

/-- The shape represented by the equation x + y = 1 in the Cartesian coordinate system -/
def shape (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 1}

/-- Definition of a rhombus -/
def is_rhombus (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (v w : ℝ × ℝ),
    v ≠ 0 ∧ w ≠ 0 ∧ v.1 * w.1 + v.2 * w.2 = 0 ∧
    S = {p : ℝ × ℝ | ∃ (t s : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ -1 ≤ s ∧ s ≤ 1 ∧ p = center + t • v + s • w}

/-- Definition of a square -/
def is_square (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (v : ℝ × ℝ),
    v ≠ 0 ∧
    S = {p : ℝ × ℝ | ∃ (t s : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ -1 ≤ s ∧ s ≤ 1 ∧ p = center + t • v + s • ((-v.2, v.1) : ℝ × ℝ)}

theorem shape_is_rhombus_not_square (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  is_rhombus (shape a b) ∧ ¬is_square (shape a b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_rhombus_not_square_l273_27378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sequence_sum_l273_27300

def sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem alternating_sequence_sum : 
  let positive_start : ℤ := 3
  let positive_diff : ℤ := 10
  let negative_start : ℤ := -8
  let negative_diff : ℤ := -10
  let last_term : ℤ := 73
  let n : ℕ := ((last_term - positive_start) / positive_diff + 1).natAbs
  sequence_sum positive_start positive_diff n + 
  sequence_sum negative_start negative_diff (n - 1) = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sequence_sum_l273_27300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l273_27387

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, x^2 + y^2 = 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l273_27387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_f_values_l273_27343

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.sqrt (4^x + 1) - Real.sqrt (4^x + 2)
  else Real.sqrt (4^(-x) + 1) - Real.sqrt (4^(-x) + 2)

-- State the theorem
theorem order_of_f_values :
  let a := f (Real.log 0.2 / Real.log 3)
  let b := f (3^(-0.2 : ℝ))
  let c := f (-3^(1.1 : ℝ))
  (∀ x, f (-x) = f x) →  -- f is even
  (c > a ∧ a > b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_f_values_l273_27343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_radii_l273_27346

/-- Triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the semiperimeter of a triangle -/
noncomputable def semiperimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

/-- Calculate the area of a triangle using Heron's formula -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := semiperimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Calculate the radius of the inscribed circle -/
noncomputable def inradius (t : Triangle) : ℝ :=
  area t / semiperimeter t

/-- Calculate the radius of an escribed circle -/
noncomputable def exradius (t : Triangle) (side : ℝ) : ℝ :=
  area t / (semiperimeter t - side)

theorem triangle_circle_radii (t : Triangle) 
  (h1 : t.a = 5) (h2 : t.b = 12) (h3 : t.c = 13) :
  inradius t = 2 ∧ 
  exradius t t.a = 3 ∧ 
  exradius t t.b = 10 ∧ 
  exradius t t.c = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_radii_l273_27346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_five_in_factorial_30_l273_27364

theorem exponent_of_five_in_factorial_30 : ∃ k : ℕ, 
  (Nat.factorial 30 = 5^7 * k) ∧ (k % 5 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_five_in_factorial_30_l273_27364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_arithmetic_product_l273_27311

/-- An arithmetic sequence with a given first term and common difference -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ+ → ℝ := λ n ↦ a₁ + (n.val - 1 : ℝ) * d

theorem not_arithmetic_product (a₁ b₁ d₁ d₂ : ℝ) (hd₁ : d₁ ≠ 0) (hd₂ : d₂ ≠ 0) :
  ¬ ∃ (c : ℝ), ∀ (n : ℕ+),
    (arithmeticSequence a₁ d₁ (n + 1)) * (arithmeticSequence b₁ d₂ (n + 1)) -
    (arithmeticSequence a₁ d₁ n) * (arithmeticSequence b₁ d₂ n) = c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_arithmetic_product_l273_27311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnsFinalPushTimeIs42_5_l273_27377

/-- The time of John's final push in a race, given:
  * John starts 15 meters behind Steve
  * John's speed is 4.2 m/s
  * Steve's speed is 3.8 m/s
  * John finishes 2 meters ahead of Steve
-/
noncomputable def johnsFinalPushTime : ℝ := by
  -- Define the initial distance between John and Steve
  let initialGap : ℝ := 15
  -- Define John's speed
  let johnSpeed : ℝ := 4.2
  -- Define Steve's speed
  let steveSpeed : ℝ := 3.8
  -- Define the final distance between John and Steve
  let finalGap : ℝ := 2
  -- Define the total distance difference
  let totalDifference : ℝ := initialGap + finalGap
  -- Calculate the time
  exact totalDifference / (johnSpeed - steveSpeed)

/-- Theorem stating that John's final push time is 42.5 seconds -/
theorem johnsFinalPushTimeIs42_5 : johnsFinalPushTime = 42.5 := by
  -- Unfold the definition of johnsFinalPushTime
  unfold johnsFinalPushTime
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnsFinalPushTimeIs42_5_l273_27377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l273_27307

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem f_properties :
  -- The smallest positive period of f is π
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
    (∀ S : ℝ, S > 0 → (∀ x : ℝ, f (x + S) = f x) → T ≤ S)) ∧
  -- The intervals where f is monotonically increasing
  (∀ k : ℤ, ∀ x y : ℝ,
    k * Real.pi - 3 * Real.pi / 8 ≤ x ∧
    x < y ∧
    y ≤ k * Real.pi + Real.pi / 8 →
    f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l273_27307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l273_27394

-- Define the line l
def line_l (k x y : ℝ) : Prop := k * x - y - k + 1 = 0

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 2 then x^3 - 3*x^2 + 2*x + 1
  else a*x - 2*a + 1

-- Define the intersection property
def intersects_at_three_points (k a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
  line_l k x₁ (f a x₁) ∧ line_l k x₂ (f a x₂) ∧ line_l k x₃ (f a x₃)

-- Define the sum property
def sum_property (k a : ℝ) : Prop :=
  ∀ x₁ x₂ x₃ : ℝ, 0 < k → k < 3 → 
  intersects_at_three_points k a → x₁ + x₂ + x₃ < 3

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (∀ k : ℝ, sum_property k a) → a ∈ Set.Ici 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l273_27394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l273_27360

open Real

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = 2/5 ∧ 
    ∀ (x y : ℝ), 1/4 ≤ x ∧ x ≤ 2/3 → 1/5 ≤ y ∧ y ≤ 1/2 → f x y ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l273_27360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_through_origin_and_negative_one_l273_27310

/-- The slope angle of a line passing through (0,0) and (-1, -1) is 135°. -/
theorem slope_angle_through_origin_and_negative_one : 
  let line := {p : ℝ × ℝ | ∃ t : ℝ, p = (t, t)}
  ∃ (slope_angle : Set (ℝ × ℝ) → ℝ), slope_angle line = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_through_origin_and_negative_one_l273_27310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l273_27312

/-- A rhombus with given diagonal and perimeter -/
structure Rhombus where
  diagonal1 : ℝ
  perimeter : ℝ

/-- The length of the other diagonal in the rhombus -/
noncomputable def other_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (((r.perimeter / 4) ^ 2) - ((r.diagonal1 / 2) ^ 2))

theorem rhombus_other_diagonal (r : Rhombus) 
  (h1 : r.diagonal1 = 72) 
  (h2 : r.perimeter = 156) : 
  other_diagonal r = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l273_27312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_properties_l273_27372

-- Define the circles
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def circle_M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 4*p.1 - 2*p.2 + 4 = 0}

-- Define the intersection points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define IsTangentLine
def IsTangentLine (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem circles_properties :
  -- The circles have two common tangents
  ∃ (t1 t2 : Set (ℝ × ℝ)), t1 ≠ t2 ∧ IsTangentLine t1 circle_O ∧ IsTangentLine t1 circle_M ∧
                           IsTangentLine t2 circle_O ∧ IsTangentLine t2 circle_M ∧
  -- The length of segment AB is 4√5/5
  ‖A - B‖ = 4 * Real.sqrt 5 / 5 ∧
  -- The maximum distance between any point on circle O and any point on circle M is √5 + 3
  (∀ (E F : ℝ × ℝ), E ∈ circle_O → F ∈ circle_M → ‖E - F‖ ≤ Real.sqrt 5 + 3) ∧
  (∃ (E F : ℝ × ℝ), E ∈ circle_O ∧ F ∈ circle_M ∧ ‖E - F‖ = Real.sqrt 5 + 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_properties_l273_27372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stand_distance_l273_27303

noncomputable section

/-- The distance to the bus stand in kilometers -/
def distance_to_bus_stand : ℝ := 2.2

/-- The speed of the first walk in km/h -/
def speed1 : ℝ := 3

/-- The speed of the second walk in km/h -/
def speed2 : ℝ := 6

/-- The time difference in hours when walking at speed1 (12 minutes late) -/
def time_diff1 : ℝ := 12 / 60

/-- The time difference in hours when walking at speed2 (10 minutes early) -/
def time_diff2 : ℝ := 10 / 60

theorem bus_stand_distance :
  distance_to_bus_stand / speed1 - time_diff1 = distance_to_bus_stand / speed2 + time_diff2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stand_distance_l273_27303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_negative_iff_a_in_range_l273_27340

/-- The function g(x) defined in terms of parameter a -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + 2 * Real.log x - 2 * a * x

/-- Theorem stating the equivalence between g(x) < 0 for all x > 1 and a ∈ [-1, 1] -/
theorem g_negative_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x > 1 → g a x < 0) ↔ a ∈ Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_negative_iff_a_in_range_l273_27340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l273_27350

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (x^2 + 1)

-- State the theorem
theorem odd_function_property (a b : ℝ) :
  (∀ x, f a b (-x) = -(f a b x)) →  -- f is an odd function
  f a b (1/2) = 2/5 →               -- f(1/2) = 2/5
  f a b 2 = 2/5 :=                  -- Then f(2) = 2/5
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l273_27350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_is_95_l273_27317

/-- Represents a score in the test -/
structure Score where
  value : ℕ

/-- Represents the frequency of each score -/
def frequency : Score → ℕ := sorry

/-- The set of all scores in the test -/
def scores : Set Score := sorry

/-- The maximum possible score on the test -/
def maxScore : ℕ := 120

/-- Definition of mode: the score that appears most frequently -/
def isMode (s : Score) : Prop :=
  ∀ t ∈ scores, frequency s ≥ frequency t

/-- The mode of the scores is 95 -/
theorem mode_is_95 : ∃ s ∈ scores, s.value = 95 ∧ isMode s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_is_95_l273_27317
