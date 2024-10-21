import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_is_correct_l1034_103456

/-- A structure representing a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop := λ x y => a * x + b * y + c = 0

/-- The reflection axis --/
def reflection_axis : Line :=
  { a := 1, b := 1, c := -5 }

/-- The line on which the incident light ray reflects --/
def incident_line : Line :=
  { a := 2, b := -1, c := 2 }

/-- The reflected light ray --/
def reflected_ray : Line :=
  { a := 1, b := -2, c := 7 }

/-- Theorem stating that the given reflected ray is correct --/
theorem reflected_ray_is_correct :
  ∃ (incident_point reflected_point : ℝ × ℝ),
    incident_line.eq incident_point.1 incident_point.2 ∧
    reflection_axis.eq incident_point.1 incident_point.2 ∧
    reflected_ray.eq reflected_point.1 reflected_point.2 ∧
    -- Additional conditions to ensure reflection properties
    (reflected_point.1 - incident_point.1) = (incident_point.1 - 0) ∧
    (reflected_point.2 - incident_point.2) = (incident_point.2 - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_is_correct_l1034_103456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elephant_weight_equation_l1034_103465

/-- The weight of each porter in catties -/
def porter_weight : ℕ := 120

/-- The number of blocks in the first scenario -/
def blocks_1 : ℕ := 20

/-- The number of porters in the first scenario -/
def porters_1 : ℕ := 3

/-- The number of blocks in the second scenario -/
def blocks_2 : ℕ := 21

/-- The number of porters in the second scenario -/
def porters_2 : ℕ := 1

/-- The weight of the elephant in catties -/
def y : ℕ → ℕ := fun _ => 0  -- Placeholder function

/-- The weight of each block in catties -/
def x : ℕ → ℕ := fun _ => 0  -- Placeholder function

theorem elephant_weight_equation :
  ∀ (y : ℕ), (y - porters_1 * porter_weight) / blocks_1 = (y - porters_2 * porter_weight) / blocks_2 :=
by
  intro y
  sorry  -- Proof skipped

#check elephant_weight_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elephant_weight_equation_l1034_103465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_specific_line_l1034_103429

/-- The distance from the origin to a line ax + by + c = 0 is |c| / √(a² + b²) -/
noncomputable def distance_origin_to_line (a b c : ℝ) : ℝ :=
  |c| / Real.sqrt (a^2 + b^2)

/-- The line equation 4x + 3y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  4 * x + 3 * y - 1 = 0

theorem distance_origin_to_specific_line :
  distance_origin_to_line 4 3 (-1) = 1/5 := by
  -- Unfold the definition of distance_origin_to_line
  unfold distance_origin_to_line
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_specific_line_l1034_103429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_v₃_l1034_103401

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 + 5x^4 + 8x^3 + 7x^2 - 6x + 11 -/
def f : List ℝ := [11, -6, 7, 8, 5, 2]

/-- v₃ is the third step in Horner's method for f(x) when x = 3 -/
def v₃ : ℝ := (horner (f.take 4) 3) * 3 + f[4]!

theorem horner_method_v₃ : v₃ = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_v₃_l1034_103401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_squared_prop_inv_R_l1034_103493

/-- Represents the gravitational constant -/
noncomputable def G : ℝ := sorry

/-- Represents the mass of Saturn -/
noncomputable def M : ℝ := sorry

/-- Represents the mass of a particle in Saturn's ring -/
noncomputable def m : ℝ := sorry

/-- Represents the distance from the center of Saturn to a particle in its ring -/
noncomputable def R : ℝ → ℝ := sorry

/-- Represents the velocity of a particle in Saturn's ring -/
noncomputable def v : ℝ → ℝ := sorry

/-- States that the gravitational force equals the centripetal force for a particle in Saturn's ring -/
axiom force_balance (t : ℝ) : (G * M * m) / (R t)^2 = m * (v t)^2 / (R t)

/-- Theorem stating that v^2 is inversely proportional to R for particles in Saturn's ring -/
theorem v_squared_prop_inv_R : ∃ (k : ℝ), ∀ (t : ℝ), (v t)^2 * (R t) = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_squared_prop_inv_R_l1034_103493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l1034_103483

noncomputable def vector_a : ℝ × ℝ := (Real.sqrt 3, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)

theorem vector_angle_problem (m : ℝ) :
  let a := vector_a
  let b := vector_b m
  (a.1 * b.1 + a.2 * b.2) = (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2) * Real.cos (2 * Real.pi / 3)) →
  m = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l1034_103483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_one_l1034_103414

theorem log_sum_equals_one (a b : ℝ) (h1 : (10 : ℝ)^a = 2) (h2 : b = Real.log 5) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_one_l1034_103414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l1034_103424

noncomputable def line (x : ℝ) : ℝ := 2 * x + 3

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 12 = 0

noncomputable def distance_to_line (a b c : ℝ) (x₀ y₀ : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

theorem min_tangent_length :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = line x₀ ∧ 
    circle_eq x₀ y₀ ∧
    ∀ (x y : ℝ), y = line x ∧ circle_eq x y → 
      Real.sqrt ((x - x₀)^2 + (y - y₀)^2) ≥ Real.sqrt 19 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l1034_103424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_l1034_103470

-- Define the spinners
def spinnerA : Finset ℕ := {4, 5, 6}
def spinnerB : Finset ℕ := {1, 2, 3}
def spinnerC : Finset ℕ := {7, 8, 9}

-- Define a function to check if a number is odd
def isOdd (n : ℕ) : Bool := n % 2 = 1

-- Define a function to calculate the probability of an event
def probability (event : Finset (ℕ × ℕ × ℕ)) (sample_space : Finset (ℕ × ℕ × ℕ)) : ℚ :=
  (event.card : ℚ) / (sample_space.card : ℚ)

-- Theorem statement
theorem odd_sum_probability :
  let sample_space := spinnerA.product (spinnerB.product spinnerC)
  let odd_sum_outcomes := sample_space.filter (fun (a, bc) => isOdd (a + bc.1 + bc.2))
  probability odd_sum_outcomes sample_space = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_probability_l1034_103470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_l1034_103476

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + x / 2 - Real.sin x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 1 / 2 - Real.cos x

-- Theorem statement
theorem f_decreasing_intervals :
  ∀ x ∈ Set.Ioo (0 : ℝ) (2 * π),
    (f' x < 0 ↔ (x ∈ Set.Ioo 0 (π / 3) ∨ x ∈ Set.Ioo (5 * π / 3) (2 * π))) :=
by sorry

-- Here, Set.Ioo represents an open interval (a, b)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_l1034_103476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_intersection_l1034_103494

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given elements
variable (M N A B : Point)
variable (MN : Line)

-- Define the symmetric point B'
noncomputable def B' : Point := sorry

-- Define the intersection point X
noncomputable def X : Point := sorry

-- Define the angle between a line and a segment
noncomputable def angle (l : Line) (p q : Point) : ℝ := sorry

-- Define a membership relation for Point and Line
def on_line (p : Point) (l : Line) : Prop := sorry

-- State the theorem
theorem equal_angles_intersection :
  on_line X MN ∧ on_line X (Line.mk sorry sorry sorry) →  -- X is on both MN and AB'
  angle MN A X = angle MN B X := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_intersection_l1034_103494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_intersecting_or_disjoint_l1034_103462

/-- A line segment on a line --/
structure LineSegment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- The property we want to prove --/
def hasEightIntersectingOrDisjoint (segments : Finset LineSegment) : Prop :=
  (∃ (point : ℝ) (intersecting : Finset LineSegment), 
    intersecting.card = 8 ∧ intersecting ⊆ segments ∧
    ∀ seg, seg ∈ intersecting → seg.left ≤ point ∧ point ≤ seg.right) ∨
  (∃ (disjoint : Finset LineSegment),
    disjoint.card = 8 ∧ disjoint ⊆ segments ∧
    ∀ seg1 seg2, seg1 ∈ disjoint → seg2 ∈ disjoint → seg1 ≠ seg2 → 
      seg1.right < seg2.left ∨ seg2.right < seg1.left)

/-- The main theorem --/
theorem eight_intersecting_or_disjoint (segments : Finset LineSegment) 
    (h : segments.card = 50) :
  hasEightIntersectingOrDisjoint segments := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_intersecting_or_disjoint_l1034_103462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_PBC_l1034_103411

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a point on the y-axis
def on_y_axis (x y : ℝ) : Prop := x = 0

-- Define the inscribed circle
def inscribed_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the triangle PBC
structure Triangle_PBC where
  P : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_P : parabola P.1 P.2
  h_B : on_y_axis B.1 B.2
  h_C : on_y_axis C.1 C.2
  h_inscribed : inscribed_circle 1 0  -- The circle is centered at (1, 0)

-- Define the area of a triangle given its vertices
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Theorem statement
theorem min_area_triangle_PBC :
  ∀ t : Triangle_PBC, ∃ min_area : ℝ, min_area = 8 ∧
  ∀ area : ℝ, area = triangle_area t.P t.B t.C → area ≥ min_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_PBC_l1034_103411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1034_103431

theorem tan_difference (α β : Real) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) :
  Real.tan (α - β) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1034_103431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_three_equals_fourteen_l1034_103453

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 18 / (x + 2)

noncomputable def g (x : ℝ) : ℝ := 4 * (Function.invFun f x) - 2

-- State the theorem
theorem g_of_three_equals_fourteen : g 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_three_equals_fourteen_l1034_103453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_two_integer_solutions_l1034_103408

theorem inequality_two_integer_solutions (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x ∈ s, (a * x - 1)^2 < x^2) ↔ 
  (-3/2 < a ∧ a ≤ -4/3) ∨ (4/3 ≤ a ∧ a < 3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_two_integer_solutions_l1034_103408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1034_103405

/-- The asymptotes of the hyperbola (x²/16) - (y²/9) = -1 are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  let hyperbola := fun (x y : ℝ) => x^2 / 16 - y^2 / 9 = -1
  let asymptote := fun (x y : ℝ) => y = (3/4) * x ∨ y = -(3/4) * x
  ∀ (x y : ℝ), hyperbola x y → asymptote x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1034_103405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_theorem_l1034_103427

/-- Represents the properties of a train -/
structure Train where
  length : ℝ
  speed : ℝ

/-- Calculates the time for two trains to meet -/
noncomputable def time_to_meet (train_a : Train) (train_b : Train) (initial_distance : ℝ) : ℝ :=
  (initial_distance + train_a.length + train_b.length) / 
  ((train_a.speed + train_b.speed) * 1000 / 3600)

/-- Calculates the position of a train after a given time -/
noncomputable def position_after_time (train : Train) (time : ℝ) : ℝ :=
  (train.speed * 1000 / 3600) * time

/-- Theorem stating the time for trains A and B to meet and the position of train C -/
theorem train_meeting_theorem (train_a train_b train_c : Train) 
  (h1 : train_a.length = 210)
  (h2 : train_a.speed = 74)
  (h3 : train_b.length = 120)
  (h4 : train_b.speed = 92)
  (h5 : train_c.length = 150)
  (h6 : train_c.speed = 110)
  (h7 : initial_distance = 160) :
  let meeting_time := time_to_meet train_a train_b initial_distance
  let train_c_position := position_after_time train_c meeting_time
  abs (meeting_time - 10.63) < 0.01 ∧ abs (train_c_position - 325.05) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_theorem_l1034_103427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l1034_103403

/-- Represents a rectangular farm -/
structure RectangularFarm where
  area : ℝ
  shortSide : ℝ

/-- Calculates the cost of fencing for a rectangular farm -/
noncomputable def fencingCost (farm : RectangularFarm) (costPerMeter : ℝ) : ℝ :=
  let longSide := farm.area / farm.shortSide
  let diagonal := Real.sqrt (longSide^2 + farm.shortSide^2)
  costPerMeter * (longSide + farm.shortSide + diagonal)

/-- Theorem stating the fencing cost for the given farm -/
theorem fencing_cost_theorem (farm : RectangularFarm) (h1 : farm.area = 1200) (h2 : farm.shortSide = 30) :
  fencingCost farm 13 = 1560 := by
  sorry

#check fencing_cost_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l1034_103403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_system_l1034_103447

theorem unique_solution_system : ∃! (x y : ℝ), 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x - 2) + (y - 2)) ∧
  x = 5 ∧ y = 2 := by
  sorry

#check unique_solution_system

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_system_l1034_103447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1034_103432

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ Real.sqrt (a + 1) + Real.sqrt (b + 3)) →
  Real.sqrt (a + 1) + Real.sqrt (b + 3) = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1034_103432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_in_figure_l1034_103469

/-- Given a figure with straight lines BD, FC, GC, and FE, prove that the sum of angles a, b, c, d, e, f, and g equals 540° -/
theorem sum_of_angles_in_figure (a b c d e f g : ℝ) 
  (h1 : a + b + g + 180 = 360) -- Sum of angles in quadrilateral ABHG
  (h2 : c + f = e + d) -- Exterior angle of triangle CJH
  (h3 : c + f + e + d + 180 = 360) -- Sum of angles in quadrilateral JHDE
  : a + b + c + d + e + f + g = 540 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_in_figure_l1034_103469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_sum_l1034_103472

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def is_even_sum (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 0

def even_sum_pairs : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2 ∧ is_even_sum pair)

theorem probability_even_sum :
  (even_sum_pairs.card : ℚ) / (Finset.card (card_set.powerset.filter (λ s => s.card = 2))) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_sum_l1034_103472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_alpha_with_irrational_cos_and_rational_multiples_l1034_103418

theorem no_alpha_with_irrational_cos_and_rational_multiples :
  ¬ ∃ α : ℝ, Irrational (Real.cos α) ∧
    (∃ q : ℚ, Real.cos (2 * α) = ↑q) ∧
    (∃ q : ℚ, Real.cos (3 * α) = ↑q) ∧
    (∃ q : ℚ, Real.cos (4 * α) = ↑q) ∧
    (∃ q : ℚ, Real.cos (5 * α) = ↑q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_alpha_with_irrational_cos_and_rational_multiples_l1034_103418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l1034_103457

/-- Represents a cyclist with their speeds for different terrains -/
structure Cyclist where
  flat_speed : ℝ
  downhill_speed : ℝ
  uphill_speed : ℝ

/-- Represents a segment of the route -/
structure RouteSegment where
  distance : ℝ
  terrain : String

/-- Calculates the time taken for a cyclist to complete a route segment -/
noncomputable def time_for_segment (c : Cyclist) (s : RouteSegment) : ℝ :=
  match s.terrain with
  | "flat" => s.distance / c.flat_speed
  | "downhill" => s.distance / c.downhill_speed
  | "uphill" => s.distance / c.uphill_speed
  | _ => 0  -- Default case, should not occur in our problem

/-- Calculates the total time for a cyclist to complete a route -/
noncomputable def total_time (c : Cyclist) (route : List RouteSegment) : ℝ :=
  route.foldl (fun acc segment => acc + time_for_segment c segment) 0

/-- Helper function to check if a number is approximately equal to another -/
def approx_equal (x y : ℝ) (ε : ℝ := 0.01) : Prop :=
  abs (x - y) < ε

/-- Notation for approximate equality -/
notation:50 x " ≈ " y => approx_equal x y

/-- The main theorem statement -/
theorem journey_time_difference (minnie penny : Cyclist) (route_minnie route_penny : List RouteSegment) :
  minnie.flat_speed = 25 ∧ minnie.downhill_speed = 35 ∧ minnie.uphill_speed = 10 ∧
  penny.flat_speed = 35 ∧ penny.downhill_speed = 45 ∧ penny.uphill_speed = 15 ∧
  route_minnie = [
    {distance := 15, terrain := "uphill"},
    {distance := 20, terrain := "downhill"},
    {distance := 25, terrain := "flat"}
  ] ∧
  route_penny = [
    {distance := 25, terrain := "flat"},
    {distance := 20, terrain := "uphill"},
    {distance := 15, terrain := "downhill"}
  ] →
  (total_time minnie route_minnie - total_time penny route_penny) * 60 ≈ 414.29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l1034_103457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_decreasing_l1034_103481

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ) + Real.sqrt 3 * Real.cos (2 * x + φ)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

def is_decreasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → g y < g x

theorem f_symmetry_and_decreasing :
  let φ := 2 * Real.pi / 3
  is_odd (f · φ) ∧ is_decreasing (f · φ) 0 (Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_decreasing_l1034_103481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_driving_distance_l1034_103473

theorem john_driving_distance : ∃ (total_distance : ℝ), total_distance = 235 := by
  let first_segment_speed : ℝ := 35
  let first_segment_time : ℝ := 2
  let second_segment_speed : ℝ := 55
  let second_segment_time : ℝ := 3
  let first_segment_distance := first_segment_speed * first_segment_time
  let second_segment_distance := second_segment_speed * second_segment_time
  let total_distance := first_segment_distance + second_segment_distance
  exists total_distance
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_driving_distance_l1034_103473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1034_103485

/-- A function f is bounded on a set S if there exists a constant M such that |f(x)| ≤ M for all x in S -/
def is_bounded (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x ∈ S, |f x| ≤ M

/-- Definition of function f -/
noncomputable def f (x : ℝ) : ℝ := (1/4)^x + (1/2)^x - 1

/-- Definition of function g -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1 - m * 2^x) / (1 + m * 2^x)

theorem problem_solution :
  (¬ is_bounded f (Set.Iio 0)) ∧ 
  (∀ x, g 1 (-x) = -(g 1 x)) ∧
  (is_bounded (g 1) Set.univ) ∧
  (∀ m, m > 0 → m < 1/2 → ∀ G, (∀ x ∈ Set.Icc 0 1, g m x ≤ G) → G ≥ (1 - m) / (1 + m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1034_103485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_lengths_l1034_103435

-- Define the triangle and point
variable (A B C M : ℝ × ℝ)

-- Define the intersections
variable (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the lengths
variable (AM BM CM : ℝ)

-- Assumptions
axiom acute_triangle : ∀ (X Y Z : ℝ × ℝ), X ≠ Y → Y ≠ Z → Z ≠ X → (X.1 - Y.1)^2 + (X.2 - Y.2)^2 + (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 + (Z.1 - X.1)^2 + (Z.2 - X.2)^2 < 2 * ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)

axiom point_inside : ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧ M = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)

axiom intersections : 
  ∃ (t₁ t₂ t₃ : ℝ), 0 < t₁ ∧ t₁ < 1 ∧ 0 < t₂ ∧ t₂ < 1 ∧ 0 < t₃ ∧ t₃ < 1 ∧
  A₁ = (t₁ * B.1 + (1 - t₁) * C.1, t₁ * B.2 + (1 - t₁) * C.2) ∧
  B₁ = (t₂ * A.1 + (1 - t₂) * C.1, t₂ * A.2 + (1 - t₂) * C.2) ∧
  C₁ = (t₃ * A.1 + (1 - t₃) * B.1, t₃ * A.2 + (1 - t₃) * B.2)

axiom equal_segments : Real.sqrt ((M.1 - A₁.1)^2 + (M.2 - A₁.2)^2) = 3 ∧ 
                       Real.sqrt ((M.1 - B₁.1)^2 + (M.2 - B₁.2)^2) = 3 ∧ 
                       Real.sqrt ((M.1 - C₁.1)^2 + (M.2 - C₁.2)^2) = 3

axiom sum_of_lengths : AM + BM + CM = 43

-- Theorem to prove
theorem product_of_lengths : AM * BM * CM = 441 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_lengths_l1034_103435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_difference_l1034_103410

theorem cats_difference (hogs : ℕ) (cats : ℕ) : 
  hogs = 75 → hogs = 3 * cats → (0.6 * (cats : ℝ) - 10 : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_difference_l1034_103410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1034_103446

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x / (x - 1)

-- Define the domain
def D : Set ℝ := {x : ℝ | x ≥ 0 ∧ x ≠ 1}

-- Theorem statement
theorem f_domain : ∀ x : ℝ, x ∈ D ↔ f x ∈ Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1034_103446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_14_dividing_40_factorial_l1034_103421

theorem greatest_power_of_14_dividing_40_factorial : 
  (∃ m : ℕ, m = 5 ∧ 
   (∀ k : ℕ, (Nat.factorial 40) % (14^k) = 0 → k ≤ m) ∧
   (Nat.factorial 40) % (14^m) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_14_dividing_40_factorial_l1034_103421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1034_103442

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h_ineq : ∀ x, deriv f (2 * x) > (log 2 / 2) * f (2 * x)) :
  f 2 > 2 * f 0 ∧ 2 * f 0 > 4 * f (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1034_103442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dedekind_cut_no_simultaneous_extrema_l1034_103458

/-- Definition of a Dedekind Cut -/
structure DedekindCut (Q : Type) [LinearOrder Q] where
  M : Set Q
  N : Set Q
  nonempty_M : M.Nonempty
  nonempty_N : N.Nonempty
  union_eq_Q : M ∪ N = Set.univ
  intersection_empty : M ∩ N = ∅
  M_lt_N : ∀ (x y : Q), x ∈ M → y ∈ N → x < y

/-- Theorem: In a Dedekind Cut, M cannot have a greatest element and N cannot have a least element simultaneously -/
theorem dedekind_cut_no_simultaneous_extrema {Q : Type} [LinearOrder Q] (cut : DedekindCut Q) :
  ¬(∃ (m : Q), m ∈ cut.M ∧ ∀ (x : Q), x ∈ cut.M → x ≤ m) ∨
  ¬(∃ (n : Q), n ∈ cut.N ∧ ∀ (y : Q), y ∈ cut.N → n ≤ y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dedekind_cut_no_simultaneous_extrema_l1034_103458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_oddness_l1034_103474

-- Define the function f
noncomputable def f (x : ℝ) := Real.log ((1 + x) / (1 - x))

-- Theorem for the domain and oddness of f
theorem f_domain_and_oddness :
  (∀ x : ℝ, f x ≠ 0 ↔ -1 < x ∧ x < 1) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x) := by
  sorry

#check f_domain_and_oddness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_oddness_l1034_103474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_C_l1034_103444

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a point in Cartesian coordinates -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Conversion from polar to Cartesian coordinates -/
noncomputable def polarToCartesian (p : PolarPoint) : CartesianPoint :=
  { x := p.ρ * Real.cos p.θ
  , y := p.ρ * Real.sin p.θ }

/-- The curve C in polar coordinates -/
def C (p : PolarPoint) : Prop :=
  p.ρ * Real.sin (p.θ - Real.pi/4) = Real.sqrt 2

/-- The proposed Cartesian equation of curve C -/
def CCartesian (p : CartesianPoint) : Prop :=
  p.x - p.y + 2 = 0

/-- Theorem stating that the polar equation of C is equivalent to its Cartesian equation -/
theorem polar_to_cartesian_C :
    ∀ p : PolarPoint, C p ↔ CCartesian (polarToCartesian p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_C_l1034_103444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1034_103448

open Real Set

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * sin x + cos x

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x ∈ Ioo (3 * π / 2) (5 * π / 2), 
    deriv f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1034_103448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l1034_103415

/-- Represents a racer in the competition -/
inductive Racer : Type
  | X : Racer
  | Y : Racer
  | Z : Racer
deriving BEq, Repr

/-- Represents the ranking of racers -/
def Ranking := List Racer

/-- The initial ranking of the race -/
def initial_ranking : Ranking := [Racer.X, Racer.Y, Racer.Z]

/-- The number of position changes for each racer -/
def position_changes : Racer → Nat
  | Racer.X => 5
  | Racer.Y => 0  -- Not specified in the problem, so we set it to 0
  | Racer.Z => 6

/-- Predicate to check if a racer finished before another -/
def finished_before (r1 r2 : Racer) (final_ranking : Ranking) : Prop :=
  final_ranking.indexOf r1 < final_ranking.indexOf r2

/-- The theorem to prove -/
theorem race_result (final_ranking : Ranking) :
  (position_changes Racer.X = 5) →
  (position_changes Racer.Z = 6) →
  (finished_before Racer.Y Racer.X final_ranking) →
  (final_ranking = [Racer.Y, Racer.X, Racer.Z]) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l1034_103415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_slant_angle_l1034_103439

/-- The angle between the slant height and the base plane of a regular triangular pyramid -/
noncomputable def slant_height_angle (R : ℝ) (b : ℝ) : ℝ :=
  Real.arcsin ((Real.sqrt 13 - 1) / 3)

/-- Theorem: In a regular triangular pyramid where the radius of the circumscribed sphere
    equals the slant height, the angle between the slant height and the base plane
    is arcsin((√13 - 1)/3). -/
theorem regular_triangular_pyramid_slant_angle (R b : ℝ) 
    (h_positive : R > 0)
    (h_equal : R = b) : 
  slant_height_angle R b = Real.arcsin ((Real.sqrt 13 - 1) / 3) := by
  sorry

#check regular_triangular_pyramid_slant_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_slant_angle_l1034_103439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flora_finished_eleventh_l1034_103478

-- Define the type for racers
inductive Racer
| Alice
| Bob
| Chris
| Dana
| Ethan
| Flora
| Other : Nat → Racer

-- Define the function that maps racers to their finishing positions
def finish_position : Racer → Nat := sorry

-- Define the total number of racers
def total_racers : Nat := 15

-- State the theorem
theorem flora_finished_eleventh :
  (finish_position Racer.Dana = finish_position Racer.Ethan + 3) →
  (finish_position Racer.Alice + 2 = finish_position Racer.Bob) →
  (finish_position Racer.Chris + 5 = finish_position Racer.Flora) →
  (finish_position Racer.Flora = finish_position Racer.Dana + 2) →
  (finish_position Racer.Ethan + 3 = finish_position Racer.Alice) →
  (finish_position Racer.Bob = 7) →
  (finish_position Racer.Flora = 11) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flora_finished_eleventh_l1034_103478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1034_103455

-- Define the two curves
def C₁ (x y : ℝ) : Prop := x^2 - y + 1 = 0
def C₂ (x y : ℝ) : Prop := y^2 - x + 1 = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 / 4 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ → 
    distance x₁ y₁ x₂ y₂ ≥ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1034_103455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_changes_percentage_theorem_l1034_103402

/-- Represents the distribution of student responses --/
structure ResponseDistribution where
  yes : Int
  no : Int
  undecided : Int
  total : Int
  sum_constraint : yes + no + undecided = total

/-- The problem setup --/
def initialDistribution : ResponseDistribution := {
  yes := 30
  no := 40
  undecided := 30
  total := 100
  sum_constraint := by simp
}

def finalDistribution : ResponseDistribution := {
  yes := 50
  no := 20
  undecided := 30
  total := 100
  sum_constraint := by simp
}

/-- Calculate the minimum number of students who must have changed their answers --/
def minChanges (initial final : ResponseDistribution) : Int :=
  max (abs (initial.yes - final.yes)) (abs (initial.no - final.no))

/-- Calculate the maximum number of students who could have changed their answers --/
def maxChanges (initial final : ResponseDistribution) : Int :=
  abs (initial.yes - final.yes) + abs (initial.no - final.no)

/-- The main theorem to prove --/
theorem changes_percentage_theorem :
  minChanges initialDistribution finalDistribution = 20 ∧
  maxChanges initialDistribution finalDistribution = 30 := by
  sorry

#eval minChanges initialDistribution finalDistribution
#eval maxChanges initialDistribution finalDistribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_changes_percentage_theorem_l1034_103402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_interval_f_decreasing_interval_is_subset_of_domain_l1034_103480

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 6 - x)

theorem f_strictly_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 * Real.pi / 3 → f x₂ < f x₁ :=
by sorry

def f_domain : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3 * Real.pi / 2}

theorem f_decreasing_interval_is_subset_of_domain :
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi / 3} ⊆ f_domain :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_interval_f_decreasing_interval_is_subset_of_domain_l1034_103480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_problem_l1034_103450

theorem average_problem (x : ℚ) : 
  (((List.range 50).map (λ i => (i + 1 : ℚ))).sum + x) / 51 = 50 * x → x = 1275 / 2549 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_problem_l1034_103450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l1034_103433

/-- The curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

/-- The line l in Cartesian coordinates -/
def l (x y : ℝ) : Prop := 2 * x - y - 6 = 0

/-- The distance function from a point to a line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |2 * x - y - 6| / Real.sqrt 5

/-- The theorem stating that the point (-3/2, 1) on C₁ has the maximum distance to l -/
theorem max_distance_point :
  C₁ (-3/2) 1 ∧
  (∀ x y : ℝ, C₁ x y → distance_to_line x y ≤ distance_to_line (-3/2) 1) ∧
  distance_to_line (-3/2) 1 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l1034_103433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selected_coins_cover_all_amounts_selected_coins_subset_available_correct_coin_selection_l1034_103484

/-- Available coin denominations -/
def available_coins : List ℕ := [1, 3, 5, 10, 20, 50]

/-- The selected set of coins -/
def selected_coins : List ℕ := [1, 1, 3, 5, 10, 10, 20, 50]

/-- Function to check if a given amount can be paid using the selected coins -/
def can_pay (amount : ℕ) (coins : List ℕ) : Prop :=
  ∃ (coin_counts : List ℕ), 
    coin_counts.length = coins.length ∧ 
    (List.zip coins coin_counts).foldl (λ sum (c, n) => sum + c * n) 0 = amount

/-- Theorem stating that the selected coins can pay any amount from 1 to 100 -/
theorem selected_coins_cover_all_amounts :
  ∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ 100 → can_pay amount selected_coins :=
by sorry

/-- Theorem stating that the selected coins are a subset of available coins -/
theorem selected_coins_subset_available :
  ∀ (coin : ℕ), coin ∈ selected_coins → coin ∈ available_coins :=
by sorry

/-- Main theorem proving the correctness of the selected coins -/
theorem correct_coin_selection :
  (selected_coins.length = 8) ∧
  (∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ 100 → can_pay amount selected_coins) ∧
  (∀ (coin : ℕ), coin ∈ selected_coins → coin ∈ available_coins) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selected_coins_cover_all_amounts_selected_coins_subset_available_correct_coin_selection_l1034_103484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1034_103440

noncomputable def f (a x : ℝ) : ℝ := (a * x + 2) * Real.log x - (x^2 + a * x - a - 1)

theorem f_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) (a * Real.log x - 2 * x + 2 / x) x) →
  (HasDerivAt (f a) (2 / Real.exp 1 - 2 * Real.exp 1) (Real.exp 1)) →
  (∃ x₀, ∀ x, f a x ≤ f a x₀) ∧ (¬∃ x₀, ∀ x, f a x ≥ f a x₀) ∧
  (∀ x > 1, f a x < 0 ↔ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1034_103440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_equation_l1034_103461

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  -- Center at origin and major axis along y-axis are implicit in the structure
  a : ℝ  -- Semi-major axis length
  c : ℝ  -- Semi-focal distance
  h : a = Real.sqrt 2 * c  -- Distance from foci to minor axis endpoints
  k : a - c = Real.sqrt 2 - 1  -- Shortest distance from foci to ellipse

/-- The equation of the special ellipse -/
def ellipse_equation (e : SpecialEllipse) : Prop :=
  ∀ x y : ℝ, y^2 / 2 + x^2 = 1 ↔ (x, y) ∈ Set.range (fun (t : ℝ × ℝ) => (t.1, t.2))

/-- Theorem stating that the given ellipse has the equation y²/2 + x² = 1 -/
theorem special_ellipse_equation (e : SpecialEllipse) : ellipse_equation e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_equation_l1034_103461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l1034_103467

/-- The function f(x) = log(x^2 - ax + 2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 2)

/-- The theorem stating that if f(x) is strictly decreasing on (-∞, a/2], then 1 < a < 2√2 -/
theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ a/2 → f a x₁ > f a x₂) →
  1 < a ∧ a < 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l1034_103467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1034_103454

-- Define the line
def line (x y : ℝ) : Prop := 2*x + 3*y - 10 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y+1)^2 = 13

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem circle_tangent_to_line : 
  -- The circle's center is on the y-axis
  ∃ b : ℝ, circle_eq 0 b
  -- The circle is tangent to the line at the given point
  ∧ line (point_of_tangency.1) (point_of_tangency.2)
  -- The circle passes through the point of tangency
  ∧ circle_eq (point_of_tangency.1) (point_of_tangency.2)
  -- The line is tangent to the circle (only one point of intersection)
  ∧ ∀ x y : ℝ, line x y ∧ circle_eq x y → (x, y) = point_of_tangency :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1034_103454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_partitions_l1034_103477

-- Define a set with 3 elements
def A : Set Nat := {1, 2, 3}

-- Define a partition
def is_partition (A₁ A₂ : Set Nat) (A : Set Nat) : Prop :=
  A₁ ∪ A₂ = A ∧ A₁ ∩ A₂ = ∅

-- Define when two partitions are considered the same
def same_partition (A₁ A₂ B₁ B₂ : Set Nat) : Prop :=
  (A₁ = B₁ ∧ A₂ = B₂) ∨ (A₁ = B₂ ∧ A₂ = B₁)

-- Theorem statement
theorem num_distinct_partitions :
  ∃ (partitions : List (Set Nat × Set Nat)),
    (∀ p, p ∈ partitions → is_partition p.1 p.2 A) ∧
    (∀ p q, p ∈ partitions → q ∈ partitions → p ≠ q → ¬same_partition p.1 p.2 q.1 q.2) ∧
    partitions.length = 4 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_partitions_l1034_103477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_domain_sum_l1034_103422

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_domain_sum (f : ℝ → ℝ) (a b : ℝ) :
  is_odd_function f →
  (∀ x : ℝ, f x ≠ 0 → x ∈ ({-1, 2, a, b} : Set ℝ)) →
  a + b = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_domain_sum_l1034_103422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_6_l1034_103482

-- Define the sequence T
def T : ℕ → ℕ
  | 0 => 11  -- Add a case for 0 to cover all natural numbers
  | 1 => 11
  | n + 2 => 11^(T (n + 1))

-- State the theorem
theorem t_50_mod_6 : T 50 ≡ 5 [ZMOD 6] := by
  sorry  -- Placeholder for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_6_l1034_103482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_theorem_l1034_103466

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- The foci of the ellipse -/
def F1 : Point := ⟨0, 2⟩
def F2 : Point := ⟨6, 2⟩

/-- The sum of distances from any point on the ellipse to the foci -/
def focal_distance_sum : ℝ := 10

/-- Predicate to check if a point is on the ellipse -/
def on_ellipse (E : Ellipse) (P : Point) : Prop :=
  distance P F1 + distance P F2 = focal_distance_sum

/-- Theorem stating the sum of center coordinates and axes lengths -/
theorem ellipse_sum_theorem (E : Ellipse) :
  (∀ (P : Point), on_ellipse E P) →
  E.center.x + E.center.y + E.a + E.b = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_theorem_l1034_103466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1034_103406

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) - 2 * x - a

-- Define the curve function
def curve (x : ℝ) : ℝ := x^3 + x + 1

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ,
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) 1 ∧
   f a (curve x₀) = curve x₀) →
  a ∈ Set.Icc (Real.exp (-3) - 9) (Real.exp 1 + 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1034_103406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_worked_36_hours_l1034_103499

/-- Payment structure for employees -/
structure PaymentStructure where
  base_rate : ℝ
  base_hours : ℝ
  overtime_rate : ℝ

/-- Calculate total pay given a payment structure and hours worked -/
noncomputable def calculate_pay (ps : PaymentStructure) (hours : ℝ) : ℝ :=
  if hours ≤ ps.base_hours then
    ps.base_rate * hours
  else
    ps.base_rate * ps.base_hours + ps.overtime_rate * (hours - ps.base_hours)

/-- Theorem stating that Harry worked 36 hours given the conditions -/
theorem harry_worked_36_hours (x : ℝ) (h_pos : x > 0) :
  let harry_ps := PaymentStructure.mk x 24 (1.5 * x)
  let james_ps := PaymentStructure.mk x 40 (2 * x)
  let james_hours := 41
  calculate_pay harry_ps 36 = calculate_pay james_ps james_hours := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_worked_36_hours_l1034_103499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_id_ends_5_most_representative_l1034_103419

/-- Represents a student in the middle school -/
structure Student :=
  (id : ℕ)
  (grade : ℕ)
  (gender : Bool)
  (plays_basketball : Bool)

/-- Represents the middle school -/
structure School :=
  (students : Set Student)

/-- Defines a sampling method -/
def SamplingMethod := School → Set Student

/-- Sampling method A: Third-grade students -/
def sample_third_grade : SamplingMethod := 
  λ school => { s ∈ school.students | s.grade = 3 }

/-- Sampling method B: All female students -/
def sample_female : SamplingMethod :=
  λ school => { s ∈ school.students | s.gender = true }

/-- Sampling method C: Students with student numbers ending in 5 -/
def sample_id_ends_5 : SamplingMethod :=
  λ school => { s ∈ school.students | s.id % 10 = 5 }

/-- Sampling method D: Students playing basketball -/
def sample_basketball : SamplingMethod :=
  λ school => { s ∈ school.students | s.plays_basketball = true }

/-- Defines representativeness of a sampling method -/
def is_most_representative (method : SamplingMethod) (school : School) : Prop :=
  ∀ other_method : SamplingMethod, 
    method school ≠ ∅ → 
    (∀ s ∈ school.students, s ∈ method school ↔ s.id % 10 = 5) →
    method school ⊆ other_method school → 
    method = other_method

/-- Theorem: Sampling students with ID ending in 5 is the most representative method -/
theorem id_ends_5_most_representative (school : School) : 
  is_most_representative sample_id_ends_5 school :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_id_ends_5_most_representative_l1034_103419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l1034_103479

/-- A regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- Volume of a regular tetrahedron -/
noncomputable def RegularTetrahedron.volume (t : RegularTetrahedron) : ℝ :=
  (1 / 12) * Real.sqrt 2 * t.edgeLength ^ 3

/-- Given two regular tetrahedrons in space with edge lengths in ratio 1:2, 
    their volume ratio is 1:8 -/
theorem tetrahedron_volume_ratio (t1 t2 : RegularTetrahedron) 
  (h : t1.edgeLength = (1 / 2) * t2.edgeLength) : 
  t1.volume = (1 / 8) * t2.volume := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_ratio_l1034_103479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_inverse_l1034_103496

theorem cube_sum_inverse (b : ℝ) (hb : b ≠ 0) :
  b^3 + b⁻¹^3 = (b + b⁻¹)^3 - 3*(b + b⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_inverse_l1034_103496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l1034_103437

/-- The time taken for two trains to pass each other -/
noncomputable def time_to_pass (speed_a speed_b length_a length_b : ℝ) : ℝ :=
  (length_a + length_b) / (speed_a - speed_b)

/-- Conversion factor from km/h to m/s -/
noncomputable def km_per_hour_to_m_per_sec : ℝ := 1000 / 3600

theorem trains_passing_time :
  let speed_a := 50 * km_per_hour_to_m_per_sec
  let speed_b := 40 * km_per_hour_to_m_per_sec
  let length_a := 125
  let length_b := 125.02
  abs (time_to_pass speed_a speed_b length_a length_b - 90) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l1034_103437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_sector_properties_l1034_103491

/-- Represents a circular sector --/
structure CircularSector where
  radius : ℝ
  arcLength : ℝ

/-- Calculates the central angle of a circular sector in radians --/
noncomputable def centralAngle (s : CircularSector) : ℝ :=
  s.arcLength / s.radius

/-- Calculates the area of a circular sector --/
noncomputable def sectorArea (s : CircularSector) : ℝ :=
  (1/2) * s.radius * s.arcLength

/-- Theorem about a specific circular sector --/
theorem specific_sector_properties :
  let s : CircularSector := { radius := 8, arcLength := 12 }
  centralAngle s = 3/2 ∧ sectorArea s = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_sector_properties_l1034_103491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_number_of_tilings_l1034_103441

/-- Represents a valid tiling of the lateral surface of a rectangular parallelepiped --/
def LateralSurfaceTiling (a b c : ℕ) : Type := Unit

/-- Counts the number of valid tilings for given dimensions --/
def countValidTilings (a b c : ℕ) : ℕ := sorry

/-- Proves that the number of tilings is even when c is odd --/
theorem even_number_of_tilings (a b c : ℕ) (h_c_odd : Odd c) :
  Even (countValidTilings a b c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_number_of_tilings_l1034_103441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1034_103463

-- Define the ellipse parameters
variable (a b : ℝ)

-- Define points A, B, and C
variable (A B C : ℝ × ℝ)

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- State the theorem
theorem ellipse_eccentricity :
  a > b ∧ b > 0 ∧  -- Ellipse condition
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ ∃ t : ℝ, (x, y) = (a * Real.cos t, b * Real.sin t)) ∧  -- Ellipse equation
  A = (-a, 0) ∧  -- Left vertex of ellipse
  B.2 = (1/3) * (B.1 + a) ∧  -- Line l equation
  C.1 = 0 ∧  -- C is on y-axis
  (B.1 - A.1) * (C.2 - A.2) = -(B.2 - C.2) * (C.1 - A.1) ∧  -- AC ⟂ BC
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2  -- |AC| = |BC|
  →
  eccentricity a b = Real.sqrt 6 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1034_103463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_locus_l1034_103400

-- Define the cube vertices
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def D : ℝ × ℝ × ℝ := (0, 1, 0)
def A' : ℝ × ℝ × ℝ := (0, 0, 1)

-- Define the edges
def edge_a : Set (ℝ × ℝ × ℝ) := {p | ∃ t, p = (1, 0, t) ∧ 0 ≤ t ∧ t ≤ 1}
def edge_b : Set (ℝ × ℝ × ℝ) := {p | ∃ t, p = (t, 1, 0) ∧ 0 ≤ t ∧ t ≤ 1}
def edge_c : Set (ℝ × ℝ × ℝ) := {p | ∃ t, p = (0, t, 1) ∧ 0 ≤ t ∧ t ≤ 1}

-- Define the distance function from a point to an edge
noncomputable def dist_to_edge (p : ℝ × ℝ × ℝ) (edge : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the property of being inside the cube
def inside_cube (p : ℝ × ℝ × ℝ) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2.1 ∧ p.2.1 ≤ 1 ∧ 0 ≤ p.2.2 ∧ p.2.2 ≤ 1

-- State the theorem
theorem equidistant_points_locus (p : ℝ × ℝ × ℝ) :
  inside_cube p →
  (dist_to_edge p edge_a = dist_to_edge p edge_b ∧
   dist_to_edge p edge_b = dist_to_edge p edge_c) ↔
  p.1 = p.2.1 ∧ p.2.1 = p.2.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_locus_l1034_103400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1034_103434

theorem right_triangle_hypotenuse (h : ℝ) : 
  let a := Real.log 512 / Real.log 9
  let b := Real.log 64 / Real.log 3
  a^2 + b^2 = h^2 → 
  (9 : ℝ)^h = 32768 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1034_103434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_slope_ranges_l1034_103407

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the center of the ellipse
def center : ℝ × ℝ := (0, 0)

-- Define the foci
def left_focus : ℝ × ℝ := (-3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Define the vertices
def left_vertex : ℝ × ℝ := (-5, 0)
def right_vertex : ℝ × ℝ := (5, 0)

-- Define a line through the center of the ellipse
def line_through_center (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define the perimeter of a triangle
noncomputable def triangle_perimeter (A B C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) +
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)

-- Define the slope of a line between two points
noncomputable def line_slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Theorem 1: Minimum perimeter of triangle PQF₂
theorem min_perimeter_triangle (P Q : ℝ × ℝ) (m : ℝ) :
  ellipse P.1 P.2 → ellipse Q.1 Q.2 →
  line_through_center m P.1 P.2 → line_through_center m Q.1 Q.2 →
  P ≠ Q →
  (∀ P' Q' : ℝ × ℝ, ellipse P'.1 P'.2 → ellipse Q'.1 Q'.2 →
    line_through_center m P'.1 P'.2 → line_through_center m Q'.1 Q'.2 →
    P' ≠ Q' →
    triangle_perimeter P' Q' right_focus ≥ triangle_perimeter P Q right_focus) →
  triangle_perimeter P Q right_focus = 18 := by
sorry

-- Theorem 2: Slope ranges
theorem slope_ranges (P : ℝ × ℝ) :
  ellipse P.1 P.2 →
  (2/5 ≤ line_slope P left_vertex ∧ line_slope P left_vertex ≤ 8/5) ↔
  (-8/5 ≤ line_slope P right_vertex ∧ line_slope P right_vertex ≤ -2/5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_slope_ranges_l1034_103407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_implies_k_value_l1034_103468

/-- The line equation kx + y + 4 = 0 --/
def line (k : ℝ) (x y : ℝ) : Prop := k * x + y + 4 = 0

/-- The circle equation x^2 + y^2 - 2y = 0 --/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

/-- The center of the circle --/
def circle_center : ℝ × ℝ := (0, 1)

/-- The radius of the circle --/
def circle_radius : ℝ := 1

/-- Point P is on the line --/
def point_on_line (k : ℝ) (P : ℝ × ℝ) : Prop := line k P.1 P.2

/-- PA and PB are tangents to the circle --/
noncomputable def tangents_to_circle (P A B : ℝ × ℝ) : Prop := sorry

/-- The area of quadrilateral PACB --/
noncomputable def area_PACB (P A C B : ℝ × ℝ) : ℝ := sorry

/-- The main theorem --/
theorem min_area_implies_k_value (k : ℝ) (P A B : ℝ × ℝ) :
  k > 0 →
  point_on_line k P →
  tangents_to_circle P A B →
  (∀ P' A' B', point_on_line k P' → tangents_to_circle P' A' B' →
    area_PACB P' A' circle_center B' ≥ 2) →
  (∃ P' A' B', point_on_line k P' ∧ tangents_to_circle P' A' B' ∧
    area_PACB P' A' circle_center B' = 2) →
  k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_implies_k_value_l1034_103468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1034_103497

-- Define the random variable x
noncomputable def x : ℝ := sorry

-- Define the conditions
axiom x_range : 50 ≤ x ∧ x ≤ 250
axiom x_floor_sqrt : ⌊Real.sqrt x⌋ = 14

-- Define the event
def event : Prop := ⌊Real.sqrt (50 * x)⌋ = 105

-- Define the probability function
noncomputable def prob : ℝ := sorry

-- State the theorem
theorem probability_theorem : prob = 193 / 1450 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1034_103497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l1034_103413

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The sum of volumes of three spheres with radii 2, 3, and 5 -/
noncomputable def total_volume : ℝ := sphere_volume 2 + sphere_volume 3 + sphere_volume 5

theorem snowman_volume : total_volume = (640 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l1034_103413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_condition_l1034_103423

/-- The eccentricity of a hyperbola with equation x^2 - y^2/m = 1 -/
noncomputable def eccentricity (m : ℝ) : ℝ := Real.sqrt (1 + m)

/-- Theorem stating that m > 2 is a sufficient but not necessary condition for 
    the eccentricity of the hyperbola x^2 - y^2/m = 1 to be greater than √2 -/
theorem hyperbola_eccentricity_condition (m : ℝ) (h : m > 0) :
  (m > 2 → eccentricity m > Real.sqrt 2) ∧
  ¬(eccentricity m > Real.sqrt 2 → m > 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_condition_l1034_103423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trig_identity_l1034_103430

open Real

theorem right_triangle_trig_identity (A B : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : A + B = π / 2) :
  let C := π / 2
  Real.sin A * Real.sin B * Real.sin (A - B) + Real.sin B * Real.sin C * Real.sin (B - C) + 
  Real.sin C * Real.sin A * Real.sin (C - A) + Real.sin (A - B) * Real.sin (B - C) * Real.sin (C - A) = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trig_identity_l1034_103430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_proof_l1034_103488

theorem exponent_proof (number : ℚ) (x : ℕ) 
  (h1 : number * (1/4)^2 = 4^x) 
  (h2 : 4^x = 1024) : 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_proof_l1034_103488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l1034_103425

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem stating the value of n that satisfies the given conditions -/
theorem geometric_series_problem (a₁ a₂ : ℝ) (n : ℝ) :
  a₁ = 12 →
  a₂ = 12 →
  let r₁ := 1 / 3
  let r₂ := (4 + n) / 12
  infiniteGeometricSeriesSum a₂ r₂ = 4 * infiniteGeometricSeriesSum a₁ r₁ →
  n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l1034_103425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_bisector_angles_l1034_103452

/-- A point in 2D space -/
structure Point : Type where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line : Type where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangle in 2D space -/
structure Triangle : Type where
  A : Point
  B : Point
  C : Point
  angle : ℝ

/-- Angle bisector of an angle -/
def AngleBisector (θ : ℝ) (l : Line) : Prop := sorry

/-- Membership of a point on a line -/
def PointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Given a triangle ABC with angles A, B, C and angle bisectors intersecting at point M,
    the angles of the six smaller triangles formed by the angle bisectors are as stated. -/
theorem triangle_angle_bisector_angles (A B C : ℝ) (h_sum : A + B + C = π) :
  let angles_small_triangles := [
    A/2, A/2 + C/2, B + C/2,
    B/2, B/2 + A/2, C + A/2,
    C/2, C/2 + B/2, A + B/2
  ]
  ∃ (M : Point) (AA₁ BB₁ CC₁ : Line),
    AngleBisector A AA₁ ∧ AngleBisector B BB₁ ∧ AngleBisector C CC₁ ∧
    PointOnLine M AA₁ ∧ PointOnLine M BB₁ ∧ PointOnLine M CC₁ ∧
    (∀ θ ∈ angles_small_triangles, ∃ T : Triangle, T.angle = θ ∨ T.angle = π - θ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_bisector_angles_l1034_103452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_bound_exists_configuration_exceeding_bound_l1034_103471

/-- A set of points in a plane forming a convex n-gon -/
structure ConvexPolygon (n : ℕ) where
  points : Fin n → ℝ × ℝ
  convex : Convex ℝ (Set.range points)

/-- The number of equilateral triangles with side length 1 in a given ConvexPolygon -/
noncomputable def count_equilateral_triangles (p : ConvexPolygon n) : ℕ := sorry

/-- Theorem: The number of equilateral triangles is less than 2n/3 -/
theorem equilateral_triangle_bound {n : ℕ} (h : n > 3) (p : ConvexPolygon n) :
  count_equilateral_triangles p < 2 * n / 3 := by sorry

/-- Theorem: There exists a configuration where the ratio of equilateral triangles to n exceeds 0.666 -/
theorem exists_configuration_exceeding_bound (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℕ) (p : ConvexPolygon n), 
    (count_equilateral_triangles p : ℝ) / n > 0.666 - ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_bound_exists_configuration_exceeding_bound_l1034_103471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_three_in_30_factorial_is_14_l1034_103451

/-- The exponent of 3 in the prime factorization of 30! -/
def exponent_of_three_in_30_factorial : ℕ :=
  (30 / 3) + (30 / 9) + (30 / 27)

/-- Theorem stating that the exponent of 3 in the prime factorization of 30! is 14 -/
theorem exponent_of_three_in_30_factorial_is_14 :
  exponent_of_three_in_30_factorial = 14 := by
  rfl

#eval exponent_of_three_in_30_factorial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_three_in_30_factorial_is_14_l1034_103451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1034_103487

theorem sin_theta_value (θ : ℝ) (h : Real.sin (π/4 - θ/2) = 2/3) : Real.sin θ = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1034_103487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recommendation_methods_count_l1034_103426

/-- Represents a university --/
inductive University
| tsinghua
| peking
| fudan

/-- Represents a student's gender --/
inductive Gender
| male
| female

/-- Represents the number of spots available for each university --/
def spots : University → Nat
| University.tsinghua => 2
| University.peking => 2
| University.fudan => 1

/-- Represents the number of candidates for each gender --/
def candidates : Gender → Nat
| Gender.male => 3
| Gender.female => 2

/-- Represents whether a university requires male students --/
def requires_male : University → Bool
| University.tsinghua => true
| University.peking => true
| University.fudan => false

/-- Theorem: The total number of different recommendation methods is 24 --/
theorem recommendation_methods_count : 
  (∀ u, spots u > 0) →
  (∀ g, candidates g > 0) →
  (spots University.tsinghua + spots University.peking + spots University.fudan = 
   candidates Gender.male + candidates Gender.female) →
  (∃ u, requires_male u) →
  (∃ count : Nat, count = 24 ∧ 
   count = (Nat.choose 3 1 * Nat.choose 2 1 * Nat.choose 1 1 + 
            Nat.choose 3 1 * 2) * Nat.choose 2 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recommendation_methods_count_l1034_103426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_coloring_count_l1034_103449

/-- Represents a 7x7 chessboard -/
def Chessboard := Fin 7 × Fin 7

/-- A coloring method is a pair of distinct squares on the chessboard -/
def ColoringMethod := { pair : Chessboard × Chessboard // pair.1 ≠ pair.2 }

/-- Rotations of the chessboard -/
noncomputable def rotations : List (Chessboard → Chessboard) := sorry

/-- Two coloring methods are equivalent if one can be obtained by rotating the other -/
def equivalent (m1 m2 : ColoringMethod) : Prop :=
  ∃ r ∈ rotations, (r m1.val.1, r m1.val.2) = m2.val ∨ (r m1.val.2, r m1.val.1) = m2.val

/-- The set of all unique coloring methods -/
noncomputable def uniqueColorings : Finset ColoringMethod :=
  sorry

/-- The theorem stating that there are 300 unique coloring methods -/
theorem unique_coloring_count :
  Finset.card uniqueColorings = 300 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_coloring_count_l1034_103449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_12_13_5_l1034_103412

/-- The area of a triangle given its three side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 12, 13, and 5 is 30 -/
theorem triangle_area_12_13_5 : triangleArea 12 13 5 = 30 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_12_13_5_l1034_103412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_existence_l1034_103464

def S : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem unique_k_existence : ∃! k : ℕ, 
  (Finset.filter (fun p : ℕ × ℕ => p.1 = k * p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S) (Finset.product S S)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_k_existence_l1034_103464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_symmetry_angle_equality_l1034_103460

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A trapezoid with vertices A, B, C, D -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point
  is_trapezoid : True  -- We assume this property holds

/-- The intersection point of the diagonals -/
def diagonalIntersection (t : Trapezoid) : Point :=
  sorry

/-- The angle bisector of ∠BOC -/
def angleBisector (t : Trapezoid) : Point → Point :=
  sorry

/-- Symmetric point with respect to a line -/
def symmetricPoint (p : Point) (l : Point → Point) : Point :=
  sorry

/-- Angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

theorem trapezoid_symmetry_angle_equality (t : Trapezoid) : 
  let O := diagonalIntersection t
  let bisector := angleBisector t
  let B' := symmetricPoint t.B bisector
  let C' := symmetricPoint t.C bisector
  angle C' t.A t.C = angle B' t.D t.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_symmetry_angle_equality_l1034_103460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangements_bound_l1034_103404

def n : ℕ := 2015^100

-- B(n) is the number of ways to arrange n identical cubes into one or more congruent rectangular solids
def B (n : ℕ) : ℕ := sorry

theorem cube_arrangements_bound :
  10^14 < B n ∧ B n < 10^15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangements_bound_l1034_103404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l1034_103495

/-- The set of digits that can be used to form the numbers -/
def digits : Finset ℕ := {0, 1, 2, 3, 4}

/-- A function that checks if a number is odd -/
def isOdd (n : ℕ) : Bool := n % 2 = 1

/-- A function that generates all valid three-digit odd numbers using the given digits -/
def validNumbers : Finset ℕ := 
  Finset.filter (λ n => n ≥ 100 ∧ n < 1000 ∧ 
                        (n / 100) ∈ digits ∧ 
                        ((n / 10) % 10) ∈ digits ∧ 
                        (n % 10) ∈ digits ∧
                        isOdd n) 
                (Finset.range 1000)

/-- The theorem stating that the sum of all valid numbers is 10880 -/
theorem sum_of_valid_numbers : (Finset.sum validNumbers id) = 10880 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l1034_103495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1034_103420

noncomputable def f (x : ℝ) := Real.sin x ^ 4 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 4

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), f x ≥ -2) ∧
  (MonotoneOn f (Set.Icc 0 (Real.pi / 3))) ∧
  (MonotoneOn f (Set.Icc (5 * Real.pi / 6) Real.pi)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1034_103420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_distance_sum_l1034_103443

/-- Parabola defined by y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- Distance from a point to the origin -/
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- The four intersection points of the circle and parabola -/
noncomputable def intersection_points : Finset (ℝ × ℝ) := {(3, 9), (1, 1), (-4, 16), (0, 0)}

/-- Sum of distances from the focus to all intersection points -/
noncomputable def sum_of_distances : ℝ := 
  Finset.sum intersection_points (λ p => distance_to_origin p.1 p.2)

theorem parabola_circle_intersection_distance_sum :
  sum_of_distances = Real.sqrt 90 + Real.sqrt 2 + Real.sqrt 272 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_distance_sum_l1034_103443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_satisfied_l1034_103438

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a r : ℝ) : ℝ := a / (1 - r)

/-- The left-hand side of the equation -/
noncomputable def lhs : ℝ := (geometric_sum 1 (1/3)) * (geometric_sum 1 (-1/3))

/-- The right-hand side of the equation -/
noncomputable def rhs (y : ℝ) : ℝ := geometric_sum 1 (1/y)

/-- The theorem stating that the equation is satisfied when y = 9 -/
theorem equation_satisfied : lhs = rhs 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_satisfied_l1034_103438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_distance_7_jumps_l1034_103417

/-- The total distance traveled by Laura after n jumps -/
noncomputable def laura_distance (n : ℕ) : ℝ :=
  10 * (1 - (3/4)^n)

/-- The theorem stating that Laura's distance after 7 jumps is approximately 7.67 -/
theorem laura_distance_7_jumps :
  abs (laura_distance 7 - 7.67) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_distance_7_jumps_l1034_103417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1034_103409

/-- Circle C with equation (x+1)^2 + (y-2)^2 = 2 -/
def circleC (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 2

/-- Point P is outside the circle -/
def outside_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 > 2

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- |PM| = |PO| condition -/
def equal_distances (x y : ℝ) : Prop :=
  distance x y (-1) 2 = distance x y 0 0

theorem min_distance_point :
  ∃ (x y : ℝ),
    outside_circle x y ∧
    equal_distances x y ∧
    (∀ (x' y' : ℝ), outside_circle x' y' ∧ equal_distances x' y' →
      distance x y (-1) 2 ≤ distance x' y' (-1) 2) ∧
    x = -3/10 ∧ y = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1034_103409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l1034_103486

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 3]
def b : Fin 2 → ℝ := ![-1, 2]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define the magnitude of a vector
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

-- Define the projection
noncomputable def projection (v w : Fin 2 → ℝ) : ℝ := (dot_product v w) / (magnitude w)

-- Theorem statement
theorem projection_a_on_b :
  projection a b = (4 * Real.sqrt 5) / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l1034_103486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_f_l1034_103475

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

-- State the theorem
theorem min_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 1 2, Monotone (fun x => f a x)) →
  a ≥ Real.exp (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_f_l1034_103475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1034_103416

/-- Triangle PQR with vertices P(1, 1), Q(4, 1), and R(3, 4) -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The specific triangle PQR from the problem -/
def trianglePQR : Triangle :=
  { P := (1, 1)
    Q := (4, 1)
    R := (3, 4) }

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let (x1, y1) := t.P
  let (x2, y2) := t.Q
  let (x3, y3) := t.R
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem stating that the area of triangle PQR is 9/2 -/
theorem area_of_triangle_PQR : triangleArea trianglePQR = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1034_103416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rosencrantz_win_probability_l1034_103489

/-- Sequence definition for a_n --/
def a : ℕ → ℕ
  | 0 => 4  -- Adding case for 0 to cover all natural numbers
  | 1 => 4
  | 2 => 3
  | n + 3 => a (n + 2) + a (n + 1)

/-- The game outcome after n flips --/
noncomputable def game_outcome (n : ℕ) : ℝ := sorry

/-- The probability of the first player ending up with more money after n flips --/
noncomputable def win_probability (n : ℕ) : ℝ := sorry

/-- The main theorem --/
theorem rosencrantz_win_probability :
  win_probability 2010 = 1/2 - 1/2^1341 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rosencrantz_win_probability_l1034_103489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merger_in_2015_l1034_103436

/-- Market share of Company A in the n-th year -/
noncomputable def market_share_A (A : ℝ) (n : ℕ) : ℝ := (A / 40) * (n^2 - n + 40)

/-- Market share of Company B in the n-th year -/
noncomputable def market_share_B (A : ℝ) (n : ℕ) : ℝ := A * (2 - 1 / 2^(n - 1))

/-- Merger condition: true if Company B should be merged into Company A -/
def should_merge (A : ℝ) (n : ℕ) : Prop :=
  market_share_B A n < 0.2 * market_share_A A n

theorem merger_in_2015 (A : ℝ) (h : A > 0) : should_merge A 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merger_in_2015_l1034_103436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l1034_103498

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | (x - 3) / (4 - x) < 0}

-- State the theorem
theorem complement_of_M : Set.compl M = Set.Icc 3 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l1034_103498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_range_l1034_103492

open Real

/-- The function f(x) = (1/2)x^2 - a*ln(x) + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * log x + 1

/-- f(x) has a minimum value in the interval (0,1) -/
def has_min_in_interval (a : ℝ) : Prop :=
  ∃ x₀ ∈ Set.Ioo 0 1, ∀ x ∈ Set.Ioo 0 1, f a x₀ ≤ f a x

theorem min_value_implies_a_range (a : ℝ) :
  has_min_in_interval a → 0 < a ∧ a < 1 := by
  sorry

#check min_value_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_range_l1034_103492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_proof_l1034_103445

theorem integer_sum_proof (x y : ℤ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x - y = 8) (h4 : x * y = 144) : x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_proof_l1034_103445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_circumcenter_orthocenter_collinear_l1034_103459

/-- Non-equilateral triangle -/
structure NonEquilateralTriangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)
  not_equilateral : ¬ (dist A B = dist B C ∧ dist B C = dist C A)

/-- Incenter of a triangle -/
noncomputable def incenter (t : NonEquilateralTriangle) : EuclideanSpace ℝ (Fin 2) := sorry

/-- Circumcenter of a triangle -/
noncomputable def circumcenter (t : NonEquilateralTriangle) : EuclideanSpace ℝ (Fin 2) := sorry

/-- Tangency points of the incircle with the sides of the triangle -/
noncomputable def tangency_points (t : NonEquilateralTriangle) : 
  (EuclideanSpace ℝ (Fin 2)) × (EuclideanSpace ℝ (Fin 2)) × (EuclideanSpace ℝ (Fin 2)) := sorry

/-- Orthocenter of a triangle -/
noncomputable def orthocenter (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

/-- Collinearity of three points -/
def collinear (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

theorem incenter_circumcenter_orthocenter_collinear 
  (t : NonEquilateralTriangle) : 
  let I := incenter t
  let O := circumcenter t
  let (T₁, T₂, T₃) := tangency_points t
  let H := orthocenter T₁ T₂ T₃
  collinear I O H := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_circumcenter_orthocenter_collinear_l1034_103459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydra_defeat_bound_l1034_103490

/-- A Hydra is represented as an undirected graph -/
structure Hydra where
  vertices : Finset ℕ
  edges : Finset (Prod ℕ ℕ)
  edge_count : edges.card = 100

/-- Represents a single strike on the Hydra -/
def strike (H : Hydra) (v : ℕ) : Hydra :=
  sorry

/-- Checks if the Hydra is defeated (disconnected into two parts) -/
def is_defeated (H : Hydra) : Prop :=
  sorry

/-- The main theorem: Any Hydra with 100 necks can be defeated in at most 10 strikes -/
theorem hydra_defeat_bound (H : Hydra) :
  ∃ (strikes : List ℕ), strikes.length ≤ 10 ∧ is_defeated (strikes.foldl strike H) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydra_defeat_bound_l1034_103490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_a_charge_first_8_minutes_l1034_103428

/-- Represents the charge for a phone call under a given plan -/
structure PhoneCharge where
  fixedCharge : ℝ  -- Fixed charge for the first 8 minutes
  perMinuteRate : ℝ  -- Rate per minute after 8 minutes
  duration : ℝ  -- Duration of the call in minutes

/-- Calculate the total charge for a call -/
noncomputable def totalCharge (c : PhoneCharge) : ℝ :=
  if c.duration ≤ 8 then c.fixedCharge
  else c.fixedCharge + (c.duration - 8) * c.perMinuteRate

/-- Theorem stating the charge for the first 8 minutes under Plan A -/
theorem plan_a_charge_first_8_minutes :
  ∃ (x : ℝ),
    (∀ (d : ℝ), d ≤ 8 → totalCharge { fixedCharge := x, perMinuteRate := 0.06, duration := d } = x) ∧
    (totalCharge { fixedCharge := x, perMinuteRate := 0.06, duration := 6 } =
     totalCharge { fixedCharge := 0, perMinuteRate := 0.08, duration := 6 }) ∧
    x = 0.48 := by
  sorry

#check plan_a_charge_first_8_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_a_charge_first_8_minutes_l1034_103428
