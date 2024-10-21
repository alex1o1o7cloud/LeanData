import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l87_8778

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a - 1/2) * x else a^x - 4

-- State the theorem
theorem range_of_a_for_increasing_f :
  (∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0)) ↔
  (∃ a : ℝ, 1 < a ∧ a ≤ 3) :=
sorry

-- Additional lemmas that might be useful for the proof
lemma f_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  (a - 1/2 > 0 ∧ a > 1 ∧ a^2 - 4 ≤ 2*(a - 1/2)) :=
sorry

lemma range_condition (a : ℝ) :
  (a - 1/2 > 0 ∧ a > 1 ∧ a^2 - 4 ≤ 2*(a - 1/2)) →
  (1 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l87_8778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_rectangle_l87_8750

/-- Given a rectangle with dimensions 6 by 8 and points A, B, C on its sides,
    prove that the area of triangle ABC is 36.5 square units. -/
theorem triangle_area_in_rectangle : 
  ∀ (A B C : ℝ × ℝ),
    -- Rectangle dimensions
    let width : ℝ := 6
    let height : ℝ := 8
    -- Point A on the left side
    A.1 = 0 ∧ A.2 = 2 →
    -- Point B on the bottom side
    B.1 = 5 ∧ B.2 = 0 →
    -- Point C on the right side
    C.1 = width ∧ C.2 = height - 3 →
    -- Area of triangle ABC
    (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) = 36.5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_rectangle_l87_8750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sausage_pieces_l87_8794

noncomputable def sausage_length : ℝ := 34.29
noncomputable def piece1_length : ℝ := 3/5
noncomputable def piece2_length : ℝ := 7/8

theorem sausage_pieces : 
  ∃ (n : ℕ), n = 46 ∧ 
  (n : ℝ) * (piece1_length + piece2_length) ≤ sausage_length ∧
  ((n + 2) : ℝ) * (piece1_length + piece2_length) > sausage_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sausage_pieces_l87_8794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_credit_conversion_l87_8715

/-- Given that P cans of soda can be charged for C credit points,
    and 3 credit points equals 2 cans of soda,
    prove that R credit points can be charged for PR/C cans of soda. -/
theorem soda_credit_conversion (P C R : ℚ) (h1 : P > 0) (h2 : C > 0) (h3 : R > 0) :
  (2 / 3 : ℚ) * R = P * R / C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_credit_conversion_l87_8715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l87_8724

-- Define the curves
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := ((2 + t) / 6, Real.sqrt t)
noncomputable def C₂ (s : ℝ) : ℝ × ℝ := (-(2 + s) / 6, -Real.sqrt s)
noncomputable def C₃ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- State the theorem
theorem curve_intersections :
  -- 1. Cartesian equation of C₁
  (∀ x y, y ≥ 0 → (∃ t, C₁ t = (x, y)) ↔ y^2 = 6*x - 2) ∧
  -- 2. Intersection points of C₃ with C₁
  (∃ θ₁ θ₂, C₃ θ₁ = (1/2, 1) ∧ C₃ θ₂ = (1, 2)) ∧
  -- 3. Intersection points of C₃ with C₂
  (∃ θ₃ θ₄, C₃ θ₃ = (-1/2, -1) ∧ C₃ θ₄ = (-1, -2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l87_8724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_albums_count_l87_8743

/-- Represents the number of albums in a person's collection -/
def Collection := Nat

/-- The number of albums shared by all three collections -/
def shared_albums : Nat := 12

/-- Andrew's collection size -/
def andrew_collection : Nat := 25

/-- Number of albums unique to John -/
def john_unique : Nat := 8

/-- Number of albums unique to Mary -/
def mary_unique : Nat := 5

/-- The number of albums in either Andrew's, John's, or Mary's collection, but not shared among all three -/
def unique_albums : Nat := andrew_collection - shared_albums + john_unique + mary_unique

theorem unique_albums_count : unique_albums = 26 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_albums_count_l87_8743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_player_wins_l87_8733

/-- Represents a rectangular table --/
structure Table where
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

/-- Represents a coin placement on the table --/
structure CoinPlacement where
  x : ℝ
  y : ℝ

/-- Represents the state of the game --/
structure GameState where
  table : Table
  placements : List CoinPlacement

/-- Represents a player --/
inductive Player
  | First
  | Second

/-- Represents a player's strategy --/
def Strategy := GameState → Option CoinPlacement

/-- Checks if a coin placement is valid --/
def is_valid_placement (state : GameState) (placement : CoinPlacement) : Prop :=
  placement.x ≥ 0 ∧ placement.x ≤ state.table.width ∧
  placement.y ≥ 0 ∧ placement.y ≤ state.table.height ∧
  ∀ p ∈ state.placements, (placement.x - p.x)^2 + (placement.y - p.y)^2 ≥ 1

/-- Defines the winning strategy for the starting player --/
noncomputable def winning_strategy : Strategy :=
  fun state => sorry

/-- Simulates playing the game --/
noncomputable def play_game (table : Table) (strategy1 strategy2 : Strategy) : Player :=
  sorry

/-- Theorem stating that the starting player always wins --/
theorem starting_player_wins (table : Table) :
  ∃ (strategy : Strategy), ∀ (opponent_strategy : Strategy),
    play_game table strategy opponent_strategy = Player.First := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_player_wins_l87_8733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l87_8796

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (((x + y) / (2 * z)) ^ (1/3 : ℝ)) + (((y + z) / (2 * x)) ^ (1/3 : ℝ)) + (((z + x) / (2 * y)) ^ (1/3 : ℝ)) ≤ (5 * (x + y + z) + 9) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l87_8796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_l87_8746

theorem divisible_by_nine (n : ℕ) : ∃ k : ℤ, (2^(2*n - 1) : ℤ) + 3*n + 4 = 9*k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_l87_8746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_english_failure_percentage_l87_8707

theorem english_failure_percentage 
  (failed_hindi : ℝ) 
  (failed_both : ℝ) 
  (passed_both : ℝ) 
  (h1 : failed_hindi = 25)
  (h2 : failed_both = 25)
  (h3 : passed_both = 50) : 
  ℝ := by
  -- The percentage of students who failed in English
  exact 50


end NUMINAMATH_CALUDE_ERRORFEEDBACK_english_failure_percentage_l87_8707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l87_8755

/-- Given a line l: x - y + 1 = 0 and a point P(1, -1), 
    this theorem proves that x + y = 0 is the equation of the line 
    passing through P and perpendicular to l. -/
theorem perpendicular_line_through_point 
  (l : ℝ → ℝ → Prop) 
  (hl : ∀ x y, l x y ↔ x - y + 1 = 0) 
  (P : ℝ × ℝ) 
  (hP : P = (1, -1)) : 
  ∃ m b, (∀ x y, y = m * x + b ↔ x + y = 0) ∧ 
         (m * 1 = -1) ∧
         (P.1 + P.2 = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l87_8755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revolution_volume_of_unit_cube_l87_8711

/-- A unit cube in 3D space -/
structure UnitCube where
  vertices : Fin 8 → Fin 3 → ℝ
  is_unit_cube : ∀ i, vertices i ∈ List.map (fun (x, y, z) => (fun j => 
    match j with
    | 0 => x
    | 1 => y
    | 2 => z
    | _ => 0
  )) [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

/-- The volume of the solid generated by revolving a unit cube around its main diagonal -/
noncomputable def revolutionVolume (cube : UnitCube) : ℝ := sorry

/-- Theorem stating that the volume of the solid generated by revolving a unit cube 
    around its main diagonal is (15 + 4√3)π / 27 -/
theorem revolution_volume_of_unit_cube (cube : UnitCube) :
  revolutionVolume cube = (15 + 4 * Real.sqrt 3) * π / 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revolution_volume_of_unit_cube_l87_8711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equation_solution_l87_8740

noncomputable def f (x : ℝ) : ℝ := 
  let rec aux (n : ℕ) : ℝ := 
    if n = 0 then x else Real.sqrt (x + aux (n-1))
  aux 1000  -- Approximate the infinite nested radical with a large finite depth

noncomputable def g (x : ℝ) : ℝ := 
  let rec aux (n : ℕ) : ℝ := 
    if n = 0 then x else Real.sqrt (x - aux (n-1))
  aux 1000  -- Approximate the infinite nested radical with a large finite depth

theorem nested_radical_equation_solution :
  ∃! x : ℝ, x > 0 ∧ f x = g x ∧ x = (3 + Real.sqrt 5) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equation_solution_l87_8740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_distance_l87_8700

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a tangent line to the circle
def TangentLine (c : Circle) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | ∃ (m : ℝ), q.2 - p.2 = m * (q.1 - p.1) ∧
       PointOnCircle c p ∧
       ∀ (x y : ℝ), (x - p.1)^2 + (y - p.2)^2 = c.radius^2 → y - p.2 = m * (x - p.1)}

-- Define the distance between a point and a line
noncomputable def DistancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

-- State the theorem
theorem tangent_circle_distance (c : Circle) (a m : ℝ × ℝ) :
  PointOnCircle c a →
  PointOnCircle c m →
  a ∈ TangentLine c a →
  let d := 2 * c.radius
  let D := DistancePointToLine m (TangentLine c a)
  (m.1 - a.1)^2 + (m.2 - a.2)^2 = d * D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_distance_l87_8700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthographic_projection_of_rhombus_not_rhombus_l87_8769

/-- A rhombus in 2D space -/
structure Rhombus where
  vertices : Fin 4 → ℝ × ℝ
  is_rhombus : ∀ i j : Fin 4, i ≠ j → 
    ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 
    ((vertices ((i + 1) % 4)).1 - (vertices ((j + 1) % 4)).1)^2 + 
    ((vertices ((i + 1) % 4)).2 - (vertices ((j + 1) % 4)).2)^2

/-- Orthographic projection function -/
noncomputable def orthographic_projection (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 / 2)

/-- The theorem stating that the orthographic projection of a rhombus is not a rhombus -/
theorem orthographic_projection_of_rhombus_not_rhombus (R : Rhombus) :
  ¬ (∃ R' : Rhombus, ∀ i : Fin 4, R'.vertices i = orthographic_projection (R.vertices i)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthographic_projection_of_rhombus_not_rhombus_l87_8769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_odd_with_pi_period_l87_8744

noncomputable def f_A (x : ℝ) := Real.cos (3 * Real.pi / 2 - 2 * x)
noncomputable def f_B (x : ℝ) := abs (Real.cos x)
noncomputable def f_C (x : ℝ) := Real.sin (Real.pi / 2 + 2 * x)
noncomputable def f_D (x : ℝ) := abs (Real.sin x)

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) :=
  has_period f p ∧ p > 0 ∧ ∀ q, (has_period f q ∧ q > 0) → p ≤ q

theorem unique_odd_with_pi_period :
  (is_odd f_A ∧ smallest_positive_period f_A Real.pi) ∧
  ¬(is_odd f_B ∧ smallest_positive_period f_B Real.pi) ∧
  ¬(is_odd f_C ∧ smallest_positive_period f_C Real.pi) ∧
  ¬(is_odd f_D ∧ smallest_positive_period f_D Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_odd_with_pi_period_l87_8744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_range_l87_8762

theorem isosceles_triangle_side_length_range 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (isosceles : dist A B = dist A C) 
  (base_length : dist B C = 10) 
  (perimeter_bound : dist A B + dist A C + dist B C ≤ 44) :
  5 < dist A B ∧ dist A B ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_range_l87_8762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l87_8704

-- Define the points A, B, C
noncomputable def A : ℝ × ℝ := (2, 4)
noncomputable def B : ℝ × ℝ := (1, -3)
noncomputable def C : ℝ × ℝ := (-2, 1)

-- Define the midpoint D of AC
noncomputable def D : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the equation of a line ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the altitude from A to BC
noncomputable def altitude : Line := { a := 3, b := -4, c := 10 }

-- Define the area of triangle DBC
noncomputable def area_DBC : ℝ := 25 / 4

-- Theorem statement
theorem triangle_properties :
  (altitude.a * A.1 + altitude.b * A.2 + altitude.c = 0) ∧
  (altitude.a * (B.1 - C.1) + altitude.b * (B.2 - C.2) = 0) ∧
  (2 * area_DBC = abs ((B.1 - D.1) * (C.2 - D.2) - (C.1 - D.1) * (B.2 - D.2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l87_8704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_pi_squared_over_two_l87_8710

open Real

/-- The function representing the curve y = x sin x --/
noncomputable def f (x : ℝ) : ℝ := x * sin x

/-- The derivative of f --/
noncomputable def f' (x : ℝ) : ℝ := sin x + x * cos x

/-- The slope of the tangent line at x = -π/2 --/
noncomputable def m : ℝ := f' (-π/2)

/-- The y-intercept of the tangent line --/
noncomputable def b : ℝ := π/2 + m * (π/2)

/-- The x-intercept of the tangent line --/
noncomputable def x_intercept : ℝ := -b / m

theorem triangle_area_is_pi_squared_over_two :
  (1/2) * π * x_intercept = π^2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_pi_squared_over_two_l87_8710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_point_T_l87_8798

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x : ℝ) : ℝ := k * (x - 1)

-- Define the condition for the line not being perpendicular to coordinate axes
def line_not_perpendicular (k : ℝ) : Prop := k ≠ 0

-- Define the intersection points P and Q
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, ellipse x y ∧ y = line_through_focus k x ∧ p = (x, y)}

-- Define the point T
def point_T (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the dot product condition
def dot_product_condition (P Q T : ℝ × ℝ) : Prop :=
  let QP := (P.1 - Q.1, P.2 - Q.2)
  let TP := (P.1 - T.1, P.2 - T.2)
  let PQ := (Q.1 - P.1, Q.2 - P.2)
  let TQ := (Q.1 - T.1, Q.2 - T.2)
  QP.1 * TP.1 + QP.2 * TP.2 = PQ.1 * TQ.1 + PQ.2 * TQ.2

-- The main theorem
theorem ellipse_intersection_point_T :
  ∃ t : ℝ, t ∈ Set.Ioo 0 (1/4) ∧
  ∀ k : ℝ, line_not_perpendicular k →
  ∃ P Q : ℝ × ℝ, P ∈ intersection_points k ∧ Q ∈ intersection_points k ∧
  dot_product_condition P Q (point_T t) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_point_T_l87_8798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_18_hours_l87_8745

/-- Represents the journey described in the problem -/
structure Journey where
  total_distance : ℝ
  car_speed : ℝ
  walk_speed : ℝ

/-- Calculates the total time of the journey -/
noncomputable def journey_time (j : Journey) (d1 d2 : ℝ) : ℝ :=
  d1 / j.car_speed + (j.total_distance - d1) / j.walk_speed

/-- Theorem stating that the journey time is 18 hours -/
theorem journey_time_is_18_hours (j : Journey) 
  (h1 : j.total_distance = 150)
  (h2 : j.car_speed = 30)
  (h3 : j.walk_speed = 4) :
  ∃ d1 d2 : ℝ, 
    d1 > 0 ∧ 
    d2 > 0 ∧ 
    d1 + d2 < j.total_distance ∧
    journey_time j d1 d2 = 18 ∧
    journey_time j (d1 + d2) 0 = 18 ∧
    d2 / j.walk_speed + (j.total_distance - (d1 + d2)) / j.walk_speed = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_18_hours_l87_8745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_count_l87_8787

theorem triangle_third_side_count :
  let side1 : ℝ := 5
  let side2 : ℝ := 7
  let possible_lengths := Finset.filter (fun n : ℕ => 
    n > 2 ∧ 
    n < 12 ∧ 
    side1 + n > side2 ∧
    side2 + n > side1 ∧
    side1 + side2 > n) (Finset.range 12)
  Finset.card possible_lengths = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_count_l87_8787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_total_journey_time_l87_8741

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ → Prop := sorry

/-- The speed of the river current in km/hr -/
def river_speed : ℝ := 2

/-- The distance traveled upstream and downstream in km -/
def distance : ℝ := 40

/-- The total journey time in hours -/
def total_time : ℝ := 15

/-- Theorem stating that the boat's speed in still water is 6 km/hr -/
theorem boat_speed_in_still_water :
  boat_speed 6 := by sorry

/-- Helper function to calculate the time taken for a leg of the journey -/
noncomputable def journey_leg_time (boat_speed : ℝ) (is_upstream : Bool) : ℝ :=
  distance / (boat_speed + if is_upstream then -river_speed else river_speed)

/-- Theorem stating that the total journey time equals 15 hours -/
theorem total_journey_time (b : ℝ) (h : boat_speed b) :
  journey_leg_time b true + journey_leg_time b false = total_time := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_total_journey_time_l87_8741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l87_8764

open Real

-- Define the type for a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the fixed points
def F1 : Point2D := ⟨-3, 0⟩
def F2 : Point2D := ⟨3, 0⟩

-- Define the condition for point P
def satisfiesCondition (P : Point2D) : Prop :=
  distance P F1 + distance P F2 = 6

-- Define what it means for a point to be on the line segment between F1 and F2
def onLineSegment (P : Point2D) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.x = -3 + 6*t ∧ P.y = 0

-- Theorem statement
theorem trajectory_is_line_segment :
  ∀ P : Point2D, satisfiesCondition P → onLineSegment P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l87_8764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_has_two_elements_l87_8754

/-- The function g(x) = (x + 4) / x -/
noncomputable def g (x : ℝ) : ℝ := (x + 4) / x

/-- The sequence of functions g_n -/
noncomputable def g_n : ℕ → (ℝ → ℝ)
  | 0 => g
  | (n + 1) => g ∘ (g_n n)

/-- The set T of all real numbers x such that g_n(x) = x for some positive integer n -/
def T : Set ℝ := {x | ∃ n : ℕ, n > 0 ∧ g_n n x = x}

/-- Theorem: The set T contains exactly 2 elements -/
theorem T_has_two_elements : ∃ (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ T ↔ x ∈ s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_has_two_elements_l87_8754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_negative_slope_line_l87_8723

-- Define the set of angles whose terminal side lies on y = -x
def S : Set ℝ := {β : ℝ | ∃ n : ℤ, β = 135 + n * 180}

-- Define the condition for angles between -360° and 360°
def InRange (β : ℝ) : Prop := -360 < β ∧ β < 360

-- Theorem statement
theorem angle_on_negative_slope_line :
  -- The set S is correctly defined
  S = {β : ℝ | ∃ n : ℤ, β = 135 + n * 180} ∧
  -- The elements of S in the given range are as specified
  {β ∈ S | InRange β} = {-225, -45, 135, 315} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_negative_slope_line_l87_8723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_range_l87_8717

-- Define the equation and its roots
def equation (m n x : ℝ) : Prop := (x^2 - 2*x + m) * (x^2 - 2*x + n) = 0

-- Define the arithmetic progression of roots
def roots_in_ap (m n : ℝ) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ : ℝ), 
    equation m n a₁ ∧ equation m n a₂ ∧ equation m n a₃ ∧ equation m n a₄ ∧
    a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧
    a₁ = 1/4 ∧ 
    (a₂ - a₁ = a₃ - a₂) ∧ (a₃ - a₂ = a₄ - a₃)

-- Define the triangle properties
def triangle_properties (m n a b : ℝ) (A B : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < A ∧ 0 < B ∧
  A = 2 * B ∧
  b = 4 * |m - n| ∧
  A < Real.pi ∧ B < Real.pi/2  -- Ensure acute triangle

theorem side_a_range (m n : ℝ) (a b : ℝ) (A B : ℝ) :
  roots_in_ap m n →
  triangle_properties m n a b A B →
  2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_range_l87_8717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l87_8714

/-- The area of a quadrilateral with vertices at (7,1), (2,6), (4,5), and (11,11) is 25.5 -/
theorem quadrilateral_area : ∃ (area : ℝ), area = 25.5 := by
  let x₁ : ℝ := 7
  let y₁ : ℝ := 1
  let x₂ : ℝ := 2
  let y₂ : ℝ := 6
  let x₃ : ℝ := 4
  let y₃ : ℝ := 5
  let x₄ : ℝ := 11
  let y₄ : ℝ := 11
  let area : ℝ := (1/2) * |x₁*y₂ + x₂*y₃ + x₃*y₄ + x₄*y₁ - (y₁*x₂ + y₂*x₃ + y₃*x₄ + y₄*x₁)|
  exists area
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l87_8714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_increasing_on_interval_l87_8791

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos (x - 2)

-- State the theorem
theorem cos_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_increasing_on_interval_l87_8791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_result_l87_8777

/-- A linear function that satisfies the given conditions -/
noncomputable def f : ℝ → ℝ := 
  sorry

/-- The property that f(f(x)) ≠ x + 1 for all x -/
axiom no_solution : ∀ x : ℝ, f (f x) ≠ x + 1

/-- The linearity of f -/
axiom f_linear : ∃ k b : ℝ, ∀ x : ℝ, f x = k * x + b

/-- The main theorem to prove -/
theorem main_result : f (f (f (f (f 2022)))) - f (f (f 2022)) - f (f 2022) = -2022 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_result_l87_8777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_two_reals_satisfying_inequality_l87_8722

theorem exist_two_reals_satisfying_inequality (S : Finset ℝ) (h : S.card = 7) :
  ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ 0 ≤ (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) < Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_two_reals_satisfying_inequality_l87_8722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l87_8719

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    and eccentricity √5/2, prove that its asymptotes have equation y = ±(1/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c/a = Real.sqrt 5/2) →
  (∃ (f : ℝ → ℝ), ∀ x, f x = (1/2) * x ∧ 
    (∀ ε > 0, ∃ M > 0, ∀ x y, x^2/a^2 - y^2/b^2 = 1 → 
      |x| > M → |y - f x| < ε * |x|)) ∧
  (∃ (g : ℝ → ℝ), ∀ x, g x = -(1/2) * x ∧ 
    (∀ ε > 0, ∃ M > 0, ∀ x y, x^2/a^2 - y^2/b^2 = 1 → 
      |x| > M → |y - g x| < ε * |x|)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l87_8719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l87_8790

theorem triangle_side_calculation (a b : ℝ) (A B : ℝ) :
  b = 5 →
  B = π / 4 →
  Real.tan A = 2 →
  a = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l87_8790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alkane_carbon_percentage_implies_n_l87_8782

/-- Represents the number of carbon atoms in an alkane -/
def n : ℕ := 7

/-- The mass percentage of carbon in the alkane -/
def carbon_percentage : ℝ := 0.84

/-- The atomic mass of carbon -/
def carbon_mass : ℝ := 12

/-- The atomic mass of hydrogen -/
def hydrogen_mass : ℝ := 1

/-- Theorem: If the mass percentage of carbon in an alkane CₙH₂ₙ₊₂ is 84%, then n = 7 -/
theorem alkane_carbon_percentage_implies_n : 
  carbon_percentage = carbon_mass * n / (carbon_mass * n + hydrogen_mass * (2 * n + 2)) → n = 7 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alkane_carbon_percentage_implies_n_l87_8782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_at_one_g_monotone_increasing_range_l87_8772

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log x

-- Define the function g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + Real.log x

-- Statement 1: When a = -2, f(x) has a minimum at x = 1 with f(1) = 1
theorem f_minimum_at_one :
  let a := -2
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f a x ≤ f a y ∧ f a 1 = 1 :=
by sorry

-- Statement 2: The range of a for which g(x) is monotonically increasing on [1, +∞) is [0, +∞)
theorem g_monotone_increasing_range :
  {a : ℝ | ∀ (x y : ℝ), 1 ≤ x ∧ x < y → g a x < g a y} = Set.Ici 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_at_one_g_monotone_increasing_range_l87_8772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_equals_five_l87_8786

def series_sum (n : ℕ) : ℚ :=
  ((-1)^n * n) / (if n % 2 = 0 then 3^(n/2) else 2^((n+1)/2))

theorem fraction_sum_equals_five (a b : ℕ+) (h1 : Nat.Coprime a b) 
  (h2 : (a : ℚ) / b = ∑' n, series_sum n) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_equals_five_l87_8786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l87_8779

/-- The area of an equilateral triangle with side length 1 -/
noncomputable def equilateral_triangle_area : ℝ := Real.sqrt 3 / 4

/-- The ratio of the visual diagram's area to the original triangle's area -/
noncomputable def area_ratio : ℝ := 1 / (2 * Real.sqrt 2)

/-- The theorem stating the area of the original triangle -/
theorem original_triangle_area : 
  equilateral_triangle_area / area_ratio = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l87_8779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_line_l1_minimizes_area_l87_8735

/-- The equation of line l -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (2 + m) * x + (1 - 2 * m) * y + (4 - 3 * m) = 0

/-- The fixed point M -/
def point_M : ℝ × ℝ := (-1, -2)

/-- The equation of line l₁ -/
def line_l1 (x y : ℝ) : Prop :=
  y = 2 * x + 4

/-- The area of the triangle formed by a line through point_M with negative half-axes -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  (1 / 2) * abs ((k - 2) / (-k)) * abs (k - 2)

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_l m (point_M.1) (point_M.2) := by
  sorry

theorem line_l1_minimizes_area :
  ∀ k : ℝ, k < 0 → triangle_area (-2) ≤ triangle_area k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_line_l1_minimizes_area_l87_8735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_flight_time_l87_8773

/-- Calculates the total flight time for a round trip given the outbound and return speeds and the distance for each way. -/
theorem total_flight_time 
  (outbound_speed : ℝ) 
  (return_speed : ℝ) 
  (distance : ℝ) 
  (h1 : outbound_speed = 300) 
  (h2 : return_speed = 500) 
  (h3 : distance = 1500) : 
  distance / outbound_speed + distance / return_speed = 8 := by
  -- Replace all hypothesis with their actual values
  rw [h1, h2, h3]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check total_flight_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_flight_time_l87_8773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sum_l87_8727

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def b_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ := 2^(a n - 2)

noncomputable def sum_geometric (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem arithmetic_and_geometric_sum 
  (a : ℕ → ℝ) 
  (h_arith : ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d)
  (h_a₁ : a 1 = 3)
  (h_a₅ : a 5 = 7) :
  (∀ n, a n = n + 2) ∧ 
  (∀ n, sum_geometric 2 2 n = 2^(n+1) - 2) := by
  sorry

#check arithmetic_and_geometric_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sum_l87_8727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l87_8729

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a - y^2 = 1

-- Define the asymptote line
def asymptote (x y : ℝ) : Prop :=
  x + 2*y = 0

-- Define the eccentricity
noncomputable def eccentricity (e : ℝ) : Prop :=
  e = Real.sqrt 5 / 2

-- Theorem statement
theorem hyperbola_eccentricity (a : ℝ) :
  a > 0 →
  (∃ x y, hyperbola a x y ∧ asymptote x y) →
  ∃ e, eccentricity e :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l87_8729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dots_on_left_faces_l87_8792

/-- Represents the number of dots on a cube face -/
inductive Dots
  | one
  | two
  | three

/-- Represents a cube with dots on its faces -/
structure Cube where
  front : Dots
  back : Dots
  left : Dots
  right : Dots
  top : Dots
  bottom : Dots

/-- Predicate to check if two cubes are touching -/
def CubesAreTouching (cubes : Vector Cube 7) (i j : Fin 7) (f₁ f₂ : Cube → Dots) : Prop := sorry

/-- Represents the configuration of 7 cubes in a "П" shape -/
structure Configuration where
  cubes : Vector Cube 7
  /-- Ensure that any two touching faces have the same number of dots -/
  touching_faces_same : ∀ (i j : Fin 7) (f₁ f₂ : Cube → Dots), 
    CubesAreTouching cubes i j f₁ f₂ → f₁ (cubes.get i) = f₂ (cubes.get j)

/-- The main theorem statement -/
theorem dots_on_left_faces (config : Configuration) :
  ∃ (c₁ c₂ c₃ : Cube),
    c₁ ∈ config.cubes.toList ∧
    c₂ ∈ config.cubes.toList ∧
    c₃ ∈ config.cubes.toList ∧
    c₁.left = Dots.two ∧
    c₂.left = Dots.two ∧
    c₃.left = Dots.three := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dots_on_left_faces_l87_8792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_single_fractions_l87_8793

/-- A structure representing a mathematical expression -/
structure Expression where
  value : ℚ → ℚ → ℚ → ℚ → ℚ

/-- A predicate to determine if an expression is a single fraction -/
def is_single_fraction (e : Expression) : Bool :=
  match e with
  | ⟨f⟩ => 
    match f 0 0 0 0 with
    | 0 => true  -- Consider 0 as a single fraction
    | _ => true  -- For simplicity, we'll consider all expressions as single fractions
                 -- In a real implementation, we'd need a more sophisticated check

/-- The list of expressions from the problem -/
def expressions : List Expression :=
  [ ⟨λ m n _ _ => (m - n) / 2⟩,
    ⟨λ _ _ _ y => y / 3⟩,  -- Using 3 instead of π for simplicity
    ⟨λ _ _ x _ => (2 * x) / (x + 2)⟩,
    ⟨λ _ _ x y => x / 7 + y / 8⟩,
    ⟨λ _ _ _ y => 2 / y⟩ ]

/-- The theorem statement -/
theorem count_single_fractions :
  (expressions.filter is_single_fraction).length = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_single_fractions_l87_8793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l87_8776

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49
def C₂ (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 100

-- Define the tangent line segment
def is_tangent (P Q : ℝ × ℝ) : Prop :=
  C₁ P.1 P.2 ∧ C₂ Q.1 Q.2 ∧
  ∀ R : ℝ × ℝ, C₁ R.1 R.2 → C₂ R.1 R.2 → 
    (P.1 - R.1)^2 + (P.2 - R.2)^2 ≤ (Q.1 - R.1)^2 + (Q.2 - R.2)^2

-- State the theorem
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent P Q ∧
    (∀ P' Q' : ℝ × ℝ, is_tangent P' Q' →
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ 
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) ∧
    abs (Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) - 47.60) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l87_8776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_square_meter_is_three_l87_8771

/-- A rectangular lawn with two intersecting roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  road_width : ℝ
  total_cost : ℝ

/-- Calculate the cost per square meter for traveling the roads -/
noncomputable def cost_per_square_meter (lawn : LawnWithRoads) : ℝ :=
  lawn.total_cost / ((lawn.length * lawn.road_width) + (lawn.width * lawn.road_width) - (lawn.road_width * lawn.road_width))

/-- Theorem stating that the cost per square meter is 3 for the given lawn -/
theorem cost_per_square_meter_is_three :
  let lawn : LawnWithRoads := {
    length := 110,
    width := 60,
    road_width := 10,
    total_cost := 4800
  }
  cost_per_square_meter lawn = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_square_meter_is_three_l87_8771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l87_8753

noncomputable section

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the distance sum property -/
def distance_sum_property (a : ℝ) : Prop :=
  ∀ (x y : ℝ), ellipse x y a (Real.sqrt (a^2 - 1)) → 
    Real.sqrt ((x + 1)^2 + y^2) + Real.sqrt ((x - 1)^2 + y^2) = 4

/-- Definition of eccentricity -/
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

/-- Definition of the line l -/
def line_l (x y K : ℝ) : Prop :=
  y = K * (x - 4) ∧ K > 0

/-- Definition of the perpendicular bisector l' -/
def line_l' (x y x0 y0 K : ℝ) : Prop :=
  y - y0 = -(1/K) * (x - x0)

/-- Main theorem -/
theorem ellipse_properties (a b : ℝ) (h1 : distance_sum_property a) 
    (h2 : eccentricity a b = 1/2) :
  (∀ (x y : ℝ), ellipse x y a b ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ (K m : ℝ), (∃ (x1 y1 x2 y2 x0 y0 : ℝ),
    ellipse x1 y1 a b ∧ 
    ellipse x2 y2 a b ∧
    line_l x1 y1 K ∧
    line_l x2 y2 K ∧
    x0 = (x1 + x2) / 2 ∧
    y0 = (y1 + y2) / 2 ∧
    line_l' 0 m x0 y0 K) →
    -1/2 < m ∧ m < 1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l87_8753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_factors_720_l87_8749

def sum_even_factors (n : ℕ) : ℕ := 
  (Finset.filter (λ x ↦ Even x ∧ x ∣ n) (Finset.range (n + 1))).sum id

theorem sum_even_factors_720 : sum_even_factors 720 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_factors_720_l87_8749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_decimal_rearrangement_rational_l87_8709

/-- Represents an infinite sequence of digits -/
def InfiniteDecimal := ℕ → Fin 10

/-- Represents a permutation of natural numbers -/
def Permutation := ℕ → ℕ

/-- Check if a number is rational -/
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

/-- Converts a rearranged infinite decimal to a real number -/
noncomputable def real_of_rearranged_decimal (d : InfiniteDecimal) (p : Permutation) : ℝ :=
  sorry -- Implementation details omitted for brevity

/-- The theorem statement -/
theorem infinite_decimal_rearrangement_rational
  (d : InfiniteDecimal) :
  ∃ (p : Permutation), IsRational (real_of_rearranged_decimal d p) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_decimal_rearrangement_rational_l87_8709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_two_thirds_l87_8703

/-- An isosceles triangle ABC with midpoint D on side AC and BD = 1 -/
structure IsoscelesTriangle where
  -- AB = AC = b
  b : ℝ
  -- D is the midpoint of AC
  is_midpoint : b > 0
  -- BD = 1
  bd_length : b > 0

/-- The area of the isosceles triangle -/
noncomputable def triangle_area (t : IsoscelesTriangle) : ℝ :=
  (t.b^2 / 2) * Real.sqrt (1 - ((5 * t.b^2 - 4) / (4 * t.b^2))^2)

/-- The maximum area of the isosceles triangle is 2/3 -/
theorem max_area_is_two_thirds :
  ∃ (t : IsoscelesTriangle), ∀ (u : IsoscelesTriangle), triangle_area t ≥ triangle_area u ∧ triangle_area t = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_two_thirds_l87_8703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l87_8756

/-- An ellipse with equation 2x^2 + 3y^2 = 6 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | 2 * p.1^2 + 3 * p.2^2 = 6}

/-- An isosceles triangle inscribed in the ellipse -/
structure InscribedTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A_in_ellipse : A ∈ Ellipse
  B_in_ellipse : B ∈ Ellipse
  C_in_ellipse : C ∈ Ellipse
  isosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2
  altitude_on_x_axis : C.2 = 0

/-- The area of the triangle -/
noncomputable def triangleArea (t : InscribedTriangle) : ℝ :=
  abs ((t.A.1 - t.C.1) * (t.B.2 - t.C.2) - (t.B.1 - t.C.1) * (t.A.2 - t.C.2)) / 2

theorem inscribed_triangle_area :
  ∀ t : InscribedTriangle,
    t.A = (1, Real.sqrt 2) →
    triangleArea t = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l87_8756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l87_8737

noncomputable section

-- Define the pyramid
def pyramid_base_side : ℝ := 8
def pyramid_height : ℝ := 10

-- Define the total surface area function
noncomputable def total_surface_area (base_side height : ℝ) : ℝ :=
  let base_area := base_side ^ 2
  let slant_height := Real.sqrt (height ^ 2 + (base_side / 2) ^ 2)
  let lateral_area := 4 * (1 / 2 * base_side * slant_height)
  base_area + lateral_area

-- Theorem statement
theorem pyramid_surface_area :
  total_surface_area pyramid_base_side pyramid_height = 64 + 32 * Real.sqrt 29 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l87_8737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_property_l87_8734

def vector_reflection (v w : ℝ × ℝ) (x : ℝ × ℝ) : ℝ × ℝ := sorry

theorem reflection_property :
  let r := vector_reflection (2, -3) (6, 1)
  r (1, 4) = (-1, -4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_property_l87_8734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_positive_sequence_a_integer_sequence_a_recurrence_l87_8706

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (sequence_a (n + 1))^2 + (-1)^n / sequence_a n

theorem sequence_a_positive (n : ℕ) : 
  0 < sequence_a n := by
  sorry

theorem sequence_a_integer (n : ℕ) : 
  ∃ k : ℤ, sequence_a n = k := by
  sorry

theorem sequence_a_recurrence (n : ℕ) : 
  sequence_a (n + 2) = sequence_a (n + 1) + sequence_a n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_positive_sequence_a_integer_sequence_a_recurrence_l87_8706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l87_8712

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_property (a₁ d : ℝ) (h : d ≠ 0) :
  arithmetic_sequence a₁ d 6 = 3 * arithmetic_sequence a₁ d 4 ∧
  arithmetic_sum a₁ d 9 = 18 * arithmetic_sequence a₁ d 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l87_8712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l87_8747

def a : ℤ := sorry

axiom a_is_even_multiple_of_1171 : ∃ k : ℤ, a = 2 * k * 1171

theorem gcd_of_polynomials : 
  Nat.gcd (Int.natAbs (3 * a^2 + 34 * a + 72)) (Int.natAbs (a + 15)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l87_8747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silverware_probability_l87_8718

/-- The probability of selecting exactly 1 fork, 2 spoons, and 1 knife
    when drawing 4 pieces of silverware from a drawer containing
    8 forks, 5 spoons, and 10 knives -/
theorem silverware_probability : ℚ := by
  -- Define the number of each type of silverware
  let forks : ℕ := 8
  let spoons : ℕ := 5
  let knives : ℕ := 10

  -- Define the total number of pieces and the number to be drawn
  let total_pieces : ℕ := forks + spoons + knives
  let draw_count : ℕ := 4

  -- Define the number of each type to be drawn
  let drawn_forks : ℕ := 1
  let drawn_spoons : ℕ := 2
  let drawn_knives : ℕ := 1

  -- Calculate the probability
  have h : (Nat.choose forks drawn_forks * Nat.choose spoons drawn_spoons * Nat.choose knives drawn_knives : ℚ) / (Nat.choose total_pieces draw_count : ℚ) = 800 / 8855 := by sorry

  exact 800 / 8855


end NUMINAMATH_CALUDE_ERRORFEEDBACK_silverware_probability_l87_8718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_is_random_variable_l87_8702

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]
variable {n : ℕ}

/-- A random variable is a measurable function from a probability space to the real numbers. -/
def RandomVariable (ξ : Ω → ℝ) : Prop := Measurable ξ

/-- A Borel function is a measurable function between two measurable spaces. -/
def BorelFunction (φ : (Fin n → ℝ) → ℝ) : Prop := Measurable φ

theorem composition_is_random_variable
  (ξ : Fin n → Ω → ℝ)
  (φ : (Fin n → ℝ) → ℝ)
  (h_ξ : ∀ i, RandomVariable (ξ i))
  (h_φ : BorelFunction φ) :
  RandomVariable (fun ω => φ (fun i => ξ i ω)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_is_random_variable_l87_8702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_number_fifth_row_l87_8726

/-- Represents an inverted triangular array of numbers -/
def InvertedTriangularArray (n : ℕ) : ℕ → ℕ → ℕ := sorry

/-- The first row of the array contains the arithmetic sequence 1, 3, 5, ..., 2n-1 -/
axiom first_row_def (n k : ℕ) (h : k ≤ n) : InvertedTriangularArray n 1 k = 2 * k - 1

/-- Each number in subsequent rows is the sum of the two numbers above it -/
axiom subsequent_rows_def (n i j : ℕ) (hi : i > 1) (hj : j < n - i + 2) :
  InvertedTriangularArray n i j = InvertedTriangularArray n (i-1) j + InvertedTriangularArray n (i-1) (j+1)

/-- The theorem to be proved -/
theorem seventh_number_fifth_row (n : ℕ) (hn : n ≥ 5) :
  InvertedTriangularArray n 5 7 = 272 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_number_fifth_row_l87_8726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l87_8701

noncomputable def z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I

theorem z_in_first_quadrant : 
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l87_8701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_bought_l87_8780

def total_money : ℚ := 40
def sandwich_cost : ℚ := 5
def soft_drink_cost : ℚ := 3/2

def max_sandwiches : ℕ := Int.toNat ((total_money / sandwich_cost).floor)

def remaining_money : ℚ := total_money - (↑max_sandwiches * sandwich_cost)

def max_soft_drinks : ℕ := Int.toNat ((remaining_money / soft_drink_cost).floor)

theorem max_items_bought :
  max_sandwiches + max_soft_drinks = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_bought_l87_8780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_specific_hyperbola_l87_8763

open Real

/-- The distance between the foci of a hyperbola -/
noncomputable def foci_distance (a b c d e f : ℝ) : ℝ :=
  let center_x := -b / (2 * a)
  let center_y := -d / (2 * c)
  let normalized_a := sqrt ((f - e + (b^2 / (4 * a)) + (d^2 / (4 * c))) / a)
  let normalized_b := sqrt ((f - e + (b^2 / (4 * a)) + (d^2 / (4 * c))) / (-c))
  2 * sqrt (normalized_a^2 + normalized_b^2)

/-- The distance between the foci of the hyperbola 9x^2 - 18x - 16y^2 + 32y = -72 -/
theorem foci_distance_specific_hyperbola :
  foci_distance 9 (-18) (-16) 32 0 (-72) = sqrt 1711 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_specific_hyperbola_l87_8763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_4x_mod_9_l87_8728

theorem remainder_4x_mod_9 (x : ℕ) (h : x % 9 = 5) : (4 * x) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_4x_mod_9_l87_8728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_20_degrees_l87_8767

/-- Represents the volume of gas at a given temperature -/
structure GasVolume where
  temp : ℚ
  volume : ℚ

/-- Represents the relationship between temperature and volume change -/
def volumeChangeRate : ℚ := 4 / 3

/-- The reference point for temperature and volume -/
def referencePoint : GasVolume := { temp := 32, volume := 24 }

/-- Calculates the volume change based on temperature difference -/
def volumeChange (tempDiff : ℚ) : ℚ :=
  tempDiff * volumeChangeRate

/-- Theorem stating that the volume of gas at 20° is 8 cubic centimeters -/
theorem volume_at_20_degrees :
  let finalTemp : ℚ := 20
  let tempDiff : ℚ := finalTemp - referencePoint.temp
  let volumeDiff : ℚ := volumeChange tempDiff
  referencePoint.volume + volumeDiff = 8 := by
  -- Unfold definitions
  unfold referencePoint volumeChange volumeChangeRate
  -- Simplify arithmetic
  simp [add_comm, mul_comm, mul_assoc]
  -- Prove equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_20_degrees_l87_8767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_investment_calculation_l87_8713

/-- A's initial investment -/
noncomputable def A_investment : ℝ := 3500

/-- Number of months A invested -/
noncomputable def A_months : ℝ := 12

/-- Number of months B invested -/
noncomputable def B_months : ℝ := 7

/-- Ratio of A's share to B's share in profits -/
noncomputable def profit_ratio : ℝ := 2 / 3

/-- B's investment -/
noncomputable def B_investment : ℝ := 4500

theorem B_investment_calculation :
  A_investment * A_months / (B_investment * B_months) = profit_ratio := by
  sorry

#check B_investment_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_investment_calculation_l87_8713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clear_time_l87_8766

-- Define the train properties
noncomputable def train1_length : ℝ := 220
noncomputable def train2_length : ℝ := 275
noncomputable def train1_speed : ℝ := 120
noncomputable def train2_speed : ℝ := 90

-- Define the function to calculate the time
noncomputable def clear_time (l1 l2 s1 s2 : ℝ) : ℝ :=
  (l1 + l2) / ((s1 + s2) * 1000 / 3600)

-- Theorem statement
theorem trains_clear_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |clear_time train1_length train2_length train1_speed train2_speed - 8.48| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clear_time_l87_8766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_of_circle_and_tangent_line_l87_8721

-- Define the circle type
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the line type
structure Line where
  point : EuclideanSpace ℝ (Fin 2)
  direction : EuclideanSpace ℝ (Fin 2)

-- Define the inversion transformation
noncomputable def inversion (inversionCircle : Circle) (p : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry

-- Define tangency between a circle and a line
def isTangent (c : Circle) (l : Line) : Prop :=
  sorry

-- Define tangency between two circles
def areTangent (c1 c2 : Circle) : Prop :=
  sorry

-- Define parallel lines
def areParallel (l1 l2 : Line) : Prop :=
  sorry

-- Helper functions for type conversion
def toCircle (x : Circle ⊕ Line) : Circle :=
  match x with
  | Sum.inl c => c
  | Sum.inr _ => sorry -- Default circle if input is a line

def toLine (x : Circle ⊕ Line) : Line :=
  match x with
  | Sum.inl _ => sorry -- Default line if input is a circle
  | Sum.inr l => l

def isCircle (x : Circle ⊕ Line) : Prop :=
  match x with
  | Sum.inl _ => True
  | Sum.inr _ => False

def isLine (x : Circle ⊕ Line) : Prop :=
  match x with
  | Sum.inl _ => False
  | Sum.inr _ => True

-- The main theorem
theorem inversion_of_circle_and_tangent_line 
  (c : Circle) (l : Line) (inversionCircle : Circle) :
  isTangent c l →
  ∃ (result1 result2 : Circle ⊕ Line),
    (areTangent (toCircle result1) (toCircle result2) ∨
     (isCircle result1 ∧ isLine result2 ∧ isTangent (toCircle result1) (toLine result2)) ∨
     (isLine result1 ∧ isLine result2 ∧ areParallel (toLine result1) (toLine result2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_of_circle_and_tangent_line_l87_8721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l87_8789

open Set
open Real

noncomputable def f (x : ℝ) := (3 * cos x + 1) / (2 - cos x)

theorem range_of_f :
  let S := {y | ∃ x, -π/3 < x ∧ x < π/3 ∧ f x = y}
  S = Ioo (5/3) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l87_8789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocal_l87_8738

theorem tan_sum_reciprocal (u v : ℝ) 
  (h1 : Real.sin u / Real.cos v + Real.sin v / Real.cos u = 2)
  (h2 : Real.cos u / Real.sin v + Real.cos v / Real.sin u = 3) :
  Real.tan u / Real.tan v + Real.tan v / Real.tan u = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocal_l87_8738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_y_value_l87_8761

/-- Two vectors in ℝ³ are parallel if one is a scalar multiple of the other -/
def parallel (a b : Fin 3 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 3, k * a i = b i

/-- Given vectors a and b, if they are parallel, then y = 7.5 -/
theorem parallel_vectors_y_value (a b : Fin 3 → ℝ) 
  (ha : a = ![2, 4, 5]) 
  (hb : b = ![3, 6, b 2]) 
  (h_parallel : parallel a b) : 
  b 2 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_y_value_l87_8761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l87_8775

noncomputable def vector1 : ℝ × ℝ := (-3, 4)
noncomputable def vector2 : ℝ × ℝ := (1, 6)

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  ((dot_product / magnitude_squared) * w.1, (dot_product / magnitude_squared) * w.2)

theorem projection_equality :
  ∃ (v : ℝ × ℝ), projection vector1 v = projection vector2 v ∧
                 projection vector1 v = (-2.2, 4.4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l87_8775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_tree_distance_l87_8757

/-- The distance between consecutive trees in a garden -/
noncomputable def distance_between_trees (total_length : ℝ) (num_trees : ℕ) : ℝ :=
  total_length / (num_trees - 1)

/-- Theorem: The distance between consecutive trees is 32 meters -/
theorem garden_tree_distance :
  let total_length : ℝ := 800
  let num_trees : ℕ := 26
  distance_between_trees total_length num_trees = 32 := by
  -- Unfold the definition and simplify
  unfold distance_between_trees
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_tree_distance_l87_8757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fourth_twelfth_terms_l87_8765

/-- An arithmetic progression with a first term and common difference. -/
structure ArithmeticProgression where
  first_term : ℝ
  common_diff : ℝ

/-- The nth term of an arithmetic progression. -/
noncomputable def nth_term (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.first_term + (n - 1 : ℝ) * ap.common_diff

/-- The sum of the first n terms of an arithmetic progression. -/
noncomputable def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  n / 2 * (2 * ap.first_term + (n - 1 : ℝ) * ap.common_diff)

/-- Theorem: In an arithmetic progression where the sum of the first 15 terms is 60,
    the sum of the 4th and 12th terms is 8. -/
theorem sum_fourth_twelfth_terms (ap : ArithmeticProgression) 
  (h : sum_n_terms ap 15 = 60) : 
  nth_term ap 4 + nth_term ap 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fourth_twelfth_terms_l87_8765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l87_8736

-- Define the triangles and their side lengths
structure Triangle where
  A : Real
  B : Real
  C : Real

def BC (t : Triangle) : Real := abs (t.B - t.C)
def AC (t : Triangle) : Real := abs (t.A - t.C)
def DF (t : Triangle) : Real := abs (t.A - t.B)  -- Using A and B as proxies for D and F
def EF (t : Triangle) : Real := abs (t.B - t.C)  -- Reusing BC definition for EF

-- Define similarity relation
def Similar (t1 t2 : Triangle) : Prop := 
  ∃ k : Real, k > 0 ∧ BC t1 = k * BC t2 ∧ AC t1 = k * AC t2 ∧ DF t1 = k * DF t2

-- State the theorem
theorem similar_triangles_side_length 
  (ABC DEF : Triangle) 
  (h_similar : Similar ABC DEF) 
  (h_BC : BC ABC = 6) 
  (h_EF : EF DEF = 4) 
  (h_AC : AC ABC = 9) : 
  DF DEF = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l87_8736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l87_8716

noncomputable def sequence_a (n : ℕ) : ℝ := 3 * n + 1

noncomputable def sequence_b (n : ℕ) : ℝ := 3 / (sequence_a n * sequence_a (n + 1))

noncomputable def sum_s (n : ℕ) : ℝ := (n * (sequence_a 1 + sequence_a n)) / 2

noncomputable def sum_t (n : ℕ) : ℝ := 1 / 4 - 1 / (3 * n + 4)

theorem sequence_properties :
  (∀ n : ℕ, sequence_a n > 0) ∧
  (∀ n : ℕ, (sequence_a n)^2 + 3 * sequence_a n = 6 * sum_s n + 4) ∧
  (∀ n : ℕ, sequence_a n = 3 * n + 1) ∧
  (∀ n : ℕ, sum_t n = (1 : ℝ) / 4 - 1 / (3 * n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l87_8716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l87_8739

/-- Represents the total journey distance in kilometers -/
def total_distance : ℝ := 253

/-- Represents the total journey time in hours -/
def total_time : ℝ := 12

/-- Represents the speed for the first third of the journey in km/hr -/
def speed1 : ℝ := 18

/-- Represents the speed for the second third of the journey in km/hr -/
def speed2 : ℝ := 24

/-- Represents the speed for the final third of the journey in km/hr -/
def speed3 : ℝ := 30

/-- Represents the total break time in hours -/
def break_time : ℝ := 1

/-- Theorem stating that the total distance is approximately 253 km given the conditions -/
theorem journey_distance : 
  ∃ (d : ℝ), (d ≥ total_distance - 1 ∧ d ≤ total_distance + 1) ∧
  (d / 3) / speed1 + (d / 3) / speed2 + (d / 3) / speed3 + break_time = total_time :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l87_8739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_vector_l87_8759

/-- Given two perpendicular vectors OA and OB with lengths 3 and 4 respectively,
    prove that the vector OC that minimizes its length is a specific linear combination of OA and OB. -/
theorem min_length_vector (O A B : ℝ × ℝ × ℝ) (h1 : ‖A - O‖ = 3) (h2 : ‖B - O‖ = 4) 
    (h3 : (A - O) • (B - O) = 0) : 
    ∃ (C : ℝ × ℝ × ℝ), C = O + (16/25 : ℝ) • (A - O) + (9/25 : ℝ) • (B - O) ∧ 
    ∀ (D : ℝ × ℝ × ℝ) (θ : ℝ), D = O + Real.sin θ ^ 2 • (A - O) + Real.cos θ ^ 2 • (B - O) → 
    ‖C - O‖ ≤ ‖D - O‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_vector_l87_8759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_product_l87_8705

theorem cosine_difference_product (a b : ℝ) : 
  Real.cos (a + b) - Real.cos (a - b) = -2 * Real.sin a * Real.sin b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_product_l87_8705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l87_8774

noncomputable def f (x : ℝ) := Real.sqrt (Real.sin x - 1/2)

theorem domain_of_f :
  {x : ℝ | ∃ k : ℤ, π/6 + 2*k*π ≤ x ∧ x ≤ 5*π/6 + 2*k*π} =
  {x : ℝ | ∃ y : ℝ, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l87_8774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_tenth_l87_8795

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The sum of 81.76 and 34.587 rounded to the nearest tenth equals 116.3 -/
theorem sum_and_round_to_tenth : roundToNearestTenth (81.76 + 34.587) = 116.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_tenth_l87_8795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l87_8748

-- Define the vectors
def a : ℝ × ℝ × ℝ := (4, 3, 1)
def b : ℝ × ℝ × ℝ := (1, -2, 1)
def c : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define coplanarity
def coplanar (v₁ v₂ v₃ : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y z : ℝ), x • v₁ + y • v₂ + z • v₃ = (0, 0, 0) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)

-- Theorem stating that the vectors are not coplanar
theorem vectors_not_coplanar : ¬(coplanar a b c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l87_8748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l87_8732

theorem trigonometric_identities (α : ℝ) 
  (h1 : Real.cos α = -Real.sqrt 5 / 5)
  (h2 : α ∈ Set.Ioo π (3 * π / 2)) : 
  Real.sin α = -(2 * Real.sqrt 5 / 5) ∧ 
  (Real.sin (π + α) + 2 * Real.sin (3 * π / 2 + α)) / (Real.cos (3 * π - α) + 1) = Real.sqrt 5 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l87_8732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_min_f_max_l87_8708

/-- A quadratic function f(x) = x^2 + ax + b satisfying f(0) = 6 and f(1) = 5 -/
def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

/-- The conditions f(0) = 6 and f(1) = 5 -/
axiom f_0 (a b : ℝ) : f 0 a b = 6
axiom f_1 (a b : ℝ) : f 1 a b = 5

/-- The domain of x is [-2, 2] -/
def domain : Set ℝ := Set.Icc (-2) 2

/-- Theorem: The quadratic function f(x) = x^2 - 2x + 6 -/
theorem f_expression : ∃ a b : ℝ, ∀ x : ℝ, f x a b = x^2 - 2*x + 6 := by sorry

/-- Theorem: The minimum value of f(x) on [-2, 2] is 5 -/
theorem f_min : ∃ a b : ℝ, ∃ x ∈ domain, ∀ y ∈ domain, f x a b ≤ f y a b ∧ f x a b = 5 := by sorry

/-- Theorem: The maximum value of f(x) on [-2, 2] is 14 -/
theorem f_max : ∃ a b : ℝ, ∃ x ∈ domain, ∀ y ∈ domain, f y a b ≤ f x a b ∧ f x a b = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_min_f_max_l87_8708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_region_in_half_l87_8742

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a rectangle given its width and height -/
noncomputable def rectangleArea (width height : ℝ) : ℝ := width * height

/-- Calculates the area of a right triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (base * height) / 2

/-- The L-shaped region in the problem -/
structure LShapedRegion where
  rectangleVertices : Fin 4 → Point
  triangleVertices : Fin 3 → Point

/-- Defines the specific L-shaped region from the problem -/
def problemRegion : LShapedRegion := {
  rectangleVertices := λ i => match i with
    | 0 => ⟨0, 0⟩
    | 1 => ⟨0, 4⟩
    | 2 => ⟨5, 4⟩
    | 3 => ⟨5, 2⟩
  triangleVertices := λ i => match i with
    | 0 => ⟨5, 2⟩
    | 1 => ⟨7, 2⟩
    | 2 => ⟨5, 0⟩
}

/-- Calculates the total area of the L-shaped region -/
noncomputable def totalArea (region : LShapedRegion) : ℝ :=
  rectangleArea 5 4 + triangleArea 2 2

/-- Theorem: The line through (0,0) and (11/4, 4) divides the L-shaped region in half -/
theorem line_divides_region_in_half (region : LShapedRegion) :
  let slope := 16 / 11
  let totalArea := totalArea region
  let halfArea := totalArea / 2
  let intersectionPoint : Point := ⟨11/4, 4⟩
  slope * intersectionPoint.x = intersectionPoint.y ∧
  halfArea = rectangleArea intersectionPoint.x 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_region_in_half_l87_8742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sachin_age_l87_8751

/-- Given that Rahul is 18 years older than Sachin and their ages are in the ratio 7:9, 
    prove that Sachin's age is 63 years. -/
theorem sachin_age (sachin rahul : ℝ) 
  (age_diff : rahul = sachin + 18)
  (age_ratio : sachin / rahul = 7 / 9) :
  sachin = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sachin_age_l87_8751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_condition_l87_8752

/-- The sequence a_n defined as n^2 - 2λn for n ∈ ℕ* -/
def a (lambda : ℝ) (n : ℕ+) : ℝ := n.val^2 - 2*lambda*n.val

/-- The property that the sequence a_n is increasing -/
def is_increasing (lambda : ℝ) : Prop :=
  ∀ n : ℕ+, a lambda n < a lambda (n + 1)

/-- The statement that λ < 1 is sufficient but not necessary for a_n to be increasing -/
theorem lambda_condition (lambda : ℝ) :
  (lambda < 1 → is_increasing lambda) ∧ ¬(is_increasing lambda → lambda < 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_condition_l87_8752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_exchange_impossibility_l87_8783

theorem coupon_exchange_impossibility : ¬ ∃ (x y : ℤ), 
  x + y = 1991 ∧ 
  x - y = 0 ∧ 
  x ≥ 0 ∧ 
  y ≥ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_exchange_impossibility_l87_8783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l87_8799

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 12) + φ)

theorem g_properties (φ : ℝ) (h : |φ| < Real.pi / 2) :
  (∀ x, g (-x) φ = g x φ) →
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → g y φ < g x φ) ∧
  (∀ x, g (Real.pi / 2 + x) φ = g (Real.pi / 2 - x) φ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l87_8799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l87_8731

theorem cube_root_equation_solution :
  ∃ x : ℝ, (5 - 2 / x)^(1/3 : ℝ) = -3 ↔ x = 1/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l87_8731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problems_l87_8720

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (-1, k)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define perpendicular vectors
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ (c : ℝ), v.1 * w.2 = c * v.2 * w.1

-- Define the angle between vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ := Real.arccos (dot_product v w / (magnitude v * magnitude w))

theorem vector_problems :
  -- Part 1: If a ⊥ b, then k = 2
  (perpendicular a (b 2)) ∧
  -- Part 2: If a ∥ b, then a · b = -5/2
  (parallel a (b (-1/2)) → dot_product a (b (-1/2)) = -5/2) ∧
  -- Part 3: If angle between a and b is 135°, then k = -3 or k = 1/3
  (angle a (b (-3)) = 3*π/4 ∨ angle a (b (1/3)) = 3*π/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problems_l87_8720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_distance_l87_8770

/-- The length of the line segment between the x-axis projection of A(3,5,-7) 
    and the z-axis projection of B(-2,4,3) is 3. -/
theorem projection_distance (A B : ℝ × ℝ × ℝ) : 
  A = (3, 5, -7) → 
  B = (-2, 4, 3) → 
  let A' : ℝ × ℝ × ℝ := (A.fst, 0, 0)
  let B' : ℝ × ℝ × ℝ := (0, 0, B.snd.snd)
  Real.sqrt ((A'.fst - B'.fst)^2 + (A'.snd.fst - B'.snd.fst)^2 + (A'.snd.snd - B'.snd.snd)^2) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_distance_l87_8770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_smallest_n_l87_8781

def S_n (n : ℕ) : ℚ :=
  ((n - 1) * n * (n + 1) * (3 * n + 2)) / 24

def is_divisible_by_five (q : ℚ) : Prop :=
  ∃ k : ℤ, q = k / 5

def smallest_valid_n : Set ℕ :=
  {n : ℕ | n ≥ 2 ∧ is_divisible_by_five (S_n n)}

theorem sum_of_eight_smallest_n : 
  ∃ (ns : Finset ℕ), ns.card = 8 ∧ 
    (∀ n ∈ ns, n ∈ smallest_valid_n) ∧
    (∀ m, m ∈ smallest_valid_n → m ∉ ns → ∃ n ∈ ns, n < m) ∧
    ns.sum id = 148 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_smallest_n_l87_8781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leah_wins_prob_l87_8758

/-- The probability of Leah's coin landing heads -/
noncomputable def leah_prob : ℝ := 1/4

/-- The probability of Ben's coin landing heads -/
noncomputable def ben_prob : ℝ := 1/3

/-- The probability that Leah's result is different first -/
noncomputable def leah_different_first : ℝ := 3/5

theorem leah_wins_prob : 
  leah_different_first = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leah_wins_prob_l87_8758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l87_8785

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b * Real.cos t.C + t.b * Real.sin t.C = t.a ∧
  t.a / 4 = t.b * Real.sin t.B

-- Theorem statement
theorem triangle_proof (t : Triangle) (h : triangle_conditions t) :
  t.B = π / 4 ∧ Real.cos t.A = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l87_8785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_265_l87_8784

/-- The volume of a parallelepiped formed by three vectors is equal to the absolute value of the scalar triple product of these vectors. -/
def parallelepiped_volume (v w u : ℝ × ℝ × ℝ) : ℝ :=
  let (v₁, v₂, v₃) := v
  let (w₁, w₂, w₃) := w
  let (u₁, u₂, u₃) := u
  |v₁ * (w₂ * u₃ - w₃ * u₂) - v₂ * (w₁ * u₃ - w₃ * u₁) + v₃ * (w₁ * u₂ - w₂ * u₁)|

/-- The volume of the parallelepiped formed by the given vectors is 265. -/
theorem volume_is_265 : parallelepiped_volume (7, -4, 3) (13, -1, 2) (1, 0, 6) = 265 := by
  -- Unfold the definition of parallelepiped_volume
  unfold parallelepiped_volume
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_265_l87_8784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_sequence_l87_8788

def sequence_sum (start : ℕ) (stop : ℕ) (step : ℕ) : ℕ := 
  sorry

theorem sum_of_specific_sequence :
  sequence_sum 199 9901 99 = 499950 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_sequence_l87_8788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andreas_rhinestones_l87_8768

theorem andreas_rhinestones (n : ℕ) : 
  (0.35 * (n : ℝ) + 0.20 * (n : ℝ) + 51 = (n : ℝ)) → n = 114 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andreas_rhinestones_l87_8768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_equals_naturals_l87_8730

theorem subset_equals_naturals (X : Set ℕ) 
  (h_nonempty : X.Nonempty)
  (h_mult_four : ∀ x ∈ X, (4 * x) ∈ X)
  (h_floor_sqrt : ∀ x ∈ X, ⌊Real.sqrt (x : ℝ)⌋.toNat ∈ X) :
  X = Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_equals_naturals_l87_8730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homologous_functions_l87_8760

-- Define the type for our functions
def RealFunction := ℝ → ℝ

-- Define the property of having the same range for different domains
def HasSameRangeForDifferentDomains (f : RealFunction) : Prop :=
  ∃ (D1 D2 : Set ℝ), D1 ≠ D2 ∧ f '' D1 = f '' D2

-- Define our six functions
noncomputable def tan : RealFunction := Real.tan
noncomputable def cos : RealFunction := Real.cos
noncomputable def cube : RealFunction := λ x => x^3
noncomputable def exp2 : RealFunction := λ x => 2^x
noncomputable def log : RealFunction := Real.log
noncomputable def fourth : RealFunction := λ x => x^4

-- State the theorem
theorem homologous_functions :
  HasSameRangeForDifferentDomains tan ∧
  HasSameRangeForDifferentDomains cos ∧
  HasSameRangeForDifferentDomains fourth ∧
  ¬HasSameRangeForDifferentDomains cube ∧
  ¬HasSameRangeForDifferentDomains exp2 ∧
  ¬HasSameRangeForDifferentDomains log := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_homologous_functions_l87_8760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_distance_theorem_l87_8725

/-- Represents a line in a plane -/
structure Line where
  id : String

/-- Represents the distance between two lines -/
noncomputable def distance (l1 l2 : Line) : ℝ := sorry

/-- Represents that two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

theorem parallel_distance_theorem 
  (a b c : Line) 
  (h_parallel_ab : parallel a b)
  (h_parallel_bc : parallel b c)
  (h_parallel_ac : parallel a c)
  (h_dist_ab : distance a b = 4)
  (h_dist_bc : distance b c = 1) :
  distance a c = 3 ∨ distance a c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_distance_theorem_l87_8725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_range_l87_8797

/-- A function that represents the equation of a conic section -/
def conic_equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (1 + k) + y^2 / (1 - k) = 1

/-- The condition for the equation to represent a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (1 + k > 0 ∧ 1 - k < 0) ∨ (1 + k < 0 ∧ 1 - k > 0)

/-- The theorem stating the range of k for which the equation represents a hyperbola -/
theorem hyperbola_range (k : ℝ) : 
  is_hyperbola k ↔ k ∈ Set.Ioi 1 ∪ Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_range_l87_8797
