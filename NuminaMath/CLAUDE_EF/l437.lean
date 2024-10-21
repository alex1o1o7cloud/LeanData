import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_completes_work_in_50_days_l437_43740

/-- The number of days needed for r to complete the work alone -/
noncomputable def r_days : ℝ := 50

/-- The rate at which p can do the work -/
noncomputable def p_rate : ℝ := 1 / r_days

/-- The rate at which q can do the work -/
noncomputable def q_rate : ℝ := 1 / 24.999999999999996

/-- The rate at which r can do the work -/
noncomputable def r_rate : ℝ := 1 / r_days

/-- p can do the work in the same time as q and r together -/
axiom p_equals_q_plus_r : p_rate = q_rate + r_rate

/-- p and q together can complete the work in 10 days -/
axiom p_and_q_in_10_days : p_rate + q_rate = 1 / 10

theorem r_completes_work_in_50_days :
  r_rate = 1 / r_days := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_completes_work_in_50_days_l437_43740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l437_43771

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x + Real.sqrt (3 - 3*x)

-- State the theorem
theorem f_range : 
  (∀ y : ℝ, (∃ x : ℝ, f x = y) → 1 ≤ y ∧ y ≤ 2) ∧ 
  (∃ x₁ x₂ : ℝ, f x₁ = 1 ∧ f x₂ = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l437_43771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_four_is_one_twelfth_l437_43730

/-- A fair six-sided die -/
def Die : Finset (Fin 6) := Finset.univ

/-- The set of all possible outcomes when rolling two dice -/
def TwoRolls : Finset (Fin 6 × Fin 6) := Die.product Die

/-- The sum of two dice rolls -/
def sum_rolls (roll : Fin 6 × Fin 6) : Nat :=
  roll.1.val + roll.2.val + 2

/-- The set of all outcomes where the sum is 4 -/
def sum_is_four : Finset (Fin 6 × Fin 6) :=
  TwoRolls.filter (λ roll => sum_rolls roll = 4)

/-- The probability of an event in a finite probability space -/
def prob (event : Finset (Fin 6 × Fin 6)) : Rat :=
  event.card / TwoRolls.card

theorem prob_sum_four_is_one_twelfth :
  prob sum_is_four = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_four_is_one_twelfth_l437_43730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l437_43741

/-- Calculates the average speed given distances traveled in two consecutive hours -/
noncomputable def averageSpeed (distance1 : ℝ) (distance2 : ℝ) : ℝ :=
  (distance1 + distance2) / 2

theorem car_average_speed :
  let distance1 : ℝ := 100  -- Distance traveled in the first hour
  let distance2 : ℝ := 30   -- Distance traveled in the second hour
  averageSpeed distance1 distance2 = 65
  := by
    -- Unfold the definition of averageSpeed
    unfold averageSpeed
    -- Simplify the expression
    simp
    -- The proof is completed with sorry for now
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l437_43741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_range_implies_a_eq_neg_one_l437_43784

/-- A function f from ℝ to ℝ defined by a parameter a -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

/-- The theorem stating that if f has domain and range ℝ, then a = -1 -/
theorem f_domain_range_implies_a_eq_neg_one (a : ℝ) :
  (∀ x, ∃ y, f a y = x) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_range_implies_a_eq_neg_one_l437_43784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_sum_l437_43774

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (Real.sqrt 100) * x₁^3 - 210 * x₁^2 + 3 = 0 ∧
  (Real.sqrt 100) * x₂^3 - 210 * x₂^2 + 3 = 0 ∧
  (Real.sqrt 100) * x₃^3 - 210 * x₃^2 + 3 = 0 →
  x₂ * (x₁ + x₃) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_sum_l437_43774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l437_43790

theorem polynomial_divisibility (k : ℝ) : 
  (∃ q : Polynomial ℝ, 2 • X^3 - 8 • X^2 + k • X - (10 : Polynomial ℝ) = (X - 2) * q) ↔ k = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l437_43790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_45_l437_43750

-- Define the clock
def clock_degrees : ℚ := 360

-- Define the number of hours on the clock
def clock_hours : ℕ := 12

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the time
def hours : ℚ := 3
def minutes : ℚ := 45

-- Function to calculate the position of the hour hand
noncomputable def hour_hand_position (h : ℚ) (m : ℚ) : ℚ :=
  (h * (clock_degrees / clock_hours)) + (m * (clock_degrees / clock_hours / minutes_per_hour))

-- Function to calculate the position of the minute hand
noncomputable def minute_hand_position (m : ℚ) : ℚ :=
  m * (clock_degrees / minutes_per_hour)

-- Function to calculate the smaller angle between two positions on the clock
noncomputable def smaller_angle (pos1 : ℚ) (pos2 : ℚ) : ℚ :=
  min (abs (pos1 - pos2)) (clock_degrees - abs (pos1 - pos2))

-- Theorem statement
theorem clock_angle_at_3_45 :
  smaller_angle (hour_hand_position hours minutes) (minute_hand_position minutes) = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_45_l437_43750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_subsegment_length_l437_43794

-- Define the triangle
structure Triangle (α : Type*) [LinearOrderedField α] where
  x : α
  y : α
  z : α
  side_ratio : x / y = 3 / 4 ∧ y / z = 4 / 5

-- Define the angle bisector
def angle_bisector {α : Type*} [LinearOrderedField α] (t : Triangle α) (e : α) : Prop :=
  e > 0 ∧ e < t.z ∧ (t.x / (t.z - e) = t.y / e)

-- Main theorem
theorem shortest_subsegment_length 
  {α : Type*} [LinearOrderedField α]
  (t : Triangle α) 
  (e : α) 
  (h1 : angle_bisector t e) 
  (h2 : t.z = 12) : 
  min e (t.z - e) = 36 / 7 := by
  sorry

-- Example usage
example : ∃ (t : Triangle ℝ) (e : ℝ), 
  angle_bisector t e ∧ t.z = 12 ∧ min e (t.z - e) = 36 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_subsegment_length_l437_43794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_acute_triangles_is_seven_l437_43747

/-- An isosceles triangle with an angle of 108° between its equal sides -/
structure IsoscelesTriangle108 where
  /-- The triangle is isosceles -/
  isIsosceles : Bool
  /-- The angle between the equal sides is 108° -/
  angleBetweenEqualSides : ℝ
  /-- The angle between the equal sides is 108° -/
  angleBetweenEqualSides_eq : angleBetweenEqualSides = 108

/-- A function that returns the minimum number of acute-angled triangles needed to divide the given triangle -/
def minAcuteTriangles (t : IsoscelesTriangle108) : ℕ := 7

/-- Theorem stating that the minimum number of acute-angled triangles is 7 -/
theorem min_acute_triangles_is_seven (t : IsoscelesTriangle108) :
  minAcuteTriangles t = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_acute_triangles_is_seven_l437_43747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_iff_t_equals_four_fifths_l437_43715

theorem x_equals_y_iff_t_equals_four_fifths (t : ℚ) : 
  (1 - 3 * t = 2 * t - 3) ↔ t = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_iff_t_equals_four_fifths_l437_43715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l437_43732

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

-- State the theorem
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = 2 * Real.pi ∧ ∀ (x : ℝ), f x = f (x + p)) ∧
  (∀ (x : ℝ), f ((-Real.pi/4) + x) = f ((-Real.pi/4) - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l437_43732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_l437_43796

theorem orthogonal_vectors (y : ℝ) : 
  (y = 3/4) ↔ ((-1 : ℝ) * 3 + 4 * y = 0) := by
  sorry

#check orthogonal_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_l437_43796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_exceeds_target_in_seven_years_years_to_target_minimal_l437_43744

/-- The initial production of a factory in 2015 -/
def initial_production : ℕ := 20000

/-- The annual growth rate of production -/
def growth_rate : ℝ := 1.2

/-- The target production to exceed -/
def target_production : ℕ := 60000

/-- The number of years after 2015 when production exceeds the target -/
def years_to_target : ℕ := 7

/-- Predicate to check if production exceeds target after n years -/
def production_exceeds_target (n : ℕ) : Prop :=
  (initial_production : ℝ) * growth_rate ^ n > target_production

theorem production_exceeds_target_in_seven_years :
  production_exceeds_target years_to_target :=
sorry

theorem years_to_target_minimal :
  ∀ m : ℕ, m < years_to_target → ¬production_exceeds_target m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_exceeds_target_in_seven_years_years_to_target_minimal_l437_43744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l437_43736

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The distance between a point and a line -/
noncomputable def distancePointLine (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- The length of a chord in a circle -/
noncomputable def chordLength (r d : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - d^2)

theorem common_chord_length 
  (circle1 : Circle) 
  (circle2 : Circle) 
  (h1 : circle1.equation = λ x y => x^2 + y^2 + x - 2*y - 20 = 0)
  (h2 : circle2.equation = λ x y => x^2 + y^2 = 25) :
  ∃ (chord : ℝ), chord = 4 * Real.sqrt 5 ∧ 
  (∃ (x1 y1 x2 y2 : ℝ), 
    circle1.equation x1 y1 ∧ 
    circle1.equation x2 y2 ∧
    circle2.equation x1 y1 ∧ 
    circle2.equation x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = chord^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l437_43736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_larger_volume_condition_l437_43711

/-- The sum of all dimensions of the suitcase -/
noncomputable def total_dimension : ℝ := 150

/-- The length of the long dimension for the elongated suitcase -/
noncomputable def long_dimension : ℝ := 220

/-- The ratio of the long dimension to each of the other two dimensions -/
noncomputable def k : ℝ := long_dimension / ((total_dimension - long_dimension) / 2)

/-- The volume of a cubic suitcase -/
noncomputable def cubic_volume : ℝ := (total_dimension / 3) ^ 3

/-- The volume of a long suitcase -/
noncomputable def long_volume : ℝ := long_dimension ^ 3 / k ^ 2

theorem cubic_larger_volume_condition :
  cubic_volume > long_volume ↔ k > (4.4 : ℝ) ^ (3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_larger_volume_condition_l437_43711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_of_cube_vertices_l437_43772

theorem sphere_surface_area_of_cube_vertices (cube_volume : ℝ) (h1 : cube_volume = 8) :
  ∃ (sphere_radius : ℝ),
    (cube_volume ^ (1/3 : ℝ) * (3 : ℝ).sqrt = 2 * sphere_radius) ∧
    (4 * Real.pi * sphere_radius^2 = 12 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_of_cube_vertices_l437_43772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l437_43716

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then 3 * a * x + 1
  else if 0 ≤ x ∧ x ≤ 3 then x^2 + 1
  else 3 * x - b

theorem continuous_piecewise_function_sum (a b : ℝ) :
  Continuous (f a b) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l437_43716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_diametrically_opposite_points_l437_43733

/-- A circle with 3k points dividing its circumference into arcs of lengths 1, 2, and 3 -/
structure CircleWithArcs (k : ℕ) where
  points : Finset (ℝ × ℝ)
  arcs : Finset (ℝ × ℝ × ℝ)
  point_count : points.card = 3 * k
  arc_count : arcs.card = 3 * k
  arc_lengths : ∃ (a b c : Finset (ℝ × ℝ × ℝ)), 
    a.card = k ∧ b.card = k ∧ c.card = k ∧
    (∀ x ∈ a, x.2 = 1) ∧ 
    (∀ x ∈ b, x.2 = 2) ∧ 
    (∀ x ∈ c, x.2 = 3) ∧
    a ∪ b ∪ c = arcs

/-- Two points are diametrically opposite if their distance equals the diameter of the circle -/
def DiametricallyOpposite (p q : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4 -- Assuming circle with radius 1

/-- There exist two diametrically opposite points in the circle -/
theorem exists_diametrically_opposite_points (k : ℕ) (circle : CircleWithArcs k) :
  ∃ p q, p ∈ circle.points ∧ q ∈ circle.points ∧ DiametricallyOpposite p q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_diametrically_opposite_points_l437_43733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_1_proposition_2_proposition_3_proposition_4_l437_43700

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Axioms for the relations
axiom perpendicular_def (l : Line) (p : Plane) : 
  perpendicular l p ↔ perpendicular l p

axiom parallel_def (l : Line) (p : Plane) :
  parallel l p ↔ parallel l p

axiom contains_def (p : Plane) (l : Line) :
  contains p l ↔ contains p l

axiom plane_perpendicular_def (p1 p2 : Plane) :
  plane_perpendicular p1 p2 ↔ plane_perpendicular p1 p2

axiom line_parallel_def (l1 l2 : Line) :
  line_parallel l1 l2 ↔ line_parallel l1 l2

-- Theorem statements
theorem proposition_1 (a : Line) (α β : Plane) :
  perpendicular a α → contains β a → plane_perpendicular α β := by sorry

theorem proposition_2 (a : Line) (α β : Plane) :
  ¬(parallel a α → plane_perpendicular α β → perpendicular a β) := by sorry

theorem proposition_3 (a : Line) (α β : Plane) :
  ¬(perpendicular a β → plane_perpendicular α β → parallel a α) := by sorry

theorem proposition_4 (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → line_parallel a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_1_proposition_2_proposition_3_proposition_4_l437_43700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_plus_one_l437_43722

def y : ℕ → ℕ
  | 0 => 100  -- Adding the base case for 0
  | 1 => 100
  | (k + 2) => y (k + 1) ^ 2 + 2 * y (k + 1) + 1

theorem sum_reciprocal_y_plus_one : 
  (∑' k : ℕ, (1 : ℝ) / (y k + 1)) = 1 / 101 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_plus_one_l437_43722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_three_pi_half_plus_theta_l437_43761

theorem tan_three_pi_half_plus_theta 
  (θ : ℝ) 
  (h1 : Real.sin θ = 1/3) 
  (h2 : θ ∈ Set.Ioo (π/2) π) : 
  Real.tan (3*π/2 + θ) = 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_three_pi_half_plus_theta_l437_43761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l437_43745

/-- Helper function to represent the focus of a parabola -/
def focus_of_parabola (a : ℝ) : ℝ × ℝ := (a, 0)

/-- The focus of the parabola y² = 4x is at (1, 0) -/
theorem parabola_focus (x y : ℝ) : y^2 = 4*x → focus_of_parabola 1 = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l437_43745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_half_l437_43749

noncomputable section

/-- The function f(x) with parameters ω and b -/
def f (ω : ℝ) (b : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

/-- The theorem statement -/
theorem f_value_at_pi_half 
  (ω : ℝ) 
  (b : ℝ) 
  (T : ℝ) 
  (h1 : ω > 0)
  (h2 : T = 2 * Real.pi / ω)
  (h3 : 2 * Real.pi / 3 < T)
  (h4 : T < Real.pi)
  (h5 : ∀ x, f ω b x = f ω b (3 * Real.pi - x))
  (h6 : f ω b (3 * Real.pi / 2) = 2) :
  f ω b (Real.pi / 2) = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_half_l437_43749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_game_theorem_l437_43799

/-- Probability that player A rolls on the nth turn -/
def p : ℕ → ℝ := sorry

/-- Probability that player B rolls on the nth turn -/
def q : ℕ → ℝ := sorry

/-- The game conditions -/
axiom game_conditions :
  (p 1 = 1) ∧ (q 1 = 0) ∧
  (∀ n : ℕ, n ≥ 2 → p n = p (n-1) * (1/6) + q (n-1) * (5/6)) ∧
  (∀ n : ℕ, n ≥ 2 → q n = q (n-1) * (1/6) + p (n-1) * (5/6)) ∧
  (∀ n : ℕ, p n + q n = 1)

/-- The main theorem to prove -/
theorem dice_game_theorem :
  (p 2 = 1/6) ∧
  (p 3 = 26/36) ∧
  (∀ n : ℕ, n ≥ 2 → p n - q n = (-2/3)^(n-1)) ∧
  (∀ n : ℕ, n ≥ 1 → p n = 1/2 * ((-2/3)^(n-1) + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_game_theorem_l437_43799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l437_43742

noncomputable def p (k : ℝ) : Prop := k^2 + 3*k - 4 ≤ 0

noncomputable def f (k x : ℝ) : ℝ := (1/2) * x^2 + k*x + Real.log x

def q (k : ℝ) : Prop := ∀ x > 0, ∀ y > x, f k y > f k x

theorem k_range (k : ℝ) : 
  (¬(p k ∧ q k) ∧ (p k ∨ q k)) → (k ∈ Set.Icc (-4) (-2) ∪ Set.Ioi 1) := by
  sorry

#check k_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l437_43742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l437_43779

-- Define the power function f(x) = x^α
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

-- Theorem statement
theorem power_function_inequality (α : ℝ) (h : f α 2 = Real.sqrt 2) :
  {a : ℝ | f α a < f α (a + 1)} = Set.Ici (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l437_43779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_equal_division_l437_43723

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line segment in a 2D plane -/
structure Segment where
  start : Point
  endpoint : Point

/-- Represents an intersection point between two segments -/
structure Intersection where
  point : Point
  segment1 : Segment
  segment2 : Segment

/-- Represents a closed self-intersecting polygonal line -/
structure PolygonalLine where
  segments : List Segment
  intersections : List Intersection

/-- Checks if a point divides a segment into equal halves -/
def divides_equally (p : Point) (s : Segment) : Prop :=
  let midpoint : Point := ⟨(s.start.x + s.endpoint.x) / 2, (s.start.y + s.endpoint.y) / 2⟩
  p = midpoint

/-- Main theorem: It's impossible for all intersection points to divide their segments equally -/
theorem impossibility_of_equal_division (pl : PolygonalLine) : 
  ¬ (∀ i ∈ pl.intersections, divides_equally i.point i.segment1 ∧ divides_equally i.point i.segment2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_equal_division_l437_43723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_vertices_l437_43782

/-- Calculates the area of a square given two adjacent vertices. -/
def area_of_square_from_adjacent_vertices (v1 v2 : ℝ × ℝ) : ℝ :=
  let dx := v2.1 - v1.1
  let dy := v2.2 - v1.2
  (dx * dx + dy * dy)  -- Square of the side length

/-- A square with adjacent vertices at (0,3) and (4,0) has an area of 25 square units. -/
theorem square_area_from_vertices : 
  let v1 : ℝ × ℝ := (0, 3)
  let v2 : ℝ × ℝ := (4, 0)
  area_of_square_from_adjacent_vertices v1 v2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_vertices_l437_43782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equality_l437_43728

theorem exponential_equality (x : ℝ) (h : (3 : ℝ)^(x + 1) = 81) : (3 : ℝ)^(2*x + 1) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equality_l437_43728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_m_values_l437_43759

theorem sum_of_m_values : ∃ (S : Finset ℤ), 
  (∀ m ∈ S, 
    (∀ x : ℝ, (x - m) / 2 ≥ 0 ∧ x + 3 < 3 * (x - 1) ↔ x > 3) ∧
    (∃ y : ℕ, (3 - y : ℚ) / (2 - y) + m / (y - 2) = 3)) ∧
  (S.sum id = -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_m_values_l437_43759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sign_around_minimum_l437_43704

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Define the conditions
axiom unique_extremum : ∃! x, ∀ y, f y ≥ f x
axiom local_min_at_one : IsLocalMin f 1

-- State the theorem
theorem derivative_sign_around_minimum :
  (∀ x < 1, HasDerivAt f (f' x) x → f' x ≤ 0) ∧
  (∀ x > 1, HasDerivAt f (f' x) x → f' x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_sign_around_minimum_l437_43704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_of_sequence_l437_43737

def arithmetic_sequence (a₁ a₂ a₃ a₄ a₅ : ℤ) : Prop :=
  ∃ d : ℤ, a₂ - a₁ = d ∧ a₃ - a₂ = d ∧ a₄ - a₃ = d ∧ a₅ - a₄ = d

theorem third_term_of_sequence (a₁ a₂ a₄ a₅ : ℤ) :
  a₁ = 8 → a₂ = 62 → a₄ = -4 → a₅ = -12 →
  ∃ x : ℤ, x = 29 ∧ arithmetic_sequence a₁ a₂ x a₄ a₅ := by
  sorry

#check third_term_of_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_of_sequence_l437_43737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_sqrt_seven_distance_l437_43729

/-- A regular hexagon with side length 1 -/
structure RegularHexagon where
  sideLength : ℝ
  isSideOne : sideLength = 1
  vertices : Finset Point

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  sorry

/-- Theorem: In a regular hexagon with side length 1, there exists a point G 
    constructed by the intersection of lines connecting vertices, such that 
    the distance between G and a vertex of the hexagon is √7 -/
theorem hexagon_sqrt_seven_distance (h : RegularHexagon) :
  ∃ (A G : Point) (l1 l2 : Line),
    (A ∈ h.vertices) ∧ 
    (l1.p1 ∈ h.vertices) ∧ (l1.p2 ∈ h.vertices) ∧
    (l2.p1 ∈ h.vertices) ∧ (l2.p2 ∈ h.vertices) ∧
    (G = intersectionPoint l1 l2) ∧
    (distance A G = Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_sqrt_seven_distance_l437_43729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_double_square_numbers_l437_43738

/-- A positive integer is a "double number" if its decimal representation 
    consists of a block of digits that does not start with 0 followed by 
    an identical block. -/
def is_double_number (n : ℕ+) : Prop :=
  ∃ (k : ℕ+) (d : ℕ+), 
    n = d + d * (10 ^ (Nat.log 10 d + 1)) ∧ 
    10 ≤ d

/-- There exists an infinite set of positive integers that are both 
    "double numbers" and perfect squares. -/
theorem infinite_double_square_numbers : 
  ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    (∀ n ∈ S, is_double_number n ∧ ∃ m : ℕ+, n = m ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_double_square_numbers_l437_43738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_abs_sine_l437_43746

/-- The smallest positive period of y = |5 sin(2x + π/3)| -/
theorem smallest_positive_period_abs_sine :
  ∃ T : ℝ, T > 0 ∧
    (∀ x : ℝ, |5 * Real.sin (2 * x + π/3)| = |5 * Real.sin (2 * (x + T) + π/3)|) ∧
    (∀ T' : ℝ, T' > 0 →
      (∀ x : ℝ, |5 * Real.sin (2 * x + π/3)| = |5 * Real.sin (2 * (x + T') + π/3)|) → T ≤ T') ∧
    T = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_abs_sine_l437_43746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_log_range_l437_43743

/-- Given that 0 < a ≠ 1 and f(x) = log_a(6ax^2 - 2x + 3) is monotonically increasing on [3/2, 2],
    prove that a ∈ (1/24, 1/12] ∪ (1, +∞) -/
theorem monotonic_log_range (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => (Real.log (6 * a * x^2 - 2 * x + 3)) / (Real.log a)
  (∀ x ∈ Set.Icc (3/2) 2, StrictMono f) →
  a ∈ Set.union (Set.Ioc (1/24) (1/12)) (Set.Ioi 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_log_range_l437_43743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MCN_constant_l437_43706

-- Define the circle Ω
variable (Ω : Set (ℝ × ℝ))

-- Define points P, A, B, C, M, N
variable (P A B C M N : ℝ × ℝ)

-- Define the line l
variable (l : Set (ℝ × ℝ))

-- Axioms representing the given conditions
axiom P_outside : P ∉ Ω
axiom l_through_P : P ∈ l
axiom A_on_circle : A ∈ Ω
axiom B_on_circle : B ∈ Ω
axiom A_on_l : A ∈ l
axiom B_on_l : B ∈ l
axiom C_on_AB : C ∈ Set.Icc A B
axiom power_of_point : (P.1 - A.1)^2 + (P.2 - A.2)^2 * ((P.1 - B.1)^2 + (P.2 - B.2)^2) = ((P.1 - C.1)^2 + (P.2 - C.2)^2)^2
axiom M_midpoint : M ∈ Ω ∧ M = (A + B) / 2
axiom N_midpoint : N ∈ Ω ∧ N = (A + B) / 2

-- Define the angle MCN
noncomputable def angle_MCN (C M N : ℝ × ℝ) : ℝ := 
  Real.arccos ((M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2)) / 
  (Real.sqrt ((M.1 - C.1)^2 + (M.2 - C.2)^2) * Real.sqrt ((N.1 - C.1)^2 + (N.2 - C.2)^2))

-- Theorem statement
theorem angle_MCN_constant :
  ∀ l₁ l₂ : Set (ℝ × ℝ), 
  (P ∈ l₁ ∧ P ∈ l₂) → 
  (∃ A₁ B₁ C₁ M₁ N₁, 
    A₁ ∈ Ω ∧ B₁ ∈ Ω ∧ A₁ ∈ l₁ ∧ B₁ ∈ l₁ ∧ 
    C₁ ∈ Set.Icc A₁ B₁ ∧
    (P.1 - A₁.1)^2 + (P.2 - A₁.2)^2 * ((P.1 - B₁.1)^2 + (P.2 - B₁.2)^2) = ((P.1 - C₁.1)^2 + (P.2 - C₁.2)^2)^2 ∧
    M₁ ∈ Ω ∧ M₁ = (A₁ + B₁) / 2 ∧
    N₁ ∈ Ω ∧ N₁ = (A₁ + B₁) / 2) →
  (∃ A₂ B₂ C₂ M₂ N₂, 
    A₂ ∈ Ω ∧ B₂ ∈ Ω ∧ A₂ ∈ l₂ ∧ B₂ ∈ l₂ ∧ 
    C₂ ∈ Set.Icc A₂ B₂ ∧
    (P.1 - A₂.1)^2 + (P.2 - A₂.2)^2 * ((P.1 - B₂.1)^2 + (P.2 - B₂.2)^2) = ((P.1 - C₂.1)^2 + (P.2 - C₂.2)^2)^2 ∧
    M₂ ∈ Ω ∧ M₂ = (A₂ + B₂) / 2 ∧
    N₂ ∈ Ω ∧ N₂ = (A₂ + B₂) / 2) →
  angle_MCN C₁ M₁ N₁ = angle_MCN C₂ M₂ N₂ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MCN_constant_l437_43706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_l437_43783

theorem function_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + f y) = (x + y) * f (x + y)) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_l437_43783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l437_43705

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
structure SimilarTriangles (Triangle1 Triangle2 : Type) :=
  (ratio : ℝ)
  (side_ratios : ∀ (s1 : ℝ) (s2 : ℝ), ratio = s1 / s2)

/-- Given two similar triangles HIJ and KLM, prove that if HI = 9, IJ = 12, and KL = 6, then LM = 8 -/
theorem similar_triangles_side_length 
  (HIJ KLM : Type) 
  (sim : SimilarTriangles HIJ KLM) 
  (HI IJ KL LM : ℝ) 
  (h1 : HI = 9) 
  (h2 : IJ = 12) 
  (h3 : KL = 6) : 
  LM = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l437_43705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_l437_43702

theorem matrix_determinant (a b c : ℝ) : 
  Matrix.det !![2, a, b;
                 2, a + b, b + c;
                 2, a, a + c] = 2 * (a * b + b * c - b^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_l437_43702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_totalWatchTime5Shows_l437_43785

def showDuration (n : ℕ) : ℚ :=
  30 * (3/2) ^ n

def totalWatchTime (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) showDuration

theorem totalWatchTime5Shows :
  totalWatchTime 5 = 395625 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_totalWatchTime5Shows_l437_43785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y5_l437_43765

def binomial_expansion (a b : ℚ) (n : ℕ) : ℚ → ℚ → ℚ :=
  λ x y ↦ (a * x + b * y) ^ n

theorem coefficient_x3y5 :
  ∃ (c : ℚ), c = -12500 / 123 ∧
  ∀ (x y : ℚ),
    binomial_expansion (2/3) (-5/4) 8 x y =
    c * x^3 * y^5 + (binomial_expansion (2/3) (-5/4) 8 x y - c * x^3 * y^5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y5_l437_43765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l437_43755

theorem first_group_size (total_count : ℕ) (total_avg : ℚ) (first_group_avg : ℚ) 
  (last_group_count : ℕ) (last_group_avg : ℚ) (fifth_result : ℚ) 
  (h1 : total_count = 11)
  (h2 : total_avg = 42)
  (h3 : first_group_avg = 49)
  (h4 : last_group_count = 7)
  (h5 : last_group_avg = 52)
  (h6 : fifth_result = 147)
  : (total_count - last_group_count : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l437_43755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_sheet_area_copper_sheet_area_m2_l437_43797

/-- The area of a sheet rolled from a copper billet -/
theorem copper_sheet_area (length width height thickness : ℝ) (h1 : length = 80) 
  (h2 : width = 20) (h3 : height = 5) (h4 : thickness = 0.1) : 
  (length * width * height) / thickness = 80000 := by
  sorry

/-- Convert square centimeters to square meters -/
noncomputable def cm2_to_m2 (area_cm2 : ℝ) : ℝ := area_cm2 / 10000

/-- The area of the copper sheet in square meters -/
theorem copper_sheet_area_m2 (length width height thickness : ℝ) (h1 : length = 80) 
  (h2 : width = 20) (h3 : height = 5) (h4 : thickness = 0.1) : 
  cm2_to_m2 ((length * width * height) / thickness) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_sheet_area_copper_sheet_area_m2_l437_43797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_delivery_count_l437_43727

def delivery_sequences : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| n + 3 => delivery_sequences (n + 2) + delivery_sequences (n + 1) + delivery_sequences n

theorem paperboy_delivery_count :
  delivery_sequences 10 = 504 := by
  rfl

#eval delivery_sequences 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_delivery_count_l437_43727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_ratio_l437_43757

/-- The perimeter of both shapes -/
noncomputable def perimeter : ℝ := 60

/-- The number of sides in an equilateral triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a regular pentagon -/
def pentagon_sides : ℕ := 5

/-- The side length of the equilateral triangle -/
noncomputable def triangle_side_length : ℝ := perimeter / triangle_sides

/-- The side length of the regular pentagon -/
noncomputable def pentagon_side_length : ℝ := perimeter / pentagon_sides

theorem side_length_ratio :
  triangle_side_length / pentagon_side_length = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_ratio_l437_43757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linlin_cards_l437_43754

/-- The number of cards Linlin has originally -/
def L : ℕ := sorry

/-- The number of cards Tongtong has originally -/
def T : ℕ := sorry

/-- If Tongtong gives 6 cards to Linlin, Linlin will have 3 times as many cards as Tongtong -/
axiom condition1 : L + 6 = 3 * (T - 6)

/-- If Linlin gives 2 cards to Tongtong, Linlin will have 2 times as many cards as Tongtong -/
axiom condition2 : T + 2 = 2 * (L - 2)

/-- Prove that Linlin originally has 46 cards -/
theorem linlin_cards : L = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linlin_cards_l437_43754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_exists_expectation_is_correct_l437_43786

noncomputable def excellent_timely : ℕ := 24
noncomputable def excellent_not_timely : ℕ := 8
noncomputable def not_excellent_timely : ℕ := 6
noncomputable def not_excellent_not_timely : ℕ := 22
noncomputable def total_sample : ℕ := 60
noncomputable def critical_value : ℝ := 10.828

noncomputable def chi_square : ℝ :=
  (total_sample * (excellent_timely * not_excellent_not_timely - not_excellent_timely * excellent_not_timely)^2 : ℝ) /
  ((excellent_timely + not_excellent_timely) * (excellent_not_timely + not_excellent_not_timely) *
   (excellent_timely + excellent_not_timely) * (not_excellent_timely + not_excellent_not_timely) : ℝ)

theorem relationship_exists : chi_square > critical_value := by sorry

noncomputable def N : ℕ := 8  -- Total number of students with excellent math performance
noncomputable def K : ℕ := 6  -- Number of students with excellent math performance who review timely
noncomputable def n : ℕ := 3  -- Number of students selected

noncomputable def hypergeometric_expectation : ℚ := (n : ℚ) * (K : ℚ) / (N : ℚ)

theorem expectation_is_correct : hypergeometric_expectation = 9/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_exists_expectation_is_correct_l437_43786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_positive_l437_43768

-- Define α as a real number
variable (α : ℝ)

-- Define the condition that α is not a multiple of π/2
def not_multiple_of_pi_half (α : ℝ) : Prop :=
  ∀ k : ℤ, α ≠ k * (Real.pi / 2)

-- Define T as a function of α
noncomputable def T (α : ℝ) : ℝ :=
  (Real.sin α + Real.tan α) / (Real.cos α + (1 / Real.tan α))

-- Theorem statement
theorem T_is_positive (α : ℝ) (h : not_multiple_of_pi_half α) : T α > 0 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_positive_l437_43768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_four_statements_can_be_true_max_true_statements_is_four_l437_43720

theorem max_true_statements (c d : ℝ) : 
  ¬(1/c < 1/d ∧ c^3 > d^3 ∧ c < d ∧ c < 0 ∧ d < 0) :=
by sorry

theorem four_statements_can_be_true (c d : ℝ) :
  ∃ (c d : ℝ), c^3 > d^3 ∧ c < d ∧ c < 0 ∧ d < 0 :=
by sorry

theorem max_true_statements_is_four :
  ∃ (n : ℕ), n = 4 ∧
  (∀ (c d : ℝ), 
    (if 1/c < 1/d then 1 else 0) + 
    (if c^3 > d^3 then 1 else 0) + 
    (if c < d then 1 else 0) + 
    (if c < 0 then 1 else 0) + 
    (if d < 0 then 1 else 0) ≤ n) ∧
  (∃ (c d : ℝ), 
    (if 1/c < 1/d then 1 else 0) + 
    (if c^3 > d^3 then 1 else 0) + 
    (if c < d then 1 else 0) + 
    (if c < 0 then 1 else 0) + 
    (if d < 0 then 1 else 0) = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_four_statements_can_be_true_max_true_statements_is_four_l437_43720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_remainder_one_l437_43760

theorem smallest_integer_with_remainder_one (n : ℕ) : 
  (∀ k ∈ ({4, 5, 6, 7, 8, 9, 10} : Set ℕ), n % k = 1) ∧ 
  (∀ m : ℕ, m > 1 ∧ m < n → ∃ k ∈ ({4, 5, 6, 7, 8, 9, 10} : Set ℕ), m % k ≠ 1) →
  n = 2521 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_remainder_one_l437_43760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l437_43725

/-- Given a quadratic function y = ax^2 - 4ax + 1, if points P(1, m) and Q(2a+2, n) lie on the graph of y and m > n, then the range of values for a is 0 < a < 1/2 or a < -1/2 -/
theorem quadratic_function_range (a m n : ℝ) : 
  let y : ℝ → ℝ := fun x ↦ a * x^2 - 4 * a * x + 1
  (y 1 = m) → 
  (y (2 * a + 2) = n) → 
  (m > n) → 
  ((0 < a ∧ a < 1/2) ∨ a < -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l437_43725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_perpendicular_vectors_l437_43721

def A : Fin 3 → ℝ := ![(-2), 0, 2]
def B : Fin 3 → ℝ := ![(-1), 1, 2]
def C : Fin 3 → ℝ := ![(-3), 0, 4]

def a : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]
def b : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

theorem vector_angle_cosine :
  dot_product a b / (magnitude a * magnitude b) = -Real.sqrt 10 / 10 := by sorry

theorem perpendicular_vectors (k : ℝ) :
  dot_product (fun i => k * (a i) + b i) (fun i => k * (a i) - 2 * (b i)) = 0 ↔ k = -5/2 ∨ k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_perpendicular_vectors_l437_43721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_periodic_l437_43735

def prime_factor_sum (n : ℕ) : ℕ := sorry

def f (n : ℕ) : ℕ := prime_factor_sum n + 1

def sequence_f (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => f (sequence_f start n)

def eventually_periodic (seq : ℕ → ℕ) : Prop :=
  ∃ (n m : ℕ), ∀ k, k ≥ n → seq (k + m) = seq k

theorem sequence_eventually_periodic :
  ∀ K : ℕ, K ≥ 9 → eventually_periodic (sequence_f K) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_periodic_l437_43735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l437_43701

/-- For an infinite geometric series with first term a and sum S, the common ratio r is given by r = (S - a) / S -/
noncomputable def geometric_series_ratio (a : ℝ) (S : ℝ) : ℝ := (S - a) / S

/-- The problem statement -/
theorem infinite_geometric_series_ratio :
  let a : ℝ := 172
  let S : ℝ := 400
  geometric_series_ratio a S = 57 / 100 := by
  -- Unfold the definition of geometric_series_ratio
  unfold geometric_series_ratio
  -- Simplify the expression
  simp
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l437_43701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_functions_properties_l437_43717

/-- b(n) denotes the number of positive divisors of n -/
def b (n : ℕ+) : ℕ := sorry

/-- p(n) denotes the sum of all positive divisors of n -/
def p (n : ℕ+) : ℕ := sorry

theorem divisor_functions_properties (k : ℕ+) (h : k > 1) :
  (Set.Infinite {n : ℕ+ | b n = k^2 - k + 1}) ∧
  (Set.Finite {n : ℕ+ | p n = k^2 - k + 1}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_functions_properties_l437_43717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l437_43709

-- Define the function f
noncomputable def f (α : Real) : Real :=
  (Real.sin (α - Real.pi/2) * Real.cos (3*Real.pi/2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

-- Theorem statement
theorem f_simplification_and_value (α : Real) :
  (α > Real.pi ∧ α < 3*Real.pi/2) →  -- α is in the third quadrant
  (f α = -Real.cos α) ∧
  (Real.cos (α - 3*Real.pi/2) = 3/5 → f α = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l437_43709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l437_43724

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (3, m)

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vector_angle_problem (m : ℝ) :
  angle_between a (b m) = π / 6 → m = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l437_43724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l437_43708

theorem quadratic_root_difference (p q : ℕ) : 
  (∃ x y : ℝ, 5 * x^2 - 2 * x - 15 = 0 ∧ 
              5 * y^2 - 2 * y - 15 = 0 ∧ 
              x ≠ y ∧ 
              |x - y| = (Real.sqrt (p : ℝ)) / (q : ℝ)) →
  (q > 0) →
  (∀ (prime : ℕ), Nat.Prime prime → ¬(prime^2 ∣ p)) →
  p + q = 309 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l437_43708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_numbers_appear_approx_l437_43751

/-- The probability of each number from 1 to 6 appearing at least once when 10 fair dice are thrown -/
noncomputable def probability_all_numbers_appear : ℝ :=
  1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10)

/-- Theorem stating that the probability of each number from 1 to 6 appearing at least once
    when 10 fair dice are thrown is approximately 0.729 -/
theorem probability_all_numbers_appear_approx :
  abs (probability_all_numbers_appear - 0.729) < 0.001 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_numbers_appear_approx_l437_43751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kenny_lawn_mowing_earnings_l437_43777

/-- The amount Kenny charges per lawn -/
def charge_per_lawn : ℚ := 15

/-- The number of lawns Kenny mows -/
def lawns_mowed : ℕ := 35

/-- The number of video games Kenny buys -/
def video_games_bought : ℕ := 5

/-- The cost of each video game -/
def video_game_cost : ℚ := 45

/-- The number of books Kenny buys -/
def books_bought : ℕ := 60

/-- The cost of each book -/
def book_cost : ℚ := 5

theorem kenny_lawn_mowing_earnings :
  charge_per_lawn = 15 :=
by
  -- Define total earnings
  let total_earnings : ℚ := charge_per_lawn * lawns_mowed
  
  -- Define costs for video games and books
  let video_games_cost : ℚ := video_games_bought * video_game_cost
  let books_cost : ℚ := books_bought * book_cost
  
  -- Define total spent
  let total_spent : ℚ := video_games_cost + books_cost
  
  -- Assert that earnings equal spending
  have earnings_equal_spending : total_earnings = total_spent := by sorry
  
  -- The main proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kenny_lawn_mowing_earnings_l437_43777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_a_b_l437_43719

theorem min_abs_diff_a_b (a b : ℕ) (h : a * b + 4 * a - 5 * b = 90) :
  ∃ (c d : ℕ), c * d + 4 * c - 5 * d = 90 ∧ 
  (∀ (x y : ℕ), x * y + 4 * x - 5 * y = 90 → |Int.ofNat x - Int.ofNat y| ≥ |Int.ofNat c - Int.ofNat d|) ∧
  |Int.ofNat c - Int.ofNat d| = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_a_b_l437_43719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_104_98_l437_43791

theorem abs_diff_squares_104_98 : |((104 : ℤ)^2 - (98 : ℤ)^2)| = 1212 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_104_98_l437_43791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_imply_B_l437_43713

def polynomial (A B : ℤ) (z : ℝ) : ℝ := z^4 - 6*z^3 + A*z^2 + B*z + 9

def has_roots (p : ℝ → ℝ) (roots : List ℝ) : Prop :=
  ∀ z, p z = 0 ↔ z ∈ roots

def all_positive_integers (roots : List ℝ) : Prop :=
  ∀ r, r ∈ roots → r > 0 ∧ ∃ n : ℤ, r = n

theorem polynomial_roots_imply_B (A B : ℤ) :
  let p := polynomial A B
  let roots := [2, 2, 1, 1]
  has_roots p roots ∧ all_positive_integers roots →
  B = -13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_imply_B_l437_43713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l437_43766

theorem count_integers_in_pi_range : 
  (Finset.range (Int.toNat (Int.floor (7 * Real.pi) - Int.ceil (-5 * Real.pi) + 1))).card = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l437_43766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l437_43739

/-- Given a function f: ℝ → ℝ with a tangent line at x=2 described by the equation 2x+y-3=0,
    prove that f(2) + f'(2) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, 2*x + y - 3 = 0 ↔ y = f 2 + (deriv f) 2 * (x - 2)) :
  f 2 + (deriv f) 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l437_43739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_iff_divisible_by_three_l437_43703

def sequence_a : ℕ → ℕ
  | 0 => 8  -- Adding case for 0
  | 1 => 18
  | (n + 2) => sequence_a (n + 1) * sequence_a n

theorem perfect_square_iff_divisible_by_three (n : ℕ) :
  (∃ m : ℕ, sequence_a n = m ^ 2) ↔ 3 ∣ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_iff_divisible_by_three_l437_43703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrigation_tank_problem_l437_43714

/-- Calculates the remaining water in a tank after a given number of days -/
def remainingWater (initialAmount : ℝ) (evaporationRate : ℝ) (additionRate : ℝ) (days : ℕ) : ℝ :=
  initialAmount + (additionRate - evaporationRate) * (days : ℝ)

/-- The problem statement -/
theorem irrigation_tank_problem :
  let initialAmount : ℝ := 300
  let evaporationRate : ℝ := 1
  let additionRate : ℝ := 0.3
  let days : ℕ := 45
  remainingWater initialAmount evaporationRate additionRate days = 268.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrigation_tank_problem_l437_43714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_13_l437_43788

theorem three_digit_numbers_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 900)).card = 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_13_l437_43788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_all_quadrants_l437_43769

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + a * x^2 - 2 * a * x + 2 * a + 1

-- State the theorem
theorem function_passes_through_all_quadrants (a : ℝ) :
  (∀ x y : ℝ, ∃ z : ℝ, f a z = x ∧ f a (-z) = y) ↔ -1/3 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_all_quadrants_l437_43769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrap_counts_l437_43792

/-- A package is a 3D shape that can be wrapped with string. -/
class Package :=
  (wrap : ℕ)

/-- A cube is a package with all sides equal. -/
def Cube : Package :=
  { wrap := 3 }

/-- A rectangular prism with two equal sides and one double side. -/
def RectPrism1 : Package :=
  { wrap := 5 }

/-- A rectangular prism with sides in the ratio 1:2:3. -/
def RectPrism2 : Package :=
  { wrap := 7 }

/-- The number of distinct ways to wrap a package along its midplanes. -/
def wrapCount (p : Package) : ℕ := p.wrap

theorem wrap_counts :
  (wrapCount Cube = 3) ∧
  (wrapCount RectPrism1 = 5) ∧
  (wrapCount RectPrism2 = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrap_counts_l437_43792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l437_43764

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a - 1) / (x^2 + 1)

-- Theorem statement
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a x = -f a (-x)) ∧ -- f is odd on [-1, 1]
    (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f a x < f a y) ∧ -- f is increasing on [-1, 1]
    (∀ t : ℝ, f a (t - 1) + f a (2 * t) < 0 ↔ 0 ≤ t ∧ t < 1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l437_43764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_purchase_cost_l437_43762

/-- The price of type A equipment in terms of type B -/
noncomputable def price_ratio : ℝ := 1.2

/-- The total number of units to be purchased -/
def total_units : ℕ := 30

/-- The minimum ratio of type A to type B equipment -/
noncomputable def min_ratio : ℝ := 1/4

/-- The unit price of type B equipment -/
noncomputable def price_B : ℝ := 2500

/-- The unit price of type A equipment -/
noncomputable def price_A : ℝ := price_B * price_ratio

/-- The cost function in terms of the number of type A units purchased -/
noncomputable def cost (a : ℝ) : ℝ := price_A * a + price_B * (total_units - a)

/-- The minimum number of type A units that can be purchased -/
noncomputable def min_A : ℝ := min_ratio * (total_units : ℝ) / (1 + min_ratio)

theorem min_purchase_cost :
  ∃ (a : ℝ), a ≥ min_A ∧ a ≤ total_units ∧ cost a = 78000 ∧
  ∀ (b : ℝ), b ≥ min_A ∧ b ≤ total_units → cost b ≥ 78000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_purchase_cost_l437_43762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_l437_43793

theorem max_tan_B (A B : Real) (h_acute_A : 0 < A ∧ A < Real.pi / 2) (h_acute_B : 0 < B ∧ B < Real.pi / 2) 
  (h_tan_sum : Real.tan (A + B) = 2 * Real.tan A) : 
  ∀ B', (0 < B' ∧ B' < Real.pi / 2 ∧ Real.tan (A + B') = 2 * Real.tan A) → Real.tan B ≤ Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_l437_43793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l437_43778

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter condition
def perimeter_condition (a b c : ℝ) : Prop :=
  a + b + c = 10 + 2 * Real.sqrt 7

-- Define the sine ratio condition
def sine_ratio_condition (a b c : ℝ) : Prop :=
  a / 2 = b / Real.sqrt 7 ∧ b / Real.sqrt 7 = c / 3

-- Define the area formula
noncomputable def area_formula (a b c : ℝ) : ℝ :=
  Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2))

-- Theorem statement
theorem triangle_area_proof (a b c : ℝ) :
  triangle_ABC a b c →
  perimeter_condition a b c →
  sine_ratio_condition a b c →
  area_formula a b c = 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l437_43778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coefficients_l437_43718

theorem sum_of_y_coefficients (x y : ℝ) : 
  let expanded := (3*x + 2*y + 1) * (x + 4*y + 5)
  let y_terms := [14*x, 8*y, 14]
  y_terms.sum = 36 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coefficients_l437_43718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_roots_P_of_P_l437_43775

-- Define the polynomial P
variable (n : ℕ) (P : ℤ → ℤ)

-- Define the conditions
variable (hn : n ≥ 5)
variable (hdegree : ∀ (k : ℕ), k > n → ∀ (x : ℤ), (P x).natAbs ≤ x.natAbs ^ k)
variable (hroots : ∃ (roots : Finset ℤ), roots.card = n ∧ (∀ x ∈ roots, P x = 0) ∧ (∀ x : ℤ, P x = 0 → x ∈ roots))
variable (hzero : P 0 = 0)

-- State the theorem
theorem num_roots_P_of_P (n : ℕ) (P : ℤ → ℤ) 
  (hn : n ≥ 5)
  (hdegree : ∀ (k : ℕ), k > n → ∀ (x : ℤ), (P x).natAbs ≤ x.natAbs ^ k)
  (hroots : ∃ (roots : Finset ℤ), roots.card = n ∧ (∀ x ∈ roots, P x = 0) ∧ (∀ x : ℤ, P x = 0 → x ∈ roots))
  (hzero : P 0 = 0) :
  ∃ (roots : Finset ℤ), roots.card = n ∧ (∀ x ∈ roots, P (P x) = 0) ∧ (∀ x : ℤ, P (P x) = 0 → x ∈ roots) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_roots_P_of_P_l437_43775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_abs_equation_l437_43773

theorem sum_of_solutions_abs_equation : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, |x - 5|^2 + |x - 5| = 20) ∧ 
                    (∀ x : ℝ, |x - 5|^2 + |x - 5| = 20 → x ∈ S) ∧
                    (S.sum id = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_abs_equation_l437_43773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l437_43726

-- Define the piecewise function
noncomputable def f (b c : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 3 then b * x + 4
  else if 0 ≤ x then x^2 - 1
  else 3 * x - c

-- State the theorem
theorem continuous_piecewise_function_sum (b c : ℝ) :
  (∀ x : ℝ, ContinuousAt (f b c) x) → b + c = 7/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l437_43726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l437_43781

/-- A quadratic radical expression is in its simplest form if each factor within the radicand
    has an exponent less than 2 and the radicand does not include a denominator. -/
noncomputable def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ n > 0 ∧ ¬ (∃ (m : ℕ), m > 1 ∧ m * m ∣ n)

/-- The given quadratic radical expressions -/
noncomputable def expressions : List ℝ := [Real.sqrt (1/2), Real.sqrt 8, Real.sqrt 4, Real.sqrt 5]

theorem simplest_quadratic_radical :
  ∃ (x : ℝ), x ∈ expressions ∧ is_simplest_quadratic_radical x ∧
  ∀ (y : ℝ), y ∈ expressions → is_simplest_quadratic_radical y → y = x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l437_43781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l437_43787

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 12*x^3 + 36*x^4) / (9 - x^3)

-- State the theorem
theorem f_nonnegative_iff (x : ℝ) : f x ≥ 0 ↔ x ∈ Set.Icc 0 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l437_43787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_price_inflation_proof_l437_43798

/-- Calculates the inflation rate given initial and final sugar prices over two years -/
noncomputable def calculate_inflation_rate 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (sugar_rate_increase : ℝ) : ℝ :=
  Real.sqrt (final_price / initial_price) - 1 - sugar_rate_increase

/-- Proves that the calculated inflation rate is correct for the given problem -/
theorem sugar_price_inflation_proof 
  (initial_price : ℝ)
  (final_price : ℝ)
  (sugar_rate_increase : ℝ)
  (h1 : initial_price = 25)
  (h2 : final_price = 33.0625)
  (h3 : sugar_rate_increase = 0.03)
  : calculate_inflation_rate initial_price final_price sugar_rate_increase = 0.12 := by
  sorry

-- This will not be evaluated due to noncomputable definition
-- #eval calculate_inflation_rate 25 33.0625 0.03

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_price_inflation_proof_l437_43798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_implies_a_nonpositive_l437_43748

/-- A function f is decreasing on an interval if for any two points x and y in that interval, 
    x < y implies f(x) ≥ f(y) -/
def DecreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x ≥ f y

theorem decreasing_function_implies_a_nonpositive 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : f = fun x ↦ x^2 + |x - a| + b)
  (h2 : DecreasingOn f (Set.Iic 0)) :
  a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_implies_a_nonpositive_l437_43748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l437_43763

/-- The distance between two points in three-dimensional space -/
noncomputable def distance3D (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem distance_between_specific_points :
  distance3D (1, 4, -3) (-2, 1, 2) = Real.sqrt 43 := by
  -- Unfold the definition of distance3D
  unfold distance3D
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l437_43763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_linear_system_l437_43756

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; 3, 2]
def B : Matrix (Fin 2) (Fin 1) ℝ := !![4; 7]

theorem solve_linear_system :
  ∃ X : Matrix (Fin 2) (Fin 1) ℝ, A * X = B ∧ X = !![1; 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_linear_system_l437_43756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_is_four_l437_43758

/-- Represents the number of workers in the first brigade -/
def n : ℕ := sorry

/-- The second brigade has 6 more workers than the first -/
def second_brigade : ℕ := n + 6

/-- The area of the second road section is three times larger than the first -/
def area_ratio : ℝ := 3

/-- The first brigade completes their work faster -/
axiom faster_completion : (1 : ℝ) / n < area_ratio / second_brigade

/-- The minimum number of workers in the first brigade -/
def min_workers : ℕ := 4

/-- Theorem stating that the minimum number of workers in the first brigade is 4 -/
theorem min_workers_is_four : 
  (∀ m : ℕ, m < min_workers → ¬((1 : ℝ) / m < area_ratio / (m + 6))) ∧
  ((1 : ℝ) / min_workers < area_ratio / (min_workers + 6)) :=
by sorry

#check min_workers_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_is_four_l437_43758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l437_43752

theorem sum_remainder_theorem (a b c : ℕ) 
  (ha : a % 53 = 31)
  (hb : b % 53 = 22)
  (hc : c % 53 = 7) :
  (a + b + c) % 53 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l437_43752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_turns_for_2012_rocks_l437_43731

/-- Represents the rules for removing rocks from the table -/
def remove_rocks (n : ℕ) : ℕ → ℕ
| 0 => 1  -- First turn, remove 1 rock
| m + 1 => max (2 * remove_rocks n m) (remove_rocks n m)  -- Subsequent turns

/-- Calculates the total number of rocks removed after a given number of turns -/
def total_rocks_removed (turns : ℕ) : ℕ :=
  (List.range turns).foldl (λ acc i => acc + remove_rocks 2012 i) 0

/-- The theorem stating that 18 turns is the minimum required to remove 2012 rocks -/
theorem min_turns_for_2012_rocks :
  (∀ k < 18, total_rocks_removed k < 2012) ∧ total_rocks_removed 18 = 2012 := by
  sorry

#eval total_rocks_removed 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_turns_for_2012_rocks_l437_43731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_eight_l437_43770

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem total_distance_equals_eight :
  let p1 : point := (2, 3)
  let p2 : point := (5, 3)
  let p3 : point := (5, -2)
  distance p1 p2 + distance p2 p3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_eight_l437_43770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_double_l437_43780

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Calculates the area of a quadrilateral given its four vertices -/
noncomputable def quadrilateralArea (a b c d : Point) : ℝ := sorry

/-- Constructs a vector from two points -/
def vectorFromPoints (p q : Point) : Vec :=
  { x := q.x - p.x, y := q.y - p.y }

theorem quadrilateral_area_double (a b c d o a' b' c' d' : Point) :
  (vectorFromPoints o a' = vectorFromPoints a b) →
  (vectorFromPoints o b' = vectorFromPoints b c) →
  (vectorFromPoints o c' = vectorFromPoints c d) →
  (vectorFromPoints o d' = vectorFromPoints d a) →
  quadrilateralArea a' b' c' d' = 2 * quadrilateralArea a b c d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_double_l437_43780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l437_43710

/-- Represents the speed of a train in meters per second -/
noncomputable def train_speed (pole_pass_time : ℝ) (crossing_time : ℝ) (stationary_train_length : ℝ) : ℝ :=
  stationary_train_length / (crossing_time - pole_pass_time)

/-- Converts speed from meters per second to kilometers per hour -/
noncomputable def mps_to_kmh (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

theorem train_speed_problem (pole_pass_time crossing_time stationary_train_length : ℝ) 
  (h1 : pole_pass_time = 10)
  (h2 : crossing_time = 35)
  (h3 : stationary_train_length = 500) :
  mps_to_kmh (train_speed pole_pass_time crossing_time stationary_train_length) = 72 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l437_43710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_common_remainder_l437_43753

theorem least_subtraction_for_common_remainder (n : ℕ) : 
  (∀ d : ℕ, d ∈ ({5, 9, 11} : Set ℕ) → (997 - n) % d = 3) ∧ 
  (∀ m : ℕ, m < n → ∃ d : ℕ, d ∈ ({5, 9, 11} : Set ℕ) ∧ (997 - m) % d ≠ 3) → 
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_common_remainder_l437_43753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_final_position_probability_l437_43795

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a random walk in 2D space --/
structure RandomWalk where
  steps : List Vector2D

/-- The length of a 2D vector --/
noncomputable def vectorLength (v : Vector2D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

/-- The sum of a list of 2D vectors --/
def vectorSum (vs : List Vector2D) : Vector2D :=
  { x := (vs.map (·.x)).sum, y := (vs.map (·.y)).sum }

/-- Checks if the final position is within a given distance of the starting point --/
def isWithinDistance (walk : RandomWalk) (distance : ℝ) : Prop :=
  vectorLength (vectorSum walk.steps) ≤ distance

/-- The probability of the final position being within a given distance of the starting point --/
noncomputable def probabilityWithinDistance (walk : RandomWalk) (distance : ℝ) : ℝ :=
  sorry

/-- The frog's random walk --/
def frogWalk : RandomWalk :=
  { steps := [
    { x := 1.5, y := 0 },
    { x := 0, y := 1.5 },
    { x := -1.5, y := 0 },
    { x := 0, y := 2 }
  ] }

theorem frog_final_position_probability :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |probabilityWithinDistance frogWalk 2 - 1/5| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_final_position_probability_l437_43795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fg_negative_solution_set_l437_43789

-- Define the real number type
variable (x : ℝ)

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define f as an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Define g as an even function
axiom g_even : ∀ x, g (-x) = g x

-- Define the derivative condition
axiom derivative_condition : ∀ x, x < 0 → (deriv f) x * g x + f x * (deriv g) x > 0

-- Define g(-3) = 0
axiom g_at_neg_three : g (-3) = 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ (0 < x ∧ x < 3)}

-- Theorem statement
theorem fg_negative_solution_set :
  {x : ℝ | f x * g x < 0} = solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fg_negative_solution_set_l437_43789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l437_43707

theorem equation_solution_range (a : ℝ) :
  (∃ x : ℝ, (3 : ℝ)^(2*x) + a * (3 : ℝ)^x + 4 = 0) →
  (a ≤ -4 ∧ ∀ b < -4, ∃ x : ℝ, (3 : ℝ)^(2*x) + b * (3 : ℝ)^x + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l437_43707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_slab_cut_l437_43767

theorem potato_slab_cut (total_length : ℕ) (difference : ℕ) (longer_piece : ℕ) : 
  total_length = 600 → difference = 50 → 
  longer_piece = total_length / 2 + difference / 2 →
  longer_piece = 325 := by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

#check potato_slab_cut

end NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_slab_cut_l437_43767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kola_sugar_percentage_after_addition_l437_43776

/-- Represents the composition of a kola solution -/
structure KolaSolution where
  volume : ℝ
  water_percent : ℝ
  kola_percent : ℝ

/-- Calculates the percentage of sugar in the solution -/
noncomputable def sugar_percentage (solution : KolaSolution) : ℝ :=
  100 - solution.water_percent - solution.kola_percent

/-- Calculates the volume of sugar in the solution -/
noncomputable def sugar_volume (solution : KolaSolution) : ℝ :=
  (sugar_percentage solution / 100) * solution.volume

/-- Theorem: The percentage of sugar in the final kola solution is approximately 14.11% -/
theorem kola_sugar_percentage_after_addition 
  (initial_solution : KolaSolution)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ) :
  initial_solution.volume = 340 ∧
  initial_solution.water_percent = 80 ∧
  initial_solution.kola_percent = 6 ∧
  added_sugar = 3.2 ∧
  added_water = 10 ∧
  added_kola = 6.8 →
  let final_sugar_volume := sugar_volume initial_solution + added_sugar
  let final_volume := initial_solution.volume + added_sugar + added_water + added_kola
  let final_sugar_percentage := (final_sugar_volume / final_volume) * 100
  abs (final_sugar_percentage - 14.11) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kola_sugar_percentage_after_addition_l437_43776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trigonometric_inequality_right_triangle_trigonometric_equality_achievable_l437_43712

noncomputable section

open Real

theorem right_triangle_trigonometric_inequality (A B C : ℝ) :
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ A + B + C = π ∧ C = π / 2 →
  sin A + sin B + (sin A) ^ 2 ≤ sqrt 2 + 1 / 2 :=
by sorry

theorem right_triangle_trigonometric_equality_achievable :
  ∃ A B C : ℝ, 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ A + B + C = π ∧ C = π / 2 ∧
  sin A + sin B + (sin A) ^ 2 = sqrt 2 + 1 / 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trigonometric_inequality_right_triangle_trigonometric_equality_achievable_l437_43712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_19_10_l437_43734

theorem binomial_coefficient_19_10 (h1 : Nat.choose 17 7 = 19448) (h2 : Nat.choose 17 9 = 24310) :
  Nat.choose 19 10 = 92378 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_19_10_l437_43734
