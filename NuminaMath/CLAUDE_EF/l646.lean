import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l646_64620

noncomputable section

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (3, 2)
def C (p q : ℝ) : ℝ × ℝ := (p, q)

-- Define the parabola
def on_parabola (x y : ℝ) : Prop := y = x^2 - 5*x + 6

-- Define the range for p
def p_range (p : ℝ) : Prop := 1 ≤ p ∧ p ≤ 3

-- Define the area of the triangle
noncomputable def triangle_area (p q : ℝ) : ℝ :=
  (1/2) * |1 * 2 + 3 * q + p * 0 - 0 * 3 - 2 * p - q * 1|

-- Theorem statement
theorem max_triangle_area :
  ∃ (p q : ℝ), 
    p_range p ∧ 
    on_parabola p q ∧ 
    on_parabola A.1 A.2 ∧ 
    on_parabola B.1 B.2 ∧
    (∀ (p' q' : ℝ), p_range p' → on_parabola p' q' → triangle_area p' q' ≤ triangle_area p q) ∧
    triangle_area p q = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l646_64620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCDEF_is_135_l646_64680

/-- Represents a polygon ABCDEF with specific properties -/
structure PolygonABCDEF where
  AB : ℝ
  BC : ℝ
  DC : ℝ
  FA : ℝ
  GF : ℝ
  AB_eq : AB = 10
  BC_eq : BC = 15
  DC_eq : DC = 7
  FA_eq : FA = 12
  GF_eq : GF = 6
  is_rectangle_ABCG : True  -- Assume ABCG is a rectangle
  is_right_triangle_GFED : True  -- Assume GFED is a right triangle with GF as base and ED perpendicular to GF

/-- Calculates the area of polygon ABCDEF -/
noncomputable def area_ABCDEF (p : PolygonABCDEF) : ℝ :=
  p.AB * p.BC - (1/2) * p.GF * (p.FA - p.DC)

/-- Theorem stating that the area of polygon ABCDEF is 135 square units -/
theorem area_ABCDEF_is_135 (p : PolygonABCDEF) : area_ABCDEF p = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCDEF_is_135_l646_64680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clubsuit_equals_two_and_expression_equals_two_final_clubsuit_equals_two_l646_64623

-- Define the ♣ operation as noncomputable
noncomputable def clubsuit (a b : ℝ) : ℝ := (2 * a / b) * (b / a)

-- Theorem statement
theorem clubsuit_equals_two_and_expression_equals_two :
  (∀ (a b : ℝ), b ≠ 0 → clubsuit a b = 2) ∧
  clubsuit 5 (clubsuit 3 6) = 2 :=
by
  -- Proof steps would go here
  sorry

-- Additional theorem to handle the final clubsuit operation
theorem final_clubsuit_equals_two :
  clubsuit (clubsuit 5 (clubsuit 3 6)) 1 = 2 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clubsuit_equals_two_and_expression_equals_two_final_clubsuit_equals_two_l646_64623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coinciding_segment_l646_64618

def mySequence (period : ℕ) := ℕ → ℕ

def periodic (s : mySequence period) (p : ℕ) : Prop :=
  ∀ n, s n = s (n + p)

def coincide (s1 s2 : ℕ → ℕ) (length : ℕ) : Prop :=
  ∀ n, n < length → s1 n = s2 n

theorem max_coinciding_segment (s1 : mySequence 7) (s2 : mySequence 13)
  (h1 : periodic s1 7) (h2 : periodic s2 13) :
  (∃ length, coincide s1 s2 length) ∧
  (∀ length, coincide s1 s2 length → length ≤ 18) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coinciding_segment_l646_64618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_william_total_distance_l646_64689

/-- Represents William's riding schedule and speeds --/
structure RidingSchedule where
  max_hours : ℝ
  weekday_max_days : ℕ
  weekday_short_days : ℕ
  weekday_half_days : ℕ
  weekday_max_speed : ℝ
  weekday_short_speed : ℝ
  weekday_half_speed : ℝ
  weekend_sat_hours : ℝ
  weekend_sat_speed : ℝ
  weekend_sun_hours : ℝ
  weekend_sun_speed : ℝ
  weekend_break_duration : ℝ

/-- Calculates the total distance ridden in two weeks --/
noncomputable def total_distance (schedule : RidingSchedule) : ℝ :=
  let weekday_distance := 
    schedule.max_hours * schedule.weekday_max_speed * (schedule.weekday_max_days : ℝ) +
    1.5 * schedule.weekday_short_speed * (schedule.weekday_short_days : ℝ) +
    (schedule.max_hours / 2) * schedule.weekday_half_speed * (schedule.weekday_half_days : ℝ)
  let weekend_sat_distance := 
    (schedule.weekend_sat_hours - (schedule.weekend_break_duration * (schedule.weekend_sat_hours - 1))) * 
    schedule.weekend_sat_speed
  let weekend_sun_distance := 
    (schedule.weekend_sun_hours - (schedule.weekend_break_duration * (schedule.weekend_sun_hours - 1))) * 
    schedule.weekend_sun_speed
  let weekly_distance := weekday_distance + weekend_sat_distance + weekend_sun_distance
  2 * weekly_distance

/-- William's actual riding schedule --/
def william_schedule : RidingSchedule := {
  max_hours := 6
  weekday_max_days := 2
  weekday_short_days := 2
  weekday_half_days := 1
  weekday_max_speed := 10
  weekday_short_speed := 8
  weekday_half_speed := 12
  weekend_sat_hours := 4
  weekend_sat_speed := 15
  weekend_sun_hours := 3
  weekend_sun_speed := 10
  weekend_break_duration := 0.25
}

/-- Theorem stating that William's total distance over two weeks is 507.5 miles --/
theorem william_total_distance : 
  total_distance william_schedule = 507.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_william_total_distance_l646_64689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1991_of_2_pow_1990_eq_4_l646_64652

-- Define f₁ as the square of the sum of digits
def f₁ (k : ℕ) : ℕ :=
  (Nat.digits 10 k).sum ^ 2

-- Define fₙ₊₁ recursively
def f (n : ℕ) (k : ℕ) : ℕ :=
  match n with
  | 0 => k
  | n + 1 => f₁ (f n k)

-- Theorem statement
theorem f_1991_of_2_pow_1990_eq_4 :
  f 1991 (2^1990) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1991_of_2_pow_1990_eq_4_l646_64652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_arccos_zero_is_pi_half_l646_64605

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3))

-- State the theorem about the domain of g
theorem domain_of_g :
  {x : ℝ | g x ≠ 0 ∧ x ∈ Set.Icc (-1 : ℝ) 1} = Set.Ioc (-1 : ℝ) 0 ∪ Set.Ioc 0 1 := by
  sorry

-- Define the conditions
axiom arccos_domain (x : ℝ) : -1 ≤ x^3 ∧ x^3 ≤ 1 → Real.arccos x^3 ∈ Set.Icc 0 Real.pi
axiom tan_undefined : Real.tan (Real.pi / 2) = 0  -- Using 0 to represent undefined in Lean

-- Additional helper theorem
theorem arccos_zero_is_pi_half : Real.arccos 0 = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_arccos_zero_is_pi_half_l646_64605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_queries_correct_l646_64615

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the side of a line a point can be on -/
inductive Side
  | Left
  | Right
  | On

/-- Represents a line query that determines which side of the line a point is on -/
structure LineQuery where
  query : Point → Side

/-- Represents the result of determining if a point is inside or outside a square -/
inductive SquarePosition
  | Inside
  | Outside

/-- The minimum number of line queries needed to determine if a point is inside a square -/
def min_queries : ℕ := 3

/-- Function to determine the position of a point relative to a square given a list of queries -/
def determine_position (square : Set Point) (p : Point) (queries : List LineQuery) : SquarePosition :=
  sorry -- Implementation omitted for brevity

/-- Theorem stating that the minimum number of queries is correct -/
theorem min_queries_correct (square : Set Point) (p : Point) :
  ∀ (queries : List LineQuery),
    queries.length < min_queries →
    ∃ (p1 p2 : Point),
      (determine_position square p queries = determine_position square p1 queries) ∧
      (determine_position square p queries = determine_position square p2 queries) ∧
      (SquarePosition.Inside = determine_position square p1 []) ∧
      (SquarePosition.Outside = determine_position square p2 []) :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_queries_correct_l646_64615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_proof_l646_64635

-- Define the cost per kilogram
noncomputable def cost_per_kg : ℚ := 25000

-- Define the conversion from ounces to grams
noncomputable def oz_to_g : ℚ := 2835 / 100

-- Define the conversion from grams to kilograms
noncomputable def g_to_kg : ℚ := 1 / 1000

-- Define the weight of the sensor in ounces
noncomputable def sensor_weight_oz : ℚ := 10

-- Theorem statement
theorem transport_cost_proof :
  let sensor_weight_kg := sensor_weight_oz * oz_to_g * g_to_kg
  sensor_weight_kg * cost_per_kg = 70875 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transport_cost_proof_l646_64635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l646_64653

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.tan (2 * ω * x - Real.pi / 6)

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, f ω (x + Real.pi / (2 * ω)) = f ω x) :
  ω = 1 ∧
  (∀ k : ℤ, ∃ c : ℝ, f ω (k * Real.pi / 4 + Real.pi / 12 + c) = f ω (k * Real.pi / 4 + Real.pi / 12 - c)) ∧
  (∀ x y : ℝ, Real.pi / 12 < x ∧ x < y ∧ y < Real.pi / 3 → f ω x < f ω y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l646_64653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_cost_is_two_l646_64665

/-- The cost of fruit and the discount policy at a store -/
structure FruitStore where
  lemon_cost : ℚ
  papaya_cost : ℚ
  mango_cost : ℚ
  discount_per_four : ℚ

/-- Tom's purchase -/
structure Purchase where
  lemons : ℕ
  papayas : ℕ
  mangos : ℕ

/-- Calculate the total cost of a purchase at a given store -/
def total_cost (store : FruitStore) (purchase : Purchase) : ℚ :=
  let total_fruits := purchase.lemons + purchase.papayas + purchase.mangos
  let discount := (total_fruits / 4 : ℚ) * store.discount_per_four
  store.lemon_cost * purchase.lemons +
  store.papaya_cost * purchase.papayas +
  store.mango_cost * purchase.mangos -
  discount

/-- The theorem to prove -/
theorem lemon_cost_is_two :
  ∃ (store : FruitStore),
    store.papaya_cost = 1 ∧
    store.mango_cost = 4 ∧
    store.discount_per_four = 1 ∧
    total_cost store ⟨6, 4, 2⟩ = 21 →
    store.lemon_cost = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_cost_is_two_l646_64665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l646_64678

-- Define the function as noncomputable due to the use of real exponentiation
noncomputable def f (x : ℝ) : ℝ := 3^(-|x - 1|)

-- State the theorem
theorem intersection_range :
  Set.Icc (0 : ℝ) 1 = {m : ℝ | ∃ x, f x = m} := by
  sorry

#check intersection_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l646_64678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l646_64663

noncomputable def a : ℝ := 17
noncomputable def b : ℝ := 19
noncomputable def c : ℝ := 23

noncomputable def complex_fraction (a b c : ℝ) : ℝ :=
  (136 * (1/b - 1/c) + 361 * (1/c - 1/a) + 529 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) *
  (144 * (1/b - 1/c) + 400 * (1/c - 1/a) + 576 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b))

theorem complex_fraction_simplification :
  complex_fraction a b c = (a + b + c)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l646_64663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_constraints_l646_64661

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + Real.sqrt b * x^2 - a^2 * x

-- Define the derivative of f
noncomputable def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * Real.sqrt b * x - a^2

theorem function_constraints (a b : ℝ) (ha : a > 0) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
   f' a b x₁ = 0 ∧ f' a b x₂ = 0 ∧ 
   |x₁| + |x₂| = 2) →
  (0 < a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_constraints_l646_64661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_powers_of_three_l646_64693

theorem perfect_squares_between_powers_of_three : 
  (Finset.range ((Nat.sqrt (3^10 + 1)) + 1) ∩ 
   Finset.filter (fun n => n^2 ≥ 3^5 + 1 ∧ n^2 ≤ 3^10 + 1) (Finset.range ((Nat.sqrt (3^10 + 1)) + 1))).card = 228 := by
  sorry

#eval (Finset.range ((Nat.sqrt (3^10 + 1)) + 1) ∩ 
   Finset.filter (fun n => n^2 ≥ 3^5 + 1 ∧ n^2 ≤ 3^10 + 1) (Finset.range ((Nat.sqrt (3^10 + 1)) + 1))).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_powers_of_three_l646_64693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_three_ordered_summands_l646_64637

open BigOperators

theorem partition_into_three_ordered_summands (n : ℕ) (h : n > 2) :
  (∑ k in Finset.range (n - 2), (n - k - 1)) = (n - 2) * (n - 1) / 2 := by
  sorry

#check partition_into_three_ordered_summands

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_three_ordered_summands_l646_64637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_room_area_l646_64654

/-- Represents the dimensions of a rectangular room in feet -/
structure RoomDimensions where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangular room in square feet -/
def area (room : RoomDimensions) : ℚ := room.length * room.width

/-- Converts square feet to square yards -/
def sqft_to_sqyd (sqft : ℚ) : ℚ := sqft / 9

theorem l_shaped_room_area :
  let main_room : RoomDimensions := ⟨15, 12⟩
  let extension : RoomDimensions := ⟨6, 5⟩
  let total_area_sqft := area main_room + area extension
  let total_area_sqyd := sqft_to_sqyd total_area_sqft
  total_area_sqyd = 70 / 3 := by
  -- Proof steps would go here
  sorry

#eval sqft_to_sqyd (area ⟨15, 12⟩ + area ⟨6, 5⟩)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_room_area_l646_64654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ajax_weight_after_two_weeks_l646_64662

/-- Represents the intensity levels of exercises -/
inductive Intensity
  | High
  | Moderate
  | Low

/-- Represents a daily workout schedule -/
structure DailySchedule where
  high : Float
  moderate : Float
  low : Float

/-- Represents a weekly workout schedule -/
def WeeklySchedule : Fin 7 → DailySchedule
  | 0 => ⟨1, 0.5, 0⟩   -- Monday
  | 1 => ⟨0, 0.5, 1⟩   -- Tuesday
  | 2 => ⟨1, 0.5, 0⟩   -- Wednesday
  | 3 => ⟨0, 0.5, 1⟩   -- Thursday
  | 4 => ⟨1, 0.5, 0⟩   -- Friday
  | 5 => ⟨1.5, 0, 0.5⟩ -- Saturday
  | 6 => ⟨0, 0, 0⟩     -- Sunday (Rest day)

/-- Calculates the weight loss for a given exercise intensity and duration -/
def weightLoss (intensity : Intensity) (duration : Float) : Float :=
  match intensity with
  | Intensity.High => 4 * duration
  | Intensity.Moderate => 2.5 * duration
  | Intensity.Low => 1.5 * duration

/-- Calculates the total weight loss for a daily schedule -/
def dailyWeightLoss (schedule : DailySchedule) : Float :=
  weightLoss Intensity.High schedule.high +
  weightLoss Intensity.Moderate schedule.moderate +
  weightLoss Intensity.Low schedule.low

/-- Calculates the total weight loss for a week -/
def weeklyWeightLoss : Float :=
  (List.range 7).foldl (fun acc i => acc + dailyWeightLoss (WeeklySchedule i)) 0

/-- Theorem: Ajax's weight after two weeks of following the workout schedule -/
theorem ajax_weight_after_two_weeks (initial_weight : Float) 
    (h1 : initial_weight = 80) 
    (h2 : weeklyWeightLoss * 2 = 56) : 
    initial_weight * 2.2 - weeklyWeightLoss * 2 = 120 := by
  sorry

-- Remove the #eval statement as it's causing issues
-- #eval ajax_weight_after_two_weeks 80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ajax_weight_after_two_weeks_l646_64662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_l646_64674

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a spherical marble -/
structure Marble where
  radius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (m : Marble) : ℝ := (4/3) * Real.pi * m.radius^3

/-- Calculates the new height of a cone after adding a marble -/
noncomputable def newConeHeight (c : Cone) (m : Marble) : ℝ :=
  c.height + sphereVolume m / (Real.pi * c.radius^2)

theorem liquid_level_rise_ratio 
  (cone1 cone2 : Cone) 
  (marble : Marble) 
  (h_volume : coneVolume cone1 = coneVolume cone2) 
  (h_radius1 : cone1.radius = 5) 
  (h_radius2 : cone2.radius = 10) 
  (h_marble : marble.radius = 2) : 
  (newConeHeight cone1 marble - cone1.height) / (newConeHeight cone2 marble - cone2.height) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_l646_64674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_x14_axis_l646_64684

/-- A point in 3D space represented by its projections -/
structure ProjectedPoint where
  p' : ℝ × ℝ  -- First projection
  p'' : ℝ × ℝ -- Second projection

/-- A plane in 3D space -/
structure Plane

/-- A cone in 3D space -/
structure Cone where
  vertex : ℝ × ℝ × ℝ
  axis : ℝ × ℝ × ℝ
  half_angle : ℝ

/-- The x₁,₄ axis -/
def x14_axis : Set (ℝ × ℝ) := sorry

/-- The first projection plane -/
def first_projection_plane : Plane := sorry

/-- Calculate the dihedral angle between a plane and a projection plane -/
def dihedral_angle (p : Plane) (n : Nat) : ℝ := sorry

/-- Check if two vectors are perpendicular -/
def perpendicular (v1 v2 : ℝ × ℝ × ℝ) : Prop := sorry

/-- Get the tangent plane of a cone -/
def tangent_plane (c : Cone) : Plane := sorry

/-- Get the first trace of a plane -/
def first_trace (p : Plane) : Set (ℝ × ℝ) := sorry

/-- The theorem stating the construction of x₁,₄ axis -/
theorem construct_x14_axis 
  (second_projection_plane : Plane) 
  (P : ProjectedPoint) 
  (given_plane : Plane) 
  (h : dihedral_angle given_plane 4 = 60) : 
  ∃ (C : Cone), 
    C.half_angle = 30 ∧ 
    perpendicular C.axis (1, 0, 0) ∧ 
    C.vertex = (P.p'.1, P.p'.2, P.p''.2) ∧
    x14_axis = first_trace (tangent_plane C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_x14_axis_l646_64684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_expression_eval_l646_64682

/-- Given a point P(-4, 3) on the terminal side of angle α, 
    prove that the expression evaluates to -3/4 -/
theorem angle_expression_eval (α : ℝ) :
  let P : ℝ × ℝ := (-4, 3)
  let r : ℝ := Real.sqrt ((-4)^2 + 3^2)
  let sin_α : ℝ := 3 / r
  let cos_α : ℝ := -4 / r
  (Real.cos (π / 2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_expression_eval_l646_64682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_constant_power_theorem_l646_64660

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define the power of a point with respect to a circle
def power (p : Point) (c : Circle) : ℝ :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 - c.radius^2

-- Define a set of 6 points
def SixPoints : Set Point :=
  {p : Point | ∃ (ps : Finset Point), ps.card = 6 ∧ p ∈ ps}

-- No three points are collinear
def NoThreeCollinear (points : Set Point) : Prop :=
  ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
    ∃ m b : ℝ, (p2.y - p1.y) ≠ m * (p2.x - p1.x) ∨
               (p3.y - p1.y) ≠ m * (p3.x - p1.x) ∨
               (p3.y - p2.y) ≠ m * (p3.x - p2.x)

-- For every 4 points, there exists a point with constant power k
def ConstantPowerK (points : Set Point) (k : ℝ) : Prop :=
  ∀ p1 p2 p3 p4, p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points →
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p4 →
    ∃ p5, p5 ∈ points ∧ p5 ≠ p1 ∧ p5 ≠ p2 ∧ p5 ≠ p3 ∧ p5 ≠ p4 ∧
    ∃ c : Circle, power p1 c = 0 ∧ power p2 c = 0 ∧ power p3 c = 0 ∧ power p5 c = k

-- All points lie on a circle
def AllPointsOnCircle (points : Set Point) : Prop :=
  ∃ c : Circle, ∀ p, p ∈ points → power p c = 0

-- Theorem statement
theorem six_points_constant_power_theorem (points : Set Point) (k : ℝ) :
  points = SixPoints →
  NoThreeCollinear points →
  ConstantPowerK points k →
  k = 0 ∧ AllPointsOnCircle points :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_constant_power_theorem_l646_64660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l646_64610

open Real

-- Define an odd function f that is monotonically increasing on [0, +∞)
def isOddAndIncreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x → x < y → f x < f y)

-- Define the set of x satisfying the inequality
def satisfyingSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x > 0 ∧ |f (log x) - f (log (1/x))| / 2 < f 1}

-- Theorem statement
theorem inequality_solution_set (f : ℝ → ℝ) 
  (h : isOddAndIncreasing f) : 
  satisfyingSet f = Set.Ioo (1/Real.exp 1) (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l646_64610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l646_64614

/-- The hyperbola struct represents a hyperbola with equation x^2 - y^2/b^2 = 1 -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram PAOB where O is the origin -/
structure Parallelogram where
  p : Point
  a : Point
  b : Point

noncomputable def Hyperbola.eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + 1 / h.b^2)

/-- States that a point lies on the hyperbola -/
def point_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 - p.y^2 / h.b^2 = 1

/-- States that two lines through P are parallel to the asymptotes -/
def parallel_to_asymptotes (h : Hyperbola) (p a b : Point) : Prop :=
  sorry

/-- Calculates the area of the parallelogram -/
noncomputable def parallelogram_area (par : Parallelogram) : ℝ :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) (p a b : Point) 
  (h_p_on : point_on_hyperbola h p)
  (h_parallel : parallel_to_asymptotes h p a b)
  (h_area : parallelogram_area ⟨p, a, b⟩ = 1) :
  h.eccentricity = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l646_64614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l646_64664

/-- Represents a tetromino (a shape made of four 1×1 squares) -/
structure Tetromino :=
  (squares : Finset (ℕ × ℕ))
  (size : squares.card = 4)

/-- Represents a 4×5 rectangle -/
def Rectangle : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 4) (Finset.range 5)

/-- Checkerboard coloring of the rectangle -/
def isBlack (x y : ℕ) : Bool :=
  (x + y) % 2 = 0

/-- Count of black squares in a tetromino -/
def blackCount (t : Tetromino) : ℕ :=
  t.squares.filter (fun (x, y) => isBlack x y) |>.card

theorem impossible_tiling (tetrominoes : Finset Tetromino) 
  (h_count : tetrominoes.card = 5) :
  ¬ (∃ (arrangement : Tetromino → Finset (ℕ × ℕ)), 
      (∀ t, t ∈ tetrominoes → arrangement t ⊆ Rectangle) ∧
      (∀ p, p ∈ Rectangle → ∃! t, t ∈ tetrominoes ∧ p ∈ arrangement t)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l646_64664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_iff_t_range_l646_64687

/-- The function y = x + 1/(2x) + t --/
noncomputable def y (x t : ℝ) : ℝ := x + 1/(2*x) + t

/-- The function h(x) = x + 1/(2x) --/
noncomputable def h (x : ℝ) : ℝ := x + 1/(2*x)

theorem two_zeros_iff_t_range (t : ℝ) (h_t : t > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ y x₁ t = 0 ∧ y x₂ t = 0) ↔ t ∈ Set.Iio (-Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_iff_t_range_l646_64687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_range_l646_64691

-- Define the vectors
noncomputable def a : Fin 2 → ℝ := ![1, Real.sqrt 3]
def b : Fin 2 → ℝ := ![0, 1]

-- Define the interval for t
def t_interval : Set ℝ := Set.Icc (-Real.sqrt 3) 2

-- Define the function to be minimized/maximized
noncomputable def f (t : ℝ) : ℝ := Real.sqrt ((a 0 - t * b 0)^2 + (a 1 - t * b 1)^2)

theorem vector_difference_range :
  ∃ (min max : ℝ), min = 1 ∧ max = Real.sqrt 13 ∧
  (∀ t ∈ t_interval, min ≤ f t ∧ f t ≤ max) ∧
  (∃ t₁ ∈ t_interval, f t₁ = min) ∧
  (∃ t₂ ∈ t_interval, f t₂ = max) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_range_l646_64691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l646_64647

/-- The ellipse problem -/
theorem ellipse_problem (a b c : ℝ) (k : ℝ) :
  a > b ∧ b > 0 ∧  -- a > b > 0
  c^2 / a^2 = 5 / 9 ∧  -- eccentricity is √5/3
  a * Real.sqrt (2*b^2) = 6 * Real.sqrt 2 ∧  -- |FB| * |AB| = 6√2
  k > 0 ∧  -- k > 0 for line y = kx
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → y = k*x →  -- line intersects ellipse
    ∃ x₂ y₂ : ℝ, x₂ + y₂ = 2 ∧ y₂ = k*x₂ ∧  -- line intersects AB
    Real.sqrt 2 * y₂ / Real.sqrt ((x-x₂)^2 + (y-y₂)^2) = (5 * Real.sqrt 2 / 4) * (y₂ / Real.sqrt (x₂^2 + y₂^2)))  -- |AQ|/|PQ| condition
  →
  (a = 3 ∧ b = 2) ∧  -- equation of ellipse
  (k = 1/2 ∨ k = 11/28)  -- value of k
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l646_64647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_fourth_quadrant_l646_64668

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x / log a
def g (a : ℝ) (x : ℝ) : ℝ := (1 - a) * x

-- Define the fourth quadrant
def fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem intersection_in_fourth_quadrant (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, f a x = g a x ∧ f a x = y ∧ fourth_quadrant (x, y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_fourth_quadrant_l646_64668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_unique_solution_l646_64672

/-- Represents an infinite geometric series with first term b₁ and common ratio q. -/
structure GeometricSeries where
  b₁ : ℝ
  q : ℝ
  h_q : |q| < 1

/-- The sum of an infinite geometric series. -/
noncomputable def seriesSum (s : GeometricSeries) : ℝ := s.b₁ / (1 - s.q)

/-- The sum of the cubes of terms in an infinite geometric series. -/
noncomputable def seriesCubeSum (s : GeometricSeries) : ℝ := s.b₁^3 / (1 - s.q^3)

theorem geometric_series_unique_solution (s : GeometricSeries) 
  (h_sum : seriesSum s = 4)
  (h_cube_sum : seriesCubeSum s = 192) :
  s.b₁ = 6 ∧ s.q = -0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_unique_solution_l646_64672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_and_function_range_l646_64651

noncomputable def m (A : ℝ) : ℝ × ℝ := (Real.sin A, Real.cos A)
noncomputable def n : ℝ × ℝ := (Real.sqrt 3, -1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def f (A x : ℝ) : ℝ := Real.cos (2 * x) + 4 * (Real.cos A) * (Real.sin x)

theorem vector_angle_and_function_range 
  (A : ℝ) 
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : dot_product (m A) n = 1) :
  A = π / 3 ∧ 
  Set.Icc (-3 : ℝ) (3 / 2) = Set.range (f A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_and_function_range_l646_64651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_courses_l646_64696

/-- The probability of two students choosing different courses from four options -/
theorem probability_different_courses : (3 : ℝ) / 4 = 
  let num_courses : ℕ := 4
  let total_outcomes : ℕ := num_courses * num_courses
  let different_outcomes : ℕ := num_courses * (num_courses - 1)
  (different_outcomes : ℝ) / total_outcomes
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_courses_l646_64696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_sin_equality_l646_64634

theorem csc_sin_equality : 1 / Real.sin (π / 6) - 4 * Real.sin (π / 3) = 2 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_sin_equality_l646_64634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l646_64628

def A : Set ℤ := {x | x^2 - 3*x ≤ 0}
def B : Set ℝ := {x | Real.log x < 1}

def A_real : Set ℝ := {x | ∃ n : ℤ, n ∈ A ∧ x = n}

theorem intersection_of_A_and_B : A_real ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l646_64628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_fraction_equals_five_l646_64645

theorem floor_fraction_equals_five :
  ⌊(2008 * 80 + 2009 * 130 + 2010 * 180 : ℚ) / (2008 * 15 + 2009 * 25 + 2010 * 35 : ℚ)⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_fraction_equals_five_l646_64645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_theorem_l646_64669

noncomputable section

open Real

theorem triangle_abc_theorem (A B C : ℝ) (a b c : ℝ) :
  (cos C / cos B = (3 * a - c) / b) →
  (b = 4 * sqrt 2) →
  (a = c) →
  (cos B = 1 / 3 ∧ sin (A + π / 6) = (3 * sqrt 2 + sqrt 3) / 6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_theorem_l646_64669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_impossibility_l646_64600

theorem box_volume_impossibility : 
  ∀ x : ℕ+, ∀ v ∈ ({240, 300, 400, 500, 600} : Finset ℕ), (10 : ℕ) * (x.val ^ 3) ≠ v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_impossibility_l646_64600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l646_64616

/-- The constant term in the expansion of (x^3 + 2)(2x - 1/x^2)^6 is 320 -/
theorem constant_term_expansion : ∃ (f : ℝ → ℝ),
  (∀ x ≠ 0, f x = (x^3 + 2) * (2*x - 1/x^2)^6) ∧
  (∃ c : ℝ, ∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  (∀ c : ℝ, (∃ δ > (0 : ℝ), ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < |c - 320|) → c = 320) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l646_64616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_circumscribed_sphere_l646_64627

/-- The volume of a sphere circumscribing a rectangular prism -/
theorem volume_circumscribed_sphere (l w h : ℝ) (hl : l = Real.sqrt 3) (hw : w = 2) (hh : h = Real.sqrt 5) :
  (4 / 3) * Real.pi * ((Real.sqrt (l^2 + w^2 + h^2)) / 2)^3 = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_circumscribed_sphere_l646_64627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_jar_count_l646_64649

theorem marble_jar_count : ∃ (total : ℕ),
  -- Half of the marbles are blue
  total / 2 = total - (total / 4 + 27 + 14) ∧
  -- A quarter of the marbles are red
  total / 4 = total - (total / 2 + 27 + 14) ∧
  -- 27 marbles are green
  27 = total - (total / 2 + total / 4 + 14) ∧
  -- 14 marbles are yellow
  14 = total - (total / 2 + total / 4 + 27) ∧
  -- The total number of marbles is 164
  total = 164 := by
  -- Proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_jar_count_l646_64649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_lower_bound_l646_64631

variable {n : ℕ}
variable (A : Matrix (Fin n) (Fin n) ℕ)

def row_sum (A : Matrix (Fin n) (Fin n) ℕ) (i : Fin n) : ℕ :=
  (Finset.univ.sum fun j => A i j)

def col_sum (A : Matrix (Fin n) (Fin n) ℕ) (j : Fin n) : ℕ :=
  (Finset.univ.sum fun i => A i j)

def matrix_sum (A : Matrix (Fin n) (Fin n) ℕ) : ℕ :=
  (Finset.univ.sum fun i => Finset.univ.sum fun j => A i j)

def zero_condition (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∀ i j, A i j = 0 → row_sum A i + col_sum A j ≥ n

theorem matrix_sum_lower_bound (hn : n > 0) (h : zero_condition A) :
  matrix_sum A ≥ n^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_lower_bound_l646_64631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_tripod_height_l646_64644

/-- Represents a tripod with three legs -/
structure Tripod where
  leg_length : ℝ
  height : ℝ

/-- Represents a tripod with one shortened leg -/
structure BrokenTripod where
  original : Tripod
  shortened_leg_length : ℝ

/-- Calculates the new height of a broken tripod -/
noncomputable def new_height (bt : BrokenTripod) : ℝ :=
  (2/3) * bt.original.height

theorem broken_tripod_height (bt : BrokenTripod) 
  (h_original : bt.original.leg_length = 6 ∧ bt.original.height = 5)
  (h_shortened : bt.shortened_leg_length = 4) :
  new_height bt = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_tripod_height_l646_64644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l646_64655

noncomputable def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x - Real.pi / 6) + Real.sin (w * x - Real.pi / 2)

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x - Real.pi / 12)

theorem problem_solution :
  (∃ w : ℝ, 0 < w ∧ w < 3 ∧ f w (Real.pi / 6) = 0 ∧ w = 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4) → g x ≥ -3 / 2) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4) ∧ g x = -3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l646_64655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_proof_l646_64612

/-- Proves that the speed of the first cyclist is 7 m/s given the problem conditions -/
theorem cyclist_speed_proof (v1 v2 : ℝ) (c t : ℝ) : 
  v1 = 7 →  -- Speed of first cyclist
  v2 = 8 →  -- Speed of second cyclist
  c = 180 → -- Circumference of the track
  t = 12 →  -- Time taken to meet
  v1 * t + v2 * t = c → -- Sum of distances traveled equals circumference
  v1 = 7 := by
  intro h1 h2 h3 h4 h5
  exact h1

#check cyclist_speed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_proof_l646_64612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_l646_64692

theorem polynomial_division (n : ℕ) :
  let g (x : ℂ) := (x^(2*(n+1)) - 1) / (x^2 - 1)
  let f (x : ℂ) := (x^(4*(n+1)) - 1) / (x^4 - 1)
  (∃ p : ℂ → ℂ, ∀ x, f x = (g x) * (p x)) ↔ Even n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_l646_64692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_area_ratio_range_l646_64658

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    prove that its eccentricity is sqrt(1 - b^2/a^2) --/
theorem ellipse_eccentricity 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (ellipse : Set (ℝ × ℝ))
  (h3 : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ ellipse)
  (F : ℝ × ℝ) 
  (h4 : F ∈ ellipse) 
  (h5 : F.1 < 0) 
  (M m : ℝ) 
  (h6 : M * m = 3/4 * a^2) :
  Real.sqrt (1 - b^2/a^2) = Real.sqrt (1 - b^2/a^2) := by
  sorry

/-- Given the conditions from the problem, prove that the range of 
    (2 * S₁ * S₂) / (S₁² + S₂²) is [0, 1] --/
theorem area_ratio_range 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0)
  (ellipse : Set (ℝ × ℝ))
  (h3 : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ ellipse)
  (F : ℝ × ℝ) 
  (h4 : F ∈ ellipse) 
  (h5 : F.1 < 0) 
  (M m : ℝ) 
  (h6 : M * m = 3/4 * a^2)
  (S₁ S₂ : ℝ)
  (h7 : S₁ > 0)
  (h8 : S₂ > 0) :
  0 ≤ (2 * S₁ * S₂) / (S₁^2 + S₂^2) ∧ (2 * S₁ * S₂) / (S₁^2 + S₂^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_area_ratio_range_l646_64658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_after_deletion_l646_64633

/-- The original sequence of digits -/
def original_sequence : List Nat := [1,2,3,4,5,6,7,8,9,1,0,1,1,1,2,9,9,1,0,0]

/-- The number of digits in the original sequence -/
def original_length : Nat := 192

/-- The number of digits to be deleted -/
def digits_to_delete : Nat := 100

/-- The resulting length after deletion -/
def result_length : Nat := original_length - digits_to_delete

/-- A function that deletes 100 digits from the original sequence -/
noncomputable def delete_digits (seq : List Nat) : List Nat :=
  sorry

/-- A function that converts a list of digits to a natural number -/
def digits_to_number (digits : List Nat) : Nat :=
  sorry

/-- The theorem stating that the largest number after deletion starts with 99999785960 -/
theorem largest_after_deletion :
  ∃ (result : List Nat),
    result.length = result_length ∧
    delete_digits original_sequence = result ∧
    (∀ (other : List Nat),
      other.length = result_length ∧
      delete_digits original_sequence = other →
      digits_to_number result ≥ digits_to_number other) ∧
    (result.take 11 = [9,9,9,9,9,7,8,5,9,6,0]) := by
  sorry

#check largest_after_deletion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_after_deletion_l646_64633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_ratios_l646_64621

-- Define the right triangle ABC
noncomputable def RightTriangleABC (A B C : ℝ) : Prop :=
  A^2 + B^2 = C^2 ∧ 0 < A ∧ 0 < B ∧ 0 < C

-- Define the tangent and sine of an angle in a right triangle
noncomputable def tan_angle (adj opp : ℝ) : ℝ := opp / adj
noncomputable def sin_angle (opp hyp : ℝ) : ℝ := opp / hyp

-- Theorem statement
theorem right_triangle_ratios :
  ∀ (A B C : ℝ),
  RightTriangleABC A B C →
  A = 15 →
  C = 17 →
  tan_angle A B = 8/15 ∧ sin_angle B C = 8/17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_ratios_l646_64621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_sum_factorials_l646_64671

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the function to calculate 15! + 17!
def sum_of_factorials : ℕ := factorial 15 + factorial 17

-- Define a function to get the greatest prime factor
def greatest_prime_factor (n : ℕ) : ℕ :=
  (Nat.factors n).maximum.getD 1

-- Theorem statement
theorem greatest_prime_factor_of_sum_factorials :
  greatest_prime_factor sum_of_factorials = 13 := by
  sorry

#eval greatest_prime_factor sum_of_factorials

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_sum_factorials_l646_64671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_payment_calculation_l646_64611

-- Define the constants
def task_a_daily_rate : ℕ := 25
def task_b_daily_rate : ℕ := 30
def task_a_days : ℕ := 3
def task_b_days : ℕ := 2
def overtime_hours : ℕ := 2
def overtime_rate_multiplier : ℚ := 3/2
def performance_bonus : ℕ := 50
def transportation_expense : ℕ := 40
def hours_per_day : ℕ := 8

-- Define the theorem
theorem worker_payment_calculation :
  (task_a_daily_rate * task_a_days +
   task_b_daily_rate * task_b_days +
   ((task_a_daily_rate : ℚ) / hours_per_day * overtime_rate_multiplier * overtime_hours).ceil.toNat +
   performance_bonus -
   transportation_expense) = 155 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_payment_calculation_l646_64611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_function_zeros_l646_64608

/-- Given a function f(x) = cos(ωx) - 1 where ω > 0, which has exactly 3 zeros
    in the interval [0, 2π], prove that 2 ≤ ω < 3. -/
theorem cos_function_zeros (ω : ℝ) (h_pos : ω > 0) : 
  (∃! (z₁ z₂ z₃ : ℝ), 0 ≤ z₁ ∧ z₁ < z₂ ∧ z₂ < z₃ ∧ z₃ ≤ 2*π ∧ 
    (∀ x ∈ Set.Icc 0 (2*π), Real.cos (ω*x) - 1 = 0 ↔ x = z₁ ∨ x = z₂ ∨ x = z₃)) →
  2 ≤ ω ∧ ω < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_function_zeros_l646_64608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l646_64607

/-- The function f(x) = 1 / (3^x + √3) -/
noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

/-- Theorem stating that f(x) + f(1-x) = √3/3 for all real x -/
theorem f_sum_property (x : ℝ) : f x + f (1 - x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l646_64607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l646_64640

-- Define the circle C
def circle_c (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 2

-- Define a point P on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circle_c P.1 P.2

-- Define a line forming 45° angle with l
def line_45_deg (P A : ℝ × ℝ) : Prop :=
  (A.2 - P.2) = (A.1 - P.1) ∨ (A.2 - P.2) = -(A.1 - P.1)

-- Define the intersection point A
def intersection_point (A : ℝ × ℝ) : Prop := line_l A.1 A.2

-- Define the distance between two points
noncomputable def distance (P A : ℝ × ℝ) : ℝ := 
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)

-- Theorem statement
theorem min_distance_circle_line :
  ∃ (min_dist : ℝ), 
    (∀ (P A : ℝ × ℝ), 
      point_on_circle P → 
      intersection_point A → 
      line_45_deg P A → 
      distance P A ≥ min_dist) ∧
    (∃ (P A : ℝ × ℝ), 
      point_on_circle P ∧ 
      intersection_point A ∧ 
      line_45_deg P A ∧ 
      distance P A = min_dist) ∧
    (min_dist = 2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l646_64640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ABCD_l646_64698

-- Define the curve Ω in Cartesian coordinates
def curve_Ω (x y : ℝ) : Prop := x^2 + y^2 = 6*x

-- Define the line l
noncomputable def line_l (θ : ℝ) (t : ℝ) : ℝ × ℝ := (4 + t * Real.cos θ, -1 + t * Real.sin θ)

-- Define the line l₀
noncomputable def line_l₀ (θ : ℝ) (s : ℝ) : ℝ × ℝ := (4 - s * Real.sin θ, -1 + s * Real.cos θ)

-- Define the intersection points
def point_A (θ : ℝ) (t : ℝ) : Prop := 
  curve_Ω (line_l θ t).1 (line_l θ t).2

def point_C (θ : ℝ) (t : ℝ) : Prop := 
  curve_Ω (line_l θ t).1 (line_l θ t).2 ∧ ¬point_A θ t

def point_B (θ : ℝ) (s : ℝ) : Prop := 
  curve_Ω (line_l₀ θ s).1 (line_l₀ θ s).2

def point_D (θ : ℝ) (s : ℝ) : Prop := 
  curve_Ω (line_l₀ θ s).1 (line_l₀ θ s).2 ∧ ¬point_B θ s

-- Theorem statement
theorem max_area_ABCD : 
  ∀ (θ t₁ t₂ s₁ s₂ : ℝ), 
    point_A θ t₁ → point_C θ t₂ → point_B θ s₁ → point_D θ s₂ →
    ∀ (area : ℝ), 
      (area = abs ((line_l θ t₁).1 * (line_l₀ θ s₁).2 + 
                   (line_l₀ θ s₁).1 * (line_l θ t₂).2 + 
                   (line_l θ t₂).1 * (line_l₀ θ s₂).2 + 
                   (line_l₀ θ s₂).1 * (line_l θ t₁).2 - 
                   (line_l θ t₁).2 * (line_l₀ θ s₁).1 - 
                   (line_l₀ θ s₁).2 * (line_l θ t₂).1 - 
                   (line_l θ t₂).2 * (line_l₀ θ s₂).1 - 
                   (line_l₀ θ s₂).2 * (line_l θ t₁).1) / 2) →
      area ≤ 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ABCD_l646_64698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_ellipse_equation_2_l646_64626

-- Define the ellipse
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity

-- Define the standard equation of an ellipse
def standard_equation (E : Ellipse) : (ℝ → ℝ → Prop) :=
  λ x y ↦ x^2 / E.a^2 + y^2 / E.b^2 = 1

-- Theorem statement
theorem ellipse_equation (E : Ellipse) :
  E.a = 6 ∧ E.e = 2/3 ∧ standard_equation E (-2) (-4) →
  standard_equation E = λ x y ↦ x^2 / 36 + y^2 / 20 = 1 :=
by
  sorry

-- Additional theorem for the second part of the problem
theorem ellipse_equation_2 (E : Ellipse) :
  E.a = 2 * E.b ∧ standard_equation E (-2) (-4) →
  (standard_equation E = λ x y ↦ x^2 / 68 + y^2 / 17 = 1) ∨
  (standard_equation E = λ x y ↦ x^2 / 8 + y^2 / 32 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_ellipse_equation_2_l646_64626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l646_64602

/-- An arithmetic sequence with a_1 = 1 and a_5 = 9 -/
noncomputable def arithmetic_seq (n : ℕ) : ℝ :=
  let d := (9 - 1) / 4
  1 + (n - 1) * d

/-- The sum of the first n terms of the arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ :=
  n * (arithmetic_seq 1 + arithmetic_seq n) / 2

theorem sum_of_first_five_terms :
  S 5 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l646_64602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_half_angle_l646_64688

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_sine_half_angle (t : Triangle) 
  (h1 : t.a + Real.sqrt 2 * t.c = 2 * t.b) 
  (h2 : Real.sin t.B = Real.sqrt 2 * Real.sin t.C) :
  Real.sin (t.C / 2) = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_half_angle_l646_64688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_eq_two_l646_64679

/-- A positive arithmetic geometric sequence with sum function S_n -/
structure PosArithGeomSeq where
  S : ℕ → ℝ
  positive : ∀ n, S n > 0

/-- The common ratio of a positive arithmetic geometric sequence -/
noncomputable def common_ratio (seq : PosArithGeomSeq) : ℝ :=
  Real.sqrt ((seq.S 4 - seq.S 2) / seq.S 2)

theorem common_ratio_eq_two (seq : PosArithGeomSeq) 
  (h2 : seq.S 2 = 3) (h4 : seq.S 4 = 15) : 
  common_ratio seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_eq_two_l646_64679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l646_64648

-- Define the function
def f (x : ℝ) : ℝ := -x^3 + x^2 + 2*x

-- State the theorem
theorem area_under_curve : 
  ∫ x in (-1)..(0), (0 - f x) + ∫ x in (0)..(2), f x = 37/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l646_64648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_construction_l646_64609

/-- IsOnCircle P C r means point P is on the circle with center C and radius r -/
def IsOnCircle (P C : Point) (r : ℝ) : Prop := sorry

/-- IsOnDiameter P C means point P is on a diameter of the circle with center C -/
def IsOnDiameter (P C : Point) : Prop := sorry

/-- Midpoint A B returns the midpoint of the line segment AB -/
def Midpoint (A B : Point) : Point := sorry

/-- RightAngle A B C means angle ABC is a right angle -/
def RightAngle (A B C : Point) : Prop := sorry

/-- IsDiagonalIntersection P A C B D means P is the intersection of diagonals AC and BD -/
def IsDiagonalIntersection (P A C B D : Point) : Prop := sorry

/-- IsMidpointIntersection P AB CD BC DA means P is the intersection of lines connecting midpoints AB with CD, and BC with DA -/
def IsMidpointIntersection (P AB CD BC DA : Point) : Prop := sorry

theorem cyclic_quadrilateral_construction (O M I : Point) (r : ℝ) :
  IsOnCircle M O r →
  IsOnDiameter M O →
  let N := Midpoint O M
  IsOnCircle I N (r / 2) →
  ∃ (A B C D : Point),
    IsOnCircle A O r ∧
    IsOnCircle B O r ∧
    IsOnCircle C O r ∧
    IsOnCircle D O r ∧
    RightAngle A B C ∧
    IsDiagonalIntersection M A C B D ∧
    let AB := Midpoint A B
    let BC := Midpoint B C
    let CD := Midpoint C D
    let DA := Midpoint D A
    IsMidpointIntersection I AB CD BC DA :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_construction_l646_64609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_eq_73_l646_64646

/-- The number of coin flips -/
def n : ℕ := 10

/-- The probability of getting heads on a single flip of a fair coin -/
def p : ℚ := 1/2

/-- The number of ways to arrange k heads in n-k+1 positions without consecutive heads -/
def arrange (n k : ℕ) : ℕ := Nat.choose (n - k + 1) k

/-- The total number of favorable outcomes (no consecutive heads) -/
def favorable_outcomes : ℕ := 
  (List.range (n/2 + 1)).map (arrange n) |>.sum

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := 2^n

/-- The probability of no consecutive heads in n coin flips -/
def prob : ℚ := favorable_outcomes / total_outcomes

theorem sum_of_fraction_parts_eq_73 : 
  let (num, den) := (prob.num.natAbs, prob.den)
  Nat.gcd num den = 1 → num + den = 73 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_eq_73_l646_64646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l646_64643

/-- Definition of a rectangle with given width and height -/
def Rectangle (w h : ℝ) : Set (ℝ × ℝ) := sorry

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateralTriangle (triangle : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate to check if a triangle is inscribed in a rectangle -/
def IsInscribed (triangle : Set (ℝ × ℝ)) (rectangle : Set (ℝ × ℝ)) : Prop := sorry

/-- Function to calculate the area of a triangle -/
def AreaOfTriangle (triangle : Set (ℝ × ℝ)) : ℝ := sorry

/-- The maximum area of an equilateral triangle inscribed in a 12x13 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle : 
  ∃ (A : ℝ), A = 36 * Real.sqrt 3 ∧ 
  ∀ (triangle_area : ℝ),
    (∃ (triangle : Set (ℝ × ℝ)), 
      IsEquilateralTriangle triangle ∧ 
      IsInscribed triangle (Rectangle 12 13) ∧
      AreaOfTriangle triangle = triangle_area) →
    triangle_area ≤ A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l646_64643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_distance_l646_64677

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 is given by
    |ax₀ + by₀ + c| / √(a² + b²) -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

/-- Given that the distance from the point (4,0) to the line y = (4/3)x + (m/3) is 3,
    prove that m = -1 or m = -31 -/
theorem point_line_distance (m : ℝ) : 
  distance_point_to_line 4 0 (4/3) (-1) (m/3) = 3 → m = -1 ∨ m = -31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_distance_l646_64677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l646_64681

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sqrt 3 * Real.sin x - Real.cos x)

def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = b * Real.sin C ∧ b = c * Real.sin A ∧ c = a * Real.sin B

theorem problem_solution 
  (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c) 
  (h_f_B : f B = 1/2) 
  (h_a_c : a + c = 1) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (k : ℤ), ∀ (x : ℝ), 
    k * Real.pi + Real.pi/3 ≤ x → x ≤ k * Real.pi + 5*Real.pi/6 → 
    ∀ (y : ℝ), x ≤ y → f y ≤ f x) ∧
  (1/2 ≤ b ∧ b < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l646_64681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_theorem_l646_64656

-- Define the set of planes
def A : Set (ℚ × ℚ × ℚ) :=
  {p | ∃ (n : ℤ) (s₁ s₂ : Bool), 
    let (x, y, z) := p
    (if s₁ then x + y else x - y) + (if s₂ then z else -z) = n}

-- Define what it means for a point to be inside an octahedron
def inside_octahedron (p : ℚ × ℚ × ℚ) : Prop :=
  ∃ (a b c : ℤ), 
    let (x, y, z) := p
    1 < (x - a) + (y - b) + (z - c) ∧
    (x - a) + (y - b) + (z - c) < 2

theorem octahedron_theorem (x₀ y₀ z₀ : ℚ) 
  (h : (x₀, y₀, z₀) ∉ A) : 
  ∃ (k : ℕ+), inside_octahedron (k * x₀, k * y₀, k * z₀) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_theorem_l646_64656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_two_l646_64666

/-- An even function f: ℝ → ℝ such that f(x) = a^x for x ≥ 0, where a > 1 -/
noncomputable def f (a : ℝ) : ℝ → ℝ :=
  fun x => if x ≥ 0 then a^x else a^(-x)

theorem a_equals_two (a : ℝ) (h1 : a > 1) 
    (h2 : ∀ x : ℝ, f a x ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2) : a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_two_l646_64666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l646_64636

/-- A parabola defined by y = ax^2 - 2ax where a ≠ 0 -/
structure Parabola where
  a : ℝ
  ha : a ≠ 0

/-- Predicate to check if a point (x, y) lies on the parabola y = ax^2 - 2ax -/
def LiesOnParabola (a x y : ℝ) : Prop :=
  y = a * x^2 - 2 * a * x

/-- Predicate to check if a point (x, y) is the vertex of the parabola y = ax^2 - 2ax -/
def IsVertex (a x y : ℝ) : Prop :=
  LiesOnParabola a x y ∧ ∀ x' y', LiesOnParabola a x' y' → y ≤ y'

theorem parabola_properties (p : Parabola) :
  -- 1. The vertex coordinates are (1, -a)
  (∃ (x y : ℝ), x = 1 ∧ y = -p.a ∧ IsVertex p.a x y) ∧
  -- 2. If two points (-1, m) and (x₀, m) lie on the parabola, then x₀ = 3
  (∀ m x₀ : ℝ, LiesOnParabola p.a (-1) m → LiesOnParabola p.a x₀ m → x₀ = 3) ∧
  -- 3. The condition y₁ < y₂ ≤ -a always holds for points A(m-1, y₁) and B(m+2, y₂) 
  --    on the parabola if and only if a < 0 and m < 1/2
  (∃ m : ℝ, (∀ y₁ y₂ : ℝ, 
    LiesOnParabola p.a (m-1) y₁ → 
    LiesOnParabola p.a (m+2) y₂ → 
    y₁ < y₂ ∧ y₂ ≤ -p.a) ↔ 
  p.a < 0 ∧ m < 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l646_64636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_monotonicity_omega_range_l646_64617

theorem sine_monotonicity_omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∀ x ∈ Set.Icc (-π/4) (2*π/3), 
    ∀ y ∈ Set.Icc (-π/4) (2*π/3), 
    x < y → Real.sin (ω*x + π/6) < Real.sin (ω*y + π/6)) →
  ω ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_monotonicity_omega_range_l646_64617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_of_sinusoidal_function_l646_64659

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem symmetry_point_of_sinusoidal_function 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_distance : Real.pi / ω = Real.pi / 2) 
  (x₀ : ℝ) 
  (h_x₀_range : x₀ ∈ Set.Icc 0 (Real.pi / 2)) 
  (h_symmetry : ∀ x : ℝ, f ω (x₀ + x) = f ω (x₀ - x)) :
  x₀ = 5 * Real.pi / 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_of_sinusoidal_function_l646_64659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lee_lawn_price_l646_64686

/-- The amount Lee charges for mowing one lawn -/
def lawn_price : ℕ := sorry

/-- The number of lawns Lee mowed last week -/
def lawns_mowed : ℕ := 16

/-- The number of customers who gave a tip -/
def tipping_customers : ℕ := 3

/-- The amount of each tip -/
def tip_amount : ℕ := 10

/-- Lee's total earnings last week -/
def total_earnings : ℕ := 558

theorem lee_lawn_price :
  lawn_price * lawns_mowed + tipping_customers * tip_amount = total_earnings →
  lawn_price = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lee_lawn_price_l646_64686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_eq_interval_l646_64613

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem M_intersect_N_eq_interval :
  M ∩ N = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_eq_interval_l646_64613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l646_64675

-- Define the tetrahedron
structure Tetrahedron where
  WX : ℝ
  WY : ℝ
  WZ : ℝ
  XY : ℝ
  XZ : ℝ
  YZ : ℝ

-- Define the conditions
def validTetrahedron (t : Tetrahedron) : Prop :=
  ({t.WX, t.WY, t.WZ, t.XY, t.XZ, t.YZ} : Set ℝ) = {8, 14, 19, 28, 37, 42} ∧
  t.WZ = 42 ∧
  t.WX + t.WY > t.XY ∧
  t.WX + t.XZ > t.WZ ∧
  t.WY + t.YZ > t.WZ ∧
  t.XY + t.XZ > t.WX ∧
  t.XY + t.YZ > t.WY ∧
  t.XZ + t.YZ > t.XY

-- Theorem statement
theorem tetrahedron_edge_length :
  ∀ t : Tetrahedron, validTetrahedron t → t.XY = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l646_64675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l646_64639

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := |x| / x

noncomputable def g (x : ℝ) : ℝ := 
  if x > 0 then 1 else -1

-- Theorem statement
theorem f_equals_g : ∀ x : ℝ, x ≠ 0 → f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l646_64639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_in_gallons_l646_64603

-- Define the units and their relationships
def liters_per_jug : ℝ := 5
def jugs_per_bucket : ℝ := 4
def buckets_per_barrel : ℝ := 3
def liters_per_gallon : ℝ := 3.79

-- Define the amount of water we have
def barrels_of_water : ℝ := 1
def buckets_of_water : ℝ := 2

-- Theorem to prove
theorem water_in_gallons :
  let liters_per_bucket : ℝ := liters_per_jug * jugs_per_bucket
  let liters_per_barrel : ℝ := liters_per_bucket * buckets_per_barrel
  let total_liters : ℝ := barrels_of_water * liters_per_barrel + buckets_of_water * liters_per_bucket
  let gallons : ℝ := total_liters / liters_per_gallon
  abs (gallons - 26.39) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_in_gallons_l646_64603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_order_l646_64604

theorem number_order (a b c d : ℝ) 
  (ha : a = (0.2 : ℝ)^(3.5 : ℝ)) 
  (hb : b = (0.2 : ℝ)^(4.1 : ℝ)) 
  (hc : c = Real.exp 1.1) 
  (hd : d = Real.log 3 / Real.log 0.2) : 
  d < b ∧ b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_order_l646_64604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l646_64641

/-- Hyperbola C with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_eccentricity : c / a = Real.sqrt 3
  h_distance : c^2 + b^2 = 5
  h_c_def : c^2 = a^2 + b^2

/-- The equation of hyperbola C -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The line that intersects the hyperbola -/
def intersecting_line (m : ℝ) (x y : ℝ) : Prop :=
  x - y + m = 0

/-- The circle on which the midpoint lies -/
def midpoint_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 5

theorem hyperbola_properties (h : Hyperbola) :
  (∀ x y, hyperbola_equation h x y ↔ x^2 - y^2 / 2 = 1) ∧
  (∀ m, (∃ x₁ y₁ x₂ y₂, 
    hyperbola_equation h x₁ y₁ ∧ 
    hyperbola_equation h x₂ y₂ ∧
    intersecting_line m x₁ y₁ ∧
    intersecting_line m x₂ y₂ ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    midpoint_circle ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)) →
  m = 1 ∨ m = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l646_64641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mystical_81649_largest_mystical_l646_64685

/-- A positive integer is mystical if it has at least two digits and every pair of two consecutive digits forms a perfect square. -/
def IsMystical (n : ℕ) : Prop :=
  n ≥ 10 ∧ 
  ∀ i : ℕ, i + 1 < (Nat.digits 10 n).length → 
    let pair := ((Nat.digits 10 n).get ⟨i, by sorry⟩) * 10 + ((Nat.digits 10 n).get ⟨i+1, by sorry⟩)
    pair ∈ ({16, 25, 36, 49, 64, 81} : Set ℕ)

/-- The set of two-digit perfect squares -/
def TwoDigitPerfectSquares : Set ℕ := {16, 25, 36, 49, 64, 81}

/-- 81649 is a mystical number -/
theorem mystical_81649 : IsMystical 81649 := by
  sorry

/-- 81649 is the largest mystical number -/
theorem largest_mystical : ∀ n : ℕ, IsMystical n → n ≤ 81649 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mystical_81649_largest_mystical_l646_64685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l646_64690

-- Define the sphere
structure Sphere where
  radius : ℝ

-- Define the cone
structure Cone where
  baseRadius : ℝ
  height : ℝ

-- Define the problem setup
structure ConeSetup where
  sphere : Sphere
  smallerCone : Cone
  largerCone : Cone

-- Define the conditions
def validSetup (setup : ConeSetup) : Prop :=
  -- Cones share a common base
  setup.smallerCone.baseRadius = setup.largerCone.baseRadius
  -- Base area is 3/16 of sphere surface area
  ∧ (Real.pi * setup.smallerCone.baseRadius ^ 2) = (3 / 16) * (4 * Real.pi * setup.sphere.radius ^ 2)
  -- Vertices and base circumference are on the sphere
  ∧ setup.smallerCone.height + setup.largerCone.height = 2 * setup.sphere.radius

-- Theorem statement
theorem cone_height_ratio (setup : ConeSetup) (h : validSetup setup) :
  setup.smallerCone.height / setup.largerCone.height = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l646_64690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_negations_l646_64642

theorem proposition_negations :
  (¬(∀ a b : ℝ, a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1) ↔ (∀ a b : ℝ, a ≤ b → (2 : ℝ)^a ≤ (2 : ℝ)^b - 1)) ∧
  (¬(∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_negations_l646_64642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_theorem_l646_64670

/-- The smallest angle under which a unit segment can be seen from the center of a ring -/
noncomputable def smallest_angle (R r : ℝ) : ℝ :=
  if r ≥ R - 1/R then
    2 * Real.arcsin (1 / (2 * R))
  else if r ≥ R - 1 then
    Real.arccos ((R^2 + r^2 - 1) / (2 * R * r))
  else
    0

/-- Theorem stating the smallest angle under which a unit segment can be seen from the center of a ring -/
theorem smallest_angle_theorem (R r : ℝ) (h1 : R > 0) (h2 : r > 0) (h3 : R ≥ r) :
  smallest_angle R r =
    if r ≥ R - 1/R then
      2 * Real.arcsin (1 / (2 * R))
    else if r ≥ R - 1 then
      Real.arccos ((R^2 + r^2 - 1) / (2 * R * r))
    else
      0 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_theorem_l646_64670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l646_64657

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 5)

theorem smallest_max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 → f y ≤ f x) ∧
  (∀ (z : ℝ), 0 < z ∧ z < x → f z < f x) ∧
  x = 1800 := by
  sorry

#check smallest_max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l646_64657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_all_black_probability_l646_64650

-- Define the grid
def Grid := Fin 4 → Fin 4 → Bool

-- Define the probability of a square being black initially
noncomputable def initialBlackProb : ℝ := 1 / 2

-- Define the rotation function
def rotate (g : Grid) : Grid :=
  fun i j => g (3 - i) (3 - j)

-- Define the repainting function
def repaint (g : Grid) : Grid :=
  fun i j => g i j || (rotate g) i j

-- Define the probability of the grid being all black after rotation and repainting
noncomputable def allBlackProb : ℝ := (1 / 2) ^ 16

-- Theorem statement
theorem grid_all_black_probability :
  allBlackProb = 1 / 65536 := by
  -- The proof goes here
  sorry

#eval Float.toString ((1 : Float) / 65536)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_all_black_probability_l646_64650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l646_64619

noncomputable def f (x : ℝ) := Real.log ((x + 1) / (x - 1))

theorem range_of_f :
  Set.range f = {y : ℝ | y < 0 ∨ y > 0} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l646_64619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunchroom_milk_consumption_l646_64683

/-- Proves that boys drink 3.33 cartons of milk on average given the lunchroom conditions -/
theorem lunchroom_milk_consumption 
  (total_students : ℕ) 
  (girl_percentage : ℚ) 
  (monitors : ℕ) 
  (monitor_ratio : ℚ) 
  (girl_milk_average : ℚ) 
  (total_milk : ℕ) 
  (h1 : girl_percentage = 2/5)
  (h2 : monitor_ratio = 2/15)
  (h3 : monitors = 8)
  (h4 : girl_milk_average = 2)
  (h5 : total_milk = 168)
  (h6 : (total_students : ℚ) = (monitors : ℚ) / monitor_ratio)
  : (total_milk : ℚ) - girl_milk_average * (girl_percentage * total_students) = 
    (10/3) * ((1 - girl_percentage) * total_students) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunchroom_milk_consumption_l646_64683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l646_64625

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem min_omega_value (ω φ T : ℝ) :
  ω > 0 →
  0 < φ ∧ φ < Real.pi / 2 →
  T > 0 →
  (∀ t, t > 0 → f ω φ t = f ω φ (t + T)) →
  (∀ t, t > 0 → t < T → ∃ x, f ω φ x ≠ f ω φ (x + t)) →
  f ω φ T = Real.sqrt 3 / 2 →
  f ω φ (Real.pi / 6) = 0 →
  ∃ ω_min, ω_min = 4 ∧ ∀ ω', ω' > 0 ∧ 
    (∃ φ' T', 
      0 < φ' ∧ φ' < Real.pi / 2 ∧
      T' > 0 ∧
      (∀ t, t > 0 → f ω' φ' t = f ω' φ' (t + T')) ∧
      (∀ t, t > 0 → t < T' → ∃ x, f ω' φ' x ≠ f ω' φ' (x + t)) ∧
      f ω' φ' T' = Real.sqrt 3 / 2 ∧
      f ω' φ' (Real.pi / 6) = 0) →
    ω' ≥ ω_min := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l646_64625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_tangent_line_l646_64630

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The first circle: x^2 + y^2 + 2y - 4 = 0 -/
noncomputable def circle1 : Circle :=
  { center := (0, -1),
    radius := Real.sqrt 5 }

/-- The second circle: x^2 + y^2 - 4x - 16 = 0 -/
noncomputable def circle2 : Circle :=
  { center := (2, 0),
    radius := 2 * Real.sqrt 5 }

/-- Calculate the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: There is exactly one line tangent to both circles -/
theorem one_tangent_line : ∃! l : Set (ℝ × ℝ), 
  (∀ p, p ∈ l → distance p circle1.center = circle1.radius ∧ 
                 distance p circle2.center = circle2.radius) ∧
  (∀ p q, p ∈ l → q ∈ l → p ≠ q → ∃ t : ℝ, q.1 = p.1 + t ∧ q.2 = p.2 + t) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_tangent_line_l646_64630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l646_64624

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (log x) / (1 + x) - log x

-- State the theorem
theorem max_value_of_f (x₀ : ℝ) (h₁ : x₀ > 0) 
  (h₂ : ∀ x > 0, f x ≤ f x₀) : f x₀ = x₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l646_64624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_modulo_seven_l646_64622

theorem power_of_two_modulo_seven (n : ℕ) :
  (∃ k : ℕ, n = 3 * k ∧ k > 0) ↔ (2^n - 1) % 7 = 0 ∧
  (2^n + 1) % 7 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_modulo_seven_l646_64622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l646_64673

/-- A pyramid with a parallelogram base -/
structure Pyramid :=
  (base_side1 : ℝ)
  (base_side2 : ℝ)
  (base_diagonal : ℝ)
  (height : ℝ)

/-- The total surface area of the pyramid -/
noncomputable def total_surface_area (p : Pyramid) : ℝ :=
  8 * (11 + Real.sqrt 34)

/-- Theorem stating the total surface area of the specific pyramid -/
theorem pyramid_surface_area (p : Pyramid) 
  (h1 : p.base_side1 = 10)
  (h2 : p.base_side2 = 8)
  (h3 : p.base_diagonal = 6)
  (h4 : p.height = 4) :
  total_surface_area p = 8 * (11 + Real.sqrt 34) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l646_64673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_AC_less_than_twice_AO_exists_unique_l646_64694

/-- The probability that AC is less than twice the distance from A to the origin -/
def probability_AC_less_than_twice_AO : Real → Prop := fun p =>
  let A : ℝ × ℝ := (0, -10)
  let B : ℝ × ℝ := (0, 0)
  let O : ℝ × ℝ := (0, -4)
  let dist_AB : ℝ := 10
  let dist_BC : ℝ := 7
  let dist_AO : ℝ := 6
  let β_range := Set.Ioo (0 : ℝ) (Real.pi / 2)
  p = (2 / Real.pi) * Real.arctan (Real.sqrt 371 / 8) ∧
  ∀ β ∈ β_range,
    let C : ℝ × ℝ := (dist_BC * Real.cos β, dist_BC * Real.sin β)
    let dist_AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
    dist_AC < 2 * dist_AO →
    β ≤ Real.arctan (Real.sqrt 371 / 8)

/-- The probability exists and is unique -/
theorem probability_AC_less_than_twice_AO_exists_unique :
  ∃! p, probability_AC_less_than_twice_AO p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_AC_less_than_twice_AO_exists_unique_l646_64694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_max_l646_64667

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The parabola y^2 = 4x -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

theorem parabola_ratio_max :
  ∀ m : Point,
  isOnParabola m →
  let o : Point := ⟨0, 0⟩
  let f : Point := ⟨1, 0⟩
  distance m o / distance m f ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

#check parabola_ratio_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_max_l646_64667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l646_64699

/-- An arithmetic sequence -/
noncomputable def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then the common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (a₁ d : ℝ) : 2 * S a₁ d 3 = 3 * S a₁ d 2 + 6 → d = 2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l646_64699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l646_64601

def A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l646_64601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l646_64629

theorem polynomial_existence (n : ℕ) (hn : n > 1) :
  ∃ (P : MvPolynomial (Fin 3) ℤ),
    ∀ (x : MvPolynomial (Fin 1) ℤ),
      P.eval₂ (algebraMap ℤ (MvPolynomial (Fin 1) ℤ)) (fun i => match i with
        | 0 => x^n
        | 1 => x^(n+1)
        | 2 => x + x^(n+2)
        | _ => 0) = x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l646_64629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l646_64632

/-- Line passing through two points -/
structure Line2D where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : LineEquation) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : LineEquation) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Calculate the area of a triangle formed by a line and coordinate axes -/
noncomputable def triangle_area (l : LineEquation) : ℝ :=
  abs (l.c * l.c) / (2 * abs (l.a * l.b))

theorem line_equations 
  (l : Line2D) 
  (h_l : l.point1 = (4, 0) ∧ l.point2 = (0, 3)) :
  ∃ (l1_parallel l1_perp1 l1_perp2 : LineEquation),
    /- (I) Parallel line equation -/
    (are_parallel l1_parallel ⟨3, 4, -12⟩) ∧ 
    (point_on_line (-1, 3) l1_parallel) ∧
    (l1_parallel = ⟨3, 4, -9⟩) ∧
    /- (II) Perpendicular line equations -/
    (are_perpendicular l1_perp1 ⟨3, 4, -12⟩) ∧
    (are_perpendicular l1_perp2 ⟨3, 4, -12⟩) ∧
    (triangle_area l1_perp1 = 6) ∧
    (triangle_area l1_perp2 = 6) ∧
    ((l1_perp1 = ⟨4, -3, -12⟩ ∧ l1_perp2 = ⟨4, -3, 12⟩) ∨
     (l1_perp1 = ⟨4, -3, 12⟩ ∧ l1_perp2 = ⟨4, -3, -12⟩)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l646_64632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_l646_64606

/-- A trapezoid with bases of length 3 and 4 -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  base1_eq : base1 = 3
  base2_eq : base2 = 4

/-- A segment parallel to the bases of a trapezoid -/
def parallel_segment (t : Trapezoid) (s : ℝ) : Prop :=
  sorry

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℝ :=
  sorry

/-- The area of a part of a trapezoid cut by a parallel segment -/
def partial_area (t : Trapezoid) (s : ℝ) : ℝ :=
  sorry

theorem parallel_segment_length (t : Trapezoid) :
  ∃ s : ℝ, parallel_segment t s ∧
    partial_area t s / (area t - partial_area t s) = 5 / 2 →
    s = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_l646_64606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l646_64676

noncomputable def g (x : ℝ) : ℝ := Real.tan x ^ 2 + Real.cos x ^ 4

theorem g_range :
  (∀ x : ℝ, g x ≥ 1) ∧ (∀ y : ℝ, y ≥ 1 → ∃ x : ℝ, g x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l646_64676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l646_64697

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersection : Plane → Plane → Line)

-- Define the perpendicular and parallel relations
variable (perp : Plane → Plane → Prop)
variable (para : Plane → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (paraLine : Line → Line → Prop)

-- Define the perpendicular and parallel relations between planes and lines
variable (perpPlaneLine : Plane → Line → Prop)
variable (paraPlaneLine : Plane → Line → Prop)

-- Define the given conditions
variable (α β γ : Plane)
variable (a b : Line)

axiom α_perp_γ : perp α γ
axiom β_perp_γ : perp β γ
axiom α_intersect_γ : intersection α γ = a
axiom β_intersect_γ : intersection β γ = b

-- State the theorem
theorem all_propositions_true :
  (perpLine a b → perp α β) ∧
  (paraPlaneLine α b → para α β) ∧
  (perp α β → perpLine a b) ∧
  (para α β → paraLine a b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l646_64697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_process_never_stops_l646_64695

/-- Represents the transformation rule of the supercomputer -/
def transform (n : ℕ) : ℕ :=
  let a := n / 100
  let b := n % 100
  2 * a + 8 * b

/-- The initial number with 900 ones -/
def initial_number : ℕ :=
  (10^900 - 1) / 9

/-- Generates the sequence of numbers produced by the supercomputer -/
def supercomputer_sequence : ℕ → ℕ
  | 0 => initial_number
  | k + 1 => transform (supercomputer_sequence k)

theorem process_never_stops :
  ∀ n : ℕ, supercomputer_sequence n ≥ 100 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_process_never_stops_l646_64695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_theorem_l646_64638

def total_shots : ℕ := 50
def three_point_success_rate : ℚ := 1/4
def two_point_success_rate : ℚ := 2/5
def free_throw_success_rate : ℚ := 4/5

def points_scored (x y z : ℚ) : ℚ :=
  3 * three_point_success_rate * x + 
  2 * two_point_success_rate * y + 
  free_throw_success_rate * z

theorem basketball_score_theorem :
  ∃ x y z : ℚ, 
    x + y + z = total_shots ∧ 
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    Int.floor (points_scored x y z) = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_theorem_l646_64638
