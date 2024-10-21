import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_removal_theorem_l662_66209

theorem digit_removal_theorem :
  (∀ m n : ℕ, 
    (∃ k : ℕ, m = 6 * 10^(n-1) + k ∧ k = m / 25) → 
    (∃ p : ℕ, n = 4*p + 2 ∧ m = 625 * 10^p)) ∧
  (∀ m n : ℕ, ¬(∃ k : ℕ, m = 6 * 10^(n-1) + k ∧ k = m / 35)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_removal_theorem_l662_66209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l662_66280

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

-- State the theorem
theorem even_function_implies_a_equals_two (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l662_66280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_together_l662_66229

/-- The time (in hours) it takes P to complete the job alone -/
noncomputable def p_time : ℝ := 3

/-- The time (in hours) it takes Q to complete the job alone -/
noncomputable def q_time : ℝ := 9

/-- The time (in hours) P works alone after working with Q -/
noncomputable def p_extra_time : ℝ := 1/3

/-- The proportion of the job completed when both P and Q work together -/
noncomputable def combined_work (t : ℝ) : ℝ := t * (1/p_time + 1/q_time)

/-- The proportion of the job P completes in the extra time working alone -/
noncomputable def p_extra_work : ℝ := p_extra_time / p_time

/-- The theorem stating that P and Q work together for 2 hours -/
theorem work_time_together : ∃ t : ℝ, t = 2 ∧ combined_work t + p_extra_work = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_together_l662_66229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cinema_lineup_arrangements_l662_66282

theorem cinema_lineup_arrangements (n : ℕ) (m : ℕ) : 
  n = 8 → m = 2 → (Nat.factorial (n - m + 1)) * (Nat.factorial m) = 10080 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cinema_lineup_arrangements_l662_66282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_and_monotonicity_l662_66291

open Real

/-- The function f(x) = x - a ln x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * log x

theorem f_tangent_and_monotonicity (a : ℝ) :
  (∀ x > 0,
    (a = 2 → (∀ y, y = f 2 x → x + y - 2 = 0 ↔ (x = 1 ∧ y = f 2 1))) ∧
    (a ≤ 0 → Monotone (f a)) ∧
    (a > 0 → StrictAntiOn (f a) (Set.Ioo 0 a) ∧
              StrictMonoOn (f a) (Set.Ioi a))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_and_monotonicity_l662_66291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_is_two_fifths_l662_66245

/-- The number of white balls in the bag -/
def white_balls : ℕ := 2

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 2

/-- The probability of drawing two balls of the same color -/
def same_color_probability : ℚ := 2 / 5

/-- Theorem stating that the probability of drawing two balls of the same color is 2/5 -/
theorem same_color_probability_is_two_fifths :
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls : ℚ) / 
  (Nat.choose total_balls drawn_balls : ℚ) = same_color_probability := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_is_two_fifths_l662_66245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_given_numbers_l662_66288

theorem largest_among_given_numbers :
  ∀ (x : ℝ), x ∈ ({1, -1, 0, Real.sqrt 2} : Set ℝ) → x ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_given_numbers_l662_66288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tank_water_proof_l662_66218

/-- The amount of water needed for each of the first two fish tanks -/
def X : ℝ := sorry

/-- The total number of fish tanks -/
def total_tanks : ℕ := 4

/-- The number of weeks for water change -/
def weeks : ℕ := 4

/-- The total amount of water needed for all tanks in four weeks -/
def total_water : ℝ := 112

/-- The amount of water needed for each of the other two tanks -/
def other_tank_water : ℝ := X - 2

theorem fish_tank_water_proof :
  (2 * X + 2 * other_tank_water) * weeks = total_water →
  X = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tank_water_proof_l662_66218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_terminal_side_l662_66260

/-- If the terminal side of angle α passes through point P(-3, -4), then sin α = -4/5 -/
theorem sin_alpha_terminal_side (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = -3 ∧ r * (Real.sin α) = -4) → 
  Real.sin α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_terminal_side_l662_66260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_destroyed_rose_bushes_l662_66286

/-- Calculates the number of destroyed rose bushes given the gardening project costs and details --/
theorem destroyed_rose_bushes (rose_bush_cost labor_rate hours_per_day work_days soil_volume soil_cost_per_unit total_cost : ℚ) : 
  rose_bush_cost = 150 →
  labor_rate = 30 →
  hours_per_day = 5 →
  work_days = 4 →
  soil_volume = 100 →
  soil_cost_per_unit = 5 →
  total_cost = 4100 →
  (total_cost - (labor_rate * hours_per_day * work_days + soil_volume * soil_cost_per_unit)) / rose_bush_cost = 20 := by
  sorry

#eval (4100 : ℚ) - (30 * 5 * 4 + 100 * 5)
#eval ((4100 : ℚ) - (30 * 5 * 4 + 100 * 5)) / 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_destroyed_rose_bushes_l662_66286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_insufficient_l662_66284

/-- Represents the relationship between accidents, overtime hours, and workers -/
structure AccidentModel where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of accidents given overtime hours and workers -/
noncomputable def accidents (model : AccidentModel) (H : ℝ) (W : ℝ) : ℝ :=
  model.a * (model.b ^ H) * (model.c ^ W)

/-- Represents a data point of accidents, overtime hours, and workers -/
structure DataPoint where
  A : ℝ
  H : ℝ
  W : ℝ

/-- Theorem stating that two data points are insufficient to uniquely determine the model -/
theorem two_points_insufficient (d1 d2 : DataPoint) :
  ∃ (m1 m2 : AccidentModel), m1 ≠ m2 ∧
    accidents m1 d1.H d1.W = d1.A ∧
    accidents m1 d2.H d2.W = d2.A ∧
    accidents m2 d1.H d1.W = d1.A ∧
    accidents m2 d2.H d2.W = d2.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_insufficient_l662_66284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_downhill_speed_l662_66254

/-- Calculates the downhill speed given uphill speed, distances, and average speed -/
theorem calculate_downhill_speed (uphill_speed uphill_distance downhill_distance average_speed : ℝ)
  (h1 : uphill_speed = 30)
  (h2 : uphill_distance = 100)
  (h3 : downhill_distance = 50)
  (h4 : average_speed = 37.89)
  (h5 : uphill_speed > 0)
  (h6 : downhill_distance > 0)
  (h7 : average_speed > 0) :
  ∃ downhill_speed : ℝ,
    downhill_speed > 0 ∧ 
    (average_speed * (uphill_distance / uphill_speed + downhill_distance / downhill_speed) 
      = uphill_distance + downhill_distance) ∧
    (abs (downhill_speed - 79.97) < 0.01) := by
  sorry

#check calculate_downhill_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_downhill_speed_l662_66254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_when_t_is_three_fourths_l662_66242

theorem x_equals_y_when_t_is_three_fourths (t : ℚ) :
  (1 - 2 * t = 2 * t - 2) ↔ t = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_when_t_is_three_fourths_l662_66242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l662_66248

/-- The hyperbola with equation x²/4 - y²/2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) - (p.2^2 / 2) = 1}

/-- A focus of the hyperbola -/
noncomputable def Focus : ℝ × ℝ := (Real.sqrt 6, 0)

/-- An asymptote of the hyperbola -/
def Asymptote : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (Real.sqrt 2 / 2) * p.1}

/-- The distance from a point to a line in ℝ² -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

theorem distance_focus_to_asymptote :
  distancePointToLine Focus Asymptote = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l662_66248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_plus_two_alpha_l662_66267

theorem sin_pi_half_plus_two_alpha (α : ℝ) 
  (h : Real.cos α = -Real.sqrt 2 / 3) : 
  Real.sin (π / 2 + 2 * α) = -5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_plus_two_alpha_l662_66267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_distance_inequality_l662_66206

/-- Given a triangle ABC with side lengths a, b, c, and a point P in the plane,
    the sum of the products of distances from P to each pair of vertices,
    divided by the corresponding pair of side lengths, is at least 1. -/
theorem triangle_point_distance_inequality 
  (A B C P : ℝ × ℝ) (a b c : ℝ) : 
  let d := fun X Y : ℝ × ℝ ↦ Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  (a = d B C ∧ b = d C A ∧ c = d A B) →
  (d P B * d P C) / (b * c) + 
  (d P C * d P A) / (c * a) + 
  (d P A * d P B) / (a * b) ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_distance_inequality_l662_66206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_pump_time_l662_66285

/-- Represents the basement dimensions and water properties --/
structure BasementData where
  length : ℚ
  width : ℚ
  depth_inches : ℚ
  gallons_per_cubic_foot : ℚ

/-- Represents the pumping rates of the three pumps --/
structure PumpRates where
  pump1 : ℚ
  pump2 : ℚ
  pump3 : ℚ

/-- Calculates the time needed to pump out the water --/
def pumpOutTime (b : BasementData) (p : PumpRates) : ℚ :=
  let volume_cubic_feet := b.length * b.width * (b.depth_inches / 12)
  let volume_gallons := volume_cubic_feet * b.gallons_per_cubic_foot
  let total_pump_rate := p.pump1 + p.pump2 + p.pump3
  volume_gallons / total_pump_rate

theorem basement_pump_time :
  let basement := BasementData.mk 40 30 24 (15/2)
  let pumps := PumpRates.mk 8 8 12
  ⌈pumpOutTime basement pumps⌉ = 643 := by
  sorry

#eval ⌈pumpOutTime (BasementData.mk 40 30 24 (15/2)) (PumpRates.mk 8 8 12)⌉

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_pump_time_l662_66285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z₁_div_z₂_l662_66207

def z₁ : ℂ := 1 + 3 * Complex.I
def z₂ : ℂ := 3 + Complex.I

theorem imaginary_part_of_z₁_div_z₂ : 
  (z₁ / z₂).im = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z₁_div_z₂_l662_66207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l662_66281

/-- The length of a chord AB in an ellipse with equation x^2/2 + y^2 = 1,
    passing through the right focus and inclined at 45° -/
theorem ellipse_chord_length : ∃ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (x₁^2/2 + y₁^2 = 1) ∧
  (x₂^2/2 + y₂^2 = 1) ∧
  (y₂ - y₁ = x₂ - x₁) ∧
  (1, 0) ∈ Set.Icc A B ∧
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * Real.sqrt 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l662_66281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_intersection_on_ellipse_l662_66226

/-- An ellipse with foci at (0,3) and (4,0) that intersects the x-axis at (0,0) -/
structure Ellipse where
  /-- The sum of distances from any point on the ellipse to the two foci is constant -/
  constant_sum : ℝ
  /-- The constant sum is equal to the sum of distances from (0,0) to the foci -/
  sum_property : constant_sum = Real.sqrt (0^2 + 3^2) + Real.sqrt (4^2 + 0^2)
  /-- The origin (0,0) lies on the ellipse -/
  origin_on_ellipse : Real.sqrt (0^2 + 3^2) + Real.sqrt (4^2 + 0^2) = constant_sum

/-- The other intersection point of the ellipse with the x-axis -/
noncomputable def other_intersection (e : Ellipse) : ℝ × ℝ := (56/11, 0)

/-- Theorem stating that (56/11, 0) is the other intersection point of the ellipse with the x-axis -/
theorem other_intersection_on_ellipse (e : Ellipse) :
  let (x, y) := other_intersection e
  y = 0 ∧ Real.sqrt (x^2 + 3^2) + Real.sqrt ((x - 4)^2 + 0^2) = e.constant_sum :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_intersection_on_ellipse_l662_66226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l662_66296

/-- Represents a pyramid with a rectangular base -/
structure RectangularPyramid where
  -- Base dimensions
  ab : ℝ
  bc : ℝ
  -- Length of edge from apex to a base corner
  pb : ℝ

/-- Calculates the volume of a rectangular pyramid -/
noncomputable def volume (p : RectangularPyramid) : ℝ :=
  (1 / 3) * p.ab * p.bc * Real.sqrt (p.pb^2 - p.ab^2)

/-- The main theorem stating the volume of the specific pyramid -/
theorem pyramid_volume : 
  let p : RectangularPyramid := { ab := 12, bc := 6, pb := 25 }
  volume p = 24 * Real.sqrt 481 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l662_66296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_condition_l662_66266

/-- Given a positive integer n, we define a sequence of weights from 1 to n -/
def weights (n : ℕ) : List ℕ := List.range n

/-- The sum of weights from 1 to n -/
def sum_weights (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : ℕ) : Prop := n % 3 = 0

/-- Predicate to check if n satisfies the condition for equal division -/
def satisfies_condition (n : ℕ) : Prop :=
  divisible_by_three n ∨ divisible_by_three (n + 1)

/-- The main theorem stating the condition for equal division of weights -/
theorem equal_division_condition (n : ℕ) (h : n > 3) :
  (∃ (a b c : List ℕ), a ++ b ++ c = weights n ∧
    a.sum = b.sum ∧ b.sum = c.sum) ↔ satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_condition_l662_66266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l662_66251

-- Define the expression as noncomputable
noncomputable def expression : ℝ := ((12.983 * 26) / 200)^3 * Real.log 5 / Real.log 10

-- State the theorem
theorem expression_value : 
  ∃ ε > 0, |expression - 3.361| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l662_66251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l662_66210

-- Define the complex number z
noncomputable def z : ℂ := (2 + Complex.I) / (1 + Complex.I)^2

-- Theorem statement
theorem imaginary_part_of_z :
  Complex.im z = -1 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l662_66210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_turn_all_on_l662_66228

/-- Represents the state of a lightbulb (on or off) -/
inductive BulbState
| On : BulbState
| Off : BulbState

/-- Represents the configuration of all lightbulbs -/
def BulbConfiguration := (Fin 24 → BulbState) × BulbState

/-- Represents a single operation on the lightbulbs -/
inductive Operation
| ToggleOdd (i j : Fin 24) : Operation
| ToggleTriangle (i j k : Fin 24) : Operation

/-- Applies an operation to a bulb configuration -/
def applyOperation (config : BulbConfiguration) (op : Operation) : BulbConfiguration :=
  sorry

/-- Checks if all bulbs are on in a given configuration -/
def allOn (config : BulbConfiguration) : Prop :=
  sorry

/-- Theorem stating that any configuration can be transformed to all-on -/
theorem can_turn_all_on (initial : BulbConfiguration) :
  ∃ (ops : List Operation), allOn (ops.foldl applyOperation initial) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_turn_all_on_l662_66228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_distance_ratio_l662_66290

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the distance of a point from a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem to prove -/
theorem line_through_point_with_distance_ratio :
  let P : Point := ⟨2, -5⟩
  let A : Point := ⟨3, -2⟩
  let B : Point := ⟨-1, 6⟩
  let line1 : Line := ⟨1, 1, 3⟩
  let line2 : Line := ⟨17, 1, -29⟩
  (pointOnLine P line1 ∧ pointOnLine P line2) ∧
  (distancePointToLine A line1 / distancePointToLine B line1 = 1 / 2) ∧
  (distancePointToLine A line2 / distancePointToLine B line2 = 1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_distance_ratio_l662_66290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l662_66287

def is_root (z : ℂ) : Prop := z^2 + 2*z = 10 - 2*Complex.I

theorem roots_of_equation :
  let roots : List ℂ := [3 - Complex.I, 3 + Complex.I, -1 - 4*Complex.I, -5 - Complex.I, -5 + Complex.I, -1 + 4*Complex.I]
  (∀ z ∈ roots, is_root z) ∧ 
  (∀ w : ℂ, is_root w → w ∈ roots) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l662_66287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_to_sphere_volume_ratio_l662_66213

/-- Two spheres with equal radius, where the center of one is on the surface of the other -/
structure IntersectingSpheres where
  r : ℝ
  (r_pos : r > 0)

/-- Volume of the intersection of two spheres -/
noncomputable def intersection_volume (s : IntersectingSpheres) : ℝ := (5 / 12) * Real.pi * s.r^3

/-- Volume of one sphere -/
noncomputable def sphere_volume (s : IntersectingSpheres) : ℝ := (4 / 3) * Real.pi * s.r^3

/-- The ratio of the intersection volume to the volume of one sphere is 5/16 -/
theorem intersection_to_sphere_volume_ratio (s : IntersectingSpheres) :
  intersection_volume s / sphere_volume s = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_to_sphere_volume_ratio_l662_66213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_mod_period_l662_66217

/-- Sequence a_n defined recursively -/
noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2
  | 1 => 2
  | (n + 2) => a (n + 1) * (a n)^2

/-- Sequence of remainders when a_n is divided by 2014 -/
noncomputable def a_mod (n : ℕ) : ℕ := Int.natMod (Int.floor (a n)) 2014

/-- The minimal period of the sequence a_mod -/
def minimal_period : ℕ := 36

/-- Theorem stating that the minimal period of a_mod is 36 -/
theorem a_mod_period :
  ∃ (N : ℕ), ∀ (m : ℕ), m ≥ N → 
    (∀ (k : ℕ), a_mod (m + k * minimal_period) = a_mod m) ∧ 
    (∀ (p : ℕ), p < minimal_period → 
      ∃ (n : ℕ), n ≥ N ∧ a_mod (n + p) ≠ a_mod n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_mod_period_l662_66217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l662_66271

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  directrix : ℝ → ℝ × ℝ
  eq : (x : ℝ) → (y : ℝ) → Prop

/-- Point on a parabola -/
def PointOnParabola (para : Parabola) (point : ℝ × ℝ) : Prop :=
  para.eq point.1 point.2

/-- Perpendicular lines -/
def Perpendicular (l1 l2 : ℝ → ℝ × ℝ) : Prop := sorry

/-- Intersection of line and parabola -/
def Intersects (para : Parabola) (l : ℝ → ℝ × ℝ) (point : ℝ × ℝ) : Prop := sorry

/-- Main theorem -/
theorem parabola_property (para : Parabola) (A B M : ℝ × ℝ) (l : ℝ → ℝ × ℝ) :
  para.p > 0 →
  A = (0, 2) →
  para.eq = (fun x y => y^2 = 2 * para.p * x) →
  Intersects para (fun t => ((1 - t) * para.focus.1 + t * A.1,
                             (1 - t) * para.focus.2 + t * A.2)) B →
  Perpendicular (fun t => (B.1, t)) para.directrix →
  M ∈ Set.range para.directrix →
  Perpendicular (fun t => ((1 - t) * A.1 + t * M.1,
                           (1 - t) * A.2 + t * M.2))
                (fun t => ((1 - t) * M.1 + t * para.focus.1,
                           (1 - t) * M.2 + t * para.focus.2)) →
  para.p = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l662_66271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l662_66293

-- Define our own unit of measurement (renamed to avoid conflict with built-in Unit)
inductive MeasurementUnit
  | Gram
  | Centimeter
  | Other

-- Define a measurement as a pair of a number and a unit
structure Measurement where
  value : ℝ
  unit : MeasurementUnit

-- Define the egg's weight
def eggWeight : Measurement := ⟨60, MeasurementUnit.Gram⟩

-- Define Xiaoqiang's height
def xiaoqiangHeight : Measurement := ⟨142, MeasurementUnit.Centimeter⟩

-- Theorem to prove the appropriate units
theorem appropriate_units :
  eggWeight.unit = MeasurementUnit.Gram ∧ xiaoqiangHeight.unit = MeasurementUnit.Centimeter := by
  apply And.intro
  · rfl
  · rfl

#check appropriate_units

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l662_66293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_complex_equations_l662_66232

noncomputable def parallelogram_area (v1 v2 v3 v4 : ℂ) : ℝ :=
  Complex.abs (v1.im * v3.re - v1.re * v3.im + v2.im * v4.re - v2.re * v4.im)

theorem parallelogram_area_complex_equations : ∃ (v1 v2 v3 v4 : ℂ),
  (v1^2 = 5 + 5 * Complex.I * Real.sqrt 7 ∧
   v2^2 = 5 + 5 * Complex.I * Real.sqrt 7 ∧
   v3^2 = 3 + 3 * Complex.I * Real.sqrt 2 ∧
   v4^2 = 3 + 3 * Complex.I * Real.sqrt 2) →
  (parallelogram_area v1 v2 v3 v4 = 5 * Real.sqrt 7 - 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_complex_equations_l662_66232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_on_board_l662_66236

theorem max_numbers_on_board : 
  ∃ (n : ℕ) (S : Finset ℝ), 
    (∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → x ≠ z → y ≠ z → x^2 > y*z) ∧ 
    (Finset.card S = n) ∧
    (∀ (T : Finset ℝ), (∀ x y z, x ∈ T → y ∈ T → z ∈ T → x ≠ y → x ≠ z → y ≠ z → x^2 > y*z) → Finset.card T ≤ n) ∧
    n = 3 :=
by sorry

#check max_numbers_on_board

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_on_board_l662_66236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_neg_two_l662_66258

theorem complex_expression_equals_neg_two :
  |1 - Real.sqrt 3| - Real.tan (60 * π / 180) + (π - 2023)^(0:ℝ) + (-1/2)^(-1:ℝ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_neg_two_l662_66258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l662_66298

/-- Given a hyperbola and a circle with specific properties, prove that the eccentricity of the hyperbola is 3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := λ (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let circle := λ (x y : ℝ) => (x - 3)^2 + y^2 = 9
  let asymptote := λ (x y : ℝ) => b * x - a * y = 0
  ∃ (A B : ℝ × ℝ),
    (asymptote A.1 A.2 ∧ circle A.1 A.2) ∧
    (asymptote B.1 B.2 ∧ circle B.1 B.2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 →
  let c := Real.sqrt (a^2 + b^2)
  c / a = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l662_66298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l662_66274

-- Define the necessary constants and functions
noncomputable def tan30 : ℝ := Real.sqrt 3 / 3
noncomputable def inv_quarter : ℝ := (1/4)⁻¹
noncomputable def sqrt12 : ℝ := Real.sqrt 12
noncomputable def abs_neg_sqrt3 : ℝ := |-(Real.sqrt 3)|

-- State the theorem
theorem calculate_expression : 
  3 * tan30 - inv_quarter - sqrt12 + abs_neg_sqrt3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l662_66274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_amount_in_punch_l662_66261

theorem water_amount_in_punch (water_parts juice_parts : ℕ) 
  (total_gallons : ℚ) (quarts_per_gallon : ℕ) 
  (h1 : water_parts = 5)
  (h2 : juice_parts = 3)
  (h3 : total_gallons = 2)
  (h4 : quarts_per_gallon = 4)
  : (water_parts : ℚ) / (water_parts + juice_parts : ℚ) * 
    (total_gallons * quarts_per_gallon) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_amount_in_punch_l662_66261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_intervals_l662_66265

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x - Real.pi / 3)

theorem f_monotone_increasing_intervals :
  let domain := Set.Icc (-Real.pi) 0
  ∃ (I₁ I₂ : Set ℝ),
    I₁ = Set.Icc (-Real.pi) (-7 * Real.pi / 12) ∧
    I₂ = Set.Icc (-Real.pi / 12) 0 ∧
    I₁ ⊆ domain ∧
    I₂ ⊆ domain ∧
    StrictMonoOn f I₁ ∧
    StrictMonoOn f I₂ ∧
    ∀ (I : Set ℝ), I ⊆ domain → StrictMonoOn f I →
      I ⊆ I₁ ∨ I ⊆ I₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_intervals_l662_66265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_l662_66239

theorem function_extrema (x y : ℝ) (h1 : x ≠ y) (h2 : x ≥ 0) (h3 : y ≥ 0) 
  (h4 : (x^2 + y^2) / (x + y) ≤ 4) :
  ∃ (z : ℝ), y - 2*x = z ∧ -2 - 2*Real.sqrt 10 ≤ z ∧ z ≤ -2 + 2*Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_l662_66239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diff_invariant_specific_case_l662_66263

/-- Represents a triplet of real numbers -/
structure Triplet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Performs one transformation on a triplet -/
def transform (t : Triplet) : Triplet :=
  { a := t.b + t.c
  , b := t.a + t.c
  , c := t.a + t.b }

/-- Returns the difference between the largest and smallest numbers in a triplet -/
noncomputable def diffMaxMin (t : Triplet) : ℝ :=
  max t.a (max t.b t.c) - min t.a (min t.b t.c)

/-- Theorem: The difference between the largest and smallest numbers 
    remains constant after any number of transformations -/
theorem diff_invariant (t : Triplet) (n : ℕ) : 
  diffMaxMin t = diffMaxMin (Nat.iterate transform n t) := by
  sorry

/-- Corollary: For the specific initial triplet (20, 1, 6), 
    the difference after 2016 transformations is 19 -/
theorem specific_case : 
  diffMaxMin (Nat.iterate transform 2016 { a := 20, b := 1, c := 6 }) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diff_invariant_specific_case_l662_66263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_calculation_l662_66249

theorem incorrect_calculation : Real.sqrt (2/3) / Real.sqrt (3/2) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_calculation_l662_66249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l662_66262

noncomputable def f (m : ℝ) (x : ℝ) := x^2 - 2*m*x + 1

def has_two_zeros (m : ℝ) : Prop := ∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0

def condition_q (m : ℝ) : Prop := ∃ x > 0, x^2 - 2*Real.exp 1*(Real.log x) ≤ m

theorem range_of_m :
  (∀ m : ℝ, (has_two_zeros m ∨ condition_q m) ∧ ¬(has_two_zeros m ∧ condition_q m)) →
  (∀ m : ℝ, m ∈ Set.Iic (-1) ∪ Set.Icc 0 1) ∧ 
  (∀ m : ℝ, m ∈ Set.Iic (-1) ∪ Set.Icc 0 1 → (has_two_zeros m ∨ condition_q m) ∧ ¬(has_two_zeros m ∧ condition_q m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l662_66262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l662_66225

theorem problem_solution : 
  ((-2.8) - (-3.6) + (-1.5) - 3.6 = -4.3) ∧ 
  ((-5/6 + 1/3 - 3/4) * (-24) = 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l662_66225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_sum_l662_66223

/-- Represents a cube with numbers on its faces -/
structure Cube where
  a1 : ℕ
  a2 : ℕ
  b1 : ℕ
  b2 : ℕ
  c1 : ℕ
  c2 : ℕ

/-- The sum of products of adjacent face numbers for each vertex -/
def vertexProductSum (cube : Cube) : ℕ :=
  cube.a1 * cube.b1 * cube.c1 +
  cube.a1 * cube.b1 * cube.c2 +
  cube.a1 * cube.b2 * cube.c1 +
  cube.a1 * cube.b2 * cube.c2 +
  cube.a2 * cube.b1 * cube.c1 +
  cube.a2 * cube.b1 * cube.c2 +
  cube.a2 * cube.b2 * cube.c1 +
  cube.a2 * cube.b2 * cube.c2

/-- The sum of all numbers on the cube's faces -/
def faceSum (cube : Cube) : ℕ :=
  cube.a1 + cube.a2 + cube.b1 + cube.b2 + cube.c1 + cube.c2

/-- Theorem: If the sum of products of adjacent face numbers for each vertex is 1001,
    then the sum of all numbers on the cube's faces is 31 -/
theorem cube_face_sum (cube : Cube) :
  vertexProductSum cube = 1001 → faceSum cube = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_sum_l662_66223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_vector_l662_66268

/-- Given vectors c and d, prove that the unit vector u satisfies the condition
that d bisects the angle between c and 3u. -/
theorem bisecting_vector (c d u : ℝ × ℝ × ℝ) : 
  c = (4, -3, 0) →
  d = (2, 0, 2) →
  u = (-16/25, 16/25, 1) →
  ∃ (k : ℝ), d = k • ((c + 3 • u) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_vector_l662_66268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trinomial_expansion_fourth_power_l662_66233

/-- The number of terms in the expansion of (x+y+z)^n -/
def trinomial_expansion_terms (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem trinomial_expansion_fourth_power :
  trinomial_expansion_terms 4 = 15 := by
  rfl

#eval trinomial_expansion_terms 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trinomial_expansion_fourth_power_l662_66233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_differential_equation_l662_66257

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := -x * Real.cos x + 3 * x

-- State the theorem
theorem y_satisfies_differential_equation :
  ∀ x : ℝ, x * (deriv y x) = y x + x^2 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_differential_equation_l662_66257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_reciprocal_of_24_and_182_l662_66259

theorem lcm_reciprocal_of_24_and_182 :
  (1 : ℚ) / (Nat.lcm 24 182) = 1 / 2184 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_reciprocal_of_24_and_182_l662_66259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_count_less_than_100_l662_66269

theorem odd_divisors_count_less_than_100 : 
  (Finset.filter (fun n : ℕ => n < 100 ∧ Nat.card (Nat.divisors n) % 2 = 1) (Finset.range 100)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_count_less_than_100_l662_66269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_sum_l662_66277

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 4*x + 3
def g (x : ℝ) : ℝ := -f x
def h (x : ℝ) : ℝ := f (-x)

-- Define the number of intersections
def a : ℕ := 2  -- number of intersections between f and g
def b : ℕ := 0  -- number of intersections between f and h

-- Theorem statement
theorem intersection_and_sum : 
  (∃ S : Finset ℝ, S.card = a ∧ ∀ x ∈ S, f x = g x) ∧
  (∃ T : Finset ℝ, T.card = b ∧ ∀ x ∈ T, f x = h x) ∧
  7 * a + 2 * b = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_sum_l662_66277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_over_six_l662_66272

theorem tan_a_pi_over_six (a : ℝ) (h : (2 : ℝ)^a = 4) : Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_over_six_l662_66272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_universal_polynomial_with_n_integer_roots_l662_66216

theorem no_universal_polynomial_with_n_integer_roots :
  ¬ ∃ (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0),
    ∀ (n : ℕ) (h_n : n > 3),
      ∃ (P : Polynomial ℝ),
        (P.leadingCoeff = 1) ∧
        (P.natDegree = n) ∧
        (P.coeff 2 = a) ∧
        (P.coeff 1 = b) ∧
        (P.coeff 0 = c) ∧
        (∃ (roots : Finset ℤ), roots.card = n ∧ ∀ x : ℤ, x ∈ roots → P.eval (↑x : ℝ) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_universal_polynomial_with_n_integer_roots_l662_66216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_l662_66283

-- Define the four functions
noncomputable def g₁ (x : ℝ) : ℝ := Real.cos x
noncomputable def g₂ (x : ℝ) : ℝ := if x ≥ 0 then -x^2 else x^2
def g₃ (x : ℝ) : ℝ := x^3 - x
noncomputable def g₄ (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- Statement of the theorem
theorem function_existence :
  (¬ ∃ f : ℝ → ℝ, ∀ x, f (g₁ x) = x) ∧
  (¬ ∃ f : ℝ → ℝ, ∀ x, f (g₃ x) = x) ∧
  (∃ f : ℝ → ℝ, ∀ x, f (g₂ x) = x) ∧
  (∃ f : ℝ → ℝ, ∀ x, f (g₄ x) = x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_l662_66283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_percentage_approx_33_percent_l662_66211

/-- Represents a school with a total number of students and a percentage of students who prefer badminton -/
structure School where
  total_students : ℕ
  badminton_percentage : ℚ

/-- Calculates the overall percentage of students who prefer badminton in two schools combined -/
def overall_badminton_percentage (school1 school2 : School) : ℚ :=
  let total_badminton_students := (school1.total_students : ℚ) * school1.badminton_percentage +
                                  (school2.total_students : ℚ) * school2.badminton_percentage
  let total_students := (school1.total_students + school2.total_students : ℚ)
  total_badminton_students / total_students

theorem badminton_percentage_approx_33_percent : 
  let north : School := { total_students := 1500, badminton_percentage := 30/100 }
  let south : School := { total_students := 1800, badminton_percentage := 35/100 }
  abs (overall_badminton_percentage north south - 33/100) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_percentage_approx_33_percent_l662_66211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_candies_remaining_l662_66292

/-- Represents the initial number of candies of each color -/
structure CandyBowl where
  red : ℝ
  orange : ℝ
  green : ℝ
  blue : ℝ
  yellow : ℝ

/-- Calculates the total number of candies -/
noncomputable def totalCandies (bowl : CandyBowl) : ℝ :=
  bowl.red + bowl.orange + bowl.green + bowl.blue + bowl.yellow

/-- Represents the remaining candies after consumption -/
noncomputable def remainingCandies (bowl : CandyBowl) : ℝ :=
  0.4 * bowl.red + 0.4875 * bowl.orange + 0.1 * bowl.blue + (2/3) * bowl.yellow

/-- The main theorem stating that 40% of red candies remain -/
theorem red_candies_remaining (bowl : CandyBowl) :
  remainingCandies bowl = 0.1 * totalCandies bowl →
  0.4 * bowl.red / bowl.red = 0.4 := by
  sorry

#check red_candies_remaining

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_candies_remaining_l662_66292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_midpoint_l662_66275

/-- The polar curve defined by ρ² - 10ρcosθ + 4 = 0 -/
def polar_curve (ρ θ : ℝ) : Prop := ρ^2 - 10*ρ*(Real.cos θ) + 4 = 0

/-- The line defined by θ = π/3 -/
def polar_line (θ : ℝ) : Prop := θ = Real.pi/3

/-- The midpoint of two points in polar coordinates -/
noncomputable def polar_midpoint (ρ₁ θ₁ ρ₂ θ₂ : ℝ) : ℝ × ℝ := 
  ((ρ₁ + ρ₂) / 2, (θ₁ + θ₂) / 2)

theorem intersection_midpoint :
  ∀ ρ₁ ρ₂ : ℝ, 
    polar_curve ρ₁ (Real.pi/3) → 
    polar_curve ρ₂ (Real.pi/3) → 
    ρ₁ ≠ ρ₂ →
    polar_midpoint ρ₁ (Real.pi/3) ρ₂ (Real.pi/3) = (5/2, Real.pi/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_midpoint_l662_66275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_l662_66220

/-- Conversion from spherical coordinates to rectangular coordinates --/
theorem spherical_to_rectangular (ρ θ φ : Real) (hρ : ρ = 3) (hθ : θ = π/2) (hφ : φ = π/3) :
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) = (0, 3 * Real.sqrt 3 / 2, 3 / 2) := by
  sorry

#check spherical_to_rectangular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_l662_66220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l662_66241

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the inclination angle of a line
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m * (180 / Real.pi)

-- Theorem statement
theorem line_inclination_angle :
  inclination_angle 1 = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l662_66241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_factorization_l662_66253

/-- The product of positive integers from 1 to 8 -/
def x : ℕ := (List.range 8).map (λ x => x + 1) |>.prod

/-- The prime factorization of x -/
theorem x_factorization (i k m p : ℕ+) :
  x = 2^(i.val) * 3^(k.val) * 5^(m.val) * 7^(p.val) ∧ i.val + k.val + m.val + p.val = 11 →
  m.val = 1 := by
  sorry

#eval x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_factorization_l662_66253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l662_66255

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 - 3*x - 1/x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 3 - 2 * Real.sqrt 3 ∧
  (∀ x : ℝ, x > 0 → f x ≤ M) ∧
  (∃ x : ℝ, x > 0 ∧ f x = M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l662_66255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sin_cos_l662_66250

theorem max_distance_sin_cos : 
  (∀ a : ℝ, dist (a, Real.sin a) (a, Real.cos a) ≤ Real.sqrt 2) ∧ 
  (∃ a₀ : ℝ, dist (a₀, Real.sin a₀) (a₀, Real.cos a₀) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sin_cos_l662_66250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l662_66221

/-- Represents the annual interest rate as a real number between 0 and 1 -/
noncomputable def annual_interest_rate : ℝ := 0.04

/-- Represents the investment period in years -/
noncomputable def investment_period : ℝ := 3 / 12

/-- Represents the final amount after interest is credited -/
noncomputable def final_amount : ℝ := 10204

/-- Represents the original investment amount -/
noncomputable def original_amount : ℝ := 10104

/-- Proves that the original investment amount is correct given the conditions -/
theorem investment_calculation :
  final_amount = original_amount * (1 + annual_interest_rate * investment_period) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l662_66221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l662_66234

/-- A square with side length 10 feet -/
structure Square :=
  (side : ℝ)
  (is_ten : side = 10)

/-- A circle passing through opposite vertices of the square and tangent to one side -/
structure CircleInSquare (s : Square) :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (passes_through_vertices : Prop) -- Changed to Prop
  (tangent_to_side : Prop) -- Changed to Prop

/-- The radius of the circle is 7.5 feet -/
theorem circle_radius (s : Square) (c : CircleInSquare s) : c.radius = 7.5 := by
  sorry

/-- Helper function to check if a circle passes through opposite vertices of a square -/
def passes_through_opposite_vertices (center : ℝ × ℝ) (radius : ℝ) (s : Square) : Prop :=
  sorry -- Implementation details omitted

/-- Helper function to check if a circle is tangent to a side of a square -/
def tangent_to_side (center : ℝ × ℝ) (radius : ℝ) (s : Square) : Prop :=
  sorry -- Implementation details omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l662_66234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l662_66276

/-- The x-coordinate of the focus with the larger x-coordinate for the hyperbola
    (x-1)²/9² - (y+2)²/16² = 1 -/
noncomputable def focus_x : ℝ := 1 + Real.sqrt 337

/-- The y-coordinate of the focus with the larger x-coordinate for the hyperbola
    (x-1)²/9² - (y+2)²/16² = 1 -/
def focus_y : ℝ := -2

/-- The equation of the hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop :=
  (x - 1)^2 / 9^2 - (y + 2)^2 / 16^2 = 1

theorem focus_coordinates :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ 
  x ≥ 1 ∧ 
  (∀ (x' y' : ℝ), hyperbola_eq x' y' → x' ≤ x) ∧
  x = focus_x ∧ 
  y = focus_y := by
  sorry

#check focus_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l662_66276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_given_range_l662_66219

-- Define the function f(x) = 1 / (x - 1)
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

-- Define the range of f
def range_f : Set ℝ := Set.Iio (-1) ∪ Set.Ioi 1

-- Define the domain we want to prove
def domain_f : Set ℝ := Set.Ioo 0 1 ∪ Set.Ioo 1 2

-- Theorem statement
theorem domain_of_f_given_range :
  (∀ y ∈ range_f, ∃ x, f x = y) →
  (∀ x, f x ∈ range_f ↔ x ∈ domain_f) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_given_range_l662_66219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_less_than_one_l662_66279

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 4*x else 4*x - x^2

-- State the theorem
theorem f_inequality_implies_a_less_than_one (a : ℝ) :
  f (2 - a) > f a → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_less_than_one_l662_66279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l662_66264

/-- Represents the daily profit function for a company selling a product -/
def daily_profit (cost selling_price initial_price initial_quantity price_step quantity_step : ℝ) : ℝ → ℝ :=
  λ x ↦ (x - cost) * (initial_quantity - (x - initial_price) * quantity_step)

theorem profit_maximization (cost initial_price initial_quantity price_step quantity_step : ℝ) :
  cost = 8 ∧ 
  initial_price = 10 ∧ 
  initial_quantity = 100 ∧ 
  price_step = 1 ∧ 
  quantity_step = 10 →
  (let f := daily_profit cost initial_price initial_price initial_quantity price_step quantity_step
   let profit_at_13 := f 13
   let max_price := 14
   let max_profit := f max_price
   profit_at_13 = 350 ∧ max_profit = 360 ∧ 
   ∀ x, 10 ≤ x ∧ x ≤ 20 → f x ≤ max_profit) :=
by
  sorry

#check profit_maximization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l662_66264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_image_classification_l662_66202

-- Define the type for the image of p(ℝ²)
inductive ImageType : Type
  | singleton (k : ℝ) : ImageType  -- [k, k]
  | rightInfinite (k : ℝ) : ImageType  -- [k, ∞)
  | leftInfinite (k : ℝ) : ImageType  -- (-∞, k]
  | allReals : ImageType  -- (-∞, ∞)
  | rightOpen (k : ℝ) : ImageType  -- (k, ∞)
  | leftOpen (k : ℝ) : ImageType  -- (-∞, k)

-- Define the polynomial function type
def PolynomialFunction := (ℝ × ℝ) → ℝ

-- Statement of the theorem
theorem polynomial_image_classification (p : PolynomialFunction) :
  ∃ (img : ImageType), 
    (∀ y : ℝ, y ∈ Set.range p ↔ 
      (∃ k, img = ImageType.singleton k ∧ y = k) ∨
      (∃ k, img = ImageType.rightInfinite k ∧ y ≥ k) ∨
      (∃ k, img = ImageType.leftInfinite k ∧ y ≤ k) ∨
      (img = ImageType.allReals) ∨
      (∃ k, img = ImageType.rightOpen k ∧ y > k) ∨
      (∃ k, img = ImageType.leftOpen k ∧ y < k)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_image_classification_l662_66202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_permutations_proof_l662_66214

/-- The sum of all numbers obtained by permutations of the digits 1234567 -/
def sum_of_permutations : ℕ := 22399997760

/-- The set of digits in the original number -/
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The number of digits -/
def n : ℕ := Finset.card digits

/-- The factorial of the number of digits -/
def factorial_n : ℕ := Nat.factorial n

/-- Theorem stating that the sum of all permutations of the digits 1234567 is 22399997760 -/
theorem sum_of_permutations_proof :
  (Finset.sum (Finset.powerset digits) (λ s => if Finset.card s = n then
    Finset.sum (Finset.powerset s) (λ p => if Finset.card p = n then
      Finset.sum (Finset.range n) (λ i => (Finset.toList p).get ⟨i, sorry⟩ * 10^i)
    else
      0)
  else
    0)) = sum_of_permutations := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_permutations_proof_l662_66214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l662_66201

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to avoid missing cases error
  | 1 => 1
  | (n + 2) => (1 / 16) * (1 + 4 * a (n + 1) + Real.sqrt (1 + 24 * a (n + 1)))

theorem a_general_term (n : ℕ) (h : n ≥ 1) : 
  a n = 1/3 + (1/2)^n + (1/3) * (1/2)^(2*n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l662_66201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l662_66252

/-- A circle centered at the origin with radius 2 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- A line with equation ax + by = 4 -/
def Line (a b : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 = 4}

/-- The distance between a point and the origin -/
noncomputable def distanceFromOrigin (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

/-- The distance between a line ax + by = 4 and the origin -/
noncomputable def distanceLineOrigin (a b : ℝ) : ℝ := 4 / Real.sqrt (a^2 + b^2)

theorem line_intersects_circle (a b : ℝ) :
  distanceFromOrigin (a, b) > 2 →
  (∃ p : ℝ × ℝ, p ∈ Circle ∧ p ∈ Line a b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l662_66252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_required_l662_66215

-- Define the constants from the problem
def yard_area : ℝ := 240
def unfenced_side : ℝ := 40
def garden_length : ℝ := 10
def garden_width : ℝ := 8
def tree_diameter : ℝ := 4

-- Define pi as a constant (approximation)
def π : ℝ := 3.14

-- Define the theorem
theorem fencing_required : ℝ :=
  let garden_perimeter := 2 * (garden_length + garden_width)
  let tree_circumference := π * tree_diameter
  garden_perimeter + tree_circumference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_required_l662_66215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_l662_66235

theorem product_of_roots_quadratic : 
  let f : ℝ → ℝ := fun x ↦ x^2 + 3*x - 5
  let roots := {x : ℝ | f x = 0}
  ∀ x y, x ∈ roots → y ∈ roots → x * y = -5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_l662_66235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_length_l662_66200

/-- The length of the other diagonal of a rhombus -/
noncomputable def other_diagonal (d1 : ℝ) (area : ℝ) : ℝ :=
  (2 * area) / d1

/-- Theorem: The other diagonal of a rhombus with one diagonal 16 cm and area 160 cm² is 20 cm -/
theorem rhombus_diagonal_length :
  let d1 : ℝ := 16
  let area : ℝ := 160
  other_diagonal d1 area = 20 := by
  -- Unfold the definition of other_diagonal
  unfold other_diagonal
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_length_l662_66200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_18_7_l662_66295

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 6) / (2 * x^2 - 5 * x + 2)

/-- The horizontal asymptote of g(x) -/
noncomputable def horizontal_asymptote : ℝ := 3 / 2

theorem g_crosses_asymptote_at_18_7 :
  ∃ (x : ℝ), x = 18 / 7 ∧ g x = horizontal_asymptote := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_at_18_7_l662_66295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_derivative_positive_l662_66208

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - a * x^2 - Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -2 * f a x - (2 * a + 1) * x^2 + a * x

-- Define the derivative of g
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 2 / x - 2 * x - a

-- State the theorem
theorem g_derivative_positive (a : ℝ) (x₁ x₂ : ℝ) :
  g a x₁ = 0 → g a x₂ = 0 → 0 < x₁ → x₁ < x₂ → x₂ < 4 * x₁ →
  g' a ((2 * x₁ + x₂) / 3) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_derivative_positive_l662_66208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_on_unit_circle_l662_66203

theorem dot_product_on_unit_circle (x₁ y₁ x₂ y₂ θ : ℝ) :
  x₁^2 + y₁^2 = 1 →
  x₂^2 + y₂^2 = 1 →
  θ > π/2 →
  θ < π →
  Real.sin (θ + π/4) = 3/5 →
  x₁*x₂ + y₁*y₂ = -Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_on_unit_circle_l662_66203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l662_66247

theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) (lambda : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  c - b = 2*b*Real.cos A →
  (∀ (A B C : ℝ), 0 < A ∧ A < π/2 → 0 < B ∧ B < π/2 → 0 < C ∧ C < π/2 → 
    A + B + C = π → lambda * Real.sin A - Real.cos (C - B) < 2) →
  lambda ≤ 5 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l662_66247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_AB_l662_66294

/-- Curve C defined by parametric equations -/
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sin α + Real.cos α, Real.sin α - Real.cos α)

/-- Line l defined in polar coordinates -/
noncomputable def line_l (ρ θ : ℝ) : Prop := Real.sqrt 2 * ρ * Real.sin (Real.pi / 4 - θ) + 1 / 2 = 0

/-- Theorem stating the length of chord AB -/
theorem chord_length_AB : 
  ∃ (A B : ℝ × ℝ) (α₁ α₂ ρ₁ ρ₂ θ₁ θ₂ : ℝ),
    curve_C α₁ = A ∧ 
    curve_C α₂ = B ∧
    line_l ρ₁ θ₁ ∧
    line_l ρ₂ θ₂ ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 30 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_AB_l662_66294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_transformation_l662_66289

theorem matrix_vector_transformation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (a b : Fin 2 → ℝ) 
  (h1 : N.vecMul a = ![1, 2]) 
  (h2 : N.vecMul b = ![3, -1]) : 
  N.vecMul (2 • a - b) = ![-1, 5] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_transformation_l662_66289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l662_66205

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.arccos x)^3 + (Real.arcsin x)^3

-- State the theorem
theorem range_of_f :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  ∃ y ∈ Set.Icc ((π^3) / 32) ((7 * π^3) / 8),
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc ((π^3) / 32) ((7 * π^3) / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l662_66205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l662_66243

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the circle
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * Real.sqrt 2 * p.1 = 0}

-- Define the line passing through P(0, 1)
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

-- Main theorem
theorem ellipse_and_triangle_area :
  ∃ (a b : ℝ),
    -- Ellipse conditions
    (a^2 = 4 ∧ b^2 = 2) ∧
    -- One focus coincides with the center of the circle
    (Real.sqrt 2, 0) ∈ Circle ∧
    -- Ellipse passes through (√2, 1)
    (Real.sqrt 2, 1) ∈ Ellipse a b ∧
    -- Line intersects ellipse at A and B
    ∃ (k : ℝ) (A B : ℝ × ℝ),
      A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧
      A ∈ Line k ∧ B ∈ Line k ∧
      -- AP = 2PB condition
      (A.1 - 0)^2 + (A.2 - 1)^2 = 4 * ((0 - B.1)^2 + (1 - B.2)^2) ∧
      -- Area of triangle AOB
      (1/2 * abs (A.1 * B.2 - B.1 * A.2) = (3 * Real.sqrt 14) / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l662_66243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_l662_66299

theorem least_subtraction_for_divisibility (n a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ (k : ℕ), k ≤ n ∧
  (n - k) % a = 0 ∧
  (n - k) % b = 0 ∧
  (n - k) % c = 0 ∧
  ∀ (m : ℕ), m < k →
    (n - m) % a ≠ 0 ∨
    (n - m) % b ≠ 0 ∨
    (n - m) % c ≠ 0 :=
by
  sorry

def compute_least_subtraction (n a b c : ℕ) : ℕ :=
  let lcm := Nat.lcm (Nat.lcm a b) c
  n % lcm

#eval compute_least_subtraction 3830 3 7 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_l662_66299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sided_faces_exist_l662_66270

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ
  sides_ge_three : sides ≥ 3

/-- A convex polyhedron with a finite number of faces -/
structure ConvexPolyhedron where
  faces : Finset Face
  convex : ∀ f ∈ faces, f.sides ≥ 3 -- Simplified convexity condition
  nonempty : faces.Nonempty

/-- The theorem stating that in any convex polyhedron, there exist two faces with an equal number of sides -/
theorem equal_sided_faces_exist (P : ConvexPolyhedron) : 
  ∃ f₁ f₂ : Face, f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.sides = f₂.sides := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sided_faces_exist_l662_66270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_and_initial_conditions_l662_66240

noncomputable def x (t : ℝ) : ℝ := t + Real.exp (-t)
noncomputable def y (t : ℝ) : ℝ := Real.exp t

theorem solution_satisfies_system_and_initial_conditions :
  (∀ t, deriv x t = 1 - 1 / (y t)) ∧
  (∀ t, deriv y t = 1 / (x t - t)) ∧
  x 0 = 1 ∧
  y 0 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_and_initial_conditions_l662_66240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l662_66238

/-- The parabola defined by y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola y^2 = 4x -/
def FocusF : ℝ × ℝ := (1, 0)

/-- Point P inside the parabola -/
def P : ℝ × ℝ := (3, 2)

/-- Assertion that P is inside the parabola -/
axiom P_inside : P.2^2 < 4 * P.1

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_sum :
  ∃ (c : ℝ), c = 4 ∧
  ∀ (M : ℝ × ℝ), M ∈ Parabola →
    c ≤ distance M P + distance M FocusF ∧
    ∃ (M₀ : ℝ × ℝ), M₀ ∈ Parabola ∧
      c = distance M₀ P + distance M₀ FocusF := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l662_66238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l662_66237

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, and β are non-zero real numbers,
    if f(1988) = 3, then f(2013) = 5. -/
theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f := λ x : ℝ ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  (f 1988 = 3) → (f 2013 = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l662_66237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l662_66231

/-- A triangle with unequal sides and a specified area -/
structure Triangle where
  area : ℝ
  unequal_sides : True

/-- The medians of the triangle -/
structure Medians (t : Triangle) where
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ

/-- The theorem statement -/
theorem third_median_length (t : Triangle) (m : Medians t) (h₁ : m.m₁ = 5) (h₂ : m.m₂ = 7)
  (h_area : t.area = 4 * Real.sqrt 21) :
  m.m₃ = 3 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l662_66231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PQRS_equals_one_l662_66244

noncomputable def P : ℝ := Real.sqrt 2010 + Real.sqrt 2009 + Real.sqrt 2008
noncomputable def Q : ℝ := -Real.sqrt 2010 - Real.sqrt 2009 + Real.sqrt 2008
noncomputable def R : ℝ := Real.sqrt 2010 - Real.sqrt 2009 - Real.sqrt 2008
noncomputable def S : ℝ := -Real.sqrt 2010 + Real.sqrt 2009 - Real.sqrt 2008

theorem PQRS_equals_one : P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PQRS_equals_one_l662_66244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_solve_mixture_volume_l662_66224

theorem mixture_volume (initial_water_percentage : Real) 
                       (final_water_percentage : Real) 
                       (added_water : Real) 
                       (mixture_volume : Real) : Prop :=
  initial_water_percentage = 0.20 ∧
  final_water_percentage = 0.25 ∧
  added_water = 12 ∧
  (initial_water_percentage * mixture_volume + added_water) / 
    (mixture_volume + added_water) = final_water_percentage

theorem solve_mixture_volume : ∃ (v : Real), mixture_volume 0.20 0.25 12 v := by
  sorry

#check mixture_volume
#check solve_mixture_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_solve_mixture_volume_l662_66224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_escalator_walk_time_l662_66297

/-- Represents the time taken for a child to walk up a running escalator -/
noncomputable def time_to_walk_up_escalator (escalator_length : ℝ) (time_to_walk_stopped : ℝ) (time_escalator_running : ℝ) : ℝ :=
  let child_speed := escalator_length / time_to_walk_stopped
  let escalator_speed := escalator_length / time_escalator_running
  let effective_speed := child_speed + escalator_speed
  escalator_length / effective_speed

/-- Theorem stating the time taken for a child to walk up a specific running escalator -/
theorem specific_escalator_walk_time :
  time_to_walk_up_escalator 60 90 60 = 36 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_escalator_walk_time_l662_66297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l662_66222

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculates the inclination angle of a line in degrees -/
noncomputable def inclinationAngle (l : Line) : ℝ :=
  (Real.arctan l.slope) * (180 / Real.pi)

/-- Calculates the x-intercept of a line -/
noncomputable def xIntercept (l : Line) : ℝ :=
  -l.intercept / l.slope

/-- The main theorem -/
theorem line_equation_proof (l : Line) :
  l.slope = -1 ∧ l.intercept = 2 →
  inclinationAngle l = 135 ∧ xIntercept l = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l662_66222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_problem_l662_66204

theorem integer_sum_problem : ∃ a b : ℕ, 
  (a * b + a + b = 254) ∧ 
  (Nat.gcd a b = 1) ∧ 
  (a < 30 ∧ b < 30) ∧ 
  (a + b = 30) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_problem_l662_66204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_BM_equation_l662_66227

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_equation (x₁ y₁ x₂ y₂ : ℚ) (x y : ℚ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- The midpoint of two points (x₁, y₁) and (x₂, y₂) -/
def midpoint_coords (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

theorem line_BM_equation :
  let A : ℚ × ℚ := (1, 2)
  let B : ℚ × ℚ := (4, 1)
  let C : ℚ × ℚ := (3, 6)
  let M := midpoint_coords A.1 A.2 C.1 C.2
  ∀ x y : ℚ, line_equation B.1 B.2 M.1 M.2 x y ↔ (3 : ℚ) / 7 * x - (2 : ℚ) / 7 * y + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_BM_equation_l662_66227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l662_66246

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = 1

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.fst A.snd

-- Define a line tangent to the circle
def line_tangent_to_circle (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), circle_M ((1 - t) * A.fst + t * B.fst) ((1 - t) * A.snd + t * B.snd)

theorem parabola_circle_tangency :
  (∀ x y, line_l x → parabola_C x y → ∃ P Q : ℝ × ℝ, P ≠ Q ∧ parabola_C P.fst P.snd ∧ parabola_C Q.fst Q.snd) →
  (∃ P Q : ℝ × ℝ, parabola_C P.fst P.snd ∧ parabola_C Q.fst Q.snd ∧ P.fst = Q.fst ∧ P.fst = 1 ∧ P.snd = -Q.snd) →
  (∀ x y, line_l x → circle_M x y → x = 1) →
  (∀ A₁ A₂ A₃ : ℝ × ℝ, 
    point_on_parabola A₁ → point_on_parabola A₂ → point_on_parabola A₃ →
    line_tangent_to_circle A₁ A₂ → line_tangent_to_circle A₁ A₃ →
    line_tangent_to_circle A₂ A₃) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l662_66246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l662_66256

-- Define the constants
noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

-- State the theorem
theorem order_of_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l662_66256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_to_nonshaded_ratio_l662_66230

/-- Right triangle ABC with given side lengths and midpoints -/
structure RightTriangle where
  -- Define points as pairs of real numbers
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Define midpoints
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  -- Conditions
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  ab_length : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36
  ac_length : (C.1 - A.1)^2 + (C.2 - A.2)^2 = 64
  bc_length : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 100
  d_midpoint : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  e_midpoint : E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  f_midpoint : F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  g_midpoint : G = ((D.1 + F.1) / 2, (D.2 + F.2) / 2)
  h_midpoint : H = ((F.1 + E.1) / 2, (F.2 + E.2) / 2)

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

/-- Theorem: The ratio of shaded to non-shaded area is 23/25 -/
theorem shaded_to_nonshaded_ratio (t : RightTriangle) : 
  let total_area := triangleArea t.A t.B t.C
  let shaded_area := triangleArea t.D t.F t.G + triangleArea t.F t.E t.H
  let non_shaded_area := total_area - shaded_area
  shaded_area / non_shaded_area = 23 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_to_nonshaded_ratio_l662_66230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sofia_run_time_l662_66278

noncomputable def total_time (laps : ℕ) (lap_distance : ℝ) (first_segment : ℝ) (second_segment : ℝ) 
  (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  laps * ((first_segment / speed1) + ((lap_distance - first_segment) / speed2))

theorem sofia_run_time : 
  let laps : ℕ := 5
  let lap_distance : ℝ := 500
  let first_segment : ℝ := 200
  let second_segment : ℝ := 300
  let speed1 : ℝ := 4
  let speed2 : ℝ := 6
  total_time laps lap_distance first_segment second_segment speed1 speed2 = 500 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues
-- #eval total_time 5 500 200 300 4 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sofia_run_time_l662_66278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l662_66273

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, -1]
def b (l : ℝ) : Fin 2 → ℝ := ![2, l]
def c : Fin 2 → ℝ := ![3, 1]

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

/-- The dot product of two vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0 * w 0) + (v 1 * w 1)

/-- Two vectors are perpendicular if their dot product is zero -/
def are_perpendicular (v w : Fin 2 → ℝ) : Prop :=
  dot_product v w = 0

/-- Main theorem -/
theorem vector_relations :
  (∃ l, are_parallel a (b l) ∧ l = -2) ∧
  (∃ k, are_perpendicular (fun i => k * (a i) + (c i)) c ∧ k = -5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l662_66273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_19_consecutive_integers_twice_square_l662_66212

theorem smallest_sum_of_19_consecutive_integers_twice_square (s : ℕ) : 
  (∃ n : ℕ, s = (n + 9) * 19 ∧ 
            (∃ k : ℕ, s = 2 * k^2) ∧
            (∀ m : ℕ, m < n → ¬(∃ j : ℕ, (m + 9) * 19 = 2 * j^2))) →
  s = 722 := by
  
  -- The proof goes here
  sorry

#check smallest_sum_of_19_consecutive_integers_twice_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_19_consecutive_integers_twice_square_l662_66212
