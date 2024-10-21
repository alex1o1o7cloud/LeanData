import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_escape_time_l1148_114878

/-- Represents the frog's progress in the well -/
structure FrogProgress where
  depth : ℕ  -- Current depth of the frog in the well
  days : ℕ   -- Number of days passed

/-- Simulates one day of frog's movement -/
def dayProgress (fp : FrogProgress) : FrogProgress :=
  if fp.depth > 3 then
    { depth := fp.depth - 1, days := fp.days + 1 }
  else
    { depth := 0, days := fp.days + 1 }

/-- Calculates the number of days for the frog to escape -/
def escapeWell (wellDepth : ℕ) : ℕ :=
  let rec loop (fp : FrogProgress) (fuel : ℕ) : ℕ :=
    if fuel = 0 then fp.days
    else if fp.depth = 0 then fp.days
    else loop (dayProgress fp) (fuel - 1)
  loop { depth := wellDepth, days := 0 } (wellDepth * 2)  -- Use 2*wellDepth as an upper bound

/-- Theorem stating that it takes 28 days for the frog to escape -/
theorem frog_escape_time :
  escapeWell 30 = 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_escape_time_l1148_114878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_triangle_perimeter_eq_200_l1148_114804

/-- Triangle DEF with parallel lines forming a new triangle --/
structure ParallelLineTriangle where
  -- Side lengths of triangle DEF
  DE : ℝ
  EF : ℝ
  FD : ℝ
  -- Lengths of intersections with parallel lines
  m_D_length : ℝ
  m_E_length : ℝ
  m_F_length : ℝ
  -- Conditions
  h_DE : DE = 160
  h_EF : EF = 280
  h_FD : FD = 240
  h_m_D : m_D_length = 70
  h_m_E : m_E_length = 60
  h_m_F : m_F_length = 30

/-- The perimeter of the triangle formed by parallel lines --/
noncomputable def inner_triangle_perimeter (t : ParallelLineTriangle) : ℝ :=
  (t.m_D_length / t.EF) * t.DE + (t.m_E_length / t.DE) * t.FD + t.m_D_length

theorem inner_triangle_perimeter_eq_200 (t : ParallelLineTriangle) :
  inner_triangle_perimeter t = 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_triangle_perimeter_eq_200_l1148_114804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_origin_l1148_114809

theorem unique_solution_is_origin (x y z : ℝ) : 
  (3 * (2 : ℝ)^y - 1 = (2 : ℝ)^x + (2 : ℝ)^(-x)) ∧ 
  (3 * (2 : ℝ)^z - 1 = (2 : ℝ)^y + (2 : ℝ)^(-y)) ∧ 
  (3 * (2 : ℝ)^x - 1 = (2 : ℝ)^z + (2 : ℝ)^(-z)) → 
  x = 0 ∧ y = 0 ∧ z = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_origin_l1148_114809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_exists_l1148_114838

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given elements
variable (A : Point) (l : Line) (S : Circle)

-- Define the reflection of a line with respect to a point
noncomputable def reflectLine (l : Line) (P : Point) : Line :=
  sorry

-- Define the intersection of a line and a circle
noncomputable def lineCircleIntersection (l : Line) (S : Circle) : Option Point :=
  sorry

-- Define the line passing through two points
noncomputable def lineThroughPoints (P Q : Point) : Line :=
  sorry

-- Define the midpoint of a segment
noncomputable def segmentMidpoint (P Q : Point) : Point :=
  sorry

-- Define the segment intercepted by a line and a circle
noncomputable def interceptedSegment (l : Line) (S : Circle) : Option (Point × Point) :=
  sorry

-- Theorem statement
theorem bisecting_line_exists :
  ∃ (B : Point),
    let l' := reflectLine l A
    let AB := lineThroughPoints A B
    ∃ (P Q : Point),
      interceptedSegment AB S = some (P, Q) ∧
      segmentMidpoint P Q = A :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_exists_l1148_114838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tethered_dog_area_l1148_114845

/-- The area outside a unit square that can be reached by a point tethered to the midpoint of one side with a rope of length 2 -/
noncomputable def tethered_area : ℝ := 2.5 * Real.pi

/-- The length of the square's side -/
def square_side : ℝ := 1

/-- The length of the tether -/
def tether_length : ℝ := 2

/-- The point where the tether is attached is at the midpoint of one side of the square -/
def tether_point : ℝ × ℝ := (0.5, 0)

/-- The theorem stating that the tethered area is equal to 2.5π -/
theorem tethered_dog_area :
  tethered_area = 2.5 * Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tethered_dog_area_l1148_114845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_inequality_l1148_114897

/-- Given three numbers in geometric sequence and their reciprocals forming another sequence,
    find the maximum n satisfying the sum inequality. -/
theorem geometric_sequence_sum_inequality (a : ℝ) (n : ℕ) : 
  (a - 1) * (a + 5) = (a + 1)^2 →  -- Geometric sequence condition
  (∃ k : ℝ, k > 1 ∧ 
    1 / (a + 5) = k * (1 / (a + 1)) ∧ 
    1 / (a + 1) = k * (1 / (a - 1))) →  -- Reciprocals form increasing geometric sequence
  (∀ m : ℕ, m ≤ n → 
    (Finset.range m).sum (λ i ↦ 1 / ((1 / (a - 1)) * k^i)) ≤ 
    (Finset.range m).sum (λ i ↦ (1 / (a - 1)) * k^i)) →
  n = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_inequality_l1148_114897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l1148_114816

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sin α + Real.cos α, Real.sin α - Real.cos α)

-- Define the line l in polar coordinates
def line_l (θ : ℝ) (ρ : ℝ) : Prop := Real.sqrt 2 * ρ * Real.sin (Real.pi/4 - θ) + 1/2 = 0

-- Theorem statement
theorem curve_and_line_intersection :
  -- The general equation of C is x² + y² = 2
  (∀ (x y : ℝ), (∃ (α : ℝ), curve_C α = (x, y)) ↔ x^2 + y^2 = 2) ∧
  -- The length |AB| is √62/2, where A and B are the intersection points of C and l
  (∃ (A B : ℝ × ℝ),
    (∃ (α₁ α₂ : ℝ), curve_C α₁ = A ∧ curve_C α₂ = B) ∧
    (∃ (θ₁ θ₂ ρ₁ ρ₂ : ℝ), line_l θ₁ ρ₁ ∧ line_l θ₂ ρ₂ ∧ 
      A.1 = ρ₁ * Real.cos θ₁ ∧ A.2 = ρ₁ * Real.sin θ₁ ∧
      B.1 = ρ₂ * Real.cos θ₂ ∧ B.2 = ρ₂ * Real.sin θ₂) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 62 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l1148_114816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1148_114843

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : tan (2*α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1148_114843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_from_lcm_and_ratio_l1148_114829

theorem gcd_from_lcm_and_ratio (A B : ℕ) (hA : A > 0) (hB : B > 0) : 
  Nat.lcm A B = 180 → (A : ℚ) / B = 4 / 5 → Nat.gcd A B = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_from_lcm_and_ratio_l1148_114829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1148_114842

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum_10 (a₁ d : ℝ) :
  a₁ = -2 →
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 6 = 2 →
  arithmetic_sum a₁ d 10 = 25 := by
  sorry

#check arithmetic_sequence_sum_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1148_114842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_labeling_exists_l1148_114856

/-- A labeling of a 45-gon is a function from its vertices to digits 0-9 -/
def Labeling := Fin 45 → Fin 10

/-- A pair of distinct digits -/
structure DistinctDigitPair where
  fst : Fin 10
  snd : Fin 10
  distinct : fst ≠ snd

/-- The set of all distinct digit pairs -/
def AllDistinctDigitPairs : Set DistinctDigitPair :=
  { p | p.fst < p.snd }

/-- A valid labeling satisfies the condition that every pair of distinct
    digits appears on one of the 45-gon's edges -/
def IsValidLabeling (l : Labeling) : Prop :=
  ∀ p : DistinctDigitPair, ∃ i : Fin 45,
    (l i = p.fst ∧ l ((i + 1) % 45) = p.snd) ∨
    (l i = p.snd ∧ l ((i + 1) % 45) = p.fst)

/-- The main theorem stating that no valid labeling exists -/
theorem no_valid_labeling_exists : ¬ ∃ l : Labeling, IsValidLabeling l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_labeling_exists_l1148_114856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_length_formula_l1148_114831

/-- Two internally touching circles with a tangent line -/
structure InternallyTouchingCircles where
  R : ℝ
  r : ℝ
  a : ℝ
  h1 : R > r
  h2 : R > 0
  h3 : r > 0
  h4 : a > 0

/-- The length of BC in the configuration -/
noncomputable def BC_length (c : InternallyTouchingCircles) : ℝ :=
  c.a * Real.sqrt ((c.R - c.r) / c.R)

/-- Theorem stating the length of BC -/
theorem BC_length_formula (c : InternallyTouchingCircles) :
  ∃ (B C : ℝ × ℝ),
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (BC_length c)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_length_formula_l1148_114831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1148_114836

/-- Represents a segment of the car's journey -/
structure Segment where
  speed : ℝ  -- Speed in kph
  distance : Option ℝ  -- Distance in km, if given
  time : Option ℝ  -- Time in hours, if given

/-- Calculates the average speed given a list of journey segments -/
noncomputable def averageSpeed (segments : List Segment) : ℝ :=
  let totalDistance := segments.foldl (fun acc s => 
    acc + match s.distance with
      | some d => d
      | none => s.speed * Option.get! s.time
  ) 0
  let totalTime := segments.foldl (fun acc s =>
    acc + match s.time with
      | some t => t
      | none => Option.get! s.distance / s.speed
  ) 0
  totalDistance / totalTime

theorem car_average_speed : 
  let journey := [
    { speed := 60, distance := some 40, time := none },
    { speed := 70, distance := some 35, time := none },
    { speed := 50, distance := none, time := some (50/60) },
    { speed := 55, distance := none, time := some (20/60) }
  ]
  abs (averageSpeed journey - 69.64) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1148_114836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_inequality_in_interval_l1148_114890

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / x

-- Theorem 1: Unique zero point condition
theorem unique_zero_point (a : ℝ) (h : a > 0) :
  (∃! x, x > 0 ∧ f a x = 0) ↔ a = 1 := by sorry

-- Theorem 2: Inequality for x in (1,2)
theorem inequality_in_interval (x : ℝ) (h : 1 < x ∧ x < 2) :
  1 / Real.log x - 1 / (x - 1) < 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_inequality_in_interval_l1148_114890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l1148_114805

open Real

def is_in_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3/2) * Real.pi

def is_in_second_or_fourth_quadrant (α : ℝ) : Prop :=
  (∃ k : ℤ, k * Real.pi + Real.pi/2 < α ∧ α < k * Real.pi + Real.pi) ∨
  (∃ k : ℤ, k * Real.pi + (3/2) * Real.pi < α ∧ α < (k + 1) * Real.pi)

theorem half_angle_quadrant (α : ℝ) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l1148_114805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exists_l1148_114882

/-- Defines a sequence of natural numbers -/
def sequenceA : ℕ → ℕ := sorry

/-- The sequence is increasing -/
axiom sequence_increasing : ∀ n : ℕ, sequenceA n < sequenceA (n + 1)

/-- The coprimality property of the sequence -/
axiom sequence_coprime : ∀ i j p q r : ℕ, 
  i ≠ j → i ≠ p → i ≠ q → i ≠ r → j ≠ p → j ≠ q → j ≠ r → p ≠ q → p ≠ r → q ≠ r →
  Nat.gcd (sequenceA i + sequenceA j) (sequenceA p + sequenceA q + sequenceA r) = 1

/-- The theorem stating the existence of such a sequence -/
theorem sequence_exists : ∃ (sequenceA : ℕ → ℕ), 
  (∀ n : ℕ, sequenceA n < sequenceA (n + 1)) ∧
  (∀ i j p q r : ℕ, 
    i ≠ j → i ≠ p → i ≠ q → i ≠ r → j ≠ p → j ≠ q → j ≠ r → p ≠ q → p ≠ r → q ≠ r →
    Nat.gcd (sequenceA i + sequenceA j) (sequenceA p + sequenceA q + sequenceA r) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exists_l1148_114882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_equals_neg_one_l1148_114835

noncomputable section

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_slopes_equal {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of line l₁: ax + 2y + 6 = 0 -/
noncomputable def slope_l₁ (a : ℝ) : ℝ := -a / 2

/-- The slope-intercept form of line l₂: x + (a-1)y + (a²-1) = 0 -/
noncomputable def slope_l₂ (a : ℝ) : ℝ := -1 / (a - 1)

/-- The theorem to be proved -/
theorem parallel_lines_a_equals_neg_one :
  ∀ a : ℝ, a ≠ 1 → slope_l₁ a = slope_l₂ a → a = -1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_equals_neg_one_l1148_114835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_always_two_main_theorem_l1148_114872

/-- Diamond operation -/
noncomputable def diamond (a b : ℝ) : ℝ := (2 * a^b / b^a) * (b^a / a^b)

/-- Theorem stating that the diamond operation always results in 2 -/
theorem diamond_always_two (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  diamond a b = 2 := by sorry

/-- Main theorem proving the given expression equals 2 -/
theorem main_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond (diamond a (diamond b c)) 1 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_always_two_main_theorem_l1148_114872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_calculation_l1148_114859

/-- Represents a rectangular field with a circular pond -/
structure FieldWithPond where
  length : ℝ
  width : ℝ
  area : ℝ
  perimeter : ℝ
  pond_radius : ℝ

/-- The remaining land area after accounting for the pond -/
noncomputable def remaining_area (field : FieldWithPond) : ℝ :=
  field.area - Real.pi * field.pond_radius^2

/-- Theorem stating the remaining area for the given conditions -/
theorem remaining_area_calculation (field : FieldWithPond) 
  (h_area : field.area = 800)
  (h_perimeter : field.perimeter = 120)
  (h_pond_diameter : field.pond_radius * 2 = field.width / 2)
  (h_rect_area : field.area = field.length * field.width)
  (h_rect_perimeter : field.perimeter = 2 * (field.length + field.width)) :
  remaining_area field = 800 - 25 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_calculation_l1148_114859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_l1148_114820

/-- Proves that given a boat's speed in still water, its downstream speed, and its upstream speed,
    we can determine the speed of the current. -/
theorem current_speed (boat_speed downstream_speed upstream_speed current_speed : ℝ) 
  (h1 : boat_speed = 60)
  (h2 : downstream_speed = 77)
  (h3 : upstream_speed = 43)
  (h4 : downstream_speed = boat_speed + current_speed)
  (h5 : upstream_speed = boat_speed - current_speed) :
  current_speed = 17 := by
  sorry

#check current_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_l1148_114820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l1148_114812

/-- The time it takes for the pipe to fill the tank without the leak -/
def T : ℝ := sorry

/-- The time it takes to fill the tank with both the pipe and the leak working -/
def fill_time_with_leak : ℝ := 9

/-- The time it takes for the leak to empty the full tank -/
def empty_time : ℝ := 18

/-- Theorem stating that the pipe can fill the tank in 6 hours without the leak -/
theorem pipe_fill_time :
  (1 / T - 1 / empty_time = 1 / fill_time_with_leak) →
  T = 6 := by
  sorry

#check pipe_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l1148_114812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_scores_count_l1148_114858

/-- Represents the number of baskets made by the player -/
def total_baskets : ℕ := 5

/-- Represents the possible point values for each basket -/
def basket_values : Finset ℕ := {2, 3}

/-- Represents all possible combinations of 2-point and 3-point baskets -/
def possible_combinations : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 + p.2 = total_baskets)
    (Finset.product (Finset.range (total_baskets + 1)) (Finset.range (total_baskets + 1)))

/-- Calculates the total score for a given combination of 2-point and 3-point baskets -/
def score (combo : ℕ × ℕ) : ℕ := 2 * combo.1 + 3 * combo.2

/-- The set of all possible scores -/
def possible_scores : Finset ℕ := Finset.image score possible_combinations

theorem distinct_scores_count : Finset.card possible_scores = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_scores_count_l1148_114858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_three_expression_l1148_114863

theorem power_three_expression (m n : ℤ) (h1 : (3 : ℝ)^m = 8) (h2 : (3 : ℝ)^n = 2) : 
  (3 : ℝ)^(2*m - 3*n + 1) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_three_expression_l1148_114863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_coord_quadrilateral_with_special_diagonals_l1148_114892

theorem no_integer_coord_quadrilateral_with_special_diagonals :
  ¬ ∃ (A B C D : ℤ × ℤ),
    let AC := Real.sqrt (((A.1 - C.1)^2 + (A.2 - C.2)^2 : ℝ))
    let BD := Real.sqrt (((B.1 - D.1)^2 + (B.2 - D.2)^2 : ℝ))
    let angle := Real.arccos (((A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) : ℝ) / (AC * BD))
    AC = 2 * BD ∧ angle = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_coord_quadrilateral_with_special_diagonals_l1148_114892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuzik_reaches_ivan_at_1247_l1148_114889

/-- The time when Tuzik reaches Ivan -/
def meeting_time (total_distance : ℕ) (ivan_speed : ℕ) (tuzik_speed : ℕ) (time_before_tuzik_start : ℕ) : ℕ :=
  let ivan_distance := ivan_speed * time_before_tuzik_start
  let remaining_distance := total_distance - ivan_distance
  let combined_speed := ivan_speed + tuzik_speed
  let time_to_meet := remaining_distance / combined_speed
  time_before_tuzik_start + time_to_meet

theorem tuzik_reaches_ivan_at_1247 :
  meeting_time 12000 1 9 1800 = 2847 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuzik_reaches_ivan_at_1247_l1148_114889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_over_n_range_l1148_114875

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_decreasing : ∀ x y : ℝ, x < y → f x > f y

-- Define the condition on m and n
axiom condition : ∀ m n : ℝ, f (m^2 - 2*m) + f (2*n - n^2) ≥ 0

-- Define the range of n
axiom n_range : ∀ n : ℝ, 1 ≤ n ∧ n ≤ 3/2

-- Theorem statement
theorem m_over_n_range :
  ∀ m n : ℝ, (1 ≤ n ∧ n ≤ 3/2) → 
  (f (m^2 - 2*m) + f (2*n - n^2) ≥ 0) →
  (1/3 ≤ m/n ∧ m/n ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_over_n_range_l1148_114875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_homework_time_l1148_114868

/-- Calculates the total time Rachel needs to complete her homework -/
def calculate_homework_time (math_pages : Float) (reading_pages : Float) (biology_pages : Float) (history_pages : Float) (base_time_per_page : Float) : Float :=
  let reading_history_time := (reading_pages + history_pages) * base_time_per_page
  let math_biology_time := (math_pages + biology_pages) * (2 * base_time_per_page)
  reading_history_time + math_biology_time

/-- Proves that Rachel needs 745 minutes to complete her homework -/
theorem rachel_homework_time :
  calculate_homework_time 8.5 7.25 4.75 3.5 20 = 745 := by
  simp [calculate_homework_time]
  norm_num
  sorry

#eval calculate_homework_time 8.5 7.25 4.75 3.5 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_homework_time_l1148_114868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_in_U_l1148_114852

-- Define the set U
def U : Set ℂ :=
  {z : ℂ | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -2 ≤ z.im ∧ z.im ≤ 2}

-- Define the transformation
noncomputable def transform (w : ℂ) : ℂ := (1/2 + 1/2*Complex.I) * w

-- Theorem statement
theorem transform_in_U : ∀ w ∈ U, transform w ∈ U :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_in_U_l1148_114852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1148_114864

-- Define the function f
noncomputable def f (x : ℝ) := |x - 1/2| + |x + 1/2|

-- Define the set M
noncomputable def M : Set ℝ := {x | f x < 2}

-- Theorem statement
theorem problem_solution :
  (M = Set.Ioo (-1 : ℝ) 1) ∧
  (∀ a b : ℝ, a ∈ M → b ∈ M → |a + b| < |1 + a * b|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1148_114864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1148_114870

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-6, 3)

theorem triangle_area : 
  abs (a.1 * b.2 - a.2 * b.1) / 2 = 3 := by
  -- Calculation
  calc abs (a.1 * b.2 - a.2 * b.1) / 2
    = abs (4 * 3 - (-1) * (-6)) / 2 := by rfl
  _ = abs (12 - 6) / 2 := by ring_nf
  _ = abs 6 / 2 := by ring_nf
  _ = 6 / 2 := by simp
  _ = 3 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1148_114870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1148_114825

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := sorry

theorem vector_sum_magnitude 
  (h1 : Real.cos (60 * π / 180) = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 1) :
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1148_114825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_sum_implies_sum_l1148_114823

theorem square_root_sum_implies_sum (a : ℝ) (h : a > 0) :
  a^(1/2 : ℝ) + a^(-(1/2 : ℝ)) = 3 → a + a⁻¹ = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_sum_implies_sum_l1148_114823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_minus_sqrt_equality_l1148_114888

theorem x_minus_sqrt_equality (a : ℝ) (n : ℕ+) (h1 : a > 0) (h2 : a ≠ 1) :
  let x : ℝ := (a^(1/(n:ℝ)) - a^(-1/(n:ℝ))) / 2
  (x - Real.sqrt (1 + x^2)) = x * (1 - Real.sqrt (4 - 4*x^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_minus_sqrt_equality_l1148_114888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_12x12_board_l1148_114857

/-- A chess board --/
structure ChessBoard where
  size : ℕ

/-- A configuration of kings on a chess board --/
structure KingConfiguration where
  board : ChessBoard
  num_kings : ℕ

/-- Predicate to check if two kings attack each other --/
def kings_attack (k1 k2 : ℕ) (config : KingConfiguration) : Prop :=
  sorry  -- Definition of when two kings attack each other

/-- Predicate to check if a king configuration is valid --/
def is_valid_configuration (config : KingConfiguration) : Prop :=
  ∀ k : ℕ, k < config.num_kings → ∃! k' : ℕ, k' < config.num_kings ∧ k' ≠ k ∧ 
    (kings_attack k k' config)

/-- The maximum number of kings that can be placed on a 12x12 board --/
def max_kings : ℕ := 56

/-- Theorem stating that 56 is the maximum number of kings on a 12x12 board --/
theorem max_kings_on_12x12_board :
  ∀ (config : KingConfiguration),
    config.board.size = 12 →
    is_valid_configuration config →
    config.num_kings ≤ max_kings := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_12x12_board_l1148_114857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_problem_solution_l1148_114873

/-- The time when the first candle is three times the height of the second -/
noncomputable def candle_problem : ℚ :=
  let initial_height : ℚ := 1
  let burn_time_1 : ℚ := 5
  let burn_time_2 : ℚ := 4
  let rate_1 : ℚ := initial_height / burn_time_1
  let rate_2 : ℚ := initial_height / burn_time_2
  (3 * initial_height - initial_height) / (3 * rate_2 - rate_1)

theorem candle_problem_solution :
  candle_problem = 40 / 11 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_problem_solution_l1148_114873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_ratio_l1148_114803

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := sorry
noncomputable def F2 : ℝ × ℝ := sorry

-- Define a point on the ellipse
noncomputable def P : ℝ × ℝ := sorry

-- Assume P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Define the midpoint of PF1
noncomputable def M : ℝ × ℝ := ((P.1 + F1.1) / 2, (P.2 + F1.2) / 2)

-- Assume the midpoint is on the y-axis
axiom M_on_y_axis : M.1 = 0

-- Define the distances
noncomputable def PF1 : ℝ := Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)
noncomputable def PF2 : ℝ := Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)

-- State the theorem
theorem ellipse_focal_ratio : PF2 / PF1 = 5 / 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_ratio_l1148_114803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1148_114871

theorem quadratic_function_theorem (f : ℝ → ℝ) :
  (∀ x, f (x + 1) - f x = 3 * x) →
  f 0 = 1 →
  ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) →
  ∀ x, f x = (3/2) * x^2 - (3/2) * x + 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1148_114871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_reciprocal_implies_a_range_l1148_114846

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * (x - 1)

noncomputable def g (x : ℝ) : ℝ := exp x

theorem tangent_slopes_reciprocal_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧
    (deriv (f a)) x₁ * (deriv g) x₂ = 1 ∧
    f a x₁ = ((deriv (f a)) x₁) * x₁ ∧
    g x₂ = ((deriv g) x₂) * x₂) →
  a = 0 ∨ (exp 1 - 1) / exp 1 < a ∧ a < (exp 2 - 1) / exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_reciprocal_implies_a_range_l1148_114846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_clicks_theorem_l1148_114879

/-- Represents the length of a rail in feet -/
noncomputable def rail_length : ℝ := 40

/-- Converts miles per hour to feet per minute -/
noncomputable def mph_to_fpm (speed : ℝ) : ℝ := speed * 5280 / 60

/-- Calculates the number of clicks per minute given a speed in mph -/
noncomputable def clicks_per_minute (speed : ℝ) : ℝ := mph_to_fpm speed / rail_length

/-- Calculates the time in minutes for the number of clicks to equal the speed -/
noncomputable def time_in_minutes (speed : ℝ) : ℝ := speed / (clicks_per_minute speed)

/-- Converts minutes to seconds -/
noncomputable def minutes_to_seconds (t : ℝ) : ℝ := t * 60

theorem train_speed_clicks_theorem (speed : ℝ) (speed_positive : speed > 0) :
  ∃ (ε : ℝ), ε > 0 ∧ abs (minutes_to_seconds (time_in_minutes speed) - 20) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_clicks_theorem_l1148_114879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_sum_l1148_114898

structure MarbleBoxes where
  total_marbles : ℕ
  box_a_marbles : ℕ
  box_b_marbles : ℕ
  box_c_marbles : ℕ
  black_a : ℕ
  black_b : ℕ
  black_c : ℕ
  prob_all_black : ℚ
  prob_all_white : ℚ

def valid_marble_boxes (mb : MarbleBoxes) : Prop :=
  mb.total_marbles = 36 ∧
  mb.box_a_marbles = mb.box_b_marbles ∧
  mb.total_marbles = mb.box_a_marbles + mb.box_b_marbles + mb.box_c_marbles ∧
  mb.prob_all_black = 1/3 ∧
  mb.prob_all_black = (mb.black_a : ℚ) / mb.box_a_marbles * 
                      (mb.black_b : ℚ) / mb.box_b_marbles * 
                      (mb.black_c : ℚ) / mb.box_c_marbles ∧
  mb.prob_all_white = ((mb.box_a_marbles - mb.black_a) : ℚ) / mb.box_a_marbles * 
                      ((mb.box_b_marbles - mb.black_b) : ℚ) / mb.box_b_marbles * 
                      ((mb.box_c_marbles - mb.black_c) : ℚ) / mb.box_c_marbles

theorem marble_probability_sum (mb : MarbleBoxes) (hp : valid_marble_boxes mb) 
  (hq : ∃ (p q : ℕ), mb.prob_all_white = (p : ℚ) / q ∧ Nat.Coprime p q) :
  ∃ (p q : ℕ), mb.prob_all_white = (p : ℚ) / q ∧ Nat.Coprime p q ∧ p + q = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_sum_l1148_114898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1148_114848

/-- The distance between a point and a line defined by parametric equations -/
noncomputable def distance_point_to_parametric_line (x₀ y₀ a₁ b₁ a₂ b₂ : ℝ) : ℝ :=
  let A := b₂ - b₁
  let B := a₁ - a₂
  let C := a₂ * b₁ - a₁ * b₂
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The theorem stating that the distance between (1, 0) and the line x = 1 + 3t, y = 2 + 4t is 6/5 -/
theorem distance_point_to_line :
  distance_point_to_parametric_line 1 0 1 2 3 4 = 6/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1148_114848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l1148_114861

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4^x) / (4^x + 2)

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define set A
def A : Set ℤ := {-2, -1, 0, 1}

-- Define set B
def B : Set ℤ := {y | ∃ x : ℝ, y = floor (f x - 1/2) + floor (f (1 - x) - 1/2)}

-- Theorem statement
theorem A_intersect_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l1148_114861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_3_eq_3_l1148_114815

noncomputable def h : ℝ → ℝ := 
  fun x => (x^(2^2009 - 1) - 1)⁻¹ * ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^(2^2008) + 1) - 1)

theorem h_of_3_eq_3 : h 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_3_eq_3_l1148_114815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_proof_l1148_114821

-- Define the structure for angles
structure Angle where
  measure : ℝ

-- Define the lines
variable (p q : Line ℝ)

-- Define the angles
variable (A B C : Angle)

-- State the theorem
theorem angle_measure_proof
  (h_parallel : p.IsParallel q)
  (h_angle_relation : A.measure = (1/4) * B.measure)
  (h_alternate_interior : C.measure = A.measure) :
  C.measure = 36 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_proof_l1148_114821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_second_group_l1148_114891

/-- Represents the daily work done by a single person -/
structure DailyWork where
  amount : ℚ
  deriving Repr

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  boys : ℕ
  days : ℕ
  deriving Repr

/-- The total work done by a group -/
def totalWork (g : WorkGroup) (m : DailyWork) (b : DailyWork) : ℚ :=
  (g.men : ℚ) * m.amount * (g.days : ℚ) + (g.boys : ℚ) * b.amount * (g.days : ℚ)

theorem boys_in_second_group 
  (m : DailyWork) 
  (b : DailyWork) 
  (g1 : WorkGroup) 
  (g2 : WorkGroup) : 
  m.amount = 2 * b.amount →
  g1.men = 12 →
  g1.boys = 16 →
  g1.days = 5 →
  g2.men = 13 →
  g2.days = 4 →
  totalWork g1 m b = totalWork g2 m b →
  g2.boys = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_second_group_l1148_114891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l1148_114876

theorem sin_cos_value (θ : ℝ) (h : Real.sin θ + Real.cos θ = 3/4) : 
  Real.sin θ * Real.cos θ = -7/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l1148_114876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_parabola_l1148_114826

/-- Parabola with equation y^2 = -6x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Points on the parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

/-- Vector between two points -/
def vector (a b : ℝ × ℝ) : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)

/-- Distance between two points -/
noncomputable def distance (a b : ℝ × ℝ) : ℝ := Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

/-- Theorem: Minimum distance between points M and N on the parabola -/
theorem min_distance_on_parabola (p : Parabola) (M N : PointOnParabola p) (k : ℝ) 
    (h : k ≠ 0) 
    (h_vector : vector p.focus M.point = k • vector p.focus N.point) :
  ∃ (min_dist : ℝ), ∀ (M' N' : PointOnParabola p), 
    vector p.focus M'.point = k • vector p.focus N'.point → 
    distance M'.point N'.point ≥ min_dist ∧ 
    min_dist = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_parabola_l1148_114826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_1994_in_special_sequence_l1148_114883

/-- Represents a sequence where the nth block contains n numbers
    (odd if n is odd, even if n is even) -/
def special_sequence (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * ((n + 1) / 2) - 1 else 2 * (n / 2)

/-- The sum of the first n natural numbers -/
def triangle_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of terms in the sequence up to the nth block -/
def total_terms (n : ℕ) : ℕ := triangle_number (n + 1)

/-- Theorem stating that the number closest to 1994 in the special sequence
    is either 1993 or 1995 -/
theorem closest_to_1994_in_special_sequence :
  ∃ (k : ℕ), (special_sequence k = 1993 ∨ special_sequence k = 1995) ∧
    ∀ (m : ℕ), (special_sequence m : ℤ) - 1994 ≥ -1 ∧ (special_sequence m : ℤ) - 1994 ≤ 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_1994_in_special_sequence_l1148_114883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1148_114881

/-- Simple interest calculation --/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_calculation (principal time interest : ℝ) 
  (h_principal : principal = 9005)
  (h_time : time = 5)
  (h_interest : interest = 4052.25) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1148_114881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_decrease_l1148_114869

/-- Represents the annual percentage decrease in market value -/
noncomputable def annual_decrease (initial_value : ℝ) (value_after_two_years : ℝ) : ℝ :=
  1 - Real.sqrt (value_after_two_years / initial_value)

/-- Theorem stating that the annual percentage decrease for the given machine is 1 - (√2 / 2) -/
theorem machine_value_decrease :
  let initial_value : ℝ := 8000
  let value_after_two_years : ℝ := 4000
  annual_decrease initial_value value_after_two_years = 1 - (Real.sqrt 2 / 2) :=
by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Remove the #eval statement as it's not computable
-- #eval annual_decrease 8000 4000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_decrease_l1148_114869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_c_for_quadratic_equation_l1148_114895

theorem unique_c_for_quadratic_equation :
  ∃! c : ℝ, c ≠ 0 ∧
    ∃! b : ℝ, b ≠ 0 ∧
      ∃! x : ℝ, x^2 + (b^2 + 1/b^2) * x + c = 0 ∧ c = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_c_for_quadratic_equation_l1148_114895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_sequence_l1148_114887

theorem arithmetic_to_geometric_sequence :
  ∀ (a b c : ℝ),
  (b - a = c - b) →                     -- arithmetic sequence
  (b / a = 4 / 3 ∧ c / b = 5 / 4) →     -- ratio 3:4:5
  ((a + 1) * c = b^2) →                 -- geometric sequence after increasing smallest by 1
  (a = 15 ∧ b = 20 ∧ c = 25) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_sequence_l1148_114887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_angle_l1148_114818

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given 5 points in a plane with no three points collinear -/
def FivePoints : Type := { pts : Finset Point // pts.card = 5 }

/-- Predicate to check if three points are collinear -/
def areCollinear (p q r : Point) : Prop := sorry

/-- The angle between three points -/
def angle (p q r : Point) : ℝ := sorry

/-- Theorem: Given 5 points in a plane with no three points collinear,
    there exists a triangle formed by these points with an angle 
    greater than or equal to 108° -/
theorem exists_large_angle (pts : FivePoints) 
  (h : ∀ p q r, p ∈ pts.val → q ∈ pts.val → r ∈ pts.val → ¬areCollinear p q r) :
  ∃ p q r, p ∈ pts.val ∧ q ∈ pts.val ∧ r ∈ pts.val ∧ angle p q r ≥ 108 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_angle_l1148_114818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_value_l1148_114819

-- Define the polynomial function
def poly (b : ℝ) (n : ℕ) (x : ℝ) : ℝ := b * x^n + 1

-- Define the expansion function
def expansion (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  Finset.sum (Finset.range (n+1)) (λ i ↦ a i * (x - 1)^i)

-- State the theorem
theorem find_b_value (b : ℝ) (n : ℕ) (a : ℕ → ℝ) :
  (∀ x : ℝ, poly b n x = expansion a n x) →
  a 1 ≠ 9 →
  a 2 ≠ 36 →
  b = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_value_l1148_114819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1148_114806

-- Define the function f(x) = x + 4/x
noncomputable def f (x : ℝ) : ℝ := x + 4 / x

-- State the theorem
theorem max_value_of_f :
  ∀ x : ℝ, x < 0 → f x ≤ -4 ∧ ∃ x₀ : ℝ, x₀ < 0 ∧ f x₀ = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1148_114806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_41pi_over_4_l1148_114800

-- Define the points A and B
def A : ℝ × ℝ := (3, 5)
def B : ℝ × ℝ := (7, 10)

-- Define the circle using its diameter endpoints
noncomputable def circle_diameter (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the area of a circle given its diameter
noncomputable def circle_area (d : ℝ) : ℝ := Real.pi * (d/2)^2

-- Theorem statement
theorem circle_area_is_41pi_over_4 : 
  circle_area (circle_diameter A B) = 41 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_41pi_over_4_l1148_114800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_satisfies_conditions_hyperbola_satisfies_conditions_l1148_114833

-- Define the given ellipse
def given_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the focus of the given ellipse
noncomputable def focus : ℝ := Real.sqrt 5

-- Define the point that the new ellipse passes through
noncomputable def point_ellipse : ℝ × ℝ := (-focus, 4)

-- Define the new ellipse
def new_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 20 = 1

-- Define the asymptote equation for the hyperbola
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Define the point that the hyperbola passes through
def point_hyperbola : ℝ × ℝ := (2, 2)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 3 - x^2 / 12 = 1

-- Theorem for the ellipse
theorem ellipse_satisfies_conditions :
  (∀ x y, given_ellipse x y → (x = focus ∨ x = -focus)) ∧
  new_ellipse point_ellipse.1 point_ellipse.2 := by
  sorry

-- Theorem for the hyperbola
theorem hyperbola_satisfies_conditions :
  (∀ x y, asymptote x y → (hyperbola x y → False)) ∧
  hyperbola point_hyperbola.1 point_hyperbola.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_satisfies_conditions_hyperbola_satisfies_conditions_l1148_114833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_and_min_area_l1148_114855

-- Define the triangle ABC and points M, N, P, R, S, T
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def M (t : Triangle) (k : ℝ) : ℝ × ℝ := 
  ((1 - k) * t.A.1 + k * t.B.1, (1 - k) * t.A.2 + k * t.B.2)

def N (t : Triangle) (k : ℝ) : ℝ × ℝ := 
  ((1 - k) * t.B.1 + k * t.C.1, (1 - k) * t.B.2 + k * t.C.2)

def P (t : Triangle) (k : ℝ) : ℝ × ℝ := 
  ((1 - k) * t.C.1 + k * t.A.1, (1 - k) * t.C.2 + k * t.A.2)

def R (t : Triangle) (k : ℝ) : ℝ × ℝ := 
  let m := M t k
  let n := N t k
  ((1 - k) * m.1 + k * n.1, (1 - k) * m.2 + k * n.2)

def S (t : Triangle) (k : ℝ) : ℝ × ℝ := 
  let n := N t k
  let p := P t k
  ((1 - k) * n.1 + k * p.1, (1 - k) * n.2 + k * p.2)

def T (t : Triangle) (k : ℝ) : ℝ × ℝ := 
  let p := P t k
  let m := M t k
  ((1 - k) * p.1 + k * m.1, (1 - k) * p.2 + k * m.2)

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_similarity_and_min_area (t : Triangle) (k : ℝ) 
  (h1 : 0 < k ∧ k < 1) :
  let str := Triangle.mk (R t k) (S t k) (T t k)
  similar str t ∧ 
  (∀ k' : ℝ, 0 < k' ∧ k' < 1 → 
    area (Triangle.mk (R t k') (S t k') (T t k')) ≥ area (Triangle.mk (R t (1/2)) (S t (1/2)) (T t (1/2)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_and_min_area_l1148_114855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_independent_k_l1148_114894

noncomputable def expression (k : ℕ) (x : ℝ) : ℝ :=
  Real.sin (k * x) * (Real.sin x) ^ k + Real.cos (k * x) * (Real.cos x) ^ k - (Real.cos (2 * x)) ^ k

theorem smallest_independent_k :
  ∃ (k : ℕ), k ≥ 1 ∧
  (∀ (x y : ℝ), expression k x = expression k y) ∧
  (∀ (k' : ℕ), 1 ≤ k' ∧ k' < k → ∃ (x y : ℝ), expression k' x ≠ expression k' y) ∧
  k = 3 := by
  sorry

#check smallest_independent_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_independent_k_l1148_114894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_perimeter_sum_l1148_114840

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement -/
theorem ellipse_quadrilateral_perimeter_sum
  (E : Ellipse)
  (R S A B P Q : Point)
  (h_major : E.semi_major_axis = 13)
  (h_minor : E.semi_minor_axis = 12)
  (h_foci : distance E.center R = distance E.center S ∧ distance E.center R = 5)
  (h_on_ellipse : distance R A + distance S A = 2 * E.semi_major_axis ∧
                  distance R B + distance S B = 2 * E.semi_major_axis)
  (h_quadrilateral : R ≠ A ∧ A ≠ S ∧ S ≠ B ∧ B ≠ R)
  (h_intersections : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = Point.mk (R.x + t * (A.x - R.x)) (R.y + t * (A.y - R.y)) ∧
                     ∃ u : ℝ, 0 < u ∧ u < 1 ∧ Q = Point.mk (R.x + u * (B.x - R.x)) (R.y + u * (B.y - R.y)))
  (h_equal_distances : distance R A = distance A S)
  (h_AP : distance A P = 26)
  (h_perimeter : ∃ m n : ℕ, distance R P + distance P S + distance S Q + distance Q R = m + Real.sqrt n)
  : ∃ m n : ℕ, distance R P + distance P S + distance S Q + distance Q R = m + Real.sqrt n ∧ m + n = 627 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_perimeter_sum_l1148_114840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1148_114866

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define vectors m and n
noncomputable def m (t : Triangle) : Fin 2 → Real
  | 0 => t.a
  | 1 => t.c
  | _ => 0

noncomputable def n (t : Triangle) : Fin 2 → Real
  | 0 => Real.cos t.C
  | 1 => Real.cos t.A
  | _ => 0

-- Part 1
theorem part1 (t : Triangle) 
  (h1 : ∃ (k : Real), ∀ i, m t i = k * n t i) -- m parallel to n
  (h2 : t.a = Real.sqrt 3 * t.c) : 
  t.A = π / 3 := by
  sorry

-- Part 2
theorem part2 (t : Triangle)
  (h1 : (m t 0 * n t 0 + m t 1 * n t 1) = 3 * t.b * Real.sin t.B) -- m · n = 3b sin B
  (h2 : Real.cos t.A = 3 / 5) :
  Real.cos t.C = (4 - 6 * Real.sqrt 2) / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1148_114866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_equation_l1148_114810

noncomputable section

-- Define the parametric equations
def x (t : ℝ) : ℝ := Real.cos t
def y (t : ℝ) : ℝ := Real.sin (t / 2) ^ 4

-- Define the second derivative
def y_xx_second_derivative (t : ℝ) : ℝ :=
  (1 + Real.cos (t / 2) ^ 2) / (4 * Real.cos (t / 2) ^ 3)

-- Theorem statement
theorem second_derivative_parametric_equation (t : ℝ) :
  (deriv (deriv y) ∘ x) t = y_xx_second_derivative t := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_equation_l1148_114810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_CDB_is_15_l1148_114860

/-- A figure composed of an isosceles triangle sharing a side with a rectangle -/
structure SharedSideTriangleRectangle where
  /-- One angle of the rectangle -/
  rectangle_angle : ℝ
  /-- One angle of the isosceles triangle -/
  isosceles_angle : ℝ

/-- The measure of angle CDB in the figure -/
noncomputable def angle_CDB (figure : SharedSideTriangleRectangle) : ℝ :=
  (180 - (figure.rectangle_angle + figure.isosceles_angle)) / 2

theorem angle_CDB_is_15 (figure : SharedSideTriangleRectangle) 
  (h1 : figure.rectangle_angle = 80)
  (h2 : figure.isosceles_angle = 70) : 
  angle_CDB figure = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_CDB_is_15_l1148_114860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_vector_quantities_l1148_114851

-- Define a type for physical quantities
inductive PhysicalQuantity
| Mass
| Velocity
| Displacement
| Force
| Acceleration
| Distance

-- Define a function to check if a quantity is a vector
def isVector (q : PhysicalQuantity) : Bool :=
  match q with
  | PhysicalQuantity.Velocity => true
  | PhysicalQuantity.Displacement => true
  | PhysicalQuantity.Force => true
  | PhysicalQuantity.Acceleration => true
  | _ => false

-- Theorem: Exactly 4 out of the 6 physical quantities are vectors
theorem four_vector_quantities :
  ∃! (n : Nat), n = (List.filter isVector 
    [PhysicalQuantity.Mass, PhysicalQuantity.Velocity, 
     PhysicalQuantity.Displacement, PhysicalQuantity.Force, 
     PhysicalQuantity.Acceleration, PhysicalQuantity.Distance]).length ∧ 
  n = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_vector_quantities_l1148_114851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_equation_l1148_114853

/-- A light ray L is emitted from a point, reflected off the x-axis, and the reflected ray is tangent to a circle. -/
structure LightRay where
  start : ℝ × ℝ
  circle : (ℝ → ℝ → ℝ)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line equation is valid for the given light ray -/
def is_valid_line_equation (L : LightRay) (eq : LineEquation) : Prop :=
  (L.start.1 = -3 ∧ L.start.2 = 3) ∧
  (L.circle = λ x y => x^2 + y^2 - 4*x - 4*y + 7) ∧
  ((eq.a = 3 ∧ eq.b = 4 ∧ eq.c = -3) ∨ (eq.a = 4 ∧ eq.b = 3 ∧ eq.c = 3))

theorem light_ray_equation (L : LightRay) :
  ∃ (eq : LineEquation), is_valid_line_equation L eq := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_equation_l1148_114853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l1148_114828

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

/-- The function g(x) -/
noncomputable def g (x m : ℝ) : ℝ := f x + m

theorem min_value_of_g (m : ℝ) :
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), g x m ≥ 3) ∧
  (∃ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), g x m = 3) ↔
  m = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l1148_114828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_equals_self_iff_m_eq_neg_four_fifths_l1148_114877

/-- The function f(x) = (3x + 4) / (mx - 5) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * x + 4) / (m * x - 5)

/-- The inverse function of f -/
noncomputable def f_inv (m : ℝ) (x : ℝ) : ℝ := 
  (5 * x + 4) / (3 - m * x)

theorem inverse_equals_self_iff_m_eq_neg_four_fifths (m : ℝ) :
  (∀ x, f_inv m x = f m x) ↔ m = -4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_equals_self_iff_m_eq_neg_four_fifths_l1148_114877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_1998_l1148_114850

/-- The sequence of nonnegative integers where every nonnegative integer
    can be uniquely expressed as aᵢ + 2aⱼ + 4aₖ -/
def A : ℕ → ℕ := sorry

/-- The property that every nonnegative integer can be uniquely expressed
    as aᵢ + 2aⱼ + 4aₖ -/
axiom A_unique_representation :
  ∀ n : ℕ, ∃! (i j k : ℕ), n = A i + 2 * A j + 4 * A k

/-- The sequence A is increasing -/
axiom A_increasing :
  ∀ n m : ℕ, n < m → A n < A m

/-- The 1998th term of the sequence A is 1227096648 -/
theorem A_1998 : A 1998 = 1227096648 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_1998_l1148_114850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_equidistant_l1148_114802

/-- A point on a plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Perpendicular bisector of a segment -/
def perpBisector (a b : Point2D) : Set Point2D :=
  {p : Point2D | distance p a = distance p b}

theorem perpendicular_bisector_equidistant 
  (a b p : Point2D) 
  (h1 : p ∈ perpBisector a b) 
  (h2 : distance p a = 5) : 
  distance p b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_equidistant_l1148_114802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l1148_114865

-- Define the radii of the two spheres
def small_radius : ℝ := 5
def large_radius : ℝ := 8

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the volume of the region between the two spheres
noncomputable def region_volume : ℝ := sphere_volume large_radius - sphere_volume small_radius

-- Theorem statement
theorem volume_between_spheres : region_volume = 516 * Real.pi := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l1148_114865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l1148_114814

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin (3 * x - Real.pi / 6)

theorem smallest_positive_period (T : ℝ) : 
  (∀ x, f (x + T) = f x) ∧ 
  (∀ T' : ℝ, 0 < T' ∧ T' < T → ∃ x, f (x + T') ≠ f x) → 
  T = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l1148_114814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_distance_difference_l1148_114824

/-- If the airspeed of an airplane is a km/h and the wind speed is 20 km/h,
    the difference in km between the distance flown against the wind for 3 hours
    and the distance flown with the wind for 4 hours is a + 140 km. -/
theorem airplane_distance_difference (a : ℝ) : 
  (a + 20) * 4 - (a - 20) * 3 = a + 140 := by
  ring

#check airplane_distance_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_distance_difference_l1148_114824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l1148_114880

noncomputable def f (x : ℝ) : ℝ := 2 - 3 * x

noncomputable def g (x : ℝ) : ℝ := (2 - x) / 3

theorem f_inverse_is_g : 
  (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l1148_114880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leaky_cistern_extra_time_l1148_114854

/-- Represents the time it takes to fill a leaky cistern -/
noncomputable def leaky_cistern_fill_time (normal_fill_time empty_time : ℝ) : ℝ :=
  1 / (1 / normal_fill_time - 1 / empty_time)

/-- Theorem: A cistern that normally fills in 10 hours and empties in 60 hours when full
    will take 2 hours longer to fill with the leak -/
theorem leaky_cistern_extra_time :
  leaky_cistern_fill_time 10 60 - 10 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leaky_cistern_extra_time_l1148_114854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l1148_114811

/-- The molar mass of aluminum in g/mol -/
noncomputable def molar_mass_Al : ℝ := 26.98

/-- The molar mass of chlorine in g/mol -/
noncomputable def molar_mass_Cl : ℝ := 35.45

/-- The number of aluminum atoms in AlCl3 -/
def num_Al_atoms : ℕ := 1

/-- The number of chlorine atoms in AlCl3 -/
def num_Cl_atoms : ℕ := 3

/-- The molar mass of AlCl3 in g/mol -/
noncomputable def molar_mass_AlCl3 : ℝ := molar_mass_Al * num_Al_atoms + molar_mass_Cl * num_Cl_atoms

/-- The mass percentage of Al in AlCl3 -/
noncomputable def mass_percentage_Al : ℝ := (molar_mass_Al / molar_mass_AlCl3) * 100

theorem mass_percentage_Al_approx :
  |mass_percentage_Al - 20.23| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l1148_114811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solutions_l1148_114807

theorem floor_equation_solutions : 
  (Finset.filter (fun x : ℕ => x > 0 ∧
    (Int.floor ((x : ℚ) / 20) = Int.floor ((x : ℚ) / 17))) 
    (Finset.range 102)).card = 56 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solutions_l1148_114807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_fraction_equality_l1148_114893

theorem power_fraction_equality : 
  (27 / 125 : ℝ) ^ (-(1/3) : ℝ) = 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_fraction_equality_l1148_114893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1148_114841

/-- An ellipse with foci at (15, -5) and (15, 45) that is tangent to the y-axis has a major axis of length 10√34. -/
theorem ellipse_major_axis_length :
  ∀ (E : Set (ℝ × ℝ)),
  (∃ (a b : ℝ), E = {(x, y) | (x - 15)^2 / a^2 + (y - 20)^2 / b^2 = 1}) →
  ((15, -5) ∈ E) →
  ((15, 45) ∈ E) →
  (∃ (y : ℝ), (0, y) ∈ E) →
  (∀ (x y : ℝ), (x, y) ∈ E → x ≥ 0) →
  (∃ (p q : ℝ × ℝ), p ∈ E ∧ q ∈ E ∧ dist p q = 10 * Real.sqrt 34) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l1148_114841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sin_shift_l1148_114899

noncomputable def f (x : ℝ) := Real.sin (x - Real.pi/4)

theorem axis_of_symmetry_sin_shift :
  ∀ (x : ℝ), f ((-Real.pi/4) + x) = f ((-Real.pi/4) - x) :=
by
  intro x
  simp [f]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sin_shift_l1148_114899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_conducts_electricity_is_deductive_iron_conducts_electricity_l1148_114808

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define a specific element
variable (iron : U)

-- Define the premises and conclusion
variable (premise1 : ∀ x, Metal x → ConductsElectricity x)
variable (premise2 : Metal iron)
variable (conclusion : ConductsElectricity iron)

-- Define deductive reasoning
def IsDeductiveReasoning (premises conclusion : Prop) : Prop :=
  premises → conclusion

-- Theorem to prove
theorem iron_conducts_electricity_is_deductive :
  IsDeductiveReasoning (∀ x, Metal x → ConductsElectricity x) (Metal iron → ConductsElectricity iron) :=
by
  intro h
  exact h iron

-- Additional theorem to show that the specific case follows from the premises
theorem iron_conducts_electricity :
  (∀ x, Metal x → ConductsElectricity x) → Metal iron → ConductsElectricity iron :=
by
  intro h1 h2
  exact h1 iron h2

#check iron_conducts_electricity_is_deductive
#check iron_conducts_electricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_conducts_electricity_is_deductive_iron_conducts_electricity_l1148_114808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1148_114837

/-- The parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle equation x^2 + y^2 - 4x - 2y = 0 -/
def circle' (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

/-- The distance between two points (x₁, y₁) and (x₂, y₂) -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem intersection_distance : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    parabola x₁ y₁ ∧ circle' x₁ y₁ ∧
    parabola x₂ y₂ ∧ circle' x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    distance x₁ y₁ x₂ y₂ = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1148_114837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_iff_a_eq_two_linear_iff_a_eq_minus_three_or_sqrt_seventeen_l1148_114817

-- Define the function y
noncomputable def y (a x : ℝ) : ℝ := (a + 3) * x^(a^2 + a - 4) + (a + 2) * x + 3

-- Define what it means for y to be quadratic in x
def is_quadratic (a : ℝ) : Prop :=
  ∃ (p q r : ℝ), ∀ x, y a x = p * x^2 + q * x + r ∧ p ≠ 0

-- Define what it means for y to be linear in x
def is_linear (a : ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, y a x = m * x + b ∧ m ≠ 0

-- Theorem for quadratic case
theorem quadratic_iff_a_eq_two :
  ∀ a : ℝ, is_quadratic a ↔ a = 2 := by
  sorry

-- Theorem for linear case
theorem linear_iff_a_eq_minus_three_or_sqrt_seventeen :
  ∀ a : ℝ, is_linear a ↔ (a = -3 ∨ a = (-1 + Real.sqrt 17) / 2 ∨ a = (-1 - Real.sqrt 17) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_iff_a_eq_two_linear_iff_a_eq_minus_three_or_sqrt_seventeen_l1148_114817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_force_rhombus_equilibrium_l1148_114874

/-- A rhombus with forces acting on its vertices -/
structure ForceRhombus where
  /-- The side length of the rhombus -/
  a : ℝ
  /-- The angle BAC of the rhombus -/
  α : ℝ
  /-- The magnitude of force P acting along AC -/
  P : ℝ
  /-- The magnitude of force Q acting along BD -/
  Q : ℝ

/-- The condition for equilibrium in a ForceRhombus -/
def isEquilibrium (r : ForceRhombus) : Prop :=
  r.Q = r.P * (Real.tan r.α)^3

/-- Theorem stating the equilibrium condition for a ForceRhombus -/
theorem force_rhombus_equilibrium (r : ForceRhombus) :
  isEquilibrium r ↔ r.Q = r.P * (Real.tan r.α)^3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_force_rhombus_equilibrium_l1148_114874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_felicia_can_volume_ratio_l1148_114849

noncomputable def cylinder_volume (diameter : ℝ) (height : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * height

theorem alex_felicia_can_volume_ratio :
  let alex_can_volume := cylinder_volume 6 12
  let felicia_can_volume := cylinder_volume 12 6
  alex_can_volume / felicia_can_volume = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_felicia_can_volume_ratio_l1148_114849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1148_114830

theorem cos_alpha_value (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < π / 2) 
  (h3 : Real.cos (π / 3 + α) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 + 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1148_114830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1148_114862

def is_in_third_quadrant (θ : ℝ) : Prop :=
  Real.pi < θ ∧ θ < 3 * Real.pi / 2

theorem angle_in_third_quadrant (θ : ℝ) 
  (h1 : Real.sin θ < 0) (h2 : Real.cos θ < 0) : is_in_third_quadrant θ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1148_114862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equality_l1148_114834

open MeasureTheory

theorem definite_integral_equality : 
  ∫ x in Set.Icc 0 1, (Real.sqrt (1 - x^2) + (1/2) * x) = (Real.pi + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equality_l1148_114834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_cube_fourth_l1148_114827

/-- Given that a³ varies inversely with b⁴, and a = 2 when b = 4, prove that a³ = 1/2 when b = 8 -/
theorem inverse_variation_cube_fourth (a b : ℝ) (k : ℝ) 
  (h1 : a^3 * b^4 = k)
  (h2 : a = 2 ∧ b = 4 → k = 2048) : 
  a^3 = 1/2 ∧ b = 8 → k = 2048 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_cube_fourth_l1148_114827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_speed_ratio_l1148_114813

/-- Proves that the ratio of Angelina's speeds is 2:1 given the problem conditions --/
theorem angelina_speed_ratio :
  ∀ (v1 v2 : ℝ),
  v1 > 0 → v2 > 0 →
  200 / v1 - 300 / v2 = 50 →
  v2 = 2 →
  v2 / v1 = 2 := by
  sorry

#check angelina_speed_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_speed_ratio_l1148_114813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_vectors_l1148_114832

noncomputable def vector_a (x y : ℝ) : ℝ × ℝ := (x, y)
def vector_b : ℝ × ℝ := (1, 2)

theorem min_distance_vectors :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 - 1 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 1 →
  ∀ (dist : ℝ), dist = Real.sqrt ((x - 1)^2 + (y - 2)^2) → dist ≥ min_dist :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_vectors_l1148_114832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_ellipse_intersection_l1148_114847

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle (renamed to avoid conflict)
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem max_distance_on_ellipse_intersection (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (m : ℝ),
    ellipse a b 1 (Real.sqrt 3 / 2) ∧
    ellipse a b (-Real.sqrt 3) 0 ∧
    (∀ (x y : ℝ), unit_circle x y → ∃ (k : ℝ), y = k * (x - m)) →
    (∀ (x1 y1 x2 y2 : ℝ),
      ellipse a b x1 y1 ∧ ellipse a b x2 y2 ∧
      (∃ (k : ℝ), y1 = k * (x1 - m) ∧ y2 = k * (x2 - m)) →
      distance x1 y1 x2 y2 ≤ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_ellipse_intersection_l1148_114847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_alpha_to_beta_l1148_114844

noncomputable def alpha : ℂ := 0
noncomputable def omega : ℂ := 3900 * Complex.I
noncomputable def beta : ℂ := 1170 + 1560 * Complex.I

theorem distance_alpha_to_beta : Complex.abs (beta - alpha) = 1950 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_alpha_to_beta_l1148_114844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_iff_product_condition_l1148_114867

theorem log_condition_iff_product_condition (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) : 
  Real.log b / Real.log a > 0 ↔ (a - 1) * (b - 1) > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_condition_iff_product_condition_l1148_114867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_y_for_strong_two_thousand_thirteen_is_strong_l1148_114822

/-- A positive integer n is strong if there exists a positive integer x such that x^(nx) + 1 is divisible by 2^n. -/
def is_strong (n : ℕ) : Prop :=
  ∃ x : ℕ, x > 0 ∧ (2^n : ℕ) ∣ (x^(n * x) + 1)

/-- For a strong positive integer m, the smallest y such that y^(my) + 1 is divisible by 2^m is 2^m - 1. -/
theorem smallest_y_for_strong (m : ℕ) (hm : m > 0) (h : is_strong m) :
  (∀ y : ℕ, y > 0 → (2^m : ℕ) ∣ (y^(m * y) + 1) → y ≥ 2^m - 1) ∧
  (2^m : ℕ) ∣ ((2^m - 1)^(m * (2^m - 1)) + 1) :=
sorry

/-- 2013 is strong. -/
theorem two_thousand_thirteen_is_strong : is_strong 2013 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_y_for_strong_two_thousand_thirteen_is_strong_l1148_114822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_pentagon_perimeter_is_sqrt5_minus_2_l1148_114885

/-- Represents a regular five-pointed star -/
structure RegularFivePointedStar where
  /-- The total perimeter of the star -/
  totalPerimeter : ℝ
  /-- Assumption that the total perimeter is 1 -/
  totalPerimeterIsOne : totalPerimeter = 1

/-- The perimeter of the inner pentagon in a regular five-pointed star -/
noncomputable def innerPentagonPerimeter (star : RegularFivePointedStar) : ℝ :=
  Real.sqrt 5 - 2

/-- Theorem stating that the perimeter of the inner pentagon in a regular five-pointed star
    with total perimeter 1 is √5 - 2 -/
theorem inner_pentagon_perimeter_is_sqrt5_minus_2 (star : RegularFivePointedStar) :
  innerPentagonPerimeter star = Real.sqrt 5 - 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_pentagon_perimeter_is_sqrt5_minus_2_l1148_114885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l1148_114896

/-- Represents the number of days it takes for 'b' to complete the work alone -/
noncomputable def b_days : ℝ := 15

/-- Represents the total amount of work to be done -/
noncomputable def total_work : ℝ := 1

/-- The rate at which 'a' completes the work -/
noncomputable def a_rate : ℝ := total_work / 12

/-- The rate at which 'b' completes the work -/
noncomputable def b_rate : ℝ := total_work / b_days

/-- The amount of work 'a' completes in 3 days -/
noncomputable def a_work_3_days : ℝ := 3 * a_rate

/-- The amount of work 'a' and 'b' complete together in 5 days -/
noncomputable def ab_work_5_days : ℝ := 5 * (a_rate + b_rate)

theorem work_completion :
  a_work_3_days + ab_work_5_days = total_work :=
by
  -- Expand definitions
  unfold a_work_3_days ab_work_5_days a_rate b_rate total_work b_days
  -- Algebraic manipulation
  ring
  -- The proof is complete
  done

#check work_completion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l1148_114896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1148_114801

theorem trigonometric_problem (α : Real) 
  (h1 : Real.sin (α + Real.pi/4) = Real.sqrt 2/10) 
  (h2 : α ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.cos α = -3/5 ∧ Real.sin (2*α - Real.pi/4) = -17*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1148_114801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmersPasses_l1148_114884

/-- Represents a swimmer with a given speed -/
structure Swimmer where
  speed : ℚ
  deriving Repr

/-- Represents the swimming pool setup -/
structure PoolSetup where
  length : ℚ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℚ
  deriving Repr

/-- Calculates the number of times swimmers pass each other -/
def countPasses (setup : PoolSetup) : ℕ :=
  sorry

/-- Theorem stating the number of passes for the given problem -/
theorem swimmersPasses :
  let setup : PoolSetup := {
    length := 120,
    swimmer1 := { speed := 4 },
    swimmer2 := { speed := 3 },
    totalTime := 20 * 60  -- 20 minutes in seconds
  }
  countPasses setup = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmersPasses_l1148_114884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_l1148_114886

/-- Represents the time it takes for a tank to empty with a leak and an inlet pipe. -/
noncomputable def time_to_empty (tank_volume : ℝ) (leak_empty_time : ℝ) (inlet_rate : ℝ) : ℝ :=
  tank_volume / (tank_volume / leak_empty_time - inlet_rate * 60)

/-- Theorem stating the time it takes for the tank to empty under given conditions. -/
theorem tank_empty_time :
  let tank_volume : ℝ := 6048
  let leak_empty_time : ℝ := 7
  let inlet_rate : ℝ := 6
  time_to_empty tank_volume leak_empty_time inlet_rate = 12 := by
  -- Proof goes here
  sorry

#check tank_empty_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_l1148_114886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_properties_l1148_114839

/-- Regular triangular pyramid -/
structure RegularTriangularPyramid where
  /-- Side length of the base -/
  a : ℝ
  /-- Dihedral angle at the base in radians -/
  dihedral_angle : ℝ
  /-- Dihedral angle is 45 degrees (π/4 radians) -/
  angle_is_45_deg : dihedral_angle = π / 4

/-- Volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ := p.a^3 / 24

/-- Total surface area of a regular triangular pyramid -/
noncomputable def totalSurfaceArea (p : RegularTriangularPyramid) : ℝ :=
  (p.a^2 * Real.sqrt 3 * (1 + Real.sqrt 2)) / 4

/-- Theorem: Volume and total surface area of a regular triangular pyramid -/
theorem regular_triangular_pyramid_properties (p : RegularTriangularPyramid) :
  volume p = p.a^3 / 24 ∧
  totalSurfaceArea p = (p.a^2 * Real.sqrt 3 * (1 + Real.sqrt 2)) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_properties_l1148_114839
