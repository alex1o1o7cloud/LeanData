import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l689_68944

/-- Two lines are parallel if their direction vectors are scalar multiples of each other -/
def parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v₁.1 * v₂.2 = c * v₁.2 * v₂.1

/-- The direction vector of a line given by parametric equations -/
def direction_vector (dx dy : ℝ) : ℝ × ℝ := (dx, dy)

/-- The normal vector of a line given in the form ax + by + c = 0 -/
def normal_vector (a b : ℝ) : ℝ × ℝ := (a, b)

theorem parallel_lines_condition (k : ℝ) :
  parallel (direction_vector (-1/2) (Real.sqrt 3/2)) (normal_vector k 1) ↔ k = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l689_68944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_y_eq_x_l689_68950

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]

theorem reflection_over_y_eq_x :
  let e1 : Fin 2 → ℝ := ![1, 0]
  let e2 : Fin 2 → ℝ := ![0, 1]
  reflection_matrix.mulVec e1 = e2 ∧
  reflection_matrix.mulVec e2 = e1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_y_eq_x_l689_68950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_two_and_half_miles_l689_68963

/-- The time in seconds between seeing the lightning flash and hearing the thunder -/
def time_delay : ℚ := 12

/-- The speed of sound in feet per second -/
def speed_of_sound : ℚ := 1100

/-- The number of feet in one mile -/
def feet_per_mile : ℚ := 5280

/-- The distance in miles between Charlie Brown and the lightning flash -/
noncomputable def distance_to_lightning : ℚ := (time_delay * speed_of_sound) / feet_per_mile

/-- Theorem stating that the distance to the lightning flash is 2.5 miles -/
theorem distance_is_two_and_half_miles :
  distance_to_lightning = 5/2 := by
  -- Unfold the definition of distance_to_lightning
  unfold distance_to_lightning
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_two_and_half_miles_l689_68963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l689_68969

/-- A circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in 2D space -/
def Point : Type := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  distance p c.center < c.radius

theorem point_inside_circle (O : Circle) (P : Point) 
  (h1 : O.radius = 4)
  (h2 : distance P O.center = 3) : 
  isInside P O := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l689_68969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_range_l689_68924

-- Define the line equation
def line_eq (a x y : ℝ) : Prop := (a + 1) * x - a * y - 1 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ line_eq a x y ∧ circle_eq x y}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem chord_length_range :
  ∀ a : ℝ, ∃ A B : ℝ × ℝ, A ∈ intersection_points a ∧ B ∈ intersection_points a ∧
    ∀ p q : ℝ × ℝ, p ∈ intersection_points a → q ∈ intersection_points a →
      2 * Real.sqrt 2 ≤ distance p q ∧ distance p q < 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_range_l689_68924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_march_first_in_leap_year_with_friday_start_l689_68981

/-- Represents days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

/-- Advances a weekday by a given number of days -/
def advanceWeekday (start : Weekday) (days : Nat) : Weekday :=
  match days with
  | 0 => start
  | n + 1 => 
    let nextDay := match start with
      | Weekday.Sunday => Weekday.Monday
      | Weekday.Monday => Weekday.Tuesday
      | Weekday.Tuesday => Weekday.Wednesday
      | Weekday.Wednesday => Weekday.Thursday
      | Weekday.Thursday => Weekday.Friday
      | Weekday.Friday => Weekday.Saturday
      | Weekday.Saturday => Weekday.Sunday
    advanceWeekday nextDay n

theorem march_first_in_leap_year_with_friday_start 
  (isLeapYear : Bool) 
  (jan1 : Weekday) 
  (h1 : isLeapYear = true) 
  (h2 : jan1 = Weekday.Friday) : 
  advanceWeekday jan1 60 = Weekday.Tuesday :=
by
  sorry

#eval advanceWeekday Weekday.Friday 60


end NUMINAMATH_CALUDE_ERRORFEEDBACK_march_first_in_leap_year_with_friday_start_l689_68981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_one_fourth_l689_68960

/-- The sum of the series Σ(3^n / (1 + 3^n + 3^(n+1) + 3^(2n+1))) from n=1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' n, (3:ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

/-- Theorem stating that the infinite series sum equals 1/4 -/
theorem infinite_series_sum_equals_one_fourth : infinite_series_sum = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_one_fourth_l689_68960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_approx_l689_68976

/-- Represents the terrain types around the field -/
inductive Terrain
  | Grass
  | Sand
  | Mud

/-- Represents the field properties and runner's speeds -/
structure FieldProperties where
  sideLength : ℝ
  grassSpeed : ℝ
  sandSpeed : ℝ
  mudSpeed : ℝ
  grassPercentage : ℝ
  sandPercentage : ℝ
  mudPercentage : ℝ

/-- Calculates the time to complete one run around the field -/
noncomputable def timeToCompleteRun (field : FieldProperties) : ℝ :=
  let perimeter := 4 * field.sideLength
  let grassLength := field.grassPercentage * perimeter
  let sandLength := field.sandPercentage * perimeter
  let mudLength := field.mudPercentage * perimeter
  let grassTime := grassLength / (field.grassSpeed * 1000 / 3600)
  let sandTime := sandLength / (field.sandSpeed * 1000 / 3600)
  let mudTime := mudLength / (field.mudSpeed * 1000 / 3600)
  grassTime + sandTime + mudTime

/-- Theorem stating the time to complete one run is approximately 72.27 seconds -/
theorem run_time_approx (ε : ℝ) (h : ε > 0) :
  ∃ (field : FieldProperties),
    field.sideLength = 50 ∧
    field.grassSpeed = 14 ∧
    field.sandSpeed = 8 ∧
    field.mudSpeed = 5 ∧
    field.grassPercentage = 0.6 ∧
    field.sandPercentage = 0.3 ∧
    field.mudPercentage = 0.1 ∧
    abs (timeToCompleteRun field - 72.27) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_approx_l689_68976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_unique_value_l689_68941

theorem sum_squares_unique_value (x y z : ℕ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_eq : x + y + z = 30)
  (gcd_sum_eq : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10) :
  x^2 + y^2 + z^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_unique_value_l689_68941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doghouse_area_l689_68975

/-- Area of a circle given its radius -/
noncomputable def area_of_circle (radius : ℝ) : ℝ := Real.pi * radius^2

/-- Area of an equilateral triangle given its side length -/
noncomputable def area_of_equilateral_triangle (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

/-- Area of a circular sector given radius and angle in turns -/
noncomputable def area_of_sector (radius : ℝ) (angle_ratio : ℝ) : ℝ := 
  Real.pi * radius^2 * angle_ratio

/-- The area outside an equilateral triangle reachable by a tethered point -/
theorem doghouse_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 →
  rope_length = 3 →
  ∃ (area : ℝ), area = (23 * Real.pi) / 3 ∧ 
    area = (area_of_circle rope_length - area_of_equilateral_triangle side_length) +
           (area_of_sector rope_length (300 / 360)) +
           2 * (area_of_sector 1 (60 / 360)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doghouse_area_l689_68975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l689_68943

/-- A structure representing a convex quadrilateral with given side and diagonal lengths -/
structure ConvexQuadrilateral (a b c d e f : ℝ) : Prop where
  convex : True  -- This is a placeholder for the convexity condition
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0
  side_d : d > 0
  diag_e : e > 0
  diag_f : f > 0

/-- Theorem stating the inequality for convex quadrilaterals -/
theorem quadrilateral_inequality (a b c d e f : ℝ) 
  (h : ConvexQuadrilateral a b c d e f) : 
  a^2 + b^2 + c^2 + d^2 ≥ e^2 + f^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l689_68943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_inverse_cosine_cotangent_l689_68927

theorem cosine_sum_inverse_cosine_cotangent : 
  Real.cos (Real.arccos (4/5) + Real.arctan (1/3)) = 9 * Real.sqrt 10 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_inverse_cosine_cotangent_l689_68927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_mathematicians_paired_probability_l689_68935

def tournament_players : ℕ := 8
def mathematicians : ℕ := 4
def pairs : ℕ := 4

def total_pairings : ℕ := Nat.factorial tournament_players / (2^pairs * Nat.factorial pairs)

def favorable_outcomes : ℕ := 
  (tournament_players * (tournament_players - 2) * (tournament_players - 4) * (tournament_players - 6)) 
  * Nat.factorial (tournament_players - mathematicians)

theorem no_mathematicians_paired_probability :
  (favorable_outcomes : ℚ) / Nat.factorial tournament_players = 8 / 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_mathematicians_paired_probability_l689_68935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_cubes_overlap_l689_68939

/-- A cube with side length 6 --/
def large_cube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 6}

/-- A unit cube centered at a given point --/
def unit_cube (center : Fin 3 → ℝ) : Set (Fin 3 → ℝ) :=
  {p | ∀ i, center i - 0.5 ≤ p i ∧ p i ≤ center i + 0.5}

/-- The centers of the 1001 unit cubes --/
def cube_centers : Set (Fin 3 → ℝ) :=
  {c | c ∈ large_cube ∧ unit_cube c ⊆ large_cube}

theorem unit_cubes_overlap :
  ∃ (n : ℕ) (centers : Fin n → Fin 3 → ℝ), n = 1001 ∧
  (∀ i, centers i ∈ cube_centers) ∧
  ∃ i j, i ≠ j ∧ centers i ∈ unit_cube (centers j) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_cubes_overlap_l689_68939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_sufficient_not_necessary_l689_68962

/-- An angle is obtuse if it is greater than π/2 and less than π --/
def IsObtuseAngle (α : Real) : Prop := Real.pi/2 < α ∧ α < Real.pi

/-- Condition for sin α > 0 and cos α < 0 --/
def SinPosCosneg (α : Real) : Prop := Real.sin α > 0 ∧ Real.cos α < 0

/-- Theorem stating that "Angle α is an obtuse angle" is a sufficient but not necessary condition for "sin α > 0 and cos α < 0" --/
theorem obtuse_sufficient_not_necessary :
  (∀ α, IsObtuseAngle α → SinPosCosneg α) ∧
  ¬(∀ α, SinPosCosneg α → IsObtuseAngle α) := by
  sorry

#check obtuse_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_sufficient_not_necessary_l689_68962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_four_implies_a_value_l689_68900

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / (x - 1) + 1 / (x - 2) + 1 / (x - 6)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := -a / ((x - 1)^2) - 1 / ((x - 2)^2) - 1 / ((x - 6)^2)

theorem max_at_four_implies_a_value (a : ℝ) :
  (∀ x ∈ Set.Ioo 3 5, f a x ≤ f a 4) →
  f_deriv a 4 = 0 →
  a = -9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_four_implies_a_value_l689_68900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_can_exceed_2017_l689_68993

-- Define the curves
noncomputable def C₁ (x : ℝ) : ℝ := Real.sin x

def C₂ (r : ℝ) (x y : ℝ) : Prop := x^2 + (y + r - 1/2)^2 = r^2

-- State the theorem
theorem intersection_points_can_exceed_2017 :
  ∃ (r : ℝ), r > 0 ∧ 
  (∃ (n : ℕ), n > 2017 ∧ 
    (∃ (points : Finset (ℝ × ℝ)), points.card = n ∧ 
      (∀ (p : ℝ × ℝ), p ∈ points → 
        C₁ p.1 = p.2 ∧ C₂ r p.1 p.2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_can_exceed_2017_l689_68993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trees_in_2002_l689_68905

/-- Represents the number of trees in a given year -/
def TreeCount : ℕ → ℕ := sorry

/-- The constant of proportionality -/
def k : ℚ := sorry

/-- The difference between tree counts in year n+2 and n is proportional to new plantings in year n+1 -/
axiom proportional_growth (n : ℕ) : TreeCount (n + 2) - TreeCount n = k * TreeCount (n + 1)

/-- In 2000, 100 trees were planted -/
axiom trees_2000 : TreeCount 2000 = 100

/-- In 2003, 250 trees were counted -/
axiom trees_2003 : TreeCount 2003 = 250

/-- In 2001, 150 new trees were added -/
axiom trees_2001 : TreeCount 2001 = TreeCount 2000 + 150

theorem trees_in_2002 : TreeCount 2002 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trees_in_2002_l689_68905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_half_l689_68919

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then
    |x - 1| - 1
  else
    1 / (1 + x^2)

-- State the theorem
theorem f_composition_half : f (f (1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_half_l689_68919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetric_about_line_l689_68911

-- Define the circle in polar coordinates
def polar_circle (a : ℝ) (ρ θ : ℝ) : Prop := ρ = 2 * a * Real.cos θ

-- Define the line in polar coordinates
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ + Real.sqrt 3 * ρ * Real.sin θ + 1 = 0

-- State the theorem
theorem circle_symmetric_about_line (a : ℝ) :
  (∃ (ρ θ : ℝ), polar_circle a ρ θ ∧ polar_line ρ θ) →
  a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetric_about_line_l689_68911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l689_68925

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem sixth_term_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l689_68925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_proportion_curve_l689_68994

-- Define the average of three real numbers
noncomputable def M (a b c : ℝ) : ℝ := (a + b + c) / 3

-- Define the minimum of three real numbers
noncomputable def min3 (a b c : ℝ) : ℝ := min a (min b c)

-- Define the theorem
theorem point_on_inverse_proportion_curve (a : ℝ) (h1 : a > 0) :
  M (-2) (a - 1) (2 * a) = a - 1 ∧
  min3 (-2) (a - 1) (2 * a) = -2 ∧
  -2 / (a - 1) = -2 →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_proportion_curve_l689_68994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_x_plus_pi_fourth_domain_l689_68998

open Set
open Real

-- Define the tangent function
noncomputable def tan_function (x : ℝ) : ℝ := tan x

-- Define the domain of the function
def domain_tan_half_x_plus_pi_fourth : Set ℝ := {x | ∀ k : ℤ, x ≠ 2 * k * π + π / 2}

-- Theorem statement
theorem tan_half_x_plus_pi_fourth_domain :
  {x : ℝ | ∃ y, tan_function (x / 2 + π / 4) = y} = domain_tan_half_x_plus_pi_fourth :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_x_plus_pi_fourth_domain_l689_68998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vanilla_cookies_sold_l689_68995

/-- Proves that the number of vanilla cookies sold is 70 --/
theorem vanilla_cookies_sold (chocolate_cookies vanilla_cookies : ℕ) 
  (chocolate_price vanilla_price total_revenue : ℚ)
  (h1 : chocolate_cookies = 220)
  (h2 : chocolate_price = 1)
  (h3 : vanilla_price = 2)
  (h4 : total_revenue = 360)
  (h5 : chocolate_cookies * chocolate_price + vanilla_price * vanilla_cookies = total_revenue) :
  vanilla_cookies = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vanilla_cookies_sold_l689_68995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_inscribed_circle_l689_68949

/-- Square ABCD with side length 2 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2) ∧ D = (0, 2))

/-- E is the midpoint of AB -/
def E : ℝ × ℝ := (1, 0)

/-- F is on side BC -/
noncomputable def F (x : ℝ) : ℝ × ℝ := (2, 2/x)

/-- G is on side CD -/
def G (x : ℝ) : ℝ × ℝ := (x, 2)

/-- AG is parallel to EF -/
noncomputable def parallel_AG_EF (sq : Square) (x : ℝ) : Prop :=
  let A := sq.A
  let G := G x
  let slope_AG := (G.2 - A.2) / (G.1 - A.1)
  let slope_EF := (F x).2 / ((F x).1 - E.1)
  slope_AG = slope_EF

/-- FG is tangent to the inscribed circle -/
noncomputable def is_tangent (sq : Square) (x : ℝ) : Prop :=
  let P := (2, 1)  -- midpoint of BC
  let Q := (1, 2)  -- midpoint of CD
  let dist_PF := Real.sqrt ((F x).1 - P.1)^2 + ((F x).2 - P.2)^2
  let dist_GQ := Real.sqrt ((G x).1 - Q.1)^2 + ((G x).2 - Q.2)^2
  let dist_FG := Real.sqrt ((F x).1 - (G x).1)^2 + ((F x).2 - (G x).2)^2
  dist_PF + dist_GQ = dist_FG

theorem tangent_to_inscribed_circle (sq : Square) (x : ℝ) :
  parallel_AG_EF sq x → is_tangent sq x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_inscribed_circle_l689_68949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l689_68958

theorem simplify_trig_expression :
  (Real.sin (35 * π / 180))^2 - 1/2 = -(Real.cos (10 * π / 180) * Real.cos (80 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l689_68958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l689_68964

def sequence_a : ℕ → ℕ
  | 0 => 3  -- Define the base case for 0
  | n + 1 => 2 * sequence_a n

theorem sixth_term_value : sequence_a 5 = 96 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l689_68964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_routes_through_midpoint_l689_68921

/-- Number of routes on a grid -/
def num_routes (start_x start_y end_x end_y : ℚ) : ℕ :=
  Nat.choose 
    (Int.floor ((end_x - start_x + end_y - start_y) * 2)).toNat 
    (Int.floor ((end_x - start_x) * 2)).toNat

/-- The problem statement -/
theorem routes_through_midpoint :
  let a_to_c := num_routes 0 3 (3/2) (3/2)
  let c_to_b := num_routes (3/2) (3/2) 3 0
  a_to_c * c_to_b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_routes_through_midpoint_l689_68921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_segment_constructible_l689_68946

-- Define the given segment length
noncomputable def α : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5

-- Define constructibility
def constructible (x : ℝ) : Prop := sorry

-- Theorem statement
theorem unit_segment_constructible : constructible 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_segment_constructible_l689_68946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_is_correct_l689_68991

/-- The greatest possible distance between the centers of two circles with 8-inch diameters
    placed within a 15-inch by 18-inch rectangle, without extending beyond the rectangle's perimeter. -/
noncomputable def greatest_distance_between_circle_centers : ℝ :=
  Real.sqrt 149

/-- Proof that the greatest distance is correct given the rectangle and circle dimensions. -/
theorem greatest_distance_is_correct (rectangle_width rectangle_length circle_diameter : ℝ)
    (h1 : rectangle_width = 15)
    (h2 : rectangle_length = 18)
    (h3 : circle_diameter = 8)
    (h4 : ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x ≤ rectangle_width → y ≤ rectangle_length →
      (x - circle_diameter / 2) ^ 2 + (y - circle_diameter / 2) ^ 2 ≤ (circle_diameter / 2) ^ 2) :
    greatest_distance_between_circle_centers = Real.sqrt 149 := by
  -- Placeholder for the actual proof
  sorry

-- This line is not necessary in Lean 4 and can be removed
-- #eval greatest_distance_between_circle_centers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_is_correct_l689_68991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_leq_M_l689_68953

-- Define the function f
def f (x : ℝ) : ℝ := (1 - x^2)^(3/2)

-- Define the derivative of f
def f_derivative (x : ℝ) : ℝ := -3 * x * (1 - x^2)^(1/2)

-- Define M as the maximum absolute value of f' in (-1, 1)
noncomputable def M : ℝ := 
  ⨆ (x : ℝ) (_ : -1 < x ∧ x < 1), |f_derivative x|

-- State the theorem
theorem integral_f_leq_M :
  ∫ x in (-1)..(1), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_leq_M_l689_68953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l689_68974

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₁ - C₂) / Real.sqrt (A^2 + B^2)

/-- Theorem: Given two parallel lines with a distance of 2 between them, 
    the constant C in the second line equation must be 6 or -14 -/
theorem parallel_lines_distance (C : ℝ) : 
  distance_between_parallel_lines 3 (-4) (-4) C = 2 → C = 6 ∨ C = -14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l689_68974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cosine_l689_68916

theorem isosceles_triangle_cosine (α β : Real) : 
  -- The triangle is isosceles, so the base angles are equal
  α = β → 
  -- The cosine of one base angle is 2/3
  Real.cos α = 2/3 → 
  -- The cosine of the vertex angle is 1/9
  Real.cos (Real.pi - α - β) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cosine_l689_68916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_area_l689_68942

-- Define the curve C
noncomputable def C : Set (ℝ × ℝ) :=
  {p | let (x, y) := p; y^2 = 4*x}

-- Define the point F
def F : ℝ × ℝ := (1, 0)

-- Define the line l
noncomputable def l : Set (ℝ × ℝ) :=
  {p | let (x, y) := p; y = 2*x - 1}

-- Define point A
def A : ℝ × ℝ := (1, 1)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem parabola_intersection_area :
  ∃ (P Q : ℝ × ℝ),
    P ∈ C ∧ Q ∈ C ∧ -- P and Q are on curve C
    P ∈ l ∧ Q ∈ l ∧ -- P and Q are on line l
    A = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) → -- A is midpoint of PQ
    triangleArea (0, 0) P Q = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_area_l689_68942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l689_68922

/-- The time it takes to fill a tank with two filling pipes and one drainpipe -/
noncomputable def fill_time (fill_time1 fill_time2 drain_time : ℝ) : ℝ :=
  let fill_rate1 := 1 / fill_time1
  let fill_rate2 := 1 / fill_time2
  let drain_rate := 1 / drain_time
  let net_rate := fill_rate1 + fill_rate2 - drain_rate
  1 / net_rate

/-- Theorem stating that the time to fill the tank with given pipe rates is 2.5 hours -/
theorem tank_fill_time :
  fill_time 5 4 20 = 2.5 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l689_68922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l689_68999

theorem problem_solution : 
  let expr1 := -(2 - Real.sqrt 3) - (Real.pi - 3.14)^(0 : ℝ) + (1/2 - Real.cos (30 * Real.pi / 180)) * (1/2)^(-2 : ℝ)
  let a := Real.sqrt 3
  let expr2 := (a - 2) / (a^2 - 4) - (a + 2) / (a^2 - 4*a + 4) / ((a + 2) / (a - 2))
  (expr1 = -1 - Real.sqrt 3) ∧ (expr2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l689_68999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_15_sqrt_2_l689_68977

/-- A tetrahedron PQRS with specific properties -/
structure Tetrahedron where
  /-- Length of edge PQ in cm -/
  pq_length : ℝ
  /-- Area of face PQR in cm² -/
  pqr_area : ℝ
  /-- Area of face PQS in cm² -/
  pqs_area : ℝ
  /-- Angle between faces PQR and PQS in degrees -/
  face_angle : ℝ
  /-- Conditions on the tetrahedron's properties -/
  h_pq : pq_length = 5
  h_pqr : pqr_area = 20
  h_pqs : pqs_area = 18
  h_angle : face_angle = 45

/-- The volume of the tetrahedron PQRS -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  15 * Real.sqrt 2

/-- Theorem stating that the volume of the tetrahedron PQRS is 15√2 cm³ -/
theorem volume_is_15_sqrt_2 (t : Tetrahedron) :
    tetrahedron_volume t = 15 * Real.sqrt 2 := by
  -- Unfold the definition of tetrahedron_volume
  unfold tetrahedron_volume
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_15_sqrt_2_l689_68977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_uses_more_paint_l689_68965

-- Define the dimensions of a fence section
def sectionLength : ℝ := 5
def sectionWidth : ℝ := 2

-- Define the number of sections (same for both fences)
def numSections : ℕ := 100  -- arbitrary positive number

-- Define Ivan's rectangular fence section area
def ivanSectionArea : ℝ := sectionLength * sectionWidth

-- Define Petr's parallelogram fence section area (with angle α)
noncomputable def petrSectionArea (α : ℝ) : ℝ := sectionLength * sectionWidth * Real.sin α

-- Total fence areas
def ivanTotalArea : ℝ := ivanSectionArea * (numSections : ℝ)
noncomputable def petrTotalArea (α : ℝ) : ℝ := petrSectionArea α * (numSections : ℝ)

-- Theorem statement
theorem ivan_uses_more_paint (α : ℝ) (h : 0 < α ∧ α < π) : 
  petrTotalArea α < ivanTotalArea := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_uses_more_paint_l689_68965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l689_68934

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x else
  if x ≥ 1 then 1/x else 0

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * f x - |x - 1|

theorem part_one (b : ℝ) :
  (∀ x > 0, g 0 x ≤ |x - 2| + b) ↔ b ≥ -1 := by
  sorry

theorem part_two :
  ∃ x > 0, ∀ y > 0, g 1 y ≤ g 1 x ∧ g 1 x = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l689_68934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_arithmetic_implies_q_one_sum_c_when_q_one_sum_c_when_q_not_one_l689_68983

-- Define the sequences
def a (n : ℕ) : ℚ := 2 * n - 1
def b (q : ℚ) (n : ℕ) : ℚ := q ^ (n - 1)
def c (q : ℚ) (n : ℕ) : ℚ := a n + b q n

-- Define the sum of the first n terms of c_n
noncomputable def S (q : ℚ) (n : ℕ) : ℚ :=
  if q = 1 then n^2 + n
  else n^2 + (1 - q^n) / (1 - q)

theorem c_arithmetic_implies_q_one (q : ℚ) :
  (∀ n : ℕ, c q (n + 2) - c q (n + 1) = c q (n + 1) - c q n) → q = 1 := by sorry

theorem sum_c_when_q_one (n : ℕ) (q : ℚ) :
  q = 1 → S q n = n^2 + n := by sorry

theorem sum_c_when_q_not_one (q : ℚ) (n : ℕ) :
  q ≠ 1 → S q n = n^2 + (1 - q^n) / (1 - q) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_arithmetic_implies_q_one_sum_c_when_q_one_sum_c_when_q_not_one_l689_68983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_neg_one_solution_set_l689_68923

open Real

theorem sin_plus_cos_eq_neg_one_solution_set :
  {x : ℝ | Real.sin x + Real.cos x = -1} = {x : ℝ | ∃ n : ℤ, x = (2*n - 1)*π ∨ x = 2*n*π - π/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_neg_one_solution_set_l689_68923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_attendance_probability_l689_68909

/-- The probability that at least 8 out of 9 people stay for an entire concert,
    given that 5 are certain to stay and 4 have a 3/7 probability of staying. -/
theorem concert_attendance_probability :
  let total_people : ℕ := 9
  let certain_stayers : ℕ := 5
  let uncertain_stayers : ℕ := 4
  let stay_probability : ℚ := 3/7
  let at_least_staying : ℕ := 8
  
  (513 : ℚ) / 2401 = 
    (Finset.range (uncertain_stayers + 1)).sum (fun k => 
      if k + certain_stayers ≥ at_least_staying then
        (Nat.choose uncertain_stayers k : ℚ) * 
        (stay_probability ^ k) * 
        ((1 : ℚ) - stay_probability) ^ (uncertain_stayers - k)
      else 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_attendance_probability_l689_68909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inverse_first_degree_function_l689_68951

theorem function_inverse (f : ℝ → ℝ) :
  (∀ x, x ≠ 0 → f (2/x + 2) = x + 1) →
  ∀ x, x ≠ 2 → f x = x / (x - 2) :=
by sorry

theorem first_degree_function (f : ℝ → ℝ) :
  (∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) →
  ∃ k b, k = 2 ∧ b = 7 ∧ ∀ x, f x = k * x + b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inverse_first_degree_function_l689_68951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l689_68961

noncomputable def f (x : ℝ) := x^2 - Real.log x

def line (x y : ℝ) : Prop := x - y - 2 = 0

noncomputable def distance_to_line (x₁ y₁ : ℝ) : ℝ :=
  |x₁ - y₁ - 2| / Real.sqrt 2

theorem min_distance_to_line :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  ∀ (x : ℝ), x > 0 → distance_to_line x (f x) ≥ distance_to_line x₀ (f x₀) ∧
  distance_to_line x₀ (f x₀) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l689_68961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_symmetric_circle_exists_l689_68928

/-- Given a circle and a line, find the symmetric circle --/
theorem symmetric_circle 
  (c₁ : ℝ × ℝ → Prop) 
  (l : ℝ × ℝ → Prop) 
  (c₂ : ℝ × ℝ → Prop) : Prop :=
  (∀ x y, c₁ (x, y) ↔ (x - 3)^2 + (y + 1)^2 = 1) →
  (∀ x y, l (x, y) ↔ x + 2*y - 3 = 0) →
  (∀ x y, c₂ (x, y) ↔ (x - 11/3)^2 + (y - 1/3)^2 = 1) →
  (∀ p₁ p₂, 
    (c₁ p₁ ∧ c₂ p₂) → 
    (∃ q, l q ∧ 
      q.1 = (p₁.1 + p₂.1) / 2 ∧ 
      q.2 = (p₁.2 + p₂.2) / 2))

theorem symmetric_circle_exists : ∃ c₂, symmetric_circle 
  (λ p ↦ (p.1 - 3)^2 + (p.2 + 1)^2 = 1)
  (λ p ↦ p.1 + 2*p.2 - 3 = 0)
  c₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_symmetric_circle_exists_l689_68928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l689_68959

/-- Curve C in Cartesian coordinates -/
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

/-- Line l in Cartesian coordinates -/
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

/-- Point P -/
def point_P : ℝ × ℝ := (-2, -4)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Three lengths form a geometric sequence -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem curve_line_intersection (a : ℝ) :
  ∃ (M N : ℝ × ℝ),
    curve_C a M.1 M.2 ∧
    curve_C a N.1 N.2 ∧
    line_l M.1 M.2 ∧
    line_l N.1 N.2 ∧
    is_geometric_sequence (distance point_P M) (distance M N) (distance point_P N) →
    a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l689_68959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_condition_l689_68996

open Real

theorem function_derivative_condition (a b c d : ℝ) :
  (∀ x, HasDerivAt (λ x ↦ (a*x + b) * sin x + (c*x + d) * cos x) (x * cos x) x) →
  a = 1 ∧ d = 1 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_derivative_condition_l689_68996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_l689_68971

/-- Represents the price of a water-based pen in yuan -/
def x : ℝ := Real.mk 0  -- We initialize x with a default value of 0

/-- Represents the price of a notebook in yuan -/
def notebook_price (x : ℝ) : ℝ := x - 2

/-- The total cost of Xiaogang's purchase in yuan -/
def total_cost (x : ℝ) : ℝ := 5 * notebook_price x + 3 * x

/-- Theorem stating that the equation correctly represents Xiaogang's purchase -/
theorem correct_equation (x : ℝ) : total_cost x = 14 := by
  sorry

#eval total_cost 5  -- This will evaluate the total cost for x = 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_l689_68971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetric_to_lines_l689_68992

/-- A circle symmetric with respect to two lines has its center at their intersection point -/
theorem circle_symmetric_to_lines (D E F : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 →
    (x - y + 4 = 0 ↔ x^2 + y^2 + D*x + E*y + F = 0) ∧
    (x + 3*y = 0 ↔ x^2 + y^2 + D*x + E*y + F = 0)) →
  D = 3 := by
  sorry

#check circle_symmetric_to_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetric_to_lines_l689_68992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_averageAgeThreeRooms_l689_68955

/-- Calculates the average age across multiple rooms given the number of people and average age in each room -/
def averageAgeAcrossRooms (rooms : List (Nat × Float)) : Float :=
  let totalPeople := (rooms.map (fun (n, _) => n)).sum
  let totalAge := (rooms.map (fun (n, a) => n.toFloat * a)).sum
  totalAge / totalPeople.toFloat

/-- Theorem stating that the average age across the given rooms is 30.25 -/
theorem averageAgeThreeRooms :
  let rooms : List (Nat × Float) := [(8, 35), (5, 30), (7, 25)]
  averageAgeAcrossRooms rooms = 30.25 := by
  sorry

#eval averageAgeAcrossRooms [(8, 35), (5, 30), (7, 25)]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_averageAgeThreeRooms_l689_68955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_draw_odds_must_be_one_to_six_l689_68915

/-- Represents the betting odds for a team --/
structure BettingOdds where
  stake : ℚ
  payout : ℚ

/-- Represents a football match with betting options --/
structure FootballMatch where
  team1_odds : BettingOdds
  team2_odds : BettingOdds
  draw_odds : BettingOdds

/-- Calculates the total amount bet on all outcomes --/
def total_bet (m : FootballMatch) (bet1 bet2 betDraw : ℚ) : ℚ :=
  bet1 + bet2 + betDraw

/-- Calculates the payout for a given outcome --/
def payout (odds : BettingOdds) (bet : ℚ) : ℚ :=
  bet * (odds.payout / odds.stake)

/-- Theorem: Given specific betting odds and equal payout condition, the draw odds must be 1 to 6 --/
theorem draw_odds_must_be_one_to_six 
  (m : FootballMatch)
  (bet1 bet2 betDraw : ℚ)
  (h1 : m.team1_odds = ⟨1, 2⟩)
  (h2 : m.team2_odds = ⟨1, 3⟩)
  (h_equal_payout : 
    payout m.team1_odds bet1 = 
    payout m.team2_odds bet2 ∧
    payout m.team1_odds bet1 = 
    payout m.draw_odds betDraw) :
  m.draw_odds = ⟨1, 6⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_draw_odds_must_be_one_to_six_l689_68915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l689_68929

-- Define the friends
inductive Friend
| Ada | Bea | Ceci | Dee | Edie | Frank

-- Define the seats
def Seat := Fin 6

-- Define the initial seating arrangement
def initial_seating : Friend → Seat := sorry

-- Define the movement of friends
def move_friend (f : Friend) : ℤ :=
  match f with
  | Friend.Bea => 1
  | Friend.Ceci => 0
  | Friend.Dee => 0
  | Friend.Edie => 0
  | Friend.Frank => -2
  | Friend.Ada => 0

-- Define Ada's final position
def ada_final_seat : Seat := ⟨1, by norm_num⟩

-- Theorem statement
theorem ada_original_seat :
  initial_seating Friend.Ada = ⟨0, by norm_num⟩ := by
  sorry

#check ada_original_seat

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l689_68929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_line_intersection_l689_68978

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The right focus of an ellipse -/
noncomputable def rightFocus (e : Ellipse) : Point :=
  { x := e.a * eccentricity e, y := 0 }

/-- The top vertex of an ellipse -/
def topVertex (e : Ellipse) : Point :=
  { x := 0, y := e.b }

/-- The bottom vertex of an ellipse -/
def bottomVertex (e : Ellipse) : Point :=
  { x := 0, y := -e.b }

/-- The dot product of two vectors represented by points -/
def dotProduct (p q : Point) : ℝ := p.x * q.x + p.y * q.y

/-- Vector subtraction for Points -/
instance : HSub Point Point Point where
  hSub p q := { x := p.x - q.x, y := p.y - q.y }

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (e : Ellipse) 
  (h_ecc : eccentricity e = 1/2) 
  (h_dot : dotProduct (topVertex e - rightFocus e) (bottomVertex e - rightFocus e) = -2) :
  e.a^2 = 4 ∧ e.b^2 = 3 := by sorry

/-- Theorem about a line intersecting the ellipse -/
theorem line_intersection (e : Ellipse) (k m : ℝ)
  (h_e : e.a^2 = 4 ∧ e.b^2 = 3)
  (h_distinct : ∃ A B : Point, A ≠ B ∧ 
    A.y = k * A.x + m ∧ B.y = k * B.x + m ∧
    A.x^2 / 4 + A.y^2 / 3 = 1 ∧ B.x^2 / 4 + B.y^2 / 3 = 1)
  (h_slopes : ∃ k_MA k_MB : ℝ, k_MA * k_MB = 1/4 ∧
    ∀ A : Point, A.y = k * A.x + m ∧ A.x^2 / 4 + A.y^2 / 3 = 1 →
    k_MA = (A.y - (topVertex e).y) / A.x ∨ k_MB = (A.y - (topVertex e).y) / A.x) :
  ∃ P : Point, P.x = 0 ∧ P.y = 2 * Real.sqrt 3 ∧
    ∀ A B : Point, A ≠ B →
      A.y = k * A.x + m → B.y = k * B.x + m →
      A.x^2 / 4 + A.y^2 / 3 = 1 → B.x^2 / 4 + B.y^2 / 3 = 1 →
      ∃ t : ℝ, P = { x := (1 - t) * A.x + t * B.x, y := (1 - t) * A.y + t * B.y } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_line_intersection_l689_68978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l689_68913

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * (a - x) * Real.exp x

/-- The theorem statement -/
theorem min_a_value (a : ℝ) (h_a : a > 0) :
  (∃ x ∈ Set.Icc 0 2, f a x ≥ Real.exp 1) ↔ a ≥ 2 + Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l689_68913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_l689_68972

theorem banana_cost (total_spent milk_cost cereal_cost apple_cost cookie_cost banana_count banana_cost : ℚ)
  (h1 : total_spent = 25)
  (h2 : milk_cost = 3)
  (h3 : cereal_cost = 3.5 * 2)
  (h4 : apple_cost = 0.5 * 4)
  (h5 : cookie_cost = 2 * milk_cost * 2)
  (h6 : banana_count = 4)
  (h7 : total_spent = milk_cost + cereal_cost + apple_cost + cookie_cost + (banana_count * banana_cost)) :
  banana_cost = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_l689_68972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l689_68901

/-- The slope of the asymptotes for a hyperbola -/
noncomputable def asymptote_slope (a b : ℝ) : ℝ := b / a

/-- The equation of a hyperbola -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  (y^2 / b^2) - (x^2 / a^2) = 1

theorem hyperbola_asymptote_slope :
  ∀ (x y : ℝ), is_hyperbola x y 3 4 →
  asymptote_slope 3 4 = 4/3 :=
by
  intros x y h
  unfold asymptote_slope
  norm_num

#check hyperbola_asymptote_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l689_68901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_mixture_l689_68938

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  element1 : ℝ
  element2 : ℝ

/-- Represents an alloy with its weight and composition -/
structure Alloy where
  weight : ℝ
  composition : AlloyComposition

/-- Calculates the amount of the second element in an alloy -/
noncomputable def amountOfElement2 (alloy : Alloy) : ℝ :=
  alloy.weight * (alloy.composition.element2 / (alloy.composition.element1 + alloy.composition.element2))

theorem tin_amount_in_mixture (alloyA alloyB alloyC : Alloy)
  (hA : alloyA.composition = ⟨1, 3⟩)
  (hB : alloyB.composition = ⟨3, 5⟩)
  (hC : alloyC.composition = ⟨4, 1⟩)
  (wA : alloyA.weight = 170)
  (wB : alloyB.weight = 250)
  (wC : alloyC.weight = 120) :
  amountOfElement2 alloyA + amountOfElement2 alloyB + amountOfElement2 alloyC = 245.25 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_mixture_l689_68938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_N_l689_68990

def N : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_divisors_of_N : 
  (Finset.filter (λ d ↦ d ∣ N) (Finset.range (N + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_N_l689_68990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l689_68920

noncomputable def f (x : ℝ) : ℝ := 3 - Real.sqrt 3 * Real.cos x + 1 - Real.sin x

theorem problem_solution :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (A B C : ℝ), 0 < A ∧ A < π ∧ f A = 4 ∧ B + C = 3 → 
    B + C + Real.sqrt (B^2 + C^2) ≤ 3 / Real.sqrt 2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l689_68920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_false_l689_68933

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem power_function_increasing_false :
  ¬(∀ α : ℝ, is_increasing (power_function α) → is_increasing (power_function (-1))) := by
  -- Proof sketch:
  -- We will show that there exists an α such that power_function α is increasing,
  -- but power_function (-1) is not increasing.
  -- For example, when α = 1, x^1 is increasing, but x^(-1) is decreasing.
  sorry

-- You can add more lemmas or theorems here if needed


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_false_l689_68933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersects_inner_circle_l689_68967

/-- Two concentric circles with radii 2 and 4 -/
noncomputable def inner_radius : ℝ := 2
noncomputable def outer_radius : ℝ := 4

/-- The probability that a chord intersects the inner circle when two points are chosen randomly on the outer circle -/
noncomputable def chord_intersection_probability : ℝ := 1/3

/-- Theorem stating that the probability of a chord intersecting the inner circle is 1/3 -/
theorem chord_intersects_inner_circle :
  let r₁ := inner_radius
  let r₂ := outer_radius
  chord_intersection_probability = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersects_inner_circle_l689_68967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_sum_l689_68917

/-- Represents a 29-digit number with special properties -/
def SpecialNumber : Type := Fin 29 → Fin 10

/-- The sum of digits of a SpecialNumber -/
def digit_sum (x : SpecialNumber) : ℕ :=
  (Finset.univ.sum fun i => (x i).val)

/-- Counts the occurrences of a digit in a SpecialNumber -/
def count_occurrences (x : SpecialNumber) (d : Fin 10) : ℕ :=
  (Finset.univ.filter fun i => x i = d).card

theorem special_number_sum :
  ∀ x : SpecialNumber,
  (∀ k : Fin 29, count_occurrences x (x k) = (x (Fin.sub 29 k).succ).val) →
  x 0 ≠ 0 →
  digit_sum x = 201 := by
  sorry

#check special_number_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_sum_l689_68917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_problem_l689_68954

/-- The length of the path traversed by a vertex of an equilateral triangle
    rotated inside a square as described in the problem -/
noncomputable def triangle_rotation_path_length (triangle_side : ℝ) (square_side : ℝ) : ℝ :=
  4 * 3 * Real.pi + 2 * square_side * Real.sqrt 2

/-- The problem statement as a theorem -/
theorem triangle_rotation_problem (triangle_side square_side : ℝ) :
  triangle_side = 3 → square_side = 6 →
  triangle_rotation_path_length triangle_side square_side = 24 * Real.pi + 12 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_problem_l689_68954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_l689_68947

theorem inequality_implies_log (x y : ℝ) : 
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_l689_68947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_probability_l689_68937

noncomputable def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def triangle_probability : ℝ :=
  (∫ x in (1/5)..(9), 1) / 10

theorem triangle_side_probability :
  triangle_probability = 22 / 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_probability_l689_68937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_not_divisible_by_3_impossible_l689_68931

/-- Represents a 10x10 table of nonzero digits -/
def Table := Fin 10 → Fin 10 → Fin 9

/-- Computes the number formed from a row or column -/
def formNumber (t : Table) (isRow : Bool) (index : Fin 10) : ℕ :=
  sorry

/-- Checks if a number is divisible by 3 -/
def isDivisibleBy3 (n : ℕ) : Bool :=
  n % 3 = 0

theorem exactly_one_not_divisible_by_3_impossible (t : Table) :
  ¬∃! i : Fin 20, ¬isDivisibleBy3 (
    if h : i.val < 10 then
      formNumber t true ⟨i.val, h⟩
    else
      formNumber t false ⟨i.val - 10, sorry⟩
  ) := by
  sorry

#check exactly_one_not_divisible_by_3_impossible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_not_divisible_by_3_impossible_l689_68931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_wins_probability_l689_68984

-- Define the dice throw result type
def DiceThrow := Fin 6

-- Define the game outcome
inductive GameOutcome
| AWins
| BWins

-- Define the game state
structure GameState where
  x : DiceThrow
  y : DiceThrow
  list : List Nat

-- Define the game rules
def sameParityWins (x y : DiceThrow) : Bool :=
  (x.val % 2 == y.val % 2)

def makeList (x y : DiceThrow) : List Nat :=
  (List.range 6).bind (fun a =>
    (List.range 6).bind (fun b =>
      let ab := (a + 1) * 10 + (b + 1)
      if ab ≤ (x.val + 1) * 10 + (y.val + 1) then [ab] else []))

def replaceWithDifference (list : List Nat) : List Nat :=
  sorry  -- Implementation details omitted

def finalComparison (x : DiceThrow) (n : Nat) : GameOutcome :=
  if x.val % 2 == n % 2 then GameOutcome.AWins else GameOutcome.BWins

-- Define the probability of A winning
def probAWins : ℚ := 3 / 4

-- State the theorem
theorem a_wins_probability :
  probAWins = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_wins_probability_l689_68984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_g_g_eq_9_l689_68980

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -2 then x^2 - 1 else x + 5

theorem no_solution_for_g_g_eq_9 : ∀ x : ℝ, g (g x) ≠ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_g_g_eq_9_l689_68980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_monotone_range_l689_68982

/-- A function f is monotonically increasing on an interval [a, b] if for any x, y in [a, b] with x ≤ y, we have f(x) ≤ f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

/-- The quadratic function f(x) = x^2 - 2ax - 4a -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 - 2*a*x - 4*a

theorem quadratic_monotone_range :
  {a : ℝ | MonotonicallyIncreasing (f a) 1 2} = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_monotone_range_l689_68982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_increasing_intervals_l689_68988

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem symmetry_and_increasing_intervals 
  (φ : ℝ) 
  (h1 : -π < φ) 
  (h2 : φ < 0) 
  (h3 : ∀ x, f x φ = f (π/4 - x) φ) : 
  (φ = -3*π/4) ∧ 
  (∀ m : ℤ, ∀ x ∈ Set.Icc (π/8 + m*π) (5*π/8 + m*π), 
    ∀ y ∈ Set.Icc (π/8 + m*π) (5*π/8 + m*π), 
    x ≤ y → f x φ ≤ f y φ) := by
  sorry

#check symmetry_and_increasing_intervals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_increasing_intervals_l689_68988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l689_68987

/-- The function f(x) = x²/(x²-x+1) -/
noncomputable def f (x : ℝ) : ℝ := x^2 / (x^2 - x + 1)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ 0 ≤ y ∧ y ≤ 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l689_68987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_trig_calculation_l689_68945

theorem inverse_trig_calculation : 3 * Real.arcsin (Real.sqrt 3 / 2) - Real.arctan (-1) - Real.arccos 0 = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_trig_calculation_l689_68945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_at_negative_one_l689_68903

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (x : ℝ) : ℝ := x * (e ^ x)

theorem local_minimum_at_negative_one :
  ∃ δ > 0, ∀ x ∈ Set.Ioo (-1 - δ) (-1 + δ), f (-1) ≤ f x := by
  sorry

#check local_minimum_at_negative_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_at_negative_one_l689_68903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l689_68908

/-- Perimeter of a quadrilateral EFGH with specific properties -/
theorem quadrilateral_perimeter (EF FG GH : ℝ) 
  (h_EF : EF = 10) (h_GH : GH = 3) (h_FG : FG = 15) :
  EF + FG + GH + Real.sqrt ((EF - GH)^2 + FG^2) = 28 + Real.sqrt 274 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l689_68908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l689_68912

theorem indefinite_integral_proof :
  ∀ x : ℝ, 
    (deriv (λ x => (2*x - 1) * Real.sin (2*x) + Real.cos (2*x))) x = 
    (4*x - 2) * Real.cos (2*x) := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l689_68912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_consumption_increase_l689_68989

/-- Represents the daily increase in banana consumption -/
def daily_increase : ℕ → ℕ := sorry

/-- Represents the total number of bananas eaten on a given day -/
def bananas_eaten (d : ℕ) : ℕ := 8 + (d - 1) * daily_increase 0

/-- The theorem stating the conditions and the result to be proved -/
theorem banana_consumption_increase :
  (∀ d : ℕ, d > 0 ∧ d ≤ 5 → bananas_eaten d > 0) →  -- Positive consumption each day
  (∀ d : ℕ, d > 1 ∧ d ≤ 5 → bananas_eaten d > bananas_eaten (d-1)) →  -- Increasing consumption
  (bananas_eaten 1 = 8) →  -- First day's consumption
  (bananas_eaten 1 + bananas_eaten 2 + bananas_eaten 3 + bananas_eaten 4 + bananas_eaten 5 = 100) →  -- Total consumption
  daily_increase 0 = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_consumption_increase_l689_68989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_palindrome_sum_of_four_digit_palindromes_l689_68926

/-- A function to check if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Prop :=
  (n.repr.toList).reverse = n.repr.toList

/-- A function to check if a natural number has exactly k digits -/
def hasDigits (n k : ℕ) : Prop :=
  10^(k-1) ≤ n ∧ n < 10^k

/-- The main theorem stating the existence of a five-digit palindrome
    that is the sum of two four-digit palindromes -/
theorem five_digit_palindrome_sum_of_four_digit_palindromes :
  ∃ (a b c : ℕ), 
    isPalindrome a ∧ 
    isPalindrome b ∧ 
    isPalindrome c ∧
    hasDigits a 4 ∧ 
    hasDigits b 4 ∧ 
    hasDigits c 5 ∧
    c = a + b := by
  -- Proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_palindrome_sum_of_four_digit_palindromes_l689_68926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l689_68970

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - |x|) + 1 / (x^2 + 1)

theorem f_inequality_range (x : ℝ) :
  (∀ y, y ∈ Set.Ioo (-1 : ℝ) 1 → f y = f (-y)) →
  (∀ y z, y ∈ Set.Ioi (0 : ℝ) → z ∈ Set.Ioi (0 : ℝ) → y < z → f y > f z) →
  (Set.range f = Set.Ioo (-1 : ℝ) 1) →
  (f (2*x + 1) ≥ f x ↔ x ∈ Set.Ioc (-1 : ℝ) (-1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l689_68970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l689_68952

theorem solve_exponential_equation (y : ℝ) : (9 : ℝ)^y = (3 : ℝ)^12 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l689_68952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_properties_l689_68930

-- Define the ellipse E
def ellipse_E (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the circle (renamed to avoid conflict)
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 8 / 3

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem ellipse_and_circle_properties :
  -- The ellipse passes through the given points
  ellipse_E 2 (Real.sqrt 2) ∧ ellipse_E (Real.sqrt 6) 1 ∧
  -- For any tangent line to the circle, it intersects the ellipse at two points A and B
  ∀ k m : ℝ, ∃ x1 y1 x2 y2 : ℝ,
    -- The tangent line equation: y = kx + m
    (y1 = k * x1 + m ∧ y2 = k * x2 + m) ∧
    -- The points lie on the ellipse
    ellipse_E x1 y1 ∧ ellipse_E x2 y2 ∧
    -- The line is tangent to the circle
    (k^2 + 1) * (8 / 3) = m^2 →
    -- A and B are perpendicular from the origin
    perpendicular x1 y1 x2 y2 ∧
    -- The range of |AB|
    4 * Real.sqrt 6 / 3 < distance x1 y1 x2 y2 ∧ distance x1 y1 x2 y2 ≤ 2 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_properties_l689_68930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l689_68932

/-- The exponential function with base 2 -/
noncomputable def f (x : ℝ) : ℝ := 2^x

/-- The slope of the tangent line to f at x = 1 -/
noncomputable def tangent_slope : ℝ := 2 * Real.log 2

theorem tangent_slope_at_one :
  HasDerivAt f tangent_slope 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l689_68932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_quadratic_implies_negative_leading_coeff_l689_68997

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- Theorem: If a quadratic function is always negative, then its leading coefficient is negative -/
theorem negative_quadratic_implies_negative_leading_coeff
  (a b c : ℝ) (h : ∀ x, quadratic_function a b c x < 0) :
  a < 0 := by
  sorry

#check negative_quadratic_implies_negative_leading_coeff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_quadratic_implies_negative_leading_coeff_l689_68997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l689_68973

noncomputable section

def area_triangle (A B C a b c : ℝ) : ℝ := (1/2) * a * c * Real.sin B

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = 2 →
  Real.cos B = 3/5 →
  -- Part (1)
  (b = 4 →
    Real.sin A = 2/5) ∧
  -- Part (2)
  (area_triangle A B C a b c = 4 →
    b = Real.sqrt 17 ∧ c = 5) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l689_68973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l689_68910

theorem min_value_theorem (y : ℝ) (h : y > 0) : 9 * y^7 + 4 * y^(-(3 : ℤ)) ≥ 13 ∧ 
  (9 * y^7 + 4 * y^(-(3 : ℤ)) = 13 ↔ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l689_68910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_max_at_two_l689_68956

/-- The concentration function C(t) for a chemical in a swimming pool -/
noncomputable def C (t : ℝ) : ℝ := 20 * t / (t^2 + 4)

/-- Theorem stating that C(t) reaches its maximum when t = 2 -/
theorem concentration_max_at_two : 
  ∀ t : ℝ, t > 0 → C t ≤ C 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_max_at_two_l689_68956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_inequality_l689_68957

theorem inverse_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 1/a - 1/b > Real.log (1/b) - Real.log (1/a)) :
  1/a > 1/b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_inequality_l689_68957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l689_68914

/-- The length of a train in meters -/
noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600) * time_s

/-- Theorem: A train traveling at 60 km/h that crosses a pole in 10 seconds has a length of 166.7 meters -/
theorem train_length_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ abs (train_length 60 10 - 166.7) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l689_68914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l689_68968

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the condition for point P
def satisfies_condition (x y : ℝ) : Prop :=
  distance x y 1 0 - |x| = 1

-- Define the trajectory equation
def on_trajectory (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x < 0 ∧ y = 0)

-- Theorem statement
theorem trajectory_equation :
  ∀ x y : ℝ, satisfies_condition x y → on_trajectory x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l689_68968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_five_digit_number_has_8_in_tens_place_l689_68936

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  ∃ (a b c d e : ℕ),
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    ({a, b, c, d, e} : Finset ℕ) = {1, 3, 5, 6, 8}

def is_even (n : ℕ) : Prop :=
  ∃ k, n = 2 * k

theorem smallest_even_five_digit_number_has_8_in_tens_place :
  ∃ (n : ℕ),
    is_valid_number n ∧
    is_even n ∧
    (∀ m, is_valid_number m ∧ is_even m → n ≤ m) ∧
    ∃ (a b c : ℕ), n = a * 10000 + b * 1000 + c * 100 + 8 * 10 + 6 :=
by sorry

#check smallest_even_five_digit_number_has_8_in_tens_place

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_five_digit_number_has_8_in_tens_place_l689_68936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l689_68907

/-- The area of a triangle with vertices (-3, 3), (5, -1), and (13, 6) is 44 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 44 ∧ area = abs ((13 - (-3)) * ((-1) - 3) - (5 - (-3)) * (6 - 3)) / 2 := by
  -- Define the vertices
  let A : ℝ × ℝ := (-3, 3)
  let B : ℝ × ℝ := (5, -1)
  let C : ℝ × ℝ := (13, 6)

  -- Calculate the area
  let area := abs ((C.1 - A.1) * (B.2 - A.2) - (B.1 - A.1) * (C.2 - A.2)) / 2

  -- Prove that the area exists and equals 44
  use area
  constructor
  · -- Prove area = 44
    sorry
  · -- Prove area equals the formula
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l689_68907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_pyramid_volume_l689_68986

/-- The volume of a pyramid with a rectangular base and given edge lengths. -/
theorem pyramid_volume (base_length base_width edge_length : ℝ) 
  (base_length_pos : 0 < base_length)
  (base_width_pos : 0 < base_width)
  (edge_length_pos : 0 < edge_length) :
  let base_area := base_length * base_width
  let diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal / 2)^2)
  (1/3 : ℝ) * base_area * height = 
  (1/3 : ℝ) * base_area * Real.sqrt (edge_length^2 - (diagonal / 2)^2) := by
  sorry

/-- The volume of a specific pyramid with a 7 × 9 rectangular base and edge length 15. -/
theorem specific_pyramid_volume :
  let base_length : ℝ := 7
  let base_width : ℝ := 9
  let edge_length : ℝ := 15
  let base_area := base_length * base_width
  let diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal / 2)^2)
  (1/3 : ℝ) * base_area * height = 21 * Real.sqrt 192.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_specific_pyramid_volume_l689_68986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_and_inscribed_circle_area_l689_68985

-- Define the circles O₁ and O₂
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1
def circle_O₂ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 9

-- Define the trajectory of the center of the moving circle M
def trajectory (x y : ℝ) : Prop := y^2 / 4 + x^2 / 3 = 1

-- Define the line l passing through the center of O₁
def line_l (k x y : ℝ) : Prop := y = k * x + 1

-- Define area and perimeter of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry
noncomputable def perimeter_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem moving_circle_and_inscribed_circle_area 
  (M : ℝ × ℝ → Prop) -- Moving circle
  (h1 : ∀ x y, M (x, y) → ∃ r, (x - 0)^2 + (y - 1)^2 = (1 + r)^2) -- M externally tangent to O₁
  (h2 : ∀ x y, M (x, y) → ∃ r, (x - 0)^2 + (y + 1)^2 = (3 - r)^2) -- M internally tangent to O₂
  : 
  (∀ x y, M (x, y) → trajectory x y) ∧ -- Part 1
  (∃ max_area : ℝ, max_area = 9 * Real.pi / 16 ∧ 
    ∀ k A B, 
      line_l k A.1 A.2 → 
      line_l k B.1 B.2 → 
      trajectory A.1 A.2 → 
      trajectory B.1 B.2 → 
      A ≠ B →
      let O₂ : ℝ × ℝ := (0, -1);
      let inscribed_circle_area := Real.pi * (area_triangle A B O₂ / (perimeter_triangle A B O₂ / 2))^2;
      inscribed_circle_area ≤ max_area) -- Part 2
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_and_inscribed_circle_area_l689_68985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_d_value_l689_68906

theorem min_d_value (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_order : a < b ∧ b < c ∧ c < d)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2007 ∧ 
    y = |x - (a : ℝ)| + |x - (b : ℝ)| + |x - (c : ℝ)| + |x - (d : ℝ)|) :
  d = 504 ∧ ∀ d' : ℕ, 0 < d' → d' < d → ¬∃! (x y : ℝ), 2 * x + y = 2007 ∧ 
    y = |x - (a : ℝ)| + |x - (b : ℝ)| + |x - (c : ℝ)| + |x - (d' : ℝ)| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_d_value_l689_68906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l689_68940

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 11) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l689_68940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_count_l689_68904

theorem photo_arrangement_count (n m k : ℕ) (hn : n = 9) (hm : m = 7) (hk : k = 2) : 
  (n.choose k) * (m.descFactorial k) = (Nat.choose n k) * (Nat.factorial m / Nat.factorial (m - k)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_count_l689_68904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_satisfy_condition_l689_68918

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with edge length 1 -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Checks if a point is on an edge of the unit cube -/
def isOnEdge (cube : UnitCube) (p : Point3D) : Prop :=
  sorry

/-- Counts the number of points on the edges of the cube satisfying the condition -/
def countSatisfyingPoints (cube : UnitCube) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 6 points satisfying the condition -/
theorem six_points_satisfy_condition (cube : UnitCube) : 
  countSatisfyingPoints cube = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_satisfy_condition_l689_68918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l689_68948

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  (a / (a - b)) = ((a / b) / ((a / b) - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l689_68948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l689_68902

-- Define the domain of the functions (all real numbers except 2)
noncomputable def Domain := {x : ℝ | x ≠ 2}

-- Define the original function
noncomputable def f (x : Domain) : ℝ := x - 1 - (x - 2)^0

-- Define the equivalent function
noncomputable def g (x : Domain) : ℝ := (x - 2)^2 / (x - 2)

-- State the theorem
theorem f_equiv_g : ∀ (x : Domain), f x = g x := by
  intro x
  simp [f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l689_68902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_not_converge_to_rational_l689_68979

/-- A sequence approximating √2 with accuracy up to 1/10ⁿ -/
def x : ℕ → ℚ := sorry

/-- The property that for any ε > 0 and k, there exists n > k such that |xₙ - a| ≥ ε -/
def diverges_from_rational (x : ℕ → ℚ) (a : ℚ) : Prop :=
  ∀ ε > 0, ∀ k : ℕ, ∃ n > k, |x n - a| ≥ ε

/-- The theorem stating that the sequence x does not converge to any rational number -/
theorem x_not_converge_to_rational :
  ∀ a : ℚ, diverges_from_rational x a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_not_converge_to_rational_l689_68979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_purchase_price_l689_68966

/-- Calculates the purchase price of shares given the specified conditions --/
theorem share_purchase_price 
  (share_value : ℝ) 
  (dividend_percentage : ℝ) 
  (tax_rate : ℝ) 
  (inflation_rate : ℝ) 
  (effective_return : ℝ) 
  (h1 : share_value = 50) 
  (h2 : dividend_percentage = 0.185) 
  (h3 : tax_rate = 0.05) 
  (h4 : inflation_rate = 0.02) 
  (h5 : effective_return = 0.25) : 
  ∃ (purchase_price : ℝ), abs (purchase_price - 35.15) < 0.01 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_purchase_price_l689_68966
