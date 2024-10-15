import Mathlib

namespace NUMINAMATH_CALUDE_angle_positions_l367_36715

-- Define the quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define the position of an angle
inductive AnglePosition
  | InQuadrant (q : Quadrant)
  | OnPositiveYAxis

-- Function to determine the position of 2θ
def doubleThetaPosition (θ : Real) : AnglePosition := sorry

-- Function to determine the position of θ/2
def halfThetaPosition (θ : Real) : Quadrant := sorry

-- Theorem statement
theorem angle_positions (θ : Real) 
  (h : ∃ (k : ℤ), 180 + k * 360 < θ ∧ θ < 270 + k * 360) : 
  (doubleThetaPosition θ = AnglePosition.InQuadrant Quadrant.First ∨
   doubleThetaPosition θ = AnglePosition.InQuadrant Quadrant.Second ∨
   doubleThetaPosition θ = AnglePosition.OnPositiveYAxis) ∧
  (halfThetaPosition θ = Quadrant.Second ∨
   halfThetaPosition θ = Quadrant.Fourth) := by
  sorry

end NUMINAMATH_CALUDE_angle_positions_l367_36715


namespace NUMINAMATH_CALUDE_earth_rotation_certain_l367_36764

-- Define the type for events
inductive Event : Type
  | EarthRotation : Event
  | RainTomorrow : Event
  | TimeBackwards : Event
  | SnowfallWinter : Event

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.EarthRotation => True
  | _ => False

-- Define the conditions given in the problem
axiom earth_rotation_continuous : ∀ (t : ℝ), ∃ (angle : ℝ), angle ≥ 0 ∧ angle < 360
axiom weather_probabilistic : ∃ (p : ℝ), 0 < p ∧ p < 1
axiom time_forwards : ∀ (t1 t2 : ℝ), t1 < t2 → t1 ≠ t2
axiom snowfall_not_guaranteed : ∃ (winter : Set ℝ), ∃ (day : ℝ), day ∈ winter ∧ ¬∃ (snow : ℝ), snow > 0

-- The theorem to prove
theorem earth_rotation_certain : is_certain Event.EarthRotation :=
  sorry

end NUMINAMATH_CALUDE_earth_rotation_certain_l367_36764


namespace NUMINAMATH_CALUDE_intersection_sum_l367_36761

/-- Given two functions f and g defined as:
    f(x) = -2|x-a| + b
    g(x) = 2|x-c| + d
    If f and g intersect at points (10, 15) and (18, 7),
    then a + c = 28 -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -2 * |x - a| + b = 2 * |x - c| + d → x = 10 ∨ x = 18) →
  -2 * |10 - a| + b = 15 →
  -2 * |18 - a| + b = 7 →
  2 * |10 - c| + d = 15 →
  2 * |18 - c| + d = 7 →
  a + c = 28 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l367_36761


namespace NUMINAMATH_CALUDE_spherical_to_cartesian_l367_36791

/-- Conversion from spherical coordinates to Cartesian coordinates -/
theorem spherical_to_cartesian :
  let r : ℝ := 8
  let θ : ℝ := π / 3
  let φ : ℝ := π / 6
  let x : ℝ := r * Real.sin θ * Real.cos φ
  let y : ℝ := r * Real.sin θ * Real.sin φ
  let z : ℝ := r * Real.cos θ
  (x, y, z) = (6, 2 * Real.sqrt 3, 4) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_cartesian_l367_36791


namespace NUMINAMATH_CALUDE_total_peaches_sum_l367_36726

/-- The total number of peaches after picking more -/
def total_peaches (initial : Float) (picked : Float) : Float :=
  initial + picked

/-- Theorem: The total number of peaches is the sum of initial and picked peaches -/
theorem total_peaches_sum (initial picked : Float) :
  total_peaches initial picked = initial + picked := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_sum_l367_36726


namespace NUMINAMATH_CALUDE_power_product_equality_l367_36769

theorem power_product_equality : 3^5 * 6^5 = 1889568 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l367_36769


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l367_36728

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 25 years older than his son and the son's current age is 23. -/
theorem age_ratio_in_two_years (son_age : ℕ) (man_age : ℕ) : 
  son_age = 23 →
  man_age = son_age + 25 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l367_36728


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l367_36745

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (5, 0)

/-- Theorem: The intersection point satisfies both the line equation and lies on the x-axis -/
theorem line_x_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_x_axis x y :=
by sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l367_36745


namespace NUMINAMATH_CALUDE_lansing_elementary_schools_l367_36779

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 6175 / 247

/-- Theorem: There are 25 elementary schools in Lansing -/
theorem lansing_elementary_schools : num_schools = 25 := by
  sorry

end NUMINAMATH_CALUDE_lansing_elementary_schools_l367_36779


namespace NUMINAMATH_CALUDE_equation_solutions_l367_36705

theorem equation_solutions (x : ℝ) : 
  (x^3 + 2*x)^(1/5) = (x^5 - 2*x)^(1/3) ↔ x = 0 ∨ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l367_36705


namespace NUMINAMATH_CALUDE_decision_block_two_exits_other_blocks_not_two_exits_l367_36778

/-- Enumeration of program block types -/
inductive ProgramBlock
  | Output
  | Processing
  | Decision
  | StartEnd

/-- Function to determine the number of exits for each program block -/
def num_exits (block : ProgramBlock) : Nat :=
  match block with
  | ProgramBlock.Output => 1
  | ProgramBlock.Processing => 1
  | ProgramBlock.Decision => 2
  | ProgramBlock.StartEnd => 0

/-- Theorem stating that only the Decision block has two exits -/
theorem decision_block_two_exits :
  ∀ (block : ProgramBlock), num_exits block = 2 ↔ block = ProgramBlock.Decision :=
by sorry

/-- Corollary: No other block type has two exits -/
theorem other_blocks_not_two_exits :
  ∀ (block : ProgramBlock), block ≠ ProgramBlock.Decision → num_exits block ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_decision_block_two_exits_other_blocks_not_two_exits_l367_36778


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_false_proposition_3_l367_36739

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations and operations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect : Line → Line → Prop → Prop)
variable (point_on : Prop → Line → Prop)
variable (coplanar : Line → Line → Prop)

-- Theorem 1
theorem proposition_1 (m l : Line) (α : Plane) (A : Prop) :
  contains α m →
  perpendicular l α →
  point_on A l →
  ¬point_on A m →
  ¬coplanar l m :=
sorry

-- Theorem 2
theorem proposition_2_false (l m : Line) (α β : Plane) :
  ¬(∀ (l m : Line) (α β : Plane),
    parallel l α →
    parallel m β →
    parallel_planes α β →
    parallel_lines l m) :=
sorry

-- Theorem 3
theorem proposition_3 (l m : Line) (α β : Plane) (A : Prop) :
  contains α l →
  contains α m →
  intersect l m A →
  parallel l β →
  parallel m β →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_false_proposition_3_l367_36739


namespace NUMINAMATH_CALUDE_small_semicircle_radius_l367_36730

/-- Configuration of tangent shapes --/
structure TangentShapes where
  R : ℝ  -- Radius of large semicircle
  r : ℝ  -- Radius of circle
  x : ℝ  -- Radius of small semicircle

/-- Predicate for valid configuration --/
def is_valid_config (shapes : TangentShapes) : Prop :=
  shapes.R = 12 ∧ shapes.r = 6 ∧ shapes.x > 0

/-- Theorem stating the radius of the small semicircle --/
theorem small_semicircle_radius (shapes : TangentShapes) 
  (h : is_valid_config shapes) : shapes.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_semicircle_radius_l367_36730


namespace NUMINAMATH_CALUDE_ramsey_bound_exists_l367_36754

/-- The maximum degree of a graph -/
def maxDegree (G : SimpleGraph α) : ℕ := sorry

/-- The Ramsey number of a graph -/
def ramseyNumber (G : SimpleGraph α) : ℕ := sorry

/-- The order (number of vertices) of a graph -/
def graphOrder (G : SimpleGraph α) : ℕ := sorry

/-- For every positive integer Δ, there exists a constant c such that
    all graphs H with maximum degree at most Δ have R(H) ≤ c|H| -/
theorem ramsey_bound_exists {α : Type*} :
  ∀ Δ : ℕ, Δ > 0 →
  ∃ c : ℝ, c > 0 ∧
  ∀ (H : SimpleGraph α), maxDegree H ≤ Δ →
  (ramseyNumber H : ℝ) ≤ c * (graphOrder H) :=
sorry

end NUMINAMATH_CALUDE_ramsey_bound_exists_l367_36754


namespace NUMINAMATH_CALUDE_problem_solution_l367_36777

theorem problem_solution (x y z : ℝ) :
  (1.5 * x = 0.3 * y) →
  (x = 20) →
  (0.6 * y = z) →
  z = 60 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l367_36777


namespace NUMINAMATH_CALUDE_valid_digits_l367_36736

/-- Given a digit x, construct the number 20x06 -/
def construct_number (x : Nat) : Nat := 20000 + x * 100 + 6

/-- Predicate to check if a given digit satisfies the divisibility condition -/
def is_valid_digit (x : Nat) : Prop :=
  x < 10 ∧ (construct_number x) % 7 = 0

theorem valid_digits :
  ∀ x, is_valid_digit x ↔ (x = 0 ∨ x = 7) :=
sorry

end NUMINAMATH_CALUDE_valid_digits_l367_36736


namespace NUMINAMATH_CALUDE_email_difference_l367_36711

def morning_emails : ℕ := 10
def afternoon_emails : ℕ := 7

theorem email_difference : morning_emails - afternoon_emails = 3 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_l367_36711


namespace NUMINAMATH_CALUDE_gcd_102_238_l367_36768

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l367_36768


namespace NUMINAMATH_CALUDE_largest_special_number_last_digit_l367_36700

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_three_digits (n : ℕ) : ℕ := n / 10

theorem largest_special_number_last_digit :
  ∃ (n : ℕ), is_four_digit n ∧ 
             n % 9 = 0 ∧ 
             (first_three_digits n) % 4 = 0 ∧
             ∀ (m : ℕ), (is_four_digit m ∧ 
                         m % 9 = 0 ∧ 
                         (first_three_digits m) % 4 = 0) → 
                         m ≤ n ∧
             n % 10 = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_special_number_last_digit_l367_36700


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l367_36738

/-- Represents the maximum distance a car can travel with tire switching -/
def max_distance (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  min front_tire_life rear_tire_life

/-- Theorem stating the maximum distance a car can travel with specific tire lifespans -/
theorem max_distance_for_given_tires :
  max_distance 42000 56000 = 42000 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l367_36738


namespace NUMINAMATH_CALUDE_binomial_11_10_l367_36727

theorem binomial_11_10 : Nat.choose 11 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_binomial_11_10_l367_36727


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l367_36716

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation for a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "subset of" relation for a line and a plane
variable (subset_of : Line → Plane → Prop)

variable (α β : Plane)
variable (m : Line)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary
  (h1 : α ≠ β)
  (h2 : subset_of m α) :
  (parallel_planes α β → parallel_line_plane m β) ∧
  ¬(parallel_line_plane m β → parallel_planes α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l367_36716


namespace NUMINAMATH_CALUDE_trigonometric_identities_l367_36772

theorem trigonometric_identities (α β γ : Real) (h : α + β + γ = Real.pi) :
  (Real.cos α + Real.cos β + Real.cos γ = 4 * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) + 1) ∧
  (Real.cos α + Real.cos β - Real.cos γ = 4 * Real.cos (α/2) * Real.cos (β/2) * Real.sin (γ/2) - 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l367_36772


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l367_36748

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 3 - 6 * Complex.I) :
  z.im = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l367_36748


namespace NUMINAMATH_CALUDE_rectangle_area_l367_36771

/-- Given a rectangle with length 16 and diagonal 20, prove its area is 192. -/
theorem rectangle_area (length width diagonal : ℝ) : 
  length = 16 → 
  diagonal = 20 → 
  length^2 + width^2 = diagonal^2 → 
  length * width = 192 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l367_36771


namespace NUMINAMATH_CALUDE_park_walk_distance_l367_36701

theorem park_walk_distance (area : ℝ) (π_approx : ℝ) (extra_distance : ℝ) : 
  area = 616 →
  π_approx = 22 / 7 →
  extra_distance = 3 →
  2 * π_approx * (area / π_approx).sqrt + extra_distance = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_park_walk_distance_l367_36701


namespace NUMINAMATH_CALUDE_day_crew_load_fraction_l367_36782

/-- 
Proves that the day crew loads 8/11 of all boxes given the conditions about night and day crews.
-/
theorem day_crew_load_fraction 
  (D : ℝ) -- Number of boxes loaded by each day crew worker
  (W : ℝ) -- Number of workers in the day crew
  (h1 : D > 0) -- Assumption that D is positive
  (h2 : W > 0) -- Assumption that W is positive
  : (D * W) / ((D * W) + ((3/4 * D) * (1/2 * W))) = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_day_crew_load_fraction_l367_36782


namespace NUMINAMATH_CALUDE_cookie_recipe_average_l367_36794

/-- Represents the cookie recipe and calculates the average pieces per cookie. -/
def average_pieces_per_cookie (total_cookies : ℕ) (chocolate_chips : ℕ) : ℚ :=
  let mms : ℕ := chocolate_chips / 3
  let white_chips : ℕ := mms / 2
  let raisins : ℕ := white_chips * 2
  let total_pieces : ℕ := chocolate_chips + mms + white_chips + raisins
  (total_pieces : ℚ) / total_cookies

/-- Theorem stating that the average pieces per cookie is 4.125 given the specified recipe. -/
theorem cookie_recipe_average :
  average_pieces_per_cookie 48 108 = 4.125 := by
  sorry


end NUMINAMATH_CALUDE_cookie_recipe_average_l367_36794


namespace NUMINAMATH_CALUDE_franks_initial_money_l367_36765

/-- Frank's initial amount of money -/
def initial_money : ℕ := sorry

/-- The amount Frank spent on toys -/
def money_spent : ℕ := 8

/-- The amount Frank had left after spending -/
def money_left : ℕ := 8

/-- Theorem stating that Frank's initial money was $16 -/
theorem franks_initial_money : initial_money = 16 := by sorry

end NUMINAMATH_CALUDE_franks_initial_money_l367_36765


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l367_36796

theorem cube_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 20)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 13 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l367_36796


namespace NUMINAMATH_CALUDE_seven_row_triangle_pieces_l367_36763

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Calculates the sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Represents the structure of the triangle -/
structure TriangleStructure where
  rows : ℕ
  rodIncrease : ℕ
  extraConnectorRow : ℕ

/-- Calculates the total number of pieces in the triangle -/
def totalPieces (t : TriangleStructure) : ℕ :=
  let totalRods := arithmeticSum 3 t.rodIncrease t.rows
  let totalConnectors := triangularNumber (t.rows + t.extraConnectorRow)
  totalRods + totalConnectors

/-- The main theorem to prove -/
theorem seven_row_triangle_pieces :
  let t : TriangleStructure := {
    rows := 7,
    rodIncrease := 3,
    extraConnectorRow := 1
  }
  totalPieces t = 120 := by sorry

end NUMINAMATH_CALUDE_seven_row_triangle_pieces_l367_36763


namespace NUMINAMATH_CALUDE_apple_street_length_in_km_l367_36725

/-- The length of Apple Street in meters -/
def apple_street_length : ℝ := 3200

/-- The distance between intersections in meters -/
def intersection_distance : ℝ := 200

/-- The number of numbered intersections -/
def numbered_intersections : ℕ := 15

/-- The total number of intersections -/
def total_intersections : ℕ := numbered_intersections + 1

theorem apple_street_length_in_km :
  apple_street_length / 1000 = 3.2 := by sorry

end NUMINAMATH_CALUDE_apple_street_length_in_km_l367_36725


namespace NUMINAMATH_CALUDE_smallest_angle_25_sided_polygon_l367_36714

/-- Represents a convex polygon with n sides and angles in an arithmetic sequence --/
structure ConvexPolygon (n : ℕ) where
  -- The common difference of the arithmetic sequence of angles
  d : ℕ
  -- The smallest angle in the polygon
  smallest_angle : ℕ
  -- Ensure the polygon is convex (all angles less than 180°)
  convex : smallest_angle + (n - 1) * d < 180
  -- Ensure the sum of angles is correct for an n-sided polygon
  angle_sum : smallest_angle * n + (n * (n - 1) * d) / 2 = (n - 2) * 180

theorem smallest_angle_25_sided_polygon :
  ∃ (p : ConvexPolygon 25), p.smallest_angle = 154 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_25_sided_polygon_l367_36714


namespace NUMINAMATH_CALUDE_bob_investment_l367_36706

theorem bob_investment (fund_a_initial : ℝ) (fund_a_interest : ℝ) (fund_b_interest : ℝ) (difference : ℝ) :
  fund_a_initial = 2000 →
  fund_a_interest = 0.12 →
  fund_b_interest = 0.30 →
  difference = 549.9999999999998 →
  ∃ (fund_b_initial : ℝ),
    fund_a_initial * (1 + fund_a_interest) = 
    fund_b_initial * (1 + fund_b_interest)^2 + difference ∧
    fund_b_initial = 1000 :=
by sorry

end NUMINAMATH_CALUDE_bob_investment_l367_36706


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l367_36712

theorem average_age_when_youngest_born (total_people : ℕ) (current_average_age : ℝ) (youngest_age : ℝ) :
  total_people = 7 →
  current_average_age = 30 →
  youngest_age = 7 →
  (total_people * current_average_age - (total_people - 1) * youngest_age) / (total_people - 1) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l367_36712


namespace NUMINAMATH_CALUDE_monotonicity_of_f_l367_36752

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

theorem monotonicity_of_f (a : ℝ) :
  (a > 0 → (∀ x y, x < y → x < -2*a/3 → f a x < f a y) ∧
            (∀ x y, x < y → 0 < x → f a x < f a y) ∧
            (∀ x y, -2*a/3 < x → x < y → y < 0 → f a x > f a y)) ∧
  (a = 0 → (∀ x y, x < y → f a x < f a y)) ∧
  (a < 0 → (∀ x y, x < y → y < 0 → f a x < f a y) ∧
            (∀ x y, x < y → -2*a/3 < x → f a x < f a y) ∧
            (∀ x y, 0 < x → x < y → y < -2*a/3 → f a x > f a y)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_of_f_l367_36752


namespace NUMINAMATH_CALUDE_analysis_time_proof_l367_36755

/-- The number of bones in a human body -/
def num_bones : ℕ := 206

/-- The time (in hours) required to analyze one bone -/
def time_per_bone : ℕ := 1

/-- The total time required to analyze all bones in a human body -/
def total_analysis_time : ℕ := num_bones * time_per_bone

theorem analysis_time_proof : total_analysis_time = 206 := by
  sorry

end NUMINAMATH_CALUDE_analysis_time_proof_l367_36755


namespace NUMINAMATH_CALUDE_exact_three_blue_marbles_probability_l367_36733

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def num_trials : ℕ := 6
def num_blue_selections : ℕ := 3

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exact_three_blue_marbles_probability :
  Nat.choose num_trials num_blue_selections *
  (prob_blue ^ num_blue_selections) *
  (prob_red ^ (num_trials - num_blue_selections)) =
  3512320 / 11390625 := by
  sorry

end NUMINAMATH_CALUDE_exact_three_blue_marbles_probability_l367_36733


namespace NUMINAMATH_CALUDE_connie_calculation_l367_36786

theorem connie_calculation (x : ℤ) : x + 2 = 80 → x - 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_connie_calculation_l367_36786


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l367_36717

theorem least_subtrahend_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem problem_solution : 
  ∃ (k : Nat), k < 37 ∧ (1234567 - k) % 37 = 0 ∧ ∀ (m : Nat), m < k → (1234567 - m) % 37 ≠ 0 ∧ k = 13 :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l367_36717


namespace NUMINAMATH_CALUDE_inequalities_proof_l367_36703

theorem inequalities_proof (a b c : ℝ) (ha : a > 0) (hbc : a < b ∧ b < c) : 
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a + b < b + c) ∧ 
  (c / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l367_36703


namespace NUMINAMATH_CALUDE_total_fireworks_count_l367_36785

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of boxes Cherie has -/
def cherie_boxes : ℕ := 1

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box) + 
  cherie_boxes * (cherie_sparklers + cherie_whistlers)

theorem total_fireworks_count : total_fireworks = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_fireworks_count_l367_36785


namespace NUMINAMATH_CALUDE_david_average_marks_l367_36735

def david_marks : List ℝ := [70, 63, 80, 63, 65]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℝ) = 68.2 := by
  sorry

end NUMINAMATH_CALUDE_david_average_marks_l367_36735


namespace NUMINAMATH_CALUDE_parabola_points_product_l367_36756

/-- Two distinct points on a parabola with opposite slopes to a fixed point -/
structure ParabolaPoints where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  distinct : (x₁, y₁) ≠ (x₂, y₂)
  on_parabola₁ : y₁^2 = x₁
  on_parabola₂ : y₂^2 = x₂
  same_side : y₁ * y₂ > 0
  opposite_slopes : (y₁ / (x₁ - 1)) = -(y₂ / (x₂ - 1))

/-- The product of y-coordinates equals 1 -/
theorem parabola_points_product (p : ParabolaPoints) : p.y₁ * p.y₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_product_l367_36756


namespace NUMINAMATH_CALUDE_least_value_x_l367_36720

theorem least_value_x (x y z : ℕ+) (hy : y = 7) (h_least : ∀ (a b c : ℕ+), a - b - c ≥ x - y - z → a - b - c ≥ 17) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_least_value_x_l367_36720


namespace NUMINAMATH_CALUDE_investment_split_l367_36744

/-- Proves the amount invested at 6% given total investment, interest rates, and total interest earned --/
theorem investment_split (total_investment : ℝ) (rate1 rate2 : ℝ) (total_interest : ℝ) 
  (h1 : total_investment = 15000)
  (h2 : rate1 = 0.06)
  (h3 : rate2 = 0.075)
  (h4 : total_interest = 1023)
  (h5 : ∃ (x y : ℝ), x + y = total_investment ∧ 
                     rate1 * x + rate2 * y = total_interest) :
  ∃ (x : ℝ), x = 6800 ∧ 
              ∃ (y : ℝ), y = total_investment - x ∧
                          rate1 * x + rate2 * y = total_interest :=
sorry

end NUMINAMATH_CALUDE_investment_split_l367_36744


namespace NUMINAMATH_CALUDE_min_sum_of_digits_f_l367_36780

def f (n : ℕ) : ℕ := 17 * n^2 - 11 * n + 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem min_sum_of_digits_f :
  ∀ n : ℕ, sum_of_digits (f n) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_f_l367_36780


namespace NUMINAMATH_CALUDE_smallest_number_proof_l367_36709

theorem smallest_number_proof (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : a + b + c = 73)
  (h4 : c - b = 5)
  (h5 : b - a = 6) :
  a = 56 / 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l367_36709


namespace NUMINAMATH_CALUDE_twenty_dollar_bills_l367_36743

theorem twenty_dollar_bills (total_amount : ℕ) (bill_denomination : ℕ) (h1 : total_amount = 280) (h2 : bill_denomination = 20) :
  total_amount / bill_denomination = 14 := by
sorry

end NUMINAMATH_CALUDE_twenty_dollar_bills_l367_36743


namespace NUMINAMATH_CALUDE_age_sum_proof_l367_36721

/-- Given that Ashley's age is 8 and the ratio of Ashley's age to Mary's age is 4:7,
    prove that the sum of their ages is 22. -/
theorem age_sum_proof (ashley_age mary_age : ℕ) : 
  ashley_age = 8 → 
  ashley_age * 7 = mary_age * 4 → 
  ashley_age + mary_age = 22 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l367_36721


namespace NUMINAMATH_CALUDE_pointDifference_l367_36746

/-- Represents a team's performance in a soccer tournament --/
structure TeamPerformance where
  wins : ℕ
  draws : ℕ

/-- Calculates the total points for a team based on their performance --/
def calculatePoints (team : TeamPerformance) : ℕ :=
  team.wins * 3 + team.draws * 1

/-- The scoring system and match results for Joe's team and the first-place team --/
def joesTeam : TeamPerformance := ⟨1, 3⟩
def firstPlaceTeam : TeamPerformance := ⟨2, 2⟩

/-- The theorem stating the difference in points between the first-place team and Joe's team --/
theorem pointDifference : calculatePoints firstPlaceTeam - calculatePoints joesTeam = 2 := by
  sorry

end NUMINAMATH_CALUDE_pointDifference_l367_36746


namespace NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l367_36749

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (-1/8, 1/2)

/-- First line equation: y = -4x -/
def line1 (x y : ℚ) : Prop := y = -4 * x

/-- Second line equation: y - 2 = 12x -/
def line2 (x y : ℚ) : Prop := y - 2 = 12 * x

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l367_36749


namespace NUMINAMATH_CALUDE_distribute_three_books_twelve_students_l367_36799

/-- The number of ways to distribute n identical objects among k people,
    where no person can receive more than one object. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose k n

theorem distribute_three_books_twelve_students :
  distribute 3 12 = 220 := by
  sorry

end NUMINAMATH_CALUDE_distribute_three_books_twelve_students_l367_36799


namespace NUMINAMATH_CALUDE_box_paint_area_l367_36793

/-- The total area to paint inside a cuboid box -/
def total_paint_area (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height) + length * width

/-- Theorem: The total area to paint inside a cuboid box with dimensions 18 cm long, 10 cm wide, and 2 cm high is 292 square centimeters -/
theorem box_paint_area :
  total_paint_area 18 10 2 = 292 := by
  sorry

end NUMINAMATH_CALUDE_box_paint_area_l367_36793


namespace NUMINAMATH_CALUDE_dans_age_proof_l367_36775

/-- Dan's present age -/
def dans_age : ℕ := 8

/-- Theorem stating that Dan's age after 20 years will be 7 times his age 4 years ago -/
theorem dans_age_proof : dans_age + 20 = 7 * (dans_age - 4) := by
  sorry

end NUMINAMATH_CALUDE_dans_age_proof_l367_36775


namespace NUMINAMATH_CALUDE_students_per_table_is_three_l367_36790

/-- The number of students sitting at each table in Miss Smith's English class --/
def students_per_table : ℕ :=
  let total_students : ℕ := 47
  let num_tables : ℕ := 6
  let students_in_bathroom : ℕ := 3
  let students_in_canteen : ℕ := 3 * students_in_bathroom
  let new_students : ℕ := 2 * 4
  let foreign_exchange_students : ℕ := 3 * 3
  let absent_students : ℕ := students_in_bathroom + students_in_canteen + new_students + foreign_exchange_students
  let present_students : ℕ := total_students - absent_students
  present_students / num_tables

theorem students_per_table_is_three : students_per_table = 3 := by
  sorry

end NUMINAMATH_CALUDE_students_per_table_is_three_l367_36790


namespace NUMINAMATH_CALUDE_not_perfect_square_l367_36787

-- Define a function to create a number with n ones
def ones (n : ℕ) : ℕ := 
  (10^n - 1) / 9

-- Define our specific number N
def N (k : ℕ) : ℕ := 
  ones 300 * 10^k

-- Theorem statement
theorem not_perfect_square (k : ℕ) : 
  ¬ ∃ (m : ℕ), N k = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l367_36787


namespace NUMINAMATH_CALUDE_cubic_factorization_l367_36702

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l367_36702


namespace NUMINAMATH_CALUDE_average_weight_increase_l367_36783

/-- Proves that replacing a person weighing 76 kg with a person weighing 119.4 kg
    in a group of 7 people increases the average weight by 6.2 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 7 * initial_average
  let new_total := initial_total - 76 + 119.4
  let new_average := new_total / 7
  new_average - initial_average = 6.2 := by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l367_36783


namespace NUMINAMATH_CALUDE_triangle_side_length_l367_36741

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  Real.cos A = 3/5 ∧
  Real.sin B = Real.sqrt 5 / 5 ∧
  a = 2 →
  c = 11 * Real.sqrt 5 / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l367_36741


namespace NUMINAMATH_CALUDE_number_of_d_values_l367_36731

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def value (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + b * 10 + c

def sum_equation (a b c d : ℕ) : Prop :=
  value a b b c + value b c d b = value d a c d

def one_carryover (a b c d : ℕ) : Prop :=
  (a + b) % 10 = d ∧ a + b ≥ 10

theorem number_of_d_values :
  ∃ (s : Finset ℕ),
    (∀ a b c d : ℕ,
      is_digit a → is_digit b → is_digit c → is_digit d →
      distinct a b c d →
      sum_equation a b c d →
      one_carryover a b c d →
      d ∈ s) ∧
    s.card = 5 := by sorry

end NUMINAMATH_CALUDE_number_of_d_values_l367_36731


namespace NUMINAMATH_CALUDE_compute_expression_l367_36762

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l367_36762


namespace NUMINAMATH_CALUDE_ali_baba_max_coins_l367_36719

/-- Represents the game state -/
structure GameState where
  piles : List Nat
  deriving Repr

/-- The initial game state -/
def initialState : GameState :=
  { piles := List.replicate 10 10 }

/-- Ali Baba's strategy -/
def aliBabaStrategy (state : GameState) : GameState :=
  sorry

/-- Thief's strategy -/
def thiefStrategy (state : GameState) : GameState :=
  sorry

/-- Play the game for a given number of rounds -/
def playGame (rounds : Nat) : GameState :=
  sorry

/-- Calculate the maximum number of coins Ali Baba can take -/
def maxCoinsAliBaba (finalState : GameState) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem ali_baba_max_coins :
  ∃ (rounds : Nat), maxCoinsAliBaba (playGame rounds) = 72 :=
sorry

end NUMINAMATH_CALUDE_ali_baba_max_coins_l367_36719


namespace NUMINAMATH_CALUDE_teresas_age_at_birth_l367_36795

/-- Proves Teresa's age when Michiko was born, given the current ages and Morio's age at Michiko's birth -/
theorem teresas_age_at_birth (teresa_current_age morio_current_age morio_age_at_birth : ℕ) 
  (h1 : teresa_current_age = 59)
  (h2 : morio_current_age = 71)
  (h3 : morio_age_at_birth = 38) :
  teresa_current_age - (morio_current_age - morio_age_at_birth) = 26 := by
  sorry

#check teresas_age_at_birth

end NUMINAMATH_CALUDE_teresas_age_at_birth_l367_36795


namespace NUMINAMATH_CALUDE_sequence_property_main_theorem_l367_36723

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then 1 else (1 / 3) * (4 / 3) ^ (n - 2)

def sequence_S (n : ℕ) : ℚ := (4 / 3) ^ (n - 1)

theorem sequence_property : ∀ n : ℕ, n ≥ 1 → 3 * sequence_a (n + 1) = sequence_S n :=
  sorry

theorem main_theorem : ∀ n : ℕ, n ≥ 1 → 
  sequence_a n = if n = 1 then 1 else (1 / 3) * (4 / 3) ^ (n - 2) :=
  sorry

end NUMINAMATH_CALUDE_sequence_property_main_theorem_l367_36723


namespace NUMINAMATH_CALUDE_complex_equation_solution_l367_36757

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l367_36757


namespace NUMINAMATH_CALUDE_dogwood_trees_after_planting_l367_36776

/-- The number of dogwood trees in the park after planting -/
def total_trees (current : ℕ) (today : ℕ) (tomorrow : ℕ) : ℕ :=
  current + today + tomorrow

/-- Theorem stating that the total number of dogwood trees after planting is 100 -/
theorem dogwood_trees_after_planting :
  total_trees 39 41 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_after_planting_l367_36776


namespace NUMINAMATH_CALUDE_photo_calculation_l367_36788

theorem photo_calculation (total_photos : ℕ) (claire_photos : ℕ) : 
  (claire_photos : ℚ) + 3 * claire_photos + 5/4 * claire_photos + 
  5/2 * 5/4 * claire_photos + (claire_photos + 3 * claire_photos) / 2 + 
  (claire_photos + 3 * claire_photos) / 4 = total_photos ∧ total_photos = 840 → 
  claire_photos = 74 := by
sorry

end NUMINAMATH_CALUDE_photo_calculation_l367_36788


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l367_36732

/-- Represents the number of tickets sold for a theater performance --/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total revenue from ticket sales --/
def totalRevenue (tickets : TheaterTickets) : ℕ :=
  12 * tickets.orchestra + 8 * tickets.balcony

/-- Represents the conditions of the theater ticket sales --/
structure TicketSalesConditions where
  tickets : TheaterTickets
  totalRevenue : ℕ
  balconyExcess : ℕ

/-- The theorem to be proved --/
theorem theater_ticket_sales 
  (conditions : TicketSalesConditions) 
  (h1 : totalRevenue conditions.tickets = conditions.totalRevenue)
  (h2 : conditions.tickets.balcony = conditions.tickets.orchestra + conditions.balconyExcess)
  (h3 : conditions.totalRevenue = 3320)
  (h4 : conditions.balconyExcess = 115) :
  conditions.tickets.orchestra + conditions.tickets.balcony = 355 := by
  sorry

#check theater_ticket_sales

end NUMINAMATH_CALUDE_theater_ticket_sales_l367_36732


namespace NUMINAMATH_CALUDE_range_of_quadratic_l367_36751

/-- The quadratic function under consideration -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The domain of the function -/
def domain : Set ℝ := Set.Ioc 1 4

/-- The range of the function on the given domain -/
def range : Set ℝ := f '' domain

theorem range_of_quadratic : range = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_range_of_quadratic_l367_36751


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l367_36708

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Theorem: 72 is the least positive integer with exactly 12 positive factors -/
theorem least_integer_with_12_factors :
  (∀ m : ℕ+, m < 72 → num_factors m ≠ 12) ∧ num_factors 72 = 12 := by sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l367_36708


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l367_36784

theorem quadratic_roots_expression (x₁ x₂ : ℝ) 
  (h1 : x₁^2 + 5*x₁ + 1 = 0) 
  (h2 : x₂^2 + 5*x₂ + 1 = 0) : 
  (x₁*Real.sqrt 6 / (1 + x₂))^2 + (x₂*Real.sqrt 6 / (1 + x₁))^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l367_36784


namespace NUMINAMATH_CALUDE_max_reflections_l367_36760

/-- Represents the angle between lines AD and CD in degrees -/
def angle_CDA : ℝ := 12

/-- Represents the number of reflections -/
def n : ℕ := 7

/-- Theorem stating that n is the maximum number of reflections possible -/
theorem max_reflections (angle : ℝ) (num_reflections : ℕ) :
  angle = angle_CDA →
  num_reflections = n →
  (∀ m : ℕ, m > num_reflections → angle * m > 90) ∧
  angle * num_reflections ≤ 90 :=
sorry

#check max_reflections

end NUMINAMATH_CALUDE_max_reflections_l367_36760


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l367_36770

theorem quadratic_inequality_coefficient_sum (a b : ℝ) :
  (∀ x, ax^2 + b*x - 4 > 0 ↔ 1 < x ∧ x < 2) →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l367_36770


namespace NUMINAMATH_CALUDE_german_enrollment_l367_36767

theorem german_enrollment (total_students : ℕ) (both_subjects : ℕ) (only_english : ℕ) 
  (h1 : total_students = 40)
  (h2 : both_subjects = 12)
  (h3 : only_english = 18)
  (h4 : ∃ (german : ℕ), german > 0)
  (h5 : total_students = both_subjects + only_english + (total_students - both_subjects - only_english)) :
  total_students - only_english = 22 := by
  sorry

end NUMINAMATH_CALUDE_german_enrollment_l367_36767


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l367_36774

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

theorem complement_of_M_in_U : Set.compl M = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l367_36774


namespace NUMINAMATH_CALUDE_calculate_expression_l367_36737

theorem calculate_expression : 15 * (216 / 3 + 36 / 9 + 16 / 25 + 2^2) = 30240 / 25 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l367_36737


namespace NUMINAMATH_CALUDE_max_angle_between_tangents_l367_36753

/-- The parabola C₁ defined by y² = 4x -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The circle C₂ defined by (x-3)² + y² = 2 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 2}

/-- The angle between two tangents drawn from a point to a circle -/
def angleBetweenTangents (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The maximum angle between tangents theorem -/
theorem max_angle_between_tangents :
  ∃ (θ : ℝ), θ = 60 * π / 180 ∧
  ∀ (p : ℝ × ℝ), p ∈ C₁ →
    angleBetweenTangents p C₂ ≤ θ ∧
    ∃ (q : ℝ × ℝ), q ∈ C₁ ∧ angleBetweenTangents q C₂ = θ :=
  sorry

end NUMINAMATH_CALUDE_max_angle_between_tangents_l367_36753


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l367_36792

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 1 < 0 ↔ 1/2 < x ∧ x < 2) → a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l367_36792


namespace NUMINAMATH_CALUDE_total_cotton_yield_l367_36713

/-- 
Given two cotton fields:
- Field 1 has m hectares and produces an average of a kilograms per hectare
- Field 2 has n hectares and produces an average of b kilograms per hectare
This theorem proves that the total cotton yield is am + bn kilograms
-/
theorem total_cotton_yield 
  (m n a b : ℝ) 
  (h1 : m ≥ 0) 
  (h2 : n ≥ 0) 
  (h3 : a ≥ 0) 
  (h4 : b ≥ 0) : 
  m * a + n * b = m * a + n * b := by
  sorry

end NUMINAMATH_CALUDE_total_cotton_yield_l367_36713


namespace NUMINAMATH_CALUDE_D_72_l367_36797

/-- D(n) represents the number of ways to write n as a product of integers > 1, where order matters -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) = 48 -/
theorem D_72 : D 72 = 48 := by sorry

end NUMINAMATH_CALUDE_D_72_l367_36797


namespace NUMINAMATH_CALUDE_determinant_evaluation_l367_36707

theorem determinant_evaluation (x y z : ℝ) : 
  Matrix.det !![x + 1, y, z; y, x + 1, z; z, y, x + 1] = 
    x^3 + 3*x^2 + 3*x + 1 - x*y*z - x*y^2 - y*z^2 - z*x^2 - z*x + y*z^2 + z*y^2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_evaluation_l367_36707


namespace NUMINAMATH_CALUDE_remainder_problem_l367_36781

theorem remainder_problem (d r : ℤ) : 
  d > 1 ∧ 
  1225 % d = r ∧ 
  1681 % d = r ∧ 
  2756 % d = r → 
  d - r = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l367_36781


namespace NUMINAMATH_CALUDE_family_ages_l367_36750

structure Family where
  father_age : ℝ
  eldest_son_age : ℝ
  daughter_age : ℝ
  youngest_son_age : ℝ

def is_valid_family (f : Family) : Prop :=
  f.father_age = f.eldest_son_age + 20 ∧
  f.father_age + 2 = 2 * (f.eldest_son_age + 2) ∧
  f.daughter_age = f.eldest_son_age - 5 ∧
  f.youngest_son_age = f.daughter_age / 2

theorem family_ages : 
  ∃ (f : Family), is_valid_family f ∧ 
    f.father_age = 38 ∧ 
    f.eldest_son_age = 18 ∧ 
    f.daughter_age = 13 ∧ 
    f.youngest_son_age = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_l367_36750


namespace NUMINAMATH_CALUDE_probability_of_drawing_heart_l367_36718

-- Define the total number of cards
def total_cards : ℕ := 5

-- Define the number of heart cards
def heart_cards : ℕ := 3

-- Define the number of spade cards
def spade_cards : ℕ := 2

-- Theorem statement
theorem probability_of_drawing_heart :
  (heart_cards : ℚ) / total_cards = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_heart_l367_36718


namespace NUMINAMATH_CALUDE_unique_triangle_with_perimeter_8_l367_36758

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of an integer triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Two triangles are congruent if they have the same side lengths (up to permutation) -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.a ∧ t1.b = t2.c ∧ t1.c = t2.b) ∨
  (t1.a = t2.b ∧ t1.b = t2.a ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b) ∨
  (t1.a = t2.c ∧ t1.b = t2.b ∧ t1.c = t2.a)

theorem unique_triangle_with_perimeter_8 :
  ∃! (t : IntTriangle), perimeter t = 8 ∧
  ∀ (t' : IntTriangle), perimeter t' = 8 → congruent t t' := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_with_perimeter_8_l367_36758


namespace NUMINAMATH_CALUDE_football_game_cost_l367_36747

/-- The cost of a football game, given the total spent and the costs of two other games. -/
theorem football_game_cost (total_spent strategy_cost batman_cost : ℚ) :
  total_spent = 35.52 ∧ strategy_cost = 9.46 ∧ batman_cost = 12.04 →
  total_spent - (strategy_cost + batman_cost) = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_football_game_cost_l367_36747


namespace NUMINAMATH_CALUDE_no_common_points_condition_l367_36773

theorem no_common_points_condition (d : ℝ) : 
  (∀ x y : ℝ × ℝ, (x.1 - y.1)^2 + (x.2 - y.2)^2 = d^2 → 
    ((x.1^2 + x.2^2 ≤ 4 ∧ y.1^2 + y.2^2 ≤ 9) ∨ 
     (x.1^2 + x.2^2 ≤ 9 ∧ y.1^2 + y.2^2 ≤ 4)) → 
    (x.1^2 + x.2^2 - 4) * (y.1^2 + y.2^2 - 9) > 0) ↔ 
  (0 ≤ d ∧ d < 1) ∨ d > 5 :=
sorry

end NUMINAMATH_CALUDE_no_common_points_condition_l367_36773


namespace NUMINAMATH_CALUDE_stationery_store_pencils_l367_36729

theorem stationery_store_pencils (pens pencils markers : ℕ) : 
  pens * 6 = pencils * 5 →  -- ratio of pens to pencils is 5:6
  pens * 7 = markers * 5 →  -- ratio of pens to markers is 5:7
  pencils = pens + 4 →      -- 4 more pencils than pens
  markers = pens + 20 →     -- 20 more markers than pens
  pencils = 24 :=           -- prove that the number of pencils is 24
by sorry

end NUMINAMATH_CALUDE_stationery_store_pencils_l367_36729


namespace NUMINAMATH_CALUDE_john_memory_card_cost_l367_36724

/-- Calculates the amount spent on memory cards given the following conditions:
  * Pictures taken per day
  * Number of years
  * Images per memory card
  * Cost per memory card
-/
def memory_card_cost (pictures_per_day : ℕ) (years : ℕ) (images_per_card : ℕ) (cost_per_card : ℕ) : ℕ :=
  let total_pictures := pictures_per_day * years * 365
  let cards_needed := (total_pictures + images_per_card - 1) / images_per_card
  cards_needed * cost_per_card

/-- Theorem stating that under the given conditions, John spends $13140 on memory cards -/
theorem john_memory_card_cost :
  memory_card_cost 10 3 50 60 = 13140 := by
  sorry


end NUMINAMATH_CALUDE_john_memory_card_cost_l367_36724


namespace NUMINAMATH_CALUDE_initial_roses_count_l367_36798

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 3

/-- The number of roses after adding more flowers -/
def final_roses : ℕ := 12

/-- The number of orchids after adding more flowers -/
def final_orchids : ℕ := 2

/-- The difference between the number of roses and orchids after adding flowers -/
def rose_orchid_difference : ℕ := 10

theorem initial_roses_count : initial_roses = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_count_l367_36798


namespace NUMINAMATH_CALUDE_greatest_even_perfect_square_under_200_l367_36710

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem greatest_even_perfect_square_under_200 :
  ∀ n : ℕ, is_perfect_square n → is_even n → n < 200 → n ≤ 196 :=
sorry

end NUMINAMATH_CALUDE_greatest_even_perfect_square_under_200_l367_36710


namespace NUMINAMATH_CALUDE_average_production_before_today_l367_36722

theorem average_production_before_today 
  (n : ℕ) 
  (today_production : ℕ) 
  (new_average : ℕ) 
  (h1 : n = 9)
  (h2 : today_production = 90)
  (h3 : new_average = 45) :
  (n * (n + 1) * new_average - (n + 1) * today_production) / n = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_production_before_today_l367_36722


namespace NUMINAMATH_CALUDE_log_calculation_l367_36759

-- Define the common logarithm (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_calculation : log10 5 * log10 20 + (log10 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_calculation_l367_36759


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_l367_36742

theorem smallest_x_absolute_value : ∃ x : ℝ, 
  (∀ y : ℝ, |5*y - 3| = 15 → x ≤ y) ∧ |5*x - 3| = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_l367_36742


namespace NUMINAMATH_CALUDE_lucy_speed_calculation_l367_36734

-- Define the cycling speeds
def eugene_speed : ℚ := 5
def carlos_relative_speed : ℚ := 4/5
def lucy_relative_speed : ℚ := 6/7

-- Theorem to prove
theorem lucy_speed_calculation :
  let carlos_speed := eugene_speed * carlos_relative_speed
  let lucy_speed := carlos_speed * lucy_relative_speed
  lucy_speed = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_lucy_speed_calculation_l367_36734


namespace NUMINAMATH_CALUDE_large_number_proof_l367_36766

/-- A number composed of 80 hundred millions, 5 ten millions, and 6 ten thousands -/
def large_number : ℕ := 80 * 100000000 + 5 * 10000000 + 6 * 10000

/-- The same number expressed in units of ten thousand -/
def large_number_in_ten_thousands : ℕ := large_number / 10000

theorem large_number_proof :
  large_number = 8050060000 ∧ large_number_in_ten_thousands = 805006 := by
  sorry

end NUMINAMATH_CALUDE_large_number_proof_l367_36766


namespace NUMINAMATH_CALUDE_abs_sum_equals_eight_l367_36740

theorem abs_sum_equals_eight (x : ℝ) (θ : ℝ) (h : Real.log x / Real.log 3 = 1 + Real.sin θ) :
  |x - 1| + |x - 9| = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_equals_eight_l367_36740


namespace NUMINAMATH_CALUDE_quadratic_sets_solution_l367_36704

/-- Given sets A and B defined by quadratic equations, prove the values of a, b, and c -/
theorem quadratic_sets_solution :
  ∀ (a b c : ℝ),
  let A := {x : ℝ | x^2 + a*x + b = 0}
  let B := {x : ℝ | x^2 + c*x + 15 = 0}
  (A ∪ B = {3, 5} ∧ A ∩ B = {3}) →
  (a = -6 ∧ b = 9 ∧ c = -8) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_sets_solution_l367_36704


namespace NUMINAMATH_CALUDE_unique_number_sum_of_digits_l367_36789

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem unique_number_sum_of_digits :
  ∃! N : ℕ, 
    400 < N ∧ N < 600 ∧ 
    N % 2 = 1 ∧ 
    N % 5 = 0 ∧ 
    N % 11 = 0 ∧
    sum_of_digits N = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_sum_of_digits_l367_36789
