import Mathlib

namespace polynomial_simplification_l3830_383014

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 - 2 * q^3 + 3 * q^2 - 7 * q + 9) + (5 * q^3 - 8 * q^2 + 6 * q - 1) =
  4 * q^4 + 3 * q^3 - 5 * q^2 - q + 8 := by
  sorry

end polynomial_simplification_l3830_383014


namespace sum_of_squares_quadratic_solutions_l3830_383087

theorem sum_of_squares_quadratic_solutions : ∀ x₁ x₂ : ℝ, 
  x₁^2 - 16*x₁ + 15 = 0 → 
  x₂^2 - 16*x₂ + 15 = 0 → 
  x₁ ≠ x₂ → 
  x₁^2 + x₂^2 = 226 := by
  sorry

end sum_of_squares_quadratic_solutions_l3830_383087


namespace milk_ratio_is_two_fifths_l3830_383013

/-- The number of milk boxes Lolita drinks on weekdays -/
def weekday_boxes : ℕ := 3

/-- The number of milk boxes Lolita drinks on Sundays -/
def sunday_boxes : ℕ := 3 * weekday_boxes

/-- The total number of milk boxes Lolita drinks per week -/
def total_boxes : ℕ := 30

/-- The number of milk boxes Lolita drinks on Saturdays -/
def saturday_boxes : ℕ := total_boxes - (5 * weekday_boxes + sunday_boxes)

/-- The ratio of milk boxes on Saturdays to weekdays -/
def milk_ratio : ℚ := saturday_boxes / (5 * weekday_boxes)

theorem milk_ratio_is_two_fifths : milk_ratio = 2 / 5 := by sorry

end milk_ratio_is_two_fifths_l3830_383013


namespace length_of_DH_l3830_383056

-- Define the triangle and points
structure Triangle :=
  (A B C D E F G H : ℝ × ℝ)

-- Define the properties of the triangle and points
def EquilateralTriangle (t : Triangle) : Prop :=
  let d := Real.sqrt 3
  t.A = (0, 0) ∧ t.B = (2, 0) ∧ t.C = (1, d)

def PointsOnSides (t : Triangle) : Prop :=
  ∃ x y z w : ℝ,
    0 ≤ x ∧ x ≤ 2 ∧
    0 ≤ y ∧ y ≤ 2 ∧
    0 ≤ z ∧ z ≤ 2 ∧
    0 ≤ w ∧ w ≤ 2 ∧
    t.D = (x, 0) ∧
    t.F = (y, 0) ∧
    t.E = (1 - z/2, z * Real.sqrt 3 / 2) ∧
    t.G = (1 - w/2, w * Real.sqrt 3 / 2)

def ParallelLines (t : Triangle) : Prop :=
  (t.E.2 - t.D.2) / (t.E.1 - t.D.1) = Real.sqrt 3 ∧
  (t.G.2 - t.F.2) / (t.G.1 - t.F.1) = Real.sqrt 3

def SpecificLengths (t : Triangle) : Prop :=
  t.D.1 - t.A.1 = 0.5 ∧
  Real.sqrt ((t.E.1 - t.D.1)^2 + (t.E.2 - t.D.2)^2) = 1 ∧
  t.F.1 - t.D.1 = 0.5 ∧
  Real.sqrt ((t.G.1 - t.F.1)^2 + (t.G.2 - t.F.2)^2) = 1 ∧
  t.B.1 - t.F.1 = 0.5

def ParallelDH (t : Triangle) : Prop :=
  ∃ k : ℝ, t.H = (k * t.C.1 + (1 - k) * t.A.1, k * t.C.2 + (1 - k) * t.A.2)

-- State the theorem
theorem length_of_DH (t : Triangle) :
  EquilateralTriangle t →
  PointsOnSides t →
  ParallelLines t →
  SpecificLengths t →
  ParallelDH t →
  Real.sqrt ((t.H.1 - t.D.1)^2 + (t.H.2 - t.D.2)^2) = 1 :=
by sorry

end length_of_DH_l3830_383056


namespace books_read_first_week_l3830_383016

/-- The number of books read in the first week of a 7-week reading plan -/
def books_first_week (total_books : ℕ) (second_week : ℕ) (later_weeks : ℕ) : ℕ :=
  total_books - second_week - (later_weeks * 5)

theorem books_read_first_week :
  books_first_week 54 3 9 = 6 := by
  sorry

end books_read_first_week_l3830_383016


namespace no_infinite_sequence_positive_integers_l3830_383064

theorem no_infinite_sequence_positive_integers :
  ¬ ∃ (a : ℕ → ℕ+), ∀ (n : ℕ), (a (n-1))^2 ≥ 2 * (a n) * (a (n+2)) := by
  sorry

end no_infinite_sequence_positive_integers_l3830_383064


namespace prime_solution_equation_l3830_383033

theorem prime_solution_equation : 
  ∃! (p q : ℕ), Prime p ∧ Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ∧ p = 17 ∧ q = 3 := by
  sorry

end prime_solution_equation_l3830_383033


namespace race_completion_time_l3830_383022

theorem race_completion_time (total_runners : ℕ) (avg_time_all : ℝ) (fastest_time : ℝ) : 
  total_runners = 4 →
  avg_time_all = 30 →
  fastest_time = 15 →
  (((avg_time_all * total_runners) - fastest_time) / (total_runners - 1) : ℝ) = 35 :=
by
  sorry

end race_completion_time_l3830_383022


namespace game_points_theorem_l3830_383097

/-- The total points of four players in a game -/
def total_points (eric_points mark_points samanta_points daisy_points : ℕ) : ℕ :=
  eric_points + mark_points + samanta_points + daisy_points

/-- Theorem stating the total points of the four players given the conditions -/
theorem game_points_theorem (eric_points mark_points samanta_points daisy_points : ℕ) :
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points = mark_points + 8 →
  daisy_points = (samanta_points + mark_points + eric_points) - (samanta_points + mark_points + eric_points) / 4 →
  total_points eric_points mark_points samanta_points daisy_points = 56 :=
by
  sorry

#check game_points_theorem

end game_points_theorem_l3830_383097


namespace jake_final_bitcoin_count_l3830_383055

/-- Calculates the final number of bitcoins Jake has after a series of transactions -/
def final_bitcoin_count (initial : ℕ) (first_donation : ℕ) (second_donation : ℕ) : ℕ :=
  let after_first_donation := initial - first_donation
  let after_giving_to_brother := after_first_donation / 2
  let after_tripling := after_giving_to_brother * 3
  after_tripling - second_donation

/-- Theorem stating that Jake ends up with 80 bitcoins -/
theorem jake_final_bitcoin_count :
  final_bitcoin_count 80 20 10 = 80 := by
  sorry

end jake_final_bitcoin_count_l3830_383055


namespace fourth_side_length_l3830_383043

/-- A rhombus inscribed in a circle -/
structure InscribedRhombus where
  -- The radius of the circumscribed circle
  radius : ℝ
  -- The length of three sides of the rhombus
  side_length : ℝ
  -- Assumption that the rhombus is actually inscribed in the circle
  is_inscribed : True

/-- Theorem: In a rhombus inscribed in a circle with radius 100√2, 
    if three sides have length 100, then the fourth side also has length 100 -/
theorem fourth_side_length (r : InscribedRhombus) 
    (h1 : r.radius = 100 * Real.sqrt 2) 
    (h2 : r.side_length = 100) : 
  r.side_length = 100 := by
  sorry


end fourth_side_length_l3830_383043


namespace factor_ab_squared_minus_25a_l3830_383068

theorem factor_ab_squared_minus_25a (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) := by
  sorry

end factor_ab_squared_minus_25a_l3830_383068


namespace xy_value_l3830_383018

theorem xy_value (x y : ℝ) (h : Real.sqrt (x - 1) + (y - 2)^2 = 0) : x * y = 2 := by
  sorry

end xy_value_l3830_383018


namespace total_rejection_is_0_75_percent_l3830_383058

-- Define the rejection rates and inspection proportion
def john_rejection_rate : ℝ := 0.005
def jane_rejection_rate : ℝ := 0.009
def jane_inspection_proportion : ℝ := 0.625

-- Define the total rejection percentage
def total_rejection_percentage : ℝ :=
  jane_rejection_rate * jane_inspection_proportion +
  john_rejection_rate * (1 - jane_inspection_proportion)

-- Theorem statement
theorem total_rejection_is_0_75_percent :
  total_rejection_percentage = 0.0075 := by
  sorry

#eval total_rejection_percentage

end total_rejection_is_0_75_percent_l3830_383058


namespace snow_leopard_arrangement_l3830_383044

theorem snow_leopard_arrangement (n : ℕ) (h : n = 7) : 
  2 * Nat.factorial (n - 2) = 240 := by
  sorry

end snow_leopard_arrangement_l3830_383044


namespace geometric_sequence_first_term_l3830_383004

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_q_pos : q > 0)
  (h_condition : a 5 * a 7 = 4 * (a 4)^2)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 := by
sorry

end geometric_sequence_first_term_l3830_383004


namespace pool_filling_times_l3830_383069

theorem pool_filling_times (t₁ t₂ : ℝ) : 
  (t₁ > 0 ∧ t₂ > 0) →  -- Ensure positive times
  (1 / t₁ + 1 / t₂ = 1 / 2.4) →  -- Combined filling rate
  (t₂ / (4 * t₁) + t₁ / (4 * t₂) = 11 / 24) →  -- Fraction filled by individual operations
  (t₁ = 4 ∧ t₂ = 6) :=
by sorry

end pool_filling_times_l3830_383069


namespace f_minimum_and_inequality_l3830_383042

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

theorem f_minimum_and_inequality :
  (∃ (x_min : ℝ), ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = 1) ∧
  (∀ (x : ℝ), x > 0 → x * (Real.exp x) * f x + (x * Real.exp x - 1) * Real.log x - Real.exp x + 1/2 > 0) :=
sorry

end f_minimum_and_inequality_l3830_383042


namespace solution_f_gt_2_min_value_f_l3830_383078

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for the solution of f(x) > 2
theorem solution_f_gt_2 (x : ℝ) : f x > 2 ↔ x < -7 ∨ x > 5/3 := by sorry

-- Theorem for the minimum value of f
theorem min_value_f : ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -9/2 := by sorry

end solution_f_gt_2_min_value_f_l3830_383078


namespace opposite_values_imply_result_l3830_383028

theorem opposite_values_imply_result (a b : ℝ) : 
  |a + 2| = -(b - 3)^2 → a^b + 3*(a - b) = -23 := by
  sorry

end opposite_values_imply_result_l3830_383028


namespace geometric_mean_of_2_and_8_l3830_383054

theorem geometric_mean_of_2_and_8 : 
  ∃ (b : ℝ), b^2 = 2 * 8 ∧ (b = 4 ∨ b = -4) := by
  sorry

end geometric_mean_of_2_and_8_l3830_383054


namespace polygon_area_bounds_l3830_383046

/-- Represents a polygon in 2D space -/
structure Polygon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents the projections of a polygon -/
structure Projections where
  x_axis : ℝ
  y_axis : ℝ
  bisector_1_3 : ℝ
  bisector_2_4 : ℝ

/-- Given a polygon, return its projections -/
def get_projections (p : Polygon) : Projections :=
  sorry

/-- Calculate the area of a polygon -/
def area (p : Polygon) : ℝ :=
  sorry

/-- Check if a polygon is convex -/
def is_convex (p : Polygon) : Prop :=
  sorry

theorem polygon_area_bounds (M : Polygon) 
    (h_proj : get_projections M = Projections.mk 4 5 (3 * Real.sqrt 2) (4 * Real.sqrt 2)) : 
  (area M ≤ 17.5) ∧ (is_convex M → area M ≥ 10) := by
  sorry

end polygon_area_bounds_l3830_383046


namespace geologists_can_reach_station_l3830_383065

/-- Represents the problem of geologists traveling to a station. -/
structure GeologistsProblem where
  totalDistance : ℝ
  timeLimit : ℝ
  motorcycleSpeed : ℝ
  walkingSpeed : ℝ
  numberOfGeologists : ℕ

/-- Checks if the geologists can reach the station within the time limit. -/
def canReachStation (problem : GeologistsProblem) : Prop :=
  ∃ (strategy : Unit), 
    let twoGeologistsTime := problem.totalDistance / problem.motorcycleSpeed
    let walkingTime := problem.totalDistance / problem.walkingSpeed
    let meetingTime := (problem.totalDistance - problem.walkingSpeed) / (problem.motorcycleSpeed + problem.walkingSpeed)
    let returnTime := (problem.totalDistance - problem.walkingSpeed * meetingTime) / problem.motorcycleSpeed
    twoGeologistsTime ≤ problem.timeLimit ∧ 
    walkingTime ≤ problem.timeLimit ∧
    meetingTime + returnTime ≤ problem.timeLimit

/-- The specific problem instance. -/
def geologistsProblem : GeologistsProblem :=
  { totalDistance := 60
  , timeLimit := 3
  , motorcycleSpeed := 50
  , walkingSpeed := 5
  , numberOfGeologists := 3 }

/-- Theorem stating that the geologists can reach the station within the time limit. -/
theorem geologists_can_reach_station : canReachStation geologistsProblem := by
  sorry


end geologists_can_reach_station_l3830_383065


namespace max_current_speed_is_26_l3830_383029

/-- The maximum possible integer value for the river current speed --/
def max_current_speed : ℕ := 26

/-- The speed at which Mumbo runs --/
def mumbo_speed : ℕ := 11

/-- The speed at which Yumbo walks --/
def yumbo_speed : ℕ := 6

/-- Represents the travel scenario described in the problem --/
structure TravelScenario where
  x : ℝ  -- distance from origin to Mumbo's raft storage
  y : ℝ  -- distance from origin to Yumbo's raft storage
  v : ℕ  -- speed of the river current

/-- Condition that Yumbo arrives earlier than Mumbo --/
def yumbo_arrives_earlier (s : TravelScenario) : Prop :=
  s.y / yumbo_speed < s.x / mumbo_speed + (s.x + s.y) / s.v

/-- Main theorem stating that 26 is the maximum possible current speed --/
theorem max_current_speed_is_26 :
  ∀ s : TravelScenario,
    s.x > 0 ∧ s.y > 0 ∧ s.x < s.y ∧ s.v ≥ 6 ∧ yumbo_arrives_earlier s
    → s.v ≤ max_current_speed :=
by sorry

#check max_current_speed_is_26

end max_current_speed_is_26_l3830_383029


namespace trig_identities_l3830_383025

theorem trig_identities (θ : Real) (h : Real.sin (θ - π/3) = 1/3) :
  (Real.sin (θ + 2*π/3) = -1/3) ∧ (Real.cos (θ - 5*π/6) = 1/3) := by
  sorry

end trig_identities_l3830_383025


namespace no_leg_longer_than_both_l3830_383066

-- Define two right triangles
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Theorem statement
theorem no_leg_longer_than_both (t1 t2 : RightTriangle) 
  (h : t1.hypotenuse = t2.hypotenuse) : 
  ¬(t1.leg1 > t2.leg1 ∧ t1.leg1 > t2.leg2) ∨ 
  ¬(t1.leg2 > t2.leg1 ∧ t1.leg2 > t2.leg2) :=
sorry

end no_leg_longer_than_both_l3830_383066


namespace inequality_proof_l3830_383086

theorem inequality_proof (x y : ℝ) (n k : ℕ) 
  (h1 : x > y) (h2 : y > 0) (h3 : n > k) :
  (x^k - y^k)^n < (x^n - y^n)^k := by
  sorry

end inequality_proof_l3830_383086


namespace arithmetic_sequence_formula_l3830_383026

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  geometric_subsequence : (a 2) ^ 2 = a 1 * a 6
  sum_condition : 2 * a 1 + a 2 = 1

/-- The main theorem stating the explicit formula for the nth term -/
theorem arithmetic_sequence_formula (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 5/3 - n := by
  sorry

end arithmetic_sequence_formula_l3830_383026


namespace system_solution_l3830_383045

theorem system_solution (x y z : ℝ) : 
  x^2 + y^2 + z^2 = 1 ∧ x^3 + y^3 + z^3 = 1 → 
  (x = 1 ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = 1 ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = 1) :=
by sorry

end system_solution_l3830_383045


namespace sin_theta_value_l3830_383009

theorem sin_theta_value (θ : Real) 
  (h1 : 5 * Real.tan θ = 2 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = (-5 + Real.sqrt 41) / 4 := by
  sorry

end sin_theta_value_l3830_383009


namespace M_equals_interval_l3830_383001

/-- The set of real numbers m for which there exists an x in (-1, 1) satisfying x^2 - x - m = 0 -/
def M : Set ℝ := {m : ℝ | ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0}

/-- The theorem stating that M is equal to [-1/4, 2) -/
theorem M_equals_interval : M = Set.Icc (-1/4) 2 := by
  sorry

end M_equals_interval_l3830_383001


namespace statue_cost_l3830_383036

theorem statue_cost (selling_price : ℚ) (profit_percentage : ℚ) (original_cost : ℚ) : 
  selling_price = 670 ∧ 
  profit_percentage = 25 ∧ 
  selling_price = original_cost * (1 + profit_percentage / 100) → 
  original_cost = 536 := by
sorry

end statue_cost_l3830_383036


namespace line_equation_l3830_383073

theorem line_equation (slope_angle : Real) (y_intercept : Real) :
  slope_angle = Real.pi / 4 → y_intercept = 2 →
  ∃ f : Real → Real, f = λ x => x + 2 :=
by
  sorry

end line_equation_l3830_383073


namespace unknown_blanket_rate_l3830_383089

/-- Given the following conditions:
    - 3 blankets at Rs. 100 each
    - 6 blankets at Rs. 150 each
    - 2 blankets at an unknown rate
    - The average price of all blankets is Rs. 150
    Prove that the unknown rate must be Rs. 225 per blanket -/
theorem unknown_blanket_rate (price1 : ℕ) (price2 : ℕ) (unknown_price : ℕ) 
    (h1 : price1 = 100)
    (h2 : price2 = 150)
    (h3 : (3 * price1 + 6 * price2 + 2 * unknown_price) / 11 = 150) :
    unknown_price = 225 := by
  sorry

end unknown_blanket_rate_l3830_383089


namespace reciprocals_from_product_l3830_383082

theorem reciprocals_from_product (x y : ℝ) (h : x * y = 1) : x = 1 / y ∧ y = 1 / x := by
  sorry

end reciprocals_from_product_l3830_383082


namespace planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l3830_383050

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_to_plane : Plane → Plane → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel 
  (P Q R : Plane) 
  (h1 : parallel_to_plane P R) 
  (h2 : parallel_to_plane Q R) : 
  parallel_planes P Q :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_same_plane_are_parallel 
  (l1 l2 : Line) 
  (P : Plane) 
  (h1 : perpendicular_to_plane l1 P) 
  (h2 : perpendicular_to_plane l2 P) : 
  parallel_lines l1 l2 :=
sorry

end planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l3830_383050


namespace seeds_in_fourth_pot_l3830_383060

/-- Given 10 total seeds, 4 pots, and 3 seeds per pot for the first 3 pots,
    prove that the number of seeds in the fourth pot is 1. -/
theorem seeds_in_fourth_pot
  (total_seeds : ℕ)
  (num_pots : ℕ)
  (seeds_per_pot : ℕ)
  (h1 : total_seeds = 10)
  (h2 : num_pots = 4)
  (h3 : seeds_per_pot = 3)
  : total_seeds - (seeds_per_pot * (num_pots - 1)) = 1 := by
  sorry

end seeds_in_fourth_pot_l3830_383060


namespace polyhedron_20_faces_l3830_383030

/-- A polyhedron with triangular faces -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- The Euler characteristic for polyhedra -/
def euler_characteristic (p : Polyhedron) : ℕ :=
  p.vertices - p.edges + p.faces

/-- Theorem: A polyhedron with 20 triangular faces has 30 edges and 12 vertices -/
theorem polyhedron_20_faces (p : Polyhedron) 
  (h_faces : p.faces = 20) 
  (h_triangular : p.edges * 2 = p.faces * 3) 
  (h_euler : euler_characteristic p = 2) : 
  p.edges = 30 ∧ p.vertices = 12 := by
  sorry


end polyhedron_20_faces_l3830_383030


namespace three_intersection_points_l3830_383000

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 8

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
by sorry

end three_intersection_points_l3830_383000


namespace permutation_equation_solution_l3830_383083

theorem permutation_equation_solution (x : ℕ) : 
  (3 * (Nat.factorial 8 / Nat.factorial (8 - x)) = 4 * (Nat.factorial 9 / Nat.factorial (10 - x))) ∧ 
  (1 ≤ x) ∧ (x ≤ 8) → 
  x = 6 := by sorry

end permutation_equation_solution_l3830_383083


namespace total_segment_length_l3830_383076

-- Define the grid dimensions
def grid_width : ℕ := 5
def grid_height : ℕ := 6

-- Define the number of unit squares
def total_squares : ℕ := 30

-- Define the lengths of the six line segments
def segment_lengths : List ℕ := [5, 1, 4, 2, 3, 3]

-- Theorem statement
theorem total_segment_length :
  grid_width = 5 ∧ 
  grid_height = 6 ∧ 
  total_squares = 30 ∧ 
  segment_lengths = [5, 1, 4, 2, 3, 3] →
  List.sum segment_lengths = 18 := by
  sorry

end total_segment_length_l3830_383076


namespace milk_ratio_l3830_383077

def total_cartons : ℕ := 24
def regular_cartons : ℕ := 3

theorem milk_ratio :
  let chocolate_cartons := total_cartons - regular_cartons
  (chocolate_cartons : ℚ) / regular_cartons = 7 / 1 :=
by sorry

end milk_ratio_l3830_383077


namespace twelve_chairs_subsets_l3830_383049

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets containing at least four adjacent chairs
    for n chairs arranged in a circle -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs arranged in a circle,
    the number of subsets containing at least four adjacent chairs is 1701 -/
theorem twelve_chairs_subsets :
  subsets_with_adjacent_chairs n = 1701 := by sorry

end twelve_chairs_subsets_l3830_383049


namespace horner_method_result_l3830_383023

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem horner_method_result : f (-4) = 220 := by
  sorry

end horner_method_result_l3830_383023


namespace wrapping_paper_usage_l3830_383005

theorem wrapping_paper_usage
  (total_paper : ℚ)
  (small_presents : ℕ)
  (large_presents : ℕ)
  (h1 : total_paper = 5 / 12)
  (h2 : small_presents = 4)
  (h3 : large_presents = 2)
  (h4 : small_presents + large_presents = 6) :
  ∃ (small_paper large_paper : ℚ),
    small_paper * small_presents + large_paper * large_presents = total_paper ∧
    large_paper = 2 * small_paper ∧
    small_paper = 5 / 96 ∧
    large_paper = 5 / 48 := by
  sorry

end wrapping_paper_usage_l3830_383005


namespace range_of_m_l3830_383099

theorem range_of_m (p q : ℝ → Prop) (m : ℝ) 
  (hp : p = fun x => x^2 + x - 2 > 0)
  (hq : q = fun x => x > m)
  (h_suff_not_nec : ∀ x, ¬(q x) → ¬(p x)) 
  (h_not_nec : ∃ x, ¬(q x) ∧ p x) : 
  m ≥ 1 := by sorry

end range_of_m_l3830_383099


namespace det_A_l3830_383006

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 0, -2; 8, 5, -4; 3, 3, 6]

theorem det_A : Matrix.det A = 108 := by sorry

end det_A_l3830_383006


namespace division_value_problem_l3830_383035

theorem division_value_problem (x : ℝ) : (4 / x) * 12 = 8 → x = 6 := by
  sorry

end division_value_problem_l3830_383035


namespace third_wednesday_not_22nd_l3830_383008

def is_third_wednesday (day : ℕ) : Prop :=
  ∃ (first_wednesday : ℕ), 
    1 ≤ first_wednesday ∧ 
    first_wednesday ≤ 7 ∧ 
    day = first_wednesday + 14

theorem third_wednesday_not_22nd : 
  ¬ is_third_wednesday 22 :=
sorry

end third_wednesday_not_22nd_l3830_383008


namespace initial_fee_calculation_l3830_383015

/-- The initial fee for a taxi trip, given the rate per segment and total charge for a specific distance. -/
theorem initial_fee_calculation (rate_per_segment : ℝ) (total_charge : ℝ) (distance : ℝ) : 
  rate_per_segment = 0.35 →
  distance = 3.6 →
  total_charge = 5.65 →
  ∃ (initial_fee : ℝ), initial_fee = 2.50 ∧ 
    total_charge = initial_fee + (distance / (2/5)) * rate_per_segment :=
by sorry

end initial_fee_calculation_l3830_383015


namespace solution_set_implies_m_equals_two_l3830_383079

theorem solution_set_implies_m_equals_two (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, a * x^2 - 6 * x + a^2 < 0 ↔ 1 < x ∧ x < m) →
  m = 2 :=
by sorry

end solution_set_implies_m_equals_two_l3830_383079


namespace flag_design_count_l3830_383081

/-- The number of colors available for the flag -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 4

/-- A function that calculates the number of possible flag designs -/
def flag_designs (n : ℕ) (k : ℕ) : ℕ :=
  if n = 1 then k
  else k * (k - 1)^(n - 1)

/-- Theorem stating that the number of possible flag designs is 24 -/
theorem flag_design_count :
  flag_designs num_stripes num_colors = 24 := by
  sorry

end flag_design_count_l3830_383081


namespace problem_solution_l3830_383070

def f (a x : ℝ) : ℝ := a * x^2 + x - a

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≥ 1 ↔ x ≤ -2 ∨ x ≥ 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x > -2*x^2 - 3*x + 1 - 2*a) ↔ a > 2) ∧
  (∀ a : ℝ, a < 0 →
    ((-1/2 < a ∧ a < 0 ∧ ∀ x : ℝ, f a x > 1 ↔ 1 < x ∧ x < -(a+1)/a) ∨
     (a = -1/2 ∧ ∀ x : ℝ, ¬(f a x > 1)) ∨
     (a < -1/2 ∧ ∀ x : ℝ, f a x > 1 ↔ -(a+1)/a < x ∧ x < 1))) :=
by sorry

end problem_solution_l3830_383070


namespace inequality_and_max_value_l3830_383096

theorem inequality_and_max_value :
  (∀ a b c d : ℝ, (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = 5 → ∀ x : ℝ, 2*a + b ≤ x → x ≤ 5) :=
by sorry


end inequality_and_max_value_l3830_383096


namespace second_village_sales_l3830_383061

/-- Given the number of cookie packs sold in the first village and the total number of packs sold,
    calculate the number of packs sold in the second village. -/
def cookiesSoldInSecondVillage (firstVillage : ℕ) (total : ℕ) : ℕ :=
  total - firstVillage

/-- Theorem stating that the number of cookie packs sold in the second village
    is equal to the total number of packs sold minus the number sold in the first village. -/
theorem second_village_sales (firstVillage : ℕ) (total : ℕ) 
    (h : firstVillage ≤ total) :
  cookiesSoldInSecondVillage firstVillage total = total - firstVillage := by
  sorry

#eval cookiesSoldInSecondVillage 23 51  -- Expected output: 28

end second_village_sales_l3830_383061


namespace dani_pants_count_l3830_383002

/-- Calculate the final number of pants after receiving a certain number of pairs each year for a given period. -/
def final_pants_count (initial_pants : ℕ) (pairs_per_year : ℕ) (pants_per_pair : ℕ) (years : ℕ) : ℕ :=
  initial_pants + pairs_per_year * pants_per_pair * years

/-- Theorem stating that Dani will have 90 pants after 5 years -/
theorem dani_pants_count : final_pants_count 50 4 2 5 = 90 := by
  sorry

end dani_pants_count_l3830_383002


namespace product_of_x_and_y_l3830_383063

theorem product_of_x_and_y (x y : ℝ) : 
  (-3 * x + 4 * y = 28) → (3 * x - 2 * y = 8) → x * y = 264 :=
by
  sorry


end product_of_x_and_y_l3830_383063


namespace tan_two_implies_fraction_four_fifths_l3830_383010

theorem tan_two_implies_fraction_four_fifths (θ : Real) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4/5 := by
  sorry

end tan_two_implies_fraction_four_fifths_l3830_383010


namespace correct_transformation_l3830_383032

def original_expression : List Int := [-17, 3, -5, -8]

def transformed_expression : List Int := [-17, 3, 5, -8]

theorem correct_transformation :
  (original_expression.map (fun x => if x < 0 then -x else x)).foldl (· - ·) 0 =
  transformed_expression.foldl (· + ·) 0 :=
sorry

end correct_transformation_l3830_383032


namespace chef_apple_pies_l3830_383037

theorem chef_apple_pies (total pies : ℕ) (pecan pumpkin apple : ℕ) : 
  total = 13 → pecan = 4 → pumpkin = 7 → total = apple + pecan + pumpkin → apple = 2 := by
  sorry

end chef_apple_pies_l3830_383037


namespace book_selection_theorem_l3830_383080

/-- Given the number of books in each language, calculates the number of ways to select two books. -/
def book_selection (japanese : ℕ) (english : ℕ) (chinese : ℕ) :
  (ℕ × ℕ × ℕ) :=
  let different_languages := japanese * english + japanese * chinese + english * chinese
  let same_language := japanese * (japanese - 1) / 2 + english * (english - 1) / 2 + chinese * (chinese - 1) / 2
  let total := (japanese + english + chinese) * (japanese + english + chinese - 1) / 2
  (different_languages, same_language, total)

/-- Theorem stating the correct number of ways to select books given the specified quantities. -/
theorem book_selection_theorem :
  book_selection 5 7 10 = (155, 76, 231) := by
  sorry

end book_selection_theorem_l3830_383080


namespace average_of_abc_is_three_l3830_383051

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 1501 * C - 3003 * A = 6006)
  (eq2 : 1501 * B + 4504 * A = 7507)
  (eq3 : A + B = 1) :
  (A + B + C) / 3 = 3 := by
sorry

end average_of_abc_is_three_l3830_383051


namespace unique_triangle_solution_l3830_383024

theorem unique_triangle_solution (a b : ℝ) (A : ℝ) (ha : a = 30) (hb : b = 25) (hA : A = 150 * π / 180) :
  ∃! (c : ℝ) (B C : ℝ), 
    0 < c ∧ 0 < B ∧ 0 < C ∧
    a / Real.sin A = b / Real.sin B ∧
    b / Real.sin B = c / Real.sin C ∧
    A + B + C = π := by
  sorry

end unique_triangle_solution_l3830_383024


namespace greatest_divisor_four_consecutive_integers_divisibility_of_four_consecutive_integers_optimality_of_twelve_l3830_383093

/-- The greatest whole number that must be a divisor of the product of any four consecutive positive integers is 12. -/
theorem greatest_divisor_four_consecutive_integers : ℕ :=
  let f : ℕ → ℕ := λ n => n * (n + 1) * (n + 2) * (n + 3)
  12

theorem divisibility_of_four_consecutive_integers (n : ℕ) :
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

theorem optimality_of_twelve (m : ℕ) :
  (∀ n : ℕ, m ∣ (n * (n + 1) * (n + 2) * (n + 3))) → m ≤ 12 :=
sorry

end greatest_divisor_four_consecutive_integers_divisibility_of_four_consecutive_integers_optimality_of_twelve_l3830_383093


namespace fraction_proof_l3830_383007

theorem fraction_proof (t k : ℚ) (f : ℚ) 
  (h1 : t = f * (k - 32))
  (h2 : t = 105)
  (h3 : k = 221) :
  f = 5 / 9 := by
sorry

end fraction_proof_l3830_383007


namespace three_number_ratio_sum_l3830_383012

theorem three_number_ratio_sum (a b c : ℝ) : 
  (a : ℝ) > 0 ∧ b = 2 * a ∧ c = 4 * a ∧ a^2 + b^2 + c^2 = 1701 →
  a + b + c = 63 := by
sorry

end three_number_ratio_sum_l3830_383012


namespace min_area_triangle_AOB_l3830_383071

/-- Given a line l: mx + ny - 1 = 0 intersecting the x-axis at A and y-axis at B,
    and forming a chord of length 2 with the circle x² + y² = 4,
    the minimum area of triangle AOB is 3. -/
theorem min_area_triangle_AOB (m n : ℝ) :
  ∃ (A B : ℝ × ℝ),
    (m * A.1 + n * A.2 - 1 = 0) ∧
    (m * B.1 + n * B.2 - 1 = 0) ∧
    (A.2 = 0) ∧
    (B.1 = 0) ∧
    (∃ (C D : ℝ × ℝ),
      (m * C.1 + n * C.2 - 1 = 0) ∧
      (m * D.1 + n * D.2 - 1 = 0) ∧
      (C.1^2 + C.2^2 = 4) ∧
      (D.1^2 + D.2^2 = 4) ∧
      ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 4)) →
  ∃ (area_min : ℝ),
    (∀ (A' B' : ℝ × ℝ),
      (m * A'.1 + n * A'.2 - 1 = 0) →
      (A'.2 = 0) →
      (m * B'.1 + n * B'.2 - 1 = 0) →
      (B'.1 = 0) →
      area_min ≤ (1/2) * A'.1 * B'.2) ∧
    area_min = 3 :=
by sorry

end min_area_triangle_AOB_l3830_383071


namespace isosceles_triangle_exists_l3830_383084

/-- Isosceles triangle type -/
structure IsoscelesTriangle where
  /-- Base length of the isosceles triangle -/
  base : ℝ
  /-- Length of the equal sides of the isosceles triangle -/
  side : ℝ
  /-- Height of the isosceles triangle -/
  height : ℝ
  /-- Condition: base and side are positive -/
  base_pos : 0 < base
  side_pos : 0 < side
  /-- Condition: height is positive -/
  height_pos : 0 < height
  /-- Condition: triangle inequality -/
  triangle_ineq : base < 2 * side

/-- Theorem: Given a perimeter and a height, an isosceles triangle exists -/
theorem isosceles_triangle_exists (perimeter : ℝ) (height : ℝ) 
  (perimeter_pos : 0 < perimeter) (height_pos : 0 < height) : 
  ∃ (t : IsoscelesTriangle), t.base + 2 * t.side = perimeter ∧ t.height = height := by
  sorry

end isosceles_triangle_exists_l3830_383084


namespace smallest_number_with_conditions_l3830_383052

/-- A function that returns the digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with 37 -/
def ends_with_37 (n : ℕ) : Prop := sorry

theorem smallest_number_with_conditions : 
  ∀ n : ℕ, 
    n ≥ 99937 → 
    (ends_with_37 n ∧ digit_sum n = 37 ∧ n % 37 = 0) → 
    n = 99937 := by sorry

end smallest_number_with_conditions_l3830_383052


namespace negation_of_false_is_true_l3830_383034

theorem negation_of_false_is_true (p q : Prop) 
  (hp : p) (hq : ¬q) : ¬q := by sorry

end negation_of_false_is_true_l3830_383034


namespace total_apples_correct_l3830_383094

/-- The number of apples Bill picked from the orchard -/
def total_apples : ℕ := 56

/-- The number of children Bill has -/
def num_children : ℕ := 2

/-- The number of apples each child takes for teachers -/
def apples_per_child : ℕ := 3

/-- The number of teachers each child gives apples to -/
def num_teachers : ℕ := 2

/-- The number of pies Jill bakes -/
def num_pies : ℕ := 2

/-- The number of apples used per pie -/
def apples_per_pie : ℕ := 10

/-- The number of apples Bill has left -/
def apples_left : ℕ := 24

/-- Theorem stating that the total number of apples Bill picked is correct -/
theorem total_apples_correct :
  total_apples = 
    num_children * apples_per_child * num_teachers +
    num_pies * apples_per_pie +
    apples_left :=
by sorry

end total_apples_correct_l3830_383094


namespace quadratic_polynomial_property_l3830_383085

/-- A quadratic polynomial with a common root property -/
structure QuadraticPolynomial where
  P : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c
  common_root : ∃ t : ℝ, P t = 0 ∧ P (P (P t)) = 0

/-- 
For any quadratic polynomial P(x) where P(x) and P(P(P(x))) have a common root, 
P(0)P(1) = 0
-/
theorem quadratic_polynomial_property (p : QuadraticPolynomial) : 
  p.P 0 * p.P 1 = 0 := by
  sorry

end quadratic_polynomial_property_l3830_383085


namespace ginger_mat_straw_ratio_l3830_383011

/-- Given the conditions for Ginger's mat weaving, prove the ratio of green to orange straws per mat -/
theorem ginger_mat_straw_ratio :
  let red_per_mat : ℕ := 20
  let orange_per_mat : ℕ := 30
  let total_mats : ℕ := 10
  let total_straws : ℕ := 650
  let green_per_mat : ℕ := (total_straws - red_per_mat * total_mats - orange_per_mat * total_mats) / total_mats
  green_per_mat * 2 = orange_per_mat := by
  sorry

#check ginger_mat_straw_ratio

end ginger_mat_straw_ratio_l3830_383011


namespace brenda_sally_meeting_distance_l3830_383020

theorem brenda_sally_meeting_distance 
  (track_length : ℝ) 
  (sally_extra_distance : ℝ) 
  (h1 : track_length = 300)
  (h2 : sally_extra_distance = 100) :
  let first_meeting_distance := (track_length / 2 + sally_extra_distance) / 2
  first_meeting_distance = 150 := by
  sorry

end brenda_sally_meeting_distance_l3830_383020


namespace change_received_l3830_383031

/-- The change received when buying a football and baseball with given costs and payment amount -/
theorem change_received (football_cost baseball_cost payment : ℚ) : 
  football_cost = 9.14 →
  baseball_cost = 6.81 →
  payment = 20 →
  payment - (football_cost + baseball_cost) = 4.05 := by
  sorry

end change_received_l3830_383031


namespace complex_exponential_sum_l3830_383098

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1 / 3 : ℂ) + (5 / 8 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1 / 3 : ℂ) - (5 / 8 : ℂ) * Complex.I :=
by sorry

end complex_exponential_sum_l3830_383098


namespace sean_total_apples_l3830_383090

def initial_apples : ℕ := 9
def apples_per_day : ℕ := 8
def days : ℕ := 5

theorem sean_total_apples :
  initial_apples + apples_per_day * days = 49 := by
  sorry

end sean_total_apples_l3830_383090


namespace max_xy_min_inverse_sum_l3830_383019

-- Define the conditions
variable (x y : ℝ)
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : x + 4*y = 4)

-- Theorem for the maximum value of xy
theorem max_xy : ∀ x y : ℝ, x > 0 → y > 0 → x + 4*y = 4 → xy ≤ 1 := by
  sorry

-- Theorem for the minimum value of 1/x + 2/y
theorem min_inverse_sum : ∀ x y : ℝ, x > 0 → y > 0 → x + 4*y = 4 → 1/x + 2/y ≥ (9 + 4*Real.sqrt 2) / 4 := by
  sorry

end max_xy_min_inverse_sum_l3830_383019


namespace inequality_proof_l3830_383074

open BigOperators

theorem inequality_proof (n : ℕ) (δ : ℝ) (a b : ℕ → ℝ) 
  (h_pos_a : ∀ i ∈ Finset.range (n + 1), a i > 0)
  (h_pos_b : ∀ i ∈ Finset.range (n + 1), b i > 0)
  (h_delta : ∀ i ∈ Finset.range n, b (i + 1) - b i ≥ δ)
  (h_delta_pos : δ > 0)
  (h_sum_a : ∑ i in Finset.range n, a i = 1) :
  ∑ i in Finset.range n, (i + 1 : ℝ) * (∏ j in Finset.range (i + 1), (a j * b j)) ^ (1 / (i + 1 : ℝ)) / (b (i + 1) * b i) < 1 / δ :=
sorry

end inequality_proof_l3830_383074


namespace project_update_lcm_l3830_383059

theorem project_update_lcm : Nat.lcm 5 (Nat.lcm 9 (Nat.lcm 10 13)) = 1170 := by
  sorry

end project_update_lcm_l3830_383059


namespace opposite_of_negative_one_third_l3830_383075

theorem opposite_of_negative_one_third :
  -((-1 : ℚ) / 3) = 1 / 3 := by
  sorry

end opposite_of_negative_one_third_l3830_383075


namespace average_time_is_five_l3830_383021

/-- Colin's running times for each mile -/
def mile_times : List ℕ := [6, 5, 5, 4]

/-- Total number of miles run -/
def total_miles : ℕ := mile_times.length

/-- Calculates the average time per mile -/
def average_time_per_mile : ℚ :=
  (mile_times.sum : ℚ) / total_miles

/-- Theorem: The average time per mile is 5 minutes -/
theorem average_time_is_five : average_time_per_mile = 5 := by
  sorry

end average_time_is_five_l3830_383021


namespace toms_bowling_score_l3830_383047

theorem toms_bowling_score (tom jerry : ℕ) : 
  tom = jerry + 30 → 
  (tom + jerry) / 2 = 90 → 
  tom = 105 := by
sorry

end toms_bowling_score_l3830_383047


namespace train_speed_difference_l3830_383088

/-- Given two trains traveling towards each other, this theorem proves that
    the difference in their speeds is 30 km/hr under specific conditions. -/
theorem train_speed_difference 
  (distance : ℝ) 
  (meeting_time : ℝ) 
  (express_speed : ℝ) 
  (h1 : distance = 390) 
  (h2 : meeting_time = 3) 
  (h3 : express_speed = 80) : 
  express_speed - (distance / meeting_time - express_speed) = 30 := by
  sorry

end train_speed_difference_l3830_383088


namespace partnership_investment_l3830_383095

theorem partnership_investment (a c total_profit c_profit : ℚ) (ha : a = 45000) (hc : c = 72000) (htotal : total_profit = 60000) (hc_profit : c_profit = 24000) :
  ∃ b : ℚ, 
    (c_profit / total_profit = c / (a + b + c)) ∧
    b = 63000 := by
  sorry

end partnership_investment_l3830_383095


namespace bus_network_property_l3830_383053

-- Define the type for bus stops
variable {V : Type}

-- Define the "can be reached from" relation
def can_reach (G : V → V → Prop) (x y : V) : Prop := G x y

-- Define the "comes after" relation
def comes_after (G : V → V → Prop) (x y : V) : Prop :=
  ∀ z, can_reach G z x → can_reach G z y ∧ ∀ w, can_reach G y w → can_reach G x w

-- State the theorem
theorem bus_network_property (G : V → V → Prop) 
  (h : ∀ x y : V, x ≠ y → (can_reach G x y ↔ comes_after G x y)) :
  ∀ a b : V, a ≠ b → (can_reach G a b ∨ can_reach G b a) ∧ ¬(can_reach G a b ∧ can_reach G b a) :=
sorry

end bus_network_property_l3830_383053


namespace factorial_division_l3830_383067

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end factorial_division_l3830_383067


namespace committee_count_12_5_l3830_383057

/-- The number of ways to choose a committee of size k from a group of n people -/
def committeeCount (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the entire group -/
def groupSize : ℕ := 12

/-- The size of the committee to be chosen -/
def committeeSize : ℕ := 5

/-- Theorem stating that the number of ways to choose a 5-person committee from 12 people is 792 -/
theorem committee_count_12_5 : 
  committeeCount groupSize committeeSize = 792 := by sorry

end committee_count_12_5_l3830_383057


namespace complex_number_equality_l3830_383048

theorem complex_number_equality (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z + 3) ∧ 
  Complex.abs (z - 1) = Complex.abs (z - Complex.I) →
  z = -1 - Complex.I := by
  sorry

end complex_number_equality_l3830_383048


namespace move_left_result_l3830_383072

/-- Moving a point 2 units to the left in a Cartesian coordinate system. -/
def moveLeft (x y : ℝ) : ℝ × ℝ := (x - 2, y)

/-- The theorem stating that moving (-2, 3) 2 units to the left results in (-4, 3). -/
theorem move_left_result : moveLeft (-2) 3 = (-4, 3) := by
  sorry

end move_left_result_l3830_383072


namespace leading_coefficient_is_negative_six_l3830_383038

def polynomial (x : ℝ) : ℝ :=
  -5 * (x^5 - 2*x^3 + x) + 8 * (x^5 + x^3 - 3) - 3 * (3*x^5 + x^3 + 2)

theorem leading_coefficient_is_negative_six :
  ∃ (p : ℝ → ℝ), ∀ x, ∃ (r : ℝ), polynomial x = -6 * x^5 + r ∧ (∀ y, |y| ≥ 1 → |r| ≤ |y|^5 * |-6 * x^5|) :=
sorry

end leading_coefficient_is_negative_six_l3830_383038


namespace tank_fill_time_l3830_383091

/-- Proves that the time required to fill 3/4 of a 4000-gallon tank at a rate of 10 gallons per hour is 300 hours. -/
theorem tank_fill_time (tank_capacity : ℝ) (fill_rate : ℝ) (fill_fraction : ℝ) (fill_time : ℝ) :
  tank_capacity = 4000 →
  fill_rate = 10 →
  fill_fraction = 3/4 →
  fill_time = (fill_fraction * tank_capacity) / fill_rate →
  fill_time = 300 :=
by
  sorry

end tank_fill_time_l3830_383091


namespace angle_between_vectors_l3830_383017

def vector1 : Fin 2 → ℝ := ![2, 5]
def vector2 : Fin 2 → ℝ := ![-3, 7]

theorem angle_between_vectors (v1 v2 : Fin 2 → ℝ) :
  v1 = vector1 → v2 = vector2 →
  Real.arccos ((v1 0 * v2 0 + v1 1 * v2 1) /
    (Real.sqrt (v1 0 ^ 2 + v1 1 ^ 2) * Real.sqrt (v2 0 ^ 2 + v2 1 ^ 2))) =
  45 * π / 180 := by
  sorry

end angle_between_vectors_l3830_383017


namespace f_negative_2017_l3830_383003

noncomputable def f (x : ℝ) : ℝ :=
  ((x + 1)^2 + Real.log (Real.sqrt (1 + 9*x^2) - 3*x) * Real.cos x) / (x^2 + 1)

theorem f_negative_2017 (h : f 2017 = 2016) : f (-2017) = -2014 := by
  sorry

end f_negative_2017_l3830_383003


namespace thread_length_calculation_l3830_383062

theorem thread_length_calculation (original_length : ℝ) (additional_fraction : ℝ) : 
  original_length = 12 →
  additional_fraction = 3/4 →
  original_length + (additional_fraction * original_length) = 21 := by
  sorry

end thread_length_calculation_l3830_383062


namespace quadratic_equation_from_means_l3830_383027

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 15 → 
  ∀ x, x^2 - 16*x + 225 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end quadratic_equation_from_means_l3830_383027


namespace initial_car_cost_l3830_383041

/-- The initial cost of John's car, given his Uber profit and the car's trade-in value -/
theorem initial_car_cost (profit : ℕ) (trade_in_value : ℕ) 
  (h1 : profit = 18000)
  (h2 : trade_in_value = 6000) :
  profit + trade_in_value = 24000 := by
  sorry

end initial_car_cost_l3830_383041


namespace right_triangle_area_l3830_383092

theorem right_triangle_area (α : Real) (hypotenuse : Real) :
  α = 30 * π / 180 →
  hypotenuse = 20 →
  ∃ (area : Real), area = 50 * Real.sqrt 3 ∧
    area = (1 / 2) * (hypotenuse / 2) * (hypotenuse / 2 * Real.sqrt 3) :=
by sorry

end right_triangle_area_l3830_383092


namespace triangle_angle_calculation_l3830_383039

/-- 
Given a triangle XYZ:
- ext_angle_x is the exterior angle at vertex X
- angle_y is the angle at vertex Y
- angle_z is the angle at vertex Z

This theorem states that if the exterior angle at X is 150° and the angle at Y is 140°, 
then the angle at Z must be 110°.
-/
theorem triangle_angle_calculation 
  (ext_angle_x angle_y angle_z : ℝ) 
  (h1 : ext_angle_x = 150)
  (h2 : angle_y = 140) :
  angle_z = 110 := by
  sorry

end triangle_angle_calculation_l3830_383039


namespace sector_area_l3830_383040

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 16 → central_angle = 2 → area = 16 := by
  sorry

end sector_area_l3830_383040
