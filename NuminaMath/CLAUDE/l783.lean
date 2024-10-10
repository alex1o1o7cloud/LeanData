import Mathlib

namespace soccer_ball_weight_l783_78357

theorem soccer_ball_weight :
  ∀ (soccer_ball_weight bicycle_weight : ℝ),
    8 * soccer_ball_weight = 5 * bicycle_weight →
    4 * bicycle_weight = 120 →
    soccer_ball_weight = 18.75 := by
  sorry

end soccer_ball_weight_l783_78357


namespace equation_solution_l783_78391

theorem equation_solution : ∃! x : ℝ, 9 / (5 + x / 0.75) = 1 ∧ x = 3 := by
  sorry

end equation_solution_l783_78391


namespace gcd_of_squares_sum_l783_78348

theorem gcd_of_squares_sum : Nat.gcd (130^2 + 250^2 + 360^2) (129^2 + 249^2 + 361^2) = 1 := by
  sorry

end gcd_of_squares_sum_l783_78348


namespace quadratic_equation_solutions_l783_78366

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  {x : ℝ | f x = 0} = {0, 2} := by sorry

end quadratic_equation_solutions_l783_78366


namespace absolute_value_expression_l783_78383

theorem absolute_value_expression : |-2| * (|-25| - |5|) = -40 := by
  sorry

end absolute_value_expression_l783_78383


namespace power_of_four_l783_78302

theorem power_of_four (k : ℕ) (h : 4^k = 5) : 4^(2*k + 2) = 400 := by
  sorry

end power_of_four_l783_78302


namespace collinear_points_iff_k_eq_neg_one_l783_78384

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- The theorem stating the condition for collinearity of the given points -/
theorem collinear_points_iff_k_eq_neg_one (k : ℝ) :
  collinear ⟨3, 1⟩ ⟨6, 4⟩ ⟨10, k + 9⟩ ↔ k = -1 := by
  sorry

end collinear_points_iff_k_eq_neg_one_l783_78384


namespace add_base6_example_l783_78375

/-- Represents a number in base 6 --/
def Base6 : Type := Fin 6 → ℕ

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : Base6 := sorry

/-- Converts a base 6 number to its natural number representation --/
def fromBase6 (b : Base6) : ℕ := sorry

/-- Adds two base 6 numbers --/
def addBase6 (a b : Base6) : Base6 := sorry

/-- The number 5 in base 6 --/
def five_base6 : Base6 := toBase6 5

/-- The number 23 in base 6 --/
def twentythree_base6 : Base6 := toBase6 23

/-- The number 32 in base 6 --/
def thirtytwo_base6 : Base6 := toBase6 32

theorem add_base6_example : addBase6 five_base6 twentythree_base6 = thirtytwo_base6 := by
  sorry

end add_base6_example_l783_78375


namespace no_valid_tetrahedron_labeling_l783_78388

/-- Represents a labeling of a tetrahedron's vertices -/
def TetrahedronLabeling := Fin 4 → Fin 4

/-- Checks if a labeling is valid (uses each number exactly once) -/
def isValidLabeling (l : TetrahedronLabeling) : Prop :=
  ∀ i : Fin 4, ∃! j : Fin 4, l j = i

/-- Represents a face of the tetrahedron as a set of three vertex indices -/
def TetrahedronFace := Fin 3 → Fin 4

/-- The four faces of a tetrahedron -/
def tetrahedronFaces : Fin 4 → TetrahedronFace := sorry

/-- The sum of labels on a face -/
def faceSum (l : TetrahedronLabeling) (f : TetrahedronFace) : Nat :=
  (f 0).val + 1 + (f 1).val + 1 + (f 2).val + 1

/-- Theorem: No valid labeling exists such that all face sums are equal -/
theorem no_valid_tetrahedron_labeling :
  ¬∃ (l : TetrahedronLabeling),
    isValidLabeling l ∧
    ∃ (s : Nat), ∀ (f : Fin 4), faceSum l (tetrahedronFaces f) = s :=
  sorry

end no_valid_tetrahedron_labeling_l783_78388


namespace tough_week_sales_800_l783_78354

/-- The amount Haji's mother sells on a good week -/
def good_week_sales : ℝ := sorry

/-- The amount Haji's mother sells on a tough week -/
def tough_week_sales : ℝ := sorry

/-- The total amount Haji's mother makes in 5 good weeks and 3 tough weeks -/
def total_sales : ℝ := 10400

/-- Tough week sales are half of good week sales -/
axiom tough_week_half_good : tough_week_sales = good_week_sales / 2

/-- Total sales equation -/
axiom total_sales_equation : 5 * good_week_sales + 3 * tough_week_sales = total_sales

theorem tough_week_sales_800 : tough_week_sales = 800 := by
  sorry

end tough_week_sales_800_l783_78354


namespace sqrt_less_than_3x_plus_1_l783_78395

theorem sqrt_less_than_3x_plus_1 (x : ℝ) (hx : x > 0) : Real.sqrt x < 3 * x + 1 := by
  sorry

end sqrt_less_than_3x_plus_1_l783_78395


namespace tan_sum_equation_l783_78369

theorem tan_sum_equation : ∀ (x y : Real),
  x + y = 60 * π / 180 →
  Real.tan (60 * π / 180) = Real.sqrt 3 →
  Real.tan x + Real.tan y + Real.sqrt 3 * Real.tan x * Real.tan y = Real.sqrt 3 := by
  sorry

end tan_sum_equation_l783_78369


namespace perfect_square_trinomial_l783_78371

theorem perfect_square_trinomial (x y : ℝ) : x^2 + 4*y^2 - 4*x*y = (x - 2*y)^2 := by
  sorry

end perfect_square_trinomial_l783_78371


namespace third_side_of_triangle_l783_78308

theorem third_side_of_triangle (a b c : ℝ) : 
  a = 3 → b = 5 → c = 4 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
  (c < a + b ∧ a < b + c ∧ b < c + a) := by
  sorry

end third_side_of_triangle_l783_78308


namespace sum_of_factors_36_l783_78346

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_36 : sum_of_factors 36 = 91 := by
  sorry

end sum_of_factors_36_l783_78346


namespace train_length_l783_78386

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → time_s = 2.7497800175985923 → 
  speed_kmh * (5/18) * time_s = 110.9912007039437 := by
  sorry

#check train_length

end train_length_l783_78386


namespace difference_m_n_l783_78336

theorem difference_m_n (m n : ℕ+) (h : 10 * 2^(m : ℕ) = 2^(n : ℕ) + 2^((n : ℕ) + 2)) :
  (n : ℕ) - (m : ℕ) = 1 := by
  sorry

end difference_m_n_l783_78336


namespace octagon_diagonals_l783_78329

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l783_78329


namespace min_value_sum_squares_l783_78318

theorem min_value_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 6) :
  ∃ (min : ℝ), min = 18/7 ∧ x^2 + y^2 + z^2 ≥ min ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ + 2*y₀ + 3*z₀ = 6 ∧ x₀^2 + y₀^2 + z₀^2 = min :=
by sorry

end min_value_sum_squares_l783_78318


namespace sqrt_180_simplification_l783_78311

theorem sqrt_180_simplification : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end sqrt_180_simplification_l783_78311


namespace ball_bounce_height_l783_78305

/-- Theorem: For a ball that rises with each bounce exactly one-half as high as it had fallen,
    and bounces 4 times, if the total distance traveled is 44.5 meters,
    then the initial height from which the ball was dropped is 9.9 meters. -/
theorem ball_bounce_height (h : ℝ) : 
  (h + 2*h + h + (1/2)*h + (1/4)*h = 44.5) → h = 9.9 := by
  sorry

end ball_bounce_height_l783_78305


namespace cuboid_non_parallel_edges_l783_78342

/-- Represents a cuboid with integer side lengths -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of edges not parallel to a given edge in a cuboid -/
def nonParallelEdges (c : Cuboid) : ℕ := sorry

/-- Theorem stating that a cuboid with side lengths 8, 6, and 4 has 8 edges not parallel to any given edge -/
theorem cuboid_non_parallel_edges :
  let c : Cuboid := { length := 8, width := 6, height := 4 }
  nonParallelEdges c = 8 := by sorry

end cuboid_non_parallel_edges_l783_78342


namespace floor_properties_l783_78379

theorem floor_properties (x : ℝ) : 
  (x - 1 < ⌊x⌋ ∧ ⌊x⌋ ≤ x) ∧ ⌊2*x⌋ - 2*⌊x⌋ ∈ ({0, 1} : Set ℤ) := by sorry

end floor_properties_l783_78379


namespace perpendicular_line_through_intersection_l783_78393

/-- Given a line l with equation 2x - y - 4 = 0, prove that the line with equation
    x + 2y - 2 = 0 is perpendicular to l and passes through the point where l
    intersects the x-axis. -/
theorem perpendicular_line_through_intersection (x y : ℝ) : 
  let l : ℝ → ℝ → Prop := λ x y ↦ 2 * x - y - 4 = 0
  let m : ℝ × ℝ := (2, 0)  -- Intersection point of l with x-axis
  let perp : ℝ → ℝ → Prop := λ x y ↦ x + 2 * y - 2 = 0
  (∀ x y, l x y → (x - m.1) * (x - m.1) + (y - m.2) * (y - m.2) ≠ 0 →
    (perp x y ↔ (x - m.1) * (2) + (y - m.2) * (-1) = 0)) ∧
  perp m.1 m.2 := by
sorry

end perpendicular_line_through_intersection_l783_78393


namespace sum_x_coordinates_above_line_l783_78362

def points : List (ℚ × ℚ) := [(2, 8), (5, 15), (10, 25), (15, 36), (19, 45), (22, 52), (25, 66)]

def isAboveLine (p : ℚ × ℚ) : Bool :=
  p.2 > 2 * p.1 + 5

def pointsAboveLine : List (ℚ × ℚ) :=
  points.filter isAboveLine

theorem sum_x_coordinates_above_line :
  (pointsAboveLine.map (·.1)).sum = 81 := by
  sorry

end sum_x_coordinates_above_line_l783_78362


namespace root_of_equation_l783_78396

theorem root_of_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x > 0, (Real.sqrt (a * b * x * (a + b + x)) + 
             Real.sqrt (b * c * x * (b + c + x)) + 
             Real.sqrt (c * a * x * (c + a + x)) = 
             Real.sqrt (a * b * c * (a + b + c))) ∧
           (x = (a * b * c) / (a * b + b * c + c * a + 2 * Real.sqrt (a * b * c * (a + b + c)))) :=
by sorry

end root_of_equation_l783_78396


namespace shortest_distance_C1_C2_l783_78331

/-- The curve C1 in Cartesian coordinates -/
def C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2

/-- The curve C2 as a line in Cartesian coordinates -/
def C2 (x y : ℝ) : Prop := x + y = 4

/-- The shortest distance between C1 and C2 -/
theorem shortest_distance_C1_C2 :
  ∃ (p q : ℝ × ℝ), C1 p.1 p.2 ∧ C2 q.1 q.2 ∧
    ∀ (p' q' : ℝ × ℝ), C1 p'.1 p'.2 → C2 q'.1 q'.2 →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 2 / 2 :=
sorry

end shortest_distance_C1_C2_l783_78331


namespace triangle_pqr_area_l783_78314

/-- Triangle PQR with given properties -/
structure Triangle where
  inradius : ℝ
  circumradius : ℝ
  angle_relation : ℝ → ℝ → ℝ → Prop

/-- The area of a triangle given its inradius and semiperimeter -/
def triangle_area (r : ℝ) (s : ℝ) : ℝ := r * s

/-- Theorem: Area of triangle PQR with given properties -/
theorem triangle_pqr_area (T : Triangle) 
  (h_inradius : T.inradius = 6)
  (h_circumradius : T.circumradius = 17)
  (h_angle : T.angle_relation = fun P Q R => 3 * Real.cos Q = Real.cos P + Real.cos R) :
  ∃ (s : ℝ), triangle_area T.inradius s = (102 * Real.sqrt 47) / 3 := by
  sorry

end triangle_pqr_area_l783_78314


namespace gadget_production_l783_78307

/-- Represents the time (in hours) required for one worker to produce one gizmo -/
def gizmo_time : ℚ := sorry

/-- Represents the time (in hours) required for one worker to produce one gadget -/
def gadget_time : ℚ := sorry

/-- The number of gadgets produced by 30 workers in 4 hours -/
def n : ℕ := sorry

theorem gadget_production :
  -- In 1 hour, 80 workers produce 200 gizmos and 160 gadgets
  80 * (200 * gizmo_time + 160 * gadget_time) = 1 →
  -- In 2 hours, 40 workers produce 160 gizmos and 240 gadgets
  40 * (160 * gizmo_time + 240 * gadget_time) = 2 →
  -- In 4 hours, 30 workers produce 120 gizmos and n gadgets
  30 * (120 * gizmo_time + n * gadget_time) = 4 →
  -- The number of gadgets produced by 30 workers in 4 hours is 135680
  n = 135680 := by
  sorry

end gadget_production_l783_78307


namespace hyperbola_asymptotes_l783_78373

/-- The asymptote equations of the hyperbola x^2 - y^2/4 = 1 are y = 2x and y = -2x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 - y^2/4 = 1) → (∃ (k : ℝ), k = 2 ∨ k = -2) ∧ (y = k*x) :=
by sorry

end hyperbola_asymptotes_l783_78373


namespace min_shift_sine_graph_l783_78313

theorem min_shift_sine_graph (φ : ℝ) : 
  (φ > 0 ∧ ∀ x, Real.sin (2*x + 2*φ + π/3) = Real.sin (2*x)) → φ ≥ 5*π/6 :=
by sorry

end min_shift_sine_graph_l783_78313


namespace computer_game_cost_l783_78321

/-- The cost of the computer game Mr. Grey purchased, given the following conditions:
  * He bought 3 polo shirts for $26 each
  * He bought 2 necklaces for $83 each
  * He received a $12 rebate
  * The total cost after the rebate was $322
-/
theorem computer_game_cost : ℕ := by
  let polo_shirt_cost : ℕ := 26
  let polo_shirt_count : ℕ := 3
  let necklace_cost : ℕ := 83
  let necklace_count : ℕ := 2
  let rebate : ℕ := 12
  let total_cost_after_rebate : ℕ := 322

  have h1 : polo_shirt_cost * polo_shirt_count + necklace_cost * necklace_count + 90 = total_cost_after_rebate + rebate := by sorry

  exact 90

end computer_game_cost_l783_78321


namespace right_triangle_hypotenuse_l783_78355

theorem right_triangle_hypotenuse (shorter_leg : ℝ) (longer_leg : ℝ) (area : ℝ) :
  shorter_leg > 0 →
  longer_leg = 3 * shorter_leg - 3 →
  area = (1 / 2) * shorter_leg * longer_leg →
  area = 84 →
  (shorter_leg ^ 2 + longer_leg ^ 2).sqrt = Real.sqrt 505 := by
  sorry

#check right_triangle_hypotenuse

end right_triangle_hypotenuse_l783_78355


namespace count_square_family_with_range_14_l783_78353

/-- A function family is characterized by its analytic expression and range -/
structure FunctionFamily where
  expression : ℝ → ℝ
  range : Set ℝ

/-- Count the number of functions in a family with different domains -/
def countFunctionsInFamily (f : FunctionFamily) : ℕ :=
  sorry

/-- The specific function family we're interested in -/
def squareFamilyWithRange14 : FunctionFamily :=
  { expression := fun x ↦ x^2,
    range := {1, 4} }

/-- Theorem stating that the number of functions in our specific family is 9 -/
theorem count_square_family_with_range_14 :
  countFunctionsInFamily squareFamilyWithRange14 = 9 := by
  sorry

end count_square_family_with_range_14_l783_78353


namespace smallest_n_for_20_colors_l783_78349

/-- Represents a ball with a color -/
structure Ball :=
  (color : Nat)

/-- Represents a circular arrangement of balls -/
def CircularArrangement := List Ball

/-- Checks if a sequence of balls has at least k different colors -/
def hasAtLeastKColors (sequence : List Ball) (k : Nat) : Prop :=
  (sequence.map Ball.color).toFinset.card ≥ k

theorem smallest_n_for_20_colors 
  (total_balls : Nat) 
  (num_colors : Nat) 
  (balls_per_color : Nat) 
  (h1 : total_balls = 1000) 
  (h2 : num_colors = 40) 
  (h3 : balls_per_color = 25) 
  (h4 : total_balls = num_colors * balls_per_color) :
  ∃ (n : Nat), 
    (∀ (arrangement : CircularArrangement), 
      arrangement.length = total_balls → 
      ∃ (subsequence : List Ball), 
        subsequence.length = n ∧ 
        hasAtLeastKColors subsequence 20) ∧
    (∀ (m : Nat), m < n → 
      ∃ (arrangement : CircularArrangement), 
        arrangement.length = total_balls ∧ 
        ∀ (subsequence : List Ball), 
          subsequence.length = m → 
          ¬(hasAtLeastKColors subsequence 20)) ∧
    n = 352 :=
sorry

end smallest_n_for_20_colors_l783_78349


namespace square_sum_of_xy_l783_78390

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + 2 * x + 2 * y = 152)
  (h2 : x ^ 2 * y + x * y ^ 2 = 1512) :
  x ^ 2 + y ^ 2 = 1136 ∨ x ^ 2 + y ^ 2 = 221 := by
sorry

end square_sum_of_xy_l783_78390


namespace zach_current_tickets_l783_78367

def ferris_wheel_cost : ℕ := 2
def roller_coaster_cost : ℕ := 7
def log_ride_cost : ℕ := 1
def additional_tickets_needed : ℕ := 9

def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + log_ride_cost

theorem zach_current_tickets : total_cost - additional_tickets_needed = 1 := by
  sorry

end zach_current_tickets_l783_78367


namespace hamburger_combinations_l783_78343

/-- The number of condiments available -/
def num_condiments : ℕ := 9

/-- The number of bun choices available -/
def num_bun_choices : ℕ := 2

/-- The number of meat patty choices available -/
def num_patty_choices : ℕ := 3

/-- The total number of different hamburger combinations -/
def total_hamburgers : ℕ := 2^num_condiments * num_bun_choices * num_patty_choices

theorem hamburger_combinations :
  total_hamburgers = 3072 :=
sorry

end hamburger_combinations_l783_78343


namespace simplify_and_evaluate_l783_78351

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 - 1 / a) / ((a^2 - 1) / a) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l783_78351


namespace max_individual_score_l783_78380

theorem max_individual_score (total_points : ℕ) (num_players : ℕ) (min_points : ℕ) 
  (h1 : total_points = 100)
  (h2 : num_players = 12)
  (h3 : min_points = 8)
  (h4 : ∀ i : ℕ, i < num_players → min_points ≤ (total_points / num_players)) :
  ∃ max_score : ℕ, max_score = 12 ∧ 
    ∀ player_score : ℕ, player_score ≤ max_score ∧
    (num_players - 1) * min_points + max_score = total_points :=
by sorry

end max_individual_score_l783_78380


namespace water_needed_for_solution_l783_78356

theorem water_needed_for_solution (total_volume : ℝ) (water_ratio : ℝ) (desired_volume : ℝ) :
  water_ratio = 1/3 →
  desired_volume = 0.48 →
  water_ratio * desired_volume = 0.16 :=
by sorry

end water_needed_for_solution_l783_78356


namespace sum_of_repeating_decimals_l783_78344

-- Define the repeating decimal 0.333...
def repeating_3 : ℚ := 1 / 3

-- Define the repeating decimal 0.0202...
def repeating_02 : ℚ := 2 / 99

-- Theorem statement
theorem sum_of_repeating_decimals : repeating_3 + repeating_02 = 35 / 99 := by
  sorry

end sum_of_repeating_decimals_l783_78344


namespace marble_arrangement_l783_78377

/-- Represents the color of a marble -/
inductive Color
| Blue
| Yellow

/-- Calculates the number of ways to arrange marbles -/
def arrange_marbles (blue : ℕ) (yellow : ℕ) : ℕ :=
  Nat.choose (yellow + blue - 1) (blue - 1)

/-- The main theorem -/
theorem marble_arrangement :
  let blue := 6
  let max_yellow := 17
  let arrangements := arrange_marbles blue max_yellow
  arrangements = 12376 ∧ arrangements % 1000 = 376 := by
  sorry


end marble_arrangement_l783_78377


namespace complex_fraction_simplification_l783_78350

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 3 * i) / (1 + 4 * i) = -10/17 - (11/17) * i :=
by
  sorry

end complex_fraction_simplification_l783_78350


namespace min_value_expression_min_value_achievable_l783_78303

theorem min_value_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  8 * a^4 + 12 * b^4 + 40 * c^4 + 18 * d^4 + 9 / (4 * a * b * c * d) ≥ 12 * Real.sqrt 2 :=
by sorry

theorem min_value_achievable :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    8 * a^4 + 12 * b^4 + 40 * c^4 + 18 * d^4 + 9 / (4 * a * b * c * d) = 12 * Real.sqrt 2 :=
by sorry

end min_value_expression_min_value_achievable_l783_78303


namespace remainder_123456789012_mod_180_l783_78337

theorem remainder_123456789012_mod_180 : 123456789012 % 180 = 12 := by
  sorry

end remainder_123456789012_mod_180_l783_78337


namespace oranges_per_box_l783_78309

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) 
  (h1 : total_oranges = 56) (h2 : num_boxes = 8) :
  total_oranges / num_boxes = 7 := by
  sorry

end oranges_per_box_l783_78309


namespace expand_polynomial_l783_78324

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 8 * x + 5) = 4 * x^3 + 4 * x^2 - 19 * x + 15 := by
  sorry

end expand_polynomial_l783_78324


namespace simplest_quadratic_radical_l783_78397

/-- A quadratic radical is considered "simple" if it cannot be simplified further. -/
def IsSimpleQuadraticRadical (x : ℝ) : Prop :=
  x ≥ 0 ∧ ∀ y z : ℝ, x = y * y * z → y = 1 ∨ z < 0

/-- The set of quadratic radicals to consider -/
def QuadraticRadicals : Set ℝ := {4, 7, 12, 0.5}

theorem simplest_quadratic_radical :
  ∃ (x : ℝ), x ∈ QuadraticRadicals ∧
    IsSimpleQuadraticRadical (Real.sqrt x) ∧
    ∀ y ∈ QuadraticRadicals, IsSimpleQuadraticRadical (Real.sqrt y) → y = x :=
by
  sorry

end simplest_quadratic_radical_l783_78397


namespace forgotten_lawns_l783_78381

/-- 
Given that:
- Roger earns $9 for each lawn he mows
- He had 14 lawns to mow
- He actually earned $54

Prove that the number of lawns Roger forgot to mow is equal to 14 minus the quotient of 54 and 9.
-/
theorem forgotten_lawns (earnings_per_lawn : ℕ) (total_lawns : ℕ) (actual_earnings : ℕ) :
  earnings_per_lawn = 9 →
  total_lawns = 14 →
  actual_earnings = 54 →
  total_lawns - (actual_earnings / earnings_per_lawn) = 8 :=
by sorry

end forgotten_lawns_l783_78381


namespace tree_height_proof_l783_78310

/-- Represents the height of a tree as a function of its breast diameter -/
def tree_height (x : ℝ) : ℝ := 25 * x + 15

theorem tree_height_proof :
  (tree_height 0.2 = 20) ∧
  (tree_height 0.28 = 22) ∧
  (tree_height 0.3 = 22.5) := by
  sorry

end tree_height_proof_l783_78310


namespace distance_to_y_axis_l783_78301

/-- The distance from a point to the y-axis is equal to the absolute value of its x-coordinate -/
theorem distance_to_y_axis (P : ℝ × ℝ) : 
  let (x, y) := P
  abs x = Real.sqrt ((x - 0)^2 + (y - y)^2) :=
by sorry

end distance_to_y_axis_l783_78301


namespace tom_typing_time_l783_78361

/-- Calculates the time required to type a given number of pages -/
def typing_time (words_per_minute : ℕ) (words_per_page : ℕ) (num_pages : ℕ) : ℕ :=
  (words_per_page * num_pages) / words_per_minute

theorem tom_typing_time :
  typing_time 90 450 10 = 50 := by
  sorry

end tom_typing_time_l783_78361


namespace sandwich_non_condiments_percentage_l783_78300

theorem sandwich_non_condiments_percentage 
  (total_weight : ℝ) 
  (condiments_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : condiments_weight = 50) : 
  (total_weight - condiments_weight) / total_weight * 100 = 75 := by
  sorry

end sandwich_non_condiments_percentage_l783_78300


namespace ping_pong_balls_l783_78332

theorem ping_pong_balls (y w : ℕ) : 
  y = 2 * (w - 10) →
  w - 10 = 5 * (y - 9) →
  y = 10 ∧ w = 15 := by
sorry

end ping_pong_balls_l783_78332


namespace polynomial_root_difference_sum_l783_78376

theorem polynomial_root_difference_sum (a b c d : ℝ) (x₁ x₂ : ℝ) : 
  a + b + c = 0 →
  a * x₁^3 + b * x₁^2 + c * x₁ + d = 0 →
  a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 →
  x₁ = 1 →
  a ≥ b →
  b ≥ c →
  a > 0 →
  c < 0 →
  ∃ (min_val max_val : ℝ),
    (∀ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 → |x₁^2 - x₂^2| ≥ min_val) ∧
    (∃ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 ∧ |x₁^2 - x₂^2| = min_val) ∧
    (∀ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 → |x₁^2 - x₂^2| ≤ max_val) ∧
    (∃ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 ∧ |x₁^2 - x₂^2| = max_val) ∧
    min_val + max_val = 3 :=
by sorry

end polynomial_root_difference_sum_l783_78376


namespace circumscribed_sphere_surface_area_l783_78333

/-- The surface area of the circumscribed sphere of a rectangular parallelepiped
    with face diagonal lengths 2, √3, and √5 is 6π. -/
theorem circumscribed_sphere_surface_area 
  (x y z : ℝ) 
  (h1 : x^2 + y^2 = 4) 
  (h2 : y^2 + z^2 = 3) 
  (h3 : z^2 + x^2 = 5) : 
  4 * Real.pi * ((x^2 + y^2 + z^2) / 4) = 6 * Real.pi :=
sorry

end circumscribed_sphere_surface_area_l783_78333


namespace geometric_sequence_second_term_l783_78317

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, n > 0 → a n = a 1 * q^(n-1) ∧ S n = (a 1 * (1 - q^n)) / (1 - q)

/-- The theorem stating that for a geometric sequence with q = 2 and S_4 = 60, a_2 = 8 -/
theorem geometric_sequence_second_term 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : geometric_sequence a 2 S) 
  (h_sum : S 4 = 60) : 
  a 2 = 8 := by
sorry

end geometric_sequence_second_term_l783_78317


namespace small_square_area_l783_78320

-- Define the tile and its components
def TileArea : ℝ := 49
def HypotenuseLength : ℝ := 5
def NumTriangles : ℕ := 8

-- Theorem statement
theorem small_square_area :
  ∀ (small_square_area : ℝ),
    small_square_area = TileArea - NumTriangles * (HypotenuseLength^2 / 2) →
    small_square_area = 1 :=
by sorry

end small_square_area_l783_78320


namespace perpendicular_vectors_m_value_l783_78330

/-- Given vectors a and b in ℝ², if a-b is perpendicular to ma+b, then m = 1/4 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (2, 1))
  (h2 : b = (1, -1))
  (h3 : (a.1 - b.1, a.2 - b.2) • (m * a.1 + b.1, m * a.2 + b.2) = 0) :
  m = 1/4 := by
  sorry

end perpendicular_vectors_m_value_l783_78330


namespace quadratic_equation_roots_l783_78312

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + a*x₁ + 3 = 0 ∧ 
   x₂^2 + a*x₂ + 3 = 0 ∧ 
   x₁^3 - 99/(2*x₂^2) = x₂^3 - 99/(2*x₁^2)) → 
  a = -6 := by
sorry

end quadratic_equation_roots_l783_78312


namespace johns_annual_profit_l783_78328

/-- John's apartment subletting profit calculation --/
theorem johns_annual_profit :
  ∀ (num_subletters : ℕ) 
    (subletter_payment : ℕ) 
    (rent_cost : ℕ) 
    (months_in_year : ℕ),
  num_subletters = 3 →
  subletter_payment = 400 →
  rent_cost = 900 →
  months_in_year = 12 →
  (num_subletters * subletter_payment - rent_cost) * months_in_year = 3600 :=
by
  sorry

end johns_annual_profit_l783_78328


namespace square_area_ratio_l783_78334

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 45) (h2 : side_D = 60) :
  (side_C^2) / (side_D^2) = 9 / 16 := by
  sorry

end square_area_ratio_l783_78334


namespace sum_of_angles_in_quadrilateral_figure_l783_78327

/-- A geometric figure with six angles that form a quadrilateral -/
structure QuadrilateralFigure where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  F : ℝ
  G : ℝ

/-- The sum of angles in a quadrilateral is 360° -/
theorem sum_of_angles_in_quadrilateral_figure (q : QuadrilateralFigure) :
  q.A + q.B + q.C + q.D + q.F + q.G = 360 := by
  sorry

end sum_of_angles_in_quadrilateral_figure_l783_78327


namespace sequence_property_l783_78387

theorem sequence_property (m : ℤ) (a : ℕ → ℤ) (r s : ℕ) :
  (∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n) →
  (|m| ≥ 2) →
  (a 1 ≠ 0 ∨ a 2 ≠ 0) →
  (r > s) →
  (s ≥ 2) →
  (a r = a 1) →
  (a s = a 1) →
  (r - s : ℤ) ≥ |m| :=
by sorry

end sequence_property_l783_78387


namespace bank_account_withdrawal_l783_78359

theorem bank_account_withdrawal (initial_balance deposit1 deposit2 final_balance_increase : ℕ) :
  initial_balance = 150 →
  deposit1 = 17 →
  deposit2 = 21 →
  final_balance_increase = 16 →
  ∃ withdrawal : ℕ, 
    initial_balance + deposit1 - withdrawal + deposit2 = initial_balance + final_balance_increase ∧
    withdrawal = 22 :=
by sorry

end bank_account_withdrawal_l783_78359


namespace min_draws_for_fifteen_l783_78335

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem min_draws_for_fifteen (counts : BallCounts) :
  counts.red = 30 ∧ counts.green = 25 ∧ counts.yellow = 23 ∧
  counts.blue = 14 ∧ counts.white = 13 ∧ counts.black = 10 →
  minDraws counts 15 = 80 := by
  sorry

end min_draws_for_fifteen_l783_78335


namespace compound_weight_proof_l783_78316

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of iodine atoms in the compound -/
def num_I_atoms : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 408

/-- Theorem stating that the molecular weight of the compound with 1 Al atom and 3 I atoms 
    is approximately equal to 408 g/mol -/
theorem compound_weight_proof : 
  ∃ ε > 0, |atomic_weight_Al + num_I_atoms * atomic_weight_I - molecular_weight| < ε :=
sorry

end compound_weight_proof_l783_78316


namespace cistern_emptying_time_l783_78352

/-- Given a cistern with specific properties, prove the time it takes to empty -/
theorem cistern_emptying_time 
  (capacity : ℝ)
  (leak_empty_time : ℝ)
  (tap_rate : ℝ)
  (h1 : capacity = 480)
  (h2 : leak_empty_time = 20)
  (h3 : tap_rate = 4)
  : (capacity / (capacity / leak_empty_time - tap_rate) = 24) :=
by
  sorry

end cistern_emptying_time_l783_78352


namespace min_value_expression_l783_78385

theorem min_value_expression (a b : ℝ) (h : a - b^2 = 4) :
  ∃ (m : ℝ), m = 5 ∧ ∀ (x y : ℝ), x - y^2 = 4 → x^2 - 3*y^2 + x - 15 ≥ m :=
by sorry

end min_value_expression_l783_78385


namespace binomial_9_5_l783_78394

theorem binomial_9_5 : Nat.choose 9 5 = 756 := by
  sorry

end binomial_9_5_l783_78394


namespace chess_tournament_games_l783_78315

theorem chess_tournament_games (n : ℕ) (h : n = 20) : 
  (n * (n - 1)) / 2 = 190 := by
  sorry

end chess_tournament_games_l783_78315


namespace perfect_squares_difference_four_digit_sqrt_difference_l783_78389

theorem perfect_squares_difference (m n p : ℕ) 
  (h1 : m > n) 
  (h2 : Real.sqrt m - Real.sqrt n = p) : 
  ∃ (a b : ℕ), m = a^2 ∧ n = b^2 := by
  sorry

theorem four_digit_sqrt_difference : 
  ∃! (abcd : ℕ), 
    1000 ≤ abcd ∧ abcd < 10000 ∧
    ∃ (a b c d : ℕ),
      abcd = 1000 * a + 100 * b + 10 * c + d ∧
      100 * a + 10 * c + d < abcd ∧
      Real.sqrt (abcd) - Real.sqrt (100 * a + 10 * c + d) = 11 * b := by
  sorry

end perfect_squares_difference_four_digit_sqrt_difference_l783_78389


namespace cereal_sugar_percentage_l783_78326

/-- The percentage of sugar in cereal A -/
def sugar_a : ℝ := 10

/-- The ratio of cereal A to cereal B -/
def ratio : ℝ := 1

/-- The percentage of sugar in the final mixture -/
def sugar_mixture : ℝ := 6

/-- The percentage of sugar in cereal B -/
def sugar_b : ℝ := 2

theorem cereal_sugar_percentage :
  (sugar_a * ratio + sugar_b * ratio) / (ratio + ratio) = sugar_mixture :=
by sorry

end cereal_sugar_percentage_l783_78326


namespace extremum_of_g_and_range_of_a_l783_78364

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * x^2
def g (x : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem extremum_of_g_and_range_of_a :
  (a > 0 → ∃ (x_min : ℝ), x_min = Real.log (2 * a) ∧ 
    (∀ y, g a y ≥ g a x_min) ∧ 
    g a x_min = 2 * a - 2 * a * Real.log (2 * a)) ∧
  ((∀ x ≥ 0, f a x ≥ x + (1 - x) * Real.exp x) → a ≤ 1) :=
sorry

end

end extremum_of_g_and_range_of_a_l783_78364


namespace larger_number_proof_l783_78358

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 60) 
  (h2 : Nat.lcm a b = 60 * 11 * 15) : max a b = 900 := by
  sorry

end larger_number_proof_l783_78358


namespace t_value_l783_78323

theorem t_value (x y t : ℝ) (h1 : 2^x = t) (h2 : 5^y = t) (h3 : 1/x + 1/y = 2) (h4 : t ≠ 1) : t = Real.sqrt 10 := by
  sorry

end t_value_l783_78323


namespace black_squares_on_33x33_board_l783_78341

/-- Represents a checkerboard with alternating colors and black corners -/
structure Checkerboard where
  size : Nat
  has_black_corners : Bool
  is_alternating : Bool

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard) : Nat :=
  sorry

/-- The theorem stating that a 33x33 checkerboard with alternating colors and black corners has 545 black squares -/
theorem black_squares_on_33x33_board :
  ∀ (board : Checkerboard),
    board.size = 33 →
    board.has_black_corners = true →
    board.is_alternating = true →
    count_black_squares board = 545 :=
  sorry

end black_squares_on_33x33_board_l783_78341


namespace modulus_of_complex_expression_l783_78360

theorem modulus_of_complex_expression : 
  Complex.abs ((1 - 2 * Complex.I)^2 / Complex.I) = 5 := by sorry

end modulus_of_complex_expression_l783_78360


namespace extremum_at_one_implies_a_equals_four_l783_78304

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_at_one_implies_a_equals_four (a b : ℝ) :
  f_derivative a b 1 = 0 → f a b 1 = 10 → a = 4 := by
  sorry

#check extremum_at_one_implies_a_equals_four

end extremum_at_one_implies_a_equals_four_l783_78304


namespace fruit_basket_count_l783_78399

/-- The number of ways to choose from n identical items -/
def chooseFromIdentical (n : ℕ) : ℕ := n + 1

/-- The number of fruit baskets with at least one fruit -/
def fruitBaskets (pears bananas : ℕ) : ℕ :=
  chooseFromIdentical pears * chooseFromIdentical bananas - 1

theorem fruit_basket_count :
  fruitBaskets 8 12 = 116 := by
  sorry

end fruit_basket_count_l783_78399


namespace performance_orders_count_l783_78347

/-- The number of programs available --/
def total_programs : ℕ := 8

/-- The number of programs to be selected --/
def selected_programs : ℕ := 4

/-- The number of special programs (A and B) --/
def special_programs : ℕ := 2

/-- Calculate the number of different performance orders --/
def calculate_orders : ℕ :=
  -- First category: only one of A or B is selected
  (special_programs.choose 1) * ((total_programs - special_programs).choose (selected_programs - 1)) * (selected_programs.factorial) +
  -- Second category: both A and B are selected
  ((total_programs - special_programs).choose (selected_programs - special_programs)) * (special_programs.factorial) * ((selected_programs - special_programs).factorial)

/-- The theorem to be proved --/
theorem performance_orders_count : calculate_orders = 1140 := by
  sorry

end performance_orders_count_l783_78347


namespace nonagon_diagonals_count_l783_78345

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex polygon with 9 sides -/
def nonagon : ℕ := 9

/-- Theorem stating that the number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals_count : nonagon_diagonals = (nonagon * (nonagon - 3)) / 2 := by
  sorry

end nonagon_diagonals_count_l783_78345


namespace two_integer_pairs_satisfy_equation_l783_78372

theorem two_integer_pairs_satisfy_equation :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 2 ∧ 
  (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 2) :=
by sorry

end two_integer_pairs_satisfy_equation_l783_78372


namespace max_a_value_l783_78306

/-- The function f as defined in the problem -/
def f (x k a : ℝ) : ℝ := x^2 - (k^2 - 5*a*k + 3)*x + 7

/-- The theorem stating the maximum value of a -/
theorem max_a_value :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (k : ℝ), k ∈ Set.Icc 0 2 → 
    ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc k (k+a) → x₂ ∈ Set.Icc (k+2*a) (k+4*a) → 
      f x₁ k a ≥ f x₂ k a) ∧
  (∀ (a' : ℝ), a' > a → 
    ∃ (k : ℝ), k ∈ Set.Icc 0 2 ∧ 
      ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc k (k+a') ∧ x₂ ∈ Set.Icc (k+2*a') (k+4*a') ∧ 
        f x₁ k a' < f x₂ k a') ∧
  a = (2 * Real.sqrt 6 - 4) / 5 := by
sorry

end max_a_value_l783_78306


namespace cos_36_degrees_l783_78370

theorem cos_36_degrees : Real.cos (36 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end cos_36_degrees_l783_78370


namespace tenth_term_of_sequence_l783_78325

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem tenth_term_of_sequence (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 3 = 10 →
  arithmetic_sequence a₁ d 6 = 16 →
  arithmetic_sequence a₁ d 10 = 24 := by
sorry

end tenth_term_of_sequence_l783_78325


namespace total_instruments_is_19_l783_78338

-- Define the number of instruments for Charlie
def charlie_flutes : ℕ := 1
def charlie_horns : ℕ := 2
def charlie_harps : ℕ := 1
def charlie_drums : ℕ := 1

-- Define the number of instruments for Carli
def carli_flutes : ℕ := 2 * charlie_flutes
def carli_horns : ℕ := charlie_horns / 2
def carli_harps : ℕ := 0
def carli_drums : ℕ := 3

-- Define the number of instruments for Nick
def nick_flutes : ℕ := charlie_flutes + carli_flutes
def nick_horns : ℕ := charlie_horns - carli_horns
def nick_harps : ℕ := 0
def nick_drums : ℕ := 4

-- Define the total number of instruments
def total_instruments : ℕ := 
  charlie_flutes + charlie_horns + charlie_harps + charlie_drums +
  carli_flutes + carli_horns + carli_harps + carli_drums +
  nick_flutes + nick_horns + nick_harps + nick_drums

-- Theorem statement
theorem total_instruments_is_19 : total_instruments = 19 := by
  sorry

end total_instruments_is_19_l783_78338


namespace double_root_values_l783_78398

/-- A polynomial with integer coefficients of the form x^4 + a₃x³ + a₂x² + a₁x + 18 -/
def P (a₃ a₂ a₁ : ℤ) (x : ℝ) : ℝ := x^4 + a₃*x^3 + a₂*x^2 + a₁*x + 18

/-- r is a double root of P if (x - r)² divides P -/
def is_double_root (r : ℤ) (a₃ a₂ a₁ : ℤ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, P a₃ a₂ a₁ x = (x - r)^2 * q x

theorem double_root_values (a₃ a₂ a₁ : ℤ) (r : ℤ) :
  is_double_root r a₃ a₂ a₁ → r = -3 ∨ r = -1 ∨ r = 1 ∨ r = 3 := by
  sorry

end double_root_values_l783_78398


namespace triangle_median_properties_l783_78339

/-- Properties of triangle medians -/
theorem triangle_median_properties
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (P p : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : P = a + b + c)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_median_a : 4 * ma^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h_median_b : 4 * mb^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h_median_c : 4 * mc^2 = 2 * a^2 + 2 * b^2 - c^2) :
  (ma + mb ≤ 3/4 * P) ∧ (ma + mb ≥ 3/4 * p) :=
by sorry

end triangle_median_properties_l783_78339


namespace matthew_crackers_l783_78368

def crackers_problem (total_crackers : ℕ) (crackers_per_friend : ℕ) : Prop :=
  total_crackers / crackers_per_friend = 4

theorem matthew_crackers : crackers_problem 8 2 := by
  sorry

end matthew_crackers_l783_78368


namespace two_real_roots_for_radical_equation_l783_78322

-- Define the function f(x) derived from the original equation
def f (a b c x : ℝ) : ℝ :=
  3 * x^2 - 2 * (a + b + c) * x - (a^2 + b^2 + c^2) + 2 * (a * b + b * c + c * a)

-- Main theorem statement
theorem two_real_roots_for_radical_equation (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (∀ x : ℝ, f a b c x = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end two_real_roots_for_radical_equation_l783_78322


namespace prob_zeros_not_adjacent_is_point_six_l783_78319

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements to be arranged -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when arranging num_ones ones and num_zeros zeros in a row -/
def prob_zeros_not_adjacent : ℚ :=
  1 - (2 * (Nat.factorial (total_elements - 1))) / (Nat.factorial total_elements)

theorem prob_zeros_not_adjacent_is_point_six :
  prob_zeros_not_adjacent = 3/5 :=
sorry

end prob_zeros_not_adjacent_is_point_six_l783_78319


namespace max_distance_theorem_l783_78374

/-- Represents the characteristics of a motor boat on a river -/
structure RiverBoat where
  upstream_distance : ℝ  -- Distance the boat can travel upstream on a full tank
  downstream_distance : ℝ -- Distance the boat can travel downstream on a full tank

/-- Calculates the maximum round trip distance for a given boat -/
def max_round_trip_distance (boat : RiverBoat) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating the maximum round trip distance for the given boat -/
theorem max_distance_theorem (boat : RiverBoat) 
  (h1 : boat.upstream_distance = 40)
  (h2 : boat.downstream_distance = 60) :
  max_round_trip_distance boat = 24 := by
  sorry

end max_distance_theorem_l783_78374


namespace triangle_dot_product_l783_78363

/-- Given a triangle ABC with |AB| = 4, |AC| = 1, and area = √3,
    prove that the dot product of AB and AC is ±2 -/
theorem triangle_dot_product (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1^2 + AB.2^2 = 16) →  -- |AB| = 4
  (AC.1^2 + AC.2^2 = 1) →   -- |AC| = 1
  (abs (AB.1 * AC.2 - AB.2 * AC.1) = 2 * Real.sqrt 3) →  -- Area = √3
  ((AB.1 * AC.1 + AB.2 * AC.2)^2 = 4) :=  -- Dot product squared = 4
by sorry

end triangle_dot_product_l783_78363


namespace evaluate_expression_l783_78392

theorem evaluate_expression : (64 : ℝ) ^ (0.125 : ℝ) * (64 : ℝ) ^ (0.375 : ℝ) = 8 := by
  sorry

end evaluate_expression_l783_78392


namespace sarah_won_30_games_l783_78340

/-- Sarah's tic-tac-toe game results -/
structure TicTacToeResults where
  total_games : ℕ
  tied_games : ℕ
  total_money : ℤ
  win_money : ℕ
  tie_money : ℕ
  lose_money : ℕ

/-- Theorem: Sarah won 30 games -/
theorem sarah_won_30_games (results : TicTacToeResults)
  (h1 : results.total_games = 100)
  (h2 : results.tied_games = 40)
  (h3 : results.total_money = -30)
  (h4 : results.win_money = 1)
  (h5 : results.tie_money = 0)
  (h6 : results.lose_money = 2) :
  ∃ (won_games lost_games : ℕ),
    won_games + results.tied_games + lost_games = results.total_games ∧
    won_games * results.win_money - lost_games * results.lose_money = results.total_money ∧
    won_games = 30 :=
  sorry

end sarah_won_30_games_l783_78340


namespace bens_initial_marbles_l783_78378

theorem bens_initial_marbles (B : ℕ) : 
  (17 + B / 2 = B / 2 + 17) → B = 34 := by sorry

end bens_initial_marbles_l783_78378


namespace quadratic_positive_function_m_range_l783_78382

/-- A function is positive on a domain if there exists a subinterval where the function maps the interval to itself -/
def PositiveFunction (f : ℝ → ℝ) (D : Set ℝ) :=
  ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧ Set.Icc a b = f '' Set.Icc a b

/-- The quadratic function g(x) = x^2 - m -/
def g (m : ℝ) : ℝ → ℝ := fun x ↦ x^2 - m

theorem quadratic_positive_function_m_range :
  (∃ m, PositiveFunction (g m) (Set.Iio 0)) → ∃ m, m ∈ Set.Ioo (3/4) 1 :=
by sorry

end quadratic_positive_function_m_range_l783_78382


namespace sequence_range_l783_78365

/-- Given an infinite sequence {a_n} satisfying the recurrence relation
    a_{n+1} = p * a_n + 1 / a_n for n ∈ ℕ*, where p is a positive real number,
    a_1 = 2, and {a_n} is monotonically decreasing, prove that p ∈ (1/2, 3/4). -/
theorem sequence_range (p : ℝ) (a : ℕ+ → ℝ) 
  (h_pos : p > 0)
  (h_rec : ∀ n : ℕ+, a (n + 1) = p * a n + 1 / a n)
  (h_init : a 1 = 2)
  (h_decr : ∀ n : ℕ+, a (n + 1) ≤ a n) :
  p > 1/2 ∧ p < 3/4 := by
sorry


end sequence_range_l783_78365
