import Mathlib

namespace initial_wage_solution_l131_13124

def initial_wage_problem (x : ℝ) : Prop :=
  let after_raise := x * 1.20
  let after_cut := after_raise * 0.75
  after_cut = 9

theorem initial_wage_solution :
  ∃ x : ℝ, initial_wage_problem x ∧ x = 10 := by
  sorry

end initial_wage_solution_l131_13124


namespace chess_tournament_games_l131_13198

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 7 players, where each player plays twice with every other player, the total number of games played is 84 -/
theorem chess_tournament_games :
  tournament_games 7 * 2 = 84 := by
  sorry

end chess_tournament_games_l131_13198


namespace factorial_simplification_l131_13131

theorem factorial_simplification : (15 : ℕ).factorial / ((12 : ℕ).factorial + 3 * (11 : ℕ).factorial) = 4680 := by
  sorry

end factorial_simplification_l131_13131


namespace camphor_ball_shrinkage_l131_13199

/-- The time it takes for a camphor ball to shrink to a specific volume -/
theorem camphor_ball_shrinkage (a k : ℝ) (h1 : a > 0) (h2 : k > 0) : 
  let V : ℝ → ℝ := λ t => a * Real.exp (-k * t)
  (V 50 = 4/9 * a) → (V 75 = 8/27 * a) := by
  sorry

end camphor_ball_shrinkage_l131_13199


namespace matching_probability_five_pairs_l131_13119

/-- A box containing shoes -/
structure ShoeBox where
  pairs : ℕ
  total : ℕ
  total_eq : total = 2 * pairs

/-- The probability of selecting a matching pair of shoes -/
def matchingProbability (box : ShoeBox) : ℚ :=
  box.pairs / (box.total * (box.total - 1) / 2)

/-- Theorem: The probability of selecting a matching pair from a box with 5 pairs is 1/9 -/
theorem matching_probability_five_pairs :
  let box : ShoeBox := ⟨5, 10, rfl⟩
  matchingProbability box = 1 / 9 := by
  sorry


end matching_probability_five_pairs_l131_13119


namespace f_satisfies_conditions_l131_13129

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 := by sorry

end f_satisfies_conditions_l131_13129


namespace max_value_expression_max_value_achievable_l131_13195

theorem max_value_expression (y : ℝ) (h : y > 0) :
  (y^2 + 3 - Real.sqrt (y^4 + 9)) / y ≤ 6 / (2 * Real.sqrt 3 + Real.sqrt 6) :=
sorry

theorem max_value_achievable :
  ∃ y : ℝ, y > 0 ∧ (y^2 + 3 - Real.sqrt (y^4 + 9)) / y = 6 / (2 * Real.sqrt 3 + Real.sqrt 6) :=
sorry

end max_value_expression_max_value_achievable_l131_13195


namespace rachels_brownies_l131_13153

/-- Rachel's brownie problem -/
theorem rachels_brownies (total : ℕ) (left_at_home : ℕ) (brought_to_school : ℕ) : 
  total = 40 → left_at_home = 24 → brought_to_school = total - left_at_home → brought_to_school = 16 := by
  sorry

#check rachels_brownies

end rachels_brownies_l131_13153


namespace sufficient_not_necessary_condition_l131_13143

theorem sufficient_not_necessary_condition :
  ∃ (S : Set ℝ), 
    (∀ x ∈ S, x^2 - 4*x < 0) ∧ 
    (S ⊂ {x : ℝ | 0 < x ∧ x < 4}) ∧
    S = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end sufficient_not_necessary_condition_l131_13143


namespace hiker_final_distance_l131_13114

-- Define the hiker's movements
def east_distance : ℝ := 15
def south_distance : ℝ := 20
def west_distance : ℝ := 15
def north_distance : ℝ := 5

-- Define the net horizontal and vertical movements
def net_horizontal : ℝ := east_distance - west_distance
def net_vertical : ℝ := south_distance - north_distance

-- Theorem to prove
theorem hiker_final_distance :
  Real.sqrt (net_horizontal ^ 2 + net_vertical ^ 2) = 15 := by
  sorry

end hiker_final_distance_l131_13114


namespace book_purchase_theorem_l131_13188

/-- The number of people who purchased only book A -/
def Z : ℕ := 1000

/-- The number of people who purchased only book B -/
def X : ℕ := 250

/-- The number of people who purchased both books A and B -/
def Y : ℕ := 500

/-- The total number of people who purchased book A -/
def A : ℕ := Z + Y

/-- The total number of people who purchased book B -/
def B : ℕ := X + Y

theorem book_purchase_theorem :
  (A = 2 * B) ∧             -- The number of people who purchased book A is twice the number of people who purchased book B
  (Y = 500) ∧               -- The number of people who purchased both books A and B is 500
  (Y = 2 * X) ∧             -- The number of people who purchased both books A and B is twice the number of people who purchased only book B
  (Z = 1000) :=             -- The number of people who purchased only book A is 1000
by sorry

end book_purchase_theorem_l131_13188


namespace only_point_distance_no_conditional_l131_13125

-- Define the four types of mathematical problems
inductive MathProblem
  | QuadraticEquation
  | LineCircleRelationship
  | StudentRanking
  | PointDistance

-- Define a function that determines if a problem requires conditional statements
def requiresConditionalStatements (problem : MathProblem) : Prop :=
  match problem with
  | MathProblem.QuadraticEquation => true
  | MathProblem.LineCircleRelationship => true
  | MathProblem.StudentRanking => true
  | MathProblem.PointDistance => false

-- Theorem stating that only PointDistance does not require conditional statements
theorem only_point_distance_no_conditional :
  ∀ (problem : MathProblem),
    ¬(requiresConditionalStatements problem) ↔ problem = MathProblem.PointDistance := by
  sorry

#check only_point_distance_no_conditional

end only_point_distance_no_conditional_l131_13125


namespace total_bike_rides_l131_13164

theorem total_bike_rides (billy_rides : ℕ) (john_rides : ℕ) (mother_rides : ℕ) : 
  billy_rides = 17 →
  john_rides = 2 * billy_rides →
  mother_rides = john_rides + 10 →
  billy_rides + john_rides + mother_rides = 95 := by
sorry

end total_bike_rides_l131_13164


namespace smallest_fraction_l131_13133

theorem smallest_fraction (x : ℝ) (h : x > 2022) :
  min (x / 2022) (min (2022 / (x - 1)) (min ((x + 1) / 2022) (min (2022 / x) (2022 / (x + 1))))) = 2022 / (x + 1) := by
  sorry

end smallest_fraction_l131_13133


namespace sum_of_f_values_l131_13135

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem sum_of_f_values : 
  Real.sqrt 3 * (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + 
                 f 1 + f 2 + f 3 + f 4 + f 5 + f 6) = 6 := by
  sorry

end sum_of_f_values_l131_13135


namespace triangle_properties_l131_13136

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides opposite to A, B, C respectively

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.A = Real.sqrt 3 * abc.a * Real.cos abc.B)
  (h2 : abc.b = 3)
  (h3 : Real.sin abc.C = 2 * Real.sin abc.A) :
  (abc.B = π/3) ∧ (abc.a = Real.sqrt 3) ∧ (abc.c = 2 * Real.sqrt 3) := by
  sorry

end triangle_properties_l131_13136


namespace right_triangle_solution_l131_13197

theorem right_triangle_solution (A B C : Real) (a b c : ℝ) :
  -- Given conditions
  (A + B + C = π) →  -- Sum of angles in a triangle
  (C = π / 2) →      -- Right angle at C
  (a = Real.sqrt 5) →
  (b = Real.sqrt 15) →
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem
  (Real.tan A = a / b) →  -- Definition of tangent
  -- Conclusions
  (c = 2 * Real.sqrt 5) ∧
  (A = π / 6) ∧  -- 30 degrees in radians
  (B = π / 3) :=  -- 60 degrees in radians
by sorry

end right_triangle_solution_l131_13197


namespace cubic_root_odd_and_increasing_l131_13127

-- Define the function
def f (x : ℝ) : ℝ := x^(1/3)

-- State the theorem
theorem cubic_root_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end cubic_root_odd_and_increasing_l131_13127


namespace triangle_determinant_zero_l131_13175

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = π) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.cos A ^ 2, Real.tan A, 1],
    ![Real.cos B ^ 2, Real.tan B, 1],
    ![Real.cos C ^ 2, Real.tan C, 1]
  ]
  Matrix.det M = 0 := by
  sorry

end triangle_determinant_zero_l131_13175


namespace problem_solution_l131_13178

theorem problem_solution (m : ℝ) (h : m + 1/m = 10) : 
  m^2 + 1/m^2 + m + 1/m = 108 := by
sorry

end problem_solution_l131_13178


namespace game_result_l131_13172

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 2, 5]
def carl_rolls : List ℕ := [1, 4, 3, 6, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : total_points allie_rolls * total_points carl_rolls = 594 := by
  sorry

end game_result_l131_13172


namespace trigonometric_calculations_l131_13154

theorem trigonometric_calculations :
  (2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180) + Real.sqrt 9 = Real.sqrt 3) ∧
  (Real.cos (30 * π / 180) / (1 + Real.sin (30 * π / 180)) + Real.tan (60 * π / 180) = 4 * Real.sqrt 3 / 3) :=
by sorry

end trigonometric_calculations_l131_13154


namespace no_solution_condition_l131_13120

theorem no_solution_condition (r : ℝ) :
  (∀ x y : ℝ, x^2 = y^2 ∧ (x - r)^2 + y^2 = 1 → False) ↔ r < -Real.sqrt 2 ∨ r > Real.sqrt 2 := by
  sorry

end no_solution_condition_l131_13120


namespace chess_tournament_participants_l131_13169

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 171 → n = 19 := by
  sorry

end chess_tournament_participants_l131_13169


namespace unique_zero_addition_l131_13160

theorem unique_zero_addition (x : ℤ) :
  (∀ n : ℤ, n + x = n) ↔ x = 0 := by
  sorry

end unique_zero_addition_l131_13160


namespace expression_equality_equation_solutions_l131_13113

-- Problem 1
theorem expression_equality : 
  |Real.sqrt 3 - 1| - 2 * Real.cos (π / 3) + (Real.sqrt 3 - 2)^2 + Real.sqrt 12 = 5 - Real.sqrt 3 := by
  sorry

-- Problem 2
theorem equation_solutions (x : ℝ) : 
  2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 := by
  sorry

end expression_equality_equation_solutions_l131_13113


namespace integral_x_cos_x_plus_cube_root_x_squared_l131_13156

open Real
open MeasureTheory
open Interval

theorem integral_x_cos_x_plus_cube_root_x_squared : 
  ∫ x in (-1)..1, (x * cos x + (x^2)^(1/3)) = 6/5 := by
  sorry

end integral_x_cos_x_plus_cube_root_x_squared_l131_13156


namespace sum_c_d_eq_eight_l131_13111

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  c : ℝ
  d : ℝ
  h : (2 * 4 + c = 16) ∧ (4 * 4 + d = 16)

/-- The sum of c and d for intersecting lines -/
def sum_c_d (lines : IntersectingLines) : ℝ := lines.c + lines.d

/-- Theorem: The sum of c and d equals 8 -/
theorem sum_c_d_eq_eight (lines : IntersectingLines) : sum_c_d lines = 8 := by
  sorry

end sum_c_d_eq_eight_l131_13111


namespace michael_digging_time_l131_13191

/-- Given the conditions of Michael and his father's digging, prove that Michael will take 700 hours to dig his hole. -/
theorem michael_digging_time (father_rate : ℝ) (father_time : ℝ) (depth_difference : ℝ) :
  father_rate = 4 →
  father_time = 400 →
  depth_difference = 400 →
  let father_depth := father_rate * father_time
  let michael_depth := 2 * father_depth - depth_difference
  michael_depth / father_rate = 700 := by
  sorry

end michael_digging_time_l131_13191


namespace quadrilateral_ABCD_is_parallelogram_l131_13170

-- Define the vertices of the quadrilateral
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, -1)
def C : ℝ × ℝ := (4, 2)
def D : ℝ × ℝ := (2, 3)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def DC : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)

-- Define a function to check if two vectors are equal
def vectors_equal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 = v2.1 ∧ v1.2 = v2.2

-- Define what it means for a quadrilateral to be a parallelogram
def is_parallelogram (a b c d : ℝ × ℝ) : Prop :=
  vectors_equal (b.1 - a.1, b.2 - a.2) (c.1 - d.1, c.2 - d.2) ∧
  vectors_equal (c.1 - b.1, c.2 - b.2) (d.1 - a.1, d.2 - a.2)

-- Theorem statement
theorem quadrilateral_ABCD_is_parallelogram :
  is_parallelogram A B C D :=
by sorry

end quadrilateral_ABCD_is_parallelogram_l131_13170


namespace circles_covering_path_implies_odd_l131_13152

/-- A configuration of n circles on a plane. -/
structure CircleConfiguration (n : ℕ) where
  /-- The set of circles. -/
  circles : Fin n → Set (ℝ × ℝ)
  /-- Any two circles intersect at exactly two points. -/
  two_intersections : ∀ (i j : Fin n), i ≠ j → ∃! (p q : ℝ × ℝ), p ≠ q ∧ p ∈ circles i ∧ p ∈ circles j ∧ q ∈ circles i ∧ q ∈ circles j
  /-- No three circles have a common point. -/
  no_triple_intersection : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ¬∃ (p : ℝ × ℝ), p ∈ circles i ∧ p ∈ circles j ∧ p ∈ circles k

/-- A path that covers all circles in the configuration. -/
def CoveringPath (n : ℕ) (config : CircleConfiguration n) :=
  ∃ (path : ℕ → Fin n), ∀ (i : Fin n), ∃ (k : ℕ), path k = i

/-- The main theorem: if there exists a covering path for n circles satisfying the given conditions,
    then n must be odd. -/
theorem circles_covering_path_implies_odd (n : ℕ) (config : CircleConfiguration n) :
  CoveringPath n config → Odd n :=
sorry

end circles_covering_path_implies_odd_l131_13152


namespace polar_to_cartesian_equivalence_l131_13112

def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.sin θ - ρ * (Real.cos θ)^2 - Real.sin θ = 0

def cartesian_equation (x y : ℝ) : Prop :=
  x = 1 ∨ (x^2 + y^2 + y = 0 ∧ y ≠ 0)

theorem polar_to_cartesian_equivalence :
  ∀ x y ρ θ, θ ∈ Set.Ioo 0 Real.pi →
  x = ρ * Real.cos θ → y = ρ * Real.sin θ →
  (polar_equation ρ θ ↔ cartesian_equation x y) := by sorry

end polar_to_cartesian_equivalence_l131_13112


namespace ring_toss_earnings_l131_13144

theorem ring_toss_earnings (daily_earnings : ℕ) (days : ℕ) (total_earnings : ℕ) : 
  daily_earnings = 144 → days = 22 → total_earnings = daily_earnings * days → total_earnings = 3168 := by
  sorry

end ring_toss_earnings_l131_13144


namespace pie_chart_probability_l131_13146

theorem pie_chart_probability (W X Y Z : ℝ) : 
  W = 1/4 → X = 1/3 → Z = 1/6 → W + X + Y + Z = 1 → Y = 1/4 := by
  sorry

end pie_chart_probability_l131_13146


namespace sqrt_x_minus_5_meaningful_l131_13109

theorem sqrt_x_minus_5_meaningful (x : ℝ) : 
  ∃ y : ℝ, y ^ 2 = x - 5 ↔ x ≥ 5 := by
  sorry

end sqrt_x_minus_5_meaningful_l131_13109


namespace area_relation_l131_13130

-- Define the triangles
structure Triangle :=
  (O A B : ℝ × ℝ)

-- Define properties of isosceles right triangles
def IsIsoscelesRight (t : Triangle) : Prop :=
  let (xO, yO) := t.O
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  (xA - xO)^2 + (yA - yO)^2 = (xB - xO)^2 + (yB - yO)^2 ∧
  (xB - xA)^2 + (yB - yA)^2 = 2 * ((xA - xO)^2 + (yA - yO)^2)

-- Define the area of a triangle
def Area (t : Triangle) : ℝ :=
  let (xO, yO) := t.O
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  0.5 * abs ((xA - xO) * (yB - yO) - (xB - xO) * (yA - yO))

-- Theorem statement
theorem area_relation (OAB OBC OCD : Triangle) :
  IsIsoscelesRight OAB ∧ IsIsoscelesRight OBC ∧ IsIsoscelesRight OCD →
  Area OCD = 12 →
  Area OAB = 3 :=
by sorry

end area_relation_l131_13130


namespace percentage_difference_l131_13179

theorem percentage_difference (x y : ℝ) (h : x = 8 * y) :
  (x - y) / x * 100 = 87.5 := by
  sorry

end percentage_difference_l131_13179


namespace nina_widget_problem_l131_13142

theorem nina_widget_problem (x : ℝ) 
  (h1 : 15 * x = 25 * (x - 5)) : 
  15 * x = 187.50 := by
  sorry

end nina_widget_problem_l131_13142


namespace probability_reach_target_l131_13196

-- Define the step type
inductive Step
| Left
| Right
| Up
| Down

-- Define the position type
structure Position :=
  (x : Int) (y : Int)

-- Define the function to take a step
def takeStep (pos : Position) (step : Step) : Position :=
  match step with
  | Step.Left  => ⟨pos.x - 1, pos.y⟩
  | Step.Right => ⟨pos.x + 1, pos.y⟩
  | Step.Up    => ⟨pos.x, pos.y + 1⟩
  | Step.Down  => ⟨pos.x, pos.y - 1⟩

-- Define the probability of a single step
def stepProbability : ℚ := 1/4

-- Define the function to check if a position is (3,1)
def isTarget (pos : Position) : Bool :=
  pos.x = 3 ∧ pos.y = 1

-- Define the theorem
theorem probability_reach_target :
  ∃ (paths : Finset (List Step)),
    (∀ path ∈ paths, path.length ≤ 8) ∧
    (∀ path ∈ paths, isTarget (path.foldl takeStep ⟨0, 0⟩)) ∧
    (paths.card : ℚ) * stepProbability ^ 8 = 7/128 :=
sorry

end probability_reach_target_l131_13196


namespace solve_linear_system_l131_13166

theorem solve_linear_system (a b : ℤ) 
  (eq1 : 2009 * a + 2013 * b = 2021)
  (eq2 : 2011 * a + 2015 * b = 2023) :
  a - b = -5 := by
  sorry

end solve_linear_system_l131_13166


namespace arithmetic_computation_l131_13116

theorem arithmetic_computation : 5 * 7 + 6 * 12 + 10 * 4 + 7 * 6 + 30 / 5 = 195 := by
  sorry

end arithmetic_computation_l131_13116


namespace intersection_implies_solution_l131_13126

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem intersection_implies_solution (k b : ℝ) :
  linear_function k b (-3) = 0 →
  (∃ x : ℝ, -k * x + b = 0) ∧
  (∀ x : ℝ, -k * x + b = 0 → x = 3) :=
by sorry

end intersection_implies_solution_l131_13126


namespace troy_vegetable_purchase_l131_13117

/-- The number of pounds of vegetables Troy buys -/
def vegetable_pounds : ℝ := 6

/-- The number of pounds of beef Troy buys -/
def beef_pounds : ℝ := 4

/-- The cost of vegetables per pound in dollars -/
def vegetable_cost_per_pound : ℝ := 2

/-- The total cost of Troy's purchase in dollars -/
def total_cost : ℝ := 36

/-- Theorem stating that the number of pounds of vegetables Troy buys is 6 -/
theorem troy_vegetable_purchase :
  vegetable_pounds = 6 ∧
  beef_pounds = 4 ∧
  vegetable_cost_per_pound = 2 ∧
  total_cost = 36 ∧
  (3 * vegetable_cost_per_pound * beef_pounds + vegetable_cost_per_pound * vegetable_pounds = total_cost) :=
by sorry

end troy_vegetable_purchase_l131_13117


namespace min_value_reciprocal_sum_l131_13165

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / a + 2 / b) ≥ 3 / 2 + Real.sqrt 2 := by
  sorry

end min_value_reciprocal_sum_l131_13165


namespace palmer_photos_l131_13193

def total_photos (initial : ℕ) (first_week : ℕ) (third_fourth_week : ℕ) : ℕ :=
  initial + first_week + 2 * first_week + third_fourth_week

theorem palmer_photos : total_photos 100 50 80 = 330 := by
  sorry

end palmer_photos_l131_13193


namespace imaginary_part_of_3_minus_4i_l131_13159

theorem imaginary_part_of_3_minus_4i :
  Complex.im (3 - 4 * Complex.I) = -4 := by sorry

end imaginary_part_of_3_minus_4i_l131_13159


namespace eugenes_living_room_length_l131_13139

/-- Represents the properties of a rectangular room --/
structure RectangularRoom where
  width : ℝ
  area : ℝ
  length : ℝ

/-- Theorem stating the length of Eugene's living room --/
theorem eugenes_living_room_length (room : RectangularRoom)
  (h1 : room.width = 14)
  (h2 : room.area = 215.6)
  (h3 : room.area = room.length * room.width) :
  room.length = 15.4 := by
  sorry

end eugenes_living_room_length_l131_13139


namespace hole_perimeter_formula_l131_13105

/-- Represents an isosceles trapezium -/
structure IsoscelesTrapezium where
  a : ℝ  -- Length of non-parallel sides
  b : ℝ  -- Length of longer parallel side

/-- Represents an equilateral triangle with a hole formed by three congruent isosceles trapeziums -/
structure TriangleWithHole where
  trapezium : IsoscelesTrapezium
  -- Assumption that three of these trapeziums form an equilateral triangle with a hole

/-- The perimeter of the hole in a TriangleWithHole -/
def holePerimeter (t : TriangleWithHole) : ℝ :=
  6 * t.trapezium.a - 3 * t.trapezium.b

/-- Theorem stating that the perimeter of the hole is 6a - 3b -/
theorem hole_perimeter_formula (t : TriangleWithHole) :
  holePerimeter t = 6 * t.trapezium.a - 3 * t.trapezium.b :=
by
  sorry

end hole_perimeter_formula_l131_13105


namespace difference_of_odd_squares_divisible_by_eight_l131_13194

theorem difference_of_odd_squares_divisible_by_eight (a b : Int) 
  (ha : a % 2 = 1) (hb : b % 2 = 1) : 
  ∃ k : Int, a^2 - b^2 = 8 * k := by
sorry

end difference_of_odd_squares_divisible_by_eight_l131_13194


namespace system_solution_l131_13123

theorem system_solution :
  ∃ (m n : ℚ), m / 3 + n / 2 = 1 ∧ m - 2 * n = 2 ∧ m = 18 / 7 ∧ n = 2 / 7 := by
  sorry

end system_solution_l131_13123


namespace scallop_dinner_cost_l131_13107

/-- Represents the problem of calculating the cost of scallops for Nate's dinner. -/
theorem scallop_dinner_cost :
  let scallops_per_pound : ℕ := 8
  let cost_per_pound : ℚ := 24
  let scallops_per_person : ℕ := 2
  let number_of_people : ℕ := 8
  
  let total_scallops : ℕ := scallops_per_person * number_of_people
  let pounds_needed : ℚ := total_scallops / scallops_per_pound
  let total_cost : ℚ := pounds_needed * cost_per_pound
  
  total_cost = 48 :=
by
  sorry


end scallop_dinner_cost_l131_13107


namespace unique_positive_solution_l131_13187

theorem unique_positive_solution : 
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 - 7 * x - 6 = 0 :=
by
  -- The proof goes here
  sorry

end unique_positive_solution_l131_13187


namespace equation_solution_l131_13180

theorem equation_solution (a b c : ℝ) (h : 1 / a - 1 / b = 2 / c) : c = a * b * (b - a) / 2 := by
  sorry

end equation_solution_l131_13180


namespace sum_of_squares_l131_13183

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 4)
  (eq2 : y^2 - 5*z = 5)
  (eq3 : z^2 - 7*x = -8) :
  x^2 + y^2 + z^2 = 83/4 := by
sorry

end sum_of_squares_l131_13183


namespace inequality_chain_l131_13176

theorem inequality_chain (a b x : ℝ) (h1 : 0 < b) (h2 : b < x) (h3 : x < a) :
  b * x < x^2 ∧ x^2 < a^2 := by
  sorry

end inequality_chain_l131_13176


namespace expression_equals_3840_factorial_l131_13171

/-- Custom factorial definition for positive p and b -/
def custom_factorial (p b : ℕ) : ℕ :=
  sorry

/-- The result of the expression 120₁₀!/20₃! + (10₂!)! -/
def expression_result : ℕ :=
  sorry

/-- Theorem stating that the expression equals (3840)! -/
theorem expression_equals_3840_factorial :
  expression_result = Nat.factorial 3840 :=
  sorry

end expression_equals_3840_factorial_l131_13171


namespace non_congruent_squares_on_6x6_grid_l131_13103

/-- A square on a lattice grid --/
structure LatticeSquare where
  side_length : ℕ
  is_diagonal : Bool

/-- The size of the grid --/
def grid_size : ℕ := 6

/-- Counts the number of squares with a given side length that fit on the grid --/
def count_squares (s : ℕ) : ℕ :=
  (grid_size - s + 1) ^ 2

/-- Counts all non-congruent squares on the grid --/
def total_non_congruent_squares : ℕ :=
  (List.range 5).map (λ i => count_squares (i + 1)) |> List.sum

/-- The main theorem stating the number of non-congruent squares on a 6x6 grid --/
theorem non_congruent_squares_on_6x6_grid :
  total_non_congruent_squares = 110 := by
  sorry


end non_congruent_squares_on_6x6_grid_l131_13103


namespace number_wall_m_equals_one_l131_13149

/-- Represents a simplified version of the number wall structure -/
structure NumberWall where
  m : ℤ
  top : ℤ
  left : ℤ
  right : ℤ

/-- The number wall satisfies the given conditions -/
def valid_wall (w : NumberWall) : Prop :=
  w.top = w.left + w.right ∧ w.left = w.m + 22 ∧ w.right = 35 ∧ w.top = 58

/-- Theorem: In the given number wall structure, m = 1 -/
theorem number_wall_m_equals_one (w : NumberWall) (h : valid_wall w) : w.m = 1 := by
  sorry

end number_wall_m_equals_one_l131_13149


namespace three_numbers_sum_l131_13157

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 := by
  sorry

end three_numbers_sum_l131_13157


namespace distance_from_origin_l131_13184

theorem distance_from_origin (x y : ℝ) (h1 : |x| = 8) 
  (h2 : Real.sqrt ((x - 7)^2 + (y - 3)^2) = 8) (h3 : y > 3) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (136 + 6 * Real.sqrt 63) :=
by sorry

end distance_from_origin_l131_13184


namespace right_triangle_roots_l131_13150

theorem right_triangle_roots (p : ℝ) : 
  (∃ a b c : ℝ, 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (a^3 - 2*p*(p+1)*a^2 + (p^4 + 4*p^3 - 1)*a - 3*p^3 = 0) ∧
    (b^3 - 2*p*(p+1)*b^2 + (p^4 + 4*p^3 - 1)*b - 3*p^3 = 0) ∧
    (c^3 - 2*p*(p+1)*c^2 + (p^4 + 4*p^3 - 1)*c - 3*p^3 = 0) ∧
    (a^2 + b^2 = c^2)) ↔ 
  p = Real.sqrt 2 :=
sorry

end right_triangle_roots_l131_13150


namespace square_of_difference_with_sqrt_l131_13147

theorem square_of_difference_with_sqrt (x : ℝ) : 
  (7 - Real.sqrt (x^2 - 49*x + 169))^2 = x^2 - 49*x + 218 - 14*Real.sqrt (x^2 - 49*x + 169) := by
  sorry

end square_of_difference_with_sqrt_l131_13147


namespace randy_lunch_cost_l131_13192

theorem randy_lunch_cost (initial_amount : ℝ) (ice_cream_cost : ℝ) : 
  initial_amount = 30 →
  ice_cream_cost = 5 →
  ∃ (lunch_cost : ℝ),
    lunch_cost = 10 ∧
    (1/4) * (initial_amount - lunch_cost) = ice_cream_cost :=
by
  sorry

end randy_lunch_cost_l131_13192


namespace cube_volume_from_surface_area_l131_13140

theorem cube_volume_from_surface_area :
  ∀ (surface_area : ℝ) (volume : ℝ),
    surface_area = 384 →
    volume = (surface_area / 6) ^ (3/2) →
    volume = 512 :=
by
  sorry

end cube_volume_from_surface_area_l131_13140


namespace side_is_one_third_perimeter_l131_13132

-- Define a triangle with an inscribed circle
structure TriangleWithInscribedCircle where
  -- We don't need to explicitly define the triangle or circle, 
  -- just the properties we're interested in
  side_length : ℝ
  perimeter : ℝ
  midpoint : ℝ × ℝ
  altitude_foot : ℝ × ℝ
  tangency_point : ℝ × ℝ

-- Define the symmetry condition
def is_symmetrical (t : TriangleWithInscribedCircle) : Prop :=
  let midpoint_distance := (t.midpoint.1 - t.tangency_point.1)^2 + (t.midpoint.2 - t.tangency_point.2)^2
  let altitude_foot_distance := (t.altitude_foot.1 - t.tangency_point.1)^2 + (t.altitude_foot.2 - t.tangency_point.2)^2
  midpoint_distance = altitude_foot_distance

-- State the theorem
theorem side_is_one_third_perimeter (t : TriangleWithInscribedCircle) 
  (h : is_symmetrical t) : t.side_length = t.perimeter / 3 := by
  sorry

end side_is_one_third_perimeter_l131_13132


namespace shoe_selection_problem_l131_13182

theorem shoe_selection_problem (n : ℕ) (h : n = 10) : 
  (n.choose 1) * ((n - 1).choose 2) * (2^2) = 1440 := by
  sorry

end shoe_selection_problem_l131_13182


namespace problem_solution_l131_13134

noncomputable def θ : ℝ := sorry

-- The terminal side of angle θ lies on the ray y = 2x (x ≥ 0)
axiom h : ∀ x : ℝ, x ≥ 0 → Real.tan θ * x = 2 * x

theorem problem_solution :
  (Real.tan θ = 2) ∧
  ((2 * Real.cos θ + 3 * Real.sin θ) / (Real.cos θ - 3 * Real.sin θ) + Real.sin θ * Real.cos θ = -6/5) :=
by sorry

end problem_solution_l131_13134


namespace a_grazing_months_l131_13118

/-- Represents the number of months 'a' put his oxen for grazing -/
def a_months : ℕ := sorry

/-- Represents the number of oxen 'a' put for grazing -/
def a_oxen : ℕ := 10

/-- Represents the number of oxen 'b' put for grazing -/
def b_oxen : ℕ := 12

/-- Represents the number of months 'b' put his oxen for grazing -/
def b_months : ℕ := 5

/-- Represents the number of oxen 'c' put for grazing -/
def c_oxen : ℕ := 15

/-- Represents the number of months 'c' put his oxen for grazing -/
def c_months : ℕ := 3

/-- Represents the total rent of the pasture in Rs. -/
def total_rent : ℕ := 105

/-- Represents 'c's share of the rent in Rs. -/
def c_share : ℕ := 27

/-- Theorem stating that 'a' put his oxen for grazing for 7 months -/
theorem a_grazing_months : a_months = 7 := by sorry

end a_grazing_months_l131_13118


namespace dog_collar_nylon_l131_13167

/-- The number of inches of nylon needed for one cat collar -/
def cat_collar_nylon : ℕ := 10

/-- The total number of inches of nylon needed for 9 dog collars and 3 cat collars -/
def total_nylon : ℕ := 192

/-- The number of dog collars made -/
def num_dog_collars : ℕ := 9

/-- The number of cat collars made -/
def num_cat_collars : ℕ := 3

/-- Theorem stating that 18 inches of nylon are needed for one dog collar -/
theorem dog_collar_nylon : 
  ∃ (x : ℕ), x * num_dog_collars + cat_collar_nylon * num_cat_collars = total_nylon ∧ x = 18 := by
  sorry

end dog_collar_nylon_l131_13167


namespace intersection_of_A_and_B_l131_13137

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l131_13137


namespace coins_probability_theorem_l131_13104

def total_coins : ℕ := 15
def num_quarters : ℕ := 3
def num_dimes : ℕ := 5
def num_nickels : ℕ := 7
def coins_drawn : ℕ := 8

def value_quarter : ℚ := 25 / 100
def value_dime : ℚ := 10 / 100
def value_nickel : ℚ := 5 / 100

def target_value : ℚ := 3 / 2

def probability_at_least_target : ℚ := 316 / 6435

theorem coins_probability_theorem :
  let total_outcomes := Nat.choose total_coins coins_drawn
  let successful_outcomes := 
    Nat.choose num_quarters 3 * Nat.choose num_dimes 5 +
    Nat.choose num_quarters 2 * Nat.choose num_dimes 4 * Nat.choose num_nickels 2
  (successful_outcomes : ℚ) / total_outcomes = probability_at_least_target :=
sorry

end coins_probability_theorem_l131_13104


namespace girls_in_class_l131_13106

theorem girls_in_class (total : ℕ) (boy_ratio girl_ratio : ℕ) (h1 : total = 260) (h2 : boy_ratio = 5) (h3 : girl_ratio = 8) :
  (girl_ratio * total) / (boy_ratio + girl_ratio) = 160 := by
sorry

end girls_in_class_l131_13106


namespace min_value_theorem_min_value_is_four_l131_13148

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 3/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 3/b = 1 → x/2 + y/3 ≤ a/2 + b/3 :=
by sorry

theorem min_value_is_four (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 3/y = 1) :
  x/2 + y/3 = 4 :=
by sorry

end min_value_theorem_min_value_is_four_l131_13148


namespace quadratic_factorization_l131_13110

theorem quadratic_factorization (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 15 = (m * x + n) * (p * x + q)) →
  (∃ k : ℤ, b = 2 * k) ∧ 
  ¬(∀ k : ℤ, ∃ (m n p q : ℤ), 15 * x^2 + (2 * k) * x + 15 = (m * x + n) * (p * x + q)) :=
by sorry

end quadratic_factorization_l131_13110


namespace isosceles_triangle_side_length_l131_13138

/-- An isosceles triangle with base 8 and side difference 2 has sides of length 10 or 6 -/
theorem isosceles_triangle_side_length (AC BC : ℝ) : 
  BC = 8 → 
  |AC - BC| = 2 → 
  (AC = 10 ∨ AC = 6) :=
by sorry

end isosceles_triangle_side_length_l131_13138


namespace geometric_sequence_sum_l131_13121

/-- A sequence where each term is twice the previous term -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h1 : GeometricSequence a) 
  (h2 : a 1 + a 4 = 2) : 
  a 5 + a 8 = 32 := by
sorry

end geometric_sequence_sum_l131_13121


namespace tank_capacity_l131_13186

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 1/4 →
  final_fraction = 2/3 →
  added_amount = 180 →
  (final_fraction - initial_fraction) * (added_amount / (final_fraction - initial_fraction)) = 432 := by
  sorry

end tank_capacity_l131_13186


namespace store_refusal_illegal_l131_13145

/-- Represents a banknote --/
structure Banknote where
  issued_by_bank_of_russia : Bool
  has_tears : Bool

/-- Represents the store's action --/
inductive StoreAction
  | Accept
  | Refuse

/-- Determines if a banknote is legal tender --/
def is_legal_tender (note : Banknote) : Prop :=
  note.issued_by_bank_of_russia ∧ (note.has_tears ∨ ¬note.has_tears)

/-- Determines if the store's action is legal --/
def is_legal_action (note : Banknote) (action : StoreAction) : Prop :=
  is_legal_tender note → action = StoreAction.Accept

/-- The main theorem --/
theorem store_refusal_illegal 
  (lydia_note : Banknote)
  (h1 : lydia_note.has_tears)
  (h2 : lydia_note.issued_by_bank_of_russia)
  (h3 : ∀ (note : Banknote), note.has_tears → is_legal_tender note)
  (store_action : StoreAction)
  (h4 : store_action = StoreAction.Refuse) :
  ¬(is_legal_action lydia_note store_action) :=
by sorry

end store_refusal_illegal_l131_13145


namespace two_std_dev_below_mean_l131_13163

/-- For a normal distribution with mean 16.5 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 13.5. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (h_μ : μ = 16.5) (h_σ : σ = 1.5) :
  μ - 2 * σ = 13.5 := by
  sorry

end two_std_dev_below_mean_l131_13163


namespace molecular_weight_proof_l131_13141

/-- Given a compound where 3 moles have a molecular weight of 222,
    prove that the molecular weight of 1 mole is 74 g/mol. -/
theorem molecular_weight_proof (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 222)
  (h2 : num_moles = 3) :
  total_weight / num_moles = 74 := by
  sorry

end molecular_weight_proof_l131_13141


namespace gcd_digits_bound_l131_13115

theorem gcd_digits_bound (a b : ℕ) (ha : a < 100000) (hb : b < 100000)
  (hlcm : 10000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 100000000) :
  Nat.gcd a b < 1000 := by
  sorry

end gcd_digits_bound_l131_13115


namespace no_integer_roots_l131_13122

theorem no_integer_roots (a b : ℤ) : ¬ ∃ x : ℤ, x^2 + 3*a*x + 3*(2 - b^2) = 0 := by
  sorry

end no_integer_roots_l131_13122


namespace initial_workers_correct_l131_13101

/-- The number of initial workers required to complete a job -/
def initialWorkers : ℕ := 6

/-- The total amount of work for the job -/
def totalWork : ℕ := initialWorkers * 8

/-- Proves that the initial number of workers is correct given the problem conditions -/
theorem initial_workers_correct :
  totalWork = initialWorkers * 3 + (initialWorkers + 4) * 3 :=
by sorry

end initial_workers_correct_l131_13101


namespace second_grade_survey_size_l131_13174

/-- Represents a school with three grades and a stratified sampling plan. -/
structure School where
  total_students : ℕ
  grade_ratio : Fin 3 → ℕ
  survey_size : ℕ

/-- Calculates the number of students to be surveyed from a specific grade. -/
def students_surveyed_in_grade (school : School) (grade : Fin 3) : ℕ :=
  (school.survey_size * school.grade_ratio grade) / (school.grade_ratio 0 + school.grade_ratio 1 + school.grade_ratio 2)

/-- The main theorem stating that 50 second-grade students should be surveyed. -/
theorem second_grade_survey_size (school : School) 
  (h1 : school.total_students = 1500)
  (h2 : school.grade_ratio 0 = 4)
  (h3 : school.grade_ratio 1 = 5)
  (h4 : school.grade_ratio 2 = 6)
  (h5 : school.survey_size = 150) :
  students_surveyed_in_grade school 1 = 50 := by
  sorry


end second_grade_survey_size_l131_13174


namespace n_value_is_six_l131_13161

/-- The cost of a water bottle in cents -/
def water_cost : ℕ := 50

/-- The cost of a fruit in cents -/
def fruit_cost : ℕ := 25

/-- The cost of a snack in cents -/
def snack_cost : ℕ := 100

/-- The number of water bottles in a bundle -/
def water_in_bundle : ℕ := 1

/-- The number of snacks in a bundle -/
def snacks_in_bundle : ℕ := 3

/-- The number of fruits in a bundle -/
def fruits_in_bundle : ℕ := 2

/-- The regular selling price of a bundle in cents -/
def bundle_price : ℕ := 460

/-- The special price for every nth bundle in cents -/
def special_price : ℕ := 200

/-- The function to calculate the cost of a regular bundle in cents -/
def bundle_cost : ℕ := 
  water_cost * water_in_bundle + 
  snack_cost * snacks_in_bundle + 
  fruit_cost * fruits_in_bundle

/-- The function to calculate the profit from a regular bundle in cents -/
def bundle_profit : ℕ := bundle_price - bundle_cost

/-- The function to calculate the cost of a special bundle in cents -/
def special_bundle_cost : ℕ := bundle_cost + snack_cost

/-- The function to calculate the loss from a special bundle in cents -/
def special_bundle_loss : ℕ := special_bundle_cost - special_price

/-- Theorem stating that the value of n is 6 -/
theorem n_value_is_six : 
  ∃ n : ℕ, n > 0 ∧ (n - 1) * bundle_profit = special_bundle_loss ∧ n = 6 := by
  sorry

end n_value_is_six_l131_13161


namespace bono_jelly_beans_l131_13162

/-- Given the number of jelly beans for Alida, Bono, and Cate, prove that Bono has 4t - 1 jelly beans. -/
theorem bono_jelly_beans (t : ℕ) (A B C : ℕ) : 
  A + B = 6 * t + 3 →
  A + C = 4 * t + 5 →
  B + C = 6 * t →
  B = 4 * t - 1 := by
  sorry

end bono_jelly_beans_l131_13162


namespace largest_fraction_l131_13177

theorem largest_fraction (a b c d : ℝ) 
  (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d)
  (h5 : a = 2) (h6 : b = 3) (h7 : c = 5) (h8 : d = 8) :
  (c + d) / (a + b) = max ((a + b) / (c + d)) 
                         (max ((a + d) / (b + c)) 
                              (max ((b + c) / (a + d)) 
                                   ((b + d) / (a + c)))) := by
  sorry

end largest_fraction_l131_13177


namespace transaction_result_l131_13128

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  carValue : Int
  hascar : Bool

/-- Represents a car transaction between two people -/
def carTransaction (buyer seller : FinancialState) (price : Int) : FinancialState × FinancialState :=
  let newBuyer : FinancialState := {
    cash := buyer.cash - price,
    carValue := seller.carValue,
    hascar := true
  }
  let newSeller : FinancialState := {
    cash := seller.cash + price,
    carValue := 0,
    hascar := false
  }
  (newBuyer, newSeller)

/-- Calculates the net worth of a person -/
def netWorth (state : FinancialState) : Int :=
  state.cash + (if state.hascar then state.carValue else 0)

theorem transaction_result (initialCarValue : Int) :
  let mrAInitial : FinancialState := { cash := 8000, carValue := initialCarValue, hascar := true }
  let mrBInitial : FinancialState := { cash := 9000, carValue := 0, hascar := false }
  let (mrBAfterFirst, mrAAfterFirst) := carTransaction mrBInitial mrAInitial 10000
  let (mrAFinal, mrBFinal) := carTransaction mrAAfterFirst mrBAfterFirst 7000
  (netWorth mrAFinal - netWorth mrAInitial = 3000) ∧
  (netWorth mrBFinal - netWorth mrBInitial = -3000) :=
by
  sorry


end transaction_result_l131_13128


namespace lcm_gcf_ratio_l131_13185

theorem lcm_gcf_ratio : (Nat.lcm 240 630) / (Nat.gcd 240 630) = 168 := by
  sorry

end lcm_gcf_ratio_l131_13185


namespace set_equivalence_l131_13181

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem set_equivalence : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end set_equivalence_l131_13181


namespace bowling_ball_weight_l131_13151

theorem bowling_ball_weight (kayak_weight : ℝ) (ball_weight : ℝ) :
  kayak_weight = 36 →
  9 * ball_weight = 2 * kayak_weight →
  ball_weight = 8 := by
sorry

end bowling_ball_weight_l131_13151


namespace trapezoid_sides_l131_13158

/-- Proves that a trapezoid with given area, height, and difference between parallel sides has specific lengths for its parallel sides -/
theorem trapezoid_sides (area : ℝ) (height : ℝ) (side_diff : ℝ) 
  (h_area : area = 594) 
  (h_height : height = 22) 
  (h_side_diff : side_diff = 6) :
  ∃ (a b : ℝ), 
    (a + b) * height / 2 = area ∧ 
    a - b = side_diff ∧ 
    a = 30 ∧ 
    b = 24 := by
  sorry

end trapezoid_sides_l131_13158


namespace mothers_day_rose_ratio_l131_13173

/-- The number of roses Kyle picked last year -/
def last_year_roses : ℕ := 12

/-- The cost of one rose at the grocery store in dollars -/
def rose_cost : ℕ := 3

/-- The total amount Kyle spent on roses at the grocery store in dollars -/
def total_spent : ℕ := 54

/-- The ratio of roses in this year's bouquet to roses picked last year -/
def rose_ratio : Rat := 3 / 2

theorem mothers_day_rose_ratio :
  (total_spent / rose_cost : ℚ) / last_year_roses = rose_ratio :=
sorry

end mothers_day_rose_ratio_l131_13173


namespace smallest_number_l131_13108

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The given numbers in their respective bases --/
def number_A : List Nat := [0, 2]
def number_B : List Nat := [0, 3]
def number_C : List Nat := [3, 2]
def number_D : List Nat := [1, 3]

/-- The bases of the given numbers --/
def base_A : Nat := 7
def base_B : Nat := 5
def base_C : Nat := 6
def base_D : Nat := 4

theorem smallest_number :
  to_decimal number_D base_D < to_decimal number_A base_A ∧
  to_decimal number_D base_D < to_decimal number_B base_B ∧
  to_decimal number_D base_D < to_decimal number_C base_C :=
by sorry

end smallest_number_l131_13108


namespace no_integer_roots_l131_13168

theorem no_integer_roots : ∀ x : ℤ, x^2 + 2^2018 * x + 2^2019 ≠ 0 := by
  sorry

end no_integer_roots_l131_13168


namespace modulus_of_z_l131_13100

theorem modulus_of_z (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end modulus_of_z_l131_13100


namespace vector_parallel_problem_l131_13190

theorem vector_parallel_problem (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![-1, 1]
  let c : Fin 2 → ℝ := ![3, 0]
  (∃ (k : ℝ), a = k • (b + c)) → m = (1 : ℝ) / 2 := by
  sorry

end vector_parallel_problem_l131_13190


namespace propositions_truth_values_l131_13155

theorem propositions_truth_values :
  (∃ a b : ℝ, a + b < 2 * Real.sqrt (a * b)) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (1/x + 9/y = 1) ∧ (x + y < 16)) ∧
  (∀ x : ℝ, x^2 + 4/x^2 ≥ 4) ∧
  (∀ a b : ℝ, (a * b > 0) → (b/a + a/b ≥ 2)) :=
by sorry


end propositions_truth_values_l131_13155


namespace arkos_population_2070_l131_13189

def population_growth (initial_population : ℕ) (growth_factor : ℕ) (years : ℕ) : ℕ :=
  initial_population * growth_factor ^ (years / 10)

theorem arkos_population_2070 :
  let initial_population := 250
  let years := 50
  let growth_factor := 2
  population_growth initial_population growth_factor years = 8000 := by
sorry

end arkos_population_2070_l131_13189


namespace circle_and_line_properties_l131_13102

-- Define the circle C
def circle_C (x y b : ℝ) : Prop := (x + 2)^2 + (y - b)^2 = 3

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = x + m

-- Define the point that the circle passes through
def point_on_circle (b : ℝ) : Prop := circle_C (-2 + Real.sqrt 2) 0 b

-- Define the tangency condition
def is_tangent (m b : ℝ) : Prop :=
  (|(-2) - 1 + m| / Real.sqrt 2) = Real.sqrt 3

-- Define the perpendicular condition
def is_perpendicular (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_C x₁ y₁ 1 ∧ circle_C x₂ y₂ 1 ∧
    line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
    x₁ * x₂ + y₁ * y₂ = 0

theorem circle_and_line_properties :
  (∃ b : ℝ, b > 0 ∧ point_on_circle b) ∧
  (∃ m : ℝ, is_tangent m 1) ∧
  (∃ m : ℝ, is_perpendicular m) :=
sorry

end circle_and_line_properties_l131_13102
