import Mathlib

namespace grill_coal_consumption_l1991_199176

theorem grill_coal_consumption (total_time : ℕ) (bags : ℕ) (coals_per_bag : ℕ) 
  (h1 : total_time = 240)
  (h2 : bags = 3)
  (h3 : coals_per_bag = 60) :
  (bags * coals_per_bag) / (total_time / 20) = 15 := by
  sorry

end grill_coal_consumption_l1991_199176


namespace valid_numbers_l1991_199168

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a b : ℕ) (q : ℚ),
    N = 10 * a + b ∧
    0 < a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    b = a * q ∧
    N = 3 * (a * q^2)

theorem valid_numbers :
  {N : ℕ | is_valid_number N} = {12, 24, 36, 48} :=
sorry

end valid_numbers_l1991_199168


namespace table_tennis_outcomes_count_l1991_199135

/-- Represents the number of possible outcomes in a table tennis match -/
def table_tennis_outcomes : ℕ := 30

/-- The winning condition for the match -/
def winning_games : ℕ := 3

/-- Theorem stating that the number of possible outcomes in a table tennis match
    where the first to win 3 games wins the match is 30 -/
theorem table_tennis_outcomes_count :
  (∀ (match_length : ℕ), match_length ≥ winning_games →
    (∃ (winner_games loser_games : ℕ),
      winner_games = winning_games ∧
      winner_games + loser_games = match_length ∧
      winner_games > loser_games)) →
  table_tennis_outcomes = 30 := by
  sorry

end table_tennis_outcomes_count_l1991_199135


namespace trajectory_and_intersection_l1991_199196

/-- The trajectory of point G -/
def trajectory (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- The condition that the product of slopes of GE and FG is -4 -/
def slope_condition (x y : ℝ) : Prop := 
  y ≠ 0 → (y / (x - 1)) * (y / (x + 1)) = -4

/-- The line passing through (0, -1) with slope k -/
def line (k x : ℝ) : ℝ := k * x - 1

/-- The x-coordinates of the intersection points sum to 8 -/
def intersection_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    trajectory x₁ (line k x₁) ∧ 
    trajectory x₂ (line k x₂) ∧ 
    x₁ + x₂ = 8

theorem trajectory_and_intersection :
  (∀ x y : ℝ, slope_condition x y → trajectory x y) ∧
  (∀ k : ℝ, intersection_condition k → k = 2) :=
sorry

end trajectory_and_intersection_l1991_199196


namespace largest_non_sum_of_composite_odd_l1991_199122

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = n

/-- A function that checks if a number is odd -/
def isOdd (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2 * k + 1

/-- A function that checks if a number can be expressed as the sum of two composite odd positive integers -/
def isSumOfTwoCompositeOdd (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ isOdd a ∧ isOdd b ∧ isComposite a ∧ isComposite b ∧ a + b = n

/-- The main theorem stating that 38 is the largest even positive integer that cannot be expressed as the sum of two composite odd positive integers -/
theorem largest_non_sum_of_composite_odd :
  (∀ (n : ℕ), n > 38 → n % 2 = 0 → isSumOfTwoCompositeOdd n) ∧
  ¬(isSumOfTwoCompositeOdd 38) :=
sorry

end largest_non_sum_of_composite_odd_l1991_199122


namespace A_value_l1991_199156

theorem A_value (a : ℝ) (h : a * (a + 2) = 8 ∨ a^2 + a = 8 - a) :
  2 / (a^2 - 4) - 1 / (a * (a - 2)) = 1 / 8 :=
by sorry

end A_value_l1991_199156


namespace regression_line_properties_l1991_199148

-- Define random variables x and y
variable (x y : ℝ → ℝ)

-- Define that x and y are correlated
variable (h_correlated : Correlated x y)

-- Define the mean of x and y
def x_mean : ℝ := sorry
def y_mean : ℝ := sorry

-- Define the regression line
def regression_line (x y : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define the slope and intercept of the regression line
def a : ℝ := 0.2
def b : ℝ := 12

theorem regression_line_properties :
  (∀ t : ℝ, regression_line x y t = a * t + b) →
  (regression_line x y x_mean = y_mean) ∧
  (∀ δ : ℝ, regression_line x y (x_mean + δ) - regression_line x y x_mean = a * δ) :=
sorry

end regression_line_properties_l1991_199148


namespace triangle_problem_l1991_199118

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < Real.pi) ∧
  (0 < B) ∧ (B < Real.pi) ∧
  (0 < C) ∧ (C < Real.pi) ∧
  (A + B + C = Real.pi) ∧
  (Real.sin A)^2 - (Real.sin B)^2 - (Real.sin C)^2 = Real.sin B * Real.sin C ∧
  a = 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  A = 2 * Real.pi / 3 ∧
  (a + b + c) ≤ 3 + 2 * Real.sqrt 3 :=
by sorry

end triangle_problem_l1991_199118


namespace reflection_of_point_across_x_axis_l1991_199188

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Theorem: The reflection of the point P(-3,1) across the x-axis is (-3,-1) -/
theorem reflection_of_point_across_x_axis :
  let P : Point2D := { x := -3, y := 1 }
  reflectAcrossXAxis P = { x := -3, y := -1 } := by
  sorry

end reflection_of_point_across_x_axis_l1991_199188


namespace max_term_is_9_8_l1991_199124

/-- The sequence defined by n^2 / 2^n for n ≥ 1 -/
def a (n : ℕ) : ℚ := (n^2 : ℚ) / 2^n

/-- The maximum term of the sequence occurs at n = 3 -/
def max_term_index : ℕ := 3

/-- The maximum value of the sequence -/
def max_term_value : ℚ := 9/8

/-- Theorem stating that the maximum term of the sequence a(n) is 9/8 -/
theorem max_term_is_9_8 :
  (∀ n : ℕ, n ≥ 1 → a n ≤ max_term_value) ∧ 
  (∃ n : ℕ, n ≥ 1 ∧ a n = max_term_value) :=
sorry

end max_term_is_9_8_l1991_199124


namespace cubic_tangent_perpendicular_l1991_199131

/-- Given a cubic function f(x) = ax³ + x + 1, if its tangent line at x = 1 is
    perpendicular to the line x + 4y = 0, then a = 1. -/
theorem cubic_tangent_perpendicular (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + x + 1
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 1
  (f' 1) * (-1/4) = -1 →
  a = 1 := by
sorry

end cubic_tangent_perpendicular_l1991_199131


namespace jaces_remaining_money_l1991_199195

/-- Proves that Jace's remaining money after transactions is correct -/
theorem jaces_remaining_money
  (earnings : ℚ)
  (debt : ℚ)
  (neighbor_percentage : ℚ)
  (exchange_rate : ℚ)
  (h1 : earnings = 1500)
  (h2 : debt = 358)
  (h3 : neighbor_percentage = 1/4)
  (h4 : exchange_rate = 121/100) :
  earnings - debt - (earnings - debt) * neighbor_percentage = 8565/10 :=
by sorry


end jaces_remaining_money_l1991_199195


namespace minimum_garden_width_minimum_garden_width_is_ten_l1991_199155

theorem minimum_garden_width (w : ℝ) : w > 0 → w * (w + 10) ≥ 150 → w ≥ 10 := by
  sorry

theorem minimum_garden_width_is_ten : ∃ w : ℝ, w > 0 ∧ w * (w + 10) ≥ 150 ∧ ∀ x : ℝ, x > 0 → x * (x + 10) ≥ 150 → x ≥ w := by
  sorry

end minimum_garden_width_minimum_garden_width_is_ten_l1991_199155


namespace square_side_length_l1991_199191

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1/4 → side^2 = area → side = 1/2 := by
sorry

end square_side_length_l1991_199191


namespace initial_population_theorem_l1991_199178

def village_population (P : ℕ) : Prop :=
  ⌊(P : ℝ) * 0.95 * 0.80⌋ = 3553

theorem initial_population_theorem :
  ∃ P : ℕ, village_population P ∧ P ≥ 4678 ∧ P < 4679 :=
sorry

end initial_population_theorem_l1991_199178


namespace runner_distance_at_click_l1991_199174

/-- The time in seconds for which the camera timer is set -/
def timer_setting : ℝ := 45

/-- The runner's speed in yards per second -/
def runner_speed : ℝ := 10

/-- The speed of sound in feet per second without headwind -/
def sound_speed : ℝ := 1100

/-- The reduction factor of sound speed due to headwind -/
def sound_speed_reduction : ℝ := 0.1

/-- The effective speed of sound in feet per second with headwind -/
def effective_sound_speed : ℝ := sound_speed * (1 - sound_speed_reduction)

/-- The distance the runner travels in feet at time t -/
def runner_distance (t : ℝ) : ℝ := runner_speed * 3 * t

/-- The distance sound travels in feet at time t after the camera click -/
def sound_distance (t : ℝ) : ℝ := effective_sound_speed * (t - timer_setting)

/-- The time when the runner hears the camera click -/
noncomputable def hearing_time : ℝ := 
  (effective_sound_speed * timer_setting) / (effective_sound_speed - runner_speed * 3)

theorem runner_distance_at_click : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |runner_distance hearing_time / 3 - 464| < ε :=
sorry

end runner_distance_at_click_l1991_199174


namespace quadratic_inequality_l1991_199181

theorem quadratic_inequality (x : ℝ) : 3 * x^2 - 4 * x - 4 < 0 ↔ -2/3 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_l1991_199181


namespace binomial_difference_divisibility_l1991_199167

theorem binomial_difference_divisibility (k : ℕ) (h : k ≥ 2) :
  ∃ n : ℕ, (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) = n * 2^(3*k) ∧
           (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) % 2^(3*k+1) ≠ 0 := by
  sorry

end binomial_difference_divisibility_l1991_199167


namespace sum_of_squares_representation_l1991_199120

theorem sum_of_squares_representation (n : ℕ) :
  ∃ (m : ℤ), (∃ (representations : Finset (ℤ × ℤ)), 
    (∀ (pair : ℤ × ℤ), pair ∈ representations → m = pair.1^2 + pair.2^2) ∧
    representations.card ≥ n) ∧ 
  m = 5^(2*n) :=
by sorry

end sum_of_squares_representation_l1991_199120


namespace bills_final_money_bills_final_money_is_3180_l1991_199112

/-- Calculates Bill's final amount of money after Frank and Bill's pizza purchase --/
theorem bills_final_money (initial_money : ℝ) (pizza_cost : ℝ) (num_pizzas : ℕ) 
  (topping1_cost : ℝ) (topping2_cost : ℝ) (discount_rate : ℝ) (bills_initial_money : ℝ) : ℝ :=
  let total_pizza_cost := pizza_cost * num_pizzas
  let total_topping_cost := (topping1_cost + topping2_cost) * num_pizzas
  let total_cost_before_discount := total_pizza_cost + total_topping_cost
  let discount := discount_rate * total_pizza_cost
  let final_cost := total_cost_before_discount - discount
  let remaining_money := initial_money - final_cost
  bills_initial_money + remaining_money

/-- Proves that Bill's final amount of money is $31.80 --/
theorem bills_final_money_is_3180 : 
  bills_final_money 42 11 3 1.5 2 0.1 30 = 31.80 := by
  sorry

end bills_final_money_bills_final_money_is_3180_l1991_199112


namespace intersection_of_three_lines_l1991_199117

/-- Given three lines that intersect at a single point, prove the value of k -/
theorem intersection_of_three_lines (k : ℚ) : 
  (∃! p : ℚ × ℚ, 
    (p.2 = 4 * p.1 + 2) ∧ 
    (p.2 = -2 * p.1 - 8) ∧ 
    (p.2 = 2 * p.1 + k)) → 
  k = -4/3 := by
sorry

end intersection_of_three_lines_l1991_199117


namespace rowing_time_ratio_l1991_199177

theorem rowing_time_ratio (man_speed : ℝ) (current_speed : ℝ) 
  (h1 : man_speed = 3.3) (h2 : current_speed = 1.1) :
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  sorry

end rowing_time_ratio_l1991_199177


namespace intersection_of_A_and_B_l1991_199138

def A : Set ℝ := {x | (1 : ℝ) / (x - 1) ≤ 1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 2} := by sorry

end intersection_of_A_and_B_l1991_199138


namespace function_identity_l1991_199121

def f_condition (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2

theorem function_identity (f : ℝ → ℝ) (h : f_condition f) :
  ∀ x : ℝ, f x = x + 1 :=
by sorry

end function_identity_l1991_199121


namespace remainder_8437_div_9_l1991_199164

theorem remainder_8437_div_9 : 8437 % 9 = 4 := by
  sorry

end remainder_8437_div_9_l1991_199164


namespace smartphone_savings_proof_l1991_199147

/-- Calculates the required weekly savings to reach a target amount. -/
def weekly_savings (smartphone_cost : ℚ) (current_savings : ℚ) (saving_weeks : ℕ) : ℚ :=
  (smartphone_cost - current_savings) / saving_weeks

/-- Proves that the weekly savings required to buy a $160 smartphone
    with $40 current savings over 8 weeks is $15. -/
theorem smartphone_savings_proof :
  let smartphone_cost : ℚ := 160
  let current_savings : ℚ := 40
  let saving_weeks : ℕ := 8
  weekly_savings smartphone_cost current_savings saving_weeks = 15 := by
sorry

end smartphone_savings_proof_l1991_199147


namespace expected_adjacent_red_pairs_l1991_199142

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of decks used -/
def num_decks : ℕ := 2

/-- The total number of cards in the combined deck -/
def total_cards : ℕ := standard_deck_size * num_decks

/-- The number of red cards in the combined deck -/
def red_cards : ℕ := standard_deck_size

/-- The expected number of pairs of adjacent red cards -/
def expected_red_pairs : ℚ := 2652 / 103

theorem expected_adjacent_red_pairs :
  let p := red_cards / total_cards
  expected_red_pairs = red_cards * (red_cards - 1) / (total_cards - 1) := by
  sorry

end expected_adjacent_red_pairs_l1991_199142


namespace first_player_wins_l1991_199105

/-- Represents a stick with a certain length -/
structure Stick :=
  (length : ℝ)

/-- Represents the state of the game -/
structure GameState :=
  (sticks : List Stick)

/-- Represents a player's move, breaking a stick into two parts -/
def breakStick (s : Stick) : Stick × Stick :=
  sorry

/-- Checks if three sticks can form a triangle -/
def canFormTriangle (s1 s2 s3 : Stick) : Prop :=
  sorry

/-- Represents a player's strategy -/
def Strategy := GameState → Option (Stick × (Stick × Stick))

/-- The first player's strategy -/
def firstPlayerStrategy : Strategy :=
  sorry

/-- The second player's strategy -/
def secondPlayerStrategy : Strategy :=
  sorry

/-- Simulates the game for three moves -/
def gameSimulation (s1 : Strategy) (s2 : Strategy) : GameState :=
  sorry

/-- Checks if the given game state allows forming two triangles -/
def canFormTwoTriangles (gs : GameState) : Prop :=
  sorry

/-- The main theorem stating that the first player can guarantee a win -/
theorem first_player_wins :
  ∀ (s2 : Strategy),
  ∃ (s1 : Strategy),
  canFormTwoTriangles (gameSimulation s1 s2) :=
sorry

end first_player_wins_l1991_199105


namespace money_left_calculation_l1991_199187

/-- The amount of money John has left after buying pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let medium_pizza_cost := 3 * q
  let large_pizza_cost := 4 * q
  let total_spent := 4 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  50 - total_spent

/-- Theorem stating that the money left is 50 - 15q -/
theorem money_left_calculation (q : ℝ) : money_left q = 50 - 15 * q := by
  sorry

end money_left_calculation_l1991_199187


namespace sum_of_root_pairs_is_124_l1991_199139

def root_pairs : List (Nat × Nat) := [(1, 48), (2, 24), (3, 16), (4, 12), (6, 8)]

def sum_of_pairs (pairs : List (Nat × Nat)) : Nat :=
  pairs.map (fun (a, b) => a + b) |> List.sum

theorem sum_of_root_pairs_is_124 : sum_of_pairs root_pairs = 124 := by
  sorry

end sum_of_root_pairs_is_124_l1991_199139


namespace collinear_points_sum_l1991_199130

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p₁ p₂ p₃ : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), p₃ = (1 - t₁ - t₂) • p₁ + t₁ • p₂ + t₂ • p₃

/-- If the points (2, a, b), (a, 3, b), and (a, b, 4) are collinear, then a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

end collinear_points_sum_l1991_199130


namespace area_between_circles_and_xaxis_l1991_199127

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_xaxis :
  let c1_center : ℝ × ℝ := (3, 5)
  let c2_center : ℝ × ℝ := (9, 5)
  let radius : ℝ := 3
  let rectangle_area : ℝ := (c2_center.1 - c1_center.1) * radius
  let sector_area : ℝ := (1/4) * π * radius^2
  rectangle_area - 2 * sector_area = 18 - (9/2) * π := by sorry

end area_between_circles_and_xaxis_l1991_199127


namespace box_of_balls_l1991_199186

theorem box_of_balls (x : ℕ) : 
  (25 - x = 30 - 25) → x = 20 := by sorry

end box_of_balls_l1991_199186


namespace all_transformed_points_in_S_l1991_199179

def S : Set ℂ := {z | -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

theorem all_transformed_points_in_S : ∀ z ∈ S, (1/2 + 1/2*I)*z ∈ S := by
  sorry

end all_transformed_points_in_S_l1991_199179


namespace problem_proof_l1991_199113

theorem problem_proof (x v : ℝ) (hx : x = 2) (hv : v = 3 * x) :
  (2 * v - 5) - (2 * x - 5) = 8 := by
  sorry

end problem_proof_l1991_199113


namespace regular_polygon_angle_relation_l1991_199154

theorem regular_polygon_angle_relation (m : ℕ) : m ≥ 3 →
  (120 : ℝ) = 4 * (360 / m) → m = 12 := by sorry

end regular_polygon_angle_relation_l1991_199154


namespace sweet_potatoes_theorem_l1991_199152

def sweet_potatoes_problem (total_harvested sold_to_adams sold_to_lenon : ℕ) : Prop :=
  total_harvested - (sold_to_adams + sold_to_lenon) = 45

theorem sweet_potatoes_theorem :
  sweet_potatoes_problem 80 20 15 := by
  sorry

end sweet_potatoes_theorem_l1991_199152


namespace digit_sum_equation_l1991_199197

/-- Given that a000 + a998 + a999 = 22997, prove that a = 7 -/
theorem digit_sum_equation (a : ℕ) : 
  a * 1000 + a * 998 + a * 999 = 22997 → a = 7 := by
  sorry

end digit_sum_equation_l1991_199197


namespace profit_percent_calculation_l1991_199109

theorem profit_percent_calculation (selling_price cost_price : ℝ) :
  cost_price = 0.25 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = 300 := by
  sorry

end profit_percent_calculation_l1991_199109


namespace parallelogram_may_not_have_symmetry_l1991_199144

-- Define the basic geometric shapes
inductive GeometricShape
  | LineSegment
  | Rectangle
  | Angle
  | Parallelogram

-- Define a property for having an axis of symmetry
def has_axis_of_symmetry (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.LineSegment => True
  | GeometricShape.Rectangle => True
  | GeometricShape.Angle => True
  | GeometricShape.Parallelogram => sorry  -- This can be True or False

-- Theorem: Only parallelograms may not have an axis of symmetry
theorem parallelogram_may_not_have_symmetry :
  ∀ (shape : GeometricShape),
    ¬(has_axis_of_symmetry shape) → shape = GeometricShape.Parallelogram :=
by sorry

-- Note: The actual proof is omitted and replaced with 'sorry'

end parallelogram_may_not_have_symmetry_l1991_199144


namespace problem_solution_l1991_199108

theorem problem_solution (a b : ℝ) : (a + b)^2 + Real.sqrt (2 * b - 4) = 0 → a = -2 := by
  sorry

end problem_solution_l1991_199108


namespace college_student_count_l1991_199141

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The total number of students in the college -/
def College.total (c : College) : ℕ := c.boys + c.girls

/-- Theorem: In a college where the ratio of boys to girls is 8:5 and there are 300 girls, 
    the total number of students is 780 -/
theorem college_student_count : 
  ∀ (c : College), 
  c.boys * 5 = c.girls * 8 → 
  c.girls = 300 → 
  c.total = 780 := by
sorry

end college_student_count_l1991_199141


namespace factorial_equation_solutions_l1991_199169

theorem factorial_equation_solutions :
  ∀ x y z : ℕ+, 2^x.val + 3^y.val - 7 = Nat.factorial z.val →
    ((x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4)) :=
by sorry

end factorial_equation_solutions_l1991_199169


namespace john_savings_period_l1991_199165

/-- Calculates the number of years saved given monthly savings, recent expense, and remaining balance -/
def years_saved (monthly_saving : ℕ) (recent_expense : ℕ) (remaining_balance : ℕ) : ℚ :=
  (recent_expense + remaining_balance) / (monthly_saving * 12)

theorem john_savings_period :
  let monthly_saving : ℕ := 25
  let recent_expense : ℕ := 400
  let remaining_balance : ℕ := 200
  years_saved monthly_saving recent_expense remaining_balance = 2 := by sorry

end john_savings_period_l1991_199165


namespace volume_of_extended_parallelepiped_with_caps_l1991_199128

/-- The volume of a set of points that are inside or within one unit of a rectangular parallelepiped
    with semi-spherical caps on the longest side vertices. -/
theorem volume_of_extended_parallelepiped_with_caps : ℝ := by
  -- Define the dimensions of the parallelepiped
  let length : ℝ := 6
  let width : ℝ := 3
  let height : ℝ := 2

  -- Define the radius of the semi-spherical caps
  let cap_radius : ℝ := 1

  -- Define the number of semi-spherical caps
  let num_caps : ℕ := 4

  -- Calculate the volume
  have volume : ℝ := (324 + 8 * Real.pi) / 3

  sorry

#check volume_of_extended_parallelepiped_with_caps

end volume_of_extended_parallelepiped_with_caps_l1991_199128


namespace initial_clean_and_jerk_was_80kg_l1991_199159

/-- Represents John's weightlifting progress --/
structure Weightlifting where
  initial_snatch : ℝ
  initial_clean_and_jerk : ℝ
  new_combined_total : ℝ

/-- Calculates the new Snatch weight after an 80% increase --/
def new_snatch (w : Weightlifting) : ℝ :=
  w.initial_snatch * 1.8

/-- Calculates the new Clean & Jerk weight after doubling --/
def new_clean_and_jerk (w : Weightlifting) : ℝ :=
  w.initial_clean_and_jerk * 2

/-- Theorem stating that John's initial Clean & Jerk weight was 80 kg --/
theorem initial_clean_and_jerk_was_80kg (w : Weightlifting) 
  (h1 : w.initial_snatch = 50)
  (h2 : new_snatch w + new_clean_and_jerk w = w.new_combined_total)
  (h3 : w.new_combined_total = 250) : 
  w.initial_clean_and_jerk = 80 := by
  sorry


end initial_clean_and_jerk_was_80kg_l1991_199159


namespace binomial_prob_half_l1991_199189

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_prob_half (ξ : BinomialRV) 
  (h_exp : expected_value ξ = 2)
  (h_var : variance ξ = 1) : 
  ξ.p = 0.5 := by
  sorry

end binomial_prob_half_l1991_199189


namespace parabola_intersection_length_l1991_199125

/-- Parabola defined by y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line with slope √3 passing through a point -/
def line_with_slope_sqrt3 (x y x0 y0 : ℝ) : Prop :=
  y - y0 = Real.sqrt 3 * (x - x0)

/-- Focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Definition of the length of a line segment on the parabola -/
def segment_length (x1 x2 : ℝ) : ℝ := x1 + x2 + 2

theorem parabola_intersection_length :
  ∀ A B : ℝ × ℝ,
  (∃ x1 x2 y1 y2 : ℝ,
    A = (x1, y1) ∧ B = (x2, y2) ∧
    parabola x1 y1 ∧ parabola x2 y2 ∧
    line_with_slope_sqrt3 x1 y1 focus.1 focus.2 ∧
    line_with_slope_sqrt3 x2 y2 focus.1 focus.2) →
  segment_length A.1 B.1 = 16/3 := by
  sorry

end parabola_intersection_length_l1991_199125


namespace min_pumps_for_given_reservoir_l1991_199192

/-- Represents the characteristics of a reservoir with a leakage problem -/
structure Reservoir where
  single_pump_time : ℝ
  double_pump_time : ℝ
  target_time : ℝ

/-- Calculates the minimum number of pumps needed to fill the reservoir within the target time -/
def min_pumps_needed (r : Reservoir) : ℕ :=
  sorry

/-- Theorem stating that for the given reservoir conditions, at least 3 pumps are needed -/
theorem min_pumps_for_given_reservoir :
  let r : Reservoir := {
    single_pump_time := 8,
    double_pump_time := 3.2,
    target_time := 2
  }
  min_pumps_needed r = 3 := by
  sorry

end min_pumps_for_given_reservoir_l1991_199192


namespace greatest_integer_solution_l1991_199104

theorem greatest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, 8 - 3 * (2 * y + 1) > 26 → y ≤ x) ∧ (8 - 3 * (2 * x + 1) > 26) ↔ x = -4 := by
  sorry

end greatest_integer_solution_l1991_199104


namespace valid_tiling_conditions_l1991_199136

/-- Represents a tile shape -/
inductive TileShape
  | L  -- L-shaped tile covering 3 squares
  | T  -- T-shaped tile covering 4 squares

/-- Represents a valid tiling of an n×n board -/
def ValidTiling (n : ℕ) : Prop :=
  ∃ (tiling : List (TileShape × ℕ × ℕ)), 
    (∀ (t : TileShape) (x y : ℕ), (t, x, y) ∈ tiling → x < n ∧ y < n) ∧ 
    (∀ (x y : ℕ), x < n ∧ y < n → ∃! (t : TileShape), (t, x, y) ∈ tiling)

/-- The main theorem stating the conditions for a valid tiling -/
theorem valid_tiling_conditions (n : ℕ) : 
  ValidTiling n ↔ (n % 4 = 0 ∧ n > 4) :=
sorry

end valid_tiling_conditions_l1991_199136


namespace debora_has_twelve_more_dresses_l1991_199160

/-- The number of dresses each person has -/
structure Dresses where
  emily : ℕ
  melissa : ℕ
  debora : ℕ

/-- The conditions of the problem -/
def problem_conditions (d : Dresses) : Prop :=
  d.emily = 16 ∧
  d.melissa = d.emily / 2 ∧
  d.debora > d.melissa ∧
  d.emily + d.melissa + d.debora = 44

/-- The theorem to prove -/
theorem debora_has_twelve_more_dresses (d : Dresses) 
  (h : problem_conditions d) : d.debora = d.melissa + 12 := by
  sorry

#check debora_has_twelve_more_dresses

end debora_has_twelve_more_dresses_l1991_199160


namespace probability_letter_in_mathematical_l1991_199180

def alphabet : Finset Char := sorry

def mathematical : String := "MATHEMATICAL"

theorem probability_letter_in_mathematical :
  let unique_letters := mathematical.toList.toFinset
  (unique_letters.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 := by sorry

end probability_letter_in_mathematical_l1991_199180


namespace lizzy_initial_money_l1991_199129

def loan_amount : ℝ := 15
def interest_rate : ℝ := 0.20
def final_amount : ℝ := 33

theorem lizzy_initial_money :
  ∃ (initial_money : ℝ),
    initial_money = loan_amount ∧
    final_amount = initial_money + loan_amount + (interest_rate * loan_amount) :=
by sorry

end lizzy_initial_money_l1991_199129


namespace uncolored_area_rectangle_with_circles_l1991_199198

/-- The uncolored area of a rectangle with four tangent circles --/
theorem uncolored_area_rectangle_with_circles (w h r : Real) 
  (hw : w = 30) 
  (hh : h = 50) 
  (hr : r = w / 4) 
  (circles_fit : 4 * r = w) 
  (circles_tangent : 2 * r = h / 2) : 
  w * h - 4 * Real.pi * r^2 = 1500 - 225 * Real.pi := by
  sorry

end uncolored_area_rectangle_with_circles_l1991_199198


namespace product_quotient_puzzle_l1991_199111

theorem product_quotient_puzzle :
  ∃ (x y t : ℕ+),
    100 ≤ (x * y : ℕ) ∧ (x * y : ℕ) ≤ 999 ∧
    x * y = t^3 ∧
    (x : ℚ) / y = t^2 ∧
    x = 243 ∧ y = 3 :=
by sorry

end product_quotient_puzzle_l1991_199111


namespace alberts_number_l1991_199199

theorem alberts_number (a b c : ℚ) 
  (h1 : a = 2 * b + 1)
  (h2 : b = 2 * c + 1)
  (h3 : c = 2 * a + 2) :
  a = -11/7 := by
sorry

end alberts_number_l1991_199199


namespace marias_trip_distance_l1991_199115

/-- The total distance of Maria's trip -/
def total_distance : ℝ := 480

/-- Theorem stating that the total distance of Maria's trip is 480 miles -/
theorem marias_trip_distance :
  ∃ (D : ℝ),
    D / 2 + (D / 2) / 4 + 180 = D ∧
    D = total_distance :=
by sorry

end marias_trip_distance_l1991_199115


namespace purely_imaginary_iff_x_eq_one_l1991_199123

theorem purely_imaginary_iff_x_eq_one (x : ℝ) : 
  let z : ℂ := (x^2 - 1) + (x + 1)*I
  (∃ y : ℝ, z = y*I) ↔ x = 1 := by sorry

end purely_imaginary_iff_x_eq_one_l1991_199123


namespace angle_conversions_correct_l1991_199106

theorem angle_conversions_correct :
  let deg_to_rad (d : ℝ) := d * (π / 180)
  let rad_to_deg (r : ℝ) := r * (180 / π)
  (deg_to_rad 60 = π / 3) ∧
  (rad_to_deg (-10 * π / 3) = -600) ∧
  (deg_to_rad (-150) = -5 * π / 6) ∧
  (rad_to_deg (π / 12) = 15) := by
  sorry

end angle_conversions_correct_l1991_199106


namespace combination_properties_l1991_199126

theorem combination_properties (n m : ℕ+) (h : n > m) :
  (Nat.choose n m = Nat.choose n (n - m)) ∧
  (Nat.choose n m + Nat.choose n (m - 1) = Nat.choose (n + 1) m) := by
  sorry

end combination_properties_l1991_199126


namespace sequence_sum_l1991_199153

theorem sequence_sum (a b c d : ℕ+) : 
  (∃ r : ℚ, b = a * r ∧ c = a * r^2) →  -- geometric progression
  (d = a + 40) →                        -- arithmetic progression and difference
  a + b + c + d = 110 := by
sorry

end sequence_sum_l1991_199153


namespace intersection_S_T_l1991_199190

-- Define the sets S and T
def S : Set ℝ := {x | x < -5 ∨ x > 5}
def T : Set ℝ := {x | -7 < x ∧ x < 3}

-- State the theorem
theorem intersection_S_T : S ∩ T = {x | -7 < x ∧ x < -5} := by sorry

end intersection_S_T_l1991_199190


namespace expected_wins_equal_l1991_199103

/-- The total number of balls in the lottery box -/
def total_balls : ℕ := 8

/-- The number of red balls in the lottery box -/
def red_balls : ℕ := 4

/-- The number of black balls in the lottery box -/
def black_balls : ℕ := 4

/-- The number of draws made -/
def num_draws : ℕ := 2

/-- Represents the outcome of a single lottery draw -/
inductive DrawResult
| Red
| Black

/-- Represents the result of two draws -/
inductive TwoDrawResult
| Win  -- Two balls of the same color
| Lose -- Two balls of different colors

/-- The probability of winning in a single draw with replacement -/
def prob_win_with_replacement : ℚ :=
  (red_balls.choose 2 + black_balls.choose 2) / total_balls.choose 2

/-- The expected number of wins with replacement -/
def expected_wins_with_replacement : ℚ :=
  num_draws * prob_win_with_replacement

/-- The probability of winning in a single draw without replacement -/
def prob_win_without_replacement : ℚ :=
  (red_balls.choose 2 + black_balls.choose 2) / total_balls.choose 2

/-- The expected number of wins without replacement -/
def expected_wins_without_replacement : ℚ :=
  (0 * (12 / 35) + 1 * (16 / 35) + 2 * (7 / 35))

/-- Theorem stating that the expected number of wins is 6/7 for both cases -/
theorem expected_wins_equal :
  expected_wins_with_replacement = 6/7 ∧
  expected_wins_without_replacement = 6/7 := by
  sorry


end expected_wins_equal_l1991_199103


namespace square_difference_equality_l1991_199102

theorem square_difference_equality : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by
  sorry

end square_difference_equality_l1991_199102


namespace contracting_schemes_l1991_199150

def number_of_projects : ℕ := 6
def projects_for_A : ℕ := 3
def projects_for_B : ℕ := 2
def projects_for_C : ℕ := 1

theorem contracting_schemes :
  (number_of_projects.choose projects_for_A) *
  ((number_of_projects - projects_for_A).choose projects_for_B) *
  ((number_of_projects - projects_for_A - projects_for_B).choose projects_for_C) = 60 := by
  sorry

end contracting_schemes_l1991_199150


namespace sum_a_d_l1991_199173

theorem sum_a_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 42)
  (h2 : b + c = 6) : 
  a + d = 7 := by sorry

end sum_a_d_l1991_199173


namespace triangle_properties_l1991_199193

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 - t.c^2 = t.b^2 - (8 * t.b * t.c) / 5)
  (h2 : t.a = 6)
  (h3 : Real.sin t.B = 4/5) :
  (Real.sin t.A = 3/5) ∧ 
  ((1/2 * t.b * t.c * Real.sin t.A = 24) ∨ 
   (1/2 * t.b * t.c * Real.sin t.A = 168/25)) := by
  sorry


end triangle_properties_l1991_199193


namespace prism_18_edges_8_faces_l1991_199161

/-- A prism is a polyhedron with two congruent parallel faces (bases) and other faces (lateral faces) that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  2 + (p.edges / 3)

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ p : Prism, p.edges = 18 → num_faces p = 8 := by
  sorry


end prism_18_edges_8_faces_l1991_199161


namespace janet_total_distance_l1991_199132

/-- Represents Janet's training schedule for a week --/
structure WeekSchedule where
  running_days : Nat
  running_miles : Nat
  cycling_days : Nat
  cycling_miles : Nat
  swimming_days : Nat
  swimming_miles : Nat
  hiking_days : Nat
  hiking_miles : Nat

/-- Calculates the total distance for a given week schedule --/
def weekTotalDistance (schedule : WeekSchedule) : Nat :=
  schedule.running_days * schedule.running_miles +
  schedule.cycling_days * schedule.cycling_miles +
  schedule.swimming_days * schedule.swimming_miles +
  schedule.hiking_days * schedule.hiking_miles

/-- Janet's training schedule for three weeks --/
def janetSchedule : List WeekSchedule := [
  { running_days := 5, running_miles := 8, cycling_days := 3, cycling_miles := 7, swimming_days := 0, swimming_miles := 0, hiking_days := 0, hiking_miles := 0 },
  { running_days := 4, running_miles := 10, cycling_days := 0, cycling_miles := 0, swimming_days := 2, swimming_miles := 2, hiking_days := 0, hiking_miles := 0 },
  { running_days := 5, running_miles := 6, cycling_days := 0, cycling_miles := 0, swimming_days := 0, swimming_miles := 0, hiking_days := 2, hiking_miles := 3 }
]

/-- Theorem: Janet's total training distance is 141 miles --/
theorem janet_total_distance :
  (janetSchedule.map weekTotalDistance).sum = 141 := by
  sorry

end janet_total_distance_l1991_199132


namespace travel_cost_for_twenty_days_l1991_199151

/-- Calculate the total travel cost for a given number of working days and one-way trip cost. -/
def totalTravelCost (workingDays : ℕ) (oneWayCost : ℕ) : ℕ :=
  workingDays * (2 * oneWayCost)

/-- Theorem: The total travel cost for 20 working days with a one-way cost of $24 is $960. -/
theorem travel_cost_for_twenty_days :
  totalTravelCost 20 24 = 960 := by
  sorry

end travel_cost_for_twenty_days_l1991_199151


namespace sum_base8_equals_467_l1991_199194

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Adds two base-8 numbers --/
def addBase8 (a b : ℕ) : ℕ := base10ToBase8 (base8ToBase10 a + base8ToBase10 b)

theorem sum_base8_equals_467 :
  addBase8 (addBase8 236 157) 52 = 467 := by sorry

end sum_base8_equals_467_l1991_199194


namespace octadecagon_diagonals_l1991_199114

/-- The number of sides in an octadecagon -/
def octadecagon_sides : ℕ := 18

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in an octadecagon is 135 -/
theorem octadecagon_diagonals : 
  num_diagonals octadecagon_sides = 135 := by
  sorry

end octadecagon_diagonals_l1991_199114


namespace h_degree_three_iff_c_eq_three_l1991_199157

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 3 - 8*x + 2*x^2 - 7*x^3 + 6*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 2 - 3*x + x^3 - 2*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * (g x)

/-- Theorem stating that h(x) has degree 3 if and only if c = 3 -/
theorem h_degree_three_iff_c_eq_three :
  ∃! c : ℝ, (∀ x : ℝ, h c x = 3 - 8*x + 2*x^2 - 4*x^3) ∧ c = 3 :=
sorry

end h_degree_three_iff_c_eq_three_l1991_199157


namespace jeans_price_ratio_l1991_199183

theorem jeans_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let sale_price := marked_price / 2
  let cost := (5 / 8) * sale_price
  cost / marked_price = 5 / 16 := by
sorry

end jeans_price_ratio_l1991_199183


namespace point_on_y_axis_l1991_199171

/-- 
Given a point P with coordinates (1-x, 2x+1) that lies on the y-axis,
prove that its coordinates are (0, 3).
-/
theorem point_on_y_axis (x : ℝ) :
  (1 - x = 0) ∧ (∃ y, y = 2*x + 1) → (1 - x = 0 ∧ 2*x + 1 = 3) :=
by sorry

end point_on_y_axis_l1991_199171


namespace greatest_gcd_square_successor_l1991_199172

theorem greatest_gcd_square_successor (n : ℕ+) : 
  ∃ (k : ℕ+), Nat.gcd (6 * n^2) (n + 1) ≤ 6 ∧ 
  Nat.gcd (6 * k^2) (k + 1) = 6 :=
sorry

end greatest_gcd_square_successor_l1991_199172


namespace water_drinkers_l1991_199158

theorem water_drinkers (total : ℕ) (juice_percent : ℚ) (water_percent : ℚ) (juice_drinkers : ℕ) : ℕ :=
  let water_drinkers : ℕ := 60
  have h1 : juice_percent = 70 / 100 := by sorry
  have h2 : water_percent = 30 / 100 := by sorry
  have h3 : juice_percent + water_percent = 1 := by sorry
  have h4 : juice_drinkers = 140 := by sorry
  have h5 : ↑juice_drinkers / ↑total = juice_percent := by sorry
  have h6 : ↑water_drinkers / ↑total = water_percent := by sorry
  water_drinkers

#check water_drinkers

end water_drinkers_l1991_199158


namespace remaining_jellybeans_l1991_199101

/-- Calculates the number of jelly beans remaining in a container after distribution --/
def jellybeans_remaining (initial_count : ℕ) (people : ℕ) (first_group : ℕ) (last_group : ℕ) (last_group_beans : ℕ) : ℕ :=
  initial_count - (first_group * 2 * last_group_beans + last_group * last_group_beans)

/-- Theorem stating the number of jelly beans remaining in the container --/
theorem remaining_jellybeans : 
  jellybeans_remaining 8000 10 6 4 400 = 1600 := by
  sorry

end remaining_jellybeans_l1991_199101


namespace last_digit_of_one_over_three_to_ten_l1991_199185

theorem last_digit_of_one_over_three_to_ten (n : ℕ) :
  n = 10 →
  ∃ (k : ℕ), (1 : ℚ) / 3^n = k / 10^10 ∧ k % 10 = 3 :=
by sorry

end last_digit_of_one_over_three_to_ten_l1991_199185


namespace first_month_sale_is_5921_l1991_199137

/-- Calculates the sale in the first month given the sales for months 2 to 6 and the average sale -/
def first_month_sale (sales_2_to_5 : List ℕ) (sale_6 : ℕ) (average : ℕ) : ℕ :=
  6 * average - (sales_2_to_5.sum + sale_6)

/-- Theorem stating that the sale in the first month is 5921 -/
theorem first_month_sale_is_5921 :
  first_month_sale [5468, 5568, 6088, 6433] 5922 5900 = 5921 := by
  sorry

end first_month_sale_is_5921_l1991_199137


namespace negative_one_minus_two_times_negative_two_l1991_199119

theorem negative_one_minus_two_times_negative_two : -1 - 2 * (-2) = 3 := by
  sorry

end negative_one_minus_two_times_negative_two_l1991_199119


namespace intersection_not_solution_quadratic_solution_l1991_199134

theorem intersection_not_solution : ∀ x y : ℝ,
  (y = x ∧ y = x - 4) → (x ≠ 2 ∨ y ≠ 2) :=
by sorry

theorem quadratic_solution : ∀ x : ℝ,
  x^2 - 4*x + 4 = 0 → x = 2 :=
by sorry

end intersection_not_solution_quadratic_solution_l1991_199134


namespace intersection_points_form_hyperbola_l1991_199143

/-- Given real t, the point (x, y) satisfies both equations -/
def satisfies_equations (x y t : ℝ) : Prop :=
  2 * t * x - 3 * y - 4 * t = 0 ∧ 2 * x - 3 * t * y + 4 = 0

/-- The locus of points (x, y) satisfying the equations for all t forms a hyperbola -/
theorem intersection_points_form_hyperbola :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, (∃ t : ℝ, satisfies_equations x y t) →
  x^2 / a^2 - y^2 / b^2 = 1 :=
sorry

end intersection_points_form_hyperbola_l1991_199143


namespace equation_true_iff_m_zero_l1991_199175

theorem equation_true_iff_m_zero (m n : ℝ) :
  21 * (m + n) + 21 = 21 * (-m + n) + 21 ↔ m = 0 := by
  sorry

end equation_true_iff_m_zero_l1991_199175


namespace max_value_ab_l1991_199163

theorem max_value_ab (a b : ℝ) (g : ℝ → ℝ) (ha : a > 0) (hb : b > 0)
  (hg : ∀ x, g x = 2^x) (h_prod : g a * g b = 2) :
  ∀ x y, x > 0 → y > 0 → g x * g y = 2 → x * y ≤ (1/4 : ℝ) :=
by sorry

end max_value_ab_l1991_199163


namespace tray_height_l1991_199133

/-- The height of a tray formed from a square paper with specific cuts -/
theorem tray_height (side_length : ℝ) (cut_start : ℝ) (cut_angle : ℝ) : 
  side_length = 50 →
  cut_start = Real.sqrt 5 →
  cut_angle = π / 4 →
  (Real.sqrt 10) / 2 = 
    cut_start * Real.sin (cut_angle / 2) := by
  sorry

end tray_height_l1991_199133


namespace razorback_tshirt_revenue_l1991_199166

/-- The amount of money made from selling t-shirts at the Razorback shop -/
theorem razorback_tshirt_revenue :
  let profit_per_tshirt : ℕ := 62
  let tshirts_sold : ℕ := 183
  let total_profit : ℕ := profit_per_tshirt * tshirts_sold
  total_profit = 11346 := by sorry

end razorback_tshirt_revenue_l1991_199166


namespace tims_income_percentage_l1991_199140

theorem tims_income_percentage (tim mart juan : ℝ) 
  (h1 : mart = tim + 0.6 * tim) 
  (h2 : mart = 0.9599999999999999 * juan) : 
  tim = 0.6 * juan := by sorry

end tims_income_percentage_l1991_199140


namespace quadratic_equation_solution_l1991_199107

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = -9 ∧ x₂ = -3 ∧ 
  (∀ x : ℝ, x^2 + 12*x + 27 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_equation_solution_l1991_199107


namespace intersection_complement_theorem_l1991_199149

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (x + 1)}

-- State the theorem
theorem intersection_complement_theorem : A ∩ (U \ B) = {x | -2 ≤ x ∧ x ≤ -1} := by sorry

end intersection_complement_theorem_l1991_199149


namespace standard_deviation_of_scores_l1991_199110

def scores : List ℝ := [10, 10, 10, 9, 10, 8, 8, 10, 10, 8]

theorem standard_deviation_of_scores :
  let n : ℕ := scores.length
  let mean : ℝ := (scores.sum) / n
  let variance : ℝ := (scores.map (λ x => (x - mean)^2)).sum / n
  Real.sqrt variance = 0.9 := by sorry

end standard_deviation_of_scores_l1991_199110


namespace double_room_percentage_l1991_199170

theorem double_room_percentage (total_students : ℝ) (h : total_students > 0) :
  let students_in_double_rooms := 0.75 * total_students
  let double_rooms := students_in_double_rooms / 2
  let students_in_single_rooms := 0.25 * total_students
  let single_rooms := students_in_single_rooms
  let total_rooms := double_rooms + single_rooms
  (double_rooms / total_rooms) * 100 = 60 := by
  sorry

end double_room_percentage_l1991_199170


namespace probability_one_or_two_pascal_l1991_199116

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the occurrences of a specific value in Pascal's Triangle up to n rows -/
def countOccurrences (n : ℕ) (value : ℕ) : ℕ := sorry

/-- Calculates the total number of elements in Pascal's Triangle up to n rows -/
def totalElements (n : ℕ) : ℕ := sorry

/-- The main theorem stating the probability of selecting 1 or 2 from the first 20 rows of Pascal's Triangle -/
theorem probability_one_or_two_pascal : 
  (countOccurrences 20 1 + countOccurrences 20 2) / totalElements 20 = 37 / 105 := by sorry

end probability_one_or_two_pascal_l1991_199116


namespace constant_term_quadratic_l1991_199100

theorem constant_term_quadratic (x : ℝ) : 
  (2 * x^2 = x + 4) → 
  (∃ a b : ℝ, 2 * x^2 - x - 4 = a * x^2 + b * x + (-4)) :=
by sorry

end constant_term_quadratic_l1991_199100


namespace max_product_of_externally_tangent_circles_l1991_199145

/-- Circle C₁ with center (a, -2) and radius 2 -/
def C₁ (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y + 2)^2 = 4

/-- Circle C₂ with center (-b, -2) and radius 1 -/
def C₂ (b : ℝ) (x y : ℝ) : Prop := (x + b)^2 + (y + 2)^2 = 1

/-- Circles C₁ and C₂ are externally tangent -/
def externally_tangent (a b : ℝ) : Prop := (a + b)^2 = 3^2

theorem max_product_of_externally_tangent_circles (a b : ℝ) 
  (h : externally_tangent a b) : 
  a * b ≤ 9/4 := by sorry

end max_product_of_externally_tangent_circles_l1991_199145


namespace simplify_and_rationalize_l1991_199146

theorem simplify_and_rationalize :
  (Real.sqrt 6 / Real.sqrt 5) * (Real.sqrt 8 / Real.sqrt 9) * (Real.sqrt 10 / Real.sqrt 11) = 4 * Real.sqrt 66 / 33 := by
  sorry

end simplify_and_rationalize_l1991_199146


namespace difference_of_squares_example_l1991_199184

theorem difference_of_squares_example : (15 + 12)^2 - (15 - 12)^2 = 720 := by
  sorry

end difference_of_squares_example_l1991_199184


namespace line_parallel_to_plane_relationship_l1991_199182

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (parallelToPlane : Line → Plane → Prop)
variable (withinPlane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_relationship 
  (a b : Line) (α : Plane)
  (h1 : parallelToPlane a α)
  (h2 : withinPlane b α) :
  parallel a b ∨ skew a b :=
sorry

end line_parallel_to_plane_relationship_l1991_199182


namespace green_peppers_weight_equal_pepper_weights_l1991_199162

/-- The weight of green peppers bought by Dale's Vegetarian Restaurant -/
def green_peppers : ℝ := 2.8333333335

/-- The total weight of peppers bought by Dale's Vegetarian Restaurant -/
def total_peppers : ℝ := 5.666666667

/-- Theorem stating that the weight of green peppers is half the total weight of peppers -/
theorem green_peppers_weight :
  green_peppers = total_peppers / 2 :=
by sorry

/-- Theorem stating that the weight of green peppers is equal to the weight of red peppers -/
theorem equal_pepper_weights :
  green_peppers = total_peppers - green_peppers :=
by sorry

end green_peppers_weight_equal_pepper_weights_l1991_199162
