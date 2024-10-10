import Mathlib

namespace vector_operation_result_l2020_202045

def a : ℝ × ℝ × ℝ := (3, 5, 1)
def b : ℝ × ℝ × ℝ := (2, 2, 3)
def c : ℝ × ℝ × ℝ := (4, -1, -3)

theorem vector_operation_result :
  (2 : ℝ) • a - (3 : ℝ) • b + (4 : ℝ) • c = (16, 0, -19) := by sorry

end vector_operation_result_l2020_202045


namespace solve_age_problem_l2020_202025

def age_problem (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 9)

theorem solve_age_problem :
  ∃ (a b : ℕ), age_problem a b ∧ b = 39 :=
sorry

end solve_age_problem_l2020_202025


namespace kellys_apples_l2020_202020

/-- The total number of apples Kelly has after picking more apples -/
def total_apples (initial : Float) (picked : Float) : Float :=
  initial + picked

/-- Theorem stating that Kelly's total apples is 161.0 -/
theorem kellys_apples :
  let initial := 56.0
  let picked := 105.0
  total_apples initial picked = 161.0 := by
  sorry

end kellys_apples_l2020_202020


namespace fashion_show_total_time_l2020_202019

/-- Represents the different types of clothing in the fashion show -/
inductive ClothingType
  | EveningWear
  | BathingSuit
  | FormalWear
  | CasualWear

/-- Returns the time in minutes for a runway walk based on the clothing type -/
def walkTime (c : ClothingType) : ℝ :=
  match c with
  | ClothingType.EveningWear => 4
  | ClothingType.BathingSuit => 2
  | ClothingType.FormalWear => 3
  | ClothingType.CasualWear => 2.5

/-- The number of models in the show -/
def numModels : ℕ := 10

/-- Returns the number of sets for each clothing type -/
def numSets (c : ClothingType) : ℕ :=
  match c with
  | ClothingType.EveningWear => 4
  | ClothingType.BathingSuit => 2
  | ClothingType.FormalWear => 3
  | ClothingType.CasualWear => 5

/-- Calculates the total time for all runway walks of a specific clothing type -/
def totalTimeForClothingType (c : ClothingType) : ℝ :=
  (walkTime c) * (numSets c : ℝ) * (numModels : ℝ)

/-- Theorem: The total time for all runway trips during the fashion show is 415 minutes -/
theorem fashion_show_total_time :
  (totalTimeForClothingType ClothingType.EveningWear) +
  (totalTimeForClothingType ClothingType.BathingSuit) +
  (totalTimeForClothingType ClothingType.FormalWear) +
  (totalTimeForClothingType ClothingType.CasualWear) = 415 := by
  sorry


end fashion_show_total_time_l2020_202019


namespace seven_square_side_length_l2020_202090

/-- Represents a shape composed of seven equal squares -/
structure SevenSquareShape :=
  (side_length : ℝ)

/-- Represents a line that divides the shape into two equal areas -/
structure DividingLine :=
  (shape : SevenSquareShape)
  (divides_equally : Bool)

/-- Represents the intersection points of the dividing line with the shape -/
structure IntersectionPoints :=
  (line : DividingLine)
  (cf_length : ℝ)
  (ae_length : ℝ)
  (sum_cf_ae : cf_length + ae_length = 91)

/-- Theorem: The side length of each small square is 26 cm -/
theorem seven_square_side_length
  (shape : SevenSquareShape)
  (line : DividingLine)
  (points : IntersectionPoints)
  (h1 : line.shape = shape)
  (h2 : line.divides_equally = true)
  (h3 : points.line = line)
  : shape.side_length = 26 :=
by sorry

end seven_square_side_length_l2020_202090


namespace f_g_f_3_equals_1360_l2020_202059

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x + 4
def g (x : ℝ) : ℝ := x^2 + 5 * x + 3

-- State the theorem
theorem f_g_f_3_equals_1360 : f (g (f 3)) = 1360 := by
  sorry

end f_g_f_3_equals_1360_l2020_202059


namespace power_calculation_l2020_202082

theorem power_calculation : (-8 : ℝ)^2023 * (1/8 : ℝ)^2024 = -1/8 := by sorry

end power_calculation_l2020_202082


namespace permutations_three_distinct_l2020_202073

/-- The number of distinct permutations of three distinct elements -/
def num_permutations_three_distinct : ℕ := 6

/-- Theorem: The number of distinct permutations of three distinct elements is 6 -/
theorem permutations_three_distinct :
  num_permutations_three_distinct = 6 := by
  sorry

end permutations_three_distinct_l2020_202073


namespace sqrt_equation_solution_l2020_202029

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end sqrt_equation_solution_l2020_202029


namespace smallest_divisible_by_18_and_24_l2020_202013

theorem smallest_divisible_by_18_and_24 : 
  ∃ n : ℕ, (n > 0 ∧ n % 18 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 18 = 0 ∧ m % 24 = 0) → n ≤ m) ∧ n = 72 :=
by sorry

end smallest_divisible_by_18_and_24_l2020_202013


namespace bob_pizza_calorie_intake_l2020_202078

/-- Calculates the average calorie intake per slice for the slices Bob ate from a pizza -/
def average_calorie_intake (total_slices : ℕ) (low_cal_slices : ℕ) (high_cal_slices : ℕ) (low_cal : ℕ) (high_cal : ℕ) : ℚ :=
  (low_cal_slices * low_cal + high_cal_slices * high_cal) / (low_cal_slices + high_cal_slices)

/-- Theorem stating that the average calorie intake per slice for the slices Bob ate is approximately 357.14 calories -/
theorem bob_pizza_calorie_intake :
  average_calorie_intake 12 3 4 300 400 = 2500 / 7 := by
  sorry

end bob_pizza_calorie_intake_l2020_202078


namespace cubic_function_property_l2020_202095

/-- A cubic function g(x) with specific properties -/
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem cubic_function_property (p q r s : ℝ) :
  g p q r s 1 = 1 →
  g p q r s 3 = 1 →
  g p q r s 2 = 2 →
  (fun x ↦ 3 * p * x^2 + 2 * q * x + r) 2 = 0 →
  3 * p - 2 * q + r - 4 * s = -2 := by
  sorry

end cubic_function_property_l2020_202095


namespace edward_money_theorem_l2020_202040

def edward_money_problem (initial_money spent1 spent2 : ℕ) : Prop :=
  let total_spent := spent1 + spent2
  let remaining_money := initial_money - total_spent
  remaining_money = 17

theorem edward_money_theorem :
  edward_money_problem 34 9 8 := by
  sorry

end edward_money_theorem_l2020_202040


namespace fair_coin_toss_is_fair_l2020_202069

-- Define a fair coin
def fair_coin (outcome : Bool) : ℝ :=
  if outcome then 0.5 else 0.5

-- Define fairness of a decision method
def is_fair (decision_method : Bool → ℝ) : Prop :=
  decision_method true = decision_method false

-- Theorem statement
theorem fair_coin_toss_is_fair :
  is_fair fair_coin :=
sorry

end fair_coin_toss_is_fair_l2020_202069


namespace complement_of_P_P_subset_Q_range_P_inter_Q_eq_Q_range_final_range_of_m_l2020_202093

-- Define sets P and Q
def P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}
def Q (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem for the complement of P
theorem complement_of_P : (Set.univ \ P) = {x | x < -2 ∨ x > 10} := by sorry

-- Theorem for P being a subset of Q
theorem P_subset_Q_range (m : ℝ) : P ⊆ Q m ↔ m ≥ 9 := by sorry

-- Theorem for intersection of P and Q equals Q
theorem P_inter_Q_eq_Q_range (m : ℝ) : P ∩ Q m = Q m ↔ m ≤ 9 := by sorry

-- Theorem for the final range of m satisfying both conditions
theorem final_range_of_m : 
  {m : ℝ | P ⊆ Q m ∧ P ∩ Q m = Q m} = {m : ℝ | 9 ≤ m ∧ m ≤ 9} := by sorry

end complement_of_P_P_subset_Q_range_P_inter_Q_eq_Q_range_final_range_of_m_l2020_202093


namespace jason_pokemon_cards_l2020_202050

theorem jason_pokemon_cards (initial : ℕ) : 
  initial - 9 = 4 → initial = 13 := by
  sorry

end jason_pokemon_cards_l2020_202050


namespace opposite_face_is_A_l2020_202080

/-- Represents the labels of the squares --/
inductive Label
  | A | B | C | D | E | F

/-- Represents a cube formed by folding six squares --/
structure Cube where
  top : Label
  bottom : Label
  front : Label
  back : Label
  left : Label
  right : Label

/-- Represents the linear arrangement of squares before folding --/
def LinearArrangement := List Label

/-- Function to create a cube from a linear arrangement of squares --/
def foldCube (arrangement : LinearArrangement) (top : Label) : Cube :=
  sorry

/-- The theorem to be proved --/
theorem opposite_face_is_A 
  (arrangement : LinearArrangement) 
  (h1 : arrangement = [Label.A, Label.B, Label.C, Label.D, Label.E, Label.F]) 
  (cube : Cube) 
  (h2 : cube = foldCube arrangement Label.B) : 
  cube.bottom = Label.A :=
sorry

end opposite_face_is_A_l2020_202080


namespace man_rowing_speed_l2020_202076

/-- Given a man's downstream speed and speed in still water, calculate his upstream speed -/
theorem man_rowing_speed (downstream_speed still_water_speed : ℝ) 
  (h1 : downstream_speed = 31)
  (h2 : still_water_speed = 28) :
  still_water_speed - (downstream_speed - still_water_speed) = 25 := by
  sorry

end man_rowing_speed_l2020_202076


namespace article_cost_l2020_202022

theorem article_cost (C : ℝ) (S : ℝ) : 
  S = 1.25 * C →                            -- 25% profit
  (0.8 * C + 0.3 * (0.8 * C) = S - 10.50) → -- 30% profit on reduced cost and price
  C = 50 := by
sorry

end article_cost_l2020_202022


namespace travel_problem_solvable_l2020_202017

/-- A strategy for three friends to travel between two cities --/
structure TravelStrategy where
  /-- The time taken for all friends to reach their destinations --/
  total_time : ℝ
  /-- Assertion that the strategy is valid --/
  is_valid : Prop

/-- The travel problem setup --/
structure TravelProblem where
  /-- Distance between the two cities in km --/
  distance : ℝ
  /-- Maximum walking speed in km/h --/
  walk_speed : ℝ
  /-- Maximum cycling speed in km/h --/
  cycle_speed : ℝ

/-- The existence of a valid strategy for the given travel problem --/
def exists_valid_strategy (problem : TravelProblem) : Prop :=
  ∃ (strategy : TravelStrategy), 
    strategy.is_valid ∧ 
    strategy.total_time ≤ 160/60 ∧  -- 2 hours and 40 minutes in hours
    problem.distance = 24 ∧
    problem.walk_speed ≤ 6 ∧
    problem.cycle_speed ≤ 18

/-- Theorem stating that there exists a valid strategy for the given problem --/
theorem travel_problem_solvable : 
  ∃ (problem : TravelProblem), exists_valid_strategy problem :=
sorry

end travel_problem_solvable_l2020_202017


namespace arithmetic_sequence_y_value_l2020_202003

/-- 
Given an arithmetic sequence with the first three terms 2/3, y-2, and 4y+1,
prove that y = -17/6.
-/
theorem arithmetic_sequence_y_value :
  ∀ y : ℚ,
  let a₁ : ℚ := 2/3
  let a₂ : ℚ := y - 2
  let a₃ : ℚ := 4*y + 1
  (a₂ - a₁ = a₃ - a₂) →
  y = -17/6 := by
sorry

end arithmetic_sequence_y_value_l2020_202003


namespace scale_model_height_l2020_202098

/-- The scale ratio of the model to the actual skyscraper -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the skyscraper in feet -/
def actual_height : ℕ := 1250

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The height of the scale model in inches -/
def model_height_inches : ℕ := 600

/-- Theorem stating that the height of the scale model in inches is 600 -/
theorem scale_model_height :
  (actual_height : ℚ) * scale_ratio * inches_per_foot = model_height_inches := by
  sorry

end scale_model_height_l2020_202098


namespace quadratic_solution_sum_l2020_202066

theorem quadratic_solution_sum (c d : ℝ) : 
  (∀ x, x^2 - 6*x + 15 = 27 ↔ x = c ∨ x = d) →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
  sorry

end quadratic_solution_sum_l2020_202066


namespace number_of_fours_is_even_l2020_202049

theorem number_of_fours_is_even (x y z : ℕ) : 
  x + y + z = 80 →
  3 * x + 4 * y + 5 * z = 276 →
  Even y :=
by sorry

end number_of_fours_is_even_l2020_202049


namespace cube_cutting_theorem_l2020_202092

/-- A plane in 3D space --/
structure Plane where
  normal : ℝ × ℝ × ℝ
  distance : ℝ

/-- A part of a cube resulting from cuts --/
structure CubePart where
  points : Set (ℝ × ℝ × ℝ)

/-- Function to calculate the maximum distance between any two points in a set --/
def maxDistance (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- Function to cut a unit cube with given planes --/
def cutCube (planes : List Plane) : List CubePart := sorry

theorem cube_cutting_theorem :
  (∃ (planes : List Plane), planes.length = 4 ∧ 
    ∀ part ∈ cutCube planes, maxDistance part.points < 4/5) ∧
  (¬ ∃ (planes : List Plane), planes.length = 4 ∧ 
    ∀ part ∈ cutCube planes, maxDistance part.points < 4/7) := by
  sorry

end cube_cutting_theorem_l2020_202092


namespace sqrt_3_times_sqrt_12_l2020_202000

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l2020_202000


namespace total_trees_planted_l2020_202072

theorem total_trees_planted (apricot_trees peach_trees : ℕ) : 
  apricot_trees = 58 →
  peach_trees = 3 * apricot_trees →
  apricot_trees + peach_trees = 232 := by
sorry

end total_trees_planted_l2020_202072


namespace cherry_weekly_earnings_l2020_202032

/-- Represents Cherry's delivery service earnings --/
def cherry_earnings : ℕ → ℚ
| 5 => 2.5  -- $2.50 for 5 kg cargo
| 8 => 4    -- $4 for 8 kg cargo
| _ => 0    -- Default case

/-- Calculates Cherry's daily earnings --/
def daily_earnings : ℚ :=
  4 * cherry_earnings 5 + 2 * cherry_earnings 8

/-- Theorem: Cherry's weekly earnings are $126 --/
theorem cherry_weekly_earnings :
  7 * daily_earnings = 126 := by
  sorry

end cherry_weekly_earnings_l2020_202032


namespace special_function_characterization_l2020_202057

/-- A function f: ℝ² → ℝ satisfying specific conditions -/
def SpecialFunction (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f x y = f y x) ∧
  (∀ x y z : ℝ, (f x y - f y z) * (f y z - f z x) * (f z x - f x y) = 0) ∧
  (∀ x y a : ℝ, f (x + a) (y + a) = f x y + a) ∧
  (∀ x y : ℝ, x ≤ y → f 0 x ≤ f 0 y)

/-- The theorem stating that any SpecialFunction must be of a specific form -/
theorem special_function_characterization (f : ℝ → ℝ → ℝ) (hf : SpecialFunction f) :
  ∃ a : ℝ, (∀ x y : ℝ, f x y = a + min x y) ∨ (∀ x y : ℝ, f x y = a + max x y) := by
  sorry

end special_function_characterization_l2020_202057


namespace pirate_costume_cost_l2020_202024

theorem pirate_costume_cost (num_friends : ℕ) (cost_per_costume : ℕ) : 
  num_friends = 8 → cost_per_costume = 5 → num_friends * cost_per_costume = 40 :=
by
  sorry

end pirate_costume_cost_l2020_202024


namespace zoo_ticket_price_l2020_202014

theorem zoo_ticket_price (regular_price : ℝ) (discount_percentage : ℝ) (discounted_price : ℝ) : 
  regular_price = 15 →
  discount_percentage = 40 →
  discounted_price = regular_price * (1 - discount_percentage / 100) →
  discounted_price = 9 := by
sorry

end zoo_ticket_price_l2020_202014


namespace initial_toy_cost_l2020_202070

theorem initial_toy_cost (total_toys : ℕ) (total_cost : ℕ) (teddy_bears : ℕ) (teddy_cost : ℕ) (initial_toys : ℕ) :
  total_toys = initial_toys + teddy_bears →
  total_cost = teddy_bears * teddy_cost + initial_toys * 10 →
  teddy_bears = 20 →
  teddy_cost = 15 →
  initial_toys = 28 →
  total_cost = 580 →
  10 = total_cost / total_toys - (teddy_bears * teddy_cost) / initial_toys :=
by sorry

end initial_toy_cost_l2020_202070


namespace largest_divisor_of_n4_minus_n2_l2020_202053

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 
  (∃ (k : ℤ), n^4 - n^2 = 12 * k) ∧ 
  (∀ (m : ℤ), m > 12 → ∃ (n : ℤ), ¬∃ (k : ℤ), n^4 - n^2 = m * k) :=
by sorry

end largest_divisor_of_n4_minus_n2_l2020_202053


namespace initial_condition_recurrence_relation_diamonds_in_25th_figure_l2020_202067

/-- The number of diamonds in the n-th figure of the sequence -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 * n^2 + 2 * n - 3

/-- The sequence starts with one diamond in the first figure -/
theorem initial_condition : num_diamonds 1 = 1 := by sorry

/-- The recurrence relation for n ≥ 2 -/
theorem recurrence_relation (n : ℕ) (h : n ≥ 2) :
  num_diamonds n = num_diamonds (n-1) + 4*n := by sorry

/-- The main theorem: The 25th figure contains 1297 diamonds -/
theorem diamonds_in_25th_figure : num_diamonds 25 = 1297 := by sorry

end initial_condition_recurrence_relation_diamonds_in_25th_figure_l2020_202067


namespace dog_weight_problem_l2020_202036

theorem dog_weight_problem (x y : ℝ) :
  -- Define the weights of the dogs
  let w₂ : ℝ := 31
  let w₃ : ℝ := 35
  let w₄ : ℝ := 33
  let w₅ : ℝ := y
  -- The average of the first 4 dogs equals the average of all 5 dogs
  (x + w₂ + w₃ + w₄) / 4 = (x + w₂ + w₃ + w₄ + w₅) / 5 →
  -- The weight of the fifth dog is 31 pounds
  y = 31 →
  -- The weight of the first dog is 25 pounds
  x = 25 := by
sorry

end dog_weight_problem_l2020_202036


namespace multiply_powers_of_x_l2020_202075

theorem multiply_powers_of_x (x : ℝ) : 2 * x * (3 * x^2) = 6 * x^3 := by
  sorry

end multiply_powers_of_x_l2020_202075


namespace x_wins_in_six_moves_l2020_202035

/-- Represents a position on the infinite grid -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents a player in the game -/
inductive Player
  | X
  | O

/-- Represents the state of the game -/
structure GameState :=
  (moves : List (Player × Position))
  (currentPlayer : Player)

/-- Checks if a given list of positions forms a winning line -/
def isWinningLine (line : List Position) : Bool :=
  sorry

/-- Checks if the current game state is a win for the given player -/
def isWin (state : GameState) (player : Player) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Position

/-- The theorem stating that X has a winning strategy in at most 6 moves -/
theorem x_wins_in_six_moves :
  ∃ (strategy : Strategy),
    ∀ (opponent_strategy : Strategy),
      ∃ (final_state : GameState),
        (final_state.moves.length ≤ 6) ∧
        (isWin final_state Player.X) :=
  sorry

end x_wins_in_six_moves_l2020_202035


namespace second_shop_amount_calculation_l2020_202081

/-- The amount paid for books from the second shop -/
def second_shop_amount (books_shop1 books_shop2 : ℕ) (amount_shop1 avg_price : ℚ) : ℚ :=
  (books_shop1 + books_shop2 : ℚ) * avg_price - amount_shop1

/-- Theorem stating the amount paid for books from the second shop -/
theorem second_shop_amount_calculation :
  second_shop_amount 27 20 581 25 = 594 := by
  sorry

end second_shop_amount_calculation_l2020_202081


namespace complex_arithmetic_equation_l2020_202039

theorem complex_arithmetic_equation : 
  -1^4 + (4 - (3/8 + 1/6 - 3/4) * 24) / 5 = 0.8 := by
  sorry

end complex_arithmetic_equation_l2020_202039


namespace carly_grill_capacity_l2020_202088

/-- The number of burgers Carly can fit on the grill at once -/
def burgers_on_grill (guests : ℕ) (cooking_time_per_burger : ℕ) (total_cooking_time : ℕ) : ℕ :=
  let total_burgers := guests / 2 * 2 + guests / 2 * 1
  total_burgers * cooking_time_per_burger / total_cooking_time

theorem carly_grill_capacity :
  burgers_on_grill 30 8 72 = 5 := by
  sorry

end carly_grill_capacity_l2020_202088


namespace reservoir_capacity_difference_l2020_202055

/-- Proves that the difference between total capacity and normal level is 25 million gallons --/
theorem reservoir_capacity_difference (current_amount : ℝ) (normal_level : ℝ) (total_capacity : ℝ)
  (h1 : current_amount = 30)
  (h2 : current_amount = 2 * normal_level)
  (h3 : current_amount = 0.75 * total_capacity) :
  total_capacity - normal_level = 25 := by
  sorry

end reservoir_capacity_difference_l2020_202055


namespace alcohol_mixture_theorem_alcohol_mixture_validity_l2020_202091

/-- Proves that adding the calculated amount of alcohol results in the desired concentration -/
theorem alcohol_mixture_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c ≥ 0) (hd : d ≥ 0) 
  (hac : a ≠ c) (had : a ≠ d) (hcd : c ≠ d) :
  let x := b * (d - c) / (a - d)
  (b * c + x * a) / (b + x) = d :=
by sorry

/-- Proves that the solution is valid when d is between a and c -/
theorem alcohol_mixture_validity (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c ≥ 0) (hd : d ≥ 0) 
  (hac : a ≠ c) (had : a ≠ d) (hcd : c ≠ d) :
  (min a c < d ∧ d < max a c) → 
  let x := b * (d - c) / (a - d)
  x > 0 :=
by sorry

end alcohol_mixture_theorem_alcohol_mixture_validity_l2020_202091


namespace village_foods_monthly_sales_l2020_202041

/-- Represents the monthly sales data for Village Foods --/
structure VillageFoodsSales where
  customers_per_month : ℕ
  lettuce_per_customer : ℕ
  lettuce_price : ℚ
  tomatoes_per_customer : ℕ
  tomato_price : ℚ

/-- Calculates the total monthly sales of lettuce and tomatoes --/
def total_monthly_sales (sales : VillageFoodsSales) : ℚ :=
  sales.customers_per_month * 
  (sales.lettuce_per_customer * sales.lettuce_price + 
   sales.tomatoes_per_customer * sales.tomato_price)

/-- Theorem stating that the total monthly sales of lettuce and tomatoes is $2000 --/
theorem village_foods_monthly_sales :
  let sales : VillageFoodsSales := {
    customers_per_month := 500,
    lettuce_per_customer := 2,
    lettuce_price := 1,
    tomatoes_per_customer := 4,
    tomato_price := 1/2
  }
  total_monthly_sales sales = 2000 := by sorry

end village_foods_monthly_sales_l2020_202041


namespace place_value_difference_power_l2020_202043

/-- Given a natural number, returns the count of a specific digit in it. -/
def countDigit (n : ℕ) (digit : ℕ) : ℕ := sorry

/-- Given a natural number, returns a list of place values for specific digits. -/
def getPlaceValues (n : ℕ) (digits : List ℕ) : List ℕ := sorry

/-- Calculates the sum of differences between consecutive place values. -/
def sumOfDifferences (placeValues : List ℕ) : ℕ := sorry

/-- The main theorem to prove. -/
theorem place_value_difference_power (n : ℕ) (h : n = 58219435) :
  let placeValues := getPlaceValues n [1, 5, 8]
  let diffSum := sumOfDifferences placeValues
  let numTwos := countDigit n 2
  diffSum ^ numTwos = 420950000 := by sorry

end place_value_difference_power_l2020_202043


namespace red_knights_magical_swords_fraction_l2020_202063

/-- Represents the color of a knight -/
inductive KnightColor
  | Red
  | Blue
  | Green

/-- Represents the total number of knights -/
def totalKnights : ℕ := 40

/-- The fraction of knights that are red -/
def redFraction : ℚ := 3/8

/-- The fraction of knights that are blue -/
def blueFraction : ℚ := 1/4

/-- The fraction of knights that are green -/
def greenFraction : ℚ := 1 - redFraction - blueFraction

/-- The fraction of all knights that wield magical swords -/
def magicalSwordsFraction : ℚ := 1/5

/-- The ratio of red knights with magical swords to blue knights with magical swords -/
def redToBlueMagicalRatio : ℚ := 3/2

/-- The ratio of red knights with magical swords to green knights with magical swords -/
def redToGreenMagicalRatio : ℚ := 2

theorem red_knights_magical_swords_fraction :
  ∃ (redMagicalFraction : ℚ),
    redMagicalFraction = 48/175 ∧
    redMagicalFraction * redFraction * totalKnights +
    (redMagicalFraction / redToBlueMagicalRatio) * blueFraction * totalKnights +
    (redMagicalFraction / redToGreenMagicalRatio) * greenFraction * totalKnights =
    magicalSwordsFraction * totalKnights :=
by sorry

end red_knights_magical_swords_fraction_l2020_202063


namespace exchange_impossibility_l2020_202062

theorem exchange_impossibility : ¬ ∃ (N : ℤ), 5 * N = 2001 := by sorry

end exchange_impossibility_l2020_202062


namespace binomial_square_constant_l2020_202015

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 300*x + c = (x + a)^2) → c = 22500 := by
  sorry

end binomial_square_constant_l2020_202015


namespace least_multiple_36_with_digit_product_multiple_9_l2020_202051

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let digits := n.digits 10
  digits.foldl (· * ·) 1

theorem least_multiple_36_with_digit_product_multiple_9 :
  ∀ n : ℕ, n > 0 →
    is_multiple_of n 36 →
    is_multiple_of (digit_product n) 9 →
    n ≥ 36 :=
sorry

end least_multiple_36_with_digit_product_multiple_9_l2020_202051


namespace total_students_is_17_l2020_202046

/-- Represents the total number of students in a class with various sports preferences. -/
def total_students : ℕ :=
  let baseball_and_football := 7
  let only_baseball := 3
  let only_football := 4
  let basketball_as_well := 2
  let basketball_and_football_not_baseball := 1
  let all_three_sports := 2
  let no_sports := 5
  let only_basketball := basketball_as_well - basketball_and_football_not_baseball - all_three_sports

  (baseball_and_football - all_three_sports) + 
  only_baseball + 
  only_football + 
  basketball_and_football_not_baseball + 
  all_three_sports + 
  no_sports + 
  only_basketball

/-- Theorem stating that the total number of students in the class is 17. -/
theorem total_students_is_17 : total_students = 17 := by
  sorry

end total_students_is_17_l2020_202046


namespace function_behavior_l2020_202031

theorem function_behavior (f : ℝ → ℝ) (h : ∀ x : ℝ, f x < f (x + 1)) :
  (∃ a b : ℝ, a < b ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∃ g : ℝ → ℝ, (∀ x : ℝ, g x < g (x + 1)) ∧
    ∀ a b : ℝ, a < b → ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ g x ≥ g y) :=
by sorry

end function_behavior_l2020_202031


namespace yoojung_notebooks_l2020_202096

theorem yoojung_notebooks :
  ∀ (initial : ℕ), 
  (initial ≥ 5) →
  (initial - 5) % 2 = 0 →
  ((initial - 5) / 2 - (initial - 5) / 2 / 2 = 4) →
  initial = 13 :=
by
  sorry

end yoojung_notebooks_l2020_202096


namespace smallest_consecutive_sequence_sum_l2020_202068

theorem smallest_consecutive_sequence_sum (B : ℤ) : B = 1011 ↔ 
  (∀ k < B, ¬∃ n : ℕ+, (n : ℤ) * (2 * k + n - 1) = 2023 ∧ n > 1) ∧
  (∃ n : ℕ+, (n : ℤ) * (2 * B + n - 1) = 2023 ∧ n > 1) :=
by sorry

end smallest_consecutive_sequence_sum_l2020_202068


namespace duck_flying_days_l2020_202006

/-- The number of days a duck spends flying during winter, summer, and spring -/
def total_flying_days (south_days : ℕ) (east_days : ℕ) : ℕ :=
  south_days + 2 * south_days + east_days

/-- Theorem: The duck spends 180 days flying during winter, summer, and spring -/
theorem duck_flying_days : total_flying_days 40 60 = 180 := by
  sorry

end duck_flying_days_l2020_202006


namespace geometric_sequence_product_l2020_202071

/-- A geometric sequence with positive terms where a₁ and a₉₉ are roots of x² - 10x + 16 = 0 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ r, ∀ n, a (n + 1) = r * a n) ∧
  (a 1 * a 99 = 16) ∧
  (a 1 + a 99 = 10)

/-- The product of specific terms in the geometric sequence equals 64 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 20 * a 50 * a 80 = 64 := by
  sorry

end geometric_sequence_product_l2020_202071


namespace part_to_whole_ratio_l2020_202016

theorem part_to_whole_ratio (N P : ℝ) 
  (h1 : (1/4) * (1/3) * P = 10)
  (h2 : 0.40 * N = 120) : 
  P/N = 1/2.5 := by
sorry

end part_to_whole_ratio_l2020_202016


namespace ones_digit_of_17_power_l2020_202085

theorem ones_digit_of_17_power : ∃ n : ℕ, 17^(17*(13^13)) ≡ 7 [ZMOD 10] := by sorry

end ones_digit_of_17_power_l2020_202085


namespace new_assessed_value_calculation_l2020_202034

/-- Represents the property tax calculation in Township K -/
structure PropertyTax where
  initialValue : ℝ
  newValue : ℝ
  taxRate : ℝ
  taxIncrease : ℝ

/-- Theorem stating the relationship between tax increase and new assessed value -/
theorem new_assessed_value_calculation (p : PropertyTax)
  (h1 : p.initialValue = 20000)
  (h2 : p.taxRate = 0.1)
  (h3 : p.taxIncrease = 800)
  (h4 : p.taxRate * p.newValue - p.taxRate * p.initialValue = p.taxIncrease) :
  p.newValue = 28000 := by
  sorry

end new_assessed_value_calculation_l2020_202034


namespace circles_intersect_l2020_202097

def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

theorem circles_intersect : ∃ (x y : ℝ), circle_C1 x y ∧ circle_C2 x y := by
  sorry

end circles_intersect_l2020_202097


namespace max_altitude_product_right_triangle_l2020_202038

/-- Given a fixed side length and area, the product of altitudes is maximum for a right triangle --/
theorem max_altitude_product_right_triangle 
  (l : ℝ) (S : ℝ) (h_pos_l : l > 0) (h_pos_S : S > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    (1/2) * l * b = S ∧
    ∀ (x y : ℝ), x > 0 → y > 0 → (1/2) * l * y = S →
      (2*S/l) * (2*S/(l*x)) * (2*S/(l*y)) ≤ (2*S/l) * (2*S/(l*a)) * (2*S/(l*b)) :=
sorry

end max_altitude_product_right_triangle_l2020_202038


namespace inequality_proof_l2020_202087

theorem inequality_proof (x₁ x₂ y₁ y₂ : ℝ) (h : x₁^2 + x₂^2 ≤ 1) :
  (x₁*y₁ + x₂*y₂ - 1)^2 ≥ (x₁^2 + x₂^2 - 1)*(y₁^2 + y₂^2 - 1) :=
by sorry

end inequality_proof_l2020_202087


namespace amy_balloons_l2020_202061

theorem amy_balloons (red green blue : ℕ) (h1 : red = 29) (h2 : green = 17) (h3 : blue = 21) :
  red + green + blue = 67 := by
  sorry

end amy_balloons_l2020_202061


namespace afternoon_sales_l2020_202044

/-- Represents the amount of pears sold in kilograms during different parts of the day -/
structure PearSales where
  morning : ℝ
  afternoon : ℝ
  evening : ℝ

/-- Defines the relationship between sales in different parts of the day -/
def valid_sales (s : PearSales) : Prop :=
  s.afternoon = 2 * s.morning ∧ 
  s.evening = 3 * s.afternoon ∧ 
  s.morning + s.afternoon + s.evening = 510

/-- Theorem stating that the afternoon sales equal 510 / 4.5 kg given the conditions -/
theorem afternoon_sales (s : PearSales) (h : valid_sales s) : 
  s.afternoon = 510 / 4.5 := by
  sorry

end afternoon_sales_l2020_202044


namespace sum_of_specific_numbers_l2020_202023

theorem sum_of_specific_numbers : 
  217 + 2.017 + 0.217 + 2.0017 = 221.2357 := by
  sorry

end sum_of_specific_numbers_l2020_202023


namespace nested_sqrt_24_l2020_202033

/-- The solution to the equation x = √(24 + x), where x is non-negative -/
theorem nested_sqrt_24 : 
  ∃ x : ℝ, x ≥ 0 ∧ x = Real.sqrt (24 + x) → x = 6 := by sorry

end nested_sqrt_24_l2020_202033


namespace inverse_function_difference_l2020_202001

-- Define a function f and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Define the property that f and f_inv are inverse functions
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define the property that f(x+2) and f^(-1)(x-1) are inverse functions
def special_inverse_property (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv (x - 1) + 2) = x ∧ f_inv (f (x + 2) - 1) = x

-- Theorem statement
theorem inverse_function_difference
  (h1 : is_inverse f f_inv)
  (h2 : special_inverse_property f f_inv) :
  f_inv 2004 - f_inv 1 = 4006 :=
by sorry

end inverse_function_difference_l2020_202001


namespace square_difference_l2020_202060

theorem square_difference (a b : ℝ) : a^2 - 2*a*b + b^2 = (a - b)^2 := by
  sorry

end square_difference_l2020_202060


namespace angle_terminal_side_l2020_202058

theorem angle_terminal_side (α : Real) (x : Real) :
  (∃ P : Real × Real, P = (x, 4) ∧ P.1 = x * Real.cos α ∧ P.2 = 4 * Real.sin α) →
  Real.sin α = 4/5 →
  x = 3 ∨ x = -3 := by
sorry

end angle_terminal_side_l2020_202058


namespace second_integer_value_l2020_202018

theorem second_integer_value (a b c d : ℤ) : 
  (∃ x : ℤ, a = x ∧ b = x + 2 ∧ c = x + 4 ∧ d = x + 6) →  -- consecutive even integers
  (a + d = 156) →                                        -- sum of first and fourth is 156
  b = 77                                                 -- second integer is 77
:= by sorry

end second_integer_value_l2020_202018


namespace jose_wrong_questions_l2020_202084

theorem jose_wrong_questions (total_questions : ℕ) (marks_per_question : ℕ) 
  (meghan_score jose_score alisson_score : ℕ) : 
  total_questions = 50 →
  marks_per_question = 2 →
  meghan_score = jose_score - 20 →
  jose_score = alisson_score + 40 →
  meghan_score + jose_score + alisson_score = 210 →
  total_questions * marks_per_question - jose_score = 5 * marks_per_question :=
by sorry

end jose_wrong_questions_l2020_202084


namespace fractional_inequality_solution_set_l2020_202026

theorem fractional_inequality_solution_set (x : ℝ) : 
  (x - 1) / (2 * x - 1) ≤ 0 ↔ 1/2 < x ∧ x ≤ 1 :=
by sorry

end fractional_inequality_solution_set_l2020_202026


namespace ratio_problem_l2020_202065

theorem ratio_problem (a b : ℕ) (h1 : a = 55) (h2 : a = 5 * b) : b = 11 := by
  sorry

end ratio_problem_l2020_202065


namespace gcd_5039_3427_l2020_202010

theorem gcd_5039_3427 : Nat.gcd 5039 3427 = 7 := by
  sorry

end gcd_5039_3427_l2020_202010


namespace trajectory_of_M_is_ellipse_l2020_202009

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 32 = 0

-- Define point A
def point_A : ℝ × ℝ := (2, 0)

-- Define a moving point P on circle C
def point_P (x y : ℝ) : Prop := circle_C x y

-- Define point M as the intersection of perpendicular bisector of AP and line PC
def point_M (x y : ℝ) : Prop :=
  ∃ (px py : ℝ), point_P px py ∧
  ((x - 2)^2 + y^2 = (x - px)^2 + (y - py)^2) ∧
  ((x - 2) * (px - 2) + y * py = 0)

-- Theorem: The trajectory of point M is an ellipse
theorem trajectory_of_M_is_ellipse :
  ∀ (x y : ℝ), point_M x y ↔ x^2/9 + y^2/5 = 1 :=
sorry

end trajectory_of_M_is_ellipse_l2020_202009


namespace shoe_size_ratio_l2020_202042

def jasmine_shoe_size : ℕ := 7
def combined_shoe_size : ℕ := 21

def alexa_shoe_size : ℕ := combined_shoe_size - jasmine_shoe_size

theorem shoe_size_ratio : 
  alexa_shoe_size / jasmine_shoe_size = 2 := by sorry

end shoe_size_ratio_l2020_202042


namespace player_B_more_consistent_l2020_202030

def player_A_scores : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def player_B_scores : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance (scores : List ℕ) : ℚ :=
  let m := mean scores
  (scores.map (fun x => ((x : ℚ) - m) ^ 2)).sum / scores.length

theorem player_B_more_consistent :
  mean player_A_scores = mean player_B_scores ∧
  variance player_B_scores < variance player_A_scores := by
  sorry

#eval mean player_A_scores
#eval mean player_B_scores
#eval variance player_A_scores
#eval variance player_B_scores

end player_B_more_consistent_l2020_202030


namespace subset_condition_disjoint_condition_l2020_202027

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 < x ∧ x < 2*a + 3}

-- Theorem 1: A ⊆ B iff a ∈ [-1/2, 0]
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a ∈ Set.Icc (-1/2) 0 := by sorry

-- Theorem 2: A ∩ B = ∅ iff a ∈ (-∞, -2] ∪ [3/2, +∞)
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ ↔ a ∈ Set.Iic (-2) ∪ Set.Ici (3/2) := by sorry

end subset_condition_disjoint_condition_l2020_202027


namespace integer_pair_inequality_l2020_202021

theorem integer_pair_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 ≤ m^n - n^m ∧ m^n - n^m ≤ m*n) ↔ 
  ((n = 1 ∧ m ≥ 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 3 ∧ n = 2)) := by
sorry

end integer_pair_inequality_l2020_202021


namespace value_of_expression_l2020_202037

-- Define the function g
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

-- State the theorem
theorem value_of_expression (p q r s : ℝ) : g p q r s 3 = 6 → 6*p - 3*q + 2*r - s = 0 := by
  sorry

end value_of_expression_l2020_202037


namespace symmetric_function_value_l2020_202083

/-- A function symmetric about x=1 -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (2 - x)

/-- The main theorem -/
theorem symmetric_function_value (f : ℝ → ℝ) 
  (h_sym : SymmetricAboutOne f)
  (h_def : ∀ x ≥ 1, f x = x * (1 - x)) : 
  f (-2) = -12 := by
  sorry

end symmetric_function_value_l2020_202083


namespace drowned_ratio_l2020_202011

/-- Proves the ratio of drowned cows to drowned sheep given the initial conditions -/
theorem drowned_ratio (initial_sheep initial_cows initial_dogs : ℕ)
  (drowned_sheep : ℕ) (total_survived : ℕ) :
  initial_sheep = 20 →
  initial_cows = 10 →
  initial_dogs = 14 →
  drowned_sheep = 3 →
  total_survived = 35 →
  (initial_cows - (total_survived - (initial_sheep - drowned_sheep) - initial_dogs)) /
  drowned_sheep = 2 := by
  sorry

end drowned_ratio_l2020_202011


namespace prime_square_in_A_implies_prime_in_A_l2020_202052

/-- The set of positive integers of the form a^2 + 2b^2, where a and b are integers and b ≠ 0 -/
def A : Set ℕ+ :=
  {n : ℕ+ | ∃ (a b : ℤ), (b ≠ 0) ∧ (n : ℤ) = a^2 + 2*b^2}

/-- Theorem: If p is a prime number and p^2 is in A, then p is in A -/
theorem prime_square_in_A_implies_prime_in_A (p : ℕ+) (hp : Nat.Prime p) 
    (h_p_sq : (p^2 : ℕ+) ∈ A) : p ∈ A := by
  sorry

end prime_square_in_A_implies_prime_in_A_l2020_202052


namespace bobs_weight_l2020_202008

theorem bobs_weight (j b : ℝ) : 
  j + b = 200 → 
  b - 3 * j = b / 4 → 
  b = 2400 / 14 := by
sorry

end bobs_weight_l2020_202008


namespace cupcakes_remaining_l2020_202077

def cupcake_problem (packages : ℕ) (cupcakes_per_package : ℕ) (eaten : ℕ) : Prop :=
  let total := packages * cupcakes_per_package
  let remaining := total - eaten
  remaining = 7

theorem cupcakes_remaining :
  cupcake_problem 3 4 5 :=
by
  sorry

end cupcakes_remaining_l2020_202077


namespace complex_number_location_l2020_202047

theorem complex_number_location (z : ℂ) (h : (1 : ℂ) + Complex.I = Complex.I / z) :
  0 < z.re ∧ 0 < z.im := by sorry

end complex_number_location_l2020_202047


namespace girls_in_college_l2020_202005

theorem girls_in_college (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) :
  total_students = 455 →
  ratio_boys = 8 →
  ratio_girls = 5 →
  ∃ (num_girls : ℕ), num_girls * (ratio_boys + ratio_girls) = total_students * ratio_girls ∧ num_girls = 175 :=
by
  sorry

end girls_in_college_l2020_202005


namespace latest_sixty_degree_time_l2020_202004

/-- Temperature model as a function of time -/
def T (t : ℝ) : ℝ := -2 * t^2 + 16 * t + 40

/-- The statement to prove -/
theorem latest_sixty_degree_time :
  ∃ t_max : ℝ, t_max = 5 ∧ 
  T t_max = 60 ∧ 
  ∀ t : ℝ, T t = 60 → t ≤ t_max :=
sorry

end latest_sixty_degree_time_l2020_202004


namespace transylvanian_identity_l2020_202094

-- Define the possible states of being
inductive State
| Human
| Vampire

-- Define the possible states of mind
inductive Mind
| Sane
| Insane

-- Define a person as a combination of state and mind
structure Person :=
  (state : State)
  (mind : Mind)

-- Define the statement made by the Transylvanian
def transylvanian_statement (p : Person) : Prop :=
  p.state = State.Human ∨ p.mind = Mind.Sane

-- Define the condition that insane vampires only make true statements
axiom insane_vampire_truth (p : Person) :
  p.state = State.Vampire ∧ p.mind = Mind.Insane → transylvanian_statement p

-- Theorem: The Transylvanian must be a human and sane
theorem transylvanian_identity :
  ∃ (p : Person), p.state = State.Human ∧ p.mind = Mind.Sane ∧ transylvanian_statement p :=
by sorry

end transylvanian_identity_l2020_202094


namespace not_perfect_square_l2020_202064

theorem not_perfect_square (m n : ℕ) : ¬ ∃ k : ℕ, 1 + 3^m + 3^n = k^2 := by
  sorry

end not_perfect_square_l2020_202064


namespace find_x_l2020_202002

theorem find_x : ∃ x : ℝ, 3 * x = (26 - x) + 26 ∧ x = 13 := by sorry

end find_x_l2020_202002


namespace valuable_heirlooms_percentage_l2020_202054

theorem valuable_heirlooms_percentage
  (useful_percentage : Real)
  (junk_percentage : Real)
  (useful_items : Nat)
  (junk_items : Nat)
  (h1 : useful_percentage = 0.2)
  (h2 : junk_percentage = 0.7)
  (h3 : useful_items = 8)
  (h4 : junk_items = 28) :
  ∃ (total_items : Nat),
    (useful_items : Real) / total_items = useful_percentage ∧
    (junk_items : Real) / total_items = junk_percentage ∧
    1 - useful_percentage - junk_percentage = 0.1 := by
  sorry

end valuable_heirlooms_percentage_l2020_202054


namespace aristocrat_spending_l2020_202086

theorem aristocrat_spending (total_people : ℕ) (men_amount : ℕ) (women_amount : ℕ)
  (men_fraction : ℚ) (women_fraction : ℚ) :
  total_people = 3552 →
  men_amount = 45 →
  women_amount = 60 →
  men_fraction = 1/9 →
  women_fraction = 1/12 →
  ∃ (men women : ℕ),
    men + women = total_people ∧
    (men_fraction * men * men_amount + women_fraction * women * women_amount : ℚ) = 17760 :=
by sorry

end aristocrat_spending_l2020_202086


namespace field_trip_students_l2020_202099

theorem field_trip_students (teachers : ℕ) (student_ticket_cost adult_ticket_cost total_cost : ℚ) :
  teachers = 4 →
  student_ticket_cost = 1 →
  adult_ticket_cost = 3 →
  total_cost = 24 →
  ∃ (students : ℕ), students * student_ticket_cost + teachers * adult_ticket_cost = total_cost ∧ students = 12 :=
by sorry

end field_trip_students_l2020_202099


namespace some_number_solution_l2020_202028

theorem some_number_solution : 
  ∃ x : ℝ, 4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470 ∧ x = 13.26 := by
  sorry

end some_number_solution_l2020_202028


namespace like_terms_sum_of_exponents_l2020_202089

/-- Given two terms 5a^m * b^4 and -4a^3 * b^(n+2) are like terms, prove that m + n = 5 -/
theorem like_terms_sum_of_exponents (m n : ℕ) : 
  (∃ (a b : ℝ), 5 * a^m * b^4 = -4 * a^3 * b^(n+2)) → m + n = 5 := by
  sorry

end like_terms_sum_of_exponents_l2020_202089


namespace remainder_theorem_l2020_202079

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 60 * k - 3) :
  (n^3 + 2*n^2 + 3*n + 4) % 60 = 46 := by
  sorry

end remainder_theorem_l2020_202079


namespace tire_repair_cost_is_seven_l2020_202048

/-- The cost of repairing one tire without sales tax, given the total cost for 4 tires and the sales tax per tire. -/
def tire_repair_cost (total_cost : ℚ) (sales_tax : ℚ) : ℚ :=
  (total_cost - 4 * sales_tax) / 4

/-- Theorem stating that the cost of repairing one tire without sales tax is $7,
    given a total cost of $30 for 4 tires and a sales tax of $0.50 per tire. -/
theorem tire_repair_cost_is_seven :
  tire_repair_cost 30 0.5 = 7 := by
  sorry

end tire_repair_cost_is_seven_l2020_202048


namespace sqrt_sum_difference_l2020_202074

theorem sqrt_sum_difference (x : ℝ) : 
  Real.sqrt 8 + Real.sqrt 18 - 4 * Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_sum_difference_l2020_202074


namespace equal_angles_l2020_202007

-- Define the basic structures
variable (Circle₁ Circle₂ : Set (ℝ × ℝ))
variable (K M A B C D : ℝ × ℝ)

-- Define the conditions
variable (h1 : K ∈ Circle₁ ∩ Circle₂)
variable (h2 : M ∈ Circle₁ ∩ Circle₂)
variable (h3 : A ∈ Circle₁)
variable (h4 : B ∈ Circle₂)
variable (h5 : C ∈ Circle₁)
variable (h6 : D ∈ Circle₂)
variable (h7 : ∃ ray₁ : Set (ℝ × ℝ), K ∈ ray₁ ∧ A ∈ ray₁ ∧ B ∈ ray₁)
variable (h8 : ∃ ray₂ : Set (ℝ × ℝ), K ∈ ray₂ ∧ C ∈ ray₂ ∧ D ∈ ray₂)

-- Define the angle function
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem equal_angles : angle M A B = angle M C D := sorry

end equal_angles_l2020_202007


namespace complex_equation_solution_l2020_202056

theorem complex_equation_solution (z : ℂ) : z + Complex.abs z * Complex.I = 3 + 9 * Complex.I → z = 3 + 4 * Complex.I :=
by sorry

end complex_equation_solution_l2020_202056


namespace volume_three_triangular_pyramids_l2020_202012

/-- The volume of three identical triangular pyramids -/
theorem volume_three_triangular_pyramids 
  (base_measurement : ℝ) 
  (base_height : ℝ) 
  (pyramid_height : ℝ) 
  (h1 : base_measurement = 40) 
  (h2 : base_height = 20) 
  (h3 : pyramid_height = 30) : 
  3 * (1/3 * (1/2 * base_measurement * base_height) * pyramid_height) = 12000 := by
  sorry

#check volume_three_triangular_pyramids

end volume_three_triangular_pyramids_l2020_202012
