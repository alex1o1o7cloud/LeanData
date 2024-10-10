import Mathlib

namespace base_2_digit_difference_l359_35972

theorem base_2_digit_difference : ∀ (n m : ℕ), n = 300 → m = 1500 → 
  (Nat.log 2 m + 1) - (Nat.log 2 n + 1) = 2 := by
  sorry

end base_2_digit_difference_l359_35972


namespace horizontal_line_slope_l359_35931

/-- The slope of a horizontal line y + 3 = 0 is 0 -/
theorem horizontal_line_slope (x y : ℝ) : y + 3 = 0 → (∀ x₁ x₂, x₁ ≠ x₂ → (y - y) / (x₁ - x₂) = 0) := by
  sorry

end horizontal_line_slope_l359_35931


namespace alexa_lemonade_profit_l359_35989

/-- Calculates the profit from a lemonade stand given the price per cup,
    cost of ingredients, and number of cups sold. -/
def lemonade_profit (price_per_cup : ℕ) (ingredient_cost : ℕ) (cups_sold : ℕ) : ℕ :=
  price_per_cup * cups_sold - ingredient_cost

/-- Proves that given the specific conditions of Alexa's lemonade stand,
    her desired profit is $80. -/
theorem alexa_lemonade_profit :
  lemonade_profit 2 20 50 = 80 := by
  sorry

end alexa_lemonade_profit_l359_35989


namespace smaller_root_of_equation_l359_35910

theorem smaller_root_of_equation :
  let f : ℝ → ℝ := λ x => (x - 1/3)^2 + (x - 1/3)*(x + 1/6)
  (f (1/12) = 0) ∧ (∀ y < 1/12, f y ≠ 0) :=
by sorry

end smaller_root_of_equation_l359_35910


namespace octagon_side_length_l359_35982

/-- Given a regular pentagon with side length 16 cm, prove that if the same total length of yarn
    is used to make a regular octagon, then the length of one side of the octagon is 10 cm. -/
theorem octagon_side_length (pentagon_side : ℝ) (octagon_side : ℝ) : 
  pentagon_side = 16 → 5 * pentagon_side = 8 * octagon_side → octagon_side = 10 := by
  sorry

end octagon_side_length_l359_35982


namespace angle_d_is_190_l359_35912

/-- A quadrilateral with angles A, B, C, and D. -/
structure Quadrilateral where
  angleA : Real
  angleB : Real
  angleC : Real
  angleD : Real
  sum_360 : angleA + angleB + angleC + angleD = 360

/-- Theorem: In a quadrilateral ABCD, if ∠A = 70°, ∠B = 60°, and ∠C = 40°, then ∠D = 190°. -/
theorem angle_d_is_190 (q : Quadrilateral) 
  (hA : q.angleA = 70)
  (hB : q.angleB = 60)
  (hC : q.angleC = 40) : 
  q.angleD = 190 := by
  sorry

end angle_d_is_190_l359_35912


namespace polynomial_division_theorem_l359_35904

theorem polynomial_division_theorem (x : ℝ) : 
  ∃ (q r : ℝ → ℝ), 
    (x^4 - 3*x^2 + 1 = (x^2 - x + 1) * q x + r x) ∧ 
    (∀ y, r y = -3*y^2 + y + 1) ∧
    (∀ z, z^2 - z + 1 = 0 → r z = 0) :=
by sorry

end polynomial_division_theorem_l359_35904


namespace cosine_identity_from_system_l359_35905

theorem cosine_identity_from_system (A B C a b c : ℝ) 
  (eq1 : a = b * Real.cos C + c * Real.cos B)
  (eq2 : b = c * Real.cos A + a * Real.cos C)
  (eq3 : c = a * Real.cos B + b * Real.cos A)
  (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 + 2 * Real.cos A * Real.cos B * Real.cos C = 1 := by
sorry

end cosine_identity_from_system_l359_35905


namespace sunday_letters_zero_l359_35992

/-- Represents the number of letters written on each day of the week -/
structure WeeklyLetters where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- The average number of letters written per day -/
def averageLettersPerDay : ℕ := 9

/-- The total number of days in a week -/
def daysInWeek : ℕ := 7

/-- Calculates the total number of letters written in a week -/
def totalLetters (w : WeeklyLetters) : ℕ :=
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday

/-- States that the total number of letters written in a week equals the average per day times the number of days -/
axiom total_letters_axiom (w : WeeklyLetters) :
  totalLetters w = averageLettersPerDay * daysInWeek

/-- Defines the known number of letters written on specific days -/
def knownLetters (w : WeeklyLetters) : Prop :=
  w.wednesday ≥ 13 ∧ w.thursday ≥ 12 ∧ w.friday ≥ 9 ∧ w.saturday ≥ 7

/-- Theorem stating that given the conditions, the number of letters written on Sunday must be zero -/
theorem sunday_letters_zero (w : WeeklyLetters) 
  (h : knownLetters w) : w.sunday = 0 := by
  sorry


end sunday_letters_zero_l359_35992


namespace problem_1_problem_2_l359_35942

variable (a b : ℝ)

theorem problem_1 : (a - b)^2 - (2*a + b)*(b - 2*a) = 5*a^2 - 2*a*b := by sorry

theorem problem_2 : (3 / (a + 1) - a + 1) / ((a^2 + 4*a + 4) / (a + 1)) = (2 - a) / (a + 2) := by sorry

end problem_1_problem_2_l359_35942


namespace moon_distance_scientific_notation_l359_35940

def moon_distance : ℝ := 384000

theorem moon_distance_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), moon_distance = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.84 ∧ n = 5 := by
  sorry

end moon_distance_scientific_notation_l359_35940


namespace restaurant_production_l359_35974

/-- Represents a restaurant's daily production of pizzas and hot dogs -/
structure Restaurant where
  hotdogs : ℕ
  pizza_excess : ℕ

/-- Calculates the total number of pizzas and hot dogs made in a given number of days -/
def total_production (r : Restaurant) (days : ℕ) : ℕ :=
  (r.hotdogs + (r.hotdogs + r.pizza_excess)) * days

/-- Theorem stating that a restaurant making 40 more pizzas than hot dogs daily,
    and 60 hot dogs per day, will produce 4800 pizzas and hot dogs in 30 days -/
theorem restaurant_production :
  ∀ (r : Restaurant),
    r.hotdogs = 60 →
    r.pizza_excess = 40 →
    total_production r 30 = 4800 :=
by
  sorry

end restaurant_production_l359_35974


namespace hemisphere_surface_area_l359_35952

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 3) : 2 * π * r^2 + π * r^2 = 9 := by
  sorry

end hemisphere_surface_area_l359_35952


namespace interest_rate_is_six_percent_l359_35911

/-- Calculates the simple interest rate given the principal, final amount, and time period. -/
def simple_interest_rate (principal : ℚ) (final_amount : ℚ) (time : ℚ) : ℚ :=
  ((final_amount - principal) * 100) / (principal * time)

/-- Theorem stating that for the given conditions, the simple interest rate is 6% -/
theorem interest_rate_is_six_percent :
  simple_interest_rate 12500 15500 4 = 6 := by
  sorry

#eval simple_interest_rate 12500 15500 4

end interest_rate_is_six_percent_l359_35911


namespace star_seven_two_l359_35953

def star (a b : ℤ) : ℤ := 4 * a - 4 * b

theorem star_seven_two : star 7 2 = 20 := by
  sorry

end star_seven_two_l359_35953


namespace circle_equation_l359_35927

/-- Given a circle with center (a, -2a) passing through (2, -1) and tangent to x + y = 1,
    prove its equation is (x-1)^2 + (y+2)^2 = 2 -/
theorem circle_equation (a : ℝ) :
  (∀ x y : ℝ, y = -2 * x → (x - a)^2 + (y + 2*a)^2 = (2 - a)^2 + (-1 + 2*a)^2) →
  (∀ x y : ℝ, x + y = 1 → ((x - a)^2 + (y + 2*a)^2).sqrt = |x - a + y + 2*a| / Real.sqrt 2) →
  (∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2) :=
by sorry

end circle_equation_l359_35927


namespace alex_score_l359_35969

theorem alex_score (total_students : ℕ) (graded_students : ℕ) (initial_average : ℚ) (final_average : ℚ) :
  total_students = 20 →
  graded_students = 19 →
  initial_average = 72 →
  final_average = 74 →
  (graded_students * initial_average + (total_students - graded_students) * 
    ((total_students * final_average - graded_students * initial_average) / (total_students - graded_students))) / total_students = final_average →
  (total_students * final_average - graded_students * initial_average) = 112 := by
  sorry

end alex_score_l359_35969


namespace inequality_proof_l359_35925

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧ 
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end inequality_proof_l359_35925


namespace horizontal_arrangement_possible_l359_35916

/-- Represents a chessboard with 65 cells -/
structure ExtendedChessboard :=
  (cells : Fin 65 → Bool)

/-- Represents a domino (1x2 rectangle) -/
structure Domino :=
  (start : Fin 65)
  (horizontal : Bool)

/-- Represents the state of the chessboard with dominos -/
structure BoardState :=
  (board : ExtendedChessboard)
  (dominos : Fin 32 → Domino)

/-- Checks if two cells are adjacent on the extended chessboard -/
def are_adjacent (a b : Fin 65) : Bool := sorry

/-- Checks if a domino placement is valid -/
def valid_domino_placement (board : ExtendedChessboard) (domino : Domino) : Prop := sorry

/-- Checks if all dominos are placed horizontally -/
def all_horizontal (state : BoardState) : Prop := sorry

/-- Represents a move of a domino to adjacent empty cells -/
def valid_move (state₁ state₂ : BoardState) : Prop := sorry

/-- Main theorem: It's always possible to arrange all dominos horizontally -/
theorem horizontal_arrangement_possible (initial_state : BoardState) : 
  (∀ d, valid_domino_placement initial_state.board (initial_state.dominos d)) → 
  ∃ final_state, (valid_move initial_state final_state ∧ all_horizontal final_state) := sorry

end horizontal_arrangement_possible_l359_35916


namespace jello_bathtub_cost_l359_35920

/-- Calculate the total cost of filling a bathtub with jello mix -/
theorem jello_bathtub_cost :
  let bathtub_capacity : ℝ := 6  -- cubic feet
  let cubic_foot_to_gallon : ℝ := 7.5
  let gallon_weight : ℝ := 8  -- pounds
  let jello_per_pound : ℝ := 1.5  -- tablespoons
  let red_jello_cost : ℝ := 0.5  -- dollars per tablespoon
  let blue_jello_cost : ℝ := 0.4  -- dollars per tablespoon
  let green_jello_cost : ℝ := 0.6  -- dollars per tablespoon
  let red_jello_ratio : ℝ := 0.6
  let blue_jello_ratio : ℝ := 0.3
  let green_jello_ratio : ℝ := 0.1

  let total_water_weight := bathtub_capacity * cubic_foot_to_gallon * gallon_weight
  let total_jello_needed := total_water_weight * jello_per_pound
  let red_jello_amount := total_jello_needed * red_jello_ratio
  let blue_jello_amount := total_jello_needed * blue_jello_ratio
  let green_jello_amount := total_jello_needed * green_jello_ratio

  let total_cost := red_jello_amount * red_jello_cost +
                    blue_jello_amount * blue_jello_cost +
                    green_jello_amount * green_jello_cost

  total_cost = 259.2 := by sorry

end jello_bathtub_cost_l359_35920


namespace base_b_cube_iff_six_l359_35996

/-- Represents a number in base b --/
def base_b_number (b : ℕ) : ℕ := b^2 + 4*b + 4

/-- Checks if a natural number is a perfect cube --/
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

/-- The main theorem: 144 in base b is a cube iff b = 6 --/
theorem base_b_cube_iff_six (b : ℕ) : 
  (b > 0) → (is_cube (base_b_number b) ↔ b = 6) := by
sorry

end base_b_cube_iff_six_l359_35996


namespace perpendicular_equal_magnitude_vectors_l359_35985

/-- Given two vectors m and n in ℝ², prove that if n is obtained by swapping and negating one component of m, then m and n are perpendicular and have equal magnitudes. -/
theorem perpendicular_equal_magnitude_vectors
  (a b : ℝ) :
  let m : ℝ × ℝ := (a, b)
  let n : ℝ × ℝ := (b, -a)
  (m.1 * n.1 + m.2 * n.2 = 0) ∧ 
  (m.1^2 + m.2^2 = n.1^2 + n.2^2) :=
by sorry

end perpendicular_equal_magnitude_vectors_l359_35985


namespace marble_count_l359_35928

theorem marble_count (white purple red blue green total : ℕ) : 
  white + purple + red + blue + green = total →
  2 * purple = 3 * white →
  5 * white = 2 * red →
  2 * blue = white →
  3 * green = white →
  blue = 24 →
  total = 120 := by
sorry

end marble_count_l359_35928


namespace cubic_roots_sum_l359_35967

theorem cubic_roots_sum (p q r : ℝ) : 
  (6 * p^3 + 500 * p + 1234 = 0) → 
  (6 * q^3 + 500 * q + 1234 = 0) → 
  (6 * r^3 + 500 * r + 1234 = 0) → 
  (p + q)^3 + (q + r)^3 + (r + p)^3 + 100 = 717 := by
sorry

end cubic_roots_sum_l359_35967


namespace system_solution_l359_35987

theorem system_solution : 
  let x : ℚ := -135/41
  let y : ℚ := 192/41
  (7 * x = -9 - 3 * y) ∧ (2 * x = 5 * y - 30) := by
  sorry

end system_solution_l359_35987


namespace recliner_sales_increase_l359_35946

theorem recliner_sales_increase 
  (price_decrease : ℝ) 
  (gross_increase : ℝ) 
  (h1 : price_decrease = 0.20) 
  (h2 : gross_increase = 0.20000000000000014) : 
  (1 + gross_increase) / (1 - price_decrease) - 1 = 0.5 := by sorry

end recliner_sales_increase_l359_35946


namespace taller_tree_height_l359_35973

/-- Given two trees where one is 20 feet taller than the other and their heights
    are in the ratio 2:3, prove that the height of the taller tree is 60 feet. -/
theorem taller_tree_height (h : ℝ) (h_pos : h > 0) : 
  (h - 20) / h = 2 / 3 → h = 60 := by
  sorry

end taller_tree_height_l359_35973


namespace percentage_difference_in_gain_l359_35915

/-- Given an article with cost price, and two selling prices, calculate the percentage difference in gain -/
theorem percentage_difference_in_gain 
  (cost_price : ℝ) 
  (selling_price1 : ℝ) 
  (selling_price2 : ℝ) 
  (h1 : cost_price = 250) 
  (h2 : selling_price1 = 350) 
  (h3 : selling_price2 = 340) : 
  (selling_price1 - cost_price - (selling_price2 - cost_price)) / (selling_price2 - cost_price) * 100 = 100 / 9 := by
sorry

end percentage_difference_in_gain_l359_35915


namespace no_nonneg_int_solutions_l359_35968

theorem no_nonneg_int_solutions :
  ¬∃ (x₁ x₂ : ℕ), 96 * x₁ + 97 * x₂ = 1000 := by
  sorry

end no_nonneg_int_solutions_l359_35968


namespace value_of_a_l359_35984

theorem value_of_a (a : ℝ) : (0.005 * a = 0.75) → a = 150 := by
  sorry

end value_of_a_l359_35984


namespace unique_solution_proof_l359_35993

/-- The positive value of k for which the equation 4x^2 + kx + 4 = 0 has exactly one solution -/
def unique_solution_k : ℝ := 8

theorem unique_solution_proof :
  ∃! (k : ℝ), k > 0 ∧
  (∃! (x : ℝ), 4 * x^2 + k * x + 4 = 0) :=
by sorry

end unique_solution_proof_l359_35993


namespace rhombus_property_l359_35961

structure Rhombus (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)
  (is_rhombus : (B - A) = (C - B) ∧ (C - B) = (D - C) ∧ (D - C) = (A - D))

theorem rhombus_property {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (ABCD : Rhombus V) (E F P Q : V) :
  (∃ t : ℝ, E = ABCD.A + t • (ABCD.B - ABCD.A)) →
  (∃ s : ℝ, F = ABCD.A + s • (ABCD.D - ABCD.A)) →
  (ABCD.A - E = ABCD.D - F) →
  (∃ u : ℝ, P = ABCD.B + u • (ABCD.C - ABCD.B)) →
  (∃ v : ℝ, P = ABCD.D + v • (E - ABCD.D)) →
  (∃ w : ℝ, Q = ABCD.C + w • (ABCD.D - ABCD.C)) →
  (∃ x : ℝ, Q = ABCD.B + x • (F - ABCD.B)) →
  (∃ y z : ℝ, P - E = y • (P - ABCD.D) ∧ Q - F = z • (Q - ABCD.B) ∧ y + z = 1) ∧
  (∃ a : ℝ, ABCD.A - P = a • (Q - P)) :=
sorry

end rhombus_property_l359_35961


namespace stock_investment_change_l359_35918

theorem stock_investment_change (x : ℝ) (x_pos : x > 0) : 
  x * (1 + 0.75) * (1 - 0.30) = 1.225 * x := by
  sorry

#check stock_investment_change

end stock_investment_change_l359_35918


namespace bank_layoff_optimization_l359_35958

/-- Represents the economic benefit function for the bank -/
def economic_benefit (x : ℕ) : ℚ :=
  (320 - x) * (20 + 0.2 * x) - 6 * x

/-- Represents the constraint on the number of employees that can be laid off -/
def valid_layoff (x : ℕ) : Prop :=
  x ≤ 80

theorem bank_layoff_optimization :
  ∃ (x : ℕ), valid_layoff x ∧
    (∀ (y : ℕ), valid_layoff y → economic_benefit x ≥ economic_benefit y) ∧
    economic_benefit x = 9160 :=
sorry

end bank_layoff_optimization_l359_35958


namespace tyrone_eric_marbles_l359_35902

theorem tyrone_eric_marbles (tyrone_initial : ℕ) (eric_initial : ℕ) 
  (h1 : tyrone_initial = 97) 
  (h2 : eric_initial = 11) : 
  ∃ (marbles_given : ℕ), 
    marbles_given = 25 ∧ 
    (tyrone_initial - marbles_given = 2 * (eric_initial + marbles_given)) := by
  sorry

end tyrone_eric_marbles_l359_35902


namespace rachel_age_when_emily_half_l359_35957

theorem rachel_age_when_emily_half (emily_age rachel_age : ℕ) : 
  rachel_age = emily_age + 4 → 
  ∃ (x : ℕ), x = rachel_age ∧ x / 2 = x - 4 → 
  x = 8 := by sorry

end rachel_age_when_emily_half_l359_35957


namespace sin_120_degrees_l359_35934

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l359_35934


namespace g_13_equals_218_l359_35963

-- Define the function g
def g (n : ℕ) : ℕ := n^2 + 2*n + 23

-- State the theorem
theorem g_13_equals_218 : g 13 = 218 := by
  sorry

end g_13_equals_218_l359_35963


namespace odd_operations_l359_35933

theorem odd_operations (a b : ℤ) (h_even : Even a) (h_odd : Odd b) :
  Odd (a + b) ∧ Odd (a - b) ∧ Odd ((a + b)^2) ∧ 
  ¬(∀ (a b : ℤ), Even a → Odd b → Odd (a * b)) ∧
  ¬(∀ (a b : ℤ), Even a → Odd b → Odd ((a + b) / 2)) :=
by sorry

end odd_operations_l359_35933


namespace min_max_y_sum_l359_35951

theorem min_max_y_sum (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 5) 
  (h3 : x * z = 1) : 
  ∃ (m M : ℝ), (∀ y', x + y' + z = 3 ∧ x^2 + y'^2 + z^2 = 5 → m ≤ y' ∧ y' ≤ M) ∧ m = 0 ∧ M = 0 ∧ m + M = 0 := by
  sorry

end min_max_y_sum_l359_35951


namespace arithmetic_sequence_common_difference_l359_35978

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def common_difference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 105)
  (h_sum2 : a 2 + a 4 + a 6 = 99) :
  common_difference a = -2 := by
  sorry

end arithmetic_sequence_common_difference_l359_35978


namespace profit_of_c_l359_35900

def total_profit : ℕ := 56700
def ratio_a : ℕ := 8
def ratio_b : ℕ := 9
def ratio_c : ℕ := 10

theorem profit_of_c :
  let total_ratio := ratio_a + ratio_b + ratio_c
  let part_value := total_profit / total_ratio
  part_value * ratio_c = 21000 := by sorry

end profit_of_c_l359_35900


namespace x_to_y_equals_negative_eight_l359_35959

theorem x_to_y_equals_negative_eight (x y : ℝ) (h : |x + 2| + (y - 3)^2 = 0) : x^y = -8 := by
  sorry

end x_to_y_equals_negative_eight_l359_35959


namespace remainder_71_73_mod_9_l359_35906

theorem remainder_71_73_mod_9 : (71 * 73) % 9 = 8 := by
  sorry

end remainder_71_73_mod_9_l359_35906


namespace second_number_is_ninety_l359_35962

theorem second_number_is_ninety (x y z : ℝ) : 
  z = 4 * y →
  y = 2 * x →
  (x + y + z) / 3 = 165 →
  y = 90 := by
sorry

end second_number_is_ninety_l359_35962


namespace smallest_coin_set_l359_35990

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin in cents --/
def coinValue : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A function that checks if a given set of coins can pay any amount from 1 to n cents --/
def canPayAllAmounts (coins : List Coin) (n : ℕ) : Prop :=
  ∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ n →
    ∃ (subset : List Coin), subset ⊆ coins ∧ (subset.map coinValue).sum = amount

/-- The main theorem stating that 10 is the smallest number of coins needed --/
theorem smallest_coin_set :
  ∃ (coins : List Coin),
    coins.length = 10 ∧
    canPayAllAmounts coins 149 ∧
    ∀ (other_coins : List Coin),
      canPayAllAmounts other_coins 149 →
      other_coins.length ≥ 10 := by
  sorry

end smallest_coin_set_l359_35990


namespace parallel_vectors_imply_k_equals_five_l359_35903

/-- Given vectors in ℝ², prove that if (a - c) is parallel to b, then k = 5 --/
theorem parallel_vectors_imply_k_equals_five (a b c : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 3) →
  c = (k, 7) →
  ∃ (t : ℝ), (a.1 - c.1, a.2 - c.2) = (t * b.1, t * b.2) →
  k = 5 := by
sorry

end parallel_vectors_imply_k_equals_five_l359_35903


namespace unique_integer_satisfying_conditions_l359_35979

theorem unique_integer_satisfying_conditions :
  ∃! (x : ℤ), 1 < x ∧ x < 9 ∧ 2 < x ∧ x < 15 ∧ -1 < x ∧ x < 7 ∧ 0 < x ∧ x < 4 ∧ x + 1 < 5 ∧ x = 3 := by
  sorry

end unique_integer_satisfying_conditions_l359_35979


namespace unique_representation_theorem_l359_35977

theorem unique_representation_theorem (n : ℕ) :
  ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
sorry

end unique_representation_theorem_l359_35977


namespace diamond_equation_solution_l359_35917

def diamond (X Y : ℚ) : ℚ := 4 * X + 3 * Y + 7

theorem diamond_equation_solution :
  ∃! X : ℚ, diamond X 5 = 75 ∧ X = 53 / 4 :=
by
  sorry

end diamond_equation_solution_l359_35917


namespace difference_divisible_by_99_l359_35924

/-- Represents a three-digit number with digits a, b, and c -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : ℕ :=
  100 * n.a + 10 * n.b + n.c

/-- The value of a three-digit number with hundreds and units digits exchanged -/
def exchangedValue (n : ThreeDigitNumber) : ℕ :=
  100 * n.c + 10 * n.b + n.a

/-- The difference between the exchanged value and the original value -/
def difference (n : ThreeDigitNumber) : ℤ :=
  (exchangedValue n : ℤ) - (value n : ℤ)

/-- Theorem stating that the difference is always divisible by 99 -/
theorem difference_divisible_by_99 (n : ThreeDigitNumber) :
  ∃ k : ℤ, difference n = 99 * k := by
  sorry


end difference_divisible_by_99_l359_35924


namespace stratified_sample_size_l359_35908

/-- Represents a stratified sample from a population -/
structure StratifiedSample where
  teachers : ℕ
  male_students : ℕ
  female_students : ℕ
  sample_teachers : ℕ
  sample_male_students : ℕ
  sample_female_students : ℕ

/-- Calculates the total sample size -/
def total_sample_size (s : StratifiedSample) : ℕ :=
  s.sample_teachers + s.sample_male_students + s.sample_female_students

/-- Theorem: If 100 out of 800 male students are selected in a stratified sample
    from a population of 200 teachers, 800 male students, and 600 female students,
    then the total sample size is 200 -/
theorem stratified_sample_size
  (s : StratifiedSample)
  (h1 : s.teachers = 200)
  (h2 : s.male_students = 800)
  (h3 : s.female_students = 600)
  (h4 : s.sample_male_students = 100)
  (h5 : s.sample_teachers = s.teachers / 8)
  (h6 : s.sample_female_students = s.female_students / 8) :
  total_sample_size s = 200 := by
  sorry

end stratified_sample_size_l359_35908


namespace square_difference_equals_150_l359_35947

theorem square_difference_equals_150 : (15 + 5)^2 - (5^2 + 15^2) = 150 := by
  sorry

end square_difference_equals_150_l359_35947


namespace car_travel_distance_l359_35941

/-- Proves that a car traveling for 12 hours at 68 km/h covers 816 km -/
theorem car_travel_distance (travel_time : ℝ) (average_speed : ℝ) (h1 : travel_time = 12) (h2 : average_speed = 68) : travel_time * average_speed = 816 := by
  sorry

end car_travel_distance_l359_35941


namespace sufficient_not_necessary_condition_l359_35965

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b ∧ b > 0) := by
  sorry

end sufficient_not_necessary_condition_l359_35965


namespace trevors_brother_age_l359_35991

/-- Trevor's age a decade ago -/
def trevors_age_decade_ago : ℕ := 16

/-- Current year -/
def current_year : ℕ := 2023

/-- Trevor's current age -/
def trevors_current_age : ℕ := trevors_age_decade_ago + 10

/-- Trevor's age 20 years ago -/
def trevors_age_20_years_ago : ℕ := trevors_current_age - 20

/-- Trevor's brother's age 20 years ago -/
def brothers_age_20_years_ago : ℕ := 2 * trevors_age_20_years_ago

/-- Trevor's brother's current age -/
def brothers_current_age : ℕ := brothers_age_20_years_ago + 20

theorem trevors_brother_age : brothers_current_age = 32 := by
  sorry

end trevors_brother_age_l359_35991


namespace equation_solution_l359_35944

theorem equation_solution (x : ℝ) : 
  (x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2) ↔ (2*x + 1)^2 = 3 :=
by sorry

end equation_solution_l359_35944


namespace clothing_distribution_l359_35964

def total_clothing : ℕ := 135
def first_load : ℕ := 29
def num_small_loads : ℕ := 7

theorem clothing_distribution :
  (total_clothing - first_load) / num_small_loads = 15 :=
by sorry

end clothing_distribution_l359_35964


namespace set_operations_l359_35935

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 < 0}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

-- Theorem statement
theorem set_operations :
  (A ∩ B = B) ∧
  (B ⊆ A) ∧
  (A \ B = {x | x ≤ -1 ∨ (1 ≤ x ∧ x < 2)}) := by
  sorry

end set_operations_l359_35935


namespace seven_hash_three_l359_35939

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- Axioms for the # operation
axiom hash_zero (r : ℝ) : hash r 0 = r + 1
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 2) s = hash r s + s + 2

-- Theorem to prove
theorem seven_hash_three : hash 7 3 = 28 := by
  sorry

end seven_hash_three_l359_35939


namespace article_percentage_gain_l359_35922

/-- Calculates the percentage gain when selling an article --/
def percentage_gain (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating the percentage gain for the given problem --/
theorem article_percentage_gain :
  let cost_price : ℚ := 40
  let selling_price : ℚ := 350
  percentage_gain cost_price selling_price = 775 := by
  sorry

end article_percentage_gain_l359_35922


namespace kid_ticket_cost_prove_kid_ticket_cost_l359_35945

theorem kid_ticket_cost (adult_price : ℝ) (total_tickets : ℕ) (total_profit : ℝ) (kid_tickets : ℕ) : ℝ :=
  let adult_tickets := total_tickets - kid_tickets
  let adult_profit := adult_tickets * adult_price
  let kid_profit := total_profit - adult_profit
  let kid_price := kid_profit / kid_tickets
  kid_price

theorem prove_kid_ticket_cost :
  kid_ticket_cost 6 175 750 75 = 2 := by
  sorry

end kid_ticket_cost_prove_kid_ticket_cost_l359_35945


namespace angle_sum_pi_half_l359_35995

theorem angle_sum_pi_half (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π/2 := by sorry

end angle_sum_pi_half_l359_35995


namespace min_tetrahedra_decomposition_l359_35981

/-- A tetrahedron is a polyhedron with four triangular faces -/
structure Tetrahedron

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube

/-- Represents a decomposition of a cube into tetrahedra -/
structure CubeDecomposition (c : Cube) where
  tetrahedra : Finset Tetrahedron
  is_valid : Bool  -- This would be a complex condition in practice

/-- The number of tetrahedra in a decomposition -/
def num_tetrahedra (d : CubeDecomposition c) : Nat :=
  d.tetrahedra.card

/-- A predicate that checks if a decomposition is minimal -/
def is_minimal_decomposition (d : CubeDecomposition c) : Prop :=
  ∀ d' : CubeDecomposition c, num_tetrahedra d ≤ num_tetrahedra d'

theorem min_tetrahedra_decomposition (c : Cube) :
  ∃ (d : CubeDecomposition c), is_minimal_decomposition d ∧ num_tetrahedra d = 5 :=
sorry

end min_tetrahedra_decomposition_l359_35981


namespace least_number_with_divisibility_conditions_l359_35986

theorem least_number_with_divisibility_conditions : ∃ n : ℕ, 
  (∀ k : ℕ, 2 ≤ k → k ≤ 7 → n % k = 1) ∧ 
  (n % 8 = 0) ∧
  (∀ m : ℕ, m < n → ¬((∀ k : ℕ, 2 ≤ k → k ≤ 7 → m % k = 1) ∧ (m % 8 = 0))) ∧
  n = 1681 := by
sorry

end least_number_with_divisibility_conditions_l359_35986


namespace sum_of_fractions_inequality_l359_35954

theorem sum_of_fractions_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end sum_of_fractions_inequality_l359_35954


namespace coin_arrangement_coin_arrangement_proof_l359_35943

theorem coin_arrangement (total_coins : ℕ) (walls : ℕ) (coins_per_wall : ℕ → Prop) : Prop :=
  (total_coins = 12 ∧ walls = 4) →
  (∀ n, coins_per_wall n → n ≥ 2 ∧ n ≤ 6) →
  (∃! n, coins_per_wall n ∧ n * walls = total_coins)

-- The proof goes here
theorem coin_arrangement_proof : coin_arrangement 12 4 (λ n ↦ n = 3) := by
  sorry

end coin_arrangement_coin_arrangement_proof_l359_35943


namespace logan_driving_time_l359_35994

/-- Proves that Logan drove for 5 hours given the conditions of the problem -/
theorem logan_driving_time (tamika_speed : ℝ) (tamika_time : ℝ) (logan_speed : ℝ) (distance_difference : ℝ)
  (h_tamika_speed : tamika_speed = 45)
  (h_tamika_time : tamika_time = 8)
  (h_logan_speed : logan_speed = 55)
  (h_distance_difference : distance_difference = 85) :
  (tamika_speed * tamika_time - distance_difference) / logan_speed = 5 := by
sorry

end logan_driving_time_l359_35994


namespace total_age_now_l359_35998

-- Define Xavier's and Yasmin's ages as natural numbers
variable (xavier_age yasmin_age : ℕ)

-- Define the conditions
axiom xavier_twice_yasmin : xavier_age = 2 * yasmin_age
axiom xavier_future_age : xavier_age + 6 = 30

-- Theorem to prove
theorem total_age_now : xavier_age + yasmin_age = 36 := by
  sorry

end total_age_now_l359_35998


namespace a_range_when_f_decreasing_l359_35937

/-- A piecewise function f(x) defined based on the parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (a - 3) * x + 3 * a else Real.log x / Real.log a

/-- Theorem stating that if f is decreasing on ℝ, then a is in the open interval (3/4, 1) -/
theorem a_range_when_f_decreasing (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Set.Ioo (3/4) 1 := by
  sorry

end a_range_when_f_decreasing_l359_35937


namespace no_valid_base_for_122_square_l359_35948

theorem no_valid_base_for_122_square : ¬ ∃ (b : ℕ), b > 1 ∧ ∃ (n : ℕ), b^2 + 2*b + 2 = n^2 := by
  sorry

end no_valid_base_for_122_square_l359_35948


namespace arithmetic_sequence_common_difference_l359_35999

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms

/-- Properties of the arithmetic sequence -/
def ArithmeticSequenceProperties (seq : ArithmeticSequence) : Prop :=
  (seq.a 2 = 3) ∧ (seq.S 9 = 6 * seq.S 3)

/-- Theorem: The common difference of the arithmetic sequence is 1 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : ArithmeticSequenceProperties seq) :
  seq.d = 1 := by sorry

end arithmetic_sequence_common_difference_l359_35999


namespace product_xy_is_zero_l359_35913

theorem product_xy_is_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : x * y = 0 := by
  sorry

end product_xy_is_zero_l359_35913


namespace quadratic_root_sum_l359_35997

theorem quadratic_root_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - (2*m - 2)*x + (m^2 - 2*m) = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ = 10 →
  m = 6 := by sorry

end quadratic_root_sum_l359_35997


namespace tax_difference_proof_l359_35938

-- Define the item price and tax rates
def item_price : ℝ := 15
def tax_rate_1 : ℝ := 0.08
def tax_rate_2 : ℝ := 0.072
def discount_rate : ℝ := 0.005

-- Define the effective tax rate after discount
def effective_tax_rate : ℝ := tax_rate_2 - discount_rate

-- Theorem statement
theorem tax_difference_proof :
  (item_price * (1 + tax_rate_1)) - (item_price * (1 + effective_tax_rate)) = 0.195 := by
  sorry

end tax_difference_proof_l359_35938


namespace min_value_reciprocal_sum_l359_35930

theorem min_value_reciprocal_sum (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 3 / 2 := by
  sorry

end min_value_reciprocal_sum_l359_35930


namespace cubic_function_properties_l359_35988

-- Define the function f
def f (m n x : ℝ) : ℝ := m * x^3 + n * x^2

-- Define the derivative of f
def f' (m n x : ℝ) : ℝ := 3 * m * x^2 + 2 * n * x

-- Theorem statement
theorem cubic_function_properties (m : ℝ) (h : m ≠ 0) :
  ∃ n : ℝ,
    f' m n 2 = 0 ∧
    n = -3 * m ∧
    (∀ x : ℝ, m > 0 → (x < 0 ∨ x > 2) → (f' m n x > 0)) ∧
    (∀ x : ℝ, m < 0 → (x > 0 ∧ x < 2) → (f' m n x > 0)) :=
by sorry

end cubic_function_properties_l359_35988


namespace monkey_peaches_l359_35914

/-- Represents the number of peaches each monkey gets -/
structure MonkeyShares :=
  (eldest : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- The problem statement -/
theorem monkey_peaches (total : ℕ) (shares : MonkeyShares) : shares.second = 20 :=
  sorry

/-- Conditions of the problem -/
axiom divide_ratio (n m : ℕ) : n / (n + m) = 5 / 9
axiom eldest_share (total : ℕ) (shares : MonkeyShares) : shares.eldest = (total * 5) / 9
axiom second_share (total : ℕ) (shares : MonkeyShares) : 
  shares.second = ((total - shares.eldest) * 5) / 9
axiom third_share (total : ℕ) (shares : MonkeyShares) : 
  shares.third = total - shares.eldest - shares.second
axiom eldest_third_difference (shares : MonkeyShares) : shares.eldest - shares.third = 29

end monkey_peaches_l359_35914


namespace count_congruent_is_71_l359_35956

/-- The number of positive integers less than 500 that are congruent to 3 (mod 7) -/
def count_congruent : ℕ :=
  (Finset.filter (fun n => n % 7 = 3) (Finset.range 500)).card

/-- Theorem: The count of positive integers less than 500 that are congruent to 3 (mod 7) is 71 -/
theorem count_congruent_is_71 : count_congruent = 71 := by
  sorry

end count_congruent_is_71_l359_35956


namespace difference_of_squares_81_49_l359_35901

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end difference_of_squares_81_49_l359_35901


namespace number_division_problem_l359_35949

theorem number_division_problem (n : ℕ) : 
  n % 37 = 26 ∧ n / 37 = 2 → 48 - n / 4 = 23 := by
  sorry

end number_division_problem_l359_35949


namespace max_squares_covered_l359_35907

/-- Represents a square card with side length 2 inches -/
structure Card :=
  (side_length : ℝ)
  (h_side_length : side_length = 2)

/-- Represents a checkerboard with squares of side length 1 inch -/
structure Checkerboard :=
  (square_side_length : ℝ)
  (h_square_side_length : square_side_length = 1)

/-- The number of squares covered by the card on the checkerboard -/
def squares_covered (card : Card) (board : Checkerboard) : ℕ := sorry

/-- The theorem stating the maximum number of squares that can be covered -/
theorem max_squares_covered (card : Card) (board : Checkerboard) :
  ∃ (n : ℕ), squares_covered card board ≤ n ∧ n = 9 := by sorry

end max_squares_covered_l359_35907


namespace X_equals_Y_l359_35960

def X : Set ℝ := {x | ∃ n : ℤ, x = (2 * n + 1) * Real.pi}
def Y : Set ℝ := {y | ∃ k : ℤ, y = (4 * k + 1) * Real.pi ∨ y = (4 * k - 1) * Real.pi}

theorem X_equals_Y : X = Y := by sorry

end X_equals_Y_l359_35960


namespace right_triangle_log_identity_l359_35926

theorem right_triangle_log_identity 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle_inequality : c > b) :
  Real.log a / Real.log (b + c) + Real.log a / Real.log (c - b) = 
  2 * (Real.log a / Real.log (c + b)) * (Real.log a / Real.log (c - b)) := by
  sorry

end right_triangle_log_identity_l359_35926


namespace quadratic_inequality_solution_sets_l359_35983

theorem quadratic_inequality_solution_sets (a b c : ℝ) :
  (∀ x, ax^2 - b*x + c ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2) →
  (∀ x, c*x^2 + b*x + a ≤ 0 ↔ x ≤ -1 ∨ x ≥ -1/2) :=
by sorry

end quadratic_inequality_solution_sets_l359_35983


namespace f_equals_f_inv_at_zero_l359_35975

/-- The function f(x) = 3x^2 - 6x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

/-- The inverse function of f -/
noncomputable def f_inv (x : ℝ) : ℝ := 1 + Real.sqrt ((1 + x) / 3)

/-- Theorem stating that f(0) = f⁻¹(0) -/
theorem f_equals_f_inv_at_zero : f 0 = f_inv 0 := by
  sorry

end f_equals_f_inv_at_zero_l359_35975


namespace max_valid_sequence_length_l359_35923

/-- A sequence of natural numbers satisfying the given conditions -/
def ValidSequence (a : Fin k → ℕ) : Prop :=
  (∀ i j, i < j → a i < a j) ∧ 
  (∀ i, 1 ≤ a i ∧ a i ≤ 50) ∧
  (∀ i j, i ≠ j → ¬(7 ∣ (a i + a j)))

/-- The maximum length of a valid sequence -/
def MaxValidSequenceLength : ℕ := 23

theorem max_valid_sequence_length :
  (∃ (k : ℕ) (a : Fin k → ℕ), ValidSequence a ∧ k = MaxValidSequenceLength) ∧
  (∀ (k : ℕ) (a : Fin k → ℕ), ValidSequence a → k ≤ MaxValidSequenceLength) :=
sorry

end max_valid_sequence_length_l359_35923


namespace odd_function_property_l359_35909

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h_odd : IsOdd f) (h_sum : ∀ x, f (x + 1) + f x = 0) :
  f 5 = 0 := by
  sorry

end odd_function_property_l359_35909


namespace find_m_l359_35921

def A : Set ℕ := {1, 2, 3}
def B (m : ℕ) : Set ℕ := {2, m, 4}

theorem find_m : ∃ m : ℕ, (A ∩ B m = {2, 3}) → m = 3 := by
  sorry

end find_m_l359_35921


namespace power_of_product_l359_35950

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end power_of_product_l359_35950


namespace chef_total_plates_l359_35971

theorem chef_total_plates (lobster_rolls spicy_hot_noodles seafood_noodles : ℕ) 
  (h1 : lobster_rolls = 25)
  (h2 : spicy_hot_noodles = 14)
  (h3 : seafood_noodles = 16) :
  lobster_rolls + spicy_hot_noodles + seafood_noodles = 55 := by
  sorry

end chef_total_plates_l359_35971


namespace f_geq_a_iff_a_in_range_l359_35970

/-- The function f(x) defined as x^2 - 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

/-- The theorem stating the equivalence between the condition and the range of a -/
theorem f_geq_a_iff_a_in_range (a : ℝ) :
  (∀ x ≥ -1, f a x ≥ a) ↔ a ∈ Set.Icc (-3) 1 :=
by sorry

#check f_geq_a_iff_a_in_range

end f_geq_a_iff_a_in_range_l359_35970


namespace group_size_calculation_l359_35980

/-- Given a group of people where:
  1. The average weight increase is 1.5 kg
  2. The total weight increase is 12 kg (77 kg - 65 kg)
  3. The total weight increase equals the average weight increase multiplied by the number of people
  Prove that the number of people in the group is 8. -/
theorem group_size_calculation (avg_increase : ℝ) (total_increase : ℝ) :
  avg_increase = 1.5 →
  total_increase = 12 →
  total_increase = avg_increase * 8 :=
by sorry

end group_size_calculation_l359_35980


namespace max_gcd_13n_plus_4_7n_plus_2_l359_35966

theorem max_gcd_13n_plus_4_7n_plus_2 :
  (∃ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) = 2) ∧
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) ≤ 2) := by
  sorry

end max_gcd_13n_plus_4_7n_plus_2_l359_35966


namespace system_solution_l359_35976

theorem system_solution (x y z : ℤ) : 
  (x + y + z = 6 ∧ x + y * z = 7) ↔ 
  ((x, y, z) = (7, 0, -1) ∨ 
   (x, y, z) = (7, -1, 0) ∨ 
   (x, y, z) = (1, 3, 2) ∨ 
   (x, y, z) = (1, 2, 3)) :=
by sorry

end system_solution_l359_35976


namespace geometry_class_eligibility_l359_35929

def minimum_score (q1 q2 q3 : ℚ) : ℚ :=
  4 * (85 : ℚ) / 100 - (q1 + q2 + q3)

theorem geometry_class_eligibility 
  (q1 q2 q3 : ℚ) 
  (h1 : q1 = 85 / 100) 
  (h2 : q2 = 80 / 100) 
  (h3 : q3 = 90 / 100) : 
  minimum_score q1 q2 q3 = 85 / 100 := by
sorry

end geometry_class_eligibility_l359_35929


namespace total_pencils_l359_35936

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who bought color boxes -/
def number_of_people : ℕ := 3

/-- The number of pencils in each color box -/
def pencils_per_box : ℕ := rainbow_colors

theorem total_pencils :
  rainbow_colors * number_of_people = 21 :=
sorry

end total_pencils_l359_35936


namespace largest_n_satisfying_inequality_l359_35955

theorem largest_n_satisfying_inequality :
  ∀ n : ℕ, (1/4 : ℚ) + (n : ℚ)/8 < 2 ↔ n ≤ 13 :=
by sorry

end largest_n_satisfying_inequality_l359_35955


namespace cubic_equation_properties_l359_35932

theorem cubic_equation_properties :
  (∀ x y : ℕ, x^3 + 5*y = y^3 + 5*x → x = y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5*y = y^3 + 5*x) :=
by sorry

end cubic_equation_properties_l359_35932


namespace function_m_minus_n_l359_35919

def M (m : ℕ) : Set ℕ := {1, 2, 3, m}
def N (n : ℕ) : Set ℕ := {4, 7, n^4, n^2 + 3*n}

def f (x : ℕ) : ℕ := 3*x + 1

theorem function_m_minus_n (m n : ℕ) : 
  (∀ x ∈ M m, f x ∈ N n) → m - n = 3 := by
  sorry

end function_m_minus_n_l359_35919
