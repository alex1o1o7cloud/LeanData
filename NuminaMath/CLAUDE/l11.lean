import Mathlib

namespace compare_exponentials_l11_1190

theorem compare_exponentials (a b c : ℝ) : 
  a = (0.4 : ℝ) ^ (0.3 : ℝ) → 
  b = (0.3 : ℝ) ^ (0.4 : ℝ) → 
  c = (0.3 : ℝ) ^ (-(0.2 : ℝ)) → 
  b < a ∧ a < c :=
by sorry

end compare_exponentials_l11_1190


namespace vector_sum_problem_l11_1114

theorem vector_sum_problem :
  let v1 : Fin 2 → ℝ := ![5, -3]
  let v2 : Fin 2 → ℝ := ![-2, 4]
  (v1 + 3 • v2) = ![-1, 9] := by
  sorry

end vector_sum_problem_l11_1114


namespace geometric_sequence_fifth_term_l11_1108

theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ r : ℝ, ∀ n, a (n + 1) = r * a n)
  (h_third_term : a 3 = 9)
  (h_seventh_term : a 7 = 1) :
  a 5 = 3 := by sorry

end geometric_sequence_fifth_term_l11_1108


namespace consecutive_integers_sum_and_average_l11_1106

theorem consecutive_integers_sum_and_average (n : ℤ) :
  let consecutive_integers := [n+1, n+2, n+3, n+4, n+5, n+6]
  (consecutive_integers.sum = 6*n + 21) ∧ 
  (consecutive_integers.sum / 6 : ℚ) = n + (21 : ℚ) / 6 := by
sorry

end consecutive_integers_sum_and_average_l11_1106


namespace range_of_m_and_n_l11_1192

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + m > 0}
def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - n ≤ 0}

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- State the theorem
theorem range_of_m_and_n (m n : ℝ) : 
  P ∈ A m ∧ P ∉ B n → m > -1 ∧ n < 5 := by
  sorry


end range_of_m_and_n_l11_1192


namespace b_invests_after_six_months_l11_1170

/-- A partnership with three investors A, B, and C -/
structure Partnership where
  x : ℝ  -- A's investment
  m : ℝ  -- Months after which B invests
  total_gain : ℝ  -- Total annual gain
  a_share : ℝ  -- A's share of the gain

/-- The conditions of the partnership -/
def partnership_conditions (p : Partnership) : Prop :=
  p.total_gain = 24000 ∧ 
  p.a_share = 8000 ∧ 
  0 < p.x ∧ 
  0 < p.m ∧ 
  p.m < 12

/-- The theorem stating that B invests after 6 months -/
theorem b_invests_after_six_months (p : Partnership) 
  (h : partnership_conditions p) : p.m = 6 := by
  sorry


end b_invests_after_six_months_l11_1170


namespace annika_hiking_rate_l11_1112

/-- Annika's hiking problem -/
theorem annika_hiking_rate (initial_distance : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  initial_distance = 2.75 →
  total_distance = 3.5 →
  total_time = 51 →
  (total_time / (2 * (total_distance - initial_distance))) = 34 :=
by
  sorry

#check annika_hiking_rate

end annika_hiking_rate_l11_1112


namespace polynomial_B_value_l11_1121

def polynomial (z A B C D : ℤ) : ℤ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36

theorem polynomial_B_value :
  ∀ (A B C D : ℤ),
  (∀ r : ℤ, polynomial r A B C D = 0 → r > 0) →
  (∃ r1 r2 r3 r4 r5 r6 : ℤ, 
    ∀ z : ℤ, polynomial z A B C D = (z - r1) * (z - r2) * (z - r3) * (z - r4) * (z - r5) * (z - r6)) →
  B = -136 := by
sorry

end polynomial_B_value_l11_1121


namespace factorial_divisibility_l11_1157

theorem factorial_divisibility (a : ℕ) : 
  (a.factorial + (a + 2).factorial) ∣ (a + 4).factorial ↔ a = 0 ∨ a = 3 := by
  sorry

end factorial_divisibility_l11_1157


namespace unique_solution_l11_1116

def equation (x y : ℤ) : Prop := 3 * x + y = 10

theorem unique_solution :
  (equation 2 4) ∧
  ¬(equation 1 6) ∧
  ¬(equation (-2) 12) ∧
  ¬(equation (-1) 11) :=
sorry

end unique_solution_l11_1116


namespace total_marble_weight_l11_1176

def marble_weights : List Float := [
  0.3333333333333333,
  0.3333333333333333,
  0.08333333333333333,
  0.21666666666666667,
  0.4583333333333333,
  0.12777777777777778
]

theorem total_marble_weight :
  marble_weights.sum = 1.5527777777777777 := by sorry

end total_marble_weight_l11_1176


namespace noemi_initial_amount_l11_1135

/-- Calculates the initial amount of money Noemi had before gambling --/
def initial_amount (roulette_loss blackjack_loss poker_loss baccarat_loss purse_left : ℕ) : ℕ :=
  roulette_loss + blackjack_loss + poker_loss + baccarat_loss + purse_left

/-- Proves that Noemi's initial amount is correct given her losses and remaining money --/
theorem noemi_initial_amount : 
  initial_amount 600 800 400 700 1500 = 4000 := by
  sorry

end noemi_initial_amount_l11_1135


namespace charles_city_population_l11_1138

theorem charles_city_population (C G : ℕ) : 
  G + 119666 = C → 
  C + G = 845640 → 
  C = 482653 := by
sorry

end charles_city_population_l11_1138


namespace range_of_m_l11_1129

def A (m : ℝ) : Set ℝ := {x | x^2 + Real.sqrt m * x + 1 = 0}

theorem range_of_m (m : ℝ) : (A m ∩ Set.univ = ∅) → (0 ≤ m ∧ m < 4) :=
sorry

end range_of_m_l11_1129


namespace correct_num_arrangements_l11_1156

/-- The number of arrangements of 3 girls and 6 boys in a row, 
    with boys at both ends and no two girls adjacent. -/
def num_arrangements : ℕ := 43200

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The number of boys -/
def num_boys : ℕ := 6

/-- Theorem stating that the number of arrangements satisfying the given conditions is 43200 -/
theorem correct_num_arrangements : 
  (num_girls = 3 ∧ num_boys = 6) → num_arrangements = 43200 := by
  sorry

end correct_num_arrangements_l11_1156


namespace textbook_cost_ratio_l11_1132

/-- The ratio of the cost of bookstore textbooks to online ordered books -/
theorem textbook_cost_ratio : 
  ∀ (sale_price online_price bookstore_price total_price : ℕ),
  sale_price = 5 * 10 →
  online_price = 40 →
  total_price = 210 →
  bookstore_price = total_price - sale_price - online_price →
  ∃ (k : ℕ), bookstore_price = k * online_price →
  (bookstore_price : ℚ) / online_price = 3 := by
  sorry

end textbook_cost_ratio_l11_1132


namespace two_numbers_with_sum_and_gcd_lcm_sum_l11_1160

theorem two_numbers_with_sum_and_gcd_lcm_sum (a b : ℕ) : 
  a + b = 60 ∧ 
  Nat.gcd a b + Nat.lcm a b = 84 → 
  (a = 24 ∧ b = 36) ∨ (a = 36 ∧ b = 24) := by
sorry

end two_numbers_with_sum_and_gcd_lcm_sum_l11_1160


namespace paula_candy_distribution_l11_1146

def minimum_candies (initial_candies : ℕ) (num_friends : ℕ) : ℕ :=
  let total := initial_candies + (num_friends - initial_candies % num_friends) % num_friends
  total

theorem paula_candy_distribution (initial_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : num_friends = 10) :
  minimum_candies initial_candies num_friends = 30 ∧
  minimum_candies initial_candies num_friends / num_friends = 3 :=
by sorry

end paula_candy_distribution_l11_1146


namespace total_squares_blocked_5x6_grid_l11_1165

/-- Represents a partially blocked grid -/
structure PartialGrid :=
  (rows : Nat)
  (cols : Nat)
  (squares_1x1 : Nat)
  (squares_2x2 : Nat)
  (squares_3x3 : Nat)
  (squares_4x4 : Nat)

/-- Calculates the total number of squares in a partially blocked grid -/
def total_squares (grid : PartialGrid) : Nat :=
  grid.squares_1x1 + grid.squares_2x2 + grid.squares_3x3 + grid.squares_4x4

/-- Theorem: The total number of squares in the given partially blocked 5x6 grid is 57 -/
theorem total_squares_blocked_5x6_grid :
  ∃ (grid : PartialGrid),
    grid.rows = 5 ∧
    grid.cols = 6 ∧
    grid.squares_1x1 = 30 ∧
    grid.squares_2x2 = 18 ∧
    grid.squares_3x3 = 7 ∧
    grid.squares_4x4 = 2 ∧
    total_squares grid = 57 :=
by
  sorry

end total_squares_blocked_5x6_grid_l11_1165


namespace intersection_slope_l11_1115

/-- Given two lines p and q that intersect at (4, 11), 
    where p has equation y = 2x + 3 and q has equation y = mx + 1,
    prove that m = 2.5 -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 3 → y = m*x + 1 → x = 4 ∧ y = 11) →
  m = 2.5 := by
  sorry

end intersection_slope_l11_1115


namespace hyperbola_equation_l11_1158

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if its eccentricity is √3 and the directrix of the parabola y² = 12x
    passes through one of its foci, then the equation of the hyperbola is
    x²/3 - y²/6 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c / a = Real.sqrt 3 ∧ c = 3) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 / 6 = 1) :=
by sorry

end hyperbola_equation_l11_1158


namespace division_simplification_l11_1159

theorem division_simplification (x y : ℝ) : -6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y := by
  sorry

end division_simplification_l11_1159


namespace largest_angle_after_change_l11_1117

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ)

-- Define the initial conditions
def initial_triangle : Triangle :=
  { D := 60, E := 60, F := 60 }

-- Define the angle decrease
def angle_decrease : ℝ := 20

-- Theorem statement
theorem largest_angle_after_change (t : Triangle) :
  t = initial_triangle →
  ∃ (new_t : Triangle),
    new_t.D = t.D - angle_decrease ∧
    new_t.D + new_t.E + new_t.F = 180 ∧
    new_t.E = new_t.F ∧
    max new_t.D (max new_t.E new_t.F) = 70 := by
  sorry

end largest_angle_after_change_l11_1117


namespace largest_two_digit_product_l11_1128

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_single_digit (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 9

theorem largest_two_digit_product :
  ∃ (n x : ℕ), 
    is_two_digit n ∧
    is_single_digit x ∧
    n = x * (10 * x + 2 * x) ∧
    ∀ (m y : ℕ), 
      is_two_digit m → 
      is_single_digit y → 
      m = y * (10 * y + 2 * y) → 
      m ≤ n ∧
    n = 48 :=
sorry

end largest_two_digit_product_l11_1128


namespace license_plate_increase_l11_1184

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^3
  let new_plates := 26^4 * 10^2
  new_plates / old_plates = 26^2 / 10 := by
sorry

end license_plate_increase_l11_1184


namespace inequality_condition_l11_1167

theorem inequality_condition :
  (∀ a b c d : ℝ, (a > b ∧ c > d) → (a + c > b + d)) ∧
  (∃ a b c d : ℝ, (a + c > b + d) ∧ ¬(a > b ∧ c > d)) :=
sorry

end inequality_condition_l11_1167


namespace negative_inequality_l11_1102

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end negative_inequality_l11_1102


namespace student_grouping_l11_1147

theorem student_grouping (total_students : ℕ) (students_per_group : ℕ) (h1 : total_students = 30) (h2 : students_per_group = 5) :
  total_students / students_per_group = 6 := by
  sorry

end student_grouping_l11_1147


namespace triangle_equilateral_l11_1163

theorem triangle_equilateral (a b c : ℝ) (A B C : ℝ) :
  (a + b + c) * (b + c - a) = 3 * a * b * c →
  Real.sin A = 2 * Real.sin B * Real.cos C →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a = b ∧ b = c ∧ A = B ∧ B = C ∧ A = Real.pi / 3 :=
sorry

end triangle_equilateral_l11_1163


namespace eric_jogging_time_l11_1122

/-- Proves the time Eric spent jogging given his running time and return trip time -/
theorem eric_jogging_time 
  (total_time_to_park : ℕ) 
  (running_time : ℕ) 
  (jogging_time : ℕ) 
  (return_trip_time : ℕ) 
  (h1 : total_time_to_park = running_time + jogging_time)
  (h2 : running_time = 20)
  (h3 : return_trip_time = 90)
  (h4 : return_trip_time = 3 * total_time_to_park) :
  jogging_time = 10 := by
sorry

end eric_jogging_time_l11_1122


namespace basketball_probability_l11_1193

/-- The number of basketballs -/
def total_balls : ℕ := 8

/-- The number of new basketballs -/
def new_balls : ℕ := 4

/-- The number of old basketballs -/
def old_balls : ℕ := 4

/-- The number of balls selected in each training session -/
def selected_balls : ℕ := 2

/-- The probability of selecting exactly one new ball in the second training session -/
def prob_one_new_second : ℚ := 51 / 98

theorem basketball_probability :
  total_balls = new_balls + old_balls →
  prob_one_new_second = (
    (Nat.choose old_balls selected_balls * Nat.choose new_balls 1 * Nat.choose old_balls 1 +
     Nat.choose new_balls 1 * Nat.choose old_balls 1 * Nat.choose (new_balls - 1) 1 * Nat.choose (old_balls + 1) 1 +
     Nat.choose new_balls selected_balls * Nat.choose (new_balls - selected_balls) 1 * Nat.choose (old_balls + selected_balls) 1) /
    (Nat.choose total_balls selected_balls * Nat.choose total_balls selected_balls)
  ) := by sorry

end basketball_probability_l11_1193


namespace fence_posts_count_l11_1155

/-- Represents the fence setup for a rectangular field -/
structure FenceSetup where
  wallLength : ℕ
  rectLength : ℕ
  rectWidth : ℕ
  postSpacing : ℕ
  gateWidth : ℕ

/-- Calculates the number of posts required for the fence setup -/
def calculatePosts (setup : FenceSetup) : ℕ :=
  sorry

/-- Theorem stating that the specific fence setup requires 19 posts -/
theorem fence_posts_count :
  let setup : FenceSetup := {
    wallLength := 120,
    rectLength := 80,
    rectWidth := 50,
    postSpacing := 10,
    gateWidth := 20
  }
  calculatePosts setup = 19 := by
  sorry

end fence_posts_count_l11_1155


namespace not_lucky_1982_1983_l11_1182

/-- Checks if a given year is a lucky year -/
def isLuckyYear (year : Nat) : Prop :=
  ∃ (month day : Nat),
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

theorem not_lucky_1982_1983 :
  ¬(isLuckyYear 1982) ∧ ¬(isLuckyYear 1983) :=
by sorry

end not_lucky_1982_1983_l11_1182


namespace inequality_solution_l11_1139

def solution_set (a : ℝ) : Set ℝ :=
  if a < 1 then {x | x < a ∨ x > 1}
  else if a = 1 then {x | x ≠ 1}
  else {x | x < 1 ∨ x > a}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | x^2 - (a + 1) * x + a > 0} = solution_set a :=
sorry

end inequality_solution_l11_1139


namespace fraction_simplification_l11_1179

theorem fraction_simplification :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := by sorry

end fraction_simplification_l11_1179


namespace sum_due_theorem_l11_1173

/-- Calculates the sum due given the true discount, interest rate, and time period. -/
def sum_due (true_discount : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  let present_value := (true_discount * 100) / (interest_rate * time)
  present_value + true_discount

/-- Proves that the sum due is 568 given the specified conditions. -/
theorem sum_due_theorem :
  sum_due 168 14 3 = 568 := by
  sorry

end sum_due_theorem_l11_1173


namespace coin_game_probability_l11_1154

/-- Represents a player in the coin game -/
inductive Player : Type
| Abby : Player
| Bernardo : Player
| Carl : Player
| Debra : Player

/-- Represents a ball color in the game -/
inductive BallColor : Type
| Green : BallColor
| Red : BallColor
| Blue : BallColor

/-- The number of rounds in the game -/
def numRounds : Nat := 5

/-- The initial number of coins each player has -/
def initialCoins : Nat := 5

/-- The number of coins transferred when green and red balls are drawn -/
def coinTransfer : Nat := 2

/-- The total number of balls in the urn -/
def totalBalls : Nat := 5

/-- The number of green balls in the urn -/
def greenBalls : Nat := 1

/-- The number of red balls in the urn -/
def redBalls : Nat := 1

/-- The number of blue balls in the urn -/
def blueBalls : Nat := 3

/-- Represents the state of the game after each round -/
structure GameState :=
  (coins : Player → Nat)

/-- The probability of a specific pair (green/red) occurring in one round -/
def pairProbability : ℚ := 1 / 20

/-- 
Theorem: The probability that each player has exactly 5 coins after 5 rounds is 1/3,200,000
-/
theorem coin_game_probability : 
  ∀ (finalState : GameState),
    (∀ p : Player, finalState.coins p = initialCoins) →
    (pairProbability ^ numRounds : ℚ) = 1 / 3200000 := by
  sorry


end coin_game_probability_l11_1154


namespace pencil_price_l11_1161

theorem pencil_price (num_pens num_pencils total_spent pen_avg_price : ℝ) 
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_spent = 450)
  (h4 : pen_avg_price = 10) :
  (total_spent - num_pens * pen_avg_price) / num_pencils = 2 := by
  sorry

end pencil_price_l11_1161


namespace ball_attendees_l11_1151

theorem ball_attendees :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end ball_attendees_l11_1151


namespace binary_1010_is_10_l11_1178

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1010_is_10 :
  binary_to_decimal [false, true, false, true] = 10 := by
  sorry

end binary_1010_is_10_l11_1178


namespace function_properties_l11_1194

/-- Given functions f and g on ℝ satisfying certain properties, 
    prove specific characteristics of their derivatives. -/
theorem function_properties
  (f g : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f (-x + 2))
  (h2 : ∀ x, g (-x + 1) - 2 = -(g (x + 1) - 2))
  (h3 : ∀ x, f (3 - x) + g (x - 1) = 2) :
  (deriv f 2022 = 0) ∧
  (∀ x, deriv g (-x) = -(deriv g x)) :=
by sorry

end function_properties_l11_1194


namespace probability_is_175_323_l11_1169

-- Define the number of black and white balls
def black_balls : ℕ := 10
def white_balls : ℕ := 9
def total_balls : ℕ := black_balls + white_balls

-- Define the number of balls drawn
def drawn_balls : ℕ := 3

-- Define the probability function
def probability_at_least_two_black : ℚ :=
  (Nat.choose black_balls 2 * Nat.choose white_balls 1 +
   Nat.choose black_balls 3) /
  Nat.choose total_balls drawn_balls

-- Theorem statement
theorem probability_is_175_323 :
  probability_at_least_two_black = 175 / 323 :=
by sorry

end probability_is_175_323_l11_1169


namespace disjunction_true_false_l11_1137

theorem disjunction_true_false (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end disjunction_true_false_l11_1137


namespace tank_capacity_proof_l11_1143

/-- Represents the capacity of a tank with a leak and two inlet pipes. -/
def tank_capacity : Real :=
  let leak_rate : Real := tank_capacity / 6
  let pipe_a_rate : Real := 3.5 * 60
  let pipe_b_rate : Real := 4.5 * 60
  let net_rate_both_pipes : Real := pipe_a_rate + pipe_b_rate - leak_rate
  let net_rate_pipe_a : Real := pipe_a_rate - leak_rate
  tank_capacity

/-- Theorem stating the capacity of the tank under given conditions. -/
theorem tank_capacity_proof :
  let leak_rate : Real := tank_capacity / 6
  let pipe_a_rate : Real := 3.5 * 60
  let pipe_b_rate : Real := 4.5 * 60
  let net_rate_both_pipes : Real := pipe_a_rate + pipe_b_rate - leak_rate
  let net_rate_pipe_a : Real := pipe_a_rate - leak_rate
  (net_rate_pipe_a * 1 + net_rate_both_pipes * 7 - leak_rate * 8 = 0) →
  tank_capacity = 1338.75 := by
  sorry

#eval tank_capacity

end tank_capacity_proof_l11_1143


namespace quadratic_no_real_roots_l11_1124

theorem quadratic_no_real_roots (m : ℝ) :
  (∀ x : ℝ, x^2 - x + m ≠ 0) → m > 1/4 := by
  sorry

end quadratic_no_real_roots_l11_1124


namespace equation_solution_l11_1168

theorem equation_solution : ∃ X : ℝ, 
  (1.5 * ((X * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002) ∧ 
  (abs (X - 3.6000000000000005) < 1e-10) := by
  sorry

end equation_solution_l11_1168


namespace solution_set_of_increasing_function_l11_1127

/-- Given an increasing function f: ℝ → ℝ with f(0) = -1 and f(3) = 1,
    the solution set of |f(x+1)| < 1 is (-1, 2). -/
theorem solution_set_of_increasing_function (f : ℝ → ℝ) 
  (h_incr : StrictMono f) (h_f0 : f 0 = -1) (h_f3 : f 3 = 1) :
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by
  sorry

end solution_set_of_increasing_function_l11_1127


namespace sara_quarters_sum_l11_1183

-- Define the initial number of quarters Sara had
def initial_quarters : ℝ := 783.0

-- Define the number of quarters Sara's dad gave her
def dad_quarters : ℝ := 271.0

-- Define the total number of quarters Sara has now
def total_quarters : ℝ := initial_quarters + dad_quarters

-- Theorem to prove
theorem sara_quarters_sum :
  total_quarters = 1054.0 := by sorry

end sara_quarters_sum_l11_1183


namespace bianca_candy_eaten_l11_1134

theorem bianca_candy_eaten (total : ℕ) (piles : ℕ) (per_pile : ℕ) 
  (h1 : total = 78)
  (h2 : piles = 6)
  (h3 : per_pile = 8) :
  total - (piles * per_pile) = 30 := by
  sorry

end bianca_candy_eaten_l11_1134


namespace tangent_line_at_neg_one_max_value_on_interval_min_value_on_interval_l11_1162

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

-- Define the interval [0, 4]
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

-- Theorem for the tangent line equation at x = -1
theorem tangent_line_at_neg_one :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ 4 * x - y + 4 = 0 :=
sorry

-- Theorem for the maximum value of f(x) on the interval [0, 4]
theorem max_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 45 :=
sorry

-- Theorem for the minimum value of f(x) on the interval [0, 4]
theorem min_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = 0 :=
sorry

end tangent_line_at_neg_one_max_value_on_interval_min_value_on_interval_l11_1162


namespace unique_positive_solution_l11_1120

theorem unique_positive_solution (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  x^y = z ∧ y^z = x ∧ z^x = y →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end unique_positive_solution_l11_1120


namespace quadratic_inequality_range_l11_1131

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + 1 < 0) ↔ a < 1 := by
  sorry

end quadratic_inequality_range_l11_1131


namespace min_rows_required_l11_1181

/-- The number of seats in each row -/
def seats_per_row : ℕ := 168

/-- The total number of students -/
def total_students : ℕ := 2016

/-- The maximum number of students from each school -/
def max_students_per_school : ℕ := 40

/-- Represents the seating arrangement in the arena -/
structure Arena where
  rows : ℕ
  students_seated : ℕ
  school_integrity : Bool  -- True if students from each school are in a single row

/-- A function to check if a seating arrangement is valid -/
def is_valid_arrangement (a : Arena) : Prop :=
  a.students_seated = total_students ∧
  a.school_integrity ∧
  a.rows * seats_per_row ≥ total_students

/-- The main theorem stating the minimum number of rows required -/
theorem min_rows_required : 
  ∀ a : Arena, is_valid_arrangement a → a.rows ≥ 15 :=
sorry

end min_rows_required_l11_1181


namespace symmetric_points_imply_fourth_quadrant_l11_1119

/-- Given two points A and B symmetric with respect to the y-axis, 
    prove that point C lies in the fourth quadrant. -/
theorem symmetric_points_imply_fourth_quadrant 
  (a b : ℝ) 
  (h_symmetric : (a - 2, 3) = (-(-1), b + 5)) : 
  (a > 0 ∧ b < 0) := by
  sorry

end symmetric_points_imply_fourth_quadrant_l11_1119


namespace negation_of_universal_proposition_l11_1149

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x ≥ x + 1) ↔ (∃ x₀ : ℝ, Real.exp x₀ < x₀ + 1) := by
  sorry

end negation_of_universal_proposition_l11_1149


namespace number_of_coaches_l11_1180

theorem number_of_coaches (pouches_per_pack : ℕ) (packs_bought : ℕ) (team_members : ℕ) (helpers : ℕ) :
  pouches_per_pack = 6 →
  packs_bought = 3 →
  team_members = 13 →
  helpers = 2 →
  packs_bought * pouches_per_pack = team_members + helpers + 3 :=
by
  sorry

end number_of_coaches_l11_1180


namespace prob_third_grade_parent_is_three_fifths_l11_1150

/-- Represents the number of parents in each grade's committee -/
structure ParentCommittee where
  grade1 : Nat
  grade2 : Nat
  grade3 : Nat

/-- Represents the number of parents sampled from each grade -/
structure SampledParents where
  grade1 : Nat
  grade2 : Nat
  grade3 : Nat

/-- Calculates the total number of parents in all committees -/
def totalParents (pc : ParentCommittee) : Nat :=
  pc.grade1 + pc.grade2 + pc.grade3

/-- Calculates the stratified sample for each grade -/
def calculateSample (pc : ParentCommittee) (totalSample : Nat) : SampledParents :=
  let ratio := totalSample / (totalParents pc)
  { grade1 := pc.grade1 * ratio
  , grade2 := pc.grade2 * ratio
  , grade3 := pc.grade3 * ratio }

/-- Calculates the probability of selecting at least one third-grade parent -/
def probThirdGradeParent (sp : SampledParents) : Rat :=
  let totalCombinations := (sp.grade1 + sp.grade2 + sp.grade3).choose 2
  let favorableCombinations := sp.grade3 * (sp.grade1 + sp.grade2) + sp.grade3.choose 2
  favorableCombinations / totalCombinations

theorem prob_third_grade_parent_is_three_fifths 
  (pc : ParentCommittee) 
  (h1 : pc.grade1 = 54)
  (h2 : pc.grade2 = 18)
  (h3 : pc.grade3 = 36)
  (totalSample : Nat)
  (h4 : totalSample = 6) :
  probThirdGradeParent (calculateSample pc totalSample) = 3/5 := by
  sorry

end prob_third_grade_parent_is_three_fifths_l11_1150


namespace line_segment_slope_l11_1101

theorem line_segment_slope (m n p : ℝ) : 
  (m = 4 * n + 5) → 
  (m + 2 = 4 * (n + p) + 5) → 
  p = 1/2 := by
sorry

end line_segment_slope_l11_1101


namespace ferry_distance_ratio_l11_1152

/-- The ratio of distances covered by two ferries --/
theorem ferry_distance_ratio :
  let v_p : ℝ := 6  -- Speed of ferry P in km/h
  let t_p : ℝ := 3  -- Time taken by ferry P in hours
  let v_q : ℝ := v_p + 3  -- Speed of ferry Q in km/h
  let t_q : ℝ := t_p + 1  -- Time taken by ferry Q in hours
  let d_p : ℝ := v_p * t_p  -- Distance covered by ferry P
  let d_q : ℝ := v_q * t_q  -- Distance covered by ferry Q
  d_q / d_p = 2 :=
by sorry

end ferry_distance_ratio_l11_1152


namespace girls_in_choir_l11_1199

theorem girls_in_choir (orchestra_students band_students choir_students total_students boys_in_choir : ℕ)
  (h1 : orchestra_students = 20)
  (h2 : band_students = 2 * orchestra_students)
  (h3 : boys_in_choir = 12)
  (h4 : total_students = 88)
  (h5 : total_students = orchestra_students + band_students + choir_students) :
  choir_students - boys_in_choir = 16 := by
  sorry

end girls_in_choir_l11_1199


namespace tangent_equation_solutions_l11_1174

open Real

theorem tangent_equation_solutions (t : ℝ) :
  cos t ≠ 0 →
  (tan t = (sin t ^ 2 + sin (2 * t) - 1) / (cos t ^ 2 - sin (2 * t) + 1)) ↔
  (∃ k : ℤ, t = π / 4 + π * k) ∨
  (∃ n : ℤ, t = arctan ((1 - Real.sqrt 5) / 2) + π * n) ∨
  (∃ l : ℤ, t = arctan ((1 + Real.sqrt 5) / 2) + π * l) :=
by sorry

end tangent_equation_solutions_l11_1174


namespace cousins_ages_sum_l11_1126

/-- Given 4 non-negative integers representing ages, if their mean is 8 and
    their median is 5, then the sum of the smallest and largest of these
    integers is 22. -/
theorem cousins_ages_sum (a b c d : ℕ) : 
  a ≤ b ∧ b ≤ c ∧ c ≤ d →  -- Sorted in ascending order
  (a + b + c + d) / 4 = 8 →  -- Mean is 8
  (b + c) / 2 = 5 →  -- Median is 5
  a + d = 22 := by sorry

end cousins_ages_sum_l11_1126


namespace taco_truck_profit_l11_1111

/-- Calculates the profit for a taco truck given the specified conditions -/
theorem taco_truck_profit
  (total_beef : ℝ)
  (beef_per_taco : ℝ)
  (selling_price : ℝ)
  (cost_per_taco : ℝ)
  (h1 : total_beef = 100)
  (h2 : beef_per_taco = 0.25)
  (h3 : selling_price = 2)
  (h4 : cost_per_taco = 1.5) :
  (total_beef / beef_per_taco) * (selling_price - cost_per_taco) = 200 :=
by sorry

end taco_truck_profit_l11_1111


namespace least_possible_a_2000_l11_1171

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ m n, m ∣ n → m < n → a m ∣ a n ∧ a m < a n

theorem least_possible_a_2000 (a : ℕ → ℕ) (h : sequence_property a) : a 2000 ≥ 128 := by
  sorry

end least_possible_a_2000_l11_1171


namespace only_ShouZhuDaiTu_describes_random_event_l11_1109

-- Define the type for idioms
inductive Idiom
  | HaiKuShiLan
  | ShouZhuDaiTu
  | HuaBingChongJi
  | GuaShuDiLuo

-- Define a property for describing a random event
def describesRandomEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.ShouZhuDaiTu => True
  | _ => False

-- Theorem statement
theorem only_ShouZhuDaiTu_describes_random_event :
  ∀ i : Idiom, describesRandomEvent i ↔ i = Idiom.ShouZhuDaiTu :=
by
  sorry


end only_ShouZhuDaiTu_describes_random_event_l11_1109


namespace units_digit_of_sum_factorials_l11_1141

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def sum_factorials (n : ℕ) : ℕ := (Finset.range n).sum (λ i => factorial (i + 1))

theorem units_digit_of_sum_factorials :
  (sum_factorials 100) % 10 = 3 := by
  sorry

end units_digit_of_sum_factorials_l11_1141


namespace quadratic_equation_coefficients_l11_1175

/-- Given a quadratic equation x^2 = 5x - 1, prove that its coefficients are 1, -5, and 1 -/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), (∀ x, x^2 = 5*x - 1 ↔ a*x^2 + b*x + c = 0) ∧ a = 1 ∧ b = -5 ∧ c = 1 := by
  sorry

end quadratic_equation_coefficients_l11_1175


namespace apple_profit_percentage_l11_1130

/-- Calculates the total profit percentage for a stock of apples -/
theorem apple_profit_percentage 
  (total_stock : ℝ)
  (first_portion : ℝ)
  (second_portion : ℝ)
  (profit_rate : ℝ)
  (h1 : total_stock = 280)
  (h2 : first_portion = 0.4)
  (h3 : second_portion = 0.6)
  (h4 : profit_rate = 0.3)
  (h5 : first_portion + second_portion = 1) :
  let total_sp := (first_portion * total_stock * (1 + profit_rate)) + 
                  (second_portion * total_stock * (1 + profit_rate))
  let total_profit := total_sp - total_stock
  let total_profit_percentage := (total_profit / total_stock) * 100
  total_profit_percentage = 30 := by
sorry


end apple_profit_percentage_l11_1130


namespace smallest_positive_multiple_of_45_l11_1113

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l11_1113


namespace sum_of_ones_and_twos_2020_l11_1118

theorem sum_of_ones_and_twos_2020 :
  (Finset.filter (fun p : ℕ × ℕ => 4 * p.1 + 5 * p.2 = 2020) (Finset.product (Finset.range 505) (Finset.range 404))).card = 102 :=
by sorry

end sum_of_ones_and_twos_2020_l11_1118


namespace min_value_x_plus_2y_min_value_exact_l11_1185

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 4/w = 1 → x + 2*y ≤ z + 2*w :=
by sorry

theorem min_value_exact (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + 2*y = 9 + 4*Real.sqrt 2 :=
by sorry

end min_value_x_plus_2y_min_value_exact_l11_1185


namespace function_properties_l11_1188

open Real

theorem function_properties (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = Real.sin x * Real.cos x)
  (hg : ∀ x, g x = Real.sin x + Real.cos x) :
  (∀ x y, 0 < x ∧ x < y ∧ y < π/4 → f x < f y ∧ g x < g y) ∧
  (∃ x, f x + g x = 1/2 + Real.sqrt 2 ∧
    ∀ y, f y + g y ≤ 1/2 + Real.sqrt 2) := by
  sorry

end function_properties_l11_1188


namespace contractor_engagement_days_l11_1144

theorem contractor_engagement_days
  (daily_wage : ℕ)
  (daily_fine : ℚ)
  (total_pay : ℕ)
  (absent_days : ℕ)
  (h_daily_wage : daily_wage = 25)
  (h_daily_fine : daily_fine = 7.5)
  (h_total_pay : total_pay = 425)
  (h_absent_days : absent_days = 10) :
  ∃ (work_days : ℕ),
    (daily_wage : ℚ) * work_days - daily_fine * absent_days = total_pay ∧
    work_days + absent_days = 30 := by
  sorry

end contractor_engagement_days_l11_1144


namespace polynomial_product_constraint_l11_1166

theorem polynomial_product_constraint (a b : ℝ) : 
  (∀ x, (a * x + b) * (2 * x + 1) = 2 * a * x^2 + b) ∧ b = 6 → a + b = -6 := by
  sorry

end polynomial_product_constraint_l11_1166


namespace borrowed_sum_calculation_l11_1100

/-- Proves that given a sum of money borrowed at 6% per annum simple interest, 
    if the interest after 6 years is Rs. 672 less than the borrowed sum, 
    then the borrowed sum is Rs. 1050. -/
theorem borrowed_sum_calculation (P : ℝ) : 
  (P * 0.06 * 6 = P - 672) → P = 1050 := by
  sorry

end borrowed_sum_calculation_l11_1100


namespace arithmetic_sequence_ninth_term_l11_1110

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_ninth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_diff : a 3 - a 2 = -2) 
  (h_seventh : a 7 = -2) : 
  a 9 = -6 := by
sorry

end arithmetic_sequence_ninth_term_l11_1110


namespace no_consecutive_integers_with_square_diff_2000_l11_1125

theorem no_consecutive_integers_with_square_diff_2000 :
  ¬ ∃ (x : ℤ), (x + 1)^2 - x^2 = 2000 := by sorry

end no_consecutive_integers_with_square_diff_2000_l11_1125


namespace decimal_division_l11_1186

theorem decimal_division (x y : ℚ) (hx : x = 0.12) (hy : y = 0.04) :
  x / y = 3 := by
  sorry

end decimal_division_l11_1186


namespace unique_function_property_l11_1191

theorem unique_function_property (k : ℕ) (f : ℕ → ℕ) 
  (h1 : ∀ n, f n < f (n + 1)) 
  (h2 : ∀ n, f (f n) = n + 2 * k) : 
  ∀ n, f n = n + k := by
  sorry

end unique_function_property_l11_1191


namespace union_of_A_and_B_intersection_of_complements_l11_1145

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def B : Set ℝ := {x | |2*x + 1| ≤ 1}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

-- Define the intersection of complements of A and B
def ACompIntBComp : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statements
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

theorem intersection_of_complements : Aᶜ ∩ Bᶜ = ACompIntBComp := by sorry

end union_of_A_and_B_intersection_of_complements_l11_1145


namespace hotel_air_conditioning_l11_1107

theorem hotel_air_conditioning (total_rooms : ℝ) (total_rooms_pos : 0 < total_rooms) : 
  let rented_rooms := (3/4 : ℝ) * total_rooms
  let air_conditioned_rooms := (3/5 : ℝ) * total_rooms
  let rented_air_conditioned := (2/3 : ℝ) * air_conditioned_rooms
  let not_rented_rooms := total_rooms - rented_rooms
  let not_rented_air_conditioned := air_conditioned_rooms - rented_air_conditioned
  (not_rented_air_conditioned / not_rented_rooms) * 100 = 80 := by
sorry

end hotel_air_conditioning_l11_1107


namespace total_crayons_l11_1198

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 9

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- Theorem stating that the total number of crayons after adding is 12 -/
theorem total_crayons : initial_crayons + added_crayons = 12 := by
  sorry

end total_crayons_l11_1198


namespace hockey_games_per_month_l11_1172

/-- 
Given a hockey season with the following properties:
- The season lasts for 14 months
- There are 182 hockey games in the season

This theorem proves that the number of hockey games played each month is 13.
-/
theorem hockey_games_per_month (season_length : ℕ) (total_games : ℕ) 
  (h1 : season_length = 14) (h2 : total_games = 182) :
  total_games / season_length = 13 := by
  sorry

end hockey_games_per_month_l11_1172


namespace sufficient_not_necessary_l11_1195

theorem sufficient_not_necessary (a b x : ℝ) :
  (∀ a b x : ℝ, x > a^2 + b^2 → x > 2*a*b) ∧
  (∃ a b x : ℝ, x > 2*a*b ∧ x ≤ a^2 + b^2) :=
by sorry

end sufficient_not_necessary_l11_1195


namespace hyperbola_asymptotes_l11_1177

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - x^2 / 25 = 1

/-- The asymptote equation -/
def asymptote (m b x y : ℝ) : Prop :=
  y = m * x + b

/-- Theorem stating that the given equations are the asymptotes of the hyperbola -/
theorem hyperbola_asymptotes :
  ∃ (m b : ℝ), m > 0 ∧ 
    (∀ (x y : ℝ), hyperbola x y → 
      (asymptote m b x y ∨ asymptote (-m) b x y)) ∧
    m = 4/5 ∧ b = 1 := by
  sorry

end hyperbola_asymptotes_l11_1177


namespace sprained_wrist_frosting_time_l11_1196

/-- The time it takes Ann to frost a cake with her sprained wrist -/
def sprained_wrist_time : ℝ := 8

/-- The normal time it takes Ann to frost a cake -/
def normal_time : ℝ := 5

/-- The additional time it takes to frost 10 cakes with a sprained wrist -/
def additional_time : ℝ := 30

theorem sprained_wrist_frosting_time :
  sprained_wrist_time = (10 * normal_time + additional_time) / 10 := by
  sorry

end sprained_wrist_frosting_time_l11_1196


namespace people_per_column_l11_1189

theorem people_per_column (total_people : ℕ) (x : ℕ) : 
  total_people = 16 * x ∧ total_people = 12 * 40 → x = 30 := by
  sorry

end people_per_column_l11_1189


namespace hyperbola_s_squared_l11_1140

/-- A hyperbola centered at the origin passing through specific points -/
structure Hyperbola where
  -- The hyperbola passes through (5, -3)
  point1 : (5 : ℝ)^2 - (-3 : ℝ)^2 * a = b
  -- The hyperbola passes through (3, 0)
  point2 : (3 : ℝ)^2 = b
  -- The hyperbola passes through (s, -1)
  point3 : s^2 - (-1 : ℝ)^2 * a = b
  -- Ensure a and b are positive
  a_pos : a > 0
  b_pos : b > 0
  -- s is a real number
  s : ℝ

/-- The theorem stating the value of s^2 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.s^2 = 873/81 := by
  sorry

end hyperbola_s_squared_l11_1140


namespace remainder_proof_l11_1148

/-- The largest integer n such that 5^n divides 12^2015 + 13^2015 -/
def n : ℕ := 3

/-- The theorem statement -/
theorem remainder_proof :
  (12^2015 + 13^2015) / 5^n % 1000 = 625 :=
sorry

end remainder_proof_l11_1148


namespace arithmetic_proof_l11_1103

theorem arithmetic_proof : (139 + 27) * 2 + (23 + 11) = 366 := by
  sorry

end arithmetic_proof_l11_1103


namespace angle_A_measure_l11_1164

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Define a triangle ABC with sides a, b, c opposite to angles A, B, C
  true  -- Placeholder, as we don't need to specify all triangle properties

theorem angle_A_measure (A B C : ℝ) (a b c : ℝ) :
  triangle_ABC A B C a b c →
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  A = π / 6 := by
  sorry


end angle_A_measure_l11_1164


namespace complement_of_union_l11_1136

def U : Set ℤ := {x | 0 < x ∧ x ≤ 8}
def M : Set ℤ := {1, 3, 5, 7}
def N : Set ℤ := {5, 6, 7}

theorem complement_of_union :
  (U \ (M ∪ N)) = {2, 4, 8} := by sorry

end complement_of_union_l11_1136


namespace rare_integer_existence_and_uniqueness_l11_1123

-- Define the property of a function satisfying the given condition
def SatisfiesCondition (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (f (x + y) + y) = f (f x + y)

-- Define the set X_v
def X_v (f : ℤ → ℤ) (v : ℤ) : Set ℤ :=
  {x : ℤ | f x = v}

-- Define what it means for an integer to be rare under f
def IsRare (f : ℤ → ℤ) (v : ℤ) : Prop :=
  (X_v f v).Nonempty ∧ (X_v f v).Finite

-- Theorem statement
theorem rare_integer_existence_and_uniqueness :
  (∃ f : ℤ → ℤ, SatisfiesCondition f ∧ ∃ v : ℤ, IsRare f v) ∧
  (∀ f : ℤ → ℤ, SatisfiesCondition f → ∀ v w : ℤ, IsRare f v → IsRare f w → v = w) :=
sorry

end rare_integer_existence_and_uniqueness_l11_1123


namespace amount_paid_is_correct_l11_1153

/-- Calculates the amount paid to Jerry after discount --/
def amount_paid_after_discount (
  painting_hours : ℕ)
  (painting_rate : ℚ)
  (mowing_hours : ℕ)
  (mowing_rate : ℚ)
  (plumbing_hours : ℕ)
  (plumbing_rate : ℚ)
  (counter_time_multiplier : ℕ)
  (discount_rate : ℚ) : ℚ :=
  let painting_cost := painting_hours * painting_rate
  let mowing_cost := mowing_hours * mowing_rate
  let plumbing_cost := plumbing_hours * plumbing_rate
  let total_cost := painting_cost + mowing_cost + plumbing_cost
  let discount := total_cost * discount_rate
  total_cost - discount

/-- Theorem stating that Miss Stevie paid $226.80 after the discount --/
theorem amount_paid_is_correct : 
  amount_paid_after_discount 8 15 6 10 4 18 3 (1/10) = 226.8 := by
  sorry

end amount_paid_is_correct_l11_1153


namespace inequality_and_equality_condition_l11_1187

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) ≥ Real.sqrt (a^2 + a*c + c^2)) ∧
  ((Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) = Real.sqrt (a^2 + a*c + c^2)) ↔ 
   (1/b = 1/a + 1/c)) :=
by sorry

end inequality_and_equality_condition_l11_1187


namespace least_number_for_divisibility_l11_1197

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(23 ∣ (1056 + y))) ∧ (23 ∣ (1056 + x)) → x = 2 :=
by sorry

end least_number_for_divisibility_l11_1197


namespace complement_A_intersect_B_l11_1142

def U : Set Int := {-1, -2, -3, -4, 0}
def A : Set Int := {-1, -2, 0}
def B : Set Int := {-3, -4, 0}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {-3, -4} := by sorry

end complement_A_intersect_B_l11_1142


namespace g_12_equals_191_l11_1104

def g (n : ℕ) : ℕ := n^2 + 2*n + 23

theorem g_12_equals_191 : g 12 = 191 := by
  sorry

end g_12_equals_191_l11_1104


namespace caffeine_content_proof_l11_1105

/-- The amount of caffeine in one energy drink -/
def caffeine_per_drink : ℕ := sorry

/-- The maximum safe amount of caffeine per day -/
def max_safe_caffeine : ℕ := 500

/-- The number of energy drinks Brandy consumes -/
def num_drinks : ℕ := 4

/-- The additional amount of caffeine Brandy can safely consume after drinking the energy drinks -/
def additional_safe_caffeine : ℕ := 20

theorem caffeine_content_proof :
  caffeine_per_drink * num_drinks + additional_safe_caffeine = max_safe_caffeine ∧
  caffeine_per_drink = 120 := by sorry

end caffeine_content_proof_l11_1105


namespace alfonso_helmet_weeks_l11_1133

/-- Calculates the number of weeks Alfonso needs to work to buy a helmet -/
def weeks_to_buy_helmet (daily_earnings : ℚ) (days_per_week : ℕ) (helmet_cost : ℚ) (savings : ℚ) : ℚ :=
  (helmet_cost - savings) / (daily_earnings * days_per_week)

/-- Proves that Alfonso needs 10 weeks to buy the helmet -/
theorem alfonso_helmet_weeks : 
  let daily_earnings : ℚ := 6
  let days_per_week : ℕ := 5
  let helmet_cost : ℚ := 340
  let savings : ℚ := 40
  weeks_to_buy_helmet daily_earnings days_per_week helmet_cost savings = 10 := by
sorry

#eval weeks_to_buy_helmet 6 5 340 40

end alfonso_helmet_weeks_l11_1133
