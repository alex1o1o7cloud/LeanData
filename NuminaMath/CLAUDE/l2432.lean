import Mathlib

namespace recipe_pancakes_l2432_243274

/-- The number of pancakes Bobby ate -/
def bobby_pancakes : ℕ := 5

/-- The number of pancakes Bobby's dog ate -/
def dog_pancakes : ℕ := 7

/-- The number of pancakes left -/
def leftover_pancakes : ℕ := 9

/-- The total number of pancakes made by the recipe -/
def total_pancakes : ℕ := bobby_pancakes + dog_pancakes + leftover_pancakes

theorem recipe_pancakes : total_pancakes = 21 := by
  sorry

end recipe_pancakes_l2432_243274


namespace apple_purchase_cost_l2432_243277

/-- Represents a purchase option for apples -/
structure AppleOption where
  count : ℕ
  price : ℕ

/-- Calculates the total cost of purchasing apples -/
def totalCost (option1 : AppleOption) (option2 : AppleOption) (count1 : ℕ) (count2 : ℕ) : ℕ :=
  option1.price * count1 + option2.price * count2

/-- Calculates the total number of apples purchased -/
def totalApples (option1 : AppleOption) (option2 : AppleOption) (count1 : ℕ) (count2 : ℕ) : ℕ :=
  option1.count * count1 + option2.count * count2

theorem apple_purchase_cost (option1 : AppleOption) (option2 : AppleOption) :
  option1.count = 4 →
  option1.price = 15 →
  option2.count = 7 →
  option2.price = 25 →
  ∃ (count1 count2 : ℕ),
    count1 = count2 ∧
    totalApples option1 option2 count1 count2 = 28 ∧
    totalCost option1 option2 count1 count2 = 120 :=
by
  sorry

end apple_purchase_cost_l2432_243277


namespace frank_payment_l2432_243271

/-- The amount of money Frank handed to the cashier -/
def amount_handed (chocolate_bars : ℕ) (chips : ℕ) (chocolate_price : ℕ) (chips_price : ℕ) (change : ℕ) : ℕ :=
  chocolate_bars * chocolate_price + chips * chips_price + change

/-- Proof that Frank handed $20 to the cashier -/
theorem frank_payment : amount_handed 5 2 2 3 4 = 20 := by
  sorry

end frank_payment_l2432_243271


namespace converse_of_proposition_l2432_243262

theorem converse_of_proposition (a b : ℝ) : 
  (∀ x y : ℝ, x ≥ y → x^3 ≥ y^3) → 
  (∀ x y : ℝ, x^3 ≥ y^3 → x ≥ y) :=
sorry

end converse_of_proposition_l2432_243262


namespace P_necessary_not_sufficient_for_Q_l2432_243246

-- Define the conditions P and Q
def P (x : ℝ) : Prop := |x - 2| < 3
def Q (x : ℝ) : Prop := x^2 - 8*x + 15 < 0

-- Theorem statement
theorem P_necessary_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) := by
  sorry

end P_necessary_not_sufficient_for_Q_l2432_243246


namespace watch_time_theorem_l2432_243266

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts Time to total seconds -/
def Time.toSeconds (t : Time) : ℕ :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Represents a watch that loses time at a constant rate -/
structure Watch where
  lossRate : ℚ  -- Rate at which the watch loses time (in seconds per hour)

def Watch.actualTimeWhenShowing (w : Watch) (setTime : Time) (actualSetTime : Time) (showingTime : Time) : Time :=
  sorry  -- Implementation not required for the statement

theorem watch_time_theorem (w : Watch) :
  let noonTime : Time := ⟨12, 0, 0⟩
  let threeTime : Time := ⟨15, 0, 0⟩
  let watchAtThree : Time := ⟨14, 54, 30⟩
  let eightPM : Time := ⟨20, 0, 0⟩
  let actualEightPM : Time := ⟨20, 15, 8⟩
  w.actualTimeWhenShowing noonTime noonTime eightPM = actualEightPM :=
by sorry


end watch_time_theorem_l2432_243266


namespace probability_of_event_b_l2432_243288

theorem probability_of_event_b 
  (prob_a : ℝ) 
  (prob_a_and_b : ℝ) 
  (prob_neither_a_nor_b : ℝ) 
  (h1 : prob_a = 0.20)
  (h2 : prob_a_and_b = 0.15)
  (h3 : prob_neither_a_nor_b = 0.5499999999999999) :
  ∃ (prob_b : ℝ), prob_b = 0.40 :=
by sorry

end probability_of_event_b_l2432_243288


namespace inequality_proof_l2432_243221

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : x^2/(1+x^2) + y^2/(1+y^2) + z^2/(1+z^2) = 2) : 
  x/(1+x^2) + y/(1+y^2) + z/(1+z^2) ≤ Real.sqrt 2 := by
  sorry

end inequality_proof_l2432_243221


namespace first_day_over_1000_l2432_243299

def fungi_count (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_over_1000 : ∃ n : ℕ, fungi_count n > 1000 ∧ ∀ m : ℕ, m < n → fungi_count m ≤ 1000 :=
by
  -- The proof goes here
  sorry

end first_day_over_1000_l2432_243299


namespace selling_price_calculation_l2432_243244

theorem selling_price_calculation (cost_price : ℝ) (gain_percent : ℝ) 
  (h1 : cost_price = 100)
  (h2 : gain_percent = 15) :
  cost_price * (1 + gain_percent / 100) = 115 :=
by
  sorry

end selling_price_calculation_l2432_243244


namespace box_height_proof_l2432_243217

/-- Given a box with specified dimensions and cube requirements, prove its height --/
theorem box_height_proof (length width : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) 
  (h_length : length = 9)
  (h_width : width = 12)
  (h_cube_volume : cube_volume = 3)
  (h_min_cubes : min_cubes = 108) :
  (cube_volume * min_cubes) / (length * width) = 3 := by
  sorry

end box_height_proof_l2432_243217


namespace f_less_than_g_for_n_ge_5_l2432_243269

theorem f_less_than_g_for_n_ge_5 (n : ℕ) (h : n ≥ 5) : n^2 + n < 2^n := by
  sorry

end f_less_than_g_for_n_ge_5_l2432_243269


namespace max_soccer_balls_buyable_l2432_243290

/-- The cost of 6 soccer balls in yuan -/
def cost_of_six_balls : ℕ := 168

/-- The number of balls in a set -/
def balls_in_set : ℕ := 6

/-- The amount of money available to spend in yuan -/
def available_money : ℕ := 500

/-- The maximum number of soccer balls that can be bought -/
def max_balls_bought : ℕ := 17

theorem max_soccer_balls_buyable :
  (cost_of_six_balls * max_balls_bought) / balls_in_set ≤ available_money ∧
  (cost_of_six_balls * (max_balls_bought + 1)) / balls_in_set > available_money :=
by sorry

end max_soccer_balls_buyable_l2432_243290


namespace angle_in_fourth_quadrant_l2432_243224

/-- A point P in ℝ² is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

/-- An angle θ is in the fourth quadrant if sin θ < 0 and cos θ > 0 -/
def in_fourth_quadrant (θ : ℝ) : Prop :=
  Real.sin θ < 0 ∧ Real.cos θ > 0

/-- If P(sin θ cos θ, 2cos θ) is in the second quadrant, then θ is in the fourth quadrant -/
theorem angle_in_fourth_quadrant (θ : ℝ) :
  in_second_quadrant (Real.sin θ * Real.cos θ, 2 * Real.cos θ) → in_fourth_quadrant θ :=
by sorry

end angle_in_fourth_quadrant_l2432_243224


namespace prob_heart_or_king_correct_l2432_243231

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either hearts or kings -/
def heart_or_king_count : ℕ := 16

/-- The probability of drawing at least one heart or king in two draws with replacement -/
def prob_at_least_one_heart_or_king : ℚ :=
  1 - (1 - heart_or_king_count / deck_size) ^ 2

theorem prob_heart_or_king_correct :
  prob_at_least_one_heart_or_king = 88 / 169 := by
  sorry

end prob_heart_or_king_correct_l2432_243231


namespace boat_rental_problem_l2432_243216

theorem boat_rental_problem (total_students : ℕ) 
  (large_boat_capacity small_boat_capacity : ℕ) :
  total_students = 104 →
  large_boat_capacity = 12 →
  small_boat_capacity = 5 →
  ∃ (num_large_boats num_small_boats : ℕ),
    num_large_boats * large_boat_capacity + 
    num_small_boats * small_boat_capacity = total_students ∧
    (num_large_boats = 2 ∨ num_large_boats = 7) :=
by sorry

end boat_rental_problem_l2432_243216


namespace increase_by_percentage_increase_350_by_175_percent_l2432_243237

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial + (percentage / 100) * initial = initial * (1 + percentage / 100) := by sorry

theorem increase_350_by_175_percent :
  350 + (175 / 100) * 350 = 962.5 := by sorry

end increase_by_percentage_increase_350_by_175_percent_l2432_243237


namespace johns_arcade_spending_l2432_243279

theorem johns_arcade_spending (total_allowance : ℚ) 
  (remaining_after_toy_store : ℚ) (toy_store_fraction : ℚ) 
  (h1 : total_allowance = 9/4)
  (h2 : remaining_after_toy_store = 3/5)
  (h3 : toy_store_fraction = 1/3) : 
  ∃ (arcade_fraction : ℚ), 
    arcade_fraction = 3/5 ∧ 
    remaining_after_toy_store = (1 - arcade_fraction) * total_allowance * (1 - toy_store_fraction) :=
by sorry

end johns_arcade_spending_l2432_243279


namespace newborn_count_l2432_243212

theorem newborn_count (total_children : ℕ) (toddlers : ℕ) : 
  total_children = 40 →
  toddlers = 6 →
  total_children = 5 * toddlers + toddlers + (total_children - 5 * toddlers - toddlers) →
  (total_children - 5 * toddlers - toddlers) = 4 :=
by sorry

end newborn_count_l2432_243212


namespace equation_solution_l2432_243281

theorem equation_solution : ∃! x : ℚ, (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 ∧ x = -2/3 := by
  sorry

end equation_solution_l2432_243281


namespace gross_profit_percentage_l2432_243278

theorem gross_profit_percentage (sales_price gross_profit : ℝ) 
  (h1 : sales_price = 91)
  (h2 : gross_profit = 56) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 160 := by
sorry

end gross_profit_percentage_l2432_243278


namespace bus_purchase_problem_l2432_243253

-- Define the variables
variable (a b : ℝ)
variable (x : ℝ)  -- Number of A model buses

-- Define the conditions
def total_buses : ℝ := 10
def fuel_savings_A : ℝ := 2.4
def fuel_savings_B : ℝ := 2
def price_difference : ℝ := 2
def model_cost_difference : ℝ := 6
def total_fuel_savings : ℝ := 22.4

-- State the theorem
theorem bus_purchase_problem :
  (a - b = price_difference) →
  (3 * b - 2 * a = model_cost_difference) →
  (fuel_savings_A * x + fuel_savings_B * (total_buses - x) = total_fuel_savings) →
  (a = 120 ∧ b = 100 ∧ x = 6 ∧ a * x + b * (total_buses - x) = 1120) := by
  sorry

end bus_purchase_problem_l2432_243253


namespace eggs_to_buy_l2432_243256

theorem eggs_to_buy (total_needed : ℕ) (given_by_andrew : ℕ) 
  (h1 : total_needed = 222) (h2 : given_by_andrew = 155) : 
  total_needed - given_by_andrew = 67 := by
  sorry

end eggs_to_buy_l2432_243256


namespace colored_pencils_ratio_l2432_243242

/-- Proves that given the conditions in the problem, the ratio of Cheryl's colored pencils to Cyrus's is 3:1 -/
theorem colored_pencils_ratio (madeline_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : madeline_pencils = 63)
  (h2 : total_pencils = 231) : ∃ (cheryl_pencils cyrus_pencils : ℕ),
  cheryl_pencils = 2 * madeline_pencils ∧
  total_pencils = cheryl_pencils + cyrus_pencils + madeline_pencils ∧
  cheryl_pencils / cyrus_pencils = 3 := by
  sorry


end colored_pencils_ratio_l2432_243242


namespace probability_5_heart_ace_l2432_243295

/-- Represents a standard deck of 52 playing cards. -/
def StandardDeck : ℕ := 52

/-- Represents the number of 5s in a standard deck. -/
def NumberOf5s : ℕ := 4

/-- Represents the number of hearts in a standard deck. -/
def NumberOfHearts : ℕ := 13

/-- Represents the number of Aces in a standard deck. -/
def NumberOfAces : ℕ := 4

/-- Theorem stating the probability of drawing a 5 as the first card, 
    a heart as the second card, and an Ace as the third card from a standard 52-card deck. -/
theorem probability_5_heart_ace : 
  (NumberOf5s : ℚ) / StandardDeck * 
  NumberOfHearts / (StandardDeck - 1) * 
  NumberOfAces / (StandardDeck - 2) = 1 / 650 := by
  sorry

end probability_5_heart_ace_l2432_243295


namespace dice_probability_l2432_243209

-- Define a die
def Die := Fin 6

-- Define the sum of three dice rolls
def diceSum (d1 d2 d3 : Die) : ℕ := d1.val + d2.val + d3.val + 3

-- Define the condition for the sum to be even and greater than 15
def validRoll (d1 d2 d3 : Die) : Prop :=
  Even (diceSum d1 d2 d3) ∧ diceSum d1 d2 d3 > 15

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := 216

-- Define the number of favorable outcomes
def favorableOutcomes : ℕ := 10

-- Theorem statement
theorem dice_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 108 := by sorry

end dice_probability_l2432_243209


namespace triangle_acute_angled_l2432_243289

theorem triangle_acute_angled (a b c : ℝ) 
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (sides_relation : a^4 + b^4 = c^4) : 
  c^2 < a^2 + b^2 := by
sorry

end triangle_acute_angled_l2432_243289


namespace theater_occupancy_l2432_243282

theorem theater_occupancy (total_chairs : ℕ) (total_people : ℕ) : 
  (3 * total_people = 5 * (4 * total_chairs / 5)) →  -- Three-fifths of people occupy four-fifths of chairs
  (total_chairs - (4 * total_chairs / 5) = 5) →      -- 5 chairs are empty
  (total_people = 33) :=                             -- Total people is 33
by
  sorry

#check theater_occupancy

end theater_occupancy_l2432_243282


namespace longest_altitudes_sum_is_14_l2432_243236

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : a = 6
  h_b : b = 8
  h_c : c = 10
  h_right : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the triangle -/
def longest_altitudes_sum (t : RightTriangle) : ℝ := t.a + t.b

theorem longest_altitudes_sum_is_14 (t : RightTriangle) :
  longest_altitudes_sum t = 14 := by
  sorry

end longest_altitudes_sum_is_14_l2432_243236


namespace integer_quotient_problem_l2432_243263

theorem integer_quotient_problem (x y : ℤ) :
  1996 * x + y / 96 = x + y →
  x / y = 1 / 2016 ∨ y / x = 2016 := by
sorry

end integer_quotient_problem_l2432_243263


namespace dividend_calculation_l2432_243273

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 10 * quotient)
  (h2 : divisor = 5 * remainder)
  (h3 : remainder = 46) :
  divisor * quotient + remainder = 5336 := by
  sorry

end dividend_calculation_l2432_243273


namespace x_equals_three_l2432_243280

theorem x_equals_three (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 3 * x^2 + 18 * x * y = x^3 + 3 * x^2 * y + 6 * x) : x = 3 := by
  sorry

end x_equals_three_l2432_243280


namespace valid_triples_are_solutions_l2432_243232

def is_valid_triple (x y z : ℕ+) : Prop :=
  ∃ (n : ℤ), (Real.sqrt (2005 / (x + y : ℝ)) + 
              Real.sqrt (2005 / (y + z : ℝ)) + 
              Real.sqrt (2005 / (z + x : ℝ))) = n

def is_solution_triple (x y z : ℕ+) : Prop :=
  (x = 2005 * 2 ∧ y = 2005 * 2 ∧ z = 2005 * 14) ∨
  (x = 2005 * 2 ∧ y = 2005 * 14 ∧ z = 2005 * 2) ∨
  (x = 2005 * 14 ∧ y = 2005 * 2 ∧ z = 2005 * 2)

theorem valid_triples_are_solutions (x y z : ℕ+) :
  is_valid_triple x y z ↔ is_solution_triple x y z := by
  sorry

end valid_triples_are_solutions_l2432_243232


namespace rectangle_area_proof_l2432_243252

theorem rectangle_area_proof (square_area : ℝ) (rectangle_width rectangle_length : ℝ) : 
  square_area = 25 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 2 * rectangle_width →
  rectangle_width * rectangle_length = 50 := by
  sorry

end rectangle_area_proof_l2432_243252


namespace problem_1_problem_2_l2432_243238

-- Problem 1
theorem problem_1 (a : ℝ) (h : a ≠ 1) : 
  a^2 / (a - 1) - a - 1 = 1 / (a - 1) := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) :
  (2 * x * y) / (x^2 - y^2) / ((1 / (x - y)) + (1 / (x + y))) = y := by sorry

end problem_1_problem_2_l2432_243238


namespace gcd_lcm_relation_l2432_243276

theorem gcd_lcm_relation (a b c : ℕ+) :
  (Nat.gcd a (Nat.gcd b c))^2 * Nat.lcm a b * Nat.lcm b c * Nat.lcm c a =
  (Nat.lcm a (Nat.lcm b c))^2 * Nat.gcd a b * Nat.gcd b c * Nat.gcd c a :=
by sorry

end gcd_lcm_relation_l2432_243276


namespace jihye_wallet_money_l2432_243291

/-- The total amount of money in Jihye's wallet -/
def total_money (note_value : ℕ) (note_count : ℕ) (coin_value : ℕ) : ℕ :=
  note_value * note_count + coin_value

/-- Theorem stating the total amount of money in Jihye's wallet -/
theorem jihye_wallet_money : total_money 1000 2 560 = 2560 := by
  sorry

end jihye_wallet_money_l2432_243291


namespace exterior_angle_theorem_l2432_243206

/-- The measure of the exterior angle BAC formed by a square and a regular octagon sharing a common side --/
def exterior_angle_measure : ℝ := 135

/-- A square and a regular octagon are coplanar and share a common side AD --/
axiom share_common_side : True

/-- Theorem: The measure of the exterior angle BAC is 135 degrees --/
theorem exterior_angle_theorem : exterior_angle_measure = 135 := by
  sorry

end exterior_angle_theorem_l2432_243206


namespace angle_terminal_side_point_l2432_243292

/-- Given an angle α whose terminal side passes through the point P(-4m, 3m) where m < 0,
    prove that 2sin(α) + cos(α) = -2/5 -/
theorem angle_terminal_side_point (m : ℝ) (α : ℝ) (h1 : m < 0) 
  (h2 : Real.cos α = 4 * m / (5 * abs m)) (h3 : Real.sin α = 3 * m / (5 * abs m)) :
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end angle_terminal_side_point_l2432_243292


namespace hyperbola_focus_asymptote_distance_l2432_243272

/-- The distance from the focus of a hyperbola to its asymptote -/
def distance_focus_to_asymptote (b : ℝ) : ℝ := 
  sorry

/-- The theorem stating the distance from the focus to the asymptote for a specific hyperbola -/
theorem hyperbola_focus_asymptote_distance : 
  ∀ b : ℝ, b > 0 → 
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) → 
  (∃ x : ℝ, x^2 / 4 + b^2 = 9) →
  (∀ x y : ℝ, y^2 = 12*x) →
  distance_focus_to_asymptote b = Real.sqrt 5 :=
sorry

end hyperbola_focus_asymptote_distance_l2432_243272


namespace minimum_sugar_amount_l2432_243245

theorem minimum_sugar_amount (f s : ℝ) : 
  (f ≥ 8 + (3 * s) / 4) → 
  (f ≤ 2 * s) → 
  s ≥ 32 / 5 :=
by
  sorry

#eval (32 : ℚ) / 5  -- To show that 32/5 = 6.4

end minimum_sugar_amount_l2432_243245


namespace johns_final_push_pace_l2432_243241

/-- Proves that John's pace during his final push was 4.2 m/s given the race conditions --/
theorem johns_final_push_pace (initial_distance : ℝ) (steve_speed : ℝ) (final_distance : ℝ) (push_duration : ℝ) :
  initial_distance = 12 →
  steve_speed = 3.7 →
  final_distance = 2 →
  push_duration = 28 →
  (push_duration * steve_speed + initial_distance + final_distance) / push_duration = 4.2 :=
by sorry

end johns_final_push_pace_l2432_243241


namespace expression_simplification_l2432_243286

theorem expression_simplification (a b : ℤ) (h1 : a = 1) (h2 : b = -2) :
  2 * (a^2 - 3*a*b + 1) - (2*a^2 - b^2) + 5*a*b = 8 := by
  sorry

end expression_simplification_l2432_243286


namespace min_value_sum_cubes_l2432_243201

/-- Given positive real numbers x and y satisfying x³ + y³ + 3xy = 1,
    the expression (x + 1/x)³ + (y + 1/y)³ has a minimum value of 125/4. -/
theorem min_value_sum_cubes (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : x^3 + y^3 + 3*x*y = 1) : 
    ∃ m : ℝ, m = 125/4 ∧ ∀ a b : ℝ, a > 0 → b > 0 → a^3 + b^3 + 3*a*b = 1 → 
    (a + 1/a)^3 + (b + 1/b)^3 ≥ m :=
  sorry

end min_value_sum_cubes_l2432_243201


namespace math_test_problem_count_l2432_243208

theorem math_test_problem_count :
  ∀ (total_points three_point_count four_point_count : ℕ),
    total_points = 100 →
    four_point_count = 10 →
    total_points = 3 * three_point_count + 4 * four_point_count →
    three_point_count + four_point_count = 30 :=
by sorry

end math_test_problem_count_l2432_243208


namespace simplify_fraction_l2432_243228

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * y) / (-(x^2 * y)) = -2 / x :=
by sorry

end simplify_fraction_l2432_243228


namespace candle_burn_time_l2432_243297

/-- Proves that given a candle that lasts 8 nights when burned for 1 hour per night, 
    if 6 candles are used over 24 nights, then the average burn time per night is 2 hours. -/
theorem candle_burn_time 
  (candle_duration : ℕ) 
  (burn_time_per_night : ℕ) 
  (num_candles : ℕ) 
  (total_nights : ℕ) 
  (h1 : candle_duration = 8)
  (h2 : burn_time_per_night = 1)
  (h3 : num_candles = 6)
  (h4 : total_nights = 24) :
  (candle_duration * burn_time_per_night * num_candles) / total_nights = 2 := by
  sorry

end candle_burn_time_l2432_243297


namespace equal_radii_of_intersecting_triangles_l2432_243223

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  vertices : Fin 3 → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Configuration of two intersecting triangles -/
structure IntersectingTriangles where
  triangle1 : TriangleWithInscribedCircle
  triangle2 : TriangleWithInscribedCircle
  smallTriangles : Fin 6 → TriangleWithInscribedCircle
  hexagon : Set (ℝ × ℝ)

/-- The theorem stating that the radii of the inscribed circles of the two original triangles are equal -/
theorem equal_radii_of_intersecting_triangles (config : IntersectingTriangles) 
  (h : ∀ i j : Fin 6, (config.smallTriangles i).radius = (config.smallTriangles j).radius) :
  config.triangle1.radius = config.triangle2.radius :=
sorry

end equal_radii_of_intersecting_triangles_l2432_243223


namespace grid_game_winner_parity_second_player_wins_when_even_first_player_wins_when_odd_l2432_243270

/-- Represents the outcome of the grid game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Determines the winner of the grid game based on the dimensions of the grid -/
def gridGameWinner (m n : ℕ) : GameOutcome :=
  if (m + n) % 2 = 0 then
    GameOutcome.SecondPlayerWins
  else
    GameOutcome.FirstPlayerWins

/-- Theorem stating the winning condition for the grid game -/
theorem grid_game_winner_parity (m n : ℕ) :
  gridGameWinner m n = 
    if (m + n) % 2 = 0 then 
      GameOutcome.SecondPlayerWins
    else 
      GameOutcome.FirstPlayerWins := by
  sorry

/-- Corollary: The second player wins when m + n is even -/
theorem second_player_wins_when_even (m n : ℕ) (h : (m + n) % 2 = 0) :
  gridGameWinner m n = GameOutcome.SecondPlayerWins := by
  sorry

/-- Corollary: The first player wins when m + n is odd -/
theorem first_player_wins_when_odd (m n : ℕ) (h : (m + n) % 2 ≠ 0) :
  gridGameWinner m n = GameOutcome.FirstPlayerWins := by
  sorry

end grid_game_winner_parity_second_player_wins_when_even_first_player_wins_when_odd_l2432_243270


namespace remaining_erasers_l2432_243267

theorem remaining_erasers (total : ℕ) (yeonju_fraction : ℚ) (minji_fraction : ℚ)
  (h_total : total = 28)
  (h_yeonju : yeonju_fraction = 1 / 4)
  (h_minji : minji_fraction = 3 / 7) :
  total - (↑total * yeonju_fraction).floor - (↑total * minji_fraction).floor = 9 := by
  sorry

end remaining_erasers_l2432_243267


namespace log_cos_acute_angle_l2432_243247

theorem log_cos_acute_angle (A m n : ℝ) : 
  0 < A → A < π/2 →
  Real.log (1 + Real.sin A) = m →
  Real.log (1 / (1 - Real.sin A)) = n →
  Real.log (Real.cos A) = (1/2) * (m - n) := by
  sorry

end log_cos_acute_angle_l2432_243247


namespace total_hats_bought_l2432_243265

theorem total_hats_bought (blue_cost green_cost total_price green_count : ℕ)
  (h1 : blue_cost = 6)
  (h2 : green_cost = 7)
  (h3 : total_price = 548)
  (h4 : green_count = 38)
  (h5 : ∃ blue_count : ℕ, blue_cost * blue_count + green_cost * green_count = total_price) :
  ∃ total_count : ℕ, total_count = green_count + (total_price - green_cost * green_count) / blue_cost :=
by sorry

end total_hats_bought_l2432_243265


namespace class_composition_l2432_243296

theorem class_composition (num_boys : ℕ) (avg_boys avg_girls avg_class : ℚ) :
  num_boys = 12 →
  avg_boys = 84 →
  avg_girls = 92 →
  avg_class = 86 →
  ∃ (num_girls : ℕ), 
    (num_boys : ℚ) * avg_boys + (num_girls : ℚ) * avg_girls = 
    ((num_boys : ℚ) + (num_girls : ℚ)) * avg_class ∧
    num_girls = 4 :=
by sorry

end class_composition_l2432_243296


namespace expression_positivity_l2432_243294

theorem expression_positivity (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) :
  5*x^2 + 5*y^2 + 5*z^2 + 6*x*y - 8*x*z - 8*y*z > 0 := by
  sorry

end expression_positivity_l2432_243294


namespace taxi_speed_taxi_speed_is_45_l2432_243258

/-- The speed of a taxi that overtakes a bus under specific conditions. -/
theorem taxi_speed : ℝ → Prop :=
  fun v =>
    (∀ (bus_distance : ℝ),
      bus_distance = 4 * (v - 30) →  -- Distance covered by bus in 4 hours
      bus_distance + 2 * (v - 30) = 2 * v) →  -- Taxi covers bus distance in 2 hours
    v = 45

/-- Proof of the taxi speed theorem. -/
theorem taxi_speed_is_45 : taxi_speed 45 := by
  sorry

end taxi_speed_taxi_speed_is_45_l2432_243258


namespace minimum_employees_proof_l2432_243230

/-- Represents the number of employees handling customer service -/
def customer_service : ℕ := 95

/-- Represents the number of employees handling technical support -/
def technical_support : ℕ := 80

/-- Represents the number of employees handling both customer service and technical support -/
def both : ℕ := 30

/-- Calculates the minimum number of employees needed to be hired -/
def min_employees : ℕ := (customer_service - both) + (technical_support - both) + both

theorem minimum_employees_proof :
  min_employees = 145 :=
sorry

end minimum_employees_proof_l2432_243230


namespace square_roots_problem_l2432_243203

theorem square_roots_problem (a m : ℝ) (ha : 0 < a) 
  (h1 : (2 * m - 1)^2 = a) (h2 : (m + 4)^2 = a) : a = 9 := by
  sorry

end square_roots_problem_l2432_243203


namespace trig_identity_l2432_243214

theorem trig_identity : 
  (Real.cos (12 * π / 180) - Real.cos (18 * π / 180) * Real.sin (60 * π / 180)) / 
  Real.sin (18 * π / 180) = 1 / 2 := by
  sorry

end trig_identity_l2432_243214


namespace adam_strawberries_l2432_243226

/-- The number of strawberries Adam had left -/
def strawberries_left : ℕ := 33

/-- The number of strawberries Adam ate -/
def strawberries_eaten : ℕ := 2

/-- The initial number of strawberries Adam picked -/
def initial_strawberries : ℕ := strawberries_left + strawberries_eaten

theorem adam_strawberries : initial_strawberries = 35 := by
  sorry

end adam_strawberries_l2432_243226


namespace combined_surface_area_theorem_l2432_243218

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents the combined shape of two cubes -/
structure CombinedShape where
  largerCube : Cube
  smallerCube : Cube

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℝ := 6 * c.edgeLength^2

/-- Calculates the surface area of the combined shape -/
def combinedSurfaceArea (cs : CombinedShape) : ℝ :=
  surfaceArea cs.largerCube + surfaceArea cs.smallerCube - 4 * cs.smallerCube.edgeLength^2

/-- The main theorem stating the surface area of the combined shape -/
theorem combined_surface_area_theorem (cs : CombinedShape) 
  (h1 : cs.largerCube.edgeLength = 2)
  (h2 : cs.smallerCube.edgeLength = cs.largerCube.edgeLength / 2) :
  combinedSurfaceArea cs = 32 := by
  sorry

#check combined_surface_area_theorem

end combined_surface_area_theorem_l2432_243218


namespace candy_distribution_l2432_243285

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 16 → 
  num_bags = 2 → 
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 8 := by
sorry

end candy_distribution_l2432_243285


namespace triangle_theorem_l2432_243234

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C)
  (h2 : t.b + t.c = Real.sqrt 2 * t.a)
  (h3 : t.a * t.b * Real.sin t.A / 2 = Real.sqrt 3 / 12) : 
  t.A = Real.pi / 3 ∧ t.a = 1 := by
  sorry


end triangle_theorem_l2432_243234


namespace icosahedron_cube_relation_l2432_243235

/-- Given a cube with edge length a and an inscribed icosahedron, 
    m is the length of the line segment connecting two vertices 
    of the icosahedron on a face of the cube -/
def icosahedron_in_cube (a m : ℝ) : Prop :=
  a > 0 ∧ m > 0 ∧ a^2 - a*m - m^2 = 0

/-- Theorem stating the relationship between the cube's edge length 
    and the distance between icosahedron vertices on a face -/
theorem icosahedron_cube_relation {a m : ℝ} 
  (h : icosahedron_in_cube a m) : a^2 - a*m - m^2 = 0 := by
  sorry

end icosahedron_cube_relation_l2432_243235


namespace max_value_of_expression_max_value_achievable_l2432_243251

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27) ≤ 1 / (6*Real.sqrt 3 + 6) :=
sorry

theorem max_value_achievable :
  ∃ x : ℝ, x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27) = 1 / (6*Real.sqrt 3 + 6) :=
sorry

end max_value_of_expression_max_value_achievable_l2432_243251


namespace rectangle_area_l2432_243259

theorem rectangle_area (length width : ℚ) (h1 : length = 2 / 3) (h2 : width = 3 / 5) :
  length * width = 2 / 5 := by
  sorry

end rectangle_area_l2432_243259


namespace coefficient_sum_l2432_243293

theorem coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10 + a₁₁*(x+1)^11) →
  a₁ + a₂ + a₁₁ = 781 :=
by sorry

end coefficient_sum_l2432_243293


namespace percentage_increase_60_to_80_l2432_243298

/-- The percentage increase when a value changes from 60 to 80 -/
theorem percentage_increase_60_to_80 : 
  (80 - 60) / 60 * 100 = 100 / 3 := by sorry

end percentage_increase_60_to_80_l2432_243298


namespace projection_area_eq_projection_length_l2432_243268

/-- A cube with edge length 1 -/
structure UnitCube where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- A plane onto which the cube is projected -/
class ProjectionPlane

/-- A line perpendicular to the projection plane -/
class PerpendicularLine (P : ProjectionPlane)

/-- The area of the projection of a cube onto a plane -/
noncomputable def projection_area (cube : UnitCube) (P : ProjectionPlane) : ℝ :=
  sorry

/-- The length of the projection of a cube onto a line perpendicular to the projection plane -/
noncomputable def projection_length (cube : UnitCube) (P : ProjectionPlane) (L : PerpendicularLine P) : ℝ :=
  sorry

/-- Theorem stating that the area of the projection of a unit cube onto a plane
    is equal to the length of its projection onto a perpendicular line -/
theorem projection_area_eq_projection_length
  (cube : UnitCube) (P : ProjectionPlane) (L : PerpendicularLine P) :
  projection_area cube P = projection_length cube P L :=
sorry

end projection_area_eq_projection_length_l2432_243268


namespace only_25_satisfies_l2432_243260

theorem only_25_satisfies : ∀ n : ℕ, 
  (n > 5 * (n % 10) ∧ n ≠ 25) → False :=
by sorry

end only_25_satisfies_l2432_243260


namespace multiply_divide_sqrt_l2432_243275

theorem multiply_divide_sqrt (x y : ℝ) : 
  x = 0.7142857142857143 → 
  x ≠ 0 → 
  Real.sqrt ((x * y) / 7) = x → 
  y = 5 := by
sorry

end multiply_divide_sqrt_l2432_243275


namespace blocks_remaining_l2432_243257

theorem blocks_remaining (initial : ℕ) (used : ℕ) (remaining : ℕ) : 
  initial = 78 → used = 19 → remaining = initial - used → remaining = 59 := by
  sorry

end blocks_remaining_l2432_243257


namespace number_of_inequalities_l2432_243210

-- Define a function to check if an expression is an inequality
def isInequality (expr : String) : Bool :=
  match expr with
  | "3 < 5" => true
  | "x > 0" => true
  | "2x ≠ 3" => true
  | "a = 3" => false
  | "2a + 1" => false
  | "(1-x)/5 > 1" => true
  | _ => false

-- Define the list of expressions
def expressions : List String :=
  ["3 < 5", "x > 0", "2x ≠ 3", "a = 3", "2a + 1", "(1-x)/5 > 1"]

-- Theorem stating that the number of inequalities is 4
theorem number_of_inequalities :
  (expressions.filter isInequality).length = 4 := by
  sorry

end number_of_inequalities_l2432_243210


namespace ellipse_properties_l2432_243202

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def short_axis_length (b : ℝ) : Prop :=
  2 * b = 2 * Real.sqrt 3

def slope_product (a : ℝ) (x y : ℝ) : Prop :=
  y^2 / (x^2 - a^2) = 3 / 4

-- Define the theorem
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : short_axis_length b) (h4 : ∀ x y, ellipse a b x y → slope_product a x y) :
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ m : ℝ, m ≠ 0 → 
    ∃ Q : ℝ × ℝ, 
      (∃ A B : ℝ × ℝ, 
        ellipse a b A.1 A.2 ∧ 
        ellipse a b B.1 B.2 ∧
        A.1 = m * A.2 + 1 ∧
        B.1 = m * B.2 + 1 ∧
        Q.1 = A.1 + (Q.2 - A.2) * (A.1 + a) / A.2 ∧
        Q.1 = B.1 + (Q.2 - B.2) * (B.1 - a) / B.2) →
      Q.1 = 4) :=
sorry

end ellipse_properties_l2432_243202


namespace fraction_simplification_l2432_243207

theorem fraction_simplification :
  ((2^2010)^2 - (2^2008)^2) / ((2^2009)^2 - (2^2007)^2) = 4 := by
  sorry

end fraction_simplification_l2432_243207


namespace total_signup_combinations_l2432_243239

/-- The number of ways for one person to sign up -/
def signup_options : ℕ := 2

/-- The number of people signing up -/
def num_people : ℕ := 3

/-- Theorem: The total number of different ways for three people to sign up, 
    each with two independent choices, is 8 -/
theorem total_signup_combinations : signup_options ^ num_people = 8 := by
  sorry

end total_signup_combinations_l2432_243239


namespace largest_integral_y_l2432_243255

theorem largest_integral_y : ∃ y : ℤ, y = 4 ∧ 
  (∀ z : ℤ, (1/4 : ℚ) < (z : ℚ)/7 ∧ (z : ℚ)/7 < 7/11 → z ≤ y) ∧
  (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/11 :=
by sorry

end largest_integral_y_l2432_243255


namespace blue_balloons_most_l2432_243249

/-- Represents the color of a balloon -/
inductive BalloonColor
  | Red
  | Blue
  | Yellow

/-- Counts the number of balloons of a given color -/
def count_balloons (color : BalloonColor) : ℕ :=
  match color with
  | BalloonColor.Red => 6
  | BalloonColor.Blue => 12
  | BalloonColor.Yellow => 6

theorem blue_balloons_most : 
  (∀ c : BalloonColor, c ≠ BalloonColor.Blue → count_balloons BalloonColor.Blue > count_balloons c) ∧ 
  count_balloons BalloonColor.Red + count_balloons BalloonColor.Blue + count_balloons BalloonColor.Yellow = 24 ∧
  count_balloons BalloonColor.Blue = count_balloons BalloonColor.Red + 6 ∧
  count_balloons BalloonColor.Red = 24 / 4 := by
  sorry

end blue_balloons_most_l2432_243249


namespace all_calculations_incorrect_l2432_243283

theorem all_calculations_incorrect : 
  (-|-3| ≠ 3) ∧ 
  (∀ a b : ℝ, (a + b)^2 ≠ a^2 + b^2) ∧ 
  (∀ a : ℝ, a ≠ 0 → a^3 * a^4 ≠ a^12) ∧ 
  (|-3^2| ≠ 3) := by
  sorry

end all_calculations_incorrect_l2432_243283


namespace candies_to_remove_for_even_distribution_l2432_243213

def total_candies : ℕ := 24
def num_sisters : ℕ := 4

theorem candies_to_remove_for_even_distribution :
  (total_candies % num_sisters = 0) ∧
  (total_candies / num_sisters * num_sisters = total_candies) :=
by sorry

end candies_to_remove_for_even_distribution_l2432_243213


namespace cube_root_of_eight_l2432_243287

theorem cube_root_of_eight (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end cube_root_of_eight_l2432_243287


namespace carson_octopus_legs_l2432_243225

/-- The number of octopuses Carson saw -/
def num_octopuses : ℕ := 5

/-- The number of legs each octopus has -/
def legs_per_octopus : ℕ := 8

/-- The total number of octopus legs Carson saw -/
def total_octopus_legs : ℕ := num_octopuses * legs_per_octopus

theorem carson_octopus_legs : total_octopus_legs = 40 := by
  sorry

end carson_octopus_legs_l2432_243225


namespace lcm_of_12_and_18_l2432_243200

theorem lcm_of_12_and_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_of_12_and_18_l2432_243200


namespace equation_solutions_l2432_243243

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -1 ∧ 
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solutions_l2432_243243


namespace parallelepiped_volume_l2432_243219

theorem parallelepiped_volume 
  (a b c : ℝ) 
  (h1 : Real.sqrt (a^2 + b^2 + c^2) = 13)
  (h2 : Real.sqrt (a^2 + b^2) = 3 * Real.sqrt 17)
  (h3 : Real.sqrt (b^2 + c^2) = 4 * Real.sqrt 10) :
  a * b * c = 144 := by
sorry

end parallelepiped_volume_l2432_243219


namespace distance_circle_center_to_point_l2432_243229

/-- The distance between the center of a circle and a point in polar coordinates -/
theorem distance_circle_center_to_point 
  (ρ : ℝ → ℝ) -- Radius function for the circle
  (θ : ℝ) -- Angle parameter
  (r : ℝ) -- Radius of point D
  (φ : ℝ) -- Angle of point D
  (h1 : ∀ θ, ρ θ = 2 * Real.sin θ) -- Circle equation
  (h2 : r = 1) -- Radius of point D
  (h3 : φ = Real.pi) -- Angle of point D
  : Real.sqrt 2 = Real.sqrt ((0 - r * Real.cos φ)^2 + (1 - r * Real.sin φ)^2) :=
sorry

end distance_circle_center_to_point_l2432_243229


namespace trapezoid_constructible_l2432_243264

/-- A trapezoid with side lengths a, b, c, and d, where a and b are the bases and c and d are the legs. -/
structure Trapezoid (a b c d : ℝ) : Prop where
  base1 : a > 0
  base2 : b > 0
  leg1 : c > 0
  leg2 : d > 0

/-- The condition for constructibility of a trapezoid. -/
def isConstructible (a b c d : ℝ) : Prop :=
  c > d ∧ c - d < a - b ∧ a - b < c + d

/-- Theorem stating the necessary and sufficient conditions for constructing a trapezoid. -/
theorem trapezoid_constructible {a b c d : ℝ} (t : Trapezoid a b c d) :
  isConstructible a b c d ↔ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = a - b :=
sorry

end trapezoid_constructible_l2432_243264


namespace nonnegative_rational_function_l2432_243215

theorem nonnegative_rational_function (x : ℝ) :
  (x - 12 * x^2 + 36 * x^3) / (9 - x^3) ≥ 0 ↔ 0 ≤ x ∧ x < 3 := by
  sorry

end nonnegative_rational_function_l2432_243215


namespace smallest_two_digit_multiple_of_three_l2432_243261

theorem smallest_two_digit_multiple_of_three : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 99) ∧ 
  n % 3 = 0 ∧ 
  (∀ m : ℕ, (m ≥ 10 ∧ m ≤ 99) ∧ m % 3 = 0 → n ≤ m) ∧
  n = 12 := by
  sorry

end smallest_two_digit_multiple_of_three_l2432_243261


namespace distance_to_focus_is_six_l2432_243205

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ∀ x y, y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

theorem distance_to_focus_is_six (p : Parabola) (P : PointOnParabola p) 
  (h : |P.x - (-3)| = 5) : 
  Real.sqrt ((P.x - focus.1)^2 + (P.y - focus.2)^2) = 6 := by
  sorry

end distance_to_focus_is_six_l2432_243205


namespace x_plus_y_between_52_and_53_l2432_243220

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the problem conditions
def problem_conditions (x y : ℝ) : Prop :=
  y = 4 * (floor x) + 2 ∧
  y = 5 * (floor (x - 3)) + 7 ∧
  ∀ n : ℤ, x ≠ n

-- Theorem statement
theorem x_plus_y_between_52_and_53 (x y : ℝ) 
  (h : problem_conditions x y) : 
  52 < x + y ∧ x + y < 53 := by
  sorry

end x_plus_y_between_52_and_53_l2432_243220


namespace probability_nine_heads_in_twelve_flips_l2432_243204

theorem probability_nine_heads_in_twelve_flips : 
  let n : ℕ := 12  -- number of coin flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 55/1024 :=
by sorry

end probability_nine_heads_in_twelve_flips_l2432_243204


namespace trigonometric_ratio_proof_trigonometric_expression_simplification_l2432_243250

theorem trigonometric_ratio_proof (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by
  sorry

theorem trigonometric_expression_simplification (α : Real) :
  (Real.sin (π/2 + α) * Real.cos (5*π/2 - α) * Real.tan (-π + α)) /
  (Real.tan (7*π - α) * Real.sin (π + α)) = Real.cos α := by
  sorry

end trigonometric_ratio_proof_trigonometric_expression_simplification_l2432_243250


namespace oranges_packed_l2432_243248

/-- Calculates the total number of oranges packed given the number of oranges per box and the number of boxes used. -/
def totalOranges (orangesPerBox : ℕ) (boxesUsed : ℕ) : ℕ :=
  orangesPerBox * boxesUsed

/-- Proves that packing 10 oranges per box in 265 boxes results in 2650 oranges packed. -/
theorem oranges_packed :
  let orangesPerBox : ℕ := 10
  let boxesUsed : ℕ := 265
  totalOranges orangesPerBox boxesUsed = 2650 := by
  sorry

end oranges_packed_l2432_243248


namespace budget_theorem_l2432_243222

/-- Represents a budget with three categories in a given ratio -/
structure Budget where
  ratio_1 : ℕ
  ratio_2 : ℕ
  ratio_3 : ℕ
  amount_2 : ℚ

/-- Calculates the total amount allocated in a budget -/
def total_amount (b : Budget) : ℚ :=
  (b.ratio_1 + b.ratio_2 + b.ratio_3) * (b.amount_2 / b.ratio_2)

/-- Theorem stating that for a budget with ratio 5:4:1 and $720 allocated to the second category,
    the total amount is $1800 -/
theorem budget_theorem (b : Budget) 
  (h1 : b.ratio_1 = 5)
  (h2 : b.ratio_2 = 4)
  (h3 : b.ratio_3 = 1)
  (h4 : b.amount_2 = 720) :
  total_amount b = 1800 := by
  sorry

end budget_theorem_l2432_243222


namespace age_equation_solution_l2432_243233

/-- Given a person's current age of 50, prove that the equation
    5 * (A + 5) - 5 * (A - X) = A is satisfied when X = 5. -/
theorem age_equation_solution :
  let A : ℕ := 50
  let X : ℕ := 5
  5 * (A + 5) - 5 * (A - X) = A :=
by sorry

end age_equation_solution_l2432_243233


namespace sine_inequality_l2432_243254

theorem sine_inequality (x y : Real) : 
  x ∈ Set.Icc 0 (Real.pi / 2) → 
  y ∈ Set.Icc 0 Real.pi → 
  Real.sin (x + y) ≥ Real.sin x - Real.sin y := by
sorry

end sine_inequality_l2432_243254


namespace household_gas_fee_l2432_243240

def gas_fee (usage : ℕ) : ℚ :=
  if usage ≤ 60 then
    0.8 * usage
  else
    0.8 * 60 + 1.2 * (usage - 60)

theorem household_gas_fee :
  ∃ (usage : ℕ),
    usage > 60 ∧
    gas_fee usage / usage = 0.88 ∧
    gas_fee usage = 66 := by
  sorry

end household_gas_fee_l2432_243240


namespace f_is_linear_l2432_243284

/-- A function f: ℝ → ℝ is linear if there exist constants m and b such that f(x) = mx + b for all x ∈ ℝ -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- The function f(x) = -x -/
def f : ℝ → ℝ := fun x ↦ -x

/-- Theorem: The function f(x) = -x is a linear function -/
theorem f_is_linear : IsLinearFunction f := by
  sorry

end f_is_linear_l2432_243284


namespace set_intersection_complement_l2432_243211

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set A
def A : Set Nat := {1, 2, 3, 5}

-- Define set B
def B : Set Nat := {2, 4, 6}

-- Theorem statement
theorem set_intersection_complement : B ∩ (U \ A) = {4, 6} := by
  sorry

end set_intersection_complement_l2432_243211


namespace closest_point_on_line_l2432_243227

/-- The point on the line y = 2x - 1 that is closest to (3, 4) is (13/5, 21/5) -/
theorem closest_point_on_line (x y : ℝ) : 
  y = 2 * x - 1 → 
  (x - 3)^2 + (y - 4)^2 ≥ (13/5 - 3)^2 + (21/5 - 4)^2 :=
by sorry

end closest_point_on_line_l2432_243227
