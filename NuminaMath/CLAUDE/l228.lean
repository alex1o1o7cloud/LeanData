import Mathlib

namespace NUMINAMATH_CALUDE_no_right_triangle_with_75_median_l228_22823

theorem no_right_triangle_with_75_median (a b c : ℕ) : 
  (a * a + b * b = c * c) →  -- Pythagorean theorem
  (Nat.gcd a (Nat.gcd b c) = 1) →  -- (a, b, c) = 1
  ¬(((a * a + 4 * b * b : ℚ) / 4 = 15 * 15 / 4) ∨  -- median to leg
    (2 * a * a + 2 * b * b - c * c : ℚ) / 4 = 15 * 15 / 4)  -- median to hypotenuse
:= by sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_75_median_l228_22823


namespace NUMINAMATH_CALUDE_promotion_difference_l228_22870

/-- Represents a shoe promotion strategy -/
inductive Promotion
  | A  -- Buy one pair, get second pair half price
  | B  -- Buy one pair, get $15 off second pair

/-- Calculates the total cost of two pairs of shoes under a given promotion -/
def calculateCost (p : Promotion) (price1 : ℕ) (price2 : ℕ) : ℕ :=
  match p with
  | Promotion.A => price1 + price2 / 2
  | Promotion.B => price1 + price2 - 15

/-- Theorem stating the difference in cost between Promotion B and A -/
theorem promotion_difference :
  ∀ (price1 price2 : ℕ),
  price1 = 50 →
  price2 = 40 →
  calculateCost Promotion.B price1 price2 - calculateCost Promotion.A price1 price2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_promotion_difference_l228_22870


namespace NUMINAMATH_CALUDE_brokerage_percentage_approx_l228_22894

/-- Calculates the brokerage percentage given the cash realized and total amount --/
def brokerage_percentage (cash_realized : ℚ) (total_amount : ℚ) : ℚ :=
  ((cash_realized - total_amount) / total_amount) * 100

/-- Theorem stating that the brokerage percentage is approximately 0.24% --/
theorem brokerage_percentage_approx :
  let cash_realized : ℚ := 10425 / 100
  let total_amount : ℚ := 104
  abs (brokerage_percentage cash_realized total_amount - 24 / 100) < 1 / 1000 := by
  sorry

#eval brokerage_percentage (10425 / 100) 104

end NUMINAMATH_CALUDE_brokerage_percentage_approx_l228_22894


namespace NUMINAMATH_CALUDE_find_t_l228_22875

theorem find_t (x y z t : ℝ) : 
  (x + y + z) / 3 = 10 →
  (x + y + z + t) / 4 = 12 →
  t = 18 := by
sorry

end NUMINAMATH_CALUDE_find_t_l228_22875


namespace NUMINAMATH_CALUDE_total_shirts_made_l228_22817

/-- The number of shirts a machine can make per minute -/
def shirts_per_minute : ℕ := 6

/-- The number of minutes the machine worked yesterday -/
def minutes_yesterday : ℕ := 12

/-- The number of minutes the machine worked today -/
def minutes_today : ℕ := 14

/-- Theorem: The total number of shirts made by the machine is 156 -/
theorem total_shirts_made : 
  shirts_per_minute * minutes_yesterday + shirts_per_minute * minutes_today = 156 := by
  sorry

end NUMINAMATH_CALUDE_total_shirts_made_l228_22817


namespace NUMINAMATH_CALUDE_c_value_for_four_distinct_roots_l228_22860

/-- The polynomial P(x) -/
def P (c : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 5) * (x^2 - c*x + 2) * (x^2 - 5*x + 10)

/-- The theorem stating the relationship between c and the number of distinct roots of P(x) -/
theorem c_value_for_four_distinct_roots (c : ℂ) : 
  (∃ (S : Finset ℂ), S.card = 4 ∧ (∀ x ∈ S, P c x = 0) ∧ (∀ x, P c x = 0 → x ∈ S)) →
  Complex.abs c = Real.sqrt (22.5 - Real.sqrt 165) := by
  sorry

end NUMINAMATH_CALUDE_c_value_for_four_distinct_roots_l228_22860


namespace NUMINAMATH_CALUDE_floor_of_e_equals_two_l228_22880

theorem floor_of_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_equals_two_l228_22880


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l228_22810

theorem isosceles_triangle_angle_measure (D E F : ℝ) : 
  D + E + F = 180 →  -- sum of angles in a triangle is 180°
  E = F →            -- isosceles triangle condition
  F = 3 * D →        -- angle F is three times angle D
  E = 540 / 7 :=     -- measure of angle E
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l228_22810


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l228_22840

theorem min_value_trig_expression (α β : ℝ) :
  9 * (Real.cos α)^2 - 10 * Real.cos α * Real.sin β - 8 * Real.cos β * Real.sin α + 17 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l228_22840


namespace NUMINAMATH_CALUDE_independence_test_confidence_l228_22857

/-- The critical value for the independence test -/
def critical_value : ℝ := 5.024

/-- The confidence level for "X and Y are related" given k > critical_value -/
def confidence_level (k : ℝ) : ℝ := 97.5

/-- Theorem stating that when k > critical_value, the confidence level is 97.5% -/
theorem independence_test_confidence (k : ℝ) (h : k > critical_value) :
  confidence_level k = 97.5 := by sorry

end NUMINAMATH_CALUDE_independence_test_confidence_l228_22857


namespace NUMINAMATH_CALUDE_multiply_and_add_l228_22869

theorem multiply_and_add : 19 * 42 + 81 * 19 = 2337 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l228_22869


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l228_22816

/-- Given the total amount of paint and the amount of white paint used,
    calculate the amount of blue paint used. -/
theorem blue_paint_calculation (total_paint white_paint : ℕ) 
    (h1 : total_paint = 6689)
    (h2 : white_paint = 660) :
    total_paint - white_paint = 6029 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_calculation_l228_22816


namespace NUMINAMATH_CALUDE_same_terminal_side_as_negative_120_degrees_l228_22897

theorem same_terminal_side_as_negative_120_degrees :
  ∃ (k : ℤ), 0 ≤ -120 + k * 360 ∧ -120 + k * 360 < 360 ∧ -120 + k * 360 = 240 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_negative_120_degrees_l228_22897


namespace NUMINAMATH_CALUDE_f_f_10_equals_1_l228_22862

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 10^(x-1) else Real.log x / Real.log 10

-- State the theorem
theorem f_f_10_equals_1 : f (f 10) = 1 := by sorry

end NUMINAMATH_CALUDE_f_f_10_equals_1_l228_22862


namespace NUMINAMATH_CALUDE_coffee_per_day_l228_22849

/-- The number of times Maria goes to the coffee shop per day. -/
def visits_per_day : ℕ := 2

/-- The number of cups of coffee Maria orders each visit. -/
def cups_per_visit : ℕ := 3

/-- Theorem: Maria orders 6 cups of coffee per day. -/
theorem coffee_per_day : visits_per_day * cups_per_visit = 6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_per_day_l228_22849


namespace NUMINAMATH_CALUDE_f_of_five_eq_six_elevenths_l228_22807

/-- Given a function f(x) = (x+1) / (3x-4), prove that f(5) = 6/11 -/
theorem f_of_five_eq_six_elevenths :
  let f : ℝ → ℝ := λ x ↦ (x + 1) / (3 * x - 4)
  f 5 = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_eq_six_elevenths_l228_22807


namespace NUMINAMATH_CALUDE_factorial_division_l228_22856

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l228_22856


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l228_22866

theorem coefficient_x_squared_in_binomial_expansion :
  let n : ℕ := 8
  let k : ℕ := 3
  let coeff : ℤ := (-1)^k * 2^k * Nat.choose n k
  coeff = -448 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l228_22866


namespace NUMINAMATH_CALUDE_cattle_market_problem_l228_22890

/-- The number of animals each person brought to the market satisfies the given conditions --/
theorem cattle_market_problem (j h d : ℕ) : 
  (j + 5 = 2 * (h - 5)) →  -- Condition 1
  (h + 13 = 3 * (d - 13)) →  -- Condition 2
  (d + 3 = 6 * (j - 3)) →  -- Condition 3
  j = 7 ∧ h = 11 ∧ d = 21 := by
sorry

end NUMINAMATH_CALUDE_cattle_market_problem_l228_22890


namespace NUMINAMATH_CALUDE_estimate_correct_l228_22813

/-- Represents the sample data of homework times in minutes -/
def sample_data : List Nat := [75, 80, 85, 65, 95, 100, 70, 55, 65, 75, 85, 110, 120, 80, 85, 80, 75, 90, 90, 95, 70, 60, 60, 75, 90, 95, 65, 75, 80, 80]

/-- The total number of students in the school -/
def total_students : Nat := 2100

/-- The size of the sample -/
def sample_size : Nat := 30

/-- The threshold time in minutes -/
def threshold : Nat := 90

/-- Counts the number of elements in the list that are greater than or equal to the threshold -/
def count_above_threshold (data : List Nat) (threshold : Nat) : Nat :=
  data.filter (λ x => x ≥ threshold) |>.length

/-- Estimates the number of students in the entire school population who spend at least the threshold time on homework -/
def estimate_students_above_threshold : Nat :=
  let count := count_above_threshold sample_data threshold
  (count * total_students) / sample_size

theorem estimate_correct : estimate_students_above_threshold = 630 := by
  sorry

end NUMINAMATH_CALUDE_estimate_correct_l228_22813


namespace NUMINAMATH_CALUDE_rest_stop_location_l228_22867

/-- The location of the rest stop between two towns -/
theorem rest_stop_location (town_a town_b rest_stop_fraction : ℚ) : 
  town_a = 30 → 
  town_b = 210 → 
  rest_stop_fraction = 4/5 → 
  town_a + rest_stop_fraction * (town_b - town_a) = 174 := by
sorry

end NUMINAMATH_CALUDE_rest_stop_location_l228_22867


namespace NUMINAMATH_CALUDE_excavator_transport_theorem_l228_22886

/-- Represents the transportation problem for excavators after an earthquake. -/
structure ExcavatorTransport where
  area_a_need : ℕ := 27
  area_b_need : ℕ := 25
  province_a_donate : ℕ := 28
  province_b_donate : ℕ := 24
  cost_a_to_a : ℚ := 0.4
  cost_a_to_b : ℚ := 0.3
  cost_b_to_a : ℚ := 0.5
  cost_b_to_b : ℚ := 0.2

/-- The functional relationship between total cost y and number of excavators x
    transported from Province A to Area A. -/
def total_cost (et : ExcavatorTransport) (x : ℕ) : ℚ :=
  et.cost_a_to_a * x + et.cost_a_to_b * (et.province_a_donate - x) +
  et.cost_b_to_a * (et.area_a_need - x) + et.cost_b_to_b * (x - 3)

/-- The theorem stating the functional relationship and range of x. -/
theorem excavator_transport_theorem (et : ExcavatorTransport) :
  ∀ x : ℕ, 3 ≤ x ∧ x ≤ 27 →
    total_cost et x = -0.2 * x + 21.3 ∧
    (∀ y : ℚ, y = total_cost et x → -0.2 * x + 21.3 = y) := by
  sorry

#check excavator_transport_theorem

end NUMINAMATH_CALUDE_excavator_transport_theorem_l228_22886


namespace NUMINAMATH_CALUDE_langsley_commute_time_l228_22801

theorem langsley_commute_time :
  let first_bus_time : ℕ := 40
  let first_bus_delay : ℕ := 10
  let first_wait_time : ℕ := 10
  let second_bus_time : ℕ := 50
  let second_bus_delay : ℕ := 5
  let second_wait_time : ℕ := 15
  let third_bus_time : ℕ := 95
  let third_bus_delay : ℕ := 15
  first_bus_time + first_bus_delay + first_wait_time +
  second_bus_time + second_bus_delay + second_wait_time +
  third_bus_time + third_bus_delay = 240 := by
sorry

end NUMINAMATH_CALUDE_langsley_commute_time_l228_22801


namespace NUMINAMATH_CALUDE_snowfall_total_l228_22805

theorem snowfall_total (morning_snowfall afternoon_snowfall : ℝ) 
  (h1 : morning_snowfall = 0.12)
  (h2 : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.62 := by
sorry

end NUMINAMATH_CALUDE_snowfall_total_l228_22805


namespace NUMINAMATH_CALUDE_cafeteria_pies_l228_22852

theorem cafeteria_pies (total_apples : ℕ) (handout_percentage : ℚ) (apples_per_pie : ℕ) : 
  total_apples = 800 →
  handout_percentage = 65 / 100 →
  apples_per_pie = 15 →
  (total_apples - (total_apples * handout_percentage).floor) / apples_per_pie = 18 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l228_22852


namespace NUMINAMATH_CALUDE_seashell_count_l228_22896

theorem seashell_count (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) 
  (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_l228_22896


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l228_22888

theorem square_perimeter_ratio (x y : ℝ) (h : x * Real.sqrt 2 = 1.5 * y * Real.sqrt 2) :
  (4 * x) / (4 * y) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l228_22888


namespace NUMINAMATH_CALUDE_probability_all_8_cards_l228_22837

/-- Represents a player in the card game --/
structure Player where
  cards : ℕ

/-- Represents the state of the game --/
structure GameState where
  players : Fin 6 → Player
  cardsDealt : ℕ

/-- The dealing process for a single card --/
def dealCard (state : GameState) : GameState :=
  sorry

/-- The final state after dealing all cards --/
def finalState : GameState :=
  sorry

/-- Checks if all players have exactly 8 cards --/
def allPlayersHave8Cards (state : GameState) : Prop :=
  ∀ i : Fin 6, (state.players i).cards = 8

/-- The probability of all players having 8 cards after dealing --/
def probabilityAllHave8Cards : ℚ :=
  sorry

/-- Theorem stating the probability of all players having 8 cards is 5/6 --/
theorem probability_all_8_cards : probabilityAllHave8Cards = 5/6 :=
  sorry

end NUMINAMATH_CALUDE_probability_all_8_cards_l228_22837


namespace NUMINAMATH_CALUDE_same_heads_probability_l228_22828

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes when tossing n pennies -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- The number of ways to get k heads when tossing n pennies -/
def ways_to_get_heads (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of favorable outcomes where Keiko and Ephraim get the same number of heads -/
def favorable_outcomes : ℕ :=
  (ways_to_get_heads keiko_pennies 0 * ways_to_get_heads ephraim_pennies 0) +
  (ways_to_get_heads keiko_pennies 1 * ways_to_get_heads ephraim_pennies 1) +
  (ways_to_get_heads keiko_pennies 2 * ways_to_get_heads ephraim_pennies 2)

/-- The probability of Ephraim getting the same number of heads as Keiko -/
theorem same_heads_probability :
  (favorable_outcomes : ℚ) / (total_outcomes keiko_pennies * total_outcomes ephraim_pennies) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_same_heads_probability_l228_22828


namespace NUMINAMATH_CALUDE_probability_one_or_two_first_20_rows_l228_22818

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (rows : ℕ) : Type := Unit

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := if n ≥ 1 then 2 * n - 1 else 0

/-- The number of 2's in the first n rows of Pascal's Triangle -/
def countTwos (n : ℕ) : ℕ := if n ≥ 3 then 2 * (n - 2) else 0

/-- The probability of selecting either 1 or 2 from the first n rows of Pascal's Triangle -/
def probabilityOneOrTwo (n : ℕ) : ℚ :=
  (countOnes n + countTwos n : ℚ) / (totalElements n : ℚ)

theorem probability_one_or_two_first_20_rows :
  probabilityOneOrTwo 20 = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_probability_one_or_two_first_20_rows_l228_22818


namespace NUMINAMATH_CALUDE_find_k_l228_22854

theorem find_k : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4) → k = -16 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l228_22854


namespace NUMINAMATH_CALUDE_problem_statement_l228_22843

theorem problem_statement (a b c t : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (ht : t ≥ 1) 
  (sum_eq : a + b + c = 1/2) 
  (sqrt_eq : Real.sqrt (a + 1/2 * (b - c)^2) + Real.sqrt b + Real.sqrt c = Real.sqrt (6*t) / 2) :
  a^(2*t) + b^(2*t) + c^(2*t) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l228_22843


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_with_pair_l228_22874

/-- The number of ways to distribute n distinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable objects into k distinguishable boxes,
    where two specific objects must always be together -/
def distributeWithPair (n k : ℕ) : ℕ := k * (distribute (n - 1) k)

theorem five_balls_three_boxes_with_pair :
  distributeWithPair 5 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_with_pair_l228_22874


namespace NUMINAMATH_CALUDE_bobs_age_l228_22853

/-- Given that the sum of Bob's and Carol's ages is 66, and Carol's age is 2 more than 3 times Bob's age, prove that Bob is 16 years old. -/
theorem bobs_age (b c : ℕ) (h1 : b + c = 66) (h2 : c = 3 * b + 2) : b = 16 := by
  sorry

end NUMINAMATH_CALUDE_bobs_age_l228_22853


namespace NUMINAMATH_CALUDE_unqualified_pieces_l228_22871

theorem unqualified_pieces (total_products : ℕ) (pass_rate : ℚ) : 
  total_products = 400 → pass_rate = 98 / 100 → 
  ↑total_products * (1 - pass_rate) = 8 := by
  sorry

end NUMINAMATH_CALUDE_unqualified_pieces_l228_22871


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l228_22803

/-- A function f is even if f(x) = f(-x) for all x in its domain. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = (a - 2)x^2 + (a - 1)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (a - 2) * x^2 + (a - 1) * x + 3

/-- If f(x) = (a - 2)x^2 + (a - 1)x + 3 is an even function, then a = 1 -/
theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l228_22803


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_area_of_overlapping_squares_is_252_l228_22872

/-- The area of the region covered by two congruent squares with side length 12 units,
    where one corner of one square coincides with a corner of the other square. -/
theorem area_of_overlapping_squares : ℝ :=
  let square_side_length : ℝ := 12
  let single_square_area : ℝ := square_side_length ^ 2
  let total_area_without_overlap : ℝ := 2 * single_square_area
  let overlap_area : ℝ := single_square_area / 4
  total_area_without_overlap - overlap_area

/-- The area of the region covered by the two squares is 252 square units. -/
theorem area_of_overlapping_squares_is_252 :
  area_of_overlapping_squares = 252 := by sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_area_of_overlapping_squares_is_252_l228_22872


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l228_22855

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating the property of the geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_sum : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l228_22855


namespace NUMINAMATH_CALUDE_mary_walking_distance_approx_l228_22848

/-- Represents the journey Mary took to her sister's house -/
structure Journey where
  total_distance : ℝ
  bike_speed : ℝ
  walk_speed : ℝ
  bike_portion : ℝ
  total_time : ℝ

/-- Calculates the walking distance for a given journey -/
def walking_distance (j : Journey) : ℝ :=
  (1 - j.bike_portion) * j.total_distance

/-- The theorem stating that Mary's walking distance is approximately 0.3 km -/
theorem mary_walking_distance_approx (j : Journey) 
  (h1 : j.bike_speed = 15)
  (h2 : j.walk_speed = 4)
  (h3 : j.bike_portion = 0.4)
  (h4 : j.total_time = 0.6) : 
  ∃ (ε : ℝ), ε > 0 ∧ abs (walking_distance j - 0.3) < ε := by
  sorry

#check mary_walking_distance_approx

end NUMINAMATH_CALUDE_mary_walking_distance_approx_l228_22848


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l228_22820

theorem pizza_slices_per_person
  (coworkers : ℕ)
  (pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (h1 : coworkers = 18)
  (h2 : pizzas = 4)
  (h3 : slices_per_pizza = 10)
  : (pizzas * slices_per_pizza) / coworkers = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l228_22820


namespace NUMINAMATH_CALUDE_min_mutually_visible_pairs_l228_22836

/-- A configuration of birds on a circle. -/
structure BirdConfiguration where
  /-- The total number of birds. -/
  total_birds : ℕ
  /-- The number of points on the circle where birds can sit. -/
  num_points : ℕ
  /-- The distribution of birds across the points. -/
  distribution : Fin num_points → ℕ
  /-- The sum of birds across all points equals the total number of birds. -/
  sum_constraint : (Finset.univ.sum distribution) = total_birds

/-- The number of mutually visible pairs in a given configuration. -/
def mutually_visible_pairs (config : BirdConfiguration) : ℕ :=
  Finset.sum Finset.univ (fun i => config.distribution i * (config.distribution i - 1) / 2)

/-- The theorem stating the minimum number of mutually visible pairs. -/
theorem min_mutually_visible_pairs :
  ∀ (config : BirdConfiguration),
    config.total_birds = 155 →
    mutually_visible_pairs config ≥ 270 :=
  sorry

end NUMINAMATH_CALUDE_min_mutually_visible_pairs_l228_22836


namespace NUMINAMATH_CALUDE_exp_sum_lt_four_l228_22814

noncomputable def f (x : ℝ) := Real.exp x - x^2 - x

theorem exp_sum_lt_four (x₁ x₂ : ℝ) 
  (h1 : x₁ < Real.log 2) 
  (h2 : x₂ > Real.log 2) 
  (h3 : deriv f x₁ = deriv f x₂) : 
  Real.exp (x₁ + x₂) < 4 := by sorry

end NUMINAMATH_CALUDE_exp_sum_lt_four_l228_22814


namespace NUMINAMATH_CALUDE_positive_solution_form_l228_22895

theorem positive_solution_form (x : ℝ) : 
  x^2 - 18*x = 80 → 
  x > 0 → 
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ x = Real.sqrt c - d ∧ c = 161 ∧ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_form_l228_22895


namespace NUMINAMATH_CALUDE_rectangular_field_area_l228_22868

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 98 rupees at 25 paise per metre has an area of 9408 square meters -/
theorem rectangular_field_area (length width : ℝ) (cost_per_metre : ℚ) (total_cost : ℚ) : 
  length / width = 4 / 3 →
  cost_per_metre = 25 / 100 →
  total_cost = 98 →
  2 * (length + width) * cost_per_metre = total_cost →
  length * width = 9408 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l228_22868


namespace NUMINAMATH_CALUDE_square_divisibility_l228_22847

theorem square_divisibility (m n : ℕ) (h1 : m > n) (h2 : m % 2 = n % 2) 
  (h3 : (m^2 - n^2 + 1) ∣ (n^2 - 1)) : 
  ∃ k : ℕ, m^2 - n^2 + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l228_22847


namespace NUMINAMATH_CALUDE_terrys_trip_distance_l228_22825

/-- Proves that given the conditions of Terry's trip, the total distance driven is 780 miles. -/
theorem terrys_trip_distance :
  ∀ (scenic_road_mpg freeway_mpg : ℝ),
  freeway_mpg = scenic_road_mpg + 6.5 →
  (9 * scenic_road_mpg + 17 * freeway_mpg) / (9 + 17) = 30 →
  9 * scenic_road_mpg + 17 * freeway_mpg = 780 :=
by
  sorry

#check terrys_trip_distance

end NUMINAMATH_CALUDE_terrys_trip_distance_l228_22825


namespace NUMINAMATH_CALUDE_find_number_l228_22804

theorem find_number : ∃ x : ℝ, 1.2 * x = 2 * (0.8 * (x - 20)) ∧ x = 80 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l228_22804


namespace NUMINAMATH_CALUDE_polynomial_factorization_l228_22898

theorem polynomial_factorization (x : ℝ) :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l228_22898


namespace NUMINAMATH_CALUDE_set_operation_equality_l228_22861

universe u

def U : Set (Fin 5) := {0, 1, 2, 3, 4}

def M : Set (Fin 5) := {0, 3}

def N : Set (Fin 5) := {0, 2, 4}

theorem set_operation_equality :
  M ∪ (Mᶜ ∩ N) = {0, 2, 3, 4} :=
by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l228_22861


namespace NUMINAMATH_CALUDE_cards_per_page_l228_22819

/-- Given Will's baseball card organization problem, prove that he puts 3 cards on each page. -/
theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) :
  (new_cards + old_cards) / pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l228_22819


namespace NUMINAMATH_CALUDE_sales_increase_percentage_l228_22876

theorem sales_increase_percentage (original_price : ℝ) (original_quantity : ℝ) 
  (discount_rate : ℝ) (income_increase_rate : ℝ) (new_quantity : ℝ)
  (h1 : discount_rate = 0.1)
  (h2 : income_increase_rate = 0.125)
  (h3 : original_price * original_quantity * (1 + income_increase_rate) = 
        original_price * (1 - discount_rate) * new_quantity) :
  new_quantity / original_quantity - 1 = 0.25 := by
sorry

end NUMINAMATH_CALUDE_sales_increase_percentage_l228_22876


namespace NUMINAMATH_CALUDE_cindy_added_25_pens_l228_22892

/-- Calculates the number of pens Cindy added given the initial count, pens received, pens given away, and final count. -/
def pens_added_by_cindy (initial : ℕ) (received : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial + received - given_away)

/-- Theorem stating that Cindy added 25 pens given the specific conditions of the problem. -/
theorem cindy_added_25_pens :
  pens_added_by_cindy 5 20 10 40 = 25 := by
  sorry

#eval pens_added_by_cindy 5 20 10 40

end NUMINAMATH_CALUDE_cindy_added_25_pens_l228_22892


namespace NUMINAMATH_CALUDE_negative_roots_quadratic_l228_22884

/-- For a quadratic polynomial x^2 + 2(p+1)x + 9p - 5, both roots are negative if and only if 5/9 < p ≤ 1 or p ≥ 6 -/
theorem negative_roots_quadratic (p : ℝ) : 
  (∀ x : ℝ, x^2 + 2*(p+1)*x + 9*p - 5 = 0 → x < 0) ↔ 
  (5/9 < p ∧ p ≤ 1) ∨ p ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_negative_roots_quadratic_l228_22884


namespace NUMINAMATH_CALUDE_marias_carrots_l228_22881

theorem marias_carrots (initial thrown_out picked_more final : ℕ) : 
  thrown_out = 11 →
  picked_more = 15 →
  final = 52 →
  initial - thrown_out + picked_more = final →
  initial = 48 := by
sorry

end NUMINAMATH_CALUDE_marias_carrots_l228_22881


namespace NUMINAMATH_CALUDE_equation_solution_l228_22883

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ 2 * ((1 / x) + (3 / x) / (6 / x)) - (1 / x) = 1.5 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l228_22883


namespace NUMINAMATH_CALUDE_circle_area_l228_22841

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4*y = 3

-- Define the center and radius of the circle
def circle_center : ℝ × ℝ := (-3, 2)
def circle_radius : ℝ := 4

-- Theorem statement
theorem circle_area :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (center = circle_center) ∧
    (radius = circle_radius) ∧
    (Real.pi * radius^2 = 16 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_circle_area_l228_22841


namespace NUMINAMATH_CALUDE_tree_height_from_shadows_l228_22873

/-- Given a tree and a flag pole, calculate the height of the tree using similar triangles -/
theorem tree_height_from_shadows 
  (tree_shadow : ℝ) 
  (pole_height : ℝ) 
  (pole_shadow : ℝ) 
  (h : tree_shadow = 8)
  (i : pole_height = 150)
  (j : pole_shadow = 100) :
  tree_shadow * pole_height / pole_shadow = 12 :=
by sorry

end NUMINAMATH_CALUDE_tree_height_from_shadows_l228_22873


namespace NUMINAMATH_CALUDE_composite_sum_product_l228_22815

def first_composite : ℕ := 4
def second_composite : ℕ := 6
def third_composite : ℕ := 8
def fourth_composite : ℕ := 9
def fifth_composite : ℕ := 10

theorem composite_sum_product : 
  (first_composite * second_composite * third_composite) + 
  (fourth_composite * fifth_composite) = 282 := by
sorry

end NUMINAMATH_CALUDE_composite_sum_product_l228_22815


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_one_l228_22832

-- Define the binary logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem logarithm_expression_equals_one :
  lg 2 * lg 50 + lg 25 - lg 5 * lg 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_one_l228_22832


namespace NUMINAMATH_CALUDE_custom_mult_three_four_l228_22842

/-- Custom multiplication operation -/
def custom_mult (a b : ℤ) : ℤ := 4*a + 3*b - a*b

/-- Theorem stating that 3 * 4 = 12 under the custom multiplication -/
theorem custom_mult_three_four : custom_mult 3 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_three_four_l228_22842


namespace NUMINAMATH_CALUDE_total_buttons_is_117_l228_22808

/-- The number of buttons Jack uses for all shirts -/
def total_buttons : ℕ :=
  let jack_kids : ℕ := 3
  let jack_shirts_per_kid : ℕ := 3
  let jack_buttons_per_shirt : ℕ := 7
  let neighbor_kids : ℕ := 2
  let neighbor_shirts_per_kid : ℕ := 3
  let neighbor_buttons_per_shirt : ℕ := 9
  
  let jack_total_shirts := jack_kids * jack_shirts_per_kid
  let jack_total_buttons := jack_total_shirts * jack_buttons_per_shirt
  
  let neighbor_total_shirts := neighbor_kids * neighbor_shirts_per_kid
  let neighbor_total_buttons := neighbor_total_shirts * neighbor_buttons_per_shirt
  
  jack_total_buttons + neighbor_total_buttons

/-- Theorem stating that the total number of buttons Jack uses is 117 -/
theorem total_buttons_is_117 : total_buttons = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_buttons_is_117_l228_22808


namespace NUMINAMATH_CALUDE_trig_ratios_for_point_l228_22859

theorem trig_ratios_for_point (m : ℝ) (α : ℝ) (h : m < 0) :
  let x : ℝ := 3 * m
  let y : ℝ := -2 * m
  let r : ℝ := Real.sqrt (x^2 + y^2)
  (x, y) = (3 * m, -2 * m) →
  Real.sin α = 2 * Real.sqrt 13 / 13 ∧
  Real.cos α = -(3 * Real.sqrt 13 / 13) ∧
  Real.tan α = -2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_trig_ratios_for_point_l228_22859


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l228_22887

-- Define the vectors
def a : ℝ × ℝ := (2, -1)
def b (m : ℝ) : ℝ × ℝ := (-1, m)
def c : ℝ × ℝ := (-1, 2)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_parallel_condition (m : ℝ) :
  parallel (a.1 + (b m).1, a.2 + (b m).2) c → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l228_22887


namespace NUMINAMATH_CALUDE_june_greatest_drop_l228_22851

/-- Represents the months in the first half of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => 1.50
  | Month.February => -2.25
  | Month.March => 0.75
  | Month.April => -3.00
  | Month.May => 1.00
  | Month.June => -4.00

/-- The month with the greatest price drop -/
def greatest_drop : Month := Month.June

theorem june_greatest_drop :
  ∀ m : Month, price_change m ≥ price_change greatest_drop → m = greatest_drop :=
by sorry

end NUMINAMATH_CALUDE_june_greatest_drop_l228_22851


namespace NUMINAMATH_CALUDE_linear_system_no_solution_l228_22878

/-- A system of two linear equations in two variables -/
structure LinearSystem (a : ℝ) :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)
  (h1 : ∀ x y, eq1 x y = a * x + 2 * y - 3)
  (h2 : ∀ x y, eq2 x y = 2 * x + a * y - 2)

/-- The system has no solution -/
def NoSolution (s : LinearSystem a) : Prop :=
  ∀ x y, ¬(s.eq1 x y = 0 ∧ s.eq2 x y = 0)

theorem linear_system_no_solution (a : ℝ) :
  (∃ s : LinearSystem a, NoSolution s) → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_no_solution_l228_22878


namespace NUMINAMATH_CALUDE_sum_of_squares_equality_l228_22877

variables {a b c x y z : ℝ}

theorem sum_of_squares_equality 
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 0)
  : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equality_l228_22877


namespace NUMINAMATH_CALUDE_max_value_3cos_minus_sin_l228_22829

theorem max_value_3cos_minus_sin :
  ∀ x : ℝ, 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 ∧
  ∃ x : ℝ, 3 * Real.cos x - Real.sin x = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_3cos_minus_sin_l228_22829


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l228_22806

/-- Given a geometric sequence where the first three terms are a-1, a+1, and a+4,
    prove that the general term formula is a_n = 4 × (3/2)^(n-1) -/
theorem geometric_sequence_general_term (a : ℝ) (n : ℕ) :
  (a - 1 : ℝ) * (a + 4 : ℝ) = (a + 1 : ℝ)^2 →
  ∃ (seq : ℕ → ℝ), seq 1 = a - 1 ∧ seq 2 = a + 1 ∧ seq 3 = a + 4 ∧
    (∀ k : ℕ, seq (k + 1) / seq k = seq 2 / seq 1) →
    ∀ m : ℕ, seq m = 4 * (3/2 : ℝ)^(m - 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l228_22806


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l228_22844

/-- Given a car traveling for two hours with a speed of 80 km/h in the first hour
    and an average speed of 60 km/h over the two hours,
    prove that the speed in the second hour must be 40 km/h. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 80)
  (h2 : average_speed = 60) :
  let speed_second_hour := 2 * average_speed - speed_first_hour
  speed_second_hour = 40 := by
sorry


end NUMINAMATH_CALUDE_car_speed_second_hour_l228_22844


namespace NUMINAMATH_CALUDE_candy_bar_difference_l228_22809

theorem candy_bar_difference (lena kevin nicole : ℕ) : 
  lena = 16 →
  lena + 5 = 3 * kevin →
  kevin + 4 = nicole →
  lena - nicole = 5 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l228_22809


namespace NUMINAMATH_CALUDE_unique_factorial_sum_l228_22821

/-- factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Function to get the hundreds digit of a natural number -/
def hundreds_digit (n : ℕ) : ℕ := 
  (n / 100) % 10

/-- Function to get the tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ := 
  (n / 10) % 10

/-- Function to get the units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := 
  n % 10

/-- Theorem stating that 145 is the only three-digit number with 1 as its hundreds digit 
    that is equal to the sum of the factorials of its digits -/
theorem unique_factorial_sum : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ hundreds_digit n = 1 → 
  (n = factorial (hundreds_digit n) + factorial (tens_digit n) + factorial (units_digit n) ↔ n = 145) := by
  sorry

end NUMINAMATH_CALUDE_unique_factorial_sum_l228_22821


namespace NUMINAMATH_CALUDE_cherry_tomatoes_weight_l228_22846

/-- Calculates the total weight of cherry tomatoes in grams -/
def total_weight_grams (initial_kg : ℝ) (additional_g : ℝ) : ℝ :=
  initial_kg * 1000 + additional_g

/-- Theorem: The total weight of cherry tomatoes is 2560 grams -/
theorem cherry_tomatoes_weight :
  total_weight_grams 2 560 = 2560 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomatoes_weight_l228_22846


namespace NUMINAMATH_CALUDE_sibling_product_l228_22811

/-- Represents a family with a specific sibling structure -/
structure Family :=
  (total_sisters : ℕ)
  (total_brothers : ℕ)

/-- Calculates the number of sisters for a given family member -/
def sisters_count (f : Family) : ℕ := f.total_sisters - 1

/-- Calculates the number of brothers for a given family member -/
def brothers_count (f : Family) : ℕ := f.total_brothers

theorem sibling_product (f : Family) 
  (h1 : f.total_sisters = 5) 
  (h2 : f.total_brothers = 7) : 
  sisters_count f * brothers_count f = 24 := by
  sorry

#check sibling_product

end NUMINAMATH_CALUDE_sibling_product_l228_22811


namespace NUMINAMATH_CALUDE_jimmy_garden_servings_l228_22830

/-- The number of servings produced by a carrot plant -/
def carrot_servings : ℕ := 4

/-- The number of green bean plants -/
def green_bean_plants : ℕ := 10

/-- The number of carrot plants -/
def carrot_plants : ℕ := 8

/-- The number of corn plants -/
def corn_plants : ℕ := 12

/-- The number of tomato plants -/
def tomato_plants : ℕ := 15

/-- The number of servings produced by a corn plant -/
def corn_servings : ℕ := 5 * carrot_servings

/-- The number of servings produced by a green bean plant -/
def green_bean_servings : ℕ := corn_servings / 2

/-- The number of servings produced by a tomato plant -/
def tomato_servings : ℕ := carrot_servings + 3

/-- The total number of servings in Jimmy's garden -/
def total_servings : ℕ :=
  green_bean_plants * green_bean_servings +
  carrot_plants * carrot_servings +
  corn_plants * corn_servings +
  tomato_plants * tomato_servings

theorem jimmy_garden_servings :
  total_servings = 477 := by sorry

end NUMINAMATH_CALUDE_jimmy_garden_servings_l228_22830


namespace NUMINAMATH_CALUDE_range_of_m_l228_22812

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - m| < 1 → x^2 - 8*x + 12 < 0) ∧ 
  (∃ x, x^2 - 8*x + 12 < 0 ∧ |x - m| ≥ 1)) →
  (3 ≤ m ∧ m ≤ 5) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l228_22812


namespace NUMINAMATH_CALUDE_subtract_equations_l228_22863

theorem subtract_equations (x y : ℝ) :
  (4 * x - 3 * y = 2) ∧ (4 * x + y = 10) → 4 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_equations_l228_22863


namespace NUMINAMATH_CALUDE_nested_cube_roots_l228_22838

theorem nested_cube_roots (N M : ℝ) (hN : N > 1) (hM : M > 1) :
  (N * (M * (N * M^(1/3))^(1/3))^(1/3))^(1/3) = N^(2/3) * M^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_roots_l228_22838


namespace NUMINAMATH_CALUDE_cylinder_diagonal_angle_l228_22831

theorem cylinder_diagonal_angle (m n : ℝ) (h : m > 0 ∧ n > 0) :
  let α := if m / n < Real.pi / 4 
           then 2 * Real.arctan (4 * m / (Real.pi * n))
           else 2 * Real.arctan (Real.pi * n / (4 * m))
  ∃ (R H : ℝ), R > 0 ∧ H > 0 ∧ 
    (Real.pi * R^2) / (2 * R * H) = m / n ∧
    α = Real.arctan (2 * R / H) + Real.arctan (2 * R / H) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_diagonal_angle_l228_22831


namespace NUMINAMATH_CALUDE_sheetrock_width_l228_22858

/-- Given a rectangular piece of sheetrock with length 6 feet and area 30 square feet, its width is 5 feet. -/
theorem sheetrock_width (length : ℝ) (area : ℝ) (width : ℝ) : 
  length = 6 → area = 30 → area = length * width → width = 5 := by
  sorry

end NUMINAMATH_CALUDE_sheetrock_width_l228_22858


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l228_22827

theorem shaded_area_between_circles (r : Real) : 
  r > 0 → -- radius of smaller circle is positive
  (2 * r = 6) → -- diameter of smaller circle is 6 units
  π * (3 * r)^2 - π * r^2 = 72 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l228_22827


namespace NUMINAMATH_CALUDE_arthur_walked_seven_miles_l228_22824

/-- The distance Arthur walked in miles -/
def arthur_distance (blocks_east blocks_north blocks_west : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north + blocks_west : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 7 miles -/
theorem arthur_walked_seven_miles :
  arthur_distance 8 15 5 (1/4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walked_seven_miles_l228_22824


namespace NUMINAMATH_CALUDE_gym_treadmills_l228_22802

def gym_problem (num_gyms : ℕ) (bikes_per_gym : ℕ) (ellipticals_per_gym : ℕ) 
  (bike_cost : ℚ) (total_cost : ℚ) : Prop :=
  let treadmill_cost : ℚ := bike_cost * (3/2)
  let elliptical_cost : ℚ := treadmill_cost * 2
  let total_bike_cost : ℚ := num_gyms * bikes_per_gym * bike_cost
  let total_elliptical_cost : ℚ := num_gyms * ellipticals_per_gym * elliptical_cost
  let treadmill_cost_per_gym : ℚ := (total_cost - total_bike_cost - total_elliptical_cost) / num_gyms
  let treadmills_per_gym : ℚ := treadmill_cost_per_gym / treadmill_cost
  treadmills_per_gym = 5

theorem gym_treadmills : 
  gym_problem 20 10 5 700 455000 := by
  sorry

end NUMINAMATH_CALUDE_gym_treadmills_l228_22802


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_prism_l228_22826

/-- The volume of a sphere that circumscribes a rectangular prism with dimensions 2 × 1 × 1 is √6π -/
theorem sphere_volume_circumscribing_prism :
  let l : ℝ := 2
  let w : ℝ := 1
  let h : ℝ := 1
  let diagonal := Real.sqrt (l^2 + w^2 + h^2)
  let radius := diagonal / 2
  let volume := (4/3) * Real.pi * radius^3
  volume = Real.sqrt 6 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_prism_l228_22826


namespace NUMINAMATH_CALUDE_paper_pack_sheets_l228_22822

theorem paper_pack_sheets : ∃ (S P : ℕ), S = 115 ∧ S - P = 100 ∧ 5 * P + 35 = S := by
  sorry

end NUMINAMATH_CALUDE_paper_pack_sheets_l228_22822


namespace NUMINAMATH_CALUDE_cube_root_of_number_with_given_square_roots_l228_22864

theorem cube_root_of_number_with_given_square_roots (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (3*a + 1)^2 = x ∧ (a + 11)^2 = x) →
  ∃ (y : ℝ), y^3 = x ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_number_with_given_square_roots_l228_22864


namespace NUMINAMATH_CALUDE_white_balls_fewest_l228_22889

/-- Represents the number of balls of each color -/
structure BallCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- The conditions of the ball counting problem -/
def ballProblemConditions (counts : BallCounts) : Prop :=
  counts.red + counts.blue + counts.white = 108 ∧
  counts.blue = counts.red / 3 ∧
  counts.white = counts.blue / 2

theorem white_balls_fewest (counts : BallCounts) 
  (h : ballProblemConditions counts) : 
  counts.white = 12 ∧ 
  counts.white < counts.blue ∧ 
  counts.white < counts.red :=
sorry

end NUMINAMATH_CALUDE_white_balls_fewest_l228_22889


namespace NUMINAMATH_CALUDE_y2k_game_second_player_strategy_l228_22800

/-- Represents a player in the Y2K Game -/
inductive Player : Type
  | First : Player
  | Second : Player

/-- Represents a letter that can be placed on the board -/
inductive Letter : Type
  | S : Letter
  | O : Letter

/-- Represents the state of a square on the board -/
inductive Square : Type
  | Empty : Square
  | Filled : Letter → Square

/-- Represents the game board -/
def Board : Type := Fin 2000 → Square

/-- Represents a move in the game -/
structure Move where
  position : Fin 2000
  letter : Letter

/-- Represents the game state -/
structure GameState where
  board : Board
  currentPlayer : Player

/-- Represents a strategy for a player -/
def Strategy : Type := GameState → Move

/-- Checks if a player has won the game -/
def hasWon (board : Board) (player : Player) : Prop := sorry

/-- Checks if the game is a draw -/
def isDraw (board : Board) : Prop := sorry

/-- The Y2K Game theorem -/
theorem y2k_game_second_player_strategy :
  ∃ (strategy : Strategy),
    ∀ (initialState : GameState),
      initialState.currentPlayer = Player.Second →
        (∃ (finalState : GameState),
          (hasWon finalState.board Player.Second ∨ isDraw finalState.board)) :=
sorry

end NUMINAMATH_CALUDE_y2k_game_second_player_strategy_l228_22800


namespace NUMINAMATH_CALUDE_san_antonio_austin_bus_passes_l228_22865

/-- Represents the time interval between bus departures in hours -/
def departure_interval : ℕ := 2

/-- Represents the duration of the journey between cities in hours -/
def journey_duration : ℕ := 7

/-- Represents the offset between San Antonio and Austin departures in hours -/
def departure_offset : ℕ := 1

/-- Calculates the number of buses passed during the journey -/
def buses_passed : ℕ :=
  journey_duration / departure_interval + 1

theorem san_antonio_austin_bus_passes :
  buses_passed = 4 :=
sorry

end NUMINAMATH_CALUDE_san_antonio_austin_bus_passes_l228_22865


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_complement_B_union_A_a_range_for_C_subset_B_l228_22882

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorems
theorem complement_intersection_A_B :
  (Set.univ : Set ℝ) \ (A ∩ B) = {x | x < 3 ∨ x ≥ 6} := by sorry

theorem complement_B_union_A :
  ((Set.univ : Set ℝ) \ B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} := by sorry

theorem a_range_for_C_subset_B :
  {a : ℝ | C a ⊆ B} = Set.Icc 2 8 := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_complement_B_union_A_a_range_for_C_subset_B_l228_22882


namespace NUMINAMATH_CALUDE_division_remainder_l228_22891

def largest_three_digit : Nat := 975
def smallest_two_digit : Nat := 23

theorem division_remainder :
  largest_three_digit % smallest_two_digit = 9 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l228_22891


namespace NUMINAMATH_CALUDE_tree_growth_fraction_l228_22839

/-- Represents the growth of a tree over time -/
def TreeGrowth (initial_height : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_height + growth_rate * years

theorem tree_growth_fraction :
  let initial_height : ℝ := 4
  let growth_rate : ℝ := 0.5
  let height_at_4_years := TreeGrowth initial_height growth_rate 4
  let height_at_6_years := TreeGrowth initial_height growth_rate 6
  (height_at_6_years - height_at_4_years) / height_at_4_years = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_fraction_l228_22839


namespace NUMINAMATH_CALUDE_soap_calculation_l228_22835

/-- Given a number of packs and bars per pack, calculates the total number of bars -/
def total_bars (packs : ℕ) (bars_per_pack : ℕ) : ℕ := packs * bars_per_pack

/-- Theorem stating that 6 packs with 5 bars each results in 30 total bars -/
theorem soap_calculation : total_bars 6 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_soap_calculation_l228_22835


namespace NUMINAMATH_CALUDE_problem_solution_l228_22833

theorem problem_solution (a b c d e x : ℝ) 
  (h : ((x + a) ^ b) / c - d = e / 2) : 
  x = (c * e / 2 + c * d) ^ (1 / b) - a := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l228_22833


namespace NUMINAMATH_CALUDE_john_distance_l228_22899

theorem john_distance (john jill jim : ℝ) 
  (h1 : jill = john - 5)
  (h2 : jim = 0.2 * jill)
  (h3 : jim = 2) : 
  john = 15 := by
sorry

end NUMINAMATH_CALUDE_john_distance_l228_22899


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l228_22885

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  ¬(((a > 0 ∧ b > 0) → (a * b < ((a + b) / 2)^2)) ∧
    ((a * b < ((a + b) / 2)^2) → (a > 0 ∧ b > 0))) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l228_22885


namespace NUMINAMATH_CALUDE_sum_of_squares_l228_22850

theorem sum_of_squares (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 3)
  (h2 : a / x + b / y + c / z = 0)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l228_22850


namespace NUMINAMATH_CALUDE_set_operation_equality_l228_22879

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define sets A and B
def A : Set Nat := {0, 3, 4}
def B : Set Nat := {1, 3}

-- State the theorem
theorem set_operation_equality :
  (Aᶜ ∪ A) ∪ B = {1, 2, 3} :=
by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l228_22879


namespace NUMINAMATH_CALUDE_hotel_cost_per_night_l228_22834

theorem hotel_cost_per_night (nights : ℕ) (discount : ℕ) (total_paid : ℕ) (cost_per_night : ℕ) : 
  nights = 3 → 
  discount = 100 → 
  total_paid = 650 → 
  nights * cost_per_night - discount = total_paid → 
  cost_per_night = 250 := by
sorry

end NUMINAMATH_CALUDE_hotel_cost_per_night_l228_22834


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l228_22893

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = 3 * p.1 + 4}
def N : Set (ℝ × ℝ) := {p | p.2 = p.1 ^ 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {(-1, 1), (4, 16)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l228_22893


namespace NUMINAMATH_CALUDE_equation_solution_l228_22845

theorem equation_solution : 
  ∀ x y : ℕ, x^2 + x*y = y + 92 ↔ (x = 2 ∧ y = 88) ∨ (x = 8 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l228_22845
