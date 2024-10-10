import Mathlib

namespace bouquet_cost_l1234_123496

/-- The cost of the bouquet given Michael's budget and other expenses --/
theorem bouquet_cost (michael_money : ℕ) (cake_cost : ℕ) (balloons_cost : ℕ) (extra_needed : ℕ) : 
  michael_money = 50 →
  cake_cost = 20 →
  balloons_cost = 5 →
  extra_needed = 11 →
  michael_money + extra_needed = cake_cost + balloons_cost + 36 := by
sorry

end bouquet_cost_l1234_123496


namespace vector_parallel_value_l1234_123464

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem vector_parallel_value (x : ℝ) :
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (6, 8)
  parallel a b → x = 8/3 := by
  sorry

end vector_parallel_value_l1234_123464


namespace range_of_f_minus_x_l1234_123477

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x < -3 then -4
  else if x < -2 then -2
  else if x < -1 then -1
  else if x < 0 then 0
  else if x < 1 then 1
  else if x < 2 then 2
  else if x < 3 then 3
  else 4

-- Define the domain
def domain : Set ℝ := Set.Icc (-4) 4

-- State the theorem
theorem range_of_f_minus_x :
  Set.range (fun x => f x - x) ∩ (Set.Icc 0 1) = Set.Icc 0 1 :=
sorry

end range_of_f_minus_x_l1234_123477


namespace intersection_M_N_l1234_123423

def M : Set ℝ := {0, 1, 2}

def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end intersection_M_N_l1234_123423


namespace three_digit_odd_count_l1234_123482

theorem three_digit_odd_count : 
  (Finset.filter 
    (fun n => n ≥ 100 ∧ n < 1000 ∧ n % 2 = 1) 
    (Finset.range 1000)).card = 450 := by
  sorry

end three_digit_odd_count_l1234_123482


namespace trigonometric_identity_l1234_123412

theorem trigonometric_identity (x : ℝ) : 
  8.435 * (Real.sin (3 * x))^10 + (Real.cos (3 * x))^10 = 
  4 * ((Real.sin (3 * x))^6 + (Real.cos (3 * x))^6) / 
  (4 * (Real.cos (6 * x))^2 + (Real.sin (6 * x))^2) ↔ 
  ∃ k : ℤ, x = k * Real.pi / 6 := by
sorry

end trigonometric_identity_l1234_123412


namespace max_value_fraction_l1234_123432

theorem max_value_fraction (x : ℝ) : 
  (3 * x^2 + 9 * x + 17) / (3 * x^2 + 9 * x + 7) ≤ 41 ∧ 
  ∃ y : ℝ, (3 * y^2 + 9 * y + 17) / (3 * y^2 + 9 * y + 7) = 41 := by
  sorry

end max_value_fraction_l1234_123432


namespace geometric_sequence_fourth_term_l1234_123443

/-- Given a geometric sequence where the first three terms are x, 3x+3, and 6x+6,
    this theorem proves that the fourth term is -24. -/
theorem geometric_sequence_fourth_term :
  ∀ x : ℝ,
  (3*x + 3)^2 = x*(6*x + 6) →
  ∃ (a r : ℝ),
    (a = x) ∧
    (a * r = 3*x + 3) ∧
    (a * r^2 = 6*x + 6) ∧
    (a * r^3 = -24) :=
by sorry

end geometric_sequence_fourth_term_l1234_123443


namespace prob_first_ace_equal_l1234_123438

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)

/-- Represents the card game -/
structure CardGame :=
  (num_players : ℕ)
  (deck : Deck)

/-- The probability of a player getting the first ace -/
def prob_first_ace (game : CardGame) (player : ℕ) : ℚ :=
  1 / game.num_players

theorem prob_first_ace_equal (game : CardGame) (player : ℕ) 
  (h1 : game.num_players = 4)
  (h2 : game.deck.total_cards = 32)
  (h3 : game.deck.num_aces = 4)
  (h4 : player ≤ game.num_players) :
  prob_first_ace game player = 1/8 :=
sorry

#check prob_first_ace_equal

end prob_first_ace_equal_l1234_123438


namespace stating_min_time_to_find_faulty_bulb_l1234_123400

/-- Represents the time in seconds for a single bulb operation (screwing or unscrewing) -/
def bulb_operation_time : ℕ := 10

/-- Represents the total number of bulbs in the series -/
def total_bulbs : ℕ := 4

/-- Represents the number of spare bulbs available -/
def spare_bulbs : ℕ := 1

/-- Represents the number of faulty bulbs in the series -/
def faulty_bulbs : ℕ := 1

/-- 
Theorem stating that the minimum time to identify a faulty bulb 
in a series of 4 bulbs is 60 seconds, given the conditions of the problem.
-/
theorem min_time_to_find_faulty_bulb : 
  (bulb_operation_time * 2 * (total_bulbs - 1) : ℕ) = 60 := by
  sorry

end stating_min_time_to_find_faulty_bulb_l1234_123400


namespace pages_read_first_day_l1234_123498

theorem pages_read_first_day (total_pages : ℕ) (days : ℕ) (first_day_pages : ℕ) : 
  total_pages = 130 →
  days = 7 →
  total_pages = first_day_pages + (days - 1) * (2 * first_day_pages) →
  first_day_pages = 10 :=
by sorry

end pages_read_first_day_l1234_123498


namespace absolute_value_equals_self_implies_nonnegative_l1234_123453

theorem absolute_value_equals_self_implies_nonnegative (a : ℝ) : (|a| = a) → a ≥ 0 := by
  sorry

end absolute_value_equals_self_implies_nonnegative_l1234_123453


namespace simplified_expression_ratio_l1234_123486

theorem simplified_expression_ratio (k : ℤ) : 
  let simplified := (6 * k + 12) / 6
  let a : ℤ := 1
  let b : ℤ := 2
  simplified = a * k + b ∧ a / b = 1 / 2 := by
sorry

end simplified_expression_ratio_l1234_123486


namespace expression_evaluation_l1234_123472

theorem expression_evaluation : (25 * 5 + 5^2) / (5^2 - 15) = 15 := by
  sorry

end expression_evaluation_l1234_123472


namespace factor_polynomial_l1234_123428

theorem factor_polynomial (x : ℝ) :
  x^4 - 36*x^2 + 25 = (x^2 - 6*x + 5) * (x^2 + 6*x + 5) := by
  sorry

end factor_polynomial_l1234_123428


namespace student_arrangement_theorem_l1234_123418

/-- The number of ways to arrange 6 students in two rows of three, 
    with the taller student in each column in the back row -/
def arrangement_count : ℕ := 90

/-- The number of students -/
def num_students : ℕ := 6

/-- The number of students in each row -/
def students_per_row : ℕ := 3

theorem student_arrangement_theorem :
  (num_students = 6) →
  (students_per_row = 3) →
  (∀ n : ℕ, n ≤ num_students → n > 0 → ∃! h : ℕ, h = n) →  -- All students have different heights
  arrangement_count = 90 :=
by sorry

end student_arrangement_theorem_l1234_123418


namespace key_cleaning_time_l1234_123402

/-- The time it takes to clean one key -/
def clean_time : ℝ := 3

theorem key_cleaning_time :
  let assignment_time : ℝ := 10
  let remaining_keys : ℕ := 14
  let total_time : ℝ := 52
  clean_time * remaining_keys + assignment_time = total_time :=
by sorry

end key_cleaning_time_l1234_123402


namespace star_example_l1234_123419

-- Define the star operation
def star (x y : ℚ) : ℚ := (x + y) / 4

-- Theorem statement
theorem star_example : star (star 3 8) 6 = 35 / 16 := by
  sorry

end star_example_l1234_123419


namespace parabola_directrix_tangent_to_circle_l1234_123416

/-- The value of p for which the directrix of the parabola y² = 2px (p > 0) 
    is tangent to the circle x² + y² - 4x + 2y - 4 = 0 -/
theorem parabola_directrix_tangent_to_circle :
  ∀ p : ℝ, p > 0 →
  (∀ x y : ℝ, y^2 = 2*p*x) →
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y - 4 = 0) →
  (∃ x y : ℝ, x = -p/2 ∧ (x - 2)^2 + (y + 1)^2 = 9) →
  p = 2 :=
by sorry

end parabola_directrix_tangent_to_circle_l1234_123416


namespace sqrt_50_between_consecutive_integers_product_l1234_123463

theorem sqrt_50_between_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 :=
by sorry

end sqrt_50_between_consecutive_integers_product_l1234_123463


namespace antiderivative_increment_l1234_123476

-- Define the function f(x) = 2x + 4
def f (x : ℝ) : ℝ := 2 * x + 4

-- Define what it means for F to be an antiderivative of f on the interval [-2, 0]
def is_antiderivative (F : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) 0, (deriv F) x = f x

-- Theorem statement
theorem antiderivative_increment (F : ℝ → ℝ) (h : is_antiderivative F) :
  F 0 - F (-2) = 4 := by sorry

end antiderivative_increment_l1234_123476


namespace chocolate_price_after_discount_l1234_123401

/-- The final price of a chocolate after discount -/
def final_price (original_cost discount : ℚ) : ℚ :=
  original_cost - discount

/-- Theorem: The final price of a chocolate with original cost $2 and discount $0.57 is $1.43 -/
theorem chocolate_price_after_discount :
  final_price 2 0.57 = 1.43 := by
  sorry

end chocolate_price_after_discount_l1234_123401


namespace drews_lawn_width_l1234_123444

def lawn_problem (bag_coverage : ℝ) (length : ℝ) (num_bags : ℕ) (extra_coverage : ℝ) : Prop :=
  let total_coverage := bag_coverage * num_bags
  let actual_lawn_area := total_coverage - extra_coverage
  let width := actual_lawn_area / length
  width = 36

theorem drews_lawn_width :
  lawn_problem 250 22 4 208 := by
  sorry

end drews_lawn_width_l1234_123444


namespace monthly_compound_greater_than_annual_l1234_123445

def annual_rate : ℝ := 0.03

theorem monthly_compound_greater_than_annual (t : ℝ) (h : t > 0) :
  (1 + annual_rate)^t < (1 + annual_rate / 12)^(12 * t) := by
  sorry


end monthly_compound_greater_than_annual_l1234_123445


namespace correct_average_l1234_123407

theorem correct_average (numbers : Finset ℕ) (incorrect_sum : ℕ) (incorrect_number correct_number : ℕ) :
  numbers.card = 10 →
  incorrect_sum = 17 * 10 →
  incorrect_number = 26 →
  correct_number = 56 →
  (incorrect_sum - incorrect_number + correct_number) / numbers.card = 20 := by
  sorry

end correct_average_l1234_123407


namespace calorie_allowance_for_longevity_l1234_123427

/-- Calculates the weekly calorie allowance for a person in their 60s aiming to live to 100 years old -/
def weeklyCalorieAllowance (averageDailyAllowance : ℕ) (reduction : ℕ) (daysInWeek : ℕ) : ℕ :=
  (averageDailyAllowance - reduction) * daysInWeek

/-- Theorem stating the weekly calorie allowance for a person in their 60s aiming to live to 100 years old -/
theorem calorie_allowance_for_longevity :
  weeklyCalorieAllowance 2000 500 7 = 10500 := by
  sorry

#eval weeklyCalorieAllowance 2000 500 7

end calorie_allowance_for_longevity_l1234_123427


namespace number_added_after_doubling_l1234_123411

theorem number_added_after_doubling (x : ℕ) (y : ℕ) (h : x = 13) :
  3 * (2 * x + y) = 99 → y = 7 := by
  sorry

end number_added_after_doubling_l1234_123411


namespace sqrt2_minus1_power_representation_l1234_123493

theorem sqrt2_minus1_power_representation (n : ℤ) :
  ∃ (N : ℕ), (Real.sqrt 2 - 1) ^ n = Real.sqrt N - Real.sqrt (N - 1) :=
sorry

end sqrt2_minus1_power_representation_l1234_123493


namespace train_speed_conversion_l1234_123426

theorem train_speed_conversion (speed_kmph : ℝ) (speed_ms : ℝ) : 
  speed_kmph = 216 → speed_ms = 60 → speed_kmph * (1000 / 3600) = speed_ms :=
by
  sorry

end train_speed_conversion_l1234_123426


namespace each_score_is_individual_l1234_123484

/-- Represents a candidate's math score -/
structure MathScore where
  score : ℝ

/-- Represents the population of candidates -/
structure Population where
  candidates : Finset MathScore
  size_gt_100000 : candidates.card > 100000

/-- Represents a sample of candidates -/
structure Sample where
  scores : Finset MathScore
  size_eq_1000 : scores.card = 1000

/-- Theorem stating that each math score in the sample is an individual data point -/
theorem each_score_is_individual (pop : Population) (sample : Sample) 
  (h_sample : ∀ s ∈ sample.scores, s ∈ pop.candidates) :
  ∀ s ∈ sample.scores, ∃! i : MathScore, i = s :=
sorry

end each_score_is_individual_l1234_123484


namespace function_through_point_l1234_123405

/-- Given a function f(x) = x^α that passes through the point (2, √2), prove that f(9) = 3 -/
theorem function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  f 2 = Real.sqrt 2 →
  f 9 = 3 := by
sorry

end function_through_point_l1234_123405


namespace hyperbola_vertices_distance_l1234_123473

/-- The distance between the vertices of a hyperbola with equation x²/144 - y²/64 = 1 is 24. -/
theorem hyperbola_vertices_distance : 
  ∃ (x y : ℝ), x^2/144 - y^2/64 = 1 → 
  ∃ (v₁ v₂ : ℝ × ℝ), (v₁.1^2/144 - v₁.2^2/64 = 1) ∧ 
                     (v₂.1^2/144 - v₂.2^2/64 = 1) ∧ 
                     (v₁.2 = 0) ∧ (v₂.2 = 0) ∧
                     (v₁.1 = -v₂.1) ∧
                     (Real.sqrt ((v₁.1 - v₂.1)^2 + (v₁.2 - v₂.2)^2) = 24) :=
by sorry

end hyperbola_vertices_distance_l1234_123473


namespace zero_in_interval_l1234_123461

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log x + 2 * x^2 - 4 * x

theorem zero_in_interval :
  ∃ (c : ℝ), c ∈ Set.Ioo 1 (exp 1) ∧ f c = 0 :=
sorry

end zero_in_interval_l1234_123461


namespace system_solution_unique_l1234_123488

theorem system_solution_unique :
  ∃! (x y z : ℝ), x + y + z = 11 ∧ x^2 + 2*y^2 + 3*z^2 = 66 ∧ x = 6 ∧ y = 3 ∧ z = 2 := by
sorry

end system_solution_unique_l1234_123488


namespace juan_oranges_picked_l1234_123448

theorem juan_oranges_picked (total : ℕ) (del_per_day : ℕ) (del_days : ℕ) : 
  total = 107 → del_per_day = 23 → del_days = 2 → 
  total - (del_per_day * del_days) = 61 := by
  sorry

end juan_oranges_picked_l1234_123448


namespace solve_for_T_l1234_123468

theorem solve_for_T : ∃ T : ℚ, (1/3 : ℚ) * (1/6 : ℚ) * T = (1/4 : ℚ) * (1/8 : ℚ) * 120 ∧ T = 67.5 := by
  sorry

end solve_for_T_l1234_123468


namespace binary_1011_is_11_decimal_124_is_octal_174_l1234_123487

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem binary_1011_is_11 :
  binary_to_decimal [true, true, false, true] = 11 := by sorry

theorem decimal_124_is_octal_174 :
  decimal_to_octal 124 = [1, 7, 4] := by sorry

end binary_1011_is_11_decimal_124_is_octal_174_l1234_123487


namespace tourist_guide_distribution_l1234_123447

theorem tourist_guide_distribution :
  let n_tourists : ℕ := 8
  let n_guides : ℕ := 3
  let total_distributions := n_guides ^ n_tourists
  let at_least_one_empty := n_guides * (n_guides - 1) ^ n_tourists
  let at_least_two_empty := n_guides * 1 ^ n_tourists
  total_distributions - at_least_one_empty + at_least_two_empty = 5796 :=
by sorry

end tourist_guide_distribution_l1234_123447


namespace cubic_root_between_integers_l1234_123456

theorem cubic_root_between_integers : ∃ (A B : ℤ), 
  B = A + 1 ∧ 
  ∃ (x : ℝ), A < x ∧ x < B ∧ x^3 + 5*x^2 - 3*x + 1 = 0 := by
  sorry

end cubic_root_between_integers_l1234_123456


namespace quadratic_minimum_l1234_123434

/-- Given a quadratic function y = x^2 + px + q + r where the minimum value is -r, 
    prove that q = p^2 / 4 -/
theorem quadratic_minimum (p q r : ℝ) : 
  (∀ x, x^2 + p*x + q + r ≥ -r) → 
  (∃ x, x^2 + p*x + q + r = -r) → 
  q = p^2 / 4 := by
  sorry

end quadratic_minimum_l1234_123434


namespace K_3_15_10_l1234_123441

noncomputable def K (x y z : ℝ) : ℝ := x / y + y / z + z / x

theorem K_3_15_10 : K 3 15 10 = 151 / 30 := by sorry

end K_3_15_10_l1234_123441


namespace negation_of_divisible_by_two_is_even_l1234_123467

theorem negation_of_divisible_by_two_is_even :
  ¬(∀ n : ℤ, 2 ∣ n → Even n) ↔ ∃ n : ℤ, 2 ∣ n ∧ ¬Even n :=
by sorry

end negation_of_divisible_by_two_is_even_l1234_123467


namespace magician_earnings_l1234_123489

/-- Calculates the money earned from selling magic card decks -/
def money_earned (price_per_deck : ℕ) (initial_decks : ℕ) (final_decks : ℕ) : ℕ :=
  (initial_decks - final_decks) * price_per_deck

/-- Proves that the magician earned 56 dollars -/
theorem magician_earnings :
  let price_per_deck : ℕ := 7
  let initial_decks : ℕ := 16
  let final_decks : ℕ := 8
  money_earned price_per_deck initial_decks final_decks = 56 := by
  sorry

end magician_earnings_l1234_123489


namespace sin_cos_sum_equals_sqrt2_over_2_l1234_123469

theorem sin_cos_sum_equals_sqrt2_over_2 :
  Real.sin (347 * π / 180) * Real.cos (148 * π / 180) +
  Real.sin (77 * π / 180) * Real.cos (58 * π / 180) =
  Real.sqrt 2 / 2 := by
  sorry

end sin_cos_sum_equals_sqrt2_over_2_l1234_123469


namespace valid_a_values_l1234_123406

/-- Set A defined by the quadratic equation x^2 - 2x - 3 = 0 -/
def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}

/-- Set B defined by the linear equation ax - 1 = 0, parameterized by a -/
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

/-- The set of values for a such that B is a subset of A -/
def valid_a : Set ℝ := {a | B a ⊆ A}

/-- Theorem stating that the set of valid a values is {-1, 0, 1/3} -/
theorem valid_a_values : valid_a = {-1, 0, 1/3} := by sorry

end valid_a_values_l1234_123406


namespace roots_product_l1234_123442

theorem roots_product (b c : ℤ) : 
  (∀ s : ℝ, s^2 - 2*s - 1 = 0 → s^5 - b*s^3 - c*s^2 = 0) → 
  b * c = 348 := by
sorry

end roots_product_l1234_123442


namespace koala_fiber_consumption_l1234_123455

/-- The amount of fiber a koala absorbs as a percentage of what it eats -/
def koala_absorption_rate : ℝ := 0.30

/-- The amount of fiber absorbed by the koala in one day (in ounces) -/
def fiber_absorbed : ℝ := 12

/-- Theorem: If a koala absorbs 30% of the fiber it eats and it absorbed 12 ounces of fiber in one day,
    then the total amount of fiber the koala ate that day was 40 ounces. -/
theorem koala_fiber_consumption :
  fiber_absorbed = koala_absorption_rate * 40 := by
  sorry

end koala_fiber_consumption_l1234_123455


namespace chess_tournament_25_players_l1234_123470

/-- Calculate the number of games in a chess tournament -/
def chess_tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- Theorem: In a chess tournament with 25 players, where each player plays twice against every other player, the total number of games is 1200 -/
theorem chess_tournament_25_players :
  2 * chess_tournament_games 25 = 1200 := by
  sorry

end chess_tournament_25_players_l1234_123470


namespace sarah_tic_tac_toe_wins_l1234_123420

/-- Represents the outcome of Sarah's tic-tac-toe games -/
structure TicTacToeOutcome where
  wins : ℕ
  ties : ℕ
  losses : ℕ
  total_games : ℕ
  net_earnings : ℤ

/-- Calculates the net earnings based on game outcomes -/
def calculate_earnings (outcome : TicTacToeOutcome) : ℤ :=
  4 * outcome.wins + outcome.ties - 3 * outcome.losses

theorem sarah_tic_tac_toe_wins : 
  ∀ (outcome : TicTacToeOutcome),
    outcome.total_games = 200 →
    outcome.ties = 60 →
    outcome.net_earnings = -84 →
    calculate_earnings outcome = outcome.net_earnings →
    outcome.wins + outcome.ties + outcome.losses = outcome.total_games →
    outcome.wins = 39 := by
  sorry


end sarah_tic_tac_toe_wins_l1234_123420


namespace vertical_line_properties_l1234_123435

/-- A line passing through two points with the same x-coordinate but different y-coordinates has an undefined slope and its x-intercept is equal to the common x-coordinate. -/
theorem vertical_line_properties (x y₁ y₂ : ℝ) (h : y₁ ≠ y₂) :
  let C : ℝ × ℝ := (x, y₁)
  let D : ℝ × ℝ := (x, y₂)
  let line := {P : ℝ × ℝ | ∃ t : ℝ, P = (1 - t) • C + t • D}
  (∀ P Q : ℝ × ℝ, P ∈ line → Q ∈ line → P.1 ≠ Q.1 → (Q.2 - P.2) / (Q.1 - P.1) = (0 : ℝ)/0) ∧
  (∃ y : ℝ, (x, y) ∈ line) :=
by sorry

end vertical_line_properties_l1234_123435


namespace mudits_age_l1234_123452

/-- Mudit's present age satisfies the given condition -/
theorem mudits_age : ∃ (x : ℕ), (x + 16 = 3 * (x - 4)) ∧ (x = 14) := by
  sorry

end mudits_age_l1234_123452


namespace tan_sum_with_product_l1234_123446

theorem tan_sum_with_product (x y : Real) (h1 : x + y = π / 3) 
  (h2 : Real.sqrt 3 = Real.tan (π / 3)) : 
  Real.tan x + Real.tan y + Real.sqrt 3 * Real.tan x * Real.tan y = Real.sqrt 3 := by
  sorry

end tan_sum_with_product_l1234_123446


namespace bananas_shared_l1234_123483

theorem bananas_shared (initial : ℕ) (remaining : ℕ) (shared : ℕ) : 
  initial = 88 → remaining = 84 → shared = initial - remaining → shared = 4 := by
sorry

end bananas_shared_l1234_123483


namespace line_intersects_midpoint_of_segment_l1234_123481

/-- The value of c for which the line 2x + y = c intersects the midpoint of the line segment from (1, 4) to (7, 10) -/
theorem line_intersects_midpoint_of_segment (c : ℝ) : 
  (∃ (x y : ℝ), 2*x + y = c ∧ 
   x = (1 + 7) / 2 ∧ 
   y = (4 + 10) / 2) → 
  c = 15 := by
sorry


end line_intersects_midpoint_of_segment_l1234_123481


namespace min_c_value_l1234_123457

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1002 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 1002 ∧
    ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a'| + |x - b'| + |x - 1002| :=
by sorry

end min_c_value_l1234_123457


namespace vector_perpendicular_parallel_l1234_123460

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v = (t * w.1, t * w.2)

theorem vector_perpendicular_parallel :
  (∃ k : ℝ, perpendicular (k * a.1 + b.1, k * a.2 + b.2) (a.1 - 3 * b.1, a.2 - 3 * b.2) ∧ k = 19) ∧
  (∃ k : ℝ, parallel (k * a.1 + b.1, k * a.2 + b.2) (a.1 - 3 * b.1, a.2 - 3 * b.2) ∧ k = -1/3) :=
sorry

end vector_perpendicular_parallel_l1234_123460


namespace point_coordinates_l1234_123454

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_coordinates :
  ∀ (x y : ℝ),
  fourth_quadrant x y →
  |y| = 12 →
  |x| = 4 →
  (x, y) = (4, -12) :=
by sorry

end point_coordinates_l1234_123454


namespace arithmetic_sequence_difference_l1234_123431

/-- Arithmetic sequence a_i -/
def a (d i : ℕ) : ℕ := 1 + 2 * (i - 1) * d

/-- Arithmetic sequence b_i -/
def b (d i : ℕ) : ℕ := 1 + (i - 1) * d

/-- Sum of first k terms of a_i -/
def s (d k : ℕ) : ℕ := k + k * (k - 1) * d

/-- Sum of first k terms of b_i -/
def t (d k : ℕ) : ℕ := k + k * (k - 1) * (d / 2)

/-- A_n sequence -/
def A (d n : ℕ) : ℕ := s d (t d n)

/-- B_n sequence -/
def B (d n : ℕ) : ℕ := t d (s d n)

/-- Main theorem -/
theorem arithmetic_sequence_difference (d n : ℕ) :
  A d (n + 1) - A d n = (1 + n * d)^3 ∧
  B d (n + 1) - B d n = (n * d)^3 + (1 + n * d)^3 := by
  sorry

end arithmetic_sequence_difference_l1234_123431


namespace boris_neighbors_l1234_123408

/-- Represents the six people in the circle -/
inductive Person : Type
  | Arkady : Person
  | Boris : Person
  | Vera : Person
  | Galya : Person
  | Danya : Person
  | Egor : Person

/-- Represents the circular arrangement of people -/
def Circle := List Person

/-- Check if two people are standing next to each other in the circle -/
def are_adjacent (c : Circle) (p1 p2 : Person) : Prop :=
  ∃ i : Nat, (c.get? i = some p1 ∧ c.get? ((i + 1) % c.length) = some p2) ∨
             (c.get? i = some p2 ∧ c.get? ((i + 1) % c.length) = some p1)

/-- Check if two people are standing opposite each other in the circle -/
def are_opposite (c : Circle) (p1 p2 : Person) : Prop :=
  ∃ i : Nat, c.get? i = some p1 ∧ c.get? ((i + c.length / 2) % c.length) = some p2

theorem boris_neighbors (c : Circle) :
  c.length = 6 →
  are_adjacent c Person.Danya Person.Vera →
  are_adjacent c Person.Danya Person.Egor →
  are_opposite c Person.Galya Person.Egor →
  ¬ are_adjacent c Person.Arkady Person.Galya →
  (are_adjacent c Person.Boris Person.Arkady ∧ are_adjacent c Person.Boris Person.Galya) :=
by sorry

end boris_neighbors_l1234_123408


namespace james_argument_l1234_123436

theorem james_argument (initial_friends : ℕ) (new_friends : ℕ) (current_friends : ℕ) :
  initial_friends = 20 →
  new_friends = 1 →
  current_friends = 19 →
  initial_friends - (current_friends - new_friends) = 1 :=
by sorry

end james_argument_l1234_123436


namespace max_non_managers_proof_l1234_123449

/-- Represents the number of managers in department A -/
def managers : ℕ := 9

/-- Represents the ratio of managers to non-managers in department A -/
def manager_ratio : ℚ := 7 / 37

/-- Represents the ratio of specialists to generalists in department A -/
def specialist_ratio : ℚ := 2 / 1

/-- Calculates the maximum number of non-managers in department A -/
def max_non_managers : ℕ := 39

/-- Theorem stating that 39 is the maximum number of non-managers in department A -/
theorem max_non_managers_proof :
  ∀ n : ℕ, 
    (n : ℚ) / managers > manager_ratio ∧ 
    n % 3 = 0 ∧
    (2 * n / 3 : ℚ) / (n / 3 : ℚ) = specialist_ratio →
    n ≤ max_non_managers :=
by sorry

end max_non_managers_proof_l1234_123449


namespace log_less_than_one_range_l1234_123410

theorem log_less_than_one_range (a : ℝ) :
  (∃ (x : ℝ), Real.log x / Real.log a < 1) → a ∈ Set.union (Set.Ioo 0 1) (Set.Ioi 1) := by
  sorry

end log_less_than_one_range_l1234_123410


namespace quadratic_coefficient_sum_l1234_123421

theorem quadratic_coefficient_sum (p q a b : ℝ) : 
  (∀ x, -x^2 + p*x + q = 0 ↔ x = a ∨ x = b) →
  b < 1 →
  1 < a →
  p + q > 1 := by
sorry

end quadratic_coefficient_sum_l1234_123421


namespace product_of_roots_quadratic_l1234_123499

/-- Given a quadratic equation x^2 - 4x + 3 = 0 with roots x₁ and x₂, 
    the product of the roots x₁ * x₂ equals 3. -/
theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ + 3 = 0 → x₂^2 - 4*x₂ + 3 = 0 → x₁ * x₂ = 3 := by
  sorry


end product_of_roots_quadratic_l1234_123499


namespace largest_divisor_of_n_squared_divisible_by_18_l1234_123491

theorem largest_divisor_of_n_squared_divisible_by_18 (n : ℕ) (h1 : n > 0) (h2 : 18 ∣ n^2) :
  ∃ (d : ℕ), d = 6 ∧ d ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ d :=
by sorry

end largest_divisor_of_n_squared_divisible_by_18_l1234_123491


namespace ring_cost_l1234_123415

theorem ring_cost (total_cost : ℝ) (num_rings : ℕ) (h1 : total_cost = 24) (h2 : num_rings = 2) :
  total_cost / num_rings = 12 :=
by sorry

end ring_cost_l1234_123415


namespace at_least_two_equal_books_l1234_123492

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem at_least_two_equal_books (books : Fin 4 → ℕ) 
  (h : ∀ i, books i / sum_of_digits (books i) = 13) : 
  ∃ i j, i ≠ j ∧ books i = books j := by sorry

end at_least_two_equal_books_l1234_123492


namespace photographer_theorem_l1234_123414

/-- Represents the number of birds of each species -/
structure BirdCount where
  starlings : Nat
  wagtails : Nat
  woodpeckers : Nat

/-- The initial bird count -/
def initial_birds : BirdCount :=
  { starlings := 8, wagtails := 7, woodpeckers := 5 }

/-- The total number of birds -/
def total_birds : Nat := 20

/-- The number of photos to be taken -/
def photos_taken : Nat := 7

/-- Predicate to check if the remaining birds meet the condition -/
def meets_condition (b : BirdCount) : Prop :=
  (b.starlings ≥ 4 ∧ (b.wagtails ≥ 3 ∨ b.woodpeckers ≥ 3)) ∨
  (b.wagtails ≥ 4 ∧ (b.starlings ≥ 3 ∨ b.woodpeckers ≥ 3)) ∨
  (b.woodpeckers ≥ 4 ∧ (b.starlings ≥ 3 ∨ b.wagtails ≥ 3))

theorem photographer_theorem :
  ∀ (remaining : BirdCount),
    remaining.starlings + remaining.wagtails + remaining.woodpeckers = total_birds - photos_taken →
    remaining.starlings ≤ initial_birds.starlings →
    remaining.wagtails ≤ initial_birds.wagtails →
    remaining.woodpeckers ≤ initial_birds.woodpeckers →
    meets_condition remaining :=
by
  sorry

end photographer_theorem_l1234_123414


namespace graph_transformation_l1234_123465

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the reflection operation about x = 1
def reflect (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 - x)

-- Define the left shift operation
def shift_left (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + 1)

-- Theorem statement
theorem graph_transformation (f : ℝ → ℝ) :
  shift_left (reflect f) = λ x => f (1 - x) := by sorry

end graph_transformation_l1234_123465


namespace polynomial_coefficient_value_l1234_123439

/-- Given a polynomial equation, prove the value of a specific coefficient -/
theorem polynomial_coefficient_value 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^2 + x^10 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₉ = -10 := by
sorry

end polynomial_coefficient_value_l1234_123439


namespace pelicans_remaining_theorem_l1234_123485

/-- The number of Pelicans remaining in Shark Bite Cove after some moved to Pelican Bay -/
def pelicansRemaining (sharksInPelicanBay : ℕ) : ℕ :=
  let originalPelicans := sharksInPelicanBay / 2
  let pelicansMoved := originalPelicans / 3
  originalPelicans - pelicansMoved

/-- Theorem stating that given 60 sharks in Pelican Bay, 20 Pelicans remain in Shark Bite Cove -/
theorem pelicans_remaining_theorem :
  pelicansRemaining 60 = 20 := by
  sorry

#eval pelicansRemaining 60

end pelicans_remaining_theorem_l1234_123485


namespace parabola_midpoint_locus_ratio_l1234_123451

/-- A parabola with vertex and focus -/
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

/-- The locus of midpoints of right-angled chords of a parabola -/
def midpoint_locus (P : Parabola) : Parabola :=
  sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem parabola_midpoint_locus_ratio (P : Parabola) :
  let Q := midpoint_locus P
  (distance P.focus Q.focus) / (distance P.vertex Q.vertex) = 7/8 := by
  sorry

end parabola_midpoint_locus_ratio_l1234_123451


namespace gcd_equality_from_division_l1234_123430

theorem gcd_equality_from_division (a b q r : ℤ) :
  b > 0 →
  0 ≤ r →
  r < b →
  a = b * q + r →
  Int.gcd a b = Int.gcd b r := by
  sorry

end gcd_equality_from_division_l1234_123430


namespace smallest_k_divisible_by_500_l1234_123474

theorem smallest_k_divisible_by_500 : 
  ∀ k : ℕ+, k.val < 3000 → ¬(500 ∣ (k.val * (k.val + 1) * (2 * k.val + 1) / 6)) ∧ 
  (500 ∣ (3000 * 3001 * 6001 / 6)) := by
  sorry

end smallest_k_divisible_by_500_l1234_123474


namespace compound_interest_problem_l1234_123475

/-- Calculates the total amount after compound interest -/
def totalAmountAfterCompoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem compound_interest_problem (compoundInterest : ℝ) (rate : ℝ) (time : ℕ) 
  (h1 : compoundInterest = 2828.80)
  (h2 : rate = 0.08)
  (h3 : time = 2) :
  ∃ (principal : ℝ), 
    totalAmountAfterCompoundInterest principal rate time - principal = compoundInterest ∧
    totalAmountAfterCompoundInterest principal rate time = 19828.80 := by
  sorry

#eval totalAmountAfterCompoundInterest 17000 0.08 2

end compound_interest_problem_l1234_123475


namespace car_trip_duration_l1234_123403

/-- Represents the duration of a car trip with varying speeds. -/
def CarTrip (initial_speed initial_time additional_speed average_speed : ℝ) : Prop :=
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed

/-- The car trip lasts 12 hours given the specified conditions. -/
theorem car_trip_duration :
  CarTrip 45 4 75 65 → ∃ (total_time : ℝ), total_time = 12 :=
by sorry

end car_trip_duration_l1234_123403


namespace ainsley_win_probability_l1234_123409

/-- A fair six-sided die -/
inductive Die : Type
| one | two | three | four | five | six

/-- The probability of rolling a specific outcome on a fair six-sided die -/
def prob_roll (outcome : Die) : ℚ := 1 / 6

/-- Whether a roll is a multiple of 3 -/
def is_multiple_of_three (roll : Die) : Prop :=
  roll = Die.three ∨ roll = Die.six

/-- The probability of rolling a multiple of 3 -/
def prob_multiple_of_three : ℚ :=
  (prob_roll Die.three) + (prob_roll Die.six)

/-- The probability of rolling a non-multiple of 3 -/
def prob_non_multiple_of_three : ℚ :=
  1 - prob_multiple_of_three

/-- The probability of Ainsley winning the game -/
theorem ainsley_win_probability :
  prob_multiple_of_three * prob_multiple_of_three = 1 / 9 := by
  sorry


end ainsley_win_probability_l1234_123409


namespace quadratic_solution_l1234_123466

theorem quadratic_solution : ∃ x : ℝ, x^2 - x - 1 = 0 ∧ (x = (1 + Real.sqrt 5) / 2 ∨ x = -(1 + Real.sqrt 5) / 2) := by
  sorry

end quadratic_solution_l1234_123466


namespace real_return_calculation_l1234_123422

theorem real_return_calculation (nominal_rate inflation_rate : ℝ) 
  (h1 : nominal_rate = 0.21)
  (h2 : inflation_rate = 0.10) :
  (1 + nominal_rate) / (1 + inflation_rate) - 1 = 0.10 := by
  sorry

end real_return_calculation_l1234_123422


namespace tangent_line_equation_l1234_123437

/-- A line passing through point A(1,0) and tangent to the circle (x-3)^2 + (y-4)^2 = 4 -/
def TangentLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃ k : ℝ, 
    (∀ x y, (x, y) ∈ l ↔ y = k * (x - 1)) ∧
    (abs (2 * k - 4) / Real.sqrt (k^2 + 1) = 2)

theorem tangent_line_equation :
  ∀ l : Set (ℝ × ℝ), TangentLine l →
    (∀ x y, (x, y) ∈ l ↔ x = 1 ∨ 3 * x - 4 * y - 3 = 0) :=
by sorry

end tangent_line_equation_l1234_123437


namespace blue_candy_count_l1234_123462

theorem blue_candy_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 3409)
  (h2 : red = 145)
  (h3 : blue = total - red) : blue = 3264 := by
  sorry

end blue_candy_count_l1234_123462


namespace fraction_addition_l1234_123429

theorem fraction_addition (x : ℝ) (h : x ≠ 1) : 
  (1 : ℝ) / (x - 1) + (3 : ℝ) / (x - 1) = (4 : ℝ) / (x - 1) := by
  sorry

end fraction_addition_l1234_123429


namespace mean_median_difference_l1234_123459

/-- Represents the frequency distribution of missed school days -/
def frequency_distribution : List (Nat × Nat) := [
  (0, 2), (1, 5), (2, 1), (3, 3), (4, 2), (5, 4), (6, 1), (7, 2)
]

/-- Total number of students -/
def total_students : Nat := 20

/-- Calculates the median number of days missed -/
def median (dist : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- Calculates the mean number of days missed -/
def mean (dist : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem mean_median_difference :
  mean frequency_distribution total_students - median frequency_distribution total_students = 1/5 := by
  sorry

end mean_median_difference_l1234_123459


namespace factorization_of_2m_squared_minus_2_l1234_123404

theorem factorization_of_2m_squared_minus_2 (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end factorization_of_2m_squared_minus_2_l1234_123404


namespace fraction_expression_equality_l1234_123497

theorem fraction_expression_equality : (3/7 + 4/5) / (5/11 + 2/3) = 1419/1295 := by
  sorry

end fraction_expression_equality_l1234_123497


namespace sqrt_minus_one_mod_prime_l1234_123458

theorem sqrt_minus_one_mod_prime (p : Nat) (h_prime : Prime p) (h_gt_two : p > 2) :
  (∃ x : Nat, x^2 ≡ -1 [ZMOD p]) ↔ ∃ k : Nat, p = 4*k + 1 := by
  sorry

end sqrt_minus_one_mod_prime_l1234_123458


namespace sales_amount_is_194_l1234_123425

/-- Represents the total sales amount from pencils in a stationery store. -/
def total_sales (eraser_price regular_price short_price : ℚ) 
                (eraser_sold regular_sold short_sold : ℕ) : ℚ :=
  eraser_price * eraser_sold + regular_price * regular_sold + short_price * short_sold

/-- Theorem stating that the total sales amount is $194 given the specific conditions. -/
theorem sales_amount_is_194 :
  total_sales 0.8 0.5 0.4 200 40 35 = 194 := by
  sorry

#eval total_sales 0.8 0.5 0.4 200 40 35

end sales_amount_is_194_l1234_123425


namespace stratified_sampling_under_35_l1234_123413

/-- Calculates the number of people to be drawn from a stratum in stratified sampling -/
def stratifiedSampleSize (totalPopulation : ℕ) (stratumSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (stratumSize * sampleSize) / totalPopulation

/-- The problem statement -/
theorem stratified_sampling_under_35 :
  let totalPopulation : ℕ := 500
  let under35 : ℕ := 125
  let between35and49 : ℕ := 280
  let over50 : ℕ := 95
  let sampleSize : ℕ := 100
  stratifiedSampleSize totalPopulation under35 sampleSize = 25 := by
  sorry


end stratified_sampling_under_35_l1234_123413


namespace tan_alpha_minus_beta_equals_one_l1234_123478

theorem tan_alpha_minus_beta_equals_one (α β : ℝ) 
  (h : (3 / (2 + Real.sin (2 * α))) + (2021 / (2 + Real.sin β)) = 2024) : 
  Real.tan (α - β) = 1 := by
  sorry

end tan_alpha_minus_beta_equals_one_l1234_123478


namespace fraction_nonnegative_l1234_123417

theorem fraction_nonnegative (x : ℝ) (h : x ≠ -2) : x^2 / (x + 2)^2 ≥ 0 := by sorry

end fraction_nonnegative_l1234_123417


namespace brick_width_is_11_l1234_123450

-- Define the dimensions and quantities
def wall_length : ℝ := 200 -- in cm
def wall_width : ℝ := 300  -- in cm
def wall_height : ℝ := 2   -- in cm
def brick_length : ℝ := 25 -- in cm
def brick_height : ℝ := 6  -- in cm
def num_bricks : ℝ := 72.72727272727273

-- Define the theorem
theorem brick_width_is_11 :
  ∃ (brick_width : ℝ),
    brick_width = 11 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry

end brick_width_is_11_l1234_123450


namespace midpoint_sum_and_product_l1234_123433

/-- Given a line segment with endpoints (8, 15) and (-2, -3), 
    prove that the sum of the coordinates of the midpoint is 9 
    and the product of the coordinates of the midpoint is 18. -/
theorem midpoint_sum_and_product : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := 15
  let x₂ : ℝ := -2
  let y₂ : ℝ := -3
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  (midpoint_x + midpoint_y = 9) ∧ (midpoint_x * midpoint_y = 18) := by
  sorry

end midpoint_sum_and_product_l1234_123433


namespace circle_area_ratio_l1234_123494

theorem circle_area_ratio (s r : ℝ) (hs : s > 0) (hr : r > 0) (h : r = 0.4 * s) :
  (π * (r / 2)^2) / (π * (s / 2)^2) = 0.16 := by
  sorry

end circle_area_ratio_l1234_123494


namespace heart_nested_calculation_l1234_123480

def heart (a b : ℝ) : ℝ := (a + 2*b) * (a - b)

theorem heart_nested_calculation : heart 2 (heart 3 4) = -260 := by
  sorry

end heart_nested_calculation_l1234_123480


namespace campaign_donation_percentage_l1234_123495

theorem campaign_donation_percentage :
  let max_donation : ℕ := 1200
  let max_donors : ℕ := 500
  let half_donation : ℕ := max_donation / 2
  let half_donors : ℕ := 3 * max_donors
  let total_raised : ℕ := 3750000
  let donation_sum : ℕ := max_donation * max_donors + half_donation * half_donors
  (donation_sum : ℚ) / total_raised * 100 = 40 := by
  sorry

end campaign_donation_percentage_l1234_123495


namespace solution_set_l1234_123479

/-- An even function that is monotonically decreasing on [0,+∞) and f(1) = 0 -/
def f (x : ℝ) : ℝ := sorry

/-- f is an even function -/
axiom f_even : ∀ x, f x = f (-x)

/-- f is monotonically decreasing on [0,+∞) -/
axiom f_decreasing : ∀ x y, 0 ≤ x → x < y → f y < f x

/-- f(1) = 0 -/
axiom f_one_eq_zero : f 1 = 0

/-- The solution set of f(x-3) ≥ 0 is [2,4] -/
theorem solution_set : Set.Icc 2 4 = {x | f (x - 3) ≥ 0} := by sorry

end solution_set_l1234_123479


namespace min_cars_in_group_l1234_123424

/-- Represents the group of cars -/
structure CarGroup where
  total : ℕ
  withAC : ℕ
  withStripes : ℕ

/-- The conditions of the car group -/
def validCarGroup (g : CarGroup) : Prop :=
  g.total - g.withAC = 49 ∧
  g.withStripes ≥ 51 ∧
  g.withAC - g.withStripes ≤ 49

/-- The theorem stating that the minimum number of cars in a valid group is 100 -/
theorem min_cars_in_group (g : CarGroup) (h : validCarGroup g) : g.total ≥ 100 := by
  sorry

#check min_cars_in_group

end min_cars_in_group_l1234_123424


namespace day_one_fish_count_l1234_123471

/-- The number of fish counted on day one -/
def day_one_count : ℕ := sorry

/-- The percentage of fish that are sharks -/
def shark_percentage : ℚ := 1/4

/-- The total number of sharks counted over two days -/
def total_sharks : ℕ := 15

theorem day_one_fish_count : 
  day_one_count = 15 :=
by
  have h1 : shark_percentage * (day_one_count + 3 * day_one_count) = total_sharks := sorry
  sorry


end day_one_fish_count_l1234_123471


namespace angle_A_measure_l1234_123440

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem angle_A_measure (abc : Triangle) (h1 : abc.C = 3 * abc.B) (h2 : abc.B = 15) : 
  abc.A = 120 := by
sorry

end angle_A_measure_l1234_123440


namespace certain_number_proof_l1234_123490

theorem certain_number_proof (given_division : 7125 / 1.25 = 5700) 
  (certain_number : ℝ) (certain_division : certain_number / 12.5 = 57) : 
  certain_number = 712.5 := by
sorry

end certain_number_proof_l1234_123490
