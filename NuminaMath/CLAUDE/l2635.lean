import Mathlib

namespace final_amount_theorem_l2635_263534

def initial_amount : ℚ := 1499.9999999999998

def remaining_after_clothes (initial : ℚ) : ℚ := initial - (1/3 * initial)

def remaining_after_food (after_clothes : ℚ) : ℚ := after_clothes - (1/5 * after_clothes)

def remaining_after_travel (after_food : ℚ) : ℚ := after_food - (1/4 * after_food)

theorem final_amount_theorem :
  remaining_after_travel (remaining_after_food (remaining_after_clothes initial_amount)) = 600 := by
  sorry

end final_amount_theorem_l2635_263534


namespace tangent_circle_radius_l2635_263553

/-- Given a right triangle with legs of lengths 6 and 8, and semicircles constructed
    on all its sides as diameters lying outside the triangle, the radius of the circle
    tangent to these semicircles is 144/23. -/
theorem tangent_circle_radius (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_a : a = 6) (h_b : b = 8) : ∃ r : ℝ, r = 144 / 23 ∧ 
  r > 0 ∧
  (∃ x y z : ℝ, x^2 + y^2 = (r + a/2)^2 ∧
               y^2 + z^2 = (r + b/2)^2 ∧
               z^2 + x^2 = (r + c/2)^2) :=
by sorry

end tangent_circle_radius_l2635_263553


namespace minimum_red_chips_l2635_263513

/-- Represents the number of chips of each color in the box -/
structure ChipCount where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Checks if the chip count satisfies all given conditions -/
def satisfiesConditions (c : ChipCount) : Prop :=
  c.blue ≥ (3 * c.white) / 4 ∧
  c.blue ≤ c.red / 4 ∧
  60 ≤ c.white + c.blue ∧
  c.white + c.blue ≤ 80

/-- The minimum number of red chips that satisfies all conditions -/
def minRedChips : ℕ := 108

theorem minimum_red_chips :
  ∀ c : ChipCount, satisfiesConditions c → c.red ≥ minRedChips :=
sorry


end minimum_red_chips_l2635_263513


namespace roots_in_intervals_l2635_263538

/-- The quadratic function f(x) = 7x^2 - (k+13)x + k^2 - k - 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := 7 * x^2 - (k + 13) * x + k^2 - k - 2

/-- Theorem stating the range of k for which f(x) has roots in (0,1) and (1,2) -/
theorem roots_in_intervals (k : ℝ) : 
  (∃ x y, 0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2 ∧ f k x = 0 ∧ f k y = 0) ↔ 
  ((3 < k ∧ k < 4) ∨ (-2 < k ∧ k < -1)) :=
sorry

end roots_in_intervals_l2635_263538


namespace regular_polygon_sides_l2635_263500

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 12 → n = 30 := by
  sorry

end regular_polygon_sides_l2635_263500


namespace remainder_double_mod_seven_l2635_263512

theorem remainder_double_mod_seven (n : ℤ) (h : n % 7 = 2) : (2 * n) % 7 = 4 := by
  sorry

end remainder_double_mod_seven_l2635_263512


namespace smallest_integer_x_l2635_263572

theorem smallest_integer_x : ∃ x : ℤ, (∀ y : ℤ, 3 * |y|^3 + 5 < 56 → x ≤ y) ∧ (3 * |x|^3 + 5 < 56) := by
  sorry

end smallest_integer_x_l2635_263572


namespace kim_earrings_proof_l2635_263570

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_earring : ℕ := 9

/-- The number of pairs of earrings Kim brings on the first day -/
def first_day_earrings : ℕ := 3

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The number of days the gumballs last -/
def days_gumballs_last : ℕ := 42

theorem kim_earrings_proof :
  (first_day_earrings * gumballs_per_earring + 
   2 * first_day_earrings * gumballs_per_earring + 
   (2 * first_day_earrings - 1) * gumballs_per_earring) = 
  (gumballs_eaten_per_day * days_gumballs_last) := by
  sorry

end kim_earrings_proof_l2635_263570


namespace inequality_solution_set_l2635_263581

theorem inequality_solution_set :
  {x : ℝ | x * (x - 1) ≥ x} = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by
  sorry

end inequality_solution_set_l2635_263581


namespace problem_solution_l2635_263533

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := (1/3) * x^3 - x
def g (x : ℝ) := 33 * f x + 3 * x

-- Define the sequence bₙ
def b (n : ℕ) : ℝ := g n ^ (1 / g (n + 1))

-- Theorem statement
theorem problem_solution :
  -- f(x) reaches its maximum value 2/3 when x = -1
  (f (-1) = 2/3 ∧ ∀ x, f x ≤ 2/3) ∧
  -- The graph of y = f(x+1) is symmetrical about the point (-1, 0)
  (∀ x, f (x + 1) = -f (-x - 1)) →
  -- 1. f(x) = (1/3)x³ - x is implied by the above conditions
  (∀ x, f x = (1/3) * x^3 - x) ∧
  -- 2. When x > 0, [1 + 1/g(x)]^g(x) < e
  (∀ x > 0, (1 + 1 / g x) ^ (g x) < Real.exp 1) ∧
  -- 3. The sequence bₙ has only one equal pair: b₂ = b₈
  (∀ n m : ℕ, n ≠ m → b n = b m ↔ (n = 2 ∧ m = 8) ∨ (n = 8 ∧ m = 2)) :=
sorry

end

end problem_solution_l2635_263533


namespace no_common_solution_l2635_263556

theorem no_common_solution : ¬∃ x : ℝ, (5*x - 2) / (6*x - 6) = 3/4 ∧ x^2 - 1 = 0 := by
  sorry

end no_common_solution_l2635_263556


namespace distinct_combinations_l2635_263507

def num_shirts : ℕ := 8
def num_ties : ℕ := 7
def num_jackets : ℕ := 3

theorem distinct_combinations : num_shirts * num_ties * num_jackets = 168 := by
  sorry

end distinct_combinations_l2635_263507


namespace garden_ant_count_l2635_263565

/-- Represents the dimensions and ant density of a rectangular garden --/
structure Garden where
  width : ℝ  -- width in feet
  length : ℝ  -- length in feet
  antDensity : ℝ  -- ants per square inch

/-- Conversion factor from feet to inches --/
def feetToInches : ℝ := 12

/-- Calculates the approximate number of ants in the garden --/
def approximateAntCount (g : Garden) : ℝ :=
  g.width * feetToInches * g.length * feetToInches * g.antDensity

/-- Theorem stating that the number of ants in the given garden is approximately 30 million --/
theorem garden_ant_count :
  let g : Garden := { width := 350, length := 300, antDensity := 2 }
  ∃ ε > 0, |approximateAntCount g - 30000000| < ε := by
  sorry

end garden_ant_count_l2635_263565


namespace completing_square_result_l2635_263580

theorem completing_square_result (x : ℝ) : 
  (x^2 - 4*x + 2 = 0) → ((x - 2)^2 = 2) :=
by
  sorry

end completing_square_result_l2635_263580


namespace travelers_meet_on_day_three_l2635_263543

/-- Distance traveled by the first traveler on day n -/
def d1 (n : ℕ) : ℕ := 3 * n - 1

/-- Distance traveled by the second traveler on day n -/
def d2 (n : ℕ) : ℕ := 2 * n + 1

/-- Total distance traveled by the first traveler after n days -/
def D1 (n : ℕ) : ℕ := (3 * n^2 + n) / 2

/-- Total distance traveled by the second traveler after n days -/
def D2 (n : ℕ) : ℕ := n^2 + 2 * n

theorem travelers_meet_on_day_three :
  ∃ n : ℕ, n > 0 ∧ D1 n = D2 n ∧ ∀ m : ℕ, 0 < m ∧ m < n → D1 m < D2 m :=
sorry

end travelers_meet_on_day_three_l2635_263543


namespace power_of_two_representation_l2635_263575

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), Odd x ∧ Odd y ∧ 2^n = 7*x^2 + y^2 :=
by sorry

end power_of_two_representation_l2635_263575


namespace tangent_perpendicular_point_l2635_263566

theorem tangent_perpendicular_point (x y : ℝ) : 
  y = 1 / x →  -- P is on the curve y = 1/x
  ((-1 / x^2) * (1 / 4) = -1) →  -- Tangent line is perpendicular to x - 4y - 8 = 0
  ((x = -1/2 ∧ y = -2) ∨ (x = 1/2 ∧ y = 2)) := by
  sorry

end tangent_perpendicular_point_l2635_263566


namespace quadratic_function_m_l2635_263590

/-- A quadratic function g(x) with integer coefficients -/
def g (d e f : ℤ) (x : ℤ) : ℤ := d * x^2 + e * x + f

/-- The theorem stating that under given conditions, m = -1 -/
theorem quadratic_function_m (d e f m : ℤ) : 
  g d e f 2 = 0 ∧ 
  60 < g d e f 6 ∧ g d e f 6 < 70 ∧
  80 < g d e f 9 ∧ g d e f 9 < 90 ∧
  10000 * m < g d e f 100 ∧ g d e f 100 < 10000 * (m + 1) →
  m = -1 := by
  sorry

end quadratic_function_m_l2635_263590


namespace smallest_n_for_doughnuts_l2635_263508

theorem smallest_n_for_doughnuts : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → (13 * m - 1) % 9 = 0 → m ≥ n) ∧
  (13 * n - 1) % 9 = 0 :=
by sorry

end smallest_n_for_doughnuts_l2635_263508


namespace rain_probability_l2635_263599

/-- The probability of rain on Friday -/
def prob_friday : ℝ := 0.4

/-- The probability of rain on Saturday -/
def prob_saturday : ℝ := 0.5

/-- The probability of rain on Sunday, given it didn't rain on both Friday and Saturday -/
def prob_sunday_normal : ℝ := 0.3

/-- The probability of rain on Sunday, given it rained on both Friday and Saturday -/
def prob_sunday_conditional : ℝ := 0.6

/-- The probability of rain on all three days -/
def prob_all_days : ℝ := prob_friday * prob_saturday * prob_sunday_conditional

theorem rain_probability :
  prob_all_days = 0.12 :=
sorry

end rain_probability_l2635_263599


namespace election_vote_count_l2635_263569

/-- Represents the number of votes in an election round -/
structure ElectionRound where
  totalVotes : ℕ
  firstCandidateVotes : ℕ
  secondCandidateVotes : ℕ

/-- Represents a two-round election -/
structure TwoRoundElection where
  firstRound : ElectionRound
  secondRound : ElectionRound

theorem election_vote_count (election : TwoRoundElection) : election.firstRound.totalVotes = 48000 :=
  by
  have h1 : election.firstRound.firstCandidateVotes = election.firstRound.secondCandidateVotes :=
    sorry
  have h2 : election.secondRound.totalVotes = election.firstRound.totalVotes := sorry
  have h3 : election.secondRound.firstCandidateVotes =
    election.firstRound.firstCandidateVotes - 16000 := sorry
  have h4 : election.secondRound.secondCandidateVotes =
    election.firstRound.secondCandidateVotes + 16000 := sorry
  have h5 : election.secondRound.secondCandidateVotes =
    5 * election.secondRound.firstCandidateVotes := sorry
  sorry

end election_vote_count_l2635_263569


namespace max_value_of_f_inequality_with_sum_constraint_l2635_263591

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

-- Theorem for the maximum value of f
theorem max_value_of_f : ∃ (s : ℝ), s = 3 ∧ ∀ (x : ℝ), f x ≤ s := by sorry

-- Theorem for the inequality
theorem inequality_with_sum_constraint (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) : 
  a^2 + b^2 + c^2 ≥ 3 := by sorry

end max_value_of_f_inequality_with_sum_constraint_l2635_263591


namespace brick_length_calculation_l2635_263515

theorem brick_length_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_width : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 15 →
  brick_width = 0.1 →
  total_bricks = 18750 →
  (courtyard_length * courtyard_width * 10000) / (total_bricks * brick_width) = 20 := by
  sorry

end brick_length_calculation_l2635_263515


namespace crayons_difference_is_seven_l2635_263548

/-- The number of crayons Nori gave to Lea more than Mae -/
def crayons_difference : ℕ :=
  let initial_boxes : ℕ := 4
  let crayons_per_box : ℕ := 8
  let crayons_to_mae : ℕ := 5
  let crayons_left : ℕ := 15
  let initial_crayons : ℕ := initial_boxes * crayons_per_box
  let crayons_after_mae : ℕ := initial_crayons - crayons_to_mae
  let crayons_to_lea : ℕ := crayons_after_mae - crayons_left
  crayons_to_lea - crayons_to_mae

theorem crayons_difference_is_seven :
  crayons_difference = 7 := by
  sorry

end crayons_difference_is_seven_l2635_263548


namespace expression_evaluation_l2635_263588

theorem expression_evaluation (a b : ℝ) 
  (h : |a - 2| + (b - 1/2)^2 = 0) : 
  2*(a^2*b - 3*a*b^2) - (5*a^2*b - 3*(2*a*b^2 - a^2*b) - 2) = -10 := by
  sorry

end expression_evaluation_l2635_263588


namespace profit_percentage_calculation_l2635_263563

/-- Calculate the profit percentage given the selling price and profit -/
theorem profit_percentage_calculation (selling_price profit : ℝ) :
  selling_price = 900 →
  profit = 225 →
  (profit / (selling_price - profit)) * 100 = 33.33333333333333 :=
by sorry

end profit_percentage_calculation_l2635_263563


namespace f_difference_at_3_and_neg_3_l2635_263554

def f (x : ℝ) : ℝ := x^4 + x^2 + 7*x

theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end f_difference_at_3_and_neg_3_l2635_263554


namespace will_summer_earnings_l2635_263577

/-- The amount of money Will spent on mower blades -/
def mower_blades_cost : ℕ := 41

/-- The number of games Will could buy with the remaining money -/
def number_of_games : ℕ := 7

/-- The cost of each game -/
def game_cost : ℕ := 9

/-- The total money Will made mowing lawns -/
def total_money : ℕ := mower_blades_cost + number_of_games * game_cost

theorem will_summer_earnings : total_money = 104 := by
  sorry

end will_summer_earnings_l2635_263577


namespace max_pieces_is_sixteen_l2635_263502

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 16

/-- The size of a small cake piece in inches -/
def small_piece_size : ℕ := 4

/-- The area of the large cake in square inches -/
def large_cake_area : ℕ := large_cake_size * large_cake_size

/-- The area of a small cake piece in square inches -/
def small_piece_area : ℕ := small_piece_size * small_piece_size

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := large_cake_area / small_piece_area

theorem max_pieces_is_sixteen : max_pieces = 16 := by
  sorry

end max_pieces_is_sixteen_l2635_263502


namespace at_least_one_not_less_than_two_l2635_263537

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  max ((b + c) / a) (max ((a + c) / b) ((a + b) / c)) ≥ 2 := by
  sorry

end at_least_one_not_less_than_two_l2635_263537


namespace quadratic_inequality_solution_set_l2635_263562

theorem quadratic_inequality_solution_set 
  (a b c α β : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c > 0 ↔ α < x ∧ x < β) 
  (h2 : α > 0) :
  ∀ x, c*x^2 + b*x + a < 0 ↔ x < 1/β ∨ x > 1/α :=
by sorry

end quadratic_inequality_solution_set_l2635_263562


namespace count_valid_numbers_l2635_263555

/-- The set of digits that can be used to form the numbers. -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A function that checks if a number is even. -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A function that checks if a three-digit number is less than 600. -/
def lessThan600 (n : Nat) : Bool := n < 600 ∧ n ≥ 100

/-- The set of valid hundreds digits (1 to 5). -/
def validHundreds : Finset Nat := {1, 2, 3, 4, 5}

/-- The set of valid units digits (0, 2, 4). -/
def validUnits : Finset Nat := {0, 2, 4}

/-- The main theorem stating the number of valid three-digit numbers. -/
theorem count_valid_numbers : 
  (validHundreds.card * digits.card * validUnits.card : Nat) = 90 := by sorry

end count_valid_numbers_l2635_263555


namespace die_roll_frequency_l2635_263546

/-- The frequency of a specific outcome in a series of trials -/
def frequency (successful_outcomes : ℕ) (total_trials : ℕ) : ℚ :=
  successful_outcomes / total_trials

/-- The number of times the die was rolled -/
def total_rolls : ℕ := 60

/-- The number of times six appeared -/
def six_appearances : ℕ := 10

theorem die_roll_frequency :
  frequency six_appearances total_rolls = 1 / 6 := by
  sorry

end die_roll_frequency_l2635_263546


namespace min_set_size_l2635_263536

theorem min_set_size (n : ℕ) : 
  let set_size := 2 * n + 1
  let median := 10
  let arithmetic_mean := 6
  let sum := arithmetic_mean * set_size
  let lower_bound := n * 1 + (n + 1) * 10
  sum ≥ lower_bound → n ≥ 4 :=
by
  sorry

end min_set_size_l2635_263536


namespace apple_price_36kg_l2635_263511

/-- The price of apples for a given weight --/
def apple_price (l q : ℚ) (weight : ℚ) : ℚ :=
  if weight ≤ 30 then l * weight
  else l * 30 + q * (weight - 30)

theorem apple_price_36kg (l q : ℚ) : 
  (apple_price l q 20 = 100) → 
  (apple_price l q 33 = 168) → 
  (apple_price l q 36 = 186) := by
  sorry

#check apple_price_36kg

end apple_price_36kg_l2635_263511


namespace sales_solution_l2635_263567

def sales_problem (s1 s2 s3 s4 s6 : ℕ) (average : ℕ) : Prop :=
  let total := average * 6
  let known_sum := s1 + s2 + s3 + s4 + s6
  let s5 := total - known_sum
  s5 = 6562

theorem sales_solution (s1 s2 s3 s4 s6 average : ℕ) 
  (h1 : s1 = 6435) (h2 : s2 = 6927) (h3 : s3 = 6855) (h4 : s4 = 7230) 
  (h5 : s6 = 6791) (h6 : average = 6800) :
  sales_problem s1 s2 s3 s4 s6 average :=
by
  sorry

end sales_solution_l2635_263567


namespace heartsuit_calculation_l2635_263529

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heartsuit_calculation :
  heartsuit 3 (heartsuit 4 5) = -72 := by
  sorry

end heartsuit_calculation_l2635_263529


namespace single_elimination_tournament_games_l2635_263522

/-- The number of games played in a single-elimination tournament. -/
def games_played (num_teams : ℕ) : ℕ :=
  num_teams - 1

/-- Theorem: In a single-elimination tournament with 32 teams, 31 games are played. -/
theorem single_elimination_tournament_games :
  games_played 32 = 31 := by
  sorry

#eval games_played 32  -- Should output 31

end single_elimination_tournament_games_l2635_263522


namespace min_value_quadratic_l2635_263541

theorem min_value_quadratic (x y : ℝ) :
  y = x^2 + 12*x + 5 →
  ∀ z : ℝ, y ≥ -31 ∧ (∃ w : ℝ, w^2 + 12*w + 5 = -31) :=
by sorry

end min_value_quadratic_l2635_263541


namespace spelling_contest_questions_l2635_263558

theorem spelling_contest_questions (drew_correct drew_wrong carla_correct : ℕ) 
  (h1 : drew_correct = 20)
  (h2 : drew_wrong = 6)
  (h3 : carla_correct = 14)
  (h4 : carla_correct + 2 * drew_wrong = drew_correct + drew_wrong) :
  drew_correct + drew_wrong = 26 :=
by sorry

end spelling_contest_questions_l2635_263558


namespace simple_interest_problem_l2635_263504

/-- Proves that given a simple interest of 100, an interest rate of 5% per annum,
    and a time period of 4 years, the principal sum is 500. -/
theorem simple_interest_problem (interest : ℕ) (rate : ℕ) (time : ℕ) (principal : ℕ) : 
  interest = 100 → rate = 5 → time = 4 → 
  interest = principal * rate * time / 100 →
  principal = 500 := by
sorry

end simple_interest_problem_l2635_263504


namespace bus_system_stops_l2635_263576

/-- Represents a bus system in a city -/
structure BusSystem where
  num_routes : ℕ
  stops_per_route : ℕ
  travel_without_transfer : Prop
  unique_intersection : Prop
  min_stops : Prop

/-- Theorem: In a bus system with 57 routes, where you can travel between any two stops without transferring,
    each pair of routes intersects at exactly one stop, and each route has at least three stops,
    the number of stops on each route is 40. -/
theorem bus_system_stops (bs : BusSystem) 
  (h1 : bs.num_routes = 57)
  (h2 : bs.travel_without_transfer)
  (h3 : bs.unique_intersection)
  (h4 : bs.min_stops)
  : bs.stops_per_route = 40 := by
  sorry

end bus_system_stops_l2635_263576


namespace max_cake_pieces_l2635_263594

/-- Represents the dimensions of a rectangular cake -/
structure CakeDimensions where
  m : ℕ
  n : ℕ

/-- Checks if the given dimensions satisfy the required condition -/
def satisfiesCondition (d : CakeDimensions) : Prop :=
  (d.m - 2) * (d.n - 2) = (d.m * d.n) / 2

/-- Calculates the total number of cake pieces -/
def totalPieces (d : CakeDimensions) : ℕ :=
  d.m * d.n

/-- Theorem stating the maximum number of cake pieces possible -/
theorem max_cake_pieces :
  ∃ (d : CakeDimensions), satisfiesCondition d ∧ 
    (∀ (d' : CakeDimensions), satisfiesCondition d' → totalPieces d' ≤ totalPieces d) ∧
    totalPieces d = 60 := by
  sorry

end max_cake_pieces_l2635_263594


namespace order_of_abc_l2635_263552

theorem order_of_abc (a b c : ℝ) 
  (ha : a = (1.1 : ℝ)^10)
  (hb : (5 : ℝ)^b = 3^a + 4^a)
  (hc : c = Real.exp a - a) : 
  b < a ∧ a < c := by sorry

end order_of_abc_l2635_263552


namespace weight_lifting_problem_l2635_263549

theorem weight_lifting_problem (total_weight first_lift second_lift : ℕ) : 
  total_weight = 1500 →
  2 * first_lift = second_lift + 300 →
  first_lift + second_lift = total_weight →
  first_lift = 600 := by
sorry

end weight_lifting_problem_l2635_263549


namespace new_shipment_bears_l2635_263530

def initial_stock : ℕ := 4
def bears_per_shelf : ℕ := 7
def shelves_used : ℕ := 2

theorem new_shipment_bears :
  initial_stock + (bears_per_shelf * shelves_used) - initial_stock = 10 := by
  sorry

end new_shipment_bears_l2635_263530


namespace expression_value_l2635_263535

theorem expression_value (m n : ℝ) (h : m * n = m + 3) : 3 * m - 3 * (m * n) + 10 = 1 := by
  sorry

end expression_value_l2635_263535


namespace distinct_polygons_count_l2635_263514

/-- The number of distinct convex polygons with 4 or more sides that can be drawn
    using some or all of 15 points marked on a circle as vertices -/
def num_polygons : ℕ := 32192

/-- The total number of points marked on the circle -/
def num_points : ℕ := 15

/-- A function that calculates the number of subsets of size k from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of subsets of 15 points -/
def total_subsets : ℕ := 2^num_points

theorem distinct_polygons_count :
  num_polygons = total_subsets - (choose num_points 0 + choose num_points 1 + 
                                  choose num_points 2 + choose num_points 3) :=
sorry

end distinct_polygons_count_l2635_263514


namespace multiplication_subtraction_difference_l2635_263560

theorem multiplication_subtraction_difference : ∃ (x : ℤ), x = 22 ∧ 3 * x - (62 - x) = 26 := by
  sorry

end multiplication_subtraction_difference_l2635_263560


namespace increasing_interval_of_f_l2635_263574

-- Define the function
def f (x : ℝ) : ℝ := 3*x - x^3

-- State the theorem
theorem increasing_interval_of_f :
  ∀ x : ℝ, (∀ y : ℝ, x < y → f x < f y) ↔ x ∈ Set.Ioo (-1 : ℝ) 1 := by sorry

end increasing_interval_of_f_l2635_263574


namespace intersection_A_B_l2635_263589

def A : Set ℤ := {-1, 0, 1, 2, 3}
def B : Set ℤ := {x | 3 - x < 1}

theorem intersection_A_B : A ∩ B = {3} := by
  sorry

end intersection_A_B_l2635_263589


namespace fold_theorem_l2635_263595

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line on a 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Defines a fold on graph paper -/
def Fold (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (foldLine : Line),
    -- The fold line is perpendicular to the line connecting p1 and p2
    foldLine.slope * ((p2.x - p1.x) / (p2.y - p1.y)) = -1 ∧
    -- The midpoint of p1 and p2 is on the fold line
    (p1.y + p2.y) / 2 = foldLine.slope * ((p1.x + p2.x) / 2) + foldLine.yIntercept ∧
    -- The midpoint of p3 and p4 is on the fold line
    (p3.y + p4.y) / 2 = foldLine.slope * ((p3.x + p4.x) / 2) + foldLine.yIntercept ∧
    -- The line connecting p3 and p4 is perpendicular to the fold line
    foldLine.slope * ((p4.x - p3.x) / (p4.y - p3.y)) = -1

/-- The main theorem to prove -/
theorem fold_theorem (m n : ℝ) :
  Fold ⟨0, 3⟩ ⟨5, 0⟩ ⟨8, 5⟩ ⟨m, n⟩ → m + n = 10.3 := by
  sorry

end fold_theorem_l2635_263595


namespace decreasing_function_inequality_l2635_263578

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : ∀ x y : ℝ, x < y → f x > f y) 
  (h_inequality : f a ≥ f (-2)) : 
  a ≤ -2 := by
  sorry

end decreasing_function_inequality_l2635_263578


namespace binomial_7_2_l2635_263596

theorem binomial_7_2 : Nat.choose 7 2 = 21 := by
  sorry

end binomial_7_2_l2635_263596


namespace sum_of_reciprocals_of_roots_l2635_263539

theorem sum_of_reciprocals_of_roots (p q : ℝ) : 
  p^2 - 16*p + 9 = 0 → 
  q^2 - 16*q + 9 = 0 → 
  p ≠ q → 
  1/p + 1/q = 16/9 := by
sorry

end sum_of_reciprocals_of_roots_l2635_263539


namespace latest_time_82_degrees_l2635_263568

-- Define the temperature function
def T (t : ℝ) : ℝ := -t^2 + 12*t + 55

-- Define the derivative of the temperature function
def T' (t : ℝ) : ℝ := -2*t + 12

-- Theorem statement
theorem latest_time_82_degrees (t : ℝ) :
  (T t = 82) ∧ (T' t < 0) →
  t = 6 + (3 * Real.sqrt 28) / 2 :=
by sorry

end latest_time_82_degrees_l2635_263568


namespace taqeeshas_grade_l2635_263505

theorem taqeeshas_grade (total_students : Nat) (initial_students : Nat) (initial_average : Nat) (new_average : Nat) :
  total_students = 17 →
  initial_students = 16 →
  initial_average = 77 →
  new_average = 78 →
  (initial_students * initial_average + (total_students - initial_students) * 94) / total_students = new_average :=
by sorry

end taqeeshas_grade_l2635_263505


namespace B_power_150_is_identity_l2635_263551

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end B_power_150_is_identity_l2635_263551


namespace hcf_of_156_324_672_l2635_263509

theorem hcf_of_156_324_672 : Nat.gcd 156 (Nat.gcd 324 672) = 12 := by
  sorry

end hcf_of_156_324_672_l2635_263509


namespace greatest_prime_factor_of_factorial_sum_l2635_263593

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (Nat.factorial 15 + Nat.factorial 17) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (Nat.factorial 15 + Nat.factorial 17) → q ≤ p :=
sorry

end greatest_prime_factor_of_factorial_sum_l2635_263593


namespace systems_equivalence_l2635_263526

-- Define the systems of equations
def system1 (x y a b : ℝ) : Prop :=
  2 * (x + 1) - y = 7 ∧ x + b * y = a

def system2 (x y a b : ℝ) : Prop :=
  a * x + y = b ∧ 3 * x + 2 * (y - 1) = 9

-- Theorem statement
theorem systems_equivalence :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), system1 x y a b ∧ system2 x y a b) →
  (∃! (x y : ℝ), x = 3 ∧ y = 1 ∧ system1 x y a b ∧ system2 x y a b) ∧
  (3 * a - b)^2023 = -1 :=
sorry

end systems_equivalence_l2635_263526


namespace largest_value_l2635_263510

theorem largest_value (x y z w : ℝ) (h : x + 3 = y - 4 ∧ x + 3 = z + 2 ∧ x + 3 = w - 1) :
  y = max x (max y (max z w)) :=
by sorry

end largest_value_l2635_263510


namespace addition_equality_l2635_263531

theorem addition_equality : 12 + 36 = 48 := by
  sorry

end addition_equality_l2635_263531


namespace profit_difference_is_640_l2635_263540

/-- Calculates the difference between profit shares of two partners given their investments and the profit share of a third partner. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℕ) : ℕ :=
  let total_parts := invest_a + invest_b + invest_c
  let part_value := b_profit * total_parts / invest_b
  let a_profit := part_value * invest_a / total_parts
  let c_profit := part_value * invest_c / total_parts
  c_profit - a_profit

/-- Theorem stating that given the investments and b's profit share, the difference between a's and c's profit shares is 640. -/
theorem profit_difference_is_640 :
  profit_share_difference 8000 10000 12000 1600 = 640 := by
  sorry

end profit_difference_is_640_l2635_263540


namespace object_height_properties_l2635_263598

-- Define the height function
def h (t : ℝ) : ℝ := -14 * (t - 3)^2 + 140

-- Theorem statement
theorem object_height_properties :
  (∀ t : ℝ, h t ≤ h 3) ∧ (h 5 = 84) := by
  sorry

end object_height_properties_l2635_263598


namespace closest_integer_to_6_sqrt_35_l2635_263547

theorem closest_integer_to_6_sqrt_35 : 
  ∃ n : ℤ, ∀ m : ℤ, |6 * Real.sqrt 35 - n| ≤ |6 * Real.sqrt 35 - m| ∧ n = 36 :=
sorry

end closest_integer_to_6_sqrt_35_l2635_263547


namespace f_monotone_increasing_l2635_263527

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem stating that f is monotonically increasing on ℝ
theorem f_monotone_increasing : Monotone f := by sorry

end f_monotone_increasing_l2635_263527


namespace cost_difference_formula_option1_more_cost_effective_at_50_l2635_263544

/-- Represents the cost difference between Option 2 and Option 1 for a customer
    buying 20 water dispensers and x water dispenser barrels, where x > 20. -/
def cost_difference (x : ℝ) : ℝ :=
  (45 * x + 6300) - (50 * x + 6000)

/-- Theorem stating that the cost difference between Option 2 and Option 1
    is always 300 - 5x yuan, for x > 20. -/
theorem cost_difference_formula (x : ℝ) (h : x > 20) :
  cost_difference x = 300 - 5 * x := by
  sorry

/-- Corollary stating that Option 1 is more cost-effective when x = 50. -/
theorem option1_more_cost_effective_at_50 :
  cost_difference 50 > 0 := by
  sorry

end cost_difference_formula_option1_more_cost_effective_at_50_l2635_263544


namespace faster_by_plane_l2635_263571

-- Define the driving time in minutes
def driving_time : ℕ := 3 * 60 + 15

-- Define the components of the airplane trip
def airport_drive_time : ℕ := 10
def boarding_wait_time : ℕ := 20
def offboarding_time : ℕ := 10

-- Define the flight time as one-third of the driving time
def flight_time : ℕ := driving_time / 3

-- Define the total airplane trip time
def airplane_trip_time : ℕ := airport_drive_time + boarding_wait_time + flight_time + offboarding_time

-- Theorem to prove
theorem faster_by_plane : driving_time - airplane_trip_time = 90 := by
  sorry

end faster_by_plane_l2635_263571


namespace height_conversion_l2635_263542

/-- Converts a height from inches to centimeters given the conversion factors. -/
def height_in_cm (height_in : ℚ) (in_per_ft : ℚ) (cm_per_ft : ℚ) : ℚ :=
  height_in * (cm_per_ft / in_per_ft)

/-- Theorem stating that 65 inches is equivalent to 162.5 cm given the conversion factors. -/
theorem height_conversion :
  height_in_cm 65 10 25 = 162.5 := by sorry

end height_conversion_l2635_263542


namespace repeating_decimal_to_fraction_l2635_263520

/-- Represents a repeating decimal with a two-digit repeating part -/
def repeating_decimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

theorem repeating_decimal_to_fraction :
  repeating_decimal 2 7 = 3 / 11 ∧
  3 + 11 = 14 :=
sorry

end repeating_decimal_to_fraction_l2635_263520


namespace calculate_savings_l2635_263583

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
theorem calculate_savings (income : ℕ) (income_ratio expenditure_ratio : ℕ) : 
  income_ratio > 0 ∧ expenditure_ratio > 0 ∧ income = 21000 ∧ income_ratio = 3 ∧ expenditure_ratio = 2 →
  income - (income * expenditure_ratio / income_ratio) = 7000 := by
sorry

end calculate_savings_l2635_263583


namespace second_number_calculation_second_number_is_190_l2635_263557

theorem second_number_calculation : ℝ → Prop :=
  fun x =>
    let first_number : ℝ := 1280
    let twenty_percent_of_650 : ℝ := 0.2 * 650
    let twenty_five_percent_of_first : ℝ := 0.25 * first_number
    x = twenty_five_percent_of_first - twenty_percent_of_650 → x = 190

-- The proof is omitted
theorem second_number_is_190 : ∃ x : ℝ, second_number_calculation x :=
  sorry

end second_number_calculation_second_number_is_190_l2635_263557


namespace equation_solution_l2635_263564

theorem equation_solution : ∃ x : ℝ, 
  (x + 2 ≠ 0) ∧ 
  (x - 2 ≠ 0) ∧ 
  (2 * x / (x + 2) + x / (x - 2) = 3) ∧ 
  (x = 6) := by
  sorry

end equation_solution_l2635_263564


namespace odd_function_symmetry_symmetric_about_one_period_four_l2635_263545

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Statement 1
theorem odd_function_symmetry (h : ∀ x, f x = -f (-x)) :
  ∀ x, f (x - 1) = -f (-x + 1) :=
sorry

-- Statement 2
theorem symmetric_about_one (h : ∀ x, f (x - 1) = f (x + 1)) :
  ∀ x, f (1 - x) = f (1 + x) :=
sorry

-- Statement 4
theorem period_four (h1 : ∀ x, f (x + 1) = f (1 - x)) 
                    (h2 : ∀ x, f (x + 3) = f (3 - x)) :
  ∀ x, f x = f (x + 4) :=
sorry

end odd_function_symmetry_symmetric_about_one_period_four_l2635_263545


namespace remaining_candy_l2635_263532

def initial_candy : Real := 520.75
def given_away : Real := 234.56

theorem remaining_candy : 
  (initial_candy / 2) - given_away = 25.815 := by sorry

end remaining_candy_l2635_263532


namespace inequality_solution_set_l2635_263523

theorem inequality_solution_set (x : ℝ) : 
  2 / (x + 2) + 5 / (x + 4) ≥ 3 / 2 ↔ x ∈ Set.Icc (-4 : ℝ) (2/3) :=
by sorry

end inequality_solution_set_l2635_263523


namespace geometric_sequence_sum_l2635_263503

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 10
  geometric_sum a r n = 29524/59049 := by
  sorry

end geometric_sequence_sum_l2635_263503


namespace range_when_p_true_range_when_p_or_q_true_and_p_and_q_false_l2635_263501

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-2) (-1), x^2 - a ≥ 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x - (a-2) = 0

-- Theorem 1
theorem range_when_p_true (a : ℝ) : p a → a ≤ 1 := by sorry

-- Theorem 2
theorem range_when_p_or_q_true_and_p_and_q_false (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Ioo (-2) 1 ∪ Set.Ioi 1 := by sorry

end range_when_p_true_range_when_p_or_q_true_and_p_and_q_false_l2635_263501


namespace percentage_loss_calculation_l2635_263597

theorem percentage_loss_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1400 →
  selling_price = 1120 →
  (cost_price - selling_price) / cost_price * 100 = 20 := by
sorry

end percentage_loss_calculation_l2635_263597


namespace greening_task_equation_l2635_263516

/-- Represents the greening task parameters and equation -/
theorem greening_task_equation (x : ℝ) (h : x > 0) : 
  (600 : ℝ) / (x / (1 + 0.25)) - 600 / x = 30 ↔ 
  60 * (1 + 0.25) / x - 60 / x = 30 :=
by sorry


end greening_task_equation_l2635_263516


namespace pascal_triangle_elements_l2635_263518

/-- The number of elements in a row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def sumOfElements (n : ℕ) : ℕ := 
  (List.range n).map elementsInRow |>.sum

/-- The number of elements in the first 25 rows of Pascal's Triangle is 325 -/
theorem pascal_triangle_elements : sumOfElements 25 = 325 := by
  sorry

end pascal_triangle_elements_l2635_263518


namespace inequality_and_equality_condition_l2635_263592

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  ((a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a)) ∧ 
  ((a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end inequality_and_equality_condition_l2635_263592


namespace chess_tournament_score_l2635_263550

theorem chess_tournament_score (total_games wins draws losses : ℕ) 
  (old_score : ℚ) : 
  total_games = wins + draws + losses →
  old_score = wins + (1/2 : ℚ) * draws →
  total_games = 52 →
  old_score = 35 →
  (wins : ℤ) - losses = 18 :=
by sorry

end chess_tournament_score_l2635_263550


namespace range_of_m_l2635_263584

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 5) > 0}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x < m + 1}

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (B m ⊆ (Set.univ \ A)) → (-2 ≤ m ∧ m ≤ 4) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end range_of_m_l2635_263584


namespace eight_bead_necklace_arrangements_l2635_263517

/-- The number of distinct arrangements of n beads on a necklace,
    considering rotational and reflectional symmetry -/
def necklace_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements
    of 8 beads on a necklace is 2520 -/
theorem eight_bead_necklace_arrangements :
  necklace_arrangements 8 = 2520 := by
  sorry

end eight_bead_necklace_arrangements_l2635_263517


namespace inequality_equivalence_l2635_263519

theorem inequality_equivalence (x : ℝ) :
  (x - 4) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ 4 := by
  sorry

end inequality_equivalence_l2635_263519


namespace tightrope_length_calculation_l2635_263559

-- Define the length of the tightrope
def tightrope_length : ℝ := 320

-- Define the probability of breaking in the first 50 meters
def break_probability : ℝ := 0.15625

-- Theorem statement
theorem tightrope_length_calculation :
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ tightrope_length → 
    (50 / tightrope_length = break_probability)) →
  tightrope_length = 320 := by
sorry

end tightrope_length_calculation_l2635_263559


namespace perfect_square_identification_l2635_263524

theorem perfect_square_identification :
  ¬ ∃ (x : ℕ), 7^2051 = x^2 ∧
  ∃ (a b c d : ℕ), 6^2048 = a^2 ∧ 8^2050 = b^2 ∧ 9^2052 = c^2 ∧ 10^2040 = d^2 :=
by sorry

end perfect_square_identification_l2635_263524


namespace car_owners_without_motorcycle_l2635_263525

theorem car_owners_without_motorcycle (total : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ)
  (h1 : total = 351)
  (h2 : car_owners = 331)
  (h3 : motorcycle_owners = 45)
  (h4 : car_owners + motorcycle_owners - total ≥ 0) :
  car_owners - (car_owners + motorcycle_owners - total) = 306 := by
  sorry

end car_owners_without_motorcycle_l2635_263525


namespace mrs_hilt_dogs_l2635_263585

/-- The number of dogs Mrs. Hilt saw -/
def num_dogs : ℕ := 2

/-- The number of chickens Mrs. Hilt saw -/
def num_chickens : ℕ := 2

/-- The total number of legs Mrs. Hilt saw -/
def total_legs : ℕ := 12

/-- The number of legs each dog has -/
def dog_legs : ℕ := 4

/-- The number of legs each chicken has -/
def chicken_legs : ℕ := 2

theorem mrs_hilt_dogs :
  num_dogs * dog_legs + num_chickens * chicken_legs = total_legs :=
by sorry

end mrs_hilt_dogs_l2635_263585


namespace max_notebooks_inequality_l2635_263521

/-- Represents the budget in dollars -/
def budget : ℝ := 500

/-- Represents the regular price per notebook in dollars -/
def regularPrice : ℝ := 10

/-- Represents the discount rate as a decimal -/
def discountRate : ℝ := 0.2

/-- Represents the threshold number of notebooks for the discount to apply -/
def discountThreshold : ℕ := 15

/-- Theorem stating that the maximum number of notebooks that can be purchased
    is represented by the inequality 10 × 0.8x ≤ 500 -/
theorem max_notebooks_inequality :
  ∀ x : ℝ, x > discountThreshold →
    (x = budget / (regularPrice * (1 - discountRate))) ↔ 
    (regularPrice * (1 - discountRate) * x ≤ budget) :=
by sorry

end max_notebooks_inequality_l2635_263521


namespace correct_product_l2635_263528

theorem correct_product (a b c : ℚ) (h1 : a = 0.25) (h2 : b = 3.4) (h3 : c = 0.85) 
  (h4 : (25 : ℤ) * 34 = 850) : a * b = c := by
  sorry

end correct_product_l2635_263528


namespace patrick_has_25_dollars_l2635_263579

/-- Calculates the amount of money Patrick has after saving for a bicycle and lending money to a friend. -/
def patricks_money (bicycle_price : ℕ) (amount_lent : ℕ) : ℕ :=
  bicycle_price / 2 - amount_lent

/-- Proves that Patrick has $25 after saving for a $150 bicycle and lending $50 to a friend. -/
theorem patrick_has_25_dollars :
  patricks_money 150 50 = 25 := by
  sorry

end patrick_has_25_dollars_l2635_263579


namespace six_digit_numbers_with_zero_count_six_digit_numbers_with_zero_l2635_263582

theorem six_digit_numbers_with_zero (total : ℕ) (no_zero : ℕ) : ℕ :=
  total - no_zero

theorem count_six_digit_numbers_with_zero :
  six_digit_numbers_with_zero 900000 531441 = 368559 := by
  sorry

end six_digit_numbers_with_zero_count_six_digit_numbers_with_zero_l2635_263582


namespace check_cashing_mistake_l2635_263506

theorem check_cashing_mistake (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →
  (10 ≤ y ∧ y ≤ 99) →
  (100 * y + x) - (100 * x + y) = 1820 →
  ∃ x y, y = x + 18 ∧ y = 2 * x :=
sorry

end check_cashing_mistake_l2635_263506


namespace sum_minimized_at_Q₅_l2635_263587

/-- A type representing points on a line -/
structure PointOnLine where
  position : ℝ

/-- The distance between two points on a line -/
def distance (p q : PointOnLine) : ℝ := |p.position - q.position|

/-- The sum of distances from a point Q to points Q₁, ..., Q₉ -/
def sumOfDistances (Q Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : PointOnLine) : ℝ :=
  distance Q Q₁ + distance Q Q₂ + distance Q Q₃ + distance Q Q₄ + 
  distance Q Q₅ + distance Q Q₆ + distance Q Q₇ + distance Q Q₈ + distance Q Q₉

/-- The theorem stating that the sum of distances is minimized when Q is at Q₅ -/
theorem sum_minimized_at_Q₅ 
  (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : PointOnLine) 
  (h : Q₁.position < Q₂.position ∧ Q₂.position < Q₃.position ∧ 
       Q₃.position < Q₄.position ∧ Q₄.position < Q₅.position ∧ 
       Q₅.position < Q₆.position ∧ Q₆.position < Q₇.position ∧ 
       Q₇.position < Q₈.position ∧ Q₈.position < Q₉.position) :
  ∀ Q : PointOnLine, sumOfDistances Q Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ ≥ 
                     sumOfDistances Q₅ Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ :=
sorry

end sum_minimized_at_Q₅_l2635_263587


namespace no_prime_divisor_8t_plus_5_l2635_263573

theorem no_prime_divisor_8t_plus_5 (x : ℕ+) :
  ∀ p : ℕ, Prime p → p % 8 = 5 →
    ¬(p ∣ (8 * x^4 - 2)) ∧
    ¬(p ∣ (8 * x^4 - 1)) ∧
    ¬(p ∣ (8 * x^4)) ∧
    ¬(p ∣ (8 * x^4 + 1)) :=
sorry

end no_prime_divisor_8t_plus_5_l2635_263573


namespace possible_values_of_x_l2635_263586

def S (x : ℝ) : Set ℝ := {1, 2, x^2}

theorem possible_values_of_x : {x : ℝ | x ∈ S x} = {0, 2} := by sorry

end possible_values_of_x_l2635_263586


namespace hash_six_eight_l2635_263561

-- Define the # operation
def hash (a b : ℤ) : ℤ := 3*a - 3*b + 4

-- Theorem statement
theorem hash_six_eight : hash 6 8 = -2 := by
  sorry

end hash_six_eight_l2635_263561
