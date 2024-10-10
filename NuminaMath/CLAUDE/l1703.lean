import Mathlib

namespace park_entry_exit_choices_l1703_170359

def num_gates : ℕ := 5

theorem park_entry_exit_choices :
  (num_gates * (num_gates - 1) : ℕ) = 20 := by
  sorry

end park_entry_exit_choices_l1703_170359


namespace unity_digit_of_n_l1703_170315

theorem unity_digit_of_n (n : ℕ) (h : 3 * n = 999^1000) : n % 10 = 7 := by
  sorry

end unity_digit_of_n_l1703_170315


namespace rectangle_cut_theorem_l1703_170365

/-- Represents a figure cut from the rectangle -/
structure Figure where
  area : ℕ
  perimeter : ℕ

/-- The problem statement -/
theorem rectangle_cut_theorem :
  ∃ (figures : List Figure),
    figures.length = 5 ∧
    (figures.map Figure.area).sum = 30 ∧
    (∀ f ∈ figures, f.perimeter = 2 * f.area) ∧
    (∃ x : ℕ, figures.map Figure.area = [x, x+1, x+2, x+3, x+4]) :=
by
  sorry

end rectangle_cut_theorem_l1703_170365


namespace line_parallel_to_x_axis_l1703_170369

/-- Given two points A and B, if the line AB is parallel to the x-axis, then m = -1 --/
theorem line_parallel_to_x_axis (m : ℝ) : 
  let A : ℝ × ℝ := (m + 1, -2)
  let B : ℝ × ℝ := (3, m - 1)
  (A.2 = B.2) → m = -1 := by sorry

end line_parallel_to_x_axis_l1703_170369


namespace unique_digit_divisibility_l1703_170312

theorem unique_digit_divisibility : 
  ∃! (A : ℕ), A < 10 ∧ 70 % A = 0 ∧ (546200 + 10 * A + 4) % 4 = 0 := by
  sorry

end unique_digit_divisibility_l1703_170312


namespace coprimality_preserving_polynomials_l1703_170344

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The property that a polynomial preserves coprimality -/
def PreservesCoprimality (P : IntPolynomial) : Prop :=
  ∀ a b : ℤ, Int.gcd a b = 1 → Int.gcd (P.eval a) (P.eval b) = 1

/-- Characterization of polynomials that preserve coprimality -/
theorem coprimality_preserving_polynomials :
  ∀ P : IntPolynomial,
  PreservesCoprimality P ↔
  (∃ n : ℕ, P = Polynomial.monomial n 1) ∨
  (∃ n : ℕ, P = Polynomial.monomial n (-1)) :=
sorry

end coprimality_preserving_polynomials_l1703_170344


namespace large_cheese_block_volume_l1703_170340

/-- The volume of a large cheese block is 32 cubic feet -/
theorem large_cheese_block_volume :
  ∀ (w d l : ℝ),
  w * d * l = 4 →
  (2 * w) * (2 * d) * (2 * l) = 32 :=
by sorry

end large_cheese_block_volume_l1703_170340


namespace checkerboard_coverage_l1703_170384

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- Determines if a checkerboard can be covered by dominoes -/
def can_be_covered (board : Checkerboard) : Prop :=
  Even (board.rows * board.cols)

/-- Theorem stating that a checkerboard can be covered if and only if its area is even -/
theorem checkerboard_coverage (board : Checkerboard) :
  can_be_covered board ↔ Even (board.rows * board.cols) := by sorry

end checkerboard_coverage_l1703_170384


namespace log_length_l1703_170355

/-- Represents the properties of a log that has been cut in half -/
structure LogCut where
  weight_per_foot : ℝ
  weight_of_piece : ℝ
  original_length : ℝ

/-- Theorem stating that given the conditions, the original log length is 20 feet -/
theorem log_length (log : LogCut) 
  (h1 : log.weight_per_foot = 150)
  (h2 : log.weight_of_piece = 1500) :
  log.original_length = 20 := by
  sorry

#check log_length

end log_length_l1703_170355


namespace negation_of_or_implies_both_false_l1703_170380

theorem negation_of_or_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end negation_of_or_implies_both_false_l1703_170380


namespace marbles_shared_proof_l1703_170345

/-- The number of marbles Jack starts with -/
def initial_marbles : ℕ := 62

/-- The number of marbles Jack ends with -/
def final_marbles : ℕ := 29

/-- The number of marbles Jack shared with Rebecca -/
def shared_marbles : ℕ := initial_marbles - final_marbles

theorem marbles_shared_proof : shared_marbles = 33 := by
  sorry

end marbles_shared_proof_l1703_170345


namespace blue_balls_drawn_first_probability_l1703_170372

def num_blue_balls : ℕ := 4
def num_yellow_balls : ℕ := 3
def total_balls : ℕ := num_blue_balls + num_yellow_balls

def favorable_outcomes : ℕ := Nat.choose (total_balls - 1) num_yellow_balls
def total_outcomes : ℕ := Nat.choose total_balls num_yellow_balls

theorem blue_balls_drawn_first_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 7 := by
  sorry

end blue_balls_drawn_first_probability_l1703_170372


namespace gcd_lcm_inequality_implies_divisibility_l1703_170378

theorem gcd_lcm_inequality_implies_divisibility (a b : ℕ) 
  (h : a * Nat.gcd a b + b * Nat.lcm a b < (5/2) * a * b) : 
  b ∣ a := by
  sorry

end gcd_lcm_inequality_implies_divisibility_l1703_170378


namespace three_digit_square_end_same_l1703_170376

theorem three_digit_square_end_same (A : ℕ) : 
  (100 ≤ A ∧ A < 1000) ∧ (A^2 % 1000 = A) ↔ (A = 376 ∨ A = 625) :=
by sorry

end three_digit_square_end_same_l1703_170376


namespace fourth_triangle_exists_l1703_170375

/-- Given four positive real numbers that can form three different triangles,
    prove that they can form a fourth triangle. -/
theorem fourth_triangle_exists (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab_c : a + b > c ∧ a + c > b ∧ b + c > a)
  (hab_d : a + b > d ∧ a + d > b ∧ b + d > a)
  (acd : a + c > d ∧ a + d > c ∧ c + d > a) :
  b + c > d ∧ b + d > c ∧ c + d > b := by
  sorry

end fourth_triangle_exists_l1703_170375


namespace max_min_difference_c_l1703_170358

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 2) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 12) : 
  ∃ (c_max c_min : ℝ), 
    (∀ c', (∃ a' b', a' + b' + c' = 2 ∧ a'^2 + b'^2 + c'^2 = 12) → c' ≤ c_max) ∧
    (∀ c', (∃ a' b', a' + b' + c' = 2 ∧ a'^2 + b'^2 + c'^2 = 12) → c' ≥ c_min) ∧
    c_max - c_min = 16/3 :=
by sorry

end max_min_difference_c_l1703_170358


namespace rectangular_solid_volume_l1703_170339

/-- The volume of a rectangular solid with specific face areas and sum of dimensions -/
theorem rectangular_solid_volume
  (a b c : ℝ)
  (side_area : a * b = 15)
  (front_area : b * c = 10)
  (bottom_area : c * a = 6)
  (sum_dimensions : a + b + c = 11)
  : a * b * c = 90 := by
  sorry

end rectangular_solid_volume_l1703_170339


namespace emma_hit_eleven_l1703_170373

-- Define the set of players
inductive Player : Type
| Alice : Player
| Ben : Player
| Cindy : Player
| Dave : Player
| Emma : Player
| Felix : Player

-- Define the score function
def score : Player → Nat
| Player.Alice => 21
| Player.Ben => 10
| Player.Cindy => 18
| Player.Dave => 15
| Player.Emma => 30
| Player.Felix => 22

-- Define the set of possible target values
def target_values : Finset Nat := Finset.range 12 \ {0}

-- Define a function to check if a player's score can be made up of three distinct values from the target
def valid_score (p : Player) : Prop :=
  ∃ (a b c : Nat), a ∈ target_values ∧ b ∈ target_values ∧ c ∈ target_values ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = score p

-- Theorem: Emma is the only player who could have hit the region worth 11 points
theorem emma_hit_eleven :
  ∀ (p : Player), p ≠ Player.Emma → 
    (valid_score p → ¬∃ (a b : Nat), a ∈ target_values ∧ b ∈ target_values ∧ a ≠ b ∧ a + b + 11 = score p) ∧
    (valid_score Player.Emma → ∃ (a b : Nat), a ∈ target_values ∧ b ∈ target_values ∧ a ≠ b ∧ a + b + 11 = score Player.Emma) :=
by sorry

end emma_hit_eleven_l1703_170373


namespace monkey_banana_problem_l1703_170300

/-- The number of monkeys in the initial scenario -/
def initial_monkeys : ℕ := 8

/-- The time taken to eat bananas in minutes -/
def eating_time : ℕ := 8

/-- The number of bananas eaten in the initial scenario -/
def initial_bananas : ℕ := 8

/-- The number of monkeys in the second scenario -/
def second_monkeys : ℕ := 3

/-- The number of bananas eaten in the second scenario -/
def second_bananas : ℕ := 3

theorem monkey_banana_problem :
  (initial_monkeys * eating_time = initial_bananas * eating_time) ∧
  (second_monkeys * eating_time = second_bananas * eating_time) →
  initial_monkeys = initial_bananas :=
by sorry

end monkey_banana_problem_l1703_170300


namespace benjamin_total_steps_l1703_170322

/-- Calculates the total distance traveled in steps given various modes of transportation -/
def total_steps_traveled (steps_per_mile : ℕ) (initial_walk : ℕ) (subway_miles : ℕ) (second_walk : ℕ) (cab_miles : ℕ) : ℕ :=
  initial_walk + (subway_miles * steps_per_mile) + second_walk + (cab_miles * steps_per_mile)

/-- The total steps traveled by Benjamin is 24000 -/
theorem benjamin_total_steps :
  total_steps_traveled 2000 2000 7 3000 3 = 24000 := by
  sorry


end benjamin_total_steps_l1703_170322


namespace total_questions_on_math_test_l1703_170303

/-- The number of questions on a math test -/
def math_test_questions (word_problems subtraction_problems answered_questions blank_questions : ℕ) : Prop :=
  word_problems + subtraction_problems = answered_questions + blank_questions

/-- Theorem: There are 45 questions on the math test -/
theorem total_questions_on_math_test :
  ∃ (word_problems subtraction_problems answered_questions blank_questions : ℕ),
    word_problems = 17 ∧
    subtraction_problems = 28 ∧
    answered_questions = 38 ∧
    blank_questions = 7 ∧
    math_test_questions word_problems subtraction_problems answered_questions blank_questions ∧
    answered_questions + blank_questions = 45 :=
by
  sorry

end total_questions_on_math_test_l1703_170303


namespace third_player_games_l1703_170361

/-- Represents a table tennis game with three players -/
structure TableTennisGame where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The rules of the game ensure that the total games is the sum of games played by any two players -/
def valid_game (g : TableTennisGame) : Prop :=
  g.total_games = g.player1_games + g.player2_games ∧
  g.total_games = g.player1_games + g.player3_games ∧
  g.total_games = g.player2_games + g.player3_games

/-- The theorem to be proved -/
theorem third_player_games (g : TableTennisGame) 
  (h1 : g.player1_games = 10)
  (h2 : g.player2_games = 21)
  (h3 : valid_game g) :
  g.player3_games = 11 := by
  sorry

end third_player_games_l1703_170361


namespace remainder_of_power_mod_quadratic_l1703_170379

theorem remainder_of_power_mod_quadratic (x : ℤ) : 
  (x + 2)^1004 ≡ -x [ZMOD (x^2 - x + 1)] :=
sorry

end remainder_of_power_mod_quadratic_l1703_170379


namespace cost_price_percentage_l1703_170377

theorem cost_price_percentage (selling_price cost_price : ℝ) 
  (h_profit_percent : (selling_price - cost_price) / cost_price = 1/3) :
  cost_price / selling_price = 3/4 := by
sorry

end cost_price_percentage_l1703_170377


namespace total_fish_count_l1703_170362

-- Define the number of fish for each person
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- Define the total number of fish
def total_fish : ℕ := billy_fish + tony_fish + sarah_fish + bobby_fish

-- Theorem statement
theorem total_fish_count : total_fish = 145 := by
  sorry

end total_fish_count_l1703_170362


namespace smallest_positive_integer_3003m_55555n_l1703_170396

theorem smallest_positive_integer_3003m_55555n :
  ∃ (k : ℕ), k > 0 ∧ (∀ (j : ℕ), j > 0 → (∃ (m n : ℤ), j = 3003 * m + 55555 * n) → k ≤ j) ∧
  (∃ (m n : ℤ), k = 3003 * m + 55555 * n) :=
by sorry

end smallest_positive_integer_3003m_55555n_l1703_170396


namespace power_product_equals_negative_one_l1703_170334

theorem power_product_equals_negative_one : 
  (4 : ℝ)^7 * (-0.25 : ℝ)^7 = -1 := by sorry

end power_product_equals_negative_one_l1703_170334


namespace order_of_logarithmic_expressions_l1703_170320

theorem order_of_logarithmic_expressions :
  let a := 2 * Real.log 0.99
  let b := Real.log 0.98
  let c := Real.sqrt 0.96 - 1
  c < b ∧ b < a := by sorry

end order_of_logarithmic_expressions_l1703_170320


namespace pyramid_volume_l1703_170395

/-- The volume of a pyramid with specific properties -/
theorem pyramid_volume (base_angle : Real) (lateral_edge : Real) (inclination : Real) : 
  base_angle = π/8 →
  lateral_edge = Real.sqrt 6 →
  inclination = 5*π/13 →
  ∃ (volume : Real), 
    volume = Real.sqrt 3 * Real.sin (10*π/13) * Real.cos (5*π/13) ∧
    volume = (1/3) * 
             ((lateral_edge * Real.cos inclination)^2 * Real.sin (2*base_angle)) * 
             (lateral_edge * Real.sin inclination) :=
by sorry

end pyramid_volume_l1703_170395


namespace circle_equation_l1703_170370

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line on which the center of the required circle lies
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

-- Define the equation of the required circle
def requiredCircle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 13

-- Theorem statement
theorem circle_equation : 
  ∀ x y : ℝ, 
  (circle1 x y ∧ circle2 x y) → 
  (∃ a b : ℝ, centerLine a b ∧ 
    ((x - a)^2 + (y - b)^2 = (x + 1)^2 + (y - 1)^2)) → 
  requiredCircle x y :=
sorry

end circle_equation_l1703_170370


namespace tom_average_speed_l1703_170374

theorem tom_average_speed (karen_speed : ℝ) (karen_delay : ℝ) (win_margin : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 →
  karen_delay = 4 / 60 →
  win_margin = 4 →
  tom_distance = 24 →
  ∃ (tom_speed : ℝ), tom_speed = 300 / 7 ∧
    karen_speed * (tom_distance / karen_speed) = 
    tom_speed * (tom_distance / karen_speed + karen_delay) - win_margin :=
by sorry

end tom_average_speed_l1703_170374


namespace min_value_trig_expression_l1703_170352

theorem min_value_trig_expression (x : ℝ) :
  (Real.sin x)^8 + 16 * (Real.cos x)^8 + 1 ≥ 
  4.7692 * ((Real.sin x)^6 + 4 * (Real.cos x)^6 + 1) := by
  sorry

end min_value_trig_expression_l1703_170352


namespace distance_to_origin_l1703_170319

/-- The distance from point P (-2, 4) to the origin (0, 0) is 2√5 -/
theorem distance_to_origin : 
  let P : ℝ × ℝ := (-2, 4)
  let O : ℝ × ℝ := (0, 0)
  Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end distance_to_origin_l1703_170319


namespace consecutive_integers_sum_l1703_170364

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 812) : 
  x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l1703_170364


namespace maintenance_time_is_three_minutes_l1703_170316

/-- Represents the passage scenario with order maintenance --/
structure PassageScenario where
  normal_rate : ℕ
  congested_rate : ℕ
  people_waiting : ℕ
  time_saved : ℕ

/-- Calculates the time spent maintaining order --/
def maintenance_time (scenario : PassageScenario) : ℕ :=
  let total_wait_time := scenario.people_waiting / scenario.congested_rate
  let actual_wait_time := total_wait_time - scenario.time_saved
  actual_wait_time

/-- Theorem stating that the maintenance time is 3 minutes for the given scenario --/
theorem maintenance_time_is_three_minutes 
  (scenario : PassageScenario)
  (h1 : scenario.normal_rate = 9)
  (h2 : scenario.congested_rate = 3)
  (h3 : scenario.people_waiting = 36)
  (h4 : scenario.time_saved = 6) :
  maintenance_time scenario = 3 := by
  sorry

#eval maintenance_time { normal_rate := 9, congested_rate := 3, people_waiting := 36, time_saved := 6 }

end maintenance_time_is_three_minutes_l1703_170316


namespace green_balls_count_l1703_170389

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) 
  (h_total : total = 60)
  (h_white : white = 22)
  (h_yellow : yellow = 8)
  (h_red : red = 5)
  (h_purple : purple = 7)
  (h_prob : prob_not_red_purple = 4/5) :
  ∃ green : ℕ, 
    green = total - (white + yellow + red + purple) ∧ 
    (white + green + yellow : ℚ) / total = prob_not_red_purple :=
by sorry

end green_balls_count_l1703_170389


namespace math_competition_solution_l1703_170350

/-- Represents the number of contestants from each school -/
structure ContestantCounts where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- The conditions of the math competition -/
def ValidContestantCounts (counts : ContestantCounts) : Prop :=
  counts.A + counts.B = 16 ∧
  counts.B + counts.C = 20 ∧
  counts.C + counts.D = 34 ∧
  counts.A < counts.B ∧
  counts.B < counts.C ∧
  counts.C < counts.D

/-- The theorem to prove -/
theorem math_competition_solution :
  ∃ (counts : ContestantCounts), ValidContestantCounts counts ∧
    counts.A = 7 ∧ counts.B = 9 ∧ counts.C = 11 ∧ counts.D = 23 := by
  sorry

end math_competition_solution_l1703_170350


namespace power_function_alpha_l1703_170394

/-- Given a power function y = mx^α where m and α are real numbers,
    if the graph passes through the point (8, 1/4), then α equals -2/3. -/
theorem power_function_alpha (m α : ℝ) :
  (∃ (x y : ℝ), x = 8 ∧ y = 1/4 ∧ y = m * x^α) → α = -2/3 := by
  sorry

end power_function_alpha_l1703_170394


namespace polygon_with_45_degree_exterior_angles_has_8_sides_l1703_170363

/-- A polygon with exterior angles of 45° has 8 sides. -/
theorem polygon_with_45_degree_exterior_angles_has_8_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
  n > 0 →
  exterior_angle = 45 →
  (n : ℝ) * exterior_angle = 360 →
  n = 8 := by
sorry

end polygon_with_45_degree_exterior_angles_has_8_sides_l1703_170363


namespace max_speed_theorem_l1703_170351

/-- Represents a set of observations for machine speed and defective items produced. -/
structure Observation where
  speed : ℝ
  defects : ℝ

/-- Calculates the slope of the linear regression line. -/
def calculateSlope (observations : List Observation) : ℝ :=
  sorry

/-- Calculates the y-intercept of the linear regression line. -/
def calculateIntercept (observations : List Observation) (slope : ℝ) : ℝ :=
  sorry

/-- Theorem: The maximum speed at which the machine can operate while producing
    no more than 10 defective items per hour is 15 revolutions per second. -/
theorem max_speed_theorem (observations : List Observation)
    (h1 : observations = [⟨8, 5⟩, ⟨12, 8⟩, ⟨14, 9⟩, ⟨16, 11⟩])
    (h2 : ∀ obs ∈ observations, obs.speed > 0 ∧ obs.defects > 0)
    (h3 : calculateSlope observations > 0) : 
    let slope := calculateSlope observations
    let intercept := calculateIntercept observations slope
    Int.floor ((10 - intercept) / slope) = 15 := by
  sorry

end max_speed_theorem_l1703_170351


namespace parallelogram_area_32_14_l1703_170337

/-- The area of a parallelogram given its base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 cm and height 14 cm is 448 square centimeters -/
theorem parallelogram_area_32_14 : parallelogram_area 32 14 = 448 := by
  sorry

end parallelogram_area_32_14_l1703_170337


namespace log_product_equals_one_l1703_170367

theorem log_product_equals_one : Real.log 2 / Real.log 5 * (2 * Real.log 5 / (2 * Real.log 2)) = 1 := by
  sorry

end log_product_equals_one_l1703_170367


namespace arithmetic_sequence_property_l1703_170317

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  S seq 3 = seq.a 2 + 10 * seq.a 1 →
  seq.a 5 = 9 →
  seq.a 1 = 1/3 := by
  sorry

end arithmetic_sequence_property_l1703_170317


namespace tan_pi_minus_alpha_l1703_170331

theorem tan_pi_minus_alpha (α : Real) (h : 3 * Real.sin α = Real.cos α) :
  Real.tan (π - α) = -1/3 := by
  sorry

end tan_pi_minus_alpha_l1703_170331


namespace dance_class_permutations_l1703_170348

theorem dance_class_permutations :
  Nat.factorial 8 = 40320 := by
  sorry

end dance_class_permutations_l1703_170348


namespace evaluation_ratio_l1703_170353

def relevance_percentage : ℚ := 45 / 100
def language_percentage : ℚ := 25 / 100
def structure_percentage : ℚ := 30 / 100

theorem evaluation_ratio :
  let r := relevance_percentage
  let l := language_percentage
  let s := structure_percentage
  let gcd := (r * 100).num.gcd ((l * 100).num.gcd (s * 100).num)
  ((r * 100).num / gcd, (l * 100).num / gcd, (s * 100).num / gcd) = (9, 5, 6) := by
  sorry

end evaluation_ratio_l1703_170353


namespace wall_length_proof_l1703_170308

/-- Proves that the length of a wall is 29 meters given specific brick and wall dimensions and the number of bricks required. -/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_width : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_width = 2 →
  wall_height = 0.75 →
  num_bricks = 29000 →
  ∃ (wall_length : ℝ), wall_length = 29 :=
by
  sorry

#check wall_length_proof

end wall_length_proof_l1703_170308


namespace complement_intersection_equals_set_l1703_170368

def U : Finset Int := {-1, 0, 1, 2, 3}
def A : Finset Int := {-1, 0}
def B : Finset Int := {0, 1, 2}

theorem complement_intersection_equals_set : (U \ A) ∩ B = {1, 2} := by sorry

end complement_intersection_equals_set_l1703_170368


namespace seventieth_number_with_remainder_five_seventieth_number_is_557_l1703_170318

theorem seventieth_number_with_remainder_five : ℕ → Prop :=
  fun n => ∃ k : ℕ, n = 8 * k + 5 ∧ n > 0

theorem seventieth_number_is_557 :
  ∃! n : ℕ, seventieth_number_with_remainder_five n ∧ (∃ m : ℕ, m = 70 ∧
    (∀ k < n, seventieth_number_with_remainder_five k →
      (∃ i : ℕ, i < m ∧ (∀ j < k, seventieth_number_with_remainder_five j → ∃ l : ℕ, l < i)))) ∧
  n = 557 :=
by sorry

end seventieth_number_with_remainder_five_seventieth_number_is_557_l1703_170318


namespace high_sulfur_oil_count_l1703_170356

/-- Represents the properties of an oil sample set -/
structure OilSampleSet where
  total_samples : Nat
  heavy_oil_prob : Rat
  light_low_sulfur_prob : Rat

/-- Theorem stating the number of high-sulfur oil samples in a given set -/
theorem high_sulfur_oil_count (s : OilSampleSet)
  (h1 : s.total_samples % 7 = 0)
  (h2 : s.total_samples ≤ 100 ∧ ∀ n, n % 7 = 0 → n ≤ 100 → s.total_samples ≥ n)
  (h3 : s.heavy_oil_prob = 1 / 7)
  (h4 : s.light_low_sulfur_prob = 9 / 14) :
  (s.total_samples : Rat) * s.heavy_oil_prob +
  (s.total_samples : Rat) * (1 - s.heavy_oil_prob) * (1 - s.light_low_sulfur_prob) = 44 := by
  sorry

end high_sulfur_oil_count_l1703_170356


namespace x_minus_y_values_l1703_170324

theorem x_minus_y_values (x y : ℝ) (hx : |x| = 5) (hy : |y| = 3) (hxy : y > x) :
  x - y = -8 ∨ x - y = -2 := by
  sorry

end x_minus_y_values_l1703_170324


namespace min_value_condition_inequality_condition_l1703_170310

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x + m|

-- Theorem for part 1
theorem min_value_condition (m : ℝ) :
  (∃ (x : ℝ), f x m = 2 ∧ ∀ (y : ℝ), f y m ≥ 2) ↔ (m = 3 ∨ m = -1) :=
sorry

-- Theorem for part 2
theorem inequality_condition (m : ℝ) :
  (∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f x m ≤ 2 * x + 3) ↔ (0 ≤ m ∧ m ≤ 2) :=
sorry

end min_value_condition_inequality_condition_l1703_170310


namespace discount_rate_is_four_percent_l1703_170357

def marked_price : ℝ := 125
def selling_price : ℝ := 120

theorem discount_rate_is_four_percent :
  (marked_price - selling_price) / marked_price * 100 = 4 := by
  sorry

end discount_rate_is_four_percent_l1703_170357


namespace cube_side_ratio_l1703_170335

/-- Given two cubes of the same material, this theorem proves that if their weights are in the ratio of 32:4, then their side lengths are in the ratio of 2:1. -/
theorem cube_side_ratio (s₁ s₂ : ℝ) (w₁ w₂ : ℝ) (h₁ : w₁ = 4) (h₂ : w₂ = 32) :
  w₁ * s₂^3 = w₂ * s₁^3 → s₂ / s₁ = 2 := by sorry

end cube_side_ratio_l1703_170335


namespace subset_implies_a_value_l1703_170311

theorem subset_implies_a_value (A B : Set ℤ) (a : ℤ) 
  (h1 : A = {0, 1}) 
  (h2 : B = {-1, 0, a+3}) 
  (h3 : A ⊆ B) : 
  a = -2 := by
sorry

end subset_implies_a_value_l1703_170311


namespace prime_power_equality_l1703_170306

theorem prime_power_equality (p : ℕ) (k : ℕ) (hp : Prime p) (hk : k > 1) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (m, n) ≠ (1, 1) ∧ 
    (m^p + n^p) / 2 = ((m + n) / 2)^k) ↔ k = p :=
by sorry

end prime_power_equality_l1703_170306


namespace solution_set_part1_range_of_a_part2_l1703_170342

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - t|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x < 1/2 ∨ x > 5/2} :=
by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ t ∈ Set.Icc 1 2,
    (∀ x ∈ Set.Icc (-1) 3, ∃ a : ℝ, f t x ≥ a + x) →
    ∃ a : ℝ, a ≤ -1 ∧ ∀ x ∈ Set.Icc (-1) 3, f t x ≥ a + x :=
by sorry

end solution_set_part1_range_of_a_part2_l1703_170342


namespace jugglers_balls_l1703_170327

theorem jugglers_balls (num_jugglers : ℕ) (total_balls : ℕ) 
  (h1 : num_jugglers = 378) 
  (h2 : total_balls = 2268) : 
  total_balls / num_jugglers = 6 := by
  sorry

end jugglers_balls_l1703_170327


namespace field_trip_van_occupancy_l1703_170305

theorem field_trip_van_occupancy :
  let num_vans : ℕ := 2
  let num_buses : ℕ := 3
  let people_per_bus : ℕ := 20
  let total_people : ℕ := 76
  let people_in_vans : ℕ := total_people - (num_buses * people_per_bus)
  people_in_vans / num_vans = 8 := by
  sorry

end field_trip_van_occupancy_l1703_170305


namespace sphere_intersection_l1703_170307

/-- Sphere intersection problem -/
theorem sphere_intersection (center : ℝ × ℝ × ℝ) (R : ℝ) :
  let (x₀, y₀, z₀) := center
  -- Conditions
  (x₀ = 3 ∧ y₀ = -2 ∧ z₀ = 5) →  -- Sphere center
  (R^2 = 29) →  -- Sphere radius
  -- xy-plane intersection
  ((3 - x₀)^2 + (-2 - y₀)^2 = 2^2) →
  -- yz-plane intersection
  ((0 - x₀)^2 + (5 - z₀)^2 = 3^2) →
  -- xz-plane intersection
  (∃ (x z : ℝ), (x - x₀)^2 + (z - z₀)^2 = 8 ∧ z = -x + 3) →
  -- Conclusion
  (3^2 = 3^2 ∧ 8 = (2 * Real.sqrt 2)^2) :=
by sorry

end sphere_intersection_l1703_170307


namespace alice_wins_iff_zero_l1703_170301

/-- Alice's winning condition in the quadratic equation game -/
theorem alice_wins_iff_zero (a b c : ℝ) : 
  (∀ d : ℝ, ¬(∃ x y : ℝ, x ≠ y ∧ 
    ((a + d) * x^2 + (b + d) * x + (c + d) = 0) ∧ 
    ((a + d) * y^2 + (b + d) * y + (c + d) = 0)))
  ↔ 
  (a = 0 ∧ b = 0 ∧ c = 0) :=
sorry

end alice_wins_iff_zero_l1703_170301


namespace a_squared_gt_b_squared_sufficient_not_necessary_l1703_170336

theorem a_squared_gt_b_squared_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a^2 > b^2 → abs a > b) ∧
  (∃ a b : ℝ, abs a > b ∧ a^2 ≤ b^2) :=
by sorry

end a_squared_gt_b_squared_sufficient_not_necessary_l1703_170336


namespace x1_value_l1703_170314

theorem x1_value (x₁ x₂ x₃ : ℝ) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h2 : (1-x₁)^3 + (x₁-x₂)^3 + (x₂-x₃)^3 + x₃^3 = 1/8) : 
  x₁ = 3/4 := by
  sorry

end x1_value_l1703_170314


namespace parallelogram_area_l1703_170343

structure Vector2D where
  x : ℝ
  y : ℝ

def angle (v w : Vector2D) : ℝ := sorry

def norm (v : Vector2D) : ℝ := sorry

def cross (v w : Vector2D) : ℝ := sorry

theorem parallelogram_area (p q : Vector2D) : 
  let a := Vector2D.mk (6 * p.x - q.x) (6 * p.y - q.y)
  let b := Vector2D.mk (5 * q.x + p.x) (5 * q.y + p.y)
  norm p = 1/2 →
  norm q = 4 →
  angle p q = 5 * π / 6 →
  abs (cross a b) = 31 := by sorry

end parallelogram_area_l1703_170343


namespace salem_poem_stanzas_l1703_170386

/-- Represents a poem with a specific structure -/
structure Poem where
  lines_per_stanza : ℕ
  words_per_line : ℕ
  total_words : ℕ

/-- Calculates the number of stanzas in a poem -/
def number_of_stanzas (p : Poem) : ℕ :=
  p.total_words / (p.lines_per_stanza * p.words_per_line)

/-- Theorem: A poem with 10 lines per stanza, 8 words per line, 
    and 1600 total words has 20 stanzas -/
theorem salem_poem_stanzas :
  let p : Poem := ⟨10, 8, 1600⟩
  number_of_stanzas p = 20 := by
  sorry

#check salem_poem_stanzas

end salem_poem_stanzas_l1703_170386


namespace equation_is_linear_in_two_vars_l1703_170393

/-- A linear equation in two variables -/
structure LinearEquation2Var where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop
  is_linear : ∀ x y, eq x y ↔ a * x + b * y + c = 0

/-- The equation y - x = 1 -/
def equation : ℝ → ℝ → Prop :=
  fun x y => y - x = 1

theorem equation_is_linear_in_two_vars :
  ∃ le : LinearEquation2Var, le.eq = equation :=
sorry

end equation_is_linear_in_two_vars_l1703_170393


namespace money_left_after_tickets_l1703_170328

/-- The amount of money Olivia and Nigel have left after buying tickets -/
def money_left (olivia_money : ℕ) (nigel_money : ℕ) (num_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (olivia_money + nigel_money) - (num_tickets * ticket_price)

/-- Theorem stating the amount of money left after buying tickets -/
theorem money_left_after_tickets :
  money_left 112 139 6 28 = 83 := by
  sorry

end money_left_after_tickets_l1703_170328


namespace apple_selling_price_l1703_170360

theorem apple_selling_price (cost_price : ℝ) (loss_fraction : ℝ) (selling_price : ℝ) : 
  cost_price = 21 →
  loss_fraction = 1/6 →
  selling_price = cost_price * (1 - loss_fraction) →
  selling_price = 17.50 := by
sorry

end apple_selling_price_l1703_170360


namespace oranges_per_tree_l1703_170346

/-- Represents the number of oranges picked by Betty -/
def betty_oranges : ℕ := 15

/-- Represents the number of oranges picked by Bill -/
def bill_oranges : ℕ := 12

/-- Represents the number of oranges picked by Frank -/
def frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)

/-- Represents the number of seeds Frank planted -/
def seeds_planted : ℕ := 2 * frank_oranges

/-- Represents the total number of oranges Philip can pick -/
def philip_total_oranges : ℕ := 810

/-- Theorem stating that the number of oranges per tree for Philip to pick is 5 -/
theorem oranges_per_tree :
  philip_total_oranges / seeds_planted = 5 := by sorry

end oranges_per_tree_l1703_170346


namespace integer_midpoint_exists_l1703_170341

def Point := ℤ × ℤ

theorem integer_midpoint_exists (P : Fin 5 → Point) :
  ∃ i j : Fin 5, i ≠ j ∧ 
    let (xi, yi) := P i
    let (xj, yj) := P j
    (xi + xj) % 2 = 0 ∧ (yi + yj) % 2 = 0 :=
by sorry

end integer_midpoint_exists_l1703_170341


namespace unique_root_of_sum_with_shift_l1703_170349

/-- Given a monic quadratic polynomial with two distinct roots, 
    prove that f(x) + f(x - √D) = 0 has exactly one root. -/
theorem unique_root_of_sum_with_shift 
  (b c : ℝ) 
  (h_distinct : ∃ (x y : ℝ), x ≠ y ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0) :
  ∃! x : ℝ, (x^2 + b*x + c) + ((x - Real.sqrt (b^2 - 4*c))^2 + b*(x - Real.sqrt (b^2 - 4*c)) + c) = 0 :=
sorry

end unique_root_of_sum_with_shift_l1703_170349


namespace count_words_with_e_l1703_170388

/-- The number of letters in our alphabet -/
def n : ℕ := 5

/-- The length of the words we're creating -/
def k : ℕ := 4

/-- The number of letters in our alphabet excluding E -/
def m : ℕ := 4

/-- The number of 4-letter words that can be made from 5 letters (A, B, C, D, E) with repetition allowed -/
def total_words : ℕ := n ^ k

/-- The number of 4-letter words that can be made from 4 letters (A, B, C, D) with repetition allowed -/
def words_without_e : ℕ := m ^ k

/-- The number of 4-letter words that can be made from 5 letters (A, B, C, D, E) with repetition allowed and using E at least once -/
def words_with_e : ℕ := total_words - words_without_e

theorem count_words_with_e : words_with_e = 369 := by
  sorry

end count_words_with_e_l1703_170388


namespace third_team_pies_l1703_170325

/-- Given a catering job requiring 750 mini meat pies to be made by 3 teams,
    where the first team made 235 pies and the second team made 275 pies,
    prove that the third team should make 240 pies. -/
theorem third_team_pies (total : ℕ) (teams : ℕ) (first : ℕ) (second : ℕ) 
    (h1 : total = 750)
    (h2 : teams = 3)
    (h3 : first = 235)
    (h4 : second = 275) :
  total - first - second = 240 := by
  sorry

end third_team_pies_l1703_170325


namespace number_problem_l1703_170309

theorem number_problem : 
  ∃ x : ℝ, 2 * x = (10 / 100) * 900 ∧ x = 45 := by sorry

end number_problem_l1703_170309


namespace quadratic_expression_value_l1703_170332

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 20) 
  (eq2 : x + 4 * y = 26) : 
  17 * x^2 + 20 * x * y + 17 * y^2 = 1076 := by
  sorry

end quadratic_expression_value_l1703_170332


namespace continuous_stripe_probability_l1703_170302

/-- Represents a cube with diagonal stripes on each face --/
structure StripedCube where
  faces : Fin 6 → Bool  -- True for one diagonal orientation, False for the other

/-- The probability of a continuous stripe loop on a cube --/
def probability_continuous_loop : ℚ :=
  2 / 64

theorem continuous_stripe_probability :
  probability_continuous_loop = 1 / 32 := by
  sorry

#check continuous_stripe_probability

end continuous_stripe_probability_l1703_170302


namespace problem_solution_l1703_170385

theorem problem_solution (x : ℝ) (h : x = -3007) :
  |(|Real.sqrt ((|x| - x)) - x| - x) - Real.sqrt (|x - x^2|)| = 3084 := by
  sorry

end problem_solution_l1703_170385


namespace purple_balls_count_l1703_170382

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  white = 50 ∧
  green = 20 ∧
  yellow = 10 ∧
  red = 17 ∧
  prob_not_red_purple = 4/5 →
  total - (white + green + yellow + red) = 3 := by
sorry

end purple_balls_count_l1703_170382


namespace area_inside_rectangle_outside_circles_l1703_170354

/-- The area of the region inside a rectangle but outside three quarter circles --/
theorem area_inside_rectangle_outside_circles (π : ℝ) :
  let rectangle_area : ℝ := 4 * 6
  let circle_e_area : ℝ := π * 2^2
  let circle_f_area : ℝ := π * 3^2
  let circle_g_area : ℝ := π * 4^2
  let quarter_circles_area : ℝ := (circle_e_area + circle_f_area + circle_g_area) / 4
  rectangle_area - quarter_circles_area = 24 - (29 * π) / 4 :=
by sorry

end area_inside_rectangle_outside_circles_l1703_170354


namespace probability_of_black_ball_l1703_170391

theorem probability_of_black_ball 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (prob_white : ℚ) :
  total_balls = 100 →
  red_balls = 45 →
  prob_white = 23/100 →
  (total_balls - red_balls - (total_balls * prob_white).floor) / total_balls = 32/100 :=
by sorry

end probability_of_black_ball_l1703_170391


namespace twelve_integer_chords_l1703_170323

/-- Represents a circle with a point inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceToCenter : ℝ

/-- Counts the number of integer-length chords through a point in a circle -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- Theorem stating that for a circle with radius 15 and a point 9 units from the center,
    there are exactly 12 integer-length chords through that point -/
theorem twelve_integer_chords :
  let c : CircleWithPoint := { radius := 15, distanceToCenter := 9 }
  countIntegerChords c = 12 := by
  sorry

end twelve_integer_chords_l1703_170323


namespace probability_of_selecting_boy_l1703_170392

/-- Given a class with 60 students where 24 are girls, the probability of selecting a boy is 0.6 -/
theorem probability_of_selecting_boy (total_students : ℕ) (num_girls : ℕ) 
  (h1 : total_students = 60) 
  (h2 : num_girls = 24) : 
  (total_students - num_girls : ℚ) / total_students = 0.6 := by
  sorry

end probability_of_selecting_boy_l1703_170392


namespace smaller_cube_weight_l1703_170371

/-- Represents the weight of a cube given its side length -/
def cube_weight (side_length : ℝ) : ℝ := sorry

theorem smaller_cube_weight :
  let small_side : ℝ := 1
  let large_side : ℝ := 2 * small_side
  let large_weight : ℝ := 56
  cube_weight small_side = 7 ∧ 
  cube_weight large_side = large_weight ∧
  cube_weight large_side = 8 * cube_weight small_side :=
by sorry

end smaller_cube_weight_l1703_170371


namespace find_b_l1703_170397

/-- The circle's equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 2*y - 2 = 0

/-- The line's equation -/
def line_eq (x y b : ℝ) : Prop := y = x + b

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (4, -1)

/-- The line bisects the circle's circumference -/
axiom bisects : ∃ b : ℝ, ∀ x y : ℝ, circle_eq x y → line_eq x y b

/-- The theorem to prove -/
theorem find_b : ∃ b : ℝ, b = -5 ∧ 
  (∀ x y : ℝ, circle_eq x y → line_eq x y b) ∧
  line_eq (circle_center.1) (circle_center.2) b :=
sorry

end find_b_l1703_170397


namespace team_ratio_is_correct_l1703_170338

/-- Represents a co-ed softball team -/
structure Team where
  total_players : ℕ
  men : ℕ
  women : ℕ
  h_total : men + women = total_players
  h_ratio : ∀ (group : ℕ), group * 3 ≤ total_players → group * 2 = women - men

/-- The specific team in the problem -/
def problem_team : Team where
  total_players := 25
  men := 8
  women := 17
  h_total := by sorry
  h_ratio := by sorry

theorem team_ratio_is_correct (team : Team) (h : team = problem_team) :
  team.men = 8 ∧ team.women = 17 := by sorry

end team_ratio_is_correct_l1703_170338


namespace farm_animals_l1703_170330

/-- Given a farm with chickens and buffalos, prove the number of chickens -/
theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) (chicken_legs : ℕ) (buffalo_legs : ℕ) 
  (h_total_animals : total_animals = 13)
  (h_total_legs : total_legs = 44)
  (h_chicken_legs : chicken_legs = 2)
  (h_buffalo_legs : buffalo_legs = 4) :
  ∃ (chickens : ℕ) (buffalos : ℕ),
    chickens + buffalos = total_animals ∧
    chickens * chicken_legs + buffalos * buffalo_legs = total_legs ∧
    chickens = 4 := by
  sorry

end farm_animals_l1703_170330


namespace x_gt_y_iff_x_minus_y_plus_sin_gt_zero_l1703_170366

theorem x_gt_y_iff_x_minus_y_plus_sin_gt_zero (x y : ℝ) :
  x > y ↔ x - y + Real.sin (x - y) > 0 := by sorry

end x_gt_y_iff_x_minus_y_plus_sin_gt_zero_l1703_170366


namespace square_area_from_vertices_l1703_170326

/-- The area of a square with adjacent vertices at (1,3) and (5,6) is 25 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (5, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by sorry

end square_area_from_vertices_l1703_170326


namespace polynomial_factorization_l1703_170304

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 6*x + 5) + (x^2 + 3*x - 18) = (x^2 + 10*x + 64) * (x^2 + 10*x - 1) := by
  sorry

end polynomial_factorization_l1703_170304


namespace prime_property_l1703_170383

theorem prime_property (p : ℕ) : 
  Prime p → (∃ q : ℕ, Prime q ∧ q = 2^(p+1) + p^3 - p^2 - p) → p = 3 :=
by sorry

end prime_property_l1703_170383


namespace interview_scores_properties_l1703_170321

def scores : List ℝ := [70, 85, 86, 88, 90, 90, 92, 94, 95, 100]

def sixtieth_percentile (l : List ℝ) : ℝ := sorry

def average (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def remove_extremes (l : List ℝ) : List ℝ := sorry

theorem interview_scores_properties :
  let s := scores
  let s_without_extremes := remove_extremes s
  (sixtieth_percentile s = 91) ∧
  (average s_without_extremes > average s) ∧
  (variance s_without_extremes < variance s) ∧
  (¬ (9 / 45 = 1 / 10)) ∧
  (¬ (average s > median s)) := by sorry

end interview_scores_properties_l1703_170321


namespace solve_for_c_l1703_170399

theorem solve_for_c (h1 : ∀ a b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
                    (h2 : 6 * 15 * c = 4) : c = 2 := by
  sorry

end solve_for_c_l1703_170399


namespace greater_than_theorem_l1703_170387

theorem greater_than_theorem (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : (a - b) * (b - c) * (c - a) > 0) : 
  a > c := by sorry

end greater_than_theorem_l1703_170387


namespace wire_length_for_square_field_l1703_170347

-- Define the area of the square field
def field_area : ℝ := 53824

-- Define the number of times the wire goes around the field
def num_rounds : ℕ := 10

-- Theorem statement
theorem wire_length_for_square_field :
  ∃ (side_length : ℝ),
    side_length * side_length = field_area ∧
    (4 * side_length * num_rounds : ℝ) = 9280 :=
by sorry

end wire_length_for_square_field_l1703_170347


namespace max_lcm_20_and_others_l1703_170398

theorem max_lcm_20_and_others : 
  let lcm_list := [Nat.lcm 20 2, Nat.lcm 20 4, Nat.lcm 20 6, Nat.lcm 20 8, Nat.lcm 20 10, Nat.lcm 20 12]
  List.maximum lcm_list = some 60 := by sorry

end max_lcm_20_and_others_l1703_170398


namespace largest_power_of_two_dividing_expression_l1703_170390

theorem largest_power_of_two_dividing_expression : ∃ k : ℕ, 
  (2^k : ℤ) ∣ (15^4 - 7^4 - 8) ∧ 
  ∀ m : ℕ, (2^m : ℤ) ∣ (15^4 - 7^4 - 8) → m ≤ k ∧
  k = 3 := by
  sorry

end largest_power_of_two_dividing_expression_l1703_170390


namespace max_servings_emily_l1703_170313

/-- Represents the recipe requirements for 4 servings --/
structure Recipe :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (berries : ℕ)
  (almond_milk : ℕ)

/-- Represents Emily's available ingredients --/
structure Available :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (berries : ℕ)
  (almond_milk : ℕ)

/-- Calculates the maximum number of servings possible --/
def max_servings (recipe : Recipe) (available : Available) : ℕ :=
  min
    (available.bananas * 4 / recipe.bananas)
    (min
      (available.yogurt * 4 / recipe.yogurt)
      (min
        (available.berries * 4 / recipe.berries)
        (available.almond_milk * 4 / recipe.almond_milk)))

/-- The theorem to be proved --/
theorem max_servings_emily :
  let recipe := Recipe.mk 3 2 1 1
  let available := Available.mk 9 5 3 4
  max_servings recipe available = 10 := by
  sorry

end max_servings_emily_l1703_170313


namespace anthony_pencils_l1703_170333

/-- The number of pencils Anthony has after giving some to Kathryn -/
def pencils_remaining (initial : Float) (given : Float) : Float :=
  initial - given

/-- Theorem: Anthony has 47.0 pencils after giving some to Kathryn -/
theorem anthony_pencils :
  pencils_remaining 56.0 9.0 = 47.0 := by
  sorry

end anthony_pencils_l1703_170333


namespace days_of_sending_roses_l1703_170381

def roses_per_day : ℕ := 24  -- 2 dozen roses per day
def total_roses : ℕ := 168   -- total number of roses sent

theorem days_of_sending_roses : 
  total_roses / roses_per_day = 7 :=
sorry

end days_of_sending_roses_l1703_170381


namespace no_solutions_for_cos_and_odd_multiples_of_90_l1703_170329

theorem no_solutions_for_cos_and_odd_multiples_of_90 :
  ¬ ∃ x : ℝ, 0 ≤ x ∧ x < 720 ∧ Real.cos (x * π / 180) = -0.6 ∧ ∃ n : ℕ, x = (2 * n + 1) * 90 :=
by sorry

end no_solutions_for_cos_and_odd_multiples_of_90_l1703_170329
