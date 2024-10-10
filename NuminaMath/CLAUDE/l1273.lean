import Mathlib

namespace six_digit_numbers_with_zero_l1273_127320

theorem six_digit_numbers_with_zero (total_six_digit : ℕ) (no_zero_six_digit : ℕ) :
  total_six_digit = 900000 →
  no_zero_six_digit = 531441 →
  total_six_digit - no_zero_six_digit = 368559 :=
by sorry

end six_digit_numbers_with_zero_l1273_127320


namespace M_lower_bound_l1273_127357

theorem M_lower_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by sorry

end M_lower_bound_l1273_127357


namespace jerry_money_left_l1273_127337

-- Define the quantities and prices
def mustard_oil_quantity : ℝ := 2
def mustard_oil_price : ℝ := 13
def pasta_quantity : ℝ := 3
def pasta_price : ℝ := 4
def sauce_quantity : ℝ := 1
def sauce_price : ℝ := 5
def initial_money : ℝ := 50

-- Define the total cost of groceries
def total_cost : ℝ :=
  mustard_oil_quantity * mustard_oil_price +
  pasta_quantity * pasta_price +
  sauce_quantity * sauce_price

-- Define the money left after shopping
def money_left : ℝ := initial_money - total_cost

-- Theorem statement
theorem jerry_money_left : money_left = 7 := by
  sorry

end jerry_money_left_l1273_127337


namespace candidate_votes_l1273_127387

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) : 
  total_votes = 560000 →
  invalid_percent = 15/100 →
  candidate_percent = 75/100 →
  ↑⌊(total_votes : ℚ) * (1 - invalid_percent) * candidate_percent⌋ = 357000 := by
sorry

end candidate_votes_l1273_127387


namespace perpendicular_line_proof_l1273_127363

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x - 6 * y = 9

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := y = -1/2 * x - 2

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, -3)

-- Theorem statement
theorem perpendicular_line_proof :
  -- The perpendicular line passes through the given point
  perp_line point.1 point.2 ∧
  -- The two lines are perpendicular
  (∀ x₁ y₁ x₂ y₂ : ℝ, given_line x₁ y₁ → perp_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((y₂ - y₁) / (x₂ - x₁)) * ((y₁ - y₂) / (x₁ - x₂)) = -1) :=
by
  sorry

end perpendicular_line_proof_l1273_127363


namespace largest_multiple_of_9_less_than_110_l1273_127353

theorem largest_multiple_of_9_less_than_110 : 
  ∀ n : ℕ, n % 9 = 0 → n < 110 → n ≤ 108 :=
by
  sorry

end largest_multiple_of_9_less_than_110_l1273_127353


namespace five_player_tournament_games_l1273_127348

/-- The number of games in a tournament where each player plays every other player once -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 5 players, where each player plays against every other player
    exactly once, the total number of games is 10 -/
theorem five_player_tournament_games :
  tournament_games 5 = 10 := by
  sorry

end five_player_tournament_games_l1273_127348


namespace roses_in_vase_after_actions_l1273_127370

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  roses : ℕ
  orchids : ℕ

/-- Represents the actions taken by Jessica -/
structure JessicaActions where
  addedRoses : ℕ
  addedOrchids : ℕ
  cutRoses : ℕ

def initial : FlowerVase := { roses := 15, orchids := 62 }

def actions : JessicaActions := { addedRoses := 0, addedOrchids := 34, cutRoses := 2 }

def final : FlowerVase := { roses := 96, orchids := initial.orchids + actions.addedOrchids }

theorem roses_in_vase_after_actions (R : ℕ) : 
  final.roses = 13 + R ↔ actions.addedRoses = R := by sorry

end roses_in_vase_after_actions_l1273_127370


namespace system_equation_ratio_l1273_127327

theorem system_equation_ratio (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - 2 * z = 0)
  (eq2 : x + 4 * y - 10 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 4*x*y) / (y^2 + z^2) = 96/13 := by sorry

end system_equation_ratio_l1273_127327


namespace percent_of_percent_l1273_127312

theorem percent_of_percent (y : ℝ) (hy : y ≠ 0) :
  (18 / 100) * y = (30 / 100) * ((60 / 100) * y) :=
by sorry

end percent_of_percent_l1273_127312


namespace rectangle_perimeter_bound_l1273_127356

theorem rectangle_perimeter_bound (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (area_gt_perimeter : a * b > 2 * (a + b)) : 2 * (a + b) > 16 := by
  sorry

end rectangle_perimeter_bound_l1273_127356


namespace unique_solution_complex_magnitude_one_l1273_127328

/-- There exists exactly one real value of x that satisfies |1 - (x/2)i| = 1 -/
theorem unique_solution_complex_magnitude_one :
  ∃! x : ℝ, Complex.abs (1 - (x / 2) * Complex.I) = 1 := by
  sorry

end unique_solution_complex_magnitude_one_l1273_127328


namespace white_marbles_count_l1273_127334

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 20 →
  blue = 6 →
  red = 9 →
  prob_red_or_white = 7/10 →
  total - blue - red = 5 := by
  sorry

end white_marbles_count_l1273_127334


namespace num_divisors_not_div_by_3_eq_8_l1273_127339

/-- The number of positive divisors of 210 that are not divisible by 3 -/
def num_divisors_not_div_by_3 : ℕ :=
  (Finset.filter (fun d => d ∣ 210 ∧ ¬(3 ∣ d)) (Finset.range 211)).card

/-- Theorem: The number of positive divisors of 210 that are not divisible by 3 is 8 -/
theorem num_divisors_not_div_by_3_eq_8 : num_divisors_not_div_by_3 = 8 := by
  sorry

end num_divisors_not_div_by_3_eq_8_l1273_127339


namespace phd_total_time_l1273_127398

def phd_timeline (acclimation_time : ℝ) (basics_time : ℝ) (research_factor : ℝ) (dissertation_factor : ℝ) : ℝ :=
  let research_time := basics_time * (1 + research_factor)
  let dissertation_time := acclimation_time * dissertation_factor
  acclimation_time + basics_time + research_time + dissertation_time

theorem phd_total_time :
  phd_timeline 1 2 0.75 0.5 = 7 := by
  sorry

end phd_total_time_l1273_127398


namespace remainder_x5_plus_3_div_x_minus_3_squared_l1273_127330

open Polynomial

theorem remainder_x5_plus_3_div_x_minus_3_squared (x : ℝ) :
  ∃ q : Polynomial ℝ, X^5 + C 3 = (X - C 3)^2 * q + (C 405 * X - C 969) := by
  sorry

end remainder_x5_plus_3_div_x_minus_3_squared_l1273_127330


namespace pirate_coin_problem_l1273_127379

def coin_distribution (x : ℕ) : Prop :=
  let paul_coins := x
  let pete_coins := x * (x + 1) / 2
  pete_coins = 5 * paul_coins ∧ 
  paul_coins + pete_coins = 54

theorem pirate_coin_problem :
  ∃ x : ℕ, coin_distribution x :=
sorry

end pirate_coin_problem_l1273_127379


namespace solution_satisfies_system_l1273_127389

/-- The system of linear equations -/
def system (x₁ x₂ x₃ : ℝ) : Prop :=
  x₁ + 2*x₂ + 4*x₃ = 5 ∧
  2*x₁ + x₂ + 5*x₃ = 7 ∧
  3*x₁ + 2*x₂ + 6*x₃ = 9

/-- The solution satisfies the system of equations -/
theorem solution_satisfies_system :
  system 1 0 1 := by sorry

end solution_satisfies_system_l1273_127389


namespace roots_product_minus_one_l1273_127395

theorem roots_product_minus_one (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d - 1) * (e - 1) = 1 := by
sorry

end roots_product_minus_one_l1273_127395


namespace arrangement_theorem_l1273_127329

def number_of_people : ℕ := 6
def people_per_row : ℕ := 3

def arrangement_count : ℕ := 216

theorem arrangement_theorem :
  let total_arrangements := number_of_people.factorial
  let front_row_without_A := (people_per_row - 1).choose 1
  let back_row_without_B := (people_per_row - 1).choose 1
  let remaining_arrangements := (number_of_people - 2).factorial
  front_row_without_A * back_row_without_B * remaining_arrangements = arrangement_count :=
by sorry

end arrangement_theorem_l1273_127329


namespace crayons_lost_or_given_away_l1273_127308

theorem crayons_lost_or_given_away (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 606)
  (h2 : remaining_crayons = 291) :
  initial_crayons - remaining_crayons = 315 := by
  sorry

end crayons_lost_or_given_away_l1273_127308


namespace flower_shop_purchase_l1273_127381

theorem flower_shop_purchase 
  (total_flowers : ℕ) 
  (total_cost : ℚ) 
  (carnation_price : ℚ) 
  (rose_price : ℚ) 
  (h1 : total_flowers = 400)
  (h2 : total_cost = 1020)
  (h3 : carnation_price = 6/5)  -- $1.2 as a rational number
  (h4 : rose_price = 3) :
  ∃ (carnations roses : ℕ),
    carnations + roses = total_flowers ∧
    carnation_price * carnations + rose_price * roses = total_cost ∧
    carnations = 100 ∧
    roses = 300 := by
  sorry

end flower_shop_purchase_l1273_127381


namespace fifth_individual_is_one_l1273_127301

def random_numbers : List ℕ := [65, 72, 08, 02, 63, 14, 07, 02, 43, 69, 97, 08, 01]

def is_valid (n : ℕ) : Bool := n ≥ 1 ∧ n ≤ 20

def select_individuals (numbers : List ℕ) : List ℕ :=
  numbers.filter is_valid |>.eraseDups

theorem fifth_individual_is_one :
  (select_individuals random_numbers).nthLe 4 sorry = 1 := by
  sorry

end fifth_individual_is_one_l1273_127301


namespace egypt_promotion_theorem_l1273_127343

/-- The number of tourists who went to Egypt for free -/
def free_tourists : ℕ := 29

/-- The number of tourists who came on their own -/
def solo_tourists : ℕ := 13

/-- The number of tourists who did not bring anyone -/
def no_referral_tourists : ℕ := 100

theorem egypt_promotion_theorem :
  ∃ (total_tourists : ℕ),
    total_tourists = solo_tourists + 4 * free_tourists ∧
    total_tourists = free_tourists + no_referral_tourists ∧
    free_tourists = 29 := by
  sorry

end egypt_promotion_theorem_l1273_127343


namespace sara_remaining_pears_l1273_127393

def remaining_pears (initial : ℕ) (given_to_dan : ℕ) (given_to_monica : ℕ) (given_to_jenny : ℕ) : ℕ :=
  initial - given_to_dan - given_to_monica - given_to_jenny

theorem sara_remaining_pears :
  remaining_pears 35 28 4 1 = 2 := by
  sorry

end sara_remaining_pears_l1273_127393


namespace fraction_simplification_l1273_127399

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end fraction_simplification_l1273_127399


namespace circle_symmetry_line_l1273_127368

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line equation
def line_eq (a x y : ℝ) : Prop := a*x + y + 1 = 0

-- Define symmetry condition
def is_symmetric (a : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq a (-1) 2

-- Theorem statement
theorem circle_symmetry_line (a : ℝ) :
  is_symmetric a → a = 3 := by sorry

end circle_symmetry_line_l1273_127368


namespace division_remainder_and_divisibility_l1273_127369

theorem division_remainder_and_divisibility : 
  let dividend : ℕ := 1234567
  let divisor : ℕ := 256
  let remainder : ℕ := dividend % divisor
  remainder = 933 ∧ ¬(∃ k : ℕ, remainder = 7 * k) := by
  sorry

end division_remainder_and_divisibility_l1273_127369


namespace max_factors_bound_l1273_127382

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum number of factors for a^m where 1 ≤ a ≤ 20 and 1 ≤ m ≤ 10 is 231 -/
theorem max_factors_bound :
  ∀ a m : ℕ, 1 ≤ a → a ≤ 20 → 1 ≤ m → m ≤ 10 → num_factors (a^m) ≤ 231 := by
  sorry

end max_factors_bound_l1273_127382


namespace quadratic_inequality_range_l1273_127390

theorem quadratic_inequality_range (x : ℝ) : x^2 + 3*x - 10 < 0 ↔ -5 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_range_l1273_127390


namespace donnelly_class_size_l1273_127317

/-- The number of cupcakes Quinton brought to school -/
def total_cupcakes : ℕ := 40

/-- The number of students in Ms. Delmont's class -/
def delmont_students : ℕ := 18

/-- The number of staff members who received cupcakes -/
def staff_members : ℕ := 4

/-- The number of cupcakes left over -/
def leftover_cupcakes : ℕ := 2

/-- The number of students in Mrs. Donnelly's class -/
def donnelly_students : ℕ := total_cupcakes - delmont_students - staff_members - leftover_cupcakes

theorem donnelly_class_size : donnelly_students = 16 := by
  sorry

end donnelly_class_size_l1273_127317


namespace emmas_room_length_l1273_127362

/-- The length of Emma's room, given the width, tiled area, and fraction of room tiled. -/
theorem emmas_room_length (width : ℝ) (tiled_area : ℝ) (tiled_fraction : ℝ) :
  width = 12 →
  tiled_area = 40 →
  tiled_fraction = 1/6 →
  ∃ length : ℝ, length = 20 ∧ tiled_area = tiled_fraction * (width * length) := by
  sorry

end emmas_room_length_l1273_127362


namespace ticket_cost_difference_l1273_127375

theorem ticket_cost_difference : 
  let num_adults : ℕ := 9
  let num_children : ℕ := 7
  let adult_ticket_price : ℕ := 11
  let child_ticket_price : ℕ := 7
  let adult_total_cost := num_adults * adult_ticket_price
  let child_total_cost := num_children * child_ticket_price
  adult_total_cost - child_total_cost = 50 := by
sorry

end ticket_cost_difference_l1273_127375


namespace lives_gained_l1273_127349

theorem lives_gained (initial_lives lost_lives final_lives : ℕ) :
  initial_lives = 14 →
  lost_lives = 4 →
  final_lives = 46 →
  final_lives - (initial_lives - lost_lives) = 36 := by
sorry

end lives_gained_l1273_127349


namespace arctan_sum_equals_pi_over_two_l1273_127332

theorem arctan_sum_equals_pi_over_two (y : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/10) + Real.arctan (1/30) + Real.arctan (1/y) = π/2 →
  y = 547/620 := by
  sorry

end arctan_sum_equals_pi_over_two_l1273_127332


namespace equation_solution_l1273_127373

theorem equation_solution : ∃ t : ℝ, t = 1.5 ∧ 4 * (4 : ℝ)^t + Real.sqrt (16 * 16^t) = 40 := by
  sorry

end equation_solution_l1273_127373


namespace problem_statement_l1273_127322

theorem problem_statement (a b : ℝ) (h1 : a = 2 + Real.sqrt 3) (h2 : b = 2 - Real.sqrt 3) :
  a^2 + 2*a*b - b*(3*a - b) = 13 := by sorry

end problem_statement_l1273_127322


namespace kristy_cookies_theorem_l1273_127354

/-- The number of cookies Kristy baked -/
def total_cookies : ℕ := 22

/-- The number of cookies Kristy ate -/
def cookies_eaten : ℕ := 2

/-- The number of cookies Kristy gave to her brother -/
def cookies_given_to_brother : ℕ := 1

/-- The number of cookies taken by the first friend -/
def cookies_taken_by_first_friend : ℕ := 3

/-- The number of cookies taken by the second friend -/
def cookies_taken_by_second_friend : ℕ := 5

/-- The number of cookies taken by the third friend -/
def cookies_taken_by_third_friend : ℕ := 5

/-- The number of cookies left -/
def cookies_left : ℕ := 6

/-- Theorem stating that the total number of cookies equals the sum of all distributed cookies and those left -/
theorem kristy_cookies_theorem : 
  total_cookies = 
    cookies_eaten + 
    cookies_given_to_brother + 
    cookies_taken_by_first_friend + 
    cookies_taken_by_second_friend + 
    cookies_taken_by_third_friend + 
    cookies_left :=
by
  sorry

end kristy_cookies_theorem_l1273_127354


namespace arithmetic_sequence_50th_term_l1273_127383

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) : ℕ → ℤ := fun n => a + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
def term50 : ℤ := arithmeticSequence 3 2 50

/-- Theorem: The 50th term of the arithmetic sequence with first term 3 and common difference 2 is 101 -/
theorem arithmetic_sequence_50th_term : term50 = 101 := by
  sorry

end arithmetic_sequence_50th_term_l1273_127383


namespace smallest_number_divisible_by_all_l1273_127306

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 6) % 12 = 0 ∧
  (n - 6) % 16 = 0 ∧
  (n - 6) % 18 = 0 ∧
  (n - 6) % 21 = 0 ∧
  (n - 6) % 28 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 1014 ∧
  ∀ m : ℕ, m < 1014 → ¬is_divisible_by_all m :=
by sorry

end smallest_number_divisible_by_all_l1273_127306


namespace father_son_age_ratio_l1273_127378

def father_son_ages (son_age : ℕ) (age_difference : ℕ) : Prop :=
  let father_age : ℕ := son_age + age_difference
  let son_age_in_two_years : ℕ := son_age + 2
  let father_age_in_two_years : ℕ := father_age + 2
  (father_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2

theorem father_son_age_ratio :
  father_son_ages 33 35 := by sorry

end father_son_age_ratio_l1273_127378


namespace smallest_s_value_l1273_127361

theorem smallest_s_value : ∃ s : ℚ, s = 4/7 ∧ 
  (∀ t : ℚ, (15*t^2 - 40*t + 18) / (4*t - 3) + 7*t = 9*t - 2 → s ≤ t) ∧ 
  (15*s^2 - 40*s + 18) / (4*s - 3) + 7*s = 9*s - 2 := by
  sorry

end smallest_s_value_l1273_127361


namespace expression_evaluation_l1273_127344

theorem expression_evaluation :
  ∀ (a b c d : ℝ),
    d = c + 1 →
    c = b - 8 →
    b = a + 4 →
    a = 7 →
    a + 3 ≠ 0 →
    b - 3 ≠ 0 →
    c + 10 ≠ 0 →
    d + 1 ≠ 0 →
    ((a + 5) / (a + 3)) * ((b - 2) / (b - 3)) * ((c + 7) / (c + 10)) * ((d - 4) / (d + 1)) = 0 := by
  sorry

end expression_evaluation_l1273_127344


namespace volume_of_region_l1273_127321

-- Define the region
def Region := {p : ℝ × ℝ × ℝ | 
  let (x, y, z) := p
  (|x - y + z| + |x - y - z| ≤ 12) ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0}

-- State the theorem
theorem volume_of_region : MeasureTheory.volume Region = 108 := by
  sorry

end volume_of_region_l1273_127321


namespace edward_baseball_cards_l1273_127300

/-- The number of binders Edward has -/
def num_binders : ℕ := 7

/-- The number of cards in each binder -/
def cards_per_binder : ℕ := 109

/-- The total number of baseball cards Edward has -/
def total_cards : ℕ := num_binders * cards_per_binder

theorem edward_baseball_cards : total_cards = 763 := by
  sorry

end edward_baseball_cards_l1273_127300


namespace sqrt_16_div_2_l1273_127335

theorem sqrt_16_div_2 : Real.sqrt 16 / 2 = 2 := by sorry

end sqrt_16_div_2_l1273_127335


namespace intersection_of_A_and_B_l1273_127355

def A : Set ℤ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℤ := {-2, -1, 0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 2, 3} := by
  sorry

end intersection_of_A_and_B_l1273_127355


namespace lisa_quiz_goal_l1273_127358

theorem lisa_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (current_as : ℕ) : 
  total_quizzes = 60 →
  goal_percentage = 3/4 →
  completed_quizzes = 40 →
  current_as = 26 →
  ∃ (max_non_as : ℕ), 
    max_non_as = 1 ∧
    (current_as + (total_quizzes - completed_quizzes - max_non_as) : ℚ) / total_quizzes ≥ goal_percentage ∧
    ∀ (n : ℕ), n > max_non_as →
      (current_as + (total_quizzes - completed_quizzes - n) : ℚ) / total_quizzes < goal_percentage :=
by sorry

end lisa_quiz_goal_l1273_127358


namespace negation_of_forall_exp_minus_x_minus_one_geq_zero_l1273_127336

theorem negation_of_forall_exp_minus_x_minus_one_geq_zero :
  (¬ ∀ x : ℝ, Real.exp x - x - 1 ≥ 0) ↔ (∃ x : ℝ, Real.exp x - x - 1 < 0) :=
sorry

end negation_of_forall_exp_minus_x_minus_one_geq_zero_l1273_127336


namespace greatest_five_digit_divisible_by_12_15_18_l1273_127372

theorem greatest_five_digit_divisible_by_12_15_18 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ 12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n → n ≤ 99900 := by
  sorry

#check greatest_five_digit_divisible_by_12_15_18

end greatest_five_digit_divisible_by_12_15_18_l1273_127372


namespace inequality_proof_l1273_127371

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h1 : a ≤ 2 * b) (h2 : 2 * b ≤ 4 * a) :
  4 * a * b ≤ 2 * (a^2 + b^2) ∧ 2 * (a^2 + b^2) ≤ 5 * a * b := by
  sorry

end inequality_proof_l1273_127371


namespace function_inequality_implies_a_bound_l1273_127367

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc 0 1, ∃ x₂ ∈ Set.Icc 1 2, 
    (x₁ - 1 / (x₁ + 1)) ≥ (x₂^2 - 2*a*x₂ + 4)) → 
  a ≥ 9/4 := by
sorry

end function_inequality_implies_a_bound_l1273_127367


namespace f_equals_g_l1273_127341

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := (x^2 - 1)^(1/3)

-- State the theorem
theorem f_equals_g : f = g := by sorry

end f_equals_g_l1273_127341


namespace complex_number_in_second_quadrant_l1273_127333

theorem complex_number_in_second_quadrant : ∃ (z : ℂ), 
  z = (1 + 2*I) - (3 - 4*I) ∧ 
  (z.re < 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_in_second_quadrant_l1273_127333


namespace running_to_basketball_ratio_l1273_127385

def trumpet_time : ℕ := 40

theorem running_to_basketball_ratio :
  let running_time := trumpet_time / 2
  let basketball_time := running_time + trumpet_time
  (running_time : ℚ) / basketball_time = 1 / 3 := by sorry

end running_to_basketball_ratio_l1273_127385


namespace z_real_z_pure_imaginary_z_second_quadrant_l1273_127360

/-- Definition of the complex number z in terms of real number m -/
def z (m : ℝ) : ℂ := (m^2 - 2*m - 3 : ℝ) + (m^2 + 3*m + 2 : ℝ) * Complex.I

/-- z is a real number if and only if m = -1 or m = -2 -/
theorem z_real (m : ℝ) : (z m).im = 0 ↔ m = -1 ∨ m = -2 := by sorry

/-- z is a pure imaginary number if and only if m = 3 -/
theorem z_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 3 := by sorry

/-- z is in the second quadrant of the complex plane if and only if -1 < m < 3 -/
theorem z_second_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ -1 < m ∧ m < 3 := by sorry

end z_real_z_pure_imaginary_z_second_quadrant_l1273_127360


namespace quadratic_roots_sum_minus_product_l1273_127388

theorem quadratic_roots_sum_minus_product (a b : ℝ) : 
  a^2 - 3*a + 1 = 0 → b^2 - 3*b + 1 = 0 → a + b - a*b = 2 := by
  sorry

end quadratic_roots_sum_minus_product_l1273_127388


namespace andrew_flooring_planks_l1273_127391

/-- The number of planks Andrew bought for his flooring project -/
def total_planks : ℕ := 65

/-- The number of planks used in Andrew's bedroom -/
def bedroom_planks : ℕ := 8

/-- The number of planks used in the living room -/
def living_room_planks : ℕ := 20

/-- The number of planks used in the kitchen -/
def kitchen_planks : ℕ := 11

/-- The number of planks used in the guest bedroom -/
def guest_bedroom_planks : ℕ := bedroom_planks - 2

/-- The number of planks used in each hallway -/
def hallway_planks : ℕ := 4

/-- The number of planks ruined and replaced in each bedroom -/
def ruined_planks_per_bedroom : ℕ := 3

/-- The number of leftover planks -/
def leftover_planks : ℕ := 6

/-- The number of hallways -/
def num_hallways : ℕ := 2

/-- The number of bedrooms -/
def num_bedrooms : ℕ := 2

theorem andrew_flooring_planks :
  total_planks = 
    bedroom_planks + living_room_planks + kitchen_planks + guest_bedroom_planks + 
    (num_hallways * hallway_planks) + (num_bedrooms * ruined_planks_per_bedroom) + 
    leftover_planks :=
by sorry

end andrew_flooring_planks_l1273_127391


namespace stating_time_for_one_click_approx_10_seconds_l1273_127310

/-- Represents the length of a rail in feet -/
def rail_length : ℝ := 15

/-- Represents the number of feet in a mile -/
def feet_per_mile : ℝ := 5280

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℝ := 60

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℝ := 60

/-- 
Theorem stating that the time taken to hear one click (passing over one rail joint) 
is approximately 10 seconds for a train traveling at any speed.
-/
theorem time_for_one_click_approx_10_seconds (train_speed : ℝ) : 
  train_speed > 0 → 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    |((rail_length * minutes_per_hour) / (train_speed * feet_per_mile)) * seconds_per_minute - 10| < ε :=
sorry

end stating_time_for_one_click_approx_10_seconds_l1273_127310


namespace florist_bouquet_problem_l1273_127318

theorem florist_bouquet_problem (narcissus : ℕ) (chrysanthemums : ℕ) (total_bouquets : ℕ) :
  narcissus = 75 →
  chrysanthemums = 90 →
  total_bouquets = 33 →
  (narcissus + chrysanthemums) % total_bouquets = 0 →
  (narcissus + chrysanthemums) / total_bouquets = 5 :=
by sorry

end florist_bouquet_problem_l1273_127318


namespace expand_binomials_l1273_127350

theorem expand_binomials (x : ℝ) : (2*x - 3) * (4*x + 5) = 8*x^2 - 2*x - 15 := by
  sorry

end expand_binomials_l1273_127350


namespace consecutive_odd_squares_sum_l1273_127374

theorem consecutive_odd_squares_sum (k : ℤ) (n : ℕ) :
  (2 * k - 1)^2 + (2 * k + 1)^2 = n * (n + 1) / 2 ↔ k = 1 ∧ n = 4 := by
  sorry

end consecutive_odd_squares_sum_l1273_127374


namespace two_distinct_roots_characterization_l1273_127392

noncomputable def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + x * |x| = 2 * (3 + a * x - 2 * a)) ∧
  (y^2 + y * |y| = 2 * (3 + a * y - 2 * a))

theorem two_distinct_roots_characterization (a : ℝ) :
  has_two_distinct_roots a ↔ 
  ((3/4 ≤ a ∧ a < 1) ∨ (a > 3)) ∨ (0 < a ∧ a < 3/4) :=
sorry

end two_distinct_roots_characterization_l1273_127392


namespace special_quadratic_relation_l1273_127331

theorem special_quadratic_relation (q a b : ℕ) (h : a^2 - q*a*b + b^2 = q) :
  ∃ (c : ℤ), c ≠ a ∧ c^2 - q*b*c + b^2 = q ∧ ∃ (k : ℕ), q = k^2 := by
  sorry

end special_quadratic_relation_l1273_127331


namespace identity_is_unique_strictly_increasing_double_application_less_than_successor_l1273_127365

-- Define a strictly increasing function from ℕ to ℕ
def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem identity_is_unique_strictly_increasing_double_application_less_than_successor
  (f : ℕ → ℕ)
  (h_increasing : StrictlyIncreasing f)
  (h_condition : ∀ n, f (f n) < n + 1) :
  ∀ n, f n = n :=
by sorry

end identity_is_unique_strictly_increasing_double_application_less_than_successor_l1273_127365


namespace men_to_women_ratio_l1273_127345

/-- Proves that the ratio of men to women is 2:1 given the average heights -/
theorem men_to_women_ratio (M W : ℕ) (h_total : M * 185 + W * 170 = (M + W) * 180) :
  M / W = 2 / 1 := by
  sorry

#check men_to_women_ratio

end men_to_women_ratio_l1273_127345


namespace raspberry_harvest_calculation_l1273_127302

/-- Calculates the expected raspberry harvest given garden dimensions and planting parameters. -/
theorem raspberry_harvest_calculation 
  (length width : ℕ) 
  (plants_per_sqft : ℕ) 
  (raspberries_per_plant : ℕ) : 
  length = 10 → 
  width = 7 → 
  plants_per_sqft = 5 → 
  raspberries_per_plant = 12 → 
  length * width * plants_per_sqft * raspberries_per_plant = 4200 :=
by sorry

end raspberry_harvest_calculation_l1273_127302


namespace minimum_value_implies_ratio_l1273_127346

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem minimum_value_implies_ratio (θ : ℝ) 
  (h : ∀ x, f x ≥ f θ) : 
  (Real.sin (2 * θ) + 2 * Real.cos θ) / (Real.sin (2 * θ) - 2 * Real.cos (2 * θ)) = -1/3 := by
  sorry

end minimum_value_implies_ratio_l1273_127346


namespace area_of_four_presentable_set_l1273_127315

/-- A complex number is four-presentable if there exists a complex number w with |w| = 5 such that z = (w - 1/w) / 2 -/
def FourPresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = (w - 1 / w) / 2

/-- The set of all four-presentable complex numbers -/
def S : Set ℂ :=
  {z : ℂ | FourPresentable z}

/-- The area of the closed curve formed by S -/
noncomputable def area_S : ℝ := sorry

theorem area_of_four_presentable_set :
  area_S = 18.025 * Real.pi := by sorry

end area_of_four_presentable_set_l1273_127315


namespace intersection_not_in_first_quadrant_l1273_127316

theorem intersection_not_in_first_quadrant (m : ℝ) : 
  let x := -(m + 4) / 2
  let y := m / 2 - 2
  ¬(x > 0 ∧ y > 0) := by
sorry

end intersection_not_in_first_quadrant_l1273_127316


namespace value_range_sqrt_sum_bounds_are_tight_l1273_127324

theorem value_range_sqrt_sum (x : ℝ) : 
  ∃ (y : ℝ), y = Real.sqrt (1 + 2*x) + Real.sqrt (1 - 2*x) ∧ 
  Real.sqrt 2 ≤ y ∧ y ≤ 2 :=
sorry

theorem bounds_are_tight : 
  (∃ (x : ℝ), Real.sqrt (1 + 2*x) + Real.sqrt (1 - 2*x) = Real.sqrt 2) ∧
  (∃ (x : ℝ), Real.sqrt (1 + 2*x) + Real.sqrt (1 - 2*x) = 2) :=
sorry

end value_range_sqrt_sum_bounds_are_tight_l1273_127324


namespace simplify_expression_l1273_127319

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (25 * x^3) * (12 * x^2) * (1 / (5 * x)^3) = 12/5 * x^2 := by
  sorry

end simplify_expression_l1273_127319


namespace only_5_12_13_is_right_triangle_l1273_127364

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def number_sets : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (4, 5, 6), (5, 12, 13), (5, 6, 7)]

/-- Theorem stating that only (5, 12, 13) forms a right triangle --/
theorem only_5_12_13_is_right_triangle :
  ∃! (a b c : ℕ), (a, b, c) ∈ number_sets ∧ is_right_triangle a b c :=
by sorry

end only_5_12_13_is_right_triangle_l1273_127364


namespace car_speed_l1273_127384

/-- Theorem: Given a car traveling for 5 hours and covering a distance of 800 km, its speed is 160 km/hour. -/
theorem car_speed (time : ℝ) (distance : ℝ) (speed : ℝ) : 
  time = 5 → distance = 800 → speed = distance / time → speed = 160 :=
by sorry

end car_speed_l1273_127384


namespace arithmetic_sequence_fourth_quadrant_l1273_127351

def arithmetic_sequence (n : ℕ) : ℚ := 1 - (n - 1) * (1 / 2)

def intersection_x (a_n : ℚ) : ℚ := (a_n + 1) / 3

def intersection_y (a_n : ℚ) : ℚ := (8 * a_n - 1) / 3

theorem arithmetic_sequence_fourth_quadrant :
  ∀ n : ℕ, n > 0 →
  (intersection_x (arithmetic_sequence n) > 0 ∧ 
   intersection_y (arithmetic_sequence n) < 0) →
  (n = 3 ∨ n = 4) ∧ 
  arithmetic_sequence n = -1/2 * n + 3/2 := by
sorry

end arithmetic_sequence_fourth_quadrant_l1273_127351


namespace bird_cage_problem_l1273_127394

theorem bird_cage_problem (initial_birds : ℕ) (final_birds : ℕ) : 
  initial_birds = 60 → final_birds = 8 → 
  ∃ F : ℚ, 
    (1/3 : ℚ) * (2/3 : ℚ) * initial_birds * (1 - F) = final_birds ∧ 
    F = 4/5 := by
  sorry

end bird_cage_problem_l1273_127394


namespace probability_sum_less_than_12_l1273_127303

def roll_dice : ℕ := 6

def total_outcomes : ℕ := roll_dice * roll_dice

def favorable_outcomes : ℕ := total_outcomes - 1

theorem probability_sum_less_than_12 : 
  (favorable_outcomes : ℚ) / total_outcomes = 35 / 36 := by sorry

end probability_sum_less_than_12_l1273_127303


namespace committee_choice_count_l1273_127307

/-- The number of members in the club -/
def total_members : ℕ := 18

/-- The minimum tenure required for eligibility -/
def min_tenure : ℕ := 10

/-- The number of members to be chosen for the committee -/
def committee_size : ℕ := 3

/-- The number of eligible members (those with tenure ≥ 10 years) -/
def eligible_members : ℕ := total_members - min_tenure + 1

/-- The number of ways to choose the committee -/
def committee_choices : ℕ := Nat.choose eligible_members committee_size

theorem committee_choice_count :
  committee_choices = 84 := by sorry

end committee_choice_count_l1273_127307


namespace zero_sum_points_for_m_3_unique_zero_sum_point_condition_l1273_127376

/-- Definition of a "zero-sum point" in the Cartesian coordinate system -/
def is_zero_sum_point (x y : ℝ) : Prop := x + y = 0

/-- The quadratic function y = x^2 + 3x + m -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + m

theorem zero_sum_points_for_m_3 :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_zero_sum_point x₁ y₁ ∧
    is_zero_sum_point x₂ y₂ ∧
    quadratic_function 3 x₁ = y₁ ∧
    quadratic_function 3 x₂ = y₂ ∧
    x₁ = -1 ∧ y₁ = 1 ∧
    x₂ = -3 ∧ y₂ = 3 :=
sorry

theorem unique_zero_sum_point_condition (m : ℝ) :
  (∃! (x y : ℝ), is_zero_sum_point x y ∧ quadratic_function m x = y) ↔ m = 4 :=
sorry

end zero_sum_points_for_m_3_unique_zero_sum_point_condition_l1273_127376


namespace distance_before_break_l1273_127352

/-- Proves the distance walked before the break given initial, final, and total distances -/
theorem distance_before_break 
  (initial_distance : ℕ) 
  (final_distance : ℕ) 
  (total_distance : ℕ) 
  (h1 : initial_distance = 3007)
  (h2 : final_distance = 840)
  (h3 : total_distance = 6030) :
  total_distance - (initial_distance + final_distance) = 2183 := by
  sorry

#check distance_before_break

end distance_before_break_l1273_127352


namespace rectangle_area_15_20_l1273_127396

/-- The area of a rectangular field with given length and width -/
def rectangle_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular field with length 15 meters and width 20 meters is 300 square meters -/
theorem rectangle_area_15_20 :
  rectangle_area 15 20 = 300 := by
  sorry

end rectangle_area_15_20_l1273_127396


namespace probability_two_red_cards_value_l1273_127380

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 54)
  (red_cards : ℕ := 27)
  (jokers : ℕ := 2)

/-- The probability of drawing two red cards from a standard deck -/
def probability_two_red_cards (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards * (d.red_cards - 1) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing two red cards from a standard deck -/
theorem probability_two_red_cards_value (d : Deck) :
  probability_two_red_cards d = 13 / 53 := by
  sorry

end probability_two_red_cards_value_l1273_127380


namespace forgotten_lawns_l1273_127323

/-- Proves the number of forgotten lawns given Henry's lawn mowing situation -/
theorem forgotten_lawns (dollars_per_lawn : ℕ) (total_lawns : ℕ) (actual_earnings : ℕ) : 
  dollars_per_lawn = 5 → 
  total_lawns = 12 → 
  actual_earnings = 25 → 
  total_lawns - (actual_earnings / dollars_per_lawn) = 7 := by
  sorry

end forgotten_lawns_l1273_127323


namespace union_of_A_and_B_l1273_127338

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by sorry

end union_of_A_and_B_l1273_127338


namespace symmetric_line_wrt_x_axis_l1273_127340

/-- Given a line with equation 3x-4y+5=0, this theorem states that its symmetric line
    with respect to the x-axis has the equation 3x+4y+5=0 -/
theorem symmetric_line_wrt_x_axis : 
  ∀ (x y : ℝ), (3 * x - 4 * y + 5 = 0) → 
  ∃ (x' y' : ℝ), (x' = x ∧ y' = -y) ∧ (3 * x' + 4 * y' + 5 = 0) :=
sorry

end symmetric_line_wrt_x_axis_l1273_127340


namespace fudge_price_per_pound_l1273_127325

-- Define the given quantities
def total_revenue : ℚ := 212
def fudge_pounds : ℚ := 20
def truffle_dozens : ℚ := 5
def truffle_price : ℚ := 3/2  -- $1.50 as a rational number
def pretzel_dozens : ℚ := 3
def pretzel_price : ℚ := 2

-- Define the theorem
theorem fudge_price_per_pound :
  (total_revenue - (truffle_dozens * 12 * truffle_price + pretzel_dozens * 12 * pretzel_price)) / fudge_pounds = 5/2 := by
  sorry

end fudge_price_per_pound_l1273_127325


namespace sum_interior_angles_num_diagonals_l1273_127347

/-- A regular polygon with exterior angles measuring 20° -/
structure RegularPolygon20 where
  n : ℕ
  exterior_angle : ℝ
  h_exterior : exterior_angle = 20

/-- The sum of interior angles of a regular polygon with 20° exterior angles is 2880° -/
theorem sum_interior_angles (p : RegularPolygon20) : 
  (p.n - 2) * 180 = 2880 := by sorry

/-- The number of diagonals in a regular polygon with 20° exterior angles is 135 -/
theorem num_diagonals (p : RegularPolygon20) : 
  p.n * (p.n - 3) / 2 = 135 := by sorry

end sum_interior_angles_num_diagonals_l1273_127347


namespace no_profit_after_ten_requests_l1273_127342

def genie_operation (x : ℕ) : ℕ := (x + 1000) / 2

def iterate_genie (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | m + 1 => genie_operation (iterate_genie m x)

theorem no_profit_after_ten_requests (x : ℕ) : iterate_genie 10 x ≤ x := by
  sorry


end no_profit_after_ten_requests_l1273_127342


namespace hyperbola_eccentricity_l1273_127386

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let e := Real.sqrt (1 + b^2 / a^2)
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → e = 5/3 :=
by sorry

end hyperbola_eccentricity_l1273_127386


namespace square_root_of_625_l1273_127314

theorem square_root_of_625 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 625) : x = 25 := by
  sorry

end square_root_of_625_l1273_127314


namespace value_of_M_l1273_127377

theorem value_of_M (m n p M : ℝ) 
  (h1 : M = m / (n + p))
  (h2 : M = n / (p + m))
  (h3 : M = p / (m + n)) :
  M = 1/2 ∨ M = -1 := by
sorry

end value_of_M_l1273_127377


namespace min_value_problem_l1273_127313

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 3) + 1 / (b + 3) = 1 / 4 → 
  x + 3 * y ≤ a + 3 * b ∧ x + 3 * y = 19 * Real.sqrt 3 := by
  sorry

end min_value_problem_l1273_127313


namespace discount_reduction_l1273_127397

/-- Proves that applying a 30% discount followed by a 20% discount
    results in a total reduction of 44% from the original price. -/
theorem discount_reduction (P : ℝ) (P_pos : P > 0) :
  let first_discount := 0.3
  let second_discount := 0.2
  let price_after_first := P * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let total_reduction := (P - price_after_second) / P
  total_reduction = 0.44 := by
  sorry

end discount_reduction_l1273_127397


namespace blueberry_pies_count_l1273_127304

/-- Proves that the number of blueberry pies is 10, given 30 total pies equally divided among three types -/
theorem blueberry_pies_count (total_pies : ℕ) (num_types : ℕ) (h1 : total_pies = 30) (h2 : num_types = 3) :
  total_pies / num_types = 10 := by
  sorry

#check blueberry_pies_count

end blueberry_pies_count_l1273_127304


namespace kitchen_tile_comparison_l1273_127359

theorem kitchen_tile_comparison : 
  let area_figure1 : ℝ := π / 3 - Real.sqrt 3 / 4
  let area_figure2 : ℝ := Real.sqrt 3 / 2 - π / 6
  area_figure1 > area_figure2 := by
sorry

end kitchen_tile_comparison_l1273_127359


namespace fraction_equation_solution_l1273_127311

def valid_pairs : List (Int × Int) := [
  (12, 6), (-2, 6), (12, 4), (-2, 4), (10, 10), (0, 10), (10, 0)
]

theorem fraction_equation_solution (x y : Int) :
  x + y ≠ 0 →
  (x^2 + y^2) / (x + y) = 10 ↔ (x, y) ∈ valid_pairs :=
by sorry

end fraction_equation_solution_l1273_127311


namespace integral_gt_one_minus_one_over_n_l1273_127305

theorem integral_gt_one_minus_one_over_n (n : ℕ+) :
  ∫ x in (0:ℝ)..1, (1 / (1 + x ^ (n:ℝ))) > 1 - 1 / (n:ℝ) := by sorry

end integral_gt_one_minus_one_over_n_l1273_127305


namespace comparison_and_estimation_l1273_127309

theorem comparison_and_estimation : 
  (2 * Real.sqrt 3 < 4) ∧ 
  (4 < Real.sqrt 17) ∧ 
  (Real.sqrt 17 < 5) := by sorry

end comparison_and_estimation_l1273_127309


namespace linear_equation_condition_l1273_127326

/-- The equation (m-1)x^|m|+4=0 is linear if and only if m = -1 -/
theorem linear_equation_condition (m : ℤ) : 
  (∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, (m - 1 : ℝ) * |x|^|m| + 4 = a * x + b) ↔ m = -1 :=
by sorry

end linear_equation_condition_l1273_127326


namespace largest_six_digit_divisible_by_88_l1273_127366

theorem largest_six_digit_divisible_by_88 : ∃ n : ℕ, 
  n ≤ 999999 ∧ 
  n ≥ 100000 ∧
  n % 88 = 0 ∧
  ∀ m : ℕ, m ≤ 999999 ∧ m ≥ 100000 ∧ m % 88 = 0 → m ≤ n :=
by
  -- The proof goes here
  sorry

end largest_six_digit_divisible_by_88_l1273_127366
