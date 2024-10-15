import Mathlib

namespace NUMINAMATH_GPT_y_intercept_line_l604_60457

theorem y_intercept_line : ∀ y : ℝ, (∃ x : ℝ, x = 0 ∧ x - 3 * y - 1 = 0) → y = -1/3 :=
by
  intro y
  intro h
  sorry

end NUMINAMATH_GPT_y_intercept_line_l604_60457


namespace NUMINAMATH_GPT_dima_and_serezha_meet_time_l604_60474

-- Define the conditions and the main theorem to be proven.
theorem dima_and_serezha_meet_time :
  let dima_run_time := 15 / 60.0 -- Dima runs for 15 minutes
  let dima_run_speed := 6.0 -- Dima's running speed is 6 km/h
  let serezha_boat_speed := 20.0 -- Serezha's boat speed is 20 km/h
  let serezha_boat_time := 30 / 60.0 -- Serezha's boat time is 30 minutes
  let common_run_speed := 6.0 -- Both run at 6 km/h towards each other
  let distance_to_meet := dima_run_speed * dima_run_time -- Distance Dima runs along the shore
  let total_time := distance_to_meet / (common_run_speed + common_run_speed) -- Time until they meet after parting
  total_time = 7.5 / 60.0 := -- 7.5 minutes converted to hours
sorry

end NUMINAMATH_GPT_dima_and_serezha_meet_time_l604_60474


namespace NUMINAMATH_GPT_D_coin_count_l604_60458

def A_coin_count : ℕ := 21
def B_coin_count := A_coin_count - 9
def C_coin_count := B_coin_count + 17
def sum_A_B := A_coin_count + B_coin_count
def sum_C_D := sum_A_B + 5

theorem D_coin_count :
  ∃ D : ℕ, sum_C_D - C_coin_count = D :=
sorry

end NUMINAMATH_GPT_D_coin_count_l604_60458


namespace NUMINAMATH_GPT_models_kirsty_can_buy_l604_60497

def original_price : ℝ := 0.45
def saved_for_models : ℝ := 30 * original_price
def new_price : ℝ := 0.50

theorem models_kirsty_can_buy :
  saved_for_models / new_price = 27 :=
sorry

end NUMINAMATH_GPT_models_kirsty_can_buy_l604_60497


namespace NUMINAMATH_GPT_min_chord_length_m_l604_60464

-- Definition of the circle and the line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 6 * y + 4 = 0
def line_eq (m x y : ℝ) : Prop := m * x - y + 1 = 0

-- Theorem statement: value of m that minimizes the length of the chord
theorem min_chord_length_m (m : ℝ) : m = 1 ↔
  ∃ x y : ℝ, circle_eq x y ∧ line_eq m x y := sorry

end NUMINAMATH_GPT_min_chord_length_m_l604_60464


namespace NUMINAMATH_GPT_find_y_l604_60481

theorem find_y : (12 : ℝ)^3 * (2 : ℝ)^4 / 432 = 5184 → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_y_l604_60481


namespace NUMINAMATH_GPT_problem_statement_l604_60438

variable (a b c d : ℝ)

theorem problem_statement :
  (a^2 - a + 1) * (b^2 - b + 1) * (c^2 - c + 1) * (d^2 - d + 1) ≥ (9 / 16) * (a - b) * (b - c) * (c - d) * (d - a) :=
sorry

end NUMINAMATH_GPT_problem_statement_l604_60438


namespace NUMINAMATH_GPT_largest_digit_M_divisible_by_six_l604_60499

theorem largest_digit_M_divisible_by_six :
  (∃ M : ℕ, M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ ∀ m : ℕ, m ≤ M → (45670 + m) % 6 ≠ 0) :=
sorry

end NUMINAMATH_GPT_largest_digit_M_divisible_by_six_l604_60499


namespace NUMINAMATH_GPT_bacteria_colony_first_day_exceeds_100_l604_60437

theorem bacteria_colony_first_day_exceeds_100 :
  ∃ n : ℕ, 3 * 2^n > 100 ∧ (∀ m < n, 3 * 2^m ≤ 100) :=
sorry

end NUMINAMATH_GPT_bacteria_colony_first_day_exceeds_100_l604_60437


namespace NUMINAMATH_GPT_solve_diophantine_l604_60405

theorem solve_diophantine (x y : ℕ) (h1 : 1990 * x - 1989 * y = 1991) : x = 11936 ∧ y = 11941 := by
  have h_pos_x : 0 < x := by sorry
  have h_pos_y : 0 < y := by sorry
  have h_x : 1990 * 11936 = 1990 * x := by sorry
  have h_y : 1989 * 11941 = 1989 * y := by sorry
  sorry

end NUMINAMATH_GPT_solve_diophantine_l604_60405


namespace NUMINAMATH_GPT_fraction_equality_x_eq_neg1_l604_60485

theorem fraction_equality_x_eq_neg1 (x : ℝ) (h : (5 + x) / (7 + x) = (3 + x) / (4 + x)) : x = -1 := by
  sorry

end NUMINAMATH_GPT_fraction_equality_x_eq_neg1_l604_60485


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l604_60436

theorem batsman_average_after_17th_inning (A : ℝ) :
  (16 * A + 87) / 17 = A + 3 → A + 3 = 39 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l604_60436


namespace NUMINAMATH_GPT_alpha_beta_roots_eq_l604_60426

theorem alpha_beta_roots_eq {α β : ℝ} (hα : α^2 - α - 2006 = 0) (hβ : β^2 - β - 2006 = 0) (h_sum : α + β = 1) : 
  α + β^2 = 2007 :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_roots_eq_l604_60426


namespace NUMINAMATH_GPT_f_5_eq_25sqrt5_l604_60489

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom continuous_f : Continuous f
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_2 : f 2 = 5

theorem f_5_eq_25sqrt5 : f 5 = 25 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_f_5_eq_25sqrt5_l604_60489


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l604_60465

theorem roots_of_quadratic_eq {x y : ℝ} (h1 : x + y = 10) (h2 : (x - y) * (x + y) = 48) : 
    ∃ a b c : ℝ, (a ≠ 0) ∧ (x^2 - a*x + b = 0) ∧ (y^2 - a*y + b = 0) ∧ b = 19.24 := 
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l604_60465


namespace NUMINAMATH_GPT_fifty_percent_of_2002_is_1001_l604_60444

theorem fifty_percent_of_2002_is_1001 :
  (1 / 2) * 2002 = 1001 :=
sorry

end NUMINAMATH_GPT_fifty_percent_of_2002_is_1001_l604_60444


namespace NUMINAMATH_GPT_find_f_2011_l604_60419

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 2 then 2 * x^2
  else sorry  -- Placeholder, since f is only defined in (0, 2)

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x

theorem find_f_2011 : f 2011 = -2 :=
by
  -- Use properties of f to reduce and eventually find f(2011)
  sorry

end NUMINAMATH_GPT_find_f_2011_l604_60419


namespace NUMINAMATH_GPT_sum_of_x_and_y_l604_60427

-- Definitions of conditions
variables (x y : ℤ)
variable (h1 : x - y = 60)
variable (h2 : x = 37)

-- Statement of the problem to be proven
theorem sum_of_x_and_y : x + y = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l604_60427


namespace NUMINAMATH_GPT_expression_for_x_expression_for_y_l604_60478

variables {A B C : ℝ}

-- Conditions: A, B, and C are positive numbers with A > B > C > 0
axiom h1 : A > 0
axiom h2 : B > 0
axiom h3 : C > 0
axiom h4 : A > B
axiom h5 : B > C

-- A is x% greater than B
variables {x : ℝ}
axiom h6 : A = (1 + x / 100) * B

-- A is y% greater than C
variables {y : ℝ}
axiom h7 : A = (1 + y / 100) * C

-- Proving the expressions for x and y
theorem expression_for_x : x = 100 * ((A - B) / B) :=
sorry

theorem expression_for_y : y = 100 * ((A - C) / C) :=
sorry

end NUMINAMATH_GPT_expression_for_x_expression_for_y_l604_60478


namespace NUMINAMATH_GPT_marble_sharing_l604_60434

theorem marble_sharing 
  (total_marbles : ℕ) 
  (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 30) 
  (h2 : marbles_per_friend = 6) : 
  total_marbles / marbles_per_friend = 5 := 
by 
  sorry

end NUMINAMATH_GPT_marble_sharing_l604_60434


namespace NUMINAMATH_GPT_octal_subtraction_correct_l604_60482

-- Define the octal numbers
def octal752 : ℕ := 7 * 8^2 + 5 * 8^1 + 2 * 8^0
def octal364 : ℕ := 3 * 8^2 + 6 * 8^1 + 4 * 8^0
def octal376 : ℕ := 3 * 8^2 + 7 * 8^1 + 6 * 8^0

-- Prove the octal number subtraction
theorem octal_subtraction_correct : octal752 - octal364 = octal376 := by
  sorry

end NUMINAMATH_GPT_octal_subtraction_correct_l604_60482


namespace NUMINAMATH_GPT_manager_salary_3700_l604_60403

theorem manager_salary_3700
  (salary_20_employees_avg : ℕ)
  (salary_increase : ℕ)
  (total_employees : ℕ)
  (manager_salary : ℕ)
  (h_avg : salary_20_employees_avg = 1600)
  (h_increase : salary_increase = 100)
  (h_total_employees : total_employees = 20)
  (h_manager_salary : manager_salary = 21 * (salary_20_employees_avg + salary_increase) - 20 * salary_20_employees_avg) :
  manager_salary = 3700 :=
by
  sorry

end NUMINAMATH_GPT_manager_salary_3700_l604_60403


namespace NUMINAMATH_GPT_train_cross_time_l604_60471

noncomputable def train_length : ℝ := 317.5
noncomputable def train_speed_kph : ℝ := 153.3
noncomputable def convert_speed_to_mps (speed_kph : ℝ) : ℝ :=
  (speed_kph * 1000) / 3600

noncomputable def train_speed_mps : ℝ := convert_speed_to_mps train_speed_kph
noncomputable def time_to_cross_pole (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_cross_time :
  time_to_cross_pole train_length train_speed_mps = 7.456 :=
by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_train_cross_time_l604_60471


namespace NUMINAMATH_GPT_units_digit_37_pow_37_l604_60477

theorem units_digit_37_pow_37: (37^37) % 10 = 7 :=
by sorry

end NUMINAMATH_GPT_units_digit_37_pow_37_l604_60477


namespace NUMINAMATH_GPT_bucket_capacities_l604_60407

theorem bucket_capacities (a b c : ℕ) 
  (h1 : a + b + c = 1440) 
  (h2 : a + b / 5 = c) 
  (h3 : b + a / 3 = c) : 
  a = 480 ∧ b = 400 ∧ c = 560 := 
by 
  sorry

end NUMINAMATH_GPT_bucket_capacities_l604_60407


namespace NUMINAMATH_GPT_expand_polynomial_product_l604_60447

variable (x : ℝ)

def P (x : ℝ) : ℝ := 5 * x ^ 2 + 3 * x - 4
def Q (x : ℝ) : ℝ := 6 * x ^ 3 + 2 * x ^ 2 - x + 7

theorem expand_polynomial_product :
  (P x) * (Q x) = 30 * x ^ 5 + 28 * x ^ 4 - 23 * x ^ 3 + 24 * x ^ 2 + 25 * x - 28 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_product_l604_60447


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l604_60494

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l604_60494


namespace NUMINAMATH_GPT_movement_down_l604_60496

def point := (ℤ × ℤ)

theorem movement_down (C D : point) (hC : C = (1, 2)) (hD : D = (1, -1)) :
  D = (C.1, C.2 - 3) :=
by
  sorry

end NUMINAMATH_GPT_movement_down_l604_60496


namespace NUMINAMATH_GPT_correct_proposition_l604_60413

variable (a b : ℝ)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)
variable (a_gt_b : a > b)

theorem correct_proposition : 1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end NUMINAMATH_GPT_correct_proposition_l604_60413


namespace NUMINAMATH_GPT_relationship_between_x_and_y_l604_60425

theorem relationship_between_x_and_y
  (x y : ℝ)
  (h1 : 2 * x - 3 * y > 6 * x)
  (h2 : 3 * x - 4 * y < 2 * y - x) :
  x < y ∧ x < 0 ∧ y < 0 :=
sorry

end NUMINAMATH_GPT_relationship_between_x_and_y_l604_60425


namespace NUMINAMATH_GPT_set_B_listing_method_l604_60487

variable (A : Set ℕ) (B : Set ℕ)

theorem set_B_listing_method (hA : A = {1, 2, 3}) (hB : B = {x | x ∈ A}) :
  B = {1, 2, 3} :=
  by
    sorry

end NUMINAMATH_GPT_set_B_listing_method_l604_60487


namespace NUMINAMATH_GPT_banker_discount_calculation_l604_60443

-- Define the future value function with given interest rates and periods.
def face_value (PV : ℝ) : ℝ :=
  (PV * (1 + 0.10) ^ 4) * (1 + 0.12) ^ 4

-- Define the true discount as the difference between the future value and the present value.
def true_discount (PV : ℝ) : ℝ :=
  face_value PV - PV

-- Given conditions
def banker_gain : ℝ := 900

-- Define the banker's discount.
def banker_discount (PV : ℝ) : ℝ :=
  banker_gain + true_discount PV

-- The proof statement to prove the relationship.
theorem banker_discount_calculation (PV : ℝ) :
  banker_discount PV = banker_gain + (face_value PV - PV) := by
  sorry

end NUMINAMATH_GPT_banker_discount_calculation_l604_60443


namespace NUMINAMATH_GPT_students_neither_play_l604_60448

theorem students_neither_play (total_students football_players tennis_players both_players neither_players : ℕ)
  (h1 : total_students = 40)
  (h2 : football_players = 26)
  (h3 : tennis_players = 20)
  (h4 : both_players = 17)
  (h5 : neither_players = total_students - (football_players + tennis_players - both_players)) :
  neither_players = 11 :=
by
  sorry

end NUMINAMATH_GPT_students_neither_play_l604_60448


namespace NUMINAMATH_GPT_find_b2_a2_a1_l604_60453

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def geometric_sequence (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, b (n + 1) / b n = b 1 / b 0

theorem find_b2_a2_a1 (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a1 : a 0 = a₁) (h_a2 : a 2 = a₂)
  (h_b2 : b 2 = b₂) :
  b₂ * (a₂ - a₁) = 6 ∨ b₂ * (a₂ - a₁) = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_b2_a2_a1_l604_60453


namespace NUMINAMATH_GPT_circle_equation_l604_60468

-- Define the circle's equation as a predicate
def is_circle (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Given conditions, defining the known center and passing point
def center_x : ℝ := 2
def center_y : ℝ := -3
def point_M_x : ℝ := -1
def point_M_y : ℝ := 1

-- Prove that the circle with the given conditions has the correct equation
theorem circle_equation :
  is_circle x y center_x center_y 5 ↔ 
  ∀ x y : ℝ, (x - center_x)^2 + (y + center_y)^2 = 25 := sorry

end NUMINAMATH_GPT_circle_equation_l604_60468


namespace NUMINAMATH_GPT_length_of_AP_l604_60470

variables {x : ℝ} (M B C P A : Point) (circle : Circle)
  (BC AB MP : Line)

-- Definitions of conditions
def is_midpoint_of_arc (M B C : Point) (circle : Circle) : Prop := sorry
def is_perpendicular (MP AB : Line) (P : Point) : Prop := sorry
def chord_length (BC : Line) (length : ℝ) : Prop := sorry
def segment_length (BP : Line) (length : ℝ) : Prop := sorry

-- Prove statement
theorem length_of_AP
  (h1 : is_midpoint_of_arc M B C circle)
  (h2 : is_perpendicular MP AB P)
  (h3 : chord_length BC (2 * x))
  (h4 : segment_length BP (3 * x)) :
  ∃AP : Line, segment_length AP (2 * x) :=
sorry

end NUMINAMATH_GPT_length_of_AP_l604_60470


namespace NUMINAMATH_GPT_cos_540_eq_neg_1_l604_60415

theorem cos_540_eq_neg_1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_GPT_cos_540_eq_neg_1_l604_60415


namespace NUMINAMATH_GPT_range_of_a_l604_60414

theorem range_of_a (a : ℝ) : (∃ (x : ℤ), x > 1 ∧ x ≤ a) → ∃ (x : ℤ), (x = 2 ∨ x = 3 ∨ x = 4) ∧ 4 ≤ a ∧ a < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l604_60414


namespace NUMINAMATH_GPT_clock_ticks_6_times_at_6_oclock_l604_60424

theorem clock_ticks_6_times_at_6_oclock
  (h6 : 5 * t = 25)
  (h12 : 11 * t = 55) :
  t = 5 ∧ 6 = 6 :=
by
  sorry

end NUMINAMATH_GPT_clock_ticks_6_times_at_6_oclock_l604_60424


namespace NUMINAMATH_GPT_geometric_sequence_a4_range_l604_60433

theorem geometric_sequence_a4_range
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : 0 < a 1 ∧ a 1 < 1)
  (h2 : 1 < a 1 * q ∧ a 1 * q < 2)
  (h3 : 2 < a 1 * q^2 ∧ a 1 * q^2 < 3) :
  ∃ a4 : ℝ, a4 = a 1 * q^3 ∧ 2 * Real.sqrt 2 < a4 ∧ a4 < 9 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a4_range_l604_60433


namespace NUMINAMATH_GPT_prob_both_calligraphy_is_correct_prob_one_each_is_correct_l604_60440

section ProbabilityOfVolunteerSelection

variable (C P : ℕ) -- C = number of calligraphy competition winners, P = number of painting competition winners
variable (total_pairs : ℕ := 6 * (6 - 1) / 2) -- Number of ways to choose 2 out of 6 participants, binomial coefficient (6 choose 2)

-- Condition variables
def num_calligraphy_winners : ℕ := 4
def num_painting_winners : ℕ := 2
def num_total_winners : ℕ := num_calligraphy_winners + num_painting_winners

-- Number of pairs of both calligraphy winners
def pairs_both_calligraphy : ℕ := 4 * (4 - 1) / 2
-- Number of pairs of one calligraphy and one painting winner
def pairs_one_each : ℕ := 4 * 2

-- Probability calculations
def prob_both_calligraphy : ℚ := pairs_both_calligraphy / total_pairs
def prob_one_each : ℚ := pairs_one_each / total_pairs

-- Theorem statements to prove the probabilities of selected types of volunteers
theorem prob_both_calligraphy_is_correct : 
  prob_both_calligraphy = 2/5 := sorry

theorem prob_one_each_is_correct : 
  prob_one_each = 8/15 := sorry

end ProbabilityOfVolunteerSelection

end NUMINAMATH_GPT_prob_both_calligraphy_is_correct_prob_one_each_is_correct_l604_60440


namespace NUMINAMATH_GPT_find_first_number_l604_60439

theorem find_first_number (N : ℤ) (k m : ℤ) (h1 : N = 170 * k + 10) (h2 : 875 = 170 * m + 25) : N = 860 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l604_60439


namespace NUMINAMATH_GPT_weight_of_new_man_l604_60492

theorem weight_of_new_man (avg_increase : ℝ) (num_oarsmen : ℕ) (old_weight : ℝ) (weight_increase : ℝ) 
  (h1 : avg_increase = 1.8) (h2 : num_oarsmen = 10) (h3 : old_weight = 53) (h4 : weight_increase = num_oarsmen * avg_increase) :
  ∃ W : ℝ, W = old_weight + weight_increase :=
by
  sorry

end NUMINAMATH_GPT_weight_of_new_man_l604_60492


namespace NUMINAMATH_GPT_batsman_average_l604_60428

theorem batsman_average (A : ℕ) (total_runs_before : ℕ) (new_score : ℕ) (increase : ℕ)
  (h1 : total_runs_before = 11 * A)
  (h2 : new_score = 70)
  (h3 : increase = 3)
  (h4 : 11 * A + new_score = 12 * (A + increase)) :
  (A + increase) = 37 :=
by
  -- skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_batsman_average_l604_60428


namespace NUMINAMATH_GPT_orange_profit_44_percent_l604_60418

theorem orange_profit_44_percent :
  (∀ CP SP : ℚ, 0.99 * CP = 1 ∧ SP = CP / 16 → 1 / 11 = CP * (1 + 44 / 100)) :=
by
  sorry

end NUMINAMATH_GPT_orange_profit_44_percent_l604_60418


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l604_60479

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l604_60479


namespace NUMINAMATH_GPT_quadratic_eq_roots_minus5_and_7_l604_60409

theorem quadratic_eq_roots_minus5_and_7 : ∀ x : ℝ, (x + 5) * (x - 7) = 0 ↔ x = -5 ∨ x = 7 := by
  sorry

end NUMINAMATH_GPT_quadratic_eq_roots_minus5_and_7_l604_60409


namespace NUMINAMATH_GPT_lines_parallel_l604_60449

theorem lines_parallel :
  ∀ (x y : ℝ), (x - y + 2 = 0) ∧ (x - y + 1 = 0) → False :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_lines_parallel_l604_60449


namespace NUMINAMATH_GPT_solve_for_x_l604_60463

-- Definitions of the conditions
def condition (x : ℚ) : Prop :=
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 2 * x - 24)

-- Statement of the theorem
theorem solve_for_x (x : ℚ) (h : condition x) : x = -5 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l604_60463


namespace NUMINAMATH_GPT_rectangle_area_l604_60498

theorem rectangle_area (b l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 112) : l * b = 588 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l604_60498


namespace NUMINAMATH_GPT_trapezoid_sides_and_height_l604_60451

def trapezoid_base_height (a h A: ℝ) :=
  (h = (2 * a + 3) / 2) ∧
  (A = a^2 + 3 * a + 9 / 4) ∧
  (A = 2 * a^2 - 7.75)

theorem trapezoid_sides_and_height :
  ∃ (a b h : ℝ), (b = a + 3) ∧
  trapezoid_base_height a h 7.75 ∧
  a = 5 ∧ b = 8 ∧ h = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_sides_and_height_l604_60451


namespace NUMINAMATH_GPT_inequality_amgm_l604_60441

theorem inequality_amgm (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_amgm_l604_60441


namespace NUMINAMATH_GPT_regression_passes_through_none_l604_60493

theorem regression_passes_through_none (b a x y : ℝ) (h₀ : (0, 0) ≠ (0*b + a, 0))
                                     (h₁ : (x, 0) ≠ (x*b + a, 0))
                                     (h₂ : (x, y) ≠ (x*b + a, y)) : 
                                     ¬ ((0, 0) = (0*b + a, 0) ∨ (x, 0) = (x*b + a, 0) ∨ (x, y) = (x*b + a, y)) :=
by sorry

end NUMINAMATH_GPT_regression_passes_through_none_l604_60493


namespace NUMINAMATH_GPT_man_l604_60495

-- Lean 4 statement
theorem man's_speed_against_stream (speed_with_stream : ℝ) (speed_still_water : ℝ) 
(h1 : speed_with_stream = 16) (h2 : speed_still_water = 4) : 
  |speed_still_water - (speed_with_stream - speed_still_water)| = 8 :=
by
  -- Dummy proof since only statement is required
  sorry

end NUMINAMATH_GPT_man_l604_60495


namespace NUMINAMATH_GPT_find_s_l604_60445

theorem find_s (s t : ℚ) (h1 : 8 * s + 6 * t = 120) (h2 : s = t - 3) : s = 51 / 7 := by
  sorry

end NUMINAMATH_GPT_find_s_l604_60445


namespace NUMINAMATH_GPT_first_number_lcm_14_20_l604_60421

theorem first_number_lcm_14_20 (x : ℕ) (h : Nat.lcm x (Nat.lcm 14 20) = 140) : x = 1 := sorry

end NUMINAMATH_GPT_first_number_lcm_14_20_l604_60421


namespace NUMINAMATH_GPT_simplify_fraction_l604_60473

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_simplify_fraction_l604_60473


namespace NUMINAMATH_GPT_right_triangle_condition_l604_60432

theorem right_triangle_condition (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_GPT_right_triangle_condition_l604_60432


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l604_60461

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 2) → (a + b > 4 ∧ a * b > 4) ∧ ¬((a + b > 4 ∧ a * b > 4) → (a > 2 ∧ b > 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l604_60461


namespace NUMINAMATH_GPT_calculate_expression_l604_60411

theorem calculate_expression :
  (Real.sqrt 2 - 3)^0 - Real.sqrt 9 + |(-2: ℝ)| + ((-1/3: ℝ)⁻¹)^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l604_60411


namespace NUMINAMATH_GPT_simple_interest_rate_l604_60442

theorem simple_interest_rate (P A : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ) :
  P = 800 → A = 950 → T = 5 → SI = A - P → SI = (P * R * T) / 100 → R = 3.75 :=
  by
  intros hP hA hT hSI h_formula
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l604_60442


namespace NUMINAMATH_GPT_problem1_problem2_l604_60400

-- Problem 1
theorem problem1 (x y : ℝ) :
  2 * x^2 * y - 3 * x * y + 2 - x^2 * y + 3 * x * y = x^2 * y + 2 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) :
  9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l604_60400


namespace NUMINAMATH_GPT_solution_set_empty_l604_60417

-- Define the quadratic polynomial
def quadratic (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem that the solution set of the given inequality is empty
theorem solution_set_empty : ∀ x : ℝ, quadratic x < 0 → false :=
by
  intro x
  unfold quadratic
  sorry

end NUMINAMATH_GPT_solution_set_empty_l604_60417


namespace NUMINAMATH_GPT_coalsBurnedEveryTwentyMinutes_l604_60460

-- Definitions based on the conditions
def totalGrillingTime : Int := 240
def coalsPerBag : Int := 60
def numberOfBags : Int := 3
def grillingInterval : Int := 20

-- Derived definitions based on conditions
def totalCoals : Int := numberOfBags * coalsPerBag
def numberOfIntervals : Int := totalGrillingTime / grillingInterval

-- The Lean theorem we want to prove
theorem coalsBurnedEveryTwentyMinutes : (totalCoals / numberOfIntervals) = 15 := by
  sorry

end NUMINAMATH_GPT_coalsBurnedEveryTwentyMinutes_l604_60460


namespace NUMINAMATH_GPT_correct_expression_for_representatives_l604_60462

/-- Definition for the number of representatives y given the class size x
    and the conditions that follow. -/
def elect_representatives (x : ℕ) : ℕ :=
  if 6 < x % 10 then (x + 3) / 10 else x / 10

theorem correct_expression_for_representatives (x : ℕ) :
  elect_representatives x = (x + 3) / 10 :=
by
  sorry

end NUMINAMATH_GPT_correct_expression_for_representatives_l604_60462


namespace NUMINAMATH_GPT_binary_quadratic_lines_value_m_l604_60412

theorem binary_quadratic_lines_value_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + 2 * x * y + 8 * y^2 + 14 * y + m = 0) →
  m = 7 :=
sorry

end NUMINAMATH_GPT_binary_quadratic_lines_value_m_l604_60412


namespace NUMINAMATH_GPT_count_numbers_with_digit_2_from_200_to_499_l604_60406

def count_numbers_with_digit_2 (lower upper : ℕ) : ℕ :=
  let A := 100  -- Numbers of the form 2xx (from 200 to 299)
  let B := 30   -- Numbers of the form x2x (where first digit is 2, 3, or 4, last digit can be any)
  let C := 30   -- Numbers of the form xx2 (similar reasoning as B)
  let A_and_B := 10  -- Numbers of the form 22x
  let A_and_C := 10  -- Numbers of the form 2x2
  let B_and_C := 3   -- Numbers of the form x22
  let A_and_B_and_C := 1  -- The number 222
  A + B + C - A_and_B - A_and_C - B_and_C + A_and_B_and_C

theorem count_numbers_with_digit_2_from_200_to_499 : 
  count_numbers_with_digit_2 200 499 = 138 :=
by
  unfold count_numbers_with_digit_2
  exact rfl

end NUMINAMATH_GPT_count_numbers_with_digit_2_from_200_to_499_l604_60406


namespace NUMINAMATH_GPT_factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l604_60420

-- Statements corresponding to the given problems

-- Theorem for 1)
theorem factorize_poly1 (a : ℤ) : 
  (a^7 + a^5 + 1) = (a^2 + a + 1) * (a^5 - a^4 + a^3 - a + 1) := 
by sorry

-- Theorem for 2)
theorem factorize_poly2 (a b : ℤ) : 
  (a^5 + a*b^4 + b^5) = (a + b) * (a^4 - a^3*b + a^2*b^2 - a*b^3 + b^4) := 
by sorry

-- Theorem for 3)
theorem factorize_poly3 (a : ℤ) : 
  (a^7 - 1) = (a - 1) * (a^6 + a^5 + a^4 + a^3 + a^2 + a + 1) := 
by sorry

-- Theorem for 4)
theorem factorize_poly4 (a x : ℤ) : 
  (2 * a^3 - a * x^2 - x^3) = (a - x) * (2 * a^2 + 2 * a * x + x^2) := 
by sorry

end NUMINAMATH_GPT_factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l604_60420


namespace NUMINAMATH_GPT_diagonal_in_parallelogram_l604_60435

-- Define the conditions of the problem
variable (A B C D M : Point)
variable (parallelogram : Parallelogram A B C D)
variable (height_bisects_side : Midpoint M A D)
variable (height_length : Distance B M = 2)
variable (acute_angle_30 : Angle A B D = 30)

-- Define the theorem based on the conditions
theorem diagonal_in_parallelogram (h1 : parallelogram) (h2 : height_bisects_side)
  (h3 : height_length) (h4 : acute_angle_30) : 
  ∃ (BD_length : ℝ) (angle1 angle2 : ℝ), BD_length = 4 ∧ angle1 = 30 ∧ angle2 = 120 := 
sorry

end NUMINAMATH_GPT_diagonal_in_parallelogram_l604_60435


namespace NUMINAMATH_GPT_weight_of_lighter_boxes_l604_60430

theorem weight_of_lighter_boxes :
  ∃ (x : ℝ),
  (∀ (w : ℝ), w = 20 ∨ w = x) ∧
  (20 * 18 = 360) ∧
  (∃ (n : ℕ), n = 15 → 15 * 20 = 300) ∧
  (∃ (m : ℕ), m = 5 → 5 * 12 = 60) ∧
  (360 - 300 = 60) ∧
  (∀ (l : ℝ), l = 60 / 5 → l = x) →
  x = 12 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_lighter_boxes_l604_60430


namespace NUMINAMATH_GPT_last_digit_of_2_pow_2018_l604_60480

-- Definition of the cyclic pattern
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Function to find the last digit of 2^n using the cycle
def last_digit_of_power_of_two (n : ℕ) : ℕ :=
  last_digit_cycle.get! ((n % 4) - 1)

-- Main theorem statement
theorem last_digit_of_2_pow_2018 : last_digit_of_power_of_two 2018 = 4 :=
by
  -- The proof part is omitted
  sorry

end NUMINAMATH_GPT_last_digit_of_2_pow_2018_l604_60480


namespace NUMINAMATH_GPT_solve_y_percentage_l604_60455

noncomputable def y_percentage (x y : ℝ) : ℝ :=
  100 * y / x

theorem solve_y_percentage (x y : ℝ) (h : 0.20 * (x - y) = 0.14 * (x + y)) :
  y_percentage x y = 300 / 17 :=
by
  sorry

end NUMINAMATH_GPT_solve_y_percentage_l604_60455


namespace NUMINAMATH_GPT_samantha_coins_worth_l604_60416

-- Define the conditions and the final question with an expected answer.
theorem samantha_coins_worth (n d : ℕ) (h1 : n + d = 30)
  (h2 : 10 * n + 5 * d = 5 * n + 10 * d + 120) :
  (5 * n + 10 * d) = 165 := 
sorry

end NUMINAMATH_GPT_samantha_coins_worth_l604_60416


namespace NUMINAMATH_GPT_trigonometric_identity_l604_60456

-- Define variables
variables (α : ℝ) (hα : α ∈ Ioc 0 π) (h_tan : Real.tan α = 2)

-- The Lean statement
theorem trigonometric_identity :
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4 / 5 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l604_60456


namespace NUMINAMATH_GPT_triangle_perimeter_l604_60408

-- Define the side lengths
def a : ℕ := 7
def b : ℕ := 10
def c : ℕ := 15

-- Define the perimeter
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Statement of the proof problem
theorem triangle_perimeter : perimeter 7 10 15 = 32 := by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l604_60408


namespace NUMINAMATH_GPT_calculate_expression_l604_60454

theorem calculate_expression : 4 * 6 * 8 + 24 / 4 - 10 = 188 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l604_60454


namespace NUMINAMATH_GPT_ratio_of_intercepts_l604_60475

theorem ratio_of_intercepts
  (u v : ℚ)
  (h1 : 2 = 5 * u)
  (h2 : 3 = -7 * v) :
  u / v = -14 / 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_intercepts_l604_60475


namespace NUMINAMATH_GPT_Gargamel_bought_tires_l604_60410

def original_price_per_tire := 84
def sale_price_per_tire := 75
def total_savings := 36
def discount_per_tire := original_price_per_tire - sale_price_per_tire
def num_tires (total_savings : ℕ) (discount_per_tire : ℕ) := total_savings / discount_per_tire

theorem Gargamel_bought_tires :
  num_tires total_savings discount_per_tire = 4 :=
by
  sorry

end NUMINAMATH_GPT_Gargamel_bought_tires_l604_60410


namespace NUMINAMATH_GPT_mario_meet_speed_l604_60450

noncomputable def Mario_average_speed (x : ℝ) : ℝ :=
  let t1 := x / 5
  let t2 := x / 3
  let t3 := x / 4
  let t4 := x / 10
  let T := t1 + t2 + t3 + t4
  let d_mario := 1.5 * x
  d_mario / T

theorem mario_meet_speed : ∀ (x : ℝ), x > 0 → Mario_average_speed x = 90 / 53 :=
by
  intros
  rw [Mario_average_speed]
  -- You can insert calculations similar to those in the provided solution
  sorry

end NUMINAMATH_GPT_mario_meet_speed_l604_60450


namespace NUMINAMATH_GPT_find_integer_pairs_l604_60476

theorem find_integer_pairs :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (m n : ℤ), (m, n) ∈ S ↔ mn ≤ 0 ∧ m^3 + n^3 - 37 * m * n = 343) ∧ S.card = 9 :=
sorry

end NUMINAMATH_GPT_find_integer_pairs_l604_60476


namespace NUMINAMATH_GPT_sqrt_of_square_eq_seven_l604_60490

theorem sqrt_of_square_eq_seven (x : ℝ) (h : x^2 = 7) : x = Real.sqrt 7 ∨ x = -Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_sqrt_of_square_eq_seven_l604_60490


namespace NUMINAMATH_GPT_abs_ac_bd_leq_one_l604_60446

theorem abs_ac_bd_leq_one {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) : |a * c + b * d| ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_abs_ac_bd_leq_one_l604_60446


namespace NUMINAMATH_GPT_find_fx_for_neg_x_l604_60429

-- Let f be an odd function defined on ℝ 
variable {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = - f x)

-- Given condition for x > 0
variable (h_pos : ∀ x, 0 < x → f x = x^2 + x - 1)

-- Problem: Prove that f(x) = -x^2 + x + 1 for x < 0
theorem find_fx_for_neg_x (x : ℝ) (h_neg : x < 0) : f x = -x^2 + x + 1 :=
sorry

end NUMINAMATH_GPT_find_fx_for_neg_x_l604_60429


namespace NUMINAMATH_GPT_equal_side_length_is_4_or_10_l604_60483

-- Define the conditions
def isosceles_triangle (base_length equal_side_length : ℝ) :=
  base_length = 7 ∧
  (equal_side_length > base_length ∧ equal_side_length - base_length = 3) ∨
  (equal_side_length < base_length ∧ base_length - equal_side_length = 3)

-- Lean 4 statement to prove
theorem equal_side_length_is_4_or_10 (base_length equal_side_length : ℝ) 
  (h : isosceles_triangle base_length equal_side_length) : 
  equal_side_length = 4 ∨ equal_side_length = 10 :=
by 
  sorry

end NUMINAMATH_GPT_equal_side_length_is_4_or_10_l604_60483


namespace NUMINAMATH_GPT_ratio_A_B_l604_60472

theorem ratio_A_B (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : 5 * C = 8 * B) : A / B = 2 / 3 := 
by sorry

end NUMINAMATH_GPT_ratio_A_B_l604_60472


namespace NUMINAMATH_GPT_billy_gaming_percentage_l604_60469

-- Define the conditions
def free_time_per_day := 8
def days_in_weekend := 2
def total_free_time := free_time_per_day * days_in_weekend
def books_read := 3
def pages_per_book := 80
def reading_rate := 60 -- pages per hour
def total_pages_read := books_read * pages_per_book
def reading_time := total_pages_read / reading_rate
def gaming_time := total_free_time - reading_time
def gaming_percentage := (gaming_time / total_free_time) * 100

-- State the theorem
theorem billy_gaming_percentage : gaming_percentage = 75 := by
  sorry

end NUMINAMATH_GPT_billy_gaming_percentage_l604_60469


namespace NUMINAMATH_GPT_train_passes_tree_in_16_seconds_l604_60422

noncomputable def time_to_pass_tree (length_train : ℕ) (speed_train_kmh : ℕ) : ℕ :=
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  length_train / speed_train_ms

theorem train_passes_tree_in_16_seconds :
  time_to_pass_tree 280 63 = 16 :=
  by
    sorry

end NUMINAMATH_GPT_train_passes_tree_in_16_seconds_l604_60422


namespace NUMINAMATH_GPT_gcd_of_12547_23791_l604_60404

theorem gcd_of_12547_23791 : Nat.gcd 12547 23791 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_12547_23791_l604_60404


namespace NUMINAMATH_GPT_find_b_l604_60401

theorem find_b (b : ℝ) (x y : ℝ) (h1 : 2 * x^2 + b * x = 12) (h2 : y = x + 5.5) (h3 : y^2 * x + y * x^2 + y * (b * x) = 12) :
  b = -5 :=
sorry

end NUMINAMATH_GPT_find_b_l604_60401


namespace NUMINAMATH_GPT_hexagon_side_count_l604_60484

noncomputable def convex_hexagon_sides (a b perimeter : ℕ) : ℕ := 
  if a ≠ b then 6 - (perimeter - (6 * b)) else 0

theorem hexagon_side_count (G H I J K L : ℕ)
  (a b : ℕ)
  (p : ℕ)
  (dist_a : a = 7)
  (dist_b : b = 8)
  (perimeter : p = 46)
  (cond : GHIJKL = [a, b, X, Y, Z, W] ∧ ∀ x ∈ [X, Y, Z, W], x = a ∨ x = b)
  : convex_hexagon_sides a b p = 4 :=
by 
  sorry

end NUMINAMATH_GPT_hexagon_side_count_l604_60484


namespace NUMINAMATH_GPT_select_twins_in_grid_l604_60431

theorem select_twins_in_grid (persons : Fin 8 × Fin 8 → Fin 2) :
  ∃ (selection : Fin 8 × Fin 8 → Bool), 
    (∀ i : Fin 8, ∃ j : Fin 8, selection (i, j) = true) ∧ 
    (∀ j : Fin 8, ∃ i : Fin 8, selection (i, j) = true) :=
sorry

end NUMINAMATH_GPT_select_twins_in_grid_l604_60431


namespace NUMINAMATH_GPT_find_number_l604_60466

theorem find_number (x : ℤ) 
  (h1 : 3 * (2 * x + 9) = 51) : x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l604_60466


namespace NUMINAMATH_GPT_paintings_total_l604_60452

def june_paintings : ℕ := 2
def july_paintings : ℕ := 2 * june_paintings
def august_paintings : ℕ := 3 * july_paintings
def total_paintings : ℕ := june_paintings + july_paintings + august_paintings

theorem paintings_total : total_paintings = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_paintings_total_l604_60452


namespace NUMINAMATH_GPT_mersenne_primes_less_than_1000_l604_60491

open Nat

-- Definitions and Conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

-- Theorem Statement
theorem mersenne_primes_less_than_1000 : {p : ℕ | is_mersenne_prime p ∧ p < 1000} = {3, 7, 31, 127} :=
by
  sorry

end NUMINAMATH_GPT_mersenne_primes_less_than_1000_l604_60491


namespace NUMINAMATH_GPT_disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l604_60423

variable (p q : Prop)

theorem disjunction_false_implies_neg_p_true (hpq : ¬(p ∨ q)) : ¬p :=
by 
  sorry

theorem neg_p_true_does_not_imply_disjunction_false (hnp : ¬p) : ¬(¬(p ∨ q)) :=
by 
  sorry

end NUMINAMATH_GPT_disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l604_60423


namespace NUMINAMATH_GPT_sum_first_six_terms_l604_60486

-- Define the conditions given in the problem
def a3 := 7
def a4 := 11
def a5 := 15

-- Define the common difference
def d := a4 - a3 -- 4

-- Define the first term
def a1 := a3 - 2 * d -- -1

-- Define the sum of the first six terms of the arithmetic sequence
def S6 := (6 / 2) * (2 * a1 + (6 - 1) * d) -- 54

-- The theorem we want to prove
theorem sum_first_six_terms : S6 = 54 := by
  sorry

end NUMINAMATH_GPT_sum_first_six_terms_l604_60486


namespace NUMINAMATH_GPT_percentage_of_carnations_is_44_percent_l604_60459

noncomputable def total_flowers : ℕ := sorry
def pink_percentage : ℚ := 2 / 5
def red_percentage : ℚ := 2 / 5
def yellow_percentage : ℚ := 1 / 5
def pink_roses_fraction : ℚ := 2 / 5
def red_carnations_fraction : ℚ := 1 / 2

theorem percentage_of_carnations_is_44_percent
  (F : ℕ)
  (h_pink : pink_percentage * F = 2 / 5 * F)
  (h_red : red_percentage * F = 2 / 5 * F)
  (h_yellow : yellow_percentage * F = 1 / 5 * F)
  (h_pink_roses : pink_roses_fraction * (pink_percentage * F) = 2 / 25 * F)
  (h_red_carnations : red_carnations_fraction * (red_percentage * F) = 1 / 5 * F) :
  ((6 / 25 * F + 5 / 25 * F) / F) * 100 = 44 := sorry

end NUMINAMATH_GPT_percentage_of_carnations_is_44_percent_l604_60459


namespace NUMINAMATH_GPT_loss_percentage_l604_60402

theorem loss_percentage (CP SP_gain L : ℝ) 
  (h1 : CP = 1500)
  (h2 : SP_gain = CP + 0.05 * CP)
  (h3 : SP_gain = CP - (L/100) * CP + 225) : 
  L = 10 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_l604_60402


namespace NUMINAMATH_GPT_range_of_m_l604_60488

variable (p q : Prop)
variable (m : ℝ)
variable (hp : (∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (1 - m) = 1) → (0 < m ∧ m < 1/3)))
variable (hq : (m^2 - 15 * m < 0))

theorem range_of_m (h_not_p_and_q : ¬ (p ∧ q)) (h_p_or_q : p ∨ q) :
  (1/3 ≤ m ∧ m < 15) :=
sorry

end NUMINAMATH_GPT_range_of_m_l604_60488


namespace NUMINAMATH_GPT_problems_per_hour_l604_60467

def num_math_problems : ℝ := 17.0
def num_spelling_problems : ℝ := 15.0
def total_hours : ℝ := 4.0

theorem problems_per_hour :
  (num_math_problems + num_spelling_problems) / total_hours = 8.0 := by
  sorry

end NUMINAMATH_GPT_problems_per_hour_l604_60467
