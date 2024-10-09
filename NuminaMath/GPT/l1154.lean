import Mathlib

namespace purely_imaginary_condition_l1154_115463

-- Define the necessary conditions
def real_part_eq_zero (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0
def imaginary_part_neq_zero (m : ℝ) : Prop := m^2 - 3 * m + 2 ≠ 0

-- State the theorem to be proved
theorem purely_imaginary_condition (m : ℝ) :
  real_part_eq_zero m ∧ imaginary_part_neq_zero m ↔ m = -1/2 :=
sorry

end purely_imaginary_condition_l1154_115463


namespace min_value_expression_l1154_115468

theorem min_value_expression (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 14) ∧ (∀ z : ℝ, (z = (x + 10) / Real.sqrt (x - 4)) → y ≤ z) := sorry

end min_value_expression_l1154_115468


namespace minimal_board_size_for_dominoes_l1154_115412

def board_size_is_minimal (n: ℕ) (total_area: ℕ) (domino_size: ℕ) (num_dominoes: ℕ) : Prop :=
  ∀ m: ℕ, m < n → ¬ (total_area ≥ m * m ∧ m * m = num_dominoes * domino_size)

theorem minimal_board_size_for_dominoes (n: ℕ) :
  board_size_is_minimal 77 2008 2 1004 :=
by
  sorry

end minimal_board_size_for_dominoes_l1154_115412


namespace optimal_strategy_for_father_l1154_115428

-- Define the individual players
inductive player
| Father 
| Mother 
| Son

open player

-- Define the probabilities of player defeating another
def prob_defeat (p1 p2 : player) : ℝ := sorry  -- These will be defined as per the problem's conditions.

-- Define the probability of father winning given the first matchups
def P_father_vs_mother : ℝ :=
  prob_defeat Father Mother * prob_defeat Father Son +
  prob_defeat Father Mother * prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother +
  prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son * prob_defeat Father Mother

def P_father_vs_son : ℝ :=
  prob_defeat Father Son * prob_defeat Father Mother +
  prob_defeat Father Son * prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son +
  prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother * prob_defeat Father Son

-- Define the optimality condition
theorem optimal_strategy_for_father :
  P_father_vs_mother > P_father_vs_son :=
sorry

end optimal_strategy_for_father_l1154_115428


namespace number_of_small_pipes_needed_l1154_115400

theorem number_of_small_pipes_needed :
  let diameter_large := 8
  let diameter_small := 1
  let radius_large := diameter_large / 2
  let radius_small := diameter_small / 2
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let num_small_pipes := area_large / area_small
  num_small_pipes = 64 :=
by
  sorry

end number_of_small_pipes_needed_l1154_115400


namespace factorize_square_diff_factorize_common_factor_l1154_115452

-- Problem 1: Difference of squares
theorem factorize_square_diff (x : ℝ) : 4 * x^2 - 9 = (2 * x + 3) * (2 * x - 3) := 
by
  sorry

-- Problem 2: Factoring out common terms
theorem factorize_common_factor (a b x y : ℝ) (h : y - x = -(x - y)) : 
  2 * a * (x - y) - 3 * b * (y - x) = (x - y) * (2 * a + 3 * b) := 
by
  sorry

end factorize_square_diff_factorize_common_factor_l1154_115452


namespace treasures_coins_count_l1154_115430

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l1154_115430


namespace find_mn_l1154_115422

theorem find_mn (m n : ℕ) (h : m > 0 ∧ n > 0) (eq1 : m^2 + n^2 + 4 * m - 46 = 0) :
  mn = 5 ∨ mn = 15 := by
  sorry

end find_mn_l1154_115422


namespace total_seats_value_l1154_115464

noncomputable def students_per_bus : ℝ := 14.0
noncomputable def number_of_buses : ℝ := 2.0
noncomputable def total_seats : ℝ := students_per_bus * number_of_buses

theorem total_seats_value : total_seats = 28.0 :=
by
  sorry

end total_seats_value_l1154_115464


namespace smallest_n_l1154_115461

theorem smallest_n (n : ℕ) (h : 10 - n ≥ 0) : 
  (9 / 10) * (8 / 9) * (7 / 8) * (6 / 7) * (5 / 6) * (4 / 5) < 0.5 → n = 6 :=
by
  sorry

end smallest_n_l1154_115461


namespace percent_increase_l1154_115450

theorem percent_increase (new_value old_value : ℕ) (h_new : new_value = 480) (h_old : old_value = 320) :
  ((new_value - old_value) / old_value) * 100 = 50 := by
  sorry

end percent_increase_l1154_115450


namespace smallest_positive_integer_l1154_115499

-- Given integers m and n, prove the smallest positive integer of the form 2017m + 48576n
theorem smallest_positive_integer (m n : ℤ) : 
  ∃ m n : ℤ, 2017 * m + 48576 * n = 1 := by
sorry

end smallest_positive_integer_l1154_115499


namespace equation_has_three_real_roots_l1154_115497

noncomputable def f (x : ℝ) : ℝ := 2^x - x^2 - 1

theorem equation_has_three_real_roots : ∃! (x : ℝ), f x = 0 :=
by sorry

end equation_has_three_real_roots_l1154_115497


namespace min_value_of_xy_l1154_115471

theorem min_value_of_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 2 * x + y + 6 = x * y) : 18 ≤ x * y :=
by
  sorry

end min_value_of_xy_l1154_115471


namespace correct_statement_A_l1154_115405

-- Definitions for conditions
def general_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

def actinomycetes_dilution_range : Set ℕ := {10^3, 10^4, 10^5}

def fungi_dilution_range : Set ℕ := {10^2, 10^3, 10^4}

def first_experiment_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

-- Statement to prove
theorem correct_statement_A : 
  (general_dilution_range = {10^3, 10^4, 10^5, 10^6, 10^7}) :=
sorry

end correct_statement_A_l1154_115405


namespace inverse_of_k_l1154_115409

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

noncomputable def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem inverse_of_k :
  ∀ y : ℝ, k_inv (k y) = y :=
by
  intros x
  simp [k, k_inv, f, g]
  sorry

end inverse_of_k_l1154_115409


namespace sum_of_numbers_greater_than_or_equal_to_0_1_l1154_115418

def num1 : ℝ := 0.8
def num2 : ℝ := 0.5  -- converting 1/2 to 0.5
def num3 : ℝ := 0.6

def is_greater_than_or_equal_to_0_1 (n : ℝ) : Prop :=
  n ≥ 0.1

theorem sum_of_numbers_greater_than_or_equal_to_0_1 :
  is_greater_than_or_equal_to_0_1 num1 ∧ 
  is_greater_than_or_equal_to_0_1 num2 ∧ 
  is_greater_than_or_equal_to_0_1 num3 →
  num1 + num2 + num3 = 1.9 :=
by
  sorry

end sum_of_numbers_greater_than_or_equal_to_0_1_l1154_115418


namespace maximum_value_is_one_div_sqrt_two_l1154_115462

noncomputable def maximum_value_2ab_root2_plus_2ac_plus_2bc (a b c : ℝ) : ℝ :=
  2 * a * b * Real.sqrt 2 + 2 * a * c + 2 * b * c

theorem maximum_value_is_one_div_sqrt_two (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h : a^2 + b^2 + c^2 = 1) :
  maximum_value_2ab_root2_plus_2ac_plus_2bc a b c ≤ 1 / Real.sqrt 2 :=
by
  sorry

end maximum_value_is_one_div_sqrt_two_l1154_115462


namespace proper_fraction_and_condition_l1154_115459

theorem proper_fraction_and_condition (a b : ℤ) (h1 : 1 < a) (h2 : b = 2 * a - 1) :
  0 < a ∧ a < b ∧ (a - 1 : ℚ) / (b - 1) = 1 / 2 :=
by
  sorry

end proper_fraction_and_condition_l1154_115459


namespace positive_integer_solution_l1154_115484

theorem positive_integer_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≤ y ∧ y ≤ z) (h_eq : 5 * (x * y + y * z + z * x) = 4 * x * y * z) :
  (x = 2 ∧ y = 5 ∧ z = 10) ∨ (x = 2 ∧ y = 4 ∧ z = 20) :=
sorry

end positive_integer_solution_l1154_115484


namespace max_g_f_inequality_l1154_115491

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := f x - x / 4 - 1

theorem max_g : ∃ x : ℝ, g x = 2 * Real.log 2 - 7 / 4 :=
sorry

theorem f_inequality (x : ℝ) (hx : 0 < x) : f x < (Real.exp x - 1) / x^2 :=
sorry

end max_g_f_inequality_l1154_115491


namespace fraction_sum_5625_l1154_115417

theorem fraction_sum_5625 : 
  ∃ (a b : ℕ), 0.5625 = (9 : ℚ) / 16 ∧ (a + b = 25) := 
by 
  sorry

end fraction_sum_5625_l1154_115417


namespace steve_book_sales_l1154_115467

theorem steve_book_sales
  (copies_price : ℝ)
  (agent_rate : ℝ)
  (total_earnings : ℝ)
  (net_per_copy : ℝ := copies_price * (1 - agent_rate))
  (total_copies_sold : ℝ := total_earnings / net_per_copy) :
  copies_price = 2 → agent_rate = 0.10 → total_earnings = 1620000 → total_copies_sold = 900000 :=
by
  intros
  sorry

end steve_book_sales_l1154_115467


namespace circle_origin_range_l1154_115489

theorem circle_origin_range (m : ℝ) : 
  (0 - m)^2 + (0 + m)^2 < 4 → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
sorry

end circle_origin_range_l1154_115489


namespace hyperbola_condition_l1154_115443

theorem hyperbola_condition (m n : ℝ) : (m < 0 ∧ 0 < n) → (∀ x y : ℝ, nx^2 + my^2 = 1 → (n * x^2 - m * y^2 > 0)) :=
by
  sorry

end hyperbola_condition_l1154_115443


namespace calculate_area_of_pentagon_l1154_115438

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℝ :=
  let triangle_area := (1/2 : ℝ) * b * a
  let trapezoid_area := (1/2 : ℝ) * (c + e) * d
  triangle_area + trapezoid_area

theorem calculate_area_of_pentagon : area_of_pentagon 18 25 28 30 25 = 1020 :=
sorry

end calculate_area_of_pentagon_l1154_115438


namespace hyperbola_condition_l1154_115446

theorem hyperbola_condition (k : ℝ) : 
  (-1 < k ∧ k < 1) ↔ (∃ x y : ℝ, (x^2 / (k-1) + y^2 / (k+1)) = 1) := 
sorry

end hyperbola_condition_l1154_115446


namespace project_completion_rate_l1154_115414

variables {a b c d e : ℕ} {f g : ℚ}  -- Assuming efficiency ratings can be represented by rational numbers.

theorem project_completion_rate (h : (a * f / c) = b / c) 
: (d * g / e) = bdge / ca := 
sorry

end project_completion_rate_l1154_115414


namespace snow_probability_first_week_l1154_115479

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end snow_probability_first_week_l1154_115479


namespace circus_total_tickets_sold_l1154_115448

-- Definitions from the conditions
def revenue_total : ℕ := 2100
def lower_seat_tickets_sold : ℕ := 50
def price_lower : ℕ := 30
def price_upper : ℕ := 20

-- Definition derived from the conditions
def tickets_total (L U : ℕ) : ℕ := L + U

-- The theorem we need to prove
theorem circus_total_tickets_sold (L U : ℕ) (hL: L = lower_seat_tickets_sold)
    (h₁ : price_lower * L + price_upper * U = revenue_total) : 
    tickets_total L U = 80 :=
by
  sorry  -- Proof omitted

end circus_total_tickets_sold_l1154_115448


namespace lowest_number_of_students_l1154_115498

theorem lowest_number_of_students (n : ℕ) (h1 : n % 18 = 0) (h2 : n % 24 = 0) : n = 72 := by
  sorry

end lowest_number_of_students_l1154_115498


namespace complement_of_M_l1154_115431

open Set

def M : Set ℝ := { x | (2 - x) / (x + 3) < 0 }

theorem complement_of_M : (Mᶜ = { x : ℝ | -3 ≤ x ∧ x ≤ 2 }) :=
by
  sorry

end complement_of_M_l1154_115431


namespace height_percentage_differences_l1154_115465

variable (B : ℝ) (A : ℝ) (R : ℝ)
variable (h1 : A = 1.25 * B) (h2 : R = 1.0625 * B)

theorem height_percentage_differences :
  (100 * (A - B) / B = 25) ∧
  (100 * (A - R) / A = 15) ∧
  (100 * (R - B) / B = 6.25) :=
by
  sorry

end height_percentage_differences_l1154_115465


namespace cricket_match_count_l1154_115416

theorem cricket_match_count (x : ℕ) (h_avg_1 : ℕ → ℕ) (h_avg_2 : ℕ) (h_avg_all : ℕ) (h_eq : 50 * x + 26 * 15 = 42 * (x + 15)) : x = 30 :=
by
  sorry

end cricket_match_count_l1154_115416


namespace problem1_proof_problem2_proof_l1154_115469

noncomputable def problem1 : Real :=
  Real.sqrt 2 * Real.sqrt 3 + Real.sqrt 24

theorem problem1_proof : problem1 = 3 * Real.sqrt 6 :=
  sorry

noncomputable def problem2 : Real :=
  (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3)

theorem problem2_proof : problem2 = 6 :=
  sorry

end problem1_proof_problem2_proof_l1154_115469


namespace floss_per_student_l1154_115439

theorem floss_per_student
  (students : ℕ)
  (yards_per_packet : ℕ)
  (floss_left_over : ℕ)
  (total_packets : ℕ)
  (total_floss : ℕ)
  (total_floss_bought : ℕ)
  (smallest_multiple_of_35 : ℕ)
  (each_student_needs : ℕ)
  (hs1 : students = 20)
  (hs2 : yards_per_packet = 35)
  (hs3 : floss_left_over = 5)
  (hs4 : total_floss = total_packets * yards_per_packet)
  (hs5 : total_floss_bought = total_floss + floss_left_over)
  (hs6 : total_floss_bought % 35 = 0)
  (hs7 : smallest_multiple_of_35 > total_packets * yards_per_packet - floss_left_over)
  (hs8 : 20 * each_student_needs + 5 = smallest_multiple_of_35)
  : each_student_needs = 5 :=
by
  sorry

end floss_per_student_l1154_115439


namespace find_geometric_arithmetic_progressions_l1154_115445

theorem find_geometric_arithmetic_progressions
    (b1 b2 b3 : ℚ)
    (h1 : b2^2 = b1 * b3)
    (h2 : b2 + 2 = (b1 + b3) / 2)
    (h3 : (b2 + 2)^2 = b1 * (b3 + 16)) :
    (b1 = 1 ∧ b2 = 3 ∧ b3 = 9) ∨ (b1 = 1/9 ∧ b2 = -5/9 ∧ b3 = 25/9) :=
  sorry

end find_geometric_arithmetic_progressions_l1154_115445


namespace cover_black_squares_with_L_shape_l1154_115406

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the main theorem
theorem cover_black_squares_with_L_shape (n : ℕ) (h_odd : is_odd n) (h_corner_black : ∀i j, (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 1) : n ≥ 7 :=
sorry

end cover_black_squares_with_L_shape_l1154_115406


namespace find_x_intervals_l1154_115419

theorem find_x_intervals :
  {x : ℝ | x^3 - x^2 + 11*x - 42 < 0} = { x | -2 < x ∧ x < 3 ∨ 3 < x ∧ x < 7 } :=
by sorry

end find_x_intervals_l1154_115419


namespace average_student_headcount_proof_l1154_115402

def average_student_headcount : ℕ := (11600 + 11800 + 12000 + 11400) / 4

theorem average_student_headcount_proof :
  average_student_headcount = 11700 :=
by
  -- calculation here
  sorry

end average_student_headcount_proof_l1154_115402


namespace sum_of_fractions_l1154_115420

theorem sum_of_fractions :
  (1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6) + 1 / (5 * 6 * 7) + 1 / (6 * 7 * 8)) = 3 / 16 := 
by
  sorry

end sum_of_fractions_l1154_115420


namespace necessary_but_not_sufficient_condition_l1154_115441

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem necessary_but_not_sufficient_condition (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f := 
sorry

end necessary_but_not_sufficient_condition_l1154_115441


namespace walt_total_interest_l1154_115451

noncomputable def total_investment : ℝ := 12000
noncomputable def investment_at_7_percent : ℝ := 5500
noncomputable def investment_at_9_percent : ℝ := total_investment - investment_at_7_percent
noncomputable def rate_7_percent : ℝ := 0.07
noncomputable def rate_9_percent : ℝ := 0.09

theorem walt_total_interest :
  let interest_7 : ℝ := investment_at_7_percent * rate_7_percent
  let interest_9 : ℝ := investment_at_9_percent * rate_9_percent
  interest_7 + interest_9 = 970 := by
  sorry

end walt_total_interest_l1154_115451


namespace petes_original_number_l1154_115481

theorem petes_original_number (x : ℤ) (h : 4 * (2 * x + 20) = 200) : x = 15 :=
sorry

end petes_original_number_l1154_115481


namespace alexa_weight_proof_l1154_115490

variable (totalWeight katerinaWeight alexaWeight : ℕ)

def weight_relation (totalWeight katerinaWeight alexaWeight : ℕ) : Prop :=
  totalWeight = katerinaWeight + alexaWeight

theorem alexa_weight_proof (h1 : totalWeight = 95) (h2 : katerinaWeight = 49) : alexaWeight = 46 :=
by
  have h : alexaWeight = totalWeight - katerinaWeight := by
    sorry
  rw [h1, h2] at h
  exact h

end alexa_weight_proof_l1154_115490


namespace percentage_of_sikhs_l1154_115475

theorem percentage_of_sikhs
  (total_boys : ℕ := 400)
  (percent_muslims : ℕ := 44)
  (percent_hindus : ℕ := 28)
  (other_boys : ℕ := 72) :
  ((total_boys - (percent_muslims * total_boys / 100 + percent_hindus * total_boys / 100 + other_boys)) * 100 / total_boys) = 10 :=
by
  -- proof goes here
  sorry

end percentage_of_sikhs_l1154_115475


namespace vacation_cost_eq_l1154_115493

theorem vacation_cost_eq (C : ℕ) (h : C / 3 - C / 5 = 50) : C = 375 :=
sorry

end vacation_cost_eq_l1154_115493


namespace polynomial_integer_roots_l1154_115424

theorem polynomial_integer_roots :
  ∀ x : ℤ, (x^3 - 3*x^2 - 10*x + 20 = 0) ↔ (x = -2 ∨ x = 5) :=
by
  sorry

end polynomial_integer_roots_l1154_115424


namespace nathan_ate_100_gumballs_l1154_115435

/-- Define the number of gumballs per package. -/
def gumballs_per_package : ℝ := 5.0

/-- Define the number of packages Nathan ate. -/
def number_of_packages : ℝ := 20.0

/-- Define the total number of gumballs Nathan ate. -/
def total_gumballs : ℝ := number_of_packages * gumballs_per_package

/-- Prove that Nathan ate 100.0 gumballs. -/
theorem nathan_ate_100_gumballs : total_gumballs = 100.0 :=
sorry

end nathan_ate_100_gumballs_l1154_115435


namespace xy_series_16_l1154_115455

noncomputable def series (x y : ℝ) : ℝ := ∑' n : ℕ, (n + 1) * (x * y)^n

theorem xy_series_16 (x y : ℝ) (h_series : series x y = 16) (h_abs : |x * y| < 1) :
  (x = 3 / 4 ∧ (y = 1 ∨ y = -1)) :=
sorry

end xy_series_16_l1154_115455


namespace abs_sum_values_l1154_115477

theorem abs_sum_values (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := 
by
  sorry

end abs_sum_values_l1154_115477


namespace jade_transactions_l1154_115457

theorem jade_transactions :
  ∀ (transactions_mabel transactions_anthony transactions_cal transactions_jade : ℕ),
    transactions_mabel = 90 →
    transactions_anthony = transactions_mabel + transactions_mabel / 10 →
    transactions_cal = (transactions_anthony * 2) / 3 →
    transactions_jade = transactions_cal + 19 →
    transactions_jade = 85 :=
by
  intros transactions_mabel transactions_anthony transactions_cal transactions_jade
  intros h_mabel h_anthony h_cal h_jade
  sorry

end jade_transactions_l1154_115457


namespace trapezoid_angles_sum_l1154_115466

theorem trapezoid_angles_sum {α β γ δ : ℝ} (h : α + β + γ + δ = 360) (h1 : α = 60) (h2 : β = 120) :
  γ + δ = 180 :=
by
  sorry

end trapezoid_angles_sum_l1154_115466


namespace at_least_two_foxes_met_same_number_of_koloboks_l1154_115408

-- Define the conditions
def number_of_foxes : ℕ := 14
def number_of_koloboks : ℕ := 92

-- The theorem statement to be proven
theorem at_least_two_foxes_met_same_number_of_koloboks :
  ∃ (f : Fin number_of_foxes.succ → ℕ), 
    (∀ i, f i ≤ number_of_koloboks) ∧ 
    ∃ i j, i ≠ j ∧ f i = f j :=
by
  sorry

end at_least_two_foxes_met_same_number_of_koloboks_l1154_115408


namespace unique_prime_satisfying_condition_l1154_115436

theorem unique_prime_satisfying_condition :
  ∃! p : ℕ, Prime p ∧ (∀ q : ℕ, Prime q ∧ q < p → ∀ k r : ℕ, p = k * q + r ∧ 0 ≤ r ∧ r < q → ∀ a : ℕ, a > 1 → ¬ a^2 ∣ r) ∧ p = 13 :=
sorry

end unique_prime_satisfying_condition_l1154_115436


namespace class_committee_selection_l1154_115487

theorem class_committee_selection :
  let members := ["A", "B", "C", "D", "E"]
  let admissible_entertainment_candidates := ["C", "D", "E"]
  ∃ (entertainment : String) (study : String) (sports : String),
    entertainment ∈ admissible_entertainment_candidates ∧
    study ∈ members.erase entertainment ∧
    sports ∈ (members.erase entertainment).erase study ∧
    (3 * 4 * 3 = 36) :=
sorry

end class_committee_selection_l1154_115487


namespace intersection_and_area_l1154_115444

theorem intersection_and_area (A B : ℝ × ℝ) (x y : ℝ):
  (x - 2 * y - 5 = 0) → (x ^ 2 + y ^ 2 = 50) →
  (A = (-5, -5) ∨ A = (7, 1)) → (B = (-5, -5) ∨ B = (7, 1)) →
  (A ≠ B) →
  ∃ (area : ℝ), area = 15 :=
by
  sorry

end intersection_and_area_l1154_115444


namespace simplify_and_evaluate_expression_l1154_115488

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = (Real.sqrt 2) + 1) : 
  (1 - (1 / a)) / ((a ^ 2 - 2 * a + 1) / a) = (Real.sqrt 2) / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l1154_115488


namespace P_inequality_l1154_115415

variable {α : Type*} [LinearOrderedField α]

def P (a b c : α) (x : α) : α := a * x^2 + b * x + c

theorem P_inequality (a b c x y : α) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (P a b c (x * y))^2 ≤ (P a b c (x^2)) * (P a b c (y^2)) :=
sorry

end P_inequality_l1154_115415


namespace sum_of_solutions_l1154_115473

theorem sum_of_solutions : 
  (∀ x : ℝ, (x^2 - 5 * x + 3)^(x^2 - 6 * x + 4) = 1) → 
  (∃ s : ℝ, s = 16) :=
by
  sorry

end sum_of_solutions_l1154_115473


namespace solve_equation_l1154_115470

theorem solve_equation (x : ℝ) : (x + 1) * (x - 3) = 5 ↔ (x = 4 ∨ x = -2) :=
by
  sorry

end solve_equation_l1154_115470


namespace problem_statement_l1154_115429

noncomputable def a : ℝ := (Real.tan 23) / (1 - (Real.tan 23) ^ 2)
noncomputable def b : ℝ := 2 * Real.sin 13 * Real.cos 13
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos 50) / 2)

theorem problem_statement : c < b ∧ b < a :=
by
  -- Proof omitted
  sorry

end problem_statement_l1154_115429


namespace number_of_kids_per_day_l1154_115413

theorem number_of_kids_per_day (K : ℕ) 
    (kids_charge : ℕ := 3) 
    (adults_charge : ℕ := kids_charge * 2) 
    (daily_earnings_from_adults : ℕ := 10 * adults_charge) 
    (weekly_earnings : ℕ := 588) 
    (daily_earnings : ℕ := weekly_earnings / 7) :
    (daily_earnings - daily_earnings_from_adults) / kids_charge = 8 :=
by
  sorry

end number_of_kids_per_day_l1154_115413


namespace find_n_l1154_115483

theorem find_n {n : ℕ} (avg1 : ℕ) (avg2 : ℕ) (S : ℕ) :
  avg1 = 7 →
  avg2 = 6 →
  S = 7 * n →
  6 = (S - 11) / (n + 1) →
  n = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end find_n_l1154_115483


namespace unique_solution_implies_relation_l1154_115433

open Nat

noncomputable def unique_solution (a b : ℤ) :=
  ∃! (x y z : ℤ), x + y = a - 1 ∧ x * (y + 1) - z^2 = b

theorem unique_solution_implies_relation (a b : ℤ) :
  unique_solution a b → b = (a * a) / 4 := sorry

end unique_solution_implies_relation_l1154_115433


namespace parabola_equation_l1154_115437

open Classical

noncomputable def circle_center : ℝ × ℝ := (2, 0)

theorem parabola_equation (vertex : ℝ × ℝ) (focus : ℝ × ℝ) :
  vertex = (0, 0) ∧ focus = circle_center → ∀ x y : ℝ, y^2 = 8 * x := by
  intro h
  sorry

end parabola_equation_l1154_115437


namespace triangle_angle_property_l1154_115447

variables {a b c : ℝ}
variables {A B C : ℝ} -- angles in triangle ABC

-- definition of a triangle side condition
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- condition given in the problem
def satisfies_condition (a b c : ℝ) : Prop := b^2 = a^2 + c^2

-- angle property based on given problem
def angle_B_is_right (A B C : ℝ) : Prop := B = 90

theorem triangle_angle_property (a b c : ℝ) (A B C : ℝ)
  (ht : triangle a b c) 
  (hc : satisfies_condition a b c) : 
  angle_B_is_right A B C :=
sorry

end triangle_angle_property_l1154_115447


namespace sin_minus_cos_eq_sqrt3_div2_l1154_115411

theorem sin_minus_cos_eq_sqrt3_div2
  (α : ℝ) 
  (h_range : (Real.pi / 4) < α ∧ α < (Real.pi / 2))
  (h_sincos : Real.sin α * Real.cos α = 1 / 8) :
  Real.sin α - Real.cos α = Real.sqrt 3 / 2 :=
by
  sorry

end sin_minus_cos_eq_sqrt3_div2_l1154_115411


namespace largest_divisor_of_n_l1154_115410

theorem largest_divisor_of_n (n : ℕ) (h_pos: n > 0) (h_div: 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end largest_divisor_of_n_l1154_115410


namespace min_fraction_value_l1154_115480

theorem min_fraction_value 
    (a : ℕ → ℝ) 
    (S : ℕ → ℝ) 
    (d : ℝ) 
    (n : ℕ) 
    (h1 : ∀ {n}, a n = 5 + (n - 1) * d)
    (h2 : (a 2) * (a 10) = (a 4 - 1)^2) 
    (h3 : S n = (n * (a 1 + a n)) / 2)
    (h4 : a 1 = 5)
    (h5 : d > 0) :
    2 * S n + n + 32 ≥ (20 / 3) * (a n + 1) := sorry

end min_fraction_value_l1154_115480


namespace non_formable_triangle_sticks_l1154_115494

theorem non_formable_triangle_sticks 
  (sticks : Fin 8 → ℕ) 
  (h_no_triangle : ∀ (i j k : Fin 8), i < j → j < k → sticks i + sticks j ≤ sticks k) : 
  ∃ (max_length : ℕ), (max_length = sticks (Fin.mk 7 (by norm_num))) ∧ max_length = 21 := 
by 
  sorry

end non_formable_triangle_sticks_l1154_115494


namespace Farrah_total_match_sticks_l1154_115434

def boxes := 4
def matchboxes_per_box := 20
def sticks_per_matchbox := 300

def total_matchboxes : Nat :=
  boxes * matchboxes_per_box

def total_match_sticks : Nat :=
  total_matchboxes * sticks_per_matchbox

theorem Farrah_total_match_sticks : total_match_sticks = 24000 := sorry

end Farrah_total_match_sticks_l1154_115434


namespace positive_integers_satisfy_eq_l1154_115407

theorem positive_integers_satisfy_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + 1 = c! → (a = 2 ∧ b = 1 ∧ c = 3) ∨ (a = 1 ∧ b = 2 ∧ c = 3) :=
by sorry

end positive_integers_satisfy_eq_l1154_115407


namespace solution_set_f_x_leq_m_solution_set_inequality_a_2_l1154_115458

-- Part (I)
theorem solution_set_f_x_leq_m (a m : ℝ) (h : ∀ x : ℝ, |x - a| ≤ m ↔ -1 ≤ x ∧ x ≤ 5) :
  a = 2 ∧ m = 3 :=
sorry

-- Part (II)
theorem solution_set_inequality_a_2 (t : ℝ) (h_t : t ≥ 0) :
  (∀ x : ℝ, |x - 2| + t ≥ |x + 2 * t - 2| ↔ t = 0 ∧ (∀ x : ℝ, True) ∨ t > 0 ∧ ∀ x : ℝ, x ≤ 2 - t / 2) :=
sorry

end solution_set_f_x_leq_m_solution_set_inequality_a_2_l1154_115458


namespace area_triangle_AEB_l1154_115492

theorem area_triangle_AEB :
  ∀ (A B C D F G E : Type)
    (AB AD BC CD : ℝ) 
    (AF BG : ℝ) 
    (triangle_AEB : ℝ),
  (AB = 7) →
  (BC = 4) →
  (CD = 7) →
  (AD = 4) →
  (DF = 2) →
  (GC = 1) →
  (triangle_AEB = 1/2 * 7 * (4 + 16/3)) →
  (triangle_AEB = 98 / 3) :=
by
  intros A B C D F G E AB AD BC CD AF BG triangle_AEB
  sorry

end area_triangle_AEB_l1154_115492


namespace find_p_l1154_115456

theorem find_p (m n p : ℝ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + p) + 5) :
  p = 2 / 5 :=
by sorry

end find_p_l1154_115456


namespace enthalpy_change_correct_l1154_115476

def CC_bond_energy : ℝ := 347
def CO_bond_energy : ℝ := 358
def OH_bond_energy_CH2OH : ℝ := 463
def CO_double_bond_energy_COOH : ℝ := 745
def OH_bond_energy_COOH : ℝ := 467
def OO_double_bond_energy : ℝ := 498
def OH_bond_energy_H2O : ℝ := 467

def total_bond_energy_reactants : ℝ :=
  CC_bond_energy + CO_bond_energy + OH_bond_energy_CH2OH + 1.5 * OO_double_bond_energy

def total_bond_energy_products : ℝ :=
  CO_double_bond_energy_COOH + OH_bond_energy_COOH + OH_bond_energy_H2O

def deltaH : ℝ := total_bond_energy_reactants - total_bond_energy_products

theorem enthalpy_change_correct :
  deltaH = 236 := by
  sorry

end enthalpy_change_correct_l1154_115476


namespace points_collinear_l1154_115401

theorem points_collinear 
  {a b c : ℝ} (h1 : 0 < b) (h2 : b < a) (h3 : c = Real.sqrt (a^2 - b^2))
  (α β : ℝ)
  (P : ℝ × ℝ) (hP : P = (a^2 / c, 0)) 
  (A : ℝ × ℝ) (hA : A = (a * Real.cos α, b * Real.sin α)) 
  (B : ℝ × ℝ) (hB : B = (a * Real.cos β, b * Real.sin β)) 
  (Q : ℝ × ℝ) (hQ : Q = (a * Real.cos α, -b * Real.sin α)) 
  (F : ℝ × ℝ) (hF : F = (c, 0))
  (line_through_F : (A.1 - F.1) * (B.2 - F.2) = (A.2 - F.2) * (B.1 - F.1)) :
  ∃ (k : ℝ), k * (Q.1 - P.1) = Q.2 - P.2 ∧ k * (B.1 - P.1) = B.2 - P.2 :=
by {
  sorry
}

end points_collinear_l1154_115401


namespace angles_cosine_sum_l1154_115454

theorem angles_cosine_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1)
  (h2 : Real.cos A + Real.cos B = 0) :
  12 * Real.cos (2 * A) + 4 * Real.cos (2 * B) = 8 :=
sorry

end angles_cosine_sum_l1154_115454


namespace total_distance_combined_l1154_115478

/-- The conditions for the problem
Each car has 50 liters of fuel.
Car U has a fuel efficiency of 20 liters per 100 kilometers.
Car V has a fuel efficiency of 25 liters per 100 kilometers.
Car W has a fuel efficiency of 5 liters per 100 kilometers.
Car X has a fuel efficiency of 10 liters per 100 kilometers.
-/
theorem total_distance_combined (fuel_U fuel_V fuel_W fuel_X : ℕ) (eff_U eff_V eff_W eff_X : ℕ) (fuel : ℕ)
  (hU : fuel_U = 50) (hV : fuel_V = 50) (hW : fuel_W = 50) (hX : fuel_X = 50)
  (eU : eff_U = 20) (eV : eff_V = 25) (eW : eff_W = 5) (eX : eff_X = 10) :
  (fuel_U * 100 / eff_U) + (fuel_V * 100 / eff_V) + (fuel_W * 100 / eff_W) + (fuel_X * 100 / eff_X) = 1950 := by 
  sorry

end total_distance_combined_l1154_115478


namespace problem_solution_l1154_115440

noncomputable def length_segment_AB : ℝ :=
  let k : ℝ := 1 -- derived from 3k - 3 = 0
  let A : ℝ × ℝ := (0, k) -- point (0, k)
  let C : ℝ × ℝ := (3, -1) -- center of the circle
  let r : ℝ := 1 -- radius of the circle
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) -- distance formula
  Real.sqrt (AC^2 - r^2)

theorem problem_solution :
  length_segment_AB = 2 * Real.sqrt 3 :=
by
  sorry

end problem_solution_l1154_115440


namespace bread_baked_on_monday_l1154_115486

def loaves_wednesday : ℕ := 5
def loaves_thursday : ℕ := 7
def loaves_friday : ℕ := 10
def loaves_saturday : ℕ := 14
def loaves_sunday : ℕ := 19

def increment (n m : ℕ) : ℕ := m - n

theorem bread_baked_on_monday : 
  increment loaves_wednesday loaves_thursday = 2 →
  increment loaves_thursday loaves_friday = 3 →
  increment loaves_friday loaves_saturday = 4 →
  increment loaves_saturday loaves_sunday = 5 →
  loaves_sunday + 6 = 25 :=
by 
  sorry

end bread_baked_on_monday_l1154_115486


namespace equivalence_of_complements_union_l1154_115472

open Set

-- Definitions as per the conditions
def U : Set ℝ := univ
def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 5 }
def complement_U (S : Set ℝ) : Set ℝ := U \ S

-- Mathematical statement to be proved
theorem equivalence_of_complements_union :
  (complement_U M ∪ complement_U N) = { x : ℝ | x < 1 ∨ x ≥ 5 } :=
by
  -- Non-trivial proof, hence skipped with sorry
  sorry

end equivalence_of_complements_union_l1154_115472


namespace sum_of_cubes_l1154_115404

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 17) : a^3 + b^3 = 490 := 
sorry

end sum_of_cubes_l1154_115404


namespace slope_of_line_l1154_115403

variable (s : ℝ) -- real number s

def line1 (x y : ℝ) := x + 3 * y = 9 * s + 4
def line2 (x y : ℝ) := x - 2 * y = 3 * s - 3

theorem slope_of_line (s : ℝ) :
  ∀ (x y : ℝ), (line1 s x y ∧ line2 s x y) → y = (2 / 9) * x + (13 / 9) :=
sorry

end slope_of_line_l1154_115403


namespace proposition_q_false_for_a_lt_2_l1154_115427

theorem proposition_q_false_for_a_lt_2 (a : ℝ) (h : a < 2) : 
  ¬ ∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1 :=
sorry

end proposition_q_false_for_a_lt_2_l1154_115427


namespace second_year_undeclared_fraction_l1154_115442

def total_students := 12

def fraction_first_year : ℚ := 1 / 4
def fraction_second_year : ℚ := 1 / 2
def fraction_third_year : ℚ := 1 / 6
def fraction_fourth_year : ℚ := 1 / 12

def fraction_undeclared_first_year : ℚ := 4 / 5
def fraction_undeclared_second_year : ℚ := 3 / 4
def fraction_undeclared_third_year : ℚ := 1 / 3
def fraction_undeclared_fourth_year : ℚ := 1 / 6

def students_first_year : ℚ := total_students * fraction_first_year
def students_second_year : ℚ := total_students * fraction_second_year
def students_third_year : ℚ := total_students * fraction_third_year
def students_fourth_year : ℚ := total_students * fraction_fourth_year

def undeclared_first_year : ℚ := students_first_year * fraction_undeclared_first_year
def undeclared_second_year : ℚ := students_second_year * fraction_undeclared_second_year
def undeclared_third_year : ℚ := students_third_year * fraction_undeclared_third_year
def undeclared_fourth_year : ℚ := students_fourth_year * fraction_undeclared_fourth_year

theorem second_year_undeclared_fraction :
  (undeclared_second_year / total_students) = 1 / 3 :=
by
  sorry  -- Proof to be provided

end second_year_undeclared_fraction_l1154_115442


namespace min_bills_required_l1154_115423

-- Conditions
def ten_dollar_bills := 13
def five_dollar_bills := 11
def one_dollar_bills := 17
def total_amount := 128

-- Prove that Tim can pay exactly $128 with the minimum number of bills being 16
theorem min_bills_required : (∃ ten five one : ℕ, 
    ten ≤ ten_dollar_bills ∧
    five ≤ five_dollar_bills ∧
    one ≤ one_dollar_bills ∧
    ten * 10 + five * 5 + one = total_amount ∧
    ten + five + one = 16) :=
by
  -- We will skip the proof for now
  sorry

end min_bills_required_l1154_115423


namespace actual_price_of_good_l1154_115485

theorem actual_price_of_good (P : ℝ) (h : 0.684 * P = 6600) : P = 9649.12 :=
sorry

end actual_price_of_good_l1154_115485


namespace remainder_of_5_pow_2023_mod_6_l1154_115482

theorem remainder_of_5_pow_2023_mod_6 : 5^2023 % 6 = 5 := 
by sorry

end remainder_of_5_pow_2023_mod_6_l1154_115482


namespace determine_a_l1154_115426

theorem determine_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x + 1 > 0) ↔ a = 2 := by
  sorry

end determine_a_l1154_115426


namespace p_necessary_not_sufficient_q_l1154_115453

def condition_p (x : ℝ) : Prop := abs x ≤ 2
def condition_q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_not_sufficient_q (x : ℝ) :
  (condition_p x → condition_q x) = false ∧ (condition_q x → condition_p x) = true :=
by
  sorry

end p_necessary_not_sufficient_q_l1154_115453


namespace tangent_normal_at_t1_l1154_115425

noncomputable def curve_param_x (t: ℝ) : ℝ := Real.arcsin (t / Real.sqrt (1 + t^2))
noncomputable def curve_param_y (t: ℝ) : ℝ := Real.arccos (1 / Real.sqrt (1 + t^2))

theorem tangent_normal_at_t1 : 
  curve_param_x 1 = Real.pi / 4 ∧
  curve_param_y 1 = Real.pi / 4 ∧
  ∃ (x y : ℝ), (y = 2*x - Real.pi/4) ∧ (y = -x/2 + 3*Real.pi/8) :=
  sorry

end tangent_normal_at_t1_l1154_115425


namespace sqrt_five_squared_minus_four_squared_eq_three_l1154_115421

theorem sqrt_five_squared_minus_four_squared_eq_three : Real.sqrt (5 ^ 2 - 4 ^ 2) = 3 := by
  sorry

end sqrt_five_squared_minus_four_squared_eq_three_l1154_115421


namespace sum_rows_7_8_pascal_triangle_l1154_115495

theorem sum_rows_7_8_pascal_triangle : (2^7 + 2^8 = 384) :=
by
  sorry

end sum_rows_7_8_pascal_triangle_l1154_115495


namespace edward_dunk_a_clown_tickets_l1154_115432

-- Definitions for conditions
def total_tickets : ℕ := 79
def rides : ℕ := 8
def tickets_per_ride : ℕ := 7

-- Theorem statement
theorem edward_dunk_a_clown_tickets :
  let tickets_spent_on_rides := rides * tickets_per_ride
  let tickets_remaining := total_tickets - tickets_spent_on_rides
  tickets_remaining = 23 :=
by
  sorry

end edward_dunk_a_clown_tickets_l1154_115432


namespace problem1_l1154_115496

theorem problem1 (a b : ℝ) (h1 : (a + b)^2 = 6) (h2 : (a - b)^2 = 2) : a^2 + b^2 = 4 ∧ a * b = 1 := 
by
  sorry

end problem1_l1154_115496


namespace percentage_decrease_revenue_l1154_115474

theorem percentage_decrease_revenue (old_revenue new_revenue : Float) (h_old : old_revenue = 69.0) (h_new : new_revenue = 42.0) : 
  (old_revenue - new_revenue) / old_revenue * 100 = 39.13 := by
  rw [h_old, h_new]
  norm_num
  sorry

end percentage_decrease_revenue_l1154_115474


namespace area_triangle_CMB_eq_105_l1154_115460

noncomputable def area_of_triangle (C M B : ℝ × ℝ) : ℝ :=
  0.5 * (M.1 * B.2 - M.2 * B.1)

theorem area_triangle_CMB_eq_105 :
  let C : ℝ × ℝ := (0, 0)
  let M : ℝ × ℝ := (10, 0)
  let B : ℝ × ℝ := (10, 21)
  area_of_triangle C M B = 105 := by
  sorry

end area_triangle_CMB_eq_105_l1154_115460


namespace cube_volume_from_surface_area_l1154_115449

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l1154_115449
