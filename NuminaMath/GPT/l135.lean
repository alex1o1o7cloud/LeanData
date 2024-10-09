import Mathlib

namespace total_volume_of_cubes_l135_13568

theorem total_volume_of_cubes (Jim_cubes : Nat) (Jim_side_length : Nat) 
    (Laura_cubes : Nat) (Laura_side_length : Nat)
    (h1 : Jim_cubes = 7) (h2 : Jim_side_length = 3) 
    (h3 : Laura_cubes = 4) (h4 : Laura_side_length = 4) : 
    (Jim_cubes * Jim_side_length^3 + Laura_cubes * Laura_side_length^3 = 445) :=
by
  sorry

end total_volume_of_cubes_l135_13568


namespace problem_statement_l135_13538

theorem problem_statement (a b : ℝ) (h : a > b) : a - 1 > b - 1 :=
sorry

end problem_statement_l135_13538


namespace rectangle_dimensions_l135_13537

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : 2 * l + 2 * w = 150) 
  (h2 : l = w + 15) : 
  w = 30 ∧ l = 45 := 
  by 
  sorry

end rectangle_dimensions_l135_13537


namespace find_largest_number_l135_13566

theorem find_largest_number (w x y z : ℕ) 
  (h1 : w + x + y = 190) 
  (h2 : w + x + z = 210) 
  (h3 : w + y + z = 220) 
  (h4 : x + y + z = 235) : 
  max (max w x) (max y z) = 95 := 
sorry

end find_largest_number_l135_13566


namespace remainder_of_square_l135_13558

variable (N X : Set ℤ)
variable (k : ℤ)

/-- Given any n in set N and any x in set X, where dividing n by x gives a remainder of 3,
prove that the remainder of n^2 divided by x is 9 mod x. -/
theorem remainder_of_square (n x : ℤ) (hn : n ∈ N) (hx : x ∈ X)
  (h : ∃ k, n = k * x + 3) : (n^2) % x = 9 % x :=
by
  sorry

end remainder_of_square_l135_13558


namespace max_product_of_sum_2020_l135_13553

/--
  Prove that the maximum product of two integers whose sum is 2020 is 1020100.
-/
theorem max_product_of_sum_2020 : 
  ∃ x : ℤ, (x + (2020 - x) = 2020) ∧ (x * (2020 - x) = 1020100) :=
by
  sorry

end max_product_of_sum_2020_l135_13553


namespace ral_current_age_l135_13507

variable (ral suri : ℕ)

-- Conditions
axiom age_relation : ral = 3 * suri
axiom suri_future_age : suri + 3 = 16

-- Statement
theorem ral_current_age : ral = 39 := by
  sorry

end ral_current_age_l135_13507


namespace total_visitors_over_two_days_l135_13570

-- Definitions of the conditions
def visitors_on_Saturday : ℕ := 200
def additional_visitors_on_Sunday : ℕ := 40

-- Statement of the problem
theorem total_visitors_over_two_days :
  let visitors_on_Sunday := visitors_on_Saturday + additional_visitors_on_Sunday
  let total_visitors := visitors_on_Saturday + visitors_on_Sunday
  total_visitors = 440 :=
by
  let visitors_on_Sunday := visitors_on_Saturday + additional_visitors_on_Sunday
  let total_visitors := visitors_on_Saturday + visitors_on_Sunday
  sorry

end total_visitors_over_two_days_l135_13570


namespace amanda_speed_l135_13533

-- Defining the conditions
def distance : ℝ := 6 -- 6 miles
def time : ℝ := 3 -- 3 hours

-- Stating the question with the conditions and the correct answer
theorem amanda_speed : (distance / time) = 2 :=
by 
  -- the proof is skipped as instructed
  sorry

end amanda_speed_l135_13533


namespace vector_dot_product_parallel_l135_13595

theorem vector_dot_product_parallel (m : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (m, -4))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (a.1 * b.1 + a.2 * b.2) = -10 := by
  sorry

end vector_dot_product_parallel_l135_13595


namespace domain_sqrt_l135_13547

noncomputable def domain_of_function := {x : ℝ | x ≥ 0 ∧ x - 1 ≥ 0}

theorem domain_sqrt : domain_of_function = {x : ℝ | 1 ≤ x} := by {
  sorry
}

end domain_sqrt_l135_13547


namespace handshake_count_l135_13522

def total_employees : ℕ := 50
def dept_X : ℕ := 30
def dept_Y : ℕ := 20
def handshakes_between_departments : ℕ := dept_X * dept_Y

theorem handshake_count : handshakes_between_departments = 600 :=
by
  sorry

end handshake_count_l135_13522


namespace find_profits_maximize_profit_week3_l135_13541

-- Defining the conditions of the problems
def week1_sales_A := 10
def week1_sales_B := 12
def week1_profit := 2000

def week2_sales_A := 20
def week2_sales_B := 15
def week2_profit := 3100

def total_sales_week3 := 25

-- Condition: Sales of type B exceed sales of type A but do not exceed twice the sales of type A
def sales_condition (x : ℕ) := (total_sales_week3 - x) > x ∧ (total_sales_week3 - x) ≤ 2 * x

-- Define the profits for types A and B
def profit_A (a b : ℕ) := week1_sales_A * a + week1_sales_B * b = week1_profit
def profit_B (a b : ℕ) := week2_sales_A * a + week2_sales_B * b = week2_profit

-- Define the profit function for week 3
def profit_week3 (a b x : ℕ) := a * x + b * (total_sales_week3 - x)

theorem find_profits : ∃ a b, profit_A a b ∧ profit_B a b :=
by
  use 80, 100
  sorry

theorem maximize_profit_week3 : 
  ∃ x y, 
  sales_condition x ∧ 
  x + y = total_sales_week3 ∧ 
  profit_week3 80 100 x = 2320 :=
by
  use 9, 16
  sorry

end find_profits_maximize_profit_week3_l135_13541


namespace gail_working_hours_x_l135_13536

theorem gail_working_hours_x (x : ℕ) (hx : x < 12) : 
  let hours_am := 12 - x
  let hours_pm := x
  hours_am + hours_pm = 12 := 
by {
  sorry
}

end gail_working_hours_x_l135_13536


namespace find_f_neg2_l135_13556

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - 3 else -(2^(-x) - 3)

theorem find_f_neg2 : f (-2) = -1 :=
sorry

end find_f_neg2_l135_13556


namespace smallest_n_for_modulo_eq_l135_13512

theorem smallest_n_for_modulo_eq :
  ∃ (n : ℕ), (3^n % 4 = n^3 % 4) ∧ (∀ m : ℕ, m < n → 3^m % 4 ≠ m^3 % 4) ∧ n = 7 :=
by
  sorry

end smallest_n_for_modulo_eq_l135_13512


namespace contrapositive_l135_13589

theorem contrapositive (q p : Prop) (h : q → p) : ¬p → ¬q :=
by
  -- Proof will be filled in later.
  sorry

end contrapositive_l135_13589


namespace find_all_solutions_l135_13578

def is_solution (f : ℕ → ℝ) : Prop :=
  (∀ n ≥ 1, f (n + 1) ≥ f n) ∧
  (∀ m n, Nat.gcd m n = 1 → f (m * n) = f m * f n)

theorem find_all_solutions :
  ∀ f : ℕ → ℝ, is_solution f →
    (∀ n, f n = 0) ∨ (∃ a ≥ 0, ∀ n, f n = n ^ a) :=
sorry

end find_all_solutions_l135_13578


namespace heating_time_correct_l135_13502

def initial_temp : ℤ := 20

def desired_temp : ℤ := 100

def heating_rate : ℤ := 5

def time_to_heat (initial desired rate : ℤ) : ℤ :=
  (desired - initial) / rate

theorem heating_time_correct :
  time_to_heat initial_temp desired_temp heating_rate = 16 :=
by
  sorry

end heating_time_correct_l135_13502


namespace problem1_problem2_l135_13517

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem problem1 (m : ℝ) (h₀ : m > 3) (h₁ : ∃ m, (1/2) * (((m - 1) / 2) - (-(m + 1) / 2) + 3) * (m - 3) = 7 / 2) : m = 4 := by
  sorry

theorem problem2 (a : ℝ) (h₂ : ∃ x, (0 ≤ x ∧ x ≤ 2) ∧ f x ≥ abs (a - 3)) : -2 ≤ a ∧ a ≤ 8 := by
  sorry

end problem1_problem2_l135_13517


namespace larger_number_is_25_l135_13521

-- Let x and y be real numbers, with x being the larger number
variables (x y : ℝ)

-- The sum of the two numbers is 45
axiom sum_eq_45 : x + y = 45

-- The difference of the two numbers is 5
axiom diff_eq_5 : x - y = 5

-- We need to prove that the larger number x is 25
theorem larger_number_is_25 : x = 25 :=
by
  sorry

end larger_number_is_25_l135_13521


namespace even_three_digit_numbers_l135_13534

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l135_13534


namespace no_infinite_arithmetic_progression_divisible_l135_13590

-- Definitions based on the given condition
def is_arithmetic_progression (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

def product_divisible_by_sum (a : ℕ → ℕ) (n : ℕ) : Prop :=
(a n * a (n+1) * a (n+2) * a (n+3) * a (n+4) * a (n+5) * a (n+6) * a (n+7) * a (n+8) * a (n+9)) %
(a n + a (n+1) + a (n+2) + a (n+3) + a (n+4) + a (n+5) + a (n+6) + a (n+7) + a (n+8) + a (n+9)) = 0

-- Final statement to be proven
theorem no_infinite_arithmetic_progression_divisible :
  ¬ ∃ (a : ℕ → ℕ), is_arithmetic_progression a ∧ ∀ n : ℕ, product_divisible_by_sum a n := 
sorry

end no_infinite_arithmetic_progression_divisible_l135_13590


namespace probability_single_shot_l135_13505

-- Define the event and probability given
def event_A := "shooter hits the target at least once out of three shots"
def probability_event_A : ℝ := 0.875

-- The probability of missing in one shot is q, and missing all three is q^3, 
-- which leads to hitting at least once being 1 - q^3
theorem probability_single_shot (q : ℝ) (h : 1 - q^3 = 0.875) : 1 - q = 0.5 :=
by
  sorry

end probability_single_shot_l135_13505


namespace max_distance_curve_line_l135_13542

noncomputable def curve_param_x (θ : ℝ) : ℝ := 1 + Real.cos θ
noncomputable def curve_param_y (θ : ℝ) : ℝ := Real.sin θ
noncomputable def line (x y : ℝ) : Prop := x + y + 2 = 0

theorem max_distance_curve_line 
  (θ : ℝ) 
  (x := curve_param_x θ) 
  (y := curve_param_y θ) :
  ∃ (d : ℝ), 
    (∀ t : ℝ, curve_param_x t = x ∧ curve_param_y t = y → d ≤ (abs (x + y + 2)) / Real.sqrt (1^2 + 1^2)) 
    ∧ d = (3 * Real.sqrt 2) / 2 + 1 :=
sorry

end max_distance_curve_line_l135_13542


namespace find_a_b_tangent_line_at_zero_l135_13594

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3
noncomputable def f' (a b x : ℝ) : ℝ := 2 * a * x + b

theorem find_a_b :
  ∃ a b : ℝ, (a ≠ 0) ∧ (∀ x, f' a b x = 2 * x - 8) := 
sorry

noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.sin x + x^2 - 8 * x + 3
noncomputable def g' (x : ℝ) : ℝ := Real.exp x * Real.sin x + Real.exp x * Real.cos x + 2 * x - 8

theorem tangent_line_at_zero :
  g' 0 = -7 ∧ g 0 = 3 ∧ (∀ y, y = 3 + (-7) * x) := 
sorry

end find_a_b_tangent_line_at_zero_l135_13594


namespace Will_old_cards_l135_13515

theorem Will_old_cards (new_cards pages cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : pages = 6) (h3 : cards_per_page = 3) :
  (pages * cards_per_page) - new_cards = 10 :=
by
  sorry

end Will_old_cards_l135_13515


namespace cases_in_1990_is_correct_l135_13599

-- Define the initial and final number of cases.
def initial_cases : ℕ := 600000
def final_cases : ℕ := 200

-- Define the years and time spans.
def year_1970 : ℕ := 1970
def year_1985 : ℕ := 1985
def year_2000 : ℕ := 2000

def span_1970_to_1985 : ℕ := year_1985 - year_1970 -- 15 years
def span_1985_to_2000 : ℕ := year_2000 - year_1985 -- 15 years

-- Define the rate of decrease from 1970 to 1985 as r cases per year.
-- Define the rate of decrease from 1985 to 2000 as (r / 2) cases per year.
def rate_of_decrease_1 (r : ℕ) := r
def rate_of_decrease_2 (r : ℕ) := r / 2

-- Define the intermediate number of cases in 1985.
def cases_in_1985 (r : ℕ) : ℕ := initial_cases - (span_1970_to_1985 * rate_of_decrease_1 r)

-- Define the number of cases in 1990.
def cases_in_1990 (r : ℕ) : ℕ := cases_in_1985 r - (5 * rate_of_decrease_2 r) -- 5 years from 1985 to 1990

-- Total decrease in cases over 30 years.
def total_decrease : ℕ := initial_cases - final_cases

-- Formalize the proof that the number of cases in 1990 is 133,450.
theorem cases_in_1990_is_correct : 
  ∃ (r : ℕ), 15 * rate_of_decrease_1 r + 15 * rate_of_decrease_2 r = total_decrease ∧ cases_in_1990 r = 133450 := 
by {
  sorry
}

end cases_in_1990_is_correct_l135_13599


namespace price_of_book_l135_13560

variables (D B : ℝ)

def younger_brother : ℝ := 10

theorem price_of_book 
  (h1 : D = 1/2 * (B + younger_brother))
  (h2 : B = 1/3 * (D + younger_brother)) : 
  D + B + younger_brother = 24 := 
sorry

end price_of_book_l135_13560


namespace graph_shift_correct_l135_13514

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x) - Real.sqrt 3 * Real.cos (3 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (3 * x)

theorem graph_shift_correct :
  ∀ (x : ℝ), f x = g (x - (5 * Real.pi / 18)) :=
sorry

end graph_shift_correct_l135_13514


namespace ramu_profit_percent_l135_13523

-- Definitions of the given conditions
def usd_to_inr (usd : ℤ) : ℤ := usd * 45 / 10
def eur_to_inr (eur : ℤ) : ℤ := eur * 567 / 100
def jpy_to_inr (jpy : ℤ) : ℤ := jpy * 1667 / 10000

def cost_of_car_in_inr := usd_to_inr 10000
def engine_repair_cost_in_inr := eur_to_inr 3000
def bodywork_repair_cost_in_inr := jpy_to_inr 150000
def total_cost_in_inr := cost_of_car_in_inr + engine_repair_cost_in_inr + bodywork_repair_cost_in_inr

def selling_price_in_inr : ℤ := 80000
def profit_or_loss_in_inr : ℤ := selling_price_in_inr - total_cost_in_inr

-- Profit percent calculation
def profit_percent (profit_or_loss total_cost : ℤ) : ℚ := (profit_or_loss : ℚ) / (total_cost : ℚ) * 100

-- The theorem stating the mathematically equivalent problem
theorem ramu_profit_percent :
  profit_percent profit_or_loss_in_inr total_cost_in_inr = -8.06 := by
  sorry

end ramu_profit_percent_l135_13523


namespace values_of_x_l135_13544

theorem values_of_x (x : ℝ) (h1 : x^2 - 3 * x - 10 < 0) (h2 : 1 < x) : 1 < x ∧ x < 5 := 
sorry

end values_of_x_l135_13544


namespace largest_5_digit_integer_congruent_to_19_mod_26_l135_13573

theorem largest_5_digit_integer_congruent_to_19_mod_26 :
  ∃ n : ℕ, 10000 ≤ 26 * n + 19 ∧ 26 * n + 19 < 100000 ∧ (26 * n + 19 ≡ 19 [MOD 26]) ∧ 26 * n + 19 = 99989 :=
by
  sorry

end largest_5_digit_integer_congruent_to_19_mod_26_l135_13573


namespace root_triple_condition_l135_13592

theorem root_triple_condition (a b c α β : ℝ)
  (h_eq : a * α^2 + b * α + c = 0)
  (h_β_eq : β = 3 * α)
  (h_vieta_sum : α + β = -b / a)
  (h_vieta_product : α * β = c / a) :
  3 * b^2 = 16 * a * c :=
by
  sorry

end root_triple_condition_l135_13592


namespace factorize_x_squared_minus_one_l135_13551

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l135_13551


namespace common_tangent_curves_l135_13597

theorem common_tangent_curves (s t a : ℝ) (e : ℝ) (he : e > 0) :
  (t = (1 / (2 * e)) * s^2) →
  (t = a * Real.log s) →
  (s / e = a / s) →
  a = 1 :=
by
  intro h1 h2 h3
  sorry

end common_tangent_curves_l135_13597


namespace trivia_team_students_l135_13540

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (h_not_picked : not_picked = 9) 
(h_groups : groups = 3) (h_students_per_group : students_per_group = 9) :
    not_picked + (groups * students_per_group) = 36 := by
  sorry

end trivia_team_students_l135_13540


namespace units_digit_47_4_plus_28_4_l135_13508

theorem units_digit_47_4_plus_28_4 (units_digit_47 : Nat := 7) (units_digit_28 : Nat := 8) :
  (47^4 + 28^4) % 10 = 7 :=
by
  sorry

end units_digit_47_4_plus_28_4_l135_13508


namespace xiaoming_grandfather_age_l135_13531

def grandfather_age (x xm_diff : ℕ) :=
  xm_diff = 60 ∧ x > 7 * (x - xm_diff) ∧ x < 70

theorem xiaoming_grandfather_age (x : ℕ) (h_cond : grandfather_age x 60) : x = 69 :=
by
  sorry

end xiaoming_grandfather_age_l135_13531


namespace value_of_x_l135_13571

theorem value_of_x (x y : ℕ) (h1 : x / y = 3) (h2 : y = 25) : x = 75 := by
  sorry

end value_of_x_l135_13571


namespace obtain_26_kg_of_sand_l135_13545

theorem obtain_26_kg_of_sand :
  ∃ (x y : ℕ), (37 - x = x + 3) ∧ (20 - y = y + 2) ∧ (x + y = 26) := by
  sorry

end obtain_26_kg_of_sand_l135_13545


namespace john_swimming_improvement_l135_13588

theorem john_swimming_improvement :
  let initial_lap_time := 35 / 15 -- initial lap time in minutes per lap
  let current_lap_time := 33 / 18 -- current lap time in minutes per lap
  initial_lap_time - current_lap_time = 1 / 9 := 
by
  -- Definition of initial and current lap times are implied in Lean.
  sorry

end john_swimming_improvement_l135_13588


namespace car_b_speed_l135_13586

theorem car_b_speed
  (v_A v_B : ℝ) (d_A d_B d : ℝ)
  (h1 : v_A = 5 / 3 * v_B)
  (h2 : d_A = v_A * 5)
  (h3 : d_B = v_B * 5)
  (h4 : d = d_A + d_B)
  (h5 : d_A = d / 2 + 25) :
  v_B = 15 := 
sorry

end car_b_speed_l135_13586


namespace veronica_cans_of_food_is_multiple_of_4_l135_13504

-- Definitions of the given conditions
def number_of_water_bottles : ℕ := 20
def number_of_kits : ℕ := 4

-- Proof statement
theorem veronica_cans_of_food_is_multiple_of_4 (F : ℕ) :
  F % number_of_kits = 0 :=
sorry

end veronica_cans_of_food_is_multiple_of_4_l135_13504


namespace total_games_in_season_l135_13587

theorem total_games_in_season :
  let num_teams := 14
  let teams_per_division := 7
  let games_within_division_per_team := 6 * 3
  let games_against_other_division_per_team := 7
  let games_per_team := games_within_division_per_team + games_against_other_division_per_team
  let total_initial_games := games_per_team * num_teams
  let total_games := total_initial_games / 2
  total_games = 175 :=
by
  sorry

end total_games_in_season_l135_13587


namespace zed_to_wyes_l135_13532

theorem zed_to_wyes (value_ex: ℝ) (value_wye: ℝ) (value_zed: ℝ)
  (h1: 2 * value_ex = 29 * value_wye)
  (h2: value_zed = 16 * value_ex) : value_zed = 232 * value_wye := by
  sorry

end zed_to_wyes_l135_13532


namespace simplify_expression_l135_13550

-- Define the conditions as parameters
variable (x y : ℕ)

-- State the theorem with the required conditions and proof goal
theorem simplify_expression (hx : x = 2) (hy : y = 3) :
  (8 * x * y^2) / (6 * x^2 * y) = 2 := by
  -- We'll provide the outline and leave the proof as sorry
  sorry

end simplify_expression_l135_13550


namespace sum_of_coefficients_l135_13581

noncomputable def P (x : ℤ) : ℤ := (x ^ 2 - 3 * x + 1) ^ 100

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

end sum_of_coefficients_l135_13581


namespace find_common_difference_l135_13526

variable (a an Sn d : ℚ)
variable (n : ℕ)

def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def sum_arithmetic_sequence (a : ℚ) (an : ℚ) (n : ℕ) : ℚ :=
  n * (a + an) / 2

theorem find_common_difference
  (h1 : a = 3)
  (h2 : an = 50)
  (h3 : Sn = 318)
  (h4 : an = arithmetic_sequence a d n)
  (h5 : Sn = sum_arithmetic_sequence a an n) :
  d = 47 / 11 :=
by
  sorry

end find_common_difference_l135_13526


namespace solve_equation_l135_13546

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^2 = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by
  sorry

end solve_equation_l135_13546


namespace ratio_of_black_to_blue_l135_13503

universe u

-- Define the types of black and red pens
variables (B R : ℕ)

-- Define the conditions
def condition1 : Prop := 2 + B + R = 12
def condition2 : Prop := R = 2 * B - 2

-- Define the proof statement
theorem ratio_of_black_to_blue (h1 : condition1 B R) (h2 : condition2 B R) : B / 2 = 1 :=
by
  sorry

end ratio_of_black_to_blue_l135_13503


namespace div_condition_for_lcm_l135_13528

theorem div_condition_for_lcm (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h : Nat.lcm (x + 2) (y + 2) - Nat.lcm (x + 1) (y + 1) = Nat.lcm (x + 1) (y + 1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x :=
sorry

end div_condition_for_lcm_l135_13528


namespace ivy_has_20_collectors_dolls_l135_13563

theorem ivy_has_20_collectors_dolls
  (D : ℕ) (I : ℕ) (C : ℕ)
  (h1 : D = 60)
  (h2 : D = 2 * I)
  (h3 : C = 2 * I / 3) 
  : C = 20 :=
by sorry

end ivy_has_20_collectors_dolls_l135_13563


namespace fraction_value_condition_l135_13579

theorem fraction_value_condition (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 :=
sorry

end fraction_value_condition_l135_13579


namespace quadratic_solutions_l135_13513

theorem quadratic_solutions :
  ∀ x : ℝ, (x^2 - 4 * x = 0) → (x = 0 ∨ x = 4) :=
by sorry

end quadratic_solutions_l135_13513


namespace paul_lives_on_story_5_l135_13562

/-- 
Given:
1. Each story is 10 feet tall.
2. Paul makes 3 trips out from and back to his apartment each day.
3. Over a week (7 days), he travels 2100 feet vertically in total.

Prove that the story on which Paul lives \( S \) is 5.
-/
theorem paul_lives_on_story_5 (height_per_story : ℕ)
  (trips_per_day : ℕ)
  (number_of_days : ℕ)
  (total_feet_travelled : ℕ)
  (S : ℕ) :
  height_per_story = 10 → 
  trips_per_day = 3 → 
  number_of_days = 7 → 
  total_feet_travelled = 2100 → 
  2 * height_per_story * trips_per_day * number_of_days * S = total_feet_travelled → 
  S = 5 :=
by
  intros
  sorry

end paul_lives_on_story_5_l135_13562


namespace simplify_expression_l135_13591

theorem simplify_expression : 1 + 3 / (2 + 5 / 6) = 35 / 17 := 
  sorry

end simplify_expression_l135_13591


namespace find_f_three_l135_13574

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end find_f_three_l135_13574


namespace odd_power_preserves_order_l135_13510

theorem odd_power_preserves_order {n : ℤ} (h1 : n > 0) (h2 : n % 2 = 1) :
  ∀ (a b : ℝ), a > b → a^n > b^n :=
by
  sorry

end odd_power_preserves_order_l135_13510


namespace income_of_A_l135_13530

theorem income_of_A (x y : ℝ) (hx₁ : 5 * x - 3 * y = 1600) (hx₂ : 4 * x - 2 * y = 1600) : 
  5 * x = 4000 :=
by
  sorry

end income_of_A_l135_13530


namespace pages_per_inch_l135_13555

theorem pages_per_inch (number_of_books : ℕ) (average_pages_per_book : ℕ) (total_thickness : ℕ) 
                        (H1 : number_of_books = 6)
                        (H2 : average_pages_per_book = 160)
                        (H3 : total_thickness = 12) :
  (number_of_books * average_pages_per_book) / total_thickness = 80 :=
by
  -- Placeholder for proof
  sorry

end pages_per_inch_l135_13555


namespace equilibrium_possible_l135_13559

variables {a b θ : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : (b / 2) < a) (h4 : a ≤ b)

theorem equilibrium_possible :
  θ = 0 ∨ θ = Real.arccos ((b^2 + 2 * a^2) / (3 * a * b)) → 
  (b / 2) < a ∧ a ≤ b ∧ (0 ≤ θ ∧ θ ≤ π) :=
sorry

end equilibrium_possible_l135_13559


namespace second_month_sale_l135_13576

theorem second_month_sale 
  (sale_1st: ℕ) (sale_3rd: ℕ) (sale_4th: ℕ) (sale_5th: ℕ) (sale_6th: ℕ) (avg_sale: ℕ)
  (h1: sale_1st = 5266) (h3: sale_3rd = 5864)
  (h4: sale_4th = 6122) (h5: sale_5th = 6588)
  (h6: sale_6th = 4916) (h_avg: avg_sale = 5750) :
  ∃ sale_2nd, (sale_1st + sale_2nd + sale_3rd + sale_4th + sale_5th + sale_6th) / 6 = avg_sale :=
by
  sorry

end second_month_sale_l135_13576


namespace first_discount_is_20_percent_l135_13577

-- Define the problem parameters
def original_price : ℝ := 200
def final_price : ℝ := 152
def second_discount : ℝ := 0.05

-- Define the function to compute the price after two discounts
def price_after_discounts (first_discount : ℝ) : ℝ := 
  original_price * (1 - first_discount) * (1 - second_discount)

-- Define the statement that we need to prove
theorem first_discount_is_20_percent : 
  ∃ (first_discount : ℝ), price_after_discounts first_discount = final_price ∧ first_discount = 0.20 :=
by
  sorry

end first_discount_is_20_percent_l135_13577


namespace problem_statement_l135_13580

variables {totalBuyers : ℕ}
variables {C M K CM CK MK CMK : ℕ}

-- Given conditions
def conditions (totalBuyers : ℕ) (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : Prop :=
  totalBuyers = 150 ∧
  C = 70 ∧
  M = 60 ∧
  K = 50 ∧
  CM = 25 ∧
  CK = 15 ∧
  MK = 10 ∧
  CMK = 5

-- Number of buyers who purchase at least one mixture
def buyersAtLeastOne (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : ℕ :=
  C + M + K - CM - CK - MK + CMK

-- Number of buyers who purchase none
def buyersNone (totalBuyers : ℕ) (buyersAtLeastOne : ℕ) : ℕ :=
  totalBuyers - buyersAtLeastOne

-- Probability computation
def probabilityNone (totalBuyers : ℕ) (buyersNone : ℕ) : ℚ :=
  buyersNone / totalBuyers

-- Theorem statement
theorem problem_statement : conditions totalBuyers C M K CM CK MK CMK →
  probabilityNone totalBuyers (buyersNone totalBuyers (buyersAtLeastOne C M K CM CK MK CMK)) = 0.1 :=
by
  intros h
  -- Assumptions from the problem
  have h_total : totalBuyers = 150 := h.left
  have hC : C = 70 := h.right.left
  have hM : M = 60 := h.right.right.left
  have hK : K = 50 := h.right.right.right.left
  have hCM : CM = 25 := h.right.right.right.right.left
  have hCK : CK = 15 := h.right.right.right.right.right.left
  have hMK : MK = 10 := h.right.right.right.right.right.right.left
  have hCMK : CMK = 5 := h.right.right.right.right.right.right.right
  sorry

end problem_statement_l135_13580


namespace bmw_cars_sold_l135_13569

def percentage_non_bmw (ford_pct nissan_pct chevrolet_pct : ℕ) : ℕ :=
  ford_pct + nissan_pct + chevrolet_pct

def percentage_bmw (total_pct non_bmw_pct : ℕ) : ℕ :=
  total_pct - non_bmw_pct

def number_of_bmws (total_cars bmw_pct : ℕ) : ℕ :=
  (total_cars * bmw_pct) / 100

theorem bmw_cars_sold (total_cars ford_pct nissan_pct chevrolet_pct : ℕ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 20)
  (h_nissan_pct : nissan_pct = 25)
  (h_chevrolet_pct : chevrolet_pct = 10) :
  number_of_bmws total_cars (percentage_bmw 100 (percentage_non_bmw ford_pct nissan_pct chevrolet_pct)) = 135 := by
  sorry

end bmw_cars_sold_l135_13569


namespace scientific_notation_29150000_l135_13575

theorem scientific_notation_29150000 :
  29150000 = 2.915 * 10^7 := sorry

end scientific_notation_29150000_l135_13575


namespace find_factorial_number_l135_13500

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_factorial_number (n : ℕ) : Prop :=
  ∃ x y z : ℕ, (0 ≤ x ∧ x ≤ 5) ∧
               (0 ≤ y ∧ y ≤ 5) ∧
               (0 ≤ z ∧ z ≤ 5) ∧
               n = 100 * x + 10 * y + z ∧
               n = x.factorial + y.factorial + z.factorial

theorem find_factorial_number : ∃ n, is_three_digit_number n ∧ is_factorial_number n ∧ n = 145 :=
by {
  sorry
}

end find_factorial_number_l135_13500


namespace common_property_of_rectangles_rhombuses_and_squares_l135_13529

-- Definitions of shapes and properties

-- Assume properties P1 = "Diagonals are equal", P2 = "Diagonals bisect each other", 
-- P3 = "Diagonals are perpendicular to each other", and P4 = "Diagonals bisect each other and are equal"

def is_rectangle (R : Type) : Prop := sorry
def is_rhombus (R : Type) : Prop := sorry
def is_square (R : Type) : Prop := sorry

def diagonals_bisect_each_other (R : Type) : Prop := sorry

-- Theorem stating the common property
theorem common_property_of_rectangles_rhombuses_and_squares 
  (R : Type)
  (H_rect : is_rectangle R)
  (H_rhomb : is_rhombus R)
  (H_square : is_square R) :
  diagonals_bisect_each_other R := 
  sorry

end common_property_of_rectangles_rhombuses_and_squares_l135_13529


namespace ordered_pairs_count_l135_13564

theorem ordered_pairs_count : 
  (∃ s : Finset (ℕ × ℕ), (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 6) ∧ s.card = 15) :=
by
  -- The proof would go here
  sorry

end ordered_pairs_count_l135_13564


namespace some_athletes_not_members_honor_society_l135_13535

universe u

variable {U : Type u} -- Assume U is our universe of discourse, e.g., individuals.
variables (Athletes Disciplined HonorSociety : U → Prop)

-- Conditions
def some_athletes_not_disciplined := ∃ x, Athletes x ∧ ¬Disciplined x
def all_honor_society_disciplined := ∀ x, HonorSociety x → Disciplined x

-- Correct Answer
theorem some_athletes_not_members_honor_society :
  some_athletes_not_disciplined Athletes Disciplined →
  all_honor_society_disciplined HonorSociety Disciplined →
  ∃ y, Athletes y ∧ ¬HonorSociety y :=
by
  intros h1 h2
  sorry

end some_athletes_not_members_honor_society_l135_13535


namespace traffic_light_probability_change_l135_13511

theorem traffic_light_probability_change :
  let cycle_time := 100
  let intervals := [(0, 50), (50, 55), (55, 100)]
  let time_changing := [((45, 50), 5), ((50, 55), 5), ((95, 100), 5)]
  let total_change_time := time_changing.map Prod.snd |>.sum
  let probability := (total_change_time : ℚ) / cycle_time
  probability = 3 / 20 := sorry

end traffic_light_probability_change_l135_13511


namespace find_two_numbers_l135_13583

theorem find_two_numbers (x y : ℕ) : 
  (x + y = 20) ∧
  (x * y = 96) ↔ 
  ((x = 12 ∧ y = 8) ∨ (x = 8 ∧ y = 12)) := 
by
  sorry

end find_two_numbers_l135_13583


namespace vectors_form_basis_l135_13527

-- Define the vectors in set B
def e1 : ℝ × ℝ := (-1, 2)
def e2 : ℝ × ℝ := (3, 7)

-- Define a function that checks if two vectors form a basis
def form_basis (v1 v2 : ℝ × ℝ) : Prop :=
  let det := v1.1 * v2.2 - v1.2 * v2.1
  det ≠ 0

-- State the theorem that vectors e1 and e2 form a basis
theorem vectors_form_basis : form_basis e1 e2 :=
by
  -- Add the proof here
  sorry

end vectors_form_basis_l135_13527


namespace population_decrease_is_25_percent_l135_13516

def initial_population : ℕ := 20000
def final_population_first_year : ℕ := initial_population + (initial_population * 25 / 100)
def final_population_second_year : ℕ := 18750

def percentage_decrease (initial final : ℕ) : ℚ :=
  ((initial - final : ℚ) * 100) / initial 

theorem population_decrease_is_25_percent :
  percentage_decrease final_population_first_year final_population_second_year = 25 :=
by
  sorry

end population_decrease_is_25_percent_l135_13516


namespace triangle_to_pentagon_ratio_l135_13518

theorem triangle_to_pentagon_ratio (t p : ℕ) 
  (h1 : 3 * t = 15) 
  (h2 : 5 * p = 15) : (t : ℚ) / (p : ℚ) = 5 / 3 :=
by
  sorry

end triangle_to_pentagon_ratio_l135_13518


namespace find_f_l135_13548

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 1) (h₁ : ∀ x y, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2) : 
  ∀ x, f x = 1 - 2 * x :=
by
  sorry  -- Proof not required

end find_f_l135_13548


namespace work_duration_l135_13554

theorem work_duration (work_rate_x work_rate_y : ℚ) (time_x : ℕ) (total_work : ℚ) :
  work_rate_x = (1 / 20) → 
  work_rate_y = (1 / 12) → 
  time_x = 4 → 
  total_work = 1 →
  ((time_x * work_rate_x) + ((total_work - (time_x * work_rate_x)) / (work_rate_x + work_rate_y))) = 10 := 
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end work_duration_l135_13554


namespace part1_69_part1_97_not_part2_difference_numbers_in_range_l135_13598

def is_difference_number (n : ℕ) : Prop :=
  (n % 7 = 6) ∧ (n % 5 = 4)

theorem part1_69 : is_difference_number 69 :=
sorry

theorem part1_97_not : ¬is_difference_number 97 :=
sorry

theorem part2_difference_numbers_in_range :
  {n : ℕ | is_difference_number n ∧ 500 < n ∧ n < 600} = {524, 559, 594} :=
sorry

end part1_69_part1_97_not_part2_difference_numbers_in_range_l135_13598


namespace goldfish_cost_graph_is_finite_set_of_points_l135_13543

theorem goldfish_cost_graph_is_finite_set_of_points :
  ∀ (n : ℤ), (1 ≤ n ∧ n ≤ 12) → ∃ (C : ℤ), C = 15 * n ∧ ∀ m ≠ n, C ≠ 15 * m :=
by
  -- The proof goes here
  sorry

end goldfish_cost_graph_is_finite_set_of_points_l135_13543


namespace square_side_length_equals_nine_l135_13572

-- Definitions based on the conditions
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * length + 2 * width
def side_length_of_square (perimeter : ℕ) : ℕ := perimeter / 4

-- The theorem we want to prove
theorem square_side_length_equals_nine : 
  side_length_of_square (rectangle_perimeter rectangle_length rectangle_width) = 9 :=
by
  -- proof goes here
  sorry

end square_side_length_equals_nine_l135_13572


namespace sets_are_equal_l135_13557

def setA : Set ℤ := {x | ∃ a b : ℤ, x = 12 * a + 8 * b}
def setB : Set ℤ := {y | ∃ c d : ℤ, y = 20 * c + 16 * d}

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l135_13557


namespace rectangle_area_l135_13524

variable (a b : ℝ)

-- Given conditions
axiom h1 : (a + b)^2 = 16 
axiom h2 : (a - b)^2 = 4

-- Objective: Prove that the area of the rectangle ab equals 3
theorem rectangle_area : a * b = 3 := by
  sorry

end rectangle_area_l135_13524


namespace sufficient_not_necessary_condition_l135_13525

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x^2 > 1 → 1 / x < 1) ∧ (¬(1 / x < 1 → x^2 > 1)) :=
by sorry

end sufficient_not_necessary_condition_l135_13525


namespace trapezoid_not_isosceles_l135_13582

noncomputable def is_trapezoid (BC AD AC : ℝ) : Prop :=
BC = 3 ∧ AD = 4 ∧ AC = 6

def is_isosceles_trapezoid_not_possible (BC AD AC : ℝ) : Prop :=
is_trapezoid BC AD AC → ¬(BC = AD)

theorem trapezoid_not_isosceles (BC AD AC : ℝ) :
  is_isosceles_trapezoid_not_possible BC AD AC :=
sorry

end trapezoid_not_isosceles_l135_13582


namespace min_value_ineq_l135_13565

open Real

theorem min_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 2) * (y^2 + 5 * y + 2) * (z^2 + 5 * z + 2) / (x * y * z) ≥ 512 :=
by sorry

noncomputable def optimal_min_value : ℝ := 512

end min_value_ineq_l135_13565


namespace natalie_bushes_needed_l135_13506

theorem natalie_bushes_needed (b c p : ℕ) 
  (h1 : ∀ b, b * 10 = c) 
  (h2 : ∀ c, c * 2 = p)
  (target_p : p = 36) :
  ∃ b, b * 10 ≥ 72 :=
by
  sorry

end natalie_bushes_needed_l135_13506


namespace find_integer_l135_13567

-- Definition of the given conditions
def conditions (x : ℤ) (r : ℤ) : Prop :=
  (0 ≤ r ∧ r < 7) ∧ ((x - 77) * 8 = 259 + r)

-- Statement of the theorem to be proved
theorem find_integer : ∃ x : ℤ, ∃ r : ℤ, conditions x r ∧ (x = 110) :=
by
  sorry

end find_integer_l135_13567


namespace min_max_x_l135_13539

theorem min_max_x (n : ℕ) (hn : 0 < n) (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = n * x + n * y) : 
  n + 1 ≤ x ∧ x ≤ n * (n + 1) :=
by {
  sorry  -- Proof goes here
}

end min_max_x_l135_13539


namespace sum_equals_120_l135_13519

def rectangular_parallelepiped := (3, 4, 5)

def face_dimensions : List (ℕ × ℕ) := [(4, 5), (3, 5), (3, 4)]

def number_assignment (d : ℕ × ℕ) : ℕ :=
  if d = (4, 5) then 9
  else if d = (3, 5) then 8
  else if d = (3, 4) then 5
  else 0

def sum_checkerboard_ring_one_width (rect_dims : ℕ × ℕ × ℕ) (number_assignment : ℕ × ℕ → ℕ) : ℕ :=
  let (x, y, z) := rect_dims
  let l1 := number_assignment (4, 5) * 2 * (4 * 5)
  let l2 := number_assignment (3, 5) * 2 * (3 * 5)
  let l3 := number_assignment (3, 4) * 2 * (3 * 4) 
  l1 + l2 + l3

theorem sum_equals_120 : ∀ rect_dims number_assignment,
  rect_dims = rectangular_parallelepiped → sum_checkerboard_ring_one_width rect_dims number_assignment = 720 := sorry

end sum_equals_120_l135_13519


namespace gcd_m_n_eq_one_l135_13520

/-- Mathematical definitions of m and n. --/
def m : ℕ := 123^2 + 235^2 + 347^2
def n : ℕ := 122^2 + 234^2 + 348^2

/-- Listing the conditions and deriving the result that gcd(m, n) = 1. --/
theorem gcd_m_n_eq_one : gcd m n = 1 :=
by sorry

end gcd_m_n_eq_one_l135_13520


namespace marks_chemistry_l135_13584

-- Definitions based on conditions
def marks_english : ℕ := 96
def marks_math : ℕ := 98
def marks_physics : ℕ := 99
def marks_biology : ℕ := 98
def average_marks : ℝ := 98.2
def num_subjects : ℕ := 5

-- Statement to prove
theorem marks_chemistry :
  ((marks_english + marks_math + marks_physics + marks_biology : ℕ) + (x : ℕ)) / num_subjects = average_marks →
  x = 100 :=
by
  sorry

end marks_chemistry_l135_13584


namespace max_f_on_interval_l135_13561

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + (Real.sqrt 3) * Real.sin x * Real.cos x

theorem max_f_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ ∀ y ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f y ≤ f x ∧ f x = 3 / 2 :=
  sorry

end max_f_on_interval_l135_13561


namespace geometric_sequence_value_l135_13593

theorem geometric_sequence_value (a : ℝ) (h_pos : 0 < a) 
    (h_geom1 : ∃ r, 25 * r = a)
    (h_geom2 : ∃ r, a * r = 7 / 9) : 
    a = 5 * Real.sqrt 7 / 3 :=
by
  sorry

end geometric_sequence_value_l135_13593


namespace domain_of_log_sqrt_l135_13585

noncomputable def domain_of_function := {x : ℝ | (2 * x - 1 > 0) ∧ (2 * x - 1 ≠ 1) ∧ (3 * x - 2 > 0)}

theorem domain_of_log_sqrt : domain_of_function = {x : ℝ | (2 / 3 < x ∧ x < 1) ∨ (1 < x)} :=
by sorry

end domain_of_log_sqrt_l135_13585


namespace chameleons_to_blue_l135_13596

-- Define a function that simulates the biting between chameleons and their resulting color changes
def color_transition (color_biter : ℕ) (color_bitten : ℕ) : ℕ :=
  if color_bitten = 1 then color_biter + 1
  else if color_bitten = 2 then color_biter + 2
  else if color_bitten = 3 then color_biter + 3
  else if color_bitten = 4 then color_biter + 4
  else 5  -- Once it reaches color 5 (blue), it remains blue.

-- Define the main theorem statement that given 5 red chameleons, all can be turned to blue.
theorem chameleons_to_blue : ∀ (red_chameleons : ℕ), red_chameleons = 5 → 
  ∃ (sequence_of_bites : ℕ → (ℕ × ℕ)), (∀ (c : ℕ), c < 5 → color_transition c (sequence_of_bites c).fst = 5) :=
by sorry

end chameleons_to_blue_l135_13596


namespace sum_powers_l135_13549

open Complex

theorem sum_powers (ω : ℂ) (h₁ : ω^5 = 1) (h₂ : ω ≠ 1) : 
  ω^10 + ω^12 + ω^14 + ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 = 1 := sorry

end sum_powers_l135_13549


namespace fraction_power_l135_13509

theorem fraction_power : (2 / 5 : ℚ) ^ 3 = 8 / 125 := by
  sorry

end fraction_power_l135_13509


namespace count_bottom_right_arrows_l135_13552

/-!
# Problem Statement
Each blank cell on the edge is to be filled with an arrow. The number in each square indicates the number of arrows pointing to that number. The arrows can point in the following directions: up, down, left, right, top-left, top-right, bottom-left, and bottom-right. Each arrow must point to a number. Figure 3 is provided and based on this, determine the number of arrows pointing to the bottom-right direction.
-/

def bottom_right_arrows_count : Nat :=
  2

theorem count_bottom_right_arrows :
  bottom_right_arrows_count = 2 :=
by
  sorry

end count_bottom_right_arrows_l135_13552


namespace part1_part2_part3_l135_13501

-- Part 1
theorem part1 (x : ℝ) :
  (2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4) ↔ x = 2 :=
sorry

-- Part 2
theorem part2 (x : ℤ) :
  (x - 1 / 4 < 1 ∧ 4 + 2 * x > -7 * x + 5) ↔ x = 1 :=
sorry

-- Part 3
theorem part3 (m : ℝ) :
  (∀ x, m < x ∧ x <= m + 2 → (x = 3 ∨ x = 2)) ↔ 1 ≤ m ∧ m < 2 :=
sorry

end part1_part2_part3_l135_13501
