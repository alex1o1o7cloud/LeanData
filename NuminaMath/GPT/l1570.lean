import Mathlib

namespace NUMINAMATH_GPT_cos_sum_proof_l1570_157027

theorem cos_sum_proof (x : ℝ) (h : Real.cos (x - (Real.pi / 6)) = Real.sqrt 3 / 3) :
  Real.cos x + Real.cos (x - Real.pi / 3) = 1 := 
sorry

end NUMINAMATH_GPT_cos_sum_proof_l1570_157027


namespace NUMINAMATH_GPT_ella_max_book_price_l1570_157066

/--
Given that Ella needs to buy 20 identical books and her total budget, 
after deducting the $5 entry fee, is $195. Each book has the same 
cost in whole dollars, and an 8% sales tax is applied to the price of each book. 
Prove that the highest possible price per book that Ella can afford is $9.
-/
theorem ella_max_book_price : 
  ∀ (n : ℕ) (B T : ℝ), n = 20 → B = 195 → T = 1.08 → 
  ∃ (p : ℕ), (↑p ≤ B / T / n) → (9 ≤ p) := 
by 
  sorry

end NUMINAMATH_GPT_ella_max_book_price_l1570_157066


namespace NUMINAMATH_GPT_revenue_correct_l1570_157073

def calculate_revenue : Real :=
  let pumpkin_pie_revenue := 4 * 8 * 5
  let custard_pie_revenue := 5 * 6 * 6
  let apple_pie_revenue := 3 * 10 * 4
  let pecan_pie_revenue := 2 * 12 * 7
  let cookie_revenue := 15 * 2
  let red_velvet_revenue := 6 * 8 * 9
  pumpkin_pie_revenue + custard_pie_revenue + apple_pie_revenue + pecan_pie_revenue + cookie_revenue + red_velvet_revenue

theorem revenue_correct : calculate_revenue = 1090 :=
by
  sorry

end NUMINAMATH_GPT_revenue_correct_l1570_157073


namespace NUMINAMATH_GPT_total_pages_in_book_l1570_157025

-- Given conditions
def pages_first_chapter : ℕ := 13
def pages_second_chapter : ℕ := 68

-- The theorem to prove the total number of pages in the book
theorem total_pages_in_book :
  pages_first_chapter + pages_second_chapter = 81 := by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l1570_157025


namespace NUMINAMATH_GPT_amelia_remaining_money_l1570_157024

variable {m b n : ℚ}

theorem amelia_remaining_money (h : (1 / 4) * m = (1 / 2) * n * b) : 
  m - n * b = (1 / 2) * m :=
by
  sorry

end NUMINAMATH_GPT_amelia_remaining_money_l1570_157024


namespace NUMINAMATH_GPT_find_coefficients_l1570_157059

theorem find_coefficients (c d : ℝ)
  (h : ∃ u v : ℝ, u ≠ v ∧ (u^3 + c * u^2 + 10 * u + 4 = 0) ∧ (v^3 + c * v^2 + 10 * v + 4 = 0)
     ∧ (u^3 + d * u^2 + 13 * u + 5 = 0) ∧ (v^3 + d * v^2 + 13 * v + 5 = 0)) :
  (c, d) = (7, 8) :=
by
  sorry

end NUMINAMATH_GPT_find_coefficients_l1570_157059


namespace NUMINAMATH_GPT_commute_time_absolute_difference_l1570_157094

theorem commute_time_absolute_difference 
  (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) :
  |x - y| = 4 :=
by sorry

end NUMINAMATH_GPT_commute_time_absolute_difference_l1570_157094


namespace NUMINAMATH_GPT_simplify_expression_l1570_157090

theorem simplify_expression : (245^2 - 225^2) / 20 = 470 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1570_157090


namespace NUMINAMATH_GPT_least_subtracted_to_divisible_by_10_l1570_157003

theorem least_subtracted_to_divisible_by_10 (n : ℕ) (k : ℕ) (h : n = 724946) (div_cond : (n - k) % 10 = 0) : k = 6 :=
by
  sorry

end NUMINAMATH_GPT_least_subtracted_to_divisible_by_10_l1570_157003


namespace NUMINAMATH_GPT_stock_percent_change_l1570_157088

theorem stock_percent_change (y : ℝ) : 
  let value_after_day1 := 0.85 * y
  let value_after_day2 := 1.25 * value_after_day1
  (value_after_day2 - y) / y * 100 = 6.25 := by
  sorry

end NUMINAMATH_GPT_stock_percent_change_l1570_157088


namespace NUMINAMATH_GPT_april_earnings_l1570_157015

def price_per_rose := 7
def price_per_lily := 5
def initial_roses := 9
def initial_lilies := 6
def remaining_roses := 4
def remaining_lilies := 2

def total_roses_sold := initial_roses - remaining_roses
def total_lilies_sold := initial_lilies - remaining_lilies

def total_earnings := (total_roses_sold * price_per_rose) + (total_lilies_sold * price_per_lily)

theorem april_earnings : total_earnings = 55 := by
  sorry

end NUMINAMATH_GPT_april_earnings_l1570_157015


namespace NUMINAMATH_GPT_remainder_of_k_div_11_l1570_157076

theorem remainder_of_k_div_11 {k : ℕ} (hk1 : k % 5 = 2) (hk2 : k % 6 = 5)
  (hk3 : 0 ≤ k % 7 ∧ k % 7 < 7) (hk4 : k < 38) : (k % 11) = 6 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_k_div_11_l1570_157076


namespace NUMINAMATH_GPT_inequality_solution_l1570_157017

noncomputable def solution_set : Set ℝ :=
  {x | -4 < x ∧ x < (17 - Real.sqrt 201) / 4} ∪ {x | (17 + Real.sqrt 201) / 4 < x ∧ x < 2 / 3}

theorem inequality_solution (x : ℝ) (h1 : x ≠ -4) (h2 : x ≠ 2 / 3) :
  (2 * x - 3) / (x + 4) > (4 * x + 1) / (3 * x - 2) ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1570_157017


namespace NUMINAMATH_GPT_brownie_pieces_count_l1570_157080

theorem brownie_pieces_count :
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := tray_length * tray_width
  let brownie_area := brownie_length * brownie_width
  let pieces_count := tray_area / brownie_area
  pieces_count = 80 :=
by
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := 24 * 20
  let brownie_area := 3 * 2
  let pieces_count := tray_area / brownie_area
  have h1 : tray_length * tray_width = 480 := by norm_num
  have h2 : brownie_length * brownie_width = 6 := by norm_num
  have h3 : pieces_count = 80 := by norm_num
  exact h3

end NUMINAMATH_GPT_brownie_pieces_count_l1570_157080


namespace NUMINAMATH_GPT_popularity_order_is_correct_l1570_157034

noncomputable def fraction_liking_dodgeball := (13 : ℚ) / 40
noncomputable def fraction_liking_karaoke := (9 : ℚ) / 30
noncomputable def fraction_liking_magicshow := (17 : ℚ) / 60
noncomputable def fraction_liking_quizbowl := (23 : ℚ) / 120

theorem popularity_order_is_correct :
  (fraction_liking_dodgeball ≥ fraction_liking_karaoke) ∧
  (fraction_liking_karaoke ≥ fraction_liking_magicshow) ∧
  (fraction_liking_magicshow ≥ fraction_liking_quizbowl) ∧
  (fraction_liking_dodgeball ≠ fraction_liking_karaoke) ∧
  (fraction_liking_karaoke ≠ fraction_liking_magicshow) ∧
  (fraction_liking_magicshow ≠ fraction_liking_quizbowl) := by
  sorry

end NUMINAMATH_GPT_popularity_order_is_correct_l1570_157034


namespace NUMINAMATH_GPT_find_sum_a7_a8_l1570_157082

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 q : ℝ), ∀ n : ℕ, a n = a1 * q ^ n

variable (a : ℕ → ℝ)

axiom h_geom : geometric_sequence a
axiom h1 : a 0 + a 1 = 16
axiom h2 : a 2 + a 3 = 32

theorem find_sum_a7_a8 : a 6 + a 7 = 128 :=
sorry

end NUMINAMATH_GPT_find_sum_a7_a8_l1570_157082


namespace NUMINAMATH_GPT_car_meeting_distance_l1570_157063

theorem car_meeting_distance
  (distance_AB : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (midpoint_C : ℝ)
  (meeting_distance_from_C : ℝ) 
  (h1 : distance_AB = 245)
  (h2 : speed_A = 70)
  (h3 : speed_B = 90)
  (h4 : midpoint_C = distance_AB / 2) :
  meeting_distance_from_C = 15.31 := 
sorry

end NUMINAMATH_GPT_car_meeting_distance_l1570_157063


namespace NUMINAMATH_GPT_average_of_last_20_students_l1570_157019

theorem average_of_last_20_students 
  (total_students : ℕ) (first_group_size : ℕ) (second_group_size : ℕ) 
  (total_average : ℕ) (first_group_average : ℕ) (second_group_average : ℕ) 
  (total_students_eq : total_students = 50) 
  (first_group_size_eq : first_group_size = 30)
  (second_group_size_eq : second_group_size = 20)
  (total_average_eq : total_average = 92) 
  (first_group_average_eq : first_group_average = 90) :
  second_group_average = 95 :=
by
  sorry

end NUMINAMATH_GPT_average_of_last_20_students_l1570_157019


namespace NUMINAMATH_GPT_no_integer_solutions_l1570_157071

theorem no_integer_solutions (m n : ℤ) : ¬ (m ^ 3 + 6 * m ^ 2 + 5 * m = 27 * n ^ 3 + 9 * n ^ 2 + 9 * n + 1) :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l1570_157071


namespace NUMINAMATH_GPT_range_of_ϕ_l1570_157020

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (2 * x + ϕ) + 1

theorem range_of_ϕ (ϕ : ℝ) (h1 : abs ϕ ≤ Real.pi / 2) 
    (h2 : ∀ (x : ℝ), -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ϕ > 1) :
  Real.pi / 6 ≤ ϕ ∧ ϕ ≤ Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_range_of_ϕ_l1570_157020


namespace NUMINAMATH_GPT_problem_solution_l1570_157079

-- Define the structure of the dartboard and scoring
structure Dartboard where
  inner_radius : ℝ
  intermediate_radius : ℝ
  outer_radius : ℝ
  regions : List (List ℤ) -- List of lists representing scores in the regions

-- Define the probability calculation function
noncomputable def probability_odd_score (d : Dartboard) : ℚ := sorry

-- Define the specific dartboard with given conditions
def revised_dartboard : Dartboard :=
  { inner_radius := 4.5,
    intermediate_radius := 6.75,
    outer_radius := 9,
    regions := [[3, 2, 2], [2, 1, 1], [1, 1, 3]] }

-- The theorem to prove the solution to the problem
theorem problem_solution : probability_odd_score revised_dartboard = 265 / 855 :=
  sorry

end NUMINAMATH_GPT_problem_solution_l1570_157079


namespace NUMINAMATH_GPT_trader_profit_percentage_l1570_157005

theorem trader_profit_percentage (P : ℝ) (h₀ : 0 ≤ P) : 
  let discount := 0.40
  let increase := 0.80
  let purchase_price := P * (1 - discount)
  let selling_price := purchase_price * (1 + increase)
  let profit := selling_price - P
  (profit / P) * 100 = 8 := 
by
  sorry

end NUMINAMATH_GPT_trader_profit_percentage_l1570_157005


namespace NUMINAMATH_GPT_x_increase_80_percent_l1570_157096

noncomputable def percentage_increase (x1 x2 : ℝ) : ℝ :=
  ((x2 / x1) - 1) * 100

theorem x_increase_80_percent
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 * y1 = x2 * y2)
  (h2 : y2 = (5 / 9) * y1) :
  percentage_increase x1 x2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_x_increase_80_percent_l1570_157096


namespace NUMINAMATH_GPT_team_size_is_nine_l1570_157007

noncomputable def number_of_workers (n x y : ℕ) : ℕ :=
  if 7 * n = (n - 2) * x ∧ 7 * n = (n - 6) * y then n else 0

theorem team_size_is_nine (x y : ℕ) :
  number_of_workers 9 x y = 9 :=
by
  sorry

end NUMINAMATH_GPT_team_size_is_nine_l1570_157007


namespace NUMINAMATH_GPT_book_arrangement_count_l1570_157030

-- Conditions
def num_math_books := 4
def num_history_books := 5

-- The number of arrangements is
def arrangements (n m : Nat) : Nat :=
  let choose_end_books := n * (n - 1)
  let choose_middle_book := (n - 2)
  let remaining_books := (n - 3) + m
  choose_end_books * choose_middle_book * Nat.factorial remaining_books

theorem book_arrangement_count (n m : Nat) (h1 : n = num_math_books) (h2 : m = num_history_books) :
  arrangements n m = 120960 :=
by
  rw [h1, h2, arrangements]
  norm_num
  sorry

end NUMINAMATH_GPT_book_arrangement_count_l1570_157030


namespace NUMINAMATH_GPT_white_seeds_per_slice_l1570_157032

theorem white_seeds_per_slice (W : ℕ) (black_seeds_per_slice : ℕ) (number_of_slices : ℕ) 
(total_seeds : ℕ) (total_black_seeds : ℕ) (total_white_seeds : ℕ) 
(h1 : black_seeds_per_slice = 20)
(h2 : number_of_slices = 40)
(h3 : total_seeds = 1600)
(h4 : total_black_seeds = black_seeds_per_slice * number_of_slices)
(h5 : total_white_seeds = total_seeds - total_black_seeds)
(h6 : W = total_white_seeds / number_of_slices) :
W = 20 :=
by
  sorry

end NUMINAMATH_GPT_white_seeds_per_slice_l1570_157032


namespace NUMINAMATH_GPT_crayons_total_l1570_157051

theorem crayons_total (blue_crayons : ℕ) (red_crayons : ℕ) 
  (H1 : red_crayons = 4 * blue_crayons) (H2 : blue_crayons = 3) : 
  blue_crayons + red_crayons = 15 := 
by
  sorry

end NUMINAMATH_GPT_crayons_total_l1570_157051


namespace NUMINAMATH_GPT_speed_of_river_l1570_157052

theorem speed_of_river (speed_still_water : ℝ) (total_time : ℝ) (total_distance : ℝ) 
  (h_still_water: speed_still_water = 6) 
  (h_total_time: total_time = 1) 
  (h_total_distance: total_distance = 16/3) : 
  ∃ (speed_river : ℝ), speed_river = 2 :=
by 
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_speed_of_river_l1570_157052


namespace NUMINAMATH_GPT_negation_of_proposition_l1570_157095

open Classical

theorem negation_of_proposition :
  (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 + 2 * x + 5 > 0) := by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1570_157095


namespace NUMINAMATH_GPT_tina_work_time_l1570_157068

theorem tina_work_time (T : ℕ) (h1 : ∀ Ann_hours, Ann_hours = 9)
                       (h2 : ∀ Tina_worked_hours, Tina_worked_hours = 8)
                       (h3 : ∀ Ann_worked_hours, Ann_worked_hours = 3)
                       (h4 : (8 : ℚ) / T + (1 : ℚ) / 3 = 1) : T = 12 :=
by
  sorry

end NUMINAMATH_GPT_tina_work_time_l1570_157068


namespace NUMINAMATH_GPT_problem_l1570_157053

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem problem (surj_f : ∀ y, ∃ x, f x = y) 
                (inj_g : ∀ x1 x2, g x1 = g x2 → x1 = x2)
                (f_ge_g : ∀ n, f n ≥ g n) :
  ∀ n, f n = g n := 
by 
  sorry

end NUMINAMATH_GPT_problem_l1570_157053


namespace NUMINAMATH_GPT_mary_blue_marbles_l1570_157038

theorem mary_blue_marbles (dan_blue_marbles mary_blue_marbles : ℕ)
  (h1 : dan_blue_marbles = 5)
  (h2 : mary_blue_marbles = 2 * dan_blue_marbles) : mary_blue_marbles = 10 := 
by
  sorry

end NUMINAMATH_GPT_mary_blue_marbles_l1570_157038


namespace NUMINAMATH_GPT_vector_subtraction_l1570_157000

def a : ℝ × ℝ := (5, 3)
def b : ℝ × ℝ := (1, -2)
def scalar : ℝ := 2

theorem vector_subtraction :
  a.1 - scalar * b.1 = 3 ∧ a.2 - scalar * b.2 = 7 :=
by {
  -- here goes the proof
  sorry
}

end NUMINAMATH_GPT_vector_subtraction_l1570_157000


namespace NUMINAMATH_GPT_hours_in_one_year_l1570_157008

/-- Given that there are 24 hours in a day and 365 days in a year,
    prove that there are 8760 hours in one year. -/
theorem hours_in_one_year (hours_per_day : ℕ) (days_per_year : ℕ) (hours_value : ℕ := 8760) : hours_per_day = 24 → days_per_year = 365 → hours_per_day * days_per_year = hours_value :=
by
  intros
  sorry

end NUMINAMATH_GPT_hours_in_one_year_l1570_157008


namespace NUMINAMATH_GPT_non_positive_sequence_l1570_157040

theorem non_positive_sequence
  (N : ℕ)
  (a : ℕ → ℝ)
  (h₀ : a 0 = 0)
  (hN : a N = 0)
  (h_rec : ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2) :
  ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 := sorry

end NUMINAMATH_GPT_non_positive_sequence_l1570_157040


namespace NUMINAMATH_GPT_boss_contribution_l1570_157012

variable (boss_contrib : ℕ) (todd_contrib : ℕ) (employees_contrib : ℕ)
variable (cost : ℕ) (n_employees : ℕ) (emp_payment : ℕ)
variable (total_payment : ℕ)

-- Conditions
def birthday_gift_conditions :=
  cost = 100 ∧
  todd_contrib = 2 * boss_contrib ∧
  employees_contrib = n_employees * emp_payment ∧
  n_employees = 5 ∧
  emp_payment = 11 ∧
  total_payment = boss_contrib + todd_contrib + employees_contrib

-- The proof goal
theorem boss_contribution
  (h : birthday_gift_conditions boss_contrib todd_contrib employees_contrib cost n_employees emp_payment total_payment) :
  boss_contrib = 15 :=
by
  sorry

end NUMINAMATH_GPT_boss_contribution_l1570_157012


namespace NUMINAMATH_GPT_original_number_l1570_157077

theorem original_number (x : ℝ) (h : x - x / 3 = 36) : x = 54 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l1570_157077


namespace NUMINAMATH_GPT_part1_part2_l1570_157014

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem part1 : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 := by
  sorry

theorem part2 (m : ℝ) : 1 ≤ m →
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f x ≤ f m) ∧ 
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f 1 ≤ f x) →
  f m - f 1 = 1 / 2 →
  m = 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1570_157014


namespace NUMINAMATH_GPT_find_C_l1570_157043

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 700) 
  (h2 : A + C = 300) 
  (h3 : B + C = 600) 
  : C = 200 := sorry

end NUMINAMATH_GPT_find_C_l1570_157043


namespace NUMINAMATH_GPT_unique_prime_solution_l1570_157018

-- Define the variables and properties
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the proof goal
theorem unique_prime_solution (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (hp_pos : 0 < p) (hq_pos : 0 < q) :
  p^2 - q^3 = 1 → (p = 3 ∧ q = 2) :=
by sorry

end NUMINAMATH_GPT_unique_prime_solution_l1570_157018


namespace NUMINAMATH_GPT_mike_marbles_l1570_157047

theorem mike_marbles (original : ℕ) (given : ℕ) (final : ℕ) 
  (h1 : original = 8) 
  (h2 : given = 4)
  (h3 : final = original - given) : 
  final = 4 :=
by sorry

end NUMINAMATH_GPT_mike_marbles_l1570_157047


namespace NUMINAMATH_GPT_blue_pill_cost_is_25_l1570_157085

variable (blue_pill_cost red_pill_cost : ℕ)

-- Clara takes one blue pill and one red pill each day for 10 days.
-- A blue pill costs $2 more than a red pill.
def pill_cost_condition (blue_pill_cost red_pill_cost : ℕ) : Prop :=
  blue_pill_cost = red_pill_cost + 2 ∧
  10 * blue_pill_cost + 10 * red_pill_cost = 480

-- Prove that the cost of one blue pill is $25.
theorem blue_pill_cost_is_25 (h : pill_cost_condition blue_pill_cost red_pill_cost) : blue_pill_cost = 25 :=
  sorry

end NUMINAMATH_GPT_blue_pill_cost_is_25_l1570_157085


namespace NUMINAMATH_GPT_problem_statement_l1570_157065

theorem problem_statement
  (x y : ℝ)
  (h1 : 4 * x + 2 * y = 12)
  (h2 : 2 * x + 4 * y = 20) :
  20 * x^2 + 24 * x * y + 20 * y^2 = 544 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l1570_157065


namespace NUMINAMATH_GPT_find_other_endpoint_l1570_157009

theorem find_other_endpoint (x y : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, (x1 + x2)/2 = 2 ∧ (y1 + y2)/2 = 3 ∧ x1 = -1 ∧ y1 = 7 ∧ x2 = x ∧ y2 = y) → (x = 5 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_other_endpoint_l1570_157009


namespace NUMINAMATH_GPT_rain_probability_l1570_157013

-- Define the probability of rain on any given day, number of trials, and specific number of successful outcomes.
def prob_rain_each_day : ℚ := 1/5
def num_days : ℕ := 10
def num_rainy_days : ℕ := 3

-- Define the binomial probability mass function
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

-- Statement to prove
theorem rain_probability : binomial_prob num_days num_rainy_days prob_rain_each_day = 1966080 / 9765625 :=
by
  sorry

end NUMINAMATH_GPT_rain_probability_l1570_157013


namespace NUMINAMATH_GPT_robbie_weight_l1570_157086

theorem robbie_weight (R P : ℝ) 
  (h1 : P = 4.5 * R - 235)
  (h2 : P = R + 115) :
  R = 100 := 
by 
  sorry

end NUMINAMATH_GPT_robbie_weight_l1570_157086


namespace NUMINAMATH_GPT_not_square_l1570_157006

open Int

theorem not_square (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ k : ℤ, (a^2 : ℤ) + ⌈(4 * a^2 : ℤ) / b⌉ = k^2 :=
by
  sorry

end NUMINAMATH_GPT_not_square_l1570_157006


namespace NUMINAMATH_GPT_KrystianaChargesForSecondFloorRooms_Theorem_l1570_157045

noncomputable def KrystianaChargesForSecondFloorRooms (X : ℝ) : Prop :=
  let costFirstFloor := 3 * 15
  let costThirdFloor := 3 * (2 * 15)
  let totalEarnings := costFirstFloor + 3 * X + costThirdFloor
  totalEarnings = 165 → X = 10

-- This is the statement only. The proof is not included.
theorem KrystianaChargesForSecondFloorRooms_Theorem : KrystianaChargesForSecondFloorRooms 10 :=
sorry

end NUMINAMATH_GPT_KrystianaChargesForSecondFloorRooms_Theorem_l1570_157045


namespace NUMINAMATH_GPT_remainder_of_x50_div_by_x_sub_1_cubed_l1570_157031

theorem remainder_of_x50_div_by_x_sub_1_cubed :
  (x^50 % (x-1)^3) = (1225*x^2 - 2500*x + 1276) :=
sorry

end NUMINAMATH_GPT_remainder_of_x50_div_by_x_sub_1_cubed_l1570_157031


namespace NUMINAMATH_GPT_smallest_reducible_fraction_l1570_157033

theorem smallest_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ d > 1, d ∣ (n - 17) ∧ d ∣ (7 * n + 8)) ∧ n = 144 := by
  sorry

end NUMINAMATH_GPT_smallest_reducible_fraction_l1570_157033


namespace NUMINAMATH_GPT_find_complex_number_l1570_157016

def i := Complex.I
def z := -Complex.I - 1
def complex_equation (z : ℂ) := i * z = 1 - i

theorem find_complex_number : complex_equation z :=
by
  -- skip the proof here
  sorry

end NUMINAMATH_GPT_find_complex_number_l1570_157016


namespace NUMINAMATH_GPT_number_of_first_year_students_to_be_sampled_l1570_157072

-- Definitions based on the conditions
def total_students_in_each_grade (x : ℕ) : List ℕ := [4*x, 5*x, 5*x, 6*x]
def total_undergraduate_students (x : ℕ) : ℕ := 4*x + 5*x + 5*x + 6*x
def sample_size : ℕ := 300
def sampling_fraction (x : ℕ) : ℚ := sample_size / total_undergraduate_students x
def first_year_sampling (x : ℕ) : ℕ := (4*x) * sample_size / total_undergraduate_students x

-- Statement to prove
theorem number_of_first_year_students_to_be_sampled {x : ℕ} (hx_pos : x > 0) :
  first_year_sampling x = 60 := 
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_number_of_first_year_students_to_be_sampled_l1570_157072


namespace NUMINAMATH_GPT_cubic_eq_real_roots_roots_product_eq_neg_nine_l1570_157023

theorem cubic_eq_real_roots :
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 + (3 / 2) * (1 - a) * x1^2 - 3 * a * x1 + b = 0 ∧
    x2^3 + (3 / 2) * (1 - a) * x2^2 - 3 * a * x2 + b = 0 ∧
    x3^3 + (3 / 2) * (1 - a) * x3^2 - 3 * a * x3 + b = 0) :=
sorry

theorem roots_product_eq_neg_nine :
  let a := 1
  let b := 1
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 - 3 * x1 + 1 = 0 ∧
    x2^3 - 3 * x2 + 1 = 0 ∧
    x3^3 - 3 * x3 + 1 = 0 ∧
    (x1^2 - 2 - x2) * (x2^2 - 2 - x3) * (x3^2 - 2 - x1) = -9) :=
sorry

end NUMINAMATH_GPT_cubic_eq_real_roots_roots_product_eq_neg_nine_l1570_157023


namespace NUMINAMATH_GPT_greatest_number_same_remainder_l1570_157078

theorem greatest_number_same_remainder (d : ℕ) :
  d ∣ (57 - 25) ∧ d ∣ (105 - 57) ∧ d ∣ (105 - 25) → d ≤ 16 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_same_remainder_l1570_157078


namespace NUMINAMATH_GPT_number_of_female_democrats_l1570_157092

variables (F M D_f : ℕ)

def total_participants := F + M = 660
def female_democrats := D_f = F / 2
def male_democrats := (F / 2) + (M / 4) = 220

theorem number_of_female_democrats 
  (h1 : total_participants F M) 
  (h2 : female_democrats F D_f) 
  (h3 : male_democrats F M) : 
  D_f = 110 := by
  sorry

end NUMINAMATH_GPT_number_of_female_democrats_l1570_157092


namespace NUMINAMATH_GPT_more_boys_after_initial_l1570_157021

theorem more_boys_after_initial (X Y Z : ℕ) (hX : X = 22) (hY : Y = 35) : Z = Y - X :=
by
  sorry

end NUMINAMATH_GPT_more_boys_after_initial_l1570_157021


namespace NUMINAMATH_GPT_probability_of_green_apples_l1570_157048

def total_apples : ℕ := 8
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def apples_chosen : ℕ := 3
noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_green_apples :
  (binomial green_apples apples_chosen : ℚ) / (binomial total_apples apples_chosen : ℚ) = 1 / 56 :=
  sorry

end NUMINAMATH_GPT_probability_of_green_apples_l1570_157048


namespace NUMINAMATH_GPT_chestnuts_distribution_l1570_157074

theorem chestnuts_distribution:
  ∃ (chestnuts_Alya chestnuts_Valya chestnuts_Galya : ℕ),
    chestnuts_Alya + chestnuts_Valya + chestnuts_Galya = 70 ∧
    4 * chestnuts_Valya = 3 * chestnuts_Alya ∧
    6 * chestnuts_Galya = 7 * chestnuts_Alya ∧
    chestnuts_Alya = 24 ∧
    chestnuts_Valya = 18 ∧
    chestnuts_Galya = 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_chestnuts_distribution_l1570_157074


namespace NUMINAMATH_GPT_largest_angle_in_hexagon_l1570_157004

-- Defining the conditions
variables (A B x y : ℝ)
variables (C D E F : ℝ)
variable (sum_of_angles_in_hexagon : ℝ) 

-- Given conditions
def condition1 : A = 100 := by sorry
def condition2 : B = 120 := by sorry
def condition3 : C = x := by sorry
def condition4 : D = x := by sorry
def condition5 : E = (2 * x + y) / 3 + 30 := by sorry
def condition6 : 100 + 120 + C + D + E + F = 720 := by sorry

-- Statement to prove
theorem largest_angle_in_hexagon :
  ∃ (largest_angle : ℝ), largest_angle = max A (max B (max C (max D (max E F)))) ∧ largest_angle = 147.5 := sorry

end NUMINAMATH_GPT_largest_angle_in_hexagon_l1570_157004


namespace NUMINAMATH_GPT_range_of_m_l1570_157042

theorem range_of_m 
  (h : ∀ x : ℝ, x^2 + m * x + m^2 - 1 > 0) :
  m ∈ (Set.Ioo (-(2 * Real.sqrt 3) / 3) (-(2 * Real.sqrt 3) / 3)).union (Set.Ioi ((2 * Real.sqrt 3) / 3)) := 
sorry

end NUMINAMATH_GPT_range_of_m_l1570_157042


namespace NUMINAMATH_GPT_expand_polynomial_l1570_157075

theorem expand_polynomial (t : ℝ) : (2 * t^3 - 3 * t + 2) * (-3 * t^2 + 3 * t - 5) = 
  -6 * t^5 + 6 * t^4 - t^3 + 3 * t^2 + 21 * t - 10 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l1570_157075


namespace NUMINAMATH_GPT_a_plus_b_value_l1570_157035

noncomputable def find_a_plus_b (a b : ℕ) (h_neq : a ≠ b) (h_pos : 0 < a ∧ 0 < b) (h_eq : a^2 - b^2 = 2018 - 2 * a) : ℕ :=
  a + b

theorem a_plus_b_value {a b : ℕ} (h_neq : a ≠ b) (h_pos : 0 < a ∧ 0 < b) (h_eq : a^2 - b^2 = 2018 - 2 * a) : find_a_plus_b a b h_neq h_pos h_eq = 672 :=
  sorry

end NUMINAMATH_GPT_a_plus_b_value_l1570_157035


namespace NUMINAMATH_GPT_new_equation_incorrect_l1570_157028

-- Definition of a function to change each digit of a number by +1 or -1 randomly.
noncomputable def modify_digit (num : ℕ) : ℕ := sorry

-- Proposition stating the original problem's condition and conclusion.
theorem new_equation_incorrect (a b : ℕ) (c := a + b) (a' b' c' : ℕ)
    (h1 : a' = modify_digit a)
    (h2 : b' = modify_digit b)
    (h3 : c' = modify_digit c) :
    a' + b' ≠ c' :=
sorry

end NUMINAMATH_GPT_new_equation_incorrect_l1570_157028


namespace NUMINAMATH_GPT_ferry_travel_time_l1570_157070

theorem ferry_travel_time:
  ∀ (v_P v_Q : ℝ) (d_P d_Q : ℝ) (t_P t_Q : ℝ),
    v_P = 8 →
    v_Q = v_P + 1 →
    d_Q = 3 * d_P →
    t_Q = t_P + 5 →
    d_P = v_P * t_P →
    d_Q = v_Q * t_Q →
    t_P = 3 := by
  sorry

end NUMINAMATH_GPT_ferry_travel_time_l1570_157070


namespace NUMINAMATH_GPT_units_digit_squares_eq_l1570_157037

theorem units_digit_squares_eq (x y : ℕ) (hx : x % 10 + y % 10 = 10) :
  (x * x) % 10 = (y * y) % 10 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_squares_eq_l1570_157037


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l1570_157036

def on_hyperbola (x y k : ℝ) : Prop := y = k / x

theorem relationship_y1_y2_y3 (y1 y2 y3 k : ℝ) (h1 : on_hyperbola (-5) y1 k) (h2 : on_hyperbola (-1) y2 k) (h3 : on_hyperbola 2 y3 k) (hk : k > 0) :
  y2 < y1 ∧ y1 < y3 :=
sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l1570_157036


namespace NUMINAMATH_GPT_heesu_has_greatest_sum_l1570_157060

theorem heesu_has_greatest_sum :
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  have h1 : Heesu_sum > Sora_sum := by sorry
  have h2 : Heesu_sum > Jiyeon_sum := by sorry
  exact And.intro h1 h2

end NUMINAMATH_GPT_heesu_has_greatest_sum_l1570_157060


namespace NUMINAMATH_GPT_system_solution_l1570_157044

theorem system_solution (x y : ℝ) 
  (h1 : (x^2 + x * y + y^2) / (x^2 - x * y + y^2) = 3) 
  (h2 : x^3 + y^3 = 2) : x = 1 ∧ y = 1 :=
  sorry

end NUMINAMATH_GPT_system_solution_l1570_157044


namespace NUMINAMATH_GPT_fraction_sum_l1570_157067

variable {w x y : ℚ}  -- assuming w, x, and y are rational numbers

theorem fraction_sum (h1 : w / x = 1 / 3) (h2 : w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end NUMINAMATH_GPT_fraction_sum_l1570_157067


namespace NUMINAMATH_GPT_walter_coins_value_l1570_157099

theorem walter_coins_value :
  let pennies : ℕ := 2
  let nickels : ℕ := 2
  let dimes : ℕ := 1
  let quarters : ℕ := 1
  let half_dollars : ℕ := 1
  let penny_value : ℕ := 1
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let quarter_value : ℕ := 25
  let half_dollar_value : ℕ := 50
  (pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value) = 97 := 
sorry

end NUMINAMATH_GPT_walter_coins_value_l1570_157099


namespace NUMINAMATH_GPT_find_floor_at_same_time_l1570_157091

def timeTaya (n : ℕ) : ℕ := 15 * (n - 22)
def timeJenna (n : ℕ) : ℕ := 120 + 3 * (n - 22)

theorem find_floor_at_same_time (n : ℕ) : n = 32 :=
by
  -- The goal is to show that Taya and Jenna arrive at the same floor at the same time
  have ht : 15 * (n - 22) = timeTaya n := rfl
  have hj : 120 + 3 * (n - 22) = timeJenna n := rfl
  -- equate the times
  have h : timeTaya n = timeJenna n := by sorry
  -- solving the equation for n = 32
  sorry

end NUMINAMATH_GPT_find_floor_at_same_time_l1570_157091


namespace NUMINAMATH_GPT_angle_B_triangle_perimeter_l1570_157083

variable {A B C a b c : Real}

-- Definitions and conditions for part 1
def sides_relation (a b c : ℝ) (A : ℝ) : Prop :=
  2 * c = a + 2 * b * Real.cos A

-- Definitions and conditions for part 2
def triangle_area (a b c : ℝ) (B : ℝ) : Prop :=
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3

def side_b_value (b : ℝ) : Prop :=
  b = Real.sqrt 13

-- Theorem statement for part 1 
theorem angle_B (a b c A : ℝ) (h1: sides_relation a b c A) : B = Real.pi / 3 :=
sorry

-- Theorem statement for part 2 
theorem triangle_perimeter (a b c B : ℝ) (h1 : triangle_area a b c B) (h2 : side_b_value b) (h3 : B = Real.pi / 3) : a + b + c = 5 + Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_angle_B_triangle_perimeter_l1570_157083


namespace NUMINAMATH_GPT_rectangle_area_l1570_157039

theorem rectangle_area (l w r: ℝ) (h1 : l = 2 * r) (h2 : w = r) : l * w = 2 * r^2 :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l1570_157039


namespace NUMINAMATH_GPT_problem_statement_l1570_157097

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem problem_statement :
  f (5 * Real.pi / 24) = Real.sqrt 2 ∧
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1570_157097


namespace NUMINAMATH_GPT_working_light_bulbs_count_l1570_157058

def lamps := 60
def bulbs_per_lamp := 7

def fraction_with_2_burnt := 1 / 3
def fraction_with_1_burnt := 1 / 4
def fraction_with_3_burnt := 1 / 5

def lamps_with_2_burnt := fraction_with_2_burnt * lamps
def lamps_with_1_burnt := fraction_with_1_burnt * lamps
def lamps_with_3_burnt := fraction_with_3_burnt * lamps
def lamps_with_all_working := lamps - (lamps_with_2_burnt + lamps_with_1_burnt + lamps_with_3_burnt)

def working_bulbs_from_2_burnt := lamps_with_2_burnt * (bulbs_per_lamp - 2)
def working_bulbs_from_1_burnt := lamps_with_1_burnt * (bulbs_per_lamp - 1)
def working_bulbs_from_3_burnt := lamps_with_3_burnt * (bulbs_per_lamp - 3)
def working_bulbs_from_all_working := lamps_with_all_working * bulbs_per_lamp

def total_working_bulbs := working_bulbs_from_2_burnt + working_bulbs_from_1_burnt + working_bulbs_from_3_burnt + working_bulbs_from_all_working

theorem working_light_bulbs_count : total_working_bulbs = 329 := by
  sorry

end NUMINAMATH_GPT_working_light_bulbs_count_l1570_157058


namespace NUMINAMATH_GPT_range_of_m_l1570_157081

theorem range_of_m (m : ℝ) :
  (m + 4 - 4)*(2 + 2 * m - 4) < 0 → 0 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1570_157081


namespace NUMINAMATH_GPT_product_not_ending_in_1_l1570_157002

theorem product_not_ending_in_1 : ∃ a b : ℕ, 111111 = a * b ∧ (a % 10 ≠ 1) ∧ (b % 10 ≠ 1) := 
sorry

end NUMINAMATH_GPT_product_not_ending_in_1_l1570_157002


namespace NUMINAMATH_GPT_sqrt_subtraction_result_l1570_157049

theorem sqrt_subtraction_result : 
  (Real.sqrt (49 + 36) - Real.sqrt (36 - 0)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_subtraction_result_l1570_157049


namespace NUMINAMATH_GPT_evaluate_expression_l1570_157098

theorem evaluate_expression :
  (↑(2 ^ (6 / 4))) ^ 8 = 4096 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1570_157098


namespace NUMINAMATH_GPT_original_average_weight_l1570_157087

theorem original_average_weight 
  (W : ℝ)
  (h1 : 7 * W + 110 + 60 = 9 * 78) : 
  W = 76 := 
by
  sorry

end NUMINAMATH_GPT_original_average_weight_l1570_157087


namespace NUMINAMATH_GPT_no_common_point_in_all_circles_l1570_157057

variable {Point : Type}
variable {Circle : Type}
variable (center : Circle → Point)
variable (contains : Circle → Point → Prop)

-- Given six circles in the plane
variables (C1 C2 C3 C4 C5 C6 : Circle)

-- Condition: None of the circles contain the center of any other circle
axiom condition_1 : ∀ (C D : Circle), C ≠ D → ¬ contains C (center D)

-- Question: Prove that there does not exist a point P that lies in all six circles
theorem no_common_point_in_all_circles : 
  ¬ ∃ (P : Point), (contains C1 P) ∧ (contains C2 P) ∧ (contains C3 P) ∧ (contains C4 P) ∧ (contains C5 P) ∧ (contains C6 P) :=
sorry

end NUMINAMATH_GPT_no_common_point_in_all_circles_l1570_157057


namespace NUMINAMATH_GPT_median_ratio_within_bounds_l1570_157093

def median_ratio_limits (α : ℝ) (hα : 0 < α ∧ α < π) : Prop :=
  ∀ (s_c s_b : ℝ), s_b = 1 → (1 / 2) ≤ (s_c / s_b) ∧ (s_c / s_b) ≤ 2

theorem median_ratio_within_bounds (α : ℝ) (hα : 0 < α ∧ α < π) : 
  median_ratio_limits α hα :=
by
  sorry

end NUMINAMATH_GPT_median_ratio_within_bounds_l1570_157093


namespace NUMINAMATH_GPT_problem1_problem2_l1570_157054

-- Problem 1: Calculation
theorem problem1 :
  (1:Real) - 1^2 + Real.sqrt 12 + Real.sqrt (4 / 3) = -1 + (8 * Real.sqrt 3) / 3 :=
by
  sorry
  
-- Problem 2: Solve the equation 2x^2 - x - 1 = 0
theorem problem2 (x : Real) :
  (2 * x^2 - x - 1 = 0) → (x = -1/2 ∨ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1570_157054


namespace NUMINAMATH_GPT_base_b_of_200_has_5_digits_l1570_157046

theorem base_b_of_200_has_5_digits : ∃ (b : ℕ), (b^4 ≤ 200) ∧ (200 < b^5) ∧ (b = 3) := by
  sorry

end NUMINAMATH_GPT_base_b_of_200_has_5_digits_l1570_157046


namespace NUMINAMATH_GPT_total_amount_before_brokerage_l1570_157050

variable (A : ℝ)

theorem total_amount_before_brokerage 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 106.25) 
  (h2 : brokerage_rate = 1 / 400) :
  A = 42500 / 399 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_before_brokerage_l1570_157050


namespace NUMINAMATH_GPT_probability_drawing_3_one_color_1_other_l1570_157041

theorem probability_drawing_3_one_color_1_other (black white : ℕ) (total_balls drawn_balls : ℕ) 
    (total_ways : ℕ) (ways_3_black_1_white : ℕ) (ways_1_black_3_white : ℕ) :
    black = 10 → white = 5 → total_balls = 15 → drawn_balls = 4 →
    total_ways = Nat.choose total_balls drawn_balls →
    ways_3_black_1_white = Nat.choose black 3 * Nat.choose white 1 →
    ways_1_black_3_white = Nat.choose black 1 * Nat.choose white 3 →
    (ways_3_black_1_white + ways_1_black_3_white) / total_ways = 140 / 273 := 
by
  intros h_black h_white h_total_balls h_drawn_balls h_total_ways h_ways_3_black_1_white h_ways_1_black_3_white
  -- The proof would go here, but is not required for this task.
  sorry

end NUMINAMATH_GPT_probability_drawing_3_one_color_1_other_l1570_157041


namespace NUMINAMATH_GPT_sin_alpha_l1570_157029

theorem sin_alpha (α : ℝ) (hα : 0 < α ∧ α < π) (hcos : Real.cos (π + α) = 3 / 5) :
  Real.sin α = 4 / 5 :=
sorry

end NUMINAMATH_GPT_sin_alpha_l1570_157029


namespace NUMINAMATH_GPT_probability_adjacent_vertices_dodecagon_l1570_157089

noncomputable def prob_adjacent_vertices_dodecagon : ℚ :=
  let total_vertices := 12
  let favorable_outcomes := 2  -- adjacent vertices per chosen vertex
  let total_outcomes := total_vertices - 1  -- choosing any other vertex
  favorable_outcomes / total_outcomes

theorem probability_adjacent_vertices_dodecagon :
  prob_adjacent_vertices_dodecagon = 2 / 11 := by
  sorry

end NUMINAMATH_GPT_probability_adjacent_vertices_dodecagon_l1570_157089


namespace NUMINAMATH_GPT_at_least_one_nonnegative_l1570_157061

theorem at_least_one_nonnegative (x y z : ℝ) : 
  (x^2 + y + 1/4 ≥ 0) ∨ (y^2 + z + 1/4 ≥ 0) ∨ (z^2 + x + 1/4 ≥ 0) :=
sorry

end NUMINAMATH_GPT_at_least_one_nonnegative_l1570_157061


namespace NUMINAMATH_GPT_solve_fraction_eq_l1570_157062

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 3) 
    (h₃ : 3 / (x - 2) = 6 / (x - 3)) : x = 1 :=
by 
  sorry

end NUMINAMATH_GPT_solve_fraction_eq_l1570_157062


namespace NUMINAMATH_GPT_brendas_age_l1570_157026

theorem brendas_age (A B J : ℝ) 
  (h1 : A = 4 * B)
  (h2 : J = B + 7)
  (h3 : A = J) : 
  B = 7 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_brendas_age_l1570_157026


namespace NUMINAMATH_GPT_permutation_of_digits_l1570_157010

-- Definition of factorial
def fact : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * fact n

-- Given conditions
def n := 8
def n1 := 3
def n2 := 2
def n3 := 1
def n4 := 2

-- Statement
theorem permutation_of_digits :
  fact n / (fact n1 * fact n2 * fact n3 * fact n4) = 1680 :=
by
  sorry

end NUMINAMATH_GPT_permutation_of_digits_l1570_157010


namespace NUMINAMATH_GPT_find_quotient_l1570_157056

theorem find_quotient (divisor remainder dividend : ℕ) (h_divisor : divisor = 24) (h_remainder : remainder = 5) (h_dividend : dividend = 1565) : 
  (dividend - remainder) / divisor = 65 :=
by
  sorry

end NUMINAMATH_GPT_find_quotient_l1570_157056


namespace NUMINAMATH_GPT_mul_65_35_l1570_157069

theorem mul_65_35 : (65 * 35) = 2275 := by
  -- define a and b
  let a := 50
  let b := 15
  -- use the equivalence (a + b) and (a - b)
  have h1 : 65 = a + b := by rfl
  have h2 : 35 = a - b := by rfl
  -- use the difference of squares formula
  have h_diff_squares : (a + b) * (a - b) = a^2 - b^2 := by sorry
  -- calculate each square
  have ha_sq : a^2 = 2500 := by sorry
  have hb_sq : b^2 = 225 := by sorry
  -- combine the results
  have h_result : a^2 - b^2 = 2500 - 225 := by sorry
  -- finish the proof
  have final_result : (65 * 35) = 2275 := by sorry
  exact final_result

end NUMINAMATH_GPT_mul_65_35_l1570_157069


namespace NUMINAMATH_GPT_oldest_child_age_l1570_157055

theorem oldest_child_age 
  (x : ℕ)
  (h1 : (6 + 8 + 10 + x) / 4 = 9)
  (h2 : 6 + 8 + 10 = 24) :
  x = 12 := 
by 
  sorry

end NUMINAMATH_GPT_oldest_child_age_l1570_157055


namespace NUMINAMATH_GPT_crystal_discount_is_50_percent_l1570_157001

noncomputable def discount_percentage_original_prices_and_revenue
  (original_price_cupcake : ℝ)
  (original_price_cookie : ℝ)
  (total_cupcakes_sold : ℕ)
  (total_cookies_sold : ℕ)
  (total_revenue : ℝ)
  (percentage_discount : ℝ) :
  Prop :=
  total_cupcakes_sold * (original_price_cupcake * (1 - percentage_discount / 100)) +
  total_cookies_sold * (original_price_cookie * (1 - percentage_discount / 100)) = total_revenue

theorem crystal_discount_is_50_percent :
  discount_percentage_original_prices_and_revenue 3 2 16 8 32 50 :=
by sorry

end NUMINAMATH_GPT_crystal_discount_is_50_percent_l1570_157001


namespace NUMINAMATH_GPT_H_perimeter_is_44_l1570_157011

-- Defining the dimensions of the rectangles
def vertical_rectangle_length : ℕ := 6
def vertical_rectangle_width : ℕ := 3
def horizontal_rectangle_length : ℕ := 6
def horizontal_rectangle_width : ℕ := 2

-- Defining the perimeter calculations, excluding overlapping parts
def vertical_rectangle_perimeter : ℕ := 2 * vertical_rectangle_length + 2 * vertical_rectangle_width
def horizontal_rectangle_perimeter : ℕ := 2 * horizontal_rectangle_length + 2 * horizontal_rectangle_width

-- Non-overlapping combined perimeter calculation for the 'H'
def H_perimeter : ℕ := 2 * vertical_rectangle_perimeter + horizontal_rectangle_perimeter - 2 * (2 * horizontal_rectangle_width)

-- Main theorem statement
theorem H_perimeter_is_44 : H_perimeter = 44 := by
  -- Provide a proof here
  sorry

end NUMINAMATH_GPT_H_perimeter_is_44_l1570_157011


namespace NUMINAMATH_GPT_cube_of_square_of_third_smallest_prime_is_correct_l1570_157084

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end NUMINAMATH_GPT_cube_of_square_of_third_smallest_prime_is_correct_l1570_157084


namespace NUMINAMATH_GPT_papers_delivered_to_sunday_only_houses_l1570_157064

-- Define the number of houses in the route and the days
def houses_in_route : ℕ := 100
def days_monday_to_saturday : ℕ := 6

-- Define the number of customers that do not get the paper on Sunday
def non_customers_sunday : ℕ := 10
def total_papers_per_week : ℕ := 720

-- Define the required number of papers delivered on Sunday to houses that only get the paper on Sunday
def papers_only_on_sunday : ℕ :=
  total_papers_per_week - (houses_in_route * days_monday_to_saturday) - (houses_in_route - non_customers_sunday)

theorem papers_delivered_to_sunday_only_houses : papers_only_on_sunday = 30 :=
by
  sorry

end NUMINAMATH_GPT_papers_delivered_to_sunday_only_houses_l1570_157064


namespace NUMINAMATH_GPT_swimmer_distance_l1570_157022

noncomputable def effective_speed := 4.4 - 2.5
noncomputable def time := 3.684210526315789
noncomputable def distance := effective_speed * time

theorem swimmer_distance :
  distance = 7 := by
  sorry

end NUMINAMATH_GPT_swimmer_distance_l1570_157022
