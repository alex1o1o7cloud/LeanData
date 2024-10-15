import Mathlib

namespace NUMINAMATH_GPT_boat_fuel_cost_per_hour_l2357_235774

variable (earnings_per_photo : ℕ)
variable (shark_frequency_minutes : ℕ)
variable (hunting_hours : ℕ)
variable (expected_profit : ℕ)

def cost_of_fuel_per_hour (earnings_per_photo shark_frequency_minutes hunting_hours expected_profit : ℕ) : ℕ :=
  sorry

theorem boat_fuel_cost_per_hour
  (h₁ : earnings_per_photo = 15)
  (h₂ : shark_frequency_minutes = 10)
  (h₃ : hunting_hours = 5)
  (h₄ : expected_profit = 200) :
  cost_of_fuel_per_hour earnings_per_photo shark_frequency_minutes hunting_hours expected_profit = 50 :=
  sorry

end NUMINAMATH_GPT_boat_fuel_cost_per_hour_l2357_235774


namespace NUMINAMATH_GPT_number_of_integer_pairs_l2357_235714

theorem number_of_integer_pairs (n : ℕ) : 
  ∃ (count : ℕ), count = 2 * n^2 + 2 * n + 1 ∧ 
  ∀ x y : ℤ, abs x + abs y ≤ n ↔
  count = 2 * n^2 + 2 * n + 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integer_pairs_l2357_235714


namespace NUMINAMATH_GPT_number_of_tires_l2357_235719

theorem number_of_tires (n : ℕ)
  (repair_cost : ℕ → ℝ)
  (sales_tax : ℕ → ℝ)
  (total_cost : ℝ) :
  (∀ t, repair_cost t = 7) →
  (∀ t, sales_tax t = 0.5) →
  (total_cost = n * (repair_cost 0 + sales_tax 0)) →
  total_cost = 30 →
  n = 4 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_tires_l2357_235719


namespace NUMINAMATH_GPT_find_focus_parabola_l2357_235780

theorem find_focus_parabola
  (x y : ℝ) 
  (h₁ : y = 9 * x^2 + 6 * x - 4) :
  ∃ (h k p : ℝ), (x + 1/3)^2 = 1/3 * (y + 5) ∧ 4 * p = 1/3 ∧ h = -1/3 ∧ k = -5 ∧ (h, k + p) = (-1/3, -59/12) :=
sorry

end NUMINAMATH_GPT_find_focus_parabola_l2357_235780


namespace NUMINAMATH_GPT_f_2008th_derivative_at_0_l2357_235783

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4))^6 + (Real.cos (x / 4))^6

theorem f_2008th_derivative_at_0 : (deriv^[2008] f) 0 = 3 / 8 :=
sorry

end NUMINAMATH_GPT_f_2008th_derivative_at_0_l2357_235783


namespace NUMINAMATH_GPT_increase_circumference_l2357_235763

theorem increase_circumference (d1 d2 : ℝ) (increase : ℝ) (P : ℝ) : 
  increase = 2 * Real.pi → 
  P = Real.pi * increase → 
  P = 2 * Real.pi ^ 2 := 
by 
  intros h_increase h_P
  rw [h_P, h_increase]
  sorry

end NUMINAMATH_GPT_increase_circumference_l2357_235763


namespace NUMINAMATH_GPT_simplify_fractions_l2357_235741

theorem simplify_fractions : (360 / 32) * (10 / 240) * (16 / 6) = 10 := by
  sorry

end NUMINAMATH_GPT_simplify_fractions_l2357_235741


namespace NUMINAMATH_GPT_trirectangular_tetrahedron_max_volume_l2357_235767

noncomputable def max_volume_trirectangular_tetrahedron (S : ℝ) : ℝ :=
  S^3 * (Real.sqrt 2 - 1)^3 / 162

theorem trirectangular_tetrahedron_max_volume
  (a b c : ℝ) (H : a > 0 ∧ b > 0 ∧ c > 0)
  (S : ℝ)
  (edge_sum :
    S = a + b + c + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2))
  : ∃ V, V = max_volume_trirectangular_tetrahedron S :=
by
  sorry

end NUMINAMATH_GPT_trirectangular_tetrahedron_max_volume_l2357_235767


namespace NUMINAMATH_GPT_manager_salary_calculation_l2357_235724

theorem manager_salary_calculation :
  let percent_marketers := 0.60
  let salary_marketers := 50000
  let percent_engineers := 0.20
  let salary_engineers := 80000
  let percent_sales_reps := 0.10
  let salary_sales_reps := 70000
  let percent_managers := 0.10
  let total_average_salary := 75000
  let total_contribution := percent_marketers * salary_marketers + percent_engineers * salary_engineers + percent_sales_reps * salary_sales_reps
  let managers_total_contribution := total_average_salary - total_contribution
  let manager_salary := managers_total_contribution / percent_managers
  manager_salary = 220000 :=
by
  sorry

end NUMINAMATH_GPT_manager_salary_calculation_l2357_235724


namespace NUMINAMATH_GPT_sweets_distribution_l2357_235717

theorem sweets_distribution (S : ℕ) (N : ℕ) (h1 : N - 70 > 0) (h2 : S = N * 24) (h3 : S = (N - 70) * 38) : N = 190 :=
by
  sorry

end NUMINAMATH_GPT_sweets_distribution_l2357_235717


namespace NUMINAMATH_GPT_grid_game_winner_l2357_235744

theorem grid_game_winner {m n : ℕ} :
  (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") = (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") := by
  sorry

end NUMINAMATH_GPT_grid_game_winner_l2357_235744


namespace NUMINAMATH_GPT_fraction_of_total_amount_l2357_235745

-- Conditions
variable (p q r : ℕ)
variable (total_amount amount_r : ℕ)
variable (total_amount_eq : total_amount = 6000)
variable (amount_r_eq : amount_r = 2400)

-- Mathematical statement
theorem fraction_of_total_amount :
  amount_r / total_amount = 2 / 5 :=
by
  -- Sorry to skip the proof, as instructed
  sorry

end NUMINAMATH_GPT_fraction_of_total_amount_l2357_235745


namespace NUMINAMATH_GPT_problem_statement_l2357_235761

variables {a b x : ℝ}

theorem problem_statement (h1 : x = a / b + 2) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + 2 * b) / (a - 2 * b) = x / (x - 4) := 
sorry

end NUMINAMATH_GPT_problem_statement_l2357_235761


namespace NUMINAMATH_GPT_percentage_value_l2357_235781

variables {P a b c : ℝ}

theorem percentage_value (h1 : (P / 100) * a = 12) (h2 : (12 / 100) * b = 6) (h3 : c = b / a) : c = P / 24 :=
by
  sorry

end NUMINAMATH_GPT_percentage_value_l2357_235781


namespace NUMINAMATH_GPT_original_number_is_64_l2357_235786

theorem original_number_is_64 (x : ℕ) : 500 + x = 9 * x - 12 → x = 64 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_64_l2357_235786


namespace NUMINAMATH_GPT_machine_production_l2357_235710

theorem machine_production
  (rate_per_minute : ℕ)
  (machines_total : ℕ)
  (production_minute : ℕ)
  (machines_sub : ℕ)
  (time_minutes : ℕ)
  (total_production : ℕ) :
  machines_total * rate_per_minute = production_minute →
  rate_per_minute = production_minute / machines_total →
  machines_sub * rate_per_minute = total_production / time_minutes →
  time_minutes * total_production / time_minutes = 900 :=
by
  sorry

end NUMINAMATH_GPT_machine_production_l2357_235710


namespace NUMINAMATH_GPT_value_op_and_add_10_l2357_235785

def op_and (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem value_op_and_add_10 : op_and 8 5 + 10 = 49 :=
by
  sorry

end NUMINAMATH_GPT_value_op_and_add_10_l2357_235785


namespace NUMINAMATH_GPT_zero_point_interval_l2357_235709

noncomputable def f (x : ℝ) := 6 / x - x ^ 2

theorem zero_point_interval : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_point_interval_l2357_235709


namespace NUMINAMATH_GPT_positive_integers_square_less_than_three_times_l2357_235755

theorem positive_integers_square_less_than_three_times (n : ℕ) (hn : 0 < n) (ineq : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
by sorry

end NUMINAMATH_GPT_positive_integers_square_less_than_three_times_l2357_235755


namespace NUMINAMATH_GPT_sum_of_solutions_eq_zero_l2357_235712

theorem sum_of_solutions_eq_zero (x : ℝ) (h : 6 * x / 30 = 7 / x) :
  (∃ x₁ x₂ : ℝ, x₁^2 = 35 ∧ x₂^2 = 35 ∧ x₁ + x₂ = 0) :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_zero_l2357_235712


namespace NUMINAMATH_GPT_sprouted_percentage_l2357_235754

-- Define the initial conditions
def cherryPits := 80
def saplingsSold := 6
def saplingsLeft := 14

-- Define the calculation of the total saplings that sprouted
def totalSaplingsSprouted := saplingsSold + saplingsLeft

-- Define the percentage calculation
def percentageSprouted := (totalSaplingsSprouted / cherryPits) * 100

-- The theorem to be proved
theorem sprouted_percentage : percentageSprouted = 25 := by
  sorry

end NUMINAMATH_GPT_sprouted_percentage_l2357_235754


namespace NUMINAMATH_GPT_y_payment_is_approximately_272_73_l2357_235771

noncomputable def calc_y_payment : ℝ :=
  let total_payment : ℝ := 600
  let percent_x_to_y : ℝ := 1.2
  total_payment / (percent_x_to_y + 1)

theorem y_payment_is_approximately_272_73
  (total_payment : ℝ)
  (percent_x_to_y : ℝ)
  (h1 : total_payment = 600)
  (h2 : percent_x_to_y = 1.2) :
  calc_y_payment = 272.73 :=
by
  sorry

end NUMINAMATH_GPT_y_payment_is_approximately_272_73_l2357_235771


namespace NUMINAMATH_GPT_range_of_a_l2357_235705

variable (a : ℝ)
def p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def q := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (hpq_or : p a ∨ q a) (hpq_and_false : ¬ (p a ∧ q a)) : 
    a ∈ Set.Iio 0 ∪ Set.Ioo (1/4) 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2357_235705


namespace NUMINAMATH_GPT_remainder_of_x13_plus_1_by_x_minus_1_l2357_235737

-- Define the polynomial f(x) = x^13 + 1
def f (x : ℕ) : ℕ := x ^ 13 + 1

-- State the theorem using the Polynomial Remainder Theorem
theorem remainder_of_x13_plus_1_by_x_minus_1 : f 1 = 2 := by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_remainder_of_x13_plus_1_by_x_minus_1_l2357_235737


namespace NUMINAMATH_GPT_committee_selection_l2357_235793

-- Definitions based on the conditions
def num_people := 12
def num_women := 7
def num_men := 5
def committee_size := 5
def min_women := 2

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Required number of ways to form the committee
def num_ways_5_person_committee_with_at_least_2_women : ℕ :=
  binom num_women min_women * binom (num_people - min_women) (committee_size - min_women)

-- Statement to be proven
theorem committee_selection : num_ways_5_person_committee_with_at_least_2_women = 2520 :=
by
  sorry

end NUMINAMATH_GPT_committee_selection_l2357_235793


namespace NUMINAMATH_GPT_son_present_age_l2357_235749

-- Definitions
variables (S M : ℕ)
-- Conditions
def age_diff : Prop := M = S + 22
def future_age_condition : Prop := M + 2 = 2 * (S + 2)

-- Theorem statement with proof placeholder
theorem son_present_age (H1 : age_diff S M) (H2 : future_age_condition S M) : S = 20 :=
by sorry

end NUMINAMATH_GPT_son_present_age_l2357_235749


namespace NUMINAMATH_GPT_min_x2_y2_l2357_235729

theorem min_x2_y2 (x y : ℝ) (h : x * y - x - y = 1) : x^2 + y^2 ≥ 6 - 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_x2_y2_l2357_235729


namespace NUMINAMATH_GPT_circumference_of_jogging_track_l2357_235789

noncomputable def trackCircumference (Deepak_speed : ℝ) (Wife_speed : ℝ) (meet_time_minutes : ℝ) : ℝ :=
  let relative_speed := Deepak_speed + Wife_speed
  let meet_time_hours := meet_time_minutes / 60
  relative_speed * meet_time_hours

theorem circumference_of_jogging_track :
  trackCircumference 20 17 37 = 1369 / 60 :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_jogging_track_l2357_235789


namespace NUMINAMATH_GPT_expression_simplification_l2357_235759

theorem expression_simplification : 2 + 1 / (3 + 1 / (2 + 2)) = 30 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_expression_simplification_l2357_235759


namespace NUMINAMATH_GPT_car_passing_problem_l2357_235778

noncomputable def maxCarsPerHourDividedBy10 : ℕ :=
  let unit_length (n : ℕ) := 5 * (n + 1)
  let cars_passed_in_one_hour (n : ℕ) := 10000 * n / unit_length n
  Nat.div (2000) (10)

theorem car_passing_problem : maxCarsPerHourDividedBy10 = 200 :=
  by
  sorry

end NUMINAMATH_GPT_car_passing_problem_l2357_235778


namespace NUMINAMATH_GPT_cost_of_double_burger_l2357_235747

-- Definitions based on conditions
def total_cost : ℝ := 64.50
def total_burgers : ℕ := 50
def single_burger_cost : ℝ := 1.00
def double_burgers : ℕ := 29

-- Proof goal
theorem cost_of_double_burger : (total_cost - single_burger_cost * (total_burgers - double_burgers)) / double_burgers = 1.50 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_double_burger_l2357_235747


namespace NUMINAMATH_GPT_actual_revenue_percent_of_projected_l2357_235707

noncomputable def projected_revenue (R : ℝ) : ℝ := 1.2 * R
noncomputable def actual_revenue (R : ℝ) : ℝ := 0.75 * R

theorem actual_revenue_percent_of_projected (R : ℝ) :
  (actual_revenue R / projected_revenue R) * 100 = 62.5 :=
  sorry

end NUMINAMATH_GPT_actual_revenue_percent_of_projected_l2357_235707


namespace NUMINAMATH_GPT_gilled_mushrooms_count_l2357_235756

def mushrooms_problem (G S : ℕ) : Prop :=
  (S = 9 * G) ∧ (G + S = 30) → (G = 3)

-- The theorem statement corresponding to the problem
theorem gilled_mushrooms_count (G S : ℕ) : mushrooms_problem G S :=
by {
  sorry
}

end NUMINAMATH_GPT_gilled_mushrooms_count_l2357_235756


namespace NUMINAMATH_GPT_ellen_legos_final_count_l2357_235706

-- Definitions based on conditions
def initial_legos : ℕ := 380
def lost_legos_first_week : ℕ := 57
def additional_legos_second_week (remaining_legos : ℕ) : ℕ := 32
def borrowed_legos_third_week (total_legos : ℕ) : ℕ := 88

-- Computed values based on conditions
def legos_after_first_week (initial : ℕ) (lost : ℕ) : ℕ := initial - lost
def legos_after_second_week (remaining : ℕ) (additional : ℕ) : ℕ := remaining + additional
def legos_after_third_week (total : ℕ) (borrowed : ℕ) : ℕ := total - borrowed

-- Proof statement
theorem ellen_legos_final_count : 
  legos_after_third_week 
    (legos_after_second_week 
      (legos_after_first_week initial_legos lost_legos_first_week)
      (additional_legos_second_week (legos_after_first_week initial_legos lost_legos_first_week)))
    (borrowed_legos_third_week (legos_after_second_week 
                                  (legos_after_first_week initial_legos lost_legos_first_week)
                                  (additional_legos_second_week (legos_after_first_week initial_legos lost_legos_first_week)))) 
  = 267 :=
by 
  sorry

end NUMINAMATH_GPT_ellen_legos_final_count_l2357_235706


namespace NUMINAMATH_GPT_find_initial_crayons_l2357_235742

namespace CrayonProblem

variable (gave : ℕ) (lost : ℕ) (additional_lost : ℕ) 

def correct_answer (gave lost additional_lost : ℕ) :=
  gave + lost = gave + (gave + additional_lost) ∧ gave + lost = 502

theorem find_initial_crayons
  (gave := 90)
  (lost := 412)
  (additional_lost := 322)
  : correct_answer gave lost additional_lost :=
by 
  sorry

end CrayonProblem

end NUMINAMATH_GPT_find_initial_crayons_l2357_235742


namespace NUMINAMATH_GPT_min_value_of_expression_l2357_235748

theorem min_value_of_expression (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 10) : 
  ∃ B, B = x^2 + y^2 + z^2 + x^2 * y ∧ B ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2357_235748


namespace NUMINAMATH_GPT_gcd_40_120_45_l2357_235746

theorem gcd_40_120_45 : Nat.gcd (Nat.gcd 40 120) 45 = 5 :=
by
  sorry

end NUMINAMATH_GPT_gcd_40_120_45_l2357_235746


namespace NUMINAMATH_GPT_complement_union_correct_l2357_235736

open Set

variable (U : Set Int)
variable (A B : Set Int)

theorem complement_union_correct (hU : U = {-2, -1, 0, 1, 2}) (hA : A = {1, 2}) (hB : B = {-2, 1, 2}) :
  A ∪ (U \ B) = {-1, 0, 1, 2} := by
  rw [hU, hA, hB]
  simp
  sorry

end NUMINAMATH_GPT_complement_union_correct_l2357_235736


namespace NUMINAMATH_GPT_find_coins_l2357_235765

-- Definitions based on conditions
structure Wallet where
  coin1 : ℕ
  coin2 : ℕ
  h_total_value : coin1 + coin2 = 15
  h_not_five : coin1 ≠ 5 ∨ coin2 ≠ 5

-- Theorem statement based on the proof problem
theorem find_coins (w : Wallet) : (w.coin1 = 5 ∧ w.coin2 = 10) ∨ (w.coin1 = 10 ∧ w.coin2 = 5) := by
  sorry

end NUMINAMATH_GPT_find_coins_l2357_235765


namespace NUMINAMATH_GPT_swiss_slices_correct_l2357_235769

-- Define the variables and conditions
variables (S : ℕ) (cheddar_slices : ℕ := 12) (total_cheddar_slices : ℕ := 84) (total_swiss_slices : ℕ := 84)

-- Define the statement to be proved
theorem swiss_slices_correct (H : total_cheddar_slices = total_swiss_slices) : S = 12 :=
sorry

end NUMINAMATH_GPT_swiss_slices_correct_l2357_235769


namespace NUMINAMATH_GPT_computer_additions_per_hour_l2357_235777

def operations_per_second : ℕ := 15000
def additions_per_second : ℕ := operations_per_second / 2
def seconds_per_hour : ℕ := 3600

theorem computer_additions_per_hour : 
  additions_per_second * seconds_per_hour = 27000000 := by
  sorry

end NUMINAMATH_GPT_computer_additions_per_hour_l2357_235777


namespace NUMINAMATH_GPT_crayons_loss_l2357_235787

def initial_crayons : ℕ := 479
def final_crayons : ℕ := 134
def crayons_lost : ℕ := initial_crayons - final_crayons

theorem crayons_loss :
  crayons_lost = 345 := by
  sorry

end NUMINAMATH_GPT_crayons_loss_l2357_235787


namespace NUMINAMATH_GPT_bacteria_doubling_time_l2357_235726

noncomputable def doubling_time_population 
    (initial final : ℝ) 
    (time : ℝ) 
    (growth_factor : ℕ) : ℝ :=
    time / (Real.log growth_factor / Real.log 2)

theorem bacteria_doubling_time :
  doubling_time_population 1000 500000 26.897352853986263 500 = 0.903 :=
by
  sorry

end NUMINAMATH_GPT_bacteria_doubling_time_l2357_235726


namespace NUMINAMATH_GPT_find_m_value_l2357_235722

open Real

-- Define the vectors a and b as specified in the problem
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (3, -2)

-- Define the sum of vectors a and b
def vec_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the dot product of the vector sum with vector b to be zero as the given condition
def dot_product (m : ℝ) : ℝ := (vec_sum m).1 * vec_b.1 + (vec_sum m).2 * vec_b.2

-- The theorem to prove that given the defined conditions, m equals 8
theorem find_m_value (m : ℝ) (h : dot_product m = 0) : m = 8 := by
  sorry

end NUMINAMATH_GPT_find_m_value_l2357_235722


namespace NUMINAMATH_GPT_radio_show_length_l2357_235720

theorem radio_show_length :
  let s3 := 10
  let s2 := s3 + 5
  let s4 := s2 / 2
  let s5 := 2 * s4
  let s1 := 2 * (s2 + s3 + s4 + s5)
  s1 + s2 + s3 + s4 + s5 = 142.5 :=
by
  sorry

end NUMINAMATH_GPT_radio_show_length_l2357_235720


namespace NUMINAMATH_GPT_find_f_l2357_235796

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f (x : ℝ) :
  (∀ t : ℝ, t = (1 - x) / (1 + x) → f t = (1 - x^2) / (1 + x^2)) →
  f x = (2 * x) / (1 + x^2) :=
by
  intros h
  specialize h ((1 - x) / (1 + x))
  specialize h rfl
  exact sorry

end NUMINAMATH_GPT_find_f_l2357_235796


namespace NUMINAMATH_GPT_standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l2357_235730

theorem standing_in_a_row (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 → 
  ∃ (ways : ℕ), ways = 120 :=
by
  sorry

theorem standing_in_a_row_AB_adj_CD_not_adj (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 24 :=
by
  sorry

theorem assign_to_classes (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 150 :=
by
  sorry

end NUMINAMATH_GPT_standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l2357_235730


namespace NUMINAMATH_GPT_equivalent_condition_for_continuity_l2357_235770

theorem equivalent_condition_for_continuity {x c d : ℝ} (g : ℝ → ℝ) (h1 : g x = 5 * x - 3) (h2 : ∀ x, |g x - 1| < c → |x - 1| < d) (hc : c > 0) (hd : d > 0) : d ≤ c / 5 :=
sorry

end NUMINAMATH_GPT_equivalent_condition_for_continuity_l2357_235770


namespace NUMINAMATH_GPT_num_5_digit_numbers_is_six_l2357_235775

-- Define that we have the digits 2, 45, and 68
def digits : List Nat := [2, 45, 68]

-- Function to generate all permutations of given digits
def permute : List Nat → List (List Nat)
| [] => [[]]
| (x::xs) =>
  List.join (List.map (λ ys =>
    List.map (λ zs => x :: zs) (permute xs)) (permute xs))

-- Calculate the number of distinct 5-digit numbers
def numberOf5DigitNumbers : Int := 
  (permute digits).length

-- Theorem to prove the number of distinct 5-digit numbers formed
theorem num_5_digit_numbers_is_six : numberOf5DigitNumbers = 6 := by
  sorry

end NUMINAMATH_GPT_num_5_digit_numbers_is_six_l2357_235775


namespace NUMINAMATH_GPT_determine_A_l2357_235733

open Real

theorem determine_A (A B C : ℝ)
  (h_decomposition : ∀ x, x ≠ 4 ∧ x ≠ -2 -> (x + 2) / (x^3 - 9 * x^2 + 14 * x + 24) = A / (x - 4) + B / (x - 3) + C / (x + 2)^2)
  (h_factorization : ∀ x, (x^3 - 9 * x^2 + 14 * x + 24) = (x - 4) * (x - 3) * (x + 2)^2) :
  A = 1 / 6 := 
sorry

end NUMINAMATH_GPT_determine_A_l2357_235733


namespace NUMINAMATH_GPT_ratio_girls_to_boys_l2357_235798

-- Define the number of students and conditions
def num_students : ℕ := 25
def girls_more_than_boys : ℕ := 3

-- Define the variables
variables (g b : ℕ)

-- Define the conditions
def total_students := g + b = num_students
def girls_boys_relationship := b = g - girls_more_than_boys

-- Lean theorem statement
theorem ratio_girls_to_boys (g b : ℕ) (h1 : total_students g b) (h2 : girls_boys_relationship g b) : (g : ℚ) / b = 14 / 11 :=
sorry

end NUMINAMATH_GPT_ratio_girls_to_boys_l2357_235798


namespace NUMINAMATH_GPT_find_R_l2357_235782

theorem find_R (a b Q R : ℕ) (ha_prime : Prime a) (hb_prime : Prime b) (h_distinct : a ≠ b)
  (h1 : a^2 - a * Q + R = 0) (h2 : b^2 - b * Q + R = 0) : R = 6 :=
sorry

end NUMINAMATH_GPT_find_R_l2357_235782


namespace NUMINAMATH_GPT_root_condition_l2357_235721

noncomputable def f (x t : ℝ) := x^2 + t * x - t

theorem root_condition {t : ℝ} : (t ≥ 0 → ∃ x : ℝ, f x t = 0) ∧ (∃ x : ℝ, f x t = 0 → t ≥ 0 ∨ t ≤ -4) := 
  sorry

end NUMINAMATH_GPT_root_condition_l2357_235721


namespace NUMINAMATH_GPT_chairs_to_remove_l2357_235713

/-- A conference hall is setting up seating for a lecture with specific conditions.
    Given the total number of chairs, chairs per row, and participants expected to attend,
    prove the number of chairs to be removed to have complete rows with the least number of empty seats. -/
theorem chairs_to_remove
  (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_participants : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : total_chairs = 225)
  (h3 : expected_participants = 140) :
  total_chairs - (chairs_per_row * ((expected_participants + chairs_per_row - 1) / chairs_per_row)) = 75 :=
by
  sorry

end NUMINAMATH_GPT_chairs_to_remove_l2357_235713


namespace NUMINAMATH_GPT_sum_eq_two_l2357_235738

theorem sum_eq_two (x y : ℝ) (hx : x^3 - 3 * x^2 + 5 * x = 1) (hy : y^3 - 3 * y^2 + 5 * y = 5) : x + y = 2 := 
sorry

end NUMINAMATH_GPT_sum_eq_two_l2357_235738


namespace NUMINAMATH_GPT_geom_prog_terms_exist_l2357_235750

theorem geom_prog_terms_exist (b3 b6 : ℝ) (h1 : b3 = -1) (h2 : b6 = 27 / 8) :
  ∃ (b1 q : ℝ), b1 = -4 / 9 ∧ q = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geom_prog_terms_exist_l2357_235750


namespace NUMINAMATH_GPT_initial_blue_balls_l2357_235762

-- Define the problem conditions
variable (R B : ℕ) -- Number of red balls and blue balls originally in the box.

-- Condition 1: Blue balls are 17 more than red balls
axiom h1 : B = R + 17

-- Condition 2: Ball addition and removal scenario
noncomputable def total_balls_after_changes : ℕ :=
  (B + 57) + (R + 18) - 44

-- Condition 3: Total balls after all changes is 502
axiom h2 : total_balls_after_changes R B = 502

-- We need to prove the initial number of blue balls
theorem initial_blue_balls : B = 244 :=
by
  sorry

end NUMINAMATH_GPT_initial_blue_balls_l2357_235762


namespace NUMINAMATH_GPT_max_cars_div_10_l2357_235757

noncomputable def max_cars (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) : ℕ :=
  let k := 2000
  2000 -- Maximum number of cars passing the sensor

theorem max_cars_div_10 (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) :
  car_length = 5 →
  (∀ k : ℕ, distance_for_speed k = k) →
  (∀ k : ℕ, speed k = 10 * k) →
  (max_cars car_length distance_for_speed speed) = 2000 → 
  (max_cars car_length distance_for_speed speed) / 10 = 200 := by
  intros
  sorry

end NUMINAMATH_GPT_max_cars_div_10_l2357_235757


namespace NUMINAMATH_GPT_regular_polygon_sides_l2357_235784

theorem regular_polygon_sides (n : ℕ) (h : 0 < n) (h_angle : (n - 2) * 180 = 144 * n) :
  n = 10 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2357_235784


namespace NUMINAMATH_GPT_probability_red_red_red_l2357_235792

-- Definition of probability for picking three red balls without replacement
def total_balls := 21
def red_balls := 7
def blue_balls := 9
def green_balls := 5

theorem probability_red_red_red : 
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) * ((red_balls - 2) / (total_balls - 2)) = 1 / 38 := 
by sorry

end NUMINAMATH_GPT_probability_red_red_red_l2357_235792


namespace NUMINAMATH_GPT_sandwiches_ordered_l2357_235703

-- Definitions of the given conditions
def sandwichCost : ℕ := 5
def payment : ℕ := 20
def change : ℕ := 5

-- Statement to prove how many sandwiches Jack ordered
theorem sandwiches_ordered : (payment - change) / sandwichCost = 3 := by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_sandwiches_ordered_l2357_235703


namespace NUMINAMATH_GPT_tangent_and_parallel_l2357_235772

noncomputable def parabola1 (x : ℝ) (b1 c1 : ℝ) : ℝ := -x^2 + b1 * x + c1
noncomputable def parabola2 (x : ℝ) (b2 c2 : ℝ) : ℝ := -x^2 + b2 * x + c2
noncomputable def parabola3 (x : ℝ) (b3 c3 : ℝ) : ℝ := x^2 + b3 * x + c3

theorem tangent_and_parallel (b1 b2 b3 c1 c2 c3 : ℝ) :
  (b3 - b1)^2 = 8 * (c3 - c1) → (b3 - b2)^2 = 8 * (c3 - c2) →
  ((b2^2 - b1^2 + 2 * b3 * (b2 - b1)) / (4 * (b2 - b1))) = 
  ((4 * (c1 - c2) - 2 * b3 * (b1 - b2)) / (2 * (b2 - b1))) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_tangent_and_parallel_l2357_235772


namespace NUMINAMATH_GPT_problem1_problem2_l2357_235731

variable (a b : ℝ)

-- (1) Prove a + b = 2 given the conditions
theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x : ℝ, abs (x - a) + abs (x + b) ≥ 2) : a + b = 2 :=
sorry

-- (2) Prove it is not possible for both a^2 + a > 2 and b^2 + b > 2 to hold simultaneously
theorem problem2 (h1: a + b = 2) (h2 : a^2 + a > 2) (h3 : b^2 + b > 2) : False :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2357_235731


namespace NUMINAMATH_GPT_correct_option_c_l2357_235791

variable (a b c : ℝ)

def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom symmetry_axis : -b / (2 * a) = 1

theorem correct_option_c (h : b = -2 * a) : c > 2 * b :=
 sorry

end NUMINAMATH_GPT_correct_option_c_l2357_235791


namespace NUMINAMATH_GPT_circle_area_in_sq_cm_l2357_235788

theorem circle_area_in_sq_cm (diameter_meters : ℝ) (h : diameter_meters = 5) : 
  let radius_meters := diameter_meters / 2
  let area_square_meters := π * radius_meters^2
  let area_square_cm := area_square_meters * 10000
  area_square_cm = 62500 * π :=
by
  sorry

end NUMINAMATH_GPT_circle_area_in_sq_cm_l2357_235788


namespace NUMINAMATH_GPT_solve_for_y_l2357_235701

theorem solve_for_y (y : ℝ) : 5 * y - 100 = 125 ↔ y = 45 := by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2357_235701


namespace NUMINAMATH_GPT_sqrt_11_custom_op_l2357_235752

noncomputable def sqrt := Real.sqrt

def custom_op (x y : Real) := (x + y) ^ 2 - (x - y) ^ 2

theorem sqrt_11_custom_op : custom_op (sqrt 11) (sqrt 11) = 44 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_11_custom_op_l2357_235752


namespace NUMINAMATH_GPT_karen_average_speed_correct_l2357_235773

def karen_time_duration : ℚ := (22 : ℚ) / 3
def karen_distance : ℚ := 230

def karen_average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed_correct :
  karen_average_speed karen_distance karen_time_duration = (31 + 4/11 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_karen_average_speed_correct_l2357_235773


namespace NUMINAMATH_GPT_arithmetic_result_l2357_235794

/-- Define the constants involved in the arithmetic operation. -/
def a : ℕ := 999999999999
def b : ℕ := 888888888888
def c : ℕ := 111111111111

/-- The theorem stating that the given arithmetic operation results in the expected answer. -/
theorem arithmetic_result :
  a - b + c = 222222222222 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_result_l2357_235794


namespace NUMINAMATH_GPT_no_integer_solution_for_triples_l2357_235790

theorem no_integer_solution_for_triples :
  ∀ (x y z : ℤ),
    x^2 - 2*x*y + 3*y^2 - z^2 = 17 →
    -x^2 + 4*y*z + z^2 = 28 →
    x^2 + 2*x*y + 5*z^2 = 42 →
    false :=
by
  intros x y z h1 h2 h3
  sorry

end NUMINAMATH_GPT_no_integer_solution_for_triples_l2357_235790


namespace NUMINAMATH_GPT_find_x_y_sum_l2357_235716

def is_perfect_square (n : ℕ) : Prop := ∃ (k : ℕ), k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

theorem find_x_y_sum (n x y : ℕ) (hn : n = 450) (hx : x > 0) (hy : y > 0)
  (hxsq : is_perfect_square (n * x))
  (hycube : is_perfect_cube (n * y)) :
  x + y = 62 :=
  sorry

end NUMINAMATH_GPT_find_x_y_sum_l2357_235716


namespace NUMINAMATH_GPT_veggies_minus_fruits_l2357_235700

-- Definitions of quantities as given in the conditions
def cucumbers : ℕ := 6
def tomatoes : ℕ := 8
def apples : ℕ := 2
def bananas : ℕ := 4

-- Problem Statement
theorem veggies_minus_fruits : (cucumbers + tomatoes) - (apples + bananas) = 8 :=
by 
  -- insert proof here
  sorry

end NUMINAMATH_GPT_veggies_minus_fruits_l2357_235700


namespace NUMINAMATH_GPT_sum_of_five_consecutive_squares_not_perfect_square_l2357_235743

theorem sum_of_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ¬ ∃ (k : ℤ), k^2 = (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_five_consecutive_squares_not_perfect_square_l2357_235743


namespace NUMINAMATH_GPT_no_maximum_y_coordinate_for_hyperbola_l2357_235751

theorem no_maximum_y_coordinate_for_hyperbola :
  ∀ y : ℝ, ∃ x : ℝ, y = 3 + (3 / 5) * x :=
by
  sorry

end NUMINAMATH_GPT_no_maximum_y_coordinate_for_hyperbola_l2357_235751


namespace NUMINAMATH_GPT_range_of_x_coordinate_of_Q_l2357_235768

def Point := ℝ × ℝ

def parabola (P : Point) : Prop :=
  P.2 = P.1 ^ 2

def vector (P Q : Point) : Point :=
  (Q.1 - P.1, Q.2 - P.2)

def dot_product (u v : Point) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (P Q R : Point) : Prop :=
  dot_product (vector P Q) (vector P R) = 0

theorem range_of_x_coordinate_of_Q:
  ∀ (A P Q: Point), 
    A = (-1, 1) →
    parabola P →
    parabola Q →
    perpendicular P A Q →
    (Q.1 ≤ -3 ∨ Q.1 ≥ 1) :=
by
  intros A P Q hA hParabP hParabQ hPerp
  sorry

end NUMINAMATH_GPT_range_of_x_coordinate_of_Q_l2357_235768


namespace NUMINAMATH_GPT_find_k_circle_radius_l2357_235727

theorem find_k_circle_radius (k : ℝ) :
  (∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) → ((x + 4)^2 + (y + 2)^2 = 7^2)) → k = 29 :=
sorry

end NUMINAMATH_GPT_find_k_circle_radius_l2357_235727


namespace NUMINAMATH_GPT_hotel_ticket_ratio_l2357_235735

theorem hotel_ticket_ratio (initial_amount : ℕ) (remaining_amount : ℕ) (ticket_cost : ℕ) (hotel_cost : ℕ) :
  initial_amount = 760 →
  remaining_amount = 310 →
  ticket_cost = 300 →
  initial_amount - remaining_amount - ticket_cost = hotel_cost →
  (hotel_cost : ℚ) / (ticket_cost : ℚ) = 1 / 2 :=
by
  intros h_initial h_remaining h_ticket h_hotel
  sorry

end NUMINAMATH_GPT_hotel_ticket_ratio_l2357_235735


namespace NUMINAMATH_GPT_probability_of_spade_or_king_in_two_draws_l2357_235776

def total_cards : ℕ := 52
def spades_count : ℕ := 13
def kings_count : ℕ := 4
def king_of_spades_count : ℕ := 1
def spades_or_kings_count : ℕ := spades_count + kings_count - king_of_spades_count
def probability_not_spade_or_king : ℚ := (total_cards - spades_or_kings_count) / total_cards
def probability_both_not_spade_or_king : ℚ := probability_not_spade_or_king^2
def probability_at_least_one_spade_or_king : ℚ := 1 - probability_both_not_spade_or_king

theorem probability_of_spade_or_king_in_two_draws :
  probability_at_least_one_spade_or_king = 88 / 169 :=
sorry

end NUMINAMATH_GPT_probability_of_spade_or_king_in_two_draws_l2357_235776


namespace NUMINAMATH_GPT_company_production_n_l2357_235797

theorem company_production_n (n : ℕ) (P : ℕ) 
  (h1 : P = n * 50) 
  (h2 : (P + 90) / (n + 1) = 58) : n = 4 := by 
  sorry

end NUMINAMATH_GPT_company_production_n_l2357_235797


namespace NUMINAMATH_GPT_num_games_round_robin_l2357_235704

-- There are 10 classes in the second grade, each class forms one team.
def num_teams := 10

-- A round-robin format means each team plays against every other team once.
def num_games (n : Nat) := n * (n - 1) / 2

-- Proving the total number of games played with num_teams equals to 45
theorem num_games_round_robin : num_games num_teams = 45 := by
  sorry

end NUMINAMATH_GPT_num_games_round_robin_l2357_235704


namespace NUMINAMATH_GPT_candy_sampling_l2357_235779

theorem candy_sampling (total_customers caught_sampling not_caught_sampling : ℝ) :
  caught_sampling = 0.22 * total_customers →
  not_caught_sampling = 0.12 * (total_customers * sampling_percent) →
  (sampling_percent * total_customers = caught_sampling / 0.78) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_candy_sampling_l2357_235779


namespace NUMINAMATH_GPT_carriage_and_people_l2357_235728

variable {x y : ℕ}

theorem carriage_and_people :
  (3 * (x - 2) = y) ∧ (2 * x + 9 = y) :=
sorry

end NUMINAMATH_GPT_carriage_and_people_l2357_235728


namespace NUMINAMATH_GPT_sandwich_is_not_condiments_l2357_235795

theorem sandwich_is_not_condiments (sandwich_weight condiments_weight : ℕ)
  (h1 : sandwich_weight = 150)
  (h2 : condiments_weight = 45) :
  (sandwich_weight - condiments_weight) / sandwich_weight * 100 = 70 := 
by sorry

end NUMINAMATH_GPT_sandwich_is_not_condiments_l2357_235795


namespace NUMINAMATH_GPT_students_not_picked_l2357_235711

/-- There are 36 students trying out for the school's trivia teams. 
If some of them didn't get picked and the rest were put into 3 groups with 9 students in each group,
prove that the number of students who didn't get picked is 9. -/

theorem students_not_picked (total_students groups students_per_group picked_students not_picked_students : ℕ)
    (h1 : total_students = 36)
    (h2 : groups = 3)
    (h3 : students_per_group = 9)
    (h4 : picked_students = groups * students_per_group)
    (h5 : not_picked_students = total_students - picked_students) :
    not_picked_students = 9 :=
by
  sorry

end NUMINAMATH_GPT_students_not_picked_l2357_235711


namespace NUMINAMATH_GPT_lucy_apples_per_week_l2357_235723

-- Define the conditions
def chandler_apples_per_week := 23
def total_apples_per_month := 168
def weeks_per_month := 4
def chandler_apples_per_month := chandler_apples_per_week * weeks_per_month
def lucy_apples_per_month := total_apples_per_month - chandler_apples_per_month

-- Define the proof problem statement
theorem lucy_apples_per_week :
  lucy_apples_per_month / weeks_per_month = 19 :=
  by sorry

end NUMINAMATH_GPT_lucy_apples_per_week_l2357_235723


namespace NUMINAMATH_GPT_fathers_age_more_than_three_times_son_l2357_235764

variable (F S x : ℝ)

theorem fathers_age_more_than_three_times_son :
  F = 27 →
  F = 3 * S + x →
  F + 3 = 2 * (S + 3) + 8 →
  x = 3 :=
by
  intros hF h1 h2
  sorry

end NUMINAMATH_GPT_fathers_age_more_than_three_times_son_l2357_235764


namespace NUMINAMATH_GPT_rotated_parabola_eq_l2357_235753

theorem rotated_parabola_eq :
  ∀ x y : ℝ, y = x^2 → ∃ y' x' : ℝ, (y' = (-x':ℝ)^2) := sorry

end NUMINAMATH_GPT_rotated_parabola_eq_l2357_235753


namespace NUMINAMATH_GPT_sonic_leads_by_19_2_meters_l2357_235740

theorem sonic_leads_by_19_2_meters (v_S v_D : ℝ)
  (h1 : ∀ t, t = 200 / v_S → 200 = v_S * t)
  (h2 : ∀ t, t = 184 / v_D → 184 = v_D * t)
  (h3 : v_S / v_D = 200 / 184)
  :  240 / v_S - (200 / v_S / (200 / 184) * 240) = 19.2 := by
  sorry

end NUMINAMATH_GPT_sonic_leads_by_19_2_meters_l2357_235740


namespace NUMINAMATH_GPT_intersection_lines_l2357_235732

theorem intersection_lines (x y : ℝ) :
  (2 * x - y - 10 = 0) ∧ (3 * x + 4 * y - 4 = 0) → (x = 4) ∧ (y = -2) :=
by
  -- The proof is provided here
  sorry

end NUMINAMATH_GPT_intersection_lines_l2357_235732


namespace NUMINAMATH_GPT_rectangle_area_l2357_235758

theorem rectangle_area (a b : ℕ) 
  (h1 : 2 * (a + b) = 16)
  (h2 : a^2 + b^2 - 2 * a * b - 4 = 0) :
  a * b = 30 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2357_235758


namespace NUMINAMATH_GPT_kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l2357_235760

/-- Conditions: -/
def kitchen_clock_gain_rate : ℝ := 1.5 -- minutes per hour
def bedroom_clock_lose_rate : ℝ := 0.5 -- minutes per hour
def synchronization_time : ℝ := 0 -- time in hours when both clocks were correct

/-- Problem 1: -/
theorem kitchen_clock_correct_again :
  ∃ t : ℝ, 1.5 * t = 720 :=
by {
  sorry
}

/-- Problem 2: -/
theorem bedroom_clock_correct_again :
  ∃ t : ℝ, 0.5 * t = 720 :=
by {
  sorry
}

/-- Problem 3: -/
theorem both_clocks_same_time_again :
  ∃ t : ℝ, 2 * t = 720 :=
by {
  sorry
}

end NUMINAMATH_GPT_kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l2357_235760


namespace NUMINAMATH_GPT_increasing_interval_of_f_l2357_235708

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 15 * x ^ 2 + 36 * x - 24

theorem increasing_interval_of_f : (∀ x : ℝ, x = 2 → deriv f x = 0) → ∀ x : ℝ, 3 < x → 0 < deriv f x :=
by
  intro h x hx
  -- We know that the function has an extreme value at x = 2
  have : deriv f 2 = 0 := h 2 rfl
  -- Require to prove the function is increasing in interval (3, +∞)
  sorry

end NUMINAMATH_GPT_increasing_interval_of_f_l2357_235708


namespace NUMINAMATH_GPT_sum_of_four_consecutive_integers_prime_factor_l2357_235766

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_four_consecutive_integers_prime_factor_l2357_235766


namespace NUMINAMATH_GPT_part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l2357_235734

noncomputable def y (x : ℝ) : ℝ := -10 * x + 400
noncomputable def W (x : ℝ) : ℝ := -10 * x^2 + 500 * x - 4000

theorem part_1_relationship (x : ℝ) (h₀ : 0 < x) (h₁ : x ≤ 40) :
  W x = -10 * x^2 + 500 * x - 4000 := by
  sorry

theorem part_2_solution (x : ℝ) (h₀ : W x = 1250) :
  x = 15 ∨ x = 35 := by
  sorry

theorem part_2_preferred (x : ℝ) (h₀ : W x = 1250) (h₁ : y 15 ≥ y 35) :
  x = 15 := by
  sorry

theorem part_3_max_W (x : ℝ) (h₀ : 28 ≤ x) (h₁ : x ≤ 35) :
  W x ≤ 2160 := by
  sorry

theorem part_3_max_at_28 :
  W 28 = 2160 := by
  sorry

end NUMINAMATH_GPT_part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l2357_235734


namespace NUMINAMATH_GPT_quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l2357_235725

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 2

-- Problem 1: Prove that the quadratic function passes through the origin for m = 1 or m = -2
theorem quadratic_passes_through_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -2) ∧ quadratic m 0 = 0 := by
  sorry

-- Problem 2: Prove that the quadratic function is symmetric about the y-axis for m = 0
theorem quadratic_symmetric_about_y_axis :
  ∃ m : ℝ, m = 0 ∧ ∀ x : ℝ, quadratic m x = quadratic m (-x) := by
  sorry

end NUMINAMATH_GPT_quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l2357_235725


namespace NUMINAMATH_GPT_fraction_power_equiv_l2357_235702

theorem fraction_power_equiv : (75000^4) / (25000^4) = 81 := by
  sorry

end NUMINAMATH_GPT_fraction_power_equiv_l2357_235702


namespace NUMINAMATH_GPT_no_pieces_left_impossible_l2357_235739

/-- Starting with 100 pieces and 1 pile, and given the ability to either:
1. Remove one piece from a pile of at least 3 pieces and divide the remaining pile into two non-empty piles,
2. Eliminate a pile containing a single piece,
prove that it is impossible to reach a situation with no pieces left. -/
theorem no_pieces_left_impossible :
  ∀ (p t : ℕ), p = 100 → t = 1 →
  (∀ (p' t' : ℕ),
    (p' = p - 1 ∧ t' = t + 1 ∧ 3 ≤ p) ∨
    (p' = p - 1 ∧ t' = t - 1 ∧ ∃ k, k = 1 ∧ t ≠ 0) →
    false) :=
by
  intros
  sorry

end NUMINAMATH_GPT_no_pieces_left_impossible_l2357_235739


namespace NUMINAMATH_GPT_sqrt_prod_plus_one_equals_341_l2357_235715

noncomputable def sqrt_prod_plus_one : ℕ :=
  Nat.sqrt ((20 * 19 * 18 * 17) + 1)

theorem sqrt_prod_plus_one_equals_341 :
  sqrt_prod_plus_one = 341 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_prod_plus_one_equals_341_l2357_235715


namespace NUMINAMATH_GPT_total_tires_parking_lot_l2357_235799

-- Definitions for each condition in a)
def four_wheel_drive_cars := 30
def motorcycles := 20
def six_wheel_trucks := 10
def bicycles := 5
def unicycles := 3
def baby_strollers := 2

def extra_roof_tires := 4
def flat_bike_tires_removed := 3
def extra_unicycle_wheel := 1

def tires_per_car := 4 + 1
def tires_per_motorcycle := 2 + 2
def tires_per_truck := 6 + 1
def tires_per_bicycle := 2
def tires_per_unicycle := 1
def tires_per_stroller := 4

-- Define total tires calculation
def total_tires (four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
                 extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel : ℕ) :=
  (four_wheel_drive_cars * tires_per_car + extra_roof_tires) +
  (motorcycles * tires_per_motorcycle) +
  (six_wheel_trucks * tires_per_truck) +
  (bicycles * tires_per_bicycle - flat_bike_tires_removed) +
  (unicycles * tires_per_unicycle + extra_unicycle_wheel) +
  (baby_strollers * tires_per_stroller)

-- The Lean statement for the proof problem
theorem total_tires_parking_lot : 
  total_tires four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
              extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel = 323 :=
by 
  sorry

end NUMINAMATH_GPT_total_tires_parking_lot_l2357_235799


namespace NUMINAMATH_GPT_num_people_price_item_equation_l2357_235718

theorem num_people_price_item_equation
  (x y : ℕ)
  (h1 : 8 * x = y + 3)
  (h2 : 7 * x = y - 4) :
  (y + 3) / 8 = (y - 4) / 7 :=
by
  sorry

end NUMINAMATH_GPT_num_people_price_item_equation_l2357_235718
