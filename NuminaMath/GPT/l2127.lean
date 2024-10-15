import Mathlib

namespace NUMINAMATH_GPT_range_of_m_l2127_212721

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > g x m) → m < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2127_212721


namespace NUMINAMATH_GPT_magazine_cost_l2127_212775

variable (b m : ℝ)

theorem magazine_cost (h1 : 2 * b + 2 * m = 26) (h2 : b + 3 * m = 27) : m = 7 :=
by
  sorry

end NUMINAMATH_GPT_magazine_cost_l2127_212775


namespace NUMINAMATH_GPT_tim_same_age_tina_l2127_212753

-- Define the ages of Tim and Tina
variables (x y : ℤ)

-- Given conditions
def condition_tim := x + 2 = 2 * (x - 2)
def condition_tina := y + 3 = 3 * (y - 3)

-- The goal is to prove that Tim is the same age as Tina
theorem tim_same_age_tina (htim : condition_tim x) (htina : condition_tina y) : x = y :=
by 
  sorry

end NUMINAMATH_GPT_tim_same_age_tina_l2127_212753


namespace NUMINAMATH_GPT_hispanic_population_in_west_l2127_212727

theorem hispanic_population_in_west (p_NE p_MW p_South p_West : ℕ)
  (h_NE : p_NE = 4)
  (h_MW : p_MW = 5)
  (h_South : p_South = 12)
  (h_West : p_West = 20) :
  ((p_West : ℝ) / (p_NE + p_MW + p_South + p_West : ℝ)) * 100 = 49 :=
by sorry

end NUMINAMATH_GPT_hispanic_population_in_west_l2127_212727


namespace NUMINAMATH_GPT_minimum_cable_length_l2127_212717

def station_positions : List ℝ := [0, 3, 7, 11, 14]

def total_cable_length (x : ℝ) : ℝ :=
  abs x + abs (x - 3) + abs (x - 7) + abs (x - 11) + abs (x - 14)

theorem minimum_cable_length :
  (∀ x : ℝ, total_cable_length x ≥ 22) ∧ total_cable_length 7 = 22 :=
by
  sorry

end NUMINAMATH_GPT_minimum_cable_length_l2127_212717


namespace NUMINAMATH_GPT_excluded_students_count_l2127_212785

theorem excluded_students_count 
  (N : ℕ) 
  (x : ℕ) 
  (average_marks : ℕ) 
  (excluded_average_marks : ℕ) 
  (remaining_average_marks : ℕ) 
  (total_students : ℕ)
  (h1 : average_marks = 80)
  (h2 : excluded_average_marks = 70)
  (h3 : remaining_average_marks = 90)
  (h4 : total_students = 10)
  (h5 : N = total_students)
  (h6 : 80 * N = 70 * x + 90 * (N - x))
  : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_excluded_students_count_l2127_212785


namespace NUMINAMATH_GPT_joe_needs_more_cars_l2127_212792

-- Definitions based on conditions
def current_cars : ℕ := 50
def total_cars : ℕ := 62

-- Theorem based on the problem question and correct answer
theorem joe_needs_more_cars : (total_cars - current_cars) = 12 :=
by
  sorry

end NUMINAMATH_GPT_joe_needs_more_cars_l2127_212792


namespace NUMINAMATH_GPT_raft_drift_time_l2127_212787

-- Define the conditions from the problem
variable (distance : ℝ := 1)
variable (steamboat_time : ℝ := 1) -- in hours
variable (motorboat_time : ℝ := 3 / 4) -- 45 minutes in hours
variable (motorboat_speed_ratio : ℝ := 2)

-- Variables for speeds
variable (vs vf : ℝ)

-- Conditions: the speeds and conditions of traveling from one village to another
variable (steamboat_eqn : vs + vf = distance / steamboat_time := by sorry)
variable (motorboat_eqn : (2 * vs) + vf = distance / motorboat_time := by sorry)

-- Time for the raft to travel the distance
theorem raft_drift_time : 90 = (distance / vf) * 60 := by
  -- Proof comes here
  sorry

end NUMINAMATH_GPT_raft_drift_time_l2127_212787


namespace NUMINAMATH_GPT_subtracted_number_from_32_l2127_212711

theorem subtracted_number_from_32 (x : ℕ) (h : 32 - x = 23) : x = 9 := 
by 
  sorry

end NUMINAMATH_GPT_subtracted_number_from_32_l2127_212711


namespace NUMINAMATH_GPT_five_digit_number_divisible_by_B_is_multiple_of_1000_l2127_212768

-- Definitions
def is_five_digit_number (A : ℕ) : Prop := 10000 ≤ A ∧ A < 100000
def B (A : ℕ) := (A / 1000 * 100) + (A % 100)
def is_four_digit_number (B : ℕ) : Prop := 1000 ≤ B ∧ B < 10000

-- Main theorem
theorem five_digit_number_divisible_by_B_is_multiple_of_1000
  (A : ℕ) (hA : is_five_digit_number A)
  (hAB : ∃ k : ℕ, B A = k) :
  A % 1000 = 0 := 
sorry

end NUMINAMATH_GPT_five_digit_number_divisible_by_B_is_multiple_of_1000_l2127_212768


namespace NUMINAMATH_GPT_range_of_a_l2127_212704

def proposition_p (a : ℝ) : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a

def proposition_q (a : ℝ) : Prop := ∃ (x₀ : ℝ), x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) : proposition_p a ∧ proposition_q a ↔ (a = 1 ∨ a ≤ -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2127_212704


namespace NUMINAMATH_GPT_matchsticks_left_l2127_212748

def initial_matchsticks : ℕ := 30
def matchsticks_needed_2 : ℕ := 5
def matchsticks_needed_0 : ℕ := 6
def num_2s : ℕ := 3
def num_0s : ℕ := 1

theorem matchsticks_left : 
  initial_matchsticks - (num_2s * matchsticks_needed_2 + num_0s * matchsticks_needed_0) = 9 :=
by sorry

end NUMINAMATH_GPT_matchsticks_left_l2127_212748


namespace NUMINAMATH_GPT_cab_driver_income_l2127_212715

theorem cab_driver_income (incomes : Fin 5 → ℝ)
  (h1 : incomes 0 = 250)
  (h2 : incomes 1 = 400)
  (h3 : incomes 2 = 750)
  (h4 : incomes 3 = 400)
  (avg_income : (incomes 0 + incomes 1 + incomes 2 + incomes 3 + incomes 4) / 5 = 460) : 
  incomes 4 = 500 :=
sorry

end NUMINAMATH_GPT_cab_driver_income_l2127_212715


namespace NUMINAMATH_GPT_remainder_when_nm_div_61_l2127_212735

theorem remainder_when_nm_div_61 (n m : ℕ) (k j : ℤ):
  n = 157 * k + 53 → m = 193 * j + 76 → (n + m) % 61 = 7 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_remainder_when_nm_div_61_l2127_212735


namespace NUMINAMATH_GPT_total_cookies_dropped_throughout_entire_baking_process_l2127_212731

def initially_baked_by_alice := 74 + 45 + 15
def initially_baked_by_bob := 7 + 32 + 18

def initially_dropped_by_alice := 5 + 8
def initially_dropped_by_bob := 10 + 6

def additional_baked_by_alice := 5 + 4 + 12
def additional_baked_by_bob := 22 + 36 + 14

def edible_cookies := 145

theorem total_cookies_dropped_throughout_entire_baking_process :
  initially_baked_by_alice + initially_baked_by_bob +
  additional_baked_by_alice + additional_baked_by_bob -
  edible_cookies = 139 := by
  sorry

end NUMINAMATH_GPT_total_cookies_dropped_throughout_entire_baking_process_l2127_212731


namespace NUMINAMATH_GPT_jenna_discount_l2127_212740

def normal_price : ℝ := 50
def tickets_from_website : ℝ := 2 * normal_price
def scalper_initial_price_per_ticket : ℝ := 2.4 * normal_price
def scalper_total_initial : ℝ := 2 * scalper_initial_price_per_ticket
def friend_discounted_ticket : ℝ := 0.6 * normal_price
def total_price_five_tickets : ℝ := tickets_from_website + scalper_total_initial + friend_discounted_ticket
def amount_paid_by_friends : ℝ := 360

theorem jenna_discount : 
    total_price_five_tickets - amount_paid_by_friends = 10 :=
by
  -- The proof would go here, but we leave it as sorry for now.
  sorry

end NUMINAMATH_GPT_jenna_discount_l2127_212740


namespace NUMINAMATH_GPT_maximize_area_partition_l2127_212790

noncomputable def optimLengthPartition (material: ℝ) (partitions: ℕ) : ℝ :=
  (material / (4 + partitions))

theorem maximize_area_partition :
  optimLengthPartition 24 (2 * 1) = 3 / 100 :=
by
  sorry

end NUMINAMATH_GPT_maximize_area_partition_l2127_212790


namespace NUMINAMATH_GPT_base4_arithmetic_l2127_212725

theorem base4_arithmetic : 
  ∀ (a b c : ℕ),
  a = 2 * 4^2 + 3 * 4^1 + 1 * 4^0 →
  b = 2 * 4^1 + 4 * 4^0 →
  c = 3 * 4^0 →
  (a * b) / c = 2 * 4^3 + 3 * 4^2 + 1 * 4^1 + 0 * 4^0 :=
by
  intros a b c ha hb hc
  sorry

end NUMINAMATH_GPT_base4_arithmetic_l2127_212725


namespace NUMINAMATH_GPT_prove_weight_loss_l2127_212718

variable (W : ℝ) -- Original weight
variable (x : ℝ) -- Percentage of weight lost

def weight_equation := W - (x / 100) * W + (2 / 100) * W = (89.76 / 100) * W

theorem prove_weight_loss (h : weight_equation W x) : x = 12.24 :=
by
  sorry

end NUMINAMATH_GPT_prove_weight_loss_l2127_212718


namespace NUMINAMATH_GPT_cash_sales_amount_l2127_212745

-- Definitions for conditions
def total_sales : ℕ := 80
def credit_sales : ℕ := (2 * total_sales) / 5

-- Statement of the proof problem
theorem cash_sales_amount :
  ∃ cash_sales : ℕ, cash_sales = total_sales - credit_sales ∧ cash_sales = 48 :=
by
  sorry

end NUMINAMATH_GPT_cash_sales_amount_l2127_212745


namespace NUMINAMATH_GPT_annual_income_of_a_l2127_212736

-- Definitions based on the conditions
def monthly_income_ratio (a_income b_income : ℝ) : Prop := a_income / b_income = 5 / 2
def income_percentage (part whole : ℝ) : Prop := part / whole = 12 / 100
def c_monthly_income : ℝ := 15000
def b_monthly_income (c_income : ℝ) := c_income + 0.12 * c_income

-- The theorem to prove
theorem annual_income_of_a : ∀ (a_income b_income c_income : ℝ),
  monthly_income_ratio a_income b_income ∧
  b_income = b_monthly_income c_income ∧
  c_income = c_monthly_income →
  (a_income * 12) = 504000 :=
by
  -- Here we do not need to fill out the proof, so we use sorry
  sorry

end NUMINAMATH_GPT_annual_income_of_a_l2127_212736


namespace NUMINAMATH_GPT_polynomial_solution_l2127_212786

variable {R : Type*} [CommRing R]

theorem polynomial_solution (p : Polynomial R) :
  (∀ (a b c : R), 
    p.eval (a + b - 2 * c) + p.eval (b + c - 2 * a) + p.eval (c + a - 2 * b)
      = 3 * p.eval (a - b) + 3 * p.eval (b - c) + 3 * p.eval (c - a)
  ) →
  ∃ (a1 a2 : R), p = Polynomial.C a2 * Polynomial.X^2 + Polynomial.C a1 * Polynomial.X :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l2127_212786


namespace NUMINAMATH_GPT_original_number_l2127_212780

theorem original_number (x : ℝ) (h : x + 0.5 * x = 90) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l2127_212780


namespace NUMINAMATH_GPT_paper_area_l2127_212769

variable (L W : ℕ)

theorem paper_area (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : L * W = 140 := by
  sorry

end NUMINAMATH_GPT_paper_area_l2127_212769


namespace NUMINAMATH_GPT_standard_deviation_less_than_mean_l2127_212765

theorem standard_deviation_less_than_mean 
  (μ : ℝ) (σ : ℝ) (x : ℝ) 
  (h1 : μ = 14.5) 
  (h2 : σ = 1.5) 
  (h3 : x = 11.5) : 
  (μ - x) / σ = 2 :=
by
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_standard_deviation_less_than_mean_l2127_212765


namespace NUMINAMATH_GPT_alexander_total_payment_l2127_212733

variable (initialFee : ℝ) (dailyRent : ℝ) (costPerMile : ℝ) (daysRented : ℕ) (milesDriven : ℝ)

def totalCost (initialFee dailyRent costPerMile : ℝ) (daysRented : ℕ) (milesDriven : ℝ) : ℝ :=
  initialFee + (dailyRent * daysRented) + (costPerMile * milesDriven)

theorem alexander_total_payment :
  totalCost 15 30 0.25 3 350 = 192.5 :=
by
  unfold totalCost
  norm_num

end NUMINAMATH_GPT_alexander_total_payment_l2127_212733


namespace NUMINAMATH_GPT_ratio_female_to_male_members_l2127_212791

theorem ratio_female_to_male_members (f m : ℕ)
  (h1 : 35 * f = SumAgesFemales)
  (h2 : 20 * m = SumAgesMales)
  (h3 : (35 * f + 20 * m) / (f + m) = 25) :
  f / m = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_female_to_male_members_l2127_212791


namespace NUMINAMATH_GPT_valid_pair_l2127_212705

-- Definitions of the animals
inductive Animal
| lion
| tiger
| leopard
| elephant

open Animal

-- Given conditions
def condition1 (selected : Animal → Prop) : Prop :=
  selected lion → selected tiger

def condition2 (selected : Animal → Prop) : Prop :=
  ¬selected leopard → ¬selected tiger

def condition3 (selected : Animal → Prop) : Prop :=
  selected leopard → ¬selected elephant

-- Main theorem to prove
theorem valid_pair (selected : Animal → Prop) (pair : Animal × Animal) :
  (pair = (tiger, leopard)) ↔ 
  (condition1 selected ∧ condition2 selected ∧ condition3 selected) :=
sorry

end NUMINAMATH_GPT_valid_pair_l2127_212705


namespace NUMINAMATH_GPT_positive_integer_expression_l2127_212798

-- Define the existence conditions for a given positive integer n
theorem positive_integer_expression (n : ℕ) (h : 0 < n) : ∃ a b c : ℤ, (n = a^2 + b^2 + c^2 + c) := 
sorry

end NUMINAMATH_GPT_positive_integer_expression_l2127_212798


namespace NUMINAMATH_GPT_distance_train_A_when_meeting_l2127_212781

noncomputable def distance_traveled_by_train_A : ℝ :=
  let distance := 375
  let time_A := 36
  let time_B := 24
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let relative_speed := speed_A + speed_B
  let time_meeting := distance / relative_speed
  speed_A * time_meeting

theorem distance_train_A_when_meeting :
  distance_traveled_by_train_A = 150 := by
  sorry

end NUMINAMATH_GPT_distance_train_A_when_meeting_l2127_212781


namespace NUMINAMATH_GPT_lowest_price_per_component_l2127_212713

theorem lowest_price_per_component (cost_per_component shipping_per_component fixed_costs num_components : ℕ) 
  (h_cost_per_component : cost_per_component = 80)
  (h_shipping_per_component : shipping_per_component = 5)
  (h_fixed_costs : fixed_costs = 16500)
  (h_num_components : num_components = 150) :
  (cost_per_component + shipping_per_component) * num_components + fixed_costs = 29250 ∧
  29250 / 150 = 195 :=
by
  sorry

end NUMINAMATH_GPT_lowest_price_per_component_l2127_212713


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2127_212777

theorem necessary_and_sufficient_condition (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
    (∃ x : ℝ, 0 < x ∧ a^x = 2) ↔ (1 < a) := 
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2127_212777


namespace NUMINAMATH_GPT_matt_paper_piles_l2127_212793

theorem matt_paper_piles (n : ℕ) (h_n1 : 1000 < n) (h_n2 : n < 2000)
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 4 = 1)
  (h5 : n % 5 = 1) (h6 : n % 6 = 1) (h7 : n % 7 = 1)
  (h8 : n % 8 = 1) : 
  ∃ k : ℕ, k ≠ 1 ∧ k ≠ n ∧ n = 1681 ∧ k = 41 :=
by
  use 41
  sorry

end NUMINAMATH_GPT_matt_paper_piles_l2127_212793


namespace NUMINAMATH_GPT_choose_9_3_eq_84_l2127_212702

theorem choose_9_3_eq_84 : Nat.choose 9 3 = 84 :=
by
  sorry

end NUMINAMATH_GPT_choose_9_3_eq_84_l2127_212702


namespace NUMINAMATH_GPT_find_sample_size_l2127_212707

def sports_team (total: Nat) (soccer: Nat) (basketball: Nat) (table_tennis: Nat) : Prop :=
  total = soccer + basketball + table_tennis

def valid_sample_size (total: Nat) (n: Nat) :=
  (n > 0) ∧ (total % n == 0) ∧ (n % 6 == 0)

def systematic_sampling_interval (total: Nat) (n: Nat): Nat :=
  total / n

theorem find_sample_size :
  ∀ (total soccer basketball table_tennis: Nat),
  sports_team total soccer basketball table_tennis →
  total = 36 →
  soccer = 18 →
  basketball = 12 →
  table_tennis = 6 →
  (∃ n, valid_sample_size total n ∧ valid_sample_size (total - 1) (n + 1)) →
  ∃ n, n = 6 := by
  sorry

end NUMINAMATH_GPT_find_sample_size_l2127_212707


namespace NUMINAMATH_GPT_meryll_questions_l2127_212741

theorem meryll_questions (M P : ℕ) (h1 : (3/5 : ℚ) * M + (2/3 : ℚ) * P = 31) (h2 : P = 15) : M = 35 :=
sorry

end NUMINAMATH_GPT_meryll_questions_l2127_212741


namespace NUMINAMATH_GPT_greatest_possible_perimeter_l2127_212766

theorem greatest_possible_perimeter :
  ∃ (x : ℤ), x ≥ 4 ∧ x ≤ 5 ∧ (x + 4 * x + 18 = 43 ∧
    ∀ (y : ℤ), y ≥ 4 ∧ y ≤ 5 → y + 4 * y + 18 ≤ 43) :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_perimeter_l2127_212766


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2127_212788

def x : ℝ := 8
def y : ℝ := -3

theorem point_in_fourth_quadrant (h1 : x > 0) (h2 : y < 0) : (x > 0 ∧ y < 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2127_212788


namespace NUMINAMATH_GPT_find_f2_l2127_212714

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x y : ℝ, x * f y = y * f x) (h10 : f 10 = 30) : f 2 = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_f2_l2127_212714


namespace NUMINAMATH_GPT_solve_equation_l2127_212759

noncomputable def unique_solution (x : ℝ) : Prop :=
  2 * x * Real.log x + x - 1 = 0 → x = 1

-- Statement of our theorem
theorem solve_equation (x : ℝ) (h : 0 < x) : unique_solution x := sorry

end NUMINAMATH_GPT_solve_equation_l2127_212759


namespace NUMINAMATH_GPT_student_marks_l2127_212778

def max_marks : ℕ := 600
def passing_percentage : ℕ := 30
def fail_by : ℕ := 100

theorem student_marks :
  ∃ x : ℕ, x + fail_by = (passing_percentage * max_marks) / 100 :=
sorry

end NUMINAMATH_GPT_student_marks_l2127_212778


namespace NUMINAMATH_GPT_equilibrium_and_stability_l2127_212764

def system_in_equilibrium (G Q m r : ℝ) : Prop :=
    -- Stability conditions for points A and B, instability at C
    (G < (m-r)/(m-2*r)) ∧ (G > (m-r)/m)

-- Create a theorem to prove the system's equilibrium and stability
theorem equilibrium_and_stability (G Q m r : ℝ) 
  (h_gt_zero : G > 0) 
  (Q_gt_zero : Q > 0) 
  (m_gt_r : m > r) 
  (r_gt_zero : r > 0) : system_in_equilibrium G Q m r :=
by
  sorry   -- Proof omitted

end NUMINAMATH_GPT_equilibrium_and_stability_l2127_212764


namespace NUMINAMATH_GPT_sum_of_squares_l2127_212760

theorem sum_of_squares (a b c : ℝ) (h₁ : a + b + c = 31) (h₂ : ab + bc + ca = 10) :
  a^2 + b^2 + c^2 = 941 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2127_212760


namespace NUMINAMATH_GPT_divisors_of_10_factorial_larger_than_9_factorial_l2127_212701

theorem divisors_of_10_factorial_larger_than_9_factorial :
  ∃ n, n = 9 ∧ (∀ d, d ∣ (Nat.factorial 10) → d > (Nat.factorial 9) → d > (Nat.factorial 1) → n = 9) :=
sorry

end NUMINAMATH_GPT_divisors_of_10_factorial_larger_than_9_factorial_l2127_212701


namespace NUMINAMATH_GPT_quadratic_complete_square_l2127_212719

theorem quadratic_complete_square (b m : ℝ) (h1 : b > 0)
    (h2 : (x : ℝ) → (x + m)^2 + 8 = x^2 + bx + 20) : b = 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l2127_212719


namespace NUMINAMATH_GPT_max_ab_l2127_212729

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ab ≤ 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_l2127_212729


namespace NUMINAMATH_GPT_find_x_l2127_212720

theorem find_x (x : ℚ) (h : ⌊x⌋ + x = 15/4) : x = 15/4 := by
  sorry

end NUMINAMATH_GPT_find_x_l2127_212720


namespace NUMINAMATH_GPT_overlapped_squares_area_l2127_212752

/-- 
Theorem: The area of the figure formed by overlapping four identical squares, 
each with an area of \(3 \, \text{cm}^2\), and with an overlapping region 
that double-counts 6 small squares is \(10.875 \, \text{cm}^2\).
-/
theorem overlapped_squares_area (area_of_square : ℝ) (num_squares : ℕ) (overlap_small_squares : ℕ) :
  area_of_square = 3 → 
  num_squares = 4 → 
  overlap_small_squares = 6 →
  ∃ total_area : ℝ, total_area = (num_squares * area_of_square) - (overlap_small_squares * (area_of_square / 16)) ∧
                         total_area = 10.875 :=
by
  sorry

end NUMINAMATH_GPT_overlapped_squares_area_l2127_212752


namespace NUMINAMATH_GPT_max_t_subsets_of_base_set_l2127_212784

theorem max_t_subsets_of_base_set (n : ℕ)
  (A : Fin (2 * n + 1) → Set (Fin n))
  (h : ∀ i j k : Fin (2 * n + 1), i < j → j < k → (A i ∩ A k) ⊆ A j) : 
  ∃ t : ℕ, t = 2 * n + 1 :=
by
  sorry

end NUMINAMATH_GPT_max_t_subsets_of_base_set_l2127_212784


namespace NUMINAMATH_GPT_sluice_fill_time_l2127_212799

noncomputable def sluice_open_equal_time (x y : ℝ) (m : ℝ) : ℝ :=
  -- Define time (t) required for both sluice gates to be open equally to fill the lake
  m / 11

theorem sluice_fill_time :
  ∀ (x y : ℝ),
    (10 * x + 14 * y = 9900) →
    (18 * x + 12 * y = 9900) →
    sluice_open_equal_time x y 9900 = 900 := sorry

end NUMINAMATH_GPT_sluice_fill_time_l2127_212799


namespace NUMINAMATH_GPT_game_A_greater_game_B_l2127_212708

-- Defining the probabilities and independence condition
def P_H := 2 / 3
def P_T := 1 / 3
def independent_tosses := true

-- Game A Probability Definition
def P_A := (P_H ^ 3) + (P_T ^ 3)

-- Game B Probability Definition
def P_B := ((P_H ^ 2) + (P_T ^ 2)) ^ 2

-- Statement to be proved
theorem game_A_greater_game_B : P_A = (27:ℚ) / 81 ∧ P_B = (25:ℚ) / 81 ∧ ((27:ℚ) / 81 - (25:ℚ) / 81 = (2:ℚ) / 81) := 
by
  -- P_A has already been computed: 1/3 = 27/81
  -- P_B has already been computed: 25/81
  sorry

end NUMINAMATH_GPT_game_A_greater_game_B_l2127_212708


namespace NUMINAMATH_GPT_johns_apartment_number_l2127_212716

theorem johns_apartment_number (car_reg : Nat) (apartment_num : Nat) 
  (h_car_reg_sum : car_reg = 834205) 
  (h_car_digits : (8 + 3 + 4 + 2 + 0 + 5 = 22)) 
  (h_apartment_digits : ∃ (d1 d2 d3 : Nat), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ d1 + d2 + d3 = 22) :
  apartment_num = 985 :=
by
  sorry

end NUMINAMATH_GPT_johns_apartment_number_l2127_212716


namespace NUMINAMATH_GPT_measure_of_angle_Q_l2127_212776

variables (R S T U Q : ℝ)
variables (angle_R angle_S angle_T angle_U : ℝ)

-- Given conditions
def sum_of_angles_in_pentagon : ℝ := 540
def angle_measure_R : ℝ := 120
def angle_measure_S : ℝ := 94
def angle_measure_T : ℝ := 115
def angle_measure_U : ℝ := 101

theorem measure_of_angle_Q :
  angle_R = angle_measure_R →
  angle_S = angle_measure_S →
  angle_T = angle_measure_T →
  angle_U = angle_measure_U →
  (angle_R + angle_S + angle_T + angle_U + Q = sum_of_angles_in_pentagon) →
  Q = 110 :=
by { sorry }

end NUMINAMATH_GPT_measure_of_angle_Q_l2127_212776


namespace NUMINAMATH_GPT_determine_denominator_of_fraction_l2127_212709

theorem determine_denominator_of_fraction (x : ℝ) (h : 57 / x = 0.0114) : x = 5000 :=
by
  sorry

end NUMINAMATH_GPT_determine_denominator_of_fraction_l2127_212709


namespace NUMINAMATH_GPT_value_of_expression_l2127_212706

theorem value_of_expression (m a b c d : ℚ) 
  (hm : |m + 1| = 4)
  (hab : a = -b) 
  (hcd : c * d = 1) :
  a + b + 3 * c * d - m = 0 ∨ a + b + 3 * c * d - m = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2127_212706


namespace NUMINAMATH_GPT_multiplication_is_correct_l2127_212770

theorem multiplication_is_correct : 209 * 209 = 43681 := sorry

end NUMINAMATH_GPT_multiplication_is_correct_l2127_212770


namespace NUMINAMATH_GPT_inequality_1_inequality_2_l2127_212728

variable (a b : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom sum_of_cubes_eq_two : a^3 + b^3 = 2

-- Question 1
theorem inequality_1 : (a + b) * (a^5 + b^5) ≥ 4 :=
by
  sorry

-- Question 2
theorem inequality_2 : a + b ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_l2127_212728


namespace NUMINAMATH_GPT_additional_slow_workers_needed_l2127_212797

-- Definitions based on conditions
def production_per_worker_fast (m : ℕ) (n : ℕ) (a : ℕ) : ℚ := m / (n * a)
def production_per_worker_slow (m : ℕ) (n : ℕ) (b : ℕ) : ℚ := m / (n * b)

def required_daily_production (p : ℕ) (q : ℕ) : ℚ := p / q

def contribution_fast_workers (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (m * c) / (n * a)

def remaining_production (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (p / q) - ((m * c) / (n * a))

def required_slow_workers (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : ℚ :=
  ((p * n * a - q * m * c) * b) / (q * m * a)

theorem additional_slow_workers_needed (m n a b p q c : ℕ) :
  required_slow_workers p q m n a b c = ((p * n * a - q * m * c) * b) / (q * m * a) := by
  sorry

end NUMINAMATH_GPT_additional_slow_workers_needed_l2127_212797


namespace NUMINAMATH_GPT_range_of_a_l2127_212782

theorem range_of_a (a : ℝ) : 
  (∀ (x1 : ℝ), ∃ (x2 : ℝ), |x1| = Real.log (a * x2^2 - 4 * x2 + 1)) → (0 ≤ a) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2127_212782


namespace NUMINAMATH_GPT_bees_flew_in_l2127_212772

theorem bees_flew_in (initial_bees additional_bees total_bees : ℕ) 
  (h1 : initial_bees = 16) (h2 : total_bees = 25) 
  (h3 : initial_bees + additional_bees = total_bees) : additional_bees = 9 :=
by sorry

end NUMINAMATH_GPT_bees_flew_in_l2127_212772


namespace NUMINAMATH_GPT_greatest_number_of_bouquets_l2127_212732

/--
Sara has 42 red flowers, 63 yellow flowers, and 54 blue flowers.
She wants to make bouquets with the same number of each color flower in each bouquet.
Prove that the greatest number of bouquets she can make is 21.
-/
theorem greatest_number_of_bouquets (red yellow blue : ℕ) (h_red : red = 42) (h_yellow : yellow = 63) (h_blue : blue = 54) :
  Nat.gcd (Nat.gcd red yellow) blue = 21 :=
by
  rw [h_red, h_yellow, h_blue]
  sorry

end NUMINAMATH_GPT_greatest_number_of_bouquets_l2127_212732


namespace NUMINAMATH_GPT_man_swim_upstream_distance_l2127_212755

theorem man_swim_upstream_distance (dist_downstream : ℝ) (time_downstream : ℝ) (time_upstream : ℝ) (speed_still_water : ℝ) 
  (effective_speed_downstream : ℝ) (speed_current : ℝ) (effective_speed_upstream : ℝ) (dist_upstream : ℝ) :
  dist_downstream = 36 →
  time_downstream = 6 →
  time_upstream = 6 →
  speed_still_water = 4.5 →
  effective_speed_downstream = dist_downstream / time_downstream →
  effective_speed_downstream = speed_still_water + speed_current →
  effective_speed_upstream = speed_still_water - speed_current →
  dist_upstream = effective_speed_upstream * time_upstream →
  dist_upstream = 18 :=
by
  intros h_dist_downstream h_time_downstream h_time_upstream h_speed_still_water
         h_effective_speed_downstream h_eq_speed_current h_effective_speed_upstream h_dist_upstream
  sorry

end NUMINAMATH_GPT_man_swim_upstream_distance_l2127_212755


namespace NUMINAMATH_GPT_quadratic_root_ratio_l2127_212789

theorem quadratic_root_ratio (k : ℝ) (h : ∃ r : ℝ, r ≠ 0 ∧ 3 * r * r = k * r - 12 * r + k ∧ r * r = k + 9 * r - k) : k = 27 :=
sorry

end NUMINAMATH_GPT_quadratic_root_ratio_l2127_212789


namespace NUMINAMATH_GPT_woody_savings_l2127_212723

-- Definitions from conditions
def console_cost : Int := 282
def weekly_allowance : Int := 24
def saving_weeks : Int := 10

-- Theorem to prove that the amount Woody already has is $42
theorem woody_savings :
  (console_cost - (weekly_allowance * saving_weeks)) = 42 := 
by
  sorry

end NUMINAMATH_GPT_woody_savings_l2127_212723


namespace NUMINAMATH_GPT_intersection_M_N_l2127_212734

open Set

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2127_212734


namespace NUMINAMATH_GPT_find_pairs_l2127_212744
open Nat

theorem find_pairs (x p : ℕ) (hp : p.Prime) (hxp : x ≤ 2 * p) (hdiv : x^(p-1) ∣ (p-1)^x + 1) : 
  (x = 1 ∧ p.Prime) ∨ (x = 2 ∧ p = 2) ∨ (x = 1 ∧ p.Prime) ∨ (x = 3 ∧ p = 3) := 
by
  sorry


end NUMINAMATH_GPT_find_pairs_l2127_212744


namespace NUMINAMATH_GPT_reflected_ray_eq_l2127_212743

theorem reflected_ray_eq:
  ∀ (x y : ℝ), 
    (3 * x + 4 * y - 18 = 0) ∧ (3 * x + 2 * y - 12 = 0) →
    63 * x + 16 * y - 174 = 0 :=
by
  intro x y
  intro h
  sorry

end NUMINAMATH_GPT_reflected_ray_eq_l2127_212743


namespace NUMINAMATH_GPT_area_of_picture_l2127_212700

theorem area_of_picture
  (paper_width : ℝ)
  (paper_height : ℝ)
  (left_margin : ℝ)
  (right_margin : ℝ)
  (top_margin_cm : ℝ)
  (bottom_margin_cm : ℝ)
  (cm_per_inch : ℝ)
  (converted_top_margin : ℝ := top_margin_cm * (1 / cm_per_inch))
  (converted_bottom_margin : ℝ := bottom_margin_cm * (1 / cm_per_inch))
  (picture_width : ℝ := paper_width - left_margin - right_margin)
  (picture_height : ℝ := paper_height - converted_top_margin - converted_bottom_margin)
  (area : ℝ := picture_width * picture_height)
  (h1 : paper_width = 8.5)
  (h2 : paper_height = 10)
  (h3 : left_margin = 1.5)
  (h4 : right_margin = 1.5)
  (h5 : top_margin_cm = 2)
  (h6 : bottom_margin_cm = 2.5)
  (h7 : cm_per_inch = 2.54)
  : area = 45.255925 :=
by sorry

end NUMINAMATH_GPT_area_of_picture_l2127_212700


namespace NUMINAMATH_GPT_booklet_cost_l2127_212796

theorem booklet_cost (b : ℝ) : 
  (10 * b < 15) ∧ (12 * b > 17) → b = 1.42 := by
  sorry

end NUMINAMATH_GPT_booklet_cost_l2127_212796


namespace NUMINAMATH_GPT_min_value_of_sum_of_reciprocals_l2127_212710

theorem min_value_of_sum_of_reciprocals 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.log (1 / a + 1 / b) / Real.log 4 = Real.log (1 / Real.sqrt (a * b)) / Real.log 2) : 
  1 / a + 1 / b ≥ 4 := 
by 
  sorry

end NUMINAMATH_GPT_min_value_of_sum_of_reciprocals_l2127_212710


namespace NUMINAMATH_GPT_right_triangle_perpendicular_ratio_l2127_212771

theorem right_triangle_perpendicular_ratio {a b c r s : ℝ}
 (h : a^2 + b^2 = c^2)
 (perpendicular : r + s = c)
 (ratio_ab : a / b = 2 / 3) :
 r / s = 4 / 9 :=
sorry

end NUMINAMATH_GPT_right_triangle_perpendicular_ratio_l2127_212771


namespace NUMINAMATH_GPT_find_english_marks_l2127_212763

variable (mathematics science social_studies english biology : ℕ)
variable (average_marks : ℕ)
variable (number_of_subjects : ℕ := 5)

-- Conditions
axiom score_math : mathematics = 76
axiom score_sci : science = 65
axiom score_ss : social_studies = 82
axiom score_bio : biology = 95
axiom average : average_marks = 77

-- The proof problem
theorem find_english_marks :
  english = 67 :=
  sorry

end NUMINAMATH_GPT_find_english_marks_l2127_212763


namespace NUMINAMATH_GPT_remainder_when_divided_by_10_l2127_212757

theorem remainder_when_divided_by_10 : 
  (2468 * 7391 * 90523) % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_10_l2127_212757


namespace NUMINAMATH_GPT_quadratic_roots_relation_l2127_212724

theorem quadratic_roots_relation (m p q : ℝ) (h_m_ne_zero : m ≠ 0) (h_p_ne_zero : p ≠ 0) (h_q_ne_zero : q ≠ 0) :
  (∀ r1 r2 : ℝ, (r1 + r2 = -q ∧ r1 * r2 = m) → (3 * r1 + 3 * r2 = -m ∧ (3 * r1) * (3 * r2) = p)) →
  p / q = 27 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_quadratic_roots_relation_l2127_212724


namespace NUMINAMATH_GPT_drinkable_amount_l2127_212761

variable {LiquidBeforeTest : ℕ}
variable {Threshold : ℕ}

def can_drink_more (LiquidBeforeTest : ℕ) (Threshold : ℕ): ℕ :=
  Threshold - LiquidBeforeTest

theorem drinkable_amount :
  LiquidBeforeTest = 24 ∧ Threshold = 32 →
  can_drink_more LiquidBeforeTest Threshold = 8 := by
  sorry

end NUMINAMATH_GPT_drinkable_amount_l2127_212761


namespace NUMINAMATH_GPT_fans_attended_show_l2127_212779

-- Definitions from the conditions
def total_seats : ℕ := 60000
def sold_percentage : ℝ := 0.75
def fans_stayed_home : ℕ := 5000

-- The proof statement
theorem fans_attended_show :
  let sold_seats := sold_percentage * total_seats
  let fans_attended := sold_seats - fans_stayed_home
  fans_attended = 40000 :=
by
  -- Auto-generated proof placeholder.
  sorry

end NUMINAMATH_GPT_fans_attended_show_l2127_212779


namespace NUMINAMATH_GPT_initially_calculated_average_l2127_212737

theorem initially_calculated_average :
  ∀ (S : ℕ), (S / 10 = 18) →
  ((S - 46 + 26) / 10 = 16) :=
by
  sorry

end NUMINAMATH_GPT_initially_calculated_average_l2127_212737


namespace NUMINAMATH_GPT_cost_of_600_candies_l2127_212738

-- Definitions based on conditions
def costOfBox : ℕ := 6       -- The cost of one box of 25 candies in dollars
def boxSize   : ℕ := 25      -- The number of candies in one box
def cost (n : ℕ) : ℕ := (n / boxSize) * costOfBox -- The cost function for n candies

-- Theorem to be proven
theorem cost_of_600_candies : cost 600 = 144 :=
by sorry

end NUMINAMATH_GPT_cost_of_600_candies_l2127_212738


namespace NUMINAMATH_GPT_wilson_fraction_l2127_212722

theorem wilson_fraction (N : ℝ) (result : ℝ) (F : ℝ) (h1 : N = 8) (h2 : result = 16 / 3) (h3 : N - F * N = result) : F = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_wilson_fraction_l2127_212722


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2127_212783

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0) ↔ (-3 < k ∧ k < 0) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2127_212783


namespace NUMINAMATH_GPT_greatest_integer_sum_of_integers_l2127_212726

-- Definition of the quadratic function
def quadratic_expr (n : ℤ) : ℤ := n^2 - 15 * n + 56

-- The greatest integer n such that quadratic_expr n ≤ 0
theorem greatest_integer (n : ℤ) (h : quadratic_expr n ≤ 0) : n ≤ 8 := 
  sorry

-- All integers that satisfy the quadratic inequality
theorem sum_of_integers (sum_n : ℤ) (h : ∀ n : ℤ, 7 ≤ n ∧ n ≤ 8 → quadratic_expr n ≤ 0) 
  (sum_eq : sum_n = 7 + 8) : sum_n = 15 :=
  sorry

end NUMINAMATH_GPT_greatest_integer_sum_of_integers_l2127_212726


namespace NUMINAMATH_GPT_percentage_girls_not_attended_college_l2127_212749

-- Definitions based on given conditions
def total_boys : ℕ := 300
def total_girls : ℕ := 240
def percent_boys_not_attended_college : ℚ := 0.30
def percent_class_attended_college : ℚ := 0.70

-- The goal is to prove that the percentage of girls who did not attend college is 30%
theorem percentage_girls_not_attended_college 
  (total_boys : ℕ)
  (total_girls : ℕ)
  (percent_boys_not_attended_college : ℚ)
  (percent_class_attended_college : ℚ)
  (total_students := total_boys + total_girls)
  (boys_not_attended := percent_boys_not_attended_college * total_boys)
  (students_attended := percent_class_attended_college * total_students)
  (students_not_attended := total_students - students_attended)
  (girls_not_attended := students_not_attended - boys_not_attended) :
  (girls_not_attended / total_girls) * 100 = 30 := 
  sorry

end NUMINAMATH_GPT_percentage_girls_not_attended_college_l2127_212749


namespace NUMINAMATH_GPT_problem_solution_l2127_212762

theorem problem_solution
  (x y : ℝ)
  (h1 : (x - y)^2 = 25)
  (h2 : x * y = -10) :
  x^2 + y^2 = 5 := sorry

end NUMINAMATH_GPT_problem_solution_l2127_212762


namespace NUMINAMATH_GPT_minimum_total_number_of_balls_l2127_212795

theorem minimum_total_number_of_balls (x y z t : ℕ) 
  (h1 : x ≥ 4)
  (h2 : x ≥ 3 ∧ y ≥ 1)
  (h3 : x ≥ 2 ∧ y ≥ 1 ∧ z ≥ 1)
  (h4 : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ t ≥ 1) :
  x + y + z + t = 21 :=
  sorry

end NUMINAMATH_GPT_minimum_total_number_of_balls_l2127_212795


namespace NUMINAMATH_GPT_equilateral_right_triangle_impossible_l2127_212746
-- Import necessary library

-- Define the conditions and the problem statement
theorem equilateral_right_triangle_impossible :
  ¬(∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A = B ∧ B = C ∧ (A^2 + B^2 = C^2) ∧ (A + B + C = 180)) := sorry

end NUMINAMATH_GPT_equilateral_right_triangle_impossible_l2127_212746


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l2127_212773

theorem quadratic_distinct_real_roots (a : ℝ) (h : a ≠ 1) : 
(a < 2) → 
(∃ x y : ℝ, x ≠ y ∧ (a-1)*x^2 - 2*x + 1 = 0 ∧ (a-1)*y^2 - 2*y + 1 = 0) :=
sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l2127_212773


namespace NUMINAMATH_GPT_coffee_shop_distance_l2127_212751

theorem coffee_shop_distance (resort_distance mall_distance : ℝ) 
  (coffee_dist : ℝ)
  (h_resort_distance : resort_distance = 400) 
  (h_mall_distance : mall_distance = 700)
  (h_equidistant : ∀ S, (S - resort_distance) ^ 2 + resort_distance ^ 2 = S ^ 2 ∧ 
  (mall_distance - S) ^ 2 + resort_distance ^ 2 = S ^ 2 → coffee_dist = S):
  coffee_dist = 464 := 
sorry

end NUMINAMATH_GPT_coffee_shop_distance_l2127_212751


namespace NUMINAMATH_GPT_frustumViews_l2127_212767

-- Define the notion of a frustum
structure Frustum where
  -- You may add necessary geometric properties of a frustum if needed
  
-- Define a function to describe the view of the frustum
def frontView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def sideView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def topView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type

-- Define the properties of the views
def isCongruentIsoscelesTrapezoid (fig : Type) : Prop := sorry -- Define property for congruent isosceles trapezoid
def isTwoConcentricCircles (fig : Type) : Prop := sorry -- Define property for two concentric circles

-- State the theorem based on the given problem
theorem frustumViews (f : Frustum) :
  isCongruentIsoscelesTrapezoid (frontView f) ∧ 
  isCongruentIsoscelesTrapezoid (sideView f) ∧ 
  isTwoConcentricCircles (topView f) := 
sorry

end NUMINAMATH_GPT_frustumViews_l2127_212767


namespace NUMINAMATH_GPT_find_number_l2127_212758

theorem find_number (x : ℝ) (h : (x + 0.005) / 2 = 0.2025) : x = 0.400 :=
sorry

end NUMINAMATH_GPT_find_number_l2127_212758


namespace NUMINAMATH_GPT_bold_o_lit_cells_l2127_212756

-- Define the conditions
def grid_size : ℕ := 5
def original_o_lit_cells : ℕ := 12 -- Number of cells lit in the original 'o'
def additional_lit_cells : ℕ := 12 -- Additional cells lit in the bold 'o'

-- Define the property to be proved
theorem bold_o_lit_cells : (original_o_lit_cells + additional_lit_cells) = 24 :=
by
  -- computation skipped
  sorry

end NUMINAMATH_GPT_bold_o_lit_cells_l2127_212756


namespace NUMINAMATH_GPT_squares_not_all_congruent_l2127_212739

/-- Proof that the statement "all squares are congruent to each other" is false. -/
theorem squares_not_all_congruent : ¬(∀ (a b : ℝ), a = b ↔ a = b) :=
by 
  sorry

end NUMINAMATH_GPT_squares_not_all_congruent_l2127_212739


namespace NUMINAMATH_GPT_correct_survey_method_l2127_212774

def service_life_of_light_tubes (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def viewership_rate_of_spring_festival_gala (survey_method : String) : Prop :=
  survey_method = "comprehensive"

def crash_resistance_of_cars (survey_method : String) : Prop :=
  survey_method = "sample"

def fastest_student_for_sports_meeting (survey_method : String) : Prop :=
  survey_method = "sample"

theorem correct_survey_method :
  ¬(service_life_of_light_tubes "comprehensive") ∧
  ¬(viewership_rate_of_spring_festival_gala "comprehensive") ∧
  ¬(crash_resistance_of_cars "sample") ∧
  (fastest_student_for_sports_meeting "sample") :=
sorry

end NUMINAMATH_GPT_correct_survey_method_l2127_212774


namespace NUMINAMATH_GPT_solve_for_k_l2127_212794

theorem solve_for_k (p q : ℝ) (k : ℝ) (hpq : 3 * p^2 + 6 * p + k = 0) (hq : 3 * q^2 + 6 * q + k = 0) 
    (h_diff : |p - q| = (1 / 2) * (p^2 + q^2)) : k = -16 + 12 * Real.sqrt 2 ∨ k = -16 - 12 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l2127_212794


namespace NUMINAMATH_GPT_binary_to_decimal_l2127_212754

theorem binary_to_decimal (x : ℕ) (h : x = 0b110010) : x = 50 := by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l2127_212754


namespace NUMINAMATH_GPT_largest_apartment_size_l2127_212730

theorem largest_apartment_size (cost_per_sqft : ℝ) (budget : ℝ) (s : ℝ) 
    (h₁ : cost_per_sqft = 1.20) 
    (h₂ : budget = 600) 
    (h₃ : 1.20 * s = 600) : 
    s = 500 := 
  sorry

end NUMINAMATH_GPT_largest_apartment_size_l2127_212730


namespace NUMINAMATH_GPT_unique_positive_integer_appending_digits_eq_sum_l2127_212742

-- Define the problem in terms of Lean types and properties
theorem unique_positive_integer_appending_digits_eq_sum :
  ∃! (A : ℕ), (A > 0) ∧ (∃ (B : ℕ), (0 ≤ B ∧ B < 1000) ∧ (1000 * A + B = (A * (A + 1)) / 2)) :=
sorry

end NUMINAMATH_GPT_unique_positive_integer_appending_digits_eq_sum_l2127_212742


namespace NUMINAMATH_GPT_curtains_length_needed_l2127_212750

def room_height_feet : ℕ := 8
def additional_material_inches : ℕ := 5

def height_in_inches : ℕ := room_height_feet * 12

def total_length_curtains : ℕ := height_in_inches + additional_material_inches

theorem curtains_length_needed : total_length_curtains = 101 := by
  sorry

end NUMINAMATH_GPT_curtains_length_needed_l2127_212750


namespace NUMINAMATH_GPT_sum_of_sides_of_regular_pentagon_l2127_212712

theorem sum_of_sides_of_regular_pentagon (s : ℝ) (n : ℕ)
    (h : s = 15) (hn : n = 5) : 5 * 15 = 75 :=
sorry

end NUMINAMATH_GPT_sum_of_sides_of_regular_pentagon_l2127_212712


namespace NUMINAMATH_GPT_find_value_of_sum_of_squares_l2127_212747

theorem find_value_of_sum_of_squares (x y : ℝ) (h : x^2 + y^2 + x^2 * y^2 - 4 * x * y + 1 = 0) :
  (x + y)^2 = 4 :=
sorry

end NUMINAMATH_GPT_find_value_of_sum_of_squares_l2127_212747


namespace NUMINAMATH_GPT_no_arithmetic_progression_exists_l2127_212703

theorem no_arithmetic_progression_exists 
  (a : ℕ) (d : ℕ) (a_n : ℕ → ℕ) 
  (h_seq : ∀ n, a_n n = a + n * d) :
  ¬ ∃ (a_n : ℕ → ℕ), (∀ n, a_n (n+1) > a_n n ∧ 
  ∀ n, (a_n n) * (a_n (n+1)) * (a_n (n+2)) * (a_n (n+3)) * (a_n (n+4)) * 
        (a_n (n+5)) * (a_n (n+6)) * (a_n (n+7)) * (a_n (n+8)) * (a_n (n+9)) % 
        ((a_n n) + (a_n (n+1)) + (a_n (n+2)) + (a_n (n+3)) + (a_n (n+4)) + 
         (a_n (n+5)) + (a_n (n+6)) + (a_n (n+7)) + (a_n (n+8)) + (a_n (n+9)) ) = 0 ) := 
sorry

end NUMINAMATH_GPT_no_arithmetic_progression_exists_l2127_212703
