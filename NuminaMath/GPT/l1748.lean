import Mathlib

namespace functional_equation_solution_l1748_174887

noncomputable def quadratic_polynomial (P : ℝ → ℝ) :=
  ∃ a b c : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x + c

theorem functional_equation_solution (P : ℝ → ℝ) (f : ℝ → ℝ)
  (h_poly : quadratic_polynomial P)
  (h_additive : ∀ x y : ℝ, f (x + y) = f x + f y)
  (h_preserves_poly : ∀ x : ℝ, f (P x) = f x) :
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_equation_solution_l1748_174887


namespace right_handed_players_total_l1748_174842

theorem right_handed_players_total
    (total_players : ℕ)
    (throwers : ℕ)
    (left_handed : ℕ)
    (right_handed : ℕ) :
    total_players = 150 →
    throwers = 60 →
    left_handed = (total_players - throwers) / 2 →
    right_handed = (total_players - throwers) / 2 →
    total_players - throwers = 2 * left_handed →
    left_handed + right_handed + throwers = total_players →
    ∀ throwers : ℕ, throwers = 60 →
    right_handed + throwers = 105 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end right_handed_players_total_l1748_174842


namespace max_odd_integers_chosen_l1748_174862

theorem max_odd_integers_chosen (a b c d e f : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h_prod_even : a * b * c * d * e * f % 2 = 0) : 
  (∀ n : ℕ, n = 5 → ∃ a b c d e, (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1) ∧ f % 2 = 0) :=
sorry

end max_odd_integers_chosen_l1748_174862


namespace percentage_increase_l1748_174836

variables (J T P : ℝ)

def income_conditions (J T P : ℝ) : Prop :=
  (T = 0.5 * J) ∧ (P = 0.8 * J)

theorem percentage_increase (J T P : ℝ) (h : income_conditions J T P) :
  ((P / T) - 1) * 100 = 60 :=
by
  sorry

end percentage_increase_l1748_174836


namespace consecutive_odd_integers_l1748_174803

theorem consecutive_odd_integers (x : ℤ) (h : x + 4 = 15) : 3 * x - 2 * (x + 4) = 3 :=
by
  sorry

end consecutive_odd_integers_l1748_174803


namespace vector_subtraction_l1748_174831

def a : ℝ × ℝ := (5, 3)
def b : ℝ × ℝ := (1, -2)
def scalar : ℝ := 2

theorem vector_subtraction :
  a.1 - scalar * b.1 = 3 ∧ a.2 - scalar * b.2 = 7 :=
by {
  -- here goes the proof
  sorry
}

end vector_subtraction_l1748_174831


namespace sum_of_dimensions_eq_18_sqrt_1_5_l1748_174884

theorem sum_of_dimensions_eq_18_sqrt_1_5 (P Q R : ℝ) (h1 : P * Q = 30) (h2 : P * R = 50) (h3 : Q * R = 90) :
  P + Q + R = 18 * Real.sqrt 1.5 :=
sorry

end sum_of_dimensions_eq_18_sqrt_1_5_l1748_174884


namespace total_boat_licenses_l1748_174898

/-- A state modifies its boat license requirements to include any one of the letters A, M, or S
followed by any six digits. How many different boat licenses can now be issued? -/
theorem total_boat_licenses : 
  let letters := 3
  let digits := 10
  letters * digits^6 = 3000000 := by
  sorry

end total_boat_licenses_l1748_174898


namespace order_of_scores_l1748_174843

variables (E L T N : ℝ)

-- Conditions
axiom Lana_condition_1 : L ≠ T
axiom Lana_condition_2 : L ≠ N
axiom Lana_condition_3 : T ≠ N

axiom Tom_condition : ∃ L' E', L' ≠ T ∧ E' > L' ∧ E' ≠ T ∧ E' ≠ L' 

axiom Nina_condition : N < E

-- Proof statement
theorem order_of_scores :
  N < L ∧ L < T :=
sorry

end order_of_scores_l1748_174843


namespace students_calculation_l1748_174809

def number_of_stars : ℝ := 3.0
def students_per_star : ℝ := 41.33333333
def total_students : ℝ := 124

theorem students_calculation : number_of_stars * students_per_star = total_students := 
by
  sorry

end students_calculation_l1748_174809


namespace multiplication_value_l1748_174874

theorem multiplication_value : 725143 * 999999 = 725142274857 :=
by
  sorry

end multiplication_value_l1748_174874


namespace tank_capacity_correct_l1748_174886

-- Define rates and times for each pipe
def rate_a : ℕ := 200 -- in liters per minute
def rate_b : ℕ := 50 -- in liters per minute
def rate_c : ℕ := 25 -- in liters per minute

def time_a : ℕ := 1 -- pipe A open time in minutes
def time_b : ℕ := 2 -- pipe B open time in minutes
def time_c : ℕ := 2 -- pipe C open time in minutes

def cycle_time : ℕ := time_a + time_b + time_c -- total time for one cycle in minutes
def total_time : ℕ := 40 -- total time to fill the tank in minutes

-- Net water added in one cycle
def net_water_in_cycle : ℕ :=
  (rate_a * time_a) + (rate_b * time_b) - (rate_c * time_c)

-- Number of cycles needed to fill the tank
def number_of_cycles : ℕ :=
  total_time / cycle_time

-- Total capacity of the tank
def tank_capacity : ℕ :=
  number_of_cycles * net_water_in_cycle

-- The hypothesis to prove
theorem tank_capacity_correct :
  tank_capacity = 2000 :=
  by
    sorry

end tank_capacity_correct_l1748_174886


namespace ping_pong_shaved_head_ping_pong_upset_l1748_174888

noncomputable def probability_shaved_head (pA pB : ℚ) : ℚ :=
  pA^3 + pB^3

noncomputable def probability_upset (pB pA : ℚ) : ℚ :=
  (pB^3) + (3 * (pB^2) * pA) + (6 * (pA^2) * (pB^2))

theorem ping_pong_shaved_head :
  probability_shaved_head (2/3) (1/3) = 1/3 := 
by
  sorry

theorem ping_pong_upset :
  probability_upset (1/3) (2/3) = 11/27 := 
by
  sorry

end ping_pong_shaved_head_ping_pong_upset_l1748_174888


namespace H_perimeter_is_44_l1748_174812

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

end H_perimeter_is_44_l1748_174812


namespace find_smallest_y_l1748_174844

noncomputable def x : ℕ := 5 * 15 * 35

def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, m ^ 4 = n

theorem find_smallest_y : ∃ y : ℕ, y > 0 ∧ is_perfect_fourth_power (x * y) ∧ y = 46485 := by
  sorry

end find_smallest_y_l1748_174844


namespace casey_pumping_time_l1748_174817

structure PlantRow :=
  (rows : ℕ) (plants_per_row : ℕ) (water_per_plant : ℚ)

structure Animal :=
  (count : ℕ) (water_per_animal : ℚ)

def morning_pump_rate := 3 -- gallons per minute
def afternoon_pump_rate := 5 -- gallons per minute

def corn := PlantRow.mk 4 15 0.5
def pumpkin := PlantRow.mk 3 10 0.8
def pigs := Animal.mk 10 4
def ducks := Animal.mk 20 0.25
def cows := Animal.mk 5 8

def total_water_needed_for_plants (corn pumpkin : PlantRow) : ℚ :=
  (corn.rows * corn.plants_per_row * corn.water_per_plant) +
  (pumpkin.rows * pumpkin.plants_per_row * pumpkin.water_per_plant)

def total_water_needed_for_animals (pigs ducks cows : Animal) : ℚ :=
  (pigs.count * pigs.water_per_animal) +
  (ducks.count * ducks.water_per_animal) +
  (cows.count * cows.water_per_animal)

def time_to_pump (total_water pump_rate : ℚ) : ℚ :=
  total_water / pump_rate

theorem casey_pumping_time :
  let total_water_plants := total_water_needed_for_plants corn pumpkin
  let total_water_animals := total_water_needed_for_animals pigs ducks cows
  let time_morning := time_to_pump total_water_plants morning_pump_rate
  let time_afternoon := time_to_pump total_water_animals afternoon_pump_rate
  time_morning + time_afternoon = 35 := by
sorry

end casey_pumping_time_l1748_174817


namespace team_size_is_nine_l1748_174864

noncomputable def number_of_workers (n x y : ℕ) : ℕ :=
  if 7 * n = (n - 2) * x ∧ 7 * n = (n - 6) * y then n else 0

theorem team_size_is_nine (x y : ℕ) :
  number_of_workers 9 x y = 9 :=
by
  sorry

end team_size_is_nine_l1748_174864


namespace sum_D_E_correct_sum_of_all_possible_values_of_D_E_l1748_174853

theorem sum_D_E_correct :
  ∀ (D E : ℕ), (D < 10) → (E < 10) →
  (∃ k : ℕ, (10^8 * D + 4650000 + 1000 * E + 32) = 7 * k) →
  D + E = 1 ∨ D + E = 8 ∨ D + E = 15 :=
by sorry

theorem sum_of_all_possible_values_of_D_E :
  (1 + 8 + 15) = 24 :=
by norm_num

end sum_D_E_correct_sum_of_all_possible_values_of_D_E_l1748_174853


namespace trig_identity_l1748_174877

theorem trig_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) : Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end trig_identity_l1748_174877


namespace permutation_of_digits_l1748_174813

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

end permutation_of_digits_l1748_174813


namespace number_divisors_product_l1748_174863

theorem number_divisors_product :
  ∃ N : ℕ, (∃ a b : ℕ, N = 3^a * 5^b ∧ (N^((a+1)*(b+1) / 2)) = 3^30 * 5^40) ∧ N = 3^3 * 5^4 :=
sorry

end number_divisors_product_l1748_174863


namespace value_2_stddevs_less_than_mean_l1748_174896

-- Definitions based on the conditions
def mean : ℝ := 10.5
def stddev : ℝ := 1
def value := mean - 2 * stddev

-- Theorem we aim to prove
theorem value_2_stddevs_less_than_mean : value = 8.5 := by
  -- proof will go here
  sorry

end value_2_stddevs_less_than_mean_l1748_174896


namespace sum_of_digits_div_by_11_in_consecutive_39_l1748_174806

-- Define the sum of digits function for natural numbers.
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem statement.
theorem sum_of_digits_div_by_11_in_consecutive_39 :
  ∀ (N : ℕ), ∃ k : ℕ, k < 39 ∧ (sum_of_digits (N + k)) % 11 = 0 :=
by sorry

end sum_of_digits_div_by_11_in_consecutive_39_l1748_174806


namespace carol_blocks_l1748_174892

theorem carol_blocks (initial_blocks : ℕ) (blocks_lost : ℕ) (final_blocks : ℕ) : 
  initial_blocks = 42 → blocks_lost = 25 → final_blocks = initial_blocks - blocks_lost → final_blocks = 17 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carol_blocks_l1748_174892


namespace problem_statement_l1748_174854

theorem problem_statement (x y : ℤ) (h1 : x = 8) (h2 : y = 3) :
  (x - 2 * y) * (x + 2 * y) = 28 :=
by
  sorry

end problem_statement_l1748_174854


namespace problem_not_true_equation_l1748_174814

theorem problem_not_true_equation
  (a b : ℝ)
  (h : a / b = 2 / 3) : a / b ≠ (a + 2) / (b + 2) := 
sorry

end problem_not_true_equation_l1748_174814


namespace hike_took_one_hour_l1748_174856

-- Define the constants and conditions
def initial_cups : ℕ := 6
def remaining_cups : ℕ := 1
def leak_rate : ℕ := 1 -- cups per hour
def drank_last_mile : ℚ := 1
def drank_first_3_miles_per_mile : ℚ := 2/3
def first_3_miles : ℕ := 3

-- Define the hike duration we want to prove
def hike_duration := 1

-- The total water drank
def total_drank := drank_last_mile + drank_first_3_miles_per_mile * first_3_miles

-- Prove the hike took 1 hour
theorem hike_took_one_hour :
  ∃ hours : ℕ, (initial_cups - remaining_cups = hours * leak_rate + total_drank) ∧ (hours = hike_duration) :=
by
  sorry

end hike_took_one_hour_l1748_174856


namespace stratified_sample_selection_l1748_174801

def TotalStudents : ℕ := 900
def FirstYearStudents : ℕ := 300
def SecondYearStudents : ℕ := 200
def ThirdYearStudents : ℕ := 400
def SampleSize : ℕ := 45
def SamplingRatio : ℚ := 1 / 20

theorem stratified_sample_selection :
  (FirstYearStudents * SamplingRatio = 15) ∧
  (SecondYearStudents * SamplingRatio = 10) ∧
  (ThirdYearStudents * SamplingRatio = 20) :=
by
  sorry

end stratified_sample_selection_l1748_174801


namespace abc_eq_l1748_174840

theorem abc_eq (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h : 4 * a * b - 1 ∣ (4 * a * a - 1) ^ 2) : a = b :=
sorry

end abc_eq_l1748_174840


namespace rain_probability_l1748_174866

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

end rain_probability_l1748_174866


namespace monotonicity_intervals_number_of_zeros_l1748_174816

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k / 2 * x^2

theorem monotonicity_intervals (k : ℝ) :
  (k ≤ 0 → (∀ x, x < 0 → f k x < 0) ∧ (∀ x, x ≥ 0 → f k x > 0)) ∧
  (0 < k ∧ k < 1 → 
    (∀ x, x < Real.log k → f k x < 0) ∧ (∀ x, x ≥ Real.log k ∧ x < 0 → f k x > 0) ∧ 
    (∀ x, x > 0 → f k x > 0)) ∧
  (k = 1 → ∀ x, f k x > 0) ∧
  (k > 1 → 
    (∀ x, x < 0 → f k x < 0) ∧ 
    (∀ x, x ≥ 0 ∧ x < Real.log k → f k x > 0) ∧ 
    (∀ x, x > Real.log k → f k x > 0)) :=
sorry

theorem number_of_zeros (k : ℝ) (h_nonpos : k ≤ 0) :
  (k < 0 → (∃ a b : ℝ, a < 0 ∧ b > 0 ∧ f k a = 0 ∧ f k b = 0)) ∧
  (k = 0 → f k 1 = 0 ∧ (∀ x, x ≠ 1 → f k x ≠ 0)) :=
sorry

end monotonicity_intervals_number_of_zeros_l1748_174816


namespace number_of_houses_with_neither_feature_l1748_174841

variable (T G P B : ℕ)

theorem number_of_houses_with_neither_feature 
  (hT : T = 90)
  (hG : G = 50)
  (hP : P = 40)
  (hB : B = 35) : 
  T - (G + P - B) = 35 := 
    by
      sorry

end number_of_houses_with_neither_feature_l1748_174841


namespace boss_contribution_l1748_174865

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

end boss_contribution_l1748_174865


namespace min_value_of_alpha_beta_l1748_174847

theorem min_value_of_alpha_beta 
  (k : ℝ)
  (h_k : k ≤ -4 ∨ k ≥ 5)
  (α β : ℝ)
  (h_αβ : α^2 - 2 * k * α + (k + 20) = 0 ∧ β^2 - 2 * k * β + (k + 20) = 0) :
  (α + 1) ^ 2 + (β + 1) ^ 2 = 18 → k = -4 :=
sorry

end min_value_of_alpha_beta_l1748_174847


namespace monotone_f_find_m_l1748_174838

noncomputable def f (x : ℝ) : ℝ := (2 * x - 2) / (x + 2)

theorem monotone_f : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by
  sorry

theorem find_m (m : ℝ) : 
  (∃ m, (f m - f 1 = 1/2)) ↔ m = 2 :=
by
  sorry

end monotone_f_find_m_l1748_174838


namespace sharon_distance_l1748_174802

noncomputable def usual_speed (x : ℝ) := x / 180
noncomputable def reduced_speed (x : ℝ) := (x / 180) - 0.5

theorem sharon_distance (x : ℝ) (h : 300 = (x / 2) / usual_speed x + (x / 2) / reduced_speed x) : x = 157.5 :=
by sorry

end sharon_distance_l1748_174802


namespace unique_prime_solution_l1748_174860

-- Define the variables and properties
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the proof goal
theorem unique_prime_solution (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (hp_pos : 0 < p) (hq_pos : 0 < q) :
  p^2 - q^3 = 1 → (p = 3 ∧ q = 2) :=
by sorry

end unique_prime_solution_l1748_174860


namespace simplify_sqrt_expression_l1748_174850

theorem simplify_sqrt_expression :
  2 * Real.sqrt 12 - Real.sqrt 27 - (Real.sqrt 3 * Real.sqrt (1 / 9)) = (2 * Real.sqrt 3) / 3 := 
by
  sorry

end simplify_sqrt_expression_l1748_174850


namespace rectangle_length_l1748_174882

theorem rectangle_length (side_square length_rectangle width_rectangle wire_length : ℝ) 
    (h1 : side_square = 12) 
    (h2 : width_rectangle = 6) 
    (h3 : wire_length = 4 * side_square) 
    (h4 : wire_length = 2 * width_rectangle + 2 * length_rectangle) : 
    length_rectangle = 18 := 
by 
  sorry

end rectangle_length_l1748_174882


namespace product_not_ending_in_1_l1748_174868

theorem product_not_ending_in_1 : ∃ a b : ℕ, 111111 = a * b ∧ (a % 10 ≠ 1) ∧ (b % 10 ≠ 1) := 
sorry

end product_not_ending_in_1_l1748_174868


namespace class_percentage_of_girls_l1748_174895

/-
Given:
- Initial number of boys in the class: 11
- Number of girls in the class: 13
- 1 boy is added to the class, resulting in the new total number of boys being 12

Prove:
- The percentage of the class that are girls is 52%.
-/
theorem class_percentage_of_girls (initial_boys : ℕ) (girls : ℕ) (added_boy : ℕ)
  (new_boy_total : ℕ) (total_students : ℕ) (percent_girls : ℕ) (h1 : initial_boys = 11) 
  (h2 : girls = 13) (h3 : added_boy = 1) (h4 : new_boy_total = initial_boys + added_boy) 
  (h5 : total_students = new_boy_total + girls) 
  (h6 : percent_girls = (girls * 100) / total_students) : percent_girls = 52 :=
sorry

end class_percentage_of_girls_l1748_174895


namespace least_common_multiple_prime_numbers_l1748_174846

theorem least_common_multiple_prime_numbers (x y : ℕ) (hx_prime : Prime x) (hy_prime : Prime y)
  (hxy : y < x) (h_eq : 2 * x + y = 12) : Nat.lcm x y = 10 :=
by
  sorry

end least_common_multiple_prime_numbers_l1748_174846


namespace sequence_a_n_eq_T_n_formula_C_n_formula_l1748_174808

noncomputable def sequence_S (n : ℕ) : ℕ := n * (2 * n - 1)

def arithmetic_seq (n : ℕ) : ℚ := 2 * n - 1

def a_n (n : ℕ) : ℤ := 4 * n - 3

def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℚ := (n : ℚ) / (4 * n + 1)

def c_n (n : ℕ) : ℚ := 3^(n - 1)

def C_n (n : ℕ) : ℚ := (3^n - 1) / 2

theorem sequence_a_n_eq (n : ℕ) : a_n n = 4 * n - 3 := by sorry

theorem T_n_formula (n : ℕ) : T_n n = (n : ℚ) / (4 * n + 1) := by sorry

theorem C_n_formula (n : ℕ) : C_n n = (3^n - 1) / 2 := by sorry

end sequence_a_n_eq_T_n_formula_C_n_formula_l1748_174808


namespace minimize_distance_l1748_174810

noncomputable def f (x : ℝ) := x^2 - 2 * x
noncomputable def P (x : ℝ) : ℝ × ℝ := (x, f x)
def Q : ℝ × ℝ := (4, -1)

theorem minimize_distance : ∃ (x : ℝ), dist (P x) Q = Real.sqrt 5 := by
  sorry

end minimize_distance_l1748_174810


namespace sequence_geometric_and_formula_l1748_174848

theorem sequence_geometric_and_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  (∀ n, a n + 1 = 2 ^ n) ∧ (a n = 2 ^ n - 1) :=
sorry

end sequence_geometric_and_formula_l1748_174848


namespace opposite_of_neg_six_l1748_174805

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end opposite_of_neg_six_l1748_174805


namespace quadrilateral_area_correct_l1748_174837

-- Definitions of given conditions
structure Quadrilateral :=
(W X Y Z : Type)
(WX XY YZ YW : ℝ)
(angle_WXY : ℝ)
(area : ℝ)

-- Quadrilateral satisfies given conditions
def quadrilateral_WXYZ : Quadrilateral :=
{ W := ℝ,
  X := ℝ,
  Y := ℝ,
  Z := ℝ,
  WX := 9,
  XY := 5,
  YZ := 12,
  YW := 15,
  angle_WXY := 90,
  area := 76.5 }

-- The theorem stating the area of quadrilateral WXYZ is 76.5
theorem quadrilateral_area_correct : quadrilateral_WXYZ.area = 76.5 :=
sorry

end quadrilateral_area_correct_l1748_174837


namespace inverse_function_value_l1748_174894

def g (x : ℝ) : ℝ := 4 * x ^ 3 - 5

theorem inverse_function_value (x : ℝ) : g x = -1 ↔ x = 1 :=
by
  sorry

end inverse_function_value_l1748_174894


namespace a_2016_value_l1748_174855

theorem a_2016_value (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6) 
  (rec : ∀ n, a (n + 2) = a (n + 1) - a n) : a 2016 = -3 :=
sorry

end a_2016_value_l1748_174855


namespace hours_in_one_year_l1748_174857

/-- Given that there are 24 hours in a day and 365 days in a year,
    prove that there are 8760 hours in one year. -/
theorem hours_in_one_year (hours_per_day : ℕ) (days_per_year : ℕ) (hours_value : ℕ := 8760) : hours_per_day = 24 → days_per_year = 365 → hours_per_day * days_per_year = hours_value :=
by
  intros
  sorry

end hours_in_one_year_l1748_174857


namespace negation_of_universal_statement_l1748_174823

open Real

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^3 > x^2) ↔ ∃ x : ℝ, x^3 ≤ x^2 :=
by
  sorry

end negation_of_universal_statement_l1748_174823


namespace correct_propositions_l1748_174890

-- Definitions for the propositions
def prop1 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ M) → a ∧ b
def prop2 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ ¬M) → a ∧ ¬b
def prop3 (a b M : Prop) : Prop := (a ∧ b) ∧ (b ∧ M) → a ∧ M
def prop4 (a M N : Prop) : Prop := (a ∧ ¬M) ∧ (a ∧ N) → ¬M ∧ N

-- Proof problem statement
theorem correct_propositions : 
  ∀ (a b M N : Prop), 
    (prop1 a M b = true) ∨ (prop1 a M b = false) ∧ 
    (prop2 a M b = true) ∨ (prop2 a M b = false) ∧ 
    (prop3 a b M = true) ∨ (prop3 a b M = false) ∧ 
    (prop4 a M N = true) ∨ (prop4 a M N = false) → 
    3 = 3 :=
by
  sorry

end correct_propositions_l1748_174890


namespace first_valve_fill_time_l1748_174811

theorem first_valve_fill_time (V1 V2: ℕ) (capacity: ℕ) (t_combined t1: ℕ) 
  (h1: t_combined = 48)
  (h2: V2 = V1 + 50)
  (h3: capacity = 12000)
  (h4: V1 + V2 = capacity / t_combined)
  : t1 = 2 * 60 :=
by
  -- The proof would come here
  sorry

end first_valve_fill_time_l1748_174811


namespace neg_div_neg_eq_pos_division_of_negatives_example_l1748_174830

theorem neg_div_neg_eq_pos (a b : Int) (hb : b ≠ 0) : (-a) / (-b) = a / b := by
  -- You can complete the proof here
  sorry

theorem division_of_negatives_example : (-81 : Int) / (-9) = 9 :=
  neg_div_neg_eq_pos 81 9 (by decide)

end neg_div_neg_eq_pos_division_of_negatives_example_l1748_174830


namespace find_other_endpoint_l1748_174858

theorem find_other_endpoint (x y : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, (x1 + x2)/2 = 2 ∧ (y1 + y2)/2 = 3 ∧ x1 = -1 ∧ y1 = 7 ∧ x2 = x ∧ y2 = y) → (x = 5 ∧ y = -1) :=
by
  sorry

end find_other_endpoint_l1748_174858


namespace least_subtracted_to_divisible_by_10_l1748_174869

theorem least_subtracted_to_divisible_by_10 (n : ℕ) (k : ℕ) (h : n = 724946) (div_cond : (n - k) % 10 = 0) : k = 6 :=
by
  sorry

end least_subtracted_to_divisible_by_10_l1748_174869


namespace square_area_inside_ellipse_l1748_174804

theorem square_area_inside_ellipse :
  (∃ s : ℝ, 
    ∀ (x y : ℝ), 
      (x = s ∧ y = s) → 
      (x^2 / 4 + y^2 / 8 = 1) ∧ 
      (4 * (s^2 / 3) = 1) ∧ 
      (area = 4 * (8 / 3))) →
    ∃ area : ℝ, 
      area = 32 / 3 :=
by
  sorry

end square_area_inside_ellipse_l1748_174804


namespace factor_expression_l1748_174826

theorem factor_expression (x : ℝ) : 75 * x^12 + 225 * x^24 = 75 * x^12 * (1 + 3 * x^12) :=
by sorry

end factor_expression_l1748_174826


namespace range_of_a_min_value_reciprocals_l1748_174829

noncomputable def f (x a : ℝ) : ℝ := |x - 2| + |x - a^2|

theorem range_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ a) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem min_value_reciprocals (m n a : ℝ) (h : m + 2 * n = a) (ha : a = 2) : (1/m + 1/n) ≥ (3/2 + Real.sqrt 2) := by
  sorry

end range_of_a_min_value_reciprocals_l1748_174829


namespace neither_sufficient_nor_necessary_l1748_174839

variable {a b : ℝ}

theorem neither_sufficient_nor_necessary (hab_ne_zero : a * b ≠ 0) :
  ¬ (a * b > 1 → a > (1 / b)) ∧ ¬ (a > (1 / b) → a * b > 1) :=
sorry

end neither_sufficient_nor_necessary_l1748_174839


namespace largest_angle_in_hexagon_l1748_174870

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

end largest_angle_in_hexagon_l1748_174870


namespace average_of_last_20_students_l1748_174861

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

end average_of_last_20_students_l1748_174861


namespace part1_part2_l1748_174867

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem part1 : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 := by
  sorry

theorem part2 (m : ℝ) : 1 ≤ m →
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f x ≤ f m) ∧ 
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f 1 ≤ f x) →
  f m - f 1 = 1 / 2 →
  m = 2 := by
  sorry

end part1_part2_l1748_174867


namespace integers_satisfying_condition_l1748_174873

-- Define the condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Define the theorem stating the proof problem
theorem integers_satisfying_condition :
  {x : ℤ | condition x} = {1, 2} :=
by
  sorry

end integers_satisfying_condition_l1748_174873


namespace minimum_time_to_finish_food_l1748_174828

-- Define the constants involved in the problem
def carrots_total : ℕ := 1000
def muffins_total : ℕ := 1000
def amy_carrots_rate : ℝ := 40 -- carrots per minute
def amy_muffins_rate : ℝ := 70 -- muffins per minute
def ben_carrots_rate : ℝ := 60 -- carrots per minute
def ben_muffins_rate : ℝ := 30 -- muffins per minute

-- Proof statement
theorem minimum_time_to_finish_food : 
  ∃ T : ℝ, 
  (∀ c : ℝ, c = 5 → 
  (∀ T_1 : ℝ, T_1 = (carrots_total / (amy_carrots_rate + ben_carrots_rate)) → 
  (∀ T_2 : ℝ, T_2 = ((muffins_total + (amy_muffins_rate * c)) / (amy_muffins_rate + ben_muffins_rate)) +
  (muffins_total / ben_muffins_rate) - T_1 - c →
  T = T_1 + T_2) ∧
  T = 23.5 )) :=
sorry

end minimum_time_to_finish_food_l1748_174828


namespace least_multiple_17_gt_500_l1748_174893

theorem least_multiple_17_gt_500 (n : ℕ) (h : (n = 17)) : ∃ m : ℤ, (m * n > 500 ∧ m * n = 510) :=
  sorry

end least_multiple_17_gt_500_l1748_174893


namespace river_depth_mid_June_l1748_174832

theorem river_depth_mid_June (D : ℝ) : 
    (∀ (mid_May mid_June mid_July : ℝ),
    mid_May = 5 →
    mid_June = mid_May + D →
    mid_July = 3 * mid_June →
    mid_July = 45) →
    D = 10 :=
by
    sorry

end river_depth_mid_June_l1748_174832


namespace max_sum_is_2017_l1748_174833

theorem max_sum_is_2017 (a b c : ℕ) 
  (h1 : a + b = 1014) 
  (h2 : c - b = 497) 
  (h3 : a > b) : 
  (a + b + c) ≤ 2017 := sorry

end max_sum_is_2017_l1748_174833


namespace find_constants_l1748_174815

theorem find_constants (a b c d : ℚ) :
  (6 * x^3 - 4 * x + 2) * (a * x^3 + b * x^2 + c * x + d) =
  18 * x^6 - 2 * x^5 + 16 * x^4 - (28 / 3) * x^3 + (8 / 3) * x^2 - 4 * x + 2 →
  a = 3 ∧ b = -1 / 3 ∧ c = 14 / 9 :=
by
  sorry

end find_constants_l1748_174815


namespace total_number_of_notes_l1748_174819

theorem total_number_of_notes (x : ℕ) (h₁ : 37 * 50 + x * 500 = 10350) : 37 + x = 54 :=
by
  -- We state that the total value of 37 Rs. 50 notes plus x Rs. 500 notes equals Rs. 10350.
  -- According to this information, we prove that the total number of notes is 54.
  sorry

end total_number_of_notes_l1748_174819


namespace area_of_smallest_square_containing_circle_l1748_174883

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  ∃ (a : ℝ), a = 100 :=
by
  sorry

end area_of_smallest_square_containing_circle_l1748_174883


namespace math_problem_l1748_174876

theorem math_problem (x y : ℕ) (h1 : (x + y * I)^3 = 2 + 11 * I) (h2 : 0 < x) (h3 : 0 < y) : 
  x + y * I = 2 + I :=
sorry

end math_problem_l1748_174876


namespace max_volume_tank_l1748_174821

theorem max_volume_tank (a b h : ℝ) (ha : a ≤ 1.5) (hb : b ≤ 1.5) (hh : h = 1.5) :
  a * b * h ≤ 3.375 :=
by {
  sorry
}

end max_volume_tank_l1748_174821


namespace option_A_option_B_option_C_option_D_l1748_174859

theorem option_A : (-4:ℤ)^2 ≠ -(4:ℤ)^2 := sorry
theorem option_B : (-2:ℤ)^3 = -2^3 := sorry
theorem option_C : (-1:ℤ)^2020 ≠ (-1:ℤ)^2021 := sorry
theorem option_D : ((2:ℚ)/(3:ℚ))^3 = ((2:ℚ)/(3:ℚ))^3 := sorry

end option_A_option_B_option_C_option_D_l1748_174859


namespace find_g_inv_f_8_l1748_174885

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g : ∀ x : ℝ, f_inv (g x) = x^2 - x
axiom g_bijective : Function.Bijective g

theorem find_g_inv_f_8 : g_inv (f 8) = (1 + Real.sqrt 33) / 2 :=
by
  -- proof is omitted
  sorry

end find_g_inv_f_8_l1748_174885


namespace colleague_typing_time_l1748_174834

theorem colleague_typing_time (T : ℝ) : 
  (∀ me_time : ℝ, (me_time = 180) →
  (∀ my_speed my_colleague_speed : ℝ, (my_speed = T / me_time) →
  (my_colleague_speed = 4 * my_speed) →
  (T / my_colleague_speed = 45))) :=
  sorry

end colleague_typing_time_l1748_174834


namespace inequality_solution_l1748_174818

noncomputable def solution_set : Set ℝ :=
  {x | -4 < x ∧ x < (17 - Real.sqrt 201) / 4} ∪ {x | (17 + Real.sqrt 201) / 4 < x ∧ x < 2 / 3}

theorem inequality_solution (x : ℝ) (h1 : x ≠ -4) (h2 : x ≠ 2 / 3) :
  (2 * x - 3) / (x + 4) > (4 * x + 1) / (3 * x - 2) ↔ x ∈ solution_set := by
  sorry

end inequality_solution_l1748_174818


namespace xy_equation_result_l1748_174878

theorem xy_equation_result (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -5) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -10.528 :=
by
  sorry

end xy_equation_result_l1748_174878


namespace quadratic_rewrite_l1748_174880

theorem quadratic_rewrite (a b c x : ℤ) :
  (16 * x^2 - 40 * x - 72 = a^2 * x^2 + 2 * a * b * x + b^2 + c) →
  (a = 4 ∨ a = -4) →
  (2 * a * b = -40) →
  ab = -20 := by
sorry

end quadratic_rewrite_l1748_174880


namespace smallest_AAB_l1748_174800

theorem smallest_AAB : ∃ (A B : ℕ), (1 <= A ∧ A <= 9) ∧ (1 <= B ∧ B <= 9) ∧ (AB = 10 * A + B) ∧ (AAB = 100 * A + 10 * A + B) ∧ (110 * A + B = 8 * (10 * A + B)) ∧ (AAB = 221) :=
by
  sorry

end smallest_AAB_l1748_174800


namespace square_land_perimeter_l1748_174879

theorem square_land_perimeter (a p : ℝ) (h1 : a = p^2 / 16) (h2 : 5*a = 10*p + 45) : p = 36 :=
by sorry

end square_land_perimeter_l1748_174879


namespace max_min_ab_bc_ca_l1748_174807

theorem max_min_ab_bc_ca (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 12) (h_ab_bc_ca : a * b + b * c + c * a = 30) :
  max (min (a * b) (min (b * c) (c * a))) = 9 :=
sorry

end max_min_ab_bc_ca_l1748_174807


namespace not_square_l1748_174824

open Int

theorem not_square (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ k : ℤ, (a^2 : ℤ) + ⌈(4 * a^2 : ℤ) / b⌉ = k^2 :=
by
  sorry

end not_square_l1748_174824


namespace four_digit_greater_than_three_digit_l1748_174889

theorem four_digit_greater_than_three_digit (n m : ℕ) (h₁ : 1000 ≤ n ∧ n ≤ 9999) (h₂ : 100 ≤ m ∧ m ≤ 999) : n > m :=
sorry

end four_digit_greater_than_three_digit_l1748_174889


namespace april_earnings_l1748_174851

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

end april_earnings_l1748_174851


namespace number_of_elements_in_M_l1748_174835

theorem number_of_elements_in_M :
  (∃! (M : Finset ℕ), M = {m | ∃ (n : ℕ), n > 0 ∧ m = 2*n - 1 ∧ m < 60 } ∧ M.card = 30) :=
sorry

end number_of_elements_in_M_l1748_174835


namespace base_b_equivalence_l1748_174881

theorem base_b_equivalence (b : ℕ) (h : (2 * b + 4) ^ 2 = 5 * b ^ 2 + 5 * b + 4) : b = 12 :=
sorry

end base_b_equivalence_l1748_174881


namespace petya_pencils_l1748_174822

theorem petya_pencils (x : ℕ) (promotion : x + 12 = 61) :
  x = 49 :=
by
  sorry

end petya_pencils_l1748_174822


namespace crystal_discount_is_50_percent_l1748_174825

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

end crystal_discount_is_50_percent_l1748_174825


namespace change_given_l1748_174849

theorem change_given (pants_cost : ℕ) (shirt_cost : ℕ) (tie_cost : ℕ) (total_paid : ℕ) (total_cost : ℕ) (change : ℕ) :
  pants_cost = 140 ∧ shirt_cost = 43 ∧ tie_cost = 15 ∧ total_paid = 200 ∧ total_cost = (pants_cost + shirt_cost + tie_cost) ∧ change = (total_paid - total_cost) → change = 2 :=
by
  sorry

end change_given_l1748_174849


namespace number_of_ordered_triples_l1748_174845

theorem number_of_ordered_triples (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 3969) (h4 : a * c = 3969^2) :
    ∃ n : ℕ, n = 12 := sorry

end number_of_ordered_triples_l1748_174845


namespace trader_profit_percentage_l1748_174871

theorem trader_profit_percentage (P : ℝ) (h₀ : 0 ≤ P) : 
  let discount := 0.40
  let increase := 0.80
  let purchase_price := P * (1 - discount)
  let selling_price := purchase_price * (1 + increase)
  let profit := selling_price - P
  (profit / P) * 100 = 8 := 
by
  sorry

end trader_profit_percentage_l1748_174871


namespace number_of_students_in_class_l1748_174872

theorem number_of_students_in_class :
  ∃ a : ℤ, 100 ≤ a ∧ a ≤ 200 ∧ a % 4 = 1 ∧ a % 3 = 2 ∧ a % 7 = 3 ∧ a = 101 := 
sorry

end number_of_students_in_class_l1748_174872


namespace perpendicular_line_through_point_l1748_174875

theorem perpendicular_line_through_point (a b c : ℝ) (hx : a = 2) (hy : b = -1) (hd : c = 3) :
  ∃ k d : ℝ, (k, d) = (-a / b, (a * 1 + b * (1 - c))) ∧ (b * -1, a * -1 + d, -a) = (1, 2, 3) :=
by
  sorry

end perpendicular_line_through_point_l1748_174875


namespace reciprocal_neg_sqrt_2_l1748_174897

theorem reciprocal_neg_sqrt_2 : 1 / (-Real.sqrt 2) = -Real.sqrt 2 / 2 :=
by
  sorry

end reciprocal_neg_sqrt_2_l1748_174897


namespace cubic_foot_to_cubic_inches_l1748_174827

theorem cubic_foot_to_cubic_inches (foot_to_inch : 1 = 12) : 12 ^ 3 = 1728 :=
by
  have h1 : 1^3 = 1 := by norm_num
  have h2 : (12^3) = 1728 := by norm_num
  rw [foot_to_inch] at h1
  exact h2

end cubic_foot_to_cubic_inches_l1748_174827


namespace number_of_children_l1748_174891

def pencils_per_child : ℕ := 2
def total_pencils : ℕ := 16

theorem number_of_children : total_pencils / pencils_per_child = 8 :=
by
  sorry

end number_of_children_l1748_174891


namespace find_complex_number_l1748_174852

def i := Complex.I
def z := -Complex.I - 1
def complex_equation (z : ℂ) := i * z = 1 - i

theorem find_complex_number : complex_equation z :=
by
  -- skip the proof here
  sorry

end find_complex_number_l1748_174852


namespace complex_arithmetic_problem_l1748_174899
open Complex

theorem complex_arithmetic_problem : (2 - 3 * Complex.I) * (2 + 3 * Complex.I) + (4 - 5 * Complex.I)^2 = 4 - 40 * Complex.I := by
  sorry

end complex_arithmetic_problem_l1748_174899


namespace simplest_square_root_l1748_174820

theorem simplest_square_root (a b c d : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 3) (h3 : c = (Real.sqrt 2) / 2) (h4 : d = Real.sqrt 10) :
  d = Real.sqrt 10 ∧ (a ≠ Real.sqrt 10) ∧ (b ≠ Real.sqrt 10) ∧ (c ≠ Real.sqrt 10) := 
by 
  sorry

end simplest_square_root_l1748_174820
