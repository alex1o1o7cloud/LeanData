import Mathlib

namespace possible_values_of_f2001_l403_40317

noncomputable def f : ℕ → ℝ := sorry

theorem possible_values_of_f2001 (f : ℕ → ℝ)
    (H : ∀ a b : ℕ, a > 1 → b > 1 → ∀ d : ℕ, d = Nat.gcd a b → 
           f (a * b) = f d * (f (a / d) + f (b / d))) :
    f 2001 = 0 ∨ f 2001 = 1/2 :=
sorry

end possible_values_of_f2001_l403_40317


namespace min_value_of_2x_plus_4y_l403_40302

noncomputable def minimum_value (x y : ℝ) : ℝ := 2^x + 4^y

theorem min_value_of_2x_plus_4y (x y : ℝ) (h : x + 2 * y = 3) : minimum_value x y = 4 * Real.sqrt 2 :=
by
  sorry

end min_value_of_2x_plus_4y_l403_40302


namespace abs_diff_one_l403_40304

theorem abs_diff_one (a b : ℤ) (h : |a| + |b| = 1) : |a - b| = 1 := sorry

end abs_diff_one_l403_40304


namespace translate_point_correct_l403_40396

-- Define initial point
def initial_point : ℝ × ℝ := (0, 1)

-- Define translation downward
def translate_down (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 - units)

-- Define translation to the left
def translate_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Define the expected resulting point
def expected_point : ℝ × ℝ := (-4, -1)

-- Lean statement to prove the equivalence
theorem translate_point_correct :
  (translate_left (translate_down initial_point 2) 4) = expected_point :=
by 
  -- Here, we would prove it step by step if required
  sorry

end translate_point_correct_l403_40396


namespace problem_solution_l403_40362

open Set

variable {U : Set ℕ} (M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hC : U \ M = {1, 3})

theorem problem_solution : 2 ∈ M :=
by
  sorry

end problem_solution_l403_40362


namespace sin_2pi_minus_theta_l403_40380

theorem sin_2pi_minus_theta (theta : ℝ) (k : ℤ) 
  (h1 : 3 * Real.cos theta ^ 2 = Real.tan theta + 3)
  (h2 : theta ≠ k * Real.pi) :
  Real.sin (2 * (Real.pi - theta)) = 2 / 3 := by
  sorry

end sin_2pi_minus_theta_l403_40380


namespace number_times_quarter_squared_eq_four_cubed_l403_40355

theorem number_times_quarter_squared_eq_four_cubed : 
  ∃ (number : ℕ), number * (1 / 4 : ℚ) ^ 2 = (4 : ℚ) ^ 3 ∧ number = 1024 :=
by 
  use 1024
  sorry

end number_times_quarter_squared_eq_four_cubed_l403_40355


namespace students_in_5th_6th_grades_l403_40366

-- Definitions for problem conditions
def is_three_digit_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def six_two_digit_sum_eq_twice (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧
               a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
               (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) = 2 * n

-- The proof problem statement in Lean 4
theorem students_in_5th_6th_grades :
  ∃ n : ℕ, is_three_digit_number n ∧ six_two_digit_sum_eq_twice n ∧ n = 198 :=
by
  sorry

end students_in_5th_6th_grades_l403_40366


namespace fathers_age_multiple_l403_40308

theorem fathers_age_multiple 
  (Johns_age : ℕ)
  (sum_of_ages : ℕ)
  (additional_years : ℕ)
  (m : ℕ)
  (h1 : Johns_age = 15)
  (h2 : sum_of_ages = 77)
  (h3 : additional_years = 32)
  (h4 : sum_of_ages = Johns_age + (Johns_age * m + additional_years)) :
  m = 2 := 
by 
  sorry

end fathers_age_multiple_l403_40308


namespace value_of_f_2019_l403_40398

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ)

-- Assumptions
axiom f_zero : f 0 = 2
axiom f_period : ∀ x : ℝ, f (x + 3) = -f x

-- The property to be proved
theorem value_of_f_2019 : f 2019 = -2 := sorry

end value_of_f_2019_l403_40398


namespace infinitely_many_composite_z_l403_40352

theorem infinitely_many_composite_z (m n : ℕ) (h_m : m > 1) : ¬ (Nat.Prime (n^4 + 4*m^4)) :=
by
  sorry

end infinitely_many_composite_z_l403_40352


namespace n_calculation_l403_40361

theorem n_calculation (n : ℕ) (hn : 0 < n)
  (h1 : Int.lcm 24 n = 72)
  (h2 : Int.lcm n 27 = 108) :
  n = 36 :=
sorry

end n_calculation_l403_40361


namespace min_value_of_m_l403_40378

theorem min_value_of_m (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + b * c + c * a = -1) (h3 : a * b * c = -m) : 
    m = - (min (-a ^ 3 + a ^ 2 + a ) (- (1 / 27))) := 
sorry

end min_value_of_m_l403_40378


namespace find_x_l403_40347

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 210) : x = 14 := by
  sorry

end find_x_l403_40347


namespace tom_new_collection_l403_40364

theorem tom_new_collection (initial_stamps mike_gift : ℕ) (harry_gift : ℕ := 2 * mike_gift + 10) (sarah_gift : ℕ := 3 * mike_gift - 5) (total_gifts : ℕ := mike_gift + harry_gift + sarah_gift) (new_collection : ℕ := initial_stamps + total_gifts) 
  (h_initial_stamps : initial_stamps = 3000) (h_mike_gift : mike_gift = 17) :
  new_collection = 3107 := by
  sorry

end tom_new_collection_l403_40364


namespace largest_number_of_gold_coins_l403_40326

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l403_40326


namespace speed_in_kmh_l403_40321

def distance : ℝ := 550.044
def time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem speed_in_kmh : (distance / time) * conversion_factor = 66.00528 := 
by
  sorry

end speed_in_kmh_l403_40321


namespace students_taking_chem_or_phys_not_both_l403_40329

def students_taking_both : ℕ := 12
def students_taking_chemistry : ℕ := 30
def students_taking_only_physics : ℕ := 18

theorem students_taking_chem_or_phys_not_both : 
  (students_taking_chemistry - students_taking_both) + students_taking_only_physics = 36 := 
by
  sorry

end students_taking_chem_or_phys_not_both_l403_40329


namespace range_of_a_l403_40397

noncomputable def f (x : ℝ) := Real.log (x + 1)
def A (x : ℝ) := (f (1 - 2 * x) > f x)
def B (a x : ℝ) := (a - 1 < x) ∧ (x < 2 * a^2)

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, A x ∧ B a x) ↔ (a < -1 / 2) ∨ (1 < a ∧ a < 4 / 3) :=
sorry

end range_of_a_l403_40397


namespace find_m_l403_40384

theorem find_m (m : ℝ) 
  (A : ℝ × ℝ := (-2, m))
  (B : ℝ × ℝ := (m, 4))
  (h_slope : ((B.snd - A.snd) / (B.fst - A.fst)) = -2) : 
  m = -8 :=
by 
  sorry

end find_m_l403_40384


namespace bedrooms_count_l403_40389

/-- Number of bedrooms calculation based on given conditions -/
theorem bedrooms_count (B : ℕ) (h1 : ∀ b, b = 20 * B)
  (h2 : ∀ lr, lr = 20 * B)
  (h3 : ∀ bath, bath = 2 * 20 * B)
  (h4 : ∀ out, out = 2 * (20 * B + 20 * B + 40 * B))
  (h5 : ∀ siblings, siblings = 3)
  (h6 : ∀ work_time, work_time = 4 * 60) : B = 3 :=
by
  -- proof will be provided here
  sorry

end bedrooms_count_l403_40389


namespace sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l403_40373

-- Proof 1: 
theorem sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3 :
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 :=
by
  sorry

-- Proof 2:
theorem sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12 :
  1 / Real.sqrt 24 + abs (Real.sqrt 6 - 3) + (1 / 2)⁻¹ - 2016 ^ 0 = 4 - 11 * Real.sqrt 6 / 12 :=
by
  sorry

-- Proof 3:
theorem sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6 :
  (Real.sqrt 3 + Real.sqrt 2) ^ 2 - (Real.sqrt 3 - Real.sqrt 2) ^ 2 = 4 * Real.sqrt 6 :=
by
  sorry

end sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l403_40373


namespace prob_each_student_gets_each_snack_l403_40348

-- Define the total number of snacks and their types
def total_snacks := 16
def snack_types := 4

-- Define the conditions for the problem
def students := 4
def snacks_per_type := 4

-- Define the probability calculation.
-- We would typically use combinatorial functions here, but for simplicity, use predefined values from the solution.
def prob_student_1 := 64 / 455
def prob_student_2 := 9 / 55
def prob_student_3 := 8 / 35
def prob_student_4 := 1 -- Always 1 for the final student's remaining snacks

-- Calculate the total probability
def total_prob := prob_student_1 * prob_student_2 * prob_student_3 * prob_student_4

-- The statement to prove the desired probability outcome
theorem prob_each_student_gets_each_snack : total_prob = (64 / 1225) :=
by
  sorry

end prob_each_student_gets_each_snack_l403_40348


namespace part1_solution_set_l403_40336

theorem part1_solution_set (a : ℝ) (x : ℝ) : a = -2 → (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0 ↔ x ≠ -1 :=
by sorry

end part1_solution_set_l403_40336


namespace seventy_second_number_in_S_is_573_l403_40353

open Nat

def S : Set Nat := { k | k % 8 = 5 }

theorem seventy_second_number_in_S_is_573 : ∃ k ∈ (Finset.range 650), k = 8 * 71 + 5 :=
by
  sorry -- Proof goes here

end seventy_second_number_in_S_is_573_l403_40353


namespace will_earnings_l403_40330

-- Defining the conditions
def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

-- Calculating the earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def total_earnings := monday_earnings + tuesday_earnings

-- Stating the problem
theorem will_earnings : total_earnings = 80 := by
  -- sorry is used to skip the actual proof
  sorry

end will_earnings_l403_40330


namespace find_principal_amount_l403_40399

theorem find_principal_amount
  (P R T SI : ℝ) 
  (rate_condition : R = 12)
  (time_condition : T = 20)
  (interest_condition : SI = 2100) :
  SI = (P * R * T) / 100 → P = 875 :=
by
  sorry

end find_principal_amount_l403_40399


namespace geom_seq_sum_l403_40322

theorem geom_seq_sum (q : ℝ) (a₃ a₄ a₅ : ℝ) : 
  0 < q ∧ 3 * (1 - q^3) / (1 - q) = 21 ∧ a₃ = 3 * q^2 ∧ a₄ = 3 * q^3 ∧ a₅ = 3 * q^4 
  -> a₃ + a₄ + a₅ = 84 := 
by 
  sorry

end geom_seq_sum_l403_40322


namespace find_M_plus_N_l403_40313

theorem find_M_plus_N (M N : ℕ) (h1 : (3:ℚ) / 5 = M / 45) (h2 : (3:ℚ) / 5 = 60 / N) : M + N = 127 :=
sorry

end find_M_plus_N_l403_40313


namespace final_net_worth_l403_40305

noncomputable def initial_cash_A := (20000 : ℤ)
noncomputable def initial_cash_B := (22000 : ℤ)
noncomputable def house_value := (20000 : ℤ)
noncomputable def vehicle_value := (10000 : ℤ)

noncomputable def transaction_1_cash_A := initial_cash_A + 25000
noncomputable def transaction_1_cash_B := initial_cash_B - 25000

noncomputable def transaction_2_cash_A := transaction_1_cash_A - 12000
noncomputable def transaction_2_cash_B := transaction_1_cash_B + 12000

noncomputable def transaction_3_cash_A := transaction_2_cash_A + 18000
noncomputable def transaction_3_cash_B := transaction_2_cash_B - 18000

noncomputable def transaction_4_cash_A := transaction_3_cash_A + 9000
noncomputable def transaction_4_cash_B := transaction_3_cash_B + 9000

noncomputable def final_value_A := transaction_4_cash_A
noncomputable def final_value_B := transaction_4_cash_B + house_value + vehicle_value

theorem final_net_worth :
  final_value_A - initial_cash_A = 40000 ∧ final_value_B - initial_cash_B = 8000 :=
by
  sorry

end final_net_worth_l403_40305


namespace find_k_l403_40350

noncomputable def g (x : ℕ) : ℤ := 2 * x^2 - 8 * x + 8

theorem find_k :
  (g 2 = 0) ∧ 
  (90 < g 9) ∧ (g 9 < 100) ∧
  (120 < g 10) ∧ (g 10 < 130) ∧
  ∃ (k : ℤ), 7000 * k < g 150 ∧ g 150 < 7000 * (k + 1)
  → ∃ (k : ℤ), k = 6 :=
by
  sorry

end find_k_l403_40350


namespace count_true_statements_l403_40325

open Set

variable {M P : Set α}

theorem count_true_statements (h : ¬ ∀ x ∈ M, x ∈ P) (hne : Nonempty M) :
  (¬ ∃ x, x ∈ M ∧ x ∈ P ∨ ∀ x, x ∈ M → x ∈ P) ∧ (∃ x, x ∈ M ∧ x ∉ P) ∧ 
  ¬ (∃ x, x ∈ M ∧ x ∈ P) ∧ (¬ ∀ x, x ∈ M → x ∈ P) :=
sorry

end count_true_statements_l403_40325


namespace max_cookie_price_l403_40306

theorem max_cookie_price (k p : ℕ) :
  8 * k + 3 * p < 200 →
  4 * k + 5 * p > 150 →
  k ≤ 19 :=
sorry

end max_cookie_price_l403_40306


namespace product_of_solutions_l403_40371

theorem product_of_solutions (α β : ℝ) (h : 2 * α^2 + 8 * α - 45 = 0 ∧ 2 * β^2 + 8 * β - 45 = 0 ∧ α ≠ β) :
  α * β = -22.5 :=
sorry

end product_of_solutions_l403_40371


namespace speed_first_half_proof_l403_40394

noncomputable def speed_first_half
  (total_time: ℕ) 
  (distance: ℕ) 
  (second_half_speed: ℕ) 
  (first_half_time: ℕ) :
  ℕ :=
  distance / first_half_time

theorem speed_first_half_proof
  (total_time: ℕ)
  (distance: ℕ)
  (second_half_speed: ℕ)
  (half_distance: ℕ)
  (second_half_time: ℕ)
  (first_half_time: ℕ) :
  total_time = 12 →
  distance = 560 →
  second_half_speed = 40 →
  half_distance = distance / 2 →
  second_half_time = half_distance / second_half_speed →
  first_half_time = total_time - second_half_time →
  speed_first_half total_time half_distance second_half_speed first_half_time = 56 :=
by
  sorry

end speed_first_half_proof_l403_40394


namespace five_x_minus_two_l403_40391

theorem five_x_minus_two (x : ℚ) (h : 4 * x - 8 = 13 * x + 3) : 5 * (x - 2) = -145 / 9 := by
  sorry

end five_x_minus_two_l403_40391


namespace part1_max_min_part2_triangle_inequality_l403_40328

noncomputable def f (x k : ℝ) : ℝ :=
  (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem part1_max_min (k : ℝ): 
  (∀ x : ℝ, k ≥ 1 → 1 ≤ f x k ∧ f x k ≤ (1/3) * (k + 2)) ∧ 
  (∀ x : ℝ, k < 1 → (1/3) * (k + 2) ≤ f x k ∧ f x k ≤ 1) := 
sorry

theorem part2_triangle_inequality (k : ℝ) : 
  -1/2 < k ∧ k < 4 ↔ (∀ a b c : ℝ, (f a k + f b k > f c k) ∧ (f b k + f c k > f a k) ∧ (f c k + f a k > f b k)) :=
sorry

end part1_max_min_part2_triangle_inequality_l403_40328


namespace equal_split_l403_40372

theorem equal_split (A B C : ℝ) (h1 : A < B) (h2 : B < C) : 
  (B + C - 2 * A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end equal_split_l403_40372


namespace eldorado_license_plates_count_l403_40360

def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def valid_license_plates_count : Nat :=
  let num_vowels := 5
  let num_letters := 26
  let num_digits := 10
  num_vowels * num_letters * num_letters * num_digits * num_digits

theorem eldorado_license_plates_count : valid_license_plates_count = 338000 := by
  sorry

end eldorado_license_plates_count_l403_40360


namespace sum_of_slopes_eq_zero_l403_40303

theorem sum_of_slopes_eq_zero
  (p : ℝ) (a : ℝ) (hp : p > 0) (ha : a > 0)
  (P Q : ℝ × ℝ)
  (hP : P.2 ^ 2 = 2 * p * P.1)
  (hQ : Q.2 ^ 2 = 2 * p * Q.1)
  (hcollinear : ∃ m : ℝ, ∀ (x y : (ℝ × ℝ)), y = P ∨ y = Q ∨ y = (-a, 0) → y.2 = m * (y.1 + a)) :
  let k_AP := (P.2) / (P.1 - a)
  let k_AQ := (Q.2) / (Q.1 - a)
  k_AP + k_AQ = 0 := by
    sorry

end sum_of_slopes_eq_zero_l403_40303


namespace find_real_x_l403_40356

theorem find_real_x (x : ℝ) : 
  (2 ≤ 3 * x / (3 * x - 7)) ∧ (3 * x / (3 * x - 7) < 6) ↔ (7 / 3 < x ∧ x < 42 / 15) :=
by
  sorry

end find_real_x_l403_40356


namespace number_of_rows_l403_40343

-- Definitions of the conditions
def total_students : ℕ := 23
def students_in_restroom : ℕ := 2
def students_absent : ℕ := 3 * students_in_restroom - 1
def students_per_desk : ℕ := 6
def fraction_full (r : ℕ) := (2 * r) / 3

-- The statement we need to prove 
theorem number_of_rows : (total_students - students_in_restroom - students_absent) / (students_per_desk * 2 / 3) = 4 :=
by
  sorry

end number_of_rows_l403_40343


namespace total_students_in_class_l403_40333

theorem total_students_in_class :
  ∃ x, (10 * 90 + 15 * 80 + x * 60) / (10 + 15 + x) = 72 → 10 + 15 + x = 50 :=
by
  -- Providing an existence proof and required conditions
  use 25
  intro h
  sorry

end total_students_in_class_l403_40333


namespace find_probability_of_B_l403_40307

-- Define the conditions and the problem
def system_A_malfunction_prob := 1 / 10
def at_least_one_not_malfunction_prob := 49 / 50

/-- The probability that System B malfunctions given that 
  the probability of at least one system not malfunctioning 
  is 49/50 and the probability of System A malfunctioning is 1/10 -/
theorem find_probability_of_B (p : ℝ) 
  (h1 : system_A_malfunction_prob = 1 / 10) 
  (h2 : at_least_one_not_malfunction_prob = 49 / 50) 
  (h3 : 1 - (system_A_malfunction_prob * p) = at_least_one_not_malfunction_prob) : 
  p = 1 / 5 :=
sorry

end find_probability_of_B_l403_40307


namespace mow_lawn_payment_l403_40309

theorem mow_lawn_payment (bike_cost weekly_allowance babysitting_rate babysitting_hours money_saved target_savings mowing_payment : ℕ) 
  (h1 : bike_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : babysitting_rate = 7)
  (h4 : babysitting_hours = 2)
  (h5 : money_saved = 65)
  (h6 : target_savings = 6) :
  mowing_payment = 10 :=
sorry

end mow_lawn_payment_l403_40309


namespace quotient_division_l403_40368

noncomputable def poly_division_quotient : Polynomial ℚ :=
  Polynomial.div (9 * Polynomial.X ^ 4 + 8 * Polynomial.X ^ 3 - 12 * Polynomial.X ^ 2 - 7 * Polynomial.X + 4) (3 * Polynomial.X ^ 2 + 2 * Polynomial.X + 5)

theorem quotient_division :
  poly_division_quotient = (3 * Polynomial.X ^ 2 - 2 * Polynomial.X + 2) :=
sorry

end quotient_division_l403_40368


namespace total_fruits_purchased_l403_40314

-- Defining the costs of apples and bananas
def cost_per_apple : ℝ := 0.80
def cost_per_banana : ℝ := 0.70

-- Defining the total cost the customer spent
def total_cost : ℝ := 6.50

-- Defining the total number of fruits purchased as 9
theorem total_fruits_purchased (A B : ℕ) : 
  (cost_per_apple * A + cost_per_banana * B = total_cost) → 
  (A + B = 9) :=
by
  sorry

end total_fruits_purchased_l403_40314


namespace fifth_eq_l403_40312

theorem fifth_eq :
  (1 = 1) ∧
  (2 + 3 + 4 = 9) ∧
  (3 + 4 + 5 + 6 + 7 = 25) ∧
  (4 + 5 + 6 + 7 + 8 + 9 + 10 = 49) →
  5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81 :=
by
  intros
  sorry

end fifth_eq_l403_40312


namespace width_of_door_l403_40300

theorem width_of_door 
  (L W H : ℕ) 
  (cost_per_sq_ft : ℕ) 
  (door_height window_height window_width : ℕ) 
  (num_windows total_cost : ℕ) 
  (door_width : ℕ) 
  (total_wall_area area_door area_windows area_to_whitewash : ℕ)
  (raw_area_door raw_area_windows total_walls_to_paint : ℕ) 
  (cost_per_sq_ft_eq : cost_per_sq_ft = 9)
  (total_cost_eq : total_cost = 8154)
  (room_dimensions_eq : L = 25 ∧ W = 15 ∧ H = 12)
  (door_dimensions_eq : door_height = 6)
  (window_dimensions_eq : window_height = 3 ∧ window_width = 4)
  (num_windows_eq : num_windows = 3)
  (total_wall_area_eq : total_wall_area = 2 * (L * H) + 2 * (W * H))
  (raw_area_door_eq : raw_area_door = door_height * door_width)
  (raw_area_windows_eq : raw_area_windows = num_windows * (window_width * window_height))
  (total_walls_to_paint_eq : total_walls_to_paint = total_wall_area - raw_area_door - raw_area_windows)
  (area_to_whitewash_eq : area_to_whitewash = 924 - 6 * door_width)
  (total_cost_eq_calc : total_cost = area_to_whitewash * cost_per_sq_ft) :
  door_width = 3 := sorry

end width_of_door_l403_40300


namespace max_xy_l403_40315

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 9 * y = 12) : xy ≤ 4 :=
by
sorry

end max_xy_l403_40315


namespace conic_section_is_parabola_l403_40382

-- Define the equation |y-3| = sqrt((x+4)^2 + y^2)
def equation (x y : ℝ) : Prop := |y - 3| = Real.sqrt ((x + 4) ^ 2 + y ^ 2)

-- The main theorem stating the conic section type is a parabola
theorem conic_section_is_parabola : ∀ x y : ℝ, equation x y → false := sorry

end conic_section_is_parabola_l403_40382


namespace spherical_coord_plane_l403_40311

-- Let's define spherical coordinates and the condition theta = c.
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def is_plane (c : ℝ) (p : SphericalCoordinates) : Prop :=
  p.θ = c

theorem spherical_coord_plane (c : ℝ) : 
  ∀ p : SphericalCoordinates, is_plane c p → True := 
by
  intros p hp
  sorry

end spherical_coord_plane_l403_40311


namespace pirates_total_coins_l403_40363

theorem pirates_total_coins :
  ∀ (x : ℕ), (x * (x + 1)) / 2 = 5 * x → 6 * x = 54 :=
by
  intro x
  intro h
  -- proof omitted
  sorry

end pirates_total_coins_l403_40363


namespace problem_part_a_problem_part_b_l403_40385

def is_two_squared (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2 ∧ a ≠ 0 ∧ b ≠ 0

def is_three_squared (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = a^2 + b^2 + c^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def is_four_squared (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a^2 + b^2 + c^2 + d^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def satisfies_prime_conditions (e : ℕ) : Prop :=
  Nat.Prime (e - 2) ∧ Nat.Prime e ∧ Nat.Prime (e + 4)

def satisfies_square_sum_conditions (a b c d e : ℕ) : Prop :=
  a^2 + b^2 + c^2 + d^2 + e^2 = 2020 ∧ a < b ∧ b < c ∧ c < d ∧ d < e

theorem problem_part_a : is_two_squared 2020 ∧ is_three_squared 2020 ∧ is_four_squared 2020 := sorry

theorem problem_part_b : ∃ a b c d e : ℕ, satisfies_prime_conditions e ∧ satisfies_square_sum_conditions a b c d e :=
  sorry

end problem_part_a_problem_part_b_l403_40385


namespace sqrt_inequality_l403_40386

theorem sqrt_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) : 
  x^2 + y^2 + 1 ≤ Real.sqrt ((x^3 + y + 1) * (y^3 + x + 1)) :=
sorry

end sqrt_inequality_l403_40386


namespace initial_avg_mark_l403_40379

variable (A : ℝ) -- The initial average mark

-- Conditions
def num_students : ℕ := 33
def avg_excluded_students : ℝ := 40
def num_excluded_students : ℕ := 3
def avg_remaining_students : ℝ := 95

-- Equation derived from the problem conditions
def initial_avg :=
  A * num_students - avg_excluded_students * num_excluded_students = avg_remaining_students * (num_students - num_excluded_students)

theorem initial_avg_mark :
  initial_avg A →
  A = 90 :=
by
  intro h
  sorry

end initial_avg_mark_l403_40379


namespace base_k_to_decimal_is_5_l403_40342

theorem base_k_to_decimal_is_5 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 42) : k = 5 := sorry

end base_k_to_decimal_is_5_l403_40342


namespace correct_statements_l403_40392

theorem correct_statements (a b c : ℝ) (h : ∀ x, ax^2 + bx + c > 0 ↔ -2 < x ∧ x < 3) :
  ( ∃ (x : ℝ), c*x^2 + b*x + a < 0 ↔ -1/2 < x ∧ x < 1/3 ) ∧
  ( ∃ (b : ℝ), ∀ b, 12/(3*b + 4) + b = 8/3 ) ∧
  ( ∀ m, ¬ (m < -1 ∨ m > 2) ) ∧
  ( c = 2 → ∀ n1 n2, (3*a*n1^2 + 6*b*n1 = -3 ∧ 3*a*n2^2 + 6*b*n2 = 1) → n2 - n1 ∈ [2, 4] ) :=
sorry

end correct_statements_l403_40392


namespace initial_alcohol_solution_percentage_l403_40367

noncomputable def initial_percentage_of_alcohol (P : ℝ) :=
  let initial_volume := 6 -- initial volume of solution in liters
  let added_alcohol := 1.2 -- added volume of pure alcohol in liters
  let final_volume := initial_volume + added_alcohol -- final volume in liters
  let final_percentage := 0.5 -- final percentage of alcohol
  ∃ P, (initial_volume * (P / 100) + added_alcohol) / final_volume = final_percentage

theorem initial_alcohol_solution_percentage : initial_percentage_of_alcohol 40 :=
by 
  -- Prove that initial percentage P is 40
  have hs : initial_percentage_of_alcohol 40 := by sorry
  exact hs

end initial_alcohol_solution_percentage_l403_40367


namespace Margie_can_drive_200_miles_l403_40370

/--
  Margie's car can go 40 miles per gallon of gas, and the price of gas is $5 per gallon.
  Prove that Margie can drive 200 miles with $25 worth of gas.
-/
theorem Margie_can_drive_200_miles (miles_per_gallon price_per_gallon money_available : ℕ) 
  (h1 : miles_per_gallon = 40) (h2 : price_per_gallon = 5) (h3 : money_available = 25) : 
  (money_available / price_per_gallon) * miles_per_gallon = 200 :=
by 
  /- The proof goes here -/
  sorry

end Margie_can_drive_200_miles_l403_40370


namespace number_of_valid_sets_l403_40344

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8,9,10}
def valid_set (A : Set ℕ) : Prop :=
  ∃ a1 a2 a3, A = {a1, a2, a3} ∧ a3 ∈ U ∧ a2 ∈ U ∧ a1 ∈ U ∧ a3 ≥ a2 + 1 ∧ a2 ≥ a1 + 4

theorem number_of_valid_sets : ∃ (n : ℕ), n = 56 ∧ ∃ S : Finset (Set ℕ), (∀ A ∈ S, valid_set A) ∧ S.card = n := by
  sorry

end number_of_valid_sets_l403_40344


namespace B_and_C_together_l403_40324

theorem B_and_C_together (A B C : ℕ) (h1 : A + B + C = 1000) (h2 : A + C = 700) (h3 : C = 300) :
  B + C = 600 :=
by
  sorry

end B_and_C_together_l403_40324


namespace find_value_of_expression_l403_40393

noncomputable def p : ℝ := 3
noncomputable def q : ℝ := 7
noncomputable def r : ℝ := 5

def inequality_holds (f : ℝ → ℝ) : Prop :=
  ∀ x, (f x ≥ 0 ↔ (x ∈ Set.Icc 3 7 ∨ x > 5))

def given_condition : Prop := p < q

theorem find_value_of_expression (f : ℝ → ℝ)
  (h : inequality_holds f)
  (hc : given_condition) :
  p + 2*q + 3*r = 32 := 
sorry

end find_value_of_expression_l403_40393


namespace intersection_A_B_range_m_l403_40310

-- Define set A when m = 3 as given
def A_set (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0
def A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define set B when m = 3 as given
def B_set (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

-- The intersection of A and B should be: -2 ≤ x ≤ 1
theorem intersection_A_B : ∀ (x : ℝ), A x ∧ B x ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

-- Define A for general m > 0
def A_set_general (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0

-- Define B for general m
def B_set_general (x : ℝ) (m : ℝ) : Prop := (x - 1)^2 ≤ m^2

-- Prove the range for m such that A ⊆ B
theorem range_m (m : ℝ) (h : m > 0) : (∀ x, A_set_general x → B_set_general x m) ↔ m ≥ 4 := sorry

end intersection_A_B_range_m_l403_40310


namespace min_xy_solution_l403_40377

theorem min_xy_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 2 * x + 8 * y) :
  (x = 16 ∧ y = 4) :=
by
  sorry

end min_xy_solution_l403_40377


namespace range_of_a_l403_40358

variable (a : ℝ)

def condition1 : Prop := a < 0
def condition2 : Prop := -a / 2 ≥ 1
def condition3 : Prop := -1 - a - 5 ≤ a

theorem range_of_a :
  condition1 a ∧ condition2 a ∧ condition3 a → -3 ≤ a ∧ a ≤ -2 :=
by
  sorry

end range_of_a_l403_40358


namespace tangent_line_ln_curve_l403_40323

theorem tangent_line_ln_curve (a : ℝ) :
  (∃ x y : ℝ, y = Real.log x + a ∧ x - y + 1 = 0 ∧ (∀ t : ℝ, t = x → (t - (Real.log t + a)) = -(1 - a))) → a = 2 :=
by
  sorry

end tangent_line_ln_curve_l403_40323


namespace num_shirts_sold_l403_40375

theorem num_shirts_sold (p_jeans : ℕ) (c_shirt : ℕ) (total_earnings : ℕ) (h1 : p_jeans = 10) (h2 : c_shirt = 10) (h3 : total_earnings = 400) : ℕ :=
  let c_jeans := 2 * c_shirt
  let n_shirts := 20
  have h4 : p_jeans * c_jeans + n_shirts * c_shirt = total_earnings := by sorry
  n_shirts

end num_shirts_sold_l403_40375


namespace min_value_xy_expression_l403_40318

theorem min_value_xy_expression : ∃ x y : ℝ, (xy - 2)^2 + (x^2 + y^2) = 4 :=
by
  sorry

end min_value_xy_expression_l403_40318


namespace BretCatchesFrogs_l403_40359

-- Define the number of frogs caught by Alster, Quinn, and Bret.
def AlsterFrogs : Nat := 2
def QuinnFrogs (a : Nat) : Nat := 2 * a
def BretFrogs (q : Nat) : Nat := 3 * q

-- The main theorem to prove
theorem BretCatchesFrogs : BretFrogs (QuinnFrogs AlsterFrogs) = 12 :=
by
  sorry

end BretCatchesFrogs_l403_40359


namespace square_area_l403_40383

theorem square_area (side_length : ℕ) (h : side_length = 16) : side_length * side_length = 256 := by
  sorry

end square_area_l403_40383


namespace johns_total_distance_l403_40365

theorem johns_total_distance :
  let monday := 1700
  let tuesday := monday + 200
  let wednesday := 0.7 * tuesday
  let thursday := 2 * wednesday
  let friday := 3.5 * 1000
  let saturday := 0
  monday + tuesday + wednesday + thursday + friday + saturday = 10090 := 
by
  sorry

end johns_total_distance_l403_40365


namespace find_y_values_l403_40340

def A (y : ℝ) : ℝ := 1 - y - 2 * y^2

theorem find_y_values (y : ℝ) (h₁ : y ≤ 1) (h₂ : y ≠ 0) (h₃ : y ≠ -1) (h₄ : y ≠ 0.5) :
  y^2 * A y / (y * A y) ≤ 1 ↔
  y ∈ Set.Iio (-1) ∪ Set.Ioo (-1) (1/2) ∪ Set.Ioc (1/2) 1 :=
by
  -- proof is omitted
  sorry

end find_y_values_l403_40340


namespace solve_system_eq_l403_40354

theorem solve_system_eq (x y : ℝ) :
  x^2 + y^2 + 6 * x * y = 68 ∧ 2 * x^2 + 2 * y^2 - 3 * x * y = 16 ↔
  (x = 4 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ (x = -4 ∧ y = -2) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end solve_system_eq_l403_40354


namespace pirates_treasure_l403_40345

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l403_40345


namespace smallest_number_starts_with_four_and_decreases_four_times_l403_40331

theorem smallest_number_starts_with_four_and_decreases_four_times :
  ∃ (X : ℕ), ∃ (A n : ℕ), (X = 4 * 10^n + A ∧ X = 4 * (10 * A + 4)) ∧ X = 410256 := 
by
  sorry

end smallest_number_starts_with_four_and_decreases_four_times_l403_40331


namespace days_to_learn_all_vowels_l403_40334

-- Defining the number of vowels
def number_of_vowels : Nat := 5

-- Defining the days Charles takes to learn one alphabet
def days_per_vowel : Nat := 7

-- Prove that Charles needs 35 days to learn all the vowels
theorem days_to_learn_all_vowels : number_of_vowels * days_per_vowel = 35 := by
  sorry

end days_to_learn_all_vowels_l403_40334


namespace arrange_scores_l403_40390

variable {K Q M S : ℝ}

theorem arrange_scores (h1 : Q > K) (h2 : M > S) (h3 : S < max Q (max M K)) : S < M ∧ M < Q := by
  sorry

end arrange_scores_l403_40390


namespace find_all_x_satisfying_condition_l403_40357

theorem find_all_x_satisfying_condition :
  ∃ (x : Fin 2016 → ℝ), 
  (∀ i : Fin 2016, x (i + 1) % 2016 = x 0) ∧
  (∀ i : Fin 2016, x i ^ 2 + x i - 1 = x ((i + 1) % 2016)) ∧
  (∀ i : Fin 2016, x i = 1 ∨ x i = -1) :=
sorry

end find_all_x_satisfying_condition_l403_40357


namespace problem_statement_l403_40395

-- Define the universal set U, and sets A and B
def U : Set ℕ := { n | 1 ≤ n ∧ n ≤ 10 }
def A : Set ℕ := {1, 2, 3, 5, 8}
def B : Set ℕ := {1, 3, 5, 7, 9}

-- Define the complement of set A with respect to U
def complement_U_A : Set ℕ := { n | n ∈ U ∧ n ∉ A }

-- Define the intersection of complement_U_A and B
def intersection_complement_U_A_B : Set ℕ := { n | n ∈ complement_U_A ∧ n ∈ B }

-- Prove the given statement
theorem problem_statement : intersection_complement_U_A_B = {7, 9} := by
  sorry

end problem_statement_l403_40395


namespace sum_geometric_series_l403_40332

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l403_40332


namespace number_of_typists_needed_l403_40351

theorem number_of_typists_needed :
  (∃ t : ℕ, (20 * 40) / 20 * 60 * t = 180) ↔ t = 30 :=
by sorry

end number_of_typists_needed_l403_40351


namespace remy_gallons_used_l403_40376

def roman_usage (R : ℕ) : Prop := R + (3 * R + 1) = 33

def remy_usage (R : ℕ) (Remy : ℕ) : Prop := Remy = 3 * R + 1

theorem remy_gallons_used :
  ∃ R Remy : ℕ, roman_usage R ∧ remy_usage R Remy ∧ Remy = 25 :=
  by
    sorry

end remy_gallons_used_l403_40376


namespace evaluate_sixth_iteration_of_g_at_2_l403_40374

def g (x : ℤ) : ℤ := x^2 - 4 * x + 1

theorem evaluate_sixth_iteration_of_g_at_2 :
  g (g (g (g (g (g 2))))) = 59162302643740737293922 := by
  sorry

end evaluate_sixth_iteration_of_g_at_2_l403_40374


namespace basketball_points_l403_40339

/-
In a basketball league, each game must have a winner and a loser. 
A team earns 2 points for a win and 1 point for a loss. 
A certain team expects to earn at least 48 points in all 32 games of 
the 2012-2013 season in order to have a chance to enter the playoffs. 
If this team wins x games in the upcoming matches, prove that
the relationship that x should satisfy to reach the goal is:
    2x + (32 - x) ≥ 48.
-/
theorem basketball_points (x : ℕ) (h : 0 ≤ x ∧ x ≤ 32) :
    2 * x + (32 - x) ≥ 48 :=
sorry

end basketball_points_l403_40339


namespace thirty_one_star_thirty_two_l403_40381

def complex_op (x y : ℝ) : ℝ :=
sorry

axiom op_zero (x : ℝ) : complex_op x 0 = 1

axiom op_associative (x y z : ℝ) : complex_op (complex_op x y) z = z * (x * y) + z

theorem thirty_one_star_thirty_two : complex_op 31 32 = 993 :=
by
  sorry

end thirty_one_star_thirty_two_l403_40381


namespace odd_function_fixed_point_l403_40316

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_fixed_point 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f) :
  f (0) = 0 → f (-1 + 1) - 2 = -2 :=
by
  sorry

end odd_function_fixed_point_l403_40316


namespace original_weight_l403_40387

variable (W : ℝ) -- Let W be the original weight of the side of beef

-- Conditions
def condition1 : ℝ := 0.80 * W -- Weight after first stage
def condition2 : ℝ := 0.70 * condition1 W -- Weight after second stage
def condition3 : ℝ := 0.75 * condition2 W -- Weight after third stage

-- Final weight is given as 570 pounds
theorem original_weight (h : condition3 W = 570) : W = 1357.14 :=
by 
  sorry

end original_weight_l403_40387


namespace correct_evaluation_l403_40337

noncomputable def evaluate_expression : ℚ :=
  - (2 : ℚ) ^ 3 + (6 / 5) * (2 / 5)

theorem correct_evaluation : evaluate_expression = -7 - 13 / 25 :=
by
  unfold evaluate_expression
  sorry

end correct_evaluation_l403_40337


namespace candidate_1_fails_by_40_marks_l403_40369

-- Definitions based on the conditions
def total_marks (T : ℕ) := T
def passing_marks (pass : ℕ) := pass = 160
def candidate_1_failed_by (marks_failed_by : ℕ) := ∃ (T : ℕ), (0.4 : ℝ) * T = 0.4 * T ∧ (0.6 : ℝ) * T - 20 = 160

-- Theorem to prove the first candidate fails by 40 marks
theorem candidate_1_fails_by_40_marks (marks_failed_by : ℕ) : candidate_1_failed_by marks_failed_by → marks_failed_by = 40 :=
by
  sorry

end candidate_1_fails_by_40_marks_l403_40369


namespace no_extrema_1_1_l403_40341

noncomputable def f (x : ℝ) : ℝ :=
  x^3 - 3 * x

theorem no_extrema_1_1 : ∀ x : ℝ, (x > -1) ∧ (x < 1) → ¬ (∃ c : ℝ, c ∈ Set.Ioo (-1) (1) ∧ (∀ y ∈ Set.Ioo (-1) (1), f y ≤ f c ∨ f c ≤ f y)) :=
by
  sorry

end no_extrema_1_1_l403_40341


namespace range_of_m_l403_40335

noncomputable def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
noncomputable def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, P x → Q x m ∧ P x) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end range_of_m_l403_40335


namespace gcd_of_three_numbers_l403_40338

theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd 324 243) 135 = 27 := 
by 
  sorry

end gcd_of_three_numbers_l403_40338


namespace meaningful_fraction_implies_neq_neg4_l403_40327

theorem meaningful_fraction_implies_neq_neg4 (x : ℝ) : (x + 4 ≠ 0) ↔ (x ≠ -4) := 
by
  sorry

end meaningful_fraction_implies_neq_neg4_l403_40327


namespace f_inequality_l403_40320

-- Definition of odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f is an odd function
variable {f : ℝ → ℝ}
variable (h1 : is_odd_function f)

-- f has a period of 4
variable (h2 : ∀ x, f (x + 4) = f x)

-- f is monotonically increasing on [0, 2)
variable (h3 : ∀ x y, 0 ≤ x → x < y → y < 2 → f x < f y)

theorem f_inequality : f 3 < 0 ∧ 0 < f 1 :=
by 
  -- Place proof here
  sorry

end f_inequality_l403_40320


namespace min_hypotenuse_of_right_triangle_l403_40319

theorem min_hypotenuse_of_right_triangle (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a + b + c = 6) : 
  c = 6 * (Real.sqrt 2 - 1) :=
sorry

end min_hypotenuse_of_right_triangle_l403_40319


namespace trapezium_area_l403_40301

theorem trapezium_area (a b area h : ℝ) (h1 : a = 20) (h2 : b = 15) (h3 : area = 245) :
  area = 1 / 2 * (a + b) * h → h = 14 :=
by
  sorry

end trapezium_area_l403_40301


namespace fish_population_estimate_l403_40349

theorem fish_population_estimate :
  ∃ N : ℕ, (60 * 60) / 2 = N ∧ (2 / 60 : ℚ) = (60 / N : ℚ) :=
by
  use 1800
  simp
  sorry

end fish_population_estimate_l403_40349


namespace greatest_prime_factor_of_n_l403_40346

noncomputable def n : ℕ := 4^17 - 2^29

theorem greatest_prime_factor_of_n :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ n → q ≤ p :=
sorry

end greatest_prime_factor_of_n_l403_40346


namespace largest_six_digit_number_l403_40388

/-- The largest six-digit number \( A \) that is divisible by 19, 
  the number obtained by removing its last digit is divisible by 17, 
  and the number obtained by removing the last two digits in \( A \) is divisible by 13 
  is \( 998412 \). -/
theorem largest_six_digit_number (A : ℕ) (h1 : A % 19 = 0) 
  (h2 : (A / 10) % 17 = 0) 
  (h3 : (A / 100) % 13 = 0) : 
  A = 998412 :=
sorry

end largest_six_digit_number_l403_40388
