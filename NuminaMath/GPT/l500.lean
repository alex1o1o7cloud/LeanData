import Mathlib

namespace fill_in_the_blank_correct_option_l500_50019

-- Assume each option is defined
def options := ["the other", "some", "another", "other"]

-- Define a helper function to validate the correct option
def is_correct_option (opt: String) : Prop :=
  opt = "another"

-- The main problem statement
theorem fill_in_the_blank_correct_option :
  (∀ opt, opt ∈ options → is_correct_option opt → opt = "another") :=
by
  intro opt h_option h_correct
  simp [is_correct_option] at h_correct
  exact h_correct

-- Test case to check the correct option
example : is_correct_option "another" :=
by
  simp [is_correct_option]

end fill_in_the_blank_correct_option_l500_50019


namespace beach_trip_time_l500_50020

noncomputable def totalTripTime (driveTime eachWay : ℝ) (beachTimeFactor : ℝ) : ℝ :=
  let totalDriveTime := eachWay * 2
  totalDriveTime + (totalDriveTime * beachTimeFactor)

theorem beach_trip_time :
  totalTripTime 2 2 2.5 = 14 := 
by
  sorry

end beach_trip_time_l500_50020


namespace common_difference_of_arithmetic_sequence_l500_50021

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
noncomputable def S_n (n : ℕ) : ℝ := -n^2 + 4*n

theorem common_difference_of_arithmetic_sequence :
  (∀ n : ℕ, S n = S_n n) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d ∧ d = -2 :=
by
  intro h
  use -2
  sorry

end common_difference_of_arithmetic_sequence_l500_50021


namespace find_positive_integers_n_l500_50087

open Real Int

noncomputable def satisfies_conditions (x y z : ℝ) (n : ℕ) : Prop :=
  sqrt x + sqrt y + sqrt z = 1 ∧ 
  (∃ m : ℤ, sqrt (x + n) + sqrt (y + n) + sqrt (z + n) = m)

theorem find_positive_integers_n (n : ℕ) :
  (∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ satisfies_conditions x y z n) ↔
  (∃ k : ℤ, k ≥ 1 ∧ (k % 9 = 1 ∨ k % 9 = 8) ∧ n = (k^2 - 1) / 9) :=
by
  sorry

end find_positive_integers_n_l500_50087


namespace checkerboard_black_squares_count_l500_50081

namespace Checkerboard

def is_black (n : ℕ) : Bool :=
  -- Define the alternating pattern of the checkerboard
  (n % 2 = 0)

def black_square_count (n : ℕ) : ℕ :=
  -- Calculate the number of black squares in a checkerboard of size n x n
  if n % 2 = 0 then n * n / 2 else n * n / 2 + n / 2 + 1

def additional_black_squares (n : ℕ) : ℕ :=
  -- Calculate the additional black squares due to modification of every 33rd square in every third row
  ((n - 1) / 3 + 1)

def total_black_squares (n : ℕ) : ℕ :=
  -- Calculate the total black squares considering the modified hypothesis
  black_square_count n + additional_black_squares n

theorem checkerboard_black_squares_count : total_black_squares 33 = 555 := 
  by sorry

end Checkerboard

end checkerboard_black_squares_count_l500_50081


namespace largest_common_value_less_than_1000_l500_50012

def arithmetic_sequence_1 (n : ℕ) : ℕ := 2 + 3 * n
def arithmetic_sequence_2 (m : ℕ) : ℕ := 4 + 8 * m

theorem largest_common_value_less_than_1000 :
  ∃ a n m : ℕ, a = arithmetic_sequence_1 n ∧ a = arithmetic_sequence_2 m ∧ a < 1000 ∧ a = 980 :=
by { sorry }

end largest_common_value_less_than_1000_l500_50012


namespace gcd_sum_product_pairwise_coprime_l500_50084

theorem gcd_sum_product_pairwise_coprime 
  (a b c : ℤ) 
  (h1 : Int.gcd a b = 1)
  (h2 : Int.gcd b c = 1)
  (h3 : Int.gcd a c = 1) : 
  Int.gcd (a * b + b * c + a * c) (a * b * c) = 1 := 
sorry

end gcd_sum_product_pairwise_coprime_l500_50084


namespace multiple_of_Mel_weight_l500_50083

/-- Given that Brenda weighs 10 pounds more than a certain multiple of Mel's weight,
    and given that Brenda weighs 220 pounds and Mel's weight is 70 pounds,
    show that the multiple is 3. -/
theorem multiple_of_Mel_weight 
    (Brenda_weight Mel_weight certain_multiple : ℝ) 
    (h1 : Brenda_weight = Mel_weight * certain_multiple + 10)
    (h2 : Brenda_weight = 220)
    (h3 : Mel_weight = 70) :
  certain_multiple = 3 :=
by 
  sorry

end multiple_of_Mel_weight_l500_50083


namespace number_of_oranges_l500_50066

def bananas : ℕ := 7
def apples : ℕ := 2 * bananas
def pears : ℕ := 4
def grapes : ℕ := apples / 2
def total_fruits : ℕ := 40

theorem number_of_oranges : total_fruits - (bananas + apples + pears + grapes) = 8 :=
by sorry

end number_of_oranges_l500_50066


namespace percentage_shaded_l500_50068

theorem percentage_shaded (total_squares shaded_squares : ℕ) (h1 : total_squares = 5 * 5) (h2 : shaded_squares = 9) :
  (shaded_squares:ℚ) / total_squares * 100 = 36 :=
by
  sorry

end percentage_shaded_l500_50068


namespace total_insects_l500_50024

theorem total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (ants_per_leaf : ℕ) (caterpillars_every_third_leaf : ℕ) :
  leaves = 84 →
  ladybugs_per_leaf = 139 →
  ants_per_leaf = 97 →
  caterpillars_every_third_leaf = 53 →
  (84 * 139) + (84 * 97) + (53 * (84 / 3)) = 21308 := 
by
  sorry

end total_insects_l500_50024


namespace slope_of_intersection_points_l500_50027

theorem slope_of_intersection_points {s x y : ℝ} 
  (h1 : 2 * x - 3 * y = 6 * s - 5) 
  (h2 : 3 * x + y = 9 * s + 4) : 
  ∃ m : ℝ, m = 3 ∧ (∀ s : ℝ, (∃ x y : ℝ, 2 * x - 3 * y = 6 * s - 5 ∧ 3 * x + y = 9 * s + 4) → y = m * x + (23/11)) := 
by
  sorry

end slope_of_intersection_points_l500_50027


namespace total_days_in_month_eq_l500_50070

-- Definition of the conditions
def took_capsules_days : ℕ := 27
def forgot_capsules_days : ℕ := 4

-- The statement to be proved
theorem total_days_in_month_eq : took_capsules_days + forgot_capsules_days = 31 := by
  sorry

end total_days_in_month_eq_l500_50070


namespace log_inequality_l500_50031

open Real

theorem log_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  log (1 + sqrt (a * b)) ≤ (1 / 2) * (log (1 + a) + log (1 + b)) :=
sorry

end log_inequality_l500_50031


namespace correct_subsidy_equation_l500_50032

-- Define the necessary variables and conditions
def sales_price (x : ℝ) := x  -- sales price of the mobile phone in yuan
def subsidy_rate : ℝ := 0.13  -- 13% subsidy rate
def number_of_phones : ℝ := 20  -- 20 units sold
def total_subsidy : ℝ := 2340  -- total subsidy provided

-- Lean theorem statement to prove the correct equation
theorem correct_subsidy_equation (x : ℝ) :
  number_of_phones * x * subsidy_rate = total_subsidy :=
by
  sorry -- proof to be completed

end correct_subsidy_equation_l500_50032


namespace smallest_value_of_3a_plus_2_l500_50069

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 3 * a + 2 = -1 :=
by
  sorry

end smallest_value_of_3a_plus_2_l500_50069


namespace correct_option_is_B_l500_50003

variable (f : ℝ → ℝ)
variable (h0 : f 0 = 2)
variable (h1 : ∀ x : ℝ, deriv f x > f x + 1)

theorem correct_option_is_B : 3 * Real.exp (1 : ℝ) < f 2 + 1 := sorry

end correct_option_is_B_l500_50003


namespace math_problem_l500_50010

theorem math_problem (x : ℂ) (hx : x + 1/x = 3) : x^6 + 1/x^6 = 322 := 
by 
  sorry

end math_problem_l500_50010


namespace greatest_constant_triangle_l500_50097

theorem greatest_constant_triangle (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  ∃ N : ℝ, (∀ (a b c : ℝ), a + b > c → b + c > a → c + a > b → (a^2 + b^2 + a * b) / c^2 > N) ∧ N = 3 / 4 :=
  sorry

end greatest_constant_triangle_l500_50097


namespace not_subset_T_to_S_l500_50043

def is_odd (x : ℤ) : Prop := ∃ n : ℤ, x = 2 * n + 1
def is_of_form_4k_plus_1 (y : ℤ) : Prop := ∃ k : ℤ, y = 4 * k + 1

theorem not_subset_T_to_S :
  ¬ (∀ y, is_of_form_4k_plus_1 y → is_odd y) :=
sorry

end not_subset_T_to_S_l500_50043


namespace number_division_l500_50034

theorem number_division (n : ℕ) (h1 : 555 + 445 = 1000) (h2 : 555 - 445 = 110) 
  (h3 : n % 1000 = 80) (h4 : n / 1000 = 220) : n = 220080 :=
by {
  -- proof steps would go here
  sorry
}

end number_division_l500_50034


namespace problem_rect_ratio_l500_50022

theorem problem_rect_ratio (W X Y Z U V R S : ℝ × ℝ) 
  (hYZ : Y = (0, 0))
  (hW : W = (0, 6))
  (hZ : Z = (7, 6))
  (hX : X = (7, 4))
  (hU : U = (5, 0))
  (hV : V = (4, 4))
  (hR : R = (5 / 3, 4))
  (hS : S = (0, 4))
  : (dist R S) / (dist X V) = 5 / 9 := 
sorry

end problem_rect_ratio_l500_50022


namespace proof_problem_l500_50075

-- Define the problem conditions
variables (x y : ℝ)

-- State the theorem
theorem proof_problem (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 :=
by sorry

end proof_problem_l500_50075


namespace area_of_triangle_l500_50054

theorem area_of_triangle (p : ℝ) (h_p : 0 < p ∧ p < 10) : 
    let C := (0, p)
    let O := (0, 0)
    let B := (10, 0)
    (1/2) * 10 * p = 5 * p := 
by
  sorry

end area_of_triangle_l500_50054


namespace calculate_number_of_girls_l500_50056

-- Definitions based on the conditions provided
def ratio_girls_to_boys : ℕ := 3
def ratio_boys_to_girls : ℕ := 4
def total_students : ℕ := 35

-- The proof statement
theorem calculate_number_of_girls (k : ℕ) (hk : ratio_girls_to_boys * k + ratio_boys_to_girls * k = total_students) :
  ratio_girls_to_boys * k = 15 :=
by sorry

end calculate_number_of_girls_l500_50056


namespace root_iff_coeff_sum_zero_l500_50058

theorem root_iff_coeff_sum_zero (a b c : ℝ) :
    (a * 1^2 + b * 1 + c = 0) ↔ (a + b + c = 0) := sorry

end root_iff_coeff_sum_zero_l500_50058


namespace range_of_independent_variable_x_l500_50007

noncomputable def range_of_x (x : ℝ) : Prop :=
  x > -2

theorem range_of_independent_variable_x (x : ℝ) :
  ∀ x, (x + 2 > 0) → range_of_x x :=
by
  intro x h
  unfold range_of_x
  linarith

end range_of_independent_variable_x_l500_50007


namespace people_counted_l500_50060

-- Define the conditions
def first_day_count (second_day_count : ℕ) : ℕ := 2 * second_day_count
def second_day_count : ℕ := 500

-- Define the total count
def total_count (first_day : ℕ) (second_day : ℕ) : ℕ := first_day + second_day

-- Statement of the proof problem: Prove that the total count is 1500 given the conditions
theorem people_counted : total_count (first_day_count second_day_count) second_day_count = 1500 := by
  sorry

end people_counted_l500_50060


namespace sum_two_and_four_l500_50004

theorem sum_two_and_four : 2 + 4 = 6 := by
  sorry

end sum_two_and_four_l500_50004


namespace largest_perimeter_l500_50099

noncomputable def triangle_largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : ℕ :=
7 + 8 + y

theorem largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : triangle_largest_perimeter y h1 h2 = 29 :=
sorry

end largest_perimeter_l500_50099


namespace find_value_l500_50059

-- Define the theorem with the given conditions and the expected result
theorem find_value (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + a * c^2 + a + b + c = 2 * (a * b + b * c + a * c)) :
  c^2017 / (a^2016 + b^2018) = 1 / 2 :=
sorry

end find_value_l500_50059


namespace cylindrical_to_rectangular_l500_50050

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 6) (hθ : θ = π / 3) (hz : z = -3) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, -3) :=
by
  sorry

end cylindrical_to_rectangular_l500_50050


namespace min_gennadies_l500_50063

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l500_50063


namespace smallest_total_squares_l500_50052

theorem smallest_total_squares (n : ℕ) (h : 4 * n - 4 = 2 * n) : n^2 = 4 :=
by
  sorry

end smallest_total_squares_l500_50052


namespace total_white_roses_l500_50036

-- Define the constants
def n_b : ℕ := 5
def n_t : ℕ := 7
def r_b : ℕ := 5
def r_t : ℕ := 12

-- State the theorem
theorem total_white_roses :
  n_t * r_t + n_b * r_b = 109 :=
by
  -- Automatic proof can be here; using sorry as placeholder
  sorry

end total_white_roses_l500_50036


namespace shrimp_per_pound_l500_50016

theorem shrimp_per_pound (shrimp_per_guest guests : ℕ) (cost_per_pound : ℝ) (total_spent : ℝ)
  (hshrimp_per_guest : shrimp_per_guest = 5) (hguests : guests = 40) (hcost_per_pound : cost_per_pound = 17.0) (htotal_spent : total_spent = 170.0) :
  let total_shrimp := shrimp_per_guest * guests
  let total_pounds := total_spent / cost_per_pound
  total_shrimp / total_pounds = 20 :=
by
  sorry

end shrimp_per_pound_l500_50016


namespace compare_fractions_l500_50061

theorem compare_fractions : (31 : ℚ) / 11 > (17 : ℚ) / 14 := 
by
  sorry

end compare_fractions_l500_50061


namespace smallest_int_ends_in_3_div_by_11_l500_50057

theorem smallest_int_ends_in_3_div_by_11 :
  ∃ k : ℕ, k > 0 ∧ k % 10 = 3 ∧ k % 11 = 0 ∧ k = 33 :=
by {
  sorry
}

end smallest_int_ends_in_3_div_by_11_l500_50057


namespace paul_initial_crayons_l500_50047

-- Define the variables for the crayons given away, lost, and left
def crayons_given_away : ℕ := 563
def crayons_lost : ℕ := 558
def crayons_left : ℕ := 332

-- Define the total number of crayons Paul got for his birthday
def initial_crayons : ℕ := 1453

-- The proof statement
theorem paul_initial_crayons :
  initial_crayons = crayons_given_away + crayons_lost + crayons_left :=
sorry

end paul_initial_crayons_l500_50047


namespace internal_angle_sine_l500_50002

theorem internal_angle_sine (α : ℝ) (h1 : α > 0 ∧ α < 180) (h2 : Real.sin (α * (Real.pi / 180)) = 1 / 2) : α = 30 ∨ α = 150 :=
sorry

end internal_angle_sine_l500_50002


namespace solve_equation_l500_50088

theorem solve_equation :
  ∀ x : ℝ, (1 + 2 * x ^ (1/2) - x ^ (1/3) - 2 * x ^ (1/6) = 0) ↔ (x = 1 ∨ x = 1 / 64) :=
by
  sorry

end solve_equation_l500_50088


namespace temperature_on_Monday_l500_50093

theorem temperature_on_Monday 
  (M T W Th F : ℝ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : F = 36) : 
  M = 44 := 
by 
  -- Proof omitted
  sorry

end temperature_on_Monday_l500_50093


namespace exists_m_such_that_m_poly_is_zero_mod_p_l500_50080

theorem exists_m_such_that_m_poly_is_zero_mod_p (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 7 = 1) :
  ∃ m : ℕ, m > 0 ∧ (m^3 + m^2 - 2*m - 1) % p = 0 := 
sorry

end exists_m_such_that_m_poly_is_zero_mod_p_l500_50080


namespace July_husband_age_l500_50067

namespace AgeProof

variable (HannahAge JulyAge HusbandAge : ℕ)

def double_age_condition (hannah_age : ℕ) (july_age : ℕ) : Prop :=
  hannah_age = 2 * july_age

def twenty_years_later (current_age : ℕ) : ℕ :=
  current_age + 20

def two_years_older (age : ℕ) : ℕ :=
  age + 2

theorem July_husband_age :
  ∃ (hannah_age july_age : ℕ), double_age_condition hannah_age july_age ∧
    twenty_years_later hannah_age = 26 ∧
    twenty_years_later july_age = 23 ∧
    two_years_older (twenty_years_later july_age) = 25 :=
by
  sorry
end AgeProof

end July_husband_age_l500_50067


namespace Elza_winning_strategy_l500_50009

-- Define a hypothetical graph structure
noncomputable def cities := {i : ℕ // 1 ≤ i ∧ i ≤ 2013}
def connected (c1 c2 : cities) : Prop := sorry

theorem Elza_winning_strategy 
  (N : ℕ) 
  (roads : (cities × cities) → Prop) 
  (h1 : ∀ c1 c2, roads (c1, c2) → connected c1 c2)
  (h2 : N = 1006): 
  ∃ (strategy : cities → Prop), 
  (∃ c1 c2 : cities, (strategy c1 ∧ strategy c2)) ∧ connected c1 c2 :=
by 
  sorry

end Elza_winning_strategy_l500_50009


namespace original_price_of_second_pair_l500_50091

variable (P : ℝ) -- original price of the second pair of shoes
variable (discounted_price : ℝ := P / 2)
variable (total_before_discount : ℝ := 40 + discounted_price)
variable (final_payment : ℝ := (3 / 4) * total_before_discount)
variable (payment : ℝ := 60)

theorem original_price_of_second_pair (h : final_payment = payment) : P = 80 :=
by
  -- Skipping the proof with sorry.
  sorry

end original_price_of_second_pair_l500_50091


namespace find_number_l500_50053

theorem find_number (x : ℕ) (h : x + 56 = 110) : x = 54 :=
sorry

end find_number_l500_50053


namespace find_g_inv_l500_50037

noncomputable def g (x : ℝ) : ℝ :=
  (x^7 - 1) / 4

noncomputable def g_inv_value : ℝ :=
  (51 / 32)^(1/7)

theorem find_g_inv (h : g (g_inv_value) = 19 / 128) : g_inv_value = (51 / 32)^(1/7) :=
by
  sorry

end find_g_inv_l500_50037


namespace min_abs_A_l500_50038

def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def A (a d : ℚ) (n : ℕ) : ℚ :=
  (arithmetic_sequence a d n) + (arithmetic_sequence a d (n + 1)) + 
  (arithmetic_sequence a d (n + 2)) + (arithmetic_sequence a d (n + 3)) + 
  (arithmetic_sequence a d (n + 4)) + (arithmetic_sequence a d (n + 5)) + 
  (arithmetic_sequence a d (n + 6))

theorem min_abs_A : (arithmetic_sequence 19 (-4/5) 26 = -1) ∧ 
                    (∀ n, 1 ≤ n) →
                    ∃ n : ℕ, |A 19 (-4/5) n| = 7/5 :=
by
  sorry

end min_abs_A_l500_50038


namespace number_of_hikers_in_the_morning_l500_50098

theorem number_of_hikers_in_the_morning (H : ℕ) :
  41 + 26 + H = 71 → H = 4 :=
by
  intros h_eq
  sorry

end number_of_hikers_in_the_morning_l500_50098


namespace VishalInvestedMoreThanTrishulBy10Percent_l500_50005

variables (R T V : ℝ)

-- Given conditions
def RaghuInvests (R : ℝ) : Prop := R = 2500
def TrishulInvests (R T : ℝ) : Prop := T = 0.9 * R
def TotalInvestment (R T V : ℝ) : Prop := V + T + R = 7225
def PercentageInvestedMore (T V : ℝ) (P : ℝ) : Prop := P * T = V - T

-- Main theorem to prove
theorem VishalInvestedMoreThanTrishulBy10Percent (R T V : ℝ) (P : ℝ) :
  RaghuInvests R ∧ TrishulInvests R T ∧ TotalInvestment R T V → PercentageInvestedMore T V P → P = 0.1 :=
by
  intros
  sorry

end VishalInvestedMoreThanTrishulBy10Percent_l500_50005


namespace total_cost_of_fruits_l500_50085

theorem total_cost_of_fruits (h_orange_weight : 12 * 2 = 24)
                             (h_apple_weight : 8 * 3.75 = 30)
                             (price_orange : ℝ := 1.5)
                             (price_apple : ℝ := 2.0) :
  (5 * 2 * price_orange + 4 * 3.75 * price_apple) = 45 :=
by
  sorry

end total_cost_of_fruits_l500_50085


namespace quadratic_value_at_3_l500_50030

theorem quadratic_value_at_3 (a b c : ℝ) :
  (a * (-2)^2 + b * (-2) + c = -13 / 2) →
  (a * (-1)^2 + b * (-1) + c = -4) →
  (a * 0^2 + b * 0 + c = -2.5) →
  (a * 1^2 + b * 1 + c = -2) →
  (a * 2^2 + b * 2 + c = -2.5) →
  (a * 3^2 + b * 3 + c = -4) :=
by
  sorry

end quadratic_value_at_3_l500_50030


namespace sin_double_angle_l500_50064

theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 1 / 3) (h2 : (π / 2) < α ∧ α < π) :
  Real.sin (2 * α) = - (4 * Real.sqrt 2) / 9 := sorry

end sin_double_angle_l500_50064


namespace problem_1_problem_2_l500_50062

-- Problem (1): Proving the solutions for \( x^2 - 3x = 0 \)
theorem problem_1 : ∀ x : ℝ, x^2 - 3 * x = 0 ↔ (x = 0 ∨ x = 3) :=
by
  intro x
  sorry

-- Problem (2): Proving the solutions for \( 5x + 2 = 3x^2 \)
theorem problem_2 : ∀ x : ℝ, 5 * x + 2 = 3 * x^2 ↔ (x = -1/3 ∨ x = 2) :=
by
  intro x
  sorry

end problem_1_problem_2_l500_50062


namespace sample_size_correct_l500_50078

-- Define the total number of students in a certain grade.
def total_students : ℕ := 500

-- Define the number of students selected for statistical analysis.
def selected_students : ℕ := 30

-- State the theorem to prove the selected students represent the sample size.
theorem sample_size_correct : selected_students = 30 := by
  -- The proof would go here, but we use sorry to indicate it is skipped.
  sorry

end sample_size_correct_l500_50078


namespace evaluate_N_l500_50095

theorem evaluate_N (N : ℕ) :
    988 + 990 + 992 + 994 + 996 = 5000 - N → N = 40 :=
by
  sorry

end evaluate_N_l500_50095


namespace intersection_of_M_and_N_l500_50077

def set_M : Set ℝ := {x | 0 ≤ x ∧ x < 2}
def set_N : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def intersection_M_N : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := 
by sorry

end intersection_of_M_and_N_l500_50077


namespace sufficient_condition_l500_50039

theorem sufficient_condition (a b : ℝ) : ab ≠ 0 → a ≠ 0 :=
sorry

end sufficient_condition_l500_50039


namespace difference_is_minus_four_l500_50013

def percentage_scoring_60 : ℝ := 0.15
def percentage_scoring_75 : ℝ := 0.25
def percentage_scoring_85 : ℝ := 0.40
def percentage_scoring_95 : ℝ := 1 - (percentage_scoring_60 + percentage_scoring_75 + percentage_scoring_85)

def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_85 : ℝ := 85
def score_95 : ℝ := 95

def mean_score : ℝ :=
  (percentage_scoring_60 * score_60) +
  (percentage_scoring_75 * score_75) +
  (percentage_scoring_85 * score_85) +
  (percentage_scoring_95 * score_95)

def median_score : ℝ := score_85

def difference_mean_median : ℝ := mean_score - median_score

theorem difference_is_minus_four : difference_mean_median = -4 :=
by
  sorry

end difference_is_minus_four_l500_50013


namespace box_weight_l500_50082

theorem box_weight (W : ℝ) (h : 7 * (W - 20) = 3 * W) : W = 35 := by
  sorry

end box_weight_l500_50082


namespace a_2023_le_1_l500_50011

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, (a (n+1))^2 + a n * a (n+2) ≤ a n + a (n+2))

theorem a_2023_le_1 : a 2023 ≤ 1 := by
  sorry

end a_2023_le_1_l500_50011


namespace positive_reals_condition_l500_50071

theorem positive_reals_condition (a : ℝ) (h_pos : 0 < a) : a < 2 :=
by
  -- Problem conditions:
  -- There exists a positive integer n and n pairwise disjoint infinite sets A_i
  -- such that A_1 ∪ ... ∪ A_n = ℕ* and for any two numbers b > c in each A_i,
  -- b - c ≥ a^i.

  sorry

end positive_reals_condition_l500_50071


namespace polygon_of_T_has_4_sides_l500_50072

def T (b : ℝ) (x y : ℝ) : Prop :=
  b ≤ x ∧ x ≤ 4 * b ∧
  b ≤ y ∧ y ≤ 4 * b ∧
  x + y ≥ 3 * b ∧
  x + 2 * b ≥ 2 * y ∧
  2 * y ≥ x + b

noncomputable def sides_of_T (b : ℝ) : ℕ :=
  if b > 0 then 4 else 0

theorem polygon_of_T_has_4_sides (b : ℝ) (hb : b > 0) : sides_of_T b = 4 := by
  sorry

end polygon_of_T_has_4_sides_l500_50072


namespace modulo_remainder_product_l500_50046

theorem modulo_remainder_product :
  let a := 2022
  let b := 2023
  let c := 2024
  let d := 2025
  let n := 17
  (a * b * c * d) % n = 0 :=
by
  sorry

end modulo_remainder_product_l500_50046


namespace equation_of_line_l500_50040

theorem equation_of_line (l : ℝ → ℝ) :
  (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x : ℝ, l x = (2 * l a / a) * x))
  ∨ (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x y : ℝ, 2 * x + y - 4 = 0)) := sorry

end equation_of_line_l500_50040


namespace john_mean_score_l500_50090

-- Define John's quiz scores as a list
def johnQuizScores := [95, 88, 90, 92, 94, 89]

-- Define the function to calculate the mean of a list of integers
def mean_scores (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

-- Prove that the mean of John's quiz scores is 91.3333 
theorem john_mean_score :
  mean_scores johnQuizScores = 91.3333 := by
  -- sorry is a placeholder for the missing proof
  sorry

end john_mean_score_l500_50090


namespace longer_bus_ride_l500_50025

theorem longer_bus_ride :
  let oscar := 0.75
  let charlie := 0.25
  oscar - charlie = 0.50 :=
by
  sorry

end longer_bus_ride_l500_50025


namespace largest_n_satisfying_inequality_l500_50079

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, (1/4 + n/6 < 3/2) ∧ (∀ m : ℕ, m > n → 1/4 + m/6 ≥ 3/2) :=
by
  sorry

end largest_n_satisfying_inequality_l500_50079


namespace expression_value_l500_50051

theorem expression_value 
  (x : ℝ)
  (h : x = 1/5) :
  (x^2 - 4) / (x^2 - 2 * x) = 11 :=
  by
  rw [h]
  sorry

end expression_value_l500_50051


namespace max_value_of_e_l500_50045

theorem max_value_of_e (a b c d e : ℝ) 
  (h₁ : a + b + c + d + e = 8) 
  (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  e ≤ 16 / 5 :=
sorry

end max_value_of_e_l500_50045


namespace ratio_of_a_to_b_l500_50014

variables (a b x m : ℝ)
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variables (h_x : x = a + 0.25 * a)
variables (h_m : m = b - 0.80 * b)
variables (h_ratio : m / x = 0.2)

theorem ratio_of_a_to_b (h_pos_a : 0 < a) (h_pos_b : 0 < b)
                        (h_x : x = a + 0.25 * a)
                        (h_m : m = b - 0.80 * b)
                        (h_ratio : m / x = 0.2) :
  a / b = 5 / 4 := by
  sorry

end ratio_of_a_to_b_l500_50014


namespace part1_part2_l500_50074

-- Define the initial conditions and the given inequality.
def condition1 (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def condition2 (m : ℝ) (x : ℝ) : Prop := x = (1/2)^(m - 1) ∧ 1 < m ∧ m < 2

-- Definitions of the correct ranges
def range_x (x : ℝ) : Prop := 1/2 < x ∧ x < 3/4
def range_a (a : ℝ) : Prop := 1/3 ≤ a ∧ a ≤ 1/2

-- Mathematical equivalent proof problem
theorem part1 {x : ℝ} (h1 : condition1 x (1/4)) (h2 : ∃ (m : ℝ), condition2 m x) : range_x x :=
sorry

theorem part2 {a : ℝ} (h : ∀ x : ℝ, (1/2 < x ∧ x < 1) → condition1 x a) : range_a a :=
sorry

end part1_part2_l500_50074


namespace least_common_multiple_xyz_l500_50048

theorem least_common_multiple_xyz (x y z : ℕ) 
  (h1 : Nat.lcm x y = 18) 
  (h2 : Nat.lcm y z = 20) : 
  Nat.lcm x z = 90 := 
sorry

end least_common_multiple_xyz_l500_50048


namespace shadow_problem_l500_50049

-- Define the conditions
def cube_edge_length : ℝ := 2
def shadow_area_outside : ℝ := 147
def total_shadow_area : ℝ := shadow_area_outside + cube_edge_length^2

-- The main statement to prove
theorem shadow_problem :
  let x := 4 / (Real.sqrt total_shadow_area - cube_edge_length)
  (⌊1000 * x⌋ : ℤ) = 481 :=
by
  let x := 4 / (Real.sqrt total_shadow_area - cube_edge_length)
  have h : (⌊1000 * x⌋ : ℤ) = 481 := sorry
  exact h

end shadow_problem_l500_50049


namespace problem_one_l500_50073

def S_n (n : Nat) : Nat := 
  List.foldl (fun acc x => acc * 10 + 2) 0 (List.replicate n 2)

theorem problem_one : ∃ n ∈ Finset.range 2011, S_n n % 2011 = 0 := 
  sorry

end problem_one_l500_50073


namespace polynomial_transformation_exists_l500_50028

theorem polynomial_transformation_exists (P : ℝ → ℝ → ℝ) (hP : ∀ x y, P (x - 1) (y - 2 * x + 1) = P x y) :
  ∃ Φ : ℝ → ℝ, ∀ x y, P x y = Φ (y - x^2) := by
  sorry

end polynomial_transformation_exists_l500_50028


namespace students_more_than_Yoongi_l500_50033

theorem students_more_than_Yoongi (total_players : ℕ) (less_than_Yoongi : ℕ) (total_players_eq : total_players = 21) (less_than_eq : less_than_Yoongi = 11) : 
  ∃ more_than_Yoongi : ℕ, more_than_Yoongi = (total_players - 1 - less_than_Yoongi) ∧ more_than_Yoongi = 8 :=
by
  sorry

end students_more_than_Yoongi_l500_50033


namespace range_of_set_of_three_numbers_l500_50035

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l500_50035


namespace value_of_b7b9_l500_50065

-- Define arithmetic sequence and geometric sequence with given conditions
variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- The given conditions in the problem
def a_seq_arithmetic (a : ℕ → ℝ) := ∀ n, a n = a 1 + (n - 1) • (a 2 - a 1)
def b_seq_geometric (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n, b (n + 1) = r * b n
def given_condition (a : ℕ → ℝ) := 2 * a 5 - (a 8)^2 + 2 * a 11 = 0
def b8_eq_a8 (a b : ℕ → ℝ) := b 8 = a 8

-- The statement to prove
theorem value_of_b7b9 : a_seq_arithmetic a → b_seq_geometric b → given_condition a → b8_eq_a8 a b → b 7 * b 9 = 4 := by
  intros a_arith b_geom cond b8a8
  sorry

end value_of_b7b9_l500_50065


namespace total_amount_l500_50089

theorem total_amount (T_pq r : ℝ) (h1 : r = 2/3 * T_pq) (h2 : r = 1600) : T_pq + r = 4000 :=
by
  -- proof skipped
  sorry

end total_amount_l500_50089


namespace find_a5_l500_50094

variable {a : ℕ → ℝ}

-- Condition 1: {a_n} is an arithmetic sequence
def arithmetic_sequence (a: ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition 2: a1 + a9 = 10
axiom a1_a9_sum : a 1 + a 9 = 10

theorem find_a5 (h_arith : arithmetic_sequence a) : a 5 = 5 :=
by {
  sorry
}

end find_a5_l500_50094


namespace number_of_cows_l500_50076

variable (C H : ℕ)

section
-- Condition 1: Cows have 4 legs each
def cows_legs := C * 4

-- Condition 2: Chickens have 2 legs each
def chickens_legs := H * 2

-- Condition 3: The number of legs was 10 more than twice the number of heads
def total_legs := cows_legs C + chickens_legs H = 2 * (C + H) + 10

theorem number_of_cows : total_legs C H → C = 5 :=
by
  intros h
  sorry

end

end number_of_cows_l500_50076


namespace range_of_a_l500_50042

theorem range_of_a (a : ℝ) (h_pos : a > 0)
  (p : ∀ x : ℝ, x^2 - 4 * a * x + 3 * a^2 ≤ 0)
  (q : ∀ x : ℝ, (x^2 - x - 6 < 0) ∧ (x^2 + 2 * x - 8 > 0)) :
  (a ∈ ((Set.Ioo 0 (2 / 3)) ∪ (Set.Ici 3))) :=
by
  sorry

end range_of_a_l500_50042


namespace mean_score_l500_50001

variable (mean stddev : ℝ)

-- Conditions
axiom condition1 : 42 = mean - 5 * stddev
axiom condition2 : 67 = mean + 2.5 * stddev

theorem mean_score : mean = 58.67 := 
by 
  -- You would need to provide proof here
  sorry

end mean_score_l500_50001


namespace find_x_satisfies_equation_l500_50026

theorem find_x_satisfies_equation :
  let x : ℤ := -14
  ∃ x : ℤ, (36 - x) - (14 - x) = 2 * ((36 - x) - (18 - x)) :=
by
  let x := -14
  use x
  sorry

end find_x_satisfies_equation_l500_50026


namespace simplify_expression_l500_50008

-- Define the question and conditions
theorem simplify_expression (x y : ℝ) (h : |x + 1| + (2 * y - 4)^2 = 0) :
  (2*x^2*y - 3*x*y) - 2*(x^2*y - x*y + 1/2*x*y^2) + x*y = 4 :=
by
  -- proof steps if needed, but currently replaced with 'sorry' to indicate proof needed
  sorry

end simplify_expression_l500_50008


namespace length_of_wall_correct_l500_50017

noncomputable def length_of_wall (s : ℝ) (w : ℝ) : ℝ :=
  let area_mirror := s * s
  let area_wall := 2 * area_mirror
  area_wall / w

theorem length_of_wall_correct : length_of_wall 18 32 = 20.25 :=
by
  -- This is the place for proof which is omitted deliberately
  sorry

end length_of_wall_correct_l500_50017


namespace change_in_us_volume_correct_l500_50055

-- Definition: Change in the total import and export volume of goods in a given year
def change_in_volume (country : String) : Float :=
  if country = "China" then 7.5
  else if country = "United States" then -6.4
  else 0

-- Theorem: The change in the total import and export volume of goods in the United States is correctly represented.
theorem change_in_us_volume_correct :
  change_in_volume "United States" = -6.4 := by
  sorry

end change_in_us_volume_correct_l500_50055


namespace price_sugar_salt_l500_50023

/-- The price of two kilograms of sugar and five kilograms of salt is $5.50. If a kilogram of sugar 
    costs $1.50, then how much is the price of three kilograms of sugar and some kilograms of salt, 
    if the total price is $5? -/
theorem price_sugar_salt 
  (price_sugar_per_kg : ℝ)
  (price_total_2kg_sugar_5kg_salt : ℝ)
  (total_price : ℝ) :
  price_sugar_per_kg = 1.50 →
  price_total_2kg_sugar_5kg_salt = 5.50 →
  total_price = 5 →
  2 * price_sugar_per_kg + 5 * (price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5 = 5.50 →
  3 * price_sugar_per_kg + (total_price - 3 * price_sugar_per_kg) / ((price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5) = 1 →
  true :=
by
  sorry

end price_sugar_salt_l500_50023


namespace x_intercept_perpendicular_l500_50006

theorem x_intercept_perpendicular (k m x y : ℝ) (h1 : 4 * x - 3 * y = 12) (h2 : y = -3/4 * x + 3) :
  x = 4 :=
by
  sorry

end x_intercept_perpendicular_l500_50006


namespace largest_divisor_for_consecutive_seven_odds_l500_50086

theorem largest_divisor_for_consecutive_seven_odds (n : ℤ) (h_even : 2 ∣ n) (h_pos : 0 < n) : 
  105 ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) :=
sorry

end largest_divisor_for_consecutive_seven_odds_l500_50086


namespace sum_first_11_even_numbers_is_132_l500_50015

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sum_first_11_even_numbers_is_132 : sum_first_n_even_numbers 11 = 132 := 
  by
    sorry

end sum_first_11_even_numbers_is_132_l500_50015


namespace total_students_mrs_mcgillicuddy_l500_50000

-- Define the conditions as variables
def students_registered_morning : ℕ := 25
def students_absent_morning : ℕ := 3
def students_registered_afternoon : ℕ := 24
def students_absent_afternoon : ℕ := 4

-- Prove the total number of students present over the two sessions
theorem total_students_mrs_mcgillicuddy : 
  students_registered_morning - students_absent_morning + students_registered_afternoon - students_absent_afternoon = 42 :=
by
  sorry

end total_students_mrs_mcgillicuddy_l500_50000


namespace line_through_midpoint_l500_50092

theorem line_through_midpoint (x y : ℝ) (P : x = 2 ∧ y = -1) :
  (∃ l : ℝ, ∀ t : ℝ, 
  (1 + 5 * Real.cos t = x) ∧ (5 * Real.sin t = y) →
  (x - y = 3)) :=
by
  sorry

end line_through_midpoint_l500_50092


namespace total_selling_price_correct_l500_50096

-- Defining the given conditions
def profit_per_meter : ℕ := 5
def cost_price_per_meter : ℕ := 100
def total_meters_sold : ℕ := 85

-- Using the conditions to define the total selling price
def total_selling_price := total_meters_sold * (cost_price_per_meter + profit_per_meter)

-- Stating the theorem without the proof
theorem total_selling_price_correct : total_selling_price = 8925 := by
  sorry

end total_selling_price_correct_l500_50096


namespace quotient_of_N_div_3_l500_50029

-- Define the number N
def N : ℕ := 7 * 12 + 4

-- Statement we need to prove
theorem quotient_of_N_div_3 : N / 3 = 29 :=
by
  sorry

end quotient_of_N_div_3_l500_50029


namespace largest_number_less_than_2_l500_50018

theorem largest_number_less_than_2 (a b c : ℝ) (h_a : a = 0.8) (h_b : b = 1/2) (h_c : c = 0.5) : 
  a < 2 ∧ b < 2 ∧ c < 2 ∧ (∀ x, (x = a ∨ x = b ∨ x = c) → x < 2) → 
  a = 0.8 ∧ 
  (a > b ∧ a > c) ∧ 
  (a < 2) :=
by sorry

end largest_number_less_than_2_l500_50018


namespace ten_years_less_than_average_age_l500_50041

theorem ten_years_less_than_average_age (L : ℕ) :
  (2 * L - 14) = 
    (2 * L - 4) - 10 :=
by {
  sorry
}

end ten_years_less_than_average_age_l500_50041


namespace time_no_traffic_is_4_hours_l500_50044

-- Definitions and conditions
def distance : ℕ := 200
def time_traffic : ℕ := 5

axiom traffic_speed_relation : ∃ (speed_traffic : ℕ), distance = speed_traffic * time_traffic
axiom speed_difference : ∀ (speed_traffic speed_no_traffic : ℕ), speed_no_traffic = speed_traffic + 10

-- Prove that the time when there's no traffic is 4 hours
theorem time_no_traffic_is_4_hours : ∀ (speed_traffic speed_no_traffic : ℕ), 
  distance = speed_no_traffic * (distance / speed_no_traffic) -> (distance / speed_no_traffic) = 4 :=
by
  intros speed_traffic speed_no_traffic h
  sorry

end time_no_traffic_is_4_hours_l500_50044
