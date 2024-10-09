import Mathlib

namespace expression_evaluation_l919_91965

variable (x y : ℤ)

theorem expression_evaluation (h₁ : x = -1) (h₂ : y = 1) : 
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 2 :=
by
  rw [h₁, h₂]
  have h₃ : (-1 + 1) * (-1 - 1) - (4 * (-1)^3 * 1 - 8 * (-1) * 1^3) / (2 * (-1) * 1) = (-2) - (-10 / -2) := by sorry
  have h₄ : (-2) - 5 = 2 := by sorry
  sorry

end expression_evaluation_l919_91965


namespace loss_percentage_l919_91902

theorem loss_percentage (CP SP_gain L : ℝ) 
  (h1 : CP = 1500)
  (h2 : SP_gain = CP + 0.05 * CP)
  (h3 : SP_gain = CP - (L/100) * CP + 225) : 
  L = 10 :=
by
  sorry

end loss_percentage_l919_91902


namespace parallel_lines_implies_a_eq_one_l919_91957

theorem parallel_lines_implies_a_eq_one 
(h_parallel: ∀ (a : ℝ), ∀ (x y : ℝ), (x + a * y = 2 * a + 2) → (a * x + y = a + 1) → -1/a = -a) :
  ∀ (a : ℝ), a = 1 := by
  sorry

end parallel_lines_implies_a_eq_one_l919_91957


namespace elisa_improvement_l919_91978

theorem elisa_improvement (cur_laps cur_minutes prev_laps prev_minutes : ℕ) 
  (h1 : cur_laps = 15) (h2 : cur_minutes = 30) 
  (h3 : prev_laps = 20) (h4 : prev_minutes = 50) : 
  ((prev_minutes / prev_laps : ℚ) - (cur_minutes / cur_laps : ℚ) = 0.5) :=
by
  sorry

end elisa_improvement_l919_91978


namespace triangle_bisector_length_l919_91950

theorem triangle_bisector_length (A B C : Type) [MetricSpace A] [MetricSpace B]
  [MetricSpace C] (angle_A angle_C : ℝ) (AC AB : ℝ) 
  (hAC : angle_A = 20) (hAngle_C : angle_C = 40) (hAC_minus_AB : AC - AB = 5) : ∃ BM : ℝ, BM = 5 :=
by
  sorry

end triangle_bisector_length_l919_91950


namespace percentage_of_copper_is_correct_l919_91993

-- Defining the conditions
def total_weight := 100.0
def weight_20_percent_alloy := 30.0
def weight_27_percent_alloy := total_weight - weight_20_percent_alloy

def percentage_20 := 0.20
def percentage_27 := 0.27

def copper_20 := percentage_20 * weight_20_percent_alloy
def copper_27 := percentage_27 * weight_27_percent_alloy
def total_copper := copper_20 + copper_27

-- The statement to be proved
def percentage_copper := (total_copper / total_weight) * 100

-- The theorem to prove
theorem percentage_of_copper_is_correct : percentage_copper = 24.9 := by sorry

end percentage_of_copper_is_correct_l919_91993


namespace mehki_age_l919_91927

variable (Mehki Jordyn Zrinka : ℕ)

axiom h1 : Mehki = Jordyn + 10
axiom h2 : Jordyn = 2 * Zrinka
axiom h3 : Zrinka = 6

theorem mehki_age : Mehki = 22 := by
  -- sorry to skip the proof
  sorry

end mehki_age_l919_91927


namespace find_original_number_l919_91990

theorem find_original_number (x : ℕ) :
  (43 * x - 34 * x = 1251) → x = 139 :=
by
  sorry

end find_original_number_l919_91990


namespace squares_difference_l919_91962

theorem squares_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 := by
  sorry

end squares_difference_l919_91962


namespace find_first_number_l919_91945

theorem find_first_number (N : ℤ) (k m : ℤ) (h1 : N = 170 * k + 10) (h2 : 875 = 170 * m + 25) : N = 860 :=
by
  sorry

end find_first_number_l919_91945


namespace factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l919_91954

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

end factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l919_91954


namespace g_range_l919_91921

noncomputable def g (x y z : ℝ) : ℝ :=
  (x ^ 2) / (x ^ 2 + y ^ 2) +
  (y ^ 2) / (y ^ 2 + z ^ 2) +
  (z ^ 2) / (z ^ 2 + x ^ 2)

theorem g_range (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 < g x y z ∧ g x y z < 2 :=
  sorry

end g_range_l919_91921


namespace marble_sharing_l919_91934

theorem marble_sharing 
  (total_marbles : ℕ) 
  (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 30) 
  (h2 : marbles_per_friend = 6) : 
  total_marbles / marbles_per_friend = 5 := 
by 
  sorry

end marble_sharing_l919_91934


namespace part1_part2_l919_91970

theorem part1 (a b h3 : ℝ) (C : ℝ) (h : 1 / h3 = 1 / a + 1 / b) : C ≤ 120 :=
sorry

theorem part2 (a b m3 : ℝ) (C : ℝ) (h : 1 / m3 = 1 / a + 1 / b) : C ≥ 120 :=
sorry

end part1_part2_l919_91970


namespace greening_investment_equation_l919_91996

theorem greening_investment_equation:
  ∃ (x : ℝ), 20 * (1 + x)^2 = 25 := 
sorry

end greening_investment_equation_l919_91996


namespace inequality_amgm_l919_91906

theorem inequality_amgm (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) := 
by 
  sorry

end inequality_amgm_l919_91906


namespace relationship_between_x_and_y_l919_91918

theorem relationship_between_x_and_y
  (x y : ℝ)
  (h1 : 2 * x - 3 * y > 6 * x)
  (h2 : 3 * x - 4 * y < 2 * y - x) :
  x < y ∧ x < 0 ∧ y < 0 :=
sorry

end relationship_between_x_and_y_l919_91918


namespace rosie_can_make_nine_pies_l919_91943

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l919_91943


namespace maximum_value_of_function_l919_91907

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - 2 * Real.sin x - 2

theorem maximum_value_of_function :
  ∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1 → f y ≤ 1 :=
by
  sorry

end maximum_value_of_function_l919_91907


namespace bacteria_colony_first_day_exceeds_100_l919_91946

theorem bacteria_colony_first_day_exceeds_100 :
  ∃ n : ℕ, 3 * 2^n > 100 ∧ (∀ m < n, 3 * 2^m ≤ 100) :=
sorry

end bacteria_colony_first_day_exceeds_100_l919_91946


namespace remaining_units_correct_l919_91973

-- Definitions based on conditions
def total_units : ℕ := 2000
def fraction_built_in_first_half : ℚ := 3/5
def additional_units_by_october : ℕ := 300

-- Calculate units built in the first half of the year
def units_built_in_first_half : ℚ := fraction_built_in_first_half * total_units

-- Remaining units after the first half of the year
def remaining_units_after_first_half : ℚ := total_units - units_built_in_first_half

-- Remaining units after building additional units by October
def remaining_units_to_be_built : ℚ := remaining_units_after_first_half - additional_units_by_october

-- Theorem statement: Prove remaining units to be built is 500
theorem remaining_units_correct : remaining_units_to_be_built = 500 := by
  sorry

end remaining_units_correct_l919_91973


namespace manager_salary_3700_l919_91903

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

end manager_salary_3700_l919_91903


namespace count_numbers_with_digit_2_from_200_to_499_l919_91901

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

end count_numbers_with_digit_2_from_200_to_499_l919_91901


namespace geom_seq_a12_value_l919_91972

-- Define the geometric sequence as a function from natural numbers to real numbers
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geom_seq_a12_value (a : ℕ → ℝ) 
  (H_geom : geom_seq a) 
  (H_7_9 : a 7 * a 9 = 4) 
  (H_4 : a 4 = 1) : 
  a 12 = 4 := 
by 
  sorry

end geom_seq_a12_value_l919_91972


namespace GreenValley_Absent_Percentage_l919_91925

theorem GreenValley_Absent_Percentage 
  (total_students boys girls absent_boys_frac absent_girls_frac : ℝ)
  (h1 : total_students = 120)
  (h2 : boys = 70)
  (h3 : girls = 50)
  (h4 : absent_boys_frac = 1 / 7)
  (h5 : absent_girls_frac = 1 / 5) :
  (absent_boys_frac * boys + absent_girls_frac * girls) / total_students * 100 = 16.67 := 
sorry

end GreenValley_Absent_Percentage_l919_91925


namespace m_power_of_prime_no_m_a_k_l919_91959

-- Part (i)
theorem m_power_of_prime (m : ℕ) (p : ℕ) (k : ℕ) (h1 : m ≥ 1) (h2 : Prime p) (h3 : m * (m + 1) = p^k) : m = 1 :=
by sorry

-- Part (ii)
theorem no_m_a_k (m a k : ℕ) (h1 : m ≥ 1) (h2 : a ≥ 1) (h3 : k ≥ 2) (h4 : m * (m + 1) = a^k) : False :=
by sorry

end m_power_of_prime_no_m_a_k_l919_91959


namespace find_b_l919_91926

theorem find_b (b : ℝ) (x y : ℝ) (h1 : 2 * x^2 + b * x = 12) (h2 : y = x + 5.5) (h3 : y^2 * x + y * x^2 + y * (b * x) = 12) :
  b = -5 :=
sorry

end find_b_l919_91926


namespace difference_in_roi_l919_91930

theorem difference_in_roi (E_investment : ℝ) (B_investment : ℝ) (E_rate : ℝ) (B_rate : ℝ) (years : ℕ) :
  E_investment = 300 → B_investment = 500 → E_rate = 0.15 → B_rate = 0.10 → years = 2 →
  (B_rate * B_investment * years) - (E_rate * E_investment * years) = 10 :=
by
  intros E_investment_eq B_investment_eq E_rate_eq B_rate_eq years_eq
  sorry

end difference_in_roi_l919_91930


namespace divisor_is_seven_l919_91931

theorem divisor_is_seven 
  (d x : ℤ)
  (h1 : x % d = 5)
  (h2 : 4 * x % d = 6) :
  d = 7 := 
sorry

end divisor_is_seven_l919_91931


namespace fraction_decomposition_l919_91989

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ -8/3 → (7 * x - 19) / (3 * x^2 + 5 * x - 8) = A / (x - 1) + B / (3 * x + 8)) →
  A = -12 / 11 ∧ B = 113 / 11 :=
by
  sorry

end fraction_decomposition_l919_91989


namespace find_a1_l919_91971

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 8 = 2 ∧ ∀ n, a (n + 1) = 1 / (1 - a n)

theorem find_a1 (a : ℕ → ℝ) (h : seq a) : a 1 = 1/2 := by
sorry

end find_a1_l919_91971


namespace solve_for_x_l919_91949

-- Definitions of δ and φ
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- The main proof statement
theorem solve_for_x :
  ∃ x : ℚ, delta (phi x) = 10 ∧ x = -31 / 36 :=
by
  sorry

end solve_for_x_l919_91949


namespace geometric_sequence_a4_range_l919_91933

theorem geometric_sequence_a4_range
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : 0 < a 1 ∧ a 1 < 1)
  (h2 : 1 < a 1 * q ∧ a 1 * q < 2)
  (h3 : 2 < a 1 * q^2 ∧ a 1 * q^2 < 3) :
  ∃ a4 : ℝ, a4 = a 1 * q^3 ∧ 2 * Real.sqrt 2 < a4 ∧ a4 < 9 := 
sorry

end geometric_sequence_a4_range_l919_91933


namespace value_by_which_number_is_multiplied_l919_91916

theorem value_by_which_number_is_multiplied (x : ℝ) : (5 / 6) * x = 10 ↔ x = 12 := by
  sorry

end value_by_which_number_is_multiplied_l919_91916


namespace samantha_coins_worth_l919_91909

-- Define the conditions and the final question with an expected answer.
theorem samantha_coins_worth (n d : ℕ) (h1 : n + d = 30)
  (h2 : 10 * n + 5 * d = 5 * n + 10 * d + 120) :
  (5 * n + 10 * d) = 165 := 
sorry

end samantha_coins_worth_l919_91909


namespace second_number_is_180_l919_91919

theorem second_number_is_180 
  (x : ℝ) 
  (first : ℝ := 2 * x) 
  (third : ℝ := (1/3) * first)
  (h : first + x + third = 660) : 
  x = 180 :=
sorry

end second_number_is_180_l919_91919


namespace ranking_emily_olivia_nicole_l919_91956

noncomputable def Emily_score : ℝ := sorry
noncomputable def Olivia_score : ℝ := sorry
noncomputable def Nicole_score : ℝ := sorry

theorem ranking_emily_olivia_nicole :
  (Emily_score > Olivia_score) ∧ (Emily_score > Nicole_score) → 
  (Emily_score > Olivia_score) ∧ (Olivia_score > Nicole_score) := 
by sorry

end ranking_emily_olivia_nicole_l919_91956


namespace committee_combinations_l919_91982

-- We use a broader import to ensure all necessary libraries are included.
-- Definitions and theorem

def club_member_count : ℕ := 20
def committee_member_count : ℕ := 3

theorem committee_combinations : 
  (Nat.choose club_member_count committee_member_count) = 1140 := by
sorry

end committee_combinations_l919_91982


namespace part_a_part_b_l919_91941

noncomputable def tsunami_area_center_face (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  180000 * Real.pi + 270000 * Real.sqrt 3

noncomputable def tsunami_area_mid_edge (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7

theorem part_a (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_center_face l v t = 180000 * Real.pi + 270000 * Real.sqrt 3 :=
by
  sorry

theorem part_b (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_mid_edge l v t = 720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7 :=
by
  sorry

end part_a_part_b_l919_91941


namespace arithmetic_sequence_general_formula_and_sum_max_l919_91975

theorem arithmetic_sequence_general_formula_and_sum_max :
  ∀ (a : ℕ → ℤ), 
  (a 7 = -8) → (a 17 = -28) → 
  (∀ n, a n = -2 * n + 6) ∧ 
  (∀ S : ℕ → ℤ, (∀ n, S n = -n^2 + 5 * n) → ∀ n, S n ≤ 6) :=
by
  sorry

end arithmetic_sequence_general_formula_and_sum_max_l919_91975


namespace number_of_x_intercepts_l919_91924

def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 3

theorem number_of_x_intercepts : (∃ y : ℝ, parabola y = 0) ∧ (∃! x : ℝ, parabola x = 0) :=
by
  sorry

end number_of_x_intercepts_l919_91924


namespace weight_of_lighter_boxes_l919_91929

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

end weight_of_lighter_boxes_l919_91929


namespace part1_part2_l919_91986

def f (x : ℝ) : ℝ := abs (x - 1)
def g (x : ℝ) : ℝ := abs (2 * x - 1)

theorem part1 (x : ℝ) :
  abs (x + 4) ≤ x * abs (2 * x - 1) ↔ x ≥ 2 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, abs ((x + 2) - 1) + abs (x - 1) + a = 0 → False) ↔ a ≤ -2 :=
sorry

end part1_part2_l919_91986


namespace gcd_of_12547_23791_l919_91904

theorem gcd_of_12547_23791 : Nat.gcd 12547 23791 = 1 :=
by
  sorry

end gcd_of_12547_23791_l919_91904


namespace sum_of_integers_remainder_l919_91974

-- Definitions of the integers and their properties
variables (a b c : ℕ)

-- Conditions
axiom h1 : a % 53 = 31
axiom h2 : b % 53 = 17
axiom h3 : c % 53 = 8
axiom h4 : a % 5 = 0

-- The proof goal
theorem sum_of_integers_remainder :
  (a + b + c) % 53 = 3 :=
by
  sorry -- Proof to be provided

end sum_of_integers_remainder_l919_91974


namespace coeffs_equal_implies_a_plus_b_eq_4_l919_91997

theorem coeffs_equal_implies_a_plus_b_eq_4 (a b : ℕ) (h_rel_prime : Nat.gcd a b = 1) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq_coeffs : (Nat.choose 2000 1998) * (a ^ 2) * (b ^ 1998) = (Nat.choose 2000 1997) * (a ^ 3) * (b ^ 1997)) :
  a + b = 4 := 
sorry

end coeffs_equal_implies_a_plus_b_eq_4_l919_91997


namespace sum_digits_18_to_21_l919_91998

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_18_to_21 :
  sum_of_digits 18 + sum_of_digits 19 + sum_of_digits 20 + sum_of_digits 21 = 24 :=
by
  sorry

end sum_digits_18_to_21_l919_91998


namespace problem_statement_l919_91944

variable (a b c d : ℝ)

theorem problem_statement :
  (a^2 - a + 1) * (b^2 - b + 1) * (c^2 - c + 1) * (d^2 - d + 1) ≥ (9 / 16) * (a - b) * (b - c) * (c - d) * (d - a) :=
sorry

end problem_statement_l919_91944


namespace correct_proposition_l919_91914

variable (a b : ℝ)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)
variable (a_gt_b : a > b)

theorem correct_proposition : 1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end correct_proposition_l919_91914


namespace one_add_i_cubed_eq_one_sub_i_l919_91920

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
sorry

end one_add_i_cubed_eq_one_sub_i_l919_91920


namespace Gargamel_bought_tires_l919_91955

def original_price_per_tire := 84
def sale_price_per_tire := 75
def total_savings := 36
def discount_per_tire := original_price_per_tire - sale_price_per_tire
def num_tires (total_savings : ℕ) (discount_per_tire : ℕ) := total_savings / discount_per_tire

theorem Gargamel_bought_tires :
  num_tires total_savings discount_per_tire = 4 :=
by
  sorry

end Gargamel_bought_tires_l919_91955


namespace absolute_value_expression_evaluation_l919_91994

theorem absolute_value_expression_evaluation : abs (-2) * (abs (-Real.sqrt 25) - abs (Real.sin (5 * Real.pi / 2))) = 8 := by
  sorry

end absolute_value_expression_evaluation_l919_91994


namespace percentage_received_certificates_l919_91977

theorem percentage_received_certificates (boys girls : ℕ) (pct_boys pct_girls : ℝ) :
    boys = 30 ∧ girls = 20 ∧ pct_boys = 0.1 ∧ pct_girls = 0.2 →
    ((pct_boys * boys + pct_girls * girls) / (boys + girls) * 100) = 14 := by
  sorry

end percentage_received_certificates_l919_91977


namespace binary_quadratic_lines_value_m_l919_91932

theorem binary_quadratic_lines_value_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + 2 * x * y + 8 * y^2 + 14 * y + m = 0) →
  m = 7 :=
sorry

end binary_quadratic_lines_value_m_l919_91932


namespace positive_rational_as_sum_of_cubes_l919_91979

theorem positive_rational_as_sum_of_cubes (q : ℚ) (h_q_pos : q > 0) : 
  ∃ (a b c d : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = ((a^3 + b^3) / (c^3 + d^3)) :=
sorry

end positive_rational_as_sum_of_cubes_l919_91979


namespace alpha_beta_roots_eq_l919_91928

theorem alpha_beta_roots_eq {α β : ℝ} (hα : α^2 - α - 2006 = 0) (hβ : β^2 - β - 2006 = 0) (h_sum : α + β = 1) : 
  α + β^2 = 2007 :=
by
  sorry

end alpha_beta_roots_eq_l919_91928


namespace solution_set_empty_l919_91911

-- Define the quadratic polynomial
def quadratic (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem that the solution set of the given inequality is empty
theorem solution_set_empty : ∀ x : ℝ, quadratic x < 0 → false :=
by
  intro x
  unfold quadratic
  sorry

end solution_set_empty_l919_91911


namespace compound_interest_time_l919_91905

theorem compound_interest_time 
  (P : ℝ) (r : ℝ) (A₁ : ℝ) (A₂ : ℝ) (t₁ t₂ : ℕ)
  (h1 : r = 0.10)
  (h2 : A₁ = P * (1 + r) ^ t₁)
  (h3 : A₂ = P * (1 + r) ^ t₂)
  (h4 : A₁ = 2420)
  (h5 : A₂ = 2662)
  (h6 : t₂ = t₁ + 3) :
  t₁ = 3 := 
sorry

end compound_interest_time_l919_91905


namespace sum_of_cubes_of_consecutive_even_integers_l919_91980

theorem sum_of_cubes_of_consecutive_even_integers (x : ℤ) (h : x^2 + (x+2)^2 + (x+4)^2 = 2960) :
  x^3 + (x + 2)^3 + (x + 4)^3 = 90117 :=
sorry

end sum_of_cubes_of_consecutive_even_integers_l919_91980


namespace find_y_value_l919_91968

theorem find_y_value : (12 ^ 3 * 6 ^ 4) / 432 = 5184 := by
  sorry

end find_y_value_l919_91968


namespace chi_square_hypothesis_test_l919_91976

-- Definitions based on the conditions
def males_like_sports := "Males like to participate in sports activities"
def females_dislike_sports := "Females do not like to participate in sports activities"
def activities_related_to_gender := "Liking to participate in sports activities is related to gender"
def activities_not_related_to_gender := "Liking to participate in sports activities is not related to gender"

-- Statement to prove that D is the correct null hypothesis
theorem chi_square_hypothesis_test :
  activities_not_related_to_gender = "H₀: Liking to participate in sports activities is not related to gender" :=
sorry

end chi_square_hypothesis_test_l919_91976


namespace right_triangle_condition_l919_91938

theorem right_triangle_condition (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : a^2 + b^2 = c^2 :=
by sorry

end right_triangle_condition_l919_91938


namespace geometric_sequence_a6_l919_91917

variable {α : Type} [LinearOrderedSemiring α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ a₁ q : α, ∀ n, a n = a₁ * q ^ n

theorem geometric_sequence_a6 
  (a : ℕ → α) 
  (h_seq : is_geometric_sequence a) 
  (h1 : a 2 + a 4 = 20) 
  (h2 : a 3 + a 5 = 40) : 
  a 6 = 64 :=
by
  sorry

end geometric_sequence_a6_l919_91917


namespace min_value_inequality_l919_91991

noncomputable def minValue : ℝ := 17 / 2

theorem min_value_inequality (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_cond : a + 2 * b = 1) :
  a^2 + 4 * b^2 + 1 / (a * b) = minValue := 
by
  sorry

end min_value_inequality_l919_91991


namespace batsman_average_l919_91923

theorem batsman_average (A : ℕ) (total_runs_before : ℕ) (new_score : ℕ) (increase : ℕ)
  (h1 : total_runs_before = 11 * A)
  (h2 : new_score = 70)
  (h3 : increase = 3)
  (h4 : 11 * A + new_score = 12 * (A + increase)) :
  (A + increase) = 37 :=
by
  -- skipping the proof with sorry
  sorry

end batsman_average_l919_91923


namespace remaining_dogs_eq_200_l919_91969

def initial_dogs : ℕ := 200
def additional_dogs : ℕ := 100
def first_adoption : ℕ := 40
def second_adoption : ℕ := 60

def total_dogs_after_adoption : ℕ :=
  initial_dogs + additional_dogs - first_adoption - second_adoption

theorem remaining_dogs_eq_200 : total_dogs_after_adoption = 200 :=
by
  -- Omitted the proof as requested
  sorry

end remaining_dogs_eq_200_l919_91969


namespace disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l919_91953

variable (p q : Prop)

theorem disjunction_false_implies_neg_p_true (hpq : ¬(p ∨ q)) : ¬p :=
by 
  sorry

theorem neg_p_true_does_not_imply_disjunction_false (hnp : ¬p) : ¬(¬(p ∨ q)) :=
by 
  sorry

end disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l919_91953


namespace Robin_needs_to_buy_more_bottles_l919_91985

/-- Robin wants to drink exactly nine bottles of water each day.
    She initially bought six hundred seventeen bottles.
    Prove that she will need to buy 4 more bottles on the last day
    to meet her goal of drinking exactly nine bottles each day. -/
theorem Robin_needs_to_buy_more_bottles :
  ∀ total_bottles bottles_per_day : ℕ, total_bottles = 617 → bottles_per_day = 9 → 
  ∃ extra_bottles : ℕ, (617 % 9) + extra_bottles = 9 ∧ extra_bottles = 4 :=
by
  sorry

end Robin_needs_to_buy_more_bottles_l919_91985


namespace average_discount_rate_l919_91995

theorem average_discount_rate
  (bag_marked_price : ℝ) (bag_sold_price : ℝ)
  (shoes_marked_price : ℝ) (shoes_sold_price : ℝ)
  (jacket_marked_price : ℝ) (jacket_sold_price : ℝ)
  (h_bag : bag_marked_price = 80) (h_bag_sold : bag_sold_price = 68)
  (h_shoes : shoes_marked_price = 120) (h_shoes_sold : shoes_sold_price = 96)
  (h_jacket : jacket_marked_price = 150) (h_jacket_sold : jacket_sold_price = 135) : 
  (15 : ℝ) =
  (((bag_marked_price - bag_sold_price) / bag_marked_price * 100) + 
   ((shoes_marked_price - shoes_sold_price) / shoes_marked_price * 100) + 
   ((jacket_marked_price - jacket_sold_price) / jacket_marked_price * 100)) / 3 :=
by {
  sorry
}

end average_discount_rate_l919_91995


namespace price_white_stamp_l919_91984

variable (price_per_white_stamp : ℝ)

theorem price_white_stamp (simon_red_stamps : ℕ)
                          (peter_white_stamps : ℕ)
                          (price_per_red_stamp : ℝ)
                          (money_difference : ℝ)
                          (h1 : simon_red_stamps = 30)
                          (h2 : peter_white_stamps = 80)
                          (h3 : price_per_red_stamp = 0.50)
                          (h4 : money_difference = 1) :
    money_difference = peter_white_stamps * price_per_white_stamp - simon_red_stamps * price_per_red_stamp →
    price_per_white_stamp = 1 / 5 :=
by
  intros
  sorry

end price_white_stamp_l919_91984


namespace train_passes_tree_in_16_seconds_l919_91952

noncomputable def time_to_pass_tree (length_train : ℕ) (speed_train_kmh : ℕ) : ℕ :=
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  length_train / speed_train_ms

theorem train_passes_tree_in_16_seconds :
  time_to_pass_tree 280 63 = 16 :=
  by
    sorry

end train_passes_tree_in_16_seconds_l919_91952


namespace orange_profit_44_percent_l919_91912

theorem orange_profit_44_percent :
  (∀ CP SP : ℚ, 0.99 * CP = 1 ∧ SP = CP / 16 → 1 / 11 = CP * (1 + 44 / 100)) :=
by
  sorry

end orange_profit_44_percent_l919_91912


namespace problem1_problem2_l919_91908

-- Problem 1
theorem problem1 (x y : ℝ) :
  2 * x^2 * y - 3 * x * y + 2 - x^2 * y + 3 * x * y = x^2 * y + 2 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) :
  9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n :=
by sorry

end problem1_problem2_l919_91908


namespace sum_of_x_and_y_l919_91922

-- Definitions of conditions
variables (x y : ℤ)
variable (h1 : x - y = 60)
variable (h2 : x = 37)

-- Statement of the problem to be proven
theorem sum_of_x_and_y : x + y = 14 :=
by
  sorry

end sum_of_x_and_y_l919_91922


namespace bucket_capacities_l919_91936

theorem bucket_capacities (a b c : ℕ) 
  (h1 : a + b + c = 1440) 
  (h2 : a + b / 5 = c) 
  (h3 : b + a / 3 = c) : 
  a = 480 ∧ b = 400 ∧ c = 560 := 
by 
  sorry

end bucket_capacities_l919_91936


namespace cost_of_first_shipment_1100_l919_91964

variables (S J : ℝ)
-- conditions
def second_shipment (S J : ℝ) := 5 * S + 15 * J = 550
def first_shipment (S J : ℝ) := 10 * S + 20 * J

-- goal
theorem cost_of_first_shipment_1100 (S J : ℝ) (h : second_shipment S J) : first_shipment S J = 1100 :=
sorry

end cost_of_first_shipment_1100_l919_91964


namespace solve_diophantine_l919_91900

theorem solve_diophantine (x y : ℕ) (h1 : 1990 * x - 1989 * y = 1991) : x = 11936 ∧ y = 11941 := by
  have h_pos_x : 0 < x := by sorry
  have h_pos_y : 0 < y := by sorry
  have h_x : 1990 * 11936 = 1990 * x := by sorry
  have h_y : 1989 * 11941 = 1989 * y := by sorry
  sorry

end solve_diophantine_l919_91900


namespace range_of_S_on_ellipse_l919_91935

theorem range_of_S_on_ellipse :
  ∀ (x y : ℝ),
    (x ^ 2 / 2 + y ^ 2 / 3 = 1) →
    -Real.sqrt 5 ≤ x + y ∧ x + y ≤ Real.sqrt 5 :=
by
  intro x y
  intro h
  sorry

end range_of_S_on_ellipse_l919_91935


namespace polynomial_evaluation_l919_91961

theorem polynomial_evaluation :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 :=
by
  sorry

end polynomial_evaluation_l919_91961


namespace det_A_l919_91948

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![2, -4, 5],
  ![0, 6, -2],
  ![3, -1, 2]
]

theorem det_A : A.det = -46 := by
  sorry

end det_A_l919_91948


namespace curve_intersects_x_axis_at_4_over_5_l919_91983

-- Define the function for the curve
noncomputable def curve (x : ℝ) : ℝ :=
  (3 * x - 1) * (Real.sqrt (9 * x ^ 2 - 6 * x + 5) + 1) +
  (2 * x - 3) * (Real.sqrt (4 * x ^ 2 - 12 * x + 13) + 1)

-- Prove that curve(x) = 0 when x = 4 / 5
theorem curve_intersects_x_axis_at_4_over_5 :
  curve (4 / 5) = 0 :=
by
  sorry

end curve_intersects_x_axis_at_4_over_5_l919_91983


namespace cos_540_eq_neg_1_l919_91942

theorem cos_540_eq_neg_1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end cos_540_eq_neg_1_l919_91942


namespace quadratic_eq_roots_minus5_and_7_l919_91940

theorem quadratic_eq_roots_minus5_and_7 : ∀ x : ℝ, (x + 5) * (x - 7) = 0 ↔ x = -5 ∨ x = 7 := by
  sorry

end quadratic_eq_roots_minus5_and_7_l919_91940


namespace find_fraction_l919_91910

variable (x y z : ℝ)

theorem find_fraction (h : (x - y) / (z - y) = -10) : (x - z) / (y - z) = 11 := 
by
  sorry

end find_fraction_l919_91910


namespace find_f_2011_l919_91937

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 2 then 2 * x^2
  else sorry  -- Placeholder, since f is only defined in (0, 2)

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x

theorem find_f_2011 : f 2011 = -2 :=
by
  -- Use properties of f to reduce and eventually find f(2011)
  sorry

end find_f_2011_l919_91937


namespace real_inequality_l919_91999

theorem real_inequality
  (a1 a2 a3 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (S : ℝ)
  (hS : S = a1 + a2 + a3)
  (h4 : ∀ i ∈ [a1, a2, a3], (i^2 / (i - 1) > S)) :
  (1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1) := 
by
  sorry

end real_inequality_l919_91999


namespace dart_probability_l919_91967

noncomputable def area_hexagon (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

noncomputable def area_circle (s : ℝ) : ℝ := Real.pi * s^2

noncomputable def probability (s : ℝ) : ℝ := 
  (area_circle s) / (area_hexagon s)

theorem dart_probability (s : ℝ) (hs : s > 0) :
  probability s = (2 * Real.pi) / (3 * Real.sqrt 3) :=
by
  sorry

end dart_probability_l919_91967


namespace sum_of_elements_in_M_l919_91981

theorem sum_of_elements_in_M (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0) :
  (∀ x : ℝ, x ∈ {x | x^2 - 2 * x + m = 0} → x = 1) ∧ m = 1 ∨
  (∃ x1 x2 : ℝ, x1 ∈ {x | x^2 - 2 * x + m = 0} ∧ x2 ∈ {x | x^2 - 2 * x + m = 0} ∧ x1 ≠ x2 ∧
   x1 + x2 = 2 ∧ m < 1) :=
sorry

end sum_of_elements_in_M_l919_91981


namespace max_mean_weight_BC_l919_91963

theorem max_mean_weight_BC
  (A_n B_n C_n : ℕ)
  (w_A w_B : ℕ)
  (mean_A mean_B mean_AB mean_AC : ℤ)
  (hA : mean_A = 30)
  (hB : mean_B = 55)
  (hAB : mean_AB = 35)
  (hAC : mean_AC = 32)
  (h1 : mean_A * A_n + mean_B * B_n = mean_AB * (A_n + B_n))
  (h2 : mean_A * A_n + mean_AC * C_n = mean_AC * (A_n + C_n)) :
  ∃ n : ℕ, n ≤ 62 ∧ (mean_B * B_n + w_A * C_n) / (B_n + C_n) = n := 
sorry

end max_mean_weight_BC_l919_91963


namespace prove_geomSeqSumFirst3_l919_91992

noncomputable def geomSeqSumFirst3 {a₁ a₆ : ℕ} (h₁ : a₁ = 1) (h₂ : a₆ = 32) : ℕ :=
  let r := 2 -- since r^5 = 32 which means r = 2
  let S3 := a₁ * (1 - r^3) / (1 - r)
  S3

theorem prove_geomSeqSumFirst3 : 
  geomSeqSumFirst3 (h₁ : 1 = 1) (h₂ : 32 = 32) = 7 := by
  sorry

end prove_geomSeqSumFirst3_l919_91992


namespace range_of_a_l919_91915

theorem range_of_a (a : ℝ) : (∃ (x : ℤ), x > 1 ∧ x ≤ a) → ∃ (x : ℤ), (x = 2 ∨ x = 3 ∨ x = 4) ∧ 4 ≤ a ∧ a < 5 :=
by
  sorry

end range_of_a_l919_91915


namespace hyperbola_asymptote_m_value_l919_91951

theorem hyperbola_asymptote_m_value
  (m : ℝ)
  (h1 : m > 0)
  (h2 : ∀ x y : ℝ, (5 * x - 2 * y = 0) → ((x^2 / 4) - (y^2 / m^2) = 1)) :
  m = 5 :=
sorry

end hyperbola_asymptote_m_value_l919_91951


namespace area_of_quadrilateral_l919_91913

theorem area_of_quadrilateral (d h1 h2 : ℝ) (hd : d = 20) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  (1 / 2) * d * (h1 + h2) = 150 :=
by
  rw [hd, hh1, hh2]
  norm_num

end area_of_quadrilateral_l919_91913


namespace value_of_3y_l919_91966

theorem value_of_3y (x y z : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h4 : z = 3) :
  3 * y = 12 :=
by
  sorry

end value_of_3y_l919_91966


namespace first_number_lcm_14_20_l919_91947

theorem first_number_lcm_14_20 (x : ℕ) (h : Nat.lcm x (Nat.lcm 14 20) = 140) : x = 1 := sorry

end first_number_lcm_14_20_l919_91947


namespace inequality_property_l919_91958

theorem inequality_property (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : (a / b) > (b / a) := 
sorry

end inequality_property_l919_91958


namespace shipCargoCalculation_l919_91988

def initialCargo : Int := 5973
def cargoLoadedInBahamas : Int := 8723
def totalCargo (initial : Int) (loaded : Int) : Int := initial + loaded

theorem shipCargoCalculation : totalCargo initialCargo cargoLoadedInBahamas = 14696 := by
  sorry

end shipCargoCalculation_l919_91988


namespace triangle_perimeter_l919_91939

-- Define the side lengths
def a : ℕ := 7
def b : ℕ := 10
def c : ℕ := 15

-- Define the perimeter
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Statement of the proof problem
theorem triangle_perimeter : perimeter 7 10 15 = 32 := by
  sorry

end triangle_perimeter_l919_91939


namespace fraction_power_simplification_l919_91987

theorem fraction_power_simplification:
  (81000/9000)^3 = 729 → (81000^3) / (9000^3) = 729 :=
by 
  intro h
  rw [<- h]
  sorry

end fraction_power_simplification_l919_91987


namespace nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l919_91960

theorem nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3
  (a b : ℤ)
  (h : 9 ∣ (a^2 + a * b + b^2)) :
  3 ∣ a ∧ 3 ∣ b :=
sorry

end nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l919_91960
