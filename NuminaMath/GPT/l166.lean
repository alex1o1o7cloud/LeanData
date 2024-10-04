import Mathlib

namespace arccos_neg_one_eq_pi_l166_166158

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l166_166158


namespace function_increasing_value_of_a_function_decreasing_value_of_a_l166_166493

-- Part 1: Prove that if \( f(x) = x^3 - ax - 1 \) is increasing on the interval \( (1, +\infty) \), then \( a \leq 3 \)
theorem function_increasing_value_of_a (a : ℝ) :
  (∀ x > 1, 3 * x^2 - a ≥ 0) → a ≤ 3 := by
  sorry

-- Part 2: Prove that if the decreasing interval of \( f(x) = x^3 - ax - 1 \) is \( (-1, 1) \), then \( a = 3 \)
theorem function_decreasing_value_of_a (a : ℝ) :
  (∀ x, -1 < x ∧ x < 1 → 3 * x^2 - a < 0) ∧ (3 * (-1)^2 - a = 0 ∧ 3 * (1)^2 - a = 0) → a = 3 := by
  sorry

end function_increasing_value_of_a_function_decreasing_value_of_a_l166_166493


namespace calc_g_f_neg_2_l166_166970

def f (x : ℝ) : ℝ := x^3 - 4 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 1

theorem calc_g_f_neg_2 : g (f (-2)) = 25 := by
  sorry

end calc_g_f_neg_2_l166_166970


namespace smallest_k_for_inequality_l166_166731

theorem smallest_k_for_inequality :
  ∃ k : ℕ, (∀ m : ℕ, m < k → 64^m ≤ 7) ∧ 64^k > 7 :=
by
  sorry

end smallest_k_for_inequality_l166_166731


namespace m_le_n_l166_166239

theorem m_le_n (k m n : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hn_pos : 0 < n) (h : m^2 + n = k^2 + k) : m ≤ n := 
sorry

end m_le_n_l166_166239


namespace simplify_expression_l166_166677

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l166_166677


namespace number_of_guest_cars_l166_166787

-- Definitions and conditions
def total_wheels : ℕ := 48
def mother_car_wheels : ℕ := 4
def father_jeep_wheels : ℕ := 4
def wheels_per_car : ℕ := 4

-- Theorem statement
theorem number_of_guest_cars (total_wheels mother_car_wheels father_jeep_wheels wheels_per_car : ℕ) : ℕ :=
  (total_wheels - (mother_car_wheels + father_jeep_wheels)) / wheels_per_car

-- Specific instance for the problem
example : number_of_guest_cars 48 4 4 4 = 10 := 
by
  sorry

end number_of_guest_cars_l166_166787


namespace intersection_M_N_l166_166628

def M : Set ℝ := {x | x < 1/2}
def N : Set ℝ := {y | y ≥ -4}

theorem intersection_M_N :
  (M ∩ N = {x | -4 ≤ x ∧ x < 1/2}) :=
sorry

end intersection_M_N_l166_166628


namespace number_of_houses_in_block_l166_166267

theorem number_of_houses_in_block (pieces_per_house pieces_per_block : ℕ) (h1 : pieces_per_house = 32) (h2 : pieces_per_block = 640) :
  pieces_per_block / pieces_per_house = 20 :=
by
  sorry

end number_of_houses_in_block_l166_166267


namespace not_divisible_by_1955_l166_166066

theorem not_divisible_by_1955 (n : ℤ) : ¬ ∃ k : ℤ, (n^2 + n + 1) = 1955 * k :=
by
  sorry

end not_divisible_by_1955_l166_166066


namespace exists_max_pile_division_l166_166657

theorem exists_max_pile_division (k : ℝ) (hk : k < 2) : 
  ∃ (N_k : ℕ), ∀ (A : Multiset ℝ) (m : ℝ), (∀ a ∈ A, a < 2 * m) → 
    ¬(∃ B : Multiset ℝ, B.card > N_k ∧ (∀ b ∈ B, b ∈ A ∧ b < 2 * m)) :=
sorry

end exists_max_pile_division_l166_166657


namespace problem1_problem2_l166_166228

def prop_p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem problem1 (a : ℝ) (h_a : a = 1) (h_pq : ∃ x, prop_p x a ∧ prop_q x) :
  ∃ x, 2 < x ∧ x < 3 :=
by sorry

theorem problem2 (h_qp : ∀ x (a : ℝ), prop_q x → prop_p x a) :
  ∃ a, 1 < a ∧ a ≤ 2 :=
by sorry

end problem1_problem2_l166_166228


namespace intersection_A_complementB_l166_166840

universe u

def R : Type := ℝ

def A (x : ℝ) : Prop := 0 < x ∧ x < 2

def B (x : ℝ) : Prop := x ≥ 1

def complement_B (x : ℝ) : Prop := x < 1

theorem intersection_A_complementB : 
  ∀ x : ℝ, (A x ∧ complement_B x) ↔ (0 < x ∧ x < 1) := 
by 
  sorry

end intersection_A_complementB_l166_166840


namespace least_possible_b_l166_166085

open Nat

/-- 
  Given conditions:
  a and b are consecutive Fibonacci numbers with a > b,
  and their sum is 100 degrees.
  We need to prove that the least possible value of b is 21 degrees.
-/
theorem least_possible_b (a b : ℕ) (h1 : fib a = fib (b + 1))
  (h2 : a > b) (h3 : a + b = 100) : b = 21 :=
sorry

end least_possible_b_l166_166085


namespace find_a_l166_166626

theorem find_a (a n : ℕ) (h1 : (2 : ℕ) ^ n = 32) (h2 : (a + 1) ^ n = 243) : a = 2 := by
  sorry

end find_a_l166_166626


namespace complete_square_l166_166983

theorem complete_square (a b c : ℕ) (h : 49 * x ^ 2 + 70 * x - 121 = 0) :
  a = 7 ∧ b = 5 ∧ c = 146 ∧ a + b + c = 158 :=
by sorry

end complete_square_l166_166983


namespace eq1_eq2_eq3_l166_166271

theorem eq1 (x : ℝ) : (x - 2)^2 - 5 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by 
  intro h
  sorry

theorem eq2 (x : ℝ) : x^2 + 4 * x = -3 → x = -1 ∨ x = -3 := 
by 
  intro h
  sorry
  
theorem eq3 (x : ℝ) : 4 * x * (x - 2) = x - 2 → x = 2 ∨ x = 1/4 := 
by 
  intro h
  sorry

end eq1_eq2_eq3_l166_166271


namespace diff_present_students_l166_166980

theorem diff_present_students (T A1 A2 A3 P1 P2 : ℕ) 
  (hT : T = 280)
  (h_total_absent : A1 + A2 + A3 = 240)
  (h_absent_ratio : A2 = 2 * A3)
  (h_absent_third_day : A3 = 280 / 7) 
  (hP1 : P1 = T - A1)
  (hP2 : P2 = T - A2) :
  P2 - P1 = 40 :=
sorry

end diff_present_students_l166_166980


namespace quadrilateral_area_l166_166645

theorem quadrilateral_area (a b x : ℝ)
  (h1: ∀ (y z : ℝ), y^2 + z^2 = a^2 ∧ (x + y)^2 + (x + z)^2 = b^2)
  (hx_perp: ∀ (p q : ℝ), x * q = 0 ∧ x * p = 0) :
  S = (1 / 4) * |b^2 - a^2| :=
by
  sorry

end quadrilateral_area_l166_166645


namespace product_of_first_three_terms_of_arithmetic_sequence_l166_166256

theorem product_of_first_three_terms_of_arithmetic_sequence {a d : ℕ} (ha : a + 6 * d = 20) (hd : d = 2) : a * (a + d) * (a + 2 * d) = 960 := by
  sorry

end product_of_first_three_terms_of_arithmetic_sequence_l166_166256


namespace time_for_B_to_complete_work_l166_166748

theorem time_for_B_to_complete_work 
  (A B C : ℝ)
  (h1 : A = 1 / 4) 
  (h2 : B + C = 1 / 3) 
  (h3 : A + C = 1 / 2) :
  1 / B = 12 :=
by
  -- Proof is omitted, as per instruction.
  sorry

end time_for_B_to_complete_work_l166_166748


namespace proposition_false_l166_166254

theorem proposition_false (a b : ℝ) (h : a + b > 0) : ¬ (a > 0 ∧ b > 0) := 
by {
  sorry -- this is a placeholder for the proof
}

end proposition_false_l166_166254


namespace candy_left_l166_166721

theorem candy_left (total_candy : ℕ) (eaten_per_person : ℕ) (number_of_people : ℕ)
  (h_total_candy : total_candy = 68)
  (h_eaten_per_person : eaten_per_person = 4)
  (h_number_of_people : number_of_people = 2) :
  total_candy - (eaten_per_person * number_of_people) = 60 :=
by
  sorry

end candy_left_l166_166721


namespace circle_representation_l166_166404

theorem circle_representation (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 + y^2 + x + 2*m*y + m = 0)) → m ≠ 1/2 :=
by
  sorry

end circle_representation_l166_166404


namespace num_factors_of_1320_l166_166918

theorem num_factors_of_1320 : ∃ n : ℕ, (n = 24) ∧ (∃ a b c d : ℕ, 1320 = 2^a * 3^b * 5^c * 11^d ∧ (a + 1) * (b + 1) * (c + 1) * (d + 1) = 24) :=
by
  sorry

end num_factors_of_1320_l166_166918


namespace min_value_of_m_squared_plus_n_squared_l166_166959

theorem min_value_of_m_squared_plus_n_squared (m n : ℝ) 
  (h : 4 * m - 3 * n - 5 * Real.sqrt 2 = 0) : m^2 + n^2 = 2 :=
sorry

end min_value_of_m_squared_plus_n_squared_l166_166959


namespace average_of_three_l166_166397

theorem average_of_three {a b c d e : ℚ}
    (h1 : (a + b + c + d + e) / 5 = 12)
    (h2 : (d + e) / 2 = 24) :
    (a + b + c) / 3 = 4 := by
  sorry

end average_of_three_l166_166397


namespace expected_digits_icosahedral_die_l166_166788

theorem expected_digits_icosahedral_die :
  let faces := (1 : Finset ℕ).filter (λ n, n ≤ 20) in
  let one_digit_faces := faces.filter (λ n, n < 10) in
  let two_digit_faces := faces.filter (λ n, n ≥ 10) in
  let probability_one_digit := (one_digit_faces.card : ℚ) / faces.card in
  let probability_two_digit := (two_digit_faces.card : ℚ) / faces.card in
  let expected_digits := (probability_one_digit * 1) + (probability_two_digit * 2) in
  expected_digits = 31 / 20 := sorry

end expected_digits_icosahedral_die_l166_166788


namespace find_k_values_l166_166718

theorem find_k_values (k : ℝ) : 
  (∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) → 
  (k = 0 ∨ k = 1 ∨ k = 2) ∧
  (k = 0 ∨ k = 1 ∨ k = 2 → ∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) :=
by
  sorry

end find_k_values_l166_166718


namespace find_two_digit_numbers_l166_166715

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem find_two_digit_numbers :
  ∀ (A : ℕ), (10 ≤ A ∧ A ≤ 99) →
    (sum_of_digits A)^2 = sum_of_digits (A^2) →
    (A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31 ∨ A = 50) :=
by sorry

end find_two_digit_numbers_l166_166715


namespace volume_of_released_gas_l166_166004

def mol_co2 : ℝ := 2.4
def molar_volume : ℝ := 22.4

theorem volume_of_released_gas : mol_co2 * molar_volume = 53.76 := by
  sorry -- proof to be filled in

end volume_of_released_gas_l166_166004


namespace set_operation_equivalence_l166_166535

variable {U : Type} -- U is the universal set
variables {X Y Z : Set U} -- X, Y, and Z are subsets of the universal set U

def star (A B : Set U) : Set U := A ∩ B  -- Define the operation "∗" as intersection

theorem set_operation_equivalence :
  star (star X Y) Z = (X ∩ Y) ∩ Z :=  -- Formulate the problem as a theorem to prove
by
  sorry  -- Proof is omitted

end set_operation_equivalence_l166_166535


namespace work_completion_time_l166_166569

/-- q can complete the work in 9 days, r can complete the work in 12 days, they work together
for 3 days, and p completes the remaining work in 10.000000000000002 days. Prove that
p alone can complete the work in approximately 24 days. -/
theorem work_completion_time (W : ℝ) (q : ℝ) (r : ℝ) (p : ℝ) :
  q = 9 → r = 12 → (p * 10.000000000000002 = (5 / 12) * W) →
  p = 24.000000000000004 :=
by 
  intros hq hr hp
  sorry

end work_completion_time_l166_166569


namespace total_fencing_cost_l166_166409

-- Conditions
def length : ℝ := 55
def cost_per_meter : ℝ := 26.50

-- We derive breadth from the given conditions
def breadth : ℝ := length - 10

-- Calculate the perimeter of the rectangular plot
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost of fencing the plot
def total_cost : ℝ := cost_per_meter * perimeter

-- The theorem to prove that total cost is equal to 5300
theorem total_fencing_cost : total_cost = 5300 := by
  -- Calculation goes here
  sorry

end total_fencing_cost_l166_166409


namespace best_ketchup_deal_l166_166779

def cost_per_ounce (price : ℝ) (ounces : ℝ) : ℝ := price / ounces

theorem best_ketchup_deal :
  let price_10oz := 1 in
  let ounces_10oz := 10 in
  let price_16oz := 2 in
  let ounces_16oz := 16 in
  let price_25oz := 2.5 in
  let ounces_25oz := 25 in
  let price_50oz := 5 in
  let ounces_50oz := 50 in
  let price_200oz := 10 in
  let ounces_200oz := 200 in
  let money := 10 in
  (∀ p o, cost_per_ounce p o ≥ cost_per_ounce price_200oz ounces_200oz) ∧ money = price_200oz :=
1
by
  sorry

end best_ketchup_deal_l166_166779


namespace sarahs_packages_l166_166081

def num_cupcakes_before : ℕ := 60
def num_cupcakes_ate : ℕ := 22
def cupcakes_per_package : ℕ := 10

theorem sarahs_packages : (num_cupcakes_before - num_cupcakes_ate) / cupcakes_per_package = 3 :=
by
  sorry

end sarahs_packages_l166_166081


namespace exam_items_count_l166_166511

theorem exam_items_count (x : ℝ) (hLiza : Liza_correct = 0.9 * x) (hRoseCorrect : Rose_correct = 0.9 * x + 2) (hRoseTotal : Rose_total = x) (hRoseIncorrect : Rose_incorrect = x - (0.9 * x + 2) ):
    Liza_correct + Rose_incorrect = Rose_total :=
by
    sorry

end exam_items_count_l166_166511


namespace product_first_three_terms_arithmetic_seq_l166_166260

theorem product_first_three_terms_arithmetic_seq :
  ∀ (a₇ d : ℤ), 
  a₇ = 20 → d = 2 → 
  let a₁ := a₇ - 6 * d in
  let a₂ := a₁ + d in
  let a₃ := a₂ + d in
  a₁ * a₂ * a₃ = 960 := 
by
  intros a₇ d a₇_20 d_2
  let a₁ := a₇ - 6 * d
  let a₂ := a₁ + d
  let a₃ := a₂ + d
  sorry

end product_first_three_terms_arithmetic_seq_l166_166260


namespace arccos_neg_one_eq_pi_l166_166161

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
  sorry

end arccos_neg_one_eq_pi_l166_166161


namespace nurses_count_l166_166266

theorem nurses_count (total_medical_staff : ℕ) (ratio_doctors : ℕ) (ratio_nurses : ℕ) 
  (total_ratio_parts : ℕ) (h1 : total_medical_staff = 200) 
  (h2 : ratio_doctors = 4) (h3 : ratio_nurses = 6) (h4 : total_ratio_parts = ratio_doctors + ratio_nurses) :
  (ratio_nurses * total_medical_staff) / total_ratio_parts = 120 :=
by
  sorry

end nurses_count_l166_166266


namespace anchuria_certification_prob_higher_in_2012_l166_166355

noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

theorem anchuria_certification_prob_higher_in_2012
    (p : ℝ) (h : p = 0.25) :
  let prob_2011 := 1 - (binomial 20 0 p + binomial 20 1 p + binomial 20 2 p)
  let prob_2012 := 1 - (binomial 40 0 p + binomial 40 1 p + binomial 40 2 p + binomial 40 3 p +
                        binomial 40 4 p + binomial 40 5 p)
  prob_2012 > prob_2011 :=
by
  intros
  have h_prob_2011 : prob_2011 = 1 - ((binomial 20 0 p) + (binomial 20 1 p) + (binomial 20 2 p)), sorry
  have h_prob_2012 : prob_2012 = 1 - ((binomial 40 0 p) + (binomial 40 1 p) + (binomial 40 2 p) +
                                      (binomial 40 3 p) + (binomial 40 4 p) + (binomial 40 5 p)), sorry
  have pf_correct_prob_2011 : prob_2011 = 0.909, sorry
  have pf_correct_prob_2012 : prob_2012 = 0.957, sorry
  have pf_final : 0.957 > 0.909, from by norm_num
  exact pf_final

end anchuria_certification_prob_higher_in_2012_l166_166355


namespace paul_money_duration_l166_166115

theorem paul_money_duration
  (mow_earnings : ℕ)
  (weed_earnings : ℕ)
  (weekly_expenses : ℕ)
  (earnings_mow : mow_earnings = 3)
  (earnings_weed : weed_earnings = 3)
  (expenses : weekly_expenses = 3) :
  (mow_earnings + weed_earnings) / weekly_expenses = 2 := 
by
  sorry

end paul_money_duration_l166_166115


namespace smallest_Y_l166_166068

noncomputable def S : ℕ := 111111111000

theorem smallest_Y : ∃ Y : ℕ, Y = S / 18 ∧ Y = 6172839500 := by
  use 6172839500
  split
  · calc
      S / 18 = 111111111000 / 18 := by sorry
    _ = 6172839500 := by sorry
  · exact rfl

end smallest_Y_l166_166068


namespace problem_equivalent_proof_l166_166617

noncomputable def sqrt (x : ℝ) : ℝ := x.sqrt -- Define the sqrt function for real numbers

theorem problem_equivalent_proof (x : ℝ) :
  sqrt ((3 + sqrt 8) ^ x) + sqrt ((3 - sqrt 8) ^ x) = 6 ↔ x = 2 ∨ x = -2 :=
begin
  sorry -- This is where the proof would be, omitted as instructed
end

end problem_equivalent_proof_l166_166617


namespace minimum_value_expression_l166_166052

theorem minimum_value_expression (a b : ℝ) (h : a * b > 0) : 
  ∃ m : ℝ, (∀ x y : ℝ, x * y > 0 → (4 * y / x + (x - 2 * y) / y) ≥ m) ∧ m = 2 :=
by
  sorry

end minimum_value_expression_l166_166052


namespace simplify_fraction_l166_166688

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l166_166688


namespace melissa_work_hours_l166_166846

variable (f : ℝ) (f_d : ℝ) (h_d : ℝ)

theorem melissa_work_hours (hf : f = 56) (hfd : f_d = 4) (hhd : h_d = 3) : 
  (f / f_d) * h_d = 42 := by
  sorry

end melissa_work_hours_l166_166846


namespace teresa_age_when_michiko_born_l166_166867

theorem teresa_age_when_michiko_born (teresa_current_age morio_current_age morio_age_when_michiko_born : ℕ) 
  (h1 : teresa_current_age = 59) 
  (h2 : morio_current_age = 71) 
  (h3 : morio_age_when_michiko_born = 38) : 
  teresa_current_age - (morio_current_age - morio_age_when_michiko_born) = 26 := 
by 
  sorry

end teresa_age_when_michiko_born_l166_166867


namespace exists_a_perfect_power_l166_166176

def is_perfect_power (n : ℕ) : Prop :=
  ∃ b k : ℕ, b > 0 ∧ k ≥ 2 ∧ n = b^k

theorem exists_a_perfect_power :
  ∃ a > 0, ∀ n, 2015 ≤ n ∧ n ≤ 2558 → is_perfect_power (n * a) :=
sorry

end exists_a_perfect_power_l166_166176


namespace distance_point_line_l166_166485

theorem distance_point_line (m : ℝ) : 
  abs (m + 1) = 2 ↔ (m = 1 ∨ m = -3) := by
  sorry

end distance_point_line_l166_166485


namespace midpoint_of_line_segment_l166_166064

theorem midpoint_of_line_segment :
  let z1 := Complex.mk (-7) 5
  let z2 := Complex.mk 5 (-3)
  (z1 + z2) / 2 = Complex.mk (-1) 1 := by sorry

end midpoint_of_line_segment_l166_166064


namespace technician_percent_round_trip_l166_166453

noncomputable def round_trip_percentage_completed (D : ℝ) : ℝ :=
  let total_round_trip := 2 * D
  let distance_completed := D + 0.10 * D
  (distance_completed / total_round_trip) * 100

theorem technician_percent_round_trip (D : ℝ) (h : D > 0) : 
  round_trip_percentage_completed D = 55 := 
by 
  sorry

end technician_percent_round_trip_l166_166453


namespace scale_of_map_l166_166303

theorem scale_of_map 
  (map_distance : ℝ)
  (travel_time : ℝ)
  (average_speed : ℝ)
  (actual_distance : ℝ)
  (scale : ℝ)
  (h1 : map_distance = 5)
  (h2 : travel_time = 6.5)
  (h3 : average_speed = 60)
  (h4 : actual_distance = average_speed * travel_time)
  (h5 : scale = map_distance / actual_distance) :
  scale = 0.01282 :=
by
  sorry

end scale_of_map_l166_166303


namespace bar_charts_as_line_charts_l166_166150

-- Given that line charts help to visualize trends of increase and decrease
axiom trends_visualization (L : Type) : Prop

-- Bar charts can be drawn as line charts, which helps in visualizing trends
theorem bar_charts_as_line_charts (L B : Type) (h : trends_visualization L) : trends_visualization B := sorry

end bar_charts_as_line_charts_l166_166150


namespace second_discount_percentage_l166_166143

def normal_price : ℝ := 49.99
def first_discount : ℝ := 0.10
def final_price : ℝ := 36.0

theorem second_discount_percentage : 
  ∃ p : ℝ, (((normal_price - (first_discount * normal_price)) - final_price) / (normal_price - (first_discount * normal_price))) * 100 = p ∧ p = 20 :=
by
  sorry

end second_discount_percentage_l166_166143


namespace annual_interest_rate_l166_166394

/-- Suppose you invested $10000, part at a certain annual interest rate and the rest at 9% annual interest.
After one year, you received $684 in interest. You invested $7200 at this rate and the rest at 9%.
What is the annual interest rate of the first investment? -/
theorem annual_interest_rate (r : ℝ) 
  (h : 7200 * r + 2800 * 0.09 = 684) : r = 0.06 :=
by
  sorry

end annual_interest_rate_l166_166394


namespace find_integer_l166_166927

theorem find_integer (n : ℤ) (h1 : n ≥ 50) (h2 : n ≤ 100) (h3 : n % 7 = 0) (h4 : n % 9 = 3) (h5 : n % 6 = 3) : n = 84 := 
by 
  sorry

end find_integer_l166_166927


namespace directrix_of_parabola_l166_166181

theorem directrix_of_parabola : 
  let y := 3 * x^2 - 6 * x + 1
  y = -25 / 12 :=
sorry

end directrix_of_parabola_l166_166181


namespace lisa_flight_time_l166_166841

theorem lisa_flight_time :
  ∀ (d s : ℕ), (d = 256) → (s = 32) → ((d / s) = 8) :=
by
  intros d s h_d h_s
  sorry

end lisa_flight_time_l166_166841


namespace largest_B_div_by_4_l166_166651

-- Given conditions
def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- The seven-digit integer is 4B6792X
def number (B X : ℕ) : ℕ := 4000000 + B * 100000 + 60000 + 7000 + 900 + 20 + X

-- Problem statement: Prove that the largest digit B so that the seven-digit integer 4B6792X is divisible by 4
theorem largest_B_div_by_4 
(B X : ℕ) 
(hX : is_digit X)
(div_4 : divisible_by_4 (number B X)) : 
B = 9 := sorry

end largest_B_div_by_4_l166_166651


namespace percentage_reduction_l166_166911

theorem percentage_reduction (P : ℝ) (h1 : 700 / P + 3 = 700 / 70) : 
  ((P - 70) / P) * 100 = 30 :=
by
  sorry

end percentage_reduction_l166_166911


namespace hyperbola_through_point_has_asymptotes_l166_166319

-- Definitions based on condition (1)
def hyperbola_asymptotes (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Definition of the problem
def hyperbola_eqn (x y : ℝ) : Prop := (x^2 / 5) - (y^2 / 20) = 1

-- Main statement including all conditions and proving the correct answer
theorem hyperbola_through_point_has_asymptotes :
  ∀ x y : ℝ, hyperbola_eqn x y ↔ (hyperbola_asymptotes x y ∨ (x, y) = (-3, 4)) :=
by
  -- The proof part is skipped with sorry
  sorry

end hyperbola_through_point_has_asymptotes_l166_166319


namespace at_least_one_less_than_two_l166_166790

theorem at_least_one_less_than_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : 2 < a + b) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := 
by
  sorry

end at_least_one_less_than_two_l166_166790


namespace pentagon_diagonal_l166_166413

theorem pentagon_diagonal (a d : ℝ) (h : d^2 = a^2 + a * d) : 
  d = a * (Real.sqrt 5 + 1) / 2 :=
sorry

end pentagon_diagonal_l166_166413


namespace arrangements_three_events_l166_166007

theorem arrangements_three_events (volunteers : ℕ) (events : ℕ) (h_vol : volunteers = 5) (h_events : events = 3) : 
  ∃ n : ℕ, n = (events^volunteers - events * 2^volunteers + events * 1^volunteers) ∧ n = 150 := 
by
  sorry

end arrangements_three_events_l166_166007


namespace reciprocal_of_neg_three_l166_166707

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l166_166707


namespace directrix_of_parabola_l166_166180

theorem directrix_of_parabola : 
  let y := 3 * x^2 - 6 * x + 1
  y = -25 / 12 :=
sorry

end directrix_of_parabola_l166_166180


namespace geo_seq_decreasing_l166_166196

variables (a_1 q : ℝ) (a : ℕ → ℝ)
-- Define the geometric sequence
def geo_seq (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q ^ n

-- The problem statement as a Lean theorem
theorem geo_seq_decreasing (h1 : a_1 * (q - 1) < 0) (h2 : q > 0) :
  ∀ n : ℕ, geo_seq a_1 q (n + 1) < geo_seq a_1 q n :=
by
  sorry

end geo_seq_decreasing_l166_166196


namespace no_valid_m_l166_166932

theorem no_valid_m
  (m : ℕ)
  (hm : m > 0)
  (h1 : ∃ k1 : ℕ, k1 > 0 ∧ 1806 = k1 * (m^2 - 2))
  (h2 : ∃ k2 : ℕ, k2 > 0 ∧ 1806 = k2 * (m^2 + 2)) :
  false :=
sorry

end no_valid_m_l166_166932


namespace compute_fraction_l166_166784

theorem compute_fraction : (1922^2 - 1913^2) / (1930^2 - 1905^2) = (9 : ℚ) / 25 := by
  sorry

end compute_fraction_l166_166784


namespace power_function_value_l166_166034

/-- Given a power function passing through a certain point, find the value at a specific point -/
theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h : f x = x ^ α) 
  (h_passes : f (1/4) = 4) : f 2 = 1/2 :=
sorry

end power_function_value_l166_166034


namespace coin_probability_not_unique_l166_166018

variables (p : ℝ) (w : ℝ)
def binomial_prob := 10 * p^3 * (1 - p)^2

theorem coin_probability_not_unique (h : binomial_prob p = 144 / 625) : 
  ∃ p1 p2, p1 ≠ p2 ∧ binomial_prob p1 = 144 / 625 ∧ binomial_prob p2 = 144 / 625 :=
by 
  sorry

end coin_probability_not_unique_l166_166018


namespace stratified_sampling_third_grade_students_l166_166902

variable (total_students : ℕ) (second_year_female_probability : ℚ) (sample_size : ℕ)

theorem stratified_sampling_third_grade_students
  (h_total : total_students = 2000)
  (h_probability : second_year_female_probability = 0.19)
  (h_sample_size : sample_size = 64) :
  let sampling_fraction := 64 / 2000
  let third_grade_students := 2000 * sampling_fraction
  third_grade_students = 16 :=
by
  -- the proof would go here, but we're skipping it per instructions
  sorry

end stratified_sampling_third_grade_students_l166_166902


namespace sqrt_expression_identity_l166_166529

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 17 - 4

theorem sqrt_expression_identity : Real.sqrt ((-a)^3 + (b + 4)^2) = 4 :=
by
  -- Prove the statement

  sorry

end sqrt_expression_identity_l166_166529


namespace find_number_l166_166890

theorem find_number : ∃ n : ℕ, ∃ q : ℕ, ∃ r : ℕ, q = 6 ∧ r = 4 ∧ n = 9 * q + r ∧ n = 58 :=
by
  sorry

end find_number_l166_166890


namespace directrix_of_parabola_l166_166183

theorem directrix_of_parabola (x y : ℝ) : y = 3 * x^2 - 6 * x + 1 → y = -25 / 12 :=
sorry

end directrix_of_parabola_l166_166183


namespace max_value_of_expression_l166_166225

theorem max_value_of_expression {a x1 x2 : ℝ}
  (h1 : x1^2 + a * x1 + a = 2)
  (h2 : x2^2 + a * x2 + a = 2)
  (h1_ne_x2 : x1 ≠ x2) :
  ∃ a : ℝ, (x1 - 2 * x2) * (x2 - 2 * x1) = -63 / 8 :=
by
  sorry

end max_value_of_expression_l166_166225


namespace average_earnings_per_minute_l166_166547

theorem average_earnings_per_minute 
  (laps : ℕ) (meters_per_lap : ℕ) (dollars_per_100_meters : ℝ) (total_minutes : ℕ) (total_laps : ℕ)
  (h_laps : total_laps = 24)
  (h_meters_per_lap : meters_per_lap = 100)
  (h_dollars_per_100_meters : dollars_per_100_meters = 3.5)
  (h_total_minutes : total_minutes = 12)
  : (total_laps * meters_per_lap / 100 * dollars_per_100_meters / total_minutes) = 7 := 
by
  sorry

end average_earnings_per_minute_l166_166547


namespace pairs_satisfaction_l166_166175

-- Definitions for the conditions given
def condition1 (x y : ℝ) : Prop := y = (x + 2)^2
def condition2 (x y : ℝ) : Prop := x * y + 2 * y = 2

-- The statement that we need to prove
theorem pairs_satisfaction : 
  (∃ x y : ℝ, condition1 x y ∧ condition2 x y) ∧ 
  (∃ x1 x2 : ℂ, x^2 + -2*x + 1 = 0 ∧ ¬∃ (y : ℝ), y = (x1 + 2)^2 ∨ y = (x2 + 2)^2) :=
by
  sorry

end pairs_satisfaction_l166_166175


namespace negation_proposition_l166_166871

theorem negation_proposition :
  (¬ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 < 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 ≥ 0) := 
by
  sorry

end negation_proposition_l166_166871


namespace proof_problem_l166_166561

theorem proof_problem (a b : ℝ) (n : ℕ) 
  (P1 P2 : ℝ × ℝ)
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_n_gt_1 : n > 1)
  (h_P1_on_curve : P1.1 ^ n = a * P1.2 ^ n + b)
  (h_P2_on_curve : P2.1 ^ n = a * P2.2 ^ n + b)
  (h_y1_lt_y2 : P1.2 < P2.2)
  (A : ℝ) (h_A : A = (1/2) * |P1.1 * P2.2 - P2.1 * P1.2|) :
  b * P2.2 > 2 * n * P1.2 ^ (n - 1) * a ^ (1 - (1 / n)) * A :=
sorry

end proof_problem_l166_166561


namespace gcd_of_B_is_2_l166_166972

-- Definition of the set B based on the given condition.
def B : Set ℕ := {n | ∃ x, n = (x - 1) + x + (x + 1) + (x + 2)}

-- The core statement to prove, wrapped in a theorem.
theorem gcd_of_B_is_2 : gcd_set B = 2 :=
by
  sorry

end gcd_of_B_is_2_l166_166972


namespace f_neg2_range_l166_166497

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x

theorem f_neg2_range (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2) (h2 : 2 ≤ f (1) ∧ f (1) ≤ 4) :
  ∀ k, f (-2) = k → 5 ≤ k ∧ k ≤ 10 :=
  sorry

end f_neg2_range_l166_166497


namespace parabola_focus_directrix_l166_166293

noncomputable def parabola_distance_property (p : ℝ) (hp : 0 < p) : Prop :=
  let focus := (2 * p, 0)
  let directrix := -2 * p
  let distance := 4 * p
  p = distance / 4

-- Theorem: Given a parabola with equation y^2 = 8px (p > 0), p represents 1/4 of the distance from the focus to the directrix.
theorem parabola_focus_directrix (p : ℝ) (hp : 0 < p) : parabola_distance_property p hp :=
by
  sorry

end parabola_focus_directrix_l166_166293


namespace gcd_factorial_l166_166888

theorem gcd_factorial (n m l : ℕ) (h1 : n = 7) (h2 : m = 10) (h3 : l = 4): 
  Nat.gcd (Nat.factorial n) (Nat.factorial m / Nat.factorial l) = 2520 :=
by
  sorry

end gcd_factorial_l166_166888


namespace average_income_correct_l166_166563

-- Define the incomes for each day
def income_day_1 : ℕ := 300
def income_day_2 : ℕ := 150
def income_day_3 : ℕ := 750
def income_day_4 : ℕ := 400
def income_day_5 : ℕ := 500

-- Define the number of days
def number_of_days : ℕ := 5

-- Define the total income
def total_income : ℕ := income_day_1 + income_day_2 + income_day_3 + income_day_4 + income_day_5

-- Define the average income
def average_income : ℕ := total_income / number_of_days

-- State that the average income is 420
theorem average_income_correct :
  average_income = 420 := by
  sorry

end average_income_correct_l166_166563


namespace melissa_work_hours_l166_166848

theorem melissa_work_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (hours_per_dress : ℕ) (total_num_dresses : ℕ) (total_hours : ℕ) 
  (h1 : total_fabric = 56) (h2 : fabric_per_dress = 4) (h3 : hours_per_dress = 3) : 
  total_hours = (total_fabric / fabric_per_dress) * hours_per_dress := by
  sorry

end melissa_work_hours_l166_166848


namespace survey_respondents_l166_166388

theorem survey_respondents (X Y : ℕ) (hX : X = 150) (ratio : X = 5 * Y) : X + Y = 180 :=
by
  sorry

end survey_respondents_l166_166388


namespace anchurian_certificate_probability_l166_166364

open Probability

-- The probability of guessing correctly on a single question
def p : ℝ := 0.25

-- The probability of guessing incorrectly on a single question
def q : ℝ := 1.0 - p

-- Binomial Probability Mass Function
noncomputable def binomial_pmf (n : ℕ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

-- Passing probability in 2011
noncomputable def pass_prob_2011 : ℝ :=
  1 - (binomial_pmf 20 0 + binomial_pmf 20 1 + binomial_pmf 20 2)

-- Passing probability in 2012
noncomputable def pass_prob_2012 : ℝ :=
  1 - (binomial_pmf 40 0 + binomial_pmf 40 1 + binomial_pmf 40 2 + binomial_pmf 40 3 + binomial_pmf 40 4 + binomial_pmf 40 5)

theorem anchurian_certificate_probability :
  pass_prob_2012 > pass_prob_2011 :=
sorry

end anchurian_certificate_probability_l166_166364


namespace molecular_weight_of_7_moles_of_NH4_2SO4_l166_166919

theorem molecular_weight_of_7_moles_of_NH4_2SO4 :
  let N_weight := 14.01
  let H_weight := 1.01
  let S_weight := 32.07
  let O_weight := 16.00
  let N_atoms := 2
  let H_atoms := 8
  let S_atoms := 1
  let O_atoms := 4
  let moles := 7
  let molecular_weight := (N_weight * N_atoms) + (H_weight * H_atoms) + (S_weight * S_atoms) + (O_weight * O_atoms)
  let total_weight := molecular_weight * moles
  total_weight = 924.19 :=
by
  sorry

end molecular_weight_of_7_moles_of_NH4_2SO4_l166_166919


namespace decreasing_interval_l166_166943

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 15 * x^4 - 15 * x^2

-- State the theorem
theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f' x < 0 :=
by sorry

end decreasing_interval_l166_166943


namespace find_constants_l166_166831

-- Define constants and the problem
variables (C D Q : Type) [AddCommGroup Q] [Module ℝ Q]
variables (CQ QD : ℝ) (h_ratio : CQ = 3 * QD / 5)

-- Define the conjecture we want to prove
theorem find_constants (t u : ℝ) (h_t : t = 5 / (3 + 5)) (h_u : u = 3 / (3 + 5)) :
  (CQ = 3 * QD / 5) → 
  (t * CQ + u * QD = (5 / 8) * CQ + (3 / 8) * QD) :=
sorry

end find_constants_l166_166831


namespace no_prime_solution_l166_166114

theorem no_prime_solution (p : ℕ) (h_prime : Nat.Prime p) : ¬(2^p + p ∣ 3^p + p) := by
  sorry

end no_prime_solution_l166_166114


namespace perimeter_of_intersection_triangle_l166_166421

theorem perimeter_of_intersection_triangle :
  ∀ (P Q R : Type) (dist : P → Q → ℝ) (length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR : ℝ),
  (length_PQ = 150) →
  (length_QR = 250) →
  (length_PR = 200) →
  (seg_ellP = 75) →
  (seg_ellQ = 50) →
  (seg_ellR = 25) →
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  TU + US + ST = 266.67 :=
by
  intros P Q R dist length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR hPQ hQR hPR hP hQ hR
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  have : TU + US + ST = 266.67 := sorry
  exact this

end perimeter_of_intersection_triangle_l166_166421


namespace ukuleles_and_violins_l166_166022

theorem ukuleles_and_violins (U V : ℕ) : 
  (4 * U + 6 * 4 + 4 * V = 40) → (U + V = 4) :=
by
  intro h
  sorry

end ukuleles_and_violins_l166_166022


namespace other_root_l166_166625

open Complex

-- Defining the conditions that are given in the problem
def quadratic_equation (x : ℂ) (m : ℝ) : Prop :=
  x^2 + (1 - 2 * I) * x + (3 * m - I) = 0

def has_real_root (x : ℂ) : Prop :=
  ∃ α : ℝ, x = α

-- The main theorem statement we need to prove
theorem other_root (m : ℝ) (α : ℝ) (α_real_root : quadratic_equation α m) :
  quadratic_equation (-1/2 + 2 * I) m :=
sorry

end other_root_l166_166625


namespace solve_f_inv_zero_l166_166224

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)
noncomputable def f_inv (a b x : ℝ) : ℝ := sorry -- this is where the inverse function definition would go

theorem solve_f_inv_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : f_inv a b 0 = (1 / b) :=
by sorry

end solve_f_inv_zero_l166_166224


namespace movie_ticket_cost_l166_166979

/--
Movie tickets cost a certain amount on a Monday, twice as much on a Wednesday, and five times as much as on Monday on a Saturday. If Glenn goes to the movie theater on Wednesday and Saturday, he spends $35. Prove that the cost of a movie ticket on a Monday is $5.
-/
theorem movie_ticket_cost (M : ℕ) 
  (wednesday_cost : 2 * M = 2 * M)
  (saturday_cost : 5 * M = 5 * M) 
  (total_cost : 2 * M + 5 * M = 35) : 
  M = 5 := 
sorry

end movie_ticket_cost_l166_166979


namespace least_possible_value_l166_166432

theorem least_possible_value (x y : ℝ) : 
  ∃ (x y : ℝ), (xy + 1)^2 + (x + y + 1)^2 = 0 := 
sorry

end least_possible_value_l166_166432


namespace part_one_part_two_l166_166944

noncomputable def f (a x : ℝ) : ℝ := x * (a + Real.log x)
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≥ -1/Real.exp 1) : a = 0 := sorry

theorem part_two {a x : ℝ} (ha : a > 0) (hx : x > 0) :
  g x - f a x < 2 / Real.exp 1 := sorry

end part_one_part_two_l166_166944


namespace inequality_chain_l166_166192

theorem inequality_chain (a : ℝ) (h : a - 1 > 0) : -a < -1 ∧ -1 < 1 ∧ 1 < a := by
  sorry

end inequality_chain_l166_166192


namespace calculate_bus_stoppage_time_l166_166313

variable (speed_excl_stoppages speed_incl_stoppages distance_excl_stoppages distance_incl_stoppages distance_diff time_lost_stoppages : ℝ)

def bus_stoppage_time
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  Prop :=
  speed_excl_stoppages = 32 ∧
  speed_incl_stoppages = 16 ∧
  time_stopped = 30

theorem calculate_bus_stoppage_time 
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  bus_stoppage_time speed_excl_stoppages speed_incl_stoppages time_stopped :=
by
  have h1 : speed_excl_stoppages = 32 := by
    sorry
  have h2 : speed_incl_stoppages = 16 := by
    sorry
  have h3 : time_stopped = 30 := by
    sorry
  exact ⟨h1, h2, h3⟩

end calculate_bus_stoppage_time_l166_166313


namespace point_in_fourth_quadrant_l166_166640

variable (a : ℝ)

theorem point_in_fourth_quadrant (h : a < -1) : 
    let x := a^2 - 2*a - 1
    let y := (a + 1) / abs (a + 1)
    (x > 0) ∧ (y < 0) := 
by
  let x := a^2 - 2*a - 1
  let y := (a + 1) / abs (a + 1)
  sorry

end point_in_fourth_quadrant_l166_166640


namespace intersection_A_B_l166_166200

def A : Set ℝ := {x | x * (x - 4) < 0}
def B : Set ℝ := {0, 1, 5}

theorem intersection_A_B : (A ∩ B) = {1} := by
  sorry

end intersection_A_B_l166_166200


namespace initial_water_amount_l166_166121

theorem initial_water_amount (W : ℝ) 
  (evap_per_day : ℝ := 0.0008) 
  (days : ℤ := 50) 
  (percentage_evap : ℝ := 0.004) 
  (evap_total : ℝ := evap_per_day * days) 
  (evap_eq : evap_total = percentage_evap * W) : 
  W = 10 := 
by
  sorry

end initial_water_amount_l166_166121


namespace parabola_transform_correct_l166_166396

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -2 * x^2 + 1

-- Define the transformation of moving the parabola one unit to the right and one unit up
def transformed_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 2

-- The theorem to prove
theorem parabola_transform_correct :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 1) + 1 :=
by
  intros x
  sorry

end parabola_transform_correct_l166_166396


namespace triploid_fruit_fly_chromosome_periodicity_l166_166766

-- Define the conditions
def normal_chromosome_count (organism: Type) : ℕ := 8
def triploid_fruit_fly (organism: Type) : Prop := true
def XXY_sex_chromosome_composition (organism: Type) : Prop := true
def periodic_change (counts: List ℕ) : Prop := counts = [9, 18, 9]

-- State the theorem
theorem triploid_fruit_fly_chromosome_periodicity (organism: Type)
  (h1: triploid_fruit_fly organism) 
  (h2: XXY_sex_chromosome_composition organism)
  (h3: normal_chromosome_count organism = 8) : 
  periodic_change [9, 18, 9] :=
sorry

end triploid_fruit_fly_chromosome_periodicity_l166_166766


namespace annual_rent_per_square_foot_l166_166744

-- Given conditions
def dimensions_length : ℕ := 10
def dimensions_width : ℕ := 10
def monthly_rent : ℕ := 1300

-- Derived conditions
def area : ℕ := dimensions_length * dimensions_width
def annual_rent : ℕ := monthly_rent * 12

-- The problem statement as a theorem in Lean 4
theorem annual_rent_per_square_foot :
  annual_rent / area = 156 := by
  sorry

end annual_rent_per_square_foot_l166_166744


namespace f_zero_is_one_l166_166291

def f (n : ℕ) : ℕ := sorry

theorem f_zero_is_one (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (f n) + f n = 2 * n + 3)
  (h2 : f 2015 = 2016) : f 0 = 1 := 
by {
  -- proof not required
  sorry
}

end f_zero_is_one_l166_166291


namespace possible_values_a_l166_166381

noncomputable def setA (a : ℝ) : Set ℝ := { x | a * x + 2 = 0 }
def setB : Set ℝ := {-1, 2}

theorem possible_values_a :
  ∀ a : ℝ, setA a ⊆ setB ↔ a = -1 ∨ a = 0 ∨ a = 2 :=
by
  intro a
  sorry

end possible_values_a_l166_166381


namespace solve_equation_l166_166616

noncomputable def a := 3 + Real.sqrt 8
noncomputable def b := 3 - Real.sqrt 8

theorem solve_equation (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 6) ↔ (x = 2 ∨ x = -2) := 
  by
  sorry

end solve_equation_l166_166616


namespace product_of_three_consecutive_not_div_by_5_adjacency_l166_166534

theorem product_of_three_consecutive_not_div_by_5_adjacency (a b c : ℕ) (h₁ : a + 1 = b) (h₂ : b + 1 = c) (h₃ : a % 5 ≠ 0) (h₄ : b % 5 ≠ 0) (h₅ : c % 5 ≠ 0) :
  ((a * b * c) % 5 = 1) ∨ ((a * b * c) % 5 = 4) := 
sorry

end product_of_three_consecutive_not_div_by_5_adjacency_l166_166534


namespace vlad_taller_than_sister_l166_166273

-- Definitions based on the conditions
def vlad_feet : ℕ := 6
def vlad_inches : ℕ := 3
def sister_feet : ℕ := 2
def sister_inches : ℕ := 10
def inches_per_foot : ℕ := 12

-- Derived values for heights in inches
def vlad_height_in_inches : ℕ := (vlad_feet * inches_per_foot) + vlad_inches
def sister_height_in_inches : ℕ := (sister_feet * inches_per_foot) + sister_inches

-- Lean 4 statement for the proof problem
theorem vlad_taller_than_sister : vlad_height_in_inches - sister_height_in_inches = 41 := 
by 
  sorry

end vlad_taller_than_sister_l166_166273


namespace parallel_lines_condition_l166_166253

theorem parallel_lines_condition (m n : ℝ) :
  (∃x y, (m * x + y - n = 0) ∧ (x + m * y + 1 = 0)) →
  (m = 1 ∧ n ≠ -1) ∨ (m = -1 ∧ n ≠ 1) :=
by
  sorry

end parallel_lines_condition_l166_166253


namespace find_other_number_l166_166100

theorem find_other_number
  (n m lcm gcf : ℕ)
  (h_n : n = 40)
  (h_lcm : lcm = 56)
  (h_gcf : gcf = 10)
  (h_lcm_gcf : lcm * gcf = n * m) : m = 14 :=
by
  sorry

end find_other_number_l166_166100


namespace intersection_is_3_l166_166380

def setA : Set ℕ := {5, 2, 3}
def setB : Set ℕ := {9, 3, 6}

theorem intersection_is_3 : setA ∩ setB = {3} := by
  sorry

end intersection_is_3_l166_166380


namespace chocolate_per_friend_l166_166372

-- Definitions according to the conditions
def total_chocolate : ℚ := 60 / 7
def piles := 5
def friends := 3

-- Proof statement for the equivalent problem
theorem chocolate_per_friend :
  (total_chocolate / piles) * (piles - 1) / friends = 16 / 7 := by
  sorry

end chocolate_per_friend_l166_166372


namespace no_green_ball_in_bag_l166_166060

theorem no_green_ball_in_bag (bag : Set String) (h : bag = {"red", "yellow", "white"}): ¬ ("green" ∈ bag) :=
by
  sorry

end no_green_ball_in_bag_l166_166060


namespace custom_mul_of_two_and_neg_three_l166_166644

-- Define the custom operation "*"
def custom.mul (a b : Int) : Int := a * b

-- The theorem to prove that 2 * (-3) using custom.mul equals -6
theorem custom_mul_of_two_and_neg_three : custom.mul 2 (-3) = -6 :=
by
  -- This is where the proof would go
  sorry

end custom_mul_of_two_and_neg_three_l166_166644


namespace restaurant_total_earnings_l166_166300

noncomputable def restaurant_earnings (weekdays weekends : ℕ) (weekday_earnings : ℝ) 
    (weekend_min_earnings weekend_max_earnings discount special_event_earnings : ℝ) : ℝ :=
  let num_mondays := weekdays / 5 
  let weekday_earnings_with_discount := weekday_earnings - (weekday_earnings * discount)
  let earnings_mondays := num_mondays * weekday_earnings_with_discount
  let earnings_other_weekdays := (weekdays - num_mondays) * weekday_earnings
  let average_weekend_earnings := (weekend_min_earnings + weekend_max_earnings) / 2
  let total_weekday_earnings := earnings_mondays + earnings_other_weekdays
  let total_weekend_earnings := 2 * weekends * average_weekend_earnings
  total_weekday_earnings + total_weekend_earnings + special_event_earnings

theorem restaurant_total_earnings 
  (weekdays weekends : ℕ)
  (weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings total_earnings : ℝ)
  (h_weekdays : weekdays = 22)
  (h_weekends : weekends = 8)
  (h_weekday_earnings : weekday_earnings = 600)
  (h_weekend_min_earnings : weekend_min_earnings = 1000)
  (h_weekend_max_earnings : weekend_max_earnings = 1500)
  (h_discount : discount = 0.1)
  (h_special_event_earnings : special_event_earnings = 500)
  (h_total_earnings : total_earnings = 33460) :
  restaurant_earnings weekdays weekends weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings = total_earnings := 
by
  sorry

end restaurant_total_earnings_l166_166300


namespace distances_product_eq_l166_166975

-- Define the distances
variables (d_ab d_ac d_bc d_ba d_cb d_ca : ℝ)

-- State the theorem
theorem distances_product_eq : d_ab * d_bc * d_ca = d_ac * d_ba * d_cb :=
sorry

end distances_product_eq_l166_166975


namespace vertex_sum_of_cube_l166_166079

noncomputable def cube_vertex_sum (a : Fin 8 → ℕ) : ℕ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7

def face_sums (a : Fin 8 → ℕ) : List ℕ :=
  [
    a 0 + a 1 + a 2 + a 3, -- first face
    a 0 + a 1 + a 4 + a 5, -- second face
    a 0 + a 3 + a 4 + a 7, -- third face
    a 1 + a 2 + a 5 + a 6, -- fourth face
    a 2 + a 3 + a 6 + a 7, -- fifth face
    a 4 + a 5 + a 6 + a 7  -- sixth face
  ]

def total_face_sum (a : Fin 8 → ℕ) : ℕ :=
  List.sum (face_sums a)

theorem vertex_sum_of_cube (a : Fin 8 → ℕ) (h : total_face_sum a = 2019) :
  cube_vertex_sum a = 673 :=
sorry

end vertex_sum_of_cube_l166_166079


namespace decreasing_intervals_sin_decreasing_intervals_log_cos_l166_166868

theorem decreasing_intervals_sin (k : ℤ) :
  ∀ x : ℝ, 
    ( (π / 2 + 2 * k * π < x) ∧ (x < 3 * π / 2 + 2 * k * π) ) ↔
    (∃ k : ℤ, (π / 2 + 2 * k * π < x) ∧ (x < 3 * π / 2 + 2 * k * π)) :=
sorry

theorem decreasing_intervals_log_cos (k : ℤ) :
  ∀ x : ℝ, 
    ( (2 * k * π < x) ∧ (x < π / 2 + 2 * k * π) ) ↔
    (∃ k : ℤ, (2 * k * π < x) ∧ (x < π / 2 + 2 * k * π)) :=
sorry

end decreasing_intervals_sin_decreasing_intervals_log_cos_l166_166868


namespace part1_part2_l166_166999

/-
Part 1: Given the conditions of parabola and line intersection, prove the range of slope k of the line.
-/
theorem part1 (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  k > -2 + 2 * Real.sqrt 2 ∨ k < -2 - 2 * Real.sqrt 2 :=
  sorry

/-
Part 2: Given the conditions of locus of point Q on the line segment P1P2, prove the equation of the locus.
-/
theorem part2 (x y : ℝ) (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  2 * x - y + 1 = 0 ∧ (-Real.sqrt 2 - 1 < x ∧ x < Real.sqrt 2 - 1 ∧ x ≠ -1) :=
  sorry

end part1_part2_l166_166999


namespace no_arithmetic_sequence_without_square_gt1_l166_166006

theorem no_arithmetic_sequence_without_square_gt1 (a d : ℕ) (h_d : d ≠ 0) :
  ¬(∀ n : ℕ, ∃ k : ℕ, k > 0 ∧ k ∈ {a + n * d | n : ℕ} ∧ ∀ m : ℕ, m > 1 → m * m ∣ k → false) := sorry

end no_arithmetic_sequence_without_square_gt1_l166_166006


namespace ratio_fifth_terms_l166_166886

-- Define the arithmetic sequences and their sums
variables {a b : ℕ → ℕ}
variables {S T : ℕ → ℕ}

-- Assume conditions of the problem
axiom sum_condition (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom sum_condition2 (n : ℕ) : T n = n * (b 1 + b n) / 2
axiom ratio_condition : ∀ n, S n / T n = (2 * n - 3) / (3 * n - 2)

-- Prove the ratio of fifth terms a_5 / b_5
theorem ratio_fifth_terms : (a 5 : ℚ) / b 5 = 3 / 5 := by
  sorry

end ratio_fifth_terms_l166_166886


namespace gcd_of_B_is_2_l166_166973

-- Condition: B is the set of all numbers which can be represented as the sum of four consecutive positive integers
def B := { n : ℕ | ∃ y : ℕ, n = (y - 1) + y + (y + 1) + (y + 2) }

-- Question: What is the greatest common divisor of all numbers in \( B \)
-- Mathematical equivalent proof problem: Prove gcd of all elements in set \( B \) is 2

theorem gcd_of_B_is_2 : ∀ n ∈ B, ∃ y : ℕ, n = 2 * (2 * y + 1) → ∀ m ∈ B, n.gcd m = 2 :=
by
  sorry

end gcd_of_B_is_2_l166_166973


namespace total_cost_of_books_l166_166747

theorem total_cost_of_books (total_children : ℕ) (n : ℕ) (extra_payment_per_child : ℕ) (cost : ℕ) :
  total_children = 12 →
  n = 2 →
  extra_payment_per_child = 10 →
  (total_children - n) * extra_payment_per_child = 100 →
  cost = 600 :=
by
  intros h1 h2 h3 h4
  sorry

end total_cost_of_books_l166_166747


namespace complement_U_A_l166_166201

-- Define the universal set U and the subset A
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}

-- Define the complement of A relative to the universal set U
def complement (U A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

-- The theorem we want to prove
theorem complement_U_A : complement U A = {2} := by
  sorry

end complement_U_A_l166_166201


namespace smallest_four_digit_number_l166_166786

noncomputable def smallest_four_digit_solution : ℕ := 1011

theorem smallest_four_digit_number (x : ℕ) (h1 : 5 * x ≡ 25 [MOD 20]) (h2 : 3 * x + 10 ≡ 19 [MOD 7]) (h3 : x + 3 ≡ 2 * x [MOD 12]) :
  x = smallest_four_digit_solution :=
by
  sorry

end smallest_four_digit_number_l166_166786


namespace cheaper_candy_price_l166_166451

theorem cheaper_candy_price
    (mix_total_weight : ℝ) (mix_price_per_pound : ℝ)
    (cheap_weight : ℝ) (expensive_weight : ℝ) (expensive_price_per_pound : ℝ)
    (cheap_total_value : ℝ) (expensive_total_value : ℝ) (total_mix_value : ℝ) :
    mix_total_weight = 80 →
    mix_price_per_pound = 2.20 →
    cheap_weight = 64 →
    expensive_weight = mix_total_weight - cheap_weight →
    expensive_price_per_pound = 3.00 →
    cheap_total_value = cheap_weight * x →
    expensive_total_value = expensive_weight * expensive_price_per_pound →
    total_mix_value = mix_total_weight * mix_price_per_pound →
    total_mix_value = cheap_total_value + expensive_total_value →
    x = 2 := 
sorry

end cheaper_candy_price_l166_166451


namespace person_A_number_is_35_l166_166391

theorem person_A_number_is_35
    (A B : ℕ)
    (h1 : A + B = 8)
    (h2 : 10 * B + A - (10 * A + B) = 18) :
    10 * A + B = 35 :=
by
    sorry

end person_A_number_is_35_l166_166391


namespace discount_percentage_l166_166002

variable (P : ℝ) (r : ℝ) (S : ℝ)

theorem discount_percentage (hP : P = 20) (hr : r = 30 / 100) (hS : S = 13) :
  (P * (1 + r) - S) / (P * (1 + r)) * 100 = 50 := 
sorry

end discount_percentage_l166_166002


namespace melissa_work_hours_l166_166847

theorem melissa_work_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (hours_per_dress : ℕ) (total_num_dresses : ℕ) (total_hours : ℕ) 
  (h1 : total_fabric = 56) (h2 : fabric_per_dress = 4) (h3 : hours_per_dress = 3) : 
  total_hours = (total_fabric / fabric_per_dress) * hours_per_dress := by
  sorry

end melissa_work_hours_l166_166847


namespace b_charges_l166_166740

theorem b_charges (total_cost : ℕ) (a_hours b_hours c_hours : ℕ)
  (h_total_cost : total_cost = 720)
  (h_a_hours : a_hours = 9)
  (h_b_hours : b_hours = 10)
  (h_c_hours : c_hours = 13) :
  (total_cost * b_hours / (a_hours + b_hours + c_hours)) = 225 :=
by
  sorry

end b_charges_l166_166740


namespace jake_weight_l166_166055

theorem jake_weight {J S : ℝ} (h1 : J - 20 = 2 * S) (h2 : J + S = 224) : J = 156 :=
by
  sorry

end jake_weight_l166_166055


namespace stratified_sampling_expected_females_l166_166589

noncomputable def sample_size := 14
noncomputable def total_athletes := 44 + 33
noncomputable def female_athletes := 33
noncomputable def stratified_sample := (female_athletes * sample_size) / total_athletes

theorem stratified_sampling_expected_females :
  stratified_sample = 6 :=
by
  sorry

end stratified_sampling_expected_females_l166_166589


namespace problem_l166_166482

def pair_eq (a b c d : ℝ) : Prop := (a = c) ∧ (b = d)

def op_a (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, b * c - a * d)
def op_o (a b c d : ℝ) : ℝ × ℝ := (a + c, b + d)

theorem problem (x y : ℝ) :
  op_a 3 4 x y = (11, -2) →
  op_o 3 4 x y = (4, 6) :=
sorry

end problem_l166_166482


namespace arithmetic_sequence_first_term_l166_166214

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 9) (h2 : 2 * a 3 = a 2 + 6) : a 1 = -3 :=
by
  -- a_5 = a_1 + 4d
  have h3 : a 5 = a 1 + 4 * d := sorry
  
  -- 2a_3 = a_2 + 6, which means 2 * (a_1 + 2d) = (a_1 + d) + 6
  have h4 : 2 * (a 1 + 2 * d) = (a 1 + d) + 6 := sorry
  
  -- solve the system of linear equations to find a_1 = -3
  sorry

end arithmetic_sequence_first_term_l166_166214


namespace f_odd_f_monotonic_range_of_x_l166_166327

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((1 + x) / (1 - x))

theorem f_odd : ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x := by
  sorry

theorem f_monotonic : ∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂ := by
  sorry

theorem range_of_x (x : ℝ) : f (1 / (x - 3)) + f (- 1 / 3) < 0 → x < 2 ∨ x > 6 := by
  sorry

end f_odd_f_monotonic_range_of_x_l166_166327


namespace largest_A_proof_smallest_A_proof_l166_166771

def is_coprime_with_12 (n : ℕ) : Prop := Nat.gcd n 12 = 1

def obtain_A_from_B (B : ℕ) : ℕ :=
  let b := B % 10
  let k := B / 10
  b * 10^7 + k

constant B : ℕ → Prop
constant A : ℕ → ℕ → Prop

noncomputable def largest_A : ℕ :=
  99999998

noncomputable def smallest_A : ℕ :=
  14444446

theorem largest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = largest_A :=
sorry

theorem smallest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = smallest_A :=
sorry

end largest_A_proof_smallest_A_proof_l166_166771


namespace smallest_integer_solution_l166_166556

theorem smallest_integer_solution (x : ℤ) :
  (7 - 5 * x < 12) → ∃ (n : ℤ), x = n ∧ n = 0 :=
by
  intro h
  sorry

end smallest_integer_solution_l166_166556


namespace probability_higher_2012_l166_166357

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

noncomputable def passing_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  1 - ∑ i in finset.range (k), binomial_probability n i p

theorem probability_higher_2012 :
  passing_probability 40 6 0.25 > passing_probability 20 3 0.25 :=
sorry

end probability_higher_2012_l166_166357


namespace property_tax_difference_correct_l166_166503

-- Define the tax rates for different ranges
def tax_rate (value : ℕ) : ℝ :=
  if value ≤ 10000 then 0.05
  else if value ≤ 20000 then 0.075
  else if value ≤ 30000 then 0.10
  else 0.125

-- Define the progressive tax calculation for a given assessed value
def calculate_tax (value : ℕ) : ℝ :=
  if value ≤ 10000 then value * 0.05
  else if value ≤ 20000 then 10000 * 0.05 + (value - 10000) * 0.075
  else if value <= 30000 then 10000 * 0.05 + 10000 * 0.075 + (value - 20000) * 0.10
  else 10000 * 0.05 + 10000 * 0.075 + 10000 * 0.10 + (value - 30000) * 0.125

-- Define the initial and new assessed values
def initial_value : ℕ := 20000
def new_value : ℕ := 28000

-- Define the difference in tax calculation
def tax_difference : ℝ := calculate_tax new_value - calculate_tax initial_value

theorem property_tax_difference_correct : tax_difference = 550 := by
  sorry

end property_tax_difference_correct_l166_166503


namespace simplify_fraction_l166_166690

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l166_166690


namespace binom_8_3_eq_56_and_2_pow_56_l166_166000

theorem binom_8_3_eq_56_and_2_pow_56 :
  (Nat.choose 8 3 = 56) ∧ (2 ^ (Nat.choose 8 3) = 2 ^ 56) :=
by
  sorry

end binom_8_3_eq_56_and_2_pow_56_l166_166000


namespace combine_like_terms_1_simplify_expression_2_l166_166923

-- Problem 1
theorem combine_like_terms_1 (m n : ℝ) :
  2 * m^2 * n - 3 * m * n + 8 - 3 * m^2 * n + 5 * m * n - 3 = -m^2 * n + 2 * m * n + 5 :=
by 
  -- Proof goes here 
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by 
  -- Proof goes here 
  sorry

end combine_like_terms_1_simplify_expression_2_l166_166923


namespace simplify_expression_l166_166682

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l166_166682


namespace max_value_of_z_l166_166836

theorem max_value_of_z (k : ℝ) (x y : ℝ)
  (h1 : x + 2 * y - 1 ≥ 0)
  (h2 : x - y ≥ 0)
  (h3 : 0 ≤ x)
  (h4 : x ≤ k)
  (h5 : ∀ x y, x + 2 * y - 1 ≥ 0 ∧ x - y ≥ 0 ∧ 0 ≤ x ∧ x ≤ k → x + k * y ≥ -2) :
  ∃ (x y : ℝ), x + k * y = 20 := 
by
  sorry

end max_value_of_z_l166_166836


namespace sheena_sewing_weeks_l166_166859

theorem sheena_sewing_weeks (sew_time : ℕ) (bridesmaids : ℕ) (sewing_per_week : ℕ) 
    (h_sew_time : sew_time = 12) (h_bridesmaids : bridesmaids = 5) (h_sewing_per_week : sewing_per_week = 4) : 
    (bridesmaids * sew_time) / sewing_per_week = 15 := 
  by sorry

end sheena_sewing_weeks_l166_166859


namespace gcd_102_238_is_34_l166_166729

noncomputable def gcd_102_238 : ℕ :=
  Nat.gcd 102 238

theorem gcd_102_238_is_34 : gcd_102_238 = 34 := by
  -- Conditions based on the Euclidean algorithm
  have h1 : 238 = 2 * 102 + 34 := by norm_num
  have h2 : 102 = 3 * 34 := by norm_num
  have h3 : Nat.gcd 102 34 = 34 := by
    rw [Nat.gcd, Nat.gcd_rec]
    exact Nat.gcd_eq_left h2

  -- Conclusion
  show gcd_102_238 = 34 from
    calc gcd_102_238 = Nat.gcd 102 238 : rfl
                  ... = Nat.gcd 34 102 : Nat.gcd_comm 102 34
                  ... = Nat.gcd 34 (102 % 34) : by rw [Nat.gcd_rec]
                  ... = Nat.gcd 34 34 : by rw [Nat.mod_eq_of_lt (by norm_num : 34 < 102)]
                  ... = 34 : Nat.gcd_self 34

end gcd_102_238_is_34_l166_166729


namespace Penny_total_species_identified_l166_166967

/-- Penny identified 35 species of sharks, 15 species of eels, and 5 species of whales.
    Prove that the total number of species identified is 55. -/
theorem Penny_total_species_identified :
  let sharks_species := 35
  let eels_species := 15
  let whales_species := 5
  sharks_species + eels_species + whales_species = 55 :=
by
  sorry

end Penny_total_species_identified_l166_166967


namespace total_flowering_bulbs_count_l166_166559

-- Definitions for the problem conditions
def crocus_cost : ℝ := 0.35
def daffodil_cost : ℝ := 0.65
def total_budget : ℝ := 29.15
def crocus_count : ℕ := 22

-- Theorem stating the total number of bulbs that can be bought
theorem total_flowering_bulbs_count : 
  ∃ daffodil_count : ℕ, (crocus_count + daffodil_count = 55) ∧ (total_budget = crocus_cost * crocus_count + daffodil_count * daffodil_cost) :=
  sorry

end total_flowering_bulbs_count_l166_166559


namespace sufficient_but_not_necessary_l166_166401

theorem sufficient_but_not_necessary (x y : ℝ) (h : ⌊x⌋ = ⌊y⌋) : 
  |x - y| < 1 ∧ ∃ x y : ℝ, |x - y| < 1 ∧ ⌊x⌋ ≠ ⌊y⌋ :=
by 
  sorry

end sufficient_but_not_necessary_l166_166401


namespace scatter_plot_role_regression_analysis_l166_166412

theorem scatter_plot_role_regression_analysis :
  ∀ (role : String), 
  (role = "Finding the number of individuals" ∨ 
   role = "Comparing the size relationship of individual data" ∨ 
   role = "Exploring individual classification" ∨ 
   role = "Roughly judging whether variables are linearly related")
  → role = "Roughly judging whether variables are linearly related" :=
by
  intros role h
  sorry

end scatter_plot_role_regression_analysis_l166_166412


namespace train_speed_proof_l166_166762

noncomputable def train_speed_kmh (length_train : ℝ) (time_crossing : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := length_train / time_crossing
  let train_speed_ms := relative_speed - man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_proof :
  train_speed_kmh 150 8 7 = 60.5 :=
by
  sorry

end train_speed_proof_l166_166762


namespace tens_digit_of_3_pow_2013_l166_166105

theorem tens_digit_of_3_pow_2013 : (3^2013 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_3_pow_2013_l166_166105


namespace max_discount_benefit_l166_166211

theorem max_discount_benefit {S X : ℕ} (P : ℕ → Prop) :
  S = 1000 →
  X = 99 →
  (∀ s1 s2 s3 s4 : ℕ, s1 ≥ s2 ∧ s2 ≥ s3 ∧ s3 ≥ s4 ∧ s4 ≥ X ∧ s1 + s2 + s3 + s4 = S →
  ∃ N : ℕ, P N ∧ N = 504) := 
by
  intros hS hX
  sorry

end max_discount_benefit_l166_166211


namespace remainder_of_division_l166_166540

theorem remainder_of_division (x r : ℕ) (h1 : 1620 - x = 1365) (h2 : 1620 = x * 6 + r) : r = 90 :=
sorry

end remainder_of_division_l166_166540


namespace book_selection_l166_166953

theorem book_selection (total_books novels : ℕ) (choose_books : ℕ)
  (h_total : total_books = 15)
  (h_novels : novels = 5)
  (h_choose : choose_books = 3) :
  (Nat.choose 15 3 - Nat.choose 10 3) = 335 :=
by
  sorry

end book_selection_l166_166953


namespace smallest_lcm_l166_166053

theorem smallest_lcm (m n : ℕ) (hm : 10000 ≤ m ∧ m < 100000) (hn : 10000 ≤ n ∧ n < 100000) (h : Nat.gcd m n = 5) : Nat.lcm m n = 20030010 :=
sorry

end smallest_lcm_l166_166053


namespace can_form_triangle_l166_166513

theorem can_form_triangle (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_condition : c^2 ≤ 4 * a * b) : 
  a + b > c ∧ a + c > b ∧ b + c > a := 
sorry

end can_form_triangle_l166_166513


namespace overtime_hours_l166_166292

theorem overtime_hours
  (regularPayPerHour : ℝ)
  (regularHours : ℝ)
  (totalPay : ℝ)
  (overtimeRate : ℝ) 
  (h1 : regularPayPerHour = 3)
  (h2 : regularHours = 40)
  (h3 : totalPay = 168)
  (h4 : overtimeRate = 2 * regularPayPerHour) :
  (totalPay - (regularPayPerHour * regularHours)) / overtimeRate = 8 :=
by
  sorry

end overtime_hours_l166_166292


namespace find_difference_l166_166835

noncomputable def expression (x y : ℝ) : ℝ :=
  (|x + y| / (|x| + |y|))^2

theorem find_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  let m := 0
  let M := 1
  M - m = 1 :=
by
  -- Please note that the proof is omitted and replaced with sorry
  sorry

end find_difference_l166_166835


namespace martha_total_cost_l166_166884

-- Definitions for the conditions
def amount_cheese_needed : ℝ := 1.5 -- in kg
def amount_meat_needed : ℝ := 0.5 -- in kg
def cost_cheese_per_kg : ℝ := 6.0 -- in dollars per kg
def cost_meat_per_kg : ℝ := 8.0 -- in dollars per kg

-- Total cost that needs to be calculated
def total_cost : ℝ :=
  (amount_cheese_needed * cost_cheese_per_kg) +
  (amount_meat_needed * cost_meat_per_kg)

-- Statement of the theorem
theorem martha_total_cost : total_cost = 13 := by
  sorry

end martha_total_cost_l166_166884


namespace min_sum_a_b_l166_166349

theorem min_sum_a_b (a b : ℝ) (h1 : 4 * a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  a + b ≥ 16 :=
sorry

end min_sum_a_b_l166_166349


namespace direct_proportion_conditions_l166_166793

theorem direct_proportion_conditions (k b : ℝ) : 
  (y = (k - 4) * x + b → (k ≠ 4 ∧ b = 0)) ∧ ¬ (b ≠ 0 ∨ k ≠ 4) :=
sorry

end direct_proportion_conditions_l166_166793


namespace pencils_per_box_l166_166596

-- Variables and Definitions based on the problem conditions
def num_boxes : ℕ := 10
def pencils_kept : ℕ := 10
def friends : ℕ := 5
def pencils_per_friend : ℕ := 8

-- Theorem to prove the solution
theorem pencils_per_box (pencils_total : ℕ)
  (h1 : pencils_total = pencils_kept + (friends * pencils_per_friend))
  (h2 : pencils_total = num_boxes * (pencils_total / num_boxes)) :
  (pencils_total / num_boxes) = 5 :=
sorry

end pencils_per_box_l166_166596


namespace minimum_value_inequality_l166_166227

theorem minimum_value_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 9) :
  (x^3 + y^3) / (x + y) + (x^3 + z^3) / (x + z) + (y^3 + z^3) / (y + z) ≥ 27 :=
sorry

end minimum_value_inequality_l166_166227


namespace find_y_if_x_l166_166971

theorem find_y_if_x (x : ℝ) (hx : x^2 + 8 * (x / (x - 3))^2 = 53) :
  (∃ y, y = (x - 3)^3 * (x + 4) / (2 * x - 5) ∧ y = 17000 / 21) :=
  sorry

end find_y_if_x_l166_166971


namespace speed_of_A_l166_166901

theorem speed_of_A :
  ∀ (v_A : ℝ), 
    (v_A * 2 + 7 * 2 = 24) → 
    v_A = 5 :=
by
  intro v_A
  intro h
  have h1 : v_A * 2 = 10 := by linarith
  have h2 : v_A = 5 := by linarith
  exact h2

end speed_of_A_l166_166901


namespace six_digit_numbers_l166_166047

theorem six_digit_numbers :
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
  sorry

end six_digit_numbers_l166_166047


namespace gerald_paid_l166_166810

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 := by
  sorry

end gerald_paid_l166_166810


namespace football_field_width_l166_166992

theorem football_field_width (length : ℕ) (total_distance : ℕ) (laps : ℕ) (width : ℕ) 
  (h1 : length = 100) (h2 : total_distance = 1800) (h3 : laps = 6) :
  width = 50 :=
by 
  -- Proof omitted
  sorry

end football_field_width_l166_166992


namespace brianna_sandwiches_l166_166305

theorem brianna_sandwiches (meats : ℕ) (cheeses : ℕ) (h_meats : meats = 8) (h_cheeses : cheeses = 7) :
  (Nat.choose meats 2) * (Nat.choose cheeses 1) = 196 := 
by
  rw [h_meats, h_cheeses]
  norm_num
  sorry

end brianna_sandwiches_l166_166305


namespace intersection_point_on_circle_l166_166803

theorem intersection_point_on_circle :
  ∀ (m : ℝ) (x y : ℝ),
  (m * x - y = 0) → 
  (x + m * y - m - 2 = 0) → 
  (x - 1)^2 + (y - 1 / 2)^2 = 5 / 4 :=
by
  intros m x y h1 h2
  sorry

end intersection_point_on_circle_l166_166803


namespace average_minutes_run_per_day_l166_166508

variables (f : ℕ)
def third_graders := 6 * f
def fourth_graders := 2 * f
def fifth_graders := f

def total_minutes_run := 14 * third_graders f + 18 * fourth_graders f + 8 * fifth_graders f
def total_students := third_graders f + fourth_graders f + fifth_graders f

theorem average_minutes_run_per_day : 
  (total_minutes_run f) / (total_students f) = 128 / 9 :=
by
  sorry

end average_minutes_run_per_day_l166_166508


namespace solve_inequality_l166_166263

theorem solve_inequality (x : ℝ) (h : x / 3 - 2 < 0) : x < 6 :=
sorry

end solve_inequality_l166_166263


namespace geom_seq_inequality_l166_166344

theorem geom_seq_inequality 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_pos : ∀ n : ℕ, a n > 0) 
  (h_q : q ≠ 1) : 
  a 1 + a 4 > a 2 + a 3 := 
sorry

end geom_seq_inequality_l166_166344


namespace fractional_sum_equals_015025_l166_166781

theorem fractional_sum_equals_015025 :
  (2 / 20) + (8 / 200) + (3 / 300) + (5 / 40000) * 2 = 0.15025 := 
by
  sorry

end fractional_sum_equals_015025_l166_166781


namespace joshua_bottle_caps_l166_166517

theorem joshua_bottle_caps (initial_caps : ℕ) (additional_caps : ℕ) (total_caps : ℕ) 
  (h1 : initial_caps = 40) 
  (h2 : additional_caps = 7) 
  (h3 : total_caps = initial_caps + additional_caps) : 
  total_caps = 47 := 
by 
  sorry

end joshua_bottle_caps_l166_166517


namespace sector_area_l166_166399

/-- The area of a sector with a central angle of 72 degrees and a radius of 20 cm is 80π cm². -/
theorem sector_area (radius : ℝ) (angle : ℝ) (h_angle_deg : angle = 72) (h_radius : radius = 20) :
  (angle / 360) * π * radius^2 = 80 * π :=
by sorry

end sector_area_l166_166399


namespace girls_boys_ratio_l166_166912

theorem girls_boys_ratio (G B : ℕ) (h1 : G + B = 100) (h2 : 0.20 * (G : ℝ) + 0.10 * (B : ℝ) = 15) : G / B = 1 :=
by
  -- Proof steps are omitted
  sorry

end girls_boys_ratio_l166_166912


namespace max_prime_area_of_rectangle_with_perimeter_40_is_19_l166_166587

-- Predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Given conditions: perimeter of 40 units; perimeter condition and area as prime number.
def max_prime_area_of_rectangle_with_perimeter_40 : Prop :=
  ∃ (l w : ℕ), l + w = 20 ∧ is_prime (l * (20 - l)) ∧
  ∀ (l' w' : ℕ), l' + w' = 20 → is_prime (l' * (20 - l')) → (l * (20 - l)) ≥ (l' * (20 - l'))

theorem max_prime_area_of_rectangle_with_perimeter_40_is_19 :
  max_prime_area_of_rectangle_with_perimeter_40 :=
sorry

end max_prime_area_of_rectangle_with_perimeter_40_is_19_l166_166587


namespace tax_free_amount_correct_l166_166298

-- Definitions based on the problem conditions
def total_value : ℝ := 1720
def tax_paid : ℝ := 78.4
def tax_rate : ℝ := 0.07

-- Definition of the tax-free amount we need to prove
def tax_free_amount : ℝ := 600

-- Main theorem to prove
theorem tax_free_amount_correct : 
  ∃ X : ℝ, 0.07 * (total_value - X) = tax_paid ∧ X = tax_free_amount :=
by 
  use 600
  simp
  sorry

end tax_free_amount_correct_l166_166298


namespace number_of_schools_in_pythagoras_city_l166_166964

theorem number_of_schools_in_pythagoras_city (n : ℕ) (h1 : true) 
    (h2 : true) (h3 : ∃ m, m = (3 * n + 1) / 2)
    (h4 : true) (h5 : true) : n = 24 :=
by 
  have h6 : 69 < 3 * n := sorry
  have h7 : 3 * n < 79 := sorry
  sorry

end number_of_schools_in_pythagoras_city_l166_166964


namespace sum_of_decimals_as_fraction_l166_166614

axiom decimal_to_fraction :
  0.2 = 2 / 10 ∧
  0.04 = 4 / 100 ∧
  0.006 = 6 / 1000 ∧
  0.0008 = 8 / 10000 ∧
  0.00010 = 10 / 100000 ∧
  0.000012 = 12 / 1000000

theorem sum_of_decimals_as_fraction:
  0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 + 0.000012 = (3858:ℚ) / 15625 :=
by
  have h := decimal_to_fraction
  sorry

end sum_of_decimals_as_fraction_l166_166614


namespace range_of_a_l166_166199

noncomputable def f (a x : ℝ) : ℝ := (Real.log (x^2 - a * x + 5)) / (Real.log a)

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (ha0 : 0 < a) (ha1 : a ≠ 1) 
  (hx₁x₂ : x₁ < x₂) (hx₂ : x₂ ≤ a / 2) 
  (hf : (f a x₂ - f a x₁ < 0)) : 
  1 < a ∧ a < 2 * Real.sqrt 5 := 
sorry

end range_of_a_l166_166199


namespace e_count_estimation_l166_166340

-- Define the various parameters used in the conditions
def num_problems : Nat := 76
def avg_words_per_problem : Nat := 40
def avg_letters_per_word : Nat := 5
def frequency_of_e : Float := 0.1
def actual_e_count : Nat := 1661

-- The goal is to prove that the actual number of "e"s is 1661
theorem e_count_estimation : actual_e_count = 1661 := by
  -- Sorry, no proof is required.
  sorry

end e_count_estimation_l166_166340


namespace min_shots_for_probability_at_least_075_l166_166083

theorem min_shots_for_probability_at_least_075 (hit_rate : ℝ) (target_probability : ℝ) :
  hit_rate = 0.25 → target_probability = 0.75 → ∃ n : ℕ, n = 4 ∧ (1 - hit_rate)^n ≤ 1 - target_probability := by
  intros h_hit_rate h_target_probability
  sorry

end min_shots_for_probability_at_least_075_l166_166083


namespace simplify_expression_l166_166676

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l166_166676


namespace simplify_fraction_l166_166686

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l166_166686


namespace pages_written_in_a_year_l166_166827

theorem pages_written_in_a_year (pages_per_letter : ℕ) (friends : ℕ) (times_per_week : ℕ) (weeks_per_year : ℕ) :
  pages_per_letter = 3 → friends = 2 → times_per_week = 2 → weeks_per_year = 52 → 
  pages_per_letter * friends * times_per_week * weeks_per_year = 624 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end pages_written_in_a_year_l166_166827


namespace golden_fish_caught_times_l166_166419

open Nat

theorem golden_fish_caught_times :
  ∃ (x y z : ℕ), (4 * x + 2 * z = 2000) ∧ (2 * y + z = 800) ∧ (x + y + z = 900) :=
sorry

end golden_fish_caught_times_l166_166419


namespace unique_real_solution_N_l166_166542

theorem unique_real_solution_N (N : ℝ) :
  (∃! (x y : ℝ), 2 * x^2 + 4 * x * y + 7 * y^2 - 12 * x - 2 * y + N = 0) ↔ N = 23 :=
by
  sorry

end unique_real_solution_N_l166_166542


namespace gerald_price_l166_166806

-- Define the conditions provided in the problem

def price_hendricks := 200
def discount_percent := 20
def discount_ratio := 0.80 -- since 20% less means Hendricks paid 80% of what Gerald paid

-- Question to be answered: Prove that the price Gerald paid equals $250
-- P is what Gerald paid

theorem gerald_price (P : ℝ) (h : price_hendricks = discount_ratio * P) : P = 250 :=
by
  sorry

end gerald_price_l166_166806


namespace brooke_butter_price_l166_166152

variables (price_per_gallon_of_milk : ℝ)
variables (gallons_to_butter_conversion : ℝ)
variables (number_of_cows : ℕ)
variables (milk_per_cow : ℝ)
variables (number_of_customers : ℕ)
variables (milk_demand_per_customer : ℝ)
variables (total_earnings : ℝ)

theorem brooke_butter_price :
    price_per_gallon_of_milk = 3 →
    gallons_to_butter_conversion = 2 →
    number_of_cows = 12 →
    milk_per_cow = 4 →
    number_of_customers = 6 →
    milk_demand_per_customer = 6 →
    total_earnings = 144 →
    (total_earnings - number_of_customers * milk_demand_per_customer * price_per_gallon_of_milk) /
    (number_of_cows * milk_per_cow - number_of_customers * milk_demand_per_customer) *
    gallons_to_butter_conversion = 1.50 :=
by { sorry }

end brooke_butter_price_l166_166152


namespace find_x_axis_intercept_l166_166148

theorem find_x_axis_intercept : ∃ x, 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  -- The theorem states that there exists an x-intercept such that substituting y = 0 in the equation results in x = -2.5.
  sorry

end find_x_axis_intercept_l166_166148


namespace correct_assignment_statement_l166_166279

-- Definitions according to the problem conditions
def input_statement (x : Nat) : Prop := x = 3
def assignment_statement1 (A B : Nat) : Prop := A = B ∧ B = 2
def assignment_statement2 (T : Nat) : Prop := T = T * T
def output_statement (A : Nat) : Prop := A = 4

-- Lean statement for the problem. We need to prove that the assignment_statement2 is correct.
theorem correct_assignment_statement (T : Nat) : assignment_statement2 T :=
by sorry

end correct_assignment_statement_l166_166279


namespace existence_of_xyz_l166_166795

theorem existence_of_xyz (n : ℕ) (hn_pos : 0 < n)
    (a b c : ℕ) (ha : 0 < a ∧ a ≤ 3 * n^2 + 4 * n) 
                (hb : 0 < b ∧ b ≤ 3 * n^2 + 4 * n) 
                (hc : 0 < c ∧ c ≤ 3 * n^2 + 4 * n) : 
  ∃ (x y z : ℤ), (|x| ≤ 2 * n) ∧ (|y| ≤ 2 * n) ∧ (|z| ≤ 2 * n) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 := by
  sorry

end existence_of_xyz_l166_166795


namespace average_age_of_girls_l166_166505

variable (B G : ℝ)
variable (age_students age_boys age_girls : ℝ)
variable (ratio_boys_girls : ℝ)

theorem average_age_of_girls :
  age_students = 15.8 ∧ age_boys = 16.2 ∧ ratio_boys_girls = 1.0000000000000044 ∧ B / G = ratio_boys_girls →
  (B * age_boys + G * age_girls) / (B + G) = age_students →
  age_girls = 15.4 :=
by
  intros hconds haverage
  sorry

end average_age_of_girls_l166_166505


namespace coin_arrangements_l166_166532

theorem coin_arrangements (n m : ℕ) (hp_pos : n = 5) (hq_pos : m = 5) :
  ∃ (num_arrangements : ℕ), num_arrangements = 8568 :=
by
  -- Note: 'sorry' is used to indicate here that the proof is omitted.
  sorry

end coin_arrangements_l166_166532


namespace sqrt_four_ninths_l166_166548

theorem sqrt_four_ninths : 
  (∀ (x : ℚ), x * x = 4 / 9 → (x = 2 / 3 ∨ x = - (2 / 3))) :=
by sorry

end sqrt_four_ninths_l166_166548


namespace paul_lives_on_story_5_l166_166527

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

end paul_lives_on_story_5_l166_166527


namespace bunnies_burrow_exit_counts_l166_166135

theorem bunnies_burrow_exit_counts :
  let groupA_bunnies := 40
  let groupA_rate := 3  -- times per minute per bunny
  let groupB_bunnies := 30
  let groupB_rate := 5 / 2 -- times per minute per bunny
  let groupC_bunnies := 30
  let groupC_rate := 8 / 5 -- times per minute per bunny
  let total_bunnies := 100
  let minutes_per_day := 1440
  let days_per_week := 7
  let pre_change_rate_per_min := groupA_bunnies * groupA_rate + groupB_bunnies * groupB_rate + groupC_bunnies * groupC_rate
  let post_change_rate_per_min := pre_change_rate_per_min * 0.5
  let total_pre_change_counts := pre_change_rate_per_min * minutes_per_day * days_per_week
  let total_post_change_counts := post_change_rate_per_min * minutes_per_day * (days_per_week * 2)
  total_pre_change_counts + total_post_change_counts = 4897920 := by
    sorry

end bunnies_burrow_exit_counts_l166_166135


namespace compute_multiplied_difference_l166_166165

theorem compute_multiplied_difference (a b : ℕ) (h_a : a = 25) (h_b : b = 15) :
  3 * ((a + b) ^ 2 - (a - b) ^ 2) = 4500 := by
  sorry

end compute_multiplied_difference_l166_166165


namespace line_intersects_x_axis_l166_166146

theorem line_intersects_x_axis (x y : ℝ) (h : 5 * y - 6 * x = 15) (hy : y = 0) : x = -2.5 ∧ y = 0 := 
by
  sorry

end line_intersects_x_axis_l166_166146


namespace simplify_fraction_l166_166244

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l166_166244


namespace enclosed_area_is_correct_l166_166473

open Real
open IntervalIntegral

-- Definitions for the curve and the line
def parabola (x : ℝ) := x^2
def line (x : ℝ) := x + 2

-- Problem statement
theorem enclosed_area_is_correct :
  let f := λ x, line x - parabola x in
  ∫ x in -1..2, f x = 9 / 2 :=
by
  sorry

end enclosed_area_is_correct_l166_166473


namespace product_of_first_three_terms_of_arithmetic_sequence_l166_166257

theorem product_of_first_three_terms_of_arithmetic_sequence {a d : ℕ} (ha : a + 6 * d = 20) (hd : d = 2) : a * (a + d) * (a + 2 * d) = 960 := by
  sorry

end product_of_first_three_terms_of_arithmetic_sequence_l166_166257


namespace drawings_on_last_page_l166_166720

theorem drawings_on_last_page :
  let n_notebooks := 10 
  let p_pages := 50
  let d_original := 5
  let d_new := 8
  let total_drawings := n_notebooks * p_pages * d_original
  let total_pages_new := total_drawings / d_new
  let filled_complete_pages := 6 * p_pages
  let drawings_on_last_page := total_drawings - filled_complete_pages * d_new - 40 * d_new
  drawings_on_last_page == 4 :=
  sorry

end drawings_on_last_page_l166_166720


namespace petals_per_ounce_l166_166654

-- Definitions of the given conditions
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes_harvested : ℕ := 800
def bottles_produced : ℕ := 20
def ounces_per_bottle : ℕ := 12

-- Calculation of petals per bush
def petals_per_bush : ℕ := roses_per_bush * petals_per_rose

-- Calculation of total petals harvested
def total_petals_harvested : ℕ := bushes_harvested * petals_per_bush

-- Calculation of total ounces of perfume
def total_ounces_produced : ℕ := bottles_produced * ounces_per_bottle

-- Main theorem statement
theorem petals_per_ounce : total_petals_harvested / total_ounces_produced = 320 :=
by
  sorry

end petals_per_ounce_l166_166654


namespace no_common_root_l166_166020

theorem no_common_root (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ¬∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x^2 + a * x + d = 0) := 
sorry

end no_common_root_l166_166020


namespace Andrey_Gleb_distance_l166_166252

theorem Andrey_Gleb_distance (AB VG : ℕ) (AG : ℕ) (BV : ℕ) (cond1 : AB = 600) (cond2 : VG = 600) (cond3 : AG = 3 * BV) :
  AG = 900 ∨ AG = 1800 := 
sorry

end Andrey_Gleb_distance_l166_166252


namespace smallest_possible_value_l166_166650

-- Definitions of the digits
def P := 1
def A := 9
def B := 2
def H := 8
def O := 3

-- Expression for continued fraction T
noncomputable def T : ℚ :=
  P + 1 / (A + 1 / (B + 1 / (H + 1 / O)))

-- The goal is to prove that T is the smallest possible value given the conditions
theorem smallest_possible_value : T = 555 / 502 :=
by
  -- The detailed proof would be done here, but for now we use sorry because we only need the statement
  sorry

end smallest_possible_value_l166_166650


namespace eval_expression_l166_166470

theorem eval_expression :
  6 - 9 * (1 / 2 - 3^3) * 2 = 483 := 
sorry

end eval_expression_l166_166470


namespace packed_tents_and_food_truck_arrangements_minimum_transportation_cost_l166_166899

-- Define the conditions
def total_items : ℕ := 320
def tents_more_than_food : ℕ := 80
def total_trucks : ℕ := 8
def type_A_tent_capacity : ℕ := 40
def type_A_food_capacity : ℕ := 10
def type_B_tent_capacity : ℕ := 20
def type_B_food_capacity : ℕ := 20
def type_A_cost : ℕ := 4000
def type_B_cost : ℕ := 3600

-- Questions to prove:
theorem packed_tents_and_food:
  ∃ t f : ℕ, t + f = total_items ∧ t = f + tents_more_than_food ∧ t = 200 ∧ f = 120 :=
sorry

theorem truck_arrangements:
  ∃ A B : ℕ, A + B = total_trucks ∧
    (A * type_A_tent_capacity + B * type_B_tent_capacity = 200) ∧
    (A * type_A_food_capacity + B * type_B_food_capacity = 120) ∧
    ((A = 2 ∧ B = 6) ∨ (A = 3 ∧ B = 5) ∨ (A = 4 ∧ B = 4)) :=
sorry

theorem minimum_transportation_cost:
  ∃ A B : ℕ, A = 2 ∧ B = 6 ∧ A * type_A_cost + B * type_B_cost = 29600 :=
sorry

end packed_tents_and_food_truck_arrangements_minimum_transportation_cost_l166_166899


namespace Liam_cycling_speed_l166_166009

theorem Liam_cycling_speed :
  ∀ (Eugene_speed Claire_speed Liam_speed : ℝ),
    Eugene_speed = 6 →
    Claire_speed = (3/4) * Eugene_speed →
    Liam_speed = (4/3) * Claire_speed →
    Liam_speed = 6 :=
by
  intros
  sorry

end Liam_cycling_speed_l166_166009


namespace problem_statement_l166_166656

variable {ι : Type*} {a : ι → ℝ} [decidable_eq ι]

theorem problem_statement 
  (h1 : ∀ i ∈ (finset.range 50), a i ≥ a (100 - i))
  (x : ℕ → ℝ) 
  (h2 : ∀ k ∈ (finset.range 99), x k = (k * a (k + 1)) / finset.sum (finset.range k) a)
  : finset.prod (finset.range 99) (λ k, x k ^ k) ≤ 1 := 
sorry

end problem_statement_l166_166656


namespace find_angle4_l166_166620

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180) 
  (h2 : angle3 = angle4) 
  (h3 : angle3 + angle4 = 70) :
  angle4 = 35 := 
by 
  sorry

end find_angle4_l166_166620


namespace landscape_breadth_l166_166398

theorem landscape_breadth (L B : ℝ) 
  (h1 : B = 6 * L) 
  (h2 : L * B = 29400) : 
  B = 420 :=
by
  sorry

end landscape_breadth_l166_166398


namespace derivative_correct_l166_166014

noncomputable def f (x : ℝ) : ℝ := 
  (1 / (2 * Real.sqrt 2)) * (Real.sin (Real.log x) - (Real.sqrt 2 - 1) * Real.cos (Real.log x)) * x^(Real.sqrt 2 + 1)

noncomputable def df (x : ℝ) : ℝ := 
  (x^(Real.sqrt 2)) / (2 * Real.sqrt 2) * (2 * Real.cos (Real.log x) - Real.sqrt 2 * Real.cos (Real.log x) + 2 * Real.sqrt 2 * Real.sin (Real.log x))

theorem derivative_correct (x : ℝ) (hx : 0 < x) :
  deriv f x = df x := by
  sorry

end derivative_correct_l166_166014


namespace gain_percentage_is_twenty_l166_166461

theorem gain_percentage_is_twenty (SP CP Gain : ℝ) (h0 : SP = 90) (h1 : Gain = 15) (h2 : SP = CP + Gain) : (Gain / CP) * 100 = 20 :=
by
  sorry

end gain_percentage_is_twenty_l166_166461


namespace factor_expression_l166_166925

theorem factor_expression (b : ℝ) : 275 * b^2 + 55 * b = 55 * b * (5 * b + 1) := by
  sorry

end factor_expression_l166_166925


namespace prob_two_segments_same_length_l166_166974

namespace hexagon_prob

noncomputable def prob_same_length : ℚ :=
  let total_elements : ℕ := 15
  let sides : ℕ := 6
  let diagonals : ℕ := 9
  (sides / total_elements) * ((sides - 1) / (total_elements - 1)) + (diagonals / total_elements) * ((diagonals - 1) / (total_elements - 1))

theorem prob_two_segments_same_length : prob_same_length = 17 / 35 :=
by
  sorry

end hexagon_prob

end prob_two_segments_same_length_l166_166974


namespace exists_large_cube_construction_l166_166082

theorem exists_large_cube_construction (n : ℕ) :
  ∃ N : ℕ, ∀ n > N, ∃ k : ℕ, k^3 = n :=
sorry

end exists_large_cube_construction_l166_166082


namespace utensils_in_each_pack_l166_166516

/-- Prove that given John needs to buy 5 packs to get 50 spoons
    and each pack contains an equal number of knives, forks, and spoons,
    the total number of utensils in each pack is 30. -/
theorem utensils_in_each_pack
  (packs : ℕ)
  (total_spoons : ℕ)
  (equal_parts : ∀ p : ℕ, p = total_spoons / packs)
  (knives forks spoons : ℕ)
  (equal_utensils : ∀ u : ℕ, u = spoons)
  (knives_forks : knives = forks)
  (knives_spoons : knives = spoons)
  (packs_needed : packs = 5)
  (total_utensils_needed : total_spoons = 50) :
  knives + forks + spoons = 30 := by
  sorry

end utensils_in_each_pack_l166_166516


namespace maximum_cookies_by_andy_l166_166269

-- Define the conditions
def total_cookies := 36
def cookies_by_andry (a : ℕ) := a
def cookies_by_alexa (a : ℕ) := 3 * a
def cookies_by_alice (a : ℕ) := 2 * a
def sum_cookies (a : ℕ) := cookies_by_andry a + cookies_by_alexa a + cookies_by_alice a

-- The theorem stating the problem and solution
theorem maximum_cookies_by_andy :
  ∃ a : ℕ, sum_cookies a = total_cookies ∧ a = 6 :=
by
  sorry

end maximum_cookies_by_andy_l166_166269


namespace arithmetic_sequence_a15_value_l166_166939

variables {a : ℕ → ℤ}

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a15_value
  (h1 : is_arithmetic_sequence a)
  (h2 : a 3 + a 13 = 20)
  (h3 : a 2 = -2) : a 15 = 24 :=
by sorry

end arithmetic_sequence_a15_value_l166_166939


namespace triangle_acute_angle_l166_166262

theorem triangle_acute_angle 
  (a b c : ℝ) 
  (h1 : a^3 = b^3 + c^3)
  (h2 : a > b)
  (h3 : a > c)
  (h4 : b > 0) 
  (h5 : c > 0) 
  (h6 : a > 0) 
  : 
  (a^2 < b^2 + c^2) :=
sorry

end triangle_acute_angle_l166_166262


namespace vip_seat_cost_l166_166759

theorem vip_seat_cost
  (V : ℝ)
  (G V_T : ℕ)
  (h1 : 20 * G + V * V_T = 7500)
  (h2 : G + V_T = 320)
  (h3 : V_T = G - 276) :
  V = 70 := by
sorry

end vip_seat_cost_l166_166759


namespace max_sum_of_factors_of_48_l166_166070

theorem max_sum_of_factors_of_48 : ∃ (heartsuit clubsuit : ℕ), heartsuit * clubsuit = 48 ∧ heartsuit + clubsuit = 49 :=
by
  -- We insert sorry here to skip the actual proof construction.
  sorry

end max_sum_of_factors_of_48_l166_166070


namespace new_mean_rent_l166_166129

theorem new_mean_rent
  (num_friends : ℕ)
  (avg_rent : ℕ)
  (original_rent_increased : ℕ)
  (increase_percentage : ℝ)
  (new_mean_rent : ℕ) :
  num_friends = 4 →
  avg_rent = 800 →
  original_rent_increased = 1400 →
  increase_percentage = 0.2 →
  new_mean_rent = 870 :=
by
  intros h1 h2 h3 h4
  sorry

end new_mean_rent_l166_166129


namespace factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l166_166107

theorem factorization_A (x y : ℝ) : x^2 - 2 * x * y = x * (x - 2 * y) :=
  by sorry

theorem factorization_B (x y : ℝ) : x^2 - 25 * y^2 = (x - 5 * y) * (x + 5 * y) :=
  by sorry

theorem factorization_C (x : ℝ) : 4 * x^2 - 4 * x + 1 = (2 * x - 1)^2 :=
  by sorry

theorem factorization_D_incorrect (x : ℝ) : x^2 + x - 2 ≠ (x - 2) * (x + 1) :=
  by sorry

theorem factorization_D_correct (x : ℝ) : x^2 + x - 2 = (x + 2) * (x - 1) :=
  by sorry

end factorization_A_factorization_B_factorization_C_factorization_D_incorrect_factorization_D_correct_l166_166107


namespace calculate_expression_l166_166818

theorem calculate_expression : 
  let x := 7.5
  let y := 2.5
  (x ^ y + Real.sqrt x + y ^ x) - (x ^ 2 + y ^ y + Real.sqrt y) = 679.2044 :=
by
  sorry

end calculate_expression_l166_166818


namespace negative_expression_P_minus_Q_l166_166448

theorem negative_expression_P_minus_Q :
  ∀ (P Q R S T : ℝ), 
    P = -4.0 → 
    Q = -2.0 → 
    R = 0.2 → 
    S = 1.1 → 
    T = 1.7 → 
    P - Q < 0 := 
by 
  intros P Q R S T hP hQ hR hS hT
  rw [hP, hQ]
  sorry

end negative_expression_P_minus_Q_l166_166448


namespace min_value_l166_166378

noncomputable def min_expression_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) : ℝ :=
  (x^2 + k*x + 1) * (y^2 + k*y + 1) * (z^2 + k*z + 1) / (x * y * z)

theorem min_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) :
  min_expression_value x y z k hx hy hz hk ≥ (2 + k)^3 :=
by
  sorry

end min_value_l166_166378


namespace indochina_no_hunters_l166_166142

universe u

noncomputable def problem : Prop :=
let People := Type u in
let Weight : People → ℝ := sorry in
let BornInIndochina : Set People := sorry in
let CollectsStamps : Set People := sorry in
let HuntsBears : Set People := sorry in
let E := {p : People | p ∈ BornInIndochina ∧ p ∉ HuntsBears} in
let A := {p : People | p ∈ E ∧ p ∈ CollectsStamps} in
let B := {p : People | p ∈ E ∧ p ∉ CollectsStamps} in
all_of (∃ q : People, Weight q < 100 ∧ q ∉ CollectsStamps → q ∉ BornInIndochina) → 
(∀ r : People, r ∈ BornInIndochina ∧ r ∈ CollectsStamps → r ∈ HuntsBears) →
(∀ s : People, s ∈ BornInIndochina → s ∈ HuntsBears ∨ s ∈ CollectsStamps) →
A = ∅ ∧ B = ∅

theorem indochina_no_hunters : problem := by
  sorry

end indochina_no_hunters_l166_166142


namespace work_on_monday_l166_166665

variable (Tuesday Wednesday Thursday Friday TotalHours Monday : ℚ)

axiom tuesday_hours : Tuesday = 1 / 2
axiom wednesday_hours : Wednesday = 2 / 3
axiom thursday_hours : Thursday = 5 / 6
axiom friday_hours : Friday = 75 / 60
axiom total_hours : TotalHours = 4

theorem work_on_monday :
  Monday = TotalHours - (Tuesday + Wednesday + Thursday + Friday) → Monday = 3 / 4 := sorry

end work_on_monday_l166_166665


namespace factorize_expression_simplify_fraction_expr_l166_166574

-- (1) Prove the factorization of m^3 - 4m^2 + 4m
theorem factorize_expression (m : ℝ) : 
  m^3 - 4 * m^2 + 4 * m = m * (m - 2)^2 :=
by
  sorry

-- (2) Simplify the fraction operation correctly
theorem simplify_fraction_expr (x : ℝ) (h : x ≠ 1) : 
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) :=
by
  sorry

end factorize_expression_simplify_fraction_expr_l166_166574


namespace ancient_chinese_poem_l166_166822

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) := by
  sorry

end ancient_chinese_poem_l166_166822


namespace range_of_a_for_quadratic_inequality_l166_166043

theorem range_of_a_for_quadratic_inequality (a : ℝ) :
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1 > 0) →
  (a ≤ -2 ∨ a ≥ 6) :=
by
  sorry

end range_of_a_for_quadratic_inequality_l166_166043


namespace papi_calot_additional_plants_l166_166389

def initial_plants := 7 * 18

def total_plants := 141

def additional_plants := total_plants - initial_plants

theorem papi_calot_additional_plants : additional_plants = 15 :=
by
  sorry

end papi_calot_additional_plants_l166_166389


namespace total_cost_fencing_l166_166407

/-
  Given conditions:
  1. Length of the plot (l) = 55 meters
  2. Length is 10 meters more than breadth (b): l = b + 10
  3. Cost of fencing per meter (cost_per_meter) = 26.50
  
  Prove that the total cost of fencing the plot is 5300 currency units.
-/
def length : ℕ := 55
def breadth : ℕ := length - 10
def cost_per_meter : ℝ := 26.50
def perimeter : ℕ := 2 * (length + breadth)
def total_cost : ℝ := cost_per_meter * perimeter

theorem total_cost_fencing : total_cost = 5300 := by
  sorry

end total_cost_fencing_l166_166407


namespace remaining_wire_in_cm_l166_166134

theorem remaining_wire_in_cm (total_mm : ℝ) (per_mobile_mm : ℝ) (conversion_factor : ℝ) :
  total_mm = 117.6 →
  per_mobile_mm = 4 →
  conversion_factor = 10 →
  ((total_mm % per_mobile_mm) / conversion_factor) = 0.16 :=
by
  intros htotal hmobile hconv
  sorry

end remaining_wire_in_cm_l166_166134


namespace simplify_trig_expression_l166_166864

theorem simplify_trig_expression :
  (2 - Real.sin 21 * Real.sin 21 - Real.cos 21 * Real.cos 21 + 
  (Real.sin 17 * Real.sin 17) * (Real.sin 17 * Real.sin 17) + 
  (Real.sin 17 * Real.sin 17) * (Real.cos 17 * Real.cos 17) + 
  (Real.cos 17 * Real.cos 17)) = 2 :=
by
  sorry

end simplify_trig_expression_l166_166864


namespace radius_of_circle_l166_166483

theorem radius_of_circle
  (r : ℝ) (r_pos : r > 0)
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1^2 + y1^2 = r^2)
  (h2 : x2^2 + y2^2 = r^2)
  (h3 : x1 + y1 = 3)
  (h4 : x2 + y2 = 3)
  (h5 : x1 * x2 + y1 * y2 = -0.5 * r^2) : 
  r = 3 * Real.sqrt 2 :=
by
  sorry

end radius_of_circle_l166_166483


namespace base_length_of_vessel_l166_166580

def volume_of_cube (edge : ℝ) := edge^3

def volume_of_displaced_water (L width rise : ℝ) := L * width * rise

theorem base_length_of_vessel (edge width rise L : ℝ) 
  (h1 : edge = 15) (h2 : width = 15) (h3 : rise = 11.25) 
  (h4 : volume_of_displaced_water L width rise = volume_of_cube edge) : 
  L = 20 :=
by
  sorry

end base_length_of_vessel_l166_166580


namespace calculation_error_l166_166061

def percentage_error (actual expected : ℚ) : ℚ :=
  (actual - expected) / expected * 100

theorem calculation_error :
  let correct_result := (5 / 3) * 3
  let incorrect_result := (5 / 3) / 3
  percentage_error incorrect_result correct_result = 88.89 := by
  sorry

end calculation_error_l166_166061


namespace marina_total_cost_l166_166662

theorem marina_total_cost (E P R X : ℕ) 
    (h1 : 15 + E + P = 47)
    (h2 : 15 + R + X = 58) :
    15 + E + P + R + X = 90 :=
by
  -- The proof will go here
  sorry

end marina_total_cost_l166_166662


namespace sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l166_166798

variable {α : Type*}

-- Part 1
theorem sin_A_sin_C_eq_3_over_4
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

-- Part 2
theorem triangle_is_equilateral
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  A = B ∧ B = C :=
sorry

end sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l166_166798


namespace mrs_wilsborough_tickets_l166_166850

theorem mrs_wilsborough_tickets :
  ∀ (saved vip_ticket_cost regular_ticket_cost vip_tickets left : ℕ),
    saved = 500 →
    vip_ticket_cost = 100 →
    regular_ticket_cost = 50 →
    vip_tickets = 2 →
    left = 150 →
    (saved - left - (vip_tickets * vip_ticket_cost)) / regular_ticket_cost = 3 :=
by
  intros saved vip_ticket_cost regular_ticket_cost vip_tickets left
  sorry

end mrs_wilsborough_tickets_l166_166850


namespace g_neg_9_equiv_78_l166_166521

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (y : ℝ) : ℝ := 3 * (y / 2 - 3 / 2)^2 + 4 * (y / 2 - 3 / 2) - 6

theorem g_neg_9_equiv_78 : g (-9) = 78 := by
  sorry

end g_neg_9_equiv_78_l166_166521


namespace profit_percentage_is_ten_l166_166909

-- Definitions based on conditions
def cost_price := 500
def selling_price := 550

-- Defining the profit percentage
def profit := selling_price - cost_price
def profit_percentage := (profit / cost_price) * 100

-- The proof that the profit percentage is 10
theorem profit_percentage_is_ten : profit_percentage = 10 :=
by
  -- Using the definitions given
  sorry

end profit_percentage_is_ten_l166_166909


namespace sin_double_angle_identity_l166_166191

open Real

theorem sin_double_angle_identity (α : ℝ) (h : sin (α - π / 4) = 3 / 5) : sin (2 * α) = 7 / 25 :=
by
  sorry

end sin_double_angle_identity_l166_166191


namespace negation_proof_l166_166545

def P (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

theorem negation_proof : (¬(∀ x : ℝ, P x)) ↔ (∃ x : ℝ, ¬(P x)) :=
by sorry

end negation_proof_l166_166545


namespace revenue_from_full_price_tickets_l166_166756

theorem revenue_from_full_price_tickets (f h p : ℕ) (H1 : f + h = 150) (H2 : f * p + h * (p / 2) = 2450) : 
  f * p = 1150 :=
by 
  sorry

end revenue_from_full_price_tickets_l166_166756


namespace abs_inequality_k_ge_neg3_l166_166931

theorem abs_inequality_k_ge_neg3 (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k ≥ -3 :=
sorry

end abs_inequality_k_ge_neg3_l166_166931


namespace spending_on_games_l166_166633

-- Definitions converted from conditions
def totalAllowance := 48
def fractionClothes := 1 / 4
def fractionBooks := 1 / 3
def fractionSnacks := 1 / 6
def spentClothes := fractionClothes * totalAllowance
def spentBooks := fractionBooks * totalAllowance
def spentSnacks := fractionSnacks * totalAllowance
def spentGames := totalAllowance - (spentClothes + spentBooks + spentSnacks)

-- The theorem that needs to be proven
theorem spending_on_games : spentGames = 12 :=
by sorry

end spending_on_games_l166_166633


namespace max_lights_correct_l166_166325

def max_lights_on (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2

theorem max_lights_correct (n : ℕ) :
  max_lights_on n = if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2 :=
by sorry

end max_lights_correct_l166_166325


namespace simplify_expression_calculate_difference_of_squares_l166_166782

section Problem1
variable (a b : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem simplify_expression : ((-2 * a^2) ^ 2 * (-b^2)) / (4 * a^3 * b^2) = -a :=
by sorry
end Problem1

section Problem2

theorem calculate_difference_of_squares : 2023^2 - 2021 * 2025 = 4 :=
by sorry
end Problem2

end simplify_expression_calculate_difference_of_squares_l166_166782


namespace coffee_consumption_l166_166696

variables (h w g : ℝ)

theorem coffee_consumption (k : ℝ) 
  (H1 : ∀ h w g, h * g = k * w)
  (H2 : h = 8 ∧ g = 4.5 ∧ w = 2)
  (H3 : h = 4 ∧ w = 3) : g = 13.5 :=
by {
  sorry
}

end coffee_consumption_l166_166696


namespace total_acorns_proof_l166_166858

variable (x y : ℝ)

def total_acorns (x y : ℝ) : ℝ :=
  let shawna := x
  let sheila := 5.3 * x
  let danny := 5.3 * x + y
  let ella := 2 * (4.3 * x + y)
  shawna + sheila + danny + ella

theorem total_acorns_proof (x y : ℝ) :
  total_acorns x y = 20.2 * x + 3 * y :=
by
  unfold total_acorns
  sorry

end total_acorns_proof_l166_166858


namespace product_of_first_three_terms_is_960_l166_166258

-- Definitions from the conditions
def a₁ : ℤ := 20 - 6 * 2
def a₂ : ℤ := a₁ + 2
def a₃ : ℤ := a₂ + 2

-- Problem statement
theorem product_of_first_three_terms_is_960 : 
  a₁ * a₂ * a₃ = 960 :=
by
  sorry

end product_of_first_three_terms_is_960_l166_166258


namespace equation_of_line_l166_166618

theorem equation_of_line :
  ∃ m : ℝ, ∀ x y : ℝ, (y = m * x - m ∧ (m = 2 ∧ x = 1 ∧ y = 0)) ∧ 
  ∀ x : ℝ, ¬(4 * x^2 - (m * x - m)^2 - 8 * x = 12) → m = 2 → y = 2 * x - 2 :=
by sorry

end equation_of_line_l166_166618


namespace volume_of_cone_l166_166207

noncomputable def lateral_surface_area : ℝ := 8 * Real.pi

theorem volume_of_cone (l r h : ℝ)
  (h_lateral_surface : l * Real.pi = 2 * lateral_surface_area)
  (h_radius : l = 2 * r)
  (h_height : h = Real.sqrt (l^2 - r^2)) :
  (1/3) * Real.pi * r^2 * h = (8 * Real.sqrt 3 * Real.pi) / 3 :=
by
  sorry

end volume_of_cone_l166_166207


namespace time_for_B_alone_to_complete_work_l166_166750

theorem time_for_B_alone_to_complete_work :
  (∃ (A B C : ℝ), A = 1 / 4 
  ∧ B + C = 1 / 3
  ∧ A + C = 1 / 2)
  → 1 / B = 12 :=
by
  sorry

end time_for_B_alone_to_complete_work_l166_166750


namespace reciprocal_of_neg_three_l166_166698

theorem reciprocal_of_neg_three : ∃ (x : ℚ), (-3 * x = 1) ∧ (x = -1 / 3) :=
by
  use (-1 / 3)
  split
  . rw [mul_comm]
    norm_num 
  . norm_num

end reciprocal_of_neg_three_l166_166698


namespace simplify_expression_l166_166680

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l166_166680


namespace polygon_sides_eq_six_l166_166351

theorem polygon_sides_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = (2 * 360)) 
  (h2 : exterior_sum = 360) :
  n = 6 := 
by
  sorry

end polygon_sides_eq_six_l166_166351


namespace mod_remainder_of_sum_of_primes_l166_166377

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def sum_of_odd_primes : ℕ := List.sum odd_primes_less_than_32

theorem mod_remainder_of_sum_of_primes : sum_of_odd_primes % 32 = 30 := by
  sorry

end mod_remainder_of_sum_of_primes_l166_166377


namespace product_first_three_terms_arithmetic_seq_l166_166261

theorem product_first_three_terms_arithmetic_seq :
  ∀ (a₇ d : ℤ), 
  a₇ = 20 → d = 2 → 
  let a₁ := a₇ - 6 * d in
  let a₂ := a₁ + d in
  let a₃ := a₂ + d in
  a₁ * a₂ * a₃ = 960 := 
by
  intros a₇ d a₇_20 d_2
  let a₁ := a₇ - 6 * d
  let a₂ := a₁ + d
  let a₃ := a₂ + d
  sorry

end product_first_three_terms_arithmetic_seq_l166_166261


namespace even_function_derivative_zero_l166_166811

variable {ℝ : Type*} [LinearOrderedField ℝ] {f : ℝ → ℝ}

theorem even_function_derivative_zero (h_even : ∀ x, f x = f (-x)) 
  (h_deriv : ∃ f', ∀ x, deriv f x = f' x) : deriv f 0 = 0 :=
by
  sorry

end even_function_derivative_zero_l166_166811


namespace height_difference_petronas_empire_state_l166_166531

theorem height_difference_petronas_empire_state :
  let esb_height := 443
  let pt_height := 452
  pt_height - esb_height = 9 := by
  sorry

end height_difference_petronas_empire_state_l166_166531


namespace samantha_born_in_1979_l166_166405

-- Condition definitions
def first_AMC8_year := 1985
def annual_event (n : ℕ) : ℕ := first_AMC8_year + n
def seventh_AMC8_year := annual_event 6

variable (Samantha_age_in_seventh_AMC8 : ℕ)
def Samantha_age_when_seventh_AMC8 := 12
def Samantha_birth_year := seventh_AMC8_year - Samantha_age_when_seventh_AMC8

-- Proof statement
theorem samantha_born_in_1979 : Samantha_birth_year = 1979 :=
by
  sorry

end samantha_born_in_1979_l166_166405


namespace two_digit_number_conditions_l166_166733

-- Definitions for two-digit number and its conditions
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

-- The proof problem statement in Lean 4
theorem two_digit_number_conditions (N : ℕ) (c d : ℕ) :
  is_two_digit_number N ∧ N = 10 * c + d ∧ N' = N + 7 ∧ 
  N = 6 * sum_of_digits (N + 7) →
  N = 24 ∨ N = 78 :=
by
  sorry

end two_digit_number_conditions_l166_166733


namespace simplify_and_evaluate_l166_166245

theorem simplify_and_evaluate (a : ℝ) (h₁ : a^2 - 4 * a + 3 = 0) (h₂ : a ≠ 3) : 
  ( (a^2 - 9) / (a^2 - 3 * a) / ( (a^2 + 9) / a + 6 ) = 1 / 4 ) :=
by 
  sorry

end simplify_and_evaluate_l166_166245


namespace pages_written_in_a_year_l166_166826

-- Definitions based on conditions
def pages_per_letter : ℕ := 3
def letters_per_week : ℕ := 2
def friends : ℕ := 2
def weeks_per_year : ℕ := 52

-- Definition to calculate total pages written in a week
def weekly_pages (pages_per_letter : ℕ) (letters_per_week : ℕ) (friends : ℕ) : ℕ :=
  pages_per_letter * letters_per_week * friends

-- Definition to calculate total pages written in a year
def yearly_pages (weekly_pages : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weekly_pages * weeks_per_year

-- Theorem to prove the total pages written in a year
theorem pages_written_in_a_year : yearly_pages (weekly_pages pages_per_letter letters_per_week friends) weeks_per_year = 624 :=
by 
  sorry

end pages_written_in_a_year_l166_166826


namespace cube_surface_area_increase_l166_166735

theorem cube_surface_area_increase (s : ℝ) :
  let original_surface_area := 6 * s^2
  let new_side_length := 1.8 * s
  let new_surface_area := 6 * (new_side_length)^2
  let percentage_increase := (new_surface_area - original_surface_area) / original_surface_area * 100
  percentage_increase = 1844 :=
by
  unfold original_surface_area
  unfold new_side_length
  unfold new_surface_area
  unfold percentage_increase
  sorry

end cube_surface_area_increase_l166_166735


namespace store_second_reduction_percentage_l166_166452

theorem store_second_reduction_percentage (P : ℝ) :
  let first_reduction := 0.88 * P
  let second_reduction := 0.792 * P
  ∃ R : ℝ, (1 - R) * first_reduction = second_reduction ∧ R = 0.1 :=
by
  let first_reduction := 0.88 * P
  let second_reduction := 0.792 * P
  use 0.1
  sorry

end store_second_reduction_percentage_l166_166452


namespace _l166_166179

lemma right_triangle_angles (AB BC AC : ℝ) (α β : ℝ)
  (h1 : AB = 1) 
  (h2 : BC = Real.sin α)
  (h3 : AC = Real.cos α)
  (h4 : AB^2 = BC^2 + AC^2) -- Pythagorean theorem for the right triangle
  (h5 : α = (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1))) :
  β = 90 - (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1)) :=
sorry

end _l166_166179


namespace multiple_of_9_l166_166343

theorem multiple_of_9 (x : ℕ) (hx1 : ∃ k : ℕ, x = 9 * k) (hx2 : x^2 > 80) (hx3 : x < 30) : x = 9 ∨ x = 18 ∨ x = 27 :=
sorry

end multiple_of_9_l166_166343


namespace prob_same_color_is_correct_l166_166177

noncomputable def prob_same_color : ℚ :=
  let green_prob := (8 : ℚ) / 10
  let red_prob := (2 : ℚ) / 10
  (green_prob)^2 + (red_prob)^2

theorem prob_same_color_is_correct :
  prob_same_color = 17 / 25 := by
  sorry

end prob_same_color_is_correct_l166_166177


namespace simplify_expression_l166_166675

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l166_166675


namespace inf_mse_conditional_expectation_l166_166523

open MeasureTheory

noncomputable theory

variables {Ω : Type*} [MeasureSpace Ω] (ξ η : Ω → ℝ)
variable (f : Ω → ℝ)
variable [Integrable η] 
variable [Integrable ξ]

theorem inf_mse_conditional_expectation (f_star : Ω → ℝ)
  (h_star : f_star = λ ω, 𝔼[η | ξ ω]) : 
  (⨅ (f : Ω → ℝ), 𝔼[λ ω, (η ω - f (ξ ω))^2]) = 𝔼[λ ω, (η ω - 𝔼[η | ξ ω])^2] :=
by
  sorry

end inf_mse_conditional_expectation_l166_166523


namespace initial_people_in_gym_l166_166776

variable (W A : ℕ)

theorem initial_people_in_gym (W A : ℕ) (h : W + A + 5 + 2 - 3 - 4 + 2 = 20) : W + A = 18 := by
  sorry

end initial_people_in_gym_l166_166776


namespace rollins_ratio_l166_166723

noncomputable def proof_problem : Prop :=
  let johnson_amount := 2300
  let sutton_amount := (2300 : ℚ) / 2
  let rollins_amount := sutton_amount * 8
  let total_after_fees := 27048
  let total_school_raised := total_after_fees / 0.98
  (rollins_amount / total_school_raised ≈ 1 / 3)

theorem rollins_ratio : proof_problem :=
by
  sorry

end rollins_ratio_l166_166723


namespace min_value_expression_l166_166832

theorem min_value_expression (α β : ℝ) :
  ∃ x y, x = 3 * Real.cos α + 6 * Real.sin β ∧
         y = 3 * Real.sin α + 6 * Real.cos β ∧
         (x - 10)^2 + (y - 18)^2 = 121 :=
by
  sorry

end min_value_expression_l166_166832


namespace part1_l166_166333

-- Define the arithmetic progression and geometric progression sequences 
structure arith_seq (a : ℕ → ℕ) (d : ℕ) :=
(arith_prop : ∀ n : ℕ, a (n + 1) = a n + d)

structure geom_seq (b : ℕ → ℕ) (r : ℕ) :=
(geom_prop : ∀ n : ℕ, b (n + 1) = b n * r)

-- Conditions
variables (a : ℕ → ℕ) (b : ℕ → ℕ) (d : ℕ)
variables [arith_seq a d] [geom_seq b 2]

-- Given conditions
axiom cond1 : a 1 + d - 2 * b 1 = a 1 + 2 * d - 4 * b 1
axiom cond2 : a 1 + d - 2 * b 1 = 8 * b 1 - (a 1 + 3 * d)

-- Part (1) Proof
theorem part1 : a 1 = b 1 :=
by sorry

-- Part (2) Proof
noncomputable def num_elements : ℕ :=
  let m_values := {m : ℕ | 1 ≤ m ∧ m ≤ 500}
  let valid_k := {k : ℕ | 2 ≤ k ∧ k ≤ 10} in
  if ∃ k : ℕ, k ∈ valid_k then 9 else 0

#eval num_elements

end part1_l166_166333


namespace trig_identity_l166_166875

theorem trig_identity :
  (Real.cos (105 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.sin (105 * Real.pi / 180)) = 1 / 2 :=
  sorry

end trig_identity_l166_166875


namespace cubic_of_m_eq_4_l166_166501

theorem cubic_of_m_eq_4 (m : ℕ) (h : 3 ^ m = 81) : m ^ 3 = 64 := 
by
  sorry

end cubic_of_m_eq_4_l166_166501


namespace min_odd_integers_l166_166883

theorem min_odd_integers (a b c d e f g h i : ℤ)
  (h1 : a + b + c = 30)
  (h2 : a + b + c + d + e + f = 48)
  (h3 : a + b + c + d + e + f + g + h + i = 69) :
  ∃ k : ℕ, k = 1 ∧
  (∃ (aa bb cc dd ee ff gg hh ii : ℤ), (fun (x : ℤ) => x % 2 = 1 → k = 1) (aa + bb + cc + dd + ee + ff + gg + hh + ii)) :=
by
  intros
  sorry

end min_odd_integers_l166_166883


namespace gcd_102_238_l166_166725

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

theorem gcd_102_238 : gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l166_166725


namespace avg_remaining_two_l166_166087

-- Defining the given conditions
variable (six_num_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ)

-- Defining the known values
axiom avg_val : six_num_avg = 3.95
axiom avg_group1 : group1_avg = 3.6
axiom avg_group2 : group2_avg = 3.85

-- Stating the problem to prove that the average of the remaining 2 numbers is 4.4
theorem avg_remaining_two (h : six_num_avg = 3.95) 
                           (h1: group1_avg = 3.6)
                           (h2: group2_avg = 3.85) : 
  4.4 = ((six_num_avg * 6) - (group1_avg * 2 + group2_avg * 2)) / 2 := 
sorry

end avg_remaining_two_l166_166087


namespace krikor_speed_increase_l166_166238

/--
Krikor traveled to work on two consecutive days, Monday and Tuesday, at different speeds.
Both days, he covered the same distance. On Monday, he traveled for 0.5 hours, and on
Tuesday, he traveled for \( \frac{5}{12} \) hours. Prove that the percentage increase in his speed 
from Monday to Tuesday is 20%.
-/
theorem krikor_speed_increase :
  ∀ (v1 v2 : ℝ), (0.5 * v1 = (5 / 12) * v2) → (v2 = (6 / 5) * v1) → 
  ((v2 - v1) / v1 * 100 = 20) :=
by
  -- Proof goes here
  sorry

end krikor_speed_increase_l166_166238


namespace max_on_bulbs_l166_166324

theorem max_on_bulbs (n : ℕ) : 
  (∃ k : ℕ, k = n / 2 ∧ (n % 2 = 0 → max_on_bulbs_count n = n^2 / 2) ∧ (n % 2 = 1 → max_on_bulbs_count n = (n^2 - 1) / 2)) :=
by
  sorry

def max_on_bulbs_count (n : ℕ) : ℕ :=
if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2

end max_on_bulbs_l166_166324


namespace tan_of_sin_in_interval_l166_166934

theorem tan_of_sin_in_interval (α : ℝ) (h1 : Real.sin α = 4 / 5) (h2 : 0 < α ∧ α < Real.pi) :
  Real.tan α = 4 / 3 ∨ Real.tan α = -4 / 3 :=
  sorry

end tan_of_sin_in_interval_l166_166934


namespace discount_percentage_correct_l166_166855

-- Define the problem parameters as variables
variables (sale_price marked_price : ℝ) (discount_percentage : ℝ)

-- Provide the conditions from the problem
def conditions : Prop :=
  sale_price = 147.60 ∧ marked_price = 180

-- State the problem: Prove the discount percentage is 18%
theorem discount_percentage_correct (h : conditions sale_price marked_price) : 
  discount_percentage = 18 :=
by
  sorry

end discount_percentage_correct_l166_166855


namespace book_distribution_ways_l166_166585

theorem book_distribution_ways : 
  ∃ n : ℕ, n = 7 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 →
  ∃ l : ℕ, l + (8 - l) = 8 ∧ 1 ≤ l ∧ 1 ≤ 8 - l :=
by
  -- We will provide a proof here.
  sorry

end book_distribution_ways_l166_166585


namespace rachel_remaining_pictures_l166_166572

theorem rachel_remaining_pictures 
  (p1 p2 p_colored : ℕ)
  (h1 : p1 = 23)
  (h2 : p2 = 32)
  (h3 : p_colored = 44) :
  (p1 + p2 - p_colored = 11) :=
by
  sorry

end rachel_remaining_pictures_l166_166572


namespace paint_cost_of_cube_l166_166402

theorem paint_cost_of_cube (side_length cost_per_kg coverage_per_kg : ℝ) (h₀ : side_length = 10) 
(h₁ : cost_per_kg = 60) (h₂ : coverage_per_kg = 20) : 
(cost_per_kg * (6 * (side_length^2) / coverage_per_kg) = 1800) :=
by
  sorry

end paint_cost_of_cube_l166_166402


namespace wire_division_l166_166208

theorem wire_division (initial_length : ℝ) (num_parts : ℕ) (final_length : ℝ) :
  initial_length = 69.76 ∧ num_parts = 8 ∧
  final_length = (initial_length / num_parts) / num_parts →
  final_length = 1.09 :=
by
  sorry

end wire_division_l166_166208


namespace only_valid_pairs_l166_166475

theorem only_valid_pairs (a b : ℕ) (h₁ : a ≥ 1) (h₂ : b ≥ 1) :
  a^b^2 = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) :=
by
  sorry

end only_valid_pairs_l166_166475


namespace division_by_reciprocal_l166_166050

theorem division_by_reciprocal :
  (10 / 3) / (1 / 5) = 50 / 3 := 
sorry

end division_by_reciprocal_l166_166050


namespace find_g_neg_three_l166_166522

namespace ProofProblem

def g (d e f x : ℝ) : ℝ := d * x^5 + e * x^3 + f * x + 6

theorem find_g_neg_three (d e f : ℝ) (h : g d e f 3 = -9) : g d e f (-3) = 21 := by
  sorry

end ProofProblem

end find_g_neg_three_l166_166522


namespace count_three_digit_odd_increasing_order_l166_166632

theorem count_three_digit_odd_increasing_order : 
  ∃ n : ℕ, n = 10 ∧
  ∀ a b c : ℕ, (100 * a + 10 * b + c) % 2 = 1 ∧ a < b ∧ b < c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 → 
    (100 * a + 10 * b + c) % 2 = 1 := 
sorry

end count_three_digit_odd_increasing_order_l166_166632


namespace simplify_expression_l166_166681

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l166_166681


namespace find_a_l166_166802

noncomputable def tangent_condition (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 : ℝ) = (1 / (x₀ + a))

theorem find_a : ∃ a : ℝ, tangent_condition a ∧ a = 2 :=
by
  sorry

end find_a_l166_166802


namespace reciprocal_of_neg_three_l166_166700

theorem reciprocal_of_neg_three : ∃ (x : ℚ), (-3 * x = 1) ∧ (x = -1 / 3) :=
by
  use (-1 / 3)
  split
  . rw [mul_comm]
    norm_num 
  . norm_num

end reciprocal_of_neg_three_l166_166700


namespace vlad_taller_than_sister_l166_166274

-- Definitions based on the conditions
def vlad_feet : ℕ := 6
def vlad_inches : ℕ := 3
def sister_feet : ℕ := 2
def sister_inches : ℕ := 10
def inches_per_foot : ℕ := 12

-- Derived values for heights in inches
def vlad_height_in_inches : ℕ := (vlad_feet * inches_per_foot) + vlad_inches
def sister_height_in_inches : ℕ := (sister_feet * inches_per_foot) + sister_inches

-- Lean 4 statement for the proof problem
theorem vlad_taller_than_sister : vlad_height_in_inches - sister_height_in_inches = 41 := 
by 
  sorry

end vlad_taller_than_sister_l166_166274


namespace Ryan_stickers_l166_166856

def Ryan_has_30_stickers (R S T : ℕ) : Prop :=
  S = 3 * R ∧ T = S + 20 ∧ R + S + T = 230 → R = 30

theorem Ryan_stickers : ∃ R S T : ℕ, Ryan_has_30_stickers R S T :=
sorry

end Ryan_stickers_l166_166856


namespace mixed_nuts_price_l166_166299

theorem mixed_nuts_price (total_weight : ℝ) (peanut_price : ℝ) (cashew_price : ℝ) (cashew_weight : ℝ) 
  (H1 : total_weight = 100) 
  (H2 : peanut_price = 3.50) 
  (H3 : cashew_price = 4.00) 
  (H4 : cashew_weight = 60) : 
  (cashew_weight * cashew_price + (total_weight - cashew_weight) * peanut_price) / total_weight = 3.80 :=
by 
  sorry

end mixed_nuts_price_l166_166299


namespace melissa_work_hours_l166_166845

variable (f : ℝ) (f_d : ℝ) (h_d : ℝ)

theorem melissa_work_hours (hf : f = 56) (hfd : f_d = 4) (hhd : h_d = 3) : 
  (f / f_d) * h_d = 42 := by
  sorry

end melissa_work_hours_l166_166845


namespace three_term_arithmetic_seq_l166_166171

noncomputable def arithmetic_sequence_squares (x y z : ℤ) : Prop :=
  x^2 + z^2 = 2 * y^2

theorem three_term_arithmetic_seq (x y z : ℤ) :
  (∃ a b : ℤ, a = (x + z) / 2 ∧ b = (x - z) / 2 ∧ x^2 + z^2 = 2 * y^2) ↔
  arithmetic_sequence_squares x y z :=
by
  sorry

end three_term_arithmetic_seq_l166_166171


namespace arccos_neg_one_l166_166155

theorem arccos_neg_one : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_l166_166155


namespace winnie_proof_l166_166439

def winnie_problem : Prop :=
  let initial_count := 2017
  let multiples_of_3 := initial_count / 3
  let multiples_of_6 := initial_count / 6
  let multiples_of_27 := initial_count / 27
  let multiples_to_erase_3 := multiples_of_3
  let multiples_to_reinstate_6 := multiples_of_6
  let multiples_to_erase_27 := multiples_of_27
  let final_count := initial_count - multiples_to_erase_3 + multiples_to_reinstate_6 - multiples_to_erase_27
  initial_count - final_count = 373

theorem winnie_proof : winnie_problem := by
  sorry

end winnie_proof_l166_166439


namespace percentage_of_400_that_results_in_224_point_5_l166_166611

-- Let x be the unknown percentage of 400
variable (x : ℝ)

-- Condition: x% of 400 plus 45% of 250 equals 224.5
def condition (x : ℝ) : Prop := (400 * x / 100) + (250 * 45 / 100) = 224.5

theorem percentage_of_400_that_results_in_224_point_5 : condition 28 :=
by
  -- proof goes here
  sorry

end percentage_of_400_that_results_in_224_point_5_l166_166611


namespace katy_summer_reading_l166_166829

theorem katy_summer_reading :
  let b_June := 8 in
  let b_July := 2 * b_June in
  let b_August := b_July - 3 in
  b_June + b_July + b_August = 37 :=
by
  sorry

end katy_summer_reading_l166_166829


namespace lara_flowers_in_vase_l166_166221

theorem lara_flowers_in_vase:
  ∀ (total_stems mom_flowers extra_flowers: ℕ),
  total_stems = 52 →
  mom_flowers = 15 →
  extra_flowers = 6 →
  let grandma_flowers := mom_flowers + extra_flowers in
  let given_away := mom_flowers + grandma_flowers in
  let in_vase := total_stems - given_away in
  in_vase = 16 :=
by
  intros total_stems mom_flowers extra_flowers
  intros h1 h2 h3
  let grandma_flowers := mom_flowers + extra_flowers
  let given_away := mom_flowers + grandma_flowers
  let in_vase := total_stems - given_away
  rw [h1, h2, h3]
  exact sorry

end lara_flowers_in_vase_l166_166221


namespace M_intersect_N_l166_166117

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def intersection (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∈ N}

theorem M_intersect_N :
  intersection M N = {x | 1 ≤ x ∧ x < 2} := 
sorry

end M_intersect_N_l166_166117


namespace base_conversion_arithmetic_l166_166463

theorem base_conversion_arithmetic :
  let b5 := 2013
  let b3 := 11
  let b6 := 3124
  let b7 := 4321
  (b5₅ / b3₃ - b6₆ + b7₇ : ℝ) = 898.5 :=
by sorry

end base_conversion_arithmetic_l166_166463


namespace determine_X_with_7_gcd_queries_l166_166126

theorem determine_X_with_7_gcd_queries : 
  ∀ (X : ℕ), (X ≤ 100) → ∃ (f : Fin 7 → ℕ × ℕ), 
    (∀ i, (f i).1 < 100 ∧ (f i).2 < 100) ∧ (∃ (Y : Fin 7 → ℕ), 
      (∀ i, Y i = Nat.gcd (X + (f i).1) (f i).2) → 
        (∀ (X' : ℕ), (X' ≤ 100) → ((∀ i, Y i = Nat.gcd (X' + (f i).1) (f i).2) → X' = X))) :=
sorry

end determine_X_with_7_gcd_queries_l166_166126


namespace contractor_job_completion_l166_166289

theorem contractor_job_completion 
  (total_days : ℕ := 100) 
  (initial_workers : ℕ := 10) 
  (days_worked_initial : ℕ := 20) 
  (fraction_completed_initial : ℚ := 1/4) 
  (fired_workers : ℕ := 2) 
  : ∀ (remaining_days : ℕ), remaining_days = 75 → (remaining_days + days_worked_initial = 95) :=
by
  sorry

end contractor_job_completion_l166_166289


namespace average_speed_l166_166139

theorem average_speed :
  ∀ (initial_odometer final_odometer total_time : ℕ), 
    initial_odometer = 2332 →
    final_odometer = 2772 →
    total_time = 8 →
    (final_odometer - initial_odometer) / total_time = 55 :=
by
  intros initial_odometer final_odometer total_time h_initial h_final h_time
  sorry

end average_speed_l166_166139


namespace reciprocal_of_neg_three_l166_166705

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l166_166705


namespace greatest_possible_perimeter_l166_166821

theorem greatest_possible_perimeter (a b c : ℕ) 
    (h₁ : a = 4 * b ∨ b = 4 * a ∨ c = 4 * a ∨ c = 4 * b)
    (h₂ : a = 18 ∨ b = 18 ∨ c = 18)
    (triangle_ineq : a + b > c ∧ a + c > b ∧ b + c > a) :
    a + b + c = 43 :=
by {
  sorry
}

end greatest_possible_perimeter_l166_166821


namespace inequality_proof_l166_166977

theorem inequality_proof 
  (a b c x y z : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h_sum : a + b + c = 1) : 
  (x^2 + y^2 + z^2) * 
  (a^3 / (x^2 + 2 * y^2) + b^3 / (y^2 + 2 * z^2) + c^3 / (z^2 + 2 * x^2)) 
  ≥ 1 / 9 := 
by 
  sorry

end inequality_proof_l166_166977


namespace blue_pill_cost_l166_166151

theorem blue_pill_cost :
  ∃ y : ℝ, ∀ (red_pill_cost blue_pill_cost : ℝ),
    (blue_pill_cost = red_pill_cost + 2) ∧
    (21 * (blue_pill_cost + red_pill_cost) = 819) →
    blue_pill_cost = 20.5 :=
by sorry

end blue_pill_cost_l166_166151


namespace aeroplane_speed_l166_166456

theorem aeroplane_speed (D : ℝ) (S : ℝ) (h1 : D = S * 6) (h2 : D = 540 * (14 / 3)) :
  S = 420 := by
  sorry

end aeroplane_speed_l166_166456


namespace attendance_rate_correct_l166_166922

def total_students : ℕ := 50
def students_on_leave : ℕ := 2
def given_attendance_rate : ℝ := 96

theorem attendance_rate_correct :
  ((total_students - students_on_leave) / total_students * 100 : ℝ) = given_attendance_rate := sorry

end attendance_rate_correct_l166_166922


namespace total_fencing_cost_l166_166410

-- Conditions
def length : ℝ := 55
def cost_per_meter : ℝ := 26.50

-- We derive breadth from the given conditions
def breadth : ℝ := length - 10

-- Calculate the perimeter of the rectangular plot
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost of fencing the plot
def total_cost : ℝ := cost_per_meter * perimeter

-- The theorem to prove that total cost is equal to 5300
theorem total_fencing_cost : total_cost = 5300 := by
  -- Calculation goes here
  sorry

end total_fencing_cost_l166_166410


namespace part1_part2_part3_l166_166323

section Part1
variable (a b : ℕ) (q r : ℕ)

theorem part1 (h : a = 2011) (h2 : b = 91) (hq : q = 22) (hr : r = 9) : 
  a = b * q + r := by
  simp [h, h2, hq, hr]
  sorry
end Part1

section Part2
variable (A : Finset ℕ) (f : ℕ → ℕ)

theorem part2 (hA : A = {1, 2, ..., 23}) (hf : ∀ x1 x2 ∈ A, |x1 - x2| ∈ {1, 2, 3} → f x1 ≠ f x2) : 
  False := by
  sorry
end Part2

section Part3
variable (A : Finset ℕ) (B : Finset ℕ) (m : ℕ)

def is_harmonic (B : Finset ℕ) : Prop := ∃ a b ∈ B, b < a ∧ b ∣ a

theorem part3 (hA : A = {1, 2, ..., 23}) (hB : B ⊆ A) (cardB : B.card = 12) (hm : m = 7)
  (hH : ∀ B ⊆ A, cardB = 12 → m ∈ B → is_harmonic B) : 
  True := by
  sorry

end Part3

end part1_part2_part3_l166_166323


namespace max_tickets_l166_166019

-- Define the conditions
def ticket_cost (n : ℕ) : ℝ :=
  if n ≤ 6 then 15 * n
  else 13.5 * n

-- Define the main theorem
theorem max_tickets (budget : ℝ) : (∀ n : ℕ, ticket_cost n ≤ budget) → budget = 120 → n ≤ 8 :=
  by
  sorry

end max_tickets_l166_166019


namespace profitable_year_exists_option2_more_economical_l166_166774

noncomputable def total_expenses (x : ℕ) : ℝ := 2 * (x:ℝ)^2 + 10 * x  

noncomputable def annual_income (x : ℕ) : ℝ := 50 * x  

def year_profitable (x : ℕ) : Prop := annual_income x > total_expenses x + 98 / 1000

theorem profitable_year_exists : ∃ x : ℕ, year_profitable x ∧ x = 3 := sorry

noncomputable def total_profit (x : ℕ) : ℝ := 
  50 * x - 2 * (x:ℝ)^2 + 10 * x - 98 / 1000 + if x = 10 then 8 else if x = 7 then 26 else 0

theorem option2_more_economical : 
  total_profit 10 = 110 ∧ total_profit 7 = 110 ∧ 7 < 10 :=
sorry

end profitable_year_exists_option2_more_economical_l166_166774


namespace drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l166_166963

-- Definitions for the initial conditions
def initial_white_balls := 2
def initial_black_balls := 3
def initial_red_balls := 5
def total_balls := initial_white_balls + initial_black_balls + initial_red_balls

-- Statement for part 1: Drawing a red ball is a random event
theorem drawing_red_ball_random : (initial_red_balls > 0) := by
  sorry

-- Statement for part 1: Drawing a yellow ball is impossible
theorem drawing_yellow_ball_impossible : (0 = 0) := by
  sorry

-- Statement for part 2: Probability of drawing a black ball
theorem probability_black_ball : (initial_black_balls : ℚ) / total_balls = 3 / 10 := by
  sorry

-- Definitions for the conditions in part 3
def additional_black_balls (x : ℕ) := initial_black_balls + x
def new_total_balls (x : ℕ) := total_balls + x

-- Statement for part 3: Finding the number of additional black balls
theorem number_of_additional_black_balls (x : ℕ)
  (h : (additional_black_balls x : ℚ) / new_total_balls x = 3 / 4) : x = 18 := by
  sorry

end drawing_red_ball_random_drawing_yellow_ball_impossible_probability_black_ball_number_of_additional_black_balls_l166_166963


namespace expression_value_l166_166247

noncomputable def expression (x b : ℝ) : ℝ :=
  (x / (x + b) + b / (x - b)) / (b / (x + b) - x / (x - b))

theorem expression_value (b x : ℝ) (hb : b ≠ 0) (hx : x ≠ b ∧ x ≠ -b) :
  expression x b = -1 := 
by
  sorry

end expression_value_l166_166247


namespace maximum_books_l166_166851

theorem maximum_books (dollars : ℝ) (price_per_book : ℝ) (n : ℕ) 
    (h1 : dollars = 12) (h2 : price_per_book = 1.25) : n ≤ 9 :=
    sorry

end maximum_books_l166_166851


namespace common_ratio_is_two_l166_166506

-- Given a geometric sequence with specific terms
variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions: all terms are positive, a_2 = 3, a_6 = 48
axiom pos_terms : ∀ n, a n > 0
axiom a2_eq : a 2 = 3
axiom a6_eq : a 6 = 48

-- Question: Prove the common ratio q is 2
theorem common_ratio_is_two :
  (∀ n, a n = a 1 * q ^ (n - 1)) → q = 2 :=
by
  sorry

end common_ratio_is_two_l166_166506


namespace sin_600_eq_neg_sqrt_3_over_2_l166_166415

theorem sin_600_eq_neg_sqrt_3_over_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_eq_neg_sqrt_3_over_2_l166_166415


namespace intersection_lines_l166_166554

theorem intersection_lines (c d : ℝ) (h1 : 6 = 2 * 4 + c) (h2 : 6 = 5 * 4 + d) : c + d = -16 := 
by
  sorry

end intersection_lines_l166_166554


namespace distance_between_joe_and_gracie_l166_166231

open Complex

noncomputable def joe_point : ℂ := 2 + 3 * I
noncomputable def gracie_point : ℂ := -2 + 2 * I
noncomputable def distance := abs (joe_point - gracie_point)

theorem distance_between_joe_and_gracie :
  distance = Real.sqrt 17 := by
  sorry

end distance_between_joe_and_gracie_l166_166231


namespace gcd_max_1001_l166_166717

theorem gcd_max_1001 (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1001) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 143 := 
sorry

end gcd_max_1001_l166_166717


namespace last_digit_of_2_pow_2018_l166_166387

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

end last_digit_of_2_pow_2018_l166_166387


namespace books_taken_off_l166_166879

def books_initially : ℝ := 38.0
def books_remaining : ℝ := 28.0

theorem books_taken_off : books_initially - books_remaining = 10 := by
  sorry

end books_taken_off_l166_166879


namespace surface_area_of_resulting_solid_l166_166775

-- Define the original cube dimensions
def original_cube_surface_area (s : ℕ) := 6 * s * s

-- Define the smaller cube dimensions to be cut
def small_cube_surface_area (s : ℕ) := 3 * s * s

-- Define the proof problem
theorem surface_area_of_resulting_solid :
  original_cube_surface_area 3 - small_cube_surface_area 1 - small_cube_surface_area 2 + (3 * 1 + 3 * 4) = 54 :=
by
  -- The actual proof is to be filled in here
  sorry

end surface_area_of_resulting_solid_l166_166775


namespace smallest_perfect_square_gt_100_has_odd_number_of_factors_l166_166634

theorem smallest_perfect_square_gt_100_has_odd_number_of_factors : 
  ∃ n : ℕ, (n > 100) ∧ (∃ k : ℕ, n = k * k) ∧ (∀ m > 100, ∃ t : ℕ, m = t * t → n ≤ m) := 
sorry

end smallest_perfect_square_gt_100_has_odd_number_of_factors_l166_166634


namespace intersection_of_A_and_B_l166_166669

def A : Set ℝ := { x | 0 < x ∧ x < 2 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l166_166669


namespace path_inequality_l166_166812

theorem path_inequality
  (f : ℕ → ℕ → ℝ) :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
sorry

end path_inequality_l166_166812


namespace bob_sheep_and_ratio_l166_166844

-- Define the initial conditions
def mary_initial_sheep : ℕ := 300
def additional_sheep_bob_has : ℕ := 35
def sheep_mary_buys : ℕ := 266
def fewer_sheep_than_bob : ℕ := 69

-- Define the number of sheep Bob has
def bob_sheep (mary_initial_sheep : ℕ) (additional_sheep_bob_has : ℕ) : ℕ := 
  mary_initial_sheep + additional_sheep_bob_has

-- Define the number of sheep Mary has after buying more sheep
def mary_new_sheep (mary_initial_sheep : ℕ) (sheep_mary_buys : ℕ) : ℕ := 
  mary_initial_sheep + sheep_mary_buys

-- Define the relation between Mary's and Bob's sheep (after Mary buys sheep)
def mary_bob_relation (mary_new_sheep : ℕ) (fewer_sheep_than_bob : ℕ) : Prop :=
  mary_new_sheep + fewer_sheep_than_bob = bob_sheep mary_initial_sheep additional_sheep_bob_has

-- Define the proof problem
theorem bob_sheep_and_ratio : 
  bob_sheep mary_initial_sheep additional_sheep_bob_has = 635 ∧ 
  (bob_sheep mary_initial_sheep additional_sheep_bob_has) * 300 = 635 * mary_initial_sheep := 
by 
  sorry

end bob_sheep_and_ratio_l166_166844


namespace delta_max_success_ratio_l166_166510

/-- In a two-day math challenge, Gamma and Delta both attempted questions totalling 600 points. 
    Gamma scored 180 points out of 300 points attempted each day.
    Delta attempted a different number of points each day and their daily success ratios were less by both days than Gamma's, 
    whose overall success ratio was 3/5. Prove that the maximum possible two-day success ratio that Delta could have achieved was 359/600. -/
theorem delta_max_success_ratio :
  ∀ (x y z w : ℕ), (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < w) ∧ (x ≤ (3 * y) / 5) ∧ (z ≤ (3 * w) / 5) ∧ (y + w = 600) ∧ (x + z < 360)
  → (x + z) / 600 ≤ 359 / 600 :=
by
  sorry

end delta_max_success_ratio_l166_166510


namespace uneven_sons_and_daughters_probability_l166_166385

open Finset

theorem uneven_sons_and_daughters_probability :
  (∀ (s : Set (Fin 8)) (h : s.card = 4), true) →
  probability (λ (s : Set (Fin 8)), s.card = 4) (total_outcomes 256) = 70 / 256 →
  probability (λ (s : Set (Fin 8)), s.card ≠ 4) (total_outcomes 256) = 93 / 128 := 
sorry

end uneven_sons_and_daughters_probability_l166_166385


namespace average_of_last_four_numbers_l166_166693

theorem average_of_last_four_numbers
  (avg_seven : ℝ) (avg_first_three : ℝ) (avg_last_four : ℝ)
  (h1 : avg_seven = 62) (h2 : avg_first_three = 55) :
  avg_last_four = 67.25 := 
by
  sorry

end average_of_last_four_numbers_l166_166693


namespace total_points_scored_l166_166857

-- Define the points scored by Sam and his friend
def points_scored_by_sam : ℕ := 75
def points_scored_by_friend : ℕ := 12

-- The main theorem stating the total points
theorem total_points_scored : points_scored_by_sam + points_scored_by_friend = 87 := by
  -- Proof goes here
  sorry

end total_points_scored_l166_166857


namespace repeating_decimal_sum_l166_166093

theorem repeating_decimal_sum (c d : ℕ) (h : 7 / 19 = (c * 10 + d) / 99) : c + d = 9 :=
sorry

end repeating_decimal_sum_l166_166093


namespace area_of_circle_r_is_16_percent_of_circle_s_l166_166741

open Real

variables (Ds Dr Rs Rr As Ar : ℝ)

def circle_r_is_40_percent_of_circle_s (Ds Dr : ℝ) := Dr = 0.40 * Ds
def radius_of_circle (D : ℝ) (R : ℝ) := R = D / 2
def area_of_circle (R : ℝ) (A : ℝ) := A = π * R^2
def percentage_area (As Ar : ℝ) (P : ℝ) := P = (Ar / As) * 100

theorem area_of_circle_r_is_16_percent_of_circle_s :
  ∀ (Ds Dr Rs Rr As Ar : ℝ),
    circle_r_is_40_percent_of_circle_s Ds Dr →
    radius_of_circle Ds Rs →
    radius_of_circle Dr Rr →
    area_of_circle Rs As →
    area_of_circle Rr Ar →
    percentage_area As Ar 16 := by
  intros Ds Dr Rs Rr As Ar H1 H2 H3 H4 H5
  sorry

end area_of_circle_r_is_16_percent_of_circle_s_l166_166741


namespace intercept_sum_l166_166907

theorem intercept_sum (x y : ℝ) (h : y - 3 = -3 * (x + 2)) :
  (∃ (x_int : ℝ), y = 0 ∧ x_int = -1) ∧ (∃ (y_int : ℝ), x = 0 ∧ y_int = -3) →
  (-1 + (-3) = -4) := by
  sorry

end intercept_sum_l166_166907


namespace no_boys_love_cards_l166_166353

def boys_love_marbles := 13
def total_marbles := 26
def marbles_per_boy := 2

theorem no_boys_love_cards (boys_love_marbles total_marbles marbles_per_boy : ℕ)
  (h1 : boys_love_marbles * marbles_per_boy = total_marbles) : 
  ∃ no_boys_love_cards : ℕ, no_boys_love_cards = 0 :=
by
  sorry

end no_boys_love_cards_l166_166353


namespace length_of_base_of_vessel_l166_166579

noncomputable def volume_of_cube (edge : ℝ) := edge ^ 3

theorem length_of_base_of_vessel 
  (cube_edge : ℝ)
  (vessel_width : ℝ)
  (rise_in_water_level : ℝ)
  (volume_cube : ℝ)
  (h1 : cube_edge = 15)
  (h2 : vessel_width = 15)
  (h3 : rise_in_water_level = 11.25)
  (h4 : volume_cube = volume_of_cube cube_edge)
  : ∃ L : ℝ, L = volume_cube / (vessel_width * rise_in_water_level) ∧ L = 20 :=
by
  sorry

end length_of_base_of_vessel_l166_166579


namespace cups_of_flour_already_put_in_correct_l166_166235

-- Let F be the number of cups of flour Mary has already put in
def cups_of_flour_already_put_in (F : ℕ) : Prop :=
  let total_flour_needed := 12
  let cups_of_salt := 7
  let additional_flour_needed := cups_of_salt + 3
  F = total_flour_needed - additional_flour_needed

-- Theorem stating that F = 2
theorem cups_of_flour_already_put_in_correct (F : ℕ) : cups_of_flour_already_put_in F → F = 2 :=
by
  intro h
  sorry

end cups_of_flour_already_put_in_correct_l166_166235


namespace train_speed_l166_166590

theorem train_speed (distance time : ℤ) (h_distance : distance = 500)
    (h_time : time = 3) :
    distance / time = 166 :=
by
  -- Proof steps will be filled in here
  sorry

end train_speed_l166_166590


namespace jon_total_cost_l166_166601
-- Import the complete Mathlib library

-- Define the conditions
def MSRP : ℝ := 30
def insurance_rate : ℝ := 0.20
def tax_rate : ℝ := 0.50

-- Calculate intermediate values based on conditions
noncomputable def insurance_cost : ℝ := insurance_rate * MSRP
noncomputable def subtotal_before_tax : ℝ := MSRP + insurance_cost
noncomputable def state_tax : ℝ := tax_rate * subtotal_before_tax
noncomputable def total_cost : ℝ := subtotal_before_tax + state_tax

-- The theorem we need to prove
theorem jon_total_cost : total_cost = 54 := by
  -- Proof is omitted
  sorry

end jon_total_cost_l166_166601


namespace smallest_nat_div3_and_5_rem1_l166_166558

theorem smallest_nat_div3_and_5_rem1 : ∃ N : ℕ, N > 1 ∧ (N % 3 = 1) ∧ (N % 5 = 1) ∧ ∀ M : ℕ, M > 1 ∧ (M % 3 = 1) ∧ (M % 5 = 1) → N ≤ M := 
by
  sorry

end smallest_nat_div3_and_5_rem1_l166_166558


namespace number_of_zeros_l166_166635

-- Definitions based on the conditions
def five_thousand := 5 * 10 ^ 3
def one_hundred := 10 ^ 2

-- The main theorem that we want to prove
theorem number_of_zeros : (five_thousand ^ 50) * (one_hundred ^ 2) = 10 ^ 154 * 5 ^ 50 := 
by sorry

end number_of_zeros_l166_166635


namespace alex_hours_per_week_l166_166765

theorem alex_hours_per_week
  (summer_earnings : ℕ)
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (academic_year_weeks : ℕ)
  (academic_year_earnings : ℕ)
  (same_hourly_rate : Prop) :
  summer_earnings = 4000 →
  summer_weeks = 8 →
  summer_hours_per_week = 40 →
  academic_year_weeks = 32 →
  academic_year_earnings = 8000 →
  same_hourly_rate →
  (academic_year_earnings / ((summer_earnings : ℚ) / (summer_weeks * summer_hours_per_week)) / academic_year_weeks) = 20 :=
by
  sorry

end alex_hours_per_week_l166_166765


namespace third_grade_contribution_fourth_grade_contribution_l166_166369

def first_grade := 20
def second_grade := 45
def third_grade := first_grade + second_grade - 17
def fourth_grade := 2 * third_grade - 36

theorem third_grade_contribution : third_grade = 48 := by
  sorry

theorem fourth_grade_contribution : fourth_grade = 60 := by
  sorry

end third_grade_contribution_fourth_grade_contribution_l166_166369


namespace concurrent_circumcircles_l166_166283

variables {A B C A' B' C' : Type} [PlaneReal A B C A' B' C']

theorem concurrent_circumcircles 
  (hA_prime : A' ∈ line B C) 
  (hB_prime : B' ∈ line C A) 
  (hC_prime : C' ∈ line A B) : 
  ∃ I, I ∈ circumcircle A B' C' ∧ I ∈ circumcircle A' B C' ∧ I ∈ circumcircle A' B' C := 
sorry

end concurrent_circumcircles_l166_166283


namespace arccos_neg_one_l166_166157

theorem arccos_neg_one : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_l166_166157


namespace gerald_paid_l166_166808

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 :=
by
  sorry

end gerald_paid_l166_166808


namespace find_first_blend_price_l166_166526

-- Define the conditions
def first_blend_price (x : ℝ) := x
def second_blend_price : ℝ := 8.00
def total_blend_weight : ℝ := 20
def total_blend_price_per_pound : ℝ := 8.40
def first_blend_weight : ℝ := 8
def second_blend_weight : ℝ := total_blend_weight - first_blend_weight

-- Define the cost calculations
def first_blend_total_cost (x : ℝ) := first_blend_weight * x
def second_blend_total_cost := second_blend_weight * second_blend_price
def total_blend_total_cost (x : ℝ) := first_blend_total_cost x + second_blend_total_cost

-- Prove that the price per pound of the first blend is $9.00
theorem find_first_blend_price : ∃ x : ℝ, total_blend_total_cost x = total_blend_weight * total_blend_price_per_pound ∧ x = 9 :=
by
  sorry

end find_first_blend_price_l166_166526


namespace cube_surface_area_increase_l166_166734

theorem cube_surface_area_increase (s : ℝ) :
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  (A_new - A_original) / A_original * 100 = 224 :=
by
  -- Definitions from the conditions
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  -- Rest of the proof; replace sorry with the actual proof
  sorry

end cube_surface_area_increase_l166_166734


namespace cos_4theta_l166_166342

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (4 * θ) = 17/81 :=
  sorry

end cos_4theta_l166_166342


namespace ratio_of_a_b_l166_166637

-- Define the problem
theorem ratio_of_a_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it.
  sorry

end ratio_of_a_b_l166_166637


namespace sheena_weeks_to_complete_l166_166862

/- Definitions -/
def time_per_dress : ℕ := 12
def number_of_dresses : ℕ := 5
def weekly_sewing_time : ℕ := 4

/- Theorem -/
theorem sheena_weeks_to_complete : (number_of_dresses * time_per_dress) / weekly_sewing_time = 15 := 
by 
  /- Proof is omitted -/
  sorry

end sheena_weeks_to_complete_l166_166862


namespace arithmetic_sequence_sum_l166_166414

theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℕ) (h1 : S n = 54) (h2 : S (2 * n) = 72) :
  S (3 * n) = 78 :=
sorry

end arithmetic_sequence_sum_l166_166414


namespace joe_time_to_friends_house_l166_166892

theorem joe_time_to_friends_house
  (feet_moved : ℕ) (time_taken : ℕ) (remaining_distance : ℕ) (feet_in_yard : ℕ)
  (rate_of_movement : ℕ) (remaining_distance_feet : ℕ) (time_to_cover_remaining_distance : ℕ) :
  feet_moved = 80 →
  time_taken = 40 →
  remaining_distance = 90 →
  feet_in_yard = 3 →
  rate_of_movement = feet_moved / time_taken →
  remaining_distance_feet = remaining_distance * feet_in_yard →
  time_to_cover_remaining_distance = remaining_distance_feet / rate_of_movement →
  time_to_cover_remaining_distance = 135 :=
by
  sorry

end joe_time_to_friends_house_l166_166892


namespace arithmetic_geometric_progressions_l166_166332

theorem arithmetic_geometric_progressions (a b : ℕ → ℕ) (d r : ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = r * b n)
  (h_comm_ratio : r = 2)
  (h_eq1 : a 1 + d - 2 * (b 1) = a 1 + 2 * d - 4 * (b 1))
  (h_eq2 : a 1 + d - 2 * (b 1) = 8 * (b 1) - (a 1 + 3 * d)) :
  (a 1 = b 1) ∧ (∃ n, ∀ k, 1 ≤ k ∧ k ≤ 10 → (b (k + 1) = a (1 + n * d) + a 1)) := by
  sorry

end arithmetic_geometric_progressions_l166_166332


namespace quadratic_function_symmetry_l166_166917

theorem quadratic_function_symmetry
  (p : ℝ → ℝ)
  (h_sym : ∀ x, p (5.5 - x) = p (5.5 + x))
  (h_0 : p 0 = -4) :
  p 11 = -4 :=
by sorry

end quadratic_function_symmetry_l166_166917


namespace sufficient_but_not_necessary_condition_circle_l166_166173

theorem sufficient_but_not_necessary_condition_circle {a : ℝ} (h : a = 1) :
  ∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y + a = 0 → (∀ a, a < 2 → (x - 1)^2 + (y + 1)^2 = 2 - a) :=
by
  sorry

end sufficient_but_not_necessary_condition_circle_l166_166173


namespace frictional_force_is_12N_l166_166137

-- Given conditions
variables (m1 m2 a μ : ℝ)
-- Constants
def g : ℝ := 9.8

-- Frictional force on the tank
def F_friction : ℝ := μ * m1 * g

-- Proof statement
theorem frictional_force_is_12N (m1_value : m1 = 3) (m2_value : m2 = 15) (a_value : a = 4) (μ_value : μ = 0.6) :
  m1 * a = 12 :=
by
  sorry

end frictional_force_is_12N_l166_166137


namespace hyperbola_equation_l166_166223

noncomputable def h : ℝ := -4
noncomputable def k : ℝ := 2
noncomputable def a : ℝ := 1
noncomputable def c : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 1

theorem hyperbola_equation :
  (h + k + a + b) = 0 := by
  have h := -4
  have k := 2
  have a := 1
  have b := 1
  show (-4 + 2 + 1 + 1) = 0
  sorry

end hyperbola_equation_l166_166223


namespace distance_around_track_l166_166374

-- Define the conditions
def total_mileage : ℝ := 10
def distance_to_high_school : ℝ := 3
def round_trip_distance : ℝ := 2 * distance_to_high_school

-- State the question and the desired proof problem
theorem distance_around_track : 
  total_mileage - round_trip_distance = 4 := 
by
  sorry

end distance_around_track_l166_166374


namespace betty_age_l166_166783

variable (C A B : ℝ)

-- conditions
def Carol_five_times_Alice := C = 5 * A
def Alice_twelve_years_younger_than_Carol := A = C - 12
def Carol_twice_as_old_as_Betty := C = 2 * B

-- goal
theorem betty_age (hc1 : Carol_five_times_Alice C A)
                  (hc2 : Alice_twelve_years_younger_than_Carol C A)
                  (hc3 : Carol_twice_as_old_as_Betty C B) : B = 7.5 := 
  by
  sorry

end betty_age_l166_166783


namespace new_oranges_added_l166_166760

def initial_oranges : Nat := 31
def thrown_away_oranges : Nat := 9
def final_oranges : Nat := 60
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges (initial_oranges thrown_away_oranges final_oranges : Nat) : Nat := 
  final_oranges - (initial_oranges - thrown_away_oranges)

theorem new_oranges_added :
  new_oranges initial_oranges thrown_away_oranges final_oranges = 38 := by
  sorry

end new_oranges_added_l166_166760


namespace find_n_from_degree_l166_166958

theorem find_n_from_degree (n : ℕ) (h : 2 + n = 5) : n = 3 :=
by {
  sorry
}

end find_n_from_degree_l166_166958


namespace five_colored_flags_l166_166338

def num_different_flags (colors total_stripes : ℕ) : ℕ :=
  Nat.choose colors total_stripes * Nat.factorial total_stripes

theorem five_colored_flags : num_different_flags 11 5 = 55440 := by
  sorry

end five_colored_flags_l166_166338


namespace red_candies_l166_166738

theorem red_candies (R Y B : ℕ) 
  (h1 : Y = 3 * R - 20)
  (h2 : B = Y / 2)
  (h3 : R + B = 90) :
  R = 40 :=
by
  sorry

end red_candies_l166_166738


namespace fraction_power_four_l166_166431

theorem fraction_power_four :
  (5 / 6) ^ 4 = 625 / 1296 :=
by sorry

end fraction_power_four_l166_166431


namespace Janice_age_l166_166058

theorem Janice_age (x : ℝ) (h : x + 12 = 8 * (x - 2)) : x = 4 := by
  sorry

end Janice_age_l166_166058


namespace ahmed_goats_is_13_l166_166593

def adam_goats : ℕ := 7

def andrew_goats : ℕ := 2 * adam_goats + 5

def ahmed_goats : ℕ := andrew_goats - 6

theorem ahmed_goats_is_13 : ahmed_goats = 13 :=
by
  sorry

end ahmed_goats_is_13_l166_166593


namespace range_of_m_l166_166067

noncomputable def G (x m : ℝ) : ℝ := (8 * x^2 + 24 * x + 5 * m) / 8

theorem range_of_m (G_is_square : ∃ c d, ∀ x, G x m = (c * x + d) ^ 2) : 3 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l166_166067


namespace gcd_102_238_l166_166726

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l166_166726


namespace remainder_zero_l166_166321

theorem remainder_zero (x : ℂ) 
  (h : x^5 + x^4 + x^3 + x^2 + x + 1 = 0) : 
  x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0 := 
by 
  sorry

end remainder_zero_l166_166321


namespace positive_root_of_equation_l166_166184

theorem positive_root_of_equation :
  ∃ a b : ℤ, (a + b * Real.sqrt 3)^3 - 5 * (a + b * Real.sqrt 3)^2 + 2 * (a + b * Real.sqrt 3) - Real.sqrt 3 = 0 ∧
    a + b * Real.sqrt 3 > 0 ∧
    (a + b * Real.sqrt 3) = 3 + Real.sqrt 3 := 
by
  sorry

end positive_root_of_equation_l166_166184


namespace negation_of_universal_is_existential_l166_166695

theorem negation_of_universal_is_existential :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by
  sorry

end negation_of_universal_is_existential_l166_166695


namespace expand_expression_l166_166612

theorem expand_expression (x y : ℤ) : (x + 7) * (3 * y + 8) = 3 * x * y + 8 * x + 21 * y + 56 :=
by
  sorry

end expand_expression_l166_166612


namespace max_value_of_reciprocal_sums_of_zeros_l166_166037

noncomputable def quadratic_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + 2 * x - 1

noncomputable def linear_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x + 1

theorem max_value_of_reciprocal_sums_of_zeros (k : ℝ) (x1 x2 : ℝ)
  (h0 : -1 < k ∧ k < 0)
  (hx1 : x1 ∈ Set.Ioc 0 1 → quadratic_part k x1 = 0)
  (hx2 : x2 ∈ Set.Ioi 1 → linear_part k x2 = 0)
  (hx_distinct : x1 ≠ x2) :
  (1 / x1) + (1 / x2) = 9 / 4 :=
sorry

end max_value_of_reciprocal_sums_of_zeros_l166_166037


namespace ellipse_focal_distance_l166_166913

theorem ellipse_focal_distance (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 16 + y^2 / m = 1) ∧ (2 * Real.sqrt (16 - m) = 2 * Real.sqrt 7)) → m = 9 :=
by
  intro h
  sorry

end ellipse_focal_distance_l166_166913


namespace solve_a_minus_b_l166_166954

theorem solve_a_minus_b (a b : ℝ) (h1 : 2010 * a + 2014 * b = 2018) (h2 : 2012 * a + 2016 * b = 2020) : a - b = -3 :=
sorry

end solve_a_minus_b_l166_166954


namespace four_circles_max_parts_l166_166824

theorem four_circles_max_parts (n : ℕ) (h1 : ∀ n, n = 1 ∨ n = 2 ∨ n = 3 → ∃ k, k = 2^n) :
    n = 4 → ∃ k, k = 14 :=
by
  sorry

end four_circles_max_parts_l166_166824


namespace necessary_but_not_sufficient_l166_166839

-- Define the sets A and B
def A (x : ℝ) : Prop := x > 2
def B (x : ℝ) : Prop := x > 1

-- Prove that B (necessary condition x > 1) does not suffice for A (x > 2)
theorem necessary_but_not_sufficient (x : ℝ) (h : B x) : A x ∨ ¬A x :=
by
  -- B x is a necessary condition for A x
  have h1 : x > 1 := h
  -- A x is not necessarily implied by B x
  sorry

end necessary_but_not_sufficient_l166_166839


namespace smallest_pos_int_ends_in_6_divisible_by_11_l166_166103

theorem smallest_pos_int_ends_in_6_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 6 ∧ 11 ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m % 10 = 6 ∧ 11 ∣ m → n ≤ m := by
  sorry

end smallest_pos_int_ends_in_6_divisible_by_11_l166_166103


namespace arithmetic_expression_eval_l166_166602

theorem arithmetic_expression_eval : 
  5 * 7.5 + 2 * 12 + 8.5 * 4 + 7 * 6 = 137.5 :=
by
  sorry

end arithmetic_expression_eval_l166_166602


namespace root_in_interval_iff_a_outside_range_l166_166038

theorem root_in_interval_iff_a_outside_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 1 ∧ a * x + 1 = 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end root_in_interval_iff_a_outside_range_l166_166038


namespace sheepdog_catches_sheep_in_20_seconds_l166_166073

theorem sheepdog_catches_sheep_in_20_seconds :
  ∀ (sheep_speed dog_speed : ℝ) (initial_distance : ℝ),
    sheep_speed = 12 →
    dog_speed = 20 →
    initial_distance = 160 →
    (initial_distance / (dog_speed - sheep_speed)) = 20 := by
  intros sheep_speed dog_speed initial_distance h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end sheepdog_catches_sheep_in_20_seconds_l166_166073


namespace condition_relationship_l166_166331

noncomputable def M : Set ℝ := {x | x > 2}
noncomputable def P : Set ℝ := {x | x < 3}

theorem condition_relationship :
  ∀ x, (x ∈ (M ∩ P) → x ∈ (M ∪ P)) ∧ ¬ (x ∈ (M ∪ P) → x ∈ (M ∩ P)) :=
by
  sorry

end condition_relationship_l166_166331


namespace car_A_speed_l166_166154

theorem car_A_speed (s_A s_B : ℝ) (d_AB d_extra t : ℝ) (h_s_B : s_B = 50) (h_d_AB : d_AB = 40) (h_d_extra : d_extra = 8) (h_time : t = 6) 
(h_distance_traveled_by_car_B : s_B * t = 300) 
(h_distance_difference : d_AB + d_extra = 48) :
  s_A = 58 :=
by
  sorry

end car_A_speed_l166_166154


namespace fraction_power_rule_l166_166429

theorem fraction_power_rule :
  (5 / 6) ^ 4 = (625 : ℚ) / 1296 := 
by sorry

end fraction_power_rule_l166_166429


namespace forester_trees_planted_l166_166583

theorem forester_trees_planted (initial_trees : ℕ) (tripled_trees : ℕ) (trees_planted_monday : ℕ) (trees_planted_tuesday : ℕ) :
  initial_trees = 30 ∧ tripled_trees = 3 * initial_trees ∧ trees_planted_monday = tripled_trees - initial_trees ∧ trees_planted_tuesday = trees_planted_monday / 3 →
  trees_planted_monday + trees_planted_tuesday = 80 :=
by
  sorry

end forester_trees_planted_l166_166583


namespace election_vote_percentage_l166_166218

theorem election_vote_percentage 
  (total_students : ℕ)
  (winner_percentage : ℝ)
  (loser_percentage : ℝ)
  (vote_difference : ℝ)
  (P : ℝ)
  (H1 : total_students = 2000)
  (H2 : winner_percentage = 0.55)
  (H3 : loser_percentage = 0.45)
  (H4 : vote_difference = 50)
  (H5 : 0.1 * P * (total_students / 100) = vote_difference) :
  P = 25 := 
sorry

end election_vote_percentage_l166_166218


namespace scientific_notation_l166_166900

theorem scientific_notation (n : ℝ) (h : n = 1300000) : n = 1.3 * 10^6 :=
by {
  sorry
}

end scientific_notation_l166_166900


namespace inequality_proof_l166_166051

theorem inequality_proof (a b : ℝ) (h : a < b) : -a - 1 > -b - 1 :=
sorry

end inequality_proof_l166_166051


namespace arccos_neg_one_eq_pi_l166_166162

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
  sorry

end arccos_neg_one_eq_pi_l166_166162


namespace smallest_integer_to_make_perfect_square_l166_166329

-- Define the number y as specified
def y : ℕ := 2^5 * 3^6 * (2^2)^7 * 5^8 * (2 * 3)^9 * 7^10 * (2^3)^11 * (3^2)^12

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The goal statement
theorem smallest_integer_to_make_perfect_square : 
  ∃ z : ℕ, z > 0 ∧ is_perfect_square (y * z) ∧ ∀ w : ℕ, w > 0 → is_perfect_square (y * w) → z ≤ w := by
  sorry

end smallest_integer_to_make_perfect_square_l166_166329


namespace zeros_of_g_l166_166961

theorem zeros_of_g (a b : ℝ) (h : 2 * a + b = 0) :
  (∃ x : ℝ, (b * x^2 - a * x = 0) ∧ (x = 0 ∨ x = -1 / 2)) :=
by
  sorry

end zeros_of_g_l166_166961


namespace eq_or_sum_zero_l166_166629

variables (a b c d : ℝ)

theorem eq_or_sum_zero (h : (3 * a + 2 * b) / (2 * b + 4 * c) = (4 * c + 3 * d) / (3 * d + 3 * a)) :
  3 * a = 4 * c ∨ 3 * a + 3 * d + 2 * b + 4 * c = 0 :=
by sorry

end eq_or_sum_zero_l166_166629


namespace cube_div_identity_l166_166730

theorem cube_div_identity (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) :
  (a^3 + b^3) / (a^2 - a * b + b^2) = 9 := by
  sorry

end cube_div_identity_l166_166730


namespace find_train_length_l166_166138

noncomputable def speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 245.03
noncomputable def time_seconds : ℝ := 30
noncomputable def speed_ms : ℝ := (speed_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := speed_ms * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length

theorem find_train_length : train_length = 129.97 := 
by
  sorry

end find_train_length_l166_166138


namespace range_of_a_l166_166434

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ 9 ≤ a := 
by sorry

end range_of_a_l166_166434


namespace function_expression_and_min_value_l166_166524

def f (x b : ℝ) := x^2 - 2*x + b

theorem function_expression_and_min_value 
    (a b : ℝ)
    (condition1 : f (2 ^ a) b = b)
    (condition2 : f a b = 4) :
    f a b = 5 
    ∧ 
    ∃ c : ℝ, f (2^c) 5 = 4 ∧ c = 0 :=
by
  sorry

end function_expression_and_min_value_l166_166524


namespace smallest_n_for_symmetry_property_l166_166322

-- Define the setup for the problem
def has_required_symmetry (n : ℕ) : Prop :=
∀ (S : Finset (Fin n)), S.card = 5 →
∃ (l : Fin n → Fin n), (∀ v ∈ S, l v ≠ v) ∧ (∀ v ∈ S, l v ∉ S)

-- The main lemma we are proving
theorem smallest_n_for_symmetry_property : ∃ n : ℕ, (∀ m < n, ¬ has_required_symmetry m) ∧ has_required_symmetry 14 :=
by
  sorry

end smallest_n_for_symmetry_property_l166_166322


namespace functional_relationship_optimizing_profit_l166_166761

-- Define the scope of the problem with conditions and proof statements

variables (x : ℝ) (y : ℝ)

-- Conditions
def price_condition := 44 ≤ x ∧ x ≤ 52
def sales_function := y = -10 * x + 740
def profit_function (x : ℝ) := -10 * x^2 + 1140 * x - 29600

-- Lean statement to prove the first part
theorem functional_relationship (h₁ : 44 ≤ x) (h₂ : x ≤ 52) : y = -10 * x + 740 := by
  sorry

-- Lean statement to prove the second part
theorem optimizing_profit (h₃ : 44 ≤ x) (h₄ : x ≤ 52) : (profit_function 52 = 2640 ∧ (∀ x, (44 ≤ x ∧ x ≤ 52) → profit_function x ≤ 2640)) := by
  sorry

end functional_relationship_optimizing_profit_l166_166761


namespace check_correct_digit_increase_l166_166990

-- Definition of the numbers involved
def number1 : ℕ := 732
def number2 : ℕ := 648
def number3 : ℕ := 985
def given_sum : ℕ := 2455
def calc_sum : ℕ := number1 + number2 + number3
def difference : ℕ := given_sum - calc_sum

-- Specify the smallest digit that needs to be increased by 1
def smallest_digit_to_increase : ℕ := 8

-- Theorem to check the validity of the problem's claim
theorem check_correct_digit_increase :
  (smallest_digit_to_increase = 8) →
  (calc_sum + 10 = given_sum - 80) :=
by
  intro h
  sorry

end check_correct_digit_increase_l166_166990


namespace obtuse_angle_condition_l166_166950

def dot_product (a b : (ℝ × ℝ)) : ℝ := a.1 * b.1 + a.2 * b.2

def is_obtuse_angle (a b : (ℝ × ℝ)) : Prop := dot_product a b < 0

def is_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem obtuse_angle_condition :
  (∀ (x : ℝ), x > 0 → is_obtuse_angle (-1, 0) (x, 1 - x) ∧ ¬is_parallel (-1, 0) (x, 1 - x)) ∧ 
  (∀ (x : ℝ), is_obtuse_angle (-1, 0) (x, 1 - x) → x > 0) :=
sorry

end obtuse_angle_condition_l166_166950


namespace eval_diamond_expr_l166_166168

def diamond (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1, 1) => 4
  | (1, 2) => 3
  | (1, 3) => 2
  | (1, 4) => 1
  | (2, 1) => 1
  | (2, 2) => 4
  | (2, 3) => 3
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 1
  | (3, 3) => 4
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 1
  | (4, 4) => 4
  | (_, _) => 0  -- This handles any case outside of 1,2,3,4 which should ideally not happen

theorem eval_diamond_expr : diamond (diamond 3 4) (diamond 2 1) = 2 := by
  sorry

end eval_diamond_expr_l166_166168


namespace largest_A_proof_smallest_A_proof_l166_166772

def is_coprime_with_12 (n : ℕ) : Prop := Nat.gcd n 12 = 1

def obtain_A_from_B (B : ℕ) : ℕ :=
  let b := B % 10
  let k := B / 10
  b * 10^7 + k

constant B : ℕ → Prop
constant A : ℕ → ℕ → Prop

noncomputable def largest_A : ℕ :=
  99999998

noncomputable def smallest_A : ℕ :=
  14444446

theorem largest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = largest_A :=
sorry

theorem smallest_A_proof (B : ℕ) (h1 : B > 44444444) (h2 : is_coprime_with_12 B) :
  obtain_A_from_B B = smallest_A :=
sorry

end largest_A_proof_smallest_A_proof_l166_166772


namespace James_has_43_Oreos_l166_166825

variable (J : ℕ)
variable (James_Oreos : ℕ)

-- Conditions
def condition1 : Prop := James_Oreos = 4 * J + 7
def condition2 : Prop := J + James_Oreos = 52

-- The statement to prove: James has 43 Oreos given the conditions
theorem James_has_43_Oreos (h1 : condition1 J James_Oreos) (h2 : condition2 J James_Oreos) : James_Oreos = 43 :=
by
  sorry

end James_has_43_Oreos_l166_166825


namespace evaluate_exponentiation_l166_166924

theorem evaluate_exponentiation : (3 ^ 3) ^ 4 = 531441 := by
  sorry

end evaluate_exponentiation_l166_166924


namespace remaining_mushroom_pieces_l166_166619

theorem remaining_mushroom_pieces 
  (mushrooms : ℕ) 
  (pieces_per_mushroom : ℕ) 
  (pieces_used_by_kenny : ℕ) 
  (pieces_used_by_karla : ℕ) 
  (mushrooms_cut : mushrooms = 22) 
  (pieces_per_mushroom_def : pieces_per_mushroom = 4) 
  (kenny_pieces_def : pieces_used_by_kenny = 38) 
  (karla_pieces_def : pieces_used_by_karla = 42) : 
  (mushrooms * pieces_per_mushroom - (pieces_used_by_kenny + pieces_used_by_karla)) = 8 := 
by 
  sorry

end remaining_mushroom_pieces_l166_166619


namespace number_of_roots_of_unity_l166_166468

theorem number_of_roots_of_unity (n : ℕ) (z : ℂ) (c d : ℤ) (h1 : n ≥ 3) (h2 : z^n = 1) (h3 : z^3 + (c : ℂ) * z + (d : ℂ) = 0) : 
  ∃ k : ℕ, k = 4 :=
by sorry

end number_of_roots_of_unity_l166_166468


namespace best_ketchup_deal_l166_166777

/-- Given different options of ketchup bottles with their respective prices and volumes as below:
 - Bottle 1: 10 oz for $1
 - Bottle 2: 16 oz for $2
 - Bottle 3: 25 oz for $2.5
 - Bottle 4: 50 oz for $5
 - Bottle 5: 200 oz for $10
And knowing that Billy's mom gives him $10 to spend entirely on ketchup,
prove that the best deal for Billy is to buy one bottle of the $10 ketchup which contains 200 ounces. -/
theorem best_ketchup_deal :
  let price := [1, 2, 2.5, 5, 10]
  let volume := [10, 16, 25, 50, 200]
  let cost_per_ounce := [0.1, 0.125, 0.1, 0.1, 0.05]
  ∃ i, (volume[i] = 200) ∧ (price[i] = 10) ∧ (∀ j, cost_per_ounce[i] ≤ cost_per_ounce[j]) ∧ (price.sum = 10) :=
by
  sorry

end best_ketchup_deal_l166_166777


namespace equivalent_statements_l166_166891

variable (P Q : Prop)

theorem equivalent_statements (h : P → Q) :
  (¬Q → ¬P) ∧ (¬P ∨ Q) :=
by 
  sorry

end equivalent_statements_l166_166891


namespace min_price_floppy_cd_l166_166095

theorem min_price_floppy_cd (x y : ℝ) (h1 : 4 * x + 5 * y ≥ 20) (h2 : 6 * x + 3 * y ≤ 24) : 3 * x + 9 * y ≥ 22 :=
by
  -- The proof is not provided as per the instructions.
  sorry

end min_price_floppy_cd_l166_166095


namespace moores_law_2000_l166_166077

noncomputable def number_of_transistors (year : ℕ) : ℕ :=
  if year = 1990 then 1000000
  else 1000000 * 2 ^ ((year - 1990) / 2)

theorem moores_law_2000 :
  number_of_transistors 2000 = 32000000 :=
by
  unfold number_of_transistors
  rfl

end moores_law_2000_l166_166077


namespace sin_pow_cos_pow_eq_l166_166799

theorem sin_pow_cos_pow_eq (x : ℝ) (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
  Real.sin x ^ 14 + Real.cos x ^ 14 = 41 / 216 := by
  sorry

end sin_pow_cos_pow_eq_l166_166799


namespace find_solutions_l166_166926

-- Defining the system of equations as conditions
def cond1 (a b : ℕ) := a * b + 2 * a - b = 58
def cond2 (b c : ℕ) := b * c + 4 * b + 2 * c = 300
def cond3 (c d : ℕ) := c * d - 6 * c + 4 * d = 101

-- Theorem to prove the solutions satisfy the system of equations
theorem find_solutions (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0):
  cond1 a b ∧ cond2 b c ∧ cond3 c d ↔ (a, b, c, d) ∈ [(3, 26, 7, 13), (15, 2, 73, 7)] :=
by sorry

end find_solutions_l166_166926


namespace probability_between_21_and_30_l166_166076

/-- 
Given Melinda rolls two standard six-sided dice, 
forming a two-digit number with the two numbers rolled,
prove that the probability of forming a number between 21 and 30, inclusive, is 11/36.
-/
theorem probability_between_21_and_30 :
  let dice := set.range (λ n : ℕ, n + 1) ∩ {n | n ≤ 6},
      form_number (a b : ℕ) := 10 * a + b,
      valid_numbers := {n | 21 ≤ n ∧ n ≤ 30},
      probability (s : set ℕ) := (s.card : ℚ) / 36
  in probability {n | ∃ a b ∈ dice, form_number a b = n} = 11 / 36 :=
by
  sorry

end probability_between_21_and_30_l166_166076


namespace actual_time_before_storm_l166_166870

-- Define valid hour digit ranges before the storm
def valid_first_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3
def valid_second_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define valid minute digit ranges before the storm
def valid_third_digit (d : ℕ) : Prop := d = 4 ∨ d = 5 ∨ d = 6
def valid_fourth_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define a valid time in HH:MM format
def valid_time (hh mm : ℕ) : Prop :=
  hh < 24 ∧ mm < 60

-- The proof problem
theorem actual_time_before_storm (hh hh' mm mm' : ℕ) 
  (h1 : valid_first_digit hh) (h2 : valid_second_digit hh') 
  (h3 : valid_third_digit mm) (h4 : valid_fourth_digit mm') 
  (h_valid : valid_time (hh * 10 + hh') (mm * 10 + mm')) 
  (h_display : (hh + 1) * 10 + (hh' - 1) = 20 ∧ (mm + 1) * 10 + (mm' - 1) = 50) :
  hh * 10 + hh' = 19 ∧ mm * 10 + mm' = 49 :=
by
  sorry

end actual_time_before_storm_l166_166870


namespace smallest_number_to_end_in_four_zeros_l166_166615

theorem smallest_number_to_end_in_four_zeros (x : ℕ) :
  let n1 := 225
  let n2 := 525
  let factor_needed := 16
  (∃ y : ℕ, y = n1 * n2 * x) ∧ (10^4 ∣ n1 * n2 * x) ↔ x = factor_needed :=
by
  sorry

end smallest_number_to_end_in_four_zeros_l166_166615


namespace jade_handled_80_transactions_l166_166894

variable (mabel anthony cal jade : ℕ)

-- Conditions
def mabel_transactions : mabel = 90 :=
by sorry

def anthony_transactions : anthony = mabel + (10 * mabel / 100) :=
by sorry

def cal_transactions : cal = 2 * anthony / 3 :=
by sorry

def jade_transactions : jade = cal + 14 :=
by sorry

-- Proof problem
theorem jade_handled_80_transactions :
  mabel = 90 →
  anthony = mabel + (10 * mabel / 100) →
  cal = 2 * anthony / 3 →
  jade = cal + 14 →
  jade = 80 :=
by
  intros
  subst_vars
  -- The proof steps would normally go here, but we leave it with sorry.
  sorry

end jade_handled_80_transactions_l166_166894


namespace probability_product_divisible_by_4_l166_166188

open Probability

theorem probability_product_divisible_by_4 :
  (∑ x in finset.filter (λ (x : ℕ × ℕ), ((x.1 * x.2) % 4 = 0)) (finset.product (finset.range 8) (finset.range 8)), 1) / 64 = 25/64 :=
by sorry

end probability_product_divisible_by_4_l166_166188


namespace min_distance_convex_lens_l166_166438

theorem min_distance_convex_lens (t k f : ℝ) (hf : f > 0) (ht : t ≥ f)
    (h_lens: 1 / t + 1 / k = 1 / f) :
  t = 2 * f → t + k = 4 * f :=
by
  sorry

end min_distance_convex_lens_l166_166438


namespace gerald_paid_l166_166809

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 := by
  sorry

end gerald_paid_l166_166809


namespace Finn_initial_goldfish_l166_166185

variable (x : ℕ)

-- Defining the conditions
def number_of_goldfish_initial (x : ℕ) : Prop :=
  ∃ y z : ℕ, y = 32 ∧ z = 57 ∧ x = y + z 

-- Theorem statement to prove Finn's initial number of goldfish
theorem Finn_initial_goldfish (x : ℕ) (h : number_of_goldfish_initial x) : x = 89 := by
  sorry

end Finn_initial_goldfish_l166_166185


namespace soda_price_increase_l166_166764

theorem soda_price_increase (P : ℝ) (h1 : 1.5 * P = 6) : P = 4 :=
by
  -- Proof will be provided here
  sorry

end soda_price_increase_l166_166764


namespace triangle_area_right_angle_l166_166365

noncomputable def area_of_triangle (AB BC : ℝ) : ℝ :=
  1 / 2 * AB * BC

theorem triangle_area_right_angle (AB BC : ℝ) (hAB : AB = 12) (hBC : BC = 9) :
  area_of_triangle AB BC = 54 := by
  rw [hAB, hBC]
  norm_num
  sorry

end triangle_area_right_angle_l166_166365


namespace gem_stone_necklaces_sold_l166_166371

theorem gem_stone_necklaces_sold (total_earned total_cost number_bead number_gem total_necklaces : ℕ) 
    (h1 : total_earned = 36) 
    (h2 : total_cost = 6) 
    (h3 : number_bead = 3) 
    (h4 : total_necklaces = total_earned / total_cost) 
    (h5 : total_necklaces = number_bead + number_gem) : 
    number_gem = 3 := 
sorry

end gem_stone_necklaces_sold_l166_166371


namespace gcd_factorial_l166_166887

theorem gcd_factorial (n m l : ℕ) (h1 : n = 7) (h2 : m = 10) (h3 : l = 4): 
  Nat.gcd (Nat.factorial n) (Nat.factorial m / Nat.factorial l) = 2520 :=
by
  sorry

end gcd_factorial_l166_166887


namespace binomial_coefficient_divisible_by_p_l166_166240

theorem binomial_coefficient_divisible_by_p (p k : ℕ) (hp : Nat.Prime p) (hk1 : 0 < k) (hk2 : k < p) :
  p ∣ (Nat.factorial p / (Nat.factorial k * Nat.factorial (p - k))) :=
by
  sorry

end binomial_coefficient_divisible_by_p_l166_166240


namespace problem_BD_correct_l166_166026

noncomputable theory

open MeasureTheory
open ProbabilityTheory

variables (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)

variables (A B : Set Ω)
variables (P_A : P A = 0.5) (P_B : P B = 0.2)

theorem problem_BD_correct :
  (Disjoint A B → P (A ∪ B) = 0.7) ∧ 
  (condProb P B A = 0.2 → IndepEvents A B P) :=
by 
  intro HAB;
  apply And.intro;
  {
    -- Prove B
    intro h_disjoint;
    have h_Union : P (A ∪ B) = P A + P B := P.add_disjoint h_disjoint;
    rw [P_A, P_B] at h_Union;
    norm_num at h_Union;
    exact h_Union;
  };
  {
    -- Prove D
    intro h_condProb;
    refine ⟨?_⟩;
    {
      intro _ _;
      simp only [IndependentSets, MeasureTheoreticProbability, condProb, tprod, MeasureTheory.cond, habspace];
      simp at h_condProb
      assumption
    }
  };
  sorry -- further missing justification required

end problem_BD_correct_l166_166026


namespace problem_1_problem_2_l166_166621

theorem problem_1 (h : Real.tan (α / 2) = 2) : Real.tan (α + Real.arctan 1) = -1/7 :=
by
  sorry

theorem problem_2 (h : Real.tan (α / 2) = 2) : (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 :=
by
  sorry

end problem_1_problem_2_l166_166621


namespace distinct_four_digit_numbers_with_repeated_digit_l166_166048

open Finset

theorem distinct_four_digit_numbers_with_repeated_digit :
  let digits := {1, 2, 3, 4, 5}
  in (∃ d ∈ digits, 
        ∃ (positions : Finset (Fin 4)) (h_pos : positions.card = 2)
        (d1 d2 ∈ digits \ {d}),
        positions.pairwise (≠) ∧ (d = d1 ∨ d = d2) ∧ d1 ≠ d2)
      → 5 * 6 * 4 * 3 = 360 :=
by
  sorry

end distinct_four_digit_numbers_with_repeated_digit_l166_166048


namespace perpendicular_condition_l166_166046

def vector_a : ℝ × ℝ := (4, 3)
def vector_b : ℝ × ℝ := (-1, 2)

def add_vector_scaled (a b : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  (a.1 + k * b.1, a.2 + k * b.2)

def sub_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perpendicular_condition (k : ℝ) :
  dot_product (add_vector_scaled vector_a vector_b k) (sub_vector vector_a vector_b) = 0 ↔ k = 23 / 3 :=
by
  sorry

end perpendicular_condition_l166_166046


namespace length_of_real_axis_l166_166495

noncomputable def hyperbola_1 : Prop :=
  ∃ (x y: ℝ), (x^2 / 16) - (y^2 / 4) = 1

noncomputable def hyperbola_2 (a b: ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∃ (x y: ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def same_eccentricity (a b: ℝ) : Prop :=
  (1 + b^2 / a^2) = (1 + 1 / 4 / 16)

noncomputable def area_of_triangle (a b: ℝ) : Prop :=
  (a * b) = 32

theorem length_of_real_axis (a b: ℝ) (ha : 0 < a) (hb : 0 < b) :
  hyperbola_1 ∧ hyperbola_2 a b ha hb ∧ same_eccentricity a b ∧ area_of_triangle a b →
  2 * a = 16 :=
by
  sorry

end length_of_real_axis_l166_166495


namespace pentagonal_faces_count_l166_166985

theorem pentagonal_faces_count (x y : ℕ) (h : (5 * x + 6 * y) % 6 = 0) (h1 : ∃ v e f, v - e + f = 2 ∧ f = x + y ∧ e = (5 * x + 6 * y) / 2 ∧ v = (5 * x + 6 * y) / 3 ∧ (5 * x + 6 * y) / 3 * 3 = 5 * x + 6 * y) : 
  x = 12 :=
sorry

end pentagonal_faces_count_l166_166985


namespace expression_not_defined_l166_166933

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : ℝ := x^2 - 25*x + 125

-- Theorem statement that the expression is not defined for specific values of x
theorem expression_not_defined (x : ℝ) : quadratic_eq x = 0 ↔ (x = 5 ∨ x = 20) :=
by
  sorry

end expression_not_defined_l166_166933


namespace arccos_neg_one_eq_pi_l166_166160

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l166_166160


namespace arithmetic_sequence_sum_l166_166264

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 + a 6 = 18) :
  S 10 = 90 :=
sorry

end arithmetic_sequence_sum_l166_166264


namespace constant_term_in_binomial_expansion_l166_166366

theorem constant_term_in_binomial_expansion :
  (∃ c : ℚ, c = -5/2 ∧ (λ x : ℝ, (∑ i in finset.range 6, (choose 5 i * (1/(2:ℚ))^(5 - i) * (-1/(x^(1/3):ℚ))^i) * (x^(1/2))^(5 - i)) = c)) → true :=
sorry

end constant_term_in_binomial_expansion_l166_166366


namespace price_of_rice_packet_l166_166272

-- Definitions based on conditions
def initial_amount : ℕ := 500
def wheat_flour_price : ℕ := 25
def wheat_flour_quantity : ℕ := 3
def soda_price : ℕ := 150
def remaining_balance : ℕ := 235
def total_spending (P : ℕ) : ℕ := initial_amount - remaining_balance

-- Theorem to prove
theorem price_of_rice_packet (P : ℕ) (h: 2 * P + wheat_flour_quantity * wheat_flour_price + soda_price = total_spending P) : P = 20 :=
sorry

end price_of_rice_packet_l166_166272


namespace largest_and_smallest_A_l166_166769

noncomputable def is_coprime_with_12 (n : ℕ) : Prop := 
  Nat.gcd n 12 = 1

def problem_statement (A_max A_min : ℕ) : Prop :=
  ∃ B : ℕ, B > 44444444 ∧ is_coprime_with_12 B ∧
  (A_max = 9 * 10^7 + (B - 9) / 10) ∧
  (A_min = 1 * 10^7 + (B - 1) / 10)

theorem largest_and_smallest_A :
  problem_statement 99999998 14444446 := sorry

end largest_and_smallest_A_l166_166769


namespace other_number_is_36_l166_166537

theorem other_number_is_36 (hcf lcm given_number other_number : ℕ) 
  (hcf_val : hcf = 16) (lcm_val : lcm = 396) (given_number_val : given_number = 176) 
  (relation : hcf * lcm = given_number * other_number) : 
  other_number = 36 := 
by 
  sorry

end other_number_is_36_l166_166537


namespace at_least_one_admitted_prob_l166_166885

theorem at_least_one_admitted_prob (pA pB : ℝ) (hA : pA = 0.6) (hB : pB = 0.7) (independent : ∀ (P Q : Prop), P ∧ Q → P ∧ Q):
  (1 - ((1 - pA) * (1 - pB))) = 0.88 :=
by
  rw [hA, hB]
  -- more steps would follow in a complete proof
  sorry

end at_least_one_admitted_prob_l166_166885


namespace num_full_servings_l166_166131

-- Define the original amount of peanut butter as a rational number
def original_amount : ℚ := 35 + 2/3

-- Define the used amount of peanut butter as a rational number
def used_amount : ℚ := 5 + 1/3

-- Define the amount of peanut butter per serving as a rational number
def serving_size : ℚ := 3

-- Define the remaining peanut butter after using some for the recipe
def remaining_amount : ℚ := original_amount - used_amount

-- Define the total number of servings that can be made from the remaining amount
def num_servings : ℚ := remaining_amount / serving_size

-- State the theorem that we need to prove
theorem num_full_servings : num_servings.floor = 10 := 
by
  sorry

end num_full_servings_l166_166131


namespace minimum_a_value_l166_166035

theorem minimum_a_value (a : ℝ) : 
  (∀ (x y : ℝ), 0 < x → 0 < y → x^2 + 2 * x * y ≤ a * (x^2 + y^2)) ↔ a ≥ (Real.sqrt 5 + 1) / 2 := 
sorry

end minimum_a_value_l166_166035


namespace largest_A_smallest_A_l166_166768

noncomputable def is_coprime_with_12 (n : Nat) : Prop :=
  Nat.gcd n 12 = 1

noncomputable def rotated_number (n : Nat) : Option Nat :=
  if n < 10^7 then none else
  let b := n % 10
  let k := n / 10
  some (b * 10^7 + k)

noncomputable def satisfies_conditions (B : Nat) : Prop :=
  B > 44444444 ∧ is_coprime_with_12 B

theorem largest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 99999998 :=
sorry

theorem smallest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 14444446 :=
sorry

end largest_A_smallest_A_l166_166768


namespace one_set_working_communication_possible_l166_166288

variable (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1)

def P_A : ℝ := p^3
def P_B : ℝ := p^3
def P_not_A : ℝ := 1 - p^3
def P_not_B : ℝ := 1 - p^3

theorem one_set_working : 2 * P_A p - 2 * (P_A p)^2 = 2 * p^3 - 2 * p^6 :=
by 
  sorry

theorem communication_possible : 2 * P_A p - (P_A p)^2 = 2 * p^3 - p^6 :=
by 
  sorry

end one_set_working_communication_possible_l166_166288


namespace product_of_first_three_terms_is_960_l166_166259

-- Definitions from the conditions
def a₁ : ℤ := 20 - 6 * 2
def a₂ : ℤ := a₁ + 2
def a₃ : ℤ := a₂ + 2

-- Problem statement
theorem product_of_first_three_terms_is_960 : 
  a₁ * a₂ * a₃ = 960 :=
by
  sorry

end product_of_first_three_terms_is_960_l166_166259


namespace gerald_price_l166_166805

-- Define the conditions provided in the problem

def price_hendricks := 200
def discount_percent := 20
def discount_ratio := 0.80 -- since 20% less means Hendricks paid 80% of what Gerald paid

-- Question to be answered: Prove that the price Gerald paid equals $250
-- P is what Gerald paid

theorem gerald_price (P : ℝ) (h : price_hendricks = discount_ratio * P) : P = 250 :=
by
  sorry

end gerald_price_l166_166805


namespace arccos_neg_one_eq_pi_l166_166159

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l166_166159


namespace tangent_line_at_x_2_increasing_on_1_to_infinity_l166_166039

noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

-- Subpart I
theorem tangent_line_at_x_2 (a b : ℝ) :
  (a / 2 + 2 = 1) ∧ (2 + a * Real.log 2 = 2 + b) → (a = -2 ∧ b = -2 * Real.log 2) :=
by
  sorry

-- Subpart II
theorem increasing_on_1_to_infinity (a : ℝ) :
  (∀ x > 1, (x + a / x) ≥ 0) → (a ≥ -1) :=
by
  sorry

end tangent_line_at_x_2_increasing_on_1_to_infinity_l166_166039


namespace prob_below_8_correct_l166_166136

-- Defining the probabilities of hitting the 10, 9, and 8 rings
def prob_10 : ℝ := 0.20
def prob_9 : ℝ := 0.30
def prob_8 : ℝ := 0.10

-- Defining the event of scoring below 8
def prob_below_8 : ℝ := 1 - (prob_10 + prob_9 + prob_8)

-- The main theorem to prove: the probability of scoring below 8 is 0.40
theorem prob_below_8_correct : prob_below_8 = 0.40 :=
by 
  -- We need to show this proof in a separate proof phase
  sorry

end prob_below_8_correct_l166_166136


namespace max_value_of_f_l166_166320

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_of_f :
  ∃ x ∈ Set.Icc (0 : ℝ) 4, ∀ y ∈ Set.Icc (0 : ℝ) 4, f y ≤ f x ∧ f x = 1 / Real.exp 1 := 
by
  sorry

end max_value_of_f_l166_166320


namespace exists_max_piles_l166_166658

theorem exists_max_piles (k : ℝ) (hk : k < 2) : 
  ∃ Nk : ℕ, ∀ A : Multiset ℝ, 
    (∀ a ∈ A, ∃ m ∈ A, a ≤ k * m) → 
    A.card ≤ Nk :=
sorry

end exists_max_piles_l166_166658


namespace abc_sum_is_12_l166_166978

theorem abc_sum_is_12
  (a b c : ℕ)
  (h : 28 * a + 30 * b + 31 * c = 365) :
  a + b + c = 12 :=
by
  sorry

end abc_sum_is_12_l166_166978


namespace probability_of_disease_given_positive_test_l166_166989

-- Define the probabilities given in the problem
noncomputable def pr_D : ℝ := 1 / 1000
noncomputable def pr_Dc : ℝ := 1 - pr_D
noncomputable def pr_T_given_D : ℝ := 1
noncomputable def pr_T_given_Dc : ℝ := 0.05

-- Define the total probability of a positive test using the law of total probability
noncomputable def pr_T := 
  pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Using Bayes' theorem
noncomputable def pr_D_given_T := 
  pr_T_given_D * pr_D / pr_T

-- Theorem to prove the desired probability
theorem probability_of_disease_given_positive_test : 
  pr_D_given_T = 1 / 10 :=
by
  sorry

end probability_of_disease_given_positive_test_l166_166989


namespace x_squared_inverse_y_fourth_l166_166988

theorem x_squared_inverse_y_fourth (x y : ℝ) (k : ℝ) (h₁ : x = 8) (h₂ : y = 2) (h₃ : (x^2) * (y^4) = k) : x^2 = 4 :=
by
  sorry

end x_squared_inverse_y_fourth_l166_166988


namespace luke_hotdogs_ratio_l166_166236

-- Definitions
def hotdogs_per_sister : ℕ := 2
def total_sisters_hotdogs : ℕ := 2 * 2 -- Ella and Emma together
def hunter_hotdogs : ℕ := 6 -- 1.5 times the total of sisters' hotdogs
def total_hotdogs : ℕ := 14

-- Ratio proof problem statement
theorem luke_hotdogs_ratio :
  ∃ x : ℕ, total_hotdogs = total_sisters_hotdogs + 4 * x + hunter_hotdogs ∧ 
    (4 * x = 2 * 1 ∧ x = 1) := 
by 
  sorry

end luke_hotdogs_ratio_l166_166236


namespace inequality_solution_l166_166691

theorem inequality_solution (x y : ℝ) (h1 : y ≥ x^2 + 1) :
    2^y - 2 * Real.cos x + Real.sqrt (y - x^2 - 1) ≤ 0 ↔ x = 0 ∧ y = 1 :=
by
  sorry

end inequality_solution_l166_166691


namespace volume_of_parallelepiped_l166_166094

theorem volume_of_parallelepiped 
  (m n Q : ℝ) 
  (ratio_positive : 0 < m ∧ 0 < n)
  (Q_positive : 0 < Q)
  (h_square_area : ∃ a b : ℝ, a / b = m / n ∧ (a^2 + b^2) = Q) :
  ∃ (V : ℝ), V = (m * n * Q * Real.sqrt Q) / (m^2 + n^2) :=
sorry

end volume_of_parallelepiped_l166_166094


namespace similar_sizes_bound_l166_166659

theorem similar_sizes_bound (k : ℝ) (hk : k < 2) :
  ∃ (N_k : ℝ), ∀ (A : multiset ℝ), (∀ a ∈ A, a ≤ k * multiset.min A) → 
  A.card ≤ N_k := sorry

end similar_sizes_bound_l166_166659


namespace sprint_team_total_miles_l166_166876

theorem sprint_team_total_miles (number_of_people : ℝ) (miles_per_person : ℝ) 
  (h1 : number_of_people = 150.0) (h2 : miles_per_person = 5.0) : 
  number_of_people * miles_per_person = 750.0 :=
by
  rw [h1, h2]
  norm_num

end sprint_team_total_miles_l166_166876


namespace simplify_fraction_l166_166242

theorem simplify_fraction :
  5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l166_166242


namespace gerald_paid_l166_166807

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 :=
by
  sorry

end gerald_paid_l166_166807


namespace oranges_thrown_away_l166_166588

theorem oranges_thrown_away (original_oranges: ℕ) (new_oranges: ℕ) (total_oranges: ℕ) (x: ℕ)
  (h1: original_oranges = 5) (h2: new_oranges = 28) (h3: total_oranges = 31) :
  original_oranges - x + new_oranges = total_oranges → x = 2 :=
by
  intros h_eq
  -- Proof omitted
  sorry

end oranges_thrown_away_l166_166588


namespace both_true_sufficient_but_not_necessary_for_either_l166_166520

variable (p q : Prop)

theorem both_true_sufficient_but_not_necessary_for_either:
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end both_true_sufficient_but_not_necessary_for_either_l166_166520


namespace tangent_line_right_triangle_l166_166946

theorem tangent_line_right_triangle {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (tangent_condition : a^2 + b^2 = c^2) : 
  (abs c)^2 = (abs a)^2 + (abs b)^2 :=
by
  sorry

end tangent_line_right_triangle_l166_166946


namespace baker_sold_cakes_l166_166916

theorem baker_sold_cakes (S : ℕ) (h1 : 154 = S + 63) : S = 91 :=
by
  sorry

end baker_sold_cakes_l166_166916


namespace entire_meal_cost_correct_l166_166600

-- Define given conditions
def appetizer_cost : ℝ := 9.00
def entree_cost : ℝ := 20.00
def num_entrees : ℕ := 2
def dessert_cost : ℝ := 11.00
def tip_percentage : ℝ := 0.30

-- Calculate intermediate values
def total_cost_before_tip : ℝ := appetizer_cost + (entree_cost * num_entrees) + dessert_cost
def tip : ℝ := tip_percentage * total_cost_before_tip
def entire_meal_cost : ℝ := total_cost_before_tip + tip

-- Statement to be proved
theorem entire_meal_cost_correct : entire_meal_cost = 78.00 := by
  -- Proof will go here
  sorry

end entire_meal_cost_correct_l166_166600


namespace number_of_ways_to_form_team_l166_166820

theorem number_of_ways_to_form_team (boys girls : ℕ) (select_boys select_girls : ℕ)
    (H_boys : boys = 7) (H_girls : girls = 9) (H_select_boys : select_boys = 2) (H_select_girls : select_girls = 3) :
    (Nat.choose boys select_boys) * (Nat.choose girls select_girls) = 1764 := by
  rw [H_boys, H_girls, H_select_boys, H_select_girls]
  sorry

end number_of_ways_to_form_team_l166_166820


namespace parabola_units_shift_l166_166484

noncomputable def parabola_expression (A B : ℝ × ℝ) (x : ℝ) : ℝ :=
  let b := -5
  let c := 6
  x^2 + b * x + c

theorem parabola_units_shift (A B : ℝ × ℝ) (x : ℝ) (y : ℝ) :
  A = (2, 0) → B = (0, 6) → parabola_expression A B 4 = 2 →
  (y - 2 = 0) → true :=
by
  intro hA hB h4 hy
  sorry

end parabola_units_shift_l166_166484


namespace book_cost_proof_l166_166952

variable (C1 C2 : ℝ)

theorem book_cost_proof (h1 : C1 + C2 = 460)
                        (h2 : C1 * 0.85 = C2 * 1.19) :
    C1 = 268.53 := by
  sorry

end book_cost_proof_l166_166952


namespace vlad_taller_by_41_inches_l166_166275

/-- Vlad's height is 6 feet and 3 inches. -/
def vlad_height_feet : ℕ := 6

def vlad_height_inches : ℕ := 3

/-- Vlad's sister's height is 2 feet and 10 inches. -/
def sister_height_feet : ℕ := 2

def sister_height_inches : ℕ := 10

/-- There are 12 inches in a foot. -/
def inches_in_a_foot : ℕ := 12

/-- Convert height in feet and inches to total inches. -/
def convert_to_inches (feet inches : ℕ) : ℕ :=
  feet * inches_in_a_foot + inches

/-- Proof that Vlad is 41 inches taller than his sister. -/
theorem vlad_taller_by_41_inches : convert_to_inches vlad_height_feet vlad_height_inches - convert_to_inches sister_height_feet sister_height_inches = 41 :=
by
  -- Start the proof
  sorry

end vlad_taller_by_41_inches_l166_166275


namespace inequality_exponentiation_l166_166195

theorem inequality_exponentiation (a b c : ℝ) (ha : 0 < a) (hab : a < b) (hb : b < 1) (hc : c > 1) : 
  a * b^c > b * a^c := 
sorry

end inequality_exponentiation_l166_166195


namespace base_conversion_problem_l166_166001

def base_to_dec (base : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ x acc => x + base * acc) 0

theorem base_conversion_problem : 
  (base_to_dec 8 [2, 5, 3] : ℝ) / (base_to_dec 4 [1, 3] : ℝ) + 
  (base_to_dec 5 [1, 3, 2] : ℝ) / (base_to_dec 3 [2, 3] : ℝ) = 28.67 := by
  sorry

end base_conversion_problem_l166_166001


namespace tangent_line_iff_l166_166801

theorem tangent_line_iff (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 8 * y + 12 = 0 → ax + y + 2 * a = 0) ↔ a = -3 / 4 :=
by
  sorry

end tangent_line_iff_l166_166801


namespace range_of_m_l166_166486

theorem range_of_m :
  (∀ x : ℝ, (x > 0) → (x^2 - m * x + 4 ≥ 0)) ∧ (¬∃ x : ℝ, (x^2 - 2 * m * x + 7 * m - 10 = 0)) ↔ (2 < m ∧ m ≤ 4) :=
by
  sorry

end range_of_m_l166_166486


namespace equilibrium_possible_l166_166591

variables {a b θ : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : (b / 2) < a) (h4 : a ≤ b)

theorem equilibrium_possible :
  θ = 0 ∨ θ = Real.arccos ((b^2 + 2 * a^2) / (3 * a * b)) → 
  (b / 2) < a ∧ a ≤ b ∧ (0 ≤ θ ∧ θ ≤ π) :=
sorry

end equilibrium_possible_l166_166591


namespace max_k_divides_expression_l166_166309

theorem max_k_divides_expression : ∃ k, (∀ n : ℕ, n > 0 → 2^k ∣ (3^(2*n + 3) + 40*n - 27)) ∧ k = 6 :=
sorry

end max_k_divides_expression_l166_166309


namespace lives_after_game_l166_166898

theorem lives_after_game (l0 : ℕ) (ll : ℕ) (lg : ℕ) (lf : ℕ) : 
  l0 = 10 → ll = 4 → lg = 26 → lf = l0 - ll + lg → lf = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end lives_after_game_l166_166898


namespace courtyard_width_is_14_l166_166903

-- Given conditions
def length_courtyard := 24   -- 24 meters
def num_bricks := 8960       -- Total number of bricks

@[simp]
def brick_length_m : ℝ := 0.25  -- 25 cm in meters
@[simp]
def brick_width_m : ℝ := 0.15   -- 15 cm in meters

-- Correct answer
def width_courtyard : ℝ := 14

-- Prove that the width of the courtyard is 14 meters
theorem courtyard_width_is_14 : 
  (length_courtyard * width_courtyard) = (num_bricks * (brick_length_m * brick_width_m)) :=
by
  -- Lean proof will go here
  sorry

end courtyard_width_is_14_l166_166903


namespace bottle_caps_cost_l166_166312

-- Conditions
def cost_per_bottle_cap : ℕ := 2
def number_of_bottle_caps : ℕ := 6

-- Statement of the problem
theorem bottle_caps_cost : (cost_per_bottle_cap * number_of_bottle_caps) = 12 :=
by
  sorry

end bottle_caps_cost_l166_166312


namespace find_special_n_l166_166003

open Nat

def is_divisor (d n : ℕ) : Prop := n % d = 0

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def special_primes_condition (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_prime (d^2 - d + 1) ∧ is_prime (d^2 + d + 1)

theorem find_special_n (n : ℕ) (h : n > 1) :
  special_primes_condition n → n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end find_special_n_l166_166003


namespace lizzys_shipping_cost_l166_166233

def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def cost_per_crate : ℝ := 1.5

def number_of_crates (W w : ℝ) : ℝ := W / w
def total_shipping_cost (c n : ℝ) : ℝ := c * n

theorem lizzys_shipping_cost :
  total_shipping_cost cost_per_crate (number_of_crates total_weight weight_per_crate) = 27 := 
by {
  sorry
}

end lizzys_shipping_cost_l166_166233


namespace least_tiles_required_l166_166895

def floor_length : ℕ := 5000
def floor_breadth : ℕ := 1125
def gcd_floor : ℕ := Nat.gcd floor_length floor_breadth
def tile_area : ℕ := gcd_floor ^ 2
def floor_area : ℕ := floor_length * floor_breadth
def tiles_count : ℕ := floor_area / tile_area

theorem least_tiles_required : tiles_count = 360 :=
by
  sorry

end least_tiles_required_l166_166895


namespace rectangle_area_difference_196_l166_166078

noncomputable def max_min_area_difference (P : ℕ) (A_max A_min : ℕ) : Prop :=
  ( ∃ l w : ℕ, 2 * l + 2 * w = P ∧ A_max = l * w ) ∧
  ( ∃ l' w' : ℕ, 2 * l' + 2 * w' = P ∧ A_min = l' * w' ) ∧
  (A_max - A_min = 196)

theorem rectangle_area_difference_196 : max_min_area_difference 60 225 29 :=
by
  sorry

end rectangle_area_difference_196_l166_166078


namespace number_of_two_bedroom_units_l166_166123

-- Define the total number of units and costs
variables (x y : ℕ)
def total_units := (x + y = 12)
def total_cost := (360 * x + 450 * y = 4950)

-- The target is to prove that there are 7 two-bedroom units
theorem number_of_two_bedroom_units : total_units ∧ total_cost → y = 7 :=
by
  sorry

end number_of_two_bedroom_units_l166_166123


namespace sum_of_local_values_l166_166732

def local_value (digit place_value : ℕ) : ℕ := digit * place_value

theorem sum_of_local_values :
  local_value 2 1000 + local_value 3 100 + local_value 4 10 + local_value 5 1 = 2345 :=
by
  sorry

end sum_of_local_values_l166_166732


namespace intersection_of_sets_l166_166330

def setA : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x : ℝ | 2 < x }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_sets_l166_166330


namespace blueberries_cartons_proof_l166_166595

def total_needed_cartons : ℕ := 26
def strawberries_cartons : ℕ := 10
def cartons_to_buy : ℕ := 7

theorem blueberries_cartons_proof :
  strawberries_cartons + cartons_to_buy + 9 = total_needed_cartons :=
by
  sorry

end blueberries_cartons_proof_l166_166595


namespace largest_whole_number_lt_150_l166_166889

theorem largest_whole_number_lt_150 : 
  ∃ x : ℕ, (9 * x < 150) ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) :=
  sorry

end largest_whole_number_lt_150_l166_166889


namespace sum_of_divisors_of_11_squared_l166_166882

theorem sum_of_divisors_of_11_squared (a b c : ℕ) (h1 : a ∣ 11^2) (h2 : b ∣ 11^2) (h3 : c ∣ 11^2) (h4 : a * b * c = 11^2) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) :
  a + b + c = 23 :=
sorry

end sum_of_divisors_of_11_squared_l166_166882


namespace basketball_free_throws_l166_166697

theorem basketball_free_throws:
  ∀ (a b x : ℕ),
    3 * b = 4 * a →
    x = 2 * a →
    2 * a + 3 * b + x = 65 →
    x = 18 := 
by
  intros a b x h1 h2 h3
  sorry

end basketball_free_throws_l166_166697


namespace median_number_of_books_l166_166209

theorem median_number_of_books :
  ∀ (students : List ℚ), 
  students = [1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8] →
  students.length = 12 →
  let sorted_students := students.sorted in
  let n := sorted_students.length in
  2 ∣ n →
  (sorted_students[n/2 - 1] + sorted_students[n/2]) / 2 = (4.5 : ℚ) :=
by
  intro students h1 h2 sorted_students n h3
  have h4 : sorted_students = [1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8] := by
    sorry -- the sorted list equal to the given list
  have h5 : n = 12 := by
    sorry -- the length of the list is 12
  have h6 : n / 2 - 1 = 5 := by
    sorry -- computing the middle index 1 (0-based)
  have h7 : n / 2 = 6 := by
    sorry -- computing the middle index 2 (0-based)
  have h8 : sorted_students[5] = 4 := by
    sorry -- the 6th element (0-based) is 4
  have h9 : sorted_students[6] = 5 := by
    sorry -- the 7th element (0-based) is 5
  have h10 : (4 + 5) / 2 = 4.5 := by
    sorry -- the median computation
  exact h10

end median_number_of_books_l166_166209


namespace simplify_expression_l166_166673

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l166_166673


namespace remainder_23_2057_mod_25_l166_166435

theorem remainder_23_2057_mod_25 : (23^2057) % 25 = 16 := 
by
  sorry

end remainder_23_2057_mod_25_l166_166435


namespace prize_amount_l166_166910

theorem prize_amount (P : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : n = 40)
  (h2 : a = 40)
  (h3 : b = (2 / 5) * P)
  (h4 : c = (3 / 5) * 40)
  (h5 : b / c = 120) :
  P = 7200 := 
sorry

end prize_amount_l166_166910


namespace choose_questions_l166_166507

theorem choose_questions (q : ℕ) (last : ℕ) (total : ℕ) (chosen : ℕ) 
  (condition : q ≥ 3) 
  (n : last = 5) 
  (m : total = 10) 
  (k : chosen = 6) : 
  ∃ (ways : ℕ), ways = 155 := 
by
  sorry

end choose_questions_l166_166507


namespace right_triangle_ABC_l166_166335

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Points definitions
def point_A : ℝ × ℝ := (1, 2)
def point_on_line : ℝ × ℝ := (5, -2)

-- Points B and C on the parabola with parameters t and s respectively
def point_B (t : ℝ) : ℝ × ℝ := (t^2, 2 * t)
def point_C (s : ℝ) : ℝ × ℝ := (s^2, 2 * s)

-- Line equation passing through points B and C
def line_eq (s t : ℝ) (x y : ℝ) : Prop :=
  2 * x - (s + t) * y + 2 * s * t = 0

-- Proof goal: Show that triangle ABC is a right triangle
theorem right_triangle_ABC
  (t s : ℝ)
  (hB : parabola (point_B t).1 (point_B t).2)
  (hC : parabola (point_C s).1 (point_C s).2)
  (hlt : point_on_line.1 = (5 : ℝ))
  (hlx : line_eq s t point_on_line.1 point_on_line.2)
  : let A := point_A
    let B := point_B t
    let C := point_C s
    -- Conclusion: triangle ABC is a right triangle
    k_AB * k_AC = -1 :=
  sorry
  where k_AB := (2 * t - 2) / (t^2 - 1)
        k_AC := (2 * s - 2) / (s^2 - 1)
        rel_t_s := (s + 1) * (t + 1) = -4

end right_triangle_ABC_l166_166335


namespace cricket_innings_l166_166752

theorem cricket_innings (n : ℕ) 
  (average_run : ℕ := 40) 
  (next_innings_run : ℕ := 84) 
  (new_average_run : ℕ := 44) :
  (40 * n + 84) / (n + 1) = 44 ↔ n = 10 := 
by
  sorry

end cricket_innings_l166_166752


namespace widget_production_l166_166692

theorem widget_production (p q r s t : ℕ) :
  (s * q * t) / (p * r) = (sqt / pr) := 
sorry

end widget_production_l166_166692


namespace rectangle_quadrilateral_inequality_l166_166605

theorem rectangle_quadrilateral_inequality 
  (a b c d : ℝ)
  (h_a : 0 < a) (h_a_bound : a < 3)
  (h_b : 0 < b) (h_b_bound : b < 4)
  (h_c : 0 < c) (h_c_bound : c < 3)
  (h_d : 0 < d) (h_d_bound : d < 4) :
  25 ≤ ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) ∧
  ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) < 50 :=
by 
  sorry

end rectangle_quadrilateral_inequality_l166_166605


namespace weight_of_new_person_l166_166743

theorem weight_of_new_person 
  (avg_weight_increase : ℝ)
  (old_weight : ℝ) 
  (num_people : ℕ)
  (new_weight_increase : ℝ)
  (total_weight_increase : ℝ)  
  (W : ℝ)
  (h1 : avg_weight_increase = 1.8)
  (h2 : old_weight = 69)
  (h3 : num_people = 6) 
  (h4 : new_weight_increase = num_people * avg_weight_increase) 
  (h5 : total_weight_increase = new_weight_increase)
  (h6 : W = old_weight + total_weight_increase)
  : W = 79.8 := 
by
  sorry

end weight_of_new_person_l166_166743


namespace granger_buys_3_jars_of_peanut_butter_l166_166203

theorem granger_buys_3_jars_of_peanut_butter :
  ∀ (spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count: ℕ),
    spam_cost = 3 → peanut_butter_cost = 5 → bread_cost = 2 →
    spam_count = 12 → loaf_count = 4 → total_cost = 59 →
    spam_cost * spam_count + bread_cost * loaf_count + peanut_butter_cost * peanut_butter_count = total_cost →
    peanut_butter_count = 3 :=
by
  intros spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count
  intros hspam_cost hpeanut_butter_cost hbread_cost hspam_count hloaf_count htotal_cost htotal
  sorry  -- The proof step is omitted as requested.

end granger_buys_3_jars_of_peanut_butter_l166_166203


namespace number_of_games_between_men_and_women_l166_166504

theorem number_of_games_between_men_and_women
    (W M : ℕ)
    (hW : W * (W - 1) / 2 = 72)
    (hM : M * (M - 1) / 2 = 288) :
  M * W = 288 :=
by
  sorry

end number_of_games_between_men_and_women_l166_166504


namespace sum_of_interior_angles_heptagon_l166_166265

theorem sum_of_interior_angles_heptagon (n : ℕ) (h : n = 7) : (n - 2) * 180 = 900 := by
  sorry

end sum_of_interior_angles_heptagon_l166_166265


namespace total_pencils_l166_166457

theorem total_pencils (reeta_pencils anika_pencils kamal_pencils : ℕ) :
  reeta_pencils = 30 →
  anika_pencils = 2 * reeta_pencils + 4 →
  kamal_pencils = 3 * reeta_pencils - 2 →
  reeta_pencils + anika_pencils + kamal_pencils = 182 :=
by
  intros h_reeta h_anika h_kamal
  sorry

end total_pencils_l166_166457


namespace distance_A_beats_B_l166_166577

noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem distance_A_beats_B :
  let distance_A := 5 -- km
  let time_A := 10 / 60 -- hours (10 minutes)
  let time_B := 14 / 60 -- hours (14 minutes)
  let speed_A := speed distance_A time_A
  let speed_B := speed distance_A time_B
  let distance_A_in_time_B := speed_A * time_B
  distance_A_in_time_B - distance_A = 2 := -- km
by
  sorry

end distance_A_beats_B_l166_166577


namespace directrix_of_parabola_l166_166182

theorem directrix_of_parabola (x y : ℝ) : y = 3 * x^2 - 6 * x + 1 → y = -25 / 12 :=
sorry

end directrix_of_parabola_l166_166182


namespace gcd_102_238_l166_166727

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l166_166727


namespace geometric_sequence_ab_product_l166_166514

theorem geometric_sequence_ab_product (a b : ℝ) (h₁ : 2 ≤ a) (h₂ : a ≤ 16) (h₃ : 2 ≤ b) (h₄ : b ≤ 16)
  (h₅ : ∃ r : ℝ, a = 2 * r ∧ b = 2 * r^2 ∧ 16 = 2 * r^3) : a * b = 32 :=
by
  sorry

end geometric_sequence_ab_product_l166_166514


namespace greatest_difference_l166_166546

theorem greatest_difference (n m : ℕ) (hn : 1023 = 17 * n + m) (hn_pos : 0 < n) (hm_pos : 0 < m) : n - m = 57 :=
sorry

end greatest_difference_l166_166546


namespace find_principal_amount_l166_166442

variable (P : ℝ)
variable (R : ℝ := 5)
variable (T : ℝ := 13)
variable (SI : ℝ := 1300)

theorem find_principal_amount (h1 : SI = (P * R * T) / 100) : P = 2000 :=
sorry

end find_principal_amount_l166_166442


namespace reciprocal_of_neg_three_l166_166708

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l166_166708


namespace sum_gcd_lcm_168_l166_166104

def gcd_54_72 : ℕ := Nat.gcd 54 72

def lcm_50_15 : ℕ := Nat.lcm 50 15

def sum_gcd_lcm : ℕ := gcd_54_72 + lcm_50_15

theorem sum_gcd_lcm_168 : sum_gcd_lcm = 168 := by
  sorry

end sum_gcd_lcm_168_l166_166104


namespace rectangle_perimeter_l166_166217

theorem rectangle_perimeter (z w : ℝ) (h : z > w) :
  (2 * ((z - w) + w)) = 2 * z := by
  sorry

end rectangle_perimeter_l166_166217


namespace time_for_B_to_complete_work_l166_166749

theorem time_for_B_to_complete_work 
  (A B C : ℝ)
  (h1 : A = 1 / 4) 
  (h2 : B + C = 1 / 3) 
  (h3 : A + C = 1 / 2) :
  1 / B = 12 :=
by
  -- Proof is omitted, as per instruction.
  sorry

end time_for_B_to_complete_work_l166_166749


namespace value_of_a_minus_b_l166_166937

variables (a b : ℝ)

theorem value_of_a_minus_b (h1 : abs a = 3) (h2 : abs b = 5) (h3 : a > b) : a - b = 8 :=
sorry

end value_of_a_minus_b_l166_166937


namespace black_lambs_count_l166_166469

-- Definitions based on the conditions given
def total_lambs : ℕ := 6048
def white_lambs : ℕ := 193

-- Theorem statement
theorem black_lambs_count : total_lambs - white_lambs = 5855 :=
by 
  -- the proof would be provided here
  sorry

end black_lambs_count_l166_166469


namespace reciprocal_of_neg3_l166_166702

theorem reciprocal_of_neg3 : ∃ x : ℝ, (-3 : ℝ) * x = 1 :=  
begin
  use -1/3,
  norm_num,
end

end reciprocal_of_neg3_l166_166702


namespace line_passes_through_parabola_vertex_l166_166187

theorem line_passes_through_parabola_vertex : 
  ∃ (c : ℝ), (∀ (x : ℝ), y = 2 * x + c → ∃ (x0 : ℝ), (x0 = 0 ∧ y = c^2)) ∧ 
  (∀ (c1 c2 : ℝ), (y = 2 * x + c1 ∧ y = 2 * x + c2 → c1 = c2)) → 
  ∃ c : ℝ, c = 0 ∨ c = 1 :=
by 
  -- Proof should be inserted here
  sorry

end line_passes_through_parabola_vertex_l166_166187


namespace find_base_of_exponential_l166_166940

theorem find_base_of_exponential (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a ≠ 1) 
  (h₃ : a ^ 2 = 1 / 16) : 
  a = 1 / 4 := 
sorry

end find_base_of_exponential_l166_166940


namespace minor_axis_of_ellipse_l166_166454

noncomputable def length_minor_axis 
    (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (p3 : ℝ × ℝ) (p4 : ℝ × ℝ) (p5 : ℝ × ℝ) : ℝ :=
if h : (p1, p2, p3, p4, p5) = ((1, 0), (1, 3), (4, 0), (4, 3), (6, 1.5)) then 3 else 0

theorem minor_axis_of_ellipse (p1 p2 p3 p4 p5 : ℝ × ℝ) :
  p1 = (1, 0) → p2 = (1, 3) → p3 = (4, 0) → p4 = (4, 3) → p5 = (6, 1.5) →
  length_minor_axis p1 p2 p3 p4 p5 = 3 :=
by sorry

end minor_axis_of_ellipse_l166_166454


namespace gcd_324_243_135_l166_166015

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 := by
  sorry

end gcd_324_243_135_l166_166015


namespace pile_limit_exists_l166_166660

noncomputable def log_floor (b x : ℝ) : ℤ :=
  Int.floor (Real.log x / Real.log b)

theorem pile_limit_exists (k : ℝ) (hk : k < 2) : ∃ Nk : ℤ, 
  Nk = 2 * (log_floor (2 / k) 2 + 1) := 
  by
    sorry

end pile_limit_exists_l166_166660


namespace cos_alpha_plus_beta_l166_166029

variable (α β : ℝ)
variable (hα : Real.sin α = (Real.sqrt 5) / 5)
variable (hβ : Real.sin β = (Real.sqrt 10) / 10)
variable (hα_obtuse : π / 2 < α ∧ α < π)
variable (hβ_obtuse : π / 2 < β ∧ β < π)

theorem cos_alpha_plus_beta : Real.cos (α + β) = Real.sqrt 2 / 2 ∧ α + β = 7 * π / 4 := by
  sorry

end cos_alpha_plus_beta_l166_166029


namespace parabola_equation_l166_166800

theorem parabola_equation (P : ℝ × ℝ) :
  let d1 := dist P (-3, 0)
  let d2 := abs (P.1 - 2)
  (d1 = d2 + 1 ↔ P.2^2 = -12 * P.1) :=
by
  intro d1 d2
  sorry

end parabola_equation_l166_166800


namespace complete_square_form_l166_166780

theorem complete_square_form :
  ∀ x : ℝ, (3 * x^2 - 6 * x + 2 = 0) → (x - 1)^2 = (1 / 3) :=
by
  intro x h
  sorry

end complete_square_form_l166_166780


namespace cos_diff_half_l166_166789

theorem cos_diff_half (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1 / 2)
  (h2 : Real.sin α + Real.sin β = Real.sqrt 3 / 2) :
  Real.cos (α - β) = -1 / 2 :=
by
  sorry

end cos_diff_half_l166_166789


namespace right_triangle_wy_expression_l166_166376

theorem right_triangle_wy_expression (α β : ℝ) (u v w y : ℝ)
    (h1 : (∀ x : ℝ, x^2 - u * x + v = 0 → x = Real.sin α ∨ x = Real.sin β))
    (h2 : (∀ x : ℝ, x^2 - w * x + y = 0 → x = Real.cos α ∨ x = Real.cos β))
    (h3 : α + β = Real.pi / 2) :
    w * y = u * v :=
sorry

end right_triangle_wy_expression_l166_166376


namespace find_cubic_polynomial_l166_166013

theorem find_cubic_polynomial (a b c d : ℚ) :
  (a + b + c + d = -5) →
  (8 * a + 4 * b + 2 * c + d = -8) →
  (27 * a + 9 * b + 3 * c + d = -17) →
  (64 * a + 16 * b + 4 * c + d = -34) →
  a = -1/3 ∧ b = -1 ∧ c = -2/3 ∧ d = -3 :=
by
  intros h1 h2 h3 h4
  sorry

end find_cubic_polynomial_l166_166013


namespace minimum_value_expression_l166_166375

theorem minimum_value_expression (α β : ℝ) : (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 144 :=
by
  sorry

end minimum_value_expression_l166_166375


namespace find_positive_integers_l166_166472

theorem find_positive_integers (n : ℕ) (h_pos : n > 0) : 
  (∃ d : ℕ, ∀ k : ℕ, 6^n + 1 = d * (10^k - 1) / 9 → d = 7) → 
  n = 1 ∨ n = 5 :=
sorry

end find_positive_integers_l166_166472


namespace general_term_formula_l166_166194

noncomputable def xSeq : ℕ → ℝ
| 0       => 3
| (n + 1) => (xSeq n)^2 + 2 / (2 * (xSeq n) - 1)

theorem general_term_formula (n : ℕ) : 
  xSeq n = (2 * 2^2^n + 1) / (2^2^n - 1) := 
sorry

end general_term_formula_l166_166194


namespace reciprocal_of_neg3_l166_166711

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l166_166711


namespace simplify_expression_l166_166986

theorem simplify_expression (x y : ℝ) :
  ((x + y)^2 - y * (2 * x + y) - 6 * x) / (2 * x) = (1 / 2) * x - 3 :=
by
  sorry

end simplify_expression_l166_166986


namespace find_k_l166_166643

noncomputable def line1_slope : ℝ := -1
noncomputable def line2_slope (k : ℝ) : ℝ := -k / 3

theorem find_k (k : ℝ) : 
  (line2_slope k) * line1_slope = -1 → k = -3 := 
by
  sorry

end find_k_l166_166643


namespace evaluate_expression_l166_166416

theorem evaluate_expression : 202 - 101 + 9 = 110 :=
by
  sorry

end evaluate_expression_l166_166416


namespace card_at_position_52_l166_166966

def cards_order : List String := ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

theorem card_at_position_52 : cards_order[(52 % 13)] = "A" :=
by
  -- proof will be added here
  sorry

end card_at_position_52_l166_166966


namespace total_passengers_correct_l166_166373

-- Definition of the conditions
def passengers_on_time : ℕ := 14507
def passengers_late : ℕ := 213
def total_passengers : ℕ := passengers_on_time + passengers_late

-- Theorem statement
theorem total_passengers_correct : total_passengers = 14720 := by
  sorry

end total_passengers_correct_l166_166373


namespace line_intersects_x_axis_l166_166145

theorem line_intersects_x_axis (x y : ℝ) (h : 5 * y - 6 * x = 15) (hy : y = 0) : x = -2.5 ∧ y = 0 := 
by
  sorry

end line_intersects_x_axis_l166_166145


namespace distance_traveled_downstream_l166_166445

noncomputable def boat_speed_in_still_water : ℝ := 12
noncomputable def current_speed : ℝ := 4
noncomputable def travel_time_in_minutes : ℝ := 18
noncomputable def travel_time_in_hours : ℝ := travel_time_in_minutes / 60

theorem distance_traveled_downstream :
  let effective_speed := boat_speed_in_still_water + current_speed
  let distance := effective_speed * travel_time_in_hours
  distance = 4.8 := 
by
  sorry

end distance_traveled_downstream_l166_166445


namespace remaining_cookies_l166_166237

theorem remaining_cookies : 
  let naomi_cookies := 53
  let oliver_cookies := 67
  let penelope_cookies := 29
  let total_cookies := naomi_cookies + oliver_cookies + penelope_cookies
  let package_size := 15
  total_cookies % package_size = 14 :=
by
  sorry

end remaining_cookies_l166_166237


namespace cost_of_four_dozen_apples_l166_166914

-- Define the given conditions and problem
def half_dozen_cost : ℚ := 4.80 -- cost of half a dozen apples
def full_dozen_cost : ℚ := half_dozen_cost / 0.5
def four_dozen_cost : ℚ := 4 * full_dozen_cost

-- Statement of the theorem to prove
theorem cost_of_four_dozen_apples : four_dozen_cost = 38.40 :=
by
  sorry

end cost_of_four_dozen_apples_l166_166914


namespace compute_S_15_l166_166169

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def first_element_in_set (n : ℕ) : ℕ := sum_first_n (n - 1) + 1

def last_element_in_set (n : ℕ) : ℕ := first_element_in_set n + n - 1

def S (n : ℕ) : ℕ := n * (first_element_in_set n + last_element_in_set n) / 2

theorem compute_S_15 : S 15 = 1695 := by
  sorry

end compute_S_15_l166_166169


namespace ratio_of_a_b_l166_166636

-- Define the problem
theorem ratio_of_a_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it.
  sorry

end ratio_of_a_b_l166_166636


namespace ratio_is_one_to_two_l166_166425

def valentina_share_to_whole_ratio (valentina_share : ℕ) (whole_burger : ℕ) : ℕ × ℕ :=
  (valentina_share / (Nat.gcd valentina_share whole_burger), 
   whole_burger / (Nat.gcd valentina_share whole_burger))

theorem ratio_is_one_to_two : valentina_share_to_whole_ratio 6 12 = (1, 2) := 
  by
  sorry

end ratio_is_one_to_two_l166_166425


namespace reciprocal_of_neg_three_l166_166704

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l166_166704


namespace total_estate_value_l166_166849

theorem total_estate_value 
  (estate : ℝ)
  (daughter_share son_share wife_share brother_share nanny_share : ℝ)
  (h1 : daughter_share + son_share = (3/5) * estate)
  (h2 : daughter_share = 5 * son_share / 2)
  (h3 : wife_share = 3 * son_share)
  (h4 : brother_share = daughter_share)
  (h5 : nanny_share = 400) :
  estate = 825 := by
  sorry

end total_estate_value_l166_166849


namespace value_of_a_l166_166541

def quadratic_vertex (a b c : ℤ) (x : ℤ) : ℤ :=
  a * x^2 + b * x + c

def vertex_form (a h k x : ℤ) : ℤ :=
  a * (x - h)^2 + k

theorem value_of_a (a b c : ℤ) (h k x1 y1 x2 y2 : ℤ) (H_vert : h = 2) (H_vert_val : k = 3)
  (H_point : x1 = 1) (H_point_val : y1 = 5) (H_graph : ∀ x, quadratic_vertex a b c x = vertex_form a h k x) :
  a = 2 :=
by
  sorry

end value_of_a_l166_166541


namespace exists_positive_integer_m_l166_166897

noncomputable def d (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r - 1)
noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d
noncomputable def g_n (n : ℕ) (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r ^ (n - 1))

theorem exists_positive_integer_m (a1 g1 : ℝ) (r : ℝ) (h0 : g1 ≠ 0) (h1 : a1 = g1) (h2 : a2 = g2)
(h3 : a_n 10 a1 (d g1 r) = g_n 3 g1 r) :
  ∀ (p : ℕ), ∃ (m : ℕ), g_n p g1 r = a_n m a1 (d g1 r) := by
  sorry

end exists_positive_integer_m_l166_166897


namespace evaluate_product_at_3_l166_166010

theorem evaluate_product_at_3 : 
  let n := 3 in
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 :=
by 
  let n := 3
  sorry

end evaluate_product_at_3_l166_166010


namespace greatest_prime_factor_of_n_l166_166447

noncomputable def n : ℕ := 4^17 - 2^29

theorem greatest_prime_factor_of_n :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ n → q ≤ p :=
sorry

end greatest_prime_factor_of_n_l166_166447


namespace reciprocal_of_neg3_l166_166710

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l166_166710


namespace simplify_expression_l166_166674

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l166_166674


namespace minimum_value_of_ratio_l166_166796

theorem minimum_value_of_ratio 
  {a b c : ℝ} (h_a : a ≠ 0) 
  (h_f'0 : 2 * a * 0 + b > 0)
  (h_f_nonneg : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  (∃ x : ℝ, a * x^2 + b * x + c ≥ 0) ∧ (1 + (a + c) / b = 2) := sorry

end minimum_value_of_ratio_l166_166796


namespace remainder_with_conditions_l166_166437

theorem remainder_with_conditions (a b c d : ℕ) (h1 : a % 53 = 33) (h2 : b % 53 = 15) (h3 : c % 53 = 27) (h4 : d % 53 = 8) :
  ((a + b + c + d + 10) % 53) = 40 :=
by
  sorry

end remainder_with_conditions_l166_166437


namespace larger_cylinder_volume_l166_166423

theorem larger_cylinder_volume (v: ℝ) (r: ℝ) (R: ℝ) (h: ℝ) (hR : R = 2 * r) (hv : v = 100) : 
  π * R^2 * h = 4 * v := 
by 
  sorry

end larger_cylinder_volume_l166_166423


namespace loss_per_meter_is_5_l166_166758

-- Define the conditions
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 50
def quantity : ℕ := 400

-- Define the statement to prove (question == answer given conditions)
theorem loss_per_meter_is_5 : 
  ((cost_price_per_meter * quantity - selling_price) / quantity) = 5 := 
by
  sorry

end loss_per_meter_is_5_l166_166758


namespace bahs_equal_to_yahs_l166_166639

theorem bahs_equal_to_yahs (bahs rahs yahs : ℝ) 
  (h1 : 18 * bahs = 30 * rahs) 
  (h2 : 6 * rahs = 10 * yahs) : 
  1200 * yahs = 432 * bahs := 
by
  sorry

end bahs_equal_to_yahs_l166_166639


namespace count_ways_with_3_in_M_count_ways_with_2_in_M_l166_166458

structure ArrangementConfig where
  positions : Fin 9 → ℕ
  unique_positions : ∀ (i j : Fin 9) (hi hj : i ≠ j), positions i ≠ positions j
  no_adjacent_same : ∀ (i : Fin 8), positions i ≠ positions (i + 1)

def count_arrangements (fixed_value : ℕ) (fixed_position : Fin 9) : ℕ :=
  -- Implementation of counting the valid arrangements
  sorry

theorem count_ways_with_3_in_M : count_arrangements 3 0 = 6 := sorry

theorem count_ways_with_2_in_M : count_arrangements 2 0 = 12 := sorry

end count_ways_with_3_in_M_count_ways_with_2_in_M_l166_166458


namespace sqrt_solution_l166_166638

theorem sqrt_solution (x : ℝ) (h : x = Real.sqrt (1 + x)) : 1 < x ∧ x < 2 :=
by
  sorry

end sqrt_solution_l166_166638


namespace Carson_age_l166_166302

theorem Carson_age {Aunt_Anna_Age : ℕ} (h1 : Aunt_Anna_Age = 60) 
                   {Maria_Age : ℕ} (h2 : Maria_Age = 2 * Aunt_Anna_Age / 3) 
                   {Carson_Age : ℕ} (h3 : Carson_Age = Maria_Age - 7) : 
                   Carson_Age = 33 := by sorry

end Carson_age_l166_166302


namespace quadratic_m_value_l166_166642

theorem quadratic_m_value (m : ℕ) :
  (∃ x : ℝ, x^(m + 1) - (m + 1) * x - 2 = 0) →
  m + 1 = 2 →
  m = 1 :=
by {
  sorry
}

end quadratic_m_value_l166_166642


namespace number_of_common_tangents_l166_166956

/-- Define the circle C1 with center (2, -1) and radius 2. -/
def C1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 4}

/-- Define the symmetry line x + y - 3 = 0. -/
def symmetry_line := {p : ℝ × ℝ | p.1 + p.2 = 3}

/-- Circle C2 is symmetric to C1 about the line x + y = 3. -/
def C2 := {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 - 1)^2 = 4}

/-- Circle C3 with the given condition MA^2 + MO^2 = 10 for any point M on the circle. 
    A(0, 2) and O is the origin. -/
def C3 := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 4}

/-- The number of common tangents between circle C2 and circle C3 is 3. -/
theorem number_of_common_tangents
  (C1_sym_C2 : ∀ p : ℝ × ℝ, p ∈ C1 ↔ p ∈ C2)
  (M_on_C3 : ∀ M : ℝ × ℝ, M ∈ C3 → ((M.1)^2 + (M.2 - 2)^2) + ((M.1)^2 + (M.2)^2) = 10) :
  ∃ tangents : ℕ, tangents = 3 :=
sorry

end number_of_common_tangents_l166_166956


namespace tensor_12_9_l166_166467

def tensor (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

theorem tensor_12_9 : tensor 12 9 = 13 + 7 / 9 :=
by
  sorry

end tensor_12_9_l166_166467


namespace irrational_infinitely_many_approximations_l166_166670

theorem irrational_infinitely_many_approximations (x : ℝ) (hx : Irrational x) (hx_pos : 0 < x) :
  ∃ᶠ (q : ℕ) in at_top, ∃ p : ℤ, |x - p / q| < 1 / q^2 :=
sorry

end irrational_infinitely_many_approximations_l166_166670


namespace max_red_balls_l166_166446

theorem max_red_balls (R B G : ℕ) (h1 : G = 12) (h2 : R + B + G = 28) (h3 : R + G < 24) : R ≤ 11 := 
by
  sorry

end max_red_balls_l166_166446


namespace milk_transfer_equal_l166_166594

theorem milk_transfer_equal (A B C x : ℕ) (hA : A = 1200) (hB : B = A - 750) (hC : C = A - B) (h_eq : B + x = C - x) :
  x = 150 :=
by
  sorry

end milk_transfer_equal_l166_166594


namespace lucy_total_packs_l166_166842

-- Define the number of packs of cookies Lucy bought
def packs_of_cookies : ℕ := 12

-- Define the number of packs of noodles Lucy bought
def packs_of_noodles : ℕ := 16

-- Define the total number of packs of groceries Lucy bought
def total_packs_of_groceries : ℕ := packs_of_cookies + packs_of_noodles

-- Proof statement: The total number of packs of groceries Lucy bought is 28
theorem lucy_total_packs : total_packs_of_groceries = 28 := by
  sorry

end lucy_total_packs_l166_166842


namespace number_not_equal_54_l166_166880

def initial_number : ℕ := 12
def target_number : ℕ := 54
def total_time : ℕ := 60

theorem number_not_equal_54 (n : ℕ) (time : ℕ) : (time = total_time) → (n = initial_number) → 
  (∀ t : ℕ, t ≤ time → (n = n * 2 ∨ n = n / 2 ∨ n = n * 3 ∨ n = n / 3)) → n ≠ target_number :=
by
  sorry

end number_not_equal_54_l166_166880


namespace gamma_distribution_moments_l166_166477

noncomputable def gamma_density (α β x : ℝ) : ℝ :=
  (1 / (β ^ (α + 1) * Real.Gamma (α + 1))) * x ^ α * Real.exp (-x / β)

open Real

theorem gamma_distribution_moments (α β : ℝ) (x_bar D_B : ℝ) (hα : α > -1) (hβ : β > 0) :
  α = x_bar ^ 2 / D_B - 1 ∧ β = D_B / x_bar :=
by
  sorry

end gamma_distribution_moments_l166_166477


namespace recurring_decimal_product_l166_166178

theorem recurring_decimal_product : (0.3333333333 : ℝ) * (0.4545454545 : ℝ) = (5 / 33 : ℝ) :=
sorry

end recurring_decimal_product_l166_166178


namespace vlad_taller_by_41_inches_l166_166276

/-- Vlad's height is 6 feet and 3 inches. -/
def vlad_height_feet : ℕ := 6

def vlad_height_inches : ℕ := 3

/-- Vlad's sister's height is 2 feet and 10 inches. -/
def sister_height_feet : ℕ := 2

def sister_height_inches : ℕ := 10

/-- There are 12 inches in a foot. -/
def inches_in_a_foot : ℕ := 12

/-- Convert height in feet and inches to total inches. -/
def convert_to_inches (feet inches : ℕ) : ℕ :=
  feet * inches_in_a_foot + inches

/-- Proof that Vlad is 41 inches taller than his sister. -/
theorem vlad_taller_by_41_inches : convert_to_inches vlad_height_feet vlad_height_inches - convert_to_inches sister_height_feet sister_height_inches = 41 :=
by
  -- Start the proof
  sorry

end vlad_taller_by_41_inches_l166_166276


namespace inequality_solution_l166_166960

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) → a ≥ 2 :=
by
  sorry

end inequality_solution_l166_166960


namespace largest_and_smallest_A_l166_166770

noncomputable def is_coprime_with_12 (n : ℕ) : Prop := 
  Nat.gcd n 12 = 1

def problem_statement (A_max A_min : ℕ) : Prop :=
  ∃ B : ℕ, B > 44444444 ∧ is_coprime_with_12 B ∧
  (A_max = 9 * 10^7 + (B - 9) / 10) ∧
  (A_min = 1 * 10^7 + (B - 1) / 10)

theorem largest_and_smallest_A :
  problem_statement 99999998 14444446 := sorry

end largest_and_smallest_A_l166_166770


namespace multiplication_scaling_l166_166031

theorem multiplication_scaling (h : 28 * 15 = 420) : 
  (28 / 10) * (15 / 10) = 2.8 * 1.5 ∧ 
  (28 / 100) * 1.5 = 0.28 * 1.5 ∧ 
  (28 / 1000) * (15 / 100) = 0.028 * 0.15 :=
by 
  sorry

end multiplication_scaling_l166_166031


namespace min_fuse_length_l166_166904

theorem min_fuse_length 
  (safe_distance : ℝ := 70) 
  (personnel_speed : ℝ := 7) 
  (fuse_burning_speed : ℝ := 10.3) : 
  ∃ (x : ℝ), x ≥ 103 := 
by
  sorry

end min_fuse_length_l166_166904


namespace sum_of_coordinates_of_intersection_l166_166652

theorem sum_of_coordinates_of_intersection :
  let A := (0, 4)
  let B := (6, 0)
  let C := (9, 3)
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let line_AE := (fun x : ℚ => (-1/3) * x + 4)
  let line_CD := (fun x : ℚ => (1/6) * x + 1/2)
  let F_x := (21 : ℚ) / 3
  let F_y := line_AE F_x
  F_x + F_y = 26 / 3 := sorry

end sum_of_coordinates_of_intersection_l166_166652


namespace sheepdog_catches_sheep_in_20_seconds_l166_166074

noncomputable def speed_sheep : ℝ := 12 -- feet per second
noncomputable def speed_sheepdog : ℝ := 20 -- feet per second
noncomputable def initial_distance : ℝ := 160 -- feet

theorem sheepdog_catches_sheep_in_20_seconds :
  (initial_distance / (speed_sheepdog - speed_sheep)) = 20 :=
by
  sorry

end sheepdog_catches_sheep_in_20_seconds_l166_166074


namespace correct_masks_l166_166311

def elephant_mask := 6
def mouse_mask := 4
def pig_mask := 8
def panda_mask := 1

theorem correct_masks :
  (elephant_mask = 6) ∧
  (mouse_mask = 4) ∧
  (pig_mask = 8) ∧
  (panda_mask = 1) := 
by
  sorry

end correct_masks_l166_166311


namespace polynomial_equation_example_l166_166622

theorem polynomial_equation_example (a0 a1 a2 a3 a4 a5 a6 a7 a8 : ℤ)
  (h : x^5 * (x + 3)^3 = a8 * (x + 1)^8 + a7 * (x + 1)^7 + a6 * (x + 1)^6 + a5 * (x + 1)^5 + a4 * (x + 1)^4 + a3 * (x + 1)^3 + a2 * (x + 1)^2 + a1 * (x + 1) + a0) :
  7 * a7 + 5 * a5 + 3 * a3 + a1 = -8 :=
sorry

end polynomial_equation_example_l166_166622


namespace trig_identity_l166_166560

theorem trig_identity (α : ℝ) :
  4.10 * (Real.cos (45 * Real.pi / 180 - α)) ^ 2 
  - (Real.cos (60 * Real.pi / 180 + α)) ^ 2 
  - Real.cos (75 * Real.pi / 180) * Real.sin (75 * Real.pi / 180 - 2 * α) 
  = Real.sin (2 * α) := 
sorry

end trig_identity_l166_166560


namespace parking_methods_count_l166_166905

theorem parking_methods_count : 
  ∃ (n : ℕ), n = 72 ∧ (∃ (spaces cars slots remainingSlots : ℕ), 
  spaces = 7 ∧ cars = 3 ∧ slots = 1 ∧ remainingSlots = 4 ∧
  ∃ (perm_ways slot_ways : ℕ), perm_ways = 6 ∧ slot_ways = 12 ∧ n = perm_ways * slot_ways) :=
  by
    sorry

end parking_methods_count_l166_166905


namespace water_required_to_prepare_saline_solution_l166_166962

theorem water_required_to_prepare_saline_solution (water_ratio : ℝ) (required_volume : ℝ) : 
  water_ratio = 3 / 8 ∧ required_volume = 0.64 → required_volume * water_ratio = 0.24 :=
by
  sorry

end water_required_to_prepare_saline_solution_l166_166962


namespace four_nat_nums_prime_condition_l166_166318

theorem four_nat_nums_prime_condition (a b c d : ℕ) (h₁ : a = 1) (h₂ : b = 2) (h₃ : c = 3) (h₄ : d = 5) :
  Nat.Prime (a * b + c * d) ∧ Nat.Prime (a * c + b * d) ∧ Nat.Prime (a * d + b * c) :=
by
  sorry

end four_nat_nums_prime_condition_l166_166318


namespace unique_solution_arithmetic_progression_l166_166512

variable {R : Type*} [Field R]

theorem unique_solution_arithmetic_progression (a b c m x y z : R) :
  (m ≠ -2) ∧ (m ≠ 1) ∧ (a + c = 2 * b) → 
  (x + y + m * z = a) ∧ (x + m * y + z = b) ∧ (m * x + y + z = c) → 
  ∃ x y z, 2 * y = x + z :=
by
  sorry

end unique_solution_arithmetic_progression_l166_166512


namespace simplify_expression_l166_166679

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l166_166679


namespace arccos_neg_one_eq_pi_l166_166163

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
  sorry

end arccos_neg_one_eq_pi_l166_166163


namespace range_of_a_l166_166379

theorem range_of_a (a : ℝ) (h_pos : a > 0)
  (p : ∀ x : ℝ, x^2 - 4 * a * x + 3 * a^2 ≤ 0)
  (q : ∀ x : ℝ, (x^2 - x - 6 < 0) ∧ (x^2 + 2 * x - 8 > 0)) :
  (a ∈ ((Set.Ioo 0 (2 / 3)) ∪ (Set.Ici 3))) :=
by
  sorry

end range_of_a_l166_166379


namespace solve_for_A_l166_166341

def spadesuit (A B : ℝ) : ℝ := 4*A + 3*B + 6

theorem solve_for_A (A : ℝ) : spadesuit A 5 = 79 → A = 14.5 :=
by
  intros h
  sorry

end solve_for_A_l166_166341


namespace no_solution_to_system_l166_166987

theorem no_solution_to_system :
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 12 ∧ 9 * x - 12 * y = 15) :=
by
  sorry

end no_solution_to_system_l166_166987


namespace meaningful_fraction_l166_166420

theorem meaningful_fraction (x : ℝ) : (x - 1 ≠ 0) ↔ (x ≠ 1) :=
by sorry

end meaningful_fraction_l166_166420


namespace lcm_of_numbers_is_750_l166_166815

-- Define the two numbers x and y
variables (x y : ℕ)

-- Given conditions as hypotheses
def product_of_numbers := 18750
def hcf_of_numbers := 25

-- The proof problem statement
theorem lcm_of_numbers_is_750 (h_product : x * y = product_of_numbers) 
                              (h_hcf : Nat.gcd x y = hcf_of_numbers) : Nat.lcm x y = 750 :=
by
  sorry

end lcm_of_numbers_is_750_l166_166815


namespace min_log_geom_seq_l166_166479

theorem min_log_geom_seq (a : ℕ → ℝ) (h1 : ∀ (n : ℕ), a n > 0)
  (h2 : a 1 + a 3 = 5 / 16) (h3 : a 2 + a 4 = 5 / 8) :
  ∃ n, log 2 (a 1 * a 2 * a 3 * a 4 * ... * a n) = -10 :=
sorry

end min_log_geom_seq_l166_166479


namespace solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l166_166017

-- Definitions for the inequality ax^2 - 2ax + 2a - 3 < 0
def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * a - 3

-- Requirement (1): The solution set is ℝ
theorem solution_set_all_real (a : ℝ) (h : a ≤ 0) : 
  ∀ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (2): The solution set is ∅
theorem solution_set_empty (a : ℝ) (h : a ≥ 3) : 
  ¬∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (3): There is at least one real solution
theorem exists_at_least_one_solution (a : ℝ) (h : a < 3) : 
  ∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

end solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l166_166017


namespace sum_of_intercepts_l166_166573

theorem sum_of_intercepts (x₀ y₀ : ℕ) (hx₀ : 4 * x₀ ≡ 2 [MOD 25]) (hy₀ : 5 * y₀ ≡ 23 [MOD 25]) 
  (hx_cond : x₀ < 25) (hy_cond : y₀ < 25) : x₀ + y₀ = 28 :=
  sorry

end sum_of_intercepts_l166_166573


namespace daily_sales_volume_relationship_maximize_daily_sales_profit_l166_166719

variables (x : ℝ) (y : ℝ) (P : ℝ)

-- Conditions
def cost_per_box : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def decrease_in_sales_volume_per_dollar : ℝ := 20

-- The functional relationship between y and x
theorem daily_sales_volume_relationship (hx : min_selling_price ≤ x ∧ x < 80) : y = -20 * x + 1600 := by
  sorry

-- The profit function
def profit_function (x : ℝ) := (x - cost_per_box) * (initial_sales_volume - decrease_in_sales_volume_per_dollar * (x - initial_selling_price))

-- Maximizing the profit
theorem maximize_daily_sales_profit : ∃ x_max, x_max = 60 ∧ P = profit_function 60 ∧ P = 8000 := by
  sorry

end daily_sales_volume_relationship_maximize_daily_sales_profit_l166_166719


namespace train_crossing_pole_time_l166_166219

theorem train_crossing_pole_time :
  ∀ (length_of_train : ℝ) (speed_km_per_hr : ℝ) (t : ℝ),
    length_of_train = 45 →
    speed_km_per_hr = 108 →
    t = 1.5 →
    t = length_of_train / (speed_km_per_hr * 1000 / 3600) := 
  sorry

end train_crossing_pole_time_l166_166219


namespace max_mineral_value_l166_166280

/-- Jane discovers three types of minerals with given weights and values:
6-pound mineral chunks worth $16 each,
3-pound mineral chunks worth $9 each,
and 2-pound mineral chunks worth $3 each. 
There are at least 30 of each type available.
She can haul a maximum of 21 pounds in her cart.
Prove that the maximum value, in dollars, that Jane can transport is $63. -/
theorem max_mineral_value : 
  ∃ (value : ℕ), (∀ (x y z : ℕ), 6 * x + 3 * y + 2 * z ≤ 21 → 
    (x ≤ 30 ∧ y ≤ 30 ∧ z ≤ 30) → value ≥ 16 * x + 9 * y + 3 * z) ∧ value = 63 :=
by sorry

end max_mineral_value_l166_166280


namespace max_distance_sum_l166_166025

theorem max_distance_sum {P : ℝ × ℝ} 
  (C : Set (ℝ × ℝ)) 
  (hC : ∀ (P : ℝ × ℝ), P ∈ C ↔ (P.1 - 3)^2 + (P.2 - 4)^2 = 1)
  (A : ℝ × ℝ := (0, -1))
  (B : ℝ × ℝ := (0, 1)) :
  ∃ P : ℝ × ℝ, 
    P ∈ C ∧ (P = (18 / 5, 24 / 5)) :=
by
  sorry

end max_distance_sum_l166_166025


namespace find_length_of_train_l166_166893

noncomputable def speed_kmhr : ℝ := 30
noncomputable def time_seconds : ℝ := 9
noncomputable def conversion_factor : ℝ := 5 / 18
noncomputable def speed_ms : ℝ := speed_kmhr * conversion_factor
noncomputable def length_train : ℝ := speed_ms * time_seconds

theorem find_length_of_train : length_train = 74.97 := 
by
  sorry

end find_length_of_train_l166_166893


namespace find_multiple_l166_166316

-- Definitions of the conditions
def is_positive (x : ℝ) : Prop := x > 0

-- Main statement
theorem find_multiple (x : ℝ) (h : is_positive x) (hx : x = 8) : ∃ k : ℝ, x + 8 = k * (1 / x) ∧ k = 128 :=
by
  use 128
  sorry

end find_multiple_l166_166316


namespace find_a8_l166_166370

theorem find_a8 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n : ℕ, (1 / (a n + 1)) = (1 / (a 0 + 1)) + n * ((1 / (a 1 + 1 - 1)) / 3)) 
  (h2 : a 2 = 3) 
  (h5 : a 5 = 1) : 
  a 8 = 1 / 3 :=
by
  sorry

end find_a8_l166_166370


namespace work_problem_l166_166281

theorem work_problem (A B : ℝ) (hA : A = 1/4) (hB : B = 1/12) :
  (2 * (A + B) + 4 * B = 1) :=
by
  -- Work rate of A and B together
  -- Work done in 2 days by both
  -- Remaining work and time taken by B alone
  -- Final Result
  sorry

end work_problem_l166_166281


namespace ordered_triples_count_l166_166968

namespace LeanVerify

def S : Finset ℕ := {n | 1 ≤ n ∧ n ≤ 15}

def succ (a b : ℕ) : Prop := (0 < a - b ∧ a - b ≤ 7) ∨ (b - a > 7)

theorem ordered_triples_count : 
  (Finset.filter (λ (t : ℕ × ℕ × ℕ), succ t.1 t.2 ∧ succ t.2 t.3 ∧ succ t.3 t.1) 
    (S.product (S.product S))).card = 420 :=
by {
  sorry
}

end LeanVerify

end ordered_triples_count_l166_166968


namespace abs_neg_three_eq_three_l166_166395

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
sorry

end abs_neg_three_eq_three_l166_166395


namespace remainder_3_pow_100_plus_5_mod_8_l166_166555

theorem remainder_3_pow_100_plus_5_mod_8 : (3^100 + 5) % 8 = 6 := by
  sorry

end remainder_3_pow_100_plus_5_mod_8_l166_166555


namespace pet_store_animals_left_l166_166294

theorem pet_store_animals_left (initial_birds initial_puppies initial_cats initial_spiders initial_snakes : ℕ)
  (donation_fraction snakes_share_sold birds_sold puppies_adopted cats_transferred kittens_brought : ℕ)
  (spiders_loose spiders_captured : ℕ)
  (H_initial_birds : initial_birds = 12)
  (H_initial_puppies : initial_puppies = 9)
  (H_initial_cats : initial_cats = 5)
  (H_initial_spiders : initial_spiders = 15)
  (H_initial_snakes : initial_snakes = 8)
  (H_donation_fraction : donation_fraction = 25)
  (H_snakes_share_sold : snakes_share_sold = (donation_fraction * initial_snakes) / 100)
  (H_birds_sold : birds_sold = initial_birds / 2)
  (H_puppies_adopted : puppies_adopted = 3)
  (H_cats_transferred : cats_transferred = 4)
  (H_kittens_brought : kittens_brought = 2)
  (H_spiders_loose : spiders_loose = 7)
  (H_spiders_captured : spiders_captured = 5) :
  (initial_snakes - snakes_share_sold) + (initial_birds - birds_sold) + 
  (initial_puppies - puppies_adopted) + (initial_cats - cats_transferred + kittens_brought) + 
  (initial_spiders - (spiders_loose - spiders_captured)) = 34 := 
by 
  sorry

end pet_store_animals_left_l166_166294


namespace apples_picked_per_tree_l166_166528

-- Definitions
def num_trees : Nat := 4
def total_apples_picked : Nat := 28

-- Proving how many apples Rachel picked from each tree if the same number were picked from each tree
theorem apples_picked_per_tree (h : num_trees ≠ 0) :
  total_apples_picked / num_trees = 7 :=
by
  sorry

end apples_picked_per_tree_l166_166528


namespace book_distribution_ways_l166_166586

/-- 
Problem Statement: We have 8 identical books. We want to find out the number of ways to distribute these books between the library and checked out such that at least one book is in the library and at least one book is checked out. The expected answer is 7.
-/
theorem book_distribution_ways : ∃ n : ℕ, n = 7 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 ↔ k books in library means exactly 8 - k books are checked out) :=
by
  sorry

end book_distribution_ways_l166_166586


namespace flower_bed_l166_166284

def planting_schemes (A B C D E F : Prop) : Prop :=
  A ≠ B ∧ B ≠ C ∧ D ≠ E ∧ E ≠ F ∧ A ≠ D ∧ B ≠ D ∧ B ≠ E ∧ C ≠ E ∧ C ≠ F ∧ D ≠ F

theorem flower_bed (A B C D E F : Prop) (plant_choices : Finset (Fin 6))
  (h_choice : plant_choices.card = 6)
  (h_different : ∀ x ∈ plant_choices, ∀ y ∈ plant_choices, x ≠ y → x ≠ y)
  (h_adj : planting_schemes A B C D E F) :
  ∃! planting_schemes, planting_schemes ∧ plant_choices.card = 13230 :=
by sorry

end flower_bed_l166_166284


namespace not_tangent_for_any_k_k_range_l166_166491

noncomputable def f (x : ℝ) : ℝ := x / Real.log x
noncomputable def g (x k : ℝ) : ℝ := k * (x - 1)

theorem not_tangent_for_any_k (k : ℝ) : ¬∃ m ∈ (Set.Ioi 1), k = (Real.log m - 1) / (Real.log m)^2 :=
sorry

theorem k_range (k : ℝ) : (∃ x ∈ Set.Icc Real.exp (Real.exp 2), f x ≤ g x k + 1 / 2) → k ≥ 1 / 2 :=
sorry

end not_tangent_for_any_k_k_range_l166_166491


namespace solution_correct_l166_166865

noncomputable def solve_system (A1 A2 A3 A4 A5 : ℝ) (x1 x2 x3 x4 x5 : ℝ) :=
  (2 * x1 - 2 * x2 = A1) ∧
  (-x1 + 4 * x2 - 3 * x3 = A2) ∧
  (-2 * x2 + 6 * x3 - 4 * x4 = A3) ∧
  (-3 * x3 + 8 * x4 - 5 * x5 = A4) ∧
  (-4 * x4 + 10 * x5 = A5)

theorem solution_correct {A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 : ℝ} :
  solve_system A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 → 
  x1 = (5 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x2 = (2 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x3 = (A1 + 2 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x4 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 2 * A5) / 12 ∧
  x5 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 5 * A5) / 30 :=
sorry

end solution_correct_l166_166865


namespace range_of_a_l166_166027

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) := ∀ x₁ x₂ : ℝ, x₁ < x₂ → -(5 - 2 * a)^x₁ > -(5 - 2 * a)^x₂

theorem range_of_a (a : ℝ) : (p a ∨ q a) → ¬ (p a ∧ q a) → a ≤ -2 := by 
  sorry

end range_of_a_l166_166027


namespace prime_divisors_of_1890_l166_166339

theorem prime_divisors_of_1890 : ∃ (S : Finset ℕ), (S.card = 4) ∧ (∀ p ∈ S, Nat.Prime p) ∧ 1890 = S.prod id :=
by
  sorry

end prime_divisors_of_1890_l166_166339


namespace a_equals_b_l166_166837

theorem a_equals_b (a b : ℕ) (h : a^3 + a + 4 * b^2 = 4 * a * b + b + b * a^2) : a = b := 
sorry

end a_equals_b_l166_166837


namespace complex_quadrant_l166_166539

open Complex

theorem complex_quadrant (z : ℂ) (h : (1 + I) * z = 2 * I) : 
  z.re > 0 ∧ z.im < 0 :=
  sorry

end complex_quadrant_l166_166539


namespace square_form_l166_166814

theorem square_form (m n : ℤ) : 
  ∃ k l : ℤ, (2 * m^2 + n^2)^2 = 2 * k^2 + l^2 :=
by
  let x := (2 * m^2 + n^2)
  let y := x^2
  let k := 2 * m * n
  let l := 2 * m^2 - n^2
  use k, l
  sorry

end square_form_l166_166814


namespace sin_theta_tan_theta_iff_first_third_quadrant_l166_166109

open Real

-- Definitions from conditions
def in_first_or_third_quadrant (θ : ℝ) : Prop :=
  (0 < θ ∧ θ < π / 2) ∨ (π < θ ∧ θ < 3 * π / 2)

def sin_theta_plus_tan_theta_positive (θ : ℝ) : Prop :=
  sin θ + tan θ > 0

-- Proof statement
theorem sin_theta_tan_theta_iff_first_third_quadrant (θ : ℝ) :
  sin_theta_plus_tan_theta_positive θ ↔ in_first_or_third_quadrant θ :=
sorry

end sin_theta_tan_theta_iff_first_third_quadrant_l166_166109


namespace three_alpha_four_plus_eight_beta_three_eq_876_l166_166198

variable (α β : ℝ)

-- Condition 1: α and β are roots of the equation x^2 - 3x - 4 = 0
def roots_of_quadratic : Prop := α^2 - 3 * α - 4 = 0 ∧ β^2 - 3 * β - 4 = 0

-- Question: 3α^4 + 8β^3 = ?
theorem three_alpha_four_plus_eight_beta_three_eq_876 
  (h : roots_of_quadratic α β) : (3 * α^4 + 8 * β^3 = 876) := sorry

end three_alpha_four_plus_eight_beta_three_eq_876_l166_166198


namespace chef_completion_time_l166_166287

variables (start_time halfway_time total_preparation_time completion_time : Time)

def preparation_start_time := start_time = Time.mk 9 0
def halfway_prepared_time := halfway_time = Time.mk 12 30
def on_schedule := halfway_prepared_time → halfway_time = halfway_time

theorem chef_completion_time (h1 : preparation_start_time) (h2 : halfway_prepared_time) (h3 : on_schedule) :
  completion_time = Time.mk 16 0 := sorry

end chef_completion_time_l166_166287


namespace function_properties_graph_transformation_l166_166630

noncomputable def f (x : ℝ) (m : ℝ) := m * sin x + cos x

theorem function_properties :
  (∃ m, f (π / 2) m = 1) →
  f x 1 = sqrt 2 * sin (x + π / 4) ∧
  Function.Periodic (f x 1) (2 * π) ∧
  ∃ x, f x 1 = sqrt 2 :=
by
  sorry

/-
Describe the transformation required to obtain the graph of f(2x) from the graph of f(x - π/4).
Proof not required.
-/
theorem graph_transformation :
  (∀ x, f (2 * x) 1 = sqrt 2 * sin (2 * x + π / 4)) →
  (∀ x, f (x - π / 4) 1 = sqrt 2 * sin (x - π / 4 + π / 4)) →
  True :=
by
  sorry

end function_properties_graph_transformation_l166_166630


namespace sine_triangle_inequality_l166_166072

theorem sine_triangle_inequality 
  {a b c : ℝ} (h_triangle : a + b + c ≤ 2 * Real.pi) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (ha_lt_pi : a < Real.pi) (hb_lt_pi : b < Real.pi) (hc_lt_pi : c < Real.pi) :
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end sine_triangle_inequality_l166_166072


namespace area_of_rectangle_l166_166110

-- Define the conditions
def width : ℕ := 6
def perimeter : ℕ := 28

-- Define the theorem statement
theorem area_of_rectangle (w : ℕ) (p : ℕ) (h_width : w = width) (h_perimeter : p = perimeter) :
  ∃ l : ℕ, (2 * (l + w) = p) → (l * w = 48) :=
by
  use 8
  intro h
  simp only [h_width, h_perimeter] at h
  sorry

end area_of_rectangle_l166_166110


namespace C_increases_with_n_l166_166957

noncomputable def C (e n R r : ℝ) : ℝ := (e * n) / (R + n * r)

theorem C_increases_with_n (e R r : ℝ) (h_e : 0 < e) (h_R : 0 < R) (h_r : 0 < r) :
  ∀ {n₁ n₂ : ℝ}, 0 < n₁ → n₁ < n₂ → C e n₁ R r < C e n₂ R r :=
by
  sorry

end C_increases_with_n_l166_166957


namespace total_seashells_l166_166525

theorem total_seashells :
  let initial_seashells : ℝ := 6.5
  let more_seashells : ℝ := 4.25
  initial_seashells + more_seashells = 10.75 :=
by
  sorry

end total_seashells_l166_166525


namespace reciprocal_of_neg3_l166_166701

theorem reciprocal_of_neg3 : ∃ x : ℝ, (-3 : ℝ) * x = 1 :=  
begin
  use -1/3,
  norm_num,
end

end reciprocal_of_neg3_l166_166701


namespace problem1_problem2_l166_166492

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem problem1 :
  f 1 + f 2 + f 3 + f (1 / 2) + f (1 / 3) = 5 / 2 :=
by
  sorry

theorem problem2 : ∀ x : ℝ, 0 < f x ∧ f x ≤ 1 :=
by
  intro x
  sorry

end problem1_problem2_l166_166492


namespace percent_increase_correct_l166_166915

variable (p_initial p_final : ℝ)

theorem percent_increase_correct : p_initial = 25 → p_final = 28 → (p_final - p_initial) / p_initial * 100 = 12 := by
  intros h_initial h_final
  sorry

end percent_increase_correct_l166_166915


namespace positive_rational_achievable_l166_166757

open Set Finset

noncomputable theory

variable {A : Set ℕ} (h1 : ∀ n ∈ A, 2 * n ∈ A)
                             (h2 : ∀ (n : ℕ), ∃ m ∈ A, m % n = 0)
                             (h3 : ∀ (C : ℝ), C > 0 → ∃ B : Finset ℕ, (B ⊆ A.toFinset ∧ ∑ x in B, (1 / (x : ℝ)) > C))

theorem positive_rational_achievable :
  ∀ r : ℚ, r > 0 → ∃ B : Finset ℕ, (B ⊆ A.toFinset ∧ (∑ x in B, (1 / (x : ℝ)) = (r : ℝ))) :=
by
  sorry

end positive_rational_achievable_l166_166757


namespace slope_CD_l166_166400

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 40 = 0

theorem slope_CD :
  ∀ C D : ℝ × ℝ, circle1 C.1 C.2 → circle2 D.1 D.2 → 
  (C ≠ D → (D.2 - C.2) / (D.1 - C.1) = 5 / 2) := 
by
  -- proof to be completed
  sorry

end slope_CD_l166_166400


namespace sum_of_three_primes_eq_86_l166_166552

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_three_primes_eq_86 (a b c : ℕ) (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) (h_sum : a + b + c = 86) :
  (a, b, c) = (2, 5, 79) ∨ (a, b, c) = (2, 11, 73) ∨ (a, b, c) = (2, 13, 71) ∨ (a, b, c) = (2, 17, 67) ∨
  (a, b, c) = (2, 23, 61) ∨ (a, b, c) = (2, 31, 53) ∨ (a, b, c) = (2, 37, 47) ∨ (a, b, c) = (2, 41, 43) :=
by
  sorry

end sum_of_three_primes_eq_86_l166_166552


namespace largest_possible_a_l166_166969

theorem largest_possible_a (a b c d : ℕ) (ha : a < 2 * b) (hb : b < 3 * c) (hc : c < 4 * d) (hd : d < 100) : 
  a ≤ 2367 :=
sorry

end largest_possible_a_l166_166969


namespace black_pens_removed_l166_166130

theorem black_pens_removed (initial_blue : ℕ) (initial_black : ℕ) (initial_red : ℕ)
    (blue_removed : ℕ) (pens_left : ℕ)
    (h_initial_pens : initial_blue = 9 ∧ initial_black = 21 ∧ initial_red = 6)
    (h_blue_removed : blue_removed = 4)
    (h_pens_left : pens_left = 25) :
    initial_blue + initial_black + initial_red - blue_removed - (initial_blue + initial_black + initial_red - blue_removed - pens_left) = 7 :=
by
  rcases h_initial_pens with ⟨h_ib, h_ibl, h_ir⟩
  simp [h_ib, h_ibl, h_ir, h_blue_removed, h_pens_left]
  sorry

end black_pens_removed_l166_166130


namespace derivative_at_one_l166_166489

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + x^2) : f' 1 = -2 :=
by
  sorry

end derivative_at_one_l166_166489


namespace transportation_trucks_l166_166984

theorem transportation_trucks (boxes : ℕ) (total_weight : ℕ) (box_weight : ℕ) (truck_capacity : ℕ) :
  (total_weight = 10) → (∀ (b : ℕ), b ≤ boxes → box_weight ≤ 1) → (truck_capacity = 3) → 
  ∃ (trucks : ℕ), trucks = 5 :=
by
  sorry

end transportation_trucks_l166_166984


namespace rate_percent_simple_interest_l166_166102

theorem rate_percent_simple_interest
  (SI P : ℚ) (T : ℕ) (R : ℚ) : SI = 160 → P = 800 → T = 4 → (P * R * T / 100 = SI) → R = 5 :=
  by
  intros hSI hP hT hFormula
  -- Assertion that R = 5 is correct based on the given conditions and formula
  sorry

end rate_percent_simple_interest_l166_166102


namespace find_two_digit_numbers_l166_166716

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem find_two_digit_numbers :
  ∀ (A : ℕ), (10 ≤ A ∧ A ≤ 99) →
    (sum_of_digits A)^2 = sum_of_digits (A^2) →
    (A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31 ∨ A = 50) :=
by sorry

end find_two_digit_numbers_l166_166716


namespace range_of_a_l166_166057

theorem range_of_a
  (a : ℝ)
  (h : ∀ x : ℝ, |x + 1| + |x - 3| ≥ a) : a ≤ 4 :=
sorry

end range_of_a_l166_166057


namespace length_of_base_of_vessel_l166_166578

noncomputable def volume_of_cube (edge : ℝ) := edge ^ 3

theorem length_of_base_of_vessel 
  (cube_edge : ℝ)
  (vessel_width : ℝ)
  (rise_in_water_level : ℝ)
  (volume_cube : ℝ)
  (h1 : cube_edge = 15)
  (h2 : vessel_width = 15)
  (h3 : rise_in_water_level = 11.25)
  (h4 : volume_cube = volume_of_cube cube_edge)
  : ∃ L : ℝ, L = volume_cube / (vessel_width * rise_in_water_level) ∧ L = 20 :=
by
  sorry

end length_of_base_of_vessel_l166_166578


namespace difference_sum_first_100_odds_evens_l166_166993

def sum_first_n_odds (n : ℕ) : ℕ :=
  n^2

def sum_first_n_evens (n : ℕ) : ℕ :=
  n * (n-1)

theorem difference_sum_first_100_odds_evens :
  sum_first_n_odds 100 - sum_first_n_evens 100 = 100 := by
  sorry

end difference_sum_first_100_odds_evens_l166_166993


namespace average_of_middle_three_l166_166088

-- Define the conditions based on the problem statement
def isPositiveWhole (n: ℕ) := n > 0
def areDifferent (a b c d e: ℕ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def isMaximumDifference (a b c d e: ℕ) := max a (max b (max c (max d e))) - min a (min b (min c (min d e)))
def isSecondSmallest (a b c d e: ℕ) := b = 3 ∧ (a < b ∧ (c < b ∨ d < b ∨ e < b) ∧ areDifferent a b c d e)
def totalSumIs30 (a b c d e: ℕ) := a + b + c + d + e = 30

-- Average of the middle three numbers calculated
theorem average_of_middle_three {a b c d e: ℕ} (cond1: isPositiveWhole a)
  (cond2: isPositiveWhole b) (cond3: isPositiveWhole c) (cond4: isPositiveWhole d)
  (cond5: isPositiveWhole e) (cond6: areDifferent a b c d e) (cond7: b = 3)
  (cond8: max a (max c (max d e)) - min a (min c (min d e)) = 16)
  (cond9: totalSumIs30 a b c d e) : (a + c + d) / 3 = 4 :=
by sorry

end average_of_middle_three_l166_166088


namespace determine_k_values_l166_166608

theorem determine_k_values (k : ℝ) :
  (∃ a b : ℝ, 3 * a ^ 2 + 6 * a + k = 0 ∧ 3 * b ^ 2 + 6 * b + k = 0 ∧ |a - b| = 1 / 2 * (a ^ 2 + b ^ 2)) → (k = 0 ∨ k = 12) :=
by
  sorry

end determine_k_values_l166_166608


namespace average_of_consecutive_numbers_l166_166991

-- Define the 7 consecutive numbers and their properties
variables (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ) (f : ℝ) (g : ℝ)

-- Conditions given in the problem
def consecutive_numbers (a b c d e f g : ℝ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 ∧ g = a + 6

def percent_relationship (a g : ℝ) : Prop :=
  g = 1.5 * a

-- The proof problem
theorem average_of_consecutive_numbers (a b c d e f g : ℝ)
  (h1 : consecutive_numbers a b c d e f g)
  (h2 : percent_relationship a g) :
  (a + b + c + d + e + f + g) / 7 = 15 :=
by {
  sorry -- Proof goes here
}

-- To ensure it passes the type checker but without providing the actual proof, we use sorry.

end average_of_consecutive_numbers_l166_166991


namespace five_letter_words_with_at_least_two_vowels_l166_166337

theorem five_letter_words_with_at_least_two_vowels 
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'}) :
  (letters.card = 6) ∧ (vowels.card = 2) ∧ (letters ⊆ {'A', 'B', 'C', 'D', 'E', 'F'}) →
  (∃ count : ℕ, count = 4192) :=
sorry

end five_letter_words_with_at_least_two_vowels_l166_166337


namespace no_int_representation_l166_166080

theorem no_int_representation (A B : ℤ) : (99999 + 111111 * Real.sqrt 3) ≠ (A + B * Real.sqrt 3)^2 :=
by
  sorry

end no_int_representation_l166_166080


namespace factorize_a3_minus_4a_l166_166315

theorem factorize_a3_minus_4a (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) := 
by
  sorry

end factorize_a3_minus_4a_l166_166315


namespace no_nat_exists_perfect_cubes_l166_166005

theorem no_nat_exists_perfect_cubes : ¬ ∃ n : ℕ, ∃ a b : ℤ, 2^(n + 1) - 1 = a^3 ∧ 2^(n - 1)*(2^n - 1) = b^3 := 
by
  sorry

end no_nat_exists_perfect_cubes_l166_166005


namespace sufficient_but_not_necessary_condition_l166_166345

variable {a : Type} {M : Type} (line : a → Prop) (plane : M → Prop)

-- Assume the definitions of perpendicularity
def perp_to_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to plane
def perp_to_lines_in_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to countless lines

-- Mathematical statement
theorem sufficient_but_not_necessary_condition (a : a) (M : M) :
  (perp_to_plane a M → perp_to_lines_in_plane a M) ∧ ¬(perp_to_lines_in_plane a M → perp_to_plane a M) :=
by
  sorry

end sufficient_but_not_necessary_condition_l166_166345


namespace rational_function_solution_l166_166248

theorem rational_function_solution (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g (1 / x) + 3 * g x / x = x^3) :
  g (-3) = 135 / 4 := 
sorry

end rational_function_solution_l166_166248


namespace find_B_l166_166390

noncomputable def g (A B C D x : ℝ) : ℝ :=
  A * x^3 + B * x^2 + C * x + D

theorem find_B (A C D : ℝ) (h1 : ∀ x, g A (-2) C D x = A * (x + 2) * (x - 1) * (x - 2)) 
  (h2 : g A (-2) C D 0 = -8) : 
  (-2 : ℝ) = -2 := 
by
  simp [g] at h2
  sorry

end find_B_l166_166390


namespace lara_puts_flowers_in_vase_l166_166220

theorem lara_puts_flowers_in_vase : 
  ∀ (total_flowers mom_flowers flowers_given_more : ℕ), 
    total_flowers = 52 →
    mom_flowers = 15 →
    flowers_given_more = 6 →
  (total_flowers - (mom_flowers + (mom_flowers + flowers_given_more))) = 16 :=
by
  intros total_flowers mom_flowers flowers_given_more h1 h2 h3
  sorry

end lara_puts_flowers_in_vase_l166_166220


namespace problem_a_l166_166746

theorem problem_a : (1038^2 % 1000) ≠ 4 := by
  sorry

end problem_a_l166_166746


namespace percent_water_evaporated_l166_166576

theorem percent_water_evaporated (W : ℝ) (E : ℝ) (T : ℝ) (hW : W = 10) (hE : E = 0.16) (hT : T = 75) : 
  ((min (E * T) W) / W) * 100 = 100 :=
by
  sorry

end percent_water_evaporated_l166_166576


namespace min_dancers_l166_166290

theorem min_dancers (N : ℕ) (h1 : N % 4 = 0) (h2 : N % 9 = 0) (h3 : N % 10 = 0) (h4 : N > 50) : N = 180 :=
  sorry

end min_dancers_l166_166290


namespace greatest_three_digit_number_l166_166277

theorem greatest_three_digit_number :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ N % 8 = 2 ∧ N % 7 = 4 ∧ N = 978 :=
by
  sorry

end greatest_three_digit_number_l166_166277


namespace find_speeds_l166_166354

theorem find_speeds 
  (x v u : ℝ)
  (hx : x = u / 4)
  (hv : 0 < v)
  (hu : 0 < u)
  (t_car : 30 / v + 1.25 = 30 / x)
  (meeting_cars : 0.05 * v + 0.05 * u = 5) :
  x = 15 ∧ v = 40 ∧ u = 60 :=
by 
  sorry

end find_speeds_l166_166354


namespace find_larger_number_l166_166090

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1500) (h2 : y = 6 * x + 15) : y = 1797 := by
  sorry

end find_larger_number_l166_166090


namespace system_of_equations_l166_166190

theorem system_of_equations (x y : ℝ) 
  (h1 : 2019 * x + 2020 * y = 2018) 
  (h2 : 2020 * x + 2019 * y = 2021) :
  x + y = 1 ∧ x - y = 3 :=
by sorry

end system_of_equations_l166_166190


namespace Gunther_free_time_left_l166_166804

def vacuuming_time := 45
def dusting_time := 60
def folding_laundry_time := 25
def mopping_time := 30
def cleaning_bathroom_time := 40
def wiping_windows_time := 15
def brushing_cats_time := 4 * 5
def washing_dishes_time := 20
def first_tasks_total_time := 2 * 60 + 30
def available_free_time := 5 * 60

theorem Gunther_free_time_left : 
  (available_free_time - 
   (vacuuming_time + dusting_time + folding_laundry_time + 
    mopping_time + cleaning_bathroom_time + 
    wiping_windows_time + brushing_cats_time + 
    washing_dishes_time) = 45) := 
by 
  sorry

end Gunther_free_time_left_l166_166804


namespace HCF_of_numbers_l166_166872

theorem HCF_of_numbers (a b : ℕ) (h₁ : a * b = 84942) (h₂ : Nat.lcm a b = 2574) : Nat.gcd a b = 33 :=
by
  sorry

end HCF_of_numbers_l166_166872


namespace hare_total_distance_l166_166853

-- Define the conditions
def distance_between_trees : ℕ := 5
def number_of_trees : ℕ := 10

-- Define the question to be proved
theorem hare_total_distance : distance_between_trees * (number_of_trees - 1) = 45 :=
by
  sorry

end hare_total_distance_l166_166853


namespace kyle_vs_parker_l166_166598

-- Define the distances thrown by Parker, Grant, and Kyle.
def parker_distance : ℕ := 16
def grant_distance : ℕ := (125 * parker_distance) / 100
def kyle_distance : ℕ := 2 * grant_distance

-- Prove that Kyle threw the ball 24 yards farther than Parker.
theorem kyle_vs_parker : kyle_distance - parker_distance = 24 := 
by
  -- Sorry for proof
  sorry

end kyle_vs_parker_l166_166598


namespace vojta_correct_sum_l166_166101

theorem vojta_correct_sum (S A B C : ℕ)
  (h1 : S + (10 * B + C) = 2224)
  (h2 : S + (10 * A + B) = 2198)
  (h3 : S + (10 * A + C) = 2204)
  (A_digit : 0 ≤ A ∧ A < 10)
  (B_digit : 0 ≤ B ∧ B < 10)
  (C_digit : 0 ≤ C ∧ C < 10) :
  S + 100 * A + 10 * B + C = 2324 := 
sorry

end vojta_correct_sum_l166_166101


namespace find_a_minus_b_l166_166866

theorem find_a_minus_b (a b : ℚ) (h_eq : ∀ x : ℚ, (a * (-5 * x + 3) + b) = x - 9) : 
  a - b = 41 / 5 := 
by {
  sorry
}

end find_a_minus_b_l166_166866


namespace circle_symmetric_line_l166_166490

theorem circle_symmetric_line (a b : ℝ) 
  (h1 : ∃ x y, x^2 + y^2 - 4 * x + 2 * y + 1 = 0)
  (h2 : ∀ x y, (x, y) = (2, -1))
  (h3 : 2 * a + 2 * b - 1 = 0) :
  ab ≤ 1 / 16 := sorry

end circle_symmetric_line_l166_166490


namespace matt_house_wall_height_l166_166075

noncomputable def height_of_walls_in_matt_house : ℕ :=
  let living_room_side := 40
  let bedroom_side_1 := 10
  let bedroom_side_2 := 12

  let perimeter_living_room := 4 * living_room_side
  let perimeter_living_room_3_walls := perimeter_living_room - living_room_side

  let perimeter_bedroom := 2 * (bedroom_side_1 + bedroom_side_2)

  let total_perimeter_to_paint := perimeter_living_room_3_walls + perimeter_bedroom
  let total_area_to_paint := 1640

  total_area_to_paint / total_perimeter_to_paint

theorem matt_house_wall_height :
  height_of_walls_in_matt_house = 10 := by
  sorry

end matt_house_wall_height_l166_166075


namespace problem_solution_l166_166118

noncomputable def problem_expr : ℝ :=
  (64 + 5 * 12) / (180 / 3) + Real.sqrt 49 - 2^3 * Nat.factorial 4

theorem problem_solution : problem_expr = -182.93333333 :=
by 
  sorry

end problem_solution_l166_166118


namespace circle_center_coordinates_l166_166249

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (c = (1, -2)) ∧ 
  (∀ x y : ℝ, (x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - 1)^2 + (y + 2)^2 = 9)) :=
by
  sorry

end circle_center_coordinates_l166_166249


namespace athletes_same_color_probability_l166_166460

theorem athletes_same_color_probability :
  let colors := ["red", "white", "blue"]
  let total_ways := 3 * 3
  let same_color_ways := 3
  total_ways > 0 → 
  (same_color_ways : ℚ) / (total_ways : ℚ) = 1 / 3 :=
by
  sorry

end athletes_same_color_probability_l166_166460


namespace possible_values_of_Q_l166_166976

theorem possible_values_of_Q (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ∃ Q : ℝ, Q = 8 ∨ Q = -1 := 
sorry

end possible_values_of_Q_l166_166976


namespace inequality_holds_l166_166994

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_holds (h_cont : Continuous f) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, 2 * f x - (deriv f x) > 0) : 
  f 1 > (f 2) / (Real.exp 2) :=
sorry

end inequality_holds_l166_166994


namespace total_time_to_complete_work_l166_166382

-- Definitions based on conditions
variable (W : ℝ) -- W is the total work
variable (Mahesh_days : ℝ := 35) -- Mahesh can complete the work in 35 days
variable (Mahesh_working_days : ℝ := 20) -- Mahesh works for 20 days
variable (Rajesh_days : ℝ := 30) -- Rajesh finishes the remaining work in 30 days

-- Proof statement
theorem total_time_to_complete_work : Mahesh_working_days + Rajesh_days = 50 :=
by
  sorry

end total_time_to_complete_work_l166_166382


namespace simplify_fraction_l166_166687

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l166_166687


namespace points_earned_l166_166648

-- Definitions from conditions
def points_per_enemy : ℕ := 8
def total_enemies : ℕ := 7
def enemies_not_destroyed : ℕ := 2

-- The proof statement
theorem points_earned :
  points_per_enemy * (total_enemies - enemies_not_destroyed) = 40 := 
by
  sorry

end points_earned_l166_166648


namespace simplify_expression_l166_166863

variable (b c : ℝ)

theorem simplify_expression :
  3 * b * (3 * b ^ 3 + 2 * b) - 2 * b ^ 2 + c * (3 * b ^ 2 - c) = 9 * b ^ 4 + 4 * b ^ 2 + 3 * b ^ 2 * c - c ^ 2 :=
by
  sorry

end simplify_expression_l166_166863


namespace five_letter_words_with_at_least_two_vowels_five_letter_words_with_at_least_two_vowels_l166_166336

def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E'
def is_consonant (c : Char) : Prop := c ≠ 'A' ∧ c ≠ 'E'

def valid_letters := ['A', 'B', 'C', 'D', 'E', 'F']

noncomputable def count_words_with_min_vowels (n m : Nat) (letters : List Char) (min_vowels : Nat) : Nat :=
  let total_combinations := letters.length ^ n
  let less_than_min_vowels_total := 
    (∑ k in Finset.range n, if k < min_vowels then (binomial n k) * (2 : Nat) ^ k * (letters.length - 2) ^ (n - k) else 0)
  total_combinations - less_than_min_vowels_total

theorem five_letter_words_with_at_least_two_vowels :
  count_words_with_min_vowels 5 6 valid_letters 2 = 4192 :=
by
  -- Summary statement importing required libraries and setting up definitions

  def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E'
  def is_consonant (c : Char) : Prop := c ≠ 'A' ∧ c ≠ 'E'

  def valid_letters := ['A', 'B', 'C', 'D', 'E', 'F']

  noncomputable def count_words_with_min_vowels (n m : Nat) (letters : List Char) (min_vowels : Nat) : Nat :=
    let total_combinations := letters.length ^ n
    let less_than_min_vowels_total :=
      (∑ k in Finset.range n, if k < min_vowels then (binomial n k) * (2 : Nat) ^ k * (letters.length - 2) ^ (n - k) else 0)
    total_combinations - less_than_min_vowels_total

  theorem five_letter_words_with_at_least_two_vowels :
    count_words_with_min_vowels 5 6 valid_letters 2 = 4192 :=
  by
    sorry

end five_letter_words_with_at_least_two_vowels_five_letter_words_with_at_least_two_vowels_l166_166336


namespace range_is_fixed_points_l166_166623

variable (f : ℕ → ℕ)

axiom functional_eq : ∀ m n, f (m + f n) = f (f m) + f n

theorem range_is_fixed_points :
  {n : ℕ | ∃ m : ℕ, f m = n} = {n : ℕ | f n = n} :=
sorry

end range_is_fixed_points_l166_166623


namespace cows_count_24_l166_166282

-- Declare the conditions as given in the problem.
variables (D C : Nat)

-- Define the total number of legs and heads and the given condition.
def total_legs := 2 * D + 4 * C
def total_heads := D + C
axiom condition : total_legs = 2 * total_heads + 48

-- The goal is to prove that the number of cows C is 24.
theorem cows_count_24 : C = 24 :=
by
  sorry

end cows_count_24_l166_166282


namespace fraction_product_l166_166604

theorem fraction_product : 
  (4 / 2) * (3 / 6) * (10 / 5) * (15 / 30) * (20 / 10) * (45 / 90) * (50 / 25) * (60 / 120) = 1 := 
by
  sorry

end fraction_product_l166_166604


namespace prime_not_divisor_ab_cd_l166_166830

theorem prime_not_divisor_ab_cd {a b c d : ℕ} (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) (hd: 0 < d) 
  (p : ℕ) (hp : p = a + b + c + d) (hprime : Nat.Prime p) : ¬ p ∣ (a * b - c * d) := 
sorry

end prime_not_divisor_ab_cd_l166_166830


namespace find_t_from_x_l166_166568

theorem find_t_from_x (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
by
  sorry

end find_t_from_x_l166_166568


namespace speed_in_still_water_l166_166754

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) (h₁ : upstream_speed = 20) (h₂ : downstream_speed = 60) :
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

end speed_in_still_water_l166_166754


namespace sum_of_cubes_four_consecutive_integers_l166_166551

theorem sum_of_cubes_four_consecutive_integers (n : ℕ) (h : (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 = 11534) :
  (n-1)^3 + n^3 + (n+1)^3 + (n+2)^3 = 74836 :=
by
  sorry

end sum_of_cubes_four_consecutive_integers_l166_166551


namespace composite_a2_b2_l166_166255

-- Introduce the main definitions according to the conditions stated in a)
theorem composite_a2_b2 (x1 x2 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (a b : ℤ) 
  (ha : a = -(x1 + x2)) (hb : b = x1 * x2 - 1) : 
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ (a^2 + b^2) = m * n := 
by 
  sorry

end composite_a2_b2_l166_166255


namespace no_rearrangement_of_power_of_two_l166_166041

theorem no_rearrangement_of_power_of_two (k n : ℕ) (hk : k > 3) (hn : n > k) : 
  ∀ m : ℕ, 
    (m.toDigits = (2^k).toDigits → m ≠ 2^n) :=
by
  sorry

end no_rearrangement_of_power_of_two_l166_166041


namespace remainder_when_sum_div_by_8_l166_166054

theorem remainder_when_sum_div_by_8 (n : ℤ) : ((8 - n) + (n + 4)) % 8 = 4 := by
  sorry

end remainder_when_sum_div_by_8_l166_166054


namespace basketball_count_l166_166877

theorem basketball_count (s b v : ℕ) 
  (h1 : s = b + 23) 
  (h2 : v = s - 18)
  (h3 : v = 40) : b = 35 :=
by sorry

end basketball_count_l166_166877


namespace prob_all_co_captains_l166_166096

theorem prob_all_co_captains :
  let p1 := 1 / (Nat.choose 6 3) * 1 / 4,
      p2 := 1 / (Nat.choose 8 3) * 1 / 4,
      p3 := 1 / (Nat.choose 9 3) * 1 / 4,
      p4 := 1 / (Nat.choose 10 3) * 1 / 4 in
  p1 + p2 + p3 + p4 = 37 / 1680 :=
by
  let p1 := 1 / (Nat.choose 6 3) * 1 / 4,
      p2 := 1 / (Nat.choose 8 3) * 1 / 4,
      p3 := 1 / (Nat.choose 9 3) * 1 / 4,
      p4 := 1 / (Nat.choose 10 3) * 1 / 4
  have h : p1 + p2 + p3 + p4 = (1 / 20) / 4 + (1 / 56) / 4 + (1 / 84) / 4 + (1 / 120) / 4 := by sorry
  have h' : (1 / 20) / 4 + (1 / 56) / 4 + (1 / 84) / 4 + (1 / 120) / 4 = 37 / 1680 := by sorry
  exact h.trans h'

end prob_all_co_captains_l166_166096


namespace smallest_Y_l166_166069

theorem smallest_Y (S : ℕ) (h1 : (∀ d ∈ S.digits 10, d = 0 ∨ d = 1)) (h2 : 18 ∣ S) : 
  (∃ (Y : ℕ), Y = S / 18 ∧ ∀ (S' : ℕ), (∀ d ∈ S'.digits 10, d = 0 ∨ d = 1) → 18 ∣ S' → S' / 18 ≥ Y) → 
  Y = 6172839500 :=
sorry

end smallest_Y_l166_166069


namespace JiaZi_second_column_l166_166538

theorem JiaZi_second_column :
  let heavenlyStemsCycle := 10
  let earthlyBranchesCycle := 12
  let firstOccurrence := 1
  let lcmCycle := Nat.lcm heavenlyStemsCycle earthlyBranchesCycle
  let secondOccurrence := firstOccurrence + lcmCycle
  secondOccurrence = 61 :=
by
  sorry

end JiaZi_second_column_l166_166538


namespace outerCircumference_is_correct_l166_166553

noncomputable def π : ℝ := Real.pi  
noncomputable def innerCircumference : ℝ := 352 / 7
noncomputable def width : ℝ := 4.001609997739084

noncomputable def radius_inner : ℝ := innerCircumference / (2 * π)
noncomputable def radius_outer : ℝ := radius_inner + width
noncomputable def outerCircumference : ℝ := 2 * π * radius_outer

theorem outerCircumference_is_correct : outerCircumference = 341.194 := by
  sorry

end outerCircumference_is_correct_l166_166553


namespace anchuria_certification_prob_higher_in_2012_l166_166356

noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))

theorem anchuria_certification_prob_higher_in_2012
    (p : ℝ) (h : p = 0.25) :
  let prob_2011 := 1 - (binomial 20 0 p + binomial 20 1 p + binomial 20 2 p)
  let prob_2012 := 1 - (binomial 40 0 p + binomial 40 1 p + binomial 40 2 p + binomial 40 3 p +
                        binomial 40 4 p + binomial 40 5 p)
  prob_2012 > prob_2011 :=
by
  intros
  have h_prob_2011 : prob_2011 = 1 - ((binomial 20 0 p) + (binomial 20 1 p) + (binomial 20 2 p)), sorry
  have h_prob_2012 : prob_2012 = 1 - ((binomial 40 0 p) + (binomial 40 1 p) + (binomial 40 2 p) +
                                      (binomial 40 3 p) + (binomial 40 4 p) + (binomial 40 5 p)), sorry
  have pf_correct_prob_2011 : prob_2011 = 0.909, sorry
  have pf_correct_prob_2012 : prob_2012 = 0.957, sorry
  have pf_final : 0.957 > 0.909, from by norm_num
  exact pf_final

end anchuria_certification_prob_higher_in_2012_l166_166356


namespace sum_of_largest_two_l166_166417

-- Define the three numbers
def a := 10
def b := 11
def c := 12

-- Define the sum of the largest and the next largest numbers
def sum_of_largest_two_numbers (x y z : ℕ) : ℕ :=
  if x >= y ∧ y >= z then x + y
  else if x >= z ∧ z >= y then x + z
  else if y >= x ∧ x >= z then y + x
  else if y >= z ∧ z >= x then y + z
  else if z >= x ∧ x >= y then z + x
  else z + y

-- State the theorem to prove
theorem sum_of_largest_two (x y z : ℕ) : sum_of_largest_two_numbers x y z = 23 :=
by
  sorry

end sum_of_largest_two_l166_166417


namespace sector_area_l166_166797

theorem sector_area (r θ : ℝ) (hr : r = 2) (hθ : θ = (45 : ℝ) * (Real.pi / 180)) : 
  (1 / 2) * r^2 * θ = Real.pi / 2 := 
by
  sorry

end sector_area_l166_166797


namespace part1_part2_l166_166945

theorem part1 (x p : ℝ) (h : abs p ≤ 2) : (x^2 + p * x + 1 > 2 * x + p) ↔ (x < -1 ∨ 3 < x) := 
by 
  sorry

theorem part2 (x p : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : (x^2 + p * x + 1 > 2 * x + p) ↔ (-1 < p) := 
by 
  sorry

end part1_part2_l166_166945


namespace sum_first_five_terms_geometric_sequence_l166_166936

theorem sum_first_five_terms_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ):
  (∀ n, a (n+1) = a 1 * (1/2) ^ n) →
  a 1 = 16 →
  1/2 * (a 4 + a 7) = 9 / 8 →
  S 5 = (a 1 * (1 - (1 / 2) ^ 5)) / (1 - 1 / 2) →
  S 5 = 31 := by
  sorry

end sum_first_five_terms_geometric_sequence_l166_166936


namespace evaluate_expression_l166_166011

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 :=
by
  sorry

end evaluate_expression_l166_166011


namespace updated_mean_of_decremented_observations_l166_166444

theorem updated_mean_of_decremented_observations (mean : ℝ) (n : ℕ) (decrement : ℝ) 
  (h_mean : mean = 200) (h_n : n = 50) (h_decrement : decrement = 47) : 
  (mean * n - decrement * n) / n = 153 := 
by 
  sorry

end updated_mean_of_decremented_observations_l166_166444


namespace simple_interest_rate_l166_166565

-- Definitions based on conditions
def principal : ℝ := 750
def amount : ℝ := 900
def time : ℕ := 10

-- Statement to prove the rate of simple interest
theorem simple_interest_rate : 
  ∃ (R : ℝ), principal * R * time / 100 = amount - principal ∧ R = 2 :=
by
  sorry

end simple_interest_rate_l166_166565


namespace reciprocal_of_neg3_l166_166703

theorem reciprocal_of_neg3 : ∃ x : ℝ, (-3 : ℝ) * x = 1 :=  
begin
  use -1/3,
  norm_num,
end

end reciprocal_of_neg3_l166_166703


namespace vector_addition_l166_166631

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, 3)

-- Stating the problem: proving the sum of vectors a and b
theorem vector_addition : a + b = (3, 4) := 
by 
  -- Proof is not required as per the instructions
  sorry

end vector_addition_l166_166631


namespace tan_theta_eq_two_implies_expression_l166_166624

theorem tan_theta_eq_two_implies_expression (θ : ℝ) (h : Real.tan θ = 2) :
    (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
by
  -- Define trig identities and given condition
  have h_sin_cos : Real.sin θ = 2 / Real.sqrt 5 ∧ Real.cos θ = 1 / Real.sqrt 5 :=
    sorry -- This will be derived from the given condition h
  
  -- Main proof
  sorry

end tan_theta_eq_two_implies_expression_l166_166624


namespace difference_of_squares_l166_166111

theorem difference_of_squares (x y : ℝ) (h₁ : x + y = 20) (h₂ : x - y = 10) : x^2 - y^2 = 200 :=
by {
  sorry
}

end difference_of_squares_l166_166111


namespace anchurian_certificate_probability_l166_166363

open Probability

-- The probability of guessing correctly on a single question
def p : ℝ := 0.25

-- The probability of guessing incorrectly on a single question
def q : ℝ := 1.0 - p

-- Binomial Probability Mass Function
noncomputable def binomial_pmf (n : ℕ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

-- Passing probability in 2011
noncomputable def pass_prob_2011 : ℝ :=
  1 - (binomial_pmf 20 0 + binomial_pmf 20 1 + binomial_pmf 20 2)

-- Passing probability in 2012
noncomputable def pass_prob_2012 : ℝ :=
  1 - (binomial_pmf 40 0 + binomial_pmf 40 1 + binomial_pmf 40 2 + binomial_pmf 40 3 + binomial_pmf 40 4 + binomial_pmf 40 5)

theorem anchurian_certificate_probability :
  pass_prob_2012 > pass_prob_2011 :=
sorry

end anchurian_certificate_probability_l166_166363


namespace fraction_power_rule_l166_166428

theorem fraction_power_rule :
  (5 / 6) ^ 4 = (625 : ℚ) / 1296 := 
by sorry

end fraction_power_rule_l166_166428


namespace oranges_to_put_back_l166_166597

theorem oranges_to_put_back
  (price_apple price_orange : ℕ)
  (A_all O_all : ℕ)
  (mean_initial_fruit mean_final_fruit : ℕ)
  (A O x : ℕ)
  (h_price_apple : price_apple = 40)
  (h_price_orange : price_orange = 60)
  (h_total_fruit : A_all + O_all = 10)
  (h_mean_initial : mean_initial_fruit = 54)
  (h_mean_final : mean_final_fruit = 50)
  (h_total_cost_initial : price_apple * A_all + price_orange * O_all = mean_initial_fruit * (A_all + O_all))
  (h_total_cost_final : price_apple * A + price_orange * (O - x) = mean_final_fruit * (A + (O - x)))
  : x = 4 := 
  sorry

end oranges_to_put_back_l166_166597


namespace min_bought_chocolates_l166_166297

variable (a b : ℕ)

theorem min_bought_chocolates :
    ∃ a : ℕ, 
        ∃ b : ℕ, 
            b = a + 41 
            ∧ (376 - a - b = 3 * a) 
            ∧ a = 67 :=
by
  sorry

end min_bought_chocolates_l166_166297


namespace quadratic_has_one_positive_and_one_negative_root_l166_166550

theorem quadratic_has_one_positive_and_one_negative_root
  (a : ℝ) (h₁ : a ≠ 0) (h₂ : a < -1) :
  ∃ x₁ x₂ : ℝ, (a * x₁^2 + 2 * x₁ + 1 = 0) ∧ (a * x₂^2 + 2 * x₂ + 1 = 0) ∧ (x₁ > 0) ∧ (x₂ < 0) :=
by
  sorry

end quadratic_has_one_positive_and_one_negative_root_l166_166550


namespace shoe_pairing_probability_l166_166099

theorem shoe_pairing_probability :
  let m := 5
  let n := 36
  let total_probability := (m : ℚ) / n
  total_probability = 5 / 36 → m + n = 41 :=
begin
  intros h,
  exact (by injection h),
end

end shoe_pairing_probability_l166_166099


namespace certain_number_unique_l166_166878

-- Define the necessary conditions and statement
def is_certain_number (n : ℕ) : Prop :=
  (∃ k : ℕ, 25 * k = n) ∧ (∃ k : ℕ, 35 * k = n) ∧ 
  (n > 0) ∧ (∃ a b c : ℕ, 1 ≤ a * n ∧ a * n ≤ 1050 ∧ 1 ≤ b * n ∧ b * n ≤ 1050 ∧ 1 ≤ c * n ∧ c * n ≤ 1050 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem certain_number_unique :
  ∃ n : ℕ, is_certain_number n ∧ n = 350 :=
by 
  sorry

end certain_number_unique_l166_166878


namespace find_z_value_l166_166500

theorem find_z_value (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (h1 : x = 2 + 1 / z)
  (h2 : z = 3 + 1 / x) : 
  z = (3 + Real.sqrt 15) / 2 :=
by
  sorry

end find_z_value_l166_166500


namespace add_one_five_times_l166_166854

theorem add_one_five_times (m n : ℕ) (h : n = m + 5) : n - (m + 1) = 4 :=
by
  sorry

end add_one_five_times_l166_166854


namespace max_distance_line_ellipse_l166_166753

theorem max_distance_line_ellipse :
  (∀ (t : ℝ), ¬(t^2 < 5) → true) →
  ∃ t : ℝ, -sqrt 5 < t ∧ t < sqrt 5 ∧
  (let x1 := -4 * t / 5 in
   let x2 := (4 * t^2 - 4) / 5 in
   real.sqrt 2 * real.sqrt ((x1 + x1) ^ 2 - 4 * x1 * x2) ≤ 4 * real.sqrt 10 / 5) :=
begin
  sorry
end

end max_distance_line_ellipse_l166_166753


namespace inequality_proof_l166_166947

variables (a b c d : ℝ)

theorem inequality_proof 
  (h1 : a + b > abs (c - d)) 
  (h2 : c + d > abs (a - b)) : 
  a + c > abs (b - d) := 
sorry

end inequality_proof_l166_166947


namespace cost_of_shirt_l166_166566

theorem cost_of_shirt (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 71) : S = 15 :=
by
  sorry

end cost_of_shirt_l166_166566


namespace girls_attending_event_l166_166308

theorem girls_attending_event (g b : ℕ) 
  (h1 : g + b = 1500)
  (h2 : 3 / 4 * g + 2 / 5 * b = 900) :
  3 / 4 * g = 643 := 
by
  sorry

end girls_attending_event_l166_166308


namespace intersection_complement_l166_166949

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x : ℝ | x > 0 }

-- Define the complement of B
def complement_B : Set ℝ := { x : ℝ | x ≤ 0 }

-- The theorem we need to prove
theorem intersection_complement :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 0 } := 
by
  sorry

end intersection_complement_l166_166949


namespace problem_statement_l166_166024

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^4 + 2*m^3 - m + 2007 = 2007 := 
by 
  sorry

end problem_statement_l166_166024


namespace remaining_student_number_l166_166127

-- Definitions based on given conditions
def total_students := 48
def sample_size := 6
def sampled_students := [5, 21, 29, 37, 45]

-- Interval calculation and pattern definition based on systematic sampling
def sampling_interval := total_students / sample_size
def sampled_student_numbers (n : Nat) : Nat := 5 + sampling_interval * (n - 1)

-- Prove the student number within the sample
theorem remaining_student_number : ∃ n, n ∉ sampled_students ∧ sampled_student_numbers n = 13 :=
by
  sorry

end remaining_student_number_l166_166127


namespace segment_length_is_ten_l166_166785

-- Definition of the cube root function and the absolute value
def cube_root (x : ℝ) : ℝ := x^(1/3)

def absolute (x : ℝ) : ℝ := abs x

-- The prerequisites as conditions for the endpoints
def endpoints_satisfy (x : ℝ) : Prop := absolute (x - cube_root 27) = 5

-- Length of the segment determined by the endpoints
def segment_length (x1 x2 : ℝ) : ℝ := absolute (x2 - x1)

-- Theorem statement
theorem segment_length_is_ten : (∀ x, endpoints_satisfy x) → segment_length (-2) 8 = 10 :=
by
  intro h
  sorry

end segment_length_is_ten_l166_166785


namespace ratio_qp_l166_166995

theorem ratio_qp (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 6 → 
    P / (x + 3) + Q / (x * (x - 6)) = (x^2 - 4 * x + 15) / (x * (x + 3) * (x - 6))) : 
  Q / P = 5 := 
sorry

end ratio_qp_l166_166995


namespace card_S_l166_166226

def a (n : ℕ) : ℕ := 2 ^ n

def b (n : ℕ) : ℕ := 5 * n - 1

def S : Finset ℕ := 
  (Finset.range 2016).image a ∩ (Finset.range (a 2015 + 1)).image b

theorem card_S : S.card = 504 := 
  sorry

end card_S_l166_166226


namespace methane_combined_l166_166016

def balancedEquation (CH₄ O₂ CO₂ H₂O : ℕ) : Prop :=
  CH₄ = 1 ∧ O₂ = 2 ∧ CO₂ = 1 ∧ H₂O = 2

theorem methane_combined {moles_CH₄ moles_O₂ moles_H₂O : ℕ}
  (h₁ : moles_O₂ = 2)
  (h₂ : moles_H₂O = 2)
  (h_eq : balancedEquation moles_CH₄ moles_O₂ 1 moles_H₂O) : 
  moles_CH₄ = 1 :=
by
  sorry

end methane_combined_l166_166016


namespace find_a_if_perpendicular_l166_166935

def m (a : ℝ) : ℝ × ℝ := (3, a - 1)
def n (a : ℝ) : ℝ × ℝ := (a, -2)

theorem find_a_if_perpendicular (a : ℝ) (h : (m a).fst * (n a).fst + (m a).snd * (n a).snd = 0) : a = -2 :=
by sorry

end find_a_if_perpendicular_l166_166935


namespace heath_time_spent_l166_166951

variables (rows_per_carrot : ℕ) (plants_per_row : ℕ) (carrots_per_hour : ℕ) (total_hours : ℕ)

def total_carrots (rows_per_carrot plants_per_row : ℕ) : ℕ :=
  rows_per_carrot * plants_per_row

def time_spent (total_carrots carrots_per_hour : ℕ) : ℕ :=
  total_carrots / carrots_per_hour

theorem heath_time_spent
  (h1 : rows_per_carrot = 400)
  (h2 : plants_per_row = 300)
  (h3 : carrots_per_hour = 6000)
  (h4 : total_hours = 20) :
  time_spent (total_carrots rows_per_carrot plants_per_row) carrots_per_hour = total_hours :=
by
  sorry

end heath_time_spent_l166_166951


namespace cookies_per_tray_l166_166307

def num_trays : ℕ := 4
def num_packs : ℕ := 8
def cookies_per_pack : ℕ := 12
def total_cookies : ℕ := num_packs * cookies_per_pack

theorem cookies_per_tray : total_cookies / num_trays = 24 := by
  sorry

end cookies_per_tray_l166_166307


namespace find_x_l166_166813

theorem find_x (x : ℚ) (h : (3 * x + 4) / 5 = 15) : x = 71 / 3 :=
by
  sorry

end find_x_l166_166813


namespace set_subset_of_inter_union_l166_166229

variable {α : Type} [Nonempty α]
variables {A B C : Set α}

-- The main theorem based on the problem statement
theorem set_subset_of_inter_union (h : A ∩ B = B ∪ C) : C ⊆ B :=
by
  sorry

end set_subset_of_inter_union_l166_166229


namespace valid_two_digit_numbers_l166_166714

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)

-- Prove the statement about two-digit numbers satisfying the condition
theorem valid_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by
  sorry

end valid_two_digit_numbers_l166_166714


namespace probability_sum_of_remaining_bills_l166_166149

def bagA : List ℕ := [10, 10, 1, 1, 1]
def bagB : List ℕ := [5, 5, 5, 5, 1, 1, 1]
def totalWaysA := Nat.choose 5 2
def totalWaysB := Nat.choose 7 2

def remainingSums (bag : List ℕ) (draw : List ℕ) : ℕ :=
  bag.sum - draw.sum

def isFavored (remA : ℕ) (remB : ℕ) : Bool :=
  remA > remB

theorem probability_sum_of_remaining_bills :
  ∑ (drawA ∈ (bagA.combinations 2)) (drawB ∈ (bagB.combinations 2)),
    if isFavored (remainingSums bagA drawA) (remainingSums bagB drawB)
    then (1 : ℚ)
    else (0 : ℚ) = (9 / 35 : ℚ) :=
by
  sorry

end probability_sum_of_remaining_bills_l166_166149


namespace equivalence_of_min_perimeter_and_cyclic_quadrilateral_l166_166655

-- Definitions for points P, Q, R, S on sides of quadrilateral ABCD
-- Function definitions for conditions and equivalence of stated problems

variable {A B C D P Q R S : Type*} 

def is_on_side (P : Type*) (A B : Type*) : Prop := sorry
def is_interior_point (P : Type*) (A B : Type*) : Prop := sorry
def is_convex_quadrilateral (A B C D : Type*) : Prop := sorry
def is_cyclic_quadrilateral (A B C D : Type*) : Prop := sorry
def has_circumcenter_interior (A B C D : Type*) : Prop := sorry
def has_minimal_perimeter (P Q R S : Type*) : Prop := sorry

theorem equivalence_of_min_perimeter_and_cyclic_quadrilateral 
  (h1 : is_convex_quadrilateral A B C D) 
  (hP : is_on_side P A B ∧ is_interior_point P A B) 
  (hQ : is_on_side Q B C ∧ is_interior_point Q B C) 
  (hR : is_on_side R C D ∧ is_interior_point R C D) 
  (hS : is_on_side S D A ∧ is_interior_point S D A) :
  (∃ P' Q' R' S', has_minimal_perimeter P' Q' R' S') ↔ (is_cyclic_quadrilateral A B C D ∧ has_circumcenter_interior A B C D) :=
sorry

end equivalence_of_min_perimeter_and_cyclic_quadrilateral_l166_166655


namespace reciprocal_of_neg_three_l166_166706

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l166_166706


namespace eq_to_general_quadratic_l166_166403

theorem eq_to_general_quadratic (x : ℝ) : (x - 1) * (x + 1) = 1 → x^2 - 2 = 0 :=
by
  sorry

end eq_to_general_quadratic_l166_166403


namespace doves_count_l166_166326

theorem doves_count 
  (num_doves : ℕ)
  (num_eggs_per_dove : ℕ)
  (hatch_rate : ℚ)
  (initial_doves : num_doves = 50)
  (eggs_per_dove : num_eggs_per_dove = 5)
  (hatch_fraction : hatch_rate = 7/9) :
  (num_doves + Int.toNat ((hatch_rate * num_doves * num_eggs_per_dove).floor)) = 244 :=
by
  sorry

end doves_count_l166_166326


namespace man_double_son_age_in_2_years_l166_166133

def present_age_son : ℕ := 25
def age_difference : ℕ := 27
def years_to_double_age : ℕ := 2

theorem man_double_son_age_in_2_years 
  (S : ℕ := present_age_son)
  (M : ℕ := S + age_difference)
  (Y : ℕ := years_to_double_age) : 
  M + Y = 2 * (S + Y) :=
by sorry

end man_double_son_age_in_2_years_l166_166133


namespace no_solution_nat_x_satisfies_eq_l166_166071

def sum_digits (x : ℕ) : ℕ := x.digits 10 |>.sum

theorem no_solution_nat_x_satisfies_eq (x : ℕ) :
  ¬ (x + sum_digits x + sum_digits (sum_digits x) = 2014) :=
by
  sorry

end no_solution_nat_x_satisfies_eq_l166_166071


namespace allocation_schemes_count_l166_166301

open BigOperators -- For working with big operator notations
open Finset -- For working with finite sets
open Nat -- For natural number operations

-- Define the number of students and dormitories
def num_students : ℕ := 7
def num_dormitories : ℕ := 2

-- Define the constraint for minimum students in each dormitory
def min_students_in_dormitory : ℕ := 2

-- Compute the number of ways to allocate students given the conditions
noncomputable def number_of_allocation_schemes : ℕ :=
  (Nat.choose num_students 3) * (Nat.choose 4 2) + (Nat.choose num_students 2) * (Nat.choose 5 2)

-- The theorem stating the total number of allocation schemes
theorem allocation_schemes_count :
  number_of_allocation_schemes = 112 :=
  by sorry

end allocation_schemes_count_l166_166301


namespace square_nonneg_l166_166449

theorem square_nonneg (x : ℝ) : x^2 ≥ 0 :=
sorry

end square_nonneg_l166_166449


namespace cost_to_selling_ratio_l166_166873

theorem cost_to_selling_ratio (cp sp: ℚ) (h: sp = cp * (1 + 0.25)): cp / sp = 4 / 5 :=
by
  sorry

end cost_to_selling_ratio_l166_166873


namespace Peter_can_always_ensure_three_distinct_real_roots_l166_166896

noncomputable def cubic_has_three_distinct_real_roots (b d : ℝ) : Prop :=
∃ (a : ℝ), ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
  (r1 * r2 * r3 = -a) ∧ (r1 + r2 + r3 = -b) ∧ (r1 * r2 + r2 * r3 + r3 * r1 = -d)

theorem Peter_can_always_ensure_three_distinct_real_roots (b d : ℝ) :
  cubic_has_three_distinct_real_roots b d :=
sorry

end Peter_can_always_ensure_three_distinct_real_roots_l166_166896


namespace min_period_f_and_max_value_g_l166_166476

open Real

noncomputable def f (x : ℝ) : ℝ := abs (sin x) + abs (cos x)
noncomputable def g (x : ℝ) : ℝ := sin x ^ 3 - sin x

theorem min_period_f_and_max_value_g :
  (∀ m : ℝ, (∀ x : ℝ, f (x + m) = f x) -> m = π / 2) ∧ 
  (∃ n : ℝ, ∀ x : ℝ, g x ≤ n ∧ (∃ x : ℝ, g x = n)) ∧ 
  (∃ mn : ℝ, mn = (π / 2) * (2 * sqrt 3 / 9)) := 
by sorry

end min_period_f_and_max_value_g_l166_166476


namespace minimum_positive_announcements_l166_166144

theorem minimum_positive_announcements (x y : ℕ) (h : x * (x - 1) = 132) (positive_products negative_products : ℕ)
  (hp : positive_products = y * (y - 1)) (hn : negative_products = (x - y) * (x - y - 1)) 
  (h_sum : positive_products + negative_products = 132) : 
  y = 2 :=
by sorry

end minimum_positive_announcements_l166_166144


namespace intersection_P_Q_l166_166641

def P : Set ℝ := { x | x^2 - x = 0 }
def Q : Set ℝ := { x | x^2 + x = 0 }

theorem intersection_P_Q : (P ∩ Q) = {0} := 
by
  sorry

end intersection_P_Q_l166_166641


namespace baker_remaining_cakes_l166_166599

def initial_cakes : ℝ := 167.3
def sold_cakes : ℝ := 108.2
def remaining_cakes : ℝ := initial_cakes - sold_cakes

theorem baker_remaining_cakes : remaining_cakes = 59.1 := by
  sorry

end baker_remaining_cakes_l166_166599


namespace minimum_shots_required_l166_166661

noncomputable def minimum_shots_to_sink_boat : ℕ := 4000

-- Definitions for the problem conditions.
structure Boat :=
(square_side : ℕ)
(base1 : ℕ)
(base2 : ℕ)
(rotatable : Bool)

def boat : Boat := { square_side := 1, base1 := 1, base2 := 3, rotatable := true }

def grid_size : ℕ := 100

def shot_covers_triangular_half : Prop := sorry -- Assumption: Define this appropriately

-- Problem statement in Lean 4
theorem minimum_shots_required (boat_within_grid : Bool) : 
  Boat → grid_size = 100 → boat_within_grid → minimum_shots_to_sink_boat = 4000 :=
by
  -- Here you would do the full proof which we assume is "sorry" for now
  sorry

end minimum_shots_required_l166_166661


namespace initial_amount_invested_l166_166347

-- Definition of the conditions as Lean definitions
def initial_amount_interest_condition (A r : ℝ) : Prop := 25000 = A * r
def interest_rate_condition (r : ℝ) : Prop := r = 5

-- The main theorem we want to prove
theorem initial_amount_invested (A r : ℝ) (h1 : initial_amount_interest_condition A r) (h2 : interest_rate_condition r) : A = 5000 :=
by {
  sorry
}

end initial_amount_invested_l166_166347


namespace total_cost_of_fencing_l166_166996

theorem total_cost_of_fencing (length breadth : ℕ) (cost_per_metre : ℕ) 
  (h1 : length = breadth + 20) 
  (h2 : length = 200) 
  (h3 : cost_per_metre = 26): 
  2 * (length + breadth) * cost_per_metre = 20140 := 
by sorry

end total_cost_of_fencing_l166_166996


namespace reciprocal_of_neg_three_l166_166709

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l166_166709


namespace convert_speed_to_mps_l166_166314

-- Define given speeds and conversion factors
def speed_kmph : ℝ := 63
def kilometers_to_meters : ℝ := 1000
def hours_to_seconds : ℝ := 3600

-- Assert the conversion
theorem convert_speed_to_mps : speed_kmph * (kilometers_to_meters / hours_to_seconds) = 17.5 := by
  sorry

end convert_speed_to_mps_l166_166314


namespace trevor_pages_l166_166098

theorem trevor_pages (p1 p2 p3 : ℕ) (h1 : p1 = 72) (h2 : p2 = 72) (h3 : p3 = p1 + 4) : 
    p1 + p2 + p3 = 220 := 
by 
    sorry

end trevor_pages_l166_166098


namespace simplify_expression_l166_166678

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l166_166678


namespace penny_paid_amount_l166_166368

-- Definitions based on conditions
def bulk_price : ℕ := 5
def minimum_spend : ℕ := 40
def tax_rate : ℕ := 1
def excess_pounds : ℕ := 32

-- Expression for total calculated cost
def total_pounds := (minimum_spend / bulk_price) + excess_pounds
def cost_before_tax := total_pounds * bulk_price
def total_tax := total_pounds * tax_rate
def total_cost := cost_before_tax + total_tax

-- Required proof statement
theorem penny_paid_amount : total_cost = 240 := 
by 
  sorry

end penny_paid_amount_l166_166368


namespace semicircle_parametric_equation_correct_l166_166063

-- Define the conditions of the problem in terms of Lean definitions and propositions.

def semicircle_parametric_equation : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ (Real.pi / 2) →
    ∃ α : ℝ, α = 2 * θ ∧ 0 ≤ α ∧ α ≤ Real.pi ∧
    (∃ (x y : ℝ), x = 1 + Real.cos α ∧ y = Real.sin α)

-- Statement that we will prove
theorem semicircle_parametric_equation_correct : semicircle_parametric_equation :=
  sorry

end semicircle_parametric_equation_correct_l166_166063


namespace general_term_a_n_l166_166816

theorem general_term_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (2/3) * a n + 1/3) :
  ∀ n, a n = (-2)^(n-1) :=
sorry

end general_term_a_n_l166_166816


namespace cherries_used_l166_166125

theorem cherries_used (initial remaining used : ℕ) (h_initial : initial = 77) (h_remaining : remaining = 17) (h_used : used = initial - remaining) : used = 60 :=
by
  rw [h_initial, h_remaining] at h_used
  simp at h_used
  exact h_used

end cherries_used_l166_166125


namespace simplify_polynomial_l166_166392

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  x * (4 * x^2 - 2) - 5 * (x^2 - 3 * x + 5) = 4 * x^3 - 5 * x^2 + 13 * x - 25 :=
by
  sorry

end simplify_polynomial_l166_166392


namespace parabola_relationship_l166_166998

theorem parabola_relationship 
  (c : ℝ) (y1 y2 y3 : ℝ) 
  (h1 : y1 = 2*(-2 - 1)^2 + c) 
  (h2 : y2 = 2*(0 - 1)^2 + c) 
  (h3 : y3 = 2*((5:ℝ)/3 - 1)^2 + c):
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end parabola_relationship_l166_166998


namespace valid_two_digit_numbers_l166_166713

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)

-- Prove the statement about two-digit numbers satisfying the condition
theorem valid_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by
  sorry

end valid_two_digit_numbers_l166_166713


namespace max_value_x_plus_2y_max_of_x_plus_2y_l166_166938

def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 4 = 1

theorem max_value_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  x + 2 * y ≤ Real.sqrt 22 :=
sorry

theorem max_of_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  ∃θ ∈ Set.Icc 0 (2 * Real.pi), (x = Real.sqrt 6 * Real.cos θ) ∧ (y = 2 * Real.sin θ) :=
sorry

end max_value_x_plus_2y_max_of_x_plus_2y_l166_166938


namespace rebecca_groups_eq_l166_166530

-- Definitions
def total_eggs : ℕ := 15
def eggs_per_group : ℕ := 5
def expected_groups : ℕ := 3

-- Theorem to prove
theorem rebecca_groups_eq :
  total_eggs / eggs_per_group = expected_groups :=
by
  sorry

end rebecca_groups_eq_l166_166530


namespace p_plus_q_is_32_l166_166089

noncomputable def slope_sums_isosceles_trapezoid : ℚ := 
  let E := (30, 150)
  let H := (31, 159)
  let translated_E := (0, 0)
  let translated_H := (1, 9)
  let relative_prime (a b : ℕ) := ∀ d > 1, d ∣ a → d ∣ b → False
  ∑ m in { 4/5, -1, -5/4, 1 }, |m|

theorem p_plus_q_is_32 :
  ∃ p q : ℕ, relative_prime p q ∧ slope_sums_isosceles_trapezoid = p / q ∧ p + q = 32 :=
begin
  sorry
end

end p_plus_q_is_32_l166_166089


namespace evaluate_expression_l166_166462

theorem evaluate_expression : 
  3 * (-4) - ((5 * (-5)) * (-2)) + 6 = -56 := 
by 
  sorry

end evaluate_expression_l166_166462


namespace triangles_pentagons_difference_l166_166304

theorem triangles_pentagons_difference :
  ∃ x y : ℕ, 
  (x + y = 50) ∧ (3 * x + 5 * y = 170) ∧ (x - y = 30) :=
sorry

end triangles_pentagons_difference_l166_166304


namespace inequality_holds_l166_166667

variable {x y : ℝ}

theorem inequality_holds (x : ℝ) (y : ℝ) (hy : y ≥ 5) : 
  x^2 - 2 * x * Real.sqrt (y - 5) + y^2 + y - 30 ≥ 0 := 
sorry

end inequality_holds_l166_166667


namespace solution_set_of_inequality_l166_166874

-- Definition of the inequality and its transformation
def inequality (x : ℝ) : Prop :=
  (x - 2) / (x + 1) ≤ 0

noncomputable def transformed_inequality (x : ℝ) : Prop :=
  (x + 1) * (x - 2) ≤ 0 ∧ x + 1 ≠ 0

-- Statement of the theorem
theorem solution_set_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | -1 < x ∧ x ≤ 2} := 
sorry

end solution_set_of_inequality_l166_166874


namespace expected_profit_two_machines_l166_166418

noncomputable def expected_profit : ℝ :=
  let p := 0.2
  let q := 1 - p
  let loss := -50000 -- 50,000 loss when malfunction
  let profit := 100000 -- 100,000 profit when working normally
  let expected_single_machine := q * profit + p * loss
  2 * expected_single_machine

theorem expected_profit_two_machines : expected_profit = 140000 := by
  sorry

end expected_profit_two_machines_l166_166418


namespace unique_two_scoop_sundaes_l166_166773

theorem unique_two_scoop_sundaes (n : ℕ) (hn : n = 8) : ∃ k, k = Nat.choose 8 2 :=
by
  use 28
  sorry

end unique_two_scoop_sundaes_l166_166773


namespace simplify_expression_l166_166684

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l166_166684


namespace simplify_expression_l166_166683

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l166_166683


namespace quotient_transformation_l166_166092

theorem quotient_transformation (A B : ℕ) (h1 : B ≠ 0) (h2 : (A : ℝ) / B = 0.514) :
  ((10 * A : ℝ) / (B / 100)) = 514 :=
by
  -- skipping the proof
  sorry

end quotient_transformation_l166_166092


namespace flat_rate_first_night_l166_166584

theorem flat_rate_first_night
  (f n : ℚ)
  (h1 : f + 3 * n = 210)
  (h2 : f + 6 * n = 350)
  : f = 70 :=
by
  sorry

end flat_rate_first_night_l166_166584


namespace lowest_test_score_dropped_l166_166112

theorem lowest_test_score_dropped (A B C D : ℕ) 
  (h_avg_four : A + B + C + D = 140) 
  (h_avg_three : A + B + C = 120) : 
  D = 20 := 
by
  sorry

end lowest_test_score_dropped_l166_166112


namespace sqrt_sum_eq_ten_l166_166610

theorem sqrt_sum_eq_ten :
  Real.sqrt ((5 - 4*Real.sqrt 2)^2) + Real.sqrt ((5 + 4*Real.sqrt 2)^2) = 10 := 
by 
  sorry

end sqrt_sum_eq_ten_l166_166610


namespace fraction_power_four_l166_166430

theorem fraction_power_four :
  (5 / 6) ^ 4 = 625 / 1296 :=
by sorry

end fraction_power_four_l166_166430


namespace initial_pieces_of_fruit_l166_166383

-- Definitions for the given problem
def pieces_eaten_in_first_four_days : ℕ := 5
def pieces_kept_for_next_week : ℕ := 2
def pieces_brought_to_school : ℕ := 3

-- Problem statement
theorem initial_pieces_of_fruit 
  (pieces_eaten : ℕ)
  (pieces_kept : ℕ)
  (pieces_brought : ℕ)
  (h1 : pieces_eaten = pieces_eaten_in_first_four_days)
  (h2 : pieces_kept = pieces_kept_for_next_week)
  (h3 : pieces_brought = pieces_brought_to_school) :
  pieces_eaten + pieces_kept + pieces_brought = 10 := 
sorry

end initial_pieces_of_fruit_l166_166383


namespace sum_of_ages_l166_166519

-- Problem statement:
-- Given: The product of their ages is 144.
-- Prove: The sum of their ages is 16.
theorem sum_of_ages (k t : ℕ) (htwins : t > k) (hprod : 2 * t * k = 144) : 2 * t + k = 16 := 
sorry

end sum_of_ages_l166_166519


namespace students_in_both_math_and_chem_l166_166210

theorem students_in_both_math_and_chem (students total math physics chem math_physics physics_chem : ℕ) :
  total = 36 →
  students ≤ 2 →
  math = 26 →
  physics = 15 →
  chem = 13 →
  math_physics = 6 →
  physics_chem = 4 →
  math + physics + chem - math_physics - physics_chem - students = total →
  students = 8 := by
  intros h_total h_students h_math h_physics h_chem h_math_physics h_physics_chem h_equation
  sorry

end students_in_both_math_and_chem_l166_166210


namespace money_sum_l166_166562

theorem money_sum (A B : ℕ) (h₁ : (1 / 3 : ℝ) * A = (1 / 4 : ℝ) * B) (h₂ : B = 484) : A + B = 847 := by
  sorry

end money_sum_l166_166562


namespace factor_expression_l166_166012

theorem factor_expression (b : ℝ) : 45 * b^2 + 135 * b^3 = 45 * b^2 * (1 + 3 * b) :=
by
  sorry

end factor_expression_l166_166012


namespace simplify_expression_l166_166167

variable (x y : ℝ)

theorem simplify_expression (A B : ℝ) (hA : A = x^2) (hB : B = y^2) :
  (A + B) / (A - B) + (A - B) / (A + B) = 2 * (x^4 + y^4) / (x^4 - y^4) :=
by {
  sorry
}

end simplify_expression_l166_166167


namespace george_initial_candy_l166_166189

theorem george_initial_candy (number_of_bags : ℕ) (pieces_per_bag : ℕ) 
  (h1 : number_of_bags = 8) (h2 : pieces_per_bag = 81) : 
  number_of_bags * pieces_per_bag = 648 := 
by 
  sorry

end george_initial_candy_l166_166189


namespace higher_probability_in_2012_l166_166362

def bernoulli_probability (n k : ℕ) (p : ℝ) : ℝ :=
  ∑ i in finset.range (k + 1), nat.choose n i * (p ^ i) * ((1 - p) ^ (n - i))

theorem higher_probability_in_2012 : 
  let p := 0.25
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  let pass_prob_2011 := 1 - bernoulli_probability n2011 (k2011 - 1) p
  let pass_prob_2012 := 1 - bernoulli_probability n2012 (k2012 - 1) p
  pass_prob_2012 > pass_prob_2011 :=
by
  -- We would provide the actual proof here, but for now, we use sorry.
  sorry

end higher_probability_in_2012_l166_166362


namespace chairs_to_remove_l166_166128

-- Defining the conditions
def chairs_per_row : Nat := 15
def total_chairs : Nat := 180
def expected_attendees : Nat := 125

-- Main statement to prove
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ) : 
  chairs_per_row = 15 → 
  total_chairs = 180 → 
  expected_attendees = 125 → 
  ∃ n, total_chairs - (chairs_per_row * n) = 45 ∧ n * chairs_per_row ≥ expected_attendees := 
by
  intros h1 h2 h3
  sorry

end chairs_to_remove_l166_166128


namespace simplify_expression_l166_166685

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l166_166685


namespace find_matrix_N_l166_166928

open Matrix

variable (u : Fin 3 → ℝ)

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]

-- Define vector v as the fixed vector in the problem
def v : Fin 3 → ℝ := ![7, 3, -9]

-- Define matrix N as the matrix to be found
def N : Matrix (Fin 3) (Fin 3) ℝ := ![![0, 9, 3], ![-9, 0, -7], ![-3, 7, 0]]

-- Define the requirement condition
theorem find_matrix_N :
  ∀ (u : Fin 3 → ℝ), (N.mulVec u) = cross_product v u :=
by
  sorry

end find_matrix_N_l166_166928


namespace product_area_perimeter_eq_104sqrt26_l166_166982

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2).sqrt

noncomputable def side_length := distance (5, 5) (0, 4)

noncomputable def area_of_square := side_length ^ 2

noncomputable def perimeter_of_square := 4 * side_length

noncomputable def product_area_perimeter := area_of_square * perimeter_of_square

theorem product_area_perimeter_eq_104sqrt26 :
  product_area_perimeter = 104 * Real.sqrt 26 :=
by 
  -- placeholder for the proof
  sorry

end product_area_perimeter_eq_104sqrt26_l166_166982


namespace square_area_l166_166852

theorem square_area (x : ℝ) (h1 : x = 60) : x^2 = 1200 :=
by
  sorry

end square_area_l166_166852


namespace second_sweet_red_probability_l166_166285

theorem second_sweet_red_probability (x y : ℕ) : 
  (y / (x + y : ℝ)) = y / (x + y + 10) * x / (x + y) + (y + 10) / (x + y + 10) * y / (x + y) :=
by
  sorry

end second_sweet_red_probability_l166_166285


namespace min_a_add_c_l166_166502

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def angle_ABC : ℝ := 2 * Real.pi / 3
noncomputable def BD : ℝ := 1

-- The bisector of angle ABC intersects AC at point D
-- Angle bisector theorem and the given information
theorem min_a_add_c : ∃ a c : ℝ, (angle_ABC = 2 * Real.pi / 3) → (BD = 1) → (a * c = a + c) → (a + c ≥ 4) :=
by
  sorry

end min_a_add_c_l166_166502


namespace solve_for_z_l166_166030

open Complex

theorem solve_for_z (z : ℂ) (i : ℂ) (h1 : i = Complex.I) (h2 : z * i = 1 + i) : z = 1 - i :=
by sorry

end solve_for_z_l166_166030


namespace cornelia_age_l166_166059

theorem cornelia_age :
  ∃ C : ℕ, 
  (∃ K : ℕ, K = 30 ∧ (C + 20 = 2 * (K + 20))) ∧
  ((K - 5)^2 = 3 * (C - 5)) := by
  sorry

end cornelia_age_l166_166059


namespace angle_sum_of_octagon_and_triangle_l166_166823

-- Define the problem setup
def is_interior_angle_of_regular_polygon (n : ℕ) (angle : ℝ) : Prop :=
  angle = 180 * (n - 2) / n

def is_regular_octagon_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 8 angle

def is_equilateral_triangle_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 3 angle

-- The statement of the problem
theorem angle_sum_of_octagon_and_triangle :
  ∃ angle_ABC angle_ABD : ℝ,
    is_regular_octagon_angle angle_ABC ∧
    is_equilateral_triangle_angle angle_ABD ∧
    angle_ABC + angle_ABD = 195 :=
sorry

end angle_sum_of_octagon_and_triangle_l166_166823


namespace base_length_of_vessel_l166_166581

def volume_of_cube (edge : ℝ) := edge^3

def volume_of_displaced_water (L width rise : ℝ) := L * width * rise

theorem base_length_of_vessel (edge width rise L : ℝ) 
  (h1 : edge = 15) (h2 : width = 15) (h3 : rise = 11.25) 
  (h4 : volume_of_displaced_water L width rise = volume_of_cube edge) : 
  L = 20 :=
by
  sorry

end base_length_of_vessel_l166_166581


namespace min_value_of_expression_l166_166348

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end min_value_of_expression_l166_166348


namespace solve_system_l166_166106

theorem solve_system :
  ∀ (x y : ℝ) (triangle : ℝ), 
  (2 * x - 3 * y = 5) ∧ (x + y = triangle) ∧ (x = 4) →
  (y = 1) ∧ (triangle = 5) :=
by
  -- Skipping the proof steps
  sorry

end solve_system_l166_166106


namespace correlation_1_and_3_l166_166250

-- Define the conditions as types
def relationship1 : Type := ∀ (age : ℕ) (fat_content : ℝ), Prop
def relationship2 : Type := ∀ (curve_point : ℝ × ℝ), Prop
def relationship3 : Type := ∀ (production : ℝ) (climate : ℝ), Prop
def relationship4 : Type := ∀ (student : ℕ) (student_ID : ℕ), Prop

-- Define what it means for two relationships to have a correlation
def has_correlation (rel1 rel2 : Type) : Prop := 
  -- Some formal definition of correlation suitable for the context
  sorry

-- Theorem stating that relationships (1) and (3) have a correlation
theorem correlation_1_and_3 :
  has_correlation relationship1 relationship3 :=
sorry

end correlation_1_and_3_l166_166250


namespace arithmetic_seq_common_diff_l166_166212

theorem arithmetic_seq_common_diff (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 0 + a 2 = 10) 
  (h2 : a 3 + a 5 = 4)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = -1 := 
  sorry

end arithmetic_seq_common_diff_l166_166212


namespace num_zeros_of_f_l166_166997

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x - 2

theorem num_zeros_of_f : ∃! x₁ x₂ ∈ ℝ, x₁ ≠ x₂ ∧ f(x₁) = 0 ∧ f(x₂) = 0 :=
by
  sorry

end num_zeros_of_f_l166_166997


namespace hyperbola_equation_l166_166906

-- Conditions
def center_origin (P : ℝ × ℝ) : Prop := P = (0, 0)
def focus_at (F : ℝ × ℝ) : Prop := F = (0, Real.sqrt 3)
def vertex_distance (d : ℝ) : Prop := d = Real.sqrt 3 - 1

-- Statement
theorem hyperbola_equation
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (d : ℝ)
  (h_center : center_origin center)
  (h_focus : focus_at focus)
  (h_vert_dist : vertex_distance d) :
  y^2 - (x^2 / 2) = 1 := 
sorry

end hyperbola_equation_l166_166906


namespace arithmetic_sequence_product_l166_166834

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) (h1 : ∀ n m, n < m → b n < b m) 
(h2 : ∀ n, b (n + 1) - b n = d) (h3 : b 3 * b 4 = 18) : b 2 * b 5 = -80 :=
sorry

end arithmetic_sequence_product_l166_166834


namespace project_completion_days_l166_166124

-- A's work rate per day
def A_work_rate : ℚ := 1 / 20

-- B's work rate per day
def B_work_rate : ℚ := 1 / 30

-- Combined work rate per day
def combined_work_rate : ℚ := A_work_rate + B_work_rate

-- Work done by B alone in the last 5 days
def B_alone_work : ℚ := 5 * B_work_rate

-- Let variable x represent the number of days A and B work together
def x (x_days : ℚ) := x_days / combined_work_rate + B_alone_work = 1

theorem project_completion_days (x_days : ℚ) (total_days : ℚ) :
  A_work_rate = 1 / 20 → B_work_rate = 1 / 30 → combined_work_rate = 1 / 12 → x_days / 12 + 1 / 6 = 1 → x_days = 10 → total_days = x_days + 5 → total_days = 15 :=
by
  intros _ _ _ _ _ _
  sorry

end project_completion_days_l166_166124


namespace hyperbola_line_intersection_l166_166132

theorem hyperbola_line_intersection
  (A B m : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) (hm : m ≠ 0) :
  ∃ x y : ℝ, A^2 * x^2 - B^2 * y^2 = 1 ∧ Ax - By = m ∧ Bx + Ay ≠ 0 :=
by
  sorry

end hyperbola_line_intersection_l166_166132


namespace number_of_members_in_league_l166_166663

-- Define the costs of the items considering the conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 3
def shorts_cost : ℕ := sock_cost + 2

-- Define the total cost for one member
def total_cost_one_member : ℕ := 
  2 * (sock_cost + tshirt_cost + shorts_cost)

-- Given total expenditure
def total_expenditure : ℕ := 4860

-- Define the theorem to be proved
theorem number_of_members_in_league :
  total_expenditure / total_cost_one_member = 106 :=
by 
  sorry

end number_of_members_in_league_l166_166663


namespace total_ice_cream_amount_l166_166455

theorem total_ice_cream_amount (ice_cream_friday ice_cream_saturday : ℝ) 
  (h1 : ice_cream_friday = 3.25)
  (h2 : ice_cream_saturday = 0.25) : 
  ice_cream_friday + ice_cream_saturday = 3.50 :=
by
  rw [h1, h2]
  norm_num

end total_ice_cream_amount_l166_166455


namespace number_of_players_l166_166268

theorem number_of_players (x y z : ℕ) 
  (h1 : x + y + z = 10)
  (h2 : x * y + y * z + z * x = 31) : 
  (x = 2 ∧ y = 3 ∧ z = 5) ∨ (x = 2 ∧ y = 5 ∧ z = 3) ∨ (x = 3 ∧ y = 2 ∧ z = 5) ∨ 
  (x = 3 ∧ y = 5 ∧ z = 2) ∨ (x = 5 ∧ y = 2 ∧ z = 3) ∨ (x = 5 ∧ y = 3 ∧ z = 2) :=
sorry

end number_of_players_l166_166268


namespace top_weight_l166_166736

theorem top_weight (T : ℝ) : 
    (9 * 0.8 + 7 * T = 10.98) → T = 0.54 :=
by 
  intro h
  have H_sum := h
  simp only [mul_add, add_assoc, mul_assoc, mul_comm, add_comm, mul_comm 7] at H_sum
  sorry

end top_weight_l166_166736


namespace simplify_fraction_l166_166241

theorem simplify_fraction :
  5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l166_166241


namespace find_x_squared_minus_one_l166_166487

theorem find_x_squared_minus_one (x : ℕ) 
  (h : 2^x + 2^x + 2^x + 2^x = 256) : 
  x^2 - 1 = 35 :=
sorry

end find_x_squared_minus_one_l166_166487


namespace number_of_customers_before_lunch_rush_l166_166763

-- Defining the total number of customers during the lunch rush
def total_customers_during_lunch_rush : ℕ := 49 + 2

-- Defining the number of additional customers during the lunch rush
def additional_customers : ℕ := 12

-- Target statement to prove
theorem number_of_customers_before_lunch_rush : total_customers_during_lunch_rush - additional_customers = 39 :=
  by sorry

end number_of_customers_before_lunch_rush_l166_166763


namespace geometric_seq_common_ratio_l166_166480

theorem geometric_seq_common_ratio (a_n : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (hS3 : S 3 = a_n 1 * (1 - q ^ 3) / (1 - q))
  (hS2 : S 2 = a_n 1 * (1 - q ^ 2) / (1 - q))
  (h : S 3 + 3 * S 2 = 0) 
  (hq_not_one : q ≠ 1) :
  q = -2 :=
by sorry

end geometric_seq_common_ratio_l166_166480


namespace equation_of_l2_l166_166649

-- Define the initial line equation
def l1 (x : ℝ) : ℝ := -2 * x - 2

-- Define the transformed line equation after translation
def l2 (x : ℝ) : ℝ := l1 (x + 1) + 2

-- Statement to prove
theorem equation_of_l2 : ∀ x, l2 x = -2 * x - 2 := by
  sorry

end equation_of_l2_l166_166649


namespace sheena_sewing_weeks_l166_166860

theorem sheena_sewing_weeks (sew_time : ℕ) (bridesmaids : ℕ) (sewing_per_week : ℕ) 
    (h_sew_time : sew_time = 12) (h_bridesmaids : bridesmaids = 5) (h_sewing_per_week : sewing_per_week = 4) : 
    (bridesmaids * sew_time) / sewing_per_week = 15 := 
  by sorry

end sheena_sewing_weeks_l166_166860


namespace solve_f_l166_166116

open Nat

theorem solve_f (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) + f n = 2 * n + 3) : f 1993 = 1994 := by
  -- assumptions and required proof
  sorry

end solve_f_l166_166116


namespace not_equal_d_l166_166108

def frac_14_over_6 : ℚ := 14 / 6
def mixed_2_and_1_3rd : ℚ := 2 + 1 / 3
def mixed_neg_2_and_1_3rd : ℚ := -(2 + 1 / 3)
def mixed_3_and_1_9th : ℚ := 3 + 1 / 9
def mixed_2_and_4_12ths : ℚ := 2 + 4 / 12
def target_fraction : ℚ := 7 / 3

theorem not_equal_d : mixed_3_and_1_9th ≠ target_fraction :=
by sorry

end not_equal_d_l166_166108


namespace sum_of_digits_of_x_l166_166908

def two_digit_palindrome (x : ℕ) : Prop :=
  (10 ≤ x ∧ x ≤ 99) ∧ (x = (x % 10) * 10 + (x % 10))

def three_digit_palindrome (y : ℕ) : Prop :=
  (100 ≤ y ∧ y ≤ 999) ∧ (y = (y % 10) * 101 + (y % 10))

theorem sum_of_digits_of_x (x : ℕ) (h1 : two_digit_palindrome x) (h2 : three_digit_palindrome (x + 10)) : 
  (x % 10 + x / 10) = 10 :=
by
  sorry

end sum_of_digits_of_x_l166_166908


namespace third_term_is_five_l166_166481

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Suppose S_n = n^2 for n ∈ ℕ*
axiom H1 : ∀ n : ℕ, n > 0 → S n = n * n

-- The relationship a_n = S_n - S_(n-1) for n ≥ 2
axiom H2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)

-- Prove that the third term is 5
theorem third_term_is_five : a 3 = 5 := by
  sorry

end third_term_is_five_l166_166481


namespace sin_five_pi_over_six_l166_166153

theorem sin_five_pi_over_six : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
  sorry

end sin_five_pi_over_six_l166_166153


namespace eval_floor_neg_sqrt_l166_166609

theorem eval_floor_neg_sqrt : (Int.floor (-Real.sqrt (64 / 9)) = -3) := sorry

end eval_floor_neg_sqrt_l166_166609


namespace polynomial_satisfies_condition_l166_166317

theorem polynomial_satisfies_condition (P : Polynomial ℝ)
  (h : ∀ a b c : ℝ, ab + bc + ca = 0 → P(a - b) + P(b - c) + P(c - a) = 2 * P(a + b + c)) :
  ∃ α β : ℝ, P = Polynomial.C α * Polynomial.X ^ 2 + Polynomial.C β * Polynomial.X ^ 4 :=
by 
  sorry

end polynomial_satisfies_condition_l166_166317


namespace correct_product_l166_166232

theorem correct_product (a b c : ℕ) (ha : 10 * c + 1 = a) (hb : 10 * c + 7 = a) 
(hl : (10 * c + 1) * b = 255) (hw : (10 * c + 7 + 6) * b = 335) : 
  a * b = 285 := 
  sorry

end correct_product_l166_166232


namespace Ahmed_goat_count_l166_166592

theorem Ahmed_goat_count : 
  let A := 7 in
  let B := 2 * A + 5 in
  let C := B - 6 in
  C = 13 :=
by
  let A := 7
  let B := 2 * A + 5
  let C := B - 6
  show C = 13
  sorry

end Ahmed_goat_count_l166_166592


namespace jade_cal_difference_l166_166981

def Mabel_transactions : ℕ := 90

def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)

def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3

def Jade_transactions : ℕ := 85

theorem jade_cal_difference : Jade_transactions - Cal_transactions = 19 := by
  sorry

end jade_cal_difference_l166_166981


namespace odd_function_value_l166_166033

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the function definition

-- Prove that f(-1/2) = -1/2 given the conditions
theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = x) →
  f (-1/2) = -1/2 :=
by
  sorry

end odd_function_value_l166_166033


namespace broken_line_count_l166_166119

def num_right_moves : ℕ := 9
def num_up_moves : ℕ := 10
def total_moves : ℕ := num_right_moves + num_up_moves
def num_broken_lines : ℕ := Nat.choose total_moves num_right_moves

theorem broken_line_count : num_broken_lines = 92378 := by
  sorry

end broken_line_count_l166_166119


namespace density_function_Y_l166_166042

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-x^2 / 2)

theorem density_function_Y (y : ℝ) (hy : 0 < y) : 
  (∃ (g : ℝ → ℝ), (∀ y, g y = (1 / Real.sqrt (2 * Real.pi * y)) * Real.exp (- y / 2))) :=
sorry

end density_function_Y_l166_166042


namespace minimum_value_of_y_l166_166792

theorem minimum_value_of_y :
  ∀ (x : ℝ), x > 3 → let y := x + 1 / (x - 3) in y ≥ 5 :=
by
  sorry

end minimum_value_of_y_l166_166792


namespace field_trip_cost_l166_166441

def candy_bar_price : ℝ := 1.25
def candy_bars_sold : ℤ := 188
def money_from_grandma : ℝ := 250

theorem field_trip_cost : (candy_bars_sold * candy_bar_price + money_from_grandma) = 485 := 
by
  sorry

end field_trip_cost_l166_166441


namespace katy_read_books_l166_166828

theorem katy_read_books (juneBooks : ℕ) (julyBooks : ℕ) (augustBooks : ℕ)
  (H1 : juneBooks = 8)
  (H2 : julyBooks = 2 * juneBooks)
  (H3 : augustBooks = julyBooks - 3) :
  juneBooks + julyBooks + augustBooks = 37 := by
  -- Proof goes here
  sorry

end katy_read_books_l166_166828


namespace standard_equation_of_circle_l166_166549

theorem standard_equation_of_circle :
  (∃ a r, r^2 = (a + 1)^2 + (a - 1)^2 ∧ r^2 = (a - 1)^2 + (a - 3)^2 ∧ a = 1 ∧ r^2 = 4) →
  ∃ r, (x - 1)^2 + (y - 1)^2 = r^2 :=
by
  intro h
  sorry

end standard_equation_of_circle_l166_166549


namespace three_digit_number_l166_166427

theorem three_digit_number (x y z : ℕ) 
  (h1: z^2 = x * y)
  (h2: y = (x + z) / 6)
  (h3: x - z = 4) :
  100 * x + 10 * y + z = 824 := 
by sorry

end three_digit_number_l166_166427


namespace seventy_seventh_digit_is_three_l166_166056

-- Define the sequence of digits from the numbers 60 to 1 in decreasing order.
def sequence_of_digits : List Nat :=
  (List.range' 1 60).reverse.bind (fun n => n.digits 10)

-- Define a function to get the nth digit from the list.
def digit_at_position (n : Nat) : Option Nat :=
  sequence_of_digits.get? (n - 1)

-- The statement to prove
theorem seventy_seventh_digit_is_three : digit_at_position 77 = some 3 :=
sorry

end seventy_seventh_digit_is_three_l166_166056


namespace rectangular_garden_length_l166_166567

theorem rectangular_garden_length (P B L : ℕ) (h1 : P = 1800) (h2 : B = 400) (h3 : P = 2 * (L + B)) : L = 500 :=
sorry

end rectangular_garden_length_l166_166567


namespace seashells_given_to_Jessica_l166_166606

-- Define the initial number of seashells Dan had
def initialSeashells : ℕ := 56

-- Define the number of seashells Dan has left
def seashellsLeft : ℕ := 22

-- Define the number of seashells Dan gave to Jessica
def seashellsGiven : ℕ := initialSeashells - seashellsLeft

-- State the theorem to prove
theorem seashells_given_to_Jessica :
  seashellsGiven = 34 :=
by
  -- Begin the proof here
  sorry

end seashells_given_to_Jessica_l166_166606


namespace cost_of_fruits_l166_166488

-- Definitions based on the conditions
variables (x y z : ℝ)

-- Conditions
axiom h1 : 2 * x + y + 4 * z = 6
axiom h2 : 4 * x + 2 * y + 2 * z = 4

-- Question to prove
theorem cost_of_fruits : 4 * x + 2 * y + 5 * z = 8 :=
sorry

end cost_of_fruits_l166_166488


namespace calculate_fg_l166_166334

def f (x : ℝ) : ℝ := x - 4

def g (x : ℝ) : ℝ := x^2 + 5

theorem calculate_fg : f (g (-3)) = 10 := by
  sorry

end calculate_fg_l166_166334


namespace probability_higher_2012_l166_166358

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

noncomputable def passing_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  1 - ∑ i in finset.range (k), binomial_probability n i p

theorem probability_higher_2012 :
  passing_probability 40 6 0.25 > passing_probability 20 3 0.25 :=
sorry

end probability_higher_2012_l166_166358


namespace sum_powers_is_76_l166_166386

theorem sum_powers_is_76 (m n : ℕ) (h1 : m + n = 1) (h2 : m^2 + n^2 = 3)
                         (h3 : m^3 + n^3 = 4) (h4 : m^4 + n^4 = 7)
                         (h5 : m^5 + n^5 = 11) : m^9 + n^9 = 76 :=
sorry

end sum_powers_is_76_l166_166386


namespace max_ages_within_two_std_dev_l166_166570

def average_age : ℕ := 30
def std_dev : ℕ := 12
def lower_limit : ℕ := average_age - 2 * std_dev
def upper_limit : ℕ := average_age + 2 * std_dev
def max_different_ages : ℕ := upper_limit - lower_limit + 1

theorem max_ages_within_two_std_dev
  (avg : ℕ) (std : ℕ) (h_avg : avg = average_age) (h_std : std = std_dev)
  : max_different_ages = 49 :=
by
  sorry

end max_ages_within_two_std_dev_l166_166570


namespace two_bedroom_units_l166_166122

theorem two_bedroom_units {x y : ℕ} 
  (h1 : x + y = 12) 
  (h2 : 360 * x + 450 * y = 4950) : 
  y = 7 := 
by
  sorry

end two_bedroom_units_l166_166122


namespace sum_of_five_consecutive_integers_l166_166466

theorem sum_of_five_consecutive_integers (n : ℤ) :
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 5 * n :=
by
  sorry

end sum_of_five_consecutive_integers_l166_166466


namespace factor_of_polynomial_l166_166607

def polynomial (x : ℝ) : ℝ := x^4 - 4*x^2 + 16
def q1 (x : ℝ) : ℝ := x^2 + 4
def q2 (x : ℝ) : ℝ := x - 2
def q3 (x : ℝ) : ℝ := x^2 - 4*x + 4
def q4 (x : ℝ) : ℝ := x^2 + 4*x + 4

theorem factor_of_polynomial : (∃ (f g : ℝ → ℝ), polynomial x = f x * g x) ∧ (q4 = f ∨ q4 = g) := by sorry

end factor_of_polynomial_l166_166607


namespace Alex_meets_train_probability_l166_166140

noncomputable def probability_Alex_meets_train : ℚ := 11 / 72

theorem Alex_meets_train_probability :
  let time_range : set (ℚ × ℚ) := {xy | 0 ≤ xy.1 ∧ xy.1 ≤ 60 ∧ 0 ≤ xy.2 ∧ xy.2 ≤ 60}
  let shaded_region : set (ℚ × ℚ) := {xy | xy.2 - xy.1 ≤ 10 ∧ xy.2 - xy.1 ≥ -10}
  probability (shaded_region) / probability (time_range) = 11 / 72 := 
sorry

end Alex_meets_train_probability_l166_166140


namespace total_weight_of_pumpkins_l166_166843

def first_pumpkin_weight : ℝ := 12.6
def second_pumpkin_weight : ℝ := 23.4
def total_weight : ℝ := 36

theorem total_weight_of_pumpkins :
  first_pumpkin_weight + second_pumpkin_weight = total_weight :=
by
  sorry

end total_weight_of_pumpkins_l166_166843


namespace sum_inequality_l166_166794

open Real

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a * b / (a + b)) + (b * c / (b + c)) + (c * a / (c + a)) + 
             (1 / 2) * ((a * b / c) + (b * c / a) + (c * a / b)) :=
by
  sorry

end sum_inequality_l166_166794


namespace find_g3_l166_166251

-- Define a function g from ℝ to ℝ
variable (g : ℝ → ℝ)

-- Condition: ∀ x, g(3^x) + 2 * x * g(3^(-x)) = 3
axiom condition : ∀ x : ℝ, g (3^x) + 2 * x * g (3^(-x)) = 3

-- The theorem we need to prove
theorem find_g3 : g 3 = -3 := 
by 
  sorry

end find_g3_l166_166251


namespace value_of_z_l166_166544

theorem value_of_z :
  let mean_of_4_16_20 := (4 + 16 + 20) / 3
  let mean_of_8_z := (8 + z) / 2
  ∀ z : ℚ, mean_of_4_16_20 = mean_of_8_z → z = 56 / 3 := 
by
  intro z mean_eq
  sorry

end value_of_z_l166_166544


namespace calculation_result_l166_166921

theorem calculation_result :
  (2 : ℝ)⁻¹ - (1 / 2 : ℝ)^0 + (2 : ℝ)^2023 * (-0.5 : ℝ)^2023 = -3 / 2 := sorry

end calculation_result_l166_166921


namespace remy_gallons_l166_166668

noncomputable def gallons_used (R : ℝ) : ℝ :=
  let remy := 3 * R + 1
  let riley := (R + remy) - 2
  let ronan := riley / 2
  R + remy + riley + ronan

theorem remy_gallons : ∃ R : ℝ, gallons_used R = 60 ∧ (3 * R + 1) = 18.85 :=
by
  sorry

end remy_gallons_l166_166668


namespace problem_proof_l166_166036

theorem problem_proof (c d : ℝ) 
  (h1 : 5 + c = 6 - d) 
  (h2 : 6 + d = 9 + c) : 
  5 - c = 6 := 
sorry

end problem_proof_l166_166036


namespace simplify_fraction_l166_166689

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l166_166689


namespace mat_length_is_correct_l166_166296

noncomputable def mat_length (r : ℝ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / 5
  let side := 2 * r * Real.sin (θ / 2)
  let D := r * Real.cos (Real.pi / 5)
  let x := ((Real.sqrt (r^2 - ((w / 2) ^ 2))) - D + (w / 2))
  x

theorem mat_length_is_correct :
  mat_length 5 1 = 1.4 :=
by
  sorry

end mat_length_is_correct_l166_166296


namespace negation_proposition_l166_166023

theorem negation_proposition (a b c : ℝ) : 
  (¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3)) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) := 
by
  -- proof goes here
  sorry

end negation_proposition_l166_166023


namespace problem_solution_l166_166627

theorem problem_solution (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) 
  (h5 : x^2 + y^2 + z^2 + w^2 = 1) : 
  x^2 * y * z * w + x * y^2 * z * w + x * y * z^2 * w + x * y * z * w^2 ≤ 1 / 8 := 
by
  sorry

end problem_solution_l166_166627


namespace max_area_triangle_PJ1J2_l166_166833

noncomputable def triangle_PQR (PQ QR PR : ℝ) (angle_P angle_Q angle_R : ℝ) : Prop :=
  PQ = 20 ∧ QR = 21 ∧ PR = 29

noncomputable def max_area_PJ1J2 (PQ QR PR angle_P angle_Q angle_R : ℝ) (PJ1 PJ2 : ℝ) : ℝ :=
  PQ * PR * real.sin (angle_P / 2) * real.sin (angle_Q / 2) * real.sin (angle_R / 2)

theorem max_area_triangle_PJ1J2 (PQ QR PR angle_P angle_Q angle_R PJ1 PJ2 : ℝ) (h : triangle_PQR PQ QR PR angle_P angle_Q angle_R) :
  max_area_PJ1J2 PQ QR PR angle_P angle_Q angle_R PJ1 PJ2 = 20 * 29 * real.sin (angle_P / 2) * real.sin (angle_Q / 2) * real.sin (angle_R / 2) :=
sorry

end max_area_triangle_PJ1J2_l166_166833


namespace time_for_B_alone_to_complete_work_l166_166751

theorem time_for_B_alone_to_complete_work :
  (∃ (A B C : ℝ), A = 1 / 4 
  ∧ B + C = 1 / 3
  ∧ A + C = 1 / 2)
  → 1 / B = 12 :=
by
  sorry

end time_for_B_alone_to_complete_work_l166_166751


namespace line_hyperbola_unique_intersection_l166_166411

theorem line_hyperbola_unique_intersection (k : ℝ) :
  (∃ (x y : ℝ), k * x - y - 2 * k = 0 ∧ x^2 - y^2 = 2 ∧ 
  ∀ y₁, y₁ ≠ y → k * x - y₁ - 2 * k ≠ 0 ∧ x^2 - y₁^2 ≠ 2) ↔ (k = 1 ∨ k = -1) :=
by
  sorry

end line_hyperbola_unique_intersection_l166_166411


namespace sum_shade_length_l166_166213

-- Define the arithmetic sequence and the given conditions
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (is_arithmetic : ∀ n, a (n + 1) = a n + d)

-- Define the shadow lengths for each term using the arithmetic progression properties
def shade_length_seq (seq : ArithmeticSequence) : ℕ → ℝ := seq.a

variables (seq : ArithmeticSequence)

-- Given conditions
axiom sum_condition_1 : seq.a 1 + seq.a 4 + seq.a 7 = 31.5
axiom sum_condition_2 : seq.a 2 + seq.a 5 + seq.a 8 = 28.5

-- Question to prove
theorem sum_shade_length : seq.a 3 + seq.a 6 + seq.a 9 = 25.5 :=
by
  -- proof to be filled in later
  sorry

end sum_shade_length_l166_166213


namespace people_ratio_l166_166204

theorem people_ratio (pounds_coal : ℕ) (days1 : ℕ) (people1 : ℕ) (pounds_goal : ℕ) (days2 : ℕ) :
  pounds_coal = 10000 → days1 = 10 → people1 = 10 → pounds_goal = 40000 → days2 = 80 →
  (people1 * pounds_goal * days1) / (pounds_coal * days2) = 1 / 2 :=
by
  sorry

end people_ratio_l166_166204


namespace M_inter_N_is_01_l166_166838

variable (x : ℝ)

def M := { x : ℝ | Real.log (1 - x) < 0 }
def N := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }

theorem M_inter_N_is_01 : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  -- Proof will go here
  sorry

end M_inter_N_is_01_l166_166838


namespace sum_first_10_mod_8_is_7_l166_166436

-- Define the sum of the first 10 positive integers
def sum_first_10 : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10

-- Define the divisor
def divisor : ℕ := 8

-- Prove that the remainder of the sum of the first 10 positive integers divided by 8 is 7
theorem sum_first_10_mod_8_is_7 : sum_first_10 % divisor = 7 :=
by
  sorry

end sum_first_10_mod_8_is_7_l166_166436


namespace bees_leg_count_l166_166120

-- Define the number of legs per bee
def legsPerBee : Nat := 6

-- Define the number of bees
def numberOfBees : Nat := 8

-- Calculate the total number of legs for 8 bees
def totalLegsForEightBees : Nat := 48

-- The theorem statement
theorem bees_leg_count : (legsPerBee * numberOfBees) = totalLegsForEightBees := 
by
  -- Skipping the proof by using sorry
  sorry

end bees_leg_count_l166_166120


namespace rational_abs_neg_l166_166499

theorem rational_abs_neg (a : ℚ) (h : abs a = -a) : a ≤ 0 :=
by 
  sorry

end rational_abs_neg_l166_166499


namespace gcd_102_238_l166_166724

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

theorem gcd_102_238 : gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l166_166724


namespace min_value_of_y_l166_166791

theorem min_value_of_y (x : ℝ) (h : x > 3) : y = x + 1/(x-3) → y ≥ 5 :=
sorry

end min_value_of_y_l166_166791


namespace reciprocal_of_neg_three_l166_166699

theorem reciprocal_of_neg_three : ∃ (x : ℚ), (-3 * x = 1) ∧ (x = -1 / 3) :=
by
  use (-1 / 3)
  split
  . rw [mul_comm]
    norm_num 
  . norm_num

end reciprocal_of_neg_three_l166_166699


namespace ratio_P_to_A_l166_166384

variable (M P A : ℕ) -- Define variables for Matthew, Patrick, and Alvin's egg rolls

theorem ratio_P_to_A (hM : M = 6) (hM_to_P : M = 3 * P) (hA : A = 4) : P / A = 1 / 2 := by
  sorry

end ratio_P_to_A_l166_166384


namespace icosahedron_path_count_l166_166465

noncomputable def icosahedron_paths : ℕ := 
  sorry

theorem icosahedron_path_count : icosahedron_paths = 45 :=
  sorry

end icosahedron_path_count_l166_166465


namespace cardinals_to_bluebirds_ratio_l166_166533

-- Define the problem conditions
variables {C B : ℕ}

-- The condition that there are 2 swallows, which is half the number of bluebirds
axiom h1 : 2 = 1 / 2 * B

-- The total number of birds is 18
axiom h2 : C + B + 2 = 18

-- The conclusion we aim to prove
theorem cardinals_to_bluebirds_ratio : C = 12 ∧ B = 4 ∧ C / B = 3 :=
by 
  -- Proof omitted
  sorry

end cardinals_to_bluebirds_ratio_l166_166533


namespace complex_number_quadrant_l166_166215

def i_squared : ℂ := -1

def z (i : ℂ) : ℂ := (-2 + i) * i^5

def in_quadrant_III (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_number_quadrant 
  (i : ℂ) (hi : i^2 = -1) (z_val : z i = (-2 + i) * i^5) :
  in_quadrant_III (z i) :=
sorry

end complex_number_quadrant_l166_166215


namespace anchuria_cert_prob_higher_2012_l166_166359

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  binomial_coefficient n k * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def cumulative_binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Finset.range (k + 1)).sum (λ i, binomial_probability n i p)

theorem anchuria_cert_prob_higher_2012 :
  let p := 0.25
  let q := 0.75
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  P_pass_2011 = 1 - cumulative_binomial_probability n2011 (k2011 - 1) p
  P_pass_2012 = 1 - cumulative_binomial_probability n2012 (k2012 - 1) p
  in P_pass_2012 > P_pass_2011 :=
by
  let p : ℝ := 0.25
  let q : ℝ := 0.75
  let n2011 : ℕ := 20
  let k2011 : ℕ := 3
  let n2012 : ℕ := 40
  let k2012 : ℕ := 6
  let P_fewer_than_3_2011 := cumulative_binomial_probability n2011 (k2011 - 1) p
  let P_fewer_than_6_2012 := cumulative_binomial_probability n2012 (k2012 - 1) p
  let P_pass_2011 := 1 - P_fewer_than_3_2011
  let P_pass_2012 := 1 - P_fewer_than_6_2012
  show P_pass_2012 > P_pass_2011 from sorry

end anchuria_cert_prob_higher_2012_l166_166359


namespace quadratic_expression_sum_l166_166613

theorem quadratic_expression_sum :
  ∃ a h k : ℝ, (∀ x, 4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) ∧ (a + h + k = 2) :=
sorry

end quadratic_expression_sum_l166_166613


namespace prob_white_ball_is_0_25_l166_166646

-- Let's define the conditions and the statement for the proof
variable (P_red P_white P_yellow : ℝ)

-- The given conditions 
def prob_red_or_white : Prop := P_red + P_white = 0.65
def prob_yellow_or_white : Prop := P_yellow + P_white = 0.6

-- The statement we want to prove
theorem prob_white_ball_is_0_25 (h1 : prob_red_or_white P_red P_white)
                               (h2 : prob_yellow_or_white P_yellow P_white) :
  P_white = 0.25 :=
sorry

end prob_white_ball_is_0_25_l166_166646


namespace max_largest_integer_l166_166206

theorem max_largest_integer (A B C D E : ℕ) 
  (h1 : A ≤ B) 
  (h2 : B ≤ C) 
  (h3 : C ≤ D) 
  (h4 : D ≤ E)
  (h5 : (A + B + C + D + E) / 5 = 60) 
  (h6 : E - A = 10) : 
  E ≤ 290 :=
sorry

end max_largest_integer_l166_166206


namespace max_path_length_CQ_D_l166_166310

noncomputable def maxCQDPathLength (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) : ℝ :=
  let r := dAB / 2
  let dCD := dAB - dAC - dBD
  2 * Real.sqrt (r^2 - (dCD / 2)^2)

theorem max_path_length_CQ_D 
  (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) (r := dAB / 2) (dCD := dAB - dAC - dBD) :
  dAB = 16 ∧ dAC = 3 ∧ dBD = 5 ∧ r = 8 ∧ dCD = 8
  → maxCQDPathLength 16 3 5 = 8 * Real.sqrt 3 :=
by
  intros h
  cases h
  sorry

end max_path_length_CQ_D_l166_166310


namespace constant_term_binomial_expansion_l166_166367

theorem constant_term_binomial_expansion : 
  let r := 3
  let general_term (r : ℕ) (x : ℝ) := (choose 5 r) * ((sqrt x / 2) ^ (5 - r)) * ((-1 / cbrt x) ^ r)
  general_term 3 x = -5 / 2 := 
by {
  sorry
}

end constant_term_binomial_expansion_l166_166367


namespace problem_proof_l166_166737

theorem problem_proof:
  (∃ n : ℕ, 25 = n ^ 2) ∧
  (Prime 31) ∧
  (¬ ∀ p : ℕ, Prime p → p >= 3 → p = 2) ∧
  (∃ m : ℕ, 8 = m ^ 3) ∧
  (∃ a b : ℕ, Prime a ∧ Prime b ∧ 15 = a * b) :=
by
  sorry

end problem_proof_l166_166737


namespace coefficient_B_is_1_l166_166040

-- Definitions based on the conditions
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

-- Given conditions
def condition1 (A B C D : ℝ) := g A B C D (-2) = 0 
def condition2 (A B C D : ℝ) := g A B C D 0 = -1
def condition3 (A B C D : ℝ) := g A B C D 2 = 0

-- The main theorem to prove
theorem coefficient_B_is_1 (A B C D : ℝ) 
  (h1 : condition1 A B C D) 
  (h2 : condition2 A B C D) 
  (h3 : condition3 A B C D) : 
  B = 1 :=
sorry

end coefficient_B_is_1_l166_166040


namespace grade_point_average_one_third_l166_166694

theorem grade_point_average_one_third :
  ∃ (x : ℝ), 55 = (1/3) * x + (2/3) * 60 ∧ x = 45 :=
by
  sorry

end grade_point_average_one_third_l166_166694


namespace Eugene_buys_two_pairs_of_shoes_l166_166647

theorem Eugene_buys_two_pairs_of_shoes :
  let tshirt_price : ℕ := 20
  let pants_price : ℕ := 80
  let shoes_price : ℕ := 150
  let discount_rate : ℕ := 10
  let discounted_price (price : ℕ) := price - (price * discount_rate / 100)
  let total_price (count1 count2 count3 : ℕ) (price1 price2 price3 : ℕ) :=
    (count1 * price1) + (count2 * price2) + (count3 * price3)
  let total_amount_paid : ℕ := 558
  let tshirts_bought : ℕ := 4
  let pants_bought : ℕ := 3
  let amount_left := total_amount_paid - discounted_price (tshirts_bought * tshirt_price + pants_bought * pants_price)
  let shoes_bought := amount_left / discounted_price shoes_price
  shoes_bought = 2 := 
sorry

end Eugene_buys_two_pairs_of_shoes_l166_166647


namespace sin_45_degrees_l166_166164

noncomputable def Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem sin_45_degrees : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_degrees_l166_166164


namespace circle_tangent_line_l166_166496

theorem circle_tangent_line {m : ℝ} : 
  (3 * (0 : ℝ) - 4 * (1 : ℝ) - 6 = 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 2 * y + m = 0) → 
  m = -3 := by
  sorry

end circle_tangent_line_l166_166496


namespace problem_statement_l166_166045

-- Definitions of sets S and P
def S : Set ℝ := {x | x^2 - 3 * x - 10 < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2 * a + 15}

-- Proof statement
theorem problem_statement (a : ℝ) : 
  (S = {x | -2 < x ∧ x < 5}) ∧ (S ⊆ P a → a ∈ Set.Icc (-5 : ℝ) (-3 : ℝ)) :=
by
  sorry

end problem_statement_l166_166045


namespace julios_grape_soda_l166_166518

variable (a b c d e f g : ℕ)
variable (ha : a = 4)
variable (hc : c = 1)
variable (hd : d = 3)
variable (he : e = 2)
variable (hf : f = 14)
variable (hg : g = 7)

theorem julios_grape_soda : 
  let julios_soda := a * e + b * e
  let mateos_soda := (c + d) * e
  julios_soda = mateos_soda + f
  → b = g := by
  sorry

end julios_grape_soda_l166_166518


namespace solution_set_l166_166328

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

theorem solution_set:
  ∀ x : ℝ, x > -1 ∧ x < 1/3 → f (2*x + 1) < f x := 
by
  sorry

end solution_set_l166_166328


namespace inequality_solution_l166_166246

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 3 * x + 10) ≥ 0 ↔ x ≥ -2 := sorry

end inequality_solution_l166_166246


namespace range_of_m_l166_166955

theorem range_of_m (m : ℝ) : (∀ (x : ℝ), |3 - x| + |5 + x| > m) → m < 8 :=
sorry

end range_of_m_l166_166955


namespace max_value_fraction_l166_166044

theorem max_value_fraction (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ max_val, max_val = 7 / 5 ∧ ∀ (x y : ℝ), 
    (x + y - 2 ≥ 0) → (y - x - 1 ≤ 0) → (x ≤ 1) → (x + 2*y) / (2*x + y) ≤ max_val :=
sorry

end max_value_fraction_l166_166044


namespace trip_correct_graph_l166_166664

-- Define a structure representing the trip
structure Trip :=
  (initial_city_traffic_duration : ℕ)
  (highway_duration_to_mall : ℕ)
  (shopping_duration : ℕ)
  (highway_duration_from_mall : ℕ)
  (return_city_traffic_duration : ℕ)

-- Define the conditions about the trip
def conditions (t : Trip) : Prop :=
  t.shopping_duration = 1 ∧ -- Shopping for one hour
  t.initial_city_traffic_duration < t.highway_duration_to_mall ∧ -- Travel more rapidly on the highway
  t.return_city_traffic_duration < t.highway_duration_from_mall -- Return more rapidly on the highway

-- Define the graph representation of the trip
inductive Graph
| A | B | C | D | E

-- Define the property that graph B correctly represents the trip
def correct_graph (t : Trip) (g : Graph) : Prop :=
  g = Graph.B

-- The theorem stating that given the conditions, the correct graph is B
theorem trip_correct_graph (t : Trip) (h : conditions t) : correct_graph t Graph.B :=
by
  sorry

end trip_correct_graph_l166_166664


namespace forester_planted_total_trees_l166_166582

theorem forester_planted_total_trees :
  let initial_trees := 30 in
  let monday_trees := 3 * initial_trees in
  let new_trees_monday := monday_trees - initial_trees in
  let tuesday_trees := (1 / 3) * new_trees_monday in
  new_trees_monday + tuesday_trees = 80 :=
by
  repeat sorry

end forester_planted_total_trees_l166_166582


namespace fraction_sum_l166_166603

theorem fraction_sum : (3 / 8) + (9 / 12) + (5 / 6) = 47 / 24 := by
  sorry

end fraction_sum_l166_166603


namespace boys_girls_difference_l166_166097

/--
If there are 550 students in a class and the ratio of boys to girls is 7:4, 
prove that the number of boys exceeds the number of girls by 150.
-/
theorem boys_girls_difference : 
  ∀ (students boys_ratio girls_ratio : ℕ),
  students = 550 →
  boys_ratio = 7 →
  girls_ratio = 4 →
  (students * boys_ratio) % (boys_ratio + girls_ratio) = 0 ∧
  (students * girls_ratio) % (boys_ratio + girls_ratio) = 0 →
  (students * boys_ratio - students * girls_ratio) / (boys_ratio + girls_ratio) = 150 :=
by
  intros students boys_ratio girls_ratio h_students h_boys_ratio h_girls_ratio h_divisibility
  -- The detailed proof would follow here, but we add 'sorry' to bypass it.
  sorry

end boys_girls_difference_l166_166097


namespace reciprocal_of_neg3_l166_166712

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l166_166712


namespace arithmetic_sequence_geometric_subsequence_l166_166032

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℕ)
  (h1 : ∀ n, a (n + 1) = a n + 1)
  (h2 : (a 3)^2 = a 1 * a 7) :
  a 5 = 6 :=
sorry

end arithmetic_sequence_geometric_subsequence_l166_166032


namespace count_solutions_sin_equation_l166_166174

theorem count_solutions_sin_equation : 
  ∃ S : Finset ℝ, (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 3 * (Real.sin x)^4 - 7 * (Real.sin x)^3 + 5 * (Real.sin x)^2 - Real.sin x = 0) ∧ S.card = 4 :=
by
  sorry

end count_solutions_sin_equation_l166_166174


namespace number_of_walls_l166_166450

theorem number_of_walls (bricks_per_row rows_per_wall total_bricks : Nat) :
  bricks_per_row = 30 → 
  rows_per_wall = 50 → 
  total_bricks = 3000 → 
  total_bricks / (bricks_per_row * rows_per_wall) = 2 := 
by
  intros h1 h2 h3
  sorry

end number_of_walls_l166_166450


namespace deborah_total_cost_l166_166170

-- Standard postage per letter
def stdPostage : ℝ := 1.08

-- Additional charge for international shipping per letter
def intlAdditional : ℝ := 0.14

-- Number of domestic and international letters
def numDomestic : ℕ := 2
def numInternational : ℕ := 2

-- Expected total cost for four letters
def expectedTotalCost : ℝ := 4.60

theorem deborah_total_cost :
  (numDomestic * stdPostage) + (numInternational * (stdPostage + intlAdditional)) = expectedTotalCost :=
by
  -- proof skipped
  sorry

end deborah_total_cost_l166_166170


namespace total_cost_fencing_l166_166408

/-
  Given conditions:
  1. Length of the plot (l) = 55 meters
  2. Length is 10 meters more than breadth (b): l = b + 10
  3. Cost of fencing per meter (cost_per_meter) = 26.50
  
  Prove that the total cost of fencing the plot is 5300 currency units.
-/
def length : ℕ := 55
def breadth : ℕ := length - 10
def cost_per_meter : ℝ := 26.50
def perimeter : ℕ := 2 * (length + breadth)
def total_cost : ℝ := cost_per_meter * perimeter

theorem total_cost_fencing : total_cost = 5300 := by
  sorry

end total_cost_fencing_l166_166408


namespace brock_peanuts_ratio_l166_166881

theorem brock_peanuts_ratio (initial : ℕ) (bonita : ℕ) (remaining : ℕ) (brock : ℕ)
  (h1 : initial = 148) (h2 : bonita = 29) (h3 : remaining = 82) (h4 : brock = 37)
  (h5 : initial - remaining = bonita + brock) :
  (brock : ℚ) / initial = 1 / 4 :=
by {
  sorry
}

end brock_peanuts_ratio_l166_166881


namespace total_octopus_legs_l166_166464

-- Define the number of octopuses Carson saw
def num_octopuses : ℕ := 5

-- Define the number of legs per octopus
def legs_per_octopus : ℕ := 8

-- Define or state the theorem for total number of legs
theorem total_octopus_legs : num_octopuses * legs_per_octopus = 40 := by
  sorry

end total_octopus_legs_l166_166464


namespace perimeter_of_shaded_region_correct_l166_166216

noncomputable def perimeter_of_shaded_region : ℝ :=
  let r := 7
  let perimeter := 2 * r + (3 / 4) * (2 * Real.pi * r)
  perimeter

theorem perimeter_of_shaded_region_correct :
  perimeter_of_shaded_region = 14 + 10.5 * Real.pi :=
by
  sorry

end perimeter_of_shaded_region_correct_l166_166216


namespace gcd_102_238_is_34_l166_166728

noncomputable def gcd_102_238 : ℕ :=
  Nat.gcd 102 238

theorem gcd_102_238_is_34 : gcd_102_238 = 34 := by
  -- Conditions based on the Euclidean algorithm
  have h1 : 238 = 2 * 102 + 34 := by norm_num
  have h2 : 102 = 3 * 34 := by norm_num
  have h3 : Nat.gcd 102 34 = 34 := by
    rw [Nat.gcd, Nat.gcd_rec]
    exact Nat.gcd_eq_left h2

  -- Conclusion
  show gcd_102_238 = 34 from
    calc gcd_102_238 = Nat.gcd 102 238 : rfl
                  ... = Nat.gcd 34 102 : Nat.gcd_comm 102 34
                  ... = Nat.gcd 34 (102 % 34) : by rw [Nat.gcd_rec]
                  ... = Nat.gcd 34 34 : by rw [Nat.mod_eq_of_lt (by norm_num : 34 < 102)]
                  ... = 34 : Nat.gcd_self 34

end gcd_102_238_is_34_l166_166728


namespace interval_of_x₀_l166_166197

-- Definition of the problem
variable (x₀ : ℝ)

-- Conditions
def condition_1 := x₀ > 0 ∧ x₀ < Real.pi
def condition_2 := Real.sin x₀ + Real.cos x₀ = 2 / 3

-- Proof problem statement
theorem interval_of_x₀ 
  (h1 : condition_1 x₀)
  (h2 : condition_2 x₀) : 
  x₀ > 7 * Real.pi / 12 ∧ x₀ < 3 * Real.pi / 4 := 
sorry

end interval_of_x₀_l166_166197


namespace max_ab_l166_166286

theorem max_ab (a b c : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : 3 * a + 2 * b = 2) :
  ab ≤ 1 / 6 :=
sorry

end max_ab_l166_166286


namespace max_number_of_band_members_l166_166295

-- Conditions definitions
def num_band_members (r x : ℕ) : ℕ := r * x + 3

def num_band_members_new (r x : ℕ) : ℕ := (r - 1) * (x + 2)

-- The main statement
theorem max_number_of_band_members :
  ∃ (r x : ℕ), num_band_members r x = 231 ∧ num_band_members_new r x = 231 
  ∧ ∀ (r' x' : ℕ), (num_band_members r' x' < 120 ∧ num_band_members_new r' x' = num_band_members r' x') → (num_band_members r' x' ≤ 231) :=
sorry

end max_number_of_band_members_l166_166295


namespace boys_from_school_A_study_science_l166_166819

theorem boys_from_school_A_study_science (total_boys school_A_percent non_science_boys school_A_boys study_science_boys: ℕ) 
(h1 : total_boys = 300)
(h2 : school_A_percent = 20)
(h3 : non_science_boys = 42)
(h4 : school_A_boys = (school_A_percent * total_boys) / 100)
(h5 : study_science_boys = school_A_boys - non_science_boys) :
(study_science_boys * 100 / school_A_boys) = 30 :=
by
  sorry

end boys_from_school_A_study_science_l166_166819


namespace cost_of_paving_is_correct_l166_166543

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_metre : ℝ := 400
def area_of_rectangle (l: ℝ) (w: ℝ) : ℝ := l * w
def cost_of_paving_floor (area: ℝ) (rate: ℝ) : ℝ := area * rate

theorem cost_of_paving_is_correct
  (h_length: length = 5.5)
  (h_width: width = 3.75)
  (h_rate: rate_per_sq_metre = 400):
  cost_of_paving_floor (area_of_rectangle length width) rate_per_sq_metre = 8250 :=
  by {
    sorry
  }

end cost_of_paving_is_correct_l166_166543


namespace problem_solution_l166_166745

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * cos (2 * x - π / 4)

theorem problem_solution :
  (∀ k : ℤ, ∀ x ∈ Icc (k * π - 3 * π / 8) (k * π + π / 8), 0 ≤ (2 * x - π / 4)) ∧
  (∀ x ∈ Icc (-π / 8) (π / 2), 
    (f x = -1 ↔ x = π / 2) ∧
    (f x = sqrt 2 ↔ x = π / 8)) :=
by
  sorry

end problem_solution_l166_166745


namespace rectangle_perimeter_l166_166443

theorem rectangle_perimeter
  (L W : ℕ)
  (h1 : L * W = 360)
  (h2 : (L + 10) * (W - 6) = 360) :
  2 * L + 2 * W = 76 := 
sorry

end rectangle_perimeter_l166_166443


namespace probability_both_groups_stop_same_round_l166_166141

noncomputable def probability_same_round : ℚ :=
  let probability_fair_coin_stop (n : ℕ) : ℚ := (1/2)^n
  let probability_biased_coin_stop (n : ℕ) : ℚ := (2/3)^(n-1) * (1/3)
  let probability_fair_coin_group_stop (n : ℕ) : ℚ := (probability_fair_coin_stop n)^3
  let probability_biased_coin_group_stop (n : ℕ) : ℚ := (probability_biased_coin_stop n)^3
  let combined_round_probability (n : ℕ) : ℚ := 
    probability_fair_coin_group_stop n * probability_biased_coin_group_stop n
  let total_probability : ℚ := ∑' n, combined_round_probability n
  total_probability

theorem probability_both_groups_stop_same_round :
  probability_same_round = 1 / 702 := by sorry

end probability_both_groups_stop_same_round_l166_166141


namespace fraction_zero_implies_x_eq_one_l166_166817

theorem fraction_zero_implies_x_eq_one (x : ℝ) (h : (x - 1) / (x + 1) = 0) : x = 1 :=
sorry

end fraction_zero_implies_x_eq_one_l166_166817


namespace prepaid_card_cost_correct_l166_166222

noncomputable def prepaid_phone_card_cost
    (cost_per_minute : ℝ) (call_minutes : ℝ) (remaining_credit : ℝ) : ℝ :=
  remaining_credit + (call_minutes * cost_per_minute)

theorem prepaid_card_cost_correct :
  let cost_per_minute := 0.16
  let call_minutes := 22
  let remaining_credit := 26.48
  prepaid_phone_card_cost cost_per_minute call_minutes remaining_credit = 30.00 := by
  sorry

end prepaid_card_cost_correct_l166_166222


namespace smaller_circle_radius_l166_166065

-- Given conditions
def larger_circle_radius : ℝ := 10
def number_of_smaller_circles : ℕ := 7

-- The goal
theorem smaller_circle_radius :
  ∃ r : ℝ, (∃ D : ℝ, D = 2 * larger_circle_radius ∧ D = 4 * r) ∧ r = 2.5 :=
by
  sorry

end smaller_circle_radius_l166_166065


namespace greatest_three_digit_number_l166_166278

theorem greatest_three_digit_number (n : ℕ) :
  (n % 8 = 2) ∧ (n % 7 = 4) ∧ (100 ≤ n ∧ n ≤ 999) → n = 970 :=
begin
  sorry
end

end greatest_three_digit_number_l166_166278


namespace sector_area_l166_166086

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 120) : 
  (theta / 360) * π * r^2 = 3 * π :=
by 
  sorry

end sector_area_l166_166086


namespace calculate_expression_l166_166920

variables (a b : ℝ) -- declaring variables a and b to be real numbers

theorem calculate_expression :
  (-a * b^2) ^ 3 + (a * b^2) * (a * b) ^ 2 * (-2 * b) ^ 2 = 3 * a^3 * b^6 :=
by
  sorry

end calculate_expression_l166_166920


namespace base_case_inequality_induction_inequality_l166_166424

theorem base_case_inequality : 2^5 > 5^2 + 1 := by
  -- Proof not required
  sorry

theorem induction_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  -- Proof not required
  sorry

end base_case_inequality_induction_inequality_l166_166424


namespace chord_intersection_l166_166422

theorem chord_intersection {AP BP CP DP : ℝ} (hAP : AP = 2) (hBP : BP = 6) (hCP_DP : ∃ k : ℝ, CP = k ∧ DP = 3 * k) :
  DP = 6 :=
by sorry

end chord_intersection_l166_166422


namespace simplify_fraction_l166_166243

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l166_166243


namespace largest_A_smallest_A_l166_166767

noncomputable def is_coprime_with_12 (n : Nat) : Prop :=
  Nat.gcd n 12 = 1

noncomputable def rotated_number (n : Nat) : Option Nat :=
  if n < 10^7 then none else
  let b := n % 10
  let k := n / 10
  some (b * 10^7 + k)

noncomputable def satisfies_conditions (B : Nat) : Prop :=
  B > 44444444 ∧ is_coprime_with_12 B

theorem largest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 99999998 :=
sorry

theorem smallest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 14444446 :=
sorry

end largest_A_smallest_A_l166_166767


namespace one_meter_to_leaps_l166_166536

theorem one_meter_to_leaps 
  (x y z w u v : ℕ)
  (h1 : x * leaps = y * strides) 
  (h2 : z * bounds = w * leaps) 
  (h3 : u * bounds = v * meters) :
  1 * meters = (uw / vz) * leaps :=
sorry

end one_meter_to_leaps_l166_166536


namespace original_class_strength_l166_166113

theorem original_class_strength (x : ℕ) 
    (avg_original : ℕ)
    (num_new : ℕ) 
    (avg_new : ℕ) 
    (decrease : ℕ)
    (h1 : avg_original = 40)
    (h2 : num_new = 17)
    (h3 : avg_new = 32)
    (h4 : decrease = 4)
    (h5 : (40 * x + 17 * avg_new) = (x + num_new) * (40 - decrease))
    : x = 17 := 
by {
  sorry
}

end original_class_strength_l166_166113


namespace banana_distribution_correct_l166_166426

noncomputable def proof_problem : Prop :=
  let bananas := 40
  let marbles := 4
  let boys := 18
  let girls := 12
  let total_friends := 30
  let bananas_for_boys := (3/8 : ℝ) * bananas
  let bananas_for_girls := (1/4 : ℝ) * bananas
  let bananas_left := bananas - (bananas_for_boys + bananas_for_girls)
  let bananas_per_marble := bananas_left / marbles
  bananas_for_boys = 15 ∧ bananas_for_girls = 10 ∧ bananas_per_marble = 3.75

theorem banana_distribution_correct : proof_problem :=
by
  -- Proof is omitted
  sorry

end banana_distribution_correct_l166_166426


namespace candy_left_l166_166722

theorem candy_left (total_candy : ℕ) (ate_each : ℕ) : total_candy = 68 → ate_each = 4 → total_candy - 2 * ate_each = 60 :=
by
  intros h1 h2
  rw [h1, h2]
  dsimp
  norm_num
  done

end candy_left_l166_166722


namespace radius_of_inscribed_circle_XYZ_l166_166433

noncomputable def radius_of_inscribed_circle (XY XZ YZ : ℝ) : ℝ :=
  let s := (XY + XZ + YZ) / 2
  let area := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  let r := area / s
  r

theorem radius_of_inscribed_circle_XYZ :
  radius_of_inscribed_circle 26 15 17 = 2 * Real.sqrt 42 / 29 :=
by
  sorry

end radius_of_inscribed_circle_XYZ_l166_166433


namespace find_k_l166_166352

theorem find_k (k : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, 3 * x - k * y + c = 0) ∧ (∀ x y : ℝ, k * x + y + 1 = 0 → 3 * k + (-k) = 0) → k = 0 :=
by
  sorry

end find_k_l166_166352


namespace candies_count_l166_166666

variable (m_and_m : Nat) (starbursts : Nat)
variable (ratio_m_and_m_to_starbursts : Nat → Nat → Prop)

-- Definition of the ratio condition
def ratio_condition : Prop :=
  ∃ (k : Nat), (m_and_m = 7 * k) ∧ (starbursts = 4 * k)

-- The main theorem to prove
theorem candies_count (h : m_and_m = 56) (r : ratio_condition m_and_m starbursts) : starbursts = 32 :=
  by
  sorry

end candies_count_l166_166666


namespace remainder_when_divided_by_5_l166_166755

theorem remainder_when_divided_by_5 
  (k : ℕ)
  (h1 : k % 6 = 5)
  (h2 : k < 42)
  (h3 : k % 7 = 3) : 
  k % 5 = 2 := 
by 
  sorry

end remainder_when_divided_by_5_l166_166755


namespace part1_part2_l166_166942

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k)
noncomputable def f_prime (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k) + Real.exp x / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f_prime x k - 2 * (f x k + Real.exp x)
noncomputable def phi (x : ℝ) : ℝ := Real.exp x / x

theorem part1 (h : f_prime 1 k = 0) : k = -1 := sorry

theorem part2 (t : ℝ) (h_g_le_phi : ∀ x > 0, g x (-1) ≤ t * phi x) : t ≥ 1 + 1 / Real.exp 2 := sorry

end part1_part2_l166_166942


namespace Taran_original_number_is_12_l166_166084

open Nat

theorem Taran_original_number_is_12 (x : ℕ)
  (h1 : (5 * x) + 5 - 5 = 73 ∨ (5 * x) + 5 - 6 = 73 ∨ (5 * x) + 6 - 5 = 73 ∨ (5 * x) + 6 - 6 = 73 ∨ 
       (6 * x) + 5 - 5 = 73 ∨ (6 * x) + 5 - 6 = 73 ∨ (6 * x) + 6 - 5 = 73 ∨ (6 * x) + 6 - 6 = 73) : x = 12 := by
  sorry

end Taran_original_number_is_12_l166_166084


namespace range_of_a_l166_166941

noncomputable def f (x a : ℝ) := 2^(2*x) - a * 2^x + 4

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a ≤ 4 :=
by
  sorry

end range_of_a_l166_166941


namespace simplify_expression_l166_166671

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l166_166671


namespace shipping_cost_l166_166234

def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def cost_per_crate : ℝ := 1.5

/-- Lizzy's total shipping cost for 540 pounds of fish packed in 30-pound crates at $1.5 per crate is $27. -/
theorem shipping_cost : (total_weight / weight_per_crate) * cost_per_crate = 27 := by
  sorry

end shipping_cost_l166_166234


namespace four_digit_not_multiples_of_4_or_9_l166_166049

theorem four_digit_not_multiples_of_4_or_9 (h1 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 4 ∣ n ↔ (250 ≤ n / 4 ∧ n / 4 ≤ 2499))
                                         (h2 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 9 ∣ n ↔ (112 ≤ n / 9 ∧ n / 9 ≤ 1111))
                                         (h3 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 36 ∣ n ↔ (28 ≤ n / 36 ∧ n / 36 ≤ 277)) :
                                         (9000 - ((2250 : ℕ) + 1000 - 250)) = 6000 :=
by sorry

end four_digit_not_multiples_of_4_or_9_l166_166049


namespace combined_money_half_l166_166515

theorem combined_money_half
  (J S : ℚ)
  (h1 : J = S)
  (h2 : J - (3/7 * J + 2/5 * J + 1/4 * J) = 24)
  (h3 : S - (1/2 * S + 1/3 * S) = 36) :
  1.5 * J = 458.18 := 
by
  sorry

end combined_money_half_l166_166515


namespace estimate_sqrt_expr_l166_166008

theorem estimate_sqrt_expr :
  2 < (Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3)) ∧ 
  (Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3)) < 3 := 
sorry

end estimate_sqrt_expr_l166_166008


namespace min_value_expression_l166_166478

variable {m n : ℝ}

theorem min_value_expression (hm : m > 0) (hn : n > 0) (hperp : m + n = 1) :
  ∃ (m n : ℝ), (1 / m + 2 / n = 3 + 2 * Real.sqrt 2) :=
by 
  sorry

end min_value_expression_l166_166478


namespace winningTicketProbability_l166_166509

-- Given conditions
def sharpBallProbability : ℚ := 1 / 30
def prizeBallsProbability : ℚ := 1 / (Nat.descFactorial 50 6)

-- The target probability that we are supposed to prove
def targetWinningProbability : ℚ := 1 / 476721000

-- Main theorem stating the required probability calculation
theorem winningTicketProbability :
  sharpBallProbability * prizeBallsProbability = targetWinningProbability :=
  sorry

end winningTicketProbability_l166_166509


namespace line_equation_l166_166346

theorem line_equation
  (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (hP : P = (-4, 6))
  (hxA : A.2 = 0) (hyB : B.1 = 0)
  (hMidpoint : P = ((A.1 + B.1)/2, (A.2 + B.2)/2)):
  3 * A.1 - 2 * B.2 + 24 = 0 :=
by
  -- Define point P
  let P := (-4, 6)
  -- Define points A and B, knowing P is the midpoint of AB and using conditions from the problem
  let A := (-8, 0)
  let B := (0, 12)
  sorry

end line_equation_l166_166346


namespace minor_premise_l166_166172

-- Definitions
def Rectangle : Type := sorry
def Square : Type := sorry
def Parallelogram : Type := sorry

axiom rectangle_is_parallelogram : Rectangle → Parallelogram
axiom square_is_rectangle : Square → Rectangle
axiom square_is_parallelogram : Square → Parallelogram

-- Problem statement
theorem minor_premise : ∀ (S : Square), ∃ (R : Rectangle), square_is_rectangle S = R :=
by
  sorry

end minor_premise_l166_166172


namespace alcohol_concentration_l166_166575

theorem alcohol_concentration (x : ℝ) (initial_volume : ℝ) (initial_concentration : ℝ) (target_concentration : ℝ) :
  initial_volume = 6 →
  initial_concentration = 0.35 →
  target_concentration = 0.50 →
  (2.1 + x) / (6 + x) = target_concentration →
  x = 1.8 :=
by
  intros h1 h2 h3 h4
  sorry

end alcohol_concentration_l166_166575


namespace range_of_a_l166_166948

theorem range_of_a :
  (∃ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)
    ↔ a ≤ -2 ∨ a = 1) := 
sorry

end range_of_a_l166_166948


namespace solve_inequality_l166_166350

theorem solve_inequality (k : ℝ) :
  (∀ (x : ℝ), (k + 2) * x > k + 2 → x < 1) → k = -3 :=
  by
  sorry

end solve_inequality_l166_166350


namespace find_m_l166_166028

variable {α : Type*} [DecidableEq α]

-- Definitions and conditions
def A (m : ℤ) : Set ℤ := {-1, 3, m ^ 2}
def B : Set ℤ := {3, 4}

theorem find_m (m : ℤ) (h : B ⊆ A m) : m = 2 ∨ m = -2 := by
  sorry

end find_m_l166_166028


namespace part_one_solution_part_two_solution_l166_166494

-- (I) Prove the solution set for the given inequality with m = 2.
theorem part_one_solution (x : ℝ) : 
  (|x - 2| > 7 - |x - 1|) ↔ (x < -4 ∨ x > 5) :=
sorry

-- (II) Prove the range of m given the condition.
theorem part_two_solution (m : ℝ) : 
  (∃ x : ℝ, |x - m| > 7 + |x - 1|) ↔ (m ∈ Set.Iio (-6) ∪ Set.Ioi (8)) :=
sorry

end part_one_solution_part_two_solution_l166_166494


namespace problem_solution_l166_166186

-- Define the operation otimes
def otimes (x y : ℚ) : ℚ := (x * y) / (x + y / 3)

-- Define the specific values x and y
def x : ℚ := 4
def y : ℚ := 3/2 -- 1.5 in fraction form

-- Prove the mathematical statement
theorem problem_solution : (0.36 : ℚ) * (otimes x y) = 12 / 25 := by
  sorry

end problem_solution_l166_166186


namespace arccos_neg_one_l166_166156

theorem arccos_neg_one : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_l166_166156


namespace billy_buys_bottle_l166_166778

-- Definitions of costs and volumes
def money : ℝ := 10
def cost1 : ℝ := 1
def volume1 : ℝ := 10
def cost2 : ℝ := 2
def volume2 : ℝ := 16
def cost3 : ℝ := 2.5
def volume3 : ℝ := 25
def cost4 : ℝ := 5
def volume4 : ℝ := 50
def cost5 : ℝ := 10
def volume5 : ℝ := 200

-- Statement of the proof problem
theorem billy_buys_bottle : ∃ b : ℕ, b = 1 ∧ cost5 = money := by 
  sorry

end billy_buys_bottle_l166_166778


namespace find_quadratic_eq_l166_166270

theorem find_quadratic_eq (x y : ℝ) (hx : x + y = 10) (hy : |x - y| = 12) :
    ∃ a b c : ℝ, a = 1 ∧ b = -10 ∧ c = -11 ∧ (x^2 + b * x + c = 0) ∧ (y^2 + b * y + c = 0) := by
  sorry

end find_quadratic_eq_l166_166270


namespace nonagon_side_length_l166_166739

theorem nonagon_side_length (perimeter : ℝ) (n : ℕ) (h_reg_nonagon : n = 9) (h_perimeter : perimeter = 171) :
  perimeter / n = 19 := by
  sorry

end nonagon_side_length_l166_166739


namespace expr_undefined_iff_l166_166021

theorem expr_undefined_iff (x : ℝ) : (x^2 - 9 = 0) ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end expr_undefined_iff_l166_166021


namespace admission_methods_correct_l166_166062

-- Define the number of famous schools.
def famous_schools : ℕ := 8

-- Define the number of students.
def students : ℕ := 3

-- Define the total number of different admission methods:
def admission_methods (schools : ℕ) (students : ℕ) : ℕ :=
  Nat.choose schools 2 * 3

-- The theorem stating the desired result.
theorem admission_methods_correct :
  admission_methods famous_schools students = 84 :=
by
  sorry

end admission_methods_correct_l166_166062


namespace proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l166_166653

variable (a b : ℝ)

-- Condition
def condition : Prop :=
  (1 / a) - (1 / b) = 1 / (a + b)

-- Proof statement
theorem proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab (h : condition a b) :
  (1 / a^2) - (1 / b^2) = 1 / (a * b) :=
sorry

end proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l166_166653


namespace tammy_speed_proof_l166_166742

noncomputable def tammy_average_speed_second_day (v t : ℝ) :=
  v + 0.5

theorem tammy_speed_proof :
  ∃ v t : ℝ, 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    tammy_average_speed_second_day v t = 4 :=
by
  sorry

end tammy_speed_proof_l166_166742


namespace volume_of_given_cuboid_l166_166571

-- Definition of the function to compute the volume of a cuboid
def volume_of_cuboid (length width height : ℝ) : ℝ :=
  length * width * height

-- Given conditions and the proof target
theorem volume_of_given_cuboid : volume_of_cuboid 2 5 3 = 30 :=
by
  sorry

end volume_of_given_cuboid_l166_166571


namespace circumradius_of_regular_tetrahedron_l166_166929

theorem circumradius_of_regular_tetrahedron (a : ℝ) (h : a > 0) :
    ∃ R : ℝ, R = a * (Real.sqrt 6) / 4 :=
by
  sorry

end circumradius_of_regular_tetrahedron_l166_166929


namespace arithmetic_sequence_an_smallest_m_l166_166230

-- Definitions for the sequence {a_n} and its sum S_n
def Sn (n : ℕ) := 3 * n^2 - 2 * n

-- The sequence terms {a_n}
def a_n : ℕ → ℕ
| 0 => 0
| n+1 => Sn (n+1) - Sn n

-- Definitions for the sequence sum T_n
def T_n (n : ℕ) := 1/2 * (1 - 1/(6*n + 1))

-- Conditions
axiom HSn (n : ℕ) : Sn n = 3 * n^2 - 2 * n

-- Prove (1) that {a_n} is an arithmetic sequence with common difference 6
theorem arithmetic_sequence_an : ∀ (n: ℕ), ∃ d : ℕ, ∀ m: ℕ, a_n (m + 1) - a_n m = d := sorry

-- Prove (2) the smallest positive integer m such that T_n < m / 20 for all n ∈ ℕ+
theorem smallest_m : ∀ n : ℕ+, T_n n < (10 : ℕ) / 20 := sorry

end arithmetic_sequence_an_smallest_m_l166_166230


namespace fraction_ratio_l166_166471

theorem fraction_ratio (x y : ℕ) (h : (x / y : ℚ) / (2 / 3) = (3 / 5) / (6 / 7)) : 
  x = 27 ∧ y = 35 :=
by 
  sorry

end fraction_ratio_l166_166471


namespace steve_speed_back_home_l166_166869

-- Definitions based on conditions
def distance := 20 -- distance from house to work in km
def total_time := 6 -- total time on the road in hours
def speed_to_work (v : ℝ) := v -- speed to work in km/h
def speed_back_home (v : ℝ) := 2 * v -- speed back home in km/h

-- Theorem to assert the proof
theorem steve_speed_back_home (v : ℝ) (h : distance / v + distance / (2 * v) = total_time) :
  speed_back_home v = 10 := by
  -- Proof goes here but we just state sorry to skip it
  sorry

end steve_speed_back_home_l166_166869


namespace anchuria_cert_prob_higher_2012_l166_166360

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  binomial_coefficient n k * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def cumulative_binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Finset.range (k + 1)).sum (λ i, binomial_probability n i p)

theorem anchuria_cert_prob_higher_2012 :
  let p := 0.25
  let q := 0.75
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  P_pass_2011 = 1 - cumulative_binomial_probability n2011 (k2011 - 1) p
  P_pass_2012 = 1 - cumulative_binomial_probability n2012 (k2012 - 1) p
  in P_pass_2012 > P_pass_2011 :=
by
  let p : ℝ := 0.25
  let q : ℝ := 0.75
  let n2011 : ℕ := 20
  let k2011 : ℕ := 3
  let n2012 : ℕ := 40
  let k2012 : ℕ := 6
  let P_fewer_than_3_2011 := cumulative_binomial_probability n2011 (k2011 - 1) p
  let P_fewer_than_6_2012 := cumulative_binomial_probability n2012 (k2012 - 1) p
  let P_pass_2011 := 1 - P_fewer_than_3_2011
  let P_pass_2012 := 1 - P_fewer_than_6_2012
  show P_pass_2012 > P_pass_2011 from sorry

end anchuria_cert_prob_higher_2012_l166_166360


namespace find_x_axis_intercept_l166_166147

theorem find_x_axis_intercept : ∃ x, 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  -- The theorem states that there exists an x-intercept such that substituting y = 0 in the equation results in x = -2.5.
  sorry

end find_x_axis_intercept_l166_166147


namespace curve_crosses_itself_l166_166166

-- Definitions of the parametric equations
def x (t k : ℝ) : ℝ := t^2 + k
def y (t k : ℝ) : ℝ := t^3 - k * t + 5

-- The main theorem statement
theorem curve_crosses_itself (k : ℝ) (ha : ℝ) (hb : ℝ) :
  ha ≠ hb →
  x ha k = x hb k →
  y ha k = y hb k →
  k = 9 ∧ x ha k = 18 ∧ y ha k = 5 :=
by
  sorry

end curve_crosses_itself_l166_166166


namespace smallest_natural_number_condition_l166_166557

theorem smallest_natural_number_condition (N : ℕ) : 
  (∀ k : ℕ, (10^6 - 1) * k = (10^54 - 1) / 9 → k < N) →
  N = 111112 :=
by
  sorry

end smallest_natural_number_condition_l166_166557


namespace pencils_ratio_l166_166965

theorem pencils_ratio (C J : ℕ) (hJ : J = 18) 
    (hJ_to_A : J_to_A = J / 3) (hJ_left : J_left = J - J_to_A)
    (hJ_left_eq : J_left = C + 3) :
    (C : ℚ) / (J : ℚ) = 1 / 2 :=
by
  sorry

end pencils_ratio_l166_166965


namespace quadratic_roots_relation_l166_166474

theorem quadratic_roots_relation (a b s p : ℝ) (h : a^2 + b^2 = 15) (h1 : s = a + b) (h2 : p = a * b) : s^2 - 2 * p = 15 :=
by sorry

end quadratic_roots_relation_l166_166474


namespace find_a_8_l166_166406

noncomputable def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ ∃ b : ℕ → ℤ, (∀ n : ℕ, 0 < n → b n = a (n + 1) - a n) ∧
  b 3 = -2 ∧ b 10 = 12

theorem find_a_8 (a : ℕ → ℤ) (h : sequence_a a) : a 8 = 3 :=
sorry

end find_a_8_l166_166406


namespace higher_probability_in_2012_l166_166361

def bernoulli_probability (n k : ℕ) (p : ℝ) : ℝ :=
  ∑ i in finset.range (k + 1), nat.choose n i * (p ^ i) * ((1 - p) ^ (n - i))

theorem higher_probability_in_2012 : 
  let p := 0.25
  let n2011 := 20
  let k2011 := 3
  let n2012 := 40
  let k2012 := 6
  let pass_prob_2011 := 1 - bernoulli_probability n2011 (k2011 - 1) p
  let pass_prob_2012 := 1 - bernoulli_probability n2012 (k2012 - 1) p
  pass_prob_2012 > pass_prob_2011 :=
by
  -- We would provide the actual proof here, but for now, we use sorry.
  sorry

end higher_probability_in_2012_l166_166361


namespace opposite_of_neg_three_sevenths_l166_166091

theorem opposite_of_neg_three_sevenths:
  ∀ x : ℚ, (x = -3 / 7) → (∃ y : ℚ, y + x = 0 ∧ y = 3 / 7) :=
by
  sorry

end opposite_of_neg_three_sevenths_l166_166091


namespace complement_A_inter_B_l166_166202

def U : Set ℤ := { x | -1 ≤ x ∧ x ≤ 2 }
def A : Set ℤ := { x | x * (x - 1) = 0 }
def B : Set ℤ := { x | -1 < x ∧ x < 2 }

theorem complement_A_inter_B {U A B : Set ℤ} :
  A ⊆ U → B ⊆ U → 
  (A ∩ B) ⊆ (U ∩ A ∩ B) → 
  (U \ (A ∩ B)) = { -1, 2 } :=
by 
  sorry

end complement_A_inter_B_l166_166202


namespace two_digit_number_is_42_l166_166930

theorem two_digit_number_is_42 (a b : ℕ) (ha : a < 10) (hb : b < 10) (h : 10 * a + b = 42) :
  ((10 * a + b) : ℚ) / (10 * b + a) = 7 / 4 := by
  sorry

end two_digit_number_is_42_l166_166930


namespace simplify_expression_l166_166672

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l166_166672


namespace find_digits_l166_166440

theorem find_digits (A B C : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h4 : A ≠ B) (h5 : B ≠ C) (h6 : A ≠ C)
  (h7 : A + B + C =  19) (h8 : 198 * (A - C) = 792) : {A, B, C} = {8, 7, 4} :=
by
  sorry

end find_digits_l166_166440


namespace extra_money_from_customer_l166_166459

theorem extra_money_from_customer
  (price_per_craft : ℕ)
  (num_crafts_sold : ℕ)
  (deposit_amount : ℕ)
  (remaining_amount : ℕ)
  (total_amount_before_deposit : ℕ)
  (amount_made_from_crafts : ℕ)
  (extra_money : ℕ) :
  price_per_craft = 12 →
  num_crafts_sold = 3 →
  deposit_amount = 18 →
  remaining_amount = 25 →
  total_amount_before_deposit = deposit_amount + remaining_amount →
  amount_made_from_crafts = price_per_craft * num_crafts_sold →
  extra_money = total_amount_before_deposit - amount_made_from_crafts →
  extra_money = 7 :=
by
  intros; sorry

end extra_money_from_customer_l166_166459


namespace sheena_weeks_to_complete_l166_166861

/- Definitions -/
def time_per_dress : ℕ := 12
def number_of_dresses : ℕ := 5
def weekly_sewing_time : ℕ := 4

/- Theorem -/
theorem sheena_weeks_to_complete : (number_of_dresses * time_per_dress) / weekly_sewing_time = 15 := 
by 
  /- Proof is omitted -/
  sorry

end sheena_weeks_to_complete_l166_166861


namespace cost_of_each_entree_l166_166393

def cost_of_appetizer : ℝ := 10
def number_of_entrees : ℝ := 4
def tip_percentage : ℝ := 0.20
def total_spent : ℝ := 108

theorem cost_of_each_entree :
  ∃ E : ℝ, total_spent = cost_of_appetizer + number_of_entrees * E + tip_percentage * (cost_of_appetizer + number_of_entrees * E) ∧ E = 20 :=
by
  sorry

end cost_of_each_entree_l166_166393


namespace diff_lines_not_parallel_perpendicular_same_plane_l166_166193

-- Variables
variables (m n : Type) (α β : Type)

-- Conditions
-- m and n are different lines, which we can assume as different types (or elements of some type).
-- α and β are different planes, which we can assume as different types (or elements of some type).
-- There exist definitions for parallel and perpendicular relationships between lines and planes.

def areParallel (x y : Type) : Prop := sorry
def arePerpendicularToSamePlane (x y : Type) : Prop := sorry

-- Theorem Statement
theorem diff_lines_not_parallel_perpendicular_same_plane
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : ¬ areParallel m n) :
  ¬ arePerpendicularToSamePlane m n :=
sorry

end diff_lines_not_parallel_perpendicular_same_plane_l166_166193


namespace student_average_grade_l166_166564

noncomputable def average_grade_two_years : ℝ :=
  let year1_courses := 6
  let year1_average_grade := 100
  let year1_total_points := year1_courses * year1_average_grade

  let year2_courses := 5
  let year2_average_grade := 40
  let year2_total_points := year2_courses * year2_average_grade

  let total_courses := year1_courses + year2_courses
  let total_points := year1_total_points + year2_total_points

  total_points / total_courses

theorem student_average_grade : average_grade_two_years = 72.7 :=
by
  sorry

end student_average_grade_l166_166564


namespace intersection_of_sets_l166_166498

theorem intersection_of_sets (M : Set ℤ) (N : Set ℤ) (H_M : M = {0, 1, 2, 3, 4}) (H_N : N = {-2, 0, 2}) :
  M ∩ N = {0, 2} :=
by
  rw [H_M, H_N]
  ext
  simp
  sorry  -- Proof to be filled in

end intersection_of_sets_l166_166498


namespace bells_toll_together_l166_166306

theorem bells_toll_together {a b c d : ℕ} (h1 : a = 9) (h2 : b = 10) (h3 : c = 14) (h4 : d = 18) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 630 :=
by
  sorry

end bells_toll_together_l166_166306


namespace intersection_of_A_and_B_l166_166205

namespace SetsIntersectionProof

def setA : Set ℝ := { x | |x| ≤ 2 }
def setB : Set ℝ := { x | x < 1 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | -2 ≤ x ∧ x < 1 } :=
sorry

end SetsIntersectionProof

end intersection_of_A_and_B_l166_166205
