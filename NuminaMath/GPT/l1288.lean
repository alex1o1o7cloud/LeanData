import Mathlib

namespace norma_cards_left_l1288_128816

def initial_cards : ℕ := 88
def lost_cards : ℕ := 70
def remaining_cards (initial lost : ℕ) : ℕ := initial - lost

theorem norma_cards_left : remaining_cards initial_cards lost_cards = 18 := by
  sorry

end norma_cards_left_l1288_128816


namespace average_salary_for_company_l1288_128826

variable (n_m : ℕ) -- number of managers
variable (n_a : ℕ) -- number of associates
variable (avg_salary_m : ℕ) -- average salary of managers
variable (avg_salary_a : ℕ) -- average salary of associates

theorem average_salary_for_company (h_n_m : n_m = 15) (h_n_a : n_a = 75) 
  (h_avg_salary_m : avg_salary_m = 90000) (h_avg_salary_a : avg_salary_a = 30000) : 
  (n_m * avg_salary_m + n_a * avg_salary_a) / (n_m + n_a) = 40000 := 
by
  sorry

end average_salary_for_company_l1288_128826


namespace range_of_a_l1288_128877

noncomputable def a_n (n : ℕ) (a : ℝ) : ℝ :=
  (-1)^(n + 2018) * a

noncomputable def b_n (n : ℕ) : ℝ :=
  2 + (-1)^(n + 2019) / n

theorem range_of_a (a : ℝ) :
  (∀ n : ℕ, 1 ≤ n → a_n n a < b_n n) ↔ -2 ≤ a ∧ a < 3 / 2 :=
  sorry

end range_of_a_l1288_128877


namespace number_of_pencils_l1288_128837

-- Define the given conditions
def circle_radius : ℝ := 14 -- 14 feet radius
def pencil_length_inches : ℝ := 6 -- 6-inch pencil

noncomputable def pencil_length_feet : ℝ := pencil_length_inches / 12 -- convert 6 inches to feet

-- Statement of the problem in Lean
theorem number_of_pencils (r : ℝ) (p_len_inch : ℝ) (d : ℝ) (p_len_feet : ℝ) :
  r = circle_radius →
  p_len_inch = pencil_length_inches →
  d = 2 * r →
  p_len_feet = pencil_length_feet →
  d / p_len_feet = 56 :=
by
  intros hr hp hd hpl
  sorry

end number_of_pencils_l1288_128837


namespace LawOfCosines_triangle_l1288_128896

theorem LawOfCosines_triangle {a b C : ℝ} (ha : a = 9) (hb : b = 2 * Real.sqrt 3) (hC : C = Real.pi / 6 * 5) :
  ∃ c, c = 2 * Real.sqrt 30 :=
by
  sorry

end LawOfCosines_triangle_l1288_128896


namespace four_by_four_increasing_matrices_l1288_128852

noncomputable def count_increasing_matrices (n : ℕ) : ℕ := sorry

theorem four_by_four_increasing_matrices :
  count_increasing_matrices 4 = 320 :=
sorry

end four_by_four_increasing_matrices_l1288_128852


namespace smallest_possible_value_of_n_l1288_128853

theorem smallest_possible_value_of_n 
  {a b c m n : ℕ} 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (hc_pos : c > 0) 
  (h_ordering : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c = 3010) 
  (h_factorial : a.factorial * b.factorial * c.factorial = m * 10^n) 
  (h_m_not_div_10 : ¬ (10 ∣ m)) 
  : n = 746 := 
sorry

end smallest_possible_value_of_n_l1288_128853


namespace number_of_girls_who_left_l1288_128875

-- Definitions for initial conditions and event information
def initial_boys : ℕ := 24
def initial_girls : ℕ := 14
def final_students : ℕ := 30

-- Main theorem statement translating the problem question
theorem number_of_girls_who_left (B G : ℕ) (h1 : B = G) 
  (h2 : initial_boys + initial_girls - B - G = final_students) :
  G = 4 := 
sorry

end number_of_girls_who_left_l1288_128875


namespace min_value_x_y_l1288_128800

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 4/y = 1) : x + y ≥ 9 :=
sorry

end min_value_x_y_l1288_128800


namespace intersection_A_B_l1288_128898

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x^2 = x}

theorem intersection_A_B : A ∩ B = {1} := 
by
  sorry

end intersection_A_B_l1288_128898


namespace pythagorean_ratio_l1288_128883

variables (a b : ℝ)

theorem pythagorean_ratio (h1 : a > 0) (h2 : b > a) (h3 : b^2 = 13 * (b - a)^2) :
  a / b = 2 / 3 :=
sorry

end pythagorean_ratio_l1288_128883


namespace vasya_fraction_l1288_128820

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l1288_128820


namespace trains_cross_time_l1288_128835

noncomputable def time_to_cross (length_train : ℝ) (speed_train_kmph : ℝ) : ℝ :=
  let relative_speed_kmph := speed_train_kmph + speed_train_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  let total_distance := length_train + length_train
  total_distance / relative_speed_mps

theorem trains_cross_time :
  time_to_cross 180 80 = 8.1 := 
by
  sorry

end trains_cross_time_l1288_128835


namespace magician_earning_l1288_128830

-- Definitions based on conditions
def price_per_deck : ℕ := 2
def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3

-- Theorem statement
theorem magician_earning :
  let sold_decks := initial_decks - remaining_decks
  let earning := sold_decks * price_per_deck
  earning = 4 := by
  sorry

end magician_earning_l1288_128830


namespace total_weight_of_envelopes_l1288_128838

theorem total_weight_of_envelopes :
  (8.5 * 880 / 1000) = 7.48 :=
by
  sorry

end total_weight_of_envelopes_l1288_128838


namespace single_discount_equivalent_l1288_128833

theorem single_discount_equivalent :
  ∀ (original final: ℝ) (d1 d2 d3 total_discount: ℝ),
  original = 800 →
  d1 = 0.15 →
  d2 = 0.10 →
  d3 = 0.05 →
  final = original * (1 - d1) * (1 - d2) * (1 - d3) →
  total_discount = 1 - (final / original) →
  total_discount = 0.27325 :=
by
  intros original final d1 d2 d3 total_discount h1 h2 h3 h4 h5 h6
  sorry

end single_discount_equivalent_l1288_128833


namespace sum_of_digits_of_d_l1288_128809

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_of_digits_of_d (d : ℕ) 
  (h_exchange : 15 * d = 9 * (d * 5 / 3)) 
  (h_spending : (5 * d / 3) - 120 = d) 
  (h_d_eq : d = 180) : sum_of_digits d = 9 := by
  -- This is where the proof would go
  sorry

end sum_of_digits_of_d_l1288_128809


namespace question1_question2_l1288_128868

noncomputable def A (x : ℝ) : Prop := x^2 - 3 * x + 2 ≤ 0
noncomputable def B_set (x a : ℝ) : ℝ := x^2 - 2 * x + a
def B (y a : ℝ) : Prop := y ≥ a - 1
noncomputable def C (x a : ℝ) : Prop := x^2 - a * x - 4 ≤ 0

def prop_p (a : ℝ) : Prop := ∃ x, A x ∧ B (B_set x a) a
def prop_q (a : ℝ) : Prop := ∀ x, A x → C x a

theorem question1 (a : ℝ) (h : ¬ prop_p a) : a > 3 :=
sorry

theorem question2 (a : ℝ) (hp : prop_p a) (hq : prop_q a) : 0 ≤ a ∧ a ≤ 3 :=
sorry

end question1_question2_l1288_128868


namespace range_S₁₂_div_d_l1288_128893

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (a₁ d : α) (n : ℕ) : α :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem range_S₁₂_div_d (a₁ d : α) (h_a₁_pos : a₁ > 0) (h_d_neg : d < 0) 
  (h_max_S_8 : ∀ n, arithmetic_sequence_sum a₁ d n ≤ arithmetic_sequence_sum a₁ d 8) :
  -30 < (arithmetic_sequence_sum a₁ d 12) / d ∧ (arithmetic_sequence_sum a₁ d 12) / d < -18 :=
by
  have h1 : -8 < a₁ / d := by sorry
  have h2 : a₁ / d < -7 := by sorry
  have h3 : (arithmetic_sequence_sum a₁ d 12) / d = 12 * (a₁ / d) + 66 := by sorry
  sorry

end range_S₁₂_div_d_l1288_128893


namespace greatest_whole_number_lt_100_with_odd_factors_l1288_128873

theorem greatest_whole_number_lt_100_with_odd_factors :
  ∃ n, n < 100 ∧ (∃ p : ℕ, n = p * p) ∧ 
    ∀ m, (m < 100 ∧ (∃ q : ℕ, m = q * q)) → m ≤ n :=
sorry

end greatest_whole_number_lt_100_with_odd_factors_l1288_128873


namespace value_of_other_number_l1288_128841

theorem value_of_other_number (k : ℕ) (other_number : ℕ) (h1 : k = 2) (h2 : (5 + k) * (5 - k) = 5^2 - other_number) : other_number = 21 :=
  sorry

end value_of_other_number_l1288_128841


namespace product_of_two_numbers_l1288_128856

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x - y = 1 * k) 
  (h2 : x + y = 2 * k) 
  (h3 : (x * y)^2 = 18 * k) : (x * y = 16) := 
by 
    sorry


end product_of_two_numbers_l1288_128856


namespace percentage_increase_biking_time_l1288_128866

theorem percentage_increase_biking_time
  (time_young_hours : ℕ)
  (distance_young_miles : ℕ)
  (time_now_hours : ℕ)
  (distance_now_miles : ℕ)
  (time_young_minutes : ℕ := time_young_hours * 60)
  (time_now_minutes : ℕ := time_now_hours * 60)
  (time_per_mile_young : ℕ := time_young_minutes / distance_young_miles)
  (time_per_mile_now : ℕ := time_now_minutes / distance_now_miles)
  (increase_in_time_per_mile : ℕ := time_per_mile_now - time_per_mile_young)
  (percentage_increase : ℕ := (increase_in_time_per_mile * 100) / time_per_mile_young) :
  percentage_increase = 100 :=
by
  -- substitution of values for conditions
  have time_young_hours := 2
  have distance_young_miles := 20
  have time_now_hours := 3
  have distance_now_miles := 15
  sorry

end percentage_increase_biking_time_l1288_128866


namespace find_a_of_pure_imaginary_l1288_128851

noncomputable def isPureImaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = ⟨0, b⟩  -- complex number z is purely imaginary if it can be written as 0 + bi

theorem find_a_of_pure_imaginary (a : ℝ) (i : ℂ) (ha : i*i = -1) :
  isPureImaginary ((1 - i) * (a + i)) → a = -1 := by
  sorry

end find_a_of_pure_imaginary_l1288_128851


namespace at_least_one_even_difference_l1288_128858

-- Statement of the problem in Lean 4
theorem at_least_one_even_difference 
  (a b : Fin (2 * n + 1) → ℤ) 
  (hperm : ∃ σ : Equiv.Perm (Fin (2 * n + 1)), ∀ k, a k = (b ∘ σ) k) : 
  ∃ k, (a k - b k) % 2 = 0 := 
sorry

end at_least_one_even_difference_l1288_128858


namespace triangle_inequality_l1288_128813

open Real

theorem triangle_inequality (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (h_sum : A + B + C = π) :
  sin A * cos C + A * cos B > 0 :=
by
  sorry

end triangle_inequality_l1288_128813


namespace part_a_part_b_part_c_l1288_128840

-- Define initial setup and conditions
def average (scores: List ℚ) : ℚ :=
  scores.sum / scores.length

-- Part (a)
theorem part_a (A B : List ℚ) (a b : ℚ) (A' : List ℚ) (B' : List ℚ) :
  average A = a ∧ average B = b ∧ average A' = a ∧ average B' = b ∧
  average A' > a ∧ average B' > b :=
sorry

-- Part (b)
theorem part_b (A B : List ℚ) : 
  ∀ a b : ℚ, (average A = a ∧ average B = b ∧ ∀ A' : List ℚ, average A' > a ∧ ∀ B' : List ℚ, average B' > b) :=
sorry

-- Part (c)
theorem part_c (A B C : List ℚ) (a b c : ℚ) (A' B' C' A'' B'' C'' : List ℚ) :
  average A = a ∧ average B = b ∧ average C = c ∧
  average A' = a ∧ average B' = b ∧ average C' = c ∧
  average A'' = a ∧ average B'' = b ∧ average C'' = c ∧
  average A' > a ∧ average B' > b ∧ average C' > c ∧
  average A'' > average A' ∧ average B'' > average B' ∧ average C'' > average C' :=
sorry

end part_a_part_b_part_c_l1288_128840


namespace find_divisor_l1288_128884

theorem find_divisor (x d : ℕ) (h1 : x ≡ 7 [MOD d]) (h2 : (x + 11) ≡ 18 [MOD 31]) : d = 31 := 
sorry

end find_divisor_l1288_128884


namespace car_trip_time_l1288_128819

theorem car_trip_time (T A : ℕ) (h1 : 50 * T = 140 + 53 * A) (h2 : T = 4 + A) : T = 24 := by
  sorry

end car_trip_time_l1288_128819


namespace amusement_park_ticket_cost_l1288_128891

theorem amusement_park_ticket_cost (T_adult T_child : ℕ) (num_children num_adults : ℕ) 
  (h1 : T_adult = 15) (h2 : T_child = 8) 
  (h3 : num_children = 15) (h4 : num_adults = 25 + num_children) :
  num_adults * T_adult + num_children * T_child = 720 :=
by
  sorry

end amusement_park_ticket_cost_l1288_128891


namespace probability_odd_divisor_25_factorial_l1288_128863

theorem probability_odd_divisor_25_factorial : 
  let divisors := (22 + 1) * (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let odd_divisors := (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  (odd_divisors / divisors = 1 / 23) :=
sorry

end probability_odd_divisor_25_factorial_l1288_128863


namespace no_non_trivial_solutions_l1288_128897

theorem no_non_trivial_solutions (x y z : ℤ) :
  3 * x^2 + 7 * y^2 = z^4 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro h
  -- Proof goes here
  sorry

end no_non_trivial_solutions_l1288_128897


namespace part1_part2_part3_l1288_128801

-- Part 1
def harmonic_fraction (num denom : ℚ) : Prop :=
  ∃ a b : ℚ, num = a - 2 * b ∧ denom = a^2 - b^2 ∧ ¬(∃ x : ℚ, a - 2 * b = (a - b) * x)

theorem part1 (a b : ℚ) (h : harmonic_fraction (a - 2 * b) (a^2 - b^2)) : true :=
  by sorry

-- Part 2
theorem part2 (a : ℕ) (h : harmonic_fraction (x - 1) (x^2 + a * x + 4)) : a = 4 ∨ a = 5 :=
  by sorry

-- Part 3
theorem part3 (a b : ℚ) :
  (4 * a^2 / (a * b^2 - b^3) - a / b * 4 / b) = (4 * a / (ab - b^2)) :=
  by sorry

end part1_part2_part3_l1288_128801


namespace sufficient_but_not_necessary_l1288_128857

theorem sufficient_but_not_necessary (x : ℝ) : (x = 1 → x * (x - 1) = 0) ∧ ¬(x * (x - 1) = 0 → x = 1) := 
by
  sorry

end sufficient_but_not_necessary_l1288_128857


namespace polynomial_evaluation_l1288_128834

-- Define the polynomial p(x) and the condition p(x) - p'(x) = x^2 + 2x + 1
variable (p : ℝ → ℝ)
variable (hp : ∀ x, p x - (deriv p x) = x^2 + 2 * x + 1)

-- Statement to prove p(5) = 50 given the conditions
theorem polynomial_evaluation : p 5 = 50 := 
sorry

end polynomial_evaluation_l1288_128834


namespace f_sum_positive_l1288_128848

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem f_sum_positive (x1 x2 : ℝ) (hx : x1 + x2 > 0) : f x1 + f x2 > 0 :=
sorry

end f_sum_positive_l1288_128848


namespace total_peaches_is_85_l1288_128818

-- Definitions based on conditions
def initial_peaches : ℝ := 61.0
def additional_peaches : ℝ := 24.0

-- Statement to prove
theorem total_peaches_is_85 :
  initial_peaches + additional_peaches = 85.0 := 
by sorry

end total_peaches_is_85_l1288_128818


namespace gummy_cost_proof_l1288_128885

variables (lollipop_cost : ℝ) (num_lollipops : ℕ) (initial_money : ℝ) (remaining_money : ℝ)
variables (num_gummies : ℕ) (cost_per_gummy : ℝ)

-- Conditions
def conditions : Prop :=
  lollipop_cost = 1.50 ∧
  num_lollipops = 4 ∧
  initial_money = 15 ∧
  remaining_money = 5 ∧
  num_gummies = 2 ∧
  initial_money - remaining_money = (num_lollipops * lollipop_cost) + (num_gummies * cost_per_gummy)

-- Proof problem
theorem gummy_cost_proof : conditions lollipop_cost num_lollipops initial_money remaining_money num_gummies cost_per_gummy → cost_per_gummy = 2 :=
by
  sorry  -- Solution steps would be filled in here


end gummy_cost_proof_l1288_128885


namespace total_shared_amount_l1288_128808

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

axiom h1 : A = 1 / 3 * (B + C)
axiom h2 : B = 2 / 7 * (A + C)
axiom h3 : A = B + 20

theorem total_shared_amount : A + B + C = 720 := by
  sorry

end total_shared_amount_l1288_128808


namespace minimum_odd_numbers_in_A_P_l1288_128894

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l1288_128894


namespace number_of_correct_statements_l1288_128825

def statement1_condition : Prop :=
∀ a b : ℝ, (a - b > 0) → (a > 0 ∧ b > 0)

def statement2_condition : Prop :=
∀ a b : ℝ, a - b = a + (-b)

def statement3_condition : Prop :=
∀ a : ℝ, (a - (-a) = 0)

def statement4_condition : Prop :=
∀ a : ℝ, 0 - a = -a

theorem number_of_correct_statements : 
  (¬ statement1_condition ∧ statement2_condition ∧ ¬ statement3_condition ∧ statement4_condition) →
  (2 = 2) :=
by
  intros
  trivial

end number_of_correct_statements_l1288_128825


namespace annual_increase_fraction_l1288_128882

theorem annual_increase_fraction (InitAmt FinalAmt : ℝ) (f : ℝ) :
  InitAmt = 51200 ∧ FinalAmt = 64800 ∧ FinalAmt = InitAmt * (1 + f)^2 →
  f = 0.125 :=
by
  intros h
  sorry

end annual_increase_fraction_l1288_128882


namespace parabola_and_x4_value_l1288_128865

theorem parabola_and_x4_value :
  (∀ P, dist P (0, 1/2) = dist P (x, -1/2) → ∃ y, P = (x, y) ∧ x^2 = 2 * y) ∧
  (∀ (x1 x2 : ℝ), x1 = 6 → x2 = 2 → ∃ x4, 1/x4 = 1/((3/2) : ℝ) + 1/x2 ∧ x4 = 6/7) :=
by
  sorry

end parabola_and_x4_value_l1288_128865


namespace largest_number_from_hcf_factors_l1288_128870

/-- This statement checks the largest number derivable from given HCF and factors. -/
theorem largest_number_from_hcf_factors (HCF factor1 factor2 : ℕ) (hHCF : HCF = 52) (hfactor1 : factor1 = 11) (hfactor2 : factor2 = 12) :
  max (HCF * factor1) (HCF * factor2) = 624 :=
by
  sorry

end largest_number_from_hcf_factors_l1288_128870


namespace value_of_2_pow_5_plus_5_l1288_128804

theorem value_of_2_pow_5_plus_5 : 2^5 + 5 = 37 := by
  sorry

end value_of_2_pow_5_plus_5_l1288_128804


namespace product_of_areas_eq_k3_times_square_of_volume_l1288_128831

variables (a b c k : ℝ)

-- Defining the areas of bottom, side, and front of the box as provided
def area_bottom := k * a * b
def area_side := k * b * c
def area_front := k * c * a

-- Volume of the box
def volume := a * b * c

-- The lean statement to be proved
theorem product_of_areas_eq_k3_times_square_of_volume :
  (area_bottom a b k) * (area_side b c k) * (area_front c a k) = k^3 * (volume a b c)^2 :=
by
  sorry

end product_of_areas_eq_k3_times_square_of_volume_l1288_128831


namespace distinct_pairs_disjoint_subsets_l1288_128827

theorem distinct_pairs_disjoint_subsets (n : ℕ) : 
  ∃ k, k = (3^n + 1) / 2 := 
sorry

end distinct_pairs_disjoint_subsets_l1288_128827


namespace black_lambs_count_l1288_128855

def total_lambs : ℕ := 6048
def white_lambs : ℕ := 193
def brown_lambs : ℕ := 527

theorem black_lambs_count :
  total_lambs - white_lambs - brown_lambs = 5328 :=
by
  -- Proof omitted
  sorry

end black_lambs_count_l1288_128855


namespace box_interior_surface_area_l1288_128879

-- Defining the conditions
def original_length := 30
def original_width := 20
def corner_length := 5
def num_corners := 4

-- Defining the area calculations based on given dimensions and removed corners
def original_area := original_length * original_width
def area_one_corner := corner_length * corner_length
def total_area_removed := num_corners * area_one_corner
def remaining_area := original_area - total_area_removed

-- Statement to prove
theorem box_interior_surface_area :
  remaining_area = 500 :=
by 
  sorry

end box_interior_surface_area_l1288_128879


namespace product_of_two_numbers_l1288_128806

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x ^ 2 + y ^ 2 = 289)
  (h2 : x + y = 23) : 
  x * y = 120 :=
by
  sorry

end product_of_two_numbers_l1288_128806


namespace arithmetic_expression_eval_l1288_128889

theorem arithmetic_expression_eval : 2 + 8 * 3 - 4 + 10 * 2 / 5 = 26 := by
  sorry

end arithmetic_expression_eval_l1288_128889


namespace functional_eq_log_l1288_128871

theorem functional_eq_log {f : ℝ → ℝ} (h₁ : f 4 = 2) 
                           (h₂ : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → f (x1 * x2) = f x1 + f x2) : 
                           (∀ x : ℝ, 0 < x → f x = Real.log x / Real.log 2) := 
by
  sorry

end functional_eq_log_l1288_128871


namespace complement_of_intersection_l1288_128805

theorem complement_of_intersection (U M N : Set ℕ)
  (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3})
  (hN : N = {2, 3, 4}) :
  (U \ (M ∩ N)) = {1, 4} :=
by
  rw [hU, hM, hN]
  sorry

end complement_of_intersection_l1288_128805


namespace smallest_integer_problem_l1288_128867

theorem smallest_integer_problem (m : ℕ) (h1 : Nat.lcm 60 m / Nat.gcd 60 m = 28) : m = 105 := sorry

end smallest_integer_problem_l1288_128867


namespace harmonic_mean_of_3_6_12_l1288_128862

-- Defining the harmonic mean function
def harmonic_mean (a b c : ℕ) : ℚ := 
  3 / ((1 / (a : ℚ)) + (1 / (b : ℚ)) + (1 / (c : ℚ)))

-- Stating the theorem
theorem harmonic_mean_of_3_6_12 : harmonic_mean 3 6 12 = 36 / 7 :=
by
  sorry

end harmonic_mean_of_3_6_12_l1288_128862


namespace sufficient_not_necessary_condition_l1288_128876

theorem sufficient_not_necessary_condition :
  ∀ x : ℝ, (x^2 - 3 * x < 0) → (0 < x ∧ x < 2) :=
by 
  sorry

end sufficient_not_necessary_condition_l1288_128876


namespace area_ratio_GHI_JKL_l1288_128817

-- Given conditions
def side_lengths_GHI : ℕ × ℕ × ℕ := (6, 8, 10)
def side_lengths_JKL : ℕ × ℕ × ℕ := (9, 12, 15)

-- Function to calculate the area of a right triangle given the lengths of the legs
def right_triangle_area (a b : ℕ) : ℕ :=
  (a * b) / 2

-- Function to determine if a triangle is a right triangle given its side lengths
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the main theorem
theorem area_ratio_GHI_JKL :
  let (a₁, b₁, c₁) := side_lengths_GHI
  let (a₂, b₂, c₂) := side_lengths_JKL
  is_right_triangle a₁ b₁ c₁ →
  is_right_triangle a₂ b₂ c₂ →
  right_triangle_area a₁ b₁ % right_triangle_area a₂ b₂ = 4 / 9 :=
by sorry

end area_ratio_GHI_JKL_l1288_128817


namespace no_solution_xy_l1288_128839

theorem no_solution_xy (x y : ℕ) : ¬ (x * (x + 1) = 4 * y * (y + 1)) :=
sorry

end no_solution_xy_l1288_128839


namespace peter_ends_up_with_eleven_erasers_l1288_128872

def eraser_problem : Nat :=
  let initial_erasers := 8
  let additional_erasers := 3
  let total_erasers := initial_erasers + additional_erasers
  total_erasers

theorem peter_ends_up_with_eleven_erasers :
  eraser_problem = 11 :=
by
  sorry

end peter_ends_up_with_eleven_erasers_l1288_128872


namespace find_pairs_l1288_128829

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ q r : ℕ, a^2 + b^2 = (a + b) * q + r ∧ q^2 + r = 1977) →
  (a, b) = (50, 37) ∨ (a, b) = (37, 50) ∨ (a, b) = (50, 7) ∨ (a, b) = (7, 50) :=
by
  sorry

end find_pairs_l1288_128829


namespace fraction_sum_l1288_128861

namespace GeometricSequence

-- Given conditions in the problem
def q : ℕ := 2

-- Definition of the sum of the first n terms (S_n) of a geometric sequence
def S_n (a₁ : ℤ) (n : ℕ) : ℤ := 
  a₁ * (1 - q ^ n) / (1 - q)

-- Specific sum for the first 4 terms (S₄)
def S₄ (a₁ : ℤ) : ℤ := S_n a₁ 4

-- Define the 2nd term of the geometric sequence
def a₂ (a₁ : ℤ) : ℤ := a₁ * q

-- The statement to prove: $\dfrac{S_4}{a_2} = \dfrac{15}{2}$
theorem fraction_sum (a₁ : ℤ) : (S₄ a₁) / (a₂ a₁) = Rat.ofInt 15 / Rat.ofInt 2 :=
  by
  -- Implementation of proof will go here
  sorry

end GeometricSequence

end fraction_sum_l1288_128861


namespace harly_adopts_percentage_l1288_128899

/-- Definitions for the conditions -/
def initial_dogs : ℝ := 80
def dogs_taken_back : ℝ := 5
def dogs_left : ℝ := 53

/-- Define the percentage of dogs adopted out -/
def percentage_adopted (P : ℝ) := P

/-- Lean 4 statement where we prove that if the given conditions are met, then the percentage of dogs initially adopted out is 40 -/
theorem harly_adopts_percentage : 
  ∃ P : ℝ, 
    (initial_dogs - (percentage_adopted P / 100 * initial_dogs) + dogs_taken_back = dogs_left) 
    ∧ P = 40 :=
by
  sorry

end harly_adopts_percentage_l1288_128899


namespace modulusOfComplexNumber_proof_l1288_128864

noncomputable def complexNumber {a : ℝ} (h : (a - 1) + 1 * Complex.I = (0 : ℂ)) : ℂ :=
  (2 + Real.sqrt 2 * Complex.I) / (a - Complex.I)

theorem modulusOfComplexNumber_proof (a : ℝ) (h : (a - 1) + 1 * Complex.I = (0 : ℂ)) : Complex.abs (complexNumber h) = Real.sqrt 3 := by
  sorry

end modulusOfComplexNumber_proof_l1288_128864


namespace impossible_sequence_l1288_128810

theorem impossible_sequence (α : ℝ) (hα : 0 < α ∧ α < 1) (a : ℕ → ℝ) (ha : ∀ n, 0 < a n) :
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) → false :=
by
  sorry

end impossible_sequence_l1288_128810


namespace percentage_of_green_ducks_smaller_pond_l1288_128844

-- Definitions of the conditions
def num_ducks_smaller_pond : ℕ := 30
def num_ducks_larger_pond : ℕ := 50
def percentage_green_larger_pond : ℕ := 12
def percentage_green_total : ℕ := 15
def total_ducks : ℕ := num_ducks_smaller_pond + num_ducks_larger_pond

-- Calculation of the number of green ducks
def num_green_larger_pond := percentage_green_larger_pond * num_ducks_larger_pond / 100
def num_green_total := percentage_green_total * total_ducks / 100

-- Define the percentage of green ducks in the smaller pond
def percentage_green_smaller_pond (x : ℕ) :=
  x * num_ducks_smaller_pond / 100 + num_green_larger_pond = num_green_total

-- The theorem to be proven
theorem percentage_of_green_ducks_smaller_pond : percentage_green_smaller_pond 20 :=
  sorry

end percentage_of_green_ducks_smaller_pond_l1288_128844


namespace det_A_l1288_128895

-- Define the matrix A
noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.sin 1, Real.cos 2, Real.sin 3],
   ![Real.sin 4, Real.cos 5, Real.sin 6],
   ![Real.sin 7, Real.cos 8, Real.sin 9]]

-- Define the explicit determinant calculation
theorem det_A :
  Matrix.det A = Real.sin 1 * (Real.cos 5 * Real.sin 9 - Real.sin 6 * Real.cos 8) -
                 Real.cos 2 * (Real.sin 4 * Real.sin 9 - Real.sin 6 * Real.sin 7) +
                 Real.sin 3 * (Real.sin 4 * Real.cos 8 - Real.cos 5 * Real.sin 7) :=
by
  sorry

end det_A_l1288_128895


namespace hilton_final_marbles_l1288_128814

theorem hilton_final_marbles :
  let initial_marbles := 26
  let found_marbles := 6
  let lost_marbles := 10
  let gift_multiplication_factor := 2
  let marbles_after_find_and_lose := initial_marbles + found_marbles - lost_marbles
  let gift_marbles := gift_multiplication_factor * lost_marbles
  let final_marbles := marbles_after_find_and_lose + gift_marbles
  final_marbles = 42 :=
by
  -- Proof to be filled
  sorry

end hilton_final_marbles_l1288_128814


namespace members_count_l1288_128812

theorem members_count
  (n : ℝ)
  (h1 : 191.25 = n / 4) :
  n = 765 :=
by
  sorry

end members_count_l1288_128812


namespace distance_traveled_by_bus_l1288_128845

noncomputable def total_distance : ℕ := 900
noncomputable def distance_by_plane : ℕ := total_distance / 3
noncomputable def distance_by_bus : ℕ := 360
noncomputable def distance_by_train : ℕ := (2 * distance_by_bus) / 3

theorem distance_traveled_by_bus :
  distance_by_plane + distance_by_train + distance_by_bus = total_distance :=
by
  sorry

end distance_traveled_by_bus_l1288_128845


namespace johns_age_l1288_128843

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l1288_128843


namespace find_term_number_l1288_128847

-- Define the arithmetic sequence
def arithmetic_seq (a d : Int) (n : Int) := a + (n - 1) * d

-- Define the condition: first term and common difference
def a1 := 4
def d := 3

-- Prove that the 672nd term is 2017
theorem find_term_number (n : Int) (h : arithmetic_seq a1 d n = 2017) : n = 672 := by
  sorry

end find_term_number_l1288_128847


namespace february_max_diff_percentage_l1288_128854

noncomputable def max_diff_percentage (D B F : ℕ) : ℚ :=
  let avg_others := (B + F) / 2
  let high_sales := max (max D B) F
  (high_sales - avg_others) / avg_others * 100

theorem february_max_diff_percentage :
  max_diff_percentage 8 5 6 = 45.45 := by
  sorry

end february_max_diff_percentage_l1288_128854


namespace find_length_AB_l1288_128802

open Real

noncomputable def AB_length := 
  let r := 4
  let V_total := 320 * π
  ∃ (L : ℝ), 16 * π * L + (256 / 3) * π = V_total ∧ L = 44 / 3

theorem find_length_AB :
  AB_length := by
  sorry

end find_length_AB_l1288_128802


namespace bugs_meet_at_point_P_l1288_128849

theorem bugs_meet_at_point_P (r1 r2 v1 v2 t : ℝ) (h1 : r1 = 7) (h2 : r2 = 3) (h3 : v1 = 4 * Real.pi) (h4 : v2 = 3 * Real.pi) :
  t = 14 :=
by
  repeat { sorry }

end bugs_meet_at_point_P_l1288_128849


namespace find_side_length_l1288_128832

theorem find_side_length
  (n : ℕ) 
  (h : (6 * n^2) / (6 * n^3) = 1 / 3) : 
  n = 3 := 
by
  sorry

end find_side_length_l1288_128832


namespace part_one_part_two_l1288_128892

def f (x : ℝ) : ℝ := abs (3 * x + 2)

theorem part_one (x : ℝ) : f x < 4 - abs (x - 1) ↔ x ∈ Set.Ioo (-5 / 4) (1 / 2) :=
sorry

noncomputable def g (x a : ℝ) : ℝ :=
if x < -2/3 then 2 * x + 2 + a
else if x ≤ a then -4 * x - 2 + a
else -2 * x - 2 - a

theorem part_two (m n a : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) :
  (∀ (x : ℝ), abs (x - a) - f x ≤ 1 / m + 1 / n) ↔ (0 < a ∧ a ≤ 10 / 3) :=
sorry

end part_one_part_two_l1288_128892


namespace range_of_k_has_extreme_values_on_interval_l1288_128811

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x - x^2 + 3 * x

theorem range_of_k_has_extreme_values_on_interval (k : ℝ) (h : k ≠ 0) :
  -9/8 < k ∧ k < 0 :=
sorry

end range_of_k_has_extreme_values_on_interval_l1288_128811


namespace mira_result_l1288_128807

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  if n % 100 >= 50 then n / 100 * 100 + 100 else n / 100 * 100

theorem mira_result :
  round_to_nearest_hundred ((63 + 48) - 21) = 100 :=
by
  sorry

end mira_result_l1288_128807


namespace count_5_numbers_after_996_l1288_128842

theorem count_5_numbers_after_996 : 
  ∃ a b c d e, a = 997 ∧ b = 998 ∧ c = 999 ∧ d = 1000 ∧ e = 1001 :=
sorry

end count_5_numbers_after_996_l1288_128842


namespace evaluate_expression_l1288_128887

theorem evaluate_expression (b x : ℝ) (h : x = b + 9) : (2 * x - b + 5) = (b + 23) :=
by
  sorry

end evaluate_expression_l1288_128887


namespace measure_of_one_interior_angle_of_regular_nonagon_is_140_l1288_128860

-- Define the number of sides for a nonagon
def number_of_sides_nonagon : ℕ := 9

-- Define the formula for the sum of the interior angles of a regular n-gon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- The sum of the interior angles of a nonagon
def sum_of_interior_angles_nonagon : ℕ := sum_of_interior_angles number_of_sides_nonagon

-- The measure of one interior angle of a regular n-gon
def measure_of_one_interior_angle (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- The measure of one interior angle of a regular nonagon
def measure_of_one_interior_angle_nonagon : ℕ := measure_of_one_interior_angle number_of_sides_nonagon

-- The final theorem statement
theorem measure_of_one_interior_angle_of_regular_nonagon_is_140 : 
  measure_of_one_interior_angle_nonagon = 140 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_nonagon_is_140_l1288_128860


namespace no_real_roots_of_quadratic_l1288_128880

-- Given an arithmetic sequence 
variable {a : ℕ → ℝ}

-- The conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k, m = n + k → a (m + 1) - a m = a (n + 1) - a n

def condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 5 + a 8 = 9

-- Lean 4 statement for the proof problem
theorem no_real_roots_of_quadratic (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : condition a) :
  let b := a 4 + a 6
  ∃ Δ, Δ = b ^ 2 - 4 * 10 ∧ Δ < 0 :=
by
  sorry

end no_real_roots_of_quadratic_l1288_128880


namespace decreasing_intervals_sin_decreasing_intervals_log_cos_l1288_128878

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

end decreasing_intervals_sin_decreasing_intervals_log_cos_l1288_128878


namespace find_x_values_l1288_128881

open Real

theorem find_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h₁ : x + 1/y = 5) (h₂ : y + 1/x = 7/4) : 
  x = 4/7 ∨ x = 5 := 
by sorry

end find_x_values_l1288_128881


namespace value_of_first_equation_l1288_128859

variables (x y z w : ℝ)

theorem value_of_first_equation (h1 : xw + yz = 8) (h2 : (2 * x + y) * (2 * z + w) = 20) : xz + yw = 1 := by
  sorry

end value_of_first_equation_l1288_128859


namespace present_age_of_father_l1288_128850

-- Definitions based on the conditions
variables (F S : ℕ)
axiom cond1 : F = 3 * S + 3
axiom cond2 : F + 3 = 2 * (S + 3) + 8

-- The theorem to prove
theorem present_age_of_father : F = 27 :=
by
  sorry

end present_age_of_father_l1288_128850


namespace externally_tangent_circles_radius_l1288_128803

theorem externally_tangent_circles_radius :
  ∃ r : ℝ, r > 0 ∧ (∀ x y, (x^2 + y^2 = 1 ∧ ((x - 3)^2 + y^2 = r^2)) → r = 2) :=
sorry

end externally_tangent_circles_radius_l1288_128803


namespace determine_compound_impossible_l1288_128823

-- Define the conditions
def contains_Cl (compound : Type) : Prop := true -- Placeholder definition
def mass_percentage_Cl (compound : Type) : ℝ := 0 -- Placeholder definition

-- Define the main statement
theorem determine_compound_impossible (compound : Type) 
  (containsCl : contains_Cl compound) 
  (massPercentageCl : mass_percentage_Cl compound = 47.3) : 
  ∃ (distinct_element : Type), compound = distinct_element := 
sorry

end determine_compound_impossible_l1288_128823


namespace chocolates_150_satisfies_l1288_128822

def chocolates_required (chocolates : ℕ) : Prop :=
  chocolates ≥ 150 ∧ chocolates % 19 = 17

theorem chocolates_150_satisfies : chocolates_required 150 :=
by
  -- We need to show that 150 satisfies the conditions:
  -- 1. 150 ≥ 150
  -- 2. 150 % 19 = 17
  unfold chocolates_required
  -- Both conditions hold:
  exact And.intro (by linarith) (by norm_num)

end chocolates_150_satisfies_l1288_128822


namespace remainder_of_87_pow_88_plus_7_l1288_128869

theorem remainder_of_87_pow_88_plus_7 :
  (87^88 + 7) % 88 = 8 :=
by sorry

end remainder_of_87_pow_88_plus_7_l1288_128869


namespace temperature_difference_l1288_128815

theorem temperature_difference (highest lowest : ℝ) (h_high : highest = 27) (h_low : lowest = 17) :
  highest - lowest = 10 :=
by
  sorry

end temperature_difference_l1288_128815


namespace divides_power_sum_l1288_128824

theorem divides_power_sum (a b c : ℤ) (h : a + b + c ∣ a^2 + b^2 + c^2) : ∀ k : ℕ, a + b + c ∣ a^(2^k) + b^(2^k) + c^(2^k) :=
by
  intro k
  induction k with
  | zero =>
    sorry -- Base case proof
  | succ k ih =>
    sorry -- Inductive step proof

end divides_power_sum_l1288_128824


namespace canonical_equations_of_line_intersection_l1288_128836

theorem canonical_equations_of_line_intersection
  (x y z : ℝ)
  (h1 : 2 * x - 3 * y + z + 6 = 0)
  (h2 : x - 3 * y - 2 * z + 3 = 0) :
  (∃ (m n p x0 y0 z0 : ℝ), 
  m * (x + 3) = n * y ∧ n * y = p * z ∧ 
  m = 9 ∧ n = 5 ∧ p = -3 ∧ 
  x0 = -3 ∧ y0 = 0 ∧ z0 = 0) :=
sorry

end canonical_equations_of_line_intersection_l1288_128836


namespace determinant_of_matrixA_l1288_128874

def matrixA : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![3, 0, -2],
  ![5, 6, -4],
  ![1, 3, 7]
]

theorem determinant_of_matrixA : Matrix.det matrixA = 144 := by
  sorry

end determinant_of_matrixA_l1288_128874


namespace lake_crystal_frogs_percentage_l1288_128890

noncomputable def percentage_fewer_frogs (frogs_in_lassie_lake total_frogs : ℕ) : ℕ :=
  let P := (total_frogs - frogs_in_lassie_lake) * 100 / frogs_in_lassie_lake
  P

theorem lake_crystal_frogs_percentage :
  let frogs_in_lassie_lake := 45
  let total_frogs := 81
  percentage_fewer_frogs frogs_in_lassie_lake total_frogs = 20 :=
by
  sorry

end lake_crystal_frogs_percentage_l1288_128890


namespace solution_set_of_abs_2x_minus_1_ge_3_l1288_128846

theorem solution_set_of_abs_2x_minus_1_ge_3 :
  { x : ℝ | |2 * x - 1| ≥ 3 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end solution_set_of_abs_2x_minus_1_ge_3_l1288_128846


namespace math_problem_l1288_128828

noncomputable def base10_b := 25 + 1  -- 101_5 in base 10
noncomputable def base10_c := 343 + 98 + 21 + 4  -- 1234_7 in base 10
noncomputable def base10_d := 2187 + 324 + 45 + 6  -- 3456_9 in base 10

theorem math_problem (a : ℕ) (b c d : ℕ) (h_a : a = 2468)
  (h_b : b = base10_b) (h_c : c = base10_c) (h_d : d = base10_d) :
  (a / b) * c - d = 41708 :=
  by {
  sorry
}

end math_problem_l1288_128828


namespace determine_m_l1288_128886

noncomputable def function_f (m : ℝ) (x : ℝ) : ℝ := m * x - |x + 1|

def exists_constant_interval (a b c m : ℝ) : Prop :=
  a < b ∧ ∀ x, a ≤ x ∧ x ≤ b → function_f m x = c

theorem determine_m (m : ℝ) (a b c : ℝ) :
  (a < b ∧ a ≥ -2 ∧ b ≥ -2 ∧ (∀ x, a ≤ x ∧ x ≤ b → function_f m x = c)) →
  m = 1 ∨ m = -1 :=
sorry

end determine_m_l1288_128886


namespace total_stickers_used_l1288_128888

-- Define all the conditions as given in the problem
def initially_water_bottles : ℕ := 20
def lost_at_school : ℕ := 5
def found_at_park : ℕ := 3
def stolen_at_dance : ℕ := 4
def misplaced_at_library : ℕ := 2
def acquired_from_friend : ℕ := 6
def stickers_per_bottle_school : ℕ := 4
def stickers_per_bottle_dance : ℕ := 3
def stickers_per_bottle_library : ℕ := 2

-- Prove the total number of stickers used
theorem total_stickers_used : 
  (lost_at_school * stickers_per_bottle_school)
  + (stolen_at_dance * stickers_per_bottle_dance)
  + (misplaced_at_library * stickers_per_bottle_library)
  = 36 := 
by
  sorry

end total_stickers_used_l1288_128888


namespace canoe_rental_cost_l1288_128821

theorem canoe_rental_cost (C : ℕ) (K : ℕ) :
  18 * K + C * (K + 5) = 405 → 
  3 * K = 2 * (K + 5) → 
  C = 15 :=
by
  intros revenue_eq ratio_eq
  sorry

end canoe_rental_cost_l1288_128821
