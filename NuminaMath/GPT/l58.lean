import Mathlib

namespace NUMINAMATH_GPT_k_is_3_l58_5813

noncomputable def k_solution (k : ℝ) : Prop :=
  k > 1 ∧ (∑' n : ℕ, (n^2 + 3 * n - 2) / k^n = 2)

theorem k_is_3 : ∃ k : ℝ, k_solution k ∧ k = 3 :=
by
  sorry

end NUMINAMATH_GPT_k_is_3_l58_5813


namespace NUMINAMATH_GPT_relationship_m_n_l58_5838

theorem relationship_m_n (b : ℝ) (m : ℝ) (n : ℝ) (h1 : m = 2 * b + 2022) (h2 : n = b^2 + 2023) : m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_relationship_m_n_l58_5838


namespace NUMINAMATH_GPT_ratio_p_q_l58_5887

section ProbabilityProof

-- Definitions and constants as per conditions
def N := Nat.factorial 15

def num_ways_A : ℕ := 4 * (Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))
def num_ways_B : ℕ := 4 * 3

def p : ℚ := num_ways_A / N
def q : ℚ := num_ways_B / N

-- Theorem: Prove that the ratio p/q is 560
theorem ratio_p_q : p / q = 560 := by
  sorry

end ProbabilityProof

end NUMINAMATH_GPT_ratio_p_q_l58_5887


namespace NUMINAMATH_GPT_fencing_cost_proof_l58_5872

theorem fencing_cost_proof (L : ℝ) (B : ℝ) (c : ℝ) (total_cost : ℝ)
  (hL : L = 60) (hL_B : L = B + 20) (hc : c = 26.50) : 
  total_cost = 5300 :=
by
  sorry

end NUMINAMATH_GPT_fencing_cost_proof_l58_5872


namespace NUMINAMATH_GPT_find_f_values_l58_5807

noncomputable def f : ℕ → ℕ := sorry

axiom condition1 : ∀ (a b : ℕ), a ≠ b → (a * f a + b * f b > a * f b + b * f a)
axiom condition2 : ∀ (n : ℕ), f (f n) = 3 * n

theorem find_f_values : f 1 + f 6 + f 28 = 66 := 
by
  sorry

end NUMINAMATH_GPT_find_f_values_l58_5807


namespace NUMINAMATH_GPT_minimum_value_2x_plus_y_l58_5853

theorem minimum_value_2x_plus_y (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : (1 / x) + (2 / (y + 1)) = 2) : 2 * x + y ≥ 3 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_2x_plus_y_l58_5853


namespace NUMINAMATH_GPT_inequality_proof_l58_5805

variable (b c : ℝ)
variable (hb : b > 0) (hc : c > 0)

theorem inequality_proof :
  (b - c) ^ 2011 * (b + c) ^ 2011 * (c - b) ^ 2011 ≥ (b ^ 2011 - c ^ 2011) * (b ^ 2011 + c ^ 2011) * (c ^ 2011 - b ^ 2011) :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l58_5805


namespace NUMINAMATH_GPT_greatest_is_B_l58_5831

def A : ℕ := 95 - 35
def B : ℕ := A + 12
def C : ℕ := B - 19

theorem greatest_is_B : B = 72 ∧ (B > A ∧ B > C) :=
by {
  -- Proof steps would be written here to prove the theorem.
  sorry
}

end NUMINAMATH_GPT_greatest_is_B_l58_5831


namespace NUMINAMATH_GPT_sum_of_coordinates_of_point_B_l58_5801

theorem sum_of_coordinates_of_point_B
  (x y : ℝ)
  (A : (ℝ × ℝ) := (2, 1))
  (B : (ℝ × ℝ) := (x, y))
  (h_line : y = 6)
  (h_slope : (y - 1) / (x - 2) = 4 / 5) :
  x + y = 14.25 :=
by {
  -- convert hypotheses to Lean terms and finish the proof
  sorry
}

end NUMINAMATH_GPT_sum_of_coordinates_of_point_B_l58_5801


namespace NUMINAMATH_GPT_Energetics_factory_l58_5854

/-- In the country "Energetics," there are 150 factories, and some of them are connected by bus
routes that do not stop anywhere except at these factories. It turns out that any four factories
can be split into two pairs such that a bus runs between each pair of factories. Find the minimum
number of pairs of factories that can be connected by bus routes. -/
theorem Energetics_factory
  (factories : Finset ℕ) (routes : Finset (ℕ × ℕ))
  (h_factories : factories.card = 150)
  (h_routes : ∀ (X Y Z W : ℕ),
    {X, Y, Z, W} ⊆ factories →
    ∃ (X1 Y1 Z1 W1 : ℕ),
    (X1, Y1) ∈ routes ∧
    (Z1, W1) ∈ routes ∧
    (X1 = X ∨ X1 = Y ∨ X1 = Z ∨ X1 = W) ∧
    (Y1 = X ∨ Y1 = Y ∨ Y1 = Z ∨ Y1 = W) ∧
    (Z1 = X ∨ Z1 = Y ∨ Z1 = Z ∨ Z1 = W) ∧
    (W1 = X ∨ W1 = Y ∨ W1 = Z ∨ W1 = W)) :
  (2 * routes.card) ≥ 11025 := sorry

end NUMINAMATH_GPT_Energetics_factory_l58_5854


namespace NUMINAMATH_GPT_lowest_sale_price_is_30_percent_l58_5833

-- Definitions and conditions
def list_price : ℝ := 80
def max_initial_discount : ℝ := 0.50
def additional_sale_discount : ℝ := 0.20

-- Calculations
def initial_discount_amount : ℝ := list_price * max_initial_discount
def initial_discounted_price : ℝ := list_price - initial_discount_amount
def additional_discount_amount : ℝ := list_price * additional_sale_discount
def lowest_sale_price : ℝ := initial_discounted_price - additional_discount_amount

-- Proof statement (with correct answer)
theorem lowest_sale_price_is_30_percent :
  lowest_sale_price = 0.30 * list_price := 
by
  sorry

end NUMINAMATH_GPT_lowest_sale_price_is_30_percent_l58_5833


namespace NUMINAMATH_GPT_choose_8_from_16_l58_5878

theorem choose_8_from_16 :
  Nat.choose 16 8 = 12870 :=
sorry

end NUMINAMATH_GPT_choose_8_from_16_l58_5878


namespace NUMINAMATH_GPT_car_highway_mileage_l58_5895

theorem car_highway_mileage :
  (∀ (H : ℝ), 
    (H > 0) → 
    (4 / H + 4 / 20 = (8 / H) * 1.4000000000000001) → 
    (H = 36)) :=
by
  intros H H_pos h_cond
  have : H = 36 := 
    sorry
  exact this

end NUMINAMATH_GPT_car_highway_mileage_l58_5895


namespace NUMINAMATH_GPT_radius_of_larger_circle_l58_5849

theorem radius_of_larger_circle
  (r r_s : ℝ)
  (h1 : r_s = 2)
  (h2 : π * r^2 = 4 * π * r_s^2) :
  r = 4 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_larger_circle_l58_5849


namespace NUMINAMATH_GPT_distance_travel_l58_5817

-- Definition of the parameters and the proof problem
variable (W_t : ℕ)
variable (R_c : ℕ)
variable (remaining_coal : ℕ)

-- Conditions
def rate_of_coal_consumption : Prop := R_c = 4 * W_t / 1000
def remaining_coal_amount : Prop := remaining_coal = 160

-- Theorem statement
theorem distance_travel (W_t : ℕ) (R_c : ℕ) (remaining_coal : ℕ) 
  (h1 : rate_of_coal_consumption W_t R_c) 
  (h2 : remaining_coal_amount remaining_coal) : 
  (remaining_coal * 1000 / 4 / W_t) = 40000 / W_t := 
by
  sorry

end NUMINAMATH_GPT_distance_travel_l58_5817


namespace NUMINAMATH_GPT_quadratic_equals_binomial_square_l58_5803

theorem quadratic_equals_binomial_square (d : ℝ) : 
  (∃ b : ℝ, (x^2 + 60 * x + d) = (x + b)^2) → d = 900 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equals_binomial_square_l58_5803


namespace NUMINAMATH_GPT_find_PF_2_l58_5877

-- Define the hyperbola and points
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 3) = 1
def PF_1 := 3
def a := 2
def two_a := 2 * a

-- State the theorem
theorem find_PF_2 (PF_2 : ℝ) (cond1 : PF_1 = 3) (cond2 : abs (PF_1 - PF_2) = two_a) : PF_2 = 7 :=
sorry

end NUMINAMATH_GPT_find_PF_2_l58_5877


namespace NUMINAMATH_GPT_seth_spent_more_l58_5859

theorem seth_spent_more : 
  let ice_cream_cartons := 20
  let yogurt_cartons := 2
  let ice_cream_price := 6
  let yogurt_price := 1
  let ice_cream_discount := 0.10
  let yogurt_discount := 0.20
  let total_ice_cream_cost := ice_cream_cartons * ice_cream_price
  let total_yogurt_cost := yogurt_cartons * yogurt_price
  let discounted_ice_cream_cost := total_ice_cream_cost * (1 - ice_cream_discount)
  let discounted_yogurt_cost := total_yogurt_cost * (1 - yogurt_discount)
  discounted_ice_cream_cost - discounted_yogurt_cost = 106.40 :=
by
  sorry

end NUMINAMATH_GPT_seth_spent_more_l58_5859


namespace NUMINAMATH_GPT_derivative_of_odd_function_is_even_l58_5851

theorem derivative_of_odd_function_is_even (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) :
  ∀ x, (deriv f) (-x) = (deriv f) x :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_odd_function_is_even_l58_5851


namespace NUMINAMATH_GPT_correlation_1_and_3_l58_5885

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

end NUMINAMATH_GPT_correlation_1_and_3_l58_5885


namespace NUMINAMATH_GPT_test_methods_first_last_test_methods_within_six_l58_5829

open Classical

def perms (n k : ℕ) : ℕ := sorry -- placeholder for permutation function

theorem test_methods_first_last
  (prod_total : ℕ) (defective : ℕ) (first_test : ℕ) (last_test : ℕ) 
  (A4_2 : ℕ) (A5_2 : ℕ) (A6_4 : ℕ) : first_test = 2 → last_test = 8 → 
  perms 4 2 * perms 5 2 * perms 6 4 = A4_2 * A5_2 * A6_4 :=
by
  intro h_first_test h_last_test
  simp [perms]
  sorry

theorem test_methods_within_six
  (prod_total : ℕ) (defective : ℕ) 
  (A4_4 : ℕ) (A4_3_A6_1 : ℕ) (A5_3_A6_2 : ℕ) (A6_6 : ℕ)
  : perms 4 4 + 4 * perms 4 3 * perms 6 1 + 4 * perms 5 3 * perms 6 2 + perms 6 6 
  = A4_4 + 4 * A4_3_A6_1 + 4 * A5_3_A6_2 + A6_6 :=
by
  simp [perms]
  sorry

end NUMINAMATH_GPT_test_methods_first_last_test_methods_within_six_l58_5829


namespace NUMINAMATH_GPT_volume_invariant_l58_5832

noncomputable def volume_of_common_region (a b c : ℝ) : ℝ := (5/6) * a * b * c

theorem volume_invariant (a b c : ℝ) (P : ℝ × ℝ × ℝ) (hP : ∀ (x y z : ℝ), 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b ∧ 0 ≤ z ∧ z ≤ c) :
  volume_of_common_region a b c = (5/6) * a * b * c :=
by sorry

end NUMINAMATH_GPT_volume_invariant_l58_5832


namespace NUMINAMATH_GPT_rachel_homework_difference_l58_5848

def total_difference (r m h s : ℕ) : ℕ :=
  (r - m) + (s - h)

theorem rachel_homework_difference :
    ∀ (r m h s : ℕ), r = 7 → m = 5 → h = 3 → s = 6 → total_difference r m h s = 5 :=
by
  intros r m h s hr hm hh hs
  rw [hr, hm, hh, hs]
  rfl

end NUMINAMATH_GPT_rachel_homework_difference_l58_5848


namespace NUMINAMATH_GPT_sum_of_last_two_digits_l58_5863

theorem sum_of_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) : (a^15 + b^15) % 100 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_last_two_digits_l58_5863


namespace NUMINAMATH_GPT_united_airlines_discount_l58_5820

theorem united_airlines_discount :
  ∀ (delta_price original_price_u discount_delta discount_u saved_amount cheapest_price: ℝ),
    delta_price = 850 →
    original_price_u = 1100 →
    discount_delta = 0.20 →
    saved_amount = 90 →
    cheapest_price = delta_price * (1 - discount_delta) - saved_amount →
    discount_u = (original_price_u - cheapest_price) / original_price_u →
    discount_u = 0.4636363636 :=
by
  intros delta_price original_price_u discount_delta discount_u saved_amount cheapest_price δeq ueq deq saeq cpeq dueq
  -- Placeholder for the actual proof steps
  sorry

end NUMINAMATH_GPT_united_airlines_discount_l58_5820


namespace NUMINAMATH_GPT_triangle_inequality_from_condition_l58_5823

theorem triangle_inequality_from_condition (a b c : ℝ)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by 
  sorry

end NUMINAMATH_GPT_triangle_inequality_from_condition_l58_5823


namespace NUMINAMATH_GPT_traveler_arrangements_l58_5835

theorem traveler_arrangements :
  let travelers := 6
  let rooms := 3
  ∃ (arrangements : Nat), arrangements = 240 := by
  sorry

end NUMINAMATH_GPT_traveler_arrangements_l58_5835


namespace NUMINAMATH_GPT_crow_eats_quarter_in_twenty_hours_l58_5871

-- Given: The crow eats 1/5 of the nuts in 4 hours
def crow_eating_rate (N : ℕ) : ℕ := N / 5 / 4

-- Prove: It will take 20 hours to eat 1/4 of the nuts
theorem crow_eats_quarter_in_twenty_hours (N : ℕ) (h : ℕ) (h_eq : h = 20) : 
  ((N / 5) / 4 : ℝ) = ((N / 4) / h : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_crow_eats_quarter_in_twenty_hours_l58_5871


namespace NUMINAMATH_GPT_probability_one_male_correct_probability_atleast_one_female_correct_l58_5891

def total_students := 5
def female_students := 2
def male_students := 3
def number_of_selections := 2

noncomputable def probability_only_one_male : ℚ :=
  (6 : ℚ) / 10

noncomputable def probability_atleast_one_female : ℚ :=
  (7 : ℚ) / 10

theorem probability_one_male_correct :
  (6 / 10 : ℚ) = 3 / 5 :=
by
  sorry

theorem probability_atleast_one_female_correct :
  (7 / 10 : ℚ) = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_male_correct_probability_atleast_one_female_correct_l58_5891


namespace NUMINAMATH_GPT_females_in_town_l58_5814

theorem females_in_town (population : ℕ) (ratio : ℕ × ℕ) (H : population = 480) (H_ratio : ratio = (3, 5)) : 
  let m := ratio.1
  let f := ratio.2
  f * (population / (m + f)) = 300 := by
  sorry

end NUMINAMATH_GPT_females_in_town_l58_5814


namespace NUMINAMATH_GPT_people_in_first_group_l58_5826

theorem people_in_first_group (P : ℕ) (work_done_by_P : 60 = 1 / (P * (1/60))) (work_done_by_16 : 30 = 1 / (16 * (1/30))) : P = 8 :=
by
  sorry

end NUMINAMATH_GPT_people_in_first_group_l58_5826


namespace NUMINAMATH_GPT_vibrations_proof_l58_5896

-- Define the conditions
def vibrations_lowest : ℕ := 1600
def increase_percentage : ℕ := 60
def use_time_minutes : ℕ := 5

-- Convert percentage to a multiplier
def percentage_to_multiplier (p : ℕ) : ℤ := (p : ℤ) / 100

-- Calculate the vibrations per second at the highest setting
def vibrations_highest := vibrations_lowest + (vibrations_lowest * percentage_to_multiplier increase_percentage).toNat

-- Convert time from minutes to seconds
def use_time_seconds := use_time_minutes * 60

-- Calculate the total vibrations Matt experiences
noncomputable def total_vibrations : ℕ := vibrations_highest * use_time_seconds

-- State the theorem
theorem vibrations_proof : total_vibrations = 768000 := 
by
  sorry

end NUMINAMATH_GPT_vibrations_proof_l58_5896


namespace NUMINAMATH_GPT_additional_men_joined_l58_5897

theorem additional_men_joined
    (M : ℕ) (X : ℕ)
    (h1 : M = 20)
    (h2 : M * 50 = (M + X) * 25) :
    X = 20 := by
  sorry

end NUMINAMATH_GPT_additional_men_joined_l58_5897


namespace NUMINAMATH_GPT_joan_football_games_l58_5875

theorem joan_football_games (G_total G_last G_this : ℕ) (h1 : G_total = 13) (h2 : G_last = 9) (h3 : G_this = G_total - G_last) : G_this = 4 :=
by
  sorry

end NUMINAMATH_GPT_joan_football_games_l58_5875


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l58_5857

variable (a : ℕ → ℝ) (h : a 1 + a 9 = 10)

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : 
  a 5 = 5 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l58_5857


namespace NUMINAMATH_GPT_correct_formulas_l58_5846

noncomputable def S (a x : ℝ) := (a^x - a^(-x)) / 2
noncomputable def C (a x : ℝ) := (a^x + a^(-x)) / 2

variable {a x y : ℝ}

axiom h1 : a > 0
axiom h2 : a ≠ 1

theorem correct_formulas : S a (x + y) = S a x * C a y + C a x * S a y ∧ S a (x - y) = S a x * C a y - C a x * S a y :=
by 
  sorry

end NUMINAMATH_GPT_correct_formulas_l58_5846


namespace NUMINAMATH_GPT_combined_cost_of_one_item_l58_5889

-- Definitions representing the given conditions
def initial_amount : ℝ := 50
def final_amount : ℝ := 14
def mangoes_purchased : ℕ := 6
def apple_juice_purchased : ℕ := 6

-- Hypothesis: The cost of mangoes and apple juice are the same
variables (M A : ℝ)

-- Total amount spent
def amount_spent : ℝ := initial_amount - final_amount

-- Combined number of items
def total_items : ℕ := mangoes_purchased + apple_juice_purchased

-- Lean statement to prove the combined cost of one mango and one carton of apple juice is $3
theorem combined_cost_of_one_item (h : mangoes_purchased * M + apple_juice_purchased * A = amount_spent) :
    (amount_spent / total_items) = (3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_combined_cost_of_one_item_l58_5889


namespace NUMINAMATH_GPT_find_number_l58_5882

theorem find_number (x : ℝ) (h : x / 0.05 = 900) : x = 45 :=
by sorry

end NUMINAMATH_GPT_find_number_l58_5882


namespace NUMINAMATH_GPT_ball_hits_ground_time_l58_5824

noncomputable def h (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 180

theorem ball_hits_ground_time :
  ∃ t : ℝ, h t = 0 ∧ t = 2.545 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l58_5824


namespace NUMINAMATH_GPT_trapezoid_midsegment_l58_5852

theorem trapezoid_midsegment (a b : ℝ)
  (AB CD E F: ℝ) -- we need to indicate that E and F are midpoints somehow
  (h1 : AB = a)
  (h2 : CD = b)
  (h3 : AB = CD) 
  (h4 : E = (AB + CD) / 2)
  (h5 : F = (CD + AB) / 2) : 
  EF = (1/2) * (a - b) := sorry

end NUMINAMATH_GPT_trapezoid_midsegment_l58_5852


namespace NUMINAMATH_GPT_product_xyz_l58_5815

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 5) : 
  x * y * z = 1 / 9 := 
by
  sorry

end NUMINAMATH_GPT_product_xyz_l58_5815


namespace NUMINAMATH_GPT_find_average_income_of_M_and_O_l58_5818

def average_income_of_M_and_O (M N O : ℕ) : Prop :=
  M + N = 10100 ∧
  N + O = 12500 ∧
  M = 4000 ∧
  (M + O) / 2 = 5200

theorem find_average_income_of_M_and_O (M N O : ℕ):
  average_income_of_M_and_O M N O → 
  (M + O) / 2 = 5200 :=
by
  intro h
  exact h.2.2.2

end NUMINAMATH_GPT_find_average_income_of_M_and_O_l58_5818


namespace NUMINAMATH_GPT_sum_of_coefficients_l58_5856

theorem sum_of_coefficients (a : ℤ) (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (a + x) * (1 + x) ^ 4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_3 + a_5 = 32 →
  a = 3 :=
by sorry

end NUMINAMATH_GPT_sum_of_coefficients_l58_5856


namespace NUMINAMATH_GPT_number_of_yellow_marbles_l58_5845

theorem number_of_yellow_marbles (Y : ℕ) (h : Y / (7 + 11 + Y) = 1 / 4) : Y = 6 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_number_of_yellow_marbles_l58_5845


namespace NUMINAMATH_GPT_saleswoman_commission_l58_5850

theorem saleswoman_commission (S : ℝ)
  (h1 : (S > 500) )
  (h2 : (0.20 * 500 + 0.50 * (S - 500)) = 0.3125 * S) : 
  S = 800 :=
sorry

end NUMINAMATH_GPT_saleswoman_commission_l58_5850


namespace NUMINAMATH_GPT_compute_expression_l58_5825

theorem compute_expression :
  (3 + 3 / 8) ^ (2 / 3) - (5 + 4 / 9) ^ (1 / 2) + 0.008 ^ (2 / 3) / 0.02 ^ (1 / 2) * 0.32 ^ (1 / 2) / 0.0625 ^ (1 / 4) = 43 / 150 := 
sorry

end NUMINAMATH_GPT_compute_expression_l58_5825


namespace NUMINAMATH_GPT_no_two_champions_l58_5870

structure Tournament (Team : Type) :=
  (defeats : Team → Team → Prop)  -- Team A defeats Team B

def is_superior {Team : Type} (T : Tournament Team) (A B: Team) : Prop :=
  T.defeats A B ∨ ∃ C, T.defeats A C ∧ T.defeats C B

def is_champion {Team : Type} (T : Tournament Team) (A : Team) : Prop :=
  ∀ B, A ≠ B → is_superior T A B

theorem no_two_champions {Team : Type} (T : Tournament Team) :
  ¬ (∃ A B, A ≠ B ∧ is_champion T A ∧ is_champion T B) :=
sorry

end NUMINAMATH_GPT_no_two_champions_l58_5870


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l58_5874

theorem solve_eq1 : ∀ (x : ℚ), (3 / 5 - 5 / 8 * x = 2 / 5) → (x = 8 / 25) := by
  intro x
  intro h
  sorry

theorem solve_eq2 : ∀ (x : ℚ), (7 * (x - 2) = 8 * (x - 4)) → (x = 18) := by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l58_5874


namespace NUMINAMATH_GPT_number_called_2009th_position_l58_5809

theorem number_called_2009th_position :
  let sequence := [1, 2, 3, 4, 3, 2]
  ∃ n, n = 2009 → sequence[(2009 % 6) - 1] = 3 := 
by
  -- let sequence := [1, 2, 3, 4, 3, 2]
  -- 2009 % 6 = 5
  -- sequence[4] = 3
  sorry

end NUMINAMATH_GPT_number_called_2009th_position_l58_5809


namespace NUMINAMATH_GPT_distance_origin_to_point_on_parabola_l58_5881

noncomputable def origin : ℝ × ℝ := (0, 0)

noncomputable def parabola_focus (x y : ℝ) : Prop :=
  x^2 = 4 * y ∧ y = 1

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  x^2 = 4 * y

theorem distance_origin_to_point_on_parabola (x y : ℝ) (hx : x^2 = 4 * y)
 (hf : (0, 1) = (0, 1)) (hPF : (x - 0)^2 + (y - 1)^2 = 25) : (x^2 + y^2 = 32) :=
by
  sorry

end NUMINAMATH_GPT_distance_origin_to_point_on_parabola_l58_5881


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l58_5816

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_no_real_roots
  (a b c: ℝ)
  (h1: ((b - 1)^2 - 4 * a * (c + 1) = 0))
  (h2: ((b + 2)^2 - 4 * a * (c - 2) = 0)) :
  ∀ x : ℝ, f a b c x ≠ 0 := 
sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l58_5816


namespace NUMINAMATH_GPT_x_plus_y_eq_3012_plus_pi_div_2_l58_5869

theorem x_plus_y_eq_3012_plus_pi_div_2
  (x y : ℝ)
  (h1 : x + Real.cos y = 3012)
  (h2 : x + 3012 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3012 + Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_x_plus_y_eq_3012_plus_pi_div_2_l58_5869


namespace NUMINAMATH_GPT_machine_present_value_l58_5888

theorem machine_present_value
  (depreciation_rate : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (dep_years : ℕ)
  (value_after_depreciation : ℝ)
  (present_value : ℝ) :

  depreciation_rate = 0.8 →
  selling_price = 118000.00000000001 →
  profit = 22000 →
  dep_years = 2 →
  value_after_depreciation = (selling_price - profit) →
  value_after_depreciation = 96000.00000000001 →
  present_value * (depreciation_rate ^ dep_years) = value_after_depreciation →
  present_value = 150000.00000000002 :=
by sorry

end NUMINAMATH_GPT_machine_present_value_l58_5888


namespace NUMINAMATH_GPT_alok_paid_rs_811_l58_5840

/-
 Assume Alok ordered the following items at the given prices:
 - 16 chapatis, each costing Rs. 6
 - 5 plates of rice, each costing Rs. 45
 - 7 plates of mixed vegetable, each costing Rs. 70
 - 6 ice-cream cups

 Prove that the total cost Alok paid is Rs. 811.
-/
theorem alok_paid_rs_811 :
  let chapati_cost := 6
  let rice_plate_cost := 45
  let mixed_vegetable_plate_cost := 70
  let chapatis := 16 * chapati_cost
  let rice_plates := 5 * rice_plate_cost
  let mixed_vegetable_plates := 7 * mixed_vegetable_plate_cost
  chapatis + rice_plates + mixed_vegetable_plates = 811 := by
  sorry

end NUMINAMATH_GPT_alok_paid_rs_811_l58_5840


namespace NUMINAMATH_GPT_parabola_locus_l58_5855

variables (a c k : ℝ) (a_pos : 0 < a) (c_pos : 0 < c) (k_pos : 0 < k)

theorem parabola_locus :
  ∀ t : ℝ, ∃ x y : ℝ,
    x = -kt / (2 * a) ∧ y = - k^2 * t^2 / (4 * a) + c ∧
    y = - (k^2 / (4 * a)) * x^2 + c :=
sorry

end NUMINAMATH_GPT_parabola_locus_l58_5855


namespace NUMINAMATH_GPT_isosceles_triangle_ratio_HD_HA_l58_5862

theorem isosceles_triangle_ratio_HD_HA (A B C D H : ℝ) :
  let AB := 13;
  let AC := 13;
  let BC := 10;
  let s := (AB + AC + BC) / 2;
  let area := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC));
  let h := (2 * area) / BC;
  let AD := h;
  let HA := h;
  let HD := 0;
  HD / HA = 0 := sorry

end NUMINAMATH_GPT_isosceles_triangle_ratio_HD_HA_l58_5862


namespace NUMINAMATH_GPT_no_rational_solutions_l58_5873

theorem no_rational_solutions : 
  ¬ ∃ (x y z : ℚ), 11 = x^5 + 2 * y^5 + 5 * z^5 := 
sorry

end NUMINAMATH_GPT_no_rational_solutions_l58_5873


namespace NUMINAMATH_GPT_sum_of_digits_18_to_21_l58_5868

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_18_to_21 : 
  (sum_digits 18 + sum_digits 19 + sum_digits 20 + sum_digits 21) = 24 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_digits_18_to_21_l58_5868


namespace NUMINAMATH_GPT_solution_set_for_inequality_l58_5879

open Set Real

theorem solution_set_for_inequality : 
  { x : ℝ | (2 * x) / (x + 1) ≤ 1 } = Ioc (-1 : ℝ) 1 := 
sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l58_5879


namespace NUMINAMATH_GPT_students_exceed_hamsters_l58_5886

-- Definitions corresponding to the problem conditions
def students_per_classroom : ℕ := 20
def hamsters_per_classroom : ℕ := 1
def number_of_classrooms : ℕ := 5

-- Lean 4 statement to express the problem
theorem students_exceed_hamsters :
  (students_per_classroom * number_of_classrooms) - (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end NUMINAMATH_GPT_students_exceed_hamsters_l58_5886


namespace NUMINAMATH_GPT_library_width_l58_5842

theorem library_width 
  (num_libraries : ℕ) 
  (length_per_library : ℕ) 
  (total_area_km2 : ℝ) 
  (conversion_factor : ℝ) 
  (total_area : ℝ) 
  (area_of_one_library : ℝ) 
  (width_of_library : ℝ) :

  num_libraries = 8 →
  length_per_library = 300 →
  total_area_km2 = 0.6 →
  conversion_factor = 1000000 →
  total_area = total_area_km2 * conversion_factor →
  area_of_one_library = total_area / num_libraries →
  width_of_library = area_of_one_library / length_per_library →
  width_of_library = 250 :=
by
  intros;
  sorry

end NUMINAMATH_GPT_library_width_l58_5842


namespace NUMINAMATH_GPT_equation_has_three_distinct_solutions_iff_l58_5893

theorem equation_has_three_distinct_solutions_iff (a : ℝ) : 
  (∃ x_1 x_2 x_3 : ℝ, x_1 ≠ x_2 ∧ x_2 ≠ x_3 ∧ x_1 ≠ x_3 ∧ 
    (x_1 * |x_1 - a| = 1) ∧ (x_2 * |x_2 - a| = 1) ∧ (x_3 * |x_3 - a| = 1)) ↔ a > 2 :=
by
  sorry


end NUMINAMATH_GPT_equation_has_three_distinct_solutions_iff_l58_5893


namespace NUMINAMATH_GPT_sqrt_of_4_l58_5876

theorem sqrt_of_4 (y : ℝ) : y^2 = 4 → (y = 2 ∨ y = -2) :=
sorry

end NUMINAMATH_GPT_sqrt_of_4_l58_5876


namespace NUMINAMATH_GPT_smallest_3a_plus_1_l58_5839

theorem smallest_3a_plus_1 (a : ℝ) (h : 8 * a ^ 2 + 6 * a + 2 = 4) : 
  ∃ a, (8 * a ^ 2 + 6 * a + 2 = 4) ∧ min (3 * (-1) + 1) (3 * (1 / 4) + 1) = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_3a_plus_1_l58_5839


namespace NUMINAMATH_GPT_days_kept_first_book_l58_5828

def cost_per_day : ℝ := 0.50
def total_days_in_may : ℝ := 31
def total_cost_paid : ℝ := 41

theorem days_kept_first_book (x : ℝ) : 0.50 * x + 2 * (0.50 * 31) = 41 → x = 20 :=
by sorry

end NUMINAMATH_GPT_days_kept_first_book_l58_5828


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l58_5819

-- Definitions and conditions
def radius (r : ℝ) := r = 3
def slant_height (l : ℝ) := l = 5
def lateral_surface_area (A : ℝ) (C : ℝ) (l : ℝ) := A = 0.5 * C * l
def circumference (C : ℝ) (r : ℝ) := C = 2 * Real.pi * r

-- Proof (statement only)
theorem cone_lateral_surface_area :
  ∀ (r l C A : ℝ), 
    radius r → 
    slant_height l → 
    circumference C r → 
    lateral_surface_area A C l → 
    A = 15 * Real.pi := 
by intros; sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l58_5819


namespace NUMINAMATH_GPT_percent_of_a_is_4b_l58_5867

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : (4 * b) / a = 20 / 9 :=
by sorry

end NUMINAMATH_GPT_percent_of_a_is_4b_l58_5867


namespace NUMINAMATH_GPT_time_to_cover_length_l58_5892

def escalator_speed : ℝ := 8  -- The speed of the escalator in feet per second
def person_speed : ℝ := 2     -- The speed of the person in feet per second
def escalator_length : ℝ := 160 -- The length of the escalator in feet

theorem time_to_cover_length : 
  (escalator_length / (escalator_speed + person_speed) = 16) :=
by 
  sorry

end NUMINAMATH_GPT_time_to_cover_length_l58_5892


namespace NUMINAMATH_GPT_find_Roe_speed_l58_5830

-- Definitions from the conditions
def Teena_speed : ℝ := 55
def time_in_hours : ℝ := 1.5
def initial_distance_difference : ℝ := 7.5
def final_distance_difference : ℝ := 15

-- Main theorem statement
theorem find_Roe_speed (R : ℝ) (h1 : R * time_in_hours + final_distance_difference = Teena_speed * time_in_hours - initial_distance_difference) :
  R = 40 :=
  sorry

end NUMINAMATH_GPT_find_Roe_speed_l58_5830


namespace NUMINAMATH_GPT_total_cost_l58_5843

-- Define the given conditions
def total_tickets : Nat := 10
def discounted_tickets : Nat := 4
def full_price : ℝ := 2.00
def discounted_price : ℝ := 1.60

-- Calculation of the total cost Martin spent
theorem total_cost : (discounted_tickets * discounted_price) + ((total_tickets - discounted_tickets) * full_price) = 18.40 := by
  sorry

end NUMINAMATH_GPT_total_cost_l58_5843


namespace NUMINAMATH_GPT_percent_profit_l58_5847

theorem percent_profit (C S : ℝ) (h : 58 * C = 50 * S) : 
  (S - C) / C * 100 = 16 :=
by
  sorry

end NUMINAMATH_GPT_percent_profit_l58_5847


namespace NUMINAMATH_GPT_tan_half_angle_product_l58_5806

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0) :
  (Real.tan (a / 2)) * (Real.tan (b / 2)) = 5 ∨ (Real.tan (a / 2)) * (Real.tan (b / 2)) = -5 :=
by 
  sorry

end NUMINAMATH_GPT_tan_half_angle_product_l58_5806


namespace NUMINAMATH_GPT_compute_scalar_dot_product_l58_5899

open Matrix 

def vec1 : Fin 2 → ℤ
| 0 => -2
| 1 => 3

def vec2 : Fin 2 → ℤ
| 0 => 4
| 1 => -5

def dot_product (v1 v2 : Fin 2 → ℤ) : ℤ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1)

theorem compute_scalar_dot_product :
  3 * dot_product vec1 vec2 = -69 := 
by 
  sorry

end NUMINAMATH_GPT_compute_scalar_dot_product_l58_5899


namespace NUMINAMATH_GPT_determine_xyz_l58_5858

theorem determine_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + 1/y = 5) (h5 : y + 1/z = 2) (h6 : z + 1/x = 8/3) :
  x * y * z = 8 + 3 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_determine_xyz_l58_5858


namespace NUMINAMATH_GPT_evaluate_expression_l58_5894

variable (x y : ℝ)

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 - y^2 = x * y) :
  (1 / x^2) - (1 / y^2) = - (1 / (x * y)) :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l58_5894


namespace NUMINAMATH_GPT_fraction_zero_iff_x_neg_one_l58_5865

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h : 1 - |x| = 0) (h_non_zero : 1 - x ≠ 0) : x = -1 :=
sorry

end NUMINAMATH_GPT_fraction_zero_iff_x_neg_one_l58_5865


namespace NUMINAMATH_GPT_first_even_number_l58_5811

theorem first_even_number (x : ℤ) (h : x + (x + 2) + (x + 4) = 1194) : x = 396 :=
by
  -- the proof is skipped as per instructions
  sorry

end NUMINAMATH_GPT_first_even_number_l58_5811


namespace NUMINAMATH_GPT_product_of_possible_values_N_l58_5890

theorem product_of_possible_values_N 
  (L M : ℤ) 
  (h1 : M = L + N) 
  (h2 : M - 7 = L + N - 7)
  (h3 : L + 5 = L + 5)
  (h4 : |(L + N - 7) - (L + 5)| = 4) : 
  N = 128 := 
  sorry

end NUMINAMATH_GPT_product_of_possible_values_N_l58_5890


namespace NUMINAMATH_GPT_find_m_l58_5898

-- Define the set A
def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3 * m + 2}

-- Main theorem statement
theorem find_m (m : ℝ) (h : 2 ∈ A m) : m = 3 := by
  sorry

end NUMINAMATH_GPT_find_m_l58_5898


namespace NUMINAMATH_GPT_max_S_2017_l58_5800

noncomputable def max_S (a b c : ℕ) : ℕ := a + b + c

theorem max_S_2017 :
  ∀ (a b c : ℕ),
  a + b = 1014 →
  c - b = 497 →
  a > b →
  max_S a b c = 2017 :=
by
  intros a b c h1 h2 h3
  sorry

end NUMINAMATH_GPT_max_S_2017_l58_5800


namespace NUMINAMATH_GPT_ancient_chinese_silver_problem_l58_5884

theorem ancient_chinese_silver_problem :
  ∃ (x y : ℤ), 7 * x = y - 4 ∧ 9 * x = y + 8 :=
by
  sorry

end NUMINAMATH_GPT_ancient_chinese_silver_problem_l58_5884


namespace NUMINAMATH_GPT_find_number_l58_5810

theorem find_number (x : ℝ) : (45 * x = 0.45 * 900) → (x = 9) :=
by sorry

end NUMINAMATH_GPT_find_number_l58_5810


namespace NUMINAMATH_GPT_cookies_recipes_count_l58_5821

theorem cookies_recipes_count 
  (total_students : ℕ)
  (attending_percentage : ℚ)
  (cookies_per_student : ℕ)
  (cookies_per_batch : ℕ) : 
  (total_students = 150) →
  (attending_percentage = 0.60) →
  (cookies_per_student = 3) →
  (cookies_per_batch = 18) →
  (total_students * attending_percentage * cookies_per_student / cookies_per_batch = 15) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cookies_recipes_count_l58_5821


namespace NUMINAMATH_GPT_Alan_has_eight_pine_trees_l58_5864

noncomputable def number_of_pine_trees (total_pine_cones_per_tree : ℕ) (percentage_on_roof : ℚ) 
                                       (weight_per_pine_cone : ℚ) (total_weight_on_roof : ℚ) : ℚ :=
  total_weight_on_roof / (total_pine_cones_per_tree * percentage_on_roof * weight_per_pine_cone)

theorem Alan_has_eight_pine_trees :
  number_of_pine_trees 200 (30 / 100) 4 1920 = 8 :=
by
  sorry

end NUMINAMATH_GPT_Alan_has_eight_pine_trees_l58_5864


namespace NUMINAMATH_GPT_a_b_sum_of_powers_l58_5844

variable (a b : ℝ)

-- Conditions
def condition1 := a + b = 1
def condition2 := a^2 + b^2 = 3
def condition3 := a^3 + b^3 = 4
def condition4 := a^4 + b^4 = 7
def condition5 := a^5 + b^5 = 11

-- Theorem statement
theorem a_b_sum_of_powers (h1 : condition1 a b) (h2 : condition2 a b) (h3 : condition3 a b) 
  (h4 : condition4 a b) (h5 : condition5 a b) : a^10 + b^10 = 123 :=
sorry

end NUMINAMATH_GPT_a_b_sum_of_powers_l58_5844


namespace NUMINAMATH_GPT_a_b_c_relationship_l58_5804

noncomputable def a (f : ℝ → ℝ) : ℝ := 25 * f (0.2^2)
noncomputable def b (f : ℝ → ℝ) : ℝ := f 1
noncomputable def c (f : ℝ → ℝ) : ℝ := - (Real.log 3 / Real.log 5) * f (Real.log 5 / Real.log 3)

axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom decreasing_g (f : ℝ → ℝ) : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → 0 < x2 → (f x1 / x1) > (f x2 / x2)

theorem a_b_c_relationship (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_decreasing : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → 0 < x2 → (f x1 / x1) > (f x2 / x2)) :
  a f > b f ∧ b f > c f :=
sorry

end NUMINAMATH_GPT_a_b_c_relationship_l58_5804


namespace NUMINAMATH_GPT_correct_option_C_l58_5834

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complements of sets A and B in U
def complA : Set ℕ := {2, 4}
def complB : Set ℕ := {3, 4}

-- Define sets A and B using the complements
def A : Set ℕ := U \ complA
def B : Set ℕ := U \ complB

-- Mathematical proof problem statement
theorem correct_option_C : 3 ∈ A ∧ 3 ∉ B := by
  sorry

end NUMINAMATH_GPT_correct_option_C_l58_5834


namespace NUMINAMATH_GPT_greatest_value_of_4a_l58_5837

-- Definitions of the given conditions
def hundreds_digit (x : ℕ) : ℕ := x / 100
def tens_digit (x : ℕ) : ℕ := (x / 10) % 10
def units_digit (x : ℕ) : ℕ := x % 10

def satisfies_conditions (a b c x : ℕ) : Prop :=
  hundreds_digit x = a ∧
  tens_digit x = b ∧
  units_digit x = c ∧
  4 * a = 2 * b ∧
  2 * b = c ∧
  a > 0

def difference_of_two_greatest_x : ℕ := 124

theorem greatest_value_of_4a (x1 x2 a1 a2 b1 b2 c1 c2 : ℕ) :
  satisfies_conditions a1 b1 c1 x1 →
  satisfies_conditions a2 b2 c2 x2 →
  x1 - x2 = difference_of_two_greatest_x →
  4 * a1 = 8 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_of_4a_l58_5837


namespace NUMINAMATH_GPT_sum_of_squares_xy_l58_5827

theorem sum_of_squares_xy (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_xy_l58_5827


namespace NUMINAMATH_GPT_simplify_expression_l58_5841

theorem simplify_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
    a * (1 / b + 1 / c) + b * (1 / a + 1 / c) + c * (1 / a + 1 / b) = -3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l58_5841


namespace NUMINAMATH_GPT_equilateral_triangle_l58_5808

theorem equilateral_triangle (a b c : ℝ) (h1 : a^4 = b^4 + c^4 - b^2 * c^2) (h2 : b^4 = a^4 + c^4 - a^2 * c^2) : 
  a = b ∧ b = c ∧ c = a :=
by sorry

end NUMINAMATH_GPT_equilateral_triangle_l58_5808


namespace NUMINAMATH_GPT_pay_per_task_l58_5861

def tasks_per_day : ℕ := 100
def days_per_week : ℕ := 6
def weekly_pay : ℕ := 720

theorem pay_per_task :
  (weekly_pay : ℚ) / (tasks_per_day * days_per_week) = 1.20 := 
sorry

end NUMINAMATH_GPT_pay_per_task_l58_5861


namespace NUMINAMATH_GPT_relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l58_5860

-- Prove that w - 2z = 0
theorem relation_w_z (w z : ℝ) : w - 2 * z = 0 :=
sorry

-- Prove that 2s + t - 8 = 0
theorem relation_s_t (s t : ℝ) : 2 * s + t - 8 = 0 :=
sorry

-- Prove that x - r - 2 = 0
theorem relation_x_r (x r : ℝ) : x - r - 2 = 0 :=
sorry

-- Prove that y + q - 6 = 0
theorem relation_y_q (y q : ℝ) : y + q - 6 = 0 :=
sorry

-- Prove that 3z - x - 2t + 6 = 0
theorem relation_z_x_t (z x t : ℝ) : 3 * z - x - 2 * t + 6 = 0 :=
sorry

-- Prove that 8z - 4t - v + 12 = 0
theorem relation_z_t_v (z t v : ℝ) : 8 * z - 4 * t - v + 12 = 0 :=
sorry

end NUMINAMATH_GPT_relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l58_5860


namespace NUMINAMATH_GPT_number_of_points_l58_5802

theorem number_of_points (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 :=
by
  -- Proof to be done here
  sorry

end NUMINAMATH_GPT_number_of_points_l58_5802


namespace NUMINAMATH_GPT_isabella_houses_problem_l58_5822

theorem isabella_houses_problem 
  (yellow green red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  (green + red = 160) := 
sorry

end NUMINAMATH_GPT_isabella_houses_problem_l58_5822


namespace NUMINAMATH_GPT_eat_cereal_in_time_l58_5812

noncomputable def time_to_eat_pounds (pounds : ℕ) (rate1 rate2 : ℚ) :=
  pounds / (rate1 + rate2)

theorem eat_cereal_in_time :
  time_to_eat_pounds 5 ((1:ℚ)/15) ((1:ℚ)/40) = 600/11 := 
by 
  sorry

end NUMINAMATH_GPT_eat_cereal_in_time_l58_5812


namespace NUMINAMATH_GPT_cheburashkas_erased_l58_5836

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end NUMINAMATH_GPT_cheburashkas_erased_l58_5836


namespace NUMINAMATH_GPT_area_BCD_l58_5883

open Real EuclideanGeometry

noncomputable def point := (ℝ × ℝ)
noncomputable def A : point := (0, 0)
noncomputable def B : point := (10, 24)
noncomputable def C : point := (30, 0)
noncomputable def D : point := (40, 0)

def area_triangle (p1 p2 p3 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * |x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)|

theorem area_BCD : area_triangle B C D = 12 := sorry

end NUMINAMATH_GPT_area_BCD_l58_5883


namespace NUMINAMATH_GPT_max_objective_value_l58_5880

theorem max_objective_value (x y : ℝ) (h1 : x - y - 2 ≥ 0) (h2 : 2 * x + y - 2 ≤ 0) (h3 : y + 4 ≥ 0) :
  ∃ (z : ℝ), z = 4 * x + 3 * y ∧ z ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_objective_value_l58_5880


namespace NUMINAMATH_GPT_divisor_of_p_l58_5866

theorem divisor_of_p (p q r s : ℕ) (hpq : Nat.gcd p q = 40)
  (hqr : Nat.gcd q r = 45) (hrs : Nat.gcd r s = 60)
  (hspr : 100 < Nat.gcd s p ∧ Nat.gcd s p < 150)
  : 7 ∣ p :=
sorry

end NUMINAMATH_GPT_divisor_of_p_l58_5866
