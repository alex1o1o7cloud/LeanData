import Mathlib

namespace angle_division_quadrant_l17_17194

variable (k : ℤ)
variable (α : ℝ)
variable (h : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi)

theorem angle_division_quadrant 
  (hα_sec_quadrant : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi) : 
  (∃ m : ℤ, (m = 0 ∧ Real.pi / 4 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
             (m = 1 ∧ Real.pi * (2 * m + 1) / 4 < α / 2 ∧ α / 2 < Real.pi * (2 * m + 1) / 2)) :=
sorry

end angle_division_quadrant_l17_17194


namespace num_even_digit_numbers_l17_17052

open Nat

theorem num_even_digit_numbers (S : Finset ℕ) (hs : S = {1, 2, 3, 4, 5}) : 
  ∃ n, n = 48 ∧ ∀ (N : ℕ), 
    (∀ d ∈ S, ∃! l : List ℕ, l.to_finset = S ∧ l.length = 5 ∧ 
    ∃ k ∈ l, k % 2 = 0 ∧ N = list_to_nat l) → (N = n) :=
by sorry

end num_even_digit_numbers_l17_17052


namespace interior_angle_regular_octagon_l17_17796

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l17_17796


namespace inequality_holds_for_all_real_numbers_l17_17245

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l17_17245


namespace days_gumballs_last_l17_17726

def pairs_day_1 := 3
def gumballs_per_pair := 9
def gumballs_day_1 := pairs_day_1 * gumballs_per_pair

def pairs_day_2 := pairs_day_1 * 2
def gumballs_day_2 := pairs_day_2 * gumballs_per_pair

def pairs_day_3 := pairs_day_2 - 1
def gumballs_day_3 := pairs_day_3 * gumballs_per_pair

def total_gumballs := gumballs_day_1 + gumballs_day_2 + gumballs_day_3
def gumballs_eaten_per_day := 3

theorem days_gumballs_last : total_gumballs / gumballs_eaten_per_day = 42 :=
by
  sorry

end days_gumballs_last_l17_17726


namespace total_minutes_worked_l17_17444

def minutesWorked : Nat :=
  -- Monday
  let monday := 45 + 30
  -- Tuesday
  let tuesday := 90 + 45 - 15
  -- Wednesday
  let wednesday := 40 + 60
  -- Thursday
  let thursday := 90 + 75 - 30
  -- Friday
  let friday := 55 + 20
  -- Saturday
  let saturday := 120 + 60 - 40
  -- Sunday
  let sunday := 105 + 135 - 45
  -- Total
  let total := monday + tuesday + wednesday + thursday + friday + saturday + sunday
  total

theorem total_minutes_worked : minutesWorked = 840 := by
  sorry

end total_minutes_worked_l17_17444


namespace age_of_B_l17_17640

variables (A B C : ℝ)

theorem age_of_B :
  (A + B + C) / 3 = 26 →
  (A + C) / 2 = 29 →
  B = 20 :=
by
  intro h1 h2
  sorry

end age_of_B_l17_17640


namespace range_of_2a_minus_b_l17_17549

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : 2 < b) (h4 : b < 4) : 
  -2 < 2 * a - b ∧ 2 * a - b < 4 := 
by 
  sorry

end range_of_2a_minus_b_l17_17549


namespace regular_octagon_interior_angle_measure_l17_17778

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l17_17778


namespace regular_octagon_interior_angle_l17_17806

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17806


namespace floor_sqrt_80_l17_17917

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l17_17917


namespace cos_square_theta_plus_pi_over_4_eq_one_fourth_l17_17542

variable (θ : ℝ)

theorem cos_square_theta_plus_pi_over_4_eq_one_fourth
  (h : Real.tan θ + 1 / Real.tan θ = 4) :
  Real.cos (θ + Real.pi / 4) ^ 2 = 1 / 4 :=
sorry

end cos_square_theta_plus_pi_over_4_eq_one_fourth_l17_17542


namespace smallest_positive_and_largest_negative_l17_17771

theorem smallest_positive_and_largest_negative:
  (∃ (a : ℤ), a > 0 ∧ ∀ (b : ℤ), b > 0 → b ≥ a ∧ a = 1) ∧
  (∃ (c : ℤ), c < 0 ∧ ∀ (d : ℤ), d < 0 → d ≤ c ∧ c = -1) :=
by
  sorry

end smallest_positive_and_largest_negative_l17_17771


namespace original_price_of_pants_l17_17370

theorem original_price_of_pants (P : ℝ) 
  (sale_discount : ℝ := 0.50)
  (saturday_additional_discount : ℝ := 0.20)
  (savings : ℝ := 50.40)
  (saturday_effective_discount : ℝ := 0.40) :
  savings = 0.60 * P ↔ P = 84.00 :=
by
  sorry

end original_price_of_pants_l17_17370


namespace calculate_sum_of_squares_l17_17956

variables {a b : ℤ}
theorem calculate_sum_of_squares (h1 : (a + b)^2 = 17) (h2 : (a - b)^2 = 11) : a^2 + b^2 = 14 :=
by
  sorry

end calculate_sum_of_squares_l17_17956


namespace ratio_area_ADE_BCED_is_8_over_9_l17_17087

noncomputable def ratio_area_ADE_BCED 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) : ℝ := 
  sorry

theorem ratio_area_ADE_BCED_is_8_over_9 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) :
  ratio_area_ADE_BCED AB BC AC AD AE hAB hBC hAC hAD hAE = 8 / 9 :=
  sorry

end ratio_area_ADE_BCED_is_8_over_9_l17_17087


namespace problem_statement_l17_17167

namespace CoinFlipping

/-- 
Define the probability that Alice and Bob both get the same number of heads
when flipping three coins where two are fair and one is biased with a probability
of 3/5 for heads. We aim to calculate p + q where p/q is this probability and 
output the final result - p + q should equal 263.
-/
def same_heads_probability_sum : ℕ :=
  let p := 63
  let q := 200
  p + q

theorem problem_statement : same_heads_probability_sum = 263 :=
  by
  -- proof to be filled in
  sorry

end CoinFlipping

end problem_statement_l17_17167


namespace inequality_proof_l17_17281

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l17_17281


namespace bridge_weight_excess_l17_17159

theorem bridge_weight_excess :
  ∀ (Kelly_weight Megan_weight Mike_weight : ℕ),
  Kelly_weight = 34 →
  Kelly_weight = 85 * Megan_weight / 100 →
  Mike_weight = Megan_weight + 5 →
  (Kelly_weight + Megan_weight + Mike_weight) - 100 = 19 :=
by
  intros Kelly_weight Megan_weight Mike_weight
  intros h1 h2 h3
  sorry

end bridge_weight_excess_l17_17159


namespace smallest_sum_B_c_l17_17560

theorem smallest_sum_B_c (B : ℕ) (c : ℕ) (hB : B < 5) (hc : c > 6) :
  31 * B = 4 * c + 4 → (B + c) = 34 :=
by
  sorry

end smallest_sum_B_c_l17_17560


namespace interest_rate_is_4_l17_17022

-- Define the conditions based on the problem statement
def principal : ℕ := 500
def time : ℕ := 8
def simple_interest : ℕ := 160

-- Assuming the formula for simple interest
def simple_interest_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- The interest rate we aim to prove
def interest_rate : ℕ := 4

-- The statement we want to prove: Given the conditions, the interest rate is 4%
theorem interest_rate_is_4 : simple_interest_formula principal interest_rate time = simple_interest := by
  -- The proof steps would go here
  sorry

end interest_rate_is_4_l17_17022


namespace odd_prime_does_not_divide_odd_nat_number_increment_l17_17734

theorem odd_prime_does_not_divide_odd_nat_number_increment (p n : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_odd : n % 2 = 1) :
  ¬ (p * n + 1 ∣ p ^ p - 1) :=
by
  sorry

end odd_prime_does_not_divide_odd_nat_number_increment_l17_17734


namespace solve_for_x_l17_17989

theorem solve_for_x (x : ℝ) : 
  x^2 - 2 * x - 8 = -(x + 2) * (x - 6) → (x = 5 ∨ x = -2) :=
by
  intro h
  sorry

end solve_for_x_l17_17989


namespace sugar_initial_weight_l17_17316

theorem sugar_initial_weight (packs : ℕ) (pack_weight : ℕ) (leftover : ℕ) (used_percentage : ℝ)
  (h1 : packs = 30)
  (h2 : pack_weight = 350)
  (h3 : leftover = 50)
  (h4 : used_percentage = 0.60) : 
  (packs * pack_weight + leftover) = 10550 :=
by 
  sorry

end sugar_initial_weight_l17_17316


namespace sara_peaches_l17_17757

theorem sara_peaches (initial_peaches : ℕ) (picked_peaches : ℕ) (total_peaches : ℕ) 
  (h1 : initial_peaches = 24) (h2 : picked_peaches = 37) : 
  total_peaches = 61 :=
by
  sorry

end sara_peaches_l17_17757


namespace max_quotient_l17_17991

-- Define the given conditions
def conditions (a b : ℝ) :=
  100 ≤ a ∧ a ≤ 250 ∧ 700 ≤ b ∧ b ≤ 1400

-- State the theorem for the largest value of the quotient b / a
theorem max_quotient (a b : ℝ) (h : conditions a b) : b / a ≤ 14 :=
by
  sorry

end max_quotient_l17_17991


namespace inequality_holds_for_real_numbers_l17_17269

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l17_17269


namespace chocolates_brought_by_friend_l17_17446

-- Definitions corresponding to the conditions in a)
def total_chocolates := 50
def chocolates_not_in_box := 5
def number_of_boxes := 3
def additional_boxes := 2

-- Theorem statement: we need to prove the number of chocolates her friend brought
theorem chocolates_brought_by_friend (C : ℕ) : 
  (C + total_chocolates = total_chocolates + (chocolates_not_in_box + number_of_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes + additional_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes) - total_chocolates) 
  → C = 30 := 
sorry

end chocolates_brought_by_friend_l17_17446


namespace min_value_fraction_sum_l17_17392

theorem min_value_fraction_sum : 
  ∀ (n : ℕ), n > 0 → (n / 3 + 27 / n) ≥ 6 :=
by
  sorry

end min_value_fraction_sum_l17_17392


namespace determine_m_l17_17205

open Set Real

theorem determine_m (m : ℝ) : (∀ x, x ∈ { x | x ≥ 3 } ∪ { x | x < m }) ∧ (∀ x, x ∉ { x | x ≥ 3 } ∩ { x | x < m }) → m = 3 :=
by
  intros h
  sorry

end determine_m_l17_17205


namespace haley_initial_music_files_l17_17700

theorem haley_initial_music_files (M : ℕ) 
  (h1 : M + 42 - 11 = 58) : M = 27 := 
by
  sorry

end haley_initial_music_files_l17_17700


namespace slope_of_line_eq_slope_of_line_l17_17185

theorem slope_of_line_eq (x y : ℝ) (h : 4 * x + 6 * y = 24) : (6 * y = -4 * x + 24) → (y = - (2 : ℝ) / 3 * x + 4) :=
by
  intro h1
  sorry

theorem slope_of_line (x y m : ℝ) (h1 : 4 * x + 6 * y = 24) (h2 : y = - (2 : ℝ) / 3 * x + 4) : m = - (2 : ℝ) / 3 :=
by
  sorry

end slope_of_line_eq_slope_of_line_l17_17185


namespace tenth_term_l17_17357

noncomputable def sequence_term (n : ℕ) : ℝ :=
  (-1)^(n+1) * (Real.sqrt (1 + 2*(n - 1))) / (2^n)

theorem tenth_term :
  sequence_term 10 = Real.sqrt 19 / (2^10) :=
by
  sorry

end tenth_term_l17_17357


namespace floor_neg_seven_fourths_l17_17889

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l17_17889


namespace grade_on_second_test_l17_17371

variable (first_test_grade second_test_average : ℕ)
#check first_test_grade
#check second_test_average

theorem grade_on_second_test :
  first_test_grade = 78 →
  second_test_average = 81 →
  (first_test_grade + (second_test_average * 2 - first_test_grade)) / 2 = second_test_average →
  second_test_grade = 84 :=
by
  intros h1 h2 h3
  sorry

end grade_on_second_test_l17_17371


namespace minimum_trips_needed_l17_17011

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def capacity : ℕ := 200

theorem minimum_trips_needed (masses : List ℕ) (capacity : ℕ) : 
  masses = [150, 60, 70, 71, 72, 100, 101, 102, 103] →
  capacity = 200 →
  ∃ trips : ℕ, trips = 5 :=
by
  sorry

end minimum_trips_needed_l17_17011


namespace miranda_saved_per_month_l17_17588

-- Definition of the conditions and calculation in the problem
def total_cost : ℕ := 260
def sister_contribution : ℕ := 50
def months : ℕ := 3
def miranda_savings : ℕ := total_cost - sister_contribution
def saved_per_month : ℕ := miranda_savings / months

-- Theorem statement with the expected answer
theorem miranda_saved_per_month : saved_per_month = 70 :=
by
  sorry

end miranda_saved_per_month_l17_17588


namespace remainder_is_three_l17_17051

def P (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem remainder_is_three : P 1 = 3 :=
by
  -- Proof goes here.
  sorry

end remainder_is_three_l17_17051


namespace sara_letters_ratio_l17_17758

variable (L_J : ℕ) (L_F : ℕ) (L_T : ℕ)

theorem sara_letters_ratio (hLJ : L_J = 6) (hLF : L_F = 9) (hLT : L_T = 33) : 
  (L_T - (L_J + L_F)) / L_J = 3 := by
  sorry

end sara_letters_ratio_l17_17758


namespace total_blood_cells_correct_l17_17655

def first_sample : ℕ := 4221
def second_sample : ℕ := 3120
def total_blood_cells : ℕ := first_sample + second_sample

theorem total_blood_cells_correct : total_blood_cells = 7341 := by
  -- proof goes here
  sorry

end total_blood_cells_correct_l17_17655


namespace domain_of_k_l17_17760

noncomputable def domain_of_h := Set.Icc (-10 : ℝ) 6

def h (x : ℝ) : Prop := x ∈ domain_of_h
def k (x : ℝ) : Prop := h (-3 * x + 1)

theorem domain_of_k : ∀ x : ℝ, k x ↔ x ∈ Set.Icc (-5/3) (11/3) :=
by
  intro x
  change (-3 * x + 1 ∈ Set.Icc (-10 : ℝ) 6) ↔ (x ∈ Set.Icc (-5/3 : ℝ) (11/3))
  sorry

end domain_of_k_l17_17760


namespace sum_200_to_299_l17_17210

variable (a : ℕ)

-- Condition: Sum of the first 100 natural numbers is equal to a
def sum_100 := (100 * 101) / 2

-- Main Theorem: Sum from 200 to 299 in terms of a
theorem sum_200_to_299 (h : sum_100 = a) : (299 * 300 / 2 - 199 * 200 / 2) = 19900 + a := by
  sorry

end sum_200_to_299_l17_17210


namespace delta_comparison_eps_based_on_gamma_l17_17227

-- Definitions for the problem
variable {α β γ δ ε : ℝ}
variable {A B C : Type}
variable (s f m : Type)

-- Conditions from problem
variable (triangle_ABC : α ≠ β)
variable (median_s_from_C : s)
variable (angle_bisector_f : f)
variable (altitude_m : m)
variable (angle_between_f_m : δ = sorry)
variable (angle_between_f_s : ε = sorry)
variable (angle_at_vertex_C : γ = sorry)

-- Main statement to prove
theorem delta_comparison_eps_based_on_gamma (h1 : α ≠ β) (h2 : δ = sorry) (h3 : ε = sorry) (h4 : γ = sorry) :
  if γ < 90 then δ < ε else if γ = 90 then δ = ε else δ > ε :=
sorry

end delta_comparison_eps_based_on_gamma_l17_17227


namespace exterior_angle_hexagon_l17_17693

theorem exterior_angle_hexagon (θ : ℝ) (hθ : θ = 60) (h_sum : θ * 6 = 360) : n = 6 :=
sorry

end exterior_angle_hexagon_l17_17693


namespace not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l17_17574

theorem not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles : 
  ¬ ∃ (rectangles : ℕ × ℕ), rectangles.1 = 1 ∧ rectangles.2 = 7 ∧ rectangles.1 * 4 + rectangles.2 * 3 = 25 :=
by
  sorry

end not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l17_17574


namespace floor_sqrt_80_l17_17905

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l17_17905


namespace smallest_n_l17_17002

theorem smallest_n (n : ℕ) :
  (1 / 4 : ℚ) + (n / 8 : ℚ) > 1 ↔ n ≥ 7 := by
  sorry

end smallest_n_l17_17002


namespace larger_of_two_numbers_l17_17770

theorem larger_of_two_numbers
  (A B hcf : ℕ)
  (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 9)
  (h_factor2 : factor2 = 10)
  (h_lcm : (A * B) / (hcf) = (hcf * factor1 * factor2))
  (h_A : A = hcf * 9)
  (h_B : B = hcf * 10) :
  max A B = 230 := by
  sorry

end larger_of_two_numbers_l17_17770


namespace correct_calculation_l17_17004

theorem correct_calculation :
  ∃ (a : ℤ), (a^2 + a^2 = 2 * a^2) ∧ 
  (¬(3*a + 4*(a : ℤ) = 12*a*(a : ℤ))) ∧ 
  (¬((a*(a : ℤ)^2)^3 = a*(a : ℤ)^6)) ∧ 
  (¬((a + 3)^2 = a^2 + 9)) :=
by
  sorry

end correct_calculation_l17_17004


namespace fg_of_5_eq_163_l17_17071

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem fg_of_5_eq_163 : f (g 5) = 163 :=
by
  sorry

end fg_of_5_eq_163_l17_17071


namespace regular_octagon_interior_angle_l17_17807

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17807


namespace rotted_tomatoes_is_correct_l17_17095

noncomputable def shipment_1 : ℕ := 1000
noncomputable def sold_Saturday : ℕ := 300
noncomputable def shipment_2 : ℕ := 2 * shipment_1
noncomputable def tomatoes_Tuesday : ℕ := 2500

-- Define remaining tomatoes after the first shipment accounting for Saturday's sales
noncomputable def remaining_tomatoes_1 : ℕ := shipment_1 - sold_Saturday

-- Define total tomatoes after second shipment arrives
noncomputable def total_tomatoes_after_second_shipment : ℕ := remaining_tomatoes_1 + shipment_2

-- Define the amount of tomatoes that rotted
noncomputable def rotted_tomatoes : ℕ :=
  total_tomatoes_after_second_shipment - tomatoes_Tuesday

theorem rotted_tomatoes_is_correct :
  rotted_tomatoes = 200 := by
  sorry

end rotted_tomatoes_is_correct_l17_17095


namespace interior_angle_regular_octagon_l17_17814

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l17_17814


namespace find_probability_p_l17_17453

noncomputable section

open ProbabilityTheory

variable {Ω : Type*} [MeasureSpace Ω]

def binomial (n : ℕ) (p : ℝ) :=  binom n p

theorem find_probability_p (X Y : Ω → ℕ) (p : ℝ) 
  (hX : ∀ ω, distribute X ω = binomial 2 p) 
  (hY : ∀ ω, distribute Y ω = binomial 3 p) 
  (hPX : ∫ ω, indicator (λ ω, X ω ≥ 1) ω =
         5 / 9) :
  ∫ ω, indicator (λ ω, Y ω ≥ 1) ω = 19 / 27 :=
sorry

end find_probability_p_l17_17453


namespace drum_oil_ratio_l17_17673

theorem drum_oil_ratio (C_X C_Y : ℝ) (h1 : (1 / 2) * C_X + (1 / 5) * C_Y = 0.45 * C_Y) : 
  C_Y / C_X = 2 :=
by
  -- Cannot provide the proof
  sorry

end drum_oil_ratio_l17_17673


namespace reasoning_is_wrong_l17_17470

-- Definitions of the conditions
def some_rationals_are_proper_fractions := ∃ q : ℚ, ∃ f : ℚ, q = f ∧ f.den ≠ 1
def integers_are_rationals := ∀ z : ℤ, ∃ q : ℚ, q = z

-- Proof that the form of reasoning is wrong given the conditions
theorem reasoning_is_wrong 
  (h₁ : some_rationals_are_proper_fractions) 
  (h₂ : integers_are_rationals) :
  ¬ (∀ z : ℤ, ∃ f : ℚ, z = f ∧ f.den ≠ 1) := 
sorry

end reasoning_is_wrong_l17_17470


namespace remainder_of_f_div_x_minus_2_is_48_l17_17848

-- Define the polynomial f(x)
noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 8 * x^3 + 25 * x^2 - 14 * x - 40

-- State the theorem to prove that the remainder of f(x) when divided by x - 2 is 48
theorem remainder_of_f_div_x_minus_2_is_48 : f 2 = 48 :=
by sorry

end remainder_of_f_div_x_minus_2_is_48_l17_17848


namespace interior_angle_of_regular_octagon_l17_17786

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l17_17786


namespace floor_of_sqrt_80_l17_17912

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l17_17912


namespace probability_one_defective_l17_17015

def total_bulbs : ℕ := 20
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs
def probability_non_defective_both : ℚ := (16 / 20) * (15 / 19)
def probability_at_least_one_defective : ℚ := 1 - probability_non_defective_both

theorem probability_one_defective :
  probability_at_least_one_defective = 7 / 19 :=
by
  sorry

end probability_one_defective_l17_17015


namespace total_cost_l17_17236

theorem total_cost
  (cost_berries   : ℝ := 11.08)
  (cost_apples    : ℝ := 14.33)
  (cost_peaches   : ℝ := 9.31)
  (cost_grapes    : ℝ := 7.50)
  (cost_bananas   : ℝ := 5.25)
  (cost_pineapples: ℝ := 4.62)
  (total_cost     : ℝ := cost_berries + cost_apples + cost_peaches + cost_grapes + cost_bananas + cost_pineapples) :
  total_cost = 52.09 :=
by
  sorry

end total_cost_l17_17236


namespace polynomial_quotient_l17_17328

theorem polynomial_quotient : 
  (12 * x^3 + 20 * x^2 - 7 * x + 4) / (3 * x + 4) = 4 * x^2 + (4/3) * x - 37/9 :=
by
  sorry

end polynomial_quotient_l17_17328


namespace cube_roots_not_arithmetic_progression_l17_17451

theorem cube_roots_not_arithmetic_progression
  (p q r : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (h_distinct: p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ¬ ∃ (d : ℝ) (m n : ℤ), (n ≠ m) ∧ (↑q)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (m : ℝ) * d ∧ (↑r)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (n : ℝ) * d :=
by sorry

end cube_roots_not_arithmetic_progression_l17_17451


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17823

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17823


namespace sum_of_reciprocals_l17_17605

theorem sum_of_reciprocals (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
  (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 31 / 21) :=
sorry

end sum_of_reciprocals_l17_17605


namespace greatest_multiple_of_4_less_than_100_l17_17481

theorem greatest_multiple_of_4_less_than_100 : ∃ n : ℕ, n % 4 = 0 ∧ n < 100 ∧ ∀ m : ℕ, (m % 4 = 0 ∧ m < 100) → m ≤ n 
:= by
  sorry

end greatest_multiple_of_4_less_than_100_l17_17481


namespace total_distance_l17_17046

variable {D : ℝ}

theorem total_distance (h1 : D / 3 > 0)
                       (h2 : (2 / 3 * D) - (1 / 6 * D) > 0)
                       (h3 : (1 / 2 * D) - (1 / 10 * D) = 180) :
    D = 450 := 
sorry

end total_distance_l17_17046


namespace roots_squared_sum_l17_17935

theorem roots_squared_sum (x1 x2 : ℝ) (h₁ : x1^2 - 5 * x1 + 3 = 0) (h₂ : x2^2 - 5 * x2 + 3 = 0) :
  x1^2 + x2^2 = 19 :=
by
  sorry

end roots_squared_sum_l17_17935


namespace apps_more_than_files_l17_17389

theorem apps_more_than_files
  (initial_apps : ℕ)
  (initial_files : ℕ)
  (deleted_apps : ℕ)
  (deleted_files : ℕ)
  (remaining_apps : ℕ)
  (remaining_files : ℕ)
  (h1 : initial_apps - deleted_apps = remaining_apps)
  (h2 : initial_files - deleted_files = remaining_files)
  (h3 : initial_apps = 24)
  (h4 : initial_files = 9)
  (h5 : remaining_apps = 12)
  (h6 : remaining_files = 5) :
  remaining_apps - remaining_files = 7 :=
by {
  sorry
}

end apps_more_than_files_l17_17389


namespace interior_angle_regular_octagon_l17_17795

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l17_17795


namespace time_taken_by_x_alone_l17_17490

theorem time_taken_by_x_alone 
  (W : ℝ)
  (Rx Ry Rz : ℝ)
  (h1 : Ry = W / 24)
  (h2 : Ry + Rz = W / 6)
  (h3 : Rx + Rz = W / 4) :
  (W / Rx) = 16 :=
by
  sorry

end time_taken_by_x_alone_l17_17490


namespace ratio_second_to_third_l17_17121

-- Define the three numbers A, B, C, and their conditions.
variables (A B C : ℕ)

-- Conditions derived from the problem statement.
def sum_condition : Prop := A + B + C = 98
def ratio_condition : Prop := 3 * A = 2 * B
def second_number_value : Prop := B = 30

-- The main theorem stating the problem to prove.
theorem ratio_second_to_third (h1 : sum_condition A B C) (h2 : ratio_condition A B) (h3 : second_number_value B) :
  B = 30 ∧ A = 20 ∧ C = 48 → B / C = 5 / 8 :=
by
  sorry

end ratio_second_to_third_l17_17121


namespace luke_games_l17_17736

theorem luke_games (F G : ℕ) (H1 : G = 2) (H2 : F + G - 2 = 2) : F = 2 := by
  sorry

end luke_games_l17_17736


namespace problem_solution_l17_17667

noncomputable def product(problem1: ℕ, problem2: ℕ): ℂ :=
∏ k in (finset.range problem1).image (λ (n : ℕ), e ^ (2 * π * complex.I * n / 17)),
  ∏ j in (finset.range problem2).image (λ (m : ℕ), e ^ (2 * π * complex.I * m / 13)),
    (j - k)

theorem problem_solution : product 15 12 = 13 := 
sorry

end problem_solution_l17_17667


namespace ryan_hours_english_is_6_l17_17395

def hours_chinese : Nat := 2

def hours_english (C : Nat) : Nat := C + 4

theorem ryan_hours_english_is_6 (C : Nat) (hC : C = hours_chinese) : hours_english C = 6 :=
by
  sorry

end ryan_hours_english_is_6_l17_17395


namespace length_of_platform_l17_17014

variable (Vtrain : Real := 55)
variable (str_len : Real := 360)
variable (cross_time : Real := 57.59539236861051)
variable (conversion_factor : Real := 5/18)

theorem length_of_platform :
  let Vtrain_mps := Vtrain * conversion_factor
  let distance_covered := Vtrain_mps * cross_time
  ∃ L : Real, str_len + L = distance_covered → L = 520 :=
by
  let Vtrain_mps := Vtrain * conversion_factor
  let distance_covered := Vtrain_mps * cross_time
  exists (distance_covered - str_len)
  intro h
  have h1 : distance_covered - str_len = 520 := sorry
  exact h1


end length_of_platform_l17_17014


namespace greatest_divisor_of_remainders_l17_17134

theorem greatest_divisor_of_remainders (x : ℕ) :
  (1442 % x = 12) ∧ (1816 % x = 6) ↔ x = 10 :=
by
  sorry

end greatest_divisor_of_remainders_l17_17134


namespace doughnuts_served_initially_l17_17173

def initial_doughnuts_served (staff_count : Nat) (doughnuts_per_staff : Nat) (doughnuts_left : Nat) : Nat :=
  staff_count * doughnuts_per_staff + doughnuts_left

theorem doughnuts_served_initially :
  ∀ (staff_count doughnuts_per_staff doughnuts_left : Nat), staff_count = 19 → doughnuts_per_staff = 2 → doughnuts_left = 12 →
  initial_doughnuts_served staff_count doughnuts_per_staff doughnuts_left = 50 :=
by
  intros staff_count doughnuts_per_staff doughnuts_left hstaff hdonuts hleft
  rw [hstaff, hdonuts, hleft]
  rfl

#check doughnuts_served_initially

end doughnuts_served_initially_l17_17173


namespace actual_price_of_good_l17_17028

variables (P : Real)

theorem actual_price_of_good:
  (∀ (P : ℝ), 0.5450625 * P = 6500 → P = 6500 / 0.5450625) :=
  by sorry

end actual_price_of_good_l17_17028


namespace min_positive_numbers_l17_17320

theorem min_positive_numbers (n : ℕ) (numbers : ℕ → ℤ) 
  (h_length : n = 103) 
  (h_consecutive : ∀ i : ℕ, i < n → (∃ (p1 p2 : ℕ), p1 < 5 ∧ p2 < 5 ∧ p1 ≠ p2 ∧ numbers (i + p1) > 0 ∧ numbers (i + p2) > 0)) :
  ∃ (min_positive : ℕ), min_positive = 42 :=
by
  sorry

end min_positive_numbers_l17_17320


namespace factorize_problem_1_factorize_problem_2_l17_17526

theorem factorize_problem_1 (a b : ℝ) : -3 * a ^ 3 + 12 * a ^ 2 * b - 12 * a * b ^ 2 = -3 * a * (a - 2 * b) ^ 2 := 
sorry

theorem factorize_problem_2 (m n : ℝ) : 9 * (m + n) ^ 2 - (m - n) ^ 2 = 4 * (2 * m + n) * (m + 2 * n) := 
sorry

end factorize_problem_1_factorize_problem_2_l17_17526


namespace pyramid_volume_l17_17606

noncomputable def volume_of_pyramid (a α β : ℝ) : ℝ :=
  (a^3 * Real.sin (α / 2) * Real.tan β) / 6

theorem pyramid_volume (a α β : ℝ) : (volume_of_pyramid a α β ) = (a^3 * Real.sin (α / 2) * Real.tan β) / 6 :=
by sorry

end pyramid_volume_l17_17606


namespace fair_people_ratio_l17_17462

def next_year_ratio (this_year next_year last_year : ℕ) (total : ℕ) :=
  this_year = 600 ∧
  last_year = next_year - 200 ∧
  this_year + last_year + next_year = total → 
  next_year = 2 * this_year

theorem fair_people_ratio :
  ∀ (next_year : ℕ),
  next_year_ratio 600 next_year (next_year - 200) 2800 → next_year = 2 * 600 := by
sorry

end fair_people_ratio_l17_17462


namespace lateral_surface_area_of_cone_l17_17456

-- Definitions from the conditions
def base_radius : ℝ := 6
def slant_height : ℝ := 15

-- Theorem statement to be proved
theorem lateral_surface_area_of_cone (r l : ℝ) (hr : r = base_radius) (hl : l = slant_height) : 
  (π * r * l) = 90 * π :=
by
  sorry

end lateral_surface_area_of_cone_l17_17456


namespace regular_octagon_interior_angle_l17_17842

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17842


namespace sqrt_floor_eight_l17_17911

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l17_17911


namespace mark_charged_more_hours_than_kate_l17_17751

variables (K P M : ℝ)
variables (h1 : K + P + M = 198) (h2 : P = 2 * K) (h3 : M = 3 * P)

theorem mark_charged_more_hours_than_kate : M - K = 110 :=
by
  sorry

end mark_charged_more_hours_than_kate_l17_17751


namespace proj_v_w_l17_17188

noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
  let w_dot_w := dot_product w w
  let v_dot_w := dot_product v w
  let scalar := v_dot_w / w_dot_w
  (scalar * w.1, scalar * w.2)

theorem proj_v_w :
  let v := (4, -3)
  let w := (12, 5)
  proj v w = (396 / 169, 165 / 169) :=
by
  sorry

end proj_v_w_l17_17188


namespace part1_extreme_value_at_2_part2_increasing_function_l17_17201

noncomputable def f (a x : ℝ) := a * x - a / x - 2 * Real.log x

theorem part1_extreme_value_at_2 (a : ℝ) :
  (∃ x : ℝ, x = 2 ∧ ∀ y : ℝ, f a x ≥ f a y) → a = 4 / 5 ∧ f a 1/2 = 2 * Real.log 2 - 6 / 5 := by
  sorry

theorem part2_increasing_function (a : ℝ) :
  (∀ x : ℝ, 0 < x → deriv (f a) x ≥ 0) → a ≥ 1 := by
  sorry

end part1_extreme_value_at_2_part2_increasing_function_l17_17201


namespace transformation_g_from_f_l17_17472

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (8 * x + 3 * Real.pi / 2)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem transformation_g_from_f :
  (∀ x, g x = f (x + Real.pi / 4) * 2) ∨ (∀ x, g x = f (x - Real.pi / 4) * 2) := 
by
  sorry

end transformation_g_from_f_l17_17472


namespace option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l17_17137

theorem option_A_correct (a : ℝ) : a ^ 2 * a ^ 3 = a ^ 5 := by {
  -- Here, we would provide the proof if required,
  -- but we are only stating the theorem.
  sorry
}

-- You may optionally add definitions of incorrect options for completeness.
theorem option_B_incorrect (a : ℝ) : ¬(a + 2 * a = 3 * a ^ 2) := by {
  sorry
}

theorem option_C_incorrect (a b : ℝ) : ¬((a * b) ^ 3 = a * b ^ 3) := by {
  sorry
}

theorem option_D_incorrect (a : ℝ) : ¬((-a ^ 3) ^ 2 = -a ^ 6) := by {
  sorry
}

end option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l17_17137


namespace find_date_behind_l17_17447

variables (x y : ℕ)
-- Conditions
def date_behind_C := x
def date_behind_A := x + 1
def date_behind_B := x + 13
def date_behind_P := x + 14

-- Statement to prove
theorem find_date_behind : (x + y = (x + 1) + (x + 13)) → (y = date_behind_P) :=
by
  sorry

end find_date_behind_l17_17447


namespace regular_octagon_interior_angle_measure_l17_17782

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l17_17782


namespace simplify_fraction_expression_l17_17334

theorem simplify_fraction_expression (d : ℤ) :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 :=
by
  -- skip the proof by adding sorry
  sorry

end simplify_fraction_expression_l17_17334


namespace initial_people_count_is_16_l17_17036

-- Define the conditions
def initial_people (x : ℕ) : Prop :=
  let people_came_in := 5 in
  let people_left := 2 in
  let final_people := 19 in
  x + people_came_in - people_left = final_people

-- Define the theorem
theorem initial_people_count_is_16 (x : ℕ) (h : initial_people x) : x = 16 :=
by
  sorry

end initial_people_count_is_16_l17_17036


namespace kimberly_gumballs_last_days_l17_17727

theorem kimberly_gumballs_last_days :
  (let earrings_day1 := 3 in
   let earrings_day2 := 2 * earrings_day1 in
   let earrings_day3 := earrings_day2 - 1 in
   let total_earrings := earrings_day1 + earrings_day2 + earrings_day3 in
   let total_gumballs := 9 * total_earrings in
   let days_last := total_gumballs / 3 in
   days_last = 42) :=
by {
  let earrings_day1 := 3,
  let earrings_day2 := 2 * earrings_day1,
  let earrings_day3 := earrings_day2 - 1,
  let total_earrings := earrings_day1 + earrings_day2 + earrings_day3,
  let total_gumballs := 9 * total_earrings,
  let days_last := total_gumballs / 3,
  exact sorry
}

end kimberly_gumballs_last_days_l17_17727


namespace James_baked_muffins_l17_17876

theorem James_baked_muffins (arthur_muffins : Nat) (multiplier : Nat) (james_muffins : Nat) : 
  arthur_muffins = 115 → 
  multiplier = 12 → 
  james_muffins = arthur_muffins * multiplier → 
  james_muffins = 1380 :=
by
  intros haf ham hmul
  rw [haf, ham] at hmul
  simp at hmul
  exact hmul

end James_baked_muffins_l17_17876


namespace compute_modulo_l17_17381

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l17_17381


namespace hair_cut_length_l17_17343

-- Definitions corresponding to the conditions in the problem
def initial_length : ℕ := 18
def current_length : ℕ := 9

-- Statement to prove
theorem hair_cut_length : initial_length - current_length = 9 :=
by
  sorry

end hair_cut_length_l17_17343


namespace cube_painted_faces_l17_17151

noncomputable def painted_faces_count (side_length painted_cubes_edge middle_cubes_edge : ℕ) : ℕ :=
  let total_corners := 8
  let total_edges := 12
  total_corners + total_edges * middle_cubes_edge

theorem cube_painted_faces :
  ∀ side_length : ℕ, side_length = 4 →
  ∀ painted_cubes_edge middle_cubes_edge total_cubes : ℕ,
  total_cubes = side_length * side_length * side_length →
  painted_cubes_edge = 3 →
  middle_cubes_edge = 2 →
  painted_faces_count side_length painted_cubes_edge middle_cubes_edge = 32 := sorry

end cube_painted_faces_l17_17151


namespace Sn_eq_S9_l17_17554

-- Definition of the arithmetic sequence sum formula.
def Sn (n a1 d : ℕ) : ℕ := (n * a1) + (n * (n - 1) / 2 * d)

theorem Sn_eq_S9 (a1 d : ℕ) (h1 : Sn 3 a1 d = 9) (h2 : Sn 6 a1 d = 36) : Sn 9 a1 d = 81 := by
  sorry

end Sn_eq_S9_l17_17554


namespace inequality_holds_for_all_real_numbers_l17_17248

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l17_17248


namespace fractions_order_l17_17332

theorem fractions_order :
  let frac1 := (21 : ℚ) / (17 : ℚ)
  let frac2 := (23 : ℚ) / (19 : ℚ)
  let frac3 := (25 : ℚ) / (21 : ℚ)
  frac3 < frac2 ∧ frac2 < frac1 :=
by sorry

end fractions_order_l17_17332


namespace objective_function_range_l17_17199

theorem objective_function_range:
  (∃ x y : ℝ, x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) ∧
  (∀ x y : ℝ, (x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) →
  (3*x + y ≥ (19:ℝ) / 9 ∧ 3*x + y ≤ 6)) :=
sorry

-- We have defined the conditions, the objective function, and the assertion in Lean 4.

end objective_function_range_l17_17199


namespace possible_distances_between_andrey_and_gleb_l17_17314

theorem possible_distances_between_andrey_and_gleb (A B V G : Point) 
  (d_AB : ℝ) (d_VG : ℝ) (d_BV : ℝ) (d_AG : ℝ)
  (h1 : d_AB = 600) 
  (h2 : d_VG = 600) 
  (h3 : d_AG = 3 * d_BV) : 
  d_AG = 900 ∨ d_AG = 1800 :=
by {
  sorry
}

end possible_distances_between_andrey_and_gleb_l17_17314


namespace determine_m_l17_17682

variable (A B : Set ℝ)
variable (m : ℝ)

theorem determine_m (hA : A = {-1, 3, m}) (hB : B = {3, 4}) (h_inter : B ∩ A = B) : m = 4 :=
sorry

end determine_m_l17_17682


namespace Amy_work_hours_l17_17510

theorem Amy_work_hours (summer_weeks: ℕ) (summer_hours_per_week: ℕ) (summer_total_earnings: ℕ)
                       (school_weeks: ℕ) (school_total_earnings: ℕ) (hourly_wage: ℕ) 
                       (school_hours_per_week: ℕ):
    summer_weeks = 8 →
    summer_hours_per_week = 40 →
    summer_total_earnings = 3200 →
    school_weeks = 32 →
    school_total_earnings = 4800 →
    hourly_wage = summer_total_earnings / (summer_weeks * summer_hours_per_week) →
    school_hours_per_week = school_total_earnings / (hourly_wage * school_weeks) →
    school_hours_per_week = 15 :=
by
  intros
  sorry

end Amy_work_hours_l17_17510


namespace sqrt_sum_inequality_l17_17106

-- Define variables a and b as positive real numbers
variable {a b : ℝ}

-- State the theorem to be proved
theorem sqrt_sum_inequality (ha : 0 < a) (hb : 0 < b) : 
  (a.sqrt + b.sqrt)^8 ≥ 64 * a * b * (a + b)^2 :=
sorry

end sqrt_sum_inequality_l17_17106


namespace find_n_l17_17874

theorem find_n (x y m n : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) 
  (h1 : 100 * y + x = (x + y) * m) (h2 : 100 * x + y = (x + y) * n) : n = 101 - m :=
by
  sorry

end find_n_l17_17874


namespace count_empty_intersection_image_l17_17547

open Finset

-- Define the set A
def A : Finset ℕ := {1, 2, 3}

-- Define the function type from A to A
def FuncType (A : Type) := A → A

theorem count_empty_intersection_image :
  ∃ (n : ℕ), n = 42 ∧ ∀ (f g : FuncType ℕ), f '' (A : Set ℕ) ∩ g '' (A : Set ℕ) = ∅ :=
  sorry

end count_empty_intersection_image_l17_17547


namespace biggest_number_in_ratio_l17_17535

theorem biggest_number_in_ratio (A B C D : ℕ) (h1 : 2 * D = 5 * A) (h2 : 3 * D = 5 * B) (h3 : 4 * D = 5 * C) (h_sum : A + B + C + D = 1344) : D = 480 := 
by
  sorry

end biggest_number_in_ratio_l17_17535


namespace greatest_possible_sum_l17_17503

noncomputable def eight_products_sum_max : ℕ :=
  let a := 3
  let b := 4
  let c := 5
  let d := 8
  let e := 6
  let f := 7
  7 * (c + d) * (e + f)

theorem greatest_possible_sum (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) :
  eight_products_sum_max = 1183 :=
by
  sorry

end greatest_possible_sum_l17_17503


namespace clea_ride_time_l17_17880

noncomputable def walk_down_stopped (x y : ℝ) : Prop := 90 * x = y
noncomputable def walk_down_moving (x y k : ℝ) : Prop := 30 * (x + k) = y
noncomputable def ride_time (y k t : ℝ) : Prop := t = y / k

theorem clea_ride_time (x y k t : ℝ) (h1 : walk_down_stopped x y) (h2 : walk_down_moving x y k) :
  ride_time y k t → t = 45 :=
sorry

end clea_ride_time_l17_17880


namespace systematic_sampling_interval_l17_17125

-- Definitions based on conditions
def population_size : ℕ := 1000
def sample_size : ℕ := 40

-- Theorem statement 
theorem systematic_sampling_interval :
  population_size / sample_size = 25 :=
by
  sorry

end systematic_sampling_interval_l17_17125


namespace inequality_proof_l17_17282

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l17_17282


namespace interior_angle_regular_octagon_l17_17797

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l17_17797


namespace max_non_overlapping_areas_l17_17861

theorem max_non_overlapping_areas (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, k = 4 * n + 4 := 
sorry

end max_non_overlapping_areas_l17_17861


namespace regular_octagon_angle_measure_l17_17833

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l17_17833


namespace inequality_holds_for_all_real_numbers_l17_17244

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l17_17244


namespace polynomial_factorization_l17_17705

variable (x y : ℝ)

theorem polynomial_factorization (m : ℝ) :
  (∃ (a b : ℝ), 6 * x^2 - 5 * x * y - 4 * y^2 - 11 * x + 22 * y + m = (3 * x - 4 * y + a) * (2 * x + y + b)) →
  m = -10 :=
sorry

end polynomial_factorization_l17_17705


namespace equilibrium_price_without_subsidy_increase_in_quantity_due_to_subsidy_l17_17619

-- Definitions for supply and demand functions
def Qs (p : ℝ) : ℝ := 2 + 8 * p
def Qd (p : ℝ) : ℝ := -2 * p + 12

-- Equilibrium without subsidy
theorem equilibrium_price_without_subsidy : (∃ p q, Qs p = q ∧ Qd p = q ∧ p = 1 ∧ q = 10) :=
sorry

-- New supply function with subsidy
def Qs_with_subsidy (p : ℝ) : ℝ := 10 + 8 * p

-- Increase in quantity sold due to subsidy
theorem increase_in_quantity_due_to_subsidy : 
  (∃ Δq, Δq = Qd 0.2 - Qd 1 ∧ Δq = 1.6) :=
sorry

end equilibrium_price_without_subsidy_increase_in_quantity_due_to_subsidy_l17_17619


namespace number_of_cars_l17_17725

theorem number_of_cars (total_wheels cars_bikes trash_can tricycle roller_skates : ℕ) 
  (h1 : cars_bikes = 2) 
  (h2 : trash_can = 2) 
  (h3 : tricycle = 3) 
  (h4 : roller_skates = 4) 
  (h5 : total_wheels = 25) 
  : (total_wheels - (cars_bikes * 2 + trash_can * 2 + tricycle * 3 + roller_skates * 4)) / 4 = 3 :=
by
  sorry

end number_of_cars_l17_17725


namespace initial_average_age_l17_17766

theorem initial_average_age (A : ℕ) (h1 : ∀ x : ℕ, 10 * A = 10 * A)
  (h2 : 5 * 17 + 10 * A = 15 * (A + 1)) : A = 14 :=
by 
  sorry

end initial_average_age_l17_17766


namespace two_digit_perfect_squares_divisible_by_3_l17_17701

theorem two_digit_perfect_squares_divisible_by_3 :
  ∃! n1 n2 : ℕ, (10 ≤ n1^2 ∧ n1^2 < 100 ∧ n1^2 % 3 = 0) ∧
               (10 ≤ n2^2 ∧ n2^2 < 100 ∧ n2^2 % 3 = 0) ∧
                (n1 ≠ n2) :=
by sorry

end two_digit_perfect_squares_divisible_by_3_l17_17701


namespace area_of_circle_l17_17131

theorem area_of_circle :
  (∃ (x y : ℝ), x^2 + y^2 - 6 * x + 8 * y = -9) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  -- We need to prove the area is 16π
  sorry

end area_of_circle_l17_17131


namespace white_roses_needed_l17_17101

theorem white_roses_needed (bouquets table_decorations white_roses_per_table_decoration white_roses_per_bouquet : ℕ)
  (h_bouquets : bouquets = 5)
  (h_table_decorations : table_decorations = 7)
  (h_white_roses_per_table_decoration : white_roses_per_table_decoration = 12)
  (h_white_roses_per_bouquet : white_roses_per_bouquet = 5):
  bouquets * white_roses_per_bouquet + table_decorations * white_roses_per_table_decoration = 109 := by
  sorry

end white_roses_needed_l17_17101


namespace sample_size_l17_17500

theorem sample_size (n : ℕ) (h1 : n ∣ 36) (h2 : 36 / n ∣ 6) (h3 : (n + 1) ∣ 35) : n = 6 := 
sorry

end sample_size_l17_17500


namespace upper_side_length_trapezoid_l17_17330

theorem upper_side_length_trapezoid
  (L U : ℝ) 
  (h : ℝ := 8) 
  (A : ℝ := 72) 
  (cond1 : U = L - 6)
  (cond2 : 1/2 * (L + U) * h = A) :
  U = 6 := 
by 
  sorry

end upper_side_length_trapezoid_l17_17330


namespace tiles_in_each_row_l17_17306

theorem tiles_in_each_row (area : ℝ) (tile_side : ℝ) (feet_to_inches : ℝ) (number_of_tiles : ℕ) :
  area = 400 ∧ tile_side = 8 ∧ feet_to_inches = 12 → number_of_tiles = 30 :=
by
  intros h
  cases h with h_area h_rest
  cases h_rest with h_tile_side h_feet_to_inches
  sorry

end tiles_in_each_row_l17_17306


namespace pencils_added_l17_17773

theorem pencils_added (initial_pencils total_pencils Mike_pencils : ℕ) 
    (h1 : initial_pencils = 41) 
    (h2 : total_pencils = 71) 
    (h3 : total_pencils = initial_pencils + Mike_pencils) :
    Mike_pencils = 30 := by
  sorry

end pencils_added_l17_17773


namespace area_of_region_l17_17133

noncomputable def area_of_enclosed_region : Real :=
  -- equation of the circle after completing square
  let circle_eqn := fun (x y : Real) => ((x - 3)^2 + (y + 4)^2 = 16)
  if circle_eqn then 
    Real.pi * 4^2
  else
    0

theorem area_of_region (h : ∀ x y, x^2 + y^2 - 6x + 8y = -9 → ((x-3)^2 + (y+4)^2 = 16)) :
  area_of_enclosed_region = 16 * Real.pi :=
by
  -- This is a statement, so just include a sorry to skip the proof.
  sorry

end area_of_region_l17_17133


namespace odd_function_value_l17_17962

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 + x else -(x^2 + x)

theorem odd_function_value : f (-3) = -12 :=
by
  -- proof goes here
  sorry

end odd_function_value_l17_17962


namespace interest_rate_part1_l17_17449

-- Definitions according to the problem statement
def total_amount : ℝ := 4000
def P1 : ℝ := 2799.9999999999995
def P2 := total_amount - P1
def annual_interest : ℝ := 144
def P2_interest_rate : ℝ := 0.05

-- Formal statement of the problem
theorem interest_rate_part1 :
  (P2 * P2_interest_rate) + (P1 * (3 / 100)) = annual_interest :=
by
  sorry

end interest_rate_part1_l17_17449


namespace integer_roots_of_quadratic_eq_l17_17180

theorem integer_roots_of_quadratic_eq (a : ℤ) :
  (∃ x1 x2 : ℤ, x1 + x2 = a ∧ x1 * x2 = 9 * a) ↔
  a = 100 ∨ a = -64 ∨ a = 48 ∨ a = -12 ∨ a = 36 ∨ a = 0 :=
by sorry

end integer_roots_of_quadratic_eq_l17_17180


namespace bridge_weight_excess_l17_17158

theorem bridge_weight_excess :
  ∀ (Kelly_weight Megan_weight Mike_weight : ℕ),
  Kelly_weight = 34 →
  Kelly_weight = 85 * Megan_weight / 100 →
  Mike_weight = Megan_weight + 5 →
  (Kelly_weight + Megan_weight + Mike_weight) - 100 = 19 :=
by
  intros Kelly_weight Megan_weight Mike_weight
  intros h1 h2 h3
  sorry

end bridge_weight_excess_l17_17158


namespace flower_pot_arrangements_l17_17402

/-- From 7 different pots of flowers, 5 are to be selected and placed 
in front of the podium such that two specific pots are not allowed to be placed 
in the very center. Prove that the number of different arrangements is 1800. -/
theorem flower_pot_arrangements : 
  let pots := {1, 2, 3, 4, 5, 6, 7} in
  let center_not_allowed := {1, 2} in
  finset.card {arr : finset (finset ℕ) // 
    arr ∈ (finset.powerset pots) ∧ 
    finset.card arr = 5 ∧ 
    ∃ center, center ∉ center_not_allowed} = 1800 :=
sorry

end flower_pot_arrangements_l17_17402


namespace inequality_proof_l17_17283

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l17_17283


namespace largest_B_at_45_l17_17367

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def B (k : ℕ) : ℝ :=
  if k ≤ 500 then (binomial_coeff 500 k) * (0.1)^k else 0

theorem largest_B_at_45 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ 500 → B k ≤ B 45 :=
by
  intros k hk
  sorry

end largest_B_at_45_l17_17367


namespace magician_cannot_determine_polynomial_l17_17646

theorem magician_cannot_determine_polynomial (n : ℕ) (hn : 0 < n) 
  (x : Fin 2n → ℝ) (hx : ∀ i j, i < j → x i < x j)
  (P : ℝ[X]) (hdeg : P.degree ≤ n) : 
  ∃ Q : ℝ[X], Q.degree ≤ n ∧
    (∀ i : Fin 2n, eval (x i) P = eval (x i) Q) ∧ P ≠ Q := 
begin
  -- Placeholder for the actual proof
  sorry
end

end magician_cannot_determine_polynomial_l17_17646


namespace pelican_fish_count_l17_17648

theorem pelican_fish_count 
(P K F : ℕ) 
(h1: K = P + 7) 
(h2: F = 3 * (P + K)) 
(h3: F = P + 86) : P = 13 := 
by 
  sorry

end pelican_fish_count_l17_17648


namespace emily_selects_green_apples_l17_17885

theorem emily_selects_green_apples :
  let total_apples := 10
  let red_apples := 6
  let green_apples := 4
  let selected_apples := 3
  let total_combinations := Nat.choose total_apples selected_apples
  let green_combinations := Nat.choose green_apples selected_apples
  (green_combinations / total_combinations : ℚ) = 1 / 30 :=
by
  sorry

end emily_selects_green_apples_l17_17885


namespace regular_octagon_interior_angle_l17_17819

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l17_17819


namespace monthly_earnings_l17_17603

-- Defining the initial conditions and known information
def current_worth : ℝ := 90
def months : ℕ := 5

-- Let I be the initial investment, and E be the earnings per month.

noncomputable def initial_investment (I : ℝ) := I * 3 = current_worth
noncomputable def earned_twice_initial (E : ℝ) (I : ℝ) := E * months = 2 * I

-- Proving the monthly earnings
theorem monthly_earnings (I E : ℝ) (h1 : initial_investment I) (h2 : earned_twice_initial E I) : E = 12 :=
sorry

end monthly_earnings_l17_17603


namespace find_ab_l17_17691

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
sorry

end find_ab_l17_17691


namespace find_original_acid_amount_l17_17017

noncomputable def original_amount_of_acid (a w : ℝ) : Prop :=
  3 * a = w + 2 ∧ 5 * a = 3 * w - 10

theorem find_original_acid_amount (a w : ℝ) (h : original_amount_of_acid a w) : a = 4 :=
by
  sorry

end find_original_acid_amount_l17_17017


namespace coordinates_of_point_on_x_axis_l17_17940

theorem coordinates_of_point_on_x_axis (m : ℤ) 
  (h : 2 * m + 8 = 0) : (m + 5, 2 * m + 8) = (1, 0) :=
sorry

end coordinates_of_point_on_x_axis_l17_17940


namespace find_exponent_l17_17062

theorem find_exponent (n : ℝ) (hn: (3:ℝ)^n = Real.sqrt 3) : n = 1 / 2 :=
by sorry

end find_exponent_l17_17062


namespace quartic_polynomial_root_l17_17178

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 6*x^2 - 4*x - 2

theorem quartic_polynomial_root :
  Q (Real.sqrt (Real.sqrt 3) + 1) = 0 :=
by
  sorry

end quartic_polynomial_root_l17_17178


namespace injective_function_equality_l17_17494

def injective (f : ℕ → ℕ) : Prop :=
  ∀ ⦃a b : ℕ⦄, f a = f b → a = b

theorem injective_function_equality
  {f : ℕ → ℕ}
  (h_injective : injective f)
  (h_eq : ∀ n m : ℕ, (1 / f n) + (1 / f m) = 4 / (f n + f m)) :
  ∀ n m : ℕ, m = n :=
by
  sorry

end injective_function_equality_l17_17494


namespace solve_quadratic_eq_l17_17604

theorem solve_quadratic_eq (x y : ℝ) :
  (x = 3 ∧ y = 1) ∨ (x = -1 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) ∨ (x = -1 ∧ y = -5) ↔
  x ^ 2 - x * y + y ^ 2 - x + 3 * y - 7 = 0 := sorry

end solve_quadratic_eq_l17_17604


namespace tan_alpha_eq_neg_five_twelfths_l17_17684

-- Define the angle α and the given conditions
variables (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π)

-- The goal is to prove that tan α = -5 / 12
theorem tan_alpha_eq_neg_five_twelfths (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π) :
  Real.tan α = -5 / 12 :=
sorry

end tan_alpha_eq_neg_five_twelfths_l17_17684


namespace speed_in_still_water_l17_17852

theorem speed_in_still_water (u d s : ℝ) (hu : u = 20) (hd : d = 60) (hs : s = (u + d) / 2) : s = 40 := 
by 
  sorry

end speed_in_still_water_l17_17852


namespace new_volume_l17_17356

variable (l w h : ℝ)

-- Given conditions
def volume := l * w * h = 5000
def surface_area := l * w + l * h + w * h = 975
def sum_of_edges := l + w + h = 60

-- Statement to prove
theorem new_volume (h1 : volume l w h) (h2 : surface_area l w h) (h3 : sum_of_edges l w h) :
  (l + 2) * (w + 2) * (h + 2) = 7198 :=
by
  sorry

end new_volume_l17_17356


namespace smallest_even_integer_l17_17001

theorem smallest_even_integer :
  ∃ (x : ℤ), |3 * x - 4| ≤ 20 ∧ (∀ (y : ℤ), |3 * y - 4| ≤ 20 → (2 ∣ y) → x ≤ y) ∧ (2 ∣ x) :=
by
  use -4
  sorry

end smallest_even_integer_l17_17001


namespace fishing_problem_l17_17029

theorem fishing_problem
  (P : ℕ) -- weight of the fish Peter caught
  (H1 : Ali_weight = 2 * P) -- Ali caught twice as much as Peter
  (H2 : Joey_weight = P + 1) -- Joey caught 1 kg more than Peter
  (H3 : P + 2 * P + (P + 1) = 25) -- Together they caught 25 kg
  : Ali_weight = 12 :=
by
  sorry

end fishing_problem_l17_17029


namespace shaded_region_area_l17_17614

-- Given conditions
def diagonal_PQ : ℝ := 10
def number_of_squares : ℕ := 20

-- Definition of the side length of the squares
noncomputable def side_length := diagonal_PQ / (4 * Real.sqrt 2)

-- Area of one smaller square
noncomputable def one_square_area := side_length * side_length

-- Total area of the shaded region
noncomputable def total_area_of_shaded_region := number_of_squares * one_square_area

-- The theorem to be proven
theorem shaded_region_area : total_area_of_shaded_region = 62.5 := by
  sorry

end shaded_region_area_l17_17614


namespace biggest_number_in_ratio_l17_17536

theorem biggest_number_in_ratio (A B C D : ℕ) (h1 : 2 * D = 5 * A) (h2 : 3 * D = 5 * B) (h3 : 4 * D = 5 * C) (h_sum : A + B + C + D = 1344) : D = 480 := 
by
  sorry

end biggest_number_in_ratio_l17_17536


namespace pentagon_coloring_valid_l17_17226

-- Define the colors
inductive Color
| Red
| Blue

-- Define the vertices as a type
inductive Vertex
| A | B | C | D | E

open Vertex Color

-- Define an edge as a pair of vertices
def Edge := Vertex × Vertex

-- Define the coloring function
def color : Edge → Color := sorry

-- Define the pentagon
def pentagon_edges : List Edge :=
  [(A, B), (B, C), (C, D), (D, E), (E, A), (A, C), (A, D), (A, E), (B, D), (B, E), (C, E)]

-- Define the condition for a valid triangle coloring
def valid_triangle_coloring (e1 e2 e3 : Edge) : Prop :=
  (color e1 = Red ∧ (color e2 = Blue ∨ color e3 = Blue)) ∨
  (color e2 = Red ∧ (color e1 = Blue ∨ color e3 = Blue)) ∨
  (color e3 = Red ∧ (color e1 = Blue ∨ color e2 = Blue))

-- Define the condition for all triangles formed by the vertices of the pentagon
def all_triangles_valid : Prop :=
  ∀ v1 v2 v3 : Vertex,
    v1 ≠ v2 → v2 ≠ v3 → v1 ≠ v3 →
    valid_triangle_coloring (v1, v2) (v2, v3) (v1, v3)

-- Statement: Prove that there are 12 valid ways to color the pentagon
theorem pentagon_coloring_valid : (∃ (coloring : Edge → Color), all_triangles_valid) :=
  sorry

end pentagon_coloring_valid_l17_17226


namespace inequality_proof_l17_17249

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l17_17249


namespace inequality_inequality_l17_17258

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l17_17258


namespace steel_scrap_problem_l17_17626

theorem steel_scrap_problem 
  (x y : ℝ)
  (h1 : x + y = 140)
  (h2 : 0.05 * x + 0.40 * y = 42) :
  x = 40 ∧ y = 100 :=
by
  -- Solution steps are not required here
  sorry

end steel_scrap_problem_l17_17626


namespace wall_cost_equal_l17_17152

theorem wall_cost_equal (A B C : ℝ) (d_1 d_2 : ℝ) (h1 : A = B) (h2 : B = C) : d_1 = d_2 :=
by
  -- sorry is used to skip the proof
  sorry

end wall_cost_equal_l17_17152


namespace tiles_per_row_proof_l17_17308

def area_square_room : ℝ := 400 -- in square feet
def tile_size : ℝ := 8 / 12 -- each tile is 8 inches, converted to feet (8/12 feet)

noncomputable def number_of_tiles_per_row : ℕ :=
  let side_length := Math.sqrt area_square_room in
  let side_length_in_inch := side_length * 12 in
  Nat.floor (side_length_in_inch / tile_size)

theorem tiles_per_row_proof :
  number_of_tiles_per_row = 30 := by
  sorry

end tiles_per_row_proof_l17_17308


namespace perfect_square_trinomial_implies_possible_m_values_l17_17419

theorem perfect_square_trinomial_implies_possible_m_values (m : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, (x - a)^2 = x^2 - 2*m*x + 16) → (m = 4 ∨ m = -4) :=
by
  sorry

end perfect_square_trinomial_implies_possible_m_values_l17_17419


namespace tank_empties_in_4320_minutes_l17_17342

-- Define the initial conditions
def tankVolumeCubicFeet: ℝ := 30
def inletPipeRateCubicInchesPerMin: ℝ := 5
def outletPipe1RateCubicInchesPerMin: ℝ := 9
def outletPipe2RateCubicInchesPerMin: ℝ := 8
def feetToInches: ℝ := 12

-- Conversion from cubic feet to cubic inches
def tankVolumeCubicInches: ℝ := tankVolumeCubicFeet * feetToInches^3

-- Net rate of emptying in cubic inches per minute
def netRateOfEmptying: ℝ := (outletPipe1RateCubicInchesPerMin + outletPipe2RateCubicInchesPerMin) - inletPipeRateCubicInchesPerMin

-- Time to empty the tank
noncomputable def timeToEmptyTank: ℝ := tankVolumeCubicInches / netRateOfEmptying

-- The theorem to prove
theorem tank_empties_in_4320_minutes :
  timeToEmptyTank = 4320 := by
  sorry

end tank_empties_in_4320_minutes_l17_17342


namespace max_value_of_f_l17_17228

variable (n : ℕ)

-- Define the quadratic function with coefficients a, b, and c.
noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
axiom f_n : ∃ a b c, f n a b c = 6
axiom f_n1 : ∃ a b c, f (n + 1) a b c = 14
axiom f_n2 : ∃ a b c, f (n + 2) a b c = 14

-- The main goal is to prove the maximum value of f(x) is 15.
theorem max_value_of_f : ∃ a b c, (∀ x : ℝ, f x a b c ≤ 15) :=
by
  sorry

end max_value_of_f_l17_17228


namespace inequality_holds_for_all_reals_l17_17295

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l17_17295


namespace initial_customers_l17_17653

theorem initial_customers (x : ℝ) : (x - 8 + 4 = 9) → x = 13 :=
by
  sorry

end initial_customers_l17_17653


namespace number_of_friends_l17_17141

theorem number_of_friends (n : ℕ) (h1 : 100 % n = 0) (h2 : 100 % (n + 5) = 0) (h3 : 100 / n - 1 = 100 / (n + 5)) : n = 20 :=
by
  sorry

end number_of_friends_l17_17141


namespace annual_interest_rate_l17_17656

theorem annual_interest_rate (r : ℝ) :
  (6000 * r + 4000 * 0.09 = 840) → r = 0.08 :=
by sorry

end annual_interest_rate_l17_17656


namespace inequality_inequality_l17_17261

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l17_17261


namespace birdseed_needed_weekly_birdseed_needed_l17_17753

def parakeet_daily_consumption := 2
def parrot_daily_consumption := 14
def finch_daily_consumption := parakeet_daily_consumption / 2
def num_parakeets := 3
def num_parrots := 2
def num_finches := 4
def days_in_week := 7

theorem birdseed_needed :
  num_parakeets * parakeet_daily_consumption +
  num_parrots * parrot_daily_consumption +
  num_finches * finch_daily_consumption = 38 :=
by
  sorry

theorem weekly_birdseed_needed :
  38 * days_in_week = 266 :=
by
  sorry

end birdseed_needed_weekly_birdseed_needed_l17_17753


namespace find_boys_and_girls_l17_17968

noncomputable def number_of_boys_and_girls (a b c d : Nat) : (Nat × Nat) := sorry

theorem find_boys_and_girls : 
  ∃ m d : Nat,
  (∀ (a b c : Nat), 
    ((a = 15 ∨ b = 18 ∨ c = 13) ∧ 
    (a.mod 4 = 3 ∨ b.mod 4 = 2 ∨ c.mod 4 = 1)) 
    → number_of_boys_and_girls a b c d = (16, 14)) :=
sorry

end find_boys_and_girls_l17_17968


namespace min_dist_l17_17718

open Real

theorem min_dist (a b : ℝ) :
  let A := (0, -1)
  let B := (1, 3)
  let C := (2, 6)
  let D := (0, b)
  let E := (1, a + b)
  let F := (2, 2 * a + b)
  let AD_sq := (b + 1) ^ 2
  let BE_sq := (a + b - 3) ^ 2
  let CF_sq := (2 * a + b - 6) ^ 2
  AD_sq + BE_sq + CF_sq = (b + 1) ^ 2 + (a + b - 3) ^ 2 + (2 * a + b - 6) ^ 2 → 
  a = 7 / 2 ∧ b = -5 / 6 :=
sorry

end min_dist_l17_17718


namespace initial_money_correct_l17_17474

def initial_money (total: ℕ) (allowance: ℕ): ℕ :=
  total - allowance

theorem initial_money_correct: initial_money 18 8 = 10 :=
  by sorry

end initial_money_correct_l17_17474


namespace sum_of_non_common_roots_zero_l17_17695

theorem sum_of_non_common_roots_zero (m α β γ : ℝ) 
  (h1 : α + β = -(m + 1))
  (h2 : α * β = -3)
  (h3 : α + γ = 4)
  (h4 : α * γ = -m)
  (h_common : α^2 + (m + 1)*α - 3 = 0)
  (h_common2 : α^2 - 4*α - m = 0)
  : β + γ = 0 := sorry

end sum_of_non_common_roots_zero_l17_17695


namespace no_positive_integer_solutions_l17_17559

def f (x : ℕ) : ℕ := x*x + x

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), a > 0 → b > 0 → 4 * (f a) ≠ (f b) :=
by
  intro a b a_pos b_pos
  sorry

end no_positive_integer_solutions_l17_17559


namespace first_player_wins_l17_17622

theorem first_player_wins : 
  ∃ (strategy: ℕ → bool), -- strategy is a function that decides which pile to take from
  ∀ (turn: ℕ), -- turn is the number of turns taken
  let remaining_nuts := 10 - turn in
  if remaining_nuts = 3 then -- when 3 nuts are left
    ∀ (pile1 pile2 pile3 : ℕ), pile1 + pile2 + pile3 = 3 -> -- three piles sum up to 3
    ¬ (pile1 = 1 ∧ pile2 = 1 ∧ pile3 = 1) -- these should not be in three separate piles 
  else
    (turn % 2 = 0 → strategy turn = true) ∧ -- first player's turns follow the strategy
    (turn % 2 = 1 → strategy turn = false)  -- second player's turns follow the strategy
:= sorry

end first_player_wins_l17_17622


namespace mean_score_l17_17930

theorem mean_score (μ σ : ℝ)
  (h1 : 86 = μ - 7 * σ)
  (h2 : 90 = μ + 3 * σ) : μ = 88.8 := by
  -- Proof steps are not included as per requirements.
  sorry

end mean_score_l17_17930


namespace daniel_earnings_l17_17882

theorem daniel_earnings :
  let monday_fabric := 20
  let monday_yarn := 15
  let tuesday_fabric := 2 * monday_fabric
  let tuesday_yarn := monday_yarn + 10
  let wednesday_fabric := (1 / 4) * tuesday_fabric
  let wednesday_yarn := (1 / 2) * tuesday_yarn
  let total_fabric := monday_fabric + tuesday_fabric + wednesday_fabric
  let total_yarn := monday_yarn + tuesday_yarn + wednesday_yarn
  let fabric_cost := 2
  let yarn_cost := 3
  let fabric_earnings_before_discount := total_fabric * fabric_cost
  let yarn_earnings_before_discount := total_yarn * yarn_cost
  let fabric_discount := if total_fabric > 30 then 0.10 * fabric_earnings_before_discount else 0
  let yarn_discount := if total_yarn > 20 then 0.05 * yarn_earnings_before_discount else 0
  let fabric_earnings_after_discount := fabric_earnings_before_discount - fabric_discount
  let yarn_earnings_after_discount := yarn_earnings_before_discount - yarn_discount
  let total_earnings := fabric_earnings_after_discount + yarn_earnings_after_discount
  total_earnings = 275.625 := by
  {
    sorry
  }

end daniel_earnings_l17_17882


namespace power_sum_l17_17659

theorem power_sum : 1 ^ 2009 + (-1) ^ 2009 = 0 := 
by 
  sorry

end power_sum_l17_17659


namespace geometric_progression_solution_l17_17172

theorem geometric_progression_solution (x : ℝ) :
  (2 * x + 10) ^ 2 = x * (5 * x + 10) → x = 15 + 5 * Real.sqrt 5 :=
by
  intro h
  sorry

end geometric_progression_solution_l17_17172


namespace units_digit_n_squared_plus_two_pow_n_l17_17233

theorem units_digit_n_squared_plus_two_pow_n
  (n : ℕ)
  (h : n = 2018^2 + 2^2018) : 
  (n^2 + 2^n) % 10 = 5 := by
  sorry

end units_digit_n_squared_plus_two_pow_n_l17_17233


namespace postage_problem_l17_17929

noncomputable def sum_all_positive_integers (n1 n2 : ℕ) : ℕ :=
  n1 + n2

theorem postage_problem : sum_all_positive_integers 21 22 = 43 :=
by
  have h1 : ∀ x y z : ℕ, 7 * x + 21 * y + 23 * z ≠ 120 := sorry
  have h2 : ∀ x y z : ℕ, 7 * x + 22 * y + 24 * z ≠ 120 := sorry
  exact rfl

end postage_problem_l17_17929


namespace number_of_tiles_per_row_l17_17305

-- Define the conditions 
def area_floor_sq_ft : ℕ := 400
def tile_side_inch : ℕ := 8
def feet_to_inch (f : ℕ) := 12 * f

-- Define the theorem using the conditions
theorem number_of_tiles_per_row (h : Nat.sqrt area_floor_sq_ft = 20) :
  (feet_to_inch 20) / tile_side_inch = 30 :=
by
  sorry

end number_of_tiles_per_row_l17_17305


namespace inequality_holds_for_all_reals_l17_17293

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l17_17293


namespace reliefSuppliesCalculation_l17_17126

noncomputable def totalReliefSupplies : ℝ := 644

theorem reliefSuppliesCalculation
    (A_capacity : ℝ)
    (B_capacity : ℝ)
    (A_capacity_per_day : A_capacity = 64.4)
    (capacity_ratio : A_capacity = 1.75 * B_capacity)
    (additional_transport : ∃ t : ℝ, A_capacity * t - B_capacity * t = 138 ∧ A_capacity * t = 322) :
  totalReliefSupplies = 644 := by
  sorry

end reliefSuppliesCalculation_l17_17126


namespace value_of_abc_l17_17219

noncomputable def f (x a b c : ℝ) := |(1 - x^2) * (x^2 + a * x + b)| - c

theorem value_of_abc :
  (∀ x : ℝ, f (x + 4) 8 15 9 = f (-x) 8 15 9) ∧
  (∃ x : ℝ, f x 8 15 9 = 0) ∧
  (∃ x : ℝ, f (-(x-4)) 8 15 9 = 0) ∧
  (∀ c : ℝ, c ≠ 0) →
  8 + 15 + 9 = 32 :=
by sorry

end value_of_abc_l17_17219


namespace mod_2_pow_1000_by_13_l17_17136

theorem mod_2_pow_1000_by_13 :
  (2 ^ 1000) % 13 = 3 := by
  sorry

end mod_2_pow_1000_by_13_l17_17136


namespace liza_final_balance_l17_17594

def initial_balance : ℕ := 800
def rent : ℕ := 450
def paycheck : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

theorem liza_final_balance :
  initial_balance - rent + paycheck - (electricity_bill + internet_bill) - phone_bill = 1563 := by
  sorry

end liza_final_balance_l17_17594


namespace simplify_cbrt_8000_eq_21_l17_17632

theorem simplify_cbrt_8000_eq_21 :
  ∃ (a b : ℕ), a * (b^(1/3)) = 20 * (1^(1/3)) ∧ b = 1 ∧ a + b = 21 :=
by
  sorry

end simplify_cbrt_8000_eq_21_l17_17632


namespace floor_sqrt_80_eq_8_l17_17919

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l17_17919


namespace particle_motion_inverse_relationship_l17_17498

theorem particle_motion_inverse_relationship 
  {k : ℝ} 
  (inverse_relationship : ∀ {n : ℕ}, ∃ t_n d_n, d_n = k / t_n)
  (second_mile : ∃ t_2 d_2, t_2 = 2 ∧ d_2 = 1) : 
  ∃ t_4 d_4, t_4 = 4 ∧ d_4 = 0.5 :=
by
  sorry

end particle_motion_inverse_relationship_l17_17498


namespace circle_radius_three_points_on_line_l17_17694

theorem circle_radius_three_points_on_line :
  ∀ R : ℝ,
  (∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = R^2 → (4 * x + 3 * y = 11) → (dist (x, y) (1, -1) = 1)) →
  R = 3
:= sorry

end circle_radius_three_points_on_line_l17_17694


namespace inequality_inequality_l17_17260

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l17_17260


namespace cost_of_car_l17_17443

theorem cost_of_car (initial_payment : ℕ) (num_installments : ℕ) (installment_amount : ℕ) : 
  initial_payment = 3000 →
  num_installments = 6 →
  installment_amount = 2500 →
  initial_payment + num_installments * installment_amount = 18000 :=
by
  intros h_initial h_num h_installment
  sorry

end cost_of_car_l17_17443


namespace inequality_proof_l17_17285

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l17_17285


namespace manuscript_fee_l17_17497

noncomputable def tax (x : ℝ) : ℝ :=
  if x ≤ 800 then 0
  else if x <= 4000 then 0.14 * (x - 800)
  else 0.11 * x

theorem manuscript_fee (x : ℝ) (h₁ : tax x = 420)
  (h₂ : 800 < x ∧ x ≤ 4000 ∨ x > 4000) :
  x = 3800 :=
sorry

end manuscript_fee_l17_17497


namespace clothes_donation_l17_17504

variable (initial_clothes : ℕ)
variable (clothes_thrown_away : ℕ)
variable (final_clothes : ℕ)
variable (x : ℕ)

theorem clothes_donation (h1 : initial_clothes = 100) 
                        (h2 : clothes_thrown_away = 15) 
                        (h3 : final_clothes = 65) 
                        (h4 : 4 * x = initial_clothes - final_clothes - clothes_thrown_away) :
  x = 5 := by
  sorry

end clothes_donation_l17_17504


namespace alpha_beta_square_eq_eight_l17_17070

theorem alpha_beta_square_eq_eight (α β : ℝ) 
  (hα : α^2 = 2*α + 1) 
  (hβ : β^2 = 2*β + 1) 
  (h_distinct : α ≠ β) : 
  (α - β)^2 = 8 := 
sorry

end alpha_beta_square_eq_eight_l17_17070


namespace Sally_fries_total_l17_17598

theorem Sally_fries_total 
  (sally_fries_initial : ℕ)
  (mark_fries_initial : ℕ)
  (fries_given_by_mark : ℕ)
  (one_third_of_mark_fries : mark_fries_initial = 36 → fries_given_by_mark = mark_fries_initial / 3) :
  sally_fries_initial = 14 → mark_fries_initial = 36 → fries_given_by_mark = 12 →
  let sally_fries_final := sally_fries_initial + fries_given_by_mark
  in sally_fries_final = 26 := 
by
  intros h1 h2 h3
  unfold sally_fries_final
  rw [h1, h3]
  exact rfl

end Sally_fries_total_l17_17598


namespace interior_angle_of_regular_octagon_l17_17784

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l17_17784


namespace double_binom_6_2_l17_17668

theorem double_binom_6_2 : 2 * Nat.choose 6 2 = 30 := by
  sorry

end double_binom_6_2_l17_17668


namespace linen_tablecloth_cost_l17_17237

def num_tables : ℕ := 20
def cost_per_place_setting : ℕ := 10
def num_place_settings_per_table : ℕ := 4
def cost_per_rose : ℕ := 5
def num_roses_per_centerpiece : ℕ := 10
def cost_per_lily : ℕ := 4
def num_lilies_per_centerpiece : ℕ := 15
def total_decoration_cost : ℕ := 3500

theorem linen_tablecloth_cost :
  (total_decoration_cost - (num_tables * num_place_settings_per_table * cost_per_place_setting + num_tables * (num_roses_per_centerpiece * cost_per_rose + num_lilies_per_centerpiece * cost_per_lily))) / num_tables = 25 :=
  sorry

end linen_tablecloth_cost_l17_17237


namespace inequality_holds_for_real_numbers_l17_17268

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l17_17268


namespace inequality_holds_for_all_reals_l17_17294

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l17_17294


namespace floor_sqrt_80_l17_17908

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l17_17908


namespace bubble_bath_per_guest_l17_17972

def rooms_couple : ℕ := 13
def rooms_single : ℕ := 14
def total_bubble_bath : ℕ := 400

theorem bubble_bath_per_guest :
  (total_bubble_bath / (rooms_couple * 2 + rooms_single)) = 10 :=
by
  sorry

end bubble_bath_per_guest_l17_17972


namespace inequality_holds_for_all_real_numbers_l17_17242

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l17_17242


namespace regular_octagon_interior_angle_l17_17840

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17840


namespace fraction_proof_l17_17938

theorem fraction_proof (a b : ℚ) (h : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
by
  sorry

end fraction_proof_l17_17938


namespace smallest_c_for_3_in_range_l17_17391

theorem smallest_c_for_3_in_range : 
  ∀ c : ℝ, (∃ x : ℝ, (x^2 - 6 * x + c) = 3) ↔ (c ≥ 12) :=
by {
  sorry
}

end smallest_c_for_3_in_range_l17_17391


namespace labourer_income_l17_17309

noncomputable def monthly_income : ℤ := 75

theorem labourer_income:
  ∃ (I D : ℤ),
  (80 * 6 = 480) ∧
  (I * 6 - D + (I * 4) = 480 + 240 + D + 30) →
  I = monthly_income :=
by
  sorry

end labourer_income_l17_17309


namespace monotonic_intervals_inequality_condition_l17_17411

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x

theorem monotonic_intervals (m : ℝ) :
  (m ≤ 0 → ∀ x > 0, ∀ y > 0, x < y → f x m < f y m) ∧
  (m > 0 → (∀ x > 0, x < 1/m → ∀ y > x, y < 1/m → f x m < f y m) ∧ (∀ x ≥ 1/m, ∀ y > x, f x m > f y m)) :=
sorry

theorem inequality_condition (m : ℝ) (h : ∀ x ≥ 1, f x m ≤ (m - 1) / x - 2 * m + 1) :
  m ≥ 1/2 :=
sorry

end monotonic_intervals_inequality_condition_l17_17411


namespace custom_op_neg2_neg3_l17_17354

  def custom_op (a b : ℤ) : ℤ := b^2 - a

  theorem custom_op_neg2_neg3 : custom_op (-2) (-3) = 11 :=
  by
    sorry
  
end custom_op_neg2_neg3_l17_17354


namespace equilibrium_price_without_subsidy_increase_in_quantity_due_to_subsidy_l17_17618

-- Definitions for supply and demand functions
def Qs (p : ℝ) : ℝ := 2 + 8 * p
def Qd (p : ℝ) : ℝ := -2 * p + 12

-- Equilibrium without subsidy
theorem equilibrium_price_without_subsidy : (∃ p q, Qs p = q ∧ Qd p = q ∧ p = 1 ∧ q = 10) :=
sorry

-- New supply function with subsidy
def Qs_with_subsidy (p : ℝ) : ℝ := 10 + 8 * p

-- Increase in quantity sold due to subsidy
theorem increase_in_quantity_due_to_subsidy : 
  (∃ Δq, Δq = Qd 0.2 - Qd 1 ∧ Δq = 1.6) :=
sorry

end equilibrium_price_without_subsidy_increase_in_quantity_due_to_subsidy_l17_17618


namespace find_number_l17_17135

theorem find_number (x : ℝ) (h : x = (1 / 3) * x + 120) : x = 180 :=
by
  sorry

end find_number_l17_17135


namespace problem_solution_l17_17220

theorem problem_solution :
  (∀ (p q : ℚ), 
    (∀ (x : ℚ), (x + 3 * p) * (x^2 - x + (1 / 3) * q) = x^3 + (3 * p - 1) * x^2 + ((1 / 3) * q - 3 * p) * x + p * q) →
    (3 * p - 1 = 0) →
    ((1 / 3) * q - 3 * p = 0) →
    p = 1 / 3 ∧ q = 3)
  ∧ ((1 / 3) ^ 2020 * 3 ^ 2021 = 3) :=
by
  sorry

end problem_solution_l17_17220


namespace intersection_P_Q_l17_17980

def setP : Set ℝ := {1, 2, 3, 4}
def setQ : Set ℝ := {x | abs x ≤ 2}

theorem intersection_P_Q : (setP ∩ setQ) = {1, 2} :=
by
  sorry

end intersection_P_Q_l17_17980


namespace pirate_coins_l17_17020

theorem pirate_coins (x : ℕ) (hn : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 15 → ∃ y : ℕ, y = (2 * k * x) / 15) : 
  ∃ y : ℕ, y = 630630 :=
by sorry

end pirate_coins_l17_17020


namespace least_positive_integer_reducible_fraction_l17_17530

theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, (n > 0) ∧ Nat.gcd (n - 17) (3 * n + 4) > 1 ∧ ∀ m : ℕ, (m > 0) ∧ Nat.gcd (m - 17) (3 * m + 4) > 1 → n ≤ m :=
begin
  use 22,
  split,
  { exact nat.succ_pos' 21 },
  split,
  { calc
      Nat.gcd (22 - 17) (3 * 22 + 4)
          = Nat.gcd 5 70 : by rw [sub_eq_add_neg, add_comm 17]; norm_num
      ... = 5 : by norm_num },
  { intros m hm,
    cases hm with _ h,
    have h1 : 3 * m + 4 = 3 * m + 4, by refl,
    rw [←Nat.gcd_zero_left (3 * m + 4), ←mod_eq_sub_mod hm.1, gcd_add_mul_right_left _ _ 3, gcd_comm_self, gcd_great_or_equal_iff_eq_if_dvd] at h,
    { cases h,
      { subst m },
      { refl } },
    { exact Iff.find_left sorry (Nat.gcd_pos_of_pos_right _ (nat.succ_pos' m)) } }
end

end least_positive_integer_reducible_fraction_l17_17530


namespace parabola_focus_coordinates_l17_17050

theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), x^2 = 8 * y → ∃ F : ℝ × ℝ, F = (0, 2) :=
  sorry

end parabola_focus_coordinates_l17_17050


namespace marks_in_biology_l17_17045

theorem marks_in_biology (E M P C : ℝ) (A B : ℝ)
  (h1 : E = 90)
  (h2 : M = 92)
  (h3 : P = 85)
  (h4 : C = 87)
  (h5 : A = 87.8) 
  (h6 : (E + M + P + C + B) / 5 = A) : 
  B = 85 := 
by
  -- Placeholder for the proof
  sorry

end marks_in_biology_l17_17045


namespace mira_jogs_hours_each_morning_l17_17743

theorem mira_jogs_hours_each_morning 
  (h : ℝ) -- number of hours Mira jogs each morning
  (speed : ℝ) -- Mira's jogging speed in miles per hour
  (days : ℝ) -- number of days Mira jogs
  (total_distance : ℝ) -- total distance Mira jogs

  (H1 : speed = 5) 
  (H2 : days = 5) 
  (H3 : total_distance = 50) 
  (H4 : total_distance = speed * h * days) :

  h = 2 :=
by
  sorry

end mira_jogs_hours_each_morning_l17_17743


namespace regular_octagon_interior_angle_l17_17800

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l17_17800


namespace square_not_end_with_four_identical_digits_l17_17601

theorem square_not_end_with_four_identical_digits (n : ℕ) (d : ℕ) :
  n = d * d → ¬ (d ≠ 0 ∧ (n % 10000 = d ^ 4)) :=
by
  sorry

end square_not_end_with_four_identical_digits_l17_17601


namespace rose_days_to_complete_work_l17_17091

theorem rose_days_to_complete_work (R : ℝ) (h1 : 1 / 10 + 1 / R = 1 / 8) : R = 40 := 
sorry

end rose_days_to_complete_work_l17_17091


namespace people_eating_vegetarian_l17_17854

theorem people_eating_vegetarian (only_veg : ℕ) (both_veg_nonveg : ℕ) (total_veg : ℕ) :
  only_veg = 13 ∧ both_veg_nonveg = 6 → total_veg = 19 := 
by
  sorry

end people_eating_vegetarian_l17_17854


namespace floor_sqrt_80_l17_17899

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l17_17899


namespace find_c_l17_17179

theorem find_c (a b c : ℤ) (h1 : c ≥ 0) (h2 : ¬∃ m : ℤ, 2 * a * b = m^2)
  (h3 : ∀ n : ℕ, n > 0 → (a^n + (2 : ℤ)^n) ∣ (b^n + c)) :
  c = 0 ∨ c = 1 :=
by
  sorry

end find_c_l17_17179


namespace correct_rounded_result_l17_17735

def round_to_nearest_ten (n : ℤ) : ℤ :=
  (n + 5) / 10 * 10

theorem correct_rounded_result :
  round_to_nearest_ten ((57 + 68) * 2) = 250 :=
by
  sorry

end correct_rounded_result_l17_17735


namespace find_inverse_sum_l17_17204

theorem find_inverse_sum (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + a = b / a + b) : 1 / a + 1 / b = -1 :=
sorry

end find_inverse_sum_l17_17204


namespace floor_neg_seven_fourths_l17_17887

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l17_17887


namespace cube_traversal_count_l17_17521

-- Defining the cube traversal problem
def cube_traversal (num_faces : ℕ) (adj_faces : ℕ) (visits : ℕ) : ℕ :=
  if (num_faces = 6 ∧ adj_faces = 4) then
    4 * 2
  else
    0

-- Theorem statement
theorem cube_traversal_count : 
  cube_traversal 6 4 1 = 8 :=
by
  -- Skipping the proof with sorry for now
  sorry

end cube_traversal_count_l17_17521


namespace inequality_solution_l17_17055

theorem inequality_solution (m : ℝ) (h : m < -1) :
  (if m = -3 then
    {x : ℝ | x > 1} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if -3 < m ∧ m < -1 then
    ({x : ℝ | x < m / (m + 3)} ∪ {x : ℝ | x > 1}) =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if m < -3 then
    {x : ℝ | 1 < x ∧ x < m / (m + 3)} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else
    False) :=
by
  sorry

end inequality_solution_l17_17055


namespace total_tickets_needed_l17_17127

-- Define the conditions
def rollercoaster_rides (n : Nat) := 3
def catapult_rides (n : Nat) := 2
def ferris_wheel_rides (n : Nat) := 1
def rollercoaster_cost (n : Nat) := 4
def catapult_cost (n : Nat) := 4
def ferris_wheel_cost (n : Nat) := 1

-- Prove the total number of tickets needed
theorem total_tickets_needed : 
  rollercoaster_rides 0 * rollercoaster_cost 0 +
  catapult_rides 0 * catapult_cost 0 +
  ferris_wheel_rides 0 * ferris_wheel_cost 0 = 21 :=
by 
  sorry

end total_tickets_needed_l17_17127


namespace correct_calculation_l17_17005

-- Definitions of the conditions
def condition_A (a : ℝ) : Prop := a^2 + a^2 = a^4
def condition_B (a : ℝ) : Prop := 3 * a^2 + 2 * a^2 = 5 * a^2
def condition_C (a : ℝ) : Prop := a^4 - a^2 = a^2
def condition_D (a : ℝ) : Prop := 3 * a^2 - 2 * a^2 = 1

-- The theorem statement
theorem correct_calculation (a : ℝ) : condition_B a := by 
sorry

end correct_calculation_l17_17005


namespace point_M_coordinates_l17_17595

theorem point_M_coordinates :
  ∃ M : ℝ × ℝ × ℝ, 
    M.1 = 0 ∧ M.2.1 = 0 ∧  
    (dist (1, 0, 2) (M.1, M.2.1, M.2.2) = dist (1, -3, 1) (M.1, M.2.1, M.2.2)) ∧ 
    M = (0, 0, -3) :=
by
  sorry

end point_M_coordinates_l17_17595


namespace percent_decrease_is_30_l17_17724

def original_price : ℝ := 100
def sale_price : ℝ := 70
def decrease_in_price : ℝ := original_price - sale_price

theorem percent_decrease_is_30 : (decrease_in_price / original_price) * 100 = 30 :=
by
  sorry

end percent_decrease_is_30_l17_17724


namespace inequality_holds_for_real_numbers_l17_17271

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l17_17271


namespace length_of_arc_l17_17995

theorem length_of_arc (C : ℝ) (θ : ℝ) (DE : ℝ) (c_circ : C = 100) (angle : θ = 120) :
  DE = 100 / 3 :=
by
  -- Place the actual proof here.
  sorry

end length_of_arc_l17_17995


namespace correct_option_l17_17138

-- Definitions for conditions
def C1 (a : ℕ) : Prop := a^2 * a^3 = a^5
def C2 (a : ℕ) : Prop := a + 2 * a = 3 * a^2
def C3 (a b : ℕ) : Prop := (a * b)^3 = a * b^3
def C4 (a : ℕ) : Prop := (-a^3)^2 = -a^6

-- The correct option is C1
theorem correct_option (a : ℕ) : C1 a := by
  sorry

end correct_option_l17_17138


namespace avg_salary_increase_l17_17454

def initial_avg_salary : ℝ := 1700
def num_employees : ℕ := 20
def manager_salary : ℝ := 3800

theorem avg_salary_increase :
  ((num_employees * initial_avg_salary + manager_salary) / (num_employees + 1)) - initial_avg_salary = 100 :=
by
  sorry

end avg_salary_increase_l17_17454


namespace power_of_expression_l17_17439

theorem power_of_expression (a b c d e : ℝ)
  (h1 : a - b - c + d = 18)
  (h2 : a + b - c - d = 6)
  (h3 : c + d - e = 5) :
  (2 * b - d + e) ^ 3 = 13824 :=
by
  sorry

end power_of_expression_l17_17439


namespace remainder_2011_2015_mod_23_l17_17000

theorem remainder_2011_2015_mod_23 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 23 = 5 := 
by
  sorry

end remainder_2011_2015_mod_23_l17_17000


namespace floor_sqrt_80_l17_17914

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l17_17914


namespace balloons_remaining_l17_17401
-- Importing the necessary libraries

-- Defining the conditions
def originalBalloons : Nat := 709
def givenBalloons : Nat := 221

-- Stating the theorem
theorem balloons_remaining : originalBalloons - givenBalloons = 488 := by
  sorry

end balloons_remaining_l17_17401


namespace simplify_expression_l17_17326

variable (x : ℝ)

theorem simplify_expression :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x) ^ 3) = (25 / 8) * x^2 :=
by
  sorry

end simplify_expression_l17_17326


namespace length_OC_l17_17009

theorem length_OC (a b : ℝ) (h_perpendicular : ∀ x, x^2 + a * x + b = 0 → x = 1 ∨ x = b) : 
  1 = 1 :=
by 
  sorry

end length_OC_l17_17009


namespace intersection_point_l17_17608

theorem intersection_point :
  ∃ (x y : ℝ), (y = 2 * x) ∧ (x + y = 3) ∧ (x = 1) ∧ (y = 2) := 
by
  sorry

end intersection_point_l17_17608


namespace xy_condition_l17_17704

theorem xy_condition (x y z : ℝ) (hxz : x ≠ z) (hxy : x ≠ y) (hyz : y ≠ z) (posx : 0 < x) (posy : 0 < y) (posz : 0 < z) 
  (h : y / (x - z) = (x + y) / z ∧ (x + y) / z = x / y) : x / y = 2 :=
by
  sorry

end xy_condition_l17_17704


namespace find_natural_numbers_l17_17992

theorem find_natural_numbers (x y z : ℕ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_ordered : x < y ∧ y < z)
  (h_reciprocal_sum_nat : ∃ a : ℕ, 1/x + 1/y + 1/z = a) : (x, y, z) = (2, 3, 6) := 
sorry

end find_natural_numbers_l17_17992


namespace solve_x_l17_17707

theorem solve_x (x : ℝ) (h : (x / 3) / 5 = 5 / (x / 3)) : x = 15 ∨ x = -15 :=
by sorry

end solve_x_l17_17707


namespace smallest_side_of_triangle_l17_17568

theorem smallest_side_of_triangle (A B C : ℝ) (a b c : ℝ) 
  (hA : A = 60) (hC : C = 45) (hb : b = 4) (h_sum : A + B + C = 180) : 
  c = 4 * Real.sqrt 3 - 4 := 
sorry

end smallest_side_of_triangle_l17_17568


namespace students_in_diligence_before_transfer_l17_17967

theorem students_in_diligence_before_transfer (D I P : ℕ)
  (h_total : D + I + P = 75)
  (h_equal : D + 2 = I - 2 + 3 ∧ D + 2 = P - 3) :
  D = 23 :=
by
  sorry

end students_in_diligence_before_transfer_l17_17967


namespace perfect_square_difference_of_solutions_l17_17231

theorem perfect_square_difference_of_solutions
  (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℤ, k^2 = x - y := 
sorry

end perfect_square_difference_of_solutions_l17_17231


namespace log_expression_simplify_l17_17300

variable (x y : ℝ)

theorem log_expression_simplify (hx : 0 < x) (hx' : x ≠ 1) (hy : 0 < y) (hy' : y ≠ 1) :
  (Real.log x^2 / Real.log y^4) * 
  (Real.log y^3 / Real.log x^3) * 
  (Real.log x^4 / Real.log y^5) * 
  (Real.log y^5 / Real.log x^2) * 
  (Real.log x^3 / Real.log y^3) = (1 / 3) * Real.log x / Real.log y := 
sorry

end log_expression_simplify_l17_17300


namespace mary_max_earnings_l17_17587

theorem mary_max_earnings
  (max_hours : ℕ)
  (regular_rate : ℕ)
  (overtime_rate_increase_percent : ℕ)
  (first_hours : ℕ)
  (total_max_hours : ℕ)
  (total_hours_payable : ℕ) :
  max_hours = 60 →
  regular_rate = 8 →
  overtime_rate_increase_percent = 25 →
  first_hours = 20 →
  total_max_hours = 60 →
  total_hours_payable = 560 →
  ((first_hours * regular_rate) + ((total_max_hours - first_hours) * (regular_rate + (regular_rate * overtime_rate_increase_percent / 100)))) = total_hours_payable :=
by
  intros
  sorry

end mary_max_earnings_l17_17587


namespace log_inequality_sqrt_inequality_l17_17297

-- Proof problem for part (1)
theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((a + b) / 2) ≥ (Real.log a + Real.log b) / 2 :=
sorry

-- Proof problem for part (2)
theorem sqrt_inequality :
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end log_inequality_sqrt_inequality_l17_17297


namespace initial_people_lifting_weights_l17_17035

theorem initial_people_lifting_weights (x : ℕ) (h : x + 3 = 19) : x = 16 :=
by
  sorry

end initial_people_lifting_weights_l17_17035


namespace norb_age_is_47_l17_17746

section NorbAge

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def exactlyHalfGuessesTooLow (guesses : List ℕ) (age : ℕ) : Prop :=
  (guesses.filter (λ x => x < age)).length = (guesses.length / 2)

def oneGuessOffByTwo (guesses : List ℕ) (age : ℕ) : Prop :=
  guesses.any (λ x => x = age + 2 ∨ x = age - 2)

def validAge (guesses : List ℕ) (age : ℕ) : Prop :=
  exactlyHalfGuessesTooLow guesses age ∧ oneGuessOffByTwo guesses age ∧ isPrime age

theorem norb_age_is_47 : validAge [23, 29, 33, 35, 39, 41, 46, 48, 50, 54] 47 :=
sorry

end NorbAge

end norb_age_is_47_l17_17746


namespace manny_paula_weight_l17_17093

   variable (m n o p : ℕ)

   -- Conditions
   variable (h1 : m + n = 320) 
   variable (h2 : n + o = 295) 
   variable (h3 : o + p = 310) 

   theorem manny_paula_weight : m + p = 335 :=
   by
     sorry
   
end manny_paula_weight_l17_17093


namespace biggest_number_in_ratio_l17_17537

theorem biggest_number_in_ratio (x : ℕ) (h_sum : 2 * x + 3 * x + 4 * x + 5 * x = 1344) : 5 * x = 480 := 
by
  sorry

end biggest_number_in_ratio_l17_17537


namespace quinary_to_octal_444_l17_17041

theorem quinary_to_octal_444 :
  (let quinary := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  let decimal := 124 in
  let octal := 1 * 8^2 + 7 * 8^1 + 4 * 8^0 in
  quinary = decimal ∧ decimal = octal :=
  quinary = 4 * 25 + 4 * 5 + 4 ∧ 124 = 1 * 64 + 7 * 8 + 4) :=
by
  sorry

end quinary_to_octal_444_l17_17041


namespace possible_k_values_l17_17971

def triangle_right_k_values (AB AC : ℝ × ℝ) (k : ℝ) : Prop :=
  let BC := (AC.1 - AB.1, AC.2 - AB.2)
  let angle_A := AB.1 * AC.1 + AB.2 * AC.2 = 0   -- Condition for ∠A = 90°
  let angle_B := AB.1 * BC.1 + AB.2 * BC.2 = 0   -- Condition for ∠B = 90°
  let angle_C := BC.1 * AC.1 + BC.2 * AC.2 = 0   -- Condition for ∠C = 90°
  (angle_A ∨ angle_B ∨ angle_C)

theorem possible_k_values (k : ℝ) :
  triangle_right_k_values (2, 3) (1, k) k ↔
  k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13) / 2 :=
by
  sorry

end possible_k_values_l17_17971


namespace product_of_numbers_l17_17122

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := 
sorry

end product_of_numbers_l17_17122


namespace ratio_of_volume_to_surface_area_l17_17155

-- Define conditions
def unit_cube_volume : ℕ := 1
def unit_cube_exposed_faces_at_ends : ℕ := 5
def unit_cube_exposed_faces_in_middle : ℕ := 4
def num_end_cubes : ℕ := 2
def num_middle_cubes : ℕ := 6
def total_cubes : ℕ := 8

-- Define volume
def volume_of_shape : ℕ := total_cubes * unit_cube_volume

-- Define surface area
def surface_area_of_shape : ℕ := 
  num_end_cubes * unit_cube_exposed_faces_at_ends + 
  num_middle_cubes * unit_cube_exposed_faces_in_middle

-- Define ratio
def volume_to_surface_area_ratio : ℚ := 
  (volume_of_shape : ℚ) / (surface_area_of_shape : ℚ)

-- Proposition stating the ratio
theorem ratio_of_volume_to_surface_area :
  volume_to_surface_area_ratio = 4 / 17 := by
  sorry

end ratio_of_volume_to_surface_area_l17_17155


namespace floor_neg_seven_fourths_l17_17896

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l17_17896


namespace additional_sugar_is_correct_l17_17627

def sugar_needed : ℝ := 450
def sugar_in_house : ℝ := 287
def sugar_in_basement_kg : ℝ := 50
def kg_to_lbs : ℝ := 2.20462

def sugar_in_basement : ℝ := sugar_in_basement_kg * kg_to_lbs
def total_sugar : ℝ := sugar_in_house + sugar_in_basement
def additional_sugar_needed : ℝ := sugar_needed - total_sugar

theorem additional_sugar_is_correct : additional_sugar_needed = 52.769 := by
  sorry

end additional_sugar_is_correct_l17_17627


namespace smallest_sum_B_c_l17_17563

theorem smallest_sum_B_c 
  (B c : ℕ) 
  (h1 : B ≤ 4) 
  (h2 : 6 < c) 
  (h3 : 31 * B = 4 * (c + 1)) : 
  B + c = 34 := 
sorry

end smallest_sum_B_c_l17_17563


namespace arithmetic_sequence_sum_l17_17086

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) 
(h : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) 
(h_a : ∀ n, a n = a 1 + (n - 1) * d) : a 2 + a 10 = 120 :=
by
  sorry

end arithmetic_sequence_sum_l17_17086


namespace transformed_equation_sum_l17_17079

theorem transformed_equation_sum (a b : ℝ) (h_eqn : ∀ x : ℝ, x^2 - 6 * x - 5 = 0 ↔ (x + a)^2 = b) :
  a + b = 11 :=
sorry

end transformed_equation_sum_l17_17079


namespace smallest_sum_B_c_l17_17561

theorem smallest_sum_B_c (B : ℕ) (c : ℕ) (hB : B < 5) (hc : c > 6) :
  31 * B = 4 * c + 4 → (B + c) = 34 :=
by
  sorry

end smallest_sum_B_c_l17_17561


namespace correct_propositions_l17_17997

variables (a b : ℝ) (x : ℝ) (a_max : ℝ)

/-- Given propositions to analyze. -/
noncomputable def propositions :=
  ((a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3) ∧
  ((¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ∧
  (a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max))

/-- The main theorem stating which propositions are correct -/
theorem correct_propositions (h1 : a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3)
                            (h2 : (¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0)
                            (h3 : a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max) :
  propositions a b a_max :=
by
  sorry

end correct_propositions_l17_17997


namespace probability_of_yellow_jelly_bean_l17_17864

theorem probability_of_yellow_jelly_bean (P_red P_orange P_yellow : ℝ) 
  (h1 : P_red = 0.2) 
  (h2 : P_orange = 0.5) 
  (h3 : P_red + P_orange + P_yellow = 1) : 
  P_yellow = 0.3 :=
sorry

end probability_of_yellow_jelly_bean_l17_17864


namespace sum_of_x_and_y_l17_17712

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 240) : x + y = 680 :=
by
  sorry

end sum_of_x_and_y_l17_17712


namespace smallest_possible_fourth_chair_l17_17539

/--
Four special prizes at a school sports day are hidden beneath each chair numbered with two-digit positive integers. 
The first three numbers found are 45, 26, and 63, but the label on the last chair is partially ripped, leaving the number unreadable. 
The sum of the digits of all four numbers equals one-third of the sum of all four numbers. 
Additionally, the total sum of all numbers must be a multiple of 7. 
Prove that the smallest possible number for the fourth chair is 37.
-/
theorem smallest_possible_fourth_chair : 
  ∃ (x : ℕ), (134 + x) % 7 = 0 ∧ x < 100 ∧ ( (26 + x.digits.sum) * 3 = 134 + x ) ∧ x = 37 := 
by 
  sorry

end smallest_possible_fourth_chair_l17_17539


namespace pies_difference_l17_17364

theorem pies_difference (time : ℕ) (alice_time : ℕ) (bob_time : ℕ) (charlie_time : ℕ)
    (h_time : time = 90) (h_alice : alice_time = 5) (h_bob : bob_time = 6) (h_charlie : charlie_time = 7) :
    (time / alice_time - time / bob_time) + (time / alice_time - time / charlie_time) = 9 := by
  sorry

end pies_difference_l17_17364


namespace craig_apples_after_sharing_l17_17043

-- Defining the initial conditions
def initial_apples_craig : ℕ := 20
def shared_apples : ℕ := 7

-- The proof statement
theorem craig_apples_after_sharing : 
  initial_apples_craig - shared_apples = 13 := 
by
  sorry

end craig_apples_after_sharing_l17_17043


namespace solution_set_f_ge_0_l17_17697

noncomputable def f (x a : ℝ) : ℝ := 1 / Real.exp x - a / x

theorem solution_set_f_ge_0 (a m n : ℝ) (h : ∀ x, m ≤ x ∧ x ≤ n ↔ 1 / Real.exp x - a / x ≥ 0) : 
  0 < a ∧ a < 1 / Real.exp 1 :=
  sorry

end solution_set_f_ge_0_l17_17697


namespace problem1_proof_problem2_proof_l17_17692

noncomputable def problem1 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a| + |b| ≤ Real.sqrt 2

noncomputable def problem2 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a^3 / b| + |b^3 / a| ≥ 1

theorem problem1_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem1 a b h₁ h₂ h₃ :=
  sorry

theorem problem2_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem2 a b h₁ h₂ h₃ :=
  sorry

end problem1_proof_problem2_proof_l17_17692


namespace find_B_l17_17966

noncomputable def B_solution (A a b : ℝ) : set ℝ :=
  {B | A = π / 6 ∧ a = 1 ∧ b = sqrt 3 → B = π / 3 ∨ B = 2 * π / 3}

theorem find_B (A a b B : ℝ) (hA : A = π / 6) (ha : a = 1) (hb : b = sqrt 3) : B_solution A a b B :=
by
  have h : ∃ B, A = π / 6 ∧ a = 1 ∧ b = sqrt 3 → B = π / 3 ∨ B = 2 * π / 3 := 
  sorry
  exact h

end find_B_l17_17966


namespace odd_number_as_diff_of_squares_l17_17416

theorem odd_number_as_diff_of_squares :
    ∀ (x y : ℤ), 63 = x^2 - y^2 ↔ (x = 32 ∧ y = 31) ∨ (x = 12 ∧ y = 9) ∨ (x = 8 ∧ y = 1) := 
by
  sorry

end odd_number_as_diff_of_squares_l17_17416


namespace max_sum_clock_digits_l17_17351

theorem max_sum_clock_digits : ∃ t : ℕ, 0 ≤ t ∧ t < 24 ∧ 
  (∃ h1 h2 m1 m2 : ℕ, t = h1 * 10 + h2 + m1 * 10 + m2 ∧ 
   (0 ≤ h1 ∧ h1 ≤ 2) ∧ (0 ≤ h2 ∧ h2 ≤ 9) ∧ (0 ≤ m1 ∧ m1 ≤ 5) ∧ (0 ≤ m2 ∧ m2 ≤ 9) ∧ 
   h1 + h2 + m1 + m2 = 24) := sorry

end max_sum_clock_digits_l17_17351


namespace find_total_cards_l17_17769

def numCardsInStack (n : ℕ) : Prop :=
  let cards : List ℕ := List.range' 1 (2 * n + 1)
  let pileA := cards.take n
  let pileB := cards.drop n
  let restack := List.zipWith (fun x y => [y, x]) pileA pileB |> List.join
  (restack.take 13).getLastD 0 = 13 ∧ 2 * n = 26

theorem find_total_cards : ∃ (n : ℕ), numCardsInStack n :=
sorry

end find_total_cards_l17_17769


namespace hyperbola_eccentricity_range_l17_17412

-- Lean 4 statement for the given problem.
theorem hyperbola_eccentricity_range {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (h : ∀ (x y : ℝ), y = x * Real.sqrt 3 → y^2 / b^2 - x^2 / a^2 = 1 ∨ ∃ (z : ℝ), y = x * Real.sqrt 3 ∧ z^2 / b^2 - x^2 / a^2 = 1) :
  1 < Real.sqrt (a^2 + b^2) / a ∧ Real.sqrt (a^2 + b^2) / a < 2 :=
by
  sorry

end hyperbola_eccentricity_range_l17_17412


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17792

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17792


namespace floor_sqrt_80_l17_17904

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l17_17904


namespace ratio_of_x_intercepts_l17_17628

theorem ratio_of_x_intercepts (b s t : ℝ) (hb : b ≠ 0) (h1 : s = -b / 8) (h2 : t = -b / 4) : s / t = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l17_17628


namespace floor_sqrt_80_l17_17901

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l17_17901


namespace dogs_not_liking_any_l17_17573

variables (totalDogs : ℕ) (dogsLikeWatermelon : ℕ) (dogsLikeSalmon : ℕ) (dogsLikeBothSalmonWatermelon : ℕ)
          (dogsLikeChicken : ℕ) (dogsLikeWatermelonNotSalmon : ℕ) (dogsLikeSalmonChickenNotWatermelon : ℕ)

theorem dogs_not_liking_any : totalDogs = 80 → dogsLikeWatermelon = 21 → dogsLikeSalmon = 58 →
  dogsLikeBothSalmonWatermelon = 12 → dogsLikeChicken = 15 →
  dogsLikeWatermelonNotSalmon = 7 → dogsLikeSalmonChickenNotWatermelon = 10 →
  (totalDogs - ((dogsLikeSalmon - (dogsLikeBothSalmonWatermelon + dogsLikeSalmonChickenNotWatermelon)) +
                (dogsLikeWatermelon - (dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon)) +
                (dogsLikeChicken - (dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) +
                dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) = 13 :=
by
  intros h_totalDogs h_dogsLikeWatermelon h_dogsLikeSalmon h_dogsLikeBothSalmonWatermelon 
         h_dogsLikeChicken h_dogsLikeWatermelonNotSalmon h_dogsLikeSalmonChickenNotWatermelon
  sorry

end dogs_not_liking_any_l17_17573


namespace picture_area_l17_17557

theorem picture_area (x y : ℕ) (hx : 1 < x) (hy : 1 < y) 
  (h_area : (3 * x + 4) * (y + 3) = 60) : x * y = 15 := 
by 
  sorry

end picture_area_l17_17557


namespace students_with_certificates_l17_17369

variable (C N : ℕ)

theorem students_with_certificates :
  (C + N = 120) ∧ (C = N + 36) → C = 78 :=
by
  sorry

end students_with_certificates_l17_17369


namespace annual_interest_rate_l17_17102

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate :
  compound_interest_rate 150 181.50 2 1 (0.2 : ℝ) :=
by
  unfold compound_interest_rate
  sorry

end annual_interest_rate_l17_17102


namespace inequality_xyz_l17_17280

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l17_17280


namespace inequality_holds_for_real_numbers_l17_17265

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l17_17265


namespace ursula_annual_salary_l17_17324

def hourly_wage : ℝ := 8.50
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

noncomputable def daily_earnings : ℝ := hourly_wage * hours_per_day
noncomputable def monthly_earnings : ℝ := daily_earnings * days_per_month
noncomputable def annual_salary : ℝ := monthly_earnings * months_per_year

theorem ursula_annual_salary : annual_salary = 16320 := 
  by sorry

end ursula_annual_salary_l17_17324


namespace polynomial_roots_property_l17_17731

theorem polynomial_roots_property (a b : ℝ) (h : ∀ x, x^2 + x - 2024 = 0 → x = a ∨ x = b) : 
  a^2 + 2 * a + b = 2023 :=
by
  sorry

end polynomial_roots_property_l17_17731


namespace eval_otimes_l17_17522

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem eval_otimes : otimes 4 2 = 18 :=
by
  sorry

end eval_otimes_l17_17522


namespace porter_previous_painting_price_l17_17596

variable (P : ℝ)

-- Conditions
def condition1 : Prop := 3.5 * P - 1000 = 49000

-- Correct Answer
def answer : ℝ := 14285.71

-- Theorem stating that the answer holds given the conditions
theorem porter_previous_painting_price (h : condition1 P) : P = answer :=
sorry

end porter_previous_painting_price_l17_17596


namespace count_irreducible_fractions_l17_17085

theorem count_irreducible_fractions (s : Finset ℕ) (h1 : ∀ n ∈ s, 15*n > 15/16) (h2 : ∀ n ∈ s, n < 1) (h3 : ∀ n ∈ s, Nat.gcd n 15 = 1) :
  s.card = 8 := 
sorry

end count_irreducible_fractions_l17_17085


namespace box_filling_possibilities_l17_17676

def possible_numbers : List ℕ := [2015, 2016, 2017, 2018, 2019]

def fill_the_boxes (D O G C W : ℕ) : Prop :=
  D + O + G = C + O + W

theorem box_filling_possibilities :
  (∃ D O G C W : ℕ, 
    D ∈ possible_numbers ∧
    O ∈ possible_numbers ∧
    G ∈ possible_numbers ∧
    C ∈ possible_numbers ∧
    W ∈ possible_numbers ∧
    D ≠ O ∧ D ≠ G ∧ D ≠ C ∧ D ≠ W ∧
    O ≠ G ∧ O ≠ C ∧ O ≠ W ∧
    G ≠ C ∧ G ≠ W ∧
    C ≠ W ∧
    fill_the_boxes D O G C W) → 
    ∃ ways : ℕ, ways = 24 :=
  sorry

end box_filling_possibilities_l17_17676


namespace range_of_a_l17_17081

variable (x a : ℝ)
def inequality_sys := x < a ∧ x < 3
def solution_set := x < a

theorem range_of_a (h : ∀ x, inequality_sys x a → solution_set x a) : a ≤ 3 := by
  sorry

end range_of_a_l17_17081


namespace regular_octagon_interior_angle_l17_17838

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17838


namespace cricket_team_captain_age_l17_17311

theorem cricket_team_captain_age
    (C W : ℕ)
    (h1 : W = C + 3)
    (h2 : (23 * 11) = (22 * 9) + C + W)
    : C = 26 :=
by
    sorry

end cricket_team_captain_age_l17_17311


namespace total_crayons_correct_l17_17142

-- Define the number of crayons each child has
def crayons_per_child : ℕ := 12

-- Define the number of children
def number_of_children : ℕ := 18

-- Define the total number of crayons
def total_crayons : ℕ := crayons_per_child * number_of_children

-- State the theorem
theorem total_crayons_correct : total_crayons = 216 :=
by
  -- Proof goes here
  sorry

end total_crayons_correct_l17_17142


namespace vector_norm_sq_sum_l17_17730

theorem vector_norm_sq_sum (a b : ℝ × ℝ) (m : ℝ × ℝ) (h_m : m = (4, 6))
  (h_midpoint : m = ((2 * a.1 + 2 * b.1) / 2, (2 * a.2 + 2 * b.2) / 2))
  (h_dot : a.1 * b.1 + a.2 * b.2 = 10) :
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 32 :=
by 
  sorry

end vector_norm_sq_sum_l17_17730


namespace num_squares_in_6_by_6_grid_l17_17644

def squares_in_grid (m n : ℕ) : ℕ :=
  (m - 1) * (m - 1) + (m - 2) * (m - 2) + 
  (m - 3) * (m - 3) + (m - 4) * (m - 4) + 
  (m - 5) * (m - 5)

theorem num_squares_in_6_by_6_grid : squares_in_grid 6 6 = 55 := 
by 
  sorry

end num_squares_in_6_by_6_grid_l17_17644


namespace inequality_proof_l17_17256

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l17_17256


namespace correct_proposition_l17_17333

-- Define the propositions as Lean 4 statements.
def PropA (a : ℝ) : Prop := a^4 + a^2 = a^6
def PropB (a : ℝ) : Prop := (-2 * a^2)^3 = -6 * a^8
def PropC (a : ℝ) : Prop := 6 * a - a = 5
def PropD (a : ℝ) : Prop := a^2 * a^3 = a^5

-- The main theorem statement that only PropD is true.
theorem correct_proposition (a : ℝ) : ¬ PropA a ∧ ¬ PropB a ∧ ¬ PropC a ∧ PropD a :=
by
  sorry

end correct_proposition_l17_17333


namespace floor_sqrt_80_l17_17902

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l17_17902


namespace perimeter_of_inner_polygon_le_outer_polygon_l17_17597

-- Definitions of polygons (for simplicity considered as list of points or sides)
structure Polygon where
  sides : List ℝ  -- assuming sides lengths are given as list of real numbers
  convex : Prop   -- a property stating that the polygon is convex

-- Definition of the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := p.sides.sum

-- Conditions from the problem
variable {P_in P_out : Polygon}
variable (h_convex_in : P_in.convex) (h_convex_out : P_out.convex)
variable (h_inside : ∀ s ∈ P_in.sides, s ∈ P_out.sides) -- simplifying the "inside" condition

-- The theorem statement
theorem perimeter_of_inner_polygon_le_outer_polygon :
  perimeter P_in ≤ perimeter P_out :=
by {
  sorry
}

end perimeter_of_inner_polygon_le_outer_polygon_l17_17597


namespace liza_final_balance_l17_17593

def initial_balance : ℕ := 800
def rent : ℕ := 450
def paycheck : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

theorem liza_final_balance :
  initial_balance - rent + paycheck - (electricity_bill + internet_bill) - phone_bill = 1563 := by
  sorry

end liza_final_balance_l17_17593


namespace Janet_horses_l17_17976

theorem Janet_horses (acres : ℕ) (gallons_per_acre : ℕ) (spread_acres_per_day : ℕ) (total_days : ℕ)
  (gallons_per_day_per_horse : ℕ) (total_gallons_needed : ℕ) (total_gallons_spread : ℕ) (horses : ℕ) :
  acres = 20 ->
  gallons_per_acre = 400 ->
  spread_acres_per_day = 4 ->
  total_days = 25 ->
  gallons_per_day_per_horse = 5 ->
  total_gallons_needed = acres * gallons_per_acre ->
  total_gallons_spread = spread_acres_per_day * gallons_per_acre * total_days ->
  horses = total_gallons_needed / (gallons_per_day_per_horse * total_days) ->
  horses = 64 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end Janet_horses_l17_17976


namespace probability_at_least_one_boy_and_one_girl_l17_17763

noncomputable def mathematics_club_prob : ℚ :=
  let boys := 14
  let girls := 10
  let total_members := 24
  let total_committees := Nat.choose total_members 5
  let boys_committees := Nat.choose boys 5
  let girls_committees := Nat.choose girls 5
  let committees_with_at_least_one_boy_and_one_girl := total_committees - (boys_committees + girls_committees)
  let probability := (committees_with_at_least_one_boy_and_one_girl : ℚ) / (total_committees : ℚ)
  probability

theorem probability_at_least_one_boy_and_one_girl :
  mathematics_club_prob = (4025 : ℚ) / 4251 :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l17_17763


namespace scientific_notation_470000000_l17_17747

theorem scientific_notation_470000000 : 470000000 = 4.7 * 10^8 :=
by
  sorry

end scientific_notation_470000000_l17_17747


namespace fraction_relation_l17_17212

theorem fraction_relation (a b : ℝ) (h : a / b = 2 / 3) : (a - b) / b = -1 / 3 :=
by
  sorry

end fraction_relation_l17_17212


namespace fraction_money_left_zero_l17_17037

-- Defining variables and conditions
variables {m c : ℝ} -- m: total money, c: total cost of CDs

-- Condition under the problem statement
def uses_one_fourth_of_money_to_buy_one_fourth_of_CDs (m c : ℝ) := (1 / 4) * m = (1 / 4) * c

-- The conjecture to be proven
theorem fraction_money_left_zero 
  (h: uses_one_fourth_of_money_to_buy_one_fourth_of_CDs m c) 
  (h_eq: c = m) : 
  (m - c) / m = 0 := 
by
  sorry

end fraction_money_left_zero_l17_17037


namespace range_of_x1_f_x2_l17_17944

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 * Real.exp 1 else Real.exp x / x^2

theorem range_of_x1_f_x2:
  ∃ (x1 x2 : ℝ), x1 ≤ 0 ∧ 0 < x2 ∧ f x1 = f x2 ∧ -4 * (Real.exp 1)^2 ≤ x1 * f x2 ∧ x1 * f x2 ≤ 0 :=
sorry

end range_of_x1_f_x2_l17_17944


namespace sad_children_count_l17_17590

theorem sad_children_count (total_children happy_children neither_happy_nor_sad children sad_children : ℕ)
  (h_total : total_children = 60)
  (h_happy : happy_children = 30)
  (h_neither : neither_happy_nor_sad = 20)
  (boys girls happy_boys sad_girls neither_boys : ℕ)
  (h_boys : boys = 17)
  (h_girls : girls = 43)
  (h_happy_boys : happy_boys = 6)
  (h_sad_girls : sad_girls = 4)
  (h_neither_boys : neither_boys = 5) :
  sad_children = total_children - happy_children - neither_happy_nor_sad :=
by sorry

end sad_children_count_l17_17590


namespace tap_B_fill_time_l17_17993

theorem tap_B_fill_time :
  ∃ t : ℝ, 
    (3 * 10 + (12 / t) * 10 = 36) →
    t = 20 :=
by
  sorry

end tap_B_fill_time_l17_17993


namespace average_age_of_both_teams_l17_17426

theorem average_age_of_both_teams (n_men : ℕ) (age_men : ℕ) (n_women : ℕ) (age_women : ℕ) :
  n_men = 8 → age_men = 35 → n_women = 6 → age_women = 30 → 
  (8 * 35 + 6 * 30) / (8 + 6) = 32.857 := 
by
  intros h1 h2 h3 h4
  -- Proof is omitted
  sorry

end average_age_of_both_teams_l17_17426


namespace total_ladybugs_and_ants_l17_17624

def num_leaves : ℕ := 84
def ladybugs_per_leaf : ℕ := 139
def ants_per_leaf : ℕ := 97

def total_ladybugs := ladybugs_per_leaf * num_leaves
def total_ants := ants_per_leaf * num_leaves
def total_insects := total_ladybugs + total_ants

theorem total_ladybugs_and_ants : total_insects = 19824 := by
  sorry

end total_ladybugs_and_ants_l17_17624


namespace cistern_wet_surface_area_l17_17008

def cistern (length : ℕ) (width : ℕ) (water_height : ℝ) : ℝ :=
  (length * width : ℝ) + 2 * (water_height * length) + 2 * (water_height * width)

theorem cistern_wet_surface_area :
  cistern 7 5 1.40 = 68.6 :=
by
  sorry

end cistern_wet_surface_area_l17_17008


namespace gcd_104_156_l17_17629

theorem gcd_104_156 : Nat.gcd 104 156 = 52 :=
by
  -- the proof steps will go here, but we can use sorry to skip it
  sorry

end gcd_104_156_l17_17629


namespace cost_of_soda_l17_17775

-- Define the system of equations
theorem cost_of_soda (b s f : ℕ): 
  3 * b + s = 390 ∧ 
  2 * b + 3 * s = 440 ∧ 
  b + 2 * f = 230 ∧ 
  s + 3 * f = 270 → 
  s = 234 := 
by 
  sorry

end cost_of_soda_l17_17775


namespace find_ab_l17_17709
-- Import the necessary Lean libraries 

-- Define the statement for the proof problem
theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : ab = 9 :=
by {
    sorry
}

end find_ab_l17_17709


namespace least_possible_number_l17_17853

theorem least_possible_number :
  ∃ x : ℕ, (∃ q r : ℕ, x = 34 * q + r ∧ 0 ≤ r ∧ r < 34) ∧
            (∃ q' : ℕ, x = 5 * q' ∧ q' = r + 8) ∧
            x = 75 :=
by
  sorry

end least_possible_number_l17_17853


namespace max_additional_bags_correct_l17_17774

-- Definitions from conditions
def num_people : ℕ := 6
def bags_per_person : ℕ := 5
def weight_per_bag : ℕ := 50
def max_plane_capacity : ℕ := 6000

-- Derived definitions from conditions
def total_bags : ℕ := num_people * bags_per_person
def total_weight_of_bags : ℕ := total_bags * weight_per_bag
def remaining_capacity : ℕ := max_plane_capacity - total_weight_of_bags 
def max_additional_bags : ℕ := remaining_capacity / weight_per_bag

-- Theorem statement
theorem max_additional_bags_correct : max_additional_bags = 90 := by
  -- Proof skipped
  sorry

end max_additional_bags_correct_l17_17774


namespace floor_of_sqrt_80_l17_17913

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l17_17913


namespace inequality_holds_for_all_real_numbers_l17_17246

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l17_17246


namespace frank_fence_length_l17_17680

theorem frank_fence_length (L W total_fence : ℝ) 
  (hW : W = 40) 
  (hArea : L * W = 200) 
  (htotal_fence : total_fence = 2 * L + W) : 
  total_fence = 50 := 
by 
  sorry

end frank_fence_length_l17_17680


namespace perpendicular_lines_parallel_lines_l17_17406

-- Define the given lines
def l1 (m : ℝ) (x y : ℝ) : ℝ := (m-2)*x + 3*y + 2*m
def l2 (m x y : ℝ) : ℝ := x + m*y + 6

-- The slope conditions for the lines to be perpendicular
def slopes_perpendicular (m : ℝ) : Prop :=
  (m - 2) * m = 3

-- The slope conditions for the lines to be parallel
def slopes_parallel (m : ℝ) : Prop :=
  m = -1

-- Perpendicular lines proof statement
theorem perpendicular_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_perpendicular m :=
sorry

-- Parallel lines proof statement
theorem parallel_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_parallel m :=
sorry

end perpendicular_lines_parallel_lines_l17_17406


namespace community_members_l17_17642

theorem community_members (n k : ℕ) (hk : k = 2) (hn : n = 6) 
    (H : ∀ (C1 C2 : Fin n) (hC1C2 : C1 ≠ C2), 
          ∃! (x : Fin ((n * (n - 1)) / 2)), 
          (x:Fin ((n * (n - 1)) / 2)) = (C1.val * n + C2.val - C1.val * (C1.val + 1) / 2)) :
    (∃ (x : ℕ), x = (n * (n - 1)) / (2 * k)) :=
by sorry

end community_members_l17_17642


namespace complete_square_identity_l17_17881

theorem complete_square_identity (x : ℝ) : ∃ (d e : ℤ), (x^2 - 10 * x + 13 = 0 → (x + d)^2 = e ∧ d + e = 7) :=
sorry

end complete_square_identity_l17_17881


namespace intersection_of_curves_l17_17532

theorem intersection_of_curves (x : ℝ) (y : ℝ) (h₁ : y = 9 / (x^2 + 3)) (h₂ : x + y = 3) : x = 0 :=
sorry

end intersection_of_curves_l17_17532


namespace initial_men_count_l17_17124

theorem initial_men_count (M : ℕ) (h1 : ∃ F : ℕ, F = M * 22) (h2 : ∃ F_remaining : ℕ, F_remaining = M * 20) (h3 : ∃ F_remaining_2 : ℕ, F_remaining_2 = (M + 1140) * 8) : 
  M = 760 := 
by
  -- Code to prove the theorem goes here.
  sorry

end initial_men_count_l17_17124


namespace quadratic_radical_simplified_l17_17222

theorem quadratic_radical_simplified (a : ℕ) : 
  (∃ (b : ℕ), a = 3 * b^2) -> a = 3 := 
by
  sorry

end quadratic_radical_simplified_l17_17222


namespace Christopher_joggers_eq_80_l17_17363

variable (T A C : ℕ)

axiom Tyson_joggers : T > 0                  -- Tyson bought a positive number of joggers.

axiom Alexander_condition : A = T + 22        -- Alexander bought 22 more joggers than Tyson.

axiom Christopher_condition : C = 20 * T      -- Christopher bought twenty times as many joggers as Tyson.

axiom Christopher_Alexander : C = A + 54     -- Christopher bought 54 more joggers than Alexander.

theorem Christopher_joggers_eq_80 : C = 80 := 
by
  sorry

end Christopher_joggers_eq_80_l17_17363


namespace zero_in_interval_l17_17765

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := -2 * x + 3
noncomputable def h (x : ℝ) : ℝ := f x + 2 * x - 3

theorem zero_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ h x = 0 := 
sorry

end zero_in_interval_l17_17765


namespace first_expression_second_expression_l17_17375

-- Define the variables
variables {a x y : ℝ}

-- Statement for the first expression
theorem first_expression (a : ℝ) : (2 * a^2)^3 + (-3 * a^3)^2 = 17 * a^6 := sorry

-- Statement for the second expression
theorem second_expression (x y : ℝ) : (x + 3 * y) * (x - y) = x^2 + 2 * x * y - 3 * y^2 := sorry

end first_expression_second_expression_l17_17375


namespace scientific_notation_470000000_l17_17748

theorem scientific_notation_470000000 : 470000000 = 4.7 * 10^8 :=
by
  sorry

end scientific_notation_470000000_l17_17748


namespace inequality_proof_l17_17288

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l17_17288


namespace floor_neg_seven_fourths_l17_17888

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l17_17888


namespace pictures_per_album_l17_17491

-- Define the conditions
def uploaded_pics_phone : ℕ := 22
def uploaded_pics_camera : ℕ := 2
def num_albums : ℕ := 4

-- Define the total pictures uploaded
def total_pictures : ℕ := uploaded_pics_phone + uploaded_pics_camera

-- Define the target statement as the theorem
theorem pictures_per_album : (total_pictures / num_albums) = 6 := by
  sorry

end pictures_per_album_l17_17491


namespace find_q_l17_17611

-- Given conditions
noncomputable def digits_non_zero (p q r : Nat) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0

noncomputable def three_digit_number (p q r : Nat) : Nat :=
  100 * p + 10 * q + r

noncomputable def two_digit_number (q r : Nat) : Nat :=
  10 * q + r

noncomputable def one_digit_number (r : Nat) : Nat := r

noncomputable def numbers_sum_to (p q r sum : Nat) : Prop :=
  three_digit_number p q r + two_digit_number q r + one_digit_number r = sum

-- The theorem to prove
theorem find_q (p q r : Nat) (hpq : digits_non_zero p q r)
  (hsum : numbers_sum_to p q r 912) : q = 5 := sorry

end find_q_l17_17611


namespace regular_octagon_angle_measure_l17_17835

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l17_17835


namespace count_integers_between_cubes_l17_17207

noncomputable def a := (10.1)^3
noncomputable def b := (10.4)^3

theorem count_integers_between_cubes : 
  ∃ (count : ℕ), count = 94 ∧ (1030.031 < a) ∧ (a < b) ∧ (b < 1124.864) := 
  sorry

end count_integers_between_cubes_l17_17207


namespace maximum_x_y_value_l17_17548

theorem maximum_x_y_value (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h1 : x + 2 * y ≤ 6) (h2 : 2 * x + y ≤ 6) : x + y ≤ 4 := 
sorry

end maximum_x_y_value_l17_17548


namespace train_pass_bridge_in_56_seconds_l17_17487

noncomputable def time_for_train_to_pass_bridge 
(length_of_train : ℕ) (speed_of_train_kmh : ℕ) (length_of_bridge : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  total_distance / speed_of_train_ms

theorem train_pass_bridge_in_56_seconds :
  time_for_train_to_pass_bridge 560 45 140 = 56 := by
  sorry

end train_pass_bridge_in_56_seconds_l17_17487


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17844

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17844


namespace partial_fraction_decomposition_l17_17317

noncomputable def partial_fraction_product (A B C : ℤ) : ℤ :=
  A * B * C

theorem partial_fraction_decomposition:
  ∃ A B C : ℤ, 
  (∀ x : ℤ, (x^2 - 19 = A * (x + 2) * (x - 3) 
                    + B * (x - 1) * (x - 3) 
                    + C * (x - 1) * (x + 2) )) 
  → partial_fraction_product A B C = 3 :=
by
  sorry

end partial_fraction_decomposition_l17_17317


namespace ratio_of_number_halving_l17_17321

theorem ratio_of_number_halving (x y : ℕ) (h1 : y = x / 2) (h2 : y = 9) : x / y = 2 :=
by
  sorry

end ratio_of_number_halving_l17_17321


namespace percentage_difference_between_maximum_and_minimum_changes_is_40_l17_17163

-- Definitions of initial and final survey conditions
def initialYesPercentage : ℝ := 0.40
def initialNoPercentage : ℝ := 0.60
def finalYesPercentage : ℝ := 0.80
def finalNoPercentage : ℝ := 0.20
def absenteePercentage : ℝ := 0.10

-- Main theorem stating the problem
theorem percentage_difference_between_maximum_and_minimum_changes_is_40 :
  let attendeesPercentage := 1 - absenteePercentage
  let adjustedFinalYesPercentage := finalYesPercentage / attendeesPercentage
  let minChange := adjustedFinalYesPercentage - initialYesPercentage
  let maxChange := initialYesPercentage + minChange
  maxChange - minChange = 0.40 :=
by
  -- Proof is omitted
  sorry

end percentage_difference_between_maximum_and_minimum_changes_is_40_l17_17163


namespace inequality_inequality_l17_17264

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l17_17264


namespace coefficient_square_sum_l17_17959

theorem coefficient_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by
  sorry

end coefficient_square_sum_l17_17959


namespace floor_neg_7_over_4_l17_17892

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l17_17892


namespace interval_length_of_solutions_l17_17523

theorem interval_length_of_solutions (a b : ℝ) :
  (∃ x : ℝ, a ≤ 3*x + 6 ∧ 3*x + 6 ≤ b) ∧ (∃ (l : ℝ), l = (b - a) / 3 ∧ l = 15) → b - a = 45 :=
by sorry

end interval_length_of_solutions_l17_17523


namespace man_savings_l17_17496

theorem man_savings (I : ℝ) (S : ℝ) (h1 : S = 0.35) (h2 : 2 * (0.65 * I) = 0.65 * I + 0.70 * I) :
  S = 0.35 :=
by
  -- Introduce necessary assumptions
  let savings_first_year := S * I
  let expenditure_first_year := I - savings_first_year
  let savings_second_year := 2 * savings_first_year

  have h3 : expenditure_first_year = 0.65 * I := by sorry
  have h4 : savings_first_year = 0.35 * I := by sorry

  -- Using given condition to resolve S
  exact h1

end man_savings_l17_17496


namespace mod_product_2023_2024_2025_2026_l17_17385

theorem mod_product_2023_2024_2025_2026 :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 :=
by
  have h2023 : 2023 % 7 = 6 := by norm_num
  have h2024 : 2024 % 7 = 0 := by norm_num
  have h2025 : 2025 % 7 = 1 := by norm_num
  have h2026 : 2026 % 7 = 2 := by norm_num
  calc
    (2023 * 2024 * 2025 * 2026) % 7
      = ((2023 % 7) * (2024 % 7) * (2025 % 7) * (2026 % 7)) % 7 : by rw [Nat.mul_mod, Nat.mul_mod, Nat.mul_mod, Nat.mul_mod]
  ... = (6 * 0 * 1 * 2) % 7 : by rw [h2023, h2024, h2025, h2026]
  ... = 0 % 7 : by norm_num
  ... = 0 : by norm_num

end mod_product_2023_2024_2025_2026_l17_17385


namespace greatest_int_with_gcd_18_is_138_l17_17478

theorem greatest_int_with_gcd_18_is_138 :
  ∃ n : ℕ, n < 150 ∧ int.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ int.gcd m 18 = 6 → m ≤ n := by
  sorry

end greatest_int_with_gcd_18_is_138_l17_17478


namespace equilibrium_price_quantity_quantity_increase_due_to_subsidy_l17_17617

theorem equilibrium_price_quantity (p Q : ℝ) :
  (∀ p, Q^S(p) = 2 + 8 * p) →
  (Q^D(2) = 8 ∧ Q^D(3) = 6) →
  (∀ p, Q^D(p) = -2 * p + 12) →
  ∃ p Q, Q^D(p) = Q^S(p) ∧ Q = 10 :=
by
  intros
  have h₁ : Q^D(p) = -2 * p + 12 := sorry
  have h₂ : Q^S(p) = 2 + 8 * p := sorry
  use 1
  use 10
  simp [Q^D, Q^S]
  split
  sorry -- detailed steps to show Q^D(1) = 10 and Q^S(1) = 10

theorem quantity_increase_due_to_subsidy (p Q : ℝ) (s : ℝ) :
  s = 1 →
  (∀ p, Q^S(p) = 2 + 8 * p) →
  (∀ p, Q^D(p) = -2 * p + 12) →
  ∃ ΔQ, ΔQ = 1.6 :=
by
  intros
  have Q_s : Q^S(p + s) = 2 + 8 * (p + 1) := sorry
  have Q_d : Q^D(p) = -2 * p + 12 := sorry
  have new_p : p = 0.2 := sorry
  have new_Q : Q^S(0.2) = 11.6 := sorry
  use 1.6
  simp
  sorry -- detailed steps to show ΔQ = 1.6.

end equilibrium_price_quantity_quantity_increase_due_to_subsidy_l17_17617


namespace regular_octagon_interior_angle_measure_l17_17780

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l17_17780


namespace greatest_integer_with_gcd_6_l17_17476

theorem greatest_integer_with_gcd_6 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 18 = 6) : n = 138 :=
sorry

end greatest_integer_with_gcd_6_l17_17476


namespace ribbon_per_box_l17_17094

def total_ribbon : ℝ := 4.5
def remaining_ribbon : ℝ := 1
def number_of_boxes : ℕ := 5

theorem ribbon_per_box :
  (total_ribbon - remaining_ribbon) / number_of_boxes = 0.7 :=
by
  sorry

end ribbon_per_box_l17_17094


namespace solve_positive_integer_l17_17584

theorem solve_positive_integer (n : ℕ) (h : ∀ m : ℕ, m > 0 → n^m ≥ m^n) : n = 3 :=
sorry

end solve_positive_integer_l17_17584


namespace remainder_of_2x_plus_3uy_l17_17410

theorem remainder_of_2x_plus_3uy (x y u v : ℤ) (hxy : x = u * y + v) (hv : 0 ≤ v) (hv_ub : v < y) :
  (if 2 * v < y then (2 * v % y) else ((2 * v % y) % -y % y)) = 
  (if 2 * v < y then 2 * v else 2 * v - y) :=
by {
  sorry
}

end remainder_of_2x_plus_3uy_l17_17410


namespace inequality_proof_l17_17254

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l17_17254


namespace find_rectangle_length_l17_17964

-- Define the problem conditions
def length_is_three_times_breadth (l b : ℕ) : Prop := l = 3 * b
def area_of_rectangle (l b : ℕ) : Prop := l * b = 6075

-- Define the theorem to prove the length of the rectangle given the conditions
theorem find_rectangle_length (l b : ℕ) (h1 : length_is_three_times_breadth l b) (h2 : area_of_rectangle l b) : l = 135 := 
sorry

end find_rectangle_length_l17_17964


namespace max_books_l17_17424

theorem max_books (price_per_book available_money : ℕ) (h1 : price_per_book = 15) (h2 : available_money = 200) :
  ∃ n : ℕ, n = 13 ∧ n ≤ available_money / price_per_book :=
by {
  sorry
}

end max_books_l17_17424


namespace Anna_needs_308_tulips_l17_17366

-- Define conditions as assertions or definitions
def number_of_eyes := 2
def red_tulips_per_eye := 8 
def number_of_eyebrows := 2
def purple_tulips_per_eyebrow := 5
def red_tulips_for_nose := 12
def red_tulips_for_smile := 18
def yellow_tulips_background := 9 * red_tulips_for_smile
def additional_purple_tulips_eyebrows := 4 * number_of_eyes * red_tulips_per_eye - number_of_eyebrows * purple_tulips_per_eyebrow
def yellow_tulips_for_nose := 3 * red_tulips_for_nose

-- Define total number of tulips for each color
def total_red_tulips := number_of_eyes * red_tulips_per_eye + red_tulips_for_nose + red_tulips_for_smile
def total_purple_tulips := number_of_eyebrows * purple_tulips_per_eyebrow + additional_purple_tulips_eyebrows
def total_yellow_tulips := yellow_tulips_background + yellow_tulips_for_nose

-- Define the total number of tulips
def total_tulips := total_red_tulips + total_purple_tulips + total_yellow_tulips

theorem Anna_needs_308_tulips :
  total_tulips = 308 :=
sorry

end Anna_needs_308_tulips_l17_17366


namespace cylindrical_to_rectangular_l17_17670

theorem cylindrical_to_rectangular (r θ z : ℝ) 
  (h₁ : r = 7) (h₂ : θ = 5 * Real.pi / 4) (h₃ : z = 6) : 
  (r * Real.cos θ, r * Real.sin θ, z) = 
  (-7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2, 6) := 
by 
  sorry

end cylindrical_to_rectangular_l17_17670


namespace sqrt_x_minus_2_range_l17_17428

theorem sqrt_x_minus_2_range (x : ℝ) : x - 2 ≥ 0 → x ≥ 2 :=
by sorry

end sqrt_x_minus_2_range_l17_17428


namespace units_digit_base_9_l17_17679

theorem units_digit_base_9 (a b : ℕ) (h1 : a = 3 * 9 + 5) (h2 : b = 4 * 9 + 7) : 
  ((a + b) % 9) = 3 := by
  sorry

end units_digit_base_9_l17_17679


namespace no_base_satisfies_l17_17467

def e : ℕ := 35

theorem no_base_satisfies :
  ∀ (base : ℝ), (1 / 5)^e * (1 / 4)^18 ≠ 1 / 2 * (base)^35 :=
by
  sorry

end no_base_satisfies_l17_17467


namespace pyramid_volume_l17_17399

noncomputable def volume_of_pyramid (a b c d: ℝ) (diagonal: ℝ) (angle: ℝ) : ℝ :=
  if (a = 10 ∧ d = 10 ∧ b = 5 ∧ c = 5 ∧ diagonal = 4 * Real.sqrt 5 ∧ angle = 45) then
    let base_area := 1 / 2 * (diagonal) * (Real.sqrt ((c * c) + (b * b)))
    let height := 10 / 3
    let volume := 1 / 3 * base_area * height
    volume
  else 0

theorem pyramid_volume :
  volume_of_pyramid 10 5 5 10 (4 * Real.sqrt 5) 45 = 500 / 9 :=
by
  sorry

end pyramid_volume_l17_17399


namespace abs_diff_inequality_l17_17963

theorem abs_diff_inequality (m : ℝ) : (∃ x : ℝ, |x + 2| - |x + 3| > m) ↔ m < -1 :=
sorry

end abs_diff_inequality_l17_17963


namespace line_intersects_ellipse_two_points_l17_17080

theorem line_intersects_ellipse_two_points (k b : ℝ) : 
  (-2 < b) ∧ (b < 2) ↔ ∀ x y : ℝ, (y = k * x + b) ↔ (x ^ 2 / 9 + y ^ 2 / 4 = 1) → true :=
sorry

end line_intersects_ellipse_two_points_l17_17080


namespace inequality_holds_for_all_reals_l17_17296

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l17_17296


namespace leadership_board_stabilizes_l17_17111

theorem leadership_board_stabilizes :
  ∃ n : ℕ, 2 ^ n - 1 ≤ 2020 ∧ 2020 < 2 ^ (n + 1) - 1 := by
  sorry

end leadership_board_stabilizes_l17_17111


namespace regular_octagon_interior_angle_l17_17801

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l17_17801


namespace prime_1011_n_l17_17400

theorem prime_1011_n (n : ℕ) (h : n ≥ 2) : 
  n = 2 ∨ n = 3 ∨ (∀ m : ℕ, m ∣ (n^3 + n + 1) → m = 1 ∨ m = n^3 + n + 1) :=
by sorry

end prime_1011_n_l17_17400


namespace sum_squares_of_roots_of_polynomial_l17_17519

noncomputable def roots (n : ℕ) (p : Polynomial ℂ) : List ℂ :=
  if h : n = p.natDegree then Multiset.toList p.roots else []

theorem sum_squares_of_roots_of_polynomial :
  (roots 2018 (Polynomial.C 404 + Polynomial.C 3 * X ^ 3 + Polynomial.C 44 * X ^ 2015 + X ^ 2018)).sum = 0 :=
by
  sorry

end sum_squares_of_roots_of_polynomial_l17_17519


namespace smallest_fraction_numerator_l17_17506

theorem smallest_fraction_numerator (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : (a * 4) > (b * 3)) : a = 73 :=
  sorry

end smallest_fraction_numerator_l17_17506


namespace price_increase_for_1620_profit_maximizing_profit_l17_17047

-- To state the problem, we need to define some variables and the associated conditions.

def cost_price : ℝ := 13
def initial_selling_price : ℝ := 20
def initial_monthly_sales : ℝ := 200
def decrease_in_sales_per_yuan : ℝ := 10
def profit_condition (x : ℝ) : ℝ := (initial_selling_price + x - cost_price) * (initial_monthly_sales - decrease_in_sales_per_yuan * x)
def profit_function (x : ℝ) : ℝ := -(10 * x ^ 2) + (130 * x) + 140

-- Part (1): Prove the price increase x such that the profit is 1620 yuan
theorem price_increase_for_1620_profit :
  ∃ (x : ℝ), profit_condition x = 1620 ∧ (x = 2 ∨ x = 11) :=
sorry

-- Part (2): Prove that the selling price that maximizes profit is 26.5 yuan and max profit is 1822.5 yuan
theorem maximizing_profit :
  ∃ (x : ℝ), (x = 13 / 2) ∧ profit_function (13 / 2) = 3645 / 2 :=
sorry

end price_increase_for_1620_profit_maximizing_profit_l17_17047


namespace postage_problem_l17_17928

noncomputable def sum_all_positive_integers (n1 n2 : ℕ) : ℕ :=
  n1 + n2

theorem postage_problem : sum_all_positive_integers 21 22 = 43 :=
by
  have h1 : ∀ x y z : ℕ, 7 * x + 21 * y + 23 * z ≠ 120 := sorry
  have h2 : ∀ x y z : ℕ, 7 * x + 22 * y + 24 * z ≠ 120 := sorry
  exact rfl

end postage_problem_l17_17928


namespace age_sum_l17_17338

variable (b : ℕ)
variable (a : ℕ := b + 2)
variable (c : ℕ := b / 2)

theorem age_sum : b = 10 → a + b + c = 27 :=
by
  intros h
  rw [h]
  sorry

end age_sum_l17_17338


namespace true_proposition_l17_17689

-- Definitions of propositions
def p := ∃ (x : ℝ), x - x + 1 ≥ 0
def q := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- Theorem statement
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end true_proposition_l17_17689


namespace necessary_and_sufficient_condition_l17_17756

theorem necessary_and_sufficient_condition {a : ℝ} :
    (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end necessary_and_sufficient_condition_l17_17756


namespace values_of_x_l17_17404

theorem values_of_x (x : ℕ) (h : Nat.choose 18 x = Nat.choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end values_of_x_l17_17404


namespace G_is_even_l17_17216

noncomputable def G (F : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := 
  F x * (1 / (a^x - 1) + 1 / 2)

theorem G_is_even (a : ℝ) (F : ℝ → ℝ) 
  (h₀ : a > 0) 
  (h₁ : a ≠ 1)
  (hF : ∀ x : ℝ, F (-x) = - F x) : 
  ∀ x : ℝ, G F a (-x) = G F a x :=
by 
  sorry

end G_is_even_l17_17216


namespace floor_sqrt_80_l17_17898

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l17_17898


namespace find_f4_l17_17941

-- Let f be a function from ℝ to ℝ with the following properties:
variable (f : ℝ → ℝ)

-- 1. f(x + 1) is an odd function
axiom f_odd : ∀ x, f (-(x + 1)) = -f (x + 1)

-- 2. f(x - 1) is an even function
axiom f_even : ∀ x, f (-(x - 1)) = f (x - 1)

-- 3. f(0) = 2
axiom f_zero : f 0 = 2

-- Prove that f(4) = -2
theorem find_f4 : f 4 = -2 :=
by
  sorry

end find_f4_l17_17941


namespace children_exceed_bridge_limit_l17_17156

theorem children_exceed_bridge_limit :
  ∀ (Kelly_weight : ℕ) (Megan_weight : ℕ) (Mike_weight : ℕ),
  Kelly_weight = 34 ∧
  Kelly_weight = (85 * Megan_weight) / 100 ∧
  Mike_weight = Megan_weight + 5 →
  Kelly_weight + Megan_weight + Mike_weight - 100 = 19 :=
by sorry

end children_exceed_bridge_limit_l17_17156


namespace sum_of_roots_is_k_over_5_l17_17436

noncomputable def sum_of_roots 
  (x1 x2 k d : ℝ) 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : ℝ :=
x1 + x2

theorem sum_of_roots_is_k_over_5 
  {x1 x2 k d : ℝ} 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : 
  sum_of_roots x1 x2 k d hx h1 h2 = k / 5 :=
sorry

end sum_of_roots_is_k_over_5_l17_17436


namespace inequality_holds_for_all_reals_l17_17289

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l17_17289


namespace expression_result_l17_17040

theorem expression_result :
  ( (9 + (1 / 2)) + (7 + (1 / 6)) + (5 + (1 / 12)) + (3 + (1 / 20)) + (1 + (1 / 30)) ) * 12 = 310 := by
  sorry

end expression_result_l17_17040


namespace integer_solution_unique_l17_17397

theorem integer_solution_unique
  (a b c d : ℤ)
  (h : a^2 + 5 * b^2 - 2 * c^2 - 2 * c * d - 3 * d^2 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end integer_solution_unique_l17_17397


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17845

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17845


namespace mother_older_than_twice_petra_l17_17466

def petra_age : ℕ := 11
def mother_age : ℕ := 36

def twice_petra_age : ℕ := 2 * petra_age

theorem mother_older_than_twice_petra : mother_age - twice_petra_age = 14 := by
  sorry

end mother_older_than_twice_petra_l17_17466


namespace inequality_inequality_l17_17257

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l17_17257


namespace Nell_cards_difference_l17_17745

-- Definitions
def initial_baseball_cards : ℕ := 438
def initial_ace_cards : ℕ := 18
def given_ace_cards : ℕ := 55
def given_baseball_cards : ℕ := 178

-- Theorem statement
theorem Nell_cards_difference :
  given_baseball_cards - given_ace_cards = 123 := 
by
  sorry

end Nell_cards_difference_l17_17745


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17843

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17843


namespace Amy_work_hours_l17_17511

theorem Amy_work_hours (summer_weeks: ℕ) (summer_hours_per_week: ℕ) (summer_total_earnings: ℕ)
                       (school_weeks: ℕ) (school_total_earnings: ℕ) (hourly_wage: ℕ) 
                       (school_hours_per_week: ℕ):
    summer_weeks = 8 →
    summer_hours_per_week = 40 →
    summer_total_earnings = 3200 →
    school_weeks = 32 →
    school_total_earnings = 4800 →
    hourly_wage = summer_total_earnings / (summer_weeks * summer_hours_per_week) →
    school_hours_per_week = school_total_earnings / (hourly_wage * school_weeks) →
    school_hours_per_week = 15 :=
by
  intros
  sorry

end Amy_work_hours_l17_17511


namespace initial_puppies_count_l17_17355

-- Define the initial conditions
def initial_birds : Nat := 12
def initial_cats : Nat := 5
def initial_spiders : Nat := 15
def initial_total_animals : Nat := 25
def half_birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_lost : Nat := 7

-- Define the remaining animals
def remaining_birds : Nat := initial_birds - half_birds_sold
def remaining_cats : Nat := initial_cats
def remaining_spiders : Nat := initial_spiders - spiders_lost

-- Define the total number of remaining animals excluding puppies
def remaining_non_puppy_animals : Nat := remaining_birds + remaining_cats + remaining_spiders

-- Define the remaining puppies
def remaining_puppies : Nat := initial_total_animals - remaining_non_puppy_animals
def initial_puppies : Nat := remaining_puppies + puppies_adopted

-- State the theorem
theorem initial_puppies_count :
  ∀ puppies : Nat, initial_puppies = 9 :=
by
  sorry

end initial_puppies_count_l17_17355


namespace inequality_xyz_l17_17279

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l17_17279


namespace original_average_weight_l17_17469

theorem original_average_weight 
  (W : ℝ)  -- Define W as the original average weight
  (h1 : 0 < W)  -- Define conditions
  (w_new1 : ℝ := 110)
  (w_new2 : ℝ := 60)
  (num_initial_players : ℝ := 7)
  (num_total_players : ℝ := 9)
  (new_average_weight : ℝ := 92)
  (total_weight_initial := num_initial_players * W)
  (total_weight_additional := w_new1 + w_new2)
  (total_weight_total := new_average_weight * num_total_players) : 
  total_weight_initial + total_weight_additional = total_weight_total → W = 94 :=
by 
  sorry

end original_average_weight_l17_17469


namespace students_present_in_class_l17_17855

theorem students_present_in_class :
  ∀ (total_students absent_percentage : ℕ), 
    total_students = 50 → absent_percentage = 12 → 
    (88 * total_students / 100) = 44 :=
by
  intros total_students absent_percentage h1 h2
  sorry

end students_present_in_class_l17_17855


namespace floor_neg_seven_fourths_l17_17894

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l17_17894


namespace elevator_min_trips_l17_17010

theorem elevator_min_trips :
  let masses := [150, 60, 70, 71, 72, 100, 101, 102, 103] in
  let max_load := 200 in
  (min_trips masses max_load = 5) :=
begin
  -- Sorry is used to skip the proof.
  sorry
end

end elevator_min_trips_l17_17010


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17827

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17827


namespace inequality_proof_l17_17253

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l17_17253


namespace mod_product_2023_2024_2025_2026_l17_17384

theorem mod_product_2023_2024_2025_2026 :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 :=
by
  have h2023 : 2023 % 7 = 6 := by norm_num
  have h2024 : 2024 % 7 = 0 := by norm_num
  have h2025 : 2025 % 7 = 1 := by norm_num
  have h2026 : 2026 % 7 = 2 := by norm_num
  calc
    (2023 * 2024 * 2025 * 2026) % 7
      = ((2023 % 7) * (2024 % 7) * (2025 % 7) * (2026 % 7)) % 7 : by rw [Nat.mul_mod, Nat.mul_mod, Nat.mul_mod, Nat.mul_mod]
  ... = (6 * 0 * 1 * 2) % 7 : by rw [h2023, h2024, h2025, h2026]
  ... = 0 % 7 : by norm_num
  ... = 0 : by norm_num

end mod_product_2023_2024_2025_2026_l17_17384


namespace convert_444_quinary_to_octal_l17_17042

def quinary_to_decimal (n : ℕ) : ℕ :=
  let d2 := (n / 100) * 25
  let d1 := ((n % 100) / 10) * 5
  let d0 := (n % 10)
  d2 + d1 + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let r2 := (n / 64)
  let n2 := (n % 64)
  let r1 := (n2 / 8)
  let r0 := (n2 % 8)
  r2 * 100 + r1 * 10 + r0

theorem convert_444_quinary_to_octal :
  decimal_to_octal (quinary_to_decimal 444) = 174 := by
  sorry

end convert_444_quinary_to_octal_l17_17042


namespace winning_percentage_votes_l17_17715

theorem winning_percentage_votes (P : ℝ) (votes_total : ℝ) (majority_votes : ℝ) (winning_votes : ℝ) : 
  votes_total = 4500 → majority_votes = 900 → 
  winning_votes = (P / 100) * votes_total → 
  majority_votes = winning_votes - ((100 - P) / 100) * votes_total → P = 60 := 
by
  intros h_total h_majority h_winning_votes h_majority_eq
  sorry

end winning_percentage_votes_l17_17715


namespace infinitely_many_not_sum_of_three_fourth_powers_l17_17299

theorem infinitely_many_not_sum_of_three_fourth_powers : ∀ n : ℕ, n > 0 → n ≡ 5 [MOD 16] → ¬(∃ a b c : ℤ, n = a^4 + b^4 + c^4) :=
by sorry

end infinitely_many_not_sum_of_three_fourth_powers_l17_17299


namespace min_value_of_reciprocal_sum_l17_17435

open Real

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : x + y = 12) (h4 : x * y = 20) : (1 / x + 1 / y) = 3 / 5 :=
sorry

end min_value_of_reciprocal_sum_l17_17435


namespace original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l17_17612

variable (a_n : ℕ → ℝ) (n : ℕ+)

-- To prove the original proposition
theorem original_proposition : (a_n n + a_n (n + 1)) / 2 < a_n n → (∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the inverse proposition
theorem inverse_proposition : ((a_n n + a_n (n + 1)) / 2 ≥ a_n n → ¬ ∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the converse proposition
theorem converse_proposition : (∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 < a_n n := 
sorry

-- To prove the contrapositive proposition
theorem contrapositive_proposition : (¬ ∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 ≥ a_n n :=
sorry

end original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l17_17612


namespace luke_games_l17_17739

variables (F G : ℕ)

theorem luke_games (G_eq_2 : G = 2) (total_games : F + G - 2 = 2) : F = 2 :=
by
  rw [G_eq_2] at total_games
  simp at total_games
  exact total_games

-- sorry

end luke_games_l17_17739


namespace infinitely_many_positive_integers_l17_17007

open Nat

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n ≥ 1, (n+1)^3 * a (n+1) = 2 * n^2 * (2 * n + 1) * a n + 2 * (3 * n + 1)

theorem infinitely_many_positive_integers (a : ℕ → ℤ) (h : sequence a) : 
  ∃ᶠ n in at_top, a n > 0 :=
sorry

end infinitely_many_positive_integers_l17_17007


namespace lamp_probability_l17_17109

theorem lamp_probability :
  ∀ (red_lamps blue_lamps : ℕ), 
  red_lamps = 4 → blue_lamps = 2 →
  (∀ lamps_on : ℕ, lamps_on = 3 →
    (1 / (Nat.choose (red_lamps + blue_lamps) 2 * Nat.choose (red_lamps + blue_lamps) 3 / 
      (Nat.choose (5) 1 * Nat.choose (4) 2)) = 0.1)) :=
by
  intros red_lamps blue_lamps h_rl h_bl lamps_on h_lo
  apply eq_div_iff_mul_eq.mpr _
  norm_num
  sorry

end lamp_probability_l17_17109


namespace exists_n_gt_2_divisible_by_1991_l17_17988

theorem exists_n_gt_2_divisible_by_1991 :
  ∃ n > 2, 1991 ∣ (2 * 10^(n+1) - 9) :=
by
  existsi (1799 : Nat)
  have h1 : 1799 > 2 := by decide
  have h2 : 1991 ∣ (2 * 10^(1799+1) - 9) := sorry
  constructor
  · exact h1
  · exact h2

end exists_n_gt_2_divisible_by_1991_l17_17988


namespace find_j_l17_17115

theorem find_j (j k : ℝ) :
  (∃ a d : ℝ, a ≠ 0 ∧ d ≠ 0 ∧ (∀ i ∈ {0, 1, 2, 3}, a + i * d ≠ a + j * d) ∧
  (∀ x : ℝ, (x = a ∨ x = a + d ∨ x = a + 2 * d ∨ x = a + 3 * d) →
  x^4 + j*x^2 + k*x + 400 = 0)) → j = -40 :=
by
  sorry

end find_j_l17_17115


namespace parabola_focus_distance_l17_17582

theorem parabola_focus_distance
  (F P Q : ℝ × ℝ)
  (hF : F = (1 / 2, 0))
  (hP : ∃ y, P = (2 * y^2, y))
  (hQ : Q = (1 / 2, Q.2))
  (h_parallel : P.2 = Q.2)
  (h_distance : dist P Q = dist Q F) :
  dist P F = 2 :=
by
  sorry

end parabola_focus_distance_l17_17582


namespace minister_can_organize_traffic_l17_17083

-- Definition of cities and roads
structure City (α : Type) :=
(road : α → α → Prop)

-- Defining the Minister's goal
def organize_traffic {α : Type} (c : City α) (num_days : ℕ) : Prop :=
∀ x y : α, c.road x y → num_days ≤ 214

theorem minister_can_organize_traffic :
  ∃ (c : City ℕ) (num_days : ℕ), (num_days ≤ 214 ∧ organize_traffic c num_days) :=
by {
  sorry
}

end minister_can_organize_traffic_l17_17083


namespace fraction_equality_l17_17708

theorem fraction_equality (a b : ℚ) (h₁ : a = 1/2) (h₂ : b = 2/3) : 
    (6 * a + 18 * b) / (12 * a + 6 * b) = 3 / 2 := by
  sorry

end fraction_equality_l17_17708


namespace solve_quadratics_l17_17662

theorem solve_quadratics :
  ∃ x y : ℝ, (9 * x^2 - 36 * x - 81 = 0) ∧ (y^2 + 6 * y + 9 = 0) ∧ (x + y = -1 + Real.sqrt 13 ∨ x + y = -1 - Real.sqrt 13) := 
by 
  sorry

end solve_quadratics_l17_17662


namespace price_of_when_you_rescind_cd_l17_17955

variable (W : ℕ) -- Defining W as a natural number since prices can't be negative

theorem price_of_when_you_rescind_cd
  (price_life_journey : ℕ := 100)
  (price_day_life : ℕ := 50)
  (num_cds_each : ℕ := 3)
  (total_spent : ℕ := 705) :
  3 * price_life_journey + 3 * price_day_life + 3 * W = total_spent → 
  W = 85 :=
by
  intros h
  sorry

end price_of_when_you_rescind_cd_l17_17955


namespace perfect_square_divisors_count_l17_17209

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def product_of_factorials : Nat := factorial 1 * factorial 2 * factorial 3 * factorial 4 * factorial 5 *
                                   factorial 6 * factorial 7 * factorial 8 * factorial 9 * factorial 10

def count_perfect_square_divisors (n : Nat) : Nat := sorry -- This would involve the correct function implementation.

theorem perfect_square_divisors_count :
  count_perfect_square_divisors product_of_factorials = 2160 :=
sorry

end perfect_square_divisors_count_l17_17209


namespace inequality_holds_for_real_numbers_l17_17272

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l17_17272


namespace inequality_holds_for_all_reals_l17_17292

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l17_17292


namespace total_teachers_correct_l17_17024

noncomputable def total_teachers (x : ℕ) : ℕ := 26 + 104 + x

theorem total_teachers_correct
    (x : ℕ)
    (h : (x : ℝ) / (26 + 104 + x) = 16 / 56) :
  total_teachers x = 182 :=
sorry

end total_teachers_correct_l17_17024


namespace proof_problem_l17_17552

variable {a1 a2 b1 b2 b3 : ℝ}

-- Condition: -2, a1, a2, -8 form an arithmetic sequence
def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = -2 / 3 * (-2 - 8)

-- Condition: -2, b1, b2, b3, -8 form a geometric sequence
def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  b2^2 = (-2) * (-8) ∧ b1^2 = (-2) * b2 ∧ b3^2 = b2 * (-8)

theorem proof_problem (h1 : arithmetic_sequence a1 a2) (h2 : geometric_sequence b1 b2 b3) : b2 * (a2 - a1) = 8 :=
by
  admit -- Convert to sorry to skip the proof

end proof_problem_l17_17552


namespace range_of_a_l17_17947

theorem range_of_a (a : ℝ) (h : a > 0) (h1 : ∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) : a ≥ 2 := 
sorry

end range_of_a_l17_17947


namespace fraction_arithmetic_l17_17878

theorem fraction_arithmetic : 
  (2 / 5 + 3 / 7) / (4 / 9 * 1 / 8) = 522 / 35 := by
  sorry

end fraction_arithmetic_l17_17878


namespace max_value_of_f_l17_17931

noncomputable def f (x : ℝ) : ℝ :=
  min (3 * x + 3) (min (-x / 3 + 3) (x / 3 + 9))

theorem max_value_of_f : ∃ x : ℝ, f x = 6 :=
by
  sorry

end max_value_of_f_l17_17931


namespace find_S40_l17_17175

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

theorem find_S40 (a r : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = geometric_sequence_sum a r n)
  (h2 : S 10 = 10)
  (h3 : S 30 = 70) :
  S 40 = 150 ∨ S 40 = 110 := 
sorry

end find_S40_l17_17175


namespace inequality_proof_l17_17255

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l17_17255


namespace original_sugar_amount_l17_17235

theorem original_sugar_amount (f : ℕ) (s t r : ℕ) (h1 : f = 5) (h2 : r = 10) (h3 : t = 14) (h4 : f * 2 = r):
  s = t / 2 := sorry

end original_sugar_amount_l17_17235


namespace radius_decrease_l17_17119

theorem radius_decrease (r r' : ℝ) (A A' : ℝ) (h_original_area : A = π * r^2)
  (h_area_decrease : A' = 0.25 * A) (h_new_area : A' = π * r'^2) : r' = 0.5 * r :=
by
  sorry

end radius_decrease_l17_17119


namespace large_cube_side_length_painted_blue_l17_17495

   theorem large_cube_side_length_painted_blue (n : ℕ) (h : 6 * n^2 = (1 / 3) * 6 * n^3) : n = 3 :=
   by
     sorry
   
end large_cube_side_length_painted_blue_l17_17495


namespace no_solution_prob1_l17_17990

theorem no_solution_prob1 : ¬ ∃ x : ℝ, x ≠ 2 ∧ (1 / (x - 2) + 3 = (1 - x) / (2 - x)) :=
by
  sorry

end no_solution_prob1_l17_17990


namespace white_roses_total_l17_17099

theorem white_roses_total (bq_num : ℕ) (tbl_num : ℕ) (roses_per_bq : ℕ) (roses_per_tbl : ℕ)
  (total_roses : ℕ) 
  (h1 : bq_num = 5) 
  (h2 : tbl_num = 7) 
  (h3 : roses_per_bq = 5) 
  (h4 : roses_per_tbl = 12)
  (h5 : total_roses = 109) : 
  bq_num * roses_per_bq + tbl_num * roses_per_tbl = total_roses := 
by 
  rw [h1, h2, h3, h4, h5]
  exact rfl

end white_roses_total_l17_17099


namespace circle_center_distance_travelled_l17_17349

theorem circle_center_distance_travelled :
  ∀ (r : ℝ) (a b c : ℝ), r = 2 ∧ a = 9 ∧ b = 12 ∧ c = 15 → (a^2 + b^2 = c^2) → 
  ∃ (d : ℝ), d = 24 :=
by
  intros r a b c h1 h2
  sorry

end circle_center_distance_travelled_l17_17349


namespace triangle_side_c_l17_17059

noncomputable def area_of_triangle (a b C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C

noncomputable def law_of_cosines (a b C : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)

theorem triangle_side_c (a b C : ℝ) (h1 : a = 3) (h2 : C = Real.pi * 2 / 3) (h3 : area_of_triangle a b C = 15 * Real.sqrt 3 / 4) : law_of_cosines a b C = 2 :=
by
  sorry

end triangle_side_c_l17_17059


namespace inequality_xyz_l17_17277

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l17_17277


namespace parametric_to_standard_equation_l17_17613

theorem parametric_to_standard_equation (x y t : ℝ) 
(h1 : x = 4 * t + 1) 
(h2 : y = -2 * t - 5) : 
x + 2 * y + 9 = 0 :=
by
  sorry

end parametric_to_standard_equation_l17_17613


namespace ab_zero_proof_l17_17358

-- Given conditions
def square_side : ℝ := 3
def rect_short_side : ℝ := 3
def rect_long_side : ℝ := 6
def rect_area : ℝ := rect_short_side * rect_long_side
def split_side_proof (a b : ℝ) : Prop := a + b = rect_short_side

-- Lean theorem proving that ab = 0 given the conditions
theorem ab_zero_proof (a b : ℝ) 
  (h1 : square_side = 3)
  (h2 : rect_short_side = 3)
  (h3 : rect_long_side = 6)
  (h4 : rect_area = 18)
  (h5 : split_side_proof a b) : a * b = 0 := by
  sorry

end ab_zero_proof_l17_17358


namespace quadratic_root_square_of_another_l17_17932

theorem quadratic_root_square_of_another (a : ℚ) :
  (∃ x y : ℚ, x^2 - (15/4) * x + a^3 = 0 ∧ (x = y^2 ∨ y = x^2) ∧ (x*y = a^3)) →
  (a = 3/2 ∨ a = -5/2) :=
sorry

end quadratic_root_square_of_another_l17_17932


namespace smallest_positive_value_l17_17675

noncomputable def exprA := 30 - 4 * Real.sqrt 14
noncomputable def exprB := 4 * Real.sqrt 14 - 30
noncomputable def exprC := 25 - 6 * Real.sqrt 15
noncomputable def exprD := 75 - 15 * Real.sqrt 30
noncomputable def exprE := 15 * Real.sqrt 30 - 75

theorem smallest_positive_value :
  exprC = 25 - 6 * Real.sqrt 15 ∧
  exprC < exprA ∧
  exprC < exprB ∧
  exprC < exprD ∧
  exprC < exprE ∧
  exprC > 0 :=
by sorry

end smallest_positive_value_l17_17675


namespace stamp_problem_l17_17926

/-- Define the context where we have stamps of 7, n, and (n + 2) cents, and 120 cents being the largest
    value that cannot be formed using these stamps -/
theorem stamp_problem (n : ℕ) (h : ∀ k, k > 120 → ∃ a b c, k = 7 * a + n * b + (n + 2) * c) (hn : ¬ ∃ a b c, 120 = 7 * a + n * b + (n + 2) * c) : n = 22 :=
sorry

end stamp_problem_l17_17926


namespace f_at_4_l17_17609

-- Define the conditions on the function f
variable (f : ℝ → ℝ)
variable (h_domain : true) -- All ℝ → ℝ functions have ℝ as their domain.

-- f is an odd function
axiom h_odd : ∀ x : ℝ, f (-x) = -f x

-- Given functional equation
axiom h_eqn : ∀ x : ℝ, f (2 * x - 3) - 2 * f (3 * x - 10) + f (x - 3) = 28 - 6 * x 

-- The goal is to determine the value of f(4), which should be 8.
theorem f_at_4 : f 4 = 8 :=
sorry

end f_at_4_l17_17609


namespace interior_angle_regular_octagon_l17_17816

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l17_17816


namespace number_of_tiles_per_row_in_square_room_l17_17304

theorem number_of_tiles_per_row_in_square_room 
  (area_of_floor: ℝ)
  (tile_side_length: ℝ)
  (conversion_rate: ℝ)
  (h1: area_of_floor = 400) 
  (h2: tile_side_length = 8) 
  (h3: conversion_rate = 12) 
: (sqrt area_of_floor * conversion_rate / tile_side_length) = 30 := 
by
  sorry

end number_of_tiles_per_row_in_square_room_l17_17304


namespace inequality_xyz_l17_17278

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l17_17278


namespace three_is_square_root_of_nine_l17_17006

theorem three_is_square_root_of_nine :
  ∃ x : ℝ, x * x = 9 ∧ x = 3 :=
sorry

end three_is_square_root_of_nine_l17_17006


namespace locus_of_centers_of_tangent_circles_l17_17994

noncomputable def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
noncomputable def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25
noncomputable def locus (a b : ℝ) : Prop := 4 * a^2 + 4 * b^2 - 6 * a - 25 = 0

theorem locus_of_centers_of_tangent_circles :
  (∃ (a b r : ℝ), a^2 + b^2 = (r + 1)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2) →
  (∃ a b : ℝ, locus a b) :=
sorry

end locus_of_centers_of_tangent_circles_l17_17994


namespace roots_of_quadratic_identity_l17_17711

namespace RootProperties

theorem roots_of_quadratic_identity (a b : ℝ) 
(h1 : a^2 - 2*a - 1 = 0) 
(h2 : b^2 - 2*b - 1 = 0) 
(h3 : a ≠ b) 
: a^2 + b^2 = 6 := 
by sorry

end RootProperties

end roots_of_quadratic_identity_l17_17711


namespace dishes_combinations_is_correct_l17_17661

-- Define the number of dishes
def num_dishes : ℕ := 15

-- Define the number of appetizers
def num_appetizers : ℕ := 5

-- Compute the total number of combinations
def combinations_of_dishes : ℕ :=
  num_dishes * num_dishes * num_appetizers

-- The theorem that states the total number of combinations is 1125
theorem dishes_combinations_is_correct :
  combinations_of_dishes = 1125 := by
  sorry

end dishes_combinations_is_correct_l17_17661


namespace equation_of_tangent_line_l17_17945

-- Definitions for the given conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x
def P : ℝ × ℝ := (-1, 4)
def slope_of_tangent (a : ℝ) (x : ℝ) : ℝ := -6 * x^2 - 2

-- The main theorem to prove the equation of the tangent line
theorem equation_of_tangent_line (a : ℝ) (ha : f a (-1) = 4) :
  8 * x + y + 4 = 0 := by
  sorry

end equation_of_tangent_line_l17_17945


namespace parabola_directrix_l17_17567

theorem parabola_directrix (p : ℝ) (h : p > 0) (h_directrix : -p / 2 = -4) : p = 8 :=
by
  sorry

end parabola_directrix_l17_17567


namespace rectangle_length_l17_17144

theorem rectangle_length (P B L : ℕ) (h1 : P = 800) (h2 : B = 300) (h3 : P = 2 * (L + B)) : L = 100 :=
by
  sorry

end rectangle_length_l17_17144


namespace pages_and_cost_calculation_l17_17575

noncomputable def copy_pages_cost (cents_per_5_pages : ℕ) (total_cents : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
if total_cents < discount_threshold * (cents_per_5_pages / 5) then
  total_cents / (cents_per_5_pages / 5)
else
  let num_pages_before_discount := discount_threshold
  let remaining_pages := total_cents / (cents_per_5_pages / 5) - num_pages_before_discount
  let cost_before_discount := num_pages_before_discount * (cents_per_5_pages / 5)
  let discounted_cost := remaining_pages * (cents_per_5_pages / 5) * (1 - discount_rate)
  cost_before_discount + discounted_cost

theorem pages_and_cost_calculation :
  let cents_per_5_pages := 10
  let total_cents := 5000
  let discount_threshold := 1000
  let discount_rate := 0.10
  let num_pages := (cents_per_5_pages * 2500) / 5
  let cost := copy_pages_cost cents_per_5_pages total_cents discount_threshold discount_rate
  (num_pages = 2500) ∧ (cost = 4700) :=
by
  sorry

end pages_and_cost_calculation_l17_17575


namespace min_value_frac_inv_l17_17939

theorem min_value_frac_inv {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (∃ m, (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 2 → m ≤ (1 / x + 1 / y)) ∧ (m = 2)) :=
by
  sorry

end min_value_frac_inv_l17_17939


namespace interior_angle_regular_octagon_l17_17817

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l17_17817


namespace triangle_angle_contradiction_l17_17331

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (sum_angles : A + B + C = 180) : 
  (¬ (A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60)) = (A > 60 ∧ B > 60 ∧ C > 60) :=
by sorry

end triangle_angle_contradiction_l17_17331


namespace water_tank_capacity_l17_17654

theorem water_tank_capacity :
  ∃ (x : ℝ), 0.9 * x - 0.4 * x = 30 → x = 60 :=
by
  sorry

end water_tank_capacity_l17_17654


namespace min_value_of_f_range_of_a_l17_17583

def f (x : ℝ) : ℝ := 2 * |x - 2| - x + 5

theorem min_value_of_f : ∃ (m : ℝ), m = 3 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use 3
  sorry

theorem range_of_a (a : ℝ) : (|a + 2| ≥ 3 ↔ a ≤ -5 ∨ a ≥ 1) :=
sorry

end min_value_of_f_range_of_a_l17_17583


namespace biggest_number_in_ratio_l17_17538

theorem biggest_number_in_ratio (x : ℕ) (h_sum : 2 * x + 3 * x + 4 * x + 5 * x = 1344) : 5 * x = 480 := 
by
  sorry

end biggest_number_in_ratio_l17_17538


namespace sum_of_min_x_y_l17_17586

theorem sum_of_min_x_y : ∃ (x y : ℕ), 
  (∃ a b c : ℕ, 180 = 2^a * 3^b * 5^c) ∧
  (∃ u v w : ℕ, 180 * x = 2^u * 3^v * 5^w ∧ u % 4 = 0 ∧ v % 4 = 0 ∧ w % 4 = 0) ∧
  (∃ p q r : ℕ, 180 * y = 2^p * 3^q * 5^r ∧ p % 6 = 0 ∧ q % 6 = 0 ∧ r % 6 = 0) ∧
  (x + y = 4054500) :=
sorry

end sum_of_min_x_y_l17_17586


namespace parallel_lines_iff_l17_17056

theorem parallel_lines_iff (a : ℝ) :
  (∀ x y : ℝ, x - y - 1 = 0 → x + a * y - 2 = 0) ↔ (a = -1) :=
by
  sorry

end parallel_lines_iff_l17_17056


namespace mass_percentage_Cl_correct_l17_17398

-- Define the given condition
def mass_percentage_of_Cl := 66.04

-- Statement to prove
theorem mass_percentage_Cl_correct : mass_percentage_of_Cl = 66.04 :=
by
  -- This is where the proof would go, but we use sorry as placeholder.
  sorry

end mass_percentage_Cl_correct_l17_17398


namespace imaginary_part_is_empty_l17_17459

def imaginary_part_empty (z : ℂ) : Prop :=
  z.im = 0

theorem imaginary_part_is_empty (z : ℂ) (h : z.im = 0) : imaginary_part_empty z :=
by
  -- proof skipped
  sorry

end imaginary_part_is_empty_l17_17459


namespace repair_cost_l17_17986

theorem repair_cost (purchase_price transport_cost sale_price : ℝ) (profit_percentage : ℝ) (repair_cost : ℝ) :
  purchase_price = 14000 →
  transport_cost = 1000 →
  sale_price = 30000 →
  profit_percentage = 50 →
  sale_price = (1 + profit_percentage / 100) * (purchase_price + repair_cost + transport_cost) →
  repair_cost = 5000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end repair_cost_l17_17986


namespace find_initial_balance_l17_17139

-- Define the initial balance (X)
def initial_balance (X : ℝ) := 
  ∃ (X : ℝ), (X / 2 + 30 + 50 - 20 = 160)

theorem find_initial_balance (X : ℝ) (h : initial_balance X) : 
  X = 200 :=
sorry

end find_initial_balance_l17_17139


namespace cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l17_17713

theorem cos_B_arithmetic_sequence (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180) :
  Real.cos B = 1 / 2 :=
by
  sorry

theorem sin_A_sin_C_geometric_sequence (A B C a b c : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180)
  (h3 : b^2 = a * c) (h4 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) :
  Real.sin A * Real.sin C = 3 / 4 :=
by
  sorry

end cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l17_17713


namespace total_weekly_earnings_l17_17097

-- Define the total weekly hours and earnings
def weekly_hours_weekday : ℕ := 5 * 5
def weekday_rate : ℕ := 3
def weekday_earnings : ℕ := weekly_hours_weekday * weekday_rate

-- Define the total weekend hours and earnings
def weekend_days : ℕ := 2
def weekend_hours_per_day : ℕ := 3
def weekend_rate : ℕ := 3 * 2
def weekend_hours : ℕ := weekend_days * weekend_hours_per_day
def weekend_earnings : ℕ := weekend_hours * weekend_rate

-- Prove that Mitch's total earnings per week are $111
theorem total_weekly_earnings : weekday_earnings + weekend_earnings = 111 := by
  sorry

end total_weekly_earnings_l17_17097


namespace calc_expression_l17_17564

noncomputable def x := (3 + Real.sqrt 5) / 2 -- chosen from one of the roots of the quadratic equation x^2 - 3x + 1

theorem calc_expression (h : x + 1 / x = 3) : 
  (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 7 + 3 * Real.sqrt 5 := 
by 
  sorry

end calc_expression_l17_17564


namespace class_A_scores_more_uniform_l17_17346

-- Define the variances of the test scores for classes A and B
def variance_A := 13.2
def variance_B := 26.26

-- Theorem: Prove that the scores of the 10 students from class A are more uniform than those from class B
theorem class_A_scores_more_uniform :
  variance_A < variance_B :=
  by
    -- Assume the given variances and state the comparison
    have h : 13.2 < 26.26 := by sorry
    exact h

end class_A_scores_more_uniform_l17_17346


namespace value_of_c_l17_17217

theorem value_of_c (b c : ℝ) (h1 : (x : ℝ) → (x + 4) * (x + b) = x^2 + c * x + 12) : c = 7 :=
by
  have h2 : 4 * b = 12 := by sorry
  have h3 : b = 3 := by sorry
  have h4 : c = b + 4 := by sorry
  rw [h3] at h4
  rw [h4]
  exact by norm_num

end value_of_c_l17_17217


namespace quadratic_eq_roots_quadratic_eq_positive_integer_roots_l17_17545

theorem quadratic_eq_roots (m : ℝ) (hm : m ≠ 0 ∧ m ≤ 9 / 8) :
  ∃ x1 x2 : ℝ, (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

theorem quadratic_eq_positive_integer_roots (m : ℕ) (hm : m = 1) :
  ∃ x1 x2 : ℝ, (x1 = -1) ∧ (x2 = -2) ∧ (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

end quadratic_eq_roots_quadratic_eq_positive_integer_roots_l17_17545


namespace sqrt_floor_eight_l17_17910

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l17_17910


namespace violet_balloons_remaining_l17_17430

def initial_count : ℕ := 7
def lost_count : ℕ := 3

theorem violet_balloons_remaining : initial_count - lost_count = 4 :=
by sorry

end violet_balloons_remaining_l17_17430


namespace product_complex_l17_17666

noncomputable def P (x : ℂ) : ℂ := ∏ k in finset.range 15, (x - complex.exp (2 * real.pi * complex.I * k / 17))
noncomputable def Q (x : ℂ) : ℂ := ∏ j in finset.range 12, (x - complex.exp (2 * real.pi * complex.I * j / 13))

theorem product_complex : 
  ∏ k in finset.range 15, (∏ j in finset.range 12, (complex.exp (2 * real.pi * complex.I * j / 13) - complex.exp (2 * real.pi * complex.I * k / 17))) = 1 := 
sorry

end product_complex_l17_17666


namespace find_other_number_l17_17639

theorem find_other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 192) (h_hcf : Nat.gcd A B = 16) (h_A : A = 48) : B = 64 :=
by
  sorry

end find_other_number_l17_17639


namespace calculation_correct_l17_17658

theorem calculation_correct :
  (Int.ceil ((15 : ℚ) / 8 * ((-35 : ℚ) / 4)) - 
  Int.floor (((15 : ℚ) / 8) * Int.floor ((-35 : ℚ) / 4 + (1 : ℚ) / 4))) = 1 := by
  sorry

end calculation_correct_l17_17658


namespace scientific_notation_of_29_47_thousand_l17_17344

theorem scientific_notation_of_29_47_thousand :
  (29.47 * 1000 = 2.947 * 10^4) :=
sorry

end scientific_notation_of_29_47_thousand_l17_17344


namespace directrix_of_parabola_l17_17182

theorem directrix_of_parabola :
  ∀ (x : ℝ), (∃ k : ℝ, y = (x^2 - 8 * x + 16) / 8 → k = -2) :=
by
  sorry

end directrix_of_parabola_l17_17182


namespace mukesh_total_debt_l17_17103

-- Define the initial principal, additional loan, interest rate, and time periods
def principal₁ : ℝ := 10000
def principal₂ : ℝ := 12000
def rate : ℝ := 0.06
def time₁ : ℝ := 2
def time₂ : ℝ := 3

-- Define the interest calculations
def interest₁ : ℝ := principal₁ * rate * time₁
def total_after_2_years : ℝ := principal₁ + interest₁ + principal₂
def interest₂ : ℝ := total_after_2_years * rate * time₂

-- Define the total amount owed after 5 years
def amount_owed : ℝ := total_after_2_years + interest₂

-- The goal is to prove that Mukesh owes 27376 Rs after 5 years
theorem mukesh_total_debt : amount_owed = 27376 := by sorry

end mukesh_total_debt_l17_17103


namespace michael_needs_more_money_l17_17096

-- Define the initial conditions
def michael_money : ℝ := 50
def cake_cost : ℝ := 20
def bouquet_cost : ℝ := 36
def balloons_cost : ℝ := 5
def perfume_gbp : ℝ := 30
def gbp_to_usd : ℝ := 1.4
def perfume_cost : ℝ := perfume_gbp * gbp_to_usd
def photo_album_eur : ℝ := 25
def eur_to_usd : ℝ := 1.2
def photo_album_cost : ℝ := photo_album_eur * eur_to_usd

-- Sum the costs
def total_cost : ℝ := cake_cost + bouquet_cost + balloons_cost + perfume_cost + photo_album_cost

-- Define the required amount
def additional_money_needed : ℝ := total_cost - michael_money

-- The theorem statement
theorem michael_needs_more_money : additional_money_needed = 83 := by
  sorry

end michael_needs_more_money_l17_17096


namespace predicted_value_y_at_x_5_l17_17987

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem predicted_value_y_at_x_5 :
  let x_values := [-2, -1, 0, 1, 2]
  let y_values := [5, 4, 2, 2, 1]
  let x_bar := mean x_values
  let y_bar := mean y_values
  let a_hat := y_bar
  (∀ x, y = -x + a_hat) →
  (x = 5 → y = -2.2) :=
by
  sorry

end predicted_value_y_at_x_5_l17_17987


namespace regular_octagon_interior_angle_l17_17798

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l17_17798


namespace vector_decomposition_unique_l17_17414

variable {m : ℝ}
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (m - 1, m + 3)

theorem vector_decomposition_unique (m : ℝ) : (m + 3 ≠ 2 * (m - 1)) ↔ (m ≠ 5) := 
sorry

end vector_decomposition_unique_l17_17414


namespace base_rate_first_company_proof_l17_17873

noncomputable def base_rate_first_company : ℝ := 8.00
def charge_per_minute_first_company : ℝ := 0.25
def base_rate_second_company : ℝ := 12.00
def charge_per_minute_second_company : ℝ := 0.20
def minutes : ℕ := 80

theorem base_rate_first_company_proof :
  base_rate_first_company = 8.00 :=
sorry

end base_rate_first_company_proof_l17_17873


namespace total_shirts_correct_l17_17625

def machine_A_production_rate := 6
def machine_A_yesterday_minutes := 12
def machine_A_today_minutes := 10

def machine_B_production_rate := 8
def machine_B_yesterday_minutes := 10
def machine_B_today_minutes := 15

def machine_C_production_rate := 5
def machine_C_yesterday_minutes := 20
def machine_C_today_minutes := 0

def total_shirts_produced : Nat :=
  (machine_A_production_rate * machine_A_yesterday_minutes +
  machine_A_production_rate * machine_A_today_minutes) +
  (machine_B_production_rate * machine_B_yesterday_minutes +
  machine_B_production_rate * machine_B_today_minutes) +
  (machine_C_production_rate * machine_C_yesterday_minutes +
  machine_C_production_rate * machine_C_today_minutes)

theorem total_shirts_correct : total_shirts_produced = 432 :=
by 
  sorry 

end total_shirts_correct_l17_17625


namespace regular_octagon_interior_angle_l17_17811

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l17_17811


namespace parabola_properties_l17_17529

-- Define the conditions
def vertex (f : ℝ → ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ (x : ℝ), f (v.1) ≤ f x

def vertical_axis_of_symmetry (f : ℝ → ℝ) (h : ℝ) : Prop :=
  ∀ (x : ℝ), f x = f (2 * h - x)

def contains_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

-- Define f as the given parabola equation
def f (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 16

-- The main statement to prove
theorem parabola_properties :
  vertex f (3, -2) ∧ vertical_axis_of_symmetry f 3 ∧ contains_point f (6, 16) := sorry

end parabola_properties_l17_17529


namespace ratio_of_boxes_loaded_l17_17368

variable (D N B : ℕ) 

-- Definitions as conditions
def night_crew_workers (D : ℕ) : ℕ := (4 * D) / 9
def day_crew_boxes (B : ℕ) : ℕ := (3 * B) / 4
def night_crew_boxes (B : ℕ) : ℕ := B / 4

theorem ratio_of_boxes_loaded :
  ∀ {D B : ℕ}, 
    night_crew_workers D ≠ 0 → 
    D ≠ 0 → 
    B ≠ 0 → 
    ((night_crew_boxes B) / (night_crew_workers D)) / ((day_crew_boxes B) / D) = 3 / 4 :=
by
  -- Proof
  sorry

end ratio_of_boxes_loaded_l17_17368


namespace determine_a_perpendicular_l17_17458

theorem determine_a_perpendicular 
  (a : ℝ)
  (h1 : 2 * x + 3 * y + 5 = 0)
  (h2 : a * x + 3 * y - 4 = 0) 
  (h_perpendicular : ∀ x y, (2 * x + 3 * y + 5 = 0) → ∀ x y, (a * x + 3 * y - 4 = 0) → (-(2 : ℝ) / (3 : ℝ)) * (-(a : ℝ) / (3 : ℝ)) = -1) :
  a = -9 / 2 :=
by
  sorry

end determine_a_perpendicular_l17_17458


namespace smallest_sum_B_c_l17_17562

theorem smallest_sum_B_c 
  (B c : ℕ) 
  (h1 : B ≤ 4) 
  (h2 : 6 < c) 
  (h3 : 31 * B = 4 * (c + 1)) : 
  B + c = 34 := 
sorry

end smallest_sum_B_c_l17_17562


namespace interior_angle_of_regular_octagon_l17_17785

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l17_17785


namespace possible_slopes_of_line_intersects_ellipse_l17_17865

/-- 
A line whose y-intercept is (0, 3) intersects the ellipse 4x^2 + 9y^2 = 36. 
Find all possible slopes of this line. 
-/
theorem possible_slopes_of_line_intersects_ellipse :
  (∀ m : ℝ, ∃ x : ℝ, 4 * x^2 + 9 * (m * x + 3)^2 = 36) ↔ 
  (m <= - (Real.sqrt 5) / 3 ∨ m >= (Real.sqrt 5) / 3) :=
sorry

end possible_slopes_of_line_intersects_ellipse_l17_17865


namespace sum_of_factors_eq_l17_17113

theorem sum_of_factors_eq :
  ∃ (d e f : ℤ), (∀ (x : ℤ), x^2 + 21 * x + 110 = (x + d) * (x + e)) ∧
                 (∀ (x : ℤ), x^2 - 19 * x + 88 = (x - e) * (x - f)) ∧
                 (d + e + f = 30) :=
sorry

end sum_of_factors_eq_l17_17113


namespace find_k_plus_a_l17_17023

theorem find_k_plus_a (k a : ℤ) (h1 : k > a) (h2 : a > 0) 
(h3 : 2 * (Int.natAbs (a - k)) * (Int.natAbs (a + k)) = 32) : k + a = 8 :=
by
  sorry

end find_k_plus_a_l17_17023


namespace polar_to_rectangular_l17_17650

theorem polar_to_rectangular :
  let x := 16
  let y := 12
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  let new_r := 2 * r
  let new_θ := θ / 2
  let cos_half_θ := Real.sqrt ((1 + (x / r)) / 2)
  let sin_half_θ := Real.sqrt ((1 - (x / r)) / 2)
  let new_x := new_r * cos_half_θ
  let new_y := new_r * sin_half_θ
  new_x = 40 * Real.sqrt 0.9 ∧ new_y = 40 * Real.sqrt 0.1 := by
  sorry

end polar_to_rectangular_l17_17650


namespace regular_octagon_interior_angle_l17_17802

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l17_17802


namespace ones_digit_7_pow_35_l17_17327

theorem ones_digit_7_pow_35 : (7^35) % 10 = 3 := 
by
  sorry

end ones_digit_7_pow_35_l17_17327


namespace max_pies_without_ingredients_l17_17982

theorem max_pies_without_ingredients :
  let total_pies := 48
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 2
  let cayenne_pies := 3 * total_pies / 8
  let soy_nut_pies := total_pies / 8
  total_pies - max chocolate_pies (max marshmallow_pies (max cayenne_pies soy_nut_pies)) = 24 := by
{
  sorry
}

end max_pies_without_ingredients_l17_17982


namespace line_x_intercept_l17_17146

-- Define the given points
def Point1 : ℝ × ℝ := (10, 3)
def Point2 : ℝ × ℝ := (-10, -7)

-- Define the x-intercept problem
theorem line_x_intercept (x : ℝ) : 
  ∃ m b : ℝ, (Point1.2 = m * Point1.1 + b) ∧ (Point2.2 = m * Point2.1 + b) ∧ (0 = m * x + b) → x = 4 :=
by
  sorry

end line_x_intercept_l17_17146


namespace train_length_l17_17161

theorem train_length (v_train : ℝ) (v_man : ℝ) (t : ℝ) (length_train : ℝ)
  (h1 : v_train = 55) (h2 : v_man = 7) (h3 : t = 10.45077684107852) :
  length_train = 180 :=
by
  sorry

end train_length_l17_17161


namespace cost_price_is_800_l17_17118

theorem cost_price_is_800 (mp sp cp : ℝ) (h1 : mp = 1100) (h2 : sp = 0.8 * mp) (h3 : sp = 1.1 * cp) :
  cp = 800 :=
by
  sorry

end cost_price_is_800_l17_17118


namespace license_plates_count_l17_17952

/--
Define the conditions and constants.
-/
def num_letters := 26
def num_first_digit := 5  -- Odd digits
def num_second_digit := 5 -- Even digits

theorem license_plates_count : num_letters ^ 3 * num_first_digit * num_second_digit = 439400 := by
  sorry

end license_plates_count_l17_17952


namespace true_prices_for_pie_and_mead_l17_17717

-- Definitions for true prices
variable (k m : ℕ)

-- Definitions for conditions
def honest_pravdoslav (k m : ℕ) : Prop :=
  4*k = 3*(m + 2) ∧ 4*(m+2) = 3*k + 14

theorem true_prices_for_pie_and_mead (k m : ℕ) (h : honest_pravdoslav k m) : k = 6 ∧ m = 6 := sorry

end true_prices_for_pie_and_mead_l17_17717


namespace soda_relationship_l17_17440

theorem soda_relationship (J : ℝ) (L : ℝ) (A : ℝ) (hL : L = 1.75 * J) (hA : A = 1.20 * J) : 
  (L - A) / A = 0.46 := 
by
  sorry

end soda_relationship_l17_17440


namespace total_money_from_tshirts_l17_17764

def num_tshirts_sold := 20
def money_per_tshirt := 215

theorem total_money_from_tshirts :
  num_tshirts_sold * money_per_tshirt = 4300 :=
by
  sorry

end total_money_from_tshirts_l17_17764


namespace probability_function_increasing_l17_17849

theorem probability_function_increasing : 
  let outcomes := [(m, n) | m ∈ Finset.range(1, 7), n ∈ Finset.range(1, 7)],
      condition := (fun (mn: ℕ × ℕ) => let (m, n) := mn in (2 * m - n ≤ 6)),
      favorable := filter condition outcomes in
  (favorable.card : ℚ) / (outcomes.card : ℚ) = 3 / 4 :=
by sorry

end probability_function_increasing_l17_17849


namespace regular_octagon_angle_measure_l17_17837

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l17_17837


namespace climb_stairs_l17_17856

noncomputable def u (n : ℕ) : ℝ :=
  let Φ := (1 + Real.sqrt 5) / 2
  let φ := (1 - Real.sqrt 5) / 2
  let A := (1 + Real.sqrt 5) / (2 * Real.sqrt 5)
  let B := (Real.sqrt 5 - 1) / (2 * Real.sqrt 5)
  A * (Φ ^ n) + B * (φ ^ n)

theorem climb_stairs (n : ℕ) (hn : n ≥ 1) : u n = A * (Φ ^ n) + B * (φ ^ n) := sorry

end climb_stairs_l17_17856


namespace compute_j_in_polynomial_arithmetic_progression_l17_17114

theorem compute_j_in_polynomial_arithmetic_progression 
  (P : Polynomial ℝ)
  (roots : Fin 4 → ℝ)
  (hP : P = Polynomial.C 400 + Polynomial.X * (Polynomial.C k + Polynomial.X * (Polynomial.C j + Polynomial.X * (Polynomial.C 0 + Polynomial.X))))
  (arithmetic_progression : ∃ b d : ℝ, roots 0 = b ∧ roots 1 = b + d ∧ roots 2 = b + 2 * d ∧ roots 3 = b + 3 * d ∧ Polynomial.degree P = 4) :
  j = -200 :=
by
  sorry

end compute_j_in_polynomial_arithmetic_progression_l17_17114


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17828

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17828


namespace function_two_common_points_with_xaxis_l17_17202

theorem function_two_common_points_with_xaxis (c : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x + c = 0 → x = -1 ∨ x = 1) → (c = -2 ∨ c = 2) :=
by
  sorry

end function_two_common_points_with_xaxis_l17_17202


namespace slope_undefined_iff_vertical_l17_17407

theorem slope_undefined_iff_vertical (m : ℝ) :
  let M := (2 * m + 3, m)
  let N := (m - 2, 1)
  (2 * m + 3 - (m - 2) = 0 ∧ m - 1 ≠ 0) ↔ m = -5 :=
by
  sorry

end slope_undefined_iff_vertical_l17_17407


namespace certain_number_k_l17_17420

theorem certain_number_k (x : ℕ) (k : ℕ) (h1 : x = 14) (h2 : 2^x - 2^(x-2) = k * 2^12) : k = 3 := by
  sorry

end certain_number_k_l17_17420


namespace quadratic_function_expression_l17_17403

-- Given conditions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def intersects_A (a b c : ℝ) : Prop := f a b c (-2) = 0

def intersects_B (a b c : ℝ) : Prop := f a b c 4 = 0

def maximum_value (a b c : ℝ) : Prop := f a b c 1 = 9

-- Prove the function expression
theorem quadratic_function_expression :
  ∃ a b c : ℝ, intersects_A a b c ∧ intersects_B a b c ∧ maximum_value a b c ∧ 
  ∀ x : ℝ, f a b c x = -x^2 + 2 * x + 8 := 
sorry

end quadratic_function_expression_l17_17403


namespace inequality_proof_l17_17286

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l17_17286


namespace sequence_expression_l17_17068

theorem sequence_expression (a : ℕ → ℕ) (h₀ : a 1 = 33) (h₁ : ∀ n, a (n + 1) - a n = 2 * n) : 
  ∀ n, a n = n^2 - n + 33 :=
by
  sorry

end sequence_expression_l17_17068


namespace inequality_proof_l17_17252

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l17_17252


namespace derivative_equals_l17_17925

noncomputable def func (x : ℝ) : ℝ :=
  (3 / (8 * Real.sqrt 2) * Real.log ((Real.sqrt 2 + Real.tanh x) / (Real.sqrt 2 - Real.tanh x)))
  - (Real.tanh x / (4 * (2 - (Real.tanh x)^2)))

theorem derivative_equals :
  ∀ x : ℝ, deriv func x = 1 / (2 + (Real.cosh x)^2)^2 :=
by {
  sorry
}

end derivative_equals_l17_17925


namespace xiaomin_house_position_l17_17636

-- Define the initial position of the school at the origin
def school_pos : ℝ × ℝ := (0, 0)

-- Define the movement east and south from the school's position
def xiaomin_house_pos (east_distance south_distance : ℝ) : ℝ × ℝ :=
  (school_pos.1 + east_distance, school_pos.2 - south_distance)

-- The given conditions
def east_distance := 200
def south_distance := 150

-- The theorem stating Xiaomin's house position
theorem xiaomin_house_position :
  xiaomin_house_pos east_distance south_distance = (200, -150) :=
by
  -- Skipping the proof steps
  sorry

end xiaomin_house_position_l17_17636


namespace floor_sqrt_80_l17_17916

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l17_17916


namespace painting_area_l17_17600

theorem painting_area
  (wall_height : ℝ) (wall_length : ℝ)
  (window_height : ℝ) (window_length : ℝ)
  (door_height : ℝ) (door_length : ℝ)
  (cond1 : wall_height = 10) (cond2 : wall_length = 15)
  (cond3 : window_height = 3) (cond4 : window_length = 5)
  (cond5 : door_height = 2) (cond6 : door_length = 7) :
  wall_height * wall_length - window_height * window_length - door_height * door_length = 121 := 
by
  simp [cond1, cond2, cond3, cond4, cond5, cond6]
  sorry

end painting_area_l17_17600


namespace whitney_spent_179_l17_17486

def total_cost (books_whales books_fish magazines book_cost magazine_cost : ℕ) : ℕ :=
  (books_whales + books_fish) * book_cost + magazines * magazine_cost

theorem whitney_spent_179 :
  total_cost 9 7 3 11 1 = 179 :=
by
  sorry

end whitney_spent_179_l17_17486


namespace boat_speed_in_still_water_l17_17149

/-- Prove the speed of the boat in still water given the conditions -/
theorem boat_speed_in_still_water (V_s : ℝ) (T : ℝ) (D : ℝ) (V_b : ℝ) :
  V_s = 4 ∧ T = 4 ∧ D = 112 ∧ (D / T = V_b + V_s) → V_b = 24 := sorry

end boat_speed_in_still_water_l17_17149


namespace inequality_inequality_l17_17259

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l17_17259


namespace regular_octagon_interior_angle_l17_17822

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l17_17822


namespace perpendicular_distance_H_to_plane_EFG_l17_17386

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def E : Point3D := ⟨5, 0, 0⟩
def F : Point3D := ⟨0, 3, 0⟩
def G : Point3D := ⟨0, 0, 4⟩
def H : Point3D := ⟨0, 0, 0⟩

def distancePointToPlane (H E F G : Point3D) : ℝ := sorry

theorem perpendicular_distance_H_to_plane_EFG :
  distancePointToPlane H E F G = 1.8 := sorry

end perpendicular_distance_H_to_plane_EFG_l17_17386


namespace cara_neighbors_l17_17517

theorem cara_neighbors (friends : Finset Person) (mark : Person) (cara : Person) (h_mark : mark ∈ friends) (h_len : friends.card = 8) :
  ∃ pairs : Finset (Person × Person), pairs.card = 6 ∧
    ∀ (p : Person × Person), p ∈ pairs → p.1 = mark ∨ p.2 = mark :=
by
  -- The proof goes here.
  sorry

end cara_neighbors_l17_17517


namespace derivative_f_eq_l17_17181

noncomputable def f (x : ℝ) : ℝ :=
  (7^x * (3 * Real.sin (3 * x) + Real.cos (3 * x) * Real.log 7)) / (9 + Real.log 7 ^ 2)

theorem derivative_f_eq :
  ∀ x : ℝ, deriv f x = 7^x * Real.cos (3 * x) :=
by
  intro x
  sorry

end derivative_f_eq_l17_17181


namespace scientific_notation_470M_l17_17749

theorem scientific_notation_470M :
  (470000000 : ℝ) = 4.7 * 10^8 :=
sorry

end scientific_notation_470M_l17_17749


namespace lines_through_origin_l17_17162

-- Define that a, b, c are in geometric progression
def geo_prog (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

-- Define the property of the line passing through the common point (0, 0)
def passes_through_origin (a b c : ℝ) : Prop :=
  ∀ x y, (a * x + b * y = c) → (x = 0 ∧ y = 0)

theorem lines_through_origin (a b c : ℝ) (h : geo_prog a b c) : passes_through_origin a b c :=
by
  sorry

end lines_through_origin_l17_17162


namespace coordinates_of_B_l17_17225

-- Define the initial coordinates of point A
def A : ℝ × ℝ := (1, -2)

-- Define the transformation to get point B from A
def B : ℝ × ℝ := (A.1 - 2, A.2 + 3)

theorem coordinates_of_B : B = (-1, 1) :=
by
  sorry

end coordinates_of_B_l17_17225


namespace total_white_roses_needed_l17_17100

theorem total_white_roses_needed : 
  let bouquets := 5 
  let table_decorations := 7 
  let roses_per_bouquet := 5 
  let roses_per_table_decoration := 12 in
  (bouquets * roses_per_bouquet) + (table_decorations * roses_per_table_decoration) = 109 := by
  sorry

end total_white_roses_needed_l17_17100


namespace floor_sqrt_80_eq_8_l17_17921

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l17_17921


namespace range_of_a_l17_17200

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ :=
  (m * x + n) / (x ^ 2 + 1)

example (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1) : 
  m = 2 ∧ n = 0 :=
sorry

theorem range_of_a (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1)
  (h_m : m = 2) (h_n : n = 0) {a : ℝ} : f (a-1) m n + f (a^2-1) m n < 0 ↔ 0 ≤ a ∧ a < 1 :=
sorry

end range_of_a_l17_17200


namespace Avianna_red_candles_l17_17105

theorem Avianna_red_candles (R : ℕ) : 
  (R / 27 = 5 / 3) → R = 45 := 
by
  sorry

end Avianna_red_candles_l17_17105


namespace inequality_holds_for_all_real_numbers_l17_17243

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l17_17243


namespace regular_octagon_angle_measure_l17_17834

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l17_17834


namespace both_miss_probability_l17_17984

-- Define the probabilities of hitting the target for Persons A and B 
def prob_hit_A : ℝ := 0.85
def prob_hit_B : ℝ := 0.8

-- Calculate the probabilities of missing the target
def prob_miss_A : ℝ := 1 - prob_hit_A
def prob_miss_B : ℝ := 1 - prob_hit_B

-- Prove that the probability of both missing the target is 0.03
theorem both_miss_probability : prob_miss_A * prob_miss_B = 0.03 :=
by
  sorry

end both_miss_probability_l17_17984


namespace regular_octagon_interior_angle_l17_17818

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l17_17818


namespace intersection_of_sets_l17_17683

theorem intersection_of_sets :
  let M := {-2, -1, 0, 1, 2}
  let N := {x | x < 0 ∨ x > 3}
  M ∩ N = {-2, -1} :=
by
  intro M N
  rw Set.inter_def
  rfl
  sorry

end intersection_of_sets_l17_17683


namespace tree_cost_l17_17415

theorem tree_cost (fence_length_yards : ℝ) (tree_width_feet : ℝ) (total_cost : ℝ) 
(h1 : fence_length_yards = 25) 
(h2 : tree_width_feet = 1.5) 
(h3 : total_cost = 400) : 
(total_cost / ((fence_length_yards * 3) / tree_width_feet) = 8) := 
by
  sorry

end tree_cost_l17_17415


namespace Mitch_weekly_earnings_l17_17098

theorem Mitch_weekly_earnings :
  (let weekdays_hours := 5 * 5
       weekend_hours := 3 * 2
       weekday_rate := 3
       weekend_rate := 2 * 3 in
   (weekdays_hours * weekday_rate + weekend_hours * weekend_rate = 111)) :=
by
  sorry

end Mitch_weekly_earnings_l17_17098


namespace floor_sqrt_80_eq_8_l17_17920

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l17_17920


namespace tetrahedron_formable_l17_17688

theorem tetrahedron_formable (x : ℝ) (hx_pos : 0 < x) (hx_bound : x < (Real.sqrt 6 + Real.sqrt 2) / 2) :
  true := 
sorry

end tetrahedron_formable_l17_17688


namespace sequence_term_expression_l17_17942

theorem sequence_term_expression (a : ℕ → ℝ) (S : ℕ → ℝ) (C : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, S n + n * a n = C)
  (h3 : ∀ n ≥ 2, (n + 1) * a n = (n - 1) * a (n - 1)) :
  ∀ n, a n = 2 / (n * (n + 1)) :=
by
  sorry

end sequence_term_expression_l17_17942


namespace john_twice_as_old_in_x_years_l17_17933

def frank_is_younger (john_age frank_age : ℕ) : Prop :=
  frank_age = john_age - 15

def frank_future_age (frank_age : ℕ) : ℕ :=
  frank_age + 4

def john_future_age (john_age : ℕ) : ℕ :=
  john_age + 4

theorem john_twice_as_old_in_x_years (john_age frank_age x : ℕ) 
  (h1 : frank_is_younger john_age frank_age)
  (h2 : frank_future_age frank_age = 16)
  (h3 : john_age = frank_age + 15) :
  (john_age + x) = 2 * (frank_age + x) → x = 3 :=
by 
  -- Skip the proof part
  sorry

end john_twice_as_old_in_x_years_l17_17933


namespace smallest_fraction_numerator_l17_17507

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (a * 4 > b * 3) ∧ (a = 73) := 
sorry

end smallest_fraction_numerator_l17_17507


namespace largest_A_divisible_by_8_l17_17850

theorem largest_A_divisible_by_8 (A B C : ℕ) (h1 : A = 8 * B + C) (h2 : B = C) (h3 : C < 8) : A ≤ 9 * 7 :=
by sorry

end largest_A_divisible_by_8_l17_17850


namespace cone_lateral_surface_area_l17_17934

theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 3) (hh : h = 4) : 15 * Real.pi = Real.pi * r * (Real.sqrt (r^2 + h^2)) :=
by
  -- Prove that 15π = π * r * sqrt(r^2 + h^2) for r = 3 and h = 4
  sorry

end cone_lateral_surface_area_l17_17934


namespace projection_correct_l17_17699

open Real
open Finset

-- Define the vectors a and b 
def a : EuclideanSpace ℝ (Fin 2) := ![1, 2]
def b : EuclideanSpace ℝ (Fin 2) := ![-1, 3]

-- Define the projection operation
def proj (u v : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  (dot_product u v / (dot_product v v)) • v

-- The specific projection statement to prove
theorem projection_correct :
  proj a b = ![-(1/2 : ℝ), 3/2] :=
  sorry

end projection_correct_l17_17699


namespace find_range_of_values_l17_17936

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem find_range_of_values (f : ℝ → ℝ) (h_even : is_even f)
  (h_increasing : is_increasing_on_nonneg f) (h_f1_zero : f 1 = 0) :
  { x : ℝ | f (Real.log x / Real.log (1/2)) > 0 } = 
  { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by 
  sorry

end find_range_of_values_l17_17936


namespace total_yellow_leaves_l17_17164

noncomputable def calculate_yellow_leaves (total : ℕ) (percent_brown : ℕ) (percent_green : ℕ) : ℕ :=
  let brown_leaves := (total * percent_brown + 50) / 100
  let green_leaves := (total * percent_green + 50) / 100
  total - (brown_leaves + green_leaves)

theorem total_yellow_leaves :
  let t_yellow := calculate_yellow_leaves 15 25 40
  let f_yellow := calculate_yellow_leaves 22 30 20
  let s_yellow := calculate_yellow_leaves 30 15 50
  t_yellow + f_yellow + s_yellow = 26 :=
by
  sorry

end total_yellow_leaves_l17_17164


namespace cube_root_simplification_l17_17634

theorem cube_root_simplification : 
  ∃ (a b : ℕ), (∃ c : ℕ, 8000 = c ^ 3) ∧ a * b ^ c = 8000 ∧ b = 1 ∧ a + b = 21 :=
by
  use 20
  use 1
  use 20
  sorry

end cube_root_simplification_l17_17634


namespace Sue_button_count_l17_17740

variable (K S : ℕ)

theorem Sue_button_count (H1 : 64 = 5 * K + 4) (H2 : S = K / 2) : S = 6 := 
by
sorry

end Sue_button_count_l17_17740


namespace simplify_expression_l17_17148

theorem simplify_expression : 
    2 * Real.sqrt 12 + 3 * Real.sqrt (4 / 3) - Real.sqrt (16 / 3) - (2 / 3) * Real.sqrt 48 = 2 * Real.sqrt 3 :=
by
  sorry

end simplify_expression_l17_17148


namespace max_value_fraction_squares_l17_17733

-- Let x and y be positive real numbers
variable (x y : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)

theorem max_value_fraction_squares (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ k, (x + 2 * y)^2 / (x^2 + y^2) ≤ k) ∧ (∀ z, (x + 2 * y)^2 / (x^2 + y^2) ≤ z) → k = 9 / 2 :=
by
  sorry

end max_value_fraction_squares_l17_17733


namespace average_of_quantities_l17_17310

theorem average_of_quantities (a1 a2 a3 a4 a5 : ℝ) :
  ((a1 + a2 + a3) / 3 = 4) →
  ((a4 + a5) / 2 = 21.5) →
  ((a1 + a2 + a3 + a4 + a5) / 5 = 11) :=
by
  intros h3 h2
  sorry

end average_of_quantities_l17_17310


namespace distance_PF_equilateral_l17_17581

-- Given conditions as definitions
def F : ℝ × ℝ := (1/2, 0)
def directrix l : ℝ := -1/2
def parabola (P : ℝ × ℝ) : Prop := P.2 ^ 2 = 2 * P.1
def lies_on_directrix (Q : ℝ × ℝ) : Prop := Q.1 = -1/2
def parallel_to_x_axis (PQ : ℝ × ℝ) : Prop := PQ.2 = 0
def equidistant (PQ QF : ℝ) : Prop := PQ = QF

-- The key property we want to prove
theorem distance_PF_equilateral (P Q : ℝ × ℝ) (hP : parabola P) (hQ : lies_on_directrix Q) (h1 : parallel_to_x_axis (P.1 - Q.1, 0)) (h2 : equidistant ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ((Q.1 - F.1)^2 + (Q.2 - F.2)^2)) : 
  ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 2^2 :=
by sorry

end distance_PF_equilateral_l17_17581


namespace find_cos_C_l17_17970

noncomputable def cos_C (a b c : ℝ) : ℝ :=
(a^2 + b^2 - c^2) / (2 * a * b)

theorem find_cos_C (a b c : ℝ) (h1 : b^2 = a * c) (h2 : c = 2 * a) :
  cos_C a b c = -√2/4 :=
by
  sorry

end find_cos_C_l17_17970


namespace scientific_notation_of_15510000_l17_17502

/--
Express 15,510,000 in scientific notation.

Theorem: 
Given that the scientific notation for large numbers is of the form \(a \times 10^n\) where \(1 \leq |a| < 10\),
prove that expressing 15,510,000 in scientific notation results in 1.551 × 10^7.
-/
theorem scientific_notation_of_15510000 : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 15510000 = a * 10 ^ n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end scientific_notation_of_15510000_l17_17502


namespace transition_algebraic_expression_l17_17776

theorem transition_algebraic_expression (k : ℕ) (hk : k > 0) :
  (k + 1 + k) * (k + 1 + k + 1) / (k + 1) = 4 * k + 2 :=
sorry

end transition_algebraic_expression_l17_17776


namespace lava_lamp_probability_l17_17110

/-- Ryan has 4 red lava lamps and 2 blue lava lamps; 
    he arranges them in a row on a shelf randomly, and then randomly turns 3 of them on. 
    Prove that the probability that the leftmost lamp is blue and off, 
    and the rightmost lamp is red and on is 2/25. -/
theorem lava_lamp_probability : 
  let total_arrangements := (Nat.choose 6 2) 
  let total_on := (Nat.choose 6 3)
  let favorable_arrangements := (Nat.choose 4 1)
  let favorable_on := (Nat.choose 4 2)
  let favorable_outcomes := 4 * 6
  let probability := (favorable_outcomes : ℚ) / (total_arrangements * total_on : ℚ)
  probability = 2 / 25 := 
by
  sorry

end lava_lamp_probability_l17_17110


namespace interior_angle_regular_octagon_l17_17815

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l17_17815


namespace half_angle_quadrant_l17_17193

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end half_angle_quadrant_l17_17193


namespace age_sum_l17_17339

variable (b : ℕ)
variable (a : ℕ := b + 2)
variable (c : ℕ := b / 2)

theorem age_sum : b = 10 → a + b + c = 27 :=
by
  intros h
  rw [h]
  sorry

end age_sum_l17_17339


namespace train_length_correct_l17_17025

noncomputable def length_bridge : ℝ := 300
noncomputable def time_to_cross : ℝ := 45
noncomputable def speed_train_kmh : ℝ := 44

-- Conversion from km/h to m/s
noncomputable def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

-- Total distance covered
noncomputable def total_distance_covered : ℝ := speed_train_ms * time_to_cross

-- Length of the train
noncomputable def length_train : ℝ := total_distance_covered - length_bridge

theorem train_length_correct : abs (length_train - 249.9) < 0.1 :=
by
  sorry

end train_length_correct_l17_17025


namespace pages_in_book_l17_17445

-- Define the initial conditions
variable (P : ℝ) -- total number of pages in the book
variable (h_read_20_percent : 0.20 * P = 320 * 0.20 / 0.80) -- Nate has read 20% of the book and the rest 80%

-- The goal is to show that P = 400
theorem pages_in_book (P : ℝ) :
  (0.80 * P = 320) → P = 400 :=
by
  sorry

end pages_in_book_l17_17445


namespace coupon_value_l17_17431

theorem coupon_value
  (bill : ℝ)
  (milk_cost : ℝ)
  (bread_cost : ℝ)
  (detergent_cost : ℝ)
  (banana_cost_per_pound : ℝ)
  (banana_weight : ℝ)
  (half_off : ℝ)
  (amount_left : ℝ)
  (total_without_coupon : ℝ)
  (total_spent : ℝ)
  (coupon_value : ℝ) :
  bill = 20 →
  milk_cost = 4 →
  bread_cost = 3.5 →
  detergent_cost = 10.25 →
  banana_cost_per_pound = 0.75 →
  banana_weight = 2 →
  half_off = 0.5 →
  amount_left = 4 →
  total_without_coupon = milk_cost * half_off + bread_cost + detergent_cost + banana_cost_per_pound * banana_weight →
  total_spent = bill - amount_left →
  coupon_value = total_without_coupon - total_spent →
  coupon_value = 1.25 :=
by
  sorry

end coupon_value_l17_17431


namespace calculate_fraction_value_l17_17879

theorem calculate_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 := 
  sorry

end calculate_fraction_value_l17_17879


namespace g_five_eq_one_l17_17999

noncomputable def g : ℝ → ℝ := sorry

theorem g_five_eq_one (hx : ∀ x y : ℝ, g (x * y) = g x * g y) (h1 : g 1 ≠ 0) : g 5 = 1 :=
sorry

end g_five_eq_one_l17_17999


namespace area_triangle_BFC_l17_17108

-- Definitions based on conditions
def Rectangle (A B C D : Type) (AB BC CD DA : ℝ) := AB = 5 ∧ BC = 12 ∧ CD = 5 ∧ DA = 12

def PointOnDiagonal (F A C : Type) := True  -- Simplified definition as being on the diagonal
def Perpendicular (B F A C : Type) := True  -- Simplified definition as being perpendicular

-- Main theorem statement
theorem area_triangle_BFC 
  (A B C D F : Type)
  (rectangle_ABCD : Rectangle A B C D 5 12 5 12)
  (F_on_AC : PointOnDiagonal F A C)
  (BF_perpendicular_AC : Perpendicular B F A C) :
  ∃ (area : ℝ), area = 30 :=
sorry

end area_triangle_BFC_l17_17108


namespace find_a1_l17_17191

noncomputable def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a (n+1) + a n = 4*n

theorem find_a1 (a : ℕ → ℕ) (h : is_arithmetic_sequence a) : a 1 = 1 := by
  sorry

end find_a1_l17_17191


namespace mr_kishore_savings_l17_17488

theorem mr_kishore_savings :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 3940
  let total_expenses := rent + milk + groceries + education + petrol + misc
  let savings_percentage := 0.10
  let salary := total_expenses / (1 - savings_percentage)
  let savings := savings_percentage * salary
  savings = 1937.78 := by
  sorry

end mr_kishore_savings_l17_17488


namespace domain_of_g_l17_17390

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊ x^2 - 9 * x + 21 ⌋

theorem domain_of_g :
  { x : ℝ | ∃ y : ℝ, g x = y } = { x : ℝ | x ≤ 4 ∨ x ≥ 5 } :=
by
  sorry

end domain_of_g_l17_17390


namespace grooming_time_5_dogs_3_cats_l17_17088

theorem grooming_time_5_dogs_3_cats :
  (2.5 * 5 + 0.5 * 3) * 60 = 840 :=
by
  -- Prove that grooming 5 dogs and 3 cats takes 840 minutes.
  sorry

end grooming_time_5_dogs_3_cats_l17_17088


namespace shauna_lowest_score_l17_17450

theorem shauna_lowest_score :
  ∀ (scores : List ℕ) (score1 score2 score3 : ℕ), 
    scores = [score1, score2, score3] → 
    score1 = 82 →
    score2 = 88 →
    score3 = 93 →
    (∃ (s4 s5 : ℕ), s4 + s5 = 162 ∧ s4 ≤ 100 ∧ s5 ≤ 100) ∧
    score1 + score2 + score3 + s4 + s5 = 425 →
    min s4 s5 = 62 := 
by 
  sorry

end shauna_lowest_score_l17_17450


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17830

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17830


namespace regular_octagon_interior_angle_l17_17805

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17805


namespace inequality_xyz_l17_17276

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l17_17276


namespace radius_any_positive_real_l17_17221

theorem radius_any_positive_real (r : ℝ) (h₁ : r > 0) 
    (h₂ : r * (2 * Real.pi * r) = 2 * Real.pi * r^2) : True :=
by
  sorry

end radius_any_positive_real_l17_17221


namespace range_of_m_l17_17686

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

noncomputable def is_monotonically_decreasing_in_domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

theorem range_of_m (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_mono : is_monotonically_decreasing_in_domain f (-2) 2) :
  ∀ m : ℝ, (f (1 - m) + f (1 - m^2) < 0) → -2 < m ∧ m < 1 :=
sorry

end range_of_m_l17_17686


namespace max_xy_l17_17977

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x / 3 + y / 4 = 1) : xy ≤ 3 :=
by {
  -- proof omitted
  sorry
}

end max_xy_l17_17977


namespace factor_expression_l17_17396

-- Define the variables
variables (x : ℝ)

-- State the theorem to prove
theorem factor_expression : 3 * x * (x + 1) + 7 * (x + 1) = (3 * x + 7) * (x + 1) :=
by
  sorry

end factor_expression_l17_17396


namespace sum_of_edges_l17_17168

theorem sum_of_edges (n : ℕ) (total_length large_edge small_edge : ℤ) : 
  n = 27 → 
  total_length = 828 → -- convert to millimeters
  large_edge = total_length / 12 → 
  small_edge = large_edge / 3 → 
  (large_edge + small_edge) / 10 = 92 :=
by
  intros
  sorry

end sum_of_edges_l17_17168


namespace sufficient_but_not_necessary_condition_for_square_l17_17012

theorem sufficient_but_not_necessary_condition_for_square (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ (¬(x^2 > 4 → x > 3)) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_square_l17_17012


namespace choose_8_3_l17_17716

/- 
  Prove that the number of ways to choose 3 elements out of 8 is 56 
-/
theorem choose_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end choose_8_3_l17_17716


namespace scientific_notation_470M_l17_17750

theorem scientific_notation_470M :
  (470000000 : ℝ) = 4.7 * 10^8 :=
sorry

end scientific_notation_470M_l17_17750


namespace ladder_geometric_sequence_solution_l17_17154

-- A sequence {aₙ} is a 3rd-order ladder geometric sequence given by a_{n+3}^2 = a_n * a_{n+6} for any positive integer n
def ladder_geometric_3rd_order (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 3) ^ 2 = a n * a (n + 6)

-- Initial conditions
def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 4 = 2

-- Main theorem to be proven in Lean 4
theorem ladder_geometric_sequence_solution :
  ∃ a : ℕ → ℝ, ladder_geometric_3rd_order a ∧ initial_conditions a ∧ a 10 = 8 :=
by
  sorry

end ladder_geometric_sequence_solution_l17_17154


namespace square_roots_equal_implication_l17_17965

theorem square_roots_equal_implication (b : ℝ) (h : 5 * b = 3 + 2 * b) : -b = -1 := 
by sorry

end square_roots_equal_implication_l17_17965


namespace polygon_number_of_sides_and_interior_sum_l17_17868

-- Given conditions
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)
def exterior_angle_sum : ℝ := 360

-- Proof problem statement
theorem polygon_number_of_sides_and_interior_sum (n : ℕ)
  (h : interior_angle_sum n = 3 * exterior_angle_sum) :
  n = 8 ∧ interior_angle_sum n = 1080 :=
by
  sorry

end polygon_number_of_sides_and_interior_sum_l17_17868


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17790

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17790


namespace inequality_proof_l17_17284

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l17_17284


namespace valid_combinations_l17_17361

-- Definitions based on conditions
def h : Nat := 4  -- number of herbs
def c : Nat := 6  -- number of crystals
def r : Nat := 3  -- number of negative reactions

-- Theorem statement based on the problem and solution
theorem valid_combinations : (h * c) - r = 21 := by
  sorry

end valid_combinations_l17_17361


namespace tax_percentage_excess_income_l17_17569

theorem tax_percentage_excess_income :
  ∀ (rate : ℝ) (total_tax income : ℝ), 
  rate = 0.15 →
  total_tax = 8000 →
  income = 50000 →
  (total_tax - income * rate) / (income - 40000) = 0.2 :=
by
  intros rate total_tax income hrate htotal hincome
  -- proof omitted
  sorry

end tax_percentage_excess_income_l17_17569


namespace inequality_holds_for_real_numbers_l17_17270

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l17_17270


namespace inequality_holds_for_all_real_numbers_l17_17241

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l17_17241


namespace floor_neg_seven_fourths_l17_17895

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l17_17895


namespace surcharge_X_is_2_17_percent_l17_17322

def priceX : ℝ := 575
def priceY : ℝ := 530
def surchargeY : ℝ := 0.03
def totalSaved : ℝ := 41.60

theorem surcharge_X_is_2_17_percent :
  let surchargeX := (2.17 / 100)
  let totalCostX := priceX + (priceX * surchargeX)
  let totalCostY := priceY + (priceY * surchargeY)
  (totalCostX - totalCostY = totalSaved) →
  surchargeX * 100 = 2.17 :=
by
  sorry

end surcharge_X_is_2_17_percent_l17_17322


namespace log_roots_equivalence_l17_17186

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 5 / Real.log 3
noncomputable def c : ℝ := Real.log 2 / Real.log 5

theorem log_roots_equivalence :
  (x : ℝ) → (x = a ∨ x = b ∨ x = c) ↔ (x^3 - (a + b + c)*x^2 + (a*b + b*c + c*a)*x - a*b*c = 0) := by
  sorry

end log_roots_equivalence_l17_17186


namespace log_expression_value_l17_17660

theorem log_expression_value (lg : ℕ → ℤ) :
  (lg 4 = 2 * lg 2) →
  (lg 20 = lg 4 + lg 5) →
  lg 4 + lg 5 * lg 20 + (lg 5)^2 = 2 :=
by
  intros h1 h2
  sorry

end log_expression_value_l17_17660


namespace c_payment_l17_17150

theorem c_payment 
  (A_rate : ℝ) (B_rate : ℝ) (days : ℝ) (total_payment : ℝ) (C_fraction : ℝ) 
  (hA : A_rate = 1 / 6) 
  (hB : B_rate = 1 / 8) 
  (hdays : days = 3) 
  (hpayment : total_payment = 3200)
  (hC_fraction : C_fraction = 1 / 8) :
  total_payment * C_fraction = 400 :=
by {
  -- The proof would go here
  sorry
}

end c_payment_l17_17150


namespace rectangle_area_l17_17303

theorem rectangle_area (x : ℝ) (h1 : x > 0) (h2 : x * 4 = 28) : x = 7 :=
sorry

end rectangle_area_l17_17303


namespace bakery_flour_total_l17_17348

theorem bakery_flour_total :
  (0.2 + 0.1 + 0.15 + 0.05 + 0.1 = 0.6) :=
by {
  sorry
}

end bakery_flour_total_l17_17348


namespace weight_loss_total_l17_17053

theorem weight_loss_total :
  ∀ (weight1 weight2 weight3 weight4 : ℕ),
    weight1 = 27 →
    weight2 = weight1 - 7 →
    weight3 = 28 →
    weight4 = 28 →
    weight1 + weight2 + weight3 + weight4 = 103 :=
by
  intros weight1 weight2 weight3 weight4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end weight_loss_total_l17_17053


namespace total_computers_sold_l17_17589

theorem total_computers_sold (T : ℕ) (h_half_sales_laptops : 2 * T / 2 = T)
        (h_third_sales_netbooks : 3 * T / 3 = T)
        (h_desktop_sales : T - T / 2 - T / 3 = 12) : T = 72 :=
by
  sorry

end total_computers_sold_l17_17589


namespace cost_of_largest_pot_l17_17741

theorem cost_of_largest_pot
  (total_cost : ℝ)
  (n : ℕ)
  (a b : ℝ)
  (h_total_cost : total_cost = 7.80)
  (h_n : n = 6)
  (h_b : b = 0.25)
  (h_small_cost : ∃ x : ℝ, ∃ is_odd : ℤ → Prop, (∃ c: ℤ, x = c / 100 ∧ is_odd c) ∧
                  total_cost = x + (x + b) + (x + 2 * b) + (x + 3 * b) + (x + 4 * b) + (x + 5 * b)) :
  ∃ y, y = (x + 5*b) ∧ y = 1.92 :=
  sorry

end cost_of_largest_pot_l17_17741


namespace john_volunteer_hours_l17_17723

noncomputable def total_volunteer_hours :=
  let first_six_months_hours := 2 * 3 * 6
  let next_five_months_hours := 1 * 2 * 4 * 5
  let december_hours := 3 * 2
  first_six_months_hours + next_five_months_hours + december_hours

theorem john_volunteer_hours : total_volunteer_hours = 82 := by
  sorry

end john_volunteer_hours_l17_17723


namespace lebesgue_measure_invariance_l17_17759

-- Definitions of the properties of Lebesgue measure to translate the problem conditions

open Set
open MeasureTheory

variables {n : ℕ} (hn1 : n > 1) (hn2 : n ≥ 1)

def lebesgue_measure_invariant_under_translations (λ : MeasureTheory.Measure ℝ^n) : Prop :=
  ∀ (x : ℝ^n) (E : Set ℝ^n), MeasurableSet E → λ (x +ᵥ E) = λ E

def lebesgue_measure_invariant_under_rotations (λ : MeasureTheory.Measure ℝ^n) : Prop :=
  ∀ (φ : ℝ^n →ₗ[ℝ] ℝ^n), φ.IsLinearMap φ → φ ∘ φ⁻¹ = id →
  ∀ E : Set ℝ^n, MeasurableSet E → λ (E) = λ (φ '' E)

theorem lebesgue_measure_invariance {λ : MeasureTheory.Measure ℝ^n} :
  (∀ B ∈ MeasurableSet ℝ^n, λ(B) = inf { λ(G) | B ⊆ G ∧ IsOpen G }) →
  lebesgue_measure_invariant_under_translations λ →
  lebesgue_measure_invariant_under_rotations λ :=
by 
  sorry

# Let λ be the Lebesgue measure on ℝ^n, the theorem states that
# λ is invariant under translations for n ≥ 1 and under rotations for n > 1.

end lebesgue_measure_invariance_l17_17759


namespace problem_1_problem_2_l17_17754

theorem problem_1 (P_A P_B P_notA P_notB : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) (hNotA: P_notA = 1/2) (hNotB: P_notB = 3/5) : 
  P_A * P_notB + P_B * P_notA = 1/2 := 
by 
  rw [hA, hB, hNotA, hNotB]
  -- exact calculations here
  sorry

theorem problem_2 (P_A P_B : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) :
  (1 - (P_A * P_A * (1 - P_B) * (1 - P_B))) = 91/100 := 
by 
  rw [hA, hB]
  -- exact calculations here
  sorry

end problem_1_problem_2_l17_17754


namespace distance_from_Bangalore_l17_17768

noncomputable def calculate_distance (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) : ℕ :=
  let total_travel_minutes := (end_hour * 60 + end_minute) - (start_hour * 60 + start_minute) - halt_minutes
  let total_travel_hours := total_travel_minutes / 60
  speed * total_travel_hours

theorem distance_from_Bangalore (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) :
  speed = 87 ∧ start_hour = 9 ∧ start_minute = 0 ∧ end_hour = 13 ∧ end_minute = 45 ∧ halt_minutes = 45 →
  calculate_distance speed start_hour start_minute end_hour end_minute halt_minutes = 348 := by
  sorry

end distance_from_Bangalore_l17_17768


namespace larger_number_l17_17313

theorem larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end larger_number_l17_17313


namespace seq_15_l17_17405

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 2 else 2 * (n - 1) + 1 -- form inferred from solution

theorem seq_15 : seq 15 = 29 := by
  sorry

end seq_15_l17_17405


namespace find_milk_ounces_l17_17090

def bathroom_limit : ℕ := 32
def grape_juice_ounces : ℕ := 16
def water_ounces : ℕ := 8
def total_liquid_limit : ℕ := bathroom_limit
def total_liquid_intake : ℕ := grape_juice_ounces + water_ounces
def milk_ounces := total_liquid_limit - total_liquid_intake

theorem find_milk_ounces : milk_ounces = 8 := by
  sorry

end find_milk_ounces_l17_17090


namespace least_positive_integer_x_l17_17483

theorem least_positive_integer_x : ∃ x : ℕ, ((2 * x)^2 + 2 * 43 * (2 * x) + 43^2) % 53 = 0 ∧ 0 < x ∧ (∀ y : ℕ, ((2 * y)^2 + 2 * 43 * (2 * y) + 43^2) % 53 = 0 → 0 < y → x ≤ y) := 
by
  sorry

end least_positive_integer_x_l17_17483


namespace remainder_of_polynomial_division_l17_17531

noncomputable def evaluate_polynomial (x : ℂ) : ℂ :=
  x^100 + x^75 + x^50 + x^25 + 1

noncomputable def divisor_polynomial (x : ℂ) : ℂ :=
  x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_polynomial_division : 
  ∀ β : ℂ, divisor_polynomial β = 0 → evaluate_polynomial β = -1 :=
by
  intros β hβ
  sorry

end remainder_of_polynomial_division_l17_17531


namespace greatest_integer_less_than_150_gcd_18_eq_6_l17_17480

theorem greatest_integer_less_than_150_gcd_18_eq_6 :
  ∃ n : ℕ, n < 150 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ gcd m 18 = 6 → m ≤ n :=
by
  use 132
  split
  { 
    -- proof that 132 < 150 
    exact sorry 
  }
  split
  { 
    -- proof that gcd 132 18 = 6
    exact sorry 
  }
  {
    -- proof that 132 is the greatest such integer
    exact sorry 
  }

end greatest_integer_less_than_150_gcd_18_eq_6_l17_17480


namespace lines_intersect_hyperbola_once_l17_17066

theorem lines_intersect_hyperbola_once
  (l : LinearMap ℝ (ℝ × ℝ) ℝ)
  (h : ∀ (x y : ℝ), l (x, y) = if x = 1 then y = 2 * x - 1 ∨ y = -2 * x + 3 ∨ y = 4 * x - 3 ∨ x = 1 else true)
  (hyperbola : ∀ (x y : ℝ), x^2 - y^2 / 4 = 1)
  : (∃ (x y : ℝ), l (x, y) = 0 ∧ hyperbola x y ∧ x = 1 ∧ y = 1) :=
sorry

end lines_intersect_hyperbola_once_l17_17066


namespace problem_part1_problem_part2_l17_17948

def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def CR_A : Set ℝ := { x | x < 3 ∨ x > 7 }

theorem problem_part1 : A ∪ B = { x | 3 ≤ x ∧ x ≤ 7 } := by
  sorry

theorem problem_part2 : (CR_A ∩ B) = { x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10) } := by
  sorry

end problem_part1_problem_part2_l17_17948


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17789

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17789


namespace number_multiplies_p_plus_1_l17_17078

theorem number_multiplies_p_plus_1 (p q x : ℕ) 
  (hp : 1 < p) (hq : 1 < q)
  (hEq : x * (p + 1) = 25 * (q + 1))
  (hSum : p + q = 40) :
  x = 325 :=
sorry

end number_multiplies_p_plus_1_l17_17078


namespace even_sum_probability_l17_17388

-- Definition of probabilities for the first wheel
def prob_first_even : ℚ := 2 / 6
def prob_first_odd  : ℚ := 4 / 6

-- Definition of probabilities for the second wheel
def prob_second_even : ℚ := 3 / 8
def prob_second_odd  : ℚ := 5 / 8

-- The expected probability of the sum being even
theorem even_sum_probability : prob_first_even * prob_second_even + prob_first_odd * prob_second_odd = 13 / 24 := by
  sorry

end even_sum_probability_l17_17388


namespace geometric_arithmetic_sequence_l17_17551

theorem geometric_arithmetic_sequence 
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (q : ℝ) 
  (h0 : 0 < q) (h1 : q ≠ 1)
  (h2 : ∀ n, a_n n = a_n 1 * q ^ (n - 1)) -- a_n is a geometric sequence
  (h3 : 2 * a_n 3 * a_n 5 = a_n 4 * (a_n 3 + a_n 5)) -- a3, a5, a4 form an arithmetic sequence
  (h4 : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) -- S_n is the sum of the first n terms
  : S 6 / S 3 = 9 / 8 :=
by
  sorry

end geometric_arithmetic_sequence_l17_17551


namespace union_of_sets_l17_17058

-- Definitions based on conditions
def A : Set ℕ := {2, 3}
def B (a : ℕ) : Set ℕ := {1, a}
def condition (a : ℕ) : Prop := A ∩ (B a) = {2}

-- Main theorem to be proven
theorem union_of_sets (a : ℕ) (h : condition a) : A ∪ (B a) = {1, 2, 3} :=
sorry

end union_of_sets_l17_17058


namespace min_height_of_cuboid_l17_17862

theorem min_height_of_cuboid (h : ℝ) (side_len : ℝ) (small_spheres_r : ℝ) (large_sphere_r : ℝ) :
  side_len = 4 → 
  small_spheres_r = 1 → 
  large_sphere_r = 2 → 
  ∃ h_min : ℝ, h_min = 2 + 2 * Real.sqrt 7 ∧ h ≥ h_min := 
by
  sorry

end min_height_of_cuboid_l17_17862


namespace mod_product_2023_2024_2025_2026_l17_17383

theorem mod_product_2023_2024_2025_2026 :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 :=
by
  have h2023 : 2023 % 7 = 6 := by norm_num
  have h2024 : 2024 % 7 = 0 := by norm_num
  have h2025 : 2025 % 7 = 1 := by norm_num
  have h2026 : 2026 % 7 = 2 := by norm_num
  calc
    (2023 * 2024 * 2025 * 2026) % 7
      = ((2023 % 7) * (2024 % 7) * (2025 % 7) * (2026 % 7)) % 7 : by rw [Nat.mul_mod, Nat.mul_mod, Nat.mul_mod, Nat.mul_mod]
  ... = (6 * 0 * 1 * 2) % 7 : by rw [h2023, h2024, h2025, h2026]
  ... = 0 % 7 : by norm_num
  ... = 0 : by norm_num

end mod_product_2023_2024_2025_2026_l17_17383


namespace regular_octagon_interior_angle_l17_17841

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17841


namespace union_of_sets_l17_17949

theorem union_of_sets (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2 * a - 1 | a ∈ M}) :
  M ∪ N = {1, 2, 3} := by
  sorry

end union_of_sets_l17_17949


namespace present_age_of_son_l17_17021

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 32) (h2 : M + 2 = 2 * (S + 2)) : S = 30 :=
by
  sorry

end present_age_of_son_l17_17021


namespace system_sol_l17_17543

theorem system_sol {x y : ℝ} (h1 : x + 2 * y = -1) (h2 : 2 * x + y = 3) : x - y = 4 := by
  sorry

end system_sol_l17_17543


namespace count_integers_between_powers_l17_17206

noncomputable def power (a : ℝ) (b : ℝ) : ℝ := a^b

theorem count_integers_between_powers:
  let a := 10
  let b1 := 0.1
  let b2 := 0.4
  have exp1 : Float := (a + b1)
  have exp2 : Float := (a + b2)
  have n1 : ℤ := exp1^3.ceil
  have n2 : ℤ := exp2^3.floor
  n2 - n1 + 1 = 94 := 
begin
  sorry
end

end count_integers_between_powers_l17_17206


namespace median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l17_17620

-- Definition of points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- The problem statements as Lean theorems
theorem median_on_AB_eq : ∀ (A B : ℝ × ℝ), A = (4, 0) ∧ B = (6, 7) → ∃ (x y : ℝ), x - 10 * y + 30 = 0 := by
  intros
  sorry

theorem altitude_on_BC_eq : ∀ (B C : ℝ × ℝ), B = (6, 7) ∧ C = (0, 3) → ∃ (x y : ℝ), 3 * x + 2 * y - 12 = 0 := by
  intros
  sorry

theorem perp_bisector_on_AC_eq : ∀ (A C : ℝ × ℝ), A = (4, 0) ∧ C = (0, 3) → ∃ (x y : ℝ), 8 * x - 6 * y - 7 = 0 := by
  intros
  sorry

end median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l17_17620


namespace points_on_line_y1_gt_y2_l17_17546

theorem points_on_line_y1_gt_y2 (y1 y2 : ℝ) : 
    (∀ x y, y = -x + 3 → 
    ((x = -4 → y = y1) ∧ (x = 2 → y = y2))) → 
    y1 > y2 :=
by
  sorry

end points_on_line_y1_gt_y2_l17_17546


namespace esteban_exercise_days_l17_17744

theorem esteban_exercise_days
  (natasha_exercise_per_day : ℕ)
  (natasha_days : ℕ)
  (esteban_exercise_per_day : ℕ)
  (total_exercise_hours : ℕ)
  (hours_to_minutes : ℕ)
  (natasha_exercise_total : ℕ)
  (total_exercise_minutes : ℕ)
  (esteban_exercise_total : ℕ)
  (esteban_days : ℕ) :
  natasha_exercise_per_day = 30 →
  natasha_days = 7 →
  esteban_exercise_per_day = 10 →
  total_exercise_hours = 5 →
  hours_to_minutes = 60 →
  natasha_exercise_total = natasha_exercise_per_day * natasha_days →
  total_exercise_minutes = total_exercise_hours * hours_to_minutes →
  esteban_exercise_total = total_exercise_minutes - natasha_exercise_total →
  esteban_days = esteban_exercise_total / esteban_exercise_per_day →
  esteban_days = 9 :=
by
  sorry

end esteban_exercise_days_l17_17744


namespace three_digit_odd_number_is_803_l17_17223

theorem three_digit_odd_number_is_803 :
  ∃ (a b c : ℕ), 0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ c % 2 = 1 ∧
  100 * a + 10 * b + c = 803 ∧ (100 * a + 10 * b + c) / 11 = a^2 + b^2 + c^2 :=
by {
  sorry
}

end three_digit_odd_number_is_803_l17_17223


namespace prod_mod7_eq_zero_l17_17377

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l17_17377


namespace regular_octagon_interior_angle_l17_17810

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l17_17810


namespace regular_octagon_interior_angle_l17_17839

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17839


namespace probability_of_exactly_one_solves_l17_17752

variable (p1 p2 : ℝ)

theorem probability_of_exactly_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end probability_of_exactly_one_solves_l17_17752


namespace xy_sufficient_not_necessary_l17_17077

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy_lt_zero : x * y < 0) → abs (x - y) = abs x + abs y ∧ (abs (x - y) = abs x + abs y → x * y ≥ 0) := 
by
  sorry

end xy_sufficient_not_necessary_l17_17077


namespace amount_amys_money_l17_17033

def initial_dollars : ℝ := 2
def chores_payment : ℝ := 5 * 13
def birthday_gift : ℝ := 3
def total_after_gift : ℝ := initial_dollars + chores_payment + birthday_gift

def investment_percentage : ℝ := 0.20
def invested_amount : ℝ := investment_percentage * total_after_gift

def interest_rate : ℝ := 0.10
def interest_amount : ℝ := interest_rate * invested_amount
def total_investment : ℝ := invested_amount + interest_amount

def cost_of_toy : ℝ := 12
def remaining_after_toy : ℝ := total_after_gift - cost_of_toy

def grandparents_gift : ℝ := 2 * remaining_after_toy
def total_including_investment : ℝ := grandparents_gift + total_investment

def donation_percentage : ℝ := 0.25
def donated_amount : ℝ := donation_percentage * total_including_investment

def final_amount : ℝ := total_including_investment - donated_amount

theorem amount_amys_money :
  final_amount = 98.55 := by
  sorry

end amount_amys_money_l17_17033


namespace net_income_difference_l17_17302

theorem net_income_difference
    (terry_daily_income : ℝ := 24) (terry_daily_hours : ℝ := 6) (terry_days : ℕ := 7)
    (jordan_daily_income : ℝ := 30) (jordan_daily_hours : ℝ := 8) (jordan_days : ℕ := 6)
    (standard_week_hours : ℝ := 40) (overtime_rate_multiplier : ℝ := 1.5)
    (terry_tax_rate : ℝ := 0.12) (jordan_tax_rate : ℝ := 0.15) :
    jordan_daily_income * jordan_days - jordan_daily_income * jordan_days * jordan_tax_rate 
      + jordan_daily_income * jordan_days * jordan_daily_hours * (overtime_rate_multiplier - 1) * jordan_tax_rate
    - (terry_daily_income * terry_days - terry_daily_income * terry_days * terry_tax_rate 
      + terry_daily_income * terry_days * terry_daily_hours * (overtime_rate_multiplier - 1) * terry_tax_rate) 
      = 32.85 := 
sorry

end net_income_difference_l17_17302


namespace inequality_proof_l17_17287

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l17_17287


namespace inequality_holds_for_real_numbers_l17_17267

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l17_17267


namespace combined_degrees_l17_17301

theorem combined_degrees (S J W : ℕ) (h1 : S = 150) (h2 : J = S - 5) (h3 : W = S - 3) : S + J + W = 442 :=
by
  sorry

end combined_degrees_l17_17301


namespace Ali_catch_weight_l17_17030

-- Define the conditions and the goal
def Ali_Peter_Joey_fishing (p: ℝ) : Prop :=
  let Ali := 2 * p in
  let Joey := p + 1 in
  p + Ali + Joey = 25

-- State the problem
theorem Ali_catch_weight :
  ∃ p: ℝ, Ali_Peter_Joey_fishing p ∧ (2 * p = 12) :=
by
  sorry

end Ali_catch_weight_l17_17030


namespace y_relation_l17_17116

noncomputable def f (x : ℝ) : ℝ := -2 * x + 5

theorem y_relation (x1 y1 y2 y3 : ℝ) (h1 : y1 = f x1) (h2 : y2 = f (x1 - 2)) (h3 : y3 = f (x1 + 3)) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end y_relation_l17_17116


namespace value_of_x_l17_17075

theorem value_of_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 :=
by
  sorry

end value_of_x_l17_17075


namespace solution_interval_l17_17678

theorem solution_interval (x : ℝ) : 
  (3/8 + |x - 1/4| < 7/8) ↔ (-1/4 < x ∧ x < 3/4) := 
sorry

end solution_interval_l17_17678


namespace circumcircle_touches_iff_ratios_l17_17013

variables {A B C D I_A I_B : Point}
variables (k : Circle) (circABC : Circumcircle A B C) 
          (D_on_arc_AB_not_C : Arc A B k (¬ C ∈ Arc A B k))

-- Define D which is a point on the arc AB not passing through C
def point_D_on_arc : Prop :=
  D ∈ Arc A B k ∧ ¬ (C ∈ Arc A B k)

-- Define I_A and I_B as the centers of the incircles of triangles ADC and BDC respectively
def center_incircle_ADC : Prop :=
  is_incenter I_A triangle.ADC

def center_incircle_BDC : Prop :=
  is_incenter I_B triangle.BDC

-- Statement of the problem
theorem circumcircle_touches_iff_ratios {I_A I_B C : Point} :
  circumcircle.I_AI_BC.I_A_I_B_C_touches_circABC k I_A I_B C ↔
  (∃ D, point_D_on_arc D circumcircle.ABC ∧ ratio.AC_CD_AD_BD C D A B) :=
sorry

end circumcircle_touches_iff_ratios_l17_17013


namespace initial_population_l17_17145

theorem initial_population (P : ℝ) (h1 : P * 1.05 * 0.95 = 9975) : P = 10000 :=
by
  sorry

end initial_population_l17_17145


namespace break_room_capacity_l17_17393

theorem break_room_capacity :
  let people_per_table := 8
  let number_of_tables := 4
  people_per_table * number_of_tables = 32 :=
by
  let people_per_table := 8
  let number_of_tables := 4
  have h : people_per_table * number_of_tables = 32 := by sorry
  exact h

end break_room_capacity_l17_17393


namespace first_player_wins_l17_17621

noncomputable def game_win_guarantee : Prop :=
  ∃ (first_can_guarantee_win : Bool),
    first_can_guarantee_win = true

theorem first_player_wins :
  ∀ (nuts : ℕ) (players : (ℕ × ℕ)) (move : ℕ → ℕ) (end_condition : ℕ → Prop),
    nuts = 10 →
    players = (1, 2) →
    (∀ n, 0 < n ∧ n ≤ nuts → move n = n - 1) →
    (end_condition 3 = true) →
    (∀ x y z, x + y + z = 3 ↔ end_condition (x + y + z)) → 
    game_win_guarantee :=
by
  intros nuts players move end_condition H1 H2 H3 H4 H5
  sorry

end first_player_wins_l17_17621


namespace determine_m_minus_n_l17_17883

-- Definitions of the conditions
variables {m n : ℝ}

-- The proof statement
theorem determine_m_minus_n (h_eq : ∀ x y : ℝ, x^(4 - 3 * |m|) + y^(3 * |n|) = 2009 → x + y = 2009)
  (h_prod_lt_zero : m * n < 0)
  (h_sum : 0 < m + n ∧ m + n ≤ 3) : m - n = 4/3 := 
sorry

end determine_m_minus_n_l17_17883


namespace triangle_identity_l17_17082

theorem triangle_identity
  (A B C : ℝ) (a b c: ℝ)
  (h1: A + B + C = Real.pi)
  (h2: a = 2 * R * Real.sin A)
  (h3: b = 2 * R * Real.sin B)
  (h4: c = 2 * R * Real.sin C)
  (h5: Real.sin A = Real.sin B * Real.cos C + Real.cos B * Real.sin C) :
  (b * Real.cos C + c * Real.cos B) / a = 1 := 
  by 
  sorry

end triangle_identity_l17_17082


namespace shaded_region_area_l17_17996

theorem shaded_region_area (r_s r_l chord_AB : ℝ) (hs : r_s = 40) (hl : r_l = 60) (hc : chord_AB = 100) :
    chord_AB / 2 = 50 →
    60^2 - (chord_AB / 2)^2 = r_s^2 →
    (π * r_l^2) - (π * r_s^2) = 2500 * π :=
by
  intros h1 h2
  sorry

end shaded_region_area_l17_17996


namespace lizas_final_balance_l17_17592

-- Define the initial condition and subsequent changes
def initial_balance : ℕ := 800
def rent_payment : ℕ := 450
def paycheck_deposit : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

-- Calculate the final balance
def final_balance : ℕ :=
  let balance_after_rent := initial_balance - rent_payment
  let balance_after_paycheck := balance_after_rent + paycheck_deposit
  let balance_after_bills := balance_after_paycheck - (electricity_bill + internet_bill)
  balance_after_bills - phone_bill

-- Theorem to prove that the final balance is 1563
theorem lizas_final_balance : final_balance = 1563 :=
by
  sorry

end lizas_final_balance_l17_17592


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17791

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17791


namespace odd_number_expression_parity_l17_17433

theorem odd_number_expression_parity (o n : ℕ) (ho : ∃ k : ℕ, o = 2 * k + 1) :
  (o^2 + n * o) % 2 = 1 ↔ n % 2 = 0 :=
by
  sorry

end odd_number_expression_parity_l17_17433


namespace sum_of_series_l17_17857

theorem sum_of_series :
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := 
by
  sorry

end sum_of_series_l17_17857


namespace age_sum_is_27_l17_17340

noncomputable def a : ℕ := 12
noncomputable def b : ℕ := 10
noncomputable def c : ℕ := 5

theorem age_sum_is_27
  (h1: a = b + 2)
  (h2: b = 2 * c)
  (h3: b = 10) :
  a + b + c = 27 :=
  sorry

end age_sum_is_27_l17_17340


namespace interior_angle_regular_octagon_l17_17794

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l17_17794


namespace two_numbers_solution_l17_17473

noncomputable def a := 8 + Real.sqrt 58
noncomputable def b := 8 - Real.sqrt 58

theorem two_numbers_solution : 
  (Real.sqrt (a * b) = Real.sqrt 6) ∧ ((2 * a * b) / (a + b) = 3 / 4) → 
  (a = 8 + Real.sqrt 58 ∧ b = 8 - Real.sqrt 58) ∨ (a = 8 - Real.sqrt 58 ∧ b = 8 + Real.sqrt 58) := 
by
  sorry

end two_numbers_solution_l17_17473


namespace correct_subtraction_l17_17558

theorem correct_subtraction (x : ℕ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end correct_subtraction_l17_17558


namespace ratio_of_selling_prices_l17_17652

theorem ratio_of_selling_prices (C SP1 SP2 : ℝ)
  (h1 : SP1 = C + 0.20 * C)
  (h2 : SP2 = C + 1.40 * C) :
  SP2 / SP1 = 2 := by
  sorry

end ratio_of_selling_prices_l17_17652


namespace sum_of_first_three_cards_l17_17577

theorem sum_of_first_three_cards :
  ∀ (G Y : ℕ → ℕ) (cards : ℕ → ℕ),
  (∀ n, G n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) →
  (∀ n, Y n ∈ ({4, 5, 6, 7, 8} : Set ℕ)) →
  (∀ n, cards (2 * n) = G (cards n) → cards (2 * n + 1) = Y (cards n + 1)) →
  (∀ n, Y n = G (n + 1) ∨ ∃ k, Y n = k * G (n + 1)) →
  (cards 0 + cards 1 + cards 2 = 14) :=
by
  sorry

end sum_of_first_three_cards_l17_17577


namespace max_quarters_l17_17448

theorem max_quarters (a b c : ℕ) (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) : c ≤ 19 :=
sorry

example : ∃ a b c : ℕ, a + b + c = 120 ∧ 5 * a + 10 * b + 25 * c = 1000 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ c = 19 :=
sorry

end max_quarters_l17_17448


namespace lamplighter_monkey_distance_traveled_l17_17492

-- Define the parameters
def running_speed : ℕ := 15
def running_time : ℕ := 5
def swinging_speed : ℕ := 10
def swinging_time : ℕ := 10

-- Define the proof statement
theorem lamplighter_monkey_distance_traveled :
  (running_speed * running_time) + (swinging_speed * swinging_time) = 175 := by
  sorry

end lamplighter_monkey_distance_traveled_l17_17492


namespace prob_within_0_to_80_l17_17084

open MeasureTheory

noncomputable def normal_dist (μ σ : ℝ) : Measure ℝ := measure_theory.measureGaussian μ σ

theorem prob_within_0_to_80 {σ : ℝ} (hσ : 0 < σ)
  (h1 : ∀ x, normal_dist 100 σ (set.Ioc 80 120) = 0.8) :
  normal_dist 100 σ (set.Ioc 0 80) = 0.1 := 
sorry 

end prob_within_0_to_80_l17_17084


namespace quadratic_roots_imaginary_l17_17166

theorem quadratic_roots_imaginary :
  ∃ (a b c : ℝ), a = 1 ∧ b = -2 * Real.sqrt 5 ∧ c = 7 ∧
  (∀ (Δ : ℝ), Δ = b^2 - 4 * a * c → Δ < 0 → ∃ (x1 x2 : ℂ), x1 = Real.sqrt 5 + Complex.i * Real.sqrt 2 ∧ x2 = Real.sqrt 5 - Complex.i * Real.sqrt 2) :=
by
  let a := 1
  let b := -2 * Real.sqrt 5
  let c := 7
  use [a, b, c]
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  intros Δ hΔ
  use Δ
  split 
  { exact hΔ }
  intros hΔ_lt
  have h := hΔ_lt
  -- Proof omitted
  sorry

end quadratic_roots_imaginary_l17_17166


namespace regular_octagon_angle_measure_l17_17836

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l17_17836


namespace percentage_subtraction_l17_17003

theorem percentage_subtraction (P : ℝ) : (700 - (P / 100 * 7000) = 700) → P = 0 :=
by
  sorry

end percentage_subtraction_l17_17003


namespace standard_parts_bounds_l17_17651

noncomputable def n : ℕ := 900
noncomputable def p : ℝ := 0.9
noncomputable def confidence_level : ℝ := 0.95
noncomputable def lower_bound : ℝ := 792
noncomputable def upper_bound : ℝ := 828

theorem standard_parts_bounds : 
  792 ≤ n * p - 1.96 * (n * p * (1 - p)).sqrt ∧ 
  n * p + 1.96 * (n * p * (1 - p)).sqrt ≤ 828 :=
sorry

end standard_parts_bounds_l17_17651


namespace polynomial_distinct_positive_roots_l17_17229

theorem polynomial_distinct_positive_roots (a b : ℝ) (P : ℝ → ℝ) (hP : ∀ x, P x = x^3 + a * x^2 + b * x - 1) 
(hroots : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0) : 
  P (-1) < -8 := 
by
  sorry

end polynomial_distinct_positive_roots_l17_17229


namespace f_g_5_eq_163_l17_17074

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem f_g_5_eq_163 : f (g 5) = 163 := by
  sorry

end f_g_5_eq_163_l17_17074


namespace prod_mod7_eq_zero_l17_17378

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l17_17378


namespace train_length_proof_l17_17643

-- Define the conditions
def train_speed_kmph := 72
def platform_length_m := 290
def crossing_time_s := 26

-- Conversion factor
def kmph_to_mps := 5 / 18

-- Convert speed to m/s
def train_speed_mps := train_speed_kmph * kmph_to_mps

-- Distance covered by train while crossing the platform (in meters)
def distance_covered := train_speed_mps * crossing_time_s

-- Length of the train (in meters)
def train_length := distance_covered - platform_length_m

-- The theorem to be proved
theorem train_length_proof : train_length = 230 :=
by 
  -- proof would be placed here 
  sorry

end train_length_proof_l17_17643


namespace elsa_data_usage_l17_17176

theorem elsa_data_usage (D : ℝ) 
  (h_condition : D - 300 - (2/5) * (D - 300) = 120) : D = 500 := 
sorry

end elsa_data_usage_l17_17176


namespace original_bales_correct_l17_17471

-- Definitions
def total_bales_now : Nat := 54
def bales_stacked_today : Nat := 26
def bales_originally_in_barn : Nat := total_bales_now - bales_stacked_today

-- Theorem statement
theorem original_bales_correct :
  bales_originally_in_barn = 28 :=
by {
  -- We will prove this later
  sorry
}

end original_bales_correct_l17_17471


namespace inequality_problem_l17_17434

-- Define the conditions and the problem statement
theorem inequality_problem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end inequality_problem_l17_17434


namespace shells_put_back_l17_17441

def shells_picked_up : ℝ := 324.0
def shells_left : ℝ := 32.0

theorem shells_put_back : shells_picked_up - shells_left = 292 := by
  sorry

end shells_put_back_l17_17441


namespace find_a_l17_17720

theorem find_a 
  (x y z a : ℤ)
  (h1 : z + a = -2)
  (h2 : y + z = 1)
  (h3 : x + y = 0) : 
  a = -2 := 
  by 
    sorry

end find_a_l17_17720


namespace time_b_started_walking_l17_17160

/-- A's speed is 7 kmph, B's speed is 7.555555555555555 kmph, and B overtakes A after 1.8 hours. -/
theorem time_b_started_walking (t : ℝ) (A_speed : ℝ) (B_speed : ℝ) (overtake_time : ℝ)
    (hA : A_speed = 7) (hB : B_speed = 7.555555555555555) (hOvertake : overtake_time = 1.8) 
    (distance_A : ℝ) (distance_B : ℝ)
    (hDistanceA : distance_A = (t + overtake_time) * A_speed)
    (hDistanceB : distance_B = B_speed * overtake_time) :
  t = 8.57 / 60 := by
  sorry

end time_b_started_walking_l17_17160


namespace find_a_l17_17729

-- Definitions of the conditions
def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

-- The proof goal
theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 := 
by 
  sorry

end find_a_l17_17729


namespace cube_side_length_l17_17427

-- Given definitions and conditions
variables (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

-- Statement of the theorem
theorem cube_side_length (x : ℝ) : 
  ( ∃ (y z : ℝ), 
      y + x + z = c ∧ 
      x + z = c * a / b ∧
      y = c * x / b ∧
      z = c * x / a 
  ) → x = a * b * c / (a * b + b * c + c * a) :=
sorry

end cube_side_length_l17_17427


namespace compute_fraction_l17_17520

theorem compute_fraction (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end compute_fraction_l17_17520


namespace trig_matrix_determinant_l17_17177

noncomputable def trig_matrix (θ φ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [Real.sin θ * Real.cos φ, Real.sin θ * Real.sin φ, Real.cos θ],
    [Real.cos φ, -Real.sin φ, 0],
    [Real.cos θ * Real.cos φ, Real.cos θ * Real.sin φ, Real.sin θ]
  ]

theorem trig_matrix_determinant (θ φ : ℝ) : (trig_matrix θ φ).det = Real.cos θ :=
by
  sorry

end trig_matrix_determinant_l17_17177


namespace count_perfect_cubes_between_10_and_2000_l17_17953

theorem count_perfect_cubes_between_10_and_2000 : 
  (∃ n_min n_max, n_min^3 ≥ 10 ∧ n_max^3 ≤ 2000 ∧ 
  (n_max - n_min + 1 = 10)) := 
sorry

end count_perfect_cubes_between_10_and_2000_l17_17953


namespace find_picture_area_l17_17703

variable (x y : ℕ)
    (h1 : x > 1)
    (h2 : y > 1)
    (h3 : (3 * x + 2) * (y + 4) - x * y = 62)

theorem find_picture_area : x * y = 10 :=
by
  sorry

end find_picture_area_l17_17703


namespace floor_sqrt_80_l17_17923

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l17_17923


namespace brittany_second_test_grade_l17_17374

theorem brittany_second_test_grade
  (first_test_grade second_test_grade : ℕ) 
  (average_after_second : ℕ)
  (h1 : first_test_grade = 78)
  (h2 : average_after_second = 81) 
  (h3 : (first_test_grade + second_test_grade) / 2 = average_after_second) :
  second_test_grade = 84 :=
by
  sorry

end brittany_second_test_grade_l17_17374


namespace can_be_divided_into_6_triangles_l17_17681

-- Define the initial rectangle dimensions
def initial_rectangle_length := 6
def initial_rectangle_width := 5

-- Define the cut-out rectangle dimensions
def cutout_rectangle_length := 2
def cutout_rectangle_width := 1

-- Total area before the cut-out
def total_area : Nat := initial_rectangle_length * initial_rectangle_width

-- Cut-out area
def cutout_area : Nat := cutout_rectangle_length * cutout_rectangle_width

-- Remaining area after the cut-out
def remaining_area : Nat := total_area - cutout_area

-- The statement to be proved
theorem can_be_divided_into_6_triangles :
  remaining_area = 28 → (∃ (triangles : List (Nat × Nat × Nat)), triangles.length = 6) :=
by 
  intros h
  sorry

end can_be_divided_into_6_triangles_l17_17681


namespace no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l17_17672

theorem no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), n^2012 - 1 = 2^k :=
by
  sorry  

end no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l17_17672


namespace problem_1_problem_2_problem_3_l17_17943

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_1 :
  (∀ x : ℝ, f 1 x ≥ f 1 1) :=
by sorry

theorem problem_2 (x e : ℝ) (hx : x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) (hf : f a x = 1) :
  0 ≤ a ∧ a ≤ 1 :=
by sorry

theorem problem_3 (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici 1 → f a x ≥ f a (1 / x)) → 1 ≤ a :=
by sorry

end problem_1_problem_2_problem_3_l17_17943


namespace age_sum_is_27_l17_17341

noncomputable def a : ℕ := 12
noncomputable def b : ℕ := 10
noncomputable def c : ℕ := 5

theorem age_sum_is_27
  (h1: a = b + 2)
  (h2: b = 2 * c)
  (h3: b = 10) :
  a + b + c = 27 :=
  sorry

end age_sum_is_27_l17_17341


namespace part_I_part_II_l17_17065

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x - 1|

theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ 2 - |x - 1|) : 0 ≤ a ∧ a ≤ 4 := 
sorry

theorem part_II (a : ℝ) (h₁ : a < 2) (h₂ : ∀ x : ℝ, f x a ≥ 3) : a = -4 := 
sorry

end part_I_part_II_l17_17065


namespace calc_expression_l17_17038

theorem calc_expression :
  (8^5 / 8^3) * 3^6 = 46656 := by
  sorry

end calc_expression_l17_17038


namespace simplify_fraction_expression_l17_17337

variable (d : ℝ)

theorem simplify_fraction_expression :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := 
by
  sorry

end simplify_fraction_expression_l17_17337


namespace race_outcomes_l17_17501

def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fiona"]

theorem race_outcomes (h : ¬ "Fiona" ∈ ["Abe", "Bobby", "Charles", "Devin", "Edwin"]) : 
  (participants.length - 1) * (participants.length - 2) * (participants.length - 3) = 60 :=
by
  sorry

end race_outcomes_l17_17501


namespace regular_octagon_interior_angle_measure_l17_17779

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l17_17779


namespace sum_of_squares_not_square_l17_17615

theorem sum_of_squares_not_square (a : ℕ) : 
  ¬ ∃ b : ℕ, (a - 1)^2 + a^2 + (a + 1)^2 = b^2 := 
by {
  sorry
}

end sum_of_squares_not_square_l17_17615


namespace distance_covered_l17_17863

-- Define the conditions
def speed_still_water : ℕ := 30   -- 30 kmph
def current_speed : ℕ := 6        -- 6 kmph
def time_downstream : ℕ := 24     -- 24 seconds

-- Proving the distance covered downstream
theorem distance_covered (s_still s_current t : ℕ) (h_s_still : s_still = speed_still_water) (h_s_current : s_current = current_speed) (h_t : t = time_downstream):
  (s_still + s_current) * 1000 / 3600 * t = 240 :=
by sorry

end distance_covered_l17_17863


namespace inequality_holds_for_real_numbers_l17_17266

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l17_17266


namespace absolute_value_inequality_l17_17985

theorem absolute_value_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| :=
by
  sorry

end absolute_value_inequality_l17_17985


namespace smallest_fraction_numerator_l17_17508

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (a * 4 > b * 3) ∧ (a = 73) := 
sorry

end smallest_fraction_numerator_l17_17508


namespace solve_pos_int_a_l17_17677

theorem solve_pos_int_a :
  ∀ a : ℕ, (0 < a) →
  (∀ n : ℕ, (n ≥ 5) → ((2^n - n^2) ∣ (a^n - n^a))) →
  (a = 2 ∨ a = 4) :=
by
  sorry

end solve_pos_int_a_l17_17677


namespace sally_fries_count_l17_17599

theorem sally_fries_count (sally_initial_fries mark_initial_fries : ℕ) 
  (mark_gave_fraction : ℤ) 
  (h_sally_initial : sally_initial_fries = 14) 
  (h_mark_initial : mark_initial_fries = 36) 
  (h_mark_give : mark_gave_fraction = 1 / 3) :
  sally_initial_fries + (mark_initial_fries * mark_gave_fraction).natAbs = 26 :=
by
  sorry

end sally_fries_count_l17_17599


namespace cuboid_surface_area_l17_17143

-- Definitions
def Length := 12  -- meters
def Breadth := 14  -- meters
def Height := 7  -- meters

-- Surface area of a cuboid formula
def surfaceAreaOfCuboid (l b h : Nat) : Nat :=
  2 * (l * b + l * h + b * h)

-- Proof statement
theorem cuboid_surface_area : surfaceAreaOfCuboid Length Breadth Height = 700 := by
  sorry

end cuboid_surface_area_l17_17143


namespace angle_in_second_quadrant_l17_17958

/-- If α is an angle in the first quadrant, then π - α is an angle in the second quadrant -/
theorem angle_in_second_quadrant (α : Real) (h : 0 < α ∧ α < π / 2) : π - α > π / 2 ∧ π - α < π :=
by
  sorry

end angle_in_second_quadrant_l17_17958


namespace percentage_no_job_diploma_l17_17224

def percentage_with_university_diploma {total_population : ℕ} (has_diploma : ℕ) : ℕ :=
  (has_diploma / total_population) * 100

variables {total_population : ℕ} (p_no_diploma_and_job : ℕ) (p_with_job : ℕ) (p_diploma : ℕ)

axiom percentage_no_diploma_job :
  p_no_diploma_and_job = 10

axiom percentage_with_job :
  p_with_job = 40

axiom percentage_diploma :
  p_diploma = 39

theorem percentage_no_job_diploma :
  ∃ p : ℕ, p = (9 / 60) * 100 := sorry

end percentage_no_job_diploma_l17_17224


namespace problem_part1_problem_part2_problem_part3_l17_17064

noncomputable def given_quadratic (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 - (Real.sqrt 3 + 1) * x + m

noncomputable def sin_cos_eq_quadratic_roots (θ m : ℝ) : Prop := 
  let sinθ := Real.sin θ
  let cosθ := Real.cos θ
  given_quadratic sinθ m = 0 ∧ given_quadratic cosθ m = 0

theorem problem_part1 (θ : ℝ) (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  (Real.sin θ / (1 - Real.cos θ) + Real.cos θ / (1 - Real.tan θ)) = (3 + 5 * Real.sqrt 3) / 4 :=
sorry

theorem problem_part2 {θ : ℝ} (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  m = Real.sqrt 3 / 4 :=
sorry

theorem problem_part3 (m : ℝ) (sinθ1 cosθ1 sinθ2 cosθ2 : ℝ) (θ1 θ2 : ℝ)
  (H1 : sinθ1 = Real.sqrt 3 / 2 ∧ cosθ1 = 1 / 2 ∧ θ1 = Real.pi / 3)
  (H2 : sinθ2 = 1 / 2 ∧ cosθ2 = Real.sqrt 3 / 2 ∧ θ2 = Real.pi / 6) : 
  ∃ θ, sin_cos_eq_quadratic_roots θ m ∧ 
       (Real.sin θ = sinθ1 ∧ Real.cos θ = cosθ1 ∨ Real.sin θ = sinθ2 ∧ Real.cos θ = cosθ2) :=
sorry

end problem_part1_problem_part2_problem_part3_l17_17064


namespace area_of_circle_l17_17130

theorem area_of_circle :
  (∃ (x y : ℝ), x^2 + y^2 - 6 * x + 8 * y = -9) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  -- We need to prove the area is 16π
  sorry

end area_of_circle_l17_17130


namespace shepherd_flock_l17_17871

theorem shepherd_flock (x y : ℕ) (h1 : (x - 1) * 5 = 7 * y) (h2 : x * 3 = 5 * (y - 1)) :
  x + y = 25 :=
sorry

end shepherd_flock_l17_17871


namespace chord_length_of_circle_intersected_by_line_l17_17312

open Real

-- Definitions for the conditions given in the problem
def line_eqn (x y : ℝ) : Prop := x - y - 1 = 0
def circle_eqn (x y : ℝ) : Prop := x^2 - 4 * x + y^2 = 4

-- The proof statement (problem) in Lean 4
theorem chord_length_of_circle_intersected_by_line :
  ∀ (x y : ℝ), circle_eqn x y → line_eqn x y → ∃ L : ℝ, L = sqrt 17 := by
  sorry

end chord_length_of_circle_intersected_by_line_l17_17312


namespace area_increase_factor_l17_17924

theorem area_increase_factor (s : ℝ) :
  let A_original := s^2
  let A_new := (3 * s)^2
  A_new / A_original = 9 := by
  sorry

end area_increase_factor_l17_17924


namespace well_diameter_l17_17018

noncomputable def calculateDiameter (volume depth : ℝ) : ℝ :=
  2 * Real.sqrt (volume / (Real.pi * depth))

theorem well_diameter :
  calculateDiameter 678.5840131753953 24 = 6 :=
by
  sorry

end well_diameter_l17_17018


namespace floor_neg_7_over_4_l17_17891

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l17_17891


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17825

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17825


namespace phillip_spent_on_oranges_l17_17239

theorem phillip_spent_on_oranges 
  (M : ℕ) (A : ℕ) (C : ℕ) (L : ℕ) (O : ℕ)
  (hM : M = 95) (hA : A = 25) (hC : C = 6) (hL : L = 50)
  (h_total_spending : O + A + C = M - L) : 
  O = 14 := 
sorry

end phillip_spent_on_oranges_l17_17239


namespace target_hit_prob_l17_17027

-- Probability definitions for A, B, and C
def prob_A := 1 / 2
def prob_B := 1 / 3
def prob_C := 1 / 4

-- Theorem to prove the probability of the target being hit
theorem target_hit_prob :
  (1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)) = 3 / 4 :=
by
  sorry

end target_hit_prob_l17_17027


namespace cube_root_simplification_l17_17635

theorem cube_root_simplification : 
  ∃ (a b : ℕ), (∃ c : ℕ, 8000 = c ^ 3) ∧ a * b ^ c = 8000 ∧ b = 1 ∧ a + b = 21 :=
by
  use 20
  use 1
  use 20
  sorry

end cube_root_simplification_l17_17635


namespace JackOfHeartsIsSane_l17_17884

inductive Card
  | Ace
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | JackOfHearts

open Card

def Sane (c : Card) : Prop := sorry

axiom Condition1 : Sane Three → ¬ Sane Ace
axiom Condition2 : Sane Four → (¬ Sane Three ∨ ¬ Sane Two)
axiom Condition3 : Sane Five → (Sane Ace ↔ Sane Four)
axiom Condition4 : Sane Six → (Sane Ace ∧ Sane Two)
axiom Condition5 : Sane Seven → ¬ Sane Five
axiom Condition6 : Sane JackOfHearts → (¬ Sane Six ∨ ¬ Sane Seven)

theorem JackOfHeartsIsSane : Sane JackOfHearts := by
  sorry

end JackOfHeartsIsSane_l17_17884


namespace rectangle_area_l17_17153

-- Define the rectangular properties
variables {w l d x : ℝ}
def width (w : ℝ) : ℝ := w
def length (w : ℝ) : ℝ := 3 * w
def diagonal (w : ℝ) : ℝ := x

theorem rectangle_area (w x : ℝ) (hw : w ^ 2 + (3 * w) ^ 2 = x ^ 2) : w * 3 * w = 3 / 10 * x ^ 2 :=
by 
  sorry

end rectangle_area_l17_17153


namespace complex_number_real_l17_17485

theorem complex_number_real (m : ℝ) (z : ℂ) 
  (h1 : z = ⟨1 / (m + 5), 0⟩ + ⟨0, m^2 + 2 * m - 15⟩)
  (h2 : m^2 + 2 * m - 15 = 0)
  (h3 : m ≠ -5) :
  m = 3 :=
sorry

end complex_number_real_l17_17485


namespace tan_identity_l17_17566

open Real

-- Definition of conditions
def isPureImaginary (z : Complex) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem tan_identity (theta : ℝ) :
  isPureImaginary ((cos theta - 4/5) + (sin theta - 3/5) * Complex.I) →
  tan (theta - π / 4) = -7 :=
by
  sorry

end tan_identity_l17_17566


namespace coffee_cost_per_week_l17_17089

def num_people: ℕ := 4
def cups_per_person_per_day: ℕ := 2
def ounces_per_cup: ℝ := 0.5
def cost_per_ounce: ℝ := 1.25

theorem coffee_cost_per_week : 
  (num_people * cups_per_person_per_day * ounces_per_cup * 7 * cost_per_ounce) = 35 :=
by
  sorry

end coffee_cost_per_week_l17_17089


namespace luke_games_l17_17737

theorem luke_games (F G : ℕ) (H1 : G = 2) (H2 : F + G - 2 = 2) : F = 2 := by
  sorry

end luke_games_l17_17737


namespace final_mark_is_correct_l17_17875

def term_mark : ℝ := 80
def term_weight : ℝ := 0.70
def exam_mark : ℝ := 90
def exam_weight : ℝ := 0.30

theorem final_mark_is_correct :
  (term_mark * term_weight + exam_mark * exam_weight) = 83 :=
by
  sorry

end final_mark_is_correct_l17_17875


namespace simplify_cbrt_8000_eq_21_l17_17633

theorem simplify_cbrt_8000_eq_21 :
  ∃ (a b : ℕ), a * (b^(1/3)) = 20 * (1^(1/3)) ∧ b = 1 ∧ a + b = 21 :=
by
  sorry

end simplify_cbrt_8000_eq_21_l17_17633


namespace percentage_cross_pollinated_l17_17493

-- Definitions and known conditions:
variables (F C T : ℕ)
variables (h1 : F + C = 221)
variables (h2 : F = 3 * T / 4)
variables (h3 : T = F + 39 + C)

-- Theorem statement for the percentage of cross-pollinated trees
theorem percentage_cross_pollinated : ((C : ℚ) / T) * 100 = 10 :=
by sorry

end percentage_cross_pollinated_l17_17493


namespace floor_sqrt_80_l17_17922

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l17_17922


namespace cos_210_eq_neg_sqrt3_over_2_l17_17516

theorem cos_210_eq_neg_sqrt3_over_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_over_2_l17_17516


namespace discount_for_multiple_rides_l17_17637

-- Definitions based on given conditions
def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def coupon_value : ℝ := 1.0
def total_tickets_needed : ℝ := 7.0

-- The proof problem
theorem discount_for_multiple_rides : 
  (ferris_wheel_cost + roller_coaster_cost) - (total_tickets_needed - coupon_value) = 2.0 :=
by
  sorry

end discount_for_multiple_rides_l17_17637


namespace jogging_track_circumference_l17_17610

/-- 
Given:
- Deepak's speed = 20 km/hr
- His wife's speed = 12 km/hr
- They meet for the first time in 32 minutes

Then:
The circumference of the jogging track is 17.0667 km.
-/
theorem jogging_track_circumference (deepak_speed : ℝ) (wife_speed : ℝ) (meet_time : ℝ)
  (h1 : deepak_speed = 20)
  (h2 : wife_speed = 12)
  (h3 : meet_time = (32 / 60) ) : 
  ∃ circumference : ℝ, circumference = 17.0667 :=
by
  sorry

end jogging_track_circumference_l17_17610


namespace dans_average_rate_l17_17489

/-- Dan's average rate for the entire trip, given the conditions, equals 0.125 miles per minute --/
theorem dans_average_rate :
  ∀ (d_run d_swim : ℝ) (r_run r_swim : ℝ) (time_run time_swim : ℝ),
  d_run = 3 ∧ d_swim = 3 ∧ r_run = 10 ∧ r_swim = 6 ∧ 
  time_run = (d_run / r_run) * 60 ∧ time_swim = (d_swim / r_swim) * 60 →
  ((d_run + d_swim) / (time_run + time_swim)) = 0.125 :=
by
  intros d_run d_swim r_run r_swim time_run time_swim h
  sorry

end dans_average_rate_l17_17489


namespace proof_problem_l17_17664

-- Define complex numbers for the roots
def P (x : ℂ) := ∏ k in Finset.range 15, (x - complex.exp (2 * real.pi * k * complex.I / 17))

def Q (x : ℂ) := ∏ j in Finset.range 12, (x - complex.exp (2 * real.pi * j * complex.I / 13))

-- Conditions as Lean definitions
noncomputable def e_k (k : ℕ) (h : k < 16) : ℂ := complex.exp (2 * real.pi * k * complex.I / 17)
noncomputable def e_j (j : ℕ) (h : j < 13) : ℂ := complex.exp (2 * real.pi * j * complex.I / 13)

theorem proof_problem : 
  (∏ k in Finset.range 15, ∏ j in Finset.range 12, (e_j j (by linarith) - e_k k (by linarith))) = 1 :=
sorry

end proof_problem_l17_17664


namespace floor_sqrt_80_eq_8_l17_17907

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l17_17907


namespace password_decryption_probability_l17_17026

theorem password_decryption_probability :
  let A := (1:ℚ)/5
  let B := (1:ℚ)/3
  let C := (1:ℚ)/4
  let P_decrypt := 1 - (1 - A) * (1 - B) * (1 - C)
  P_decrypt = 3/5 := 
  by
    -- Calculations and logic will be provided here
    sorry

end password_decryption_probability_l17_17026


namespace min_value_of_x_plus_y_l17_17544

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)  
  (h : 19 / x + 98 / y = 1) : x + y ≥ 203 :=
sorry

end min_value_of_x_plus_y_l17_17544


namespace final_solution_concentration_l17_17347

def concentration (mass : ℕ) (volume : ℕ) : ℕ := 
  (mass * 100) / volume

theorem final_solution_concentration :
  let volume1 := 4
  let conc1 := 4 -- percentage
  let volume2 := 2
  let conc2 := 10 -- percentage
  let mass1 := volume1 * conc1 / 100
  let mass2 := volume2 * conc2 / 100
  let total_mass := mass1 + mass2
  let total_volume := volume1 + volume2
  concentration total_mass total_volume = 6 :=
by
  sorry

end final_solution_concentration_l17_17347


namespace ratio_addition_l17_17960

theorem ratio_addition (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := 
by sorry

end ratio_addition_l17_17960


namespace polygon_sides_l17_17019

theorem polygon_sides (n : ℕ) (a1 d : ℝ) (h1 : a1 = 100) (h2 : d = 10)
  (h3 : ∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d < 180) : n = 8 :=
by
  sorry

end polygon_sides_l17_17019


namespace swim_distance_downstream_l17_17647

theorem swim_distance_downstream 
  (V_m V_s : ℕ) 
  (t d : ℕ) 
  (h1 : V_m = 9) 
  (h2 : t = 3) 
  (h3 : 3 * (V_m - V_s) = 18) : 
  t * (V_m + V_s) = 36 := 
by 
  sorry

end swim_distance_downstream_l17_17647


namespace interior_angle_of_regular_octagon_l17_17783

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l17_17783


namespace polynomial_roots_property_l17_17732

theorem polynomial_roots_property (a b : ℝ) (h : ∀ x, x^2 + x - 2024 = 0 → x = a ∨ x = b) : 
  a^2 + 2 * a + b = 2023 :=
by
  sorry

end polynomial_roots_property_l17_17732


namespace find_a_plus_b_l17_17937

theorem find_a_plus_b (a b : ℝ) (x y : ℝ) 
  (h1 : x = 2) 
  (h2 : y = -1) 
  (h3 : a * x - 2 * y = 4) 
  (h4 : 3 * x + b * y = -7) : a + b = 14 := 
by 
  -- Begin the proof
  sorry

end find_a_plus_b_l17_17937


namespace half_angle_quadrant_l17_17192

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end half_angle_quadrant_l17_17192


namespace books_count_is_8_l17_17714

theorem books_count_is_8
  (k a p_k p_a : ℕ)
  (h1 : k = a + 6)
  (h2 : k * p_k = 1056)
  (h3 : a * p_a = 56)
  (h4 : p_k > p_a + 100) :
  k = 8 := 
sorry

end books_count_is_8_l17_17714


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17831

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17831


namespace carson_gold_stars_yesterday_l17_17165

def goldStarsEarnedYesterday (total: ℕ) (earnedToday: ℕ) : ℕ :=
  total - earnedToday

theorem carson_gold_stars_yesterday :
  goldStarsEarnedYesterday 15 9 = 6 :=
by 
  sorry

end carson_gold_stars_yesterday_l17_17165


namespace number_of_tricycles_l17_17623

def num_bicycles : Nat := 24
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3
def total_wheels : Nat := 90

theorem number_of_tricycles : ∃ T : Nat, (wheels_per_bicycle * num_bicycles) + (wheels_per_tricycle * T) = total_wheels ∧ T = 14 := by
  sorry

end number_of_tricycles_l17_17623


namespace gift_spending_l17_17069

def total_amount : ℝ := 700.00
def wrapping_expenses : ℝ := 139.00
def amount_spent_on_gifts : ℝ := 700.00 - 139.00

theorem gift_spending :
  (total_amount - wrapping_expenses) = 561.00 :=
by
  sorry

end gift_spending_l17_17069


namespace inequality_xyz_l17_17274

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l17_17274


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17788

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17788


namespace value_of_x_l17_17076

theorem value_of_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 :=
by
  sorry

end value_of_x_l17_17076


namespace find_n_l17_17698

theorem find_n (x y n : ℝ) (h1 : 2 * x - 5 * y = 3 * n + 7) (h2 : x - 3 * y = 4) 
  (h3 : x = y):
  n = -1 / 3 := 
by 
  sorry

end find_n_l17_17698


namespace jason_daily_charge_l17_17578

theorem jason_daily_charge 
  (total_cost_eric : ℕ) (days_eric : ℕ) (daily_charge : ℕ)
  (h1 : total_cost_eric = 800) (h2 : days_eric = 20)
  (h3 : daily_charge = total_cost_eric / days_eric) :
  daily_charge = 40 := 
by
  sorry

end jason_daily_charge_l17_17578


namespace positive_integer_triples_satisfying_conditions_l17_17527

theorem positive_integer_triples_satisfying_conditions :
  ∀ (a b c : ℕ), a^2 + b^2 + c^2 = 2005 ∧ a ≤ b ∧ b ≤ c →
  (a, b, c) = (23, 24, 30) ∨
  (a, b, c) = (12, 30, 31) ∨
  (a, b, c) = (9, 30, 32) ∨
  (a, b, c) = (4, 30, 33) ∨
  (a, b, c) = (15, 22, 36) ∨
  (a, b, c) = (9, 18, 40) ∨
  (a, b, c) = (4, 15, 42) :=
sorry

end positive_integer_triples_satisfying_conditions_l17_17527


namespace count_valid_m_l17_17298

def is_divisor (a b : ℕ) : Prop := b % a = 0

def valid_m (m : ℕ) : Prop :=
  m > 1 ∧ is_divisor m 480 ∧ (480 / m) > 1

theorem count_valid_m : (∃ m, valid_m m) → Nat.card {m // valid_m m} = 22 :=
by sorry

end count_valid_m_l17_17298


namespace polynomial_expansion_a5_l17_17422

theorem polynomial_expansion_a5 :
  (x - 1) ^ 8 = (1 : ℤ) + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 →
  a₅ = -56 :=
by
  intro h
  -- The proof is omitted.
  sorry

end polynomial_expansion_a5_l17_17422


namespace fg_of_5_eq_163_l17_17072

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem fg_of_5_eq_163 : f (g 5) = 163 :=
by
  sorry

end fg_of_5_eq_163_l17_17072


namespace sum_of_coefficients_l17_17671

theorem sum_of_coefficients (x y : ℝ) : 
  (2 * x - 3 * y) ^ 9 = -1 :=
by
  sorry

end sum_of_coefficients_l17_17671


namespace prove_fraction_l17_17442

noncomputable def michael_brothers_problem (M O Y : ℕ) :=
  Y = 5 ∧
  M + O + Y = 28 ∧
  O = 2 * (M - 1) + 1 →
  Y / O = 1 / 3

theorem prove_fraction (M O Y : ℕ) : michael_brothers_problem M O Y :=
  sorry

end prove_fraction_l17_17442


namespace a_sufficient_but_not_necessary_l17_17345

theorem a_sufficient_but_not_necessary (a : ℝ) : 
  (a = 1 → |a| = 1) ∧ (¬ (|a| = 1 → a = 1)) :=
by 
  sorry

end a_sufficient_but_not_necessary_l17_17345


namespace f_g_5_eq_163_l17_17073

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem f_g_5_eq_163 : f (g 5) = 163 := by
  sorry

end f_g_5_eq_163_l17_17073


namespace trig_identity_l17_17054

open Real

theorem trig_identity (α β : ℝ) (h : cos α * cos β - sin α * sin β = 0) : sin α * cos β + cos α * sin β = 1 ∨ sin α * cos β + cos α * sin β = -1 :=
by
  sorry

end trig_identity_l17_17054


namespace greatest_integer_with_gcd_6_l17_17475

theorem greatest_integer_with_gcd_6 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 18 = 6) : n = 138 :=
sorry

end greatest_integer_with_gcd_6_l17_17475


namespace line_exists_symmetric_diagonals_l17_17687

-- Define the initial conditions
def Circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 6 * x = 0
def Line_l1 (x y : ℝ) : Prop := y = 2 * x + 1

-- Define the symmetric circle C about the line l1
def Symmetric_Circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the origion and intersection points
def Point_O : (ℝ × ℝ) := (0, 0)
def Point_Intersection (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop := ∃ x_A y_A x_B y_B : ℝ,
  l x_A = y_A ∧ l x_B = y_B ∧ Symmetric_Circle x_A y_A ∧ Symmetric_Circle x_B y_B

-- Define diagonal equality condition
def Diagonals_Equal (O A S B : ℝ × ℝ) : Prop := 
  let (xO, yO) := O
  let (xA, yA) := A
  let (xS, yS) := S
  let (xB, yB) := B
  (xA - xO)^2 + (yA - yO)^2 = (xB - xS)^2 + (yB - yS)^2

-- Prove existence of line where diagonals are equal and find the equation
theorem line_exists_symmetric_diagonals :
  ∃ l : ℝ → ℝ, (l (-1) = 0) ∧
    (∃ (A B S : ℝ × ℝ), Point_Intersection l A B ∧ Diagonals_Equal Point_O A S B) ∧
    (∀ x : ℝ, l x = x + 1) :=
by
  sorry

end line_exists_symmetric_diagonals_l17_17687


namespace probability_of_answering_phone_in_4_rings_l17_17353

/-- A proof statement that asserts the probability of answering the phone within the first four rings is equal to 9/10. -/
theorem probability_of_answering_phone_in_4_rings :
  (1/10) + (3/10) + (2/5) + (1/10) = 9/10 :=
by
  sorry

end probability_of_answering_phone_in_4_rings_l17_17353


namespace inequality_holds_for_all_reals_l17_17291

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l17_17291


namespace expression_eq_one_if_and_only_if_k_eq_one_l17_17438

noncomputable def expression (a b c k : ℝ) :=
  (k * a^2 * b^2 + a^2 * c^2 + b^2 * c^2) /
  ((a^2 - b * c) * (b^2 - a * c) + (a^2 - b * c) * (c^2 - a * b) + (b^2 - a * c) * (c^2 - a * b))

theorem expression_eq_one_if_and_only_if_k_eq_one
  (a b c k : ℝ) (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0) :
  expression a b c k = 1 ↔ k = 1 :=
by
  sorry

end expression_eq_one_if_and_only_if_k_eq_one_l17_17438


namespace fraction_of_boxes_loaded_by_day_crew_l17_17514

theorem fraction_of_boxes_loaded_by_day_crew
    (dayCrewBoxesPerWorker : ℚ)
    (dayCrewWorkers : ℚ)
    (nightCrewBoxesPerWorker : ℚ := (3 / 4) * dayCrewBoxesPerWorker)
    (nightCrewWorkers : ℚ := (3 / 4) * dayCrewWorkers) :
    (dayCrewBoxesPerWorker * dayCrewWorkers) / ((dayCrewBoxesPerWorker * dayCrewWorkers) + (nightCrewBoxesPerWorker * nightCrewWorkers)) = 16 / 25 :=
by
  sorry

end fraction_of_boxes_loaded_by_day_crew_l17_17514


namespace apples_total_l17_17515

theorem apples_total (Benny_picked Dan_picked : ℕ) (hB : Benny_picked = 2) (hD : Dan_picked = 9) : Benny_picked + Dan_picked = 11 :=
by
  -- Definitions
  sorry

end apples_total_l17_17515


namespace required_run_rate_equivalence_l17_17719

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.5
def overs_first_phase : ℝ := 10
def total_target_runs : ℝ := 350
def remaining_overs : ℝ := 35
def total_overs : ℝ := 45

-- Define the already scored runs
def runs_scored_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_phase

-- Define the required runs for the remaining overs
def runs_needed : ℝ := total_target_runs - runs_scored_first_10_overs

-- Theorem stating the required run rate in the remaining 35 overs
theorem required_run_rate_equivalence :
  runs_needed / remaining_overs = 9 :=
by
  sorry

end required_run_rate_equivalence_l17_17719


namespace lizas_final_balance_l17_17591

-- Define the initial condition and subsequent changes
def initial_balance : ℕ := 800
def rent_payment : ℕ := 450
def paycheck_deposit : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

-- Calculate the final balance
def final_balance : ℕ :=
  let balance_after_rent := initial_balance - rent_payment
  let balance_after_paycheck := balance_after_rent + paycheck_deposit
  let balance_after_bills := balance_after_paycheck - (electricity_bill + internet_bill)
  balance_after_bills - phone_bill

-- Theorem to prove that the final balance is 1563
theorem lizas_final_balance : final_balance = 1563 :=
by
  sorry

end lizas_final_balance_l17_17591


namespace new_volume_of_balloon_l17_17859

def initial_volume : ℝ := 2.00  -- Initial volume in liters
def initial_pressure : ℝ := 745  -- Initial pressure in mmHg
def initial_temperature : ℝ := 293.15  -- Initial temperature in Kelvin
def final_pressure : ℝ := 700  -- Final pressure in mmHg
def final_temperature : ℝ := 283.15  -- Final temperature in Kelvin
def final_volume : ℝ := 2.06  -- Expected final volume in liters

theorem new_volume_of_balloon :
  (initial_pressure * initial_volume / initial_temperature) = (final_pressure * final_volume / final_temperature) :=
  sorry  -- Proof to be filled in later

end new_volume_of_balloon_l17_17859


namespace elena_pen_cost_l17_17394

theorem elena_pen_cost (cost_X : ℝ) (cost_Y : ℝ) (total_pens : ℕ) (brand_X_pens : ℕ) 
    (purchased_X_cost : cost_X = 4.0) (purchased_Y_cost : cost_Y = 2.8)
    (total_pens_condition : total_pens = 12) (brand_X_pens_condition : brand_X_pens = 8) :
    cost_X * brand_X_pens + cost_Y * (total_pens - brand_X_pens) = 43.20 :=
    sorry

end elena_pen_cost_l17_17394


namespace curve_is_line_l17_17170

def curve_theta (theta : ℝ) : Prop :=
  theta = Real.pi / 4

theorem curve_is_line : curve_theta θ → (curve_type = "line") :=
by
  intros h
  cases h
  -- This is where the proof would go, but we'll use a placeholder for now.
  -- The essence of the proof will show that all points making an angle of π/4 with the x-axis lie on a line.
  exact sorry

end curve_is_line_l17_17170


namespace multiply_polynomials_l17_17104

theorem multiply_polynomials (x : ℝ) :
  (x^4 + 8 * x^2 + 64) * (x^2 - 8) = x^4 + 16 * x^2 :=
by
  sorry

end multiply_polynomials_l17_17104


namespace solve_for_x_l17_17777

-- Define the custom operation for real numbers
def custom_op (a b c d : ℝ) : ℝ := a * c - b * d

-- The theorem to prove
theorem solve_for_x (x : ℝ) (h : custom_op (-x) 3 (x - 2) (-6) = 10) :
  x = 4 ∨ x = -2 :=
sorry

end solve_for_x_l17_17777


namespace henry_added_water_l17_17951

theorem henry_added_water (F : ℕ) (h2 : F = 32) (α β : ℚ) (h3 : α = 3/4) (h4 : β = 7/8) :
  (F * β) - (F * α) = 4 := by
  sorry

end henry_added_water_l17_17951


namespace values_of_z_l17_17533

theorem values_of_z (z : ℤ) (hz : 0 < z) :
  (z^2 - 50 * z + 550 ≤ 10) ↔ (20 ≤ z ∧ z ≤ 30) := sorry

end values_of_z_l17_17533


namespace children_exceed_bridge_limit_l17_17157

theorem children_exceed_bridge_limit :
  ∀ (Kelly_weight : ℕ) (Megan_weight : ℕ) (Mike_weight : ℕ),
  Kelly_weight = 34 ∧
  Kelly_weight = (85 * Megan_weight) / 100 ∧
  Mike_weight = Megan_weight + 5 →
  Kelly_weight + Megan_weight + Mike_weight - 100 = 19 :=
by sorry

end children_exceed_bridge_limit_l17_17157


namespace complement_A_in_U_l17_17203

open Set

-- Definitions for sets
def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

-- The proof goal: prove that the complement of A in U is {4}
theorem complement_A_in_U : (U \ A) = {4} := by
  sorry

end complement_A_in_U_l17_17203


namespace find_fixed_monthly_fee_l17_17031

noncomputable def fixed_monthly_fee (f h : ℝ) (february_bill march_bill : ℝ) : Prop :=
  (f + h = february_bill) ∧ (f + 3 * h = march_bill)

theorem find_fixed_monthly_fee (h : ℝ):
  fixed_monthly_fee 13.44 h 20.72 35.28 :=
by 
  sorry

end find_fixed_monthly_fee_l17_17031


namespace exposed_surface_area_hemisphere_l17_17645

-- Given conditions
def radius : ℝ := 10
def height_above_liquid : ℝ := 5

-- The attempt to state the problem as a proposition
theorem exposed_surface_area_hemisphere : 
  (π * radius ^ 2) + (π * radius * height_above_liquid) = 200 * π :=
by
  sorry

end exposed_surface_area_hemisphere_l17_17645


namespace luke_games_l17_17738

variables (F G : ℕ)

theorem luke_games (G_eq_2 : G = 2) (total_games : F + G - 2 = 2) : F = 2 :=
by
  rw [G_eq_2] at total_games
  simp at total_games
  exact total_games

-- sorry

end luke_games_l17_17738


namespace two_digit_multiples_of_4_and_9_l17_17954

theorem two_digit_multiples_of_4_and_9 :
  ∃ (count : ℕ), 
    (∀ (n : ℕ), (10 ≤ n ∧ n ≤ 99) → (n % 4 = 0 ∧ n % 9 = 0) → (n = 36 ∨ n = 72)) ∧ count = 2 :=
by
  sorry

end two_digit_multiples_of_4_and_9_l17_17954


namespace volume_and_surface_area_of_inscribed_sphere_l17_17197

theorem volume_and_surface_area_of_inscribed_sphere (edge_length : ℝ) (h_edge : edge_length = 10) :
    let r := edge_length / 2
    let V := (4 / 3) * π * r^3
    let A := 4 * π * r^2
    V = (500 / 3) * π ∧ A = 100 * π := 
by
  sorry

end volume_and_surface_area_of_inscribed_sphere_l17_17197


namespace regular_octagon_interior_angle_l17_17821

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l17_17821


namespace angle_division_quadrant_l17_17195

variable (k : ℤ)
variable (α : ℝ)
variable (h : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi)

theorem angle_division_quadrant 
  (hα_sec_quadrant : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi) : 
  (∃ m : ℤ, (m = 0 ∧ Real.pi / 4 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
             (m = 1 ∧ Real.pi * (2 * m + 1) / 4 < α / 2 ∧ α / 2 < Real.pi * (2 * m + 1) / 2)) :=
sorry

end angle_division_quadrant_l17_17195


namespace count_measures_of_angle_A_l17_17315

theorem count_measures_of_angle_A :
  ∃ n : ℕ, n = 17 ∧
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ A + B = 180 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (∀ (A' B' : ℕ), A' > 0 ∧ B' > 0 ∧ A' + B' = 180 ∧ (∀ k : ℕ, k ≥ 1 ∧ A' = k * B') → n = 17) :=
sorry

end count_measures_of_angle_A_l17_17315


namespace prime_is_good_iff_not_2_l17_17365

open Nat

def is_good_prime (p : ℕ) [fact p.prime] : Prop :=
  (∃ k > 1, ∃ (n : Fin k → ℕ), (∀ i : Fin k, n i ≥ (p+1)/2) ∧ (∀ i : Fin k, (p^(n i) - 1) % n ((i+1) % k) = 0 ∧ Nat.coprime ((p^(n i) - 1) / n ((i+1) % k)) (n ((i+1) % k))))

theorem prime_is_good_iff_not_2 (p : ℕ) [fact p.prime] : is_good_prime p ↔ p ≠ 2 := 
by {
  sorry
}

end prime_is_good_iff_not_2_l17_17365


namespace algebraic_expression_domain_l17_17565

theorem algebraic_expression_domain (x : ℝ) : (∃ y : ℝ, y = 1 / (x + 2)) ↔ (x ≠ -2) := 
sorry

end algebraic_expression_domain_l17_17565


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17829

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17829


namespace regular_octagon_interior_angle_measure_l17_17781

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l17_17781


namespace probability_of_two_specific_suits_l17_17421

noncomputable def probability_two_suits (suit1: Fin 4) (suit2: Fin 4) : ℝ :=
  let p_each := 1 / 4
  in p_each ^ 6

theorem probability_of_two_specific_suits 
  (suit1: Fin 4) (suit2: Fin 4) (h_suit1: suit1 ≠ suit2) :
  probability_two_suits suit1 suit2 = 1 / 4096 :=
by
  sorry

end probability_of_two_specific_suits_l17_17421


namespace algebraic_identity_l17_17630

theorem algebraic_identity (a b : ℕ) (h1 : a = 753) (h2 : b = 247)
  (identity : ∀ a b, (a^2 + b^2 - a * b) / (a^3 + b^3) = 1 / (a + b)) : 
  (753^2 + 247^2 - 753 * 247) / (753^3 + 247^3) = 0.001 := 
by
  sorry

end algebraic_identity_l17_17630


namespace average_age_of_adults_l17_17767

theorem average_age_of_adults (n_total n_girls n_boys n_adults : ℕ) 
                              (avg_age_total avg_age_girls avg_age_boys avg_age_adults : ℕ)
                              (h1 : n_total = 60)
                              (h2 : avg_age_total = 18)
                              (h3 : n_girls = 30)
                              (h4 : avg_age_girls = 16)
                              (h5 : n_boys = 20)
                              (h6 : avg_age_boys = 17)
                              (h7 : n_adults = 10) :
                              avg_age_adults = 26 :=
sorry

end average_age_of_adults_l17_17767


namespace expected_digits_die_l17_17350

noncomputable def expected_number_of_digits (numbers : List ℕ) : ℚ :=
  let one_digit_numbers := numbers.filter (λ n => n < 10)
  let two_digit_numbers := numbers.filter (λ n => n >= 10)
  let p_one_digit := (one_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  let p_two_digit := (two_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  p_one_digit * 1 + p_two_digit * 2

theorem expected_digits_die :
  expected_number_of_digits [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] = 1.5833 := 
by
  sorry

end expected_digits_die_l17_17350


namespace number_of_numbers_is_ten_l17_17455

open Nat

-- Define the conditions as given
variable (n : ℕ) -- Total number of numbers
variable (incorrect_average correct_average incorrect_value correct_value : ℤ)
variable (h1 : incorrect_average = 16)
variable (h2 : correct_average = 17)
variable (h3 : incorrect_value = 25)
variable (h4 : correct_value = 35)

-- Define the proof problem
theorem number_of_numbers_is_ten
  (h1 : incorrect_average = 16)
  (h2 : correct_average = 17)
  (h3 : incorrect_value = 25)
  (h4 : correct_value = 35)
  (h5 : ∀ (x : ℤ), x ≠ incorrect_value → incorrect_average * (n : ℤ) + x = correct_average * (n : ℤ) + correct_value - incorrect_value)
  : n = 10 := 
sorry

end number_of_numbers_is_ten_l17_17455


namespace greatest_int_with_gcd_18_is_138_l17_17477

theorem greatest_int_with_gcd_18_is_138 :
  ∃ n : ℕ, n < 150 ∧ int.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ int.gcd m 18 = 6 → m ≤ n := by
  sorry

end greatest_int_with_gcd_18_is_138_l17_17477


namespace drum_oil_capacity_l17_17524

theorem drum_oil_capacity (C : ℝ) (Y : ℝ) 
  (hX : DrumX_Oil = 0.5 * C) 
  (hY : DrumY_Cap = 2 * C) 
  (hY_filled : Y + 0.5 * C = 0.65 * (2 * C)) :
  Y = 0.8 * C :=
by
  sorry

end drum_oil_capacity_l17_17524


namespace area_of_sector_l17_17218

theorem area_of_sector (L θ : ℝ) (hL : L = 4) (hθ : θ = 2) : 
  (1 / 2) * ((L / θ) ^ 2) * θ = 4 := by
  sorry

end area_of_sector_l17_17218


namespace sum_first_2017_terms_l17_17319

-- Given sequence definition
def a : ℕ → ℕ
| 0       => 0 -- a_0 (dummy term for 1-based index convenience)
| 1       => 1
| (n + 2) => 3 * 2^(n) - a (n + 1)

-- Sum of the first n terms of the sequence {a_n}
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + a (n + 1)

-- Theorem to prove
theorem sum_first_2017_terms : S 2017 = 2^2017 - 1 :=
sorry

end sum_first_2017_terms_l17_17319


namespace complex_fraction_simplification_l17_17602

theorem complex_fraction_simplification :
  (4 + 7 * complex.i) / (4 - 7 * complex.i) + (4 - 7 * complex.i) / (4 + 7 * complex.i) = -66 / 65 :=
by
  sorry

end complex_fraction_simplification_l17_17602


namespace total_oranges_and_weight_l17_17174

theorem total_oranges_and_weight 
  (oranges_per_child : ℕ) (num_children : ℕ) (average_weight_per_orange : ℝ)
  (h1 : oranges_per_child = 3)
  (h2 : num_children = 4)
  (h3 : average_weight_per_orange = 0.3) :
  oranges_per_child * num_children = 12 ∧ (oranges_per_child * num_children : ℝ) * average_weight_per_orange = 3.6 :=
by
  sorry

end total_oranges_and_weight_l17_17174


namespace max_months_to_build_l17_17762

theorem max_months_to_build (a b c x : ℝ) (h1 : 1/a + 1/b = 1/6)
                            (h2 : 1/a + 1/c = 1/5)
                            (h3 : 1/c + 1/b = 1/4)
                            (h4 : (1/a + 1/b + 1/c) * x = 1) :
                            x = 4 :=
sorry

end max_months_to_build_l17_17762


namespace tank_fill_time_l17_17147

noncomputable def fill_time (T rA rB rC : ℝ) : ℝ :=
  let cycle_fill := rA + rB + rC
  let cycles := T / cycle_fill
  let cycle_time := 3
  cycles * cycle_time

theorem tank_fill_time
  (T : ℝ) (rA rB rC : ℝ) (hT : T = 800) (hrA : rA = 40) (hrB : rB = 30) (hrC : rC = -20) :
  fill_time T rA rB rC = 48 :=
by
  sorry

end tank_fill_time_l17_17147


namespace length_of_third_side_l17_17710

-- Define the properties and setup for the problem
variables {a b : ℝ} (h1 : a = 4) (h2 : b = 8)

-- Define the condition for an isosceles triangle
def isosceles_triangle (x y z : ℝ) : Prop :=
  (x = y ∧ x ≠ z) ∨ (x = z ∧ x ≠ y) ∨ (y = z ∧ y ≠ x)

-- Define the condition for a valid triangle
def valid_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- State the theorem to be proved
theorem length_of_third_side (c : ℝ) (h : isosceles_triangle a b c ∧ valid_triangle a b c) : c = 8 :=
sorry

end length_of_third_side_l17_17710


namespace find_sum_A_B_l17_17464

-- Define ω as a root of the polynomial x^2 + x + 1
noncomputable def ω : ℂ := sorry

-- Define the polynomial P
noncomputable def P (x : ℂ) (A B : ℂ) : ℂ := x^101 + A * x + B

-- State the main theorem
theorem find_sum_A_B (A B : ℂ) : 
  (∀ x : ℂ, (x^2 + x + 1 = 0) → P x A B = 0) → A + B = 2 :=
by
  intros Divisibility
  -- Here, you would provide the steps to prove the theorem if necessary
  sorry

end find_sum_A_B_l17_17464


namespace energy_equivalence_l17_17553

def solar_energy_per_sqm := 1.3 * 10^8
def china_land_area := 9.6 * 10^6
def expected_coal_energy := 1.248 * 10^15

theorem energy_equivalence : 
  solar_energy_per_sqm * china_land_area = expected_coal_energy := 
by
  sorry

end energy_equivalence_l17_17553


namespace FO_greater_DI_l17_17969

-- The quadrilateral FIDO is assumed to be convex with specified properties
variables {F I D O E : Type*}

variables (length_FI length_DI length_DO length_FO : ℝ)
variables (angle_FIO angle_DIO : ℝ)
variables (E : I)

-- Given conditions
variables (convex_FIDO : Prop) -- FIDO is convex
variables (h1 : length_FI = length_DO)
variables (h2 : length_FI > length_DI)
variables (h3 : angle_FIO = angle_DIO)

-- Use given identity IE = ID
variables (length_IE : ℝ) (h4 : length_IE = length_DI)

theorem FO_greater_DI 
    (length_FI length_DI length_DO length_FO : ℝ)
    (angle_FIO angle_DIO : ℝ)
    (convex_FIDO : Prop)
    (h1 : length_FI = length_DO)
    (h2 : length_FI > length_DI)
    (h3 : angle_FIO = angle_DIO)
    (length_IE : ℝ)
    (h4 : length_IE = length_DI) : 
    length_FO > length_DI :=
sorry

end FO_greater_DI_l17_17969


namespace initial_cookies_count_l17_17140

theorem initial_cookies_count (x : ℕ) (h_ate : ℕ) (h_left : ℕ) :
  h_ate = 2 → h_left = 5 → (x - h_ate = h_left) → x = 7 :=
by
  intros
  sorry

end initial_cookies_count_l17_17140


namespace inequality_holds_for_all_reals_l17_17290

theorem inequality_holds_for_all_reals (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
by
  sorry

end inequality_holds_for_all_reals_l17_17290


namespace floor_neg_seven_fourths_l17_17897

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l17_17897


namespace washing_whiteboards_l17_17534

/-- Define the conditions from the problem:
1. Four kids can wash three whiteboards in 20 minutes.
2. It takes one kid 160 minutes to wash a certain number of whiteboards. -/
def four_kids_wash_in_20_min : ℕ := 3
def time_per_batch : ℕ := 20
def one_kid_time : ℕ := 160
def intervals : ℕ := one_kid_time / time_per_batch

/-- Proving the answer based on the conditions:
one kid can wash six whiteboards in 160 minutes given these conditions. -/
theorem washing_whiteboards : intervals * (four_kids_wash_in_20_min / 4) = 6 :=
by
  sorry

end washing_whiteboards_l17_17534


namespace major_arc_circumference_l17_17728

noncomputable def circumference_major_arc 
  (A B C : Point) (r : ℝ) (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) : ℝ :=
  let total_circumference := 2 * Real.pi * r
  let major_arc_angle := 360 - angle_ACB
  major_arc_angle / 360 * total_circumference

theorem major_arc_circumference (A B C : Point) (r : ℝ)
  (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) :
  circumference_major_arc A B C r angle_ACB h1 h2 = (500 / 3) * Real.pi :=
  sorry

end major_arc_circumference_l17_17728


namespace complex_power_difference_l17_17213

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 40 - (1 - i) ^ 40 = 0 := by 
  sorry

end complex_power_difference_l17_17213


namespace regular_octagon_interior_angle_l17_17809

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l17_17809


namespace range_G_l17_17171

noncomputable def G (x : ℝ) : ℝ := |x + 2| - 2 * |x - 2|

theorem range_G : Set.range G = Set.Icc (-8 : ℝ) 8 := sorry

end range_G_l17_17171


namespace solution_set_inequality_l17_17465

theorem solution_set_inequality : {x : ℝ | (x-1)*(x-2) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end solution_set_inequality_l17_17465


namespace ball_box_arrangement_l17_17702

-- Given n distinguishable balls and m distinguishable boxes,
-- prove that the number of ways to place the n balls into the m boxes is m^n.
-- Specifically for n = 6 and m = 3.

theorem ball_box_arrangement : (3^6 = 729) :=
by
  sorry

end ball_box_arrangement_l17_17702


namespace equilibrium_price_quantity_quantity_increase_due_to_subsidy_l17_17616

theorem equilibrium_price_quantity (p Q : ℝ) :
  (∀ p, Q^S(p) = 2 + 8 * p) →
  (Q^D(2) = 8 ∧ Q^D(3) = 6) →
  (∀ p, Q^D(p) = -2 * p + 12) →
  ∃ p Q, Q^D(p) = Q^S(p) ∧ Q = 10 :=
by
  intros
  have h₁ : Q^D(p) = -2 * p + 12 := sorry
  have h₂ : Q^S(p) = 2 + 8 * p := sorry
  use 1
  use 10
  simp [Q^D, Q^S]
  split
  sorry -- detailed steps to show Q^D(1) = 10 and Q^S(1) = 10

theorem quantity_increase_due_to_subsidy (p Q : ℝ) (s : ℝ) :
  s = 1 →
  (∀ p, Q^S(p) = 2 + 8 * p) →
  (∀ p, Q^D(p) = -2 * p + 12) →
  ∃ ΔQ, ΔQ = 1.6 :=
by
  intros
  have Q_s : Q^S(p + s) = 2 + 8 * (p + 1) := sorry
  have Q_d : Q^D(p) = -2 * p + 12 := sorry
  have new_p : p = 0.2 := sorry
  have new_Q : Q^S(0.2) = 11.6 := sorry
  use 1.6
  simp
  sorry -- detailed steps to show ΔQ = 1.6.

end equilibrium_price_quantity_quantity_increase_due_to_subsidy_l17_17616


namespace braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l17_17525

section braking_distance

variables {t k v s : ℝ}

-- Problem 1
theorem braking_distance_non_alcohol: 
  (t = 0.5) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 15) :=
by intros; sorry

-- Problem 2a
theorem reaction_time_after_alcohol:
  (v = 15) ∧ (s = 52.5) ∧ (k = 0.1) → (s = t * v + k * v^2) → (t = 2) :=
by intros; sorry

-- Problem 2b
theorem braking_distance_after_alcohol:
  (t = 2) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 30) :=
by intros; sorry

-- Problem 2c
theorem increase_in_braking_distance:
  (s_after = 30) ∧ (s_before = 15) → (diff = s_after - s_before) → (diff = 15) :=
by intros; sorry

-- Problem 3
theorem max_reaction_time:
  (v = 12) ∧ (k = 0.1) ∧ (s ≤ 42) → (s = t * v + k * v^2) → (t ≤ 2.3) :=
by intros; sorry

end braking_distance

end braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l17_17525


namespace meal_cost_is_seven_l17_17983

-- Defining the given conditions
def total_cost : ℕ := 21
def number_of_meals : ℕ := 3

-- The amount each meal costs
def meal_cost : ℕ := total_cost / number_of_meals

-- Prove that each meal costs 7 dollars given the conditions
theorem meal_cost_is_seven : meal_cost = 7 :=
by
  -- The result follows directly from the definition of meal_cost
  unfold meal_cost
  have h : 21 / 3 = 7 := by norm_num
  exact h


end meal_cost_is_seven_l17_17983


namespace number_of_terms_in_product_l17_17418

theorem number_of_terms_in_product 
  (a b c d e f g h i : ℕ) :
  (a + b + c + d) * (e + f + g + h + i) = 20 :=
sorry

end number_of_terms_in_product_l17_17418


namespace no_such_set_exists_l17_17187

theorem no_such_set_exists :
  ¬ ∃ (A : Finset ℕ), A.card = 11 ∧
  (∀ (s : Finset ℕ), s ⊆ A → s.card = 6 → ¬ 6 ∣ s.sum id) :=
sorry

end no_such_set_exists_l17_17187


namespace inequality_xyz_l17_17275

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l17_17275


namespace regular_octagon_interior_angle_l17_17804

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17804


namespace inequality_xyz_l17_17273

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l17_17273


namespace parabola_equation_l17_17866

noncomputable def parabola_focus : (ℝ × ℝ) := (5, -2)

noncomputable def parabola_directrix (x y : ℝ) : Prop := 4 * x - 5 * y = 20

theorem parabola_equation (x y : ℝ) :
  (parabola_focus = (5, -2)) →
  (parabola_directrix x y) →
  25 * x^2 + 40 * x * y + 16 * y^2 - 650 * x + 184 * y + 1009 = 0 :=
by
  sorry

end parabola_equation_l17_17866


namespace cost_per_pack_l17_17318

variable (total_amount : ℕ) (number_of_packs : ℕ)

theorem cost_per_pack (h1 : total_amount = 132) (h2 : number_of_packs = 11) : 
  total_amount / number_of_packs = 12 := by
  sorry

end cost_per_pack_l17_17318


namespace temperature_difference_l17_17238

theorem temperature_difference (high low : ℝ) (h_high : high = 5) (h_low : low = -3) :
  high - low = 8 :=
by {
  -- Proof goes here
  sorry
}

end temperature_difference_l17_17238


namespace george_total_payment_in_dollars_l17_17189
noncomputable def total_cost_in_dollars : ℝ := 
  let sandwich_cost : ℝ := 4
  let juice_cost : ℝ := 2 * sandwich_cost * 0.9
  let coffee_cost : ℝ := sandwich_cost / 2
  let milk_cost : ℝ := 0.75 * (sandwich_cost + juice_cost)
  let milk_cost_dollars : ℝ := milk_cost * 1.2
  let chocolate_bar_cost_pounds : ℝ := 3
  let chocolate_bar_cost_dollars : ℝ := chocolate_bar_cost_pounds * 1.25
  let total_euros_in_items : ℝ := 2 * sandwich_cost + juice_cost + coffee_cost
  let total_euros_to_dollars : ℝ := total_euros_in_items * 1.2
  total_euros_to_dollars + milk_cost_dollars + chocolate_bar_cost_dollars

theorem george_total_payment_in_dollars : total_cost_in_dollars = 38.07 := by
  sorry

end george_total_payment_in_dollars_l17_17189


namespace inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l17_17429

theorem inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d
  (a b c d : ℚ) 
  (h1 : a * d > b * c) 
  (h2 : (a : ℚ) / b > (c : ℚ) / d) : 
  (a / b > (a + c) / (b + d)) ∧ ((a + c) / (b + d) > c / d) :=
by 
  sorry

end inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l17_17429


namespace simplify_fraction_expression_l17_17336

variable (d : ℝ)

theorem simplify_fraction_expression :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := 
by
  sorry

end simplify_fraction_expression_l17_17336


namespace regular_octagon_interior_angle_l17_17799

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l17_17799


namespace inequality_solution_set_l17_17772

theorem inequality_solution_set (x : ℝ) : (x - 1) * abs (x + 2) ≥ 0 ↔ (x ≥ 1 ∨ x = -2) :=
by
  sorry

end inequality_solution_set_l17_17772


namespace parabola_y_intercepts_l17_17556

theorem parabola_y_intercepts : 
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x : ℝ), x = 0 → 
  ∃ (y : ℝ), 3 * y^2 - 5 * y - 2 = 0 :=
sorry

end parabola_y_intercepts_l17_17556


namespace moving_circle_fixed_point_l17_17198

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

def tangent_line (c : ℝ × ℝ) (r : ℝ) : Prop :=
  abs (c.1 + 1) = r

theorem moving_circle_fixed_point :
  ∀ (c : ℝ × ℝ) (r : ℝ),
    parabola c →
    tangent_line c r →
    (1, 0) ∈ {p : ℝ × ℝ | dist c p = r} :=
by
  intro c r hc ht
  sorry

end moving_circle_fixed_point_l17_17198


namespace additional_people_required_l17_17049

-- Define conditions
def people := 8
def time1 := 3
def total_work := people * time1 -- This gives us the constant k

-- Define the second condition where 12 people are needed to complete in 2 hours
def required_people (t : Nat) := total_work / t

-- The number of additional people required
def additional_people := required_people 2 - people

-- State the theorem
theorem additional_people_required : additional_people = 4 :=
by 
  show additional_people = 4
  sorry

end additional_people_required_l17_17049


namespace brittany_second_test_grade_l17_17373

theorem brittany_second_test_grade
  (first_test_grade second_test_grade : ℕ) 
  (average_after_second : ℕ)
  (h1 : first_test_grade = 78)
  (h2 : average_after_second = 81) 
  (h3 : (first_test_grade + second_test_grade) / 2 = average_after_second) :
  second_test_grade = 84 :=
by
  sorry

end brittany_second_test_grade_l17_17373


namespace find_rate_of_interest_l17_17867

-- Definitions based on conditions
def Principal : ℝ := 7200
def SimpleInterest : ℝ := 3150
def Time : ℝ := 2.5
def RatePerAnnum (R : ℝ) : Prop := SimpleInterest = (Principal * R * Time) / 100

-- Theorem statement
theorem find_rate_of_interest (R : ℝ) (h : RatePerAnnum R) : R = 17.5 :=
by { sorry }

end find_rate_of_interest_l17_17867


namespace total_weight_of_towels_is_40_lbs_l17_17742

def number_of_towels_Mary := 24
def factor_Mary_Frances := 4
def weight_Frances_towels_oz := 128
def pounds_per_ounce := 1 / 16

def number_of_towels_Frances := number_of_towels_Mary / factor_Mary_Frances

def total_number_of_towels := number_of_towels_Mary + number_of_towels_Frances
def weight_per_towel_oz := weight_Frances_towels_oz / number_of_towels_Frances

def total_weight_oz := total_number_of_towels * weight_per_towel_oz
def total_weight_lbs := total_weight_oz * pounds_per_ounce

theorem total_weight_of_towels_is_40_lbs :
  total_weight_lbs = 40 :=
sorry

end total_weight_of_towels_is_40_lbs_l17_17742


namespace B_join_months_after_A_l17_17359

-- Definitions based on conditions
def capitalA (monthsA : ℕ) : ℕ := 3500 * monthsA
def capitalB (monthsB : ℕ) : ℕ := 9000 * monthsB

-- The condition that profit is in ratio 2:3 implies the ratio of their capitals should equal 2:3
def ratio_condition (x : ℕ) : Prop := 2 * (capitalB (12 - x)) = 3 * (capitalA 12)

-- Main theorem stating that B joined the business 5 months after A started
theorem B_join_months_after_A : ∃ x, ratio_condition x ∧ x = 5 :=
by
  use 5
  -- Proof would go here
  sorry

end B_join_months_after_A_l17_17359


namespace find_x_l17_17325

theorem find_x 
  (AB AC BC : ℝ) 
  (x : ℝ)
  (hO : π * (AB / 2)^2 = 12 + 2 * x)
  (hP : π * (AC / 2)^2 = 24 + x)
  (hQ : π * (BC / 2)^2 = 108 - x)
  : AC^2 + BC^2 = AB^2 → x = 60 :=
by {
   sorry
}

end find_x_l17_17325


namespace find_y_given_conditions_l17_17211

theorem find_y_given_conditions : 
  ∀ (x y : ℝ), (1.5 * x = 0.75 * y) ∧ (x = 24) → (y = 48) :=
by
  intros x y h
  cases h with h1 h2
  rw h2 at h1
  sorry

end find_y_given_conditions_l17_17211


namespace floor_sqrt_80_l17_17903

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l17_17903


namespace smallest_fraction_numerator_l17_17505

theorem smallest_fraction_numerator (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : (a * 4) > (b * 3)) : a = 73 :=
  sorry

end smallest_fraction_numerator_l17_17505


namespace area_of_closed_shape_l17_17039

theorem area_of_closed_shape :
  ∫ y in (-2 : ℝ)..3, ((2:ℝ)^y + 2 - (2:ℝ)^y) = 10 := by
  sorry

end area_of_closed_shape_l17_17039


namespace interior_angle_of_regular_octagon_l17_17787

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l17_17787


namespace final_population_l17_17034

theorem final_population (P0 : ℕ) (r1 r2 : ℝ) (P2 : ℝ) 
  (h0 : P0 = 1000)
  (h1 : r1 = 1.20)
  (h2 : r2 = 1.30)
  (h3 : P2 = P0 * r1 * r2) : 
  P2 = 1560 := 
sorry

end final_population_l17_17034


namespace floor_neg_seven_fourths_l17_17886

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l17_17886


namespace betty_eggs_per_teaspoon_vanilla_l17_17484

theorem betty_eggs_per_teaspoon_vanilla
  (sugar_cream_cheese_ratio : ℚ)
  (vanilla_cream_cheese_ratio : ℚ)
  (sugar_in_cups : ℚ)
  (eggs_used : ℕ)
  (expected_ratio : ℚ) :
  sugar_cream_cheese_ratio = 1/4 →
  vanilla_cream_cheese_ratio = 1/2 →
  sugar_in_cups = 2 →
  eggs_used = 8 →
  expected_ratio = 2 →
  (eggs_used / (sugar_in_cups * 4 * vanilla_cream_cheese_ratio)) = expected_ratio :=
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_eggs_per_teaspoon_vanilla_l17_17484


namespace floor_sqrt_80_eq_8_l17_17906

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l17_17906


namespace water_ratio_horse_pig_l17_17518

-- Definitions based on conditions
def num_pigs : ℕ := 8
def water_per_pig : ℕ := 3
def num_horses : ℕ := 10
def water_for_chickens : ℕ := 30
def total_water : ℕ := 114

-- Statement of the problem
theorem water_ratio_horse_pig : 
  (total_water - (num_pigs * water_per_pig) - water_for_chickens) / num_horses / water_per_pig = 2 := 
by sorry

end water_ratio_horse_pig_l17_17518


namespace train_speed_km_per_hr_l17_17360

theorem train_speed_km_per_hr 
  (length : ℝ) 
  (time : ℝ) 
  (h_length : length = 150) 
  (h_time : time = 9.99920006399488) : 
  length / time * 3.6 = 54.00287976961843 :=
by
  sorry

end train_speed_km_per_hr_l17_17360


namespace total_socks_l17_17975

-- Definitions based on conditions
def red_pairs : ℕ := 20
def red_socks : ℕ := red_pairs * 2
def black_socks : ℕ := red_socks / 2
def white_socks : ℕ := 2 * (red_socks + black_socks)

-- The main theorem we want to prove
theorem total_socks :
  (red_socks + black_socks + white_socks) = 180 := by
  sorry

end total_socks_l17_17975


namespace tiles_per_row_l17_17307

noncomputable def area_square_room : ℕ := 400
noncomputable def side_length_feet : ℕ := nat.sqrt area_square_room
noncomputable def side_length_inches : ℕ := side_length_feet * 12
noncomputable def tile_size : ℕ := 8

theorem tiles_per_row : (side_length_inches / tile_size) = 30 := 
by
  have h1: side_length_feet = 20 := by sorry
  have h2: side_length_inches = 240 := by sorry
  have h3: side_length_inches / tile_size = 30 := by sorry
  exact h3

end tiles_per_row_l17_17307


namespace remainder_when_divided_by_100_l17_17706

theorem remainder_when_divided_by_100 (n : ℤ) (h : ∃ a : ℤ, n = 100 * a - 1) : 
  (n^3 + n^2 + 2 * n + 3) % 100 = 1 :=
by 
  sorry

end remainder_when_divided_by_100_l17_17706


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17826

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17826


namespace ratio_A_to_B_l17_17499

/--
Proof problem statement:
Given that A and B together can finish the work in 4 days,
and B alone can finish the work in 24 days,
prove that the ratio of the time A takes to finish the work to the time B takes to finish the work is 1:5.
-/
theorem ratio_A_to_B
  (A_time B_time working_together_time : ℝ) 
  (h1 : working_together_time = 4)
  (h2 : B_time = 24)
  (h3 : 1 / A_time + 1 / B_time = 1 / working_together_time) :
  A_time / B_time = 1 / 5 :=
sorry

end ratio_A_to_B_l17_17499


namespace johns_average_speed_last_hour_l17_17580

theorem johns_average_speed_last_hour
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (distance_last_hour : ℕ)
  (average_speed_last_hour : ℕ)
  (H1 : total_distance = 120)
  (H2 : total_time = 3)
  (H3 : speed_first_hour = 40)
  (H4 : speed_second_hour = 50)
  (H5 : distance_last_hour = total_distance - (speed_first_hour + speed_second_hour))
  (H6 : average_speed_last_hour = distance_last_hour / 1)
  : average_speed_last_hour = 30 := 
by
  -- Placeholder for the proof
  sorry

end johns_average_speed_last_hour_l17_17580


namespace strawberries_jam_profit_l17_17657

noncomputable def betty_strawberries : ℕ := 25
noncomputable def matthew_strawberries : ℕ := betty_strawberries + 30
noncomputable def natalie_strawberries : ℕ := matthew_strawberries / 3  -- Integer division rounds down
noncomputable def total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries
noncomputable def strawberries_per_jar : ℕ := 12
noncomputable def jars_of_jam : ℕ := total_strawberries / strawberries_per_jar  -- Integer division rounds down
noncomputable def money_per_jar : ℕ := 6
noncomputable def total_money_made : ℕ := jars_of_jam * money_per_jar

theorem strawberries_jam_profit :
  total_money_made = 48 := by
  sorry

end strawberries_jam_profit_l17_17657


namespace regular_octagon_interior_angle_l17_17812

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l17_17812


namespace amy_school_year_hours_l17_17513

noncomputable def summer_hours_per_week := 40
noncomputable def summer_weeks := 8
noncomputable def summer_earnings := 3200
noncomputable def school_year_weeks := 32
noncomputable def school_year_earnings_needed := 4800

theorem amy_school_year_hours
  (H1 : summer_earnings = summer_hours_per_week * summer_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  (H2 : school_year_earnings_needed = school_year_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  : (school_year_earnings_needed / school_year_weeks / (summer_earnings / (summer_hours_per_week * summer_weeks))) = 15 :=
by
  sorry

end amy_school_year_hours_l17_17513


namespace evaluate_expression_l17_17674

theorem evaluate_expression :
  1002^3 - 1001 * 1002^2 - 1001^2 * 1002 + 1001^3 - 1000^3 = 2009007 :=
by
  sorry

end evaluate_expression_l17_17674


namespace both_A_and_B_are_Gnomes_l17_17425

inductive Inhabitant
| Elf
| Gnome

open Inhabitant

def lies_about_gold (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def tells_truth_about_others (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def A_statement : Prop := ∀ i : Inhabitant, lies_about_gold i → i = Gnome
def B_statement : Prop := ∀ i : Inhabitant, tells_truth_about_others i → i = Gnome

theorem both_A_and_B_are_Gnomes (A_statement_true : A_statement) (B_statement_true : B_statement) :
  ∀ i : Inhabitant, (lies_about_gold i ∧ tells_truth_about_others i) → i = Gnome :=
by
  sorry

end both_A_and_B_are_Gnomes_l17_17425


namespace partition_nat_set_l17_17755

theorem partition_nat_set :
  ∃ (P : ℕ → ℕ), (∀ (n : ℕ), P n < 100) ∧ (∀ (a b c : ℕ), a + 99 * b = c → (P a = P b ∨ P b = P c ∨ P c = P a)) :=
sorry

end partition_nat_set_l17_17755


namespace hyperbola_eccentricity_range_l17_17061

theorem hyperbola_eccentricity_range (a : ℝ) (h_range: 0 < a ∧ a ≤ 1) :
  ∃ e : Set ℝ, e = Set.Ico (Real.sqrt 2) (Real.sqrt 21 / 3) :=
by
  sorry

end hyperbola_eccentricity_range_l17_17061


namespace interior_angle_regular_octagon_l17_17813

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l17_17813


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17824

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17824


namespace num_races_necessary_l17_17468

/-- There are 300 sprinters registered for a 200-meter dash at a local track meet,
where the track has only 8 lanes. In each race, 3 of the competitors advance to the
next round, while the rest are eliminated immediately. Determine how many races are
needed to identify the champion sprinter. -/
def num_races_to_champion (total_sprinters : ℕ) (lanes : ℕ) (advance_per_race : ℕ) : ℕ :=
  if h : advance_per_race < lanes ∧ lanes > 0 then
    let eliminations_per_race := lanes - advance_per_race
    let total_eliminations := total_sprinters - 1
    Nat.ceil (total_eliminations / eliminations_per_race)
  else
    0

theorem num_races_necessary
  (total_sprinters : ℕ)
  (lanes : ℕ)
  (advance_per_race : ℕ)
  (h_total_sprinters : total_sprinters = 300)
  (h_lanes : lanes = 8)
  (h_advance_per_race : advance_per_race = 3) :
  num_races_to_champion total_sprinters lanes advance_per_race = 60 := by
  sorry

end num_races_necessary_l17_17468


namespace termites_count_l17_17362

theorem termites_count (total_workers monkeys : ℕ) (h1 : total_workers = 861) (h2 : monkeys = 239) : total_workers - monkeys = 622 :=
by
  -- The proof steps will go here
  sorry

end termites_count_l17_17362


namespace find_length_QR_l17_17452

-- Define the provided conditions as Lean definitions
variables (Q P R : ℝ) (h_cos : Real.cos Q = 0.3) (QP : ℝ) (h_QP : QP = 15)
  
-- State the theorem we need to prove
theorem find_length_QR (QR : ℝ) (h_triangle : QP / QR = Real.cos Q) : QR = 50 := sorry

end find_length_QR_l17_17452


namespace part1_general_formula_part2_sum_S_l17_17057

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => a n + 1

theorem part1_general_formula (n : ℕ) : a n = n + 1 := by
  sorry

noncomputable def b (n : ℕ) : ℝ := 1 / (↑n * ↑(n + 2))

noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i => b (i + 1))

theorem part2_sum_S (n : ℕ) : 
  S n = (1/2) * ((3/2) - (1 / (n + 1)) - (1 / (n + 2))) := by
  sorry

end part1_general_formula_part2_sum_S_l17_17057


namespace rate_is_five_l17_17870

noncomputable def rate_per_sq_meter (total_cost : ℕ) (total_area : ℕ) : ℕ :=
  total_cost / total_area

theorem rate_is_five :
  let length := 80
  let breadth := 60
  let road_width := 10
  let total_cost := 6500
  let area_road1 := road_width * breadth
  let area_road2 := road_width * length
  let area_intersection := road_width * road_width
  let total_area := area_road1 + area_road2 - area_intersection
  rate_per_sq_meter total_cost total_area = 5 :=
by
  sorry

end rate_is_five_l17_17870


namespace gcd_gx_x_l17_17107

def g (x : ℕ) : ℕ := (5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (3 * x + 8)

theorem gcd_gx_x (x : ℕ) (h : 27720 ∣ x) : Nat.gcd (g x) x = 168 := by
  sorry

end gcd_gx_x_l17_17107


namespace area_of_region_l17_17132

noncomputable def area_of_enclosed_region : Real :=
  -- equation of the circle after completing square
  let circle_eqn := fun (x y : Real) => ((x - 3)^2 + (y + 4)^2 = 16)
  if circle_eqn then 
    Real.pi * 4^2
  else
    0

theorem area_of_region (h : ∀ x y, x^2 + y^2 - 6x + 8y = -9 → ((x-3)^2 + (y+4)^2 = 16)) :
  area_of_enclosed_region = 16 * Real.pi :=
by
  -- This is a statement, so just include a sorry to skip the proof.
  sorry

end area_of_region_l17_17132


namespace problem_f_neg2_l17_17946

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 2007 + b * x + 1

theorem problem_f_neg2 (a b : ℝ) (h : f a b 2 = 2) : f a b (-2) = 0 :=
by
  sorry

end problem_f_neg2_l17_17946


namespace probability_at_A_after_8_steps_l17_17432

variables (A B C D : Type) {T : Type}
variable [hab: MetricSpace A,B,C,D]
variables (P : ℕ → ℚ)

axiom start_prob : P 0 = 1
axiom recursive_prob :
  ∀ n, P (n + 1) = 1/3 * (1 - P n)

/-- Prove that the probability of the bug being at vertex A after crawling
exactly 8 meters is 547/2187, expressed as p = n/2187, and find n, n = 547. -/
theorem probability_at_A_after_8_steps :
  P 8 = 547 / 2187 :=
sorry

end probability_at_A_after_8_steps_l17_17432


namespace regression_equation_l17_17120

-- Define the regression coefficient and correlation
def negatively_correlated (x y : ℝ) : Prop :=
  ∃ (a : ℝ), a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100

-- The question is to prove that given x and y are negatively correlated,
-- the regression equation is \hat{y} = -2x + 100
theorem regression_equation (x y : ℝ) (h : negatively_correlated x y) :
  (∃ a, a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100) → ∃ (b : ℝ), b = -2 ∧ ∀ (x_val : ℝ), y = b * x_val + 100 :=
by
  sorry

end regression_equation_l17_17120


namespace total_number_of_balls_l17_17048

-- Define the conditions
def balls_per_box : Nat := 3
def number_of_boxes : Nat := 2

-- Define the proposition
theorem total_number_of_balls : (balls_per_box * number_of_boxes) = 6 :=
by
  sorry

end total_number_of_balls_l17_17048


namespace compute_modulo_l17_17380

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l17_17380


namespace product_of_solutions_eq_zero_l17_17184

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (3 * x + 5) / (6 * x + 5) = (5 * x + 4) / (9 * x + 4) → (x = 0 ∨ x = 8 / 3)) →
  0 * (8 / 3) = 0 :=
by
  intro h
  sorry

end product_of_solutions_eq_zero_l17_17184


namespace intersection_of_A_and_B_l17_17690

def setA : Set ℝ := { x | x - 2 ≥ 0 }
def setB : Set ℝ := { x | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | 2 ≤ x ∧ x < 4 } :=
sorry

end intersection_of_A_and_B_l17_17690


namespace special_op_equality_l17_17215

def special_op (x y : ℕ) : ℕ := x * y - x - 2 * y

theorem special_op_equality : (special_op 7 4) - (special_op 4 7) = 3 := by
  sorry

end special_op_equality_l17_17215


namespace sum_xyz_l17_17214

theorem sum_xyz (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 := 
by
  sorry

end sum_xyz_l17_17214


namespace greatest_integer_less_than_150_gcd_18_eq_6_l17_17479

theorem greatest_integer_less_than_150_gcd_18_eq_6 :
  ∃ n : ℕ, n < 150 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 ∧ gcd m 18 = 6 → m ≤ n :=
by
  use 132
  split
  { 
    -- proof that 132 < 150 
    exact sorry 
  }
  split
  { 
    -- proof that gcd 132 18 = 6
    exact sorry 
  }
  {
    -- proof that 132 is the greatest such integer
    exact sorry 
  }

end greatest_integer_less_than_150_gcd_18_eq_6_l17_17479


namespace ratio_is_five_over_twelve_l17_17067

theorem ratio_is_five_over_twelve (a b c d : ℚ) (h1 : b = 4 * a) (h2 : d = 2 * c) :
    (a + b) / (c + d) = 5 / 12 :=
sorry

end ratio_is_five_over_twelve_l17_17067


namespace parabola_directrix_l17_17555

theorem parabola_directrix (p : ℝ) (hp : p > 0) 
  (x1 x2 t : ℝ) 
  (h_intersect : ∃ y1 y2, y1 = x1 + t ∧ y2 = x2 + t ∧ x1^2 = 2 * p * y1 ∧ x2^2 = 2 * p * y2)
  (h_midpoint : (x1 + x2) / 2 = 2) :
  p = 2 → ∃ d : ℝ, d = -1 := 
by
  sorry

end parabola_directrix_l17_17555


namespace binomial_identity_l17_17978

-- Given:
variables {k n : ℕ}

-- Conditions:
axiom h₁ : 1 < k
axiom h₂ : 1 < n

-- Statement:
theorem binomial_identity (h₁ : 1 < k) (h₂ : 1 < n) : 
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := 
sorry

end binomial_identity_l17_17978


namespace dasha_paper_strip_l17_17044

theorem dasha_paper_strip (a b c : ℕ) (h1 : a < b) (h2 : 2 * a * b + 2 * a * c - a^2 = 43) :
    ∃ (length width : ℕ), length = a ∧ width = b + c := by
  sorry

end dasha_paper_strip_l17_17044


namespace area_of_PQRS_l17_17240

noncomputable def length_EF := 6
noncomputable def width_EF := 4

noncomputable def area_PQRS := (length_EF + 6 * Real.sqrt 3) * (width_EF + 4 * Real.sqrt 3)

theorem area_of_PQRS :
  area_PQRS = 60 + 48 * Real.sqrt 3 := by
  sorry

end area_of_PQRS_l17_17240


namespace line_equation_solution_l17_17413

noncomputable def line_equation (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  ∃ (l : ℝ → ℝ), (l P.fst = P.snd) ∧ (∀ (x : ℝ), l x = 4 * x - 2) ∨ (∀ (x : ℝ), x = 1)

theorem line_equation_solution : line_equation (1, 2) (2, 3) (0, -5) :=
sorry

end line_equation_solution_l17_17413


namespace range_of_m_l17_17060

theorem range_of_m (m : ℝ) (h1 : (m - 3) < 0) (h2 : (m + 1) > 0) : -1 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l17_17060


namespace security_deposit_correct_l17_17230

-- Definitions (Conditions)
def daily_rate : ℝ := 125
def pet_fee_per_dog : ℝ := 100
def number_of_dogs : ℕ := 2
def tourism_tax_rate : ℝ := 0.10
def service_fee_rate : ℝ := 0.20
def activity_cost_per_person : ℝ := 45
def number_of_activities_per_person : ℕ := 3
def number_of_people : ℕ := 2
def security_deposit_rate : ℝ := 0.50
def usd_to_euro_conversion_rate : ℝ := 0.83

-- Function to calculate total cost
def total_cost_in_euros : ℝ :=
  let rental_cost := daily_rate * 14
  let pet_cost := pet_fee_per_dog * number_of_dogs
  let tourism_tax := tourism_tax_rate * rental_cost
  let service_fee := service_fee_rate * rental_cost
  let cabin_total := rental_cost + pet_cost + tourism_tax + service_fee
  let activities_total := number_of_activities_per_person * activity_cost_per_person * number_of_people
  let total_cost := cabin_total + activities_total
  let security_deposit_usd := security_deposit_rate * total_cost
  security_deposit_usd * usd_to_euro_conversion_rate

-- Theorem to prove
theorem security_deposit_correct :
  total_cost_in_euros = 1139.18 := 
sorry

end security_deposit_correct_l17_17230


namespace least_number_added_1054_l17_17329

theorem least_number_added_1054 (x d: ℕ) (h_cond: 1054 + x = 1058) (h_div: d = 2) : 1058 % d = 0 :=
by
  sorry

end least_number_added_1054_l17_17329


namespace juice_cans_count_l17_17872

theorem juice_cans_count :
  let original_price := 12 
  let discount := 2 
  let tub_sale_price := original_price - discount 
  let tub_quantity := 2 
  let ice_cream_total := tub_quantity * tub_sale_price 
  let total_payment := 24 
  let juice_cost_per_5cans := 2 
  let remaining_amount := total_payment - ice_cream_total 
  let sets_of_juice_cans := remaining_amount / juice_cost_per_5cans 
  let cans_per_set := 5 
  2 * cans_per_set = 10 :=
by
  sorry

end juice_cans_count_l17_17872


namespace simplify_fraction_expression_l17_17335

theorem simplify_fraction_expression (d : ℤ) :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 :=
by
  -- skip the proof by adding sorry
  sorry

end simplify_fraction_expression_l17_17335


namespace participants_with_exactly_five_problems_l17_17570

theorem participants_with_exactly_five_problems (n : ℕ) 
  (p : Fin 6 → Fin 6 → ℕ)
  (h1 : ∀ i j : Fin 6, i ≠ j → p i j > 2 * n / 5)
  (h2 : ¬ ∃ i : Fin 6, ∀ j : Fin 6, j ≠ i → p i j = n)
  : ∃ k1 k2 : Fin n, k1 ≠ k2 ∧ (∀ i : Fin 6, (p i k1 = 5) ∧ (p i k2 = 5)) :=
sorry

end participants_with_exactly_five_problems_l17_17570


namespace regular_octagon_interior_angle_l17_17808

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l17_17808


namespace james_total_socks_l17_17974

theorem james_total_socks :
  ∀ (red_socks_pairs : ℕ) (black_socks_ratio red_socks_ratio : ℕ),
    red_socks_pairs = 20 →
    black_socks_ratio = 2 →
    red_socks_ratio = 2 →
    let red_socks := red_socks_pairs * 2 in
    let black_socks := red_socks / black_socks_ratio in
    let red_black_combined := red_socks + black_socks in
    let white_socks := red_black_combined * red_socks_ratio in
    red_socks + black_socks + white_socks = 180 :=
by
  intros red_socks_pairs black_socks_ratio red_socks_ratio
  intro h1 h2 h3
  let red_socks := red_socks_pairs * 2
  let black_socks := red_socks / black_socks_ratio
  let red_black_combined := red_socks + black_socks
  let white_socks := red_black_combined * red_socks_ratio
  have step1 : red_socks = 40 := by rw [h1]; refl
  have step2 : black_socks = 20 := by rw [step1, h2]; refl
  have step3 : red_black_combined = 60 := by rw [step1, step2]; refl
  have step4 : white_socks = 120 := by rw [step3, h3]; refl
  calc 
    red_socks + black_socks + white_socks 
      = 40 + 20 + 120 : by rw [step1, step2, step4]
      ... = 180 : by norm_num

end james_total_socks_l17_17974


namespace minimum_positive_period_l17_17460

open Real

noncomputable def function := fun x : ℝ => 3 * sin (2 * x + π / 3)

theorem minimum_positive_period : ∃ T > 0, ∀ x, function (x + T) = function x ∧ (∀ T', T' > 0 → (∀ x, function (x + T') = function x) → T ≤ T') :=
  sorry

end minimum_positive_period_l17_17460


namespace marie_erasers_l17_17234

-- Define the initial conditions
def initial_erasers : ℝ := 95.0
def additional_erasers : ℝ := 42.0

-- Define the target final erasers count
def final_erasers : ℝ := 137.0

-- The theorem we need to prove
theorem marie_erasers :
  initial_erasers + additional_erasers = final_erasers := by
  sorry

end marie_erasers_l17_17234


namespace prod_mod7_eq_zero_l17_17379

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l17_17379


namespace parabola_passes_through_points_and_has_solution_4_l17_17463

theorem parabola_passes_through_points_and_has_solution_4 
  (a h k m: ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k → 
    (y = 0 → (x = -1 → x = 5))) → 
  (∃ m, ∀ x, (a * (x - h + m) ^ 2 + k = 0) → x = 4) → 
  m = -5 ∨ m = 1 :=
sorry

end parabola_passes_through_points_and_has_solution_4_l17_17463


namespace compute_product_l17_17665

noncomputable def P (x : ℂ) : ℂ := ∏ k in (finset.range 15).map (nat.cast), (x - exp (2 * pi * complex.I * k / 17))

noncomputable def Q (x : ℂ) : ℂ := ∏ j in (finset.range 12).map (nat.cast), (x - exp (2 * pi * complex.I * j / 13))

theorem compute_product : 
  (∏ k in (finset.range 15).map (nat.cast), ∏ j in (finset.range 12).map (nat.cast), (exp (2 * pi * complex.I * j / 13) - exp (2 * pi * complex.I * k / 17))) = 1 :=
by
  sorry

end compute_product_l17_17665


namespace not_taking_ship_probability_l17_17649

-- Real non-negative numbers as probabilities
variables (P_train P_ship P_car P_airplane : ℝ)

-- Conditions
axiom h_train : 0 ≤ P_train ∧ P_train ≤ 1 ∧ P_train = 0.3
axiom h_ship : 0 ≤ P_ship ∧ P_ship ≤ 1 ∧ P_ship = 0.1
axiom h_car : 0 ≤ P_car ∧ P_car ≤ 1 ∧ P_car = 0.4
axiom h_airplane : 0 ≤ P_airplane ∧ P_airplane ≤ 1 ∧ P_airplane = 0.2

-- Prove that the probability of not taking a ship is 0.9
theorem not_taking_ship_probability : 1 - P_ship = 0.9 :=
by
  sorry

end not_taking_ship_probability_l17_17649


namespace inequality_proof_l17_17251

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l17_17251


namespace greatest_x_solution_l17_17183

theorem greatest_x_solution : ∀ x : ℝ, 
    (x ≠ 6) ∧ (x ≠ -4) ∧ ((x^2 - 3*x - 18)/(x-6) = 2 / (x+4)) → x ≤ -2 :=
by
  sorry

end greatest_x_solution_l17_17183


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17846

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17846


namespace floor_sqrt_80_eq_8_l17_17918

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l17_17918


namespace complement_union_l17_17232

open Set

variable (U M N : Set ℕ)

def complement_U (A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem complement_union (hU : U = {0, 1, 2, 3, 4, 5, 6})
                          (hM : M = {1, 3, 5})
                          (hN : N = {2, 4, 6}) :
  (complement_U U M) ∪ (complement_U U N) = {0, 1, 2, 3, 4, 5, 6} :=
by 
  sorry

end complement_union_l17_17232


namespace max_value_of_expression_l17_17092

variable (a b c : ℝ)

theorem max_value_of_expression : 
  ∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c = Real.sqrt (a^2 + b^2) + c := by
sorry

end max_value_of_expression_l17_17092


namespace dealership_truck_sales_l17_17877

theorem dealership_truck_sales (SUVs Trucks : ℕ) (h1 : SUVs = 45) (h2 : 3 * Trucks = 5 * SUVs) : Trucks = 75 :=
by
  sorry

end dealership_truck_sales_l17_17877


namespace rectangle_area_l17_17860

theorem rectangle_area (x : ℝ) (w : ℝ) (h : w^2 + (2 * w)^2 = x^2) : 
  2 * (w^2) = (2 / 5) * x^2 :=
by
  sorry

end rectangle_area_l17_17860


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17847

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17847


namespace regular_octagon_interior_angle_l17_17820

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l17_17820


namespace sum_possible_values_l17_17961

theorem sum_possible_values (x : ℤ) (h : ∃ y : ℤ, y = (3 * x + 13) / (x + 6)) :
  ∃ s : ℤ, s = -2 + 8 + 2 + 4 :=
sorry

end sum_possible_values_l17_17961


namespace inequality_proof_l17_17250

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l17_17250


namespace no_real_solution_abs_eq_quadratic_l17_17528

theorem no_real_solution_abs_eq_quadratic (x : ℝ) : abs (2 * x - 6) ≠ x^2 - x + 2 := by
  sorry

end no_real_solution_abs_eq_quadratic_l17_17528


namespace floor_sqrt_80_l17_17900

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l17_17900


namespace find_b7_l17_17128

/-- We represent the situation with twelve people in a circle, each with an integer number. The
     average announced by a person is the average of their two immediate neighbors. Given the
     person who announced the average of 7, we aim to find the number they initially chose. --/
theorem find_b7 (b : ℕ → ℕ) (announced_avg : ℕ → ℕ) :
  (announced_avg 1 = (b 12 + b 2) / 2) ∧
  (announced_avg 2 = (b 1 + b 3) / 2) ∧
  (announced_avg 3 = (b 2 + b 4) / 2) ∧
  (announced_avg 4 = (b 3 + b 5) / 2) ∧
  (announced_avg 5 = (b 4 + b 6) / 2) ∧
  (announced_avg 6 = (b 5 + b 7) / 2) ∧
  (announced_avg 7 = (b 6 + b 8) / 2) ∧
  (announced_avg 8 = (b 7 + b 9) / 2) ∧
  (announced_avg 9 = (b 8 + b 10) / 2) ∧
  (announced_avg 10 = (b 9 + b 11) / 2) ∧
  (announced_avg 11 = (b 10 + b 12) / 2) ∧
  (announced_avg 12 = (b 11 + b 1) / 2) ∧
  (announced_avg 7 = 7) →
  b 7 = 12 := 
sorry

end find_b7_l17_17128


namespace floor_neg_7_over_4_l17_17893

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l17_17893


namespace find_value_of_a5_l17_17973

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a_1 d : ℝ), ∀ n, a n = a_1 + (n - 1) * d

variable (h_arith : is_arithmetic_sequence a)
variable (h : a 2 + a 8 = 12)

theorem find_value_of_a5 : a 5 = 6 :=
by
  sorry

end find_value_of_a5_l17_17973


namespace find_a_l17_17408

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then a ^ x - 1 else 2 * x ^ 2

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ m n : ℝ, f a m ≤ f a n ↔ m ≤ n)
  (h4 : f a a = 5 * a - 2) : a = 2 :=
sorry

end find_a_l17_17408


namespace measure_of_one_interior_angle_of_regular_octagon_l17_17832

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l17_17832


namespace find_value_of_expression_l17_17685

theorem find_value_of_expression (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) : 2 * a + 2 * b - 3 * (a * b) = 9 :=
by
  sorry

end find_value_of_expression_l17_17685


namespace find_number_l17_17869

theorem find_number (m : ℤ) (h1 : ∃ k1 : ℤ, k1 * k1 = m + 100) (h2 : ∃ k2 : ℤ, k2 * k2 = m + 168) : m = 156 :=
sorry

end find_number_l17_17869


namespace ratio_x_to_w_as_percentage_l17_17585

theorem ratio_x_to_w_as_percentage (x y z w : ℝ) 
    (h1 : x = 1.20 * y) 
    (h2 : y = 0.30 * z) 
    (h3 : z = 1.35 * w) : 
    (x / w) * 100 = 48.6 := 
by sorry

end ratio_x_to_w_as_percentage_l17_17585


namespace GODOT_value_l17_17437

theorem GODOT_value (G O D I T : ℕ) (h1 : G ≠ 0) (h2 : D ≠ 0) 
  (eq1 : 1000 * G + 100 * O + 10 * G + O + 1000 * D + 100 * I + 10 * D + I = 10000 * G + 1000 * O + 100 * D + 10 * O + T) : 
  10000 * G + 1000 * O + 100 * D + 10 * O + T = 10908 :=
by {
  sorry
}

end GODOT_value_l17_17437


namespace sum_of_slopes_range_l17_17190

theorem sum_of_slopes_range (p b : ℝ) (hpb : 2 * p > b) (hp : p > 0) 
  (K1 K2 : ℝ) (A B : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1) (hB : B.2^2 = 2 * p * B.1)
  (hl1 : A.2 = A.1 + b) (hl2 : B.2 = B.1 + b) 
  (hA_pos : A.2 > 0) (hB_pos : B.2 > 0) :
  4 < K1 + K2 :=
sorry

end sum_of_slopes_range_l17_17190


namespace interior_angle_regular_octagon_l17_17793

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l17_17793


namespace solution_for_factorial_equation_l17_17169

theorem solution_for_factorial_equation:
  { (n, k) : ℕ × ℕ | 0 < n ∧ 0 < k ∧ n! + n = n^k } = {(2,2), (3,2), (5,3)} :=
by
  sorry

end solution_for_factorial_equation_l17_17169


namespace floor_sqrt_80_l17_17909

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l17_17909


namespace journey_distance_l17_17851

theorem journey_distance :
  ∃ D T : ℝ,
    D = 100 * T ∧
    D = 80 * (T + 1/3) ∧
    D = 400 / 3 :=
by
  sorry

end journey_distance_l17_17851


namespace diet_soda_bottles_l17_17352

-- Define the conditions and then state the problem
theorem diet_soda_bottles (R D : ℕ) (h1 : R = 67) (h2 : R = D + 58) : D = 9 :=
by
  -- The proof goes here
  sorry

end diet_soda_bottles_l17_17352


namespace candies_problem_max_children_l17_17123

theorem candies_problem_max_children (u v : ℕ → ℕ) (n : ℕ) :
  (∀ i : ℕ, u i = v i + 2) →
  (∀ i : ℕ, u i + 2 = u (i + 1)) →
  (u (n - 1) / u 0 = 13) →
  n = 25 :=
by
  -- Proof not required as per the instructions.
  sorry

end candies_problem_max_children_l17_17123


namespace max_k_C_l17_17387

theorem max_k_C (n : ℕ) (h1 : Odd n) (h2 : 0 < n) :
  ∃ k : ℕ, (k = ((n + 1) / 2) ^ 2) := 
sorry

end max_k_C_l17_17387


namespace rice_price_per_kg_l17_17208

theorem rice_price_per_kg (price1 price2 : ℝ) (amount1 amount2 : ℝ) (total_cost total_weight : ℝ) (P : ℝ)
  (h1 : price1 = 6.60)
  (h2 : amount1 = 49)
  (h3 : price2 = 9.60)
  (h4 : amount2 = 56)
  (h5 : total_cost = price1 * amount1 + price2 * amount2)
  (h6 : total_weight = amount1 + amount2)
  (h7 : P = total_cost / total_weight) :
  P = 8.20 := 
by sorry

end rice_price_per_kg_l17_17208


namespace M_necessary_for_N_l17_17950

open Set

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem M_necessary_for_N : ∀ a : ℝ, a ∈ N → a ∈ M ∧ ¬(a ∈ M → a ∈ N) :=
by
  sorry

end M_necessary_for_N_l17_17950


namespace proof_seq_l17_17858

open Nat

-- Definition of sequence {a_n}
def seq_a : ℕ → ℕ
| 0 => 1
| n + 1 => 3 * seq_a n

-- Definition of sum S_n of sequence {b_n}
def sum_S : ℕ → ℕ
| 0 => 0
| n + 1 => sum_S n + (2^n)

-- Definition of sequence {b_n}
def seq_b : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * seq_b n

-- Definition of sequence {c_n}
def seq_c (n : ℕ) : ℕ := seq_b n * log 3 (seq_a n) -- Note: log base 3

-- Sum of first n terms of {c_n}
def sum_T : ℕ → ℕ
| 0 => 0
| n + 1 => sum_T n + seq_c n

-- Proof statement
theorem proof_seq (n : ℕ) :
  (seq_a n = 3 ^ n) ∧
  (2 * seq_b n - 1 = sum_S 0 * sum_S n) ∧
  (sum_T n = (n - 2) * 2 ^ (n + 2)) :=
sorry

end proof_seq_l17_17858


namespace coefficients_divisible_by_seven_l17_17576

theorem coefficients_divisible_by_seven {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  a % 7 = 0 ∧ b % 7 = 0 ∧ c % 7 = 0 ∧ d % 7 = 0 ∧ e % 7 = 0 := 
  sorry

end coefficients_divisible_by_seven_l17_17576


namespace sufficient_but_not_necessary_condition_l17_17979

variables (A B C : Prop)

theorem sufficient_but_not_necessary_condition (h1 : B → A) (h2 : C → B) (h3 : ¬(B → C)) : (C → A) ∧ ¬(A → C) :=
by
  sorry

end sufficient_but_not_necessary_condition_l17_17979


namespace expectation_is_correct_l17_17063

variable (m : ℝ)
variable (ξ : ℕ → ℝ)
variable (P : ℕ → ℝ)

-- Conditions from the problem
axiom prob_1 : P 1 = 0.3
axiom prob_2 : P 2 = m
axiom prob_3 : P 3 = 0.4
axiom total_prob : P 1 + P 2 + P 3 = 1

-- Proving the expectation is 2.1
theorem expectation_is_correct : E ξ = 2.1 := by
sorry

end expectation_is_correct_l17_17063


namespace contrapositive_of_x_squared_lt_one_is_true_l17_17607

variable {x : ℝ}

theorem contrapositive_of_x_squared_lt_one_is_true
  (h : ∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) :
  ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1 :=
by
  sorry

end contrapositive_of_x_squared_lt_one_is_true_l17_17607


namespace square_side_length_l17_17423

theorem square_side_length (x : ℝ) (h : x ^ 2 = 4 * 3) : x = 2 * Real.sqrt 3 :=
by sorry

end square_side_length_l17_17423


namespace inequality_inequality_l17_17262

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l17_17262


namespace evaluate_expression_at_two_l17_17631

theorem evaluate_expression_at_two: 
  (3 * 2^2 - 4 * 2 + 2) = 6 := 
by 
  sorry

end evaluate_expression_at_two_l17_17631


namespace standard_colony_condition_l17_17482

noncomputable def StandardBacterialColony : Prop := sorry

theorem standard_colony_condition (visible_mass_of_microorganisms : Prop) 
                                   (single_mother_cell : Prop) 
                                   (solid_culture_medium : Prop) 
                                   (not_multiple_types : Prop) 
                                   : StandardBacterialColony :=
sorry

end standard_colony_condition_l17_17482


namespace Sn_minimum_value_l17_17981

theorem Sn_minimum_value {a : ℕ → ℤ} (n : ℕ) (S : ℕ → ℤ)
  (h1 : a 1 = -11)
  (h2 : a 4 + a 6 = -6)
  (S_def : ∀ n, S n = n * (-12 + n)) :
  ∃ n, S n = S 6 :=
sorry

end Sn_minimum_value_l17_17981


namespace floor_neg_7_over_4_l17_17890

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l17_17890


namespace regular_octagon_interior_angle_l17_17803

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l17_17803


namespace min_value_a_plus_9b_l17_17550

theorem min_value_a_plus_9b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 1 / b = 1) : a + 9 * b ≥ 16 :=
  sorry

end min_value_a_plus_9b_l17_17550


namespace no_negative_roots_of_polynomial_l17_17417

def polynomial (x : ℝ) := x^4 - 5 * x^3 - 4 * x^2 - 7 * x + 4

theorem no_negative_roots_of_polynomial :
  ¬ ∃ (x : ℝ), x < 0 ∧ polynomial x = 0 :=
by
  sorry

end no_negative_roots_of_polynomial_l17_17417


namespace sin_alpha_value_l17_17540

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_value_l17_17540


namespace range_of_a_l17_17196

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 7 * x1^2 - (a + 13) * x1 + a^2 - a - 2 = 0 ∧
                 7 * x2^2 - (a + 13) * x2 + a^2 - a - 2 = 0 ∧
                 0 < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ x2 < 2) →
  (-2 < a ∧ a < -1) ∨ (3 < a ∧ a < 4) :=
by
  intro h
  sorry

end range_of_a_l17_17196


namespace inequality_holds_for_all_real_numbers_l17_17247

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l17_17247


namespace garden_area_l17_17117

-- Given conditions:
def width := 16
def length (W : ℕ) := 3 * W

-- Proof statement:
theorem garden_area (W : ℕ) (hW : W = width) : length W * W = 768 :=
by
  rw [hW]
  exact rfl

end garden_area_l17_17117


namespace total_action_figures_l17_17579

-- Definitions based on conditions
def initial_figures : ℕ := 8
def figures_per_set : ℕ := 5
def added_sets : ℕ := 2
def total_added_figures : ℕ := added_sets * figures_per_set
def total_figures : ℕ := initial_figures + total_added_figures

-- Theorem statement with conditions and expected result
theorem total_action_figures : total_figures = 18 := by
  sorry

end total_action_figures_l17_17579


namespace tyler_age_l17_17323

theorem tyler_age (T C : ℕ) (h1 : T = 3 * C + 1) (h2 : T + C = 21) : T = 16 :=
by
  sorry

end tyler_age_l17_17323


namespace example_problem_l17_17509

theorem example_problem
  (h1 : 0.25 < 1) 
  (h2 : 0.15 < 0.25) : 
  3.04 / 0.25 > 1 :=
by
  sorry

end example_problem_l17_17509


namespace inequality_inequality_l17_17263

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l17_17263


namespace probability_two_girls_l17_17572

-- Define the conditions
def total_students := 8
def total_girls := 5
def total_boys := 3
def choose_two_from_n (n : ℕ) := n * (n - 1) / 2

-- Define the question as a statement that the probability equals 5/14
theorem probability_two_girls
    (h1 : choose_two_from_n total_students = 28)
    (h2 : choose_two_from_n total_girls = 10) :
    (choose_two_from_n total_girls : ℚ) / choose_two_from_n total_students = 5 / 14 :=
by
  sorry

end probability_two_girls_l17_17572


namespace thabo_number_of_hardcover_nonfiction_books_l17_17761

variables (P_f H_f P_nf H_nf A : ℕ)

theorem thabo_number_of_hardcover_nonfiction_books
  (h1 : P_nf = H_nf + 15)
  (h2 : H_f = P_f + 10)
  (h3 : P_f = 3 * A)
  (h4 : A + H_f = 70)
  (h5 : P_f + H_f + P_nf + H_nf + A = 250) :
  H_nf = 30 :=
by {
  sorry
}

end thabo_number_of_hardcover_nonfiction_books_l17_17761


namespace line_equation_with_equal_intercepts_l17_17696

theorem line_equation_with_equal_intercepts 
  (a : ℝ) 
  (l : ℝ → ℝ → Prop) 
  (h : ∀ x y, l x y ↔ (a+1)*x + y + 2 - a = 0) 
  (intercept_condition : ∀ x y, l x 0 = l 0 y) : 
  (∀ x y, l x y ↔ x + y + 2 = 0) ∨ (∀ x y, l x y ↔ 3*x + y = 0) :=
sorry

end line_equation_with_equal_intercepts_l17_17696


namespace Pythagorean_triple_l17_17663

theorem Pythagorean_triple : ∃ (a b c : ℕ), (a = 6) ∧ (b = 8) ∧ (c = 10) ∧ (a^2 + b^2 = c^2) :=
by
  use 6, 8, 10
  split; try {refl}
  calc
    6^2 + 8^2 = 36 + 64 := by rw [pow_two, pow_two]
            ... = 100 := by norm_num
            ... = 10^2 := by norm_num
  done

end Pythagorean_triple_l17_17663


namespace train_speed_l17_17638

theorem train_speed (length : ℕ) (time : ℕ) (h1 : length = 1600) (h2 : time = 40) : length / time = 40 := 
by
  -- use the given conditions here
  sorry

end train_speed_l17_17638


namespace find_t_l17_17541

-- Definitions of the vectors involved
def vector_AB : ℝ × ℝ := (2, 3)
def vector_AC (t : ℝ) : ℝ × ℝ := (3, t)
def vector_BC (t : ℝ) : ℝ × ℝ := ((vector_AC t).1 - (vector_AB).1, (vector_AC t).2 - (vector_AB).2)

-- Condition for orthogonality
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Main statement to be proved
theorem find_t : ∃ t : ℝ, is_perpendicular vector_AB (vector_BC t) ∧ t = 7 / 3 :=
by
  sorry

end find_t_l17_17541


namespace ellipse_semi_focal_distance_range_l17_17409

theorem ellipse_semi_focal_distance_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) (h_ellipse : a^2 = b^2 + c^2) :
  1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 := 
sorry

end ellipse_semi_focal_distance_range_l17_17409


namespace h_value_l17_17957

theorem h_value (h : ℝ) : (∃ x : ℝ, x^3 + h * x + 5 = 0 ∧ x = 3) → h = -32 / 3 := by
  sorry

end h_value_l17_17957


namespace compute_modulo_l17_17382

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l17_17382


namespace men_in_second_group_l17_17641

theorem men_in_second_group (M : ℕ) (h1 : 36 * 18 = M * 24) : M = 27 :=
by {
  sorry
}

end men_in_second_group_l17_17641


namespace hotel_P_charge_less_than_G_l17_17112

open Real

variable (G R P : ℝ)

-- Given conditions
def charge_R_eq_2G : Prop := R = 2 * G
def charge_P_eq_R_minus_55percent : Prop := P = R - 0.55 * R

-- Goal: Prove the percentage by which P's charge is less than G's charge is 10%
theorem hotel_P_charge_less_than_G : charge_R_eq_2G G R → charge_P_eq_R_minus_55percent R P → P = 0.9 * G := by
  intros h1 h2
  sorry

end hotel_P_charge_less_than_G_l17_17112


namespace discount_percentage_is_10_l17_17457

-- Definitions of the conditions directly translated
def CP (MP : ℝ) : ℝ := 0.7 * MP
def GainPercent : ℝ := 0.2857142857142857
def SP (MP : ℝ) : ℝ := CP MP * (1 + GainPercent)

-- Using the alternative expression for selling price involving discount percentage
def DiscountSP (MP : ℝ) (D : ℝ) : ℝ := MP * (1 - D)

-- The theorem to prove the discount percentage is 10%
theorem discount_percentage_is_10 (MP : ℝ) : ∃ D : ℝ, DiscountSP MP D = SP MP ∧ D = 0.1 := 
by
  use 0.1
  sorry

end discount_percentage_is_10_l17_17457


namespace john_paid_correct_amount_l17_17722

def cost_bw : ℝ := 160
def markup_percentage : ℝ := 0.5

def cost_color : ℝ := cost_bw * (1 + markup_percentage)

theorem john_paid_correct_amount : 
  cost_color = 240 := 
by
  -- proof required here
  sorry

end john_paid_correct_amount_l17_17722


namespace rightmost_three_digits_of_7_pow_1997_l17_17129

theorem rightmost_three_digits_of_7_pow_1997 :
  7^1997 % 1000 = 207 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1997_l17_17129


namespace floor_sqrt_80_l17_17915

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l17_17915


namespace ratio_of_raspberries_l17_17016

theorem ratio_of_raspberries (B R K L : ℕ) (h1 : B = 42) (h2 : L = 7) (h3 : K = B / 3) (h4 : B = R + K + L) :
  R / Nat.gcd R B = 1 ∧ B / Nat.gcd R B = 2 :=
by
  sorry

end ratio_of_raspberries_l17_17016


namespace correct_transformation_l17_17032

-- Given transformations
def transformation_A (a : ℝ) : Prop := - (1 / a) = -1 / a
def transformation_B (a b : ℝ) : Prop := (1 / a) + (1 / b) = 1 / (a + b)
def transformation_C (a b : ℝ) : Prop := (2 * b^2) / a^2 = (2 * b) / a
def transformation_D (a b : ℝ) : Prop := (a + a * b) / (b + a * b) = a / b

-- Correct transformation is A.
theorem correct_transformation (a b : ℝ) : transformation_A a ∧ ¬transformation_B a b ∧ ¬transformation_C a b ∧ ¬transformation_D a b :=
sorry

end correct_transformation_l17_17032


namespace amy_school_year_hours_l17_17512

noncomputable def summer_hours_per_week := 40
noncomputable def summer_weeks := 8
noncomputable def summer_earnings := 3200
noncomputable def school_year_weeks := 32
noncomputable def school_year_earnings_needed := 4800

theorem amy_school_year_hours
  (H1 : summer_earnings = summer_hours_per_week * summer_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  (H2 : school_year_earnings_needed = school_year_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  : (school_year_earnings_needed / school_year_weeks / (summer_earnings / (summer_hours_per_week * summer_weeks))) = 15 :=
by
  sorry

end amy_school_year_hours_l17_17512


namespace stamp_problem_l17_17927

/-- Define the context where we have stamps of 7, n, and (n + 2) cents, and 120 cents being the largest
    value that cannot be formed using these stamps -/
theorem stamp_problem (n : ℕ) (h : ∀ k, k > 120 → ∃ a b c, k = 7 * a + n * b + (n + 2) * c) (hn : ¬ ∃ a b c, 120 = 7 * a + n * b + (n + 2) * c) : n = 22 :=
sorry

end stamp_problem_l17_17927


namespace ratio_of_flowers_given_l17_17376

-- Definitions based on conditions
def Collin_flowers : ℕ := 25
def Ingrid_flowers_initial : ℕ := 33
def petals_per_flower : ℕ := 4
def Collin_petals_total : ℕ := 144

-- The ratio of the number of flowers Ingrid gave to Collin to the number of flowers Ingrid had initially
theorem ratio_of_flowers_given :
  let Ingrid_flowers_given := (Collin_petals_total - (Collin_flowers * petals_per_flower)) / petals_per_flower
  let ratio := Ingrid_flowers_given / Ingrid_flowers_initial
  ratio = 1 / 3 :=
by
  sorry

end ratio_of_flowers_given_l17_17376


namespace flowers_per_bouquet_l17_17721

-- Defining the problem parameters
def total_flowers : ℕ := 66
def wilted_flowers : ℕ := 10
def num_bouquets : ℕ := 7

-- The goal is to prove that the number of flowers per bouquet is 8
theorem flowers_per_bouquet :
  (total_flowers - wilted_flowers) / num_bouquets = 8 :=
by
  sorry

end flowers_per_bouquet_l17_17721


namespace determine_m_to_satisfy_conditions_l17_17998

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 5) * x^(m - 1)

theorem determine_m_to_satisfy_conditions : 
  ∃ (m : ℝ), (m = 3) ∧ ∀ (x : ℝ), (0 < x → (m^2 - m - 5 > 0) ∧ (m - 1 > 0)) :=
by
  sorry

end determine_m_to_satisfy_conditions_l17_17998


namespace number_of_people_in_range_l17_17571

open ProbabilityTheory

noncomputable def normal_distribution (μ σ : ℝ) (X : ℝ → Measure ℝ) : Prop :=
  ∀ a b, X a b = (1 / (σ * sqrt (2 * π))) * exp (-((b - μ)^2) / (2 * σ^2))

theorem number_of_people_in_range :
  let μ := 72
  let σ := 8
  let n := 20000
  let prob_1σ := 0.6827
  let prob_2σ := 0.9545
  
  let z₁ := (80 - μ) / σ
  let z₂ := (88 - μ) / σ
  
  let prob_80_88 := (prob_2σ - prob_1σ) / 2
  let num_people := prob_80_88 * n
  
  (num_people ≈ 2718) :=
sorry

end number_of_people_in_range_l17_17571


namespace grade_on_second_test_l17_17372

variable (first_test_grade second_test_average : ℕ)
#check first_test_grade
#check second_test_average

theorem grade_on_second_test :
  first_test_grade = 78 →
  second_test_average = 81 →
  (first_test_grade + (second_test_average * 2 - first_test_grade)) / 2 = second_test_average →
  second_test_grade = 84 :=
by
  intros h1 h2 h3
  sorry

end grade_on_second_test_l17_17372


namespace negation_of_proposition_l17_17461

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℝ, 2^x_0 < x_0^2) ↔ (∀ x : ℝ, 2^x ≥ x^2) :=
by sorry

end negation_of_proposition_l17_17461


namespace ellipse_tangency_construction_l17_17669

theorem ellipse_tangency_construction
  (a : ℝ)
  (e1 e2 : ℝ → Prop)  -- Representing the parallel lines as propositions
  (F1 F2 : ℝ × ℝ)  -- Foci represented as points in the plane
  (d : ℝ)  -- Distance between the parallel lines
  (angle_condition : ℝ)
  (conditions : 2 * a > d ∧ angle_condition = 1 / 3) : 
  ∃ O : ℝ × ℝ,  -- Midpoint O
    ∃ (T1 T1' T2 T2' : ℝ × ℝ),  -- Points of tangency
      (∃ E1 E2 : ℝ, e1 E1 ∧ e2 E2) ∧  -- Intersection points on the lines
      (F1.1 * (T1.1 - F1.1) + F1.2 * (T1.2 - F1.2)) / 
      (F2.1 * (T2.1 - F2.1) + F2.2 * (T2.2 - F2.2)) = 1 / 3 :=
sorry

end ellipse_tangency_construction_l17_17669
