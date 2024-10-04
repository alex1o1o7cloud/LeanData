import Mathlib

namespace min_segments_required_l253_253880

noncomputable def min_segments (n : ℕ) : ℕ := (3 * n - 2 + 1) / 2

theorem min_segments_required (n : ℕ) (h : ∀ (A B : ℕ) (hA : A < n) (hB : B < n) (hAB : A ≠ B), 
  ∃ (C : ℕ), C < n ∧ (C ≠ A) ∧ (C ≠ B)) : 
  min_segments n = ⌈ (3 * n - 2 : ℝ) / 2 ⌉ := 
sorry

end min_segments_required_l253_253880


namespace infinite_series_sum_eq_3_div_4_l253_253797

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253797


namespace age_of_new_person_l253_253018

theorem age_of_new_person (T : ℝ) (A : ℝ) (h : T / 20 - 4 = (T - 60 + A) / 20) : A = 40 :=
sorry

end age_of_new_person_l253_253018


namespace sum_geometric_series_l253_253712

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253712


namespace sum_of_values_satisfying_eq_l253_253043

theorem sum_of_values_satisfying_eq (x : ℝ) :
  (x^2 - 5 * x + 5 = 16) → ∀ r s : ℝ, (r + s = 5) :=
by
  sorry  -- Proof is omitted, looking to verify the structure only.

end sum_of_values_satisfying_eq_l253_253043


namespace series_sum_l253_253735

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253735


namespace library_visitors_on_sundays_l253_253503

theorem library_visitors_on_sundays 
  (average_other_days : ℕ) 
  (average_per_day : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ) 
  (total_visitors_month : ℕ)
  (visitors_other_days : ℕ) 
  (total_visitors_sundays : ℕ) :
  average_other_days = 240 →
  average_per_day = 285 →
  total_days = 30 →
  sundays = 5 →
  other_days = total_days - sundays →
  total_visitors_month = average_per_day * total_days →
  visitors_other_days = average_other_days * other_days →
  total_visitors_sundays + visitors_other_days = total_visitors_month →
  total_visitors_sundays = sundays * (510 : ℕ) :=
by
  sorry


end library_visitors_on_sundays_l253_253503


namespace smallest_n_terminating_decimal_l253_253978

theorem smallest_n_terminating_decimal : ∃ n : ℕ, (∀ m : ℕ, m < n → (∀ k : ℕ, (n = 103 + k) → (∃ a b : ℕ, k = 2^a * 5^b)) → (k ≠ 0 → k = 125)) ∧ n = 22 := 
sorry

end smallest_n_terminating_decimal_l253_253978


namespace correct_choice_is_C_l253_253045

-- Define the proposition C.
def prop_C : Prop := ∃ x : ℝ, |x - 1| < 0

-- The problem statement in Lean 4.
theorem correct_choice_is_C : ¬ prop_C :=
by
  sorry

end correct_choice_is_C_l253_253045


namespace dominic_domino_problem_l253_253386

theorem dominic_domino_problem 
  (num_dominoes : ℕ)
  (pips_pairs : ℕ → ℕ)
  (hexagonal_ring : ℕ → ℕ → Prop) : 
  ∀ (adj : ℕ → ℕ → Prop), 
  num_dominoes = 6 → 
  (∀ i j, hexagonal_ring i j → pips_pairs i = pips_pairs j) →
  ∃ k, k = 2 :=
by {
  sorry
}

end dominic_domino_problem_l253_253386


namespace intersection_complement_l253_253147

def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}
def compl_U_N : Set ℕ := {x ∈ U | x ∉ N}

theorem intersection_complement :
  M ∩ compl_U_N = {4} :=
by
  have h1 : compl_U_N = {2, 4, 8} := by sorry
  have h2 : M ∩ compl_U_N = {4} := by sorry
  exact h2

end intersection_complement_l253_253147


namespace sheep_daddy_input_l253_253022

-- Conditions for black box transformations
def black_box (k : ℕ) : ℕ :=
  if k % 2 = 1 then 4 * k + 1 else k / 2

-- The transformation chain with three black boxes
def black_box_chain (k : ℕ) : ℕ :=
  black_box (black_box (black_box k))

-- Theorem statement capturing the problem:
-- Final output m is 2, and the largest input leading to this is 64.
theorem sheep_daddy_input : ∃ k : ℕ, ∀ (k1 k2 k3 k4 : ℕ), 
  black_box_chain k1 = 2 ∧ 
  black_box_chain k2 = 2 ∧ 
  black_box_chain k3 = 2 ∧ 
  black_box_chain k4 = 2 ∧ 
  k1 ≠ k2 ∧ k2 ≠ k3 ∧ k3 ≠ k4 ∧ k4 ≠ k1 ∧ 
  k = max k1 (max k2 (max k3 k4)) → k = 64 :=
sorry  -- Proof is not required

end sheep_daddy_input_l253_253022


namespace sqrt_extraction_count_l253_253460

theorem sqrt_extraction_count (p : ℕ) [Fact p.Prime] : 
    ∃ k, k = (p + 1) / 2 ∧ ∀ n < p, ∃ x < p, x^2 ≡ n [MOD p] ↔ n < k := 
by
  sorry

end sqrt_extraction_count_l253_253460


namespace series_result_l253_253838

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253838


namespace quadratic_k_value_l253_253268

theorem quadratic_k_value (a b c k : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : 4 * b * b - k * a * c = 0): 
  k = 16 / 3 :=
by
  sorry

end quadratic_k_value_l253_253268


namespace smallest_n_terminating_decimal_l253_253977

theorem smallest_n_terminating_decimal : 
  ∃ (n : ℕ), (n > 0) ∧ (∀ p, prime p → p ∣ (n + 103) → (p = 2 ∨ p = 5)) ∧ (n = 22) := 
by
  sorry

end smallest_n_terminating_decimal_l253_253977


namespace reduced_price_l253_253653

variable (P R : ℝ)
variable (price_reduction : R = 0.75 * P)
variable (buy_more_oil : 700 / R = 700 / P + 5)

theorem reduced_price (non_zero_P : P ≠ 0) (non_zero_R : R ≠ 0) : R = 35 := 
by
  sorry

end reduced_price_l253_253653


namespace area_is_12_l253_253284

-- Definitions based on conditions
def isosceles_triangle (a b m : ℝ) : Prop :=
  a = b ∧ m > 0 ∧ a > 0

def median (height base_length : ℝ) : Prop :=
  height > 0 ∧ base_length > 0

noncomputable def area_of_isosceles_triangle_with_given_median (a m : ℝ) : ℝ :=
  let base_half := Real.sqrt (a^2 - m^2)
  let base := 2 * base_half
  (1 / 2) * base * m

-- Prove that the area of the isosceles triangle is correct given conditions
theorem area_is_12 :
  ∀ (a m : ℝ), isosceles_triangle a a m → median m (2 * Real.sqrt (a^2 - m^2)) → area_of_isosceles_triangle_with_given_median a m = 12 := 
by
  intros a m hiso hmed
  sorry  -- Proof steps are omitted

end area_is_12_l253_253284


namespace expected_number_of_ones_on_three_dice_l253_253955

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l253_253955


namespace number_of_tiles_l253_253994

theorem number_of_tiles (floor_length : ℝ) (floor_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) 
  (h1 : floor_length = 9) 
  (h2 : floor_width = 12) 
  (h3 : tile_length = 1 / 2) 
  (h4 : tile_width = 2 / 3) 
  : (floor_length * floor_width) / (tile_length * tile_width) = 324 := 
by
  sorry

end number_of_tiles_l253_253994


namespace distinct_solutions_equation_number_of_solutions_a2019_l253_253346

theorem distinct_solutions_equation (a : ℕ) (ha : a > 1) : 
  ∃ (x y : ℕ), (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / (a : ℚ)) ∧ x > 0 ∧ y > 0 ∧ (x ≠ y) ∧ 
  ∃ (x₁ y₁ x₂ y₂ : ℕ), (1 / (x₁ : ℚ) + 1 / (y₁ : ℚ) = 1 / (a : ℚ)) ∧
  (1 / (x₂ : ℚ) + 1 / (y₂ : ℚ) = 1 / (a : ℚ)) ∧
  x₁ ≠ y₁ ∧ x₂ ≠ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) := 
sorry

theorem number_of_solutions_a2019 :
  ∃ n, n = (3 * 3) := 
by {
  -- use 2019 = 3 * 673 and divisor count
  sorry 
}

end distinct_solutions_equation_number_of_solutions_a2019_l253_253346


namespace initial_velocity_l253_253219

noncomputable def displacement (t : ℝ) : ℝ := 3 * t - t^2

theorem initial_velocity :
  (deriv displacement 0) = 3 :=
by
  sorry

end initial_velocity_l253_253219


namespace average_visitors_on_Sundays_l253_253506

theorem average_visitors_on_Sundays (S : ℕ) (h1 : 30 = 5 + 25) (h2 : 25 * 240 + 5 * S = 30 * 285) :
  S = 510 := sorry

end average_visitors_on_Sundays_l253_253506


namespace purely_imaginary_number_eq_l253_253006

theorem purely_imaginary_number_eq (z : ℂ) (a : ℝ) (i : ℂ) (h_imag : z.im = 0 ∧ z = 0 ∧ (3 - i) * z = a + i + i) :
  a = 1 / 3 :=
  sorry

end purely_imaginary_number_eq_l253_253006


namespace sum_binomial_coeffs_equal_sum_k_values_l253_253091

theorem sum_binomial_coeffs_equal (k : ℕ) 
  (h1 : nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  k = 6 ∨ k = 20 := sorry

theorem sum_k_values (k : ℕ) (h1 :
  nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  6 + 20 = 26 := by 
  have h : k = 6 ∨ k = 20 := sum_binomial_coeffs_equal k h1 h2
  sorry

end sum_binomial_coeffs_equal_sum_k_values_l253_253091


namespace machines_work_together_l253_253035

theorem machines_work_together (x : ℝ) (h_pos : 0 < x) :
  (1 / (x + 2) + 1 / (x + 3) + 1 / (x + 1) = 1 / x) → x = 1 :=
by
  sorry

end machines_work_together_l253_253035


namespace james_bought_dirt_bikes_l253_253440

variable (D : ℕ)

-- Definitions derived from conditions
def cost_dirt_bike := 150
def cost_off_road_vehicle := 300
def registration_fee := 25
def num_off_road_vehicles := 4
def total_paid := 1825

-- Auxiliary definitions
def total_cost_dirt_bike := cost_dirt_bike + registration_fee
def total_cost_off_road_vehicle := cost_off_road_vehicle + registration_fee
def total_cost_off_road_vehicles := num_off_road_vehicles * total_cost_off_road_vehicle
def total_cost_dirt_bikes := total_paid - total_cost_off_road_vehicles

-- The final statement we need to prove
theorem james_bought_dirt_bikes : D = total_cost_dirt_bikes / total_cost_dirt_bike ↔ D = 3 := by
  sorry

end james_bought_dirt_bikes_l253_253440


namespace extremum_at_one_eq_a_one_l253_253567

theorem extremum_at_one_eq_a_one 
  (a : ℝ) 
  (h : ∃ f' : ℝ → ℝ, (∀ x, f' x = 3 * a * x^2 - 3) ∧ f' 1 = 0) : 
  a = 1 :=
sorry

end extremum_at_one_eq_a_one_l253_253567


namespace pascal_triangle_row_num_l253_253980

theorem pascal_triangle_row_num (n k : ℕ) (hn : n = 50) (hk : k = 2) : 
  nat.choose 50 2 = 1225 :=
by
  rw [nat.choose, hn, hk]
  sorry

end pascal_triangle_row_num_l253_253980


namespace additional_time_to_fill_l253_253670

variable (capacity : ℝ) -- The full capacity of the pool
variable (PA PB : ℝ) -- Rates of pipes A and B
variable (time_simultaneous : ℝ)

noncomputable def additional_time (PA PB capacity : ℝ) : ℝ :=
  let initial_capacity := (1 / 18) * capacity
  let middle_capacity := (2 / 9) * capacity
  let rateA := PA
  let rateB := PB
  let timeA := 81 -- Pipe A continues alone
  let timeB := 49 -- Pipe B continues alone
  let X := 63 -- Simultaneous time initially

  let filled_A := PA * X
  let filled_B := PB * X

  let total_fill_A := rateA * timeA
  let total_fill_B := rateB * timeB
 
  let partial_fill_together := (1/6) * capacity -- From fraction conversion: [2/9 - 1/18]
 
  let remaining_fill := capacity - (initial_capacity + 2 * partial_fill_together)
  let combined_rate := PA + PB

  let additional_time := 231 -- Time for 11/18 capacity for both pipes

  additional_time

theorem additional_time_to_fill 
  (capacity : ℝ) -- Pool capacity
  (PA PB : ℝ) -- Rates of pipes A and B
  (time_simultaneous : 63 = additional_time PA PB capacity) -- Condition of simultaneous fill time
: additional_time PA PB capacity = 231 :=
by
  sorry

end additional_time_to_fill_l253_253670


namespace sum_series_eq_3_div_4_l253_253828

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253828


namespace factory_production_l253_253588

theorem factory_production (y x : ℝ) (h1 : y + 40 * x = 1.2 * y) (h2 : y + 0.6 * y * x = 2.5 * y) 
  (hx : x = 2.5) : y = 500 ∧ 1 + x = 3.5 :=
by
  sorry

end factory_production_l253_253588


namespace c_left_before_completion_l253_253202

def a_one_day_work : ℚ := 1 / 24
def b_one_day_work : ℚ := 1 / 30
def c_one_day_work : ℚ := 1 / 40
def total_work_completed (days : ℚ) : Prop := days = 11

theorem c_left_before_completion (days_left : ℚ) (h : total_work_completed 11) :
  (11 - days_left) * (a_one_day_work + b_one_day_work + c_one_day_work) +
  (days_left * (a_one_day_work + b_one_day_work)) = 1 :=
sorry

end c_left_before_completion_l253_253202


namespace sum_of_series_eq_three_fourths_l253_253781

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253781


namespace janet_needs_more_money_l253_253441

theorem janet_needs_more_money :
  let janet_savings := 2225
  let monthly_rent := 1250
  let months_in_advance := 2
  let deposit := 500
  let total_required := (monthly_rent * months_in_advance) + deposit
  let additional_money_needed := total_required - janet_savings
  in additional_money_needed = 775 :=
by
  let janet_savings := 2225
  let monthly_rent := 1250
  let months_in_advance := 2
  let deposit := 500
  let total_required := (monthly_rent * months_in_advance) + deposit
  let additional_money_needed := total_required - janet_savings
  have h1 : total_required = 3000, by sorry
  have h2 : additional_money_needed = total_required - janet_savings, by sorry
  have h3 : total_required - janet_savings = 775, by sorry
  exact Eq.trans h2 h3

end janet_needs_more_money_l253_253441


namespace arrangement_count_example_l253_253481

theorem arrangement_count_example 
  (teachers : Finset String) 
  (students : Finset String) 
  (locations : Finset String) 
  (h_teachers : teachers.card = 2) 
  (h_students : students.card = 4) 
  (h_locations : locations.card = 2)
  : ∃ n : ℕ, n = 12 := 
sorry

end arrangement_count_example_l253_253481


namespace evaluate_series_sum_l253_253765

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253765


namespace union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l253_253401

open Set

def setA : Set ℝ := {x | -4 < x ∧ x < 2}
def setB : Set ℝ := {x | x < -5 ∨ x > 1}
def setComplementB : Set ℝ := {x | -5 ≤ x ∧ x ≤ 1}

theorem union_of_A_and_B : setA ∪ setB = {x | x < -5 ∨ x > -4} := by
  sorry

theorem intersection_of_A_and_complementB : setA ∩ setComplementB = {x | -4 < x ∧ x ≤ 1} := by
  sorry

noncomputable def setC (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m + 1}

theorem range_of_m (m : ℝ) (h : setB ∩ (setC m) = ∅) : -4 ≤ m ∧ m ≤ 0 := by
  sorry

end union_of_A_and_B_intersection_of_A_and_complementB_range_of_m_l253_253401


namespace expected_number_of_ones_l253_253949

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l253_253949


namespace eggs_per_snake_l253_253912

-- Define the conditions
def num_snakes : ℕ := 3
def price_regular : ℕ := 250
def price_super_rare : ℕ := 1000
def total_revenue : ℕ := 2250

-- Prove for the number of eggs each snake lays
theorem eggs_per_snake (E : ℕ) 
  (h1 : E * (num_snakes - 1) * price_regular + E * price_super_rare = total_revenue) : 
  E = 2 :=
sorry

end eggs_per_snake_l253_253912


namespace solve_inequality_l253_253627

theorem solve_inequality : 
  {x : ℝ | -x^2 - 2*x + 3 ≤ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end solve_inequality_l253_253627


namespace infinite_series_sum_eq_3_over_4_l253_253847

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253847


namespace ratio_of_fifth_to_second_l253_253033

-- Definitions based on the conditions
def first_stack := 7
def second_stack := first_stack + 3
def third_stack := second_stack - 6
def fourth_stack := third_stack + 10

def total_blocks := 55

-- The number of blocks in the fifth stack
def fifth_stack := total_blocks - (first_stack + second_stack + third_stack + fourth_stack)

-- The ratio of the fifth stack to the second stack
def ratio := fifth_stack / second_stack

-- The theorem we want to prove
theorem ratio_of_fifth_to_second: ratio = 2 := by
  sorry

end ratio_of_fifth_to_second_l253_253033


namespace expected_ones_three_dice_l253_253970

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l253_253970


namespace solution_exists_l253_253373

def valid_grid (grid : List (List Nat)) : Prop :=
  grid = [[2, 3, 6], [6, 3, 2]] ∨
  grid = [[2, 4, 8], [8, 4, 2]]

theorem solution_exists :
  ∃ (grid : List (List Nat)), valid_grid grid := by
  sorry

end solution_exists_l253_253373


namespace series_sum_l253_253743

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253743


namespace min_value_2a_3b_equality_case_l253_253260

theorem min_value_2a_3b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) : 
  2 * a + 3 * b ≥ 25 :=
sorry

theorem equality_case (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) :
  (a = 5) ∧ (b = 5) → 2 * a + 3 * b = 25 :=
sorry

end min_value_2a_3b_equality_case_l253_253260


namespace lines_coplanar_parameter_l253_253617

/-- 
  Two lines are given in parametric form: 
  L1: (2 + 2s, 4s, -3 + rs)
  L2: (-1 + 3t, 2t, 1 + 2t)
  Prove that if these lines are coplanar, then r = 4.
-/
theorem lines_coplanar_parameter (s t r : ℝ) :
  ∃ (k : ℝ), 
  (∀ s t, 
    ∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0
      ∧
      (2 + 2 * s, 4 * s, -3 + r * s) = (k * (-1 + 3 * t), k * 2 * t, k * (1 + 2 * t))
  ) → r = 4 := sorry

end lines_coplanar_parameter_l253_253617


namespace find_num_round_balloons_l253_253913

variable (R : ℕ) -- Number of bags of round balloons that Janeth bought
variable (RoundBalloonsPerBag : ℕ := 20)
variable (LongBalloonsPerBag : ℕ := 30)
variable (BagsLongBalloons : ℕ := 4)
variable (BurstRoundBalloons : ℕ := 5)
variable (BalloonsLeft : ℕ := 215)

def total_long_balloons : ℕ := BagsLongBalloons * LongBalloonsPerBag
def total_balloons : ℕ := R * RoundBalloonsPerBag + total_long_balloons - BurstRoundBalloons

theorem find_num_round_balloons :
  BalloonsLeft = total_balloons → R = 5 := by
  sorry

end find_num_round_balloons_l253_253913


namespace cylinder_side_surface_area_l253_253501

-- Define the given conditions
def base_circumference : ℝ := 4
def height_of_cylinder : ℝ := 4

-- Define the relation we need to prove
theorem cylinder_side_surface_area : 
  base_circumference * height_of_cylinder = 16 := 
by
  sorry

end cylinder_side_surface_area_l253_253501


namespace no_such_function_exists_l253_253014

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f^[n] n = n + 1 :=
by
  sorry

end no_such_function_exists_l253_253014


namespace muffin_to_banana_ratio_l253_253917

variables (m b : ℝ) -- initial cost of a muffin and a banana

-- John's total cost for muffins and bananas
def johns_cost (m b : ℝ) : ℝ :=
  3 * m + 4 * b

-- Martha's total cost for muffins and bananas based on increased prices
def marthas_cost_increased (m b : ℝ) : ℝ :=
  5 * (1.2 * m) + 12 * (1.5 * b)

-- John's total cost times three
def marthas_cost_original_times_three (m b : ℝ) : ℝ :=
  3 * (johns_cost m b)

-- The theorem to prove
theorem muffin_to_banana_ratio
  (h3m4b_eq : johns_cost m b * 3 = marthas_cost_increased m b)
  (hm_eq_2b : m = 2 * b) :
  (1.2 * m) / (1.5 * b) = 4 / 5 := by
  sorry

end muffin_to_banana_ratio_l253_253917


namespace remainder_n_sq_plus_3n_5_mod_25_l253_253900

theorem remainder_n_sq_plus_3n_5_mod_25 (k : ℤ) (n : ℤ) (h : n = 25 * k - 1) : 
  (n^2 + 3 * n + 5) % 25 = 3 := 
by
  sorry

end remainder_n_sq_plus_3n_5_mod_25_l253_253900


namespace total_students_l253_253907

-- Define the conditions
def ratio_girls_boys (G B : ℕ) : Prop := G / B = 1 / 2
def ratio_math_girls (M N : ℕ) : Prop := M / N = 3 / 1
def ratio_sports_boys (S T : ℕ) : Prop := S / T = 4 / 1

-- Define the problem statement
theorem total_students (G B M N S T : ℕ) 
  (h1 : ratio_girls_boys G B)
  (h2 : ratio_math_girls M N)
  (h3 : ratio_sports_boys S T)
  (h4 : M = 12)
  (h5 : G = M + N)
  (h6 : G = 16) 
  (h7 : B = 32) : 
  G + B = 48 :=
sorry

end total_students_l253_253907


namespace ernaldo_friends_count_l253_253207

-- Define the members of the group
inductive Member
| Arnaldo
| Bernaldo
| Cernaldo
| Dernaldo
| Ernaldo

open Member

-- Define the number of friends for each member
def number_of_friends : Member → ℕ
| Arnaldo  => 1
| Bernaldo => 2
| Cernaldo => 3
| Dernaldo => 4
| Ernaldo  => 0  -- This will be our unknown to solve

-- The main theorem we need to prove
theorem ernaldo_friends_count : number_of_friends Ernaldo = 2 :=
sorry

end ernaldo_friends_count_l253_253207


namespace janes_score_is_110_l253_253463

-- Definitions and conditions
def sarah_score_condition (x y : ℕ) : Prop := x = y + 50
def average_score_condition (x y : ℕ) : Prop := (x + y) / 2 = 110
def janes_score (x y : ℕ) : ℕ := (x + y) / 2

-- The proof problem statement
theorem janes_score_is_110 (x y : ℕ) 
  (h_sarah : sarah_score_condition x y) 
  (h_avg   : average_score_condition x y) : 
  janes_score x y = 110 := 
by
  sorry

end janes_score_is_110_l253_253463


namespace two_color_K6_contains_monochromatic_triangle_l253_253425

theorem two_color_K6_contains_monochromatic_triangle (V : Type) [Fintype V] [DecidableEq V]
  (hV : Fintype.card V = 6)
  (color : V → V → Fin 2) :
  ∃ (a b c : V), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (color a b = color b c ∧ color b c = color c a) := by
  sorry

end two_color_K6_contains_monochromatic_triangle_l253_253425


namespace ratio_five_to_one_l253_253199

theorem ratio_five_to_one (x : ℕ) : (5 : ℕ) * 13 = 1 * x → x = 65 := 
by 
  intro h
  linarith

end ratio_five_to_one_l253_253199


namespace infinite_series_sum_eq_l253_253717

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253717


namespace sum_geometric_series_l253_253705

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253705


namespace total_value_of_coins_l253_253643

theorem total_value_of_coins :
  (∀ (coins : List (String × ℕ)), coins.length = 12 →
    (∃ Q N : ℕ, 
      Q = 4 ∧ N = 8 ∧
      (∀ (coin : String × ℕ), coin ∈ coins → 
        (coin = ("quarter", Q) → Q = 4 ∧ (Q * 25 = 100)) ∧ 
        (coin = ("nickel", N) → N = 8 ∧ (N * 5 = 40)) ∧
      (Q * 25 + N * 5 = 140)))) :=
sorry

end total_value_of_coins_l253_253643


namespace series_result_l253_253837

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253837


namespace repeating_decimals_count_l253_253250

theorem repeating_decimals_count :
  { n : ℕ | 1 ≤ n ∧ n ≤ 20 ∧ (¬∃ k, n = 9 * k) }.to_finset.card = 18 :=
by
  sorry

end repeating_decimals_count_l253_253250


namespace find_number_l253_253988

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
sorry

end find_number_l253_253988


namespace price_per_sq_ft_l253_253634

def house_sq_ft : ℕ := 2400
def barn_sq_ft : ℕ := 1000
def total_property_value : ℝ := 333200

theorem price_per_sq_ft : 
  (total_property_value / (house_sq_ft + barn_sq_ft)) = 98 := 
by 
  sorry

end price_per_sq_ft_l253_253634


namespace parabola_tangent_parameter_l253_253334

theorem parabola_tangent_parameter (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hp : p ≠ 0) :
  ∃ p : ℝ, (∀ y, y^2 + (2 * p * b / a) * y + (2 * p * c^2 / a) = 0) ↔ (p = 2 * a * c^2 / b^2) := 
by
  sorry

end parabola_tangent_parameter_l253_253334


namespace irrigation_system_flow_rates_l253_253391

-- Define the conditions
variable (q0 : ℝ) -- Flow rate in channel BC

-- Channels' flow rates
variable (qAB qAH q_total : ℝ)

-- Define the conditions as hypotheses
axiom H1 : qAB = 1/2 * q0
axiom H2 : qAH = 3/4 * q0
axiom H3 : q_total = qAB + qAH

-- Prove the results
theorem irrigation_system_flow_rates
  (q0 : ℝ)
  (qAB qAH q_total : ℝ)
  (H1 : qAB = 1/2 * q0)
  (H2 : qAH = 3/4 * q0)
  (H3 : q_total = qAB + qAH) :
  qAB = 1/2 * q0 ∧ qAH = 3/4 * q0 ∧ q_total = 7/4 * q0 :=
by {
  split,
  exact H1,
  split,
  exact H2,
  rw [H3, H1, H2],
  linarith
}

end irrigation_system_flow_rates_l253_253391


namespace value_of_fraction_power_series_l253_253044

theorem value_of_fraction_power_series (x : ℕ) (h : x = 3) :
  (x^3 * x^5 * x^7 * x^9 * x^11 * x^13 * x^15 * x^17 * x^19 * x^21) /
  (x^4 * x^8 * x^12 * x^16 * x^20 * x^24) = 3^36 :=
by
  subst h
  sorry

end value_of_fraction_power_series_l253_253044


namespace functional_equation_solution_l253_253547

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y) + f (f x + f y) = y * f x + f (x + f y)) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) :=
sorry

end functional_equation_solution_l253_253547


namespace calculate_expression_l253_253395

theorem calculate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  (1 / (y + 1)) / (1 / (x + 2)) = 1 := by
  sorry

end calculate_expression_l253_253395


namespace remainder_2007_div_81_l253_253487

theorem remainder_2007_div_81 : 2007 % 81 = 63 :=
by
  sorry

end remainder_2007_div_81_l253_253487


namespace solve_for_x_l253_253335

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 ↔ x = 2 / 9 := by
  sorry

end solve_for_x_l253_253335


namespace subset_intersection_exists_l253_253448

theorem subset_intersection_exists {n : ℕ} (A : Fin (n + 1) → Finset (Fin n)) 
    (h_distinct : ∀ i j : Fin (n + 1), i ≠ j → A i ≠ A j)
    (h_size : ∀ i : Fin (n + 1), (A i).card = 3) : 
    ∃ (i j : Fin (n + 1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
by
  sorry

end subset_intersection_exists_l253_253448


namespace certain_number_is_65_l253_253221

-- Define the conditions
variables (N : ℕ)
axiom condition1 : N < 81
axiom condition2 : ∀ k : ℕ, k ≤ 15 → N + k < 81
axiom last_consecutive : N + 15 = 80

-- Prove the theorem
theorem certain_number_is_65 (h1 : N < 81) (h2 : ∀ k : ℕ, k ≤ 15 → N + k < 81) (h3 : N + 15 = 80) : N = 65 :=
sorry

end certain_number_is_65_l253_253221


namespace average_visitors_on_Sundays_l253_253508

theorem average_visitors_on_Sundays (S : ℕ) 
  (visitors_other_days : ℕ := 240)
  (avg_per_day : ℕ := 285)
  (days_in_month : ℕ := 30)
  (month_starts_with_sunday : true) :
  let num_sundays := 5
  let num_other_days := days_in_month - num_sundays
  let total_visitors := (num_sundays * S) + (num_other_days * visitors_other_days)
  total_visitors = avg_per_day * days_in_month → S = 510 := 
by
  intros _ _ _ _ _ total_visitors_eq
  sorry

end average_visitors_on_Sundays_l253_253508


namespace tan_135_eq_neg1_l253_253532

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg1_l253_253532


namespace expected_ones_in_three_dice_rolls_l253_253947

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l253_253947


namespace second_frog_hops_eq_18_l253_253180

-- Define the given conditions
variables (x : ℕ) (h3 : ℕ)

def second_frog_hops := 2 * h3
def first_frog_hops := 4 * second_frog_hops
def total_hops := h3 + second_frog_hops + first_frog_hops

-- The proof goal
theorem second_frog_hops_eq_18 (H : total_hops = 99) : second_frog_hops = 18 :=
by
  sorry

end second_frog_hops_eq_18_l253_253180


namespace correct_average_weight_l253_253472

noncomputable def initial_average_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def misread_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 66

theorem correct_average_weight : 
  (initial_average_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.9 := 
by
  sorry

end correct_average_weight_l253_253472


namespace marathon_y_distance_l253_253209

theorem marathon_y_distance (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (total_yards : ℕ) (y : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : yards_per_marathon = 312) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 8) 
  (H5 : total_yards = num_marathons * yards_per_marathon) 
  (H6 : total_yards % yards_per_mile = y) 
  (H7 : 0 ≤ y) 
  (H8 : y < yards_per_mile) : 
  y = 736 :=
by 
  sorry

end marathon_y_distance_l253_253209


namespace sum_of_first_six_terms_geometric_sequence_l253_253873

-- conditions
def a : ℚ := 1/4
def r : ℚ := 1/4

-- geometric series sum function
def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- target sum of first six terms
def S_6 : ℚ := geom_sum a r 6

-- proof statement
theorem sum_of_first_six_terms_geometric_sequence :
  S_6 = 1365 / 4096 :=
by 
  sorry

end sum_of_first_six_terms_geometric_sequence_l253_253873


namespace value_of_c_l253_253894

theorem value_of_c (a b c : ℚ) (h1 : a / b = 2 / 3) (h2 : b / c = 3 / 7) (h3 : a - b + 3 = c - 2 * b) : c = 21 / 2 :=
sorry

end value_of_c_l253_253894


namespace xiao_ming_completion_days_l253_253663

/-
  Conditions:
  1. The total number of pages is 960.
  2. The planned number of days to finish the book is 20.
  3. Xiao Ming actually read 12 more pages per day than planned.

  Question:
  How many days did it actually take Xiao Ming to finish the book?

  Answer:
  The actual number of days to finish the book is 16 days.
-/

open Nat

theorem xiao_ming_completion_days :
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  actual_days = 16 :=
by
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  show actual_days = 16
  sorry

end xiao_ming_completion_days_l253_253663


namespace gcd_ab_conditions_l253_253454

theorem gcd_ab_conditions 
  (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = 1) : 
  Nat.gcd (a + b) (a - b) = 1 ∨ Nat.gcd (a + b) (a - b) = 2 := 
sorry

end gcd_ab_conditions_l253_253454


namespace solution_set_M_minimum_value_expr_l253_253412

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

-- Proof problem (1): Prove that the solution set M of the inequality f(x) ≥ -1 is {x | 2/3 ≤ x ≤ 6}.
theorem solution_set_M : 
  { x : ℝ | f x ≥ -1 } = { x : ℝ | 2/3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Define the requirement for t and the expression to minimize
noncomputable def t : ℝ := 6
noncomputable def expr (a b c : ℝ) : ℝ := 1 / (2 * a + b) + 1 / (2 * a + c)

-- Proof problem (2): Given t = 6 and 4a + b + c = 6, prove that the minimum value of expr is 2/3.
theorem minimum_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = t) :
  expr a b c ≥ 2/3 :=
sorry

end solution_set_M_minimum_value_expr_l253_253412


namespace arithmetic_sequence_properties_l253_253559

noncomputable def arithmetic_sequence (a3 a5_a7_sum : ℝ) : Prop :=
  ∃ (a d : ℝ), a + 2*d = a3 ∧ 2*a + 10*d = a5_a7_sum

noncomputable def sequence_a_n (a d n : ℝ) : ℝ := a + (n - 1)*d

noncomputable def sum_S_n (a d n : ℝ) : ℝ := n/2 * (2*a + (n-1)*d)

noncomputable def sequence_b_n (a d n : ℝ) : ℝ := 1 / (sequence_a_n a d n ^ 2 - 1)

noncomputable def sum_T_n (a d n : ℝ) : ℝ :=
  (1 / 4) * (1 - 1/(n+1))

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 7 26) →
  (∀ n : ℕ+, sequence_a_n 3 2 n = 2 * n + 1) ∧
  (∀ n : ℕ+, sum_S_n 3 2 n = n^2 + 2 * n) ∧
  (∀ n : ℕ+, sum_T_n 3 2 n = n / (4 * (n + 1))) :=
by
  sorry

end arithmetic_sequence_properties_l253_253559


namespace tammy_speed_on_second_day_l253_253657

-- Definitions of the conditions
variables (t v : ℝ)
def total_hours := 14
def total_distance := 52

-- Distance equation
def distance_eq := v * t + (v + 0.5) * (t - 2) = total_distance

-- Time equation
def time_eq := t + (t - 2) = total_hours

theorem tammy_speed_on_second_day :
  (time_eq t ∧ distance_eq v t) → v + 0.5 = 4 :=
by sorry

end tammy_speed_on_second_day_l253_253657


namespace evaluate_series_sum_l253_253769

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253769


namespace quadratic_distinct_real_roots_l253_253887

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 = 0 → 
  (k ≠ 0 ∧ ((-2)^2 - 4 * k * (-1) > 0))) ↔ (k > -1 ∧ k ≠ 0) := 
sorry

end quadratic_distinct_real_roots_l253_253887


namespace coloring_count_is_2_l253_253318

noncomputable def count_colorings (initial_color : String) : Nat := 
  if initial_color = "R" then 2 else 0 -- Assumes only the case of initial red color is valid for simplicity

theorem coloring_count_is_2 (h1 : True) (h2 : True) (h3 : True) (h4 : True):
  count_colorings "R" = 2 := by
  sorry

end coloring_count_is_2_l253_253318


namespace n_minus_m_eq_singleton_6_l253_253382

def set_difference (A B : Set α) : Set α :=
  {x | x ∈ A ∧ x ∉ B}

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 6}

theorem n_minus_m_eq_singleton_6 : set_difference N M = {6} :=
by
  sorry

end n_minus_m_eq_singleton_6_l253_253382


namespace angle_C_of_quadrilateral_ABCD_l253_253287

theorem angle_C_of_quadrilateral_ABCD
  (AB CD BC AD : ℝ) (D : ℝ) (h_AB_CD : AB = CD) (h_BC_AD : BC = AD) (h_ang_D : D = 120) :
  ∃ C : ℝ, C = 60 :=
by
  sorry

end angle_C_of_quadrilateral_ABCD_l253_253287


namespace circle_intersection_probability_l253_253191

noncomputable def probability_circles_intersect : ℝ :=
  1

theorem circle_intersection_probability :
  ∀ (A_X B_X : ℝ), (0 ≤ A_X) → (A_X ≤ 2) → (0 ≤ B_X) → (B_X ≤ 2) →
  (∃ y, y ≥ 1 ∧ y ≤ 2) →
  ∃ p : ℝ, p = probability_circles_intersect ∧
  p = 1 :=
by
  sorry

end circle_intersection_probability_l253_253191


namespace oranges_in_bowl_l253_253134

theorem oranges_in_bowl (bananas : Nat) (apples : Nat) (pears : Nat) (total_fruits : Nat) (h_bananas : bananas = 4) (h_apples : apples = 3 * bananas) (h_pears : pears = 5) (h_total_fruits : total_fruits = 30) :
  total_fruits - (bananas + apples + pears) = 9 :=
by
  subst h_bananas
  subst h_apples
  subst h_pears
  subst h_total_fruits
  sorry

end oranges_in_bowl_l253_253134


namespace find_a_l253_253124

theorem find_a (a b c : ℕ) (h₁ : a + b = c) (h₂ : b + 2 * c = 10) (h₃ : c = 4) : a = 2 := by
  sorry

end find_a_l253_253124


namespace correct_option_l253_253495

theorem correct_option :
  (∀ (a b : ℝ),  3 * a^2 * b - 4 * b * a^2 = -a^2 * b) ∧
  ¬(1 / 7 * (-7) + (-1 / 7) * 7 = 1) ∧
  ¬((-3 / 5)^2 = 9 / 5) ∧
  ¬(∀ (a b : ℝ), 3 * a + 5 * b = 8 * a * b) :=
by
  sorry

end correct_option_l253_253495


namespace unvisited_planet_exists_l253_253145

theorem unvisited_planet_exists (n : ℕ) (h : 1 ≤ n)
  (planets : Fin (2 * n + 1) → ℝ) 
  (distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → planets i ≠ planets j) 
  (expeditions : Fin (2 * n + 1) → Fin (2 * n + 1))
  (closest : ∀ i : Fin (2 * n + 1), expeditions i = i ↔ False) :
  ∃ p : Fin (2 * n + 1), ∀ q : Fin (2 * n + 1), expeditions q ≠ p := sorry

end unvisited_planet_exists_l253_253145


namespace sum_series_eq_3_over_4_l253_253746

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253746


namespace probability_of_at_most_one_white_ball_l253_253585

open Nat

-- Definitions based on conditions in a)
def black_balls : ℕ := 10
def red_balls : ℕ := 12
def white_balls : ℕ := 3
def total_balls : ℕ := black_balls + red_balls + white_balls
def select_balls : ℕ := 3

-- The combinatorial function C(n, k) as defined in combinatorics
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Defining the expression and correct answer
def expr : ℚ := (C white_balls 1 * C (black_balls + red_balls) 2 + C (black_balls + red_balls) 3 : ℚ) / (C total_balls 3 : ℚ)
def correct_answer : ℚ := (C white_balls 0 * C (black_balls + red_balls) 3 + C white_balls 1 * C (black_balls + red_balls) 2 : ℚ) / (C total_balls 3 : ℚ)

-- Lean 4 theorem statement
theorem probability_of_at_most_one_white_ball :
  expr = correct_answer := sorry

end probability_of_at_most_one_white_ball_l253_253585


namespace two_thirds_greater_l253_253933

theorem two_thirds_greater :
  let epsilon : ℚ := (2 : ℚ) / (3 * 10^8)
  let decimal_part : ℚ := 66666666 / 10^8
  (2 / 3) - decimal_part = epsilon := by
  sorry

end two_thirds_greater_l253_253933


namespace expected_ones_three_standard_dice_l253_253959

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l253_253959


namespace smallest_AAB_value_l253_253512

theorem smallest_AAB_value (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (hAB : 10 * A + B = (1 / 7) * (110 * A + B)) : 110 * A + B = 332 :=
by
  have h1 : 70 * A + 6 * B = 110 * A, from sorry
  have h2 : 6 * B = 40 * A, from sorry
  have h3 : 3 * B = 20 * A, from sorry
  have h4 : B = (20 // 3) * A, from sorry
  have smallest_A : A = 3, from sorry
  have smallest_B : B = 2, from sorry
  show 110 * A + B = 332, from sorry

end smallest_AAB_value_l253_253512


namespace shara_shells_l253_253303

def initial_shells : ℕ := 20
def first_vacation_day1_3 : ℕ := 5 * 3
def first_vacation_day4 : ℕ := 6
def second_vacation_day1_2 : ℕ := 4 * 2
def second_vacation_day3 : ℕ := 7
def third_vacation_day1 : ℕ := 8
def third_vacation_day2 : ℕ := 4
def third_vacation_day3_4 : ℕ := 3 * 2

def total_shells : ℕ :=
  initial_shells + 
  (first_vacation_day1_3 + first_vacation_day4) +
  (second_vacation_day1_2 + second_vacation_day3) + 
  (third_vacation_day1 + third_vacation_day2 + third_vacation_day3_4)

theorem shara_shells : total_shells = 74 :=
by
  sorry

end shara_shells_l253_253303


namespace fraction_calculation_l253_253939

theorem fraction_calculation : (36 - 12) / (12 - 4) = 3 :=
by
  sorry

end fraction_calculation_l253_253939


namespace spurs_total_basketballs_l253_253309

theorem spurs_total_basketballs (players : ℕ) (basketballs_per_player : ℕ) (h1 : players = 22) (h2 : basketballs_per_player = 11) : players * basketballs_per_player = 242 := by
  sorry

end spurs_total_basketballs_l253_253309


namespace series_result_l253_253844

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253844


namespace infinite_series_sum_eq_3_div_4_l253_253802

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253802


namespace infinite_series_sum_eq_l253_253720

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253720


namespace quadratic_expression_positive_l253_253387

theorem quadratic_expression_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5) * x + k + 2 > 0) ↔ (7 - 4 * Real.sqrt 2 < k ∧ k < 7 + 4 * Real.sqrt 2) :=
by
  sorry

end quadratic_expression_positive_l253_253387


namespace series_converges_to_three_fourths_l253_253701

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253701


namespace solve_system_of_equations_l253_253168

def solution_set : Set (ℝ × ℝ) := {(0, 0), (-1, 1), (-2 / (3^(1/3)), -2 * (3^(1/3)))}

theorem solve_system_of_equations (x y : ℝ) :
  (x * y^2 - 2 * y + 3 * x^2 = 0 ∧ y^2 + x^2 * y + 2 * x = 0) ↔ (x, y) ∈ solution_set := sorry

end solve_system_of_equations_l253_253168


namespace probability_B_win_probability_game_ends_with_B_shot_2_balls_l253_253989

-- Define the probability of A and B making a shot.
def p_A : ℚ := 1 / 3
def p_B : ℚ := 1 / 2

-- Define the condition that each shot is independent; in Lean, 
-- this can be assumed inherently, so we do not define independence explicitly.

-- Problem 1: Prove that the probability that B wins.
theorem probability_B_win : (p_A * p_B) + (p_A^2 * p_B^2 * p_B) + (p_A^3 * p_B^2 * p_B^2 * p_B) = 13 / 27 :=
begin
  sorry
end

-- Problem 2: Prove that the probability that the game ends with B having shot only 2 balls.
theorem probability_game_ends_with_B_shot_2_balls :
  (p_A^2 * p_B * 1/2) + (1/2 * 1/2 * p_A^2 * 1/3) = 4 / 27 :=
begin
  sorry
end

end probability_B_win_probability_game_ends_with_B_shot_2_balls_l253_253989


namespace betty_height_in_feet_l253_253685

theorem betty_height_in_feet (dog_height carter_height betty_height : ℕ) (h1 : dog_height = 24) 
  (h2 : carter_height = 2 * dog_height) (h3 : betty_height = carter_height - 12) : betty_height / 12 = 3 :=
by
  sorry

end betty_height_in_feet_l253_253685


namespace total_paint_area_l253_253614

structure Room where
  length : ℕ
  width : ℕ
  height : ℕ

def livingRoom : Room := { length := 40, width := 40, height := 10 }
def bedroom : Room := { length := 12, width := 10, height := 10 }

def wallArea (room : Room) (n_walls : ℕ) : ℕ :=
  let longWallsArea := 2 * (room.length * room.height)
  let shortWallsArea := 2 * (room.width * room.height)
  if n_walls <= 2 then
    longWallsArea * n_walls / 2
  else if n_walls <= 4 then
    longWallsArea + (shortWallsArea * (n_walls - 2) / 2)
  else
    0

def totalWallArea (livingRoom : Room) (bedroom : Room) (n_livingRoomWalls n_bedroomWalls : ℕ) : ℕ :=
  wallArea livingRoom n_livingRoomWalls + wallArea bedroom n_bedroomWalls

theorem total_paint_area : totalWallArea livingRoom bedroom 3 4 = 1640 := by
  sorry

end total_paint_area_l253_253614


namespace solve_for_a_l253_253254

theorem solve_for_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : x - a * y = 3) : a = -1 :=
sorry

end solve_for_a_l253_253254


namespace return_time_possibilities_l253_253360

variables (d v w : ℝ) (t_return : ℝ)

-- Condition 1: Flight against wind takes 84 minutes
axiom flight_against_wind : d / (v - w) = 84

-- Condition 2: Return trip with wind takes 9 minutes less than without wind
axiom return_wind_condition : d / (v + w) = d / v - 9

-- Problem Statement: Find the possible return times
theorem return_time_possibilities :
  t_return = d / (v + w) → t_return = 63 ∨ t_return = 12 :=
sorry

end return_time_possibilities_l253_253360


namespace sum_series_div_3_powers_l253_253758

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253758


namespace sum_of_consecutive_integers_l253_253205

theorem sum_of_consecutive_integers (a : ℤ) (n : ℕ) (h : a = -49) (h_n : n = 100) 
  : ∑ i in Finset.range n, (a + i : ℤ) = 50 := by
  sorry

end sum_of_consecutive_integers_l253_253205


namespace infinite_series_sum_eq_3_over_4_l253_253846

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253846


namespace range_of_a_l253_253162

/-- 
Proof problem statement derived from the given math problem and solution:
Prove that if the conditions:
1. ∀ x > 0, x + 1/x > a
2. ∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0
3. ¬ (∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0) is false
4. (∀ x > 0, x + 1/x > a) ∧ (∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0) is false
hold, then a ≥ 2.
-/
theorem range_of_a (a : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → x + 1 / x > a)
  (h2 : ∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0)
  (h3 : ¬ (¬ (∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0)))
  (h4 : ¬ ((∀ x : ℝ, x > 0 → x + 1 / x > a) ∧ (∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0))) :
  a ≥ 2 :=
sorry

end range_of_a_l253_253162


namespace sum_series_equals_three_fourths_l253_253822

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253822


namespace expected_number_of_ones_when_three_dice_rolled_l253_253960

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l253_253960


namespace expected_ones_on_three_dice_l253_253964

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l253_253964


namespace total_number_of_baseball_cards_l253_253446

def baseball_cards_total : Nat :=
  let carlos := 20
  let matias := carlos - 6
  let jorge := matias
  carlos + matias + jorge
   
theorem total_number_of_baseball_cards :
  baseball_cards_total = 48 :=
by
  rfl

end total_number_of_baseball_cards_l253_253446


namespace cubeRootThree_expression_value_l253_253893

-- Define the approximate value of cube root of 3
def cubeRootThree : ℝ := 1.442

-- Lean theorem statement
theorem cubeRootThree_expression_value :
  cubeRootThree - 3 * cubeRootThree - 98 * cubeRootThree = -144.2 := by
  sorry

end cubeRootThree_expression_value_l253_253893


namespace sum_of_series_eq_three_fourths_l253_253777

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253777


namespace find_abc_l253_253659

noncomputable def log (x : ℝ) : ℝ := sorry -- Replace sorry with an actual implementation of log function if needed

theorem find_abc (a b c : ℝ) 
    (h1 : 1 ≤ a) 
    (h2 : 1 ≤ b) 
    (h3 : 1 ≤ c)
    (h4 : a * b * c = 10)
    (h5 : a^(log a) * b^(log b) * c^(log c) ≥ 10) :
    (a = 1 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 10) := 
by
  sorry

end find_abc_l253_253659


namespace number_of_possible_outcomes_l253_253178

theorem number_of_possible_outcomes : 
  ∃ n : ℕ, n = 30 ∧
  ∀ (total_shots successful_shots consecutive_hits : ℕ),
  total_shots = 8 ∧ successful_shots = 3 ∧ consecutive_hits = 2 →
  n = 30 := 
by
  sorry

end number_of_possible_outcomes_l253_253178


namespace basketball_teams_l253_253926

theorem basketball_teams (boys girls : ℕ) (total_players : ℕ) (team_size : ℕ) (ways : ℕ) :
  boys = 7 → girls = 3 → total_players = 10 → team_size = 5 → ways = 105 → 
  ∃ (girls_in_team1 girls_in_team2 : ℕ), 
    girls_in_team1 + girls_in_team2 = 3 ∧ 
    1 ≤ girls_in_team1 ∧ 
    1 ≤ girls_in_team2 ∧ 
    girls_in_team1 ≠ 0 ∧ 
    girls_in_team2 ≠ 0 ∧ 
    ways = 105 :=
by 
  sorry

end basketball_teams_l253_253926


namespace infinite_series_sum_eq_3_div_4_l253_253800

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253800


namespace num_repeating_decimals_between_1_and_20_l253_253248

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end num_repeating_decimals_between_1_and_20_l253_253248


namespace tan_135_eq_neg_one_l253_253534

theorem tan_135_eq_neg_one : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg_one_l253_253534


namespace infinite_series_sum_eq_3_div_4_l253_253795

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253795


namespace percent_of_b_l253_253127

variables (a b c : ℝ)

theorem percent_of_b (h1 : c = 0.30 * a) (h2 : b = 1.20 * a) : c = 0.25 * b :=
by sorry

end percent_of_b_l253_253127


namespace person_speed_approx_l253_253359

noncomputable def convertDistance (meters : ℝ) : ℝ := meters * 0.000621371
noncomputable def convertTime (minutes : ℝ) (seconds : ℝ) : ℝ := (minutes + (seconds / 60)) / 60
noncomputable def calculateSpeed (distance_miles : ℝ) (time_hours : ℝ) : ℝ := distance_miles / time_hours

theorem person_speed_approx (street_length_meters : ℝ) (time_min : ℝ) (time_sec : ℝ) :
  street_length_meters = 900 →
  time_min = 3 →
  time_sec = 20 →
  abs ((calculateSpeed (convertDistance street_length_meters) (convertTime time_min time_sec)) - 10.07) < 0.01 :=
by
  sorry

end person_speed_approx_l253_253359


namespace polynomial_remainder_l253_253338

noncomputable def f (x : ℝ) : ℝ := x^4 + 2 * x^2 - 3
noncomputable def g (x : ℝ) : ℝ := x^2 + x - 2
noncomputable def r (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 3

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = g x * q x + r x :=
sorry

end polynomial_remainder_l253_253338


namespace sum_geometric_series_l253_253711

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253711


namespace lily_milk_left_l253_253154

theorem lily_milk_left : 
  let initial_milk := 5 
  let given_to_james := 18 / 7
  ∃ r : ℚ, r = 2 + 3/7 ∧ (initial_milk - given_to_james) = r :=
by
  sorry

end lily_milk_left_l253_253154


namespace average_percentage_decrease_l253_253927

theorem average_percentage_decrease : 
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ ((2000 * (1 - x)^2 = 1280) ↔ (x = 0.18)) :=
by
  sorry

end average_percentage_decrease_l253_253927


namespace infinite_series_sum_eq_3_over_4_l253_253852

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253852


namespace correct_operation_l253_253494

variable (a : ℕ)

theorem correct_operation :
  (3 * a + 2 * a ≠ 5 * a^2) ∧
  (3 * a - 2 * a ≠ 1) ∧
  a^2 * a^3 = a^5 ∧
  (a / a^2 ≠ a) :=
by
  sorry

end correct_operation_l253_253494


namespace inequality_proof_l253_253882

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 := 
sorry

end inequality_proof_l253_253882


namespace kate_change_is_correct_l253_253918

-- Define prices of items
def gum_price : ℝ := 0.89
def chocolate_price : ℝ := 1.25
def chips_price : ℝ := 2.49

-- Define sales tax rate
def tax_rate : ℝ := 0.06

-- Define the total money Kate gave to the clerk
def payment : ℝ := 10.00

-- Define total cost of items before tax
def total_before_tax := gum_price + chocolate_price + chips_price

-- Define the sales tax
def sales_tax := tax_rate * total_before_tax

-- Define the correct answer for total cost
def total_cost := total_before_tax + sales_tax

-- Define the correct amount of change Kate should get back
def change := payment - total_cost

theorem kate_change_is_correct : abs (change - 5.09) < 0.01 :=
by
  sorry

end kate_change_is_correct_l253_253918


namespace audrey_sleep_time_l253_253518

theorem audrey_sleep_time (T : ℝ) (h1 : (3 / 5) * T = 6) : T = 10 :=
by
  sorry

end audrey_sleep_time_l253_253518


namespace factor_x12_minus_4096_l253_253526

theorem factor_x12_minus_4096 (x : ℝ) : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) :=
by
  sorry

end factor_x12_minus_4096_l253_253526


namespace intersection_in_fourth_quadrant_l253_253910

theorem intersection_in_fourth_quadrant (a : ℝ) : 
  (∃ x y : ℝ, y = -x + 1 ∧ y = x - 2 * a ∧ x > 0 ∧ y < 0) → a > 1 / 2 := 
by 
  sorry

end intersection_in_fourth_quadrant_l253_253910


namespace probability_x_satisfies_inequality_l253_253671

open Set

theorem probability_x_satisfies_inequality :
  let interval := Ioo 0 5
  let subinterval := Ioo 0 2
  (volume subinterval / volume interval) = 2 / 5 :=
by
  sorry

end probability_x_satisfies_inequality_l253_253671


namespace p_necessary_not_sufficient_for_q_l253_253263

open Real

noncomputable def p (x : ℝ) : Prop := |x| < 3
noncomputable def q (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem p_necessary_not_sufficient_for_q : 
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l253_253263


namespace largest_distinct_arithmetic_sequence_number_l253_253197

theorem largest_distinct_arithmetic_sequence_number :
  ∃ a b c d : ℕ, 
    (100 * a + 10 * b + c = 789) ∧ 
    (b - a = d) ∧ 
    (c - b = d) ∧ 
    (a ≠ b) ∧ 
    (b ≠ c) ∧ 
    (a ≠ c) ∧ 
    (a < 10) ∧ 
    (b < 10) ∧ 
    (c < 10) :=
sorry

end largest_distinct_arithmetic_sequence_number_l253_253197


namespace sum_of_excluded_numbers_l253_253065

theorem sum_of_excluded_numbers (S : ℕ) (X : ℕ) (n m : ℕ) (averageN : ℕ) (averageM : ℕ)
  (h1 : S = 34 * 8) 
  (h2 : n = 8) 
  (h3 : m = 6) 
  (h4 : averageN = 34) 
  (h5 : averageM = 29) 
  (hS : S = n * averageN) 
  (hX : S - X = m * averageM) : 
  X = 98 := by
  sorry

end sum_of_excluded_numbers_l253_253065


namespace larger_rectangle_area_l253_253312

/-- Given a smaller rectangle made out of three squares each of area 25 cm²,
    where two vertices of the smaller rectangle lie on the midpoints of the
    shorter sides of the larger rectangle and the other two vertices lie on
    the longer sides, prove the area of the larger rectangle is 150 cm². -/
theorem larger_rectangle_area (s : ℝ) (l W S_Larger W_Larger : ℝ)
  (h_s : s^2 = 25) 
  (h_small_dim : l = 3 * s ∧ W = s ∧ l * W = 3 * s^2) 
  (h_vertices : 2 * W = W_Larger ∧ l = S_Larger) :
  (S_Larger * W_Larger = 150) := 
by
  sorry

end larger_rectangle_area_l253_253312


namespace Lowella_score_l253_253133

theorem Lowella_score
  (Mandy_score : ℕ)
  (Pamela_score : ℕ)
  (Lowella_score : ℕ)
  (h1 : Mandy_score = 84) 
  (h2 : Mandy_score = 2 * Pamela_score)
  (h3 : Pamela_score = Lowella_score + 20) :
  Lowella_score = 22 := by
  sorry

end Lowella_score_l253_253133


namespace number_of_zeros_l253_253474

noncomputable def f (x : ℝ) : ℝ := |2^x - 1| - 3^x

theorem number_of_zeros : ∃! x : ℝ, f x = 0 := sorry

end number_of_zeros_l253_253474


namespace fraction_of_6_l253_253379

theorem fraction_of_6 (x y : ℕ) (h : (x / y : ℚ) * 6 + 6 = 10) : (x / y : ℚ) = 2 / 3 :=
by
  sorry

end fraction_of_6_l253_253379


namespace betty_height_correct_l253_253688

-- Definitions for the conditions
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height_inches : ℕ := carter_height - 12
def betty_height_feet : ℕ := betty_height_inches / 12

-- Theorem that we need to prove
theorem betty_height_correct : betty_height_feet = 3 :=
by
  sorry

end betty_height_correct_l253_253688


namespace no_nat_number_with_perfect_square_l253_253461

theorem no_nat_number_with_perfect_square (n : Nat) : 
  ¬ ∃ m : Nat, m * m = n^6 + 3 * n^5 - 5 * n^4 - 15 * n^3 + 4 * n^2 + 12 * n + 3 := 
  by
  sorry

end no_nat_number_with_perfect_square_l253_253461


namespace number_of_sides_of_polygon_l253_253935

theorem number_of_sides_of_polygon :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 2 * n + 7 ∧ n = 8 := 
by
  sorry

end number_of_sides_of_polygon_l253_253935


namespace carson_can_ride_giant_slide_exactly_twice_l253_253226

noncomputable def Carson_Carnival : Prop := 
  let total_time_available := 240
  let roller_coaster_time := 30
  let tilt_a_whirl_time := 60
  let giant_slide_time := 15
  let vortex_time := 45
  let bumper_cars_time := 25
  let roller_coaster_rides := 4
  let tilt_a_whirl_rides := 2
  let vortex_rides := 1
  let bumper_cars_rides := 3

  let total_time_spent := 
    roller_coaster_time * roller_coaster_rides +
    tilt_a_whirl_time * tilt_a_whirl_rides +
    vortex_time * vortex_rides +
    bumper_cars_time * bumper_cars_rides

  total_time_available - (total_time_spent + giant_slide_time * 2) = 0

theorem carson_can_ride_giant_slide_exactly_twice : Carson_Carnival :=
by
  unfold Carson_Carnival
  sorry -- proof will be provided here

end carson_can_ride_giant_slide_exactly_twice_l253_253226


namespace geometric_progression_vertex_l253_253404

theorem geometric_progression_vertex (a b c d : ℝ) (q : ℝ)
  (h1 : b = 1)
  (h2 : c = 2)
  (h3 : q = c / b)
  (h4 : a = b / q)
  (h5 : d = c * q) :
  a + d = 9 / 2 :=
sorry

end geometric_progression_vertex_l253_253404


namespace tayzia_tip_l253_253308

theorem tayzia_tip (haircut_women : ℕ) (haircut_children : ℕ) (num_women : ℕ) (num_children : ℕ) (tip_percentage : ℕ) :
  ((num_women * haircut_women + num_children * haircut_children) * tip_percentage / 100) = 24 :=
by
  -- Given conditions
  let haircut_women := 48
  let haircut_children := 36
  let num_women := 1
  let num_children := 2
  let tip_percentage := 20
  -- Perform the calculations as shown in the solution steps
  sorry

end tayzia_tip_l253_253308


namespace actual_distance_traveled_l253_253126

theorem actual_distance_traveled (D t : ℝ) 
  (h1 : D = 15 * t)
  (h2 : D + 50 = 35 * t) : 
  D = 37.5 :=
by
  sorry

end actual_distance_traveled_l253_253126


namespace evaluate_series_sum_l253_253770

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253770


namespace isosceles_obtuse_triangle_angles_l253_253516

def isosceles (A B C : ℝ) : Prop := A = B ∨ B = C ∨ C = A
def obtuse (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90

noncomputable def sixty_percent_larger_angle : ℝ := 1.6 * 90

theorem isosceles_obtuse_triangle_angles 
  (A B C : ℝ) 
  (h_iso : isosceles A B C) 
  (h_obt : obtuse A B C) 
  (h_large_angle : A = sixty_percent_larger_angle ∨ B = sixty_percent_larger_angle ∨ C = sixty_percent_larger_angle) 
  (h_sum : A + B + C = 180) : 
  (A = 18 ∨ B = 18 ∨ C = 18) := 
sorry

end isosceles_obtuse_triangle_angles_l253_253516


namespace initial_chocolate_amount_l253_253040

-- Define the problem conditions

def initial_dough (d : ℕ) := d = 36
def left_over_chocolate (lo_choc : ℕ) := lo_choc = 4
def chocolate_percentage (p : ℚ) := p = 0.20
def total_weight (d : ℕ) (c_choc : ℕ) := d + c_choc - 4
def chocolate_used (c_choc : ℕ) (lo_choc : ℕ) := c_choc - lo_choc

-- The main proof goal
theorem initial_chocolate_amount (d : ℕ) (lo_choc : ℕ) (p : ℚ) (C : ℕ) :
  initial_dough d → left_over_chocolate lo_choc → chocolate_percentage p →
  p * (total_weight d C) = chocolate_used C lo_choc → C = 13 :=
by
  intros hd hlc hp h
  sorry

end initial_chocolate_amount_l253_253040


namespace triangle_angle_equality_l253_253189

theorem triangle_angle_equality
  (α β γ α₁ β₁ γ₁ : ℝ)
  (hABC : α + β + γ = 180)
  (hA₁B₁C₁ : α₁ + β₁ + γ₁ = 180)
  (angle_relation : (α = α₁ ∨ α + α₁ = 180) ∧ (β = β₁ ∨ β + β₁ = 180) ∧ (γ = γ₁ ∨ γ + γ₁ = 180)) :
  α = α₁ ∧ β = β₁ ∧ γ = γ₁ :=
by {
  sorry
}

end triangle_angle_equality_l253_253189


namespace perfect_square_iff_l253_253163

theorem perfect_square_iff (A : ℕ) : (∃ k : ℕ, A = k^2) ↔ (∀ n : ℕ, n > 0 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n ∣ ((A + k)^2 - A)) :=
by
  sorry

end perfect_square_iff_l253_253163


namespace B_time_to_complete_work_l253_253345

variable {W : ℝ} {R_b : ℝ} {T_b : ℝ}

theorem B_time_to_complete_work (h1 : 3 * R_b * (T_b - 10) = R_b * T_b) : T_b = 15 :=
by
  sorry

end B_time_to_complete_work_l253_253345


namespace geometric_sequence_sixth_term_correct_l253_253637

noncomputable def geometric_sequence_sixth_term (a r : ℝ) (pos_a : 0 < a) (pos_r : 0 < r)
    (third_term : a * r^2 = 27)
    (ninth_term : a * r^8 = 3) : ℝ :=
  a * r^5

theorem geometric_sequence_sixth_term_correct (a r : ℝ) (pos_a : 0 < a) (pos_r : 0 < r) 
    (third_term : a * r^2 = 27)
    (ninth_term : a * r^8 = 3) : geometric_sequence_sixth_term a r pos_a pos_r third_term ninth_term = 9 := 
sorry

end geometric_sequence_sixth_term_correct_l253_253637


namespace lines_parallel_l253_253297

theorem lines_parallel (m : ℝ) : 
  (m = 2 ↔ ∀ x y : ℝ, (2 * x - m * y - 1 = 0) ∧ ((m - 1) * x - y + 1 = 0) → 
  (∃ k : ℝ, (2 * x - m * y - 1 = k * ((m - 1) * x - y + 1)))) :=
by sorry

end lines_parallel_l253_253297


namespace solution_exists_l253_253237

noncomputable def equation (x : ℝ) := 
  (x^2 - 5 * x + 4) / (x - 1) + (2 * x^2 + 7 * x - 4) / (2 * x - 1)

theorem solution_exists : equation 2 = 4 := by
  sorry

end solution_exists_l253_253237


namespace binom_solution_l253_253550

theorem binom_solution (x y : ℕ) (hxy : x > 0 ∧ y > 0) (bin_eq : Nat.choose x y = 1999000) : x = 1999000 ∨ x = 2000 := 
by
  sorry

end binom_solution_l253_253550


namespace probability_math_majors_consecutive_l253_253484

theorem probability_math_majors_consecutive :
  (5 / 12) * (4 / 11) * (3 / 10) * (2 / 9) * (1 / 8) * 12 = 1 / 66 :=
by
  sorry

end probability_math_majors_consecutive_l253_253484


namespace sum_of_first_six_terms_geometric_sequence_l253_253874

-- conditions
def a : ℚ := 1/4
def r : ℚ := 1/4

-- geometric series sum function
def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- target sum of first six terms
def S_6 : ℚ := geom_sum a r 6

-- proof statement
theorem sum_of_first_six_terms_geometric_sequence :
  S_6 = 1365 / 4096 :=
by 
  sorry

end sum_of_first_six_terms_geometric_sequence_l253_253874


namespace kelsey_total_distance_l253_253003

-- Define the constants and variables involved
def total_distance (total_time : ℕ) (speed1 speed2 half_dist1 half_dist2 : ℕ) : ℕ :=
  let T1 := half_dist1 / speed1
  let T2 := half_dist2 / speed2
  let T := T1 + T2
  total_time

-- Prove the equivalency given the conditions
theorem kelsey_total_distance (total_time : ℕ) (speed1 speed2 : ℕ) : 
  (total_time = 10) ∧ (speed1 = 25) ∧ (speed2 = 40)  →
  ∃ D, D = 307 ∧ (10 = D / 50 + D / 80) :=
by 
  intro h
  have h_total_time := h.1
  have h_speed1 := h.2.1
  have h_speed2 := h.2.2
  -- Need to prove the statement using provided conditions
  let D := 307
  sorry

end kelsey_total_distance_l253_253003


namespace total_fish_caught_l253_253074

-- Definitions based on conditions
def brenden_morning_fish := 8
def brenden_fish_thrown_back := 3
def brenden_afternoon_fish := 5
def dad_fish := 13

-- Theorem representing the main question and its answer
theorem total_fish_caught : 
  (brenden_morning_fish + brenden_afternoon_fish - brenden_fish_thrown_back) + dad_fish = 23 :=
by
  sorry -- Proof goes here

end total_fish_caught_l253_253074


namespace find_genuine_coin_in_three_weighings_l253_253324

theorem find_genuine_coin_in_three_weighings (coins : Fin 15 → ℝ)
  (even_number_of_counterfeit : ∃ n : ℕ, 2 * n < 15 ∧ (∀ i, coins i = 1) ∨ (∃ j, coins j = 0.5)) : 
  ∃ i, coins i = 1 :=
by sorry

end find_genuine_coin_in_three_weighings_l253_253324


namespace sum_series_div_3_powers_l253_253763

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253763


namespace ratio_of_first_term_to_common_difference_l253_253201

theorem ratio_of_first_term_to_common_difference
  (a d : ℝ)
  (h : (8 / 2 * (2 * a + 7 * d)) = 3 * (5 / 2 * (2 * a + 4 * d))) :
  a / d = 2 / 7 :=
by
  sorry

end ratio_of_first_term_to_common_difference_l253_253201


namespace compound_interest_rate_l253_253042

theorem compound_interest_rate
  (P : ℝ) (t : ℕ) (A : ℝ) (interest : ℝ)
  (hP : P = 6000)
  (ht : t = 2)
  (hA : A = 7260)
  (hInterest : interest = 1260.000000000001)
  (hA_eq : A = P + interest) :
  ∃ r : ℝ, (1 + r)^(t : ℝ) = A / P ∧ r = 0.1 :=
by
  sorry

end compound_interest_rate_l253_253042


namespace area_of_inscribed_triangle_l253_253486

-- Define the square with a given diagonal
def diagonal (d : ℝ) : Prop := d = 16
def side_length_of_square (s : ℝ) : Prop := s = 8 * Real.sqrt 2
def side_length_of_equilateral_triangle (a : ℝ) : Prop := a = 8 * Real.sqrt 2

-- Define the area of the equilateral triangle
def area_of_equilateral_triangle (area : ℝ) : Prop :=
  area = 32 * Real.sqrt 3

-- The theorem: Given the above conditions, prove the area of the equilateral triangle
theorem area_of_inscribed_triangle (d s a area : ℝ) 
  (h1 : diagonal d) 
  (h2 : side_length_of_square s) 
  (h3 : side_length_of_equilateral_triangle a) 
  (h4 : s = a) : 
  area_of_equilateral_triangle area :=
sorry

end area_of_inscribed_triangle_l253_253486


namespace smallest_n_for_three_pairs_l253_253872

theorem smallest_n_for_three_pairs :
  ∃ (n : ℕ), (0 < n) ∧
    (∀ (x y : ℕ), (x^2 - y^2 = n) → (0 < x) ∧ (0 < y)) ∧
    (∃ (a b c : ℕ), 
      (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
      (∃ (x y : ℕ), (x^2 - y^2 = n) ∧
        (((x, y) = (a, b)) ∨ ((x, y) = (b, c)) ∨ ((x, y) = (a, c))))) :=
sorry

end smallest_n_for_three_pairs_l253_253872


namespace time_to_run_above_tree_l253_253674

-- Defining the given conditions
def tiger_length : ℕ := 5
def tree_trunk_length : ℕ := 20
def time_to_pass_grass : ℕ := 1

-- Defining the speed of the tiger
def tiger_speed : ℕ := tiger_length / time_to_pass_grass

-- Defining the total distance the tiger needs to run
def total_distance : ℕ := tree_trunk_length + tiger_length

-- The theorem stating the time it takes for the tiger to run above the fallen tree trunk
theorem time_to_run_above_tree :
  (total_distance / tiger_speed) = 5 :=
by
  -- Trying to fit the solution steps as formal Lean statements
  sorry

end time_to_run_above_tree_l253_253674


namespace find_x_value_l253_253007

theorem find_x_value (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (heq: (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 := 
sorry

end find_x_value_l253_253007


namespace part1_part2_l253_253423

-- Problem conditions and target statement
theorem part1
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : C = 2 / 3 * Real.pi)
  (h2 : (a, b, c) = (c - 4, c - 2, c))
  (h3 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) :
  c = 7 :=
sorry

theorem part2
  (a b c A B : ℝ)
  (h0 : c = Real.sqrt 3)
  (h1 : C = 2 / 3 * Real.pi)
  (h2 : a = 2 * Real.sin B)
  (h3 : b = 2 * Real.sin (Real.pi / 3 - B))
  (h4 : A = Real.pi - B - C)
  (h5 : 0 < A) :
  let f : ℝ → ℝ := λ θ, 2 * Real.sin (θ + Real.pi / 3) + Real.sqrt 3 in
  ∃ max_value, max_value = 2 + Real.sqrt 3 ∧
  (∀ θ, 0 < θ ∧ θ < Real.pi / 3 → (f θ ≤ max_value)) :=
sorry

end part1_part2_l253_253423


namespace tom_has_hours_to_spare_l253_253184

-- Conditions as definitions
def numberOfWalls : Nat := 5
def wallWidth : Nat := 2 -- in meters
def wallHeight : Nat := 3 -- in meters
def paintingRate : Nat := 10 -- in minutes per square meter
def totalAvailableTime : Nat := 10 -- in hours

-- Lean 4 statement of the problem
theorem tom_has_hours_to_spare :
  let areaOfOneWall := wallWidth * wallHeight -- 2 * 3
  let totalArea := numberOfWalls * areaOfOneWall -- 5 * (2 * 3)
  let totalTimeToPaint := (totalArea * paintingRate) / 60 -- (30 * 10) / 60
  totalAvailableTime - totalTimeToPaint = 5 :=
by
  sorry

end tom_has_hours_to_spare_l253_253184


namespace exists_nat_numbers_except_two_three_l253_253868

theorem exists_nat_numbers_except_two_three (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ (k ≠ 2 ∧ k ≠ 3) :=
by
  sorry

end exists_nat_numbers_except_two_three_l253_253868


namespace triangle_perimeter_is_720_l253_253028

-- Definitions corresponding to conditions
variables (x : ℕ)
noncomputable def shortest_side := 5 * x
noncomputable def middle_side := 6 * x
noncomputable def longest_side := 7 * x

-- Given the length of the longest side is 280 cm
axiom longest_side_eq : longest_side x = 280

-- Prove that the perimeter of the triangle is 720 cm
theorem triangle_perimeter_is_720 : 
  shortest_side x + middle_side x + longest_side x = 720 :=
by
  sorry

end triangle_perimeter_is_720_l253_253028


namespace slower_speed_l253_253058

theorem slower_speed (x : ℝ) (h_walk_faster : 12 * (100 / x) - 100 = 20) : x = 10 :=
by sorry

end slower_speed_l253_253058


namespace union_of_A_and_B_eq_C_l253_253400

open Set

def A := {x : ℝ | -3 < x ∧ x < 3}
def B := {x : ℝ | x^2 - x - 6 ≤ 0}
def C := {x : ℝ | -3 < x ∧ x ≤ 3}

theorem union_of_A_and_B_eq_C : A ∪ B = C := 
by 
  sorry

end union_of_A_and_B_eq_C_l253_253400


namespace smallest_n_congruence_l253_253647

theorem smallest_n_congruence :
  ∃ n : ℕ+, 537 * (n : ℕ) % 30 = 1073 * (n : ℕ) % 30 ∧ (∀ m : ℕ+, 537 * (m : ℕ) % 30 = 1073 * (m : ℕ) % 30 → (m : ℕ) < n → false) :=
  sorry

end smallest_n_congruence_l253_253647


namespace tangent_line_condition_l253_253480

theorem tangent_line_condition (k : ℝ) : 
  (∀ x y : ℝ, (x-2)^2 + (y-1)^2 = 1 → x - k * y - 1 = 0 → False) ↔ k = 0 :=
sorry

end tangent_line_condition_l253_253480


namespace series_converges_to_three_fourths_l253_253699

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253699


namespace arccos_range_l253_253396

theorem arccos_range (a : ℝ) (x : ℝ) (h1 : x = Real.sin a) (h2 : a ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4)) :
  Set.Icc 0 (3 * Real.pi / 4) = Set.image Real.arccos (Set.Icc (-Real.sqrt 2 / 2) 1) :=
by
  sorry

end arccos_range_l253_253396


namespace infinite_series_sum_value_l253_253805

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253805


namespace perimeter_of_combined_figure_l253_253596

theorem perimeter_of_combined_figure (P1 P2 : ℕ) (s1 s2 : ℕ) (overlap : ℕ) :
    P1 = 40 →
    P2 = 100 →
    s1 = P1 / 4 →
    s2 = P2 / 4 →
    overlap = 2 * s1 →
    (P1 + P2 - overlap) = 120 := 
by
  intros hP1 hP2 hs1 hs2 hoverlap
  rw [hP1, hP2, hs1, hs2, hoverlap]
  norm_num
  sorry

end perimeter_of_combined_figure_l253_253596


namespace math_problem_l253_253477

variable (a b c : ℝ)

variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a ≠ -b) (h5 : b ≠ -c) (h6 : c ≠ -a)

theorem math_problem 
    (h₁ : (a * b) / (a + b) = 4)
    (h₂ : (b * c) / (b + c) = 5)
    (h₃ : (c * a) / (c + a) = 7) :
    (a * b * c) / (a * b + b * c + c * a) = 280 / 83 := 
sorry

end math_problem_l253_253477


namespace colony_fungi_day_l253_253428

theorem colony_fungi_day (n : ℕ): 
  (4 * 2^n > 150) = (n = 6) :=
sorry

end colony_fungi_day_l253_253428


namespace inequality_additive_l253_253290

variable {a b c d : ℝ}

theorem inequality_additive (h1 : a > b) (h2 : c > d) : a + c > b + d :=
by
  sorry

end inequality_additive_l253_253290


namespace smallest_a_b_sum_l253_253258

theorem smallest_a_b_sum :
  ∃ (a b : ℕ), 3^6 * 5^3 * 7^2 = a^b ∧ a + b = 317 := 
sorry

end smallest_a_b_sum_l253_253258


namespace mass_percentage_O_in_N2O_is_approximately_36_35_l253_253242

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def number_of_N : ℕ := 2
noncomputable def number_of_O : ℕ := 1

noncomputable def molar_mass_N2O : ℝ := (number_of_N * atomic_mass_N) + (number_of_O * atomic_mass_O)

noncomputable def mass_percentage_O : ℝ := (atomic_mass_O / molar_mass_N2O) * 100

theorem mass_percentage_O_in_N2O_is_approximately_36_35 :
  abs (mass_percentage_O - 36.35) < 0.01 := sorry

end mass_percentage_O_in_N2O_is_approximately_36_35_l253_253242


namespace find_m_l253_253398

-- Define the arithmetic sequence and its properties
variable {α : Type*} [OrderedRing α]
variable (a : Nat → α) (S : Nat → α) (m : ℕ)

-- The conditions from the problem
variable (is_arithmetic_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
variable (sum_of_terms : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2)
variable (m_gt_one : m > 1)
variable (condition1 : a (m - 1) + a (m + 1) - a m ^ 2 - 1 = 0)
variable (condition2 : S (2 * m - 1) = 39)

-- Prove that m = 20
theorem find_m : m = 20 :=
sorry

end find_m_l253_253398


namespace parallel_lines_find_m_l253_253886

theorem parallel_lines_find_m :
  (∀ (m : ℝ), ∀ (x y : ℝ), (2 * x + (m + 1) * y + 4 = 0) ∧ (m * x + 3 * y - 2 = 0) → (m = -3 ∨ m = 2)) := 
sorry

end parallel_lines_find_m_l253_253886


namespace sum_binomial_coeffs_equal_sum_k_values_l253_253090

theorem sum_binomial_coeffs_equal (k : ℕ) 
  (h1 : nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  k = 6 ∨ k = 20 := sorry

theorem sum_k_values (k : ℕ) (h1 :
  nat.choose 25 5 + nat.choose 25 6 = nat.choose 26 k) 
  (h2 : nat.choose 26 6 = nat.choose 26 20) : 
  6 + 20 = 26 := by 
  have h : k = 6 ∨ k = 20 := sum_binomial_coeffs_equal k h1 h2
  sorry

end sum_binomial_coeffs_equal_sum_k_values_l253_253090


namespace series_sum_l253_253744

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253744


namespace max_profit_l253_253286

noncomputable def total_cost (Q : ℝ) : ℝ := 5 * Q^2

noncomputable def demand_non_slytherin (P : ℝ) : ℝ := 26 - 2 * P

noncomputable def demand_slytherin (P : ℝ) : ℝ := 10 - P

noncomputable def combined_demand (P : ℝ) : ℝ :=
  if P >= 13 then demand_non_slytherin P else demand_non_slytherin P + demand_slytherin P

noncomputable def inverse_demand (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q / 2 else 12 - Q / 3

noncomputable def revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then Q * (13 - Q / 2) else Q * (12 - Q / 3)

noncomputable def marginal_revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q else 12 - 2 * Q / 3

noncomputable def marginal_cost (Q : ℝ) : ℝ := 10 * Q

theorem max_profit :
  ∃ Q P TR TC π,
    P = inverse_demand Q ∧
    TR = P * Q ∧
    TC = total_cost Q ∧
    π = TR - TC ∧
    π = 7.69 :=
sorry

end max_profit_l253_253286


namespace sum_series_div_3_powers_l253_253764

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253764


namespace sum_integers_75_to_95_l253_253489

theorem sum_integers_75_to_95 : ∑ k in finset.Icc 75 95, k = 1785 := by
  sorry

end sum_integers_75_to_95_l253_253489


namespace infinite_series_sum_eq_3_over_4_l253_253850

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253850


namespace find_z_percentage_of_1000_l253_253888

noncomputable def x := (3 / 5) * 4864
noncomputable def y := (2 / 3) * 9720
noncomputable def z := (1 / 4) * 800

theorem find_z_percentage_of_1000 :
  (2 / 3) * x + (1 / 2) * y = z → (z / 1000) * 100 = 20 :=
by
  sorry

end find_z_percentage_of_1000_l253_253888


namespace repeating_decimals_count_l253_253251

theorem repeating_decimals_count :
  { n : ℕ | 1 ≤ n ∧ n ≤ 20 ∧ (¬∃ k, n = 9 * k) }.to_finset.card = 18 :=
by
  sorry

end repeating_decimals_count_l253_253251


namespace sneaker_final_price_l253_253998

-- Definitions of the conditions
def original_price : ℝ := 120
def coupon_value : ℝ := 10
def discount_percent : ℝ := 0.1

-- The price after the coupon is applied
def price_after_coupon := original_price - coupon_value

-- The membership discount amount
def membership_discount := price_after_coupon * discount_percent

-- The final price the man will pay
def final_price := price_after_coupon - membership_discount

theorem sneaker_final_price : final_price = 99 := by
  sorry

end sneaker_final_price_l253_253998


namespace evaluate_series_sum_l253_253771

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253771


namespace seating_arrangements_l253_253638

theorem seating_arrangements (n_seats : ℕ) (n_people : ℕ) (n_adj_empty : ℕ) (h1 : n_seats = 6) 
    (h2 : n_people = 3) (h3 : n_adj_empty = 2) : 
    ∃ arrangements : ℕ, arrangements = 48 := 
by
  sorry

end seating_arrangements_l253_253638


namespace binomial_sum_sum_of_binomial_solutions_l253_253096

theorem binomial_sum (k : ℕ) (h1 : Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (h2 : k = 6 ∨ k = 20) :
  k = 6 ∨ k = 20 → k = 6 ∨ k = 20 :=
by
  sorry

theorem sum_of_binomial_solutions :
  ∑ k in {6, 20}, k = 26 :=
by
  sorry

end binomial_sum_sum_of_binomial_solutions_l253_253096


namespace total_plates_l253_253055

-- define the variables for the number of plates
def plates_lobster_rolls : Nat := 25
def plates_spicy_hot_noodles : Nat := 14
def plates_seafood_noodles : Nat := 16

-- state the problem as a theorem
theorem total_plates :
  plates_lobster_rolls + plates_spicy_hot_noodles + plates_seafood_noodles = 55 := by
  sorry

end total_plates_l253_253055


namespace evaluate_series_sum_l253_253774

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253774


namespace factorial_sum_perfect_square_iff_l253_253238

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, m * m = n

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map Nat.factorial |>.sum

theorem factorial_sum_perfect_square_iff (n : Nat) :
  n = 1 ∨ n = 3 ↔ is_perfect_square (sum_of_factorials n) := by {
  sorry
}

end factorial_sum_perfect_square_iff_l253_253238


namespace usual_time_to_school_l253_253645

theorem usual_time_to_school (R T : ℕ) (h : 7 * R * (T - 4) = 6 * R * T) : T = 28 :=
sorry

end usual_time_to_school_l253_253645


namespace series_sum_correct_l253_253787

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253787


namespace largest_possible_s_l253_253604

theorem largest_possible_s (r s : ℕ) 
  (hr : r ≥ s) 
  (hs : s ≥ 3) 
  (h_angle : (101 : ℚ) / 97 * ((s - 2) * 180 / s : ℚ) = ((r - 2) * 180 / r : ℚ)) :
  s = 100 :=
by
  sorry

end largest_possible_s_l253_253604


namespace series_sum_l253_253739

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253739


namespace expected_number_of_ones_l253_253951

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l253_253951


namespace value_of_fraction_l253_253113

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = (-4 - (-1)) / (4 - 1)

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
  b2 * b2 = (-4) * (-1) ∧ b2 < 0

theorem value_of_fraction (a1 a2 b2 : ℝ)
  (h1 : arithmetic_sequence a1 a2)
  (h2 : geometric_sequence b2) :
  (a2 - a1) / b2 = 1 / 2 :=
by
  sorry

end value_of_fraction_l253_253113


namespace proof_problem_l253_253383

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2, (0 ≤ x1) → (0 ≤ x2) → (x1 ≠ x2) → (x1 - x2) * (f x1 - f x2) > 0

theorem proof_problem (f : ℝ → ℝ) (hf_even : even_function f) (hf_condition : condition f) :
  f 1 < f (-2) ∧ f (-2) < f 3 := sorry

end proof_problem_l253_253383


namespace series_result_l253_253840

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253840


namespace difference_between_advertised_and_actual_mileage_l253_253009

def advertised_mileage : ℕ := 35

def city_mileage_regular : ℕ := 30
def highway_mileage_premium : ℕ := 40
def traffic_mileage_diesel : ℕ := 32

def gallons_regular : ℕ := 4
def gallons_premium : ℕ := 4
def gallons_diesel : ℕ := 4

def total_miles_driven : ℕ :=
  (gallons_regular * city_mileage_regular) + 
  (gallons_premium * highway_mileage_premium) + 
  (gallons_diesel * traffic_mileage_diesel)

def total_gallons_used : ℕ :=
  gallons_regular + gallons_premium + gallons_diesel

def weighted_average_mpg : ℤ :=
  total_miles_driven / total_gallons_used

theorem difference_between_advertised_and_actual_mileage :
  advertised_mileage - weighted_average_mpg = 1 :=
by
  -- proof to be filled in
  sorry

end difference_between_advertised_and_actual_mileage_l253_253009


namespace area_of_sector_l253_253408

theorem area_of_sector (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 60) : (θ / 360 * π * r^2 = 6 * π) :=
by sorry

end area_of_sector_l253_253408


namespace trees_died_l253_253514

theorem trees_died (initial_trees dead surviving : ℕ) 
  (h_initial : initial_trees = 11) 
  (h_surviving : surviving = dead + 7) 
  (h_total : dead + surviving = initial_trees) : 
  dead = 2 :=
by
  sorry

end trees_died_l253_253514


namespace reciprocal_div_calculate_fraction_reciprocal_div_result_l253_253620

-- Part 1
theorem reciprocal_div {a b c : ℚ} (h : (a + b) / c = -2) : c / (a + b) = -1 / 2 :=
sorry

-- Part 2
theorem calculate_fraction : (5 / 12 - 1 / 9 + 2 / 3) / (1 / 36) = 35 :=
sorry

-- Part 3
theorem reciprocal_div_result : (5 / 12 - 1 / 9 + 2 / 3) / (1 / 36) = 35 →
 (-1 / 36) / (5 / 12 - 1 / 9 + 2 / 3) = -1 / 35 :=
sorry

end reciprocal_div_calculate_fraction_reciprocal_div_result_l253_253620


namespace sequence_formula_l253_253026

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 0)
  (h : ∀ n, a (n + 1) = 1 / (2 - a n)) :
  ∀ n, a n = (n - 1) / n :=
sorry

end sequence_formula_l253_253026


namespace infinite_series_sum_eq_3_over_4_l253_253845

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253845


namespace sum_series_eq_3_over_4_l253_253748

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253748


namespace sum_of_coordinates_l253_253301

theorem sum_of_coordinates (x : ℚ) : (0, 0) = (0, 0) ∧ (x, -3) = (x, -3) ∧ ((-3 - 0) / (x - 0) = 4 / 5) → x - 3 = -27 / 4 := 
sorry

end sum_of_coordinates_l253_253301


namespace repeating_decimals_for_n_div_18_l253_253245

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end repeating_decimals_for_n_div_18_l253_253245


namespace chip_drawing_probability_l253_253662

theorem chip_drawing_probability :
  let total_chips := 12,
      purple_chips := 4,
      orange_chips := 3,
      green_chips := 5,
      total_permutations := total_chips.factorial,
      constrained_permutations := 2.factorial * purple_chips.factorial * orange_chips.factorial * green_chips.factorial
  in (constrained_permutations : ℚ) / total_permutations = 1 / 13860 := by
  sorry

end chip_drawing_probability_l253_253662


namespace percentage_selected_in_state_A_l253_253427

-- Definitions
def num_candidates : ℕ := 8000
def percentage_selected_state_B : ℕ := 7
def extra_selected_candidates : ℕ := 80

-- Question
theorem percentage_selected_in_state_A :
  ∃ (P : ℕ), ((P / 100) * 8000 + 80 = 560) ∧ (P = 6) := sorry

end percentage_selected_in_state_A_l253_253427


namespace correct_average_of_10_numbers_l253_253348

theorem correct_average_of_10_numbers
  (incorrect_avg : ℕ)
  (n : ℕ)
  (incorrect_read : ℕ)
  (correct_read : ℕ)
  (incorrect_total_sum : ℕ) :
  incorrect_avg = 19 →
  n = 10 →
  incorrect_read = 26 →
  correct_read = 76 →
  incorrect_total_sum = incorrect_avg * n →
  (correct_total_sum : ℕ) = incorrect_total_sum - incorrect_read + correct_read →
  (correct_avg : ℕ) = correct_total_sum / n →
  correct_avg = 24 :=
by
  intros
  sorry

end correct_average_of_10_numbers_l253_253348


namespace minimum_distance_l253_253276

-- Define conditions and problem

def lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 4 = 0

theorem minimum_distance (P : ℝ × ℝ) (h : lies_on_line P) : P.1^2 + P.2^2 ≥ 8 :=
sorry

end minimum_distance_l253_253276


namespace min_value_l253_253104

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 :=
sorry

end min_value_l253_253104


namespace expected_value_of_ones_on_three_dice_l253_253942

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l253_253942


namespace remaining_sand_fraction_l253_253990

theorem remaining_sand_fraction (total_weight : ℕ) (used_weight : ℕ) (h1 : total_weight = 50) (h2 : used_weight = 30) : 
  (total_weight - used_weight) / total_weight = 2 / 5 :=
by 
  sorry

end remaining_sand_fraction_l253_253990


namespace find_distance_l253_253509

variable (D V : ℕ)

axiom normal_speed : V = 25
axiom time_difference : (D / V) - (D / (V + 5)) = 2

theorem find_distance : D = 300 :=
by
  sorry

end find_distance_l253_253509


namespace num_elements_in_set_is_one_l253_253320

noncomputable def log_eq_condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ Real.log (x^3 + (1 / 3) * y^3 + 1 / 9) = Real.log x + Real.log y

theorem num_elements_in_set_is_one :
  (Set.to_finset {p : ℝ × ℝ | log_eq_condition p.1 p.2}).card = 1 :=
by
  sorry

end num_elements_in_set_is_one_l253_253320


namespace r_exceeds_s_by_six_l253_253270

theorem r_exceeds_s_by_six (x y : ℚ) (h1 : 3 * x + 2 * y = 16) (h2 : x + 3 * y = 26 / 5) :
  x - y = 6 := by
  sorry

end r_exceeds_s_by_six_l253_253270


namespace pages_per_day_l253_253467

variable (P : ℕ) (D : ℕ)

theorem pages_per_day (hP : P = 66) (hD : D = 6) : P / D = 11 :=
by
  sorry

end pages_per_day_l253_253467


namespace more_sqft_to_mow_l253_253598

-- Defining the parameters given in the original problem
def rate_per_sqft : ℝ := 0.10
def book_cost : ℝ := 150.0
def lawn_dimensions : ℝ × ℝ := (20, 15)
def num_lawns_mowed : ℕ := 3

-- The theorem stating how many more square feet LaKeisha needs to mow
theorem more_sqft_to_mow : 
  let area_one_lawn := (lawn_dimensions.1 * lawn_dimensions.2 : ℝ)
  let total_area_mowed := area_one_lawn * (num_lawns_mowed : ℝ)
  let money_earned := total_area_mowed * rate_per_sqft
  let remaining_amount := book_cost - money_earned
  let more_sqft_needed := remaining_amount / rate_per_sqft
  more_sqft_needed = 600 := 
by 
  sorry

end more_sqft_to_mow_l253_253598


namespace find_radius_l253_253013

theorem find_radius :
  ∃ (r : ℝ), 
  (∀ (x : ℝ), y = x^2 + r) ∧ 
  (∀ (x : ℝ), y = x) ∧ 
  (∀ (x : ℝ), x^2 + r = x) ∧ 
  (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 0) → 
  r = 1 / 4 :=
by
  sorry

end find_radius_l253_253013


namespace distinct_digits_sum_base7_l253_253919

theorem distinct_digits_sum_base7
    (A B C : ℕ)
    (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
    (h_nonzero : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
    (h_base7 : A < 7 ∧ B < 7 ∧ C < 7)
    (h_sum_eq : ((7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B)) = (7^3 * A + 7^2 * A + 7 * A)) :
    B + C = 6 :=
by {
    sorry
}

end distinct_digits_sum_base7_l253_253919


namespace days_to_complete_work_l253_253465

-- Let's define the conditions as Lean definitions based on the problem.

variables (P D : ℕ)
noncomputable def original_work := P * D
noncomputable def half_work_by_double_people := 2 * P * 3

-- Here is our theorem statement
theorem days_to_complete_work : original_work P D = 2 * half_work_by_double_people P :=
by sorry

end days_to_complete_work_l253_253465


namespace border_area_l253_253212

theorem border_area (photo_height photo_width border_width : ℕ) (h1 : photo_height = 12) (h2 : photo_width = 16) (h3 : border_width = 3) : 
  let framed_height := photo_height + 2 * border_width 
  let framed_width := photo_width + 2 * border_width 
  let area_of_photo := photo_height * photo_width
  let area_of_framed := framed_height * framed_width 
  let area_of_border := area_of_framed - area_of_photo 
  area_of_border = 204 := 
by
  sorry

end border_area_l253_253212


namespace common_difference_range_l253_253473

noncomputable def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

theorem common_difference_range :
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  (a10 > 0) ∧ (a9 <= 0) → 8 / 3 < d ∧ d <= 3 :=
by
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  intro h
  sorry

end common_difference_range_l253_253473


namespace infinite_series_sum_eq_l253_253716

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253716


namespace expected_ones_three_dice_l253_253968

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l253_253968


namespace tan_135_eq_neg1_l253_253530

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg1_l253_253530


namespace intersection_is_4_l253_253150

-- Definitions of the sets
def U : Set Int := {0, 1, 2, 4, 6, 8}
def M : Set Int := {0, 4, 6}
def N : Set Int := {0, 1, 6}

-- Definition of the complement
def complement_U_N : Set Int := U \ N

-- Definition of the intersection
def intersection_M_complement_U_N : Set Int := M ∩ complement_U_N

-- Statement of the theorem
theorem intersection_is_4 : intersection_M_complement_U_N = {4} :=
by
  sorry

end intersection_is_4_l253_253150


namespace annual_population_change_l253_253364

theorem annual_population_change (initial_population : Int) (moved_in : Int) (moved_out : Int) (final_population : Int) (years : Int) : 
  initial_population = 780 → 
  moved_in = 100 →
  moved_out = 400 →
  final_population = 60 →
  years = 4 →
  (initial_population + moved_in - moved_out - final_population) / years = 105 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end annual_population_change_l253_253364


namespace mass_percentage_H_correct_l253_253333

noncomputable def mass_percentage_H_in_CaH2 : ℝ :=
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_H : ℝ := 1.01
  let molar_mass_CaH2 : ℝ := molar_mass_Ca + 2 * molar_mass_H
  (2 * molar_mass_H / molar_mass_CaH2) * 100

theorem mass_percentage_H_correct :
  |mass_percentage_H_in_CaH2 - 4.80| < 0.01 :=
by
  sorry

end mass_percentage_H_correct_l253_253333


namespace points_on_square_diagonal_l253_253161

theorem points_on_square_diagonal (a : ℝ) (ha : a > 1) (Q : ℝ × ℝ) (hQ : Q = (a + 1, 4 * a + 1)) 
    (line : ℝ × ℝ → Prop) (hline : ∀ (x y : ℝ), line (x, y) ↔ y = a * x + 3) :
    ∃ (P R : ℝ × ℝ), line Q ∧ P = (6, 3) ∧ R = (-3, 6) :=
by
  sorry

end points_on_square_diagonal_l253_253161


namespace two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l253_253080

theorem two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one :
  (2.85 = 2850 * 0.001) := by
  sorry

end two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l253_253080


namespace probability_of_urn_contains_nine_red_and_four_blue_after_operations_l253_253067

-- Definition of the initial urn state
def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1

-- Definition of the number of operations
def num_operations : ℕ := 5

-- Definition of the final state
def final_red_balls : ℕ := 9
def final_blue_balls : ℕ := 4

-- Definition of total number of balls after five operations
def total_balls_after_operations : ℕ := 13

-- The probability we aim to prove
def target_probability : ℚ := 1920 / 10395

noncomputable def george_experiment_probability_theorem 
  (initial_red_balls initial_blue_balls num_operations final_red_balls final_blue_balls : ℕ)
  (total_balls_after_operations : ℕ) : ℚ :=
if initial_red_balls = 2 ∧ initial_blue_balls = 1 ∧ num_operations = 5 ∧ final_red_balls = 9 ∧ final_blue_balls = 4 ∧ total_balls_after_operations = 13 then
  target_probability
else
  0

-- The theorem statement, no proof provided (using sorry).
theorem probability_of_urn_contains_nine_red_and_four_blue_after_operations :
  george_experiment_probability_theorem 2 1 5 9 4 13 = target_probability := sorry

end probability_of_urn_contains_nine_red_and_four_blue_after_operations_l253_253067


namespace ratio_of_boys_to_girls_l253_253204

-- Define the given conditions and provable statement
theorem ratio_of_boys_to_girls (S G : ℕ) (h : (2/3 : ℚ) * G = (1/5 : ℚ) * S) : (S - G) * 3 = 7 * G :=
by
  -- This is a placeholder for solving the proof
  sorry

end ratio_of_boys_to_girls_l253_253204


namespace solve_for_x_l253_253077

theorem solve_for_x : 
  (∃ x : ℝ, (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 4 * x - 21) 
  ∧ x = 4.5) := by
{
  sorry
}

end solve_for_x_l253_253077


namespace LaKeisha_needs_to_mow_more_sqft_l253_253601

noncomputable def LaKeisha_price_per_sqft : ℝ := 0.10
noncomputable def LaKeisha_book_cost : ℝ := 150
noncomputable def LaKeisha_mowed_sqft : ℕ := 3 * 20 * 15
noncomputable def LaKeisha_earnings_so_far : ℝ := LaKeisha_mowed_sqft * LaKeisha_price_per_sqft

theorem LaKeisha_needs_to_mow_more_sqft (additional_sqft_needed : ℝ) :
  additional_sqft_needed = (LaKeisha_book_cost - LaKeisha_earnings_so_far) / LaKeisha_price_per_sqft → 
  additional_sqft_needed = 600 :=
by
  sorry

end LaKeisha_needs_to_mow_more_sqft_l253_253601


namespace total_number_of_baseball_cards_l253_253447

def baseball_cards_total : Nat :=
  let carlos := 20
  let matias := carlos - 6
  let jorge := matias
  carlos + matias + jorge
   
theorem total_number_of_baseball_cards :
  baseball_cards_total = 48 :=
by
  rfl

end total_number_of_baseball_cards_l253_253447


namespace father_son_age_ratio_l253_253665

theorem father_son_age_ratio :
  ∃ S : ℕ, (45 = S + 15 * 2) ∧ (45 / S = 3) := 
sorry

end father_son_age_ratio_l253_253665


namespace avg_two_ab_l253_253020

-- Defining the weights and conditions
variables (A B C : ℕ)

-- The conditions provided in the problem
def avg_three (A B C : ℕ) := (A + B + C) / 3 = 45
def avg_two_bc (B C : ℕ) := (B + C) / 2 = 43
def weight_b (B : ℕ) := B = 35

-- The target proof statement
theorem avg_two_ab (A B C : ℕ) (h1 : avg_three A B C) (h2 : avg_two_bc B C) (h3 : weight_b B) : (A + B) / 2 = 42 := 
sorry

end avg_two_ab_l253_253020


namespace sum_fraction_equals_two_l253_253005

theorem sum_fraction_equals_two
  (a b c d : ℝ) (h₁ : a ≠ -1) (h₂ : b ≠ -1) (h₃ : c ≠ -1) (h₄ : d ≠ -1)
  (ω : ℂ) (h₅ : ω^4 = 1) (h₆ : ω ≠ 1)
  (h₇ : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = (4 / (ω^2))) 
  (h₈ : a + b + c + d = a * b * c * d)
  (h₉ : a * b + a * c + a * d + b * c + b * d + c * d = a * b * c + a * b * d + a * c * d + b * c * d) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := 
sorry

end sum_fraction_equals_two_l253_253005


namespace infinite_series_sum_eq_3_div_4_l253_253799

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253799


namespace Mr_Pendearly_optimal_speed_l253_253615

noncomputable def optimal_speed (d t : ℝ) : ℝ := d / t

theorem Mr_Pendearly_optimal_speed :
  ∀ (d t : ℝ),
  (d = 45 * (t + 1/15)) →
  (d = 75 * (t - 1/15)) →
  optimal_speed d t = 56.25 :=
by
  intros d t h1 h2
  have h_d_eq_45 := h1
  have h_d_eq_75 := h2
  sorry

end Mr_Pendearly_optimal_speed_l253_253615


namespace discount_percentage_l253_253220

theorem discount_percentage (C M A : ℝ) (h1 : M = 1.40 * C) (h2 : A = 1.05 * C) :
    (M - A) / M * 100 = 25 :=
by
  sorry

end discount_percentage_l253_253220


namespace count_two_digit_decimals_between_0_40_and_0_50_l253_253122

theorem count_two_digit_decimals_between_0_40_and_0_50 : 
  ∃ (n : ℕ), n = 9 ∧ ∀ x : ℝ, 0.40 < x ∧ x < 0.50 → (exists d : ℕ, (1 ≤ d ∧ d ≤ 9 ∧ x = 0.4 + d * 0.01)) :=
by
  sorry

end count_two_digit_decimals_between_0_40_and_0_50_l253_253122


namespace expected_number_of_ones_when_three_dice_rolled_l253_253963

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l253_253963


namespace number_of_square_tiles_l253_253664

theorem number_of_square_tiles (a b : ℕ) (h1 : a + b = 32) (h2 : 3 * a + 4 * b = 110) : b = 14 :=
by
  -- the proof steps are skipped
  sorry

end number_of_square_tiles_l253_253664


namespace lily_milk_left_l253_253153

theorem lily_milk_left (initial : ℚ) (given : ℚ) : initial = 5 ∧ given = 18/7 → initial - given = 17/7 :=
by
  intros h,
  cases h with h_initial h_given,
  rw [h_initial, h_given],
  sorry

end lily_milk_left_l253_253153


namespace valid_values_of_X_Y_l253_253329

-- Stating the conditions
def odd_combinations := 125
def even_combinations := 64
def revenue_diff (X Y : ℕ) := odd_combinations * X - even_combinations * Y = 5
def valid_limit (n : ℕ) := 0 < n ∧ n < 250

-- The theorem we want to prove
theorem valid_values_of_X_Y (X Y : ℕ) :
  revenue_diff X Y ∧ valid_limit X ∧ valid_limit Y ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) :=
  sorry

end valid_values_of_X_Y_l253_253329


namespace abs_neg_five_not_eq_five_l253_253648

theorem abs_neg_five_not_eq_five : -(abs (-5)) ≠ 5 := by
  sorry

end abs_neg_five_not_eq_five_l253_253648


namespace expected_number_of_ones_when_three_dice_rolled_l253_253961

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l253_253961


namespace sum_geometric_series_l253_253707

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253707


namespace unique_nets_of_a_cube_l253_253121

-- Definitions based on the conditions and the properties of the cube
def is_net (net: ℕ) : Prop :=
  -- A placeholder definition of a valid net
  sorry

def is_distinct_by_rotation_or_reflection (net1 net2: ℕ) : Prop :=
  -- Two nets are distinct if they cannot be transformed into each other by rotation or reflection
  sorry

-- The statement to be proved
theorem unique_nets_of_a_cube : ∃ n, n = 11 ∧ (∀ net, is_net net → ∃! net', is_net net' ∧ is_distinct_by_rotation_or_reflection net net') :=
sorry

end unique_nets_of_a_cube_l253_253121


namespace ratio_of_r_to_pq_l253_253048

theorem ratio_of_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 7000) (h₂ : r = 2800) :
  r / (p + q) = 2 / 3 :=
by sorry

end ratio_of_r_to_pq_l253_253048


namespace infinite_series_sum_eq_l253_253724

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253724


namespace acute_angle_tan_eq_one_l253_253111

theorem acute_angle_tan_eq_one (A : ℝ) (h1 : 0 < A ∧ A < π / 2) (h2 : Real.tan A = 1) : A = π / 4 :=
by
  sorry

end acute_angle_tan_eq_one_l253_253111


namespace compute_ratio_d_e_l253_253023

open Polynomial

noncomputable def quartic_polynomial (a b c d e : ℚ) : Polynomial ℚ := 
  C a * X^4 + C b * X^3 + C c * X^2 + C d * X + C e

def roots_of_quartic (a b c d e: ℚ) : Prop :=
  (quartic_polynomial a b c d e).roots = {1, 2, 3, 5}

theorem compute_ratio_d_e (a b c d e : ℚ) 
    (h : roots_of_quartic a b c d e) :
    d / e = -61 / 30 :=
  sorry

end compute_ratio_d_e_l253_253023


namespace perimeter_of_quadrilateral_l253_253500

theorem perimeter_of_quadrilateral 
  (WXYZ_area : ℝ)
  (h_area : WXYZ_area = 2500)
  (WQ XQ YQ ZQ : ℝ)
  (h_WQ : WQ = 30)
  (h_XQ : XQ = 40)
  (h_YQ : YQ = 35)
  (h_ZQ : ZQ = 50) :
  ∃ (P : ℝ), P = 155 + 10 * Real.sqrt 34 + 5 * Real.sqrt 113 :=
by
  sorry

end perimeter_of_quadrilateral_l253_253500


namespace tank_ratio_two_l253_253102

variable (T1 : ℕ) (F1 : ℕ) (F2 : ℕ) (T2 : ℕ)

-- Assume the given conditions
axiom h1 : T1 = 48
axiom h2 : F1 = T1 / 3
axiom h3 : F1 - 1 = F2 + 3
axiom h4 : T2 = F2 * 2

-- The theorem to prove
theorem tank_ratio_two (h1 : T1 = 48) (h2 : F1 = T1 / 3) (h3 : F1 - 1 = F2 + 3) (h4 : T2 = F2 * 2) : T1 / T2 = 2 := by
  sorry

end tank_ratio_two_l253_253102


namespace area_of_inscribed_triangle_l253_253362

theorem area_of_inscribed_triangle 
  (x : ℝ) 
  (h1 : (2:ℝ) * x ≤ (3:ℝ) * x ∧ (3:ℝ) * x ≤ (4:ℝ) * x) 
  (h2 : (4:ℝ) * x = 2 * 4) :
  ∃ (area : ℝ), area = 12.00 :=
by
  sorry

end area_of_inscribed_triangle_l253_253362


namespace repeated_digit_percentage_l253_253273

theorem repeated_digit_percentage (total : ℕ := 90000) (non_repeated_count : ℕ := 9 * 9 * 8 * 7 * 6) : 
  let repeated_count := total - non_repeated_count in
  let y := (repeated_count : ℚ) / total * 100 in
  y ≈ 69.8 :=
by
  sorry

end repeated_digit_percentage_l253_253273


namespace infinite_series_sum_eq_3_over_4_l253_253854

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253854


namespace find_line_equation_l253_253903

theorem find_line_equation (k : ℝ) (x y : ℝ) :
  (∀ k, (∃ x y, y = k * x + 1 ∧ x^2 + y^2 - 2 * x - 3 = 0) ↔ x - y + 1 = 0) :=
by
  sorry

end find_line_equation_l253_253903


namespace rewrite_expression_l253_253462

theorem rewrite_expression (k : ℝ) :
  ∃ d r s : ℝ, (8 * k^2 - 12 * k + 20 = d * (k + r)^2 + s) ∧ (r + s = 14.75) := 
sorry

end rewrite_expression_l253_253462


namespace series_sum_eq_l253_253732

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253732


namespace meaningful_expression_iff_l253_253892

theorem meaningful_expression_iff (x : ℝ) : (∃ y, y = (2 : ℝ) / (2*x - 1)) ↔ x ≠ (1 / 2 : ℝ) :=
by
  sorry

end meaningful_expression_iff_l253_253892


namespace number_of_pups_in_second_round_l253_253071

-- Define the conditions
variable (initialMice : Nat := 8)
variable (firstRoundPupsPerMouse : Nat := 6)
variable (secondRoundEatenPupsPerMouse : Nat := 2)
variable (finalMice : Nat := 280)

-- Define the proof problem
theorem number_of_pups_in_second_round (P : Nat) :
  initialMice + initialMice * firstRoundPupsPerMouse = 56 → 
  56 + 56 * P - 56 * secondRoundEatenPupsPerMouse = finalMice →
  P = 6 := by
  intros h1 h2
  sorry

end number_of_pups_in_second_round_l253_253071


namespace fraction_dehydrated_l253_253179

theorem fraction_dehydrated (total_men tripped fraction_dnf finished : ℕ) (fraction_tripped fraction_dehydrated_dnf : ℚ)
  (htotal_men : total_men = 80)
  (hfraction_tripped : fraction_tripped = 1 / 4)
  (htripped : tripped = total_men * fraction_tripped)
  (hfinished : finished = 52)
  (hfraction_dnf : fraction_dehydrated_dnf = 1 / 5)
  (hdnf : total_men - finished = tripped + fraction_dehydrated_dnf * (total_men - tripped) * x)
  (hx : x = 2 / 3) :
  x = 2 / 3 := sorry

end fraction_dehydrated_l253_253179


namespace infinite_series_sum_value_l253_253812

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253812


namespace series_sum_l253_253740

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253740


namespace k_valid_iff_l253_253866

open Nat

theorem k_valid_iff (k : ℕ) :
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by
  sorry

end k_valid_iff_l253_253866


namespace ram_actual_distance_from_base_l253_253497

def map_distance_between_mountains : ℝ := 312
def actual_distance_between_mountains : ℝ := 136
def ram_map_distance_from_base : ℝ := 28

theorem ram_actual_distance_from_base :
  ram_map_distance_from_base * (actual_distance_between_mountains / map_distance_between_mountains) = 12.205 :=
by sorry

end ram_actual_distance_from_base_l253_253497


namespace intersection_complement_l253_253118

-- Definitions based on the conditions in the problem
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

-- Definition of complement of set M in the universe U
def complement_U (M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

-- The proof statement
theorem intersection_complement :
  N ∩ (complement_U M) = {3, 5} :=
by
  sorry

end intersection_complement_l253_253118


namespace series_sum_correct_l253_253786

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253786


namespace find_b_from_conditions_l253_253115

theorem find_b_from_conditions (x y z k : ℝ) (h1 : (x + y) / 2 = k) (h2 : (z + x) / 3 = k) (h3 : (y + z) / 4 = k) (h4 : x + y + z = 36) : x + y = 16 := 
by 
  sorry

end find_b_from_conditions_l253_253115


namespace percentage_second_question_correct_l253_253897

theorem percentage_second_question_correct (a b c : ℝ) 
  (h1 : a = 0.75) (h2 : b = 0.20) (h3 : c = 0.50) :
  (1 - b) - (a - c) + c = 0.55 :=
by
  sorry

end percentage_second_question_correct_l253_253897


namespace sum_series_eq_3_div_4_l253_253832

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253832


namespace way_to_cut_grid_l253_253591

def grid_ways : ℕ := 17

def rectangles (size : ℕ × ℕ) (count : ℕ) := 
  size = (1, 2) ∧ count = 8

def square (size : ℕ × ℕ) (count : ℕ) := 
  size = (1, 1) ∧ count = 1

theorem way_to_cut_grid :
  (∃ ways : ℕ, ways = 10) ↔ 
  ∀ g ways, g = grid_ways → 
  (rectangles (1, 2) 8 ∧ square (1, 1) 1 → ways = 10) :=
by 
  sorry

end way_to_cut_grid_l253_253591


namespace sum_series_div_3_powers_l253_253755

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253755


namespace sum_of_series_eq_three_fourths_l253_253779

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253779


namespace g_2_eq_8_l253_253468

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 4) / x

noncomputable def g (x : ℝ) : ℝ := 1 / f_inv x + 7

theorem g_2_eq_8 : g 2 = 8 := 
by 
  unfold g
  unfold f_inv
  sorry

end g_2_eq_8_l253_253468


namespace broken_shells_count_l253_253418

-- Definitions from conditions
def total_perfect_shells := 17
def non_spiral_perfect_shells := 12
def extra_broken_spiral_shells := 21

-- Derived definitions
def perfect_spiral_shells : ℕ := total_perfect_shells - non_spiral_perfect_shells
def broken_spiral_shells : ℕ := perfect_spiral_shells + extra_broken_spiral_shells
def broken_shells : ℕ := 2 * broken_spiral_shells

-- The theorem to be proved
theorem broken_shells_count : broken_shells = 52 := by
  sorry

end broken_shells_count_l253_253418


namespace infinite_series_sum_value_l253_253813

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253813


namespace selection_of_projects_l253_253357

-- Mathematical definitions
def numberOfWaysToSelect2ProjectsFrom4KeyAnd6General (key: Finset ℕ) (general: Finset ℕ) : ℕ :=
  (key.card.choose 2) * (general.card.choose 2)

def numberOfWaysToSelectAtLeastOneProjectAorB (key: Finset ℕ) (general: Finset ℕ) (A B: ℕ) : ℕ :=
  let total_ways := (key.card.choose 2) * (general.card.choose 2)
  let ways_without_A := ((key.erase A).card.choose 2) * (general.card.choose 2)
  let ways_without_B := (key.card.choose 2) * ((general.erase B).card.choose 2)
  let ways_without_A_and_B := ((key.erase A).card.choose 2) * ((general.erase B).card.choose 2)
  total_ways - ways_without_A_and_B

-- Theorem we need to prove
theorem selection_of_projects (key general: Finset ℕ) (A B: ℕ) (hA: A ∈ key) (hB: B ∈ general) (h_key_card: key.card = 4) (h_general_card: general.card = 6) :
  numberOfWaysToSelectAtLeastOneProjectAorB key general A B = 60 := 
sorry

end selection_of_projects_l253_253357


namespace no_nontrivial_solutions_l253_253624

theorem no_nontrivial_solutions :
  ∀ (x y z t : ℤ), (¬(x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0)) → ¬(x^2 = 2 * y^2 ∧ x^4 + 3 * y^4 + 27 * z^4 = 9 * t^4) :=
by
  intros x y z t h_nontrivial h_eqs
  sorry

end no_nontrivial_solutions_l253_253624


namespace series_sum_correct_l253_253790

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253790


namespace sample_mean_and_variance_l253_253673

def sample : List ℕ := [10, 12, 9, 14, 13]
def n : ℕ := 5

-- Definition of sample mean
noncomputable def sampleMean : ℝ := (sample.sum / n)

-- Definition of sample variance using population formula
noncomputable def sampleVariance : ℝ := (sample.map (λ x_i => (x_i - sampleMean)^2)).sum / n

theorem sample_mean_and_variance :
  sampleMean = 11.6 ∧ sampleVariance = 3.44 := by
  sorry

end sample_mean_and_variance_l253_253673


namespace circle_equation_passing_through_P_l253_253106

-- Define the problem conditions
def P : ℝ × ℝ := (3, 1)
def l₁ (x y : ℝ) := x + 2 * y + 3 = 0
def l₂ (x y : ℝ) := x + 2 * y - 7 = 0

-- The main theorem statement
theorem circle_equation_passing_through_P :
  ∃ (α β : ℝ), 
    ((α = 4 ∧ β = -1) ∨ (α = 4 / 5 ∧ β = 3 / 5)) ∧ 
    ((x - α)^2 + (y - β)^2 = 5) :=
  sorry

end circle_equation_passing_through_P_l253_253106


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253857

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253857


namespace range_of_k_for_real_roots_l253_253129

theorem range_of_k_for_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by 
  sorry

end range_of_k_for_real_roots_l253_253129


namespace polar_to_rectangular_coords_l253_253689

theorem polar_to_rectangular_coords (r θ : ℝ) (x y : ℝ) 
  (hr : r = 5) (hθ : θ = 5 * Real.pi / 4)
  (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x = - (5 * Real.sqrt 2) / 2 ∧ y = - (5 * Real.sqrt 2) / 2 := 
by
  rw [hr, hθ] at hx hy
  simp [Real.cos, Real.sin] at hx hy
  rw [hx, hy]
  constructor
  . sorry
  . sorry

end polar_to_rectangular_coords_l253_253689


namespace total_dogs_l253_253683

variable (U : Type) [Fintype U]
variable (jump fetch shake : U → Prop)
variable [DecidablePred jump] [DecidablePred fetch] [DecidablePred shake]

theorem total_dogs (h_jump : Fintype.card {u | jump u} = 70)
  (h_jump_and_fetch : Fintype.card {u | jump u ∧ fetch u} = 30)
  (h_fetch : Fintype.card {u | fetch u} = 40)
  (h_fetch_and_shake : Fintype.card {u | fetch u ∧ shake u} = 20)
  (h_shake : Fintype.card {u | shake u} = 50)
  (h_jump_and_shake : Fintype.card {u | jump u ∧ shake u} = 25)
  (h_all_three : Fintype.card {u | jump u ∧ fetch u ∧ shake u} = 15)
  (h_none : Fintype.card {u | ¬jump u ∧ ¬fetch u ∧ ¬shake u} = 15) :
  Fintype.card U = 115 :=
by
  sorry

end total_dogs_l253_253683


namespace find_product_of_roots_plus_one_l253_253406

-- Define the problem conditions
variables (x1 x2 : ℝ)
axiom sum_roots : x1 + x2 = 3
axiom prod_roots : x1 * x2 = 2

-- State the theorem corresponding to the proof problem
theorem find_product_of_roots_plus_one : (x1 + 1) * (x2 + 1) = 6 :=
by 
  sorry

end find_product_of_roots_plus_one_l253_253406


namespace variance_binom_4_half_l253_253264

-- Define the binomial variance function
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Define the conditions
def n := 4
def p := 1 / 2

-- The target statement
theorem variance_binom_4_half : binomial_variance n p = 1 :=
by
  -- The proof goes here
  sorry

end variance_binom_4_half_l253_253264


namespace total_money_collected_is_140_l253_253513

def total_attendees : ℕ := 280
def child_attendees : ℕ := 80
def adult_attendees : ℕ := total_attendees - child_attendees
def adult_ticket_cost : ℝ := 0.60
def child_ticket_cost : ℝ := 0.25

def money_collected_from_adults : ℝ := adult_attendees * adult_ticket_cost
def money_collected_from_children : ℝ := child_attendees * child_ticket_cost
def total_money_collected : ℝ := money_collected_from_adults + money_collected_from_children

theorem total_money_collected_is_140 : total_money_collected = 140 := by
  sorry

end total_money_collected_is_140_l253_253513


namespace infinite_series_sum_value_l253_253807

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253807


namespace exists_nat_numbers_except_two_three_l253_253869

theorem exists_nat_numbers_except_two_three (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ (k ≠ 2 ∧ k ≠ 3) :=
by
  sorry

end exists_nat_numbers_except_two_three_l253_253869


namespace find_ab_l253_253316

noncomputable def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

theorem find_ab (a b : ℝ) :
  (f 1 a b = 10) ∧ ((3 * 1^2 - 2 * a * 1 - b = 0)) → (a, b) = (-4, 11) ∨ (a, b) = (3, -3) :=
by
  sorry

end find_ab_l253_253316


namespace posters_total_l253_253612

-- Definitions based on conditions
def Mario_posters : Nat := 18
def Samantha_posters : Nat := Mario_posters + 15

-- Statement to prove: They made 51 posters altogether
theorem posters_total : Mario_posters + Samantha_posters = 51 := 
by sorry

end posters_total_l253_253612


namespace simplify_expression_l253_253015

theorem simplify_expression (x y : ℝ) :
  (3 * x^2 * y)^3 + (4 * x * y) * y^4 = 27 * x^6 * y^3 + 4 * x * y^5 :=
by 
  sorry

end simplify_expression_l253_253015


namespace train_crosses_bridge_in_30_seconds_l253_253891

noncomputable def train_length : ℝ := 100
noncomputable def bridge_length : ℝ := 200
noncomputable def train_speed_kmph : ℝ := 36

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def total_distance : ℝ := train_length + bridge_length

noncomputable def crossing_time : ℝ := total_distance / train_speed_mps

theorem train_crosses_bridge_in_30_seconds :
  crossing_time = 30 := 
by
  sorry

end train_crosses_bridge_in_30_seconds_l253_253891


namespace inequality_solution_l253_253083

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x ∈ (Set.Ioi 0 ∩ Set.Iic (1/2)) ∪ (Set.Ioi 1.5 ∩ Set.Iio 2)) 
  ↔ ( (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ) := by
  sorry

end inequality_solution_l253_253083


namespace isosceles_right_triangle_C_coordinates_l253_253283

theorem isosceles_right_triangle_C_coordinates :
  ∃ C : ℝ × ℝ, (let A : ℝ × ℝ := (1, 0)
                let B : ℝ × ℝ := (3, 1) 
                ∃ (x y: ℝ), C = (x, y) ∧ 
                ((x-1)^2 + y^2 = 10) ∧ 
                (((x-3)^2 + (y-1)^2 = 10))) ∨
                ((x = 2 ∧ y = 3) ∨ (x = 4 ∧ y = -1)) :=
by
  sorry

end isosceles_right_triangle_C_coordinates_l253_253283


namespace smallest_x_l253_253384

theorem smallest_x (x : ℕ) (M : ℕ) (h : 1800 * x = M^3) :
  x = 30 :=
by
  sorry

end smallest_x_l253_253384


namespace greatest_is_B_l253_253640

def A : ℕ := 95 - 35
def B : ℕ := A + 12
def C : ℕ := B - 19

theorem greatest_is_B : B = 72 ∧ (B > A ∧ B > C) :=
by {
  -- Proof steps would be written here to prove the theorem.
  sorry
}

end greatest_is_B_l253_253640


namespace find_m_l253_253117

theorem find_m (x1 x2 m : ℝ) (h_eq : ∀ x, x^2 + x + m = 0 → (x = x1 ∨ x = x2))
  (h_abs : |x1| + |x2| = 3)
  (h_sum : x1 + x2 = -1)
  (h_prod : x1 * x2 = m) :
  m = -2 :=
sorry

end find_m_l253_253117


namespace jose_is_21_l253_253002

-- Define the ages of the individuals based on the conditions
def age_of_inez := 12
def age_of_zack := age_of_inez + 4
def age_of_jose := age_of_zack + 5

-- State the proposition we want to prove
theorem jose_is_21 : age_of_jose = 21 := 
by 
  sorry

end jose_is_21_l253_253002


namespace series_sum_eq_l253_253731

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253731


namespace count_repeating_decimals_l253_253252

theorem count_repeating_decimals (s : Set ℕ) :
  (∀ n, n ∈ s ↔ 1 ≤ n ∧ n ≤ 20 ∧ ¬∃ k, k * 3 = n) →
  (s.card = 14) :=
by 
  sorry

end count_repeating_decimals_l253_253252


namespace largest_k_exists_l253_253114

theorem largest_k_exists (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n → (c - b) ≥ k ∧ (b - a) ≥ k ∧ (a + b ≥ c + 1)) ∧ 
  (k = (n - 1) / 3) :=
  sorry

end largest_k_exists_l253_253114


namespace boys_and_girls_in_class_l253_253426

theorem boys_and_girls_in_class (b g : ℕ) (h1 : b + g = 21) (h2 : 5 * b + 2 * g = 69) 
: b = 9 ∧ g = 12 := by
  sorry

end boys_and_girls_in_class_l253_253426


namespace minimum_value_inequality_l253_253476

theorem minimum_value_inequality
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y - 3 = 0) :
  ∃ t : ℝ, (∀ (x y : ℝ), (2 * x + y = 3) → (0 < x) → (0 < y) → (t = (4 * y - x + 6) / (x * y)) → 9 ≤ t) ∧
          (∃ (x_ y_: ℝ), 2 * x_ + y_ = 3 ∧ 0 < x_ ∧ 0 < y_ ∧ (4 * y_ - x_ + 6) / (x_ * y_) = 9) :=
sorry

end minimum_value_inequality_l253_253476


namespace shaded_percentage_of_grid_l253_253336

def percent_shaded (total_squares shaded_squares : ℕ) : ℚ :=
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100

theorem shaded_percentage_of_grid :
  percent_shaded 36 16 = 44.44 :=
by 
  sorry

end shaded_percentage_of_grid_l253_253336


namespace tan_135_eq_neg1_l253_253539

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 :=
by
  have h_cos : Real.cos (135 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180) := 
    by 
      apply Real.cos_angle_of_pi_sub_angle; 
      sorry
  have h_cos_45 : Real.cos (45 * Real.pi / 180) = (1 / Real.sqrt 2) := 
    by 
      rw Real.cos_pi_div_four;
      sorry
  have h_sin : Real.sin (135 * Real.pi / 180) = Real.sin (45 * Real.pi / 180) := 
    by
      apply Real.sin_of_pi_sub_angle;
      sorry
  have h_sin_45 : Real.sin (45 * Real.pi / 180) = (1 / Real.sqrt 2) := 
    by 
      rw Real.sin_pi_div_four;
      sorry
  rw [← h_sin, h_sin_45, ← h_cos, h_cos_45]
  rw [Real.tan_eq_sin_div_cos, div_eq_mul_inv, mul_comm, inv_mul_cancel]
  norm_num
  sorry

end tan_135_eq_neg1_l253_253539


namespace repeated_digit_percentage_l253_253274

noncomputable def percentage_repeated_digits : ℝ :=
  let total_numbers := 90000
  let non_repeated_digits := 9 * 9 * 8 * 7 * 6
  let repeated_digits := total_numbers - non_repeated_digits
  ((repeated_digits.toReal / total_numbers.toReal) * 100).round / 10

theorem repeated_digit_percentage (y : ℝ) : 
  percentage_repeated_digits = y → y = 69.8 :=
by
  intro h
  have : percentage_repeated_digits = 69.8 := sorry
  rw this at h
  exact h

end repeated_digit_percentage_l253_253274


namespace terry_tomato_types_l253_253630

theorem terry_tomato_types (T : ℕ) (h1 : 2 * T * 4 * 2 = 48) : T = 3 :=
by
  -- Proof goes here
  sorry

end terry_tomato_types_l253_253630


namespace land_area_in_acres_l253_253159

-- Define the conditions given in the problem.
def length_cm : ℕ := 30
def width_cm : ℕ := 20
def scale_cm_to_mile : ℕ := 1  -- 1 cm corresponds to 1 mile.
def sq_mile_to_acres : ℕ := 640  -- 1 square mile corresponds to 640 acres.

-- Define the statement to be proved.
theorem land_area_in_acres :
  (length_cm * width_cm * sq_mile_to_acres) = 384000 := 
  by sorry

end land_area_in_acres_l253_253159


namespace expected_ones_on_three_dice_l253_253967

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l253_253967


namespace altitude_of_isosceles_triangle_l253_253190

noncomputable def radius_X (C : ℝ) := C / (2 * Real.pi)
noncomputable def radius_Y (radius_X : ℝ) := radius_X
noncomputable def a (radius_Y : ℝ) := radius_Y / 2

-- Define the theorem to be proven
theorem altitude_of_isosceles_triangle (C : ℝ) (h_C : C = 14 * Real.pi) (radius_X := radius_X C) (radius_Y := radius_Y radius_X) (a := a radius_Y) :
  ∃ h : ℝ, h = a * Real.sqrt 3 :=
sorry

end altitude_of_isosceles_triangle_l253_253190


namespace sum_of_series_eq_three_fourths_l253_253778

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253778


namespace sum_series_div_3_powers_l253_253762

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253762


namespace sum_of_squares_of_consecutive_even_numbers_l253_253987

theorem sum_of_squares_of_consecutive_even_numbers :
  ∃ (x : ℤ), x + (x + 2) + (x + 4) + (x + 6) = 36 → (x ^ 2 + (x + 2) ^ 2 + (x + 4) ^ 2 + (x + 6) ^ 2 = 344) :=
by
  sorry

end sum_of_squares_of_consecutive_even_numbers_l253_253987


namespace expected_ones_three_dice_l253_253969

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l253_253969


namespace find_f_of_f_neg2_l253_253555

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f_of_f_neg2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end find_f_of_f_neg2_l253_253555


namespace triangle_BPC_area_l253_253132

universe u

variables {T : Type u} [LinearOrderedField T]

-- Define the points
variables (A B C E F P : T)
variables (area : T → T → T → T) -- A function to compute the area of a triangle

-- Hypotheses
def conditions :=
  E ∈ [A, B] ∧
  F ∈ [A, C] ∧
  (∃ P, P ∈ [B, F] ∧ P ∈ [C, E]) ∧
  area A E P + area E P F + area P F A = 4 ∧ -- AEPF
  area B E P = 4 ∧ -- BEP
  area C F P = 4   -- CFP

-- The theorem to prove
theorem triangle_BPC_area (h : conditions A B C E F P area) : area B P C = 12 :=
sorry

end triangle_BPC_area_l253_253132


namespace HCF_of_two_numbers_l253_253038

theorem HCF_of_two_numbers (a b : ℕ) (h1 : a * b = 2562) (h2 : Nat.lcm a b = 183) : Nat.gcd a b = 14 := 
by
  sorry

end HCF_of_two_numbers_l253_253038


namespace cost_of_antibiotics_for_a_week_l253_253517

noncomputable def antibiotic_cost : ℕ := 3
def doses_per_day : ℕ := 3
def days_in_week : ℕ := 7

theorem cost_of_antibiotics_for_a_week : doses_per_day * days_in_week * antibiotic_cost = 63 :=
by
  sorry

end cost_of_antibiotics_for_a_week_l253_253517


namespace arithmetic_sequence_sum_l253_253432

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (n : ℕ) 
  (h_arith : ∀ n, a (n+1) = a n + 3)
  (h_a1_a2 : a 1 + a 2 = 7)
  (h_a3 : a 3 = 8)
  (h_bn : ∀ n, b n = 1 / (a n * a (n+1)))
  :
  (∀ n, a n = 3 * n - 1) ∧ (T n = n / (2 * (3 * n + 2))) :=
by 
  sorry

end arithmetic_sequence_sum_l253_253432


namespace maximum_area_of_garden_l253_253076

theorem maximum_area_of_garden (w l : ℝ) 
  (h_perimeter : 2 * w + l = 400) : 
  ∃ (A : ℝ), A = 20000 ∧ A = w * l ∧ l = 400 - 2 * w ∧ ∀ (w' : ℝ) (l' : ℝ),
    2 * w' + l' = 400 → w' * l' ≤ 20000 :=
by
  sorry

end maximum_area_of_garden_l253_253076


namespace no_positive_integer_solutions_l253_253164

theorem no_positive_integer_solutions (x n r : ℕ) (h1 : x > 1) (h2 : x > 0) (h3 : n > 0) (h4 : r > 0) :
  ¬(x^(2*n + 1) = 2^r + 1 ∨ x^(2*n + 1) = 2^r - 1) :=
sorry

end no_positive_integer_solutions_l253_253164


namespace quadratic_solution_l253_253524

theorem quadratic_solution 
  (x : ℝ)
  (h : x^2 - 2 * x - 1 = 0) : 
  x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
sorry

end quadratic_solution_l253_253524


namespace betty_height_correct_l253_253687

-- Definitions for the conditions
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height_inches : ℕ := carter_height - 12
def betty_height_feet : ℕ := betty_height_inches / 12

-- Theorem that we need to prove
theorem betty_height_correct : betty_height_feet = 3 :=
by
  sorry

end betty_height_correct_l253_253687


namespace first_person_days_l253_253628

-- Define the condition that Tanya is 25% more efficient than the first person and that Tanya takes 12 days to do the work.
def tanya_more_efficient (x : ℕ) : Prop :=
  -- Efficiency relationship: tanya (12 days) = 3 days less than the first person
  12 = x - (x / 4)

-- Define the theorem that the first person takes 15 days to do the work
theorem first_person_days : ∃ x : ℕ, tanya_more_efficient x ∧ x = 15 := 
by
  sorry -- proof is not required

end first_person_days_l253_253628


namespace series_sum_l253_253736

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253736


namespace sum_of_series_eq_three_fourths_l253_253783

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253783


namespace sum_series_equals_three_fourths_l253_253820

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253820


namespace temperature_on_Friday_l253_253019

variable {M T W Th F : ℝ}

theorem temperature_on_Friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (hM : M = 41) :
  F = 33 :=
by
  -- Proof goes here
  sorry

end temperature_on_Friday_l253_253019


namespace infinite_series_sum_eq_3_over_4_l253_253851

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253851


namespace sum_series_equals_three_fourths_l253_253817

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253817


namespace infinite_series_sum_value_l253_253806

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253806


namespace value_of_xyz_l253_253920

theorem value_of_xyz (x y z : ℂ) 
  (h1 : x * y + 5 * y = -20)
  (h2 : y * z + 5 * z = -20)
  (h3 : z * x + 5 * x = -20) :
  x * y * z = 80 := 
by
  sorry

end value_of_xyz_l253_253920


namespace jessica_cut_roses_l253_253482

/-- There were 13 roses and 84 orchids in the vase. Jessica cut some more roses and 
orchids from her flower garden. There are now 91 orchids and 14 roses in the vase. 
How many roses did she cut? -/
theorem jessica_cut_roses :
  let initial_roses := 13
  let new_roses := 14
  ∃ cut_roses : ℕ, new_roses = initial_roses + cut_roses ∧ cut_roses = 1 :=
by
  sorry

end jessica_cut_roses_l253_253482


namespace molly_christmas_shipping_cost_l253_253438

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_sisters_in_law_per_brother : ℕ := 1
def num_children_per_brother : ℕ := 2

def total_relatives : ℕ :=
  num_parents + num_brothers + (num_brothers * num_sisters_in_law_per_brother) + (num_brothers * num_children_per_brother)

theorem molly_christmas_shipping_cost : total_relatives * cost_per_package = 70 :=
by
  sorry

end molly_christmas_shipping_cost_l253_253438


namespace find_function_l253_253865

variable (R : Type) [LinearOrderedField R]

theorem find_function
  (f : R → R)
  (h : ∀ x y : R, f (x + y) + y ≤ f (f (f x))) :
  ∃ c : R, ∀ x : R, f x = c - x :=
sorry

end find_function_l253_253865


namespace sum_series_div_3_powers_l253_253761

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253761


namespace find_age_of_b_l253_253655

variable (a b : ℤ)

-- Conditions
axiom cond1 : a + 10 = 2 * (b - 10)
axiom cond2 : a = b + 9

-- Goal
theorem find_age_of_b : b = 39 :=
sorry

end find_age_of_b_l253_253655


namespace factor_x12_minus_4096_l253_253527

theorem factor_x12_minus_4096 (x : ℝ) : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) :=
by
  sorry

end factor_x12_minus_4096_l253_253527


namespace surface_area_of_circumscribed_sphere_l253_253694

/-- 
  Problem: Determine the surface area of the sphere circumscribed about a cube with edge length 2.

  Given:
  - The edge length of the cube is 2.
  - The space diagonal of a cube with edge length \(a\) is given by \(d = \sqrt{3} \cdot a\).
  - The diameter of the circumscribed sphere is equal to the space diagonal of the cube.
  - The surface area \(S\) of a sphere with radius \(R\) is given by \(S = 4\pi R^2\).

  To Prove:
  - The surface area of the sphere circumscribed about the cube is \(12\pi\).
-/
theorem surface_area_of_circumscribed_sphere (a : ℝ) (π : ℝ) (h1 : a = 2) 
  (h2 : ∀ a, d = Real.sqrt 3 * a) (h3 : ∀ d, R = d / 2) (h4 : ∀ R, S = 4 * π * R^2) : 
  S = 12 * π := 
by
  sorry

end surface_area_of_circumscribed_sphere_l253_253694


namespace expected_ones_in_three_dice_rolls_l253_253945

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l253_253945


namespace gcd_2_pow_2010_minus_3_2_pow_2001_minus_3_l253_253331

-- Definitions based on conditions
def a := (2:ℤ) ^ 2010 - 3
def b := (2:ℤ) ^ 2001 - 3

-- The proof statement
theorem gcd_2_pow_2010_minus_3_2_pow_2001_minus_3 :
  Int.gcd a b = 1533 := by
  sorry

end gcd_2_pow_2010_minus_3_2_pow_2001_minus_3_l253_253331


namespace fraction_value_l253_253105

theorem fraction_value (x : ℝ) (h₀ : x^2 - 3 * x - 1 = 0) (h₁ : x ≠ 0) : 
  x^2 / (x^4 + x^2 + 1) = 1 / 12 := 
by
  sorry

end fraction_value_l253_253105


namespace series_converges_to_three_fourths_l253_253703

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253703


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253855

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253855


namespace series_converges_to_three_fourths_l253_253702

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253702


namespace suitable_chart_for_air_composition_l253_253642

/-- Given that air is a mixture of various gases, prove that the most suitable
    type of statistical chart to depict this data, while introducing it
    succinctly and effectively, is a pie chart. -/
theorem suitable_chart_for_air_composition :
  ∀ (air_composition : String) (suitable_for_introduction : String → Prop),
  (air_composition = "mixture of various gases") →
  (suitable_for_introduction "pie chart") →
  suitable_for_introduction "pie chart" :=
by
  intros air_composition suitable_for_introduction h_air_composition h_pie_chart
  sorry

end suitable_chart_for_air_composition_l253_253642


namespace sum_series_equals_three_fourths_l253_253819

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253819


namespace find_principal_l253_253654

theorem find_principal
  (P R : ℝ)
  (h : (P * (R + 2) * 7) / 100 = (P * R * 7) / 100 + 140) :
  P = 1000 := by
sorry

end find_principal_l253_253654


namespace find_reflection_line_l253_253188

/-*
Triangle ABC has vertices with coordinates A(2,3), B(7,8), and C(-4,6).
The triangle is reflected about line L.
The image points are A'(2,-5), B'(7,-10), and C'(-4,-8).
Prove that the equation of line L is y = -1.
*-/
theorem find_reflection_line :
  ∃ (L : ℝ), (∀ (x : ℝ), (∃ (k : ℝ), L = k) ∧ (L = -1)) :=
by sorry

end find_reflection_line_l253_253188


namespace zero_intersections_l253_253405

noncomputable def Line : Type := sorry  -- Define Line as a type
noncomputable def is_skew (a b : Line) : Prop := sorry  -- Predicate for skew lines
noncomputable def is_common_perpendicular (EF a b : Line) : Prop := sorry  -- Predicate for common perpendicular
noncomputable def is_parallel (l EF : Line) : Prop := sorry  -- Predicate for parallel lines
noncomputable def count_intersections (l a b : Line) : ℕ := sorry  -- Function to count intersections

theorem zero_intersections (EF a b l : Line) 
  (h_skew : is_skew a b) 
  (h_common_perpendicular : is_common_perpendicular EF a b)
  (h_parallel : is_parallel l EF) : 
  count_intersections l a b = 0 := 
sorry

end zero_intersections_l253_253405


namespace train_speed_correct_l253_253511

noncomputable def train_speed_kmh (length : ℝ) (time : ℝ) (conversion_factor : ℝ) : ℝ :=
  (length / time) * conversion_factor

theorem train_speed_correct 
  (length : ℝ := 350) 
  (time : ℝ := 8.7493) 
  (conversion_factor : ℝ := 3.6) : 
  train_speed_kmh length time conversion_factor = 144.02 := 
sorry

end train_speed_correct_l253_253511


namespace find_t_l253_253392

def vector (α : Type) : Type := (α × α)

def dot_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_t (t : ℝ) :
  let a : vector ℝ := (1, -1)
  let b : vector ℝ := (2, t)
  orthogonal a b → t = 2 := by
  sorry

end find_t_l253_253392


namespace skill_testing_question_l253_253327

theorem skill_testing_question : (5 * (10 - 6) / 2) = 10 := by
  sorry

end skill_testing_question_l253_253327


namespace sum_series_eq_3_over_4_l253_253752

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253752


namespace find_radius_of_base_of_cone_l253_253176

noncomputable def radius_of_cone (CSA : ℝ) (l : ℝ) : ℝ :=
  CSA / (Real.pi * l)

theorem find_radius_of_base_of_cone :
  radius_of_cone 527.7875658030853 14 = 12 :=
by
  sorry

end find_radius_of_base_of_cone_l253_253176


namespace last_digit_3_pow_1991_plus_1991_pow_3_l253_253982

theorem last_digit_3_pow_1991_plus_1991_pow_3 :
  (3 ^ 1991 + 1991 ^ 3) % 10 = 8 :=
  sorry

end last_digit_3_pow_1991_plus_1991_pow_3_l253_253982


namespace closest_ratio_l253_253499

theorem closest_ratio
  (a_0 : ℝ)
  (h_pos : a_0 > 0)
  (a_10 : ℝ)
  (h_eq : a_10 = a_0 * (1 + 0.05) ^ 10) :
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.5) ∧
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.7) ∧
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.8) := 
sorry

end closest_ratio_l253_253499


namespace unique_solution_quadratic_l253_253169

theorem unique_solution_quadratic (x : ℚ) (b : ℚ) (h_b_nonzero : b ≠ 0) (h_discriminant_zero : 625 - 36 * b = 0) : 
  (b = 625 / 36) ∧ (x = -18 / 25) → b * x^2 + 25 * x + 9 = 0 :=
by 
  -- We assume b = 625 / 36 and x = -18 / 25
  rintro ⟨hb, hx⟩
  -- Substitute b and x into the quadratic equation and simplify
  rw [hb, hx]
  -- Show the left-hand side evaluates to zero
  sorry

end unique_solution_quadratic_l253_253169


namespace smallest_n_for_g_n_eq_4_l253_253293

/-- 
  Let g(n) be the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n.
  Prove that the smallest positive integer n for which g(n) = 4 is 25.
-/
def g (n : ℕ) : ℕ :=
  (finset.univ.product finset.univ).filter (λ (ab : ℕ × ℕ), ab.1 ^ 2 + ab.2 ^ 2 = n ∧ ab.1 ≠ ab.2).card

theorem smallest_n_for_g_n_eq_4 :
  ∃ n : ℕ, g n = 4 ∧ (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 25
  sorry

end smallest_n_for_g_n_eq_4_l253_253293


namespace geometric_series_sum_l253_253522

  theorem geometric_series_sum :
    let a := (1 / 4 : ℚ)
    let r := (1 / 4 : ℚ)
    let n := 4
    let S_n := a * (1 - r^n) / (1 - r)
    S_n = 255 / 768 := by
  sorry
  
end geometric_series_sum_l253_253522


namespace find_x_l253_253210

theorem find_x (x n q r : ℕ) (h_n : n = 220080) (h_sum : n = (x + 445) * (2 * (x - 445)) + r) (h_r : r = 80) : 
  x = 555 :=
by
  have eq1 : n = 220080 := h_n
  have eq2 : n =  (x + 445) * (2 * (x - 445)) + r := h_sum
  have eq3 : r = 80 := h_r
  sorry

end find_x_l253_253210


namespace weight_loss_challenge_l253_253047

noncomputable def percentage_weight_loss (W : ℝ) : ℝ :=
  ((W - (0.918 * W)) / W) * 100

theorem weight_loss_challenge (W : ℝ) (h : W > 0) :
  percentage_weight_loss W = 8.2 :=
by
  sorry

end weight_loss_challenge_l253_253047


namespace expected_ones_three_standard_dice_l253_253957

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l253_253957


namespace mia_receives_chocolate_l253_253151

-- Given conditions
def total_chocolate : ℚ := 72 / 7
def piles : ℕ := 6
def piles_to_Mia : ℕ := 2

-- Weight of one pile
def weight_of_one_pile (total_chocolate : ℚ) (piles : ℕ) := total_chocolate / piles

-- Total weight Mia receives
def mia_chocolate (weight_of_one_pile : ℚ) (piles_to_Mia : ℕ) := piles_to_Mia * weight_of_one_pile

theorem mia_receives_chocolate : mia_chocolate (weight_of_one_pile total_chocolate piles) piles_to_Mia = 24 / 7 :=
by
  sorry

end mia_receives_chocolate_l253_253151


namespace repeating_decimals_for_n_div_18_l253_253244

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end repeating_decimals_for_n_div_18_l253_253244


namespace sets_of_three_teams_l253_253136

-- Definitions based on the conditions
def total_teams : ℕ := 20
def won_games : ℕ := 12
def lost_games : ℕ := 7

-- Main theorem to prove
theorem sets_of_three_teams : 
  (total_teams * (total_teams - 1) * (total_teams - 2)) / 6 / 2 = 570 := by
  sorry

end sets_of_three_teams_l253_253136


namespace tan_135_eq_neg1_l253_253543

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 :=
by
  have h1 : 135 * Real.pi / 180 = Real.pi - Real.pi / 4 := by norm_num
  rw [h1, Real.tan_sub_pi_div_two]
  norm_num
  sorry

end tan_135_eq_neg1_l253_253543


namespace probability_different_colors_l253_253278

def total_chips : ℕ := 12

def blue_chips : ℕ := 5
def yellow_chips : ℕ := 3
def red_chips : ℕ := 4

def prob_diff_color (x y : ℕ) : ℚ :=
(x / total_chips) * (y / total_chips) + (y / total_chips) * (x / total_chips)

theorem probability_different_colors :
  prob_diff_color blue_chips yellow_chips +
  prob_diff_color blue_chips red_chips +
  prob_diff_color yellow_chips red_chips = 47 / 72 := by
sorry

end probability_different_colors_l253_253278


namespace find_three_digit_number_l253_253936

theorem find_three_digit_number (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : 0 ≤ c ∧ c ≤ 9)
    (h₄ : (10 * a + b) / 99 + (100 * a + 10 * b + c) / 999 = 33 / 37) :
    100 * a + 10 * b + c = 447 :=
sorry

end find_three_digit_number_l253_253936


namespace parabola_shift_units_l253_253633

theorem parabola_shift_units (h : ℝ) :
  (∃ h, (0 + 3 - h)^2 - 1 = 0) ↔ (h = 2 ∨ h = 4) :=
by 
  sorry

end parabola_shift_units_l253_253633


namespace series_sum_eq_l253_253726

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253726


namespace series_sum_correct_l253_253794

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253794


namespace base4_arithmetic_l253_253520

theorem base4_arithmetic : 
  ∀ (a b c : ℕ),
  a = 2 * 4^2 + 3 * 4^1 + 1 * 4^0 →
  b = 2 * 4^1 + 4 * 4^0 →
  c = 3 * 4^0 →
  (a * b) / c = 2 * 4^3 + 3 * 4^2 + 1 * 4^1 + 0 * 4^0 :=
by
  intros a b c ha hb hc
  sorry

end base4_arithmetic_l253_253520


namespace jeffrey_walks_to_mailbox_l253_253492

theorem jeffrey_walks_to_mailbox :
  ∀ (D total_steps net_gain_per_set steps_per_set sets net_gain : ℕ),
    steps_per_set = 3 ∧ 
    net_gain = 1 ∧ 
    total_steps = 330 ∧ 
    net_gain_per_set = net_gain ∧ 
    sets = total_steps / steps_per_set ∧ 
    D = sets * net_gain →
    D = 110 :=
by
  intro D total_steps net_gain_per_set steps_per_set sets net_gain
  intro h
  sorry

end jeffrey_walks_to_mailbox_l253_253492


namespace sum_of_first_six_terms_of_geom_seq_l253_253875

theorem sum_of_first_six_terms_of_geom_seq :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let S6 := a * (1 - r^6) / (1 - r)
  S6 = 4095 / 12288 := by
sorry

end sum_of_first_six_terms_of_geom_seq_l253_253875


namespace series_sum_correct_l253_253785

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253785


namespace greatest_integer_x_l253_253194

theorem greatest_integer_x (x : ℤ) : (5 : ℚ)/8 > (x : ℚ)/15 → x ≤ 9 :=
by {
  sorry
}

end greatest_integer_x_l253_253194


namespace find_x_l253_253100

theorem find_x (x y : ℕ) (h1 : y = 144) (h2 : x^3 * 6^2 / 432 = y) : x = 12 := 
by
  sorry

end find_x_l253_253100


namespace red_lettuce_cost_l253_253142

-- Define the known conditions
def cost_per_pound : Nat := 2
def total_pounds : Nat := 7
def cost_green_lettuce : Nat := 8

-- Define the total cost calculation
def total_cost : Nat := total_pounds * cost_per_pound
def cost_red_lettuce : Nat := total_cost - cost_green_lettuce

-- Statement to prove: cost_red_lettuce = 6
theorem red_lettuce_cost :
  cost_red_lettuce = 6 :=
by
  sorry

end red_lettuce_cost_l253_253142


namespace nonagon_line_segments_not_adjacent_l253_253120

def nonagon_segments (n : ℕ) : ℕ :=
(n * (n - 3)) / 2

theorem nonagon_line_segments_not_adjacent (h : ∃ n, n = 9) :
  nonagon_segments 9 = 27 :=
by
  -- proof omitted
  sorry

end nonagon_line_segments_not_adjacent_l253_253120


namespace total_bars_is_7_l253_253230

variable (x : ℕ)

-- Each chocolate bar costs $3
def cost_per_bar := 3

-- Olivia sold all but 4 bars
def bars_sold (total_bars : ℕ) := total_bars - 4

-- Olivia made $9
def amount_made (total_bars : ℕ) := cost_per_bar * bars_sold total_bars

-- Given conditions
def condition1 (total_bars : ℕ) := amount_made total_bars = 9

-- Proof that the total number of bars is 7
theorem total_bars_is_7 : condition1 x -> x = 7 := by
  sorry

end total_bars_is_7_l253_253230


namespace remaining_battery_life_l253_253222

theorem remaining_battery_life :
  let capacity1 := 60
  let capacity2 := 80
  let capacity3 := 120
  let used1 := capacity1 * (3 / 4 : ℚ)
  let used2 := capacity2 * (1 / 2 : ℚ)
  let used3 := capacity3 * (2 / 3 : ℚ)
  let remaining1 := capacity1 - used1 - 2
  let remaining2 := capacity2 - used2 - 2
  let remaining3 := capacity3 - used3 - 2
  remaining1 + remaining2 + remaining3 = 89 := 
by
  sorry

end remaining_battery_life_l253_253222


namespace total_and_average_games_l253_253914

def football_games_per_month : List Nat := [29, 35, 48, 43, 56, 36]
def baseball_games_per_month : List Nat := [15, 19, 23, 14, 18, 17]
def basketball_games_per_month : List Nat := [17, 21, 14, 32, 22, 27]

def total_games (games_per_month : List Nat) : Nat :=
  List.sum games_per_month

def average_games (total : Nat) (months : Nat) : Nat :=
  total / months

theorem total_and_average_games :
  total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month = 486
  ∧ average_games (total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month) 6 = 81 :=
by
  sorry

end total_and_average_games_l253_253914


namespace luke_plays_14_rounds_l253_253611

theorem luke_plays_14_rounds (total_points : ℕ) (points_per_round : ℕ)
  (h1 : total_points = 154) (h2 : points_per_round = 11) : 
  total_points / points_per_round = 14 := by
  sorry

end luke_plays_14_rounds_l253_253611


namespace altitudes_sum_eq_l253_253651

variables {α : Type*} [LinearOrderedField α]

structure Triangle (α) :=
(A B C : α)
(R : α)   -- circumradius
(r : α)   -- inradius

variables (T : Triangle α)
(A B C : α)
(m n p : α)  -- points on respective arcs
(h1 h2 h3 : α)  -- altitudes of the segments

theorem altitudes_sum_eq (T : Triangle α) (A B C m n p h1 h2 h3 : α) :
  h1 + h2 + h3 = 2 * T.R - T.r :=
sorry

end altitudes_sum_eq_l253_253651


namespace num_repeating_decimals_1_to_20_l253_253246

theorem num_repeating_decimals_1_to_20 : 
  (∃ count_repeating : ℕ, count_repeating = 18 ∧ 
    ∀ n, 1 ≤ n ∧ n ≤ 20 → ((∃ k, n = 9 * k ∨ n = 18 * k) → false) → 
        (∃ d, (∃ normalized, n / 18 = normalized ∧ normalized.has_repeating_decimal))) :=
sorry

end num_repeating_decimals_1_to_20_l253_253246


namespace expected_number_of_ones_l253_253948

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l253_253948


namespace ax_by_n_sum_l253_253902

theorem ax_by_n_sum {a b x y : ℝ} 
  (h1 : a * x + b * y = 2)
  (h2 : a * x^2 + b * y^2 = 5)
  (h3 : a * x^3 + b * y^3 = 15)
  (h4 : a * x^4 + b * y^4 = 35) :
  a * x^5 + b * y^5 = 10 :=
sorry

end ax_by_n_sum_l253_253902


namespace log_xy_l253_253402

-- Definitions from conditions
def log (z : ℝ) : ℝ := sorry -- Assume a definition of log function
variables (x y : ℝ)
axiom h1 : log (x^2 * y^2) = 1
axiom h2 : log (x^3 * y) = 2

-- The proof goal
theorem log_xy (x y : ℝ) (h1 : log (x^2 * y^2) = 1) (h2 : log (x^3 * y) = 2) : log (x * y) = 1/2 :=
sorry

end log_xy_l253_253402


namespace sum_series_div_3_powers_l253_253760

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253760


namespace vector_c_solution_l253_253573

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def vector_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_c_solution
  (a b c : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (2, -3))
  (h3 : vector_parallel (c.1 + 1, c.2 + 2) b)
  (h4 : vector_perpendicular c (3, -1)) :
  c = (-7/9, -7/3) :=
sorry

end vector_c_solution_l253_253573


namespace sufficient_but_not_necessary_condition_for_prop_l253_253024

theorem sufficient_but_not_necessary_condition_for_prop :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
sorry

end sufficient_but_not_necessary_condition_for_prop_l253_253024


namespace unique_shape_determination_l253_253158

theorem unique_shape_determination (ratio_sides_median : Prop) (ratios_three_sides : Prop) 
                                   (ratio_circumradius_side : Prop) (ratio_two_angles : Prop) 
                                   (length_one_side_heights : Prop) :
  ¬(ratio_circumradius_side → (ratio_sides_median ∧ ratios_three_sides ∧ ratio_two_angles ∧ length_one_side_heights)) := 
sorry

end unique_shape_determination_l253_253158


namespace discount_rate_pony_jeans_l253_253203

theorem discount_rate_pony_jeans
  (fox_price pony_price : ℕ)
  (fox_pairs pony_pairs : ℕ)
  (total_savings total_discount_rate : ℕ)
  (F P : ℕ)
  (h1 : fox_price = 15)
  (h2 : pony_price = 20)
  (h3 : fox_pairs = 3)
  (h4 : pony_pairs = 2)
  (h5 : total_savings = 9)
  (h6 : total_discount_rate = 22)
  (h7 : F + P = total_discount_rate)
  (h8 : fox_pairs * fox_price * F / 100 + pony_pairs * pony_price * P / 100 = total_savings) : 
  P = 18 :=
sorry

end discount_rate_pony_jeans_l253_253203


namespace largest_arithmetic_seq_3digit_l253_253196

theorem largest_arithmetic_seq_3digit : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (∃ a b c : ℕ, n = 100*a + 10*b + c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a = 9 ∧ ∃ d, b = a - d ∧ c = a - 2*d) ∧ n = 963 :=
by sorry

end largest_arithmetic_seq_3digit_l253_253196


namespace expected_value_of_ones_on_three_dice_l253_253940

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l253_253940


namespace average_visitors_on_Sundays_l253_253507

theorem average_visitors_on_Sundays (S : ℕ) 
  (visitors_other_days : ℕ := 240)
  (avg_per_day : ℕ := 285)
  (days_in_month : ℕ := 30)
  (month_starts_with_sunday : true) :
  let num_sundays := 5
  let num_other_days := days_in_month - num_sundays
  let total_visitors := (num_sundays * S) + (num_other_days * visitors_other_days)
  total_visitors = avg_per_day * days_in_month → S = 510 := 
by
  intros _ _ _ _ _ total_visitors_eq
  sorry

end average_visitors_on_Sundays_l253_253507


namespace sum_of_perimeters_l253_253017

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) : 
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * x + 4 * y := by
  sorry

end sum_of_perimeters_l253_253017


namespace jasmine_money_left_l253_253443

theorem jasmine_money_left 
  (initial_amount : ℝ)
  (apple_cost : ℝ) (num_apples : ℕ)
  (orange_cost : ℝ) (num_oranges : ℕ)
  (pear_cost : ℝ) (num_pears : ℕ)
  (h_initial : initial_amount = 100.00)
  (h_apple_cost : apple_cost = 1.50)
  (h_num_apples : num_apples = 5)
  (h_orange_cost : orange_cost = 2.00)
  (h_num_oranges : num_oranges = 10)
  (h_pear_cost : pear_cost = 2.25)
  (h_num_pears : num_pears = 4) : 
  initial_amount - (num_apples * apple_cost + num_oranges * orange_cost + num_pears * pear_cost) = 63.50 := 
by 
  sorry

end jasmine_money_left_l253_253443


namespace evening_campers_l253_253498

theorem evening_campers (morning_campers afternoon_campers total_campers : ℕ) (h_morning : morning_campers = 36) (h_afternoon : afternoon_campers = 13) (h_total : total_campers = 98) :
  total_campers - (morning_campers + afternoon_campers) = 49 :=
by
  sorry

end evening_campers_l253_253498


namespace initial_number_of_friends_l253_253032

theorem initial_number_of_friends (F : ℕ) (h : 6 * (F + 2) = 60) : F = 8 :=
by {
  sorry
}

end initial_number_of_friends_l253_253032


namespace arithmetic_seq_sum_ratio_l253_253107

theorem arithmetic_seq_sum_ratio
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : S 25 / a 23 = 5)
  (h3 : S 45 / a 33 = 25) :
  S 65 / a 43 = 45 :=
by sorry

end arithmetic_seq_sum_ratio_l253_253107


namespace fourth_term_geometric_series_l253_253140

theorem fourth_term_geometric_series (a₁ a₅ : ℕ) (r : ℕ) :
  a₁ = 6 → a₅ = 1458 → (∀ n, aₙ = a₁ * r^(n-1)) → r = 3 → (∃ a₄, a₄ = a₁ * r^(4-1) ∧ a₄ = 162) :=
by intros h₁ h₅ H r_sol
   sorry

end fourth_term_geometric_series_l253_253140


namespace part1_part2_part3_l253_253564

noncomputable def m := -1
noncomputable def cosAlpha := -((2 * Real.sqrt 5) / 5)
noncomputable def sinAlpha := -(Real.sqrt 5 / 5)
noncomputable def tanAlpha := 1 / 2

theorem part1 (α : ℝ) (A : ℝ × ℝ) (hA : A = (-2, m)) (h_sin : Real.sin α = sinAlpha) : 
  A.snd = m := 
by
  unfold m
  sorry

theorem part2 (α : ℝ) (h_sin : Real.sin α = sinAlpha) : 
  Real.cos α = cosAlpha := 
by
  unfold cosAlpha sinAlpha
  sorry

theorem part3 (α : ℝ) (h_sin : Real.sin α = sinAlpha) (h_cos : Real.cos α = cosAlpha) : 
  (Real.cos ((Real.pi / 2) + α) * Real.sin (-Real.pi - α)) / (Real.cos ((11 * Real.pi / 2) - α) * Real.sin ((9 * Real.pi / 2) + α)) = (1 / 2) :=
by
  unfold sinAlpha cosAlpha
  sorry

end part1_part2_part3_l253_253564


namespace gcd_of_factorials_l253_253521

-- Define factorials
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define 7!
def seven_factorial : ℕ := factorial 7

-- Define (11! / 4!)
def eleven_div_four_factorial : ℕ := factorial 11 / factorial 4

-- GCD function based on prime factorization (though a direct gcd function also exists, we follow the steps)
def prime_factorization_gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Proof statement
theorem gcd_of_factorials : prime_factorization_gcd seven_factorial eleven_div_four_factorial = 5040 := by
  sorry

end gcd_of_factorials_l253_253521


namespace range_of_k_l253_253213

theorem range_of_k (k : Real) : 
  (∀ (x y : Real), x^2 + y^2 - 12 * x - 4 * y + 37 = 0)
  → ((k < -Real.sqrt 2) ∨ (k > Real.sqrt 2)) :=
by
  sorry

end range_of_k_l253_253213


namespace series_sum_l253_253737

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253737


namespace teams_face_each_other_l253_253586

theorem teams_face_each_other (n : ℕ) (total_games : ℕ) (k : ℕ)
  (h1 : n = 20)
  (h2 : total_games = 760)
  (h3 : total_games = n * (n - 1) * k / 2) :
  k = 4 :=
by
  sorry

end teams_face_each_other_l253_253586


namespace max_individual_score_l253_253054

open Nat

theorem max_individual_score (n : ℕ) (total_points : ℕ) (minimum_points : ℕ) (H1 : n = 12) (H2 : total_points = 100) (H3 : ∀ i : Fin n, 7 ≤ minimum_points) :
  ∃ max_points : ℕ, max_points = 23 :=
by 
  sorry

end max_individual_score_l253_253054


namespace disproof_of_Alitta_l253_253677

-- Definition: A prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition: A number is odd
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

-- The value is a specific set of odd primes including 11
def contains (p : ℕ) : Prop :=
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11

-- Main statement: There exists an odd prime p in the given options such that p^2 - 2 is not a prime
theorem disproof_of_Alitta :
  ∃ p : ℕ, contains p ∧ is_prime p ∧ is_odd p ∧ ¬ is_prime (p^2 - 2) :=
by
  sorry

end disproof_of_Alitta_l253_253677


namespace molly_total_cost_l253_253436

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_children_per_brother : ℕ := 2
def num_spouse_per_brother : ℕ := 1

def total_num_relatives : ℕ := 
  let parents_and_siblings := num_parents + num_brothers
  let additional_relatives := num_brothers * (1 + num_spouse_per_brother + num_children_per_brother)
  parents_and_siblings + additional_relatives

def total_cost : ℕ :=
  total_num_relatives * cost_per_package

theorem molly_total_cost : total_cost = 85 := sorry

end molly_total_cost_l253_253436


namespace betty_height_in_feet_l253_253686

theorem betty_height_in_feet (dog_height carter_height betty_height : ℕ) (h1 : dog_height = 24) 
  (h2 : carter_height = 2 * dog_height) (h3 : betty_height = carter_height - 12) : betty_height / 12 = 3 :=
by
  sorry

end betty_height_in_feet_l253_253686


namespace vampires_after_two_nights_l253_253641

def initial_population : ℕ := 300
def initial_vampires : ℕ := 3
def conversion_rate : ℕ := 7

theorem vampires_after_two_nights :
  let first_night := initial_vampires * conversion_rate
  let total_first_night := initial_vampires + first_night
  let second_night := total_first_night * conversion_rate
  let total_second_night := total_first_night + second_night
  total_second_night = 192 :=
by
  let first_night := initial_vampires * conversion_rate
  let total_first_night := initial_vampires + first_night
  let second_night := total_first_night * conversion_rate
  let total_second_night := total_first_night + second_night
  have h1 : first_night = 21 := rfl
  have h2 : total_first_night = 24 := rfl
  have h3 : second_night = 168 := rfl
  have h4 : total_second_night = 192 := rfl
  exact rfl

end vampires_after_two_nights_l253_253641


namespace rectangle_perimeter_l253_253059

theorem rectangle_perimeter (a b : ℤ) (h1 : a ≠ b) (h2 : 2 * (2 * a + 2 * b) - a * b = 12) : 2 * (a + b) = 26 :=
sorry

end rectangle_perimeter_l253_253059


namespace shortest_paths_in_grid_l253_253545

-- Define a function that computes the binomial coefficient
def binom (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

-- Proof problem: Prove that the number of shortest paths in an m x n grid is binom(m, n)
theorem shortest_paths_in_grid (m n : ℕ) : binom m n = Nat.choose (m + n) n :=
by
  -- Intentionally left blank: proof is skipped
  sorry

end shortest_paths_in_grid_l253_253545


namespace cost_to_make_each_pop_l253_253215

-- Define the conditions as given in step a)
def selling_price : ℝ := 1.50
def pops_sold : ℝ := 300
def pencil_cost : ℝ := 1.80
def pencils_to_buy : ℝ := 100

-- Define the total revenue from selling the ice-pops
def total_revenue : ℝ := pops_sold * selling_price

-- Define the total cost to buy the pencils
def total_pencil_cost : ℝ := pencils_to_buy * pencil_cost

-- Define the total profit
def total_profit : ℝ := total_revenue - total_pencil_cost

-- Define the cost to make each ice-pop
theorem cost_to_make_each_pop : total_profit / pops_sold = 0.90 :=
by
  sorry

end cost_to_make_each_pop_l253_253215


namespace polynomial_is_linear_l253_253558

theorem polynomial_is_linear (a : ℕ → ℝ) (n : ℕ) (h_rec : ∀ i : ℕ, 1 ≤ i → a (i - 1) + a (i + 1) = 2 * a i) (h_diff : a 0 ≠ a 1) :
  ∃ c d : ℝ, ∀ x : ℝ, (a n * (1 - x)^n + ∑ i in Finset.range n, a i * Nat.choose n i * x^i * (1 - x)^(n - i)) = c + d * x :=
by
  sorry

end polynomial_is_linear_l253_253558


namespace trains_cross_time_l253_253974

theorem trains_cross_time
  (length_each_train : ℝ)
  (speed_each_train_kmh : ℝ)
  (relative_speed_m_s : ℝ)
  (total_distance : ℝ)
  (conversion_factor : ℝ) :
  length_each_train = 120 →
  speed_each_train_kmh = 27 →
  conversion_factor = 1000 / 3600 →
  relative_speed_m_s = speed_each_train_kmh * conversion_factor →
  total_distance = 2 * length_each_train →
  total_distance / relative_speed_m_s = 16 :=
by
  sorry

end trains_cross_time_l253_253974


namespace percent_fair_hair_l253_253661

theorem percent_fair_hair (total_employees : ℕ) (total_women_fair_hair : ℕ)
  (percent_fair_haired_women : ℕ) (percent_women_fair_hair : ℕ)
  (h1 : total_women_fair_hair = (total_employees * percent_women_fair_hair) / 100)
  (h2 : percent_fair_haired_women * total_women_fair_hair = total_employees * 10) :
  (25 * total_employees = 100 * total_women_fair_hair) :=
by {
  sorry
}

end percent_fair_hair_l253_253661


namespace minimum_expression_value_l253_253089

theorem minimum_expression_value (a b c : ℝ) (hbpos : b > 0) (hab : b > a) (hcb : b > c) (hca : c > a) :
  (a + 2 * b) ^ 2 / b ^ 2 + (b - 2 * c) ^ 2 / b ^ 2 + (c - 2 * a) ^ 2 / b ^ 2 ≥ 65 / 16 := 
sorry

end minimum_expression_value_l253_253089


namespace abs_neg_seven_l253_253470

theorem abs_neg_seven : |(-7 : ℤ)| = 7 := by
  sorry

end abs_neg_seven_l253_253470


namespace area_of_inscribed_triangle_l253_253363

theorem area_of_inscribed_triangle 
  (x : ℝ) 
  (h1 : (2:ℝ) * x ≤ (3:ℝ) * x ∧ (3:ℝ) * x ≤ (4:ℝ) * x) 
  (h2 : (4:ℝ) * x = 2 * 4) :
  ∃ (area : ℝ), area = 12.00 :=
by
  sorry

end area_of_inscribed_triangle_l253_253363


namespace remainder_of_N_mod_16_is_7_l253_253602

-- Let N be the product of all odd primes less than 16
def odd_primes : List ℕ := [3, 5, 7, 11, 13]

-- Calculate the product N of these primes
def N : ℕ := odd_primes.foldr (· * ·) 1

-- Prove the remainder of N when divided by 16 is 7
theorem remainder_of_N_mod_16_is_7 : N % 16 = 7 := by
  sorry

end remainder_of_N_mod_16_is_7_l253_253602


namespace point_in_second_quadrant_l253_253911

-- Definitions for the coordinates of the points
def A : ℤ × ℤ := (3, 2)
def B : ℤ × ℤ := (-3, -2)
def C : ℤ × ℤ := (3, -2)
def D : ℤ × ℤ := (-3, 2)

-- Definition for the second quadrant condition
def isSecondQuadrant (p : ℤ × ℤ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- The theorem we need to prove
theorem point_in_second_quadrant : isSecondQuadrant D :=
by
  sorry

end point_in_second_quadrant_l253_253911


namespace part1_solution_set_part2_minimum_value_l253_253413

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

theorem part1_solution_set (x : ℝ) :
  (f x ≥ -1) ↔ (2 / 3 ≤ x ∧ x ≤ 6) := sorry

variables {a b c : ℝ}
theorem part2_minimum_value (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = 6) :
  (1 / (2 * a + b) + 1 / (2 * a + c) ≥ 2 / 3) := 
sorry

end part1_solution_set_part2_minimum_value_l253_253413


namespace infinite_series_sum_eq_l253_253722

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253722


namespace series_sum_eq_l253_253725

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253725


namespace smallest_constant_inequality_l253_253552

open Real

theorem smallest_constant_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
    sqrt (x / (y + z + w)) + sqrt (y / (x + z + w)) + sqrt (z / (x + y + w)) + sqrt (w / (x + y + z)) ≤ 2 := by
  sorry

end smallest_constant_inequality_l253_253552


namespace diagonal_length_of_regular_hexagon_l253_253088

-- Define a structure for the hexagon with a given side length
structure RegularHexagon (s : ℝ) :=
(side_length : ℝ := s)

-- Prove that the length of diagonal DB in a regular hexagon with side length 12 is 12√3
theorem diagonal_length_of_regular_hexagon (H : RegularHexagon 12) : 
  ∃ DB : ℝ, DB = 12 * Real.sqrt 3 :=
by
  sorry

end diagonal_length_of_regular_hexagon_l253_253088


namespace rectangle_in_right_triangle_dimensions_l253_253622

theorem rectangle_in_right_triangle_dimensions :
  ∀ (DE EF DF x y : ℝ),
  DE = 6 → EF = 8 → DF = 10 →
  -- Assuming isosceles right triangle (interchange sides for the proof)
  ∃ (G H I J : ℝ),
  (G = 0 ∧ H = 0 ∧ I = y ∧ J = x ∧ x * y = GH * GI) → -- Rectangle GH parallel to DE
  (x = 10 / 8 * y) →
  ∃ (GH GI : ℝ), 
  GH = 8 / 8.33 ∧ GI = 6.67 / 8.33 →
  (x = 25 / 3 ∧ y = 40 / 6) :=
by
  sorry

end rectangle_in_right_triangle_dimensions_l253_253622


namespace problem_statement_l253_253452

-- Define the universal set
def U : Set ℕ := {x | x ≤ 6}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {4, 5, 6}

-- Define the complement of A with respect to U
def complement_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- Define the intersection of the complement of A and B
def intersect_complement_A_B : Set ℕ := {x | x ∈ complement_A ∧ x ∈ B}

-- Theorem statement to be proven
theorem problem_statement : intersect_complement_A_B = {4, 6} :=
by
  sorry

end problem_statement_l253_253452


namespace retirement_hire_year_l253_253992

theorem retirement_hire_year (A : ℕ) (R : ℕ) (Y : ℕ) (W : ℕ) 
  (h1 : A + W = 70) 
  (h2 : A = 32) 
  (h3 : R = 2008) 
  (h4 : W = R - Y) : Y = 1970 :=
by
  sorry

end retirement_hire_year_l253_253992


namespace foil_covered_prism_width_l253_253031

def inner_prism_length (l : ℝ) := l
def inner_prism_width (l : ℝ) := 2 * l
def inner_prism_height (l : ℝ) := l
def inner_prism_volume (l : ℝ) := l * (2 * l) * l

theorem foil_covered_prism_width :
  (∃ l : ℝ, inner_prism_volume l = 128) → (inner_prism_width l + 2 = 8) := by
sorry

end foil_covered_prism_width_l253_253031


namespace sequence_formula_l253_253890

theorem sequence_formula (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_formula_l253_253890


namespace abs_neg_six_l253_253635

theorem abs_neg_six : abs (-6) = 6 :=
sorry

end abs_neg_six_l253_253635


namespace evaluate_series_sum_l253_253772

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253772


namespace Billy_weighs_more_l253_253684

-- Variables and assumptions
variable (Billy Brad Carl : ℕ)
variable (b_weight : Billy = 159)
variable (c_weight : Carl = 145)
variable (brad_formula : Brad = Carl + 5)

-- Theorem statement to prove the required condition
theorem Billy_weighs_more :
  Billy - Brad = 9 :=
by
  -- Here we put the proof steps, but it's omitted as per instructions.
  sorry

end Billy_weighs_more_l253_253684


namespace ian_investment_percentage_change_l253_253277

theorem ian_investment_percentage_change :
  let initial_investment := 200
  let first_year_loss := 0.10
  let second_year_gain := 0.25
  let amount_after_loss := initial_investment * (1 - first_year_loss)
  let amount_after_gain := amount_after_loss * (1 + second_year_gain)
  let percentage_change := (amount_after_gain - initial_investment) / initial_investment * 100
  percentage_change = 12.5 := 
by
  sorry

end ian_investment_percentage_change_l253_253277


namespace find_x_l253_253275

variables (a b c d x : ℤ)

theorem find_x (h1 : a - b = c + d + 9) (h2 : a - c = 3) (h3 : a + b = c - d - x) : x = 3 :=
sorry

end find_x_l253_253275


namespace math_problem_l253_253551

theorem math_problem:
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 :=
by
  sorry

end math_problem_l253_253551


namespace minimum_value_of_2m_plus_n_solution_set_for_inequality_l253_253889

namespace MathProof

-- Definitions and conditions
def f (x m n : ℝ) : ℝ := |x + m| + |2 * x - n|

-- Part (I)
theorem minimum_value_of_2m_plus_n
  (m n : ℝ)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_f_nonneg : ∀ x : ℝ, f x m n ≥ 1) :
  2 * m + n ≥ 2 :=
sorry

-- Part (II)
theorem solution_set_for_inequality
  (x : ℝ) :
  (f x 2 3 > 5 ↔ (x < 0 ∨ x > 2)) :=
sorry

end MathProof

end minimum_value_of_2m_plus_n_solution_set_for_inequality_l253_253889


namespace exp_decreasing_function_range_l253_253421

theorem exp_decreasing_function_range (a : ℝ) (x : ℝ) (h_a : 0 < a ∧ a < 1) (h_f : a^(x+1) ≥ 1) : x ≤ -1 :=
sorry

end exp_decreasing_function_range_l253_253421


namespace quadratic_always_positive_l253_253086

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 3) * x - 2 * k + 12 > 0) ↔ -7 < k ∧ k < 5 :=
sorry

end quadratic_always_positive_l253_253086


namespace salon_visitors_l253_253368

noncomputable def total_customers (x : ℕ) : ℕ :=
  let revenue_customers_with_one_visit := 10 * x
  let revenue_customers_with_two_visits := 30 * 18
  let revenue_customers_with_three_visits := 10 * 26
  let total_revenue := revenue_customers_with_one_visit + revenue_customers_with_two_visits + revenue_customers_with_three_visits
  if total_revenue = 1240 then
    x + 30 + 10
  else
    0

theorem salon_visitors : 
  ∃ x, total_customers x = 84 :=
by
  use 44
  sorry

end salon_visitors_l253_253368


namespace minimum_pizzas_needed_l253_253001

variables (p : ℕ)

def income_per_pizza : ℕ := 12
def gas_cost_per_pizza : ℕ := 4
def maintenance_cost_per_pizza : ℕ := 1
def car_cost : ℕ := 6500

theorem minimum_pizzas_needed :
  p ≥ 929 ↔ (income_per_pizza * p - (gas_cost_per_pizza + maintenance_cost_per_pizza) * p) ≥ car_cost :=
sorry

end minimum_pizzas_needed_l253_253001


namespace triangle_side_ratio_triangle_area_l253_253904

-- Definition of Problem 1
theorem triangle_side_ratio {A B C a b c : ℝ} 
  (h1 : 4 * Real.sin A = 3 * Real.sin B)
  (h2 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h3 : a / b = Real.sin A / Real.sin B)
  (h4 : b / c = Real.sin B / Real.sin C)
  : c / b = 5 / 4 :=
sorry

-- Definition of Problem 2
theorem triangle_area {A B C a b c : ℝ} 
  (h1 : C = 2 * Real.pi / 3)
  (h2 : c - a = 8)
  (h3 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h4 : a + c = 2 * b)
  : (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 :=
sorry

end triangle_side_ratio_triangle_area_l253_253904


namespace hire_charges_paid_by_b_l253_253344

theorem hire_charges_paid_by_b (total_cost : ℕ) (hours_a : ℕ) (hours_b : ℕ) (hours_c : ℕ) 
  (total_hours : ℕ) (cost_per_hour : ℕ) : 
  total_cost = 520 → hours_a = 7 → hours_b = 8 → hours_c = 11 → total_hours = hours_a + hours_b + hours_c 
  → cost_per_hour = total_cost / total_hours → 
  (hours_b * cost_per_hour) = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hire_charges_paid_by_b_l253_253344


namespace tan_135_eq_neg1_l253_253531

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg1_l253_253531


namespace general_term_sequence_sum_of_cn_l253_253922

theorem general_term_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n) :
  ∀ n, a n = n :=
by
  sorry

theorem sum_of_cn (S : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n)
  (ha : ∀ n, a n = n)
  (hc_odd : ∀ n, c (2 * n - 1) = a (2 * n))
  (hc_even : ∀ n, c (2 * n) = 3 * 2^(a (2 * n - 1)) + 1) :
  ∀ n, T (2 * n) = 2^(2 * n + 1) + n^2 + 2 * n - 2 :=
by
  sorry

end general_term_sequence_sum_of_cn_l253_253922


namespace third_number_correct_l253_253981

-- Given that the row of Pascal's triangle with 51 numbers corresponds to the binomial coefficients of 50.
def third_number_in_51_pascal_row : ℕ := Nat.choose 50 2

-- Prove that the third number in this row of Pascal's triangle is 1225.
theorem third_number_correct : third_number_in_51_pascal_row = 1225 := 
by 
  -- Calculation part can be filled in for the full proof.
  sorry

end third_number_correct_l253_253981


namespace g_difference_l253_253451

noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((3 + Real.sqrt 3) / 6) ^ n + (3 - 2 * Real.sqrt 3) / 6 * ((3 - Real.sqrt 3) / 6) ^ n

theorem g_difference (n : ℕ) : g (n + 2) - g n = (1 / 4) * g n := 
sorry

end g_difference_l253_253451


namespace sum_series_eq_3_over_4_l253_253754

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253754


namespace ben_points_l253_253985

theorem ben_points (B : ℕ) 
  (h1 : 42 = B + 21) : B = 21 := 
by 
-- Proof can be filled in here
sorry

end ben_points_l253_253985


namespace part1_range_of_a_part2_area_l253_253050

theorem part1_range_of_a (a : ℝ) : 
  (∃ t : ℝ, a ≠ 0 ∧ -1/6 < a ∧ (a > 0 ∨ (a > -1/6 ∧ a < 0)) ∧ 
    (∃ t : ℝ, (2 * a - (8 / 9) * t ^ 2) ^ 2 = 4 * (a ^ 2 + (16 / 27) * t ^ 3) ∧ 
    t = (3 + Real.sqrt (9 + 54 * a)) / 2)) → -1/6 < a < 0 ∨ a > 0 :=
sorry

theorem part2_area (a : ℝ) (h : -1/6 < a < 0 ∨ a > 0) : 
  ∀ x1 x2 : ℝ, x2 = -a + (4 / 9) * ((3 + Real.sqrt (9 + 54 * a)) / 2) ^ 2 - x1 → 
  x1 = -a + (4 / 9) * ((3 - Real.sqrt (9 + 54 * a)) / 2) ^ 2 → 
  ∫ x in x1..x2, (x - x1)^2 + ∫ x in x1..x2, (x - x2)^2 = (16 / 3) * (2 * a + 1) ^ (3/2) :=
sorry

end part1_range_of_a_part2_area_l253_253050


namespace greatest_perfect_square_less_than_500_has_odd_factors_l253_253453

-- We need to state that a number has an odd number of positive factors if and only if it is a perfect square
lemma odd_factors_iff_perfect_square (n : ℕ) :
  (∃ m, m * m = n) ↔ (∃ k, k * k = n) :=
by sorry

-- Define the specific problem conditions
def is_perfect_square (n : ℕ) : Prop := ∃ m, m * m = n

def less_than_500 (n : ℕ) : Prop := n < 500

-- Final statement combining the conditions and conclusion
theorem greatest_perfect_square_less_than_500_has_odd_factors :
  ∃ n, is_perfect_square n ∧ less_than_500 n ∧ ∀ m, (is_perfect_square m ∧ less_than_500 m) → m ≤ n ∧ n = 484 :=
by sorry

end greatest_perfect_square_less_than_500_has_odd_factors_l253_253453


namespace union_A_B_eq_C_l253_253562

open Set

variable {R : Type} [LinearOrderedField R] (x : R)

def A : Set R := { x | x^2 + 5 * x - 6 < 0 }
def B : Set R := { x | x^2 - 5 * x - 6 < 0 }
def C : Set R := { x | -6 < x ∧ x < 6 }

theorem union_A_B_eq_C : (A ∪ B) = C := by
  sorry

end union_A_B_eq_C_l253_253562


namespace problem_statement_l253_253109

noncomputable def a := 2 * Real.sqrt 2
noncomputable def b := 2
def ellipse_eq (x y : ℝ) := (x^2) / 8 + (y^2) / 4 = 1
def line_eq (x y m : ℝ) := y = x + m
def circle_eq (x y : ℝ) := x^2 + y^2 = 1

theorem problem_statement (x1 y1 x2 y2 x0 y0 m : ℝ) (h1 : ellipse_eq x1 y1) (h2 : ellipse_eq x2 y2) 
  (hm : line_eq x0 y0 m) (h0 : (x1 + x2) / 2 = -2 * m / 3) (h0' : (y1 + y2) / 2 = m / 3) : 
  (ellipse_eq x y ∧ line_eq x y m ∧ circle_eq x0 y0) → m = (3 * Real.sqrt 5) / 5 ∨ m = -(3 * Real.sqrt 5) / 5 := 
by {
  sorry
}

end problem_statement_l253_253109


namespace expected_number_of_ones_l253_253950

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l253_253950


namespace quadratic_real_roots_l253_253130

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_l253_253130


namespace pizza_ratio_l253_253592

/-- Define a function that represents the ratio calculation -/
def ratio (a b : ℕ) : ℕ × ℕ := (a / (Nat.gcd a b), b / (Nat.gcd a b))

/-- State the main problem to be proved -/
theorem pizza_ratio (total_slices friend_eats james_eats remaining_slices gcd : ℕ)
  (h1 : total_slices = 8)
  (h2 : friend_eats = 2)
  (h3 : james_eats = 3)
  (h4 : remaining_slices = total_slices - friend_eats)
  (h5 : gcd = Nat.gcd james_eats remaining_slices)
  (h6 : ratio james_eats remaining_slices = (1, 2)) :
  ratio james_eats remaining_slices = (1, 2) :=
by
  sorry

end pizza_ratio_l253_253592


namespace find_m_l253_253112

theorem find_m (x y m : ℤ) (h1 : x = 1) (h2 : y = -1) (h3 : 2 * x + m + y = 0) : m = -1 := by
  -- Proof can be completed here
  sorry

end find_m_l253_253112


namespace sum_of_series_eq_three_fourths_l253_253782

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253782


namespace gas_pressure_inversely_proportional_l253_253681

theorem gas_pressure_inversely_proportional :
  ∀ (p v : ℝ), (p * v = 27.2) → (8 * 3.4 = 27.2) → (v = 6.8) → p = 4 :=
by
  intros p v h1 h2 h3
  have h4 : 27.2 = 8 * 3.4 := by sorry
  have h5 : p * 6.8 = 27.2 := by sorry
  exact sorry

end gas_pressure_inversely_proportional_l253_253681


namespace prev_geng_yin_year_2010_is_1950_l253_253063

def heavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def earthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "You", "Xu", "Hai"]

def cycleLength : Nat := Nat.lcm 10 12

def prev_geng_yin_year (current_year : Nat) : Nat :=
  if cycleLength ≠ 0 then
    current_year - cycleLength
  else
    current_year -- This line is just to handle the case where LCM is incorrectly zero, which shouldn't happen practically.

theorem prev_geng_yin_year_2010_is_1950 : prev_geng_yin_year 2010 = 1950 := by
  sorry

end prev_geng_yin_year_2010_is_1950_l253_253063


namespace sum_of_series_eq_three_fourths_l253_253780

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253780


namespace expected_value_of_ones_on_three_dice_l253_253941

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l253_253941


namespace boat_distance_l253_253353

theorem boat_distance (v_b : ℝ) (v_s : ℝ) (t_downstream : ℝ) (t_upstream : ℝ) (d : ℝ) :
  v_b = 7 ∧ t_downstream = 2 ∧ t_upstream = 5 ∧ d = (v_b + v_s) * t_downstream ∧ d = (v_b - v_s) * t_upstream → d = 20 :=
by {
  sorry
}

end boat_distance_l253_253353


namespace analytical_expression_f_l253_253569

def f : ℝ → ℝ := sorry

theorem analytical_expression_f :
  (∀ x : ℝ, f (x + 2) = x^2 - x + 1) →
  (∀ y : ℝ, f y = y^2 - 5*y + 7) :=
by
  sorry

end analytical_expression_f_l253_253569


namespace perfect_squares_count_in_range_l253_253575

theorem perfect_squares_count_in_range :
  ∃ (n : ℕ), (
    (∀ (k : ℕ), (50 < k^2 ∧ k^2 < 500) → (8 ≤ k ∧ k ≤ 22)) ∧
    (15 = 22 - 8 + 1)
  ) := sorry

end perfect_squares_count_in_range_l253_253575


namespace eunice_pots_l253_253235

theorem eunice_pots (total_seeds pots_with_3_seeds last_pot_seeds : ℕ)
  (h1 : total_seeds = 10)
  (h2 : pots_with_3_seeds * 3 + last_pot_seeds = total_seeds)
  (h3 : last_pot_seeds = 1) : pots_with_3_seeds + 1 = 4 :=
by
  -- Proof omitted
  sorry

end eunice_pots_l253_253235


namespace expression_for_f_in_positive_domain_l253_253895

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def given_f (x : ℝ) : ℝ :=
  if x < 0 then 3 * Real.sin x + 4 * Real.cos x + 1 else 0 -- temp def for Lean proof

theorem expression_for_f_in_positive_domain (f : ℝ → ℝ) (h_odd : is_odd_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = 3 * Real.sin x + 4 * Real.cos x + 1) :
  ∀ x : ℝ, x > 0 → f x = 3 * Real.sin x - 4 * Real.cos x - 1 :=
by
  intros x hx_pos
  sorry

end expression_for_f_in_positive_domain_l253_253895


namespace series_sum_l253_253738

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253738


namespace solve_system_eqns_l253_253930

noncomputable def eq1 (x y z : ℚ) : Prop := x^2 + 2 * y * z = x
noncomputable def eq2 (x y z : ℚ) : Prop := y^2 + 2 * z * x = y
noncomputable def eq3 (x y z : ℚ) : Prop := z^2 + 2 * x * y = z

theorem solve_system_eqns (x y z : ℚ) :
  (eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) ↔
  ((x, y, z) = (0, 0, 0) ∨
   (x, y, z) = (1/3, 1/3, 1/3) ∨
   (x, y, z) = (1, 0, 0) ∨
   (x, y, z) = (0, 1, 0) ∨
   (x, y, z) = (0, 0, 1) ∨
   (x, y, z) = (2/3, -1/3, -1/3) ∨
   (x, y, z) = (-1/3, 2/3, -1/3) ∨
   (x, y, z) = (-1/3, -1/3, 2/3)) :=
by sorry

end solve_system_eqns_l253_253930


namespace evaluate_expression_l253_253223

theorem evaluate_expression :
  (π - 2023) ^ 0 + |(-9)| - 3 ^ 2 = 1 :=
by
  sorry

end evaluate_expression_l253_253223


namespace complement_intersection_l253_253298

open Set

variable (I : Set ℕ) (A B : Set ℕ)

-- Given the universal set and specific sets A and B
def universal_set : Set ℕ := {1,2,3,4,5}
def set_A : Set ℕ := {2,3,5}
def set_B : Set ℕ := {1,2}

-- To prove that the complement of B in I intersects A to be {3,5}
theorem complement_intersection :
  (universal_set \ set_B) ∩ set_A = {3,5} :=
sorry

end complement_intersection_l253_253298


namespace solution_l253_253146

noncomputable def problem_statement : ℝ :=
  let a := 6
  let b := 5
  let x := 10 * a + b
  let y := 10 * b + a
  let m := 16.5
  x + y + m

theorem solution : problem_statement = 137.5 :=
by
  sorry

end solution_l253_253146


namespace sum_of_series_eq_three_fourths_l253_253776

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253776


namespace largest_arithmetic_seq_3digit_l253_253195

theorem largest_arithmetic_seq_3digit : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (∃ a b c : ℕ, n = 100*a + 10*b + c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a = 9 ∧ ∃ d, b = a - d ∧ c = a - 2*d) ∧ n = 963 :=
by sorry

end largest_arithmetic_seq_3digit_l253_253195


namespace cost_of_fencing_l253_253323

/-- The sides of a rectangular field are in the ratio 3:4.
If the area of the field is 10092 sq. m and the cost of fencing the field is 25 paise per meter,
then the cost of fencing the field is 101.5 rupees. --/
theorem cost_of_fencing (area : ℕ) (fencing_cost : ℝ) (ratio1 ratio2 perimeter : ℝ)
  (h_area : area = 10092)
  (h_ratio : ratio1 = 3 ∧ ratio2 = 4)
  (h_fencing_cost : fencing_cost = 0.25)
  (h_perimeter : perimeter = 406) :
  perimeter * fencing_cost = 101.5 := by
  sorry

end cost_of_fencing_l253_253323


namespace series_result_l253_253835

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253835


namespace sum_series_eq_3_over_4_l253_253750

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253750


namespace pencil_length_after_sharpening_l253_253000

-- Definition of the initial length of the pencil
def initial_length : ℕ := 22

-- Definition of the amount sharpened each day
def sharpened_each_day : ℕ := 2

-- Final length of the pencil after sharpening on Monday and Tuesday
def final_length (initial_length : ℕ) (sharpened_each_day : ℕ) : ℕ :=
  initial_length - sharpened_each_day * 2

-- Theorem stating that the final length is 18 inches
theorem pencil_length_after_sharpening : final_length initial_length sharpened_each_day = 18 := by
  sorry

end pencil_length_after_sharpening_l253_253000


namespace deepak_age_l253_253049

-- Defining the problem with the given conditions in Lean:
theorem deepak_age (x : ℕ) (rahul_current : ℕ := 4 * x) (deepak_current : ℕ := 3 * x) :
  (rahul_current + 6 = 38) → (deepak_current = 24) :=
by
  sorry

end deepak_age_l253_253049


namespace range_of_c_l253_253879

variable (c : ℝ)

def p : Prop := ∀ x : ℝ, x > 0 → c^x = c^(x+1) / c
def q : Prop := ∀ x : ℝ, (1/2 ≤ x ∧ x ≤ 2) → x + 1/x > 1/c

theorem range_of_c (h1 : c > 0) (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) :
  (0 < c ∧ c ≤ 1/2) ∨ (c ≥ 1) :=
sorry

end range_of_c_l253_253879


namespace interval_of_monotonic_increase_fx_lt_x_minus_1_k_range_values_l253_253410

noncomputable def f (x : ℝ) : ℝ := log x - (x - 1)^2 / 2

theorem interval_of_monotonic_increase :
  {x : ℝ | 0 < x ∧ x < (1 + Real.sqrt 5) / 2}.Nonempty := sorry

theorem fx_lt_x_minus_1 (x : ℝ) (h : 1 < x) : f x < x - 1 := sorry

theorem k_range_values (k : ℝ) : (k < 1) ↔ ∃ x0 > 1, ∀ x ∈ Set.Ioo 1 x0, f x > k * (x - 1) := sorry

end interval_of_monotonic_increase_fx_lt_x_minus_1_k_range_values_l253_253410


namespace cube_sum_of_quadratic_roots_ratio_sum_of_quadratic_roots_l253_253566

theorem cube_sum_of_quadratic_roots (x₁ x₂ : ℝ) (h : x₁^2 - 3 * x₁ + 1 = 0) (h' : x₂^2 - 3 * x₂ + 1 = 0) :
  x₁^3 + x₂^3 = 18 :=
sorry

theorem ratio_sum_of_quadratic_roots (x₁ x₂ : ℝ) (h : x₁^2 - 3 * x₁ + 1 = 0) (h' : x₂^2 - 3 * x₂ + 1 = 0) :
  (x₂ / x₁) + (x₁ / x₂) = 7 :=
sorry

end cube_sum_of_quadratic_roots_ratio_sum_of_quadratic_roots_l253_253566


namespace sum_series_equals_three_fourths_l253_253818

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253818


namespace garden_perimeter_is_24_l253_253011

def perimeter_of_garden(a b c x: ℕ) (h1: a + b + c = 3) : ℕ :=
  3 + 5 + a + x + b + 4 + c + 4 + 5 - x

theorem garden_perimeter_is_24 (a b c x : ℕ) (h1 : a + b + c = 3) :
  perimeter_of_garden a b c x h1 = 24 :=
  by
  sorry

end garden_perimeter_is_24_l253_253011


namespace cylinder_ratio_max_volume_l253_253234

theorem cylinder_ratio_max_volume 
    (l w : ℝ) 
    (r : ℝ) 
    (h : ℝ)
    (H_perimeter : 2 * l + 2 * w = 12)
    (H_length_circumference : l = 2 * π * r)
    (H_width_height : w = h) :
    (∀ V : ℝ, V = π * r^2 * h) →
    (∀ r : ℝ, r = 2 / π) →
    ((2 * π * r) / h = 2) :=
sorry

end cylinder_ratio_max_volume_l253_253234


namespace largest_distinct_arithmetic_sequence_number_l253_253198

theorem largest_distinct_arithmetic_sequence_number :
  ∃ a b c d : ℕ, 
    (100 * a + 10 * b + c = 789) ∧ 
    (b - a = d) ∧ 
    (c - b = d) ∧ 
    (a ≠ b) ∧ 
    (b ≠ c) ∧ 
    (a ≠ c) ∧ 
    (a < 10) ∧ 
    (b < 10) ∧ 
    (c < 10) :=
sorry

end largest_distinct_arithmetic_sequence_number_l253_253198


namespace parts_of_alloys_l253_253679

def ratio_of_metals_in_alloy (a1 a2 a3 b1 b2 : ℚ) (x y : ℚ) : Prop :=
  let first_metal := (1 / a3) * x + (a1 / b2) * y
  let second_metal := (2 / a3) * x + (b1 / b2) * y
  (first_metal / second_metal) = (17 / 27)

theorem parts_of_alloys
  (x y : ℚ)
  (a1 a2 a3 b1 b2 : ℚ)
  (h1 : a1 = 1)
  (h2 : a2 = 2)
  (h3 : a3 = 3)
  (h4 : b1 = 2)
  (h5 : b2 = 5)
  (h6 : ratio_of_metals_in_alloy a1 a2 a3 b1 b2 x y) :
  x = 9 ∧ y = 35 :=
sorry

end parts_of_alloys_l253_253679


namespace sum_series_eq_3_div_4_l253_253830

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253830


namespace seating_arrangement_l253_253909

theorem seating_arrangement (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) : 
  (∃ n : ℕ, n = (boys.factorial * girls.factorial) + (girls.factorial * boys.factorial) ∧ n = 288) :=
by 
  sorry

end seating_arrangement_l253_253909


namespace find_parallelepiped_dimensions_l253_253241

theorem find_parallelepiped_dimensions :
  ∃ (x y z : ℕ),
    (x * y * z = 2 * (x * y + y * z + z * x)) ∧
    (x = 6 ∧ y = 6 ∧ z = 6 ∨
     x = 5 ∧ y = 5 ∧ z = 10 ∨
     x = 4 ∧ y = 8 ∧ z = 8 ∨
     x = 3 ∧ y = 12 ∧ z = 12 ∨
     x = 3 ∧ y = 7 ∧ z = 42 ∨
     x = 3 ∧ y = 8 ∧ z = 24 ∨
     x = 3 ∧ y = 9 ∧ z = 18 ∨
     x = 3 ∧ y = 10 ∧ z = 15 ∨
     x = 4 ∧ y = 5 ∧ z = 20 ∨
     x = 4 ∧ y = 6 ∧ z = 12) :=
by
  sorry

end find_parallelepiped_dimensions_l253_253241


namespace expected_ones_in_three_dice_rolls_l253_253946

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l253_253946


namespace sum_of_integers_75_to_95_l253_253491

theorem sum_of_integers_75_to_95 : (∑ i in Finset.range (95 - 75 + 1), (i + 75)) = 1785 := by
  sorry

end sum_of_integers_75_to_95_l253_253491


namespace smallest_n_for_g4_l253_253292

def g (n : ℕ) : ℕ :=
  ((sum (λ a, if a > 0 ∧ (∃ b > 0, a^2 + b^2 = n) then 1 else 0)) -
  ((sum (λ ⟨a, b⟩, if a > 0 ∧ b > 0 ∧ a^2 + b^2 = n then 1 else 0)) / 2)) + 1

theorem smallest_n_for_g4 :
  (∃ n : ℕ, n > 0 ∧ g(n) = 4 ∧ ∀ m : ℕ, m > 0 ∧ m < n → g(m) ≠ 4) ∧
  (∀ n : ℕ, n = 65 → g(n) = 4) :=
begin
  sorry
end

end smallest_n_for_g4_l253_253292


namespace least_area_in_rectangle_l253_253211

theorem least_area_in_rectangle
  (x y : ℤ)
  (h1 : 2 * (x + y) = 150)
  (h2 : x > 0)
  (h3 : y > 0) :
  ∃ x y : ℤ, (2 * (x + y) = 150) ∧ (x * y = 74) := by
  sorry

end least_area_in_rectangle_l253_253211


namespace debra_probability_theorem_l253_253381

-- Define event for Debra's coin flipping game starting with "HTT"
def debra_coin_game_event : Prop := 
  let heads_probability : ℝ := 0.5
  let tails_probability : ℝ := 0.5
  let initial_prob : ℝ := heads_probability * tails_probability * tails_probability
  let Q : ℝ := 1 / 3  -- the computed probability of getting HH after HTT
  let final_probability : ℝ := initial_prob * Q
  final_probability = 1 / 24

-- The theorem statement
theorem debra_probability_theorem :
  debra_coin_game_event := 
by
  sorry

end debra_probability_theorem_l253_253381


namespace find_f_5_l253_253578

theorem find_f_5 : 
  ∀ (f : ℝ → ℝ) (y : ℝ), 
  (∀ x, f x = 2 * x ^ 2 + y) ∧ f 2 = 60 -> f 5 = 102 :=
by
  sorry

end find_f_5_l253_253578


namespace sum_series_eq_3_div_4_l253_253833

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253833


namespace sum_series_eq_3_over_4_l253_253745

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253745


namespace base_five_product_l253_253371

theorem base_five_product (n1 n2 : ℕ) (h1 : n1 = 1 * 5^2 + 3 * 5^1 + 1 * 5^0) 
                          (h2 : n2 = 1 * 5^1 + 2 * 5^0) :
  let product_dec := (n1 * n2 : ℕ)
  let product_base5 := 2 * 125 + 1 * 25 + 2 * 5 + 2 * 1
  product_dec = 287 ∧ product_base5 = 2122 := by
                                -- calculations to verify statement omitted
                                sorry

end base_five_product_l253_253371


namespace conditional_probability_second_sci_given_first_sci_l253_253281

-- Definitions based on the conditions
def total_questions : ℕ := 6
def science_questions : ℕ := 4
def humanities_questions : ℕ := 2
def first_draw_is_science : Prop := true

-- The statement we want to prove
theorem conditional_probability_second_sci_given_first_sci : 
    first_draw_is_science → (science_questions - 1) / (total_questions - 1) = 3 / 5 := 
by
  intro h
  have num_sci_after_first : ℕ := science_questions - 1
  have total_after_first : ℕ := total_questions - 1
  have prob_second_sci := num_sci_after_first / total_after_first
  sorry

end conditional_probability_second_sci_given_first_sci_l253_253281


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253856

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253856


namespace B_holds_32_l253_253138

variable (x y z : ℝ)

-- Conditions
def condition1 : Prop := x + 1/2 * (y + z) = 90
def condition2 : Prop := y + 1/2 * (x + z) = 70
def condition3 : Prop := z + 1/2 * (x + y) = 56

-- Theorem to prove
theorem B_holds_32 (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : y = 32 :=
sorry

end B_holds_32_l253_253138


namespace hillary_stops_short_of_summit_l253_253119

noncomputable def distance_to_summit_from_base_camp : ℝ := 4700
noncomputable def hillary_climb_rate : ℝ := 800
noncomputable def eddy_climb_rate : ℝ := 500
noncomputable def hillary_descent_rate : ℝ := 1000
noncomputable def time_of_departure : ℝ := 6
noncomputable def time_of_passing : ℝ := 12

theorem hillary_stops_short_of_summit :
  ∃ x : ℝ, 
    (time_of_passing - time_of_departure) * hillary_climb_rate = distance_to_summit_from_base_camp - x →
    (time_of_passing - time_of_departure) * eddy_climb_rate = x →
    x = 2900 :=
by
  sorry

end hillary_stops_short_of_summit_l253_253119


namespace find_number_l253_253352

theorem find_number (x : ℝ) : (0.75 * x = 0.45 * 1500 + 495) -> x = 1560 :=
by
  sorry

end find_number_l253_253352


namespace min_value_ax_over_rR_l253_253921

theorem min_value_ax_over_rR (a b c r R : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_le_b : a ≤ b) (h_le_c : a ≤ c) (h_inradius : ∀ (a b c : ℝ), r = 2 * area / (a + b + c))
  (h_circumradius : ∀ (a b c : ℝ), R = (a * b * c) / (4 * area))
  (x : ℝ) (h_x : x = (b + c - a) / 2) (area : ℝ) :
  (a * x / (r * R)) ≥ 3 :=
sorry

end min_value_ax_over_rR_l253_253921


namespace sum_series_eq_3_div_4_l253_253826

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253826


namespace katy_summer_reading_total_l253_253143

def katy_books_in_summer (june_books july_books august_books : ℕ) : ℕ := june_books + july_books + august_books

theorem katy_summer_reading_total (june_books : ℕ) (july_books : ℕ) (august_books : ℕ) 
  (h1 : june_books = 8)
  (h2 : july_books = 2 * june_books)
  (h3 : august_books = july_books - 3) :
  katy_books_in_summer june_books july_books august_books = 37 :=
by
  sorry

end katy_summer_reading_total_l253_253143


namespace Lily_gallons_left_l253_253155

theorem Lily_gallons_left (initial_gallons : ℚ) (given_gallons : ℚ) (remaining_gallons : ℚ) 
  (h_initial : initial_gallons = 5) (h_given : given_gallons = 18 / 7) : 
  initial_gallons - given_gallons = remaining_gallons := 
begin
  have h_fraction : initial_gallons = 35 / 7, 
  { rw h_initial,
    norm_num, },
  rw [h_fraction, h_given],
  norm_num,
end

end Lily_gallons_left_l253_253155


namespace percentage_increase_of_numerator_l253_253475

theorem percentage_increase_of_numerator (N D : ℝ) (P : ℝ) (h1 : N / D = 0.75)
  (h2 : (N + (P / 100) * N) / (D - (8 / 100) * D) = 15 / 16) :
  P = 15 :=
sorry

end percentage_increase_of_numerator_l253_253475


namespace infinite_series_sum_value_l253_253808

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253808


namespace buckets_required_l253_253036

theorem buckets_required (C : ℚ) (N : ℕ) (h : 250 * (4/5 : ℚ) * C = N * C) : N = 200 :=
by
  sorry

end buckets_required_l253_253036


namespace tan_135_eq_neg1_l253_253542

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 :=
by
  have h1 : 135 * Real.pi / 180 = Real.pi - Real.pi / 4 := by norm_num
  rw [h1, Real.tan_sub_pi_div_two]
  norm_num
  sorry

end tan_135_eq_neg1_l253_253542


namespace unpainted_cube_count_is_correct_l253_253053

def unit_cube_count : ℕ := 6 * 6 * 6
def opposite_faces_painted_squares : ℕ := 16 * 2
def remaining_faces_painted_squares : ℕ := 9 * 4
def total_painted_squares (overlap_count : ℕ) : ℕ :=
  opposite_faces_painted_squares + remaining_faces_painted_squares - overlap_count
def overlap_count : ℕ := 4 * 2
def painted_cubes : ℕ := total_painted_squares overlap_count
def unpainted_cubes : ℕ := unit_cube_count - painted_cubes

theorem unpainted_cube_count_is_correct : unpainted_cubes = 156 := by
  sorry

end unpainted_cube_count_is_correct_l253_253053


namespace georg_can_identify_fake_coins_l253_253232

theorem georg_can_identify_fake_coins :
  ∀ (coins : ℕ) (baron : ℕ → ℕ → ℕ) (queries : ℕ),
    coins = 100 →
    ∃ (fake_count : ℕ → ℕ) (exaggeration : ℕ),
      (∀ group_size : ℕ, 10 ≤ group_size ∧ group_size ≤ 20) →
      (∀ (show_coins : ℕ), show_coins ≤ group_size → fake_count show_coins = baron show_coins exaggeration) →
      queries < 120 :=
by
  sorry

end georg_can_identify_fake_coins_l253_253232


namespace solution_set_of_inequality_l253_253101

theorem solution_set_of_inequality (x : ℝ) (n : ℕ) (h1 : n ≤ x ∧ x < n + 1 ∧ 0 < n) :
  4 * (⌊x⌋ : ℝ)^2 - 36 * (⌊x⌋ : ℝ) + 45 < 0 ↔ ∃ k : ℕ, (2 ≤ k ∧ k < 8 ∧ ⌊x⌋ = k) :=
by sorry

end solution_set_of_inequality_l253_253101


namespace tom_has_hours_to_spare_l253_253183

-- Conditions as definitions
def numberOfWalls : Nat := 5
def wallWidth : Nat := 2 -- in meters
def wallHeight : Nat := 3 -- in meters
def paintingRate : Nat := 10 -- in minutes per square meter
def totalAvailableTime : Nat := 10 -- in hours

-- Lean 4 statement of the problem
theorem tom_has_hours_to_spare :
  let areaOfOneWall := wallWidth * wallHeight -- 2 * 3
  let totalArea := numberOfWalls * areaOfOneWall -- 5 * (2 * 3)
  let totalTimeToPaint := (totalArea * paintingRate) / 60 -- (30 * 10) / 60
  totalAvailableTime - totalTimeToPaint = 5 :=
by
  sorry

end tom_has_hours_to_spare_l253_253183


namespace evaluate_series_sum_l253_253773

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253773


namespace students_study_all_three_l253_253366

open Finset

variables (U : Finset ℕ) (M L C : Finset ℕ)
variables (students_total : U.card = 425)
variables (M_card : M.card = 351) (L_card : L.card = 71) (C_card : C.card = 203)
variables (more_than_one_subject : (M ∩ L).card + (M ∩ C).card + (L ∩ C).card - 2 * (M ∩ L ∩ C).card = 199)
variables (no_subject : U.card - (M ∪ L ∪ C).card = 8)

theorem students_study_all_three
    (U M L C : Finset ℕ)
    (students_total : U.card = 425)
    (M_card : M.card = 351)
    (L_card : L.card = 71)
    (C_card : C.card = 203)
    (more_than_one_subject : (M ∩ L).card + (M ∩ C).card + (L ∩ C).card - 2 * (M ∩ L ∩ C).card = 199)
    (no_subject : U.card - (M ∪ L ∪ C).card = 8)
    : (M ∩ L ∩ C).card = 9 :=
begin
  sorry
end

end students_study_all_three_l253_253366


namespace verify_Fermat_point_l253_253589

open Real

theorem verify_Fermat_point :
  let D := (0, 0)
  let E := (6, 4)
  let F := (3, -2)
  let Q := (2, 1)
  let distance (P₁ P₂ : ℝ × ℝ) : ℝ := sqrt ((P₂.1 - P₁.1)^2 + (P₂.2 - P₁.2)^2)
  distance D Q + distance E Q + distance F Q = 5 + sqrt 5 + sqrt 10 := by
sorry

end verify_Fermat_point_l253_253589


namespace doughnut_problem_l253_253008

theorem doughnut_problem :
  ∀ (total_doughnuts first_two_box_doughnuts boxes : ℕ),
  total_doughnuts = 72 →
  first_two_box_doughnuts = 12 →
  boxes = 4 →
  (total_doughnuts - 2 * first_two_box_doughnuts) / boxes = 12 :=
by
  intros total_doughnuts first_two_box_doughnuts boxes ht12 hb12 b4
  sorry

end doughnut_problem_l253_253008


namespace total_cost_div_selling_price_eq_23_div_13_l253_253925

-- Conditions from part (a)
def pencil_count := 140
def pen_count := 90
def eraser_count := 60

def loss_pencils := 70
def loss_pens := 30
def loss_erasers := 20

def pen_cost (P : ℝ) := P
def pencil_cost (P : ℝ) := 2 * P
def eraser_cost (P : ℝ) := 1.5 * P

def total_cost (P : ℝ) :=
  pencil_count * pencil_cost P +
  pen_count * pen_cost P +
  eraser_count * eraser_cost P

def loss (P : ℝ) :=
  loss_pencils * pencil_cost P +
  loss_pens * pen_cost P +
  loss_erasers * eraser_cost P

def selling_price (P : ℝ) :=
  total_cost P - loss P

-- Statement to be proved: the total cost is 23/13 times the selling price.
theorem total_cost_div_selling_price_eq_23_div_13 (P : ℝ) :
  total_cost P / selling_price P = 23 / 13 := by
  sorry

end total_cost_div_selling_price_eq_23_div_13_l253_253925


namespace sum_geometric_series_l253_253708

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253708


namespace series_sum_correct_l253_253791

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253791


namespace segment_problem_l253_253227

theorem segment_problem 
  (A C : ℝ) (B D : ℝ) (P Q : ℝ) (x y k : ℝ)
  (hA : A = 0) (hC : C = 0) 
  (hB : B = 6) (hD : D = 9)
  (hx : x = P - A) (hy : y = Q - C) 
  (hxk : x = 3 * k)
  (hxyk : x + y = 12 * k) :
  k = 2 :=
  sorry

end segment_problem_l253_253227


namespace sum_geometric_series_l253_253713

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253713


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253862

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253862


namespace distance_between_points_l253_253172

/-- Given points P1 and P2 in the plane, prove that the distance between 
P1 and P2 is 5 units. -/
theorem distance_between_points : 
  let P1 : ℝ × ℝ := (-1, 1)
  let P2 : ℝ × ℝ := (2, 5)
  dist P1 P2 = 5 :=
by 
  sorry

end distance_between_points_l253_253172


namespace scientific_notation_chip_gate_width_l253_253326

theorem scientific_notation_chip_gate_width :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end scientific_notation_chip_gate_width_l253_253326


namespace series_result_l253_253839

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253839


namespace palindrome_clock_count_l253_253582

-- Definitions based on conditions from the problem statement.
def is_valid_hour (h : ℕ) : Prop := h < 24
def is_valid_minute (m : ℕ) : Prop := m < 60
def is_palindrome (h m : ℕ) : Prop :=
  (h < 10 ∧ m / 10 = h ∧ m % 10 = h) ∨
  (h >= 10 ∧ (h / 10) = (m % 10) ∧ (h % 10) = (m / 10 % 10))

-- Main theorem statement
theorem palindrome_clock_count : 
  (∃ n : ℕ, n = 66 ∧ ∀ (h m : ℕ), is_valid_hour h → is_valid_minute m → is_palindrome h m) := 
sorry

end palindrome_clock_count_l253_253582


namespace slope_angle_of_perpendicular_line_l253_253116

theorem slope_angle_of_perpendicular_line (l : ℝ → ℝ) (h_perp : ∀ x y : ℝ, l x = y ↔ x - y - 1 = 0) : ∃ α : ℝ, α = 135 :=
by
  sorry

end slope_angle_of_perpendicular_line_l253_253116


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253859

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253859


namespace number_of_birds_is_20_l253_253305

-- Define the given conditions
def distance_jim_disney : ℕ := 50
def distance_disney_london : ℕ := 60
def total_travel_distance : ℕ := 2200

-- Define the number of birds
def num_birds (B : ℕ) : Prop :=
  (distance_jim_disney + distance_disney_london) * B = total_travel_distance

-- The theorem stating the number of birds
theorem number_of_birds_is_20 : num_birds 20 :=
by
  unfold num_birds
  sorry

end number_of_birds_is_20_l253_253305


namespace food_requirement_l253_253160

/-- Peter has six horses. Each horse eats 5 pounds of oats, three times a day, and 4 pounds of grain twice a day. -/
def totalFoodRequired (horses : ℕ) (days : ℕ) (oatsMeal : ℕ) (oatsMealsPerDay : ℕ) (grainMeal : ℕ) (grainMealsPerDay : ℕ) : ℕ :=
  let dailyOats := oatsMeal * oatsMealsPerDay
  let dailyGrain := grainMeal * grainMealsPerDay
  let dailyFood := dailyOats + dailyGrain
  let totalDailyFood := dailyFood * horses
  totalDailyFood * days

theorem food_requirement :
  totalFoodRequired 6 5 5 3 4 2 = 690 :=
by sorry

end food_requirement_l253_253160


namespace intersection_P_Q_l253_253603

-- Defining the two sets P and Q
def P := { x : ℤ | abs x ≤ 2 }
def Q := { x : ℝ | -1 < x ∧ x < 5/2 }

-- Statement to prove
theorem intersection_P_Q : 
  { x : ℤ | abs x ≤ 2 } ∩ { x : ℤ | -1 < ((x : ℝ)) ∧ ((x : ℝ)) < 5/2 } = {0, 1, 2} := sorry

end intersection_P_Q_l253_253603


namespace molly_total_cost_l253_253437

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_children_per_brother : ℕ := 2
def num_spouse_per_brother : ℕ := 1

def total_num_relatives : ℕ := 
  let parents_and_siblings := num_parents + num_brothers
  let additional_relatives := num_brothers * (1 + num_spouse_per_brother + num_children_per_brother)
  parents_and_siblings + additional_relatives

def total_cost : ℕ :=
  total_num_relatives * cost_per_package

theorem molly_total_cost : total_cost = 85 := sorry

end molly_total_cost_l253_253437


namespace teachers_photos_l253_253200

theorem teachers_photos (n : ℕ) (ht : n = 5) : 6 * 7 = 42 :=
by
  sorry

end teachers_photos_l253_253200


namespace circle_center_sum_l253_253690

theorem circle_center_sum (x y : ℝ) :
  (x^2 + y^2 = 10*x - 12*y + 40) →
  x + y = -1 :=
by {
  sorry
}

end circle_center_sum_l253_253690


namespace sum_series_eq_3_over_4_l253_253749

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253749


namespace express_y_in_terms_of_x_l253_253314

theorem express_y_in_terms_of_x (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 := 
by { sorry }

end express_y_in_terms_of_x_l253_253314


namespace zeroes_y_minus_a_l253_253267

open Real

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then |2 ^ x - 1| else 3 / (x - 1)

theorem zeroes_y_minus_a (a : ℝ) : (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) → (0 < a ∧ a < 1) :=
sorry

end zeroes_y_minus_a_l253_253267


namespace series_result_l253_253843

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253843


namespace pizza_area_increase_l253_253471

theorem pizza_area_increase (A1 A2 r1 r2 : ℝ) (r1_eq : r1 = 7) (r2_eq : r2 = 5) (A1_eq : A1 = Real.pi * r1^2) (A2_eq : A2 = Real.pi * r2^2) :
  ((A1 - A2) / A2) * 100 = 96 := by
  sorry

end pizza_area_increase_l253_253471


namespace one_fourth_to_fourth_power_is_decimal_l253_253193

def one_fourth : ℚ := 1 / 4

theorem one_fourth_to_fourth_power_is_decimal :
  (one_fourth ^ 4 : ℚ) = 0.00390625 := 
by sorry

end one_fourth_to_fourth_power_is_decimal_l253_253193


namespace percentage_repeated_digits_five_digit_numbers_l253_253272

theorem percentage_repeated_digits_five_digit_numbers : 
  let total_five_digit_numbers := 90000
  let non_repeated_digits_number := 9 * 9 * 8 * 7 * 6
  let repeated_digits_number := total_five_digit_numbers - non_repeated_digits_number
  let y := (repeated_digits_number.toFloat / total_five_digit_numbers.toFloat) * 100 
  y = 69.8 :=
by
  sorry

end percentage_repeated_digits_five_digit_numbers_l253_253272


namespace bn_is_arithmetic_seq_an_general_term_l253_253415

def seq_an (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ ∀ n, (a (n + 1) - 1) * (a n - 1) = 3 * (a n - a (n + 1))

def seq_bn (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n, b n = 1 / (a n - 1)

theorem bn_is_arithmetic_seq (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, b (n + 1) - b n = 1 / 3 :=
sorry

theorem an_general_term (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, a n = (n + 5) / (n + 2) :=
sorry

end bn_is_arithmetic_seq_an_general_term_l253_253415


namespace part1_part2_l253_253399

variables (a b : ℝ) (f g : ℝ → ℝ)

-- Step 1: Given a > 0, b > 0 and f(x) = |x - a| - |x + b|, prove that if max(f) = 3, then a + b = 3.
theorem part1 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3) :
  a + b = 3 :=
sorry

-- Step 2: For g(x) = -x^2 - ax - b, if g(x) < f(x) for all x ≥ a, prove that 1/2 < a < 3.
theorem part2 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3)
    (hg : ∀ x, g x = -x^2 - a * x - b) (hcond : ∀ x, x ≥ a → g x < f x) :
    1 / 2 < a ∧ a < 3 :=
sorry

end part1_part2_l253_253399


namespace visitors_current_day_l253_253218

-- Define the number of visitors on the previous day and the additional visitors
def v_prev : ℕ := 600
def v_add : ℕ := 61

-- Prove that the number of visitors on the current day is 661
theorem visitors_current_day : v_prev + v_add = 661 :=
by
  sorry

end visitors_current_day_l253_253218


namespace binomial_sum_sum_of_binomial_solutions_l253_253097

theorem binomial_sum (k : ℕ) (h1 : Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (h2 : k = 6 ∨ k = 20) :
  k = 6 ∨ k = 20 → k = 6 ∨ k = 20 :=
by
  sorry

theorem sum_of_binomial_solutions :
  ∑ k in {6, 20}, k = 26 :=
by
  sorry

end binomial_sum_sum_of_binomial_solutions_l253_253097


namespace sum_of_series_eq_three_fourths_l253_253775

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253775


namespace taxi_fare_distance_l253_253420

theorem taxi_fare_distance (initial_fare : ℝ) (subsequent_fare : ℝ) (initial_distance : ℝ) (total_fare : ℝ) : 
  initial_fare = 2.0 →
  subsequent_fare = 0.60 →
  initial_distance = 1 / 5 →
  total_fare = 25.4 →
  ∃ d : ℝ, d = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end taxi_fare_distance_l253_253420


namespace weekly_earnings_l253_253390

theorem weekly_earnings (total_earnings : ℕ) (weeks : ℕ) (h1 : total_earnings = 133) (h2 : weeks = 19) : 
  round (total_earnings / weeks : ℝ) = 7 := 
by 
  sorry

end weekly_earnings_l253_253390


namespace LaKeisha_needs_to_mow_more_sqft_l253_253600

noncomputable def LaKeisha_price_per_sqft : ℝ := 0.10
noncomputable def LaKeisha_book_cost : ℝ := 150
noncomputable def LaKeisha_mowed_sqft : ℕ := 3 * 20 * 15
noncomputable def LaKeisha_earnings_so_far : ℝ := LaKeisha_mowed_sqft * LaKeisha_price_per_sqft

theorem LaKeisha_needs_to_mow_more_sqft (additional_sqft_needed : ℝ) :
  additional_sqft_needed = (LaKeisha_book_cost - LaKeisha_earnings_so_far) / LaKeisha_price_per_sqft → 
  additional_sqft_needed = 600 :=
by
  sorry

end LaKeisha_needs_to_mow_more_sqft_l253_253600


namespace proof_problem_l253_253419

def otimes (a b : ℕ) : ℕ := (a^2 - b) / (a - b)

theorem proof_problem : otimes (otimes 7 5) 2 = 24 := by
  sorry

end proof_problem_l253_253419


namespace sum_series_eq_3_div_4_l253_253834

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253834


namespace four_digit_sum_l253_253606

theorem four_digit_sum (A B : ℕ) (hA : 1000 ≤ A ∧ A < 10000) (hB : 1000 ≤ B ∧ B < 10000) (h : A * B = 16^5 + 2^10) : A + B = 2049 := 
by sorry

end four_digit_sum_l253_253606


namespace barbie_earrings_l253_253369

theorem barbie_earrings (total_earrings_alissa : ℕ) (alissa_triple_given : ℕ → ℕ) 
  (given_earrings_double_bought : ℕ → ℕ) (pairs_of_earrings : ℕ) : 
  total_earrings_alissa = 36 → 
  alissa_triple_given (total_earrings_alissa / 3) = total_earrings_alissa → 
  given_earrings_double_bought (total_earrings_alissa / 3) = total_earrings_alissa →
  pairs_of_earrings = 12 :=
by
  intros h1 h2 h3
  sorry

end barbie_earrings_l253_253369


namespace crayons_and_erasers_difference_l253_253300

theorem crayons_and_erasers_difference 
  (initial_crayons : ℕ) (initial_erasers : ℕ) (remaining_crayons : ℕ) 
  (h1 : initial_crayons = 601) (h2 : initial_erasers = 406) (h3 : remaining_crayons = 336) : 
  initial_erasers - remaining_crayons = 70 :=
by
  sorry

end crayons_and_erasers_difference_l253_253300


namespace tan_x_eq_2_solution_set_l253_253177

theorem tan_x_eq_2_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2} = {x : ℝ | Real.tan x = 2} :=
sorry

end tan_x_eq_2_solution_set_l253_253177


namespace joan_missed_games_l253_253593

variable (total_games : ℕ) (night_games : ℕ) (attended_games : ℕ)

theorem joan_missed_games (h1 : total_games = 864) (h2 : night_games = 128) (h3 : attended_games = 395) : 
  total_games - attended_games = 469 :=
  by
    sorry

end joan_missed_games_l253_253593


namespace smallest_n_for_g4_l253_253294

def g (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc a => acc + (List.range (n + 1)).countP (λ b => a * a + b * b = n)) 0

theorem smallest_n_for_g4 : ∃ n : ℕ, g n = 4 ∧ 
  (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 65
  -- Proof would go here
  sorry

end smallest_n_for_g4_l253_253294


namespace more_sqft_to_mow_l253_253599

-- Defining the parameters given in the original problem
def rate_per_sqft : ℝ := 0.10
def book_cost : ℝ := 150.0
def lawn_dimensions : ℝ × ℝ := (20, 15)
def num_lawns_mowed : ℕ := 3

-- The theorem stating how many more square feet LaKeisha needs to mow
theorem more_sqft_to_mow : 
  let area_one_lawn := (lawn_dimensions.1 * lawn_dimensions.2 : ℝ)
  let total_area_mowed := area_one_lawn * (num_lawns_mowed : ℝ)
  let money_earned := total_area_mowed * rate_per_sqft
  let remaining_amount := book_cost - money_earned
  let more_sqft_needed := remaining_amount / rate_per_sqft
  more_sqft_needed = 600 := 
by 
  sorry

end more_sqft_to_mow_l253_253599


namespace inequality_transpose_l253_253878

variable (a b : ℝ)

theorem inequality_transpose (h : a < b) (hab : b < 0) : (1 / a) > (1 / b) := by
  sorry

end inequality_transpose_l253_253878


namespace shaded_region_area_proof_l253_253380

/-- Define the geometric properties of the problem -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

noncomputable def shaded_region_area (rect : Rectangle) (circle1 circle2 : Circle) : ℝ :=
  let rect_area := rect.width * rect.height
  let circle_area := (Real.pi * circle1.radius ^ 2) + (Real.pi * circle2.radius ^ 2)
  rect_area - circle_area

theorem shaded_region_area_proof : shaded_region_area 
  {width := 10, height := 12} 
  {radius := 3, center := (0, 0)} 
  {radius := 3, center := (12, 10)} = 120 - 18 * Real.pi :=
by
  sorry

end shaded_region_area_proof_l253_253380


namespace number_of_trees_l253_253678

theorem number_of_trees (l d : ℕ) (h_l : l = 441) (h_d : d = 21) : (l / d) + 1 = 22 :=
by
  sorry

end number_of_trees_l253_253678


namespace minimum_distance_between_tracks_l253_253166

-- Problem statement as Lean definitions and theorem to prove
noncomputable def rational_man_track (t : ℝ) : ℝ × ℝ :=
  (Real.cos t, Real.sin t)

noncomputable def hyperbolic_man_track (t : ℝ) : ℝ × ℝ :=
  (-1 + 3 * Real.cos (t / 2), 5 * Real.sin (t / 2))

noncomputable def circle_eq := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

noncomputable def ellipse_eq := {p : ℝ × ℝ | (p.1 + 1)^2 / 9 + p.2^2 / 25 = 1}

theorem minimum_distance_between_tracks : 
  ∃ A ∈ circle_eq, ∃ B ∈ ellipse_eq, dist A B = Real.sqrt 14 - 1 := 
sorry

end minimum_distance_between_tracks_l253_253166


namespace two_real_solutions_only_if_c_zero_l253_253580

theorem two_real_solutions_only_if_c_zero (x y c : ℝ) :
  (|x + y| = 99 ∧ |x - y| = c → (∃! (x y : ℝ), |x + y| = 99 ∧ |x - y| = c)) ↔ c = 0 :=
by
  sorry

end two_real_solutions_only_if_c_zero_l253_253580


namespace clock_palindromes_l253_253584

theorem clock_palindromes : 
  let valid_hours := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22]
  let valid_minutes := [0, 1, 2, 3, 4, 5]
  let two_digit_palindromes := 9 * 6
  let four_digit_palindromes := 6
  (two_digit_palindromes + four_digit_palindromes) = 60 := 
by
  sorry

end clock_palindromes_l253_253584


namespace peanuts_weight_l253_253597

theorem peanuts_weight (total_snacks raisins : ℝ) (h_total : total_snacks = 0.5) (h_raisins : raisins = 0.4) : (total_snacks - raisins) = 0.1 :=
by
  rw [h_total, h_raisins]
  norm_num

end peanuts_weight_l253_253597


namespace correct_operation_l253_253341

theorem correct_operation (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 :=
sorry

end correct_operation_l253_253341


namespace infinite_series_sum_eq_l253_253715

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253715


namespace smallest_determinant_and_min_ab_l253_253397

def determinant (a b : ℤ) : ℤ :=
  36 * b - 81 * a

theorem smallest_determinant_and_min_ab :
  (∃ (a b : ℤ), 0 < determinant a b ∧ determinant a b = 9 ∧ ∀ a' b', determinant a' b' = 9 → a' + b' ≥ a + b) ∧
  (∃ (a b : ℤ), a = 3 ∧ b = 7) :=
sorry

end smallest_determinant_and_min_ab_l253_253397


namespace evaluate_series_sum_l253_253768

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253768


namespace combination_sum_l253_253523

noncomputable def combination (n r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem combination_sum :
  combination 3 2 + combination 4 2 + combination 5 2 + combination 6 2 + 
  combination 7 2 + combination 8 2 + combination 9 2 + combination 10 2 = 164 :=
by
  sorry

end combination_sum_l253_253523


namespace average_visitors_on_Sundays_l253_253505

theorem average_visitors_on_Sundays (S : ℕ) (h1 : 30 = 5 + 25) (h2 : 25 * 240 + 5 * S = 30 * 285) :
  S = 510 := sorry

end average_visitors_on_Sundays_l253_253505


namespace expected_ones_three_standard_dice_l253_253956

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l253_253956


namespace evaluate_expression_l253_253372

theorem evaluate_expression :
  2 - (-3) - 4 + (-5) - 6 + 7 = -3 :=
by
  sorry

end evaluate_expression_l253_253372


namespace molly_christmas_shipping_cost_l253_253439

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_sisters_in_law_per_brother : ℕ := 1
def num_children_per_brother : ℕ := 2

def total_relatives : ℕ :=
  num_parents + num_brothers + (num_brothers * num_sisters_in_law_per_brother) + (num_brothers * num_children_per_brother)

theorem molly_christmas_shipping_cost : total_relatives * cost_per_package = 70 :=
by
  sorry

end molly_christmas_shipping_cost_l253_253439


namespace expected_number_of_ones_on_three_dice_l253_253953

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l253_253953


namespace infinite_series_sum_eq_l253_253723

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253723


namespace triangle_sine_sum_leq_l253_253139

theorem triangle_sine_sum_leq (A B C : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) (h₄ : A + B + C = π) :
  sin A + sin B + sin C ≤ (3 * sqrt 3) / 2 :=
begin
  sorry
end

end triangle_sine_sum_leq_l253_253139


namespace expected_ones_on_three_dice_l253_253966

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l253_253966


namespace estimate_students_l253_253906

open ProbTheory

noncomputable def number_of_students_above_110 {n : ℕ} (X : PMF ℝ) : ℕ :=
  let μ := 100
  let σ := 10
  let number_of_students := 50
  if X.pmf 90 |>.val = 0.3 then
    (number_of_students * (1 - (X.pmf (μ-σ) |>.val + X.pmf μ |>.val)))
  else
    0

theorem estimate_students :
  let X := PMF.uniform_of_finset (set.Icc 90 110) 50
  (X.pmf 90.0 |>.val, X.pmf 100.0 |>.val = (0.3, 0.3)) → 
  number_of_students_above_110 X = 10 :=
by
  sorry

end estimate_students_l253_253906


namespace john_has_hours_to_spare_l253_253185

def total_wall_area (num_walls : ℕ) (wall_width wall_height : ℕ) : ℕ :=
  num_walls * wall_width * wall_height

def time_to_paint_area (area : ℕ) (rate_per_square_meter_in_minutes : ℕ) : ℕ :=
  area * rate_per_square_meter_in_minutes

def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem john_has_hours_to_spare 
  (num_walls : ℕ) (wall_width wall_height : ℕ)
  (rate_per_square_meter_in_minutes : ℕ) (total_available_hours : ℕ)
  (to_spare_hours : ℕ)
  (h : total_wall_area num_walls wall_width wall_height = num_walls * wall_width * wall_height)
  (h1 : time_to_paint_area (num_walls * wall_width * wall_height) rate_per_square_meter_in_minutes = num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes)
  (h2 : minutes_to_hours (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes) = (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes) / 60)
  (h3 : total_available_hours = 10) 
  (h4 : to_spare_hours = total_available_hours - (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes / 60)) : 
  to_spare_hours = 5 := 
sorry

end john_has_hours_to_spare_l253_253185


namespace tan_135_eq_neg1_l253_253538

theorem tan_135_eq_neg1 :
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in
  Q = -(Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * Complex.I →
  Complex.tan (135 * Real.pi / 180 * Complex.I) = -1 :=
by
  intro hQ
  have Q_coords : Q = -(Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * Complex.I := hQ
  sorry

end tan_135_eq_neg1_l253_253538


namespace sum_k_binomial_l253_253093

theorem sum_k_binomial :
  (∃ k1 k2, k1 ≠ k2 ∧ nat.choose 26 k1 = nat.choose 25 5 + nat.choose 25 6 ∧
              nat.choose 26 k2 = nat.choose 25 5 + nat.choose 25 6 ∧ k1 + k2 = 26) :=
by
  use [6, 20]
  split
  { sorry } -- proof of k1 ≠ k2
  { split
    { simp [nat.choose] }
    { split
      { simp [nat.choose] }
      { simp }
    }
  }

end sum_k_binomial_l253_253093


namespace number_of_students_like_basketball_but_not_table_tennis_l253_253280

-- Given definitions
def total_students : Nat := 40
def students_like_basketball : Nat := 24
def students_like_table_tennis : Nat := 16
def students_dislike_both : Nat := 6

-- Proposition to prove
theorem number_of_students_like_basketball_but_not_table_tennis : 
  students_like_basketball - (students_like_basketball + students_like_table_tennis - (total_students - students_dislike_both)) = 18 := 
by
  sorry

end number_of_students_like_basketball_but_not_table_tennis_l253_253280


namespace not_possible_to_color_plane_l253_253435

theorem not_possible_to_color_plane :
  ¬ ∃ (color : ℕ → ℕ × ℕ → ℕ) (c : ℕ), 
    (c = 2016) ∧
    (∀ (A B C : ℕ × ℕ), (A ≠ B ∧ B ≠ C ∧ C ≠ A) → 
                        (color c A = color c B) ∨ (color c B = color c C) ∨ (color c C = color c A)) :=
by
  sorry

end not_possible_to_color_plane_l253_253435


namespace square_of_binomial_l253_253576

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (9:ℝ) * x^2 + 24 * x + a = (3 * x + b)^2) → a = 16 :=
by
  sorry

end square_of_binomial_l253_253576


namespace part1_part2_l253_253393

open Set

def f (x : ℝ) : ℝ := abs (x + 2) - abs (2 * x - 1)

def M : Set ℝ := { x | f x > 0 }

theorem part1 :
  M = { x | - (1 / 3 : ℝ) < x ∧ x < 3 } :=
sorry

theorem part2 :
  ∀ (x y : ℝ), x ∈ M → y ∈ M → abs (x + y + x * y) < 15 :=
sorry

end part1_part2_l253_253393


namespace hermione_max_profit_l253_253285

def TC (Q : ℝ) : ℝ := 5 * Q^2

def demand_ws (P : ℝ) : ℝ := 26 - 2 * P
def demand_s (P : ℝ) : ℝ := 10 - P

noncomputable def max_profit : ℝ := 7.69

theorem hermione_max_profit :
  ∃ P Q, (P > 0 ∧ Q > 0) ∧ (Q = demand_ws P + demand_s P) ∧
  (P * Q - TC Q = max_profit) := sorry

end hermione_max_profit_l253_253285


namespace tan_135_eq_neg1_l253_253540

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 :=
by
  have h_cos : Real.cos (135 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180) := 
    by 
      apply Real.cos_angle_of_pi_sub_angle; 
      sorry
  have h_cos_45 : Real.cos (45 * Real.pi / 180) = (1 / Real.sqrt 2) := 
    by 
      rw Real.cos_pi_div_four;
      sorry
  have h_sin : Real.sin (135 * Real.pi / 180) = Real.sin (45 * Real.pi / 180) := 
    by
      apply Real.sin_of_pi_sub_angle;
      sorry
  have h_sin_45 : Real.sin (45 * Real.pi / 180) = (1 / Real.sqrt 2) := 
    by 
      rw Real.sin_pi_div_four;
      sorry
  rw [← h_sin, h_sin_45, ← h_cos, h_cos_45]
  rw [Real.tan_eq_sin_div_cos, div_eq_mul_inv, mul_comm, inv_mul_cancel]
  norm_num
  sorry

end tan_135_eq_neg1_l253_253540


namespace factorization_sum_l253_253315

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x ^ 2 + 9 * x + 18 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x ^ 2 + 19 * x + 90 = (x + b) * (x + c)) :
  a + b + c = 22 := by
sorry

end factorization_sum_l253_253315


namespace minimum_reciprocal_sum_of_roots_l253_253409

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := 2 * x^2 + b * x + c

theorem minimum_reciprocal_sum_of_roots {b c : ℝ} {x1 x2 : ℝ} 
  (h1: f (-10) b c = f 12 b c)
  (h2: f x1 b c = 0)
  (h3: f x2 b c = 0)
  (h4: 0 < x1)
  (h5: 0 < x2)
  (h6: x1 + x2 = 2) :
  (1 / x1 + 1 / x2) = 2 :=
sorry

end minimum_reciprocal_sum_of_roots_l253_253409


namespace probability_at_least_one_blue_l253_253433

-- Definitions of the setup
def red_balls := 2
def blue_balls := 2
def total_balls := red_balls + blue_balls
def total_outcomes := (total_balls * (total_balls - 1)) / 2  -- choose 2 out of total
def favorable_outcomes := 10  -- by counting outcomes with at least one blue ball

-- Definition of the proof problem
theorem probability_at_least_one_blue (a b : ℕ) (h1: a = red_balls) (h2: b = blue_balls) :
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 5 / 6 := by
  sorry  

end probability_at_least_one_blue_l253_253433


namespace least_four_digit_divisible_1_2_4_8_l253_253332

theorem least_four_digit_divisible_1_2_4_8 : ∃ n : ℕ, ∀ d1 d2 d3 d4 : ℕ, 
  n = d1*1000 + d2*100 + d3*10 + d4 ∧
  1000 ≤ n ∧ n < 10000 ∧
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d4 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d4 ∧
  n % 1 = 0 ∧
  n % 2 = 0 ∧
  n % 4 = 0 ∧
  n % 8 = 0 ∧
  n = 1248 :=
by
  sorry

end least_four_digit_divisible_1_2_4_8_l253_253332


namespace largest_common_value_under_800_l253_253310

-- Let's define the problem conditions as arithmetic sequences
def sequence1 (a : ℤ) : Prop := ∃ n : ℤ, a = 4 + 5 * n
def sequence2 (a : ℤ) : Prop := ∃ m : ℤ, a = 7 + 8 * m

-- Now we state the theorem that the largest common value less than 800 is 799
theorem largest_common_value_under_800 : 
  ∃ a : ℤ, sequence1 a ∧ sequence2 a ∧ a < 800 ∧ ∀ b : ℤ, sequence1 b ∧ sequence2 b ∧ b < 800 → b ≤ a :=
sorry

end largest_common_value_under_800_l253_253310


namespace infinite_series_sum_value_l253_253809

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253809


namespace sum_powers_of_i_l253_253378

variable (n : ℕ) (i : ℂ) (h_multiple_of_6 : n % 6 = 0) (h_i : i^2 = -1)

theorem sum_powers_of_i (h_n6 : n = 6) :
    1 + 2*i + 3*i^2 + 4*i^3 + 5*i^4 + 6*i^5 + 7*i^6 = 6*i - 7 := by
  sorry

end sum_powers_of_i_l253_253378


namespace solution_set_l253_253052

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set (c1 : ∀ x : ℝ, f x + f' x > 1)
                     (c2 : f 0 = 2) :
  {x : ℝ | e^x * f x > e^x + 1} = {x : ℝ | 0 < x} :=
sorry

end solution_set_l253_253052


namespace part1_part2_part3_l253_253570

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)
def g (x : ℝ) : ℝ := f x - abs (x - 2)

theorem part1 : ∀ x : ℝ, f x ≤ 8 ↔ (-11 ≤ x ∧ x ≤ 5) := by sorry

theorem part2 : ∃ x : ℝ, g x = 5 := by sorry

theorem part3 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 5) : 
  1 / a + 9 / b = 16 / 5 := by sorry

end part1_part2_part3_l253_253570


namespace distance_in_scientific_notation_l253_253021

theorem distance_in_scientific_notation :
  ∃ a n : ℝ, 1 ≤ |a| ∧ |a| < 10 ∧ n = 4 ∧ 38000 = a * 10^n ∧ a = 3.8 :=
by
  sorry

end distance_in_scientific_notation_l253_253021


namespace series_sum_eq_l253_253727

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253727


namespace piles_stones_l253_253457

theorem piles_stones (a b c d : ℕ)
  (h₁ : a = 2011)
  (h₂ : b = 2010)
  (h₃ : c = 2009)
  (h₄ : d = 2008) :
  ∃ (k l m n : ℕ), (k, l, m, n) = (0, 0, 0, 2) ∧
  ((∃ x y z w : ℕ, k = x - y ∧ l = y - z ∧ m = z - w ∧ x + l + m + w = 0) ∨
   (∃ u : ℕ, k = a - u ∧ l = b - u ∧ m = c - u ∧ n = d - u)) :=
sorry

end piles_stones_l253_253457


namespace series_sum_correct_l253_253793

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253793


namespace line_does_not_pass_through_second_quadrant_l253_253061
-- Import the Mathlib library

-- Define the properties of the line
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the condition for a point to be in the second quadrant:
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the proof statement
theorem line_does_not_pass_through_second_quadrant:
  ∀ x y : ℝ, line_eq x y → ¬ in_second_quadrant x y :=
by
  sorry

end line_does_not_pass_through_second_quadrant_l253_253061


namespace mouse_jumps_28_inches_further_than_grasshopper_l253_253317

theorem mouse_jumps_28_inches_further_than_grasshopper :
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  g_to_m_difference = 28 :=
by
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  show g_to_m_difference = 28
  sorry

end mouse_jumps_28_inches_further_than_grasshopper_l253_253317


namespace center_of_circle_l253_253239

theorem center_of_circle :
  ∀ (x y : ℝ), (x^2 - 8 * x + y^2 - 4 * y = 16) → (x, y) = (4, 2) :=
by
  sorry

end center_of_circle_l253_253239


namespace minute_hand_angle_l253_253064

theorem minute_hand_angle (minutes_slow : ℕ) (total_minutes : ℕ) (full_rotation : ℝ) (h1 : minutes_slow = 5) (h2 : total_minutes = 60) (h3 : full_rotation = 2 * Real.pi) : 
  (minutes_slow / total_minutes : ℝ) * full_rotation = Real.pi / 6 :=
by
  sorry

end minute_hand_angle_l253_253064


namespace number_of_cartons_of_pencils_l253_253456

theorem number_of_cartons_of_pencils (P E : ℕ) 
  (h1 : P + E = 100) 
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end number_of_cartons_of_pencils_l253_253456


namespace axis_of_symmetry_parabola_l253_253870

theorem axis_of_symmetry_parabola (x y : ℝ) : 
  (∃ k : ℝ, (y^2 = -8 * k) → (y^2 = -8 * x) → x = -1) :=
by
  sorry

end axis_of_symmetry_parabola_l253_253870


namespace black_car_speed_l253_253485

theorem black_car_speed
  (red_speed black_speed : ℝ)
  (initial_distance time : ℝ)
  (red_speed_eq : red_speed = 10)
  (initial_distance_eq : initial_distance = 20)
  (time_eq : time = 0.5)
  (distance_eq : black_speed * time = initial_distance + red_speed * time) :
  black_speed = 50 := by
  rw [red_speed_eq, initial_distance_eq, time_eq] at distance_eq
  sorry

end black_car_speed_l253_253485


namespace cylinder_original_radius_l253_253231

theorem cylinder_original_radius 
  (r h : ℝ) 
  (hr_eq : h = 3)
  (volume_increase_radius : Real.pi * (r + 8)^2 * 3 = Real.pi * r^2 * 11) :
  r = 8 :=
by
  -- the proof steps will be here
  sorry

end cylinder_original_radius_l253_253231


namespace inequality_solution_l253_253085

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Ioo (-1/4) 0 ∪ Set.Ioo 3/2 2) ↔ 
  (1 ≤ (x - 2) * 4 ∧ x ≠ 2) ∧ (x > 0 ∨ x ≠ 0) := 
sorry

end inequality_solution_l253_253085


namespace sum_series_equals_three_fourths_l253_253816

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253816


namespace expected_number_of_ones_on_three_dice_l253_253952

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l253_253952


namespace total_distance_proof_l253_253066

-- Define the conditions
def amoli_speed : ℕ := 42      -- Amoli's speed in miles per hour
def amoli_time : ℕ := 3        -- Amoli's driving time in hours
def anayet_speed : ℕ := 61     -- Anayet's speed in miles per hour
def anayet_time : ℕ := 2       -- Anayet's driving time in hours
def remaining_distance : ℕ := 121  -- Remaining distance to be traveled in miles

-- Total distance calculation
def total_distance : ℕ :=
  amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance

-- The theorem to prove
theorem total_distance_proof : total_distance = 369 :=
by
  -- Proof goes here
  sorry

end total_distance_proof_l253_253066


namespace sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5_l253_253618

theorem sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5 : Real.sqrt 3 - Real.sqrt 2 > Real.sqrt 6 - Real.sqrt 5 :=
sorry

end sqrt3_sub_sqrt2_gt_sqrt6_sub_sqrt5_l253_253618


namespace tangent_line_equation_l253_253173

-- Definitions used as conditions in the problem
def curve (x : ℝ) : ℝ := 2 * x - x^3
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Lean 4 statement representing the proof problem
theorem tangent_line_equation :
  let x₀ := 1
  let y₀ := 1
  let m := deriv curve x₀
  m = -1 ∧ curve x₀ = y₀ →
  ∀ x y : ℝ, x + y - 2 = 0 → curve x₀ + m * (x - x₀) = y :=
by
  -- Proof would go here
  sorry

end tangent_line_equation_l253_253173


namespace difference_in_mileage_l253_253010

-- Define the conditions
def advertised_mpg : ℝ := 35
def tank_capacity : ℝ := 12
def regular_gasoline_mpg : ℝ := 30
def premium_gasoline_mpg : ℝ := 40
def diesel_mpg : ℝ := 32
def fuel_proportion : ℝ := 1 / 3

-- Define the weighted average function
def weighted_average_mpg (mpg1 mpg2 mpg3 : ℝ) (proportion : ℝ) : ℝ :=
  (mpg1 * proportion) + (mpg2 * proportion) + (mpg3 * proportion)

-- Proof
theorem difference_in_mileage :
  advertised_mpg - weighted_average_mpg regular_gasoline_mpg premium_gasoline_mpg diesel_mpg fuel_proportion = 1 := by
  sorry

end difference_in_mileage_l253_253010


namespace sum_series_div_3_powers_l253_253759

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253759


namespace express_y_l253_253081

theorem express_y (x y : ℝ) (h : 3 * x + 2 * y = 1) : y = (1 - 3 * x) / 2 :=
by {
  sorry
}

end express_y_l253_253081


namespace length_of_train_l253_253675

theorem length_of_train :
  ∀ (L : ℝ) (V : ℝ),
  (∀ t p : ℝ, t = 14 → p = 535.7142857142857 → V = L / t) →
  (∀ t p : ℝ, t = 39 → p = 535.7142857142857 → V = (L + p) / t) →
  L = 300 :=
by
  sorry

end length_of_train_l253_253675


namespace fishing_problem_l253_253483

theorem fishing_problem :
  ∃ F : ℕ, (F % 3 = 1 ∧
            ((F - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3) % 3 = 1) ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3 - 1) = 0) :=
sorry

end fishing_problem_l253_253483


namespace inequality_sqrt_sum_leq_one_plus_sqrt_l253_253174

theorem inequality_sqrt_sum_leq_one_plus_sqrt (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  Real.sqrt (a * (1 - b) * (1 - c)) + Real.sqrt (b * (1 - a) * (1 - c)) + Real.sqrt (c * (1 - a) * (1 - b)) 
  ≤ 1 + Real.sqrt (a * b * c) :=
sorry

end inequality_sqrt_sum_leq_one_plus_sqrt_l253_253174


namespace subset_strict_M_P_l253_253004

-- Define the set M
def M : Set ℕ := {x | ∃ a : ℕ, a > 0 ∧ x = a^2 + 1}

-- Define the set P
def P : Set ℕ := {y | ∃ b : ℕ, b > 0 ∧ y = b^2 - 4*b + 5}

-- Prove that M is strictly a subset of P
theorem subset_strict_M_P : M ⊆ P ∧ ∃ x ∈ P, x ∉ M :=
by
  sorry

end subset_strict_M_P_l253_253004


namespace total_pepper_weight_l253_253546

theorem total_pepper_weight :
  let green_peppers := 2.8333333333333335
  let red_peppers := 3.254
  let yellow_peppers := 1.375
  let orange_peppers := 0.567
  (green_peppers + red_peppers + yellow_peppers + orange_peppers) = 8.029333333333333 := 
by
  sorry

end total_pepper_weight_l253_253546


namespace count_repeating_decimals_l253_253253

theorem count_repeating_decimals (s : Set ℕ) :
  (∀ n, n ∈ s ↔ 1 ≤ n ∧ n ≤ 20 ∧ ¬∃ k, k * 3 = n) →
  (s.card = 14) :=
by 
  sorry

end count_repeating_decimals_l253_253253


namespace second_train_speed_l253_253192

theorem second_train_speed (d : ℝ) (s₁ : ℝ) (t₁ : ℝ) (t₂ : ℝ) (meet_time : ℝ) (total_distance : ℝ) :
  d = 110 ∧ s₁ = 20 ∧ t₁ = 3 ∧ t₂ = 2 ∧ meet_time = 10 ∧ total_distance = d →
  60 + 2 * (total_distance - 60) / 2 = 110 →
  (total_distance - 60) / 2 = 25 :=
by
  intro h1 h2
  sorry

end second_train_speed_l253_253192


namespace sum_of_letters_l253_253660

def A : ℕ := 0
def B : ℕ := 1
def C : ℕ := 2
def M : ℕ := 12

theorem sum_of_letters :
  A + B + M + C = 15 :=
by
  sorry

end sum_of_letters_l253_253660


namespace lisa_speed_l253_253299

-- Define conditions
def distance : ℕ := 256
def time : ℕ := 8

-- Define the speed calculation theorem
theorem lisa_speed : (distance / time) = 32 := 
by {
  sorry
}

end lisa_speed_l253_253299


namespace range_of_m_l253_253458

-- Definitions of Propositions p and q
def Proposition_p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m > 0) ∧ (1 > 0)  -- where x₁ + x₂ = -m > 0 and x₁x₂ = 1

def Proposition_q (m : ℝ) : Prop :=
  16 * (m + 2)^2 - 16 < 0  -- discriminant of 4x^2 + 4(m+2)x + 1 = 0 is less than 0

-- Given: "Proposition p or Proposition q" is true
def given (m : ℝ) : Prop :=
  Proposition_p m ∨ Proposition_q m

-- Prove: Range of values for m is (-∞, -1)
theorem range_of_m (m : ℝ) (h : given m) : m < -1 :=
sorry

end range_of_m_l253_253458


namespace max_b_minus_a_l253_253554

theorem max_b_minus_a (a b : ℝ) (h_a: a < 0) (h_ineq: ∀ x : ℝ, (3 * x^2 + a) * (2 * x + b) ≥ 0) : 
b - a = 1 / 3 := 
sorry

end max_b_minus_a_l253_253554


namespace arithmetic_seq_a11_l253_253108

variable (a : ℕ → ℤ)
variable (d : ℕ → ℤ)

-- Conditions
def arithmetic_sequence : Prop := ∀ n, a (n + 2) - a n = 6
def a1 : Prop := a 1 = 1

-- Statement of the problem
theorem arithmetic_seq_a11 : arithmetic_sequence a ∧ a1 a → a 11 = 31 :=
by sorry

end arithmetic_seq_a11_l253_253108


namespace find_side_length_of_triangle_l253_253572

noncomputable def triangle_side_length
  (a b : ℝ)
  (angle_C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hC : angle_C = real.pi / 3) : ℝ :=
  let c_squared := a^2 + b^2 - 2 * a * b * real.cos angle_C in
  real.sqrt c_squared

theorem find_side_length_of_triangle :
  ∀ (a b angle_C : ℝ), a = 2 ∧ b = 3 ∧ angle_C = real.pi / 3 →
  triangle_side_length a b angle_C 2 3 (real.pi / 3) = real.sqrt 7 :=
by
  intros a b angle_C h,
  unfold triangle_side_length,
  rw [h.1, h.2.1, h.2.2],
  sorry -- The actual proof would go here

end find_side_length_of_triangle_l253_253572


namespace dampening_factor_l253_253356

theorem dampening_factor (s r : ℝ) 
  (h1 : s / (1 - r) = 16) 
  (h2 : s * r / (1 - r^2) = -6) :
  r = -3 / 11 := 
sorry

end dampening_factor_l253_253356


namespace series_sum_correct_l253_253788

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253788


namespace sravan_distance_l253_253656

theorem sravan_distance {D : ℝ} :
  (D / 90 + D / 60 = 15) ↔ (D = 540) :=
by sorry

end sravan_distance_l253_253656


namespace x_coordinate_of_equidistant_point_l253_253975

theorem x_coordinate_of_equidistant_point (x : ℝ) : 
  ((-3 - x)^2 + (-2 - 0)^2) = ((2 - x)^2 + (-6 - 0)^2) → x = 2.7 :=
by
  sorry

end x_coordinate_of_equidistant_point_l253_253975


namespace total_earnings_l253_253375

def oil_change_cost : ℕ := 20
def repair_cost : ℕ := 30
def car_wash_cost : ℕ := 5

def num_oil_changes : ℕ := 5
def num_repairs : ℕ := 10
def num_car_washes : ℕ := 15

theorem total_earnings :
  (num_oil_changes * oil_change_cost) +
  (num_repairs * repair_cost) +
  (num_car_washes * car_wash_cost) = 475 :=
by
  sorry

end total_earnings_l253_253375


namespace solution_count_l253_253478

theorem solution_count (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) → 
  (∃ (num_solutions : ℕ), 
    (num_solutions = 3 ∧ a = 1 ∨ a = -1) ∨ 
    (num_solutions = 2 ∧ a = Real.sqrt 2 ∨ a = -Real.sqrt 2)) :=
by sorry

end solution_count_l253_253478


namespace grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l253_253070

theorem grid_spiral_infinite_divisible_by_68 (n : ℕ) :
  ∃ (k : ℕ), ∃ (m : ℕ), ∃ (t : ℕ), 
  let A := t + 0;
  let B := t + 4;
  let C := t + 12;
  let D := t + 8;
  (k = n * 68 ∧ (n ≥ 1)) ∧ 
  (m = A + B + C + D) ∧ (m % 68 = 0) := by
  sorry

theorem grid_spiral_unique_center_sums (n : ℕ) :
  ∀ (i j : ℕ), 
  let Si := n * 68 + i;
  let Sj := n * 68 + j;
  ¬ (Si = Sj) := by
  sorry

end grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l253_253070


namespace three_zeros_of_f_l253_253394

noncomputable def f (a x b : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + a + 2) * x + (2 * a + 2) * (Real.log x) + b

theorem three_zeros_of_f (a b : ℝ) (h1 : a > 3) (h2 : a^2 + a + 1 < b) (h3 : b < 2 * a^2 - 2 * a + 2) : 
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 b = 0 ∧ f a x2 b = 0 ∧ f a x3 b = 0 :=
by
  sorry

end three_zeros_of_f_l253_253394


namespace angle_addition_l253_253259

open Real

theorem angle_addition (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : tan α = 1 / 3) (h₄ : cos β = 3 / 5) : α + 3 * β = 3 * π / 4 :=
by
  sorry

end angle_addition_l253_253259


namespace unique_plants_count_l253_253034

open Finset

variable (A B C : Finset ℕ)

def card_A : ℕ := 600
def card_B : ℕ := 550
def card_C : ℕ := 400
def card_AB : ℕ := 60
def card_AC : ℕ := 110
def card_BC : ℕ := 90
def card_ABC : ℕ := 30

theorem unique_plants_count :  
  ∀ A B C : Finset ℕ,  
  A.card = card_A ∧ 
  B.card = card_B ∧ 
  C.card = card_C ∧ 
  (A ∩ B).card = card_AB ∧ 
  (A ∩ C).card = card_AC ∧ 
  (B ∩ C).card = card_BC ∧ 
  (A ∩ B ∩ C).card = card_ABC → 
  (A ∪ B ∪ C).card = 1320 := 
by sorry

end unique_plants_count_l253_253034


namespace min_people_wearing_both_hat_and_glove_l253_253135

theorem min_people_wearing_both_hat_and_glove (n : ℕ) (x : ℕ) 
  (h1 : 2 * n = 5 * (8 : ℕ)) -- 2/5 of n people wear gloves
  (h2 : 3 * n = 4 * (15 : ℕ)) -- 3/4 of n people wear hats
  (h3 : n = 20): -- total number of people is 20
  x = 3 := -- minimum number of people wearing both a hat and a glove is 3
by sorry

end min_people_wearing_both_hat_and_glove_l253_253135


namespace total_baseball_cards_l253_253445

theorem total_baseball_cards (Carlos Matias Jorge : ℕ) (h1 : Carlos = 20) (h2 : Matias = Carlos - 6) (h3 : Jorge = Matias) : Carlos + Matias + Jorge = 48 :=
by
  sorry

end total_baseball_cards_l253_253445


namespace infinite_series_sum_eq_l253_253721

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253721


namespace recreation_percentage_l253_253144

variable (W : ℝ) -- John's wages last week
variable (recreation_last_week : ℝ := 0.35 * W) -- Amount spent on recreation last week
variable (wages_this_week : ℝ := 0.70 * W) -- Wages this week
variable (recreation_this_week : ℝ := 0.25 * wages_this_week) -- Amount spent on recreation this week

theorem recreation_percentage :
  (recreation_this_week / recreation_last_week) * 100 = 50 := by
  sorry

end recreation_percentage_l253_253144


namespace infinite_series_sum_eq_3_div_4_l253_253798

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253798


namespace infinite_series_sum_eq_3_div_4_l253_253803

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253803


namespace no_three_digit_number_l253_253385

theorem no_three_digit_number :
  ¬ ∃ (a b c : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (100 * a + 10 * b + c = 3 * (100 * b + 10 * c + a)) :=
by
  sorry

end no_three_digit_number_l253_253385


namespace geom_seq_product_equals_16_l253_253282

theorem geom_seq_product_equals_16
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : ∀ m n, a (m + 1) - a m = a (n + 1) - a n)
  (non_zero_diff : ∃ d, d ≠ 0 ∧ ∀ n, a (n + 1) - a n = d)
  (h_cond : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  (h_geom : ∀ m n, b (m + 1) / b m = b (n + 1) / b n)
  (h_b7 : b 7 = a 7):
  b 6 * b 8 = 16 := 
sorry

end geom_seq_product_equals_16_l253_253282


namespace tan_135_eq_neg_one_l253_253533

theorem tan_135_eq_neg_one : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg_one_l253_253533


namespace sum_series_eq_3_over_4_l253_253751

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253751


namespace tan_a_values_l253_253877

theorem tan_a_values (a : ℝ) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1 / 2 :=
by
  sorry

end tan_a_values_l253_253877


namespace loan_amount_is_900_l253_253621

theorem loan_amount_is_900 (P R T SI : ℕ) (hR : R = 9) (hT : T = 9) (hSI : SI = 729)
    (h_simple_interest : SI = (P * R * T) / 100) : P = 900 := by
  sorry

end loan_amount_is_900_l253_253621


namespace proof_value_of_expression_l253_253271

theorem proof_value_of_expression (a b c d m : ℝ) 
  (h1: a + b = 0)
  (h2: c * d = 1)
  (h3: |m| = 4) : 
  m + c * d + (a + b) / m = 5 ∨ m + c * d + (a + b) / m = -3 := by
  sorry

end proof_value_of_expression_l253_253271


namespace total_games_attended_l253_253916

def games_in_months (this_month previous_month next_month following_month fifth_month : ℕ) : ℕ :=
  this_month + previous_month + next_month + following_month + fifth_month

theorem total_games_attended :
  games_in_months 24 32 29 19 34 = 138 :=
by
  -- Proof will be provided, but ignored for this problem
  sorry

end total_games_attended_l253_253916


namespace moskvich_halfway_from_zhiguli_to_b_l253_253068

-- Define the Moskvich's and Zhiguli's speeds as real numbers
variables (u v : ℝ)

-- Define the given conditions as named hypotheses
axiom speed_condition : u = v
axiom halfway_condition : u = (1 / 2) * (u + v) 

-- The mathematical statement we want to prove
theorem moskvich_halfway_from_zhiguli_to_b (speed_condition : u = v) (halfway_condition : u = (1 / 2) * (u + v)) : 
  ∃ t : ℝ, t = 2 := 
sorry -- Proof omitted

end moskvich_halfway_from_zhiguli_to_b_l253_253068


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253861

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253861


namespace fa_plus_fb_gt_zero_l253_253568

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Define the conditions for a and b
variables (a b : ℝ)
axiom ab_pos : a + b > 0

-- State the theorem
theorem fa_plus_fb_gt_zero : f a + f b > 0 :=
sorry

end fa_plus_fb_gt_zero_l253_253568


namespace total_earnings_l253_253374

def oil_change_cost : ℕ := 20
def repair_cost : ℕ := 30
def car_wash_cost : ℕ := 5

def num_oil_changes : ℕ := 5
def num_repairs : ℕ := 10
def num_car_washes : ℕ := 15

theorem total_earnings :
  (num_oil_changes * oil_change_cost) +
  (num_repairs * repair_cost) +
  (num_car_washes * car_wash_cost) = 475 :=
by
  sorry

end total_earnings_l253_253374


namespace scienceStudyTime_l253_253233

def totalStudyTime : ℕ := 60
def mathStudyTime : ℕ := 35

theorem scienceStudyTime : totalStudyTime - mathStudyTime = 25 :=
by sorry

end scienceStudyTime_l253_253233


namespace calc_a_squared_plus_b_squared_and_ab_l253_253123

theorem calc_a_squared_plus_b_squared_and_ab (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) :
  a^2 + b^2 = 5 ∧ a * b = 1 :=
by
  sorry

end calc_a_squared_plus_b_squared_and_ab_l253_253123


namespace john_has_hours_to_spare_l253_253186

def total_wall_area (num_walls : ℕ) (wall_width wall_height : ℕ) : ℕ :=
  num_walls * wall_width * wall_height

def time_to_paint_area (area : ℕ) (rate_per_square_meter_in_minutes : ℕ) : ℕ :=
  area * rate_per_square_meter_in_minutes

def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem john_has_hours_to_spare 
  (num_walls : ℕ) (wall_width wall_height : ℕ)
  (rate_per_square_meter_in_minutes : ℕ) (total_available_hours : ℕ)
  (to_spare_hours : ℕ)
  (h : total_wall_area num_walls wall_width wall_height = num_walls * wall_width * wall_height)
  (h1 : time_to_paint_area (num_walls * wall_width * wall_height) rate_per_square_meter_in_minutes = num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes)
  (h2 : minutes_to_hours (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes) = (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes) / 60)
  (h3 : total_available_hours = 10) 
  (h4 : to_spare_hours = total_available_hours - (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes / 60)) : 
  to_spare_hours = 5 := 
sorry

end john_has_hours_to_spare_l253_253186


namespace expected_number_of_ones_when_three_dice_rolled_l253_253962

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l253_253962


namespace sum_series_eq_3_div_4_l253_253829

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253829


namespace num_repeating_decimals_between_1_and_20_l253_253249

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end num_repeating_decimals_between_1_and_20_l253_253249


namespace contrapositive_equivalence_l253_253340

-- Definitions based on the conditions
variables (R S : Prop)

-- Statement of the proof
theorem contrapositive_equivalence (h : ¬R → S) : ¬S → R := 
sorry

end contrapositive_equivalence_l253_253340


namespace part_I_part_II_l253_253027

noncomputable def seq_a : ℕ → ℝ 
| 0       => 1   -- Normally, we start with n = 1, so we set a_0 to some default value.
| (n+1)   => (1 + 1 / (n^2 + n)) * seq_a n + 1 / (2^n)

theorem part_I (n : ℕ) (h: n ≥ 2) : seq_a n ≥ 2 :=
sorry

theorem part_II (n : ℕ) : seq_a n < Real.exp 2 :=
sorry

-- Assumption: ln(1 + x) < x for all x > 0
axiom ln_ineq (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x

end part_I_part_II_l253_253027


namespace green_minus_blue_is_40_l253_253549

noncomputable def number_of_green_minus_blue_disks (total_disks : ℕ) (ratio_blue : ℕ) (ratio_yellow : ℕ) (ratio_green : ℕ) : ℕ :=
  let total_ratio := ratio_blue + ratio_yellow + ratio_green
  let disks_per_part := total_disks / total_ratio
  let blue_disks := ratio_blue * disks_per_part
  let green_disks := ratio_green * disks_per_part
  green_disks - blue_disks

theorem green_minus_blue_is_40 :
  number_of_green_minus_blue_disks 144 3 7 8 = 40 :=
sorry

end green_minus_blue_is_40_l253_253549


namespace light_match_first_l253_253337

-- Define the conditions
def dark_room : Prop := true
def has_candle : Prop := true
def has_kerosene_lamp : Prop := true
def has_ready_to_use_stove : Prop := true
def has_single_match : Prop := true

-- Define the main question as a theorem
theorem light_match_first (h1 : dark_room) (h2 : has_candle) (h3 : has_kerosene_lamp) (h4 : has_ready_to_use_stove) (h5 : has_single_match) : true :=
by
  sorry

end light_match_first_l253_253337


namespace f_plus_one_odd_l253_253125

noncomputable def f : ℝ → ℝ := sorry

theorem f_plus_one_odd (f : ℝ → ℝ)
  (h : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 1) :
  ∀ x : ℝ, f x + 1 = -(f (-x) + 1) :=
sorry

end f_plus_one_odd_l253_253125


namespace sum_geometric_series_l253_253710

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253710


namespace prove_cardinality_l253_253295

-- Definitions used in Lean 4 Statement adapted from conditions
variable (a b : ℕ)
variable (A B : Finset ℕ)

-- Hypotheses
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_disjoint : Disjoint A B)
variable (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B)

-- The statement to prove
theorem prove_cardinality (a b : ℕ) (A B : Finset ℕ)
  (ha : a > 0) (hb : b > 0) (h_disjoint : Disjoint A B)
  (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B) :
  a * A.card = b * B.card :=
by 
  sorry

end prove_cardinality_l253_253295


namespace sum_series_eq_3_over_4_l253_253747

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253747


namespace total_earnings_correct_l253_253376

-- Given conditions
def charge_oil_change : ℕ := 20
def charge_repair : ℕ := 30
def charge_car_wash : ℕ := 5

def number_oil_changes : ℕ := 5
def number_repairs : ℕ := 10
def number_car_washes : ℕ := 15

-- Calculation of earnings based on the conditions
def earnings_from_oil_changes : ℕ := charge_oil_change * number_oil_changes
def earnings_from_repairs : ℕ := charge_repair * number_repairs
def earnings_from_car_washes : ℕ := charge_car_wash * number_car_washes

-- The total earnings
def total_earnings : ℕ := earnings_from_oil_changes + earnings_from_repairs + earnings_from_car_washes

-- Proof statement: Prove that the total earnings are $475
theorem total_earnings_correct : total_earnings = 475 := by -- our proof will go here
  sorry

end total_earnings_correct_l253_253376


namespace max_value_of_expression_l253_253607

noncomputable def max_expression_value (a b c : ℝ) : ℝ :=
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2)))

theorem max_value_of_expression (a b c : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) (hc : -1 < c ∧ c < 1) :
  max_expression_value a b c ≤ 2 :=
by sorry

end max_value_of_expression_l253_253607


namespace intersection_complement_l253_253148

def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}
def compl_U_N : Set ℕ := {x ∈ U | x ∉ N}

theorem intersection_complement :
  M ∩ compl_U_N = {4} :=
by
  have h1 : compl_U_N = {2, 4, 8} := by sorry
  have h2 : M ∩ compl_U_N = {4} := by sorry
  exact h2

end intersection_complement_l253_253148


namespace expected_ones_in_three_dice_rolls_l253_253944

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l253_253944


namespace Seojun_apples_decimal_l253_253012

theorem Seojun_apples_decimal :
  let total_apples := 100
  let seojun_apples := 11
  seojun_apples / total_apples = 0.11 :=
by
  let total_apples := 100
  let seojun_apples := 11
  sorry

end Seojun_apples_decimal_l253_253012


namespace find_a_value_l253_253243

theorem find_a_value : (15^2 * 8^3 / 256 = 450) :=
by
  sorry

end find_a_value_l253_253243


namespace range_of_a_plus_c_l253_253884

-- Let a, b, c be the sides of the triangle opposite to angles A, B, and C respectively.
variable (a b c A B C : ℝ)

-- Given conditions
variable (h1 : b = Real.sqrt 3)
variable (h2 : (2 * c - a) / b * Real.cos B = Real.cos A)
variable (h3 : 0 < A ∧ A < Real.pi / 2)
variable (h4 : 0 < B ∧ B < Real.pi / 2)
variable (h5 : 0 < C ∧ C < Real.pi / 2)
variable (h6 : A + B + C = Real.pi)

-- The range of a + c
theorem range_of_a_plus_c (a b c A B C : ℝ) (h1 : b = Real.sqrt 3)
  (h2 : (2 * c - a) / b * Real.cos B = Real.cos A) (h3 : 0 < A ∧ A < Real.pi / 2)
  (h4 : 0 < B ∧ B < Real.pi / 2) (h5 : 0 < C ∧ C < Real.pi / 2) (h6 : A + B + C = Real.pi) :
  a + c ∈ Set.Ioc (Real.sqrt 3) (2 * Real.sqrt 3) :=
  sorry

end range_of_a_plus_c_l253_253884


namespace line_tangent_to_ellipse_l253_253228

theorem line_tangent_to_ellipse (m : ℝ) (a : ℝ) (b : ℝ) (h_a : a = 3) (h_b : b = 1) :
  m^2 = 1 / 3 := by
  sorry

end line_tangent_to_ellipse_l253_253228


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253860

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253860


namespace car_speeds_l253_253225

theorem car_speeds (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v) / 2) :=
sorry

end car_speeds_l253_253225


namespace candidate1_fails_by_l253_253355

-- Define the total marks (T), passing marks (P), percentage marks (perc1 and perc2), and the extra marks.
def T : ℝ := 600
def P : ℝ := 160
def perc1 : ℝ := 0.20
def perc2 : ℝ := 0.30
def extra_marks : ℝ := 20

-- Define the marks obtained by the candidates.
def marks_candidate1 : ℝ := perc1 * T
def marks_candidate2 : ℝ := perc2 * T

-- The theorem stating the number of marks by which the first candidate fails.
theorem candidate1_fails_by (h_pass: perc2 * T = P + extra_marks) : P - marks_candidate1 = 40 :=
by
  -- The proof would go here.
  sorry

end candidate1_fails_by_l253_253355


namespace article_initial_cost_l253_253676

theorem article_initial_cost (x : ℝ) (h : 0.44 * x = 4400) : x = 10000 :=
by
  sorry

end article_initial_cost_l253_253676


namespace find_cost_of_apple_l253_253304

theorem find_cost_of_apple (A O : ℝ) 
  (h1 : 6 * A + 3 * O = 1.77) 
  (h2 : 2 * A + 5 * O = 1.27) : 
  A = 0.21 :=
by 
  sorry

end find_cost_of_apple_l253_253304


namespace order_of_fractions_l253_253493

theorem order_of_fractions (a b c d : ℚ)
  (h₁ : a = 21/14)
  (h₂ : b = 25/18)
  (h₃ : c = 23/16)
  (h₄ : d = 27/19)
  (h₅ : a > b)
  (h₆ : a > c)
  (h₇ : a > d)
  (h₈ : b < c)
  (h₉ : b < d)
  (h₁₀ : c > d) :
  b < d ∧ d < c ∧ c < a := 
sorry

end order_of_fractions_l253_253493


namespace min_value_l253_253262

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) : 
  ∃ m, m = (1 / x + 1 / y) ∧ m = 9 :=
by
  sorry

end min_value_l253_253262


namespace union_P_Q_l253_253269

noncomputable def P : Set ℝ := {x : ℝ | abs x ≥ 3}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x - 1}

theorem union_P_Q :
  (P ∪ Q) = Set.Iic (-3) ∪ Set.Ici (-1) :=
by {
  sorry
}

end union_P_Q_l253_253269


namespace evaluate_series_sum_l253_253767

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253767


namespace time_per_lawn_in_minutes_l253_253915

def jason_lawns := 16
def total_hours_cutting := 8
def minutes_per_hour := 60

theorem time_per_lawn_in_minutes : 
  (total_hours_cutting / jason_lawns) * minutes_per_hour = 30 :=
by
  sorry

end time_per_lawn_in_minutes_l253_253915


namespace domain_of_sqrt_function_l253_253313

theorem domain_of_sqrt_function (x : ℝ) :
  (x + 4 ≥ 0) ∧ (1 - x ≥ 0) ∧ (x ≠ 0) ↔ (-4 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) := 
sorry

end domain_of_sqrt_function_l253_253313


namespace length_of_CD_l253_253025

theorem length_of_CD (L : ℝ) (r : ℝ) (V_total : ℝ) (cylinder_vol : ℝ) (hemisphere_vol : ℝ) : 
  r = 5 ∧ V_total = 900 * Real.pi ∧ cylinder_vol = Real.pi * r^2 * L ∧ hemisphere_vol = (2/3) *Real.pi * r^3 → 
  V_total = cylinder_vol + 2 * hemisphere_vol → 
  L = 88 / 3 := 
by
  sorry

end length_of_CD_l253_253025


namespace area_enclosed_by_graph_eq_160_l253_253692

theorem area_enclosed_by_graph_eq_160 :
  ∃ (area : ℝ), area = 160 ∧
  (∀ (x y : ℝ), |2 * x| + |5 * y| = 20 → abs x ≤ 10 ∧ abs y ≤ 4) :=
begin
  sorry
end

end area_enclosed_by_graph_eq_160_l253_253692


namespace second_rectangle_area_l253_253931

theorem second_rectangle_area (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hbx : x < h):
  2 * b * x * (h - 3 * x) / h = (2 * b * x * (h - 3 * x))/h := 
sorry

end second_rectangle_area_l253_253931


namespace sum_k_binomial_l253_253092

theorem sum_k_binomial :
  (∃ k1 k2, k1 ≠ k2 ∧ nat.choose 26 k1 = nat.choose 25 5 + nat.choose 25 6 ∧
              nat.choose 26 k2 = nat.choose 25 5 + nat.choose 25 6 ∧ k1 + k2 = 26) :=
by
  use [6, 20]
  split
  { sorry } -- proof of k1 ≠ k2
  { split
    { simp [nat.choose] }
    { split
      { simp [nat.choose] }
      { simp }
    }
  }

end sum_k_binomial_l253_253092


namespace store_cost_comparison_l253_253469

noncomputable def store_A_cost (x : ℕ) : ℝ := 1760 + 40 * x
noncomputable def store_B_cost (x : ℕ) : ℝ := 1920 + 32 * x

theorem store_cost_comparison (x : ℕ) (h : x > 16) :
  (x > 20 → store_B_cost x < store_A_cost x) ∧ (x < 20 → store_A_cost x < store_B_cost x) :=
by
  sorry

end store_cost_comparison_l253_253469


namespace valid_parameterizations_l253_253934

-- Define the parameterization as a structure
structure LineParameterization where
  x : ℝ
  y : ℝ
  dx : ℝ
  dy : ℝ

-- Define the line equation
def line_eq (p : ℝ × ℝ) : Prop :=
  p.snd = -(2/3) * p.fst + 4

-- Proving which parameterizations are valid
theorem valid_parameterizations :
  (line_eq (3 + t * 3, 4 + t * (-2)) ∧
   line_eq (0 + t * 1.5, 4 + t * (-1)) ∧
   line_eq (1 + t * (-6), 3.33 + t * 4) ∧
   line_eq (5 + t * 1.5, (2/3) + t * (-1)) ∧
   line_eq (-6 + t * 9, 8 + t * (-6))) = 
  false ∧ true ∧ false ∧ true ∧ false :=
by
  sorry

end valid_parameterizations_l253_253934


namespace sqrt_sq_eq_abs_l253_253339

theorem sqrt_sq_eq_abs (a : ℝ) : Real.sqrt (a^2) = |a| :=
sorry

end sqrt_sq_eq_abs_l253_253339


namespace ratio_of_ages_l253_253046

variables (X Y : ℕ)

theorem ratio_of_ages (h1 : X - 6 = 24) (h2 : X + Y = 36) : X / Y = 2 :=
by 
  have h3 : X = 30 - 6 := by sorry
  have h4 : X = 24 := by sorry
  have h5 : X + Y = 36 := by sorry
  have h6 : Y = 12 := by sorry
  have h7 : X / Y = 2 := by sorry
  exact h7

end ratio_of_ages_l253_253046


namespace solution_set_M_minimum_value_expr_l253_253411

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

-- Proof problem (1): Prove that the solution set M of the inequality f(x) ≥ -1 is {x | 2/3 ≤ x ≤ 6}.
theorem solution_set_M : 
  { x : ℝ | f x ≥ -1 } = { x : ℝ | 2/3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Define the requirement for t and the expression to minimize
noncomputable def t : ℝ := 6
noncomputable def expr (a b c : ℝ) : ℝ := 1 / (2 * a + b) + 1 / (2 * a + c)

-- Proof problem (2): Given t = 6 and 4a + b + c = 6, prove that the minimum value of expr is 2/3.
theorem minimum_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = t) :
  expr a b c ≥ 2/3 :=
sorry

end solution_set_M_minimum_value_expr_l253_253411


namespace fraction_of_males_on_time_l253_253367

open Real

variable (A : ℝ) (M : ℝ)

theorem fraction_of_males_on_time
  (h1 : (3/5) * A + (2/5) * A = A)
  (h2 : (9/10) * (2/5) * A + M * (3/5) * A = 0.885 * A) :
  M = 0.875 :=
by 
  have h3 : (3 / 5 + 2 / 5) * A = A := by sorry -- This simplifies from h1
  have h4 : (3 / 5 * M + 0.36 * 2 / 5 = 0.885) := by sorry -- This converts the equation to a simpler form
  have h5 : (3 * M + 1.8 = 4.425) := by sorry -- Another form simplification
  have h6 : (3 * M = 2.625) := by sorry -- Isolate the term related to M
  have h7 : (M = 0.875) := by sorry -- Solve for M
  exact h7 -- Complete the proof

end fraction_of_males_on_time_l253_253367


namespace value_of_f_at_2019_l253_253208

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_positive : ∀ x : ℝ, f x > 0)
variable (h_functional : ∀ x : ℝ, f (x + 2) = 1 / (f x))

theorem value_of_f_at_2019 : f 2019 = 1 :=
by
  sorry

end value_of_f_at_2019_l253_253208


namespace cuts_for_20_pentagons_l253_253668

theorem cuts_for_20_pentagons (K : ℕ) : 20 * 540 + (K - 19) * 180 ≤ 360 * K + 540 ↔ K ≥ 38 :=
by
  sorry

end cuts_for_20_pentagons_l253_253668


namespace find_missing_number_l253_253431

noncomputable def missing_number : Prop :=
  ∃ (y x a b : ℝ),
    a = y + x ∧
    b = x + 630 ∧
    28 = y * a ∧
    660 = a * b ∧
    y = 13

theorem find_missing_number : missing_number :=
  sorry

end find_missing_number_l253_253431


namespace y_directly_proportional_x_l253_253577

-- Definition for direct proportionality
def directly_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x

-- Theorem stating the relationship between y and x given the condition
theorem y_directly_proportional_x (x y : ℝ) (h : directly_proportional x y) :
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x :=
by
  sorry

end y_directly_proportional_x_l253_253577


namespace hotdogs_per_hour_l253_253502

-- Define the necessary conditions
def price_per_hotdog : ℝ := 2
def total_hours : ℝ := 10
def total_sales : ℝ := 200

-- Prove that the number of hot dogs sold per hour equals 10
theorem hotdogs_per_hour : (total_sales / total_hours) / price_per_hotdog = 10 :=
by
  sorry

end hotdogs_per_hour_l253_253502


namespace solution_set_inequality_l253_253937

noncomputable def solution_set := {x : ℝ | (x + 1) * (x - 2) ≤ 0 ∧ x ≠ -1}

theorem solution_set_inequality :
  solution_set = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by {
-- Insert proof here
sorry
}

end solution_set_inequality_l253_253937


namespace sum_series_eq_3_div_4_l253_253827

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253827


namespace sum_series_equals_three_fourths_l253_253823

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253823


namespace series_sum_correct_l253_253792

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253792


namespace solve_equation_l253_253929

theorem solve_equation (x : ℝ) : 
  (9 - x - 2 * (31 - x) = 27) → (x = 80) :=
by
  sorry

end solve_equation_l253_253929


namespace product_of_roots_l253_253388

theorem product_of_roots :
  let a := 24
  let c := -216
  ∀ x : ℝ, (24 * x^2 + 36 * x - 216 = 0) → (c / a = -9) :=
by
  intros
  sorry

end product_of_roots_l253_253388


namespace two_digit_number_count_four_digit_number_count_l253_253051

-- Defining the set of digits
def digits : Finset ℕ := {1, 2, 3, 4}

-- Problem 1 condition and question
def two_digit_count := Nat.choose 4 2 * 2

-- Problem 2 condition and question
def four_digit_count := Nat.choose 4 4 * 24

-- Theorem statement for Problem 1
theorem two_digit_number_count : two_digit_count = 12 :=
sorry

-- Theorem statement for Problem 2
theorem four_digit_number_count : four_digit_count = 24 :=
sorry

end two_digit_number_count_four_digit_number_count_l253_253051


namespace sum_series_equals_three_fourths_l253_253821

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253821


namespace series_converges_to_three_fourths_l253_253698

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253698


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253864

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253864


namespace find_principal_l253_253171

variable (P : ℝ) (r : ℝ) (t : ℕ) (CI : ℝ) (SI : ℝ)

-- Define simple and compound interest
def simple_interest (P r : ℝ) (t : ℕ) : ℝ := P * r * t
def compound_interest (P r : ℝ) (t : ℕ) : ℝ := P * (1 + r)^t - P

-- Given conditions
axiom H1 : r = 0.05
axiom H2 : t = 2
axiom H3 : compound_interest P r t - simple_interest P r t = 18

-- The principal sum is 7200
theorem find_principal : P = 7200 := 
by sorry

end find_principal_l253_253171


namespace sum_series_eq_3_over_4_l253_253753

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l253_253753


namespace final_price_is_99_l253_253997

-- Conditions:
def original_price : ℝ := 120
def coupon_discount : ℝ := 10
def membership_discount_rate : ℝ := 0.10

-- Define final price calculation
def final_price (original_price coupon_discount membership_discount_rate : ℝ) : ℝ :=
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  price_after_coupon - membership_discount

-- Question: Is the final price equal to $99?
theorem final_price_is_99 :
  final_price original_price coupon_discount membership_discount_rate = 99 :=
by
  sorry

end final_price_is_99_l253_253997


namespace final_price_is_99_l253_253996

-- Conditions:
def original_price : ℝ := 120
def coupon_discount : ℝ := 10
def membership_discount_rate : ℝ := 0.10

-- Define final price calculation
def final_price (original_price coupon_discount membership_discount_rate : ℝ) : ℝ :=
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  price_after_coupon - membership_discount

-- Question: Is the final price equal to $99?
theorem final_price_is_99 :
  final_price original_price coupon_discount membership_discount_rate = 99 :=
by
  sorry

end final_price_is_99_l253_253996


namespace ratio_of_diamonds_to_spades_l253_253669

-- Given conditions
variable (total_cards : Nat := 13)
variable (black_cards : Nat := 7)
variable (red_cards : Nat := 6)
variable (clubs : Nat := 6)
variable (diamonds : Nat)
variable (spades : Nat)
variable (hearts : Nat := 2 * diamonds)
variable (cards_distribution : clubs + diamonds + hearts + spades = total_cards)
variable (black_distribution : clubs + spades = black_cards)

-- Define the proof theorem
theorem ratio_of_diamonds_to_spades : (diamonds / spades : ℝ) = 2 :=
 by
  -- temporarily we insert sorry to skip the proof
  sorry

end ratio_of_diamonds_to_spades_l253_253669


namespace A_and_B_mutually_exclusive_l253_253187

-- Definitions of events based on conditions
def A (a : ℕ) : Prop := a = 3
def B (a : ℕ) : Prop := a = 4

-- Define mutually exclusive
def mutually_exclusive (P Q : ℕ → Prop) : Prop :=
  ∀ a, P a → Q a → false

-- Problem statement: Prove A and B are mutually exclusive.
theorem A_and_B_mutually_exclusive :
  mutually_exclusive A B :=
sorry

end A_and_B_mutually_exclusive_l253_253187


namespace third_root_of_cubic_l253_253973

theorem third_root_of_cubic (a b : ℚ) (h1 : a ≠ 0) 
  (h2 : eval (-2 : ℚ) (a * X^3 + (a + 2 * b) * X^2 + (b - 3 * a) * X + (8 - a)) = 0)
  (h3 : eval (3 : ℚ) (a * X^3 + (a + 2 * b) * X^2 + (b - 3 * a) * X + (8 - a)) = 0) 
  : a * (4 / 3) ^ 3 + (a + 2 * b) * (4 / 3) ^ 2 + (b - 3 * a) * (4 / 3) + (8 - a) = 0 :=
by
  sorry

end third_root_of_cubic_l253_253973


namespace simplify_and_evaluate_l253_253625

theorem simplify_and_evaluate (x : ℝ) (hx : x = 6) :
  (1 + 2 / (x + 1)) * (x^2 + x) / (x^2 - 9) = 2 :=
by
  rw hx
  sorry

end simplify_and_evaluate_l253_253625


namespace sum_geometric_series_l253_253706

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253706


namespace num_repeating_decimals_1_to_20_l253_253247

theorem num_repeating_decimals_1_to_20 : 
  (∃ count_repeating : ℕ, count_repeating = 18 ∧ 
    ∀ n, 1 ≤ n ∧ n ≤ 20 → ((∃ k, n = 9 * k ∨ n = 18 * k) → false) → 
        (∃ d, (∃ normalized, n / 18 = normalized ∧ normalized.has_repeating_decimal))) :=
sorry

end num_repeating_decimals_1_to_20_l253_253247


namespace winning_candidate_percentage_is_57_l253_253351

def candidate_votes : List ℕ := [1136, 7636, 11628]

def total_votes : ℕ := candidate_votes.sum

def winning_votes : ℕ := candidate_votes.maximum?.getD 0

def winning_percentage (votes : ℕ) (total : ℕ) : ℚ :=
  (votes * 100) / total

theorem winning_candidate_percentage_is_57 :
  winning_percentage winning_votes total_votes = 57 := by
  sorry

end winning_candidate_percentage_is_57_l253_253351


namespace sum_ends_in_zero_squares_end_same_digit_l253_253302

theorem sum_ends_in_zero_squares_end_same_digit (a b : ℕ) (h : (a + b) % 10 = 0) : (a^2 % 10) = (b^2 % 10) := 
sorry

end sum_ends_in_zero_squares_end_same_digit_l253_253302


namespace distribute_books_l253_253079

theorem distribute_books : 
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5
  total_ways - subtract_one_student_none + add_two_students_none = 240 :=
by
  -- Definitions based on conditions in a)
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5

  -- The final calculation
  have h : total_ways - subtract_one_student_none + add_two_students_none = 240 := by sorry
  exact h

end distribute_books_l253_253079


namespace part1_solution_set_part2_minimum_value_l253_253414

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

theorem part1_solution_set (x : ℝ) :
  (f x ≥ -1) ↔ (2 / 3 ≤ x ∧ x ≤ 6) := sorry

variables {a b c : ℝ}
theorem part2_minimum_value (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = 6) :
  (1 / (2 * a + b) + 1 / (2 * a + c) ≥ 2 / 3) := 
sorry

end part1_solution_set_part2_minimum_value_l253_253414


namespace quinn_frogs_caught_l253_253165

-- Defining the conditions
def Alster_frogs : Nat := 2

def Quinn_frogs (Alster_caught: Nat) : Nat := Alster_caught

def Bret_frogs (Quinn_caught: Nat) : Nat := 3 * Quinn_caught

-- Given that Bret caught 12 frogs, prove the amount Quinn caught
theorem quinn_frogs_caught (Bret_caught: Nat) (h1: Bret_caught = 12) : Quinn_frogs Alster_frogs = 4 :=
by
  sorry

end quinn_frogs_caught_l253_253165


namespace smallest_positive_m_l253_253548

theorem smallest_positive_m (m : ℕ) (h : ∀ (n : ℕ), n % 2 = 1 → (529^n + m * 132^n) % 262417 = 0) : m = 1 :=
sorry

end smallest_positive_m_l253_253548


namespace continuous_implies_defined_defined_does_not_imply_continuous_l253_253229

-- Define function continuity at a point x = a
def continuous_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - a) < δ → abs (f x - f a) < ε

-- Prove that if f is continuous at x = a, then f is defined at x = a
theorem continuous_implies_defined (f : ℝ → ℝ) (a : ℝ) : 
  continuous_at f a → ∃ y, f a = y :=
by
  sorry  -- Proof omitted

-- Prove that the definition of f at x = a does not guarantee continuity at x = a
theorem defined_does_not_imply_continuous (f : ℝ → ℝ) (a : ℝ) :
  (∃ y, f a = y) → ¬ continuous_at f a :=
by
  sorry  -- Proof omitted

end continuous_implies_defined_defined_does_not_imply_continuous_l253_253229


namespace income_before_taxes_l253_253347

/-- Define given conditions -/
def net_income (x : ℝ) : ℝ := x - 0.10 * (x - 3000)

/-- Prove that the income before taxes must have been 13000 given the conditions. -/
theorem income_before_taxes (x : ℝ) (hx : net_income x = 12000) : x = 13000 :=
by sorry

end income_before_taxes_l253_253347


namespace equivalent_proof_l253_253407

theorem equivalent_proof :
  let a := 4
  let b := Real.sqrt 17 - a
  b^2020 * (a + Real.sqrt 17)^2021 = Real.sqrt 17 + 4 :=
by
  let a := 4
  let b := Real.sqrt 17 - a
  sorry

end equivalent_proof_l253_253407


namespace tan_135_eq_neg_one_l253_253535

theorem tan_135_eq_neg_one : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg_one_l253_253535


namespace rectangle_area_l253_253672

-- Conditions: 
-- 1. The length of the rectangle is three times its width.
-- 2. The diagonal length of the rectangle is x.

theorem rectangle_area (x : ℝ) (w l : ℝ) (h1 : w * 3 = l) (h2 : w^2 + l^2 = x^2) :
  l * w = (3 / 10) * x^2 :=
by
  sorry

end rectangle_area_l253_253672


namespace train_speed_is_60_0131_l253_253217

noncomputable def train_speed (speed_of_man_kmh : ℝ) (length_of_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_of_man_ms := speed_of_man_kmh * 1000 / 3600
  let relative_speed := length_of_train_m / time_s
  let train_speed_ms := relative_speed - speed_of_man_ms
  train_speed_ms * 3600 / 1000

theorem train_speed_is_60_0131 :
  train_speed 6 330 17.998560115190788 = 60.0131 := by
  sorry

end train_speed_is_60_0131_l253_253217


namespace circular_sequence_zero_if_equidistant_l253_253610

noncomputable def circular_sequence_property (x y z : ℤ): Prop :=
  (x = 0 ∧ y = 0 ∧ dist x y = dist y z) → z = 0

theorem circular_sequence_zero_if_equidistant {x y z : ℤ} :
  (x = 0 ∧ y = 0 ∧ dist x y = dist y z) → z = 0 :=
by sorry

end circular_sequence_zero_if_equidistant_l253_253610


namespace gcd_72_and_120_l253_253087

theorem gcd_72_and_120 : Nat.gcd 72 120 = 24 := 
by
  sorry

end gcd_72_and_120_l253_253087


namespace no_consecutive_even_square_and_three_times_square_no_consecutive_square_and_seven_times_square_l253_253619

-- Problem 1: Square of an even number followed by three times a square number
theorem no_consecutive_even_square_and_three_times_square :
  ∀ (k n : ℕ), ¬(3 * n ^ 2 = 4 * k ^ 2 + 1) :=
by sorry

-- Problem 2: Square number followed by seven times another square number
theorem no_consecutive_square_and_seven_times_square :
  ∀ (r s : ℕ), ¬(7 * s ^ 2 = r ^ 2 + 1) :=
by sorry

end no_consecutive_even_square_and_three_times_square_no_consecutive_square_and_seven_times_square_l253_253619


namespace tan_135_eq_neg1_l253_253536

theorem tan_135_eq_neg1 :
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in
  Q = -(Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * Complex.I →
  Complex.tan (135 * Real.pi / 180 * Complex.I) = -1 :=
by
  intro hQ
  have Q_coords : Q = -(Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * Complex.I := hQ
  sorry

end tan_135_eq_neg1_l253_253536


namespace a3_eq_5_l253_253257

variable {a_n : ℕ → Real} (S : ℕ → Real)
variable (a1 d : Real)

-- Define arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → Real) (a1 d : Real) : Prop :=
  ∀ n : ℕ, n > 0 → a_n n = a1 + (n - 1) * d

-- Define sum of first n terms
def sum_of_arithmetic (S : ℕ → Real) (a_n : ℕ → Real) : Prop :=
  ∀ n : ℕ, S n = n / 2 * (a_n 1 + a_n n)

-- Given conditions: S_5 = 25
def S_5_eq_25 (S : ℕ → Real) : Prop :=
  S 5 = 25

-- Goal: prove a_3 = 5
theorem a3_eq_5 (h_arith : is_arithmetic_sequence a_n a1 d)
                (h_sum : sum_of_arithmetic S a_n)
                (h_S5 : S_5_eq_25 S) : a_n 3 = 5 :=
  sorry

end a3_eq_5_l253_253257


namespace new_game_cost_l253_253496

theorem new_game_cost (G : ℕ) (h_initial_money : 83 = G + 9 * 4) : G = 47 := by
  sorry

end new_game_cost_l253_253496


namespace odd_function_and_monotonic_decreasing_l253_253255

variable (f : ℝ → ℝ)

-- Given conditions:
axiom condition_1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom condition_2 : ∀ x : ℝ, x > 0 → f x < 0

-- Statement to prove:
theorem odd_function_and_monotonic_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2) := by
  sorry

end odd_function_and_monotonic_decreasing_l253_253255


namespace series_sum_l253_253741

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253741


namespace sugar_and_granulated_sugar_delivered_l253_253510

theorem sugar_and_granulated_sugar_delivered (total_bags : ℕ) (percentage_more : ℚ) (mass_ratio : ℚ) (total_weight : ℚ)
    (h_total_bags : total_bags = 63)
    (h_percentage_more : percentage_more = 1.25)
    (h_mass_ratio : mass_ratio = 3 / 4)
    (h_total_weight : total_weight = 4.8) :
    ∃ (sugar_weight granulated_sugar_weight : ℚ),
        (granulated_sugar_weight = 1.8) ∧ (sugar_weight = 3) ∧
        ((sugar_weight + granulated_sugar_weight = total_weight) ∧
        (sugar_weight / 28 = (granulated_sugar_weight / 35) * mass_ratio)) :=
by
    sorry

end sugar_and_granulated_sugar_delivered_l253_253510


namespace series_converges_to_three_fourths_l253_253697

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253697


namespace area_of_region_R_l253_253605

def greatest_integer (x : ℝ) : ℤ :=
  ⌊x⌋

def region_R (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + y + greatest_integer x + greatest_integer y ≤ 7

theorem area_of_region_R :
  let R := { p : ℝ × ℝ | region_R p.1 p.2 } in
  is_measurable R ∧ measure_theory.measure_space.volume.measure R = 8 :=
by
  sorry

end area_of_region_R_l253_253605


namespace probability_at_least_two_red_balls_l253_253306

noncomputable def prob_red_balls (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (drawn_balls : ℕ) : ℚ :=
if total_balls = 6 ∧ red_balls = 3 ∧ white_balls = 2 ∧ black_balls = 1 ∧ drawn_balls = 3 then
  1 / 2
else
  0

theorem probability_at_least_two_red_balls :
  prob_red_balls 6 3 2 1 3 = 1 / 2 :=
by 
  sorry

end probability_at_least_two_red_balls_l253_253306


namespace scientific_notation_of_number_l253_253325

theorem scientific_notation_of_number :
  (0.000000014 : ℝ) = 1.4 * 10 ^ (-8) :=
sorry

end scientific_notation_of_number_l253_253325


namespace infinite_series_sum_eq_l253_253718

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253718


namespace first_number_is_45_l253_253652

theorem first_number_is_45 (a b : ℕ) (h1 : a / gcd a b = 3) (h2 : b / gcd a b = 4) (h3 : lcm a b = 180) : a = 45 := by
  sorry

end first_number_is_45_l253_253652


namespace sum_of_integers_k_sum_of_all_integers_k_l253_253095
open Nat

theorem sum_of_integers_k (k : ℕ) (h : choose 25 5 + choose 25 6 = choose 26 k) : k = 6 ∨ k = 20 :=
begin
  sorry,
end

theorem sum_of_all_integers_k : 
  (∃ k, (choose 25 5 + choose 25 6 = choose 26 k) → k = 6 ∨ k = 20) → 6 + 20 = 26 :=
begin
  sorry,
end

end sum_of_integers_k_sum_of_all_integers_k_l253_253095


namespace infinite_series_sum_eq_3_over_4_l253_253848

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253848


namespace Ron_needs_to_drink_80_percent_l253_253349

theorem Ron_needs_to_drink_80_percent 
  (volume_each : ℕ)
  (volume_intelligence : ℕ)
  (volume_beauty : ℕ)
  (volume_strength : ℕ)
  (volume_second_pitcher : ℕ)
  (effective_volume : ℕ)
  (volume_intelligence_left : ℕ)
  (volume_beauty_left : ℕ)
  (volume_strength_left : ℕ)
  (total_volume : ℕ)
  (Ron_needs : ℕ)
  (intelligence_condition : effective_volume = 30)
  (initial_volumes : volume_each = 300)
  (first_drink : volume_intelligence = volume_each / 2)
  (mix_before_second_drink : volume_second_pitcher = volume_intelligence + volume_beauty)
  (Hermione_drink : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left)
  (Harry_drink : volume_strength_left = volume_each / 2)
  (second_mix : volume_second_pitcher = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (final_mix : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (Ron_needs_condition : Ron_needs = effective_volume / volume_intelligence_left * 100)
  : Ron_needs = 80 := sorry

end Ron_needs_to_drink_80_percent_l253_253349


namespace circle_radius_l253_253693

theorem circle_radius 
  (x y : ℝ)
  (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = Real.sqrt 117 :=
by 
  sorry

end circle_radius_l253_253693


namespace scale_drawing_l253_253214

theorem scale_drawing (length_cm : ℝ) (representation : ℝ) : length_cm * representation = 3750 :=
by
  let length_cm := 7.5
  let representation := 500
  sorry

end scale_drawing_l253_253214


namespace intersection_is_4_l253_253149

-- Definitions of the sets
def U : Set Int := {0, 1, 2, 4, 6, 8}
def M : Set Int := {0, 4, 6}
def N : Set Int := {0, 1, 6}

-- Definition of the complement
def complement_U_N : Set Int := U \ N

-- Definition of the intersection
def intersection_M_complement_U_N : Set Int := M ∩ complement_U_N

-- Statement of the theorem
theorem intersection_is_4 : intersection_M_complement_U_N = {4} :=
by
  sorry

end intersection_is_4_l253_253149


namespace factorization_of_x12_minus_4096_l253_253528

variable (x : ℝ)

theorem factorization_of_x12_minus_4096 : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end factorization_of_x12_minus_4096_l253_253528


namespace relationship_l253_253039

-- Given definitions
def S : ℕ := 31
def L : ℕ := 124 - S

-- Proving the relationship
theorem relationship: S + L = 124 ∧ S = 31 → L = S + 62 := by
  sorry

end relationship_l253_253039


namespace find_f2_l253_253422

def f (x : ℝ) : ℝ := sorry

theorem find_f2 : (∀ x, f (x-1) = x / (x-1)) → f 2 = 3 / 2 :=
by
  sorry

end find_f2_l253_253422


namespace smallest_a_for_quadratic_poly_l253_253557

theorem smallest_a_for_quadratic_poly (a : ℕ) (a_pos : 0 < a) :
  (∃ b c : ℤ, ∀ x : ℝ, 0 < x ∧ x < 1 → a*x^2 + b*x + c = 0 → (2 : ℝ)^2 - (4 : ℝ)*(a * c) < 0 ∧ b^2 - 4*a*c ≥ 1) → a ≥ 5 := 
sorry

end smallest_a_for_quadratic_poly_l253_253557


namespace evaluate_series_sum_l253_253766

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l253_253766


namespace expression_simplification_l253_253896

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 2 * x + y / 2 ≠ 0) :
  (2 * x + y / 2)⁻¹ * ((2 * x)⁻¹ + (y / 2)⁻¹) = (x * y)⁻¹ := 
sorry

end expression_simplification_l253_253896


namespace buses_needed_l253_253636

def total_students : ℕ := 111
def seats_per_bus : ℕ := 3

theorem buses_needed : total_students / seats_per_bus = 37 :=
by
  sorry

end buses_needed_l253_253636


namespace merchant_discount_percentage_l253_253667

theorem merchant_discount_percentage
  (CP MP SP : ℝ)
  (h1 : MP = CP + 0.40 * CP)
  (h2 : SP = CP + 0.26 * CP)
  : ((MP - SP) / MP) * 100 = 10 := by
  sorry

end merchant_discount_percentage_l253_253667


namespace master_zhang_must_sell_100_apples_l253_253613

-- Define the given conditions
def buying_price_per_apple : ℚ := 1 / 4 -- 1 yuan for 4 apples
def selling_price_per_apple : ℚ := 2 / 5 -- 2 yuan for 5 apples
def profit_per_apple : ℚ := selling_price_per_apple - buying_price_per_apple

-- Define the target profit
def target_profit : ℚ := 15

-- Define the number of apples to sell
def apples_to_sell : ℚ := target_profit / profit_per_apple

-- The theorem statement: Master Zhang must sell 100 apples to achieve the target profit of 15 yuan
theorem master_zhang_must_sell_100_apples :
  apples_to_sell = 100 :=
sorry

end master_zhang_must_sell_100_apples_l253_253613


namespace optimal_washing_effect_l253_253350

noncomputable def total_capacity : ℝ := 20 -- kilograms
noncomputable def weight_clothes : ℝ := 5 -- kilograms
noncomputable def weight_detergent_existing : ℝ := 2 * 0.02 -- kilograms
noncomputable def optimal_concentration : ℝ := 0.004 -- kilograms per kilogram of water

theorem optimal_washing_effect :
  ∃ (additional_detergent additional_water : ℝ),
    additional_detergent = 0.02 ∧ additional_water = 14.94 ∧
    weight_clothes + additional_water + weight_detergent_existing + additional_detergent = total_capacity ∧
    weight_detergent_existing + additional_detergent = optimal_concentration * additional_water :=
by
  sorry

end optimal_washing_effect_l253_253350


namespace probability_XiaoCong_project_A_probability_same_project_not_C_l253_253650

-- Definition of projects and conditions
inductive Project
| A | B | C

def XiaoCong : Project := sorry
def XiaoYing : Project := sorry

-- (1) Probability of Xiao Cong assigned to project A
theorem probability_XiaoCong_project_A : 
  (1 / 3 : ℝ) = 1 / 3 := 
by sorry

-- (2) Probability of Xiao Cong and Xiao Ying being assigned to the same project, given Xiao Ying not assigned to C
theorem probability_same_project_not_C : 
  (2 / 6 : ℝ) = 1 / 3 :=
by sorry

end probability_XiaoCong_project_A_probability_same_project_not_C_l253_253650


namespace inequality_solution_l253_253084

theorem inequality_solution (x : ℝ) :
  (\frac{x + 1}{x - 2} + \frac{x + 3}{3*x} ≥ 4) ↔ (0 < x ∧ x ≤ 1/4) ∨ (1 < x ∧ x ≤ 2) :=
sorry

end inequality_solution_l253_253084


namespace series_sum_eq_l253_253730

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253730


namespace parabolas_equation_l253_253871

theorem parabolas_equation (vertex_origin : (0, 0) ∈ {(x, y) | y = x^2} ∨ (0, 0) ∈ {(x, y) | x = -y^2})
  (focus_on_axis : ∀ F : ℝ × ℝ, (F ∈ {(x, y) | y = x^2} ∨ F ∈ {(x, y) | x = -y^2}) → (F.1 = 0 ∨ F.2 = 0))
  (through_point : (-2, 4) ∈ {(x, y) | y = x^2} ∨ (-2, 4) ∈ {(x, y) | x = -y^2}) :
  {(x, y) | y = x^2} ∪ {(x, y) | x = -y^2} ≠ ∅ :=
by
  sorry

end parabolas_equation_l253_253871


namespace sum_of_integers_75_to_95_l253_253490

def arithmeticSumOfIntegers (a l : ℕ) : ℕ :=
  let n := l - a + 1
  n / 2 * (a + l)

theorem sum_of_integers_75_to_95 : arithmeticSumOfIntegers 75 95 = 1785 :=
  by
  sorry

end sum_of_integers_75_to_95_l253_253490


namespace infinite_series_sum_value_l253_253811

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253811


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253863

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253863


namespace partition_sum_nine_times_l253_253082

theorem partition_sum_nine_times (k : ℕ) :
  ∃ (A B : Set ℕ), 
    (S = {1994 + 3 * i | i ∈ Finset.range (k + 1)}) ∧
    (A ∪ B = S) ∧
    (A ∩ B = ∅) ∧
    (∑ x in A, x = 9 * ∑ x in B, x) →
  (∃ t : ℕ, k = 20 * t - 1 ∨ k = 20 * t + 4) :=
begin
  sorry
end

end partition_sum_nine_times_l253_253082


namespace Mildred_final_oranges_l253_253156

def initial_oranges : ℕ := 215
def father_oranges : ℕ := 3 * initial_oranges
def total_after_father : ℕ := initial_oranges + father_oranges
def sister_takes_away : ℕ := 174
def after_sister : ℕ := total_after_father - sister_takes_away
def final_oranges : ℕ := 2 * after_sister

theorem Mildred_final_oranges : final_oranges = 1372 := by
  sorry

end Mildred_final_oranges_l253_253156


namespace fraction_of_painted_surface_area_l253_253060

def total_surface_area_of_smaller_prisms : ℕ := 
  let num_smaller_prisms := 27
  let num_square_faces := num_smaller_prisms * 3
  let num_triangular_faces := num_smaller_prisms * 2
  num_square_faces + num_triangular_faces

def painted_surface_area_of_larger_prism : ℕ :=
  let painted_square_faces := 3 * 9
  let painted_triangular_faces := 2 * 9
  painted_square_faces + painted_triangular_faces

theorem fraction_of_painted_surface_area : 
  (painted_surface_area_of_larger_prism : ℚ) / (total_surface_area_of_smaller_prisms : ℚ) = 1 / 3 :=
by sorry

end fraction_of_painted_surface_area_l253_253060


namespace num_males_in_group_l253_253430

-- Definitions based on the given conditions
def num_females (f : ℕ) : Prop := f = 16
def num_males_choose_malt (m_malt : ℕ) : Prop := m_malt = 6
def num_females_choose_malt (f_malt : ℕ) : Prop := f_malt = 8
def num_choose_malt (m_malt f_malt n_malt : ℕ) : Prop := n_malt = m_malt + f_malt
def num_choose_coke (c : ℕ) (n_malt : ℕ) : Prop := n_malt = 2 * c
def total_cheerleaders (t : ℕ) (n_malt c : ℕ) : Prop := t = n_malt + c
def num_males (m f t : ℕ) : Prop := m = t - f

theorem num_males_in_group
  (f m_malt f_malt n_malt c t m : ℕ)
  (hf : num_females f)
  (hmm : num_males_choose_malt m_malt)
  (hfm : num_females_choose_malt f_malt)
  (hmalt : num_choose_malt m_malt f_malt n_malt)
  (hc : num_choose_coke c n_malt)
  (ht : total_cheerleaders t n_malt c)
  (hm : num_males m f t) :
  m = 5 := 
sorry

end num_males_in_group_l253_253430


namespace mean_score_seniors_138_l253_253037

def total_students : ℕ := 200
def mean_score_all : ℕ := 120

variable (s n : ℕ) -- number of seniors and non-seniors
variable (ms mn : ℚ) -- mean score of seniors and non-seniors

def non_seniors_twice_seniors := n = 2 * s
def mean_score_non_seniors := mn = 0.8 * ms
def total_students_eq := s + n = total_students

def total_score := (s : ℚ) * ms + (n : ℚ) * mn = (total_students : ℚ) * mean_score_all

theorem mean_score_seniors_138 :
  ∃ s n ms mn,
    non_seniors_twice_seniors s n ∧
    mean_score_non_seniors ms mn ∧
    total_students_eq s n ∧
    total_score s n ms mn → 
    ms = 138 :=
sorry

end mean_score_seniors_138_l253_253037


namespace proof_l253_253403

noncomputable def problem_statement (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∀ x : ℝ, |x + a| + |x - b| + c ≥ 4)

theorem proof (a b c : ℝ) (h : problem_statement a b c) :
  a + b + c = 4 ∧ (∀ x : ℝ, 1 / a + 4 / b + 9 / c ≥ 9) :=
by
  sorry

end proof_l253_253403


namespace geometric_sequence_arithmetic_progression_l253_253556

theorem geometric_sequence_arithmetic_progression
  (q : ℝ) (h_q : q ≠ 1)
  (a : ℕ → ℝ) (m n p : ℕ)
  (h1 : ∃ a1, ∀ k, a k = a1 * q ^ (k - 1))
  (h2 : a n ^ 2 = a m * a p) :
  2 * n = m + p := 
by
  sorry

end geometric_sequence_arithmetic_progression_l253_253556


namespace triangle_heights_inequality_l253_253609

variable {R : Type} [OrderedRing R]

theorem triangle_heights_inequality (m_a m_b m_c s : R) 
  (h_m_a_nonneg : 0 ≤ m_a) (h_m_b_nonneg : 0 ≤ m_b) (h_m_c_nonneg : 0 ≤ m_c)
  (h_s_nonneg : 0 ≤ s) : 
  m_a^2 + m_b^2 + m_c^2 ≤ s^2 := 
by
  sorry

end triangle_heights_inequality_l253_253609


namespace problem1_problem2_l253_253224

theorem problem1 : (-(3 / 4) - (5 / 8) + (9 / 12)) * (-24) = 15 := by
  sorry

theorem problem2 : (-1 ^ 6 + |(-2) ^ 3 - 10| - (-3) / (-1) ^ 2023) = 14 := by
  sorry

end problem1_problem2_l253_253224


namespace team_combinations_l253_253632

/-- 
The math club at Walnutridge High School has five girls and seven boys. 
How many different teams, comprising two girls and two boys, can be formed 
if one boy on each team must also be designated as the team leader?
-/
theorem team_combinations (girls boys : ℕ) (h_girls : girls = 5) (h_boys : boys = 7) :
  ∃ n, n = 420 :=
by
  sorry

end team_combinations_l253_253632


namespace sum_series_div_3_powers_l253_253756

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253756


namespace expected_value_of_ones_on_three_dice_l253_253943

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l253_253943


namespace part1_part2_l253_253417

open Set

variable (A B : Set ℝ) (m : ℝ)

def setA : Set ℝ := {x | x ^ 2 - 2 * x - 8 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | x ^ 2 - (2 * m - 3) * x + m ^ 2 - 3 * m ≤ 0}

theorem part1 (h : (setA ∩ setB 5) = Icc 2 4) : m = 5 := sorry

theorem part2 (h : setA ⊆ compl (setB m)) :
  m ∈ Iio (-2) ∪ Ioi 7 := sorry

end part1_part2_l253_253417


namespace solution_set_of_inequality_l253_253608

theorem solution_set_of_inequality (x : ℝ) : (|x - 3| < 1) → (2 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_inequality_l253_253608


namespace slope_positive_if_and_only_if_l253_253479

/-- Given points A(2, 1) and B(1, m^2), the slope of the line passing through them is positive,
if and only if m is in the range -1 < m < 1. -/
theorem slope_positive_if_and_only_if
  (m : ℝ) : 1 - m^2 > 0 ↔ -1 < m ∧ m < 1 :=
by
  sorry

end slope_positive_if_and_only_if_l253_253479


namespace pq_sum_l253_253296

open Real

section Problem
variables (p q : ℝ)
  (hp : p^3 - 21 * p^2 + 35 * p - 105 = 0)
  (hq : 5 * q^3 - 35 * q^2 - 175 * q + 1225 = 0)

theorem pq_sum : p + q = 21 / 2 :=
sorry
end Problem

end pq_sum_l253_253296


namespace car_dealership_l253_253680

variable (sportsCars : ℕ) (sedans : ℕ) (trucks : ℕ)

theorem car_dealership (h1 : 3 * sedans = 5 * sportsCars) 
  (h2 : 3 * trucks = 3 * sportsCars) 
  (h3 : sportsCars = 45) : 
  sedans = 75 ∧ trucks = 45 := by
  sorry

end car_dealership_l253_253680


namespace completing_square_correct_l253_253928

theorem completing_square_correct :
  ∀ x : ℝ, (x^2 - 4 * x + 2 = 0) ↔ ((x - 2)^2 = 2) := 
by
  intros x
  sorry

end completing_square_correct_l253_253928


namespace expected_number_of_ones_on_three_dice_l253_253954

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l253_253954


namespace series_sum_eq_l253_253734

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253734


namespace train_stop_times_l253_253938

theorem train_stop_times :
  ∀ (speed_without_stops_A speed_with_stops_A speed_without_stops_B speed_with_stops_B : ℕ),
  speed_without_stops_A = 45 →
  speed_with_stops_A = 30 →
  speed_without_stops_B = 60 →
  speed_with_stops_B = 40 →
  (60 * (speed_without_stops_A - speed_with_stops_A) / speed_without_stops_A = 20) ∧
  (60 * (speed_without_stops_B - speed_with_stops_B) / speed_without_stops_B = 20) :=
by
  intros
  sorry

end train_stop_times_l253_253938


namespace series_sum_eq_l253_253729

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253729


namespace johnny_ways_to_choose_l253_253141

def num_ways_to_choose_marbles (total_marbles : ℕ) (marbles_to_choose : ℕ) (blue_must_be_included : ℕ) : ℕ :=
  Nat.choose (total_marbles - blue_must_be_included) (marbles_to_choose - blue_must_be_included)

-- Given conditions
def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def blue_must_be_included : ℕ := 1

-- Theorem to prove the number of ways to choose the marbles
theorem johnny_ways_to_choose :
  num_ways_to_choose_marbles total_marbles marbles_to_choose blue_must_be_included = 56 := by
  sorry

end johnny_ways_to_choose_l253_253141


namespace fraction_bounds_l253_253330

theorem fraction_bounds (n : ℕ) (h : 0 < n) : (1 : ℚ) / 2 ≤ n / (n + 1 : ℚ) ∧ n / (n + 1 : ℚ) < 1 :=
by
  sorry

end fraction_bounds_l253_253330


namespace reciprocal_sum_neg_l253_253561

theorem reciprocal_sum_neg (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c = 8) : (1/a) + (1/b) + (1/c) < 0 := 
sorry

end reciprocal_sum_neg_l253_253561


namespace palindrome_clock_count_l253_253581

-- Definitions based on conditions from the problem statement.
def is_valid_hour (h : ℕ) : Prop := h < 24
def is_valid_minute (m : ℕ) : Prop := m < 60
def is_palindrome (h m : ℕ) : Prop :=
  (h < 10 ∧ m / 10 = h ∧ m % 10 = h) ∨
  (h >= 10 ∧ (h / 10) = (m % 10) ∧ (h % 10) = (m / 10 % 10))

-- Main theorem statement
theorem palindrome_clock_count : 
  (∃ n : ℕ, n = 66 ∧ ∀ (h m : ℕ), is_valid_hour h → is_valid_minute m → is_palindrome h m) := 
sorry

end palindrome_clock_count_l253_253581


namespace correct_quotient_l253_253587

variable (D : ℕ) (q1 q2 : ℕ)
variable (h1 : q1 = 4900) (h2 : D - 1000 = 1200 * q1)

theorem correct_quotient : q2 = D / 2100 → q2 = 2800 :=
by
  sorry

end correct_quotient_l253_253587


namespace remainder_n_squared_plus_3n_plus_5_l253_253898

theorem remainder_n_squared_plus_3n_plus_5 (n : ℕ) (h : n % 25 = 24) : (n^2 + 3 * n + 5) % 25 = 3 :=
by
  sorry

end remainder_n_squared_plus_3n_plus_5_l253_253898


namespace find_b_l253_253157

theorem find_b (a b : ℕ) (h1 : (a + b) % 10 = 5) (h2 : (a + b) % 7 = 4) : b = 2 := 
sorry

end find_b_l253_253157


namespace factorization_of_x12_minus_4096_l253_253529

variable (x : ℝ)

theorem factorization_of_x12_minus_4096 : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end factorization_of_x12_minus_4096_l253_253529


namespace sum_series_equals_three_fourths_l253_253815

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253815


namespace simplify_T_l253_253449

theorem simplify_T (x : ℝ) : 
  (x + 2)^6 + 6 * (x + 2)^5 + 15 * (x + 2)^4 + 20 * (x + 2)^3 + 15 * (x + 2)^2 + 6 * (x + 2) + 1 = (x + 3)^6 :=
by
  sorry

end simplify_T_l253_253449


namespace no_three_consecutive_geo_prog_l253_253525

theorem no_three_consecutive_geo_prog (n k m: ℕ) (h: n ≠ k ∧ n ≠ m ∧ k ≠ m) :
  ¬(∃ a b c: ℕ, 
    (a = 2^n + 1 ∧ b = 2^k + 1 ∧ c = 2^m + 1) ∧ 
    (b^2 = a * c)) :=
by sorry

end no_three_consecutive_geo_prog_l253_253525


namespace geometric_seq_a4_l253_253429

theorem geometric_seq_a4 (a : ℕ → ℕ) (q : ℕ) (h_q : q = 2) 
  (h_a1a3 : a 0 * a 2 = 6 * a 1) : a 3 = 24 :=
by
  -- Skipped proof
  sorry

end geometric_seq_a4_l253_253429


namespace series_result_l253_253836

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253836


namespace number_of_members_l253_253434

def cost_knee_pads : ℤ := 6
def cost_jersey : ℤ := cost_knee_pads + 7
def total_cost_per_member : ℤ := 2 * (cost_knee_pads + cost_jersey)
def total_expenditure : ℤ := 3120

theorem number_of_members (n : ℤ) (h : n * total_cost_per_member = total_expenditure) : n = 82 :=
sorry

end number_of_members_l253_253434


namespace find_m_plus_n_l253_253029

variable (U : Set ℝ) (A : Set ℝ) (CUA : Set ℝ) (m n : ℝ)
  -- Condition 1: The universal set U is the set of all real numbers
  (hU : U = Set.univ)
  -- Condition 2: A is defined as the set of all x such that (x - 1)(x - m) > 0
  (hA : A = { x : ℝ | (x - 1) * (x - m) > 0 })
  -- Condition 3: The complement of A in U is [-1, -n]
  (hCUA : CUA = { x : ℝ | x ∈ U ∧ x ∉ A } ∧ CUA = Icc (-1) (-n))

theorem find_m_plus_n : m + n = -2 :=
  sorry 

end find_m_plus_n_l253_253029


namespace second_frog_hops_l253_253181

theorem second_frog_hops (x : ℕ) :
  let first_frog_hops := 8 * x,
      second_frog_hops := 2 * x,
      third_frog_hops := x,
      total_hops := first_frog_hops + second_frog_hops + third_frog_hops in
  total_hops = 99 → second_frog_hops = 18 :=
by
  intro h
  rw [←Nat.mul_assoc, ←add_assoc, add_comm (2 * x) x, add_assoc, ←two_mul, add_assoc] at h
  have : 11 * x = 99 := by simp [h]
  calc
    2 * x = 2 * 9 : by rw [←Nat.div_eq_self (by simp [h]),_nat_cast_mul_cancel"],
    2 * 9 = 18 : by norm_num

#check second_frog_hops

end second_frog_hops_l253_253181


namespace teddy_bears_per_shelf_l253_253307

theorem teddy_bears_per_shelf :
  (98 / 14 = 7) := 
by
  sorry

end teddy_bears_per_shelf_l253_253307


namespace rate_per_square_meter_is_3_l253_253319

def floor_painting_rate 
  (length : ℝ) 
  (total_cost : ℝ)
  (length_more_than_breadth_by_percentage : ℝ)
  (expected_rate : ℝ) : Prop :=
  ∃ (breadth : ℝ) (rate : ℝ),
    length = (1 + length_more_than_breadth_by_percentage / 100) * breadth ∧
    total_cost = length * breadth * rate ∧
    rate = expected_rate

-- Given conditions
theorem rate_per_square_meter_is_3 :
  floor_painting_rate 15.491933384829668 240 200 3 :=
by
  sorry

end rate_per_square_meter_is_3_l253_253319


namespace child_stops_incur_yearly_cost_at_age_18_l253_253289

def john_contribution (years: ℕ) (cost_per_year: ℕ) : ℕ :=
  years * cost_per_year / 2

def university_contribution (university_cost: ℕ) : ℕ :=
  university_cost / 2

def total_contribution (years_after_8: ℕ) : ℕ :=
  john_contribution 8 10000 +
  john_contribution years_after_8 20000 +
  university_contribution 250000

theorem child_stops_incur_yearly_cost_at_age_18 :
  (total_contribution n = 265000) → (n + 8 = 18) :=
by
  sorry

end child_stops_incur_yearly_cost_at_age_18_l253_253289


namespace infinite_series_sum_eq_3_over_4_l253_253849

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253849


namespace algebra_books_needed_l253_253358

theorem algebra_books_needed (A' H' S' M' E' : ℕ) (x y : ℝ) (z : ℝ)
  (h1 : y > x)
  (h2 : A' ≠ H' ∧ A' ≠ S' ∧ A' ≠ M' ∧ A' ≠ E' ∧ H' ≠ S' ∧ H' ≠ M' ∧ H' ≠ E' ∧ S' ≠ M' ∧ S' ≠ E' ∧ M' ≠ E')
  (h3 : A' * x + H' * y = z)
  (h4 : S' * x + M' * y = z)
  (h5 : E' * x = 2 * z) :
  E' = (2 * A' * M' - 2 * S' * H') / (M' - H') :=
by
  sorry

end algebra_books_needed_l253_253358


namespace coefficient_comparison_expansion_l253_253579

theorem coefficient_comparison_expansion (n : ℕ) (h₁ : 2 * n * (n - 1) = 14 * n) : n = 8 :=
by
  sorry

end coefficient_comparison_expansion_l253_253579


namespace series_sum_correct_l253_253789

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l253_253789


namespace infinite_series_sum_eq_3_div_4_l253_253801

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253801


namespace find_k_l253_253923

theorem find_k 
  (k : ℝ) 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 3 * x + 5)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 7) 
  (intersection : ∃ x y : ℝ, (y = 3 * x + 5) ∧ (y = k * x - 7) ∧ x = -4 ∧ y = -7) :
  k = 0 :=
by
  sorry

end find_k_l253_253923


namespace pieces_per_serving_l253_253030

-- Definitions based on conditions
def jaredPopcorn : Nat := 90
def friendPopcorn : Nat := 60
def numberOfFriends : Nat := 3
def totalServings : Nat := 9

-- Statement to verify
theorem pieces_per_serving : 
  ((jaredPopcorn + numberOfFriends * friendPopcorn) / totalServings) = 30 :=
by
  sorry

end pieces_per_serving_l253_253030


namespace series_result_l253_253841

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253841


namespace johns_number_is_1500_l253_253594

def is_multiple_of (a b : Nat) : Prop := ∃ k, a = k * b

theorem johns_number_is_1500 (n : ℕ) (h1 : is_multiple_of n 125) (h2 : is_multiple_of n 30) (h3 : 1000 ≤ n ∧ n ≤ 3000) : n = 1500 :=
by
  -- proof structure goes here
  sorry

end johns_number_is_1500_l253_253594


namespace infinite_series_sum_eq_3_div_4_l253_253804

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253804


namespace solve_system_l253_253464

theorem solve_system : 
  ∀ (a b c : ℝ), 
  (a * (b^2 + c) = c * (c + a * b) ∧ 
   b * (c^2 + a) = a * (a + b * c) ∧ 
   c * (a^2 + b) = b * (b + c * a)) 
   → (∃ t : ℝ, a = t ∧ b = t ∧ c = t) :=
by
  intros a b c h
  sorry

end solve_system_l253_253464


namespace expected_ones_on_three_dice_l253_253965

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l253_253965


namespace exists_m_n_for_d_l253_253459

theorem exists_m_n_for_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) := 
sorry

end exists_m_n_for_d_l253_253459


namespace pascal_third_number_in_51_row_l253_253979

-- Definition and conditions
def pascal_row_num := 50
def third_number_index := 2

-- Statement of the problem
theorem pascal_third_number_in_51_row : 
  (nat.choose pascal_row_num third_number_index) = 1225 :=
by {
  -- The proof step will be skipped using sorry
  sorry
}

end pascal_third_number_in_51_row_l253_253979


namespace sara_caught_five_trout_l253_253623

theorem sara_caught_five_trout (S M : ℕ) (h1 : M = 2 * S) (h2 : M = 10) : S = 5 :=
by
  sorry

end sara_caught_five_trout_l253_253623


namespace probability_largest_value_five_l253_253991

open ProbabilityTheory

theorem probability_largest_value_five :
  let cards := Set.range (6 : ℕ)
  let events := {s : Set ℕ | s ⊆ cards ∧ s.card = 3}
  let largest_five := {s : Set ℕ | s ≠ ∅ ∧ s.max' (Finset.singleton_nonempty s) = 5}
  P (events ∩ largest_five) = 3 / 10 := by
  sorry

end probability_largest_value_five_l253_253991


namespace subtract_rational_from_zero_yields_additive_inverse_l253_253629

theorem subtract_rational_from_zero_yields_additive_inverse (a : ℚ) : 0 - a = -a := by
  sorry

end subtract_rational_from_zero_yields_additive_inverse_l253_253629


namespace series_converges_to_three_fourths_l253_253700

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253700


namespace perfect_squares_between_50_and_500_l253_253574

theorem perfect_squares_between_50_and_500 : 
  let n := 8 in
  let m := 22 in
  (∀ k, n ≤ k ∧ k ≤ m → (50 ≤ k^2 ∧ k^2 ≤ 500)) → (m - n + 1 = 15) := 
by
  let n := 8
  let m := 22
  assume h
  sorry

end perfect_squares_between_50_and_500_l253_253574


namespace is_quadratic_l253_253983

theorem is_quadratic (A B C D : Prop) :
  (A = (∀ x : ℝ, x + (1 / x) = 0)) ∧
  (B = (∀ x y : ℝ, x + x * y + 1 = 0)) ∧
  (C = (∀ x : ℝ, 3 * x + 2 = 0)) ∧
  (D = (∀ x : ℝ, x^2 + 2 * x = 1)) →
  D := 
by
  sorry

end is_quadratic_l253_253983


namespace total_baseball_cards_l253_253444

theorem total_baseball_cards (Carlos Matias Jorge : ℕ) (h1 : Carlos = 20) (h2 : Matias = Carlos - 6) (h3 : Jorge = Matias) : Carlos + Matias + Jorge = 48 :=
by
  sorry

end total_baseball_cards_l253_253444


namespace wonderland_cities_l253_253424

theorem wonderland_cities (V E B : ℕ) (hE : E = 45) (hB : B = 42) (h_connected : connected_graph) (h_simple : simple_graph) (h_bridges : count_bridges = 42) : V = 45 :=
sorry

end wonderland_cities_l253_253424


namespace range_of_k_for_real_roots_l253_253128

theorem range_of_k_for_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by 
  sorry

end range_of_k_for_real_roots_l253_253128


namespace three_digit_number_is_275_l253_253361

noncomputable def digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100 % 10, n / 10 % 10, n % 10)

theorem three_digit_number_is_275 :
  ∃ (n : ℕ), n / 100 % 10 + n % 10 = n / 10 % 10 ∧
              7 * (n / 100 % 10) = n % 10 + n / 10 % 10 + 2 ∧
              n / 100 % 10 + n / 10 % 10 + n % 10 = 14 ∧
              n = 275 :=
by
  sorry

end three_digit_number_is_275_l253_253361


namespace polynomial_product_l253_253343

theorem polynomial_product (x : ℝ) : (x - 1) * (x + 3) * (x + 5) = x^3 + 7*x^2 + 7*x - 15 :=
by
  sorry

end polynomial_product_l253_253343


namespace tan_135_eq_neg1_l253_253544

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 :=
by
  have h1 : 135 * Real.pi / 180 = Real.pi - Real.pi / 4 := by norm_num
  rw [h1, Real.tan_sub_pi_div_two]
  norm_num
  sorry

end tan_135_eq_neg1_l253_253544


namespace option_A_option_C_l253_253553

variable {a : ℕ → ℝ} (q : ℝ)
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n + 1) = q * (a n)

def decreasing_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n > a (n + 1)

theorem option_A (h₁ : a 1 > 0) (hq : geometric_sequence a q) : 0 < q ∧ q < 1 → decreasing_sequence a := 
  sorry

theorem option_C (h₁ : a 1 < 0) (hq : geometric_sequence a q) : q > 1 → decreasing_sequence a := 
  sorry

end option_A_option_C_l253_253553


namespace total_fish_caught_l253_253073

-- Definitions based on conditions
def brenden_morning_fish := 8
def brenden_fish_thrown_back := 3
def brenden_afternoon_fish := 5
def dad_fish := 13

-- Theorem representing the main question and its answer
theorem total_fish_caught : 
  (brenden_morning_fish + brenden_afternoon_fish - brenden_fish_thrown_back) + dad_fish = 23 :=
by
  sorry -- Proof goes here

end total_fish_caught_l253_253073


namespace remainder_n_sq_plus_3n_5_mod_25_l253_253901

theorem remainder_n_sq_plus_3n_5_mod_25 (k : ℤ) (n : ℤ) (h : n = 25 * k - 1) : 
  (n^2 + 3 * n + 5) % 25 = 3 := 
by
  sorry

end remainder_n_sq_plus_3n_5_mod_25_l253_253901


namespace center_of_circle_l253_253170

theorem center_of_circle : ∃ c : ℝ × ℝ, (∀ x y : ℝ, (x^2 + y^2 - 2*x + 4*y + 3 = 0 ↔ ((x - c.1)^2 + (y + c.2)^2 = 2))) ∧ (c = (1, -2)) :=
by
  -- Proof is omitted
  sorry

end center_of_circle_l253_253170


namespace intersection_of_A_and_B_l253_253110

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x)}
def B : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x ≤ 4} :=
sorry

end intersection_of_A_and_B_l253_253110


namespace coefficient_of_x2_in_expansion_l253_253240

def binomial_coefficient (n k : Nat) : Nat := Nat.choose k n

def binomial_term (a x : ℕ) (n r : ℕ) : ℕ :=
  a^(n-r) * binomial_coefficient n r * x^r

theorem coefficient_of_x2_in_expansion : 
  binomial_term 2 1 5 2 = 80 := by sorry

end coefficient_of_x2_in_expansion_l253_253240


namespace quadratic_real_roots_l253_253131

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_l253_253131


namespace opens_door_on_third_attempt_l253_253057

def probability_opens_door_on_third_attempt (keys : List ℕ) (correct_key : ℕ) : ℕ → ℝ :=
  sorry

noncomputable def solution : ℝ :=
  0.2

theorem opens_door_on_third_attempt :
  ∀ (keys : List ℕ) (correct_key : ℕ), 
    (keys.length = 5) →
    (List.mem correct_key keys) →
    (probability_opens_door_on_third_attempt keys correct_key 3 = solution) :=
by
  intros keys correct_key hlength hmem
  sorry

end opens_door_on_third_attempt_l253_253057


namespace infinite_series_sum_eq_3_over_4_l253_253853

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l253_253853


namespace weighted_averages_correct_l253_253908

def group_A_boys : ℕ := 20
def group_B_boys : ℕ := 25
def group_C_boys : ℕ := 15

def group_A_weight : ℝ := 50.25
def group_B_weight : ℝ := 45.15
def group_C_weight : ℝ := 55.20

def group_A_height : ℝ := 160
def group_B_height : ℝ := 150
def group_C_height : ℝ := 165

def group_A_age : ℝ := 15
def group_B_age : ℝ := 14
def group_C_age : ℝ := 16

def group_A_athletic : ℝ := 0.60
def group_B_athletic : ℝ := 0.40
def group_C_athletic : ℝ := 0.75

noncomputable def total_boys : ℕ := group_A_boys + group_B_boys + group_C_boys

noncomputable def weighted_average_height : ℝ := 
    (group_A_boys * group_A_height + group_B_boys * group_B_height + group_C_boys * group_C_height) / total_boys

noncomputable def weighted_average_weight : ℝ := 
    (group_A_boys * group_A_weight + group_B_boys * group_B_weight + group_C_boys * group_C_weight) / total_boys

noncomputable def weighted_average_age : ℝ := 
    (group_A_boys * group_A_age + group_B_boys * group_B_age + group_C_boys * group_C_age) / total_boys

noncomputable def weighted_average_athletic : ℝ := 
    (group_A_boys * group_A_athletic + group_B_boys * group_B_athletic + group_C_boys * group_C_athletic) / total_boys

theorem weighted_averages_correct :
  weighted_average_height = 157.08 ∧
  weighted_average_weight = 49.36 ∧
  weighted_average_age = 14.83 ∧
  weighted_average_athletic = 0.5542 := 
  by
    sorry

end weighted_averages_correct_l253_253908


namespace total_red_beads_l253_253639

theorem total_red_beads (total_beads : ℕ) (pattern_length : ℕ) (green_beads : ℕ) (red_beads : ℕ) (yellow_beads : ℕ) 
                         (h_total: total_beads = 85) 
                         (h_pattern: pattern_length = green_beads + red_beads + yellow_beads) 
                         (h_cycle: green_beads = 3 ∧ red_beads = 4 ∧ yellow_beads = 1) : 
                         (red_beads * (total_beads / pattern_length)) + (min red_beads (total_beads % pattern_length)) = 42 :=
by
  sorry

end total_red_beads_l253_253639


namespace sum_series_div_3_powers_l253_253757

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l253_253757


namespace twenty_four_game_solution_l253_253137

theorem twenty_four_game_solution :
  let a := 4
  let b := 8
  (a - (b / b)) * b = 24 :=
by
  let a := 4
  let b := 8
  show (a - (b / b)) * b = 24
  sorry

end twenty_four_game_solution_l253_253137


namespace find_sum_f_neg1_f_3_l253_253261

noncomputable def f : ℝ → ℝ := sorry

-- condition: odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f x

-- condition: symmetry around x=1
def symmetric_around_one (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (1 - x) = f (1 + x)

-- condition: specific value at x=1
def value_at_one (f : ℝ → ℝ) : Prop := f 1 = 2

-- Theorem to prove
theorem find_sum_f_neg1_f_3 (h1 : odd_function f) (h2 : symmetric_around_one f) (h3 : value_at_one f) : f (-1) + f 3 = -4 := by
  sorry

end find_sum_f_neg1_f_3_l253_253261


namespace solve_for_y_l253_253265

theorem solve_for_y (x y : ℝ) (h : 5 * x + 3 * y = 1) : y = (1 - 5 * x) / 3 :=
by
  sorry

end solve_for_y_l253_253265


namespace problem1_problem2_l253_253883

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + (a - 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem problem1 (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) → a = 2 ∨ a = 3 := sorry

theorem problem2 (m : ℝ) : (∀ x, x ∈ A → x ∈ C m) → m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) := sorry

end problem1_problem2_l253_253883


namespace series_sum_eq_l253_253728

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253728


namespace sum_series_eq_3_div_4_l253_253825

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253825


namespace janet_additional_money_needed_l253_253442

def janet_savings : ℕ := 2225
def monthly_rent : ℕ := 1250
def advance_months : ℕ := 2
def deposit : ℕ := 500

theorem janet_additional_money_needed :
  (advance_months * monthly_rent + deposit - janet_savings) = 775 :=
by
  sorry

end janet_additional_money_needed_l253_253442


namespace infinite_series_sum_value_l253_253810

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253810


namespace expected_ones_three_standard_dice_l253_253958

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l253_253958


namespace number_of_propositions_is_4_l253_253515

def is_proposition (s : String) : Prop :=
  s = "The Earth is a planet in the solar system" ∨ 
  s = "{0} ∈ ℕ" ∨ 
  s = "1+1 > 2" ∨ 
  s = "Elderly people form a set"

theorem number_of_propositions_is_4 : 
  (is_proposition "The Earth is a planet in the solar system" ∨ 
   is_proposition "{0} ∈ ℕ" ∨ 
   is_proposition "1+1 > 2" ∨ 
   is_proposition "Elderly people form a set") → 
  4 = 4 :=
by
  sorry

end number_of_propositions_is_4_l253_253515


namespace benny_spending_l253_253519

variable (S D V : ℝ)

theorem benny_spending :
  (200 - 45) = S + (D / 110) + (V / 0.75) :=
by
  sorry

end benny_spending_l253_253519


namespace gap_between_rails_should_be_12_24_mm_l253_253646

noncomputable def initial_length : ℝ := 15
noncomputable def temperature_initial : ℝ := -8
noncomputable def temperature_max : ℝ := 60
noncomputable def expansion_coefficient : ℝ := 0.000012
noncomputable def change_in_temperature : ℝ := temperature_max - temperature_initial
noncomputable def final_length : ℝ := initial_length * (1 + expansion_coefficient * change_in_temperature)
noncomputable def gap : ℝ := (final_length - initial_length) * 1000  -- converted to mm

theorem gap_between_rails_should_be_12_24_mm
  : gap = 12.24 := by
  sorry

end gap_between_rails_should_be_12_24_mm_l253_253646


namespace min_living_allowance_inequality_l253_253322

variable (x : ℝ)

-- The regulation stipulates that the minimum living allowance should not be less than 300 yuan.
def min_living_allowance_regulation (x : ℝ) : Prop := x >= 300

theorem min_living_allowance_inequality (x : ℝ) :
  min_living_allowance_regulation x ↔ x ≥ 300 := by
  sorry

end min_living_allowance_inequality_l253_253322


namespace passengers_on_third_plane_l253_253182

theorem passengers_on_third_plane (
  P : ℕ
) (h1 : 600 - 2 * 50 = 500) -- Speed of the first plane
  (h2 : 600 - 2 * 60 = 480) -- Speed of the second plane
  (h_avg : (500 + 480 + (600 - 2 * P)) / 3 = 500) -- Average speed condition
  : P = 40 := by sorry

end passengers_on_third_plane_l253_253182


namespace mode_and_median_are_8_l253_253342

open Finset

/-- A problem to verify that in a given dataset, the mode and median are both equal to 8. -/
theorem mode_and_median_are_8 :
  let data := [11, 9, 7, 8, 6, 8, 12, 8] in
  let mode := (data.filter(λ x, x = 8)).length in
  let sorted_data := sort data in
  let median_index := data.length / 2 in
  mode = 3 ∧ sorted_data.nth median_index = 8 :=
by
  sorry

end mode_and_median_are_8_l253_253342


namespace library_visitors_on_sundays_l253_253504

theorem library_visitors_on_sundays 
  (average_other_days : ℕ) 
  (average_per_day : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ) 
  (total_visitors_month : ℕ)
  (visitors_other_days : ℕ) 
  (total_visitors_sundays : ℕ) :
  average_other_days = 240 →
  average_per_day = 285 →
  total_days = 30 →
  sundays = 5 →
  other_days = total_days - sundays →
  total_visitors_month = average_per_day * total_days →
  visitors_other_days = average_other_days * other_days →
  total_visitors_sundays + visitors_other_days = total_visitors_month →
  total_visitors_sundays = sundays * (510 : ℕ) :=
by
  sorry


end library_visitors_on_sundays_l253_253504


namespace series_sum_l253_253742

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l253_253742


namespace expression_of_f_l253_253103

theorem expression_of_f (f : ℤ → ℤ) (h : ∀ x, f (x - 1) = x^2 + 4 * x - 5) : ∀ x, f x = x^2 + 6 * x :=
by
  sorry

end expression_of_f_l253_253103


namespace arithmetic_seq_properties_l253_253590

theorem arithmetic_seq_properties (a : ℕ → ℝ) (d a1 : ℝ) (S : ℕ → ℝ) :
  (a 1 + a 3 = 8) ∧ (a 4 ^ 2 = a 2 * a 9) →
  ((a1 = 4 ∧ d = 0 ∧ (∀ n, S n = 4 * n)) ∨
   (a1 = 1 ∧ d = 3 ∧ (∀ n, S n = (3 * n^2 - n) / 2))) := 
sorry

end arithmetic_seq_properties_l253_253590


namespace find_a11_times_a55_l253_253365

noncomputable def a_ij (i j : ℕ) : ℝ := 
  if i = 4 ∧ j = 1 then -2 else
  if i = 4 ∧ j = 3 then 10 else
  if i = 2 ∧ j = 4 then 4 else sorry

theorem find_a11_times_a55 
  (arithmetic_first_row : ∀ j, a_ij 1 (j + 1) = a_ij 1 1 + (j * 6))
  (geometric_columns : ∀ i j, a_ij (i + 1) j = a_ij 1 j * (2 ^ i) ∨ a_ij (i + 1) j = a_ij 1 j * ((-2) ^ i))
  (a24_eq_4 : a_ij 2 4 = 4)
  (a41_eq_neg2 : a_ij 4 1 = -2)
  (a43_eq_10 : a_ij 4 3 = 10) :
  a_ij 1 1 * a_ij 5 5 = -11 :=
by sorry

end find_a11_times_a55_l253_253365


namespace initial_bananas_per_child_l253_253616

theorem initial_bananas_per_child (B x : ℕ) (total_children : ℕ := 780) (absent_children : ℕ := 390) :
  390 * (x + 2) = total_children * x → x = 2 :=
by
  intros h
  sorry

end initial_bananas_per_child_l253_253616


namespace max_value_of_expr_l253_253976

noncomputable def max_value (t : ℕ) : ℝ := (3^t - 2*t)*t / 9^t

theorem max_value_of_expr :
  ∃ t : ℕ, max_value t = 1 / 8 :=
sorry

end max_value_of_expr_l253_253976


namespace investment_interests_l253_253466

theorem investment_interests (x y : ℝ) (h₁ : x + y = 24000)
  (h₂ : 0.045 * x + 0.06 * y = 0.05 * 24000) : (x = 16000) ∧ (y = 8000) :=
  by
  sorry

end investment_interests_l253_253466


namespace sum_series_equals_three_fourths_l253_253824

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l253_253824


namespace inequality_proof_l253_253167

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) :
  (x / (y + 1) + y / (x + 1)) ≥ (2 / 3) ∧ (x = 1 / 2 ∧ y = 1 / 2 → x / (y + 1) + y / (x + 1) = 2 / 3) := by
  sorry

end inequality_proof_l253_253167


namespace sum_of_perimeters_of_squares_l253_253631

theorem sum_of_perimeters_of_squares (x : ℝ) (h₁ : x = 3) :
  let area1 := x^2 + 4 * x + 4
  let area2 := 4 * x^2 - 12 * x + 9
  let side1 := Real.sqrt area1
  let side2 := Real.sqrt area2
  let perim1 := 4 * side1
  let perim2 := 4 * side2
  perim1 + perim2 = 32 :=
by
  sorry

end sum_of_perimeters_of_squares_l253_253631


namespace accommodation_ways_l253_253666

-- Definition of the problem
def triple_room_count : ℕ := 1
def double_room_count : ℕ := 2
def adults_count : ℕ := 3
def children_count : ℕ := 2
def total_ways : ℕ := 60

-- Main statement to be proved
theorem accommodation_ways :
  (triple_room_count = 1) →
  (double_room_count = 2) →
  (adults_count = 3) →
  (children_count = 2) →
  -- Children must be accompanied by adults, and not all rooms need to be occupied.
  -- We are to prove that the number of valid ways to assign the rooms is 60
  total_ways = 60 :=
by sorry

end accommodation_ways_l253_253666


namespace simplify_and_evaluate_expression_l253_253626

theorem simplify_and_evaluate_expression (x : ℝ) (hx : x = 6) :
  (1 + (2 / (x + 1))) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end simplify_and_evaluate_expression_l253_253626


namespace distance_of_point_P_to_base_AB_l253_253972

theorem distance_of_point_P_to_base_AB :
  ∀ (P : ℝ) (A B C : ℝ → ℝ)
    (h : ∀ (x : ℝ), A x = B x)
    (altitude : ℝ)
    (area_ratio : ℝ),
  altitude = 6 →
  area_ratio = 1 / 3 →
  (∃ d : ℝ, d = 6 - (2 / 3) * 6 ∧ d = 2) := 
  sorry

end distance_of_point_P_to_base_AB_l253_253972


namespace percentage_assigned_exam_l253_253279

-- Define the conditions of the problem
def total_students : ℕ := 100
def average_assigned : ℝ := 0.55
def average_makeup : ℝ := 0.95
def average_total : ℝ := 0.67

-- Define the proof problem statement
theorem percentage_assigned_exam :
  ∃ (x : ℝ), (x / total_students) * average_assigned + ((total_students - x) / total_students) * average_makeup = average_total ∧ x = 70 :=
by
  sorry

end percentage_assigned_exam_l253_253279


namespace flowers_total_l253_253069

def red_roses := 1491
def yellow_carnations := 3025
def white_roses := 1768
def purple_tulips := 2150
def pink_daisies := 3500
def blue_irises := 2973
def orange_marigolds := 4234
def lavender_orchids := 350
def sunflowers := 815
def violet_lilies := 26

theorem flowers_total :
  red_roses +
  yellow_carnations +
  white_roses +
  purple_tulips +
  pink_daisies +
  blue_irises +
  orange_marigolds +
  lavender_orchids +
  sunflowers +
  violet_lilies = 21332 := 
by
  -- Simplify and add up all given numbers
  sorry

end flowers_total_l253_253069


namespace total_fish_caught_l253_253072

-- Definitions based on conditions
def brenden_morning_fish := 8
def brenden_fish_thrown_back := 3
def brenden_afternoon_fish := 5
def dad_fish := 13

-- Theorem representing the main question and its answer
theorem total_fish_caught : 
  (brenden_morning_fish + brenden_afternoon_fish - brenden_fish_thrown_back) + dad_fish = 23 :=
by
  sorry -- Proof goes here

end total_fish_caught_l253_253072


namespace infinite_series_sum_eq_l253_253719

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l253_253719


namespace area_of_triangle_l253_253932

theorem area_of_triangle {a b c : ℝ} (S : ℝ) (h1 : (a^2) * (Real.sin C) = 4 * (Real.sin A))
                          (h2 : (a + c)^2 = 12 + b^2)
                          (h3 : S = Real.sqrt ((1/4) * (a^2 * c^2 - ( (a^2 + c^2 - b^2)/2 )^2))) :
  S = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l253_253932


namespace carnival_earnings_l253_253175

theorem carnival_earnings (days : ℕ) (total_earnings : ℕ) (h1 : days = 22) (h2 : total_earnings = 3168) : 
  (total_earnings / days) = 144 := 
by
  -- The proof would go here
  sorry

end carnival_earnings_l253_253175


namespace nth_equation_proof_l253_253455

theorem nth_equation_proof (n : ℕ) (h : n ≥ 1) :
  1 / (n + 1 : ℚ) + 1 / (n * (n + 1)) = 1 / n := 
sorry

end nth_equation_proof_l253_253455


namespace factorization_problem_l253_253649

theorem factorization_problem (a b c x : ℝ) :
  ¬(2 * a^2 - b^2 = (a + b) * (a - b) + a^2) ∧
  ¬(2 * a * (b + c) = 2 * a * b + 2 * a * c) ∧
  (x^3 - 2 * x^2 + x = x * (x - 1)^2) ∧
  ¬ (x^2 + x = x^2 * (1 + 1 / x)) :=
by
  sorry

end factorization_problem_l253_253649


namespace sum_of_integers_k_sum_of_all_integers_k_l253_253094
open Nat

theorem sum_of_integers_k (k : ℕ) (h : choose 25 5 + choose 25 6 = choose 26 k) : k = 6 ∨ k = 20 :=
begin
  sorry,
end

theorem sum_of_all_integers_k : 
  (∃ k, (choose 25 5 + choose 25 6 = choose 26 k) → k = 6 ∨ k = 20) → 6 + 20 = 26 :=
begin
  sorry,
end

end sum_of_integers_k_sum_of_all_integers_k_l253_253094


namespace series_sum_eq_l253_253733

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l253_253733


namespace sequence_general_formula_l253_253881

noncomputable def sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2^(n-2)

theorem sequence_general_formula {a : ℕ → ℝ} {S : ℕ → ℝ} (hpos : ∀ n, a n > 0)
  (hSn : ∀ n, 2 * a n = S n + 0.5) : ∀ n, a n = sequence_formula a S n :=
by 
  sorry

end sequence_general_formula_l253_253881


namespace infinite_series_sum_value_l253_253814

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l253_253814


namespace average_height_of_trees_l253_253216

theorem average_height_of_trees :
  ∃ (h : ℕ → ℕ), (h 2 = 12) ∧ (∀ i, h i = 2 * h (i+1) ∨ h i = h (i+1) / 2) ∧ (h 1 * h 2 * h 3 * h 4 * h 5 * h 6 = 4608) →
  (h 1 + h 2 + h 3 + h 4 + h 5 + h 6) / 6 = 21 :=
sorry

end average_height_of_trees_l253_253216


namespace preservation_time_at_33_degrees_l253_253321

noncomputable def preservation_time (x : ℝ) (k : ℝ) (b : ℝ) : ℝ :=
  Real.exp (k * x + b)

theorem preservation_time_at_33_degrees (k b : ℝ) 
  (h1 : Real.exp b = 192)
  (h2 : Real.exp (22 * k + b) = 48) :
  preservation_time 33 k b = 24 := by
  sorry

end preservation_time_at_33_degrees_l253_253321


namespace probability_female_wears_glasses_l253_253905

def prob_female_wears_glasses (total_females : ℕ) (females_no_glasses : ℕ) : ℚ :=
  let females_with_glasses := total_females - females_no_glasses
  females_with_glasses / total_females

theorem probability_female_wears_glasses :
  prob_female_wears_glasses 18 8 = 5 / 9 := by
  sorry  -- Proof is skipped

end probability_female_wears_glasses_l253_253905


namespace min_value_of_exponential_l253_253565

theorem min_value_of_exponential (x y : ℝ) (h : x + 2 * y = 3) : 2^x + 4^y = 4 * Real.sqrt 2 := by
  sorry

end min_value_of_exponential_l253_253565


namespace tan_135_eq_neg1_l253_253537

theorem tan_135_eq_neg1 :
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in
  Q = -(Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * Complex.I →
  Complex.tan (135 * Real.pi / 180 * Complex.I) = -1 :=
by
  intro hQ
  have Q_coords : Q = -(Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * Complex.I := hQ
  sorry

end tan_135_eq_neg1_l253_253537


namespace number_of_bushes_needed_l253_253924

-- Definitions from the conditions
def containers_per_bush : ℕ := 10
def containers_per_zucchini : ℕ := 3
def zucchinis_required : ℕ := 72

-- Statement to prove
theorem number_of_bushes_needed : 
  ∃ bushes_needed : ℕ, bushes_needed = 22 ∧ 
  (zucchinis_required * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush = bushes_needed := 
by
  sorry

end number_of_bushes_needed_l253_253924


namespace huahuan_initial_cards_l253_253658

theorem huahuan_initial_cards
  (a b c : ℕ) -- let a, b, c be the initial number of cards Huahuan, Yingying, and Nini have
  (total : a + b + c = 2712)
  (condition_after_50_rounds : ∃ d, b = a + d ∧ c = a + 2 * d) -- after 50 rounds, form an arithmetic sequence
  : a = 754 := sorry

end huahuan_initial_cards_l253_253658


namespace sum_geometric_series_l253_253714

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253714


namespace remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l253_253389

def p (x : ℝ) : ℝ := x^3 - 4 * x^2 + 3 * x + 2

theorem remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1 :
  p 1 = 2 := by
  -- solution needed, for now we put a placeholder
  sorry

end remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l253_253389


namespace sum_series_eq_3_div_4_l253_253831

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l253_253831


namespace sum_of_first_six_terms_of_geom_seq_l253_253876

theorem sum_of_first_six_terms_of_geom_seq :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let S6 := a * (1 - r^6) / (1 - r)
  S6 = 4095 / 12288 := by
sorry

end sum_of_first_six_terms_of_geom_seq_l253_253876


namespace infinite_triangles_with_sides_x_y_10_l253_253563

theorem infinite_triangles_with_sides_x_y_10 (x y : Nat) (hx : 0 < x) (hy : 0 < y) : 
  (∃ n : Nat, n > 5 ∧ ∀ m ≥ n, ∃ x y : Nat, 0 < x ∧ 0 < y ∧ x + y > 10 ∧ x + 10 > y ∧ y + 10 > x) :=
sorry

end infinite_triangles_with_sides_x_y_10_l253_253563


namespace combined_perimeter_of_squares_l253_253595

theorem combined_perimeter_of_squares (p1 p2 : ℝ) (s1 s2 : ℝ) :
  p1 = 40 → p2 = 100 → 4 * s1 = p1 → 4 * s2 = p2 →
  (p1 + p2 - 2 * s1) = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end combined_perimeter_of_squares_l253_253595


namespace expected_ones_three_dice_l253_253971

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l253_253971


namespace series_converges_to_three_fourths_l253_253695

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253695


namespace replace_all_cardio_machines_cost_l253_253056

noncomputable def totalReplacementCost : ℕ :=
  let numGyms := 20
  let bikesPerGym := 10
  let treadmillsPerGym := 5
  let ellipticalsPerGym := 5
  let costPerBike := 700
  let costPerTreadmill := costPerBike * 3 / 2
  let costPerElliptical := costPerTreadmill * 2
  let totalBikes := numGyms * bikesPerGym
  let totalTreadmills := numGyms * treadmillsPerGym
  let totalEllipticals := numGyms * ellipticalsPerGym
  (totalBikes * costPerBike) + (totalTreadmills * costPerTreadmill) + (totalEllipticals * costPerElliptical)

theorem replace_all_cardio_machines_cost :
  totalReplacementCost = 455000 :=
by
  -- All the calculation steps provided as conditions and intermediary results need to be verified here.
  sorry

end replace_all_cardio_machines_cost_l253_253056


namespace brian_breath_proof_l253_253370

def breath_holding_time (initial_time: ℕ) (week1_factor: ℝ) (week2_factor: ℝ) 
  (missed_days: ℕ) (missed_decrease: ℝ) (week3_factor: ℝ): ℝ := by
  let week1_time := initial_time * week1_factor
  let hypothetical_week2_time := week1_time * (1 + week2_factor)
  let missed_decrease_total := week1_time * missed_decrease * missed_days
  let effective_week2_time := hypothetical_week2_time - missed_decrease_total
  let final_time := effective_week2_time * (1 + week3_factor)
  exact final_time

theorem brian_breath_proof :
  breath_holding_time 10 2 0.75 2 0.1 0.5 = 46.5 := 
by
  sorry

end brian_breath_proof_l253_253370


namespace stocks_closed_higher_l253_253206

-- Definition of the conditions:
def stocks : Nat := 1980
def increased (H L : Nat) : Prop := H = (1.20 : ℝ) * L
def total_stocks (H L : Nat) : Prop := H + L = stocks

-- Claim to prove
theorem stocks_closed_higher (H L : Nat) (h1 : increased H L) (h2 : total_stocks H L) : H = 1080 :=
by
  sorry

end stocks_closed_higher_l253_253206


namespace part1_part2_l253_253560

-- Problem 1: Given |x| = 9, |y| = 5, x < 0, y > 0, prove x + y = -4
theorem part1 (x y : ℚ) (h1 : |x| = 9) (h2 : |y| = 5) (h3 : x < 0) (h4 : y > 0) : x + y = -4 :=
sorry

-- Problem 2: Given |x| = 9, |y| = 5, |x + y| = x + y, prove x - y = 4 or x - y = 14
theorem part2 (x y : ℚ) (h1 : |x| = 9) (h2 : |y| = 5) (h3 : |x + y| = x + y) : x - y = 4 ∨ x - y = 14 :=
sorry

end part1_part2_l253_253560


namespace milk_left_l253_253152

theorem milk_left (initial_milk : ℝ) (given_milk : ℝ) : initial_milk = 5 ∧ given_milk = 18/7 → (initial_milk - given_milk = 17/7) :=
by
  assume h
  cases h with h_initial h_given
  rw [h_initial, h_given]
  norm_num
  sorry

end milk_left_l253_253152


namespace symmetric_point_origin_l253_253311

-- Define the point P
structure Point3D where
  x : Int
  y : Int
  z : Int

def P : Point3D := { x := 1, y := 3, z := -5 }

-- Define the symmetric function w.r.t. the origin
def symmetric_with_origin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Define the expected result
def Q : Point3D := { x := -1, y := -3, z := 5 }

-- The theorem to prove
theorem symmetric_point_origin : symmetric_with_origin P = Q := by
  sorry

end symmetric_point_origin_l253_253311


namespace series_converges_to_three_fourths_l253_253704

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253704


namespace three_digit_number_count_correct_l253_253266

noncomputable
def count_three_digit_numbers (digits : List ℕ) : ℕ :=
  if h : digits.length = 5 then
    (5 * 4 * 3 : ℕ)
  else
    0

theorem three_digit_number_count_correct :
  count_three_digit_numbers [1, 3, 5, 7, 9] = 60 :=
by
  unfold count_three_digit_numbers
  simp only [List.length, if_pos]
  rfl

end three_digit_number_count_correct_l253_253266


namespace infinite_series_sum_eq_3_div_4_l253_253796

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l253_253796


namespace sum_k_over_3_pow_k_eq_three_fourths_l253_253858

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l253_253858


namespace sum_geometric_series_l253_253709

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l253_253709


namespace area_of_absolute_value_sum_l253_253691

theorem area_of_absolute_value_sum :
  ∃ area : ℝ, (area = 80) ∧ (∀ x y : ℝ, |2 * x| + |5 * y| = 20 → area = 80) :=
by
  sorry

end area_of_absolute_value_sum_l253_253691


namespace two_legged_birds_count_l253_253682

def count_birds (b m i : ℕ) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 6 * i = 680 → b = 280

theorem two_legged_birds_count : ∃ b m i : ℕ, count_birds b m i :=
by
  have h1 : count_birds 280 0 20 := sorry
  exact ⟨280, 0, 20, h1⟩

end two_legged_birds_count_l253_253682


namespace simplify_expression_l253_253016

theorem simplify_expression (x y : ℝ) (m : ℤ) : 
  ((x + y)^(2 * m + 1) / (x + y)^(m - 1) = (x + y)^(m + 2)) :=
by sorry

end simplify_expression_l253_253016


namespace student_weighted_avg_larger_l253_253062

variable {u v w : ℚ}

theorem student_weighted_avg_larger (h1 : u < v) (h2 : v < w) :
  (4 * u + 6 * v + 20 * w) / 30 > (2 * u + 3 * v + 4 * w) / 9 := by
  sorry

end student_weighted_avg_larger_l253_253062


namespace find_k_l253_253995

theorem find_k (k : ℝ) : 
  (k - 10) / (-8) = (5 - k) / (-8) → k = 7.5 :=
by
  intro h
  let slope1 := (k - 10) / (-8)
  let slope2 := (5 - k) / (-8)
  have h_eq : slope1 = slope2 := h
  sorry

end find_k_l253_253995


namespace clock_palindromes_l253_253583

theorem clock_palindromes : 
  let valid_hours := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22]
  let valid_minutes := [0, 1, 2, 3, 4, 5]
  let two_digit_palindromes := 9 * 6
  let four_digit_palindromes := 6
  (two_digit_palindromes + four_digit_palindromes) = 60 := 
by
  sorry

end clock_palindromes_l253_253583


namespace min_red_chips_l253_253354

variable (w b r : ℕ)

theorem min_red_chips :
  (b ≥ w / 3) → (b ≤ r / 4) → (w + b ≥ 70) → r ≥ 72 :=
by
  sorry

end min_red_chips_l253_253354


namespace sum_of_series_eq_three_fourths_l253_253784

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l253_253784


namespace amount_left_after_spending_l253_253041

-- Define the initial amount and percentage spent
def initial_amount : ℝ := 500
def percentage_spent : ℝ := 0.30

-- Define the proof statement that the amount left is 350
theorem amount_left_after_spending : 
  (initial_amount - (percentage_spent * initial_amount)) = 350 :=
by
  sorry

end amount_left_after_spending_l253_253041


namespace largest_value_is_B_l253_253984

def exprA := 1 + 2 * 3 + 4
def exprB := 1 + 2 + 3 * 4
def exprC := 1 + 2 + 3 + 4
def exprD := 1 * 2 + 3 + 4
def exprE := 1 * 2 + 3 * 4

theorem largest_value_is_B : exprB = 15 ∧ exprB > exprA ∧ exprB > exprC ∧ exprB > exprD ∧ exprB > exprE := 
by
  sorry

end largest_value_is_B_l253_253984


namespace linear_equation_in_two_variables_l253_253078

def is_linear_equation_two_variables (eq : String → Prop) : Prop :=
  eq "D"

-- Given Conditions
def eqA (x y z : ℝ) : Prop := 2 * x + 3 * y = z
def eqB (x y : ℝ) : Prop := 4 / x + y = 5
def eqC (x y : ℝ) : Prop := 1 / 2 * x^2 + y = 0
def eqD (x y : ℝ) : Prop := y = 1 / 2 * (x + 8)

-- Problem Statement to be Proved
theorem linear_equation_in_two_variables :
  is_linear_equation_two_variables (λ s =>
    ∃ x y z : ℝ, 
      (s = "A" → eqA x y z) ∨ 
      (s = "B" → eqB x y) ∨ 
      (s = "C" → eqC x y) ∨ 
      (s = "D" → eqD x y)
  ) :=
sorry

end linear_equation_in_two_variables_l253_253078


namespace sum_of_slopes_eq_zero_l253_253256

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

end sum_of_slopes_eq_zero_l253_253256


namespace neg_3_14_gt_neg_pi_l253_253236

theorem neg_3_14_gt_neg_pi (π : ℝ) (h : 0 < π) : -3.14 > -π := 
sorry

end neg_3_14_gt_neg_pi_l253_253236


namespace range_of_a_l253_253571

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 < 0 → a < x) → a ≤ -1 :=
by
  sorry

end range_of_a_l253_253571


namespace series_converges_to_three_fourths_l253_253696

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l253_253696


namespace k_valid_iff_l253_253867

open Nat

theorem k_valid_iff (k : ℕ) :
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by
  sorry

end k_valid_iff_l253_253867


namespace ratio_of_sums_l253_253450

variable {α : Type*} [LinearOrderedField α] 

variable (a : ℕ → α) (S : ℕ → α)
variable (a1 d : α)

def isArithmeticSequence (a : ℕ → α) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + n * d

def sumArithmeticSequence (a : α) (d : α) (n : ℕ) : α :=
  n / 2 * (2 * a + (n - 1) * d)

theorem ratio_of_sums (h_arith : isArithmeticSequence a) (h_S : ∀ n, S n = sumArithmeticSequence a1 d n)
  (h_a5_5a3 : a 5 = 5 * a 3) : S 9 / S 5 = 9 := by sorry

end ratio_of_sums_l253_253450


namespace sculpture_and_base_height_l253_253075

def height_sculpture_ft : ℕ := 2
def height_sculpture_in : ℕ := 10
def height_base_in : ℕ := 2

def total_height_in (ft : ℕ) (inch1 inch2 : ℕ) : ℕ :=
  (ft * 12) + inch1 + inch2

def total_height_ft (total_in : ℕ) : ℕ :=
  total_in / 12

theorem sculpture_and_base_height :
  total_height_ft (total_height_in height_sculpture_ft height_sculpture_in height_base_in) = 3 :=
by
  sorry

end sculpture_and_base_height_l253_253075


namespace overlap_percentage_l253_253644

noncomputable def square_side_length : ℝ := 10
noncomputable def rectangle_length : ℝ := 18
noncomputable def rectangle_width : ℝ := square_side_length
noncomputable def overlap_length : ℝ := 2
noncomputable def overlap_width : ℝ := rectangle_width

noncomputable def rectangle_area : ℝ :=
  rectangle_length * rectangle_width

noncomputable def overlap_area : ℝ :=
  overlap_length * overlap_width

noncomputable def percentage_shaded : ℝ :=
  (overlap_area / rectangle_area) * 100

theorem overlap_percentage :
  percentage_shaded = 100 * (1 / 9) :=
sorry

end overlap_percentage_l253_253644


namespace total_earnings_correct_l253_253377

-- Given conditions
def charge_oil_change : ℕ := 20
def charge_repair : ℕ := 30
def charge_car_wash : ℕ := 5

def number_oil_changes : ℕ := 5
def number_repairs : ℕ := 10
def number_car_washes : ℕ := 15

-- Calculation of earnings based on the conditions
def earnings_from_oil_changes : ℕ := charge_oil_change * number_oil_changes
def earnings_from_repairs : ℕ := charge_repair * number_repairs
def earnings_from_car_washes : ℕ := charge_car_wash * number_car_washes

-- The total earnings
def total_earnings : ℕ := earnings_from_oil_changes + earnings_from_repairs + earnings_from_car_washes

-- Proof statement: Prove that the total earnings are $475
theorem total_earnings_correct : total_earnings = 475 := by -- our proof will go here
  sorry

end total_earnings_correct_l253_253377


namespace break_even_number_of_books_l253_253288

-- Definitions from conditions.
def fixed_cost : ℝ := 50000
def variable_cost_per_book : ℝ := 4
def selling_price_per_book : ℝ := 9

-- Main statement proving the break-even point.
theorem break_even_number_of_books 
  (x : ℕ) : (selling_price_per_book * x = fixed_cost + variable_cost_per_book * x) → (x = 10000) :=
by
  sorry

end break_even_number_of_books_l253_253288


namespace sneaker_final_price_l253_253999

-- Definitions of the conditions
def original_price : ℝ := 120
def coupon_value : ℝ := 10
def discount_percent : ℝ := 0.1

-- The price after the coupon is applied
def price_after_coupon := original_price - coupon_value

-- The membership discount amount
def membership_discount := price_after_coupon * discount_percent

-- The final price the man will pay
def final_price := price_after_coupon - membership_discount

theorem sneaker_final_price : final_price = 99 := by
  sorry

end sneaker_final_price_l253_253999


namespace series_result_l253_253842

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l253_253842


namespace lemon_juice_fraction_l253_253993

theorem lemon_juice_fraction :
  ∃ L : ℚ, 30 - 30 * L - (1 / 3) * (30 - 30 * L) = 6 ∧ L = 7 / 10 :=
sorry

end lemon_juice_fraction_l253_253993


namespace remainder_n_squared_plus_3n_plus_5_l253_253899

theorem remainder_n_squared_plus_3n_plus_5 (n : ℕ) (h : n % 25 = 24) : (n^2 + 3 * n + 5) % 25 = 3 :=
by
  sorry

end remainder_n_squared_plus_3n_plus_5_l253_253899


namespace sum_of_valid_k_equals_26_l253_253099

theorem sum_of_valid_k_equals_26 :
  (∑ k in Finset.filter (λ k => Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (Finset.range 27)) = 26 :=
by
  sorry

end sum_of_valid_k_equals_26_l253_253099


namespace sum_integers_75_to_95_l253_253488

theorem sum_integers_75_to_95 :
  let a := 75
  let l := 95
  let n := 95 - 75 + 1
  ∑ k in Finset.range n, (a + k) = 1785 := by
  sorry

end sum_integers_75_to_95_l253_253488


namespace part_I_part_II_l253_253885

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -4 ∨ x > -2}
def C (m : ℝ) : Set ℝ := {x | 3 - 2 * m ≤ x ∧ x ≤ 2 + m}
def D : Set ℝ := {y | y < -6 ∨ y > -5}

theorem part_I (m : ℝ) : (∀ x, x ∈ A ∧ x ∈ B → x ∈ C m) → m ≥ 5 / 2 :=
sorry

theorem part_II (m : ℝ) : 
  (B ∪ (C m) = Set.univ) ∧ 
  (C m ⊆ D) → 
  7 / 2 ≤ m ∧ m < 4 :=
sorry

end part_I_part_II_l253_253885


namespace sum_of_valid_k_equals_26_l253_253098

theorem sum_of_valid_k_equals_26 :
  (∑ k in Finset.filter (λ k => Nat.choose 25 5 + Nat.choose 25 6 = Nat.choose 26 k) (Finset.range 27)) = 26 :=
by
  sorry

end sum_of_valid_k_equals_26_l253_253098


namespace correct_answer_l253_253328

def coin_events : Finset (Finset Char) := 
  {{'H', 'H', 'H'}, {'H', 'H', 'T'}, {'H', 'T', 'H'}, {'H', 'T', 'T'}, {'T', 'H', 'H'}, {'T', 'H', 'T'}, {'T', 'T', 'H'}, {'T', 'T', 'T'}}

def at_least_one_heads (s : Finset Char) : Prop := ∃ h ∈ s, h = 'H'

def at_most_one_heads (s : Finset Char) : Prop := s.count('H') ≤ 1

def exactly_two_heads (s : Finset Char) : Prop := s.count('H') = 2

def at_least_two_heads (s : Finset Char) : Prop := s.count('H') ≥ 2

def at_most_two_heads (s : Finset Char) : Prop := s.count('H') ≤ 2

theorem correct_answer :
  ∃ s1 s2 : Finset Char,
  s1 ∈ coin_events ∧
  (at_most_one_heads s1 ∧ exactly_two_heads s2) ∧
  (s1 ∩ s2 = ∅) ∧ ¬(s1 ∪ s2 = coin_events) :=
sorry

end correct_answer_l253_253328


namespace intersection_A_B_l253_253416

def A : Set ℤ := {x | abs x < 2}
def B : Set ℤ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_A_B_l253_253416


namespace tan_135_eq_neg1_l253_253541

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 :=
by
  have h_cos : Real.cos (135 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180) := 
    by 
      apply Real.cos_angle_of_pi_sub_angle; 
      sorry
  have h_cos_45 : Real.cos (45 * Real.pi / 180) = (1 / Real.sqrt 2) := 
    by 
      rw Real.cos_pi_div_four;
      sorry
  have h_sin : Real.sin (135 * Real.pi / 180) = Real.sin (45 * Real.pi / 180) := 
    by
      apply Real.sin_of_pi_sub_angle;
      sorry
  have h_sin_45 : Real.sin (45 * Real.pi / 180) = (1 / Real.sqrt 2) := 
    by 
      rw Real.sin_pi_div_four;
      sorry
  rw [← h_sin, h_sin_45, ← h_cos, h_cos_45]
  rw [Real.tan_eq_sin_div_cos, div_eq_mul_inv, mul_comm, inv_mul_cancel]
  norm_num
  sorry

end tan_135_eq_neg1_l253_253541


namespace electronics_weight_l253_253986

variable (B C E : ℝ)

-- Conditions
def initial_ratio : Prop := B / 5 = C / 4 ∧ C / 4 = E / 2
def removed_clothes : Prop := B / 10 = (C - 9) / 4

-- Proof statement
theorem electronics_weight (h1 : initial_ratio B C E) (h2 : removed_clothes B C) : E = 9 := 
by
  sorry

end electronics_weight_l253_253986


namespace min_value_expression_l253_253291

open Real

theorem min_value_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 60 := 
  sorry

end min_value_expression_l253_253291
