import Mathlib

namespace solve_for_a_plus_b_l128_12865

theorem solve_for_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, a * (x + b) = 3 * x + 12) → a + b = 7 :=
by
  intros h
  sorry

end solve_for_a_plus_b_l128_12865


namespace ten_numbers_property_l128_12899

theorem ten_numbers_property (x : ℕ → ℝ) (h : ∀ i : ℕ, 1 ≤ i → i ≤ 9 → x i + 2 * x (i + 1) = 1) : 
  x 1 + 512 * x 10 = 171 :=
by
  sorry

end ten_numbers_property_l128_12899


namespace remainder_of_5n_minus_9_l128_12862

theorem remainder_of_5n_minus_9 (n : ℤ) (h : n % 11 = 3) : (5 * n - 9) % 11 = 6 :=
by
  sorry -- Proof is omitted, as per instruction.

end remainder_of_5n_minus_9_l128_12862


namespace height_of_stack_correct_l128_12834

namespace PaperStack

-- Define the problem conditions
def sheets_per_package : ℕ := 500
def thickness_per_sheet_mm : ℝ := 0.1
def packages_per_stack : ℕ := 60
def mm_to_m : ℝ := 1000.0

-- Statement: the height of the stack of 60 paper packages
theorem height_of_stack_correct :
  (sheets_per_package * thickness_per_sheet_mm * packages_per_stack) / mm_to_m = 3 :=
sorry

end PaperStack

end height_of_stack_correct_l128_12834


namespace principal_amount_l128_12880

theorem principal_amount (A r t : ℝ) (hA : A = 1120) (hr : r = 0.11) (ht : t = 2.4) :
  abs ((A / (1 + r * t)) - 885.82) < 0.01 :=
by
  -- This theorem is stating that given A = 1120, r = 0.11, and t = 2.4,
  -- the principal amount (calculated using the simple interest formula)
  -- is approximately 885.82 with a margin of error less than 0.01.
  sorry

end principal_amount_l128_12880


namespace find_integer_pairs_l128_12891

theorem find_integer_pairs (a b : ℤ) (h₁ : 1 < a) (h₂ : 1 < b) 
    (h₃ : a ∣ (b + 1)) (h₄ : b ∣ (a^3 - 1)) : 
    ∃ (s : ℤ), (s ≥ 2 ∧ (a, b) = (s, s^3 - 1)) ∨ (s ≥ 3 ∧ (a, b) = (s, s - 1)) :=
  sorry

end find_integer_pairs_l128_12891


namespace intersection_M_N_l128_12815

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {x | x ≥ 3}

theorem intersection_M_N : M ∩ N = {3, 4} := 
by
  sorry

end intersection_M_N_l128_12815


namespace trebled_principal_after_5_years_l128_12801

theorem trebled_principal_after_5_years 
(P R : ℝ) (T total_interest : ℝ) (n : ℝ) 
(h1 : T = 10) 
(h2 : total_interest = 800) 
(h3 : (P * R * 10) / 100 = 400) 
(h4 : (P * R * n) / 100 + (3 * P * R * (10 - n)) / 100 = 800) :
n = 5 :=
by
-- The Lean proof will go here
sorry

end trebled_principal_after_5_years_l128_12801


namespace cistern_empty_time_l128_12855

noncomputable def time_to_empty_cistern (fill_no_leak_time fill_with_leak_time : ℝ) (filled_cistern : ℝ) : ℝ :=
  let R := filled_cistern / fill_no_leak_time
  let L := (R - filled_cistern / fill_with_leak_time)
  filled_cistern / L

theorem cistern_empty_time :
  time_to_empty_cistern 12 14 1 = 84 :=
by
  unfold time_to_empty_cistern
  simp
  sorry

end cistern_empty_time_l128_12855


namespace gcd_f_x_l128_12800

-- Define that x is a multiple of 23478
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

-- Define the function f(x)
noncomputable def f (x : ℕ) : ℕ := (2 * x + 3) * (7 * x + 2) * (13 * x + 7) * (x + 13)

-- Assert the proof problem
theorem gcd_f_x (x : ℕ) (h : is_multiple_of x 23478) : Nat.gcd (f x) x = 546 :=
by 
  sorry

end gcd_f_x_l128_12800


namespace original_number_of_men_l128_12853

theorem original_number_of_men (W : ℝ) (M : ℝ) (total_work : ℝ) :
  (M * W * 11 = (M + 10) * W * 8) → M = 27 :=
by
  sorry

end original_number_of_men_l128_12853


namespace matts_weight_l128_12818

theorem matts_weight (protein_per_powder_rate : ℝ)
                     (weekly_intake_powder : ℝ)
                     (daily_protein_required_per_kg : ℝ)
                     (days_in_week : ℝ)
                     (expected_weight : ℝ)
    (h1 : protein_per_powder_rate = 0.8)
    (h2 : weekly_intake_powder = 1400)
    (h3 : daily_protein_required_per_kg = 2)
    (h4 : days_in_week = 7)
    (h5 : expected_weight = 80) :
    (weekly_intake_powder / days_in_week) * protein_per_powder_rate / daily_protein_required_per_kg = expected_weight := by
  sorry

end matts_weight_l128_12818


namespace find_numbers_l128_12839

theorem find_numbers (x y : ℤ) (h_sum : x + y = 40) (h_diff : x - y = 12) : x = 26 ∧ y = 14 :=
sorry

end find_numbers_l128_12839


namespace smallest_positive_integer_for_divisibility_l128_12877

def is_divisible_by (a b : ℕ) : Prop :=
  ∃ k, a = b * k

def smallest_n (n : ℕ) : Prop :=
  (is_divisible_by (n^2) 50) ∧ (is_divisible_by (n^3) 288) ∧ (∀ m : ℕ, m > 0 → m < n → ¬ (is_divisible_by (m^2) 50 ∧ is_divisible_by (m^3) 288))

theorem smallest_positive_integer_for_divisibility : smallest_n 60 :=
by
  sorry

end smallest_positive_integer_for_divisibility_l128_12877


namespace trout_ratio_l128_12872

theorem trout_ratio (caleb_trouts dad_trouts : ℕ) (h_c : caleb_trouts = 2) (h_d : dad_trouts = caleb_trouts + 4) :
  dad_trouts / (Nat.gcd dad_trouts caleb_trouts) = 3 ∧ caleb_trouts / (Nat.gcd dad_trouts caleb_trouts) = 1 :=
by
  sorry

end trout_ratio_l128_12872


namespace calculation_result_l128_12840

theorem calculation_result : 
  (16 = 2^4) → 
  (8 = 2^3) → 
  (4 = 2^2) → 
  (16^6 * 8^3 / 4^10 = 8192) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end calculation_result_l128_12840


namespace inequality_ay_bz_cx_lt_k_squared_l128_12849

theorem inequality_ay_bz_cx_lt_k_squared
  (a b c x y z k : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hk1 : a + x = k) (hk2 : b + y = k) (hk3 : c + z = k) :
  (a * y + b * z + c * x) < k^2 :=
sorry

end inequality_ay_bz_cx_lt_k_squared_l128_12849


namespace cobs_count_l128_12846

theorem cobs_count (bushel_weight : ℝ) (ear_weight : ℝ) (num_bushels : ℕ)
  (h1 : bushel_weight = 56) (h2 : ear_weight = 0.5) (h3 : num_bushels = 2) : 
  ((num_bushels * bushel_weight) / ear_weight) = 224 :=
by 
  sorry

end cobs_count_l128_12846


namespace range_of_x_l128_12803

variable (a x : ℝ)

theorem range_of_x :
  (∃ a ∈ Set.Icc 2 4, a * x ^ 2 + (a - 3) * x - 3 > 0) →
  x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 4) :=
by
  sorry

end range_of_x_l128_12803


namespace part1_solution_part2_solution_l128_12866

def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

theorem part1_solution (x : ℝ) : 
  f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := sorry

theorem part2_solution (a : ℝ) :
  (∃ x : ℝ, f x a < 2 * a) ↔ 3 < a := sorry

end part1_solution_part2_solution_l128_12866


namespace no_real_b_for_inequality_l128_12837

theorem no_real_b_for_inequality : ¬ ∃ b : ℝ, (∃ x : ℝ, |x^2 + 3 * b * x + 4 * b| = 5 ∧ ∀ y : ℝ, y ≠ x → |y^2 + 3 * b * y + 4 * b| > 5) := sorry

end no_real_b_for_inequality_l128_12837


namespace four_digit_number_count_l128_12809

theorem four_digit_number_count (A : ℕ → ℕ → ℕ)
  (odd_digits even_digits : Finset ℕ)
  (odds : ∀ x ∈ odd_digits, x % 2 = 1)
  (evens : ∀ x ∈ even_digits, x % 2 = 0) :
  odd_digits = {1, 3, 5, 7, 9} ∧ 
  even_digits = {2, 4, 6, 8} →
  A 5 2 * A 7 2 = 840 :=
by
  intros h1
  sorry

end four_digit_number_count_l128_12809


namespace sum_end_digit_7_l128_12850

theorem sum_end_digit_7 (n : ℕ) : ¬ (n * (n + 1) ≡ 14 [MOD 20]) :=
by
  intro h
  -- Place where you'd continue the proof, but for now we use sorry
  sorry

end sum_end_digit_7_l128_12850


namespace probability_two_asian_countries_probability_A1_not_B1_l128_12861

-- Scope: Definitions for the problem context
def countries : List String := ["A1", "A2", "A3", "B1", "B2", "B3"]

-- Probability of picking two Asian countries from a pool of six (three Asian, three European)
theorem probability_two_asian_countries : 
  (3 / 15) = (1 / 5) := by
  sorry

-- Probability of picking one country from the Asian group and 
-- one from the European group, including A1 but not B1
theorem probability_A1_not_B1 : 
  (2 / 9) = (2 / 9) := by
  sorry

end probability_two_asian_countries_probability_A1_not_B1_l128_12861


namespace find_solutions_l128_12883

noncomputable def equation (x : ℝ) : ℝ :=
  (1 / (x^2 + 11*x - 8)) + (1 / (x^2 + 2*x - 8)) + (1 / (x^2 - 13*x - 8))

theorem find_solutions : 
  {x : ℝ | equation x = 0} = {1, -8, 8, -1} := by
  sorry

end find_solutions_l128_12883


namespace find_smallest_n_l128_12832

-- Definitions of the condition that m and n are relatively prime and that the fraction includes the digits 4, 5, and 6 consecutively
def is_coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

def has_digits_456 (m n : ℕ) : Prop := 
  ∃ k : ℕ, ∃ c : ℕ, 10^k * m % (10^k * n) = 456 * 10^c

-- The theorem to prove the smallest value of n
theorem find_smallest_n (m n : ℕ) (h1 : is_coprime m n) (h2 : m < n) (h3 : has_digits_456 m n) : n = 230 :=
sorry

end find_smallest_n_l128_12832


namespace divide_segment_mean_proportional_l128_12884

theorem divide_segment_mean_proportional (a : ℝ) (x : ℝ) : 
  ∃ H : ℝ, H > 0 ∧ H < a ∧ H = (a * (Real.sqrt 5 - 1) / 2) :=
sorry

end divide_segment_mean_proportional_l128_12884


namespace tangent_circle_equation_l128_12810

theorem tangent_circle_equation :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi →
    ∃ c : ℝ × ℝ, ∃ r : ℝ,
      (∀ (a b : ℝ), c = (a, b) →
        (|a * Real.cos θ + b * Real.sin θ - Real.cos θ - 2 * Real.sin θ - 2| = r) ∧
        (r = 2)) ∧
      (∃ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = r^2)) :=
by
  sorry

end tangent_circle_equation_l128_12810


namespace no_intersection_of_graphs_l128_12805

theorem no_intersection_of_graphs :
  ∃ x y : ℝ, y = |3 * x + 6| ∧ y = -|4 * x - 3| → false := by
  sorry

end no_intersection_of_graphs_l128_12805


namespace sqrt_product_equals_l128_12824

noncomputable def sqrt128 : ℝ := Real.sqrt 128
noncomputable def sqrt50 : ℝ := Real.sqrt 50
noncomputable def sqrt18 : ℝ := Real.sqrt 18

theorem sqrt_product_equals : sqrt128 * sqrt50 * sqrt18 = 240 * Real.sqrt 2 := 
by
  sorry

end sqrt_product_equals_l128_12824


namespace calculate_x_l128_12857

theorem calculate_x :
  529 + 2 * 23 * 11 + 121 = 1156 :=
by
  -- Begin the proof (which we won't complete here)
  -- The proof steps would go here
  sorry  -- placeholder for the actual proof steps

end calculate_x_l128_12857


namespace seconds_hand_revolution_l128_12838

theorem seconds_hand_revolution (revTimeSeconds revTimeMinutes : ℕ) : 
  (revTimeSeconds = 60) ∧ (revTimeMinutes = 1) :=
sorry

end seconds_hand_revolution_l128_12838


namespace minimum_ribbon_length_l128_12873

def side_length : ℚ := 13 / 12

def perimeter_of_equilateral_triangle (a : ℚ) : ℚ := 3 * a

theorem minimum_ribbon_length :
  perimeter_of_equilateral_triangle side_length = 3.25 := 
by
  sorry

end minimum_ribbon_length_l128_12873


namespace minimum_tasks_for_18_points_l128_12886

def task_count (points : ℕ) : ℕ :=
  if points <= 9 then
    (points / 3) * 1
  else if points <= 15 then
    3 + (points - 9 + 2) / 3 * 2
  else
    3 + 4 + (points - 15 + 2) / 3 * 3

theorem minimum_tasks_for_18_points : task_count 18 = 10 := by
  sorry

end minimum_tasks_for_18_points_l128_12886


namespace sequence_term_1000_l128_12804

theorem sequence_term_1000 :
  ∃ (a : ℕ → ℤ), a 1 = 2007 ∧ a 2 = 2008 ∧ (∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) ∧ a 1000 = 2340 := 
by
  sorry

end sequence_term_1000_l128_12804


namespace total_amount_paid_l128_12812

theorem total_amount_paid :
  let pizzas := 3
  let cost_per_pizza := 8
  let total_cost := pizzas * cost_per_pizza
  total_cost = 24 :=
by
  sorry

end total_amount_paid_l128_12812


namespace gcd_problem_l128_12811

def a : ℕ := 101^5 + 1
def b : ℕ := 101^5 + 101^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := by
  sorry

end gcd_problem_l128_12811


namespace original_speed_l128_12831

noncomputable def circumference_feet := 10
noncomputable def feet_to_miles := 5280
noncomputable def seconds_to_hours := 3600
noncomputable def shortened_time := 1 / 18000
noncomputable def speed_increase := 6

theorem original_speed (r : ℝ) (t : ℝ) : 
  r * t = (circumference_feet / feet_to_miles) * seconds_to_hours ∧ 
  (r + speed_increase) * (t - shortened_time) = (circumference_feet / feet_to_miles) * seconds_to_hours
  → r = 6 := 
by
  sorry

end original_speed_l128_12831


namespace contrapositive_equivalence_l128_12856

variable (p q : Prop)

theorem contrapositive_equivalence : (p → ¬q) ↔ (q → ¬p) := by
  sorry

end contrapositive_equivalence_l128_12856


namespace eval_expression_l128_12830

theorem eval_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end eval_expression_l128_12830


namespace vincent_total_loads_l128_12844

def loads_wednesday : Nat := 2 + 1 + 3

def loads_thursday : Nat := 2 * loads_wednesday

def loads_friday : Nat := loads_thursday / 2

def loads_saturday : Nat := loads_wednesday / 3

def total_loads : Nat := loads_wednesday + loads_thursday + loads_friday + loads_saturday

theorem vincent_total_loads : total_loads = 20 := by
  -- Proof will be filled in here
  sorry

end vincent_total_loads_l128_12844


namespace no_distinct_natural_numbers_eq_sum_and_cubes_eq_l128_12881

theorem no_distinct_natural_numbers_eq_sum_and_cubes_eq:
  ∀ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
  → a^3 + b^3 = c^3 + d^3
  → a + b = c + d
  → false := 
by
  intros
  sorry

end no_distinct_natural_numbers_eq_sum_and_cubes_eq_l128_12881


namespace standard_equation_of_circle_l128_12833

theorem standard_equation_of_circle :
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ (h - 2) / 2 = k / 1 + 3 / 2 ∧ 
  ((h - 2)^2 + (k + 3)^2 = r^2) ∧ ((h + 2)^2 + (k + 5)^2 = r^2) ∧ 
  h = -1 ∧ k = -2 ∧ r^2 = 10 :=
by
  sorry

end standard_equation_of_circle_l128_12833


namespace total_treats_l128_12817

theorem total_treats (children : ℕ) (hours : ℕ) (houses_per_hour : ℕ) (treats_per_house_per_kid : ℕ) :
  children = 3 → hours = 4 → houses_per_hour = 5 → treats_per_house_per_kid = 3 → 
  (children * hours * houses_per_hour * treats_per_house_per_kid) = 180 :=
by
  intros
  sorry

end total_treats_l128_12817


namespace number_of_pictures_l128_12821

theorem number_of_pictures (x : ℕ) (h : x - (x / 2 - 1) = 25) : x = 48 :=
sorry

end number_of_pictures_l128_12821


namespace arithmetic_sequence_term_count_l128_12848

theorem arithmetic_sequence_term_count (a d n an : ℕ) (h₀ : a = 5) (h₁ : d = 7) (h₂ : an = 126) (h₃ : an = a + (n - 1) * d) : n = 18 := by
  sorry

end arithmetic_sequence_term_count_l128_12848


namespace intersection_of_M_and_N_is_12_l128_12845

def M : Set ℤ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℤ := {1, 2, 3}

theorem intersection_of_M_and_N_is_12 : M ∩ N = {1, 2} :=
by
  sorry

end intersection_of_M_and_N_is_12_l128_12845


namespace price_increase_percentage_l128_12888

theorem price_increase_percentage (x : ℝ) :
  (0.9 * (1 + x / 100) * 0.9259259259259259 = 1) → x = 20 :=
by
  intros
  sorry

end price_increase_percentage_l128_12888


namespace right_angled_triangle_other_angle_isosceles_triangle_base_angle_l128_12852

theorem right_angled_triangle_other_angle (a : ℝ) (h1 : 0 < a) (h2 : a < 90) (h3 : 40 = a) :
  50 = 90 - a :=
sorry

theorem isosceles_triangle_base_angle (v : ℝ) (h1 : 0 < v) (h2 : v < 180) (h3 : 80 = v) :
  50 = (180 - v) / 2 :=
sorry

end right_angled_triangle_other_angle_isosceles_triangle_base_angle_l128_12852


namespace least_three_digit_product_18_l128_12874

theorem least_three_digit_product_18 : ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ N = 100 * H + 10 * T + U ∧ H * T * U = 18) ∧ ∀ M : ℕ, (100 ≤ M ∧ M ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ M = 100 * H + 10 * T + U ∧ H * T * U = 18)) → N ≤ M :=
    sorry

end least_three_digit_product_18_l128_12874


namespace log_inequality_l128_12875

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 2 / Real.log 5
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem log_inequality : c > a ∧ a > b := 
by
  sorry

end log_inequality_l128_12875


namespace smallest_positive_value_l128_12854

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℝ), k = 2 ∧ k = (↑(a - b) / ↑(a + b) + ↑(a + b) / ↑(a - b)) :=
sorry

end smallest_positive_value_l128_12854


namespace rank_A_second_l128_12842

-- We define the conditions provided in the problem
variables (a b c : ℕ) -- defining the scores of A, B, and C as natural numbers

-- Conditions given
def A_said (a b c : ℕ) := b < a ∧ c < a
def B_said (b c : ℕ) := b > c
def C_said (a b c : ℕ) := a > c ∧ b > c

-- Conditions as hypotheses
variable (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) -- the scores are different
variable (h2 : A_said a b c ∨ B_said b c ∨ C_said a b c) -- exactly one of the statements is incorrect

-- The theorem to prove
theorem rank_A_second : ∃ (rankA : ℕ), rankA = 2 := by
  sorry

end rank_A_second_l128_12842


namespace cost_of_one_basketball_deck_l128_12814

theorem cost_of_one_basketball_deck (total_money_spent : ℕ) 
  (mary_sunglasses_cost : ℕ) (mary_jeans_cost : ℕ) 
  (rose_shoes_cost : ℕ) (rose_decks_count : ℕ) 
  (mary_total_cost : total_money_spent = 2 * mary_sunglasses_cost + mary_jeans_cost)
  (rose_total_cost : total_money_spent = rose_shoes_cost + 2 * (total_money_spent - rose_shoes_cost) / rose_decks_count) :
  (total_money_spent - rose_shoes_cost) / rose_decks_count = 25 := 
by 
  sorry

end cost_of_one_basketball_deck_l128_12814


namespace sara_initial_quarters_l128_12887

theorem sara_initial_quarters (total_quarters dad_gift initial_quarters : ℕ) (h1 : dad_gift = 49) (h2 : total_quarters = 70) (h3 : total_quarters = initial_quarters + dad_gift) : initial_quarters = 21 :=
by sorry

end sara_initial_quarters_l128_12887


namespace value_of_x_l128_12897

theorem value_of_x (x : ℝ) : 3 - 5 + 7 = 6 - x → x = 1 :=
by
  intro h
  sorry

end value_of_x_l128_12897


namespace pirate_15_gets_coins_l128_12876

def coins_required_for_pirates : ℕ :=
  Nat.factorial 14 * ((2 ^ 4) * (3 ^ 9)) / 15 ^ 14

theorem pirate_15_gets_coins :
  coins_required_for_pirates = 314928 := 
by sorry

end pirate_15_gets_coins_l128_12876


namespace seventh_graders_problems_l128_12879

theorem seventh_graders_problems (n : ℕ) (S : ℕ) (a : ℕ) (h1 : a > (S - a) / 5) (h2 : a < (S - a) / 3) : n = 5 :=
  sorry

end seventh_graders_problems_l128_12879


namespace people_with_fewer_than_7_cards_l128_12864

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l128_12864


namespace vector_addition_l128_12827

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (6, 2)
def vector_b : ℝ × ℝ := (-2, 4)

-- Theorem statement to prove the sum of vector_a and vector_b equals (4, 6)
theorem vector_addition :
  vector_a + vector_b = (4, 6) :=
sorry

end vector_addition_l128_12827


namespace michael_meets_truck_once_l128_12841

def michael_speed := 5  -- feet per second
def pail_distance := 150  -- feet
def truck_speed := 15  -- feet per second
def truck_stop_time := 20  -- seconds

def initial_michael_position (t : ℕ) : ℕ := t * michael_speed
def initial_truck_position (t : ℕ) : ℕ := pail_distance + t * truck_speed - (t / (truck_speed * truck_stop_time))

def distance (t : ℕ) : ℕ := initial_truck_position t - initial_michael_position t

theorem michael_meets_truck_once :
  ∃ t, (distance t = 0) :=  
sorry

end michael_meets_truck_once_l128_12841


namespace Lisa_quiz_goal_l128_12828

theorem Lisa_quiz_goal (total_quizzes : ℕ) (required_percentage : ℝ) (a_scored : ℕ) (completed_quizzes : ℕ) : 
  total_quizzes = 60 → 
  required_percentage = 0.75 → 
  a_scored = 30 → 
  completed_quizzes = 40 → 
  ∃ lower_than_a_quizzes : ℕ, lower_than_a_quizzes = 5 :=
by
  intros total_quizzes_eq req_percent_eq a_scored_eq completed_quizzes_eq
  sorry

end Lisa_quiz_goal_l128_12828


namespace find_a6_l128_12898

-- Define the geometric sequence conditions
noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the specific sequence with given initial conditions and sum of first three terms
theorem find_a6 : 
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (0 < q) ∧ (q ≠ 1) ∧ geom_seq a q ∧ 
    a 1 = 96 ∧ 
    (a 1 + a 2 + a 3 = 168) ∧
    a 6 = 3 := 
by
  sorry

end find_a6_l128_12898


namespace area_ratio_of_isosceles_triangle_l128_12807

variable (x : ℝ)
variable (hx : 0 < x)

def isosceles_triangle (AB AC : ℝ) (BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 * x ∧ BC = x

def extend_side (B_length AB_length : ℝ) : Prop :=
  B_length = 2 * AB_length

def ratio_of_areas (area_AB'B'C' area_ABC : ℝ) : Prop :=
  area_AB'B'C' / area_ABC = 9

theorem area_ratio_of_isosceles_triangle
  (AB AC BC : ℝ) (BB' B'C' area_ABC area_AB'B'C' : ℝ)
  (h_isosceles : isosceles_triangle x AB AC BC)
  (h_extend_A : extend_side BB' AB)
  (h_extend_C : extend_side B'C' AC) :
  ratio_of_areas area_AB'B'C' area_ABC := by
  sorry

end area_ratio_of_isosceles_triangle_l128_12807


namespace michelle_will_have_four_crayons_l128_12869

def michelle_crayons (m j : ℕ) : ℕ := m + j

theorem michelle_will_have_four_crayons (H₁ : michelle_crayons 2 2 = 4) : michelle_crayons 2 2 = 4 :=
by
  sorry

end michelle_will_have_four_crayons_l128_12869


namespace eighteen_mnp_eq_P_np_Q_2mp_l128_12867

theorem eighteen_mnp_eq_P_np_Q_2mp (m n p : ℕ) (P Q : ℕ) (hP : P = 2 ^ m) (hQ : Q = 3 ^ n) :
  18 ^ (m * n * p) = P ^ (n * p) * Q ^ (2 * m * p) :=
by
  sorry

end eighteen_mnp_eq_P_np_Q_2mp_l128_12867


namespace first_movie_series_seasons_l128_12858

theorem first_movie_series_seasons (S : ℕ) : 
  (∀ E : ℕ, E = 16) → 
  (∀ L : ℕ, L = 2) → 
  (∀ T : ℕ, T = 364) → 
  (∀ second_series_seasons : ℕ, second_series_seasons = 14) → 
  (∀ second_series_remaining : ℕ, second_series_remaining = second_series_seasons * (E - L)) → 
  (E - L = 14) → 
  (second_series_remaining = 196) → 
  (T - second_series_remaining = S * (E - L)) → 
  S = 12 :=
by 
  intros E_16 L_2 T_364 second_series_14 second_series_remaining_196 E_L second_series_total_episodes remaining_episodes
  sorry

end first_movie_series_seasons_l128_12858


namespace centroid_midpoint_triangle_eq_centroid_original_triangle_l128_12859

/-
Prove that the centroid of the triangle formed by the midpoints of the sides of another triangle
is the same as the centroid of the original triangle.
-/
theorem centroid_midpoint_triangle_eq_centroid_original_triangle
  (A B C M N P : ℝ × ℝ)
  (hM : M = (A + B) / 2)
  (hN : N = (A + C) / 2)
  (hP : P = (B + C) / 2) :
  (M.1 + N.1 + P.1) / 3 = (A.1 + B.1 + C.1) / 3 ∧
  (M.2 + N.2 + P.2) / 3 = (A.2 + B.2 + C.2) / 3 :=
by
  sorry

end centroid_midpoint_triangle_eq_centroid_original_triangle_l128_12859


namespace boys_collected_in_all_l128_12882

-- Definition of the problem’s conditions
variables (solomon juwan levi : ℕ)

-- Given conditions as assumptions
def conditions : Prop :=
  solomon = 66 ∧
  solomon = 3 * juwan ∧
  levi = juwan / 2

-- Total cans collected by all boys
def total_cans (solomon juwan levi : ℕ) : ℕ := solomon + juwan + levi

theorem boys_collected_in_all : ∃ solomon juwan levi : ℕ, 
  conditions solomon juwan levi ∧ total_cans solomon juwan levi = 99 :=
by {
  sorry
}

end boys_collected_in_all_l128_12882


namespace last_digit_of_large_prime_l128_12820

theorem last_digit_of_large_prime :
  let n := 2^859433 - 1
  let last_digit := n % 10
  last_digit = 1 :=
by
  sorry

end last_digit_of_large_prime_l128_12820


namespace Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l128_12819

/-- Definitions for phone plans A and B and phone call durations -/
def fixed_cost_A : ℕ := 18
def free_minutes_A : ℕ := 1500
def price_per_minute_A : ℕ → ℚ := λ t => 0.1 * t

def fixed_cost_B : ℕ := 38
def free_minutes_B : ℕ := 4000
def price_per_minute_B : ℕ → ℚ := λ t => 0.07 * t

def call_duration_October : ℕ := 2600
def total_bill_November_December : ℚ := 176
def total_call_duration_November_December : ℕ := 5200

/-- Problem statements to be proven -/

theorem Phone_Bill_October : 
  fixed_cost_A + price_per_minute_A (call_duration_October - free_minutes_A) = 128 :=
  sorry

theorem Phone_Bill_November_December (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ total_call_duration_November_December) : 
  let bill_November := fixed_cost_A + price_per_minute_A (x - free_minutes_A)
  let bill_December := fixed_cost_B + price_per_minute_B (total_call_duration_November_December - x - free_minutes_B)
  bill_November + bill_December = total_bill_November_December :=
  sorry
  
theorem Extra_Cost_November_December :
  let actual_cost := 138 + 38
  let hypothetical_cost := fixed_cost_A + price_per_minute_A (total_call_duration_November_December - free_minutes_A)
  hypothetical_cost - actual_cost = 80 :=
  sorry

end Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l128_12819


namespace isosceles_triangle_largest_angle_l128_12826

theorem isosceles_triangle_largest_angle (a b c : ℝ) 
  (h1 : a = b)
  (h2 : c + 50 + 50 = 180) : 
  c = 80 :=
by sorry

end isosceles_triangle_largest_angle_l128_12826


namespace original_price_of_article_l128_12894

theorem original_price_of_article :
  ∃ P : ℝ, (P * 0.55 * 0.85 = 920) ∧ P = 1968.04 :=
by
  sorry

end original_price_of_article_l128_12894


namespace range_of_b_l128_12829

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  - (1/2) * (x - 2)^2 + b * Real.log x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 1 < x → f x b ≤ f 1 b) → b ≤ -1 :=
by
  sorry

end range_of_b_l128_12829


namespace Peter_buys_more_hot_dogs_than_hamburgers_l128_12890

theorem Peter_buys_more_hot_dogs_than_hamburgers :
  let chicken := 16
  let hamburgers := chicken / 2
  (exists H : Real, 16 + hamburgers + H + H / 2 = 39 ∧ (H - hamburgers = 2)) := sorry

end Peter_buys_more_hot_dogs_than_hamburgers_l128_12890


namespace solve_inequality_range_of_a_l128_12895

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 2|

theorem solve_inequality : {x : ℝ | f x > 5} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 4 / 3} :=
by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x, ¬ (f x < a)) : a ≤ 2 :=
by
  sorry

end solve_inequality_range_of_a_l128_12895


namespace accurate_scale_l128_12892

-- Definitions for the weights on each scale
variables (a b c d e x : ℝ)

-- Given conditions
def condition1 := c = b - 0.3
def condition2 := d = c - 0.1
def condition3 := e = a - 0.1
def condition4 := c = e - 0.1
def condition5 := 5 * x = a + b + c + d + e

-- Proof statement
theorem accurate_scale 
  (h1 : c = b - 0.3)
  (h2 : d = c - 0.1)
  (h3 : e = a - 0.1)
  (h4 : c = e - 0.1)
  (h5 : 5 * x = a + b + c + d + e) : e = x :=
by
  sorry

end accurate_scale_l128_12892


namespace solution_set_of_inequality_l128_12860

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_2 : f 2 = 1 / 2
axiom f_prime_lt_exp : ∀ x : ℝ, deriv f x < Real.exp x

theorem solution_set_of_inequality :
  {x : ℝ | f x < Real.exp x - 1 / 2} = {x : ℝ | 0 < x} :=
by
  sorry

end solution_set_of_inequality_l128_12860


namespace bake_four_pans_l128_12802

-- Define the conditions
def bake_time_one_pan : ℕ := 7
def total_bake_time (n : ℕ) : ℕ := 28

-- Define the theorem statement
theorem bake_four_pans : total_bake_time 4 = 28 :=
by
  -- Proof is omitted
  sorry

end bake_four_pans_l128_12802


namespace top_card_is_queen_probability_l128_12885

-- Define the conditions of the problem
def standard_deck_size := 52
def number_of_queens := 4

-- Problem statement: The probability that the top card is a Queen
theorem top_card_is_queen_probability : 
  (number_of_queens : ℚ) / standard_deck_size = 1 / 13 := 
sorry

end top_card_is_queen_probability_l128_12885


namespace eduardo_needs_l128_12808

variable (flour_per_24_cookies sugar_per_24_cookies : ℝ)
variable (num_cookies : ℝ)

axiom h_flour : flour_per_24_cookies = 1.5
axiom h_sugar : sugar_per_24_cookies = 0.5
axiom h_cookies : num_cookies = 120

theorem eduardo_needs (scaling_factor : ℝ) 
    (flour_needed : ℝ)
    (sugar_needed : ℝ)
    (h_scaling : scaling_factor = num_cookies / 24)
    (h_flour_needed : flour_needed = flour_per_24_cookies * scaling_factor)
    (h_sugar_needed : sugar_needed = sugar_per_24_cookies * scaling_factor) :
  flour_needed = 7.5 ∧ sugar_needed = 2.5 :=
sorry

end eduardo_needs_l128_12808


namespace main_factor_is_D_l128_12851

-- Let A, B, C, and D be the factors where A is influenced by 1, B by 2, C by 3, and D by 4
def A := 1
def B := 2
def C := 3
def D := 4

-- Defining the main factor influenced by the plan
def main_factor_influenced_by_plan := D

-- The problem statement translated to a Lean theorem statement
theorem main_factor_is_D : main_factor_influenced_by_plan = D := 
by sorry

end main_factor_is_D_l128_12851


namespace remainder_of_xyz_l128_12806

theorem remainder_of_xyz {x y z : ℕ} (hx: x < 9) (hy: y < 9) (hz: z < 9)
  (h1: (x + 3*y + 2*z) % 9 = 0)
  (h2: (2*x + 2*y + z) % 9 = 7)
  (h3: (x + 2*y + 3*z) % 9 = 5) :
  (x * y * z) % 9 = 5 :=
sorry

end remainder_of_xyz_l128_12806


namespace range_of_m_l128_12816

theorem range_of_m (m : ℝ) :
  (∀ x, |x^2 - 4 * x + m| ≤ x + 4 ↔ (-4 ≤ m ∧ m ≤ 4)) ∧
  (∀ x, (x = 0 → |0^2 - 4 * 0 + m| ≤ 0 + 4) ∧ (x = 2 → ¬(|2^2 - 4 * 2 + m| ≤ 2 + 4))) →
  (-4 ≤ m ∧ m < -2) :=
by
  sorry

end range_of_m_l128_12816


namespace lisa_flight_time_l128_12835

theorem lisa_flight_time :
  ∀ (d s : ℕ), (d = 256) → (s = 32) → ((d / s) = 8) :=
by
  intros d s h_d h_s
  sorry

end lisa_flight_time_l128_12835


namespace factorize_expression_l128_12896

theorem factorize_expression (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 :=
by
  sorry

end factorize_expression_l128_12896


namespace total_worth_of_presents_l128_12893

-- Define the costs as given in the conditions
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def bracelet_cost : ℕ := 2 * ring_cost

-- Define the total worth of the presents
def total_worth : ℕ := ring_cost + car_cost + bracelet_cost

-- Statement: Prove the total worth is 14000
theorem total_worth_of_presents : total_worth = 14000 :=
by
  -- Here is the proof statement
  sorry

end total_worth_of_presents_l128_12893


namespace number_of_registration_methods_l128_12868

theorem number_of_registration_methods
  (students : ℕ) (groups : ℕ) (registration_methods : ℕ)
  (h_students : students = 4) (h_groups : groups = 3) :
  registration_methods = groups ^ students :=
by
  rw [h_students, h_groups]
  exact sorry

end number_of_registration_methods_l128_12868


namespace values_of_m_l128_12847

theorem values_of_m (m n : ℕ) (hmn : m * n = 900) (hm: m > 1) (hn: n ≥ 1) : 
  (∃ (k : ℕ), ∀ (m : ℕ), (1 < m ∧ (900 / m) ≥ 1 ∧ 900 % m = 0) ↔ k = 25) :=
sorry

end values_of_m_l128_12847


namespace sum_arithmetic_series_l128_12878

theorem sum_arithmetic_series : 
    let a₁ := 1
    let d := 2
    let n := 9
    let a_n := a₁ + (n - 1) * d
    let S_n := n * (a₁ + a_n) / 2
    a_n = 17 → S_n = 81 :=
by intros
   sorry

end sum_arithmetic_series_l128_12878


namespace find_width_of_room_l128_12843

variable (length : ℕ) (total_carpet_owned : ℕ) (additional_carpet_needed : ℕ)
variable (total_area : ℕ) (width : ℕ)

theorem find_width_of_room
  (h1 : length = 11) 
  (h2 : total_carpet_owned = 16) 
  (h3 : additional_carpet_needed = 149)
  (h4 : total_area = total_carpet_owned + additional_carpet_needed) 
  (h5 : total_area = length * width) :
  width = 15 := by
    sorry

end find_width_of_room_l128_12843


namespace solve_system_eq_l128_12871

theorem solve_system_eq (x y : ℚ) 
  (h1 : 3 * x - 7 * y = 31) 
  (h2 : 5 * x + 2 * y = -10) : 
  x = -336 / 205 := 
sorry

end solve_system_eq_l128_12871


namespace B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l128_12822

-- Define the sets A and B
def setA : Set ℝ := {x : ℝ | x < 1 ∨ x > 2}
def setB (m : ℝ) : Set ℝ := 
  if m = 0 then {x : ℝ | x > 1} 
  else if m < 0 then {x : ℝ | x > 1 ∨ x < (2/m)}
  else if 0 < m ∧ m < 2 then {x : ℝ | 1 < x ∧ x < (2/m)}
  else if m = 2 then ∅
  else {x : ℝ | (2/m) < x ∧ x < 1}

-- Complement of set A
def complementA : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}

-- Proposition: if B subset of complement of A
theorem B_subset_complementA (m : ℝ) : setB m ⊆ complementA ↔ 1 ≤ m ∧ m ≤ 2 := by
  sorry

-- Similarly, we can define the other two propositions
theorem A_intersection_B_nonempty (m : ℝ) : (setA ∩ setB m).Nonempty ↔ m < 1 ∨ m > 2 := by
  sorry

theorem A_union_B_eq_A (m : ℝ) : setA ∪ setB m = setA ↔ m ≥ 2 := by
  sorry

end B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l128_12822


namespace jims_investment_l128_12863

theorem jims_investment
  {total_investment : ℝ} 
  (h1 : total_investment = 127000)
  {john_ratio : ℕ} 
  (h2 : john_ratio = 8)
  {james_ratio : ℕ} 
  (h3 : james_ratio = 11)
  {jim_ratio : ℕ} 
  (h4 : jim_ratio = 15)
  {jordan_ratio : ℕ} 
  (h5 : jordan_ratio = 19) :
  jim_ratio / (john_ratio + james_ratio + jim_ratio + jordan_ratio) * total_investment = 35943.40 :=
by {
  sorry
}

end jims_investment_l128_12863


namespace solve_equation_l128_12825

def equation_holds (x : ℝ) : Prop := 
  (1 / (x + 10)) + (1 / (x + 8)) = (1 / (x + 11)) + (1 / (x + 7))

theorem solve_equation : equation_holds (-9) :=
by
  sorry

end solve_equation_l128_12825


namespace Tom_sold_games_for_240_l128_12836

-- Define the value of games and perform operations as per given conditions
def original_value : ℕ := 200
def tripled_value : ℕ := 3 * original_value
def sold_percentage : ℕ := 40
def sold_value : ℕ := (sold_percentage * tripled_value) / 100

-- Assert the proof problem
theorem Tom_sold_games_for_240 : sold_value = 240 := 
by
  sorry

end Tom_sold_games_for_240_l128_12836


namespace rectangle_y_value_l128_12823

theorem rectangle_y_value (y : ℝ) (h₁ : (-2, y) ≠ (10, y))
  (h₂ : (-2, -1) ≠ (10, -1))
  (h₃ : 12 * (y + 1) = 108)
  (y_pos : 0 < y) :
  y = 8 :=
by
  sorry

end rectangle_y_value_l128_12823


namespace range_of_a_l128_12889

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1) ^ 2 > 4 → x > a) → a ≥ 1 := sorry

end range_of_a_l128_12889


namespace power_equation_l128_12813

theorem power_equation (m : ℤ) (h : 16 = 2 ^ 4) : (16 : ℝ) ^ (3 / 4) = (2 : ℝ) ^ (m : ℝ) → m = 3 := by
  intros
  sorry

end power_equation_l128_12813


namespace smallest_integer_in_consecutive_set_l128_12870

theorem smallest_integer_in_consecutive_set (n : ℤ) (h : n + 6 < 2 * (n + 3)) : n > 0 := by
  sorry

end smallest_integer_in_consecutive_set_l128_12870
