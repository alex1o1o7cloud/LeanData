import Mathlib

namespace lcm_two_numbers_l2318_231805

theorem lcm_two_numbers
  (a b : ℕ)
  (hcf_ab : Nat.gcd a b = 20)
  (product_ab : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end lcm_two_numbers_l2318_231805


namespace geometric_sequence_sum_l2318_231875

theorem geometric_sequence_sum (a_1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  (∀ n, S (n+1) = a_1 * (1 - q^(n+1)) / (1 - q)) →
  S 4 / a_1 = 15 :=
by
  intros hq hsum
  sorry

end geometric_sequence_sum_l2318_231875


namespace min_omega_sin_two_max_l2318_231823

theorem min_omega_sin_two_max (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → ∃ k : ℤ, (ω * x = (2 + 2 * k) * π)) →
  ∃ ω_min : ℝ, ω_min = 4 * π :=
by
  sorry

end min_omega_sin_two_max_l2318_231823


namespace arithmetic_sequence_problem_l2318_231850

variable (n : ℕ) (a S : ℕ → ℕ)

theorem arithmetic_sequence_problem
  (h1 : a 2 + a 8 = 82)
  (h2 : S 41 = S 9)
  (hSn : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n, a n = 51 - 2 * n) ∧ (∀ n, S n ≤ 625) := sorry

end arithmetic_sequence_problem_l2318_231850


namespace ways_to_distribute_books_into_bags_l2318_231876

theorem ways_to_distribute_books_into_bags : 
  let books := 5
  let bags := 4
  ∃ (ways : ℕ), ways = 41 := 
sorry

end ways_to_distribute_books_into_bags_l2318_231876


namespace number_of_young_fish_l2318_231842

-- Define the conditions
def tanks : ℕ := 3
def pregnantFishPerTank : ℕ := 4
def youngPerFish : ℕ := 20

-- Define the proof problem
theorem number_of_young_fish : (tanks * pregnantFishPerTank * youngPerFish) = 240 := by
  sorry

end number_of_young_fish_l2318_231842


namespace total_marbles_proof_l2318_231891

def red_marble_condition (b r : ℕ) : Prop :=
  r = b + (3 * b / 10)

def yellow_marble_condition (r y : ℕ) : Prop :=
  y = r + (5 * r / 10)

def total_marbles (b r y : ℕ) : ℕ :=
  r + b + y

theorem total_marbles_proof (b r y : ℕ)
  (h1 : red_marble_condition b r)
  (h2 : yellow_marble_condition r y) :
  total_marbles b r y = 425 * r / 130 :=
by {
  sorry
}

end total_marbles_proof_l2318_231891


namespace find_m_range_l2318_231862

-- Define the mathematical objects and conditions
def condition_p (m : ℝ) : Prop :=
  (|1 - m| / Real.sqrt 2) > 1

def condition_q (m : ℝ) : Prop :=
  m < 4

-- Define the proof problem
theorem find_m_range (p q : Prop) (m : ℝ) 
  (hp : ¬ p) (hq : q) (hpq : p ∨ q)
  (hP_imp : p → condition_p m)
  (hQ_imp : q → condition_q m) : 
  1 - Real.sqrt 2 ≤ m ∧ m ≤ 1 + Real.sqrt 2 := 
sorry

end find_m_range_l2318_231862


namespace max_value_of_x_l2318_231816

theorem max_value_of_x (x y : ℝ) (h : x^2 + y^2 = 18 * x + 20 * y) : x ≤ 9 + Real.sqrt 181 :=
by
  sorry

end max_value_of_x_l2318_231816


namespace total_apples_l2318_231847

theorem total_apples (apples_per_person : ℕ) (num_people : ℕ) (h1 : apples_per_person = 25) (h2 : num_people = 6) : apples_per_person * num_people = 150 := by
  sorry

end total_apples_l2318_231847


namespace find_m_l2318_231886

theorem find_m (m : ℤ) (a := (3, m)) (b := (1, -2)) (h : a.1 * b.1 + a.2 * b.2 = b.1^2 + b.2^2) : m = -1 :=
sorry

end find_m_l2318_231886


namespace part_a_solution_part_b_solution_l2318_231807

-- Part (a) Statement in Lean 4
theorem part_a_solution (N : ℕ) (a b : ℕ) (h : N = a * 10^n + b * 10^(n-1)) :
  ∃ (m : ℕ), (N / 10 = m) -> m * 10 = N := sorry

-- Part (b) Statement in Lean 4
theorem part_b_solution (N : ℕ) (a b c : ℕ) (h : N = a * 10^n + b * 10^(n-1) + c * 10^(n-2)) :
  ∃ (m : ℕ), (N / 10^(n-1) = m) -> m * 10^(n-1) = N := sorry

end part_a_solution_part_b_solution_l2318_231807


namespace bus_ride_cost_l2318_231844

noncomputable def bus_cost : ℝ := 1.75

theorem bus_ride_cost (B T : ℝ) (h1 : T = B + 6.35) (h2 : T + B = 9.85) : B = bus_cost :=
by
  sorry

end bus_ride_cost_l2318_231844


namespace mashed_potatoes_suggestion_count_l2318_231860

def number_of_students_suggesting_bacon := 394
def extra_students_suggesting_mashed_potatoes := 63
def number_of_students_suggesting_mashed_potatoes := number_of_students_suggesting_bacon + extra_students_suggesting_mashed_potatoes

theorem mashed_potatoes_suggestion_count :
  number_of_students_suggesting_mashed_potatoes = 457 := by
  sorry

end mashed_potatoes_suggestion_count_l2318_231860


namespace slope_of_tangent_at_A_l2318_231806

def f (x : ℝ) : ℝ := x^2 + 3 * x

def f' (x : ℝ) : ℝ := 2 * x + 3

theorem slope_of_tangent_at_A : f' 2 = 7 := by
  sorry

end slope_of_tangent_at_A_l2318_231806


namespace contrapositive_of_sum_of_squares_l2318_231856

theorem contrapositive_of_sum_of_squares
  (a b : ℝ)
  (h : a ≠ 0 ∨ b ≠ 0) :
  a^2 + b^2 ≠ 0 := 
sorry

end contrapositive_of_sum_of_squares_l2318_231856


namespace time_ratio_l2318_231871

theorem time_ratio (distance : ℝ) (initial_time : ℝ) (new_speed : ℝ) :
  distance = 600 → initial_time = 5 → new_speed = 80 → (distance / new_speed) / initial_time = 1.5 :=
by
  intros hdist htime hspeed
  sorry

end time_ratio_l2318_231871


namespace trig_identity_l2318_231817

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 :=
by 
  sorry

end trig_identity_l2318_231817


namespace find_p_q_r_l2318_231872

theorem find_p_q_r  (t : ℝ) (p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
                    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = (p / q) - Real.sqrt r)
                    (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
                    (rel_prime : Nat.gcd p q = 1) : 
                    p + q + r = 5 := 
by
  sorry

end find_p_q_r_l2318_231872


namespace total_fencing_cost_is_5300_l2318_231808

-- Define the conditions
def length_more_than_breadth_condition (l b : ℕ) := l = b + 40
def fencing_cost_per_meter : ℝ := 26.50
def given_length : ℕ := 70

-- Define the perimeter calculation
def perimeter (l b : ℕ) := 2 * l + 2 * b

-- Define the total cost calculation
def total_cost (P : ℕ) (cost_per_meter : ℝ) := P * cost_per_meter

-- State the theorem to be proven
theorem total_fencing_cost_is_5300 (b : ℕ) (l := given_length) :
  length_more_than_breadth_condition l b →
  total_cost (perimeter l b) fencing_cost_per_meter = 5300 :=
by
  sorry

end total_fencing_cost_is_5300_l2318_231808


namespace probability_not_black_l2318_231826

theorem probability_not_black (white_balls black_balls red_balls : ℕ) (total_balls : ℕ) (non_black_balls : ℕ) :
  white_balls = 7 → black_balls = 6 → red_balls = 4 →
  total_balls = white_balls + black_balls + red_balls →
  non_black_balls = white_balls + red_balls →
  (non_black_balls / total_balls : ℚ) = 11 / 17 :=
by
  sorry

end probability_not_black_l2318_231826


namespace custom_op_2006_l2318_231858

def custom_op (n : ℕ) : ℕ := 
  match n with 
  | 0 => 1
  | (n+1) => 2 + custom_op n

theorem custom_op_2006 : custom_op 2005 = 4011 :=
by {
  sorry
}

end custom_op_2006_l2318_231858


namespace toys_calculation_l2318_231852

-- Define the number of toys each person has as variables
variables (Jason John Rachel : ℕ)

-- State the conditions
variables (h1 : Jason = 3 * John)
variables (h2 : John = Rachel + 6)
variables (h3 : Jason = 21)

-- Define the theorem to prove the number of toys Rachel has
theorem toys_calculation : Rachel = 1 :=
by {
  sorry
}

end toys_calculation_l2318_231852


namespace isosceles_triangle_perimeter_l2318_231854

theorem isosceles_triangle_perimeter : 
  ∀ a b c : ℝ, a^2 - 6 * a + 5 = 0 → b^2 - 6 * b + 5 = 0 → 
    (a = b ∨ b = c ∨ a = c) →
    (a + b > c ∧ b + c > a ∧ a + c > b) →
    a + b + c = 11 := 
by
  intros a b c ha hb hiso htri
  sorry

end isosceles_triangle_perimeter_l2318_231854


namespace initial_lives_l2318_231832

theorem initial_lives (x : ℕ) (h1 : x - 23 + 46 = 70) : x = 47 := 
by 
  sorry

end initial_lives_l2318_231832


namespace first_year_exceeds_two_million_l2318_231824

-- Definition of the initial R&D investment in 2015
def initial_investment : ℝ := 1.3

-- Definition of the annual growth rate
def growth_rate : ℝ := 1.12

-- Definition of the investment function for year n
def investment (n : ℕ) : ℝ := initial_investment * growth_rate ^ (n - 2015)

-- The problem statement to be proven
theorem first_year_exceeds_two_million : ∃ n : ℕ, n > 2015 ∧ investment n > 2 ∧ ∀ m : ℕ, (m < n ∧ m > 2015) → investment m ≤ 2 := by
  sorry

end first_year_exceeds_two_million_l2318_231824


namespace initial_bottle_caps_l2318_231889

variable (x : Nat)

theorem initial_bottle_caps (h : x + 3 = 29) : x = 26 := by
  sorry

end initial_bottle_caps_l2318_231889


namespace find_n_with_divisors_conditions_l2318_231881

theorem find_n_with_divisors_conditions :
  ∃ n : ℕ, 
    (∀ d : ℕ, d ∣ n → d ∈ [1, n] ∧ 
    (∃ a b c : ℕ, a = 1 ∧ b = d / a ∧ c = d / b ∧ b = 7 * a ∧ d = 10 + b)) →
    n = 2891 :=
by
  sorry

end find_n_with_divisors_conditions_l2318_231881


namespace air_conditioner_sale_price_l2318_231840

theorem air_conditioner_sale_price (P : ℝ) (d1 d2 : ℝ) (hP : P = 500) (hd1 : d1 = 0.10) (hd2 : d2 = 0.20) :
  ((P * (1 - d1)) * (1 - d2)) / P * 100 = 72 :=
by
  sorry

end air_conditioner_sale_price_l2318_231840


namespace range_of_a_l2318_231899

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1/x) + a

theorem range_of_a (a : ℝ) (h : f a 0 = a^2) : (f a 0 = f a 0 -> 0 ≤ a ∧ a ≤ 2) := by
  sorry

end range_of_a_l2318_231899


namespace secondTrain_speed_l2318_231829

/-
Conditions:
1. Two trains start from A and B and travel towards each other.
2. The distance between them is 1100 km.
3. At the time of their meeting, one train has traveled 100 km more than the other.
4. The first train's speed is 50 kmph.
-/

-- Let v be the speed of the second train
def secondTrainSpeed (v : ℝ) : Prop :=
  ∃ d : ℝ, 
    d > 0 ∧
    v > 0 ∧
    (d + (d - 100) = 1100) ∧
    ((d / 50) = ((d - 100) / v))

-- Here is the main theorem translating the problem statement:
theorem secondTrain_speed :
  secondTrainSpeed (250 / 6) :=
by
  sorry

end secondTrain_speed_l2318_231829


namespace find_E_l2318_231878

theorem find_E (A H S M E : ℕ) (h1 : A ≠ 0) (h2 : H ≠ 0) (h3 : S ≠ 0) (h4 : M ≠ 0) (h5 : E ≠ 0) 
  (cond1 : A + H = E)
  (cond2 : S + M = E)
  (cond3 : E = (A * M - S * H) / (M - H)) : 
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end find_E_l2318_231878


namespace remainder_division_of_product_l2318_231811

theorem remainder_division_of_product
  (h1 : 1225 % 12 = 1)
  (h2 : 1227 % 12 = 3) :
  ((1225 * 1227 * 1) % 12) = 3 :=
by
  sorry

end remainder_division_of_product_l2318_231811


namespace min_value_x3_y2_z_w2_l2318_231898

theorem min_value_x3_y2_z_w2 (x y z w : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < w)
  (h : (1/x) + (1/y) + (1/z) + (1/w) = 8) : x^3 * y^2 * z * w^2 ≥ 1/432 :=
by
  sorry

end min_value_x3_y2_z_w2_l2318_231898


namespace estimate_total_observations_in_interval_l2318_231866

def total_observations : ℕ := 1000
def sample_size : ℕ := 50
def frequency_in_sample : ℝ := 0.12

theorem estimate_total_observations_in_interval : 
  frequency_in_sample * (total_observations : ℝ) = 120 :=
by
  -- conditions defined above
  -- use given frequency to estimate the total observations in the interval
  -- actual proof omitted
  sorry

end estimate_total_observations_in_interval_l2318_231866


namespace arithmetic_sequence_sum_l2318_231802

theorem arithmetic_sequence_sum (b : ℕ → ℝ) (h_arith : ∀ n, b (n+1) - b n = b 2 - b 1) (h_b5 : b 5 = 2) :
  b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 = 18 := 
sorry

end arithmetic_sequence_sum_l2318_231802


namespace scooter_cost_l2318_231825

variable (saved needed total_cost : ℕ)

-- The conditions given in the problem
def greg_saved_57 : saved = 57 := sorry
def greg_needs_33_more : needed = 33 := sorry

-- The proof goal
theorem scooter_cost (h1 : saved = 57) (h2 : needed = 33) :
  total_cost = saved + needed → total_cost = 90 := by
  sorry

end scooter_cost_l2318_231825


namespace base_conversion_arithmetic_l2318_231870

theorem base_conversion_arithmetic :
  let b5 := 2013
  let b3 := 11
  let b6 := 3124
  let b7 := 4321
  (b5₅ / b3₃ - b6₆ + b7₇ : ℝ) = 898.5 :=
by sorry

end base_conversion_arithmetic_l2318_231870


namespace negation_of_exists_l2318_231818

theorem negation_of_exists (x : ℝ) (h : ∃ x : ℝ, x^2 - x + 1 ≤ 0) : 
  (∀ x : ℝ, x^2 - x + 1 > 0) :=
sorry

end negation_of_exists_l2318_231818


namespace max_sum_of_squares_eq_l2318_231821

theorem max_sum_of_squares_eq (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end max_sum_of_squares_eq_l2318_231821


namespace function_positive_for_x_gt_neg1_l2318_231845

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (3*x^2 + 6*x + 9)

theorem function_positive_for_x_gt_neg1 : ∀ (x : ℝ), x > -1 → f x > 0.5 :=
by
  sorry

end function_positive_for_x_gt_neg1_l2318_231845


namespace find_max_n_l2318_231813

variables {α : Type*} [LinearOrderedField α]

-- Define the sum S_n of the first n terms of an arithmetic sequence
noncomputable def S_n (a d : α) (n : ℕ) : α := 
  (n : α) / 2 * (2 * a + (n - 1) * d)

-- Given conditions
variable {a d : α}
axiom S11_pos : S_n a d 11 > 0
axiom S12_neg : S_n a d 12 < 0

theorem find_max_n : ∃ (n : ℕ), ∀ k < n, S_n a d k ≤ S_n a d n ∧ (k ≠ n → S_n a d k < S_n a d n) :=
sorry

end find_max_n_l2318_231813


namespace math_problem_l2318_231843

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 3) : a^(2008 : ℕ) + b^(2008 : ℕ) + c^(2008 : ℕ) = 3 :=
by 
  let h1' : a + b + c = 3 := h1
  let h2' : a^2 + b^2 + c^2 = 3 := h2
  sorry

end math_problem_l2318_231843


namespace find_b_if_even_function_l2318_231822

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem find_b_if_even_function (h : ∀ x : ℝ, f (-x) = f (x)) : b = 0 := by
  sorry

end find_b_if_even_function_l2318_231822


namespace value_two_stddev_below_mean_l2318_231867

def mean : ℝ := 16.2
def standard_deviation : ℝ := 2.3

theorem value_two_stddev_below_mean : mean - 2 * standard_deviation = 11.6 :=
by
  sorry

end value_two_stddev_below_mean_l2318_231867


namespace find_shirt_numbers_calculate_profit_l2318_231819

def total_shirts_condition (x y : ℕ) : Prop := x + y = 200
def total_cost_condition (x y : ℕ) : Prop := 25 * x + 15 * y = 3500
def profit_calculation (x y : ℕ) : ℕ := (50 - 25) * x + (35 - 15) * y

theorem find_shirt_numbers (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  x = 50 ∧ y = 150 :=
sorry

theorem calculate_profit (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  profit_calculation x y = 4250 :=
sorry

end find_shirt_numbers_calculate_profit_l2318_231819


namespace triangle_inequality_third_side_l2318_231873

theorem triangle_inequality_third_side (a : ℝ) (h1 : 3 + a > 7) (h2 : 7 + a > 3) (h3 : 3 + 7 > a) : 
  4 < a ∧ a < 10 :=
by sorry

end triangle_inequality_third_side_l2318_231873


namespace percentage_less_than_m_add_d_l2318_231869

def symmetric_about_mean (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P m - x = P m + x

def within_one_stdev (P : ℝ → ℝ) (m d : ℝ) : Prop :=
  P m - d = 0.68 ∧ P m + d = 0.68

theorem percentage_less_than_m_add_d 
  (P : ℝ → ℝ) (m d : ℝ) 
  (symm : symmetric_about_mean P m)
  (within_stdev : within_one_stdev P m d) : 
  ∃ f, f = 0.84 :=
by
  sorry

end percentage_less_than_m_add_d_l2318_231869


namespace abs_z1_purely_imaginary_l2318_231892

noncomputable def z1 (a : ℝ) : Complex := ⟨a, 2⟩
def z2 : Complex := ⟨2, -1⟩

theorem abs_z1_purely_imaginary (a : ℝ) (ha : 2 * a - 2 = 0) : Complex.abs (z1 a) = Real.sqrt 5 :=
by
  sorry

end abs_z1_purely_imaginary_l2318_231892


namespace min_sum_four_consecutive_nat_nums_l2318_231831

theorem min_sum_four_consecutive_nat_nums (a : ℕ) (h1 : a % 11 = 0) (h2 : (a + 1) % 7 = 0)
    (h3 : (a + 2) % 5 = 0) (h4 : (a + 3) % 3 = 0) : a + (a + 1) + (a + 2) + (a + 3) = 1458 :=
  sorry

end min_sum_four_consecutive_nat_nums_l2318_231831


namespace estimate_2_sqrt_5_l2318_231836

theorem estimate_2_sqrt_5: 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 :=
by
  sorry

end estimate_2_sqrt_5_l2318_231836


namespace wooden_parallelepiped_length_l2318_231849

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l2318_231849


namespace ext_9_implication_l2318_231883

theorem ext_9_implication (a b : ℝ) (h1 : 3 + 2 * a + b = 0) (h2 : 1 + a + b + a^2 = 10) : (2 : ℝ)^3 + a * (2 : ℝ)^2 + b * (2 : ℝ) + a^2 - 1 = 17 := by
  sorry

end ext_9_implication_l2318_231883


namespace prob_score_3_points_l2318_231841

-- Definitions for the probabilities
def probability_hit_A := 3/4
def score_hit_A := 1
def score_miss_A := -1

def probability_hit_B := 2/3
def score_hit_B := 2
def score_miss_B := 0

-- Conditional probabilities and their calculations
noncomputable def prob_scenario_1 : ℚ := 
  probability_hit_A * 2 * probability_hit_B * (1 - probability_hit_B)

noncomputable def prob_scenario_2 : ℚ := 
  (1 - probability_hit_A) * probability_hit_B^2

noncomputable def total_prob : ℚ := 
  prob_scenario_1 + prob_scenario_2

-- The final proof statement
theorem prob_score_3_points : total_prob = 4/9 := sorry

end prob_score_3_points_l2318_231841


namespace find_width_of_brick_l2318_231885

theorem find_width_of_brick (l h : ℝ) (SurfaceArea : ℝ) (w : ℝ) :
  l = 8 → h = 2 → SurfaceArea = 152 → 2*l*w + 2*l*h + 2*w*h = SurfaceArea → w = 6 :=
by
  intro l_value
  intro h_value
  intro SurfaceArea_value
  intro surface_area_equation
  sorry

end find_width_of_brick_l2318_231885


namespace youngest_child_age_l2318_231815

theorem youngest_child_age (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 55) : x = 7 := 
by
  sorry

end youngest_child_age_l2318_231815


namespace angle_in_first_quadrant_l2318_231864

theorem angle_in_first_quadrant (x : ℝ) (h1 : Real.tan x > 0) (h2 : Real.sin x + Real.cos x > 0) : 
  0 < Real.sin x ∧ 0 < Real.cos x := 
by 
  sorry

end angle_in_first_quadrant_l2318_231864


namespace digit_B_condition_l2318_231857

theorem digit_B_condition {B : ℕ} (h10 : ∃ d : ℕ, 58709310 = 10 * d)
  (h5 : ∃ e : ℕ, 58709310 = 5 * e)
  (h6 : ∃ f : ℕ, 58709310 = 6 * f)
  (h4 : ∃ g : ℕ, 58709310 = 4 * g)
  (h3 : ∃ h : ℕ, 58709310 = 3 * h)
  (h2 : ∃ i : ℕ, 58709310 = 2 * i) :
  B = 0 := by
  sorry

end digit_B_condition_l2318_231857


namespace correct_calculation_l2318_231839

variable (a : ℝ) -- assuming a ∈ ℝ

theorem correct_calculation : (a ^ 3) ^ 2 = a ^ 6 :=
by {
  sorry
}

end correct_calculation_l2318_231839


namespace work_together_l2318_231855

theorem work_together (A B : ℝ) (hA : A = 1/3) (hB : B = 1/6) : (1 / (A + B)) = 2 := by
  sorry

end work_together_l2318_231855


namespace shaniqua_haircuts_l2318_231884

theorem shaniqua_haircuts
  (H : ℕ) -- number of haircuts
  (haircut_income : ℕ) (style_income : ℕ)
  (total_styles : ℕ) (total_income : ℕ)
  (haircut_income_eq : haircut_income = 12)
  (style_income_eq : style_income = 25)
  (total_styles_eq : total_styles = 5)
  (total_income_eq : total_income = 221)
  (income_from_styles : ℕ := total_styles * style_income)
  (income_from_haircuts : ℕ := total_income - income_from_styles) :
  H = income_from_haircuts / haircut_income :=
sorry

end shaniqua_haircuts_l2318_231884


namespace domain_of_function_l2318_231834

theorem domain_of_function :
  ∀ x : ℝ, (2 - x > 0) ∧ (2 * x + 1 > 0) ↔ (-1 / 2 < x) ∧ (x < 2) :=
sorry

end domain_of_function_l2318_231834


namespace find_m_l2318_231861

open Set

def A (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 2}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem find_m (m : ℝ) :
  (A m ∩ B = ∅ ∧ A m ∪ B = B) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end find_m_l2318_231861


namespace parabola_midpoint_locus_minimum_slope_difference_exists_l2318_231810

open Real

def parabola_locus (x y : ℝ) : Prop :=
  x^2 = 4 * y

def slope_difference_condition (x1 x2 k1 k2 : ℝ) : Prop :=
  |k1 - k2| = 1

theorem parabola_midpoint_locus :
  ∀ (x y : ℝ), parabola_locus x y :=
by
  intros x y
  apply sorry

theorem minimum_slope_difference_exists :
  ∀ {x1 y1 x2 y2 k1 k2 : ℝ},
  slope_difference_condition x1 x2 k1 k2 :=
by
  intros x1 y1 x2 y2 k1 k2
  apply sorry

end parabola_midpoint_locus_minimum_slope_difference_exists_l2318_231810


namespace f_cos_x_l2318_231835

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 2 - Real.cos x ^ 2) : f (Real.cos x) = 2 + Real.sin x ^ 2 := by
  sorry

end f_cos_x_l2318_231835


namespace nonnegative_difference_roots_eq_12_l2318_231809

theorem nonnegative_difference_roots_eq_12 :
  ∀ (x : ℝ), (x^2 + 40 * x + 300 = -64) →
  ∃ (r₁ r₂ : ℝ), (x^2 + 40 * x + 364 = 0) ∧ 
  (r₁ = -26 ∧ r₂ = -14)
  ∧ (|r₁ - r₂| = 12) :=
by
  sorry

end nonnegative_difference_roots_eq_12_l2318_231809


namespace product_of_distinct_nonzero_real_numbers_l2318_231812

variable {x y : ℝ}

theorem product_of_distinct_nonzero_real_numbers (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 4 / x = y + 4 / y) : x * y = 4 := 
sorry

end product_of_distinct_nonzero_real_numbers_l2318_231812


namespace correct_equation_for_programmers_l2318_231893

theorem correct_equation_for_programmers (x : ℕ) 
  (hB : x > 0) 
  (programmer_b_speed : ℕ := x) 
  (programmer_a_speed : ℕ := 2 * x) 
  (data : ℕ := 2640) :
  (data / programmer_a_speed = data / programmer_b_speed - 120) :=
by
  -- sorry is used to skip the proof, focus on the statement
  sorry

end correct_equation_for_programmers_l2318_231893


namespace find_a_l2318_231803

variable (f g : ℝ → ℝ) (a : ℝ)

-- Conditions
axiom h1 : ∀ x, f x = a^x * g x
axiom h2 : ∀ x, g x ≠ 0
axiom h3 : ∀ x, f x * (deriv g x) > (deriv f x) * g x

-- Question and target proof
theorem find_a (h4 : (f 1) / (g 1) + (f (-1)) / (g (-1)) = 5 / 2) : a = 1 / 2 :=
by sorry

end find_a_l2318_231803


namespace machine_value_after_2_years_l2318_231894

section
def initial_value : ℝ := 1200
def depreciation_rate_year1 : ℝ := 0.10
def depreciation_rate_year2 : ℝ := 0.12
def repair_rate : ℝ := 0.03
def major_overhaul_rate : ℝ := 0.15

theorem machine_value_after_2_years :
  let value_after_repairs_2 := (initial_value * (1 - depreciation_rate_year1) + initial_value * repair_rate) * (1 - depreciation_rate_year2 + repair_rate)
  (value_after_repairs_2 * (1 - major_overhaul_rate)) = 863.23 := 
by
  -- proof here
  sorry
end

end machine_value_after_2_years_l2318_231894


namespace susan_ate_6_candies_l2318_231848

-- Definitions based on the problem conditions
def candies_tuesday := 3
def candies_thursday := 5
def candies_friday := 2
def candies_left := 4

-- The total number of candies bought
def total_candies_bought := candies_tuesday + candies_thursday + candies_friday

-- The number of candies eaten
def candies_eaten := total_candies_bought - candies_left

-- Theorem statement to prove that Susan ate 6 candies
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  -- Proof will be provided here
  sorry

end susan_ate_6_candies_l2318_231848


namespace log_xy_l2318_231846

-- Definitions from conditions
def log (z : ℝ) : ℝ := sorry -- Assume a definition of log function
variables (x y : ℝ)
axiom h1 : log (x^2 * y^2) = 1
axiom h2 : log (x^3 * y) = 2

-- The proof goal
theorem log_xy (x y : ℝ) (h1 : log (x^2 * y^2) = 1) (h2 : log (x^3 * y) = 2) : log (x * y) = 1/2 :=
sorry

end log_xy_l2318_231846


namespace larger_integer_value_l2318_231877

theorem larger_integer_value (x y : ℕ) (h1 : (4 * x)^2 - 2 * x = 8100) (h2 : x + 10 = 2 * y) : x = 22 :=
by
  sorry

end larger_integer_value_l2318_231877


namespace repeating_decimal_fraction_l2318_231887

noncomputable def repeating_decimal := 7 + ((789 : ℚ) / (10^4 - 1))

theorem repeating_decimal_fraction :
  repeating_decimal = (365 : ℚ) / 85 :=
by
  sorry

end repeating_decimal_fraction_l2318_231887


namespace lesser_fraction_l2318_231830

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3/4) (h₂ : x * y = 1/8) : min x y = 1/4 :=
by
  -- The proof would go here
  sorry

end lesser_fraction_l2318_231830


namespace symmetric_point_origin_l2318_231897

def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, -P.2)

theorem symmetric_point_origin :
  symmetric_point (3, -1) = (-3, 1) :=
by
  sorry

end symmetric_point_origin_l2318_231897


namespace find_different_weighted_coins_l2318_231800

-- Define the conditions and the theorem
def num_coins : Nat := 128
def weight_types : Nat := 2
def coins_of_each_weight : Nat := 64

theorem find_different_weighted_coins (weighings_at_most : Nat := 7) :
  ∃ (w1 w2 : Nat) (coins : Fin num_coins → Nat), w1 ≠ w2 ∧ 
  (∃ (pair : Fin num_coins × Fin num_coins), pair.fst ≠ pair.snd ∧ coins pair.fst ≠ coins pair.snd) :=
sorry

end find_different_weighted_coins_l2318_231800


namespace perimeter_is_36_l2318_231859

-- Define an equilateral triangle with a given side length
def equilateral_triangle_perimeter (side_length : ℝ) : ℝ :=
  3 * side_length

-- Given: The base of the equilateral triangle is 12 m
def base_length : ℝ := 12

-- Theorem: The perimeter of the equilateral triangle is 36 m
theorem perimeter_is_36 : equilateral_triangle_perimeter base_length = 36 :=
by
  -- Placeholder for the proof
  sorry

end perimeter_is_36_l2318_231859


namespace min_value_of_expression_l2318_231801

noncomputable def expression (x : ℝ) : ℝ := (15 - x) * (12 - x) * (15 + x) * (12 + x)

theorem min_value_of_expression :
  ∃ x : ℝ, (expression x) = -1640.25 :=
sorry

end min_value_of_expression_l2318_231801


namespace relationship_M_N_l2318_231868

def M : Set Int := {-1, 0, 1}
def N : Set Int := {x | ∃ a b : Int, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem relationship_M_N : N ⊆ M ∧ N ≠ M := by
  sorry

end relationship_M_N_l2318_231868


namespace debbie_total_tape_l2318_231896

def large_box_tape : ℕ := 4
def medium_box_tape : ℕ := 2
def small_box_tape : ℕ := 1
def label_tape : ℕ := 1

def large_boxes_packed : ℕ := 2
def medium_boxes_packed : ℕ := 8
def small_boxes_packed : ℕ := 5

def total_tape_used : ℕ := 
  (large_boxes_packed * (large_box_tape + label_tape)) +
  (medium_boxes_packed * (medium_box_tape + label_tape)) +
  (small_boxes_packed * (small_box_tape + label_tape))

theorem debbie_total_tape : total_tape_used = 44 := by
  sorry

end debbie_total_tape_l2318_231896


namespace polar_coordinates_to_rectangular_l2318_231895

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_coordinates_to_rectangular :
  polar_to_rectangular 10 (11 * Real.pi / 6) = (5 * Real.sqrt 3, -5) :=
by
  sorry

end polar_coordinates_to_rectangular_l2318_231895


namespace number_of_valid_sequences_l2318_231888

-- Define the sequence property
def sequence_property (b : Fin 10 → Fin 10) : Prop :=
  ∀ i : Fin 10, 2 ≤ i → (∃ j : Fin 10, j < i ∧ (b j = b i + 1 ∨ b j = b i - 1 ∨ b j = b i + 2 ∨ b j = b i - 2))

-- Define the set of such sequences
def valid_sequences : Set (Fin 10 → Fin 10) := {b | sequence_property b}

-- Define the number of such sequences
def number_of_sequences : Fin 512 :=
  sorry -- Proof omitted for brevity

-- The final statement
theorem number_of_valid_sequences : number_of_sequences = 512 :=
  sorry  -- Skip proof

end number_of_valid_sequences_l2318_231888


namespace length_of_one_side_of_regular_octagon_l2318_231838

theorem length_of_one_side_of_regular_octagon
  (a b : ℕ)
  (h_pentagon : a = 16)   -- Side length of regular pentagon
  (h_total_yarn_pentagon : b = 80)  -- Total yarn for pentagon
  (hpentagon_yarn_length : 5 * a = b)  -- Total yarn condition
  (hoctagon_total_sides : 8 = 8)   -- Number of sides of octagon
  (hoctagon_side_length : 10 = b / 8)  -- Side length condition for octagon
  : 10 = 10 :=
by
  sorry

end length_of_one_side_of_regular_octagon_l2318_231838


namespace value_of_f_f_3_l2318_231820

def f (x : ℝ) := 3 * x^2 + 3 * x - 2

theorem value_of_f_f_3 : f (f 3) = 3568 :=
by {
  -- Definition of f is already given in the conditions
  sorry
}

end value_of_f_f_3_l2318_231820


namespace product_of_b_product_of_values_l2318_231828

/-- 
If the distance between the points (3b, b+2) and (6, 3) is 3√5 units,
then the product of all possible values of b is -0.8.
-/
theorem product_of_b (b : ℝ)
  (h : (6 - 3 * b)^2 + (3 - (b + 2))^2 = (3 * Real.sqrt 5)^2) :
  b = 4 ∨ b = -0.2 := sorry

/--
The product of the values satisfying the theorem product_of_b is -0.8.
-/
theorem product_of_values : (4 : ℝ) * (-0.2) = -0.8 := 
by norm_num -- using built-in arithmetic simplification

end product_of_b_product_of_values_l2318_231828


namespace watermelon_slices_l2318_231833

theorem watermelon_slices (total_seeds slices_black seeds_white seeds_per_slice num_slices : ℕ)
  (h1 : seeds_black = 20)
  (h2 : seeds_white = 20)
  (h3 : seeds_per_slice = seeds_black + seeds_white)
  (h4 : total_seeds = 1600)
  (h5 : num_slices = total_seeds / seeds_per_slice) :
  num_slices = 40 :=
by
  sorry

end watermelon_slices_l2318_231833


namespace example_problem_l2318_231880

variable (a b c d : ℝ)

theorem example_problem :
  (a + (b + c - d) = a + b + c - d) ∧
  (a - (b - c + d) = a - b + c - d) ∧
  (a - b - (c - d) ≠ a - b - c - d) ∧
  (a + b - (-c - d) = a + b + c + d) :=
by {
  sorry
}

end example_problem_l2318_231880


namespace divisible_by_5_l2318_231882

theorem divisible_by_5 (x y : ℕ) (h1 : 2 * x^2 - 1 = y^15) (h2 : x > 1) : 5 ∣ x := sorry

end divisible_by_5_l2318_231882


namespace find_interval_for_inequality_l2318_231804

open Set

theorem find_interval_for_inequality :
  {x : ℝ | (1 / (x^2 + 2) > 4 / x + 21 / 10)} = Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end find_interval_for_inequality_l2318_231804


namespace expression_is_integer_l2318_231879

theorem expression_is_integer (m : ℕ) (hm : 0 < m) :
  ∃ k : ℤ, k = (m^4 / 24 + m^3 / 4 + 11*m^2 / 24 + m / 4 : ℚ) :=
by
  sorry

end expression_is_integer_l2318_231879


namespace Connie_total_markers_l2318_231837

/--
Connie has 41 red markers and 64 blue markers. 
We want to prove that the total number of markers Connie has is 105.
-/
theorem Connie_total_markers : 
  let red_markers := 41
  let blue_markers := 64
  let total_markers := red_markers + blue_markers
  total_markers = 105 :=
by
  sorry

end Connie_total_markers_l2318_231837


namespace find_x_l2318_231827

noncomputable def area_of_figure (x : ℝ) : ℝ :=
  let A_rectangle := 3 * x * 2 * x
  let A_square1 := x ^ 2
  let A_square2 := (4 * x) ^ 2
  let A_triangle := (3 * x * 2 * x) / 2
  A_rectangle + A_square1 + A_square2 + A_triangle

theorem find_x (x : ℝ) : area_of_figure x = 1250 → x = 6.93 :=
  sorry

end find_x_l2318_231827


namespace fraction_equality_l2318_231853

theorem fraction_equality (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 := 
sorry

end fraction_equality_l2318_231853


namespace arithmetic_sequence_product_l2318_231865

noncomputable def a_n (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sequence_product (a_1 d : ℤ) :
  (a_n 4 a_1 d) + (a_n 7 a_1 d) = 2 →
  (a_n 5 a_1 d) * (a_n 6 a_1 d) = -3 →
  a_1 * (a_n 10 a_1 d) = -323 :=
by
  sorry

end arithmetic_sequence_product_l2318_231865


namespace ad_equals_two_l2318_231851

noncomputable def geometric_sequence (a b c d : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c)

theorem ad_equals_two (a b c d : ℝ) 
  (h1 : geometric_sequence a b c d) 
  (h2 : ∃ (b c : ℝ), (1, 2) = (b, c) ∧ b = 1 ∧ c = 2) :
  a * d = 2 :=
by
  sorry

end ad_equals_two_l2318_231851


namespace zoo_visitors_l2318_231863

theorem zoo_visitors (visitors_friday : ℕ) 
  (h1 : 3 * visitors_friday = 3750) :
  visitors_friday = 1250 := 
sorry

end zoo_visitors_l2318_231863


namespace bigger_part_of_sum_and_linear_combination_l2318_231814

theorem bigger_part_of_sum_and_linear_combination (x y : ℕ) 
  (h1 : x + y = 24) 
  (h2 : 7 * x + 5 * y = 146) : x = 13 :=
by 
  sorry

end bigger_part_of_sum_and_linear_combination_l2318_231814


namespace heather_bicycled_distance_l2318_231890

def speed : ℕ := 8
def time : ℕ := 5
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

theorem heather_bicycled_distance : distance speed time = 40 := by
  sorry

end heather_bicycled_distance_l2318_231890


namespace find_units_digit_l2318_231874

theorem find_units_digit : 
  (7^1993 + 5^1993) % 10 = 2 :=
by
  sorry

end find_units_digit_l2318_231874
