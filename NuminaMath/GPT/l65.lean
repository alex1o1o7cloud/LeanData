import Mathlib

namespace NUMINAMATH_GPT_mrs_petersons_change_l65_6583

-- Define the conditions
def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def discount_rate : ℚ := 0.10
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

-- Formulate the proof statement
theorem mrs_petersons_change :
  let total_cost_before_discount := num_tumblers * cost_per_tumbler
  let discount_amount := total_cost_before_discount * discount_rate
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let total_amount_paid := num_bills * value_per_bill
  let change_received := total_amount_paid - total_cost_after_discount
  change_received = 95 := by sorry

end NUMINAMATH_GPT_mrs_petersons_change_l65_6583


namespace NUMINAMATH_GPT_valuing_fraction_l65_6513

variable {x y : ℚ}

theorem valuing_fraction (h : x / y = 1 / 2) : (x - y) / (x + y) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_valuing_fraction_l65_6513


namespace NUMINAMATH_GPT_distance_cycled_l65_6504

variable (v t d : ℝ)

theorem distance_cycled (h1 : d = v * t)
                        (h2 : d = (v + 1) * (3 * t / 4))
                        (h3 : d = (v - 1) * (t + 3)) :
                        d = 36 :=
by
  sorry

end NUMINAMATH_GPT_distance_cycled_l65_6504


namespace NUMINAMATH_GPT_pages_read_on_Monday_l65_6536

variable (P : Nat) (W : Nat)
def TotalPages : Nat := P + 12 + W

theorem pages_read_on_Monday :
  (TotalPages P W = 51) → (P = 39) :=
by
  sorry

end NUMINAMATH_GPT_pages_read_on_Monday_l65_6536


namespace NUMINAMATH_GPT_largest_value_of_y_l65_6594

theorem largest_value_of_y :
  (∃ x y : ℝ, x^2 + 3 * x * y - y^2 = 27 ∧ 3 * x^2 - x * y + y^2 = 27 ∧ y ≤ 3) → (∃ y : ℝ, y = 3) :=
by
  intro h
  obtain ⟨x, y, h1, h2, h3⟩ := h
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_largest_value_of_y_l65_6594


namespace NUMINAMATH_GPT_balls_into_boxes_l65_6525

theorem balls_into_boxes : (4 ^ 5 = 1024) :=
by
  -- The proof is omitted; the statement is required
  sorry

end NUMINAMATH_GPT_balls_into_boxes_l65_6525


namespace NUMINAMATH_GPT_inequality_proof_l65_6587

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / a + (c + a) / b + (a + b) / c ≥ ((a ^ 2 + b ^ 2 + c ^ 2) * (a * b + b * c + c * a)) / (a * b * c * (a + b + c)) + 3 := 
by
  -- Adding 'sorry' to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_inequality_proof_l65_6587


namespace NUMINAMATH_GPT_coordinates_of_A_l65_6508

-- Definition of the distance function for any point (x, y)
def distance_to_x_axis (x y : ℝ) : ℝ := abs y
def distance_to_y_axis (x y : ℝ) : ℝ := abs x

-- Point A's coordinates
def point_is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- The main theorem to prove
theorem coordinates_of_A :
  ∃ (x y : ℝ), 
  point_is_in_fourth_quadrant x y ∧ 
  distance_to_x_axis x y = 3 ∧ 
  distance_to_y_axis x y = 6 ∧ 
  (x, y) = (6, -3) :=
by 
  sorry

end NUMINAMATH_GPT_coordinates_of_A_l65_6508


namespace NUMINAMATH_GPT_range_of_a_l65_6528

variable (a x : ℝ)

def P : Prop := a < x ∧ x < a + 1
def q : Prop := x^2 - 7 * x + 10 ≤ 0

theorem range_of_a (h₁ : P a x → q x) (h₂ : ∃ x, q x ∧ ¬P a x) : 2 ≤ a ∧ a ≤ 4 := 
sorry

end NUMINAMATH_GPT_range_of_a_l65_6528


namespace NUMINAMATH_GPT_maximum_possible_value_of_k_l65_6521

theorem maximum_possible_value_of_k :
  ∀ (k : ℕ), 
    (∃ (x : ℕ → ℝ), 
      (∀ i j : ℕ, 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k → x i > 1 ∧ x i ≠ x j ∧ x i ^ ⌊x j⌋ = x j ^ ⌊x i⌋)) 
      → k ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_maximum_possible_value_of_k_l65_6521


namespace NUMINAMATH_GPT_find_a_l65_6523

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {1, 3, a}) (hB : B = {1, a^2 - a + 1}) (h_subset : B ⊆ A) :
  a = -1 ∨ a = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l65_6523


namespace NUMINAMATH_GPT_mul_neg_x_squared_cubed_l65_6552

theorem mul_neg_x_squared_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 :=
sorry

end NUMINAMATH_GPT_mul_neg_x_squared_cubed_l65_6552


namespace NUMINAMATH_GPT_line_does_not_pass_second_quadrant_l65_6548

theorem line_does_not_pass_second_quadrant (a : ℝ) (ha : a ≠ 0) :
  ∀ (x y : ℝ), (x - y - a^2 = 0) → ¬(x < 0 ∧ y > 0) :=
sorry

end NUMINAMATH_GPT_line_does_not_pass_second_quadrant_l65_6548


namespace NUMINAMATH_GPT_symmetric_about_x_axis_l65_6592

noncomputable def f (a x : ℝ) : ℝ := a - x^2
def g (x : ℝ) : ℝ := x + 1

theorem symmetric_about_x_axis (a : ℝ) :
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f a x = - g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_about_x_axis_l65_6592


namespace NUMINAMATH_GPT_frank_has_4_five_dollar_bills_l65_6530

theorem frank_has_4_five_dollar_bills
    (one_dollar_bills : ℕ := 7)
    (ten_dollar_bills : ℕ := 2)
    (twenty_dollar_bills : ℕ := 1)
    (change : ℕ := 4)
    (peanut_cost_per_pound : ℕ := 3)
    (days_in_week : ℕ := 7)
    (peanuts_per_day : ℕ := 3) :
    let initial_amount := (one_dollar_bills * 1) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)
    let total_peanuts_cost := (peanuts_per_day * days_in_week) * peanut_cost_per_pound
    let F := (total_peanuts_cost + change - initial_amount) / 5 
    F = 4 :=
by
  repeat { admit }


end NUMINAMATH_GPT_frank_has_4_five_dollar_bills_l65_6530


namespace NUMINAMATH_GPT_pear_counts_after_events_l65_6537

theorem pear_counts_after_events (Alyssa_picked Nancy_picked Carlos_picked : ℕ) (give_away : ℕ)
  (eat_fraction : ℚ) (share_fraction : ℚ) :
  Alyssa_picked = 42 →
  Nancy_picked = 17 →
  Carlos_picked = 25 →
  give_away = 5 →
  eat_fraction = 0.20 →
  share_fraction = 0.5 →
  ∃ (Alyssa_picked_final Nancy_picked_final Carlos_picked_final : ℕ),
    Alyssa_picked_final = 30 ∧
    Nancy_picked_final = 14 ∧
    Carlos_picked_final = 18 :=
by
  sorry

end NUMINAMATH_GPT_pear_counts_after_events_l65_6537


namespace NUMINAMATH_GPT_evaluate_expression_l65_6577

def my_star (A B : ℕ) : ℕ := (A + B) / 2
def my_hash (A B : ℕ) : ℕ := A * B + 1

theorem evaluate_expression : my_hash (my_star 4 6) 5 = 26 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l65_6577


namespace NUMINAMATH_GPT_percent_of_y_l65_6533

theorem percent_of_y (y : ℝ) (h : y > 0) : ((1 * y) / 20 + (3 * y) / 10) = (35/100) * y :=
by
  sorry

end NUMINAMATH_GPT_percent_of_y_l65_6533


namespace NUMINAMATH_GPT_distinct_x_sum_l65_6564

theorem distinct_x_sum (x y z : ℂ) 
(h1 : x + y * z = 9) 
(h2 : y + x * z = 12) 
(h3 : z + x * y = 12) : 
(x = 1 ∨ x = 3) ∧ (¬(x = 1 ∧ x = 3) → x ≠ 1 ∧ x ≠ 3) ∧ (1 + 3 = 4) :=
by
  sorry

end NUMINAMATH_GPT_distinct_x_sum_l65_6564


namespace NUMINAMATH_GPT_puppies_per_cage_l65_6586

theorem puppies_per_cage (initial_puppies sold_puppies cages remaining_puppies puppies_per_cage : ℕ)
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : cages = 3)
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 5 := by
  sorry

end NUMINAMATH_GPT_puppies_per_cage_l65_6586


namespace NUMINAMATH_GPT_tan_alpha_over_tan_beta_l65_6578

theorem tan_alpha_over_tan_beta (α β : ℝ) (h1 : Real.sin (α + β) = 2 / 3) (h2 : Real.sin (α - β) = 1 / 3) :
  (Real.tan α / Real.tan β = 3) :=
sorry

end NUMINAMATH_GPT_tan_alpha_over_tan_beta_l65_6578


namespace NUMINAMATH_GPT_total_distance_travelled_l65_6515

-- Definitions and propositions
def distance_first_hour : ℝ := 15
def distance_second_hour : ℝ := 18
def distance_third_hour : ℝ := 1.25 * distance_second_hour

-- Conditions based on the problem
axiom second_hour_distance : distance_second_hour = 18
axiom second_hour_20_percent_more : distance_second_hour = 1.2 * distance_first_hour
axiom third_hour_25_percent_more : distance_third_hour = 1.25 * distance_second_hour

-- Proof of the total distance James traveled
theorem total_distance_travelled : 
  distance_first_hour + distance_second_hour + distance_third_hour = 55.5 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_travelled_l65_6515


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t_eq_2_l65_6503

variable (t : ℝ)

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2 

theorem instantaneous_velocity_at_t_eq_2 :
  (deriv (displacement) 2) = 4 :=
sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t_eq_2_l65_6503


namespace NUMINAMATH_GPT_plant_height_after_year_l65_6509

theorem plant_height_after_year (current_height : ℝ) (monthly_growth : ℝ) (months_in_year : ℕ) (total_growth : ℝ)
  (h1 : current_height = 20)
  (h2 : monthly_growth = 5)
  (h3 : months_in_year = 12)
  (h4 : total_growth = monthly_growth * months_in_year) :
  current_height + total_growth = 80 :=
sorry

end NUMINAMATH_GPT_plant_height_after_year_l65_6509


namespace NUMINAMATH_GPT_max_lights_correct_l65_6543

def max_lights_on (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2

theorem max_lights_correct (n : ℕ) :
  max_lights_on n = if n % 2 = 0 then n^2 / 2 else (n^2 - 1) / 2 :=
by sorry

end NUMINAMATH_GPT_max_lights_correct_l65_6543


namespace NUMINAMATH_GPT_union_M_N_intersection_M_complement_N_l65_6557

open Set

variable (U : Set ℝ) (M N : Set ℝ)

-- Define the universal set
def is_universal_set (U : Set ℝ) : Prop :=
  U = univ

-- Define the set M
def is_set_M (M : Set ℝ) : Prop :=
  M = {x | ∃ y, y = (x - 2).sqrt}  -- or equivalently x ≥ 2

-- Define the set N
def is_set_N (N : Set ℝ) : Prop :=
  N = {x | x < 1 ∨ x > 3}

-- Define the complement of N in U
def complement_set_N (U N : Set ℝ) : Set ℝ :=
  U \ N

-- Prove M ∪ N = {x | x < 1 ∨ x ≥ 2}
theorem union_M_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∪ N = {x | x < 1 ∨ x ≥ 2} :=
  sorry

-- Prove M ∩ (complement of N in U) = {x | 2 ≤ x ≤ 3}
theorem intersection_M_complement_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∩ (complement_set_N U N) = {x | 2 ≤ x ∧ x ≤ 3} :=
  sorry

end NUMINAMATH_GPT_union_M_N_intersection_M_complement_N_l65_6557


namespace NUMINAMATH_GPT_stream_speed_l65_6585

theorem stream_speed (v : ℝ) (h1 : 36 > 0) (h2 : 80 > 0) (h3 : 40 > 0) (t_down : 80 / (36 + v) = 40 / (36 - v)) : v = 12 := 
by
  sorry

end NUMINAMATH_GPT_stream_speed_l65_6585


namespace NUMINAMATH_GPT_find_N_l65_6558

theorem find_N (a b c N : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = a + b) : N = 272 :=
sorry

end NUMINAMATH_GPT_find_N_l65_6558


namespace NUMINAMATH_GPT_age_relation_l65_6595

/--
Given that a woman is 42 years old and her daughter is 8 years old,
prove that in 9 years, the mother will be three times as old as her daughter.
-/
theorem age_relation (x : ℕ) (mother_age daughter_age : ℕ) 
  (h1 : mother_age = 42) (h2 : daughter_age = 8) 
  (h3 : 42 + x = 3 * (8 + x)) : 
  x = 9 :=
by
  sorry

end NUMINAMATH_GPT_age_relation_l65_6595


namespace NUMINAMATH_GPT_induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l65_6550

open Nat

theorem induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25 :
  ∀ n : ℕ, n > 0 → 25 ∣ (2^(n+2) * 3^n + 5*n - 4) :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l65_6550


namespace NUMINAMATH_GPT_determine_coefficients_l65_6519

theorem determine_coefficients (A B C : ℝ) 
  (h1 : 3 * A - 1 = 0)
  (h2 : 3 * A^2 + 3 * B = 0)
  (h3 : A^3 + 6 * A * B + 3 * C = 0) :
  A = 1 / 3 ∧ B = -1 / 9 ∧ C = 5 / 81 :=
by 
  sorry

end NUMINAMATH_GPT_determine_coefficients_l65_6519


namespace NUMINAMATH_GPT_tangent_line_at_P_no_zero_points_sum_of_zero_points_l65_6527

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

/-- Given that f(x) = ln(x) - 2x, prove that the tangent line at point P(1, -2) has the equation x + y + 1 = 0. -/
theorem tangent_line_at_P (a : ℝ) (h : a = 2) : ∀ x y : ℝ, x + y + 1 = 0 :=
sorry

/-- Show that for f(x) = ln(x) - ax, the function f(x) has no zero points if a > 1/e. -/
theorem no_zero_points (a : ℝ) (h : a > 1 / Real.exp 1) : ¬∃ x : ℝ, f x a = 0 :=
sorry

/-- For f(x) = ln(x) - ax and x1 ≠ x2 such that f(x1) = f(x2) = 0, prove that x1 + x2 > 2 / a. -/
theorem sum_of_zero_points (a x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) : x₁ + x₂ > 2 / a :=
sorry

end NUMINAMATH_GPT_tangent_line_at_P_no_zero_points_sum_of_zero_points_l65_6527


namespace NUMINAMATH_GPT_find_y_in_triangle_l65_6501

theorem find_y_in_triangle (BAC ABC BCA : ℝ) (y : ℝ) (h1 : BAC = 90)
  (h2 : ABC = 2 * y) (h3 : BCA = y - 10) : y = 100 / 3 :=
by
  -- The proof will be left as sorry
  sorry

end NUMINAMATH_GPT_find_y_in_triangle_l65_6501


namespace NUMINAMATH_GPT_range_of_m_l65_6559

theorem range_of_m (m : ℝ) :
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * m * x_0 + m + 2 < 0) ↔ (-1 : ℝ) ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l65_6559


namespace NUMINAMATH_GPT_power_function_value_l65_6520

-- Given conditions
def f : ℝ → ℝ := fun x => x^(1 / 3)

theorem power_function_value :
  f (Real.log 5 / (Real.log 2 * 8) + Real.log 160 / (Real.log (1 / 2))) = -2 := by
  sorry

end NUMINAMATH_GPT_power_function_value_l65_6520


namespace NUMINAMATH_GPT_total_money_divided_l65_6591

noncomputable def children_share_total (A B E : ℕ) :=
  (12 * A = 8 * B ∧ 8 * B = 6 * E ∧ A = 84) → 
  A + B + E = 378

theorem total_money_divided (A B E : ℕ) : children_share_total A B E :=
by
  intros h
  sorry

end NUMINAMATH_GPT_total_money_divided_l65_6591


namespace NUMINAMATH_GPT_bakery_item_count_l65_6568

theorem bakery_item_count : ∃ (s c : ℕ), 5 * s + 25 * c = 500 ∧ s + c = 12 := by
  sorry

end NUMINAMATH_GPT_bakery_item_count_l65_6568


namespace NUMINAMATH_GPT_jacob_calories_l65_6549

theorem jacob_calories (goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) 
  (h_goal : goal = 1800) 
  (h_breakfast : breakfast = 400) 
  (h_lunch : lunch = 900) 
  (h_dinner : dinner = 1100) : 
  (breakfast + lunch + dinner) - goal = 600 :=
by 
  sorry

end NUMINAMATH_GPT_jacob_calories_l65_6549


namespace NUMINAMATH_GPT_check_random_event_l65_6598

def random_event (A B C D : Prop) : Prop := ∃ E, D = E

def event_A : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_B : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_C : Prop :=
  ∀ (probability : ℝ), probability = 1

def event_D : Prop :=
  ∀ (probability : ℝ), 0 < probability ∧ probability < 1

theorem check_random_event :
  random_event event_A event_B event_C event_D :=
sorry

end NUMINAMATH_GPT_check_random_event_l65_6598


namespace NUMINAMATH_GPT_acute_angle_tan_eq_one_l65_6593

theorem acute_angle_tan_eq_one (A : ℝ) (h1 : 0 < A ∧ A < π / 2) (h2 : Real.tan A = 1) : A = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_tan_eq_one_l65_6593


namespace NUMINAMATH_GPT_percentage_calculation_l65_6538

-- Define total and part amounts
def total_amount : ℕ := 800
def part_amount : ℕ := 200

-- Define the percentage calculation
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Theorem to show the percentage is 25%
theorem percentage_calculation :
  percentage part_amount total_amount = 25 :=
sorry

end NUMINAMATH_GPT_percentage_calculation_l65_6538


namespace NUMINAMATH_GPT_steve_halfway_time_longer_than_danny_l65_6576

theorem steve_halfway_time_longer_than_danny 
  (T_D : ℝ) (T_S : ℝ)
  (h1 : T_D = 33) 
  (h2 : T_S = 2 * T_D):
  (T_S / 2) - (T_D / 2) = 16.5 :=
by sorry

end NUMINAMATH_GPT_steve_halfway_time_longer_than_danny_l65_6576


namespace NUMINAMATH_GPT_num_candidates_above_630_l65_6540

noncomputable def normal_distribution_candidates : Prop :=
  let μ := 530
  let σ := 50
  let total_candidates := 1000
  let probability_above_630 := (1 - 0.954) / 2  -- Probability of scoring above 630
  let expected_candidates_above_630 := total_candidates * probability_above_630
  expected_candidates_above_630 = 23

theorem num_candidates_above_630 : normal_distribution_candidates := by
  sorry

end NUMINAMATH_GPT_num_candidates_above_630_l65_6540


namespace NUMINAMATH_GPT_graph_of_g_contains_1_0_and_sum_l65_6566

noncomputable def f : ℝ → ℝ := sorry

def g (x y : ℝ) : Prop := 3 * y = 2 * f (3 * x) + 4

theorem graph_of_g_contains_1_0_and_sum :
  f 3 = -2 → g 1 0 ∧ (1 + 0 = 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_graph_of_g_contains_1_0_and_sum_l65_6566


namespace NUMINAMATH_GPT_area_ratio_of_squares_l65_6544

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 16 * b) : a ^ 2 = 16 * b ^ 2 := by
  sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l65_6544


namespace NUMINAMATH_GPT_fraction_value_l65_6511

theorem fraction_value (a b c : ℕ) (h1 : a = 2200) (h2 : b = 2096) (h3 : c = 121) :
    (a - b)^2 / c = 89 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l65_6511


namespace NUMINAMATH_GPT_largest_variable_l65_6532

theorem largest_variable {x y z w : ℤ} 
  (h1 : x + 3 = y - 4)
  (h2 : x + 3 = z + 2)
  (h3 : x + 3 = w - 1) :
  y > x ∧ y > z ∧ y > w :=
by sorry

end NUMINAMATH_GPT_largest_variable_l65_6532


namespace NUMINAMATH_GPT_correct_operations_result_greater_than_1000_l65_6574

theorem correct_operations_result_greater_than_1000
    (finalResultIncorrectOps : ℕ)
    (originalNumber : ℕ)
    (finalResultCorrectOps : ℕ)
    (H1 : finalResultIncorrectOps = 40)
    (H2 : originalNumber = (finalResultIncorrectOps + 12) * 8)
    (H3 : finalResultCorrectOps = (originalNumber * 8) + (2 * originalNumber) + 12) :
  finalResultCorrectOps > 1000 := 
sorry

end NUMINAMATH_GPT_correct_operations_result_greater_than_1000_l65_6574


namespace NUMINAMATH_GPT_paint_cans_needed_l65_6561

-- Conditions as definitions
def bedrooms : ℕ := 3
def other_rooms : ℕ := 2 * bedrooms
def paint_per_room : ℕ := 2
def color_can_capacity : ℕ := 1
def white_can_capacity : ℕ := 3

-- Total gallons needed
def total_color_gallons_needed : ℕ := paint_per_room * bedrooms
def total_white_gallons_needed : ℕ := paint_per_room * other_rooms

-- Total cans needed
def total_color_cans_needed : ℕ := total_color_gallons_needed / color_can_capacity
def total_white_cans_needed : ℕ := total_white_gallons_needed / white_can_capacity
def total_cans_needed : ℕ := total_color_cans_needed + total_white_cans_needed

theorem paint_cans_needed : total_cans_needed = 10 := by
  -- Proof steps (skipped) to show total_cans_needed = 10
  sorry

end NUMINAMATH_GPT_paint_cans_needed_l65_6561


namespace NUMINAMATH_GPT_gcd_of_polynomials_l65_6506

theorem gcd_of_polynomials (b : ℤ) (h : b % 1620 = 0) : Int.gcd (b^2 + 11 * b + 36) (b + 6) = 6 := 
by
  sorry

end NUMINAMATH_GPT_gcd_of_polynomials_l65_6506


namespace NUMINAMATH_GPT_joseph_power_cost_ratio_l65_6582

theorem joseph_power_cost_ratio
  (electric_oven_cost : ℝ)
  (total_cost : ℝ)
  (water_heater_cost : ℝ)
  (refrigerator_cost : ℝ)
  (H1 : electric_oven_cost = 500)
  (H2 : 2 * water_heater_cost = electric_oven_cost)
  (H3 : refrigerator_cost + water_heater_cost + electric_oven_cost = total_cost)
  (H4 : total_cost = 1500):
  (refrigerator_cost / water_heater_cost) = 3 := sorry

end NUMINAMATH_GPT_joseph_power_cost_ratio_l65_6582


namespace NUMINAMATH_GPT_value_of_expression_l65_6546

theorem value_of_expression (x y z : ℝ) (h : (x * y * z) / (|x * y * z|) = 1) :
  (|x| / x + y / |y| + |z| / z) = 3 ∨ (|x| / x + y / |y| + |z| / z) = -1 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l65_6546


namespace NUMINAMATH_GPT_principal_amount_borrowed_l65_6526

theorem principal_amount_borrowed 
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 9000) 
  (h2 : R = 0.12) 
  (h3 : T = 3) 
  (h4 : SI = P * R * T) : 
  P = 25000 :=
sorry

end NUMINAMATH_GPT_principal_amount_borrowed_l65_6526


namespace NUMINAMATH_GPT_lcm_gcd_product_difference_l65_6542
open Nat

theorem lcm_gcd_product_difference :
  (Nat.lcm 12 9) * (Nat.gcd 12 9) - (Nat.gcd 15 9) = 105 :=
by
  sorry

end NUMINAMATH_GPT_lcm_gcd_product_difference_l65_6542


namespace NUMINAMATH_GPT_geo_series_sum_eight_terms_l65_6599

theorem geo_series_sum_eight_terms :
  let a_0 := 1 / 3
  let r := 1 / 3 
  let S_8 := a_0 * (1 - r^8) / (1 - r)
  S_8 = 3280 / 6561 :=
by
  /- :: Proof Steps Omitted. -/
  sorry

end NUMINAMATH_GPT_geo_series_sum_eight_terms_l65_6599


namespace NUMINAMATH_GPT_contrapositive_l65_6569

theorem contrapositive (x : ℝ) (h : x^2 ≥ 1) : x ≥ 0 ∨ x ≤ -1 :=
sorry

end NUMINAMATH_GPT_contrapositive_l65_6569


namespace NUMINAMATH_GPT_gather_half_of_nuts_l65_6541

open Nat

theorem gather_half_of_nuts (a b c : ℕ) (h₀ : (a + b + c) % 2 = 0) : ∃ k, k = (a + b + c) / 2 :=
  sorry

end NUMINAMATH_GPT_gather_half_of_nuts_l65_6541


namespace NUMINAMATH_GPT_union_of_A_and_B_l65_6560

variable (a b : ℕ)

def A : Set ℕ := {3, 2^a}
def B : Set ℕ := {a, b}
def intersection_condition : A a ∩ B a b = {2} := by sorry

theorem union_of_A_and_B (h : A a ∩ B a b = {2}) : 
  A a ∪ B a b = {1, 2, 3} := by sorry

end NUMINAMATH_GPT_union_of_A_and_B_l65_6560


namespace NUMINAMATH_GPT_range_of_c_monotonicity_of_g_l65_6554

noncomputable def f (x: ℝ) : ℝ := 2 * Real.log x + 1

theorem range_of_c (c: ℝ) : (∀ x > 0, f x ≤ 2 * x + c) → c ≥ -1 := by
  sorry

noncomputable def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

theorem monotonicity_of_g (a: ℝ) (ha: a > 0) : 
  (∀ x > 0, x ≠ a → ((x < a → g x a < g a a) ∧ (x > a → g x a < g a a))) := by
  sorry

end NUMINAMATH_GPT_range_of_c_monotonicity_of_g_l65_6554


namespace NUMINAMATH_GPT_fifth_term_of_sequence_is_31_l65_6539

namespace SequenceProof

def sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = 2 * a (n - 1) + 1

theorem fifth_term_of_sequence_is_31 :
  ∃ a : ℕ → ℕ, sequence a ∧ a 5 = 31 :=
by
  sorry

end SequenceProof

end NUMINAMATH_GPT_fifth_term_of_sequence_is_31_l65_6539


namespace NUMINAMATH_GPT_tagged_fish_in_second_catch_l65_6556

-- Definitions and conditions
def total_fish_in_pond : ℕ := 1750
def tagged_fish_initial : ℕ := 70
def fish_caught_second_time : ℕ := 50
def ratio_tagged_fish : ℚ := tagged_fish_initial / total_fish_in_pond

-- Theorem statement
theorem tagged_fish_in_second_catch (T : ℕ) : (T : ℚ) / fish_caught_second_time = ratio_tagged_fish → T = 2 :=
by
  sorry

end NUMINAMATH_GPT_tagged_fish_in_second_catch_l65_6556


namespace NUMINAMATH_GPT_log_expression_equality_l65_6547

theorem log_expression_equality : 
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) + 
  (Real.log 8 / Real.log 4) + 
  2 = 11 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_log_expression_equality_l65_6547


namespace NUMINAMATH_GPT_avg_class_l65_6512

-- Problem definitions
def total_students : ℕ := 40
def num_students_95 : ℕ := 8
def num_students_0 : ℕ := 5
def num_students_70 : ℕ := 10
def avg_remaining_students : ℝ := 50

-- Assuming we have these marks
def marks_95 : ℝ := 95
def marks_0 : ℝ := 0
def marks_70 : ℝ := 70

-- We need to prove that the total average is 57.75 given the above conditions
theorem avg_class (h1 : total_students = 40)
                  (h2 : num_students_95 = 8)
                  (h3 : num_students_0 = 5)
                  (h4 : num_students_70 = 10)
                  (h5 : avg_remaining_students = 50)
                  (h6 : marks_95 = 95)
                  (h7 : marks_0 = 0)
                  (h8 : marks_70 = 70) :
                  (8 * 95 + 5 * 0 + 10 * 70 + 50 * (40 - (8 + 5 + 10))) / 40 = 57.75 :=
by sorry

end NUMINAMATH_GPT_avg_class_l65_6512


namespace NUMINAMATH_GPT_commercial_break_duration_l65_6510

theorem commercial_break_duration (n1 n2 m1 m2 : ℕ) (h1 : n1 = 3) (h2 : m1 = 5) (h3 : n2 = 11) (h4 : m2 = 2) :
  n1 * m1 + n2 * m2 = 37 :=
by
  -- Here, in a real proof, we would substitute and show the calculations.
  sorry

end NUMINAMATH_GPT_commercial_break_duration_l65_6510


namespace NUMINAMATH_GPT_larger_integer_is_neg4_l65_6580

-- Definitions of the integers used in the problem
variables (x y : ℤ)

-- Conditions given in the problem
def condition1 : x + y = -9 := sorry
def condition2 : x - y = 1 := sorry

-- The theorem to prove
theorem larger_integer_is_neg4 (h1 : x + y = -9) (h2 : x - y = 1) : x = -4 := 
sorry

end NUMINAMATH_GPT_larger_integer_is_neg4_l65_6580


namespace NUMINAMATH_GPT_sin_alpha_value_l65_6573

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 4) = (7 * Real.sqrt 2) / 10)
  (h2 : Real.cos (2 * α) = 7 / 25) : 
  Real.sin α = 3 / 5 :=
sorry

end NUMINAMATH_GPT_sin_alpha_value_l65_6573


namespace NUMINAMATH_GPT_inequality_system_solution_l65_6590

theorem inequality_system_solution {x : ℝ} (h1 : 2 * x - 1 < x + 5) (h2 : (x + 1)/3 < x - 1) : 2 < x ∧ x < 6 :=
by
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l65_6590


namespace NUMINAMATH_GPT_sam_final_investment_l65_6571

-- Definitions based on conditions
def initial_investment : ℝ := 10000
def first_interest_rate : ℝ := 0.20
def years_first_period : ℕ := 3
def triple_amount : ℕ := 3
def second_interest_rate : ℝ := 0.15
def years_second_period : ℕ := 1

-- Lean function to accumulate investment with compound interest
def compound_interest (P r: ℝ) (n: ℕ) : ℝ := P * (1 + r) ^ n

-- Sam's investment calculations
def amount_after_3_years : ℝ := compound_interest initial_investment first_interest_rate years_first_period
def new_investment : ℝ := triple_amount * amount_after_3_years
def final_amount : ℝ := compound_interest new_investment second_interest_rate years_second_period

-- Proof goal (statement with the proof skipped)
theorem sam_final_investment : final_amount = 59616 := by
  sorry

end NUMINAMATH_GPT_sam_final_investment_l65_6571


namespace NUMINAMATH_GPT_range_of_m_for_roots_greater_than_1_l65_6551

theorem range_of_m_for_roots_greater_than_1:
  ∀ m : ℝ, 
  (∀ x : ℝ, 8 * x^2 - (m - 1) * x + (m - 7) = 0 → 1 < x) ↔ 25 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_roots_greater_than_1_l65_6551


namespace NUMINAMATH_GPT_largest_five_digit_negative_int_congruent_mod_23_l65_6534

theorem largest_five_digit_negative_int_congruent_mod_23 :
  ∃ n : ℤ, 23 * n + 1 < -9999 ∧ 23 * n + 1 = -9994 := 
sorry

end NUMINAMATH_GPT_largest_five_digit_negative_int_congruent_mod_23_l65_6534


namespace NUMINAMATH_GPT_polygon_sides_eq_nine_l65_6553

theorem polygon_sides_eq_nine (n : ℕ) (h : n - 1 = 8) : n = 9 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_nine_l65_6553


namespace NUMINAMATH_GPT_total_shirts_l65_6572

def initial_shirts : ℕ := 9
def new_shirts : ℕ := 8

theorem total_shirts : initial_shirts + new_shirts = 17 := by
  sorry

end NUMINAMATH_GPT_total_shirts_l65_6572


namespace NUMINAMATH_GPT_middle_dimension_of_crate_l65_6518

theorem middle_dimension_of_crate (middle_dimension : ℝ) : 
    (∀ r : ℝ, r = 5 → ∃ w h l : ℝ, w = 5 ∧ h = 12 ∧ l = middle_dimension ∧
        (diameter = 2 * r ∧ diameter ≤ middle_dimension ∧ h ≥ 12)) → 
    middle_dimension = 10 :=
by
  sorry

end NUMINAMATH_GPT_middle_dimension_of_crate_l65_6518


namespace NUMINAMATH_GPT_non_monotonic_m_range_l65_6516

theorem non_monotonic_m_range (m : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 2, (3 * x^2 + 2 * x + m = 0)) →
  m ∈ Set.Ioo (-16 : ℝ) (1/3 : ℝ) :=
sorry

end NUMINAMATH_GPT_non_monotonic_m_range_l65_6516


namespace NUMINAMATH_GPT_range_of_a_l65_6562

noncomputable def function_with_extreme_at_zero_only (a b : ℝ) : Prop :=
∀ x : ℝ, x ≠ 0 → 4 * x^2 + 3 * a * x + 4 > 0

theorem range_of_a (a b : ℝ) (h : function_with_extreme_at_zero_only a b) : 
  -8 / 3 ≤ a ∧ a ≤ 8 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l65_6562


namespace NUMINAMATH_GPT_problem_statements_l65_6502

theorem problem_statements :
  let S1 := ∀ (x : ℤ) (k : ℤ), x = 2 * k + 1 → (x % 2 = 1)
  let S2 := (∀ (x : ℝ), x > 2 → x > 1) 
            ∧ (∀ (x : ℝ), x > 1 → (x ≥ 2 ∨ x < 2)) 
  let S3 := ∀ (x : ℝ), ¬(∃ (x : ℝ), ∃ (y : ℝ), y = x^2 + 1 ∧ x = y)
  let S4 := ¬(∀ (x : ℝ), x > 1 → x^2 - x > 0) → (∃ (x : ℝ), x > 1 ∧ x^2 - x ≤ 0)
  (S1 ∧ S2 ∧ S3 ∧ ¬S4) := by
    sorry

end NUMINAMATH_GPT_problem_statements_l65_6502


namespace NUMINAMATH_GPT_find_diagonal_length_l65_6555

noncomputable def parallelepiped_diagonal_length 
  (s : ℝ) -- Side length of square face
  (h : ℝ) -- Length of vertical edge
  (θ : ℝ) -- Angle between vertical edge and square face edges
  (hsq : s = 5) -- Length of side of the square face ABCD
  (hedge : h = 5) -- Length of vertical edge AA1
  (θdeg : θ = 60) -- Angle in degrees
  : ℝ :=
5 * Real.sqrt 3

-- The main theorem to be proved
theorem find_diagonal_length
  (s : ℝ)
  (h : ℝ)
  (θ : ℝ)
  (hsq : s = 5)
  (hedge : h = 5)
  (θdeg : θ = 60)
  : parallelepiped_diagonal_length s h θ hsq hedge θdeg = 5 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_find_diagonal_length_l65_6555


namespace NUMINAMATH_GPT_probability_both_tell_truth_l65_6524

theorem probability_both_tell_truth (pA pB : ℝ) (hA : pA = 0.80) (hB : pB = 0.60) : pA * pB = 0.48 :=
by
  subst hA
  subst hB
  sorry

end NUMINAMATH_GPT_probability_both_tell_truth_l65_6524


namespace NUMINAMATH_GPT_resulting_surface_area_l65_6545

-- Defining the initial condition for the cube structure
def cube_surface_area (side_length : ℕ) : ℕ :=
  6 * side_length^2

-- Defining the structure and the modifications
def initial_structure : ℕ :=
  64 * (cube_surface_area 2)

def removed_cubes_exposure : ℕ :=
  4 * (cube_surface_area 2)

-- The final lean statement to prove the surface area after removing central cubes
theorem resulting_surface_area : initial_structure + removed_cubes_exposure = 1632 := by
  sorry

end NUMINAMATH_GPT_resulting_surface_area_l65_6545


namespace NUMINAMATH_GPT_value_of_expression_l65_6500

theorem value_of_expression (a b c : ℚ) (h1 : a * b * c < 0) (h2 : a + b + c = 0) :
    (a - b - c) / |a| + (b - c - a) / |b| + (c - a - b) / |c| = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l65_6500


namespace NUMINAMATH_GPT_ratio_XZ_ZY_equals_one_l65_6581

theorem ratio_XZ_ZY_equals_one (A : ℕ) (B : ℕ) (C : ℕ) (total_area : ℕ) (area_bisected : ℕ)
  (decagon_area : total_area = 12) (halves_area : area_bisected = 6)
  (above_LZ : A + B = area_bisected) (below_LZ : C + D = area_bisected)
  (symmetry : XZ = ZY) :
  (XZ / ZY = 1) := 
by
  sorry

end NUMINAMATH_GPT_ratio_XZ_ZY_equals_one_l65_6581


namespace NUMINAMATH_GPT_sum_of_solutions_l65_6570

theorem sum_of_solutions (x : ℝ) (h : ∀ x, (x ≠ 1) ∧ (x ≠ -1) → ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) )) : 
  (∀ x, (x ≠ 1) ∧ (x ≠ -1) → -15 * x / (x^2 - 1) = 3 * x / (x+1) - 9 / (x-1)) → (x = ( -1 + Real.sqrt 13 ) / 2 ∨ x = ( -1 - Real.sqrt 13 ) / 2) → (x + ( -x ) = -1) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l65_6570


namespace NUMINAMATH_GPT_sum_of_x_and_y_l65_6596

theorem sum_of_x_and_y (x y : ℝ) (h1 : x + abs x + y = 5) (h2 : x + abs y - y = 6) : x + y = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l65_6596


namespace NUMINAMATH_GPT_batteries_C_equivalent_l65_6575

variables (x y z W : ℝ)

-- Conditions
def cond1 := 4 * x + 18 * y + 16 * z = W * z
def cond2 := 2 * x + 15 * y + 24 * z = W * z
def cond3 := 6 * x + 12 * y + 20 * z = W * z

-- Equivalent statement to prove
theorem batteries_C_equivalent (h1 : cond1 x y z W) (h2 : cond2 x y z W) (h3 : cond3 x y z W) : W = 48 :=
sorry

end NUMINAMATH_GPT_batteries_C_equivalent_l65_6575


namespace NUMINAMATH_GPT_roots_polynomial_sum_squares_l65_6597

theorem roots_polynomial_sum_squares (p q r : ℝ) 
  (h_roots : ∀ x : ℝ, x^3 - 15 * x^2 + 25 * x - 10 = 0 → x = p ∨ x = q ∨ x = r) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 350 := 
by {
  sorry
}

end NUMINAMATH_GPT_roots_polynomial_sum_squares_l65_6597


namespace NUMINAMATH_GPT_avg_speed_is_40_l65_6505

noncomputable def average_speed (x : ℝ) : ℝ :=
  let time1 := x / 40
  let time2 := 2 * x / 20
  let total_time := time1 + time2
  let total_distance := 5 * x
  total_distance / total_time

theorem avg_speed_is_40 (x : ℝ) (hx : x > 0) :
  average_speed x = 40 := by
  sorry

end NUMINAMATH_GPT_avg_speed_is_40_l65_6505


namespace NUMINAMATH_GPT_max_k_possible_l65_6579

-- Given the sequence formed by writing all three-digit numbers from 100 to 999 consecutively
def digits_sequence : List Nat := List.join (List.map (fun n => [n / 100, (n / 10) % 10, n % 10]) (List.range' 100 (999 - 100 + 1)))

-- Function to get a k-digit number from the sequence
def get_k_digit_number (seq : List Nat) (start k : Nat) : List Nat := seq.drop start |>.take k

-- Statement to prove the maximum k
theorem max_k_possible : ∃ k : Nat, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 5 = get_k_digit_number digits_sequence start2 5) ∧ (¬ ∃ k' > 5, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 k' = get_k_digit_number digits_sequence start2 k')) :=
sorry

end NUMINAMATH_GPT_max_k_possible_l65_6579


namespace NUMINAMATH_GPT_avg_remaining_two_l65_6514

theorem avg_remaining_two (avg5 avg3 : ℝ) (h1 : avg5 = 12) (h2 : avg3 = 4) : (5 * avg5 - 3 * avg3) / 2 = 24 :=
by sorry

end NUMINAMATH_GPT_avg_remaining_two_l65_6514


namespace NUMINAMATH_GPT_complete_square_l65_6529

theorem complete_square (a b c : ℕ) (h : 49 * x ^ 2 + 70 * x - 121 = 0) :
  a = 7 ∧ b = 5 ∧ c = 146 ∧ a + b + c = 158 :=
by sorry

end NUMINAMATH_GPT_complete_square_l65_6529


namespace NUMINAMATH_GPT_binom_10_0_eq_1_l65_6567

theorem binom_10_0_eq_1 :
  (Nat.choose 10 0) = 1 :=
by
  sorry

end NUMINAMATH_GPT_binom_10_0_eq_1_l65_6567


namespace NUMINAMATH_GPT_real_roots_iff_integer_roots_iff_l65_6584

noncomputable def discriminant (k : ℝ) : ℝ := (k + 1)^2 - 4 * k * (k - 1)

theorem real_roots_iff (k : ℝ) : 
  (discriminant k ≥ 0) ↔ (∃ (a b : ℝ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) := sorry

theorem integer_roots_iff (k : ℝ) : 
  (∃ (a b : ℤ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) ↔ 
  (k = 0 ∨ k = 1 ∨ k = -1/7) := sorry

-- These theorems need to be proven within Lean 4 itself

end NUMINAMATH_GPT_real_roots_iff_integer_roots_iff_l65_6584


namespace NUMINAMATH_GPT_largest_number_of_stores_visited_l65_6531

theorem largest_number_of_stores_visited
  (stores : ℕ) (total_visits : ℕ) (total_peopled_shopping : ℕ)
  (people_visiting_2_stores : ℕ) (people_visiting_3_stores : ℕ)
  (people_visiting_4_stores : ℕ) (people_visiting_1_store : ℕ)
  (everyone_visited_at_least_one_store : ∀ p : ℕ, 0 < people_visiting_1_store + people_visiting_2_stores + people_visiting_3_stores + people_visiting_4_stores)
  (h1 : stores = 15) (h2 : total_visits = 60) (h3 : total_peopled_shopping = 30)
  (h4 : people_visiting_2_stores = 12) (h5 : people_visiting_3_stores = 6)
  (h6 : people_visiting_4_stores = 4) (h7 : people_visiting_1_store = total_peopled_shopping - (people_visiting_2_stores + people_visiting_3_stores + people_visiting_4_stores + 2)) :
  ∃ p : ℕ, ∀ person, person ≤ p ∧ p = 4 := sorry

end NUMINAMATH_GPT_largest_number_of_stores_visited_l65_6531


namespace NUMINAMATH_GPT_cost_of_shoes_is_150_l65_6563

def cost_sunglasses : ℕ := 50
def pairs_sunglasses : ℕ := 2
def cost_jeans : ℕ := 100

def cost_basketball_cards : ℕ := 25
def decks_basketball_cards : ℕ := 2

-- Define the total amount spent by Mary and Rose
def total_mary : ℕ := cost_sunglasses * pairs_sunglasses + cost_jeans
def cost_shoes (total_rose : ℕ) (cost_cards : ℕ) : ℕ := total_rose - cost_cards

theorem cost_of_shoes_is_150 (total_spent : ℕ) :
  total_spent = total_mary →
  cost_shoes total_spent (cost_basketball_cards * decks_basketball_cards) = 150 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_of_shoes_is_150_l65_6563


namespace NUMINAMATH_GPT_race_distance_l65_6522

theorem race_distance (d x y z : ℝ) 
  (h1 : d / x = (d - 25) / y)
  (h2 : d / y = (d - 15) / z)
  (h3 : d / x = (d - 35) / z) : 
  d = 75 := 
sorry

end NUMINAMATH_GPT_race_distance_l65_6522


namespace NUMINAMATH_GPT_playerB_hit_rate_playerA_probability_l65_6589

theorem playerB_hit_rate (p : ℝ) (h : (1 - p)^2 = 1/16) : p = 3/4 :=
sorry

theorem playerA_probability (hit_rate : ℝ) (h : hit_rate = 1/2) : 
  (1 - (1 - hit_rate)^2) = 3/4 :=
sorry

end NUMINAMATH_GPT_playerB_hit_rate_playerA_probability_l65_6589


namespace NUMINAMATH_GPT_find_a_l65_6507

theorem find_a :
  ∃ a : ℝ, (∀ t1 t2 : ℝ, t1 + t2 = -a ∧ t1 * t2 = -2017 ∧ 2 * t1 = 4) → a = 1006.5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l65_6507


namespace NUMINAMATH_GPT_toes_on_bus_is_164_l65_6535

def num_toes_hoopit : Nat := 3 * 4
def num_toes_neglart : Nat := 2 * 5

def num_hoopits : Nat := 7
def num_neglarts : Nat := 8

def total_toes_on_bus : Nat :=
  num_hoopits * num_toes_hoopit + num_neglarts * num_toes_neglart

theorem toes_on_bus_is_164 : total_toes_on_bus = 164 := by
  sorry

end NUMINAMATH_GPT_toes_on_bus_is_164_l65_6535


namespace NUMINAMATH_GPT_cubes_difference_l65_6588

theorem cubes_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 65) (h3 : a + b = 6) : a^3 - b^3 = 432.25 :=
by
  sorry

end NUMINAMATH_GPT_cubes_difference_l65_6588


namespace NUMINAMATH_GPT_radio_selling_price_l65_6565

theorem radio_selling_price (CP LP Loss SP : ℝ) (h1 : CP = 1500) (h2 : LP = 11)
  (h3 : Loss = (LP / 100) * CP) (h4 : SP = CP - Loss) : SP = 1335 := 
  by
  -- hint: Apply the given conditions.
  sorry

end NUMINAMATH_GPT_radio_selling_price_l65_6565


namespace NUMINAMATH_GPT_find_a_for_tangent_l65_6517

theorem find_a_for_tangent (a : ℤ) (x : ℝ) (h : ∀ x, 3*x^2 - 4*a*x + 2*a > 0) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_for_tangent_l65_6517
