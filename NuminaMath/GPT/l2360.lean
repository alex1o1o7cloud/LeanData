import Mathlib

namespace mushroom_distribution_l2360_236080

-- Define the total number of mushrooms
def total_mushrooms : ℕ := 120

-- Define the number of girls
def number_of_girls : ℕ := 5

-- Auxiliary function to represent each girl receiving pattern
def mushrooms_received (n :ℕ) (total : ℕ) : ℝ :=
  (n + 20) + 0.04 * (total - (n + 20))

-- Define the equality function to check distribution condition
def equal_distribution (girls : ℕ) (total : ℕ) : Prop :=
  ∀ i j : ℕ, i < girls → j < girls → mushrooms_received i total = mushrooms_received j total

-- Main proof statement about the total mushrooms and number of girls following the distribution
theorem mushroom_distribution :
  total_mushrooms = 120 ∧ number_of_girls = 5 ∧ equal_distribution number_of_girls total_mushrooms := 
by 
  sorry

end mushroom_distribution_l2360_236080


namespace complex_number_solution_l2360_236049

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) / z = 2 + I) :
  im z = -1 ∧ abs z = Real.sqrt 2 ∧ z ^ 6 = -8 * I :=
by
  sorry

end complex_number_solution_l2360_236049


namespace fraction_of_rotten_fruits_l2360_236010

theorem fraction_of_rotten_fruits (a p : ℕ) (rotten_apples_eq_rotten_pears : (2 / 3) * a = (3 / 4) * p)
    (rotten_apples_fraction : 2 / 3 = 2 / 3)
    (rotten_pears_fraction : 3 / 4 = 3 / 4) :
    (4 * a) / (3 * (a + (4 / 3) * (2 * a) / 3)) = 12 / 17 :=
by
  sorry

end fraction_of_rotten_fruits_l2360_236010


namespace smartphones_discount_l2360_236081

theorem smartphones_discount
  (discount : ℝ)
  (cost_per_iphone : ℝ)
  (total_saving : ℝ)
  (num_people : ℕ)
  (num_iphones : ℕ)
  (total_cost : ℝ)
  (required_num : ℕ) :
  discount = 0.05 →
  cost_per_iphone = 600 →
  total_saving = 90 →
  num_people = 3 →
  num_iphones = 3 →
  total_cost = num_iphones * cost_per_iphone →
  required_num = num_iphones →
  required_num * cost_per_iphone * discount = total_saving →
  required_num = 3 :=
by
  intros
  sorry

end smartphones_discount_l2360_236081


namespace jelly_bean_ratio_l2360_236096

theorem jelly_bean_ratio 
  (Napoleon_jelly_beans : ℕ)
  (Sedrich_jelly_beans : ℕ)
  (Mikey_jelly_beans : ℕ)
  (h1 : Napoleon_jelly_beans = 17)
  (h2 : Sedrich_jelly_beans = Napoleon_jelly_beans + 4)
  (h3 : Mikey_jelly_beans = 19) :
  2 * (Napoleon_jelly_beans + Sedrich_jelly_beans) / Mikey_jelly_beans = 4 := 
sorry

end jelly_bean_ratio_l2360_236096


namespace find_b_in_cubic_function_l2360_236001

noncomputable def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem find_b_in_cubic_function (a b c d : ℝ) (h1: cubic_function a b c d 2 = 0)
  (h2: cubic_function a b c d (-1) = 0) (h3: cubic_function a b c d 1 = 4) :
  b = 6 :=
by
  sorry

end find_b_in_cubic_function_l2360_236001


namespace who_is_wrong_l2360_236005

theorem who_is_wrong 
  (a1 a2 a3 a4 a5 a6 : ℤ)
  (h1 : a1 + a3 + a5 = a2 + a4 + a6 + 3)
  (h2 : a2 + a4 + a6 = a1 + a3 + a5 + 5) : 
  False := 
sorry

end who_is_wrong_l2360_236005


namespace abby_correct_percentage_l2360_236062

-- Defining the scores and number of problems for each test
def score_test1 := 85 / 100
def score_test2 := 75 / 100
def score_test3 := 60 / 100
def score_test4 := 90 / 100

def problems_test1 := 30
def problems_test2 := 50
def problems_test3 := 20
def problems_test4 := 40

-- Define the total number of problems
def total_problems := problems_test1 + problems_test2 + problems_test3 + problems_test4

-- Calculate the number of problems Abby answered correctly on each test
def correct_problems_test1 := score_test1 * problems_test1
def correct_problems_test2 := score_test2 * problems_test2
def correct_problems_test3 := score_test3 * problems_test3
def correct_problems_test4 := score_test4 * problems_test4

-- Calculate the total number of correctly answered problems
def total_correct_problems := correct_problems_test1 + correct_problems_test2 + correct_problems_test3 + correct_problems_test4

-- Calculate the overall percentage of problems answered correctly
def overall_percentage_correct := (total_correct_problems / total_problems) * 100

-- The theorem to be proved
theorem abby_correct_percentage : overall_percentage_correct = 80 := by
  -- Skipping the actual proof
  sorry

end abby_correct_percentage_l2360_236062


namespace residents_rent_contribution_l2360_236039

theorem residents_rent_contribution (x R : ℝ) (hx1 : 10 * x + 88 = R) (hx2 : 10.80 * x = 1.025 * R) :
  R / x = 10.54 :=
by sorry

end residents_rent_contribution_l2360_236039


namespace geometric_sequence_common_ratio_l2360_236017

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} 
    (h1 : a 1 = 1) 
    (h4 : a 4 = 1 / 64) 
    (geom_seq : ∀ n, ∃ r, a (n + 1) = a n * r) : 
       
    ∃ q, (∀ n, a n = 1 * (q ^ (n - 1))) ∧ (a 4 = 1 * (q ^ 3)) ∧ q = 1 / 4 := 
by
    sorry

end geometric_sequence_common_ratio_l2360_236017


namespace savings_percentage_correct_l2360_236055

-- Definitions based on conditions
def food_per_week : ℕ := 100
def num_weeks : ℕ := 4
def rent : ℕ := 1500
def video_streaming : ℕ := 30
def cell_phone : ℕ := 50
def savings : ℕ := 198

-- Total spending calculations based on the conditions
def food_total : ℕ := food_per_week * num_weeks
def total_spending : ℕ := food_total + rent + video_streaming + cell_phone

-- Calculation of the percentage
def savings_percentage (savings total_spending : ℕ) : ℕ :=
  (savings * 100) / total_spending

-- The statement to prove
theorem savings_percentage_correct : savings_percentage savings total_spending = 10 := by
  sorry

end savings_percentage_correct_l2360_236055


namespace problem_solution_l2360_236012

theorem problem_solution :
  (204^2 - 196^2) / 16 = 200 :=
by
  sorry

end problem_solution_l2360_236012


namespace cos_four_times_arccos_val_l2360_236037

theorem cos_four_times_arccos_val : 
  ∀ x : ℝ, x = Real.arccos (1 / 4) → Real.cos (4 * x) = 17 / 32 :=
by
  intro x h
  sorry

end cos_four_times_arccos_val_l2360_236037


namespace min_abs_sum_of_diffs_l2360_236099

theorem min_abs_sum_of_diffs (x : ℝ) (α β : ℝ)
  (h₁ : α * α - 6 * α + 5 = 0)
  (h₂ : β * β - 6 * β + 5 = 0)
  (h_ne : α ≠ β) :
  ∃ m, ∀ x, m = min (|x - α| + |x - β|) :=
by
  use (4)
  sorry

end min_abs_sum_of_diffs_l2360_236099


namespace intersection_is_correct_l2360_236053

noncomputable def A : Set ℝ := { x : ℝ | x^2 - x - 2 ≤ 0 }
noncomputable def B : Set ℝ := { x : ℝ | x < 1 }

theorem intersection_is_correct : (A ∩ B) = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_is_correct_l2360_236053


namespace point_on_parabola_l2360_236069

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem point_on_parabola : parabola (1/2) = 0 := 
by sorry

end point_on_parabola_l2360_236069


namespace determinant_of_2x2_matrix_l2360_236088

theorem determinant_of_2x2_matrix :
  let a := 2
  let b := 4
  let c := 1
  let d := 3
  a * d - b * c = 2 := by
  sorry

end determinant_of_2x2_matrix_l2360_236088


namespace sum_after_third_rotation_max_sum_of_six_faces_l2360_236002

variable (a b c : ℕ) (a' b': ℕ)

-- Initial Conditions
axiom sum_initial : a + b + c = 42

-- Conditions after first rotation
axiom a_prime : a' = a - 8
axiom sum_first_rotation : b + c + a' = 34

-- Conditions after second rotation
axiom b_prime : b' = b + 19
axiom sum_second_rotation : c + a' + b' = 53

-- The cube always rests on the face with number 6
axiom bottom_face : c = 6

-- Prove question 1:
theorem sum_after_third_rotation : (b + 19) + a + c = 61 :=
by sorry

-- Prove question 2:
theorem max_sum_of_six_faces : 
∃ d e f: ℕ, d = a ∧ e = b ∧ f = c ∧ d + e + f + (a - 8) + (b + 19) + 6 = 100 :=
by sorry

end sum_after_third_rotation_max_sum_of_six_faces_l2360_236002


namespace tractor_efficiency_l2360_236025

theorem tractor_efficiency (x y : ℝ) (h1 : 18 / x = 24 / y) (h2 : x + y = 7) :
  x = 3 ∧ y = 4 :=
by {
  sorry
}

end tractor_efficiency_l2360_236025


namespace expected_value_max_l2360_236009

def E_max_x_y_z (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 10) : ℚ :=
  (4 * (1/6) + 5 * (1/3) + 6 * (1/4) + 7 * (1/6) + 8 * (1/12))

theorem expected_value_max (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 10) :
  E_max_x_y_z x y z h1 h2 h3 h4 = 17 / 3 := 
sorry

end expected_value_max_l2360_236009


namespace arithmetic_mean_of_fractions_l2360_236067

theorem arithmetic_mean_of_fractions :
  let a := 7 / 9
  let b := 5 / 6
  let c := 8 / 9
  2 * b = a + c :=
by
  sorry

end arithmetic_mean_of_fractions_l2360_236067


namespace chair_arrangements_48_l2360_236083

theorem chair_arrangements_48 :
  ∃ (n : ℕ), n = 8 ∧ (∀ (r c : ℕ), r * c = 48 → 2 ≤ r ∧ 2 ≤ c) := 
sorry

end chair_arrangements_48_l2360_236083


namespace smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l2360_236073

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * (Real.sin x)^2 - (1 + 2)

theorem smallest_period_of_f : ∀ x : ℝ, f (x + π) = f x := 
by sorry

theorem center_of_symmetry_of_f : ∀ k : ℤ, ∃ c : ℝ, ∀ x : ℝ, f (c - x) = f (c + x) := 
by sorry

theorem range_of_f_on_interval : 
  ∃ a b, (∀ x ∈ Set.Icc (-π / 4) (π / 4), f x ∈ Set.Icc a b) ∧ 
          (∀ y, y ∈ Set.Icc 3 5 → ∃ x ∈ Set.Icc (-π / 4) (π / 4), y = f x) := 
by sorry

end smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l2360_236073


namespace cars_at_2023_cars_less_than_15_l2360_236093

def a_recurrence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 0.9 * a n + 8

def initial_condition (a : ℕ → ℝ) : Prop :=
a 1 = 300

theorem cars_at_2023 (a : ℕ → ℝ)
  (h_recurrence : a_recurrence a)
  (h_initial : initial_condition a) :
  a 4 = 240 :=
sorry

def shifted_geom_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - 80 = 0.9 * (a n - 80)

theorem cars_less_than_15 (a : ℕ → ℝ)
  (h_recurrence : a_recurrence a)
  (h_initial : initial_condition a)
  (h_geom_seq : shifted_geom_seq a) :
  ∃ n, n ≥ 12 ∧ a n < 15 :=
sorry

end cars_at_2023_cars_less_than_15_l2360_236093


namespace cone_volume_is_correct_l2360_236054

theorem cone_volume_is_correct (r l h : ℝ) 
  (h1 : 2 * r = Real.sqrt 2 * l)
  (h2 : π * r * l = 16 * Real.sqrt 2 * π)
  (h3 : h = r) : 
  (1 / 3) * π * r ^ 2 * h = (64 / 3) * π :=
by sorry

end cone_volume_is_correct_l2360_236054


namespace bus_children_count_l2360_236090

theorem bus_children_count
  (initial_count : ℕ)
  (first_stop_add : ℕ)
  (second_stop_add : ℕ)
  (second_stop_remove : ℕ)
  (third_stop_remove : ℕ)
  (third_stop_add : ℕ)
  (final_count : ℕ)
  (h1 : initial_count = 18)
  (h2 : first_stop_add = 5)
  (h3 : second_stop_remove = 4)
  (h4 : third_stop_remove = 3)
  (h5 : third_stop_add = 5)
  (h6 : final_count = 25)
  (h7 : initial_count + first_stop_add = 23)
  (h8 : 23 + second_stop_add - second_stop_remove - third_stop_remove + third_stop_add = final_count) :
  second_stop_add = 4 :=
by
  sorry

end bus_children_count_l2360_236090


namespace find_y_l2360_236079

theorem find_y :
  ∃ (x y : ℤ), (x - 5) / 7 = 7 ∧ (x - y) / 10 = 3 ∧ y = 24 :=
by
  sorry

end find_y_l2360_236079


namespace henry_kombucha_bottles_l2360_236058

theorem henry_kombucha_bottles :
  ∀ (monthly_bottles: ℕ) (cost_per_bottle refund_rate: ℝ) (months_in_year total_bottles_in_year: ℕ),
  (monthly_bottles = 15) →
  (cost_per_bottle = 3.0) →
  (refund_rate = 0.10) →
  (months_in_year = 12) →
  (total_bottles_in_year = monthly_bottles * months_in_year) →
  (total_refund = refund_rate * total_bottles_in_year) →
  (bottles_bought_with_refund = total_refund / cost_per_bottle) →
  bottles_bought_with_refund = 6 :=
by
  intros monthly_bottles cost_per_bottle refund_rate months_in_year total_bottles_in_year
  sorry

end henry_kombucha_bottles_l2360_236058


namespace spaceship_distance_l2360_236029

-- Define the distance variables and conditions
variables (D : ℝ) -- Distance from Earth to Planet X
variable (T : ℝ) -- Total distance traveled by the spaceship

-- Conditions
variables (hx : T = 0.7) -- Total distance traveled is 0.7 light-years
variables (hy : D + 0.1 + 0.1 = T) -- Sum of distances along the path

-- Theorem statement to prove the distance from Earth to Planet X
theorem spaceship_distance (h1 : T = 0.7) (h2 : D + 0.1 + 0.1 = T) : D = 0.5 :=
by
  -- Proof steps would go here
  sorry

end spaceship_distance_l2360_236029


namespace equivalent_single_discount_l2360_236056

variable (original_price : ℝ)
variable (first_discount : ℝ)
variable (second_discount : ℝ)

-- Conditions
def sale_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)

def final_price (p : ℝ) (d1 d2 : ℝ) : ℝ :=
  let sale1 := sale_price p d1
  sale_price sale1 d2

-- Prove the equivalent single discount is as described
theorem equivalent_single_discount :
  original_price = 30 → first_discount = 0.2 → second_discount = 0.25 →
  (1 - final_price original_price first_discount second_discount / original_price) * 100 = 40 :=
by
  intros
  sorry

end equivalent_single_discount_l2360_236056


namespace square_perimeter_l2360_236076

variable (side : ℕ) (P : ℕ)

theorem square_perimeter (h : side = 19) : P = 4 * side → P = 76 := by
  intro hp
  rw [h] at hp
  norm_num at hp
  exact hp

end square_perimeter_l2360_236076


namespace least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l2360_236089

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_primes_greater_than_five : List ℕ :=
  [7, 11, 13]

theorem least_positive_integer_divisible_by_three_smallest_primes_greater_than_five : 
  ∃ n : ℕ, n > 0 ∧ (∀ p ∈ smallest_primes_greater_than_five, p ∣ n) ∧ n = 1001 := by
  sorry

end least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l2360_236089


namespace geometric_sequence_problem_l2360_236040

theorem geometric_sequence_problem 
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h1 : a 7 * a 11 = 6)
  (h2 : a 4 + a 14 = 5) :
  ∃ x : ℝ, x = 2 / 3 ∨ x = 3 / 2 := by
  sorry

end geometric_sequence_problem_l2360_236040


namespace range_of_alpha_plus_beta_l2360_236036

theorem range_of_alpha_plus_beta (α β : ℝ) (h1 : 0 < α - β) (h2 : α - β < π) (h3 : 0 < α + 2 * β) (h4 : α + 2 * β < π) :
  0 < α + β ∧ α + β < π :=
sorry

end range_of_alpha_plus_beta_l2360_236036


namespace committee_count_l2360_236061

noncomputable def num_acceptable_committees (total_people : ℕ) (committee_size : ℕ) (conditions : List (Set ℕ)) : ℕ := sorry

theorem committee_count :
  num_acceptable_committees 9 5 [ {1, 2}, {3, 4} ] = 41 := sorry

end committee_count_l2360_236061


namespace chocolate_bar_cost_l2360_236064

variable (cost_per_bar num_bars : ℝ)

theorem chocolate_bar_cost (num_scouts smores_per_scout smores_per_bar : ℕ) (total_cost : ℝ)
  (h1 : num_scouts = 15)
  (h2 : smores_per_scout = 2)
  (h3 : smores_per_bar = 3)
  (h4 : total_cost = 15)
  (h5 : num_bars = (num_scouts * smores_per_scout) / smores_per_bar)
  (h6 : total_cost = cost_per_bar * num_bars) :
  cost_per_bar = 1.50 :=
by
  sorry

end chocolate_bar_cost_l2360_236064


namespace find_f_2012_l2360_236063

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x / Real.log 2 + b * Real.log x / Real.log 3 + 2

theorem find_f_2012 (a b : ℝ) (h : f (1 / 2012) a b = 5) : f 2012 a b = -1 :=
by
  sorry

end find_f_2012_l2360_236063


namespace measure_four_liters_impossible_l2360_236070

theorem measure_four_liters_impossible (a b c : ℕ) (h1 : a = 12) (h2 : b = 9) (h3 : c = 4) :
  ¬ ∃ x y : ℕ, x * a + y * b = c := 
by
  sorry

end measure_four_liters_impossible_l2360_236070


namespace odd_number_divides_3n_plus_1_l2360_236071

theorem odd_number_divides_3n_plus_1 (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n ∣ 3^n + 1) : n = 1 :=
by
  sorry

end odd_number_divides_3n_plus_1_l2360_236071


namespace woman_l2360_236030

-- Define the variables and given conditions
variables (W S X : ℕ)
axiom s_eq : S = 27
axiom sum_eq : W + S = 84
axiom w_eq : W = 2 * S + X

theorem woman's_age_more_years : X = 3 :=
by
  -- Proof goes here
  sorry

end woman_l2360_236030


namespace largest_consecutive_even_sum_l2360_236006

theorem largest_consecutive_even_sum (a b c : ℤ) (h1 : b = a+2) (h2 : c = a+4) (h3 : a + b + c = 312) : c = 106 := 
by 
  sorry

end largest_consecutive_even_sum_l2360_236006


namespace combined_length_in_scientific_notation_l2360_236041

noncomputable def yards_to_inches (yards : ℝ) : ℝ := yards * 36
noncomputable def inches_to_cm (inches : ℝ) : ℝ := inches * 2.54
noncomputable def feet_to_inches (feet : ℝ) : ℝ := feet * 12

def sports_stadium_length_yards : ℝ := 61
def safety_margin_feet : ℝ := 2
def safety_margin_inches : ℝ := 9

theorem combined_length_in_scientific_notation :
  (inches_to_cm (yards_to_inches sports_stadium_length_yards) +
   (inches_to_cm (feet_to_inches safety_margin_feet + safety_margin_inches)) * 2) = 5.74268 * 10^3 :=
by
  sorry

end combined_length_in_scientific_notation_l2360_236041


namespace abcde_sum_l2360_236097

theorem abcde_sum : 
  ∀ (a b c d e : ℝ), 
  a + 1 = b + 2 → 
  b + 2 = c + 3 → 
  c + 3 = d + 4 → 
  d + 4 = e + 5 → 
  e + 5 = a + b + c + d + e + 10 → 
  a + b + c + d + e = -35 / 4 :=
sorry

end abcde_sum_l2360_236097


namespace bob_pennies_l2360_236000

theorem bob_pennies (a b : ℕ) 
  (h1 : b + 1 = 4 * (a - 1)) 
  (h2 : b - 1 = 3 * (a + 1)) : 
  b = 31 :=
by
  sorry

end bob_pennies_l2360_236000


namespace value_of_k_l2360_236084

theorem value_of_k :
  ∀ (k : ℝ), (∃ m : ℝ, m = 4/5 ∧ (21 - (-5)) / (k - 3) = m) →
  k = 35.5 :=
by
  intros k hk
  -- Here hk is the proof that the line through (3, -5) and (k, 21) has the same slope as 4/5
  sorry

end value_of_k_l2360_236084


namespace solve_chair_table_fraction_l2360_236034

def chair_table_fraction : Prop :=
  ∃ (C T : ℝ), T = 140 ∧ (T + 4 * C = 220) ∧ (C / T = 1 / 7)

theorem solve_chair_table_fraction : chair_table_fraction :=
  sorry

end solve_chair_table_fraction_l2360_236034


namespace bus_problem_initial_buses_passengers_l2360_236014

theorem bus_problem_initial_buses_passengers : 
  ∃ (m n : ℕ), m ≥ 2 ∧ n ≤ 32 ∧ 22 * m + 1 = n * (m - 1) ∧ n * (m - 1) = 529 ∧ m = 24 :=
sorry

end bus_problem_initial_buses_passengers_l2360_236014


namespace union_complement_A_B_l2360_236092

-- Definitions based on conditions
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | x < 6}
def C_R (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- The proof problem statement
theorem union_complement_A_B :
  (C_R B ∪ A = {x | 0 ≤ x}) :=
by 
  sorry

end union_complement_A_B_l2360_236092


namespace fill_tank_time_l2360_236075

theorem fill_tank_time 
  (tank_capacity : ℕ) (initial_fill : ℕ) (fill_rate : ℝ) 
  (drain_rate1 : ℝ) (drain_rate2 : ℝ) : 
  tank_capacity = 8000 ∧ initial_fill = 4000 ∧ fill_rate = 0.5 ∧ drain_rate1 = 0.25 ∧ drain_rate2 = 0.1667 
  → (initial_fill + fill_rate * t - (drain_rate1 + drain_rate2) * t) = tank_capacity → t = 48 := sorry

end fill_tank_time_l2360_236075


namespace boatworks_total_canoes_l2360_236046

theorem boatworks_total_canoes : 
  let jan := 5
  let feb := 3 * jan
  let mar := 3 * feb
  let apr := 3 * mar
  jan + feb + mar + apr = 200 := 
by 
  sorry

end boatworks_total_canoes_l2360_236046


namespace tank_capacity_is_780_l2360_236045

noncomputable def tank_capacity : ℕ := 
  let fill_rate_A := 40
  let fill_rate_B := 30
  let drain_rate_C := 20
  let cycle_minutes := 3
  let total_minutes := 48
  let net_fill_per_cycle := fill_rate_A + fill_rate_B - drain_rate_C
  let total_cycles := total_minutes / cycle_minutes
  let total_fill := total_cycles * net_fill_per_cycle
  let final_capacity := total_fill - drain_rate_C -- Adjust for the last minute where C opens
  final_capacity

theorem tank_capacity_is_780 : tank_capacity = 780 := by
  unfold tank_capacity
  -- Proof steps to be filled in
  sorry

end tank_capacity_is_780_l2360_236045


namespace number_of_classes_l2360_236086

theorem number_of_classes (max_val : ℕ) (min_val : ℕ) (class_interval : ℕ) (range : ℕ) (num_classes : ℕ) :
  max_val = 169 → min_val = 143 → class_interval = 3 → range = max_val - min_val → num_classes = (range + 2) / class_interval + 1 :=
sorry

end number_of_classes_l2360_236086


namespace find_P_Q_sum_l2360_236038

theorem find_P_Q_sum (P Q : ℤ) 
  (h : ∃ b c : ℤ, x^2 + 3 * x + 2 ∣ x^4 + P * x^2 + Q 
    ∧ b + 3 = 0 
    ∧ c + 3 * b + 6 = P 
    ∧ 3 * c + 2 * b = 0 
    ∧ 2 * c = Q): 
  P + Q = 3 := 
sorry

end find_P_Q_sum_l2360_236038


namespace integer_solution_count_l2360_236051

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end integer_solution_count_l2360_236051


namespace triangle_incircle_ratio_l2360_236026

theorem triangle_incircle_ratio (r p k : ℝ) (h1 : k = r * (p / 2)) : 
  p / k = 2 / r :=
by
  sorry

end triangle_incircle_ratio_l2360_236026


namespace peak_valley_usage_l2360_236027

-- Define the electricity rate constants
def normal_rate : ℝ := 0.5380
def peak_rate : ℝ := 0.5680
def valley_rate : ℝ := 0.2880

-- Define the total consumption and the savings
def total_consumption : ℝ := 200
def savings : ℝ := 16.4

-- Define the theorem to prove the peak and off-peak usage
theorem peak_valley_usage :
  ∃ (x y : ℝ), x + y = total_consumption ∧ peak_rate * x + valley_rate * y = total_consumption * normal_rate - savings ∧ x = 120 ∧ y = 80 :=
by
  sorry

end peak_valley_usage_l2360_236027


namespace f_2015_l2360_236059

def f : ℤ → ℤ := sorry

axiom f1 : f 1 = 1
axiom f2 : f 2 = 0
axiom functional_eq (x y : ℤ) : f (x + y) = f x * f (1 - y) + f (1 - x) * f y

theorem f_2015 : f 2015 = 1 ∨ f 2015 = -1 :=
sorry

end f_2015_l2360_236059


namespace sin_theta_fourth_quadrant_l2360_236035

-- Given conditions
variables {θ : ℝ} (h1 : Real.cos θ = 1 / 3) (h2 : 3 * pi / 2 < θ ∧ θ < 2 * pi)

-- Proof statement
theorem sin_theta_fourth_quadrant : Real.sin θ = -2 * Real.sqrt 2 / 3 :=
sorry

end sin_theta_fourth_quadrant_l2360_236035


namespace origin_in_ellipse_l2360_236013

theorem origin_in_ellipse (k : ℝ):
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ x = 0 ∧ y = 0) →
  0 < abs k ∧ abs k < 1 :=
by
  -- Note: Proof omitted.
  sorry

end origin_in_ellipse_l2360_236013


namespace speed_of_second_fragment_l2360_236011

noncomputable def magnitude_speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) (v_y1 : ℝ := - (u - g * t)) 
  (v_x2 : ℝ := -v_x1) (v_y2 : ℝ := v_y1) : ℝ :=
Real.sqrt ((v_x2 ^ 2) + (v_y2 ^ 2))

theorem speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) 
  (h_u : u = 20) (h_t : t = 3) (h_g : g = 10) (h_vx1 : v_x1 = 48) :
  magnitude_speed_of_second_fragment u t g v_x1 = Real.sqrt 2404 :=
by
  -- Proof
  sorry

end speed_of_second_fragment_l2360_236011


namespace not_true_expr_l2360_236003

theorem not_true_expr (x y : ℝ) (h : x < y) : -2 * x > -2 * y :=
sorry

end not_true_expr_l2360_236003


namespace combined_salaries_ABC_E_l2360_236048

-- Definitions for the conditions
def salary_D : ℝ := 7000
def avg_salary_ABCDE : ℝ := 8200

-- Defining the combined salary proof
theorem combined_salaries_ABC_E : (A B C E : ℝ) → 
  (A + B + C + D + E = 5 * avg_salary_ABCDE ∧ D = salary_D) → 
  (A + B + C + E = 34000) := 
sorry

end combined_salaries_ABC_E_l2360_236048


namespace ratio_vegan_gluten_free_cupcakes_l2360_236098

theorem ratio_vegan_gluten_free_cupcakes :
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  (vegan_gluten_free_cupcakes / vegan_cupcakes) = 1 / 2 :=
by {
  let total_cupcakes := 80
  let gluten_free_cupcakes := total_cupcakes / 2
  let vegan_cupcakes := 24
  let non_vegan_gluten_cupcakes := 28
  let vegan_gluten_free_cupcakes := gluten_free_cupcakes - non_vegan_gluten_cupcakes
  have h : vegan_gluten_free_cupcakes = 12 := by norm_num
  have r : 12 / 24 = 1 / 2 := by norm_num
  exact r
}

end ratio_vegan_gluten_free_cupcakes_l2360_236098


namespace percentage_increase_Sakshi_Tanya_l2360_236052

def efficiency_Sakshi : ℚ := 1 / 5
def efficiency_Tanya : ℚ := 1 / 4
def percentage_increase_in_efficiency (eff_Sakshi eff_Tanya : ℚ) : ℚ :=
  ((eff_Tanya - eff_Sakshi) / eff_Sakshi) * 100

theorem percentage_increase_Sakshi_Tanya :
  percentage_increase_in_efficiency efficiency_Sakshi efficiency_Tanya = 25 :=
by
  sorry

end percentage_increase_Sakshi_Tanya_l2360_236052


namespace problem_solution_l2360_236085

noncomputable def vector_magnitudes_and_angle 
  (a b : ℝ) (angle_ab : ℝ) (norma normb : ℝ) (k : ℝ) : Prop :=
(a = 4 ∧ b = 8 ∧ angle_ab = 2 * Real.pi / 3 ∧ norma = 4 ∧ normb = 8) →
((norma^2 + normb^2 + 2 * norma * normb * Real.cos angle_ab = 48) ∧
  (16 * k - 32 * k + 16 - 128 = 0))

theorem problem_solution : vector_magnitudes_and_angle 4 8 (2 * Real.pi / 3) 4 8 (-7) := 
by 
  sorry

end problem_solution_l2360_236085


namespace total_surface_area_of_cube_l2360_236066

theorem total_surface_area_of_cube : 
  ∀ (s : Real), 
  (12 * s = 36) → 
  (s * Real.sqrt 3 = 3 * Real.sqrt 3) → 
  6 * s^2 = 54 := 
by
  intros s h1 h2
  sorry

end total_surface_area_of_cube_l2360_236066


namespace evaluate_expression_l2360_236095

variables (x : ℝ)

theorem evaluate_expression :
  x * (x * (x * (3 - x) - 5) + 13) + 1 = -x^4 + 3*x^3 - 5*x^2 + 13*x + 1 :=
by 
  sorry

end evaluate_expression_l2360_236095


namespace sum_of_remainders_l2360_236020

theorem sum_of_remainders (d e f g : ℕ)
  (hd : d % 30 = 15)
  (he : e % 30 = 5)
  (hf : f % 30 = 10)
  (hg : g % 30 = 20) :
  (d + e + f + g) % 30 = 20 :=
by
  sorry

end sum_of_remainders_l2360_236020


namespace find_remainder_mod_10_l2360_236050

def inv_mod_10 (x : ℕ) : ℕ := 
  if x = 1 then 1 
  else if x = 3 then 7 
  else if x = 7 then 3 
  else if x = 9 then 9 
  else 0 -- invalid, not invertible

theorem find_remainder_mod_10 (a b c d : ℕ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ d) (hd : d ≠ a) 
  (ha' : a < 10) (hb' : b < 10) (hc' : c < 10) (hd' : d < 10)
  (ha_inv : inv_mod_10 a ≠ 0) (hb_inv : inv_mod_10 b ≠ 0)
  (hc_inv : inv_mod_10 c ≠ 0) (hd_inv : inv_mod_10 d ≠ 0) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (inv_mod_10 (a * b * c * d % 10))) % 10 = 0 :=
by
  sorry

end find_remainder_mod_10_l2360_236050


namespace remainder_sum_of_first_eight_primes_div_tenth_prime_l2360_236074

theorem remainder_sum_of_first_eight_primes_div_tenth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) % 29 = 19 :=
by norm_num

end remainder_sum_of_first_eight_primes_div_tenth_prime_l2360_236074


namespace f_neg_expression_l2360_236032

noncomputable def f : ℝ → ℝ :=
  λ x => if x > 0 then x^2 - 2*x + 3 else sorry

-- Define f by cases: for x > 0 and use the property of odd functions to conclude the expression for x < 0.

theorem f_neg_expression (x : ℝ) (h : x < 0) : f x = -x^2 - 2*x - 3 :=
by
  sorry

end f_neg_expression_l2360_236032


namespace vector_subtraction_l2360_236007

-- Lean definitions for the problem conditions
def v₁ : ℝ × ℝ := (3, -5)
def v₂ : ℝ × ℝ := (-2, 6)
def s₁ : ℝ := 4
def s₂ : ℝ := 3

-- The theorem statement
theorem vector_subtraction :
  s₁ • v₁ - s₂ • v₂ = (18, -38) :=
by
  sorry

end vector_subtraction_l2360_236007


namespace max_weight_each_shipping_box_can_hold_l2360_236008

noncomputable def max_shipping_box_weight_pounds 
  (total_plates : ℕ)
  (weight_per_plate_ounces : ℕ)
  (plates_removed : ℕ)
  (ounce_to_pound : ℕ) : ℕ :=
  (total_plates - plates_removed) * weight_per_plate_ounces / ounce_to_pound

theorem max_weight_each_shipping_box_can_hold :
  max_shipping_box_weight_pounds 38 10 6 16 = 20 :=
by
  sorry

end max_weight_each_shipping_box_can_hold_l2360_236008


namespace melanie_phil_ages_l2360_236016

theorem melanie_phil_ages (A B : ℕ) 
  (h : (A + 10) * (B + 10) = A * B + 400) :
  (A + 6) + (B + 6) = 42 :=
by
  sorry

end melanie_phil_ages_l2360_236016


namespace percentage_decrease_increase_l2360_236057

theorem percentage_decrease_increase (S : ℝ) (x : ℝ) (h : S > 0) :
  S * (1 - x / 100) * (1 + x / 100) = S * (64 / 100) → x = 6 :=
by
  sorry

end percentage_decrease_increase_l2360_236057


namespace product_of_two_two_digit_numbers_greater_than_40_is_four_digit_l2360_236028

-- Define the condition: both numbers are two-digit numbers greater than 40
def is_two_digit_and_greater_than_40 (n : ℕ) : Prop :=
  40 < n ∧ n < 100

-- Define the problem statement
theorem product_of_two_two_digit_numbers_greater_than_40_is_four_digit
  (a b : ℕ) (ha : is_two_digit_and_greater_than_40 a) (hb : is_two_digit_and_greater_than_40 b) :
  1000 ≤ a * b ∧ a * b < 10000 :=
by
  sorry

end product_of_two_two_digit_numbers_greater_than_40_is_four_digit_l2360_236028


namespace initial_children_count_l2360_236077

theorem initial_children_count (passed retake : ℝ) (h_passed : passed = 105.0) (h_retake : retake = 593) : 
    passed + retake = 698 := 
by
  sorry

end initial_children_count_l2360_236077


namespace triangle_area_condition_l2360_236018

theorem triangle_area_condition (m : ℝ) 
  (H_line : ∀ (x y : ℝ), x - m*y + 1 = 0)
  (H_circle : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 4)
  (H_area : ∃ (A B C : (ℝ × ℝ)), (x - my + 1 = 0) ∧ (∃ C : (ℝ × ℝ), (x1 - 1)^2 + y1^2 = 4 ∨ (x2 - 1)^2 + y2^2 = 4))
  : m = 2 :=
sorry

end triangle_area_condition_l2360_236018


namespace number_of_intersections_l2360_236068

theorem number_of_intersections (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x = 4) → (x = 4 ∧ y = 0) :=
by {
  sorry
}

end number_of_intersections_l2360_236068


namespace inequality_solution_set_l2360_236094

noncomputable def f (x : ℝ) : ℝ := x + 1 / (2 * x) + 2

lemma f_increasing {x₁ x₂ : ℝ} (hx₁ : 1 ≤ x₁) (hx₂ : 1 ≤ x₂) (h : x₁ < x₂) : f x₁ < f x₂ := sorry

lemma solve_inequality (x : ℝ) (hx : 1 ≤ x) : (2 * x - 1 / 2 < x + 1007) → (f (2 * x - 1 / 2) < f (x + 1007)) := sorry

theorem inequality_solution_set {x : ℝ} : (1 ≤ x) → (2 * x - 1 / 2 < x + 1007) ↔ (3 / 4 ≤ x ∧ x < 2015 / 2) := sorry

end inequality_solution_set_l2360_236094


namespace perimeter_of_triangle_l2360_236023

-- The given condition about the average length of the triangle sides.
def average_side_length (a b c : ℝ) (h : (a + b + c) / 3 = 12) : Prop :=
  a + b + c = 36

-- The theorem to prove the perimeter of triangle ABC.
theorem perimeter_of_triangle (a b c : ℝ) (h : (a + b + c) / 3 = 12) : a + b + c = 36 :=
  by
    sorry

end perimeter_of_triangle_l2360_236023


namespace converse_inverse_contrapositive_l2360_236031

-- The original statement
def original_statement (x y : ℕ) : Prop :=
  (x + y = 5) → (x = 3 ∧ y = 2)

-- Converse of the original statement
theorem converse (x y : ℕ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by
  sorry

-- Inverse of the original statement
theorem inverse (x y : ℕ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by
  sorry

-- Contrapositive of the original statement
theorem contrapositive (x y : ℕ) : (x ≠ 3 ∨ y ≠ 2) → (x + y ≠ 5) :=
by
  sorry

end converse_inverse_contrapositive_l2360_236031


namespace total_cost_chairs_l2360_236047

def living_room_chairs : Nat := 3
def kitchen_chairs : Nat := 6
def dining_room_chairs : Nat := 8
def outdoor_patio_chairs : Nat := 12

def living_room_price : Nat := 75
def kitchen_price : Nat := 50
def dining_room_price : Nat := 100
def outdoor_patio_price : Nat := 60

theorem total_cost_chairs : 
  living_room_chairs * living_room_price + 
  kitchen_chairs * kitchen_price + 
  dining_room_chairs * dining_room_price + 
  outdoor_patio_chairs * outdoor_patio_price = 2045 := by
  sorry

end total_cost_chairs_l2360_236047


namespace john_computers_fixed_count_l2360_236019

-- Define the problem conditions.
variables (C : ℕ)
variables (unfixable_ratio spare_part_ratio fixable_ratio : ℝ)
variables (fixed_right_away : ℕ)
variables (h1 : unfixable_ratio = 0.20)
variables (h2 : spare_part_ratio = 0.40)
variables (h3 : fixable_ratio = 0.40)
variables (h4 : fixed_right_away = 8)
variables (h5 : fixable_ratio * ↑C = fixed_right_away)

-- The theorem to prove.
theorem john_computers_fixed_count (h1 : C > 0) : C = 20 := by
  sorry

end john_computers_fixed_count_l2360_236019


namespace calc_num_int_values_l2360_236091

theorem calc_num_int_values (x : ℕ) (h : 121 ≤ x ∧ x < 144) : ∃ n : ℕ, n = 23 :=
by
  sorry

end calc_num_int_values_l2360_236091


namespace relationship_and_range_max_profit_find_a_l2360_236072

noncomputable def functional_relationship (x : ℝ) : ℝ :=
if 40 ≤ x ∧ x ≤ 50 then 5
else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x
else 0  -- default case to handle x out of range, though ideally this should not occur in the context.

theorem relationship_and_range : 
  ∀ (x : ℝ), (40 ≤ x ∧ x ≤ 100) →
    (functional_relationship x = 
    (if 40 ≤ x ∧ x ≤ 50 then 5 else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x else 0)) :=
sorry

noncomputable def monthly_profit (x : ℝ) : ℝ :=
(x - 40) * functional_relationship x

theorem max_profit : 
  (∀ x, 40 ≤ x ∧ x ≤ 100 → monthly_profit x ≤ 90) ∧
  (monthly_profit 70 = 90) :=
sorry

noncomputable def donation_profit (x a : ℝ) : ℝ :=
(x - 40 - a) * (10 - 0.1 * x)

theorem find_a (a : ℝ) : 
  (∀ x, x ≤ 70 → donation_profit x a ≤ 78) ∧
  (donation_profit 70 a = 78) → 
  a = 4 :=
sorry

end relationship_and_range_max_profit_find_a_l2360_236072


namespace line_equation_l2360_236021

theorem line_equation (P : ℝ × ℝ) (hP : P = (-2, 1)) :
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -5 := by
  sorry

end line_equation_l2360_236021


namespace students_remaining_after_third_stop_l2360_236022

theorem students_remaining_after_third_stop
  (initial_students : ℕ)
  (third : ℚ) (stops : ℕ)
  (one_third_off : third = 1 / 3)
  (initial_students_eq : initial_students = 64)
  (stops_eq : stops = 3)
  : 64 * ((2 / 3) ^ 3) = 512 / 27 :=
by 
  sorry

end students_remaining_after_third_stop_l2360_236022


namespace oil_used_l2360_236042

theorem oil_used (total_weight : ℕ) (ratio_oil_peanuts : ℕ) (ratio_total_parts : ℕ) 
  (ratio_peanuts : ℕ) (ratio_parts : ℕ) (peanuts_weight : ℕ) : 
  ratio_oil_peanuts = 2 → 
  ratio_peanuts = 8 → 
  ratio_total_parts = 10 → 
  ratio_parts = 20 →
  peanuts_weight = total_weight / ratio_total_parts →
  total_weight = 20 → 
  2 * peanuts_weight = 4 :=
by sorry

end oil_used_l2360_236042


namespace largest_number_l2360_236033

noncomputable def a : ℝ := 8.12331
noncomputable def b : ℝ := 8.123 + 3 / 10000 * ∑' n, 1 / (10 : ℝ)^n
noncomputable def c : ℝ := 8.12 + 331 / 100000 * ∑' n, 1 / (1000 : ℝ)^n
noncomputable def d : ℝ := 8.1 + 2331 / 1000000 * ∑' n, 1 / (10000 : ℝ)^n
noncomputable def e : ℝ := 8 + 12331 / 100000 * ∑' n, 1 / (10000 : ℝ)^n

theorem largest_number : (b > a) ∧ (b > c) ∧ (b > d) ∧ (b > e) := by
  sorry

end largest_number_l2360_236033


namespace second_derivative_of_y_l2360_236024

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.log (1 + Real.sin x)

theorem second_derivative_of_y :
  (deriv^[2] y) x = 
  2 * Real.log (1 + Real.sin x) + (4 * x * Real.cos x - x ^ 2) / (1 + Real.sin x) :=
sorry

end second_derivative_of_y_l2360_236024


namespace divisibility_of_n_l2360_236004

theorem divisibility_of_n (P : Polynomial ℤ) (k n : ℕ)
  (hk : k % 2 = 0)
  (h_odd_coeffs : ∀ i, i ≤ k → i % 2 = 1)
  (h_div : ∃ Q : Polynomial ℤ, (X + 1)^n - 1 = (P * Q)) :
  n % (k + 1) = 0 :=
sorry

end divisibility_of_n_l2360_236004


namespace find_fraction_of_ab_l2360_236044

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def x := a / b

theorem find_fraction_of_ab (h1 : a ≠ b) (h2 : a / b + (3 * a + 4 * b) / (b + 12 * a) = 2) :
  a / b = (5 - Real.sqrt 19) / 6 :=
sorry

end find_fraction_of_ab_l2360_236044


namespace parabola_vertex_l2360_236043

theorem parabola_vertex (x y : ℝ) : ∀ x y, (y^2 + 8 * y + 2 * x + 11 = 0) → (x = 5 / 2 ∧ y = -4) :=
by
  intro x y h
  sorry

end parabola_vertex_l2360_236043


namespace yield_percentage_l2360_236065

theorem yield_percentage (d : ℝ) (q : ℝ) (f : ℝ) : d = 12 → q = 150 → f = 100 → (d * f / q) * 100 = 8 :=
by
  intros h_d h_q h_f
  rw [h_d, h_q, h_f]
  sorry

end yield_percentage_l2360_236065


namespace find_b_l2360_236087

theorem find_b
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) -> True)
  (h4 : ∃ e, e = (Real.sqrt 5) / 3)
  (h5 : 2 * a = 12) :
  b = 4 :=
by
  sorry

end find_b_l2360_236087


namespace sum_of_reciprocals_l2360_236082

noncomputable def roots (p q r : ℂ) : Prop := 
  p ^ 3 - p + 1 = 0 ∧ q ^ 3 - q + 1 = 0 ∧ r ^ 3 - r + 1 = 0

theorem sum_of_reciprocals (p q r : ℂ) (h : roots p q r) : 
  (1 / (p + 2)) + (1 / (q + 2)) + (1 / (r + 2)) = - (10 / 13) := by 
  sorry

end sum_of_reciprocals_l2360_236082


namespace john_climbs_9_flights_l2360_236060

variable (fl : Real := 10)  -- Each flight of stairs is 10 feet
variable (step_height_inches : Real := 18)  -- Each step is 18 inches
variable (steps : Nat := 60)  -- John climbs 60 steps

theorem john_climbs_9_flights :
  (steps * (step_height_inches / 12) / fl = 9) :=
by
  sorry

end john_climbs_9_flights_l2360_236060


namespace difference_of_squares_l2360_236078

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : |x^2 - y^2| = 108 :=
  sorry

end difference_of_squares_l2360_236078


namespace quadratic_root_and_a_value_l2360_236015

theorem quadratic_root_and_a_value (a : ℝ) (h1 : (a + 3) * 0^2 - 4 * 0 + a^2 - 9 = 0) (h2 : a + 3 ≠ 0) : a = 3 :=
by
  sorry

end quadratic_root_and_a_value_l2360_236015
