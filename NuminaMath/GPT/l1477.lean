import Mathlib

namespace mean_and_variance_of_y_l1477_147778

noncomputable def mean (xs : List ℝ) : ℝ :=
  if h : xs.length > 0 then xs.sum / xs.length else 0

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  if h : xs.length > 0 then (xs.map (λ x => (x - m)^2)).sum / xs.length else 0

theorem mean_and_variance_of_y
  (x : List ℝ)
  (hx_len : x.length = 20)
  (hx_mean : mean x = 1)
  (hx_var : variance x = 8) :
  let y := x.map (λ xi => 2 * xi + 3)
  mean y = 5 ∧ variance y = 32 :=
by
  let y := x.map (λ xi => 2 * xi + 3)
  sorry

end mean_and_variance_of_y_l1477_147778


namespace find_pairs_satisfying_conditions_l1477_147764

theorem find_pairs_satisfying_conditions :
  ∀ (m n : ℕ), (0 < m ∧ 0 < n) →
               (∃ k : ℤ, m^2 - 4 * n = k^2) →
               (∃ l : ℤ, n^2 - 4 * m = l^2) →
               (m = 4 ∧ n = 4) ∨ (m = 5 ∧ n = 6) ∨ (m = 6 ∧ n = 5) :=
by
  intros m n hmn h1 h2
  sorry

end find_pairs_satisfying_conditions_l1477_147764


namespace multiple_of_other_number_l1477_147725

theorem multiple_of_other_number (S L k : ℤ) (h₁ : S = 18) (h₂ : L = k * S - 3) (h₃ : S + L = 51) : k = 2 :=
by
  sorry

end multiple_of_other_number_l1477_147725


namespace speed_of_mother_minimum_running_time_l1477_147790

namespace XiaotongTravel

def distance_to_binjiang : ℝ := 4320
def time_diff : ℝ := 12
def speed_rate : ℝ := 1.2

theorem speed_of_mother : 
  ∃ (x : ℝ), (distance_to_binjiang / x - distance_to_binjiang / (speed_rate * x) = time_diff) → (speed_rate * x = 72) :=
sorry

def distance_to_company : ℝ := 2940
def running_speed : ℝ := 150
def total_time : ℝ := 30

theorem minimum_running_time :
  ∃ (y : ℝ), ((distance_to_company - running_speed * y) / 72 + y ≤ total_time) → (y ≥ 10) :=
sorry

end XiaotongTravel

end speed_of_mother_minimum_running_time_l1477_147790


namespace square_area_l1477_147743

noncomputable def side_length_x (x : ℚ) : Prop :=
5 * x - 20 = 30 - 4 * x

noncomputable def side_length_s : ℚ :=
70 / 9

noncomputable def area_of_square : ℚ :=
(side_length_s)^2

theorem square_area (x : ℚ) (h : side_length_x x) : area_of_square = 4900 / 81 :=
sorry

end square_area_l1477_147743


namespace new_student_weight_l1477_147727

theorem new_student_weight (W_new : ℝ) (W : ℝ) (avg_decrease : ℝ) (num_students : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_decrease = 5 → old_weight = 86 → num_students = 8 →
  W_new = W - old_weight + new_weight → W_new = W - avg_decrease * num_students →
  new_weight = 46 :=
by
  intros avg_decrease_eq old_weight_eq num_students_eq W_new_eq avg_weight_decrease_eq
  rw [avg_decrease_eq, old_weight_eq, num_students_eq] at *
  sorry

end new_student_weight_l1477_147727


namespace calc_154_1836_minus_54_1836_l1477_147714

-- Statement of the problem in Lean 4
theorem calc_154_1836_minus_54_1836 : 154 * 1836 - 54 * 1836 = 183600 :=
by
  sorry

end calc_154_1836_minus_54_1836_l1477_147714


namespace infinite_geometric_series_sum_l1477_147750

theorem infinite_geometric_series_sum : 
  (∃ (a r : ℚ), a = 5/4 ∧ r = 1/3) → 
  ∑' n : ℕ, ((5/4 : ℚ) * (1/3 : ℚ) ^ n) = (15/8 : ℚ) :=
by
  sorry

end infinite_geometric_series_sum_l1477_147750


namespace right_triangle_area_l1477_147746

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := 1 / 2 * a * b

theorem right_triangle_area {a b : ℝ} 
  (h1 : a + b = 4) 
  (h2 : a^2 + b^2 = 14) : 
  area_of_right_triangle a b = 1 / 2 :=
by 
  sorry

end right_triangle_area_l1477_147746


namespace TV_cost_l1477_147736

theorem TV_cost (savings_furniture_fraction : ℚ)
                (original_savings : ℝ)
                (spent_on_furniture : ℝ)
                (spent_on_TV : ℝ)
                (hfurniture : savings_furniture_fraction = 2/4)
                (hsavings : original_savings = 600)
                (hspent_furniture : spent_on_furniture = original_savings * savings_furniture_fraction) :
                spent_on_TV = 300 := 
sorry

end TV_cost_l1477_147736


namespace percentage_more_research_l1477_147799

-- Defining the various times spent
def acclimation_period : ℝ := 1
def learning_basics_period : ℝ := 2
def dissertation_fraction : ℝ := 0.5
def total_time : ℝ := 7

-- Defining the time spent on each activity
def dissertation_period := dissertation_fraction * acclimation_period
def research_period := total_time - acclimation_period - learning_basics_period - dissertation_period

-- The main theorem to prove
theorem percentage_more_research : 
  ((research_period - learning_basics_period) / learning_basics_period) * 100 = 75 :=
by
  -- Placeholder for the proof
  sorry

end percentage_more_research_l1477_147799


namespace arith_seq_ratio_l1477_147779

theorem arith_seq_ratio (a_2 a_3 S_4 S_5 : ℕ) 
  (arithmetic_seq : ∀ n : ℕ, ℕ)
  (sum_of_first_n_terms : ∀ n : ℕ, ℕ)
  (h1 : (a_2 : ℚ) / a_3 = 1 / 3) 
  (h2 : S_4 = 4 * (a_2 - (a_3 - a_2)) + ((4 * 3 * (a_3 - a_2)) / 2)) 
  (h3 : S_5 = 5 * (a_2 - (a_3 - a_2)) + ((5 * 4 * (a_3 - a_2)) / 2)) :
  (S_4 : ℚ) / S_5 = 8 / 15 :=
by sorry

end arith_seq_ratio_l1477_147779


namespace system_infinite_solutions_a_eq_neg2_l1477_147789

theorem system_infinite_solutions_a_eq_neg2 
  (x y a : ℝ)
  (h1 : 2 * x + 2 * y = -1)
  (h2 : 4 * x + a^2 * y = a) 
  (infinitely_many_solutions : ∃ (a : ℝ), ∀ (c : ℝ), 4 * x + a^2 * y = c) :
  a = -2 :=
by
  sorry

end system_infinite_solutions_a_eq_neg2_l1477_147789


namespace tangent_line_at_origin_l1477_147793

-- Define the function f(x) = x^3 + ax with an extremum at x = 1
def f (x a : ℝ) : ℝ := x^3 + a * x

-- Define the condition for a local extremum at x = 1: f'(1) = 0
def extremum_condition (a : ℝ) : Prop := (3 * 1^2 + a = 0)

-- Define the derivative of f at x = 0
def derivative_at_origin (a : ℝ) : ℝ := 3 * 0^2 + a

-- Define the value of function at x = 0
def value_at_origin (a : ℝ) : ℝ := f 0 a

-- The main theorem to prove
theorem tangent_line_at_origin (a : ℝ) (ha : extremum_condition a) :
    (value_at_origin a = 0) ∧ (derivative_at_origin a = -3) → ∀ x, (3 * x + (f x a - f 0 a) / (x - 0) = 0) := by
  sorry

end tangent_line_at_origin_l1477_147793


namespace smallest_n_for_good_sequence_l1477_147752

def is_good_sequence (a : ℕ → ℝ) : Prop :=
   (∃ (a_0 : ℕ), a 0 = a_0) ∧
   (∀ i : ℕ, a (i+1) = 2 * a i + 1 ∨ a (i+1) = a i / (a i + 2)) ∧
   (∃ k : ℕ, a k = 2014)

theorem smallest_n_for_good_sequence : 
  ∀ (a : ℕ → ℝ), is_good_sequence a → ∃ n : ℕ, a n = 2014 ∧ ∀ m : ℕ, m < n → a m ≠ 2014 :=
sorry

end smallest_n_for_good_sequence_l1477_147752


namespace tetrahedron_point_choice_l1477_147758

-- Definitions
variables (h s1 s2 : ℝ) -- h, s1, s2 are positive real numbers
variables (A B C : ℝ)  -- A, B, C can be points in space

-- Hypothetical tetrahedron face areas and height
def height_condition (D : ℝ) : Prop := -- D is a point in space
  ∃ (D_height : ℝ), D_height = h

def area_ACD_condition (D : ℝ) : Prop := 
  ∃ (area_ACD : ℝ), area_ACD = s1

def area_BCD_condition (D : ℝ) : Prop := 
  ∃ (area_BCD : ℝ), area_BCD = s2

-- The main theorem
theorem tetrahedron_point_choice : 
  ∃ D, height_condition h D ∧ area_ACD_condition s1 D ∧ area_BCD_condition s2 D :=
sorry

end tetrahedron_point_choice_l1477_147758


namespace calculate_neg_three_minus_one_l1477_147773

theorem calculate_neg_three_minus_one : -3 - 1 = -4 := by
  sorry

end calculate_neg_three_minus_one_l1477_147773


namespace percentage_increase_l1477_147787

variable (x r : ℝ)

theorem percentage_increase (h_x : x = 78.4) (h_r : x = 70 * (1 + r)) : r = 0.12 :=
by
  -- Proof goes here
  sorry

end percentage_increase_l1477_147787


namespace power_function_evaluation_l1477_147741

theorem power_function_evaluation (f : ℝ → ℝ) (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = (Real.sqrt 2) / 2) :
  f 4 = 1 / 2 := by
  sorry

end power_function_evaluation_l1477_147741


namespace mean_of_six_numbers_l1477_147700

theorem mean_of_six_numbers (a b c d e f : ℚ) (h : a + b + c + d + e + f = 1 / 3) :
  (a + b + c + d + e + f) / 6 = 1 / 18 :=
by
  sorry

end mean_of_six_numbers_l1477_147700


namespace find_F_l1477_147723

theorem find_F (C F : ℝ) 
  (h1 : C = 7 / 13 * (F - 40))
  (h2 : C = 26) :
  F = 88.2857 :=
by
  sorry

end find_F_l1477_147723


namespace maximize_product_l1477_147780

theorem maximize_product (x y z : ℝ) (h1 : x ≥ 20) (h2 : y ≥ 40) (h3 : z ≥ 1675) (h4 : x + y + z = 2015) :
  x * y * z ≤ 721480000 / 27 :=
by sorry

end maximize_product_l1477_147780


namespace number_of_lawns_mowed_l1477_147769

noncomputable def ChargePerLawn : ℕ := 33
noncomputable def TotalTips : ℕ := 30
noncomputable def TotalEarnings : ℕ := 558

theorem number_of_lawns_mowed (L : ℕ) 
  (h1 : ChargePerLawn * L + TotalTips = TotalEarnings) : L = 16 := 
by
  sorry

end number_of_lawns_mowed_l1477_147769


namespace monkeys_bananas_minimum_l1477_147795

theorem monkeys_bananas_minimum (b1 b2 b3 : ℕ) (x y z : ℕ) : 
  (x = 2 * y) ∧ (z = (2 * y) / 3) ∧ 
  (x = (2 * b1) / 3 + (b2 / 3) + (5 * b3) / 12) ∧ 
  (y = (b1 / 6) + (b2 / 3) + (5 * b3) / 12) ∧ 
  (z = (b1 / 6) + (b2 / 3) + (b3 / 6)) →
  b1 = 324 ∧ b2 = 162 ∧ b3 = 72 ∧ (b1 + b2 + b3 = 558) :=
sorry

end monkeys_bananas_minimum_l1477_147795


namespace average_is_correct_l1477_147782

def nums : List ℝ := [13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_is_correct :
  (nums.sum / nums.length) = 125830.8 :=
by sorry

end average_is_correct_l1477_147782


namespace probability_cooking_is_one_fourth_l1477_147731
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l1477_147731


namespace jose_fewer_rocks_l1477_147747

theorem jose_fewer_rocks (J : ℕ) (H1 : 80 = J + 14) (H2 : J + 20 = 86) (H3 : J < 80) : J = 66 :=
by
  -- Installation of other conditions derived from the proof
  have H_albert_collected : 86 = 80 + 6 := by rfl
  have J_def : J = 86 - 20 := by sorry
  sorry

end jose_fewer_rocks_l1477_147747


namespace find_k_l1477_147748

theorem find_k 
  (k : ℝ) 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 3 * x + 5)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 7) 
  (intersection : ∃ x y : ℝ, (y = 3 * x + 5) ∧ (y = k * x - 7) ∧ x = -4 ∧ y = -7) :
  k = 0 :=
by
  sorry

end find_k_l1477_147748


namespace sequence_property_l1477_147762

theorem sequence_property
  (b : ℝ) (h₀ : b > 0)
  (u : ℕ → ℝ)
  (h₁ : u 1 = b)
  (h₂ : ∀ n ≥ 1, u (n + 1) = 1 / (2 - u n)) :
  u 10 = (4 * b - 3) / (6 * b - 5) :=
by
  sorry

end sequence_property_l1477_147762


namespace num_ways_to_place_balls_in_boxes_l1477_147716

theorem num_ways_to_place_balls_in_boxes (num_balls num_boxes : ℕ) (hB : num_balls = 4) (hX : num_boxes = 3) : 
  (num_boxes ^ num_balls) = 81 := by
  rw [hB, hX]
  sorry

end num_ways_to_place_balls_in_boxes_l1477_147716


namespace compare_negatives_l1477_147711

theorem compare_negatives : -2 > -3 :=
by
  sorry

end compare_negatives_l1477_147711


namespace tan_theta_condition_l1477_147738

open Real

theorem tan_theta_condition (k : ℤ) : 
  (∃ θ : ℝ, θ = 2 * k * π + π / 4 ∧ tan θ = 1) ∧ ¬ (∀ θ : ℝ, tan θ = 1 → ∃ k : ℤ, θ = 2 * k * π + π / 4) :=
by sorry

end tan_theta_condition_l1477_147738


namespace pencil_notebook_cost_l1477_147761

theorem pencil_notebook_cost (p n : ℝ)
  (h1 : 9 * p + 10 * n = 5.35)
  (h2 : 6 * p + 4 * n = 2.50) :
  24 * 0.9 * p + 15 * n = 9.24 :=
by 
  sorry

end pencil_notebook_cost_l1477_147761


namespace olympiad_divisors_l1477_147717

theorem olympiad_divisors :
  {n : ℕ | n > 0 ∧ n ∣ (1998 + n)} = {n : ℕ | n > 0 ∧ n ∣ 1998} :=
by {
  sorry
}

end olympiad_divisors_l1477_147717


namespace fib_fact_last_two_sum_is_five_l1477_147767

def fib_fact_last_two_sum (s : List (Fin 100)) : Fin 100 :=
  s.sum

theorem fib_fact_last_two_sum_is_five :
  fib_fact_last_two_sum [1, 1, 2, 6, 20, 20, 0] = 5 :=
by 
  sorry

end fib_fact_last_two_sum_is_five_l1477_147767


namespace ratio_of_sums_l1477_147785

open Nat

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 8 - 2 * a 3) / 7)

def arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem ratio_of_sums
    (a : ℕ → ℝ)
    (S : ℕ → ℝ)
    (a_arith : arithmetic_sequence_property a 1)
    (s_def : ∀ n, S n = sum_of_first_n_terms a n)
    (a8_eq_2a3 : a 8 = 2 * a 3) :
  S 15 / S 5 = 6 :=
sorry

end ratio_of_sums_l1477_147785


namespace min_time_to_cross_river_l1477_147784

-- Definitions for the time it takes each horse to cross the river
def timeA : ℕ := 2
def timeB : ℕ := 3
def timeC : ℕ := 7
def timeD : ℕ := 6

-- Definition for the minimum time required for all horses to cross the river
def min_crossing_time : ℕ := 18

-- Theorem stating the problem: 
theorem min_time_to_cross_river :
  ∀ (timeA timeB timeC timeD : ℕ), timeA = 2 → timeB = 3 → timeC = 7 → timeD = 6 →
  min_crossing_time = 18 :=
sorry

end min_time_to_cross_river_l1477_147784


namespace cake_cost_is_20_l1477_147765

-- Define the given conditions
def total_budget : ℕ := 50
def additional_needed : ℕ := 11
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

-- Define the derived conditions
def total_cost : ℕ := total_budget + additional_needed
def combined_bouquet_balloons_cost : ℕ := bouquet_cost + balloons_cost
def cake_cost : ℕ := total_cost - combined_bouquet_balloons_cost

-- The theorem to be proved
theorem cake_cost_is_20 : cake_cost = 20 :=
by
  -- proof steps are not required
  sorry

end cake_cost_is_20_l1477_147765


namespace total_cookies_l1477_147720

theorem total_cookies (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) : bags * cookies_per_bag = 703 :=
by
  sorry

end total_cookies_l1477_147720


namespace gear_p_revolutions_per_minute_l1477_147794

theorem gear_p_revolutions_per_minute (r : ℝ) 
  (cond2 : ℝ := 40) 
  (cond3 : 1.5 * r + 45 = 1.5 * 40) :
  r = 10 :=
by
  sorry

end gear_p_revolutions_per_minute_l1477_147794


namespace determine_x_l1477_147704

theorem determine_x
  (w : ℤ) (z : ℤ) (y : ℤ) (x : ℤ)
  (h₁ : w = 90)
  (h₂ : z = w + 25)
  (h₃ : y = z + 12)
  (h₄ : x = y + 7) : x = 134 :=
by
  sorry

end determine_x_l1477_147704


namespace circle_center_distance_travelled_l1477_147701

theorem circle_center_distance_travelled :
  ∀ (r : ℝ) (a b c : ℝ), r = 2 ∧ a = 9 ∧ b = 12 ∧ c = 15 → (a^2 + b^2 = c^2) → 
  ∃ (d : ℝ), d = 24 :=
by
  intros r a b c h1 h2
  sorry

end circle_center_distance_travelled_l1477_147701


namespace quadratic_range_l1477_147742

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x + 7

-- Defining the range of the quadratic function for the interval -1 < x < 4
theorem quadratic_range (y : ℝ) : 3 ≤ y ∧ y < 12 ↔ ∃ x : ℝ, -1 < x ∧ x < 4 ∧ y = quadratic_function x :=
by
  sorry

end quadratic_range_l1477_147742


namespace max_silver_coins_l1477_147724

theorem max_silver_coins (n : ℕ) : (n < 150) ∧ (n % 15 = 3) → n = 138 :=
by
  sorry

end max_silver_coins_l1477_147724


namespace complex_solution_count_l1477_147786

theorem complex_solution_count : 
  ∃ (s : Finset ℂ), (∀ z ∈ s, (z^3 - 8) / (z^2 - 3 * z + 2) = 0) ∧ s.card = 2 := 
by
  sorry

end complex_solution_count_l1477_147786


namespace y_share_per_x_l1477_147713

theorem y_share_per_x (total_amount y_share : ℝ) (z_share_per_x : ℝ) 
  (h_total : total_amount = 234)
  (h_y_share : y_share = 54)
  (h_z_share_per_x : z_share_per_x = 0.5) :
  ∃ a : ℝ, (forall x : ℝ, y_share = a * x) ∧ a = 9 / 20 :=
by
  use 9 / 20
  intros
  sorry

end y_share_per_x_l1477_147713


namespace triangle_land_area_l1477_147791

theorem triangle_land_area :
  let base_cm := 12
  let height_cm := 9
  let scale_cm_to_miles := 3
  let square_mile_to_acres := 640
  let area_cm2 := (1 / 2 : Float) * base_cm * height_cm
  let area_miles2 := area_cm2 * (scale_cm_to_miles ^ 2)
  let area_acres := area_miles2 * square_mile_to_acres
  area_acres = 311040 :=
by
  -- Skipped proofs
  sorry

end triangle_land_area_l1477_147791


namespace no_four_digit_number_differs_from_reverse_by_1008_l1477_147737

theorem no_four_digit_number_differs_from_reverse_by_1008 :
  ∀ a b c d : ℕ, 
  a < 10 → b < 10 → c < 10 → d < 10 → (999 * (a - d) + 90 * (b - c) ≠ 1008) :=
by
  intro a b c d ha hb hc hd h
  sorry

end no_four_digit_number_differs_from_reverse_by_1008_l1477_147737


namespace building_height_is_74_l1477_147772

theorem building_height_is_74
  (building_shadow : ℚ)
  (flagpole_height : ℚ)
  (flagpole_shadow : ℚ)
  (ratio_valid : building_shadow / flagpole_shadow = 21 / 8)
  (flagpole_height_value : flagpole_height = 28)
  (building_shadow_value : building_shadow = 84)
  (flagpole_shadow_value : flagpole_shadow = 32) :
  ∃ (h : ℚ), h = 74 := by
  sorry

end building_height_is_74_l1477_147772


namespace value_of_x_l1477_147756

theorem value_of_x (x : ℝ) : (9 - x) ^ 2 = x ^ 2 → x = 4.5 :=
by
  sorry

end value_of_x_l1477_147756


namespace length_of_AB_l1477_147792

-- Given the conditions and the question to prove, we write:
theorem length_of_AB (AB CD : ℝ) (h : ℝ) 
  (area_ABC : ℝ := 0.5 * AB * h) 
  (area_ADC : ℝ := 0.5 * CD * h)
  (ratio_areas : area_ABC / area_ADC = 5 / 2)
  (sum_AB_CD : AB + CD = 280) :
  AB = 200 :=
by
  sorry

end length_of_AB_l1477_147792


namespace net_loss_is_1_percent_l1477_147796

noncomputable def net_loss_percent (CP SP1 SP2 SP3 SP4 : ℝ) : ℝ :=
  let TCP := 4 * CP
  let TSP := SP1 + SP2 + SP3 + SP4
  ((TCP - TSP) / TCP) * 100

theorem net_loss_is_1_percent
  (CP : ℝ)
  (HCP : CP = 1000)
  (SP1 : ℝ)
  (HSP1 : SP1 = CP * 1.1 * 0.95)
  (SP2 : ℝ)
  (HSP2 : SP2 = (CP * 0.9) * 1.02)
  (SP3 : ℝ)
  (HSP3 : SP3 = (CP * 1.2) * 1.03)
  (SP4 : ℝ)
  (HSP4 : SP4 = (CP * 0.75) * 1.01) :
  abs (net_loss_percent CP SP1 SP2 SP3 SP4 + 1.09) < 0.01 :=
by
  -- Proof omitted
  sorry

end net_loss_is_1_percent_l1477_147796


namespace number_of_x_for_P_eq_zero_l1477_147776

noncomputable def P (x : ℝ) : ℂ :=
  1 + Complex.exp (Complex.I * x) - Complex.exp (2 * Complex.I * x) + Complex.exp (3 * Complex.I * x) - Complex.exp (4 * Complex.I * x)

theorem number_of_x_for_P_eq_zero : 
  ∃ (n : ℕ), n = 4 ∧ ∃ (xs : Fin n → ℝ), (∀ i, 0 ≤ xs i ∧ xs i < 2 * Real.pi ∧ P (xs i) = 0) ∧ Function.Injective xs := 
sorry

end number_of_x_for_P_eq_zero_l1477_147776


namespace linear_correlation_test_l1477_147733

theorem linear_correlation_test (n1 n2 n3 n4 : ℕ) (r1 r2 r3 r4 : ℝ) :
  n1 = 10 ∧ r1 = 0.9533 →
  n2 = 15 ∧ r2 = 0.3012 →
  n3 = 17 ∧ r3 = 0.9991 →
  n4 = 3  ∧ r4 = 0.9950 →
  abs r1 > abs r2 ∧ abs r3 > abs r4 →
  (abs r1 > abs r2 → abs r1 > abs r4) →
  (abs r3 > abs r2 → abs r3 > abs r4) →
  abs r1 ≠ abs r2 →
  abs r3 ≠ abs r4 →
  true := 
sorry

end linear_correlation_test_l1477_147733


namespace combined_size_UK_India_US_l1477_147703

theorem combined_size_UK_India_US (U : ℝ)
    (Canada : ℝ := 1.5 * U)
    (Russia : ℝ := (1 + 1/3) * Canada)
    (China : ℝ := (1 / 1.7) * Russia)
    (Brazil : ℝ := (2 / 3) * U)
    (Australia : ℝ := (1 / 2) * Brazil)
    (UK : ℝ := 2 * Australia)
    (India : ℝ := (1 / 4) * Russia)
    (India' : ℝ := 6 * UK)
    (h_India : India = India') :
  UK + India = 7 / 6 * U := 
by
  -- Proof details
  sorry

end combined_size_UK_India_US_l1477_147703


namespace find_triplets_l1477_147726

theorem find_triplets (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a ^ b ∣ b ^ c - 1) ∧ (a ^ c ∣ c ^ b - 1)) ↔ (a = 1 ∨ (b = 1 ∧ c = 1)) :=
by sorry

end find_triplets_l1477_147726


namespace range_of_a_plus_b_l1477_147749

noncomputable def range_of_sum_of_sides (a b : ℝ) (c : ℝ) : Prop :=
  (2 < a + b ∧ a + b ≤ 4)

theorem range_of_a_plus_b
  (a b c : ℝ)
  (h1 : (2 * (b ^ 2 - (1/2) * a * b) = b ^ 2 + 4 - a ^ 2))
  (h2 : c = 2) :
  range_of_sum_of_sides a b c :=
by
  -- Proof would go here, but it's omitted as per the instructions.
  sorry

end range_of_a_plus_b_l1477_147749


namespace find_a5_a7_l1477_147706

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom h1 : a 1 + a 3 = 2
axiom h2 : a 3 + a 5 = 4

theorem find_a5_a7 (a : ℕ → ℤ) (d : ℤ) (h_seq : is_arithmetic_sequence a d)
  (h1 : a 1 + a 3 = 2) (h2 : a 3 + a 5 = 4) : a 5 + a 7 = 6 :=
sorry

end find_a5_a7_l1477_147706


namespace meals_calculation_l1477_147710

def combined_meals (k a : ℕ) : ℕ :=
  k + a

theorem meals_calculation :
  ∀ (k a : ℕ), k = 8 → (2 * a = k) → combined_meals k a = 12 :=
  by
    intros k a h1 h2
    rw [h1] at h2
    have ha : a = 4 := by linarith
    rw [h1, ha]
    unfold combined_meals
    sorry

end meals_calculation_l1477_147710


namespace remainder_when_a_squared_times_b_divided_by_n_l1477_147783

theorem remainder_when_a_squared_times_b_divided_by_n (n : ℕ) (a : ℤ) (h1 : a * 3 ≡ 1 [ZMOD n]) : 
  (a^2 * 3) % n = a % n := 
by
  sorry

end remainder_when_a_squared_times_b_divided_by_n_l1477_147783


namespace average_mark_is_correct_l1477_147739

-- Define the maximum score in the exam
def max_score := 1100

-- Define the percentages scored by Amar, Bhavan, Chetan, and Deepak
def score_percentage_amar := 64 / 100
def score_percentage_bhavan := 36 / 100
def score_percentage_chetan := 44 / 100
def score_percentage_deepak := 52 / 100

-- Calculate the actual scores based on percentages
def score_amar := score_percentage_amar * max_score
def score_bhavan := score_percentage_bhavan * max_score
def score_chetan := score_percentage_chetan * max_score
def score_deepak := score_percentage_deepak * max_score

-- Define the total score
def total_score := score_amar + score_bhavan + score_chetan + score_deepak

-- Define the number of students
def number_of_students := 4

-- Define the average score
def average_score := total_score / number_of_students

-- The theorem to prove that the average score is 539
theorem average_mark_is_correct : average_score = 539 := by
  -- Proof skipped
  sorry

end average_mark_is_correct_l1477_147739


namespace simplify_and_evaluate_correct_l1477_147766

noncomputable def simplify_and_evaluate (x y : ℚ) : ℚ :=
  3 * (x^2 - 2 * x * y) - (3 * x^2 - 2 * y + 2 * (x * y + y))

theorem simplify_and_evaluate_correct : 
  simplify_and_evaluate (-1 / 2 : ℚ) (-3 : ℚ) = -12 := by
  sorry

end simplify_and_evaluate_correct_l1477_147766


namespace triangle_a_c_sin_A_minus_B_l1477_147745

theorem triangle_a_c_sin_A_minus_B (a b c : ℝ) (A B C : ℝ):
  a + c = 6 → b = 2 → Real.cos B = 7/9 →
  a = 3 ∧ c = 3 ∧ Real.sin (A - B) = (10 * Real.sqrt 2) / 27 :=
by
  intro h1 h2 h3
  sorry

end triangle_a_c_sin_A_minus_B_l1477_147745


namespace max_integer_a_l1477_147729

theorem max_integer_a :
  ∀ (a: ℤ), (∀ x: ℝ, (a + 1) * x^2 - 2 * x + 3 = 0 → (a = -2 → (-12 * a - 8) ≥ 0)) → (∀ a ≤ -2, a ≠ -1) :=
by
  sorry

end max_integer_a_l1477_147729


namespace cube_root_eval_l1477_147707

noncomputable def cube_root_nested (N : ℝ) : ℝ := (N * (N * (N * (N)))) ^ (1/81)

theorem cube_root_eval (N : ℝ) (h : N > 1) : 
  cube_root_nested N = N ^ (40 / 81) := 
sorry

end cube_root_eval_l1477_147707


namespace largest_r_in_subset_l1477_147755

theorem largest_r_in_subset (A : Finset ℕ) (hA : A.card = 500) : 
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ (B ∩ C).card ≥ 100 := sorry

end largest_r_in_subset_l1477_147755


namespace Tyrone_total_money_l1477_147763

theorem Tyrone_total_money :
  let usd_bills := 4 * 1 + 1 * 10 + 2 * 5 + 30 * 0.25 + 5 * 0.5 + 48 * 0.1 + 12 * 0.05 + 4 * 1 + 64 * 0.01 + 3 * 2 + 5 * 0.5
  let euro_to_usd := 20 * 1.1
  let pound_to_usd := 15 * 1.32
  let cad_to_usd := 6 * 0.76
  let total_usd_currency := usd_bills
  let total_foreign_usd_currency := euro_to_usd + pound_to_usd + cad_to_usd
  let total_money := total_usd_currency + total_foreign_usd_currency
  total_money = 98.90 :=
by
  sorry

end Tyrone_total_money_l1477_147763


namespace variance_defect_rate_l1477_147702

noncomputable def defect_rate : ℝ := 0.02
noncomputable def number_of_trials : ℕ := 100
noncomputable def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem variance_defect_rate :
  variance_binomial number_of_trials defect_rate = 1.96 :=
by
  sorry

end variance_defect_rate_l1477_147702


namespace remainder_of_2357916_div_8_l1477_147705

theorem remainder_of_2357916_div_8 : (2357916 % 8) = 4 := by
  sorry

end remainder_of_2357916_div_8_l1477_147705


namespace required_folders_l1477_147735

def pencil_cost : ℝ := 0.5
def folder_cost : ℝ := 0.9
def pencil_count : ℕ := 24
def total_cost : ℝ := 30

theorem required_folders : ∃ (folders : ℕ), folders = 20 ∧ 
  (pencil_count * pencil_cost + folders * folder_cost = total_cost) :=
sorry

end required_folders_l1477_147735


namespace total_profit_l1477_147754

-- Definitions based on the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Statement of the theorem
theorem total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end total_profit_l1477_147754


namespace value_of_k_l1477_147768

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 6
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k * x - 8

theorem value_of_k:
  (f 5) - (g 5 k) = 20 → k = -10.8 :=
by
  sorry

end value_of_k_l1477_147768


namespace blue_segments_count_l1477_147777

def grid_size : ℕ := 16
def total_dots : ℕ := grid_size * grid_size
def red_dots : ℕ := 133
def boundary_red_dots : ℕ := 32
def corner_red_dots : ℕ := 2
def yellow_segments : ℕ := 196

-- Dummy hypotheses representing the given conditions
axiom red_dots_on_grid : red_dots <= total_dots
axiom boundary_red_dots_count : boundary_red_dots = 32
axiom corner_red_dots_count : corner_red_dots = 2
axiom total_yellow_segments : yellow_segments = 196

-- Proving the number of blue line segments
theorem blue_segments_count :  ∃ (blue_segments : ℕ), blue_segments = 134 := 
sorry

end blue_segments_count_l1477_147777


namespace dan_spent_at_music_store_l1477_147730

def cost_of_clarinet : ℝ := 130.30
def cost_of_song_book : ℝ := 11.24
def money_left_in_pocket : ℝ := 12.32
def total_spent : ℝ := 129.22

theorem dan_spent_at_music_store : 
  cost_of_clarinet + cost_of_song_book - money_left_in_pocket = total_spent :=
by
  -- Proof omitted.
  sorry

end dan_spent_at_music_store_l1477_147730


namespace terminating_decimal_expansion_7_over_625_l1477_147781

theorem terminating_decimal_expansion_7_over_625 : (7 / 625 : ℚ) = 112 / 10000 := by
  sorry

end terminating_decimal_expansion_7_over_625_l1477_147781


namespace value_of_expression_l1477_147751

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 2 * x + 5 = 9) : 3 * x^2 + 3 * x - 7 = -1 :=
by
  -- The proof would go here
  sorry

end value_of_expression_l1477_147751


namespace total_handshakes_is_72_l1477_147744

-- Define the conditions
def number_of_players_per_team := 6
def number_of_teams := 2
def number_of_referees := 3

-- Define the total number of players
def total_players := number_of_teams * number_of_players_per_team

-- Define the total number of handshakes between players of different teams
def team_handshakes := number_of_players_per_team * number_of_players_per_team

-- Define the total number of handshakes between players and referees
def player_referee_handshakes := total_players * number_of_referees

-- Define the total number of handshakes
def total_handshakes := team_handshakes + player_referee_handshakes

-- Prove that the total number of handshakes is 72
theorem total_handshakes_is_72 : total_handshakes = 72 := by
  sorry

end total_handshakes_is_72_l1477_147744


namespace find_f_x_sq_minus_2_l1477_147775

-- Define the polynomial and its given condition
def f (x : ℝ) : ℝ := sorry  -- f is some polynomial, we'll leave it unspecified for now

-- Assume the given condition
axiom f_condition : ∀ x : ℝ, f (x^2 + 2) = x^4 + 6 * x^2 + 4

-- Prove the desired result
theorem find_f_x_sq_minus_2 (x : ℝ) : f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
sorry

end find_f_x_sq_minus_2_l1477_147775


namespace range_of_m_l1477_147728

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ (m > 2 ∨ m < -4) :=
by
  sorry

end range_of_m_l1477_147728


namespace solve_thought_of_number_l1477_147718

def thought_of_number (x : ℝ) : Prop :=
  (x / 6) + 5 = 17

theorem solve_thought_of_number :
  ∃ x, thought_of_number x ∧ x = 72 :=
by
  sorry

end solve_thought_of_number_l1477_147718


namespace hotel_rooms_count_l1477_147715

theorem hotel_rooms_count
  (TotalLamps : ℕ) (TotalChairs : ℕ) (TotalBedSheets : ℕ)
  (LampsPerRoom : ℕ) (ChairsPerRoom : ℕ) (BedSheetsPerRoom : ℕ) :
  TotalLamps = 147 → 
  TotalChairs = 84 → 
  TotalBedSheets = 210 → 
  LampsPerRoom = 7 → 
  ChairsPerRoom = 4 → 
  BedSheetsPerRoom = 10 →
  (TotalLamps / LampsPerRoom = 21) ∧ 
  (TotalChairs / ChairsPerRoom = 21) ∧ 
  (TotalBedSheets / BedSheetsPerRoom = 21) :=
by
  intros
  sorry

end hotel_rooms_count_l1477_147715


namespace unique_two_scoop_sundaes_l1477_147788

theorem unique_two_scoop_sundaes (n : ℕ) (hn : n = 8) : ∃ k, k = Nat.choose 8 2 :=
by
  use 28
  sorry

end unique_two_scoop_sundaes_l1477_147788


namespace geometric_sequence_common_ratio_l1477_147797

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 2)
  (h3 : a 5 = 1/4) :
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l1477_147797


namespace hyperbola_eccentricity_l1477_147734

theorem hyperbola_eccentricity (a b c : ℝ) (h₁ : 2 * a = 16) (h₂ : 2 * b = 12) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  (c / a) = 5 / 4 :=
by
  sorry

end hyperbola_eccentricity_l1477_147734


namespace salon_fingers_l1477_147721

theorem salon_fingers (clients non_clients total_fingers cost_per_client total_earnings : Nat)
  (h1 : cost_per_client = 20)
  (h2 : total_earnings = 200)
  (h3 : total_fingers = 210)
  (h4 : non_clients = 11)
  (h_clients : clients = total_earnings / cost_per_client)
  (h_people : total_fingers / 10 = clients + non_clients) :
  10 = total_fingers / (clients + non_clients) :=
by
  sorry

end salon_fingers_l1477_147721


namespace base4_last_digit_390_l1477_147722

theorem base4_last_digit_390 : 
  (Nat.digits 4 390).head! = 2 := sorry

end base4_last_digit_390_l1477_147722


namespace determineFinalCounts_l1477_147732

structure FruitCounts where
  plums : ℕ
  oranges : ℕ
  apples : ℕ
  pears : ℕ
  cherries : ℕ

def initialCounts : FruitCounts :=
  { plums := 10, oranges := 8, apples := 12, pears := 6, cherries := 0 }

def givenAway : FruitCounts :=
  { plums := 4, oranges := 3, apples := 5, pears := 0, cherries := 0 }

def receivedFromSam : FruitCounts :=
  { plums := 2, oranges := 0, apples := 0, pears := 1, cherries := 0 }

def receivedFromBrother : FruitCounts :=
  { plums := 0, oranges := 1, apples := 2, pears := 0, cherries := 0 }

def receivedFromNeighbor : FruitCounts :=
  { plums := 0, oranges := 0, apples := 0, pears := 3, cherries := 2 }

def finalCounts (initial given receivedSam receivedBrother receivedNeighbor : FruitCounts) : FruitCounts :=
  { plums := initial.plums - given.plums + receivedSam.plums,
    oranges := initial.oranges - given.oranges + receivedBrother.oranges,
    apples := initial.apples - given.apples + receivedBrother.apples,
    pears := initial.pears - given.pears + receivedSam.pears + receivedNeighbor.pears,
    cherries := initial.cherries - given.cherries + receivedNeighbor.cherries }

theorem determineFinalCounts :
  finalCounts initialCounts givenAway receivedFromSam receivedFromBrother receivedFromNeighbor =
  { plums := 8, oranges := 6, apples := 9, pears := 10, cherries := 2 } :=
by
  sorry

end determineFinalCounts_l1477_147732


namespace right_triangle_perimeter_l1477_147719

theorem right_triangle_perimeter 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_area : 1/2 * 30 * b = 180)
  (h_pythagorean : c^2 = 30^2 + b^2)
  : a + b + c = 42 + 2 * Real.sqrt 261 :=
sorry

end right_triangle_perimeter_l1477_147719


namespace grocer_rows_count_l1477_147757

theorem grocer_rows_count (n : ℕ) (a d S : ℕ) (h_a : a = 1) (h_d : d = 3) (h_S : S = 225)
  (h_sum : S = n * (2 * a + (n - 1) * d) / 2) : n = 16 :=
by {
  sorry
}

end grocer_rows_count_l1477_147757


namespace prob_factor_less_than_nine_l1477_147771

theorem prob_factor_less_than_nine : 
  (∃ (n : ℕ), n = 72) ∧ (∃ (total_factors : ℕ), total_factors = 12) ∧ 
  (∃ (factors_lt_9 : ℕ), factors_lt_9 = 6) → 
  (6 / 12 : ℚ) = (1 / 2 : ℚ) := 
by
  sorry

end prob_factor_less_than_nine_l1477_147771


namespace cos_double_angle_identity_l1477_147770

open Real

theorem cos_double_angle_identity (α : ℝ) 
  (h : tan (α + π / 4) = 1 / 3) : cos (2 * α) = 3 / 5 :=
sorry

end cos_double_angle_identity_l1477_147770


namespace range_of_a_l1477_147774

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

theorem range_of_a (a : ℝ) (h : a ≠ 1) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ici 1 ∧ f a x₀ < a / (a - 1)) →
  a ∈ Set.Ioo (-Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∨ a ∈ Set.Ioi 1 :=
by sorry

end range_of_a_l1477_147774


namespace remainder_3_45_plus_4_mod_5_l1477_147712

theorem remainder_3_45_plus_4_mod_5 :
  (3 ^ 45 + 4) % 5 = 2 := 
by {
  sorry
}

end remainder_3_45_plus_4_mod_5_l1477_147712


namespace sum_of_drawn_vegetable_oil_and_fruits_vegetables_l1477_147759

-- Definitions based on conditions
def varieties_of_grains : ℕ := 40
def varieties_of_vegetable_oil : ℕ := 10
def varieties_of_animal_products : ℕ := 30
def varieties_of_fruits_vegetables : ℕ := 20
def total_sample_size : ℕ := 20

def sampling_fraction : ℚ := total_sample_size / (varieties_of_grains + varieties_of_vegetable_oil + varieties_of_animal_products + varieties_of_fruits_vegetables)

def expected_drawn_vegetable_oil : ℚ := varieties_of_vegetable_oil * sampling_fraction
def expected_drawn_fruits_vegetables : ℚ := varieties_of_fruits_vegetables * sampling_fraction

-- The theorem to be proved
theorem sum_of_drawn_vegetable_oil_and_fruits_vegetables : 
  expected_drawn_vegetable_oil + expected_drawn_fruits_vegetables = 6 := 
by 
  -- Placeholder for proof
  sorry

end sum_of_drawn_vegetable_oil_and_fruits_vegetables_l1477_147759


namespace necessary_but_not_sufficient_condition_l1477_147753

variables {a b c : ℝ × ℝ}

def nonzero_vector (v : ℝ × ℝ) : Prop := v ≠ (0, 0)

theorem necessary_but_not_sufficient_condition (ha : nonzero_vector a) (hb : nonzero_vector b) (hc : nonzero_vector c) :
  (a.1 * (b.1 - c.1) + a.2 * (b.2 - c.2) = 0) ↔ (b = c) :=
sorry

end necessary_but_not_sufficient_condition_l1477_147753


namespace boys_in_other_communities_l1477_147798

theorem boys_in_other_communities (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℕ)
  (H_total : total_boys = 400)
  (H_muslim : muslim_percent = 44)
  (H_hindu : hindu_percent = 28)
  (H_sikh : sikh_percent = 10) :
  total_boys * (1 - (muslim_percent + hindu_percent + sikh_percent) / 100) = 72 :=
by
  sorry

end boys_in_other_communities_l1477_147798


namespace recreation_percentage_l1477_147708

theorem recreation_percentage (W : ℝ) (hW : W > 0) :
  (0.40 * W) / (0.15 * W) * 100 = 267 := by
  sorry

end recreation_percentage_l1477_147708


namespace correct_weight_misread_l1477_147740

theorem correct_weight_misread (initial_avg correct_avg : ℝ) (num_boys : ℕ) (misread_weight : ℝ)
  (h_initial : initial_avg = 58.4) (h_correct : correct_avg = 58.85) (h_num_boys : num_boys = 20)
  (h_misread_weight : misread_weight = 56) :
  ∃ x : ℝ, x = 65 :=
by
  sorry

end correct_weight_misread_l1477_147740


namespace remainder_7_pow_150_mod_12_l1477_147709

theorem remainder_7_pow_150_mod_12 :
  (7^150) % 12 = 1 := sorry

end remainder_7_pow_150_mod_12_l1477_147709


namespace problem_solution_l1477_147760

noncomputable def f (A B : ℝ) (x : ℝ) : ℝ := A + B / x + x

theorem problem_solution (A B : ℝ) :
  ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 →
  (x * f A B (x + 1 / y) + y * f A B y + y / x = y * f A B (y + 1 / x) + x * f A B x + x / y) :=
by
  sorry

end problem_solution_l1477_147760
