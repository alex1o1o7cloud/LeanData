import Mathlib

namespace problem1_problem2_l1459_145920

noncomputable def f (x : ℝ) : ℝ :=
let m := (2 * Real.cos x, 1)
let n := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
m.1 * n.1 + m.2 * n.2

theorem problem1 :
  ( ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi ) ∧
  ∀ k : ℤ, ∀ x ∈ Set.Icc ((1 : ℝ) * Real.pi / 6 + k * Real.pi) ((2 : ℝ) * Real.pi / 3 + k * Real.pi),
  f x < f (x + (Real.pi / 3)) :=
sorry

theorem problem2 (A : ℝ) (a b c : ℝ) :
  a ≠ 0 ∧ b = 1 ∧ f A = 2 ∧
  0 < A ∧ A < Real.pi ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2  →
  a = Real.sqrt 3 :=
sorry

end problem1_problem2_l1459_145920


namespace ratio_e_f_l1459_145930

theorem ratio_e_f (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  e / f = 9 / 4 :=
sorry

end ratio_e_f_l1459_145930


namespace fraction_inequality_l1459_145976

theorem fraction_inequality {a b : ℝ} (h1 : a < b) (h2 : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end fraction_inequality_l1459_145976


namespace total_legs_in_household_l1459_145995

def number_of_legs (humans children dogs cats : ℕ) (human_legs child_legs dog_legs cat_legs : ℕ) : ℕ :=
  humans * human_legs + children * child_legs + dogs * dog_legs + cats * cat_legs

theorem total_legs_in_household : number_of_legs 2 3 2 1 2 2 4 4 = 22 :=
  by
    -- The statement ensures the total number of legs is 22, given the defined conditions.
    sorry

end total_legs_in_household_l1459_145995


namespace sum_of_two_digit_and_reverse_l1459_145911

theorem sum_of_two_digit_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9)
  (h5 : (10 * a + b) - (10 * b + a) = 9 * (a + b)) : (10 * a + b) + (10 * b + a) = 11 :=
by
  sorry

end sum_of_two_digit_and_reverse_l1459_145911


namespace find_x_l1459_145989

theorem find_x (x : ℚ) (h1 : 8 * x^2 + 9 * x - 2 = 0) (h2 : 16 * x^2 + 35 * x - 4 = 0) : 
  x = 1 / 8 :=
by sorry

end find_x_l1459_145989


namespace sum_of_incircle_areas_l1459_145969

variables {a b c : ℝ} (ABC : Triangle ℝ) (s K r : ℝ)
  (hs : s = (a + b + c) / 2)
  (hK : K = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (hr : r = K / s)

theorem sum_of_incircle_areas :
  let larger_circle_area := π * r^2
  let smaller_circle_area := π * (r / 2)^2
  larger_circle_area + 3 * smaller_circle_area = 7 * π * r^2 / 4 :=
sorry

end sum_of_incircle_areas_l1459_145969


namespace total_boxes_moved_l1459_145971

-- Define a truck's capacity and number of trips
def truck_capacity : ℕ := 4
def trips : ℕ := 218

-- Prove that the total number of boxes is 872
theorem total_boxes_moved : truck_capacity * trips = 872 := by
  sorry

end total_boxes_moved_l1459_145971


namespace geometric_arithmetic_sequence_difference_l1459_145978

theorem geometric_arithmetic_sequence_difference
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (hq : q > 0)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = a 1 * q)
  (ha4 : a 4 = a 1 * q ^ 3)
  (ha5 : a 5 = a 1 * q ^ 4)
  (harith : 2 * (a 4 + 2 * a 5) = 2 * a 2 + (a 4 + 2 * a 5))
  (hS : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 10 - S 4 = 2016 :=
by
  sorry

end geometric_arithmetic_sequence_difference_l1459_145978


namespace non_congruent_squares_on_5x5_grid_l1459_145986

def is_lattice_point (x y : ℕ) : Prop := x ≤ 4 ∧ y ≤ 4

def is_square {a b c d : (ℕ × ℕ)} : Prop :=
((a.1 - b.1)^2 + (a.2 - b.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2) ∧ 
((c.1 - b.1)^2 + (c.2 - b.2)^2 = (a.1 - d.1)^2 + (a.2 - d.2)^2)

def number_of_non_congruent_squares : ℕ :=
  4 + -- Standard squares: 1x1, 2x2, 3x3, 4x4
  2 + -- Diagonal squares: with sides √2 and 2√2
  2   -- Diagonal sides of 1x2 and 1x3 rectangles

theorem non_congruent_squares_on_5x5_grid :
  number_of_non_congruent_squares = 8 :=
by
  -- proof goes here
  sorry

end non_congruent_squares_on_5x5_grid_l1459_145986


namespace time_left_for_nap_l1459_145949

noncomputable def total_time : ℝ := 20
noncomputable def first_train_time : ℝ := 2 + 1
noncomputable def second_train_time : ℝ := 3 + 1
noncomputable def transfer_one_time : ℝ := 0.75 + 0.5
noncomputable def third_train_time : ℝ := 2 + 1
noncomputable def transfer_two_time : ℝ := 1
noncomputable def fourth_train_time : ℝ := 1
noncomputable def transfer_three_time : ℝ := 0.5
noncomputable def fifth_train_time_before_nap : ℝ := 1.5

noncomputable def total_activities_time : ℝ :=
  first_train_time +
  second_train_time +
  transfer_one_time +
  third_train_time +
  transfer_two_time +
  fourth_train_time +
  transfer_three_time +
  fifth_train_time_before_nap

theorem time_left_for_nap : total_time - total_activities_time = 4.75 := by
  sorry

end time_left_for_nap_l1459_145949


namespace molecular_weight_of_one_mole_l1459_145923

theorem molecular_weight_of_one_mole (total_molecular_weight : ℝ) (number_of_moles : ℕ) (h1 : total_molecular_weight = 304) (h2 : number_of_moles = 4) : 
  total_molecular_weight / number_of_moles = 76 := 
by
  sorry

end molecular_weight_of_one_mole_l1459_145923


namespace rounding_proof_l1459_145919

def rounding_question : Prop :=
  let num := 9.996
  let rounded_value := ((num * 100).round / 100)
  rounded_value ≠ 10.00

theorem rounding_proof : rounding_question :=
by
  sorry

end rounding_proof_l1459_145919


namespace opposite_of_neg_2023_l1459_145972

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l1459_145972


namespace avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l1459_145907

open Real
open List

-- Conditions
def x_vals : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y_vals : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
def sum_x : ℝ := 0.6
def sum_y : ℝ := 3.9
def sum_x_squared : ℝ := 0.038
def sum_y_squared : ℝ := 1.6158
def sum_xy : ℝ := 0.2474
def total_root_area : ℝ := 186

-- Proof problems
theorem avg_root_area : (List.sum x_vals / 10) = 0.06 := by
  sorry

theorem avg_volume : (List.sum y_vals / 10) = 0.39 := by
  sorry

theorem correlation_coefficient : 
  let mean_x := List.sum x_vals / 10;
  let mean_y := List.sum y_vals / 10;
  let numerator := List.sum (List.zipWith (λ x y => (x - mean_x) * (y - mean_y)) x_vals y_vals);
  let denominator := sqrt ((List.sum (List.map (λ x => (x - mean_x) ^ 2) x_vals)) * (List.sum (List.map (λ y => (y - mean_y) ^ 2) y_vals)));
  (numerator / denominator) = 0.97 := by 
  sorry

theorem total_volume_estimate : 
  let avg_x := sum_x / 10;
  let avg_y := sum_y / 10;
  (avg_y / avg_x) * total_root_area = 1209 := by
  sorry

end avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l1459_145907


namespace tree_height_relationship_l1459_145929

theorem tree_height_relationship (x : ℕ) : ∃ h : ℕ, h = 80 + 2 * x :=
by
  sorry

end tree_height_relationship_l1459_145929


namespace local_minimum_at_one_l1459_145901

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + a^2 * x

theorem local_minimum_at_one (a : ℝ) (hfmin : ∀ x : ℝ, deriv (f a) x = 3 * a * x^2 - 4 * x + a^2) (h1 : f a 1 = f a 1) : a = 1 :=
sorry

end local_minimum_at_one_l1459_145901


namespace circle_radius_order_l1459_145944

theorem circle_radius_order (r_X r_Y r_Z : ℝ)
  (hX : r_X = π)
  (hY : 2 * π * r_Y = 8 * π)
  (hZ : π * r_Z^2 = 9 * π) :
  r_Z < r_X ∧ r_X < r_Y :=
by {
  sorry
}

end circle_radius_order_l1459_145944


namespace probability_hardcover_liberal_arts_probability_liberal_arts_then_hardcover_l1459_145903

-- Definitions based on the conditions provided
def total_books : ℕ := 100
def liberal_arts_books : ℕ := 40
def hardcover_books : ℕ := 70
def softcover_science_books : ℕ := 20
def hardcover_liberal_arts_books : ℕ := 30
def softcover_liberal_arts_books : ℕ := liberal_arts_books - hardcover_liberal_arts_books
def total_events_2 : ℕ := total_books * total_books

-- Statement part 1: Probability of selecting a hardcover liberal arts book
theorem probability_hardcover_liberal_arts :
  (hardcover_liberal_arts_books : ℝ) / total_books = 0.3 :=
sorry

-- Statement part 2: Probability of selecting a liberal arts book then a hardcover book (with replacement)
theorem probability_liberal_arts_then_hardcover :
  ((liberal_arts_books : ℝ) / total_books) * ((hardcover_books : ℝ) / total_books) = 0.28 :=
sorry

end probability_hardcover_liberal_arts_probability_liberal_arts_then_hardcover_l1459_145903


namespace marquita_gardens_l1459_145928

open Nat

theorem marquita_gardens (num_mancino_gardens : ℕ) 
  (length_mancino_garden width_mancino_garden : ℕ) 
  (num_marquita_gardens : ℕ) 
  (length_marquita_garden width_marquita_garden : ℕ)
  (total_area : ℕ) 
  (h1 : num_mancino_gardens = 3)
  (h2 : length_mancino_garden = 16)
  (h3 : width_mancino_garden = 5)
  (h4 : length_marquita_garden = 8)
  (h5 : width_marquita_garden = 4)
  (h6 : total_area = 304)
  (hmancino_area : num_mancino_gardens * (length_mancino_garden * width_mancino_garden) = 3 * (16 * 5))
  (hcombined_area : total_area = num_mancino_gardens * (length_mancino_garden * width_mancino_garden) + num_marquita_gardens * (length_marquita_garden * width_marquita_garden)) :
  num_marquita_gardens = 2 :=
sorry

end marquita_gardens_l1459_145928


namespace find_a_for_tangent_parallel_l1459_145950

theorem find_a_for_tangent_parallel : 
  ∀ a : ℝ,
  (∀ (x y : ℝ), y = Real.log x - a * x → x = 1 → 2 * x + y - 1 = 0) →
  a = 3 :=
by
  sorry

end find_a_for_tangent_parallel_l1459_145950


namespace workerB_time_to_complete_job_l1459_145914

theorem workerB_time_to_complete_job 
  (time_A : ℝ) (time_together: ℝ) (time_B : ℝ) 
  (h1 : time_A = 5) 
  (h2 : time_together = 3.333333333333333) 
  (h3 : 1 / time_A + 1 / time_B = 1 / time_together) 
  : time_B = 10 := 
  sorry

end workerB_time_to_complete_job_l1459_145914


namespace david_marks_in_physics_l1459_145934

theorem david_marks_in_physics : 
  ∀ (P : ℝ), 
  let english := 72 
  let mathematics := 60 
  let chemistry := 62 
  let biology := 84 
  let average_marks := 62.6 
  let num_subjects := 5 
  let total_marks := average_marks * num_subjects 
  let known_marks := english + mathematics + chemistry + biology 
  total_marks - known_marks = P → P = 35 :=
by
  sorry

end david_marks_in_physics_l1459_145934


namespace possible_divisor_of_p_l1459_145908

theorem possible_divisor_of_p (p q r s : ℕ)
  (hpq : ∃ x y, p = 40 * x ∧ q = 40 * y ∧ Nat.gcd p q = 40)
  (hqr : ∃ u v, q = 45 * u ∧ r = 45 * v ∧ Nat.gcd q r = 45)
  (hrs : ∃ w z, r = 60 * w ∧ s = 60 * z ∧ Nat.gcd r s = 60)
  (hsp : ∃ t, Nat.gcd s p = 100 * t ∧ 100 ≤ Nat.gcd s p ∧ Nat.gcd s p < 1000) :
  7 ∣ p :=
sorry

end possible_divisor_of_p_l1459_145908


namespace inverse_proportion_function_sol_l1459_145952

theorem inverse_proportion_function_sol (k m x : ℝ) (h1 : k ≠ 0) (h2 : (m - 1) * x ^ (m ^ 2 - 2) = k / x) : m = -1 :=
by
  sorry

end inverse_proportion_function_sol_l1459_145952


namespace find_correct_speed_l1459_145931

variables (d t : ℝ) -- Defining distance and time as real numbers

theorem find_correct_speed
  (h1 : d = 30 * (t + 5 / 60))
  (h2 : d = 50 * (t - 5 / 60)) :
  ∃ r : ℝ, r = 37.5 ∧ d = r * t :=
by 
  -- Skip the proof for now
  sorry

end find_correct_speed_l1459_145931


namespace quadratic_inequality_solution_l1459_145947

theorem quadratic_inequality_solution (x : ℝ) : -3 < x ∧ x < 4 → x^2 - x - 12 < 0 := by
  sorry

end quadratic_inequality_solution_l1459_145947


namespace add_A_to_10_eq_15_l1459_145904

theorem add_A_to_10_eq_15 (A : ℕ) (h : A + 10 = 15) : A = 5 :=
sorry

end add_A_to_10_eq_15_l1459_145904


namespace most_persuasive_method_l1459_145902

-- Survey data and conditions
def male_citizens : ℕ := 4258
def male_believe_doping : ℕ := 2360
def female_citizens : ℕ := 3890
def female_believe_framed : ℕ := 2386

def random_division_by_gender : Prop := true -- Represents the random division into male and female groups

-- Proposition to prove
theorem most_persuasive_method : 
  random_division_by_gender → 
  ∃ method : String, method = "Independence Test" := by
  sorry

end most_persuasive_method_l1459_145902


namespace factorize_polynomial_l1459_145985

theorem factorize_polynomial (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := 
by 
  sorry

end factorize_polynomial_l1459_145985


namespace min_students_solving_most_l1459_145939

theorem min_students_solving_most (students problems : Nat) 
    (total_students : students = 10) 
    (problems_per_student : Nat → Nat) 
    (problems_per_student_property : ∀ s, s < students → problems_per_student s = 3) 
    (common_problem : ∀ s1 s2, s1 < students → s2 < students → s1 ≠ s2 → ∃ p, p < problems ∧ (∃ (solves1 solves2 : Nat → Nat), (solves1 p = 1 ∧ solves2 p = 1) ∧ s1 < students ∧ s2 < students)): 
  ∃ min_students, min_students = 5 :=
by
  sorry

end min_students_solving_most_l1459_145939


namespace profit_relationship_profit_range_max_profit_l1459_145958

noncomputable def profit (x : ℝ) : ℝ :=
  -20 * x ^ 2 + 100 * x + 6000

theorem profit_relationship (x : ℝ) :
  profit (x) = (60 - x) * (300 + 20 * x) - 40 * (300 + 20 * x) :=
by
  sorry
  
theorem profit_range (x : ℝ) (h : 0 ≤ x ∧ x < 20) : 
  0 ≤ profit (x) :=
by
  sorry

theorem max_profit (x : ℝ) :
  (2.5 ≤ x ∧ x < 2.6) → profit (x) ≤ 6125 := 
by
  sorry  

end profit_relationship_profit_range_max_profit_l1459_145958


namespace tension_limit_l1459_145983

theorem tension_limit (M m g : ℝ) (hM : 0 < M) (hg : 0 < g) :
  (∀ T, (T = Mg ↔ m = 0) → (∀ ε, 0 < ε → ∃ m₀, m > m₀ → |T - 2 * M * g| < ε)) :=
by 
  sorry

end tension_limit_l1459_145983


namespace number_of_distinct_b_values_l1459_145900

theorem number_of_distinct_b_values : 
  ∃ (b : ℝ) (p q : ℤ), (∀ (x : ℝ), x*x + b*x + 12*b = 0) ∧ 
                        p + q = -b ∧ 
                        p * q = 12 * b ∧ 
                        ∃ n : ℤ, 1 ≤ n ∧ n ≤ 15 :=
sorry

end number_of_distinct_b_values_l1459_145900


namespace height_difference_between_crates_l1459_145959

theorem height_difference_between_crates 
  (n : ℕ) (diameter : ℝ) 
  (height_A : ℝ) (height_B : ℝ) :
  n = 200 →
  diameter = 12 →
  height_A = n / 10 * diameter →
  height_B = n / 20 * (diameter + 6 * Real.sqrt 3) →
  height_A - height_B = 120 - 60 * Real.sqrt 3 :=
sorry

end height_difference_between_crates_l1459_145959


namespace pounds_over_minimum_l1459_145918

def cost_per_pound : ℕ := 3
def minimum_purchase : ℕ := 15
def total_spent : ℕ := 105

theorem pounds_over_minimum : 
  (total_spent / cost_per_pound) - minimum_purchase = 20 :=
by
  sorry

end pounds_over_minimum_l1459_145918


namespace solution_set_of_inequality_l1459_145951

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} := sorry

end solution_set_of_inequality_l1459_145951


namespace minimize_expression_at_9_l1459_145960

noncomputable def minimize_expression (n : ℕ) : ℚ :=
  n / 3 + 27 / n

theorem minimize_expression_at_9 : minimize_expression 9 = 6 := by
  sorry

end minimize_expression_at_9_l1459_145960


namespace part1_part2_l1459_145927

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part (1)
theorem part1 (a : ℝ) (h : a = 3) : (P 3)ᶜ ∩ Q = {x | -2 ≤ x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : (∀ x, x ∈ P a → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P a) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end part1_part2_l1459_145927


namespace product_mod_7_l1459_145994

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end product_mod_7_l1459_145994


namespace max_abs_sum_sqrt2_l1459_145968

theorem max_abs_sum_sqrt2 (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (a : ℝ), (a = |x| + |y| ∧ a ≤ 2 * Real.sqrt 2) ∧ 
             ∀ (x y : ℝ), x^2 + y^2 = 4 → (|x| + |y|) ≤ 2 * Real.sqrt 2 :=
sorry

end max_abs_sum_sqrt2_l1459_145968


namespace optimal_room_rate_to_maximize_income_l1459_145970

noncomputable def max_income (x : ℝ) : ℝ := x * (300 - 0.5 * (x - 200))

theorem optimal_room_rate_to_maximize_income :
  ∀ x, 200 ≤ x → x ≤ 800 → max_income x ≤ max_income 400 :=
by
  sorry

end optimal_room_rate_to_maximize_income_l1459_145970


namespace frog_escape_l1459_145997

theorem frog_escape (wellDepth dayClimb nightSlide escapeDays : ℕ)
  (h_depth : wellDepth = 30)
  (h_dayClimb : dayClimb = 3)
  (h_nightSlide : nightSlide = 2)
  (h_escape : escapeDays = 28) :
  ∃ n, n = escapeDays ∧
       ((wellDepth ≤ (n - 1) * (dayClimb - nightSlide) + dayClimb)) :=
by
  sorry

end frog_escape_l1459_145997


namespace petya_vasya_problem_l1459_145999

theorem petya_vasya_problem :
  ∀ n : ℕ, (∀ x : ℕ, x = 12320 * 10 ^ (10 * n + 1) - 1 →
    (∃ p q : ℕ, (p ≠ q ∧ ∀ r : ℕ, (r ∣ x → (r = p ∨ r = q))))) → n = 0 :=
by
  sorry

end petya_vasya_problem_l1459_145999


namespace number_of_articles_l1459_145991

variables (C S N : ℝ)
noncomputable def gain : ℝ := 3 / 7

-- Cost price of 50 articles is equal to the selling price of N articles
axiom cost_price_eq_selling_price : 50 * C = N * S

-- Selling price is cost price plus gain percentage
axiom selling_price_with_gain : S = C * (1 + gain)

-- Goal: Prove that N = 35
theorem number_of_articles (h1 : 50 * C = N * C * (10 / 7)) : N = 35 := by
  sorry

end number_of_articles_l1459_145991


namespace percentage_length_more_than_breadth_l1459_145992

-- Define the basic conditions
variables {C r l b : ℝ}
variable {p : ℝ}

-- Assume the conditions
def conditions (C r l b : ℝ) : Prop :=
  C = 400 ∧ r = 3 ∧ l = 20 ∧ 20 * b = 400 / 3

-- Define the statement that we want to prove
theorem percentage_length_more_than_breadth (C r l b : ℝ) (h : conditions C r l b) :
  ∃ (p : ℝ), l = b * (1 + p / 100) ∧ p = 200 :=
sorry

end percentage_length_more_than_breadth_l1459_145992


namespace angle_A_and_area_of_triangle_l1459_145921

theorem angle_A_and_area_of_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) (h1 : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) 
(h2 : R = 2) (h3 : b^2 + c^2 = 18) :
  A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 := 
by
  sorry

end angle_A_and_area_of_triangle_l1459_145921


namespace three_digit_integer_conditions_l1459_145942

theorem three_digit_integer_conditions:
  ∃ n : ℕ, 
    n % 5 = 3 ∧ 
    n % 7 = 4 ∧ 
    n % 4 = 2 ∧
    100 ≤ n ∧ n < 1000 ∧ 
    n = 548 :=
sorry

end three_digit_integer_conditions_l1459_145942


namespace sum_even_minus_sum_odd_l1459_145974

theorem sum_even_minus_sum_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 := by
sorry

end sum_even_minus_sum_odd_l1459_145974


namespace all_zero_l1459_145933

def circle_condition (x : Fin 2007 → ℤ) : Prop :=
  ∀ i : Fin 2007, x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) = 2 * (x (i+1) + x (i+2)) + 2 * (x (i+3) + x (i+4))

theorem all_zero (x : Fin 2007 → ℤ) (h : circle_condition x) : ∀ i, x i = 0 :=
sorry

end all_zero_l1459_145933


namespace bank_deposit_exceeds_1000_on_saturday_l1459_145915

theorem bank_deposit_exceeds_1000_on_saturday:
  ∃ n: ℕ, (2 * (3^n - 1) / 2 > 1000) ∧ ((n + 1) % 7 = 0) := by
  sorry

end bank_deposit_exceeds_1000_on_saturday_l1459_145915


namespace points_distance_le_sqrt5_l1459_145940

theorem points_distance_le_sqrt5 :
  ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (0 ≤ (points i).1 ∧ (points i).1 ≤ 4) ∧ (0 ≤ (points i).2 ∧ (points i).2 ≤ 3)) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end points_distance_le_sqrt5_l1459_145940


namespace geometric_sequence_ratios_l1459_145935

theorem geometric_sequence_ratios {n : ℕ} {r : ℝ}
  (h1 : 85 = (1 - r^(2*n)) / (1 - r^2))
  (h2 : 170 = r * 85) :
  r = 2 ∧ 2*n = 8 :=
by
  sorry

end geometric_sequence_ratios_l1459_145935


namespace find_x_given_sin_interval_l1459_145961

open Real

theorem find_x_given_sin_interval (x : ℝ) (h1 : sin x = -3 / 5) (h2 : π < x ∧ x < 3 / 2 * π) :
  x = π + arcsin (3 / 5) :=
sorry

end find_x_given_sin_interval_l1459_145961


namespace congruence_example_l1459_145954

theorem congruence_example (x : ℤ) (h : 5 * x + 3 ≡ 1 [ZMOD 18]) : 3 * x + 8 ≡ 14 [ZMOD 18] :=
sorry

end congruence_example_l1459_145954


namespace lottery_blanks_l1459_145981

theorem lottery_blanks (P B : ℕ) (h₁ : P = 10) (h₂ : (P : ℝ) / (P + B) = 0.2857142857142857) : B = 25 := 
by
  sorry

end lottery_blanks_l1459_145981


namespace bus_journey_distance_l1459_145998

theorem bus_journey_distance (x : ℝ) (h1 : 0 ≤ x)
  (h2 : 0 ≤ 250 - x)
  (h3 : x / 40 + (250 - x) / 60 = 5.2) :
  x = 124 :=
sorry

end bus_journey_distance_l1459_145998


namespace claire_gerbils_l1459_145966

variables (G H : ℕ)

-- Claire's total pets
def total_pets : Prop := G + H = 92

-- One-quarter of the gerbils are male
def male_gerbils (G : ℕ) : ℕ := G / 4

-- One-third of the hamsters are male
def male_hamsters (H : ℕ) : ℕ := H / 3

-- Total males are 25
def total_males : Prop := male_gerbils G + male_hamsters H = 25

theorem claire_gerbils : total_pets G H → total_males G H → G = 68 :=
by
  intro h1 h2
  sorry

end claire_gerbils_l1459_145966


namespace calculate_fraction_l1459_145926

theorem calculate_fraction :
  (18 - 6) / ((3 + 3) * 2) = 1 := by
  sorry

end calculate_fraction_l1459_145926


namespace trig_expression_equality_l1459_145963

theorem trig_expression_equality (α : ℝ) (h : Real.tan α = 1 / 2) : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -4 :=
by
  sorry

end trig_expression_equality_l1459_145963


namespace trig_identity_example_l1459_145917

theorem trig_identity_example :
  256 * (Real.sin (10 * Real.pi / 180)) * (Real.sin (30 * Real.pi / 180)) *
    (Real.sin (50 * Real.pi / 180)) * (Real.sin (70 * Real.pi / 180)) = 16 := by
  sorry

end trig_identity_example_l1459_145917


namespace roots_geom_prog_eq_neg_cbrt_c_l1459_145936

theorem roots_geom_prog_eq_neg_cbrt_c {a b c : ℝ} (h : ∀ (x1 x2 x3 : ℝ), 
  (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ (x3^3 + a * x3^2 + b * x3 + c = 0) ∧ 
  (∃ (r : ℝ), (x2 = r * x1) ∧ (x3 = r^2 * x1))) : 
  ∃ (x : ℝ), (x^3 = c) ∧ (x = - ((c) ^ (1/3))) :=
by 
  sorry

end roots_geom_prog_eq_neg_cbrt_c_l1459_145936


namespace train_total_distance_l1459_145932

theorem train_total_distance (x : ℝ) (h1 : x > 0) 
  (h_speed_avg : 48 = ((3 * x) / (x / 8))) : 
  3 * x = 6 := 
by
  sorry

end train_total_distance_l1459_145932


namespace smallest_n_exists_l1459_145956

def connected (a b : ℕ) : Prop := -- define connection based on a picture not specified here, placeholder
sorry

def not_connected (a b : ℕ) : Prop := ¬ connected a b

def coprime (a n : ℕ) : Prop := ∀ k : ℕ, k > 1 → k ∣ a → ¬ k ∣ n

def common_divisor_greater_than_one (a n : ℕ) : Prop := ∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ n

theorem smallest_n_exists :
  ∃ n : ℕ,
  (n = 35) ∧
  ∀ (numbers : Fin 7 → ℕ),
  (∀ i j, not_connected (numbers i) (numbers j) → coprime (numbers i + numbers j) n) ∧
  (∀ i j, connected (numbers i) (numbers j) → common_divisor_greater_than_one (numbers i + numbers j) n) := 
sorry

end smallest_n_exists_l1459_145956


namespace no_line_bisected_by_P_exists_l1459_145990

theorem no_line_bisected_by_P_exists (P : ℝ × ℝ) (H : ∀ x y : ℝ, (x / 3)^2 - (y / 2)^2 = 1) : 
  P ≠ (2, 1) := 
sorry

end no_line_bisected_by_P_exists_l1459_145990


namespace sqrt_fraction_l1459_145996

theorem sqrt_fraction {a b c : ℝ}
  (h1 : a = Real.sqrt 27)
  (h2 : b = Real.sqrt 243)
  (h3 : c = Real.sqrt 48) :
  (a + b) / c = 3 := by
  sorry

end sqrt_fraction_l1459_145996


namespace solve_for_xy_l1459_145984

-- The conditions given in the problem
variables (x y : ℝ)
axiom cond1 : 1 / 2 * x - y = 5
axiom cond2 : y - 1 / 3 * x = 2

-- The theorem we need to prove
theorem solve_for_xy (x y : ℝ) (cond1 : 1 / 2 * x - y = 5) (cond2 : y - 1 / 3 * x = 2) : 
  x = 42 ∧ y = 16 := sorry

end solve_for_xy_l1459_145984


namespace solution_set_l1459_145937

open Real

noncomputable def f : ℝ → ℝ := sorry -- The function f is abstractly defined
axiom f_point : f 1 = 0 -- f passes through (1, 0)
axiom f_deriv_pos : ∀ (x : ℝ), x > 0 → x * (deriv f x) > 1 -- xf'(x) > 1 for x > 0

theorem solution_set (x : ℝ) : f x ≤ log x ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end solution_set_l1459_145937


namespace major_premise_incorrect_l1459_145922

theorem major_premise_incorrect (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
    ¬ (∀ x y : ℝ, x < y → a^x < a^y) :=
by {
  sorry
}

end major_premise_incorrect_l1459_145922


namespace even_sum_probability_l1459_145980

-- Define the probabilities for the first wheel
def prob_even_first_wheel : ℚ := 3 / 6
def prob_odd_first_wheel : ℚ := 3 / 6

-- Define the probabilities for the second wheel
def prob_even_second_wheel : ℚ := 3 / 4
def prob_odd_second_wheel : ℚ := 1 / 4

-- Probability that the sum of the two selected numbers is even
def prob_even_sum : ℚ :=
  (prob_even_first_wheel * prob_even_second_wheel) +
  (prob_odd_first_wheel * prob_odd_second_wheel)

-- The theorem to prove
theorem even_sum_probability : prob_even_sum = 13 / 24 := by
  sorry

end even_sum_probability_l1459_145980


namespace segment_MN_length_l1459_145979

theorem segment_MN_length
  (A B C D M N : ℝ)
  (hA : A < B)
  (hB : B < C)
  (hC : C < D)
  (hM : M = (A + C) / 2)
  (hN : N = (B + D) / 2)
  (hAD : D - A = 68)
  (hBC : C - B = 26) :
  |M - N| = 21 :=
sorry

end segment_MN_length_l1459_145979


namespace wheat_field_problem_l1459_145957

def equations (x F : ℕ) :=
  (6 * x - 300 = F) ∧ (5 * x + 200 = F)

theorem wheat_field_problem :
  ∃ (x F : ℕ), equations x F ∧ x = 500 ∧ F = 2700 :=
by
  sorry

end wheat_field_problem_l1459_145957


namespace height_of_building_l1459_145977

-- Define the conditions
def height_flagpole : ℝ := 18
def shadow_flagpole : ℝ := 45
def shadow_building : ℝ := 55

-- State the theorem to prove the height of the building
theorem height_of_building (h : ℝ) : (height_flagpole / shadow_flagpole) = (h / shadow_building) → h = 22 :=
by
  sorry

end height_of_building_l1459_145977


namespace inequality_proof_l1459_145987

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (h : 1/a + 1/b + 1/c = a + b + c) :
  1/(2*a + b + c)^2 + 1/(2*b + c + a)^2 + 1/(2*c + a + b)^2 ≤ 3/16 :=
by
  sorry

end inequality_proof_l1459_145987


namespace total_flour_used_l1459_145913

def wheat_flour : ℝ := 0.2
def white_flour : ℝ := 0.1

theorem total_flour_used : wheat_flour + white_flour = 0.3 :=
by
  sorry

end total_flour_used_l1459_145913


namespace second_tap_emptying_time_l1459_145909

theorem second_tap_emptying_time :
  ∀ (T : ℝ), (∀ (f e : ℝ),
  (f = 1 / 3) →
  (∀ (n : ℝ), (n = 1 / 4.5) →
  (n = f - e ↔ e = 1 / T))) →
  T = 9 :=
by
  sorry

end second_tap_emptying_time_l1459_145909


namespace unique_solution_for_y_l1459_145962

def operation (x y : ℝ) : ℝ := 4 * x - 2 * y + x^2 * y

theorem unique_solution_for_y : ∃! (y : ℝ), operation 3 y = 20 :=
by {
  sorry
}

end unique_solution_for_y_l1459_145962


namespace multiple_of_4_and_8_l1459_145946

theorem multiple_of_4_and_8 (a b : ℤ) (h1 : ∃ k1 : ℤ, a = 4 * k1) (h2 : ∃ k2 : ℤ, b = 8 * k2) :
  (∃ k3 : ℤ, b = 4 * k3) ∧ (∃ k4 : ℤ, a - b = 4 * k4) :=
by
  sorry

end multiple_of_4_and_8_l1459_145946


namespace food_insufficiency_l1459_145967

-- Given conditions
def number_of_dogs : ℕ := 5
def food_per_meal : ℚ := 3 / 4
def meals_per_day : ℕ := 3
def initial_food : ℚ := 45
def days_in_two_weeks : ℕ := 14

-- Definitions derived from conditions
def daily_food_per_dog : ℚ := food_per_meal * meals_per_day
def daily_food_for_all_dogs : ℚ := daily_food_per_dog * number_of_dogs
def total_food_in_two_weeks : ℚ := daily_food_for_all_dogs * days_in_two_weeks

-- Proof statement: proving the food consumed exceeds the initial amount
theorem food_insufficiency : total_food_in_two_weeks > initial_food :=
by {
  sorry
}

end food_insufficiency_l1459_145967


namespace nina_max_digits_l1459_145916

-- Define the conditions
def sam_digits (C : ℕ) := C + 6
def mina_digits := 24
def nina_digits (C : ℕ) := (7 * C) / 2

-- Define Carlos's digits and the sum condition
def carlos_digits := mina_digits / 6
def total_digits (C : ℕ) := C + sam_digits C + mina_digits + nina_digits C

-- Prove the maximum number of digits Nina could memorize
theorem nina_max_digits : ∀ C : ℕ, C = carlos_digits →
  total_digits C ≤ 100 → nina_digits C ≤ 62 :=
by
  intro C hC htotal
  sorry

end nina_max_digits_l1459_145916


namespace cal_fraction_of_anthony_l1459_145982

theorem cal_fraction_of_anthony (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ)
  (h_mabel : mabel_transactions = 90)
  (h_anthony : anthony_transactions = mabel_transactions + mabel_transactions / 10)
  (h_jade : jade_transactions = 82)
  (h_jade_cal : jade_transactions = cal_transactions + 16) :
  (cal_transactions : ℚ) / (anthony_transactions : ℚ) = 2 / 3 :=
by
  -- The proof would be here, but it is omitted as per the requirement.
  sorry

end cal_fraction_of_anthony_l1459_145982


namespace face_value_of_shares_l1459_145905

-- Define the problem conditions
variables (F : ℝ) (D R : ℝ)

-- Assume conditions
axiom h1 : D = 0.155 * F
axiom h2 : R = 0.25 * 31
axiom h3 : D = R

-- State the theorem
theorem face_value_of_shares : F = 50 :=
by 
  -- Here should be the proof which we are skipping
  sorry

end face_value_of_shares_l1459_145905


namespace symmetric_curve_eq_l1459_145941

-- Definitions from the problem conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 1
def line_of_symmetry (x y : ℝ) : Prop := x - y + 3 = 0

-- Problem statement derived from the translation step
theorem symmetric_curve_eq (x y : ℝ) : (x - 2) ^ 2 + (y + 1) ^ 2 = 1 ∧ x - y + 3 = 0 → (x + 4) ^ 2 + (y - 5) ^ 2 = 1 := 
by
  sorry

end symmetric_curve_eq_l1459_145941


namespace value_of_a_minus_b_l1459_145955

variable {R : Type} [Field R]

noncomputable def f (a b x : R) : R := a * x + b
noncomputable def g (x : R) : R := -2 * x + 7
noncomputable def h (a b x : R) : R := f a b (g x)

theorem value_of_a_minus_b (a b : R) (h_inv : R → R) 
  (h_def : ∀ x, h_inv x = x + 9)
  (h_eq : ∀ x, h a b x = x - 9) : 
  a - b = 5 := by
  sorry

end value_of_a_minus_b_l1459_145955


namespace no_3_digit_number_with_digit_sum_27_and_even_l1459_145993

-- Define what it means for a number to be 3-digit
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the digit-sum function
def digitSum (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

-- Define what it means for a number to be even
def isEven (n : ℕ) : Prop := n % 2 = 0

-- State the proof problem
theorem no_3_digit_number_with_digit_sum_27_and_even :
  ∀ n : ℕ, isThreeDigit n → digitSum n = 27 → isEven n → false :=
by
  -- Proof should go here
  sorry

end no_3_digit_number_with_digit_sum_27_and_even_l1459_145993


namespace blocks_for_tower_l1459_145943

theorem blocks_for_tower (total_blocks : ℕ) (house_blocks : ℕ) (extra_blocks : ℕ) (tower_blocks : ℕ) 
  (h1 : total_blocks = 95) 
  (h2 : house_blocks = 20) 
  (h3 : extra_blocks = 30) 
  (h4 : tower_blocks = house_blocks + extra_blocks) : 
  tower_blocks = 50 :=
sorry

end blocks_for_tower_l1459_145943


namespace trig_expression_value_l1459_145953

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 1/2) :
  (1 + 2 * Real.sin (π - α) * Real.cos (-2 * π - α)) / 
  (Real.sin (-α) ^ 2 - Real.sin (5 * π / 2 - α) ^ 2) = -3 :=
by
  sorry

end trig_expression_value_l1459_145953


namespace simplify_and_evaluate_expression_l1459_145964

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate_expression :
  ((x + 1) / (x^2 + 2 * x + 1)) / (1 - (2 / (x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l1459_145964


namespace max_length_CD_l1459_145938

open Real

/-- Given a circle with center O and diameter AB = 20 units,
    with points C and D positioned such that C is 6 units away from A
    and D is 7 units away from B on the diameter AB,
    prove that the maximum length of the direct path from C to D is 7 units.
-/
theorem max_length_CD {A B C D : ℝ} 
    (diameter : dist A B = 20) 
    (C_pos : dist A C = 6) 
    (D_pos : dist B D = 7) : 
    dist C D = 7 :=
by
  -- Details of the proof would go here
  sorry

end max_length_CD_l1459_145938


namespace new_paint_intensity_l1459_145948

-- Definition of the given conditions
def original_paint_intensity : ℝ := 0.15
def replacement_paint_intensity : ℝ := 0.25
def fraction_replaced : ℝ := 1.5
def original_volume : ℝ := 100

-- Proof statement
theorem new_paint_intensity :
  (original_volume * original_paint_intensity + original_volume * fraction_replaced * replacement_paint_intensity) /
  (original_volume + original_volume * fraction_replaced) = 0.21 :=
by
  sorry

end new_paint_intensity_l1459_145948


namespace distance_to_school_l1459_145945

theorem distance_to_school (d : ℝ) (h1 : d / 5 + d / 25 = 1) : d = 25 / 6 :=
by
  sorry

end distance_to_school_l1459_145945


namespace never_sunday_l1459_145975

theorem never_sunday (n : ℕ) (days_in_month : ℕ → ℕ) (is_leap_year : Bool) : 
  (∀ (month : ℕ), 1 ≤ month ∧ month ≤ 12 → (days_in_month month = 28 ∨ days_in_month month = 29 ∨ days_in_month month = 30 ∨ days_in_month month = 31) ∧
  (∃ (k : ℕ), k < 7 ∧ ∀ (d : ℕ), d < days_in_month month → (d % 7 = k ↔ n ≠ d))) → n = 31 := 
by
  sorry

end never_sunday_l1459_145975


namespace no_x_satisfies_arithmetic_mean_l1459_145924

theorem no_x_satisfies_arithmetic_mean :
  ¬ ∃ x : ℝ, (3 + 117 + 915 + 138 + 2114 + x) / 6 = 12 :=
by
  sorry

end no_x_satisfies_arithmetic_mean_l1459_145924


namespace car_clock_correctness_l1459_145906

variables {t_watch t_car : ℕ} 
--  Variable declarations for time on watch (accurate) and time on car clock.

-- Define the initial times at 8:00 AM
def initial_time_watch : ℕ := 8 * 60 -- 8:00 AM in minutes
def initial_time_car : ℕ := 8 * 60 -- also 8:00 AM in minutes

-- Define the known times in the afternoon
def afternoon_time_watch : ℕ := 14 * 60 -- 2:00 PM in minutes
def afternoon_time_car : ℕ := 14 * 60 + 10 -- 2:10 PM in minutes

-- Car clock runs 37 minutes in the time the watch runs 36 minutes
def car_clock_rate : ℕ × ℕ := (37, 36)

-- Check the car clock time when the accurate watch shows 10:00 PM
def car_time_at_10pm_watch : ℕ := 22 * 60 -- 10:00 PM in minutes

-- Define the actual time that we need to prove
def actual_time_at_10pm_car : ℕ := 21 * 60 + 47 -- 9:47 PM in minutes

theorem car_clock_correctness : 
  (t_watch = actual_time_at_10pm_car) ↔ 
  (t_car = car_time_at_10pm_watch) ∧ 
  (initial_time_watch = initial_time_car) ∧ 
  (afternoon_time_watch = 14 * 60) ∧ 
  (afternoon_time_car = 14 * 60 + 10) ∧ 
  (car_clock_rate = (37, 36)) :=
sorry

end car_clock_correctness_l1459_145906


namespace sum_of_two_numbers_l1459_145973

theorem sum_of_two_numbers (S L : ℝ) (h1 : S = 10.0) (h2 : 7 * S = 5 * L) : S + L = 24.0 :=
by
  -- proof goes here
  sorry

end sum_of_two_numbers_l1459_145973


namespace parallel_line_through_A_is_2x_3y_minus_15_line_with_twice_slope_angle_l1459_145988

open Real

-- Conditions:
def l1 (x y : ℝ) : Prop := x - 2 * y + 3 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 9 = 0
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

-- Point A is the intersection of l1 and l2
def A : ℝ × ℝ := ⟨3, 3⟩

-- Question 1
def line_parallel (x y : ℝ) (c : ℝ) : Prop := 2 * x + 3 * y + c = 0
def line_parallel_passing_through_A : Prop := line_parallel A.1 A.2 (-15)

theorem parallel_line_through_A_is_2x_3y_minus_15 : line_parallel_passing_through_A :=
sorry

-- Question 2
def slope_angle (tan_alpha : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ y, ∃ x, y ≠ 0 ∧ l x y ∧ (tan_alpha = x / y)

def required_slope (tan_alpha : ℝ) : Prop :=
  tan_alpha = 4 / 3

def line_with_slope (x y slope : ℝ) : Prop :=
  y - A.2 = slope * (x - A.1)

def line_with_required_slope : Prop := 
  line_with_slope A.1 A.2 (4 / 3)

theorem line_with_twice_slope_angle : line_with_required_slope :=
sorry

end parallel_line_through_A_is_2x_3y_minus_15_line_with_twice_slope_angle_l1459_145988


namespace equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l1459_145925
open BigOperators

-- First, we define the three equations and their constraints
def equation1_solution (k : ℤ) : ℤ × ℤ := (2 - 5 * k, -1 + 3 * k)
def equation2_solution (k : ℤ) : ℤ × ℤ := (8 - 5 * k, -4 + 3 * k)
def equation3_solution (k : ℤ) : ℤ × ℤ := (16 - 39 * k, -25 + 61 * k)

-- Define the proof that the supposed solutions hold for each equation
theorem equation1_solution_valid (k : ℤ) : 3 * (equation1_solution k).1 + 5 * (equation1_solution k).2 = 1 :=
by
  -- Proof steps would go here
  sorry

theorem equation2_solution_valid (k : ℤ) : 3 * (equation2_solution k).1 + 5 * (equation2_solution k).2 = 4 :=
by
  -- Proof steps would go here
  sorry

theorem equation3_solution_valid (k : ℤ) : 183 * (equation3_solution k).1 + 117 * (equation3_solution k).2 = 3 :=
by
  -- Proof steps would go here
  sorry

end equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l1459_145925


namespace range_of_k_l1459_145965

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ 0 ≤ k ∧ k < 4 := sorry

end range_of_k_l1459_145965


namespace measure_of_angle_E_l1459_145910

variable (D E F : ℝ)
variable (h1 : E = F)
variable (h2 : F = 3 * D)
variable (h3 : D + E + F = 180)

theorem measure_of_angle_E : E = 540 / 7 :=
by
  -- Proof omitted
  sorry

end measure_of_angle_E_l1459_145910


namespace find_f_2016_minus_f_2015_l1459_145912

-- Definitions for the given conditions

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) → f x = 2^x

-- Main theorem statement
theorem find_f_2016_minus_f_2015 {f : ℝ → ℝ} 
    (H1 : odd_function f) 
    (H2 : periodic_function f)
    (H3 : specific_values f)
    : f 2016 - f 2015 = 2 := 
sorry

end find_f_2016_minus_f_2015_l1459_145912
