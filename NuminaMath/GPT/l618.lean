import Mathlib

namespace NUMINAMATH_GPT_probability_ratio_l618_61809

theorem probability_ratio (bins balls n1 n2 n3 n4 : Nat)
  (h_balls : balls = 18)
  (h_bins : bins = 4)
  (scenarioA : n1 = 6 ∧ n2 = 2 ∧ n3 = 5 ∧ n4 = 5)
  (scenarioB : n1 = 5 ∧ n2 = 5 ∧ n3 = 4 ∧ n4 = 4) :
  ((Nat.choose bins 1) * (Nat.choose (bins - 1) 1) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) /
  ((Nat.choose bins 2) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_ratio_l618_61809


namespace NUMINAMATH_GPT_first_pair_weight_l618_61811

variable (total_weight : ℕ) (second_pair_weight : ℕ) (third_pair_weight : ℕ)

theorem first_pair_weight (h : total_weight = 32) (h_second : second_pair_weight = 5) (h_third : third_pair_weight = 8) : 
    total_weight - 2 * (second_pair_weight + third_pair_weight) = 6 :=
by
  sorry

end NUMINAMATH_GPT_first_pair_weight_l618_61811


namespace NUMINAMATH_GPT_vasya_numbers_l618_61896

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end NUMINAMATH_GPT_vasya_numbers_l618_61896


namespace NUMINAMATH_GPT_area_increase_by_40_percent_l618_61843

theorem area_increase_by_40_percent (s : ℝ) : 
  let A1 := s^2 
  let new_side := 1.40 * s 
  let A2 := new_side^2 
  (A2 - A1) / A1 * 100 = 96 := 
by 
  sorry

end NUMINAMATH_GPT_area_increase_by_40_percent_l618_61843


namespace NUMINAMATH_GPT_solve_for_x_l618_61879

theorem solve_for_x (x : ℝ) (h : 3 * x + 8 = -4 * x - 16) : x = -24 / 7 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l618_61879


namespace NUMINAMATH_GPT_optimal_garden_area_l618_61873

variable (l w : ℕ)

/-- Tiffany is building a fence around a rectangular garden. Determine the optimal area, 
    in square feet, that can be enclosed under the conditions. -/
theorem optimal_garden_area 
  (h1 : l >= 100)
  (h2 : w >= 50)
  (h3 : 2 * l + 2 * w = 400) : (l * w) ≤ 7500 := 
sorry

end NUMINAMATH_GPT_optimal_garden_area_l618_61873


namespace NUMINAMATH_GPT_solution_is_unique_zero_l618_61830

theorem solution_is_unique_zero : ∀ (x y z : ℤ), x^3 + 2 * y^3 = 4 * z^3 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros x y z h
  sorry

end NUMINAMATH_GPT_solution_is_unique_zero_l618_61830


namespace NUMINAMATH_GPT_alice_minimum_speed_l618_61822

-- Conditions
def distance : ℝ := 60 -- The distance from City A to City B in miles
def bob_speed : ℝ := 40 -- Bob's constant speed in miles per hour
def alice_delay : ℝ := 0.5 -- Alice's delay in hours before she starts

-- Question as a proof statement
theorem alice_minimum_speed : ∀ (alice_speed : ℝ), alice_speed > 60 → 
  (alice_speed * (1.5 - alice_delay) < distance) → true :=
by
  sorry

end NUMINAMATH_GPT_alice_minimum_speed_l618_61822


namespace NUMINAMATH_GPT_turkey_weight_l618_61881

theorem turkey_weight (total_time_minutes roast_time_per_pound number_of_turkeys : ℕ) 
  (h1 : total_time_minutes = 480) 
  (h2 : roast_time_per_pound = 15)
  (h3 : number_of_turkeys = 2) : 
  (total_time_minutes / number_of_turkeys) / roast_time_per_pound = 16 :=
by
  sorry

end NUMINAMATH_GPT_turkey_weight_l618_61881


namespace NUMINAMATH_GPT_largest_obtuse_prime_angle_l618_61839

theorem largest_obtuse_prime_angle (alpha beta gamma : ℕ) 
    (h_triangle_sum : alpha + beta + gamma = 180) 
    (h_alpha_gt_beta : alpha > beta) 
    (h_beta_gt_gamma : beta > gamma)
    (h_obtuse_alpha : alpha > 90) 
    (h_alpha_prime : Prime alpha) 
    (h_beta_prime : Prime beta) : 
    alpha = 173 := 
sorry

end NUMINAMATH_GPT_largest_obtuse_prime_angle_l618_61839


namespace NUMINAMATH_GPT_find_a2_l618_61890

-- Definitions from conditions
def is_arithmetic_sequence (u : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, u (n + 1) = u n + d

def is_geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

-- Main theorem statement
theorem find_a2
  (u : ℕ → ℤ) (a1 a3 a4 : ℤ)
  (h1 : is_arithmetic_sequence u 3)
  (h2 : is_geometric_sequence a1 a3 a4)
  (h3 : a1 = u 1)
  (h4 : a3 = u 3)
  (h5 : a4 = u 4) :
  u 2 = -9 :=
by  
  sorry

end NUMINAMATH_GPT_find_a2_l618_61890


namespace NUMINAMATH_GPT_smallest_num_rectangles_to_cover_square_l618_61804

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end NUMINAMATH_GPT_smallest_num_rectangles_to_cover_square_l618_61804


namespace NUMINAMATH_GPT_distribute_items_l618_61886

open Nat

def g (n k : ℕ) : ℕ :=
  -- This is a placeholder for the actual function definition
  sorry

theorem distribute_items (n k : ℕ) (h : n ≥ k ∧ k ≥ 2) :
  g (n + 1) k = k * g n (k - 1) + k * g n k :=
by
  sorry

end NUMINAMATH_GPT_distribute_items_l618_61886


namespace NUMINAMATH_GPT_neg_square_result_l618_61882

-- This definition captures the algebraic expression and its computation rule.
theorem neg_square_result (a : ℝ) : -((-3 * a) ^ 2) = -9 * (a ^ 2) := 
by
  sorry

end NUMINAMATH_GPT_neg_square_result_l618_61882


namespace NUMINAMATH_GPT_solve_for_x_l618_61888

theorem solve_for_x (x : ℝ) (h1 : 8 * x^2 + 8 * x - 2 = 0) (h2 : 32 * x^2 + 68 * x - 8 = 0) : 
    x = 1 / 8 := 
    sorry

end NUMINAMATH_GPT_solve_for_x_l618_61888


namespace NUMINAMATH_GPT_max_min_value_of_a_l618_61864

theorem max_min_value_of_a 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end NUMINAMATH_GPT_max_min_value_of_a_l618_61864


namespace NUMINAMATH_GPT_gcd_84_108_132_156_l618_61857

theorem gcd_84_108_132_156 : Nat.gcd (Nat.gcd 84 108) (Nat.gcd 132 156) = 12 := 
by
  sorry

end NUMINAMATH_GPT_gcd_84_108_132_156_l618_61857


namespace NUMINAMATH_GPT_problem_statement_l618_61865

-- Definition of the function f with the given condition
def satisfies_condition (f : ℝ → ℝ) := ∀ (α β : ℝ), f (α + β) - (f α + f β) = 2008

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) := ∀ (x : ℝ), f (-x) = -f x

-- Main statement to prove in Lean
theorem problem_statement (f : ℝ → ℝ) (h : satisfies_condition f) : is_odd (fun x => f x + 2008) :=
sorry

end NUMINAMATH_GPT_problem_statement_l618_61865


namespace NUMINAMATH_GPT_regular_polygon_interior_angle_integer_l618_61808

theorem regular_polygon_interior_angle_integer :
  ∃ l : List ℕ, l.length = 9 ∧ ∀ n ∈ l, 3 ≤ n ∧ n ≤ 15 ∧ (180 * (n - 2)) % n = 0 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_interior_angle_integer_l618_61808


namespace NUMINAMATH_GPT_purple_candy_minimum_cost_l618_61861

theorem purple_candy_minimum_cost (r g b n : ℕ) (h : 10 * r = 15 * g) (h1 : 15 * g = 18 * b) (h2 : 18 * b = 24 * n) : 
  ∃ k, k = n ∧ k ≥ 1 ∧ ∀ m, (24 * m = 360) → (m ≥ k) :=
by
  sorry

end NUMINAMATH_GPT_purple_candy_minimum_cost_l618_61861


namespace NUMINAMATH_GPT_right_angle_triangle_sets_l618_61802

def is_right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angle_triangle_sets :
  ¬ is_right_angle_triangle (2 / 3) 2 (5 / 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_right_angle_triangle_sets_l618_61802


namespace NUMINAMATH_GPT_isabella_paint_area_l618_61895

theorem isabella_paint_area 
    (bedrooms : ℕ) 
    (length width height doorway_window_area : ℕ) 
    (h1 : bedrooms = 4) 
    (h2 : length = 14) 
    (h3 : width = 12) 
    (h4 : height = 9)
    (h5 : doorway_window_area = 80) :
    (2 * (length * height) + 2 * (width * height) - doorway_window_area) * bedrooms = 1552 := by
       -- Calculate the area of the walls in one bedroom
       -- 2 * (length * height) + 2 * (width * height) - doorway_window_area = 388
       -- The total paintable area for 4 bedrooms = 388 * 4 = 1552
       sorry

end NUMINAMATH_GPT_isabella_paint_area_l618_61895


namespace NUMINAMATH_GPT_megan_savings_days_l618_61851

theorem megan_savings_days :
  let josiah_saving_rate : ℝ := 0.25
  let josiah_days : ℕ := 24
  let josiah_total := josiah_saving_rate * josiah_days

  let leah_saving_rate : ℝ := 0.5
  let leah_days : ℕ := 20
  let leah_total := leah_saving_rate * leah_days

  let total_savings : ℝ := 28.0
  let josiah_leah_total := josiah_total + leah_total
  let megan_total := total_savings - josiah_leah_total

  let megan_saving_rate := 2 * leah_saving_rate
  let megan_days := megan_total / megan_saving_rate
  
  megan_days = 12 :=
by
  sorry

end NUMINAMATH_GPT_megan_savings_days_l618_61851


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l618_61858

theorem abs_inequality_solution_set (x : ℝ) :
  |x - 1| + |x + 2| < 5 ↔ -3 < x ∧ x < 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_abs_inequality_solution_set_l618_61858


namespace NUMINAMATH_GPT_boys_to_girls_ratio_l618_61846

theorem boys_to_girls_ratio (S G B : ℕ) (h : (1/2 : ℚ) * G = (1/3 : ℚ) * S) :
  B / G = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_boys_to_girls_ratio_l618_61846


namespace NUMINAMATH_GPT_not_algebraic_expression_C_l618_61867

-- Define what it means for something to be an algebraic expression, as per given problem's conditions
def is_algebraic_expression (expr : String) : Prop :=
  expr = "A" ∨ expr = "B" ∨ expr = "D"
  
theorem not_algebraic_expression_C : ¬ (is_algebraic_expression "C") :=
by
  -- This is a placeholder; proof steps are not required per instructions
  sorry

end NUMINAMATH_GPT_not_algebraic_expression_C_l618_61867


namespace NUMINAMATH_GPT_sum_of_xy_l618_61855

theorem sum_of_xy (x y : ℝ) (h1 : x + 3 * y = 12) (h2 : 3 * x + y = 8) : x + y = 5 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_xy_l618_61855


namespace NUMINAMATH_GPT_magic_card_profit_l618_61860

theorem magic_card_profit (purchase_price : ℝ) (multiplier : ℝ) (selling_price : ℝ) (profit : ℝ) 
                          (h1 : purchase_price = 100) 
                          (h2 : multiplier = 3) 
                          (h3 : selling_price = purchase_price * multiplier) 
                          (h4 : profit = selling_price - purchase_price) : 
                          profit = 200 :=
by 
  -- Here, you can introduce intermediate steps if needed.
  sorry

end NUMINAMATH_GPT_magic_card_profit_l618_61860


namespace NUMINAMATH_GPT_gravitational_force_on_space_station_l618_61805

-- Define the problem conditions and gravitational relationship
def gravitational_force_proportionality (f d : ℝ) : Prop :=
  ∃ k : ℝ, f * d^2 = k

-- Given conditions
def earth_surface_distance : ℝ := 6371
def space_station_distance : ℝ := 100000
def surface_gravitational_force : ℝ := 980
def proportionality_constant : ℝ := surface_gravitational_force * earth_surface_distance^2

-- Statement of the proof problem
theorem gravitational_force_on_space_station :
  gravitational_force_proportionality surface_gravitational_force earth_surface_distance →
  ∃ f2 : ℝ, f2 = 3.977 ∧ gravitational_force_proportionality f2 space_station_distance :=
sorry

end NUMINAMATH_GPT_gravitational_force_on_space_station_l618_61805


namespace NUMINAMATH_GPT_boys_of_other_communities_l618_61814

theorem boys_of_other_communities (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℝ) 
  (h_tm : total_boys = 1500)
  (h_pm : percentage_muslims = 37.5)
  (h_ph : percentage_hindus = 25.6)
  (h_ps : percentage_sikhs = 8.4) : 
  ∃ (boys_other_communities : ℕ), boys_other_communities = 428 :=
by
  sorry

end NUMINAMATH_GPT_boys_of_other_communities_l618_61814


namespace NUMINAMATH_GPT_complete_square_l618_61820

theorem complete_square (x m : ℝ) : x^2 + 2 * x - 2 = 0 → (x + m)^2 = 3 → m = 1 := sorry

end NUMINAMATH_GPT_complete_square_l618_61820


namespace NUMINAMATH_GPT_extreme_values_number_of_zeros_l618_61891

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5
noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem extreme_values :
  (∀ x : ℝ, f x ≤ 12) ∧ (f (-1) = 12) ∧ (∀ x : ℝ, -15 ≤ f x) ∧ (f 2 = -15) := 
sorry

theorem number_of_zeros (m : ℝ) :
  (m > 12 ∨ m < -15 → ∃! x : ℝ, g x m = 0) ∧
  (m = 12 ∨ m = -15 → ∃ x y : ℝ, x ≠ y ∧ g x m = 0 ∧ g y m = 0) ∧
  (-15 < m ∧ m < 12 → ∃ x y z : ℝ, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ g x m = 0 ∧ g y m = 0 ∧ g z m = 0) :=
sorry

end NUMINAMATH_GPT_extreme_values_number_of_zeros_l618_61891


namespace NUMINAMATH_GPT_min_value_expression_l618_61827

variable (p q r : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)

theorem min_value_expression :
  (9 * r / (3 * p + 2 * q) + 9 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l618_61827


namespace NUMINAMATH_GPT_batches_of_engines_l618_61870

variable (total_engines : ℕ) (not_defective_engines : ℕ := 300) (engines_per_batch : ℕ := 80)

theorem batches_of_engines (h1 : 3 * total_engines / 4 = not_defective_engines) :
  total_engines / engines_per_batch = 5 := by
sorry

end NUMINAMATH_GPT_batches_of_engines_l618_61870


namespace NUMINAMATH_GPT_total_reptiles_l618_61842

theorem total_reptiles 
  (reptiles_in_s1 : ℕ := 523)
  (reptiles_in_s2 : ℕ := 689)
  (reptiles_in_s3 : ℕ := 784)
  (reptiles_in_s4 : ℕ := 392)
  (reptiles_in_s5 : ℕ := 563)
  (reptiles_in_s6 : ℕ := 842) :
  reptiles_in_s1 + reptiles_in_s2 + reptiles_in_s3 + reptiles_in_s4 + reptiles_in_s5 + reptiles_in_s6 = 3793 :=
by
  sorry

end NUMINAMATH_GPT_total_reptiles_l618_61842


namespace NUMINAMATH_GPT_mushroom_ratio_l618_61813

theorem mushroom_ratio (total_mushrooms safe_mushrooms uncertain_mushrooms : ℕ)
  (h_total : total_mushrooms = 32)
  (h_safe : safe_mushrooms = 9)
  (h_uncertain : uncertain_mushrooms = 5) :
  (total_mushrooms - safe_mushrooms - uncertain_mushrooms) / safe_mushrooms = 2 :=
by sorry

end NUMINAMATH_GPT_mushroom_ratio_l618_61813


namespace NUMINAMATH_GPT_melanie_turnips_l618_61876

theorem melanie_turnips (benny_turnips total_turnips melanie_turnips : ℕ) 
  (h1 : benny_turnips = 113) 
  (h2 : total_turnips = 252) 
  (h3 : total_turnips = benny_turnips + melanie_turnips) : 
  melanie_turnips = 139 :=
by
  sorry

end NUMINAMATH_GPT_melanie_turnips_l618_61876


namespace NUMINAMATH_GPT_find_B_in_product_l618_61859

theorem find_B_in_product (B : ℕ) (hB : B < 10) (h : (B * 100 + 2) * (900 + B) = 8016) : B = 8 := by
  sorry

end NUMINAMATH_GPT_find_B_in_product_l618_61859


namespace NUMINAMATH_GPT_rebecca_pies_l618_61807

theorem rebecca_pies 
  (P : ℕ) 
  (slices_per_pie : ℕ := 8) 
  (rebecca_slices : ℕ := P) 
  (family_and_friends_slices : ℕ := (7 * P) / 2) 
  (additional_slices : ℕ := 2) 
  (remaining_slices : ℕ := 5) 
  (total_slices : ℕ := slices_per_pie * P) :
  rebecca_slices + family_and_friends_slices + additional_slices + remaining_slices = total_slices → 
  P = 2 := 
by { sorry }

end NUMINAMATH_GPT_rebecca_pies_l618_61807


namespace NUMINAMATH_GPT_domain_of_function_l618_61825

theorem domain_of_function :
  {x : ℝ | (x^2 - 9*x + 20 ≥ 0) ∧ (|x - 5| + |x + 2| ≠ 0)} = {x : ℝ | x ≤ 4 ∨ x ≥ 5} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l618_61825


namespace NUMINAMATH_GPT_total_tiles_l618_61887

theorem total_tiles (s : ℕ) (h_black_tiles : 2 * s - 1 = 75) : s^2 = 1444 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_tiles_l618_61887


namespace NUMINAMATH_GPT_ryan_recruit_people_l618_61840

noncomputable def total_amount_needed : ℕ := 1000
noncomputable def amount_already_have : ℕ := 200
noncomputable def average_funding_per_person : ℕ := 10
noncomputable def additional_funding_needed : ℕ := total_amount_needed - amount_already_have
noncomputable def number_of_people_recruit : ℕ := additional_funding_needed / average_funding_per_person

theorem ryan_recruit_people : number_of_people_recruit = 80 := by
  sorry

end NUMINAMATH_GPT_ryan_recruit_people_l618_61840


namespace NUMINAMATH_GPT_unique_positive_integer_solution_l618_61892

theorem unique_positive_integer_solution :
  ∃! n : ℕ, n > 0 ∧ ∃ k : ℕ, n^4 - n^3 + 3*n^2 + 5 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_integer_solution_l618_61892


namespace NUMINAMATH_GPT_determine_b_l618_61894

theorem determine_b (b : ℝ) : (∀ x : ℝ, (-x^2 + b * x + 1 < 0) ↔ (x < 2 ∨ x > 6)) → b = 8 :=
by sorry

end NUMINAMATH_GPT_determine_b_l618_61894


namespace NUMINAMATH_GPT_least_prime_value_l618_61880

/-- Let q be a set of 12 distinct prime numbers. If the sum of the integers in q is odd,
the product of all the integers in q is divisible by a perfect square, and the number x is a member of q,
then the least value that x can be is 2. -/
theorem least_prime_value (q : Finset ℕ) (hq_distinct : q.card = 12) (hq_prime : ∀ p ∈ q, Nat.Prime p) 
    (hq_odd_sum : q.sum id % 2 = 1) (hq_perfect_square_div : ∃ k, q.prod id % (k * k) = 0) (x : ℕ)
    (hx : x ∈ q) : x = 2 :=
sorry

end NUMINAMATH_GPT_least_prime_value_l618_61880


namespace NUMINAMATH_GPT_equation_of_circle_l618_61877

def center : ℝ × ℝ := (3, -2)
def radius : ℝ := 5

theorem equation_of_circle (x y : ℝ) :
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔
  (x - 3)^2 + (y + 2)^2 = 25 :=
by
  simp [center, radius]
  sorry

end NUMINAMATH_GPT_equation_of_circle_l618_61877


namespace NUMINAMATH_GPT_nested_sqrt_expr_l618_61812

theorem nested_sqrt_expr (M : ℝ) (h : M > 1) : (↑(M) ^ (1 / 4) ^ (1 / 4) ^ (1 / 4)) = M ^ (21 / 64) :=
by
  sorry

end NUMINAMATH_GPT_nested_sqrt_expr_l618_61812


namespace NUMINAMATH_GPT_average_price_of_dvds_l618_61899

theorem average_price_of_dvds :
  let num_dvds_box1 := 10
  let price_per_dvd_box1 := 2.00
  let num_dvds_box2 := 5
  let price_per_dvd_box2 := 5.00
  let total_cost_box1 := num_dvds_box1 * price_per_dvd_box1
  let total_cost_box2 := num_dvds_box2 * price_per_dvd_box2
  let total_dvds := num_dvds_box1 + num_dvds_box2
  let total_cost := total_cost_box1 + total_cost_box2
  (total_cost / total_dvds) = 3.00 := 
sorry

end NUMINAMATH_GPT_average_price_of_dvds_l618_61899


namespace NUMINAMATH_GPT_find_x_in_terms_of_y_l618_61897

theorem find_x_in_terms_of_y 
(h₁ : x ≠ 0) 
(h₂ : x ≠ 3) 
(h₃ : y ≠ 0) 
(h₄ : y ≠ 5) 
(h_eq : 3 / x + 2 / y = 1 / 3) : 
x = 9 * y / (y - 6) :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_terms_of_y_l618_61897


namespace NUMINAMATH_GPT_problem1_problem2_l618_61826

open Real

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

-- Conditions:
axiom condition1 : sin (α + π / 6) = sqrt 10 / 10
axiom condition2 : cos (α + π / 6) = 3 * sqrt 10 / 10
axiom condition3 : tan (α + β) = 2 / 5

-- Prove:
theorem problem1 : sin (2 * α + π / 6) = (3 * sqrt 3 - 4) / 10 :=
by sorry

theorem problem2 : tan (2 * β - π / 3) = 17 / 144 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l618_61826


namespace NUMINAMATH_GPT_water_in_maria_jar_after_200_days_l618_61893

def arithmetic_series_sum (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem water_in_maria_jar_after_200_days :
  let initial_volume_maria : ℕ := 1000
  let days : ℕ := 200
  let odd_days : ℕ := days / 2
  let even_days : ℕ := days / 2
  let volume_odd_transfer : ℕ := arithmetic_series_sum 1 2 odd_days
  let volume_even_transfer : ℕ := arithmetic_series_sum 2 2 even_days
  let net_transfer : ℕ := volume_odd_transfer - volume_even_transfer
  let final_volume_maria := initial_volume_maria + net_transfer
  final_volume_maria = 900 :=
by
  sorry

end NUMINAMATH_GPT_water_in_maria_jar_after_200_days_l618_61893


namespace NUMINAMATH_GPT_mickey_horses_per_week_l618_61841

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end NUMINAMATH_GPT_mickey_horses_per_week_l618_61841


namespace NUMINAMATH_GPT_find_k_and_max_ck_largest_ck_for_k0_largest_ck_for_k2_l618_61878

theorem find_k_and_max_ck:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    ∃ (c_k : ℝ), c_k > 0 ∧ (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c_k * (x + y + z)^k) →
  (∀ (k : ℝ), 0 ≤ k ∧ k ≤ 2) :=
by
  sorry

theorem largest_ck_for_k0:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ 1) := 
by
  sorry

theorem largest_ck_for_k2:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ (8/9) * (x + y + z)^2) :=
by
  sorry

end NUMINAMATH_GPT_find_k_and_max_ck_largest_ck_for_k0_largest_ck_for_k2_l618_61878


namespace NUMINAMATH_GPT_min_value_fraction_l618_61854

theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2: b > 0) (h3 : a + b = 1) : 
  ∃ c : ℝ, c = 3 + 2 * Real.sqrt 2 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (x + y = 1) → x + 2 * y ≥ c) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l618_61854


namespace NUMINAMATH_GPT_trapezoid_height_l618_61800

-- We are given the lengths of the sides of the trapezoid
def length_parallel1 : ℝ := 25
def length_parallel2 : ℝ := 4
def length_non_parallel1 : ℝ := 20
def length_non_parallel2 : ℝ := 13

-- We need to prove that the height of the trapezoid is 12 cm
theorem trapezoid_height (h : ℝ) :
  (h^2 + (20^2 - 16^2) = 144 ∧ h = 12) :=
sorry

end NUMINAMATH_GPT_trapezoid_height_l618_61800


namespace NUMINAMATH_GPT_find_f79_l618_61806

noncomputable def f : ℝ → ℝ :=
  sorry

axiom condition1 : ∀ x y : ℝ, f (x * y) = x * f y
axiom condition2 : f 1 = 25

theorem find_f79 : f 79 = 1975 :=
by
  sorry

end NUMINAMATH_GPT_find_f79_l618_61806


namespace NUMINAMATH_GPT_factorize_expression_l618_61848

theorem factorize_expression (x : ℝ) : 
  (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 := 
by sorry

end NUMINAMATH_GPT_factorize_expression_l618_61848


namespace NUMINAMATH_GPT_congruence_solution_count_l618_61817

theorem congruence_solution_count :
  ∀ y : ℕ, y < 150 → (y ≡ 20 + 110 [MOD 46]) → y = 38 ∨ y = 84 ∨ y = 130 :=
by
  intro y
  intro hy
  intro hcong
  sorry

end NUMINAMATH_GPT_congruence_solution_count_l618_61817


namespace NUMINAMATH_GPT_solve_for_b_l618_61866

theorem solve_for_b (b : ℚ) (h : b + 2 * b / 5 = 22 / 5) : b = 22 / 7 :=
sorry

end NUMINAMATH_GPT_solve_for_b_l618_61866


namespace NUMINAMATH_GPT_find_k_l618_61875

open Complex

noncomputable def possible_values_of_k (a b c d e : ℂ) (k : ℂ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∧
  (a * k^4 + b * k^3 + c * k^2 + d * k + e = 0) ∧
  (b * k^4 + c * k^3 + d * k^2 + e * k + a = 0)

theorem find_k (a b c d e : ℂ) (k : ℂ) :
  possible_values_of_k a b c d e k → k^5 = 1 :=
by
  intro h
  sorry

#check find_k

end NUMINAMATH_GPT_find_k_l618_61875


namespace NUMINAMATH_GPT_baseball_league_games_l618_61828

theorem baseball_league_games
  (N M : ℕ)
  (hN_gt_2M : N > 2 * M)
  (hM_gt_4 : M > 4)
  (h_total_games : 4 * N + 5 * M = 94) :
  4 * N = 64 :=
by
  sorry

end NUMINAMATH_GPT_baseball_league_games_l618_61828


namespace NUMINAMATH_GPT_smallest_number_of_2_by_3_rectangles_l618_61845

def area_2_by_3_rectangle : Int := 2 * 3

def smallest_square_area_multiple_of_6 : Int :=
  let side_length := 6
  side_length * side_length

def number_of_rectangles_to_cover_square (square_area : Int) (rectangle_area : Int) : Int :=
  square_area / rectangle_area

theorem smallest_number_of_2_by_3_rectangles :
  number_of_rectangles_to_cover_square smallest_square_area_multiple_of_6 area_2_by_3_rectangle = 6 := by
  sorry

end NUMINAMATH_GPT_smallest_number_of_2_by_3_rectangles_l618_61845


namespace NUMINAMATH_GPT_non_zero_real_m_value_l618_61815

theorem non_zero_real_m_value (m : ℝ) (h1 : 3 - m ∈ ({1, 2, 3} : Set ℝ)) (h2 : m ≠ 0) : m = 2 := 
sorry

end NUMINAMATH_GPT_non_zero_real_m_value_l618_61815


namespace NUMINAMATH_GPT_irreducible_fraction_l618_61831

theorem irreducible_fraction (n : ℤ) : Int.gcd (3 * n + 10) (4 * n + 13) = 1 := 
sorry

end NUMINAMATH_GPT_irreducible_fraction_l618_61831


namespace NUMINAMATH_GPT_inequality_holds_l618_61850

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := sorry

end NUMINAMATH_GPT_inequality_holds_l618_61850


namespace NUMINAMATH_GPT_corey_candies_l618_61869

-- Definitions based on conditions
variable (T C : ℕ)
variable (totalCandies : T + C = 66)
variable (tapangaExtra : T = C + 8)

-- Theorem to prove Corey has 29 candies
theorem corey_candies : C = 29 :=
by
  sorry

end NUMINAMATH_GPT_corey_candies_l618_61869


namespace NUMINAMATH_GPT_bouncy_balls_total_l618_61883

theorem bouncy_balls_total :
  let red_packs := 6
  let red_per_pack := 12
  let yellow_packs := 10
  let yellow_per_pack := 8
  let green_packs := 4
  let green_per_pack := 15
  let blue_packs := 3
  let blue_per_pack := 20
  let red_balls := red_packs * red_per_pack
  let yellow_balls := yellow_packs * yellow_per_pack
  let green_balls := green_packs * green_per_pack
  let blue_balls := blue_packs * blue_per_pack
  red_balls + yellow_balls + green_balls + blue_balls = 272 := 
by
  sorry

end NUMINAMATH_GPT_bouncy_balls_total_l618_61883


namespace NUMINAMATH_GPT_profit_function_profit_for_240_barrels_barrels_for_760_profit_l618_61868

-- Define fixed costs, cost price per barrel, and selling price per barrel as constants
def fixed_costs : ℝ := 200
def cost_price_per_barrel : ℝ := 5
def selling_price_per_barrel : ℝ := 8

-- Definitions for daily sales quantity (x) and daily profit (y)
def daily_sales_quantity (x : ℝ) : ℝ := x
def daily_profit (x : ℝ) : ℝ := (selling_price_per_barrel * x) - (cost_price_per_barrel * x) - fixed_costs

-- Prove the functional relationship y = 3x - 200
theorem profit_function (x : ℝ) : daily_profit x = 3 * x - fixed_costs :=
by sorry

-- Given sales quantity is 240 barrels, prove profit is 520 yuan
theorem profit_for_240_barrels : daily_profit 240 = 520 :=
by sorry

-- Given profit is 760 yuan, prove sales quantity is 320 barrels
theorem barrels_for_760_profit : ∃ (x : ℝ), daily_profit x = 760 ∧ x = 320 :=
by sorry

end NUMINAMATH_GPT_profit_function_profit_for_240_barrels_barrels_for_760_profit_l618_61868


namespace NUMINAMATH_GPT_solve_inequality_l618_61823

theorem solve_inequality (y : ℚ) :
  (3 / 40 : ℚ) + |y - (17 / 80 : ℚ)| < (1 / 8 : ℚ) ↔ (13 / 80 : ℚ) < y ∧ y < (21 / 80 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l618_61823


namespace NUMINAMATH_GPT_problem_solution_l618_61853

def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }
def complement_N : Set ℝ := { x | x ≤ 0 ∨ x ≥ 1 }

theorem problem_solution : M ∪ complement_N = Set.univ := 
sorry

end NUMINAMATH_GPT_problem_solution_l618_61853


namespace NUMINAMATH_GPT_max_value_of_g_l618_61889

noncomputable def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g : ∃ x ∈ Set.Icc (0:ℝ) 2, g x = 25 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_max_value_of_g_l618_61889


namespace NUMINAMATH_GPT_cubed_inequality_l618_61874

variable {a b : ℝ}

theorem cubed_inequality (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cubed_inequality_l618_61874


namespace NUMINAMATH_GPT_last_digit_of_product_l618_61838

theorem last_digit_of_product :
    (3 ^ 65 * 6 ^ 59 * 7 ^ 71) % 10 = 4 := 
  by sorry

end NUMINAMATH_GPT_last_digit_of_product_l618_61838


namespace NUMINAMATH_GPT_geometric_sequence_sum_10_l618_61829

theorem geometric_sequence_sum_10 (a : ℕ) (r : ℕ) (h : r = 2) (sum5 : a + r * a + r^2 * a + r^3 * a + r^4 * a = 1) : 
    a * (1 - r^10) / (1 - r) = 33 := 
by 
    sorry

end NUMINAMATH_GPT_geometric_sequence_sum_10_l618_61829


namespace NUMINAMATH_GPT_correct_region_l618_61862

-- Define the condition for x > 1
def condition_x_gt_1 (x : ℝ) (y : ℝ) : Prop :=
  x > 1 → y^2 > x

-- Define the condition for 0 < x < 1
def condition_0_lt_x_lt_1 (x : ℝ) (y : ℝ) : Prop :=
  0 < x ∧ x < 1 → 0 < y^2 ∧ y^2 < x

-- Formal statement to check the correct region
theorem correct_region (x y : ℝ) : 
  (condition_x_gt_1 x y ∨ condition_0_lt_x_lt_1 x y) →
  y^2 > x ∨ (0 < y^2 ∧ y^2 < x) :=
sorry

end NUMINAMATH_GPT_correct_region_l618_61862


namespace NUMINAMATH_GPT_total_items_to_buy_l618_61834

theorem total_items_to_buy (total_money : ℝ) (cost_sandwich : ℝ) (cost_drink : ℝ) (num_items : ℕ) :
  total_money = 30 → cost_sandwich = 4.5 → cost_drink = 1 → num_items = 9 :=
by
  sorry

end NUMINAMATH_GPT_total_items_to_buy_l618_61834


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l618_61898

theorem necessary_but_not_sufficient_condition :
  (∀ m, -1 < m ∧ m < 5 → ∀ x, 
    x^2 - 2 * m * x + m^2 - 1 = 0 → -2 < x ∧ x < 4) ∧ 
  ¬ (∀ m, -1 < m ∧ m < 5 → ∀ x, 
    x^2 - 2 * m * x + m^2 - 1 = 0 → -2 < x ∧ x < 4) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l618_61898


namespace NUMINAMATH_GPT_xyz_value_l618_61818

theorem xyz_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
    (hx : x * (y + z) = 162)
    (hy : y * (z + x) = 180)
    (hz : z * (x + y) = 198)
    (h_sum : x + y + z = 26) :
    x * y * z = 2294.67 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l618_61818


namespace NUMINAMATH_GPT_largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l618_61824

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end NUMINAMATH_GPT_largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l618_61824


namespace NUMINAMATH_GPT_find_2x_plus_y_l618_61833

theorem find_2x_plus_y (x y : ℝ) 
  (h1 : (x + y) / 3 = 5 / 3) 
  (h2 : x + 2*y = 8) : 
  2*x + y = 7 :=
sorry

end NUMINAMATH_GPT_find_2x_plus_y_l618_61833


namespace NUMINAMATH_GPT_expenditure_on_concrete_blocks_l618_61885

def blocks_per_section : ℕ := 30
def cost_per_block : ℕ := 2
def number_of_sections : ℕ := 8

theorem expenditure_on_concrete_blocks : 
  (number_of_sections * blocks_per_section) * cost_per_block = 480 := 
by 
  sorry

end NUMINAMATH_GPT_expenditure_on_concrete_blocks_l618_61885


namespace NUMINAMATH_GPT_arithmetic_sequence_angle_l618_61810

-- Define the conditions
variables (A B C a b c : ℝ)
-- The statement assumes that A, B, C form an arithmetic sequence
-- which implies 2B = A + C
-- We need to show that 1/(a + b) + 1/(b + c) = 3/(a + b + c)

theorem arithmetic_sequence_angle
  (h : 2 * B = A + C)
  (cos_rule : b^2 = c^2 + a^2 - 2 * c * a * Real.cos B):
    1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) := sorry

end NUMINAMATH_GPT_arithmetic_sequence_angle_l618_61810


namespace NUMINAMATH_GPT_factor_1000000000001_l618_61832

theorem factor_1000000000001 : ∃ a b c : ℕ, 1000000000001 = a * b * c ∧ a = 73 ∧ b = 137 ∧ c = 99990001 :=
by {
  sorry
}

end NUMINAMATH_GPT_factor_1000000000001_l618_61832


namespace NUMINAMATH_GPT_factor_expression_l618_61803

-- Define the expressions E1 and E2
def E1 (y : ℝ) : ℝ := 12 * y^6 + 35 * y^4 - 5
def E2 (y : ℝ) : ℝ := 2 * y^6 - 4 * y^4 + 5

-- Define the target expression E
def E (y : ℝ) : ℝ := E1 y - E2 y

-- The main theorem to prove
theorem factor_expression (y : ℝ) : E y = 10 * (y^6 + 3.9 * y^4 - 1) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l618_61803


namespace NUMINAMATH_GPT_integer_triplets_prime_l618_61816

theorem integer_triplets_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ sol : ℕ, ((∃ (x y z : ℤ), (3 * x + y + z) * (x + 2 * y + z) * (x + y + z) = p) ∧
  if p = 2 then sol = 4 else sol = 12) :=
by
  sorry

end NUMINAMATH_GPT_integer_triplets_prime_l618_61816


namespace NUMINAMATH_GPT_game_score_correct_answers_l618_61844

theorem game_score_correct_answers :
  ∃ x : ℕ, (∃ y : ℕ, x + y = 30 ∧ 7 * x - 12 * y = 77) ∧ x = 23 :=
by
  use 23
  sorry

end NUMINAMATH_GPT_game_score_correct_answers_l618_61844


namespace NUMINAMATH_GPT_find_x_plus_2y_sq_l618_61847

theorem find_x_plus_2y_sq (x y : ℝ) 
  (h : 8 * y^4 + 4 * x^2 * y^2 + 4 * x * y^2 + 2 * x^3 + 2 * y^2 + 2 * x = x^2 + 1) : 
  x + 2 * y^2 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_x_plus_2y_sq_l618_61847


namespace NUMINAMATH_GPT_rectangle_side_excess_percentage_l618_61849

theorem rectangle_side_excess_percentage (A B : ℝ) (x : ℝ) (h : A * (1 + x) * B * (1 - 0.04) = A * B * 1.008) : x = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_side_excess_percentage_l618_61849


namespace NUMINAMATH_GPT_min_x_y_l618_61821

theorem min_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end NUMINAMATH_GPT_min_x_y_l618_61821


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_m_l618_61863
open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := - abs (x + 4) + m

-- Part I: Solution set for f(x) > x + 1 is (-∞, 0)
theorem part1_solution_set : { x : ℝ | f x > x + 1 } = { x : ℝ | x < 0 } :=
sorry

-- Part II: Range of m when the graphs of y = f(x) and y = g(x) have common points
theorem part2_range_m (m : ℝ) : (∃ x : ℝ, f x = g x m) → m ≥ 5 :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_m_l618_61863


namespace NUMINAMATH_GPT_interval_contains_zeros_l618_61837

-- Define the conditions and the function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c 

theorem interval_contains_zeros (a b c : ℝ) (h1 : 2 * a + c / 2 > b) (h2 : c < 0) : 
  ∃ x ∈ Set.Ioc (-2 : ℝ) 0, quadratic a b c x = 0 :=
by
  -- Problem Statement: given conditions, interval (-2, 0) contains a zero
  sorry

end NUMINAMATH_GPT_interval_contains_zeros_l618_61837


namespace NUMINAMATH_GPT_factor_difference_of_squares_l618_61872

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l618_61872


namespace NUMINAMATH_GPT_max_value_g_l618_61884

-- Defining the conditions and goal as functions and properties
def condition_1 (f : ℕ → ℕ) : Prop :=
  (Finset.range 43).sum f ≤ 2022

def condition_2 (f g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a >= b → g (a + b) ≤ f a + f b

-- Defining the main theorem to establish the maximum value
theorem max_value_g (f g : ℕ → ℕ) (h1 : condition_1 f) (h2 : condition_2 f g) :
  (Finset.range 85).sum g ≤ 7615 :=
sorry


end NUMINAMATH_GPT_max_value_g_l618_61884


namespace NUMINAMATH_GPT_wendy_initial_flowers_l618_61835

theorem wendy_initial_flowers (wilted: ℕ) (bouquets_made: ℕ) (flowers_per_bouquet: ℕ) (flowers_initially_picked: ℕ):
  wilted = 35 →
  bouquets_made = 2 →
  flowers_per_bouquet = 5 →
  flowers_initially_picked = wilted + bouquets_made * flowers_per_bouquet →
  flowers_initially_picked = 45 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_wendy_initial_flowers_l618_61835


namespace NUMINAMATH_GPT_line_of_intersection_l618_61819

theorem line_of_intersection (x y z : ℝ) :
  (2 * x + 3 * y + 3 * z - 9 = 0) ∧ (4 * x + 2 * y + z - 8 = 0) →
  ((x / 4.5 + y / 3 + z / 3 = 1) ∧ (x / 2 + y / 4 + z / 8 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_line_of_intersection_l618_61819


namespace NUMINAMATH_GPT_sum_of_common_divisors_is_10_l618_61801

-- Define the list of numbers
def numbers : List ℤ := [42, 84, -14, 126, 210]

-- Define the common divisors
def common_divisors : List ℕ := [1, 2, 7]

-- Define the function that checks if a number is a common divisor of all numbers in the list
def is_common_divisor (d : ℕ) : Prop :=
  ∀ n ∈ numbers, (d : ℤ) ∣ n

-- Specify the sum of the common divisors
def sum_common_divisors : ℕ := common_divisors.sum

-- State the theorem to be proved
theorem sum_of_common_divisors_is_10 : 
  (∀ d ∈ common_divisors, is_common_divisor d) → 
  sum_common_divisors = 10 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_common_divisors_is_10_l618_61801


namespace NUMINAMATH_GPT_scalene_triangle_not_unique_by_two_non_opposite_angles_l618_61836

theorem scalene_triangle_not_unique_by_two_non_opposite_angles
  (α β : ℝ) (h1 : α > 0) (h2 : β > 0) (h3 : α + β < π) :
  ∃ (γ δ : ℝ), γ ≠ δ ∧ γ + α + β = δ + α + β :=
sorry

end NUMINAMATH_GPT_scalene_triangle_not_unique_by_two_non_opposite_angles_l618_61836


namespace NUMINAMATH_GPT_find_n_sequence_l618_61856

theorem find_n_sequence (n : ℕ) (b : ℕ → ℝ)
  (h0 : b 0 = 45) (h1 : b 1 = 80) (hn : b n = 0)
  (hrec : ∀ k, 1 ≤ k ∧ k ≤ n-1 → b (k+1) = b (k-1) - 4 / b k) :
  n = 901 :=
sorry

end NUMINAMATH_GPT_find_n_sequence_l618_61856


namespace NUMINAMATH_GPT_trapezium_second_side_length_l618_61871

theorem trapezium_second_side_length (a b h : ℕ) (Area : ℕ) 
  (h_area : Area = (1 / 2 : ℚ) * (a + b) * h)
  (ha : a = 20) (hh : h = 12) (hA : Area = 228) : b = 18 := by
  sorry

end NUMINAMATH_GPT_trapezium_second_side_length_l618_61871


namespace NUMINAMATH_GPT_parallelogram_area_l618_61852

theorem parallelogram_area (angle_bad : ℝ) (side_ab side_ad : ℝ) (h1 : angle_bad = 150) (h2 : side_ab = 20) (h3 : side_ad = 10) :
  side_ab * side_ad * Real.sin (angle_bad * Real.pi / 180) = 100 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l618_61852
