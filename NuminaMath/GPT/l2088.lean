import Mathlib

namespace probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l2088_208849

theorem probability_winning_on_first_draw : 
  let red := 1 
  let yellow := 3 
  red / (red + yellow) = 1 / 4 :=
by 
  sorry

theorem optimal_ball_to_add_for_fine_gift :
  let red := 1 
  let yellow := 3
  -- After adding a red ball: 2 red, 3 yellow
  let p1 := (2 * 1 + 3 * 2) / (2 + 3) / (1 + 3) = (2/5)
  -- After adding a yellow ball: 1 red, 4 yellow
  let p2 := (1 * 0 + 4 * 3) / (1 + 4) / (1 + 3) = (3/5)
  p1 < p2 :=
by 
  sorry

end probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l2088_208849


namespace path_area_and_cost_l2088_208834

-- Define the initial conditions
def field_length : ℝ := 65
def field_width : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sq_m : ℝ := 2

-- Define the extended dimensions including the path
def extended_length := field_length + 2 * path_width
def extended_width := field_width + 2 * path_width

-- Define the areas
def area_with_path := extended_length * extended_width
def area_of_field := field_length * field_width
def area_of_path := area_with_path - area_of_field

-- Define the cost
def cost_of_constructing_path := area_of_path * cost_per_sq_m

theorem path_area_and_cost :
  area_of_path = 625 ∧ cost_of_constructing_path = 1250 :=
by
  sorry

end path_area_and_cost_l2088_208834


namespace problem_l2088_208811

-- Conditions
variables (x y : ℚ)
def condition1 := 3 * x + 5 = 12
def condition2 := 10 * y - 2 = 5

-- Theorem to prove
theorem problem (h1 : condition1 x) (h2 : condition2 y) : x + y = 91 / 30 := sorry

end problem_l2088_208811


namespace third_place_prize_correct_l2088_208897

-- Define the conditions and formulate the problem
def total_amount_in_pot : ℝ := 210
def third_place_percentage : ℝ := 0.15
def third_place_prize (P : ℝ) : ℝ := third_place_percentage * P

-- The theorem to be proved
theorem third_place_prize_correct : 
  third_place_prize total_amount_in_pot = 31.5 := 
by
  sorry

end third_place_prize_correct_l2088_208897


namespace smallest_n_for_2n_3n_5n_conditions_l2088_208895

theorem smallest_n_for_2n_3n_5n_conditions : 
  ∃ n : ℕ, 
    (∀ k : ℕ, 2 * n ≠ k^2) ∧          -- 2n is a perfect square
    (∀ k : ℕ, 3 * n ≠ k^3) ∧          -- 3n is a perfect cube
    (∀ k : ℕ, 5 * n ≠ k^5) ∧          -- 5n is a perfect fifth power
    n = 11250 :=
sorry

end smallest_n_for_2n_3n_5n_conditions_l2088_208895


namespace find_x_l2088_208892

theorem find_x (x y : ℕ) 
  (h1 : 3^x * 4^y = 59049) 
  (h2 : x - y = 10) : 
  x = 10 := 
by 
  sorry

end find_x_l2088_208892


namespace simplify_fractions_l2088_208819

theorem simplify_fractions :
  (36 / 51) * (35 / 24) * (68 / 49) = (20 / 7) :=
by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 51 = 3 * 17 := by norm_num
  have h3 : 35 = 5 * 7 := by norm_num
  have h4 : 24 = 2^3 * 3 := by norm_num
  have h5 : 68 = 2^2 * 17 := by norm_num
  have h6 : 49 = 7^2 := by norm_num
  sorry

end simplify_fractions_l2088_208819


namespace proof_problem_l2088_208812

def diamondsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem proof_problem :
  { (x, y) : ℝ × ℝ | diamondsuit x y = diamondsuit y x } =
  { (x, y) | x = 0 } ∪ { (x, y) | y = 0 } ∪ { (x, y) | x = y } ∪ { (x, y) | x = -y } :=
by
  sorry

end proof_problem_l2088_208812


namespace set_difference_correct_l2088_208876

-- Define the sets A and B
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}

-- Define the set difference A - B
def A_minus_B : Set ℤ := {x | x ∈ A ∧ x ∉ B} -- This is the operation A - B

-- The theorem stating the required proof
theorem set_difference_correct : A_minus_B = {1, 3, 9} :=
by {
  -- Proof goes here; however, we have requested no proof, so we put sorry.
  sorry
}

end set_difference_correct_l2088_208876


namespace problem1_problem2_l2088_208860

-- Problem 1: y(x + y) + (x + y)(x - y) = x^2
theorem problem1 (x y : ℝ) : y * (x + y) + (x + y) * (x - y) = x^2 := 
by sorry

-- Problem 2: ( (2m + 1) / (m + 1) + m - 1 ) ÷ ( (m + 2) / (m^2 + 2m + 1) ) = m^2 + m
theorem problem2 (m : ℝ) (h1 : m ≠ -1) : 
  ( (2 * m + 1) / (m + 1) + m - 1 ) / ( (m + 2) / ((m + 1)^2) ) = m^2 + m := 
by sorry

end problem1_problem2_l2088_208860


namespace part1_solution_part2_solution_l2088_208879

-- Part (1) Statement
theorem part1_solution (x : ℝ) (m : ℝ) (h_m : m = -1) :
  (3 * x - m) / 2 - (x + m) / 3 = 5 / 6 → x = 0 :=
by
  intros h_eq
  rw [h_m] at h_eq
  sorry  -- Proof to be filled in

-- Part (2) Statement
theorem part2_solution (x m : ℝ) (h_x : x = 5)
  (h_eq : (3 * x - m) / 2 - (x + m) / 3 = 5 / 6) :
  (1 / 2) * m^2 + 2 * m = 30 :=
by
  rw [h_x] at h_eq
  sorry  -- Proof to be filled in

end part1_solution_part2_solution_l2088_208879


namespace age_difference_l2088_208826

-- Define the hypothesis and statement
theorem age_difference (A B C : ℕ) 
  (h1 : A + B = B + C + 15)
  (h2 : C = A - 15) : 
  (A + B) - (B + C) = 15 :=
by
  sorry

end age_difference_l2088_208826


namespace arrangements_of_masters_and_apprentices_l2088_208817

theorem arrangements_of_masters_and_apprentices : 
  ∃ n : ℕ, n = 48 ∧ 
     let pairs := 3 
     let ways_to_arrange_pairs := pairs.factorial 
     let ways_to_arrange_within_pairs := 2 ^ pairs 
     ways_to_arrange_pairs * ways_to_arrange_within_pairs = n := 
sorry

end arrangements_of_masters_and_apprentices_l2088_208817


namespace koala_fiber_intake_l2088_208845

theorem koala_fiber_intake (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
by
  sorry

end koala_fiber_intake_l2088_208845


namespace lcm_8_13_14_is_728_l2088_208844

-- Define the numbers and their factorizations
def num1 := 8
def fact1 := 2 ^ 3

def num2 := 13  -- 13 is prime

def num3 := 14
def fact3 := 2 * 7

-- Define the function to calculate the LCM of three integers
def lcm (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- State the theorem to prove that the LCM of 8, 13, and 14 is 728
theorem lcm_8_13_14_is_728 : lcm num1 num2 num3 = 728 :=
by
  -- Prove the equality, skipping proof details with sorry
  sorry

end lcm_8_13_14_is_728_l2088_208844


namespace real_solutions_x_inequality_l2088_208835

theorem real_solutions_x_inequality (x : ℝ) :
  (∃ y : ℝ, y^2 + 6 * x * y + x + 8 = 0) ↔ (x ≤ -8 / 9 ∨ x ≥ 1) := 
sorry

end real_solutions_x_inequality_l2088_208835


namespace roots_polynomial_l2088_208842

theorem roots_polynomial (a b c : ℝ) (h1 : a + b + c = 18) (h2 : a * b + b * c + c * a = 19) (h3 : a * b * c = 8) : 
  (1 + a) * (1 + b) * (1 + c) = 46 :=
by
  sorry

end roots_polynomial_l2088_208842


namespace find_least_multiple_of_50_l2088_208898

def digits (n : ℕ) : List ℕ := n.digits 10

def product_of_digits (n : ℕ) : ℕ := (digits n).prod

theorem find_least_multiple_of_50 :
  ∃ n, (n % 50 = 0) ∧ ((product_of_digits n) % 50 = 0) ∧ (∀ m, (m % 50 = 0) ∧ ((product_of_digits m) % 50 = 0) → n ≤ m) ↔ n = 5550 :=
by sorry

end find_least_multiple_of_50_l2088_208898


namespace total_cars_produced_l2088_208866

def cars_produced_north_america : ℕ := 3884
def cars_produced_europe : ℕ := 2871
def cars_produced_asia : ℕ := 5273
def cars_produced_south_america : ℕ := 1945

theorem total_cars_produced : cars_produced_north_america + cars_produced_europe + cars_produced_asia + cars_produced_south_america = 13973 := by
  sorry

end total_cars_produced_l2088_208866


namespace robin_packages_gum_l2088_208837

/-
Conditions:
1. Robin has 14 packages of candy.
2. There are 6 pieces in each candy package.
3. Robin has 7 additional pieces.
4. Each package of gum contains 6 pieces.

Proof Problem:
Prove that the number of packages of gum Robin has is 15.
-/
theorem robin_packages_gum (candies_packages : ℕ) (pieces_per_candy_package : ℕ)
                          (additional_pieces : ℕ) (pieces_per_gum_package : ℕ) :
  candies_packages = 14 →
  pieces_per_candy_package = 6 →
  additional_pieces = 7 →
  pieces_per_gum_package = 6 →
  (candies_packages * pieces_per_candy_package + additional_pieces) / pieces_per_gum_package = 15 :=
by intros h1 h2 h3 h4; sorry

end robin_packages_gum_l2088_208837


namespace parallelogram_area_l2088_208869

theorem parallelogram_area (base height : ℕ) (h_base : base = 36) (h_height : height = 24) : base * height = 864 := by
  sorry

end parallelogram_area_l2088_208869


namespace advertisement_probability_l2088_208801

theorem advertisement_probability
  (ads_time_hour : ℕ)
  (total_time_hour : ℕ)
  (h1 : ads_time_hour = 20)
  (h2 : total_time_hour = 60) :
  ads_time_hour / total_time_hour = 1 / 3 :=
by
  sorry

end advertisement_probability_l2088_208801


namespace total_earnings_is_correct_l2088_208836

def lloyd_normal_hours : ℝ := 7.5
def lloyd_rate : ℝ := 4.5
def lloyd_overtime_rate : ℝ := 2.0
def lloyd_hours_worked : ℝ := 10.5

def casey_normal_hours : ℝ := 8
def casey_rate : ℝ := 5
def casey_overtime_rate : ℝ := 1.5
def casey_hours_worked : ℝ := 9.5

def lloyd_earnings : ℝ := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours_worked - lloyd_normal_hours) * lloyd_rate * lloyd_overtime_rate)

def casey_earnings : ℝ := (casey_normal_hours * casey_rate) + ((casey_hours_worked - casey_normal_hours) * casey_rate * casey_overtime_rate)

def total_earnings : ℝ := lloyd_earnings + casey_earnings

theorem total_earnings_is_correct : total_earnings = 112 := by
  sorry

end total_earnings_is_correct_l2088_208836


namespace total_parallelograms_in_grid_l2088_208852

theorem total_parallelograms_in_grid (n : ℕ) : 
  ∃ p : ℕ, p = 3 * Nat.choose (n + 2) 4 :=
sorry

end total_parallelograms_in_grid_l2088_208852


namespace solution_ratio_l2088_208870

-- Describe the problem conditions
variable (a b : ℝ) -- amounts of solutions A and B

-- conditions
def proportion_A : ℝ := 0.20 -- Alcohol concentration in solution A
def proportion_B : ℝ := 0.60 -- Alcohol concentration in solution B
def final_proportion : ℝ := 0.40 -- Final alcohol concentration

-- Lean statement
theorem solution_ratio (h : 0.20 * a + 0.60 * b = 0.40 * (a + b)) : a = b := by
  sorry

end solution_ratio_l2088_208870


namespace sin_double_angle_l2088_208854

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 4 / 5) : Real.sin (2 * x) = -7 / 25 := 
by 
  sorry

end sin_double_angle_l2088_208854


namespace find_w_l2088_208824

theorem find_w (p q r u v w : ℝ)
  (h₁ : (x : ℝ) → x^3 - 6 * x^2 + 11 * x + 10 = (x - p) * (x - q) * (x - r))
  (h₂ : (x : ℝ) → x^3 + u * x^2 + v * x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p)))
  (h₃ : p + q + r = 6) :
  w = 80 := sorry

end find_w_l2088_208824


namespace simplify_expression_correct_l2088_208878

def simplify_expression (x : ℝ) : Prop :=
  (5 - 2 * x) - (7 + 3 * x) = -2 - 5 * x

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
  by
    sorry

end simplify_expression_correct_l2088_208878


namespace integers_in_range_eq_l2088_208829

theorem integers_in_range_eq :
  {i : ℤ | i > -2 ∧ i ≤ 3} = {-1, 0, 1, 2, 3} :=
by
  sorry

end integers_in_range_eq_l2088_208829


namespace last_digit_expr_is_4_l2088_208807

-- Definitions for last digits.
def last_digit (n : ℕ) : ℕ := n % 10

def a : ℕ := 287
def b : ℕ := 269

def expr := (a * a) + (b * b) - (2 * a * b)

-- Conjecture stating that the last digit of the given expression is 4.
theorem last_digit_expr_is_4 : last_digit expr = 4 := 
by sorry

end last_digit_expr_is_4_l2088_208807


namespace parabola_translation_l2088_208857

theorem parabola_translation :
  ∀ (x y : ℝ), y = 3 * x^2 →
  ∃ (new_x new_y : ℝ), new_y = 3 * (new_x + 3)^2 - 3 :=
by {
  sorry
}

end parabola_translation_l2088_208857


namespace total_cans_collected_l2088_208873

def bags_on_saturday : ℕ := 6
def bags_on_sunday : ℕ := 3
def cans_per_bag : ℕ := 8
def total_cans : ℕ := 72

theorem total_cans_collected :
  (bags_on_saturday + bags_on_sunday) * cans_per_bag = total_cans :=
by
  sorry

end total_cans_collected_l2088_208873


namespace max_branch_diameter_l2088_208802

theorem max_branch_diameter (d : ℝ) (w : ℝ) (angle : ℝ) (H: w = 1 ∧ angle = 90) :
  d ≤ 2 * Real.sqrt 2 + 2 := 
sorry

end max_branch_diameter_l2088_208802


namespace min_ab_value_l2088_208850

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + 9 * b + 7) : a * b ≥ 49 :=
sorry

end min_ab_value_l2088_208850


namespace expression_bounds_l2088_208828

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 + Real.sqrt 2 ≤ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ∧ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ≤ 4 * Real.sqrt 2 := 
sorry

end expression_bounds_l2088_208828


namespace correct_sample_size_l2088_208815

-- Definitions based on conditions:
def population_size : ℕ := 1800
def sample_size : ℕ := 1000
def surveyed_parents : ℕ := 1000

-- The proof statement we need: 
-- Prove that the sample size is 1000, given the surveyed parents are 1000
theorem correct_sample_size (ps : ℕ) (sp : ℕ) (ss : ℕ) (h1 : ps = population_size) (h2 : sp = surveyed_parents) : ss = sample_size :=
  sorry

end correct_sample_size_l2088_208815


namespace functional_equation_solution_l2088_208822

theorem functional_equation_solution (f : ℤ → ℝ) (hf : ∀ x y : ℤ, f (↑((x + y) / 3)) = (f x + f y) / 2) :
    ∃ c : ℝ, ∀ x : ℤ, x ≠ 0 → f x = c :=
sorry

end functional_equation_solution_l2088_208822


namespace division_result_l2088_208862

def expr := 180 / (12 + 13 * 2)

theorem division_result : expr = 90 / 19 := by
  sorry

end division_result_l2088_208862


namespace line_through_intersection_perpendicular_l2088_208894

theorem line_through_intersection_perpendicular (x y : ℝ) :
  (2 * x - 3 * y + 10 = 0) ∧ (3 * x + 4 * y - 2 = 0) →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a = 2) ∧ (b = 3) ∧ (c = -2) ∧ (3 * a + 2 * b = 0)) :=
by
  sorry

end line_through_intersection_perpendicular_l2088_208894


namespace range_of_a_l2088_208803

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (2 * x + 2) - abs (2 * x - 2) ≤ a) ↔ 4 ≤ a :=
sorry

end range_of_a_l2088_208803


namespace complement_of_intersection_l2088_208830

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end complement_of_intersection_l2088_208830


namespace total_marks_l2088_208843

-- Define the conditions
def average_marks : ℝ := 35
def number_of_candidates : ℕ := 120

-- Define the total marks as a goal to prove
theorem total_marks : number_of_candidates * average_marks = 4200 :=
by
  sorry

end total_marks_l2088_208843


namespace total_games_l2088_208858

-- Defining the conditions.
def games_this_month : ℕ := 9
def games_last_month : ℕ := 8
def games_next_month : ℕ := 7

-- Theorem statement to prove the total number of games.
theorem total_games : games_this_month + games_last_month + games_next_month = 24 := by
  sorry

end total_games_l2088_208858


namespace time_to_carry_backpack_l2088_208888

/-- 
Given:
1. Lara takes 73 seconds to crank open the door to the obstacle course.
2. Lara traverses the obstacle course the second time in 5 minutes and 58 seconds.
3. The total time to complete the obstacle course is 874 seconds.

Prove:
The time it took Lara to carry the backpack through the obstacle course the first time is 443 seconds.
-/
theorem time_to_carry_backpack (door_time : ℕ) (second_traversal_time : ℕ) (total_time : ℕ) : 
  (door_time + second_traversal_time + 443 = total_time) :=
by
  -- Given conditions
  let door_time := 73
  let second_traversal_time := 5 * 60 + 58 -- Convert 5 minutes 58 seconds to seconds
  let total_time := 874
  -- Calculate the time to carry the backpack
  sorry

end time_to_carry_backpack_l2088_208888


namespace range_of_m_l2088_208882

-- Definitions and the main problem statement
def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ (-4 < m ∧ m ≤ 0) :=
by
  sorry

end range_of_m_l2088_208882


namespace part_I_part_II_l2088_208880

noncomputable def f (x a : ℝ) := |x - 4| + |x - a|

theorem part_I (x : ℝ) : (f x 2 > 10) ↔ (x > 8 ∨ x < -2) :=
by sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≥ 1) ↔ (a ≥ 5 ∨ a ≤ 3) :=
by sorry

end part_I_part_II_l2088_208880


namespace parallel_line_equation_l2088_208881

theorem parallel_line_equation :
  ∃ (c : ℝ), 
    (∀ x : ℝ, y = (3 / 4) * x + 6 → (y = (3 / 4) * x + c → abs (c - 6) = 4 * (5 / 4))) → c = 1 :=
by
  sorry

end parallel_line_equation_l2088_208881


namespace HCl_moles_formed_l2088_208827

-- Define the conditions for the problem:
def moles_H2SO4 := 1 -- moles of H2SO4
def moles_NaCl := 1 -- moles of NaCl
def reaction : List (Int × String) :=
  [(1, "H2SO4"), (2, "NaCl"), (2, "HCl"), (1, "Na2SO4")]  -- the reaction coefficients in (coefficient, chemical) pairs

-- Define the function that calculates the product moles based on limiting reactant
def calculate_HCl (moles_H2SO4 : Int) (moles_NaCl : Int) : Int :=
  if moles_NaCl < 2 then moles_NaCl else 2 * (moles_H2SO4 / 1)

-- Specify the theorem to be proven with the given conditions
theorem HCl_moles_formed :
  calculate_HCl moles_H2SO4 moles_NaCl = 1 :=
by
  sorry -- Proof can be filled in later

end HCl_moles_formed_l2088_208827


namespace sufficient_but_not_necessary_condition_l2088_208813

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 - m * x + 1 > 0) → -2 < m ∧ m < 2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l2088_208813


namespace orange_juice_fraction_l2088_208856

theorem orange_juice_fraction :
  let capacity1 := 500
  let capacity2 := 600
  let fraction1 := (1/4 : ℚ)
  let fraction2 := (1/3 : ℚ)
  let juice1 := capacity1 * fraction1
  let juice2 := capacity2 * fraction2
  let total_juice := juice1 + juice2
  let total_volume := capacity1 + capacity2
  (total_juice / total_volume = (13/44 : ℚ)) := sorry

end orange_juice_fraction_l2088_208856


namespace central_angle_of_section_l2088_208884

theorem central_angle_of_section (A : ℝ) (x: ℝ) (H : (1 / 8 : ℝ) = (x / 360)) : x = 45 :=
by
  sorry

end central_angle_of_section_l2088_208884


namespace initial_back_squat_weight_l2088_208853

-- Define a structure to encapsulate the conditions
structure squat_conditions where
  initial_back_squat : ℝ
  front_squat_ratio : ℝ := 0.8
  back_squat_increase : ℝ := 50
  front_squat_triple_ratio : ℝ := 0.9
  total_weight_moved : ℝ := 540

-- Using the conditions provided to prove John's initial back squat weight
theorem initial_back_squat_weight (c : squat_conditions) :
  (3 * 3 * (c.front_squat_triple_ratio * (c.front_squat_ratio * c.initial_back_squat)) = c.total_weight_moved) →
  c.initial_back_squat = 540 / 6.48 := sorry

end initial_back_squat_weight_l2088_208853


namespace percentage_change_area_l2088_208863

theorem percentage_change_area
    (L B : ℝ)
    (Area_original : ℝ) (Area_new : ℝ)
    (Length_new : ℝ) (Breadth_new : ℝ) :
    Area_original = L * B →
    Length_new = L / 2 →
    Breadth_new = 3 * B →
    Area_new = Length_new * Breadth_new →
    (Area_new - Area_original) / Area_original * 100 = 50 :=
  by
  intro h_orig_area hl_new hb_new ha_new
  sorry

end percentage_change_area_l2088_208863


namespace triangle_angle_C_l2088_208825

theorem triangle_angle_C (A B C : ℝ) (sin cos : ℝ → ℝ) 
  (h1 : 3 * sin A + 4 * cos B = 6)
  (h2 : 4 * sin B + 3 * cos A = 1)
  (triangle_sum : A + B + C = 180) :
  C = 30 :=
by
  sorry

end triangle_angle_C_l2088_208825


namespace ratio_of_areas_l2088_208872

theorem ratio_of_areas (n : ℕ) (r s : ℕ) (square_area : ℕ) (triangle_adf_area : ℕ)
  (h_square_area : square_area = 4)
  (h_triangle_adf_area : triangle_adf_area = n * square_area)
  (h_triangle_sim : s = 8 / r)
  (h_r_eq_n : r = n):
  (s / square_area) = 2 / n :=
by
  sorry

end ratio_of_areas_l2088_208872


namespace number_of_sides_of_polygon_l2088_208810

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 540) : n = 5 :=
by
  sorry

end number_of_sides_of_polygon_l2088_208810


namespace expression_value_l2088_208832

theorem expression_value (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x :=
by
  -- Insert proof here
  sorry

end expression_value_l2088_208832


namespace percentage_exceeds_l2088_208859

-- Defining the constants and conditions
variables {y z x : ℝ}

-- Conditions
def condition1 (y x : ℝ) : Prop := x = 0.6 * y
def condition2 (x z : ℝ) : Prop := z = 1.25 * x

-- Proposition to prove
theorem percentage_exceeds (hyx : condition1 y x) (hxz : condition2 x z) : y = 4/3 * z :=
by 
  -- We skip the proof as requested
  sorry

end percentage_exceeds_l2088_208859


namespace rational_solution_quadratic_l2088_208893

theorem rational_solution_quadratic (m : ℕ) (h_pos : m > 0) : 
  (∃ (x : ℚ), x * x * m + 25 * x + m = 0) ↔ m = 10 ∨ m = 12 :=
by sorry

end rational_solution_quadratic_l2088_208893


namespace part1_part2_l2088_208804

noncomputable def f (a x : ℝ) : ℝ := a * x - a * Real.log x - Real.exp x / x

theorem part1 (a : ℝ) :
  (∀ x > 0, f a x < 0) → a < Real.exp 1 :=
sorry

theorem part2 (a : ℝ) (x1 x2 x3 : ℝ) :
  (∀ x, f a x = 0 → x = x1 ∨ x = x2 ∨ x = x3) ∧
  f a x1 + f a x2 + f a x3 ≤ 3 * Real.exp 2 - Real.exp 1 →
  Real.exp 1 < a ∧ a ≤ Real.exp 2 :=
sorry

end part1_part2_l2088_208804


namespace necessary_condition_of_equilateral_triangle_l2088_208899

variable {A B C: ℝ}
variable {a b c: ℝ}

theorem necessary_condition_of_equilateral_triangle
  (h1 : B + C = 2 * A)
  (h2 : b + c = 2 * a)
  : (A = B ∧ B = C ∧ a = b ∧ b = c) ↔ (B + C = 2 * A ∧ b + c = 2 * a) := 
by
  sorry

end necessary_condition_of_equilateral_triangle_l2088_208899


namespace prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l2088_208846

section ParkingProblem

variable (P_A_more_1_no_more_2 : ℚ) (P_A_more_than_14 : ℚ)

theorem prob_A_fee_exactly_6_yuan :
  (P_A_more_1_no_more_2 = 1/3) →
  (P_A_more_than_14 = 5/12) →
  (1 - (P_A_more_1_no_more_2 + P_A_more_than_14)) = 1/4 :=
by
  -- Skipping the proof
  intros _ _
  sorry

theorem prob_sum_fees_A_B_36_yuan :
  (1/4 : ℚ) = 1/4 :=
by
  -- Skipping the proof
  exact rfl

end ParkingProblem

end prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l2088_208846


namespace board_partition_possible_l2088_208809

variable (m n : ℕ)

theorem board_partition_possible (hm : m > 15) (hn : n > 15) :
  ((∃ k1, m = 5 * k1 ∧ ∃ k2, n = 4 * k2) ∨ (∃ k3, m = 4 * k3 ∧ ∃ k4, n = 5 * k4)) :=
sorry

end board_partition_possible_l2088_208809


namespace find_number_l2088_208891

theorem find_number (x : ℝ) (h : 0.4 * x = 15) : x = 37.5 := by
  sorry

end find_number_l2088_208891


namespace number_of_ways_is_64_l2088_208805

-- Definition of the problem conditions
def ways_to_sign_up (students groups : ℕ) : ℕ :=
  groups ^ students

-- Theorem statement asserting that for 3 students and 4 groups, the number of ways is 64
theorem number_of_ways_is_64 : ways_to_sign_up 3 4 = 64 :=
by sorry

end number_of_ways_is_64_l2088_208805


namespace whose_number_is_larger_l2088_208851

theorem whose_number_is_larger
    (vasya_prod : ℕ := 4^12)
    (petya_prod : ℕ := 2^25) :
    petya_prod > vasya_prod :=
    by
    sorry

end whose_number_is_larger_l2088_208851


namespace parallel_lines_perpendicular_lines_l2088_208875

theorem parallel_lines (t s k : ℝ) :
  (∀ t, ∃ s, (1 - 2 * t = s) ∧ (2 + k * t = 1 - 2 * s)) →
  k = 4 :=
by
  sorry

theorem perpendicular_lines (t s k : ℝ) :
  (∀ t, ∃ s, (1 - 2 * t = s) ∧ (2 + k * t = 1 - 2 * s)) →
  k = -1 :=
by
  sorry

end parallel_lines_perpendicular_lines_l2088_208875


namespace prove_ratio_chickens_pigs_horses_sheep_l2088_208848

noncomputable def ratio_chickens_pigs_horses_sheep (c p h s : ℕ) : Prop :=
  (∃ k : ℕ, c = 26*k ∧ p = 5*k) ∧
  (∃ l : ℕ, s = 25*l ∧ h = 9*l) ∧
  (∃ m : ℕ, p = 10*m ∧ h = 3*m) ∧
  c = 156 ∧ p = 30 ∧ h = 9 ∧ s = 25

theorem prove_ratio_chickens_pigs_horses_sheep (c p h s : ℕ) :
  ratio_chickens_pigs_horses_sheep c p h s :=
sorry

end prove_ratio_chickens_pigs_horses_sheep_l2088_208848


namespace sum_of_angles_is_290_l2088_208877

-- Given conditions
def angle_A : ℝ := 40
def angle_C : ℝ := 70
def angle_D : ℝ := 50
def angle_F : ℝ := 60

-- Calculate angle B (which is same as angle E)
def angle_B : ℝ := 180 - angle_A - angle_C
def angle_E := angle_B  -- by the condition that B and E are identical

-- Total sum of angles
def total_angle_sum : ℝ := angle_A + angle_B + angle_C + angle_D + angle_F

-- Theorem statement
theorem sum_of_angles_is_290 : total_angle_sum = 290 := by
  sorry

end sum_of_angles_is_290_l2088_208877


namespace giuseppe_can_cut_rectangles_l2088_208889

theorem giuseppe_can_cut_rectangles : 
  let board_length := 22
  let board_width := 15
  let rectangle_length := 3
  let rectangle_width := 5
  (board_length * board_width) / (rectangle_length * rectangle_width) = 22 :=
by
  sorry

end giuseppe_can_cut_rectangles_l2088_208889


namespace powers_of_2_not_powers_of_4_below_1000000_equals_10_l2088_208838

def num_powers_of_2_not_4 (n : ℕ) : ℕ :=
  let powers_of_2 := (List.range n).filter (fun k => (2^k < 1000000));
  let powers_of_4 := (List.range n).filter (fun k => (4^k < 1000000));
  powers_of_2.length - powers_of_4.length

theorem powers_of_2_not_powers_of_4_below_1000000_equals_10 : 
  num_powers_of_2_not_4 20 = 10 :=
by
  sorry

end powers_of_2_not_powers_of_4_below_1000000_equals_10_l2088_208838


namespace grain_milling_l2088_208808

theorem grain_milling (A : ℚ) (h1 : 0.9 * A = 100) : A = 111 + 1 / 9 :=
by
  sorry

end grain_milling_l2088_208808


namespace class_mean_correct_l2088_208874

noncomputable def new_class_mean (number_students_midterm : ℕ) (avg_score_midterm : ℚ)
                                 (number_students_next_day : ℕ) (avg_score_next_day : ℚ)
                                 (number_students_final_day : ℕ) (avg_score_final_day : ℚ)
                                 (total_students : ℕ) : ℚ :=
  let total_score_midterm := number_students_midterm * avg_score_midterm
  let total_score_next_day := number_students_next_day * avg_score_next_day
  let total_score_final_day := number_students_final_day * avg_score_final_day
  let total_score := total_score_midterm + total_score_next_day + total_score_final_day
  total_score / total_students

theorem class_mean_correct :
  new_class_mean 50 65 8 85 2 55 60 = 67 :=
by
  sorry

end class_mean_correct_l2088_208874


namespace problem_inequality_solution_set_problem_minimum_value_l2088_208886

noncomputable def f (x : ℝ) := x^2 / (x - 1)

theorem problem_inequality_solution_set : 
  ∀ x : ℝ, 1 < x ∧ x < (1 + Real.sqrt 5) / 2 → f x > 2 * x + 1 :=
sorry

theorem problem_minimum_value : ∀ x : ℝ, x > 1 → (f x ≥ 4) ∧ (f 2 = 4) :=
sorry

end problem_inequality_solution_set_problem_minimum_value_l2088_208886


namespace fraction_multiplication_validity_l2088_208855

theorem fraction_multiplication_validity (a b m x : ℝ) (hb : b ≠ 0) : 
  (x ≠ m) ↔ (b * (x - m) ≠ 0) :=
by
  sorry

end fraction_multiplication_validity_l2088_208855


namespace tan_ratio_of_triangle_l2088_208841

theorem tan_ratio_of_triangle (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = (3 / 5) * c) : 
  Real.tan A / Real.tan B = 4 :=
sorry

end tan_ratio_of_triangle_l2088_208841


namespace total_money_spent_l2088_208806

theorem total_money_spent {s j : ℝ} (hs : s = 14.28) (hj : j = 4.74) : s + j = 19.02 :=
by
  sorry

end total_money_spent_l2088_208806


namespace find_sum_l2088_208814

def f (x : ℝ) : ℝ := sorry

axiom f_non_decreasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → x1 ≤ 1 → 0 ≤ x2 → x2 ≤ 1 → x1 < x2 → f x1 ≤ f x2
axiom f_at_0 : f 0 = 0
axiom f_scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (x / 3) = (1 / 2) * f x
axiom f_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (1 - x) = 1 - f x

theorem find_sum :
  f (1 / 3) + f (2 / 3) + f (1 / 9) + f (1 / 6) + f (1 / 8) = 7 / 4 :=
by
  sorry

end find_sum_l2088_208814


namespace abs_ineq_l2088_208823

open Real

noncomputable def absolute_value (x : ℝ) : ℝ := abs x

theorem abs_ineq (a b c : ℝ) (h1 : a + b ≥ 0) (h2 : b + c ≥ 0) (h3 : c + a ≥ 0) :
  a + b + c ≥ (absolute_value a + absolute_value b + absolute_value c) / 3 := by
  sorry

end abs_ineq_l2088_208823


namespace five_digit_numbers_with_alternating_parity_l2088_208871

theorem five_digit_numbers_with_alternating_parity : 
  ∃ n : ℕ, n = 5625 ∧ ∀ (x : ℕ), (10000 ≤ x ∧ x < 100000) → 
    (∀ i, i < 4 → (((x / 10^i) % 10) % 2 ≠ ((x / 10^(i+1)) % 10) % 2)) ↔ 
    (x = 5625) := 
sorry

end five_digit_numbers_with_alternating_parity_l2088_208871


namespace sale_price_after_discounts_l2088_208883

def original_price : ℝ := 400.00
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.10

theorem sale_price_after_discounts (orig : ℝ) (d1 d2 d3 : ℝ) :
  orig = original_price →
  d1 = discount1 →
  d2 = discount2 →
  d3 = discount3 →
  orig * (1 - d1) * (1 - d2) * (1 - d3) = 243.00 := by
  sorry

end sale_price_after_discounts_l2088_208883


namespace square_inscribed_in_hexagon_has_side_length_l2088_208816

-- Definitions for the conditions given
noncomputable def side_length_square (AB EF : ℝ) : ℝ :=
  if AB = 30 ∧ EF = 19 * (Real.sqrt 3 - 1) then 10 * Real.sqrt 3 else 0

-- The theorem stating the specified equality
theorem square_inscribed_in_hexagon_has_side_length (AB EF : ℝ)
  (hAB : AB = 30) (hEF : EF = 19 * (Real.sqrt 3 - 1)) :
  side_length_square AB EF = 10 * Real.sqrt 3 := 
by 
  -- This is the proof placeholder
  sorry

end square_inscribed_in_hexagon_has_side_length_l2088_208816


namespace ab_value_l2088_208868

theorem ab_value (a b : ℝ) (h : 6 * a = 20 ∧ 7 * b = 20) : 84 * (a * b) = 800 :=
by sorry

end ab_value_l2088_208868


namespace number_of_pencils_purchased_l2088_208820

variable {total_pens : ℕ} (total_cost : ℝ) (avg_price_pencil avg_price_pen : ℝ)

theorem number_of_pencils_purchased 
  (h1 : total_pens = 30)
  (h2 : total_cost = 570)
  (h3 : avg_price_pencil = 2.00)
  (h4 : avg_price_pen = 14)
  : 
  ∃ P : ℕ, P = 75 :=
by
  sorry

end number_of_pencils_purchased_l2088_208820


namespace q_evaluation_l2088_208861

def q (x y : ℤ) : ℤ :=
if x ≥ 0 ∧ y ≤ 0 then x - y
else if x < 0 ∧ y > 0 then x + 3 * y
else 4 * x - 2 * y

theorem q_evaluation : q (q 2 (-3)) (q (-4) 1) = 6 :=
by
  sorry

end q_evaluation_l2088_208861


namespace max_value_expression_l2088_208839

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l2088_208839


namespace divisor_of_12401_76_13_l2088_208833

theorem divisor_of_12401_76_13 (D : ℕ) (h1: 12401 = (D * 76) + 13) : D = 163 :=
sorry

end divisor_of_12401_76_13_l2088_208833


namespace circumscribed_sphere_surface_area_l2088_208818

theorem circumscribed_sphere_surface_area 
    (x y z : ℝ) 
    (h1 : x * y = Real.sqrt 6) 
    (h2 : y * z = Real.sqrt 2) 
    (h3 : z * x = Real.sqrt 3) : 
    4 * Real.pi * ((Real.sqrt (x^2 + y^2 + z^2)) / 2)^2 = 6 * Real.pi := 
by
  sorry

end circumscribed_sphere_surface_area_l2088_208818


namespace non_binary_listeners_l2088_208885

theorem non_binary_listeners (listen_total males_listen females_dont_listen non_binary_dont_listen dont_listen_total : ℕ) 
  (h_listen_total : listen_total = 250) 
  (h_males_listen : males_listen = 85) 
  (h_females_dont_listen : females_dont_listen = 95) 
  (h_non_binary_dont_listen : non_binary_dont_listen = 45) 
  (h_dont_listen_total : dont_listen_total = 230) : 
  (listen_total - males_listen - (dont_listen_total - females_dont_listen - non_binary_dont_listen)) = 70 :=
by 
  -- Let nbl be the number of non-binary listeners
  let nbl := listen_total - males_listen - (dont_listen_total - females_dont_listen - non_binary_dont_listen)
  -- We need to show nbl = 70
  show nbl = 70
  sorry

end non_binary_listeners_l2088_208885


namespace factorization_correct_l2088_208896

-- Define noncomputable to deal with the natural arithmetic operations
noncomputable def a : ℕ := 66
noncomputable def b : ℕ := 231

-- Define the given expressions
noncomputable def lhs (x : ℕ) : ℤ := ((a : ℤ) * x^6) - ((b : ℤ) * x^12)
noncomputable def rhs (x : ℕ) : ℤ := (33 : ℤ) * x^6 * (2 - 7 * x^6)

-- The theorem to prove the equality
theorem factorization_correct (x : ℕ) : lhs x = rhs x :=
by sorry

end factorization_correct_l2088_208896


namespace min_width_l2088_208865

theorem min_width (w : ℝ) (h : w * (w + 20) ≥ 150) : w ≥ 10 := by
  sorry

end min_width_l2088_208865


namespace find_positive_integers_l2088_208867

theorem find_positive_integers 
    (a b : ℕ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (h1 : ∃ k1 : ℤ, (a^3 * b - 1) = k1 * (a + 1))
    (h2 : ∃ k2 : ℤ, (b^3 * a + 1) = k2 * (b - 1)) : 
    (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
sorry

end find_positive_integers_l2088_208867


namespace alex_friends_invite_l2088_208840

theorem alex_friends_invite (burger_buns_per_pack : ℕ)
                            (packs_of_buns : ℕ)
                            (buns_needed_by_each_guest : ℕ)
                            (total_buns : ℕ)
                            (friends_who_dont_eat_buns : ℕ)
                            (friends_who_dont_eat_meat : ℕ)
                            (total_friends_invited : ℕ) 
                            (h1 : burger_buns_per_pack = 8)
                            (h2 : packs_of_buns = 3)
                            (h3 : buns_needed_by_each_guest = 3)
                            (h4 : total_buns = packs_of_buns * burger_buns_per_pack)
                            (h5 : friends_who_dont_eat_buns = 1)
                            (h6 : friends_who_dont_eat_meat = 1)
                            (h7 : total_friends_invited = (total_buns / buns_needed_by_each_guest) + friends_who_dont_eat_buns) :
  total_friends_invited = 9 :=
by sorry

end alex_friends_invite_l2088_208840


namespace area_of_transformed_region_l2088_208847

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 0], ![5, 3]]
def area_T : ℝ := 9

-- Theorem statement
theorem area_of_transformed_region : 
  let det_matrix := matrix.det
  (det_matrix = 9) → (area_T = 9) → (area_T * det_matrix = 81) :=
by
  intros h₁ h₂
  sorry

end area_of_transformed_region_l2088_208847


namespace value_of_f_f_2_l2088_208887

def f (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x - 1

theorem value_of_f_f_2 : f (f 2) = 164 := by
  sorry

end value_of_f_f_2_l2088_208887


namespace find_angle_l2088_208890

theorem find_angle (a b c d e : ℝ) (sum_of_hexagon_angles : ℝ) (h_sum : a = 135 ∧ b = 120 ∧ c = 105 ∧ d = 150 ∧ e = 110 ∧ sum_of_hexagon_angles = 720) : 
  ∃ P : ℝ, a + b + c + d + e + P = sum_of_hexagon_angles ∧ P = 100 :=
by
  sorry

end find_angle_l2088_208890


namespace smallest_number_of_students_l2088_208831

theorem smallest_number_of_students
  (n : ℕ)
  (h1 : 3 * 90 + (n - 3) * 65 ≤ n * 80)
  (h2 : ∀ k, k ≤ n - 3 → 65 ≤ k)
  (h3 : (3 * 90) + ((n - 3) * 65) / n = 80) : n = 5 :=
sorry

end smallest_number_of_students_l2088_208831


namespace find_alpha_l2088_208800

-- Define the problem in Lean terms
variable (x y α : ℝ)

-- Conditions
def condition1 : Prop := 3 + α + y = 4 + α + x
def condition2 : Prop := 1 + x + 3 + 3 + α + y + 4 + 1 = 2 * (4 + α + x)

-- The theorem to prove
theorem find_alpha (h1 : condition1 x y α) (h2 : condition2 x y α) : α = 5 := 
  sorry

end find_alpha_l2088_208800


namespace min_max_value_expression_l2088_208821

theorem min_max_value_expression
  (x1 x2 x3 : ℝ) 
  (hx : x1 + x2 + x3 = 1)
  (hx1 : 0 ≤ x1)
  (hx2 : 0 ≤ x2)
  (hx3 : 0 ≤ x3) :
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5) = 1 := 
sorry

end min_max_value_expression_l2088_208821


namespace sally_picked_11_pears_l2088_208864

theorem sally_picked_11_pears (total_pears : ℕ) (pears_picked_by_Sara : ℕ) (pears_picked_by_Sally : ℕ) 
    (h1 : total_pears = 56) (h2 : pears_picked_by_Sara = 45) :
    pears_picked_by_Sally = total_pears - pears_picked_by_Sara := by
  sorry

end sally_picked_11_pears_l2088_208864
