import Mathlib

namespace friend_saves_per_week_l232_232699

theorem friend_saves_per_week
  (x : ℕ) 
  (you_have : ℕ := 160)
  (you_save_per_week : ℕ := 7)
  (friend_have : ℕ := 210)
  (weeks : ℕ := 25)
  (total_you_save : ℕ := you_have + you_save_per_week * weeks)
  (total_friend_save : ℕ := friend_have + x * weeks) 
  (h : total_you_save = total_friend_save) : x = 5 := 
by 
  sorry

end friend_saves_per_week_l232_232699


namespace num_factors_60_l232_232812

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l232_232812


namespace sum_floor_sqrt_1_to_25_l232_232741

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l232_232741


namespace hannah_spent_65_l232_232627

-- Definitions based on the conditions
def sweatshirts_count : ℕ := 3
def t_shirts_count : ℕ := 2
def sweatshirt_cost : ℕ := 15
def t_shirt_cost : ℕ := 10

-- The total amount spent
def total_spent : ℕ := sweatshirts_count * sweatshirt_cost + t_shirts_count * t_shirt_cost

-- The theorem stating the problem
theorem hannah_spent_65 : total_spent = 65 :=
by
  sorry

end hannah_spent_65_l232_232627


namespace chickens_cheaper_than_buying_eggs_l232_232871

theorem chickens_cheaper_than_buying_eggs :
  ∃ W, W ≥ 80 ∧ 80 + W ≤ 2 * W :=
by
  sorry

end chickens_cheaper_than_buying_eggs_l232_232871


namespace line_intersects_circle_l232_232791

noncomputable def diameter : ℝ := 8
noncomputable def radius : ℝ := diameter / 2
noncomputable def center_to_line_distance : ℝ := 3

theorem line_intersects_circle :
  center_to_line_distance < radius → True :=
by {
  /- The proof would go here, but for now, we use sorry. -/
  sorry
}

end line_intersects_circle_l232_232791


namespace triangle_is_isosceles_l232_232230

/-- Given triangle ABC with angles A, B, and C, where C = π - (A + B),
    if 2 * sin A * cos B = sin C, then triangle ABC is an isosceles triangle -/
theorem triangle_is_isosceles
  (A B C : ℝ)
  (hC : C = π - (A + B))
  (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  A = B :=
by
  sorry

end triangle_is_isosceles_l232_232230


namespace find_coefficients_l232_232602

theorem find_coefficients (A B C D : ℚ) :
  (∀ x : ℚ, x ≠ -1 → 
  (A / (x + 1)) + (B / (x + 1)^2) + ((C * x + D) / (x^2 + x + 1)) = 
  1 / ((x + 1)^2 * (x^2 + x + 1))) →
  A = 1 ∧ B = 1 ∧ C = -1 ∧ D = -1 :=
sorry

end find_coefficients_l232_232602


namespace gcd_of_three_numbers_l232_232768

theorem gcd_of_three_numbers : 
  let a := 4560
  let b := 6080
  let c := 16560
  gcd (gcd a b) c = 80 := 
by {
  -- placeholder for the proof
  sorry
}

end gcd_of_three_numbers_l232_232768


namespace problem1_problem2_l232_232706

-- Proof problem for the first condition
theorem problem1 {p : ℕ} (hp : Nat.Prime p) 
  (h : ∃ n : ℕ, (7^(p-1) - 1) = p * n^2) : p = 3 :=
sorry

-- Proof problem for the second condition
theorem problem2 {p : ℕ} (hp : Nat.Prime p)
  (h : ∃ n : ℕ, (11^(p-1) - 1) = p * n^2) : false :=
sorry

end problem1_problem2_l232_232706


namespace six_parallelepipeds_visibility_l232_232594

structure Point (space : Type) := { coords : space }

structure Parallelepiped (space : Type) :=
  (vertices : set (Point space)) -- Set of vertices
  (opaque : Prop := True)

theorem six_parallelepipeds_visibility :
  ∃ (p : Point ℝ^3),
    ∀ (parallelepipeds : Fin 6 → Parallelepiped ℝ^3),
      (∀ i j, i ≠ j → Disjoint (parallelepipeds i).vertices (parallelepipeds j).vertices) → 
      (∀ i, p ∉ (parallelepipeds i).vertices) →
      ∀ i, ¬ (∃ vertex, vertex ∈ (parallelepipeds i).vertices ∧ 
                          visible_from p vertex) :=
begin
  sorry -- Proof goes here
end

end six_parallelepipeds_visibility_l232_232594


namespace range_of_numbers_is_six_l232_232253

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l232_232253


namespace min_value_expr_l232_232171

theorem min_value_expr (a b : ℝ) (h : a * b > 0) : (a^4 + 4 * b^4 + 1) / (a * b) ≥ 4 := 
sorry

end min_value_expr_l232_232171


namespace rectangle_area_ratio_l232_232705

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 4) (h2 : b / d = 3 / 4) :
  (a * b) / (c * d) = 9 / 16 := 
  sorry

end rectangle_area_ratio_l232_232705


namespace num_factors_of_60_l232_232830

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l232_232830


namespace num_factors_of_60_l232_232825

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l232_232825


namespace range_of_set_l232_232261

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l232_232261


namespace first_place_clay_l232_232101

def Clay := "Clay"
def Allen := "Allen"
def Bart := "Bart"
def Dick := "Dick"

-- Statements made by the participants
def Allen_statements := ["I finished right before Bart", "I am not the first"]
def Bart_statements := ["I finished right before Clay", "I am not the second"]
def Clay_statements := ["I finished right before Dick", "I am not the third"]
def Dick_statements := ["I finished right before Allen", "I am not the last"]

-- Conditions
def only_two_true_statements : Prop := sorry -- This represents the condition that only two of these statements are true.
def first_place_told_truth : Prop := sorry -- This represents the condition that the person who got first place told at least one truth.

def person_first_place := Clay

theorem first_place_clay : person_first_place = Clay ∧ only_two_true_statements ∧ first_place_told_truth := 
sorry

end first_place_clay_l232_232101


namespace math_problem_l232_232369

theorem math_problem 
  (x y z : ℚ)
  (h1 : 4 * x - 5 * y - z = 0)
  (h2 : x + 5 * y - 18 * z = 0)
  (hz : z ≠ 0) :
  (x^2 + 4 * x * y) / (y^2 + z^2) = 3622 / 9256 := 
sorry

end math_problem_l232_232369


namespace day_of_week_after_2_power_50_days_l232_232510

-- Conditions:
def today_is_monday : ℕ := 1  -- Monday corresponds to 1

def days_later (n : ℕ) : ℕ := (today_is_monday + n) % 7

theorem day_of_week_after_2_power_50_days :
  days_later (2^50) = 6 :=  -- Saturday corresponds to 6 (0 is Sunday)
by {
  -- Proof steps are skipped
  sorry
}

end day_of_week_after_2_power_50_days_l232_232510


namespace megan_pages_left_l232_232658

theorem megan_pages_left (total_problems completed_problems problems_per_page : ℕ)
    (h_total : total_problems = 40)
    (h_completed : completed_problems = 26)
    (h_problems_per_page : problems_per_page = 7) :
    (total_problems - completed_problems) / problems_per_page = 2 :=
by
  sorry

end megan_pages_left_l232_232658


namespace trigonometric_identity_l232_232612

open Real

theorem trigonometric_identity (α : ℝ) (hα : sin (2 * π - α) = 4 / 5) (hα_range : 3 * π / 2 < α ∧ α < 2 * π) : 
  (sin α + cos α) / (sin α - cos α) = 1 / 7 := 
by
  sorry

end trigonometric_identity_l232_232612


namespace amount_in_cup_after_division_l232_232673

theorem amount_in_cup_after_division (removed remaining cups : ℕ) (h : remaining + removed = 40) : 
  (40 / cups = 8) :=
by
  sorry

end amount_in_cup_after_division_l232_232673


namespace difference_of_squares_l232_232362

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l232_232362


namespace find_a_l232_232137

theorem find_a (a : ℝ) : 
  (∀ (i : ℂ), i^2 = -1 → (a * i / (2 - i) + 1 = 2 * i)) → a = 5 :=
by
  intro h
  sorry

end find_a_l232_232137


namespace increase_in_green_chameleons_is_11_l232_232054

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l232_232054


namespace quadratic_function_min_value_at_1_l232_232550

-- Define the quadratic function y = (x - 1)^2 - 3
def quadratic_function (x : ℝ) : ℝ :=
  (x - 1) ^ 2 - 3

-- The theorem to prove is that this quadratic function reaches its minimum value when x = 1.
theorem quadratic_function_min_value_at_1 : ∃ x : ℝ, quadratic_function x = quadratic_function 1 :=
by
  sorry

end quadratic_function_min_value_at_1_l232_232550


namespace plane_perpendicular_l232_232852

-- Define types for lines and planes
axiom Line : Type
axiom Plane : Type

-- Define the relationships between lines and planes
axiom Parallel (l : Line) (p : Plane) : Prop
axiom Perpendicular (l : Line) (p : Plane) : Prop
axiom PlanePerpendicular (p1 p2 : Plane) : Prop

-- The setting conditions
variables (c : Line) (α β : Plane)

-- The given conditions
axiom c_perpendicular_β : Perpendicular c β
axiom c_parallel_α : Parallel c α

-- The proof goal (without the proof body)
theorem plane_perpendicular : PlanePerpendicular α β :=
by
  sorry

end plane_perpendicular_l232_232852


namespace price_of_first_variety_l232_232875

theorem price_of_first_variety
  (P : ℝ)
  (H1 : 1 * P + 1 * 135 + 2 * 175.5 = 4 * 153) :
  P = 126 :=
by
  sorry

end price_of_first_variety_l232_232875


namespace orange_juice_serving_size_l232_232724

theorem orange_juice_serving_size (n_servings : ℕ) (c_concentrate : ℕ) (v_concentrate : ℕ) (c_water_per_concentrate : ℕ)
    (v_cans : ℕ) (expected_serving_size : ℕ) 
    (h1 : n_servings = 200)
    (h2 : c_concentrate = 60)
    (h3 : v_concentrate = 5)
    (h4 : c_water_per_concentrate = 3)
    (h5 : v_cans = 5)
    (h6 : expected_serving_size = 6) : 
   (c_concentrate * v_concentrate + c_concentrate * c_water_per_concentrate * v_cans) / n_servings = expected_serving_size := 
by 
  sorry

end orange_juice_serving_size_l232_232724


namespace perfect_square_and_integer_c_exists_l232_232384

-- Define the problem statement and conditions
theorem perfect_square_and_integer_c_exists (q a b : ℕ) (h : a ^ 2 - q * a * b + b ^ 2 = q) : 
    (∃ c : ℤ, c ≠ a ∧ c ^ 2 - q * b * c + b ^ 2 = q) ∧ ∃ k : ℕ, q = k ^ 2 := 
by
  sorry

end perfect_square_and_integer_c_exists_l232_232384


namespace cream_cheese_cost_l232_232007

theorem cream_cheese_cost
  (B C : ℝ)
  (h1 : 2 * B + 3 * C = 12)
  (h2 : 4 * B + 2 * C = 14) :
  C = 2.5 :=
by
  sorry

end cream_cheese_cost_l232_232007


namespace max_volume_of_pyramid_PABC_l232_232479

noncomputable def max_pyramid_volume (PA PB AB BC CA : ℝ) (hPA : PA = 3) (hPB : PB = 3) 
(hAB : AB = 2) (hBC : BC = 2) (hCA : CA = 2) : ℝ :=
  let D := 1 -- Midpoint of segment AB
  let PD : ℝ := Real.sqrt (PA ^ 2 - D ^ 2) -- Distance PD using Pythagorean theorem
  let S_ABC : ℝ := (Real.sqrt 3 / 4) * (AB ^ 2) -- Area of triangle ABC
  let V_PABC : ℝ := (1 / 3) * S_ABC * PD -- Volume of the pyramid
  V_PABC -- Return the volume

theorem max_volume_of_pyramid_PABC : 
  max_pyramid_volume 3 3 2 2 2  (rfl) (rfl) (rfl) (rfl) (rfl) = (2 * Real.sqrt 6) / 3 :=
by
  sorry

end max_volume_of_pyramid_PABC_l232_232479


namespace reduced_price_of_oil_is_40_l232_232580

variables 
  (P R : ℝ) 
  (hP : 0 < P)
  (hR : R = 0.75 * P)
  (hw : 800 / (0.75 * P) = 800 / P + 5)

theorem reduced_price_of_oil_is_40 : R = 40 :=
sorry

end reduced_price_of_oil_is_40_l232_232580


namespace max_withdrawal_l232_232865

def initial_balance : ℕ := 500
def withdraw_amount : ℕ := 300
def add_amount : ℕ := 198
def remaining_balance (x : ℕ) : Prop := 
  x % 6 = 0 ∧ x ≤ initial_balance

theorem max_withdrawal : ∃(max_withdrawal_amount : ℕ), 
  max_withdrawal_amount = initial_balance - 498 :=
sorry

end max_withdrawal_l232_232865


namespace two_A_plus_B_l232_232851

theorem two_A_plus_B (A B : ℕ) (h1 : A = Nat.gcd (Nat.gcd 12 18) 30) (h2 : B = Nat.lcm (Nat.lcm 12 18) 30) : 2 * A + B = 192 :=
by
  sorry

end two_A_plus_B_l232_232851


namespace number_of_factors_60_l232_232808

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l232_232808


namespace radar_placement_problem_l232_232331

noncomputable def max_distance (n : ℕ) (coverage_radius : ℝ) (central_angle : ℝ) : ℝ :=
  coverage_radius / Real.sin (central_angle / 2)

noncomputable def ring_area (inner_radius : ℝ) (outer_radius : ℝ) : ℝ :=
  Real.pi * (outer_radius ^ 2 - inner_radius ^ 2)

theorem radar_placement_problem (r : ℝ := 13) (n : ℕ := 5) (width : ℝ := 10) :
  let angle := 2 * Real.pi / n
  let max_dist := max_distance n r angle
  let inner_radius := (r ^ 2 - (r - width) ^ 2) / Real.tan (angle / 2)
  let outer_radius := inner_radius + width
  max_dist = 12 / Real.sin (angle / 2) ∧
  ring_area inner_radius outer_radius = 240 * Real.pi / Real.tan (angle / 2) :=
by
  sorry

end radar_placement_problem_l232_232331


namespace x_eq_1_sufficient_not_necessary_l232_232601

theorem x_eq_1_sufficient_not_necessary (x : ℝ) : 
    (x = 1 → (x^2 - 3 * x + 2 = 0)) ∧ ¬((x^2 - 3 * x + 2 = 0) → (x = 1)) := 
by
  sorry

end x_eq_1_sufficient_not_necessary_l232_232601


namespace total_area_to_be_painted_l232_232569

theorem total_area_to_be_painted (length width height partition_length partition_height : ℝ) 
(partition_along_length inside_outside both_sides : Bool)
(h1 : length = 15)
(h2 : width = 12)
(h3 : height = 6)
(h4 : partition_length = 15)
(h5 : partition_height = 6) 
(h_partition_along_length : partition_along_length = true)
(h_inside_outside : inside_outside = true)
(h_both_sides : both_sides = true) :
    let end_wall_area := 2 * 2 * width * height
    let side_wall_area := 2 * 2 * length * height
    let ceiling_area := length * width
    let partition_area := 2 * partition_length * partition_height
    (end_wall_area + side_wall_area + ceiling_area + partition_area) = 1008 :=
by
    sorry

end total_area_to_be_painted_l232_232569


namespace johnson_oldest_child_age_l232_232199

/-- The average age of the three Johnson children is 10 years. 
    The two younger children are 6 years old and 8 years old. 
    Prove that the age of the oldest child is 16 years. -/
theorem johnson_oldest_child_age :
  ∃ x : ℕ, (6 + 8 + x) / 3 = 10 ∧ x = 16 :=
by
  sorry

end johnson_oldest_child_age_l232_232199


namespace midpoint_is_grid_point_l232_232838

theorem midpoint_is_grid_point
  (P : Fin 5 → (ℤ × ℤ)) :
  ∃ (i j : Fin 5), i ≠ j ∧ (P i).fst % 2 = (P j).fst % 2 ∧ (P i).snd % 2 = (P j).snd % 2 :=
by
  sorry

end midpoint_is_grid_point_l232_232838


namespace equation1_solution_equation2_solution_l232_232064

theorem equation1_solution (x : ℝ) (h : 5 / (x + 1) = 1 / (x - 3)) : x = 4 :=
sorry

theorem equation2_solution (x : ℝ) (h : (2 - x) / (x - 3) + 2 = 1 / (3 - x)) : x = 7 / 3 :=
sorry

end equation1_solution_equation2_solution_l232_232064


namespace find_k_l232_232330

theorem find_k (k : ℕ) :
  (∑' n : ℕ, (5 + n * k) / 5 ^ n) = 12 → k = 90 :=
by
  sorry

end find_k_l232_232330


namespace ratio_of_sums_l232_232782

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

end ratio_of_sums_l232_232782


namespace income_of_A_l232_232564

theorem income_of_A (x y : ℝ) 
    (ratio_income : 5 * x = y * 4)
    (ratio_expenditure : 3 * x = y * 2)
    (savings_A : 5 * x - 3 * y = 1600)
    (savings_B : 4 * x - 2 * y = 1600) : 
    5 * x = 4000 := 
by
  sorry

end income_of_A_l232_232564


namespace ratio_xy_half_l232_232890

noncomputable def common_ratio_k (x y z : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) : ℝ := sorry

theorem ratio_xy_half (x y z k : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) :
  ∃ k, (x + 4) = 2 * k ∧ (y + 9) = k * (z - 3) ∧ (x + 5) = k * (z - 5) → (x / y) = 1 / 2 :=
sorry

end ratio_xy_half_l232_232890


namespace regular_polygon_perimeter_l232_232718

def exterior_angle (n : ℕ) := 360 / n

theorem regular_polygon_perimeter
  (side_length : ℕ)
  (exterior_angle_deg : ℕ)
  (polygon_perimeter : ℕ)
  (h1 : side_length = 8)
  (h2 : exterior_angle_deg = 72)
  (h3 : ∃ n : ℕ, exterior_angle n = exterior_angle_deg)
  (h4 : ∀ n : ℕ, exterior_angle n = exterior_angle_deg → polygon_perimeter = n * side_length) :
  polygon_perimeter = 40 :=
sorry

end regular_polygon_perimeter_l232_232718


namespace complex_number_solution_l232_232790

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i^2 = -1) 
  (h : -i * z = (3 + 2 * i) * (1 - i)) : z = 1 + 5 * i :=
by
  sorry

end complex_number_solution_l232_232790


namespace fraction_of_students_saying_dislike_actually_like_l232_232108

variables (total_students liking_disliking_students saying_disliking_like_students : ℚ)
          (fraction_like_dislike say_dislike : ℚ)
          (cond1 : 0.7 = liking_disliking_students / total_students) 
          (cond2 : 0.3 = (total_students - liking_disliking_students) / total_students)
          (cond3 : 0.3 * liking_disliking_students = saying_disliking_like_students)
          (cond4 : 0.8 * (total_students - liking_disliking_students) 
                    = say_dislike)

theorem fraction_of_students_saying_dislike_actually_like
    (total_students_eq: total_students = 100) : 
    fraction_like_dislike = 46.67 :=
by
  sorry

end fraction_of_students_saying_dislike_actually_like_l232_232108


namespace triangle_area_l232_232511

-- Define the conditions of the problem
variables (a b c : ℝ) (C : ℝ)
axiom cond1 : c^2 = a^2 + b^2 - 2 * a * b + 6
axiom cond2 : C = Real.pi / 3

-- Define the goal
theorem triangle_area : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_area_l232_232511


namespace cannot_be_sum_of_six_consecutive_odds_l232_232223

def is_sum_of_six_consecutive_odds (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (6 * k + 30)

theorem cannot_be_sum_of_six_consecutive_odds :
  ¬ is_sum_of_six_consecutive_odds 198 ∧ ¬ is_sum_of_six_consecutive_odds 390 := 
sorry

end cannot_be_sum_of_six_consecutive_odds_l232_232223


namespace container_could_be_emptied_l232_232212

theorem container_could_be_emptied (a b c : ℕ) (h : 0 ≤ a ∧ a ≤ b ∧ b ≤ c) :
  ∃ (a' b' c' : ℕ), (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
  (∀ x y z : ℕ, (a, b, c) = (x, y, z) → (a', b', c') = (y + y, z - y, x - y)) :=
sorry

end container_could_be_emptied_l232_232212


namespace range_of_numbers_is_six_l232_232252

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l232_232252


namespace find_prob_complement_l232_232611

variable {Ω : Type*} [MeasurableSpace Ω] (P : ProbabilityMeasure Ω)

variable (A B : Set Ω)

noncomputable def problem_conditions :=
P(B) = 0.3 ∧ P(B ∩ A) / P(A) = 0.9 ∧ P(B ∩ Aᶜ) / P(Aᶜ) = 0.2

theorem find_prob_complement (h : problem_conditions P A B) : 
  P(Aᶜ) = 6 / 7 := sorry

end find_prob_complement_l232_232611


namespace profit_value_l232_232657

variable (P : ℝ) -- Total profit made by the business in that year.
variable (MaryInvestment : ℝ) -- Mary's investment
variable (MikeInvestment : ℝ) -- Mike's investment
variable (MaryExtra : ℝ) -- Extra money received by Mary

-- Conditions
axiom mary_investment : MaryInvestment = 900
axiom mike_investment : MikeInvestment = 100
axiom mary_received_more : MaryExtra = 1600
axiom profit_shared_equally : (P / 3) / 2 + (MaryInvestment / (MaryInvestment + MikeInvestment)) * (2 * P / 3) 
                           = MikeInvestment / (MaryInvestment + MikeInvestment) * (2 * P / 3) + MaryExtra

-- Statement
theorem profit_value : P = 4000 :=
by
  sorry

end profit_value_l232_232657


namespace ratio_of_roots_ratio_l232_232604

noncomputable def sum_roots_first_eq (a b c : ℝ) := b / a
noncomputable def product_roots_first_eq (a b c : ℝ) := c / a
noncomputable def sum_roots_second_eq (a b c : ℝ) := a / c
noncomputable def product_roots_second_eq (a b c : ℝ) := b / c

theorem ratio_of_roots_ratio (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (h3 : (b ^ 2 - 4 * a * c) > 0)
  (h4 : (a ^ 2 - 4 * c * b) > 0)
  (h5 : sum_roots_first_eq a b c ≥ 0)
  (h6 : product_roots_first_eq a b c = 9 * sum_roots_second_eq a b c) :
  sum_roots_first_eq a b c / product_roots_second_eq a b c = -3 :=
sorry

end ratio_of_roots_ratio_l232_232604


namespace solution_set_16_sin_pi_x_cos_pi_x_l232_232413

theorem solution_set_16_sin_pi_x_cos_pi_x (x : ℝ) :
  (x = 1 / 4 ∨ x = -1 / 4) ↔ 16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x :=
sorry

end solution_set_16_sin_pi_x_cos_pi_x_l232_232413


namespace gamma_max_two_day_success_ratio_l232_232309

theorem gamma_max_two_day_success_ratio :
  ∃ (e g f h : ℕ), 0 < e ∧ 0 < g ∧
  e + g = 335 ∧ 
  e < f ∧ g < h ∧ 
  f + h = 600 ∧ 
  (e : ℚ) / f < (180 : ℚ) / 360 ∧ 
  (g : ℚ) / h < (150 : ℚ) / 240 ∧ 
  (e + g) / 600 = 67 / 120 :=
by
  sorry

end gamma_max_two_day_success_ratio_l232_232309


namespace price_of_jumbo_pumpkin_l232_232663

theorem price_of_jumbo_pumpkin (total_pumpkins : ℕ) (total_revenue : ℝ)
  (regular_pumpkins : ℕ) (price_regular : ℝ)
  (sold_jumbo_pumpkins : ℕ) (revenue_jumbo : ℝ): 
  total_pumpkins = 80 →
  total_revenue = 395.00 →
  regular_pumpkins = 65 →
  price_regular = 4.00 →
  sold_jumbo_pumpkins = total_pumpkins - regular_pumpkins →
  revenue_jumbo = total_revenue - (price_regular * regular_pumpkins) →
  revenue_jumbo / sold_jumbo_pumpkins = 9.00 :=
by
  intro h_total_pumpkins
  intro h_total_revenue
  intro h_regular_pumpkins
  intro h_price_regular
  intro h_sold_jumbo_pumpkins
  intro h_revenue_jumbo
  sorry

end price_of_jumbo_pumpkin_l232_232663


namespace dandelion_dog_puffs_l232_232940

theorem dandelion_dog_puffs :
  let original_puffs := 40
  let mom_puffs := 3
  let sister_puffs := 3
  let grandmother_puffs := 5
  let friends := 3
  let puffs_per_friend := 9
  original_puffs - (mom_puffs + sister_puffs + grandmother_puffs + friends * puffs_per_friend) = 2 :=
by
  sorry

end dandelion_dog_puffs_l232_232940


namespace range_of_numbers_is_six_l232_232254

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l232_232254


namespace quadratic_roots_l232_232787

-- Definitions based on the conditions provided.
def condition1 (x y : ℝ) : Prop := x^2 - 6 * x + 9 = -(abs (y - 1))

-- The main theorem we want to prove.
theorem quadratic_roots (x y : ℝ) (h : condition1 x y) : (a : ℝ) → (a - 3) * (a - 1) = a^2 - 4 * a + 3 :=
  by sorry

end quadratic_roots_l232_232787


namespace parallel_lines_slope_l232_232801

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 2 = 0 → ax * x - 8 * y - 3 = 0 → a = -6) :=
by
  sorry

end parallel_lines_slope_l232_232801


namespace complex_exponentiation_l232_232595

-- Define the imaginary unit i where i^2 = -1.
def i : ℂ := Complex.I

-- Lean statement for proving the problem.
theorem complex_exponentiation :
  (1 + i)^6 = -8 * i :=
sorry

end complex_exponentiation_l232_232595


namespace jovana_shells_l232_232168

theorem jovana_shells (initial_shells : ℕ) (added_shells : ℕ) (total_shells : ℕ) 
  (h_initial : initial_shells = 5) (h_added : added_shells = 12) :
  total_shells = 17 :=
by
  sorry

end jovana_shells_l232_232168


namespace problem_statement_l232_232649

theorem problem_statement (n : ℤ) (h_odd: Odd n) (h_pos: n > 0) (h_not_divisible_by_3: ¬(3 ∣ n)) : 24 ∣ (n^2 - 1) :=
sorry

end problem_statement_l232_232649


namespace number_of_pairs_x_y_l232_232979

theorem number_of_pairs_x_y (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 85) : 
    (1 : ℕ) + (1 : ℕ) = 2 := 
by 
  sorry

end number_of_pairs_x_y_l232_232979


namespace number_of_factors_of_60_l232_232819

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l232_232819


namespace range_of_a_l232_232025

noncomputable def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 3}

theorem range_of_a (a : ℝ) :
  ((setA a ∩ setB) = setA a) ∧ (∃ x, x ∈ (setA a ∩ setB)) →
  (a < -3 ∨ a > 3) ∧ (a < -1 ∨ a > 1) :=
by sorry

end range_of_a_l232_232025


namespace seats_needed_l232_232707

theorem seats_needed (children seats_per_seat : ℕ) (h1 : children = 58) (h2 : seats_per_seat = 2) : children / seats_per_seat = 29 :=
by sorry

end seats_needed_l232_232707


namespace solution_set_part1_solution_set_part2_l232_232796

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 1

theorem solution_set_part1 :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem solution_set_part2 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end solution_set_part1_solution_set_part2_l232_232796


namespace num_factors_of_60_l232_232828

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l232_232828


namespace squares_difference_l232_232531

theorem squares_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 := by
  sorry

end squares_difference_l232_232531


namespace a_less_than_2_l232_232497

-- Define the quadratic function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x + 2

-- Define the condition that the inequality f(x) - a > 0 has solutions in the interval [0,5]
def inequality_holds (a : ℝ) : Prop := ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → f x - a > 0

-- Theorem stating that a must be less than 2 to satisfy the above condition
theorem a_less_than_2 : ∀ (a : ℝ), (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧ f x - a > 0) → a < 2 := 
sorry

end a_less_than_2_l232_232497


namespace transform_equation_l232_232367

variable (x y : ℝ)

theorem transform_equation 
  (h : y = x + 1/x) 
  (hx : x^4 + x^3 - 5 * x^2 + x + 1 = 0):
  x^2 * (y^2 + y - 7) = 0 := by
  sorry

end transform_equation_l232_232367


namespace part1_part2_l232_232974

theorem part1 (a : ℝ) (x : ℝ) (h : a ≠ 0) :
    (|x - a| + |x + a + (1 / a)|) ≥ 2 * Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) (h : a ≠ 0) (h₁ : |2 - a| + |2 + a + 1 / a| ≤ 3) :
    a ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ico (1/2 : ℝ) 2 :=
sorry

end part1_part2_l232_232974


namespace polynomial_coefficient_sum_l232_232353

theorem polynomial_coefficient_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2 * x - 3) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  sorry

end polynomial_coefficient_sum_l232_232353


namespace no_real_roots_of_quad_eq_l232_232506

theorem no_real_roots_of_quad_eq (k : ℝ) :
  ∀ x : ℝ, ¬ (x^2 - 2*x - k = 0) ↔ k < -1 :=
by sorry

end no_real_roots_of_quad_eq_l232_232506


namespace find_m_l232_232335

theorem find_m (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1/2) : 
  m = 100 := 
by
  sorry

end find_m_l232_232335


namespace two_point_questions_l232_232081

theorem two_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
by
  sorry

end two_point_questions_l232_232081


namespace leak_empties_tank_in_24_hours_l232_232227

theorem leak_empties_tank_in_24_hours (A L : ℝ) (hA : A = 1 / 8) (h_comb : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- Proof will be here
  sorry

end leak_empties_tank_in_24_hours_l232_232227


namespace trapezoid_total_area_l232_232098

/-- 
Given a trapezoid with side lengths 4, 6, 8, and 10, where sides 4 and 8 are used as parallel bases, 
prove that the total area of the trapezoid in all possible configurations is 48√2.
-/
theorem trapezoid_total_area : 
  let a := 4
  let b := 8
  let c := 6
  let d := 10
  let h := 4 * Real.sqrt 2
  let Area := (1 / 2) * (a + b) * h
  (Area + Area) = 48 * Real.sqrt 2 :=
by 
  sorry

end trapezoid_total_area_l232_232098


namespace trig_expr_evaluation_l232_232732

theorem trig_expr_evaluation :
    (1 - (1 / cos (Real.pi / 6))) * 
    (1 + (1 / sin (Real.pi / 3))) * 
    (1 - (1 / sin (Real.pi / 6))) * 
    (1 + (1 / cos (Real.pi / 3))) = 1 := 
by
  let cos_30 := cos (Real.pi / 6)
  let sin_60 := sin (Real.pi / 3)
  let sin_30 := sin (Real.pi / 6)
  let cos_60 := cos (Real.pi / 3)
  have h_cos_30 : cos_30 = (Real.sqrt 3) / 2 := sorry
  have h_sin_60 : sin_60 = (Real.sqrt 3) / 2 := sorry
  have h_sin_30 : sin_30 = 1 / 2 := sorry
  have h_cos_60 : cos_60 = 1 / 2 := sorry
  sorry

end trig_expr_evaluation_l232_232732


namespace num_factors_60_l232_232815

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l232_232815


namespace min_value_at_1_l232_232793

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2 * a * x + 8 else x + 4 / x + 2 * a

theorem min_value_at_1 (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ (a = 5/4 ∨ a = 2 ∨ a = 4) :=
by
  sorry

end min_value_at_1_l232_232793


namespace correct_option_c_l232_232106

variable (a b c : ℝ)

def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom symmetry_axis : -b / (2 * a) = 1

theorem correct_option_c (h : b = -2 * a) : c > 2 * b :=
 sorry

end correct_option_c_l232_232106


namespace deductive_vs_inductive_l232_232551

def is_inductive_reasoning (stmt : String) : Prop :=
  match stmt with
  | "C" => True
  | _ => False

theorem deductive_vs_inductive (A B C D : String) 
  (hA : A = "All trigonometric functions are periodic functions, sin(x) is a trigonometric function, therefore sin(x) is a periodic function.")
  (hB : B = "All odd numbers cannot be divided by 2, 525 is an odd number, therefore 525 cannot be divided by 2.")
  (hC : C = "From 1=1^2, 1+3=2^2, 1+3+5=3^2, it follows that 1+3+…+(2n-1)=n^2 (n ∈ ℕ*)")
  (hD : D = "If two lines are parallel, the corresponding angles are equal. If ∠A and ∠B are corresponding angles of two parallel lines, then ∠A = ∠B.") :
  is_inductive_reasoning C :=
by
  sorry

end deductive_vs_inductive_l232_232551


namespace definite_integral_eval_l232_232467

-- Definitions based on the problem conditions
def first_integral := ∫ x in -2..2, real.sqrt (4 - x^2)
def second_integral := ∫ x in -2..2, -x^2017

-- Main theorem statement
theorem definite_integral_eval :
  (first_integral + second_integral) = 2 * real.pi :=
by sorry

end definite_integral_eval_l232_232467


namespace minimum_value_of_a_squared_plus_b_squared_l232_232348

def quadratic (a b x : ℝ) : ℝ := a * x^2 + (2 * b + 1) * x - a - 2

theorem minimum_value_of_a_squared_plus_b_squared (a b : ℝ) (hab : a ≠ 0)
  (hroot : ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 4 ∧ quadratic a b x = 0) :
  a^2 + b^2 = 1 / 100 :=
sorry

end minimum_value_of_a_squared_plus_b_squared_l232_232348


namespace range_of_set_of_three_numbers_l232_232269

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l232_232269


namespace range_of_set_l232_232295

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l232_232295


namespace find_x_value_l232_232366

-- Let's define the conditions
def equation (x y : ℝ) : Prop := x^2 - 4 * x + y = 0
def y_value : ℝ := 4

-- Define the theorem which states that x = 2 satisfies the conditions
theorem find_x_value (x : ℝ) (h : equation x y_value) : x = 2 :=
by
  sorry

end find_x_value_l232_232366


namespace water_wheel_effective_horsepower_l232_232163

noncomputable def effective_horsepower 
  (velocity : ℝ) (width : ℝ) (thickness : ℝ) (density : ℝ) 
  (diameter : ℝ) (efficiency : ℝ) (g : ℝ) (hp_conversion : ℝ) : ℝ :=
  let mass_flow_rate := velocity * width * thickness * density
  let kinetic_energy_per_second := 0.5 * mass_flow_rate * velocity^2
  let potential_energy_per_second := mass_flow_rate * diameter * g
  let indicated_power := kinetic_energy_per_second + potential_energy_per_second
  let horsepower := indicated_power / hp_conversion
  efficiency * horsepower

theorem water_wheel_effective_horsepower :
  effective_horsepower 1.4 0.5 0.13 1000 3 0.78 9.81 745.7 = 2.9 :=
by
  sorry

end water_wheel_effective_horsepower_l232_232163


namespace solution_l232_232617

-- Define M and N according to the given conditions
def M : Set ℝ := {x | x < 0 ∨ x > 2}
def N : Set ℝ := {x | x ≥ 1}

-- Define the complement of M in Real numbers
def complementM : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define the union of the complement of M and N
def problem_statement : Set ℝ := complementM ∪ N

-- State the theorem
theorem solution :
  problem_statement = { x | x ≥ 0 } :=
by
  sorry

end solution_l232_232617


namespace largest_integer_is_59_l232_232323

theorem largest_integer_is_59 
  {w x y z : ℤ} 
  (h₁ : (w + x + y) / 3 = 32)
  (h₂ : (w + x + z) / 3 = 39)
  (h₃ : (w + y + z) / 3 = 40)
  (h₄ : (x + y + z) / 3 = 44) :
  max (max w x) (max y z) = 59 :=
by {
  sorry
}

end largest_integer_is_59_l232_232323


namespace min_people_liking_both_l232_232555

theorem min_people_liking_both (A B C V : ℕ) (hA : A = 200) (hB : B = 150) (hC : C = 120) (hV : V = 80) :
  ∃ D, D = 80 ∧ D ≤ min B (A - C + V) :=
by {
  sorry
}

end min_people_liking_both_l232_232555


namespace probability_divisible_by_18_l232_232554

theorem probability_divisible_by_18 :
  let cards := {2, 3, 4, 5, 6, 7}
  let total_outcomes := 36
  let favorable_outcomes := { (2, 4), (4, 2), (2, 7), (7, 2), (3, 3), (3, 6), (6, 3), (4, 5), (5, 4) }
  finset.card favorable_outcomes = 9 →
  (9:ℚ) / (36:ℚ) = 1 / 4 :=
by
  sorry

end probability_divisible_by_18_l232_232554


namespace find_c_value_l232_232239

theorem find_c_value (b c : ℝ) 
  (h1 : 1 + b + c = 4) 
  (h2 : 25 + 5 * b + c = 4) : 
  c = 9 :=
by
  sorry

end find_c_value_l232_232239


namespace max_m_value_l232_232779

theorem max_m_value (a b m : ℝ) (ha : a > 0) (hb : b > 0) (H : (3/a + 1/b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
sorry

end max_m_value_l232_232779


namespace number_of_positive_factors_of_60_l232_232816

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l232_232816


namespace sum_of_integers_100_to_110_l232_232902

theorem sum_of_integers_100_to_110 : ∑ i in Finset.range (111 - 100), (100 + i) = 1155 :=
by
  sorry

end sum_of_integers_100_to_110_l232_232902


namespace range_of_set_is_six_l232_232301

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l232_232301


namespace range_of_values_l232_232496

theorem range_of_values (x : ℝ) (h1 : x - 1 ≥ 0) (h2 : x ≠ 0) : x ≥ 1 := 
sorry

end range_of_values_l232_232496


namespace sum_of_values_l232_232133

theorem sum_of_values (N : ℝ) (R : ℝ) (h : N ≠ 0) (h_eq : N + 5 / N = R) : N = R := 
sorry

end sum_of_values_l232_232133


namespace problem_1_problem_2_l232_232968

noncomputable theory

-- Define the function f(x)
def f (a x : ℝ) : ℝ := Real.exp x - a * x

-- Define the interval (-e, -1)
def interval_1 (x : ℝ) : Prop := (-Real.exp 1 < x) ∧ (x < -1)

-- Define the function F(x)
def F (a x : ℝ) : ℝ := f a x - (Real.exp x - 2 * a * x + 2 * Real.log x + a)

-- Define the interval (0, 1/2)
def interval_2 (x : ℝ) : Prop := (0 < x) ∧ (x < 1/2)

-- First problem: show that f(x) is decreasing on (-e, -1) iff a > 1/e
theorem problem_1 (a : ℝ) : (∀ x, interval_1 x → f' a x < 0) ↔ a > (1 / Real.exp 1) :=
sorry

-- Second problem: find maximum value of a such that F(x) has no zero points in (0, 1/2)
theorem problem_2 (a : ℝ) : (∀ x, interval_2 x → F a x ≠ 0) → a ≤ 4 * Real.log 2 :=
sorry

end problem_1_problem_2_l232_232968


namespace inequality_solution_set_l232_232004

theorem inequality_solution_set (x : ℝ) : 4 * x^2 - 4 * x + 1 ≥ 0 := 
by
  sorry

end inequality_solution_set_l232_232004


namespace first_train_takes_4_hours_less_l232_232211

-- Definitions of conditions
def distance: ℝ := 425.80645161290323
def speed_first_train: ℝ := 75
def speed_second_train: ℝ := 44

-- Lean statement to prove the correct answer
theorem first_train_takes_4_hours_less:
  (distance / speed_second_train) - (distance / speed_first_train) = 4 := 
  by
    -- Skip the actual proof
    sorry

end first_train_takes_4_hours_less_l232_232211


namespace stock_market_value_l232_232089

def face_value : ℝ := 100
def dividend_rate : ℝ := 0.05
def yield_rate : ℝ := 0.10

theorem stock_market_value :
  (dividend_rate * face_value / yield_rate = 50) :=
by
  sorry

end stock_market_value_l232_232089


namespace chameleon_increase_l232_232046

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l232_232046


namespace inequality_solution_l232_232425

theorem inequality_solution (x : ℝ) :
  (∀ y : ℝ, (0 < y) → (4 * (x^2 * y^2 + x * y^3 + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y + y)) ↔ (1 < x) :=
by
  sorry

end inequality_solution_l232_232425


namespace range_of_m_l232_232784

theorem range_of_m (m x : ℝ) :
  (m-1 < x ∧ x < m+1) → (2 < x ∧ x < 6) → (3 ≤ m ∧ m ≤ 5) :=
by
  intros hp hq
  sorry

end range_of_m_l232_232784


namespace combine_like_terms_l232_232593

theorem combine_like_terms (a : ℝ) : 2 * a + 3 * a = 5 * a := 
by sorry

end combine_like_terms_l232_232593


namespace find_y_l232_232984

theorem find_y (x y : ℤ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 :=
sorry

end find_y_l232_232984


namespace problem1_problem2_problem3_l232_232372

-- Given conditions for the sequence
axiom pos_seq {a : ℕ → ℝ} : (∀ n : ℕ, 0 < a n)
axiom relation1 {a : ℕ → ℝ} (t : ℝ) : (∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
axiom relation2 {a : ℕ → ℝ} : 2 * (a 3) = (a 2) + (a 4)

-- Proof Requirements

-- (1) Find the value of (a1 + a3) / a2
theorem problem1 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  (a 1 + a 3) / a 2 = 2 :=
sorry

-- (2) Prove that the sequence is an arithmetic sequence
theorem problem2 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  ∀ n : ℕ, a (n+2) - a (n+1) = a (n+1) - a n :=
sorry

-- (3) Show p and r such that (1/a_k), (1/a_p), (1/a_r) form an arithmetic sequence
theorem problem3 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) (k : ℕ) (hk : k ≠ 0) :
  (k = 1 → ∀ p r : ℕ, ¬((k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p))) ∧ 
  (k ≥ 2 → ∃ p r : ℕ, (k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p) ∧ p = 2 * k - 1 ∧ r = k * (2 * k - 1)) :=
sorry

end problem1_problem2_problem3_l232_232372


namespace find_r_in_geometric_sum_l232_232509

theorem find_r_in_geometric_sum (S_n : ℕ → ℕ) (r : ℤ)
  (hSn : ∀ n : ℕ, S_n n = 2 * 3^n + r)
  (hgeo : ∀ n : ℕ, n ≥ 2 → S_n n - S_n (n - 1) = 4 * 3^(n - 1))
  (hn1 : S_n 1 = 6 + r) :
  r = -2 :=
by
  sorry

end find_r_in_geometric_sum_l232_232509


namespace terminating_decimal_multiples_l232_232332

theorem terminating_decimal_multiples :
  (∃ n : ℕ, 20 = n ∧ ∀ m, 1 ≤ m ∧ m ≤ 180 → 
  (∃ k : ℕ, m = 9 * k)) :=
by
  sorry

end terminating_decimal_multiples_l232_232332


namespace sum_floor_sqrt_1_to_25_l232_232742

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l232_232742


namespace isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l232_232043

open Real

theorem isosceles_triangle_of_sine_ratio (a b c : ℝ) (A B C : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h1 : a = b * sin C + c * cos B) :
  C = π / 4 :=
sorry

theorem obtuse_triangle_of_tan_sum_neg (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h_tan_sum : tan A + tan B + tan C < 0) :
  ∃ (E : ℝ), (A = E ∨ B = E ∨ C = E) ∧ π / 2 < E :=
sorry

end isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l232_232043


namespace factors_of_60_l232_232822

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l232_232822


namespace positive_integer_condition_l232_232123

theorem positive_integer_condition (x : ℝ) (hx : x ≠ 0) : 
  (∃ (n : ℤ), n > 0 ∧ (abs (x - abs x + 2) / x) = n) ↔ x = 2 :=
by
  sorry

end positive_integer_condition_l232_232123


namespace area_intersection_A_B_l232_232144

noncomputable def A : Set (Real × Real) := {
  p | ∃ α β : ℝ, p.1 = 2 * Real.sin α + 2 * Real.sin β ∧ p.2 = 2 * Real.cos α + 2 * Real.cos β
}

noncomputable def B : Set (Real × Real) := {
  p | Real.sin (p.1 + p.2) * Real.cos (p.1 + p.2) ≥ 0
}

theorem area_intersection_A_B :
  let intersection := Set.inter A B
  let area : ℝ := 8 * Real.pi
  ∀ (x y : ℝ), (x, y) ∈ intersection → True := sorry

end area_intersection_A_B_l232_232144


namespace number_of_factors_60_l232_232807

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l232_232807


namespace distance_between_centers_of_externally_tangent_circles_l232_232882

noncomputable def external_tangent_distance (R r : ℝ) (hR : R = 2) (hr : r = 3) (tangent : R > 0 ∧ r > 0) : ℝ :=
  R + r

theorem distance_between_centers_of_externally_tangent_circles :
  external_tangent_distance 2 3 (by rfl) (by rfl) (by norm_num) = 5 :=
sorry

end distance_between_centers_of_externally_tangent_circles_l232_232882


namespace black_area_after_transformations_l232_232922

theorem black_area_after_transformations :
  let initial_fraction : ℝ := 1
  let transformation_factor : ℝ := 3 / 4
  let number_of_transformations : ℕ := 5
  let final_fraction : ℝ := transformation_factor ^ number_of_transformations
  final_fraction = 243 / 1024 :=
by
  -- Proof omitted
  sorry

end black_area_after_transformations_l232_232922


namespace even_sum_probability_l232_232422

theorem even_sum_probability :
  let p_even_w1 := 3 / 4
  let p_even_w2 := 1 / 2
  let p_even_w3 := 1 / 4
  let p_odd_w1 := 1 - p_even_w1
  let p_odd_w2 := 1 - p_even_w2
  let p_odd_w3 := 1 - p_even_w3
  (p_even_w1 * p_even_w2 * p_even_w3) +
  (p_odd_w1 * p_odd_w2 * p_even_w3) +
  (p_odd_w1 * p_even_w2 * p_odd_w3) +
  (p_even_w1 * p_odd_w2 * p_odd_w3) = 1 / 2 := by
    sorry

end even_sum_probability_l232_232422


namespace no_way_to_write_as_sum_l232_232640

def can_be_written_as_sum (S : ℕ → ℕ) (n : ℕ) (k : ℕ) : Prop :=
  n + k - 1 + (n - 1) * (k - 1) / 2 = 528 ∧ n > 0 ∧ 2 ∣ n ∧ k > 1

theorem no_way_to_write_as_sum : 
  ∀ (S : ℕ → ℕ) (n k : ℕ), can_be_written_as_sum S n k →
    0 = 0 :=
by
  -- Problem states that there are 0 valid ways to write 528 as the sum
  -- of an increasing sequence of two or more consecutive positive integers
  sorry

end no_way_to_write_as_sum_l232_232640


namespace division_identity_l232_232217

theorem division_identity
  (x y : ℕ)
  (h1 : x = 7)
  (h2 : y = 2)
  : (x^3 + y^3) / (x^2 - x * y + y^2) = 9 :=
by
  sorry

end division_identity_l232_232217


namespace find_a_l232_232944

def E (a b c : ℤ) : ℤ := a * b * b + c

theorem find_a (a : ℤ) : E a 3 1 = E a 5 11 → a = -5 / 8 := 
by sorry

end find_a_l232_232944


namespace sufficient_not_necessary_condition_l232_232131

variable (x y : ℝ)

theorem sufficient_not_necessary_condition (h : x + y ≤ 1) : x ≤ 1/2 ∨ y ≤ 1/2 := 
  sorry

end sufficient_not_necessary_condition_l232_232131


namespace two_workers_two_hours_holes_l232_232678

theorem two_workers_two_hours_holes
    (workers1: ℝ) (holes1: ℝ) (hours1: ℝ)
    (workers2: ℝ) (hours2: ℝ)
    (h1: workers1 = 1.5)
    (h2: holes1 = 1.5)
    (h3: hours1 = 1.5)
    (h4: workers2 = 2)
    (h5: hours2 = 2)
    : (workers2 * (holes1 / (workers1 * hours1)) * hours2 = 8 / 3) := 
by {
   -- To be filled with proof, currently a placeholder.
  sorry
}

end two_workers_two_hours_holes_l232_232678


namespace smallest_b_value_l232_232186

def triangle_inequality (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

def not_triangle (x y z : ℝ) : Prop :=
  ¬triangle_inequality x y z

theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
    (h3 : not_triangle 2 a b) (h4 : not_triangle (1 / b) (1 / a) 1) :
    b >= 2 :=
by
  sorry

end smallest_b_value_l232_232186


namespace f_at_2_l232_232904

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4

-- State the theorem that we need to prove
theorem f_at_2 : f 2 = 2 := by
  -- the proof will go here
  sorry

end f_at_2_l232_232904


namespace Ram_money_l232_232412

theorem Ram_money (R G K : ℕ) (h1 : R = 7 * G / 17) (h2 : G = 7 * K / 17) (h3 : K = 4046) : R = 686 := by
  sorry

end Ram_money_l232_232412


namespace petya_vacation_days_l232_232185

-- Defining the conditions
def total_days : ℕ := 90

def swims (d : ℕ) : Prop := d % 2 = 0
def shops (d : ℕ) : Prop := d % 3 = 0
def solves_math (d : ℕ) : Prop := d % 5 = 0

def does_all (d : ℕ) : Prop := swims d ∧ shops d ∧ solves_math d

def does_any_task (d : ℕ) : Prop := swims d ∨ shops d ∨ solves_math d

-- "Pleasant" days definition: swims, not shops, not solves math
def is_pleasant_day (d : ℕ) : Prop := swims d ∧ ¬shops d ∧ ¬solves_math d
-- "Boring" days definition: does nothing
def is_boring_day (d : ℕ) : Prop := ¬does_any_task d

-- Theorem stating the number of pleasant and boring days
theorem petya_vacation_days :
  (∃ pleasant_days : Finset ℕ, pleasant_days.card = 24 ∧ ∀ d ∈ pleasant_days, is_pleasant_day d)
  ∧ (∃ boring_days : Finset ℕ, boring_days.card = 24 ∧ ∀ d ∈ boring_days, is_boring_day d) :=
by
  sorry

end petya_vacation_days_l232_232185


namespace product_remainder_mod_5_l232_232955

theorem product_remainder_mod_5 :
  (1024 * 1455 * 1776 * 2018 * 2222) % 5 = 0 := 
sorry

end product_remainder_mod_5_l232_232955


namespace area_of_rectangle_l232_232229

noncomputable def area_proof : ℝ :=
  let a := 294
  let b := 147
  let c := 3
  a + b * Real.sqrt c

theorem area_of_rectangle (ABCD : ℝ × ℝ) (E : ℝ) (F : ℝ) (BE : ℝ) (AB' : ℝ) : 
  BE = 21 ∧ BE = 2 * CF → AB' = 7 → 
  (ABCD.1 * ABCD.2 = 294 + 147 * Real.sqrt 3 ∧ (294 + 147 + 3 = 444)) :=
sorry

end area_of_rectangle_l232_232229


namespace calculate_length_X_l232_232035

theorem calculate_length_X 
  (X : ℝ)
  (h1 : 3 + X + 4 = 5 + 7 + X)
  : X = 5 :=
sorry

end calculate_length_X_l232_232035


namespace trig_expression_equality_l232_232731

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end trig_expression_equality_l232_232731


namespace cos_theta_value_sin_theta_plus_pi_over_3_value_l232_232011

variable (θ : ℝ)
variable (H1 : 0 < θ ∧ θ < π / 2)
variable (H2 : Real.sin θ = 4 / 5)

theorem cos_theta_value : Real.cos θ = 3 / 5 := sorry

theorem sin_theta_plus_pi_over_3_value : 
    Real.sin (θ + π / 3) = (4 + 3 * Real.sqrt 3) / 10 := sorry

end cos_theta_value_sin_theta_plus_pi_over_3_value_l232_232011


namespace david_wins_2011th_even_l232_232408

theorem david_wins_2011th_even :
  ∃ n : ℕ, (∃ k : ℕ, k = 2011 ∧ n = 2 * k) ∧ (∀ a b : ℕ, a < b → a + b < b * a) ∧ (n % 2 = 0) := 
sorry

end david_wins_2011th_even_l232_232408


namespace decagon_diagonals_intersect_probability_l232_232568

theorem decagon_diagonals_intersect_probability :
  let n := 10  -- number of vertices in decagon
  let diagonals := n * (n - 3) / 2  -- number of diagonals in decagon
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2  -- ways to choose 2 diagonals from diagonals
  let ways_choose_4 := Nat.choose 10 4  -- ways to choose 4 vertices from 10
  let probability := (4 * ways_choose_4) / pairs_diagonals  -- four vertices chosen determine two intersecting diagonals forming a convex quadrilateral
  probability = (210 / 595) := by
  -- Definitions (diagonals, pairs_diagonals, ways_choose_4) are directly used as hypothesis

  sorry  -- skipping the proof

end decagon_diagonals_intersect_probability_l232_232568


namespace math_problem_l232_232483

open Real

variables {a b c d e f : ℝ}

theorem math_problem 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (hcond : abs (sqrt (a * b) - sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) :=
sorry

end math_problem_l232_232483


namespace exists_factor_between_10_and_20_l232_232628

theorem exists_factor_between_10_and_20 (n : ℕ) : ∃ k, (10 ≤ k ∧ k ≤ 20) ∧ k ∣ (2^n - 1) → k = 17 :=
by
  sorry

end exists_factor_between_10_and_20_l232_232628


namespace std_deviation_calc_l232_232876

theorem std_deviation_calc 
  (μ : ℝ) (σ : ℝ) (V : ℝ) (k : ℝ)
  (hμ : μ = 14.0)
  (hσ : σ = 1.5)
  (hV : V = 11)
  (hk : k = (μ - V) / σ) :
  k = 2 := by
  sorry

end std_deviation_calc_l232_232876


namespace function_neither_odd_nor_even_l232_232795

def f (x : ℝ) : ℝ := x^2 + 6 * x

theorem function_neither_odd_nor_even : 
  ¬ (∀ x, f (-x) = f x) ∧ ¬ (∀ x, f (-x) = -f x) :=
by
  sorry

end function_neither_odd_nor_even_l232_232795


namespace quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l232_232401

theorem quad_eq1_solution (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  sorry

theorem quad_eq2_solution (x : ℝ) : 2 * x^2 - 7 * x + 5 = 0 → x = 5 / 2 ∨ x = 1 :=
by
  sorry

theorem quad_eq3_solution (x : ℝ) : (x + 3)^2 - 2 * (x + 3) = 0 → x = -3 ∨ x = -1 :=
by
  sorry

end quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l232_232401


namespace meals_second_restaurant_l232_232351

theorem meals_second_restaurant (r1 r2 r3 total_weekly_meals : ℕ) 
    (H1 : r1 = 20) 
    (H3 : r3 = 50) 
    (H_total : total_weekly_meals = 770) : 
    (7 * r2) = 280 := 
by 
    sorry

example (r2 : ℕ) : (40 = r2) :=
    by sorry

end meals_second_restaurant_l232_232351


namespace total_amount_correct_l232_232654

-- Define the prices of jeans and tees
def price_jean : ℕ := 11
def price_tee : ℕ := 8

-- Define the quantities sold
def quantity_jeans_sold : ℕ := 4
def quantity_tees_sold : ℕ := 7

-- Calculate the total amount earned
def total_amount : ℕ := (price_jean * quantity_jeans_sold) + (price_tee * quantity_tees_sold)

-- Now, we state and prove the theorem
theorem total_amount_correct : total_amount = 100 :=
by
  -- Here we assert the correctness of the calculation
  sorry

end total_amount_correct_l232_232654


namespace sum_floor_sqrt_1_to_25_l232_232740

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l232_232740


namespace correct_factorization_option_A_l232_232562

variable (x y : ℝ)

theorem correct_factorization_option_A :
  (2 * x^2 + 3 * x + 1 = (2 * x + 1) * (x + 1)) :=
by {
  sorry
}

end correct_factorization_option_A_l232_232562


namespace product_xyz_equals_one_l232_232962

theorem product_xyz_equals_one (x y z : ℝ) (h1 : x + (1/y) = 2) (h2 : y + (1/z) = 2) : x * y * z = 1 := 
by
  sorry

end product_xyz_equals_one_l232_232962


namespace range_of_set_l232_232294

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l232_232294


namespace distance_between_parabola_vertices_l232_232458

theorem distance_between_parabola_vertices : 
  ∀ x y : ℝ, (sqrt (x^2 + y^2) + abs (y - 2) = 4) →
    |(6 - max y 2) - (-(2 - min y 2))| = 4 :=
by
  intros x y h_eqn
  sorry

end distance_between_parabola_vertices_l232_232458


namespace complement_union_l232_232027

theorem complement_union (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {2, 3, 5})
  (hN : N = {4, 5}) :
  U \ (M ∪ N) = {1, 6} :=
by
  sorry

end complement_union_l232_232027


namespace geom_seq_common_ratio_l232_232201

noncomputable def log_custom_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem geom_seq_common_ratio (a : ℝ) :
  let u₁ := a + log_custom_base 2 3
  let u₂ := a + log_custom_base 4 3
  let u₃ := a + log_custom_base 8 3
  u₂ / u₁ = u₃ / u₂ →
  u₂ / u₁ = 1 / 3 :=
by
  intro h
  sorry

end geom_seq_common_ratio_l232_232201


namespace correct_multiplication_l232_232494

theorem correct_multiplication (n : ℕ) (h₁ : 15 * n = 45) : 5 * n = 15 :=
by
  -- skipping the proof
  sorry

end correct_multiplication_l232_232494


namespace walk_fraction_correct_l232_232310

def bus_fraction := 1/3
def automobile_fraction := 1/5
def bicycle_fraction := 1/8
def metro_fraction := 1/15

def total_transport_fraction := bus_fraction + automobile_fraction + bicycle_fraction + metro_fraction

def walk_fraction := 1 - total_transport_fraction

theorem walk_fraction_correct : walk_fraction = 11/40 := by
  sorry

end walk_fraction_correct_l232_232310


namespace not_divisible_59_l232_232863

theorem not_divisible_59 (x y : ℕ) (hx : ¬ (59 ∣ x)) (hy : ¬ (59 ∣ y)) 
  (h : (3 * x + 28 * y) % 59 = 0) : (5 * x + 16 * y) % 59 ≠ 0 :=
by
  sorry

end not_divisible_59_l232_232863


namespace angie_age_problem_l232_232685

theorem angie_age_problem (a certain_number : ℕ) 
  (h1 : 2 * 8 + certain_number = 20) : 
  certain_number = 4 :=
by 
  sorry

end angie_age_problem_l232_232685


namespace quadratic_behavior_l232_232010

theorem quadratic_behavior (x : ℝ) : x < 3 → ∃ y : ℝ, y = 5 * (x - 3) ^ 2 + 2 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 3 ∧ x2 < 3 → (5 * (x1 - 3) ^ 2 + 2) > (5 * (x2 - 3) ^ 2 + 2) := 
by
  sorry

end quadratic_behavior_l232_232010


namespace calculate_difference_square_l232_232360

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l232_232360


namespace Alyosha_result_divisible_by_S_l232_232207

variable (a b S x y : ℤ)
variable (h1 : x + y = S)
variable (h2 : S ∣ a * x + b * y)

theorem Alyosha_result_divisible_by_S :
  S ∣ b * x + a * y :=
sorry

end Alyosha_result_divisible_by_S_l232_232207


namespace investment_three_years_ago_l232_232684

noncomputable def initial_investment (final_amount : ℝ) : ℝ :=
  final_amount / (1.08 ^ 3)

theorem investment_three_years_ago :
  abs (initial_investment 439.23 - 348.68) < 0.01 :=
by
  sorry

end investment_three_years_ago_l232_232684


namespace sum_equidistant_terms_l232_232996

def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n m : ℕ, (n < m) → a (n+1) - a n = a (m+1) - a m

variable {a : ℕ → ℤ}

theorem sum_equidistant_terms (h_seq : is_arithmetic_sequence a)
  (h_4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end sum_equidistant_terms_l232_232996


namespace geometric_sequence_third_term_l232_232798

theorem geometric_sequence_third_term (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h1 : a 1 * a 5 = 16) :
  a 3 = 4 ∨ a 3 = -4 := 
sorry

end geometric_sequence_third_term_l232_232798


namespace problem_water_percentage_l232_232919

noncomputable def percentage_water_in_mixture 
  (volA volB volC volD : ℕ) 
  (pctA pctB pctC pctD : ℝ) : ℝ :=
  let total_volume := volA + volB + volC + volD
  let total_solution := volA * pctA + volB * pctB + volC * pctC + volD * pctD
  let total_water := total_volume - total_solution
  (total_water / total_volume) * 100

theorem problem_water_percentage :
  percentage_water_in_mixture 100 90 60 50 0.25 0.3 0.4 0.2 = 71.33 :=
by
  -- proof goes here
  sorry

end problem_water_percentage_l232_232919


namespace tangent_line_m_value_l232_232547

theorem tangent_line_m_value : 
  (∀ m : ℝ, ∃ (x y : ℝ), (x = my + 2) ∧ (x + one)^2 + (y + one)^2 = 2) → 
  (m = 1 ∨ m = -7) :=
  sorry

end tangent_line_m_value_l232_232547


namespace solve_for_b_l232_232895

theorem solve_for_b (a b : ℤ) (h1 : 3 * a + 2 = 5) (h2 : b - 4 * a = 2) : b = 6 :=
by
  -- proof goes here
  sorry

end solve_for_b_l232_232895


namespace train_length_proof_l232_232915

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (5 / 18) -- convert to m/s
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_proof (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) :
  speed1 = 120 →
  speed2 = 80 →
  time = 9 →
  length2 = 270.04 →
  length_of_first_train speed1 speed2 time length2 = 230 :=
by
  intros h1 h2 h3 h4
  -- Use the defined function and simplify
  rw [h1, h2, h3, h4]
  simp [length_of_first_train]
  sorry

end train_length_proof_l232_232915


namespace normal_line_eq_l232_232954

variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem normal_line_eq (x_0 : ℝ) (h : x_0 = 1) :
  ∃ y_0 : ℝ, y_0 = f x_0 ∧ 
  ∀ x y : ℝ, y = -(x - 1) + y_0 ↔ f 1 = 0 ∧ y = -x + 1 :=
by
  sorry

end normal_line_eq_l232_232954


namespace combination_of_10_choose_3_l232_232516

theorem combination_of_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end combination_of_10_choose_3_l232_232516


namespace value_two_std_dev_less_l232_232540

noncomputable def mean : ℝ := 15.5
noncomputable def std_dev : ℝ := 1.5

theorem value_two_std_dev_less : mean - 2 * std_dev = 12.5 := by
  sorry

end value_two_std_dev_less_l232_232540


namespace gasoline_tank_capacity_l232_232236

theorem gasoline_tank_capacity (x : ℝ)
  (h1 : (7 / 8) * x - (1 / 2) * x = 12) : x = 32 := 
sorry

end gasoline_tank_capacity_l232_232236


namespace trig_expression_equality_l232_232729

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end trig_expression_equality_l232_232729


namespace floor_sum_sqrt_1_to_25_l232_232752

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l232_232752


namespace quadratic_no_real_roots_l232_232503

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x - k = 0)) ↔ k < -1 :=
by sorry

end quadratic_no_real_roots_l232_232503


namespace Saheed_earnings_l232_232062

theorem Saheed_earnings (Vika_earnings : ℕ) (Kayla_earnings : ℕ) (Saheed_earnings : ℕ)
  (h1 : Vika_earnings = 84) (h2 : Kayla_earnings = Vika_earnings - 30) (h3 : Saheed_earnings = 4 * Kayla_earnings) :
  Saheed_earnings = 216 := 
by
  sorry

end Saheed_earnings_l232_232062


namespace sale_in_fifth_month_l232_232092

-- Define the sale amounts and average sale required.
def sale_first_month : ℕ := 7435
def sale_second_month : ℕ := 7920
def sale_third_month : ℕ := 7855
def sale_fourth_month : ℕ := 8230
def sale_sixth_month : ℕ := 6000
def average_sale_required : ℕ := 7500

-- State the theorem to determine the sale in the fifth month.
theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 avg : ℕ)
  (h1 : s1 = sale_first_month)
  (h2 : s2 = sale_second_month)
  (h3 : s3 = sale_third_month)
  (h4 : s4 = sale_fourth_month)
  (h6 : s6 = sale_sixth_month)
  (havg : avg = average_sale_required) :
  s1 + s2 + s3 + s4 + s6 + x = 6 * avg →
  x = 7560 :=
by
  sorry

end sale_in_fifth_month_l232_232092


namespace batsman_average_after_11th_inning_l232_232570

theorem batsman_average_after_11th_inning (x : ℝ) (h : 10 * x + 110 = 11 * (x + 5)) : 
    (10 * x + 110) / 11 = 60 := by
  sorry

end batsman_average_after_11th_inning_l232_232570


namespace existence_of_committees_l232_232088

noncomputable def committeesExist : Prop :=
∃ (C : Fin 1990 → Fin 11 → Fin 3), 
  (∀ i j, i ≠ j → C i ≠ C j) ∧
  (∀ i j, i = j + 1 ∨ (i = 0 ∧ j = 1990 - 1) → ∃ k, C i k = C j k)

theorem existence_of_committees : committeesExist :=
sorry

end existence_of_committees_l232_232088


namespace percentage_supports_policy_l232_232924

theorem percentage_supports_policy (men women : ℕ) (men_favor women_favor : ℝ) (total_population : ℕ) (total_supporters : ℕ) (percentage_supporters : ℝ)
  (h1 : men = 200) 
  (h2 : women = 800)
  (h3 : men_favor = 0.70)
  (h4 : women_favor = 0.75)
  (h5 : total_population = men + women)
  (h6 : total_supporters = (men_favor * men) + (women_favor * women))
  (h7 : percentage_supporters = (total_supporters / total_population) * 100) :
  percentage_supporters = 74 := 
by
  sorry

end percentage_supports_policy_l232_232924


namespace line_intersects_circle_and_angle_conditions_l232_232342

noncomputable def line_circle_intersection_condition (k : ℝ) : Prop :=
  - (Real.sqrt 3) / 3 ≤ k ∧ k ≤ (Real.sqrt 3) / 3

noncomputable def inclination_angle_condition (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

theorem line_intersects_circle_and_angle_conditions (k θ : ℝ) :
  line_circle_intersection_condition k →
  inclination_angle_condition θ →
  ∃ x y : ℝ, (y = k * (x + 1)) ∧ ((x - 1)^2 + y^2 = 1) :=
by
  sorry

end line_intersects_circle_and_angle_conditions_l232_232342


namespace functions_are_identical_l232_232908

def f1 (x : ℝ) : ℝ := 1
def f2 (x : ℝ) : ℝ := x^0

theorem functions_are_identical : ∀ (x : ℝ), f1 x = f2 x :=
by
  intro x
  simp [f1, f2]
  sorry

end functions_are_identical_l232_232908


namespace num_factors_of_60_l232_232806

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l232_232806


namespace greatest_common_multiple_less_than_bound_l232_232688

-- Define the numbers and the bound
def num1 : ℕ := 15
def num2 : ℕ := 10
def bound : ℕ := 150

-- Define the LCM of num1 and num2
def lcm_num1_num2 : ℕ := Nat.lcm num1 num2

-- Define the greatest multiple of LCM less than bound
def greatest_multiple_less_than_bound (lcm : ℕ) (b : ℕ) : ℕ :=
  (b / lcm) * lcm

-- Main theorem
theorem greatest_common_multiple_less_than_bound :
  greatest_multiple_less_than_bound lcm_num1_num2 bound = 120 :=
by
  sorry

end greatest_common_multiple_less_than_bound_l232_232688


namespace amount_of_flour_per_new_bread_roll_l232_232421

theorem amount_of_flour_per_new_bread_roll :
  (24 * (1 / 8) = 3) → (16 * f = 3) → (f = 3 / 16) :=
by
  intro h1 h2
  sorry

end amount_of_flour_per_new_bread_roll_l232_232421


namespace johns_apartment_number_l232_232999

theorem johns_apartment_number (car_reg : Nat) (apartment_num : Nat) 
  (h_car_reg_sum : car_reg = 834205) 
  (h_car_digits : (8 + 3 + 4 + 2 + 0 + 5 = 22)) 
  (h_apartment_digits : ∃ (d1 d2 d3 : Nat), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ d1 + d2 + d3 = 22) :
  apartment_num = 985 :=
by
  sorry

end johns_apartment_number_l232_232999


namespace total_cookies_after_three_days_l232_232431

-- Define the initial conditions
def cookies_baked_monday : ℕ := 32
def cookies_baked_tuesday : ℕ := cookies_baked_monday / 2
def cookies_baked_wednesday_before : ℕ := cookies_baked_tuesday * 3
def brother_ate : ℕ := 4

-- Define the total cookies before brother ate any
def total_cookies_before : ℕ := cookies_baked_monday + cookies_baked_tuesday + cookies_baked_wednesday_before

-- Define the total cookies after brother ate some
def total_cookies_after : ℕ := total_cookies_before - brother_ate

-- The proof statement
theorem total_cookies_after_three_days : total_cookies_after = 92 := by
  -- Here, we would provide the proof, but we add sorry for now to compile successfully.
  sorry

end total_cookies_after_three_days_l232_232431


namespace song_book_cost_correct_l232_232178

noncomputable def cost_of_trumpet : ℝ := 145.16
noncomputable def total_spent : ℝ := 151.00
noncomputable def cost_of_song_book : ℝ := total_spent - cost_of_trumpet

theorem song_book_cost_correct : cost_of_song_book = 5.84 :=
  by
    sorry

end song_book_cost_correct_l232_232178


namespace smallest_n_for_doubling_sum_l232_232193

theorem smallest_n_for_doubling_sum :
  let D (a n : ℕ) := a * (2^n - 1)
  ∃ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 6 → ∃ a_i : ℕ, D a_i i = n) ∧ n = Nat.lcm (Nat.repeat (fun i : ℕ => 2^i - 1) 6) :=
sorry

end smallest_n_for_doubling_sum_l232_232193


namespace find_initial_amount_l232_232980

noncomputable def initial_amount (diff : ℝ) : ℝ :=
  diff / (1.4641 - 1.44)

theorem find_initial_amount
  (diff : ℝ)
  (h : diff = 964.0000000000146) :
  initial_amount diff = 40000 :=
by
  -- the steps to prove this can be added here later
  sorry

end find_initial_amount_l232_232980


namespace reflected_line_equation_l232_232543

def line_reflection_about_x_axis (x y : ℝ) : Prop :=
  x - y + 1 = 0 → y = -x - 1

theorem reflected_line_equation :
  ∀ (x y : ℝ), x - y + 1 = 0 → x + y + 1 = 0 :=
by
  intros x y h
  suffices y = -x - 1 by
    linarith
  sorry

end reflected_line_equation_l232_232543


namespace smallest_square_area_l232_232008

theorem smallest_square_area :
  (∀ (x y : ℝ), (∃ (x1 x2 y1 y2 : ℝ), y1 = 3 * x1 - 4 ∧ y2 = 3 * x2 - 4 ∧ y = x^2 + 5 ∧ 
  ∀ (k : ℝ), x1 + x2 = 3 ∧ x1 * x2 = 5 - k ∧ 16 * k^2 - 332 * k + 396 = 0 ∧ 
  ((k = 1.5 ∧ 10 * (4 * k - 11) = 50) ∨ 
  (k = 16.5 ∧ 10 * (4 * k - 11) ≠ 50))) → 
  ∃ (A: Real), A = 50) :=
sorry

end smallest_square_area_l232_232008


namespace range_of_set_l232_232263

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l232_232263


namespace fraction_simplification_l232_232983

variable (a b x : ℝ)
variable (h1 : x = a / b)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : a = b * x ^ 2)

theorem fraction_simplification : (a + b) / (a - b) = (x ^ 2 + 1) / (x ^ 2 - 1) := by
  sorry

end fraction_simplification_l232_232983


namespace find_AD_l232_232642

theorem find_AD
  (A B C D : Type)
  (BD BC CD AD : ℝ)
  (hBD : BD = 21)
  (hBC : BC = 30)
  (hCD : CD = 15)
  (hAngleBisect : true) -- Encode that D bisects the angle at C internally
  : AD = 35 := by
  sorry

end find_AD_l232_232642


namespace problem_solution_l232_232355

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l232_232355


namespace max_product_decomposition_l232_232461

theorem max_product_decomposition : ∃ x y : ℝ, x + y = 100 ∧ x * y = 50 * 50 := by
  sorry

end max_product_decomposition_l232_232461


namespace max_consecutive_integers_sum_l232_232897

theorem max_consecutive_integers_sum:
  ∃ k, ∀ n: ℕ, 3 + ∑ i in (range (n - 2)), (3 + i) ≤ 500 → k = 29 := by
sorry

end max_consecutive_integers_sum_l232_232897


namespace largest_measureable_quantity_is_1_l232_232914

theorem largest_measureable_quantity_is_1 : 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd 496 403) 713) 824) 1171 = 1 :=
  sorry

end largest_measureable_quantity_is_1_l232_232914


namespace mean_combined_l232_232548

-- Definitions for the two sets and their properties
def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

variables (set₁ set₂ : List ℕ)
-- Conditions based on the problem
axiom h₁ : set₁.length = 7
axiom h₂ : mean set₁ = 15
axiom h₃ : set₂.length = 8
axiom h₄ : mean set₂ = 30

-- Prove that the mean of the combined set is 23
theorem mean_combined (h₁ : set₁.length = 7) (h₂ : mean set₁ = 15)
  (h₃ : set₂.length = 8) (h₄ : mean set₂ = 30) : mean (set₁ ++ set₂) = 23 := 
sorry

end mean_combined_l232_232548


namespace reflect_A_across_x_axis_l232_232542

-- Define the point A
def A : ℝ × ℝ := (-3, 2)

-- Define the reflection function across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Theorem statement: The reflection of point A across the x-axis should be (-3, -2)
theorem reflect_A_across_x_axis : reflect_x A = (-3, -2) := by
  sorry

end reflect_A_across_x_axis_l232_232542


namespace number_of_factors_60_l232_232809

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l232_232809


namespace total_points_needed_l232_232415

def num_students : ℕ := 25
def num_weeks : ℕ := 2
def vegetables_per_student_per_week : ℕ := 2
def points_per_vegetable : ℕ := 2

theorem total_points_needed : 
  (num_students * (vegetables_per_student_per_week * num_weeks) * points_per_vegetable) = 200 := by
  sorry

end total_points_needed_l232_232415


namespace quadratic_no_real_roots_l232_232501

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x - k = 0)) ↔ k < -1 :=
by sorry

end quadratic_no_real_roots_l232_232501


namespace smallest_positive_period_monotonically_decreasing_range_of_f_on_interval_l232_232143

noncomputable def f (x : ℝ) : ℝ := 2 * real.sqrt 3 * real.sin x * real.cos x - 2 * real.cos x ^ 2

theorem smallest_positive_period_monotonically_decreasing :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ k : ℤ, ∀ x ∈ set.Icc (k * π + π / 3) (k * π + 5 * π / 6), isDecreasing (f x)) :=
sorry

theorem range_of_f_on_interval :
  set.image f (set.Icc 0 (π / 2)) = set.Icc (-2 : ℝ) 1 :=
sorry

end smallest_positive_period_monotonically_decreasing_range_of_f_on_interval_l232_232143


namespace decimal_equivalent_of_one_quarter_l232_232565

theorem decimal_equivalent_of_one_quarter:
  ( (1:ℚ) / (4:ℚ) )^1 = 0.25 := 
sorry

end decimal_equivalent_of_one_quarter_l232_232565


namespace Sadie_l232_232535

theorem Sadie's_homework_problems (T : ℝ) 
  (h1 : 0.40 * T = A) 
  (h2 : 0.5 * A = 28) 
  : T = 140 := 
by
  sorry

end Sadie_l232_232535


namespace least_value_of_b_l232_232836

variable {x y b : ℝ}

noncomputable def condition_inequality (x y b : ℝ) : Prop :=
  (x^2 + y^2)^2 ≤ b * (x^4 + y^4)

theorem least_value_of_b (h : ∀ x y : ℝ, condition_inequality x y b) : b ≥ 2 := 
sorry

end least_value_of_b_l232_232836


namespace circle_condition_k_l232_232877

theorem circle_condition_k (k : ℝ) : 
  (∃ (h : ℝ), (x^2 + y^2 - 2*x + 6*y + k = 0)) → k < 10 :=
by
  sorry

end circle_condition_k_l232_232877


namespace rohit_distance_from_start_l232_232397

-- Define Rohit's movements
def rohit_walked_south (d: ℕ) : ℕ := d
def rohit_turned_left_walked_east (d: ℕ) : ℕ := d
def rohit_turned_left_walked_north (d: ℕ) : ℕ := d
def rohit_turned_right_walked_east (d: ℕ) : ℕ := d

-- Rohit's total movement in east direction
def total_distance_moved_east (d1 d2 : ℕ) : ℕ :=
  rohit_turned_left_walked_east d1 + rohit_turned_right_walked_east d2

-- Prove the distance from the starting point is 35 meters
theorem rohit_distance_from_start : 
  total_distance_moved_east 20 15 = 35 :=
by
  sorry

end rohit_distance_from_start_l232_232397


namespace saline_solution_mixture_l232_232803

theorem saline_solution_mixture 
  (x : ℝ) 
  (h₁ : 20 + 0.1 * x = 0.25 * (50 + x)) 
  : x = 50 := 
by 
  sorry

end saline_solution_mixture_l232_232803


namespace total_right_handed_players_l232_232532

theorem total_right_handed_players
  (total_players throwers mp_players non_throwers L R : ℕ)
  (ratio_L_R : 2 * R = 3 * L)
  (total_eq : total_players = 120)
  (throwers_eq : throwers = 60)
  (mp_eq : mp_players = 20)
  (non_throwers_eq : non_throwers = total_players - throwers - mp_players)
  (non_thrower_sum_eq : L + R = non_throwers) :
  (throwers + mp_players + R = 104) :=
by
  sorry

end total_right_handed_players_l232_232532


namespace B_subset_A_l232_232045

def A (x : ℝ) : Prop := abs (2 * x - 3) > 1
def B (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem B_subset_A : ∀ x, B x → A x := sorry

end B_subset_A_l232_232045


namespace sum_quotient_dividend_divisor_l232_232631

theorem sum_quotient_dividend_divisor (D : ℕ) (d : ℕ) (Q : ℕ) 
  (h1 : D = 54) (h2 : d = 9) (h3 : D = Q * d) : 
  (Q + D + d) = 69 :=
by
  sorry

end sum_quotient_dividend_divisor_l232_232631


namespace kyle_age_l232_232399

theorem kyle_age :
  ∃ (kyle shelley julian frederick tyson casey : ℕ),
    shelley = kyle - 3 ∧ 
    shelley = julian + 4 ∧
    julian = frederick - 20 ∧
    frederick = 2 * tyson ∧
    tyson = 2 * casey ∧
    casey = 15 ∧ 
    kyle = 47 :=
by
  sorry

end kyle_age_l232_232399


namespace factorize_expression_l232_232470

theorem factorize_expression (m : ℝ) : 
  4 * m^2 - 64 = 4 * (m + 4) * (m - 4) :=
sorry

end factorize_expression_l232_232470


namespace relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l232_232442

variable (x y : ℝ)
variable (h1 : 2 * (x + y) = 18)
variable (h2 : x * y = 18)
variable (h3 : x > 0) (h4 : y > 0) (h5 : x > y)
variable (h6 : x * y = 21)

theorem relationship_and_range : (y = 9 - x ∧ 0 < x ∧ x < 9) :=
by sorry

theorem dimensions_when_area_18 :
  (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) :=
by sorry

theorem impossibility_of_area_21 :
  ¬(∃ x y, x * y = 21 ∧ 2 * (x + y) = 18 ∧ x > y) :=
by sorry

end relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l232_232442


namespace downstream_speed_l232_232918

-- Define the given conditions
def V_m : ℝ := 40 -- speed of the man in still water in kmph
def V_up : ℝ := 32 -- speed of the man upstream in kmph

-- Question to be proved as a statement
theorem downstream_speed : 
  ∃ (V_c V_down : ℝ), V_c = V_m - V_up ∧ V_down = V_m + V_c ∧ V_down = 48 :=
by
  -- Provide statement without proof as specified
  sorry

end downstream_speed_l232_232918


namespace wage_increase_percentage_l232_232928

theorem wage_increase_percentage (new_wage old_wage : ℝ) (h1 : new_wage = 35) (h2 : old_wage = 25) : 
  ((new_wage - old_wage) / old_wage) * 100 = 40 := 
by
  sorry

end wage_increase_percentage_l232_232928


namespace union_of_A_and_B_l232_232853

section
variable {A B : Set ℝ}
variable (a b : ℝ)

def setA := {x : ℝ | x^2 - 3 * x + a = 0}
def setB := {x : ℝ | x^2 + b = 0}

theorem union_of_A_and_B:
  setA a ∩ setB b = {2} →
  setA a ∪ setB b = ({-2, 1, 2} : Set ℝ) := by
  sorry
end

end union_of_A_and_B_l232_232853


namespace ratio_part_to_third_fraction_l232_232182

variable (P N : ℕ)

-- Definitions based on conditions
def one_fourth_one_third_P_eq_14 : Prop := (1/4 : ℚ) * (1/3 : ℚ) * (P : ℚ) = 14

def forty_percent_N_eq_168 : Prop := (40/100 : ℚ) * (N : ℚ) = 168

-- Theorem stating the required ratio
theorem ratio_part_to_third_fraction (h1 : one_fourth_one_third_P_eq_14 P) (h2 : forty_percent_N_eq_168 N) : 
  (P : ℚ) / ((1/3 : ℚ) * (N : ℚ)) = 6 / 5 := by
  sorry

end ratio_part_to_third_fraction_l232_232182


namespace max_students_in_auditorium_l232_232725

def increment (i : ℕ) : ℕ :=
  (i * (i + 1)) / 2

def seats_in_row (i : ℕ) : ℕ :=
  10 + increment i

def max_students_in_row (n : ℕ) : ℕ :=
  (n + 1) / 2

def total_max_students_up_to_row (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => max_students_in_row (seats_in_row (i + 1)))

theorem max_students_in_auditorium : total_max_students_up_to_row 20 = 335 := 
sorry

end max_students_in_auditorium_l232_232725


namespace range_of_set_l232_232246

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l232_232246


namespace problem_statement_l232_232387

def f (x : ℕ) : ℕ := x^2 + x + 4
def g (x : ℕ) : ℕ := 3 * x^3 + 2

theorem problem_statement : g (f 3) = 12290 := by
  sorry

end problem_statement_l232_232387


namespace no_solutions_for_divisibility_by_3_l232_232472

theorem no_solutions_for_divisibility_by_3 (x y : ℤ) : ¬ (x^2 + y^2 + x + y ∣ 3) :=
sorry

end no_solutions_for_divisibility_by_3_l232_232472


namespace triangle_side_length_difference_l232_232926

theorem triangle_side_length_difference :
  (∃ x : ℤ, 3 ≤ x ∧ x ≤ 17 ∧ ∀ a b c : ℤ, x + 8 > 10 ∧ x + 10 > 8 ∧ 8 + 10 > x) →
  (17 - 3 = 14) :=
by
  intros
  sorry

end triangle_side_length_difference_l232_232926


namespace gigi_additional_batches_l232_232126

-- Define the initial amount of flour in cups
def initialFlour : Nat := 20

-- Define the amount of flour required per batch in cups
def flourPerBatch : Nat := 2

-- Define the number of batches already baked
def batchesBaked : Nat := 3

-- Define the remaining flour
def remainingFlour : Nat := initialFlour - (batchesBaked * flourPerBatch)

-- Define the additional batches Gigi can make with the remaining flour
def additionalBatches : Nat := remainingFlour / flourPerBatch

-- Prove that with the given conditions, the additional batches Gigi can make is 7
theorem gigi_additional_batches : additionalBatches = 7 := by
  -- Calculate the remaining cups of flour after baking
  have h1 : remainingFlour = 20 - (3 * 2) := by rfl

  -- Calculate the additional batches of cookies Gigi can make
  have h2 : additionalBatches = h1 / 2 := by rfl

  -- Solve for the additional batches
  show additionalBatches = 7 from
    calc
      additionalBatches = (initialFlour - (batchesBaked * flourPerBatch)) / flourPerBatch : by rfl
      ...               = (20 - 6) / 2                               : by rw h1
      ...               = 14 / 2                                     : by rfl
      ...               = 7                                          : by rfl

end gigi_additional_batches_l232_232126


namespace Yeonseo_skirts_l232_232909

theorem Yeonseo_skirts
  (P : ℕ)
  (more_than_two_skirts : ∀ S : ℕ, S > 2)
  (more_than_two_pants : P > 2)
  (ways_to_choose : P + 3 = 7) :
  ∃ S : ℕ, S = 3 := by
  sorry

end Yeonseo_skirts_l232_232909


namespace combined_weight_l232_232529

theorem combined_weight (x y z : ℕ) (h1 : x + y = 110) (h2 : y + z = 130) (h3 : z + x = 150) : x + y + z = 195 :=
by
  sorry

end combined_weight_l232_232529


namespace sum_abs_arithmetic_sequence_l232_232343

variable (n : ℕ)

def S_n (n : ℕ) : ℚ :=
  - ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n

def T_n (n : ℕ) : ℚ :=
  if n ≤ 34 then
    -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
  else
    ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502

theorem sum_abs_arithmetic_sequence :
  T_n n = (if n ≤ 34 then -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
           else ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502) :=
by sorry

end sum_abs_arithmetic_sequence_l232_232343


namespace krystian_total_books_borrowed_l232_232381

/-
Conditions:
1. Krystian starts on Monday by borrowing 40 books.
2. Each day from Tuesday to Thursday, he borrows 5% more books than he did the previous day.
3. On Friday, his number of borrowed books is 40% higher than on Thursday.
4. During weekends, Krystian borrows books for his friends, and he borrows 2 additional books for every 10 books borrowed during the weekdays.

Theorem: Given these conditions, Krystian borrows a total of 283 books from Monday to Sunday.
-/
theorem krystian_total_books_borrowed : 
  let mon := 40
  let tue := mon + (5 * mon / 100)
  let wed := tue + (5 * tue / 100)
  let thu := wed + (5 * wed / 100)
  let fri := thu + (40 * thu / 100)
  let weekday_total := mon + tue + wed + thu + fri
  let weekend := 2 * (weekday_total / 10)
  weekday_total + weekend = 283 := 
by
  sorry

end krystian_total_books_borrowed_l232_232381


namespace sum_floor_sqrt_l232_232755

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l232_232755


namespace error_estimate_alternating_series_l232_232466

theorem error_estimate_alternating_series :
  let S := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4) + (-(1 / 5)) 
  let S₄ := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4)
  ∃ ΔS : ℝ, ΔS = |-(1 / 5)| ∧ ΔS < 0.2 := by
  sorry

end error_estimate_alternating_series_l232_232466


namespace range_of_numbers_is_six_l232_232274

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l232_232274


namespace contrapositive_proposition_l232_232404

theorem contrapositive_proposition (α : ℝ) :
  (α = π / 4 → Real.tan α = 1) ↔ (Real.tan α ≠ 1 → α ≠ π / 4) :=
by
  sorry

end contrapositive_proposition_l232_232404


namespace manolo_face_mask_time_l232_232776
variable (x : ℕ)
def time_to_make_mask_first_hour := x
def face_masks_made_first_hour := 60 / x
def face_masks_made_next_three_hours := 180 / 6
def total_face_masks_in_four_hours := face_masks_made_first_hour + face_masks_made_next_three_hours

theorem manolo_face_mask_time : 
  total_face_masks_in_four_hours x = 45 ↔ x = 4 := sorry

end manolo_face_mask_time_l232_232776


namespace total_revenue_is_correct_l232_232652

-- Define the constants and conditions
def price_of_jeans : ℕ := 11
def price_of_tees : ℕ := 8
def quantity_of_tees_sold : ℕ := 7
def quantity_of_jeans_sold : ℕ := 4

-- Define the total revenue calculation
def total_revenue : ℕ :=
  (price_of_tees * quantity_of_tees_sold) +
  (price_of_jeans * quantity_of_jeans_sold)

-- The theorem to prove
theorem total_revenue_is_correct : total_revenue = 100 := 
by
  -- Proof is omitted for now
  sorry

end total_revenue_is_correct_l232_232652


namespace problem_1_problem_2_l232_232012

-- Problem I
theorem problem_1 (x : ℝ) (h : |x - 2| + |x - 1| < 4) : (-1/2 : ℝ) < x ∧ x < 7/2 :=
sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 1| ≥ 2) : a ≤ -1 ∨ a ≥ 3 :=
sorry

end problem_1_problem_2_l232_232012


namespace slope_equal_angles_l232_232024

-- Define the problem
theorem slope_equal_angles (k : ℝ) :
  (∀ (l1 l2 : ℝ), l1 = 1 ∧ l2 = 2 → (abs ((k - l1) / (1 + k * l1)) = abs ((l2 - k) / (1 + l2 * k)))) →
  (k = (1 + Real.sqrt 10) / 3 ∨ k = (1 - Real.sqrt 10) / 3) :=
by
  intros
  sorry

end slope_equal_angles_l232_232024


namespace water_required_l232_232607

-- Definitions based on the conditions
def balanced_equation : Prop := ∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl + H2O = NH4OH + HCl

-- New problem with the conditions translated into Lean
theorem water_required 
  (h_eq : balanced_equation)
  (n : ℕ)
  (m : ℕ)
  (mole_NH4Cl : n = 2 * m)
  (mole_H2O : m = 2) :
  n = m :=
by
  sorry

end water_required_l232_232607


namespace math_problem_l232_232482

open Real

variables {a b c d e f : ℝ}

theorem math_problem 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (hcond : abs (sqrt (a * b) - sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) :=
sorry

end math_problem_l232_232482


namespace poods_of_sugar_problem_l232_232710

noncomputable def solve_poods_of_sugar : ℕ :=
  let x := nat.sqrt 2025 - 5 in -- basic computation to find 20
  x

theorem poods_of_sugar_problem (x p : ℕ) 
  (h1 : x * p = 500) 
  (h2 : 500 / (x + 5) = p - 5) 
  : x = 20 := by
  sorry

end poods_of_sugar_problem_l232_232710


namespace jacob_dimes_l232_232659

-- Definitions of the conditions
def mrs_hilt_total_cents : ℕ := 2 * 1 + 2 * 10 + 2 * 5
def jacob_base_cents : ℕ := 4 * 1 + 1 * 5
def difference : ℕ := 13

-- The proof problem: prove Jacob has 1 dime.
theorem jacob_dimes (d : ℕ) (h : mrs_hilt_total_cents - (jacob_base_cents + 10 * d) = difference) : d = 1 := by
  sorry

end jacob_dimes_l232_232659


namespace total_number_of_legs_is_40_l232_232456

-- Define the number of octopuses Carson saw.
def number_of_octopuses := 5

-- Define the number of legs per octopus.
def legs_per_octopus := 8

-- Define the total number of octopus legs Carson saw.
def total_octopus_legs : Nat := number_of_octopuses * legs_per_octopus

-- Prove that the total number of octopus legs Carson saw is 40.
theorem total_number_of_legs_is_40 : total_octopus_legs = 40 := by
  sorry

end total_number_of_legs_is_40_l232_232456


namespace ratio_area_A_to_C_l232_232060

noncomputable def side_length (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area (side : ℕ) : ℕ :=
  side * side

theorem ratio_area_A_to_C : 
  let A_perimeter := 16
  let B_perimeter := 40
  let C_perimeter := 2 * A_perimeter
  let side_A := side_length A_perimeter
  let side_C := side_length C_perimeter
  let area_A := area side_A
  let area_C := area side_C
  (area_A : ℚ) / area_C = 1 / 4 :=
by
  sorry

end ratio_area_A_to_C_l232_232060


namespace range_of_numbers_is_six_l232_232276

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l232_232276


namespace range_of_set_is_six_l232_232300

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l232_232300


namespace cherries_on_June_5_l232_232109

theorem cherries_on_June_5 : 
  ∃ c : ℕ, (c + (c + 8) + (c + 16) + (c + 24) + (c + 32) = 130) ∧ (c + 32 = 42) :=
by
  sorry

end cherries_on_June_5_l232_232109


namespace find_number_l232_232576

theorem find_number :
  ∃ (x : ℤ), 
  x * (x + 6) = -8 ∧ 
  x^4 + (x + 6)^4 = 272 :=
by
  sorry

end find_number_l232_232576


namespace sum_term_addition_l232_232893

theorem sum_term_addition (k : ℕ) (hk : k ≥ 2) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end sum_term_addition_l232_232893


namespace minimum_value_of_expression_l232_232388

theorem minimum_value_of_expression (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (hxyz : x * y * z = 27) : 
  (3 * x + 2 * y + z) ≥ 18 * 2 ^ (1 / 3) := 
sorry

end minimum_value_of_expression_l232_232388


namespace arithmetic_sequence_count_l232_232125

-- Definitions based on the conditions and question
def sequence_count : ℕ := 314 -- The number of common differences for 315-term sequences
def set_size : ℕ := 2014     -- The maximum number in the set {1, 2, 3, ..., 2014}
def min_seq_length : ℕ := 315 -- The length of the arithmetic sequence

-- Lean 4 statement to verify the number of ways to form the required sequence
theorem arithmetic_sequence_count :
  ∃ (ways : ℕ), ways = 5490 ∧
  (∀ (d : ℕ), 1 ≤ d ∧ d ≤ 6 →
  (set_size - (sequence_count * d - 1)) > 0 → 
  ways = (
    if d = 1 then set_size - sequence_count + 1 else
    if d = 2 then set_size - (sequence_count * 2 - 1) + 1 else
    if d = 3 then set_size - (sequence_count * 3 - 1) + 1 else
    if d = 4 then set_size - (sequence_count * 4 - 1) + 1 else
    if d = 5 then set_size - (sequence_count * 5 - 1) + 1 else
    set_size - (sequence_count * 6 - 1) + 1) - 2
  ) :=
sorry

end arithmetic_sequence_count_l232_232125


namespace find_d_l232_232538

theorem find_d :
  ∃ d : ℝ, (∀ x y : ℝ, x^2 + 3 * y^2 + 6 * x - 18 * y + d = 0 → x = -3 ∧ y = 3) ↔ d = -27 :=
by {
  sorry
}

end find_d_l232_232538


namespace michael_students_l232_232393

theorem michael_students (M N : ℕ) (h1 : M = 5 * N) (h2 : M + N + 300 = 3500) : M = 2667 := 
by 
  -- This to be filled later
  sorry

end michael_students_l232_232393


namespace ratio_of_football_to_hockey_l232_232713

variables (B F H s : ℕ)

-- Definitions from conditions
def condition1 : Prop := B = F - 50
def condition2 : Prop := F = s * H
def condition3 : Prop := H = 200
def condition4 : Prop := B + F + H = 1750

-- Proof statement
theorem ratio_of_football_to_hockey (B F H s : ℕ) 
  (h1 : condition1 B F)
  (h2 : condition2 F s H)
  (h3 : condition3 H)
  (h4 : condition4 B F H) : F / H = 4 :=
sorry

end ratio_of_football_to_hockey_l232_232713


namespace area_enclosed_by_curve_and_line_l232_232067

theorem area_enclosed_by_curve_and_line :
  let f := fun x : ℝ => x^2 + 2
  let g := fun x : ℝ => 3 * x
  let A := ∫ x in (0 : ℝ)..1, (f x - g x) + ∫ x in (1 : ℝ)..2, (g x - f x)
  A = 1 := by
    sorry

end area_enclosed_by_curve_and_line_l232_232067


namespace isosceles_right_triangle_sums_l232_232616

theorem isosceles_right_triangle_sums (m n : ℝ)
  (h1: (1 * 2 + m * m + 2 * n) = 0)
  (h2: (1 + m^2 + 4) = (4 + m^2 + n^2)) :
  m + n = -1 :=
by {
  sorry
}

end isosceles_right_triangle_sums_l232_232616


namespace range_of_set_l232_232296

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l232_232296


namespace sum_f_always_negative_l232_232132

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_always_negative
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 :=
by
  unfold f
  sorry

end sum_f_always_negative_l232_232132


namespace solution_set_of_inequality_l232_232621

open Real Set

noncomputable def f (x : ℝ) : ℝ := exp (-x) - exp x - 5 * x

theorem solution_set_of_inequality :
  { x : ℝ | f (x ^ 2) + f (-x - 6) < 0 } = Iio (-2) ∪ Ioi 3 :=
by
  sorry

end solution_set_of_inequality_l232_232621


namespace value_of_a_l232_232488

theorem value_of_a (a : ℝ) (h₁ : ∀ x : ℝ, (2 * x - (1/3) * a ≤ 0) → (x ≤ 2)) : a = 12 :=
sorry

end value_of_a_l232_232488


namespace product_cubed_roots_l232_232590

-- Given conditions
def cbrt (x : ℝ) : ℝ := x^(1/3)
def expr : ℝ := cbrt (1 + 27) * cbrt (1 + cbrt 27) * cbrt 9

-- Main statement to prove
theorem product_cubed_roots : expr = cbrt 1008 :=
by sorry

end product_cubed_roots_l232_232590


namespace sequence_a_2017_l232_232350

theorem sequence_a_2017 :
  (∃ (a : ℕ → ℚ), (a 1 = 1) ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2016 * a n / (2014 * a n + 2016)) → a 2017 = 1008 / (1007 * 2017 + 1)) :=
by
  sorry

end sequence_a_2017_l232_232350


namespace range_of_numbers_is_six_l232_232251

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l232_232251


namespace product_of_fractions_l232_232938

theorem product_of_fractions :
  ∏ k in finset.range 501, (4 + 4 * k : ℕ) / (8 + 4 * k) = 1 / 502 := 
by sorry

end product_of_fractions_l232_232938


namespace number_of_positive_factors_of_60_l232_232817

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l232_232817


namespace slope_to_y_intercept_ratio_l232_232165

theorem slope_to_y_intercept_ratio (m b : ℝ) (c : ℝ) (h1 : m = c * b) (h2 : 2 * m + b = 0) : c = -1 / 2 :=
by sorry

end slope_to_y_intercept_ratio_l232_232165


namespace range_of_set_l232_232290

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l232_232290


namespace find_y_l232_232986

theorem find_y
  (x y : ℝ)
  (h1 : x - y = 10)
  (h2 : x + y = 8) : y = -1 :=
by
  sorry

end find_y_l232_232986


namespace sum_congruence_example_l232_232906

theorem sum_congruence_example (a b c : ℤ) (h1 : a % 15 = 7) (h2 : b % 15 = 3) (h3 : c % 15 = 9) : 
  (a + b + c) % 15 = 4 :=
by 
  sorry

end sum_congruence_example_l232_232906


namespace E_midpoint_of_AI_l232_232615

-- Define the points A, B, C, D, E, F, G, H, I
variables {A B C D E F G H I : EuclideanSpace ℝ (Fin 2)}

-- Definitions for equilateral triangles with positive orientation
def equilateral_positive (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R P ∧
  (⊿PQR) > 0 -- placeholder for positive orientation condition.

axiom ABD_equilateral : equilateral_positive A B D
axiom BAE_equilateral : equilateral_positive B A E
axiom CAF_equilateral : equilateral_positive C A F
axiom DFG_equilateral : equilateral_positive D F G
axiom ECH_equilateral : equilateral_positive E C H
axiom GHI_equilateral : equilateral_positive G H I

-- The main theorem statement
theorem E_midpoint_of_AI : 
  ∃ M : EuclideanSpace ℝ (Fin 2), (M = E) ∧ (dist A E = dist E I) ∧ (affine_space.between ℝ A E I) :=
sorry

end E_midpoint_of_AI_l232_232615


namespace intersection_M_N_l232_232623

def M : Set ℝ := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
def N : Set ℝ := Set.univ

theorem intersection_M_N : M ∩ N = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l232_232623


namespace number_of_words_with_at_least_one_consonant_l232_232151

def total_5_letter_words : ℕ := 6 ^ 5

def total_5_letter_vowel_words : ℕ := 2 ^ 5

def total_5_letter_words_with_consonant : ℕ := total_5_letter_words - total_5_letter_vowel_words

theorem number_of_words_with_at_least_one_consonant :
  total_5_letter_words_with_consonant = 7744 :=
  by
    -- We assert the calculation follows correctly:
    -- total_5_letter_words == 6^5 = 7776
    -- total_5_letter_vowel_words == 2^5 = 32
    -- 7776 - 32 == 7744
    sorry

end number_of_words_with_at_least_one_consonant_l232_232151


namespace magnified_diameter_l232_232222

theorem magnified_diameter (diameter_actual : ℝ) (magnification_factor : ℕ) 
  (h_actual : diameter_actual = 0.005) (h_magnification : magnification_factor = 1000) :
  diameter_actual * magnification_factor = 5 :=
by 
  sorry

end magnified_diameter_l232_232222


namespace exponential_fixed_point_l232_232544

theorem exponential_fixed_point (a : ℝ) (hx₁ : a > 0) (hx₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a ^ x) } := by
  sorry 

end exponential_fixed_point_l232_232544


namespace fraction_subtraction_l232_232111

theorem fraction_subtraction : 
  (4 + 6 + 8 + 10) / (3 + 5 + 7) - (3 + 5 + 7 + 9) / (4 + 6 + 8) = 8 / 15 :=
  sorry

end fraction_subtraction_l232_232111


namespace angle_B_possible_values_l232_232635

theorem angle_B_possible_values
  (a b : ℝ) (A B : ℝ)
  (h_a : a = 2)
  (h_b : b = 2 * Real.sqrt 3)
  (h_A : A = Real.pi / 6) 
  (h_A_range : (0 : ℝ) < A ∧ A < Real.pi) :
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
  sorry

end angle_B_possible_values_l232_232635


namespace solve_for_a_and_b_l232_232508
-- Import the necessary library

open Classical

variable (a b x : ℝ)

theorem solve_for_a_and_b (h1 : 0 ≤ x) (h2 : x < 1) (h3 : x + 2 * a ≥ 4) (h4 : (2 * x - b) / 3 < 1) : a + b = 1 := 
by
  sorry

end solve_for_a_and_b_l232_232508


namespace total_revenue_proof_l232_232210

-- Define constants for the problem
def original_price_per_case : ℝ := 25
def first_group_customers : ℕ := 8
def first_group_cases_per_customer : ℕ := 3
def first_group_discount_percentage : ℝ := 0.15
def second_group_customers : ℕ := 4
def second_group_cases_per_customer : ℕ := 2
def second_group_discount_percentage : ℝ := 0.10
def third_group_customers : ℕ := 8
def third_group_cases_per_customer : ℕ := 1

-- Calculate the prices after discount
def discounted_price_first_group : ℝ := original_price_per_case * (1 - first_group_discount_percentage)
def discounted_price_second_group : ℝ := original_price_per_case * (1 - second_group_discount_percentage)
def regular_price : ℝ := original_price_per_case

-- Calculate the total revenue
def total_revenue_first_group : ℝ := first_group_customers * first_group_cases_per_customer * discounted_price_first_group
def total_revenue_second_group : ℝ := second_group_customers * second_group_cases_per_customer * discounted_price_second_group
def total_revenue_third_group : ℝ := third_group_customers * third_group_cases_per_customer * regular_price

def total_revenue : ℝ := total_revenue_first_group + total_revenue_second_group + total_revenue_third_group

-- Prove that the total revenue is $890
theorem total_revenue_proof : total_revenue = 890 := by
  sorry

end total_revenue_proof_l232_232210


namespace total_investment_amount_l232_232096

-- Define the conditions
def total_interest_in_one_year : ℝ := 1023
def invested_at_6_percent : ℝ := 8200
def interest_rate_6_percent : ℝ := 0.06
def interest_rate_7_5_percent : ℝ := 0.075

-- Define the equation based on the conditions
def interest_from_6_percent_investment : ℝ := invested_at_6_percent * interest_rate_6_percent

def total_investment_is_correct (T : ℝ) : Prop :=
  let interest_from_7_5_percent_investment := (T - invested_at_6_percent) * interest_rate_7_5_percent
  interest_from_6_percent_investment + interest_from_7_5_percent_investment = total_interest_in_one_year

-- Statement to prove
theorem total_investment_amount : total_investment_is_correct 15280 :=
by
  unfold total_investment_is_correct
  unfold interest_from_6_percent_investment
  simp
  sorry

end total_investment_amount_l232_232096


namespace num_factors_60_l232_232813

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l232_232813


namespace sum_of_geometric_terms_l232_232789

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

theorem sum_of_geometric_terms {a : ℕ → ℝ} 
  (hseq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_sum135 : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end sum_of_geometric_terms_l232_232789


namespace pat_kate_mark_ratio_l232_232395

variables (P K M r : ℚ) 

theorem pat_kate_mark_ratio (h1 : P + K + M = 189) 
                            (h2 : P = r * K) 
                            (h3 : P = (1 / 3) * M) 
                            (h4 : M = K + 105) :
  r = 4 / 3 :=
sorry

end pat_kate_mark_ratio_l232_232395


namespace solution_set_condition_l232_232634

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - 1| > a) ↔ a < 1 :=
by
  sorry

end solution_set_condition_l232_232634


namespace zach_cookies_total_l232_232432

theorem zach_cookies_total :
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  cookies_monday + cookies_tuesday + cookies_wednesday = 92 :=
by
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  sorry

end zach_cookies_total_l232_232432


namespace relationship_among_abc_l232_232340

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := (2 / 3) ^ (2 / 5)
noncomputable def c : ℝ := Real.log (1 / 5) / Real.log (1 / 3)

theorem relationship_among_abc : c > b ∧ b > a :=
by
  have h1 : a = (1 / 3) ^ (2 / 5) := rfl
  have h2 : b = (2 / 3) ^ (2 / 5) := rfl
  have h3 : c = Real.log (1 / 5) / Real.log (1 / 3) := rfl
  sorry

end relationship_among_abc_l232_232340


namespace gear_teeth_count_l232_232158

theorem gear_teeth_count 
  (x y z: ℕ) 
  (h1: x + y + z = 60) 
  (h2: 4 * x - 20 = 5 * y) 
  (h3: 5 * y = 10 * z):
  x = 30 ∧ y = 20 ∧ z = 10 :=
by
  sorry

end gear_teeth_count_l232_232158


namespace jean_average_mark_l232_232842

/-
  Jean writes five tests and achieves the following marks: 80, 70, 60, 90, and 80.
  Prove that her average mark on these five tests is 76.
-/
theorem jean_average_mark : 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  average_mark = 76 :=
by 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  sorry

end jean_average_mark_l232_232842


namespace pairs_divisible_by_three_l232_232837

theorem pairs_divisible_by_three (P T : ℕ) (h : 5 * P = 3 * T) : ∃ k : ℕ, P = 3 * k := 
sorry

end pairs_divisible_by_three_l232_232837


namespace factor_difference_of_squares_l232_232000

theorem factor_difference_of_squares (y : ℝ) : 81 - 16 * y^2 = (9 - 4 * y) * (9 + 4 * y) :=
by
  sorry

end factor_difference_of_squares_l232_232000


namespace simplify_expression_l232_232872

theorem simplify_expression (y : ℝ) : 3 * y + 5 * y + 6 * y + 10 = 14 * y + 10 :=
by
  sorry

end simplify_expression_l232_232872


namespace range_of_set_is_six_l232_232304

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l232_232304


namespace range_of_m_l232_232965

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x, (x^2 + 1) * (x^2 - 8 * x - 20) ≤ 0 → (x^2 - 2 * x + (1 - m^2)) ≤ 0) →
  m ≥ 9 := by
  sorry

end range_of_m_l232_232965


namespace sum_angles_eq_990_degrees_l232_232206

noncomputable def alpha_sum : ℝ :=
  (54 + 126 + 198 + 270 + 342 : ℝ)

theorem sum_angles_eq_990_degrees :
  ∑ k in (finset.range 5), real.angle.to_degrees (real.angle.of_real ((270 + 360 * k) / 5)) = 990 :=
by {
  sorry
}

end sum_angles_eq_990_degrees_l232_232206


namespace round_trip_completion_percentage_l232_232584

-- Define the distances for each section
def sectionA_distance : Float := 10
def sectionB_distance : Float := 20
def sectionC_distance : Float := 15

-- Define the speeds for each section
def sectionA_speed : Float := 50
def sectionB_speed : Float := 40
def sectionC_speed : Float := 60

-- Define the delays for each section
def sectionA_delay : Float := 1.15
def sectionB_delay : Float := 1.10

-- Calculate the time for each section without delays
def sectionA_time : Float := sectionA_distance / sectionA_speed
def sectionB_time : Float := sectionB_distance / sectionB_speed
def sectionC_time : Float := sectionC_distance / sectionC_speed

-- Calculate the time with delays for the trip to the center
def sectionA_time_with_delay : Float := sectionA_time * sectionA_delay
def sectionB_time_with_delay : Float := sectionB_time * sectionB_delay
def sectionC_time_with_delay : Float := sectionC_time

-- Total time with delays to the center
def total_time_to_center : Float := sectionA_time_with_delay + sectionB_time_with_delay + sectionC_time_with_delay

-- Total distance to the center
def total_distance_to_center : Float := sectionA_distance + sectionB_distance + sectionC_distance

-- Total round trip distance
def total_round_trip_distance : Float := total_distance_to_center * 2

-- Distance covered on the way back
def distance_back : Float := total_distance_to_center * 0.2

-- Total distance covered considering the delays and the return trip
def total_distance_covered : Float := total_distance_to_center + distance_back

-- Effective completion percentage of the round trip
def completion_percentage : Float := (total_distance_covered / total_round_trip_distance) * 100

-- The main theorem statement
theorem round_trip_completion_percentage :
  completion_percentage = 60 := by
  sorry

end round_trip_completion_percentage_l232_232584


namespace messages_tuesday_l232_232712

theorem messages_tuesday (T : ℕ) (h1 : 300 + T + (T + 300) + 2 * (T + 300) = 2000) : 
  T = 200 := by
  sorry

end messages_tuesday_l232_232712


namespace perpendicular_vectors_l232_232145

noncomputable def a (k : ℝ) : ℝ × ℝ := (2 * k - 4, 3)
noncomputable def b (k : ℝ) : ℝ × ℝ := (-3, k)

theorem perpendicular_vectors (k : ℝ) (h : (2 * k - 4) * (-3) + 3 * k = 0) : k = 4 :=
sorry

end perpendicular_vectors_l232_232145


namespace circle_and_line_properties_l232_232014

-- Define the circle C with center on the positive x-axis and passing through the origin
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l: y = kx + 2
def line_l (k x y : ℝ) : Prop := y = k * x + 2

-- Statement: the circle and line setup
theorem circle_and_line_properties (k : ℝ) : 
  ∀ (x y : ℝ), 
  circle_C x y → 
  ∃ (x1 y1 x2 y2 : ℝ), 
  line_l k x1 y1 ∧ 
  line_l k x2 y2 ∧ 
  circle_C x1 y1 ∧ 
  circle_C x2 y2 ∧ 
  (x1 ≠ x2 ∧ y1 ≠ y2) → 
  k < -3/4 ∧
  ( (y1 / x1) + (y2 / x2) = 1 ) :=
by
  sorry

end circle_and_line_properties_l232_232014


namespace ratio_of_average_speed_to_still_water_speed_l232_232571

noncomputable def speed_of_current := 6
noncomputable def speed_in_still_water := 18
noncomputable def downstream_speed := speed_in_still_water + speed_of_current
noncomputable def upstream_speed := speed_in_still_water - speed_of_current
noncomputable def distance_each_way := 1
noncomputable def total_distance := 2 * distance_each_way
noncomputable def time_downstream := (distance_each_way : ℝ) / (downstream_speed : ℝ)
noncomputable def time_upstream := (distance_each_way : ℝ) / (upstream_speed : ℝ)
noncomputable def total_time := time_downstream + time_upstream
noncomputable def average_speed := (total_distance : ℝ) / (total_time : ℝ)
noncomputable def ratio_average_speed := (average_speed : ℝ) / (speed_in_still_water : ℝ)

theorem ratio_of_average_speed_to_still_water_speed :
  ratio_average_speed = (8 : ℝ) / (9 : ℝ) :=
sorry

end ratio_of_average_speed_to_still_water_speed_l232_232571


namespace smallest_AAB_l232_232099

theorem smallest_AAB : ∃ (A B : ℕ), (1 <= A ∧ A <= 9) ∧ (1 <= B ∧ B <= 9) ∧ (AB = 10 * A + B) ∧ (AAB = 100 * A + 10 * A + B) ∧ (110 * A + B = 8 * (10 * A + B)) ∧ (AAB = 221) :=
by
  sorry

end smallest_AAB_l232_232099


namespace find_matrix_l232_232474

noncomputable def M : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![7 / 29, 5 / 29, 0], ![3 / 29, 2 / 29, 0], ![0, 0, 1]]

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![2, -5, 0], ![-3, 7, 0], ![0, 0, 1]]

def I : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 0, 0], ![0, 1, 0], ![0, 0, 1]]

theorem find_matrix : M * A = I :=
by
  sorry

end find_matrix_l232_232474


namespace find_common_ratio_l232_232140

noncomputable def common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) : ℝ :=
3

theorem find_common_ratio 
( a : ℕ → ℝ) 
( d : ℝ) 
(h1 : d ≠ 0)
(h2 : ∀ n, a (n + 1) = a n + d)
(h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) :
common_ratio_of_geometric_sequence a d h1 h2 h3 = 3 :=
sorry

end find_common_ratio_l232_232140


namespace solve_system_and_compute_l232_232777

-- Given system of equations
variables {x y : ℝ}
variables (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5)

-- Statement to prove
theorem solve_system_and_compute :
  (x - y = -1) ∧ (x + y = 3) ∧ ((1/3 * (x^2 - y^2)) * (x^2 - 2*x*y + y^2) = -1) :=
by
  sorry

end solve_system_and_compute_l232_232777


namespace area_comparison_l232_232114

namespace Quadrilaterals

open Real

-- Define the vertices of both quadrilaterals
def quadrilateral_I_vertices : List (ℝ × ℝ) := [(0, 0), (2, 0), (2, 2), (0, 1)]
def quadrilateral_II_vertices : List (ℝ × ℝ) := [(0, 0), (3, 0), (3, 1), (0, 2)]

-- Area calculation function (example function for clarity)
def area_of_quadrilateral (vertices : List (ℝ × ℝ)) : ℝ :=
  -- This would use the actual geometry to compute the area
  2.5 -- placeholder for the area of quadrilateral I
  -- 4.5 -- placeholder for the area of quadrilateral II

theorem area_comparison :
  (area_of_quadrilateral quadrilateral_I_vertices) < (area_of_quadrilateral quadrilateral_II_vertices) :=
  sorry

end Quadrilaterals

end area_comparison_l232_232114


namespace hardcover_volumes_l232_232763

theorem hardcover_volumes (h p : ℕ) (h1 : h + p = 10) (h2 : 25 * h + 15 * p = 220) : h = 7 :=
by sorry

end hardcover_volumes_l232_232763


namespace set_intersection_complement_l232_232026

/-- Definition of the universal set U. -/
def U := ({1, 2, 3, 4, 5} : Set ℕ)

/-- Definition of the set M. -/
def M := ({3, 4, 5} : Set ℕ)

/-- Definition of the set N. -/
def N := ({2, 3} : Set ℕ)

/-- Statement of the problem to be proven. -/
theorem set_intersection_complement :
  ((U \ N) ∩ M) = ({4, 5} : Set ℕ) :=
by
  sorry

end set_intersection_complement_l232_232026


namespace regular_polygon_is_octagon_l232_232242

theorem regular_polygon_is_octagon (n : ℕ) (interior_angle exterior_angle : ℝ) :
  interior_angle = 3 * exterior_angle ∧ interior_angle + exterior_angle = 180 → n = 8 :=
by
  intros h
  sorry

end regular_polygon_is_octagon_l232_232242


namespace rhombus_perimeter_l232_232703

-- Define the conditions for the rhombus
variable (d1 d2 : ℝ) (a b s : ℝ)

-- State the condition that the diagonals of a rhombus measure 24 cm and 10 cm
def diagonal_condition := (d1 = 24) ∧ (d2 = 10)

-- State the Pythagorean theorem for the lengths of half-diagonals
def pythagorean_theorem := a^2 + b^2 = s^2

-- State the relationship of diagonals bisecting each other at right angles
def bisect_condition := (a = d1 / 2) ∧ (b = d2 / 2)

-- State the definition of the perimeter for a rhombus
def perimeter (s : ℝ) : ℝ := 4 * s

-- The theorem we want to prove
theorem rhombus_perimeter : diagonal_condition d1 d2 →
                            bisect_condition d1 d2 a b →
                            pythagorean_theorem a b s →
                            perimeter s = 52 :=
by
  intros h1 h2 h3
  -- Proof would go here, but it is omitted
  sorry

end rhombus_perimeter_l232_232703


namespace pencils_per_student_l232_232889

theorem pencils_per_student (num_students total_pencils : ℕ)
  (h1 : num_students = 4) (h2 : total_pencils = 8) : total_pencils / num_students = 2 :=
by
  -- Proof omitted
  sorry

end pencils_per_student_l232_232889


namespace unique_triple_gcd_square_l232_232606

theorem unique_triple_gcd_square (m n l : ℕ) (H1 : m + n = Nat.gcd m n ^ 2)
                                  (H2 : m + l = Nat.gcd m l ^ 2)
                                  (H3 : n + l = Nat.gcd n l ^ 2) : (m, n, l) = (2, 2, 2) :=
by
  sorry

end unique_triple_gcd_square_l232_232606


namespace jill_and_bob_payment_l232_232845

-- Definitions of the conditions
def price_of_first_house (X : ℝ) := X
def price_of_second_house (Y X : ℝ) := 2 * X

theorem jill_and_bob_payment :
  ∃ X, ∃ Y, Y = 2 * X ∧ X + Y = 600000 ∧ X = 200000 :=
by
  sorry

end jill_and_bob_payment_l232_232845


namespace marble_draw_probability_l232_232452

def marble_probabilities : ℚ :=
  let prob_white_a := 5 / 10
  let prob_black_a := 5 / 10
  let prob_yellow_b := 8 / 15
  let prob_yellow_c := 3 / 10
  let prob_green_d := 6 / 10
  let prob_white_then_yellow_then_green := prob_white_a * prob_yellow_b * prob_green_d
  let prob_black_then_yellow_then_green := prob_black_a * prob_yellow_c * prob_green_d
  prob_white_then_yellow_then_green + prob_black_then_yellow_then_green

theorem marble_draw_probability :
  marble_probabilities = 17 / 50 := by
  sorry

end marble_draw_probability_l232_232452


namespace not_divisible_59_l232_232864

theorem not_divisible_59 (x y : ℕ) (hx : ¬ (59 ∣ x)) (hy : ¬ (59 ∣ y)) 
  (h : (3 * x + 28 * y) % 59 = 0) : (5 * x + 16 * y) % 59 ≠ 0 :=
by
  sorry

end not_divisible_59_l232_232864


namespace sum_of_distances_to_focus_is_ten_l232_232619

theorem sum_of_distances_to_focus_is_ten (P : ℝ × ℝ) (A B F : ℝ × ℝ)
  (hP : P = (2, 1))
  (hA : A.1^2 = 12 * A.2)
  (hB : B.1^2 = 12 * B.2)
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hFocus : F = (3, 0)) :
  |A.1 - F.1| + |B.1 - F.1| = 10 :=
by
  sorry

end sum_of_distances_to_focus_is_ten_l232_232619


namespace find_c_l232_232976

theorem find_c (c : ℝ) : (∀ x : ℝ, -2 < x ∧ x < 1 → x^2 + x - c < 0) → c = 2 :=
by
  intros h
  -- Sorry to skip the proof
  sorry

end find_c_l232_232976


namespace minimum_visible_pairs_l232_232715

-- Definitions of the problem conditions
def num_birds : ℕ := 155
def max_arc : ℝ := 10
def circle_degree : ℝ := 360
def positions : ℕ := 35
def min_pairs(visible_pairs: ℕ) : Prop :=
  ∃ x : Fin 36 → Finset ℕ,  -- x denotes the distribution of birds
  (x.Sum = 155) ∧
  (Σ p in (Finset.range positions), (x p).card.choose 2) = visible_pairs

-- Problem Statement: Prove that the smallest number of mutually visible pairs is 270
theorem minimum_visible_pairs : min_pairs 270 := by
  sorry

end minimum_visible_pairs_l232_232715


namespace chameleon_problem_l232_232057

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l232_232057


namespace find_common_ratio_l232_232041

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m, ∃ q, a (n + 1) = a n * q ∧ a (m + 1) = a m * q

theorem find_common_ratio 
  (a : ℕ → α) 
  (h : is_geometric_sequence a) 
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 1 / 4) : 
  ∃ q, q = 1 / 2 :=
by
  sorry

end find_common_ratio_l232_232041


namespace friend_saves_per_week_l232_232700

theorem friend_saves_per_week
  (x : ℕ) 
  (you_have : ℕ := 160)
  (you_save_per_week : ℕ := 7)
  (friend_have : ℕ := 210)
  (weeks : ℕ := 25)
  (total_you_save : ℕ := you_have + you_save_per_week * weeks)
  (total_friend_save : ℕ := friend_have + x * weeks) 
  (h : total_you_save = total_friend_save) : x = 5 := 
by 
  sorry

end friend_saves_per_week_l232_232700


namespace motorcycles_in_anytown_l232_232094

variable (t s m : ℕ) -- t: number of trucks, s: number of sedans, m: number of motorcycles
variable (r_trucks r_sedans r_motorcycles : ℕ) -- r_trucks : truck ratio, r_sedans : sedan ratio, r_motorcycles : motorcycle ratio
variable (n_sedans : ℕ) -- n_sedans: number of sedans

theorem motorcycles_in_anytown
  (h1 : r_trucks = 3) -- ratio of trucks
  (h2 : r_sedans = 7) -- ratio of sedans
  (h3 : r_motorcycles = 2) -- ratio of motorcycles
  (h4 : s = 9100) -- number of sedans
  (h5 : s = (r_sedans * n_sedans)) -- relationship between sedans and parts
  (h6 : t = (r_trucks * n_sedans)) -- relationship between trucks and parts
  (h7 : m = (r_motorcycles * n_sedans)) -- relationship between motorcycles and parts
  : m = 2600 := by
    sorry

end motorcycles_in_anytown_l232_232094


namespace find_n_from_equation_l232_232559

theorem find_n_from_equation : ∃ n : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 22 ∧ n = 4 :=
by
  sorry

end find_n_from_equation_l232_232559


namespace llesis_more_rice_l232_232856

theorem llesis_more_rice :
  let total_rice := 50
  let llesis_fraction := 7 / 10
  let llesis_rice := total_rice * llesis_fraction
  let everest_rice := total_rice - llesis_rice
  llesis_rice - everest_rice = 20 := by
    sorry

end llesis_more_rice_l232_232856


namespace cylinder_volume_transformation_l232_232074

variable (r h : ℝ)
variable (V_original : ℝ)
variable (V_new : ℝ)

noncomputable def original_volume : ℝ := Real.pi * r^2 * h

noncomputable def new_volume : ℝ := Real.pi * (3 * r)^2 * (2 * h)

theorem cylinder_volume_transformation 
  (h_original : original_volume r h = 15) :
  new_volume r h = 270 :=
by
  unfold original_volume at h_original
  unfold new_volume
  sorry

end cylinder_volume_transformation_l232_232074


namespace find_numbers_l232_232721

theorem find_numbers (a b : ℕ) 
  (h1 : a / b * 6 = 10)
  (h2 : a - b + 4 = 10) :
  a = 15 ∧ b = 9 := by
  sorry

end find_numbers_l232_232721


namespace total_money_in_wallet_l232_232308

-- Definitions of conditions
def initial_five_dollar_bills := 7
def initial_ten_dollar_bills := 1
def initial_twenty_dollar_bills := 3
def initial_fifty_dollar_bills := 1
def initial_one_dollar_coins := 8

def spent_groceries := 65
def paid_fifty_dollar_bill := 1
def paid_twenty_dollar_bill := 1
def received_five_dollar_bill_change := 1
def received_one_dollar_coin_change := 5

def received_twenty_dollar_bills_from_friend := 2
def received_one_dollar_bills_from_friend := 2

-- Proving total amount of money
theorem total_money_in_wallet : 
  initial_five_dollar_bills * 5 + 
  initial_ten_dollar_bills * 10 + 
  initial_twenty_dollar_bills * 20 + 
  initial_fifty_dollar_bills * 50 + 
  initial_one_dollar_coins * 1 - 
  spent_groceries + 
  received_five_dollar_bill_change * 5 + 
  received_one_dollar_coin_change * 1 + 
  received_twenty_dollar_bills_from_friend * 20 + 
  received_one_dollar_bills_from_friend * 1 
  = 150 := 
by
  -- This is where the proof would be located
  sorry

end total_money_in_wallet_l232_232308


namespace sum_of_squares_of_rates_l232_232948

theorem sum_of_squares_of_rates (c j s : ℕ) (cond1 : 3 * c + 2 * j + 2 * s = 80) (cond2 : 2 * j + 2 * s + 4 * c = 104) : 
  c^2 + j^2 + s^2 = 592 :=
sorry

end sum_of_squares_of_rates_l232_232948


namespace square_side_length_l232_232695

theorem square_side_length (area_circle perimeter_square : ℝ) (h1 : area_circle = 100) (h2 : perimeter_square = area_circle) :
  side_length_square perimeter_square = 25 :=
by
  let s := 25 -- The length of one side of the square is 25
  sorry

def side_length_square (perimeter_square : ℝ) : ℝ :=
  perimeter_square / 4

end square_side_length_l232_232695


namespace find_specified_time_l232_232997

theorem find_specified_time (distance : ℕ) (slow_time fast_time : ℕ → ℕ) (fast_is_double : ∀ x, fast_time x = 2 * slow_time x)
  (distance_value : distance = 900) (slow_time_eq : ∀ x, slow_time x = x + 1) (fast_time_eq : ∀ x, fast_time x = x - 3) :
  2 * (distance / (slow_time x)) = distance / (fast_time x) :=
by
  intros
  rw [distance_value, slow_time_eq, fast_time_eq]
  sorry

end find_specified_time_l232_232997


namespace average_age_of_4_students_l232_232198

theorem average_age_of_4_students :
  let total_age_15 := 15 * 15
  let age_15th := 25
  let total_age_9 := 16 * 9
  (total_age_15 - total_age_9 - age_15th) / 4 = 14 :=
by
  sorry

end average_age_of_4_students_l232_232198


namespace height_of_parallelogram_l232_232556

noncomputable def parallelogram_height (base area : ℝ) : ℝ :=
  area / base

theorem height_of_parallelogram :
  parallelogram_height 8 78.88 = 9.86 :=
by
  -- This is where the proof would go, but it's being omitted as per instructions.
  sorry

end height_of_parallelogram_l232_232556


namespace range_of_set_of_three_numbers_l232_232265

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l232_232265


namespace chickens_cheaper_than_buying_eggs_after_81_weeks_l232_232870

-- Definitions based on conditions
def cost_chickens (num_chickens : ℕ) (cost_per_chicken : ℕ) : ℕ := num_chickens * cost_per_chicken
def egg_production (num_chickens : ℕ) (eggs_per_chicken_per_week : ℕ) : ℕ := num_chickens * eggs_per_chicken_per_week
def weekly_savings (cost_per_dozen : ℕ) (weekly_feed_cost : ℕ) : ℕ := cost_per_dozen - weekly_feed_cost
def break_even_weeks (total_cost : ℕ) (weekly_savings : ℕ) : ℕ := total_cost / weekly_savings
def cheaper_than_after_weeks (break_even_weeks : ℕ) : ℕ := break_even_weeks + 1

-- Theorem to prove
theorem chickens_cheaper_than_buying_eggs_after_81_weeks :
  ∀ (cost_per_chicken weekly_feed_cost eggs_per_chicken_per_week cost_per_dozen num_chickens : ℕ),
  cost_per_chicken = 20 →
  weekly_feed_cost = 1 →
  eggs_per_chicken_per_week = 3 →
  cost_per_dozen = 2 →
  num_chickens = 4 →
  let total_cost := cost_chickens num_chickens cost_per_chicken,
      weekly_savings_amt := weekly_savings cost_per_dozen weekly_feed_cost,
      break_even := break_even_weeks total_cost weekly_savings_amt,
      weeks_needed := cheaper_than_after_weeks break_even
  in weeks_needed = 81 :=
begin
  intros,
  sorry
end

end chickens_cheaper_than_buying_eggs_after_81_weeks_l232_232870


namespace range_of_set_l232_232279

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l232_232279


namespace solve_for_N_l232_232464

theorem solve_for_N : ∃ N : ℕ, 32^4 * 4^5 = 2^N ∧ N = 30 := by
  sorry

end solve_for_N_l232_232464


namespace negation_of_p_range_of_m_if_p_false_l232_232977

open Real

noncomputable def neg_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 - m*x - m > 0

theorem negation_of_p (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m*x - m ≤ 0) ↔ neg_p m := 
by sorry

theorem range_of_m_if_p_false : 
  (∀ m : ℝ, neg_p m → (-4 < m ∧ m < 0)) :=
by sorry

end negation_of_p_range_of_m_if_p_false_l232_232977


namespace max_area_of_garden_l232_232567

theorem max_area_of_garden (p : ℝ) (h : p = 36) : 
  ∃ A : ℝ, (∀ l w : ℝ, l + l + w + w = p → l * w ≤ A) ∧ A = 81 :=
by
  sorry

end max_area_of_garden_l232_232567


namespace proof_problem_l232_232016

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) + f x = 0

def decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

def satisfies_neq_point (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Main problem statement to prove (with conditions)
theorem proof_problem (f : ℝ → ℝ)
  (Hodd : odd_function f)
  (Hdec : decreasing_on f {y | 0 < y})
  (Hpt : satisfies_neq_point f (-2)) :
  {x : ℝ | (x - 1) * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end proof_problem_l232_232016


namespace tank_capacity_l232_232701

theorem tank_capacity (C : ℝ) (h_leak : ∀ t, t = 6 -> C / 6 = C / t)
    (h_inlet : ∀ r, r = 240 -> r = 4 * 60)
    (h_net : ∀ t, t = 8 -> 240 - C / 6 = C / 8) :
    C = 5760 / 7 := 
by 
  sorry

end tank_capacity_l232_232701


namespace set_inter_complement_l232_232489

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 3}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem set_inter_complement :
  B ∩ (U \ A) = {2} :=
by
  sorry

end set_inter_complement_l232_232489


namespace Lizette_average_above_94_l232_232176

noncomputable def Lizette_new_weighted_average
  (score3: ℝ) (avg3: ℝ) (weight3: ℝ) (score_new1 score_new2: ℝ) (weight_new: ℝ) :=
  let total_points3 := avg3 * 3
  let total_weight3 := 3 * weight3
  let total_points := total_points3 + score_new1 + score_new2
  let total_weight := total_weight3 + 2 * weight_new
  total_points / total_weight

theorem Lizette_average_above_94:
  ∀ (score3 avg3 weight3 score_new1 score_new2 weight_new: ℝ),
  score3 = 92 →
  avg3 = 94 →
  weight3 = 0.15 →
  score_new1 > 94 →
  score_new2 > 94 →
  weight_new = 0.20 →
  Lizette_new_weighted_average score3 avg3 weight3 score_new1 score_new2 weight_new > 94 :=
by
  intros score3 avg3 weight3 score_new1 score_new2 weight_new h1 h2 h3 h4 h5 h6
  sorry

end Lizette_average_above_94_l232_232176


namespace increase_in_green_chameleons_is_11_l232_232055

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l232_232055


namespace pow_mod_remainder_l232_232690

theorem pow_mod_remainder : (3 ^ 304) % 11 = 4 := by
  sorry

end pow_mod_remainder_l232_232690


namespace range_of_numbers_is_six_l232_232271

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l232_232271


namespace problem_solution_l232_232152

theorem problem_solution (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by 
  sorry

end problem_solution_l232_232152


namespace solve_quadratic_equations_l232_232192

noncomputable def E1 := ∀ x : ℝ, x^2 - 14 * x + 21 = 0 ↔ (x = 7 + 2 * Real.sqrt 7 ∨ x = 7 - 2 * Real.sqrt 7)

noncomputable def E2 := ∀ x : ℝ, x^2 - 3 * x + 2 = 0 ↔ (x = 1 ∨ x = 2)

theorem solve_quadratic_equations :
  (E1) ∧ (E2) :=
by
  sorry

end solve_quadratic_equations_l232_232192


namespace combined_solid_sum_faces_edges_vertices_l232_232581

noncomputable def prism_faces : ℕ := 6
noncomputable def prism_edges : ℕ := 12
noncomputable def prism_vertices : ℕ := 8
noncomputable def new_pyramid_faces : ℕ := 4
noncomputable def new_pyramid_edges : ℕ := 4
noncomputable def new_pyramid_vertex : ℕ := 1

theorem combined_solid_sum_faces_edges_vertices :
  prism_faces - 1 + new_pyramid_faces + prism_edges + new_pyramid_edges + prism_vertices + new_pyramid_vertex = 34 :=
by
  -- proof would go here
  sorry

end combined_solid_sum_faces_edges_vertices_l232_232581


namespace expected_value_of_difference_l232_232110

noncomputable def expected_difference (num_days : ℕ) : ℝ :=
  let p_prime := 3 / 4
  let p_composite := 1 / 4
  let p_no_reroll := 2 / 3
  let expected_unsweetened_days := p_prime * p_no_reroll * num_days
  let expected_sweetened_days := p_composite * p_no_reroll * num_days
  expected_unsweetened_days - expected_sweetened_days

theorem expected_value_of_difference :
  expected_difference 365 = 121.667 := by
  sorry

end expected_value_of_difference_l232_232110


namespace skeleton_ratio_l232_232036

theorem skeleton_ratio (W M C : ℕ) 
  (h1 : W + M + C = 20)
  (h2 : M = C)
  (h3 : 20 * W + 25 * M + 10 * C = 375) :
  (W : ℚ) / (W + M + C) = 1 / 2 :=
by
  sorry

end skeleton_ratio_l232_232036


namespace range_of_set_l232_232287

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l232_232287


namespace age_difference_l232_232400

variables (F S M B : ℕ)

theorem age_difference:
  (F - S = 38) → (M - B = 36) → (F - M = 6) → (S - B = 4) :=
by
  intros h1 h2 h3
  -- Use the conditions to derive that S - B = 4
  sorry

end age_difference_l232_232400


namespace first_term_of_geometric_sequence_l232_232884

theorem first_term_of_geometric_sequence (a r : ℚ) 
  (h1 : a * r = 18) 
  (h2 : a * r^2 = 24) : 
  a = 27 / 2 := 
sorry

end first_term_of_geometric_sequence_l232_232884


namespace earliest_time_meet_l232_232930

open Nat

def lap_time_anna := 5
def lap_time_bob := 8
def lap_time_carol := 10

def lcm_lap_times : ℕ :=
  Nat.lcm lap_time_anna (Nat.lcm lap_time_bob lap_time_carol)

theorem earliest_time_meet : lcm_lap_times = 40 := by
  sorry

end earliest_time_meet_l232_232930


namespace f_comp_f_neg4_l232_232780

noncomputable theory

def f (x : ℝ) : ℝ :=
if x < 1 then x^2 else x - 1

theorem f_comp_f_neg4 : f (f (-4)) = 15 := by
  sorry

end f_comp_f_neg4_l232_232780


namespace count_divisible_2_3_or_5_lt_100_l232_232030
-- We need the Mathlib library for general mathematical functions

-- The main theorem statement
theorem count_divisible_2_3_or_5_lt_100 : 
  let A2 := Nat.floor (100 / 2)
  let A3 := Nat.floor (100 / 3)
  let A5 := Nat.floor (100 / 5)
  let A23 := Nat.floor (100 / 6)
  let A25 := Nat.floor (100 / 10)
  let A35 := Nat.floor (100 / 15)
  let A235 := Nat.floor (100 / 30)
  (A2 + A3 + A5 - A23 - A25 - A35 + A235) = 74 :=
by
  sorry

end count_divisible_2_3_or_5_lt_100_l232_232030


namespace digit_encoding_problem_l232_232216

theorem digit_encoding_problem :
  ∃ (A B : ℕ), 0 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 21 * A + B = 111 * B ∧ A = 5 ∧ B = 5 :=
by
  sorry

end digit_encoding_problem_l232_232216


namespace convert_rectangular_to_polar_l232_232321

theorem convert_rectangular_to_polar (x y : ℝ) (h₁ : x = -2) (h₂ : y = -2) : 
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (2 * Real.sqrt 2, 5 * Real.pi / 4) := by
  sorry

end convert_rectangular_to_polar_l232_232321


namespace exponential_function_example_l232_232103

def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a > 0, a ≠ 1 ∧ ∀ x, f x = a ^ x

theorem exponential_function_example : is_exponential_function (fun x => 3 ^ x) :=
by
  sorry

end exponential_function_example_l232_232103


namespace largest_possible_red_socks_l232_232235

theorem largest_possible_red_socks (r b : ℕ) (h1 : 0 < r) (h2 : 0 < b)
  (h3 : r + b ≤ 2500) (h4 : r > b) :
  r * (r - 1) + b * (b - 1) = 3/5 * (r + b) * (r + b - 1) → r ≤ 1164 :=
by sorry

end largest_possible_red_socks_l232_232235


namespace coefficients_identity_l232_232403

def coefficients_of_quadratic (a b c : ℤ) (x : ℤ) : Prop :=
  a * x^2 + b * x + c = 0

theorem coefficients_identity : ∀ x : ℤ,
  coefficients_of_quadratic 3 (-4) 1 x :=
by
  sorry

end coefficients_identity_l232_232403


namespace least_bulbs_needed_l232_232666

/-- Tulip bulbs come in packs of 15, and daffodil bulbs come in packs of 16.
  Rita wants to buy the same number of tulip and daffodil bulbs. 
  The goal is to prove that the least number of bulbs she needs to buy is 240, i.e.,
  the least common multiple of 15 and 16 is 240. -/
theorem least_bulbs_needed : Nat.lcm 15 16 = 240 := 
by
  sorry

end least_bulbs_needed_l232_232666


namespace log_base_change_l232_232605

-- Define the conditions: 8192 = 2 ^ 13 and change of base formula
def x : ℕ := 8192
def a : ℕ := 2
def n : ℕ := 13
def b : ℕ := 5

theorem log_base_change (log : ℕ → ℕ → ℝ) 
  (h1 : x = a ^ n) 
  (h2 : ∀ (x b c: ℕ), c ≠ 1 → log x b = (log x c) / (log b c) ): 
  log x b = 13 / (log 5 2) :=
by
  sorry

end log_base_change_l232_232605


namespace left_handed_women_percentage_l232_232512

theorem left_handed_women_percentage
  (x y : ℕ)
  (h1 : 4 * x = 5 * y)
  (h2 : 3 * x ≥ 3 * y) :
  (x / (4 * x) : ℚ) * 100 = 25 :=
by
  sorry

end left_handed_women_percentage_l232_232512


namespace number_of_factors_of_60_l232_232821

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l232_232821


namespace quadratic_eq_with_roots_l232_232785

theorem quadratic_eq_with_roots (x y : ℝ) (h : (x^2 - 6 * x + 9) = -|y - 1|) : 
  ∃ a : ℝ, (a^2 - 4 * a + 3 = 0) :=
by 
  sorry

end quadratic_eq_with_roots_l232_232785


namespace sum_of_rational_roots_is_6_l232_232117

noncomputable def h : Polynomial ℚ := Polynomial.X^3 - 6 * Polynomial.X^2 + 11 * Polynomial.X - 6

theorem sum_of_rational_roots_is_6 : (h.roots.filter (λ r, r.is_rat)).sum = 6 := by
  sorry

end sum_of_rational_roots_is_6_l232_232117


namespace range_of_x_l232_232013

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) :
  (2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
   abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2)
  ↔ (Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4) :=
by
  sorry

end range_of_x_l232_232013


namespace expression_equals_one_l232_232735

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end expression_equals_one_l232_232735


namespace larry_final_channels_l232_232169

def initial_channels : Int := 150
def removed_channels : Int := 20
def replacement_channels : Int := 12
def reduced_channels : Int := 10
def sports_package_channels : Int := 8
def supreme_sports_package_channels : Int := 7

theorem larry_final_channels :
  initial_channels 
  - removed_channels 
  + replacement_channels 
  - reduced_channels 
  + sports_package_channels 
  + supreme_sports_package_channels 
  = 147 := by
  rfl  -- Reflects the direct computation as per the problem

end larry_final_channels_l232_232169


namespace triangle_formation_segments_l232_232582

theorem triangle_formation_segments (a b c : ℝ) (h_sum : a + b + c = 1) (h_a : a < 1/2) (h_b : b < 1/2) (h_c : c < 1/2) : 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := 
by
  sorry

end triangle_formation_segments_l232_232582


namespace unique_solution_l232_232766

noncomputable def func_prop (f : ℝ → ℝ) : Prop :=
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) ∧
  (∀ x ≥ 1, f (x + 1) = (f x)^2 / x - 1 / x)

theorem unique_solution (f : ℝ → ℝ) :
  func_prop f → ∀ x ≥ 1, f x = x + 1 :=
by
  sorry

end unique_solution_l232_232766


namespace polygon_to_triangle_l232_232240

theorem polygon_to_triangle {n : ℕ} (h : n > 4) :
  ∃ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) :=
sorry

end polygon_to_triangle_l232_232240


namespace count_angles_l232_232945

open Real

noncomputable def isGeometricSequence (a b c : ℝ) : Prop :=
(a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a / b = b / c ∨ b / a = a / c ∨ c / a = a / b)

theorem count_angles (h1 : ∀ θ : ℝ, 0 < θ ∧ θ < 2 * π → (sin θ * cos θ = tan θ) ∨ (sin θ ^ 3 = cos θ ^ 2)) :
  ∃ n : ℕ, 
    (∀ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧ (θ % (π/2) ≠ 0) → isGeometricSequence (sin θ) (cos θ) (tan θ) ) → 
    n = 6 := 
sorry

end count_angles_l232_232945


namespace find_x_l232_232328

noncomputable def is_solution (x : ℝ) : Prop :=
   (⌊x * ⌊x⌋⌋ = 29)

theorem find_x (x : ℝ) (h : is_solution x) : 5.8 ≤ x ∧ x < 6 :=
sorry

end find_x_l232_232328


namespace vector_coordinates_l232_232971

theorem vector_coordinates (b : ℝ × ℝ)
  (a : ℝ × ℝ := (Real.sqrt 3, 1))
  (angle : ℝ := 2 * Real.pi / 3)
  (norm_b : ℝ := 1)
  (dot_product_eq : (a.fst * b.fst + a.snd * b.snd = -1))
  (norm_b_eq : (b.fst ^ 2 + b.snd ^ 2 = 1)) :
  b = (0, -1) ∨ b = (-Real.sqrt 3 / 2, 1 / 2) :=
sorry

end vector_coordinates_l232_232971


namespace missing_score_find_missing_score_l232_232843

theorem missing_score
  (score1 score2 score3 score4 mean total : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89)
  (hTotal : total = 445) :
  score1 + score2 + score3 + score4 + x = total :=
by
  sorry

theorem find_missing_score
  (score1 score2 score3 score4 mean : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89) :
  (score1 + score2 + score3 + score4 + x) / 5 = mean
  → x = 90 :=
by
  sorry

end missing_score_find_missing_score_l232_232843


namespace workers_problem_l232_232380

theorem workers_problem
    (n : ℕ)
    (total_workers : ℕ)
    (c_choose_2 : ℕ → ℕ)
    (probability_jack_jill : ℚ) :
    total_workers = n + 2 →
    c_choose_2 total_workers = (total_workers * (total_workers - 1)) / 2 →
    probability_jack_jill = 1 / (c_choose_2 total_workers) →
    probability_jack_jill = 1 / 6 →
    n = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1] at h2
  sorry

end workers_problem_l232_232380


namespace sum_of_altitudes_of_triangle_l232_232757

theorem sum_of_altitudes_of_triangle : 
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  altitude1 + altitude2 + altitude3 = (22 * Real.sqrt 73 + 48) / Real.sqrt 73 :=
by
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  sorry

end sum_of_altitudes_of_triangle_l232_232757


namespace exists_fixed_point_A4_l232_232038

-- Definitions for Part 1
def A1 := (1 : ℝ, 0 : ℝ)
def A2 := (-2 : ℝ, 0 : ℝ)

lemma locus_of_M (M : ℝ × ℝ) :
  (real.sqrt ((M.1 - A1.1) ^ 2 + M.2 ^ 2) / real.sqrt ((M.1 + 2) ^ 2 + M.2 ^ 2) = real.sqrt 2 / 2)
  ↔ (M.1^2 + M.2^2 - 8 * M.1 - 2 = 0) :=
sorry

-- Definitions for Part 2
def circle_N (x y : ℝ) := (x-3)^2 + y^2 = 4
def A3 := (-1 : ℝ, 0 : ℝ)
def ratio_condition (N A4 : ℝ × ℝ) := real.sqrt ((N.1 + 1) ^ 2 + N.2 ^ 2) / real.sqrt ((N.1 - A4.1) ^ 2 + (N.2 - A4.2) ^ 2) = 2

theorem exists_fixed_point_A4 (N : ℝ × ℝ) (hN : circle_N N.1 N.2) :
  ∃ A4 : ℝ × ℝ, ratio_condition N A4 ∧ A4 = (2, 0) :=
sorry

end exists_fixed_point_A4_l232_232038


namespace factorize_polynomial_l232_232326

variable (x : ℝ)

theorem factorize_polynomial : 4 * x^3 - 8 * x^2 + 4 * x = 4 * x * (x - 1)^2 := 
by 
  sorry

end factorize_polynomial_l232_232326


namespace x_intersection_difference_l232_232006

-- Define the conditions
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

theorem x_intersection_difference :
  let x₁ := (1 + Real.sqrt 6) / 5
  let x₂ := (1 - Real.sqrt 6) / 5
  (parabola1 x₁ = parabola2 x₁) → (parabola1 x₂ = parabola2 x₂) →
  (x₁ - x₂) = (2 * Real.sqrt 6) / 5 := 
by
  sorry

end x_intersection_difference_l232_232006


namespace range_of_set_l232_232285

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l232_232285


namespace impossible_configuration_l232_232557

-- Define the initial state of stones in boxes
def stones_in_box (n : ℕ) : ℕ :=
  if n ≥ 1 ∧ n ≤ 100 then n else 0

-- Define the condition for moving stones between boxes
def can_move_stones (box1 box2 : ℕ) : Prop :=
  stones_in_box box1 + stones_in_box box2 = 101

-- The proposition: it is impossible to achieve the desired configuration
theorem impossible_configuration :
  ¬ ∃ boxes : ℕ → ℕ, 
    (boxes 70 = 69) ∧ 
    (boxes 50 = 51) ∧ 
    (∀ n, n ≠ 70 → n ≠ 50 → boxes n = stones_in_box n) ∧
    (∀ n1 n2, can_move_stones n1 n2 → (boxes n1 + boxes n2 = 101)) :=
sorry

end impossible_configuration_l232_232557


namespace find_first_parrot_weight_l232_232318

def cats_weights := [7, 10, 13, 15]
def cats_sum := List.sum cats_weights
def dog1 := cats_sum - 2
def dog2 := cats_sum + 7
def dog3 := (dog1 + dog2) / 2
def dogs_sum := dog1 + dog2 + dog3
def total_parrots_weight := 2 / 3 * dogs_sum

noncomputable def parrot1 := 2 / 5 * total_parrots_weight
noncomputable def parrot2 := 3 / 5 * total_parrots_weight

theorem find_first_parrot_weight : parrot1 = 38 :=
by
  sorry

end find_first_parrot_weight_l232_232318


namespace find_two_digit_number_l232_232833

theorem find_two_digit_number (x y : ℕ) (h1 : 10 * x + y = 4 * (x + y) + 3) (h2 : 10 * x + y = 3 * x * y + 5) : 10 * x + y = 23 :=
by {
  sorry
}

end find_two_digit_number_l232_232833


namespace two_presses_printing_time_l232_232032

def printing_time (presses newspapers hours : ℕ) : ℕ := sorry

theorem two_presses_printing_time :
  ∀ (presses newspapers hours : ℕ),
    (presses = 4) →
    (newspapers = 8000) →
    (hours = 6) →
    printing_time 2 6000 hours = 9 := sorry

end two_presses_printing_time_l232_232032


namespace intersection_of_M_and_N_l232_232800

theorem intersection_of_M_and_N :
  let M := { x : ℝ | -6 ≤ x ∧ x < 4 }
  let N := { x : ℝ | -2 < x ∧ x ≤ 8 }
  M ∩ N = { x | -2 < x ∧ x < 4 } :=
by
  sorry -- Proof is omitted

end intersection_of_M_and_N_l232_232800


namespace eval_g_at_3_l232_232070

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem eval_g_at_3 : g 3 = 10 := by
  -- Proof goes here
  sorry

end eval_g_at_3_l232_232070


namespace unique_solution_is_2_or_minus_2_l232_232988

theorem unique_solution_is_2_or_minus_2 (a : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, (y^2 + a * y + 1 = 0 ↔ y = x)) → (a = 2 ∨ a = -2) :=
by sorry

end unique_solution_is_2_or_minus_2_l232_232988


namespace floor_sqrt_sum_l232_232745

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l232_232745


namespace weight_of_dry_grapes_l232_232911

theorem weight_of_dry_grapes (w_fresh : ℝ) (perc_water_fresh perc_water_dried : ℝ) (w_non_water : ℝ) (w_dry : ℝ) :
  w_fresh = 5 →
  perc_water_fresh = 0.90 →
  perc_water_dried = 0.20 →
  w_non_water = w_fresh * (1 - perc_water_fresh) →
  w_non_water = w_dry * (1 - perc_water_dried) →
  w_dry = 0.625 :=
by sorry

end weight_of_dry_grapes_l232_232911


namespace num_possible_arrangements_l232_232641

def tea_picking : Fin 6 := 0
def cherry_picking : Fin 6 := 1
def strawberry_picking : Fin 6 := 2
def weeding : Fin 6 := 3
def tree_planting : Fin 6 := 4
def cow_milking : Fin 6 := 5

def activities := {tea_picking, cherry_picking, strawberry_picking, weeding, tree_planting, cow_milking}

theorem num_possible_arrangements : 
  let A62 := @Finset.choose 6 2 activities
  let A42 := @Finset.choose 4 2 activities
  let C61 := @Finset.choose 6 1 activities
  let C51 := @Finset.choose 5 1 activities
  let C41 := @Finset.choose 4 1 activities
  2 * A62.length * A42.length + C61.length * C51.length * C41.length * 2 + (A62.length * 2) = 630 :=
by
  sorry

end num_possible_arrangements_l232_232641


namespace num_factors_of_60_l232_232829

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l232_232829


namespace simplify_and_evaluate_expression_l232_232191

theorem simplify_and_evaluate_expression (a : ℤ) (ha : a = -2) : 
  (1 + 1 / (a - 1)) / ((2 * a) / (a ^ 2 - 1)) = -1 / 2 := by
  sorry

end simplify_and_evaluate_expression_l232_232191


namespace range_of_set_l232_232247

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l232_232247


namespace triangle_area_l232_232077

theorem triangle_area (A B C : ℝ × ℝ) (hA : A = (0, 0)) (hB : B = (0, 8)) (hC : C = (10, 15)) : 
  let base := 8
  let height := 10
  let area := 1 / 2 * base * height
  area = 40.0 :=
by
  sorry

end triangle_area_l232_232077


namespace num_factors_of_60_l232_232827

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l232_232827


namespace function_increasing_on_interval_l232_232406

noncomputable def f : ℝ → ℝ := λ x, 1 + x - sin x

theorem function_increasing_on_interval :
  ∀ x ∈ (Ioo 0 (2 * real.pi)), deriv f x > 0 :=
begin
  intros x hx,
  have h_deriv : deriv f x = 1 - cos x,
  { sorry }, -- Derivation steps here
  have h_cos : -1 ≤ cos x ∧ cos x < 1,
  { sorry }, -- Range analysis here
  linarith,
end

end function_increasing_on_interval_l232_232406


namespace problem_statement_l232_232861

namespace LeanProofExample

def not_divisible (n : ℕ) (p : ℕ) : Prop :=
  ¬(p ∣ n)

theorem problem_statement (x y : ℕ) 
  (hx : not_divisible x 59) 
  (hy : not_divisible y 59)
  (h : 3 * x + 28 * y ≡ 0 [MOD 59]) :
  ¬(5 * x + 16 * y ≡ 0 [MOD 59]) :=
  sorry

end LeanProofExample

end problem_statement_l232_232861


namespace solve_m_l232_232625

theorem solve_m (m : ℝ) : (m + 1) / 6 = m / 1 → m = 1 / 5 :=
by
  intro h
  sorry

end solve_m_l232_232625


namespace rectangle_triangle_height_l232_232338

theorem rectangle_triangle_height (l : ℝ) (h : ℝ) (w : ℝ) (d : ℝ) 
  (hw : w = Real.sqrt 2 * l)
  (hd : d = Real.sqrt (l^2 + w^2))
  (A_triangle : (1 / 2) * d * h = l * w) :
  h = (2 * l * Real.sqrt 6) / 3 := by
  sorry

end rectangle_triangle_height_l232_232338


namespace probability_less_than_8_rings_l232_232377

def P_10_ring : ℝ := 0.20
def P_9_ring : ℝ := 0.30
def P_8_ring : ℝ := 0.10

theorem probability_less_than_8_rings : 
  (1 - (P_10_ring + P_9_ring + P_8_ring)) = 0.40 :=
by
  sorry

end probability_less_than_8_rings_l232_232377


namespace f_minus_ten_l232_232792

noncomputable def f : ℝ → ℝ := sorry

theorem f_minus_ten :
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) →
  (f 1 = 2) →
  f (-10) = 90 :=
by
  intros h1 h2
  sorry

end f_minus_ten_l232_232792


namespace determine_d_l232_232869

theorem determine_d (u v d c : ℝ) (p q : ℝ → ℝ)
  (hp : ∀ x, p x = x^3 + c * x + d)
  (hq : ∀ x, q x = x^3 + c * x + d + 300)
  (huv : p u = 0 ∧ p v = 0)  
  (hu5_v4 : q (u + 5) = 0 ∧ q (v - 4) = 0)
  (sum_roots_p : u + v + (-u - v) = 0)
  (sum_roots_q : (u + 5) + (v - 4) + (-u - v - 1) = 0)
  : d = -4 ∨ d = 6 :=
sorry

end determine_d_l232_232869


namespace nonnegative_integer_solutions_l232_232767

theorem nonnegative_integer_solutions :
  {ab : ℕ × ℕ | 3 * 2^ab.1 + 1 = ab.2^2} = {(0, 2), (3, 5), (4, 7)} :=
by
  sorry

end nonnegative_integer_solutions_l232_232767


namespace volume_and_surface_area_of_prism_l232_232079

theorem volume_and_surface_area_of_prism 
  (a b c : ℝ)
  (h1 : a * b = 24)
  (h2 : b * c = 18)
  (h3 : c * a = 12) :
  (a * b * c = 72) ∧ (2 * (a * b + b * c + c * a) = 108) := by
  sorry

end volume_and_surface_area_of_prism_l232_232079


namespace robert_has_2_more_years_l232_232609

theorem robert_has_2_more_years (R P T Rb M : ℕ) 
                                 (h1 : R = P + T + Rb + M)
                                 (h2 : R = 42)
                                 (h3 : P = 12)
                                 (h4 : T = 2 * Rb)
                                 (h5 : Rb = P - 4) : Rb - M = 2 := 
by 
-- skipped proof
  sorry

end robert_has_2_more_years_l232_232609


namespace min_f_when_a_neg3_range_of_a_l232_232854

open Real

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- First statement: Minimum value of f(x) when a = -3
theorem min_f_when_a_neg3 : (∀ x : ℝ, f x (-3) ≥ 4) ∧ (∃ x : ℝ,  f x (-3) = 4) := by
  sorry

-- Second statement: Range of a given the condition
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≤ 2 * a + 2 * abs (x - 1)) ↔ a ≥ 1/3 := by
  sorry

end min_f_when_a_neg3_range_of_a_l232_232854


namespace min_students_solved_both_l232_232887

/-- A simple mathematical proof problem to find the minimum number of students who solved both problems correctly --/
theorem min_students_solved_both (total_students first_problem second_problem : ℕ)
  (h₀ : total_students = 30)
  (h₁ : first_problem = 21)
  (h₂ : second_problem = 18) :
  ∃ (both_solved : ℕ), both_solved = 9 :=
by
  sorry

end min_students_solved_both_l232_232887


namespace hyperbola_properties_l232_232478

noncomputable def hyperbola_eq (x y : ℝ) : ℝ := x^2 - y^2

theorem hyperbola_properties :
  (∃ (a : ℝ) (c : ℝ) (λ : ℝ) (m : ℝ),
  -- Given conditions:
  a ≠ 0 ∧ c = sqrt 2 * a ∧ λ = 6 ∧ hyperbola_eq 4 (- sqrt 10) = λ ∧
  (x^2 - y^2 = λ) ∧ (3^2 - m^2 = λ) ∧
  let F1 : point (ℝ × ℝ) := ⟨2 * sqrt 3, 0⟩ in
  let F2 : point (ℝ × ℝ) := ⟨-2 * sqrt 3, 0⟩ in
  ∃ (M : point (ℝ × ℝ)),
  M.1 = 3 ∧ M.2 = m ∧
  -- Prove that:
  ((F1 - M) • (F2 - M) = 0) ∧   -- Perpendicular condition
  (M lies on the circle with diameter (F1F2)) ∧
  let area : ℝ := abs ((F1.1 * M.2 - F2.1 * M.2) / 2) in
  -- Area of triangle F1MF2
  area = 6
  sorry

end hyperbola_properties_l232_232478


namespace number_of_factors_of_60_l232_232820

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l232_232820


namespace sqrt_meaningful_iff_range_l232_232982

variable (x : ℝ)

theorem sqrt_meaningful_iff_range :
  (∃ (v : ℝ), v = sqrt (x - 1)) ↔ x ≥ 1 := sorry

end sqrt_meaningful_iff_range_l232_232982


namespace fencing_required_l232_232578

theorem fencing_required {length width : ℝ} 
  (uncovered_side : length = 20)
  (field_area : length * width = 50) :
  2 * width + length = 25 :=
by
  sorry

end fencing_required_l232_232578


namespace calculator_unit_prices_and_min_cost_l232_232232

-- Definitions for conditions
def unit_price_type_A (x : ℕ) : Prop :=
  ∀ y : ℕ, (y = x + 10) → (550 / x = 600 / y)

def purchase_constraint (a : ℕ) : Prop :=
  25 ≤ a ∧ a ≤ 100

def total_cost (a : ℕ) (x y : ℕ) : ℕ :=
  110 * a + 120 * (100 - a)

-- Statement to prove
theorem calculator_unit_prices_and_min_cost :
  ∃ x y, unit_price_type_A x ∧ unit_price_type_A x ∧ total_cost 100 x y = 11000 :=
by
  sorry

end calculator_unit_prices_and_min_cost_l232_232232


namespace no_real_roots_iff_l232_232499

theorem no_real_roots_iff (k : ℝ) : (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_iff_l232_232499


namespace certain_number_value_l232_232373

theorem certain_number_value (x : ℕ) (p n : ℕ) (hp : Nat.Prime p) (hx : x = 44) (h : x / (n * p) = 2) : n = 2 := 
by
  sorry

end certain_number_value_l232_232373


namespace inequality_proof_l232_232009

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) +
    (b / Real.sqrt (b^2 + 8 * a * c)) +
    (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_proof_l232_232009


namespace range_of_set_l232_232293

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l232_232293


namespace correct_conclusions_l232_232389

noncomputable def M : Set ℝ := sorry

axiom non_empty : Nonempty M
axiom mem_2 : (2 : ℝ) ∈ M
axiom closed_under_sub : ∀ {x y : ℝ}, x ∈ M → y ∈ M → (x - y) ∈ M
axiom closed_under_div : ∀ {x : ℝ}, x ∈ M → x ≠ 0 → (1 / x) ∈ M

theorem correct_conclusions :
  (0 : ℝ) ∈ M ∧
  (∀ x y : ℝ, x ∈ M → y ∈ M → (x + y) ∈ M) ∧
  (∀ x y : ℝ, x ∈ M → y ∈ M → (x * y) ∈ M) ∧
  ¬ (1 ∉ M) := sorry

end correct_conclusions_l232_232389


namespace range_of_set_l232_232291

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l232_232291


namespace range_of_set_l232_232288

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l232_232288


namespace expression_value_l232_232566

theorem expression_value (x y z : ℤ) (hx : x = -2) (hy : y = 1) (hz : z = 1) : 
  x^2 * y * z - x * y * z^2 = 6 :=
by
  rw [hx, hy, hz]
  rfl

end expression_value_l232_232566


namespace family_ages_l232_232379

theorem family_ages:
  (∀ (Peter Harriet Jane Emily father: ℕ),
  ((Peter + 12 = 2 * (Harriet + 12)) ∧
   (Jane = Emily + 10) ∧
   (Peter = 60 / 3) ∧
   (Peter = Jane + 5) ∧
   (Aunt_Lucy = 52) ∧
   (Aunt_Lucy = 4 + Peter_Jane_mother) ∧
   (father - 20 = Aunt_Lucy)) →
  (Harriet = 4) ∧ (Peter = 20) ∧ (Jane = 15) ∧ (Emily = 5) ∧ (father = 72)) :=
sorry

end family_ages_l232_232379


namespace min_apples_l232_232075

theorem min_apples (N : ℕ) : 
  (N ≡ 1 [MOD 3]) ∧ 
  (N ≡ 3 [MOD 4]) ∧ 
  (N ≡ 2 [MOD 5]) 
  → N = 67 := 
by
  sorry

end min_apples_l232_232075


namespace variance_of_dataSet_l232_232208

-- Define the given data set
def dataSet : List ℤ := [-2, -1, 0, 1, 2]

-- Define the function to calculate mean
def mean (data : List ℤ) : ℚ :=
  (data.sum : ℚ) / data.length

-- Define the function to calculate variance
def variance (data : List ℤ) : ℚ :=
  let μ := mean data
  (data.map (λ x => (x - μ) ^ 2)).sum / data.length

-- State the theorem: The variance of the given data set is 2
theorem variance_of_dataSet : variance dataSet = 2 := by
  sorry

end variance_of_dataSet_l232_232208


namespace polynomial_solution_l232_232174

noncomputable def f (n : ℕ) (X Y : ℝ) : ℝ :=
  (X - 2 * Y) * (X + Y) ^ (n - 1)

theorem polynomial_solution (n : ℕ) (f : ℝ → ℝ → ℝ)
  (h1 : ∀ (t x y : ℝ), f (t * x) (t * y) = t^n * f x y)
  (h2 : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0)
  (h3 : f 1 0 = 1) :
  ∀ (X Y : ℝ), f X Y = (X - 2 * Y) * (X + Y) ^ (n - 1) :=
by
  sorry

end polynomial_solution_l232_232174


namespace student_second_subject_percentage_l232_232720

theorem student_second_subject_percentage (x : ℝ) (h : (50 + x + 90) / 3 = 70) : x = 70 :=
by { sorry }

end student_second_subject_percentage_l232_232720


namespace multiplication_expansion_l232_232860

theorem multiplication_expansion (y : ℤ) :
  (y^4 + 9 * y^2 + 81) * (y^2 - 9) = y^6 - 729 :=
by
  sorry

end multiplication_expansion_l232_232860


namespace work_completion_days_l232_232910

theorem work_completion_days (a b c : ℝ) :
  (1/a) = 1/90 → (1/b) = 1/45 → (1/a + 1/b + 1/c) = 1/5 → c = 6 :=
by
  intros ha hb habc
  sorry

end work_completion_days_l232_232910


namespace graph_passes_through_point_l232_232878

theorem graph_passes_through_point (a : ℝ) (ha : 0 < a) (ha_ne_one : a ≠ 1) :
  let f := fun x : ℝ => a^(x - 3) + 2
  f 3 = 3 := by
  sorry

end graph_passes_through_point_l232_232878


namespace range_of_set_is_six_l232_232303

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l232_232303


namespace sum_of_floors_of_square_roots_l232_232744

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l232_232744


namespace abs_sum_less_than_two_l232_232963

theorem abs_sum_less_than_two (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) : |a + b| + |a - b| < 2 := 
sorry

end abs_sum_less_than_two_l232_232963


namespace smallest_value_wawbwcwd_l232_232172

noncomputable def g (x : ℝ) : ℝ := x^4 + 10 * x^3 + 35 * x^2 + 50 * x + 24

theorem smallest_value_wawbwcwd (w1 w2 w3 w4 : ℝ) : 
  (∀ x : ℝ, g x = 0 ↔ x = w1 ∨ x = w2 ∨ x = w3 ∨ x = w4) →
  |w1 * w2 + w3 * w4| = 12 ∨ |w1 * w3 + w2 * w4| = 12 ∨ |w1 * w4 + w2 * w3| = 12 :=
by 
  sorry

end smallest_value_wawbwcwd_l232_232172


namespace solve_for_x_l232_232773

theorem solve_for_x (x : ℝ) : 
  2.5 * ((3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002 → 
  x = 2.5 :=
by 
  sorry

end solve_for_x_l232_232773


namespace find_angle_C_find_area_of_triangle_l232_232841

variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides of the triangle

-- Proof 1: Prove \(C = \frac{\pi}{3}\) given \(a \cos B \cos C + b \cos A \cos C = \frac{c}{2}\).

theorem find_angle_C 
  (h : a * Real.cos B * Real.cos C + b * Real.cos A * Real.cos C = c / 2) : C = π / 3 :=
sorry

-- Proof 2: Prove the area of triangle \(ABC = \frac{3\sqrt{3}}{2}\) given \(c = \sqrt{7}\), \(a + b = 5\), and \(C = \frac{\pi}{3}\).

theorem find_area_of_triangle 
  (h1 : c = Real.sqrt 7) (h2 : a + b = 5) (h3 : C = π / 3) : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
sorry

end find_angle_C_find_area_of_triangle_l232_232841


namespace function_decomposition_l232_232104

open Real

noncomputable def f (x : ℝ) : ℝ := log (10^x + 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
noncomputable def h (x : ℝ) : ℝ := log (10^x + 1) - x / 2

theorem function_decomposition :
  ∀ x : ℝ, f x = g x + h x ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, h (-x) = h x) :=
by
  intro x
  sorry

end function_decomposition_l232_232104


namespace mr_bhaskar_tour_duration_l232_232704

theorem mr_bhaskar_tour_duration :
  ∃ d : Nat, 
    (d > 0) ∧ 
    (∃ original_daily_expense new_daily_expense : ℕ,
      original_daily_expense = 360 / d ∧
      new_daily_expense = original_daily_expense - 3 ∧
      360 = new_daily_expense * (d + 4)) ∧
      d = 20 :=
by
  use 20
  -- Here would come the proof steps to verify the conditions and reach the conclusion.
  sorry

end mr_bhaskar_tour_duration_l232_232704


namespace min_value_expression_l232_232162

open ProbabilityTheory

-- Definitions based on given conditions
variables {σ : ℝ} {X : ℝ → ℝ}

-- Let's assume X is a normal distribution with mean 100 and variance σ²
def normal_X : MeasureTheory.Measure ℝ := MeasureTheory.Measure.dirac 100

-- Definition of probabilities a and b
def a := ProbabilityTheory.ProbabilityMeasure.probability (X > 120)
def b := ProbabilityTheory.ProbabilityMeasure.probability (80 ≤ X ∧ X ≤ 100)

-- Given relationship between a and b
axiom rel_a_b : a + b = 1 / 2

theorem min_value_expression : (4 / a + 1 / b) = 18 :=
by
  -- The proof is skipped, as only the statement is required
  sorry

end min_value_expression_l232_232162


namespace original_three_digit_number_a_original_three_digit_number_b_l232_232085

section ProblemA

variables {x y z : ℕ}

/-- In a three-digit number, the first digit on the left was erased. Then, the resulting
  two-digit number was multiplied by 7, and the original three-digit number was obtained. -/
theorem original_three_digit_number_a (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  N = 7 * (10 * y + z)) : ∃ (N : ℕ), N = 350 :=
sorry

end ProblemA

section ProblemB

variables {x y z : ℕ}

/-- In a three-digit number, the middle digit was erased, and the resulting number 
  is 6 times smaller than the original. --/
theorem original_three_digit_number_b (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  6 * (10 * x + z) = N) : ∃ (N : ℕ), N = 108 :=
sorry

end ProblemB

end original_three_digit_number_a_original_three_digit_number_b_l232_232085


namespace car_rental_cost_per_mile_l232_232437

theorem car_rental_cost_per_mile
    (daily_rental_cost : ℕ)
    (daily_budget : ℕ)
    (miles_limit : ℕ)
    (cost_per_mile : ℕ) :
    daily_rental_cost = 30 →
    daily_budget = 76 →
    miles_limit = 200 →
    cost_per_mile = (daily_budget - daily_rental_cost) * 100 / miles_limit →
    cost_per_mile = 23 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end car_rental_cost_per_mile_l232_232437


namespace pick_two_black_cards_l232_232444

-- Definition: conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13
def black_suits : ℕ := 2
def red_suits : ℕ := 2
def total_black_cards : ℕ := black_suits * cards_per_suit

-- Theorem: number of ways to pick two different black cards
theorem pick_two_black_cards :
  (total_black_cards * (total_black_cards - 1)) = 650 :=
by
  -- proof here
  sorry

end pick_two_black_cards_l232_232444


namespace chameleon_increase_l232_232047

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l232_232047


namespace quadratic_no_real_roots_l232_232502

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x - k = 0)) ↔ k < -1 :=
by sorry

end quadratic_no_real_roots_l232_232502


namespace sawyer_saw_octopuses_l232_232189

def number_of_legs := 40
def legs_per_octopus := 8

theorem sawyer_saw_octopuses : number_of_legs / legs_per_octopus = 5 := 
by
  sorry

end sawyer_saw_octopuses_l232_232189


namespace complex_z_modulus_l232_232344

noncomputable def i : ℂ := Complex.I

theorem complex_z_modulus (z : ℂ) (h : (1 + i) * z = 2 * i) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end complex_z_modulus_l232_232344


namespace square_side_length_l232_232692

theorem square_side_length (π : ℝ) (s : ℝ) :
  (∃ r : ℝ, 100 = π * r^2) ∧ (4 * s = 100) → s = 25 := by
  sorry

end square_side_length_l232_232692


namespace equal_chessboard_numbers_l232_232181

theorem equal_chessboard_numbers (n : ℕ) (board : ℕ → ℕ → ℕ) 
  (mean_property : ∀ (x y : ℕ), board x y = (board (x-1) y + board (x+1) y + board x (y-1) + board x (y+1)) / 4) : 
  ∀ (x y : ℕ), board x y = board 0 0 :=
by
  -- Proof not required
  sorry

end equal_chessboard_numbers_l232_232181


namespace five_letter_words_with_consonant_l232_232148

theorem five_letter_words_with_consonant :
  let letters := { 'A', 'B', 'C', 'D', 'E', 'F' } in
  let consonants := { 'B', 'C', 'D', 'F' } in
  let vowels := { 'A', 'E' } in
  let total_5_letter_words := 6^5 in
  let total_vowel_only_words := 2^5 in
  total_5_letter_words - total_vowel_only_words = 7744 :=
by
  sorry

end five_letter_words_with_consonant_l232_232148


namespace range_of_m_l232_232834

theorem range_of_m {m : ℝ} (h : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ x^2 + 2 * x - m = 0) : 8 < m ∧ m < 15 :=
sorry

end range_of_m_l232_232834


namespace cos_alpha_third_quadrant_l232_232967

theorem cos_alpha_third_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : Real.tan α > 0) : Real.cos α = -12 / 13 := 
sorry

end cos_alpha_third_quadrant_l232_232967


namespace number_of_true_propositions_eq_2_l232_232459

theorem number_of_true_propositions_eq_2 :
  (¬(∀ (a b : ℝ), a < 0 → b > 0 → a + b < 0)) ∧
  (∀ (α β : ℝ), α = 90 → β = 90 → α = β) ∧
  (∀ (α β : ℝ), α + β = 90 → (∀ (γ : ℝ), γ + α = 90 → β = γ)) ∧
  (¬(∀ (ℓ m n : ℕ), (ℓ ≠ m ∧ ℓ ≠ n ∧ m ≠ n) → (∀ (α β : ℝ), α = β))) →
  2 = 2 :=
by
  sorry

end number_of_true_propositions_eq_2_l232_232459


namespace exists_natural_numbers_with_digit_sum_condition_l232_232462

def digit_sum (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

theorem exists_natural_numbers_with_digit_sum_condition :
  ∃ (a b c : ℕ), digit_sum (a + b) < 5 ∧ digit_sum (a + c) < 5 ∧ digit_sum (b + c) < 5 ∧ digit_sum (a + b + c) > 50 :=
by
  sorry

end exists_natural_numbers_with_digit_sum_condition_l232_232462


namespace ratio_of_x_to_y_l232_232507

theorem ratio_of_x_to_y (x y : ℚ) (h : (2 * x - 3 * y) / (x + 2 * y) = 5 / 4) : x / y = 22 / 3 := by
  sorry

end ratio_of_x_to_y_l232_232507


namespace range_of_set_l232_232298

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l232_232298


namespace find_q_l232_232072

def polynomial_q (x p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h₀ : r = 3)
  (h₁ : (-p / 3) = -r)
  (h₂ : (-r) = 1 + p + q + r) :
  q = -16 :=
by
  -- h₀ implies r = 3
  -- h₁ becomes (-p / 3) = -3
  -- which results in p = 9
  -- h₂ becomes -3 = 1 + 9 + q + 3
  -- leading to q = -16
  sorry

end find_q_l232_232072


namespace range_of_set_l232_232284

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l232_232284


namespace friend_saves_per_week_l232_232698

theorem friend_saves_per_week (x : ℕ) : 
  160 + 7 * 25 = 210 + x * 25 → x = 5 := 
by 
  sorry

end friend_saves_per_week_l232_232698


namespace sum_of_rational_roots_of_h_l232_232116

theorem sum_of_rational_roots_of_h :
  let h(x : ℚ) := x^3 - 6*x^2 + 11*x - 6
  h(1) = 0 ∧ h(2) = 0 ∧ h(3) = 0 →
  (1 + 2 + 3 = 6) := sorry

end sum_of_rational_roots_of_h_l232_232116


namespace range_of_set_l232_232280

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l232_232280


namespace number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l232_232146

def is_consonant (c : Char) : Prop :=
  c = 'B' ∨ c = 'C' ∨ c = 'D' ∨ c = 'F'

def count_5_letter_words_with_at_least_one_consonant : Nat :=
  let total_words := 6 ^ 5
  let vowel_words := 2 ^ 5
  total_words - vowel_words

theorem number_of_5_letter_words_with_at_least_one_consonant_equals_7744 :
  count_5_letter_words_with_at_least_one_consonant = 7744 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l232_232146


namespace total_cookies_after_three_days_l232_232430

-- Define the initial conditions
def cookies_baked_monday : ℕ := 32
def cookies_baked_tuesday : ℕ := cookies_baked_monday / 2
def cookies_baked_wednesday_before : ℕ := cookies_baked_tuesday * 3
def brother_ate : ℕ := 4

-- Define the total cookies before brother ate any
def total_cookies_before : ℕ := cookies_baked_monday + cookies_baked_tuesday + cookies_baked_wednesday_before

-- Define the total cookies after brother ate some
def total_cookies_after : ℕ := total_cookies_before - brother_ate

-- The proof statement
theorem total_cookies_after_three_days : total_cookies_after = 92 := by
  -- Here, we would provide the proof, but we add sorry for now to compile successfully.
  sorry

end total_cookies_after_three_days_l232_232430


namespace quadratic_real_roots_l232_232885

theorem quadratic_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ (x^2 - m * x + (m - 1) = 0) ∧ (y^2 - m * y + (m - 1) = 0) 
  ∨ ∃ z : ℝ, (z^2 - m * z + (m - 1) = 0) := 
sorry

end quadratic_real_roots_l232_232885


namespace range_of_a_l232_232970

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) ↔ a > -3/4 := 
sorry

end range_of_a_l232_232970


namespace product_sequence_equals_l232_232939

-- Define the form of each fraction in the sequence
def frac (k : ℕ) : ℚ := (4 * k : ℚ) / (4 * k + 4)

-- Define the product of the sequence from k=1 to k=501
def productSequence : ℚ :=
  (finset.range 501).prod (λ k => frac (k + 1))

-- The theorem that the product equals 1/502
theorem product_sequence_equals : productSequence = 1 / 502 := by
  sorry

end product_sequence_equals_l232_232939


namespace min_value_expr_l232_232961

variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = 1)

theorem min_value_expr : (1 / (2 * a)) + (1 / (2 * b)) + (8 / (a + b)) ≥ 4 :=
by
  sorry

end min_value_expr_l232_232961


namespace isosceles_trapezoid_l232_232385

-- Define a type for geometric points
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define structures for geometric properties
structure Trapezoid :=
  (A B C D M N : Point)
  (is_midpoint_M : 2 * M.x = A.x + B.x ∧ 2 * M.y = A.y + B.y)
  (is_midpoint_N : 2 * N.x = C.x + D.x ∧ 2 * N.y = C.y + D.y)
  (AB_parallel_CD : (B.y - A.y) * (D.x - C.x) = (B.x - A.x) * (D.y - C.y)) -- AB || CD
  (MN_perpendicular_AB_CD : (N.y - M.y) * (B.y - A.y) + (N.x - M.x) * (B.x - A.x) = 0 ∧
                            (N.y - M.y) * (D.y - C.y) + (N.x - M.x) * (D.x - C.x) = 0) -- MN ⊥ AB && MN ⊥ CD

-- The isosceles condition
def is_isosceles (T : Trapezoid) : Prop :=
  ((T.A.x - T.D.x) ^ 2 + (T.A.y - T.D.y) ^ 2) = ((T.B.x - T.C.x) ^ 2 + (T.B.y - T.C.y) ^ 2)

-- The theorem statement
theorem isosceles_trapezoid (T : Trapezoid) : is_isosceles T :=
by
  sorry

end isosceles_trapezoid_l232_232385


namespace polynomial_divisibility_l232_232831

def poly1 (x : ℝ) (k : ℝ) : ℝ := 3*x^3 - 9*x^2 + k*x - 12

theorem polynomial_divisibility (k : ℝ) :
  (∀ (x : ℝ), poly1 x k = (x - 3) * (3*x^2 + 4)) → (poly1 3 k = 0) := sorry

end polynomial_divisibility_l232_232831


namespace total_legs_at_pet_shop_l232_232933

theorem total_legs_at_pet_shop : 
  let birds := 3 in
  let dogs := 5 in
  let snakes := 4 in
  let spiders := 1 in
  let legs_bird := 2 in
  let legs_dog := 4 in
  let legs_snake := 0 in
  let legs_spider := 8 in
  (birds * legs_bird + dogs * legs_dog + snakes * legs_snake + spiders * legs_spider) = 34 :=
by 
  -- Proof will be here
  sorry

end total_legs_at_pet_shop_l232_232933


namespace maximal_inradius_of_tetrahedron_l232_232005

-- Define the properties and variables
variables (A B C D : ℝ) (h_A h_B h_C h_D : ℝ) (V r : ℝ)

-- Assumptions
variable (h_A_ge_1 : h_A ≥ 1)
variable (h_B_ge_1 : h_B ≥ 1)
variable (h_C_ge_1 : h_C ≥ 1)
variable (h_D_ge_1 : h_D ≥ 1)

-- Volume expressed in terms of altitudes and face areas
axiom vol_eq_Ah : V = (1 / 3) * A * h_A
axiom vol_eq_Bh : V = (1 / 3) * B * h_B
axiom vol_eq_Ch : V = (1 / 3) * C * h_C
axiom vol_eq_Dh : V = (1 / 3) * D * h_D

-- Volume expressed in terms of inradius and sum of face areas
axiom vol_eq_inradius : V = (1 / 3) * (A + B + C + D) * r

-- The theorem to prove
theorem maximal_inradius_of_tetrahedron : r = 1 / 4 :=
sorry

end maximal_inradius_of_tetrahedron_l232_232005


namespace find_d_l232_232546

noncomputable def problem_condition :=
  ∃ (v d : ℝ × ℝ) (t : ℝ) (x y : ℝ),
  (y = (5 * x - 7) / 6) ∧ 
  ((x, y) = (v.1 + t * d.1, v.2 + t * d.2)) ∧ 
  (x ≥ 4) ∧ 
  (dist (x, y) (4, 2) = t)

noncomputable def correct_answer : ℝ × ℝ := ⟨6 / 7, 5 / 7⟩

theorem find_d 
  (h : problem_condition) : 
  ∃ (d : ℝ × ℝ), d = correct_answer :=
sorry

end find_d_l232_232546


namespace joan_socks_remaining_l232_232519

-- Definitions based on conditions
def total_socks : ℕ := 1200
def white_socks : ℕ := total_socks / 4
def blue_socks : ℕ := total_socks * 3 / 8
def red_socks : ℕ := total_socks / 6
def green_socks : ℕ := total_socks / 12
def white_socks_lost : ℕ := white_socks / 3
def blue_socks_sold : ℕ := blue_socks / 2
def remaining_white_socks : ℕ := white_socks - white_socks_lost
def remaining_blue_socks : ℕ := blue_socks - blue_socks_sold

-- Theorem to prove the total number of remaining socks
theorem joan_socks_remaining :
  remaining_white_socks + remaining_blue_socks + red_socks + green_socks = 725 := by
  sorry

end joan_socks_remaining_l232_232519


namespace unique_lines_count_l232_232345

theorem unique_lines_count :
  let S := {0, 1, 2, 3, 5}
  in ∃ (A B : ℤ), A ≠ B ∧ A ∈ S ∧ B ∈ S ∧ (S.card.choose 2 + 2 * (S.card - 1) = 14) :=
by
  let S := {0, 1, 2, 3, 5}
  have h1 : S.card = 5 := by simp [S]
  have h2 : S.card.choose 2 = 10 := by simp [nat.choose_eq, S]
  sorry  -- Proof omitted

end unique_lines_count_l232_232345


namespace find_value_of_a_l232_232636

-- Define the setting for triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : b^2 - c^2 + 2 * a = 0)
variables (h2 : Real.tan C / Real.tan B = 3)

-- Given conditions and conclusion for the proof problem
theorem find_value_of_a 
  (h1 : b^2 - c^2 + 2 * a = 0) 
  (h2 : Real.tan C / Real.tan B = 3) : 
  a = 4 := 
sorry

end find_value_of_a_l232_232636


namespace find_angle_x_eq_38_l232_232040

theorem find_angle_x_eq_38
  (angle_ACD angle_ECB angle_DCE : ℝ)
  (h1 : angle_ACD = 90)
  (h2 : angle_ECB = 52)
  (h3 : angle_ACD + angle_ECB + angle_DCE = 180) :
  angle_DCE = 38 :=
by
  sorry

end find_angle_x_eq_38_l232_232040


namespace tomatoes_picked_l232_232439

theorem tomatoes_picked (initial_tomatoes picked_tomatoes : ℕ)
  (h₀ : initial_tomatoes = 17)
  (h₁ : initial_tomatoes - picked_tomatoes = 8) :
  picked_tomatoes = 9 :=
by
  sorry

end tomatoes_picked_l232_232439


namespace calculate_difference_square_l232_232359

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l232_232359


namespace range_of_set_is_six_l232_232302

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l232_232302


namespace range_of_set_l232_232260

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l232_232260


namespace mango_distribution_l232_232492

theorem mango_distribution (friends : ℕ) (initial_mangos : ℕ) 
    (share_left : ℕ) (share_right : ℕ) 
    (eat_mango : ℕ) (pass_mango_right : ℕ)
    (H1 : friends = 100) 
    (H2 : initial_mangos = 2019)
    (H3 : share_left = 2) 
    (H4 : share_right = 1) 
    (H5 : eat_mango = 1) 
    (H6 : pass_mango_right = 1) :
    ∃ final_count, final_count = 8 :=
by
  -- Proof is omitted.
  sorry

end mango_distribution_l232_232492


namespace initial_price_of_gasoline_l232_232881

theorem initial_price_of_gasoline 
  (P0 : ℝ) 
  (P1 : ℝ := 1.30 * P0)
  (P2 : ℝ := 0.75 * P1)
  (P3 : ℝ := 1.10 * P2)
  (P4 : ℝ := 0.85 * P3)
  (P5 : ℝ := 0.80 * P4)
  (h : P5 = 102.60) : 
  P0 = 140.67 :=
by sorry

end initial_price_of_gasoline_l232_232881


namespace rectangle_width_l232_232411

theorem rectangle_width (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w + l) = w * l) : w = 3 :=
by sorry

end rectangle_width_l232_232411


namespace length_of_marquita_garden_l232_232656

variable (length_marquita_garden : ℕ)

def total_area_mancino_gardens : ℕ := 3 * (16 * 5)
def total_gardens_area : ℕ := 304
def total_area_marquita_gardens : ℕ := total_gardens_area - total_area_mancino_gardens
def area_one_marquita_garden : ℕ := total_area_marquita_gardens / 2

theorem length_of_marquita_garden :
  (4 * length_marquita_garden = area_one_marquita_garden) →
  length_marquita_garden = 8 := by
  sorry

end length_of_marquita_garden_l232_232656


namespace olivia_earning_l232_232762

theorem olivia_earning
  (cost_per_bar : ℝ)
  (total_bars : ℕ)
  (unsold_bars : ℕ)
  (sold_bars : ℕ := total_bars - unsold_bars)
  (earnings : ℝ := sold_bars * cost_per_bar) :
  cost_per_bar = 3 → total_bars = 7 → unsold_bars = 4 → earnings = 9 :=
by
  sorry

end olivia_earning_l232_232762


namespace lex_read_pages_l232_232651

theorem lex_read_pages (total_pages days : ℕ) (h1 : total_pages = 240) (h2 : days = 12) :
  total_pages / days = 20 :=
by sorry

end lex_read_pages_l232_232651


namespace sequence_no_limit_l232_232958

noncomputable def sequence_limit (x : ℕ → ℝ) (a : ℝ) : Prop :=
    ∀ ε > 0, ∃ N, ∀ n > N, abs (x n - a) < ε

theorem sequence_no_limit (x : ℕ → ℝ) (a : ℝ) (ε : ℝ) (k : ℕ) :
    (ε > 0) ∧ (∀ n, n > k → abs (x n - a) ≥ ε) → ¬ sequence_limit x a :=
by
  sorry

end sequence_no_limit_l232_232958


namespace rational_function_sum_l232_232545

noncomputable def p (x : ℝ) : ℝ := (2 / 3) * (x - 2)
noncomputable def q (x : ℝ) : ℝ := (-4 / 3) * (x - 2)

theorem rational_function_sum :
  (p (-1) = 2) ∧ (q (-1) = 4) ∧ (degree q = 2) ∧ 
  (∃ l, filter.tendsto (λ (x: ℝ), p x / q x) filter.at_top (nhds 1)) ∧ 
  (∃ l, filter.tendsto (λ (x: ℝ), p x / q x) (nhds 2) filter.at_top) →
    p x + q x = (-2 / 3) * x + (4 / 3) :=
by
  sorry

end rational_function_sum_l232_232545


namespace max_min_sum_zero_l232_232071

def cubic_function (x : ℝ) : ℝ :=
  x^3 - 3 * x

def first_derivative (x : ℝ) : ℝ :=
  3 * x^2 - 3

theorem max_min_sum_zero :
  let m := cubic_function (-1);
  let n := cubic_function 1;
  m + n = 0 :=
by
  sorry

end max_min_sum_zero_l232_232071


namespace number_of_words_with_at_least_one_consonant_l232_232150

def total_5_letter_words : ℕ := 6 ^ 5

def total_5_letter_vowel_words : ℕ := 2 ^ 5

def total_5_letter_words_with_consonant : ℕ := total_5_letter_words - total_5_letter_vowel_words

theorem number_of_words_with_at_least_one_consonant :
  total_5_letter_words_with_consonant = 7744 :=
  by
    -- We assert the calculation follows correctly:
    -- total_5_letter_words == 6^5 = 7776
    -- total_5_letter_vowel_words == 2^5 = 32
    -- 7776 - 32 == 7744
    sorry

end number_of_words_with_at_least_one_consonant_l232_232150


namespace piecewise_function_solution_l232_232141

theorem piecewise_function_solution (m : ℝ) :
  (m = Real.sqrt 10 ∨ m = -1) ↔
  (if m > 0 then log m = 1 / 2 else 2 ^ m = 1 / 2) :=
by
  sorry

end piecewise_function_solution_l232_232141


namespace power_of_two_representation_l232_232533

/-- Prove that any number 2^n, where n = 3,4,5,..., can be represented 
as 7x^2 + y^2 where x and y are odd numbers. -/
theorem power_of_two_representation (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, (2*x ≠ 0 ∧ 2*y ≠ 0) ∧ 2^n = 7 * x^2 + y^2 :=
by
  sorry

end power_of_two_representation_l232_232533


namespace penny_initial_money_l232_232183

theorem penny_initial_money
    (pairs_of_socks : ℕ)
    (cost_per_pair : ℝ)
    (number_of_pairs : ℕ)
    (cost_of_hat : ℝ)
    (money_left : ℝ)
    (initial_money : ℝ)
    (H1 : pairs_of_socks = 4)
    (H2 : cost_per_pair = 2)
    (H3 : number_of_pairs = pairs_of_socks)
    (H4 : cost_of_hat = 7)
    (H5 : money_left = 5)
    (H6 : initial_money = (number_of_pairs * cost_per_pair) + cost_of_hat + money_left) : initial_money = 20 :=
sorry

end penny_initial_money_l232_232183


namespace correct_average_l232_232912

theorem correct_average (n : ℕ) (wrong_avg : ℕ) (wrong_num correct_num : ℕ) (correct_avg : ℕ)
  (h1 : n = 10) 
  (h2 : wrong_avg = 21)
  (h3 : wrong_num = 26)
  (h4 : correct_num = 36)
  (h5 : correct_avg = 22) :
  (wrong_avg * n + (correct_num - wrong_num)) / n = correct_avg :=
by
  sorry

end correct_average_l232_232912


namespace trail_length_is_48_meters_l232_232063

noncomputable def length_of_trail (d: ℝ) : Prop :=
  let normal_speed := 8 -- normal speed in m/s
  let mud_speed := normal_speed / 4 -- speed in mud in m/s

  let time_mud := (1 / 3 * d) / mud_speed -- time through the mud in seconds
  let time_normal := (2 / 3 * d) / normal_speed -- time through the normal trail in seconds

  let total_time := 12 -- total time in seconds

  total_time = time_mud + time_normal

theorem trail_length_is_48_meters : ∃ d: ℝ, length_of_trail d ∧ d = 48 :=
sorry

end trail_length_is_48_meters_l232_232063


namespace floor_sum_sqrt_1_to_25_l232_232751

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l232_232751


namespace probability_of_factor_less_than_ten_is_half_l232_232219

-- Definitions for the factors and counts
def numFactors (n : ℕ) : ℕ :=
  let psa := 1;
  let psb := 2;
  let psc := 1;
  (psa + 1) * (psb + 1) * (psc + 1)

def factorsLessThanTen (n : ℕ) : List ℕ :=
  if n = 90 then [1, 2, 3, 5, 6, 9] else []

def probabilityLessThanTen (n : ℕ) : ℚ :=
  let totalFactors := numFactors n;
  let lessThanTenFactors := factorsLessThanTen n;
  let favorableOutcomes := lessThanTenFactors.length;
  favorableOutcomes / totalFactors

-- The proof statement
theorem probability_of_factor_less_than_ten_is_half :
  probabilityLessThanTen 90 = 1 / 2 := sorry

end probability_of_factor_less_than_ten_is_half_l232_232219


namespace square_root_of_9_eq_pm_3_l232_232886

theorem square_root_of_9_eq_pm_3 (x : ℝ) : x^2 = 9 → x = 3 ∨ x = -3 :=
sorry

end square_root_of_9_eq_pm_3_l232_232886


namespace largest_four_digit_number_with_digits_sum_25_l232_232689

def four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  (n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10) = s)

theorem largest_four_digit_number_with_digits_sum_25 :
  ∃ n, four_digit n ∧ digits_sum_to n 25 ∧ ∀ m, four_digit m → digits_sum_to m 25 → m ≤ n :=
sorry

end largest_four_digit_number_with_digits_sum_25_l232_232689


namespace polynomial_evaluation_l232_232477

noncomputable def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem polynomial_evaluation : f 2 = 123 := by
  sorry

end polynomial_evaluation_l232_232477


namespace difference_of_squares_l232_232365

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l232_232365


namespace count_colorings_l232_232166

-- Define the number of disks
def num_disks : ℕ := 6

-- Define colorings with constraints: 2 black, 2 white, 2 blue considering rotations and reflections as equivalent
def valid_colorings : ℕ :=
  18  -- This is the result obtained using Burnside's Lemma as shown in the solution

theorem count_colorings : valid_colorings = 18 := by
  sorry

end count_colorings_l232_232166


namespace green_chameleon_increase_l232_232053

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l232_232053


namespace directrix_of_parabola_l232_232002

-- Define the given conditions
def parabola_eqn (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 5

-- The problem is to show that the directrix of this parabola has the equation y = 23/12
theorem directrix_of_parabola : 
  (∃ y : ℝ, ∀ x : ℝ, parabola_eqn x = y) →

  ∃ y : ℝ, y = 23 / 12 :=
sorry

end directrix_of_parabola_l232_232002


namespace car_selling_price_l232_232573

def car_material_cost : ℕ := 100
def car_production_per_month : ℕ := 4
def motorcycle_material_cost : ℕ := 250
def motorcycles_sold_per_month : ℕ := 8
def motorcycle_selling_price : ℤ := 50
def additional_motorcycle_profit : ℤ := 50

theorem car_selling_price (x : ℤ) :
  (motorcycles_sold_per_month * motorcycle_selling_price - motorcycle_material_cost)
  = (car_production_per_month * x - car_material_cost + additional_motorcycle_profit) →
  x = 50 :=
by
  sorry

end car_selling_price_l232_232573


namespace negation_of_proposition_l232_232549

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, 0 ≤ x ∧ (x^2 - 2*x - 3 = 0)) ↔ (∀ x : ℝ, 0 ≤ x → (x^2 - 2*x - 3 ≠ 0)) := 
by 
  sorry

end negation_of_proposition_l232_232549


namespace basketball_cards_price_l232_232530

theorem basketball_cards_price :
  let toys_cost := 3 * 10
  let shirts_cost := 5 * 6
  let total_cost := 70
  let basketball_cards_cost := total_cost - (toys_cost + shirts_cost)
  let packs_of_cards := 2
  (basketball_cards_cost / packs_of_cards) = 5 :=
by
  sorry

end basketball_cards_price_l232_232530


namespace find_a_l232_232527

noncomputable def A (a : ℝ) : Set ℝ := {1, 2, a}
noncomputable def B (a : ℝ) : Set ℝ := {1, a^2 - a}

theorem find_a (a : ℝ) : A a ⊇ B a → a = -1 ∨ a = 0 :=
by
  sorry

end find_a_l232_232527


namespace range_of_numbers_is_six_l232_232275

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l232_232275


namespace math_problem_l232_232964

variable {a b : ℕ → ℕ}

-- Condition 1: a_n is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

-- Condition 2: 2a₂ - a₇² + 2a₁₂ = 0
def satisfies_equation (a : ℕ → ℕ) : Prop :=
  2 * a 2 - (a 7)^2 + 2 * a 12 = 0

-- Condition 3: b_n is a geometric sequence
def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, b (n + m) = b n * b m

-- Condition 4: b₇ = a₇
def b7_eq_a7 (a b : ℕ → ℕ) : Prop :=
  b 7 = a 7

-- To prove: b₅ * b₉ = 16
theorem math_problem (a b : ℕ → ℕ)
  (h₁ : is_arithmetic_sequence a)
  (h₂ : satisfies_equation a)
  (h₃ : is_geometric_sequence b)
  (h₄ : b7_eq_a7 a b) :
  b 5 * b 9 = 16 :=
sorry

end math_problem_l232_232964


namespace unit_digit_25_pow_2010_sub_3_pow_2012_l232_232429

theorem unit_digit_25_pow_2010_sub_3_pow_2012 :
  (25^2010 - 3^2012) % 10 = 4 :=
by 
  sorry

end unit_digit_25_pow_2010_sub_3_pow_2012_l232_232429


namespace product_of_numbers_l232_232774

theorem product_of_numbers (a b : ℝ) 
  (h1 : a + b = 5 * (a - b))
  (h2 : a * b = 18 * (a - b)) : 
  a * b = 54 :=
by
  sorry

end product_of_numbers_l232_232774


namespace greatest_int_satisfying_inequality_l232_232896

theorem greatest_int_satisfying_inequality : 
  ∃ m : ℤ, (∀ x : ℤ, x - 5 > 4 * x - 1 → x ≤ -2) ∧ (∀ k : ℤ, k < -2 → k - 5 > 4 * k - 1) :=
by
  sorry

end greatest_int_satisfying_inequality_l232_232896


namespace cos_540_eq_neg_1_l232_232592

theorem cos_540_eq_neg_1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end cos_540_eq_neg_1_l232_232592


namespace triangle_side_lengths_relationship_l232_232722

variable {a b c : ℝ}

def is_quadratic_mean (a b c : ℝ) : Prop :=
  (2 * b^2 = a^2 + c^2)

def is_geometric_mean (a b c : ℝ) : Prop :=
  (b * a = c^2)

theorem triangle_side_lengths_relationship (a b c : ℝ) :
  (is_quadratic_mean a b c ∧ is_geometric_mean a b c) → 
  ∃ a b c, (2 * b^2 = a^2 + c^2) ∧ (b * a = c^2) :=
sorry

end triangle_side_lengths_relationship_l232_232722


namespace solve_eqn_l232_232951

theorem solve_eqn {x : ℝ} : x^4 + (3 - x)^4 = 130 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end solve_eqn_l232_232951


namespace find_y_l232_232987

theorem find_y
  (x y : ℝ)
  (h1 : x - y = 10)
  (h2 : x + y = 8) : y = -1 :=
by
  sorry

end find_y_l232_232987


namespace evaluate_f_at_5pi_over_6_l232_232794

noncomputable def f (x : ℝ) : ℝ := Real.sin (π / 2 + x) * Real.sin (π + x)

theorem evaluate_f_at_5pi_over_6 : f (5 * π / 6) = sqrt 3 / 4 :=
by
  sorry

end evaluate_f_at_5pi_over_6_l232_232794


namespace simplify_correct_l232_232873

def simplify_polynomial (x : Real) : Real :=
  (12 * x^10 + 6 * x^9 + 3 * x^8) + (2 * x^11 + x^10 + 4 * x^9 + x^7 + 4 * x^4 + 7 * x + 9)

theorem simplify_correct (x : Real) :
  simplify_polynomial x = 2 * x^11 + 13 * x^10 + 10 * x^9 + 3 * x^8 + x^7 + 4 * x^4 + 7 * x + 9 :=
by
  sorry

end simplify_correct_l232_232873


namespace num_factors_60_l232_232811

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l232_232811


namespace find_k_l232_232115

theorem find_k
  (S : ℝ)    -- Distance between the village and city
  (x : ℝ)    -- Speed of the truck in km/h
  (y : ℝ)    -- Speed of the car in km/h
  (H1 : 18 = 0.75 * x - 0.75 * x ^ 2 / (x + y))  -- Condition that truck leaving earlier meets 18 km closer to the city
  (H2 : 24 = x * y / (x + y))      -- Intermediate step from solving the first condition
  : (k = 8) :=    -- We need to show that k = 8
  sorry

end find_k_l232_232115


namespace time_between_last_two_rings_l232_232100

variable (n : ℕ) (x y : ℝ)

noncomputable def timeBetweenLastTwoRings : ℝ :=
  x + (n - 3) * y

theorem time_between_last_two_rings :
  timeBetweenLastTwoRings n x y = x + (n - 3) * y :=
by
  sorry

end time_between_last_two_rings_l232_232100


namespace sum_of_fractions_l232_232591

theorem sum_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (7 : ℚ) / 9
  a + b = 83 / 72 := 
by
  sorry

end sum_of_fractions_l232_232591


namespace rachel_math_homework_l232_232058

def rachel_homework (M : ℕ) (reading : ℕ) (biology : ℕ) (total : ℕ) : Prop :=
reading = 3 ∧ biology = 10 ∧ total = 15 ∧ reading + biology + M = total

theorem rachel_math_homework: ∃ M : ℕ, rachel_homework M 3 10 15 ∧ M = 2 := 
by 
  sorry

end rachel_math_homework_l232_232058


namespace sum_floor_sqrt_l232_232756

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l232_232756


namespace probability_pink_second_marble_l232_232451

def bagA := (5, 5)  -- (red, green)
def bagB := (8, 2)  -- (pink, purple)
def bagC := (3, 7)  -- (pink, purple)

def P (success total : ℕ) := success / total

def probability_red := P 5 10
def probability_green := P 5 10

def probability_pink_given_red := P 8 10
def probability_pink_given_green := P 3 10

theorem probability_pink_second_marble :
  probability_red * probability_pink_given_red +
  probability_green * probability_pink_given_green = 11 / 20 :=
sorry

end probability_pink_second_marble_l232_232451


namespace pet_shop_legs_l232_232932

theorem pet_shop_legs :
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  birds * bird_legs + dogs * dog_legs + snakes * snake_legs + spiders * spider_legs = 34 := 
by
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  sorry

end pet_shop_legs_l232_232932


namespace find_a_from_limit_l232_232476

theorem find_a_from_limit (a : ℝ) (h : (Filter.Tendsto (fun n : ℕ => (a * n - 2) / (n + 1)) Filter.atTop (Filter.principal {1}))) :
    a = 1 := 
sorry

end find_a_from_limit_l232_232476


namespace num_factors_60_l232_232814

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l232_232814


namespace calculate_value_pa_pb_l232_232378

theorem calculate_value_pa_pb :
  let P := (2 : ℝ, 2 : ℝ)
  let α := real.pi / 3
  let l_parametric_eqns (t : ℝ) := (2 + 1/2 * t, 2 + (real.sqrt 3)/2 * t) -- Parametric equations of the line through P with inclination π/3
  let C_eqn (x y : ℝ) := x^2 + y^2 = 2 * x -- Rectangular equation of the circle from its polar form
  let intersection_pts := {t : ℝ | C_eqn (2 + 1/2 * t) (2 + (real.sqrt 3)/2 * t)} -- Intersection points
  let t1 t2 := classical.some (exists_pair this intersection_pts) -- Let t1 and t2 be the roots of the equation from the intersections
  in
  (1 / |t1| + 1 / |t2|) = (2 * real.sqrt 3 + 1) / 4 :=
sorry

end calculate_value_pa_pb_l232_232378


namespace rice_difference_l232_232859

theorem rice_difference:
  ∀ (total : ℕ) (kept_fraction : ℚ) (kept : ℕ) (given : ℕ),
  total = 50 →
  kept_fraction = (7 : ℚ) / 10 →
  kept = total * kept_fraction →
  given = total - kept →
  (kept - given = 20) :=
by
  intros total kept_fraction kept given
  intros h_total h_fraction h_kept h_given
  sorry

end rice_difference_l232_232859


namespace gasoline_added_l232_232033

noncomputable def initial_amount (capacity: ℕ) : ℝ :=
  (3 / 4) * capacity

noncomputable def final_amount (capacity: ℕ) : ℝ :=
  (9 / 10) * capacity

theorem gasoline_added (capacity: ℕ) (initial_fraction final_fraction: ℝ) (initial_amount final_amount: ℝ) : 
  capacity = 54 ∧ initial_fraction = 3/4 ∧ final_fraction = 9/10 ∧ 
  initial_amount = initial_fraction * capacity ∧ 
  final_amount = final_fraction * capacity →
  final_amount - initial_amount = 8.1 :=
sorry

end gasoline_added_l232_232033


namespace least_total_cost_is_172_l232_232855

noncomputable def least_total_cost : ℕ :=
  let lcm := Nat.lcm (Nat.lcm 6 5) 8
  let strawberry_packs := lcm / 6
  let blueberry_packs := lcm / 5
  let cherry_packs := lcm / 8
  let strawberry_cost := strawberry_packs * 2
  let blueberry_cost := blueberry_packs * 3
  let cherry_cost := cherry_packs * 4
  strawberry_cost + blueberry_cost + cherry_cost

theorem least_total_cost_is_172 : least_total_cost = 172 := 
by
  sorry

end least_total_cost_is_172_l232_232855


namespace solve_system_of_inequalities_l232_232065

theorem solve_system_of_inequalities {x : ℝ} :
  (|x^2 + 5 * x| < 6) ∧ (|x + 1| ≤ 1) ↔ (0 ≤ x ∧ x < 2) ∨ (4 < x ∧ x ≤ 6) :=
by
  sorry

end solve_system_of_inequalities_l232_232065


namespace dot_product_v_w_l232_232738

def v : ℝ × ℝ := (-5, 3)
def w : ℝ × ℝ := (7, -9)

theorem dot_product_v_w : v.1 * w.1 + v.2 * w.2 = -62 := 
  by sorry

end dot_product_v_w_l232_232738


namespace expression_equality_l232_232315

theorem expression_equality (a b c : ℝ) : a * (a + b - c) = a^2 + a * b - a * c :=
by
  sorry

end expression_equality_l232_232315


namespace range_of_set_l232_232257

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l232_232257


namespace calculate_difference_square_l232_232358

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l232_232358


namespace factors_of_60_l232_232823

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l232_232823


namespace largest_value_x_l232_232769

-- Definition of the conditions
def equation (x : ℚ) : Prop :=
  (16 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 8 * x - 2

-- Statement of the proof 
theorem largest_value_x : ∀ x : ℚ, equation x → x ≤ 9 / 4 := sorry

end largest_value_x_l232_232769


namespace more_oaks_than_willows_l232_232682

theorem more_oaks_than_willows (total_trees willows : ℕ) (h1 : total_trees = 83) (h2 : willows = 36) :
  (total_trees - willows) - willows = 11 :=
by
  sorry

end more_oaks_than_willows_l232_232682


namespace solve_inequality_l232_232552

theorem solve_inequality (x : ℝ) :
  (x * (x + 2) / (x - 3) < 0) ↔ (x < -2 ∨ (0 < x ∧ x < 3)) :=
sorry

end solve_inequality_l232_232552


namespace amusement_park_line_l232_232184

theorem amusement_park_line (h1 : Eunji_position = 6) (h2 : people_behind_Eunji = 7) : total_people_in_line = 13 :=
by
  sorry

end amusement_park_line_l232_232184


namespace find_A_l232_232921

theorem find_A (A : ℕ) (B : ℕ) (h₀ : 0 ≤ B) (h₁ : B ≤ 999) :
  1000 * A + B = (A * (A + 1)) / 2 → A = 1999 := sorry

end find_A_l232_232921


namespace integer_solutions_for_exponential_equation_l232_232327

theorem integer_solutions_for_exponential_equation :
  ∃ (a b c : ℕ), 
  2 ^ a * 3 ^ b + 9 = c ^ 2 ∧ 
  (a = 4 ∧ b = 0 ∧ c = 5) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 3 ∧ c = 21) ∨ 
  (a = 3 ∧ b = 3 ∧ c = 15) ∨ 
  (a = 4 ∧ b = 5 ∧ c = 51) :=
by {
  -- This is where the proof would go.
  sorry
}

end integer_solutions_for_exponential_equation_l232_232327


namespace first_place_prize_is_200_l232_232717

-- Define the conditions from the problem
def total_prize_money : ℤ := 800
def num_winners : ℤ := 18
def second_place_prize : ℤ := 150
def third_place_prize : ℤ := 120
def fourth_to_eighteenth_prize : ℤ := 22
def fourth_to_eighteenth_winners : ℤ := num_winners - 3

-- Define the amount awarded to fourth to eighteenth place winners
def total_fourth_to_eighteenth_prize : ℤ := fourth_to_eighteenth_winners * fourth_to_eighteenth_prize

-- Define the total amount awarded to second and third place winners
def total_second_and_third_prize : ℤ := second_place_prize + third_place_prize

-- Define the total amount awarded to second to eighteenth place winners
def total_second_to_eighteenth_prize : ℤ := total_fourth_to_eighteenth_prize + total_second_and_third_prize

-- Define the amount awarded to first place
def first_place_prize : ℤ := total_prize_money - total_second_to_eighteenth_prize

-- Statement for proof required
theorem first_place_prize_is_200 : first_place_prize = 200 :=
by
  -- Assuming the conditions are correct
  sorry

end first_place_prize_is_200_l232_232717


namespace number_of_solutions_l232_232648

noncomputable def g_n (n : ℕ) (x : ℝ) := (Real.sin x)^(2 * n) + (Real.cos x)^(2 * n)

theorem number_of_solutions : ∀ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi) -> 
  8 * g_n 3 x - 6 * g_n 2 x = 3 * g_n 1 x -> false :=
by sorry

end number_of_solutions_l232_232648


namespace not_perfect_cube_of_cond_l232_232850

open Int

theorem not_perfect_cube_of_cond (n : ℤ) (h₁ : 0 < n) (k : ℤ) 
  (h₂ : n^5 + n^3 + 2 * n^2 + 2 * n + 2 = k ^ 3) : 
  ¬ ∃ m : ℤ, 2 * n^2 + n + 2 = m ^ 3 :=
sorry

end not_perfect_cube_of_cond_l232_232850


namespace alpha_gamma_shopping_ways_l232_232727

theorem alpha_gamma_shopping_ways :
  let oreos := 5
  let milks := 3
  let cookies := 2
  let total_items := oreos + milks + cookies

  let alpha_ways := binomial total_items 2
  let gamma_ways_2_items := binomial (oreos + cookies) 2 + (oreos + cookies)
  let case1 := alpha_ways * gamma_ways_2_items

  let alpha_ways_1 := total_items
  let gamma_ways_3_items :=
    binomial (oreos + cookies) 3 +
    (oreos + cookies) * (oreos + cookies - 1) +
    (oreos + cookies)
  let case2 := alpha_ways_1 * gamma_ways_3_items
  
  case1 + case2 = 2100 := by
    sorry

end alpha_gamma_shopping_ways_l232_232727


namespace sum_seven_l232_232039

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ)

axiom a2 : a 2 = 3
axiom a6 : a 6 = 11
axiom arithmetic_seq : arithmetic_sequence a
axiom sum_of_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_seven : S 7 = 49 :=
sorry

end sum_seven_l232_232039


namespace range_of_set_of_three_numbers_l232_232267

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l232_232267


namespace expression_equality_l232_232314

theorem expression_equality (a b c : ℝ) : a * (a + b - c) = a^2 + a * b - a * c :=
by
  sorry

end expression_equality_l232_232314


namespace average_last_4_matches_l232_232200

theorem average_last_4_matches (avg_10: ℝ) (avg_6: ℝ) (total_matches: ℕ) (first_matches: ℕ) :
  avg_10 = 38.9 → avg_6 = 42 → total_matches = 10 → first_matches = 6 → 
  (avg_10 * total_matches - avg_6 * first_matches) / (total_matches - first_matches) = 34.25 :=
by 
  intros h1 h2 h3 h4
  sorry

end average_last_4_matches_l232_232200


namespace range_of_numbers_is_six_l232_232250

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l232_232250


namespace ranking_of_ABC_l232_232324

-- Define the ranking type
inductive Rank
| first
| second
| third

-- Define types for people
inductive Person
| A
| B
| C

open Rank Person

-- Alias for ranking of each person
def ranking := Person → Rank

-- Define the conditions
def A_statement (r : ranking) : Prop := r A ≠ first
def B_statement (r : ranking) : Prop := A_statement r ≠ false
def C_statement (r : ranking) : Prop := r C ≠ third

def B_lied : Prop := true
def C_told_truth : Prop := true

-- The equivalent problem, asked to prove the final result
theorem ranking_of_ABC (r : ranking) : 
  (B_lied ∧ C_told_truth ∧ B_statement r = false ∧ C_statement r = true) → 
  (r A = first ∧ r B = third ∧ r C = second) :=
sorry

end ranking_of_ABC_l232_232324


namespace laboratory_painting_area_laboratory_paint_needed_l232_232716

section
variable (l w h excluded_area : ℝ)
variable (paint_per_sqm : ℝ)

def painting_area (l w h excluded_area : ℝ) : ℝ :=
  let total_area := (l * w + w * h + h * l) * 2 - (l * w)
  total_area - excluded_area

def paint_needed (painting_area paint_per_sqm : ℝ) : ℝ :=
  painting_area * paint_per_sqm

theorem laboratory_painting_area :
  painting_area 12 8 6 28.4 = 307.6 :=
by
  simp [painting_area, *]
  norm_num

theorem laboratory_paint_needed :
  paint_needed 307.6 0.2 = 61.52 :=
by
  simp [paint_needed, *]
  norm_num

end

end laboratory_painting_area_laboratory_paint_needed_l232_232716


namespace sin_over_sin_l232_232018

theorem sin_over_sin (a : Real) (h_cos : Real.cos (Real.pi / 4 - a) = 12 / 13)
  (h_quadrant : 0 < Real.pi / 4 - a ∧ Real.pi / 4 - a < Real.pi / 2) :
  Real.sin (Real.pi / 2 - 2 * a) / Real.sin (Real.pi / 4 + a) = 119 / 144 := by
sorry

end sin_over_sin_l232_232018


namespace reading_time_difference_l232_232214

theorem reading_time_difference
  (tristan_speed : ℕ := 120)
  (ella_speed : ℕ := 40)
  (book_pages : ℕ := 360) :
  let tristan_time := book_pages / tristan_speed
  let ella_time := book_pages / ella_speed
  let time_difference_hours := ella_time - tristan_time
  let time_difference_minutes := time_difference_hours * 60
  time_difference_minutes = 360 :=
by
  sorry

end reading_time_difference_l232_232214


namespace range_of_set_l232_232262

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l232_232262


namespace candle_height_problem_l232_232423

/-- Define the height functions of the two candles. -/
def h1 (t : ℚ) : ℚ := 1 - t / 5
def h2 (t : ℚ) : ℚ := 1 - t / 4

/-- The main theorem stating the time t when the first candle is three times the height of the second candle. -/
theorem candle_height_problem : 
  (∀ t : ℚ, h1 t = 3 * h2 t) → t = (40 : ℚ) / 11 :=
by
  sorry

end candle_height_problem_l232_232423


namespace problem_solution_l232_232622

def p : Prop := ∀ x : ℝ, |x| ≥ 0
def q : Prop := ∃ x : ℝ, x = 2 ∧ x + 2 = 0

theorem problem_solution : p ∧ ¬q :=
by
  -- Here we would provide the proof to show that p ∧ ¬q is true
  sorry

end problem_solution_l232_232622


namespace exam_papers_count_l232_232583

theorem exam_papers_count (F x : ℝ) :
  (∀ n : ℕ, n = 5) →    -- condition 1: equivalence of n to proportions count
  (6 * x + 7 * x + 8 * x + 9 * x + 10 * x = 40 * x) →    -- condition 2: sum of proportions
  (40 * x = 0.60 * n * F) →   -- condition 3: student obtained 60% of total marks
  (7 * x > 0.50 * F ∧ 8 * x > 0.50 * F ∧ 9 * x > 0.50 * F ∧ 10 * x > 0.50 * F ∧ 6 * x ≤ 0.50 * F) →  -- condition 4: more than 50% in 4 papers
  ∃ n : ℕ, n = 5 :=    -- prove: number of papers is 5
sorry

end exam_papers_count_l232_232583


namespace necessary_and_sufficient_condition_l232_232139

open Classical

noncomputable def f (x a : ℝ) := x + a / x

theorem necessary_and_sufficient_condition
  (a : ℝ) :
  (∀ x : ℝ, x > 0 → f x a ≥ 2) ↔ (a ≥ 1) :=
by
  sorry

end necessary_and_sufficient_condition_l232_232139


namespace joan_total_socks_l232_232646

theorem joan_total_socks (n : ℕ) (h1 : n / 3 = 60) : n = 180 :=
by
  -- Proof goes here
  sorry

end joan_total_socks_l232_232646


namespace friend_saves_per_week_l232_232697

theorem friend_saves_per_week (x : ℕ) : 
  160 + 7 * 25 = 210 + x * 25 → x = 5 := 
by 
  sorry

end friend_saves_per_week_l232_232697


namespace Mitzi_score_l232_232352

-- Definitions based on the conditions
def Gretchen_score : ℕ := 120
def Beth_score : ℕ := 85
def average_score (total_score : ℕ) (num_bowlers : ℕ) : ℕ := total_score / num_bowlers

-- Theorem stating that Mitzi's bowling score is 113
theorem Mitzi_score (m : ℕ) (h : average_score (Gretchen_score + m + Beth_score) 3 = 106) :
  m = 113 :=
by sorry

end Mitzi_score_l232_232352


namespace square_side_length_l232_232691

theorem square_side_length (π : ℝ) (s : ℝ) :
  (∃ r : ℝ, 100 = π * r^2) ∧ (4 * s = 100) → s = 25 := by
  sorry

end square_side_length_l232_232691


namespace tattoo_ratio_l232_232449

theorem tattoo_ratio (a j k : ℕ) (ha : a = 23) (hj : j = 10) (rel : a = k * j + 3) : a / j = 23 / 10 :=
by {
  -- Insert proof here
  sorry
}

end tattoo_ratio_l232_232449


namespace abs_p_minus_1_ge_2_l232_232848

theorem abs_p_minus_1_ge_2 (p : ℝ) (a : ℕ → ℝ) 
  (h₀ : a 0 = 1)
  (h₁ : a 1 = p)
  (h₂ : a 2 = p * (p - 1))
  (h₃ : ∀ n : ℕ, a (n + 3) = p * a (n + 2) - p * a (n + 1) + a n)
  (h₄ : ∀ n : ℕ, a n > 0)
  (h₅ : ∀ m n : ℕ, m ≥ n → a m * a n > a (m + 1) * a (n - 1)) :
  |p - 1| ≥ 2 :=
sorry

end abs_p_minus_1_ge_2_l232_232848


namespace A_intersect_B_eq_l232_232978

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x ≤ 1
def A_cap_B (x : ℝ) : Prop := x ∈ {y | A y} ∧ x ∈ {y | B y}

theorem A_intersect_B_eq (x : ℝ) : (A_cap_B x) ↔ (x ∈ Set.Ioc 0 1) :=
by
  sorry

end A_intersect_B_eq_l232_232978


namespace first_bakery_sacks_per_week_l232_232931

theorem first_bakery_sacks_per_week (x : ℕ) 
    (H1 : 4 * x + 4 * 4 + 4 * 12 = 72) : x = 2 :=
by 
  -- we will provide the proof here if needed
  sorry

end first_bakery_sacks_per_week_l232_232931


namespace probability_two_diff_colors_l232_232637

-- Defining the conditions based on part (a)
constant blue_chips : ℕ := 6
constant red_chips : ℕ := 5
constant yellow_chips : ℕ := 3
constant green_chips : ℕ := 2
constant total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

-- Defining the probability statement
noncomputable def prob_diff_colors : ℚ :=
  (6 / 16) * (10 / 16) + (5 / 16) * (11 / 16) + (3 / 16) * (13 / 16) + (2 / 16) * (14 / 16)

-- The goal is to prove that this probability is equal to 91/128
theorem probability_two_diff_colors : prob_diff_colors = 91 / 128 :=
by
  sorry -- Proof to be provided

end probability_two_diff_colors_l232_232637


namespace geometric_sequence_constant_l232_232160

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ)
    (h1 : ∀ n, a (n+1) = q * a n)
    (h2 : ∀ n, a n > 0)
    (h3 : (a 1 + a 3) * (a 5 + a 7) = 4 * (a 4) ^ 2) :
    ∀ n, a n = a 0 :=
by
  sorry

end geometric_sequence_constant_l232_232160


namespace value_bounds_of_expression_l232_232647

theorem value_bounds_of_expression
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (triangle_ineq1 : a + b > c)
  (triangle_ineq2 : a + c > b)
  (triangle_ineq3 : b + c > a)
  : 4 ≤ (a+b+c)^2 / (b*c) ∧ (a+b+c)^2 / (b*c) ≤ 9 := sorry

end value_bounds_of_expression_l232_232647


namespace chameleon_problem_l232_232048

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l232_232048


namespace current_speed_l232_232440

-- The main statement of our problem
theorem current_speed (v_with_current v_against_current c man_speed : ℝ) 
  (h1 : v_with_current = man_speed + c) 
  (h2 : v_against_current = man_speed - c) 
  (h_with : v_with_current = 15) 
  (h_against : v_against_current = 9.4) : 
  c = 2.8 :=
by
  sorry

end current_speed_l232_232440


namespace function_symmetric_about_point_l232_232407

theorem function_symmetric_about_point :
  ∃ x₀ y₀, (x₀, y₀) = (Real.pi / 3, 0) ∧ ∀ x y, y = Real.sin (2 * x + Real.pi / 3) →
    (Real.sin (2 * (2 * x₀ - x) + Real.pi / 3) = y) :=
sorry

end function_symmetric_about_point_l232_232407


namespace expected_worth_flip_l232_232714

/-- A biased coin lands on heads with probability 2/3 and on tails with probability 1/3.
Each heads flip gains $5, and each tails flip loses $9.
If three consecutive flips all result in tails, then an additional loss of $10 is applied.
Prove that the expected worth of a single coin flip is -1/27. -/
theorem expected_worth_flip :
  let P_heads := 2 / 3
  let P_tails := 1 / 3
  (P_heads * 5 + P_tails * -9) - (P_tails ^ 3 * 10) = -1 / 27 :=
by
  sorry

end expected_worth_flip_l232_232714


namespace range_of_a_l232_232346

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*x + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → f x a > 0) ↔ a > -3 := 
by sorry

end range_of_a_l232_232346


namespace common_difference_is_4_l232_232522

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Defining the arithmetic sequence {a_n}
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
variable (d : ℤ) (a4_a5_sum : a 4 + a 5 = 24) (S6_val : S 6 = 48)

-- Statement to prove: given the conditions, d = 4
theorem common_difference_is_4 (h_seq : is_arithmetic_sequence a d) :
  d = 4 := sorry

end common_difference_is_4_l232_232522


namespace max_consecutive_integers_sum_le_500_l232_232898

def consecutive_sum (n : ℕ) : ℕ :=
  -- Formula for sum starting from 3
  (n * (n + 1)) / 2 - 3

theorem max_consecutive_integers_sum_le_500 : ∃ n : ℕ, consecutive_sum n ≤ 500 ∧ ∀ m : ℕ, m > n → consecutive_sum m > 500 :=
by
  sorry

end max_consecutive_integers_sum_le_500_l232_232898


namespace percentage_reduction_is_58_perc_l232_232436

-- Define the conditions
def initial_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.7 * P
def increased_price (P : ℝ) : ℝ := 1.2 * (discount_price P)
def clearance_price (P : ℝ) : ℝ := 0.5 * (increased_price P)

-- The statement of the proof problem
theorem percentage_reduction_is_58_perc (P : ℝ) (h : P > 0) :
  (1 - (clearance_price P / initial_price P)) * 100 = 58 :=
by
  -- Proof omitted
  sorry

end percentage_reduction_is_58_perc_l232_232436


namespace expression_equals_one_l232_232736

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end expression_equals_one_l232_232736


namespace llesis_more_rice_l232_232857

theorem llesis_more_rice :
  let total_rice := 50
  let llesis_fraction := 7 / 10
  let llesis_rice := total_rice * llesis_fraction
  let everest_rice := total_rice - llesis_rice
  llesis_rice - everest_rice = 20 := by
    sorry

end llesis_more_rice_l232_232857


namespace cos_R_in_triangle_PQR_l232_232518

theorem cos_R_in_triangle_PQR
  (P Q R : ℝ) (hP : P = 90) (hQ : Real.sin Q = 3/5)
  (h_sum : P + Q + R = 180) (h_PQ_comp : P + Q = 90) :
  Real.cos R = 3 / 5 := 
sorry

end cos_R_in_triangle_PQR_l232_232518


namespace geometric_sequence_sum_l232_232515

variable {a : ℕ → ℕ}

def is_geometric_sequence_with_common_product (k : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem geometric_sequence_sum :
  is_geometric_sequence_with_common_product 27 a →
  a 1 = 1 →
  a 2 = 3 →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 +
   a 11 + a 12 + a 13 + a 14 + a 15 + a 16 + a 17 + a 18) = 78 :=
by
  intros h_geom h_a1 h_a2
  sorry

end geometric_sequence_sum_l232_232515


namespace triangle_ABC_area_l232_232491

-- Define the vertices of the triangle
def A := (-4, 0)
def B := (24, 0)
def C := (0, 2)

-- Function to calculate the determinant, used for the area calculation
def det (x1 y1 x2 y2 x3 y3 : ℝ) :=
  x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)

-- Area calculation for triangle given vertices using determinant method
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * |det x1 y1 x2 y2 x3 y3|

-- The goal is to prove that the area of triangle ABC is 14
theorem triangle_ABC_area :
  triangle_area (-4) 0 24 0 0 2 = 14 := sorry

end triangle_ABC_area_l232_232491


namespace converse_angles_complements_l232_232202

theorem converse_angles_complements (α β : ℝ) (h : ∀γ : ℝ, α + γ = 90 ∧ β + γ = 90 → α = β) : 
  ∀ δ, α + δ = 90 ∧ β + δ = 90 → α = β :=
by 
  sorry

end converse_angles_complements_l232_232202


namespace calculate_M_minus_m_l232_232603

def total_students : ℕ := 2001
def students_studying_spanish (S : ℕ) : Prop := 1601 ≤ S ∧ S ≤ 1700
def students_studying_french (F : ℕ) : Prop := 601 ≤ F ∧ F ≤ 800
def studying_both_languages_lower_bound (S F m : ℕ) : Prop := S + F - m = total_students
def studying_both_languages_upper_bound (S F M : ℕ) : Prop := S + F - M = total_students

theorem calculate_M_minus_m :
  ∀ (S F m M : ℕ),
    students_studying_spanish S →
    students_studying_french F →
    studying_both_languages_lower_bound S F m →
    studying_both_languages_upper_bound S F M →
    S = 1601 ∨ S = 1700 →
    F = 601 ∨ F = 800 →
    M - m = 298 :=
by
  intros S F m M hs hf hl hb Hs Hf
  sorry

end calculate_M_minus_m_l232_232603


namespace prob_sum_equals_15_is_0_l232_232177

theorem prob_sum_equals_15_is_0 (coin1 coin2 : ℕ) (die_min die_max : ℕ) (age : ℕ)
  (h1 : coin1 = 5) (h2 : coin2 = 15) (h3 : die_min = 1) (h4 : die_max = 6) (h5 : age = 15) :
  ((coin1 = 5 ∨ coin2 = 15) → die_min ≤ ((if coin1 = 5 then 5 else 15) + (die_max - die_min + 1)) ∧ 
   (die_min ≤ 6) ∧ 6 ≤ die_max) → 
  0 = 0 :=
by
  sorry

end prob_sum_equals_15_is_0_l232_232177


namespace magnitude_of_angle_B_value_of_k_l232_232374

-- Define the conditions and corresponding proofs

variable {a b c : ℝ}
variable {A B C : ℝ} -- Angles in the triangle
variable (k : ℝ) -- Define k
variable (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) -- Given condition for part 1
variable (h2 : (A + B + C) = Real.pi) -- Angle sum in triangle
variable (h3 : k > 1) -- Condition for part 2
variable (m_dot_n_max : ∀ (t : ℝ), 4 * k * t + Real.cos (2 * Real.arcsin t) = 5) -- Given condition for part 2

-- Proofs Required

theorem magnitude_of_angle_B (hA : 0 < A ∧ A < Real.pi) : B = Real.pi / 3 :=
by 
  sorry -- proof to be completed

theorem value_of_k : k = 3 / 2 :=
by 
  sorry -- proof to be completed

end magnitude_of_angle_B_value_of_k_l232_232374


namespace increase_by_thirteen_possible_l232_232042

-- Define the main condition which states the reduction of the original product
def product_increase_by_thirteen (a : Fin 7 → ℕ) : Prop :=
  let P := (List.range 7).map (fun i => a ⟨i, sorry⟩) |>.prod
  let Q := (List.range 7).map (fun i => a ⟨i, sorry⟩ - 3) |>.prod
  Q = 13 * P

-- State the theorem to be proved
theorem increase_by_thirteen_possible : ∃ (a : Fin 7 → ℕ), product_increase_by_thirteen a :=
sorry

end increase_by_thirteen_possible_l232_232042


namespace number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l232_232147

def is_consonant (c : Char) : Prop :=
  c = 'B' ∨ c = 'C' ∨ c = 'D' ∨ c = 'F'

def count_5_letter_words_with_at_least_one_consonant : Nat :=
  let total_words := 6 ^ 5
  let vowel_words := 2 ^ 5
  total_words - vowel_words

theorem number_of_5_letter_words_with_at_least_one_consonant_equals_7744 :
  count_5_letter_words_with_at_least_one_consonant = 7744 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l232_232147


namespace difference_of_squares_l232_232364

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l232_232364


namespace num_factors_of_60_l232_232804

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l232_232804


namespace difference_of_squares_not_2018_l232_232455

theorem difference_of_squares_not_2018 (a b : ℕ) : a^2 - b^2 ≠ 2018 :=
by
  sorry

end difference_of_squares_not_2018_l232_232455


namespace student_contribution_is_4_l232_232517

-- Definitions based on the conditions in the problem statement
def total_contribution := 90
def available_class_funds := 14
def number_of_students := 19

-- The theorem statement to be proven
theorem student_contribution_is_4 : 
  (total_contribution - available_class_funds) / number_of_students = 4 :=
by
  sorry  -- Proof is not required as per the instructions

end student_contribution_is_4_l232_232517


namespace winning_candidate_percentage_l232_232164

theorem winning_candidate_percentage (P: ℝ) (majority diff votes totalVotes : ℝ)
    (h1 : majority = 184)
    (h2 : totalVotes = 460)
    (h3 : diff = P * totalVotes / 100 - (100 - P) * totalVotes / 100)
    (h4 : majority = diff) : P = 70 :=
by
  sorry

end winning_candidate_percentage_l232_232164


namespace range_of_set_of_three_numbers_l232_232264

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l232_232264


namespace part1_part2_l232_232524

-- Given Definitions
variable (p : ℕ) [hp : Fact (p > 3)] [prime : Fact (Nat.Prime p)]
variable (A_l : ℕ → ℕ)

-- Assertions to Prove
theorem part1 (l : ℕ) (hl : 1 ≤ l ∧ l ≤ p - 2) : A_l l % p = 0 :=
sorry

theorem part2 (l : ℕ) (hl : 1 < l ∧ l < p ∧ l % 2 = 1) : A_l l % (p * p) = 0 :=
sorry

end part1_part2_l232_232524


namespace remainder_is_zero_l232_232159

theorem remainder_is_zero (D R r : ℕ) (h1 : D = 12 * 42 + R)
                           (h2 : D = 21 * 24 + r)
                           (h3 : r < 21) :
                           r = 0 :=
by 
  sorry

end remainder_is_zero_l232_232159


namespace ellipse_condition_l232_232370

theorem ellipse_condition (m : ℝ) :
  (m > 0) ∧ (2 * m - 1 > 0) ∧ (m ≠ 2 * m - 1) ↔ (m > 1/2) ∧ (m ≠ 1) :=
by
  sorry

end ellipse_condition_l232_232370


namespace sufficient_but_not_necessary_condition_l232_232087

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 0 → x^2 > 0) ∧ ¬(x^2 > 0 → x > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l232_232087


namespace ratio_pen_to_pencil_l232_232441

-- Define the costs
def cost_of_pencil (P : ℝ) : ℝ := P
def cost_of_pen (P : ℝ) : ℝ := 4 * P
def total_cost (P : ℝ) : ℝ := cost_of_pencil P + cost_of_pen P

-- The proof that the total cost of the pen and pencil is $6 given the provided ratio
theorem ratio_pen_to_pencil (P : ℝ) (h_total_cost : total_cost P = 6) (h_pen_cost : cost_of_pen P = 4) :
  cost_of_pen P / cost_of_pencil P = 4 :=
by
  -- Proof skipped
  sorry

end ratio_pen_to_pencil_l232_232441


namespace find_number_l232_232572

theorem find_number (x : ℝ) (h : x / 0.07 = 700) : x = 49 :=
sorry

end find_number_l232_232572


namespace inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l232_232021

-- Definitions for the conditions
def inside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 < r^2 ∧ (M.1 ≠ 0 ∨ M.2 ≠ 0)

def on_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 = r^2

def outside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 > r^2

def line_l_intersects_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 < r^2 ∨ M.1 * M.1 + M.2 * M.2 = r^2

def line_l_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 = r^2

def line_l_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 > r^2

-- Propositions
theorem inside_circle_implies_line_intersects_circle (M : ℝ × ℝ) (r : ℝ) : 
  inside_circle M r → line_l_intersects_circle M r := 
sorry

theorem on_circle_implies_line_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) :
  on_circle M r → line_l_tangent_to_circle M r :=
sorry

theorem outside_circle_implies_line_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) :
  outside_circle M r → line_l_does_not_intersect_circle M r :=
sorry

end inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l232_232021


namespace find_x_l232_232031

theorem find_x (x y : ℝ) (h₁ : x - y = 10) (h₂ : x + y = 14) : x = 12 :=
by
  sorry

end find_x_l232_232031


namespace distribute_a_eq_l232_232317

variable (a b c : ℝ)

theorem distribute_a_eq : a * (a + b - c) = a^2 + a * b - a * c := 
sorry

end distribute_a_eq_l232_232317


namespace necessary_and_sufficient_condition_l232_232135

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2 ≥ 2 * a * b) ↔ (a/b + b/a ≥ 2) :=
sorry

end necessary_and_sufficient_condition_l232_232135


namespace Peggy_needs_to_add_stamps_l232_232311

theorem Peggy_needs_to_add_stamps :
  ∀ (Peggy_stamps Bert_stamps Ernie_stamps : ℕ),
  Peggy_stamps = 75 →
  Ernie_stamps = 3 * Peggy_stamps →
  Bert_stamps = 4 * Ernie_stamps →
  Bert_stamps - Peggy_stamps = 825 :=
by
  intros Peggy_stamps Bert_stamps Ernie_stamps hPeggy hErnie hBert
  sorry

end Peggy_needs_to_add_stamps_l232_232311


namespace weight_of_11th_person_l232_232681

theorem weight_of_11th_person
  (n : ℕ) (avg1 avg2 : ℝ)
  (hn : n = 10)
  (havg1 : avg1 = 165)
  (havg2 : avg2 = 170)
  (W : ℝ) (X : ℝ)
  (hw : W = n * avg1)
  (havg2_eq : (W + X) / (n + 1) = avg2) :
  X = 220 :=
by
  sorry

end weight_of_11th_person_l232_232681


namespace minimum_value_f_range_of_m_l232_232347

noncomputable def f (x m : ℝ) := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem minimum_value_f (m : ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) : 
  if m ≤ 2 then f x m = 2 - m 
  else if m ≥ Real.exp 1 + 1 then f x m = Real.exp 1 - m - (m - 1) / Real.exp 1 
  else f x m = m - 2 - m * Real.log (m - 1) :=
sorry

theorem range_of_m (m : ℝ) :
  (m ≤ 2 ∧ ∀ x2 ∈ [-2, 0], ∃ x1 ∈ [Real.exp 1, Real.exp 2], f x1 m ≤ g x2) ↔
  (m ∈ [ (Real.exp 2 - Real.exp 1 + 1) / (Real.exp 1 + 1), 2 ]) :=
sorry

end minimum_value_f_range_of_m_l232_232347


namespace math_lovers_l232_232726

/-- The proof problem: 
Given 1256 students in total and the difference of 408 between students who like math and others,
prove that the number of students who like math is 424, given that students who like math are fewer than 500.
--/
theorem math_lovers (M O : ℕ) (h1 : M + O = 1256) (h2: O - M = 408) (h3 : M < 500) : M = 424 :=
by
  sorry

end math_lovers_l232_232726


namespace time_for_q_to_complete_work_alone_l232_232226

theorem time_for_q_to_complete_work_alone (P Q : ℝ) (h1 : (1 / P) + (1 / Q) = 1 / 40) (h2 : (20 / P) + (12 / Q) = 1) : Q = 64 / 3 :=
by
  sorry

end time_for_q_to_complete_work_alone_l232_232226


namespace green_chameleon_increase_l232_232050

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l232_232050


namespace center_of_circle_l232_232674

theorem center_of_circle (x y : ℝ) : 
  (x - 1) ^ 2 + (y + 1) ^ 2 = 4 ↔ (x^2 + y^2 - 2*x + 2*y - 2 = 0) :=
sorry

end center_of_circle_l232_232674


namespace tangent_parallel_to_BC_l232_232382

theorem tangent_parallel_to_BC {A B C D M X H P : Point} (h_isosceles : A = C) (h_BC_gt_AB : dist B C > dist A B)
  (h_midpoints_BCM : midpoint B C = D) (h_midpoints_AB : midpoint A B = M) 
  (h_BX_perp_AC : ⊥ X (line_through B C)) (h_XD_parallel_AB : parallel X D (line_through A B))
  (h_BX_inter_AD_H : intersect B X A D = H) (h_P_circumcircle_AHX : is_on_circumcircle P (triangle A H X)) :
  parallel (tangent_at A (circumcircle (triangle A M P))) (line_through B C) :=
by
  sorry

end tangent_parallel_to_BC_l232_232382


namespace max_value_of_expression_l232_232526

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 1) :
  x^4 * y^3 * z^2 ≤ 1024 / 14348907 :=
sorry

end max_value_of_expression_l232_232526


namespace find_base_of_denominator_l232_232832

theorem find_base_of_denominator 
  (some_base : ℕ)
  (h1 : (1/2)^16 * (1/81)^8 = 1 / some_base^16) : 
  some_base = 18 :=
sorry

end find_base_of_denominator_l232_232832


namespace birth_year_l232_232917

theorem birth_year (x : ℤ) (h : 1850 < x^2 - 10 - x ∧ 1849 ≤ x^2 - 10 - x ∧ x^2 - 10 - x ≤ 1880) : 
x^2 - 10 - x ≠ 1849 ∧ x^2 - 10 - x ≠ 1855 ∧ x^2 - 10 - x ≠ 1862 ∧ x^2 - 10 - x ≠ 1871 ∧ x^2 - 10 - x ≠ 1880 := 
sorry

end birth_year_l232_232917


namespace find_x_satisfying_floor_eq_l232_232950

theorem find_x_satisfying_floor_eq (x : ℝ) (hx: ⌊x⌋ * x = 152) : x = 38 / 3 :=
sorry

end find_x_satisfying_floor_eq_l232_232950


namespace find_ages_l232_232589

theorem find_ages (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 5) : x = 2 := 
sorry

end find_ages_l232_232589


namespace green_chameleon_increase_l232_232052

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l232_232052


namespace cl_mass_percentage_in_ccl4_l232_232770

noncomputable def mass_percentage_of_cl_in_ccl4 : ℝ :=
  let mass_C : ℝ := 12.01
  let mass_Cl : ℝ := 35.45
  let num_Cl : ℝ := 4
  let total_mass_Cl : ℝ := num_Cl * mass_Cl
  let total_mass_CCl4 : ℝ := mass_C + total_mass_Cl
  (total_mass_Cl / total_mass_CCl4) * 100

theorem cl_mass_percentage_in_ccl4 :
  abs (mass_percentage_of_cl_in_ccl4 - 92.19) < 0.01 := 
sorry

end cl_mass_percentage_in_ccl4_l232_232770


namespace always_positive_expression_l232_232972

variable (x a b : ℝ)

theorem always_positive_expression (h : ∀ x, (x - a)^2 + b > 0) : b > 0 :=
sorry

end always_positive_expression_l232_232972


namespace average_of_scores_l232_232563

theorem average_of_scores :
  let scores := [50, 60, 70, 80, 80]
  let total := 340
  let num_subjects := 5
  let average := total / num_subjects
  average = 68 :=
by
  sorry

end average_of_scores_l232_232563


namespace max_edges_intersected_by_plane_l232_232426

theorem max_edges_intersected_by_plane (p : ℕ) (h_pos : p > 0) : ℕ :=
  let vertices := 2 * p
  let base_edges := p
  let lateral_edges := p
  let total_edges := 3 * p
  total_edges

end max_edges_intersected_by_plane_l232_232426


namespace num_factors_of_60_l232_232826

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l232_232826


namespace intersection_of_A_and_B_l232_232156

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x > 0}
  let B := {x : ℝ | x^2 - 2*x - 3 < 0}
  (A ∩ B) = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l232_232156


namespace sum_of_floor_sqrt_1_to_25_l232_232747

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l232_232747


namespace leak_drain_time_l232_232241

theorem leak_drain_time (P L : ℝ) (h1 : P = 0.5) (h2 : (P - L) = (6 / 13)) :
    (1 / L) = 26 := by
  sorry

end leak_drain_time_l232_232241


namespace smallest_k_for_64k_greater_than_6_l232_232078

theorem smallest_k_for_64k_greater_than_6 : ∃ (k : ℕ), 64 ^ k > 6 ∧ ∀ m : ℕ, m < k → 64 ^ m ≤ 6 :=
by
  use 1
  sorry

end smallest_k_for_64k_greater_than_6_l232_232078


namespace sqrt_meaningful_range_l232_232981

theorem sqrt_meaningful_range {x : ℝ} (h : x - 1 ≥ 0) : x ≥ 1 :=
sorry

end sqrt_meaningful_range_l232_232981


namespace eq_has_infinite_solutions_l232_232122

theorem eq_has_infinite_solutions (b : ℤ) :
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by 
  sorry

end eq_has_infinite_solutions_l232_232122


namespace bruce_initial_money_l232_232934

-- Definitions of the conditions
def cost_crayons : ℕ := 5 * 5
def cost_books : ℕ := 10 * 5
def cost_calculators : ℕ := 3 * 5
def total_spent : ℕ := cost_crayons + cost_books + cost_calculators
def cost_bags : ℕ := 11 * 10
def initial_money : ℕ := total_spent + cost_bags

-- Theorem statement
theorem bruce_initial_money :
  initial_money = 200 := by
  sorry

end bruce_initial_money_l232_232934


namespace third_number_is_60_l232_232541

theorem third_number_is_60 (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 80 + 15) / 3 + 5 → x = 60 :=
by
  intro h
  sorry

end third_number_is_60_l232_232541


namespace eccentricity_of_given_hyperbola_l232_232618

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : b = 2 * a) : ℝ :=
  Real.sqrt (1 + (b * b) / (a * a))

theorem eccentricity_of_given_hyperbola (a b : ℝ) 
  (h_hyperbola : b = 2 * a)
  (h_asymptote : ∃ k, k = 2 ∧ ∀ x, y = k * x → ((y * a) = (b * x))) :
  hyperbola_eccentricity a b h_hyperbola = Real.sqrt 5 :=
by
  sorry

end eccentricity_of_given_hyperbola_l232_232618


namespace cube_edges_count_l232_232029

theorem cube_edges_count : (∀ (A : Type), A ≈ cube -> 12 edges = true :=
by
  sorry

end cube_edges_count_l232_232029


namespace numbers_are_perfect_squares_l232_232396

/-- Prove that the numbers 49, 4489, 444889, ... obtained by inserting 48 into the 
middle of the previous number are perfect squares. -/
theorem numbers_are_perfect_squares :
  ∀ n : ℕ, ∃ k : ℕ, (k ^ 2) = (Int.ofNat ((20 * (10 : ℕ) ^ n + 1) / 3)) :=
by
  sorry

end numbers_are_perfect_squares_l232_232396


namespace remainder_7_pow_2010_l232_232558

theorem remainder_7_pow_2010 :
  (7 ^ 2010) % 100 = 49 := 
by 
  sorry

end remainder_7_pow_2010_l232_232558


namespace trig_expr_evaluation_l232_232733

theorem trig_expr_evaluation :
    (1 - (1 / cos (Real.pi / 6))) * 
    (1 + (1 / sin (Real.pi / 3))) * 
    (1 - (1 / sin (Real.pi / 6))) * 
    (1 + (1 / cos (Real.pi / 3))) = 1 := 
by
  let cos_30 := cos (Real.pi / 6)
  let sin_60 := sin (Real.pi / 3)
  let sin_30 := sin (Real.pi / 6)
  let cos_60 := cos (Real.pi / 3)
  have h_cos_30 : cos_30 = (Real.sqrt 3) / 2 := sorry
  have h_sin_60 : sin_60 = (Real.sqrt 3) / 2 := sorry
  have h_sin_30 : sin_30 = 1 / 2 := sorry
  have h_cos_60 : cos_60 = 1 / 2 := sorry
  sorry

end trig_expr_evaluation_l232_232733


namespace side_of_square_is_25_l232_232694

theorem side_of_square_is_25 (area_of_circle : ℝ) (perimeter_of_square : ℝ) (h1 : area_of_circle = 100) (h2 : area_of_circle = perimeter_of_square) : perimeter_of_square / 4 = 25 :=
by {
  -- Insert the steps here if necessary.
  sorry
}

end side_of_square_is_25_l232_232694


namespace sum_of_floor_sqrt_1_to_25_l232_232748

theorem sum_of_floor_sqrt_1_to_25 : 
  ( ∑ i in Finset.range 26, ∥Real.sqrt i∥ ) = 75 :=
by
  sorry

end sum_of_floor_sqrt_1_to_25_l232_232748


namespace find_k_l232_232772

noncomputable def k_val : ℝ := 19.2

theorem find_k (k : ℝ) :
  (4 + ∑' n : ℕ, (4 + n * k) / (5^(n + 1))) = 10 ↔ k = k_val :=
  sorry

end find_k_l232_232772


namespace rose_initial_rice_l232_232187

theorem rose_initial_rice : 
  ∀ (R : ℝ), (R - 9 / 10 * R - 1 / 4 * (R - 9 / 10 * R) = 0.75) → (R = 10) :=
by
  intro R h
  sorry

end rose_initial_rice_l232_232187


namespace arithmetic_sequence_sum_l232_232840

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 = 2) (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l232_232840


namespace calculate_expression_l232_232454

theorem calculate_expression (x : ℝ) : 2 * x^3 * (-3 * x)^2 = 18 * x^5 :=
by
  sorry

end calculate_expression_l232_232454


namespace eiffel_tower_height_l232_232667

-- Define the constants for heights and difference
def BurjKhalifa : ℝ := 830
def height_difference : ℝ := 506

-- The goal: Prove that the height of the Eiffel Tower is 324 m.
theorem eiffel_tower_height : BurjKhalifa - height_difference = 324 := 
by 
sorry

end eiffel_tower_height_l232_232667


namespace shopkeeper_discount_problem_l232_232443

theorem shopkeeper_discount_problem (CP SP_with_discount SP_without_discount Discount : ℝ)
  (h1 : SP_with_discount = CP + 0.273 * CP)
  (h2 : SP_without_discount = CP + 0.34 * CP) :
  Discount = SP_without_discount - SP_with_discount →
  (Discount / SP_without_discount) * 100 = 5 := 
sorry

end shopkeeper_discount_problem_l232_232443


namespace calculate_value_l232_232560

-- Given conditions
def n : ℝ := 2.25

-- Lean statement to express the proof problem
theorem calculate_value : (n / 3) * 12 = 9 := by
  -- Proof will be supplied here
  sorry

end calculate_value_l232_232560


namespace div_by_6_l232_232664

theorem div_by_6 (m : ℕ) : 6 ∣ (m^3 + 11 * m) :=
sorry

end div_by_6_l232_232664


namespace find_circle_equation_l232_232015

noncomputable def center_of_circle_on_line (x y : ℝ) : Prop := x - y + 1 = 0

noncomputable def point_on_circle (x_c y_c r x y : ℝ) : Prop := (x - x_c)^2 + (y - y_c)^2 = r^2

theorem find_circle_equation 
  (x_A y_A x_B y_B : ℝ)
  (hA : x_A = 1 ∧ y_A = 1)
  (hB : x_B = 2 ∧ y_B = -2)
  (h_center_on_line : ∃ x_c y_c, center_of_circle_on_line x_c y_c ∧ point_on_circle x_c y_c r 1 1 ∧ point_on_circle x_c y_c r 2 (-2))
  : ∃ (x_c y_c r : ℝ), (x + 3)^2 + (y + 2)^2 = 25 :=
by
  sorry

end find_circle_equation_l232_232015


namespace actual_speed_of_valentin_l232_232196

theorem actual_speed_of_valentin
  (claimed_speed : ℕ := 50) -- Claimed speed in m/min
  (wrong_meter : ℕ := 60)   -- Valentin thought 1 meter = 60 cm
  (wrong_minute : ℕ := 100) -- Valentin thought 1 minute = 100 seconds
  (correct_speed : ℕ := 18) -- The actual speed in m/min
  : (claimed_speed * wrong_meter / wrong_minute) * 60 / 100 = correct_speed :=
by
  sorry

end actual_speed_of_valentin_l232_232196


namespace terminating_decimal_multiples_l232_232333

theorem terminating_decimal_multiples :
  {m : ℕ | 1 ≤ m ∧ m ≤ 180 ∧ m % 3 = 0}.to_finset.card = 60 := 
sorry

end terminating_decimal_multiples_l232_232333


namespace graph_not_pass_first_quadrant_l232_232128

theorem graph_not_pass_first_quadrant (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ¬ (∃ x y : ℝ, y = a^x + b ∧ x > 0 ∧ y > 0) :=
sorry

end graph_not_pass_first_quadrant_l232_232128


namespace product_of_intersection_points_l232_232220

-- Define the two circles in the plane
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 8*y + 16 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y + 21 = 0

-- Define the intersection points property
def are_intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- The theorem to be proved
theorem product_of_intersection_points : ∃ x y : ℝ, are_intersection_points x y ∧ x * y = 12 := 
by
  sorry

end product_of_intersection_points_l232_232220


namespace value_of_expression_l232_232416

theorem value_of_expression : 30 - 5^2 = 5 := by
  sorry

end value_of_expression_l232_232416


namespace problem_k_star_k_star_k_l232_232759

def star (x y : ℝ) : ℝ := 2 * x^2 - y

theorem problem_k_star_k_star_k (k : ℝ) : star k (star k k) = k :=
by
  sorry

end problem_k_star_k_star_k_l232_232759


namespace allocation_schemes_correct_l232_232539

noncomputable def allocation_schemes : Nat :=
  let C (n k : Nat) : Nat := Nat.choose n k
  -- Calculate category 1: one school gets 1 professor, two get 2 professors each
  let category1 := C 3 1 * C 5 1 * C 4 2 * C 2 2 / 2
  -- Calculate category 2: one school gets 3 professors, two get 1 professor each
  let category2 := C 3 1 * C 5 3 * C 2 1 * C 1 1 / 2
  -- Total allocation ways
  let totalWays := 6 * (category1 + category2)
  totalWays

theorem allocation_schemes_correct : allocation_schemes = 900 := by
  sorry

end allocation_schemes_correct_l232_232539


namespace value_of_y_l232_232630

theorem value_of_y (x y : ℝ) (hx : x = 3) (h : x^(3 * y) = 9) : y = 2 / 3 := by
  sorry

end value_of_y_l232_232630


namespace depression_comparative_phrase_l232_232668

def correct_comparative_phrase (phrase : String) : Prop :=
  phrase = "twice as…as"

theorem depression_comparative_phrase :
  correct_comparative_phrase "twice as…as" :=
by
  sorry

end depression_comparative_phrase_l232_232668


namespace zach_cookies_total_l232_232433

theorem zach_cookies_total :
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  cookies_monday + cookies_tuesday + cookies_wednesday = 92 :=
by
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  sorry

end zach_cookies_total_l232_232433


namespace sum_of_floors_of_square_roots_l232_232750

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l232_232750


namespace fixed_point_coordinates_l232_232959

theorem fixed_point_coordinates (k : ℝ) (M : ℝ × ℝ) (h : ∀ k : ℝ, M.2 - 2 = k * (M.1 + 1)) :
  M = (-1, 2) :=
sorry

end fixed_point_coordinates_l232_232959


namespace retirement_year_2020_l232_232891

-- Given conditions
def femaleRetirementAge := 55
def initialRetirementYear (birthYear : ℕ) := birthYear + femaleRetirementAge
def delayedRetirementYear (baseYear additionalYears : ℕ) := baseYear + additionalYears

def postponementStep := 3
def delayStartYear := 2018
def retirementAgeIn2045 := 65
def retirementYear (birthYear : ℕ) : ℕ :=
  let originalRetirementYear := initialRetirementYear birthYear
  let delayYears := ((originalRetirementYear - delayStartYear) / postponementStep) + 1
  delayedRetirementYear originalRetirementYear delayYears

-- Main theorem to prove
theorem retirement_year_2020 : retirementYear 1964 = 2020 := sorry

end retirement_year_2020_l232_232891


namespace chocolate_bars_partial_boxes_l232_232155

-- Define the total number of bars for each type
def totalA : ℕ := 853845
def totalB : ℕ := 537896
def totalC : ℕ := 729763

-- Define the box capacities for each type
def capacityA : ℕ := 9
def capacityB : ℕ := 11
def capacityC : ℕ := 15

-- State the theorem we want to prove
theorem chocolate_bars_partial_boxes :
  totalA % capacityA = 4 ∧
  totalB % capacityB = 3 ∧
  totalC % capacityC = 8 :=
by
  -- Proof omitted for this task
  sorry

end chocolate_bars_partial_boxes_l232_232155


namespace ending_number_divisible_by_3_l232_232418

theorem ending_number_divisible_by_3 : 
∃ n : ℕ, (∀ k : ℕ, (10 + k * 3) ≤ n → (10 + k * 3) % 3 = 0) ∧ 
       (∃ c : ℕ, c = 12 ∧ (n - 10) / 3 + 1 = c) ∧ 
       n = 45 := 
sorry

end ending_number_divisible_by_3_l232_232418


namespace inequality_proof_l232_232480

theorem inequality_proof
  (a b c d e f : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (he : 0 < e)
  (hf : 0 < f)
  (h_condition : abs (Real.sqrt (a * b) - Real.sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := 
sorry

end inequality_proof_l232_232480


namespace proof_sum_of_ab_l232_232998

theorem proof_sum_of_ab :
  ∃ (a b : ℕ), a ≤ b ∧ 0 < a ∧ 0 < b ∧ a ^ 2 + b ^ 2 + 8 * a * b = 2010 ∧ a + b = 42 :=
sorry

end proof_sum_of_ab_l232_232998


namespace number_of_positive_factors_of_60_l232_232818

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l232_232818


namespace rectangle_problem_l232_232069

def rectangle_perimeter (L B : ℕ) : ℕ :=
  2 * (L + B)

theorem rectangle_problem (L B : ℕ) (h1 : L - B = 23) (h2 : L * B = 2520) : rectangle_perimeter L B = 206 := by
  sorry

end rectangle_problem_l232_232069


namespace curve_transformation_l232_232799

variable (x y x0 y0 : ℝ)

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -2], ![0, 1]]

def C (x0 y0 : ℝ) : Prop := (x0 - y0)^2 + y0^2 = 1

def transform (x0 y0 : ℝ) : ℝ × ℝ :=
  let x := 2 * x0 - 2 * y0
  let y := y0
  (x, y)

def C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

theorem curve_transformation :
  ∀ x0 y0, C x0 y0 → C' (2 * x0 - 2 * y0) y0 := sorry

end curve_transformation_l232_232799


namespace total_students_is_correct_l232_232180

-- Define the number of students in each class based on the conditions
def number_of_students_finley := 24
def number_of_students_johnson := (number_of_students_finley / 2) + 10
def number_of_students_garcia := 2 * number_of_students_johnson
def number_of_students_smith := number_of_students_finley / 3
def number_of_students_patel := (3 / 4) * (number_of_students_finley + number_of_students_johnson + number_of_students_garcia)

-- Define the total number of students in all five classes combined
def total_number_of_students := 
  number_of_students_finley + 
  number_of_students_johnson + 
  number_of_students_garcia +
  number_of_students_smith + 
  number_of_students_patel

-- The theorem statement to prove
theorem total_students_is_correct : total_number_of_students = 166 := by
  sorry

end total_students_is_correct_l232_232180


namespace range_of_set_l232_232289

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l232_232289


namespace marseille_hairs_l232_232577

theorem marseille_hairs (N : ℕ) (M : ℕ) (hN : N = 2000000) (hM : M = 300001) :
  ∃ k, k ≥ 7 ∧ ∃ b : ℕ, b ≤ M ∧ b > 0 ∧ ∀ i ≤ M, ∃ l : ℕ, l ≥ k → l ≤ (N / M + 1) :=
by
  sorry

end marseille_hairs_l232_232577


namespace scientific_notation_103M_l232_232469

theorem scientific_notation_103M : 103000000 = 1.03 * 10^8 := sorry

end scientific_notation_103M_l232_232469


namespace problem_equiv_l232_232001

theorem problem_equiv {a : ℤ} : (a^2 ≡ 9 [ZMOD 10]) ↔ (a ≡ 3 [ZMOD 10] ∨ a ≡ -3 [ZMOD 10] ∨ a ≡ 7 [ZMOD 10] ∨ a ≡ -7 [ZMOD 10]) :=
sorry

end problem_equiv_l232_232001


namespace triangle_max_distance_product_l232_232781

open Real

noncomputable def max_product_of_distances
  (a b c : ℝ) (P : {p : ℝ × ℝ // True}) : ℝ :=
  let h_a := 1 -- placeholder for actual distance calculation
  let h_b := 1 -- placeholder for actual distance calculation
  let h_c := 1 -- placeholder for actual distance calculation
  h_a * h_b * h_c

theorem triangle_max_distance_product
  (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5)
  (P : {p : ℝ × ℝ // True}) :
  max_product_of_distances a b c P = (16/15 : ℝ) :=
sorry

end triangle_max_distance_product_l232_232781


namespace parabola_coefficients_l232_232537

theorem parabola_coefficients :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, (a * (x - 3)^2 + 2 = 0 → (x = 1) ∧ (a * (1 - 3)^2 + 2 = 0))
    ∧ (a = -1/2 ∧ b = 3 ∧ c = -5/2)) 
    ∧ (∀ x : ℝ, a * x^2 + b * x + c = - 1 / 2 * x^2 + 3 * x - 5 / 2) :=
sorry

end parabola_coefficients_l232_232537


namespace remainder_of_70_div_17_l232_232900

theorem remainder_of_70_div_17 : 70 % 17 = 2 :=
by
  sorry

end remainder_of_70_div_17_l232_232900


namespace find_coordinates_of_M_l232_232867

def point_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.2) = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.1) = d

theorem find_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_second_quadrant M ∧ distance_to_x_axis M 5 ∧ distance_to_y_axis M 3 ∧ M = (-3, 5) :=
by
  sorry

end find_coordinates_of_M_l232_232867


namespace problem1_problem2_l232_232090

noncomputable def circle_ast (a b : ℕ) : ℕ := sorry

axiom circle_ast_self (a : ℕ) : circle_ast a a = a
axiom circle_ast_zero (a : ℕ) : circle_ast a 0 = 2 * a
axiom circle_ast_add (a b c d : ℕ) : circle_ast a b + circle_ast c d = circle_ast (a + c) (b + d)

theorem problem1 : circle_ast (2 + 3) (0 + 3) = 7 := sorry

theorem problem2 : circle_ast 1024 48 = 2000 := sorry

end problem1_problem2_l232_232090


namespace grade_assignments_count_l232_232095

theorem grade_assignments_count (n : ℕ) (g : ℕ) (h : n = 15) (k : g = 4) : g^n = 1073741824 :=
by
  sorry

end grade_assignments_count_l232_232095


namespace range_of_set_l232_232248

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l232_232248


namespace math_problem_l232_232107

theorem math_problem
  (p q r s : ℕ)
  (hpq : p^3 = q^2)
  (hrs : r^4 = s^3)
  (hrp : r - p = 25) :
  s - q = 73 := by
  sorry

end math_problem_l232_232107


namespace children_on_playground_l232_232419

theorem children_on_playground (boys_soccer girls_soccer boys_swings girls_swings boys_snacks girls_snacks : ℕ)
(h1 : boys_soccer = 27) (h2 : girls_soccer = 35)
(h3 : boys_swings = 15) (h4 : girls_swings = 20)
(h5 : boys_snacks = 10) (h6 : girls_snacks = 5) :
boys_soccer + girls_soccer + boys_swings + girls_swings + boys_snacks + girls_snacks = 112 := by
  sorry

end children_on_playground_l232_232419


namespace percentage_increase_in_cellphone_pay_rate_l232_232447

theorem percentage_increase_in_cellphone_pay_rate
    (regular_rate : ℝ)
    (total_surveys : ℕ)
    (cellphone_surveys : ℕ)
    (total_earnings : ℝ)
    (regular_surveys : ℕ := total_surveys - cellphone_surveys)
    (higher_rate : ℝ := (total_earnings - (regular_surveys * regular_rate)) / cellphone_surveys)
    : regular_rate = 30 ∧ total_surveys = 100 ∧ cellphone_surveys = 50 ∧ total_earnings = 3300
    → ((higher_rate - regular_rate) / regular_rate) * 100 = 20 := by
  sorry

end percentage_increase_in_cellphone_pay_rate_l232_232447


namespace floor_sum_sqrt_25_l232_232754

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l232_232754


namespace no_real_roots_of_quad_eq_l232_232505

theorem no_real_roots_of_quad_eq (k : ℝ) :
  ∀ x : ℝ, ¬ (x^2 - 2*x - k = 0) ↔ k < -1 :=
by sorry

end no_real_roots_of_quad_eq_l232_232505


namespace car_total_distance_l232_232438

theorem car_total_distance (h1 h2 h3 : ℕ) :
  h1 = 180 → h2 = 160 → h3 = 220 → h1 + h2 + h3 = 560 :=
by
  intros h1_eq h2_eq h3_eq
  sorry

end car_total_distance_l232_232438


namespace part1_part2_l232_232839

-- Definition of points and given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Conditions for part 1
def A1 (a : ℝ) : Point := { x := -2, y := a + 1 }
def B1 (a : ℝ) : Point := { x := a - 1, y := 4 }

-- Definition for distance calculation
def distance (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)

-- Problem 1 Statement
theorem part1 (a : ℝ) (h : a = 3) : distance (A1 a) (B1 a) = 4 :=
by 
  sorry

-- Conditions for part 2
def C2 (b : ℝ) : Point := { x := b - 2, y := b }

-- Problem 2 Statement
theorem part2 (b : ℝ) (h : abs b = 1) :
  (C2 b = { x := -1, y := 1 } ∨ C2 b = { x := -3, y := -1 }) :=
by
  sorry

end part1_part2_l232_232839


namespace inequality_proof_l232_232481

theorem inequality_proof
  (a b c d e f : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (he : 0 < e)
  (hf : 0 < f)
  (h_condition : abs (Real.sqrt (a * b) - Real.sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := 
sorry

end inequality_proof_l232_232481


namespace number_of_grade2_students_l232_232205

theorem number_of_grade2_students (ratio1 ratio2 ratio3 : ℕ) (total_students : ℕ) (ratio_sum : ratio1 + ratio2 + ratio3 = 12)
  (total_sample_size : total_students = 240) : 
  total_students * ratio2 / (ratio1 + ratio2 + ratio3) = 80 :=
by
  have ratio1_val : ratio1 = 5 := sorry
  have ratio2_val : ratio2 = 4 := sorry
  have ratio3_val : ratio3 = 3 := sorry
  rw [ratio1_val, ratio2_val, ratio3_val] at ratio_sum
  rw [ratio1_val, ratio2_val, ratio3_val]
  exact sorry

end number_of_grade2_students_l232_232205


namespace number_of_children_on_bus_l232_232231

theorem number_of_children_on_bus (initial_children : ℕ) (additional_children : ℕ) (total_children : ℕ) 
  (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = 64 :=
by
  sorry

end number_of_children_on_bus_l232_232231


namespace interval_length_sum_l232_232953

theorem interval_length_sum (x : ℝ) : 
  (0 < x ∧ x < 2) ∧ (Real.sin x > 1/2) → 
  (\let I := setOf (λ x, (Real.sin x > 1/2) ∧ (0 < x ∧ x < 2)) in
   (∃ (a b : ℝ), a < b ∧ I = λ x, a < x ∧ x < b ∧ (Real.sin x > 1/2))) 
   ∧ abs (((5*Real.pi/6) - (Real.pi/6)) - 2.09) < 0.02 :=
by
  sorry

end interval_length_sum_l232_232953


namespace unique_three_digit_multiple_of_66_ending_in_4_l232_232493

theorem unique_three_digit_multiple_of_66_ending_in_4 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 66 = 0 ∧ n % 10 = 4 := sorry

end unique_three_digit_multiple_of_66_ending_in_4_l232_232493


namespace ratio_of_investments_l232_232677

theorem ratio_of_investments (P Q : ℝ)
  (h_ratio_profits : (20 * P) / (40 * Q) = 7 / 10) : P / Q = 7 / 5 := 
sorry

end ratio_of_investments_l232_232677


namespace inequality_proved_l232_232669

theorem inequality_proved (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_proved_l232_232669


namespace intersection_A_B_l232_232136

def setA : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, x^2)}
def setB : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, Real.sqrt x)}

theorem intersection_A_B :
  (setA ∩ setB) = {(0, 0), (1, 1)} := by
  sorry

end intersection_A_B_l232_232136


namespace ceil_evaluation_l232_232765

theorem ceil_evaluation : 
  (Int.ceil (4 * (8 - 1 / 3 : ℚ))) = 31 :=
by
  sorry

end ceil_evaluation_l232_232765


namespace Nils_has_300_geese_l232_232394

variables (A x k n : ℕ)

def condition1 (A x k n : ℕ) : Prop :=
  A = k * x * n

def condition2 (A x k n : ℕ) : Prop :=
  A = (k + 20) * x * (n - 50)

def condition3 (A x k n : ℕ) : Prop :=
  A = (k - 10) * x * (n + 100)

theorem Nils_has_300_geese (A x k n : ℕ) :
  condition1 A x k n →
  condition2 A x k n →
  condition3 A x k n →
  n = 300 :=
by
  intros h1 h2 h3
  sorry

end Nils_has_300_geese_l232_232394


namespace non_science_majors_percentage_l232_232514

-- Definitions of conditions
def women_percentage (class_size : ℝ) : ℝ := 0.6 * class_size
def men_percentage (class_size : ℝ) : ℝ := 0.4 * class_size

def women_science_majors (class_size : ℝ) : ℝ := 0.2 * women_percentage class_size
def men_science_majors (class_size : ℝ) : ℝ := 0.7 * men_percentage class_size

def total_science_majors (class_size : ℝ) : ℝ := women_science_majors class_size + men_science_majors class_size

-- Theorem to prove the percentage of the class that are non-science majors is 60%
theorem non_science_majors_percentage (class_size : ℝ) : total_science_majors class_size / class_size = 0.4 → (class_size - total_science_majors class_size) / class_size = 0.6 := 
by
  sorry

end non_science_majors_percentage_l232_232514


namespace loss_percentage_is_20_l232_232446

-- Define necessary conditions
def CP : ℕ := 2000
def gain_percent : ℕ := 6
def SP_new : ℕ := CP + ((gain_percent * CP) / 100)
def increase : ℕ := 520

-- Define the selling price condition
def SP : ℕ := SP_new - increase

-- Define the loss percentage condition
def loss_percent : ℕ := ((CP - SP) * 100) / CP

-- Prove the loss percentage is 20%
theorem loss_percentage_is_20 : loss_percent = 20 :=
by sorry

end loss_percentage_is_20_l232_232446


namespace sum_floor_sqrt_1_to_25_l232_232739

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l232_232739


namespace no_real_roots_iff_l232_232498

theorem no_real_roots_iff (k : ℝ) : (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_iff_l232_232498


namespace find_analytical_expression_of_f_l232_232975

-- Define the function f satisfying the condition
def f (x : ℝ) : ℝ := sorry

-- Lean 4 theorem statement
theorem find_analytical_expression_of_f :
  (∀ x : ℝ, f (x + 1) = x^2 + 2*x + 2) → (∀ x : ℝ, f x = x^2 + 1) :=
by
  -- The initial f definition and theorem statement are created
  -- The proof is omitted since the focus is on translating the problem
  sorry

end find_analytical_expression_of_f_l232_232975


namespace probability_not_next_to_each_other_l232_232417

theorem probability_not_next_to_each_other :
  let num_chairs := 10
  let valid_chairs := num_chairs - 1
  let total_ways := Nat.choose valid_chairs 2
  let next_to_ways := valid_chairs - 1
  let p_next_to := next_to_ways.toReal / total_ways.toReal
  let p_not_next_to := 1 - p_next_to
  p_not_next_to = (7 : ℚ) / 9 := 
by {
  -- Definitions for conditions
  let num_chairs := 10
  let valid_chairs := num_chairs - 1
  let total_ways := Nat.choose valid_chairs 2
  let next_to_ways := valid_chairs - 1
  let p_next_to := next_to_ways.toReal / total_ways.toReal

  -- Calculations
  have h1 : total_ways = 36 := by sorry
  have h2 : next_to_ways = 8 := by sorry
  have h3 : p_next_to = (2 : ℚ) / 9 := by sorry
  have h4 : p_not_next_to = 1 - (2 : ℚ) / 9 := by sorry

  -- Conclusion
  show p_not_next_to = (7 : ℚ) / 9 from by sorry
}  

end probability_not_next_to_each_other_l232_232417


namespace gray_area_l232_232892

-- Given conditions
def rect1_length : ℕ := 8
def rect1_width : ℕ := 10
def rect2_length : ℕ := 12
def rect2_width : ℕ := 9
def black_area : ℕ := 37

-- Define areas based on conditions
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width
def white_area : ℕ := area_rect1 - black_area

-- Theorem to prove the area of the gray part
theorem gray_area : area_rect2 - white_area = 65 :=
by
  sorry

end gray_area_l232_232892


namespace mrs_McGillicuddy_student_count_l232_232661

theorem mrs_McGillicuddy_student_count :
  let morning_registered := 25
  let morning_absent := 3
  let early_afternoon_registered := 24
  let early_afternoon_absent := 4
  let late_afternoon_registered := 30
  let late_afternoon_absent := 5
  let evening_registered := 35
  let evening_absent := 7
  let morning_present := morning_registered - morning_absent
  let early_afternoon_present := early_afternoon_registered - early_afternoon_absent
  let late_afternoon_present := late_afternoon_registered - late_afternoon_absent
  let evening_present := evening_registered - evening_absent
  let total_present := morning_present + early_afternoon_present + late_afternoon_present + evening_present
  total_present = 95 :=
by
  sorry

end mrs_McGillicuddy_student_count_l232_232661


namespace find_bc_div_a_l232_232142

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x + 1

variable (a b c : ℝ)

def satisfied (x : ℝ) : Prop := a * f x + b * f (x - c) = 1

theorem find_bc_div_a (ha : ∀ x, satisfied a b c x) : (b * Real.cos c / a) = -1 := 
by sorry

end find_bc_div_a_l232_232142


namespace teachers_no_conditions_percentage_l232_232925

theorem teachers_no_conditions_percentage :
  let total_teachers := 150
  let high_blood_pressure := 90
  let heart_trouble := 60
  let both_hbp_ht := 30
  let diabetes := 10
  let both_diabetes_ht := 5
  let both_diabetes_hbp := 8
  let all_three := 3

  let only_hbp := high_blood_pressure - both_hbp_ht - both_diabetes_hbp - all_three
  let only_ht := heart_trouble - both_hbp_ht - both_diabetes_ht - all_three
  let only_diabetes := diabetes - both_diabetes_hbp - both_diabetes_ht - all_three
  let both_hbp_ht_only := both_hbp_ht - all_three
  let both_hbp_diabetes_only := both_diabetes_hbp - all_three
  let both_ht_diabetes_only := both_diabetes_ht - all_three
  let any_condition := only_hbp + only_ht + only_diabetes + both_hbp_ht_only + both_hbp_diabetes_only + both_ht_diabetes_only + all_three
  let no_conditions := total_teachers - any_condition

  (no_conditions / total_teachers * 100) = 28 :=
by
  sorry

end teachers_no_conditions_percentage_l232_232925


namespace chameleon_problem_l232_232049

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l232_232049


namespace marathon_yards_l232_232093

theorem marathon_yards (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ)
  (total_miles : ℕ) (total_yards : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : extra_yards_per_marathon = 395) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 15) 
  (H5 : total_miles = num_marathons * miles_per_marathon + (num_marathons * extra_yards_per_marathon) / yards_per_mile)
  (H6 : total_yards = (num_marathons * extra_yards_per_marathon) % yards_per_mile)
  (H7 : 0 ≤ total_yards ∧ total_yards < yards_per_mile) 
  : total_yards = 645 :=
sorry

end marathon_yards_l232_232093


namespace min_tables_42_l232_232234

def min_tables_needed (total_people : ℕ) (table_sizes : List ℕ) : ℕ :=
  sorry

theorem min_tables_42 :
  min_tables_needed 42 [4, 6, 8] = 6 :=
sorry

end min_tables_42_l232_232234


namespace green_chameleon_increase_l232_232051

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l232_232051


namespace particle_at_k_l232_232195

-- Define q
def q (p : ℝ) := 1 - p

-- Define the probability of the particle being at position k at time n
noncomputable def P (n k : ℤ) (p : ℝ) : ℝ :=
  if (n - k) % 2 = 0 then
    let r := (n - k) / 2 in
    (Nat.choose n r) * p^(n - r) * (q p)^r
  else
    0

-- Main theorem statement
theorem particle_at_k (n k : ℤ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  P n k p = if (n - k) % 2 = 0 then
    let r := (n - k) / 2 in
    (Nat.choose n r : ℝ) * p^(n - r) * (1 - p)^r 
  else
    0 :=
sorry

end particle_at_k_l232_232195


namespace interval_length_l232_232952

theorem interval_length (x : ℝ) :
  (1/x > 1/2) ∧ (Real.sin x > 1/2) → (2 - Real.pi / 6 = 1.48) :=
by
  sorry

end interval_length_l232_232952


namespace dot_product_eq_neg29_l232_232320

def v := (3, -2)
def w := (-5, 7)

theorem dot_product_eq_neg29 : (v.1 * w.1 + v.2 * w.2) = -29 := 
by 
  -- this is where the detailed proof will occur
  sorry

end dot_product_eq_neg29_l232_232320


namespace weavers_problem_l232_232670

theorem weavers_problem 
  (W : ℕ) 
  (H1 : 1 = W / 4) 
  (H2 : 3.5 = 49 / 14) :
  W = 4 :=
by
  sorry

end weavers_problem_l232_232670


namespace angle_F_measure_l232_232167

-- Given conditions
def D := 74
def sum_of_angles (x E D : ℝ) := x + E + D = 180
def E_formula (x : ℝ) := 2 * x - 10

-- Proof problem statement in Lean 4
theorem angle_F_measure :
  ∃ x : ℝ, x = (116 / 3) ∧
    sum_of_angles x (E_formula x) D :=
sorry

end angle_F_measure_l232_232167


namespace complement_of_intersection_l232_232034

theorem complement_of_intersection (U M N : Set ℤ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 4}) (hN : N = {3, 4, 5}) :
   U \ (M ∩ N) = {1, 2, 3, 5} := by
   sorry

end complement_of_intersection_l232_232034


namespace unique_rectangles_l232_232783

theorem unique_rectangles (a b x y : ℝ) (h_dim : a < b) 
    (h_perimeter : 2 * (x + y) = a + b)
    (h_area : x * y = (a * b) / 2) : 
    (∃ x y : ℝ, (2 * (x + y) = a + b) ∧ (x * y = (a * b) / 2) ∧ (x < a) ∧ (y < b)) → 
    (∃! z w : ℝ, (2 * (z + w) = a + b) ∧ (z * y = (a * b) / 2) ∧ (z < a) ∧ (w < b)) :=
sorry

end unique_rectangles_l232_232783


namespace dan_spent_amount_l232_232598

-- Defining the prices of items
def candy_bar_price : ℝ := 7
def chocolate_price : ℝ := 6
def gum_price : ℝ := 3
def chips_price : ℝ := 4

-- Defining the discount and tax rates
def candy_bar_discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05

-- Defining the steps to calculate the total price including discount and tax
def total_before_discount_and_tax := candy_bar_price + chocolate_price + gum_price + chips_price
def candy_bar_discount := candy_bar_discount_rate * candy_bar_price
def candy_bar_after_discount := candy_bar_price - candy_bar_discount
def total_after_discount := candy_bar_after_discount + chocolate_price + gum_price + chips_price
def tax := tax_rate * total_after_discount
def total_with_discount_and_tax := total_after_discount + tax

theorem dan_spent_amount : total_with_discount_and_tax = 20.27 :=
by sorry

end dan_spent_amount_l232_232598


namespace person_age_in_1893_l232_232157

theorem person_age_in_1893 
    (x y : ℕ)
    (h1 : 0 ≤ x ∧ x < 10)
    (h2 : 0 ≤ y ∧ y < 10)
    (h3 : 1 + 8 + x + y = 93 - 10 * x - y) : 
    1893 - (1800 + 10 * x + y) = 24 :=
by
  sorry

end person_age_in_1893_l232_232157


namespace range_of_set_l232_232283

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l232_232283


namespace base_6_digit_divisibility_l232_232599

theorem base_6_digit_divisibility (d : ℕ) (h1 : d < 6) : ∃ t : ℤ, (655 + 42 * d) = 13 * t :=
by sorry

end base_6_digit_divisibility_l232_232599


namespace expand_polynomial_l232_232468

theorem expand_polynomial (x : ℝ) :
    (5*x^2 + 3*x - 7) * (4*x^3) = 20*x^5 + 12*x^4 - 28*x^3 :=
by
  sorry

end expand_polynomial_l232_232468


namespace minimum_overlap_l232_232037

variable (U : Finset ℕ) -- This is the set of all people surveyed
variable (B V : Finset ℕ) -- These are the sets of people who like Beethoven and Vivaldi respectively.

-- Given conditions:
axiom h_total : U.card = 120
axiom h_B : B.card = 95
axiom h_V : V.card = 80
axiom h_subset_B : B ⊆ U
axiom h_subset_V : V ⊆ U

-- Question to prove:
theorem minimum_overlap : (B ∩ V).card = 95 + 80 - 120 := by
  sorry

end minimum_overlap_l232_232037


namespace product_sequence_eq_l232_232937

theorem product_sequence_eq : (∏ k in Finset.range 501, (4 * (k + 1)) / ((4 * (k + 1)) + 4)) = (1 : ℚ) / 502 := 
sorry

end product_sequence_eq_l232_232937


namespace function_positive_for_x_gt_neg1_l232_232879

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (3*x^2 + 6*x + 9)

theorem function_positive_for_x_gt_neg1 : ∀ (x : ℝ), x > -1 → f x > 0.5 :=
by
  sorry

end function_positive_for_x_gt_neg1_l232_232879


namespace arithmetic_geometric_sequence_l232_232134

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the first term, common difference and positions of terms in geometric sequence
def a1 : ℤ := -8
def d : ℤ := 2
def a3 := arithmetic_sequence a1 d 2
def a4 := arithmetic_sequence a1 d 3

-- Conditions for the terms forming a geometric sequence
def geometric_condition (a b c : ℤ) : Prop :=
  b^2 = a * c

-- Statement to prove
theorem arithmetic_geometric_sequence :
  geometric_condition a1 a3 a4 → a1 = -8 :=
by
  intro h
  -- Proof can be filled in here
  sorry

end arithmetic_geometric_sequence_l232_232134


namespace div_by_3_l232_232170

theorem div_by_3 (a b : ℤ) : 
  (∃ (k : ℤ), a = 3 * k) ∨ 
  (∃ (k : ℤ), b = 3 * k) ∨ 
  (∃ (k : ℤ), a + b = 3 * k) ∨ 
  (∃ (k : ℤ), a - b = 3 * k) :=
sorry

end div_by_3_l232_232170


namespace sqrt_difference_l232_232313

theorem sqrt_difference : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := 
by 
  sorry

end sqrt_difference_l232_232313


namespace number_of_white_balls_l232_232639

-- Definition of the conditions
def total_balls : ℕ := 40
def prob_red : ℝ := 0.15
def prob_black : ℝ := 0.45
def prob_white := 1 - prob_red - prob_black

-- The statement that needs to be proved
theorem number_of_white_balls : (total_balls : ℝ) * prob_white = 16 :=
by
  sorry

end number_of_white_balls_l232_232639


namespace arithmetic_sum_l232_232632

variable {a : ℕ → ℝ}

def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sum :
  is_arithmetic_seq a →
  a 5 + a 6 + a 7 = 15 →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  intros
  sorry

end arithmetic_sum_l232_232632


namespace range_of_numbers_is_six_l232_232272

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l232_232272


namespace range_of_numbers_is_six_l232_232256

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l232_232256


namespace total_area_covered_by_strips_l232_232457

theorem total_area_covered_by_strips (L W : ℝ) (n : ℕ) (overlap_area : ℝ) (end_to_end_area : ℝ) :
  L = 15 → W = 1 → n = 4 → overlap_area = 15 → end_to_end_area = 30 → 
  (L * W * n - overlap_area + end_to_end_area) = 45 :=
by
  intros hL hW hn hoverlap hend_to_end
  sorry

end total_area_covered_by_strips_l232_232457


namespace odometer_problem_l232_232322

theorem odometer_problem (a b c : ℕ) (h₀ : a + b + c = 7) (h₁ : 1 ≤ a)
  (h₂ : a < 10) (h₃ : b < 10) (h₄ : c < 10) (h₅ : (c - a) % 20 = 0) : a^2 + b^2 + c^2 = 37 := 
  sorry

end odometer_problem_l232_232322


namespace g_2002_eq_1_l232_232797

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ := λ x => f x + 1 - x)

axiom f_one : f 1 = 1
axiom f_inequality_1 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

theorem g_2002_eq_1 : g 2002 = 1 := by
  sorry

end g_2002_eq_1_l232_232797


namespace number_of_vip_children_l232_232190

theorem number_of_vip_children (total_attendees children_percentage children_vip_percentage : ℕ) :
  total_attendees = 400 →
  children_percentage = 75 →
  children_vip_percentage = 20 →
  (total_attendees * children_percentage / 100) * children_vip_percentage / 100 = 60 :=
by
  intros h_total h_children_pct h_vip_pct
  sorry

end number_of_vip_children_l232_232190


namespace find_fraction_l232_232066

theorem find_fraction (c d : ℕ) (h1 : 435 = 2 * 100 + c * 10 + d) :
  (c + d) / 12 = 5 / 6 :=
by sorry

end find_fraction_l232_232066


namespace liam_savings_per_month_l232_232175

theorem liam_savings_per_month (trip_cost bill_cost left_after_bills : ℕ) 
                               (months_in_two_years : ℕ) (total_savings_per_month : ℕ) :
  trip_cost = 7000 →
  bill_cost = 3500 →
  left_after_bills = 8500 →
  months_in_two_years = 24 →
  total_savings_per_month = 19000 →
  total_savings_per_month / months_in_two_years = 79167 / 100 :=
by
  intros
  sorry

end liam_savings_per_month_l232_232175


namespace time_for_a_to_complete_one_round_l232_232082

theorem time_for_a_to_complete_one_round (T_a T_b : ℝ) 
  (h1 : 4 * T_a = 3 * T_b)
  (h2 : T_b = T_a + 10) : 
  T_a = 30 := by
  sorry

end time_for_a_to_complete_one_round_l232_232082


namespace deepak_present_age_l232_232883

theorem deepak_present_age (R D : ℕ) (h1 : R =  4 * D / 3) (h2 : R + 10 = 26) : D = 12 :=
by
  sorry

end deepak_present_age_l232_232883


namespace side_of_square_is_25_l232_232693

theorem side_of_square_is_25 (area_of_circle : ℝ) (perimeter_of_square : ℝ) (h1 : area_of_circle = 100) (h2 : area_of_circle = perimeter_of_square) : perimeter_of_square / 4 = 25 :=
by {
  -- Insert the steps here if necessary.
  sorry
}

end side_of_square_is_25_l232_232693


namespace problem1_problem2_l232_232349

theorem problem1 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := sorry

theorem problem2 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (A + C) / (2 * B + A) = 9 / 5 := sorry

end problem1_problem2_l232_232349


namespace problem_solution_l232_232357

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l232_232357


namespace cost_difference_is_360_l232_232644

def sailboat_cost_per_day : ℕ := 60
def ski_boat_cost_per_hour : ℕ := 80
def ken_days : ℕ := 2
def aldrich_hours_per_day : ℕ := 3
def aldrich_days : ℕ := 2

theorem cost_difference_is_360 :
  let ken_total_cost := sailboat_cost_per_day * ken_days
  let aldrich_total_cost_per_day := ski_boat_cost_per_hour * aldrich_hours_per_day
  let aldrich_total_cost := aldrich_total_cost_per_day * aldrich_days
  let cost_diff := aldrich_total_cost - ken_total_cost
  cost_diff = 360 :=
by
  sorry

end cost_difference_is_360_l232_232644


namespace b_arithmetic_sequence_max_S_n_l232_232138

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a m ≠ 0 → a n = a (n + 1) * a (m-1) / (a m)

axiom a_pos_terms : ∀ n, 0 < a n
axiom a11_eight : a 11 = 8
axiom b_log : ∀ n, b n = Real.log (a n) / Real.log 2
axiom b4_seventeen : b 4 = 17

-- Question I: Prove b_n is an arithmetic sequence with common difference -2
theorem b_arithmetic_sequence (d : ℝ) (h_d : d = (-2)) :
  ∃ d, ∀ n, b (n + 1) - b n = d :=
sorry

-- Question II: Find the maximum value of S_n
theorem max_S_n : ∃ n, S n = 144 :=
sorry

end b_arithmetic_sequence_max_S_n_l232_232138


namespace product_of_distinct_nonzero_real_numbers_l232_232341

variable {x y : ℝ}

theorem product_of_distinct_nonzero_real_numbers (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 4 / x = y + 4 / y) : x * y = 4 := 
sorry

end product_of_distinct_nonzero_real_numbers_l232_232341


namespace geometric_progression_fourth_term_l232_232405

theorem geometric_progression_fourth_term :
  let a1 := 3^(1/2)
  let a2 := 3^(1/3)
  let a3 := 3^(1/6)
  let r  := a3 / a2    -- Common ratio of the geometric sequence
  let a4 := a3 * r     -- Fourth term in the geometric sequence
  a4 = 1 := by
  sorry

end geometric_progression_fourth_term_l232_232405


namespace frank_eats_each_day_l232_232475

theorem frank_eats_each_day :
  ∀ (cookies_per_tray cookies_per_day days ted_eats remaining_cookies : ℕ),
  cookies_per_tray = 12 →
  cookies_per_day = 2 →
  days = 6 →
  ted_eats = 4 →
  remaining_cookies = 134 →
  (2 * cookies_per_tray * days) - (ted_eats + remaining_cookies) / days = 1 :=
  by
    intros cookies_per_tray cookies_per_day days ted_eats remaining_cookies ht hc hd hted hr
    sorry

end frank_eats_each_day_l232_232475


namespace max_consecutive_integers_sum_500_l232_232899

theorem max_consecutive_integers_sum_500 : ∀ S k max_n,
  (∀ (n : ℕ), S n = n * k + n * (n - 1) / 2) →
  (k = 3) →
  (∀ (n : ℕ), 2 * S n ≤ 1000) →
  max_n = 29 :=
by
  intros S k max_n S_def hk hineq
  sorry

end max_consecutive_integers_sum_500_l232_232899


namespace inequality_8xyz_l232_232665

theorem inequality_8xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) > 8 * x * y * z := 
  by sorry

end inequality_8xyz_l232_232665


namespace repeating_decimal_in_lowest_terms_l232_232325

def repeating_decimal_to_fraction (x : ℚ) (h : x = 6 + 182 / 999) : x = 6176 / 999 :=
by sorry

theorem repeating_decimal_in_lowest_terms : (6176, 999).gcd = 1 :=
by sorry

end repeating_decimal_in_lowest_terms_l232_232325


namespace initial_women_count_l232_232991

-- Let x be the initial number of women.
-- Let y be the initial number of men.

theorem initial_women_count (x y : ℕ) (h1 : y = 2 * (x - 15)) (h2 : (y - 45) * 5 = (x - 15)) :
  x = 40 :=
by
  -- sorry to skip the proof
  sorry

end initial_women_count_l232_232991


namespace minimum_value_expression_l232_232173

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) ≥ 24 :=
by
  sorry

end minimum_value_expression_l232_232173


namespace FriedChickenDinner_orders_count_l232_232424

-- Defining the number of pieces of chicken used by each type of order
def piecesChickenPasta := 2
def piecesBarbecueChicken := 3
def piecesFriedChickenDinner := 8

-- Defining the number of orders for Chicken Pasta and Barbecue Chicken
def numChickenPastaOrders := 6
def numBarbecueChickenOrders := 3

-- Defining the total pieces of chicken needed for all orders
def totalPiecesOfChickenNeeded := 37

-- Defining the number of pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPasta : Nat := piecesChickenPasta * numChickenPastaOrders
def piecesNeededBarbecueChicken : Nat := piecesBarbecueChicken * numBarbecueChickenOrders

-- Defining the total pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPastaAndBarbecue : Nat := piecesNeededChickenPasta + piecesNeededBarbecueChicken

-- Calculating the pieces of chicken needed for Fried Chicken Dinner orders
def piecesNeededFriedChickenDinner : Nat := totalPiecesOfChickenNeeded - piecesNeededChickenPastaAndBarbecue

-- Defining the number of Fried Chicken Dinner orders
def numFriedChickenDinnerOrders : Nat := piecesNeededFriedChickenDinner / piecesFriedChickenDinner

-- Proving Victor has 2 Fried Chicken Dinner orders
theorem FriedChickenDinner_orders_count : numFriedChickenDinnerOrders = 2 := by
  unfold numFriedChickenDinnerOrders
  unfold piecesNeededFriedChickenDinner
  unfold piecesNeededChickenPastaAndBarbecue
  unfold piecesNeededBarbecueChicken
  unfold piecesNeededChickenPasta
  unfold totalPiecesOfChickenNeeded
  unfold numBarbecueChickenOrders
  unfold piecesBarbecueChicken
  unfold numChickenPastaOrders
  unfold piecesChickenPasta
  sorry

end FriedChickenDinner_orders_count_l232_232424


namespace least_product_of_distinct_primes_greater_than_50_l232_232003

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  p ≠ q ∧ is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 : 
  ∃ p q, distinct_primes_greater_than_50 p q ∧ p * q = 3127 := 
sorry

end least_product_of_distinct_primes_greater_than_50_l232_232003


namespace value_of_a_l232_232484

theorem value_of_a (a : ℝ) : (|a| - 1 = 1) ∧ (a - 2 ≠ 0) → a = -2 :=
by
  sorry

end value_of_a_l232_232484


namespace find_a_b_find_extreme_values_l232_232023

-- Definitions based on the conditions in the problem
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 2 * b

-- The function f attains a maximum value of 2 at x = -1
def f_max_at_neg_1 (a b : ℝ) : Prop :=
  (∃ x : ℝ, x = -1 ∧ 
  (∀ y : ℝ, f x a b ≤ f y a b)) ∧ f (-1) a b = 2

-- Statement (1): Finding the values of a and b
theorem find_a_b : ∃ a b : ℝ, f_max_at_neg_1 a b ∧ a = 2 ∧ b = 1 :=
sorry

-- The function f with a=2 and b=1
def f_specific (x : ℝ) : ℝ := f x 2 1

-- Statement (2): Finding the extreme values of f(x) on the interval [-1, 1]
def extreme_values_on_interval : Prop :=
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_specific x ≤ 6 ∧ f_specific x ≥ 50/27) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 6) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 50/27)

theorem find_extreme_values : extreme_values_on_interval :=
sorry

end find_a_b_find_extreme_values_l232_232023


namespace formula1_correct_formula2_correct_formula3_correct_l232_232587

noncomputable def formula1 (n : ℕ) := (Real.sqrt 2 / 2) * (1 - (-1 : ℝ) ^ n)
noncomputable def formula2 (n : ℕ) := Real.sqrt (1 - (-1 : ℝ) ^ n)
noncomputable def formula3 (n : ℕ) := if (n % 2 = 1) then Real.sqrt 2 else 0

theorem formula1_correct (n : ℕ) : 
  (n % 2 = 1 → formula1 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula1 n = 0) := 
by
  sorry

theorem formula2_correct (n : ℕ) : 
  (n % 2 = 1 → formula2 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula2 n = 0) := 
by
  sorry
  
theorem formula3_correct (n : ℕ) : 
  (n % 2 = 1 → formula3 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula3 n = 0) := 
by
  sorry

end formula1_correct_formula2_correct_formula3_correct_l232_232587


namespace final_temperature_correct_l232_232728

-- Define the initial conditions
def initial_temperature : ℝ := 12
def decrease_per_hour : ℝ := 5
def time_duration : ℕ := 4

-- Define the expected final temperature
def expected_final_temperature : ℝ := -8

-- The theorem to prove that the final temperature after a given time is as expected
theorem final_temperature_correct :
  initial_temperature + (-decrease_per_hour * time_duration) = expected_final_temperature :=
by
  sorry

end final_temperature_correct_l232_232728


namespace fraction_value_l232_232671

theorem fraction_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 :=
by
  sorry

end fraction_value_l232_232671


namespace range_of_numbers_is_six_l232_232273

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l232_232273


namespace diameter_of_circular_field_l232_232473

theorem diameter_of_circular_field :
  ∀ (π : ℝ) (cost_per_meter total_cost circumference diameter : ℝ),
    π = Real.pi → 
    cost_per_meter = 1.50 → 
    total_cost = 94.24777960769379 → 
    circumference = total_cost / cost_per_meter →
    circumference = π * diameter →
    diameter = 20 := 
by
  intros π cost_per_meter total_cost circumference diameter hπ hcp ht cutoff_circ hcirc
  sorry

end diameter_of_circular_field_l232_232473


namespace tangent_parabola_line_l232_232633

theorem tangent_parabola_line (a : ℝ) :
  (∃ x0 : ℝ, ax0^2 + 3 = 2 * x0 + 1) ∧ (∀ x : ℝ, a * x^2 - 2 * x + 2 = 0 → x = x0) → a = 1/2 :=
by
  intro h
  sorry

end tangent_parabola_line_l232_232633


namespace find_a_of_inequality_solution_l232_232371

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 ↔ x^2 - a * x < 0) → a = 1 := 
by 
  sorry

end find_a_of_inequality_solution_l232_232371


namespace question_1_part_1_question_1_part_2_question_2_l232_232490

universe u

variables (U : Type u) [PartialOrder U]
noncomputable def A : Set ℝ := {x | (x - 2) * (x - 9) < 0}
noncomputable def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
noncomputable def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a }

theorem question_1_part_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} :=
sorry

theorem question_1_part_2 : B ∪ (Set.compl A) = {x | x ≤ 5 ∨ x ≥ 9} :=
sorry

theorem question_2 (a : ℝ) (h : C a ∪ (Set.compl B) = Set.univ) : a ≤ -3 :=
sorry

end question_1_part_1_question_1_part_2_question_2_l232_232490


namespace veronica_reroll_probability_is_correct_l232_232215

noncomputable def veronica_reroll_probability : ℚ :=
  let P := (5 : ℚ) / 54
  P

theorem veronica_reroll_probability_is_correct :
  veronica_reroll_probability = (5 : ℚ) / 54 := sorry

end veronica_reroll_probability_is_correct_l232_232215


namespace polynomial_symmetric_equiv_l232_232534

variable {R : Type*} [CommRing R]

def symmetric_about (P : R → R) (a b : R) : Prop :=
  ∀ x, P (2 * a - x) = 2 * b - P x

def polynomial_form (P : R → R) (a b : R) (Q : R → R) : Prop :=
  ∀ x, P x = b + (x - a) * Q ((x - a) * (x - a))

theorem polynomial_symmetric_equiv (P Q : R → R) (a b : R) :
  (symmetric_about P a b ↔ polynomial_form P a b Q) :=
sorry

end polynomial_symmetric_equiv_l232_232534


namespace cost_of_first_house_l232_232846

theorem cost_of_first_house (C : ℝ) (h₀ : 2 * C + C = 600000) : C = 200000 := by
  -- proof placeholder
  sorry

end cost_of_first_house_l232_232846


namespace quadratic_roots_l232_232788

-- Definitions based on the conditions provided.
def condition1 (x y : ℝ) : Prop := x^2 - 6 * x + 9 = -(abs (y - 1))

-- The main theorem we want to prove.
theorem quadratic_roots (x y : ℝ) (h : condition1 x y) : (a : ℝ) → (a - 3) * (a - 1) = a^2 - 4 * a + 3 :=
  by sorry

end quadratic_roots_l232_232788


namespace trains_cross_in_9_seconds_l232_232711

noncomputable def time_to_cross (length1 length2 : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (speed1 + speed2)

theorem trains_cross_in_9_seconds :
  time_to_cross 240 260.04 (120 * (5 / 18)) (80 * (5 / 18)) = 9 := 
by
  sorry

end trains_cross_in_9_seconds_l232_232711


namespace minimum_shirts_to_save_money_by_using_Acme_l232_232585

-- Define the cost functions for Acme and Gamma
def Acme_cost (x : ℕ) : ℕ := 60 + 8 * x
def Gamma_cost (x : ℕ) : ℕ := 12 * x

-- State the theorem to prove that for x = 16, Acme is cheaper than Gamma
theorem minimum_shirts_to_save_money_by_using_Acme : ∀ x ≥ 16, Acme_cost x < Gamma_cost x :=
by
  intros x hx
  sorry

end minimum_shirts_to_save_money_by_using_Acme_l232_232585


namespace range_of_set_l232_232281

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l232_232281


namespace cost_price_of_watch_l232_232927

theorem cost_price_of_watch (C : ℝ) 
  (h1 : ∃ (SP1 SP2 : ℝ), SP1 = 0.54 * C ∧ SP2 = 1.04 * C ∧ SP2 = SP1 + 140) : 
  C = 280 :=
by
  obtain ⟨SP1, SP2, H1, H2, H3⟩ := h1
  sorry

end cost_price_of_watch_l232_232927


namespace correct_calculation_l232_232102

theorem correct_calculation (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 :=
  sorry

end correct_calculation_l232_232102


namespace difference_of_squares_l232_232363

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l232_232363


namespace percentage_of_allowance_spent_l232_232179

noncomputable def amount_spent : ℝ := 14
noncomputable def amount_left : ℝ := 26
noncomputable def total_allowance : ℝ := amount_spent + amount_left

theorem percentage_of_allowance_spent :
  ((amount_spent / total_allowance) * 100) = 35 := 
by 
  sorry

end percentage_of_allowance_spent_l232_232179


namespace best_fit_model_l232_232992

theorem best_fit_model 
  (R2_model1 R2_model2 R2_model3 R2_model4 : ℝ)
  (h1 : R2_model1 = 0.976)
  (h2 : R2_model2 = 0.776)
  (h3 : R2_model3 = 0.076)
  (h4 : R2_model4 = 0.351) : 
  (R2_model1 > R2_model2) ∧ (R2_model1 > R2_model3) ∧ (R2_model1 > R2_model4) :=
by
  sorry

end best_fit_model_l232_232992


namespace volunteers_allocation_scheme_count_l232_232761

theorem volunteers_allocation_scheme_count :
  let volunteers := 6
  let groups_of_two := 2
  let groups_of_one := 2
  let pavilions := 4
  let calculate_combinations (n k : ℕ) := Nat.choose n k
  calculate_combinations volunteers 2 * calculate_combinations (volunteers - 2) 2 * 
  calculate_combinations pavilions 2 * Nat.factorial pavilions = 1080 := by
sorry

end volunteers_allocation_scheme_count_l232_232761


namespace sum_arithmetic_sequence_100_to_110_l232_232901

theorem sum_arithmetic_sequence_100_to_110 :
  let a := 100
  let l := 110
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 1155 := by
  sorry

end sum_arithmetic_sequence_100_to_110_l232_232901


namespace find_longer_diagonal_l232_232626

-- Define the necessary conditions
variables (d1 d2 : ℝ)
variable (A : ℝ)
axiom ratio_condition : d1 / d2 = 2 / 3
axiom area_condition : A = 12

-- Define the problem of finding the length of the longer diagonal
theorem find_longer_diagonal : ∃ (d : ℝ), d = d2 → d = 6 :=
by 
  sorry

end find_longer_diagonal_l232_232626


namespace paul_baseball_cards_l232_232662

-- Define the necessary variables and statements
variable {n : ℕ}

-- State the problem and the proof target
theorem paul_baseball_cards : ∃ k, k = 3 * n + 1 := sorry

end paul_baseball_cards_l232_232662


namespace jill_and_bob_payment_l232_232844

-- Definitions of the conditions
def price_of_first_house (X : ℝ) := X
def price_of_second_house (Y X : ℝ) := 2 * X

theorem jill_and_bob_payment :
  ∃ X, ∃ Y, Y = 2 * X ∧ X + Y = 600000 ∧ X = 200000 :=
by
  sorry

end jill_and_bob_payment_l232_232844


namespace proof_mn_eq_9_l232_232835

theorem proof_mn_eq_9 (m n : ℕ) (h1 : 2 * m + n = 8) (h2 : m - n = 1) : m^n = 9 :=
by {
  sorry 
}

end proof_mn_eq_9_l232_232835


namespace a_9_equals_18_l232_232017

def is_sequence_of_positive_integers (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, 0 < n → 0 < a n

def satisfies_recursive_relation (a : ℕ → ℕ) : Prop :=
∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q

theorem a_9_equals_18 (a : ℕ → ℕ)
  (H1 : is_sequence_of_positive_integers a)
  (H2 : satisfies_recursive_relation a)
  (H3 : a 2 = 4) : a 9 = 18 :=
sorry

end a_9_equals_18_l232_232017


namespace floor_sum_sqrt_25_l232_232753

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l232_232753


namespace introduce_people_no_three_same_acquaintances_l232_232435

theorem introduce_people_no_three_same_acquaintances (n : ℕ) :
  ∃ f : ℕ → ℕ, (∀ i, i < n → f i ≤ n - 1) ∧ (∀ i j k, i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k → ¬(f i = f j ∧ f j = f k)) := 
sorry

end introduce_people_no_three_same_acquaintances_l232_232435


namespace five_letter_words_with_consonant_l232_232149

theorem five_letter_words_with_consonant :
  let letters := { 'A', 'B', 'C', 'D', 'E', 'F' } in
  let consonants := { 'B', 'C', 'D', 'F' } in
  let vowels := { 'A', 'E' } in
  let total_5_letter_words := 6^5 in
  let total_vowel_only_words := 2^5 in
  total_5_letter_words - total_vowel_only_words = 7744 :=
by
  sorry

end five_letter_words_with_consonant_l232_232149


namespace shaded_area_is_one_third_l232_232719

noncomputable def fractional_shaded_area : ℕ → ℚ
| 0 => 1 / 4
| n + 1 => (1 / 4) * fractional_shaded_area n

theorem shaded_area_is_one_third : (∑' n, fractional_shaded_area n) = 1 / 3 := 
sorry

end shaded_area_is_one_third_l232_232719


namespace find_x_for_mean_l232_232084

theorem find_x_for_mean 
(x : ℝ) 
(h_mean : (3 + 11 + 7 + 9 + 15 + 13 + 8 + 19 + 17 + 21 + 14 + x) / 12 = 12) : 
x = 7 :=
sorry

end find_x_for_mean_l232_232084


namespace range_of_set_is_six_l232_232299

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l232_232299


namespace polynomial_root_triples_l232_232957

theorem polynomial_root_triples (a b c : ℝ) :
  (∀ x : ℝ, x > 0 → (x^4 + a * x^3 + b * x^2 + c * x + b = 0)) ↔ (a, b, c) = (-21, 112, -204) ∨ (a, b, c) = (-12, 48, -80) :=
by
  sorry

end polynomial_root_triples_l232_232957


namespace smallest_positive_period_of_f_range_of_f_in_interval_l232_232022

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem smallest_positive_period_of_f (a : ℝ) (h : f a (π / 3) = 0) :
  ∃ T : ℝ, T = 2 * π ∧ (∀ x, f a (x + T) = f a x) :=
sorry

theorem range_of_f_in_interval (a : ℝ) (h : f a (π / 3) = 0) :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2), -1 ≤ f a x ∧ f a x ≤ 2 :=
sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l232_232022


namespace incorrect_conclusions_l232_232465

theorem incorrect_conclusions (p q : ℝ) :
  (¬ ∀ x, x|x| + p*x + q = 0 → ¬ (num_real_roots x = 3)) ∧
  (¬ (∃ x, x|x| + p*x + q = 0) → false) ∧
  (¬ (x : ℝ) ((p^2 - 4*q < 0) → (¬∃ x, x|x| + p*x + q = 0))) ∧
  (p < 0 ∧ q > 0 → ¬ (num_real_roots (λ x, x|x| + p*x + q) = 3)) :=
begin
  sorry,
end

end incorrect_conclusions_l232_232465


namespace distribute_a_eq_l232_232316

variable (a b c : ℝ)

theorem distribute_a_eq : a * (a + b - c) = a^2 + a * b - a * c := 
sorry

end distribute_a_eq_l232_232316


namespace num_factors_60_l232_232810

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l232_232810


namespace Sandy_tokens_difference_l232_232188

theorem Sandy_tokens_difference :
  let total_tokens : ℕ := 1000000
  let siblings : ℕ := 4
  let Sandy_tokens : ℕ := total_tokens / 2
  let sibling_tokens : ℕ := Sandy_tokens / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  sorry

end Sandy_tokens_difference_l232_232188


namespace meal_cost_l232_232553

variable (s c p : ℝ)

axiom cond1 : 5 * s + 8 * c + p = 5.00
axiom cond2 : 7 * s + 12 * c + p = 7.20
axiom cond3 : 4 * s + 6 * c + 2 * p = 6.00

theorem meal_cost : s + c + p = 1.90 :=
by
  sorry

end meal_cost_l232_232553


namespace multiple_of_C_share_l232_232398

noncomputable def find_multiple (A B C : ℕ) (total : ℕ) (mult : ℕ) (h1 : 4 * A = mult * C) (h2 : 5 * B = mult * C) (h3 : A + B + C = total) : ℕ :=
  mult

theorem multiple_of_C_share (A B : ℕ) (h1 : 4 * A = 10 * 160) (h2 : 5 * B = 10 * 160) (h3 : A + B + 160 = 880) : find_multiple A B 160 880 10 h1 h2 h3 = 10 :=
by
  sorry

end multiple_of_C_share_l232_232398


namespace original_total_price_l232_232775

-- Definitions of the original prices
def original_price_candy_box : ℕ := 10
def original_price_soda : ℕ := 6
def original_price_chips : ℕ := 4
def original_price_chocolate_bar : ℕ := 2

-- Mathematical problem statement
theorem original_total_price :
  original_price_candy_box + original_price_soda + original_price_chips + original_price_chocolate_bar = 22 :=
by
  sorry

end original_total_price_l232_232775


namespace sum_first_10_terms_l232_232614

-- Define the general arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the conditions of the problem
def given_conditions (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = 2 * a 4 ∧ arithmetic_seq a d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Statement of the problem
theorem sum_first_10_terms (a : ℕ → ℤ) (d : ℤ) (S₁₀ : ℤ) :
  given_conditions a d →
  (S₁₀ = 20 ∨ S₁₀ = 110) :=
sorry

end sum_first_10_terms_l232_232614


namespace range_of_m_eq_l232_232204

theorem range_of_m_eq (m: ℝ) (x: ℝ) :
  (m+1 = 0 ∧ 4 > 0) ∨ 
  ((m + 1 > 0) ∧ ((m^2 - 2 * m - 3)^2 - 4 * (m + 1) * (-m + 3) < 0)) ↔ 
  (m ∈ Set.Icc (-1 : ℝ) 1 ∪ Set.Ico (1 : ℝ) 3) := 
sorry

end range_of_m_eq_l232_232204


namespace log_expansion_l232_232129

theorem log_expansion (a : ℝ) (h : a = Real.log 4 / Real.log 5) : Real.log 64 / Real.log 5 - 2 * (Real.log 20 / Real.log 5) = a - 2 :=
by
  sorry

end log_expansion_l232_232129


namespace avg_height_is_28_l232_232923

-- Define the height relationship between trees
def height_relation (a b : ℕ) := a = 2 * b ∨ a = b / 2

-- Given tree heights (partial information)
def height_tree_2 := 14
def height_tree_5 := 20

-- Define the tree heights variables
variables (height_tree_1 height_tree_3 height_tree_4 height_tree_6 : ℕ)

-- Conditions based on the given data and height relations
axiom h1 : height_relation height_tree_1 height_tree_2
axiom h2 : height_relation height_tree_2 height_tree_3
axiom h3 : height_relation height_tree_3 height_tree_4
axiom h4 : height_relation height_tree_4 height_tree_5
axiom h5 : height_relation height_tree_5 height_tree_6

-- Compute total and average height
def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4 + height_tree_5 + height_tree_6
def average_height := total_height / 6

-- Prove the average height is 28 meters
theorem avg_height_is_28 : average_height = 28 := by
  sorry

end avg_height_is_28_l232_232923


namespace total_savings_l232_232683

-- Definitions and Conditions
def thomas_monthly_savings : ℕ := 40
def joseph_saving_ratio : ℚ := 3 / 5
def saving_period_months : ℕ := 72

-- Problem Statement
theorem total_savings :
  let thomas_total := thomas_monthly_savings * saving_period_months
  let joseph_monthly_savings := thomas_monthly_savings * joseph_saving_ratio
  let joseph_total := joseph_monthly_savings * saving_period_months
  thomas_total + joseph_total = 4608 := 
by
  sorry

end total_savings_l232_232683


namespace distance_apart_after_two_hours_l232_232929

theorem distance_apart_after_two_hours :
  (Jay_walk_rate : ℝ) = 1 / 20 →
  (Paul_jog_rate : ℝ) = 3 / 40 →
  (time_duration : ℝ) = 2 * 60 →
  (distance_apart : ℝ) = 15 :=
by
  sorry

end distance_apart_after_two_hours_l232_232929


namespace shaded_area_l232_232312

/-- Prove that the shaded area of a shape formed by removing four right triangles of legs 2 from each corner of a 6 × 6 square is equal to 28 square units -/
theorem shaded_area (a b c d : ℕ) (square_side_length : ℕ) (triangle_leg_length : ℕ)
  (h1 : square_side_length = 6)
  (h2 : triangle_leg_length = 2)
  (h3 : a = 1)
  (h4 : b = 2)
  (h5 : c = b)
  (h6 : d = 4*a) : 
  a * square_side_length * square_side_length - d * (b * b / 2) = 28 := 
sorry

end shaded_area_l232_232312


namespace calculate_difference_square_l232_232361

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l232_232361


namespace a_1994_is_7_l232_232680

def f (m : ℕ) : ℕ := m % 10

def a (n : ℕ) : ℕ := f (2^(n + 1) - 1)

theorem a_1994_is_7 : a 1994 = 7 :=
by
  sorry

end a_1994_is_7_l232_232680


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l232_232935

theorem problem_1 (x y z : ℝ) (h : z = (x + y) / 2) : z = (x + y) / 2 :=
sorry

theorem problem_2 (x y w : ℝ) (h1 : w = x + y) : w = x + y :=
sorry

theorem problem_3 (x w y : ℝ) (h1 : w = x + y) (h2 : y = w - x) : y = w - x :=
sorry

theorem problem_4 (x z v : ℝ) (h1 : z = (x + y) / 2) (h2 : v = 2 * z) : v = 2 * (x + (x + y) / 2) :=
sorry

theorem problem_5 (x z u : ℝ) (h : u = - (x + z) / 5) : x + z + 5 * u = 0 :=
sorry

theorem problem_6 (y z t : ℝ) (h : t = (6 + y + z) / 2) : t = (6 + y + z) / 2 :=
sorry

theorem problem_7 (y z s : ℝ) (h : y + z + 4 * s - 10 = 0) : y + z + 4 * s - 10 = 0 :=
sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l232_232935


namespace monotone_on_interval_and_extreme_values_l232_232868

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem monotone_on_interval_and_extreme_values :
  (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2) → f x1 > f x2) ∧ (f 1 = 5 ∧ f 2 = 4) := 
by
  sorry

end monotone_on_interval_and_extreme_values_l232_232868


namespace problem_solution_l232_232354

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l232_232354


namespace new_person_weight_l232_232672

noncomputable def weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

theorem new_person_weight 
  (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) 
  (weight_eqn : weight_increase n avg_increase = new_weight - old_weight) : 
  new_weight = 87.5 :=
by
  have n := 9
  have avg_increase := 2.5
  have old_weight := 65
  have weight_increase := 9 * 2.5
  have weight_eqn := weight_increase = 87.5 - 65
  sorry

end new_person_weight_l232_232672


namespace range_of_set_l232_232259

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l232_232259


namespace selling_price_percentage_l232_232097

  variable (L : ℝ)  -- List price
  variable (C : ℝ)  -- Cost price after discount
  variable (M : ℝ)  -- Marked price
  variable (S : ℝ)  -- Selling price after discount

  -- Conditions
  def cost_price_condition (L : ℝ) : ℝ := 0.7 * L
  def profit_condition (C S : ℝ) : Prop := 0.75 * S = C
  def marked_price_condition (S M : ℝ) : Prop := 0.85 * M = S

  theorem selling_price_percentage (L : ℝ) (h1 : C = cost_price_condition L)
    (h2 : profit_condition C S) (h3 : marked_price_condition S M) :
    S = 0.9333 * L :=
  by
    -- This is where the proof would go
    sorry
  
end selling_price_percentage_l232_232097


namespace repeating_decimal_sum_num_denom_l232_232905

noncomputable def repeating_decimal_to_fraction (n d : ℕ) (rep : ℚ) : ℚ :=
(rep * (10^d) - rep) / ((10^d) - 1)

theorem repeating_decimal_sum_num_denom
  (x : ℚ)
  (h1 : x = repeating_decimal_to_fraction 45 2 0.45)
  (h2 : repeating_decimal_to_fraction 45 2 0.45 = 5/11) : 
  (5 + 11) = 16 :=
by 
  sorry

end repeating_decimal_sum_num_denom_l232_232905


namespace perimeter_of_rectangular_garden_l232_232579

theorem perimeter_of_rectangular_garden (L W : ℝ) (h : L + W = 28) : 2 * (L + W) = 56 :=
by sorry

end perimeter_of_rectangular_garden_l232_232579


namespace value_of_a4_l232_232643

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := n^2 - 3 * n - 4

-- State the main proof problem.
theorem value_of_a4 : a_n 4 = 0 := by
  sorry

end value_of_a4_l232_232643


namespace equal_saturdays_and_sundays_l232_232920

theorem equal_saturdays_and_sundays (start_day : ℕ) (h : start_day < 7) :
  ∃! d, (d < 7 ∧ ((d + 2) % 7 = 0 → (d = 5))) :=
by
  sorry

end equal_saturdays_and_sundays_l232_232920


namespace order_of_numbers_l232_232588

def base16_to_dec (s : String) : ℕ := sorry
def base6_to_dec (s : String) : ℕ := sorry
def base4_to_dec (s : String) : ℕ := sorry
def base2_to_dec (s : String) : ℕ := sorry

theorem order_of_numbers:
  let a := base16_to_dec "3E"
  let b := base6_to_dec "210"
  let c := base4_to_dec "1000"
  let d := base2_to_dec "111011"
  a = 62 ∧ b = 78 ∧ c = 64 ∧ d = 59 →
  b > c ∧ c > a ∧ a > d :=
by
  intros
  sorry

end order_of_numbers_l232_232588


namespace number_of_females_l232_232402

theorem number_of_females (total_people : ℕ) (avg_age_total : ℕ) 
  (avg_age_males : ℕ) (avg_age_females : ℕ) (females : ℕ) :
  total_people = 140 → avg_age_total = 24 →
  avg_age_males = 21 → avg_age_females = 28 → 
  females = 60 :=
by
  intros h1 h2 h3 h4
  -- Using the given conditions
  sorry

end number_of_females_l232_232402


namespace gabor_can_cross_l232_232993

open Real

-- Definitions based on conditions
def river_width : ℝ := 100
def total_island_perimeter : ℝ := 800
def banks_parallel : Prop := true

theorem gabor_can_cross (w : ℝ) (p : ℝ) (bp : Prop) : 
  w = river_width → 
  p = total_island_perimeter → 
  bp = banks_parallel → 
  ∃ d : ℝ, d ≤ 300 := 
by
  sorry

end gabor_can_cross_l232_232993


namespace sum_of_remainders_l232_232080

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  sorry

end sum_of_remainders_l232_232080


namespace range_of_set_l232_232244

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l232_232244


namespace valid_codes_count_correct_l232_232949

def valid_codes_count : ℕ :=
  let digits := {0, 1, 2, 3, 4}
  let pairs := [(1, 2), (2, 4)]
  let remaining_digits (a b : ℕ) := (digits.erase a).erase b
  let num_permutations (s : Finset ℕ) := s.toFinset.perm.card
  pairs.length * num_permutations (remaining_digits 1 2)

theorem valid_codes_count_correct : valid_codes_count = 12 :=
by
  have h_perm : ∀ (a b : ℕ), 
  let rem_digits := (digits.erase a).erase b in
  rem_digits.toFinset.perm.card = 6 := sorry
  simp [valid_codes_count, remaining_digits, pairs, h_perm]
  done

end valid_codes_count_correct_l232_232949


namespace circle_symmetric_line_l232_232989

theorem circle_symmetric_line (m : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) → (3*x + y + m = 0)) →
  m = 1 :=
by
  intro h
  sorry

end circle_symmetric_line_l232_232989


namespace cost_of_first_house_l232_232847

theorem cost_of_first_house (C : ℝ) (h₀ : 2 * C + C = 600000) : C = 200000 := by
  -- proof placeholder
  sorry

end cost_of_first_house_l232_232847


namespace math_problem_l232_232960

theorem math_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + 2) * (b + 2) = 18) :
  (∀ x, (x = 3 / (a + 2) + 3 / (b + 2)) → x ≥ Real.sqrt 2) ∧
  ¬(∃ y, (y = a * b) ∧ y ≤ 11 - 6 * Real.sqrt 2) ∧
  (∀ z, (z = 2 * a + b) → z ≥ 6) ∧
  (∀ w, (w = (a + 1) * b) → w ≤ 8) :=
sorry

end math_problem_l232_232960


namespace population_increase_time_l232_232073

theorem population_increase_time (persons_added : ℕ) (time_minutes : ℕ) (seconds_per_minute : ℕ) (total_seconds : ℕ) (time_for_one_person : ℕ) :
  persons_added = 160 →
  time_minutes = 40 →
  seconds_per_minute = 60 →
  total_seconds = time_minutes * seconds_per_minute →
  time_for_one_person = total_seconds / persons_added →
  time_for_one_person = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end population_increase_time_l232_232073


namespace max_value_2019m_2020n_l232_232778

theorem max_value_2019m_2020n (m n : ℤ) (h1 : 0 ≤ m - n) (h2 : m - n ≤ 1) (h3 : 2 ≤ m + n) (h4 : m + n ≤ 4) :
  (∀ (m' n' : ℤ), (0 ≤ m' - n') → (m' - n' ≤ 1) → (2 ≤ m' + n') → (m' + n' ≤ 4) → (m - 2 * n ≥ m' - 2 * n')) →
  2019 * m + 2020 * n = 2019 :=
by
  sorry

end max_value_2019m_2020n_l232_232778


namespace distance_from_Q_to_BC_l232_232194

-- Definitions for the problem
structure Square :=
(A B C D : ℝ × ℝ)
(side_length : ℝ)

def P : (ℝ × ℝ) := (3, 6)
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 6)^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25
def side_BC (x y : ℝ) : Prop := x = 6

-- Lean proof statement
theorem distance_from_Q_to_BC (Q : ℝ × ℝ) (hQ1 : circle1 Q.1 Q.2) (hQ2 : circle2 Q.1 Q.2) :
  Exists (fun d : ℝ => Q.1 = 6 ∧ Q.2 = d) := sorry

end distance_from_Q_to_BC_l232_232194


namespace MeganMarkers_l232_232391

def initialMarkers : Nat := 217
def additionalMarkers : Nat := 109
def totalMarkers : Nat := initialMarkers + additionalMarkers

theorem MeganMarkers : totalMarkers = 326 := by
    sorry

end MeganMarkers_l232_232391


namespace combined_average_mark_l232_232376

theorem combined_average_mark 
  (n_A n_B n_C n_D n_E : ℕ) 
  (avg_A avg_B avg_C avg_D avg_E : ℕ)
  (students_A : n_A = 22) (students_B : n_B = 28)
  (students_C : n_C = 15) (students_D : n_D = 35)
  (students_E : n_E = 25)
  (avg_marks_A : avg_A = 40) (avg_marks_B : avg_B = 60)
  (avg_marks_C : avg_C = 55) (avg_marks_D : avg_D = 75)
  (avg_marks_E : avg_E = 50) : 
  (22 * 40 + 28 * 60 + 15 * 55 + 35 * 75 + 25 * 50) / (22 + 28 + 15 + 35 + 25) = 58.08 := 
  by 
    sorry

end combined_average_mark_l232_232376


namespace problem_statement_l232_232334

theorem problem_statement (x y : ℝ) (h : -x + 2 * y = 5) :
  5 * (x - 2 * y) ^ 2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  sorry

end problem_statement_l232_232334


namespace prob_complement_A_l232_232610

theorem prob_complement_A (P : Set (Set α) → ℝ) (A B : Set α)
  (hPB : P B = 0.3)
  (hPBA : P (B \ A) / P (A) = 0.9)
  (hPBNegA : P (B \ ¬A) / P (¬A) = 0.2) :
  P (¬A) = 6 / 7 := by
  sorry

end prob_complement_A_l232_232610


namespace range_of_set_l232_232249

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l232_232249


namespace total_seashells_l232_232645

def joans_seashells : Nat := 6
def jessicas_seashells : Nat := 8

theorem total_seashells : joans_seashells + jessicas_seashells = 14 :=
by
  sorry

end total_seashells_l232_232645


namespace cost_prices_of_products_l232_232233

-- Define the variables and conditions from the problem
variables (x y : ℝ)

-- Theorem statement
theorem cost_prices_of_products (h1 : 20 * x + 15 * y = 380) (h2 : 15 * x + 10 * y = 280) : 
  x = 16 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end cost_prices_of_products_l232_232233


namespace total_copies_l232_232083

theorem total_copies (rate1 : ℕ) (rate2 : ℕ) (time : ℕ) (total : ℕ) 
  (h1 : rate1 = 25) (h2 : rate2 = 55) (h3 : time = 30) : 
  total = rate1 * time + rate2 * time := 
  sorry

end total_copies_l232_232083


namespace count_valid_48_tuples_l232_232802

open BigOperators

theorem count_valid_48_tuples : 
  ∃ n : ℕ, n = 54 ^ 48 ∧ 
  ( ∃ a : Fin 48 → ℕ, 
    (∀ i : Fin 48, 0 ≤ a i ∧ a i ≤ 100) ∧ 
    (∀ (i j : Fin 48), i < j → a i ≠ a j ∧ a i ≠ a j + 1) 
  ) :=
by
  sorry

end count_valid_48_tuples_l232_232802


namespace broken_line_coverable_l232_232076

noncomputable def cover_broken_line (length_of_line : ℝ) (radius_of_circle : ℝ) : Prop :=
  length_of_line = 5 ∧ radius_of_circle > 1.25

theorem broken_line_coverable :
  ∃ radius_of_circle, cover_broken_line 5 radius_of_circle :=
by sorry

end broken_line_coverable_l232_232076


namespace ratio_of_a_plus_b_to_b_plus_c_l232_232153

variable (a b c : ℝ)

theorem ratio_of_a_plus_b_to_b_plus_c (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 :=
by
  sorry

end ratio_of_a_plus_b_to_b_plus_c_l232_232153


namespace product_of_solutions_eq_neg_ten_l232_232427

theorem product_of_solutions_eq_neg_ten :
  (∃ x₁ x₂, -20 = -2 * x₁^2 - 6 * x₁ ∧ -20 = -2 * x₂^2 - 6 * x₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -10) :=
by
  sorry

end product_of_solutions_eq_neg_ten_l232_232427


namespace problem_solution_l232_232528

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Complement within U
def complement_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- The proof goal
theorem problem_solution : (complement_U A) ∪ B = {2, 3, 4, 5} := by
  sorry

end problem_solution_l232_232528


namespace walnut_trees_planted_today_l232_232888

-- Define the number of walnut trees before planting
def walnut_trees_before_planting : ℕ := 22

-- Define the number of walnut trees after planting
def walnut_trees_after_planting : ℕ := 55

-- Define a theorem to prove the number of walnut trees planted
theorem walnut_trees_planted_today : 
  walnut_trees_after_planting - walnut_trees_before_planting = 33 :=
by
  -- The proof will be inserted here.
  sorry

end walnut_trees_planted_today_l232_232888


namespace daria_multiple_pizzas_l232_232460

variable (m : ℝ)
variable (don_pizzas : ℝ) (total_pizzas : ℝ)

axiom don_pizzas_def : don_pizzas = 80
axiom total_pizzas_def : total_pizzas = 280

theorem daria_multiple_pizzas (m : ℝ) (don_pizzas : ℝ) (total_pizzas : ℝ) 
    (h1 : don_pizzas = 80) (h2 : total_pizzas = 280) 
    (h3 : total_pizzas = don_pizzas + m * don_pizzas) : 
    m = 2.5 :=
by sorry

end daria_multiple_pizzas_l232_232460


namespace line_passes_through_fixed_point_l232_232409

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k + 1) * (-1) - (2 * k - 1) * (1) + 3 * k = 0 :=
by
  intro k
  sorry

end line_passes_through_fixed_point_l232_232409


namespace book_configurations_l232_232575

theorem book_configurations : 
  (∃ (configurations : Finset ℕ), configurations = {1, 2, 3, 4, 5, 6, 7} ∧ configurations.card = 7) 
  ↔ 
  (∃ (n : ℕ), n = 7) :=
by 
  sorry

end book_configurations_l232_232575


namespace am_gm_inequality_l232_232525

theorem am_gm_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c) ^ 2 :=
by
  sorry

end am_gm_inequality_l232_232525


namespace greatest_abs_solution_l232_232946

theorem greatest_abs_solution :
  (∃ x : ℝ, x^2 + 18 * x + 81 = 0 ∧ ∀ y : ℝ, y^2 + 18 * y + 81 = 0 → |x| ≥ |y| ∧ |x| = 9) :=
sorry

end greatest_abs_solution_l232_232946


namespace total_fruits_is_78_l232_232390

def oranges_louis : Nat := 5
def apples_louis : Nat := 3

def oranges_samantha : Nat := 8
def apples_samantha : Nat := 7

def oranges_marley : Nat := 2 * oranges_louis
def apples_marley : Nat := 3 * apples_samantha

def oranges_edward : Nat := 3 * oranges_louis
def apples_edward : Nat := 3 * apples_louis

def total_fruits_louis : Nat := oranges_louis + apples_louis
def total_fruits_samantha : Nat := oranges_samantha + apples_samantha
def total_fruits_marley : Nat := oranges_marley + apples_marley
def total_fruits_edward : Nat := oranges_edward + apples_edward

def total_fruits_all : Nat :=
  total_fruits_louis + total_fruits_samantha + total_fruits_marley + total_fruits_edward

theorem total_fruits_is_78 : total_fruits_all = 78 := by
  sorry

end total_fruits_is_78_l232_232390


namespace equivalent_proof_problem_l232_232121

def op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem equivalent_proof_problem (x y : ℝ) : 
  op ((x + y) ^ 2) ((x - y) ^ 2) = 4 * (x ^ 2 + y ^ 2) ^ 2 := 
by 
  sorry

end equivalent_proof_problem_l232_232121


namespace no_valid_2011_matrix_l232_232913

def valid_matrix (A : ℕ → ℕ → ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 2011 →
    (∀ k, 1 ≤ k ∧ k ≤ 4021 →
      (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A i j = k) ∨ (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A j i = k))

theorem no_valid_2011_matrix :
  ¬ ∃ A : ℕ → ℕ → ℕ, (∀ i j, 1 ≤ i ∧ i ≤ 2011 ∧ 1 ≤ j ∧ j ≤ 2011 → 1 ≤ A i j ∧ A i j ≤ 4021) ∧ valid_matrix A :=
by
  sorry

end no_valid_2011_matrix_l232_232913


namespace no_real_roots_iff_l232_232500

theorem no_real_roots_iff (k : ℝ) : (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_iff_l232_232500


namespace simplify_and_evaluate_l232_232536

theorem simplify_and_evaluate (x : ℚ) (h1 : x = -1/3) :
    (3 * x + 2) * (3 * x - 2) - 5 * x * (x - 1) - (2 * x - 1)^2 = 9 * x - 5 ∧
    (9 * x - 5) = -8 := 
by sorry

end simplify_and_evaluate_l232_232536


namespace last_digit_of_product_of_consecutive_numbers_l232_232676

theorem last_digit_of_product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h1 : k > 5)
    (h2 : n = (k + 1) * (k + 2) * (k + 3) * (k + 4))
    (h3 : n % 10 ≠ 0) : n % 10 = 4 :=
sorry -- Proof not provided as per instructions.

end last_digit_of_product_of_consecutive_numbers_l232_232676


namespace prob_min_score_guaranteeing_payoff_l232_232375

-- Definitions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
def prob_single_throw (n : ℕ) : ℚ := if n ∈ dice_faces then 1 / 6 else 0
def two_throws_event : Event (Finset (ℕ × ℕ)) := 
  {e | e.1 ∈ dice_faces ∧ e.2 ∈ dice_faces}

-- Mathematical Statement
theorem prob_min_score_guaranteeing_payoff : 
  Probability (Event.filter two_throws_event (λ x : ℕ × ℕ, x.1 + x.2 = 12)) = 1 / 36 :=
by
  sorry

end prob_min_score_guaranteeing_payoff_l232_232375


namespace pets_percentage_of_cats_l232_232392

theorem pets_percentage_of_cats :
  ∀ (total_pets dogs as_percentage bunnies cats_percentage : ℕ),
    total_pets = 36 →
    dogs = total_pets * as_percentage / 100 →
    as_percentage = 25 →
    bunnies = 9 →
    cats_percentage = (total_pets - (dogs + bunnies)) * 100 / total_pets →
    cats_percentage = 50 :=
by
  intros total_pets dogs as_percentage bunnies cats_percentage
  sorry

end pets_percentage_of_cats_l232_232392


namespace purely_imaginary_sol_l232_232966

theorem purely_imaginary_sol (x : ℝ) 
  (h1 : (x^2 - 1) = 0)
  (h_imag : (x^2 + 3 * x + 2) ≠ 0) :
  x = 1 :=
sorry

end purely_imaginary_sol_l232_232966


namespace pieces_eaten_first_l232_232453

variable (initial_candy : ℕ) (remaining_candy : ℕ) (candy_eaten_second : ℕ)

theorem pieces_eaten_first 
    (initial_candy := 21) 
    (remaining_candy := 7)
    (candy_eaten_second := 9) :
    (initial_candy - remaining_candy - candy_eaten_second = 5) :=
sorry

end pieces_eaten_first_l232_232453


namespace min_rounds_for_expected_value_l232_232947

theorem min_rounds_for_expected_value 
  (p1 p2 : ℝ) (h0 : 0 ≤ p1 ∧ p1 ≤ 1) (h1 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h2 : p1 + p2 = 3 / 2)
  (indep : true) -- Assuming independence implicitly
  (X : ℕ → ℕ) (n : ℕ)
  (E_X_eq_24 : (n : ℕ) * (3 * p1 * p2 * (1 - p1 * p2)) = 24) :
  n = 32 := 
sorry

end min_rounds_for_expected_value_l232_232947


namespace find_range_of_x_l232_232487

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem find_range_of_x (x : ℝ) :
  (f (2 * x) > f (x - 3)) ↔ (x < -3 ∨ x > 1) :=
sorry

end find_range_of_x_l232_232487


namespace jose_birds_left_l232_232520

-- Define initial conditions
def chickens_initial : Nat := 28
def ducks : Nat := 18
def turkeys : Nat := 15
def chickens_sold : Nat := 12

-- Calculate remaining chickens
def chickens_left : Nat := chickens_initial - chickens_sold

-- Calculate total birds left
def total_birds_left : Nat := chickens_left + ducks + turkeys

-- Theorem statement to prove the number of birds left
theorem jose_birds_left : total_birds_left = 49 :=
by
  -- This is where the proof would typically go
  sorry

end jose_birds_left_l232_232520


namespace range_of_set_l232_232282

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l232_232282


namespace problem1_problem2_problem3_l232_232969

variable (a b : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_cond1 : a ≥ (1 / a) + (2 / b))
variable (h_cond2 : b ≥ (3 / a) + (2 / b))

/-- Statement 1: Prove that a + b ≥ 4 under the given conditions. -/
theorem problem1 : (a + b) ≥ 4 := 
by 
  sorry

/-- Statement 2: Prove that a^2 + b^2 ≥ 3 + 2√6 under the given conditions. -/
theorem problem2 : (a^2 + b^2) ≥ (3 + 2 * Real.sqrt 6) := 
by 
  sorry

/-- Statement 3: Prove that (1/a) + (1/b) < 1 + (√2/2) under the given conditions. -/
theorem problem3 : (1 / a) + (1 / b) < 1 + (Real.sqrt 2 / 2) := 
by 
  sorry

end problem1_problem2_problem3_l232_232969


namespace constant_subsequence_exists_l232_232849

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem constant_subsequence_exists (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (f : ℕ → ℕ) (c : ℕ), (∀ n m, n < m → f n < f m) ∧ (∀ n, sum_of_digits (⌊a * ↑(f n) + b⌋₊) = c) :=
sorry

end constant_subsequence_exists_l232_232849


namespace relationship_between_x_and_y_l232_232495

theorem relationship_between_x_and_y (m x y : ℝ) (h1 : x = 3 - m) (h2 : y = 2 * m + 1) : 2 * x + y = 7 :=
sorry

end relationship_between_x_and_y_l232_232495


namespace find_three_digit_number_l232_232471

-- Define the function that calculates the total number of digits required
def total_digits (x : ℕ) : ℕ :=
  (if x >= 1 then 9 else 0) +
  (if x >= 10 then 90 * 2 else 0) +
  (if x >= 100 then 3 * (x - 99) else 0)

theorem find_three_digit_number : ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ 2 * x = total_digits x := by
  sorry

end find_three_digit_number_l232_232471


namespace Saheed_earnings_l232_232061

theorem Saheed_earnings (Vika_earnings : ℕ) (Kayla_earnings : ℕ) (Saheed_earnings : ℕ)
  (h1 : Vika_earnings = 84) (h2 : Kayla_earnings = Vika_earnings - 30) (h3 : Saheed_earnings = 4 * Kayla_earnings) :
  Saheed_earnings = 216 := 
by
  sorry

end Saheed_earnings_l232_232061


namespace pairs_of_positive_integers_l232_232463

theorem pairs_of_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
    (∃ (m : ℕ), m ≥ 2 ∧ (x = m^3 + 2*m^2 - m - 1 ∧ y = m^3 + m^2 - 2*m - 1 ∨ 
                        x = m^3 + m^2 - 2*m - 1 ∧ y = m^3 + 2*m^2 - m - 1)) ∨
    (x = 1 ∧ y = 1) ↔ 
    (∃ n : ℝ, n^3 = 7*x^2 - 13*x*y + 7*y^2) ∧ (Int.natAbs (x - y) - 1 = n) :=
by
  sorry

end pairs_of_positive_integers_l232_232463


namespace polygon_sides_l232_232225

theorem polygon_sides (n : ℕ) (h : 44 = n * (n - 3) / 2) : n = 11 :=
sorry

end polygon_sides_l232_232225


namespace tan_alpha_l232_232629

variable (α : ℝ)
variable (H_cos : Real.cos α = 12/13)
variable (H_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)

theorem tan_alpha :
  Real.tan α = -5/12 :=
sorry

end tan_alpha_l232_232629


namespace range_of_set_l232_232286

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l232_232286


namespace red_light_after_two_red_light_expectation_and_variance_l232_232445

noncomputable def prob_red_light_after_two : ℚ := (2/3) * (2/3) * (1/3)
theorem red_light_after_two :
  prob_red_light_after_two = 4/27 :=
by
  -- We have defined the probability calculation directly
  sorry

noncomputable def expected_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p
noncomputable def variance_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem red_light_expectation_and_variance :
  expected_red_lights 6 (1/3) = 2 ∧ variance_red_lights 6 (1/3) = 4/3 :=
by
  -- We have defined expectation and variance calculations directly
  sorry

end red_light_after_two_red_light_expectation_and_variance_l232_232445


namespace trajectory_of_midpoint_l232_232337

open Real

theorem trajectory_of_midpoint (A : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A = (-2, 0))
    (hP_on_curve : P.1 = 2 * P.2 ^ 2)
    (hM_midpoint : M = ((A.1 + P.1) / 2, (A.2 + P.2) / 2)) :
    M.1 = 4 * M.2 ^ 2 - 1 :=
sorry

end trajectory_of_midpoint_l232_232337


namespace missing_jar_size_l232_232044

theorem missing_jar_size (total_ounces jars_16 jars_28 jars_unknown m n p: ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : p = 3)
    (total_jars : m + n + p = 9)
    (total_peanut_butter : 16 * m + 28 * n + jars_unknown * p = 252)
    : jars_unknown = 40 := by
  sorry

end missing_jar_size_l232_232044


namespace bulls_win_nba_finals_l232_232197

open ProbabilityTheory

def bulls_win_probability : ℝ := 
  let p_bulls := (2:ℝ) / 3 in
  let p_lakers := (1:ℝ) / 3 in
  ∑ k in Finset.range 6, 
    (Nat.choose (5 + k) k * (p_bulls^6) * (p_lakers^k))

theorem bulls_win_nba_finals : (bulls_win_probability * 100).round = 86 := 
  by
  sorry

end bulls_win_nba_finals_l232_232197


namespace find_number_l232_232118

theorem find_number
  (n : ℕ)
  (h : 80641 * n = 806006795) :
  n = 9995 :=
by 
  sorry

end find_number_l232_232118


namespace problem_solution_l232_232356

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l232_232356


namespace necessary_but_not_sufficient_l232_232708

theorem necessary_but_not_sufficient (x : ℝ) :
  (x - 1) * (x + 2) = 0 → (x = 1 ∨ x = -2) ∧ (x = 1 → (x - 1) * (x + 2) = 0) ∧ ¬((x - 1) * (x + 2) = 0 ↔ x = 1) :=
by
  sorry

end necessary_but_not_sufficient_l232_232708


namespace number_of_girls_who_left_l232_232586

-- Definitions for initial conditions and event information
def initial_boys : ℕ := 24
def initial_girls : ℕ := 14
def final_students : ℕ := 30

-- Main theorem statement translating the problem question
theorem number_of_girls_who_left (B G : ℕ) (h1 : B = G) 
  (h2 : initial_boys + initial_girls - B - G = final_students) :
  G = 4 := 
sorry

end number_of_girls_who_left_l232_232586


namespace blue_balls_count_l232_232161

theorem blue_balls_count (Y B : ℕ) (h_ratio : 4 * B = 3 * Y) (h_total : Y + B = 35) : B = 15 :=
sorry

end blue_balls_count_l232_232161


namespace range_of_set_l232_232278

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l232_232278


namespace eccentricity_of_ellipse_l232_232119

variable (a b c d1 d2 : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
variable (h4 : 2 * c = (d1 + d2) / 2)
variable (h5 : d1 + d2 = 2 * a)

theorem eccentricity_of_ellipse : (c / a) = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l232_232119


namespace find_y_l232_232985

theorem find_y (x y : ℤ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 :=
sorry

end find_y_l232_232985


namespace order_of_p_q_r_l232_232336

theorem order_of_p_q_r (p q r : ℝ) (h1 : p = Real.sqrt 2) (h2 : q = Real.sqrt 7 - Real.sqrt 3) (h3 : r = Real.sqrt 6 - Real.sqrt 2) :
  p > r ∧ r > q :=
by
  sorry

end order_of_p_q_r_l232_232336


namespace problem_statement_l232_232862

namespace LeanProofExample

def not_divisible (n : ℕ) (p : ℕ) : Prop :=
  ¬(p ∣ n)

theorem problem_statement (x y : ℕ) 
  (hx : not_divisible x 59) 
  (hy : not_divisible y 59)
  (h : 3 * x + 28 * y ≡ 0 [MOD 59]) :
  ¬(5 * x + 16 * y ≡ 0 [MOD 59]) :=
  sorry

end LeanProofExample

end problem_statement_l232_232862


namespace remaining_batches_l232_232127

def flour_per_batch : ℕ := 2
def batches_baked : ℕ := 3
def initial_flour : ℕ := 20

theorem remaining_batches : (initial_flour - flour_per_batch * batches_baked) / flour_per_batch = 7 := by
  sorry

end remaining_batches_l232_232127


namespace problem_statement_l232_232154

theorem problem_statement (a b c x y z : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z)
  (h7 : a^2 + b^2 + c^2 = 16) (h8 : x^2 + y^2 + z^2 = 49) (h9 : a * x + b * y + c * z = 28) : 
  (a + b + c) / (x + y + z) = 4 / 7 := 
by
  sorry

end problem_statement_l232_232154


namespace abs_inequality_l232_232613

theorem abs_inequality (x y : ℝ) (h1 : |x| < 2) (h2 : |y| < 2) : |4 - x * y| > 2 * |x - y| :=
by
  sorry

end abs_inequality_l232_232613


namespace range_of_set_l232_232292

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l232_232292


namespace cube_edge_length_eq_six_l232_232410

theorem cube_edge_length_eq_six {s : ℝ} (h : s^3 = 6 * s^2) : s = 6 :=
sorry

end cube_edge_length_eq_six_l232_232410


namespace range_of_set_of_three_numbers_l232_232266

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l232_232266


namespace trig_expression_equality_l232_232730

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end trig_expression_equality_l232_232730


namespace cos_double_alpha_two_alpha_minus_beta_l232_232709

variable (α β : ℝ)
variable (α_pos : 0 < α)
variable (α_lt_pi : α < π)
variable (tan_α : Real.tan α = 2)

variable (β_pos : 0 < β)
variable (β_lt_pi : β < π)
variable (cos_β : Real.cos β = -((7 * Real.sqrt 2) / 10))

theorem cos_double_alpha (hα : 0 < α ∧ α < π) (htan : Real.tan α = 2) : 
  Real.cos (2 * α) = -3 / 5 := by
  sorry

theorem two_alpha_minus_beta (hα : 0 < α ∧ α < π) (htan : Real.tan α = 2)
  (hβ : 0 < β ∧ β < π) (hcosβ : Real.cos β = -((7 * Real.sqrt 2) / 10)) : 
  2 * α - β = -π / 4 := by
  sorry

end cos_double_alpha_two_alpha_minus_beta_l232_232709


namespace largest_positive_integer_n_l232_232368

def binary_operation (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_n (x : ℤ) (h : x = -15) : 
  ∃ (n : ℤ), n > 0 ∧ binary_operation n < x ∧ ∀ m > 0, binary_operation m < x → m ≤ n :=
by
  sorry

end largest_positive_integer_n_l232_232368


namespace geometric_mean_unique_solution_l232_232620

-- Define the conditions
variable (k : ℕ) -- k is a natural number
variable (hk_pos : 0 < k) -- k is a positive natural number

-- The geometric mean condition translated to Lean
def geometric_mean_condition (k : ℕ) : Prop :=
  (2 * k)^2 = (k + 9) * (6 - k)

-- The main statement to prove
theorem geometric_mean_unique_solution (k : ℕ) (hk_pos : 0 < k) (h: geometric_mean_condition k) : k = 3 :=
sorry -- proof placeholder

end geometric_mean_unique_solution_l232_232620


namespace solve_fractional_equation_for_c_l232_232874

theorem solve_fractional_equation_for_c :
  (∃ c : ℝ, (c - 37) / 3 = (3 * c + 7) / 8) → c = -317 := by
sorry

end solve_fractional_equation_for_c_l232_232874


namespace walk_to_bus_stop_time_l232_232894

/-- Walking with 4/5 of my usual speed, I arrive at the bus stop 7 minutes later than normal.
    How many minutes does it take to walk to the bus stop at my usual speed? -/
theorem walk_to_bus_stop_time (S T : ℝ) (h : T > 0) 
  (d_usual : S * T = (4/5) * S * (T + 7)) : 
  T = 28 :=
by
  sorry

end walk_to_bus_stop_time_l232_232894


namespace xiao_li_hits_bullseye_14_times_l232_232686

theorem xiao_li_hits_bullseye_14_times
  (initial_rifle_bullets : ℕ := 10)
  (initial_pistol_bullets : ℕ := 14)
  (reward_per_bullseye_rifle : ℕ := 2)
  (reward_per_bullseye_pistol : ℕ := 4)
  (xiao_wang_bullseyes : ℕ := 30)
  (total_bullets : ℕ := initial_rifle_bullets + xiao_wang_bullseyes * reward_per_bullseye_rifle) :
  ∃ (xiao_li_bullseyes : ℕ), total_bullets = initial_pistol_bullets + xiao_li_bullseyes * reward_per_bullseye_pistol ∧ xiao_li_bullseyes = 14 :=
by sorry

end xiao_li_hits_bullseye_14_times_l232_232686


namespace range_of_numbers_is_six_l232_232255

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l232_232255


namespace total_revenue_is_correct_l232_232653

-- Define the constants and conditions
def price_of_jeans : ℕ := 11
def price_of_tees : ℕ := 8
def quantity_of_tees_sold : ℕ := 7
def quantity_of_jeans_sold : ℕ := 4

-- Define the total revenue calculation
def total_revenue : ℕ :=
  (price_of_tees * quantity_of_tees_sold) +
  (price_of_jeans * quantity_of_jeans_sold)

-- The theorem to prove
theorem total_revenue_is_correct : total_revenue = 100 := 
by
  -- Proof is omitted for now
  sorry

end total_revenue_is_correct_l232_232653


namespace quadratic_eq_with_roots_l232_232786

theorem quadratic_eq_with_roots (x y : ℝ) (h : (x^2 - 6 * x + 9) = -|y - 1|) : 
  ∃ a : ℝ, (a^2 - 4 * a + 3 = 0) :=
by 
  sorry

end quadratic_eq_with_roots_l232_232786


namespace find_m_value_l232_232973

noncomputable def fx (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem find_m_value (m : ℝ) : (∀ x > 0, fx m x > fx m 0) → m = 2 := by
  sorry

end find_m_value_l232_232973


namespace carnival_game_ratio_l232_232450

theorem carnival_game_ratio (L W : ℕ) (h_ratio : 4 * L = W) (h_lost : L = 7) : W = 28 :=
by {
  sorry
}

end carnival_game_ratio_l232_232450


namespace odd_integers_equality_l232_232386

-- Definitions
def is_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

def divides (d n : ℤ) := ∃ k : ℤ, n = d * k

-- Main statement
theorem odd_integers_equality (a b : ℤ) (ha_pos : 0 < a) (hb_pos : 0 < b)
 (ha_odd : is_odd a) (hb_odd : is_odd b)
 (h_div : divides (2 * a * b + 1) (a^2 + b^2 + 1))
 : a = b :=
by 
  sorry

end odd_integers_equality_l232_232386


namespace largest_inscribed_triangle_area_l232_232319

-- Definition of the conditions
def radius : ℝ := 10
def diameter : ℝ := 2 * radius

-- The theorem to be proven
theorem largest_inscribed_triangle_area (r : ℝ) (D : ℝ) (h : D = 2 * r) : 
  ∃ (A : ℝ), A = 100 := by
  have base := D
  have height := r
  have area := (1 / 2) * base * height
  use area
  sorry

end largest_inscribed_triangle_area_l232_232319


namespace square_side_length_l232_232696

theorem square_side_length (area_circle perimeter_square : ℝ) (h1 : area_circle = 100) (h2 : perimeter_square = area_circle) :
  side_length_square perimeter_square = 25 :=
by
  let s := 25 -- The length of one side of the square is 25
  sorry

def side_length_square (perimeter_square : ℝ) : ℝ :=
  perimeter_square / 4

end square_side_length_l232_232696


namespace find_horses_l232_232434

theorem find_horses {x : ℕ} :
  (841 : ℝ) = 8 * (x : ℝ) + 16 * 9 + 18 * 6 → 348 = 16 * 9 →
  x = 73 :=
by
  intros h₁ h₂
  sorry

end find_horses_l232_232434


namespace rosy_current_age_l232_232086

theorem rosy_current_age 
  (R : ℕ) 
  (h1 : ∀ (david_age rosy_age : ℕ), david_age = rosy_age + 12) 
  (h2 : ∀ (david_age_plus_4 rosy_age_plus_4 : ℕ), david_age_plus_4 = 2 * rosy_age_plus_4) : 
  R = 8 := 
sorry

end rosy_current_age_l232_232086


namespace expected_value_of_winnings_l232_232916

/-- A fair 6-sided die is rolled. If the roll is even, then you win the amount of dollars 
equal to the square of the number you roll. If the roll is odd, you win nothing. 
Prove that the expected value of your winnings is 28/3 dollars. -/
theorem expected_value_of_winnings : 
  (1 / 6) * (2^2 + 4^2 + 6^2) = 28 / 3 := by
sorry

end expected_value_of_winnings_l232_232916


namespace range_of_set_is_six_l232_232305

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l232_232305


namespace min_value_expr_l232_232600

-- Definition of the expression given a real constant k
def expr (k : ℝ) (x y : ℝ) : ℝ := 9 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

-- The proof problem statement
theorem min_value_expr (k : ℝ) (h : k = 2 / 9) : ∃ x y : ℝ, expr k x y = 1 ∧ ∀ x y : ℝ, expr k x y ≥ 1 :=
by
  sorry

end min_value_expr_l232_232600


namespace son_present_age_l232_232702

theorem son_present_age (S F : ℕ) (h1 : F = S + 34) (h2 : F + 2 = 2 * (S + 2)) : S = 32 :=
by
  sorry

end son_present_age_l232_232702


namespace range_of_set_l232_232245

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l232_232245


namespace ratio_simplified_l232_232113

variable (a b c : ℕ)
variable (n m p : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : p > 0)

theorem ratio_simplified (h_ratio : a^n = 3 * c^p ∧ b^m = 4 * c^p ∧ c^p = 7 * c^p) :
  (a^n + b^m + c^p) / c^p = 2 := sorry

end ratio_simplified_l232_232113


namespace floor_sqrt_sum_l232_232746

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l232_232746


namespace sequence_inequality_l232_232523

theorem sequence_inequality (a : ℕ → ℝ) (h1 : ∀ n m : ℕ, a (n + m) ≤ a n + a m)
  (h2 : ∀ n : ℕ, 0 ≤ a n) (n m : ℕ) (hnm : n ≥ m) : 
  a n ≤ m * a 1 + (n / m - 1) * a m :=
sorry

end sequence_inequality_l232_232523


namespace solution_of_inequality_l232_232414

-- Let us define the inequality and the solution set
def inequality (x : ℝ) := (x - 1)^2023 - 2^2023 * x^2023 ≤ x + 1
def solution_set (x : ℝ) := x ≥ -1

-- The theorem statement to prove that the solution set matches the inequality
theorem solution_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} := sorry

end solution_of_inequality_l232_232414


namespace binary_to_decimal_l232_232943

theorem binary_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by
  sorry

end binary_to_decimal_l232_232943


namespace not_square_of_expression_l232_232561

theorem not_square_of_expression (n : ℕ) (h : n > 0) : ∀ k : ℕ, (4 * n^2 + 4 * n + 4 ≠ k^2) :=
by
  sorry

end not_square_of_expression_l232_232561


namespace magician_earned_4_dollars_l232_232237

-- Define conditions
def price_per_deck := 2
def initial_decks := 5
def decks_left := 3

-- Define the number of decks sold
def decks_sold := initial_decks - decks_left

-- Define the total money earned
def money_earned := decks_sold * price_per_deck

-- Theorem to prove the money earned is 4 dollars
theorem magician_earned_4_dollars : money_earned = 4 := by
  sorry

end magician_earned_4_dollars_l232_232237


namespace sequence_eventually_constant_l232_232120

theorem sequence_eventually_constant (n : ℕ) (h : n ≥ 1) : 
  ∃ s, ∀ k ≥ s, (2 ^ (2 ^ k) % n) = (2 ^ (2 ^ (k + 1)) % n) :=
sorry

end sequence_eventually_constant_l232_232120


namespace find_x_values_for_inverse_l232_232758

def f (x : ℝ) : ℝ := x^2 - 3 * x - 4

theorem find_x_values_for_inverse :
  ∃ (x : ℝ), (f x = 2 + 2 * Real.sqrt 2 ∨ f x = 2 - 2 * Real.sqrt 2) ∧ f x = x :=
sorry

end find_x_values_for_inverse_l232_232758


namespace range_of_set_of_three_numbers_l232_232270

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l232_232270


namespace smallest_number_divisible_l232_232428

theorem smallest_number_divisible (n : ℤ) : 
  (n + 7) % 25 = 0 ∧
  (n + 7) % 49 = 0 ∧
  (n + 7) % 15 = 0 ∧
  (n + 7) % 21 = 0 ↔ n = 3668 :=
by 
 sorry

end smallest_number_divisible_l232_232428


namespace find_dividend_l232_232218

-- Definitions based on conditions from the problem
def divisor : ℕ := 13
def quotient : ℕ := 17
def remainder : ℕ := 1

-- Statement of the proof problem
theorem find_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

-- Proof statement ensuring dividend is as expected
example : find_dividend divisor quotient remainder = 222 :=
by 
  sorry

end find_dividend_l232_232218


namespace range_of_numbers_is_six_l232_232277

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l232_232277


namespace probability_truth_or_lies_l232_232638

def probability_truth := 0.30
def probability_lies := 0.20
def probability_both := 0.10

theorem probability_truth_or_lies :
  (probability_truth + probability_lies - probability_both) = 0.40 :=
by
  sorry

end probability_truth_or_lies_l232_232638


namespace num_factors_of_60_l232_232805

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l232_232805


namespace evaluate_expr_l232_232764

theorem evaluate_expr :
  (3 * Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) = 
  -1 / 6 * (Real.sqrt 21 + Real.sqrt 35 - Real.sqrt 77) - 7 / 3 := by
  sorry

end evaluate_expr_l232_232764


namespace intersecting_functions_k_range_l232_232020

theorem intersecting_functions_k_range 
  (k : ℝ) (h : 0 < k) : 
    ∃ x : ℝ, -2 * x + 3 = k / x ↔ k ≤ 9 / 8 :=
by 
  sorry

end intersecting_functions_k_range_l232_232020


namespace B_age_l232_232228

-- Define the conditions
variables (x y : ℕ)
variable (current_year : ℕ)
axiom h1 : 10 * x + y + 4 = 43
axiom reference_year : current_year = 1955

-- Define the relationship between the digit equation and the year
def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

-- Birth year calculation
def age (current_year birth_year : ℕ) : ℕ := current_year - birth_year

-- Final theorem: Age of B
theorem B_age (x y : ℕ) (current_year : ℕ) (h1 : 10 * x  + y + 4 = 43) (reference_year : current_year = 1955) :
  age current_year (birth_year x y) = 16 :=
by
  sorry

end B_age_l232_232228


namespace prove_n_eq_1_l232_232383

-- Definitions of the given conditions
def is_prime (x : ℕ) : Prop := Nat.Prime x

variable {p q r n : ℕ}
variable (hp : is_prime p) (hq : is_prime q) (hr : is_prime r)
variable (hn_pos : n > 0)
variable (h_eq : p^n + q^n = r^2)

-- Statement to prove
theorem prove_n_eq_1 : n = 1 :=
  sorry

end prove_n_eq_1_l232_232383


namespace total_flowers_is_288_l232_232105

-- Definitions from the Conditions in a)
def arwen_tulips : ℕ := 20
def arwen_roses : ℕ := 18
def elrond_tulips : ℕ := 2 * arwen_tulips
def elrond_roses : ℕ := 3 * arwen_roses
def galadriel_tulips : ℕ := 3 * elrond_tulips
def galadriel_roses : ℕ := 2 * arwen_roses

-- Total number of tulips
def total_tulips : ℕ := arwen_tulips + elrond_tulips + galadriel_tulips

-- Total number of roses
def total_roses : ℕ := arwen_roses + elrond_roses + galadriel_roses

-- Total number of flowers
def total_flowers : ℕ := total_tulips + total_roses

theorem total_flowers_is_288 : total_flowers = 288 :=
by
  -- Placeholder for proof
  sorry

end total_flowers_is_288_l232_232105


namespace pine_trees_multiple_of_27_l232_232448

noncomputable def numberOfPineTrees (n : ℕ) : ℕ := 27 * n

theorem pine_trees_multiple_of_27 (oak_trees : ℕ) (max_trees_per_row : ℕ) (rows_of_oak : ℕ) :
  oak_trees = 54 → max_trees_per_row = 27 → rows_of_oak = oak_trees / max_trees_per_row →
  ∃ n : ℕ, numberOfPineTrees n = 27 * n :=
by
  intros
  use (oak_trees - rows_of_oak * max_trees_per_row) / 27
  sorry

end pine_trees_multiple_of_27_l232_232448


namespace first_half_speed_l232_232723

noncomputable def speed_first_half : ℝ := 21

theorem first_half_speed (total_distance first_half_distance second_half_distance second_half_speed total_time : ℝ)
  (h1 : total_distance = 224)
  (h2 : first_half_distance = total_distance / 2)
  (h3 : second_half_distance = total_distance / 2)
  (h4 : second_half_speed = 24)
  (h5 : total_time = 10)
  (h6 : total_time = first_half_distance / speed_first_half + second_half_distance / second_half_speed) :
  speed_first_half = 21 :=
sorry

end first_half_speed_l232_232723


namespace minimum_reciprocal_sum_l232_232019

theorem minimum_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  4 ≤ (1 / a) + (1 / b) :=
sorry

end minimum_reciprocal_sum_l232_232019


namespace segments_divide_ratio_3_to_1_l232_232687

-- Define points and segments
structure Point :=
  (x : ℝ) (y : ℝ)

structure Segment :=
  (A B : Point)

-- Define T-shaped figure consisting of 22 unit squares
noncomputable def T_shaped_figure : ℕ := 22

-- Define line p passing through point V
structure Line :=
  (p : Point → Point)
  (passes_through : Point)

-- Define equal areas condition
def equal_areas (white_area gray_area : ℝ) : Prop := 
  white_area = gray_area

-- Define the problem
theorem segments_divide_ratio_3_to_1
  (AB : Segment)
  (V : Point)
  (white_area gray_area : ℝ)
  (p : Line)
  (h1 : equal_areas white_area gray_area)
  (h2 : T_shaped_figure = 22)
  (h3 : p.passes_through = V) :
  ∃ (C : Point), (p.p AB.A = C) ∧ ((abs (AB.A.x - C.x)) / (abs (C.x - AB.B.x))) = 3 :=
sorry

end segments_divide_ratio_3_to_1_l232_232687


namespace evaluate_expression_l232_232956

theorem evaluate_expression : 
  abs (abs (-abs (3 - 5) + 2) - 4) = 4 :=
by
  sorry

end evaluate_expression_l232_232956


namespace equivalent_functions_l232_232907

theorem equivalent_functions :
  ∀ (x t : ℝ), (x^2 - 2*x - 1 = t^2 - 2*t + 1) := 
by
  intros x t
  sorry

end equivalent_functions_l232_232907


namespace problem_statement_l232_232221

noncomputable def seq_sub_triples: ℚ :=
  let a := (5 / 6 : ℚ)
  let b := (1 / 6 : ℚ)
  let c := (1 / 4 : ℚ)
  a - b - c

theorem problem_statement : seq_sub_triples = 5 / 12 := by
  sorry

end problem_statement_l232_232221


namespace chameleon_problem_l232_232056

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l232_232056


namespace toothpicks_for_10_squares_l232_232213

theorem toothpicks_for_10_squares : (4 + 3 * (10 - 1)) = 31 :=
by 
  sorry

end toothpicks_for_10_squares_l232_232213


namespace dormouse_is_thief_l232_232513

-- Definitions of the suspects
inductive Suspect
| MarchHare
| Hatter
| Dormouse

open Suspect

-- Definitions of the statement conditions
def statement (s : Suspect) : Suspect :=
match s with
| MarchHare => Hatter
| Hatter => sorry -- Sonya and Hatter's testimonies are not recorded
| Dormouse => sorry -- Sonya and Hatter's testimonies are not recorded

-- Condition that only the thief tells the truth
def tells_truth (thief : Suspect) (s : Suspect) : Prop :=
s = thief

-- Conditions of the problem
axiom condition1 : statement MarchHare = Hatter
axiom condition2 : ∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse

-- Proposition that Dormouse (Sonya) is the thief
theorem dormouse_is_thief : (∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse) → t = Dormouse :=
sorry

end dormouse_is_thief_l232_232513


namespace breadth_of_rectangular_plot_is_18_l232_232880

/-- Problem statement:
The length of a rectangular plot is thrice its breadth. 
If the area of the rectangular plot is 972 sq m, 
this theorem proves that the breadth of the rectangular plot is 18 meters.
-/
theorem breadth_of_rectangular_plot_is_18 (b l : ℝ) (h_length : l = 3 * b) (h_area : l * b = 972) : b = 18 :=
by
  sorry

end breadth_of_rectangular_plot_is_18_l232_232880


namespace no_real_roots_of_quad_eq_l232_232504

theorem no_real_roots_of_quad_eq (k : ℝ) :
  ∀ x : ℝ, ¬ (x^2 - 2*x - k = 0) ↔ k < -1 :=
by sorry

end no_real_roots_of_quad_eq_l232_232504


namespace sum_of_floors_of_square_roots_l232_232749

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l232_232749


namespace product_of_distinct_nonzero_real_satisfying_eq_l232_232485

theorem product_of_distinct_nonzero_real_satisfying_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
    (h : x + 3/x = y + 3/y) : x * y = 3 :=
by sorry

end product_of_distinct_nonzero_real_satisfying_eq_l232_232485


namespace negation_of_prop_l232_232203

theorem negation_of_prop :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_prop_l232_232203


namespace fraction_power_l232_232936

theorem fraction_power : (2 / 5 : ℚ) ^ 3 = 8 / 125 := by
  sorry

end fraction_power_l232_232936


namespace glasses_displayed_is_correct_l232_232597

-- Definitions from the problem conditions
def tall_cupboard_capacity : Nat := 20
def wide_cupboard_capacity : Nat := 2 * tall_cupboard_capacity
def per_shelf_narrow_cupboard : Nat := 15 / 3
def usable_narrow_cupboard_capacity : Nat := 2 * per_shelf_narrow_cupboard

-- Theorem to prove that the total number of glasses displayed is 70
theorem glasses_displayed_is_correct :
  (tall_cupboard_capacity + wide_cupboard_capacity + usable_narrow_cupboard_capacity) = 70 :=
by
  sorry

end glasses_displayed_is_correct_l232_232597


namespace price_of_battery_l232_232124

def cost_of_tire : ℕ := 42
def cost_of_tires (num_tires : ℕ) : ℕ := num_tires * cost_of_tire
def total_cost : ℕ := 224
def num_tires : ℕ := 4
def cost_of_battery : ℕ := total_cost - cost_of_tires num_tires

theorem price_of_battery : cost_of_battery = 56 := by
  sorry

end price_of_battery_l232_232124


namespace feasibility_orderings_l232_232112

theorem feasibility_orderings (a : ℝ) :
  (a ≠ 0) →
  (∀ a > 0, a < 2 * a ∧ 2 * a < 3 * a + 1) ∧
  ¬∃ a, a < 3 * a + 1 ∧ 3 * a + 1 < 2 * a ∧ 2 * a < 3 * a + 1 ∧ a ≠ 0 ∧ a > 0 ∧ a < -1 / 2 ∧ a < 0 ∧ a < -1 ∧ a < -1 / 2 ∧ a < -1 / 2 ∧ a < 0 :=
sorry

end feasibility_orderings_l232_232112


namespace arithmetic_mean_of_fractions_l232_232660

theorem arithmetic_mean_of_fractions :
  let a := (9 : ℝ) / 12
  let b := (5 : ℝ) / 6
  let c := (11 : ℝ) / 12
  (a + c) / 2 = b := 
by
  sorry

end arithmetic_mean_of_fractions_l232_232660


namespace line_l2_equation_min_area_triangle_EPQ_l232_232995

theorem line_l2_equation (t PQ : ℝ) (h_t : t > 0) (h_PQ : PQ = 6) :
  (l_2 = {p | p.y = 0}) ∨ (l_2 = {p | 4 * p.x - 3 * p.y - 1 = 0}) :=
sorry

theorem min_area_triangle_EPQ (t : ℝ) (t_pos_int : t ∈ ℕ ∧ t > 0) 
  (AM_le_2BM : ∀ M, AM ≤ 2 * BM) :
  ∃ t_min : ℕ, t_min = t ∧ area_EPQ = (sqrt 15) / 2 :=
sorry

end line_l2_equation_min_area_triangle_EPQ_l232_232995


namespace pre_images_of_one_l232_232650

def f (x : ℝ) := x^3 - x + 1

theorem pre_images_of_one : {x : ℝ | f x = 1} = {-1, 0, 1} :=
by {
  sorry
}

end pre_images_of_one_l232_232650


namespace inequality_holds_for_all_real_numbers_l232_232760

theorem inequality_holds_for_all_real_numbers (x : ℝ) : 3 * x - 5 ≤ 12 - 2 * x + x^2 :=
by sorry

end inequality_holds_for_all_real_numbers_l232_232760


namespace farmer_goats_sheep_unique_solution_l232_232091

theorem farmer_goats_sheep_unique_solution:
  ∃ g h : ℕ, 0 < g ∧ 0 < h ∧ 28 * g + 30 * h = 1200 ∧ h > g :=
by
  sorry

end farmer_goats_sheep_unique_solution_l232_232091


namespace chili_pepper_cost_l232_232574

theorem chili_pepper_cost :
  ∃ x : ℝ, 
    (3 * 2.50 + 4 * 1.50 + 5 * x = 18) ∧ 
    x = 0.90 :=
by
  use 0.90
  sorry

end chili_pepper_cost_l232_232574


namespace rachel_age_is_24_5_l232_232059

/-- Rachel is 4 years older than Leah -/
def rachel_age_eq_leah_plus_4 (R L : ℝ) : Prop := R = L + 4

/-- Together, Rachel and Leah are twice as old as Sam -/
def rachel_and_leah_eq_twice_sam (R L S : ℝ) : Prop := R + L = 2 * S

/-- Alex is twice as old as Rachel -/
def alex_eq_twice_rachel (A R : ℝ) : Prop := A = 2 * R

/-- The sum of all four friends' ages is 92 -/
def sum_ages_eq_92 (R L S A : ℝ) : Prop := R + L + S + A = 92

theorem rachel_age_is_24_5 (R L S A : ℝ) :
  rachel_age_eq_leah_plus_4 R L →
  rachel_and_leah_eq_twice_sam R L S →
  alex_eq_twice_rachel A R →
  sum_ages_eq_92 R L S A →
  R = 24.5 := 
by 
  sorry

end rachel_age_is_24_5_l232_232059


namespace line_equation_passing_through_P_and_equal_intercepts_l232_232675

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition: line passes through point P(1, 3)
def passes_through_P (P : Point) (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq 1 3 = 0

-- Define the condition: equal intercepts on the x-axis and y-axis
def has_equal_intercepts (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ (∀ x y, line_eq x y = 0 ↔ x / a + y / a = 1)

-- Define the specific lines x + y - 4 = 0 and 3x - y = 0
def specific_line1 (x y : ℝ) : ℝ := x + y - 4
def specific_line2 (x y : ℝ) : ℝ := 3 * x - y

-- Define the point P(1, 3)
def P := Point.mk 1 3

theorem line_equation_passing_through_P_and_equal_intercepts :
  (passes_through_P P specific_line1 ∧ has_equal_intercepts specific_line1) ∨
  (passes_through_P P specific_line2 ∧ has_equal_intercepts specific_line2) :=
by
  sorry

end line_equation_passing_through_P_and_equal_intercepts_l232_232675


namespace exists_constant_sum_arrangement_l232_232866

open Finset

-- Define the set of circles and squares
def circles : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define each square's vertices as 4-element subsets of the set {1, 2, 3, 4, 5, 6, 7, 8, 9}
def squares : Finset (Finset ℕ) := { {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}, {5, 6, 7, 8}, {6, 7, 8, 9} }

-- Lean 4 statement representing the problem
theorem exists_constant_sum_arrangement : ∃ (f : Fin ℕ → ℕ), 
  (∀ x, x ∈ { 1, 2, 3, 4, 5, 6, 7, 8, 9 } → f x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
  ∀ s ∈ squares, (∑ i in s, f i) = k :=
begin
  sorry -- Proof to be filled in later
end

end exists_constant_sum_arrangement_l232_232866


namespace sum_integers_100_to_110_l232_232903

theorem sum_integers_100_to_110 : (∑ i in finset.Icc 100 110, i) = 1155 := by
  sorry

end sum_integers_100_to_110_l232_232903


namespace value_of_3m_2n_l232_232990

section ProofProblem

variable (m n : ℤ)
-- Condition that x-3 is a factor of 3x^3 - mx + n
def factor1 : Prop := (3 * 3^3 - m * 3 + n = 0)
-- Condition that x+4 is a factor of 3x^3 - mx + n
def factor2 : Prop := (3 * (-4)^3 - m * (-4) + n = 0)

theorem value_of_3m_2n (h₁ : factor1 m n) (h₂ : factor2 m n) : abs (3 * m - 2 * n) = 45 := by
  sorry

end ProofProblem

end value_of_3m_2n_l232_232990


namespace difference_of_integers_l232_232679

theorem difference_of_integers : ∃ (x y : ℕ), x + y = 20 ∧ x * y = 96 ∧ (x - y = 4 ∨ y - x = 4) :=
by
  sorry

end difference_of_integers_l232_232679


namespace problem_statement_l232_232339

theorem problem_statement
  (a b m n c : ℝ)
  (h1 : a = -b)
  (h2 : m * n = 1)
  (h3 : |c| = 3)
  : a + b + m * n - |c| = -2 := by
  sorry

end problem_statement_l232_232339


namespace ratio_square_pentagon_l232_232306

theorem ratio_square_pentagon (P_sq P_pent : ℕ) 
  (h_sq : P_sq = 60) (h_pent : P_pent = 60) :
  (P_sq / 4) / (P_pent / 5) = 5 / 4 :=
by 
  sorry

end ratio_square_pentagon_l232_232306


namespace find_factor_l232_232238

theorem find_factor (x : ℕ) (f : ℕ) (h1 : x = 9)
  (h2 : (2 * x + 6) * f = 72) : f = 3 := by
  sorry

end find_factor_l232_232238


namespace find_b_c_l232_232130

theorem find_b_c (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 40) 
  (h2 : a + d = 6)
  (h3 : a * b + b * c + c * d + d * a = 28) : 
  b + c = 17 / 3 := 
by
  sorry

end find_b_c_l232_232130


namespace range_of_set_of_three_numbers_l232_232268

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l232_232268


namespace total_amount_correct_l232_232655

-- Define the prices of jeans and tees
def price_jean : ℕ := 11
def price_tee : ℕ := 8

-- Define the quantities sold
def quantity_jeans_sold : ℕ := 4
def quantity_tees_sold : ℕ := 7

-- Calculate the total amount earned
def total_amount : ℕ := (price_jean * quantity_jeans_sold) + (price_tee * quantity_tees_sold)

-- Now, we state and prove the theorem
theorem total_amount_correct : total_amount = 100 :=
by
  -- Here we assert the correctness of the calculation
  sorry

end total_amount_correct_l232_232655


namespace total_insect_legs_l232_232068

/--
This Lean statement defines the conditions and question,
proving that given 5 insects in the laboratory and each insect
having 6 legs, the total number of insect legs is 30.
-/
theorem total_insect_legs (n_insects : Nat) (legs_per_insect : Nat) (h1 : n_insects = 5) (h2 : legs_per_insect = 6) : (n_insects * legs_per_insect) = 30 :=
by
  sorry

end total_insect_legs_l232_232068


namespace cube_has_12_edges_l232_232028

-- Definition of the number of edges in a cube
def number_of_edges_of_cube : Nat := 12

-- The theorem that asserts the cube has 12 edges
theorem cube_has_12_edges : number_of_edges_of_cube = 12 := by
  -- proof to be filled later
  sorry

end cube_has_12_edges_l232_232028


namespace find_a_b_and_compare_y_values_l232_232486

-- Conditions
def quadratic (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 1
def linear (a : ℝ) (x : ℝ) : ℝ := a * x

-- Problem statement
theorem find_a_b_and_compare_y_values (a b y1 y2 y3 : ℝ) (h₀ : quadratic a b (-2) = 1) (h₁ : linear a (-2) = 1)
    (h2 : y1 = quadratic a b 2) (h3 : y2 = quadratic a b b) (h4 : y3 = quadratic a b (a - b)) :
  (a = -1/2) ∧ (b = -2) ∧ y1 < y3 ∧ y3 < y2 :=
by
  -- Placeholder for the proof
  sorry

end find_a_b_and_compare_y_values_l232_232486


namespace range_of_set_l232_232243

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l232_232243


namespace trig_expr_evaluation_l232_232734

theorem trig_expr_evaluation :
    (1 - (1 / cos (Real.pi / 6))) * 
    (1 + (1 / sin (Real.pi / 3))) * 
    (1 - (1 / sin (Real.pi / 6))) * 
    (1 + (1 / cos (Real.pi / 3))) = 1 := 
by
  let cos_30 := cos (Real.pi / 6)
  let sin_60 := sin (Real.pi / 3)
  let sin_30 := sin (Real.pi / 6)
  let cos_60 := cos (Real.pi / 3)
  have h_cos_30 : cos_30 = (Real.sqrt 3) / 2 := sorry
  have h_sin_60 : sin_60 = (Real.sqrt 3) / 2 := sorry
  have h_sin_30 : sin_30 = 1 / 2 := sorry
  have h_cos_60 : cos_60 = 1 / 2 := sorry
  sorry

end trig_expr_evaluation_l232_232734


namespace factors_of_60_l232_232824

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l232_232824


namespace ratio_of_adults_to_children_l232_232307

-- Definitions based on conditions
def adult_ticket_price : ℝ := 5.50
def child_ticket_price : ℝ := 2.50
def total_receipts : ℝ := 1026
def number_of_adults : ℝ := 152

-- Main theorem to prove ratio of adults to children is 2:1
theorem ratio_of_adults_to_children : 
  ∃ (number_of_children : ℝ), adult_ticket_price * number_of_adults + child_ticket_price * number_of_children = total_receipts ∧ 
  number_of_adults / number_of_children = 2 :=
by
  sorry

end ratio_of_adults_to_children_l232_232307


namespace wizard_achievable_for_odd_n_l232_232209

-- Define what it means for the wizard to achieve his goal
def wizard_goal_achievable (n : ℕ) : Prop :=
  ∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = 2 * n ∧ 
    ∀ (sorcerer_breaks : Finset (ℕ × ℕ)), sorcerer_breaks.card = n → 
      ∃ (dwarves : Finset ℕ), dwarves.card = 2 * n ∧
      ∀ k ∈ dwarves, ((k, (k + 1) % n) ∈ pairs ∨ ((k + 1) % n, k) ∈ pairs) ∧
                     (∀ i j, (i, j) ∈ sorcerer_breaks → ¬((i, j) ∈ pairs ∨ (j, i) ∈ pairs))

theorem wizard_achievable_for_odd_n (n : ℕ) (h : Odd n) : wizard_goal_achievable n := sorry

end wizard_achievable_for_odd_n_l232_232209


namespace min_flash_drives_needed_l232_232941

theorem min_flash_drives_needed (total_files : ℕ) (capacity_per_drive : ℝ)  
  (num_files_0_9 : ℕ) (size_0_9 : ℝ) 
  (num_files_0_8 : ℕ) (size_0_8 : ℝ) 
  (size_0_6 : ℝ) 
  (remaining_files : ℕ) :
  total_files = 40 →
  capacity_per_drive = 2.88 →
  num_files_0_9 = 5 →
  size_0_9 = 0.9 →
  num_files_0_8 = 18 →
  size_0_8 = 0.8 →
  remaining_files = total_files - (num_files_0_9 + num_files_0_8) →
  size_0_6 = 0.6 →
  (num_files_0_9 * size_0_9 + num_files_0_8 * size_0_8 + remaining_files * size_0_6) / capacity_per_drive ≤ 13 :=
by {
  sorry
}

end min_flash_drives_needed_l232_232941


namespace jessica_games_attended_l232_232420

def total_games : ℕ := 6
def games_missed_by_jessica : ℕ := 4

theorem jessica_games_attended : total_games - games_missed_by_jessica = 2 := by
  sorry

end jessica_games_attended_l232_232420


namespace find_a_plus_b_l232_232596

theorem find_a_plus_b (a b : ℚ)
  (h1 : 3 = a + b / (2^2 + 1))
  (h2 : 2 = a + b / (1^2 + 1)) :
  a + b = 1 / 3 := 
sorry

end find_a_plus_b_l232_232596


namespace geometric_progression_fifth_term_sum_l232_232624

def gp_sum_fifth_term
    (p q : ℝ)
    (hpq_sum : p + q = 3)
    (hpq_6th : p^5 + q^5 = 573) : ℝ :=
p^4 + q^4

theorem geometric_progression_fifth_term_sum :
    ∃ p q : ℝ, p + q = 3 ∧ p^5 + q^5 = 573 ∧ gp_sum_fifth_term p q (by sorry) (by sorry) = 161 :=
by
  sorry

end geometric_progression_fifth_term_sum_l232_232624


namespace rice_difference_l232_232858

theorem rice_difference:
  ∀ (total : ℕ) (kept_fraction : ℚ) (kept : ℕ) (given : ℕ),
  total = 50 →
  kept_fraction = (7 : ℚ) / 10 →
  kept = total * kept_fraction →
  given = total - kept →
  (kept - given = 20) :=
by
  intros total kept_fraction kept given
  intros h_total h_fraction h_kept h_given
  sorry

end rice_difference_l232_232858


namespace sequence_term_1000_l232_232994

theorem sequence_term_1000 (a : ℕ → ℤ) 
  (h1 : a 1 = 2010) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 1000 = 2676 :=
sorry

end sequence_term_1000_l232_232994


namespace range_of_set_l232_232258

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l232_232258


namespace denise_spent_l232_232521

theorem denise_spent (price_simple : ℕ) (price_meat : ℕ) (price_fish : ℕ)
  (price_milk_smoothie : ℕ) (price_fruit_smoothie : ℕ) (price_special_smoothie : ℕ)
  (julio_spent_more : ℕ) :
  price_simple = 7 →
  price_meat = 11 →
  price_fish = 14 →
  price_milk_smoothie = 6 →
  price_fruit_smoothie = 7 →
  price_special_smoothie = 9 →
  julio_spent_more = 6 →
  ∃ (d_price : ℕ), (d_price = 14 ∨ d_price = 17) :=
by
  sorry

end denise_spent_l232_232521


namespace expression_equals_one_l232_232737

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end expression_equals_one_l232_232737


namespace remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l232_232329

theorem remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12 :
  (7 * 11 ^ 24 + 2 ^ 24) % 12 = 11 := by
sorry

end remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l232_232329


namespace sum_of_floors_of_square_roots_l232_232743

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l232_232743


namespace compute_value_l232_232942

theorem compute_value : ((-120) - (-60)) / (-30) = 2 := 
by 
  sorry

end compute_value_l232_232942


namespace modular_inverse_of_31_mod_35_is_1_l232_232608

theorem modular_inverse_of_31_mod_35_is_1 :
  ∃ a : ℕ, 0 ≤ a ∧ a < 35 ∧ 31 * a % 35 = 1 := sorry

end modular_inverse_of_31_mod_35_is_1_l232_232608


namespace profit_diff_is_560_l232_232224

-- Define the initial conditions
def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1400

-- Define the ratio parts
def ratio_A : ℕ := 4
def ratio_B : ℕ := 5
def ratio_C : ℕ := 6

-- Define the value of one part based on B's profit share and ratio part
def value_per_part : ℕ := profit_share_B / ratio_B

-- Define the profit shares of A and C
def profit_share_A : ℕ := ratio_A * value_per_part
def profit_share_C : ℕ := ratio_C * value_per_part

-- Define the difference between the profit shares of A and C
def profit_difference : ℕ := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_diff_is_560 : profit_difference = 560 := 
by sorry

end profit_diff_is_560_l232_232224


namespace range_of_set_l232_232297

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l232_232297


namespace no_real_coeff_quadratic_with_roots_sum_and_product_l232_232771

theorem no_real_coeff_quadratic_with_roots_sum_and_product (a b c : ℝ) (h : a ≠ 0) :
  ¬ ∃ (α β : ℝ), (α = a + b + c) ∧ (β = a * b * c) ∧ (α + β = -b / a) ∧ (α * β = c / a) :=
by
  sorry

end no_real_coeff_quadratic_with_roots_sum_and_product_l232_232771
