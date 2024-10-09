import Mathlib

namespace probability_second_roll_twice_first_l835_83573

theorem probability_second_roll_twice_first :
  let outcomes := [(1, 2), (2, 4), (3, 6)]
  let total_outcomes := 36
  let favorable_outcomes := 3
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 12 :=
by
  sorry

end probability_second_roll_twice_first_l835_83573


namespace exercise_l835_83500

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end exercise_l835_83500


namespace Winnie_keeps_lollipops_l835_83559

-- Definitions based on the conditions provided
def total_lollipops : ℕ := 60 + 135 + 5 + 250
def number_of_friends : ℕ := 12

-- The theorem statement we need to prove
theorem Winnie_keeps_lollipops : total_lollipops % number_of_friends = 6 :=
by
  -- proof omitted as instructed
  sorry

end Winnie_keeps_lollipops_l835_83559


namespace sum_of_roots_eq_three_l835_83553

theorem sum_of_roots_eq_three {a b : ℝ} (h₁ : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (11 - a) = 0)
  (h₂ : a * 2^3 + (a + 3 * b) * 2^2 + (b - 4 * a) * 2 + (11 - a) = 0)
  (h₃ : a * 4^3 + (a + 3 * b) * 4^2 + (b - 4 * a) * 4 + (11 - a) = 0) :
  (-3) + 2 + 4 = 3 :=
by
  sorry

end sum_of_roots_eq_three_l835_83553


namespace blue_balls_initial_count_l835_83534

theorem blue_balls_initial_count (B : ℕ)
  (h1 : 15 - 3 = 12)
  (h2 : (B - 3) / 12 = 1 / 3) :
  B = 7 :=
sorry

end blue_balls_initial_count_l835_83534


namespace cara_neighbors_l835_83555

theorem cara_neighbors (friends : Finset Person) (mark : Person) (cara : Person) (h_mark : mark ∈ friends) (h_len : friends.card = 8) :
  ∃ pairs : Finset (Person × Person), pairs.card = 6 ∧
    ∀ (p : Person × Person), p ∈ pairs → p.1 = mark ∨ p.2 = mark :=
by
  -- The proof goes here.
  sorry

end cara_neighbors_l835_83555


namespace fraction_of_repeating_decimal_l835_83562

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l835_83562


namespace total_num_animals_l835_83545

-- Given conditions
def num_pigs : ℕ := 10
def num_cows : ℕ := (2 * num_pigs) - 3
def num_goats : ℕ := num_cows + 6

-- Theorem statement
theorem total_num_animals : num_pigs + num_cows + num_goats = 50 := 
by
  sorry

end total_num_animals_l835_83545


namespace pages_allocation_correct_l835_83535

-- Define times per page for Alice, Bob, and Chandra
def t_A := 40
def t_B := 60
def t_C := 48

-- Define pages read by Alice, Bob, and Chandra
def pages_A := 295
def pages_B := 197
def pages_C := 420

-- Total pages in the novel
def total_pages := 912

-- Calculate the total time each one spends reading
def total_time_A := t_A * pages_A
def total_time_B := t_B * pages_B
def total_time_C := t_C * pages_C

-- Theorem: Prove the correct allocation of pages
theorem pages_allocation_correct : 
  total_pages = pages_A + pages_B + pages_C ∧
  total_time_A = total_time_B ∧
  total_time_B = total_time_C :=
by 
  -- Place end of proof here 
  sorry

end pages_allocation_correct_l835_83535


namespace paco_initial_cookies_l835_83554

theorem paco_initial_cookies (cookies_ate : ℕ) (cookies_left : ℕ) (cookies_initial : ℕ) 
  (h1 : cookies_ate = 15) (h2 : cookies_left = 78) :
  cookies_initial = cookies_ate + cookies_left → cookies_initial = 93 :=
by
  sorry

end paco_initial_cookies_l835_83554


namespace people_in_group_10_l835_83518

-- Let n represent the number of people in the group.
def number_of_people_in_group (n : ℕ) : Prop :=
  let average_increase : ℚ := 3.2
  let weight_of_replaced_person : ℚ := 65
  let weight_of_new_person : ℚ := 97
  let weight_increase : ℚ := weight_of_new_person - weight_of_replaced_person
  weight_increase = average_increase * n

theorem people_in_group_10 :
  ∃ n : ℕ, number_of_people_in_group n ∧ n = 10 :=
by
  sorry

end people_in_group_10_l835_83518


namespace range_of_a_l835_83526

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 :=
by {
  sorry
}

end range_of_a_l835_83526


namespace distance_squared_from_B_to_origin_l835_83591

-- Conditions:
-- 1. the radius of the circle is 10 cm
-- 2. the length of AB is 8 cm
-- 3. the length of BC is 3 cm
-- 4. the angle ABC is a right angle
-- 5. the center of the circle is at the origin
-- a^2 + b^2 is the square of the distance from B to the center of the circle (origin)

theorem distance_squared_from_B_to_origin
  (a b : ℝ)
  (h1 : a^2 + (b + 8)^2 = 100)
  (h2 : (a + 3)^2 + b^2 = 100)
  (h3 : 6 * a - 16 * b = 55) : a^2 + b^2 = 50 :=
sorry

end distance_squared_from_B_to_origin_l835_83591


namespace exam_total_students_l835_83517
-- Import the necessary Lean libraries

-- Define the problem conditions and the proof goal
theorem exam_total_students (T : ℕ) (h1 : 27 * T / 100 ≤ T) (h2 : 54 * T / 100 ≤ T) (h3 : 57 = 19 * T / 100) :
  T = 300 :=
  sorry  -- Proof is omitted here.

end exam_total_students_l835_83517


namespace max_value_of_y_is_2_l835_83595

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x

theorem max_value_of_y_is_2 (a : ℝ) (h : ∀ x : ℝ, (3 * x^2 + 2 * a * x + (a - 3)) = (3 * x^2 - 2 * a * x + (a - 3))) : 
  ∃ x : ℝ, f a x = 2 :=
sorry

end max_value_of_y_is_2_l835_83595


namespace fertilizer_production_l835_83560

theorem fertilizer_production (daily_production : ℕ) (days : ℕ) (total_production : ℕ) 
  (h1 : daily_production = 105) 
  (h2 : days = 24) 
  (h3 : total_production = daily_production * days) : 
  total_production = 2520 := 
  by 
  -- skipping the proof
  sorry

end fertilizer_production_l835_83560


namespace slope_product_l835_83590

   -- Define the hyperbola
   def hyperbola (x y : ℝ) : Prop := x^2 - (2 * y^2) / (Real.sqrt 5 + 1) = 1

   -- Define the slope calculation for points P, M, N on the hyperbola
   def slopes (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (Real.sqrt 5 + 1) / 2 = ((yP - y0) * (yP + y0)) / ((xP - x0) * (xP + x0)) := sorry
  
   -- Theorem to show the required relationship
   theorem slope_product (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (yP^2 - y0^2) / (xP^2 - x0^2) = (Real.sqrt 5 + 1) / 2 := sorry
   
end slope_product_l835_83590


namespace central_angle_of_sector_l835_83583

theorem central_angle_of_sector (r A θ : ℝ) (hr : r = 2) (hA : A = 4) :
  θ = 2 :=
by
  sorry

end central_angle_of_sector_l835_83583


namespace abs_neg_2022_eq_2022_l835_83589

theorem abs_neg_2022_eq_2022 : abs (-2022) = 2022 :=
by
  sorry

end abs_neg_2022_eq_2022_l835_83589


namespace inequality_proof_l835_83577

variables (a b c d : ℝ)

theorem inequality_proof (h1 : a > b) (h2 : c = d) : a + c > b + d :=
sorry

end inequality_proof_l835_83577


namespace apps_added_eq_sixty_l835_83569

-- Definitions derived from the problem conditions
def initial_apps : ℕ := 50
def removed_apps : ℕ := 10
def final_apps : ℕ := 100

-- Intermediate calculation based on the problem
def apps_after_removal : ℕ := initial_apps - removed_apps

-- The main theorem stating the mathematically equivalent proof problem
theorem apps_added_eq_sixty : final_apps - apps_after_removal = 60 :=
by
  sorry

end apps_added_eq_sixty_l835_83569


namespace division_multiplication_eval_l835_83575

theorem division_multiplication_eval : (18 / (5 + 2 - 3)) * 4 = 18 := 
by
  sorry

end division_multiplication_eval_l835_83575


namespace yola_past_weight_l835_83582

variable (W Y Y_past : ℕ)

-- Conditions
def condition1 : Prop := W = Y + 30
def condition2 : Prop := W = Y_past + 80
def condition3 : Prop := Y = 220

-- Theorem statement
theorem yola_past_weight : condition1 W Y → condition2 W Y_past → condition3 Y → Y_past = 170 :=
by
  intros h_condition1 h_condition2 h_condition3
  -- Placeholder for the proof, not required in the solution
  sorry

end yola_past_weight_l835_83582


namespace least_positive_integer_x_l835_83542

theorem least_positive_integer_x :
  ∃ x : ℕ, (x > 0) ∧ (∃ k : ℕ, (2 * x + 51) = k * 59) ∧ x = 4 :=
by
  -- Lean statement
  sorry

end least_positive_integer_x_l835_83542


namespace factor_expression_l835_83599

theorem factor_expression (a : ℝ) : 198 * a ^ 2 + 36 * a + 54 = 18 * (11 * a ^ 2 + 2 * a + 3) :=
by
  sorry

end factor_expression_l835_83599


namespace find_sum_l835_83504

variable (a b : ℚ)

theorem find_sum :
  2 * a + 5 * b = 31 ∧ 4 * a + 3 * b = 35 → a + b = 68 / 7 := by
  sorry

end find_sum_l835_83504


namespace f_injective_on_restricted_domain_l835_83550

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2)^2 - 5

-- Define the restricted domain
def f_restricted (x : ℝ) (h : -2 <= x) : ℝ := f x

-- The main statement to be proved
theorem f_injective_on_restricted_domain : 
  (∀ x1 x2 : {x // -2 <= x}, f_restricted x1.val x1.property = f_restricted x2.val x2.property → x1 = x2) := 
sorry

end f_injective_on_restricted_domain_l835_83550


namespace mod_inverse_13_997_l835_83529

-- The theorem statement
theorem mod_inverse_13_997 : ∃ x : ℕ, 0 ≤ x ∧ x < 997 ∧ (13 * x) % 997 = 1 ∧ x = 767 := 
by
  sorry

end mod_inverse_13_997_l835_83529


namespace frank_hamburger_goal_l835_83533

theorem frank_hamburger_goal:
  let price_per_hamburger := 5
  let group1_hamburgers := 2 * 4
  let group2_hamburgers := 2 * 2
  let current_hamburgers := group1_hamburgers + group2_hamburgers
  let extra_hamburgers_needed := 4
  let total_hamburgers := current_hamburgers + extra_hamburgers_needed
  price_per_hamburger * total_hamburgers = 80 :=
by
  sorry

end frank_hamburger_goal_l835_83533


namespace quadratic_has_two_distinct_real_roots_l835_83524

theorem quadratic_has_two_distinct_real_roots :
  ∃ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 6 ∧ a*x^2 + b*x + c = 0 → (b^2 - 4*a*c) > 0 := 
sorry

end quadratic_has_two_distinct_real_roots_l835_83524


namespace intersection_of_perpendicular_lines_l835_83521

theorem intersection_of_perpendicular_lines (x y : ℝ) : 
  (y = 3 * x + 4) ∧ (y = -1/3 * x + 4) → (x = 0 ∧ y = 4) :=
by
  sorry

end intersection_of_perpendicular_lines_l835_83521


namespace slope_positive_if_and_only_if_l835_83530

/-- Given points A(2, 1) and B(1, m^2), the slope of the line passing through them is positive,
if and only if m is in the range -1 < m < 1. -/
theorem slope_positive_if_and_only_if
  (m : ℝ) : 1 - m^2 > 0 ↔ -1 < m ∧ m < 1 :=
by
  sorry

end slope_positive_if_and_only_if_l835_83530


namespace equation_of_parabola_l835_83596

def parabola_vertex_form_vertex (a x y : ℝ) := y = a * (x - 3)^2 - 2
def parabola_passes_through_point (a : ℝ) := 1 = a * (0 - 3)^2 - 2
def parabola_equation (y x : ℝ) := y = (1/3) * x^2 - 2 * x + 1

theorem equation_of_parabola :
  ∃ a : ℝ,
    ∀ x y : ℝ,
      parabola_vertex_form_vertex a x y ∧
      parabola_passes_through_point a →
      parabola_equation y x :=
by
  sorry

end equation_of_parabola_l835_83596


namespace each_person_gets_equal_share_l835_83578

-- Definitions based on the conditions
def number_of_friends: Nat := 4
def initial_chicken_wings: Nat := 9
def additional_chicken_wings: Nat := 7

-- The proof statement
theorem each_person_gets_equal_share (total_chicken_wings := initial_chicken_wings + additional_chicken_wings) : 
       total_chicken_wings / number_of_friends = 4 := 
by 
  sorry

end each_person_gets_equal_share_l835_83578


namespace correctly_calculated_value_l835_83556

theorem correctly_calculated_value :
  ∀ (x : ℕ), (x * 15 = 45) → ((x * 5) * 10 = 150) := 
by
  intro x
  intro h
  sorry

end correctly_calculated_value_l835_83556


namespace triangle_max_third_side_l835_83508

theorem triangle_max_third_side (D E F : ℝ) (a b : ℝ) (h1 : a = 8) (h2 : b = 15) 
(h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1) 
: ∃ c : ℝ, c = 13 :=
by
  sorry

end triangle_max_third_side_l835_83508


namespace brian_books_chapters_l835_83584

variable (x : ℕ)

theorem brian_books_chapters (h1 : 1 ≤ x) (h2 : 20 + 2 * x + (20 + 2 * x) / 2 = 75) : x = 15 :=
sorry

end brian_books_chapters_l835_83584


namespace max_product_913_l835_83587

-- Define the condition that ensures the digits are from the set {3, 5, 8, 9, 1}
def valid_digits (digits : List ℕ) : Prop :=
  digits = [3, 5, 8, 9, 1]

-- Define the predicate for a valid three-digit and two-digit integer
def valid_numbers (a b c d e : ℕ) : Prop :=
  valid_digits [a, b, c, d, e] ∧
  ∃ x y, 100 * x + 10 * 1 + y = 10 * d + e ∧ d ≠ 1 ∧ a ≠ 1

-- Define the product function
def product (a b c d e : ℕ) : ℕ :=
  (100 * a + 10 * b + c) * (10 * d + e)

-- State the theorem
theorem max_product_913 : ∀ (a b c d e : ℕ), valid_numbers a b c d e → 
(product a b c d e) ≤ (product 9 1 3 8 5) :=
by
  intros a b c d e
  unfold valid_numbers product 
  sorry

end max_product_913_l835_83587


namespace contestant_final_score_l835_83586

theorem contestant_final_score (score_content score_skills score_effects : ℕ) 
                               (weight_content weight_skills weight_effects : ℕ) :
    score_content = 90 →
    score_skills  = 80 →
    score_effects = 90 →
    weight_content = 4 →
    weight_skills  = 2 →
    weight_effects = 4 →
    (score_content * weight_content + score_skills * weight_skills + score_effects * weight_effects) / 
    (weight_content + weight_skills + weight_effects) = 88 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end contestant_final_score_l835_83586


namespace option_a_correct_option_c_correct_option_d_correct_l835_83541

theorem option_a_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (1 / a > 1 / b) :=
sorry

theorem option_c_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (Real.sqrt (-a) > Real.sqrt (-b)) :=
sorry

theorem option_d_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (|a| > -b) :=
sorry

end option_a_correct_option_c_correct_option_d_correct_l835_83541


namespace lotion_cost_l835_83532

variable (shampoo_conditioner_cost lotion_total_spend: ℝ)
variable (num_lotions num_lotions_cost_target: ℕ)
variable (free_shipping_threshold additional_spend_needed: ℝ)

noncomputable def cost_of_each_lotion := lotion_total_spend / num_lotions

theorem lotion_cost
    (h1 : shampoo_conditioner_cost = 10)
    (h2 : num_lotions = 3)
    (h3 : additional_spend_needed = 12)
    (h4 : free_shipping_threshold = 50)
    (h5 : (shampoo_conditioner_cost * 2) + additional_spend_needed + lotion_total_spend = free_shipping_threshold) :
    cost_of_each_lotion = 10 :=
by
  sorry

end lotion_cost_l835_83532


namespace warriors_can_defeat_dragon_l835_83576

theorem warriors_can_defeat_dragon (n : ℕ) (h : n = 20^20) :
  (∀ n, n % 2 = 0 ∨ n % 3 = 0) → (∃ m, m = 0) := 
sorry

end warriors_can_defeat_dragon_l835_83576


namespace cans_purchased_l835_83519

variable (N P T : ℕ)

theorem cans_purchased (N P T : ℕ) : N * (5 * (T - 1)) / P = 5 * N * (T - 1) / P :=
by
  sorry

end cans_purchased_l835_83519


namespace area_ratio_of_circles_l835_83574

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end area_ratio_of_circles_l835_83574


namespace expression_for_B_A_greater_than_B_l835_83585

-- Define the polynomials A and B
def A (x : ℝ) := 3 * x^2 - 2 * x + 1
def B (x : ℝ) := 2 * x^2 - x - 3

-- Prove that the given expression for B validates the equation A + B = 5x^2 - 4x - 2.
theorem expression_for_B (x : ℝ) : A x + 2 * x^2 - x - 3 = 5 * x^2 - 4 * x - 2 :=
by {
  sorry
}

-- Prove that A is always greater than B for all values of x.
theorem A_greater_than_B (x : ℝ) : A x > B x :=
by {
  sorry
}

end expression_for_B_A_greater_than_B_l835_83585


namespace distribute_pencils_l835_83571

def number_of_ways_to_distribute_pencils (pencils friends : ℕ) : ℕ :=
  Nat.choose (pencils - friends + friends - 1) (friends - 1)

theorem distribute_pencils :
  number_of_ways_to_distribute_pencils 4 4 = 35 :=
by
  sorry

end distribute_pencils_l835_83571


namespace ratio_of_sums_l835_83515

theorem ratio_of_sums (total_sums : ℕ) (correct_sums : ℕ) (incorrect_sums : ℕ)
  (h1 : total_sums = 75)
  (h2 : incorrect_sums = 2 * correct_sums)
  (h3 : total_sums = correct_sums + incorrect_sums) :
  incorrect_sums / correct_sums = 2 :=
by
  -- Proof placeholder
  sorry

end ratio_of_sums_l835_83515


namespace prob_not_same_city_l835_83570

def prob_A_city_A : ℝ := 0.6
def prob_B_city_A : ℝ := 0.3

theorem prob_not_same_city :
  (prob_A_city_A * (1 - prob_B_city_A) + (1 - prob_A_city_A) * prob_B_city_A) = 0.54 :=
by 
  -- This is just a placeholder to indicate that the proof is skipped
  sorry

end prob_not_same_city_l835_83570


namespace running_speed_equiv_l835_83564

variable (R : ℝ)
variable (walking_speed : ℝ) (total_distance : ℝ) (total_time: ℝ) (distance_walked : ℝ) (distance_ran : ℝ)

theorem running_speed_equiv :
  walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4 →
  1 + (4 / R) = 1.5 →
  R = 8 :=
by
  intros H1 H2
  -- H1: Condition set (walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4)
  -- H2: Equation (1 + (4 / R) = 1.5)
  sorry

end running_speed_equiv_l835_83564


namespace isabella_hair_length_end_of_year_l835_83513

/--
Isabella's initial hair length.
-/
def initial_hair_length : ℕ := 18

/--
Isabella's hair growth over the year.
-/
def hair_growth : ℕ := 6

/--
Prove that Isabella's hair length at the end of the year is 24 inches.
-/
theorem isabella_hair_length_end_of_year : initial_hair_length + hair_growth = 24 := by
  sorry

end isabella_hair_length_end_of_year_l835_83513


namespace dichromate_molecular_weight_l835_83572

theorem dichromate_molecular_weight :
  let atomic_weight_Cr := 52.00
  let atomic_weight_O := 16.00
  let dichromate_num_Cr := 2
  let dichromate_num_O := 7
  (dichromate_num_Cr * atomic_weight_Cr + dichromate_num_O * atomic_weight_O) = 216.00 :=
by
  sorry

end dichromate_molecular_weight_l835_83572


namespace non_zero_real_y_satisfies_l835_83503

theorem non_zero_real_y_satisfies (y : ℝ) (h : y ≠ 0) : (8 * y) ^ 3 = (16 * y) ^ 2 → y = 1 / 2 :=
by
  -- Lean code placeholders
  sorry

end non_zero_real_y_satisfies_l835_83503


namespace lobster_distribution_l835_83539

theorem lobster_distribution :
  let HarborA := 50
  let HarborB := 70.5
  let HarborC := (2 / 3) * HarborB
  let HarborD := HarborA - 0.15 * HarborA
  let Sum := HarborA + HarborB + HarborC + HarborD
  let HooperBay := 3 * Sum
  let Total := HooperBay + Sum
  Total = 840 := by
  sorry

end lobster_distribution_l835_83539


namespace smallest_possible_denominator_l835_83552

theorem smallest_possible_denominator :
  ∃ p q : ℕ, q < 4027 ∧ (1/2014 : ℚ) < p / q ∧ p / q < (1/2013 : ℚ) → ∃ q : ℕ, q = 4027 :=
by
  sorry

end smallest_possible_denominator_l835_83552


namespace stratified_sampling_probability_l835_83544

open Finset Nat

noncomputable def combin (n k : ℕ) : ℕ := choose n k

theorem stratified_sampling_probability :
  let total_balls := 40
  let red_balls := 16
  let blue_balls := 12
  let white_balls := 8
  let yellow_balls := 4
  let n_draw := 10
  let red_draw := 4
  let blue_draw := 3
  let white_draw := 2
  let yellow_draw := 1
  
  combin yellow_balls yellow_draw * combin white_balls white_draw * combin blue_balls blue_draw * combin red_balls red_draw = combin total_balls n_draw :=
sorry

end stratified_sampling_probability_l835_83544


namespace gaoan_total_revenue_in_scientific_notation_l835_83507

theorem gaoan_total_revenue_in_scientific_notation :
  (21 * 10^9 : ℝ) = 2.1 * 10^9 :=
sorry

end gaoan_total_revenue_in_scientific_notation_l835_83507


namespace committee_count_l835_83592

theorem committee_count (club_members : Finset ℕ) (h_count : club_members.card = 30) :
  ∃ committee_count : ℕ, committee_count = 2850360 :=
by
  sorry

end committee_count_l835_83592


namespace range_of_a_l835_83516

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → 3 * x - a ≥ 0) → a ≤ 6 :=
by
  intros h
  sorry

end range_of_a_l835_83516


namespace positive_numbers_inequality_l835_83568

theorem positive_numbers_inequality
  (x y z : ℝ)
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x * y + y * z + z * x = 6) :
  (1 / (2 * Real.sqrt 2 + x^2 * (y + z)) + 
   1 / (2 * Real.sqrt 2 + y^2 * (x + z)) + 
   1 / (2 * Real.sqrt 2 + z^2 * (x + y))) <= 
  (1 / (x * y * z)) :=
by
  sorry

end positive_numbers_inequality_l835_83568


namespace value_of_a_plus_d_l835_83511

variable (a b c d : ℝ)

theorem value_of_a_plus_d 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := 
by 
  sorry

end value_of_a_plus_d_l835_83511


namespace cost_of_staying_23_days_l835_83537

def hostel_cost (days: ℕ) : ℝ :=
  if days ≤ 7 then
    days * 18
  else
    7 * 18 + (days - 7) * 14

theorem cost_of_staying_23_days : hostel_cost 23 = 350 :=
by
  sorry

end cost_of_staying_23_days_l835_83537


namespace circles_ordered_by_radius_l835_83558

def circle_radii_ordered (rA rB rC : ℝ) : Prop :=
  rA < rC ∧ rC < rB

theorem circles_ordered_by_radius :
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  circle_radii_ordered rA rB rC :=
by
  intros
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  show circle_radii_ordered rA rB rC
  sorry

end circles_ordered_by_radius_l835_83558


namespace next_term_geometric_sequence_l835_83520

noncomputable def geometric_term (a r : ℕ) (n : ℕ) : ℕ :=
a * r^n

theorem next_term_geometric_sequence (y : ℕ) :
  ∀ a₁ a₂ a₃ a₄, a₁ = 3 → a₂ = 9 * y → a₃ = 27 * y^2 → a₄ = 81 * y^3 →
  geometric_term 3 (3 * y) 4 = 243 * y^4 :=
by
  intros a₁ a₂ a₃ a₄ h₁ h₂ h₃ h₄
  sorry

end next_term_geometric_sequence_l835_83520


namespace problem1_problem2_problem3_problem4_problem5_problem6_l835_83581

-- Problem 1
theorem problem1 : (-20 + 3 - (-5) - 7 : Int) = -19 := sorry

-- Problem 2
theorem problem2 : (-2.4 - 3.7 - 4.6 + 5.7 : Real) = -5 := sorry

-- Problem 3
theorem problem3 : (-0.25 + ((-3 / 7) * (4 / 5)) : Real) = (-83 / 140) := sorry

-- Problem 4
theorem problem4 : ((-1 / 2) * (-8) + (-6)^2 : Real) = 40 := sorry

-- Problem 5
theorem problem5 : ((-1 / 12 - 1 / 36 + 1 / 6) * (-36) : Real) = -2 := sorry

-- Problem 6
theorem problem6 : (-1^4 + (-2) + (-1 / 3) - abs (-9) : Real) = -37 / 3 := sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l835_83581


namespace total_handshakes_five_people_l835_83509

theorem total_handshakes_five_people : 
  let n := 5
  let total_handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2
  total_handshakes 5 = 10 :=
by sorry

end total_handshakes_five_people_l835_83509


namespace single_interval_condition_l835_83514

-- Definitions: k and l are integers
variables (k l : ℤ)

-- Condition: The given condition for l
theorem single_interval_condition : l = Int.floor (k ^ 2 / 4) :=
sorry

end single_interval_condition_l835_83514


namespace movie_box_office_growth_l835_83566

theorem movie_box_office_growth 
  (x : ℝ) 
  (r₁ r₃ : ℝ) 
  (h₁ : r₁ = 1) 
  (h₃ : r₃ = 2.4) 
  (growth : r₃ = (1 + x) ^ 2) : 
  (1 + x) ^ 2 = 2.4 :=
by sorry

end movie_box_office_growth_l835_83566


namespace percentage_of_non_defective_products_l835_83540

-- Define the conditions
def totalProduction : ℕ := 100
def M1_production : ℕ := 25
def M2_production : ℕ := 35
def M3_production : ℕ := 40

def M1_defective_rate : ℝ := 0.02
def M2_defective_rate : ℝ := 0.04
def M3_defective_rate : ℝ := 0.05

-- Calculate the total defective units
noncomputable def total_defective_units : ℝ := 
  (M1_defective_rate * M1_production) + 
  (M2_defective_rate * M2_production) + 
  (M3_defective_rate * M3_production)

-- Calculate the percentage of defective products
noncomputable def defective_percentage : ℝ := (total_defective_units / totalProduction) * 100

-- Calculate the percentage of non-defective products
noncomputable def non_defective_percentage : ℝ := 100 - defective_percentage

-- The statement to prove
theorem percentage_of_non_defective_products :
  non_defective_percentage = 96.1 :=
by
  sorry

end percentage_of_non_defective_products_l835_83540


namespace simplify_expr_l835_83598

variable (a b : ℝ)

def expr := a * b - (a^2 - a * b + b^2)

theorem simplify_expr : expr a b = - a^2 + 2 * a * b - b^2 :=
by 
  -- No proof is provided as per the instructions
  sorry

end simplify_expr_l835_83598


namespace inequality_solution_set_l835_83543

theorem inequality_solution_set (x : ℝ) :
  (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 := by
sorry

end inequality_solution_set_l835_83543


namespace jason_current_cards_l835_83563

-- Define the initial number of Pokemon cards Jason had.
def initial_cards : ℕ := 9

-- Define the number of Pokemon cards Jason gave to his friends.
def given_away : ℕ := 4

-- Prove that the number of Pokemon cards he has now is 5.
theorem jason_current_cards : initial_cards - given_away = 5 := by
  sorry

end jason_current_cards_l835_83563


namespace rectangle_area_from_square_l835_83597

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l835_83597


namespace smaller_two_digit_product_is_34_l835_83527

theorem smaller_two_digit_product_is_34 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 5082) : min a b = 34 :=
by
  sorry

end smaller_two_digit_product_is_34_l835_83527


namespace min_n_for_binomial_constant_term_l835_83565

theorem min_n_for_binomial_constant_term : ∃ (n : ℕ), n > 0 ∧ 3 * n - 7 * ((3 * n) / 7) = 0 ∧ n = 7 :=
by {
  sorry
}

end min_n_for_binomial_constant_term_l835_83565


namespace maximize_profit_l835_83531

noncomputable def profit_function (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 40 then
  -2 * x^2 + 120 * x - 300
else if 40 < x ∧ x ≤ 100 then
  -x - 3600 / x + 1800
else
  0

theorem maximize_profit :
  profit_function 60 = 1680 ∧
  ∀ x, 0 < x ∧ x ≤ 100 → profit_function x ≤ 1680 := 
sorry

end maximize_profit_l835_83531


namespace apples_per_friend_l835_83547

def Benny_apples : Nat := 5
def Dan_apples : Nat := 2 * Benny_apples
def Total_apples : Nat := Benny_apples + Dan_apples
def Number_of_friends : Nat := 3

theorem apples_per_friend : Total_apples / Number_of_friends = 5 := by
  sorry

end apples_per_friend_l835_83547


namespace even_function_value_for_negative_x_l835_83588

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_value_for_negative_x (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_pos : ∀ (x : ℝ), 0 < x → f x = 10^x) :
  ∀ x : ℝ, x < 0 → f x = 10^(-x) :=
by
  sorry

end even_function_value_for_negative_x_l835_83588


namespace bug_total_distance_l835_83502

theorem bug_total_distance 
  (p₀ p₁ p₂ p₃ : ℤ) 
  (h₀ : p₀ = 0) 
  (h₁ : p₁ = 4) 
  (h₂ : p₂ = -3) 
  (h₃ : p₃ = 7) : 
  |p₁ - p₀| + |p₂ - p₁| + |p₃ - p₂| = 21 :=
by 
  sorry

end bug_total_distance_l835_83502


namespace number_of_girls_in_school_l835_83506

/-- Statement: There are 408 boys and some girls in a school which are to be divided into equal sections
of either boys or girls alone. The total number of sections thus formed is 26. Prove that the number 
of girls is 216. -/
theorem number_of_girls_in_school (n : ℕ) (n_boys : ℕ := 408) (total_sections : ℕ := 26)
  (h1 : n_boys = 408)
  (h2 : ∃ b g : ℕ, b + g = total_sections ∧ 408 / b = n / g ∧ b ∣ 408 ∧ g ∣ n) :
  n = 216 :=
by
  -- Proof would go here
  sorry

end number_of_girls_in_school_l835_83506


namespace john_annual_profit_l835_83536

namespace JohnProfit

def number_of_people_subletting := 3
def rent_per_person_per_month := 400
def john_rent_per_month := 900
def months_in_year := 12

theorem john_annual_profit 
  (h1 : number_of_people_subletting = 3)
  (h2 : rent_per_person_per_month = 400)
  (h3 : john_rent_per_month = 900)
  (h4 : months_in_year = 12) : 
  (number_of_people_subletting * rent_per_person_per_month - john_rent_per_month) * months_in_year = 3600 :=
by
  sorry

end JohnProfit

end john_annual_profit_l835_83536


namespace find_p_l835_83510

variables (a b c p : ℝ)

theorem find_p 
  (h1 : 9 / (a + b) = 13 / (c - b)) : 
  p = 22 :=
sorry

end find_p_l835_83510


namespace minimize_travel_expense_l835_83548

noncomputable def travel_cost_A (x : ℕ) : ℝ := 2000 * x * 0.75
noncomputable def travel_cost_B (x : ℕ) : ℝ := 2000 * (x - 1) * 0.8

theorem minimize_travel_expense (x : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 25) :
  (10 ≤ x ∧ x ≤ 15 → travel_cost_B x < travel_cost_A x) ∧
  (x = 16 → travel_cost_A x = travel_cost_B x) ∧
  (17 ≤ x ∧ x ≤ 25 → travel_cost_A x < travel_cost_B x) :=
by
  sorry

end minimize_travel_expense_l835_83548


namespace arith_seq_sum_first_110_l835_83512

variable {α : Type*} [OrderedRing α]

theorem arith_seq_sum_first_110 (a₁ d : α) :
  (10 * a₁ + 45 * d = 100) →
  (100 * a₁ + 4950 * d = 10) →
  (110 * a₁ + 5995 * d = -110) :=
by
  intros h1 h2
  sorry

end arith_seq_sum_first_110_l835_83512


namespace watermelon_percentage_l835_83538

theorem watermelon_percentage (total_drink : ℕ)
  (orange_percentage : ℕ)
  (grape_juice : ℕ)
  (watermelon_amount : ℕ)
  (W : ℕ) :
  total_drink = 300 →
  orange_percentage = 25 →
  grape_juice = 105 →
  watermelon_amount = total_drink - (orange_percentage * total_drink) / 100 - grape_juice →
  W = (watermelon_amount * 100) / total_drink →
  W = 40 :=
sorry

end watermelon_percentage_l835_83538


namespace hypotenuse_length_is_13_l835_83501

theorem hypotenuse_length_is_13 (a b c : ℝ) (ha : a = 5) (hb : b = 12)
  (hrt : a ^ 2 + b ^ 2 = c ^ 2) : c = 13 :=
by
  -- to complete the proof, fill in the details here
  sorry

end hypotenuse_length_is_13_l835_83501


namespace total_amount_l835_83528

theorem total_amount (x : ℝ) (hC : 2 * x = 70) :
  let B_share := 1.25 * x
  let C_share := 2 * x
  let D_share := 0.7 * x
  let E_share := 0.5 * x
  let A_share := x
  B_share + C_share + D_share + E_share + A_share = 190.75 :=
by
  sorry

end total_amount_l835_83528


namespace find_x2_plus_y2_l835_83594

theorem find_x2_plus_y2 (x y : ℕ) (h1 : xy + x + y = 35) (h2 : x^2 * y + x * y^2 = 306) : x^2 + y^2 = 290 :=
sorry

end find_x2_plus_y2_l835_83594


namespace stratified_sampling_counts_l835_83561

-- Defining the given conditions
def num_elderly : ℕ := 27
def num_middle_aged : ℕ := 54
def num_young : ℕ := 81
def total_sample : ℕ := 42

-- Proving the required stratified sample counts
theorem stratified_sampling_counts :
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  elderly_count = 7 ∧ middle_aged_count = 14 ∧ young_count = 21 :=
by 
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  have h1 : elderly_count = 7 := by sorry
  have h2 : middle_aged_count = 14 := by sorry
  have h3 : young_count = 21 := by sorry
  exact ⟨h1, h2, h3⟩

end stratified_sampling_counts_l835_83561


namespace problem_statement_l835_83551

noncomputable def f (x : ℝ) := Real.log 9 * (Real.log x / Real.log 3)

theorem problem_statement : deriv f 2 + deriv f 2 = 1 := sorry

end problem_statement_l835_83551


namespace salon_customers_l835_83580

theorem salon_customers (C : ℕ) (H : C * 2 + 5 = 33) : C = 14 :=
by {
  sorry
}

end salon_customers_l835_83580


namespace smallest_sum_of_four_consecutive_primes_divisible_by_five_l835_83523

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_sum_of_four_consecutive_primes_divisible_by_five :
  ∃ (a b c d : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    a < b ∧ b < c ∧ c < d ∧
    b = a + 2 ∧ c = b + 4 ∧ d = c + 2 ∧
    (a + b + c + d) % 5 = 0 ∧ (a + b + c + d = 60) := sorry

end smallest_sum_of_four_consecutive_primes_divisible_by_five_l835_83523


namespace total_cows_l835_83549

/-- A farmer divides his herd of cows among his four sons.
The first son receives 1/3 of the herd, the second son receives 1/6,
the third son receives 1/9, and the rest goes to the fourth son,
who receives 12 cows. Calculate the total number of cows in the herd
-/
theorem total_cows (n : ℕ) (h1 : (n : ℚ) * (1 / 3) + (n : ℚ) * (1 / 6) + (n : ℚ) * (1 / 9) + 12 = n) : n = 54 := by
  sorry

end total_cows_l835_83549


namespace reciprocal_of_neg3_l835_83579

theorem reciprocal_of_neg3 : 1 / (-3: ℝ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l835_83579


namespace opposite_numbers_l835_83593

theorem opposite_numbers (a b : ℝ) (h : a = -b) : b = -a := 
by 
  sorry

end opposite_numbers_l835_83593


namespace max_three_digit_sum_l835_83567

theorem max_three_digit_sum :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ 101 * A + 11 * B + 11 * C = 986 := 
sorry

end max_three_digit_sum_l835_83567


namespace remainder_division_l835_83505

theorem remainder_division (exists_quotient : ∃ q r : ℕ, r < 5 ∧ N = 5 * 5 + r)
    (exists_quotient_prime : ∃ k : ℕ, N = 11 * k + 3) :
  ∃ r : ℕ, r = 0 ∧ N % 5 = r := 
sorry

end remainder_division_l835_83505


namespace find_square_digit_l835_83525

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_digits_31_42_7s (s : ℕ) : ℕ :=
  3 + 1 + 4 + 2 + 7 + s

-- The main theorem to prove
theorem find_square_digit (d : ℕ) (h0 : is_even d) (h1 : (sum_digits_31_42_7s d) % 3 = 0) : d = 4 :=
by
  sorry

end find_square_digit_l835_83525


namespace cube_face_min_sum_l835_83557

open Set

theorem cube_face_min_sum (S : Finset ℕ)
  (hS : S = {1, 2, 3, 4, 5, 6, 7, 8})
  (h_faces_sum : ∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → 
                    (a + b + c >= 10) ∨ 
                    (a + b + d >= 10) ∨ 
                    (a + c + d >= 10) ∨ 
                    (b + c + d >= 10)) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 16 :=
sorry

end cube_face_min_sum_l835_83557


namespace smallest_six_digit_odd_div_by_125_l835_83522

theorem smallest_six_digit_odd_div_by_125 : 
  ∃ n : ℕ, n = 111375 ∧ 
           100000 ≤ n ∧ n < 1000000 ∧ 
           (∀ d : ℕ, d ∈ (n.digits 10) → d % 2 = 1) ∧ 
           n % 125 = 0 :=
by
  sorry

end smallest_six_digit_odd_div_by_125_l835_83522


namespace ellipse_x_intercepts_l835_83546

noncomputable def distances_sum (x : ℝ) (y : ℝ) (f₁ f₂ : ℝ × ℝ) :=
  (Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2)) + (Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2))

def is_on_ellipse (x y : ℝ) : Prop := 
  distances_sum x y (0, 3) (4, 0) = 7

theorem ellipse_x_intercepts 
  (h₀ : is_on_ellipse 0 0) 
  (hx_intercept : ∀ x : ℝ, is_on_ellipse x 0 → x = 0 ∨ x = 20 / 7) :
  ∀ x : ℝ, is_on_ellipse x 0 ↔ x = 0 ∨ x = 20 / 7 :=
by
  sorry

end ellipse_x_intercepts_l835_83546
