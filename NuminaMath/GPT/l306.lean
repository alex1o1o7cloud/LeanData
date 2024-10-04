import Mathlib

namespace percent_students_two_novels_l306_306346

theorem percent_students_two_novels :
  let total_students := 240
  let students_three_or_more := (1/6 : ℚ) * total_students
  let students_one := (5/12 : ℚ) * total_students
  let students_none := 16
  let students_two := total_students - students_three_or_more - students_one - students_none
  (students_two / total_students) * 100 = 35 := 
by
  sorry

end percent_students_two_novels_l306_306346


namespace area_difference_of_circles_l306_306831

theorem area_difference_of_circles : 
  let r1 := 30
  let r2 := 15
  let pi := Real.pi
  900 * pi - 225 * pi = 675 * pi := by
  sorry

end area_difference_of_circles_l306_306831


namespace sum_divisible_by_ten_l306_306700

    -- Given conditions
    def is_natural_number (n : ℕ) : Prop := true

    -- Sum S as defined in the conditions
    def S (n : ℕ) : ℕ := n ^ 2 + (n + 1) ^ 2 + (n + 2) ^ 2 + (n + 3) ^ 2

    -- The equivalent math proof problem in Lean 4 statement
    theorem sum_divisible_by_ten (n : ℕ) : S n % 10 = 0 ↔ n % 5 = 1 := by
      sorry
    
end sum_divisible_by_ten_l306_306700


namespace mistaken_fraction_l306_306449

theorem mistaken_fraction (n correct_result student_result : ℕ) (h1 : n = 384)
  (h2 : correct_result = (5 * n) / 16) (h3 : student_result = correct_result + 200) : 
  (student_result / n : ℚ) = 5 / 6 :=
by
  sorry

end mistaken_fraction_l306_306449


namespace g_at_five_l306_306269

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_five :
  (∀ x : ℝ, g (3 * x - 7) = 4 * x + 6) →
  g (5) = 22 :=
by
  intros h
  sorry

end g_at_five_l306_306269


namespace find_room_length_l306_306482

variable (width : ℝ) (cost rate : ℝ) (length : ℝ)

theorem find_room_length (h_width : width = 4.75)
  (h_cost : cost = 34200)
  (h_rate : rate = 900)
  (h_area : cost / rate = length * width) :
  length = 8 :=
sorry

end find_room_length_l306_306482


namespace sugar_in_lollipop_l306_306164

-- Definitions based on problem conditions
def chocolate_bars := 14
def sugar_per_bar := 10
def total_sugar := 177

-- The theorem to prove
theorem sugar_in_lollipop : total_sugar - (chocolate_bars * sugar_per_bar) = 37 :=
by
  -- we are not providing the proof, hence using sorry
  sorry

end sugar_in_lollipop_l306_306164


namespace player_placing_third_won_against_seventh_l306_306347

theorem player_placing_third_won_against_seventh :
  ∃ (s : Fin 8 → ℚ),
    -- Condition 1: Scores are different
    (∀ i j, i ≠ j → s i ≠ s j) ∧
    -- Condition 2: Second place score equals the sum of the bottom four scores
    (s 1 = s 4 + s 5 + s 6 + s 7) ∧
    -- Result: Third player won against the seventh player
    (s 2 > s 6) :=
sorry

end player_placing_third_won_against_seventh_l306_306347


namespace max_side_of_triangle_l306_306055

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306055


namespace coupon1_greater_l306_306804

variable (x : ℝ)

def coupon1_discount (x : ℝ) : ℝ := 0.15 * x
def coupon2_discount : ℝ := 50
def coupon3_discount (x : ℝ) : ℝ := 0.25 * x - 62.5

theorem coupon1_greater (x : ℝ) (hx1 : 333.33 < x ∧ x < 625) : 
  coupon1_discount x > coupon2_discount ∧ coupon1_discount x > coupon3_discount x := by
  sorry

end coupon1_greater_l306_306804


namespace exists_rationals_leq_l306_306968

theorem exists_rationals_leq (f : ℚ → ℤ) : ∃ a b : ℚ, (f a + f b) / 2 ≤ f (a + b) / 2 :=
by
  sorry

end exists_rationals_leq_l306_306968


namespace trader_profit_l306_306359

-- Definitions and conditions
def original_price (P : ℝ) := P
def discounted_price (P : ℝ) := 0.70 * P
def marked_up_price (P : ℝ) := 0.84 * P
def sale_price (P : ℝ) := 0.714 * P
def final_price (P : ℝ) := 1.2138 * P

-- Proof statement
theorem trader_profit (P : ℝ) : ((final_price P - original_price P) / original_price P) * 100 = 21.38 := by
  sorry

end trader_profit_l306_306359


namespace limonia_largest_none_providable_amount_l306_306745

def is_achievable (n : ℕ) (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), x = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10)

theorem limonia_largest_none_providable_amount (n : ℕ) : 
  ∃ s, ¬ is_achievable n s ∧ (∀ t, t > s → is_achievable n t) ∧ s = 12 * n^2 + 14 * n - 1 :=
by
  sorry

end limonia_largest_none_providable_amount_l306_306745


namespace naomi_total_wheels_l306_306231

theorem naomi_total_wheels 
  (regular_bikes : ℕ) (children_bikes : ℕ) (tandem_bikes_4_wheels : ℕ) (tandem_bikes_6_wheels : ℕ)
  (wheels_per_regular_bike : ℕ) (wheels_per_children_bike : ℕ) (wheels_per_tandem_4wheel : ℕ) (wheels_per_tandem_6wheel : ℕ) :
  regular_bikes = 7 →
  children_bikes = 11 →
  tandem_bikes_4_wheels = 5 →
  tandem_bikes_6_wheels = 3 →
  wheels_per_regular_bike = 2 →
  wheels_per_children_bike = 4 →
  wheels_per_tandem_4wheel = 4 →
  wheels_per_tandem_6wheel = 6 →
  (regular_bikes * wheels_per_regular_bike) + 
  (children_bikes * wheels_per_children_bike) + 
  (tandem_bikes_4_wheels * wheels_per_tandem_4wheel) + 
  (tandem_bikes_6_wheels * wheels_per_tandem_6wheel) = 96 := 
by
  intros; sorry

end naomi_total_wheels_l306_306231


namespace y_equals_x_l306_306868

theorem y_equals_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 1 + 1 / y) (h2 : y = 1 + 1 / x) : y = x :=
sorry

end y_equals_x_l306_306868


namespace min_value_144_l306_306566

noncomputable def min_expression (a b c d : ℝ) : ℝ :=
  (a + b + c) / (a * b * c * d)

theorem min_value_144 (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_d : 0 < d) (h_sum : a + b + c + d = 2) : min_expression a b c d ≥ 144 :=
by
  sorry

end min_value_144_l306_306566


namespace intersection_A_B_is_1_and_2_l306_306716

-- Define sets A and B as per the given conditions
def setA : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 3}
def setB : Set ℚ := {x : ℚ | 0 < x ∧ x < 3}

-- Assert the intersection of A and B is {1,2}
theorem intersection_A_B_is_1_and_2 : 
  (setA.inter {x : ℤ | ∃ (q : ℚ), x = q ∧ 0 < q ∧ q < 3}) = {1, 2} :=
sorry

end intersection_A_B_is_1_and_2_l306_306716


namespace max_side_length_is_11_l306_306017

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306017


namespace volleyball_club_lineups_l306_306167
-- Import the required Lean library

-- Define the main problem
theorem volleyball_club_lineups :
  let total_players := 18
  let quadruplets := 4
  let starters := 6
  let eligible_lineups := Nat.choose 18 6 - Nat.choose 14 2 - Nat.choose 14 6
  eligible_lineups = 15470 :=
by
  sorry

end volleyball_club_lineups_l306_306167


namespace geom_prog_roots_a_eq_22_l306_306116

theorem geom_prog_roots_a_eq_22 (x1 x2 x3 a : ℝ) :
  (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) → 
  (∃ b q, (x1 = b ∧ x2 = b * q ∧ x3 = b * q^2) ∧ (x1 + x2 + x3 = 11) ∧ (x1 * x2 * x3 = 8) ∧ (x1*x2 + x2*x3 + x3*x1 = a)) → 
  a = 22 :=
sorry

end geom_prog_roots_a_eq_22_l306_306116


namespace value_of_coupon_l306_306270

theorem value_of_coupon (price_per_bag : ℝ) (oz_per_bag : ℕ) (cost_per_serving_with_coupon : ℝ) (total_servings : ℕ) :
  price_per_bag = 25 → oz_per_bag = 40 → cost_per_serving_with_coupon = 0.50 → total_servings = 40 →
  (price_per_bag - (cost_per_serving_with_coupon * total_servings)) = 5 :=
by 
  intros hpb hob hcpwcs hts
  sorry

end value_of_coupon_l306_306270


namespace trigonometric_identity_l306_306141

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = -3) :
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 :=
  sorry

end trigonometric_identity_l306_306141


namespace sets_B_C_D_represent_same_function_l306_306092

theorem sets_B_C_D_represent_same_function :
  (∀ x : ℝ, (2 * x = 2 * (x ^ (3 : ℝ) ^ (1 / 3)))) ∧
  (∀ x t : ℝ, (x ^ 2 + x + 3 = t ^ 2 + t + 3)) ∧
  (∀ x : ℝ, (x ^ 2 = (x ^ 4) ^ (1 / 2))) :=
by
  sorry

end sets_B_C_D_represent_same_function_l306_306092


namespace arthur_walked_distance_in_miles_l306_306512

def blocks_west : ℕ := 8
def blocks_south : ℕ := 10
def block_length_in_miles : ℚ := 1 / 4

theorem arthur_walked_distance_in_miles : 
  (blocks_west + blocks_south) * block_length_in_miles = 4.5 := by
sorry

end arthur_walked_distance_in_miles_l306_306512


namespace novels_next_to_each_other_l306_306798

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem novels_next_to_each_other (n_essays n_novels : Nat) (condition_novels : n_novels = 2) (condition_essays : n_essays = 3) :
  let total_units := (n_novels - 1) + n_essays
  factorial total_units * factorial n_novels = 48 :=
by
  sorry

end novels_next_to_each_other_l306_306798


namespace max_three_digit_sum_l306_306724

theorem max_three_digit_sum : ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (0 ≤ A ∧ A < 10) ∧ (0 ≤ B ∧ B < 10) ∧ (0 ≤ C ∧ C < 10) ∧ (111 * A + 10 * C + 2 * B = 976) := sorry

end max_three_digit_sum_l306_306724


namespace min_value_64_l306_306462

noncomputable def min_value_expr (a b c d e f g h : ℝ) : ℝ :=
  (a * e) ^ 2 + (b * f) ^ 2 + (c * g) ^ 2 + (d * h) ^ 2

theorem min_value_64 
  (a b c d e f g h : ℝ) 
  (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16)
  (h3 : a + b + c + d = e * f * g) :
  min_value_expr a b c d e f g h = 64 := 
sorry

end min_value_64_l306_306462


namespace shaded_area_difference_l306_306478

theorem shaded_area_difference (A1 A3 A4 : ℚ) (h1 : 4 = 2 * 2) (h2 : A1 + 5 * A1 + 7 * A1 = 6) (h3 : p + q = 49) : 
  ∃ p q : ℕ, p + q = 49 ∧ p = 36 ∧ q = 13 :=
by {
  sorry
}

end shaded_area_difference_l306_306478


namespace max_side_length_l306_306007

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306007


namespace find_other_num_l306_306481

variables (a b : ℕ)

theorem find_other_num (h_gcd : Nat.gcd a b = 12) (h_lcm : Nat.lcm a b = 5040) (h_a : a = 240) :
  b = 252 :=
  sorry

end find_other_num_l306_306481


namespace positive_difference_x_coordinates_lines_l306_306257

theorem positive_difference_x_coordinates_lines :
  let l := fun x : ℝ => -2 * x + 4
  let m := fun x : ℝ => - (1 / 5) * x + 1
  let x_l := (- (10 - 4) / 2)
  let x_m := (- (10 - 1) * 5)
  abs (x_l - x_m) = 42 := by
  sorry

end positive_difference_x_coordinates_lines_l306_306257


namespace problem_solution_l306_306461

theorem problem_solution
  (N1 N2 : ℤ)
  (h : ∀ x : ℝ, 50 * x - 42 ≠ 0 → x ≠ 2 → x ≠ 3 → 
    (50 * x - 42) / (x ^ 2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) : 
  N1 * N2 = -6264 :=
sorry

end problem_solution_l306_306461


namespace converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l306_306902

theorem converse_of_P (a b : ℤ) : (a - 2 > b - 2) → (a > b) :=
by
  intro h
  exact sorry

theorem inverse_of_P (a b : ℤ) : (a ≤ b) → (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

theorem contrapositive_of_P (a b : ℤ) : (a - 2 ≤ b - 2) → (a ≤ b) :=
by
  intro h
  exact sorry

theorem negation_of_P (a b : ℤ) : (a > b) → ¬ (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

end converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l306_306902


namespace value_of_complex_fraction_l306_306758

theorem value_of_complex_fraction (i : ℂ) (h : i ^ 2 = -1) : ((1 - i) / (1 + i)) ^ 2 = -1 :=
by
  sorry

end value_of_complex_fraction_l306_306758


namespace mix_solutions_l306_306901

-- Definitions based on conditions
def solution_x_percentage : ℝ := 0.10
def solution_y_percentage : ℝ := 0.30
def volume_y : ℝ := 100
def desired_percentage : ℝ := 0.15

-- Problem statement rewrite with equivalent proof goal
theorem mix_solutions :
  ∃ Vx : ℝ, (Vx * solution_x_percentage + volume_y * solution_y_percentage) = (Vx + volume_y) * desired_percentage ∧ Vx = 300 :=
by
  sorry

end mix_solutions_l306_306901


namespace max_side_of_triangle_l306_306049

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306049


namespace hyperbola_equation_l306_306742

-- Definitions based on the conditions
def parabola_focus : (ℝ × ℝ) := (2, 0)
def point_on_hyperbola : (ℝ × ℝ) := (1, 0)
def hyperbola_center : (ℝ × ℝ) := (0, 0)
def right_focus_of_hyperbola : (ℝ × ℝ) := parabola_focus

-- Given the above definitions, we should prove that the standard equation of hyperbola C is correct
theorem hyperbola_equation :
  ∃ (a b : ℝ), (a = 1) ∧ (2^2 = a^2 + b^2) ∧
  (hyperbola_center = (0, 0)) ∧ (point_on_hyperbola = (1, 0)) →
  (x^2 - (y^2 / 3) = 1) :=
by
  sorry

end hyperbola_equation_l306_306742


namespace division_result_l306_306733

theorem division_result (x : ℝ) (h : (x - 2) / 13 = 4) : (x - 5) / 7 = 7 := by
  sorry

end division_result_l306_306733


namespace max_side_length_of_triangle_l306_306067

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306067


namespace max_product_of_two_integers_l306_306633

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end max_product_of_two_integers_l306_306633


namespace first_part_eq_19_l306_306504

theorem first_part_eq_19 (x y : ℕ) (h1 : x + y = 36) (h2 : 8 * x + 3 * y = 203) : x = 19 :=
by sorry

end first_part_eq_19_l306_306504


namespace sandwich_percentage_not_vegetables_l306_306666

noncomputable def percentage_not_vegetables (total_weight : ℝ) (vegetable_weight : ℝ) : ℝ :=
  (total_weight - vegetable_weight) / total_weight * 100

theorem sandwich_percentage_not_vegetables :
  percentage_not_vegetables 180 50 = 72.22 :=
by
  sorry

end sandwich_percentage_not_vegetables_l306_306666


namespace max_side_length_l306_306006

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306006


namespace g_inv_f_five_l306_306176

-- Declare the existence of functions f and g and their inverses
variables (f g : ℝ → ℝ)

-- Given condition from the problem
axiom inv_cond : ∀ x, f⁻¹ (g x) = 4 * x - 1

-- Define the specific problem to solve
theorem g_inv_f_five : g⁻¹ (f 5) = 3 / 2 :=
by
  sorry

end g_inv_f_five_l306_306176


namespace problem_1_problem_2_l306_306860

-- Definitions for sets A and B
def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 6
def B (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Problem (1): What is A ∩ B when m = 3
theorem problem_1 : ∀ (x : ℝ), A x → B x 3 → (-1 ≤ x ∧ x ≤ 4) := by
  intro x hA hB
  sorry

-- Problem (2): What is the range of m if A ⊆ B and m > 0
theorem problem_2 (m : ℝ) : m > 0 → (∀ x, A x → B x m) → (m ≥ 5) := by
  intros hm hAB
  sorry

end problem_1_problem_2_l306_306860


namespace smallest_part_proportional_division_l306_306146

theorem smallest_part_proportional_division (a b c d total : ℕ) (h : a + b + c + d = total) (sum_equals_360 : 360 = total * 15):
  min (4 * 15) (min (5 * 15) (min (7 * 15) (8 * 15))) = 60 :=
by
  -- Defining the proportions and overall total
  let a := 5
  let b := 7
  let c := 4
  let d := 8
  let total_parts := a + b + c + d

  -- Given that the division is proportional
  let part_value := 360 / total_parts

  -- Assert that the smallest part is equal to the smallest proportion times the value of one part
  let smallest_part := c * part_value
  trivial

end smallest_part_proportional_division_l306_306146


namespace complex_power_identity_l306_306264

-- Given condition
variable (z : Complex)
variable (h : z + 1/z = 2 * Real.cos (Real.pi / 36))

-- Proof Problem
theorem complex_power_identity :
  z ^ 1000 + 1 / (z ^ 1000) = -2 * Real.cos (Real.pi * 40 / 180) :=
sorry

end complex_power_identity_l306_306264


namespace smallest_sum_of_factors_of_8_l306_306315

theorem smallest_sum_of_factors_of_8! :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a * b * c * d = Nat.factorial 8 ∧ a + b + c + d = 102 :=
sorry

end smallest_sum_of_factors_of_8_l306_306315


namespace find_a2016_l306_306710

theorem find_a2016 (S : ℕ → ℕ)
  (a : ℕ → ℤ)
  (h₁ : S 1 = 6)
  (h₂ : S 2 = 4)
  (h₃ : ∀ n, S n > 0)
  (h₄ : ∀ n, (S (2 * n - 1))^2 = S (2 * n) * S (2 * n + 2))
  (h₅ : ∀ n, 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1))
  : a 2016 = -1009 := 
  sorry

end find_a2016_l306_306710


namespace find_unit_prices_and_evaluate_discount_schemes_l306_306679

theorem find_unit_prices_and_evaluate_discount_schemes :
  ∃ (x y : ℝ),
    40 * x + 100 * y = 280 ∧
    30 * x + 200 * y = 260 ∧
    x = 6 ∧
    y = 0.4 ∧
    (∀ m : ℝ, m > 200 → 
      (50 * 6 + 0.4 * (m - 50) < 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m < 450) ∧
      (50 * 6 + 0.4 * (m - 50) = 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m = 450) ∧
      (50 * 6 + 0.4 * (m - 50) > 50 * 6 + 0.4 * 200 + 0.4 * 0.8 * (m - 200) ↔ m > 450)) :=
sorry

end find_unit_prices_and_evaluate_discount_schemes_l306_306679


namespace value_of_a_plus_b_l306_306263

theorem value_of_a_plus_b :
  ∀ (a b x y : ℝ), x = 3 → y = -2 → 
  a * x + b * y = 2 → b * x + a * y = -3 → 
  a + b = -1 := 
by
  intros a b x y hx hy h1 h2
  subst hx
  subst hy
  sorry

end value_of_a_plus_b_l306_306263


namespace explicit_formula_inequality_solution_l306_306125

noncomputable def f (x : ℝ) : ℝ := (x : ℝ) / (x^2 + 1)

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x → y < b → x < y → f x < f y
def f_half_eq_two_fifths : Prop := f (1/2) = 2/5

-- Questions rewritten as goals
theorem explicit_formula :
  odd_function f ∧ increasing_on_interval f (-1) 1 ∧ f_half_eq_two_fifths →
  ∀ x, f x = x / (x^2 + 1) := by 
sorry

theorem inequality_solution :
  odd_function f ∧ increasing_on_interval f (-1) 1 →
  ∀ t, (f (t - 1) + f t < 0) ↔ (0 < t ∧ t < 1/2) := by 
sorry

end explicit_formula_inequality_solution_l306_306125


namespace part1_part2_part3_l306_306132

-- Part 1
theorem part1 (x : ℝ) (h : abs (x + 2) = abs (x - 4)) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h : abs (x + 2) + abs (x - 4) = 8) : x = -3 ∨ x = 5 :=
by
  sorry

-- Part 3
theorem part3 (t : ℝ) :
  let M := -2 - t
  let N := 4 - 3 * t
  (abs M = abs (M - N) → t = 1/2) ∧ 
  (N = 0 → t = 4/3) ∧
  (abs N = abs (N - M) → t = 2) ∧
  (M = N → t = 3) ∧
  (abs (M - N) = abs (2 * M) → t = 8) :=
by
  sorry

end part1_part2_part3_l306_306132


namespace areasEqualForHexagonAndOctagon_l306_306520

noncomputable def areaHexagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 6) -- Circumscribed radius
  let a := s / (2 * Real.tan (Real.pi / 6)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

noncomputable def areaOctagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 8) -- Circumscribed radius
  let a := s / (2 * Real.tan (3 * Real.pi / 8)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

theorem areasEqualForHexagonAndOctagon :
  let s := 3
  areaHexagon s = areaOctagon s := sorry

end areasEqualForHexagonAndOctagon_l306_306520


namespace units_digit_7_pow_6_pow_5_l306_306099

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 7 := by
  -- Proof will go here
  sorry

end units_digit_7_pow_6_pow_5_l306_306099


namespace AJ_stamps_l306_306830

theorem AJ_stamps (A : ℕ)
  (KJ := A / 2)
  (CJ := 2 * KJ + 5)
  (BJ := 3 * A - 3)
  (total_stamps := A + KJ + CJ + BJ)
  (h : total_stamps = 1472) :
  A = 267 :=
  sorry

end AJ_stamps_l306_306830


namespace train_length_proof_l306_306815

noncomputable def train_speed_kmh : ℝ := 50
noncomputable def crossing_time_s : ℝ := 9
noncomputable def length_of_train_m : ℝ := 125

theorem train_length_proof:
  ∀ (speed_kmh: ℝ) (time_s: ℝ), 
  speed_kmh = train_speed_kmh →
  time_s = crossing_time_s →
  (speed_kmh * (1000 / 3600) * time_s) = length_of_train_m :=
by intros speed_kmh time_s h_speed_kmh h_time_s
   -- Proof omitted
   sorry

end train_length_proof_l306_306815


namespace max_side_of_triangle_l306_306057

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306057


namespace perfect_square_form_l306_306245

theorem perfect_square_form (N : ℕ) (hN : 0 < N) : 
  ∃ x : ℤ, 2^N - 2 * (N : ℤ) = x^2 ↔ N = 1 ∨ N = 2 :=
by
  sorry

end perfect_square_form_l306_306245


namespace reading_schedule_l306_306926

-- Definitions of reading speeds and conditions
def total_pages := 910
def alice_speed := 30  -- seconds per page
def bob_speed := 60    -- seconds per page
def chandra_speed := 45  -- seconds per page

-- Mathematical problem statement
theorem reading_schedule :
  ∃ (x y : ℕ), 
    (x < y) ∧ 
    (y ≤ total_pages) ∧ 
    (30 * x = 45 * (y - x) ∧ 45 * (y - x) = 60 * (total_pages - y)) ∧ 
    x = 420 ∧ 
    y = 700 :=
  sorry

end reading_schedule_l306_306926


namespace quadratic_function_n_neg_l306_306531

theorem quadratic_function_n_neg (n : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + n = 0 → x > 0) → n < 0 :=
by
  sorry

end quadratic_function_n_neg_l306_306531


namespace regular_polygon_sides_l306_306665

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 160) : n = 18 :=
by
  sorry

end regular_polygon_sides_l306_306665


namespace angle_difference_l306_306278

-- Define the conditions
variables (A B : ℝ) 

def is_parallelogram := A + B = 180
def smaller_angle := A = 70
def larger_angle := B = 180 - 70

-- State the theorem to be proved
theorem angle_difference (A B : ℝ) (h1 : is_parallelogram A B) (h2 : smaller_angle A) : B - A = 40 := by
  sorry

end angle_difference_l306_306278


namespace geometric_sequence_common_ratio_l306_306735

theorem geometric_sequence_common_ratio :
  (∃ q : ℝ, 1 + q + q^2 = 13 ∧ (q = 3 ∨ q = -4)) :=
by
  sorry

end geometric_sequence_common_ratio_l306_306735


namespace find_a_l306_306536

theorem find_a (a : ℤ) (A : Set ℤ) (B : Set ℤ) :
  A = {-2, 3 * a - 1, a^2 - 3} ∧
  B = {a - 2, a - 1, a + 1} ∧
  A ∩ B = {-2} → a = -3 :=
by
  intro H
  sorry

end find_a_l306_306536


namespace inequality_correct_l306_306704

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end inequality_correct_l306_306704


namespace even_func_monotonic_on_negative_interval_l306_306149

variable {α : Type*} [LinearOrderedField α]
variable {f : α → α}

theorem even_func_monotonic_on_negative_interval 
  (h_even : ∀ x : α, f (-x) = f x)
  (h_mon_incr : ∀ x y : α, x < y → (x < 0 ∧ y ≤ 0) → f x < f y) :
  f 2 < f (-3 / 2) :=
sorry

end even_func_monotonic_on_negative_interval_l306_306149


namespace N_square_solutions_l306_306243

theorem N_square_solutions :
  ∀ N : ℕ, (N > 0 → ∃ k : ℕ, 2^N - 2 * N = k^2) → (N = 1 ∨ N = 2) :=
by
  sorry

end N_square_solutions_l306_306243


namespace license_plate_increase_l306_306736

theorem license_plate_increase :
  let old_license_plates := 26^2 * 10^3
  let new_license_plates := 26^2 * 10^4
  new_license_plates / old_license_plates = 10 :=
by
  sorry

end license_plate_increase_l306_306736


namespace time_for_P_to_finish_job_alone_l306_306467

variable (T : ℝ)

theorem time_for_P_to_finish_job_alone (h1 : 0 < T) (h2 : 3 * (1 / T + 1 / 20) + 0.4 * (1 / T) = 1) : T = 4 :=
by
  sorry

end time_for_P_to_finish_job_alone_l306_306467


namespace part1_part2_l306_306267
open Real

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (hm : m > 1) : ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by
  sorry

end part1_part2_l306_306267


namespace least_integer_to_add_l306_306336

theorem least_integer_to_add (n : ℕ) (h : n = 725) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 :=
by
  use 5
  split
  · exact Nat.lt_succ_self 4
  · rw [h]
    norm_num
    sorry

end least_integer_to_add_l306_306336


namespace Jack_total_money_in_dollars_l306_306753

variable (Jack_dollars : ℕ) (Jack_euros : ℕ) (euro_to_dollar : ℕ)

noncomputable def total_dollars (Jack_dollars : ℕ) (Jack_euros : ℕ) (euro_to_dollar : ℕ) : ℕ :=
  Jack_dollars + Jack_euros * euro_to_dollar

theorem Jack_total_money_in_dollars : 
  Jack_dollars = 45 → 
  Jack_euros = 36 → 
  euro_to_dollar = 2 → 
  total_dollars 45 36 2 = 117 :=
by
  intro h1 h2 h3
  unfold total_dollars
  rw [h1, h2, h3]
  -- skipping the actual proof
  sorry

end Jack_total_money_in_dollars_l306_306753


namespace max_triangle_side_24_l306_306083

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306083


namespace cricket_runs_l306_306998

variable (A B C D E : ℕ)

theorem cricket_runs
  (h1 : (A + B + C + D + E) = 180)
  (h2 : D = E + 5)
  (h3 : A = E + 8)
  (h4 : B = D + E)
  (h5 : B + C = 107) :
  E = 20 := by
  sorry

end cricket_runs_l306_306998


namespace sin_alpha_through_point_l306_306856

theorem sin_alpha_through_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (-3, -Real.sqrt 3)) :
    Real.sin α = -1 / 2 :=
by
  sorry

end sin_alpha_through_point_l306_306856


namespace Kim_morning_routine_time_l306_306879

def total_employees : ℕ := 9
def senior_employees : ℕ := 3
def overtime_employees : ℕ := 4
def regular_employees : ℕ := total_employees - senior_employees
def non_overtime_employees : ℕ := total_employees - overtime_employees

def coffee_time : ℕ := 5
def status_update_time (regular senior : ℕ) : ℕ := (regular * 2) + (senior * 3)
def payroll_update_time (overtime non_overtime : ℕ) : ℕ := (overtime * 3) + (non_overtime * 1)
def email_time : ℕ := 10
def task_allocation_time : ℕ := 7

def total_morning_routine_time : ℕ :=
  coffee_time +
  status_update_time regular_employees senior_employees +
  payroll_update_time overtime_employees non_overtime_employees +
  email_time +
  task_allocation_time

theorem Kim_morning_routine_time : total_morning_routine_time = 60 := by
  sorry

end Kim_morning_routine_time_l306_306879


namespace sufficient_drivers_and_ivan_petrovich_departure_l306_306603

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l306_306603


namespace change_received_correct_l306_306888

-- Define the prices of items and the amount paid
def price_hamburger : ℕ := 4
def price_onion_rings : ℕ := 2
def price_smoothie : ℕ := 3
def amount_paid : ℕ := 20

-- Define the total cost and the change received
def total_cost : ℕ := price_hamburger + price_onion_rings + price_smoothie
def change_received : ℕ := amount_paid - total_cost

-- Theorem stating the change received
theorem change_received_correct : change_received = 11 := by
  sorry

end change_received_correct_l306_306888


namespace solution_set_of_inequality_l306_306185

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} := sorry

end solution_set_of_inequality_l306_306185


namespace measure_six_liters_l306_306805

-- Given conditions as constants
def container_capacity : ℕ := 40
def ten_liter_bucket_capacity : ℕ := 10
def nine_liter_jug_capacity : ℕ := 9
def five_liter_jug_capacity : ℕ := 5

-- Goal: Measure out exactly 6 liters of milk using the above containers
theorem measure_six_liters (container : ℕ) (ten_bucket : ℕ) (nine_jug : ℕ) (five_jug : ℕ) :
  container = 40 →
  ten_bucket ≤ 10 →
  nine_jug ≤ 9 →
  five_jug ≤ 5 →
  ∃ (sequence_of_steps : ℕ → ℕ) (final_ten_bucket : ℕ),
    final_ten_bucket = 6 ∧ final_ten_bucket ≤ ten_bucket :=
by
  intro hcontainer hten_bucket hnine_jug hfive_jug
  sorry

end measure_six_liters_l306_306805


namespace max_side_of_triangle_l306_306046

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306046


namespace lotus_leaves_not_odd_l306_306345

theorem lotus_leaves_not_odd (n : ℕ) (h1 : n > 1) (h2 : ∀ t : ℕ, ∃ r : ℕ, 0 ≤ r ∧ r < n ∧ (t * (t + 1) / 2 - 1) % n = r) : ¬ Odd n :=
sorry

end lotus_leaves_not_odd_l306_306345


namespace car_rental_daily_rate_l306_306660

theorem car_rental_daily_rate (x : ℝ) : 
  (x + 0.18 * 48 = 18.95 + 0.16 * 48) -> 
  x = 17.99 :=
by 
  sorry

end car_rental_daily_rate_l306_306660


namespace cyclist_first_part_distance_l306_306575

theorem cyclist_first_part_distance
  (T₁ T₂ T₃ : ℝ)
  (D : ℝ)
  (h1 : D = 9 * T₁)
  (h2 : T₂ = 12 / 10)
  (h3 : T₃ = (D + 12) / 7.5)
  (h4 : T₁ + T₂ + T₃ = 7.2) : D = 18 := by
  sorry

end cyclist_first_part_distance_l306_306575


namespace simplify_expression_l306_306770

theorem simplify_expression : (225 / 10125) * 45 = 1 := by
  sorry

end simplify_expression_l306_306770


namespace no_such_function_exists_l306_306650

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = f (n + 1) - f n :=
by
  sorry

end no_such_function_exists_l306_306650


namespace part1_proof_part2_proof_part3_proof_l306_306280

-- Definitions and conditions for part 1
def P (a : ℤ) : ℤ × ℤ := (-3 * a - 4, 2 + a)
def part1_condition (a : ℤ) : Prop := (2 + a = 0)
def part1_answer : ℤ × ℤ := (2, 0)

-- Definitions and conditions for part 2
def Q : ℤ × ℤ := (5, 8)
def part2_condition (a : ℤ) : Prop := (-3 * a - 4 = 5)
def part2_answer : ℤ × ℤ := (5, -1)

-- Definitions and conditions for part 3
def part3_condition (a : ℤ) : Prop := 
  (-3 * a - 4 + 2 + a = 0) ∧ (-3 * a - 4 < 0 ∧ 2 + a > 0) -- Second quadrant
def part3_answer (a : ℤ) : ℤ := (a ^ 2023 + 2023)

-- Lean statements for proofs

theorem part1_proof (a : ℤ) (h : part1_condition a) : P a = part1_answer :=
by sorry

theorem part2_proof (a : ℤ) (h : part2_condition a) : P a = part2_answer :=
by sorry

theorem part3_proof (a : ℤ) (h : part3_condition a) : part3_answer a = 2022 :=
by sorry

end part1_proof_part2_proof_part3_proof_l306_306280


namespace yuko_in_front_of_yuri_l306_306645

theorem yuko_in_front_of_yuri (X : ℕ) (hYuri : 2 + 4 + 5 = 11) (hYuko : 1 + 5 + X > 11) : X = 6 := 
by
  sorry

end yuko_in_front_of_yuri_l306_306645


namespace largest_odd_integer_satisfying_inequality_l306_306841

theorem largest_odd_integer_satisfying_inequality : 
  ∃ (x : ℤ), (x % 2 = 1) ∧ (1 / 4 < x / 6) ∧ (x / 6 < 7 / 9) ∧ (∀ y : ℤ, (y % 2 = 1) ∧ (1 / 4 < y / 6) ∧ (y / 6 < 7 / 9) → y ≤ x) :=
sorry

end largest_odd_integer_satisfying_inequality_l306_306841


namespace stella_toilet_paper_packs_l306_306300

-- Define the relevant constants/conditions
def rolls_per_bathroom_per_day : Nat := 1
def number_of_bathrooms : Nat := 6
def days_per_week : Nat := 7
def weeks : Nat := 4
def rolls_per_pack : Nat := 12

-- Theorem statement
theorem stella_toilet_paper_packs :
  (rolls_per_bathroom_per_day * number_of_bathrooms * days_per_week * weeks) / rolls_per_pack = 14 :=
by
  sorry

end stella_toilet_paper_packs_l306_306300


namespace max_side_of_triangle_l306_306044

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306044


namespace when_did_B_join_l306_306350

theorem when_did_B_join
  (A_investment : ℝ := 27000)
  (B_investment : ℝ := 36000)
  (profit_ratio : ℝ := 2 / 1)
  (A_time : ℝ := 12)
  (B_time : ℝ)
  (profit_A : ℝ)
  (profit_B : ℝ) :
  (profit_A / profit_B = profit_ratio) → 
  (A_investment * A_time / (B_investment * B_time) = profit_ratio) →
  B_time = (12 - 7.5) :=
by
  intro hp ha
  sorry

end when_did_B_join_l306_306350


namespace wrongly_entered_mark_l306_306812

theorem wrongly_entered_mark (x : ℕ) 
    (h1 : x - 33 = 52) : x = 85 :=
by
  sorry

end wrongly_entered_mark_l306_306812


namespace range_of_a_l306_306412

theorem range_of_a (a : ℝ) : 
  {x : ℝ | x^2 - 4 * x + 3 < 0} ⊆ {x : ℝ | 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0} → 
  -4 ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l306_306412


namespace initial_red_marbles_l306_306808

theorem initial_red_marbles (R : ℕ) (blue_marbles_initial : ℕ) (red_marbles_removed : ℕ) :
  blue_marbles_initial = 30 →
  red_marbles_removed = 3 →
  (R - red_marbles_removed) + (blue_marbles_initial - 4 * red_marbles_removed) = 35 →
  R = 20 :=
by
  intros h_blue h_red h_total
  sorry

end initial_red_marbles_l306_306808


namespace b_must_be_one_l306_306323

theorem b_must_be_one (a b : ℝ) (h1 : a + b - a * b = 1) (h2 : ∀ n : ℤ, a ≠ n) : b = 1 :=
sorry

end b_must_be_one_l306_306323


namespace not_sit_next_probability_l306_306951

theorem not_sit_next_probability (n : ℕ) (h : n = 9) :
  let total_ways := Nat.choose 9 2,
  let adjacent_pairs := 8,
  let probability_adjacent := (adjacent_pairs : ℚ) / total_ways,
  let probability_not_adjacent := 1 - probability_adjacent
  in probability_not_adjacent = 7 / 9 := 
by
  sorry

end not_sit_next_probability_l306_306951


namespace trigonometric_identity_l306_306696

theorem trigonometric_identity :
  (2 * real.cos (real.pi / 18) - real.sin (real.pi / 9)) / real.sin (7 * real.pi / 18) = real.sqrt 3 :=
by
  -- Proof goes here
  sorry

end trigonometric_identity_l306_306696


namespace principle_calculation_l306_306201

noncomputable def calculate_principal (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  A / (1 + (R * T))

theorem principle_calculation :
  calculate_principal 1456 0.05 2.4 = 1300 :=
by
  sorry

end principle_calculation_l306_306201


namespace distance_A_to_focus_l306_306534

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  ((b^2 - 4*a*c) / (4*a), 0)

theorem distance_A_to_focus 
  (P : ℝ × ℝ) (parabola : ℝ → ℝ → Prop)
  (A B : ℝ × ℝ)
  (hP : P = (-2, 0))
  (hPar : ∀ x y, parabola x y ↔ y^2 = 4 * x)
  (hLine : ∃ m b, ∀ x y, y = m * x + b ∧ y^2 = 4 * x → (x, y) = A ∨ (x, y) = B)
  (hDist : dist P A = (1 / 2) * dist A B)
  (hFocus : focus_of_parabola 1 0 (-1) = (1, 0)) :
  dist A (1, 0) = 5 / 3 :=
sorry

end distance_A_to_focus_l306_306534


namespace transformed_variance_l306_306614

variables {n : ℕ} {a : ℕ → ℝ}
variable (var_a : variance (list.of_fn a) = 1)

theorem transformed_variance : variance (list.of_fn (λ i, 2 * a i - 1)) = 4 :=
by sorry

end transformed_variance_l306_306614


namespace base_of_524_l306_306356

theorem base_of_524 : 
  ∀ (b : ℕ), (5 * b^2 + 2 * b + 4 = 340) → b = 8 :=
by
  intros b h
  sorry

end base_of_524_l306_306356


namespace least_pos_int_for_multiple_of_5_l306_306331

theorem least_pos_int_for_multiple_of_5 (n : ℕ) (h1 : n = 725) : ∃ x : ℕ, x > 0 ∧ (725 + x) % 5 = 0 ∧ x = 5 :=
by
  sorry

end least_pos_int_for_multiple_of_5_l306_306331


namespace arithmetic_geometric_mean_inequality_l306_306305

open BigOperators

noncomputable def A (a : Fin n → ℝ) : ℝ := (Finset.univ.sum a) / n

noncomputable def G (a : Fin n → ℝ) : ℝ := (Finset.univ.prod a) ^ (1 / n)

theorem arithmetic_geometric_mean_inequality (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) : A a ≥ G a :=
  sorry

end arithmetic_geometric_mean_inequality_l306_306305


namespace max_side_length_is_11_l306_306020

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306020


namespace white_pairs_coincide_l306_306688

def triangles_in_each_half (red blue white: Nat) : Prop :=
  red = 5 ∧ blue = 6 ∧ white = 9

def folding_over_centerline (r_pairs b_pairs rw_pairs bw_pairs: Nat) : Prop :=
  r_pairs = 3 ∧ b_pairs = 2 ∧ rw_pairs = 3 ∧ bw_pairs = 1

theorem white_pairs_coincide
    (red_triangles blue_triangles white_triangles : Nat)
    (r_pairs b_pairs rw_pairs bw_pairs : Nat) :
    triangles_in_each_half red_triangles blue_triangles white_triangles →
    folding_over_centerline r_pairs b_pairs rw_pairs bw_pairs →
    ∃ coinciding_white_pairs, coinciding_white_pairs = 5 :=
by
  intros half_cond fold_cond
  sorry

end white_pairs_coincide_l306_306688


namespace max_side_of_triangle_l306_306050

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306050


namespace complement_of_angle_is_acute_l306_306823

theorem complement_of_angle_is_acute (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < 90) : 0 < 90 - θ ∧ 90 - θ < 90 :=
by sorry

end complement_of_angle_is_acute_l306_306823


namespace arc_length_of_sector_l306_306905

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h : r = Real.pi ∧ θ = 120) : 
  r * θ / 180 * Real.pi = 2 * Real.pi * Real.pi / 3 :=
by
  sorry

end arc_length_of_sector_l306_306905


namespace possible_galina_numbers_l306_306533

def is_divisible_by (m n : ℕ) : Prop := n % m = 0

def conditions_for_galina_number (n : ℕ) : Prop :=
  let C1 := is_divisible_by 7 n
  let C2 := is_divisible_by 11 n
  let C3 := n < 13
  let C4 := is_divisible_by 77 n
  (C1 ∧ ¬C2 ∧ C3 ∧ ¬C4) ∨ (¬C1 ∧ C2 ∧ C3 ∧ ¬C4)

theorem possible_galina_numbers (n : ℕ) :
  conditions_for_galina_number n ↔ (n = 7 ∨ n = 11) :=
by
  -- Proof to be filled in
  sorry

end possible_galina_numbers_l306_306533


namespace solve_for_x_l306_306748

theorem solve_for_x (x : ℝ) (h : (1 / 4) + (5 / x) = (12 / x) + (1 / 15)) : x = 420 / 11 := 
by
  sorry

end solve_for_x_l306_306748


namespace equation_solutions_l306_306136

noncomputable def count_solutions (a : ℝ) : ℕ :=
  if 0 < a ∧ a <= 1 ∨ a = Real.exp (1 / Real.exp 1) then 1
  else if 1 < a ∧ a < Real.exp (1 / Real.exp 1) then 2
  else if a > Real.exp (1 / Real.exp 1) then 0
  else 0

theorem equation_solutions (a : ℝ) (h₀ : 0 < a) :
  (∃! x : ℝ, a^x = x) ↔ count_solutions a = 1 ∨ count_solutions a = 2 ∨ count_solutions a = 0 := sorry

end equation_solutions_l306_306136


namespace total_right_handed_players_is_60_l306_306766

def total_players : ℕ := 70
def throwers : ℕ := 40
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def right_handed_throwers : ℕ := throwers
def total_right_handed_players : ℕ := right_handed_throwers + right_handed_non_throwers

theorem total_right_handed_players_is_60 : total_right_handed_players = 60 := by
  sorry

end total_right_handed_players_is_60_l306_306766


namespace find_sequence_l306_306547

def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → 
    a (n + 1) = (a n * a (n - 1)) / 
               Real.sqrt (a n^2 + a (n - 1)^2 + 1)

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 5

def sequence_property (F : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = Real.sqrt (1 / (Real.exp (F n * Real.log 10) - 1))

theorem find_sequence (a : ℕ → ℝ) (F : ℕ → ℝ) :
  initial_conditions a →
  recurrence_relation a →
  (∀ n : ℕ, n ≥ 2 →
    F (n + 1) = F n + F (n - 1)) →
  sequence_property F a :=
by
  intros h_initial h_recur h_F
  sorry

end find_sequence_l306_306547


namespace max_product_of_two_integers_l306_306632

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end max_product_of_two_integers_l306_306632


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306030

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306030


namespace polynomial_value_l306_306882

noncomputable def p (x : ℝ) : ℝ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) + 24 * x

theorem polynomial_value :
  (p 1 = 24) ∧ (p 2 = 48) ∧ (p 3 = 72) ∧ (p 4 = 96) →
  p 0 + p 5 = 168 := 
by
  sorry

end polynomial_value_l306_306882


namespace find_article_cost_l306_306200

noncomputable def original_cost_price (C S : ℝ) :=
  (S = 1.25 * C) ∧
  (S - 6.30 = 1.04 * C)

theorem find_article_cost (C S : ℝ) (h : original_cost_price C S) : C = 30 :=
by sorry

end find_article_cost_l306_306200


namespace five_points_distance_ratio_ge_two_sin_54_l306_306118

theorem five_points_distance_ratio_ge_two_sin_54
  (points : Fin 5 → ℝ × ℝ)
  (distinct : Function.Injective points) :
  let distances := {d : ℝ | ∃ (i j : Fin 5), i ≠ j ∧ d = dist (points i) (points j)}
  ∃ (max_dist min_dist : ℝ), max_dist ∈ distances ∧ min_dist ∈ distances ∧ max_dist / min_dist ≥ 2 * Real.sin (54 * Real.pi / 180) := by
  sorry

end five_points_distance_ratio_ge_two_sin_54_l306_306118


namespace least_pos_int_for_multiple_of_5_l306_306332

theorem least_pos_int_for_multiple_of_5 (n : ℕ) (h1 : n = 725) : ∃ x : ℕ, x > 0 ∧ (725 + x) % 5 = 0 ∧ x = 5 :=
by
  sorry

end least_pos_int_for_multiple_of_5_l306_306332


namespace sphere_in_cone_volume_l306_306959

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem sphere_in_cone_volume :
  let d := 12
  let θ := 45
  let r := 3 * Real.sqrt 2
  let V := volume_of_sphere r
  d = 12 → θ = 45 → V = 72 * Real.sqrt 2 * Real.pi := by
  intros h1 h2
  sorry

end sphere_in_cone_volume_l306_306959


namespace product_11_29_product_leq_20_squared_product_leq_half_m_squared_l306_306765

-- Definition of natural numbers
variable (a b m : ℕ)

-- Statement 1: Prove that 11 × 29 = 20^2 - 9^2
theorem product_11_29 : 11 * 29 = 20^2 - 9^2 := sorry

-- Statement 2: Prove ∀ a, b ∈ ℕ, if a + b = 40, then ab ≤ 20^2.
theorem product_leq_20_squared (a b : ℕ) (h : a + b = 40) : a * b ≤ 20^2 := sorry

-- Statement 3: Prove ∀ a, b ∈ ℕ, if a + b = m, then ab ≤ (m/2)^2.
theorem product_leq_half_m_squared (a b : ℕ) (m : ℕ) (h : a + b = m) : a * b ≤ (m / 2)^2 := sorry

end product_11_29_product_leq_20_squared_product_leq_half_m_squared_l306_306765


namespace hypotenuse_length_l306_306318

theorem hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = Real.sqrt 5) (h₂ : b = Real.sqrt 12) : c = Real.sqrt 17 :=
by
  -- Proof not required, hence skipped with 'sorry'
  sorry

end hypotenuse_length_l306_306318


namespace max_shirt_price_l306_306820

theorem max_shirt_price (total_budget : ℝ) (entrance_fee : ℝ) (num_shirts : ℝ) 
  (discount_rate : ℝ) (tax_rate : ℝ) (max_price : ℝ) 
  (budget_after_fee : total_budget - entrance_fee = 195)
  (shirt_discount : num_shirts > 15 → discounted_price = num_shirts * max_price * (1 - discount_rate))
  (price_with_tax : discounted_price * (1 + tax_rate) ≤ 195) : 
  max_price ≤ 10 := 
sorry

end max_shirt_price_l306_306820


namespace capture_probability_correct_l306_306469

structure ProblemConditions where
  rachel_speed : ℕ -- seconds per lap
  robert_speed : ℕ -- seconds per lap
  rachel_direction : Bool -- true if counterclockwise, false if clockwise
  robert_direction : Bool -- true if counterclockwise, false if clockwise
  start_time : ℕ -- 0 seconds
  end_time_start : ℕ -- 900 seconds
  end_time_end : ℕ -- 1200 seconds
  photo_coverage_fraction : ℚ -- fraction of the track covered by the photo

noncomputable def probability_capture_in_photo (pc : ProblemConditions) : ℚ :=
  sorry -- define and prove the exact probability

-- Given the conditions in the problem
def problem_instance : ProblemConditions :=
{
  rachel_speed := 120,
  robert_speed := 100,
  rachel_direction := true,
  robert_direction := false,
  start_time := 0,
  end_time_start := 900,
  end_time_end := 1200,
  photo_coverage_fraction := 1/3
}

-- The theorem statement we are asked to prove
theorem capture_probability_correct :
  probability_capture_in_photo problem_instance = 1/9 :=
sorry

end capture_probability_correct_l306_306469


namespace max_a_for_no_lattice_point_l306_306354

theorem max_a_for_no_lattice_point (a : ℝ) (hm : ∀ m : ℝ, 1 / 2 < m ∧ m < a → ¬ ∃ x y : ℤ, 0 < x ∧ x ≤ 200 ∧ y = m * x + 3) : 
  a = 101 / 201 :=
sorry

end max_a_for_no_lattice_point_l306_306354


namespace largest_unique_k_l306_306195

theorem largest_unique_k (n : ℕ) :
  (∀ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13 → False) ∧
  (∃ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13) → n = 112 :=
by sorry

end largest_unique_k_l306_306195


namespace three_correct_deliveries_probability_l306_306405

theorem three_correct_deliveries_probability (n : ℕ) (h1 : n = 5) :
  (∃ p : ℚ, p = 1/6 ∧ 
   (∃ choose3 : ℕ, choose3 = Nat.choose n 3) ∧ 
   (choose3 * 1/5 * 1/4 * 1/3 = p)) :=
by 
  sorry

end three_correct_deliveries_probability_l306_306405


namespace total_pages_in_book_l306_306929

theorem total_pages_in_book : 
  ∀ (n : ℕ), (∑ i in finset.range n, if i < 10 then 1 else if i < 100 then 2 else 3) = 990 → n = 366 := by
  intro n h
  sorry

end total_pages_in_book_l306_306929


namespace airplane_seats_l306_306511

theorem airplane_seats (s : ℝ)
  (h1 : 0.30 * s = 0.30 * s)
  (h2 : (3 / 5) * s = (3 / 5) * s)
  (h3 : 36 + 0.30 * s + (3 / 5) * s = s) : s = 360 :=
by
  sorry

end airplane_seats_l306_306511


namespace infinite_solutions_iff_a_eq_neg12_l306_306255

theorem infinite_solutions_iff_a_eq_neg12 {a : ℝ} : 
  (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + 16)) ↔ a = -12 :=
by 
  sorry

end infinite_solutions_iff_a_eq_neg12_l306_306255


namespace minimum_n_of_colored_balls_l306_306564

theorem minimum_n_of_colored_balls (n : ℕ) (h1 : n ≥ 3)
  (h2 : (n * (n + 1)) / 2 % 10 = 0) : n = 24 :=
sorry

end minimum_n_of_colored_balls_l306_306564


namespace inequality_sum_l306_306548

theorem inequality_sum 
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2)
  (h2 : a2 ≥ a3)
  (h3 : a3 > 0)
  (h4 : b1 ≥ b2)
  (h5 : b2 ≥ b3)
  (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) :
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := 
sorry

end inequality_sum_l306_306548


namespace max_side_length_l306_306008

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306008


namespace largest_good_number_smallest_bad_number_l306_306254

def is_good_number (M : ℕ) : Prop :=
  ∃ a b c d : ℤ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

theorem largest_good_number :
  ∀ M : ℕ, is_good_number M ↔ M ≤ 576 :=
by sorry

theorem smallest_bad_number :
  ∀ M : ℕ, ¬ is_good_number M ↔ M ≥ 443 :=
by sorry

end largest_good_number_smallest_bad_number_l306_306254


namespace weaving_increase_is_sixteen_over_twentynine_l306_306774

-- Conditions for the problem as definitions
def first_day_weaving := 5
def total_days := 30
def total_weaving := 390

-- The arithmetic series sum formula for 30 days
def sum_arithmetic_series (a d : ℚ) (n : ℕ) := n * a + (n * (n-1) / 2) * d

-- The question is to prove the increase in chi per day is 16/29
theorem weaving_increase_is_sixteen_over_twentynine
  (d : ℚ)
  (h : sum_arithmetic_series first_day_weaving d total_days = total_weaving) :
  d = 16 / 29 :=
sorry

end weaving_increase_is_sixteen_over_twentynine_l306_306774


namespace baker_cakes_remaining_l306_306232

def InitialCakes : ℕ := 48
def SoldCakes : ℕ := 44
def RemainingCakes (initial sold : ℕ) : ℕ := initial - sold

theorem baker_cakes_remaining : RemainingCakes InitialCakes SoldCakes = 4 := 
by {
  -- placeholder for the proof
  sorry
}

end baker_cakes_remaining_l306_306232


namespace greatest_product_l306_306636

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end greatest_product_l306_306636


namespace average_greater_than_median_by_22_l306_306722

/-- Define the weights of the siblings -/
def hammie_weight : ℕ := 120
def triplet1_weight : ℕ := 4
def triplet2_weight : ℕ := 4
def triplet3_weight : ℕ := 7
def brother_weight : ℕ := 10

/-- Define the list of weights -/
def weights : List ℕ := [hammie_weight, triplet1_weight, triplet2_weight, triplet3_weight, brother_weight]

/-- Define the median and average weight -/
def median_weight : ℕ := 7
def average_weight : ℕ := 29

theorem average_greater_than_median_by_22 : average_weight - median_weight = 22 := by
  sorry

end average_greater_than_median_by_22_l306_306722


namespace complement_A_B_eq_singleton_three_l306_306852

open Set

variable (A : Set ℕ) (B : Set ℕ) (a : ℕ)

theorem complement_A_B_eq_singleton_three (hA : A = {2, 3, 4})
    (hB : B = {a + 2, a}) (h_inter : A ∩ B = B) : A \ B = {3} :=
  sorry

end complement_A_B_eq_singleton_three_l306_306852


namespace max_side_of_triangle_l306_306039

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306039


namespace winner_more_than_third_l306_306827

theorem winner_more_than_third (W S T F : ℕ) (h1 : F = 199) 
(h2 : W = F + 105) (h3 : W = S + 53) (h4 : W + S + T + F = 979) : 
W - T = 79 :=
by
  -- Here, the proof steps would go, but they are not required as per instructions.
  sorry

end winner_more_than_third_l306_306827


namespace janets_total_pockets_l306_306457

-- Define the total number of dresses
def totalDresses : ℕ := 36

-- Define the dresses with pockets
def dressesWithPockets : ℕ := totalDresses / 2

-- Define the dresses without pockets
def dressesWithoutPockets : ℕ := totalDresses - dressesWithPockets

-- Define the dresses with one hidden pocket
def dressesWithOneHiddenPocket : ℕ := (40 * dressesWithoutPockets) / 100

-- Define the dresses with 2 pockets
def dressesWithTwoPockets : ℕ := dressesWithPockets / 3

-- Define the dresses with 3 pockets
def dressesWithThreePockets : ℕ := dressesWithPockets / 4

-- Define the dresses with 4 pockets
def dressesWithFourPockets : ℕ := dressesWithPockets - dressesWithTwoPockets - dressesWithThreePockets

-- Calculate the total number of pockets
def totalPockets : ℕ := 
  2 * dressesWithTwoPockets + 
  3 * dressesWithThreePockets + 
  4 * dressesWithFourPockets + 
  dressesWithOneHiddenPocket

-- The theorem to prove the total number of pockets
theorem janets_total_pockets : totalPockets = 63 :=
  by
    -- Proof is omitted, use 'sorry'
    sorry

end janets_total_pockets_l306_306457


namespace middle_marble_radius_l306_306404

theorem middle_marble_radius (r_1 r_5 : ℝ) (h1 : r_1 = 8) (h5 : r_5 = 18) : 
  ∃ r_3 : ℝ, r_3 = 12 :=
by
  let r_3 := Real.sqrt (r_1 * r_5)
  have h : r_3 = 12 := sorry
  exact ⟨r_3, h⟩

end middle_marble_radius_l306_306404


namespace product_of_numbers_l306_306909

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 157) : x * y = 22 := 
by 
  sorry

end product_of_numbers_l306_306909


namespace drivers_sufficiency_and_ivan_petrovich_departure_l306_306596

def one_way_trip_mins := 2 * 60 + 40
def round_trip_mins := 2 * one_way_trip_mins
def rest_time_mins := 60

axiom driver_A_return_time : 12 * 60 + 40
axiom driver_A_next_trip_time := driver_A_return_time + rest_time_mins
axiom driver_D_departure_time : 13 * 60 + 5
axiom continuous_trips : True 

noncomputable def sufficient_drivers : nat := 4
noncomputable def ivan_petrovich_departure : nat := 10 * 60 + 40

theorem drivers_sufficiency_and_ivan_petrovich_departure :
  (sufficient_drivers = 4) ∧ (ivan_petrovich_departure = 10 * 60 + 40) := by
  sorry

end drivers_sufficiency_and_ivan_petrovich_departure_l306_306596


namespace num_handshakes_l306_306670

-- Definition of the conditions
def num_teams : Nat := 4
def women_per_team : Nat := 2
def total_women : Nat := num_teams * women_per_team
def handshakes_per_woman : Nat := total_women -1 - women_per_team

-- Statement of the problem to prove
theorem num_handshakes (h: total_women = 8) : (total_women * handshakes_per_woman) / 2 = 24 := 
by
  -- Proof goes here
  sorry

end num_handshakes_l306_306670


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306036

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306036


namespace find_x_l306_306490

theorem find_x 
  (x : ℝ)
  (h : 0.4 * x + (0.6 * 0.8) = 0.56) : 
  x = 0.2 := sorry

end find_x_l306_306490


namespace problem_l306_306728

theorem problem (x y z : ℕ) (hx : x < 9) (hy : y < 9) (hz : z < 9) 
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z % 9 = 0) :=
sorry

end problem_l306_306728


namespace amount_after_two_years_l306_306971

def present_value : ℝ := 62000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

theorem amount_after_two_years:
  let amount_after_n_years (pv : ℝ) (r : ℝ) (n : ℕ) := pv * (1 + r)^n
  amount_after_n_years present_value rate_of_increase time_period = 78468.75 := 
  by 
    -- This is where your proof would go
    sorry

end amount_after_two_years_l306_306971


namespace max_side_length_is_11_l306_306018

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306018


namespace at_least_two_pass_written_test_expectation_number_of_admission_advantage_l306_306932

noncomputable def probability_of_passing_written_test_A : ℝ := 0.4
noncomputable def probability_of_passing_written_test_B : ℝ := 0.8
noncomputable def probability_of_passing_written_test_C : ℝ := 0.5

noncomputable def probability_of_passing_interview_A : ℝ := 0.8
noncomputable def probability_of_passing_interview_B : ℝ := 0.4
noncomputable def probability_of_passing_interview_C : ℝ := 0.64

theorem at_least_two_pass_written_test :
  (probability_of_passing_written_test_A * probability_of_passing_written_test_B * (1 - probability_of_passing_written_test_C) +
  probability_of_passing_written_test_A * (1 - probability_of_passing_written_test_B) * probability_of_passing_written_test_C +
  (1 - probability_of_passing_written_test_A) * probability_of_passing_written_test_B * probability_of_passing_written_test_C +
  probability_of_passing_written_test_A * probability_of_passing_written_test_B * probability_of_passing_written_test_C = 0.6) :=
sorry

theorem expectation_number_of_admission_advantage :
  (3 * (probability_of_passing_written_test_A * probability_of_passing_interview_A) +
  3 * (probability_of_passing_written_test_B * probability_of_passing_interview_B) +
  3 * (probability_of_passing_written_test_C * probability_of_passing_interview_C) = 0.96) :=
sorry

end at_least_two_pass_written_test_expectation_number_of_admission_advantage_l306_306932


namespace a_plus_b_eq_zero_l306_306572

-- Define the universal set and the relevant sets
def U : Set ℝ := Set.univ
def M (a : ℝ) : Set ℝ := {x | x^2 + a * x ≤ 0}
def C_U_M (b : ℝ) : Set ℝ := {x | x > b ∨ x < 0}

-- Define the proof theorem
theorem a_plus_b_eq_zero (a b : ℝ) (h1 : ∀ x, x ∈ M a ↔ -a < x ∧ x < 0 ∨ 0 < x ∧ x < -a)
                         (h2 : ∀ x, x ∈ C_U_M b ↔ x > b ∨ x < 0) : a + b = 0 := 
sorry

end a_plus_b_eq_zero_l306_306572


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306029

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306029


namespace distance_MN_is_2R_l306_306997

-- Definitions for the problem conditions
variable (R : ℝ) (A B C M N : ℝ) (alpha : ℝ)
variable (AC AB : ℝ)

-- Assumptions based on the problem statement
axiom circle_radius (r : ℝ) : r = R
axiom chord_length_AC (ch_AC : ℝ) : ch_AC = AC
axiom chord_length_AB (ch_AB : ℝ) : ch_AB = AB
axiom distance_M_to_AC (d_M_AC : ℝ) : d_M_AC = AC
axiom distance_N_to_AB (d_N_AB : ℝ) : d_N_AB = AB
axiom angle_BAC (ang_BAC : ℝ) : ang_BAC = alpha

-- To prove: the distance between M and N is 2R
theorem distance_MN_is_2R : |MN| = 2 * R := sorry

end distance_MN_is_2R_l306_306997


namespace first_number_Harold_says_l306_306371

/-
  Define each student's sequence of numbers.
  - Alice skips every 4th number.
  - Barbara says numbers that Alice didn't say, skipping every 4th in her sequence.
  - Subsequent students follow the same rule.
  - Harold picks the smallest prime number not said by any student.
-/

def is_skipped_by_Alice (n : Nat) : Prop :=
  n % 4 ≠ 0

def is_skipped_by_Barbara (n : Nat) : Prop :=
  is_skipped_by_Alice n ∧ (n / 4) % 4 ≠ 3

def is_skipped_by_Candice (n : Nat) : Prop :=
  is_skipped_by_Barbara n ∧ (n / 16) % 4 ≠ 3

def is_skipped_by_Debbie (n : Nat) : Prop :=
  is_skipped_by_Candice n ∧ (n / 64) % 4 ≠ 3

def is_skipped_by_Eliza (n : Nat) : Prop :=
  is_skipped_by_Debbie n ∧ (n / 256) % 4 ≠ 3

def is_skipped_by_Fatima (n : Nat) : Prop :=
  is_skipped_by_Eliza n ∧ (n / 1024) % 4 ≠ 3

def is_skipped_by_Grace (n : Nat) : Prop :=
  is_skipped_by_Fatima n

def is_skipped_by_anyone (n : Nat) : Prop :=
  ¬ is_skipped_by_Alice n ∨ ¬ is_skipped_by_Barbara n ∨ ¬ is_skipped_by_Candice n ∨
  ¬ is_skipped_by_Debbie n ∨ ¬ is_skipped_by_Eliza n ∨ ¬ is_skipped_by_Fatima n ∨
  ¬ is_skipped_by_Grace n

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ (m : Nat), m ∣ n → m = 1 ∨ m = n

theorem first_number_Harold_says : ∃ n : Nat, is_prime n ∧ ¬ is_skipped_by_anyone n ∧ n = 11 := by
  sorry

end first_number_Harold_says_l306_306371


namespace max_product_l306_306631

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end max_product_l306_306631


namespace child_support_amount_l306_306166

-- Definitions
def base_salary_1_3 := 30000
def base_salary_4_7 := 36000
def bonus_1 := 2000
def bonus_2 := 3000
def bonus_3 := 4000
def bonus_4 := 5000
def bonus_5 := 6000
def bonus_6 := 7000
def bonus_7 := 8000
def child_support_1_5 := 30 / 100
def child_support_6_7 := 25 / 100
def paid_total := 1200

-- Total Income per year
def income_year_1 := base_salary_1_3 + bonus_1
def income_year_2 := base_salary_1_3 + bonus_2
def income_year_3 := base_salary_1_3 + bonus_3
def income_year_4 := base_salary_4_7 + bonus_4
def income_year_5 := base_salary_4_7 + bonus_5
def income_year_6 := base_salary_4_7 + bonus_6
def income_year_7 := base_salary_4_7 + bonus_7

-- Child Support per year
def support_year_1 := child_support_1_5 * income_year_1
def support_year_2 := child_support_1_5 * income_year_2
def support_year_3 := child_support_1_5 * income_year_3
def support_year_4 := child_support_1_5 * income_year_4
def support_year_5 := child_support_1_5 * income_year_5
def support_year_6 := child_support_6_7 * income_year_6
def support_year_7 := child_support_6_7 * income_year_7

-- Total Support calculation
def total_owed := support_year_1 + support_year_2 + support_year_3 + 
                  support_year_4 + support_year_5 +
                  support_year_6 + support_year_7

-- Final amount owed
def amount_owed := total_owed - paid_total

-- Theorem statement
theorem child_support_amount :
  amount_owed = 75150 :=
sorry

end child_support_amount_l306_306166


namespace other_root_of_quadratic_l306_306870

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, x^2 + a * x - 2 = 0 → x = -1) → ∃ m, x = m ∧ m = 2 :=
by
  sorry

end other_root_of_quadratic_l306_306870


namespace roots_cosines_of_triangle_l306_306183

-- Condition: polynomial p(x) has three positive real roots
variables {a b c : ℝ}

-- Definition of the polynomial
def p (x : ℝ) := x^3 + a*x^2 + b*x + c

theorem roots_cosines_of_triangle (h_pos_roots : ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ p x₁ = 0 ∧ p x₂ = 0 ∧ p x₃ = 0) :
  ∃ A B C : ℝ, 
    (A + B + C = π) ∧ 
    (a^2 - 2*b - 2*c = 1) :=
sorry

end roots_cosines_of_triangle_l306_306183


namespace labeling_edges_complete_graph_condition_l306_306283

open Nat

theorem labeling_edges_complete_graph_condition (n : ℕ) : 
  (∃ (f : Fin (binom n 2) → Fin (binom n 3) → ℕ), 
  ∀ (a b c : Fin (binom n 3)), 
  (gcd (f a) (f c)) ∣ f b) → n ≤ 3 :=
sorry

end labeling_edges_complete_graph_condition_l306_306283


namespace sum_of_solutions_eq_zero_l306_306197

theorem sum_of_solutions_eq_zero (x : ℝ) (h : 6 * x / 30 = 7 / x) :
  (∃ x₁ x₂ : ℝ, x₁^2 = 35 ∧ x₂^2 = 35 ∧ x₁ + x₂ = 0) :=
sorry

end sum_of_solutions_eq_zero_l306_306197


namespace max_side_of_triangle_l306_306054

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306054


namespace num_integers_satisfying_inequality_l306_306435

theorem num_integers_satisfying_inequality (n : ℤ) (h : n ≠ 0) : (1 / |(n:ℤ)| ≥ 1 / 5) → (number_of_satisfying_integers = 10) :=
by
  sorry

end num_integers_satisfying_inequality_l306_306435


namespace mean_of_set_l306_306483

theorem mean_of_set (n : ℤ) (h_median : n + 7 = 14) : (n + (n + 4) + (n + 7) + (n + 10) + (n + 14)) / 5 = 14 := by
  sorry

end mean_of_set_l306_306483


namespace wifes_raise_l306_306103

variable (D W : ℝ)
variable (h1 : 0.08 * D = 800)
variable (h2 : 1.08 * D - 1.08 * W = 540)

theorem wifes_raise : 0.08 * W = 760 :=
by
  sorry

end wifes_raise_l306_306103


namespace my_op_eq_l306_306992

-- Define the custom operation
def my_op (m n : ℝ) : ℝ := m * n * (m - n)

-- State the theorem
theorem my_op_eq :
  ∀ (a b : ℝ), my_op (a + b) a = a^2 * b + a * b^2 :=
by intros a b; sorry

end my_op_eq_l306_306992


namespace crushing_load_example_l306_306479

noncomputable def crushing_load (T H : ℝ) : ℝ :=
  (30 * T^5) / H^3

theorem crushing_load_example : crushing_load 5 10 = 93.75 := by
  sorry

end crushing_load_example_l306_306479


namespace longest_side_obtuse_triangle_l306_306873

theorem longest_side_obtuse_triangle (a b c : ℝ) (h₀ : a = 2) (h₁ : b = 4) 
  (h₂ : a^2 + b^2 < c^2) : 
  2 * Real.sqrt 5 < c ∧ c < 6 :=
by 
  sorry

end longest_side_obtuse_triangle_l306_306873


namespace max_triangle_side_24_l306_306081

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306081


namespace calculate_expression_l306_306518

theorem calculate_expression :
  2^3 - (Real.tan (Real.pi / 3))^2 = 5 := by
  sorry

end calculate_expression_l306_306518


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306026

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306026


namespace prob_four_vertical_faces_same_color_l306_306104

noncomputable def painted_cube_probability : ℚ :=
  let total_arrangements := 3^6
  let suitable_arrangements := 3 + 18 + 6
  suitable_arrangements / total_arrangements

theorem prob_four_vertical_faces_same_color : 
  painted_cube_probability = 1 / 27 := by
  sorry

end prob_four_vertical_faces_same_color_l306_306104


namespace investment_ratio_l306_306344

variable (x : ℝ)
variable (p q t : ℝ)

theorem investment_ratio (h1 : 7 * p = 5 * q) (h2 : (7 * p * 8) / (5 * q * t) = 7 / 10) : t = 16 :=
by
  sorry

end investment_ratio_l306_306344


namespace slope_intercept_product_l306_306424

theorem slope_intercept_product (b m : ℤ) (h1 : b = -3) (h2 : m = 3) : m * b = -9 := by
  sorry

end slope_intercept_product_l306_306424


namespace smallest_k_divides_l306_306109

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (z : ℂ) : ∀ k : ℕ, (f z ∣ z^42 - 1) ∧ (∀ k' : ℕ, k' < 42 → ¬ (f z ∣ z^k' - 1)) :=
by
  sorry

end smallest_k_divides_l306_306109


namespace probability_all_genuine_given_equal_weights_l306_306934

variables (coins : Finset ℕ) (c_genuine : Finset ℕ) (c_counterfeit : Finset ℕ)
variable {coin_weight : ℕ → ℝ}

def areWeightsEqual (x y : ℕ) (a b : ℕ) : Prop :=
  coin_weight x + coin_weight y = coin_weight a + coin_weight b

noncomputable def probAllGenuineGivenEqualWeights : ℚ :=
  (conditionedProbability (allSelectedAreGenuine coins c_genuine c_counterfeit) 
    (weightsOfPairsAreEqual coins coin_weight))

axiom allSelectedAreGenuine : ProbEvent coins
axiom weightsOfPairsAreEqual : ProbEvent coins

theorem probability_all_genuine_given_equal_weights :
  probAllGenuineGivenEqualWeights = Rat.ofInt 15 / 19 := sorry

end probability_all_genuine_given_equal_weights_l306_306934


namespace solve_real_equation_l306_306693

theorem solve_real_equation (x : ℝ) (h : x^4 + (3 - x)^4 = 82) : x = 2.5 ∨ x = 0.5 :=
sorry

end solve_real_equation_l306_306693


namespace M_subsetneq_P_l306_306292

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

theorem M_subsetneq_P : M ⊂ P :=
by sorry

end M_subsetneq_P_l306_306292


namespace morgan_change_l306_306890

-- Define the costs of the items and the amount paid
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def amount_paid : ℕ := 20

-- Define total cost
def total_cost := hamburger_cost + onion_rings_cost + smoothie_cost

-- Define the change received
def change_received := amount_paid - total_cost

-- Statement of the problem in Lean 4
theorem morgan_change : change_received = 11 := by
  -- include proof steps here
  sorry

end morgan_change_l306_306890


namespace solve_for_star_l306_306202

theorem solve_for_star : ∀ (star : ℝ), (45 - (28 - (37 - (15 - star))) = 54) → star = 15 := by
  intros star h
  sorry

end solve_for_star_l306_306202


namespace distance_they_both_run_l306_306659

theorem distance_they_both_run
  (time_A time_B : ℕ)
  (distance_advantage: ℝ)
  (speed_A speed_B : ℝ)
  (D : ℝ) :
  time_A = 198 →
  time_B = 220 →
  distance_advantage = 300 →
  speed_A = D / time_A →
  speed_B = D / time_B →
  speed_A * time_B = D + distance_advantage →
  D = 2700 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end distance_they_both_run_l306_306659


namespace max_triangle_side_24_l306_306086

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306086


namespace train_speed_l306_306220

def speed_of_train (distance_m : ℕ) (time_s : ℕ) : ℕ :=
  ((distance_m : ℚ) / 1000) / ((time_s : ℚ) / 3600)

theorem train_speed (h_distance : 125 = 125) (h_time : 9 = 9) :
  speed_of_train 125 9 = 50 :=
by
  -- Proof is required here
  sorry

end train_speed_l306_306220


namespace min_n_for_constant_term_l306_306995

theorem min_n_for_constant_term :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, 3 * n = 5 * r) → ∃ n : ℕ, n = 5 :=
by
  intros h
  sorry

end min_n_for_constant_term_l306_306995


namespace stack_of_crates_count_l306_306328

open Finset

theorem stack_of_crates_count : 
  (∃ a b c : ℕ, 2 * a + 4 * b + 5 * c = 40 ∧ a + b + c = 12 ∧ 
    choose 12 a * choose (12 - a) b = 495 + 792 + 924) :=
by
  let configurations := [
    ⟨4, 8, 0⟩,
    ⟨5, 5, 2⟩,
    ⟨6, 2, 4⟩
  ]
  sorry

end stack_of_crates_count_l306_306328


namespace probability_of_double_domino_l306_306683

theorem probability_of_double_domino :
  let integers := Finset.range 13
  let all_pairs := (integers.product integers).filter (λ ⟨i, j⟩, i ≤ j)
  let doubles := all_pairs.filter (λ ⟨i, j⟩, i = j)
  (doubles.card : ℚ) / all_pairs.card = (1 : ℚ) / 13 :=
by
  -- Definitions
  let integers := Finset.range 13
  let all_pairs := (integers.product integers).filter (λ ⟨i, j⟩, i ≤ j)
  let doubles := all_pairs.filter (λ ⟨i, j⟩, i = j)
  
  -- Total number of doubles
  have h1 : doubles.card = 13 := by sorry
  
  -- Total number of pairs
  have h2 : all_pairs.card = 169 := by sorry
  
  -- Probability calculation
  have h3 : (doubles.card : ℚ) / all_pairs.card = (1 : ℚ) / 13 := by sorry
  
  exact h3


end probability_of_double_domino_l306_306683


namespace cryptarithm_base_solution_l306_306256

theorem cryptarithm_base_solution :
  ∃ (K I T : ℕ) (d : ℕ), 
    O = 0 ∧
    2 * T = I ∧
    T + 1 = K ∧
    K + I = d ∧ 
    d = 7 ∧ 
    K ≠ I ∧ K ≠ T ∧ K ≠ O ∧
    I ≠ T ∧ I ≠ O ∧
    T ≠ O :=
sorry

end cryptarithm_base_solution_l306_306256


namespace paperback_copies_sold_l306_306485

theorem paperback_copies_sold 
(H : ℕ)
(hardback_sold : H = 36000)
(P : ℕ)
(paperback_relation : P = 9 * H)
(total_copies : H + P = 440000) :
P = 324000 :=
sorry

end paperback_copies_sold_l306_306485


namespace intersecting_points_of_curves_l306_306544

theorem intersecting_points_of_curves :
  (∀ x y, (y = 2 * x^3 + x^2 - 5 * x + 2) ∧ (y = 3 * x^2 + 6 * x - 4) → 
   (x = -1 ∧ y = -7) ∨ (x = 3 ∧ y = 41)) := sorry

end intersecting_points_of_curves_l306_306544


namespace find_a_b_l306_306543

theorem find_a_b (a b : ℝ)
  (h1 : (0 - a)^2 + (-12 - b)^2 = 36)
  (h2 : (0 - a)^2 + (0 - b)^2 = 36) :
  a = 0 ∧ b = -6 :=
by
  sorry

end find_a_b_l306_306543


namespace g_15_33_eq_165_l306_306757

noncomputable def g : ℕ → ℕ → ℕ := sorry

axiom g_self (x : ℕ) : g x x = x
axiom g_comm (x y : ℕ) : g x y = g y x
axiom g_equation (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_15_33_eq_165 : g 15 33 = 165 := by sorry

end g_15_33_eq_165_l306_306757


namespace solution_l306_306621

-- Define the discount conditions
def discount (price : ℕ) : ℕ :=
  if price > 22 then price * 7 / 10 else
  if price < 20 then price * 8 / 10 else
  price

-- Define the given book prices
def book_prices : List ℕ := [25, 18, 21, 35, 12, 10]

-- Calculate total cost using the discount function
def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (λ acc price => acc + discount price) 0

def problem_statement : Prop :=
  total_cost book_prices = 95

theorem solution : problem_statement :=
  by
  unfold problem_statement
  unfold total_cost
  simp [book_prices, discount]
  sorry

end solution_l306_306621


namespace find_value_l306_306641

theorem find_value (a b : ℝ) (h : a + b + 1 = -2) : (a + b - 1) * (1 - a - b) = -16 := by
  sorry

end find_value_l306_306641


namespace a_alone_time_to_complete_work_l306_306647

theorem a_alone_time_to_complete_work :
  (W : ℝ) →
  (A : ℝ) →
  (B : ℝ) →
  (h1 : A + B = W / 6) →
  (h2 : B = W / 12) →
  A = W / 12 :=
by
  -- Given conditions
  intros W A B h1 h2
  -- Proof is not needed as per instructions
  sorry

end a_alone_time_to_complete_work_l306_306647


namespace find_circle_center_l306_306477

theorem find_circle_center : ∃ (h k : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x + 12*y + 1 = 0 ↔ (x - h)^2 + (y - k)^2 = 36) ∧ h = 1 ∧ k = -6 := 
sorry

end find_circle_center_l306_306477


namespace ladder_base_l306_306349

theorem ladder_base (h : ℝ) (b : ℝ) (l : ℝ)
  (h_eq : h = 12) (l_eq : l = 15) : b = 9 :=
by
  have hypotenuse := l
  have height := h
  have base := b
  have pythagorean_theorem : height^2 + base^2 = hypotenuse^2 := by sorry 
  sorry

end ladder_base_l306_306349


namespace find_speed_of_P_l306_306191

noncomputable def walking_speeds (v_P v_Q : ℝ) : Prop :=
  let distance_XY := 90
  let distance_meet_from_Y := 15
  let distance_P := distance_XY - distance_meet_from_Y
  let distance_Q := distance_XY + distance_meet_from_Y
  (v_Q = v_P + 3) ∧
  (distance_P / v_P = distance_Q / v_Q)

theorem find_speed_of_P : ∃ v_P : ℝ, walking_speeds v_P (v_P + 3) ∧ v_P = 7.5 :=
by
  sorry

end find_speed_of_P_l306_306191


namespace quadratic_root_value_l306_306144

theorem quadratic_root_value {m : ℝ} (h : m^2 + m - 1 = 0) : 2 * m^2 + 2 * m + 2025 = 2027 :=
sorry

end quadratic_root_value_l306_306144


namespace total_guppies_correct_l306_306091

-- Define the initial conditions as variables
def initial_guppies : ℕ := 7
def baby_guppies_1 : ℕ := 3 * 12
def baby_guppies_2 : ℕ := 9

-- Define the total number of guppies
def total_guppies : ℕ := initial_guppies + baby_guppies_1 + baby_guppies_2

-- Theorem: Proving the total number of guppies is 52
theorem total_guppies_correct : total_guppies = 52 :=
by
  sorry

end total_guppies_correct_l306_306091


namespace range_of_a_l306_306734

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → (a ≤ -1) :=
by 
  sorry

end range_of_a_l306_306734


namespace jennifer_fifth_score_l306_306878

theorem jennifer_fifth_score :
  ∀ (x : ℝ), (85 + 90 + 87 + 92 + x) / 5 = 89 → x = 91 :=
by
  sorry

end jennifer_fifth_score_l306_306878


namespace total_value_after_3_years_l306_306952

noncomputable def value_after_years (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

def machine1_initial_value : ℝ := 2500
def machine1_depreciation_rate : ℝ := 0.05
def machine2_initial_value : ℝ := 3500
def machine2_depreciation_rate : ℝ := 0.07
def machine3_initial_value : ℝ := 4500
def machine3_depreciation_rate : ℝ := 0.04
def years : ℕ := 3

theorem total_value_after_3_years :
  value_after_years machine1_initial_value machine1_depreciation_rate years +
  value_after_years machine2_initial_value machine2_depreciation_rate years +
  value_after_years machine3_initial_value machine3_depreciation_rate years = 8940 :=
by
  sorry

end total_value_after_3_years_l306_306952


namespace not_perfect_square_l306_306235

theorem not_perfect_square (x y : ℤ) : ¬ ∃ k : ℤ, k^2 = (x^2 + x + 1)^2 + (y^2 + y + 1)^2 :=
by
  sorry

end not_perfect_square_l306_306235


namespace seats_filled_percentage_l306_306554

theorem seats_filled_percentage (total_seats vacant_seats : ℕ) (h1 : total_seats = 600) (h2 : vacant_seats = 228) :
  ((total_seats - vacant_seats) / total_seats * 100 : ℝ) = 62 := by
  sorry

end seats_filled_percentage_l306_306554


namespace find_M_l306_306568

theorem find_M (A M C : ℕ) (h1 : (100 * A + 10 * M + C) * (A + M + C) = 2040)
(h2 : (A + M + C) % 2 = 0)
(h3 : A ≤ 9) (h4 : M ≤ 9) (h5 : C ≤ 9) :
  M = 7 := 
sorry

end find_M_l306_306568


namespace count_positive_integers_l306_306687

theorem count_positive_integers (n : ℕ) : ∃ k : ℕ, k = 9 ∧  ∀ n, 1 ≤ n → n < 10 → 3 * n + 20 < 50 :=
by
  sorry

end count_positive_integers_l306_306687


namespace min_x2_y2_z2_l306_306707

theorem min_x2_y2_z2 (x y z : ℝ) (h : 2 * x + 3 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 22 :=
by
  sorry

end min_x2_y2_z2_l306_306707


namespace cylinder_volume_ratio_l306_306761

theorem cylinder_volume_ratio
  (S1 S2 : ℝ) (v1 v2 : ℝ)
  (lateral_area_equal : 2 * Real.pi * S1.sqrt = 2 * Real.pi * S2.sqrt)
  (base_area_ratio : S1 / S2 = 16 / 9) :
  v1 / v2 = 4 / 3 :=
by
  sorry

end cylinder_volume_ratio_l306_306761


namespace Benjamin_has_45_presents_l306_306525

-- Define the number of presents each person has
def Ethan_presents : ℝ := 31.5
def Alissa_presents : ℝ := Ethan_presents + 22
def Benjamin_presents : ℝ := Alissa_presents - 8.5

-- The statement we need to prove
theorem Benjamin_has_45_presents : Benjamin_presents = 45 :=
by
  -- on the last line, we type sorry to skip the actual proof
  sorry

end Benjamin_has_45_presents_l306_306525


namespace intersection_A_B_l306_306861

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def setA : Set ℝ := { x | Real.log x > 0 }
def setB : Set ℝ := { x | Real.exp x * Real.exp x < 3 }

theorem intersection_A_B : setA ∩ setB = { x | 1 < x ∧ x < log2 3 } :=
by
  sorry

end intersection_A_B_l306_306861


namespace determine_sequence_parameters_l306_306555

variables {n : ℕ} {d q : ℝ} (h1 : 1 + (n-1) * d = 81) (h2 : 1 * q^(n-1) = 81) (h3 : q / d = 0.15)

theorem determine_sequence_parameters : n = 5 ∧ d = 20 ∧ q = 3 :=
by {
  -- Assumptions:
  -- h1: Arithmetic sequence, a1 = 1, an = 81
  -- h2: Geometric sequence, b1 = 1, bn = 81
  -- h3: q / d = 0.15
  -- Goal: n = 5, d = 20, q = 3
  sorry
}

end determine_sequence_parameters_l306_306555


namespace dot_product_is_ten_l306_306863

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the condition that the vectors are parallel
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 / v2.1 = v1.2 / v2.2

-- The main theorem statement
theorem dot_product_is_ten (m : ℝ) (h : parallel a (b m)) : 
  a.1 * (b m).1 + a.2 * (b m).2 = 10 := by
  sorry

end dot_product_is_ten_l306_306863


namespace max_side_length_l306_306000

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306000


namespace not_eq_positive_integers_l306_306759

theorem not_eq_positive_integers (a b : ℤ) (ha : a > 0) (hb : b > 0) : 
  a^3 + (a + b)^2 + b ≠ b^3 + a + 2 :=
by {
  sorry
}

end not_eq_positive_integers_l306_306759


namespace shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l306_306216

def false_weight_kgs (false_weight_g : ℕ) : ℚ := false_weight_g / 1000

def shopkeeper_gain_percentage (false_weight_g price_per_kg : ℕ) : ℚ :=
  let actual_price := false_weight_kgs false_weight_g * price_per_kg
  let gain := price_per_kg - actual_price
  (gain / actual_price) * 100

theorem shopkeeper_gain_first_pulse :
  shopkeeper_gain_percentage 950 10 = 5.26 := 
sorry

theorem shopkeeper_gain_second_pulse :
  shopkeeper_gain_percentage 960 15 = 4.17 := 
sorry

theorem shopkeeper_gain_third_pulse :
  shopkeeper_gain_percentage 970 20 = 3.09 := 
sorry

end shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l306_306216


namespace repeating_decimal_to_fraction_l306_306390

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l306_306390


namespace probability_of_divisible_by_3_before_not_divisible_l306_306506

noncomputable def probability_rolls (d : List ℕ) : ℚ :=
  let divisible_by_3 := [3, 6]
  let not_divisible_by_3 := [1, 2, 4, 5, 7, 8]
  -- Assuming independence and uniform probability distribution
  let p_div3 := 1 / 4
  let p_not_div3 := 3 / 4
  let p_valid_sequence := (n : ℕ) -> (p_div3 ^ (n-1)) * p_not_div3
  let p_complement := (n : ℕ) -> p_valid_sequence n * (2 / 2^(n-1))
  let p_total := ∑' n, if n >= 3 then p_valid_sequence n - p_complement n else 0
  p_total

theorem probability_of_divisible_by_3_before_not_divisible :
  probability_rolls [1, 2, 3, 4, 5, 6, 7, 8] = (3 / 128) :=
by
  sorry

end probability_of_divisible_by_3_before_not_divisible_l306_306506


namespace repeating_decimal_fraction_l306_306381

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l306_306381


namespace integer_solution_count_l306_306434

theorem integer_solution_count :
  {n : ℤ | n ≠ 0 ∧ (1 / |(n:ℚ)| ≥ 1 / 5)}.finite.card = 10 :=
by
  sorry

end integer_solution_count_l306_306434


namespace g_at_10_is_300_l306_306881

-- Define the function g and the given condition about g
def g: ℕ → ℤ := sorry

axiom g_cond (m n: ℕ) (h: m ≥ n): g (m + n) + g (m - n) = 2 * g m + 3 * g n
axiom g_1: g 1 = 3

-- Statement to be proved
theorem g_at_10_is_300 : g 10 = 300 := by
  sorry

end g_at_10_is_300_l306_306881


namespace polynomial_quotient_correct_l306_306972

noncomputable def polynomial_division_quotient : Polynomial ℝ :=
  (Polynomial.C 1 * Polynomial.X^6 + Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 8) / (Polynomial.X - Polynomial.C 1)

-- Math proof statement
theorem polynomial_quotient_correct :
  polynomial_division_quotient = Polynomial.C 1 * Polynomial.X^5 + Polynomial.C 1 * Polynomial.X^4 
                                 + Polynomial.C 1 * Polynomial.X^3 + Polynomial.C 1 * Polynomial.X^2 
                                 + Polynomial.C 3 * Polynomial.X + Polynomial.C 3 :=
by
  sorry

end polynomial_quotient_correct_l306_306972


namespace money_made_arkansas_game_is_8722_l306_306775

def price_per_tshirt : ℕ := 98
def tshirts_sold_arkansas_game : ℕ := 89
def total_money_made_arkansas_game (price_per_tshirt tshirts_sold_arkansas_game : ℕ) : ℕ :=
  price_per_tshirt * tshirts_sold_arkansas_game

theorem money_made_arkansas_game_is_8722 :
  total_money_made_arkansas_game price_per_tshirt tshirts_sold_arkansas_game = 8722 :=
by
  sorry

end money_made_arkansas_game_is_8722_l306_306775


namespace percent_of_y_eq_l306_306494

theorem percent_of_y_eq (y : ℝ) (h : y ≠ 0) : (0.3 * 0.7 * y) = (0.21 * y) := by
  sorry

end percent_of_y_eq_l306_306494


namespace boarders_initial_count_l306_306782

noncomputable def initial_boarders (x : ℕ) : ℕ := 7 * x

theorem boarders_initial_count (x : ℕ) (h1 : 80 + initial_boarders x = (2 : ℝ) * 16) :
  initial_boarders x = 560 :=
by
  sorry

end boarders_initial_count_l306_306782


namespace sum_a_b_max_power_l306_306829

theorem sum_a_b_max_power (a b : ℕ) (h_pos : 0 < a) (h_b_gt_1 : 1 < b) (h_lt_600 : a ^ b < 600) : a + b = 26 :=
sorry

end sum_a_b_max_power_l306_306829


namespace difference_of_digits_l306_306776

theorem difference_of_digits (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h_diff : (10 * x + y) - (10 * y + x) = 54) : x - y = 6 :=
sorry

end difference_of_digits_l306_306776


namespace number_of_integers_satisfying_inequality_l306_306437

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℚ) / |n| ≥ 1 / 5}.to_finset.card = 10 :=
by {
  sorry
}

end number_of_integers_satisfying_inequality_l306_306437


namespace winning_margin_l306_306617

theorem winning_margin (total_votes : ℝ) (winning_votes : ℝ) (winning_percent : ℝ) (losing_percent : ℝ) 
  (win_votes_eq: winning_votes = winning_percent * total_votes)
  (perc_eq: winning_percent + losing_percent = 1)
  (win_votes_given: winning_votes = 550)
  (winning_percent_given: winning_percent = 0.55)
  (losing_percent_given: losing_percent = 0.45) :
  winning_votes - (losing_percent * total_votes) = 100 := 
by
  sorry

end winning_margin_l306_306617


namespace pencil_notebook_cost_l306_306961

variable {p n : ℝ}

theorem pencil_notebook_cost (hp1 : 9 * p + 11 * n = 6.05) (hp2 : 6 * p + 4 * n = 2.68) :
  18 * p + 13 * n = 8.45 :=
sorry

end pencil_notebook_cost_l306_306961


namespace WorldCup_group_stage_matches_l306_306474

theorem WorldCup_group_stage_matches
  (teams : ℕ)
  (groups : ℕ)
  (teams_per_group : ℕ)
  (matches_per_group : ℕ)
  (total_matches : ℕ) :
  teams = 32 ∧ 
  groups = 8 ∧ 
  teams_per_group = 4 ∧ 
  matches_per_group = teams_per_group * (teams_per_group - 1) / 2 ∧ 
  total_matches = matches_per_group * groups →
  total_matches = 48 :=
by 
  -- sorry lets Lean skip the proof.
  sorry

end WorldCup_group_stage_matches_l306_306474


namespace binomial_square_coefficients_l306_306990

noncomputable def a : ℝ := 13.5
noncomputable def b : ℝ := 18

theorem binomial_square_coefficients (c d : ℝ) :
  (∀ x : ℝ, 6 * x ^ 2 + 18 * x + a = (c * x + d) ^ 2) ∧ 
  (∀ x : ℝ, 3 * x ^ 2 + b * x + 4 = (c * x + d) ^ 2)  → 
  a = 13.5 ∧ b = 18 := sorry

end binomial_square_coefficients_l306_306990


namespace parallel_lines_sufficient_but_not_necessary_l306_306417

theorem parallel_lines_sufficient_but_not_necessary (a : ℝ) :
  (a = 1 ↔ ((ax + y - 1 = 0) ∧ (x + ay + 1 = 0) → False)) := 
sorry

end parallel_lines_sufficient_but_not_necessary_l306_306417


namespace smaller_angle_is_85_l306_306737

-- Conditions
def isParallelogram (α β : ℝ) : Prop :=
  α + β = 180

def angleExceedsBy10 (α β : ℝ) : Prop :=
  β = α + 10

-- Proof Problem
theorem smaller_angle_is_85 (α β : ℝ)
  (h1 : isParallelogram α β)
  (h2 : angleExceedsBy10 α β) :
  α = 85 :=
by
  sorry

end smaller_angle_is_85_l306_306737


namespace nancy_initial_bottle_caps_l306_306295

theorem nancy_initial_bottle_caps (found additional_bottle_caps: ℕ) (total_bottle_caps: ℕ) (h1: additional_bottle_caps = 88) (h2: total_bottle_caps = 179) : 
  (total_bottle_caps - additional_bottle_caps) = 91 :=
by
  sorry

end nancy_initial_bottle_caps_l306_306295


namespace Polyas_probability_relation_l306_306488

variable (Z : ℕ → ℤ → ℝ)

theorem Polyas_probability_relation (n : ℕ) (k : ℤ) :
  Z n k = (1/2) * (Z (n-1) (k-1) + Z (n-1) (k+1)) :=
by
  sorry

end Polyas_probability_relation_l306_306488


namespace sin_half_angle_product_lt_quarter_l306_306577

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end sin_half_angle_product_lt_quarter_l306_306577


namespace calc_value_l306_306237

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem calc_value :
  ((diamond (diamond 3 4) 2) - (diamond 3 (diamond 4 2))) = -13 / 28 :=
by sorry

end calc_value_l306_306237


namespace days_before_reinforcement_l306_306217

/-- A garrison of 2000 men originally has provisions for 62 days.
    After some days, a reinforcement of 2700 men arrives.
    The provisions are found to last for only 20 days more after the reinforcement arrives.
    Prove that the number of days passed before the reinforcement arrived is 15. -/
theorem days_before_reinforcement 
  (x : ℕ) 
  (num_men_orig : ℕ := 2000) 
  (num_men_reinf : ℕ := 2700) 
  (days_orig : ℕ := 62) 
  (days_after_reinf : ℕ := 20) 
  (total_provisions : ℕ := num_men_orig * days_orig)
  (remaining_provisions : ℕ := num_men_orig * (days_orig - x))
  (consumption_after_reinf : ℕ := (num_men_orig + num_men_reinf) * days_after_reinf) 
  (provisions_eq : remaining_provisions = consumption_after_reinf) : 
  x = 15 := 
by 
  sorry

end days_before_reinforcement_l306_306217


namespace polynomial_coeff_sum_neg_33_l306_306438

theorem polynomial_coeff_sum_neg_33
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (2 - 3 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -33 :=
by sorry

end polynomial_coeff_sum_neg_33_l306_306438


namespace true_discount_l306_306906

theorem true_discount (BD PV TD : ℝ) (h1 : BD = 36) (h2 : PV = 180) :
  TD = 30 :=
by
  sorry

end true_discount_l306_306906


namespace solution_set_of_inequality_l306_306923

theorem solution_set_of_inequality (x : ℝ) : ((x - 1) * (2 - x) ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end solution_set_of_inequality_l306_306923


namespace fraction_simplification_l306_306373

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 3) = 516 / 455 := by
  sorry

end fraction_simplification_l306_306373


namespace min_value_collinear_l306_306415

theorem min_value_collinear (x y : ℝ) (h₁ : 2 * x + 3 * y = 3) (h₂ : 0 < x) (h₃ : 0 < y) : 
  (3 / x + 2 / y) = 8 :=
sorry

end min_value_collinear_l306_306415


namespace parabola_point_focus_distance_l306_306261

/-- 
  Given a point P on the parabola y^2 = 4x, and the distance from P to the line x = -2
  is 5 units, prove that the distance from P to the focus of the parabola is 4 units.
-/
theorem parabola_point_focus_distance {P : ℝ × ℝ} 
  (hP : P.2^2 = 4 * P.1) 
  (h_dist : (P.1 + 2)^2 + P.2^2 = 25) : 
  dist P (1, 0) = 4 :=
sorry

end parabola_point_focus_distance_l306_306261


namespace d_n_2_d_n_3_l306_306871

def d (n k : ℕ) : ℕ :=
  if k = 0 then 1
  else if n = 1 then 0
  else (0:ℕ) -- Placeholder to demonstrate that we need a recurrence relation, not strictly necessary here for the statement.

theorem d_n_2 (n : ℕ) (hn : n ≥ 2) : 
  d n 2 = (n^2 - 3*n + 2) / 2 := 
by 
  sorry

theorem d_n_3 (n : ℕ) (hn : n ≥ 3) : 
  d n 3 = (n^3 - 7*n + 6) / 6 := 
by 
  sorry

end d_n_2_d_n_3_l306_306871


namespace count_4x4_increasing_arrays_l306_306180

-- Define the notion of a 4x4 grid that satisfies the given conditions
def isInIncreasingOrder (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, i < 3 -> matrix i j < matrix (i+1) j) ∧
  (∀ i j : Fin 4, j < 3 -> matrix i j < matrix i (j+1))

def validGrid (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, 1 ≤ matrix i j ∧ matrix i j ≤ 16) ∧ isInIncreasingOrder matrix

noncomputable def countValidGrids : ℕ :=
  sorry

theorem count_4x4_increasing_arrays : countValidGrids = 13824 :=
  sorry

end count_4x4_increasing_arrays_l306_306180


namespace square_tiles_count_l306_306738

theorem square_tiles_count 
  (h s : ℕ)
  (total_tiles : h + s = 30)
  (total_edges : 6 * h + 4 * s = 128) : 
  s = 26 :=
by
  sorry

end square_tiles_count_l306_306738


namespace normal_distribution_symmetry_l306_306977

theorem normal_distribution_symmetry {σ : ℝ} (hσ : 0 < σ) :
  P(ξ < 2016) = 0.5 :=
by 
  sorry

end normal_distribution_symmetry_l306_306977


namespace find_k_l306_306451

-- Define the sequence according to the given conditions
def seq (n : ℕ) : ℕ :=
  if n = 1 then 2 else sorry

axiom seq_base : seq 1 = 2
axiom seq_recur (m n : ℕ) : seq (m + n) = seq m * seq n

-- Given condition for sum
axiom sum_condition (k : ℕ) : 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5

-- The statement to prove
theorem find_k : ∃ k : ℕ, 
  (finset.range 10).sum (λ i, seq (k + 1 + i)) = 2^15 - 2^5 ∧ k = 4 :=
by {
  sorry -- proof is not required
}

end find_k_l306_306451


namespace initial_percentage_of_chemical_x_l306_306208

theorem initial_percentage_of_chemical_x (P : ℝ) (h1 : 20 + 80 * P = 44) : P = 0.3 :=
by sorry

end initial_percentage_of_chemical_x_l306_306208


namespace max_side_of_triangle_l306_306043

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306043


namespace probability_at_least_four_girls_l306_306285

noncomputable def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_at_least_four_girls
  (n : ℕ)
  (p : ℝ)
  (q : ℝ)
  (h_pq : p + q = 1)
  (h_p : p = 0.55)
  (h_q : q = 0.45)
  (h_n : n = 7) :
  (binomial_probability n 4 p) + (binomial_probability n 5 p) + (binomial_probability n 6 p) + (binomial_probability n 7 p) = 0.59197745 :=
sorry

end probability_at_least_four_girls_l306_306285


namespace domain_of_f_2x_minus_3_l306_306984

noncomputable def f (x : ℝ) := 2 * x + 1

theorem domain_of_f_2x_minus_3 :
  (∀ x, 1 ≤ 2 * x - 3 ∧ 2 * x - 3 ≤ 5 → (2 ≤ x ∧ x ≤ 4)) :=
by
  sorry

end domain_of_f_2x_minus_3_l306_306984


namespace remainder_5_7_9_6_3_5_mod_7_l306_306196

theorem remainder_5_7_9_6_3_5_mod_7 : (5^7 + 9^6 + 3^5) % 7 = 5 :=
by sorry

end remainder_5_7_9_6_3_5_mod_7_l306_306196


namespace carlos_fraction_l306_306680

theorem carlos_fraction (f : ℝ) :
  (1 - f) ^ 4 * 64 = 4 → f = 1 / 2 :=
by
  intro h
  sorry

end carlos_fraction_l306_306680


namespace system_solutions_a_l306_306472

theorem system_solutions_a (x y z : ℝ) :
  (2 * x = (y + z) ^ 2) ∧ (2 * y = (z + x) ^ 2) ∧ (2 * z = (x + y) ^ 2) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solutions_a_l306_306472


namespace percentage_decrease_10_l306_306364

def stocks_decrease (F J M : ℝ) (X : ℝ) : Prop :=
  J = F * (1 - X / 100) ∧
  J = M * 1.20 ∧
  M = F * 0.7500000000000007

theorem percentage_decrease_10 {F J M X : ℝ} (h : stocks_decrease F J M X) :
  X = 9.99999999999992 :=
by
  sorry

end percentage_decrease_10_l306_306364


namespace meal_total_l306_306233

noncomputable def meal_price (appetizer entree dessert drink sales_tax tip : ℝ) : ℝ :=
  let total_before_tax := appetizer + (2 * entree) + dessert + (2 * drink)
  let tax_amount := (sales_tax / 100) * total_before_tax
  let subtotal := total_before_tax + tax_amount
  let tip_amount := (tip / 100) * subtotal
  subtotal + tip_amount

theorem meal_total : 
  meal_price 9 20 11 6.5 7.5 22 = 95.75 :=
by
  sorry

end meal_total_l306_306233


namespace max_side_of_triangle_l306_306058

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306058


namespace intersection_complement_correct_l306_306987

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 5}
def B : Set ℕ := {1, 4}
def C_I (s : Set ℕ) := I \ s  -- set complement

theorem intersection_complement_correct: A ∩ C_I B = {3, 5} := by
  -- proof steps go here
  sorry

end intersection_complement_correct_l306_306987


namespace expressions_not_equal_l306_306528

theorem expressions_not_equal (x : ℝ) (hx : x > 0) : 
  3 * x^x ≠ 2 * x^x + x^(2 * x) ∧ 
  x^(3 * x) ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^x ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^(3 * x) ≠ 2 * x^x + x^(2 * x) :=
by 
  sorry

end expressions_not_equal_l306_306528


namespace N_square_solutions_l306_306242

theorem N_square_solutions :
  ∀ N : ℕ, (N > 0 → ∃ k : ℕ, 2^N - 2 * N = k^2) → (N = 1 ∨ N = 2) :=
by
  sorry

end N_square_solutions_l306_306242


namespace train_speed_in_kmph_l306_306221

def train_length : ℕ := 125
def time_to_cross_pole : ℕ := 9
def conversion_factor : ℚ := 18 / 5

theorem train_speed_in_kmph
  (d : ℕ := train_length)
  (t : ℕ := time_to_cross_pole)
  (cf : ℚ := conversion_factor) :
  d / t * cf = 50 := 
sorry

end train_speed_in_kmph_l306_306221


namespace time_spent_on_aerobics_l306_306562

theorem time_spent_on_aerobics (A W : ℝ) 
  (h1 : A + W = 250) 
  (h2 : A / W = 3 / 2) : 
  A = 150 := 
sorry

end time_spent_on_aerobics_l306_306562


namespace find_m_value_l306_306123

noncomputable def x0 : ℝ := sorry

noncomputable def m : ℝ := x0^3 + 2 * x0^2 + 2

theorem find_m_value :
  (x0^2 + x0 - 1 = 0) → (m = 3) :=
by
  intro h
  have hx : x0 = sorry := sorry
  have hm : m = x0 ^ 3 + 2 * x0^2 + 2 := rfl
  rw [hx] at hm
  sorry

end find_m_value_l306_306123


namespace calculation_correct_l306_306330

theorem calculation_correct :
  15 * ( (1/3 : ℚ) + (1/4) + (1/6) )⁻¹ = 20 := sorry

end calculation_correct_l306_306330


namespace cube_volume_from_surface_area_l306_306204

theorem cube_volume_from_surface_area (A : ℝ) (h : A = 54) :
  ∃ V : ℝ, V = 27 := by
  sorry

end cube_volume_from_surface_area_l306_306204


namespace sum_of_x_and_y_l306_306148

theorem sum_of_x_and_y (x y : ℝ) (h1 : x + abs x + y = 5) (h2 : x + abs y - y = 6) : x + y = 9 / 5 :=
by
  sorry

end sum_of_x_and_y_l306_306148


namespace integer_not_in_range_of_f_l306_306290

noncomputable def f (x : ℝ) : ℤ :=
  if x > -1 then ⌈1 / (x + 1)⌉ else ⌊1 / (x + 1)⌋

theorem integer_not_in_range_of_f :
  ¬ ∃ x : ℝ, x ≠ -1 ∧ f x = 0 :=
by
  sorry

end integer_not_in_range_of_f_l306_306290


namespace radian_to_degree_conversion_l306_306097

theorem radian_to_degree_conversion
: (π : ℝ) = 180 → ((-23 / 12) * π) = -345 :=
by
  sorry

end radian_to_degree_conversion_l306_306097


namespace value_of_expression_l306_306442

theorem value_of_expression (x y z : ℕ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 1) : 
  3 * x - 2 * y + 4 * z = 9 := 
by
  sorry

end value_of_expression_l306_306442


namespace max_side_length_is_11_l306_306023

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306023


namespace negation_of_p_l306_306859

def p : Prop := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x ≤ Real.sin x :=
by sorry

end negation_of_p_l306_306859


namespace intersection_distance_l306_306126

open Real

-- Definition of the curve C in standard coordinates
def curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Definition of the line l in parametric form
def line_l (x y t : ℝ) : Prop :=
  x = 1 + t ∧ y = -1 + t

-- The length of the intersection points A and B of curve C and line l
theorem intersection_distance : ∃ t1 t2 : ℝ, (curve_C (1 + t1) (-1 + t1) ∧ curve_C (1 + t2) (-1 + t2)) ∧ (abs (t1 - t2) = 4 * sqrt 6) :=
sorry

end intersection_distance_l306_306126


namespace intersection_empty_l306_306129

def A : Set ℝ := {x | x > -1 ∧ x ≤ 3}
def B : Set ℝ := {2, 4}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l306_306129


namespace inequality_hold_l306_306426

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l306_306426


namespace greatest_third_term_arithmetic_seq_l306_306186

theorem greatest_third_term_arithmetic_seq (a d : ℤ) (h1: a > 0) (h2: d ≥ 0) (h3: 5 * a + 10 * d = 65) : 
  a + 2 * d = 13 := 
by 
  sorry

end greatest_third_term_arithmetic_seq_l306_306186


namespace distinct_positive_integer_roots_l306_306503

theorem distinct_positive_integer_roots (m a b : ℤ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : a + b = -m) (h5 : a * b = -m + 1) : m = -5 := 
by
  sorry

end distinct_positive_integer_roots_l306_306503


namespace gcd_111_148_l306_306840

theorem gcd_111_148 : Nat.gcd 111 148 = 37 :=
by
  sorry

end gcd_111_148_l306_306840


namespace larger_number_is_84_l306_306915

theorem larger_number_is_84 (x y : ℕ) (HCF LCM : ℕ)
  (h_hcf : HCF = 84)
  (h_lcm : LCM = 21)
  (h_ratio : x * 4 = y)
  (h_product : x * y = HCF * LCM) :
  y = 84 :=
by
  sorry

end larger_number_is_84_l306_306915


namespace divide_shape_into_equal_parts_l306_306686

-- Definitions and conditions
structure Shape where
  has_vertical_symmetry : Bool
  -- Other properties of the shape can be added as necessary

def vertical_line_divides_equally (s : Shape) : Prop :=
  s.has_vertical_symmetry

-- Theorem statement
theorem divide_shape_into_equal_parts (s : Shape) (h : s.has_vertical_symmetry = true) :
  vertical_line_divides_equally s :=
by
  -- Begin proof
  sorry

end divide_shape_into_equal_parts_l306_306686


namespace pizza_sales_calculation_l306_306664

def pizzas_sold_in_spring (total_sales : ℝ) (summer_sales : ℝ) (fall_percentage : ℝ) (winter_percentage : ℝ) : ℝ :=
  total_sales - summer_sales - (fall_percentage * total_sales) - (winter_percentage * total_sales)

theorem pizza_sales_calculation :
  let summer_sales := 5;
  let fall_percentage := 0.1;
  let winter_percentage := 0.2;
  ∃ (total_sales : ℝ), 0.4 * total_sales = summer_sales ∧
    pizzas_sold_in_spring total_sales summer_sales fall_percentage winter_percentage = 3.75 :=
by
  sorry

end pizza_sales_calculation_l306_306664


namespace zoo_animal_difference_l306_306224

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  monkeys - zebras = 35 := by
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  show monkeys - zebras = 35
  sorry

end zoo_animal_difference_l306_306224


namespace fewest_coach_handshakes_l306_306513

theorem fewest_coach_handshakes (n m1 m2 : ℕ) 
  (handshakes_total : (n * (n - 1)) / 2 + m1 + m2 = 465) 
  (m1_m2_eq_n : m1 + m2 = n) : 
  n * (n - 1) / 2 = 465 → m1 + m2 = 0 :=
by 
  sorry

end fewest_coach_handshakes_l306_306513


namespace repeating_decimal_fraction_l306_306382

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l306_306382


namespace g_6_eq_1_l306_306122

variable (f : ℝ → ℝ)

noncomputable def g (x : ℝ) := f x + 1 - x

theorem g_6_eq_1 
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1) :
  g f 6 = 1 :=
by
  sorry

end g_6_eq_1_l306_306122


namespace max_value_of_f_l306_306842

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_of_f : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 2 := 
by
  sorry

end max_value_of_f_l306_306842


namespace balcony_more_than_orchestra_l306_306649

theorem balcony_more_than_orchestra (O B : ℕ) 
  (h1 : O + B = 355) 
  (h2 : 12 * O + 8 * B = 3320) : 
  B - O = 115 :=
by 
  -- Sorry, this will skip the proof.
  sorry

end balcony_more_than_orchestra_l306_306649


namespace repeating_decimal_to_fraction_l306_306392

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l306_306392


namespace max_xy_max_xy_value_l306_306539

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y ≤ 3 :=
sorry

theorem max_xy_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y = 3 → x = 3 / 2 ∧ y = 2 :=
sorry

end max_xy_max_xy_value_l306_306539


namespace simplify_and_evaluate_l306_306307

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = 1/25) (h2 : y = -25) :
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = -3 :=
by
  sorry

end simplify_and_evaluate_l306_306307


namespace odd_function_neg_expression_l306_306777

theorem odd_function_neg_expression (f : ℝ → ℝ) (h₀ : ∀ x > 0, f x = x^3 + x + 1)
    (h₁ : ∀ x, f (-x) = -f x) : ∀ x < 0, f x = x^3 + x - 1 :=
by
  sorry

end odd_function_neg_expression_l306_306777


namespace last_five_digits_of_sequence_l306_306184

theorem last_five_digits_of_sequence (seq : Fin 36 → Fin 2) 
  (h0 : seq 0 = 0) (h1 : seq 1 = 0) (h2 : seq 2 = 0) (h3 : seq 3 = 0) (h4 : seq 4 = 0)
  (unique_combos : ∀ (combo: Fin 32 → Fin 2), 
    ∃ (start_index : Fin 32), ∀ (i : Fin 5),
      combo i = seq ((start_index + i) % 36)) :
  seq 31 = 1 ∧ seq 32 = 1 ∧ seq 33 = 1 ∧ seq 34 = 0 ∧ seq 35 = 1 :=
by
  sorry

end last_five_digits_of_sequence_l306_306184


namespace chris_money_before_birthday_l306_306365

-- Define the given amounts of money from each source
def money_from_grandmother : ℕ := 25
def money_from_aunt_and_uncle : ℕ := 20
def money_from_parents : ℕ := 75
def total_money_now : ℕ := 279

-- Calculate the total birthday money
def total_birthday_money := money_from_grandmother + money_from_aunt_and_uncle + money_from_parents

-- Define the amount of money Chris had before his birthday
def money_before_birthday := total_money_now - total_birthday_money

-- The proof statement
theorem chris_money_before_birthday : money_before_birthday = 159 :=
by
  sorry

end chris_money_before_birthday_l306_306365


namespace at_least_one_angle_not_less_than_sixty_l306_306171

theorem at_least_one_angle_not_less_than_sixty (A B C : ℝ)
  (hABC_sum : A + B + C = 180)
  (hA : A < 60)
  (hB : B < 60)
  (hC : C < 60) : false :=
by
  sorry

end at_least_one_angle_not_less_than_sixty_l306_306171


namespace repeating_decimal_equals_fraction_l306_306400

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l306_306400


namespace min_value_x_plus_2_div_x_minus_2_l306_306848

theorem min_value_x_plus_2_div_x_minus_2 (x : ℝ) (h : x > 2) : 
  ∃ m, m = 2 + 2 * Real.sqrt 2 ∧ x + 2/(x-2) ≥ m :=
by sorry

end min_value_x_plus_2_div_x_minus_2_l306_306848


namespace range_of_m_l306_306542

theorem range_of_m (x m : ℝ) (h1 : 2 * x - m ≤ 3) (h2 : -5 < x) (h3 : x < 4) :
  ∃ m, ∀ (x : ℝ), (-5 < x ∧ x < 4) → (2 * x - m ≤ 3) ↔ (m ≥ 5) :=
by sorry

end range_of_m_l306_306542


namespace max_side_length_is_11_l306_306019

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306019


namespace part1_part2_l306_306535

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x^2 ≤ 5 * x - 4
def q (x a : ℝ) : Prop := x^2 - (a + 2) * x + 2 * a ≤ 0

-- Theorem statement for part (1)
theorem part1 (x : ℝ) (h : p x) : 1 ≤ x ∧ x ≤ 4 := 
by sorry

-- Theorem statement for part (2)
theorem part2 (a : ℝ) : 
  (∀ x, p x → q x a) ∧ (∃ x, p x) ∧ ¬ (∀ x, q x a → p x) → 1 ≤ a ∧ a ≤ 4 := 
by sorry

end part1_part2_l306_306535


namespace number_of_handshakes_l306_306672

-- Definitions based on the conditions:
def number_of_teams : ℕ := 4
def number_of_women_per_team : ℕ := 2
def total_women : ℕ := number_of_teams * number_of_women_per_team

-- Each woman shakes hands with all others except her partner
def handshakes_per_woman : ℕ := total_women - 1 - (number_of_women_per_team - 1)

-- Calculate total handshakes, considering each handshake is counted twice
def total_handshakes : ℕ := (total_women * handshakes_per_woman) / 2

-- Statement to prove
theorem number_of_handshakes :
  total_handshakes = 24 := 
sorry

end number_of_handshakes_l306_306672


namespace minimum_turns_to_exceed_1000000_l306_306160

theorem minimum_turns_to_exceed_1000000 :
  let a : Fin 5 → ℕ := fun n => if n = 0 then 1 else 0
  (∀ n : ℕ, ∃ (b_2 b_3 b_4 b_5 : ℕ),
    a 4 + b_2 ≥ 0 ∧
    a 3 + b_3 ≥ 0 ∧
    a 2 + b_4 ≥ 0 ∧
    a 1 + b_5 ≥ 0 ∧
    b_2 * b_3 * b_4 * b_5 > 1000000 →
    b_2 + b_3 + b_4 + b_5 = n) → 
    ∃ n, n = 127 :=
by
  sorry

end minimum_turns_to_exceed_1000000_l306_306160


namespace sufficient_drivers_and_ivan_petrovich_departure_l306_306602

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l306_306602


namespace solve_z_l306_306869

noncomputable def complex_equation (z : ℂ) := (1 + 3 * Complex.I) * z = Complex.I - 3

theorem solve_z (z : ℂ) (h : complex_equation z) : z = Complex.I :=
by
  sorry

end solve_z_l306_306869


namespace loss_percentage_is_25_l306_306179

variables (C S : ℝ)
variables (h : 30 * C = 40 * S)

theorem loss_percentage_is_25 (h : 30 * C = 40 * S) : ((C - S) / C) * 100 = 25 :=
by
  -- proof skipped
  sorry

end loss_percentage_is_25_l306_306179


namespace isosceles_right_triangle_inscribed_circle_l306_306158

theorem isosceles_right_triangle_inscribed_circle
  (h r x : ℝ)
  (h_def : h = 2 * r)
  (r_def : r = Real.sqrt 2 / 4)
  (x_def : x = h - r) :
  x = Real.sqrt 2 / 4 :=
by
  sorry

end isosceles_right_triangle_inscribed_circle_l306_306158


namespace divisibility_check_l306_306701

variable (d : ℕ) (h1 : d % 2 = 1) (h2 : d % 5 ≠ 0)
variable (δ : ℕ) (h3 : ∃ m : ℕ, 10 * δ + 1 = m * d)
variable (N : ℕ)

def last_digit (N : ℕ) : ℕ := N % 10
def remove_last_digit (N : ℕ) : ℕ := N / 10

theorem divisibility_check (h4 : ∃ N' u : ℕ, N = 10 * N' + u ∧ N = N' * 10 + u ∧ N' = remove_last_digit N ∧ u = last_digit N)
  (N' : ℕ) (u : ℕ) (N1 : ℕ) (h5 : N1 = N' - δ * u) :
  d ∣ N1 → d ∣ N := by
  sorry

end divisibility_check_l306_306701


namespace standard_parabola_with_symmetry_axis_eq_1_l306_306973

-- Define the condition that the axis of symmetry is x = 1
def axis_of_symmetry_x_eq_one (x : ℝ) : Prop :=
  x = 1

-- Define the standard equation of the parabola y^2 = -4x
def standard_parabola_eq (y x : ℝ) : Prop :=
  y^2 = -4 * x

-- Theorem: Prove that given the axis of symmetry of the parabola is x = 1,
-- the standard equation of the parabola is y^2 = -4x.
theorem standard_parabola_with_symmetry_axis_eq_1 : ∀ (x y : ℝ),
  axis_of_symmetry_x_eq_one x → standard_parabola_eq y x :=
by
  intros
  sorry

end standard_parabola_with_symmetry_axis_eq_1_l306_306973


namespace sum_angles_triangle_complement_l306_306447

theorem sum_angles_triangle_complement (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 180 - C = 130) : A + B = 130 :=
by
  have hC : C = 50 := by linarith
  linarith

end sum_angles_triangle_complement_l306_306447


namespace sums_solved_correctly_l306_306509

theorem sums_solved_correctly (x : ℕ) (h : x + 2 * x = 48) : x = 16 := by
  sorry

end sums_solved_correctly_l306_306509


namespace solve_equation_l306_306585

theorem solve_equation :
  ∀ (x : ℝ), 
    x^3 + (Real.log 25 + Real.log 32 + Real.log 53) * x = (Real.log 23 + Real.log 35 + Real.log 52) * x^2 + 1 ↔ 
    x = Real.log 23 ∨ x = Real.log 35 ∨ x = Real.log 52 :=
by
  sorry

end solve_equation_l306_306585


namespace percentage_of_students_on_trip_l306_306145

-- Define the problem context
variable (total_students : ℕ)
variable (students_more_100 : ℕ)
variable (students_on_trip : ℕ)

-- Define the conditions as per the problem
def condition_1 : Prop := students_more_100 = total_students * 15 / 100
def condition_2 : Prop := students_more_100 = students_on_trip * 25 / 100

-- Define the problem statement
theorem percentage_of_students_on_trip
  (h1 : condition_1 total_students students_more_100)
  (h2 : condition_2 students_more_100 students_on_trip) :
  students_on_trip = total_students * 60 / 100 :=
by
  sorry

end percentage_of_students_on_trip_l306_306145


namespace quadratic_equation_from_absolute_value_l306_306685

theorem quadratic_equation_from_absolute_value :
  ∃ b c : ℝ, (∀ x : ℝ, |x - 8| = 3 ↔ x^2 + b * x + c = 0) ∧ (b, c) = (-16, 55) :=
sorry

end quadratic_equation_from_absolute_value_l306_306685


namespace sarah_shaded_area_l306_306899

theorem sarah_shaded_area (r : ℝ) (A_square : ℝ) (A_circle : ℝ) (A_circles : ℝ) (A_shaded : ℝ) :
  let side_length := 27
  let radius := side_length / (3 * 2)
  let area_square := side_length * side_length
  let area_one_circle := Real.pi * (radius * radius)
  let total_area_circles := 9 * area_one_circle
  let shaded_area := area_square - total_area_circles
  shaded_area = 729 - 182.25 * Real.pi := 
by
  sorry

end sarah_shaded_area_l306_306899


namespace integer_points_count_l306_306811

theorem integer_points_count :
  ∃ (n : ℤ), n = 9 ∧
  ∀ a b : ℝ, (1 < a) → (1 < b) → (ab + a - b - 10 = 0) →
  (a + b = 6) → 
  ∃ (x y : ℤ), (3 * x^2 + 2 * y^2 ≤ 6) :=
by
  sorry

end integer_points_count_l306_306811


namespace tan_half_alpha_l306_306538

theorem tan_half_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 24 / 25) : Real.tan (α / 2) = 3 / 4 :=
by
  sorry

end tan_half_alpha_l306_306538


namespace max_side_length_is_11_l306_306021

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306021


namespace gain_percent_is_correct_l306_306137

theorem gain_percent_is_correct (gain_in_paise : ℝ) (cost_price_in_rs : ℝ) (conversion_factor : ℝ)
  (gain_percent_formula : ∀ (gain : ℝ) (cost : ℝ), ℝ) : 
  gain_percent_formula (gain_in_paise / conversion_factor) cost_price_in_rs = 1 :=
by
  let gain := gain_in_paise / conversion_factor
  let cost := cost_price_in_rs
  have h : gain_percent_formula gain cost = (gain / cost) * 100 := sorry
  have h2 : gain_percent_formula (70 / 100) 70 = 1 := sorry
  exact h2

end gain_percent_is_correct_l306_306137


namespace tickets_spent_l306_306514

theorem tickets_spent (initial_tickets : ℕ) (tickets_left : ℕ) (tickets_spent : ℕ) 
  (h1 : initial_tickets = 11) (h2 : tickets_left = 8) : tickets_spent = 3 :=
by
  sorry

end tickets_spent_l306_306514


namespace fraction_equivalent_of_repeating_decimal_l306_306388

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l306_306388


namespace sin_neg_1290_l306_306515

theorem sin_neg_1290 : Real.sin (-(1290 : ℝ) * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_neg_1290_l306_306515


namespace max_side_of_triangle_l306_306052

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306052


namespace compound_interest_rate_l306_306783

theorem compound_interest_rate
(SI : ℝ) (CI : ℝ) (P1 : ℝ) (r : ℝ) (t1 t2 : ℕ) (P2 R : ℝ)
(h1 : SI = (P1 * r * t1) / 100)
(h2 : SI = CI / 2)
(h3 : CI = P2 * (1 + R / 100) ^ t2 - P2)
(h4 : P1 = 3500)
(h5 : r = 6)
(h6 : t1 = 2)
(h7 : P2 = 4000)
(h8 : t2 = 2) : R = 10 := by
  sorry

end compound_interest_rate_l306_306783


namespace dvaneft_percentage_bounds_l306_306807

theorem dvaneft_percentage_bounds (x y z : ℝ) (n m : ℕ) 
  (h1 : x * n + y * m = z * (m + n))
  (h2 : 3 * x * n = y * m)
  (h3_1 : 10 ≤ y - x)
  (h3_2 : y - x ≤ 18)
  (h4_1 : 18 ≤ z)
  (h4_2 : z ≤ 42)
  : (15 ≤ (n:ℝ) / (2 * (n + m)) * 100) ∧ ((n:ℝ) / (2 * (n + m)) * 100 ≤ 25) :=
by
  sorry

end dvaneft_percentage_bounds_l306_306807


namespace opposite_of_neg_three_l306_306920

def opposite (x : Int) : Int := -x

theorem opposite_of_neg_three : opposite (-3) = 3 := by
  -- To be proven using Lean
  sorry

end opposite_of_neg_three_l306_306920


namespace max_triangle_side_24_l306_306087

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306087


namespace max_triangle_side_24_l306_306089

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306089


namespace simplify_expr_l306_306308

theorem simplify_expr (x : ℝ) : 
  2 * x * (4 * x ^ 3 - 3 * x + 1) - 7 * (x ^ 3 - x ^ 2 + 3 * x - 4) = 
  8 * x ^ 4 - 7 * x ^ 3 + x ^ 2 - 19 * x + 28 := 
by
  sorry

end simplify_expr_l306_306308


namespace necessary_but_not_sufficient_l306_306282

variable {a b c : ℝ}

theorem necessary_but_not_sufficient (h1 : b^2 - 4 * a * c ≥ 0) (h2 : a * c > 0) (h3 : a * b < 0) : 
  ¬∀ r1 r2 : ℝ, (r1 = (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) ∧ (r2 = (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) → r1 > 0 ∧ r2 > 0 :=
sorry

end necessary_but_not_sufficient_l306_306282


namespace greatest_product_l306_306637

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end greatest_product_l306_306637


namespace max_side_length_l306_306005

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306005


namespace dilation_transform_l306_306311

theorem dilation_transform (z c : ℂ) (k : ℝ) (h₀ : z = 0 - 2 * complex.I) (h₁: c = 1 + 2 * complex.I) (h₂ : k = 4) :
  (z - c = k * (0 - 2 * complex.I - c)) → (z = -3 - 14 * complex.I) :=
by
  intro h
  have h₃ : z - (1 + 2 * complex.I) = 4 * ((0 - 2 * complex.I) - (1 + 2 * complex.I)), by rw [h₀, h₁, h₂],
  have h₄ : z - (1 + 2 * complex.I) = -4 - 16 * complex.I, by simp [h₃],
  have h₅ : z = -4 - 16 * complex.I + 1 + 2 * complex.I, by rwa [h₄],
  have h₆ : z = -3 - 14 * complex.I, by simp [h₅],
  exact h₆

end dilation_transform_l306_306311


namespace fraction_rep_finite_geom_series_036_l306_306394

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l306_306394


namespace number_of_sets_l306_306560

theorem number_of_sets (weight_per_rep reps total_weight : ℕ) 
  (h_weight_per_rep : weight_per_rep = 15)
  (h_reps : reps = 10)
  (h_total_weight : total_weight = 450) :
  (total_weight / (weight_per_rep * reps)) = 3 :=
by
  sorry

end number_of_sets_l306_306560


namespace find_smallest_subtract_l306_306903

-- Definitions for multiples
def is_mul_2 (n : ℕ) : Prop := 2 ∣ n
def is_mul_3 (n : ℕ) : Prop := 3 ∣ n
def is_mul_5 (n : ℕ) : Prop := 5 ∣ n

-- Statement of the problem
theorem find_smallest_subtract (x : ℕ) :
  (is_mul_2 (134 - x)) ∧ (is_mul_3 (134 - x)) ∧ (is_mul_5 (134 - x)) → x = 14 :=
by
  sorry

end find_smallest_subtract_l306_306903


namespace solve_for_x_l306_306771

theorem solve_for_x (x : ℝ) (h : (x - 6)^4 = (1 / 16)⁻¹) : x = 8 := 
by 
  sorry

end solve_for_x_l306_306771


namespace kostya_table_prime_l306_306459

theorem kostya_table_prime {n : ℕ} (hn : n > 3)
  (h : ∀ r s : ℕ, r ≥ 3 → s ≥ 3 → rs - (r + s) ≠ n) : Prime (n + 1) := 
sorry

end kostya_table_prime_l306_306459


namespace elliot_book_pages_l306_306188

theorem elliot_book_pages : 
  ∀ (initial_pages read_per_day days_in_week remaining_pages total_pages: ℕ), 
    initial_pages = 149 → 
    read_per_day = 20 → 
    days_in_week = 7 → 
    remaining_pages = 92 → 
    total_pages = initial_pages + (read_per_day * days_in_week) + remaining_pages → 
    total_pages = 381 :=
by
  intros initial_pages read_per_day days_in_week remaining_pages total_pages
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  assumption

end elliot_book_pages_l306_306188


namespace T_simplified_l306_306289

-- Define the polynomial expression T
def T (x : ℝ) : ℝ := (x-2)^4 - 4*(x-2)^3 + 6*(x-2)^2 - 4*(x-2) + 1

-- Prove that T simplifies to (x-3)^4
theorem T_simplified (x : ℝ) : T x = (x - 3)^4 := by
  sorry

end T_simplified_l306_306289


namespace sufficient_drivers_and_ivan_petrovich_departure_l306_306604

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l306_306604


namespace solve_real_equation_l306_306692

theorem solve_real_equation (x : ℝ) (h : x^4 + (3 - x)^4 = 82) : x = 2.5 ∨ x = 0.5 :=
sorry

end solve_real_equation_l306_306692


namespace cos_arithmetic_sequence_result_l306_306981

-- Define an arithmetic sequence as a function
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem cos_arithmetic_sequence_result (a d : ℝ) 
  (h : arithmetic_seq a d 1 + arithmetic_seq a d 5 + arithmetic_seq a d 9 = 8 * Real.pi) :
  Real.cos (arithmetic_seq a d 3 + arithmetic_seq a d 7) = -1 / 2 := by
  sorry

end cos_arithmetic_sequence_result_l306_306981


namespace vendor_sells_50_percent_on_first_day_l306_306667

variables (A : ℝ) (S : ℝ)

theorem vendor_sells_50_percent_on_first_day 
  (h : 0.2 * A * (1 - S) + 0.5 * A * (1 - S) * 0.8 = 0.3 * A) : S = 0.5 :=
  sorry

end vendor_sells_50_percent_on_first_day_l306_306667


namespace max_geometries_l306_306131

variables {α β : set (set ℝ)}

/-- Given two parallel planes α and β, each containing distinct points, determine the 
   maximum number of lines, planes, and triangular pyramids. -/
theorem max_geometries
  (h_parallel : α ∩ β = ∅)
  (hα : α ≠ ∅)
  (hβ : β ≠ ∅)
  (points_α : ∀ p ∈ α, ∃! q ∈ α, q ≠ p) -- 4 points in α
  (points_β : ∀ p ∈ β, ∃! q ∈ β, q ≠ p) -- 5 points in β
  (h_noncoplanar : ∀ s : finset (sets ℝ), s.card = 4 → ¬∃ p, p ∈ α ∪ β ∧ p ∈ s)
  (h_noncollinear : ∀ s : finset (sets ℝ), s.card = 3 → ¬∃ p, p ∈ α ∪ β ∧ p ∈ s) :
  (nat.choose 9 2 = 36) ∧
  ((nat.choose 4 1) * (nat.choose 5 2) + (nat.choose 4 2) * (nat.choose 5 1) + 2 = 72) ∧
  ((nat.choose 4 3) * (nat.choose 5 1) + (nat.choose 4 2) * (nat.choose 5 2) + (nat.choose 4 1) * (nat.choose 5 3) = 120)
:= sorry

end max_geometries_l306_306131


namespace perfect_square_of_d_l306_306170

theorem perfect_square_of_d (a b c d : ℤ) (h : d = (a + (2:ℝ)^(1/3) * b + (4:ℝ)^(1/3) * c)^2) : ∃ k : ℤ, d = k^2 :=
by
  sorry

end perfect_square_of_d_l306_306170


namespace sin_600_eq_neg_sqrt_3_div_2_l306_306175

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- proof to be provided here
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l306_306175


namespace soybeans_in_jar_l306_306957

theorem soybeans_in_jar
  (totalRedBeans : ℕ)
  (sampleSize : ℕ)
  (sampleRedBeans : ℕ)
  (totalBeans : ℕ)
  (proportion : sampleRedBeans / sampleSize = totalRedBeans / totalBeans)
  (h1 : totalRedBeans = 200)
  (h2 : sampleSize = 60)
  (h3 : sampleRedBeans = 5) :
  totalBeans = 2400 :=
by
  sorry

end soybeans_in_jar_l306_306957


namespace max_side_of_triangle_l306_306061

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306061


namespace max_side_length_of_triangle_l306_306064

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306064


namespace quadratic_roots_l306_306912

theorem quadratic_roots (a b c : ℝ) :
  (∀ (x y : ℝ), ((x, y) = (-2, 12) ∨ (x, y) = (0, -8) ∨ (x, y) = (1, -12) ∨ (x, y) = (3, -8)) → y = a * x^2 + b * x + c) →
  (a * 0^2 + b * 0 + c + 8 = 0) ∧ (a * 3^2 + b * 3 + c + 8 = 0) :=
by sorry

end quadratic_roots_l306_306912


namespace casey_correct_result_l306_306519

variable (x : ℕ)

def incorrect_divide (x : ℕ) := x / 7
def incorrect_subtract (x : ℕ) := x - 20
def incorrect_result := 19

def reverse_subtract (x : ℕ) := x + 20
def reverse_divide (x : ℕ) := x * 7

def correct_multiply (x : ℕ) := x * 7
def correct_add (x : ℕ) := x + 20

theorem casey_correct_result (x : ℕ) (h : reverse_divide (reverse_subtract incorrect_result) = x) : correct_add (correct_multiply x) = 1931 :=
by
  sorry

end casey_correct_result_l306_306519


namespace inequality_proof_l306_306431

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l306_306431


namespace min_value_l306_306855

noncomputable def min_res (a b c : ℝ) : ℝ := 
  if h : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
  then (1 / a + 2 / b + 3 / c) 
  else 0

theorem min_value (a b c : ℝ) : (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + 4 * b^2 + 9 * c^2 = 4 * b + 12 * c - 2) 
    → min_res a b c = 6 := 
sorry

end min_value_l306_306855


namespace price_per_slice_is_five_l306_306958

-- Definitions based on the given conditions
def pies_sold := 9
def slices_per_pie := 4
def total_revenue := 180

-- Definition derived from given conditions
def total_slices := pies_sold * slices_per_pie

-- The theorem to prove
theorem price_per_slice_is_five :
  total_revenue / total_slices = 5 :=
by
  sorry

end price_per_slice_is_five_l306_306958


namespace perimeter_square_C_l306_306839

theorem perimeter_square_C (pA pB pC : ℕ) (hA : pA = 16) (hB : pB = 32) (hC : pC = (pA + pB) / 2) : pC = 24 := by
  sorry

end perimeter_square_C_l306_306839


namespace quadratic_function_vertex_and_comparison_l306_306541

theorem quadratic_function_vertex_and_comparison
  (a b c : ℝ)
  (A_conds : 4 * a - 2 * b + c = 9)
  (B_conds : c = 3)
  (C_conds : 16 * a + 4 * b + c = 3) :
  (a = 1/2 ∧ b = -2 ∧ c = 3) ∧
  (∀ (m : ℝ) (y₁ y₂ : ℝ),
     y₁ = 1/2 * m^2 - 2 * m + 3 ∧
     y₂ = 1/2 * (m + 1)^2 - 2 * (m + 1) + 3 →
     (m < 3/2 → y₁ > y₂) ∧
     (m = 3/2 → y₁ = y₂) ∧
     (m > 3/2 → y₁ < y₂)) :=
by
  sorry

end quadratic_function_vertex_and_comparison_l306_306541


namespace symmetric_point_l306_306205

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def plane_eq (M : Point3D) : Prop :=
  2 * M.x - 4 * M.y - 4 * M.z - 13 = 0

-- Given Point M
def M : Point3D := { x := 3, y := -3, z := -1 }

-- Symmetric Point M'
def M' : Point3D := { x := 2, y := -1, z := 1 }

theorem symmetric_point (H : plane_eq M) : plane_eq M' ∧ 
  (M'.x = 2 * (3 + 2 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.x) ∧ 
  (M'.y = 2 * (-3 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.y) ∧ 
  (M'.z = 2 * (-1 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.z) :=
sorry

end symmetric_point_l306_306205


namespace yearly_production_target_l306_306801

-- Definitions for the conditions
def p_current : ℕ := 100
def p_add : ℕ := 50

-- The theorem to be proven
theorem yearly_production_target : (p_current + p_add) * 12 = 1800 := by
  sorry  -- Proof is omitted

end yearly_production_target_l306_306801


namespace ak_squared_eq_kl_km_l306_306767

-- Define the problem in Lean
variables {A B C D K L M : Type} [AddCommGroup A] [AffineSpace A]

-- Define A, B, C, D as points forming a parallelogram
def is_parallelogram (A B C D : A) : Prop :=
  vector.to_param (A, B, C, D) ≠ 0 ∧
  dist A B = dist C D ∧ 
  dist A D = dist B C ∧
  dist A B * dist A D = dist A C * dist A D

-- Define a point K on diagonal BD
def on_diagonal (B D K : A) : Prop :=
  K ∈ line_through B D

-- Define point L on CD and M on BC
def intersects (A K C D B L M : A) : Prop :=
  K ∈ line_through A D ∧ L ∈ line_join C D ∧ M ∈ line_join B C

theorem ak_squared_eq_kl_km 
  (A B C D K L M : A) 
  [AddCommGroup A] [AffineSpace A]
  (hParallelogram : is_parallelogram A B C D)
  (hOnDiagonal : on_diagonal B D K)
  (hIntersects : intersects A K C D B L M) :
  dist A K ^ 2 = dist K L * dist K M :=
sorry

end ak_squared_eq_kl_km_l306_306767


namespace bisection_second_iteration_value_l306_306455

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_second_iteration_value :
  f 0.25 = -0.234375 :=
by
  -- The proof steps would go here
  sorry

end bisection_second_iteration_value_l306_306455


namespace price_reduction_correct_l306_306343

noncomputable def percentage_reduction (x : ℝ) : Prop :=
  (5000 * (1 - x)^2 = 4050)

theorem price_reduction_correct {x : ℝ} (h : percentage_reduction x) : x = 0.1 :=
by
  -- proof is omitted, so we use sorry
  sorry

end price_reduction_correct_l306_306343


namespace smallest_number_of_marbles_l306_306495

theorem smallest_number_of_marbles (M : ℕ) (h1 : M ≡ 2 [MOD 5]) (h2 : M ≡ 2 [MOD 6]) (h3 : M ≡ 2 [MOD 7]) (h4 : 1 < M) : M = 212 :=
by sorry

end smallest_number_of_marbles_l306_306495


namespace grandmother_total_payment_l306_306361

theorem grandmother_total_payment
  (senior_discount : Real := 0.30)
  (children_discount : Real := 0.40)
  (num_seniors : Nat := 2)
  (num_children : Nat := 2)
  (num_regular : Nat := 2)
  (senior_ticket_price : Real := 7.50)
  (regular_ticket_price : Real := senior_ticket_price / (1 - senior_discount))
  (children_ticket_price : Real := regular_ticket_price * (1 - children_discount))
  : (num_seniors * senior_ticket_price + num_regular * regular_ticket_price + num_children * children_ticket_price) = 49.27 := 
by
  sorry

end grandmother_total_payment_l306_306361


namespace salad_dressing_vinegar_percentage_l306_306578

-- Define the initial conditions
def percentage_in_vinegar_in_Q : ℝ := 10
def percentage_of_vinegar_in_combined : ℝ := 12
def percentage_of_dressing_P_in_combined : ℝ := 0.10
def percentage_of_dressing_Q_in_combined : ℝ := 0.90
def percentage_of_vinegar_in_P (V : ℝ) : ℝ := V

-- The statement to prove
theorem salad_dressing_vinegar_percentage (V : ℝ) 
  (hQ : percentage_in_vinegar_in_Q = 10)
  (hCombined : percentage_of_vinegar_in_combined = 12)
  (hP_combined : percentage_of_dressing_P_in_combined = 0.10)
  (hQ_combined : percentage_of_dressing_Q_in_combined = 0.90)
  (hV_combined : 0.10 * percentage_of_vinegar_in_P V + 0.90 * percentage_in_vinegar_in_Q = 12) :
  V = 30 :=
by 
  sorry

end salad_dressing_vinegar_percentage_l306_306578


namespace calculate_expression_l306_306677

theorem calculate_expression : 2 * Real.sin (60 * Real.pi / 180) + (-1/2)⁻¹ + abs (2 - Real.sqrt 3) = 0 :=
by
  sorry

end calculate_expression_l306_306677


namespace coefficients_sum_l306_306274

theorem coefficients_sum :
  let A := 3
  let B := 14
  let C := 18
  let D := 19
  let E := 30
  A + B + C + D + E = 84 := by
  sorry

end coefficients_sum_l306_306274


namespace ribbon_fraction_per_box_l306_306286

theorem ribbon_fraction_per_box 
  (total_ribbon_used : ℚ)
  (number_of_boxes : ℕ)
  (h1 : total_ribbon_used = 5/8)
  (h2 : number_of_boxes = 5) :
  (total_ribbon_used / number_of_boxes = 1/8) :=
by
  sorry

end ribbon_fraction_per_box_l306_306286


namespace f_x_plus_1_even_f_x_plus_3_odd_l306_306883

variable (R : Type) [CommRing R]

variable (f : R → R)

-- Conditions
axiom condition1 : ∀ x : R, f (1 + x) = f (1 - x)
axiom condition2 : ∀ x : R, f (x - 2) + f (-x) = 0

-- Prove that f(x + 1) is an even function
theorem f_x_plus_1_even (x : R) : f (x + 1) = f (-(x + 1)) :=
by sorry

-- Prove that f(x + 3) is an odd function
theorem f_x_plus_3_odd (x : R) : f (x + 3) = - f (-(x + 3)) :=
by sorry

end f_x_plus_1_even_f_x_plus_3_odd_l306_306883


namespace part1_part2_l306_306268
open Real

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (hm : m > 1) : ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by
  sorry

end part1_part2_l306_306268


namespace reasoning_classification_correct_l306_306532

def analogical_reasoning := "reasoning from specific to specific"
def inductive_reasoning := "reasoning from part to whole and from individual to general"
def deductive_reasoning := "reasoning from general to specific"

theorem reasoning_classification_correct : 
  (analogical_reasoning, inductive_reasoning, deductive_reasoning) =
  ("reasoning from specific to specific", "reasoning from part to whole and from individual to general", "reasoning from general to specific") := 
by 
  sorry

end reasoning_classification_correct_l306_306532


namespace repeating_decimal_fraction_eq_l306_306378

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l306_306378


namespace max_side_length_of_triangle_l306_306075

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306075


namespace total_number_of_soccer_games_l306_306615

theorem total_number_of_soccer_games (teams : ℕ)
  (regular_games_per_team : ℕ)
  (promotional_games_per_team : ℕ)
  (h1 : teams = 15)
  (h2 : regular_games_per_team = 14)
  (h3 : promotional_games_per_team = 2) :
  ((teams * regular_games_per_team) / 2 + (teams * promotional_games_per_team) / 2) = 120 :=
by
  sorry

end total_number_of_soccer_games_l306_306615


namespace find_erased_number_l306_306358

theorem find_erased_number (n x : ℕ) 
  (h1 : n > 3) 
  (h2 : (rat.ofInt ((n - 2) * (n + 3) / 2) - x) / (n - 2) = rat.ofInt 454 / 9) 
  : x = 107 :=
sorry

end find_erased_number_l306_306358


namespace range_of_a_l306_306721

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | x ≥ a }

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ≤ -2 :=
by
  sorry

end range_of_a_l306_306721


namespace a7_b7_equals_29_l306_306296

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry

def cond1 := a + b = 1
def cond2 := a^2 + b^2 = 3
def cond3 := a^3 + b^3 = 4
def cond4 := a^4 + b^4 = 7
def cond5 := a^5 + b^5 = 11

theorem a7_b7_equals_29 : cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 → a^7 + b^7 = 29 :=
by
  sorry

end a7_b7_equals_29_l306_306296


namespace opposite_of_neg_three_l306_306921

def opposite (x : Int) : Int := -x

theorem opposite_of_neg_three : opposite (-3) = 3 := by
  -- To be proven using Lean
  sorry

end opposite_of_neg_three_l306_306921


namespace max_product_of_two_integers_l306_306634

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end max_product_of_two_integers_l306_306634


namespace Ma_Xiaohu_speed_l306_306573

theorem Ma_Xiaohu_speed
  (distance_home_school : ℕ := 1800)
  (distance_to_school : ℕ := 1600)
  (father_speed_factor : ℕ := 2)
  (time_difference : ℕ := 10)
  (x : ℕ)
  (hx : distance_home_school - distance_to_school = 200)
  (hspeed : father_speed_factor * x = 2 * x)
  :
  (distance_to_school / x) - (distance_to_school / (2 * x)) = time_difference ↔ x = 80 :=
by
  sorry

end Ma_Xiaohu_speed_l306_306573


namespace log_eq_solution_l306_306584

theorem log_eq_solution (x : ℝ) (h : Real.log 8 / Real.log x = Real.log 5 / Real.log 125) : x = 512 := by
  sorry

end log_eq_solution_l306_306584


namespace repeating_decimal_fraction_equiv_l306_306385

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l306_306385


namespace union_sets_a_l306_306121

theorem union_sets_a (P S : Set ℝ) (a : ℝ) :
  P = {1, 5, 10} →
  S = {1, 3, a^2 + 1} →
  S ∪ P = {1, 3, 5, 10} →
  a = 2 ∨ a = -2 ∨ a = 3 ∨ a = -3 :=
by
  intros hP hS hUnion 
  sorry

end union_sets_a_l306_306121


namespace rectangle_ratio_l306_306450

theorem rectangle_ratio (a b c d : ℝ) (h₀ : a = 4)
  (h₁ : b = (4 / 3)) (h₂ : c = (8 / 3)) (h₃ : d = 4) :
  (∃ XY YZ, XY * YZ = a * a ∧ XY / YZ = 0.9) :=
by
  -- Proof to be filled
  sorry

end rectangle_ratio_l306_306450


namespace max_triangle_side_24_l306_306077

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306077


namespace area_of_rhombus_l306_306654

-- Defining conditions for the problem
def d1 : ℝ := 40   -- Length of the first diagonal in meters
def d2 : ℝ := 30   -- Length of the second diagonal in meters

-- Calculating the area of the rhombus
noncomputable def area : ℝ := (d1 * d2) / 2

-- Statement of the theorem
theorem area_of_rhombus : area = 600 := by
  sorry

end area_of_rhombus_l306_306654


namespace units_digit_7_pow_6_pow_5_l306_306100

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 7 := by
  -- Proof will go here
  sorry

end units_digit_7_pow_6_pow_5_l306_306100


namespace max_side_length_of_triangle_l306_306073

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306073


namespace limonia_largest_none_providable_amount_l306_306744

def is_achievable (n : ℕ) (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), x = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10)

theorem limonia_largest_none_providable_amount (n : ℕ) : 
  ∃ s, ¬ is_achievable n s ∧ (∀ t, t > s → is_achievable n t) ∧ s = 12 * n^2 + 14 * n - 1 :=
by
  sorry

end limonia_largest_none_providable_amount_l306_306744


namespace shaded_region_area_l306_306556

noncomputable def side_length := 1 -- Length of each side of the squares, in cm.

-- Conditions
def top_square_center_above_edge : Prop := 
  ∀ square1 square2 square3 : ℝ, square3 = (square1 + square2) / 2

-- Question: Area of the shaded region
def area_of_shaded_region := 1 -- area in cm^2

-- Lean 4 Statement
theorem shaded_region_area :
  top_square_center_above_edge → area_of_shaded_region = 1 := 
by
  sorry

end shaded_region_area_l306_306556


namespace f_2016_is_1_l306_306851

noncomputable def f : ℤ → ℤ := sorry

axiom h1 : f 1 = 1
axiom h2 : f 2015 ≠ 1
axiom h3 : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)
axiom h4 : ∀ x : ℤ, f x = f (-x)

theorem f_2016_is_1 : f 2016 = 1 := 
by 
  sorry

end f_2016_is_1_l306_306851


namespace no_such_prime_pair_l306_306374

open Prime

theorem no_such_prime_pair :
  ∀ (p q : ℕ), Prime p → Prime q → (p > 5) → (q > 5) →
  (p * q) ∣ ((5^p - 2^p) * (5^q - 2^q)) → false :=
by
  intros p q hp hq hp_gt5 hq_gt5 hdiv
  sorry

end no_such_prime_pair_l306_306374


namespace winning_candidate_percentage_l306_306948

theorem winning_candidate_percentage
  (votes_candidate1 : ℕ) (votes_candidate2 : ℕ) (votes_candidate3 : ℕ)
  (total_votes : ℕ) (winning_votes : ℕ) (percentage : ℚ)
  (h1 : votes_candidate1 = 1000)
  (h2 : votes_candidate2 = 2000)
  (h3 : votes_candidate3 = 4000)
  (h4 : total_votes = votes_candidate1 + votes_candidate2 + votes_candidate3)
  (h5 : winning_votes = votes_candidate3)
  (h6 : percentage = (winning_votes : ℚ) / total_votes * 100) :
  percentage = 57.14 := 
sorry

end winning_candidate_percentage_l306_306948


namespace two_results_l306_306609

constant duration_one_way : Nat := 160 -- minutes, converted from 2 hours 40 minutes
constant duration_round_trip : Nat := 320 -- minutes
constant rest_time : Nat := 60 -- minutes

constant time_to_minutes : String → Nat
constant next_trip_time : Nat → Nat → Nat

axiom driverA_returns : String := "12:40"
axiom driverA_next_trip: Nat := next_trip_time (time_to_minutes "12:40") rest_time
axiom driverD_departure: Nat := time_to_minutes "13:05"

axiom driverA_fifth_trip : Nat := time_to_minutes "16:10"
axiom driverB_return_sixth_trip : Nat := time_to_minutes "16:00"
axiom driverB_sixth_trip_departure : Nat := next_trip_time (time_to_minutes "16:00") rest_time

axiom fifth_trip_done_by_A : Nat := time_to_minutes "16:10"
axiom last_trip_done_by_B : Nat := time_to_minutes "17:30"

noncomputable def required_drivers : Nat := 4
noncomputable def ivan_petrovich_departure : String := "10:40"

theorem two_results :
  required_drivers = 4 ∧
  ivan_petrovich_departure = "10:40" :=
by
  sorry

end two_results_l306_306609


namespace blocks_eaten_correct_l306_306174

def initial_blocks : ℕ := 55
def remaining_blocks : ℕ := 26

-- How many blocks were eaten by the hippopotamus?
def blocks_eaten_by_hippopotamus : ℕ := initial_blocks - remaining_blocks

theorem blocks_eaten_correct :
  blocks_eaten_by_hippopotamus = 29 := by
  sorry

end blocks_eaten_correct_l306_306174


namespace incorrect_propositions_l306_306833

theorem incorrect_propositions :
  ¬ (∀ P : Prop, P → P) ∨
  (¬ (∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x : ℝ, x^2 - x > 0)) ∨
  (∀ (R : Type) (f : R → Prop), (∀ r, f r → ∃ r', f r') = ∃ r, f r ∧ ∃ r', f r') ∨
  (∀ (x : ℝ), x ≠ 3 → abs x = 3 → x = 3) :=
by sorry

end incorrect_propositions_l306_306833


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306031

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306031


namespace fraction_repeating_decimal_l306_306376

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l306_306376


namespace same_days_to_dig_scenario_l306_306499

def volume (depth length breadth : ℝ) : ℝ :=
  depth * length * breadth

def days_to_dig (depth length breadth days : ℝ) : Prop :=
  ∃ (labors : ℝ), 
    (volume depth length breadth) * days = (volume 100 25 30) * 12

theorem same_days_to_dig_scenario :
  days_to_dig 75 20 50 12 :=
sorry

end same_days_to_dig_scenario_l306_306499


namespace fruit_salad_mixture_l306_306955

theorem fruit_salad_mixture :
  ∃ (A P G : ℝ), A / P = 12 / 8 ∧ A / G = 12 / 7 ∧ P / G = 8 / 7 ∧ A = G + 10 ∧ A + P + G = 54 :=
by
  sorry

end fruit_salad_mixture_l306_306955


namespace combined_weight_of_emma_and_henry_l306_306838

variables (e f g h : ℕ)

theorem combined_weight_of_emma_and_henry 
  (h1 : e + f = 310)
  (h2 : f + g = 265)
  (h3 : g + h = 280) : e + h = 325 :=
by
  sorry

end combined_weight_of_emma_and_henry_l306_306838


namespace stella_toilet_paper_packs_l306_306299

-- Define the relevant constants/conditions
def rolls_per_bathroom_per_day : Nat := 1
def number_of_bathrooms : Nat := 6
def days_per_week : Nat := 7
def weeks : Nat := 4
def rolls_per_pack : Nat := 12

-- Theorem statement
theorem stella_toilet_paper_packs :
  (rolls_per_bathroom_per_day * number_of_bathrooms * days_per_week * weeks) / rolls_per_pack = 14 :=
by
  sorry

end stella_toilet_paper_packs_l306_306299


namespace max_side_of_triangle_l306_306045

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306045


namespace harold_betty_choose_3_common_l306_306892

open BigOperators

-- Definition of combinations
def comb (n k : ℕ) : ℕ := nat.choose n k

-- The total number of ways each can choose 6 books from 12 books
def total_choices : ℕ := comb 12 6 * comb 12 6

-- The number of successful outcomes where exactly 3 books are the same
def successful_outcomes : ℕ := comb 12 3 * comb 9 3 * comb 9 3

-- The probability Harold and Betty choose exactly 3 books in common
def probability (total successful : ℕ) : ℚ := successful / total

theorem harold_betty_choose_3_common :
  probability total_choices successful_outcomes = 220 / 1215 :=
by
  rw [total_choices, successful_outcomes, probability]
  exact sorry

end harold_betty_choose_3_common_l306_306892


namespace no_friendly_triplet_in_range_l306_306193

open Nat

def isFriendly (a b c : ℕ) : Prop :=
  (a ∣ (b * c) ∨ b ∣ (a * c) ∨ c ∣ (a * b))

theorem no_friendly_triplet_in_range (n : ℕ) (a b c : ℕ) :
  n^2 < a ∧ a < n^2 + n → n^2 < b ∧ b < n^2 + n → n^2 < c ∧ c < n^2 + n → a ≠ b → b ≠ c → a ≠ c →
  ¬ isFriendly a b c :=
by sorry

end no_friendly_triplet_in_range_l306_306193


namespace percentage_of_copper_in_second_alloy_l306_306965

theorem percentage_of_copper_in_second_alloy
  (w₁ w₂ w_total : ℝ)
  (p₁ p_total : ℝ)
  (h₁ : w₁ = 66)
  (h₂ : p₁ = 0.10)
  (h₃ : w_total = 121)
  (h₄ : p_total = 0.15) :
  (w_total - w₁) * 0.21 = w_total * p_total - w₁ * p₁ := 
  sorry

end percentage_of_copper_in_second_alloy_l306_306965


namespace two_results_l306_306610

constant duration_one_way : Nat := 160 -- minutes, converted from 2 hours 40 minutes
constant duration_round_trip : Nat := 320 -- minutes
constant rest_time : Nat := 60 -- minutes

constant time_to_minutes : String → Nat
constant next_trip_time : Nat → Nat → Nat

axiom driverA_returns : String := "12:40"
axiom driverA_next_trip: Nat := next_trip_time (time_to_minutes "12:40") rest_time
axiom driverD_departure: Nat := time_to_minutes "13:05"

axiom driverA_fifth_trip : Nat := time_to_minutes "16:10"
axiom driverB_return_sixth_trip : Nat := time_to_minutes "16:00"
axiom driverB_sixth_trip_departure : Nat := next_trip_time (time_to_minutes "16:00") rest_time

axiom fifth_trip_done_by_A : Nat := time_to_minutes "16:10"
axiom last_trip_done_by_B : Nat := time_to_minutes "17:30"

noncomputable def required_drivers : Nat := 4
noncomputable def ivan_petrovich_departure : String := "10:40"

theorem two_results :
  required_drivers = 4 ∧
  ivan_petrovich_departure = "10:40" :=
by
  sorry

end two_results_l306_306610


namespace nancy_weight_l306_306846

theorem nancy_weight (w : ℕ) (h : (60 * w) / 100 = 54) : w = 90 :=
by
  sorry

end nancy_weight_l306_306846


namespace solve_equation_l306_306320

theorem solve_equation : ∀ x : ℝ, (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1) → x = 5 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l306_306320


namespace condition_a_condition_b_condition_c_l306_306864

-- Definitions for conditions
variable {ι : Type*} (f₁ f₂ f₃ f₄ : ι → ℝ) (x : ι)

-- First part: Condition to prove second equation is a consequence of first
theorem condition_a :
  (∀ x, f₁ x * f₄ x = f₂ x * f₃ x) →
  ((f₂ x ≠ 0) ∧ (f₂ x + f₄ x ≠ 0)) →
  (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) :=
sorry

-- Second part: Condition to prove first equation is a consequence of second
theorem condition_b :
  (∀ x, f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) →
  ((f₄ x ≠ 0) ∧ (f₂ x ≠ 0)) →
  (f₁ x * f₄ x = f₂ x * f₃ x) :=
sorry

-- Third part: Condition for equivalence of the equations
theorem condition_c :
  (∀ x, (f₁ x * f₄ x = f₂ x * f₃ x) ∧ (x ∉ {x | f₂ x + f₄ x = 0})) ↔
  (∀ x, (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) ∧ (x ∉ {x | f₄ x = 0})) :=
sorry

end condition_a_condition_b_condition_c_l306_306864


namespace vector_dot_product_zero_implies_orthogonal_l306_306228

theorem vector_dot_product_zero_implies_orthogonal
  (a b : ℝ → ℝ)
  (h0 : ∀ (x y : ℝ), a x * b y = 0) :
  ¬(a = 0 ∨ b = 0) := 
sorry

end vector_dot_product_zero_implies_orthogonal_l306_306228


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306032

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306032


namespace problem1_problem2_problem3_l306_306582

theorem problem1 : 999 * 999 + 1999 = 1000000 := by
  sorry

theorem problem2 : 9 * 72 * 125 = 81000 := by
  sorry

theorem problem3 : 416 - 327 + 184 - 273 = 0 := by
  sorry

end problem1_problem2_problem3_l306_306582


namespace proof_problem_l306_306516

noncomputable def problem_expression : ℝ :=
  50 * 39.96 * 3.996 * 500

theorem proof_problem : problem_expression = (3996 : ℝ)^2 :=
by
  sorry

end proof_problem_l306_306516


namespace quadratic_solution_l306_306922

theorem quadratic_solution
  (a c : ℝ) (h : a ≠ 0) (h_passes_through : ∃ b, b = c - 9 * a) :
  ∀ (x : ℝ), (ax^2 - 2 * a * x + c = 0) ↔ (x = -1) ∨ (x = 3) :=
by
  sorry

end quadratic_solution_l306_306922


namespace original_number_is_7_l306_306107

theorem original_number_is_7 (N : ℕ) (h : ∃ (k : ℤ), N = 12 * k + 7) : N = 7 :=
sorry

end original_number_is_7_l306_306107


namespace find_replacement_percentage_l306_306895

noncomputable def final_percentage_replacement_alcohol_solution (a₁ p₁ p₂ x : ℝ) : Prop :=
  let d := 0.4 -- gallons
  let final_solution := 1 -- gallon
  let initial_pure_alcohol := a₁ * p₁ / 100
  let remaining_pure_alcohol := initial_pure_alcohol - (d * p₁ / 100)
  let added_pure_alcohol := d * x / 100
  remaining_pure_alcohol + added_pure_alcohol = final_solution * p₂ / 100

theorem find_replacement_percentage :
  final_percentage_replacement_alcohol_solution 1 75 65 50 :=
by
  sorry

end find_replacement_percentage_l306_306895


namespace find_integer_n_l306_306502

def s : List ℤ := [8, 11, 12, 14, 15]

theorem find_integer_n (n : ℤ) (h : (s.sum + n) / (s.length + 1) = (25 / 100) * (s.sum / s.length) + (s.sum / s.length)) : n = 30 := by
  sorry

end find_integer_n_l306_306502


namespace repeating_decimal_fraction_eq_l306_306380

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l306_306380


namespace max_value_condition_l306_306708

noncomputable def f (a x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ a then
  Real.log x
else
  if x > a then
    a / x
  else
    0 -- This case should not happen given the domain conditions

theorem max_value_condition (a : ℝ) : 
  (∃ M, ∀ x > 0, x ≤ a → f a x ≤ M) ∧ (∀ x > a, f a x ≤ M) ↔ a ≥ Real.exp 1 :=
sorry

end max_value_condition_l306_306708


namespace intersection_of_A_and_B_l306_306120

open Set

theorem intersection_of_A_and_B (A B : Set ℕ) (hA : A = {1, 2, 4}) (hB : B = {2, 4, 6}) : A ∩ B = {2, 4} :=
by
  rw [hA, hB]
  apply Set.ext
  intro x
  simp
  sorry

end intersection_of_A_and_B_l306_306120


namespace christopher_sword_length_l306_306681

variable (C J U : ℤ)

def jameson_sword (C : ℤ) : ℤ := 2 * C + 3
def june_sword (J : ℤ) : ℤ := J + 5
def june_sword_christopher (C : ℤ) : ℤ := C + 23

theorem christopher_sword_length (h1 : J = jameson_sword C)
                                (h2 : U = june_sword J)
                                (h3 : U = june_sword_christopher C) :
                                C = 15 :=
by
  sorry

end christopher_sword_length_l306_306681


namespace negation_log2_property_l306_306484

theorem negation_log2_property :
  ¬(∃ x₀ : ℝ, Real.log x₀ / Real.log 2 ≤ 0) ↔ ∀ x : ℝ, Real.log x / Real.log 2 > 0 :=
by
  sorry

end negation_log2_property_l306_306484


namespace driver_schedule_l306_306607

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l306_306607


namespace sin_sum_identity_l306_306304

theorem sin_sum_identity (n : ℕ) (α : ℝ) (hα : α = π / (2^(n + 1) - 1)) :
  (∑ i in (finset.range n).map ⟨λ k, 2^(k+1), λ a b h, by simp⟩, 1 / (Real.sin (i * α))) = 1 / (Real.sin α) :=
sorry

end sin_sum_identity_l306_306304


namespace find_f_a_plus_b_plus_c_l306_306714

open Polynomial

variables {a b c p q r : ℝ}
variables (f : ℝ → ℝ)

-- Polynomial conditions
def f_poly (x : ℝ) := p * x^2 + q * x + r

-- Given distinct real numbers a, b, c
variables (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
-- Given polynomial conditions
variables (h1 : f a = b * c)
variables (h2 : f b = c * a)
variables (h3 : f c = a * b)

-- The goal statement
theorem find_f_a_plus_b_plus_c :
  f (a + b + c) = a * b + b * c + c * a := sorry

end find_f_a_plus_b_plus_c_l306_306714


namespace mia_12th_roll_last_is_approximately_027_l306_306147

noncomputable def mia_probability_last_roll_on_12th : ℚ :=
  (5/6) ^ 10 * (1/6)

theorem mia_12th_roll_last_is_approximately_027 : 
  abs (mia_probability_last_roll_on_12th - 0.027) < 0.001 :=
sorry

end mia_12th_roll_last_is_approximately_027_l306_306147


namespace shopkeeper_standard_weight_l306_306813

theorem shopkeeper_standard_weight
    (cost_price : ℝ)
    (actual_weight_used : ℝ)
    (profit_percentage : ℝ)
    (standard_weight : ℝ)
    (H1 : actual_weight_used = 800)
    (H2 : profit_percentage = 25) :
    standard_weight = 1000 :=
by 
    sorry

end shopkeeper_standard_weight_l306_306813


namespace green_fish_always_15_l306_306298

def total_fish (T : ℕ) : Prop :=
∃ (O B G : ℕ),
B = T / 2 ∧
O = B - 15 ∧
T = B + O + G ∧
G = 15

theorem green_fish_always_15 (T : ℕ) : total_fish T → ∃ G, G = 15 :=
by
  intro h
  sorry

end green_fish_always_15_l306_306298


namespace cone_volume_l306_306661

theorem cone_volume (slant_height : ℝ) (central_angle_deg : ℝ) (volume : ℝ) :
  slant_height = 1 ∧ central_angle_deg = 120 ∧ volume = (2 * Real.sqrt 2 / 81) * Real.pi →
  ∃ r h, h = Real.sqrt (slant_height^2 - r^2) ∧
    r = (1/3) ∧
    h = (2 * Real.sqrt 2 / 3) ∧
    volume = (1/3) * Real.pi * r^2 * h := 
by
  sorry

end cone_volume_l306_306661


namespace find_income_l306_306916

noncomputable def income_expenditure_proof : Prop := 
  ∃ (x : ℕ), (5 * x - 4 * x = 3600) ∧ (5 * x = 18000)

theorem find_income : income_expenditure_proof :=
  sorry

end find_income_l306_306916


namespace sum_of_three_consecutive_integers_is_21_l306_306496

theorem sum_of_three_consecutive_integers_is_21 (n : ℤ) :
    n ∈ {17, 11, 25, 21, 8} →
    (∃ a, n = a + (a + 1) + (a + 2)) →
    n = 21 :=
by
  intro h
  intro h_consec
  cases h_consec with a ha
  have sum_eq_three_a : n = 3 * a + 3 :=
    by linarith
  -- Verify that 21 is the only possible sum value.
  have h_n_values : n = 17 ∨ n = 11 ∨ n = 25 ∨ n = 21 ∨ n = 8 :=
    by simp at h; exact h
  cases h_n_values
  { rw h_n_values at sum_eq_three_a; contradiction }
  { rw h_n_values at sum_eq_three_a; contradiction }
  { rw h_n_values at sum_eq_three_a; contradiction }
  { exact h_n_values }
  { rw h_n_values at sum_eq_three_a; contradiction }
sorry

end sum_of_three_consecutive_integers_is_21_l306_306496


namespace min_value_2x_plus_y_l306_306419

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/(y + 1) = 2) : 2 * x + y = 3 :=
sorry

end min_value_2x_plus_y_l306_306419


namespace not_divisible_by_3_or_4_l306_306291

theorem not_divisible_by_3_or_4 (n : ℤ) : 
  ¬ (n^2 + 1) % 3 = 0 ∧ ¬ (n^2 + 1) % 4 = 0 := 
by
  sorry

end not_divisible_by_3_or_4_l306_306291


namespace perpendicular_vectors_l306_306153

def vector_a := (2, 0 : ℤ × ℤ)
def vector_b := (1, 1 : ℤ × ℤ)

theorem perpendicular_vectors:
  let v := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) in
  v.1 * vector_b.1 + v.2 * vector_b.2 = 0 :=
by
  sorry

end perpendicular_vectors_l306_306153


namespace incorrect_transformation_l306_306341

theorem incorrect_transformation (a b c : ℝ) (h1 : a = b) (h2 : c = 0) : ¬(a / c = b / c) :=
by
  sorry

end incorrect_transformation_l306_306341


namespace max_side_length_of_triangle_l306_306071

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306071


namespace max_product_of_sum_2024_l306_306626

theorem max_product_of_sum_2024 : 
  ∃ (x y : ℤ), x + y = 2024 ∧ x * y = 1024144 :=
by
  use 1012, 1012
  split
  · sorry
  · sorry

end max_product_of_sum_2024_l306_306626


namespace negation_of_statement_l306_306593

theorem negation_of_statement (h: ∀ x : ℝ, |x| + x^2 ≥ 0) :
  ¬ (∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end negation_of_statement_l306_306593


namespace Nancy_weighs_90_pounds_l306_306844

theorem Nancy_weighs_90_pounds (W : ℝ) (h1 : 0.60 * W = 54) : W = 90 :=
by
  sorry

end Nancy_weighs_90_pounds_l306_306844


namespace percent_of_y_eq_l306_306493

theorem percent_of_y_eq (y : ℝ) (h : y ≠ 0) : (0.3 * 0.7 * y) = (0.21 * y) := by
  sorry

end percent_of_y_eq_l306_306493


namespace nancy_weight_l306_306845

theorem nancy_weight (w : ℕ) (h : (60 * w) / 100 = 54) : w = 90 :=
by
  sorry

end nancy_weight_l306_306845


namespace toilet_paper_packs_needed_l306_306301

-- Definitions based on conditions
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def weeks : ℕ := 4
def rolls_per_pack : ℕ := 12
def daily_stock : ℕ := 1

-- The main theorem statement
theorem toilet_paper_packs_needed : 
  (bathrooms * days_per_week * weeks) / rolls_per_pack = 14 := by
sorry

end toilet_paper_packs_needed_l306_306301


namespace range_of_m_l306_306794

theorem range_of_m {m : ℝ} : (-1 ≤ 1 - m ∧ 1 - m ≤ 1) ↔ (0 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l306_306794


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306034

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306034


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306035

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306035


namespace fraction_equivalent_of_repeating_decimal_l306_306387

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l306_306387


namespace problem_l306_306119

theorem problem (x y : ℝ) (h : 2^x + 3^y = 4^x + 9^y) :
  1 < 8^x + 27^y ∧ 8^x + 27^y ≤ 2 :=
by
  sorry

end problem_l306_306119


namespace total_number_of_boys_l306_306945

-- Define the circular arrangement and the opposite positions
variable (n : ℕ)

theorem total_number_of_boys (h : (40 ≠ 10 ∧ (40 - 10) * 2 = n - 2)) : n = 62 := 
sorry

end total_number_of_boys_l306_306945


namespace parabola_distance_l306_306262

theorem parabola_distance {x y : ℝ} (h₁ : y^2 = 4 * x) (h₂ : abs (x + 2) = 5) : 
  (sqrt ((x + 1)^2 + y^2) = 4) :=
sorry

end parabola_distance_l306_306262


namespace max_side_length_is_11_l306_306012

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306012


namespace incorrect_statement_A_l306_306510

-- conditions as stated in the table
def spring_length (x : ℕ) : ℝ :=
  if x = 0 then 20
  else if x = 1 then 20.5
  else if x = 2 then 21
  else if x = 3 then 21.5
  else if x = 4 then 22
  else if x = 5 then 22.5
  else 0 -- assuming 0 for out of range for simplicity

-- questions with answers
-- Prove that statement A is incorrect
theorem incorrect_statement_A : ¬ (spring_length 0 = 20) := by
  sorry

end incorrect_statement_A_l306_306510


namespace find_x_for_g_equal_20_l306_306772

theorem find_x_for_g_equal_20 (g f : ℝ → ℝ) (h₁ : ∀ x, g x = 4 * (f⁻¹ x))
    (h₂ : ∀ x, f x = 30 / (x + 5)) :
    ∃ x, g x = 20 ∧ x = 3 := by
  sorry

end find_x_for_g_equal_20_l306_306772


namespace passing_students_this_year_l306_306553

constant initial_students : ℕ := 200 -- Initial number of students who passed three years ago
constant growth_rate : ℝ := 0.5      -- Growth rate of 50%

-- Function to calculate the number of students passing each year
def students_passing (n : ℕ) : ℕ :=
nat.rec_on n initial_students (λ n' ih, ih + (ih / 2))

-- Proposition stating the number of students passing the course this year
theorem passing_students_this_year : students_passing 3 = 675 := sorry

end passing_students_this_year_l306_306553


namespace number_of_classes_l306_306523

theorem number_of_classes (sheets_per_class_per_day : ℕ) 
                          (total_sheets_per_week : ℕ) 
                          (school_days_per_week : ℕ) 
                          (h1 : sheets_per_class_per_day = 200) 
                          (h2 : total_sheets_per_week = 9000) 
                          (h3 : school_days_per_week = 5) : 
                          total_sheets_per_week / school_days_per_week / sheets_per_class_per_day = 9 :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_classes_l306_306523


namespace trains_crossing_time_l306_306487

/-- Define the length of the first train in meters -/
def length_train1 : ℚ := 200

/-- Define the length of the second train in meters -/
def length_train2 : ℚ := 150

/-- Define the speed of the first train in kilometers per hour -/
def speed_train1_kmph : ℚ := 40

/-- Define the speed of the second train in kilometers per hour -/
def speed_train2_kmph : ℚ := 46

/-- Define conversion factor from kilometers per hour to meters per second -/
def kmph_to_mps : ℚ := 1000 / 3600

/-- Calculate the relative speed in meters per second assuming both trains are moving in the same direction -/
def relative_speed_mps : ℚ := (speed_train2_kmph - speed_train1_kmph) * kmph_to_mps

/-- Calculate the combined length of both trains in meters -/
def combined_length : ℚ := length_train1 + length_train2

/-- Prove the time in seconds for the two trains to cross each other when moving in the same direction is 210 seconds -/
theorem trains_crossing_time :
  (combined_length / relative_speed_mps) = 210 := by
  sorry

end trains_crossing_time_l306_306487


namespace find_a1_a10_value_l306_306423

variable {α : Type} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a1_a10_value (a : ℕ → α) (h1 : is_geometric_sequence a)
    (h2 : a 4 + a 7 = 2) (h3 : a 5 * a 6 = -8) : a 1 + a 10 = -7 := by
  sorry

end find_a1_a10_value_l306_306423


namespace B_joined_with_54000_l306_306950

theorem B_joined_with_54000 :
  ∀ (x : ℕ),
    (36000 * 12) / (x * 4) = 2 → x = 54000 :=
by 
  intro x h
  sorry

end B_joined_with_54000_l306_306950


namespace driver_schedule_l306_306605

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l306_306605


namespace mary_age_l306_306652

theorem mary_age :
  ∃ M R : ℕ, (R = M + 30) ∧ (R + 20 = 2 * (M + 20)) ∧ (M = 10) :=
by
  sorry

end mary_age_l306_306652


namespace distance_from_pole_to_line_l306_306369

-- Definitions based on the problem condition
def polar_equation_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = Real.sqrt 3

-- The statement of the proof problem
theorem distance_from_pole_to_line (ρ θ : ℝ) (h : polar_equation_line ρ θ) :
  ρ = Real.sqrt 6 / 2 := sorry

end distance_from_pole_to_line_l306_306369


namespace solution_set_of_inequalities_l306_306925

-- Define the conditions of the inequality system
def inequality1 (x : ℝ) : Prop := x - 2 ≥ -5
def inequality2 (x : ℝ) : Prop := 3 * x < x + 2

-- The statement to prove the solution set of the inequalities
theorem solution_set_of_inequalities :
  { x : ℝ | inequality1 x ∧ inequality2 x } = { x : ℝ | -3 ≤ x ∧ x < 1 } :=
  sorry

end solution_set_of_inequalities_l306_306925


namespace final_price_after_adjustments_l306_306476

theorem final_price_after_adjustments (p : ℝ) :
  let increased_price := p * 1.30
  let discounted_price := increased_price * 0.75
  let final_price := discounted_price * 1.10
  final_price = 1.0725 * p :=
by
  sorry

end final_price_after_adjustments_l306_306476


namespace two_results_l306_306608

constant duration_one_way : Nat := 160 -- minutes, converted from 2 hours 40 minutes
constant duration_round_trip : Nat := 320 -- minutes
constant rest_time : Nat := 60 -- minutes

constant time_to_minutes : String → Nat
constant next_trip_time : Nat → Nat → Nat

axiom driverA_returns : String := "12:40"
axiom driverA_next_trip: Nat := next_trip_time (time_to_minutes "12:40") rest_time
axiom driverD_departure: Nat := time_to_minutes "13:05"

axiom driverA_fifth_trip : Nat := time_to_minutes "16:10"
axiom driverB_return_sixth_trip : Nat := time_to_minutes "16:00"
axiom driverB_sixth_trip_departure : Nat := next_trip_time (time_to_minutes "16:00") rest_time

axiom fifth_trip_done_by_A : Nat := time_to_minutes "16:10"
axiom last_trip_done_by_B : Nat := time_to_minutes "17:30"

noncomputable def required_drivers : Nat := 4
noncomputable def ivan_petrovich_departure : String := "10:40"

theorem two_results :
  required_drivers = 4 ∧
  ivan_petrovich_departure = "10:40" :=
by
  sorry

end two_results_l306_306608


namespace grid_max_sum_l306_306370

open Finset

-- Definitions for given conditions
variables {α : Type*} [LinearOrder α] [CommSemiring α]

-- Sum of a sequence of integers.
def max_sum (a b : Fin n → α) : α :=
  ∑ i, a i * b i

-- The main statement of the problem.
theorem grid_max_sum (n : ℕ) (grid : Fin n → Fin n → Bool)
  (a b : Fin n → ℕ) (h_a : ∀ i, a i = (univ.filter (λ j, grid i j)).card)
  (h_b : ∀ j, b j = (univ.filter (λ i, ¬ grid i j)).card) :
  max_sum a b ≤ 2 * (n * (n-1) * (n+1) / 6) :=
by
  sorry

end grid_max_sum_l306_306370


namespace rotation_transform_l306_306947

theorem rotation_transform (x y α : ℝ) :
    let x' := x * Real.cos α - y * Real.sin α
    let y' := x * Real.sin α + y * Real.cos α
    (x', y') = (x * Real.cos α - y * Real.sin α, x * Real.sin α + y * Real.cos α) := by
  sorry

end rotation_transform_l306_306947


namespace max_side_of_triangle_l306_306048

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306048


namespace part_a_part_b_part_c_l306_306206

-- Given conditions and questions
variable (x y : ℝ)
variable (h : (x - y)^2 - 2 * (x + y) + 1 = 0)

-- Part (a): Prove neither x nor y can be negative
theorem part_a (h : (x - y)^2 - 2 * (x + y) + 1 = 0) : x ≥ 0 ∧ y ≥ 0 := 
sorry

-- Part (b): Prove if x > 1 and y < x, then sqrt{x} - sqrt{y} = 1
theorem part_b (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x > 1) (hy : y < x) : 
  Real.sqrt x - Real.sqrt y = 1 := 
sorry

-- Part (c): Prove if x < 1 and y < 1, then sqrt{x} + sqrt{y} = 1
theorem part_c (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x < 1) (hy : y < 1) : 
  Real.sqrt x + Real.sqrt y = 1 := 
sorry

end part_a_part_b_part_c_l306_306206


namespace total_material_weight_l306_306213

def gravel_weight : ℝ := 5.91
def sand_weight : ℝ := 8.11

theorem total_material_weight : gravel_weight + sand_weight = 14.02 := by
  sorry

end total_material_weight_l306_306213


namespace find_f_a_plus_b_plus_c_l306_306713

open Polynomial

variables {a b c p q r : ℝ}
variables (f : ℝ → ℝ)

-- Polynomial conditions
def f_poly (x : ℝ) := p * x^2 + q * x + r

-- Given distinct real numbers a, b, c
variables (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
-- Given polynomial conditions
variables (h1 : f a = b * c)
variables (h2 : f b = c * a)
variables (h3 : f c = a * b)

-- The goal statement
theorem find_f_a_plus_b_plus_c :
  f (a + b + c) = a * b + b * c + c * a := sorry

end find_f_a_plus_b_plus_c_l306_306713


namespace max_product_l306_306629

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end max_product_l306_306629


namespace final_range_a_l306_306985

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + x^2 - a * x

lemma increasing_function_range_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0) :
  a ≤ 2 * sqrt 2 :=
sorry

lemma condition_range_a (a : ℝ) (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a :=
sorry

theorem final_range_a (a : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a ∧ a ≤ 2 * sqrt 2 :=
sorry

end final_range_a_l306_306985


namespace max_side_of_triangle_l306_306062

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306062


namespace pizzas_bought_l306_306891

def slices_per_pizza := 8
def total_slices := 16

theorem pizzas_bought : total_slices / slices_per_pizza = 2 := by
  sorry

end pizzas_bought_l306_306891


namespace sarah_total_pencils_l306_306579

-- Define the number of pencils Sarah buys on each day
def pencils_monday : ℕ := 35
def pencils_tuesday : ℕ := 42
def pencils_wednesday : ℕ := 3 * pencils_tuesday
def pencils_thursday : ℕ := pencils_wednesday / 2
def pencils_friday : ℕ := 2 * pencils_monday

-- Define the total number of pencils
def total_pencils : ℕ :=
  pencils_monday + pencils_tuesday + pencils_wednesday + pencils_thursday + pencils_friday

-- Theorem statement to prove the total number of pencils equals 336
theorem sarah_total_pencils : total_pencils = 336 :=
by
  -- here goes the proof, but it is not required
  sorry

end sarah_total_pencils_l306_306579


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306033

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306033


namespace find_base_l306_306150

theorem find_base (b x y : ℝ) (h₁ : b^x * 4^y = 59049) (h₂ : x = 10) (h₃ : x - y = 10) : b = 3 :=
by
  sorry

end find_base_l306_306150


namespace students_passed_this_year_l306_306551

theorem students_passed_this_year
  (initial_students : ℕ)
  (annual_increase_rate : ℝ)
  (years_lapsed : ℕ)
  (current_students : ℕ)
  (h_initial : initial_students = 200)
  (h_rate : annual_increase_rate = 1.5)
  (h_years : years_lapsed = 3)
  (h_calc : current_students = (λ n, initial_students * (annual_increase_rate ^ n)) years_lapsed) :
  current_students = 675 :=
begin
  sorry
end

end students_passed_this_year_l306_306551


namespace yuko_in_front_of_yuri_l306_306646

theorem yuko_in_front_of_yuri (X : ℕ) (hYuri : 2 + 4 + 5 = 11) (hYuko : 1 + 5 + X > 11) : X = 6 := 
by
  sorry

end yuko_in_front_of_yuri_l306_306646


namespace number_of_matches_among_three_players_l306_306155

-- Define the given conditions
variables (n r : ℕ) -- n is the number of participants, r is the number of matches among the 3 players
variables (m : ℕ := 50) -- m is the total number of matches played

-- Given assumptions
def condition1 := m = 50
def condition2 := ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)

-- The target proof
theorem number_of_matches_among_three_players (n r : ℕ) (m : ℕ := 50)
  (h1 : m = 50)
  (h2 : ∃ (n: ℕ), 50 = Nat.choose (n-3) 2 + r + (6 - 2 * r)) :
  r = 1 :=
sorry

end number_of_matches_among_three_players_l306_306155


namespace min_text_length_l306_306558

theorem min_text_length : ∃ (L : ℕ), (∀ x : ℕ, 0.105 * (L : ℝ) < (x : ℝ) ∧ (x : ℝ) < 0.11 * (L : ℝ)) → L = 19 :=
by
  sorry

end min_text_length_l306_306558


namespace range_of_a_l306_306130

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 3 * x + a < 0) ∧ (∀ x : ℝ, 2 * x + 7 > 4 * x - 1) ∧ (∀ x : ℝ, x < 0) → a = 0 := 
by sorry

end range_of_a_l306_306130


namespace cos_thm_l306_306975

variable (θ : ℝ)

-- Conditions
def condition1 : Prop := 3 * Real.sin (2 * θ) = 4 * Real.tan θ
def condition2 : Prop := ∀ k : ℤ, θ ≠ k * Real.pi

-- Prove that cos 2θ = 1/3 given the conditions
theorem cos_thm (h1 : condition1 θ) (h2 : condition2 θ) : Real.cos (2 * θ) = 1 / 3 :=
by
  sorry

end cos_thm_l306_306975


namespace camilla_jellybeans_l306_306832

theorem camilla_jellybeans (b c : ℕ) (h1 : b = 3 * c) (h2 : b - 20 = 4 * (c - 20)) :
  b = 180 :=
by
  -- Proof steps would go here
  sorry

end camilla_jellybeans_l306_306832


namespace flowers_remaining_along_path_after_events_l306_306090

def total_flowers : ℕ := 30
def total_peonies : ℕ := 15
def total_tulips : ℕ := 15
def unwatered_flowers : ℕ := 10
def tulips_watered_by_sineglazka : ℕ := 10
def tulips_picked_by_neznaika : ℕ := 6
def remaining_flowers : ℕ := 19

theorem flowers_remaining_along_path_after_events :
  total_peonies + total_tulips = total_flowers →
  tulips_watered_by_sineglazka + unwatered_flowers = total_flowers →
  tulips_picked_by_neznaika ≤ total_tulips →
  remaining_flowers = 19 := sorry

end flowers_remaining_along_path_after_events_l306_306090


namespace clothing_value_is_correct_l306_306586

-- Define the value of the clothing to be C and the correct answer
def value_of_clothing (C : ℝ) : Prop :=
  (C + 2) = (7 / 12) * (C + 10)

-- Statement of the problem
theorem clothing_value_is_correct :
  ∃ (C : ℝ), value_of_clothing C ∧ C = 46 / 5 :=
by {
  sorry
}

end clothing_value_is_correct_l306_306586


namespace degree_of_resulting_poly_l306_306372

-- Define the polynomials involved in the problem
noncomputable def poly_1 : Polynomial ℝ := 3 * Polynomial.X ^ 5 + 2 * Polynomial.X ^ 3 - Polynomial.X - 16
noncomputable def poly_2 : Polynomial ℝ := 4 * Polynomial.X ^ 11 - 8 * Polynomial.X ^ 8 + 6 * Polynomial.X ^ 5 + 35
noncomputable def poly_3 : Polynomial ℝ := (Polynomial.X ^ 2 + 4) ^ 8

-- Define the resulting polynomial
noncomputable def resulting_poly : Polynomial ℝ :=
  poly_1 * poly_2 - poly_3

-- The goal is to prove that the degree of the resulting polynomial is 16
theorem degree_of_resulting_poly : resulting_poly.degree = 16 := 
sorry

end degree_of_resulting_poly_l306_306372


namespace greatest_product_l306_306635

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end greatest_product_l306_306635


namespace percent_equivalence_l306_306491

theorem percent_equivalence (y : ℝ) (h : y ≠ 0) : 0.21 * y = 0.21 * y :=
by sorry

end percent_equivalence_l306_306491


namespace range_of_m_l306_306994

theorem range_of_m (x m : ℝ) (h1 : x + 3 = 3 * x - m) (h2 : x ≥ 0) : m ≥ -3 := by
  sorry

end range_of_m_l306_306994


namespace middle_elementary_students_l306_306154

theorem middle_elementary_students (S S_PS S_MS S_MR : ℕ) 
  (h1 : S = 12000)
  (h2 : S_PS = (15 * S) / 16)
  (h3 : S_MS = S - S_PS)
  (h4 : S_MR + S_MS = (S_PS) / 2) : 
  S_MR = 4875 :=
by
  sorry

end middle_elementary_students_l306_306154


namespace remainder_when_xyz_divided_by_9_is_0_l306_306726

theorem remainder_when_xyz_divided_by_9_is_0
  (x y z : ℕ)
  (hx : x < 9)
  (hy : y < 9)
  (hz : z < 9)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z) % 9 = 0 := by
  sorry

end remainder_when_xyz_divided_by_9_is_0_l306_306726


namespace problem_correct_l306_306363

def decimal_to_fraction_eq_80_5 : Prop :=
  ( (0.5 + 0.25 + 0.125) / (0.5 * 0.25 * 0.125) * ((7 / 18 * (9 / 2) + 1 / 6) / (13 + 1 / 3 - (15 / 4 * 16 / 5))) = 80.5 )

theorem problem_correct : decimal_to_fraction_eq_80_5 :=
  sorry

end problem_correct_l306_306363


namespace max_side_of_triangle_l306_306059

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306059


namespace birds_per_site_average_l306_306563

def total_birds : ℕ := 5 * 7 + 5 * 5 + 10 * 8 + 7 * 10 + 3 * 6 + 8 * 12 + 4 * 9
def total_sites : ℕ := 5 + 5 + 10 + 7 + 3 + 8 + 4

theorem birds_per_site_average :
  (total_birds : ℚ) / total_sites = 360 / 42 :=
by
  -- Skip proof
  sorry

end birds_per_site_average_l306_306563


namespace maximal_segment_number_l306_306414

theorem maximal_segment_number (n : ℕ) (h : n > 4) : 
  ∃ k, k = if n % 2 = 0 then 2 * n - 4 else 2 * n - 3 :=
sorry

end maximal_segment_number_l306_306414


namespace inequality_hold_l306_306427

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l306_306427


namespace thirty_percent_less_than_80_equals_one_fourth_more_l306_306618

theorem thirty_percent_less_than_80_equals_one_fourth_more (n : ℝ) :
  80 * 0.30 = 24 → 80 - 24 = 56 → n + n / 4 = 56 → n = 224 / 5 :=
by
  intros h1 h2 h3
  sorry

end thirty_percent_less_than_80_equals_one_fourth_more_l306_306618


namespace remainder_when_xyz_divided_by_9_is_0_l306_306727

theorem remainder_when_xyz_divided_by_9_is_0
  (x y z : ℕ)
  (hx : x < 9)
  (hy : y < 9)
  (hz : z < 9)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z) % 9 = 0 := by
  sorry

end remainder_when_xyz_divided_by_9_is_0_l306_306727


namespace value_of_N_l306_306817

theorem value_of_N (N : ℕ): 6 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 7.5 ↔ N = 25 ∨ N = 26 ∨ N = 27 ∨ N = 28 ∨ N = 29 := 
by
  sorry

end value_of_N_l306_306817


namespace max_triangle_side_24_l306_306088

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306088


namespace seating_arrangements_l306_306619

-- Define the conditions and the proof problem
theorem seating_arrangements (children : Finset (Fin 6)) 
  (is_sibling_pair : (Fin 6) -> (Fin 6) -> Prop)
  (no_siblings_next_to_each_other : (Fin 6) -> (Fin 6) -> Bool)
  (no_sibling_directly_in_front : (Fin 6) -> (Fin 6) -> Bool) :
  -- Statement: There are 96 valid seating arrangements
  ∃ (arrangements : Finset (Fin 6 -> Fin (2 * 3))),
  arrangements.card = 96 :=
by
  -- Proof omitted
  sorry

end seating_arrangements_l306_306619


namespace max_side_of_triangle_l306_306038

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306038


namespace students_second_scenario_l306_306276

def total_students (R : ℕ) : ℕ := 5 * R + 6
def effective_students (R : ℕ) : ℕ := 6 * (R - 3)
def filled_rows (R : ℕ) : ℕ := R - 3
def students_per_row := 6

theorem students_second_scenario:
  ∀ (R : ℕ), R = 24 → total_students R = effective_students R → students_per_row = 6
:= by
  intro R h_eq h_total_eq_effective
  -- Insert proof steps here
  sorry

end students_second_scenario_l306_306276


namespace same_points_among_teams_l306_306872

theorem same_points_among_teams :
  ∀ (n : Nat), n = 28 → 
  ∀ (G D N : Nat), G = 378 → D >= 284 → N <= 94 →
  (∃ (team_scores : Fin n → Int), ∀ (i j : Fin n), i ≠ j → team_scores i = team_scores j) := by
sorry

end same_points_among_teams_l306_306872


namespace max_side_length_of_triangle_l306_306066

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306066


namespace find_vector_result_l306_306133

-- Define the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m: ℝ) : ℝ × ℝ := (-2, m)
def m := -4
def result := 2 • vector_a + 3 • vector_b m

-- State the theorem
theorem find_vector_result : result = (-4, -8) := 
by {
  -- skipping the proof
  sorry
}

end find_vector_result_l306_306133


namespace compare_sqrts_l306_306991

theorem compare_sqrts (a b c : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) (h3 : c = 5 * Real.sqrt 2):
  c > b ∧ b > a :=
by
  sorry

end compare_sqrts_l306_306991


namespace opposite_of_neg_three_l306_306918

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l306_306918


namespace expression_value_l306_306362

theorem expression_value : 
  (Nat.factorial 10) / (2 * (Finset.sum (Finset.range 11) id)) = 33080 := by
  sorry

end expression_value_l306_306362


namespace find_f_l306_306162

noncomputable def f (x : ℝ) : ℝ :=
  let a b c : ℝ in (c / (a - b)) * x + (c / (a + b))

theorem find_f (a b c : ℝ) (x : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ a ≠ -b) :
  a * f (x - 1) + b * f (1 - x) = c * x :=
by
  sorry

end find_f_l306_306162


namespace repeating_decimal_eq_fraction_l306_306398

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l306_306398


namespace math_homework_pages_l306_306897

-- Define Rachel's total pages, math homework pages, and reading homework pages
def total_pages : ℕ := 13
def reading_homework : ℕ := sorry
def math_homework (r : ℕ) : ℕ := r + 3

-- State the main theorem that needs to be proved
theorem math_homework_pages :
  ∃ r : ℕ, r + (math_homework r) = total_pages ∧ (math_homework r) = 8 :=
by {
  sorry
}

end math_homework_pages_l306_306897


namespace number_of_handshakes_l306_306673

-- Definitions based on the conditions:
def number_of_teams : ℕ := 4
def number_of_women_per_team : ℕ := 2
def total_women : ℕ := number_of_teams * number_of_women_per_team

-- Each woman shakes hands with all others except her partner
def handshakes_per_woman : ℕ := total_women - 1 - (number_of_women_per_team - 1)

-- Calculate total handshakes, considering each handshake is counted twice
def total_handshakes : ℕ := (total_women * handshakes_per_woman) / 2

-- Statement to prove
theorem number_of_handshakes :
  total_handshakes = 24 := 
sorry

end number_of_handshakes_l306_306673


namespace evaluate_expression_l306_306527

-- Definitions for conditions
def x := (1 / 4 : ℚ)
def y := (1 / 2 : ℚ)
def z := (3 : ℚ)

-- Statement of the problem
theorem evaluate_expression : 
  4 * (x^3 * y^2 * z^2) = 9 / 64 :=
by
  sorry

end evaluate_expression_l306_306527


namespace train_length_l306_306816

theorem train_length 
  (speed_kmh : ℝ) (time_s : ℝ)
  (h_speed : speed_kmh = 50)
  (h_time : time_s = 9) : 
  let speed_ms := (speed_kmh * 1000) / 3600 in
  let length_m := speed_ms * time_s in
  length_m = 125 :=
by
  -- Speed of the train in m/s
  have speed_ms_def : speed_ms = (speed_kmh * 1000) / 3600 := by rfl
  -- Calculation of the length of the train
  have length_m_def : length_m = speed_ms * time_s := by rfl
  -- Substituting values
  rw [h_speed, h_time, speed_ms_def, length_m_def]
  -- Converting speed to m/s
  have speed_calc : (50 * 1000) / 3600 = 125 / 9 := sorry
  -- Final calculation
  -- Simplifying the final expression
  rw [speed_calc]
  have final_calc : (125 / 9) * 9 = 125 := by sorry
  exact final_calc

end train_length_l306_306816


namespace max_consecutive_sum_l306_306792

theorem max_consecutive_sum (n : ℕ) : 
  (∀ (n : ℕ), (n*(n + 1))/2 ≤ 400 → n ≤ 27) ∧ ((27*(27 + 1))/2 ≤ 400) :=
by
  sorry

end max_consecutive_sum_l306_306792


namespace milan_long_distance_bill_l306_306114

theorem milan_long_distance_bill
  (monthly_fee : ℝ := 2)
  (per_minute_cost : ℝ := 0.12)
  (minutes_used : ℕ := 178) :
  ((minutes_used : ℝ) * per_minute_cost + monthly_fee = 23.36) :=
by
  sorry

end milan_long_distance_bill_l306_306114


namespace clock_equiv_4_cubic_l306_306574

theorem clock_equiv_4_cubic :
  ∃ x : ℕ, x > 3 ∧ x % 12 = (x^3) % 12 ∧ (∀ y : ℕ, y > 3 ∧ y % 12 = (y^3) % 12 → x ≤ y) :=
by
  use 4
  sorry

end clock_equiv_4_cubic_l306_306574


namespace part1_part2_l306_306858

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4 * x + a + 3
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := b * x + 5 - 2 * b

theorem part1 (a : ℝ) : (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem part2 (b : ℝ) : (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 4 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 4 ∧ f x2 3 = g x1 b) ↔ -1 ≤ b ∧ b ≤ 1/2 :=
sorry

end part1_part2_l306_306858


namespace sequence_k_l306_306454

theorem sequence_k (a : ℕ → ℕ) (k : ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ m n, a (m + n) = a m * a n) 
    (h3 : (finset.range 10).sum (λ i, a (k + 1 + i)) = 2^15 - 2^5) : 
    k = 4 := 
by sorry

end sequence_k_l306_306454


namespace remainder_is_neg_x_plus_60_l306_306756

theorem remainder_is_neg_x_plus_60 (R : Polynomial ℝ) :
  (R.eval 10 = 50) ∧ (R.eval 50 = 10) → 
  ∃ Q : Polynomial ℝ, R = (Polynomial.X - 10) * (Polynomial.X - 50) * Q + (- Polynomial.X + 60) :=
by
  sorry

end remainder_is_neg_x_plus_60_l306_306756


namespace tickets_per_box_l306_306989

-- Definitions
def boxes (G: Type) : ℕ := 9
def total_tickets (G: Type) : ℕ := 45

-- Theorem statement
theorem tickets_per_box (G: Type) : total_tickets G / boxes G = 5 :=
by
  sorry

end tickets_per_box_l306_306989


namespace sum_of_numbers_in_ratio_l306_306117

theorem sum_of_numbers_in_ratio 
  (x : ℕ)
  (h : 5 * x = 560) : 
  2 * x + 3 * x + 4 * x + 5 * x = 1568 := 
by 
  sorry

end sum_of_numbers_in_ratio_l306_306117


namespace repeating_decimal_fraction_eq_l306_306379

theorem repeating_decimal_fraction_eq : (let x := (0.363636 : ℚ) in x = (4 : ℚ) / 11) :=
by
  let x := (0.363636 : ℚ)
  have h₀ : x = 0.36 + 0.0036 / (1 - 0.0001)
  calc
    x = 36 / 100 + 36 / 10000 + 36 / 1000000 + ... : sorry
    _ = 9 / 25 : sorry
    _ = (9 / 25) / (1 - 0.01) : sorry
    _ = (9 / 25) / (99 / 100) : sorry
    _ = (9 / 25) * (100 / 99) : sorry
    _ = 900 / 2475 : sorry
    _ = 4 / 11 : sorry

end repeating_decimal_fraction_eq_l306_306379


namespace drivers_schedule_l306_306599

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l306_306599


namespace four_digit_numbers_divisible_by_nine_l306_306135

theorem four_digit_numbers_divisible_by_nine : 
  (count (λ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 9 = 0) = 1000) :=
by
  sorry

end four_digit_numbers_divisible_by_nine_l306_306135


namespace sin_B_in_triangle_ABC_l306_306876

noncomputable def triangle := Type*
variables {A B C : triangle}
variables (AC BC : ℝ)

-- The given conditions
def right_angle_at_A (A B C : triangle) : Prop := true -- Placeholder for \angle A = 90^\circ
def length_AC : ℝ := 4
def length_BC : ℝ := real.sqrt 41

-- The theorem we want to prove
theorem sin_B_in_triangle_ABC 
  (h_right : right_angle_at_A A B C)
  (h_AC : AC = length_AC)
  (h_BC : BC = length_BC) :
  real.sin (real.atan (AC / BC)) = 4 / real.sqrt 41 :=
sorry

end sin_B_in_triangle_ABC_l306_306876


namespace ellipse_eccentricity_proof_l306_306979

theorem ellipse_eccentricity_proof (a b c : ℝ) 
  (ha_gt_hb : a > b) (hb_gt_zero : b > 0) (hc_gt_zero : c > 0)
  (h_ellipse : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_r : ∃ r : ℝ, r = (Real.sqrt 2 / 6) * c) :
  (Real.sqrt (1 - b^2 / a^2)) = (2 * Real.sqrt 5 / 5) := by {
  sorry
}

end ellipse_eccentricity_proof_l306_306979


namespace expression_without_arithmetic_square_root_l306_306642

theorem expression_without_arithmetic_square_root : 
  ¬ (∃ x, x^2 = (-|-9|)) :=
by { intro h, cases h with y hy, 
     have hy_nonneg : y^2 ≥ 0 
       := sq_nonneg y,
     let expr := y^2,
     show false,
     calc 
       expr = -|-9| : hy
       ... = -9    : by norm_num
       ... < 0     : by linarith,
}

end expression_without_arithmetic_square_root_l306_306642


namespace inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l306_306124

variable (a : ℝ) (x : ℝ)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

-- Part (1)
theorem inequality_solution (a : ℝ) (h1 : 0 < a ∧ a < 1) : (0 ≤ x ∧ x ≤ 2*a / (1 - a^2)) → (f x a ≤ 1) :=
sorry

theorem inequality_solution_b (a : ℝ) (h2 : a ≥ 1) : (0 ≤ x) → (f x a ≤ 1) :=
sorry

-- Part (2)
theorem monotonic_increasing (a : ℝ) (h3 : a ≤ 0) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≤ f x2 a :=
sorry

theorem monotonic_decreasing (a : ℝ) (h4 : a ≥ 1) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≥ f x2 a :=
sorry

end inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l306_306124


namespace competition_scores_order_l306_306999

theorem competition_scores_order (A B C D : ℕ) (h1 : A + B = C + D) (h2 : C + A > D + B) (h3 : B > A + D) : (B > A) ∧ (A > C) ∧ (C > D) := 
by 
  sorry

end competition_scores_order_l306_306999


namespace paint_for_cube_l306_306790

theorem paint_for_cube (paint_per_unit_area : ℕ → ℕ → ℕ)
  (h2 : paint_per_unit_area 2 1 = 1) :
  paint_per_unit_area 6 1 = 9 :=
by
  sorry

end paint_for_cube_l306_306790


namespace max_side_length_is_11_l306_306016

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306016


namespace drivers_schedule_l306_306601

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l306_306601


namespace coord_relationship_M_l306_306980

theorem coord_relationship_M (x y z : ℝ) (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, -1)) (hB : B = (2, 0, 2))
  (hM : ∃ M : ℝ × ℝ × ℝ, M = (x, y, z) ∧ y = 0 ∧ |(1 - x)^2 + 2^2 + (-1 - z)^2| = |(2 - x)^2 + (0 - z)^2|) :
  x + 3 * z - 1 = 0 ∧ y = 0 := 
sorry

end coord_relationship_M_l306_306980


namespace lcm_of_2_4_5_6_l306_306340

theorem lcm_of_2_4_5_6 : Nat.lcm (Nat.lcm (Nat.lcm 2 4) 5) 6 = 60 :=
by
  sorry

end lcm_of_2_4_5_6_l306_306340


namespace additional_fee_per_minute_for_second_plan_l306_306810

theorem additional_fee_per_minute_for_second_plan :
  (∃ x : ℝ, (22 + 0.13 * 280 = 8 + x * 280) ∧ x = 0.18) :=
sorry

end additional_fee_per_minute_for_second_plan_l306_306810


namespace drivers_sufficiency_and_ivan_petrovich_departure_l306_306598

def one_way_trip_mins := 2 * 60 + 40
def round_trip_mins := 2 * one_way_trip_mins
def rest_time_mins := 60

axiom driver_A_return_time : 12 * 60 + 40
axiom driver_A_next_trip_time := driver_A_return_time + rest_time_mins
axiom driver_D_departure_time : 13 * 60 + 5
axiom continuous_trips : True 

noncomputable def sufficient_drivers : nat := 4
noncomputable def ivan_petrovich_departure : nat := 10 * 60 + 40

theorem drivers_sufficiency_and_ivan_petrovich_departure :
  (sufficient_drivers = 4) ∧ (ivan_petrovich_departure = 10 * 60 + 40) := by
  sorry

end drivers_sufficiency_and_ivan_petrovich_departure_l306_306598


namespace value_of_expression_l306_306796

theorem value_of_expression (x : ℤ) (h : x = 4) : (3 * x + 7) ^ 2 = 361 := by
  rw [h] -- Replace x with 4
  norm_num -- Simplify the expression
  done

end value_of_expression_l306_306796


namespace max_side_length_is_11_l306_306014

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306014


namespace kenneth_money_left_l306_306755

noncomputable def baguettes : ℝ := 2 * 2
noncomputable def water : ℝ := 2 * 1

noncomputable def chocolate_bars_cost_before_discount : ℝ := 2 * 1.5
noncomputable def chocolate_bars_cost_after_discount : ℝ := chocolate_bars_cost_before_discount * (1 - 0.20)
noncomputable def chocolate_bars_final_cost : ℝ := chocolate_bars_cost_after_discount * 1.08

noncomputable def milk_cost_after_discount : ℝ := 3.5 * (1 - 0.10)

noncomputable def chips_cost_before_tax : ℝ := 2.5 + (2.5 * 0.50)
noncomputable def chips_final_cost : ℝ := chips_cost_before_tax * 1.08

noncomputable def total_cost : ℝ :=
  baguettes + water + chocolate_bars_final_cost + milk_cost_after_discount + chips_final_cost

noncomputable def initial_amount : ℝ := 50
noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem kenneth_money_left : amount_left = 50 - 15.792 := by
  sorry

end kenneth_money_left_l306_306755


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306037

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306037


namespace base6_sum_correct_l306_306140

theorem base6_sum_correct {S H E : ℕ} (hS : S < 6) (hH : H < 6) (hE : E < 6) 
  (dist : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (rightmost : (E + E) % 6 = S) 
  (second_rightmost : (H + H + if E + E < 6 then 0 else 1) % 6 = E) :
  S + H + E = 11 := 
by sorry

end base6_sum_correct_l306_306140


namespace cube_plane_intersection_distance_l306_306803

theorem cube_plane_intersection_distance :
  let vertices := [(0, 0, 0), (0, 0, 6), (0, 6, 0), (0, 6, 6), (6, 0, 0), (6, 0, 6), (6, 6, 0), (6, 6, 6)]
  let P := (0, 3, 0)
  let Q := (2, 0, 0)
  let R := (2, 6, 6)
  let plane_equation := 3 * x - 2 * y - 2 * z + 6 = 0
  let S := (2, 0, 6)
  let T := (0, 6, 3)
  dist S T = 7 := sorry

end cube_plane_intersection_distance_l306_306803


namespace probability_of_matching_correctly_l306_306824

-- Define the number of plants and seedlings.
def num_plants : ℕ := 4

-- Define the number of total arrangements.
def total_arrangements : ℕ := Nat.factorial num_plants

-- Define the number of correct arrangements.
def correct_arrangements : ℕ := 1

-- Define the probability of a correct guess.
def probability_of_correct_guess : ℚ := correct_arrangements / total_arrangements

-- The problem requires to prove that the probability of correct guess is 1/24
theorem probability_of_matching_correctly :
  probability_of_correct_guess = 1 / 24 :=
  by
    sorry

end probability_of_matching_correctly_l306_306824


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306028

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306028


namespace fried_hop_edges_in_three_hops_l306_306163

noncomputable def fried_hop_probability : ℚ :=
  let moves : List (Int × Int) := [(-1, 0), (1, 0), (0, -1), (0, 1)]
  let center := (2, 2)
  let edges := [(1, 2), (1, 3), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3)]
  -- Since the exact steps of solution calculation are complex,
  -- we assume the correct probability as per our given solution.
  5 / 8

theorem fried_hop_edges_in_three_hops :
  let p := fried_hop_probability
  p = 5 / 8 := by
  sorry

end fried_hop_edges_in_three_hops_l306_306163


namespace max_side_length_is_11_l306_306024

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306024


namespace max_triangle_side_24_l306_306084

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306084


namespace units_digit_of_N_is_8_l306_306288

def product_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens * units

def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens + units

theorem units_digit_of_N_is_8 (N : ℕ) (hN_range : 10 ≤ N ∧ N < 100)
    (hN_eq : N = product_of_digits N * sum_of_digits N) : N % 10 = 8 :=
sorry

end units_digit_of_N_is_8_l306_306288


namespace sector_area_l306_306413

theorem sector_area (radius : ℝ) (central_angle : ℝ) (h1 : radius = 3) (h2 : central_angle = 2 * Real.pi / 3) : 
    (1 / 2) * radius^2 * central_angle = 6 * Real.pi :=
by
  rw [h1, h2]
  sorry

end sector_area_l306_306413


namespace find_ABC_sum_l306_306182

theorem find_ABC_sum (A B C : ℤ) (h : ∀ x : ℤ, x = -3 ∨ x = 0 ∨ x = 4 → x^3 + A * x^2 + B * x + C = 0) : 
  A + B + C = -13 := 
by 
  sorry

end find_ABC_sum_l306_306182


namespace water_left_ratio_l306_306930

theorem water_left_ratio (h1: 2 * (30 / 10) = 6)
                        (h2: 2 * (30 / 10) = 6)
                        (h3: 4 * (60 / 10) = 24)
                        (water_left: ℕ)
                        (total_water_collected: ℕ) 
                        (h4: water_left = 18)
                        (h5: total_water_collected = 36) : 
  water_left * 2 = total_water_collected :=
by
  sorry

end water_left_ratio_l306_306930


namespace max_side_length_l306_306009

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306009


namespace evaluate_expression_l306_306969

def star (A B : ℚ) : ℚ := (A + B) / 3

theorem evaluate_expression : star (star 7 15) 10 = 52 / 9 := by
  sorry

end evaluate_expression_l306_306969


namespace elmer_saving_percent_l306_306105

theorem elmer_saving_percent (x c : ℝ) (hx : x > 0) (hc : c > 0) :
  let old_car_fuel_efficiency := x
  let new_car_fuel_efficiency := 1.6 * x
  let gasoline_cost := c
  let diesel_cost := 1.25 * c
  let trip_distance := 300
  let old_car_fuel_needed := trip_distance / old_car_fuel_efficiency
  let new_car_fuel_needed := trip_distance / new_car_fuel_efficiency
  let old_car_cost := old_car_fuel_needed * gasoline_cost
  let new_car_cost := new_car_fuel_needed * diesel_cost
  let cost_saving := old_car_cost - new_car_cost
  let percent_saving := (cost_saving / old_car_cost) * 100
  percent_saving = 21.875 :=
by
  sorry

end elmer_saving_percent_l306_306105


namespace total_length_of_wire_l306_306874

-- Definitions based on conditions
def num_squares : ℕ := 15
def length_of_grid : ℕ := 10
def width_of_grid : ℕ := 5
def height_of_grid : ℕ := 3
def side_length : ℕ := length_of_grid / width_of_grid -- 2 units
def num_horizontal_wires : ℕ := height_of_grid + 1    -- 4 wires
def num_vertical_wires : ℕ := width_of_grid + 1      -- 6 wires
def total_length_horizontal_wires : ℕ := num_horizontal_wires * length_of_grid -- 40 units
def total_length_vertical_wires : ℕ := num_vertical_wires * (height_of_grid * side_length) -- 36 units

-- The theorem to prove the total length of wire needed
theorem total_length_of_wire : total_length_horizontal_wires + total_length_vertical_wires = 76 :=
by
  sorry

end total_length_of_wire_l306_306874


namespace smallest_integral_value_of_y_l306_306252

theorem smallest_integral_value_of_y :
  ∃ y : ℤ, (1 / 4 : ℝ) < y / 7 ∧ y / 7 < 2 / 3 ∧ ∀ z : ℤ, (1 / 4 : ℝ) < z / 7 ∧ z / 7 < 2 / 3 → y ≤ z :=
by
  -- The statement is defined and the proof is left as "sorry" to illustrate that no solution steps are used directly.
  sorry

end smallest_integral_value_of_y_l306_306252


namespace playground_girls_count_l306_306486

theorem playground_girls_count (boys : ℕ) (total_children : ℕ) 
  (h_boys : boys = 35) (h_total : total_children = 63) : 
  ∃ girls : ℕ, girls = 28 ∧ girls = total_children - boys := 
by 
  sorry

end playground_girls_count_l306_306486


namespace points_can_move_on_same_line_l306_306885

variable {A B C x y x' y' : ℝ}

def transform_x (x y : ℝ) : ℝ := 3 * x + 2 * y + 1
def transform_y (x y : ℝ) : ℝ := x + 4 * y - 3

noncomputable def points_on_same_line (A B C : ℝ) (x y : ℝ) : Prop :=
  A*x + B*y + C = 0 ∧
  A*(transform_x x y) + B*(transform_y x y) + C = 0

theorem points_can_move_on_same_line :
  ∃ (A B C : ℝ), ∀ (x y : ℝ), points_on_same_line A B C x y :=
sorry

end points_can_move_on_same_line_l306_306885


namespace sphere_segment_volume_l306_306781

theorem sphere_segment_volume (r : ℝ) (ratio_surface_to_base : ℝ) : r = 10 → ratio_surface_to_base = 10 / 7 → ∃ V : ℝ, V = 288 * π :=
by
  intros
  sorry

end sphere_segment_volume_l306_306781


namespace male_students_outnumber_female_students_l306_306964

-- Define the given conditions
def total_students : ℕ := 928
def male_students : ℕ := 713
def female_students : ℕ := total_students - male_students

-- The theorem to be proven
theorem male_students_outnumber_female_students :
  male_students - female_students = 498 :=
by
  sorry

end male_students_outnumber_female_students_l306_306964


namespace range_of_a_l306_306567

open Set

theorem range_of_a (a : ℝ) (h1 : (∃ x, a^x > 1 ∧ x < 0) ∨ (∀ x, ax^2 - x + a ≥ 0))
  (h2 : ¬((∃ x, a^x > 1 ∧ x < 0) ∧ (∀ x, ax^2 - x + a ≥ 0))) :
  a ∈ (Ioo 0 (1/2)) ∪ (Ici 1) :=
by {
  sorry
}

end range_of_a_l306_306567


namespace probability_equal_white_black_probability_white_ge_black_l306_306351

/-- Part (a) -/
theorem probability_equal_white_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (2 * m) / (n + m)) := 
  sorry

/-- Part (b) -/
theorem probability_white_ge_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (n - m + 1) / (n + 1)) := 
  sorry

end probability_equal_white_black_probability_white_ge_black_l306_306351


namespace probability_of_two_red_shoes_l306_306655

theorem probability_of_two_red_shoes (total_shoes red_shoes green_shoes : ℕ) 
  (h_total : total_shoes = 10)
  (h_red : red_shoes = 7)
  (h_green : green_shoes = 3) :
  let C := λ n k : ℕ, nat.choose n k in
  (C red_shoes 2 : ℚ) / (C total_shoes 2) = 7 / 15 := 
by
  let C := λ n k : ℕ, nat.choose n k
  rw [h_total, h_red, h_green]
  sorry

end probability_of_two_red_shoes_l306_306655


namespace average_speed_correct_l306_306943

noncomputable def initial_odometer := 12321
noncomputable def final_odometer := 12421
noncomputable def time_hours := 4
noncomputable def distance := final_odometer - initial_odometer
noncomputable def avg_speed := distance / time_hours

theorem average_speed_correct : avg_speed = 25 := by
  sorry

end average_speed_correct_l306_306943


namespace log_inequality_l306_306976

noncomputable def a : ℝ := Real.log 3.6 / Real.log 2
noncomputable def b : ℝ := Real.log 3.2 / Real.log 4
noncomputable def c : ℝ := Real.log 3.6 / Real.log 4

theorem log_inequality : a > c ∧ c > b :=
by {
  -- Proof goes here
  sorry
}

end log_inequality_l306_306976


namespace intersection_point_unique_m_l306_306867

theorem intersection_point_unique_m (m : ℕ) (h1 : m > 0)
  (x y : ℤ) (h2 : 13 * x + 11 * y = 700) (h3 : y = m * x - 1) : m = 6 :=
by
  sorry

end intersection_point_unique_m_l306_306867


namespace dilation_0_minus_2i_to_neg3_minus_14i_l306_306312

open Complex

def dilation_centered (z_center z zk : ℂ) (factor : ℝ) : ℂ :=
  z_center + factor * (zk - z_center)

theorem dilation_0_minus_2i_to_neg3_minus_14i :
  dilation_centered (1 + 2 * I) (0 - 2 * I) (1 + 2 * I) 4 = -3 - 14 * I :=
by
  sorry

end dilation_0_minus_2i_to_neg3_minus_14i_l306_306312


namespace least_positive_integer_to_multiple_of_5_l306_306333

theorem least_positive_integer_to_multiple_of_5 : ∃ (n : ℕ), n > 0 ∧ (725 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m :=
by
  sorry

end least_positive_integer_to_multiple_of_5_l306_306333


namespace total_math_and_biology_homework_l306_306896

-- Definitions
def math_homework_pages : ℕ := 8
def biology_homework_pages : ℕ := 3

-- Theorem stating the problem to prove
theorem total_math_and_biology_homework :
  math_homework_pages + biology_homework_pages = 11 :=
by
  sorry

end total_math_and_biology_homework_l306_306896


namespace distribute_candy_bars_l306_306773

theorem distribute_candy_bars (candies bags : ℕ) (h1 : candies = 15) (h2 : bags = 5) :
  candies / bags = 3 :=
by
  sorry

end distribute_candy_bars_l306_306773


namespace abc_ineq_l306_306420

theorem abc_ineq (a b c : ℝ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : c > 0) (h₄ : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27 / 8 :=
sorry

end abc_ineq_l306_306420


namespace distance_AB_polar_l306_306880

open Real

/-- The distance between points A and B in polar coordinates, given that θ₁ - θ₂ = π. -/
theorem distance_AB_polar (A B : ℝ × ℝ) (r1 r2 : ℝ) (θ1 θ2 : ℝ) (hA : A = (r1, θ1)) (hB : B = (r2, θ2)) (hθ : θ1 - θ2 = π) :
  dist (r1 * cos θ1, r1 * sin θ1) (r2 * cos θ2, r2 * sin θ2) = r1 + r2 :=
sorry

end distance_AB_polar_l306_306880


namespace find_k_l306_306453

open BigOperators

def a (n : ℕ) : ℕ := 2 ^ n

theorem find_k (k : ℕ) (h : a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8) + a (k+9) + a (k+10) = 2 ^ 15 - 2 ^ 5) : k = 4 :=
sorry

end find_k_l306_306453


namespace no_such_b_c_exist_l306_306246

theorem no_such_b_c_exist :
  ¬ ∃ (b c : ℝ), (∃ (k l : ℤ), (k ≠ l ∧ (k ^ 2 + b * ↑k + c = 0) ∧ (l ^ 2 + b * ↑l + c = 0))) ∧
                  (∃ (m n : ℤ), (m ≠ n ∧ (2 * (m ^ 2) + (b + 1) * ↑m + (c + 1) = 0) ∧ 
                                        (2 * (n ^ 2) + (b + 1) * ↑n + (c + 1) = 0))) :=
sorry

end no_such_b_c_exist_l306_306246


namespace max_side_length_l306_306010

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306010


namespace nineteen_times_eight_pow_n_plus_seventeen_is_composite_l306_306768

theorem nineteen_times_eight_pow_n_plus_seventeen_is_composite 
  (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
sorry

end nineteen_times_eight_pow_n_plus_seventeen_is_composite_l306_306768


namespace area_ratio_independent_l306_306763

-- Definitions related to the problem
variables (AB BC CD : ℝ) (e f g : ℝ)

-- Let the lengths be defined as follows
def AB_def : Prop := AB = 2 * e
def BC_def : Prop := BC = 2 * f
def CD_def : Prop := CD = 2 * g

-- Let the areas be defined as follows
def area_quadrilateral (e f g : ℝ) : ℝ :=
  2 * (e + f) * (f + g)

def area_enclosed (e f g : ℝ) : ℝ :=
  (e + f + g) ^ 2 + f ^ 2 - e ^ 2 - g ^ 2

-- Prove the ratio is 2 / π
theorem area_ratio_independent (e f g : ℝ) (h1 : AB_def AB e)
  (h2 : BC_def BC f) (h3 : CD_def CD g) :
  (area_quadrilateral e f g) / ((area_enclosed e f g) * (π / 2)) = 2 / π :=
by
  sorry

end area_ratio_independent_l306_306763


namespace half_product_two_consecutive_integers_mod_3_l306_306303

theorem half_product_two_consecutive_integers_mod_3 (A : ℤ) : 
  (A * (A + 1) / 2) % 3 = 0 ∨ (A * (A + 1) / 2) % 3 = 1 :=
sorry

end half_product_two_consecutive_integers_mod_3_l306_306303


namespace sequence_difference_l306_306433

theorem sequence_difference (a : ℕ → ℤ) (h_rec : ∀ n : ℕ, a (n + 1) + a n = n) (h_a1 : a 1 = 2) :
  a 4 - a 2 = 1 :=
sorry

end sequence_difference_l306_306433


namespace prime_mod_30_not_composite_l306_306966

theorem prime_mod_30_not_composite (p : ℕ) (h_prime : Prime p) (h_gt_30 : p > 30) : 
  ¬ ∃ (x : ℕ), (x > 1 ∧ ∃ (a b : ℕ), x = a * b ∧ a > 1 ∧ b > 1) ∧ (0 < x ∧ x < 30 ∧ ∃ (k : ℕ), p = 30 * k + x) :=
by
  sorry

end prime_mod_30_not_composite_l306_306966


namespace width_of_carton_is_25_l306_306507

-- Definitions for the given problem
def carton_width := 25
def carton_length := 60
def width_or_height := min carton_width carton_length

theorem width_of_carton_is_25 : width_or_height = 25 := by
  sorry

end width_of_carton_is_25_l306_306507


namespace max_side_of_triangle_l306_306056

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306056


namespace fraction_rep_finite_geom_series_036_l306_306395

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l306_306395


namespace units_digit_of_7_pow_6_pow_5_l306_306101

theorem units_digit_of_7_pow_6_pow_5 : ((7 : ℕ)^ (6^5) % 10) = 1 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l306_306101


namespace find_number_l306_306277

theorem find_number (x : ℚ) (h : (3 * x / 2) + 6 = 11) : x = 10 / 3 :=
sorry

end find_number_l306_306277


namespace max_product_of_sum_2024_l306_306628

theorem max_product_of_sum_2024 : 
  ∃ (x y : ℤ), x + y = 2024 ∧ x * y = 1024144 :=
by
  use 1012, 1012
  split
  · sorry
  · sorry

end max_product_of_sum_2024_l306_306628


namespace no_real_roots_x2_bx_8_eq_0_l306_306115

theorem no_real_roots_x2_bx_8_eq_0 (b : ℝ) :
  (∀ x : ℝ, x^2 + b * x + 5 ≠ -3) ↔ (-4 * Real.sqrt 2 < b ∧ b < 4 * Real.sqrt 2) := by
  sorry

end no_real_roots_x2_bx_8_eq_0_l306_306115


namespace gary_egg_collection_l306_306702

-- Conditions
def initial_chickens : ℕ := 4
def multiplier : ℕ := 8
def eggs_per_chicken_per_day : ℕ := 6
def days_in_week : ℕ := 7

-- Definitions derived from conditions
def current_chickens : ℕ := initial_chickens * multiplier
def eggs_per_day : ℕ := current_chickens * eggs_per_chicken_per_day
def eggs_per_week : ℕ := eggs_per_day * days_in_week

-- Proof statement
theorem gary_egg_collection : eggs_per_week = 1344 := by
  unfold eggs_per_week
  unfold eggs_per_day
  unfold current_chickens
  sorry

end gary_egg_collection_l306_306702


namespace percentage_for_x_plus_y_l306_306866

theorem percentage_for_x_plus_y (x y : Real) (P : Real) 
  (h1 : 0.60 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.5 * x) : 
  P = 20 := 
by 
  sorry

end percentage_for_x_plus_y_l306_306866


namespace total_pieces_l306_306172

def gum_packages : ℕ := 28
def candy_packages : ℕ := 14
def pieces_per_package : ℕ := 6

theorem total_pieces : (gum_packages * pieces_per_package) + (candy_packages * pieces_per_package) = 252 :=
by
  sorry

end total_pieces_l306_306172


namespace b_must_be_one_l306_306324

theorem b_must_be_one (a b : ℝ) (h1 : a + b - a * b = 1) (h2 : ∀ n : ℤ, a ≠ n) : b = 1 :=
sorry

end b_must_be_one_l306_306324


namespace b_integer_iff_a_special_form_l306_306464

theorem b_integer_iff_a_special_form (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h2 : b = (a + Real.sqrt (a ^ 2 + 1)) ^ (1 / 3) + (a - Real.sqrt (a ^ 2 + 1)) ^ (1 / 3)) : 
  (∃ (n : ℕ), a = 1 / 2 * (n * (n^2 + 3))) ↔ (∃ (n : ℕ), b = n) :=
sorry

end b_integer_iff_a_special_form_l306_306464


namespace athena_spent_correct_amount_l306_306281

-- Define the conditions
def num_sandwiches : ℕ := 3
def price_per_sandwich : ℝ := 3
def num_drinks : ℕ := 2
def price_per_drink : ℝ := 2.5

-- Define the total cost as per the given conditions
def total_cost : ℝ :=
  (num_sandwiches * price_per_sandwich) + (num_drinks * price_per_drink)

-- The theorem that states the problem and asserts the correct answer
theorem athena_spent_correct_amount : total_cost = 14 := 
  by
    sorry

end athena_spent_correct_amount_l306_306281


namespace problem1_problem2_l306_306236

theorem problem1 : (1 * (-2016 : ℝ)^0 + 32 * 2^(2/3) + (1/4)^(-1/2)) = 5 := 
by 
  sorry

-- Note: The second problem involves approximate equality. We typically do this by using an error term such as 0.001 for demonstration purposes.
theorem problem2 : (log 3 81 + log 10 20 + log 10 5 + 4^(log 4 2) + log 5 1) ≈ 8.301 := 
by 
  -- Here, we would use tactics to show that the difference is less than a small epsilon.
  -- For now, we provide a placeholder.
  sorry

end problem1_problem2_l306_306236


namespace concert_ticket_revenue_l306_306620

theorem concert_ticket_revenue :
  let price_student : ℕ := 9
  let price_non_student : ℕ := 11
  let total_tickets : ℕ := 2000
  let student_tickets : ℕ := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  revenue_student + revenue_non_student = 20960 :=
by
  -- Definitions
  let price_student := 9
  let price_non_student := 11
  let total_tickets := 2000
  let student_tickets := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  -- Proof
  sorry  -- Placeholder for the proof

end concert_ticket_revenue_l306_306620


namespace no_solution_of_fractional_equation_l306_306309

theorem no_solution_of_fractional_equation (x : ℝ) : ¬ (x - 8) / (x - 7) - 8 = 1 / (7 - x) := 
sorry

end no_solution_of_fractional_equation_l306_306309


namespace find_omega2019_value_l306_306917

noncomputable def omega_n (n : ℕ) : ℝ := (2 * n - 1) * Real.pi / 2

theorem find_omega2019_value :
  omega_n 2019 = 4037 * Real.pi / 2 :=
by
  sorry

end find_omega2019_value_l306_306917


namespace three_digit_difference_l306_306550

theorem three_digit_difference (x : ℕ) (a b c : ℕ)
  (h1 : a = x + 2)
  (h2 : b = x + 1)
  (h3 : c = x)
  (h4 : a > b)
  (h5 : b > c) :
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = 198 :=
by
  sorry

end three_digit_difference_l306_306550


namespace problem1_problem2_l306_306699

-- Problem 1 Definition: Operation ※
def operation (m n : ℚ) : ℚ := 3 * m - n

-- Lean 4 statement: Prove 2※10 = -4
theorem problem1 : operation 2 10 = -4 := by
  sorry

-- Lean 4 statement: Prove that ※ does not satisfy the distributive law
theorem problem2 (a b c : ℚ) : 
  operation a (b + c) ≠ operation a b + operation a c := by
  sorry

end problem1_problem2_l306_306699


namespace total_copies_to_save_40_each_l306_306907

-- Definitions for the conditions.
def cost_per_copy : ℝ := 0.02
def discount_rate : ℝ := 0.25
def min_copies_for_discount : ℕ := 100
def savings_required : ℝ := 0.40
def steve_copies : ℕ := 80
def dinley_copies : ℕ := 80

-- Lean 4 statement to prove the total number of copies 
-- to save $0.40 each.
theorem total_copies_to_save_40_each : 
  (steve_copies + dinley_copies) + 
  (savings_required / (cost_per_copy * discount_rate)) * 2 = 320 :=
by 
  sorry

end total_copies_to_save_40_each_l306_306907


namespace gcd_123456_789012_l306_306402

theorem gcd_123456_789012 : Nat.gcd 123456 789012 = 36 := sorry

end gcd_123456_789012_l306_306402


namespace proof_strictly_increasing_sequence_l306_306473

noncomputable def exists_strictly_increasing_sequence : Prop :=
  ∃ a : ℕ → ℕ, 
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) ∧
    (∀ n : ℕ, 0 < n → a n > n^2 / 16)

theorem proof_strictly_increasing_sequence : exists_strictly_increasing_sequence :=
  sorry

end proof_strictly_increasing_sequence_l306_306473


namespace hyperbola_equation_l306_306421

theorem hyperbola_equation (c : ℝ) (b a : ℝ) 
  (h₁ : c = 2 * Real.sqrt 5) 
  (h₂ : a^2 + b^2 = c^2) 
  (h₃ : b / a = 1 / 2) : 
  (x y : ℝ) → (x^2 / 16) - (y^2 / 4) = 1 :=
by
  sorry

end hyperbola_equation_l306_306421


namespace time_jran_l306_306287

variable (D : ℕ) (S : ℕ)

theorem time_jran (hD: D = 80) (hS : S = 10) : D / S = 8 := 
  sorry

end time_jran_l306_306287


namespace corey_lowest_score_l306_306238

theorem corey_lowest_score
  (e1 e2 e3 e4 : ℕ)
  (h1 : e1 = 84)
  (h2 : e2 = 67)
  (max_score : ∀ (e : ℕ), e ≤ 100)
  (avg_at_least_75 : (e1 + e2 + e3 + e4) / 4 ≥ 75) :
  e3 ≥ 49 ∨ e4 ≥ 49 :=
by
  sorry

end corey_lowest_score_l306_306238


namespace max_length_of_third_side_of_triangle_l306_306904

noncomputable def max_third_side_length (D E F : ℝ) (a b : ℝ) : ℝ :=
  let c_square := a^2 + b^2 - 2 * a * b * Real.cos (90 * Real.pi / 180)
  Real.sqrt c_square

theorem max_length_of_third_side_of_triangle (D E F : ℝ) (a b : ℝ) (h₁ : Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1)
    (h₂ : a = 8) (h₃ : b = 15) : 
    max_third_side_length D E F a b = 17 := 
by
  sorry

end max_length_of_third_side_of_triangle_l306_306904


namespace Lisa_income_percentage_J_M_combined_l306_306764

variables (T M J L : ℝ)

-- Conditions as definitions
def Mary_income_eq_1p6_T (M T : ℝ) : Prop := M = 1.60 * T
def Tim_income_eq_0p5_J (T J : ℝ) : Prop := T = 0.50 * J
def Lisa_income_eq_1p3_M (L M : ℝ) : Prop := L = 1.30 * M
def Lisa_income_eq_0p75_J (L J : ℝ) : Prop := L = 0.75 * J

-- Theorem statement
theorem Lisa_income_percentage_J_M_combined (M T J L : ℝ)
  (h1 : Mary_income_eq_1p6_T M T)
  (h2 : Tim_income_eq_0p5_J T J)
  (h3 : Lisa_income_eq_1p3_M L M)
  (h4 : Lisa_income_eq_0p75_J L J) :
  (L / (M + J)) * 100 = 41.67 := 
sorry

end Lisa_income_percentage_J_M_combined_l306_306764


namespace shaded_area_correct_l306_306329

-- Definitions of the given conditions
def first_rectangle_length : ℕ := 8
def first_rectangle_width : ℕ := 5
def second_rectangle_length : ℕ := 4
def second_rectangle_width : ℕ := 9
def overlapping_area : ℕ := 3

def first_rectangle_area := first_rectangle_length * first_rectangle_width
def second_rectangle_area := second_rectangle_length * second_rectangle_width

-- Problem statement in Lean 4
theorem shaded_area_correct :
  first_rectangle_area + second_rectangle_area - overlapping_area = 73 :=
by
  -- The proof is skipped
  sorry

end shaded_area_correct_l306_306329


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306027

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306027


namespace men_in_business_class_l306_306470

theorem men_in_business_class (total_passengers : ℕ) (percentage_men : ℝ)
  (fraction_business_class : ℝ) (num_men_in_business_class : ℕ) 
  (h1 : total_passengers = 160) 
  (h2 : percentage_men = 0.75) 
  (h3 : fraction_business_class = 1 / 4) 
  (h4 : num_men_in_business_class = total_passengers * percentage_men * fraction_business_class) : 
  num_men_in_business_class = 30 := 
  sorry

end men_in_business_class_l306_306470


namespace change_received_correct_l306_306887

-- Define the prices of items and the amount paid
def price_hamburger : ℕ := 4
def price_onion_rings : ℕ := 2
def price_smoothie : ℕ := 3
def amount_paid : ℕ := 20

-- Define the total cost and the change received
def total_cost : ℕ := price_hamburger + price_onion_rings + price_smoothie
def change_received : ℕ := amount_paid - total_cost

-- Theorem stating the change received
theorem change_received_correct : change_received = 11 := by
  sorry

end change_received_correct_l306_306887


namespace range_of_m_l306_306706

-- Definition of p: x / (x - 2) < 0 implies 0 < x < 2
def p (x : ℝ) : Prop := x / (x - 2) < 0

-- Definition of q: 0 < x < m
def q (x m : ℝ) : Prop := 0 < x ∧ x < m

-- Main theorem: If p is a necessary but not sufficient condition for q to hold, then the range of m is (2, +∞)
theorem range_of_m {m : ℝ} (h : ∀ x, p x → q x m) (hs : ∃ x, ¬(q x m) ∧ p x) : 
  2 < m :=
sorry

end range_of_m_l306_306706


namespace intersection_complement_l306_306762

open Set

variable (U : Set ℕ) (P Q : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (hP : P = {1, 2, 3, 4, 5})
variable (hQ : Q = {3, 4, 5, 6, 7})

theorem intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_l306_306762


namespace sale_price_of_trouser_l306_306561

theorem sale_price_of_trouser : (100 - 0.70 * 100) = 30 := by
  sorry

end sale_price_of_trouser_l306_306561


namespace quadratic_coefficient_nonzero_l306_306440

theorem quadratic_coefficient_nonzero (a : ℝ) (x : ℝ) :
  (a - 3) * x^2 - 3 * x - 4 = 0 → a ≠ 3 :=
sorry

end quadratic_coefficient_nonzero_l306_306440


namespace fraction_repeating_decimal_l306_306377

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l306_306377


namespace min_xy_min_x_plus_y_l306_306169

theorem min_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : xy ≥ 36 :=
sorry  

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end min_xy_min_x_plus_y_l306_306169


namespace age_product_difference_is_nine_l306_306826

namespace ArnoldDanny

def current_age := 4
def product_today (A : ℕ) := A * A
def product_next_year (A : ℕ) := (A + 1) * (A + 1)
def difference (A : ℕ) := product_next_year A - product_today A

theorem age_product_difference_is_nine :
  difference current_age = 9 :=
by
  sorry

end ArnoldDanny

end age_product_difference_is_nine_l306_306826


namespace repeating_decimal_equals_fraction_l306_306401

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l306_306401


namespace max_triangle_side_24_l306_306082

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306082


namespace probability_at_least_6_heads_in_9_flips_l306_306953

def fair_coin_flips := 9
def total_outcomes := 512

/-- The probability of obtaining at least 'k' consecutive heads in 'n' flips of a fair coin. -/
def at_least_k_consecutive_heads_probability (k n : ℕ) : ℚ :=
∑ i in ((list.range (n - k + 1)).map (λ start_pos, 
  let end_pos := start_pos + k - 1 in 
  ((2 : ℚ)^(start_pos) + (n - end_pos - 1)))) 
  , (1 / (2^n : ℚ))

theorem probability_at_least_6_heads_in_9_flips :
  at_least_k_consecutive_heads_probability 6 9 = 49 / 512 := sorry

end probability_at_least_6_heads_in_9_flips_l306_306953


namespace parabola_properties_l306_306894

theorem parabola_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  (∀ x, a * x^2 + b * x + c >= a * (x^2)) ∧
  (c < 0) ∧ 
  (-b / (2 * a) < 0) :=
by
  sorry

end parabola_properties_l306_306894


namespace green_sequins_per_row_correct_l306_306456

def total_blue_sequins : ℕ := 6 * 8
def total_purple_sequins : ℕ := 5 * 12
def total_green_sequins : ℕ := 162 - (total_blue_sequins + total_purple_sequins)
def green_sequins_per_row : ℕ := total_green_sequins / 9

theorem green_sequins_per_row_correct : green_sequins_per_row = 6 := 
by 
  sorry

end green_sequins_per_row_correct_l306_306456


namespace repeating_decimal_fraction_equiv_l306_306386

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l306_306386


namespace sum_of_a_b_l306_306142

theorem sum_of_a_b (a b : ℝ) (h1 : ∀ x : ℝ, (a * (b * x + a) + b = x))
  (h2 : ∀ y : ℝ, (b * (a * y + b) + a = y)) : a + b = -2 := 
sorry

end sum_of_a_b_l306_306142


namespace angle_ACD_l306_306749

theorem angle_ACD (E : ℝ) (arc_eq : ∀ (AB BC CD : ℝ), AB = BC ∧ BC = CD) (angle_eq : E = 40) : ∃ (ACD : ℝ), ACD = 15 :=
by
  sorry

end angle_ACD_l306_306749


namespace profit_difference_l306_306498

-- Define the initial capitals of A, B, and C
def capital_A := 8000
def capital_B := 10000
def capital_C := 12000

-- Define B's profit share
def profit_share_B := 3500

-- Define the total number of parts
def total_parts := 15

-- Define the number of parts for each person
def parts_A := 4
def parts_B := 5
def parts_C := 6

-- Define the total profit
noncomputable def total_profit := profit_share_B * (total_parts / parts_B)

-- Define the profit shares of A and C
noncomputable def profit_share_A := (parts_A / total_parts) * total_profit
noncomputable def profit_share_C := (parts_C / total_parts) * total_profit

-- Define the difference between the profit shares of A and C
noncomputable def profit_share_difference := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_difference :
  profit_share_difference = 1400 := by
  sorry

end profit_difference_l306_306498


namespace f_of_f_five_l306_306181

noncomputable def f : ℝ → ℝ := sorry

axiom f_periodicity (x : ℝ) : f (x + 2) = 1 / f x
axiom f_initial_value : f 1 = -5

theorem f_of_f_five : f (f 5) = -1 / 5 :=
by sorry

end f_of_f_five_l306_306181


namespace solve_inequalities_system_l306_306924

theorem solve_inequalities_system (x : ℝ) :
  (x - 2 ≥ -5) → (3x < x + 2) → (-3 ≤ x ∧ x < 1) :=
by
  intros h1 h2
  sorry

end solve_inequalities_system_l306_306924


namespace tom_killed_enemies_l306_306933

-- Define the number of points per enemy
def points_per_enemy : ℝ := 10

-- Define the bonus threshold and bonus factor
def bonus_threshold : ℝ := 100
def bonus_factor : ℝ := 1.5

-- Define the total score achieved by Tom
def total_score : ℝ := 2250

-- Define the number of enemies killed by Tom
variable (E : ℝ)

-- The proof goal
theorem tom_killed_enemies 
  (h1 : E ≥ bonus_threshold)
  (h2 : bonus_factor * points_per_enemy * E = total_score) : 
  E = 150 :=
sorry

end tom_killed_enemies_l306_306933


namespace false_statement_D_l306_306342

theorem false_statement_D :
  ¬ (∀ {α β : ℝ}, α = β → (true → true → true → α = β ↔ α = β)) :=
by
  sorry

end false_statement_D_l306_306342


namespace largest_divisor_of_odd_product_l306_306791

theorem largest_divisor_of_odd_product (n : ℕ) (h : Even n ∧ n > 0) :
  ∃ m, m > 0 ∧ (∀ k, (n+1)*(n+3)*(n+7)*(n+9)*(n+11) % k = 0 ↔ k ≤ 15) := by
  -- Proof goes here
  sorry

end largest_divisor_of_odd_product_l306_306791


namespace problem_statement_l306_306643

def has_arithmetic_square_root (x : ℝ) : Prop :=
  ∃ y : ℝ, y * y = x

theorem problem_statement :
  (¬ has_arithmetic_square_root (-abs 9)) ∧
  (has_arithmetic_square_root ((-1/4)^2)) ∧
  (has_arithmetic_square_root 0) ∧
  (has_arithmetic_square_root (10^2)) := 
sorry

end problem_statement_l306_306643


namespace sequence_k_eq_4_l306_306452

theorem sequence_k_eq_4 {a : ℕ → ℕ} (h1 : a 1 = 2) (h2 : ∀ m n, a (m + n) = a m * a n)
    (h3 : ∑ i in finset.range 10, a (4 + i) = 2^15 - 2^5) : 4 = 4 :=
by
  sorry

end sequence_k_eq_4_l306_306452


namespace problem_l306_306729

theorem problem (x y z : ℕ) (hx : x < 9) (hy : y < 9) (hz : z < 9) 
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z % 9 = 0) :=
sorry

end problem_l306_306729


namespace circle_inscribed_in_square_area_l306_306589

theorem circle_inscribed_in_square_area :
  ∀ (x y : ℝ) (h : 2 * x^2 + 2 * y^2 - 8 * x - 12 * y + 24 = 0),
  ∃ side : ℝ, 4 * (side^2) = 16 :=
by
  sorry

end circle_inscribed_in_square_area_l306_306589


namespace parabola_min_perimeter_l306_306445

noncomputable def focus_of_parabola (p : ℝ) (hp : p > 0) : ℝ × ℝ :=
(1, 0)

noncomputable def A : ℝ × ℝ := (3, 2)

noncomputable def is_on_parabola (P : ℝ × ℝ) (p : ℝ) : Prop :=
P.2 ^ 2 = 2 * p * P.1

noncomputable def area_of_triangle (A P F : ℝ × ℝ) : ℝ :=
0.5 * abs (A.1 * (P.2 - F.2) + P.1 * (F.2 - A.2) + F.1 * (A.2 - P.2))

noncomputable def perimeter (A P F : ℝ × ℝ) : ℝ := 
abs (A.1 - P.1) + abs (A.1 - F.1) + abs (P.1 - F.1)

theorem parabola_min_perimeter 
  {p : ℝ} (hp : p > 0)
  (A : ℝ × ℝ) (ha : A = (3,2))
  (P : ℝ × ℝ) (hP : is_on_parabola P p)
  {F : ℝ × ℝ} (hF : F = focus_of_parabola p hp)
  (harea : area_of_triangle A P F = 1)
  (hmin : ∀ P', is_on_parabola P' p → 
    perimeter A P' F ≥ perimeter A P F) :
  abs (P.1 - F.1) = 5/2 :=
sorry

end parabola_min_perimeter_l306_306445


namespace larger_number_of_product_56_and_sum_15_l306_306937

theorem larger_number_of_product_56_and_sum_15 (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 := 
by
  sorry

end larger_number_of_product_56_and_sum_15_l306_306937


namespace max_side_length_is_11_l306_306022

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306022


namespace determine_n_l306_306098

theorem determine_n (n : ℕ) (x : ℤ) (h : x^n + (2 + x)^n + (2 - x)^n = 0) : n = 1 :=
sorry

end determine_n_l306_306098


namespace sampling_is_stratified_l306_306207

-- Given Conditions
def number_of_male_students := 500
def number_of_female_students := 400
def sampled_male_students := 25
def sampled_female_students := 20

-- Definition of stratified sampling according to the problem context
def is_stratified_sampling (N_M F_M R_M R_F : ℕ) : Prop :=
  (R_M > 0 ∧ R_F > 0 ∧ R_M < N_M ∧ R_F < N_M ∧ N_M > 0 ∧ N_M > 0)

-- Proving that the sampling method is stratified sampling
theorem sampling_is_stratified : 
  is_stratified_sampling number_of_male_students number_of_female_students sampled_male_students sampled_female_students = true :=
by
  sorry

end sampling_is_stratified_l306_306207


namespace first_reduction_percentage_l306_306814

theorem first_reduction_percentage 
  (P : ℝ)  -- original price
  (x : ℝ)  -- first day reduction percentage
  (h : P > 0) -- price assumption
  (h2 : 0 ≤ x ∧ x ≤ 100) -- percentage assumption
  (cond : P * (1 - x / 100) * 0.86 = 0.774 * P) : 
  x = 10 := 
sorry

end first_reduction_percentage_l306_306814


namespace emily_spent_20_dollars_l306_306837

/-- Let X be the amount Emily spent on Friday. --/
variables (X : ℝ)

/-- Emily spent twice the amount on Saturday. --/
def saturday_spent := 2 * X

/-- Emily spent three times the amount on Sunday. --/
def sunday_spent := 3 * X

/-- The total amount spent over the three days is $120. --/
axiom total_spent : X + saturday_spent X + sunday_spent X = 120

/-- Prove that X = 20. --/
theorem emily_spent_20_dollars : X = 20 :=
sorry

end emily_spent_20_dollars_l306_306837


namespace remainder_of_large_number_l306_306108

theorem remainder_of_large_number (N : ℕ) (hN : N = 123456789012): 
  N % 360 = 108 :=
by
  have h1 : N % 4 = 0 := by 
    sorry
  have h2 : N % 9 = 3 := by 
    sorry
  have h3 : N % 10 = 2 := by
    sorry
  sorry

end remainder_of_large_number_l306_306108


namespace toilet_paper_packs_needed_l306_306302

-- Definitions based on conditions
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def weeks : ℕ := 4
def rolls_per_pack : ℕ := 12
def daily_stock : ℕ := 1

-- The main theorem statement
theorem toilet_paper_packs_needed : 
  (bathrooms * days_per_week * weeks) / rolls_per_pack = 14 := by
sorry

end toilet_paper_packs_needed_l306_306302


namespace integral_curves_l306_306489

theorem integral_curves (y x : ℝ) : 
  (∃ k : ℝ, (y - x) / (y + x) = k) → 
  (∃ c : ℝ, y = x * (c + 1) / (c - 1)) ∨ (y = 0) ∨ (y = x) ∨ (x = 0) :=
by
  sorry

end integral_curves_l306_306489


namespace man_speed_with_current_l306_306809

-- Define the conditions
def current_speed : ℕ := 3
def man_speed_against_current : ℕ := 14

-- Define the man's speed in still water (v) based on the given speed against the current
def man_speed_in_still_water : ℕ := man_speed_against_current + current_speed

-- Prove that the man's speed with the current is 20 kmph
theorem man_speed_with_current : man_speed_in_still_water + current_speed = 20 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end man_speed_with_current_l306_306809


namespace probability_of_drawing_green_ball_l306_306740

variable (total_balls green_balls : ℕ)
variable (total_balls_eq : total_balls = 10)
variable (green_balls_eq : green_balls = 4)

theorem probability_of_drawing_green_ball (h_total : total_balls = 10) (h_green : green_balls = 4) :
  (green_balls : ℚ) / total_balls = 2 / 5 := by
  sorry

end probability_of_drawing_green_ball_l306_306740


namespace atm_withdrawal_cost_l306_306908

theorem atm_withdrawal_cost (x y : ℝ)
  (h1 : 221 = x + 40000 * y)
  (h2 : 485 = x + 100000 * y) :
  (x + 85000 * y) = 419 := by
  sorry

end atm_withdrawal_cost_l306_306908


namespace find_q_sum_l306_306849

variable (q : ℕ → ℕ)

def conditions :=
  q 3 = 2 ∧ 
  q 8 = 20 ∧ 
  q 16 = 12 ∧ 
  q 21 = 30

theorem find_q_sum (h : conditions q) : 
  (q 1 + q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + 
   q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18 + q 19 + q 20 + q 21 + q 22) = 352 := 
  sorry

end find_q_sum_l306_306849


namespace speed_against_current_l306_306355

theorem speed_against_current (V_m V_c : ℕ) (h1 : V_m + V_c = 20) (h2 : V_c = 3) : V_m - V_c = 14 :=
by 
  sorry

end speed_against_current_l306_306355


namespace percentage_relation_l306_306723

theorem percentage_relation (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 12) : y = 60 := by
  sorry

end percentage_relation_l306_306723


namespace grasshopper_jump_l306_306914

-- Definitions for the distances jumped
variables (G F M : ℕ)

-- Conditions given in the problem
def condition1 : Prop := G = F + 19
def condition2 : Prop := M = F - 12
def condition3 : Prop := M = 8

-- The theorem statement
theorem grasshopper_jump : condition1 G F ∧ condition2 F M ∧ condition3 M → G = 39 :=
by
  sorry

end grasshopper_jump_l306_306914


namespace distance_school_house_l306_306199

def speed_to_school : ℝ := 6
def speed_from_school : ℝ := 4
def total_time : ℝ := 10

theorem distance_school_house : 
  ∃ D : ℝ, (D / speed_to_school + D / speed_from_school = total_time) ∧ (D = 24) :=
sorry

end distance_school_house_l306_306199


namespace max_side_length_l306_306003

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306003


namespace least_positive_integer_to_multiple_of_5_l306_306337

theorem least_positive_integer_to_multiple_of_5 (n : ℕ) (h₁ : n = 725) :
  ∃ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 ∧ ∀ k : ℕ, (k > 0 ∧ (725 + k) % 5 = 0) → m ≤ k :=
begin
  use 5,
  sorry
end

end least_positive_integer_to_multiple_of_5_l306_306337


namespace determine_day_from_statements_l306_306227

/-- Define the days of the week as an inductive type. -/
inductive Day where
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
  deriving DecidableEq, Repr

open Day

/-- Define the properties of the lion lying on specific days. -/
def lion_lies (d : Day) : Prop :=
  d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Define the properties of the lion telling the truth on specific days. -/
def lion_truth (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday ∨ d = Sunday

/-- Define the properties of the unicorn lying on specific days. -/
def unicorn_lies (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday

/-- Define the properties of the unicorn telling the truth on specific days. -/
def unicorn_truth (d : Day) : Prop :=
  d = Sunday ∨ d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Function to determine the day before a given day. -/
def yesterday (d : Day) : Day :=
  match d with
  | Monday    => Sunday
  | Tuesday   => Monday
  | Wednesday => Tuesday
  | Thursday  => Wednesday
  | Friday    => Thursday
  | Saturday  => Friday
  | Sunday    => Saturday

/-- Define the lion's statement: "Yesterday was a day when I lied." -/
def lion_statement (d : Day) : Prop :=
  lion_lies (yesterday d)

/-- Define the unicorn's statement: "Yesterday was a day when I lied." -/
def unicorn_statement (d : Day) : Prop :=
  unicorn_lies (yesterday d)

/-- Prove that today must be Thursday given the conditions and statements. -/
theorem determine_day_from_statements (d : Day) :
    lion_statement d ∧ unicorn_statement d → d = Thursday := by
  sorry

end determine_day_from_statements_l306_306227


namespace actual_price_of_good_l306_306944

theorem actual_price_of_good (P : ℝ) 
  (hp : 0.684 * P = 6500) : P = 9502.92 :=
by 
  sorry

end actual_price_of_good_l306_306944


namespace drivers_schedule_l306_306600

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end drivers_schedule_l306_306600


namespace distance_between_pulley_centers_l306_306825

theorem distance_between_pulley_centers (R1 R2 CD : ℝ) (R1_pos : R1 = 10) (R2_pos : R2 = 6) (CD_pos : CD = 30) :
  ∃ AB : ℝ, AB = 2 * Real.sqrt 229 :=
by
  sorry

end distance_between_pulley_centers_l306_306825


namespace henry_age_is_20_l306_306785

open Nat

def sum_ages (H J : ℕ) : Prop := H + J = 33
def age_relation (H J : ℕ) : Prop := H - 6 = 2 * (J - 6)

theorem henry_age_is_20 (H J : ℕ) (h1 : sum_ages H J) (h2 : age_relation H J) : H = 20 :=
by
  -- Proof goes here
  sorry

end henry_age_is_20_l306_306785


namespace find_k_value_l306_306886

theorem find_k_value :
  (∃ p q : ℝ → ℝ,
    (∀ x, p x = 3 * x + 5) ∧
    (∃ k : ℝ, (∀ x, q x = k * x + 3) ∧
      (p (-4) = -7) ∧ (q (-4) = -7) ∧ k = 2.5)) :=
by
  sorry

end find_k_value_l306_306886


namespace angle_equiv_330_neg390_l306_306822

theorem angle_equiv_330_neg390 : ∃ k : ℤ, 330 = -390 + 360 * k :=
by
  sorry

end angle_equiv_330_neg390_l306_306822


namespace hyperbola_eccentricity_l306_306588

theorem hyperbola_eccentricity 
  (p1 p2 : ℝ × ℝ)
  (asymptote_passes_through_p1 : p1 = (1, 2))
  (hyperbola_passes_through_p2 : p2 = (2 * Real.sqrt 2, 4)) :
  ∃ e : ℝ, e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l306_306588


namespace young_or_old_woman_lawyer_probability_l306_306210

/-- 
40 percent of the members of a study group are women.
Among these women, 30 percent are young lawyers.
10 percent are old lawyers.
Prove the probability that a member randomly selected is a young or old woman lawyer is 0.16.
-/
theorem young_or_old_woman_lawyer_probability :
  let total_members := 100
  let women_percentage := 40
  let young_lawyers_percentage := 30
  let old_lawyers_percentage := 10
  let total_women := (women_percentage * total_members) / 100
  let young_women_lawyers := (young_lawyers_percentage * total_women) / 100
  let old_women_lawyers := (old_lawyers_percentage * total_women) / 100
  let women_lawyers := young_women_lawyers + old_women_lawyers
  let probability := women_lawyers / total_members
  probability = 0.16 := 
by {
  sorry
}

end young_or_old_woman_lawyer_probability_l306_306210


namespace minimum_value_fraction_l306_306546

-- Define the conditions in Lean
theorem minimum_value_fraction
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (line_through_center : ∀ x y, x = 1 ∧ y = -2 → a * x - b * y - 1 = 0) :
  (2 / a + 1 / b) = 8 := 
sorry

end minimum_value_fraction_l306_306546


namespace triangle_DBN_side_lengths_and_trig_values_l306_306557

-- Given conditions
def is_square (ABCD : ℝ) := ABCD = 1
def is_equilateral (BCE : ℝ) := BCE = 1
def midpoint (M CE : ℝ) := M = CE / 2
def perpendicular (DN BM CE : ℝ) := BM ⊥ CE ∧ DN ⊥ BM

theorem triangle_DBN_side_lengths_and_trig_values :
  (is_square 1) →
  (is_equilateral 1) →
  (midpoint 0.5 1) →
  (perpendicular 0 0 1) →
  (
    let DB := real.sqrt 2,
    let DN := (real.sqrt 3 + 1) / 2,
    let BN := (real.sqrt 3 - 1) / 2,
    let cos₇₅ := (real.sqrt 6 - real.sqrt 2) / 4,
    let sin₇₅ := (real.sqrt 6 + real.sqrt 2) / 4,
    let tan₇₅ := (real.sqrt 6 + real.sqrt 2) / (real.sqrt 6 - real.sqrt 2),
    let cos₁₅ := (real.sqrt 6 + real.sqrt 2) / 4,
    let sin₁₅ := (real.sqrt 6 - real.sqrt 2) / 4,
    let tan₁₅ := (real.sqrt 6 - real.sqrt 2) / (real.sqrt 6 + real.sqrt 2)
  ) →
  (
    DB = real.sqrt 2 ∧
    DN = (real.sqrt 3 + 1) / 2 ∧
    BN = (real.sqrt 3 - 1) / 2 ∧
    cos₇₅ = (real.sqrt 6 - real.sqrt 2) / 4 ∧
    sin₇₅ = (real.sqrt 6 + real.sqrt 2) / 4 ∧
    tan₇₅ = (real.sqrt 6 + real.sqrt 2) / (real.sqrt 6 - real.sqrt 2) ∧
    cos₁₅ = (real.sqrt 6 + real.sqrt 2) / 4 ∧
    sin₁₅ = (real.sqrt 6 - real.sqrt 2) / 4 ∧
    tan₁₅ = (real.sqrt 6 - real.sqrt 2) / (real.sqrt 6 + real.sqrt 2)
  ) :=
by sorry

end triangle_DBN_side_lengths_and_trig_values_l306_306557


namespace max_side_of_triangle_l306_306051

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306051


namespace translation_vector_condition_l306_306911

theorem translation_vector_condition (m n : ℝ) :
  (∀ x : ℝ, 2 * (x - m) + n = 2 * x + 5) → n = 2 * m + 5 :=
by
  intro h
  -- proof can be filled here
  sorry

end translation_vector_condition_l306_306911


namespace center_of_symmetry_l306_306360

open Real

theorem center_of_symmetry:
  (∃ x: ℝ, ∃ y: ℝ, y = sin (2 * x) + cos (2 * x) ∧ y = 0) → 
  ∃ x: ℝ, x = (3 * π / 8) :=
by
  sorry

end center_of_symmetry_l306_306360


namespace inversely_proportional_percentage_change_l306_306656

variable {x y k : ℝ}
variable (a b : ℝ)

/-- Given that x and y are positive numbers and inversely proportional,
if x increases by a% and y decreases by b%, then b = 100a / (100 + a) -/
theorem inversely_proportional_percentage_change
  (hx : 0 < x) (hy : 0 < y) (hinv : y = k / x)
  (ha : 0 < a) (hb : 0 < b)
  (hchange : ((1 + a / 100) * x) * ((1 - b / 100) * y) = k) :
  b = 100 * a / (100 + a) :=
sorry

end inversely_proportional_percentage_change_l306_306656


namespace paths_A_to_C_l306_306113

theorem paths_A_to_C :
  let paths_AB := 2
  let paths_BD := 3
  let paths_DC := 3
  let paths_AC_direct := 1
  paths_AB * paths_BD * paths_DC + paths_AC_direct = 19 :=
by
  sorry

end paths_A_to_C_l306_306113


namespace max_triangle_side_24_l306_306080

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306080


namespace max_side_length_of_triangle_l306_306065

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306065


namespace determine_k_and_a_n_and_T_n_l306_306986

noncomputable def S_n (n : ℕ) (k : ℝ) : ℝ := -0.5 * n^2 + k * n

/-- Given the sequence S_n with sum of the first n terms S_n := -1/2 n^2 + k*n,
where k is a positive natural number. The maximum value of S_n is 8. -/
theorem determine_k_and_a_n_and_T_n (k : ℝ) (h : k = 4) :
  (∀ n : ℕ, S_n n k ≤ 8) ∧ 
  (∀ n : ℕ, ∃ a : ℝ, a = 9/2 - n) ∧
  (∀ n : ℕ, ∃ T : ℝ, T = 4 - (n + 2)/2^(n-1)) :=
by
  sorry

end determine_k_and_a_n_and_T_n_l306_306986


namespace inequality_hold_l306_306428

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l306_306428


namespace round_to_nearest_tenth_l306_306898

theorem round_to_nearest_tenth : 
  let x := 36.89753 
  let tenth_place := 8
  let hundredth_place := 9
  (hundredth_place > 5) → (Float.round (10 * x) / 10 = 36.9) := 
by
  intros x tenth_place hundredth_place h
  sorry

end round_to_nearest_tenth_l306_306898


namespace total_ducats_is_160_l306_306112

variable (T : ℤ) (a b c d e : ℤ) -- Variables to represent the amounts taken by the robbers

-- Conditions
axiom h1 : a = 81                                            -- The strongest robber took 81 ducats
axiom h2 : b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e    -- Each remaining robber took a different amount
axiom h3 : a + b + c + d + e = T                             -- Total amount of ducats
axiom redistribution : 
  -- Redistribution process leads to each robber having the same amount
  2*b + 2*c + 2*d + 2*e = T ∧
  2*(2*c + 2*d + 2*e) = T ∧
  2*(2*(2*d + 2*e)) = T ∧
  2*(2*(2*(2*e))) = T

-- Proof that verifies the total ducats is 160
theorem total_ducats_is_160 : T = 160 :=
by
  sorry

end total_ducats_is_160_l306_306112


namespace find_a_l306_306720

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(2 * x - 1) + 3 else 1 - Real.log 2 x

theorem find_a (a : ℝ) (h : f a = 4) : a = 1/8 :=
by
  sorry

end find_a_l306_306720


namespace max_triangle_side_24_l306_306079

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306079


namespace number_of_classes_l306_306524

theorem number_of_classes (sheets_per_class_per_day : ℕ) 
                          (total_sheets_per_week : ℕ) 
                          (school_days_per_week : ℕ) 
                          (h1 : sheets_per_class_per_day = 200) 
                          (h2 : total_sheets_per_week = 9000) 
                          (h3 : school_days_per_week = 5) : 
                          total_sheets_per_week / school_days_per_week / sheets_per_class_per_day = 9 :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_classes_l306_306524


namespace smallest_k_divides_polynomial_l306_306110

noncomputable def is_divisible (f g : polynomial complex) : Prop :=
∃ q, f = q * g

theorem smallest_k_divides_polynomial:
  let f := (polynomial.C (1:complex) * polynomial.X ^ 12 +
            polynomial.C (1:complex) * polynomial.X ^ 11 +
            polynomial.C (1:complex) * polynomial.X ^ 8 +
            polynomial.C (1:complex) * polynomial.X ^ 7 +
            polynomial.C (1:complex) * polynomial.X ^ 6 +
            polynomial.C (1:complex) * polynomial.X ^ 3 +
            polynomial.C (1:complex)) in
  ∃ k : ℕ, k > 0 ∧ is_divisible (polynomial.X ^ k - 1) f ∧ k = 120 :=
begin
  intros f,
  use [120],
  split,
  { linarith, },
  split,
  { sorry, },
  { refl, },
end

end smallest_k_divides_polynomial_l306_306110


namespace morgan_change_l306_306889

-- Define the costs of the items and the amount paid
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def amount_paid : ℕ := 20

-- Define total cost
def total_cost := hamburger_cost + onion_rings_cost + smoothie_cost

-- Define the change received
def change_received := amount_paid - total_cost

-- Statement of the problem in Lean 4
theorem morgan_change : change_received = 11 := by
  -- include proof steps here
  sorry

end morgan_change_l306_306889


namespace phone_answer_prob_within_four_rings_l306_306353

def prob_first_ring : ℚ := 1/10
def prob_second_ring : ℚ := 1/5
def prob_third_ring : ℚ := 3/10
def prob_fourth_ring : ℚ := 1/10

theorem phone_answer_prob_within_four_rings :
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring = 7/10 :=
by
  sorry

end phone_answer_prob_within_four_rings_l306_306353


namespace ladder_base_distance_l306_306348

theorem ladder_base_distance (h : real) (L : real) (H : real) (B : real) 
  (h_eq : h = 12) (L_eq : L = 15) (h_sq_plus_b_sq : h^2 + B^2 = L^2) : 
  B = 9 :=
by
  rw [h_eq, L_eq] at h_sq_plus_b_sq
  sorry

end ladder_base_distance_l306_306348


namespace product_of_repeating_decimal_l306_306967

noncomputable def repeating_decimal := 1357 / 9999
def product_with_7 (x : ℚ) := 7 * x

theorem product_of_repeating_decimal :
  product_with_7 repeating_decimal = 9499 / 9999 :=
by sorry

end product_of_repeating_decimal_l306_306967


namespace smallest_prime_sum_of_five_distinct_primes_l306_306795

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def distinct (a b c d e : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem smallest_prime_sum_of_five_distinct_primes :
  ∃ a b c d e : ℕ, is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ distinct a b c d e ∧ (a + b + c + d + e = 43) ∧ is_prime 43 :=
sorry

end smallest_prime_sum_of_five_distinct_primes_l306_306795


namespace symmetric_graph_l306_306569

variable (f : ℝ → ℝ)
variable (c : ℝ)
variable (h_nonzero : c ≠ 0)
variable (h_fx_plus_y : ∀ (x y : ℝ), f (x + y) + f (x - y) = 2 * f x * f y)
variable (h_f_half_c : f (c / 2) = 0)
variable (h_f_zero : f 0 ≠ 0)

theorem symmetric_graph (k : ℤ) : 
  ∀ (x : ℝ), f (x) = f (2*k*c - x) :=
sorry

end symmetric_graph_l306_306569


namespace max_side_length_l306_306011

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306011


namespace trenton_commission_rate_l306_306327

noncomputable def commission_rate (fixed_earnings : ℕ) (goal : ℕ) (sales : ℕ) : ℚ :=
  ((goal - fixed_earnings : ℤ) / (sales : ℤ)) * 100

theorem trenton_commission_rate :
  commission_rate 190 500 7750 = 4 := 
  by
  sorry

end trenton_commission_rate_l306_306327


namespace max_side_of_triangle_l306_306063

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306063


namespace translate_parabola_l306_306622

-- Translating the parabola y = (x-2)^2 - 8 three units left and five units up
theorem translate_parabola (x y : ℝ) :
  y = (x - 2) ^ 2 - 8 →
  y = ((x + 3) - 2) ^ 2 - 8 + 5 →
  y = (x + 1) ^ 2 - 3 := by
sorry

end translate_parabola_l306_306622


namespace percent_equivalence_l306_306492

theorem percent_equivalence (y : ℝ) (h : y ≠ 0) : 0.21 * y = 0.21 * y :=
by sorry

end percent_equivalence_l306_306492


namespace base12_addition_l306_306225

theorem base12_addition : ∀ a b : ℕ, a = 956 ∧ b = 273 → (a + b) = 1009 := by
  sorry

end base12_addition_l306_306225


namespace quadratic_polynomial_value_l306_306711

noncomputable def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem quadratic_polynomial_value (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : f 1 (-(a + b + c)) (a*b + b*c + a*c) a = b * c)
  (h2 : f 1 (-(a + b + c)) (a*b + b*c + a*c) b = c * a)
  (h3 : f 1 (-(a + b + c)) (a*b + b*c + a*c) c = a * b) :
  f 1 (-(a + b + c)) (a*b + b*c + a*c) (a + b + c) = a * b + b * c + a * c := sorry

end quadratic_polynomial_value_l306_306711


namespace problem_l306_306128

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
sorry

end problem_l306_306128


namespace provisions_last_days_after_reinforcement_l306_306956

-- Definitions based on the conditions
def initial_men := 2000
def initial_days := 40
def reinforcement_men := 2000
def days_passed := 20

-- Calculate the total provisions initially
def total_provisions := initial_men * initial_days

-- Calculate the remaining provisions after some days passed
def remaining_provisions := total_provisions - (initial_men * days_passed)

-- Total number of men after reinforcement
def total_men := initial_men + reinforcement_men

-- The Lean statement proving the duration the remaining provisions will last
theorem provisions_last_days_after_reinforcement :
  remaining_provisions / total_men = 10 := by
  sorry

end provisions_last_days_after_reinforcement_l306_306956


namespace min_value_function_l306_306251

theorem min_value_function (x y: ℝ) (hx: x > 2) (hy: y > 2) : 
  (∃c: ℝ, c = (x^3/(y - 2) + y^3/(x - 2)) ∧ ∀x y: ℝ, x > 2 → y > 2 → (x^3/(y - 2) + y^3/(x - 2)) ≥ c) ∧ c = 96 :=
sorry

end min_value_function_l306_306251


namespace inequality_proof_l306_306429

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l306_306429


namespace distance_to_Rock_Mist_Mountains_l306_306963

theorem distance_to_Rock_Mist_Mountains
  (d_Sky_Falls : ℕ) (d_Sky_Falls_eq : d_Sky_Falls = 8)
  (d_Rock_Mist : ℕ) (d_Rock_Mist_eq : d_Rock_Mist = 50 * d_Sky_Falls)
  (detour_Thunder_Pass : ℕ) (detour_Thunder_Pass_eq : detour_Thunder_Pass = 25) :
  d_Rock_Mist + detour_Thunder_Pass = 425 := by
  sorry

end distance_to_Rock_Mist_Mountains_l306_306963


namespace max_sum_n_of_arithmetic_sequence_l306_306460

/-- Let \( S_n \) be the sum of the first \( n \) terms of an arithmetic sequence \( \{a_n\} \) with 
a non-zero common difference, and \( a_1 > 0 \). If \( S_5 = S_9 \), then when \( S_n \) is maximum, \( n = 7 \). -/
theorem max_sum_n_of_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (a1_pos : a 1 > 0) (common_difference : ∀ n, a (n + 1) = a n + d)
  (s5_eq_s9 : S 5 = S 9) :
  ∃ n, (∀ m, m ≤ n → S m ≤ S n) ∧ n = 7 :=
sorry

end max_sum_n_of_arithmetic_sequence_l306_306460


namespace ninth_graders_science_only_l306_306594

theorem ninth_graders_science_only 
    (total_students : ℕ := 120)
    (science_students : ℕ := 80)
    (programming_students : ℕ := 75) 
    : (science_students - (science_students + programming_students - total_students)) = 45 :=
by
  sorry

end ninth_graders_science_only_l306_306594


namespace max_side_of_triangle_l306_306047

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306047


namespace sequence_is_k_plus_n_l306_306694

theorem sequence_is_k_plus_n (a : ℕ → ℕ) (k : ℕ) (h : ∀ n : ℕ, a (n + 2) * (a (n + 1) - 1) = a n * (a (n + 1) + 1))
  (pos: ∀ n: ℕ, a n > 0) : ∀ n: ℕ, a n = k + n := 
sorry

end sequence_is_k_plus_n_l306_306694


namespace quadratic_polynomial_value_l306_306712

noncomputable def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem quadratic_polynomial_value (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : f 1 (-(a + b + c)) (a*b + b*c + a*c) a = b * c)
  (h2 : f 1 (-(a + b + c)) (a*b + b*c + a*c) b = c * a)
  (h3 : f 1 (-(a + b + c)) (a*b + b*c + a*c) c = a * b) :
  f 1 (-(a + b + c)) (a*b + b*c + a*c) (a + b + c) = a * b + b * c + a * c := sorry

end quadratic_polynomial_value_l306_306712


namespace integer_divisibility_l306_306151

open Nat

theorem integer_divisibility (n : ℕ) (h1 : ∃ m : ℕ, 2^n - 2 = n * m) : ∃ k : ℕ, 2^((2^n) - 1) - 2 = (2^n - 1) * k := by
  sorry

end integer_divisibility_l306_306151


namespace morse_code_count_l306_306996

noncomputable def morse_code_sequences : Nat :=
  let case_1 := 2            -- 1 dot or dash
  let case_2 := 2 * 2        -- 2 dots or dashes
  let case_3 := 2 * 2 * 2    -- 3 dots or dashes
  let case_4 := 2 * 2 * 2 * 2-- 4 dots or dashes
  let case_5 := 2 * 2 * 2 * 2 * 2 -- 5 dots or dashes
  case_1 + case_2 + case_3 + case_4 + case_5

theorem morse_code_count : morse_code_sequences = 62 := by
  sorry

end morse_code_count_l306_306996


namespace number_of_machines_sold_l306_306960

-- Define the parameters and conditions given in the problem
def commission_of_first_150 (sale_price : ℕ) : ℕ := 150 * (sale_price * 3 / 100)
def commission_of_next_100 (sale_price : ℕ) : ℕ := 100 * (sale_price * 4 / 100)
def commission_of_after_250 (sale_price : ℕ) (x : ℕ) : ℕ := x * (sale_price * 5 / 100)

-- Define the total commission using these commissions
def total_commission (x : ℕ) : ℕ :=
  commission_of_first_150 10000 + 
  commission_of_next_100 9500 + 
  commission_of_after_250 9000 x

-- The main statement we want to prove
theorem number_of_machines_sold (x : ℕ) (total_commission : ℕ) : x = 398 ↔ total_commission = 150000 :=
by
  sorry

end number_of_machines_sold_l306_306960


namespace school_xx_percentage_increase_l306_306779

theorem school_xx_percentage_increase
  (X Y : ℕ) -- denote the number of students at school XX and YY last year
  (H_Y : Y = 2400) -- condition: school YY had 2400 students last year
  (H_total : X + Y = 4000) -- condition: total number of students last year was 4000
  (H_increase_YY : YY_increase = (3 * Y) / 100) -- condition: 3 percent increase at school YY
  (H_difference : XX_increase = YY_increase + 40) -- condition: school XX grew by 40 more students than YY
  : (XX_increase * 100) / X = 7 :=
by
  sorry

end school_xx_percentage_increase_l306_306779


namespace complement_M_intersect_N_l306_306570

def M : Set ℤ := {m | m ≤ -3 ∨ m ≥ 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}
def complement_M : Set ℤ := {m | -3 < m ∧ m < 2} 

theorem complement_M_intersect_N : (complement_M ∩ N) = {-1, 0, 1} := by
  sorry

end complement_M_intersect_N_l306_306570


namespace current_number_of_people_l306_306789

theorem current_number_of_people (a b : ℕ) : 0 ≤ a → 0 ≤ b → 48 - a + b ≥ 0 := by
  sorry

end current_number_of_people_l306_306789


namespace common_number_is_eight_l306_306931

theorem common_number_is_eight (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d) / 4 = 7)
  (h2 : (d + e + f + g) / 4 = 9)
  (h3 : (a + b + c + d + e + f + g) / 7 = 8) :
  d = 8 :=
by
sorry

end common_number_is_eight_l306_306931


namespace rational_solutions_quadratic_eq_l306_306409

theorem rational_solutions_quadratic_eq (k : ℕ) (h_pos : k > 0) :
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ (k = 8 ∨ k = 12) :=
by sorry

end rational_solutions_quadratic_eq_l306_306409


namespace unique_solution_l306_306689

theorem unique_solution (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hn : 2 ≤ n) (h_y_bound : y ≤ 5 * 2^(2*n)) :
  x^(2*n+1) - y^(2*n+1) = x * y * z + 2^(2*n+1) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  sorry

end unique_solution_l306_306689


namespace largest_of_eight_consecutive_summing_to_5400_l306_306927

theorem largest_of_eight_consecutive_summing_to_5400 :
  ∃ (n : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 5400)
  → (n+7 = 678) :=
by 
  sorry

end largest_of_eight_consecutive_summing_to_5400_l306_306927


namespace num_valid_m_l306_306847

theorem num_valid_m (m : ℕ) : (∃ n : ℕ, n * (m^2 - 3) = 1722) → ∃ p : ℕ, p = 3 := 
  by
  sorry

end num_valid_m_l306_306847


namespace find_cows_l306_306651

variable (D C : ℕ)

theorem find_cows (h1 : 2 * D + 4 * C = 2 * (D + C) + 36) : C = 18 :=
by
  -- Proof goes here
  sorry

end find_cows_l306_306651


namespace repeating_decimal_fraction_l306_306383

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l306_306383


namespace sahil_selling_price_l306_306501

def initial_cost : ℝ := 14000
def repair_cost : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percent : ℝ := 50

noncomputable def total_cost : ℝ := initial_cost + repair_cost + transportation_charges
noncomputable def profit : ℝ := profit_percent / 100 * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem sahil_selling_price :
  selling_price = 30000 := by
  sorry

end sahil_selling_price_l306_306501


namespace repeating_decimal_eq_fraction_l306_306396

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l306_306396


namespace shirts_sold_l306_306581

theorem shirts_sold (pants shorts shirts jackets credit_remaining : ℕ) 
  (price_shirt1 price_shirt2 price_pants : ℕ) 
  (discount tax : ℝ) :
  (pants = 3) →
  (shorts = 5) →
  (jackets = 2) →
  (price_shirt1 = 10) →
  (price_shirt2 = 12) →
  (price_pants = 15) →
  (discount = 0.10) →
  (tax = 0.05) →
  (credit_remaining = 25) →
  (store_credit : ℕ) →
  (store_credit = pants * 5 + shorts * 3 + jackets * 7 + shirts * 4) →
  (total_cost : ℝ) →
  (total_cost = (price_shirt1 + price_shirt2 + price_pants) * (1 - discount) * (1 + tax)) →
  (total_store_credit_used : ℝ) →
  (total_store_credit_used = total_cost - credit_remaining) →
  (initial_credit : ℝ) →
  (initial_credit = total_store_credit_used + (pants * 5 + shorts * 3 + jackets * 7)) →
  shirts = 2 :=
by
  intros
  sorry

end shirts_sold_l306_306581


namespace max_side_of_triangle_l306_306042

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306042


namespace simplify_fraction_addition_l306_306306

theorem simplify_fraction_addition : (3 : ℚ) / 462 + 13 / 42 = 73 / 231 :=
by
  sorry

end simplify_fraction_addition_l306_306306


namespace perpendicular_lines_have_a_zero_l306_306717

theorem perpendicular_lines_have_a_zero {a : ℝ} :
  ∀ x y : ℝ, (ax + y - 1 = 0) ∧ (x + a*y - 1 = 0) → a = 0 :=
by
  sorry

end perpendicular_lines_have_a_zero_l306_306717


namespace simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l306_306583

-- Problem 1
theorem simplify_expression1 (a b : ℝ) : 
  4 * a^3 + 2 * b - 2 * a^3 + b = 2 * a^3 + 3 * b := 
sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  2 * x^2 + 6 * x - 6 - (-2 * x^2 + 4 * x + 1) = 4 * x^2 + 2 * x - 7 := 
sorry

-- Problem 3
theorem simplify_expression3 (a b : ℝ) : 
  3 * (3 * a^2 - 2 * a * b) - 2 * (4 * a^2 - a * b) = a^2 - 4 * a * b := 
sorry

-- Problem 4
theorem simplify_expression4 (x y : ℝ) : 
  6 * x * y^2 - (2 * x - (1 / 2) * (2 * x - 4 * x * y^2) - x * y^2) = 5 * x * y^2 - x := 
sorry

end simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l306_306583


namespace proof_problem_l306_306297

def f (a b c : ℕ) : ℕ :=
  a * 100 + b * 10 + c

def special_op (a b c : ℕ) : ℕ :=
  f (a * b) (b * c / 10) (b * c % 10)

theorem proof_problem :
  special_op 5 7 4 - special_op 7 4 5 = 708 := 
    sorry

end proof_problem_l306_306297


namespace inradius_of_right_triangle_l306_306508

-- Define the side lengths of the triangle
def a : ℕ := 9
def b : ℕ := 40
def c : ℕ := 41

-- Define the semiperimeter of the triangle
def s : ℕ := (a + b + c) / 2

-- Define the area of a right triangle
def A : ℕ := (a * b) / 2

-- Define the inradius of the triangle
def inradius : ℕ := A / s

theorem inradius_of_right_triangle : inradius = 4 :=
by
  -- The proof is omitted since only the statement is requested
  sorry

end inradius_of_right_triangle_l306_306508


namespace max_side_length_is_11_l306_306015

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306015


namespace cricketer_hits_two_sixes_l306_306214

-- Definitions of the given conditions
def total_runs : ℕ := 132
def boundaries_count : ℕ := 12
def running_percent : ℚ := 54.54545454545454 / 100

-- Function to calculate runs made by running
def runs_by_running (total: ℕ) (percent: ℚ) : ℚ :=
  percent * total

-- Function to calculate runs made from boundaries
def runs_from_boundaries (count: ℕ) : ℕ :=
  count * 4

-- Function to calculate runs made from sixes
def runs_from_sixes (total: ℕ) (boundaries_runs: ℕ) (running_runs: ℚ) : ℚ :=
  total - boundaries_runs - running_runs

-- Function to calculate number of sixes hit
def number_of_sixes (sixes_runs: ℚ) : ℚ :=
  sixes_runs / 6

-- The proof statement for the cricketer hitting 2 sixes
theorem cricketer_hits_two_sixes:
  number_of_sixes (runs_from_sixes total_runs (runs_from_boundaries boundaries_count) (runs_by_running total_runs running_percent)) = 2 := by
  sorry

end cricketer_hits_two_sixes_l306_306214


namespace tallest_player_height_correct_l306_306322

-- Define the height of the shortest player
def shortest_player_height : ℝ := 68.25

-- Define the height difference between the tallest and shortest player
def height_difference : ℝ := 9.5

-- Define the height of the tallest player based on the conditions
def tallest_player_height : ℝ :=
  shortest_player_height + height_difference

-- Theorem statement
theorem tallest_player_height_correct : tallest_player_height = 77.75 := by
  sorry

end tallest_player_height_correct_l306_306322


namespace correct_average_l306_306203

theorem correct_average (n : ℕ) (average incorrect correct : ℕ) (h1 : n = 10) (h2 : average = 15) 
(h3 : incorrect = 26) (h4 : correct = 36) :
  (n * average - incorrect + correct) / n = 16 :=
  sorry

end correct_average_l306_306203


namespace max_side_length_of_integer_triangle_with_perimeter_24_l306_306025

theorem max_side_length_of_integer_triangle_with_perimeter_24
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : a + b + c = 24)
  (h4 : a ≠ b) 
  (h5 : b ≠ c) 
  (h6 : a ≠ c) 
  : c ≤ 11 :=
begin
  sorry
end

end max_side_length_of_integer_triangle_with_perimeter_24_l306_306025


namespace train_speed_l306_306962

theorem train_speed :
  ∃ V : ℝ,
    (∃ L : ℝ, L = V * 18) ∧ 
    (∃ L : ℝ, L + 260 = V * 31) ∧ 
    V * 3.6 = 72 := by
  sorry

end train_speed_l306_306962


namespace fraction_repeating_decimal_l306_306375

theorem fraction_repeating_decimal : ∃ (r : ℚ), r = (0.36 : ℚ) ∧ r = 4 / 11 :=
by
  -- The decimal number 0.36 recurring is represented by the infinite series
  have h_series : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 0.36 := sorry,
  -- Using the sum formula for the infinite series
  have h_sum : ∑' n : ℕ, (36:ℚ) / 10^(2*(n+1)) = 4 / 11 := sorry,
  -- Combining the results
  use 4 / 11,
  split,
  { exact h_series },
  { exact h_sum }

end fraction_repeating_decimal_l306_306375


namespace worker_overtime_hours_l306_306222

theorem worker_overtime_hours :
  ∃ (x y : ℕ), 60 * x + 90 * y = 3240 ∧ x + y = 50 ∧ y = 8 :=
by
  sorry

end worker_overtime_hours_l306_306222


namespace units_digit_of_7_pow_6_pow_5_l306_306102

theorem units_digit_of_7_pow_6_pow_5 : ((7 : ℕ)^ (6^5) % 10) = 1 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l306_306102


namespace limonia_largest_unachievable_l306_306747

noncomputable def largest_unachievable_amount (n : ℕ) : ℕ :=
  12 * n^2 + 14 * n - 1

theorem limonia_largest_unachievable (n : ℕ) :
  ∀ k, ¬ ∃ a b c d : ℕ, 
    k = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10) 
    → k = largest_unachievable_amount n :=
sorry

end limonia_largest_unachievable_l306_306747


namespace total_marks_secured_l306_306157

-- Define the conditions
def correct_points_per_question := 4
def wrong_points_per_question := 1
def total_questions := 60
def correct_questions := 40

-- Calculate the remaining incorrect questions
def wrong_questions := total_questions - correct_questions

-- Calculate total marks secured by the student
def total_marks := (correct_questions * correct_points_per_question) - (wrong_questions * wrong_points_per_question)

-- The statement to be proven
theorem total_marks_secured : total_marks = 140 := by
  -- This will be proven in Lean's proof assistant
  sorry

end total_marks_secured_l306_306157


namespace cookies_in_jar_l306_306448

-- Let C be the total number of cookies in the jar.
def C : ℕ := sorry

-- Conditions
def adults_eat_one_third (C : ℕ) : ℕ := C / 3
def children_get_each (C : ℕ) : ℕ := 20
def num_children : ℕ := 4

-- Proof statement
theorem cookies_in_jar (C : ℕ) (h1 : C / 3 = adults_eat_one_third C)
  (h2 : children_get_each C * num_children = 80)
  (h3 : 2 * (C / 3) = 80) :
  C = 120 :=
sorry

end cookies_in_jar_l306_306448


namespace hostel_provisions_l306_306806

theorem hostel_provisions (x : ℕ) (h1 : 250 * x = 200 * 40) : x = 32 :=
by
  sorry

end hostel_provisions_l306_306806


namespace max_deflection_angle_l306_306663

variable (M m : ℝ)
variable (h : M > m)

theorem max_deflection_angle :
  ∃ α : ℝ, α = Real.arcsin (m / M) := by
  sorry

end max_deflection_angle_l306_306663


namespace solid_is_cylinder_l306_306928

def solid_views (v1 v2 v3 : String) : Prop := 
  -- This definition makes a placeholder for the views of the solid.
  sorry

def is_cylinder (s : String) : Prop := 
  s = "Cylinder"

theorem solid_is_cylinder (v1 v2 v3 : String) (h : solid_views v1 v2 v3) :
  ∃ s : String, is_cylinder s :=
sorry

end solid_is_cylinder_l306_306928


namespace value_of_b_l306_306443

theorem value_of_b (x b : ℝ) (h₁ : x = 0.3) 
  (h₂ : (10 * x + b) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3) : 
  b = 2 :=
by
  sorry

end value_of_b_l306_306443


namespace find_breadth_of_rectangle_l306_306946

noncomputable def breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) (breadth : ℝ) : Prop :=
  A = length_to_breadth_ratio * breadth * breadth → breadth = 20

-- Now we can state the theorem.
theorem find_breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) : breadth_of_rectangle A length_to_breadth_ratio 20 :=
by
  intros h
  sorry

end find_breadth_of_rectangle_l306_306946


namespace part1_part2_l306_306266

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (h : m > 1) : ∃ x : ℝ, f x = (4 / (m - 1)) + m :=
by
  sorry

end part1_part2_l306_306266


namespace alice_has_ball_after_two_turns_l306_306821

noncomputable def prob_alice_has_ball_after_two_turns : ℚ :=
  let p_A_B := (3 : ℚ) / 5 -- Probability Alice tosses to Bob
  let p_B_A := (1 : ℚ) / 3 -- Probability Bob tosses to Alice
  let p_A_A := (2 : ℚ) / 5 -- Probability Alice keeps the ball
  (p_A_B * p_B_A) + (p_A_A * p_A_A)

theorem alice_has_ball_after_two_turns :
  prob_alice_has_ball_after_two_turns = 9 / 25 :=
by
  -- skipping the proof
  sorry

end alice_has_ball_after_two_turns_l306_306821


namespace gcd_problem_l306_306418

theorem gcd_problem 
  (b : ℤ) 
  (hb_odd : b % 2 = 1) 
  (hb_multiples_of_8723 : ∃ (k : ℤ), b = 8723 * k) : 
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 15) = 3 := 
by 
  sorry

end gcd_problem_l306_306418


namespace cubic_inequality_l306_306835

theorem cubic_inequality :
  {x : ℝ | x^3 - 12*x^2 + 47*x - 60 < 0} = {x : ℝ | 3 < x ∧ x < 5} :=
by
  sorry

end cubic_inequality_l306_306835


namespace proof_problem_l306_306854

theorem proof_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c * d = 1) : 
  (a * c + b * d) * (a * d + b * c) ≥ 1 := 
by 
  sorry

end proof_problem_l306_306854


namespace dawn_annual_salary_l306_306367

variable (M : ℝ)

theorem dawn_annual_salary (h1 : 0.10 * M = 400) : M * 12 = 48000 := by
  sorry

end dawn_annual_salary_l306_306367


namespace positive_integer_divisibility_l306_306410

theorem positive_integer_divisibility (n : ℕ) (h_pos : n > 0) (h_div : (n^2 + 1) ∣ (n + 1)) : n = 1 := 
sorry

end positive_integer_divisibility_l306_306410


namespace knights_divisible_by_4_l306_306247

-- Define the conditions: Assume n is the total number of knights (n > 0).
-- Condition 1: Knights from two opposing clans A and B
-- Condition 2: Number of knights with an enemy to the right equals number of knights with a friend to the right.

open Nat

theorem knights_divisible_by_4 (n : ℕ) (h1 : 0 < n)
  (h2 : ∃k : ℕ, 2 * k = n ∧ ∀ (i : ℕ), (i < n → ((i % 2 = 0 → (i+1) % 2 = 1) ∧ (i % 2 = 1 → (i+1) % 2 = 0)))) :
  n % 4 = 0 :=
sorry

end knights_divisible_by_4_l306_306247


namespace book_pairs_count_l306_306139

theorem book_pairs_count :
  let mystery_books := 3
  let fantasy_books := 3
  let biography_books := 3
  let science_fiction_books := 3
  let genres := 4
  let genre_pairs := Nat.choose genres 2
  let books_in_pair := mystery_books * fantasy_books  
  genre_pairs * books_in_pair = 54 :=
by
  let mystery_books := 3
  let fantasy_books := 3
  let biography_books := 3
  let science_fiction_books := 3
  let genres := 4
  let genre_pairs := Nat.choose genres 2 -- 6 ways to choose 2 genres out of 4
  let books_in_pair := mystery_books * fantasy_books -- 9 ways to choose 1 book from each genre
  have genre_pairs_eq : genre_pairs = 6 := by simp [genre_pairs]
  have books_in_pair_eq : books_in_pair = 9 := by simp [books_in_pair]
  have total_pairs := genre_pairs * books_in_pair
  have total_pairs_eq : total_pairs = 6 * 9 := by simp [total_pairs]
  have result : total_pairs = 54 := by simp [total_pairs_eq]
  exact result

end book_pairs_count_l306_306139


namespace inequality_pos_real_l306_306760

theorem inequality_pos_real (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c)) ≥ (2 / 3) := 
sorry

end inequality_pos_real_l306_306760


namespace sufficient_and_necessary_condition_l306_306850

variable (a_n : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
variable (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
variable (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d))

theorem sufficient_and_necessary_condition (d : ℚ) (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d)) :
  (d > 0) ↔ (S 4 + S 6 > 2 * S 5) := by
  sorry

end sufficient_and_necessary_condition_l306_306850


namespace initial_money_l306_306526

theorem initial_money (M : ℝ) (h1 : M - (1/4 * M) - (1/3 * (M - (1/4 * M))) = 1600) : M = 3200 :=
sorry

end initial_money_l306_306526


namespace time_jogging_l306_306730

def distance := 25     -- Distance jogged (in kilometers)
def speed := 5        -- Speed (in kilometers per hour)

theorem time_jogging :
  (distance / speed) = 5 := 
by
  sorry

end time_jogging_l306_306730


namespace max_side_of_triangle_l306_306041

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306041


namespace repeating_decimal_to_fraction_l306_306391

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l306_306391


namespace gain_percent_calculation_l306_306138

theorem gain_percent_calculation (gain_paise : ℕ) (cost_price_rupees : ℕ) (rupees_to_paise : ℕ)
  (h_gain_paise : gain_paise = 70)
  (h_cost_price_rupees : cost_price_rupees = 70)
  (h_rupees_to_paise : rupees_to_paise = 100) :
  ((gain_paise / rupees_to_paise) / cost_price_rupees) * 100 = 1 :=
by
  -- Placeholder to indicate the need for proof
  sorry

end gain_percent_calculation_l306_306138


namespace distance_between_cyclists_l306_306935

def cyclist_distance (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * t

theorem distance_between_cyclists :
  cyclist_distance 10 25 1.4285714285714286 = 50 := by
  sorry

end distance_between_cyclists_l306_306935


namespace difference_of_squares_is_39_l306_306910

theorem difference_of_squares_is_39 (L S : ℕ) (h1 : L = 8) (h2 : L - S = 3) : L^2 - S^2 = 39 :=
by
  sorry

end difference_of_squares_is_39_l306_306910


namespace max_side_length_of_triangle_l306_306068

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306068


namespace factor_polynomials_l306_306106

theorem factor_polynomials (x : ℝ) :
  (x^2 + 4 * x + 3) * (x^2 + 9 * x + 20) + (x^2 + 6 * x - 9) = 
  (x^2 + 6 * x + 6) * (x^2 + 6 * x + 3) :=
sorry

end factor_polynomials_l306_306106


namespace remainder_seven_times_quotient_l306_306638

theorem remainder_seven_times_quotient (n : ℕ) : 
  (∃ q r : ℕ, n = 23 * q + r ∧ r = 7 * q ∧ 0 ≤ r ∧ r < 23) ↔ (n = 30 ∨ n = 60 ∨ n = 90) :=
by 
  sorry

end remainder_seven_times_quotient_l306_306638


namespace symmetry_P_over_xOz_l306_306875

-- Definition for the point P and the plane xOz
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def P : Point3D := { x := 2, y := 3, z := 4 }

def symmetry_over_xOz_plane (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_P_over_xOz : symmetry_over_xOz_plane P = { x := 2, y := -3, z := 4 } :=
by
  -- The proof is omitted.
  sorry

end symmetry_P_over_xOz_l306_306875


namespace max_quarters_l306_306173

theorem max_quarters (total_value : ℝ) (n_quarters n_nickels n_dimes : ℕ) 
  (h1 : n_nickels = n_quarters) 
  (h2 : n_dimes = 2 * n_quarters)
  (h3 : 0.25 * n_quarters + 0.05 * n_nickels + 0.10 * n_dimes = total_value)
  (h4 : total_value = 3.80) : 
  n_quarters = 7 := 
by
  sorry

end max_quarters_l306_306173


namespace gcf_45_135_90_l306_306624

def gcd (a b : Nat) : Nat := Nat.gcd a b

noncomputable def gcd_of_three (a b c : Nat) : Nat :=
  gcd (gcd a b) c

theorem gcf_45_135_90 : gcd_of_three 45 135 90 = 45 := by
  sorry

end gcf_45_135_90_l306_306624


namespace find_a_l306_306463

noncomputable def pure_imaginary_simplification (a : ℝ) (i : ℂ) (hi : i * i = -1) : Prop :=
  let denom := (3 : ℂ) - (4 : ℂ) * i
  let numer := (15 : ℂ)
  let complex_num := a + numer / denom
  let simplified_real := a + (9 : ℝ) / (5 : ℝ)
  simplified_real = 0

theorem find_a (i : ℂ) (hi : i * i = -1) : pure_imaginary_simplification (- 9 / 5 : ℝ) i hi :=
by
  sorry

end find_a_l306_306463


namespace Nancy_weighs_90_pounds_l306_306843

theorem Nancy_weighs_90_pounds (W : ℝ) (h1 : 0.60 * W = 54) : W = 90 :=
by
  sorry

end Nancy_weighs_90_pounds_l306_306843


namespace inequality_hold_l306_306425

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l306_306425


namespace probability_of_at_least_6_consecutive_heads_l306_306954

-- Define the conditions
def flip_options : finset (fin 9 → bool) := 
  finset.univ

def at_least_6_consecutive_heads (seq : fin 9 → bool) : bool :=
  (seq 0 && seq 1 && seq 2 && seq 3 && seq 4 && seq 5) ||
  (seq 1 && seq 2 && seq 3 && seq 4 && seq 5 && seq 6) ||
  (seq 2 && seq 3 && seq 4 && seq 5 && seq 6 && seq 7) ||
  (seq 3 && seq 4 && seq 5 && seq 6 && seq 7 && seq 8)

-- Define the theorem to prove the probability
theorem probability_of_at_least_6_consecutive_heads :
  (flip_options.filter at_least_6_consecutive_heads).card = 11 / 512 :=
by
  sorry

end probability_of_at_least_6_consecutive_heads_l306_306954


namespace percentage_of_400_equals_100_l306_306403

def part : ℝ := 100
def whole : ℝ := 400

theorem percentage_of_400_equals_100 : (part / whole) * 100 = 25 := by
  sorry

end percentage_of_400_equals_100_l306_306403


namespace max_side_length_l306_306002

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306002


namespace least_positive_integer_to_multiple_of_5_l306_306338

theorem least_positive_integer_to_multiple_of_5 (n : ℕ) (h₁ : n = 725) :
  ∃ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 ∧ ∀ k : ℕ, (k > 0 ∧ (725 + k) % 5 = 0) → m ≤ k :=
begin
  use 5,
  sorry
end

end least_positive_integer_to_multiple_of_5_l306_306338


namespace sean_total_cost_l306_306580

noncomputable def total_cost (soda_cost soup_cost sandwich_cost : ℕ) (num_soda num_soup num_sandwich : ℕ) : ℕ :=
  num_soda * soda_cost + num_soup * soup_cost + num_sandwich * sandwich_cost

theorem sean_total_cost :
  let soda_cost := 1
  let soup_cost := 3 * soda_cost
  let sandwich_cost := 3 * soup_cost
  let num_soda := 3
  let num_soup := 2
  let num_sandwich := 1
  total_cost soda_cost soup_cost sandwich_cost num_soda num_soup num_sandwich = 18 :=
by
  sorry

end sean_total_cost_l306_306580


namespace totalGames_l306_306284

-- Define Jerry's original number of video games
def originalGames : ℕ := 7

-- Define the number of video games Jerry received for his birthday
def birthdayGames : ℕ := 2

-- Statement: Prove that the total number of games Jerry has now is 9
theorem totalGames : originalGames + birthdayGames = 9 := by
  sorry

end totalGames_l306_306284


namespace speed_difference_is_36_l306_306819

open Real

noncomputable def alex_speed : ℝ := 8 / (40 / 60)
noncomputable def jordan_speed : ℝ := 12 / (15 / 60)
noncomputable def speed_difference : ℝ := jordan_speed - alex_speed

theorem speed_difference_is_36 : speed_difference = 36 := by
  have hs1 : alex_speed = 8 / (40 / 60) := rfl
  have hs2 : jordan_speed = 12 / (15 / 60) := rfl
  have hd : speed_difference = jordan_speed - alex_speed := rfl
  rw [hs1, hs2] at hd
  simp [alex_speed, jordan_speed, speed_difference] at hd
  sorry

end speed_difference_is_36_l306_306819


namespace proof_total_distance_l306_306751

-- Define the total distance
def total_distance (D : ℕ) :=
  let by_foot := (1 : ℚ) / 6
  let by_bicycle := (1 : ℚ) / 4
  let by_bus := (1 : ℚ) / 3
  let by_car := 10
  let by_train := (1 : ℚ) / 12
  D - (by_foot + by_bicycle + by_bus + by_train) * D = by_car

-- Given proof problem
theorem proof_total_distance : ∃ D : ℕ, total_distance D ∧ D = 60 :=
sorry

end proof_total_distance_l306_306751


namespace solve_equation_l306_306690

theorem solve_equation (x : ℝ) : x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
  sorry

end solve_equation_l306_306690


namespace find_N_l306_306241

theorem find_N : {N : ℕ // N > 0 ∧ ∃ k : ℕ, 2^N - 2 * N = k^2} = {1, 2} := 
    sorry

end find_N_l306_306241


namespace six_dice_not_same_probability_l306_306938

theorem six_dice_not_same_probability :
  let total_outcomes := 6^6
  let all_same := 6
  let probability_all_same := all_same / total_outcomes
  let probability_not_all_same := 1 - probability_all_same
  probability_not_all_same = 7775 / 7776 :=
by
  sorry

end six_dice_not_same_probability_l306_306938


namespace least_positive_integer_to_multiple_of_5_l306_306334

theorem least_positive_integer_to_multiple_of_5 : ∃ (n : ℕ), n > 0 ∧ (725 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m :=
by
  sorry

end least_positive_integer_to_multiple_of_5_l306_306334


namespace min_sum_of_factors_l306_306314

theorem min_sum_of_factors (a b c d : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h : a * b * c * d = nat.factorial 8) : 
  a + b + c + d ≥ 61 := 
sorry

end min_sum_of_factors_l306_306314


namespace base_b_not_divisible_by_5_l306_306408

theorem base_b_not_divisible_by_5 (b : ℕ) : b = 4 ∨ b = 7 ∨ b = 8 → ¬ (5 ∣ (2 * b^2 * (b - 1))) :=
by
  sorry

end base_b_not_divisible_by_5_l306_306408


namespace problem_solution_l306_306529

noncomputable def solve_system : List (ℝ × ℝ × ℝ) :=
[(0, 1, -2), (-3/2, 5/2, -1/2)]

theorem problem_solution (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h1 : x^2 + y^2 = -x + 3*y + z)
  (h2 : y^2 + z^2 = x + 3*y - z)
  (h3 : z^2 + x^2 = 2*x + 2*y - z) :
  (x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2) :=
sorry

end problem_solution_l306_306529


namespace max_triangle_side_24_l306_306085

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306085


namespace g_at_pi_over_3_l306_306545

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

noncomputable def g (x : ℝ) (ω φ : ℝ) : ℝ := 3 * Real.sin (ω * x + φ) - 1

theorem g_at_pi_over_3 (ω φ : ℝ) :
  (∀ x : ℝ, f (π / 3 + x) ω φ = f (π / 3 - x) ω φ) →
  g (π / 3) ω φ = -1 :=
by sorry

end g_at_pi_over_3_l306_306545


namespace sin_cos_identity_l306_306983

theorem sin_cos_identity (α : ℝ) (hα_cos : Real.cos α = 3/5) (hα_sin : Real.sin α = 4/5) : Real.sin α + 2 * Real.cos α = 2 :=
by
  -- Proof omitted
  sorry

end sin_cos_identity_l306_306983


namespace customer_paid_amount_l306_306611

theorem customer_paid_amount 
  (cost_price : ℝ) 
  (markup_percent : ℝ) 
  (customer_payment : ℝ)
  (h1 : cost_price = 1250) 
  (h2 : markup_percent = 0.60)
  (h3 : customer_payment = cost_price + (markup_percent * cost_price)) :
  customer_payment = 2000 :=
sorry

end customer_paid_amount_l306_306611


namespace triangle_table_distinct_lines_l306_306787

theorem triangle_table_distinct_lines (a : ℕ) (h : a > 1) : 
  ∀ (n : ℕ) (line : ℕ → ℕ), 
  (line 0 = a) → 
  (∀ k, line (2*k + 1) = line k ^ 2 ∧ line (2*k + 2) = line k + 1) → 
  ∀ i j, i < 2^n → j < 2^n → (i ≠ j → line i ≠ line j) := 
by {
  sorry
}

end triangle_table_distinct_lines_l306_306787


namespace bug_total_distance_l306_306658

def total_distance (p1 p2 p3 p4 : ℤ) : ℤ :=
  abs (p2 - p1) + abs (p3 - p2) + abs (p4 - p3)

theorem bug_total_distance : total_distance (-3) (-8) 0 6 = 19 := 
by sorry

end bug_total_distance_l306_306658


namespace compound_proposition_truth_l306_306993

theorem compound_proposition_truth (p q : Prop) (h1 : ¬p ∨ ¬q = False) : (p ∧ q) ∧ (p ∨ q) :=
by
  sorry

end compound_proposition_truth_l306_306993


namespace intersection_eq_l306_306703

def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem intersection_eq : A ∩ B = {x | -1 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_eq_l306_306703


namespace polygon_sides_eq_nine_l306_306784

theorem polygon_sides_eq_nine (n : ℕ) 
  (interior_sum : ℕ := (n - 2) * 180)
  (exterior_sum : ℕ := 360)
  (condition : interior_sum = 4 * exterior_sum - 180) : 
  n = 9 :=
by {
  sorry
}

end polygon_sides_eq_nine_l306_306784


namespace solve_for_x_l306_306639

theorem solve_for_x : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by
  sorry

end solve_for_x_l306_306639


namespace num_handshakes_l306_306671

-- Definition of the conditions
def num_teams : Nat := 4
def women_per_team : Nat := 2
def total_women : Nat := num_teams * women_per_team
def handshakes_per_woman : Nat := total_women -1 - women_per_team

-- Statement of the problem to prove
theorem num_handshakes (h: total_women = 8) : (total_women * handshakes_per_woman) / 2 = 24 := 
by
  -- Proof goes here
  sorry

end num_handshakes_l306_306671


namespace men_apples_l306_306936

theorem men_apples (M W : ℕ) (h1 : M = W - 20) (h2 : 2 * M + 3 * W = 210) : M = 30 :=
by
  -- skipping the proof
  sorry

end men_apples_l306_306936


namespace find_B_share_l306_306648

-- Definitions for the conditions
def proportion (a b c d : ℕ) := 6 * a = 3 * b ∧ 3 * b = 5 * c ∧ 5 * c = 4 * d

def condition (c d : ℕ) := c = d + 1000

-- Statement of the problem
theorem find_B_share (A B C D : ℕ) (x : ℕ) 
  (h1 : proportion (6*x) (3*x) (5*x) (4*x)) 
  (h2 : condition (5*x) (4*x)) : 
  B = 3000 :=
by 
  sorry

end find_B_share_l306_306648


namespace carrie_total_sales_l306_306095

theorem carrie_total_sales :
  let tomatoes := 200
  let carrots := 350
  let price_tomato := 1.0
  let price_carrot := 1.50
  (tomatoes * price_tomato + carrots * price_carrot) = 725 := by
  -- let tomatoes := 200
  -- let carrots := 350
  -- let price_tomato := 1.0
  -- let price_carrot := 1.50
  -- show (tomatoes * price_tomato + carrots * price_carrot) = 725
  sorry

end carrie_total_sales_l306_306095


namespace max_side_length_is_11_l306_306013

theorem max_side_length_is_11 (a b c : ℕ) (h_perm : a + b + c = 24) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) (h_order : a < b ∧ b < c) : c = 11 :=
by
  sorry

end max_side_length_is_11_l306_306013


namespace problem1_problem2_l306_306678

theorem problem1 : |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.sqrt 2 = Real.sqrt 3 + Real.sqrt 2 := 
by {
  sorry
}

theorem problem2 : Real.sqrt 5 * (Real.sqrt 5 - 1 / Real.sqrt 5) = 4 := 
by {
  sorry
}

end problem1_problem2_l306_306678


namespace repeating_decimal_eq_fraction_l306_306397

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l306_306397


namespace ratio_of_heights_l306_306294

def min_height := 140
def brother_height := 180
def grow_needed := 20

def mary_height := min_height - grow_needed
def height_ratio := mary_height / brother_height

theorem ratio_of_heights : height_ratio = (2 / 3) := 
  sorry

end ratio_of_heights_l306_306294


namespace number_of_buildings_l306_306229

theorem number_of_buildings (studio_apartments : ℕ) (two_person_apartments : ℕ) (four_person_apartments : ℕ)
    (occupancy_percentage : ℝ) (current_occupancy : ℕ)
    (max_occupancy_building : ℕ) (max_occupancy_complex : ℕ) (num_buildings : ℕ)
    (h_studio : studio_apartments = 10)
    (h_two_person : two_person_apartments = 20)
    (h_four_person : four_person_apartments = 5)
    (h_occupancy_percentage : occupancy_percentage = 0.75)
    (h_current_occupancy : current_occupancy = 210)
    (h_max_occupancy_building : max_occupancy_building = 10 * 1 + 20 * 2 + 5 * 4)
    (h_max_occupancy_complex : max_occupancy_complex = current_occupancy / occupancy_percentage)
    (h_num_buildings : num_buildings = max_occupancy_complex / max_occupancy_building) :
    num_buildings = 4 :=
by
  sorry

end number_of_buildings_l306_306229


namespace B_finishes_job_in_37_5_days_l306_306662

variable (eff_A eff_B eff_C : ℝ)
variable (effA_eq_half_effB : eff_A = (1 / 2) * eff_B)
variable (effB_eq_two_thirds_effC : eff_B = (2 / 3) * eff_C)
variable (job_in_15_days : 15 * (eff_A + eff_B + eff_C) = 1)

theorem B_finishes_job_in_37_5_days :
  (1 / eff_B) = 37.5 :=
by
  sorry

end B_finishes_job_in_37_5_days_l306_306662


namespace books_left_after_sale_l306_306161

theorem books_left_after_sale (initial_books sold_books books_left : ℕ)
    (h1 : initial_books = 33)
    (h2 : sold_books = 26)
    (h3 : books_left = initial_books - sold_books) :
    books_left = 7 := by
  sorry

end books_left_after_sale_l306_306161


namespace solve_equation_l306_306691

theorem solve_equation (x : ℝ) : x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
  sorry

end solve_equation_l306_306691


namespace chris_newspapers_l306_306096

theorem chris_newspapers (C L : ℕ) 
  (h1 : L = C + 23) 
  (h2 : C + L = 65) : 
  C = 21 := 
by 
  sorry

end chris_newspapers_l306_306096


namespace length_of_cube_side_l306_306187

theorem length_of_cube_side (SA : ℝ) (h₀ : SA = 600) (h₁ : SA = 6 * a^2) : a = 10 := by
  sorry

end length_of_cube_side_l306_306187


namespace sin_double_angle_l306_306853

theorem sin_double_angle
  (α : ℝ) (h1 : Real.sin (3 * Real.pi / 2 - α) = 3 / 5) (h2 : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) :
  Real.sin (2 * α) = 24 / 25 :=
sorry

end sin_double_angle_l306_306853


namespace max_side_of_triangle_l306_306053

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306053


namespace sum_of_roots_P_l306_306111

noncomputable def P (x : ℂ) : ℂ :=
  (x - 1)^2007 + 2 * (x - 2)^2006 + 3 * (x - 3)^2005 + ∑ (k : ℕ) in finset.range 2004, (k + 3) * (x - (k + 3))^(2008 - (k + 4))

-- The theorem statement for the sum of the roots of the polynomial P
theorem sum_of_roots_P : ∑ (γ : ℂ) in (P.roots), γ = 2005 :=
by
  sorry

end sum_of_roots_P_l306_306111


namespace find_b_l306_306366

theorem find_b (a u v w : ℝ) (b : ℝ)
  (h1 : ∀ x : ℝ, 12 * x^3 + 7 * a * x^2 + 6 * b * x + b = 0 → (x = u ∨ x = v ∨ x = w))
  (h2 : 0 < u ∧ 0 < v ∧ 0 < w)
  (h3 : u ≠ v ∧ v ≠ w ∧ u ≠ w)
  (h4 : Real.log u / Real.log 3 + Real.log v / Real.log 3 + Real.log w / Real.log 3 = 3):
  b = -324 := 
sorry

end find_b_l306_306366


namespace find_exponent_l306_306732

theorem find_exponent (m x y a : ℝ) (h : y = m * x ^ a) (hx : x = 1 / 4) (hy : y = 1 / 2) : a = 1 / 2 :=
by
  sorry

end find_exponent_l306_306732


namespace inequality_proof_l306_306432

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l306_306432


namespace max_side_length_of_triangle_l306_306069

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306069


namespace find_a_to_satisfy_divisibility_l306_306537

theorem find_a_to_satisfy_divisibility (a : ℕ) (h₀ : 0 ≤ a) (h₁ : a < 11) (h₂ : (2 * 10^10 + a) % 11 = 0) : a = 9 :=
sorry

end find_a_to_satisfy_divisibility_l306_306537


namespace problem_x_sq_plus_y_sq_l306_306441

variables {x y : ℝ}

theorem problem_x_sq_plus_y_sq (h₁ : x - y = 12) (h₂ : x * y = 9) : x^2 + y^2 = 162 := 
sorry

end problem_x_sq_plus_y_sq_l306_306441


namespace perfect_square_sum_l306_306595

-- Define the numbers based on the given conditions
def A (n : ℕ) : ℕ := 4 * (10^(2 * n) - 1) / 9
def B (n : ℕ) : ℕ := 2 * (10^(n + 1) - 1) / 9
def C (n : ℕ) : ℕ := 8 * (10^n - 1) / 9

-- Define the main theorem to be proved
theorem perfect_square_sum (n : ℕ) : 
  ∃ k, A n + B n + C n + 7 = k * k :=
sorry

end perfect_square_sum_l306_306595


namespace max_side_length_of_triangle_l306_306074

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306074


namespace remainder_when_divided_by_296_l306_306218

theorem remainder_when_divided_by_296 (N : ℤ) (Q : ℤ) (R : ℤ)
  (h1 : N % 37 = 1)
  (h2 : N = 296 * Q + R)
  (h3 : 0 ≤ R) 
  (h4 : R < 296) :
  R = 260 := 
sorry

end remainder_when_divided_by_296_l306_306218


namespace driver_schedule_l306_306606

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end driver_schedule_l306_306606


namespace max_side_of_triangle_l306_306060

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l306_306060


namespace non_attacking_rooks_8x8_removed_corners_l306_306623

theorem non_attacking_rooks_8x8_removed_corners :
  let rows := Finset.range 8
  let columns := Finset.range 8
  let corners := {(0, 0), (0, 7), (7, 0), (7, 7)}
  let remaining_squares := (rows.product columns).filter (λ rc, ¬ rc ∈ corners)
  let rook_placement := {f // Function.Injective f ∧ ∀ r, (r, f r) ∈ remaining_squares}
  Finset.card rook_placement = 21600 :=
by sorry

end non_attacking_rooks_8x8_removed_corners_l306_306623


namespace abs_diff_between_sequences_l306_306190

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 + (-8) * (n - 1)

theorem abs_diff_between_sequences :
  |sequence_C 31 - sequence_D 31| = 600 :=
by
  sorry

end abs_diff_between_sequences_l306_306190


namespace sarah_tom_probability_not_next_to_each_other_l306_306739

theorem sarah_tom_probability_not_next_to_each_other : 
  let total_ways : ℕ := 45 in
  let ways_next_to_each_other : ℕ := 9 in
  let probability_not_next_to_each_other : ℚ := (total_ways - ways_next_to_each_other) / total_ways in
  probability_not_next_to_each_other = 4 / 5 :=
by
  sorry

end sarah_tom_probability_not_next_to_each_other_l306_306739


namespace wrapping_paper_l306_306769

theorem wrapping_paper (total_used : ℚ) (decoration_used : ℚ) (presents : ℕ) (other_presents : ℕ) (individual_used : ℚ) 
  (h1 : total_used = 5 / 8) 
  (h2 : decoration_used = 1 / 24) 
  (h3 : presents = 4) 
  (h4 : other_presents = 3) 
  (h5 : individual_used = (5 / 8 - 1 / 24) / 3) : 
  individual_used = 7 / 36 := 
by
  -- The theorem will be proven here.
  sorry

end wrapping_paper_l306_306769


namespace Annika_hiking_rate_is_correct_l306_306669

def AnnikaHikingRate
  (distance_partial_east distance_total_east : ℕ)
  (time_back_to_start : ℕ)
  (equality_rate : Nat) : Prop :=
  distance_partial_east = 2750 / 1000 ∧
  distance_total_east = 3500 / 1000 ∧
  time_back_to_start = 51 ∧
  equality_rate = 34

theorem Annika_hiking_rate_is_correct :
  ∃ R : ℕ, ∀ d1 d2 t,
  AnnikaHikingRate d1 d2 t R → R = 34 :=
by
  sorry

end Annika_hiking_rate_is_correct_l306_306669


namespace base8_9257_digits_product_sum_l306_306793

theorem base8_9257_digits_product_sum :
  let base10 := 9257
  let base8_digits := [2, 2, 0, 5, 1] -- base 8 representation of 9257
  let product_of_digits := 2 * 2 * 0 * 5 * 1
  let sum_of_digits := 2 + 2 + 0 + 5 + 1
  product_of_digits = 0 ∧ sum_of_digits = 10 := 
by
  sorry

end base8_9257_digits_product_sum_l306_306793


namespace smallest_possible_perimeter_l306_306522

-- Definitions for prime numbers and scalene triangles
def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions
def valid_sides (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ is_scalene_triangle a b c

def valid_perimeter (a b c : ℕ) : Prop :=
  is_prime (a + b + c)

-- The goal statement
theorem smallest_possible_perimeter : ∃ a b c : ℕ, valid_sides a b c ∧ valid_perimeter a b c ∧ (a + b + c) = 23 :=
by
  sorry

end smallest_possible_perimeter_l306_306522


namespace arithmetic_sequence_ratio_l306_306978

noncomputable def sum_first_n_terms (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_ratio (d : ℚ) (h : d ≠ 0) :
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  (7 * S₅) / (5 * S₇) = 10 / 11 :=
by 
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  sorry

end arithmetic_sequence_ratio_l306_306978


namespace complement_of_M_with_respect_to_U_l306_306862

open Set

def U : Set ℤ := {-1, -2, -3, -4}
def M : Set ℤ := {-2, -3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {-1, -4} :=
by
  sorry

end complement_of_M_with_respect_to_U_l306_306862


namespace surface_area_of_segmented_part_l306_306684

theorem surface_area_of_segmented_part (h_prism : ∀ (base_height prism_height : ℝ), base_height = 9 ∧ prism_height = 20)
  (isosceles_triangle : ∀ (a b c : ℝ), a = 18 ∧ b = 15 ∧ c = 15 ∧ b = c)
  (midpoints : ∀ (X Y Z : ℝ), X = 9 ∧ Y = 10 ∧ Z = 9) 
  : let triangle_CZX_area := 45
    let triangle_CZY_area := 45
    let triangle_CXY_area := 9
    let triangle_XYZ_area := 9
    (triangle_CZX_area + triangle_CZY_area + triangle_CXY_area + triangle_XYZ_area = 108) :=
sorry

end surface_area_of_segmented_part_l306_306684


namespace arrival_time_difference_l306_306549

theorem arrival_time_difference
  (d : ℝ) (r_H : ℝ) (r_A : ℝ) (h₁ : d = 2) (h₂ : r_H = 12) (h₃ : r_A = 6) :
  (d / r_A * 60) - (d / r_H * 60) = 10 :=
by
  sorry

end arrival_time_difference_l306_306549


namespace least_integer_to_add_l306_306335

theorem least_integer_to_add (n : ℕ) (h : n = 725) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 :=
by
  use 5
  split
  · exact Nat.lt_succ_self 4
  · rw [h]
    norm_num
    sorry

end least_integer_to_add_l306_306335


namespace correct_equation_l306_306212

-- Conditions:
def number_of_branches (x : ℕ) := x
def number_of_small_branches (x : ℕ) := x * x
def total_number (x : ℕ) := 1 + number_of_branches x + number_of_small_branches x

-- Proof Problem:
theorem correct_equation (x : ℕ) : total_number x = 43 → x^2 + x + 1 = 43 :=
by 
  sorry

end correct_equation_l306_306212


namespace friendly_point_pairs_l306_306982

def friendly_points (k : ℝ) (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (a, -1 / a) ∧ B = (-a, 1 / a) ∧
  B.2 = k * B.1 + 1 + k

theorem friendly_point_pairs : ∀ (k : ℝ), k ≥ 0 → 
  ∃ n, (n = 1 ∨ n = 2) ∧
  (∀ a : ℝ, a > 0 →
    friendly_points k a (a, -1 / a) (-a, 1 / a))
:= by
  sorry

end friendly_point_pairs_l306_306982


namespace tan_alpha_minus_pi_over_4_l306_306259

theorem tan_alpha_minus_pi_over_4 (α : Real) (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
  (Real.tan (α - π / 4) = -1/7) ∨ (Real.tan (α - π / 4) = -7) :=
by
  sorry

end tan_alpha_minus_pi_over_4_l306_306259


namespace range_of_ab_c2_l306_306258

theorem range_of_ab_c2 (a b c : ℝ) (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
    0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
sorry

end range_of_ab_c2_l306_306258


namespace remainder_8354_11_l306_306939

theorem remainder_8354_11 : 8354 % 11 = 6 := sorry

end remainder_8354_11_l306_306939


namespace exp_addition_property_l306_306592

theorem exp_addition_property (x y : ℝ) : (Real.exp (x + y)) = (Real.exp x) * (Real.exp y) := 
sorry

end exp_addition_property_l306_306592


namespace last_four_digits_5_to_2019_l306_306466

theorem last_four_digits_5_to_2019 :
  ∃ (x : ℕ), (5^2019) % 10000 = x ∧ x = 8125 :=
by
  sorry

end last_four_digits_5_to_2019_l306_306466


namespace no_diff_22_for_prime_products_l306_306406

theorem no_diff_22_for_prime_products (p1 p2 p3 p4 : ℕ) 
  (hp1_prime : Nat.Prime p1) (hp2_prime : Nat.Prime p2) (hp3_prime : Nat.Prime p3) (hp4_prime : Nat.Prime p4)
  (h_distinct : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4)
  (h_ordered : p1 < p2 ∧ p2 < p3 ∧ p3 < p4) 
  (h_bound : p1 * p2 * p3 * p4 < 1995) :
  ∀ (d8 d9 : ℕ), d8 = p1 * p4 → d9 = p2 * p3 → d9 - d8 ≠ 22 := by
    sorry

end no_diff_22_for_prime_products_l306_306406


namespace spending_example_l306_306836

theorem spending_example (X : ℝ) (h₁ : X + 2 * X + 3 * X = 120) : X = 20 := by
  sorry

end spending_example_l306_306836


namespace pizza_slices_l306_306458

theorem pizza_slices (num_people : ℕ) (slices_per_person : ℕ) (num_pizzas : ℕ) (total_slices : ℕ) :
  num_people = 6 → 
  slices_per_person = 4 → 
  num_pizzas = 3 → 
  total_slices = num_people * slices_per_person →
  total_slices / num_pizzas = 8 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have h : total_slices = 24 := h4
  rw h
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end pizza_slices_l306_306458


namespace minimum_value_f_l306_306250

open Real

noncomputable def f (x : ℝ) : ℝ :=
  x + (3 * x) / (x^2 + 3) + (x * (x + 3)) / (x^2 + 1) + (3 * (x + 1)) / (x * (x^2 + 1))

theorem minimum_value_f (x : ℝ) (hx : x > 0) : f x ≥ 7 :=
by
  -- Proof omitted
  sorry

end minimum_value_f_l306_306250


namespace calculate_paintable_area_l306_306877

noncomputable def bedroom_length : ℝ := 15
noncomputable def bedroom_width : ℝ := 11
noncomputable def bedroom_height : ℝ := 9
noncomputable def door_window_area : ℝ := 70
noncomputable def num_bedrooms : ℝ := 3

theorem calculate_paintable_area :
  (num_bedrooms * ((2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height) - door_window_area)) = 1194 := 
by
  -- conditions as definitions
  let total_wall_area := (2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height)
  let paintable_wall_in_bedroom := total_wall_area - door_window_area
  let total_paintable_area := num_bedrooms * paintable_wall_in_bedroom
  show total_paintable_area = 1194
  sorry

end calculate_paintable_area_l306_306877


namespace total_payroll_l306_306668

theorem total_payroll 
  (heavy_operator_pay : ℕ) 
  (laborer_pay : ℕ) 
  (total_people : ℕ) 
  (laborers : ℕ)
  (heavy_operators : ℕ)
  (total_payroll : ℕ)
  (h1: heavy_operator_pay = 140)
  (h2: laborer_pay = 90)
  (h3: total_people = 35)
  (h4: laborers = 19)
  (h5: heavy_operators = total_people - laborers)
  (h6: total_payroll = (heavy_operators * heavy_operator_pay) + (laborers * laborer_pay)) :
  total_payroll = 3950 :=
by sorry

end total_payroll_l306_306668


namespace find_N_value_l306_306571

-- Definitions based on given conditions
def M (n : ℕ) : ℕ := 4^n
def N (n : ℕ) : ℕ := 2^n
def condition (n : ℕ) : Prop := M n - N n = 240

-- Theorem statement to prove N == 16 given the conditions
theorem find_N_value (n : ℕ) (h : condition n) : N n = 16 := 
  sorry

end find_N_value_l306_306571


namespace gcf_45_135_90_l306_306625

theorem gcf_45_135_90 : Nat.gcd (Nat.gcd 45 135) 90 = 45 := 
by
  sorry

end gcf_45_135_90_l306_306625


namespace num_diagonals_29_sides_l306_306094

-- Define the number of sides
def n : Nat := 29

-- Calculate the combination (binomial coefficient) of selecting 2 vertices from n vertices
def binom (n k : Nat) : Nat := Nat.choose n k

-- Define the number of diagonals in a polygon with n sides
def num_diagonals (n : Nat) : Nat := binom n 2 - n

-- State the theorem to prove the number of diagonals for a polygon with 29 sides is 377
theorem num_diagonals_29_sides : num_diagonals 29 = 377 :=
by
  sorry

end num_diagonals_29_sides_l306_306094


namespace max_side_length_of_triangle_l306_306076

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306076


namespace g_45_l306_306913

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y
axiom g_30 : g 30 = 30

theorem g_45 : g 45 = 20 := by
  -- proof to be completed
  sorry

end g_45_l306_306913


namespace discount_difference_l306_306093

def single_discount (original: ℝ) (discount: ℝ) : ℝ :=
  original * (1 - discount)

def successive_discount (original: ℝ) (first_discount: ℝ) (second_discount: ℝ) : ℝ :=
  original * (1 - first_discount) * (1 - second_discount)

theorem discount_difference : 
  let original := 12000
  let single_disc := 0.30
  let first_disc := 0.20
  let second_disc := 0.10
  single_discount original single_disc - successive_discount original first_disc second_disc = 240 := 
by sorry

end discount_difference_l306_306093


namespace total_spent_is_64_l306_306192

def deck_price : ℕ := 8
def victors_decks : ℕ := 6
def friends_decks : ℕ := 2

def victors_spending : ℕ := victors_decks * deck_price
def friends_spending : ℕ := friends_decks * deck_price
def total_spending : ℕ := victors_spending + friends_spending

theorem total_spent_is_64 : total_spending = 64 := by
  sorry

end total_spent_is_64_l306_306192


namespace tennis_tournament_handshakes_l306_306674

theorem tennis_tournament_handshakes :
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team in
  (total_players * handshakes_per_player) / 2 = 24 :=
by
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team
  have h : (total_players * handshakes_per_player) / 2 = 24 := sorry
  exact h

end tennis_tournament_handshakes_l306_306674


namespace pattyCoinsValue_l306_306168

def totalCoins (q d : ℕ) : Prop := q + d = 30
def originalValue (q d : ℕ) : ℝ := 0.25 * q + 0.10 * d
def swappedValue (q d : ℕ) : ℝ := 0.10 * q + 0.25 * d
def valueIncrease (q : ℕ) : Prop := swappedValue q (30 - q) - originalValue q (30 - q) = 1.20

theorem pattyCoinsValue (q d : ℕ) (h1 : totalCoins q d) (h2 : valueIncrease q) : originalValue q d = 4.65 := 
by
  sorry

end pattyCoinsValue_l306_306168


namespace symmetric_line_equation_l306_306313

-- Definitions of the given conditions.
def original_line_equation (x y : ℝ) : Prop := 2 * x + 3 * y + 6 = 0
def line_of_symmetry (x y : ℝ) : Prop := y = x

-- The theorem statement to prove:
theorem symmetric_line_equation (x y : ℝ) : original_line_equation y x ↔ (3 * x + 2 * y + 6 = 0) :=
sorry

end symmetric_line_equation_l306_306313


namespace apple_tree_bears_fruit_in_7_years_l306_306293

def age_planted : ℕ := 4
def age_eats : ℕ := 11
def time_to_bear_fruit : ℕ := age_eats - age_planted

theorem apple_tree_bears_fruit_in_7_years :
  time_to_bear_fruit = 7 :=
by
  sorry

end apple_tree_bears_fruit_in_7_years_l306_306293


namespace domain_of_function_l306_306591

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ x + 2 ≠ 0} = {x : ℝ | x ≥ -3 ∧ x ≠ -2} :=
by
  sorry

end domain_of_function_l306_306591


namespace find_N_l306_306240

theorem find_N : {N : ℕ // N > 0 ∧ ∃ k : ℕ, 2^N - 2 * N = k^2} = {1, 2} := 
    sorry

end find_N_l306_306240


namespace initial_price_correct_l306_306612

-- Definitions based on the conditions
def initial_price : ℝ := 3  -- Rs. 3 per kg
def new_price : ℝ := 5      -- Rs. 5 per kg
def reduction_in_consumption : ℝ := 0.4  -- 40%

-- The main theorem we need to prove
theorem initial_price_correct :
  initial_price = 3 :=
sorry

end initial_price_correct_l306_306612


namespace solve_ax_plus_b_l306_306279

theorem solve_ax_plus_b (a b : ℝ) : 
  (if a ≠ 0 then "unique solution, x = -b / a"
   else if b ≠ 0 then "no solution"
   else "infinitely many solutions") = "A conditional control structure should be adopted" :=
sorry

end solve_ax_plus_b_l306_306279


namespace alpha_beta_value_l306_306275

variable (α β : ℝ)

def quadratic (x : ℝ) := x^2 + 2 * x - 2005

axiom roots_quadratic_eq : quadratic α = 0 ∧ quadratic β = 0

theorem alpha_beta_value :
  α^2 + 3 * α + β = 2003 :=
by sorry

end alpha_beta_value_l306_306275


namespace correct_option_D_l306_306940

theorem correct_option_D : -2 = -|-2| := 
by 
  sorry

end correct_option_D_l306_306940


namespace number_of_4_digit_numbers_divisible_by_9_l306_306134

theorem number_of_4_digit_numbers_divisible_by_9 :
  ∃ n : ℕ, (∀ k : ℕ, k ∈ Finset.range n → 1008 + k * 9 ≤ 9999) ∧
           (1008 + (n - 1) * 9 = 9999) ∧
           n = 1000 :=
by
  sorry

end number_of_4_digit_numbers_divisible_by_9_l306_306134


namespace faye_candy_count_l306_306800

theorem faye_candy_count :
  let initial_candy := 47
  let candy_ate := 25
  let candy_given := 40
  initial_candy - candy_ate + candy_given = 62 :=
by
  let initial_candy := 47
  let candy_ate := 25
  let candy_given := 40
  sorry

end faye_candy_count_l306_306800


namespace star_polygon_points_eq_24_l306_306156

theorem star_polygon_points_eq_24 (n : ℕ) 
  (A_i B_i : ℕ → ℝ) 
  (h_congruent_A : ∀ i j, A_i i = A_i j) 
  (h_congruent_B : ∀ i j, B_i i = B_i j) 
  (h_angle_difference : ∀ i, A_i i = B_i i - 15) : 
  n = 24 := 
sorry

end star_polygon_points_eq_24_l306_306156


namespace opposite_of_neg_three_l306_306919

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l306_306919


namespace realize_ancient_dreams_only_C_l306_306189

-- Define the available options
inductive Options
| A : Options
| B : Options
| C : Options
| D : Options

-- Define the ancient dreams condition
def realize_ancient_dreams (o : Options) : Prop :=
  o = Options.C

-- The theorem states that only Geographic Information Technology (option C) can realize the ancient dreams
theorem realize_ancient_dreams_only_C :
  realize_ancient_dreams Options.C :=
by
  -- skip the exact proof
  sorry

end realize_ancient_dreams_only_C_l306_306189


namespace overall_percentage_favoring_new_tool_l306_306219

theorem overall_percentage_favoring_new_tool (teachers students : ℕ) 
  (favor_teachers favor_students : ℚ) 
  (surveyed_teachers surveyed_students : ℕ) : 
  surveyed_teachers = 200 → 
  surveyed_students = 800 → 
  favor_teachers = 0.4 → 
  favor_students = 0.75 → 
  ( ( (favor_teachers * surveyed_teachers) + (favor_students * surveyed_students) ) / (surveyed_teachers + surveyed_students) ) * 100 = 68 := 
by 
  sorry

end overall_percentage_favoring_new_tool_l306_306219


namespace tennis_tournament_handshakes_l306_306675

theorem tennis_tournament_handshakes :
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team in
  (total_players * handshakes_per_player) / 2 = 24 :=
by
  let teams := 4
  let players_per_team := 2
  let total_players := teams * players_per_team
  let handshakes_per_player := total_players - players_per_team
  have h : (total_players * handshakes_per_player) / 2 = 24 := sorry
  exact h

end tennis_tournament_handshakes_l306_306675


namespace distance_from_desk_to_fountain_l306_306165

-- Problem definitions with given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Formulate the proof problem as a Lean theorem statement
theorem distance_from_desk_to_fountain :
  total_distance / trips = 30 :=
by
  sorry

end distance_from_desk_to_fountain_l306_306165


namespace max_product_l306_306630

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end max_product_l306_306630


namespace nina_shoe_payment_l306_306893

theorem nina_shoe_payment :
  let first_pair_original := 22
  let first_pair_discount := 0.10 * first_pair_original
  let first_pair_discounted := first_pair_original - first_pair_discount
  let first_pair_tax := 0.05 * first_pair_discounted
  let first_pair_final := first_pair_discounted + first_pair_tax

  let second_pair_original := first_pair_original * 1.50
  let second_pair_discount := 0.15 * second_pair_original
  let second_pair_discounted := second_pair_original - second_pair_discount
  let second_pair_tax := 0.07 * second_pair_discounted
  let second_pair_final := second_pair_discounted + second_pair_tax

  let total_payment := first_pair_final + second_pair_final
  total_payment = 50.80 :=
by 
  sorry

end nina_shoe_payment_l306_306893


namespace tables_capacity_l306_306505

theorem tables_capacity (invited attended : ℕ) (didn't_show_up : ℕ) (tables : ℕ) (capacity : ℕ) 
    (h1 : invited = 24) (h2 : didn't_show_up = 10) (h3 : attended = invited - didn't_show_up) 
    (h4 : attended = 14) (h5 : tables = 2) : capacity = attended / tables :=
by {
  -- Proof goes here
  sorry
}

end tables_capacity_l306_306505


namespace calculate_exponent_l306_306234

theorem calculate_exponent (m : ℝ) : (243 : ℝ)^(1 / 3) = 3^m → m = 5 / 3 :=
by
  sorry

end calculate_exponent_l306_306234


namespace inverse_proportional_p_q_l306_306475

theorem inverse_proportional_p_q (k : ℚ)
  (h1 : ∀ p q : ℚ, p * q = k)
  (h2 : (30 : ℚ) * (4 : ℚ) = k) :
  p = 12 ↔ (10 : ℚ) * p = k :=
by
  sorry

end inverse_proportional_p_q_l306_306475


namespace garden_perimeter_is_56_l306_306339

-- Define the conditions
def garden_width : ℕ := 12
def playground_length : ℕ := 16
def playground_width : ℕ := 12
def playground_area : ℕ := playground_length * playground_width
def garden_length : ℕ := playground_area / garden_width
def garden_perimeter : ℕ := 2 * (garden_length + garden_width)

-- Statement to prove
theorem garden_perimeter_is_56 :
  garden_perimeter = 56 := by
sorry

end garden_perimeter_is_56_l306_306339


namespace aladdin_can_find_heavy_coins_l306_306226

theorem aladdin_can_find_heavy_coins :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ x ≠ y ∧ (x + y ≥ 28) :=
by
  sorry

end aladdin_can_find_heavy_coins_l306_306226


namespace replace_half_black_cubes_l306_306949

-- Define the conditions
def is_black_cube (n : ℕ) (i j k : ℕ) : Prop :=
-- Mock-up property for black cubes
sorry

def valid_subprism (n : ℕ) (axis : ℕ) (index : ℕ) : Prop :=
-- Mock-up property for subprisms containing exactly two black cubes
-- and the two black cubes are separated by an even number of white cubes
sorry

-- Define the cube's property for the X and Y cubes
def is_X_cube (i j k : ℕ) : Prop :=
(i % 2 + j % 2 + k % 2) % 2 = 0

def is_Y_cube (i j k : ℕ) : Prop :=
(i % 2 + j % 2 + k % 2) % 2 = 1

-- Prove that we can replace half of the black cubes such that the condition is met
theorem replace_half_black_cubes
  (n : ℕ)
  (h1 : ∀ i j k, i < n ∧ j < n ∧ k < n → is_black_cube n i j k)
  (h2 : ∀ axis index, index < n → valid_subprism n axis index) :
  ∃ f : ℕ × ℕ × ℕ → Prop,
    (∀ i j k, i < n ∧ j < n ∧ k < n → f (i, j, k) = is_Y_cube i j k) ∧
    (∀ axis index, index < n →
      ∃! k_i, 0 ≤ k_i < n ∧ is_black_cube n k_i (index axis i) ∧ f (k_i, index axis i)) :=
sorry

end replace_half_black_cubes_l306_306949


namespace root_count_sqrt_eq_l306_306834

open Real

theorem root_count_sqrt_eq (x : ℝ) :
  (∀ y, (y = sqrt (7 - 2 * x)) → y = x * y → (∃ x, x = 7 / 2 ∨ x = 1)) ∧
  (7 - 2 * x ≥ 0) →
  ∃ s, s = 1 ∧ (7 - 2 * s = 0) → x = 1 ∨ x = 7 / 2 :=
sorry

end root_count_sqrt_eq_l306_306834


namespace max_side_length_l306_306001

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306001


namespace quadratic_root_value_l306_306143

theorem quadratic_root_value {m : ℝ} (h : m^2 + m - 1 = 0) : 2 * m^2 + 2 * m + 2025 = 2027 :=
sorry

end quadratic_root_value_l306_306143


namespace relationship_between_abcd_l306_306797

theorem relationship_between_abcd (a b c d : ℝ) (h : d ≠ 0) :
  (∀ x : ℝ, (a * x + c) / (b * x + d) = (a + c * x) / (b + d * x)) ↔ a / b = c / d :=
by
  sorry

end relationship_between_abcd_l306_306797


namespace unique_difference_count_l306_306272

open Finset

noncomputable def count_unique_differences (S : Finset ℕ) : ℕ :=
  (S.product S).filter (λ p, (p.1 ≠ p.2) ∧ (p.1 > p.2)).image (λ p, p.1 - p.2) |> toFinset |> card

theorem unique_difference_count : 
  let S := ({2, 3, 5, 7, 11, 13} : Finset ℕ) in
  count_unique_differences S = 10 :=
by
  let S := ({2, 3, 5, 7, 11, 13} : Finset ℕ)
  have h : count_unique_differences S = 10 := sorry
  exact h

end unique_difference_count_l306_306272


namespace exists_x_inequality_l306_306316

theorem exists_x_inequality (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * a * x + 9 < 0) ↔ a < -2 ∨ a > 2 :=
by
  sorry

end exists_x_inequality_l306_306316


namespace time_per_toy_is_3_l306_306223

-- Define the conditions
variable (total_toys : ℕ) (total_hours : ℕ)

-- Define the given condition
def given_condition := (total_toys = 50 ∧ total_hours = 150)

-- Define the statement to be proved
theorem time_per_toy_is_3 (h : given_condition total_toys total_hours) :
  total_hours / total_toys = 3 := by
sorry

end time_per_toy_is_3_l306_306223


namespace symmetric_points_addition_l306_306741

theorem symmetric_points_addition (m n : ℤ) (h₁ : m = 2) (h₂ : n = -3) : m + n = -1 := by
  rw [h₁, h₂]
  norm_num

end symmetric_points_addition_l306_306741


namespace rod_total_length_l306_306357

theorem rod_total_length (n : ℕ) (piece_length : ℝ) (total_length : ℝ) 
  (h1 : n = 50) 
  (h2 : piece_length = 0.85) 
  (h3 : total_length = n * piece_length) : 
  total_length = 42.5 :=
by
  -- Proof steps will go here
  sorry

end rod_total_length_l306_306357


namespace max_determinant_is_sqrt_233_l306_306565

noncomputable def max_det_of_matrix : ℝ :=
let v := ![3, 2, 0]
let w := ![1, -1, 4]
let u := ((1 : ℝ) / Real.sqrt 233) • ![8, -12, -5] in
  (Matrix.det ![
    u,
    v,
    w
  ])

theorem max_determinant_is_sqrt_233 :
  max_det_of_matrix = Real.sqrt 233 :=
sorry

end max_determinant_is_sqrt_233_l306_306565


namespace helens_mother_brought_101_l306_306271

-- Define the conditions
def total_hotdogs : ℕ := 480
def dylan_mother_hotdogs : ℕ := 379
def helens_mother_hotdogs := total_hotdogs - dylan_mother_hotdogs

-- Theorem statement: Prove that the number of hotdogs Helen's mother brought is 101
theorem helens_mother_brought_101 : helens_mother_hotdogs = 101 :=
by
  sorry

end helens_mother_brought_101_l306_306271


namespace integer_condition_l306_306731

theorem integer_condition (p : ℕ) (h : p > 0) : 
  (∃ n : ℤ, (3 * (p: ℤ) + 25) = n * (2 * (p: ℤ) - 5)) ↔ (3 ≤ p ∧ p ≤ 35) :=
sorry

end integer_condition_l306_306731


namespace percentage_speaking_both_langs_l306_306828

def diplomats_total : ℕ := 100
def diplomats_french : ℕ := 22
def diplomats_not_russian : ℕ := 32
def diplomats_neither : ℕ := 20

theorem percentage_speaking_both_langs
  (h1 : 20% diplomats_total = diplomats_neither)
  (h2 : diplomats_total - diplomats_not_russian = 68)
  (h3 : diplomats_total ≠ 0) :
  (22 + 68 - 80) / diplomats_total * 100 = 10 :=
by
  sorry

end percentage_speaking_both_langs_l306_306828


namespace fraction_problem_l306_306248

noncomputable def zero_point_one_five : ℚ := 5 / 33
noncomputable def two_point_four_zero_three : ℚ := 2401 / 999

theorem fraction_problem :
  (zero_point_one_five / two_point_four_zero_three) = (4995 / 79233) :=
by
  sorry

end fraction_problem_l306_306248


namespace infinite_solutions_l306_306695

theorem infinite_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (k : ℕ), x = k^3 + 1 ∧ y = (k^3 + 1) * k := 
sorry

end infinite_solutions_l306_306695


namespace necessary_and_sufficient_condition_l306_306177

variable {R : Type*} [LinearOrderedField R]
variable (f : R × R → R)
variable (x₀ y₀ : R)

theorem necessary_and_sufficient_condition :
  (f (x₀, y₀) = 0) ↔ ((x₀, y₀) ∈ {p : R × R | f p = 0}) :=
by
  sorry

end necessary_and_sufficient_condition_l306_306177


namespace radius_of_circle_through_points_l306_306780

theorem radius_of_circle_through_points : 
  ∃ (x : ℝ), 
  (dist (x, 0) (2, 5) = dist (x, 0) (3, 4)) →
  (∃ (r : ℝ), r = dist (x, 0) (2, 5) ∧ r = 5) :=
by
  sorry

end radius_of_circle_through_points_l306_306780


namespace cone_lateral_area_and_sector_area_l306_306802

theorem cone_lateral_area_and_sector_area 
  (slant_height : ℝ) 
  (height : ℝ) 
  (r : ℝ) 
  (h_slant : slant_height = 1) 
  (h_height : height = 0.8) 
  (h_r : r = Real.sqrt (slant_height^2 - height^2)) :
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) ∧
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) :=
by
  sorry

end cone_lateral_area_and_sector_area_l306_306802


namespace solve_for_x_l306_306725

theorem solve_for_x (x : ℚ) (h : (1 / 3 - 1 / 4 = 4 / x)) : x = 48 := by
  sorry

end solve_for_x_l306_306725


namespace sequence_sum_l306_306750

theorem sequence_sum (A B C D E F G H I J : ℤ)
  (h1 : D = 7)
  (h2 : A + B + C = 24)
  (h3 : B + C + D = 24)
  (h4 : C + D + E = 24)
  (h5 : D + E + F = 24)
  (h6 : E + F + G = 24)
  (h7 : F + G + H = 24)
  (h8 : G + H + I = 24)
  (h9 : H + I + J = 24) : 
  A + J = 105 :=
sorry

end sequence_sum_l306_306750


namespace smallest_factor_to_end_with_four_zeros_l306_306657

theorem smallest_factor_to_end_with_four_zeros :
  ∃ x : ℕ, (975 * 935 * 972 * x) % 10000 = 0 ∧
           (∀ y : ℕ, (975 * 935 * 972 * y) % 10000 = 0 → x ≤ y) ∧
           x = 20 := by
  -- The proof would go here.
  sorry

end smallest_factor_to_end_with_four_zeros_l306_306657


namespace find_breadth_l306_306249

-- Define variables and constants
variables (SA l h w : ℝ)

-- Given conditions
axiom h1 : SA = 2400
axiom h2 : l = 15
axiom h3 : h = 16

-- Define the surface area equation for a cuboid 
def surface_area := 2 * (l * w + l * h + w * h)

-- Statement to prove
theorem find_breadth : surface_area l w h = SA → w = 30.97 := sorry

end find_breadth_l306_306249


namespace min_dist_C1_to_l_l306_306743

-- Define the parametric line equation
def line_l (t : ℝ) : ℝ × ℝ := (-sqrt 5 + (sqrt 2)/2 * t, sqrt 5 + (sqrt 2)/2 * t)

-- Parametric equation of line in Cartesian form
def line_eq (x y : ℝ) : Prop :=
  x + y = 0

-- Polar equation of the curve C
def polar_eq (θ : ℝ) : ℝ :=
  4 * Real.cos θ

-- Cartesian equation of the curve C
def cartesian_eq (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

-- We define curve C_1 after transformations
def C1_eq (x y : ℝ) : Prop :=
  4 * x ^ 2 + y ^ 2 = 4

-- Minimum distance between points on C_1 and line l
def min_distance (x y : ℝ) : ℝ :=
  (abs (x + y) / Real.sqrt 2)

/-- Proof that the minimum distance from points on curve C_1 to line l is 0 -/
theorem min_dist_C1_to_l : ∀ x y : ℝ, C1_eq x y → line_eq x y → min_distance x y = 0 :=
sorry

end min_dist_C1_to_l_l306_306743


namespace problem_proof_l306_306705

theorem problem_proof (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = 3) : 3 * a^2 * b + 3 * a * b^2 = 18 := 
by
  sorry

end problem_proof_l306_306705


namespace perfect_square_form_l306_306244

theorem perfect_square_form (N : ℕ) (hN : 0 < N) : 
  ∃ x : ℤ, 2^N - 2 * (N : ℤ) = x^2 ↔ N = 1 ∨ N = 2 :=
by
  sorry

end perfect_square_form_l306_306244


namespace total_price_of_purchases_l306_306198

def price_of_refrigerator := 4275
def price_difference := 1490
def price_of_washing_machine := price_of_refrigerator - price_difference
def total_price := price_of_refrigerator + price_of_washing_machine

theorem total_price_of_purchases : total_price = 7060 :=
by
  rfl  -- This is just a placeholder; you need to solve the proof.

end total_price_of_purchases_l306_306198


namespace range_a_l306_306416

noncomputable def A (a : ℝ) : Set ℝ := {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}

noncomputable def B : Set ℝ := {x | x < -1 ∨ x > 16}

theorem range_a (a : ℝ) : (A a ∩ B = A a) → (a < 6 ∨ a > 7.5) :=
by
  intro h
  sorry

end range_a_l306_306416


namespace fraction_order_l306_306799

theorem fraction_order :
  (21:ℚ) / 17 < (23:ℚ) / 18 ∧ (23:ℚ) / 18 < (25:ℚ) / 19 :=
by
  sorry

end fraction_order_l306_306799


namespace max_triangle_side_24_l306_306078

theorem max_triangle_side_24 {a b c : ℕ} (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 24)
  (h4 : a < b + c) (h5 : b < a + c) (h6 : c < a + b) : a ≤ 11 := sorry

end max_triangle_side_24_l306_306078


namespace find_p_l306_306325

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_p (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (h1 : p + q = r + 2) (h2 : 1 < p) (h3 : p < q) :
  p = 2 := 
sorry

end find_p_l306_306325


namespace limonia_largest_unachievable_l306_306746

noncomputable def largest_unachievable_amount (n : ℕ) : ℕ :=
  12 * n^2 + 14 * n - 1

theorem limonia_largest_unachievable (n : ℕ) :
  ∀ k, ¬ ∃ a b c d : ℕ, 
    k = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10) 
    → k = largest_unachievable_amount n :=
sorry

end limonia_largest_unachievable_l306_306746


namespace surface_area_combination_l306_306786

noncomputable def smallest_surface_area : ℕ :=
  let s1 := 3
  let s2 := 5
  let s3 := 8
  let surface_area := 6 * (s1 * s1 + s2 * s2 + s3 * s3)
  let overlap_area := (s1 * s1) * 4 + (s2 * s2) * 2 
  surface_area - overlap_area

theorem surface_area_combination :
  smallest_surface_area = 502 :=
by
  -- Proof goes here
  sorry

end surface_area_combination_l306_306786


namespace flag_distance_false_l306_306697

theorem flag_distance_false (track_length : ℕ) (num_flags : ℕ) (flag1_flagN : 2 ≤ num_flags)
  (h1 : track_length = 90) (h2 : num_flags = 10) :
  ¬ (track_length / (num_flags - 1) = 9) :=
by
  sorry

end flag_distance_false_l306_306697


namespace find_value_l306_306884

noncomputable def S2013 (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) : ℂ :=
  (x / (x + y))^2013 + (y / (x + y))^2013

theorem find_value (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) :
  S2013 x y h h_eq = -2 :=
sorry

end find_value_l306_306884


namespace number_of_integers_satisfying_inequality_l306_306436

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | n ≠ 0 ∧ (1 : ℝ) / |n| ≥ 1 / 5}.finite.card = 10 :=
by
  -- Proof goes here
  sorry

end number_of_integers_satisfying_inequality_l306_306436


namespace base_length_of_isosceles_triangle_l306_306178

theorem base_length_of_isosceles_triangle (a b : ℕ) (h1 : a = 8) (h2 : b + 2 * a = 26) : b = 10 :=
by
  have h3 : 2 * 8 = 16 := by norm_num
  rw [h1] at h2
  rw [h3] at h2
  linarith

end base_length_of_isosceles_triangle_l306_306178


namespace repeating_decimal_fraction_equiv_l306_306384

open_locale big_operators

theorem repeating_decimal_fraction_equiv (a : ℚ) (r : ℚ) (h : 0 < r ∧ r < 1) :
  (0.\overline{36} : ℚ) = a / (1 - r) → (a = 36 / 100) → (r = 1 / 100) → (0.\overline{36} : ℚ) = 4 / 11 :=
by
  intro h_decimal_series h_a h_r
  sorry

end repeating_decimal_fraction_equiv_l306_306384


namespace price_of_cheaper_book_l306_306616

theorem price_of_cheaper_book
    (total_cost : ℕ)
    (sets : ℕ)
    (price_more_expensive_book_increase : ℕ)
    (h1 : total_cost = 21000)
    (h2 : sets = 3)
    (h3 : price_more_expensive_book_increase = 300) :
  ∃ x : ℕ, 3 * ((x + (x + price_more_expensive_book_increase))) = total_cost ∧ x = 3350 :=
by
  sorry

end price_of_cheaper_book_l306_306616


namespace max_side_length_of_triangle_l306_306072

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306072


namespace sum_of_three_consecutive_integers_is_21_l306_306497

theorem sum_of_three_consecutive_integers_is_21 : 
  ∃ (n : ℤ), 3 * n = 21 :=
by
  sorry

end sum_of_three_consecutive_integers_is_21_l306_306497


namespace correct_statements_are_ACD_l306_306941

theorem correct_statements_are_ACD :
  (∀ (width : ℝ), narrower_width_implies_better_fit width) ∧
  (rA rB : ℝ) (h_rA : rA = 0.97) (h_rB : rB = -0.99)
    → ¬ (stronger_A_implies_correlation rA rB) ∧
  (∀ (R2 : ℝ), smaller_R2_implies_worse_fit R2) ∧
  (num_products : ℕ) (defective_products : ℕ) (selected_products : ℕ)
    (h_num : num_products = 10) (h_def : defective_products = 3) (h_sel : selected_products = 2),
    probability_exactly_one_defective num_products defective_products selected_products = 7 / 15 :=
by
  sorry  -- Proof is not required

end correct_statements_are_ACD_l306_306941


namespace infinite_a_exists_l306_306698

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ+), ∃ (m : ℕ+), n^6 + 3 * (a : ℕ) = m^3 :=
  sorry

end infinite_a_exists_l306_306698


namespace repeating_decimal_equals_fraction_l306_306399

noncomputable def repeating_decimal_to_fraction : ℚ := 0.363636...

theorem repeating_decimal_equals_fraction :
  repeating_decimal_to_fraction = 4/11 := 
sorry

end repeating_decimal_equals_fraction_l306_306399


namespace cylinder_water_depth_l306_306215

theorem cylinder_water_depth 
  (height radius : ℝ)
  (h_ge_zero : height ≥ 0)
  (r_ge_zero : radius ≥ 0)
  (total_height : height = 1200)
  (total_radius : radius = 100)
  (above_water_vol : 1 / 3 * π * radius^2 * height = 1 / 3 * π * radius^2 * 1200) :
  height - 800 = 400 :=
by
  -- Use provided constraints and logical reasoning on structures
  sorry

end cylinder_water_depth_l306_306215


namespace correct_statements_l306_306942

variable (model : Type) (r_A r_B R2 : ℝ) (totalProducts defectiveProducts : ℕ)
variable (selectionEvents : Set (Set ℕ))

-- Conditions
def narrow_band_better_fit : Prop :=
  ∀ (residuals : model → ℝ), (∀ d d' ∈ residuals, |d| < |d'|) → True

def stronger_correlation_abs : Prop :=
  r_A = 0.97 ∧ r_B = -0.99 ∧ |r_A| < |r_B|

def worse_fit_smaller_R2 : Prop :=
  R2 ≥ 0 ∧ R2 < 1 ∧ R2 ≤ 0.5

def probability_one_defective : Prop :=
  totalProducts = 10 ∧ defectiveProducts = 3 ∧
  (∀ (selection : Set ℕ), selection ∈ selectionEvents →
    selection.card = 2 → selection.inter {1, 2, 3}.card = 1 →
    (selectionEvents.prob selection = 7 / 15))

-- Theorem
theorem correct_statements : narrow_band_better_fit model ∧ 
                              worse_fit_smaller_R2 R2 ∧ 
                              probability_one_defective totalProducts defectiveProducts selectionEvents := by
  sorry

end correct_statements_l306_306942


namespace claire_crafting_hours_l306_306521

theorem claire_crafting_hours (H1 : 24 = 24) (H2 : 8 = 8) (H3 : 4 = 4) (H4 : 2 = 2):
  let total_hours_per_day := 24
  let sleep_hours := 8
  let cleaning_hours := 4
  let cooking_hours := 2
  let working_hours := total_hours_per_day - sleep_hours
  let remaining_hours := working_hours - (cleaning_hours + cooking_hours)
  let crafting_hours := remaining_hours / 2
  crafting_hours = 5 :=
by
  sorry

end claire_crafting_hours_l306_306521


namespace q_is_false_of_pq_false_and_notp_false_l306_306446

variables (p q : Prop)

theorem q_is_false_of_pq_false_and_notp_false (hpq_false : ¬(p ∧ q)) (hnotp_false : ¬(¬p)) : ¬q := 
by 
  sorry

end q_is_false_of_pq_false_and_notp_false_l306_306446


namespace quadratic_inequality_solution_l306_306613

theorem quadratic_inequality_solution :
  {x : ℝ | -x^2 + x + 2 > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end quadratic_inequality_solution_l306_306613


namespace fraction_rep_finite_geom_series_036_l306_306393

noncomputable def expr := (36:ℚ) / (10^2 : ℚ) + (36:ℚ) / (10^4 : ℚ) + (36:ℚ) / (10^6 : ℚ) + sum (λ (n:ℕ), (36:ℚ) / (10^(2* (n+1)) : ℚ))

theorem fraction_rep_finite_geom_series_036 : expr = (4:ℚ) / (11:ℚ) := by
  sorry

end fraction_rep_finite_geom_series_036_l306_306393


namespace gcd_between_35_and_7_l306_306480

theorem gcd_between_35_and_7 {n : ℕ} (h1 : 65 < n) (h2 : n < 75) (h3 : gcd 35 n = 7) : n = 70 := 
sorry

end gcd_between_35_and_7_l306_306480


namespace fraction_equivalent_of_repeating_decimal_l306_306389

theorem fraction_equivalent_of_repeating_decimal 
  (h_series : (0.36 : ℝ) = (geom_series (9/25) (1/100))) :
  ∃ (f : ℚ), (f = 4/11) ∧ (0.36 : ℝ) = f :=
by
  sorry

end fraction_equivalent_of_repeating_decimal_l306_306389


namespace solve_abs_equation_l306_306974

theorem solve_abs_equation (x : ℝ) (h : abs (x - 20) + abs (x - 18) = abs (2 * x - 36)) : x = 19 :=
sorry

end solve_abs_equation_l306_306974


namespace minimize_PA_PB_l306_306590

theorem minimize_PA_PB 
  (A B : ℝ × ℝ) 
  (hA : A = (1, 3)) 
  (hB : B = (5, 1)) : 
  ∃ P : ℝ × ℝ, P = (4, 0) ∧ 
  ∀ P' : ℝ × ℝ, P'.snd = 0 → (dist P A + dist P B) ≤ (dist P' A + dist P' B) :=
sorry

end minimize_PA_PB_l306_306590


namespace minimum_value_of_f_l306_306719

-- Define the function y = f(x)
def f (x : ℝ) : ℝ := x^2 + 8 * x + 25

-- We need to prove that the minimum value of f(x) is 9
theorem minimum_value_of_f : ∃ x : ℝ, f x = 9 ∧ ∀ y : ℝ, f y ≥ 9 :=
by
  sorry

end minimum_value_of_f_l306_306719


namespace total_cost_of_cloth_l306_306754

/-- Define the length of the cloth in meters --/
def length_of_cloth : ℝ := 9.25

/-- Define the cost per meter in dollars --/
def cost_per_meter : ℝ := 46

/-- Theorem stating that the total cost is $425.50 given the length and cost per meter --/
theorem total_cost_of_cloth : length_of_cloth * cost_per_meter = 425.50 := by
  sorry

end total_cost_of_cloth_l306_306754


namespace max_cylinder_volume_l306_306676

/-- Given a rectangle with perimeter 18 cm, when rotating it around one side to form a cylinder, 
    the maximum volume of the cylinder and the corresponding side length of the rectangle. -/
theorem max_cylinder_volume (x y : ℝ) (h_perimeter : 2 * (x + y) = 18) (hx : x > 0) (hy : y > 0)
  (h_cylinder_volume : ∃ (V : ℝ), V = π * x * (y / 2)^2) :
  (x = 3 ∧ y = 6 ∧ ∀ V, V = 108 * π) := sorry

end max_cylinder_volume_l306_306676


namespace find_a_l306_306988

theorem find_a (a : ℝ) :
  (∃ x y : ℝ, x - 2 * y + 1 = 0 ∧ x + 3 * y - 1 = 0 ∧ ¬(∀ x y : ℝ, ax + 2 * y - 3 = 0)) →
  (∃ p q : ℝ, ax + 2 * q - 3 = 0 ∧ (a = -1 ∨ a = 2 / 3)) :=
by {
  sorry
}

end find_a_l306_306988


namespace const_seq_is_arithmetic_not_geometric_l306_306317

-- Define the sequence
def const_seq (n : ℕ) : ℕ := 0

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, seq (n + 1) = seq n + d

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = seq n * r

-- The proof statement
theorem const_seq_is_arithmetic_not_geometric :
  is_arithmetic_sequence const_seq ∧ ¬ is_geometric_sequence const_seq :=
by
  sorry

end const_seq_is_arithmetic_not_geometric_l306_306317


namespace solve_for_m_l306_306444

theorem solve_for_m (x m : ℝ) (hx : 0 < x) (h_eq : m / (x^2 - 9) + 2 / (x + 3) = 1 / (x - 3)) : m = 6 :=
sorry

end solve_for_m_l306_306444


namespace find_ratio_AF_FB_l306_306752

-- Define the vector space over reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions of points A, B, C, D, F, P
variables (a b c d f p : V)

-- Given conditions as hypotheses
variables (h1 : (p = 2 / 5 • a + 3 / 5 • d))
variables (h2 : (p = 5 / 7 • f + 2 / 7 • c))
variables (hd : (d = 1 / 3 • b + 2 / 3 • c))
variables (hf : (f = 1 / 4 • a + 3 / 4 • b))

-- Theorem statement
theorem find_ratio_AF_FB : (41 : ℝ) / 15 = (41 : ℝ) / 15 := 
by sorry

end find_ratio_AF_FB_l306_306752


namespace inequality_solution_set_l306_306422

noncomputable def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem inequality_solution_set (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_A : f 0 = -2)
  (h_B : f 3 = 2) :
  {x : ℝ | |f (x+1)| ≥ 2} = {x | x ≤ -1} ∪ {x | x ≥ 2} :=
sorry

end inequality_solution_set_l306_306422


namespace chess_tournament_participants_l306_306500

/-- If each participant of a chess tournament plays exactly one game with each of the remaining participants, and 231 games are played during the tournament, then the number of participants is 22. -/
theorem chess_tournament_participants (n : ℕ) (h : (n - 1) * n / 2 = 231) : n = 22 :=
sorry

end chess_tournament_participants_l306_306500


namespace polygon_interior_angle_sum_l306_306352

theorem polygon_interior_angle_sum (n : ℕ) (h : (n-1) * 180 = 2400 + 120) : n = 16 :=
by
  sorry

end polygon_interior_angle_sum_l306_306352


namespace intersection_A_B_l306_306715

def set_A : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

def set_B : Set ℝ := {x | x^2 - 3 * x < 0}

theorem intersection_A_B :
  {x : ℤ | x ∈ set_A ∧ x ∈ set_B} = {1, 2} :=
sorry

end intersection_A_B_l306_306715


namespace smaller_group_men_l306_306209

-- Define the main conditions of the problem
def men_work_days : ℕ := 36 * 18  -- 36 men for 18 days

-- Define the theorem we need to prove
theorem smaller_group_men (M : ℕ) (h: M * 72 = men_work_days) : M = 9 :=
by
  -- proof is not required
  sorry

end smaller_group_men_l306_306209


namespace ratio_songs_kept_to_deleted_l306_306326

theorem ratio_songs_kept_to_deleted (initial_songs deleted_songs kept_songs : ℕ) 
  (h_initial : initial_songs = 54) (h_deleted : deleted_songs = 9) (h_kept : kept_songs = initial_songs - deleted_songs) :
  (kept_songs : ℚ) / (deleted_songs : ℚ) = 5 / 1 :=
by
  sorry

end ratio_songs_kept_to_deleted_l306_306326


namespace tangent_line_proof_minimum_a_proof_l306_306857

noncomputable def f (x : ℝ) := 2 * Real.log x - 3 * x^2 - 11 * x

def tangent_equation_correct : Prop :=
  let y := f 1
  let slope := (2 / 1 - 6 * 1 - 11)
  (slope = -15) ∧ (y = -14) ∧ (∀ x y, y = -15 * (x - 1) + -14 ↔ 15 * x + y - 1 = 0)

def minimum_a_correct : Prop :=
  ∃ a : ℤ, 
    (∀ x, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x - 2) ↔ (a = 2)

theorem tangent_line_proof : tangent_equation_correct := sorry

theorem minimum_a_proof : minimum_a_correct := sorry

end tangent_line_proof_minimum_a_proof_l306_306857


namespace tallest_player_height_l306_306321

theorem tallest_player_height (shortest_player tallest_player : ℝ) (height_diff : ℝ)
  (h1 : shortest_player = 68.25)
  (h2 : height_diff = 9.5)
  (h3 : tallest_player = shortest_player + height_diff) :
  tallest_player = 77.75 :=
by {
  rw [h1, h2] at h3,
  exact h3,
  sorry
}

end tallest_player_height_l306_306321


namespace probability_greater_equal_zero_l306_306709

noncomputable def random_variable : Type := ℝ -- representing the real-valued random variable

variables (μ σ : ℝ) (X : random_variable) [NormalDist X μ σ]

theorem probability_greater_equal_zero (hX_mean : μ = 1) (hX_prob : P(X > 2) = 0.3) : P(X ≥ 0) = 0.7 :=
by sorry

end probability_greater_equal_zero_l306_306709


namespace decreasing_condition_l306_306260

variable (m : ℝ)

def quadratic_fn (x : ℝ) : ℝ := x^2 + m * x + 1

theorem decreasing_condition (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → (deriv (quadratic_fn m) x ≤ 0)) :
    m ≤ -10 := 
by
  -- Proof omitted
  sorry

end decreasing_condition_l306_306260


namespace distance_PQ_parallel_x_max_distance_PQ_l306_306230

open Real

def parabola (x : ℝ) : ℝ := x^2

/--
1. When PQ is parallel to the x-axis, find the distance from point O to PQ.
-/
theorem distance_PQ_parallel_x (m : ℝ) (h₁ : m ≠ 0) (h₂ : parabola m = 1) : 
  ∃ d : ℝ, d = 1 := by
  sorry

/--
2. Find the maximum value of the distance from point O to PQ.
-/
theorem max_distance_PQ (a b : ℝ) (h₁ : a * b = -1) (h₂ : ∀ x, ∃ y, y = a * x + b) :
  ∃ d : ℝ, d = 1 := by
  sorry

end distance_PQ_parallel_x_max_distance_PQ_l306_306230


namespace arithmetic_mean_l306_306194

theorem arithmetic_mean (a b : ℚ) (h1 : a = 3/7) (h2 : b = 6/11) :
  (a + b) / 2 = 75 / 154 :=
by
  sorry

end arithmetic_mean_l306_306194


namespace money_left_after_expenditures_l306_306239

variable (initial_amount : ℝ) (P : initial_amount = 15000)
variable (gas_percentage food_fraction clothing_fraction entertainment_percentage : ℝ) 
variable (H1 : gas_percentage = 0.35) (H2 : food_fraction = 0.2) (H3 : clothing_fraction = 0.25) (H4 : entertainment_percentage = 0.15)

theorem money_left_after_expenditures
  (money_left : ℝ):
  money_left = initial_amount * (1 - gas_percentage) *
                (1 - food_fraction) * 
                (1 - clothing_fraction) * 
                (1 - entertainment_percentage) → 
  money_left = 4972.50 :=
by
  sorry

end money_left_after_expenditures_l306_306239


namespace value_of_card_l306_306465

/-- For this problem: 
    1. Matt has 8 baseball cards worth $6 each.
    2. He trades two of them to Jane in exchange for 3 $2 cards and a card of certain value.
    3. He makes a profit of $3.
    We need to prove that the value of the card that Jane gave to Matt apart from the $2 cards is $9. -/
theorem value_of_card (value_per_card traded_cards received_dollar_cards profit received_total_value : ℤ)
  (h1 : value_per_card = 6)
  (h2 : traded_cards = 2)
  (h3 : received_dollar_cards = 6)
  (h4 : profit = 3)
  (h5 : received_total_value = 15) :
  received_total_value - received_dollar_cards = 9 :=
by {
  -- This is just left as a placeholder to signal that the proof needs to be provided.
  sorry
}

end value_of_card_l306_306465


namespace max_side_of_triangle_l306_306040

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l306_306040


namespace trucks_after_redistribution_l306_306788

/-- Problem Statement:
   Prove that the total number of trucks after redistribution is 10.
-/

theorem trucks_after_redistribution
    (num_trucks1 : ℕ)
    (boxes_per_truck1 : ℕ)
    (num_trucks2 : ℕ)
    (boxes_per_truck2 : ℕ)
    (containers_per_box : ℕ)
    (containers_per_truck_after : ℕ)
    (h1 : num_trucks1 = 7)
    (h2 : boxes_per_truck1 = 20)
    (h3 : num_trucks2 = 5)
    (h4 : boxes_per_truck2 = 12)
    (h5 : containers_per_box = 8)
    (h6 : containers_per_truck_after = 160) :
  (num_trucks1 * boxes_per_truck1 + num_trucks2 * boxes_per_truck2) * containers_per_box / containers_per_truck_after = 10 := by
  sorry

end trucks_after_redistribution_l306_306788


namespace inequality_proof_l306_306430

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l306_306430


namespace pencil_eraser_cost_l306_306576

variable (p e : ℕ)

theorem pencil_eraser_cost
  (h1 : 15 * p + 5 * e = 125)
  (h2 : p > e)
  (h3 : p > 0)
  (h4 : e > 0) :
  p + e = 11 :=
sorry

end pencil_eraser_cost_l306_306576


namespace complex_pure_imaginary_l306_306540

theorem complex_pure_imaginary (a : ℝ) : (↑a + Complex.I) / (1 - Complex.I) = 0 + b * Complex.I → a = 1 :=
by
  intro h
  -- Proof content here
  sorry

end complex_pure_imaginary_l306_306540


namespace largest_n_l306_306970

theorem largest_n (x y z n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12 → n ≤ 6 :=
by
  sorry

end largest_n_l306_306970


namespace exists_unique_pair_l306_306368

theorem exists_unique_pair (X : Set ℤ) :
  (∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n) :=
sorry

end exists_unique_pair_l306_306368


namespace difference_of_two_numbers_l306_306310

theorem difference_of_two_numbers
  (L : ℕ) (S : ℕ) 
  (hL : L = 1596) 
  (hS : 6 * S + 15 = 1596) : 
  L - S = 1333 := 
by
  sorry

end difference_of_two_numbers_l306_306310


namespace people_needed_to_mow_lawn_in_4_hours_l306_306411

-- Define the given constants and conditions
def n := 4
def t := 6
def c := n * t -- The total work that can be done in constant hours
def t' := 4

-- Define the new number of people required to complete the work in t' hours
def n' := c / t'

-- Define the problem statement
theorem people_needed_to_mow_lawn_in_4_hours : n' - n = 2 := 
sorry

end people_needed_to_mow_lawn_in_4_hours_l306_306411


namespace max_side_length_l306_306004

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l306_306004


namespace prove_fraction_l306_306718

variables {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

def forms_arithmetic_sequence (x y z : ℝ) : Prop :=
2 * y = x + z

theorem prove_fraction
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h_ar : forms_arithmetic_sequence (a 1 + 2 * b 1) (a 3 + 4 * b 3) (a 5 + 8 * b 5)) :
  (b 3 * b 7) / (b 4 ^ 2) = 1 / 4 :=
sorry

end prove_fraction_l306_306718


namespace inequality_solution_l306_306319

open Set

theorem inequality_solution (x : ℝ) : (1 - 7 / (2 * x - 1) < 0) ↔ (1 / 2 < x ∧ x < 4) := 
by
  sorry

end inequality_solution_l306_306319


namespace john_bought_two_dozens_l306_306559

theorem john_bought_two_dozens (x : ℕ) (h₁ : 21 + 3 = x * 12) : x = 2 :=
by {
    -- Placeholder for skipping the proof since it's not required.
    sorry
}

end john_bought_two_dozens_l306_306559


namespace placements_for_nine_squares_l306_306468

-- Define the parameters and conditions of the problem
def countPlacements (n : ℕ) : ℕ := sorry

theorem placements_for_nine_squares : countPlacements 9 = 25 := sorry

end placements_for_nine_squares_l306_306468


namespace task_probability_l306_306644

theorem task_probability :
  let P1 := (3 : ℚ) / 8
  let P2_not := 2 / 5
  let P3 := 5 / 9
  let P4_not := 5 / 12
  P1 * P2_not * P3 * P4_not = 5 / 72 :=
by
  sorry

end task_probability_l306_306644


namespace last_three_digits_of_expression_l306_306517

theorem last_three_digits_of_expression : 
  let prod := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349
  (prod ^ 3) % 1000 = 976 :=
by
  sorry

end last_three_digits_of_expression_l306_306517


namespace drivers_sufficiency_and_ivan_petrovich_departure_l306_306597

def one_way_trip_mins := 2 * 60 + 40
def round_trip_mins := 2 * one_way_trip_mins
def rest_time_mins := 60

axiom driver_A_return_time : 12 * 60 + 40
axiom driver_A_next_trip_time := driver_A_return_time + rest_time_mins
axiom driver_D_departure_time : 13 * 60 + 5
axiom continuous_trips : True 

noncomputable def sufficient_drivers : nat := 4
noncomputable def ivan_petrovich_departure : nat := 10 * 60 + 40

theorem drivers_sufficiency_and_ivan_petrovich_departure :
  (sufficient_drivers = 4) ∧ (ivan_petrovich_departure = 10 * 60 + 40) := by
  sorry

end drivers_sufficiency_and_ivan_petrovich_departure_l306_306597


namespace count_integers_within_range_l306_306273

theorem count_integers_within_range : 
  ∃ (count : ℕ), count = 57 ∧ ∀ n : ℤ, -5.5 * Real.pi ≤ n ∧ n ≤ 12.5 * Real.pi → n ≥ -17 ∧ n ≤ 39 :=
by
  sorry

end count_integers_within_range_l306_306273


namespace angle_of_inclination_of_line_l306_306587

theorem angle_of_inclination_of_line (x y : ℝ) (h : x - y - 1 = 0) : 
  ∃ α : ℝ, α = π / 4 := 
sorry

end angle_of_inclination_of_line_l306_306587


namespace total_participants_l306_306471

theorem total_participants (x : ℕ) (h1 : 800 / x + 60 = 800 / (x - 3)) : x = 8 :=
sorry

end total_participants_l306_306471


namespace find_second_term_of_ratio_l306_306778

theorem find_second_term_of_ratio
  (a b c d : ℕ)
  (h1 : a = 6)
  (h2 : b = 7)
  (h3 : c = 3)
  (h4 : (a - c) * 4 < a * d) :
  d = 5 :=
by
  sorry

end find_second_term_of_ratio_l306_306778


namespace vector_difference_perpendicular_l306_306152

/-- Proof that the vector difference a - b is perpendicular to b given specific vectors a and b -/
theorem vector_difference_perpendicular {a b : ℝ × ℝ} (h_a : a = (2, 0)) (h_b : b = (1, 1)) :
  (a - b) • b = 0 :=
by
  sorry

end vector_difference_perpendicular_l306_306152


namespace B_plus_C_is_330_l306_306818

-- Definitions
def A : ℕ := 170
def B : ℕ := 300
def C : ℕ := 30

axiom h1 : A + B + C = 500
axiom h2 : A + C = 200
axiom h3 : C = 30

-- Theorem statement
theorem B_plus_C_is_330 : B + C = 330 :=
by
  sorry

end B_plus_C_is_330_l306_306818


namespace max_min_values_l306_306530

namespace ProofPrimary

-- Define the polynomial function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 36 * x + 1

-- State the interval of interest
def interval : Set ℝ := Set.Icc 1 11

-- Main theorem asserting the minimum and maximum values
theorem max_min_values : 
  (∀ x ∈ interval, f x ≥ -43 ∧ f x ≤ 2630) ∧
  (∃ x ∈ interval, f x = -43) ∧
  (∃ x ∈ interval, f x = 2630) :=
by
  sorry

end ProofPrimary

end max_min_values_l306_306530


namespace machine_production_l306_306653

theorem machine_production
  (rate_per_minute : ℕ)
  (machines_total : ℕ)
  (production_minute : ℕ)
  (machines_sub : ℕ)
  (time_minutes : ℕ)
  (total_production : ℕ) :
  machines_total * rate_per_minute = production_minute →
  rate_per_minute = production_minute / machines_total →
  machines_sub * rate_per_minute = total_production / time_minutes →
  time_minutes * total_production / time_minutes = 900 :=
by
  sorry

end machine_production_l306_306653


namespace Cody_age_is_14_l306_306682

variable (CodyGrandmotherAge CodyAge : ℕ)

theorem Cody_age_is_14 (h1 : CodyGrandmotherAge = 6 * CodyAge) (h2 : CodyGrandmotherAge = 84) : CodyAge = 14 := by
  sorry

end Cody_age_is_14_l306_306682


namespace find_k_l306_306407

theorem find_k : 
  ∀ (k : ℤ), 2^4 - 6 = 3^3 + k ↔ k = -17 :=
by sorry

end find_k_l306_306407


namespace evaluate_f_at_2_l306_306127

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem evaluate_f_at_2 : f 2 = 259 := 
by
  -- Substitute x = 2 into the polynomial and simplify the expression.
  sorry

end evaluate_f_at_2_l306_306127


namespace students_passing_course_l306_306552

theorem students_passing_course :
  let students_three_years_ago := 200
  let increase_factor := 1.5
  let students_two_years_ago := students_three_years_ago * increase_factor
  let students_last_year := students_two_years_ago * increase_factor
  let students_this_year := students_last_year * increase_factor
  students_this_year = 675 :=
by
  sorry

end students_passing_course_l306_306552


namespace part1_part2_l306_306265

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (h : m > 1) : ∃ x : ℝ, f x = (4 / (m - 1)) + m :=
by
  sorry

end part1_part2_l306_306265


namespace determine_a_b_l306_306865

theorem determine_a_b (a b : ℤ) :
  (∀ x : ℤ, x^2 + a * x + b = (x - 1) * (x + 4)) → (a = 3 ∧ b = -4) :=
by
  intro h
  sorry

end determine_a_b_l306_306865


namespace standard_deviations_below_l306_306253

variable (σ : ℝ)
variable (mean : ℝ)
variable (score98 : ℝ)
variable (score58 : ℝ)

-- Conditions translated to Lean definitions
def condition_1 : Prop := score98 = mean + 3 * σ
def condition_2 : Prop := mean = 74
def condition_3 : Prop := σ = 8

-- Target statement: Prove that the score of 58 is 2 standard deviations below the mean
theorem standard_deviations_below : condition_1 σ mean score98 → condition_2 mean → condition_3 σ → score58 = 74 - 2 * σ :=
by
  intro h1 h2 h3
  sorry

end standard_deviations_below_l306_306253


namespace sum_of_first_50_digits_is_216_l306_306640

noncomputable def sum_first_50_digits_of_fraction : Nat :=
  let repeating_block := [0, 0, 0, 9, 9, 9]
  let full_cycles := 8
  let remaining_digits := [0, 0]
  let sum_full_cycles := full_cycles * (repeating_block.sum)
  let sum_remaining_digits := remaining_digits.sum
  sum_full_cycles + sum_remaining_digits

theorem sum_of_first_50_digits_is_216 :
  sum_first_50_digits_of_fraction = 216 := by
  sorry

end sum_of_first_50_digits_is_216_l306_306640


namespace value_of_M_after_subtracting_10_percent_l306_306439

-- Define the given conditions and desired result formally in Lean 4
theorem value_of_M_after_subtracting_10_percent (M : ℝ) (h : 0.25 * M = 0.55 * 2500) :
  M - 0.10 * M = 4950 :=
by
  sorry

end value_of_M_after_subtracting_10_percent_l306_306439


namespace best_fit_model_l306_306159

-- Define the coefficients of determination for each model
noncomputable def R2_Model1 : ℝ := 0.75
noncomputable def R2_Model2 : ℝ := 0.90
noncomputable def R2_Model3 : ℝ := 0.45
noncomputable def R2_Model4 : ℝ := 0.65

-- State the theorem 
theorem best_fit_model : 
  R2_Model2 ≥ R2_Model1 ∧ 
  R2_Model2 ≥ R2_Model3 ∧ 
  R2_Model2 ≥ R2_Model4 :=
by
  sorry

end best_fit_model_l306_306159


namespace max_product_of_sum_2024_l306_306627

theorem max_product_of_sum_2024 : 
  ∃ (x y : ℤ), x + y = 2024 ∧ x * y = 1024144 :=
by
  use 1012, 1012
  split
  · sorry
  · sorry

end max_product_of_sum_2024_l306_306627


namespace simplify_and_evaluate_l306_306900

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) (h3 : x ≠ 1) :
  (x = -1) → ( (x-1) / (x^2 - 2*x + 1) / ((x^2 + x - 1) / (x-1) - (x + 1)) - 1 / (x - 2) = -2 / 3 ) :=
by 
  intro hx
  rw [hx]
  sorry

end simplify_and_evaluate_l306_306900


namespace max_side_length_of_triangle_l306_306070

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l306_306070


namespace maximum_height_l306_306211

-- Define the quadratic function h(t)
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 60

-- Define our proof problem
theorem maximum_height : ∃ t : ℝ, h t = 140 :=
by
  let t := -80 / (2 * -20)
  use t
  sorry

end maximum_height_l306_306211
