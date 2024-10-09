import Mathlib

namespace min_value_of_expression_l268_26884

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) : 
  x + 2 * y ≥ 9 + 4 * Real.sqrt 2 := 
sorry

end min_value_of_expression_l268_26884


namespace patio_perimeter_is_100_feet_l268_26841

theorem patio_perimeter_is_100_feet
  (rectangle : Prop)
  (length : ℝ)
  (width : ℝ)
  (length_eq_40 : length = 40)
  (length_eq_4_times_width : length = 4 * width) :
  2 * length + 2 * width = 100 := 
by
  sorry

end patio_perimeter_is_100_feet_l268_26841


namespace symmetric_points_sum_l268_26880

variable {p q : ℤ}

theorem symmetric_points_sum (h1 : p = -6) (h2 : q = 2) : p + q = -4 := by
  sorry

end symmetric_points_sum_l268_26880


namespace necessarily_positive_l268_26811

theorem necessarily_positive (x y z : ℝ) (h1 : 0 < x ∧ x < 2) (h2 : -2 < y ∧ y < 0) (h3 : 0 < z ∧ z < 3) : 
  y + 2 * z > 0 := 
sorry

end necessarily_positive_l268_26811


namespace find_pairs_gcd_lcm_l268_26881

theorem find_pairs_gcd_lcm : 
  { (a, b) : ℕ × ℕ | Nat.gcd a b = 24 ∧ Nat.lcm a b = 360 } = {(24, 360), (72, 120)} := 
by
  sorry

end find_pairs_gcd_lcm_l268_26881


namespace elena_pens_l268_26874

theorem elena_pens (X Y : ℕ) (h1 : X + Y = 12) (h2 : 4*X + 22*Y = 420) : X = 9 := by
  sorry

end elena_pens_l268_26874


namespace largest_number_of_cakes_without_ingredients_l268_26864

theorem largest_number_of_cakes_without_ingredients :
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  ∃ (max_no_ingredients : ℕ), max_no_ingredients = 24 :=
by
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  existsi (60 - max 20 (max 30 (max 36 6))) -- max value should be used to reflect maximum coverage content
  sorry -- Proof to be completed

end largest_number_of_cakes_without_ingredients_l268_26864


namespace work_b_alone_l268_26893

theorem work_b_alone (a b : ℕ) (h1 : 2 * b = a) (h2 : a + b = 3) (h3 : (a + b) * 11 = 33) : 33 = 33 :=
by
  -- sorry is used here because we are skipping the actual proof
  sorry

end work_b_alone_l268_26893


namespace mall_incur_1_percent_loss_l268_26845

theorem mall_incur_1_percent_loss
  (a b x : ℝ)
  (ha : x = a * 1.1)
  (hb : x = b * 0.9) :
  (2 * x - (a + b)) / (a + b) = -0.01 :=
sorry

end mall_incur_1_percent_loss_l268_26845


namespace least_positive_integer_fac_6370_factorial_l268_26877

theorem least_positive_integer_fac_6370_factorial :
  ∃ (n : ℕ), (∀ m : ℕ, (6370 ∣ m.factorial) → m ≥ n) ∧ n = 14 :=
by
  sorry

end least_positive_integer_fac_6370_factorial_l268_26877


namespace haley_marble_distribution_l268_26857

theorem haley_marble_distribution (total_marbles : ℕ) (num_boys : ℕ) (h1 : total_marbles = 20) (h2 : num_boys = 2) : (total_marbles / num_boys) = 10 := 
by 
  sorry

end haley_marble_distribution_l268_26857


namespace rebecca_income_percentage_l268_26868

-- Define Rebecca's initial income
def rebecca_initial_income : ℤ := 15000
-- Define Jimmy's income
def jimmy_income : ℤ := 18000
-- Define the increase in Rebecca's income
def rebecca_income_increase : ℤ := 7000

-- Define the new income for Rebecca after increase
def rebecca_new_income : ℤ := rebecca_initial_income + rebecca_income_increase
-- Define the new combined income
def new_combined_income : ℤ := rebecca_new_income + jimmy_income

-- State the theorem to prove that Rebecca's new income is 55% of the new combined income
theorem rebecca_income_percentage : 
  (rebecca_new_income * 100) / new_combined_income = 55 :=
sorry

end rebecca_income_percentage_l268_26868


namespace find_kids_l268_26866

theorem find_kids (A K : ℕ) (h1 : A + K = 12) (h2 : 3 * A = 15) : K = 7 :=
sorry

end find_kids_l268_26866


namespace find_value_of_a_l268_26801

def pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem find_value_of_a (a : ℝ) :
  pure_imaginary ((a^3 - a) + (a / (1 - a)) * Complex.I) ↔ a = -1 := 
sorry

end find_value_of_a_l268_26801


namespace simplify_fraction_product_l268_26806

theorem simplify_fraction_product : 
  (21 / 28) * (14 / 33) * (99 / 42) = 1 := 
by 
  sorry

end simplify_fraction_product_l268_26806


namespace same_terminal_side_l268_26896

open Real

theorem same_terminal_side (k : ℤ) : (∃ k : ℤ, k * 360 - 315 = 9 / 4 * 180) :=
by
  sorry

end same_terminal_side_l268_26896


namespace solve_problem_l268_26844

theorem solve_problem : 
  ∃ p q : ℝ, 
    (p ≠ q) ∧ 
    ((∀ x : ℝ, (x = p ∨ x = q) ↔ (x-4)*(x+4) = 24*x - 96)) ∧ 
    (p > q) ∧ 
    (p - q = 16) :=
by
  sorry

end solve_problem_l268_26844


namespace fraction_of_canvas_painted_blue_l268_26872

noncomputable def square_canvas_blue_fraction : ℚ :=
  sorry

theorem fraction_of_canvas_painted_blue :
  square_canvas_blue_fraction = 3 / 8 :=
  sorry

end fraction_of_canvas_painted_blue_l268_26872


namespace race_problem_l268_26849

theorem race_problem 
    (d : ℕ) (a1 : ℕ) (a2 : ℕ) 
    (h1 : d = 60)
    (h2 : a1 = 10)
    (h3 : a2 = 20) 
    (const_speed : ∀ (x y z : ℕ), x * y = z → y ≠ 0 → x = z / y) :
  (d - d * (d - a1) / (d - a2) = 12) := 
by {
  sorry
}

end race_problem_l268_26849


namespace min_sum_of_positive_real_solution_l268_26821

theorem min_sum_of_positive_real_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y = 6 := 
by {
  sorry
}

end min_sum_of_positive_real_solution_l268_26821


namespace total_pizzas_served_l268_26889

-- Define the conditions
def pizzas_lunch : Nat := 9
def pizzas_dinner : Nat := 6

-- Define the theorem to prove
theorem total_pizzas_served : pizzas_lunch + pizzas_dinner = 15 := by
  sorry

end total_pizzas_served_l268_26889


namespace sum_of_denominators_of_fractions_l268_26825

theorem sum_of_denominators_of_fractions {a b : ℕ} (ha : 3 * a / 5 * b + 2 * a / 9 * b + 4 * a / 15 * b = 28 / 45) (gcd_ab : Nat.gcd a b = 1) :
  5 * b + 9 * b + 15 * b = 203 := sorry

end sum_of_denominators_of_fractions_l268_26825


namespace max_red_socks_l268_26808

theorem max_red_socks (r b g t : ℕ) (h1 : t ≤ 2500) (h2 : r + b + g = t) 
  (h3 : (r * (r - 1) + b * (b - 1) + g * (g - 1)) = (2 / 3) * t * (t - 1)) : 
  r ≤ 1625 :=
by 
  sorry

end max_red_socks_l268_26808


namespace jill_runs_more_than_jack_l268_26826

noncomputable def streetWidth : ℝ := 15 -- Street width in feet
noncomputable def blockSide : ℝ := 300 -- Side length of the block in feet

noncomputable def jacksPerimeter : ℝ := 4 * blockSide -- Perimeter of Jack's running path
noncomputable def jillsPerimeter : ℝ := 4 * (blockSide + 2 * streetWidth) -- Perimeter of Jill's running path on the opposite side of the street

theorem jill_runs_more_than_jack :
  jillsPerimeter - jacksPerimeter = 120 :=
by
  sorry

end jill_runs_more_than_jack_l268_26826


namespace rate_of_drawing_barbed_wire_is_correct_l268_26800

noncomputable def rate_of_drawing_barbed_wire (area cost: ℝ) (gate_width barbed_wire_extension: ℝ) : ℝ :=
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_barbed_wire := (perimeter - 2 * gate_width) + 4 * barbed_wire_extension
  cost / total_barbed_wire

theorem rate_of_drawing_barbed_wire_is_correct :
  rate_of_drawing_barbed_wire 3136 666 1 3 = 2.85 :=
by
  sorry

end rate_of_drawing_barbed_wire_is_correct_l268_26800


namespace find_tangent_point_l268_26885

theorem find_tangent_point (x : ℝ) (y : ℝ) (h_curve : y = x^2) (h_slope : 2 * x = 1) : 
    (x, y) = (1/2, 1/4) :=
sorry

end find_tangent_point_l268_26885


namespace set_inter_complement_l268_26828

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

end set_inter_complement_l268_26828


namespace cos_beta_l268_26809

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.cos α = 3/5)
  (h2 : Real.cos (α + β) = -5/13) : Real.cos β = 33/65 := 
sorry

end cos_beta_l268_26809


namespace fill_pool_time_l268_26834

theorem fill_pool_time (pool_volume : ℕ := 32000) 
                       (num_hoses : ℕ := 5) 
                       (flow_rate_per_hose : ℕ := 4) 
                       (operation_minutes : ℕ := 45) 
                       (maintenance_minutes : ℕ := 15) 
                       : ℕ :=
by
  -- Calculation steps will go here in the actual proof
  sorry

example : fill_pool_time = 47 := by
  -- Proof of the theorem fill_pool_time here
  sorry

end fill_pool_time_l268_26834


namespace savings_by_december_l268_26851

-- Define the basic conditions
def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4

-- Define the final savings calculation
def final_savings : ℕ := initial_savings + total_income - total_expenses

-- The theorem to be proved
theorem savings_by_december : final_savings = 1340840 := by
  -- Proof placeholder
  sorry

end savings_by_december_l268_26851


namespace ratio_of_beef_to_pork_l268_26853

/-- 
James buys 20 pounds of beef. 
James buys an unknown amount of pork. 
James uses 1.5 pounds of meat to make each meal. 
Each meal sells for $20. 
James made $400 from selling meals.
The ratio of the amount of beef to the amount of pork James bought is 2:1.
-/
theorem ratio_of_beef_to_pork (beef pork : ℝ) (meal_weight : ℝ) (meal_price : ℝ) (total_revenue : ℝ)
  (h_beef : beef = 20)
  (h_meal_weight : meal_weight = 1.5)
  (h_meal_price : meal_price = 20)
  (h_total_revenue : total_revenue = 400) :
  (beef / pork) = 2 :=
by
  sorry

end ratio_of_beef_to_pork_l268_26853


namespace turtle_reaches_waterhole_28_minutes_after_meeting_l268_26839

theorem turtle_reaches_waterhole_28_minutes_after_meeting (x : ℝ) (distance_lion1 : ℝ := 5 * x) 
  (speed_lion2 : ℝ := 1.5 * x) (distance_turtle : ℝ := 30) (speed_turtle : ℝ := 1/30) : 
  ∃ t_meeting : ℝ, t_meeting = 2 ∧ (distance_turtle - speed_turtle * t_meeting) / speed_turtle = 28 :=
by 
  sorry

end turtle_reaches_waterhole_28_minutes_after_meeting_l268_26839


namespace sqrt_of_sum_eq_l268_26835

noncomputable def cube_term : ℝ := 2 ^ 3
noncomputable def sum_cubes : ℝ := cube_term + cube_term + cube_term + cube_term
noncomputable def sqrt_sum : ℝ := Real.sqrt sum_cubes

theorem sqrt_of_sum_eq :
  sqrt_sum = 4 * Real.sqrt 2 :=
by
  sorry

end sqrt_of_sum_eq_l268_26835


namespace system_of_equations_solution_l268_26815

theorem system_of_equations_solution :
  ∃ (X Y: ℝ), 
    (X^2 * Y^2 + X * Y^2 + X^2 * Y + X * Y + X + Y + 3 = 0) ∧ 
    (X^2 * Y + X * Y + 1 = 0) ∧ 
    (X = -2) ∧ (Y = -1/2) :=
by
  sorry

end system_of_equations_solution_l268_26815


namespace find_matrix_A_l268_26842

-- Let A be a 2x2 matrix such that 
def A (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

theorem find_matrix_A :
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ,
  (A.mulVec ![4, 1] = ![8, 14]) ∧ (A.mulVec ![2, -3] = ![-2, 11]) ∧
  A = ![![2, 1/2], ![-1, -13/3]] :=
by
  sorry

end find_matrix_A_l268_26842


namespace number_of_tangent_and_parallel_lines_l268_26822

theorem number_of_tangent_and_parallel_lines (p : ℝ × ℝ) (a : ℝ) (h : p = (2, 4)) (hp_on_parabola : (p.1)^2 = 8 * p.2) :
  ∃ l1 l2 : (ℝ × ℝ) → Prop, 
    (l1 (2, 4) ∧ l2 (2, 4)) ∧ 
    (∀ l, (l = l1 ∨ l = l2) ↔ (∃ q, q ≠ p ∧ q ∈ {p' | (p'.1)^2 = 8 * p'.2})) ∧ 
    (∀ p' ∈ {p' | (p'.1)^2 = 8 * p'.2}, (l1 p' ∨ l2 p') → False) :=
sorry

end number_of_tangent_and_parallel_lines_l268_26822


namespace solve_for_vee_l268_26830

theorem solve_for_vee (vee : ℝ) (h : 4 * vee ^ 2 = 144) : vee = 6 ∨ vee = -6 :=
by
  -- We state that this theorem should be true for all vee and given the condition h
  sorry

end solve_for_vee_l268_26830


namespace Tim_total_money_l268_26867

theorem Tim_total_money :
  let nickels_amount := 3 * 0.05
  let dimes_amount_shoes := 13 * 0.10
  let shining_shoes := nickels_amount + dimes_amount_shoes
  let dimes_amount_tip_jar := 7 * 0.10
  let half_dollars_amount := 9 * 0.50
  let tip_jar := dimes_amount_tip_jar + half_dollars_amount
  let total := shining_shoes + tip_jar
  total = 6.65 :=
by
  sorry

end Tim_total_money_l268_26867


namespace largest_n_is_253_l268_26898

-- Define the triangle property for a set
def triangle_property (s : Set ℕ) : Prop :=
∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a < b → b < c → c < a + b

-- Define the problem statement
def largest_possible_n (n : ℕ) : Prop :=
∀ (s : Finset ℕ), (∀ (x : ℕ), x ∈ s → 4 ≤ x ∧ x ≤ n) → (s.card = 10 → triangle_property s)

-- The given proof problem
theorem largest_n_is_253 : largest_possible_n 253 :=
by
  sorry

end largest_n_is_253_l268_26898


namespace boat_problem_l268_26803

theorem boat_problem (x y : ℕ) (h : 12 * x + 5 * y = 99) :
  (x = 2 ∧ y = 15) ∨ (x = 7 ∧ y = 3) :=
sorry

end boat_problem_l268_26803


namespace find_first_number_l268_26820

theorem find_first_number (x y : ℝ) (h1 : x + y = 50) (h2 : 2 * (x - y) = 20) : x = 30 :=
by
  sorry

end find_first_number_l268_26820


namespace sum_series_l268_26876

noncomputable def series_sum := (∑' n : ℕ, (4 * (n + 1) - 2) / 3^(n + 1))

theorem sum_series : series_sum = 4 := by
  sorry

end sum_series_l268_26876


namespace g_at_neg2_eq_8_l268_26855

-- Define the functions f and g
def f (x : ℤ) : ℤ := 4 * x - 6
def g (y : ℤ) : ℤ := 3 * (y + 6/4)^2 + 4 * (y + 6/4) + 1

-- Statement of the math proof problem:
theorem g_at_neg2_eq_8 : g (-2) = 8 := 
by 
  sorry

end g_at_neg2_eq_8_l268_26855


namespace factors_of_180_multiple_of_15_count_l268_26861

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l268_26861


namespace each_child_plays_equally_l268_26894

theorem each_child_plays_equally (total_time : ℕ) (num_children : ℕ)
  (play_group_size : ℕ) (play_time : ℕ) :
  num_children = 6 ∧ play_group_size = 3 ∧ total_time = 120 ∧ play_time = (total_time * play_group_size) / num_children →
  play_time = 60 :=
by
  intros h
  sorry

end each_child_plays_equally_l268_26894


namespace part1_part2_l268_26848

noncomputable def f (x : ℝ) := |x - 3| + |x - 4|

theorem part1 (a : ℝ) (h : ∃ x : ℝ, f x < a) : a > 1 :=
sorry

theorem part2 (x : ℝ) : f x ≥ 7 + 7 * x - x ^ 2 ↔ x ≤ 0 ∨ 7 ≤ x :=
sorry

end part1_part2_l268_26848


namespace number_of_distinct_d_l268_26807

noncomputable def calculateDistinctValuesOfD (u v w x : ℂ) (h_distinct : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x) : ℕ := 
by
  sorry

theorem number_of_distinct_d (u v w x : ℂ) (h : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x)
    (h_eqs : ∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
             (z - (d * u)) * (z - (d * v)) * (z - (d * w)) * (z - (d * x))) : 
    calculateDistinctValuesOfD u v w x h = 4 :=
by
  sorry

end number_of_distinct_d_l268_26807


namespace bd_le_q2_l268_26863

theorem bd_le_q2 (a b c d p q : ℝ) (h1 : a * b + c * d = 2 * p * q) (h2 : a * c ≥ p^2 ∧ p^2 > 0) : b * d ≤ q^2 :=
sorry

end bd_le_q2_l268_26863


namespace smallest_root_of_quadratic_l268_26802

theorem smallest_root_of_quadratic (y : ℝ) (h : 4 * y^2 - 7 * y + 3 = 0) : y = 3 / 4 :=
sorry

end smallest_root_of_quadratic_l268_26802


namespace eggs_per_basket_l268_26886

theorem eggs_per_basket (n : ℕ) (total_eggs_red total_eggs_orange min_eggs_per_basket : ℕ) (h_red : total_eggs_red = 20) (h_orange : total_eggs_orange = 30) (h_min : min_eggs_per_basket = 5) (h_div_red : total_eggs_red % n = 0) (h_div_orange : total_eggs_orange % n = 0) (h_at_least : n ≥ min_eggs_per_basket) : n = 5 :=
sorry

end eggs_per_basket_l268_26886


namespace arithmetic_sequence_sum_l268_26843

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S_9 : ℚ) 
  (h_arith : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a2_a8 : a 2 + a 8 = 4 / 3) :
  S_9 = 6 :=
by
  sorry

end arithmetic_sequence_sum_l268_26843


namespace nectar_water_percentage_l268_26899

-- Definitions as per conditions
def nectar_weight : ℝ := 1.2
def honey_weight : ℝ := 1
def honey_water_ratio : ℝ := 0.4

-- Final statement to prove
theorem nectar_water_percentage : (honey_weight * honey_water_ratio + (nectar_weight - honey_weight)) / nectar_weight = 0.5 := by
  sorry

end nectar_water_percentage_l268_26899


namespace find_x_l268_26823

-- Definitions of the conditions
def eq1 (x y z : ℕ) : Prop := x + y + z = 25
def eq2 (y z : ℕ) : Prop := y + z = 14

-- Statement of the mathematically equivalent proof problem
theorem find_x (x y z : ℕ) (h1 : eq1 x y z) (h2 : eq2 y z) : x = 11 :=
by {
  -- This is where the proof would go, but we can omit it for now:
  sorry
}

end find_x_l268_26823


namespace intersection_empty_l268_26814

noncomputable def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
noncomputable def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem intersection_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_empty_l268_26814


namespace pure_alcohol_addition_problem_l268_26856

-- Define the initial conditions
def initial_volume := 6
def initial_concentration := 0.30
def final_concentration := 0.50

-- Define the amount of pure alcohol to be added
def x := 2.4

-- Proof problem statement
theorem pure_alcohol_addition_problem (initial_volume initial_concentration final_concentration x : ℝ) :
  initial_volume * initial_concentration + x = final_concentration * (initial_volume + x) :=
by
  -- Initial condition values definition
  let initial_volume := 6
  let initial_concentration := 0.30
  let final_concentration := 0.50
  let x := 2.4
  -- Skip the proof
  sorry

end pure_alcohol_addition_problem_l268_26856


namespace cannot_factor_polynomial_l268_26895

theorem cannot_factor_polynomial (a b c d : ℤ) :
  ¬(x^4 + 3 * x^3 + 6 * x^2 + 9 * x + 12 = (x^2 + a * x + b) * (x^2 + c * x + d)) := 
by {
  sorry
}

end cannot_factor_polynomial_l268_26895


namespace simplify_evaluate_l268_26850

theorem simplify_evaluate (x y : ℝ) (h : (x - 2)^2 + |1 + y| = 0) : 
  ((x - y) * (x + 2 * y) - (x + y)^2) / y = 1 :=
by
  sorry

end simplify_evaluate_l268_26850


namespace linear_function_quadrant_l268_26827

theorem linear_function_quadrant (x y : ℝ) : 
  y = 2 * x - 3 → ¬ ((x < 0 ∧ y > 0)) := 
sorry

end linear_function_quadrant_l268_26827


namespace brady_work_hours_l268_26883

theorem brady_work_hours (A : ℕ) :
    (A * 30 + 5 * 30 + 8 * 30 = 3 * 190) → 
    A = 6 :=
by sorry

end brady_work_hours_l268_26883


namespace range_of_a_l268_26804

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (2 * a - 3) * x - 1 else x ^ 2 + 1

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (3 / 2 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l268_26804


namespace sample_mean_and_variance_l268_26865

def sample : List ℕ := [10, 12, 9, 14, 13]
def n : ℕ := 5

-- Definition of sample mean
noncomputable def sampleMean : ℝ := (sample.sum / n)

-- Definition of sample variance using population formula
noncomputable def sampleVariance : ℝ := (sample.map (λ x_i => (x_i - sampleMean)^2)).sum / n

theorem sample_mean_and_variance :
  sampleMean = 11.6 ∧ sampleVariance = 3.44 := by
  sorry

end sample_mean_and_variance_l268_26865


namespace translate_parabola_l268_26819

theorem translate_parabola :
  (∀ x : ℝ, (y : ℝ) = 6 * x^2 -> y = 6 * (x + 2)^2 + 3) :=
by
  sorry

end translate_parabola_l268_26819


namespace reyn_pieces_l268_26875

-- Define the conditions
variables (total_pieces : ℕ) (pieces_each : ℕ) (pieces_left : ℕ)
variables (R : ℕ) (Rhys : ℕ) (Rory : ℕ)

-- Initial Conditions
def mrs_young_conditions :=
  total_pieces = 300 ∧
  pieces_each = total_pieces / 3 ∧
  Rhys = 2 * R ∧
  Rory = 3 * R ∧
  6 * R + pieces_left = total_pieces ∧
  pieces_left = 150

-- The statement of our proof goal
theorem reyn_pieces (h : mrs_young_conditions total_pieces pieces_each pieces_left R Rhys Rory) : R = 25 :=
sorry

end reyn_pieces_l268_26875


namespace simplify_expression_l268_26888

-- Define the variables a and b as real numbers
variables (a b : ℝ)

-- Define the expression and the simplified expression
def original_expr := -a^2 * (-2 * a * b) + 3 * a * (a^2 * b - 1)
def simplified_expr := 5 * a^3 * b - 3 * a

-- Statement that the original expression is equal to the simplified expression
theorem simplify_expression : original_expr a b = simplified_expr a b :=
by
  sorry

end simplify_expression_l268_26888


namespace solve_for_x_l268_26817

-- Define the quadratic equation condition
def quadratic_eq (x : ℝ) : Prop := 3 * x^2 - 7 * x - 6 = 0

-- The main theorem to prove
theorem solve_for_x (x : ℝ) : x > 0 ∧ quadratic_eq x → x = 3 := by
  sorry

end solve_for_x_l268_26817


namespace determine_amount_of_substance_l268_26878

noncomputable def amount_of_substance 
  (A : ℝ) (R : ℝ) (delta_T : ℝ) : ℝ :=
  (2 * A) / (R * delta_T)

theorem determine_amount_of_substance 
  (A : ℝ := 831) 
  (R : ℝ := 8.31) 
  (delta_T : ℝ := 100) 
  (nu : ℝ := amount_of_substance A R delta_T) :
  nu = 2 := by
  -- Conditions rewritten as definitions
  -- Definition: A = 831 J
  -- Definition: R = 8.31 J/(mol * K)
  -- Definition: delta_T = 100 K
  -- The correct answer to be proven: nu = 2 mol
  sorry

end determine_amount_of_substance_l268_26878


namespace arithmetic_seq_sum_equidistant_l268_26870

variable (a : ℕ → ℤ)

theorem arithmetic_seq_sum_equidistant :
  (∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) → a 4 = 12 → a 1 + a 7 = 24 :=
by
  intros h_seq h_a4
  sorry

end arithmetic_seq_sum_equidistant_l268_26870


namespace jose_julia_completion_time_l268_26891

variable (J N L : ℝ)

theorem jose_julia_completion_time :
  J + N + L = 1/4 ∧
  J * (1/3) = 1/18 ∧
  N = 1/9 ∧
  L * (1/3) = 1/18 →
  1/J = 6 ∧ 1/L = 6 := sorry

end jose_julia_completion_time_l268_26891


namespace route_down_distance_l268_26854

-- Definitions
def rate_up : ℝ := 7
def time_up : ℝ := 2
def distance_up : ℝ := rate_up * time_up
def rate_down : ℝ := 1.5 * rate_up
def time_down : ℝ := time_up
def distance_down : ℝ := rate_down * time_down

-- Theorem
theorem route_down_distance : distance_down = 21 := by
  sorry

end route_down_distance_l268_26854


namespace pythagorean_numbers_b_l268_26836

-- Define Pythagorean numbers and conditions
variable (a b c m : ℕ)
variable (h1 : a = 1/2 * m^2 - 1/2)
variable (h2 : c = 1/2 * m^2 + 1/2)
variable (h3 : m > 1 ∧ ¬ even m)

theorem pythagorean_numbers_b (h4 : c^2 = a^2 + b^2) : b = m :=
sorry

end pythagorean_numbers_b_l268_26836


namespace cubic_inches_needed_l268_26858

/-- The dimensions of each box are 20 inches by 20 inches by 12 inches. -/
def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 12

/-- The cost of each box is $0.40. -/
def box_cost : ℝ := 0.40

/-- The minimum spending required by the university on boxes is $200. -/
def min_spending : ℝ := 200

/-- Given the above conditions, the total cubic inches needed to package the collection is 2,400,000 cubic inches. -/
theorem cubic_inches_needed :
  (min_spending / box_cost) * (box_length * box_width * box_height) = 2400000 := by
  sorry

end cubic_inches_needed_l268_26858


namespace original_cost_of_each_bag_l268_26859

theorem original_cost_of_each_bag (C : ℕ) (hC : C % 13 = 0) (h4 : (85 * C) % 400 = 0) : C / 5 = 208 := by
  sorry

end original_cost_of_each_bag_l268_26859


namespace local_maximum_no_global_maximum_equation_root_condition_l268_26831

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x^2 + 2*x - 3) * Real.exp x

theorem local_maximum_no_global_maximum : (∃ x0 : ℝ, f' x0 = 0 ∧ (∀ x < x0, f x < f x0) ∧ (∀ x > x0, f x < f x0))
∧ (f 1 = -2 * Real.exp 1) 
∧ (∀ x : ℝ, ∃ b : ℝ, f x = b ∧ b > 6 * Real.exp (-3) → ¬(f x = f 1))
:= sorry

theorem equation_root_condition (b : ℝ) : (∃ x1 x2 x3 : ℝ, f x1 = b ∧ f x2 = b ∧ f x3 = b ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) 
→ (0 < b ∧ b < 6 * Real.exp (-3))
:= sorry

end local_maximum_no_global_maximum_equation_root_condition_l268_26831


namespace train_crossing_time_l268_26892

noncomputable def length_first_train : ℝ := 200  -- meters
noncomputable def speed_first_train_kmph : ℝ := 72  -- km/h
noncomputable def speed_first_train : ℝ := speed_first_train_kmph * (1000 / 3600)  -- m/s

noncomputable def length_second_train : ℝ := 300  -- meters
noncomputable def speed_second_train_kmph : ℝ := 36  -- km/h
noncomputable def speed_second_train : ℝ := speed_second_train_kmph * (1000 / 3600)  -- m/s

noncomputable def relative_speed : ℝ := speed_first_train - speed_second_train -- m/s
noncomputable def total_length : ℝ := length_first_train + length_second_train  -- meters
noncomputable def time_to_cross : ℝ := total_length / relative_speed  -- seconds

theorem train_crossing_time :
  time_to_cross = 50 := by
  sorry

end train_crossing_time_l268_26892


namespace isosceles_trapezoid_AB_length_l268_26890

theorem isosceles_trapezoid_AB_length (BC AD : ℝ) (r : ℝ) (a : ℝ) (h_isosceles : BC = a) (h_ratio : AD = 3 * a) (h_area : 4 * a * r = Real.sqrt 3 / 2) (h_radius : r = a * Real.sqrt 3 / 2) :
  2 * a = 1 :=
by
 sorry

end isosceles_trapezoid_AB_length_l268_26890


namespace Chloe_final_points_l268_26812

-- Define the points scored (or lost) in each round
def round1_points : ℤ := 40
def round2_points : ℤ := 50
def round3_points : ℤ := 60
def round4_points : ℤ := 70
def round5_points : ℤ := -4
def round6_points : ℤ := 80
def round7_points : ℤ := -6

-- Statement to prove: Chloe's total points at the end of the game
theorem Chloe_final_points : 
  round1_points + round2_points + round3_points + round4_points + round5_points + round6_points + round7_points = 290 :=
by
  sorry

end Chloe_final_points_l268_26812


namespace bike_distance_difference_l268_26832

-- Defining constants for Alex's and Bella's rates and the time duration
def Alex_rate : ℕ := 12
def Bella_rate : ℕ := 10
def time : ℕ := 6

-- The goal is to prove the difference in distance is 12 miles
theorem bike_distance_difference : (Alex_rate * time) - (Bella_rate * time) = 12 := by
  sorry

end bike_distance_difference_l268_26832


namespace fish_count_when_james_discovers_l268_26873

def fish_in_aquarium (initial_fish : ℕ) (bobbit_worm_eats : ℕ) (predatory_fish_eats : ℕ)
  (reproduction_rate : ℕ × ℕ) (days_1 : ℕ) (added_fish: ℕ) (days_2 : ℕ) : ℕ :=
  let predation_rate := bobbit_worm_eats + predatory_fish_eats
  let total_eaten_in_14_days := predation_rate * days_1
  let reproduction_events_in_14_days := days_1 / reproduction_rate.snd
  let fish_born_in_14_days := reproduction_events_in_14_days * reproduction_rate.fst
  let fish_after_14_days := initial_fish - total_eaten_in_14_days + fish_born_in_14_days
  let fish_after_14_days_non_negative := max fish_after_14_days 0
  let fish_after_addition := fish_after_14_days_non_negative + added_fish
  let total_eaten_in_7_days := predation_rate * days_2
  let reproduction_events_in_7_days := days_2 / reproduction_rate.snd
  let fish_born_in_7_days := reproduction_events_in_7_days * reproduction_rate.fst
  let fish_after_7_days := fish_after_addition - total_eaten_in_7_days + fish_born_in_7_days
  max fish_after_7_days 0

theorem fish_count_when_james_discovers :
  fish_in_aquarium 60 2 3 (2, 3) 14 8 7 = 4 :=
sorry

end fish_count_when_james_discovers_l268_26873


namespace number_of_bass_caught_l268_26838

/-
Statement:
Given:
1. An eight-pound trout.
2. Two twelve-pound salmon.
3. They need to feed 22 campers with two pounds of fish each.
Prove that the number of two-pound bass caught is 6.
-/

theorem number_of_bass_caught
  (weight_trout : ℕ := 8)
  (weight_salmon : ℕ := 12)
  (num_salmon : ℕ := 2)
  (num_campers : ℕ := 22)
  (required_per_camper : ℕ := 2)
  (weight_bass : ℕ := 2) :
  (num_campers * required_per_camper - (weight_trout + num_salmon * weight_salmon)) / weight_bass = 6 :=
by
  sorry  -- Proof to be completed

end number_of_bass_caught_l268_26838


namespace at_least_one_ge_one_l268_26829

theorem at_least_one_ge_one (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 2) : 
  max (|a - b|) (max (|b - c|) (|c - a|)) ≥ 1 :=
by 
  sorry

end at_least_one_ge_one_l268_26829


namespace find_f_neg1_l268_26833

-- Definitions based on conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable {b : ℝ} (f : ℝ → ℝ)

axiom odd_f : odd_function f
axiom f_form : ∀ x, 0 ≤ x → f x = 2^(x + 1) + 2 * x + b
axiom b_value : b = -2

theorem find_f_neg1 : f (-1) = -4 :=
sorry

end find_f_neg1_l268_26833


namespace max_value_8a_3b_5c_l268_26810

theorem max_value_8a_3b_5c (a b c : ℝ) (h_condition : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ (Real.sqrt 373) / 6 :=
by
  sorry

end max_value_8a_3b_5c_l268_26810


namespace ring_display_capacity_l268_26824

def necklace_capacity : ℕ := 12
def current_necklaces : ℕ := 5
def ring_capacity : ℕ := 18
def bracelet_capacity : ℕ := 15
def current_bracelets : ℕ := 8
def necklace_cost : ℕ := 4
def ring_cost : ℕ := 10
def bracelet_cost : ℕ := 5
def total_cost : ℕ := 183

theorem ring_display_capacity : ring_capacity + (total_cost - ((necklace_capacity - current_necklaces) * necklace_cost + (bracelet_capacity - current_bracelets) * bracelet_cost)) / ring_cost = 30 := by
  sorry

end ring_display_capacity_l268_26824


namespace sum_of_first_three_terms_l268_26846

theorem sum_of_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 8) 
  (h5 : a 5 = 12) 
  (h6 : a 6 = 16) : 
  a 1 + a 2 + a 3 = 0 :=
by
  sorry

end sum_of_first_three_terms_l268_26846


namespace refrigerator_profit_l268_26862

theorem refrigerator_profit 
  (marked_price : ℝ) 
  (cost_price : ℝ) 
  (profit_margin : ℝ ) 
  (discount1 : ℝ) 
  (profit1 : ℝ)
  (discount2 : ℝ):
  profit_margin = 0.1 → 
  profit1 = 200 → 
  cost_price = 2000 → 
  discount1 = 0.8 → 
  discount2 = 0.85 → 
  discount1 * marked_price - cost_price = profit1 → 

  (discount2 * marked_price - cost_price) = 337.5 := 
by 
  intros; 
  let marked_price := 2750; 
  sorry

end refrigerator_profit_l268_26862


namespace find_side_b_in_triangle_l268_26840

theorem find_side_b_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180) 
  (h3 : a + c = 8) 
  (h4 : a * c = 15) 
  (h5 : (b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos (B * Real.pi / 180))) : 
  b = Real.sqrt 19 := 
  by sorry

end find_side_b_in_triangle_l268_26840


namespace arrangement_of_letters_l268_26837

-- Define the set of letters with subscripts
def letters : Finset String := {"B", "A₁", "B₁", "A₂", "B₂", "A₃"}

-- Define the number of ways to arrange 6 distinct letters
theorem arrangement_of_letters : letters.card.factorial = 720 := 
by {
  sorry
}

end arrangement_of_letters_l268_26837


namespace length_DC_of_ABCD_l268_26813

open Real

structure Trapezoid (ABCD : Type) :=
  (AB DC : ℝ)
  (BC : ℝ := 0)
  (angleBCD angleCDA : ℝ)

noncomputable def given_trapezoid : Trapezoid ℝ :=
{ AB := 5,
  DC := 8 + sqrt 3, -- this is from the answer
  BC := 3 * sqrt 2,
  angleBCD := π / 4,   -- 45 degrees in radians
  angleCDA := π / 3 }  -- 60 degrees in radians

variable (ABCD : Trapezoid ℝ)

theorem length_DC_of_ABCD :
  ABCD.AB = 5 ∧
  ABCD.BC = 3 * sqrt 2 ∧
  ABCD.angleBCD = π / 4 ∧
  ABCD.angleCDA = π / 3 →
  ABCD.DC = 8 + sqrt 3 :=
sorry

end length_DC_of_ABCD_l268_26813


namespace common_chord_eqn_l268_26852

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 12 * x - 2 * y - 13 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 12 * x + 16 * y - 25 = 0

-- Define the proposition stating the common chord equation
theorem common_chord_eqn : ∀ x y : ℝ, C1 x y ∧ C2 x y → 4 * x + 3 * y - 2 = 0 :=
by
  sorry

end common_chord_eqn_l268_26852


namespace relationship_of_variables_l268_26882

theorem relationship_of_variables
  (a b c d : ℚ)
  (h : (a + b) / (b + c) = (c + d) / (d + a)) :
  a = c ∨ a + b + c + d = 0 :=
by sorry

end relationship_of_variables_l268_26882


namespace find_a_l268_26879

noncomputable def M (a : ℤ) : Set ℤ := {a, 0}
noncomputable def N : Set ℤ := { x : ℤ | 2 * x^2 - 3 * x < 0 }

theorem find_a (a : ℤ) (h : (M a ∩ N).Nonempty) : a = 1 := sorry

end find_a_l268_26879


namespace kendra_fish_count_l268_26847

variable (K : ℕ) -- Number of fish Kendra caught
variable (Ken_fish : ℕ) -- Number of fish Ken brought home

-- Conditions
axiom twice_as_many : Ken_fish = 2 * K - 3
axiom total_fish : K + Ken_fish = 87

-- The theorem we need to prove
theorem kendra_fish_count : K = 30 :=
by
  -- Lean proof goes here
  sorry

end kendra_fish_count_l268_26847


namespace find_y_l268_26818

theorem find_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 := by
  sorry

end find_y_l268_26818


namespace det_transformed_matrix_l268_26805

variables {p q r s : ℝ} -- Defining the variables over the real numbers

-- Defining the first determinant condition as an axiom
axiom det_initial_matrix : (p * s - q * r) = 10

-- Stating the theorem to be proved
theorem det_transformed_matrix : 
  (p + 2 * r) * s - (q + 2 * s) * r = 10 :=
by
  sorry -- Placeholder for the actual proof

end det_transformed_matrix_l268_26805


namespace surface_area_eighth_block_l268_26816

theorem surface_area_eighth_block {A B C D E F G H : ℕ} 
  (blockA : A = 148) 
  (blockB : B = 46) 
  (blockC : C = 72) 
  (blockD : D = 28) 
  (blockE : E = 88) 
  (blockF : F = 126) 
  (blockG : G = 58) 
  : H = 22 :=
by 
  sorry

end surface_area_eighth_block_l268_26816


namespace minimum_cost_for_28_apples_l268_26860

/--
Conditions:
  - apples can be bought at a rate of 4 for 15 cents,
  - apples can be bought at a rate of 7 for 30 cents,
  - you need to buy exactly 28 apples.
Prove that the minimum total cost to buy exactly 28 apples is 120 cents.
-/
theorem minimum_cost_for_28_apples : 
  let cost_4_for_15 := 15
  let cost_7_for_30 := 30
  let apples_needed := 28
  ∃ (n m : ℕ), n * 4 + m * 7 = apples_needed ∧ n * cost_4_for_15 + m * cost_7_for_30 = 120 := sorry

end minimum_cost_for_28_apples_l268_26860


namespace probability_each_person_selected_l268_26897

-- Define the number of initial participants
def initial_participants := 2007

-- Define the number of participants to exclude
def exclude_participants := 7

-- Define the final number of participants remaining after exclusion
def remaining_participants := initial_participants - exclude_participants

-- Define the number of participants to select
def select_participants := 50

-- Define the probability of each participant being selected
def selection_probability : ℚ :=
  select_participants * remaining_participants / (initial_participants * remaining_participants)

theorem probability_each_person_selected :
  selection_probability = (50 / 2007 : ℚ) :=
sorry

end probability_each_person_selected_l268_26897


namespace rate_per_square_meter_l268_26887

theorem rate_per_square_meter 
  (L : ℝ) (W : ℝ) (C : ℝ)
  (hL : L = 5.5) 
  (hW : W = 3.75)
  (hC : C = 20625)
  : C / (L * W) = 1000 :=
by
  sorry

end rate_per_square_meter_l268_26887


namespace arithmetic_expression_evaluation_l268_26869

theorem arithmetic_expression_evaluation :
  4 * 6 + 8 * 3 - 28 / 2 = 34 := by
  sorry

end arithmetic_expression_evaluation_l268_26869


namespace initial_apples_correct_l268_26871

-- Define the conditions
def apples_handout : Nat := 5
def pies_made : Nat := 9
def apples_per_pie : Nat := 5

-- Calculate the number of apples used for pies
def apples_for_pies := pies_made * apples_per_pie

-- Define the total number of apples initially
def apples_initial := apples_for_pies + apples_handout

-- State the theorem to prove
theorem initial_apples_correct : apples_initial = 50 :=
by
  sorry

end initial_apples_correct_l268_26871
