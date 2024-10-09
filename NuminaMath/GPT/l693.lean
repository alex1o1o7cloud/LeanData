import Mathlib

namespace final_price_hat_final_price_tie_l693_69359

theorem final_price_hat (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (h_initial : initial_price = 20) 
    (h_first : first_discount = 0.25) 
    (h_second : second_discount = 0.20) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 12 := 
by 
  rw [h_initial, h_first, h_second]
  norm_num

theorem final_price_tie (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (t_initial : initial_price = 15) 
    (t_first : first_discount = 0.10) 
    (t_second : second_discount = 0.30) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 9.45 := 
by 
  rw [t_initial, t_first, t_second]
  norm_num

end final_price_hat_final_price_tie_l693_69359


namespace coordinates_of_A_l693_69391

/-- The initial point A and the transformations applied to it -/
def initial_point : Prod ℤ ℤ := (-3, 2)

def translate_right (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1 + units, p.2)

def translate_down (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1, p.2 - units)

/-- Proof that the point A' has coordinates (1, -1) -/
theorem coordinates_of_A' : 
  translate_down (translate_right initial_point 4) 3 = (1, -1) :=
by
  sorry

end coordinates_of_A_l693_69391


namespace composite_sum_of_powers_l693_69339

theorem composite_sum_of_powers (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ a^2016 + b^2016 + c^2016 + d^2016 = x * y :=
by sorry

end composite_sum_of_powers_l693_69339


namespace cos_sq_half_diff_eq_csquared_over_a2_b2_l693_69354

theorem cos_sq_half_diff_eq_csquared_over_a2_b2
  (a b c α β : ℝ)
  (h1 : a^2 + b^2 ≠ 0)
  (h2 : a * (Real.cos α) + b * (Real.sin α) = c)
  (h3 : a * (Real.cos β) + b * (Real.sin β) = c)
  (h4 : ∀ k : ℤ, α ≠ β + 2 * k * Real.pi) :
  Real.cos (α - β) / 2 = c^2 / (a^2 + b^2) :=
by
  sorry

end cos_sq_half_diff_eq_csquared_over_a2_b2_l693_69354


namespace cost_price_of_article_l693_69352

theorem cost_price_of_article (SP CP : ℝ) (h1 : SP = 150) (h2 : SP = CP + (1 / 4) * CP) : CP = 120 :=
by
  sorry

end cost_price_of_article_l693_69352


namespace lawnmower_blade_cost_l693_69342

theorem lawnmower_blade_cost (x : ℕ) : 4 * x + 7 = 39 → x = 8 :=
by
  sorry

end lawnmower_blade_cost_l693_69342


namespace total_cost_correct_l693_69337

noncomputable def totalCost : ℝ :=
  let fuel_efficiences := [15, 12, 14, 10, 13, 15]
  let distances := [10, 6, 7, 5, 3, 9]
  let gas_prices := [3.5, 3.6, 3.4, 3.55, 3.55, 3.5]
  let gas_used := distances.zip fuel_efficiences |>.map (λ p => (p.1 : ℝ) / p.2)
  let costs := gas_prices.zip gas_used |>.map (λ p => p.1 * p.2)
  costs.sum

theorem total_cost_correct : abs (totalCost - 10.52884) < 0.01 := by
  sorry

end total_cost_correct_l693_69337


namespace even_mult_expressions_divisible_by_8_l693_69390

theorem even_mult_expressions_divisible_by_8 {a : ℤ} (h : ∃ k : ℤ, a = 2 * k) :
  (8 ∣ a * (a^2 + 20)) ∧ (8 ∣ a * (a^2 - 20)) ∧ (8 ∣ a * (a^2 - 4)) := by
  sorry

end even_mult_expressions_divisible_by_8_l693_69390


namespace copper_tin_ratio_l693_69371

theorem copper_tin_ratio 
    (w1 w2 w_new : ℝ) 
    (r1_copper r1_tin r2_copper r2_tin : ℝ) 
    (r_new_copper r_new_tin : ℝ)
    (pure_copper : ℝ)
    (h1 : w1 = 10)
    (h2 : w2 = 16)
    (h3 : r1_copper = 4 / 5 * w1)
    (h4 : r1_tin = 1 / 5 * w1)
    (h5 : r2_copper = 1 / 4 * w2)
    (h6 : r2_tin = 3 / 4 * w2)
    (h7 : r_new_copper = r1_copper + r2_copper + pure_copper)
    (h8 : r_new_tin = r1_tin + r2_tin)
    (h9 : w_new = 35)
    (h10 : r_new_copper + r_new_tin + pure_copper = w_new)
    (h11 : pure_copper = 9) :
    r_new_copper / r_new_tin = 3 / 2 :=
by
  sorry

end copper_tin_ratio_l693_69371


namespace quadratic_inequality_solution_l693_69305

theorem quadratic_inequality_solution:
  ∀ x : ℝ, (x^2 + 2 * x < 3) ↔ (-3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l693_69305


namespace max_area_of_rectangle_l693_69353

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * (x + y) = 60) : x * y ≤ 225 :=
by sorry

end max_area_of_rectangle_l693_69353


namespace ravi_work_alone_days_l693_69302

theorem ravi_work_alone_days (R : ℝ) (h1 : 1 / 75 + 1 / R = 1 / 30) : R = 50 :=
sorry

end ravi_work_alone_days_l693_69302


namespace girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l693_69397

-- Definition of the primary condition
def girls := 3
def boys := 5

-- Statement for each part of the problem
theorem girls_together (A : ℕ → ℕ → ℕ) : 
  A (girls + boys - 1) girls * A girls girls = 4320 := 
sorry

theorem girls_separated (A : ℕ → ℕ → ℕ) : 
  A boys boys * A (girls + boys - 1) girls = 14400 := 
sorry

theorem girls_not_both_ends (A : ℕ → ℕ → ℕ) : 
  A boys 2 * A (girls + boys - 2) (girls + boys - 2) = 14400 := 
sorry

theorem girls_not_both_ends_simul (P : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ) : 
  P (girls + boys) (girls + boys) - A girls 2 * A (girls + boys - 2) (girls + boys - 2) = 36000 := 
sorry

end girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l693_69397


namespace neg_p_equiv_l693_69347

theorem neg_p_equiv (p : Prop) : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end neg_p_equiv_l693_69347


namespace fill_up_mini_vans_l693_69363

/--
In a fuel station, the service costs $2.20 per vehicle and every liter of fuel costs $0.70.
Assume that mini-vans have a tank size of 65 liters, and trucks have a tank size of 143 liters.
Given that 2 trucks were filled up and the total cost was $347.7,
prove the number of mini-vans filled up is 3.
-/
theorem fill_up_mini_vans (m : ℝ) (t : ℝ) 
    (service_cost_per_vehicle fuel_cost_per_liter : ℝ)
    (van_tank_size truck_tank_size total_cost : ℝ):
    service_cost_per_vehicle = 2.20 →
    fuel_cost_per_liter = 0.70 →
    van_tank_size = 65 →
    truck_tank_size = 143 →
    t = 2 →
    total_cost = 347.7 →
    (service_cost_per_vehicle * m + service_cost_per_vehicle * t) + (fuel_cost_per_liter * van_tank_size * m) + (fuel_cost_per_liter * truck_tank_size * t) = total_cost →
    m = 3 :=
by
  intros
  sorry

end fill_up_mini_vans_l693_69363


namespace determinant_identity_l693_69398

variable (x y z w : ℝ)
variable (h1 : x * w - y * z = -3)

theorem determinant_identity :
  (x + z) * w - (y + w) * z = -3 :=
by sorry

end determinant_identity_l693_69398


namespace problem_1_1_eval_l693_69379

noncomputable def E (a b c : ℝ) : ℝ :=
  let A := (1/a - 1/(b+c))/(1/a + 1/(b+c))
  let B := 1 + (b^2 + c^2 - a^2)/(2*b*c)
  let C := (a - b - c)/(a * b * c)
  (A * B) / C

theorem problem_1_1_eval :
  E 0.02 (-11.05) 1.07 = 0.1 :=
by
  -- Proof goes here
  sorry

end problem_1_1_eval_l693_69379


namespace sampling_methods_correct_l693_69380

-- Definitions of the conditions:
def is_simple_random_sampling (method : String) : Prop := 
  method = "random selection of 24 students by the student council"

def is_systematic_sampling (method : String) : Prop := 
  method = "selection of students numbered from 001 to 240 whose student number ends in 3"

-- The equivalent math proof problem:
theorem sampling_methods_correct :
  is_simple_random_sampling "random selection of 24 students by the student council" ∧
  is_systematic_sampling "selection of students numbered from 001 to 240 whose student number ends in 3" :=
by
  sorry

end sampling_methods_correct_l693_69380


namespace natalie_list_count_l693_69372

theorem natalie_list_count : ∀ n : ℕ, (15 ≤ n ∧ n ≤ 225) → ((225 - 15 + 1) = 211) :=
by
  intros n h
  sorry

end natalie_list_count_l693_69372


namespace sqrt2_minus1_mul_sqrt2_plus1_eq1_l693_69393

theorem sqrt2_minus1_mul_sqrt2_plus1_eq1 : (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 1 :=
  sorry

end sqrt2_minus1_mul_sqrt2_plus1_eq1_l693_69393


namespace lg_eight_plus_three_lg_five_l693_69360

theorem lg_eight_plus_three_lg_five : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  sorry

end lg_eight_plus_three_lg_five_l693_69360


namespace function_is_monotonic_and_odd_l693_69394

   variable (a : ℝ) (x : ℝ)

   noncomputable def f : ℝ := (a^x - a^(-x))

   theorem function_is_monotonic_and_odd (h1 : a > 0) (h2 : a ≠ 1) : 
     (∀ x : ℝ, f (-x) = -f (x)) ∧ ((a > 1 → ∀ x y : ℝ, x < y → f x < f y) ∧ (0 < a ∧ a < 1 → ∀ x y : ℝ, x < y → f x > f y)) :=
   by
         sorry
   
end function_is_monotonic_and_odd_l693_69394


namespace rectangle_clear_area_l693_69378

theorem rectangle_clear_area (EF FG : ℝ)
  (radius_E radius_F radius_G radius_H : ℝ) : 
  EF = 4 → FG = 6 → 
  radius_E = 2 → radius_F = 3 → radius_G = 1.5 → radius_H = 2.5 → 
  abs ((EF * FG) - (π * radius_E^2 / 4 + π * radius_F^2 / 4 + π * radius_G^2 / 4 + π * radius_H^2 / 4)) - 7.14 < 0.5 :=
by sorry

end rectangle_clear_area_l693_69378


namespace valid_parametrizations_l693_69375

-- Define the line as a function
def line (x : ℝ) : ℝ := -2 * x + 7

-- Define vectors and their properties
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def on_line (v : Vector2D) : Prop :=
  v.y = line v.x

def direction_vector (v1 v2 : Vector2D) : Vector2D :=
  ⟨v2.x - v1.x, v2.y - v1.y⟩

def is_multiple (v1 v2 : Vector2D) : Prop :=
  ∃ k : ℝ, v2.x = k * v1.x ∧ v2.y = k * v1.y

-- Define the given parameterizations
def param_A (t : ℝ) : Vector2D := ⟨0 + t * 5, 7 + t * 10⟩
def param_B (t : ℝ) : Vector2D := ⟨2 + t * 1, 3 + t * -2⟩
def param_C (t : ℝ) : Vector2D := ⟨7 + t * 4, 0 + t * -8⟩
def param_D (t : ℝ) : Vector2D := ⟨-1 + t * 2, 9 + t * 4⟩
def param_E (t : ℝ) : Vector2D := ⟨3 + t * 2, 1 + t * 0⟩

-- Define the theorem
theorem valid_parametrizations :
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨0, 7⟩ (param_A t)) ∧ on_line (param_A t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨2, 3⟩ (param_B t)) ∧ on_line (param_B t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨7, 0⟩ (param_C t)) ∧ on_line (param_C t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨-1, 9⟩ (param_D t)) ∧ on_line (param_D t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨3, 1⟩ (param_E t)) ∧ on_line (param_E t) → False) :=
by
  sorry

end valid_parametrizations_l693_69375


namespace always_composite_for_x64_l693_69388

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

theorem always_composite_for_x64 (n : ℕ) : is_composite (n^4 + 64) :=
by
  sorry

end always_composite_for_x64_l693_69388


namespace angle_between_generatrix_and_base_of_cone_l693_69328

theorem angle_between_generatrix_and_base_of_cone (r R H : ℝ) (α : ℝ)
  (h_cylinder_height : H = 2 * R)
  (h_total_surface_area : 2 * Real.pi * r * H + 2 * Real.pi * r^2 = Real.pi * R^2) :
  α = Real.arctan (2 * (4 + Real.sqrt 6) / 5) :=
sorry

end angle_between_generatrix_and_base_of_cone_l693_69328


namespace construct_triangle_given_side_and_medians_l693_69387

theorem construct_triangle_given_side_and_medians
  (AB : ℝ) (m_a m_b : ℝ)
  (h1 : AB > 0) (h2 : m_a > 0) (h3 : m_b > 0) :
  ∃ (A B C : ℝ × ℝ),
    (∃ G : ℝ × ℝ, 
      dist A B = AB ∧ 
      dist A G = (2 / 3) * m_a ∧
      dist B G = (2 / 3) * m_b ∧ 
      dist G (midpoint ℝ A C) = m_b / 3 ∧ 
      dist G (midpoint ℝ B C) = m_a / 3) :=
sorry

end construct_triangle_given_side_and_medians_l693_69387


namespace flower_shop_options_l693_69364

theorem flower_shop_options:
  ∃ (S : Finset (ℕ × ℕ)), (∀ p ∈ S, 2 * p.1 + 3 * p.2 = 30 ∧ p.1 > 0 ∧ p.2 > 0) ∧ S.card = 4 :=
sorry

end flower_shop_options_l693_69364


namespace lamp_height_difference_l693_69334

def old_lamp_height : ℝ := 1
def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := new_lamp_height - old_lamp_height

theorem lamp_height_difference :
  height_difference = 1.3333333333333335 := by
  sorry

end lamp_height_difference_l693_69334


namespace problem_condition_l693_69312

noncomputable def f (x b : ℝ) := Real.exp x * (x - b)
noncomputable def f_prime (x b : ℝ) := Real.exp x * (x - b + 1)
noncomputable def g (x : ℝ) := (x^2 + 2*x) / (x + 1)

theorem problem_condition (b : ℝ) :
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x b + x * f_prime x b > 0) → b < 8 / 3 :=
by
  sorry

end problem_condition_l693_69312


namespace power_function_value_at_3_l693_69315

theorem power_function_value_at_3
  (f : ℝ → ℝ)
  (h1 : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α)
  (h2 : f 2 = 1 / 4) :
  f 3 = 1 / 9 := 
sorry

end power_function_value_at_3_l693_69315


namespace lcm_of_54_96_120_150_l693_69317

theorem lcm_of_54_96_120_150 : Nat.lcm 54 (Nat.lcm 96 (Nat.lcm 120 150)) = 21600 := by
  sorry

end lcm_of_54_96_120_150_l693_69317


namespace probability_none_A_B_C_l693_69323

-- Define the probabilities as given conditions
def P_A : ℝ := 0.25
def P_B : ℝ := 0.40
def P_C : ℝ := 0.35
def P_AB : ℝ := 0.20
def P_AC : ℝ := 0.15
def P_BC : ℝ := 0.25
def P_ABC : ℝ := 0.10

-- Prove that the probability that none of the events A, B, C occur simultaneously is 0.50
theorem probability_none_A_B_C : 1 - (P_A + P_B + P_C - P_AB - P_AC - P_BC + P_ABC) = 0.50 :=
by
  sorry

end probability_none_A_B_C_l693_69323


namespace exists_a_not_divisible_l693_69308

theorem exists_a_not_divisible (p : ℕ) (hp_prime : Prime p) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ (p^2 ∣ (a^(p-1) - 1)) ∧ ¬ (p^2 ∣ ((a+1)^(p-1) - 1))) :=
  sorry

end exists_a_not_divisible_l693_69308


namespace height_of_trapezium_l693_69384

-- Define the lengths of the parallel sides
def length_side1 : ℝ := 10
def length_side2 : ℝ := 18

-- Define the given area of the trapezium
def area_trapezium : ℝ := 210

-- The distance between the parallel sides (height) we want to prove
def height_between_sides : ℝ := 15

-- State the problem as a theorem in Lean: prove that the height is correct
theorem height_of_trapezium :
  (1 / 2) * (length_side1 + length_side2) * height_between_sides = area_trapezium :=
by
  sorry

end height_of_trapezium_l693_69384


namespace values_satisfying_ggx_eq_gx_l693_69367

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem values_satisfying_ggx_eq_gx (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 1 ∨ x = 3 ∨ x = 4 :=
by
  -- The proof is omitted
  sorry

end values_satisfying_ggx_eq_gx_l693_69367


namespace smallest_sum_of_consecutive_integers_gt_420_l693_69344

theorem smallest_sum_of_consecutive_integers_gt_420 : 
  ∃ n : ℕ, (n * (n + 1) > 420) ∧ (n + (n + 1) = 43) := sorry

end smallest_sum_of_consecutive_integers_gt_420_l693_69344


namespace fill_time_without_leak_l693_69368

theorem fill_time_without_leak (F L : ℝ)
  (h1 : (F - L) * 12 = 1)
  (h2 : L * 24 = 1) :
  1 / F = 8 := 
sorry

end fill_time_without_leak_l693_69368


namespace find_prime_numbers_of_form_p_p_plus_1_l693_69348

def has_at_most_19_digits (n : ℕ) : Prop := n < 10^19

theorem find_prime_numbers_of_form_p_p_plus_1 :
  {n : ℕ | ∃ p : ℕ, n = p^p + 1 ∧ has_at_most_19_digits n ∧ Nat.Prime n} = {2, 5, 257} :=
by
  sorry

end find_prime_numbers_of_form_p_p_plus_1_l693_69348


namespace intersection_of_sets_l693_69345

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets :
  setA ∩ setB = { z : ℝ | z ∈ [-1, 1] } :=
sorry

end intersection_of_sets_l693_69345


namespace initial_students_per_group_l693_69358

-- Define the conditions
variables {x : ℕ} (h : 3 * x - 2 = 22)

-- Lean 4 statement of the proof problem
theorem initial_students_per_group (x : ℕ) (h : 3 * x - 2 = 22) : x = 8 :=
sorry

end initial_students_per_group_l693_69358


namespace unique_x_condition_l693_69301

theorem unique_x_condition (x : ℝ) : 
  (1 ≤ x ∧ x < 2) ∧ (∀ n : ℕ, 0 < n → (⌊2^n * x⌋ % 4 = 1 ∨ ⌊2^n * x⌋ % 4 = 2)) ↔ x = 4/3 := 
by 
  sorry

end unique_x_condition_l693_69301


namespace two_positive_numbers_inequality_three_positive_numbers_am_gm_l693_69374

theorem two_positive_numbers_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  x^3 + y^3 ≥ x^2 * y + x * y^2 ∧ (x = y ↔ x^3 + y^3 = x^2 * y + x * y^2) := by
sorry

theorem three_positive_numbers_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≥ (a * b * c)^(1/3) ∧ (a = b ∧ b = c ↔ (a + b + c) / 3 = (a * b * c)^(1/3)) := by
sorry

end two_positive_numbers_inequality_three_positive_numbers_am_gm_l693_69374


namespace greatest_possible_sum_example_sum_case_l693_69355

/-- For integers x and y such that x^2 + y^2 = 50, the greatest possible value of x + y is 10. -/
theorem greatest_possible_sum (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 10 :=
sorry

-- Auxiliary theorem to state that 10 can be achieved
theorem example_sum_case : ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ x + y = 10 :=
sorry

end greatest_possible_sum_example_sum_case_l693_69355


namespace repeated_three_digit_divisible_l693_69377

theorem repeated_three_digit_divisible (μ : ℕ) (h : 100 ≤ μ ∧ μ < 1000) :
  ∃ k : ℕ, (1000 * μ + μ) = k * 7 * 11 * 13 := by
sorry

end repeated_three_digit_divisible_l693_69377


namespace automotive_test_l693_69324

theorem automotive_test (D T1 T2 T3 T_total : ℕ) (H1 : 3 * D = 180) 
  (H2 : T1 = D / 4) (H3 : T2 = D / 5) (H4 : T3 = D / 6)
  (H5 : T_total = T1 + T2 + T3) : T_total = 37 :=
  sorry

end automotive_test_l693_69324


namespace maximize_profit_at_six_l693_69332

-- Defining the functions (conditions)
def y1 (x : ℝ) : ℝ := 17 * x^2
def y2 (x : ℝ) : ℝ := 2 * x^3 - x^2
def profit (x : ℝ) : ℝ := y1 x - y2 x

-- The condition x > 0
def x_pos (x : ℝ) : Prop := x > 0

-- Proving the maximum profit is achieved at x = 6 (question == answer)
theorem maximize_profit_at_six : ∀ x > 0, (∀ y > 0, y = profit x → x = 6) :=
by 
  intros x hx y hy
  sorry

end maximize_profit_at_six_l693_69332


namespace number_of_positive_integers_l693_69341

theorem number_of_positive_integers (n : ℕ) (hpos : 0 < n) (h : 24 - 6 * n ≥ 12) : n = 1 ∨ n = 2 :=
sorry

end number_of_positive_integers_l693_69341


namespace electricity_consumption_scientific_notation_l693_69340

def electricity_consumption (x : Float) : String := 
  let scientific_notation := "3.64 × 10^4"
  scientific_notation

theorem electricity_consumption_scientific_notation :
  electricity_consumption 36400 = "3.64 × 10^4" :=
by 
  sorry

end electricity_consumption_scientific_notation_l693_69340


namespace ratio_of_flour_to_eggs_l693_69389

theorem ratio_of_flour_to_eggs (F E : ℕ) (h1 : E = 60) (h2 : F + E = 90) : F / 30 = 1 ∧ E / 30 = 2 := by
  sorry

end ratio_of_flour_to_eggs_l693_69389


namespace evaluate_expression_at_minus3_l693_69326

theorem evaluate_expression_at_minus3:
  (∀ x, x = -3 → (3 + x * (3 + x) - 3^2 + x) / (x - 3 + x^2 - x) = -3/2) :=
by
  sorry

end evaluate_expression_at_minus3_l693_69326


namespace carlotta_performance_time_l693_69300

theorem carlotta_performance_time :
  ∀ (s p t : ℕ),  -- s for singing, p for practicing, t for tantrums
  (∀ (n : ℕ), p = 3 * n ∧ t = 5 * n) →
  s = 6 →
  (s + p + t) = 54 :=
by 
  intros s p t h1 h2
  rcases h1 1 with ⟨h3, h4⟩
  sorry

end carlotta_performance_time_l693_69300


namespace calculate_subtraction_l693_69399

theorem calculate_subtraction :
  ∀ (x : ℕ), (49 = 50 - 1) → (49^2 = 50^2 - 99)
  := by
  intros x h
  sorry

end calculate_subtraction_l693_69399


namespace find_valid_pairs_l693_69336

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m : ℕ, 2 ≤ m → m ≤ p / 2 → ¬(m ∣ p)

def valid_pair (n p : ℕ) : Prop :=
  is_prime p ∧ 0 < n ∧ n ≤ 2 * p ∧ n ^ (p - 1) ∣ (p - 1) ^ n + 1

theorem find_valid_pairs (n p : ℕ) : valid_pair n p ↔ (n = 1 ∧ is_prime p) ∨ (n, p) = (2, 2) ∨ (n, p) = (3, 3) := by
  sorry

end find_valid_pairs_l693_69336


namespace fraction_meaningful_iff_l693_69392

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l693_69392


namespace water_in_bowl_after_adding_4_cups_l693_69346

def total_capacity_bowl := 20 -- Capacity of the bowl in cups

def initially_half_full (C : ℕ) : Prop :=
C = total_capacity_bowl / 2

def after_adding_4_cups (initial : ℕ) : ℕ :=
initial + 4

def seventy_percent_full (C : ℕ) : ℕ :=
7 * C / 10

theorem water_in_bowl_after_adding_4_cups :
  ∀ (C initial after_adding) (h1 : initially_half_full initial)
  (h2 : after_adding = after_adding_4_cups initial)
  (h3 : after_adding = seventy_percent_full C),
  after_adding = 14 := 
by
  intros C initial after_adding h1 h2 h3
  -- Proof goes here
  sorry

end water_in_bowl_after_adding_4_cups_l693_69346


namespace alex_plays_with_friends_l693_69314

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l693_69314


namespace saturn_moon_approximation_l693_69385

theorem saturn_moon_approximation : (1.2 * 10^5) * 10 = 1.2 * 10^6 := 
by sorry

end saturn_moon_approximation_l693_69385


namespace line_through_points_l693_69330

theorem line_through_points 
  (A1 B1 A2 B2 : ℝ) 
  (h₁ : A1 * -7 + B1 * 9 = 1) 
  (h₂ : A2 * -7 + B2 * 9 = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), (A1, B1) ≠ (A2, B2) → y = k * x + (B1 - k * A1) → -7 * x + 9 * y = 1 :=
sorry

end line_through_points_l693_69330


namespace lisa_eggs_total_l693_69318

def children_mon_tue := 4 * 2 * 2
def husband_mon_tue := 3 * 2 
def lisa_mon_tue := 2 * 2
def total_mon_tue := children_mon_tue + husband_mon_tue + lisa_mon_tue

def children_wed := 4 * 3
def husband_wed := 4
def lisa_wed := 3
def total_wed := children_wed + husband_wed + lisa_wed

def children_thu := 4 * 1
def husband_thu := 2
def lisa_thu := 1
def total_thu := children_thu + husband_thu + lisa_thu

def children_fri := 4 * 2
def husband_fri := 3
def lisa_fri := 2
def total_fri := children_fri + husband_fri + lisa_fri

def total_week := total_mon_tue + total_wed + total_thu + total_fri

def weeks_per_year := 52
def yearly_eggs := total_week * weeks_per_year

def children_holidays := 4 * 2 * 8
def husband_holidays := 2 * 8
def lisa_holidays := 2 * 8
def total_holidays := children_holidays + husband_holidays + lisa_holidays

def total_annual_eggs := yearly_eggs + total_holidays

theorem lisa_eggs_total : total_annual_eggs = 3476 := by
  sorry

end lisa_eggs_total_l693_69318


namespace total_length_of_table_free_sides_l693_69365

theorem total_length_of_table_free_sides
  (L W : ℕ) -- Define lengths of the sides
  (h1 : L = 2 * W) -- The side opposite the wall is twice the length of each of the other two free sides
  (h2 : L * W = 128) -- The area of the rectangular table is 128 square feet
  : L + 2 * W = 32 -- Prove the total length of the table's free sides is 32 feet
  :=
sorry -- proof omitted

end total_length_of_table_free_sides_l693_69365


namespace find_special_integers_l693_69331

theorem find_special_integers 
  : ∃ n : ℕ, 100 ≤ n ∧ n ≤ 1997 ∧ (2^n + 2) % n = 0 ∧ (n = 66 ∨ n = 198 ∨ n = 398 ∨ n = 798) :=
by
  sorry

end find_special_integers_l693_69331


namespace papaya_cost_is_one_l693_69373

theorem papaya_cost_is_one (lemons_cost : ℕ) (mangos_cost : ℕ) (total_fruits : ℕ) (total_cost_paid : ℕ) :
    (lemons_cost = 2) → (mangos_cost = 4) → (total_fruits = 12) → (total_cost_paid = 21) → 
    let discounts := total_fruits / 4
    let lemons_bought := 6
    let mangos_bought := 2
    let papayas_bought := 4
    let total_discount := discounts
    let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
    total_cost_before_discount - total_discount = total_cost_paid → 
    P = 1 := 
by 
  intros h1 h2 h3 h4 
  let discounts := total_fruits / 4
  let lemons_bought := 6
  let mangos_bought := 2
  let papayas_bought := 4
  let total_discount := discounts
  let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
  sorry

end papaya_cost_is_one_l693_69373


namespace kittens_count_l693_69349

def cats_taken_in : ℕ := 12
def cats_initial : ℕ := cats_taken_in / 2
def cats_post_adoption : ℕ := cats_taken_in + cats_initial - 3
def cats_now : ℕ := 19

theorem kittens_count :
  ∃ k : ℕ, cats_post_adoption + k - 1 = cats_now :=
by
  use 5
  sorry

end kittens_count_l693_69349


namespace zhang_shan_sales_prediction_l693_69333

theorem zhang_shan_sales_prediction (x : ℝ) (y : ℝ) (h : x = 34) (reg_eq : y = 2 * x + 60) : y = 128 :=
by
  sorry

end zhang_shan_sales_prediction_l693_69333


namespace total_animals_made_it_to_shore_l693_69325

def boat (total_sheep total_cows total_dogs sheep_drowned cows_drowned dogs_saved : Nat) : Prop :=
  cows_drowned = sheep_drowned * 2 ∧
  dogs_saved = total_dogs ∧
  total_sheep + total_cows + total_dogs - sheep_drowned - cows_drowned = 35

theorem total_animals_made_it_to_shore :
  boat 20 10 14 3 6 14 :=
by
  sorry

end total_animals_made_it_to_shore_l693_69325


namespace range_of_m_l693_69383

open Real

def vector_a (m : ℝ) : ℝ × ℝ := (m, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (-2 * m, m)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def not_parallel (m : ℝ) : Prop :=
  m^2 + 2 * m ≠ 0

theorem range_of_m (m : ℝ) (h1 : dot_product (vector_a m) (vector_b m) < 0) (h2 : not_parallel m) :
  m < 0 ∨ (m > (1 / 2) ∧ m ≠ -2) :=
sorry

end range_of_m_l693_69383


namespace find_a_l693_69327

-- Define the sets A and B and their union
variables (a : ℕ)
def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a^2}
def C : Set ℕ := {0, 1, 2, 3, 9}

-- Define the condition and prove that it implies a = 3
theorem find_a (h : A a ∪ B a = C) : a = 3 := 
by
  sorry

end find_a_l693_69327


namespace shortest_chord_through_point_on_circle_l693_69310

theorem shortest_chord_through_point_on_circle :
  ∀ (M : ℝ × ℝ) (x y : ℝ),
    M = (3, 0) →
    x^2 + y^2 - 8 * x - 2 * y + 10 = 0 →
    ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ a = 1 ∧ b = 1 ∧ c = -3 :=
by
  sorry

end shortest_chord_through_point_on_circle_l693_69310


namespace park_area_l693_69319

-- Definitions for the conditions
def length (breadth : ℕ) : ℕ := 4 * breadth
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the proof problem
theorem park_area (breadth : ℕ) (h1 : perimeter (length breadth) breadth = 1600) : 
  let len := length breadth
  len * breadth = 102400 := 
by 
  sorry

end park_area_l693_69319


namespace conic_section_is_parabola_l693_69396

theorem conic_section_is_parabola (x y : ℝ) : y^4 - 16 * x^2 = 2 * y^2 - 64 → ((y^2 - 1)^2 = 16 * x^2 - 63) ∧ (∃ k : ℝ, y^2 = 4 * k * x + 1) :=
sorry

end conic_section_is_parabola_l693_69396


namespace connie_initial_marbles_l693_69366

theorem connie_initial_marbles (marbles_given : ℝ) (marbles_left : ℝ) : 
  marbles_given = 183 → marbles_left = 593 → marbles_given + marbles_left = 776 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end connie_initial_marbles_l693_69366


namespace area_of_shape_l693_69316

theorem area_of_shape (x y : ℝ) (α : ℝ) (P : ℝ × ℝ) :
  (x - 2 * Real.cos α)^2 + (y - 2 * Real.sin α)^2 = 16 →
  ∃ A : ℝ, A = 32 * Real.pi :=
by
  sorry

end area_of_shape_l693_69316


namespace fifth_term_power_of_five_sequence_l693_69306

theorem fifth_term_power_of_five_sequence : 5^0 + 5^1 + 5^2 + 5^3 + 5^4 = 781 := 
by
sorry

end fifth_term_power_of_five_sequence_l693_69306


namespace heather_ends_up_with_45_blocks_l693_69376

-- Conditions
def initialBlocks (Heather : Type) : ℕ := 86
def sharedBlocks (Heather : Type) : ℕ := 41

-- The theorem to prove
theorem heather_ends_up_with_45_blocks (Heather : Type) :
  (initialBlocks Heather) - (sharedBlocks Heather) = 45 :=
by
  sorry

end heather_ends_up_with_45_blocks_l693_69376


namespace range_of_x_range_of_a_l693_69395

-- Part (1): 
theorem range_of_x (x : ℝ) : 
  (a = 1) → (x^2 - 6 * a * x + 8 * a^2 < 0) → (x^2 - 4 * x + 3 ≤ 0) → (2 < x ∧ x ≤ 3) := sorry

-- Part (2):
theorem range_of_a (a : ℝ) : 
  (a ≠ 0) → (∀ x, (x^2 - 4 * x + 3 ≤ 0) → (x^2 - 6 * a * x + 8 * a^2 < 0)) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 4) := sorry

end range_of_x_range_of_a_l693_69395


namespace sphere_radius_five_times_surface_area_l693_69304

theorem sphere_radius_five_times_surface_area (R : ℝ) (h₁ : (4 * π * R^3 / 3) = 5 * (4 * π * R^2)) : R = 15 :=
sorry

end sphere_radius_five_times_surface_area_l693_69304


namespace cosine_of_arcsine_l693_69307

theorem cosine_of_arcsine (h : -1 ≤ (8 : ℝ) / 17 ∧ (8 : ℝ) / 17 ≤ 1) : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
sorry

end cosine_of_arcsine_l693_69307


namespace square_floor_tile_count_l693_69350

/-
A square floor is tiled with congruent square tiles.
The tiles on the two diagonals of the floor are black.
If there are 101 black tiles, then the total number of tiles is 2601.
-/
theorem square_floor_tile_count  
  (s : ℕ) 
  (hs_odd : s % 2 = 1)  -- s is odd
  (h_black_tile_count : 2 * s - 1 = 101) 
  : s^2 = 2601 := 
by 
  sorry

end square_floor_tile_count_l693_69350


namespace proof_problem_l693_69321

noncomputable def a : ℚ := 2 / 3
noncomputable def b : ℚ := - 3 / 2
noncomputable def n : ℕ := 2023

theorem proof_problem :
  (a ^ n) * (b ^ n) = -1 :=
by
  sorry

end proof_problem_l693_69321


namespace rectangular_solid_edges_sum_l693_69329

theorem rectangular_solid_edges_sum 
  (a b c : ℝ) 
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b^2 = a * c) : 
  4 * (a + b + c) = 32 := 
  sorry

end rectangular_solid_edges_sum_l693_69329


namespace union_A_B_l693_69370

open Set

def A := {x : ℝ | x * (x - 2) < 3}
def B := {x : ℝ | 5 / (x + 1) ≥ 1}
def U := {x : ℝ | -1 < x ∧ x ≤ 4}

theorem union_A_B : A ∪ B = U := 
sorry

end union_A_B_l693_69370


namespace items_per_baggie_l693_69311

def num_pretzels : ℕ := 64
def num_suckers : ℕ := 32
def num_kids : ℕ := 16
def num_goldfish : ℕ := 4 * num_pretzels
def total_items : ℕ := num_pretzels + num_goldfish + num_suckers

theorem items_per_baggie : total_items / num_kids = 22 :=
by
  -- Calculation proof
  sorry

end items_per_baggie_l693_69311


namespace alpha_plus_beta_eq_l693_69361

variable {α β : ℝ}
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin (α - β) = 5 / 6)
variable (h4 : Real.tan α / Real.tan β = -1 / 4)

theorem alpha_plus_beta_eq : α + β = 7 * Real.pi / 6 := by
  sorry

end alpha_plus_beta_eq_l693_69361


namespace all_numbers_equal_l693_69351

theorem all_numbers_equal 
  (x : Fin 2007 → ℝ)
  (h : ∀ (I : Finset (Fin 2007)), I.card = 7 → ∃ (J : Finset (Fin 2007)), J.card = 11 ∧ 
  (1 / 7 : ℝ) * I.sum x = (1 / 11 : ℝ) * J.sum x) :
  ∃ c : ℝ, ∀ i : Fin 2007, x i = c :=
by sorry

end all_numbers_equal_l693_69351


namespace people_ratio_l693_69356

theorem people_ratio (pounds_coal : ℕ) (days1 : ℕ) (people1 : ℕ) (pounds_goal : ℕ) (days2 : ℕ) :
  pounds_coal = 10000 → days1 = 10 → people1 = 10 → pounds_goal = 40000 → days2 = 80 →
  (people1 * pounds_goal * days1) / (pounds_coal * days2) = 1 / 2 :=
by
  sorry

end people_ratio_l693_69356


namespace railway_tunnel_construction_days_l693_69357

theorem railway_tunnel_construction_days
  (a b t : ℝ)
  (h1 : a = 1/3)
  (h2 : b = 20/100)
  (h3 : t = 4/5 ∨ t = 0.8)
  (total_days : ℝ)
  (h_total_days : total_days = 185)
  : total_days = 180 := 
sorry

end railway_tunnel_construction_days_l693_69357


namespace speed_of_current_l693_69386

theorem speed_of_current (v_w v_c : ℝ) (h_downstream : 125 = (v_w + v_c) * 10)
                         (h_upstream : 60 = (v_w - v_c) * 10) :
  v_c = 3.25 :=
by {
  sorry
}

end speed_of_current_l693_69386


namespace number_is_450064_l693_69382

theorem number_is_450064 : (45 * 10000 + 64) = 450064 :=
by
  sorry

end number_is_450064_l693_69382


namespace find_k_l693_69338

-- Conditions
def t : ℕ := 6
def is_nonzero_digit (n : ℕ) : Prop := n > 0 ∧ n < 10

-- Given these conditions, we need to prove that k = 9
theorem find_k (k t : ℕ) (h1 : t = 6) (h2 : is_nonzero_digit k) (h3 : is_nonzero_digit t) :
    (8 * 10^2 + k * 10 + 8) + (k * 10^2 + 8 * 10 + 8) - 16 * t * 10^0 * 6 = (9 * 10 + 8) + (9 * 10^2 + 8 * 10 + 8) - (16 * 6 * 10^1 + 6) → k = 9 := 
sorry

end find_k_l693_69338


namespace part_one_part_two_l693_69369

noncomputable def f (x a : ℝ) : ℝ :=
  Real.log (1 + x) + a * Real.cos x

noncomputable def g (x : ℝ) : ℝ :=
  f x 2 - 1 / (1 + x)

theorem part_one (a : ℝ) : 
  (∀ x, f x a = Real.log (1 + x) + a * Real.cos x) ∧ 
  f 0 a = 2 ∧ 
  (∀ x, x + f (0:ℝ) a = x + 2) → 
  a = 2 := 
sorry

theorem part_two : 
  (∀ x, g x = Real.log (1 + x) + 2 * Real.cos x - 1 / (1 + x)) →
  (∃ y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) ∧ 
  (∀ x, -1 < x ∧ x < (Real.pi / 2) → g x ≠ 0) →
  (∃! y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) :=
sorry

end part_one_part_two_l693_69369


namespace arithmetic_progression_correct_l693_69381

noncomputable def nth_term_arithmetic_progression (n : ℕ) : ℝ :=
  4.2 * n + 9.3

def recursive_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  a 1 = 13.5 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n + 4.2

theorem arithmetic_progression_correct (n : ℕ) :
  (nth_term_arithmetic_progression n = 4.2 * n + 9.3) ∧
  ∀ (a : ℕ → ℝ), recursive_arithmetic_progression a → a n = 4.2 * n + 9.3 :=
by
  sorry

end arithmetic_progression_correct_l693_69381


namespace reflect_parabola_x_axis_l693_69303

theorem reflect_parabola_x_axis (x : ℝ) (a b c : ℝ) :
  (∀ y : ℝ, y = x^2 + x - 2 → -y = x^2 + x - 2) →
  (∀ y : ℝ, -y = x^2 + x - 2 → y = -x^2 - x + 2) :=
by
  intros h₁ h₂
  intro y
  sorry

end reflect_parabola_x_axis_l693_69303


namespace master_li_speeding_l693_69309

theorem master_li_speeding (distance : ℝ) (time : ℝ) (speed_limit : ℝ) (average_speed : ℝ)
  (h_distance : distance = 165)
  (h_time : time = 2)
  (h_speed_limit : speed_limit = 80)
  (h_average_speed : average_speed = distance / time)
  (h_speeding : average_speed > speed_limit) :
  True :=
sorry

end master_li_speeding_l693_69309


namespace trash_cans_street_count_l693_69322

theorem trash_cans_street_count (S B : ℕ) (h1 : B = 2 * S) (h2 : S + B = 42) : S = 14 :=
by
  sorry

end trash_cans_street_count_l693_69322


namespace eccentricity_of_ellipse_l693_69362

open Real

noncomputable def ellipse_eccentricity : ℝ :=
  let a : ℝ := 4
  let b : ℝ := 2 * sqrt 3
  let c : ℝ := sqrt (a^2 - b^2)
  c / a

theorem eccentricity_of_ellipse (a b : ℝ) (ha : a = 4) (hb : b = 2 * sqrt 3) (h_eq : ∀ A B : ℝ, |A - B| = b^2 / 2 → |A - 2 * sqrt 3| + |B - 2 * sqrt 3| ≤ 10) :
  ellipse_eccentricity = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l693_69362


namespace ranking_sequences_l693_69313

theorem ranking_sequences
    (A D B E C : Type)
    (h_no_ties : ∀ (X Y : Type), X ≠ Y)
    (h_games : (W1 = A ∨ W1 = D) ∧ (W2 = B ∨ W2 = E) ∧ (W3 = W1 ∨ W3 = C)) :
  ∃! (n : ℕ), n = 48 := 
sorry

end ranking_sequences_l693_69313


namespace determine_f_3_2016_l693_69320

noncomputable def f : ℕ → ℕ → ℕ
| 0, y       => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem determine_f_3_2016 : f 3 2016 = 2 ^ 2019 - 3 := by
  sorry

end determine_f_3_2016_l693_69320


namespace smallest_fraction_l693_69335

theorem smallest_fraction 
  (x y z t : ℝ) 
  (h1 : 1 < x) 
  (h2 : x < y) 
  (h3 : y < z) 
  (h4 : z < t) : 
  (min (min (min (min ((x + y) / (z + t)) ((x + t) / (y + z))) ((y + z) / (x + t))) ((y + t) / (x + z))) ((z + t) / (x + y))) = (x + y) / (z + t) :=
by {
    sorry
}

end smallest_fraction_l693_69335


namespace odd_function_values_l693_69343

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l693_69343
