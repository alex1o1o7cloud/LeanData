import Mathlib

namespace distance_origin_to_point_on_parabola_l808_80853

noncomputable def origin : ℝ × ℝ := (0, 0)

noncomputable def parabola_focus (x y : ℝ) : Prop :=
  x^2 = 4 * y ∧ y = 1

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  x^2 = 4 * y

theorem distance_origin_to_point_on_parabola (x y : ℝ) (hx : x^2 = 4 * y)
 (hf : (0, 1) = (0, 1)) (hPF : (x - 0)^2 + (y - 1)^2 = 25) : (x^2 + y^2 = 32) :=
by
  sorry

end distance_origin_to_point_on_parabola_l808_80853


namespace divisor_of_p_l808_80865

theorem divisor_of_p (p q r s : ℕ) (hpq : Nat.gcd p q = 40)
  (hqr : Nat.gcd q r = 45) (hrs : Nat.gcd r s = 60)
  (hspr : 100 < Nat.gcd s p ∧ Nat.gcd s p < 150)
  : 7 ∣ p :=
sorry

end divisor_of_p_l808_80865


namespace eat_cereal_in_time_l808_80832

noncomputable def time_to_eat_pounds (pounds : ℕ) (rate1 rate2 : ℚ) :=
  pounds / (rate1 + rate2)

theorem eat_cereal_in_time :
  time_to_eat_pounds 5 ((1:ℚ)/15) ((1:ℚ)/40) = 600/11 := 
by 
  sorry

end eat_cereal_in_time_l808_80832


namespace probability_at_least_one_8_l808_80825

theorem probability_at_least_one_8 (n : ℕ) (hn : n = 8) : 
  (1 - (7/8) * (7/8)) = 15 / 64 :=
by
  rw [← hn]
  sorry

end probability_at_least_one_8_l808_80825


namespace k_is_3_l808_80803

noncomputable def k_solution (k : ℝ) : Prop :=
  k > 1 ∧ (∑' n : ℕ, (n^2 + 3 * n - 2) / k^n = 2)

theorem k_is_3 : ∃ k : ℝ, k_solution k ∧ k = 3 :=
by
  sorry

end k_is_3_l808_80803


namespace rachel_homework_difference_l808_80869

def total_difference (r m h s : ℕ) : ℕ :=
  (r - m) + (s - h)

theorem rachel_homework_difference :
    ∀ (r m h s : ℕ), r = 7 → m = 5 → h = 3 → s = 6 → total_difference r m h s = 5 :=
by
  intros r m h s hr hm hh hs
  rw [hr, hm, hh, hs]
  rfl

end rachel_homework_difference_l808_80869


namespace greatest_is_B_l808_80800

def A : ℕ := 95 - 35
def B : ℕ := A + 12
def C : ℕ := B - 19

theorem greatest_is_B : B = 72 ∧ (B > A ∧ B > C) :=
by {
  -- Proof steps would be written here to prove the theorem.
  sorry
}

end greatest_is_B_l808_80800


namespace distance_travel_l808_80808

-- Definition of the parameters and the proof problem
variable (W_t : ℕ)
variable (R_c : ℕ)
variable (remaining_coal : ℕ)

-- Conditions
def rate_of_coal_consumption : Prop := R_c = 4 * W_t / 1000
def remaining_coal_amount : Prop := remaining_coal = 160

-- Theorem statement
theorem distance_travel (W_t : ℕ) (R_c : ℕ) (remaining_coal : ℕ) 
  (h1 : rate_of_coal_consumption W_t R_c) 
  (h2 : remaining_coal_amount remaining_coal) : 
  (remaining_coal * 1000 / 4 / W_t) = 40000 / W_t := 
by
  sorry

end distance_travel_l808_80808


namespace sum_of_coefficients_l808_80873

theorem sum_of_coefficients (a : ℤ) (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (a + x) * (1 + x) ^ 4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_3 + a_5 = 32 →
  a = 3 :=
by sorry

end sum_of_coefficients_l808_80873


namespace combined_cost_of_one_item_l808_80878

-- Definitions representing the given conditions
def initial_amount : ℝ := 50
def final_amount : ℝ := 14
def mangoes_purchased : ℕ := 6
def apple_juice_purchased : ℕ := 6

-- Hypothesis: The cost of mangoes and apple juice are the same
variables (M A : ℝ)

-- Total amount spent
def amount_spent : ℝ := initial_amount - final_amount

-- Combined number of items
def total_items : ℕ := mangoes_purchased + apple_juice_purchased

-- Lean statement to prove the combined cost of one mango and one carton of apple juice is $3
theorem combined_cost_of_one_item (h : mangoes_purchased * M + apple_juice_purchased * A = amount_spent) :
    (amount_spent / total_items) = (3 : ℝ) :=
by
  sorry

end combined_cost_of_one_item_l808_80878


namespace product_xyz_l808_80838

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 5) : 
  x * y * z = 1 / 9 := 
by
  sorry

end product_xyz_l808_80838


namespace probability_one_male_correct_probability_atleast_one_female_correct_l808_80851

def total_students := 5
def female_students := 2
def male_students := 3
def number_of_selections := 2

noncomputable def probability_only_one_male : ℚ :=
  (6 : ℚ) / 10

noncomputable def probability_atleast_one_female : ℚ :=
  (7 : ℚ) / 10

theorem probability_one_male_correct :
  (6 / 10 : ℚ) = 3 / 5 :=
by
  sorry

theorem probability_atleast_one_female_correct :
  (7 / 10 : ℚ) = 7 / 10 :=
by
  sorry

end probability_one_male_correct_probability_atleast_one_female_correct_l808_80851


namespace percent_profit_l808_80862

theorem percent_profit (C S : ℝ) (h : 58 * C = 50 * S) : 
  (S - C) / C * 100 = 16 :=
by
  sorry

end percent_profit_l808_80862


namespace number_called_2009th_position_l808_80822

theorem number_called_2009th_position :
  let sequence := [1, 2, 3, 4, 3, 2]
  ∃ n, n = 2009 → sequence[(2009 % 6) - 1] = 3 := 
by
  -- let sequence := [1, 2, 3, 4, 3, 2]
  -- 2009 % 6 = 5
  -- sequence[4] = 3
  sorry

end number_called_2009th_position_l808_80822


namespace quadratic_no_real_roots_l808_80839

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_no_real_roots
  (a b c: ℝ)
  (h1: ((b - 1)^2 - 4 * a * (c + 1) = 0))
  (h2: ((b + 2)^2 - 4 * a * (c - 2) = 0)) :
  ∀ x : ℝ, f a b c x ≠ 0 := 
sorry

end quadratic_no_real_roots_l808_80839


namespace sum_of_squares_xy_l808_80817

theorem sum_of_squares_xy (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end sum_of_squares_xy_l808_80817


namespace first_even_number_l808_80831

theorem first_even_number (x : ℤ) (h : x + (x + 2) + (x + 4) = 1194) : x = 396 :=
by
  -- the proof is skipped as per instructions
  sorry

end first_even_number_l808_80831


namespace inequality_proof_l808_80826

variable (b c : ℝ)
variable (hb : b > 0) (hc : c > 0)

theorem inequality_proof :
  (b - c) ^ 2011 * (b + c) ^ 2011 * (c - b) ^ 2011 ≥ (b ^ 2011 - c ^ 2011) * (b ^ 2011 + c ^ 2011) * (c ^ 2011 - b ^ 2011) :=
  sorry

end inequality_proof_l808_80826


namespace derivative_of_odd_function_is_even_l808_80857

theorem derivative_of_odd_function_is_even (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) :
  ∀ x, (deriv f) (-x) = (deriv f) x :=
by
  sorry

end derivative_of_odd_function_is_even_l808_80857


namespace quadratic_equals_binomial_square_l808_80819

theorem quadratic_equals_binomial_square (d : ℝ) : 
  (∃ b : ℝ, (x^2 + 60 * x + d) = (x + b)^2) → d = 900 :=
by
  sorry

end quadratic_equals_binomial_square_l808_80819


namespace isosceles_triangle_ratio_HD_HA_l808_80876

theorem isosceles_triangle_ratio_HD_HA (A B C D H : ℝ) :
  let AB := 13;
  let AC := 13;
  let BC := 10;
  let s := (AB + AC + BC) / 2;
  let area := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC));
  let h := (2 * area) / BC;
  let AD := h;
  let HA := h;
  let HD := 0;
  HD / HA = 0 := sorry

end isosceles_triangle_ratio_HD_HA_l808_80876


namespace radius_of_larger_circle_l808_80870

theorem radius_of_larger_circle
  (r r_s : ℝ)
  (h1 : r_s = 2)
  (h2 : π * r^2 = 4 * π * r_s^2) :
  r = 4 :=
by
  sorry

end radius_of_larger_circle_l808_80870


namespace correlation_1_and_3_l808_80860

-- Define the conditions as types
def relationship1 : Type := ∀ (age : ℕ) (fat_content : ℝ), Prop
def relationship2 : Type := ∀ (curve_point : ℝ × ℝ), Prop
def relationship3 : Type := ∀ (production : ℝ) (climate : ℝ), Prop
def relationship4 : Type := ∀ (student : ℕ) (student_ID : ℕ), Prop

-- Define what it means for two relationships to have a correlation
def has_correlation (rel1 rel2 : Type) : Prop := 
  -- Some formal definition of correlation suitable for the context
  sorry

-- Theorem stating that relationships (1) and (3) have a correlation
theorem correlation_1_and_3 :
  has_correlation relationship1 relationship3 :=
sorry

end correlation_1_and_3_l808_80860


namespace compute_expression_l808_80836

theorem compute_expression :
  (3 + 3 / 8) ^ (2 / 3) - (5 + 4 / 9) ^ (1 / 2) + 0.008 ^ (2 / 3) / 0.02 ^ (1 / 2) * 0.32 ^ (1 / 2) / 0.0625 ^ (1 / 4) = 43 / 150 := 
sorry

end compute_expression_l808_80836


namespace females_in_town_l808_80804

theorem females_in_town (population : ℕ) (ratio : ℕ × ℕ) (H : population = 480) (H_ratio : ratio = (3, 5)) : 
  let m := ratio.1
  let f := ratio.2
  f * (population / (m + f)) = 300 := by
  sorry

end females_in_town_l808_80804


namespace area_BCD_l808_80891

open Real EuclideanGeometry

noncomputable def point := (ℝ × ℝ)
noncomputable def A : point := (0, 0)
noncomputable def B : point := (10, 24)
noncomputable def C : point := (30, 0)
noncomputable def D : point := (40, 0)

def area_triangle (p1 p2 p3 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * |x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)|

theorem area_BCD : area_triangle B C D = 12 := sorry

end area_BCD_l808_80891


namespace cone_lateral_surface_area_l808_80805

-- Definitions and conditions
def radius (r : ℝ) := r = 3
def slant_height (l : ℝ) := l = 5
def lateral_surface_area (A : ℝ) (C : ℝ) (l : ℝ) := A = 0.5 * C * l
def circumference (C : ℝ) (r : ℝ) := C = 2 * Real.pi * r

-- Proof (statement only)
theorem cone_lateral_surface_area :
  ∀ (r l C A : ℝ), 
    radius r → 
    slant_height l → 
    circumference C r → 
    lateral_surface_area A C l → 
    A = 15 * Real.pi := 
by intros; sorry

end cone_lateral_surface_area_l808_80805


namespace a_b_c_relationship_l808_80820

noncomputable def a (f : ℝ → ℝ) : ℝ := 25 * f (0.2^2)
noncomputable def b (f : ℝ → ℝ) : ℝ := f 1
noncomputable def c (f : ℝ → ℝ) : ℝ := - (Real.log 3 / Real.log 5) * f (Real.log 5 / Real.log 3)

axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom decreasing_g (f : ℝ → ℝ) : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → 0 < x2 → (f x1 / x1) > (f x2 / x2)

theorem a_b_c_relationship (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_decreasing : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → 0 < x2 → (f x1 / x1) > (f x2 / x2)) :
  a f > b f ∧ b f > c f :=
sorry

end a_b_c_relationship_l808_80820


namespace lowest_sale_price_is_30_percent_l808_80824

-- Definitions and conditions
def list_price : ℝ := 80
def max_initial_discount : ℝ := 0.50
def additional_sale_discount : ℝ := 0.20

-- Calculations
def initial_discount_amount : ℝ := list_price * max_initial_discount
def initial_discounted_price : ℝ := list_price - initial_discount_amount
def additional_discount_amount : ℝ := list_price * additional_sale_discount
def lowest_sale_price : ℝ := initial_discounted_price - additional_discount_amount

-- Proof statement (with correct answer)
theorem lowest_sale_price_is_30_percent :
  lowest_sale_price = 0.30 * list_price := 
by
  sorry

end lowest_sale_price_is_30_percent_l808_80824


namespace Alan_has_eight_pine_trees_l808_80863

noncomputable def number_of_pine_trees (total_pine_cones_per_tree : ℕ) (percentage_on_roof : ℚ) 
                                       (weight_per_pine_cone : ℚ) (total_weight_on_roof : ℚ) : ℚ :=
  total_weight_on_roof / (total_pine_cones_per_tree * percentage_on_roof * weight_per_pine_cone)

theorem Alan_has_eight_pine_trees :
  number_of_pine_trees 200 (30 / 100) 4 1920 = 8 :=
by
  sorry

end Alan_has_eight_pine_trees_l808_80863


namespace number_of_yellow_marbles_l808_80854

theorem number_of_yellow_marbles (Y : ℕ) (h : Y / (7 + 11 + Y) = 1 / 4) : Y = 6 :=
by
  -- Proof to be filled in
  sorry

end number_of_yellow_marbles_l808_80854


namespace volume_invariant_l808_80801

noncomputable def volume_of_common_region (a b c : ℝ) : ℝ := (5/6) * a * b * c

theorem volume_invariant (a b c : ℝ) (P : ℝ × ℝ × ℝ) (hP : ∀ (x y z : ℝ), 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b ∧ 0 ≤ z ∧ z ≤ c) :
  volume_of_common_region a b c = (5/6) * a * b * c :=
by sorry

end volume_invariant_l808_80801


namespace a_b_sum_of_powers_l808_80861

variable (a b : ℝ)

-- Conditions
def condition1 := a + b = 1
def condition2 := a^2 + b^2 = 3
def condition3 := a^3 + b^3 = 4
def condition4 := a^4 + b^4 = 7
def condition5 := a^5 + b^5 = 11

-- Theorem statement
theorem a_b_sum_of_powers (h1 : condition1 a b) (h2 : condition2 a b) (h3 : condition3 a b) 
  (h4 : condition4 a b) (h5 : condition5 a b) : a^10 + b^10 = 123 :=
sorry

end a_b_sum_of_powers_l808_80861


namespace max_S_2017_l808_80812

noncomputable def max_S (a b c : ℕ) : ℕ := a + b + c

theorem max_S_2017 :
  ∀ (a b c : ℕ),
  a + b = 1014 →
  c - b = 497 →
  a > b →
  max_S a b c = 2017 :=
by
  intros a b c h1 h2 h3
  sorry

end max_S_2017_l808_80812


namespace additional_men_joined_l808_80850

theorem additional_men_joined
    (M : ℕ) (X : ℕ)
    (h1 : M = 20)
    (h2 : M * 50 = (M + X) * 25) :
    X = 20 := by
  sorry

end additional_men_joined_l808_80850


namespace pay_per_task_l808_80895

def tasks_per_day : ℕ := 100
def days_per_week : ℕ := 6
def weekly_pay : ℕ := 720

theorem pay_per_task :
  (weekly_pay : ℚ) / (tasks_per_day * days_per_week) = 1.20 := 
sorry

end pay_per_task_l808_80895


namespace find_Roe_speed_l808_80834

-- Definitions from the conditions
def Teena_speed : ℝ := 55
def time_in_hours : ℝ := 1.5
def initial_distance_difference : ℝ := 7.5
def final_distance_difference : ℝ := 15

-- Main theorem statement
theorem find_Roe_speed (R : ℝ) (h1 : R * time_in_hours + final_distance_difference = Teena_speed * time_in_hours - initial_distance_difference) :
  R = 40 :=
  sorry

end find_Roe_speed_l808_80834


namespace isabella_houses_problem_l808_80829

theorem isabella_houses_problem 
  (yellow green red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  (green + red = 160) := 
sorry

end isabella_houses_problem_l808_80829


namespace percent_of_a_is_4b_l808_80871

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : (4 * b) / a = 20 / 9 :=
by sorry

end percent_of_a_is_4b_l808_80871


namespace find_PF_2_l808_80899

-- Define the hyperbola and points
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 3) = 1
def PF_1 := 3
def a := 2
def two_a := 2 * a

-- State the theorem
theorem find_PF_2 (PF_2 : ℝ) (cond1 : PF_1 = 3) (cond2 : abs (PF_1 - PF_2) = two_a) : PF_2 = 7 :=
sorry

end find_PF_2_l808_80899


namespace relationship_m_n_l808_80821

theorem relationship_m_n (b : ℝ) (m : ℝ) (n : ℝ) (h1 : m = 2 * b + 2022) (h2 : n = b^2 + 2023) : m ≤ n :=
by
  sorry

end relationship_m_n_l808_80821


namespace find_x_collinear_l808_80806

theorem find_x_collinear (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (x, 1)) 
  (h_collinear : ∃ k : ℝ, (2 * 2 + x) = k * x ∧ (2 * -1 + 1) = k * 1) : x = -2 :=
by
  sorry

end find_x_collinear_l808_80806


namespace people_in_first_group_l808_80816

theorem people_in_first_group (P : ℕ) (work_done_by_P : 60 = 1 / (P * (1/60))) (work_done_by_16 : 30 = 1 / (16 * (1/30))) : P = 8 :=
by
  sorry

end people_in_first_group_l808_80816


namespace days_kept_first_book_l808_80818

def cost_per_day : ℝ := 0.50
def total_days_in_may : ℝ := 31
def total_cost_paid : ℝ := 41

theorem days_kept_first_book (x : ℝ) : 0.50 * x + 2 * (0.50 * 31) = 41 → x = 20 :=
by sorry

end days_kept_first_book_l808_80818


namespace determine_xyz_l808_80875

theorem determine_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + 1/y = 5) (h5 : y + 1/z = 2) (h6 : z + 1/x = 8/3) :
  x * y * z = 8 + 3 * Real.sqrt 7 :=
by
  sorry

end determine_xyz_l808_80875


namespace solution_set_for_inequality_l808_80888

open Set Real

theorem solution_set_for_inequality : 
  { x : ℝ | (2 * x) / (x + 1) ≤ 1 } = Ioc (-1 : ℝ) 1 := 
sorry

end solution_set_for_inequality_l808_80888


namespace max_objective_value_l808_80847

theorem max_objective_value (x y : ℝ) (h1 : x - y - 2 ≥ 0) (h2 : 2 * x + y - 2 ≤ 0) (h3 : y + 4 ≥ 0) :
  ∃ (z : ℝ), z = 4 * x + 3 * y ∧ z ≤ 8 :=
sorry

end max_objective_value_l808_80847


namespace crow_eats_quarter_in_twenty_hours_l808_80855

-- Given: The crow eats 1/5 of the nuts in 4 hours
def crow_eating_rate (N : ℕ) : ℕ := N / 5 / 4

-- Prove: It will take 20 hours to eat 1/4 of the nuts
theorem crow_eats_quarter_in_twenty_hours (N : ℕ) (h : ℕ) (h_eq : h = 20) : 
  ((N / 5) / 4 : ℝ) = ((N / 4) / h : ℝ) :=
by
  sorry

end crow_eats_quarter_in_twenty_hours_l808_80855


namespace fencing_cost_proof_l808_80877

theorem fencing_cost_proof (L : ℝ) (B : ℝ) (c : ℝ) (total_cost : ℝ)
  (hL : L = 60) (hL_B : L = B + 20) (hc : c = 26.50) : 
  total_cost = 5300 :=
by
  sorry

end fencing_cost_proof_l808_80877


namespace solve_eq1_solve_eq2_l808_80883

theorem solve_eq1 : ∀ (x : ℚ), (3 / 5 - 5 / 8 * x = 2 / 5) → (x = 8 / 25) := by
  intro x
  intro h
  sorry

theorem solve_eq2 : ∀ (x : ℚ), (7 * (x - 2) = 8 * (x - 4)) → (x = 18) := by
  intro x
  intro h
  sorry

end solve_eq1_solve_eq2_l808_80883


namespace find_y_of_pentagon_l808_80835

def y_coordinate (y : ℝ) : Prop :=
  let area_ABDE := 12
  let area_BCD := 2 * (y - 3)
  let total_area := area_ABDE + area_BCD
  total_area = 35

theorem find_y_of_pentagon :
  ∃ y : ℝ, y_coordinate y ∧ y = 14.5 :=
by
  sorry

end find_y_of_pentagon_l808_80835


namespace ancient_chinese_silver_problem_l808_80859

theorem ancient_chinese_silver_problem :
  ∃ (x y : ℤ), 7 * x = y - 4 ∧ 9 * x = y + 8 :=
by
  sorry

end ancient_chinese_silver_problem_l808_80859


namespace find_average_income_of_M_and_O_l808_80828

def average_income_of_M_and_O (M N O : ℕ) : Prop :=
  M + N = 10100 ∧
  N + O = 12500 ∧
  M = 4000 ∧
  (M + O) / 2 = 5200

theorem find_average_income_of_M_and_O (M N O : ℕ):
  average_income_of_M_and_O M N O → 
  (M + O) / 2 = 5200 :=
by
  intro h
  exact h.2.2.2

end find_average_income_of_M_and_O_l808_80828


namespace impossible_relationships_l808_80884

theorem impossible_relationships (a b : ℝ) (h : (1 / a) = (1 / b)) :
  (¬ (0 < a ∧ a < b)) ∧ (¬ (b < a ∧ a < 0)) :=
by
  sorry

end impossible_relationships_l808_80884


namespace factor_polynomial_l808_80897

theorem factor_polynomial (x : ℝ) : 
  (20 * x^3 + 100 * x - 10) - (-3 * x^3 + 5 * x - 15) = 5 * (23 * x^3 + 19 * x + 1) := 
by 
  -- Proof can be filled in here
  sorry

end factor_polynomial_l808_80897


namespace find_number_l808_80890

theorem find_number (x : ℝ) (h : x / 0.05 = 900) : x = 45 :=
by sorry

end find_number_l808_80890


namespace ratio_of_female_contestants_l808_80809

theorem ratio_of_female_contestants (T M F : ℕ) (hT : T = 18) (hM : M = 12) (hF : F = T - M) :
  F / T = 1 / 3 :=
by
  sorry

end ratio_of_female_contestants_l808_80809


namespace sum_of_last_two_digits_l808_80852

theorem sum_of_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) : (a^15 + b^15) % 100 = 0 := by
  sorry

end sum_of_last_two_digits_l808_80852


namespace saleswoman_commission_l808_80856

theorem saleswoman_commission (S : ℝ)
  (h1 : (S > 500) )
  (h2 : (0.20 * 500 + 0.50 * (S - 500)) = 0.3125 * S) : 
  S = 800 :=
sorry

end saleswoman_commission_l808_80856


namespace contractor_absent_days_l808_80898

-- Definition of problem conditions
def total_days : ℕ := 30
def daily_wage : ℝ := 25
def daily_fine : ℝ := 7.5
def total_amount_received : ℝ := 620

-- Function to define the constraint equations
def equation1 (x y : ℕ) : Prop := x + y = total_days
def equation2 (x y : ℕ) : Prop := (daily_wage * x - daily_fine * y) = total_amount_received

-- The proof problem translation as Lean 4 statement
theorem contractor_absent_days (x y : ℕ) (h1 : equation1 x y) (h2 : equation2 x y) : y = 8 :=
by
  sorry

end contractor_absent_days_l808_80898


namespace compute_scalar_dot_product_l808_80843

open Matrix 

def vec1 : Fin 2 → ℤ
| 0 => -2
| 1 => 3

def vec2 : Fin 2 → ℤ
| 0 => 4
| 1 => -5

def dot_product (v1 v2 : Fin 2 → ℤ) : ℤ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1)

theorem compute_scalar_dot_product :
  3 * dot_product vec1 vec2 = -69 := 
by 
  sorry

end compute_scalar_dot_product_l808_80843


namespace joan_football_games_l808_80886

theorem joan_football_games (G_total G_last G_this : ℕ) (h1 : G_total = 13) (h2 : G_last = 9) (h3 : G_this = G_total - G_last) : G_this = 4 :=
by
  sorry

end joan_football_games_l808_80886


namespace fraction_zero_iff_x_neg_one_l808_80864

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h : 1 - |x| = 0) (h_non_zero : 1 - x ≠ 0) : x = -1 :=
sorry

end fraction_zero_iff_x_neg_one_l808_80864


namespace arithmetic_sequence_a5_l808_80874

variable (a : ℕ → ℝ) (h : a 1 + a 9 = 10)

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : 
  a 5 = 5 :=
by sorry

end arithmetic_sequence_a5_l808_80874


namespace evaluate_expression_l808_80880

variable (x y : ℝ)

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 - y^2 = x * y) :
  (1 / x^2) - (1 / y^2) = - (1 / (x * y)) :=
sorry

end evaluate_expression_l808_80880


namespace checkerboard_corners_sum_l808_80802

theorem checkerboard_corners_sum : 
  let N : ℕ := 9 
  let corners := [1, 9, 73, 81]
  (corners.sum = 164) := by
  sorry

end checkerboard_corners_sum_l808_80802


namespace largest_angle_in_pentagon_l808_80811

theorem largest_angle_in_pentagon {R S : ℝ} (h₁: R = S) 
  (h₂: (75 : ℝ) + 110 + R + S + (3 * R - 20) = 540) : 
  (3 * R - 20) = 217 :=
by {
  -- Given conditions are assigned and now we need to prove the theorem, the proof is omitted
  sorry
}

end largest_angle_in_pentagon_l808_80811


namespace parabola_locus_l808_80867

variables (a c k : ℝ) (a_pos : 0 < a) (c_pos : 0 < c) (k_pos : 0 < k)

theorem parabola_locus :
  ∀ t : ℝ, ∃ x y : ℝ,
    x = -kt / (2 * a) ∧ y = - k^2 * t^2 / (4 * a) + c ∧
    y = - (k^2 / (4 * a)) * x^2 + c :=
sorry

end parabola_locus_l808_80867


namespace trapezoid_midsegment_l808_80841

theorem trapezoid_midsegment (a b : ℝ)
  (AB CD E F: ℝ) -- we need to indicate that E and F are midpoints somehow
  (h1 : AB = a)
  (h2 : CD = b)
  (h3 : AB = CD) 
  (h4 : E = (AB + CD) / 2)
  (h5 : F = (CD + AB) / 2) : 
  EF = (1/2) * (a - b) := sorry

end trapezoid_midsegment_l808_80841


namespace simplify_expression_l808_80844

theorem simplify_expression (x : ℝ) : (3 * x + 2) - 2 * (2 * x - 1) = 3 * x + 2 - 4 * x + 2 := 
by sorry

end simplify_expression_l808_80844


namespace cookies_recipes_count_l808_80837

theorem cookies_recipes_count 
  (total_students : ℕ)
  (attending_percentage : ℚ)
  (cookies_per_student : ℕ)
  (cookies_per_batch : ℕ) : 
  (total_students = 150) →
  (attending_percentage = 0.60) →
  (cookies_per_student = 3) →
  (cookies_per_batch = 18) →
  (total_students * attending_percentage * cookies_per_student / cookies_per_batch = 15) :=
by
  intros h1 h2 h3 h4
  sorry

end cookies_recipes_count_l808_80837


namespace ratio_p_q_l808_80882

section ProbabilityProof

-- Definitions and constants as per conditions
def N := Nat.factorial 15

def num_ways_A : ℕ := 4 * (Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))
def num_ways_B : ℕ := 4 * 3

def p : ℚ := num_ways_A / N
def q : ℚ := num_ways_B / N

-- Theorem: Prove that the ratio p/q is 560
theorem ratio_p_q : p / q = 560 := by
  sorry

end ProbabilityProof

end ratio_p_q_l808_80882


namespace correct_option_C_l808_80833

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complements of sets A and B in U
def complA : Set ℕ := {2, 4}
def complB : Set ℕ := {3, 4}

-- Define sets A and B using the complements
def A : Set ℕ := U \ complA
def B : Set ℕ := U \ complB

-- Mathematical proof problem statement
theorem correct_option_C : 3 ∈ A ∧ 3 ∉ B := by
  sorry

end correct_option_C_l808_80833


namespace total_cost_l808_80887

-- Define the given conditions
def total_tickets : Nat := 10
def discounted_tickets : Nat := 4
def full_price : ℝ := 2.00
def discounted_price : ℝ := 1.60

-- Calculation of the total cost Martin spent
theorem total_cost : (discounted_tickets * discounted_price) + ((total_tickets - discounted_tickets) * full_price) = 18.40 := by
  sorry

end total_cost_l808_80887


namespace Energetics_factory_l808_80889

/-- In the country "Energetics," there are 150 factories, and some of them are connected by bus
routes that do not stop anywhere except at these factories. It turns out that any four factories
can be split into two pairs such that a bus runs between each pair of factories. Find the minimum
number of pairs of factories that can be connected by bus routes. -/
theorem Energetics_factory
  (factories : Finset ℕ) (routes : Finset (ℕ × ℕ))
  (h_factories : factories.card = 150)
  (h_routes : ∀ (X Y Z W : ℕ),
    {X, Y, Z, W} ⊆ factories →
    ∃ (X1 Y1 Z1 W1 : ℕ),
    (X1, Y1) ∈ routes ∧
    (Z1, W1) ∈ routes ∧
    (X1 = X ∨ X1 = Y ∨ X1 = Z ∨ X1 = W) ∧
    (Y1 = X ∨ Y1 = Y ∨ Y1 = Z ∨ Y1 = W) ∧
    (Z1 = X ∨ Z1 = Y ∨ Z1 = Z ∨ Z1 = W) ∧
    (W1 = X ∨ W1 = Y ∨ W1 = Z ∨ W1 = W)) :
  (2 * routes.card) ≥ 11025 := sorry

end Energetics_factory_l808_80889


namespace correct_formulas_l808_80846

noncomputable def S (a x : ℝ) := (a^x - a^(-x)) / 2
noncomputable def C (a x : ℝ) := (a^x + a^(-x)) / 2

variable {a x y : ℝ}

axiom h1 : a > 0
axiom h2 : a ≠ 1

theorem correct_formulas : S a (x + y) = S a x * C a y + C a x * S a y ∧ S a (x - y) = S a x * C a y - C a x * S a y :=
by 
  sorry

end correct_formulas_l808_80846


namespace max_initial_value_seq_l808_80810

theorem max_initial_value_seq :
  ∀ (x : Fin 1996 → ℝ),
    (∀ i : Fin 1996, 1 ≤ x i) →
    (x 0 = x 1995) →
    (∀ i : Fin 1995, x i + 2 / x i = 2 * x (i + 1) + 1 / x (i + 1)) →
    x 0 ≤ 2 ^ 997 :=
sorry

end max_initial_value_seq_l808_80810


namespace smallest_3a_plus_1_l808_80830

theorem smallest_3a_plus_1 (a : ℝ) (h : 8 * a ^ 2 + 6 * a + 2 = 4) : 
  ∃ a, (8 * a ^ 2 + 6 * a + 2 = 4) ∧ min (3 * (-1) + 1) (3 * (1 / 4) + 1) = -2 :=
by {
  sorry
}

end smallest_3a_plus_1_l808_80830


namespace test_methods_first_last_test_methods_within_six_l808_80814

open Classical

def perms (n k : ℕ) : ℕ := sorry -- placeholder for permutation function

theorem test_methods_first_last
  (prod_total : ℕ) (defective : ℕ) (first_test : ℕ) (last_test : ℕ) 
  (A4_2 : ℕ) (A5_2 : ℕ) (A6_4 : ℕ) : first_test = 2 → last_test = 8 → 
  perms 4 2 * perms 5 2 * perms 6 4 = A4_2 * A5_2 * A6_4 :=
by
  intro h_first_test h_last_test
  simp [perms]
  sorry

theorem test_methods_within_six
  (prod_total : ℕ) (defective : ℕ) 
  (A4_4 : ℕ) (A4_3_A6_1 : ℕ) (A5_3_A6_2 : ℕ) (A6_6 : ℕ)
  : perms 4 4 + 4 * perms 4 3 * perms 6 1 + 4 * perms 5 3 * perms 6 2 + perms 6 6 
  = A4_4 + 4 * A4_3_A6_1 + 4 * A5_3_A6_2 + A6_6 :=
by
  simp [perms]
  sorry

end test_methods_first_last_test_methods_within_six_l808_80814


namespace choose_8_from_16_l808_80858

theorem choose_8_from_16 :
  Nat.choose 16 8 = 12870 :=
sorry

end choose_8_from_16_l808_80858


namespace vibrations_proof_l808_80849

-- Define the conditions
def vibrations_lowest : ℕ := 1600
def increase_percentage : ℕ := 60
def use_time_minutes : ℕ := 5

-- Convert percentage to a multiplier
def percentage_to_multiplier (p : ℕ) : ℤ := (p : ℤ) / 100

-- Calculate the vibrations per second at the highest setting
def vibrations_highest := vibrations_lowest + (vibrations_lowest * percentage_to_multiplier increase_percentage).toNat

-- Convert time from minutes to seconds
def use_time_seconds := use_time_minutes * 60

-- Calculate the total vibrations Matt experiences
noncomputable def total_vibrations : ℕ := vibrations_highest * use_time_seconds

-- State the theorem
theorem vibrations_proof : total_vibrations = 768000 := 
by
  sorry

end vibrations_proof_l808_80849


namespace cheburashkas_erased_l808_80827

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l808_80827


namespace alok_paid_rs_811_l808_80807

/-
 Assume Alok ordered the following items at the given prices:
 - 16 chapatis, each costing Rs. 6
 - 5 plates of rice, each costing Rs. 45
 - 7 plates of mixed vegetable, each costing Rs. 70
 - 6 ice-cream cups

 Prove that the total cost Alok paid is Rs. 811.
-/
theorem alok_paid_rs_811 :
  let chapati_cost := 6
  let rice_plate_cost := 45
  let mixed_vegetable_plate_cost := 70
  let chapatis := 16 * chapati_cost
  let rice_plates := 5 * rice_plate_cost
  let mixed_vegetable_plates := 7 * mixed_vegetable_plate_cost
  chapatis + rice_plates + mixed_vegetable_plates = 811 := by
  sorry

end alok_paid_rs_811_l808_80807


namespace painted_cells_possible_values_l808_80845

theorem painted_cells_possible_values (k l : ℕ) (hk : 2 * k + 1 > 0) (hl : 2 * l + 1 > 0) (h : k * l = 74) :
  (2 * k + 1) * (2 * l + 1) - 74 = 301 ∨ (2 * k + 1) * (2 * l + 1) - 74 = 373 := 
sorry

end painted_cells_possible_values_l808_80845


namespace equation_has_three_distinct_solutions_iff_l808_80868

theorem equation_has_three_distinct_solutions_iff (a : ℝ) : 
  (∃ x_1 x_2 x_3 : ℝ, x_1 ≠ x_2 ∧ x_2 ≠ x_3 ∧ x_1 ≠ x_3 ∧ 
    (x_1 * |x_1 - a| = 1) ∧ (x_2 * |x_2 - a| = 1) ∧ (x_3 * |x_3 - a| = 1)) ↔ a > 2 :=
by
  sorry


end equation_has_three_distinct_solutions_iff_l808_80868


namespace find_number_l808_80823

theorem find_number (x : ℝ) : (45 * x = 0.45 * 900) → (x = 9) :=
by sorry

end find_number_l808_80823


namespace find_m_l808_80894

-- Define the set A
def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3 * m + 2}

-- Main theorem statement
theorem find_m (m : ℝ) (h : 2 ∈ A m) : m = 3 := by
  sorry

end find_m_l808_80894


namespace seth_spent_more_l808_80893

theorem seth_spent_more : 
  let ice_cream_cartons := 20
  let yogurt_cartons := 2
  let ice_cream_price := 6
  let yogurt_price := 1
  let ice_cream_discount := 0.10
  let yogurt_discount := 0.20
  let total_ice_cream_cost := ice_cream_cartons * ice_cream_price
  let total_yogurt_cost := yogurt_cartons * yogurt_price
  let discounted_ice_cream_cost := total_ice_cream_cost * (1 - ice_cream_discount)
  let discounted_yogurt_cost := total_yogurt_cost * (1 - yogurt_discount)
  discounted_ice_cream_cost - discounted_yogurt_cost = 106.40 :=
by
  sorry

end seth_spent_more_l808_80893


namespace no_two_champions_l808_80866

structure Tournament (Team : Type) :=
  (defeats : Team → Team → Prop)  -- Team A defeats Team B

def is_superior {Team : Type} (T : Tournament Team) (A B: Team) : Prop :=
  T.defeats A B ∨ ∃ C, T.defeats A C ∧ T.defeats C B

def is_champion {Team : Type} (T : Tournament Team) (A : Team) : Prop :=
  ∀ B, A ≠ B → is_superior T A B

theorem no_two_champions {Team : Type} (T : Tournament Team) :
  ¬ (∃ A B, A ≠ B ∧ is_champion T A ∧ is_champion T B) :=
sorry

end no_two_champions_l808_80866


namespace relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l808_80892

-- Prove that w - 2z = 0
theorem relation_w_z (w z : ℝ) : w - 2 * z = 0 :=
sorry

-- Prove that 2s + t - 8 = 0
theorem relation_s_t (s t : ℝ) : 2 * s + t - 8 = 0 :=
sorry

-- Prove that x - r - 2 = 0
theorem relation_x_r (x r : ℝ) : x - r - 2 = 0 :=
sorry

-- Prove that y + q - 6 = 0
theorem relation_y_q (y q : ℝ) : y + q - 6 = 0 :=
sorry

-- Prove that 3z - x - 2t + 6 = 0
theorem relation_z_x_t (z x t : ℝ) : 3 * z - x - 2 * t + 6 = 0 :=
sorry

-- Prove that 8z - 4t - v + 12 = 0
theorem relation_z_t_v (z t v : ℝ) : 8 * z - 4 * t - v + 12 = 0 :=
sorry

end relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l808_80892


namespace sqrt_of_4_l808_80840

theorem sqrt_of_4 (y : ℝ) : y^2 = 4 → (y = 2 ∨ y = -2) :=
sorry

end sqrt_of_4_l808_80840


namespace product_of_possible_values_N_l808_80879

theorem product_of_possible_values_N 
  (L M : ℤ) 
  (h1 : M = L + N) 
  (h2 : M - 7 = L + N - 7)
  (h3 : L + 5 = L + 5)
  (h4 : |(L + N - 7) - (L + 5)| = 4) : 
  N = 128 := 
  sorry

end product_of_possible_values_N_l808_80879


namespace car_highway_mileage_l808_80848

theorem car_highway_mileage :
  (∀ (H : ℝ), 
    (H > 0) → 
    (4 / H + 4 / 20 = (8 / H) * 1.4000000000000001) → 
    (H = 36)) :=
by
  intros H H_pos h_cond
  have : H = 36 := 
    sorry
  exact this

end car_highway_mileage_l808_80848


namespace first_more_than_200_paperclips_day_l808_80896

-- Definitions based on the conditions:
def paperclips_on_day (k : ℕ) : ℕ :=
  3 * 2^k

-- The theorem stating the solution:
theorem first_more_than_200_paperclips_day :
  ∃ k : ℕ, paperclips_on_day k > 200 ∧ k = 7 :=
by
  use 7
  sorry

end first_more_than_200_paperclips_day_l808_80896


namespace sum_of_coordinates_of_point_B_l808_80813

theorem sum_of_coordinates_of_point_B
  (x y : ℝ)
  (A : (ℝ × ℝ) := (2, 1))
  (B : (ℝ × ℝ) := (x, y))
  (h_line : y = 6)
  (h_slope : (y - 1) / (x - 2) = 4 / 5) :
  x + y = 14.25 :=
by {
  -- convert hypotheses to Lean terms and finish the proof
  sorry
}

end sum_of_coordinates_of_point_B_l808_80813


namespace integer_solutions_of_prime_equation_l808_80815

theorem integer_solutions_of_prime_equation (p : ℕ) (hp : Prime p) :
  ∃ x y : ℤ, (p * (x + y) = x * y) ↔ 
    (x = (p * (p + 1)) ∧ y = (p + 1)) ∨ 
    (x = 2 * p ∧ y = 2 * p) ∨ 
    (x = 0 ∧ y = 0) ∨ 
    (x = p * (1 - p) ∧ y = (p - 1)) := 
sorry

end integer_solutions_of_prime_equation_l808_80815


namespace minimum_value_2x_plus_y_l808_80842

theorem minimum_value_2x_plus_y (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : (1 / x) + (2 / (y + 1)) = 2) : 2 * x + y ≥ 3 := 
by
  sorry

end minimum_value_2x_plus_y_l808_80842


namespace students_exceed_hamsters_l808_80881

-- Definitions corresponding to the problem conditions
def students_per_classroom : ℕ := 20
def hamsters_per_classroom : ℕ := 1
def number_of_classrooms : ℕ := 5

-- Lean 4 statement to express the problem
theorem students_exceed_hamsters :
  (students_per_classroom * number_of_classrooms) - (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end students_exceed_hamsters_l808_80881


namespace no_rational_solutions_l808_80872

theorem no_rational_solutions : 
  ¬ ∃ (x y z : ℚ), 11 = x^5 + 2 * y^5 + 5 * z^5 := 
sorry

end no_rational_solutions_l808_80872


namespace machine_present_value_l808_80885

theorem machine_present_value
  (depreciation_rate : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (dep_years : ℕ)
  (value_after_depreciation : ℝ)
  (present_value : ℝ) :

  depreciation_rate = 0.8 →
  selling_price = 118000.00000000001 →
  profit = 22000 →
  dep_years = 2 →
  value_after_depreciation = (selling_price - profit) →
  value_after_depreciation = 96000.00000000001 →
  present_value * (depreciation_rate ^ dep_years) = value_after_depreciation →
  present_value = 150000.00000000002 :=
by sorry

end machine_present_value_l808_80885
