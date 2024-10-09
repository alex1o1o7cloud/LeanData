import Mathlib

namespace solve_expression_l428_42880

theorem solve_expression : 68 + (108 * 3) + (29^2) - 310 - (6 * 9) = 869 :=
by
  sorry

end solve_expression_l428_42880


namespace ratio_of_bike_to_tractor_speed_l428_42815

theorem ratio_of_bike_to_tractor_speed (d_tr: ℝ) (t_tr: ℝ) (d_car: ℝ) (t_car: ℝ) (k: ℝ) (β: ℝ) 
  (h1: d_tr / t_tr = 25) 
  (h2: d_car / t_car = 90)
  (h3: 90 = 9 / 5 * β)
: β / (d_tr / t_tr) = 2 := 
by
  sorry

end ratio_of_bike_to_tractor_speed_l428_42815


namespace find_a_l428_42884

variable (m n a : ℝ)
variable (h1 : m = 2 * n + 5)
variable (h2 : m + a = 2 * (n + 1.5) + 5)

theorem find_a : a = 3 := by
  sorry

end find_a_l428_42884


namespace simplify_fraction_l428_42854

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l428_42854


namespace part1_part2_l428_42809

section PartOne

variables (x y : ℕ)
def condition1 := x + y = 360
def condition2 := x - y = 110

theorem part1 (h1 : condition1 x y) (h2 : condition2 x y) : x = 235 ∧ y = 125 := by {
  sorry
}

end PartOne

section PartTwo

variables (t W : ℕ)
def tents_capacity (t : ℕ) := 40 * t + 20 * (9 - t)
def food_capacity (t : ℕ) := 10 * t + 20 * (9 - t)
def transportation_cost (t : ℕ) := 4000 * t + 3600 * (9 - t)

theorem part2 
  (htents : tents_capacity t ≥ 235) 
  (hfood : food_capacity t ≥ 125) : 
  W = transportation_cost t → t = 3 ∧ W = 33600 := by {
  sorry
}

end PartTwo

end part1_part2_l428_42809


namespace number_of_triangles_2016_30_l428_42816

def f (m n : ℕ) : ℕ :=
  2 * m - n - 2

theorem number_of_triangles_2016_30 :
  f 2016 30 = 4000 := 
by
  sorry

end number_of_triangles_2016_30_l428_42816


namespace sin_3x_sin_x_solutions_l428_42895

open Real

theorem sin_3x_sin_x_solutions :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (3 * x) = sin x) ∧ s.card = 7 := 
by sorry

end sin_3x_sin_x_solutions_l428_42895


namespace cloth_coloring_problem_l428_42865

theorem cloth_coloring_problem (lengthOfCloth : ℕ) 
  (women_can_color_100m_in_1_day : 5 * 1 = 100) 
  (women_can_color_in_3_days : 6 * 3 = lengthOfCloth) : lengthOfCloth = 360 := 
sorry

end cloth_coloring_problem_l428_42865


namespace numMilkmen_rented_pasture_l428_42857

def cowMonths (cows: ℕ) (months: ℕ) : ℕ := cows * months

def totalCowMonths (a: ℕ) (b: ℕ) (c: ℕ) (d: ℕ) : ℕ := a + b + c + d

noncomputable def rentPerCowMonth (share: ℕ) (cowMonths: ℕ) : ℕ := 
  share / cowMonths

theorem numMilkmen_rented_pasture 
  (a_cows: ℕ) (a_months: ℕ) (b_cows: ℕ) (b_months: ℕ) (c_cows: ℕ) (c_months: ℕ) (d_cows: ℕ) (d_months: ℕ)
  (a_share: ℕ) (total_rent: ℕ) 
  (ha: a_cows = 24) (hma: a_months = 3) 
  (hb: b_cows = 10) (hmb: b_months = 5)
  (hc: c_cows = 35) (hmc: c_months = 4)
  (hd: d_cows = 21) (hmd: d_months = 3)
  (ha_share: a_share = 720) (htotal_rent: total_rent = 3250)
  : 4 = 4 := by
  sorry

end numMilkmen_rented_pasture_l428_42857


namespace negation_example_l428_42825

theorem negation_example : ¬ (∃ x : ℤ, x^2 + 2 * x + 1 ≤ 0) ↔ ∀ x : ℤ, x^2 + 2 * x + 1 > 0 := 
by 
  sorry

end negation_example_l428_42825


namespace plane_equation_l428_42839

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + s + 2 * t, 4 - 2 * s, 1 - s + t)

def normal_vector : ℝ × ℝ × ℝ :=
  (-2, -3, 4)

def point_on_plane : ℝ × ℝ × ℝ :=
  (3, 4, 1)

theorem plane_equation : ∀ (x y z : ℝ),
  (∃ (s t : ℝ), (x, y, z) = parametric_plane s t) ↔
  2 * x + 3 * y - 4 * z - 14 = 0 :=
sorry

end plane_equation_l428_42839


namespace metal_rods_per_sheet_l428_42887

theorem metal_rods_per_sheet :
  (∀ (metal_rod_for_sheets metal_rod_for_beams total_metal_rod num_sheet_per_panel num_panel num_rod_per_beam),
    (num_rod_per_beam = 4) →
    (total_metal_rod = 380) →
    (metal_rod_for_beams = num_panel * (2 * num_rod_per_beam)) →
    (metal_rod_for_sheets = total_metal_rod - metal_rod_for_beams) →
    (num_sheet_per_panel = 3) →
    (num_panel = 10) →
    (metal_rod_per_sheet = metal_rod_for_sheets / (num_panel * num_sheet_per_panel)) →
    metal_rod_per_sheet = 10
  ) := sorry

end metal_rods_per_sheet_l428_42887


namespace number_of_yellow_parrots_l428_42851

theorem number_of_yellow_parrots (total_parrots : ℕ) (red_fraction : ℚ) 
  (h_total_parrots : total_parrots = 108) 
  (h_red_fraction : red_fraction = 5 / 6) : 
  ∃ (yellow_parrots : ℕ), yellow_parrots = total_parrots * (1 - red_fraction) ∧ yellow_parrots = 18 := 
by
  sorry

end number_of_yellow_parrots_l428_42851


namespace solution_set_f_le_1_l428_42863

variable {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem solution_set_f_le_1 :
  is_even_function f →
  monotone_on_nonneg f →
  f (-2) = 1 →
  {x : ℝ | f x ≤ 1} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} :=
by
  intros h_even h_mono h_f_neg_2
  sorry

end solution_set_f_le_1_l428_42863


namespace volume_of_quadrilateral_pyramid_l428_42845

theorem volume_of_quadrilateral_pyramid (m α : ℝ) : 
  ∃ (V : ℝ), V = (2 / 3) * m^3 * (Real.cos α) * (Real.sin (2 * α)) :=
by
  sorry

end volume_of_quadrilateral_pyramid_l428_42845


namespace shaded_region_area_l428_42892

noncomputable def shaded_area (π_approx : ℝ := 3.14) (r : ℝ := 1) : ℝ :=
  let square_area := (r / Real.sqrt 2) ^ 2
  let quarter_circle_area := (π_approx * r ^ 2) / 4
  quarter_circle_area - square_area

theorem shaded_region_area :
  shaded_area = 0.285 :=
by
  sorry

end shaded_region_area_l428_42892


namespace negation_proposition_l428_42821

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^3 + 5*x - 2 = 0) ↔ ∀ x : ℝ, x^3 + 5*x - 2 ≠ 0 :=
by sorry

end negation_proposition_l428_42821


namespace factor_polynomial_l428_42836

theorem factor_polynomial (x y : ℝ) : 
  (x^2 - 2*x*y + y^2 - 16) = (x - y + 4) * (x - y - 4) :=
sorry

end factor_polynomial_l428_42836


namespace x_in_M_sufficient_condition_for_x_in_N_l428_42889

def M := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0}
def N := {y : ℝ | ∃ x : ℝ, y = Real.sqrt ((1 - x) / x)}

theorem x_in_M_sufficient_condition_for_x_in_N :
  (∀ x, x ∈ M → x ∈ N) ∧ ¬ (∀ x, x ∈ N → x ∈ M) :=
by sorry

end x_in_M_sufficient_condition_for_x_in_N_l428_42889


namespace five_b_value_l428_42868

theorem five_b_value (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 4) 
  (h2 : a = b - 3) : 
  5 * b = 65 / 7 := 
by
  sorry

end five_b_value_l428_42868


namespace painters_completing_rooms_l428_42893

theorem painters_completing_rooms (three_painters_three_rooms_three_hours : 3 * 3 * 3 ≥ 3 * 3) :
  9 * 3 * 9 ≥ 9 * 27 :=
by 
  sorry

end painters_completing_rooms_l428_42893


namespace sector_area_l428_42841

theorem sector_area (r : ℝ) (θ : ℝ) (arc_area : ℝ) : 
  r = 24 ∧ θ = 110 ∧ arc_area = 176 * Real.pi → 
  arc_area = (θ / 360) * (Real.pi * r ^ 2) :=
by
  intros
  sorry

end sector_area_l428_42841


namespace least_number_of_marbles_divisible_l428_42834

theorem least_number_of_marbles_divisible (n : ℕ) : 
  (∀ k ∈ [2, 3, 4, 5, 6, 7, 8], n % k = 0) -> n >= 840 :=
by sorry

end least_number_of_marbles_divisible_l428_42834


namespace no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant_l428_42885

theorem no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant :
    ∀ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
                     (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) →
                     (a^2 + b^2 + c^2 + d^2 = 100) → False := by
  sorry

end no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant_l428_42885


namespace num_ways_write_100_as_distinct_squares_l428_42817

theorem num_ways_write_100_as_distinct_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a^2 + b^2 + c^2 = 100 ∧
  (∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x^2 + y^2 + z^2 = 100 ∧ (x, y, z) ≠ (a, b, c) ∧ (x, y, z) ≠ (a, c, b) ∧ (x, y, z) ≠ (b, a, c) ∧ (x, y, z) ≠ (b, c, a) ∧ (x, y, z) ≠ (c, a, b) ∧ (x, y, z) ≠ (c, b, a)) ∧
  ∀ (p q r : ℕ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p^2 + q^2 + r^2 = 100 → (p, q, r) = (a, b, c) ∨ (p, q, r) = (a, c, b) ∨ (p, q, r) = (b, a, c) ∨ (p, q, r) = (b, c, a) ∨ (p, q, r) = (c, a, b) ∨ (p, q, r) = (c, b, a) ∨ (p, q, r) = (x, y, z) ∨ (p, q, r) = (x, z, y) ∨ (p, q, r) = (y, x, z) ∨ (p, q, r) = (y, z, x) ∨ (p, q, r) = (z, x, y) ∨ (p, q, r) = (z, y, x) :=
sorry

end num_ways_write_100_as_distinct_squares_l428_42817


namespace expression_value_l428_42899

theorem expression_value (x : ℝ) (hx1 : x ≠ -1) (hx2 : x ≠ 2) :
  (2 * x ^ 2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 := 
by
  sorry

end expression_value_l428_42899


namespace inverse_of_g_l428_42866

noncomputable def u (x : ℝ) : ℝ := sorry
noncomputable def v (x : ℝ) : ℝ := sorry
noncomputable def w (x : ℝ) : ℝ := sorry

noncomputable def u_inv (x : ℝ) : ℝ := sorry
noncomputable def v_inv (x : ℝ) : ℝ := sorry
noncomputable def w_inv (x : ℝ) : ℝ := sorry

lemma u_inverse : ∀ x, u_inv (u x) = x ∧ u (u_inv x) = x := sorry
lemma v_inverse : ∀ x, v_inv (v x) = x ∧ v (v_inv x) = x := sorry
lemma w_inverse : ∀ x, w_inv (w x) = x ∧ w (w_inv x) = x := sorry

noncomputable def g (x : ℝ) : ℝ := v (u (w x))

noncomputable def g_inv (x : ℝ) : ℝ := w_inv (u_inv (v_inv x))

theorem inverse_of_g :
  ∀ x : ℝ, g_inv (g x) = x ∧ g (g_inv x) = x :=
by
  intro x
  -- proof omitted
  sorry

end inverse_of_g_l428_42866


namespace solid_is_cylinder_l428_42876

def solid_views (v1 v2 v3 : String) : Prop := 
  -- This definition makes a placeholder for the views of the solid.
  sorry

def is_cylinder (s : String) : Prop := 
  s = "Cylinder"

theorem solid_is_cylinder (v1 v2 v3 : String) (h : solid_views v1 v2 v3) :
  ∃ s : String, is_cylinder s :=
sorry

end solid_is_cylinder_l428_42876


namespace xyz_value_l428_42819

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
                x * y * z = 6 := by
  sorry

end xyz_value_l428_42819


namespace juan_more_marbles_l428_42844

theorem juan_more_marbles (connie_marbles : ℕ) (juan_marbles : ℕ) (h1 : connie_marbles = 323) (h2 : juan_marbles = 498) :
  juan_marbles - connie_marbles = 175 :=
by
  -- Proof goes here
  sorry

end juan_more_marbles_l428_42844


namespace soccer_balls_are_20_l428_42846

variable (S : ℕ)
variable (num_baseballs : ℕ) (num_volleyballs : ℕ)
variable (condition_baseballs : num_baseballs = 5 * S)
variable (condition_volleyballs : num_volleyballs = 3 * S)
variable (condition_total : num_baseballs + num_volleyballs = 160)

theorem soccer_balls_are_20 :
  S = 20 :=
by
  sorry

end soccer_balls_are_20_l428_42846


namespace weight_of_mixture_is_112_5_l428_42853

noncomputable def weight_of_mixture (W : ℝ) : Prop :=
  (5 / 14) * W + (3 / 10) * W + (2 / 9) * W + (1 / 7) * W + 2.5 = W

theorem weight_of_mixture_is_112_5 : ∃ W : ℝ, weight_of_mixture W ∧ W = 112.5 :=
by {
  use 112.5,
  sorry
}

end weight_of_mixture_is_112_5_l428_42853


namespace divisor_of_difference_is_62_l428_42881

-- The problem conditions as definitions
def x : Int := 859622
def y : Int := 859560
def difference : Int := x - y

-- The proof statement
theorem divisor_of_difference_is_62 (d : Int) (h₁ : d ∣ y) (h₂ : d ∣ difference) : d = 62 := by
  sorry

end divisor_of_difference_is_62_l428_42881


namespace total_distance_of_the_race_l428_42875

-- Define the given conditions
def A_beats_B_by_56_meters_or_7_seconds : Prop :=
  ∃ D : ℕ, ∀ S_B S_A : ℕ, S_B = 8 ∧ S_A = D / 8 ∧ D = S_B * (8 + 7)

-- Define the question and correct answer
theorem total_distance_of_the_race : A_beats_B_by_56_meters_or_7_seconds → ∃ D : ℕ, D = 120 :=
by
  sorry

end total_distance_of_the_race_l428_42875


namespace f_1984_and_f_1985_l428_42867

namespace Proof

variable {N M : Type} [AddMonoid M] [Zero M] (f : ℕ → M)

-- Conditions
axiom f_10 : f 10 = 0
axiom f_last_digit_3 {n : ℕ} : (n % 10 = 3) → f n = 0
axiom f_mn (m n : ℕ) : f (m * n) = f m + f n

-- Prove f(1984) = 0 and f(1985) = 0
theorem f_1984_and_f_1985 : f 1984 = 0 ∧ f 1985 = 0 :=
by
  sorry

end Proof

end f_1984_and_f_1985_l428_42867


namespace vitamin_C_in_apple_juice_l428_42802

theorem vitamin_C_in_apple_juice (A O : ℝ) 
  (h₁ : A + O = 185) 
  (h₂ : 2 * A + 3 * O = 452) :
  A = 103 :=
sorry

end vitamin_C_in_apple_juice_l428_42802


namespace sum_of_converted_2016_is_correct_l428_42800

theorem sum_of_converted_2016_is_correct :
  (20.16 + 20.16 + 20.16 + 201.6 + 201.6 + 201.6 = 463.68 ∨
   2.016 + 2.016 + 2.016 + 20.16 + 20.16 + 20.16 = 46.368) :=
by
  sorry

end sum_of_converted_2016_is_correct_l428_42800


namespace ratio_of_counters_l428_42896

theorem ratio_of_counters (C_K M_K C_total M_ratio : ℕ)
  (h1 : C_K = 40)
  (h2 : M_K = 50)
  (h3 : M_ratio = 4 * M_K)
  (h4 : C_total = C_K + M_ratio)
  (h5 : C_total = 320) :
  C_K ≠ 0 → (320 - M_ratio) / C_K = 3 :=
by
  sorry

end ratio_of_counters_l428_42896


namespace predict_sales_amount_l428_42805

theorem predict_sales_amount :
  let x_data := [2, 4, 5, 6, 8]
  let y_data := [30, 40, 50, 60, 70]
  let b := 7
  let x := 10 -- corresponding to 10,000 yuan investment
  let a := 15 -- \hat{a} calculated from the regression equation and data points
  let regression (x : ℝ) := b * x + a
  regression x = 85 :=
by
  -- Proof skipped
  sorry

end predict_sales_amount_l428_42805


namespace evaluate_72_squared_minus_48_squared_l428_42810

theorem evaluate_72_squared_minus_48_squared :
  (72:ℤ)^2 - (48:ℤ)^2 = 2880 :=
by
  sorry

end evaluate_72_squared_minus_48_squared_l428_42810


namespace original_price_increased_by_total_percent_l428_42813

noncomputable def percent_increase_sequence (P : ℝ) : ℝ :=
  let step1 := P * 1.15
  let step2 := step1 * 1.40
  let step3 := step2 * 1.20
  let step4 := step3 * 0.90
  let step5 := step4 * 1.25
  (step5 - P) / P * 100

theorem original_price_increased_by_total_percent (P : ℝ) : percent_increase_sequence P = 117.35 :=
by
  -- Sorry is used here for simplicity, but the automated proof will involve calculating the exact percentage increase step-by-step.
  sorry

end original_price_increased_by_total_percent_l428_42813


namespace cos_neg_13pi_div_4_l428_42831

theorem cos_neg_13pi_div_4 : (Real.cos (-13 * Real.pi / 4)) = -Real.sqrt 2 / 2 := 
by sorry

end cos_neg_13pi_div_4_l428_42831


namespace sin2alpha_cos2beta_l428_42835

variable (α β : ℝ)

-- Conditions
def tan_add_eq : Prop := Real.tan (α + β) = -3
def tan_sub_eq : Prop := Real.tan (α - β) = 2

-- Question
theorem sin2alpha_cos2beta (h1 : tan_add_eq α β) (h2 : tan_sub_eq α β) : 
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = -1 / 7 := 
  sorry

end sin2alpha_cos2beta_l428_42835


namespace find_some_number_l428_42872

noncomputable def some_number : ℝ := 1000
def expr_approx (a b c d : ℝ) := (a * b) / c = d

theorem find_some_number :
  expr_approx 3.241 14 some_number 0.045374000000000005 :=
by sorry

end find_some_number_l428_42872


namespace quadratic_inequality_solution_range_l428_42833

theorem quadratic_inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end quadratic_inequality_solution_range_l428_42833


namespace box_prices_l428_42814

theorem box_prices (a b c : ℝ) 
  (h1 : a + b + c = 9) 
  (h2 : 3 * a + 2 * b + c = 16) : 
  c - a = 2 := 
by 
  sorry

end box_prices_l428_42814


namespace cube_side_length_l428_42898

-- Given conditions for the problem
def surface_area (a : ℝ) : ℝ := 6 * a^2

-- Theorem statement
theorem cube_side_length (h : surface_area a = 864) : a = 12 :=
by
  sorry

end cube_side_length_l428_42898


namespace total_distance_is_75_l428_42822

def distance1 : ℕ := 30
def distance2 : ℕ := 20
def distance3 : ℕ := 25

def total_distance : ℕ := distance1 + distance2 + distance3

theorem total_distance_is_75 : total_distance = 75 := by
  sorry

end total_distance_is_75_l428_42822


namespace determine_m_if_root_exists_l428_42873

def fractional_equation_has_root (x m : ℝ) : Prop :=
  (3 / (x - 4) + (x + m) / (4 - x) = 1)

theorem determine_m_if_root_exists (x : ℝ) (h : fractional_equation_has_root x m) : m = -1 :=
sorry

end determine_m_if_root_exists_l428_42873


namespace simplify_identity_l428_42894

theorem simplify_identity :
  ∀ θ : ℝ, θ = 160 → (1 / (Real.sqrt (1 + Real.tan (θ : ℝ) ^ 2))) = -Real.cos (θ : ℝ) :=
by
  intro θ h
  rw [h]
  sorry  

end simplify_identity_l428_42894


namespace solve_for_a_l428_42843

theorem solve_for_a (x a : ℝ) (h : 3 * x + 2 * a = 3) (hx : x = 5) : a = -6 :=
by
  sorry

end solve_for_a_l428_42843


namespace email_difference_l428_42852

def morning_emails_early : ℕ := 10
def morning_emails_late : ℕ := 15
def afternoon_emails_early : ℕ := 7
def afternoon_emails_late : ℕ := 12

theorem email_difference :
  (morning_emails_early + morning_emails_late) - (afternoon_emails_early + afternoon_emails_late) = 6 :=
by
  sorry

end email_difference_l428_42852


namespace range_of_a_l428_42823

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x + a + 3
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * a

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, f a x₀ < 0 ∧ g a x₀ < 0) → 7 < a :=
by
  intro h
  sorry

end range_of_a_l428_42823


namespace gravitational_force_solution_l428_42806

noncomputable def gravitational_force_proportionality (d d' : ℕ) (f f' k : ℝ) : Prop :=
  (f * (d:ℝ)^2 = k) ∧
  d = 6000 ∧
  f = 800 ∧
  d' = 36000 ∧
  f' * (d':ℝ)^2 = k

theorem gravitational_force_solution : ∃ k, gravitational_force_proportionality 6000 36000 800 (1/45) k :=
by
  sorry

end gravitational_force_solution_l428_42806


namespace range_of_a_l428_42826

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 4 * x + a - 3 > 0) ↔ (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l428_42826


namespace find_a_values_l428_42879

noncomputable def system_has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((|y - 4| + |x + 12| - 3) * (x^2 + y^2 - 12) = 0) ∧ 
    ((x + 5)^2 + (y - 4)^2 = a)

theorem find_a_values : system_has_exactly_three_solutions 16 ∨ 
                        system_has_exactly_three_solutions (41 + 4 * Real.sqrt 123) :=
  by sorry

end find_a_values_l428_42879


namespace thomas_lost_pieces_l428_42820

theorem thomas_lost_pieces (audrey_lost : ℕ) (total_pieces_left : ℕ) (initial_pieces_each : ℕ) (total_pieces_initial : ℕ) (audrey_remaining_pieces : ℕ) (thomas_remaining_pieces : ℕ) : 
  audrey_lost = 6 → total_pieces_left = 21 → initial_pieces_each = 16 → total_pieces_initial = 32 → 
  audrey_remaining_pieces = initial_pieces_each - audrey_lost → 
  thomas_remaining_pieces = total_pieces_left - audrey_remaining_pieces → 
  initial_pieces_each - thomas_remaining_pieces = 5 :=
by
  sorry

end thomas_lost_pieces_l428_42820


namespace standard_spherical_coordinates_l428_42832

theorem standard_spherical_coordinates :
  ∀ (ρ θ φ : ℝ), 
  ρ = 5 → θ = 3 * Real.pi / 4 → φ = 9 * Real.pi / 5 →
  (ρ > 0) →
  (0 ≤ θ ∧ θ < 2 * Real.pi) →
  (0 ≤ φ ∧ φ ≤ Real.pi) →
  (ρ, θ, φ) = (5, 7 * Real.pi / 4, Real.pi / 5) :=
by sorry

end standard_spherical_coordinates_l428_42832


namespace abs_eq_cases_l428_42847

theorem abs_eq_cases (a b : ℝ) : (|a| = |b|) → (a = b ∨ a = -b) :=
sorry

end abs_eq_cases_l428_42847


namespace easter_eggs_total_l428_42804

theorem easter_eggs_total (h he total : ℕ)
 (hannah_eggs : h = 42) 
 (twice_he : h = 2 * he) 
 (total_eggs : total = h + he) : 
 total = 63 := 
sorry

end easter_eggs_total_l428_42804


namespace find_distance_city_A_B_l428_42858

-- Variables and givens
variable (D : ℝ)

-- Conditions from the problem
variable (JohnSpeed : ℝ := 40) (LewisSpeed : ℝ := 60)
variable (MeetDistance : ℝ := 160)
variable (TimeJohn : ℝ := (D - MeetDistance) / JohnSpeed)
variable (TimeLewis : ℝ := (D + MeetDistance) / LewisSpeed)

-- Lean 4 theorem statement for the proof
theorem find_distance_city_A_B :
  TimeJohn = TimeLewis → D = 800 :=
by
  sorry

end find_distance_city_A_B_l428_42858


namespace john_saving_yearly_l428_42864

def old_monthly_cost : ℕ := 1200
def increase_percentage : ℕ := 40
def split_count : ℕ := 3

def old_annual_cost (monthly_cost : ℕ) := monthly_cost * 12
def new_monthly_cost (monthly_cost : ℕ) (percentage : ℕ) := monthly_cost * (100 + percentage) / 100
def new_monthly_share (new_cost : ℕ) (split : ℕ) := new_cost / split
def new_annual_cost (monthly_share : ℕ) := monthly_share * 12
def annual_savings (old_annual : ℕ) (new_annual : ℕ) := old_annual - new_annual

theorem john_saving_yearly 
  (old_cost : ℕ := old_monthly_cost)
  (increase : ℕ := increase_percentage)
  (split : ℕ := split_count) :
  annual_savings (old_annual_cost old_cost) 
                 (new_annual_cost (new_monthly_share (new_monthly_cost old_cost increase) split)) 
  = 7680 :=
by
  sorry

end john_saving_yearly_l428_42864


namespace total_distance_run_l428_42830

-- Given conditions
def number_of_students : Nat := 18
def distance_per_student : Nat := 106

-- Prove that the total distance run by the students equals 1908 meters.
theorem total_distance_run : number_of_students * distance_per_student = 1908 := by
  sorry

end total_distance_run_l428_42830


namespace algebraic_expression_value_l428_42862

theorem algebraic_expression_value (a b : ℝ) (h : a + b = 3) : a^3 + b^3 + 9 * a * b = 27 :=
by
  sorry

end algebraic_expression_value_l428_42862


namespace pull_ups_of_fourth_student_l428_42878

theorem pull_ups_of_fourth_student 
  (avg_pullups : ℕ) 
  (num_students : ℕ) 
  (pullups_first : ℕ) 
  (pullups_second : ℕ) 
  (pullups_third : ℕ) 
  (pullups_fifth : ℕ) 
  (H_avg : avg_pullups = 10) 
  (H_students : num_students = 5) 
  (H_first : pullups_first = 9) 
  (H_second : pullups_second = 12) 
  (H_third : pullups_third = 9) 
  (H_fifth : pullups_fifth = 8) : 
  ∃ (pullups_fourth : ℕ), pullups_fourth = 12 := by
  sorry

end pull_ups_of_fourth_student_l428_42878


namespace calculate_value_l428_42860

theorem calculate_value : 15 * (1 / 3) + 45 * (2 / 3) = 35 := 
by
simp -- We use simp to simplify the expression
sorry -- We put sorry as we are skipping the full proof

end calculate_value_l428_42860


namespace team_average_typing_speed_l428_42824

-- Definitions of typing speeds of each team member
def typing_speed_rudy := 64
def typing_speed_joyce := 76
def typing_speed_gladys := 91
def typing_speed_lisa := 80
def typing_speed_mike := 89

-- Number of team members
def number_of_team_members := 5

-- Total typing speed calculation
def total_typing_speed := typing_speed_rudy + typing_speed_joyce + typing_speed_gladys + typing_speed_lisa + typing_speed_mike

-- Average typing speed calculation
def average_typing_speed := total_typing_speed / number_of_team_members

-- Theorem statement
theorem team_average_typing_speed : average_typing_speed = 80 := by
  sorry

end team_average_typing_speed_l428_42824


namespace marys_balloons_l428_42856

theorem marys_balloons (x y : ℝ) (h1 : x = 4 * y) (h2 : x = 7.0) : y = 1.75 := by
  sorry

end marys_balloons_l428_42856


namespace smallest_integer_divisibility_conditions_l428_42848

theorem smallest_integer_divisibility_conditions :
  ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (900 ∣ n^3) ∧ (1024 ∣ n^4) ∧ n = 120 :=
by
  sorry

end smallest_integer_divisibility_conditions_l428_42848


namespace perfect_square_trinomial_l428_42829

theorem perfect_square_trinomial :
  120^2 - 40 * 120 + 20^2 = 10000 := sorry

end perfect_square_trinomial_l428_42829


namespace find_n_l428_42890

theorem find_n (n x y k : ℕ) (h_coprime : Nat.gcd x y = 1) (h_eq : 3^n = x^k + y^k) : n = 2 :=
sorry

end find_n_l428_42890


namespace evening_temperature_l428_42871

-- Define the given conditions
def t_noon : ℤ := 1
def d : ℤ := 3

-- The main theorem stating that the evening temperature is -2℃
theorem evening_temperature : t_noon - d = -2 := by
  sorry

end evening_temperature_l428_42871


namespace area_of_FDBG_l428_42891

noncomputable def area_quadrilateral (AB AC : ℝ) (area_ABC : ℝ) : ℝ :=
  let AD := AB / 2
  let AE := AC / 2
  let sin_A := (2 * area_ABC) / (AB * AC)
  let area_ADE := (1 / 2) * AD * AE * sin_A
  let BC := (2 * area_ABC) / (AC * sin_A)
  let GC := BC / 3
  let area_AGC := (1 / 2) * AC * GC * sin_A
  area_ABC - (area_ADE + area_AGC)

theorem area_of_FDBG (AB AC : ℝ) (area_ABC : ℝ)
  (h1 : AB = 30)
  (h2 : AC = 15) 
  (h3 : area_ABC = 90) :
  area_quadrilateral AB AC area_ABC = 37.5 :=
by
  intros
  sorry

end area_of_FDBG_l428_42891


namespace car_actual_speed_is_40_l428_42861

variable (v : ℝ) -- actual speed (we will prove it is 40 km/h)

-- Conditions
variable (hyp_speed : ℝ := v + 20) -- hypothetical speed
variable (distance : ℝ := 60) -- distance traveled
variable (time_difference : ℝ := 0.5) -- time difference in hours

-- Define the equation derived from the given conditions:
def speed_equation : Prop :=
  (distance / v) - (distance / hyp_speed) = time_difference

-- The theorem to prove:
theorem car_actual_speed_is_40 : speed_equation v → v = 40 :=
by
  sorry

end car_actual_speed_is_40_l428_42861


namespace percentage_of_stock_l428_42818

noncomputable def investment_amount : ℝ := 6000
noncomputable def income_derived : ℝ := 756
noncomputable def brokerage_percentage : ℝ := 0.25
noncomputable def brokerage_fee : ℝ := investment_amount * (brokerage_percentage / 100)
noncomputable def net_investment_amount : ℝ := investment_amount - brokerage_fee
noncomputable def dividend_yield : ℝ := (income_derived / net_investment_amount) * 100

theorem percentage_of_stock :
  ∃ (percentage_of_stock : ℝ), percentage_of_stock = dividend_yield := by
  sorry

end percentage_of_stock_l428_42818


namespace arc_length_of_sector_l428_42811

noncomputable def central_angle := 36
noncomputable def radius := 15

theorem arc_length_of_sector : (central_angle * Real.pi * radius / 180 = 3 * Real.pi) :=
by
  sorry

end arc_length_of_sector_l428_42811


namespace circles_intersect_and_common_chord_l428_42828

theorem circles_intersect_and_common_chord :
  (∃ P : ℝ × ℝ, P.1 ^ 2 + P.2 ^ 2 - P.1 + P.2 - 2 = 0 ∧
                P.1 ^ 2 + P.2 ^ 2 = 5) ∧
  (∀ x y : ℝ, (x ^ 2 + y ^ 2 - x + y - 2 = 0 ∧ x ^ 2 + y ^ 2 = 5) →
              x - y - 3 = 0) ∧
  (∃ A B : ℝ × ℝ, A.1 ^ 2 + A.2 ^ 2 - A.1 + A.2 - 2 = 0 ∧
                   A.1 ^ 2 + A.2 ^ 2 = 5 ∧
                   B.1 ^ 2 + B.2 ^ 2 - B.1 + B.2 - 2 = 0 ∧
                   B.1 ^ 2 + B.2 ^ 2 = 5 ∧
                   (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 2) := sorry

end circles_intersect_and_common_chord_l428_42828


namespace dave_paid_for_6_candy_bars_l428_42838

-- Given conditions
def number_of_candy_bars : ℕ := 20
def cost_per_candy_bar : ℝ := 1.50
def amount_paid_by_john : ℝ := 21

-- Correct answer
def number_of_candy_bars_paid_by_dave : ℝ := 6

-- The proof problem in Lean statement
theorem dave_paid_for_6_candy_bars (H : number_of_candy_bars * cost_per_candy_bar - amount_paid_by_john = 9) :
  number_of_candy_bars_paid_by_dave = 6 := by
sorry

end dave_paid_for_6_candy_bars_l428_42838


namespace cone_to_sphere_ratio_l428_42850

-- Prove the ratio of the cone's altitude to its base radius
theorem cone_to_sphere_ratio (r h : ℝ) (h_r_pos : 0 < r) 
  (vol_cone : ℝ) (vol_sphere : ℝ) 
  (hyp_vol_relation : vol_cone = (1 / 3) * vol_sphere)
  (vol_sphere_def : vol_sphere = (4 / 3) * π * r^3)
  (vol_cone_def : vol_cone = (1 / 3) * π * r^2 * h) :
  h / r = 4 / 3 :=
by
  sorry

end cone_to_sphere_ratio_l428_42850


namespace inverse_proposition_l428_42842

-- Define the variables m, n, and a^2
variables (m n : ℝ) (a : ℝ)

-- State the proof problem
theorem inverse_proposition
  (h1 : m > n)
: m * a^2 > n * a^2 :=
sorry

end inverse_proposition_l428_42842


namespace two_digit_number_system_l428_42837

theorem two_digit_number_system (x y : ℕ) :
  (10 * x + y - 3 * (x + y) = 13) ∧ (10 * x + y - 6 = 4 * (x + y)) :=
by sorry

end two_digit_number_system_l428_42837


namespace least_possible_value_expression_l428_42803

theorem least_possible_value_expression :
  ∃ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 = 2039 :=
sorry

end least_possible_value_expression_l428_42803


namespace solve_cryptarithm_l428_42827

-- Definitions for digits mapped to letters
def C : ℕ := 9
def H : ℕ := 3
def U : ℕ := 5
def K : ℕ := 4
def T : ℕ := 1
def R : ℕ := 2
def I : ℕ := 0
def G : ℕ := 6
def N : ℕ := 8
def S : ℕ := 7

-- Function to evaluate the cryptarithm sum
def cryptarithm_sum : ℕ :=
  (C*10000 + H*1000 + U*100 + C*10 + K) +
  (T*10000 + R*1000 + I*100 + G*10 + G) +
  (T*10000 + U*1000 + R*100 + N*10 + S)

-- Equation checking the result
def cryptarithm_correct : Prop :=
  cryptarithm_sum = T*100000 + R*10000 + I*1000 + C*100 + K*10 + S

-- The theorem we want to prove
theorem solve_cryptarithm : cryptarithm_correct :=
by
  -- Proof steps would be filled here
  -- but for now, we just acknowledge it is a theorem
  sorry

end solve_cryptarithm_l428_42827


namespace book_contains_300_pages_l428_42886

-- The given conditions
def total_digits : ℕ := 792
def digits_per_page_1_to_9 : ℕ := 9 * 1
def digits_per_page_10_to_99 : ℕ := 90 * 2
def remaining_digits : ℕ := total_digits - digits_per_page_1_to_9 - digits_per_page_10_to_99
def pages_with_3_digits : ℕ := remaining_digits / 3

-- The total number of pages
def total_pages : ℕ := 99 + pages_with_3_digits

theorem book_contains_300_pages : total_pages = 300 := by
  sorry

end book_contains_300_pages_l428_42886


namespace solve_for_q_l428_42807

variable (R t m q : ℝ)

def given_condition : Prop :=
  R = t / ((2 + m) ^ q)

theorem solve_for_q (h : given_condition R t m q) : 
  q = (Real.log (t / R)) / (Real.log (2 + m)) := 
sorry

end solve_for_q_l428_42807


namespace pieces_on_third_day_impossibility_of_2014_pieces_l428_42808

-- Define the process of dividing and eating chocolate pieces.
def chocolate_pieces (n : ℕ) : ℕ :=
  9 + 8 * n

-- The number of pieces after the third day.
theorem pieces_on_third_day : chocolate_pieces 3 = 25 :=
sorry

-- It's impossible for Maria to have exactly 2014 pieces on any given day.
theorem impossibility_of_2014_pieces : ∀ n : ℕ, chocolate_pieces n ≠ 2014 :=
sorry

end pieces_on_third_day_impossibility_of_2014_pieces_l428_42808


namespace remainder_proof_l428_42840

noncomputable def problem (n : ℤ) : Prop :=
  n % 9 = 4

noncomputable def solution (n : ℤ) : ℤ :=
  (4 * n - 11) % 9

theorem remainder_proof (n : ℤ) (h : problem n) : solution n = 5 := by
  sorry

end remainder_proof_l428_42840


namespace gumball_problem_l428_42882

theorem gumball_problem:
  ∀ (total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs: ℕ),
    total_gumballs = 45 →
    given_to_Todd = 4 →
    given_to_Alisha = 2 * given_to_Todd →
    remaining_gumballs = 6 →
    given_to_Todd + given_to_Alisha + given_to_Bobby + remaining_gumballs = total_gumballs →
    given_to_Bobby = 45 - 18 →
    4 * given_to_Alisha - given_to_Bobby = 5 :=
by
  intros total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs ht hTodd hAlisha hRemaining hSum hBobby
  rw [ht, hTodd] at *
  rw [hAlisha, hRemaining] at *
  sorry

end gumball_problem_l428_42882


namespace dice_probability_exactly_four_twos_l428_42870

theorem dice_probability_exactly_four_twos :
  let probability := (Nat.choose 8 4 : ℚ) * (1 / 8)^4 * (7 / 8)^4 
  probability = 168070 / 16777216 :=
by
  sorry

end dice_probability_exactly_four_twos_l428_42870


namespace book_price_net_change_l428_42812

theorem book_price_net_change (P : ℝ) :
  let decreased_price := P * 0.70
  let increased_price := decreased_price * 1.20
  let net_change := (increased_price - P) / P * 100
  net_change = -16 := 
by
  sorry

end book_price_net_change_l428_42812


namespace rectangle_area_increase_l428_42849

variable {L W : ℝ} -- Define variables for length and width

theorem rectangle_area_increase (p : ℝ) (hW : W' = 0.4 * W) (hA : A' = 1.36 * (L * W)) :
  L' = L + (240 / 100) * L :=
by
  sorry

end rectangle_area_increase_l428_42849


namespace distinct_sum_product_problem_l428_42883

theorem distinct_sum_product_problem (S : ℤ) (hS : S ≥ 100) :
  ∃ a b c P : ℤ, a > b ∧ b > c ∧ a + b + c = S ∧ a * b * c = P ∧ 
    ¬(∀ x y z : ℤ, x > y ∧ y > z ∧ x + y + z = S → a = x ∧ b = y ∧ c = z) := 
sorry

end distinct_sum_product_problem_l428_42883


namespace distributive_addition_over_multiplication_not_hold_l428_42801

def complex_add (z1 z2 : ℝ × ℝ) : ℝ × ℝ :=
(z1.1 + z2.1, z1.2 + z2.2)

def complex_mul (z1 z2 : ℝ × ℝ) : ℝ × ℝ :=
(z1.1 * z2.1 - z1.2 * z2.2, z1.1 * z2.2 + z1.2 * z2.1)

theorem distributive_addition_over_multiplication_not_hold (x y x1 y1 x2 y2 : ℝ) :
  complex_add (x, y) (complex_mul (x1, y1) (x2, y2)) ≠
    complex_mul (complex_add (x, y) (x1, y1)) (complex_add (x, y) (x2, y2)) :=
sorry

end distributive_addition_over_multiplication_not_hold_l428_42801


namespace functional_equation_solution_l428_42869

theorem functional_equation_solution {f : ℚ → ℚ} :
  (∀ x y z t : ℚ, x < y ∧ y < z ∧ z < t ∧ (y - x) = (z - y) ∧ (z - y) = (t - z) →
    f x + f t = f y + f z) → 
  ∃ c b : ℚ, ∀ q : ℚ, f q = c * q + b := 
by
  sorry

end functional_equation_solution_l428_42869


namespace base8_minus_base7_base10_eq_l428_42897

-- Definitions of the two numbers in their respective bases
def n1_base8 : ℕ := 305
def n2_base7 : ℕ := 165

-- Conversion of these numbers to base 10
def n1_base10 : ℕ := 3 * 8^2 + 0 * 8^1 + 5 * 8^0
def n2_base10 : ℕ := 1 * 7^2 + 6 * 7^1 + 5 * 7^0

-- Statement of the theorem to be proven
theorem base8_minus_base7_base10_eq :
  (n1_base10 - n2_base10 = 101) :=
  by
    -- The proof would go here
    sorry

end base8_minus_base7_base10_eq_l428_42897


namespace harry_terry_difference_l428_42874

theorem harry_terry_difference :
  let H := 12 - (3 + 6)
  let T := 12 - 3 + 6 * 2
  H - T = -18 :=
by
  sorry

end harry_terry_difference_l428_42874


namespace tens_digit_of_3_pow_2023_l428_42888

theorem tens_digit_of_3_pow_2023 : (3 ^ 2023 % 100) / 10 = 2 := 
sorry

end tens_digit_of_3_pow_2023_l428_42888


namespace range_a_if_no_solution_l428_42855

def f (x : ℝ) : ℝ := abs (x - abs (2 * x - 4))

theorem range_a_if_no_solution (a : ℝ) :
  (∀ x : ℝ, f x > 0 → false) → a < 1 :=
by
  sorry

end range_a_if_no_solution_l428_42855


namespace smallest_repeating_block_digits_l428_42877

theorem smallest_repeating_block_digits (n : ℕ) (d : ℕ) (hd_pos : d > 0) (hd_coprime : Nat.gcd n d = 1)
  (h_fraction : (n : ℚ) / d = 8 / 11) : n = 2 :=
by
  -- proof will go here
  sorry

end smallest_repeating_block_digits_l428_42877


namespace probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l428_42859

-- Defining the conditions
def p : ℚ := 4 / 5
def n : ℕ := 5
def k1 : ℕ := 2
def k2 : ℕ := 1

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Binomial probability function
def binom_prob (k n : ℕ) (p : ℚ) : ℚ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- The first proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate is 0.05 given the accuracy rate
theorem probability_of_2_out_of_5_accurate :
  binom_prob k1 n p = 0.05 := by
  sorry

-- The second proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate, with the third forecast being one of the accurate ones, is 0.02 given the accuracy rate
theorem probability_of_2_out_of_5_with_third_accurate :
  binom_prob k2 (n - 1) p = 0.02 := by
  sorry

end probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l428_42859
