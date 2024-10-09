import Mathlib

namespace find_smallest_number_l1277_127744

variable (x : ℕ)

def second_number := 2 * x
def third_number := 4 * second_number x
def average := (x + second_number x + third_number x) / 3

theorem find_smallest_number (h : average x = 165) : x = 45 := by
  sorry

end find_smallest_number_l1277_127744


namespace least_value_of_g_l1277_127773

noncomputable def g (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 1

theorem least_value_of_g : ∃ x : ℝ, ∀ y : ℝ, g y ≥ g x ∧ g x = -2 := by
  sorry

end least_value_of_g_l1277_127773


namespace greatest_common_divisor_is_one_l1277_127797

-- Define the expressions for a and b
def a : ℕ := 114^2 + 226^2 + 338^2
def b : ℕ := 113^2 + 225^2 + 339^2

-- Now state that the gcd of a and b is 1
theorem greatest_common_divisor_is_one : Nat.gcd a b = 1 := sorry

end greatest_common_divisor_is_one_l1277_127797


namespace mrs_hilt_total_payment_l1277_127790

-- Define the conditions
def number_of_hot_dogs : ℕ := 6
def cost_per_hot_dog : ℝ := 0.50

-- Define the total cost
def total_cost : ℝ := number_of_hot_dogs * cost_per_hot_dog

-- State the theorem to prove the total cost
theorem mrs_hilt_total_payment : total_cost = 3.00 := 
by
  sorry

end mrs_hilt_total_payment_l1277_127790


namespace second_candidate_percentage_l1277_127784

theorem second_candidate_percentage (V : ℝ) (h1 : 0.15 * V ≠ 0) (h2 : 0.38 * V ≠ 300) :
  (0.38 * V - 300) / (0.85 * V - 250) * 100 = 44.71 :=
by 
  -- Let the math proof be synthesized by a more detailed breakdown of conditions and theorems
  sorry

end second_candidate_percentage_l1277_127784


namespace y_equals_x_l1277_127726

theorem y_equals_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 1 + 1 / y) (h2 : y = 1 + 1 / x) : y = x :=
sorry

end y_equals_x_l1277_127726


namespace coordinates_of_A_l1277_127710

-- Definitions based on conditions
def origin : ℝ × ℝ := (0, 0)
def similarity_ratio : ℝ := 2
def point_A : ℝ × ℝ := (2, 3)
def point_A' (P : ℝ × ℝ) : Prop :=
  P = (similarity_ratio * point_A.1, similarity_ratio * point_A.2) ∨
  P = (-similarity_ratio * point_A.1, -similarity_ratio * point_A.2)

-- Statement of the theorem
theorem coordinates_of_A' :
  ∃ P : ℝ × ℝ, point_A' P :=
by
  use (4, 6)
  left
  sorry

end coordinates_of_A_l1277_127710


namespace greatest_mean_weight_l1277_127759

variable (X Y Z : Type) [Group X] [Group Y] [Group Z]

theorem greatest_mean_weight 
  (mean_X : ℝ) (mean_Y : ℝ) (mean_XY : ℝ) (mean_XZ : ℝ)
  (hX : mean_X = 30)
  (hY : mean_Y = 70)
  (hXY : mean_XY = 50)
  (hXZ : mean_XZ = 40) :
  ∃ k : ℝ, k = 70 :=
by {
  sorry
}

end greatest_mean_weight_l1277_127759


namespace mark_hourly_wage_before_raise_40_l1277_127713

-- Mark's hourly wage before the raise
def hourly_wage_before_raise (x : ℝ) : Prop :=
  let weekly_hours := 40
  let raise_percentage := 0.05
  let new_hourly_wage := x * (1 + raise_percentage)
  let new_weekly_earnings := weekly_hours * new_hourly_wage
  let old_bills := 600
  let personal_trainer := 100
  let new_expenses := old_bills + personal_trainer
  let leftover_income := 980
  new_weekly_earnings = new_expenses + leftover_income

-- Proving that Mark's hourly wage before the raise was 40 dollars
theorem mark_hourly_wage_before_raise_40 : hourly_wage_before_raise 40 :=
by
  -- Proof goes here
  sorry

end mark_hourly_wage_before_raise_40_l1277_127713


namespace find_k_l1277_127755

theorem find_k (k : ℤ) : 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    (x1, y1) = (2, 9) ∧ (x2, y2) = (5, 18) ∧ (x3, y3) = (8, 27) ∧ 
    ∃ m b : ℤ, y1 = m * x1 + b ∧ y2 = m * x2 + b ∧ y3 = m * x3 + b) 
  ∧ ∃ m b : ℤ, k = m * 42 + b
  → k = 129 :=
sorry

end find_k_l1277_127755


namespace sin_theta_of_triangle_area_side_median_l1277_127719

-- Defining the problem statement and required conditions
theorem sin_theta_of_triangle_area_side_median (A : ℝ) (a m : ℝ) (θ : ℝ) 
  (hA : A = 30)
  (ha : a = 12)
  (hm : m = 8)
  (hTriangleArea : A = 1/2 * a * m * Real.sin θ) :
  Real.sin θ = 5 / 8 :=
by
  -- Proof omitted
  sorry

end sin_theta_of_triangle_area_side_median_l1277_127719


namespace min_value_x2_y2_l1277_127772

theorem min_value_x2_y2 (x y : ℝ) (h : x + y = 2) : ∃ m, m = x^2 + y^2 ∧ (∀ (x y : ℝ), x + y = 2 → x^2 + y^2 ≥ m) ∧ m = 2 := 
sorry

end min_value_x2_y2_l1277_127772


namespace find_a_l1277_127761

theorem find_a (a : ℕ) : 
  (a >= 100 ∧ a <= 999) ∧ 7 ∣ (504000 + a) ∧ 9 ∣ (504000 + a) ∧ 11 ∣ (504000 + a) ↔ a = 711 :=
by {
  sorry
}

end find_a_l1277_127761


namespace areas_of_geometric_figures_with_equal_perimeter_l1277_127712

theorem areas_of_geometric_figures_with_equal_perimeter (l : ℝ) (h : (l > 0)) :
  let s1 := l^2 / (4 * Real.pi)
  let s2 := l^2 / 16
  let s3 := (Real.sqrt 3) * l^2 / 36
  s1 > s2 ∧ s2 > s3 := by
  sorry

end areas_of_geometric_figures_with_equal_perimeter_l1277_127712


namespace sphere_surface_area_increase_l1277_127764

theorem sphere_surface_area_increase (r : ℝ) (h_r_pos : 0 < r):
  let A := 4 * π * r ^ 2
  let r' := 1.10 * r
  let A' := 4 * π * (r') ^ 2
  let ΔA := A' - A
  (ΔA / A) * 100 = 21 := by
  sorry

end sphere_surface_area_increase_l1277_127764


namespace monday_has_greatest_temp_range_l1277_127767

-- Define the temperatures
def high_temp (day : String) : Int :=
  if day = "Monday" then 6 else
  if day = "Tuesday" then 3 else
  if day = "Wednesday" then 4 else
  if day = "Thursday" then 4 else
  if day = "Friday" then 8 else 0

def low_temp (day : String) : Int :=
  if day = "Monday" then -4 else
  if day = "Tuesday" then -6 else
  if day = "Wednesday" then -2 else
  if day = "Thursday" then -5 else
  if day = "Friday" then 0 else 0

-- Define the temperature range for a given day
def temp_range (day : String) : Int :=
  high_temp day - low_temp day

-- Statement to prove: Monday has the greatest temperature range
theorem monday_has_greatest_temp_range : 
  temp_range "Monday" > temp_range "Tuesday" ∧
  temp_range "Monday" > temp_range "Wednesday" ∧
  temp_range "Monday" > temp_range "Thursday" ∧
  temp_range "Monday" > temp_range "Friday" := 
sorry

end monday_has_greatest_temp_range_l1277_127767


namespace part_a_l1277_127798

theorem part_a (a : ℤ) : (a^2 < 4) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := 
sorry

end part_a_l1277_127798


namespace probability_of_winning_correct_l1277_127796

noncomputable def probability_of_winning (P_L : ℚ) (P_T : ℚ) : ℚ :=
  1 - (P_L + P_T)

theorem probability_of_winning_correct :
  probability_of_winning (3/7) (2/21) = 10/21 :=
by
  sorry

end probability_of_winning_correct_l1277_127796


namespace simplify_and_find_ratio_l1277_127740

theorem simplify_and_find_ratio (m : ℤ) : 
  let expr := (6 * m + 18) / 6 
  let c := 1
  let d := 3
  (c / d : ℚ) = 1 / 3 := 
by
  -- Conditions and transformations are stated here
  -- (6 * m + 18) / 6 can be simplified step-by-step
  sorry

end simplify_and_find_ratio_l1277_127740


namespace maximum_xy_l1277_127766

theorem maximum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_parallel : 2 * x + y = 2) : 
  xy ≤ 1/2 := 
  sorry

end maximum_xy_l1277_127766


namespace equivalent_single_percentage_increase_l1277_127734

noncomputable def calculate_final_price (p : ℝ) : ℝ :=
  let p1 := p * (1 + 0.15)
  let p2 := p1 * (1 + 0.20)
  let p_final := p2 * (1 - 0.10)
  p_final

theorem equivalent_single_percentage_increase (p : ℝ) : 
  calculate_final_price p = p * 1.242 :=
by
  sorry

end equivalent_single_percentage_increase_l1277_127734


namespace polynomial_coeff_sum_eq_four_l1277_127776

theorem polynomial_coeff_sum_eq_four (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ) :
  (∀ x : ℤ, (2 * x - 1)^6 * (x + 1)^2 = a * x ^ 8 + a1 * x ^ 7 + a2 * x ^ 6 + a3 * x ^ 5 + 
                      a4 * x ^ 4 + a5 * x ^ 3 + a6 * x ^ 2 + a7 * x + a8) →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 := by
  sorry

end polynomial_coeff_sum_eq_four_l1277_127776


namespace determine_k_l1277_127775

theorem determine_k (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by
  sorry

end determine_k_l1277_127775


namespace solve_quadratic_equation_l1277_127703

theorem solve_quadratic_equation (x : ℝ) : 4 * (x - 1)^2 = 36 ↔ (x = 4 ∨ x = -2) :=
by sorry

end solve_quadratic_equation_l1277_127703


namespace vector_parallel_m_eq_two_neg_two_l1277_127787

theorem vector_parallel_m_eq_two_neg_two (m : ℝ) (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 / x = m / y) : m = 2 ∨ m = -2 :=
by
  sorry

end vector_parallel_m_eq_two_neg_two_l1277_127787


namespace certain_event_at_least_one_good_product_l1277_127754

-- Define the number of products and their types
def num_products := 12
def num_good_products := 10
def num_defective_products := 2
def num_selected_products := 3

-- Statement of the problem
theorem certain_event_at_least_one_good_product :
  ∀ (selected : Finset (Fin num_products)),
  selected.card = num_selected_products →
  ∃ p ∈ selected, p.val < num_good_products :=
sorry

end certain_event_at_least_one_good_product_l1277_127754


namespace expression_equality_l1277_127725

theorem expression_equality : 
  (∀ (x : ℝ) (a k n : ℝ), (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n → a = 6 ∧ k = -5 ∧ n = -6) → 
  ∀ (a k n : ℝ), a = 6 → k = -5 → n = -6 → a - n + k = 7 :=
by
  intro h
  intros a k n ha hk hn
  rw [ha, hk, hn]
  norm_num

end expression_equality_l1277_127725


namespace problem_statement_l1277_127702

theorem problem_statement :
  ∃ (n : ℕ), n = 101 ∧
  (∀ (x : ℕ), x < 4032 → ((x^2 - 20) % 16 = 0) ∧ ((x^2 - 16) % 20 = 0) ↔ (∃ k1 k2 : ℕ, (x = 80 * k1 + 6 ∨ x = 80 * k2 + 74) ∧ k1 + k2 + 1 = n)) :=
by sorry

end problem_statement_l1277_127702


namespace area_of_square_l1277_127785

noncomputable def square_area (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) : ℝ :=
  (v * v) / 4

theorem area_of_square (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) (h_cond : ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → B = (u, 0) → C = (u, v) → 
  (u - 0) * (u - 0) + (v - 0) * (v - 0) = (u - 0) * (u - 0)) :
  square_area u v h_u h_v = v * v / 4 := 
by 
  sorry

end area_of_square_l1277_127785


namespace common_ratio_of_geometric_progression_l1277_127792

-- Define the problem conditions
variables {a b c q : ℝ}

-- The sequence a, b, c is a geometric progression
def geometric_progression (a b c : ℝ) (q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- The sequence 577a, (2020b/7), (c/7) is an arithmetic progression
def arithmetic_progression (x y z : ℝ) : Prop :=
  2 * y = x + z

-- Main theorem statement to prove
theorem common_ratio_of_geometric_progression (h1 : geometric_progression a b c q) 
  (h2 : arithmetic_progression (577 * a) (2020 * b / 7) (c / 7)) 
  (h3 : b < a ∧ c < b) : q = 4039 :=
sorry

end common_ratio_of_geometric_progression_l1277_127792


namespace income_ratio_l1277_127782

-- Define the conditions
variables (I_A I_B E_A E_B : ℝ)
variables (Savings_A Savings_B : ℝ)

-- Given conditions
def expenditure_ratio : E_A / E_B = 3 / 2 := sorry
def savings_A : Savings_A = 1600 := sorry
def savings_B : Savings_B = 1600 := sorry
def income_A : I_A = 4000 := sorry
def expenditure_A : E_A = I_A - Savings_A := sorry
def expenditure_B : E_B = I_B - Savings_B := sorry

-- Prove it's implied that the ratio of incomes is 5:4
theorem income_ratio : I_A / I_B = 5 / 4 :=
by
  sorry

end income_ratio_l1277_127782


namespace value_of_business_calculation_l1277_127742

noncomputable def value_of_business (total_shares_sold_value : ℝ) (shares_fraction_sold : ℝ) (ownership_fraction : ℝ) : ℝ :=
  (total_shares_sold_value / shares_fraction_sold) * ownership_fraction⁻¹

theorem value_of_business_calculation :
  value_of_business 45000 (3/4) (2/3) = 90000 :=
by
  sorry

end value_of_business_calculation_l1277_127742


namespace opposite_of_neg_six_is_six_l1277_127718

theorem opposite_of_neg_six_is_six : 
  ∃ (x : ℝ), (-6 + x = 0) ∧ x = 6 := by
  sorry

end opposite_of_neg_six_is_six_l1277_127718


namespace convert_mps_to_kmph_l1277_127786

-- Define the conversion factor
def conversion_factor : ℝ := 3.6

-- Define the initial speed in meters per second
def initial_speed_mps : ℝ := 50

-- Define the target speed in kilometers per hour
def target_speed_kmph : ℝ := 180

-- Problem statement: Prove the conversion is correct
theorem convert_mps_to_kmph : initial_speed_mps * conversion_factor = target_speed_kmph := by
  sorry

end convert_mps_to_kmph_l1277_127786


namespace cost_price_A_l1277_127778

theorem cost_price_A (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ) 
(h1 : CP_B = 1.20 * CP_A)
(h2 : SP_C = 1.25 * CP_B)
(h3 : SP_C = 225) : 
CP_A = 150 := 
by 
  sorry

end cost_price_A_l1277_127778


namespace total_drink_volume_l1277_127745

variable (T : ℝ)

theorem total_drink_volume :
  (0.15 * T + 0.60 * T + 0.25 * T = 35) → T = 140 :=
by
  intros h
  have h1 : (0.25 * T) = 35 := by sorry
  have h2 : T = 140 := by sorry
  exact h2

end total_drink_volume_l1277_127745


namespace find_angle_A_find_sum_b_c_l1277_127722

-- Given the necessary conditions
variables (a b c : ℝ)
variables (A B C : ℝ)
variables (sin cos : ℝ → ℝ)

-- Assuming necessary trigonometric identities
axiom sin_squared_add_cos_squared : ∀ (x : ℝ), sin x * sin x + cos x * cos x = 1
axiom cos_sum : ∀ (x y : ℝ), cos (x + y) = cos x * cos y - sin x * sin y

-- Condition: 2 sin^2(A) + 3 cos(B+C) = 0
axiom condition1 : 2 * sin A * sin A + 3 * cos (B + C) = 0

-- Condition: The area of the triangle is S = 5 √3
axiom condition2 : 1 / 2 * b * c * sin A = 5 * Real.sqrt 3

-- Condition: The length of side a = √21
axiom condition3 : a = Real.sqrt 21

-- Part (1): Prove the measure of angle A
theorem find_angle_A : A = π / 3 :=
sorry

-- Part (2): Given S = 5√3 and a = √21, find b + c.
theorem find_sum_b_c : b + c = 9 :=
sorry

end find_angle_A_find_sum_b_c_l1277_127722


namespace suitable_survey_l1277_127780

def survey_suitable_for_census (A B C D : Prop) : Prop :=
  A ∧ ¬B ∧ ¬C ∧ ¬D

theorem suitable_survey {A B C D : Prop} (h_A : A) (h_B : ¬B) (h_C : ¬C) (h_D : ¬D) : survey_suitable_for_census A B C D :=
by
  unfold survey_suitable_for_census
  exact ⟨h_A, h_B, h_C, h_D⟩

end suitable_survey_l1277_127780


namespace smartphone_cost_decrease_l1277_127707

theorem smartphone_cost_decrease :
  ∀ (cost2010 cost2020 : ℝ),
  cost2010 = 600 →
  cost2020 = 450 →
  ((cost2010 - cost2020) / cost2010) * 100 = 25 :=
by
  intros cost2010 cost2020 h1 h2
  sorry

end smartphone_cost_decrease_l1277_127707


namespace rug_area_is_180_l1277_127731

variables (w l : ℕ)

def length_eq_width_plus_eight (l w : ℕ) : Prop :=
  l = w + 8

def uniform_width_between_rug_and_room (d : ℕ) : Prop :=
  d = 8

def area_uncovered_by_rug (area : ℕ) : Prop :=
  area = 704

def area_of_rug (w l : ℕ) : ℕ :=
  l * w

theorem rug_area_is_180 (w l : ℕ) (hwld : length_eq_width_plus_eight l w)
  (huw : uniform_width_between_rug_and_room 8)
  (huar : area_uncovered_by_rug 704) :
  area_of_rug w l = 180 :=
sorry

end rug_area_is_180_l1277_127731


namespace sum_of_solutions_eq_neg4_l1277_127748

theorem sum_of_solutions_eq_neg4 :
  ∃ (n : ℕ) (solutions : Fin n → ℝ × ℝ),
    (∀ i, ∃ (x y : ℝ), solutions i = (x, y) ∧ abs (x - 3) = abs (y - 9) ∧ abs (x - 9) = 2 * abs (y - 3)) ∧
    (Finset.univ.sum (fun i => (solutions i).1 + (solutions i).2) = -4) :=
sorry

end sum_of_solutions_eq_neg4_l1277_127748


namespace product_of_four_consecutive_integers_divisible_by_24_l1277_127733

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℤ) :
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_24_l1277_127733


namespace maximum_shapes_in_grid_l1277_127717

-- Define the grid size and shape properties
def grid_width : Nat := 8
def grid_height : Nat := 14
def shape_area : Nat := 3
def shape_grid_points : Nat := 8

-- Define the total grid points in the rectangular grid
def total_grid_points : Nat := (grid_width + 1) * (grid_height + 1)

-- Define the question and the condition that needs to be proved
theorem maximum_shapes_in_grid : (total_grid_points / shape_grid_points) = 16 := by
  sorry

end maximum_shapes_in_grid_l1277_127717


namespace frank_pie_consumption_l1277_127777

theorem frank_pie_consumption :
  let Erik := 0.6666666666666666
  let MoreThanFrank := 0.3333333333333333
  let Frank := Erik - MoreThanFrank
  Frank = 0.3333333333333333 := by
sorry

end frank_pie_consumption_l1277_127777


namespace triangle_dimensions_l1277_127793

-- Define the problem in Lean 4
theorem triangle_dimensions (a m : ℕ) (h₁ : a = m + 4)
  (h₂ : (a + 12) * (m + 12) = 10 * a * m) : 
  a = 12 ∧ m = 8 := 
by
  sorry

end triangle_dimensions_l1277_127793


namespace find_s_l1277_127781

noncomputable def s_value (m : ℝ) : ℝ := m + 16.25

theorem find_s (a b m s : ℝ)
  (h1 : a + b = m) (h2 : a * b = 4) :
  s = s_value m :=
by
  sorry

end find_s_l1277_127781


namespace rocky_training_miles_l1277_127795

variable (x : ℕ)

theorem rocky_training_miles (h1 : x + 2 * x + 6 * x = 36) : x = 4 :=
by
  -- proof
  sorry

end rocky_training_miles_l1277_127795


namespace sufficient_condition_l1277_127701

theorem sufficient_condition (m : ℝ) (x : ℝ) : -3 < m ∧ m < 1 → ((m - 1) * x^2 + (m - 1) * x - 1 < 0) :=
by
  sorry

end sufficient_condition_l1277_127701


namespace smallest_five_digit_equiv_11_mod_13_l1277_127762

open Nat

theorem smallest_five_digit_equiv_11_mod_13 :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 13 = 11 ∧ n = 10009 :=
by
  sorry

end smallest_five_digit_equiv_11_mod_13_l1277_127762


namespace coefficient_of_x3_in_expansion_l1277_127757

theorem coefficient_of_x3_in_expansion :
  let coeff := 56 * 972 * Real.sqrt 2
  coeff = 54432 * Real.sqrt 2 :=
by
  let coeff := 56 * 972 * Real.sqrt 2
  have h : coeff = 54432 * Real.sqrt 2 := sorry
  exact h

end coefficient_of_x3_in_expansion_l1277_127757


namespace paths_A_to_D_through_B_and_C_l1277_127715

-- Define points and paths in a grid
structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 3⟩
def C : Point := ⟨6, 4⟩
def D : Point := ⟨9, 6⟩

-- Calculate binomial coefficient
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Number of paths from one point to another in a grid
def numPaths (p1 p2 : Point) : ℕ :=
  let stepsRight := p2.x - p1.x
  let stepsDown := p2.y - p1.y
  choose (stepsRight + stepsDown) stepsRight

theorem paths_A_to_D_through_B_and_C : numPaths A B * numPaths B C * numPaths C D = 500 := by
  -- Using the conditions provided:
  -- numPaths A B = choose 5 2 = 10
  -- numPaths B C = choose 5 1 = 5
  -- numPaths C D = choose 5 2 = 10
  -- Therefore, numPaths A B * numPaths B C * numPaths C D = 10 * 5 * 10 = 500
  sorry

end paths_A_to_D_through_B_and_C_l1277_127715


namespace g_18_value_l1277_127714

-- Define the function g as taking positive integers to positive integers
variable (g : ℕ+ → ℕ+)

-- Define the conditions for the function g
axiom increasing (n : ℕ+) : g (n + 1) > g n
axiom multiplicative (m n : ℕ+) : g (m * n) = g m * g n
axiom power_property (m n : ℕ+) (h : m ≠ n ∧ m ^ (n : ℕ) = n ^ (m : ℕ)) :
  g m = n ∨ g n = m

-- Prove that g(18) is 72
theorem g_18_value : g 18 = 72 :=
sorry

end g_18_value_l1277_127714


namespace calculate_expression_l1277_127732

theorem calculate_expression :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) = 3^128 - 5^128 :=
by
  sorry

end calculate_expression_l1277_127732


namespace find_j_l1277_127716

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_j
  (a b c : ℤ)
  (h1 : f a b c 2 = 0)
  (h2 : 200 < f a b c 10 ∧ f a b c 10 < 300)
  (h3 : 400 < f a b c 9 ∧ f a b c 9 < 500)
  (j : ℤ)
  (h4 : 1000 * j < f a b c 100 ∧ f a b c 100 < 1000 * (j + 1)) :
  j = 36 := sorry

end find_j_l1277_127716


namespace bacteria_seventh_generation_l1277_127747

/-- Represents the effective multiplication factor per generation --/
def effective_mult_factor : ℕ := 4

/-- The number of bacteria in the first generation --/
def first_generation : ℕ := 1

/-- A helper function to compute the number of bacteria in the nth generation --/
def bacteria_count (n : ℕ) : ℕ :=
  first_generation * effective_mult_factor ^ n

/-- The number of bacteria in the seventh generation --/
theorem bacteria_seventh_generation : bacteria_count 7 = 4096 := by
  sorry

end bacteria_seventh_generation_l1277_127747


namespace equilateral_triangle_side_length_l1277_127794

variable (R : ℝ)

theorem equilateral_triangle_side_length (R : ℝ) :
  (∃ (s : ℝ), s = R * Real.sqrt 3) :=
sorry

end equilateral_triangle_side_length_l1277_127794


namespace larger_angle_measure_l1277_127739

-- Defining all conditions
def is_complementary (a b : ℝ) : Prop := a + b = 90

def angle_ratio (a b : ℝ) : Prop := a / b = 5 / 4

-- Main proof statement
theorem larger_angle_measure (a b : ℝ) (h1 : is_complementary a b) (h2 : angle_ratio a b) : a = 50 :=
by
  sorry

end larger_angle_measure_l1277_127739


namespace nearest_int_to_expr_l1277_127779

theorem nearest_int_to_expr : 
  |(3 + Real.sqrt 2)^6 - 3707| < 1 :=
by 
  sorry

end nearest_int_to_expr_l1277_127779


namespace largest_first_term_geometric_progression_l1277_127768

noncomputable def geometric_progression_exists (d : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (a + d + 3) / a = (a + 2 * d + 15) / (a + d + 3)

theorem largest_first_term_geometric_progression : ∀ (d : ℝ), 
  d^2 + 6 * d - 36 = 0 → 
  ∃ (a : ℝ), a = 5 ∧ geometric_progression_exists d ∧ a = 5 ∧ 
  ∀ (a' : ℝ), geometric_progression_exists d → a' ≤ a :=
by intros d h; sorry

end largest_first_term_geometric_progression_l1277_127768


namespace root_equation_satisfies_expr_l1277_127769

theorem root_equation_satisfies_expr (a : ℝ) (h : 2 * a ^ 2 - 7 * a - 1 = 0) :
  a * (2 * a - 7) + 5 = 6 :=
by
  sorry

end root_equation_satisfies_expr_l1277_127769


namespace jason_cousins_l1277_127752

theorem jason_cousins :
  let dozen := 12
  let cupcakes_bought := 4 * dozen
  let cupcakes_per_cousin := 3
  let number_of_cousins := cupcakes_bought / cupcakes_per_cousin
  number_of_cousins = 16 :=
by
  sorry

end jason_cousins_l1277_127752


namespace fiona_observe_pairs_l1277_127728

def classroom_pairs (n : ℕ) : ℕ :=
  if n > 1 then n - 1 else 0

theorem fiona_observe_pairs :
  classroom_pairs 12 = 11 :=
by
  sorry

end fiona_observe_pairs_l1277_127728


namespace range_of_fraction_l1277_127758

theorem range_of_fraction (x1 y1 : ℝ) (h1 : y1 = -2 * x1 + 8) (h2 : 2 ≤ x1 ∧ x1 ≤ 5) :
  -1/6 ≤ (y1 + 1) / (x1 + 1) ∧ (y1 + 1) / (x1 + 1) ≤ 5/3 :=
sorry

end range_of_fraction_l1277_127758


namespace intersection_of_A_and_B_l1277_127741

namespace SetsIntersectionProof

def setA : Set ℝ := { x | |x| ≤ 2 }
def setB : Set ℝ := { x | x < 1 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | -2 ≤ x ∧ x < 1 } :=
sorry

end SetsIntersectionProof

end intersection_of_A_and_B_l1277_127741


namespace conclusion_friendly_not_large_l1277_127750

variable {Snake : Type}
variable (isLarge isFriendly canClimb canSwim : Snake → Prop)
variable (marysSnakes : Finset Snake)
variable (h1 : marysSnakes.card = 16)
variable (h2 : (marysSnakes.filter isLarge).card = 6)
variable (h3 : (marysSnakes.filter isFriendly).card = 7)
variable (h4 : ∀ s, isFriendly s → canClimb s)
variable (h5 : ∀ s, isLarge s → ¬ canSwim s)
variable (h6 : ∀ s, ¬ canSwim s → ¬ canClimb s)

theorem conclusion_friendly_not_large :
  ∀ s, isFriendly s → ¬ isLarge s :=
by
  sorry

end conclusion_friendly_not_large_l1277_127750


namespace concentration_in_third_flask_l1277_127763

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l1277_127763


namespace diff_x_y_l1277_127788

theorem diff_x_y (x y : ℤ) (h1 : x + y = 14) (h2 : x = 37) : x - y = 60 :=
sorry

end diff_x_y_l1277_127788


namespace calculate_expression_l1277_127751

theorem calculate_expression :
  (0.125: ℝ) ^ 3 * (-8) ^ 3 = -1 := 
by
  sorry

end calculate_expression_l1277_127751


namespace smallest_a_l1277_127791

theorem smallest_a (a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  a = 17 :=
by
  sorry

end smallest_a_l1277_127791


namespace least_n_probability_lt_1_over_10_l1277_127737

theorem least_n_probability_lt_1_over_10 : 
  ∃ (n : ℕ), (1 / 2 : ℝ) ^ n < 1 / 10 ∧ ∀ m < n, ¬ ((1 / 2 : ℝ) ^ m < 1 / 10) :=
by
  sorry

end least_n_probability_lt_1_over_10_l1277_127737


namespace combined_percent_of_6th_graders_l1277_127738

theorem combined_percent_of_6th_graders (num_students_pineview : ℕ) 
                                        (percent_6th_pineview : ℝ) 
                                        (num_students_oakridge : ℕ)
                                        (percent_6th_oakridge : ℝ)
                                        (num_students_maplewood : ℕ)
                                        (percent_6th_maplewood : ℝ) 
                                        (total_students : ℝ) :
    num_students_pineview = 150 →
    percent_6th_pineview = 0.15 →
    num_students_oakridge = 180 →
    percent_6th_oakridge = 0.17 →
    num_students_maplewood = 170 →
    percent_6th_maplewood = 0.15 →
    total_students = 500 →
    ((percent_6th_pineview * num_students_pineview) + 
     (percent_6th_oakridge * num_students_oakridge) + 
     (percent_6th_maplewood * num_students_maplewood)) / 
    total_students * 100 = 15.72 :=
by
  sorry

end combined_percent_of_6th_graders_l1277_127738


namespace sum_of_min_x_y_l1277_127799

theorem sum_of_min_x_y : ∃ (x y : ℕ), 
  (∃ a b c : ℕ, 180 = 2^a * 3^b * 5^c) ∧
  (∃ u v w : ℕ, 180 * x = 2^u * 3^v * 5^w ∧ u % 4 = 0 ∧ v % 4 = 0 ∧ w % 4 = 0) ∧
  (∃ p q r : ℕ, 180 * y = 2^p * 3^q * 5^r ∧ p % 6 = 0 ∧ q % 6 = 0 ∧ r % 6 = 0) ∧
  (x + y = 4054500) :=
sorry

end sum_of_min_x_y_l1277_127799


namespace arithmetic_sequence_max_sum_l1277_127709

theorem arithmetic_sequence_max_sum (a d t : ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a > 0) 
  (h2 : (9 * t) = a + 5 * d) 
  (h3 : (11 * t) = a + 4 * d) 
  (h4 : ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2) :
  n = 10 :=
sorry

end arithmetic_sequence_max_sum_l1277_127709


namespace absolute_value_example_l1277_127749

theorem absolute_value_example (x : ℝ) (h : x = 4) : |x - 5| = 1 :=
by
  sorry

end absolute_value_example_l1277_127749


namespace total_amount_paid_l1277_127705

-- Definitions based on conditions
def original_price : ℝ := 100
def discount_rate : ℝ := 0.20
def additional_discount : ℝ := 5
def sales_tax_rate : ℝ := 0.08

-- Theorem statement
theorem total_amount_paid :
  let discounted_price := original_price * (1 - discount_rate)
  let final_price := discounted_price - additional_discount
  let total_price_with_tax := final_price * (1 + sales_tax_rate)
  total_price_with_tax = 81 := sorry

end total_amount_paid_l1277_127705


namespace lele_has_enough_money_and_remaining_19_yuan_l1277_127760

def price_A : ℝ := 46.5
def price_B : ℝ := 54.5
def total_money : ℝ := 120

theorem lele_has_enough_money_and_remaining_19_yuan : 
  (price_A + price_B ≤ total_money) ∧ (total_money - (price_A + price_B) = 19) :=
by
  sorry

end lele_has_enough_money_and_remaining_19_yuan_l1277_127760


namespace contrapositive_example_l1277_127700

theorem contrapositive_example (x : ℝ) : 
  (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := 
by
  sorry

end contrapositive_example_l1277_127700


namespace ratio_of_inquisitive_tourist_l1277_127706

theorem ratio_of_inquisitive_tourist (questions_per_tourist : ℕ)
                                     (num_group1 : ℕ) (num_group2 : ℕ) (num_group3 : ℕ) (num_group4 : ℕ)
                                     (total_questions : ℕ) 
                                     (inquisitive_tourist_questions : ℕ) :
  questions_per_tourist = 2 ∧ 
  num_group1 = 6 ∧ 
  num_group2 = 11 ∧ 
  num_group3 = 8 ∧ 
  num_group4 = 7 ∧ 
  total_questions = 68 ∧ 
  inquisitive_tourist_questions = (total_questions - (num_group1 * questions_per_tourist + num_group2 * questions_per_tourist +
                                                        (num_group3 - 1) * questions_per_tourist + num_group4 * questions_per_tourist)) →
  (inquisitive_tourist_questions : ℕ) / questions_per_tourist = 3 :=
by sorry

end ratio_of_inquisitive_tourist_l1277_127706


namespace focus_of_parabola_l1277_127753

def parabola_focus (a k : ℕ) : ℚ :=
  1 / (4 * a) + k

theorem focus_of_parabola :
  parabola_focus 9 6 = 217 / 36 :=
by
  sorry

end focus_of_parabola_l1277_127753


namespace twenty_four_game_l1277_127730

theorem twenty_four_game : 8 / (3 - 8 / 3) = 24 := 
by
  sorry

end twenty_four_game_l1277_127730


namespace haley_tickets_l1277_127746

-- Conditions
def cost_per_ticket : ℕ := 4
def extra_tickets : ℕ := 5
def total_spent : ℕ := 32
def cost_extra_tickets : ℕ := extra_tickets * cost_per_ticket

-- Main proof problem
theorem haley_tickets (T : ℕ) (h : 4 * T + cost_extra_tickets = total_spent) :
  T = 3 := sorry

end haley_tickets_l1277_127746


namespace problem_solution_l1277_127735

variable {A B : Set ℝ}

def definition_A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def definition_B : Set ℝ := {x | x ≤ 3}

theorem problem_solution : definition_A ∩ definition_B = definition_A :=
by
  sorry

end problem_solution_l1277_127735


namespace susan_ate_6_candies_l1277_127771

-- Definitions for the conditions
def candies_tuesday : ℕ := 3
def candies_thursday : ℕ := 5
def candies_friday : ℕ := 2
def candies_left : ℕ := 4

-- The total number of candies bought during the week
def total_candies_bought : ℕ := candies_tuesday + candies_thursday + candies_friday

-- The number of candies Susan ate during the week
def candies_eaten : ℕ := total_candies_bought - candies_left

-- Theorem statement
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  unfold candies_eaten total_candies_bought candies_tuesday candies_thursday candies_friday candies_left
  sorry

end susan_ate_6_candies_l1277_127771


namespace simplify_expression_l1277_127727

theorem simplify_expression (a : ℤ) :
  ((36 * a^9)^4 * (63 * a^9)^4) = a^4 :=
sorry

end simplify_expression_l1277_127727


namespace a_range_l1277_127736

noncomputable def f (a x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem a_range (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ a ∈ Set.Ico (1/7 : ℝ) (1/3 : ℝ) :=
by
  sorry

end a_range_l1277_127736


namespace triangle_area_example_l1277_127704

def point := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_example :
  let A : point := (3, -2)
  let B : point := (12, 5)
  let C : point := (3, 8)
  triangle_area A B C = 45 :=
by
  sorry

end triangle_area_example_l1277_127704


namespace arithmetic_sequence_sum_l1277_127765

theorem arithmetic_sequence_sum
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_S_def : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2)
  (h_S2 : S 2 = 4)
  (h_S4 : S 4 = 16) :
  a 5 + a 6 = 20 :=
sorry

end arithmetic_sequence_sum_l1277_127765


namespace car_average_speed_l1277_127723

noncomputable def average_speed (speeds : List ℝ) (distances : List ℝ) (times : List ℝ) : ℝ :=
  (distances.sum + times.sum) / times.sum

theorem car_average_speed :
  let distances := [30, 35, 35, 52 / 3, 15]
  let times := [30 / 45, 35 / 55, 30 / 60, 20 / 60, 15 / 65]
  average_speed [45, 55, 70, 52, 65] distances times = 64.82 := by
  sorry

end car_average_speed_l1277_127723


namespace relationship_between_m_and_n_l1277_127743

variable (x : ℝ)

def m := x^2 + 2*x + 3
def n := 2

theorem relationship_between_m_and_n :
  m x ≥ n := by
  sorry

end relationship_between_m_and_n_l1277_127743


namespace area_of_enclosed_region_is_zero_l1277_127774

theorem area_of_enclosed_region_is_zero :
  (∃ (x y : ℝ), x^2 + y^2 = |x| - |y|) → (0 = 0) :=
sorry

end area_of_enclosed_region_is_zero_l1277_127774


namespace find_b_age_l1277_127721

variable (a b c : ℕ)
-- Condition 1: a is two years older than b
variable (h1 : a = b + 2)
-- Condition 2: b is twice as old as c
variable (h2 : b = 2 * c)
-- Condition 3: The total of the ages of a, b, and c is 17
variable (h3 : a + b + c = 17)

theorem find_b_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 17) : b = 6 :=
by
  sorry

end find_b_age_l1277_127721


namespace min_value_of_expression_l1277_127783

open Real

noncomputable def minValue (x y z : ℝ) : ℝ :=
  x + 3 * y + 5 * z

theorem min_value_of_expression : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 8 → minValue x y z = 14.796 :=
by
  intros x y z h
  sorry

end min_value_of_expression_l1277_127783


namespace yellow_papers_count_l1277_127729

theorem yellow_papers_count (n : ℕ) (total_papers : ℕ) (periphery_papers : ℕ) (inner_papers : ℕ) 
  (h1 : n = 10) 
  (h2 : total_papers = n * n) 
  (h3 : periphery_papers = 4 * n - 4)
  (h4 : inner_papers = total_papers - periphery_papers) :
  inner_papers = 64 :=
by
  sorry

end yellow_papers_count_l1277_127729


namespace min_value_xy_k_l1277_127756

theorem min_value_xy_k (x y k : ℝ) : ∃ x y : ℝ, (xy - k)^2 + (x + y - 1)^2 = 1 := by
  sorry

end min_value_xy_k_l1277_127756


namespace quadratic_has_negative_root_l1277_127708

def quadratic_function (m x : ℝ) : ℝ := (m - 2) * x^2 - 4 * m * x + 2 * m - 6

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks for a range of m such that the quadratic function intersects the negative x-axis
theorem quadratic_has_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ quadratic_function m x = 0) ↔ (1 ≤ m ∧ m < 2 ∨ 2 < m ∧ m < 3) :=
sorry

end quadratic_has_negative_root_l1277_127708


namespace power_of_two_sequence_invariant_l1277_127789

theorem power_of_two_sequence_invariant
  (n : ℕ)
  (a b : ℕ → ℕ)
  (h₀ : a 0 = 1)
  (h₁ : b 0 = n)
  (hi : ∀ i : ℕ, a i < b i → a (i + 1) = 2 * a i + 1 ∧ b (i + 1) = b i - a i - 1)
  (hj : ∀ i : ℕ, a i > b i → a (i + 1) = a i - b i - 1 ∧ b (i + 1) = 2 * b i + 1)
  (hk : ∀ i : ℕ, a i = b i → a (i + 1) = a i ∧ b (i + 1) = b i)
  (k : ℕ)
  (h : a k = b k) :
  ∃ m : ℕ, n + 3 = 2 ^ m :=
by
  sorry

end power_of_two_sequence_invariant_l1277_127789


namespace tank_fill_fraction_l1277_127720

theorem tank_fill_fraction (a b c : ℝ) (h1 : a=9) (h2 : b=54) (h3 : c=3/4) : (c * b + a) / b = 23 / 25 := 
by 
  sorry

end tank_fill_fraction_l1277_127720


namespace coin_combinations_l1277_127724

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l1277_127724


namespace intersection_A_B_l1277_127711

def A (x : ℝ) : Prop := ∃ y, y = Real.log (-x^2 - 2*x + 8) ∧ -x^2 - 2*x + 8 > 0
def B (x : ℝ) : Prop := Real.log x / Real.log 2 < 1 ∧ x > 0

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l1277_127711


namespace remainder_of_x_plus_2_pow_2022_l1277_127770

theorem remainder_of_x_plus_2_pow_2022 (x : ℂ) :
  ∃ r : ℂ, ∃ q : ℂ, (x + 2)^2022 = q * (x^2 - x + 1) + r ∧ (r = x) :=
by
  sorry

end remainder_of_x_plus_2_pow_2022_l1277_127770
