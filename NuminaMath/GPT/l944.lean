import Mathlib

namespace ratio_pow_eq_l944_94442

theorem ratio_pow_eq {x y : ℝ} (h : x / y = 7 / 5) : (x^3 / y^2) = 343 / 25 :=
by sorry

end ratio_pow_eq_l944_94442


namespace max_value_of_a_l944_94467

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

theorem max_value_of_a
  (odd_f : odd_function f)
  (decr_f : decreasing_function f)
  (h : ∀ x : ℝ, f (Real.cos (2 * x) + Real.sin x) + f (Real.sin x - a) ≤ 0) :
  a ≤ -3 :=
sorry

end max_value_of_a_l944_94467


namespace find_prime_number_between_50_and_60_l944_94445

theorem find_prime_number_between_50_and_60 (n : ℕ) :
  (50 < n ∧ n < 60) ∧ Prime n ∧ n % 7 = 3 ↔ n = 59 :=
by
  sorry

end find_prime_number_between_50_and_60_l944_94445


namespace product_of_two_numbers_l944_94486

theorem product_of_two_numbers (x y : ℝ) 
  (h₁ : x + y = 50) 
  (h₂ : x - y = 6) : 
  x * y = 616 := 
by
  sorry

end product_of_two_numbers_l944_94486


namespace find_HCF_of_two_numbers_l944_94479

theorem find_HCF_of_two_numbers (a b H : ℕ) 
  (H_HCF : Nat.gcd a b = H) 
  (H_LCM_Factors : Nat.lcm a b = H * 13 * 14) 
  (H_largest_number : 322 = max a b) : 
  H = 14 :=
sorry

end find_HCF_of_two_numbers_l944_94479


namespace no_nonnegative_integral_solutions_l944_94448

theorem no_nonnegative_integral_solutions :
  ¬ ∃ (x y : ℕ), (x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0) ∧ (x + y = 10) :=
by
  sorry

end no_nonnegative_integral_solutions_l944_94448


namespace value_of_x_y_mn_l944_94433

variables (x y m n : ℝ)

-- Conditions for arithmetic sequence 2, x, y, 3
def arithmetic_sequence_condition_1 : Prop := 2 * x = 2 + y
def arithmetic_sequence_condition_2 : Prop := 2 * y = 3 + x

-- Conditions for geometric sequence 2, m, n, 3
def geometric_sequence_condition_1 : Prop := m^2 = 2 * n
def geometric_sequence_condition_2 : Prop := n^2 = 3 * m

theorem value_of_x_y_mn (h1 : arithmetic_sequence_condition_1 x y) 
                        (h2 : arithmetic_sequence_condition_2 x y) 
                        (h3 : geometric_sequence_condition_1 m n)
                        (h4 : geometric_sequence_condition_2 m n) : 
  x + y + m * n = 11 :=
sorry

end value_of_x_y_mn_l944_94433


namespace minimum_value_of_quadratic_l944_94444

-- Definition of the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 6 * x + 13

-- Statement of the proof problem
theorem minimum_value_of_quadratic : ∃ (y : ℝ), ∀ x : ℝ, quadratic x >= y ∧ y = 4 := by
  sorry

end minimum_value_of_quadratic_l944_94444


namespace min_value_trig_expression_l944_94492

theorem min_value_trig_expression : (∃ x : ℝ, 3 * Real.cos x - 4 * Real.sin x = -5) :=
by
  sorry

end min_value_trig_expression_l944_94492


namespace num_pairs_of_nat_numbers_satisfying_eq_l944_94469

theorem num_pairs_of_nat_numbers_satisfying_eq (n : ℕ) :
  n = 5 ↔ ∃ (a b : ℕ), a ≥ b ∧ (1/a : ℚ) + (1/b : ℚ) = (1/6 : ℚ) := sorry

end num_pairs_of_nat_numbers_satisfying_eq_l944_94469


namespace no_solution_for_equation_l944_94472

theorem no_solution_for_equation :
  ¬(∃ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ (x+2)/(x-2) - x/(x+2) = 16/(x^2-4)) :=
by
    sorry

end no_solution_for_equation_l944_94472


namespace total_students_l944_94466

theorem total_students (students_in_front : ℕ) (position_from_back : ℕ) : 
  students_in_front = 6 ∧ position_from_back = 5 → 
  students_in_front + 1 + (position_from_back - 1) = 11 :=
by
  sorry

end total_students_l944_94466


namespace markers_needed_total_l944_94405

noncomputable def markers_needed_first_group : ℕ := 10 * 2
noncomputable def markers_needed_second_group : ℕ := 15 * 4
noncomputable def students_last_group : ℕ := 30 - (10 + 15)
noncomputable def markers_needed_last_group : ℕ := students_last_group * 6

theorem markers_needed_total : markers_needed_first_group + markers_needed_second_group + markers_needed_last_group = 110 :=
by
  sorry

end markers_needed_total_l944_94405


namespace polynomial_problem_l944_94415

theorem polynomial_problem 
  (d_1 d_2 d_3 d_4 e_1 e_2 e_3 e_4 : ℝ)
  (h : ∀ (x : ℝ),
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + d_1 * x + e_1) * (x^2 + d_2 * x + e_2) * (x^2 + d_3 * x + e_3) * (x^2 + d_4 * x + e_4)) :
  d_1 * e_1 + d_2 * e_2 + d_3 * e_3 + d_4 * e_4 = -1 := 
by
  sorry

end polynomial_problem_l944_94415


namespace base_number_eq_2_l944_94450

theorem base_number_eq_2 (x : ℝ) (n : ℕ) (h₁ : x^(2 * n) + x^(2 * n) + x^(2 * n) + x^(2 * n) = 4^28) (h₂ : n = 27) : x = 2 := by
  sorry

end base_number_eq_2_l944_94450


namespace tom_catches_48_trout_l944_94454

variable (melanie_tom_catch_ratio : ℕ := 3)
variable (melanie_catch : ℕ := 16)

theorem tom_catches_48_trout (h1 : melanie_catch = 16) (h2 : melanie_tom_catch_ratio = 3) : (melanie_tom_catch_ratio * melanie_catch) = 48 :=
by
  sorry

end tom_catches_48_trout_l944_94454


namespace truck_gasoline_rate_l944_94440

theorem truck_gasoline_rate (gas_initial gas_final : ℕ) (dist_supermarket dist_farm_turn dist_farm_final : ℕ) 
    (total_miles gas_used : ℕ) : 
  gas_initial = 12 →
  gas_final = 2 →
  dist_supermarket = 10 →
  dist_farm_turn = 4 →
  dist_farm_final = 6 →
  total_miles = dist_supermarket + dist_farm_turn + dist_farm_final →
  gas_used = gas_initial - gas_final →
  total_miles / gas_used = 2 :=
by sorry

end truck_gasoline_rate_l944_94440


namespace cost_of_nuts_l944_94426

/--
Adam bought 3 kilograms of nuts and 2.5 kilograms of dried fruits at a store. 
One kilogram of nuts costs a certain amount N and one kilogram of dried fruit costs $8. 
His purchases cost $56. Prove that one kilogram of nuts costs $12.
-/
theorem cost_of_nuts (N : ℝ) 
  (h1 : 3 * N + 2.5 * 8 = 56) 
  : N = 12 := by
  sorry

end cost_of_nuts_l944_94426


namespace row_3_seat_6_representation_l944_94483

-- Given Conditions
def seat_representation (r : ℕ) (s : ℕ) : (ℕ × ℕ) :=
  (r, s)

-- Proof Statement
theorem row_3_seat_6_representation :
  seat_representation 3 6 = (3, 6) :=
by
  sorry

end row_3_seat_6_representation_l944_94483


namespace product_of_ratios_l944_94446

theorem product_of_ratios:
  ∀ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1^3 - 3 * x1 * y1^2 = 2023) ∧ (y1^3 - 3 * x1^2 * y1 = 2022) →
    (x2^3 - 3 * x2 * y2^2 = 2023) ∧ (y2^3 - 3 * x2^2 * y2 = 2022) →
    (x3^3 - 3 * x3 * y3^2 = 2023) ∧ (y3^3 - 3 * x3^2 * y3 = 2022) →
    (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1 / 2023 :=
by
  intros x1 y1 x2 y2 x3 y3
  sorry

end product_of_ratios_l944_94446


namespace sin_2x_from_tan_pi_minus_x_l944_94419

theorem sin_2x_from_tan_pi_minus_x (x : ℝ) (h : Real.tan (Real.pi - x) = 3) : Real.sin (2 * x) = -3 / 5 := by
  sorry

end sin_2x_from_tan_pi_minus_x_l944_94419


namespace quad_in_vertex_form_addition_l944_94422

theorem quad_in_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∃ a h k, (4 * x^2 - 8 * x + 3) = a * (x - h) ^ 2 + k) →
  a + h + k = 4 :=
by
  sorry

end quad_in_vertex_form_addition_l944_94422


namespace find_T_l944_94437

theorem find_T (T : ℝ) 
  (h : (1/3) * (1/8) * T = (1/4) * (1/6) * 150) : 
  T = 150 :=
sorry

end find_T_l944_94437


namespace gcd_yz_min_value_l944_94418

theorem gcd_yz_min_value (x y z : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) 
  (hxy_gcd : Nat.gcd x y = 224) (hxz_gcd : Nat.gcd x z = 546) : 
  Nat.gcd y z = 14 := 
sorry

end gcd_yz_min_value_l944_94418


namespace circle_tangent_radii_l944_94428

theorem circle_tangent_radii (a b c : ℝ) (A : ℝ) (p : ℝ)
  (r r_a r_b r_c : ℝ)
  (h1 : p = (a + b + c) / 2)
  (h2 : r = A / p)
  (h3 : r_a = A / (p - a))
  (h4 : r_b = A / (p - b))
  (h5 : r_c = A / (p - c))
  : 1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  sorry

end circle_tangent_radii_l944_94428


namespace avg_age_all_l944_94471

-- Define the conditions
def avg_age_seventh_graders (n₁ : Nat) (a₁ : Nat) : Prop :=
  n₁ = 40 ∧ a₁ = 13

def avg_age_parents (n₂ : Nat) (a₂ : Nat) : Prop :=
  n₂ = 50 ∧ a₂ = 40

-- Define the problem to prove
def avg_age_combined (n₁ n₂ a₁ a₂ : Nat) : Prop :=
  (n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 28

-- The main theorem
theorem avg_age_all (n₁ n₂ a₁ a₂ : Nat):
  avg_age_seventh_graders n₁ a₁ → avg_age_parents n₂ a₂ → avg_age_combined n₁ n₂ a₁ a₂ :=
by 
  intros h1 h2
  sorry

end avg_age_all_l944_94471


namespace series_sum_correct_l944_94496

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l944_94496


namespace cupboard_cost_price_l944_94451

noncomputable def cost_price_of_cupboard (C : ℝ) : Prop :=
  let SP := 0.88 * C
  let NSP := 1.12 * C
  NSP - SP = 1650

theorem cupboard_cost_price : ∃ (C : ℝ), cost_price_of_cupboard C ∧ C = 6875 := by
  sorry

end cupboard_cost_price_l944_94451


namespace operation_results_in_m4_l944_94487

variable (m : ℤ)

theorem operation_results_in_m4 :
  (-m^2)^2 = m^4 :=
sorry

end operation_results_in_m4_l944_94487


namespace max_min_rounded_value_l944_94476

theorem max_min_rounded_value (n : ℝ) (h : 3.75 ≤ n ∧ n < 3.85) : 
  (∀ n, 3.75 ≤ n ∧ n < 3.85 → n ≤ 3.84 ∧ n ≥ 3.75) :=
sorry

end max_min_rounded_value_l944_94476


namespace expression_c_is_positive_l944_94432

def A : ℝ := 2.1
def B : ℝ := -0.5
def C : ℝ := -3.0
def D : ℝ := 4.2
def E : ℝ := 0.8

theorem expression_c_is_positive : |C| + |B| > 0 :=
by {
  sorry
}

end expression_c_is_positive_l944_94432


namespace number_of_days_in_first_part_l944_94470

variable {x : ℕ}

-- Conditions
def avg_exp_first_part (x : ℕ) : ℕ := 350 * x
def avg_exp_next_four_days : ℕ := 420 * 4
def total_days (x : ℕ) : ℕ := x + 4
def avg_exp_whole_week (x : ℕ) : ℕ := 390 * total_days x

-- Equation based on the conditions
theorem number_of_days_in_first_part :
  avg_exp_first_part x + avg_exp_next_four_days = avg_exp_whole_week x →
  x = 3 :=
by
  sorry

end number_of_days_in_first_part_l944_94470


namespace possible_values_for_xyz_l944_94475

theorem possible_values_for_xyz:
  (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
   x + 2 * y = z →
   x^2 - 4 * y^2 + z^2 = 310 →
   (∃ (k : ℕ), k = x * y * z ∧ (k = 11935 ∨ k = 2015))) :=
by
  intros x y z hx hy hz h1 h2
  sorry

end possible_values_for_xyz_l944_94475


namespace min_n_coloring_property_l944_94427

theorem min_n_coloring_property : ∃ n : ℕ, (∀ (coloring : ℕ → Bool), 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧ coloring a = coloring b ∧ coloring b = coloring c → 2 * a + b = c)) ∧ n = 15 := 
sorry

end min_n_coloring_property_l944_94427


namespace cos_theta_of_triangle_median_l944_94459

theorem cos_theta_of_triangle_median
  (A : ℝ) (a : ℝ) (m : ℝ) (theta : ℝ)
  (area_eq : A = 24)
  (side_eq : a = 12)
  (median_eq : m = 5)
  (area_formula : A = (1/2) * a * m * Real.sin theta) :
  Real.cos theta = 3 / 5 := 
by 
  sorry

end cos_theta_of_triangle_median_l944_94459


namespace isosceles_triangle_l944_94465

theorem isosceles_triangle
  (α β γ x y z w : ℝ)
  (h_triangle : α + β + γ = 180)
  (h_quad : x + y + z + w = 360)
  (h_conditions : (x = α + β) ∧ (y = β + γ) ∧ (z = γ + α) ∨ (w = α + β) ∧ (x = β + γ) ∧ (y = γ + α) ∨ (z = α + β) ∧ (w = β + γ) ∧ (x = γ + α) ∨ (y = α + β) ∧ (z = β + γ) ∧ (w = γ + α))
  : α = β ∨ β = γ ∨ γ = α := 
sorry

end isosceles_triangle_l944_94465


namespace percent_problem_l944_94434

theorem percent_problem (x : ℝ) (h : 0.20 * x = 1000) : 1.20 * x = 6000 := by
  sorry

end percent_problem_l944_94434


namespace parallel_lines_a_value_l944_94421

theorem parallel_lines_a_value (a : ℝ) 
  (h1 : ∀ x y : ℝ, x + a * y - 1 = 0 → x = a * (-4 * y - 2)) 
  : a = 2 :=
sorry

end parallel_lines_a_value_l944_94421


namespace b_share_is_approx_1885_71_l944_94401

noncomputable def investment_problem (x : ℝ) : ℝ := 
  let c_investment := x
  let b_investment := (2 / 3) * c_investment
  let a_investment := 3 * b_investment
  let total_investment := a_investment + b_investment + c_investment
  let b_share := (b_investment / total_investment) * 6600
  b_share

theorem b_share_is_approx_1885_71 (x : ℝ) : abs (investment_problem x - 1885.71) < 0.01 := sorry

end b_share_is_approx_1885_71_l944_94401


namespace basket_weight_l944_94452

variables 
  (B : ℕ) -- Weight of the basket
  (L : ℕ) -- Lifting capacity of one balloon

-- Condition: One balloon can lift a basket with contents weighing not more than 80 kg
axiom one_balloon_lifts (h1 : B + L ≤ 80) : Prop

-- Condition: Two balloons can lift a basket with contents weighing not more than 180 kg
axiom two_balloons_lift (h2 : B + 2 * L ≤ 180) : Prop

-- The proof problem: Determine B under the given conditions
theorem basket_weight (B : ℕ) (L : ℕ) (h1 : B + L ≤ 80) (h2 : B + 2 * L ≤ 180) : B = 20 :=
  sorry

end basket_weight_l944_94452


namespace max_marks_mike_could_have_got_l944_94412

theorem max_marks_mike_could_have_got (p : ℝ) (m_s : ℝ) (d : ℝ) (M : ℝ) :
  p = 0.30 → m_s = 212 → d = 13 → 0.30 * M = (212 + 13) → M = 750 :=
by
  intros hp hms hd heq
  sorry

end max_marks_mike_could_have_got_l944_94412


namespace remainder_7547_div_11_l944_94436

theorem remainder_7547_div_11 : 7547 % 11 = 10 :=
by
  sorry

end remainder_7547_div_11_l944_94436


namespace vasya_triangle_rotation_l944_94480

theorem vasya_triangle_rotation :
  (∀ (θ1 θ2 θ3 : ℝ), (12 * θ1 = 360) ∧ (6 * θ2 = 360) ∧ (θ1 + θ2 + θ3 = 180) → ∃ n : ℕ, (n * θ3 = 360) ∧ n ≥ 4) :=
by
  -- The formal proof is omitted, inserting "sorry" to indicate incomplete proof
  sorry

end vasya_triangle_rotation_l944_94480


namespace largest_even_number_in_sequence_of_six_l944_94455

-- Definitions and conditions
def smallest_even_number (x : ℤ) : Prop :=
  x + (x + 2) + (x+4) + (x+6) + (x + 8) + (x + 10) = 540

def sum_of_squares_of_sequence (x : ℤ) : Prop :=
  x^2 + (x + 2)^2 + (x + 4)^2 + (x + 6)^2 + (x + 8)^2 + (x + 10)^2 = 97920

-- Statement to prove
theorem largest_even_number_in_sequence_of_six (x : ℤ) (h1 : smallest_even_number x) (h2 : sum_of_squares_of_sequence x) : x + 10 = 95 :=
  sorry

end largest_even_number_in_sequence_of_six_l944_94455


namespace paper_folding_possible_layers_l944_94429

theorem paper_folding_possible_layers (n : ℕ) : 16 = 2 ^ n :=
by
  sorry

end paper_folding_possible_layers_l944_94429


namespace geometric_sequence_eighth_term_is_correct_l944_94438

noncomputable def geometric_sequence_eighth_term : ℚ :=
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8

theorem geometric_sequence_eighth_term_is_correct :
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8 = 35651584 / 4782969 := by
    sorry

end geometric_sequence_eighth_term_is_correct_l944_94438


namespace function_point_proof_l944_94416

-- Given conditions
def condition (f : ℝ → ℝ) : Prop :=
  f 1 = 3

-- Prove the statement
theorem function_point_proof (f : ℝ → ℝ) (h : condition f) : f (-1) + 1 = 4 :=
by
  -- Adding the conditions here
  sorry -- proof is not required

end function_point_proof_l944_94416


namespace quadratic_has_one_solution_implies_m_eq_3_l944_94497

theorem quadratic_has_one_solution_implies_m_eq_3 {m : ℝ} (h : ∃ x : ℝ, 3 * x^2 - 6 * x + m = 0 ∧ ∃! u, 3 * u^2 - 6 * u + m = 0) : m = 3 :=
by sorry

end quadratic_has_one_solution_implies_m_eq_3_l944_94497


namespace second_discount_percentage_l944_94485

/-- 
  Given:
  - The listed price of Rs. 560.
  - The final sale price after successive discounts of 20% and another discount is Rs. 313.6.
  Prove:
  - The second discount percentage is 30%.
-/
theorem second_discount_percentage (list_price final_price : ℝ) (first_discount_percentage : ℝ) : 
  list_price = 560 → 
  final_price = 313.6 → 
  first_discount_percentage = 20 → 
  ∃ (second_discount_percentage : ℝ), second_discount_percentage = 30 :=
by
  sorry

end second_discount_percentage_l944_94485


namespace mod_17_residue_l944_94453

theorem mod_17_residue : (255 + 7 * 51 + 9 * 187 + 5 * 34) % 17 = 0 := 
  by sorry

end mod_17_residue_l944_94453


namespace fruit_juice_conversion_needed_l944_94403

theorem fruit_juice_conversion_needed
  (A_milk_parts B_milk_parts A_fruit_juice_parts B_fruit_juice_parts : ℕ)
  (y : ℕ)
  (x : ℕ)
  (convert_liters : ℕ)
  (A_juice_ratio_milk A_juice_ratio_fruit : ℚ)
  (B_juice_ratio_milk B_juice_ratio_fruit : ℚ) :
  (A_milk_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_milk →
  (A_fruit_juice_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_fruit →
  (B_milk_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_milk →
  (B_fruit_juice_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_fruit →
  (A_juice_ratio_milk * x = A_juice_ratio_fruit * x + y) →
  y = 14 →
  x = 98 :=
by sorry

end fruit_juice_conversion_needed_l944_94403


namespace max_correct_questions_prime_score_l944_94406

-- Definitions and conditions
def total_questions := 20
def points_correct := 5
def points_no_answer := 0
def points_wrong := -2

-- Main statement to prove
theorem max_correct_questions_prime_score :
  ∃ (correct : ℕ) (no_answer wrong : ℕ), 
    correct + no_answer + wrong = total_questions ∧ 
    correct * points_correct + no_answer * points_no_answer + wrong * points_wrong = 83 ∧
    correct = 17 :=
sorry

end max_correct_questions_prime_score_l944_94406


namespace quadratic_inequality_solution_set_l944_94489

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | ax^2 - (2 + a) * x + 2 > 0} = {x | 2 / a < x ∧ x < 1} :=
sorry

end quadratic_inequality_solution_set_l944_94489


namespace ratio_Bill_Cary_l944_94494

noncomputable def Cary_height : ℝ := 72
noncomputable def Jan_height : ℝ := 42
noncomputable def Bill_height : ℝ := Jan_height - 6

theorem ratio_Bill_Cary : Bill_height / Cary_height = 1 / 2 :=
by
  sorry

end ratio_Bill_Cary_l944_94494


namespace map_a_distance_map_b_distance_miles_map_b_distance_km_l944_94484

theorem map_a_distance (distance_cm : ℝ) (scale_cm : ℝ) (scale_km : ℝ) (actual_distance : ℝ) : 
  distance_cm = 80.5 → scale_cm = 0.6 → scale_km = 6.6 → actual_distance = (distance_cm * scale_km) / scale_cm → actual_distance = 885.5 :=
by
  intros h1 h2 h3 h4
  sorry

theorem map_b_distance_miles (distance_cm : ℝ) (scale_cm : ℝ) (scale_miles : ℝ) (actual_distance_miles : ℝ) : 
  distance_cm = 56.3 → scale_cm = 1.1 → scale_miles = 7.7 → actual_distance_miles = (distance_cm * scale_miles) / scale_cm → actual_distance_miles = 394.1 :=
by
  intros h1 h2 h3 h4
  sorry

theorem map_b_distance_km (distance_miles : ℝ) (conversion_factor : ℝ) (actual_distance_km : ℝ) :
  conversion_factor = 1.60934 → distance_miles = 394.1 → actual_distance_km = distance_miles * conversion_factor → actual_distance_km = 634.3 :=
by
  intros h1 h2 h3
  sorry

end map_a_distance_map_b_distance_miles_map_b_distance_km_l944_94484


namespace calculate_product_N1_N2_l944_94462

theorem calculate_product_N1_N2 : 
  (∃ (N1 N2 : ℝ), 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → 
      (60 * x - 46) / (x^2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) ∧
      N1 * N2 = -1036) :=
  sorry

end calculate_product_N1_N2_l944_94462


namespace boat_speed_in_still_water_l944_94443

theorem boat_speed_in_still_water (V_s : ℝ) (D : ℝ) (t_down : ℝ) (t_up : ℝ) (V_b : ℝ) :
  V_s = 3 → t_down = 1 → t_up = 3 / 2 →
  (V_b + V_s) * t_down = D → (V_b - V_s) * t_up = D → V_b = 15 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end boat_speed_in_still_water_l944_94443


namespace range_of_m_l944_94482

theorem range_of_m (m : ℝ) :
  (∃ (m : ℝ), (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + 1 = 0 ∧ x2^2 + m * x2 + 1 = 0) ∧ 
  (∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≤ 0)) ↔ (m ≤ 1 ∨ m ≥ 3 ∨ m < -2) :=
by
  sorry

end range_of_m_l944_94482


namespace sum_of_squares_eq_1850_l944_94447

-- Assuming definitions for the rates
variables (b j s h : ℕ)

-- Condition from Ed's activity
axiom ed_condition : 3 * b + 4 * j + 2 * s + 3 * h = 120

-- Condition from Sue's activity
axiom sue_condition : 2 * b + 3 * j + 4 * s + 3 * h = 150

-- Sum of squares of biking, jogging, swimming, and hiking rates
def sum_of_squares (b j s h : ℕ) : ℕ := b^2 + j^2 + s^2 + h^2

-- Assertion we want to prove
theorem sum_of_squares_eq_1850 :
  ∃ b j s h : ℕ, 3 * b + 4 * j + 2 * s + 3 * h = 120 ∧ 2 * b + 3 * j + 4 * s + 3 * h = 150 ∧ sum_of_squares b j s h = 1850 :=
by
  sorry

end sum_of_squares_eq_1850_l944_94447


namespace coefficient_of_x3_in_expansion_l944_94425

noncomputable def binomial_expansion_coefficient (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem coefficient_of_x3_in_expansion : 
  (∀ k : ℕ, binomial_expansion_coefficient 6 k ≤ binomial_expansion_coefficient 6 3) →
  binomial_expansion_coefficient 6 3 = 20 :=
by
  intro h
  -- skipping the proof
  sorry

end coefficient_of_x3_in_expansion_l944_94425


namespace Donovan_Mitchell_goal_average_l944_94441

theorem Donovan_Mitchell_goal_average 
  (current_avg_pg : ℕ)     -- Donovan's current average points per game.
  (played_games : ℕ)       -- Number of games played so far.
  (required_avg_pg : ℕ)    -- Required average points per game in remaining games.
  (total_games : ℕ)        -- Total number of games in the season.
  (goal_avg_pg : ℕ)        -- Goal average points per game for the entire season.
  (H1 : current_avg_pg = 26)
  (H2 : played_games = 15)
  (H3 : required_avg_pg = 42)
  (H4 : total_games = 20) :
  goal_avg_pg = 30 :=
by
  sorry

end Donovan_Mitchell_goal_average_l944_94441


namespace walter_time_spent_at_seals_l944_94414

theorem walter_time_spent_at_seals (S : ℕ) 
(h1 : 8 * S + S + 13 = 130) : S = 13 :=
sorry

end walter_time_spent_at_seals_l944_94414


namespace black_and_white_films_l944_94410

theorem black_and_white_films (y x B : ℕ) 
  (h1 : ∀ B, B = 40 * x)
  (h2 : (4 * y : ℚ) / (((y / x : ℚ) * B / 100) + 4 * y) = 10 / 11) :
  B = 40 * x :=
by sorry

end black_and_white_films_l944_94410


namespace card_area_after_reduction_width_l944_94468

def initial_length : ℕ := 5
def initial_width : ℕ := 8
def new_width := initial_width - 2
def expected_new_area : ℕ := 24

theorem card_area_after_reduction_width :
  initial_length * new_width = expected_new_area := 
by
  -- initial_length = 5, new_width = 8 - 2 = 6
  -- 5 * 6 = 30, which was corrected to 24 given the misinterpretation mentioned.
  sorry

end card_area_after_reduction_width_l944_94468


namespace inclination_angle_of_focal_chord_l944_94478

theorem inclination_angle_of_focal_chord
  (p : ℝ)
  (h_parabola : ∀ x y : ℝ, y^2 = 2 * p * x → True)
  (h_focal_chord_length : ∀ A B : ℝ, |A - B| = 8 * p → True) :
  ∃ θ : ℝ, (θ = π / 6 ∨ θ = 5 * π / 6) :=
by
  sorry

end inclination_angle_of_focal_chord_l944_94478


namespace symmetric_points_x_axis_l944_94477

theorem symmetric_points_x_axis (a b : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (a + 2, -2))
  (hQ : Q = (4, b))
  (hx : (a + 2) = 4)
  (hy : b = 2) :
  (a^b) = 4 := by
sorry

end symmetric_points_x_axis_l944_94477


namespace younger_son_age_in_30_years_l944_94430

theorem younger_son_age_in_30_years
  (age_difference : ℕ)
  (elder_son_current_age : ℕ)
  (younger_son_age_in_30_years : ℕ) :
  age_difference = 10 →
  elder_son_current_age = 40 →
  younger_son_age_in_30_years = elder_son_current_age - age_difference + 30 →
  younger_son_age_in_30_years = 60 :=
by
  intros h_diff h_elder h_calc
  sorry

end younger_son_age_in_30_years_l944_94430


namespace range_of_b_l944_94404

theorem range_of_b (b : ℝ) :
  (∀ x y : ℝ, (x ≠ y) → (y = 1/3 * x^3 + b * x^2 + (b + 2) * x + 3) → (y ≥ 1/3 * x^3 + b * x^2 + (b + 2) * x + 3))
  ↔ (-1 ≤ b ∧ b ≤ 2) :=
sorry

end range_of_b_l944_94404


namespace jacques_initial_gumballs_l944_94495

def joanna_initial_gumballs : ℕ := 40
def each_shared_gumballs_after_purchase : ℕ := 250

theorem jacques_initial_gumballs (J : ℕ) (h : 2 * (joanna_initial_gumballs + J + 4 * (joanna_initial_gumballs + J)) = 2 * each_shared_gumballs_after_purchase) : J = 60 :=
by
  sorry

end jacques_initial_gumballs_l944_94495


namespace james_bought_100_cattle_l944_94488

noncomputable def number_of_cattle (purchase_price : ℝ) (feeding_ratio : ℝ) (weight_per_cattle : ℝ) (price_per_pound : ℝ) (profit : ℝ) : ℝ :=
  let feeding_cost := purchase_price * feeding_ratio
  let total_feeding_cost := purchase_price + feeding_cost
  let total_cost := purchase_price + total_feeding_cost
  let selling_price_per_cattle := weight_per_cattle * price_per_pound
  let total_revenue := total_cost + profit
  total_revenue / selling_price_per_cattle

theorem james_bought_100_cattle :
  number_of_cattle 40000 0.20 1000 2 112000 = 100 :=
by {
  sorry
}

end james_bought_100_cattle_l944_94488


namespace current_ratio_of_employees_l944_94474

-- Definitions for the number of current male employees and the ratio if 3 more men are hired
variables (M : ℕ) (F : ℕ)
variables (hM : M = 189)
variables (ratio_hired : (M + 3) / F = 8 / 9)

-- Conclusion we want to prove
theorem current_ratio_of_employees (M F : ℕ) (hM : M = 189) (ratio_hired : (M + 3) / F = 8 / 9) : 
  M / F = 7 / 8 :=
sorry

end current_ratio_of_employees_l944_94474


namespace power_function_no_origin_l944_94463

theorem power_function_no_origin (m : ℝ) :
  (m = 1 ∨ m = 2) → 
  (m^2 - 3 * m + 3 ≠ 0 ∧ (m - 2) * (m + 1) ≤ 0) :=
by
  intro h
  cases h
  case inl =>
    -- m = 1 case will be processed here
    sorry
  case inr =>
    -- m = 2 case will be processed here
    sorry

end power_function_no_origin_l944_94463


namespace team_selection_l944_94498

-- Define the number of boys and girls in the club
def boys : Nat := 10
def girls : Nat := 12

-- Define the number of boys and girls to be selected for the team
def boys_team : Nat := 4
def girls_team : Nat := 4

-- Calculate the number of combinations using Nat.choose
noncomputable def choosing_boys : Nat := Nat.choose boys boys_team
noncomputable def choosing_girls : Nat := Nat.choose girls girls_team

-- Calculate the total number of ways to form the team
noncomputable def total_combinations : Nat := choosing_boys * choosing_girls

-- Theorem stating the total number of combinations equals the correct answer
theorem team_selection :
  total_combinations = 103950 := by
  sorry

end team_selection_l944_94498


namespace bananas_on_first_day_l944_94499

theorem bananas_on_first_day (total_bananas : ℕ) (days : ℕ) (increment : ℕ) (bananas_first_day : ℕ) :
  (total_bananas = 100) ∧ (days = 5) ∧ (increment = 6) ∧ ((bananas_first_day + (bananas_first_day + increment) + 
  (bananas_first_day + 2*increment) + (bananas_first_day + 3*increment) + (bananas_first_day + 4*increment)) = total_bananas) → 
  bananas_first_day = 8 :=
by
  sorry

end bananas_on_first_day_l944_94499


namespace smallest_positive_cube_ends_in_112_l944_94400

theorem smallest_positive_cube_ends_in_112 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 112 ∧ (∀ m : ℕ, (m > 0 ∧ m^3 % 1000 = 112) → n ≤ m) :=
by
  sorry

end smallest_positive_cube_ends_in_112_l944_94400


namespace part1_part2_part3_l944_94402

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem part1 : determinant (-3) (-2) 4 5 = -7 := by
  sorry

theorem part2 (x: ℝ) (h: determinant 2 (-2 * x) 3 (-5 * x) = 2) : x = -1/2 := by
  sorry

theorem part3 (m n x: ℝ) 
  (h1: determinant (8 * m * x - 1) (-8/3 + 2 * x) (3/2) (-3) = 
        determinant 6 (-1) (-n) x) : 
    m = -3/8 ∧ n = -7 := by
  sorry

end part1_part2_part3_l944_94402


namespace proof_case_a_proof_case_b_l944_94464

noncomputable def proof_problem_a (x y z p q : ℝ) (n : ℕ) 
  (h1 : y = x^n + p*x + q) 
  (h2 : z = y^n + p*y + q) 
  (h3 : x = z^n + p*z + q) : Prop :=
  x^2 * y + y^2 * z + z^2 * x >= x^2 * z + y^2 * x + z^2 * y

theorem proof_case_a (x y z p q : ℝ) 
  (h1 : y = x^2 + p*x + q) 
  (h2 : z = y^2 + p*y + q) 
  (h3 : x = z^2 + p*z + q) : 
  proof_problem_a x y z p q 2 h1 h2 h3 := 
sorry

theorem proof_case_b (x y z p q : ℝ) 
  (h1 : y = x^2010 + p*x + q) 
  (h2 : z = y^2010 + p*y + q) 
  (h3 : x = z^2010 + p*z + q) : 
  proof_problem_a x y z p q 2010 h1 h2 h3 := 
sorry

end proof_case_a_proof_case_b_l944_94464


namespace value_of_x_minus_y_l944_94407

theorem value_of_x_minus_y 
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : 3 * x - y = 8) :
  x - y = 3 := by
  sorry

end value_of_x_minus_y_l944_94407


namespace fish_original_count_l944_94420

theorem fish_original_count (F : ℕ) (h : F / 2 - F / 6 = 12) : F = 36 := 
by 
  sorry

end fish_original_count_l944_94420


namespace unique_three_digit_numbers_l944_94473

theorem unique_three_digit_numbers (d1 d2 d3 : ℕ) :
  (d1 = 3 ∧ d2 = 0 ∧ d3 = 8) →
  ∃ nums : Finset ℕ, 
  (∀ n ∈ nums, (∃ h t u : ℕ, n = 100 * h + 10 * t + u ∧ 
                h ≠ 0 ∧ (h = d1 ∨ h = d2 ∨ h = d3) ∧ 
                (t = d1 ∨ t = d2 ∨ t = d3) ∧ (u = d1 ∨ u = d2 ∨ u = d3) ∧ 
                h ≠ t ∧ t ≠ u ∧ u ≠ h)) ∧ nums.card = 4 :=
by
  sorry

end unique_three_digit_numbers_l944_94473


namespace min_value_of_a_l944_94491

noncomputable def P (x : ℕ) : ℤ := sorry

def smallest_value_of_a (a : ℕ) : Prop :=
  a > 0 ∧
  (P 1 = a ∧ P 3 = a ∧ P 5 = a ∧ P 7 = a ∧ P 9 = a ∧
   P 2 = -a ∧ P 4 = -a ∧ P 6 = -a ∧ P 8 = -a ∧ P 10 = -a)

theorem min_value_of_a : ∃ a : ℕ, smallest_value_of_a a ∧ a = 6930 :=
sorry

end min_value_of_a_l944_94491


namespace mark_old_bills_l944_94413

noncomputable def old_hourly_wage : ℝ := 40
noncomputable def new_hourly_wage : ℝ := 42
noncomputable def work_hours_per_week : ℝ := 8 * 5
noncomputable def personal_trainer_cost_per_week : ℝ := 100
noncomputable def leftover_after_expenses : ℝ := 980

noncomputable def new_weekly_earnings := new_hourly_wage * work_hours_per_week
noncomputable def total_weekly_spending_after_raise := leftover_after_expenses + personal_trainer_cost_per_week
noncomputable def old_bills_per_week := new_weekly_earnings - total_weekly_spending_after_raise

theorem mark_old_bills : old_bills_per_week = 600 := by
  sorry

end mark_old_bills_l944_94413


namespace picnic_total_persons_l944_94431

-- Definitions based on given conditions
variables (W M A C : ℕ)
axiom cond1 : M = W + 80
axiom cond2 : A = C + 80
axiom cond3 : M = 120

-- Proof problem: Total persons = 240
theorem picnic_total_persons : W + M + A + C = 240 :=
by
  -- Proof will be filled here
  sorry

end picnic_total_persons_l944_94431


namespace exists_positive_integer_m_l944_94423

theorem exists_positive_integer_m (n : ℕ) (hn : 0 < n) : ∃ m : ℕ, 0 < m ∧ 7^n ∣ (3^m + 5^m - 1) :=
sorry

end exists_positive_integer_m_l944_94423


namespace sequence_sum_relation_l944_94493

theorem sequence_sum_relation (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, 4 * S n = (a n + 1) ^ 2) →
  (S 1 = a 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) = a n + 2) →
  a 2023 = 4045 :=
by
  sorry

end sequence_sum_relation_l944_94493


namespace problem_statement_l944_94490

-- Assume F is a function defined such that given the point (4,4) is on the graph y = F(x)
def F : ℝ → ℝ := sorry

-- Hypothesis: (4, 4) is on the graph of y = F(x)
axiom H : F 4 = 4

-- We need to prove that F(4) = 4
theorem problem_statement : F 4 = 4 :=
by exact H

end problem_statement_l944_94490


namespace length_of_AE_l944_94411

variable (A B C D E : Type) [AddGroup A]
variable (AB CD AC AE EC : ℝ)
variable 
  (hAB : AB = 8)
  (hCD : CD = 18)
  (hAC : AC = 20)
  (hEqualAreas : ∀ (AED BEC : Type), (area AED = area BEC) → (AED = BEC))

theorem length_of_AE (hRatio : AE / EC = 4 / 9) (hSum : AC = AE + EC) : AE = 80 / 13 :=
by
  sorry

end length_of_AE_l944_94411


namespace percent_women_surveryed_equal_40_l944_94408

theorem percent_women_surveryed_equal_40
  (W M : ℕ) 
  (h1 : W + M = 100)
  (h2 : (W / 100 * 1 / 10 : ℚ) + (M / 100 * 1 / 4 : ℚ) = (19 / 100 : ℚ))
  (h3 : (9 / 10 : ℚ) * (W / 100 : ℚ) + (3 / 4 : ℚ) * (M / 100 : ℚ) = (1 - 19 / 100 : ℚ)) :
  W = 40 := 
sorry

end percent_women_surveryed_equal_40_l944_94408


namespace range_of_a_l944_94409

variable (f : ℝ → ℝ) (a : ℝ)

-- Definitions based on provided conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x ≤ f y

-- Main statement
theorem range_of_a
    (hf_even : is_even f)
    (hf_mono : is_monotonically_increasing f)
    (h_ineq : ∀ x : ℝ, f (Real.log (a) / Real.log 2) ≤ f (x^2 - 2 * x + 2)) :
  (1/2 : ℝ) ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l944_94409


namespace parallelogram_larger_angle_l944_94449

theorem parallelogram_larger_angle (a b : ℕ) (h₁ : b = a + 50) (h₂ : a = 65) : b = 115 := 
by
  -- Use the conditions h₁ and h₂ to prove the statement.
  sorry

end parallelogram_larger_angle_l944_94449


namespace henry_games_l944_94424

theorem henry_games {N H : ℕ} (hN : N = 7) (hH : H = 4 * N) 
    (h_final: H - 6 = 4 * (N + 6)) : H = 58 :=
by
  -- Proof would be inserted here, but skipped using sorry
  sorry

end henry_games_l944_94424


namespace L_shape_area_correct_l944_94461

noncomputable def large_rectangle_area : ℕ := 12 * 7
noncomputable def small_rectangle_area : ℕ := 4 * 3
noncomputable def L_shape_area := large_rectangle_area - small_rectangle_area

theorem L_shape_area_correct : L_shape_area = 72 := by
  -- here goes your solution
  sorry

end L_shape_area_correct_l944_94461


namespace albums_total_l944_94481

noncomputable def miriam_albums (katrina_albums : ℕ) := 5 * katrina_albums
noncomputable def katrina_albums (bridget_albums : ℕ) := 6 * bridget_albums
noncomputable def bridget_albums (adele_albums : ℕ) := adele_albums - 15
noncomputable def total_albums (adele_albums : ℕ) (bridget_albums : ℕ) (katrina_albums : ℕ) (miriam_albums : ℕ) := 
  adele_albums + bridget_albums + katrina_albums + miriam_albums

theorem albums_total (adele_has_30 : adele_albums = 30) : 
  total_albums 30 (bridget_albums 30) (katrina_albums (bridget_albums 30)) (miriam_albums (katrina_albums (bridget_albums 30))) = 585 := 
by 
  -- In Lean, replace the term 'sorry' below with the necessary steps to finish the proof
  sorry

end albums_total_l944_94481


namespace schoolchildren_initial_speed_l944_94456

theorem schoolchildren_initial_speed (v : ℝ) (t t_1 t_2 : ℝ) 
  (h1 : t_1 = (6 * v) / (v + 60) + (400 - 3 * v) / (v + 60)) 
  (h2 : t_2 = (400 - 3 * v) / v) 
  (h3 : t_1 = t_2) : v = 63.24 :=
by sorry

end schoolchildren_initial_speed_l944_94456


namespace solve_2xx_eq_sqrt2_unique_solution_l944_94439

noncomputable def solve_equation_2xx_eq_sqrt2 (x : ℝ) : Prop :=
  2 * x^x = Real.sqrt 2

theorem solve_2xx_eq_sqrt2_unique_solution (x : ℝ) : solve_equation_2xx_eq_sqrt2 x ↔ (x = 1/2 ∨ x = 1/4) ∧ x > 0 :=
by
  sorry

end solve_2xx_eq_sqrt2_unique_solution_l944_94439


namespace problem_real_numbers_l944_94417

theorem problem_real_numbers (a b c d r : ℝ) 
  (h1 : b + c + d = r * a) 
  (h2 : a + c + d = r * b) 
  (h3 : a + b + d = r * c) 
  (h4 : a + b + c = r * d) : 
  r = 3 ∨ r = -1 :=
sorry

end problem_real_numbers_l944_94417


namespace line_parabola_intersections_l944_94457

theorem line_parabola_intersections (k : ℝ) :
  ((∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) ↔ k = 0) ∧
  (¬∃ x₁ x₂, x₁ ≠ x₂ ∧ (k * (x₁ - 2) + 1)^2 = 4 * x₁ ∧ (k * (x₂ - 2) + 1)^2 = 4 * x₂) ∧
  (¬∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) :=
by sorry

end line_parabola_intersections_l944_94457


namespace relationship_between_abc_l944_94458

noncomputable def a := (4 / 5) ^ (1 / 2)
noncomputable def b := (5 / 4) ^ (1 / 5)
noncomputable def c := (3 / 4) ^ (3 / 4)

theorem relationship_between_abc : c < a ∧ a < b := by
  sorry

end relationship_between_abc_l944_94458


namespace Caitlin_Sara_weight_l944_94460

variable (A C S : ℕ)

theorem Caitlin_Sara_weight 
  (h1 : A + C = 95) 
  (h2 : A = S + 8) : 
  C + S = 87 := by
  sorry

end Caitlin_Sara_weight_l944_94460


namespace hh3_eq_2943_l944_94435

-- Define the function h
def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

-- Prove that h(h(3)) = 2943
theorem hh3_eq_2943 : h (h 3) = 2943 :=
by
  sorry

end hh3_eq_2943_l944_94435
