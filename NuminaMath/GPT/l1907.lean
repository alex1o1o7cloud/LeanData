import Mathlib

namespace diff_of_roots_l1907_190797

-- Define the quadratic equation and its coefficients
def quadratic_eq (z : ℝ) : ℝ := 2 * z^2 + 5 * z - 12

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the roots of the quadratic equation using the quadratic formula
noncomputable def larger_root (a b c : ℝ) : ℝ := (-b + Real.sqrt (discriminant a b c)) / (2 * a)
noncomputable def smaller_root (a b c : ℝ) : ℝ := (-b - Real.sqrt (discriminant a b c)) / (2 * a)

-- Define the proof statement
theorem diff_of_roots : 
  ∃ (a b c z1 z2 : ℝ), 
    a = 2 ∧ b = 5 ∧ c = -12 ∧
    quadratic_eq z1 = 0 ∧ quadratic_eq z2 = 0 ∧
    z1 = smaller_root a b c ∧ z2 = larger_root a b c ∧
    z2 - z1 = 5.5 := 
by 
  sorry

end diff_of_roots_l1907_190797


namespace xy_value_l1907_190757

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := 
by sorry

end xy_value_l1907_190757


namespace fraction_meaningful_iff_l1907_190755

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = x / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l1907_190755


namespace arrangement_of_BANANA_l1907_190756

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l1907_190756


namespace solve_system1_l1907_190701

structure SystemOfEquations :=
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)

def system1 : SystemOfEquations :=
  { a₁ := 1, b₁ := -3, c₁ := 4,
    a₂ := 2, b₂ := -1, c₂ := 3 }

theorem solve_system1 :
  ∃ x y : ℝ, x - 3 * y = 4 ∧ 2 * x - y = 3 ∧ x = 1 ∧ y = -1 :=
by
  sorry

end solve_system1_l1907_190701


namespace count_true_statements_l1907_190785

theorem count_true_statements (x : ℝ) (h : x > -3) :
  (if (x > -3 → x > -6) then 1 else 0) +
  (if (¬ (x > -3 → x > -6)) then 1 else 0) +
  (if (x > -6 → x > -3) then 1 else 0) +
  (if (¬ (x > -6 → x > -3)) then 1 else 0) = 2 :=
sorry

end count_true_statements_l1907_190785


namespace ratio_of_boys_in_class_l1907_190773

noncomputable def boy_to_total_ratio (p_boy p_girl : ℚ) : ℚ :=
p_boy / (p_boy + p_girl)

theorem ratio_of_boys_in_class (p_boy p_girl total_students : ℚ)
    (h1 : p_boy = (3/4) * p_girl)
    (h2 : p_boy + p_girl = 1)
    (h3 : total_students = 1) :
    boy_to_total_ratio p_boy p_girl = 3/7 :=
by
  sorry

end ratio_of_boys_in_class_l1907_190773


namespace gcd_180_270_450_l1907_190791

theorem gcd_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 := by 
  sorry

end gcd_180_270_450_l1907_190791


namespace max_sum_composite_shape_l1907_190764

theorem max_sum_composite_shape :
  let faces_hex_prism := 8
  let edges_hex_prism := 18
  let vertices_hex_prism := 12

  let faces_hex_with_pyramid := 8 - 1 + 6
  let edges_hex_with_pyramid := 18 + 6
  let vertices_hex_with_pyramid := 12 + 1
  let sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  let faces_rec_with_pyramid := 8 - 1 + 5
  let edges_rec_with_pyramid := 18 + 4
  let vertices_rec_with_pyramid := 12 + 1
  let sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sum_hex_with_pyramid = 50 ∧ sum_rec_with_pyramid = 46 ∧ sum_hex_with_pyramid ≥ sum_rec_with_pyramid := 
by
  have faces_hex_prism := 8
  have edges_hex_prism := 18
  have vertices_hex_prism := 12

  have faces_hex_with_pyramid := 8 - 1 + 6
  have edges_hex_with_pyramid := 18 + 6
  have vertices_hex_with_pyramid := 12 + 1
  have sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  have faces_rec_with_pyramid := 8 - 1 + 5
  have edges_rec_with_pyramid := 18 + 4
  have vertices_rec_with_pyramid := 12 + 1
  have sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sorry -- proof omitted

end max_sum_composite_shape_l1907_190764


namespace sum_of_edges_l1907_190784

-- Define the number of edges for a triangle and a rectangle
def edges_triangle : Nat := 3
def edges_rectangle : Nat := 4

-- The theorem states that the sum of the edges of a triangle and a rectangle is 7
theorem sum_of_edges : edges_triangle + edges_rectangle = 7 := 
by
  -- proof omitted
  sorry

end sum_of_edges_l1907_190784


namespace exists_odd_a_b_and_positive_k_l1907_190731

theorem exists_odd_a_b_and_positive_k (m : ℤ) :
  ∃ (a b : ℤ) (k : ℕ), a % 2 = 1 ∧ b % 2 = 1 ∧ k > 0 ∧ 2 * m = a^5 + b^5 + k * 2^100 := 
sorry

end exists_odd_a_b_and_positive_k_l1907_190731


namespace small_possible_value_l1907_190798

theorem small_possible_value (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : 2^12 * 3^3 = a^b) : a + b = 110593 := by
  sorry

end small_possible_value_l1907_190798


namespace sum_of_three_consecutive_numbers_l1907_190753

theorem sum_of_three_consecutive_numbers (smallest : ℕ) (h : smallest = 29) :
  (smallest + (smallest + 1) + (smallest + 2)) = 90 :=
by
  sorry

end sum_of_three_consecutive_numbers_l1907_190753


namespace average_salary_of_officers_l1907_190795

-- Define the given conditions
def avg_salary_total := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 480

-- Define the expected result
def avg_salary_officers := 440

-- Define the problem and statement to be proved in Lean
theorem average_salary_of_officers :
  (num_officers + num_non_officers) * avg_salary_total - num_non_officers * avg_salary_non_officers = num_officers * avg_salary_officers := 
by
  sorry

end average_salary_of_officers_l1907_190795


namespace find_u_v_l1907_190730

theorem find_u_v (u v : ℤ) (huv_pos : 0 < v ∧ v < u) (area_eq : u^2 + 3 * u * v = 615) : 
  u + v = 45 :=
sorry

end find_u_v_l1907_190730


namespace student_a_score_l1907_190754

def total_questions : ℕ := 100
def correct_responses : ℕ := 87
def incorrect_responses : ℕ := total_questions - correct_responses
def score : ℕ := correct_responses - 2 * incorrect_responses

theorem student_a_score : score = 61 := by
  unfold score
  unfold correct_responses
  unfold incorrect_responses
  norm_num
  -- At this point, the theorem is stated, but we insert sorry to satisfy the requirement of not providing the proof.
  sorry

end student_a_score_l1907_190754


namespace lowest_price_per_component_l1907_190729

def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 6
def fixed_monthly_costs : ℝ := 16500
def components_per_month : ℕ := 150

theorem lowest_price_per_component (price_per_component : ℝ) :
  let total_cost_per_component := production_cost_per_component + shipping_cost_per_component
  let total_production_and_shipping_cost := total_cost_per_component * components_per_month
  let total_cost := total_production_and_shipping_cost + fixed_monthly_costs
  price_per_component = total_cost / components_per_month → price_per_component = 196 :=
by
  sorry

end lowest_price_per_component_l1907_190729


namespace crayons_left_l1907_190777

-- Define initial number of crayons and the number taken by Mary
def initial_crayons : ℝ := 7.5
def taken_crayons : ℝ := 2.25

-- Calculate remaining crayons
def remaining_crayons := initial_crayons - taken_crayons

-- Prove that the remaining crayons are 5.25
theorem crayons_left : remaining_crayons = 5.25 := by
  sorry

end crayons_left_l1907_190777


namespace find_expression_for_a_n_l1907_190724

noncomputable def a_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2^n

theorem find_expression_for_a_n (a : ℕ → ℕ) (h : a_sequence a) (initial : a 1 = 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end find_expression_for_a_n_l1907_190724


namespace exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l1907_190748

theorem exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum :
  ∃ (a b c : ℤ), 2 * (a * b + b * c + c * a) = 4 * (a + b + c) :=
by
  -- Here we prove the existence of such integers a, b, c, which is stated in the theorem
  sorry

end exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l1907_190748


namespace flag_arrangement_remainder_l1907_190732

theorem flag_arrangement_remainder :
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  M % div = 441 := 
by
  -- Definitions
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  -- Proof
  sorry

end flag_arrangement_remainder_l1907_190732


namespace boat_speed_l1907_190727

theorem boat_speed (b s : ℝ) (h1 : b + s = 7) (h2 : b - s = 5) : b = 6 := 
by
  sorry

end boat_speed_l1907_190727


namespace max_cos_half_sin_eq_1_l1907_190796

noncomputable def max_value_expression (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 - Real.sin θ)

theorem max_cos_half_sin_eq_1 : 
  ∀ θ : ℝ, 0 < θ ∧ θ < π → max_value_expression θ ≤ 1 :=
by
  intros θ h
  sorry

end max_cos_half_sin_eq_1_l1907_190796


namespace john_daily_reading_hours_l1907_190786

-- Definitions from the conditions
def reading_rate := 50  -- pages per hour
def total_pages := 2800  -- pages
def weeks := 4
def days_per_week := 7

-- Hypotheses derived from the conditions
def total_hours := total_pages / reading_rate  -- 2800 / 50 = 56 hours
def total_days := weeks * days_per_week  -- 4 * 7 = 28 days

-- Theorem to prove 
theorem john_daily_reading_hours : (total_hours / total_days) = 2 := by
  sorry

end john_daily_reading_hours_l1907_190786


namespace part_I_solution_part_II_solution_l1907_190734

-- Part (I) proof problem: Prove the solution set for a specific inequality
theorem part_I_solution (x : ℝ) : -6 < x ∧ x < 10 / 3 → |2 * x - 2| + x + 1 < 9 :=
by
  sorry

-- Part (II) proof problem: Prove the range of 'a' for a given inequality to hold
theorem part_II_solution (a : ℝ) : (-3 ≤ a ∧ a ≤ 17 / 3) →
  (∀ x : ℝ, x ≥ 2 → |a * x + a - 4| + x + 1 ≤ (x + 2)^2) :=
by
  sorry

end part_I_solution_part_II_solution_l1907_190734


namespace find_x_plus_y_l1907_190762

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1005) 
  (h2 : x + 1005 * Real.sin y = 1003) 
  (h3 : π ≤ y ∧ y ≤ 3 * π / 2) : 
  x + y = 1005 + 3 * π / 2 :=
sorry

end find_x_plus_y_l1907_190762


namespace geometric_sequence_common_ratio_l1907_190723

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_arith : -a 5 + a 6 = 2 * a 4) :
  q = -1 ∨ q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l1907_190723


namespace alok_age_proof_l1907_190745

variable (A B C : ℕ)

theorem alok_age_proof (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  sorry

end alok_age_proof_l1907_190745


namespace pupils_like_only_maths_l1907_190778

noncomputable def number_pupils_like_only_maths (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) 
(neither_lovers: ℕ) (both_lovers: ℕ) : ℕ :=
maths_lovers - both_lovers

theorem pupils_like_only_maths : 
∀ (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) (neither_lovers: ℕ) (both_lovers: ℕ),
total = 30 →
maths_lovers = 20 →
english_lovers = 18 →
both_lovers = 2 * neither_lovers →
neither_lovers + maths_lovers + english_lovers - both_lovers - both_lovers = total →
number_pupils_like_only_maths total maths_lovers english_lovers neither_lovers both_lovers = 4 :=
by
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end pupils_like_only_maths_l1907_190778


namespace gain_percent_is_40_l1907_190746

-- Define the conditions
def purchase_price : ℕ := 800
def repair_costs : ℕ := 200
def selling_price : ℕ := 1400

-- Define the total cost
def total_cost : ℕ := purchase_price + repair_costs

-- Define the gain
def gain : ℕ := selling_price - total_cost

-- Define the gain percent
def gain_percent : ℕ := (gain * 100) / total_cost

theorem gain_percent_is_40 : gain_percent = 40 := by
  -- Placeholder for the proof
  sorry

end gain_percent_is_40_l1907_190746


namespace smallest_sum_of_big_in_circle_l1907_190700

theorem smallest_sum_of_big_in_circle (arranged_circle : Fin 8 → ℕ) (h_circle : ∀ n, arranged_circle n ∈ Finset.range (9) ∧ arranged_circle n > 0) :
  (∀ n, (arranged_circle n > arranged_circle (n + 1) % 8 ∧ arranged_circle n > arranged_circle (n + 7) % 8) ∨ (arranged_circle n < arranged_circle (n + 1) % 8 ∧ arranged_circle n < arranged_circle (n + 7) % 8)) →
  ∃ big_indices : Finset (Fin 8), big_indices.card = 4 ∧ big_indices.sum arranged_circle = 23 :=
by
  sorry

end smallest_sum_of_big_in_circle_l1907_190700


namespace part1_l1907_190721

def A (a : ℝ) : Set ℝ := {x | 2 * a - 3 < x ∧ x < a + 1}
def B : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem part1 (a : ℝ) (h : a = 0) : A a ∩ B = {x | -1 < x ∧ x < 1} :=
by
  -- Proof here
  sorry

end part1_l1907_190721


namespace candy_count_l1907_190763

variables (S M L : ℕ)

theorem candy_count :
  S + M + L = 110 ∧ S + L = 100 ∧ L = S + 20 → S = 40 ∧ M = 10 ∧ L = 60 :=
by
  intros h
  sorry

end candy_count_l1907_190763


namespace inequalities_for_m_gt_n_l1907_190702

open Real

theorem inequalities_for_m_gt_n (m n : ℕ) (hmn : m > n) : 
  (1 + 1 / (m : ℝ)) ^ m > (1 + 1 / (n : ℝ)) ^ n ∧ 
  (1 + 1 / (m : ℝ)) ^ (m + 1) < (1 + 1 / (n : ℝ)) ^ (n + 1) := 
by
  sorry

end inequalities_for_m_gt_n_l1907_190702


namespace sum_f_to_2017_l1907_190790

noncomputable def f (x : ℕ) : ℝ := Real.cos (x * Real.pi / 3)

theorem sum_f_to_2017 : (Finset.range 2017).sum f = 1 / 2 :=
by
  sorry

end sum_f_to_2017_l1907_190790


namespace log_ordering_l1907_190710

theorem log_ordering 
  (a b c : ℝ) 
  (ha: a = Real.log 3 / Real.log 2) 
  (hb: b = Real.log 2 / Real.log 3) 
  (hc: c = Real.log 0.5 / Real.log 10) : 
  a > b ∧ b > c := 
by 
  sorry

end log_ordering_l1907_190710


namespace boat_distance_downstream_is_68_l1907_190739

variable (boat_speed : ℕ) (stream_speed : ℕ) (time_hours : ℕ)

-- Given conditions
def effective_speed_downstream (boat_speed stream_speed : ℕ) : ℕ := boat_speed + stream_speed
def distance_downstream (speed time : ℕ) : ℕ := speed * time

theorem boat_distance_downstream_is_68 
  (h1 : boat_speed = 13) 
  (h2 : stream_speed = 4) 
  (h3 : time_hours = 4) : 
  distance_downstream (effective_speed_downstream boat_speed stream_speed) time_hours = 68 := 
by 
  sorry

end boat_distance_downstream_is_68_l1907_190739


namespace bank_balance_after_2_years_l1907_190712

noncomputable def compound_interest (P₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P₀ * (1 + r)^n

theorem bank_balance_after_2_years :
  compound_interest 100 0.10 2 = 121 := 
  by
  sorry

end bank_balance_after_2_years_l1907_190712


namespace haley_stickers_l1907_190771

theorem haley_stickers (friends : ℕ) (stickers_per_friend : ℕ) (total_stickers : ℕ) :
  friends = 9 → stickers_per_friend = 8 → total_stickers = friends * stickers_per_friend → total_stickers = 72 :=
by
  intros h_friends h_stickers_per_friend h_total_stickers
  rw [h_friends, h_stickers_per_friend] at h_total_stickers
  exact h_total_stickers

end haley_stickers_l1907_190771


namespace tenth_term_of_arithmetic_sequence_l1907_190799

-- Define the initial conditions: first term 'a' and the common difference 'd'
def a : ℤ := 2
def d : ℤ := 1 - a

-- Define the n-th term of an arithmetic sequence formula
def nth_term (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Statement to prove
theorem tenth_term_of_arithmetic_sequence :
  nth_term a d 10 = -7 := 
by
  sorry

end tenth_term_of_arithmetic_sequence_l1907_190799


namespace isosceles_triangle_no_obtuse_l1907_190775

theorem isosceles_triangle_no_obtuse (A B C : ℝ) 
  (h1 : A = 70) 
  (h2 : B = 70) 
  (h3 : A + B + C = 180) 
  (h_iso : A = B) 
  : (A ≤ 90) ∧ (B ≤ 90) ∧ (C ≤ 90) :=
by
  sorry

end isosceles_triangle_no_obtuse_l1907_190775


namespace find_a_l1907_190703

theorem find_a (a : ℝ) (x_values y_values : List ℝ)
  (h_y : ∀ x, List.getD y_values x 0 = 2.1 * List.getD x_values x 1 - 0.3) :
  a = 10 :=
by
  have h_mean_x : (1 + 2 + 3 + 4 + 5) / 5 = 3 := by norm_num
  have h_sum_y : (2 + 3 + 7 + 8 + a) / 5 = (2.1 * 3 - 0.3) := by sorry
  sorry

end find_a_l1907_190703


namespace min_positive_integer_expression_l1907_190736

theorem min_positive_integer_expression : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (m: ℝ) / 3 + 27 / (m: ℝ) ≥ (n: ℝ) / 3 + 27 / (n: ℝ)) ∧ (n / 3 + 27 / n = 6) :=
sorry

end min_positive_integer_expression_l1907_190736


namespace find_x_l1907_190750

theorem find_x (x y : ℝ) (hx : x ≠ 0) (h1 : x / 2 = y^2) (h2 : x / 4 = 4 * y) : x = 128 :=
by
  sorry

end find_x_l1907_190750


namespace statement_C_l1907_190779

theorem statement_C (x : ℝ) (h : x^2 < 4) : x < 2 := 
sorry

end statement_C_l1907_190779


namespace fencing_rate_correct_l1907_190794

noncomputable def rate_of_fencing_per_meter (area_hectares : ℝ) (total_cost : ℝ) : ℝ :=
  let area_sqm := area_hectares * 10000
  let r_squared := area_sqm / Real.pi
  let r := Real.sqrt r_squared
  let circumference := 2 * Real.pi * r
  total_cost / circumference

theorem fencing_rate_correct :
  rate_of_fencing_per_meter 13.86 6070.778380479544 = 4.60 :=
by
  sorry

end fencing_rate_correct_l1907_190794


namespace overall_percentage_good_fruits_l1907_190707

theorem overall_percentage_good_fruits
  (oranges_bought : ℕ)
  (bananas_bought : ℕ)
  (apples_bought : ℕ)
  (pears_bought : ℕ)
  (oranges_rotten_percent : ℝ)
  (bananas_rotten_percent : ℝ)
  (apples_rotten_percent : ℝ)
  (pears_rotten_percent : ℝ)
  (h_oranges : oranges_bought = 600)
  (h_bananas : bananas_bought = 400)
  (h_apples : apples_bought = 800)
  (h_pears : pears_bought = 200)
  (h_oranges_rotten : oranges_rotten_percent = 0.15)
  (h_bananas_rotten : bananas_rotten_percent = 0.03)
  (h_apples_rotten : apples_rotten_percent = 0.12)
  (h_pears_rotten : pears_rotten_percent = 0.25) :
  let total_fruits := oranges_bought + bananas_bought + apples_bought + pears_bought
  let rotten_oranges := oranges_rotten_percent * oranges_bought
  let rotten_bananas := bananas_rotten_percent * bananas_bought
  let rotten_apples := apples_rotten_percent * apples_bought
  let rotten_pears := pears_rotten_percent * pears_bought
  let good_oranges := oranges_bought - rotten_oranges
  let good_bananas := bananas_bought - rotten_bananas
  let good_apples := apples_bought - rotten_apples
  let good_pears := pears_bought - rotten_pears
  let total_good_fruits := good_oranges + good_bananas + good_apples + good_pears
  (total_good_fruits / total_fruits) * 100 = 87.6 :=
by
  sorry

end overall_percentage_good_fruits_l1907_190707


namespace delta_y_over_delta_x_l1907_190776

variable (Δx : ℝ)

def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem delta_y_over_delta_x : (f (1 + Δx) - f 1) / Δx = 4 + 2 * Δx :=
by
  sorry

end delta_y_over_delta_x_l1907_190776


namespace jill_travel_time_to_school_is_20_minutes_l1907_190747

variables (dave_rate : ℕ) (dave_step : ℕ) (dave_time : ℕ)
variables (jill_rate : ℕ) (jill_step : ℕ)

def dave_distance : ℕ := dave_rate * dave_step * dave_time
def jill_time_to_school : ℕ := dave_distance dave_rate dave_step dave_time / (jill_rate * jill_step)

theorem jill_travel_time_to_school_is_20_minutes : 
  dave_rate = 85 → dave_step = 80 → dave_time = 18 → 
  jill_rate = 120 → jill_step = 50 → jill_time_to_school 85 80 18 120 50 = 20 :=
by
  intros
  unfold jill_time_to_school
  unfold dave_distance
  sorry

end jill_travel_time_to_school_is_20_minutes_l1907_190747


namespace plane_equation_parallel_to_Oz_l1907_190752

theorem plane_equation_parallel_to_Oz (A B D : ℝ)
  (h1 : A * 1 + B * 0 + D = 0)
  (h2 : A * (-2) + B * 1 + D = 0)
  (h3 : ∀ z : ℝ, exists c : ℝ, A * z + B * c + D = 0):
  A = 1 ∧ B = 3 ∧ D = -1 :=
  by
  sorry

end plane_equation_parallel_to_Oz_l1907_190752


namespace lowest_possible_number_of_students_l1907_190788

theorem lowest_possible_number_of_students :
  Nat.lcm 18 24 = 72 :=
by
  sorry

end lowest_possible_number_of_students_l1907_190788


namespace find_xy_l1907_190760

theorem find_xy (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : 
  2^x - 5 = 11^y ↔ (x = 4 ∧ y = 1) :=
by sorry

end find_xy_l1907_190760


namespace no_perfect_square_in_range_l1907_190743

def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem no_perfect_square_in_range :
  ∀ (n : ℕ), 4 ≤ n ∧ n ≤ 12 → ¬ isPerfectSquare (2*n*n + 3*n + 2) :=
by
  intro n
  intro h
  sorry

end no_perfect_square_in_range_l1907_190743


namespace initial_workers_number_l1907_190722

-- Define the initial problem
variables {W : ℕ} -- Number of initial workers
variables (Work1 : ℕ := W * 8) -- Work done for the first hole
variables (Work2 : ℕ := (W + 65) * 6) -- Work done for the second hole
variables (Depth1 : ℕ := 30) -- Depth of the first hole
variables (Depth2 : ℕ := 55) -- Depth of the second hole

-- Expressing the conditions and question
theorem initial_workers_number : 8 * W * 55 = 30 * (W + 65) * 6 → W = 45 :=
by
  sorry

end initial_workers_number_l1907_190722


namespace jim_loan_inequality_l1907_190726

noncomputable def A (t : ℕ) : ℝ := 1500 * (1.06 ^ t)

theorem jim_loan_inequality : ∃ t : ℕ, A t > 3000 ∧ ∀ t' : ℕ, t' < t → A t' ≤ 3000 :=
by
  sorry

end jim_loan_inequality_l1907_190726


namespace additional_miles_needed_l1907_190718

theorem additional_miles_needed :
  ∀ (h : ℝ), (25 + 75 * h) / (5 / 8 + h) = 60 → 75 * h = 62.5 := 
by
  intros h H
  -- the rest of the proof goes here
  sorry

end additional_miles_needed_l1907_190718


namespace area_of_woods_l1907_190728

def width := 8 -- the width in miles
def length := 3 -- the length in miles
def area (w : Nat) (l : Nat) : Nat := w * l -- the area function for a rectangle

theorem area_of_woods : area width length = 24 := by
  sorry

end area_of_woods_l1907_190728


namespace binom_eight_four_l1907_190733

theorem binom_eight_four : (Nat.choose 8 4) = 70 :=
by
  sorry

end binom_eight_four_l1907_190733


namespace find_a5_l1907_190789

variable {a : ℕ → ℝ}  -- Define the sequence a(n)

-- Define the conditions of the problem
variable (a1_positive : ∀ n, a n > 0)
variable (geo_seq : ∀ n, a (n + 1) = a n * 2)
variable (condition : (a 3) * (a 11) = 16)

theorem find_a5 (a1_positive : ∀ n, a n > 0) (geo_seq : ∀ n, a (n + 1) = a n * 2)
(condition : (a 3) * (a 11) = 16) : a 5 = 1 := by
  sorry

end find_a5_l1907_190789


namespace ferns_have_1260_leaves_l1907_190758

def num_ferns : ℕ := 6
def fronds_per_fern : ℕ := 7
def leaves_per_frond : ℕ := 30
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem ferns_have_1260_leaves : total_leaves = 1260 :=
by 
  -- proof goes here
  sorry

end ferns_have_1260_leaves_l1907_190758


namespace change_from_fifteen_dollars_l1907_190719

theorem change_from_fifteen_dollars : 
  ∀ (cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid : ℕ),
  cost_eggs = 3 →
  cost_pancakes = 2 →
  cost_mug_cocoa = 2 →
  num_mugs = 2 →
  tax = 1 →
  additional_pancakes = 2 →
  additional_mug = 2 →
  paid = 15 →
  paid - (cost_eggs + cost_pancakes + (num_mugs * cost_mug_cocoa) + tax + additional_pancakes + additional_mug) = 1 :=
by
  intros cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid
  sorry

end change_from_fifteen_dollars_l1907_190719


namespace geometric_prog_y_90_common_ratio_l1907_190725

theorem geometric_prog_y_90_common_ratio :
  ∀ (y : ℝ), y = 90 → ∃ r : ℝ, r = (90 + y) / (30 + y) ∧ r = (180 + y) / (90 + y) ∧ r = 3 / 2 :=
by
  intros
  sorry

end geometric_prog_y_90_common_ratio_l1907_190725


namespace totalAttendees_l1907_190759

def numberOfBuses : ℕ := 8
def studentsPerBus : ℕ := 45
def chaperonesList : List ℕ := [2, 3, 4, 5, 3, 4, 2, 6]

theorem totalAttendees : 
    numberOfBuses * studentsPerBus + chaperonesList.sum = 389 := 
by
  sorry

end totalAttendees_l1907_190759


namespace candy_bar_cost_correct_l1907_190749

noncomputable def candy_bar_cost : ℕ := 25 -- Correct answer from the solution

theorem candy_bar_cost_correct (C : ℤ) (H1 : 3 * C + 150 + 50 = 11 * 25)
  (H2 : ∃ C, C ≥ 0) : C = candy_bar_cost :=
by
  sorry

end candy_bar_cost_correct_l1907_190749


namespace ellen_painting_time_l1907_190766

def time_to_paint_lilies := 5
def time_to_paint_roses := 7
def time_to_paint_orchids := 3
def time_to_paint_vines := 2

def number_of_lilies := 17
def number_of_roses := 10
def number_of_orchids := 6
def number_of_vines := 20

def total_time := 213

theorem ellen_painting_time:
  time_to_paint_lilies * number_of_lilies +
  time_to_paint_roses * number_of_roses +
  time_to_paint_orchids * number_of_orchids +
  time_to_paint_vines * number_of_vines = total_time := by
  sorry

end ellen_painting_time_l1907_190766


namespace total_children_on_playground_l1907_190781

theorem total_children_on_playground (girls boys : ℕ) (h_girls : girls = 28) (h_boys : boys = 35) : girls + boys = 63 := 
by 
  sorry

end total_children_on_playground_l1907_190781


namespace inequality_neg_mul_l1907_190709

theorem inequality_neg_mul (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
    sorry

end inequality_neg_mul_l1907_190709


namespace solve_division_problem_l1907_190787

-- Problem Conditions
def division_problem : ℚ := 0.25 / 0.005

-- Proof Problem Statement
theorem solve_division_problem : division_problem = 50 := by
  sorry

end solve_division_problem_l1907_190787


namespace ratio_of_quadratic_roots_l1907_190783

theorem ratio_of_quadratic_roots (a b c : ℝ) (h : 2 * b^2 = 9 * a * c) : 
  ∃ (x₁ x₂ : ℝ), (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ (x₁ / x₂ = 2) :=
sorry

end ratio_of_quadratic_roots_l1907_190783


namespace sum_of_roots_l1907_190713

theorem sum_of_roots (x : ℝ) :
  (3 * x - 2) * (x - 3) + (3 * x - 2) * (2 * x - 8) = 0 ->
  x = 2 / 3 ∨ x = 11 / 3 ->
  (2 / 3) + (11 / 3) = 13 / 3 :=
by
  sorry

end sum_of_roots_l1907_190713


namespace isosceles_triangle_inequality_l1907_190741

theorem isosceles_triangle_inequality
  (a b : ℝ)
  (hb : b > 0)
  (h₁₂ : 12 * (π / 180) = π / 15) 
  (h_sin6 : Real.sin (6 * (π / 180)) > 1 / 10)
  (h_eq : a = 2 * b * Real.sin (6 * (π / 180))) : 
  b < 5 * a := 
by
  sorry

end isosceles_triangle_inequality_l1907_190741


namespace find_a7_l1907_190706

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, ∃ r, a (n + m) = (a n) * (r ^ m)

def sequence_properties (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ a 3 = 3 ∧ a 11 = 27

theorem find_a7 (a : ℕ → ℝ) (h : sequence_properties a) : a 7 = 9 := 
sorry

end find_a7_l1907_190706


namespace goods_train_speed_l1907_190768

theorem goods_train_speed
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_taken : ℝ)
  (speed_kmph : ℝ)
  (h1 : length_train = 240.0416)
  (h2 : length_platform = 280)
  (h3 : time_taken = 26)
  (h4 : speed_kmph = 72.00576) :
  speed_kmph = ((length_train + length_platform) / time_taken) * 3.6 := sorry

end goods_train_speed_l1907_190768


namespace total_hours_worked_l1907_190717

def hours_day1 : ℝ := 2.5
def increment_day2 : ℝ := 0.5
def hours_day2 : ℝ := hours_day1 + increment_day2
def hours_day3 : ℝ := 3.75

theorem total_hours_worked :
  hours_day1 + hours_day2 + hours_day3 = 9.25 :=
sorry

end total_hours_worked_l1907_190717


namespace non_real_roots_interval_l1907_190716

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l1907_190716


namespace how_many_halves_to_sum_one_and_one_half_l1907_190737

theorem how_many_halves_to_sum_one_and_one_half : 
  (3 / 2) / (1 / 2) = 3 := 
by 
  sorry

end how_many_halves_to_sum_one_and_one_half_l1907_190737


namespace number_of_puppies_l1907_190708

theorem number_of_puppies (P K : ℕ) (h1 : K = 2 * P + 14) (h2 : K = 78) : P = 32 :=
by sorry

end number_of_puppies_l1907_190708


namespace range_of_m_l1907_190767

def isDistinctRealRootsInInterval (a b x : ℝ) : Prop :=
  a * x^2 + b * x + 4 = 0 ∧ 0 < x ∧ x ≤ 3

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) x ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) y) ↔
  (3 < m ∧ m ≤ 10 / 3) :=
sorry

end range_of_m_l1907_190767


namespace smallest_circle_equation_l1907_190772

theorem smallest_circle_equation :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ (x - 1)^2 + y^2 = 1 ∧ ((x - 1)^2 + y^2 = 1) = (x^2 + y^2 = 1) := 
sorry

end smallest_circle_equation_l1907_190772


namespace vector_parallel_addition_l1907_190720

theorem vector_parallel_addition 
  (x : ℝ)
  (a : ℝ × ℝ := (2, 1))
  (b : ℝ × ℝ := (x, -2)) 
  (h_parallel : 2 / x = 1 / -2) :
  a + b = (-2, -1) := 
by
  -- While the proof is omitted, the statement is complete and correct.
  sorry

end vector_parallel_addition_l1907_190720


namespace range_of_f_is_real_l1907_190744

noncomputable def f (x : ℝ) (m : ℝ) := Real.log (5^x + 4 / 5^x + m)

theorem range_of_f_is_real (m : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x m = y) ↔ m ≤ -4 :=
sorry

end range_of_f_is_real_l1907_190744


namespace Shiela_drawings_l1907_190751

theorem Shiela_drawings (n_neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
    (h1 : n_neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
by 
  sorry

end Shiela_drawings_l1907_190751


namespace find_r_l1907_190735

theorem find_r (r : ℝ) (cone1_radius cone2_radius cone3_radius : ℝ) (sphere_radius : ℝ)
  (cone_height_eq : cone1_radius = 2 * r ∧ cone2_radius = 3 * r ∧ cone3_radius = 10 * r)
  (sphere_touch : sphere_radius = 2)
  (center_eq_dist : ∀ {P Q : ℝ}, dist P Q = 2 → dist Q r = 2) :
  r = 1 := 
sorry

end find_r_l1907_190735


namespace problem1_problem2_l1907_190705

-- Problem (1) proof statement
theorem problem1 (a : ℝ) (h : a ≠ 0) : 
  3 * a^2 * a^3 + a^7 / a^2 = 4 * a^5 :=
by
  sorry

-- Problem (2) proof statement
theorem problem2 (x : ℝ) : 
  (x - 1)^2 - x * (x + 1) + (-2023)^0 = -3 * x + 2 :=
by
  sorry

end problem1_problem2_l1907_190705


namespace num_emails_received_after_second_deletion_l1907_190770

-- Define the initial conditions and final question
variable (initialEmails : ℕ)    -- Initial number of emails
variable (deletedEmails1 : ℕ)   -- First batch of deleted emails
variable (receivedEmails1 : ℕ)  -- First batch of received emails
variable (deletedEmails2 : ℕ)   -- Second batch of deleted emails
variable (receivedEmails2 : ℕ)  -- Second batch of received emails
variable (receivedEmails3 : ℕ)  -- Third batch of received emails
variable (finalEmails : ℕ)      -- Final number of emails in the inbox

-- Conditions based on the problem description
axiom initialEmails_def : initialEmails = 0
axiom deletedEmails1_def : deletedEmails1 = 50
axiom receivedEmails1_def : receivedEmails1 = 15
axiom deletedEmails2_def : deletedEmails2 = 20
axiom receivedEmails3_def : receivedEmails3 = 10
axiom finalEmails_def : finalEmails = 30

-- Question: Prove that the number of emails received after the second deletion is 5
theorem num_emails_received_after_second_deletion : receivedEmails2 = 5 :=
by
  sorry

end num_emails_received_after_second_deletion_l1907_190770


namespace find_general_formula_sum_b_n_less_than_two_l1907_190704

noncomputable def a_n (n : ℕ) : ℕ := n

noncomputable def S_n (n : ℕ) : ℚ := (n^2 + n) / 2

noncomputable def b_n (n : ℕ) : ℚ := 1 / S_n n

theorem find_general_formula (n : ℕ) : b_n n = 2 / (n^2 + n) := by 
  sorry

theorem sum_b_n_less_than_two (n : ℕ) :
  Finset.sum (Finset.range n) (λ k => b_n (k + 1)) < 2 :=
by 
  sorry

end find_general_formula_sum_b_n_less_than_two_l1907_190704


namespace a_minus_b_value_l1907_190780

theorem a_minus_b_value (a b : ℤ) :
  (∀ x : ℝ, 9 * x^3 + y^2 + a * x - b * x^3 + x + 5 = y^2 + 5) → a - b = -10 :=
by
  sorry

end a_minus_b_value_l1907_190780


namespace union_complement_eq_univ_l1907_190742

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 7}

-- Define set M
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N
def N : Set ℕ := {3, 5}

-- Define the complement of N with respect to U
def complement_U_N : Set ℕ := {1, 2, 4, 7}

-- Prove that U = M ∪ complement_U_N
theorem union_complement_eq_univ : U = M ∪ complement_U_N := 
sorry

end union_complement_eq_univ_l1907_190742


namespace isosceles_triangle_perimeter_l1907_190792

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : a ≠ b) (h4 : a + b > b) (h5 : a + b > a) 
: ∃ p : ℝ, p = 10 :=
by
  -- Using the given conditions to determine the perimeter
  sorry

end isosceles_triangle_perimeter_l1907_190792


namespace rationalize_denominator_l1907_190761

theorem rationalize_denominator (h : Real.sqrt 200 = 10 * Real.sqrt 2) : 
  (7 / Real.sqrt 200) = (7 * Real.sqrt 2 / 20) :=
by
  sorry

end rationalize_denominator_l1907_190761


namespace pizza_slices_l1907_190782

-- Definitions of conditions
def slices (H C : ℝ) : Prop :=
  (H / 2 - 3 + 2 * C / 3 = 11) ∧ (H = C)

-- Stating the theorem to prove
theorem pizza_slices (H C : ℝ) (h : slices H C) : H = 12 :=
sorry

end pizza_slices_l1907_190782


namespace kids_wearing_shoes_l1907_190711

-- Definitions based on the problem's conditions
def total_kids := 22
def kids_with_socks := 12
def kids_with_both := 6
def barefoot_kids := 8

-- Theorem statement
theorem kids_wearing_shoes :
  (∃ (kids_with_shoes : ℕ), 
     (kids_with_shoes = (total_kids - barefoot_kids) - (kids_with_socks - kids_with_both) + kids_with_both) ∧ 
     kids_with_shoes = 8) :=
by
  sorry

end kids_wearing_shoes_l1907_190711


namespace find_number_l1907_190714

theorem find_number : ∃ n : ℕ, (∃ x : ℕ, x / 15 = 4 ∧ x^2 = n) ∧ n = 3600 := 
by
  sorry

end find_number_l1907_190714


namespace problem_solution_l1907_190740

theorem problem_solution
  {a b c d : ℝ}
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2011)
  (h3 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2011) :
  (c * d)^2012 - (a * b)^2012 = 2011 :=
by
  sorry

end problem_solution_l1907_190740


namespace car_travel_distance_l1907_190738

theorem car_travel_distance :
  let a := 36
  let d := -12
  let n := 4
  let S := (n / 2) * (2 * a + (n - 1) * d)
  S = 72 := by
    sorry

end car_travel_distance_l1907_190738


namespace expansion_coeff_sum_l1907_190715

theorem expansion_coeff_sum
  (a : ℕ → ℤ)
  (h : ∀ x y : ℤ, (x - 2 * y) ^ 5 * (x + 3 * y) ^ 4 = 
    a 9 * x ^ 9 + 
    a 8 * x ^ 8 * y + 
    a 7 * x ^ 7 * y ^ 2 + 
    a 6 * x ^ 6 * y ^ 3 + 
    a 5 * x ^ 5 * y ^ 4 + 
    a 4 * x ^ 4 * y ^ 5 + 
    a 3 * x ^ 3 * y ^ 6 + 
    a 2 * x ^ 2 * y ^ 7 + 
    a 1 * x * y ^ 8 + 
    a 0 * y ^ 9) :
  a 0 + a 8 = -2602 := by
  sorry

end expansion_coeff_sum_l1907_190715


namespace solve_simultaneous_eqns_l1907_190774

theorem solve_simultaneous_eqns :
  ∀ (x y : ℝ), 
  (1/x - 1/(2*y) = 2*y^4 - 2*x^4 ∧ 1/x + 1/(2*y) = (3*x^2 + y^2) * (x^2 + 3*y^2)) 
  ↔ 
  (x = (3^(1/5) + 1) / 2 ∧ y = (3^(1/5) - 1) / 2) :=
by sorry

end solve_simultaneous_eqns_l1907_190774


namespace pyramid_volume_l1907_190765

theorem pyramid_volume 
(EF FG QE : ℝ) 
(base_area : ℝ) 
(volume : ℝ)
(h1 : EF = 10)
(h2 : FG = 5)
(h3 : base_area = EF * FG)
(h4 : QE = 9)
(h5 : volume = (1 / 3) * base_area * QE) : 
volume = 150 :=
by
  simp [h1, h2, h3, h4, h5]
  sorry

end pyramid_volume_l1907_190765


namespace karen_starts_late_by_4_minutes_l1907_190793

-- Define conditions as Lean 4 variables/constants
noncomputable def karen_speed : ℝ := 60 -- in mph
noncomputable def tom_speed : ℝ := 45 -- in mph
noncomputable def tom_distance : ℝ := 24 -- in miles
noncomputable def karen_lead : ℝ := 4 -- in miles

-- Main theorem statement
theorem karen_starts_late_by_4_minutes : 
  ∃ (minutes_late : ℝ), minutes_late = 4 :=
by
  -- Calculations based on given conditions provided in the problem
  let t := tom_distance / tom_speed -- Time for Tom to drive 24 miles
  let tk := (tom_distance + karen_lead) / karen_speed -- Time for Karen to drive 28 miles
  let time_difference := t - tk -- Time difference between Tom and Karen
  let minutes_late := time_difference * 60 -- Convert time difference to minutes
  existsi minutes_late -- Existential quantifier to state the existence of such a time
  have h : minutes_late = 4 := sorry -- Placeholder for demonstrating equality
  exact h

end karen_starts_late_by_4_minutes_l1907_190793


namespace min_straight_line_cuts_l1907_190769

theorem min_straight_line_cuts (can_overlap : Prop) : 
  ∃ (cuts : ℕ), cuts = 4 ∧ 
  (∀ (square : ℕ), square = 3 →
   ∀ (unit : ℕ), unit = 1 → 
   ∀ (divided : Prop), divided = True → 
   (unit * unit) * 9 = (square * square)) :=
by
  sorry

end min_straight_line_cuts_l1907_190769
