import Mathlib

namespace NUMINAMATH_GPT_biased_coin_prob_three_heads_l237_23776

def prob_heads := 1/3

theorem biased_coin_prob_three_heads : prob_heads^3 = 1/27 :=
by
  sorry

end NUMINAMATH_GPT_biased_coin_prob_three_heads_l237_23776


namespace NUMINAMATH_GPT_find_a_b_find_tangent_line_l237_23774

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := 2 * x ^ 3 + 3 * a * x ^ 2 + 3 * b * x + 8

-- Define the derivative of the function f(x)
def f' (a b x : ℝ) : ℝ := 6 * x ^ 2 + 6 * a * x + 3 * b

-- Define the conditions for extreme values at x=1 and x=2
def extreme_conditions (a b : ℝ) : Prop :=
  f' a b 1 = 0 ∧ f' a b 2 = 0

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) (h : extreme_conditions a b) : a = -3 ∧ b = 4 :=
by sorry

-- Find the equation of the tangent line at x=0
def tangent_equation (a b : ℝ) (x y : ℝ) : Prop :=
  12 * x - y + 8 = 0

-- Prove the equation of the tangent line
theorem find_tangent_line (a b : ℝ) (h : extreme_conditions a b) : tangent_equation a b 0 8 :=
by sorry

end NUMINAMATH_GPT_find_a_b_find_tangent_line_l237_23774


namespace NUMINAMATH_GPT_age_of_15th_student_l237_23751

theorem age_of_15th_student (avg_age_all : ℝ) (avg_age_4 : ℝ) (avg_age_10 : ℝ) 
  (total_students : ℕ) (group_4_students : ℕ) (group_10_students : ℕ) 
  (h1 : avg_age_all = 15) (h2 : avg_age_4 = 14) (h3 : avg_age_10 = 16) 
  (h4 : total_students = 15) (h5 : group_4_students = 4) (h6 : group_10_students = 10) : 
  ∃ x : ℝ, x = 9 := 
by 
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l237_23751


namespace NUMINAMATH_GPT_symmetric_y_axis_l237_23711

-- Definition of a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of point symmetry with respect to the y-axis
def symmetric_about_y_axis (M : Point3D) : Point3D := 
  { x := -M.x, y := M.y, z := -M.z }

-- Theorem statement: proving the symmetry
theorem symmetric_y_axis (M : Point3D) : 
  symmetric_about_y_axis M = { x := -M.x, y := M.y, z := -M.z } := by
  sorry  -- Proof is left out as per instruction.

end NUMINAMATH_GPT_symmetric_y_axis_l237_23711


namespace NUMINAMATH_GPT_number_of_groups_l237_23734

noncomputable def original_students : ℕ := 22 + 2

def students_per_group : ℕ := 8

theorem number_of_groups : original_students / students_per_group = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_groups_l237_23734


namespace NUMINAMATH_GPT_find_a_l237_23784

def setA (a : ℤ) : Set ℤ := {a, 0}

def setB : Set ℤ := {x : ℤ | 3 * x^2 - 10 * x < 0}

theorem find_a (a : ℤ) (h : (setA a ∩ setB).Nonempty) : a = 1 ∨ a = 2 ∨ a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_l237_23784


namespace NUMINAMATH_GPT_find_x_plus_y_l237_23763

noncomputable def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

noncomputable def det2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem find_x_plus_y (x y : ℝ) (h1 : x ≠ y)
  (h2 : det3x3 2 5 10 4 x y 4 y x = 0)
  (h3 : det2x2 x y y x = 16) : x + y = 30 := by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l237_23763


namespace NUMINAMATH_GPT_marble_arrangement_count_l237_23768
noncomputable def countValidMarbleArrangements : Nat := 
  let totalArrangements := 120
  let restrictedPairsCount := 24
  totalArrangements - restrictedPairsCount

theorem marble_arrangement_count :
  countValidMarbleArrangements = 96 :=
  by
    sorry

end NUMINAMATH_GPT_marble_arrangement_count_l237_23768


namespace NUMINAMATH_GPT_min_value_fraction_l237_23719

variable (a b : ℝ)
variable (h1 : 2 * a - 2 * b + 2 = 0) -- This corresponds to a + b = 1 based on the given center (-1, 2)
variable (ha : a > 0)
variable (hb : b > 0)

theorem min_value_fraction (h1 : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (4 / a) + (1 / b) ≥ 9 :=
  sorry

end NUMINAMATH_GPT_min_value_fraction_l237_23719


namespace NUMINAMATH_GPT_geometric_sequence_sum_l237_23731

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ)
  (h_geometric : ∀ n, a (n + 1) = r * a n)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 5 + a 6 = 90 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l237_23731


namespace NUMINAMATH_GPT_coefficient_A_l237_23716

-- Definitions from the conditions
variable (A c₀ d : ℝ)
variable (h₁ : c₀ = 47)
variable (h₂ : A * c₀ + (d - 12) ^ 2 = 235)

-- The theorem to prove
theorem coefficient_A (h₁ : c₀ = 47) (h₂ : A * c₀ + (d - 12) ^ 2 = 235) : A = 5 :=
by sorry

end NUMINAMATH_GPT_coefficient_A_l237_23716


namespace NUMINAMATH_GPT_min_guests_at_banquet_l237_23786

theorem min_guests_at_banquet (total_food : ℕ) (max_food_per_guest : ℕ) : 
  total_food = 323 ∧ max_food_per_guest = 2 → 
  (∀ guests : ℕ, guests * max_food_per_guest >= total_food) → 
  (∃ g : ℕ, g = 162) :=
by
  -- Assuming total food and max food per guest
  intro h_cons
  -- Mathematical proof steps would go here, skipping with sorry
  sorry

end NUMINAMATH_GPT_min_guests_at_banquet_l237_23786


namespace NUMINAMATH_GPT_sum_of_four_triangles_l237_23738

theorem sum_of_four_triangles (x y : ℝ) (h1 : 3 * x + 2 * y = 27) (h2 : 2 * x + 3 * y = 23) : 4 * y = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_four_triangles_l237_23738


namespace NUMINAMATH_GPT_find_f_3_l237_23737

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f (x + y) = f x + f y
axiom f_4_eq_6 : f 4 = 6

theorem find_f_3 : f 3 = 9 / 2 :=
by sorry

end NUMINAMATH_GPT_find_f_3_l237_23737


namespace NUMINAMATH_GPT_evaluate_composite_function_l237_23771

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_composite_function : g (h 2) = 5288 := by
  sorry

end NUMINAMATH_GPT_evaluate_composite_function_l237_23771


namespace NUMINAMATH_GPT_proportion_solution_l237_23740

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 7 / 8) : x = 6 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_proportion_solution_l237_23740


namespace NUMINAMATH_GPT_sum_first_20_integers_l237_23714

def sum_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem sum_first_20_integers : sum_first_n_integers 20 = 210 :=
by
  -- Provided proof omitted
  sorry

end NUMINAMATH_GPT_sum_first_20_integers_l237_23714


namespace NUMINAMATH_GPT_reciprocal_of_neg_2023_l237_23745

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end NUMINAMATH_GPT_reciprocal_of_neg_2023_l237_23745


namespace NUMINAMATH_GPT_find_m_l237_23754

theorem find_m (S : ℕ → ℝ) (m : ℝ) (h : ∀ n, S n = m * 2^(n-1) - 3) : m = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l237_23754


namespace NUMINAMATH_GPT_triangle_with_angle_ratio_is_right_triangle_l237_23703

theorem triangle_with_angle_ratio_is_right_triangle (x : ℝ) (h1 : 1 * x + 2 * x + 3 * x = 180) : 
  ∃ A B C : ℝ, A = x ∧ B = 2 * x ∧ C = 3 * x ∧ (A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end NUMINAMATH_GPT_triangle_with_angle_ratio_is_right_triangle_l237_23703


namespace NUMINAMATH_GPT_cos_three_pi_over_four_l237_23742

theorem cos_three_pi_over_four :
  Real.cos (3 * Real.pi / 4) = -1 / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_three_pi_over_four_l237_23742


namespace NUMINAMATH_GPT_linda_original_amount_l237_23775

theorem linda_original_amount (L L2 : ℕ) 
  (h1 : L = 20) 
  (h2 : L - 5 = L2) : 
  L2 + 5 = 15 := 
sorry

end NUMINAMATH_GPT_linda_original_amount_l237_23775


namespace NUMINAMATH_GPT_paint_used_l237_23791

theorem paint_used (total_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) 
  (first_week_paint : ℚ) (remaining_paint : ℚ) (second_week_paint : ℚ) (total_used_paint : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1/6 →
  second_week_fraction = 1/5 →
  first_week_paint = first_week_fraction * total_paint →
  remaining_paint = total_paint - first_week_paint →
  second_week_paint = second_week_fraction * remaining_paint →
  total_used_paint = first_week_paint + second_week_paint →
  total_used_paint = 120 := sorry

end NUMINAMATH_GPT_paint_used_l237_23791


namespace NUMINAMATH_GPT_length_of_train_l237_23709

def speed_kmh : ℝ := 162
def time_seconds : ℝ := 2.222044458665529
def speed_ms : ℝ := 45  -- from conversion: 162 * (1000 / 3600)

theorem length_of_train :
  (speed_kmh * (1000 / 3600)) * time_seconds = 100 := by
  -- Proof is left out
  sorry 

end NUMINAMATH_GPT_length_of_train_l237_23709


namespace NUMINAMATH_GPT_last_digit_322_pow_369_l237_23767

theorem last_digit_322_pow_369 : (322^369) % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_last_digit_322_pow_369_l237_23767


namespace NUMINAMATH_GPT_problem_statement_l237_23788

variable (m n : ℝ)
noncomputable def sqrt_2_minus_1_inv := (Real.sqrt 2 - 1)⁻¹
noncomputable def sqrt_2_plus_1_inv := (Real.sqrt 2 + 1)⁻¹

theorem problem_statement 
  (hm : m = sqrt_2_minus_1_inv) 
  (hn : n = sqrt_2_plus_1_inv) : 
  m + n = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_problem_statement_l237_23788


namespace NUMINAMATH_GPT_condo_cats_l237_23726

theorem condo_cats (x y : ℕ) (h1 : 2 * x + y = 29) : 6 * x + 3 * y = 87 := by
  sorry

end NUMINAMATH_GPT_condo_cats_l237_23726


namespace NUMINAMATH_GPT_find_a_b_c_l237_23779

theorem find_a_b_c :
  ∃ a b c : ℕ, a = 1 ∧ b = 17 ∧ c = 2 ∧ (Nat.gcd a c = 1) ∧ a + b + c = 20 :=
by {
  -- the proof would go here
  sorry
}

end NUMINAMATH_GPT_find_a_b_c_l237_23779


namespace NUMINAMATH_GPT_third_side_triangle_max_l237_23787

theorem third_side_triangle_max (a b c : ℝ) (h1 : a = 5) (h2 : b = 10) (h3 : a + b > c) (h4 : a + c > b) (h5 : b + c > a) : c = 14 :=
by
  sorry

end NUMINAMATH_GPT_third_side_triangle_max_l237_23787


namespace NUMINAMATH_GPT_cos_A_eq_neg_quarter_l237_23725

-- Definitions of angles and sides in the triangle
variables (A B C : ℝ)
variables (a b c : ℝ)

-- Conditions from the math problem
axiom sin_arithmetic_sequence : 2 * Real.sin B = Real.sin A + Real.sin C
axiom side_relation : a = 2 * c

-- Question to be proved as Lean 4 statement
theorem cos_A_eq_neg_quarter (h1 : ∀ {x y z : ℝ}, 2 * y = x + z) 
                              (h2 : ∀ {a b c : ℝ}, a = 2 * c) : 
                              Real.cos A = -1/4 := 
sorry

end NUMINAMATH_GPT_cos_A_eq_neg_quarter_l237_23725


namespace NUMINAMATH_GPT_fraction_doubled_l237_23781

theorem fraction_doubled (x y : ℝ) (h_nonzero : x + y ≠ 0) : (4 * x^2) / (2 * (x + y)) = 2 * (x^2 / (x + y)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_doubled_l237_23781


namespace NUMINAMATH_GPT_james_collected_on_first_day_l237_23748

-- Conditions
variables (x : ℕ) -- the number of tins collected on the first day
variable (h1 : 500 = x + 3 * x + (3 * x - 50) + 4 * 50) -- total number of tins collected

-- Theorem to be proved
theorem james_collected_on_first_day : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_james_collected_on_first_day_l237_23748


namespace NUMINAMATH_GPT_highest_score_l237_23718

-- Definitions based on conditions
variable (H L : ℕ)

-- Condition (1): H - L = 150
def condition1 : Prop := H - L = 150

-- Condition (2): H + L = 208
def condition2 : Prop := H + L = 208

-- Condition (3): Total runs in 46 innings at an average of 60, excluding two innings averages to 58
def total_runs := 60 * 46
def excluded_runs := total_runs - 2552

theorem highest_score
  (cond1 : condition1 H L)
  (cond2 : condition2 H L)
  : H = 179 :=
by sorry

end NUMINAMATH_GPT_highest_score_l237_23718


namespace NUMINAMATH_GPT_graph_description_l237_23789

theorem graph_description : ∀ x y : ℝ, (x + y)^2 = 2 * (x^2 + y^2) → x = 0 ∧ y = 0 :=
by 
  sorry

end NUMINAMATH_GPT_graph_description_l237_23789


namespace NUMINAMATH_GPT_find_largest_element_l237_23794

noncomputable def increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ i j, 1 ≤ i → i < j → j ≤ 8 → a i < a j

noncomputable def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) (i : ℕ) : Prop :=
a (i+1) - a i = d ∧ a (i+2) - a (i+1) = d ∧ a (i+3) - a (i+2) = d

noncomputable def geometric_progression (a : ℕ → ℝ) (i : ℕ) : Prop :=
a (i+1) / a i = a (i+2) / a (i+1) ∧ a (i+2) / a (i+1) = a (i+3) / a (i+2)

theorem find_largest_element
  (a : ℕ → ℝ)
  (h_inc : increasing_sequence a)
  (h_ap1 : ∃ i, 1 ≤ i ∧ i ≤ 5 ∧ arithmetic_progression a 4 i)
  (h_ap2 : ∃ j, 1 ≤ j ∧ j ≤ 5 ∧ arithmetic_progression a 36 j)
  (h_gp : ∃ k, 1 ≤ k ∧ k ≤ 5 ∧ geometric_progression a k) :
  a 8 = 126 :=
sorry

end NUMINAMATH_GPT_find_largest_element_l237_23794


namespace NUMINAMATH_GPT_points_on_hyperbola_order_l237_23713

theorem points_on_hyperbola_order (k a b c : ℝ) (hk : k > 0)
  (h₁ : a = k / -2)
  (h₂ : b = k / 2)
  (h₃ : c = k / 3) :
  a < c ∧ c < b := 
sorry

end NUMINAMATH_GPT_points_on_hyperbola_order_l237_23713


namespace NUMINAMATH_GPT_percentage_passed_both_l237_23753

-- Define the percentages of failures
def percentage_failed_hindi : ℕ := 34
def percentage_failed_english : ℕ := 44
def percentage_failed_both : ℕ := 22

-- Statement to prove
theorem percentage_passed_both : 
  (100 - (percentage_failed_hindi + percentage_failed_english - percentage_failed_both)) = 44 := by
  sorry

end NUMINAMATH_GPT_percentage_passed_both_l237_23753


namespace NUMINAMATH_GPT_value_of_y_l237_23770

theorem value_of_y (x y : ℝ) (h1 : 3 * (x - y) = 18) (h2 : x + y = 20) : y = 7 := by
  sorry

end NUMINAMATH_GPT_value_of_y_l237_23770


namespace NUMINAMATH_GPT_car_production_total_l237_23712

theorem car_production_total (northAmericaCars europeCars : ℕ) (h1 : northAmericaCars = 3884) (h2 : europeCars = 2871) : northAmericaCars + europeCars = 6755 := by
  sorry

end NUMINAMATH_GPT_car_production_total_l237_23712


namespace NUMINAMATH_GPT_parallelogram_area_l237_23769

theorem parallelogram_area (base height : ℝ) (h_base : base = 10) (h_height : height = 7) :
  base * height = 70 := by
  rw [h_base, h_height]
  norm_num

end NUMINAMATH_GPT_parallelogram_area_l237_23769


namespace NUMINAMATH_GPT_probability_no_adjacent_standing_l237_23705

-- Define the problem conditions in Lean 4.
def total_outcomes := 2^10
def favorable_outcomes := 123

-- The probability is given by favorable outcomes over total outcomes.
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now state the theorem regarding the probability.
theorem probability_no_adjacent_standing : 
  probability = 123 / 1024 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_no_adjacent_standing_l237_23705


namespace NUMINAMATH_GPT_sin_two_alpha_l237_23777

theorem sin_two_alpha (alpha : ℝ) (h : Real.cos (π / 4 - alpha) = 4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_two_alpha_l237_23777


namespace NUMINAMATH_GPT_calculate_weight_of_6_moles_HClO2_l237_23722

noncomputable def weight_of_6_moles_HClO2 := 
  let molar_mass_H := 1.01
  let molar_mass_Cl := 35.45
  let molar_mass_O := 16.00
  let molar_mass_HClO2 := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  let moles_HClO2 := 6
  moles_HClO2 * molar_mass_HClO2

theorem calculate_weight_of_6_moles_HClO2 : weight_of_6_moles_HClO2 = 410.76 :=
by
  sorry

end NUMINAMATH_GPT_calculate_weight_of_6_moles_HClO2_l237_23722


namespace NUMINAMATH_GPT_solve_system_of_equations_l237_23799

theorem solve_system_of_equations
  (x y : ℝ)
  (h1 : 1 / x + 1 / (2 * y) = (x^2 + 3 * y^2) * (3 * x^2 + y^2))
  (h2 : 1 / x - 1 / (2 * y) = 2 * (y^4 - x^4)) :
  x = (3 ^ (1 / 5) + 1) / 2 ∧ y = (3 ^ (1 / 5) - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l237_23799


namespace NUMINAMATH_GPT_find_x_l237_23766

theorem find_x (x : ℤ) (h : (2 + 76 + x) / 3 = 5) : x = -63 := 
sorry

end NUMINAMATH_GPT_find_x_l237_23766


namespace NUMINAMATH_GPT_fruit_seller_original_apples_l237_23756

variable (x : ℝ)

theorem fruit_seller_original_apples (h : 0.60 * x = 420) : x = 700 := by
  sorry

end NUMINAMATH_GPT_fruit_seller_original_apples_l237_23756


namespace NUMINAMATH_GPT_collinear_points_sum_l237_23710

theorem collinear_points_sum (a b : ℝ) 
  (h_collin: ∃ k : ℝ, 
    (1 - a) / (a - a) = k * (a - b) / (b - b) ∧
    (a - a) / (2 - b) = k * (2 - 3) / (3 - 3) ∧
    (a - b) / (3 - 3) = k * (a - a) / (3 - b) ) : 
  a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_collinear_points_sum_l237_23710


namespace NUMINAMATH_GPT_proof_problem_l237_23749

variable {a b c : ℝ}

theorem proof_problem (h1 : ∀ x : ℝ, 4 * x^2 - 3 * x + 1 = a * (x - 1)^2 + b * (x - 1) + c) : 
  (4 * a + 2 * b + c = 28) := by
  -- The proof goes here. The goal statement is what we need.
  sorry

end NUMINAMATH_GPT_proof_problem_l237_23749


namespace NUMINAMATH_GPT_field_dimension_m_l237_23741

theorem field_dimension_m (m : ℝ) (h : (3 * m + 8) * (m - 3) = 80) : m = 6.057 := by
  sorry

end NUMINAMATH_GPT_field_dimension_m_l237_23741


namespace NUMINAMATH_GPT_sculpture_cost_in_inr_l237_23760

def convert_currency (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) : ℕ := 
  (n_cost / n_to_b_rate) * b_to_i_rate

theorem sculpture_cost_in_inr (n_cost : ℕ) (n_to_b_rate : ℕ) (b_to_i_rate : ℕ) :
  n_cost = 360 → 
  n_to_b_rate = 18 → 
  b_to_i_rate = 20 →
  convert_currency n_cost n_to_b_rate b_to_i_rate = 400 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- turns 360 / 18 * 20 = 400
  sorry

end NUMINAMATH_GPT_sculpture_cost_in_inr_l237_23760


namespace NUMINAMATH_GPT_sqrt_20_minus_1_range_l237_23704

theorem sqrt_20_minus_1_range : 
  16 < 20 ∧ 20 < 25 ∧ Real.sqrt 16 = 4 ∧ Real.sqrt 25 = 5 → (3 < Real.sqrt 20 - 1 ∧ Real.sqrt 20 - 1 < 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sqrt_20_minus_1_range_l237_23704


namespace NUMINAMATH_GPT_athletes_and_probability_l237_23747

-- Given conditions and parameters
def total_athletes_a := 27
def total_athletes_b := 9
def total_athletes_c := 18
def total_selected := 6
def athletes := ["A1", "A2", "A3", "A4", "A5", "A6"]

-- Definitions based on given conditions and solution steps
def selection_ratio := total_selected / (total_athletes_a + total_athletes_b + total_athletes_c)

def selected_from_a := total_athletes_a * selection_ratio
def selected_from_b := total_athletes_b * selection_ratio
def selected_from_c := total_athletes_c * selection_ratio

def pairs (l : List String) : List (String × String) :=
  (List.bind l (λ x => List.map (λ y => (x, y)) l)).filter (λ (x,y) => x < y)

def all_pairs := pairs athletes

def event_A (pair : String × String) : Bool :=
  pair.fst = "A5" ∨ pair.snd = "A5" ∨ pair.fst = "A6" ∨ pair.snd = "A6"

def favorable_event_A := all_pairs.filter event_A

noncomputable def probability_event_A := favorable_event_A.length / all_pairs.length

-- The main theorem: Number of athletes selected from each association and probability of event A
theorem athletes_and_probability : selected_from_a = 3 ∧ selected_from_b = 1 ∧ selected_from_c = 2 ∧ probability_event_A = 3/5 := by
  sorry

end NUMINAMATH_GPT_athletes_and_probability_l237_23747


namespace NUMINAMATH_GPT_heidi_more_nail_polishes_l237_23728

theorem heidi_more_nail_polishes :
  ∀ (k h r : ℕ), 
    k = 12 ->
    r = k - 4 ->
    h + r = 25 ->
    h - k = 5 :=
by
  intros k h r hk hr hr_sum
  sorry

end NUMINAMATH_GPT_heidi_more_nail_polishes_l237_23728


namespace NUMINAMATH_GPT_probability_of_winning_set_l237_23764

def winning_probability : ℚ :=
  let total_cards := 9
  let total_draws := 3
  let same_color_sets := 3
  let same_letter_sets := 3
  let total_ways_to_draw := Nat.choose total_cards total_draws
  let total_favorable_outcomes := same_color_sets + same_letter_sets
  let probability := total_favorable_outcomes / total_ways_to_draw
  probability

theorem probability_of_winning_set :
  winning_probability = 1 / 14 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_winning_set_l237_23764


namespace NUMINAMATH_GPT_whisker_relationship_l237_23706

theorem whisker_relationship :
  let P_whiskers := 14
  let C_whiskers := 22
  (C_whiskers - P_whiskers = 8) ∧ (C_whiskers / P_whiskers = 11 / 7) :=
by
  let P_whiskers := 14
  let C_whiskers := 22
  have h1 : C_whiskers - P_whiskers = 8 := by sorry
  have h2 : C_whiskers / P_whiskers = 11 / 7 := by sorry
  exact And.intro h1 h2

end NUMINAMATH_GPT_whisker_relationship_l237_23706


namespace NUMINAMATH_GPT_muffin_sum_l237_23792

theorem muffin_sum (N : ℕ) : 
  (N % 13 = 3) → 
  (N % 8 = 5) → 
  (N < 120) → 
  (N = 16 ∨ N = 81 ∨ N = 107) → 
  (16 + 81 + 107 = 204) := 
by sorry

end NUMINAMATH_GPT_muffin_sum_l237_23792


namespace NUMINAMATH_GPT_boat_equation_l237_23735

-- Define the conditions given in the problem
def total_boats : ℕ := 8
def large_boat_capacity : ℕ := 6
def small_boat_capacity : ℕ := 4
def total_students : ℕ := 38

-- Define the theorem to be proven
theorem boat_equation (x : ℕ) (h0 : x ≤ total_boats) : 
  large_boat_capacity * (total_boats - x) + small_boat_capacity * x = total_students := by
  sorry

end NUMINAMATH_GPT_boat_equation_l237_23735


namespace NUMINAMATH_GPT_murtha_total_items_at_day_10_l237_23743

-- Define terms and conditions
def num_pebbles (n : ℕ) : ℕ := n
def num_seashells (n : ℕ) : ℕ := 1 + 2 * (n - 1)

def total_pebbles (n : ℕ) : ℕ :=
  (n * (1 + n)) / 2

def total_seashells (n : ℕ) : ℕ :=
  (n * (1 + num_seashells n)) / 2

-- Define main proposition
theorem murtha_total_items_at_day_10 : total_pebbles 10 + total_seashells 10 = 155 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_murtha_total_items_at_day_10_l237_23743


namespace NUMINAMATH_GPT_original_number_solution_l237_23721

theorem original_number_solution (x : ℝ) (h : x^2 + 45 = 100) : x = Real.sqrt 55 ∨ x = -Real.sqrt 55 :=
by
  sorry

end NUMINAMATH_GPT_original_number_solution_l237_23721


namespace NUMINAMATH_GPT_knights_count_l237_23736

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end NUMINAMATH_GPT_knights_count_l237_23736


namespace NUMINAMATH_GPT_sum_first_100_even_numbers_divisible_by_6_l237_23757

-- Define the sequence of even numbers divisible by 6 between 100 and 300 inclusive.
def even_numbers_divisible_by_6 (n : ℕ) : ℕ := 102 + n * 6

-- Define the sum of the first 100 even numbers divisible by 6.
def sum_even_numbers_divisible_by_6 (k : ℕ) : ℕ := k / 2 * (102 + (102 + (k - 1) * 6))

-- Define the problem statement as a theorem.
theorem sum_first_100_even_numbers_divisible_by_6 :
  sum_even_numbers_divisible_by_6 100 = 39900 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_100_even_numbers_divisible_by_6_l237_23757


namespace NUMINAMATH_GPT_original_profit_margin_theorem_l237_23732

noncomputable def original_profit_margin (a : ℝ) (x : ℝ) (h : a > 0) : Prop := 
  (a * (1 + x) - a * (1 - 0.064)) / (a * (1 - 0.064)) = x + 0.08

theorem original_profit_margin_theorem (a : ℝ) (x : ℝ) (h : a > 0) :
  original_profit_margin a x h → x = 0.17 :=
sorry

end NUMINAMATH_GPT_original_profit_margin_theorem_l237_23732


namespace NUMINAMATH_GPT_quadratic_equation_roots_l237_23790

theorem quadratic_equation_roots {x y : ℝ}
  (h1 : x + y = 10)
  (h2 : |x - y| = 4)
  (h3 : x * y = 21) : (x - 7) * (x - 3) = 0 ∨ (x - 3) * (x - 7) = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_roots_l237_23790


namespace NUMINAMATH_GPT_factor_x_squared_minus_sixtyfour_l237_23765

theorem factor_x_squared_minus_sixtyfour (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end NUMINAMATH_GPT_factor_x_squared_minus_sixtyfour_l237_23765


namespace NUMINAMATH_GPT_smallest_number_divisible_by_18_70_100_84_increased_by_3_l237_23762

theorem smallest_number_divisible_by_18_70_100_84_increased_by_3 :
  ∃ n : ℕ, (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0 ∧ (n + 3) % 84 = 0 ∧ n = 6297 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_18_70_100_84_increased_by_3_l237_23762


namespace NUMINAMATH_GPT_sphere_intersection_circle_radius_l237_23773

theorem sphere_intersection_circle_radius
  (x1 y1 z1: ℝ) (x2 y2 z2: ℝ) (r1 r2: ℝ)
  (hyp1: x1 = 3) (hyp2: y1 = 5) (hyp3: z1 = 0) 
  (hyp4: r1 = 2) 
  (hyp5: x2 = 0) (hyp6: y2 = 5) (hyp7: z2 = -8) :
  r2 = Real.sqrt 59 := 
by
  sorry

end NUMINAMATH_GPT_sphere_intersection_circle_radius_l237_23773


namespace NUMINAMATH_GPT_angle_between_line_and_plane_l237_23750

open Real

def plane1 (x y z : ℝ) : Prop := 2*x - y - 3*z + 5 = 0
def plane2 (x y z : ℝ) : Prop := x + y - 2 = 0

def point_M : ℝ × ℝ × ℝ := (-2, 0, 3)
def point_N : ℝ × ℝ × ℝ := (0, 2, 2)
def point_K : ℝ × ℝ × ℝ := (3, -3, 1)

theorem angle_between_line_and_plane :
  ∃ α : ℝ, α = arcsin (22 / (3 * sqrt 102)) :=
by sorry

end NUMINAMATH_GPT_angle_between_line_and_plane_l237_23750


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l237_23727

theorem arithmetic_sequence_common_difference 
  (a1 a2 a3 a4 d : ℕ)
  (S : ℕ → ℕ)
  (h1 : S 2 = a1 + a2)
  (h2 : S 4 = a1 + a2 + a3 + a4)
  (h3 : S 2 = 4)
  (h4 : S 4 = 20)
  (h5 : a2 = a1 + d)
  (h6 : a3 = a2 + d)
  (h7 : a4 = a3 + d) :
  d = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l237_23727


namespace NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_2023_l237_23752

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 343 :=
sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_2023_l237_23752


namespace NUMINAMATH_GPT_find_integer_x_l237_23702

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  0 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 
  → x = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_integer_x_l237_23702


namespace NUMINAMATH_GPT_even_numbers_average_l237_23772

theorem even_numbers_average (n : ℕ) (h : (n / 2 * (2 + 2 * n)) / n = 16) : n = 15 :=
by
  have hn : n ≠ 0 := sorry -- n > 0 because the first some even numbers were mentioned
  have hn_pos : 0 < n / 2 * (2 + 2 * n) := sorry -- n / 2 * (2 + 2n) > 0
  sorry

end NUMINAMATH_GPT_even_numbers_average_l237_23772


namespace NUMINAMATH_GPT_distribution_of_balls_l237_23744

-- Definition for the problem conditions
inductive Ball : Type
| one : Ball
| two : Ball
| three : Ball
| four : Ball

inductive Box : Type
| box1 : Box
| box2 : Box
| box3 : Box

-- Function to count the number of ways to distribute the balls according to the conditions
noncomputable def num_ways_to_distribute_balls : Nat := 18

-- Theorem statement
theorem distribution_of_balls :
  num_ways_to_distribute_balls = 18 := by
  sorry

end NUMINAMATH_GPT_distribution_of_balls_l237_23744


namespace NUMINAMATH_GPT_largest_prime_divisor_of_sum_of_squares_l237_23746

def largest_prime_divisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_sum_of_squares :
  largest_prime_divisor (11^2 + 90^2) = 89 :=
by sorry

end NUMINAMATH_GPT_largest_prime_divisor_of_sum_of_squares_l237_23746


namespace NUMINAMATH_GPT_product_of_integers_l237_23723

theorem product_of_integers (x y : ℤ) (h1 : Int.gcd x y = 5) (h2 : Int.lcm x y = 60) : x * y = 300 :=
by
  sorry

end NUMINAMATH_GPT_product_of_integers_l237_23723


namespace NUMINAMATH_GPT_linear_system_k_value_l237_23759

theorem linear_system_k_value (x y k : ℝ) (h1 : x + 3 * y = 2 * k + 1) (h2 : x - y = 1) (h3 : x = -y) : k = -1 :=
sorry

end NUMINAMATH_GPT_linear_system_k_value_l237_23759


namespace NUMINAMATH_GPT_shaded_region_area_l237_23798

noncomputable def line1 (x : ℝ) : ℝ := -(3 / 10) * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -(5 / 7) * x + 47 / 7

noncomputable def intersection_x : ℝ := 17 / 5

noncomputable def area_under_curve (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem shaded_region_area : 
  area_under_curve line1 line2 0 intersection_x = 1.91 :=
sorry

end NUMINAMATH_GPT_shaded_region_area_l237_23798


namespace NUMINAMATH_GPT_find_largest_divisor_l237_23707

def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

theorem find_largest_divisor :
  ∃ m : ℕ, (∀ n : ℕ, f n % m = 0) ∧ m = 36 :=
sorry

end NUMINAMATH_GPT_find_largest_divisor_l237_23707


namespace NUMINAMATH_GPT_subtract_and_convert_l237_23717

theorem subtract_and_convert : (3/4 - 1/16 : ℚ) = 0.6875 :=
by
  sorry

end NUMINAMATH_GPT_subtract_and_convert_l237_23717


namespace NUMINAMATH_GPT_smallest_a_plus_b_l237_23782

theorem smallest_a_plus_b : ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2^3 * 3^7 * 7^2 = a^b ∧ a + b = 380 :=
sorry

end NUMINAMATH_GPT_smallest_a_plus_b_l237_23782


namespace NUMINAMATH_GPT_circle_E_radius_sum_l237_23755

noncomputable def radius_A := 15
noncomputable def radius_B := 5
noncomputable def radius_C := 3
noncomputable def radius_D := 3

-- We need to find that the sum of m and n for the radius of circle E is 131.
theorem circle_E_radius_sum (m n : ℕ) (h1 : Nat.gcd m n = 1) (radius_E : ℚ := (m / n)) :
  m + n = 131 :=
  sorry

end NUMINAMATH_GPT_circle_E_radius_sum_l237_23755


namespace NUMINAMATH_GPT_science_books_initially_l237_23729

def initial_number_of_books (borrowed left : ℕ) : ℕ := 
borrowed + left

theorem science_books_initially (borrowed left : ℕ) (h1 : borrowed = 18) (h2 : left = 57) :
initial_number_of_books borrowed left = 75 := by
sorry

end NUMINAMATH_GPT_science_books_initially_l237_23729


namespace NUMINAMATH_GPT_rectangle_length_width_difference_l237_23783

noncomputable def difference_between_length_and_width : ℝ :=
  let x := by sorry
  let y := by sorry
  (x - y)

theorem rectangle_length_width_difference {x y : ℝ}
  (h₁ : 2 * (x + y) = 20) (h₂ : x^2 + y^2 = 10^2) :
  difference_between_length_and_width = 10 :=
  by sorry

end NUMINAMATH_GPT_rectangle_length_width_difference_l237_23783


namespace NUMINAMATH_GPT_fermat_1000_units_digit_l237_23720

-- Define Fermat numbers
def FermatNumber (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1

-- Define a function to extract the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- The theorem to be proven
theorem fermat_1000_units_digit : units_digit (FermatNumber 1000) = 7 := 
by sorry

end NUMINAMATH_GPT_fermat_1000_units_digit_l237_23720


namespace NUMINAMATH_GPT_sum_of_cubes_is_24680_l237_23715

noncomputable def jake_age := 10
noncomputable def amy_age := 12
noncomputable def ryan_age := 28

theorem sum_of_cubes_is_24680 (j a r : ℕ) (h1 : 2 * j + 3 * a = 4 * r)
  (h2 : j^3 + a^3 = 1 / 2 * r^3) (h3 : j + a + r = 50) : j^3 + a^3 + r^3 = 24680 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_is_24680_l237_23715


namespace NUMINAMATH_GPT_minimum_for_specific_values_proof_minimum_for_arbitrary_values_proof_l237_23700

noncomputable def minimum_for_specific_values : ℝ :=
  let m := 2 
  let n := 2 
  let p := 2 
  let xyz := 8 
  let x := 2
  let y := 2
  let z := 2
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem minimum_for_specific_values_proof : minimum_for_specific_values = 36 := by
  sorry

noncomputable def minimum_for_arbitrary_values (m n p : ℝ) (h : m * n * p = 8) : ℝ :=
  let x := 2
  let y := 2
  let z := 2
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem minimum_for_arbitrary_values_proof (m n p : ℝ) (h : m * n * p = 8) : minimum_for_arbitrary_values m n p h = 12 + 4 * (m + n + p) := by
  sorry

end NUMINAMATH_GPT_minimum_for_specific_values_proof_minimum_for_arbitrary_values_proof_l237_23700


namespace NUMINAMATH_GPT_production_line_B_units_l237_23758

theorem production_line_B_units (total_units : ℕ) (A_units B_units C_units : ℕ) 
  (h1 : total_units = 16800)
  (h2 : ∃ d : ℕ, A_units + d = B_units ∧ B_units + d = C_units) :
  B_units = 5600 := 
sorry

end NUMINAMATH_GPT_production_line_B_units_l237_23758


namespace NUMINAMATH_GPT_basketball_free_throws_l237_23733

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 4 * a) 
  (h2 : x = 2 * a) 
  (h3 : 2 * a + 3 * b + x = 72) : 
  x = 18 := 
sorry

end NUMINAMATH_GPT_basketball_free_throws_l237_23733


namespace NUMINAMATH_GPT_binary_to_base4_conversion_l237_23796

theorem binary_to_base4_conversion :
  let b := 110110100
  let b_2 := Nat.ofDigits 2 [1, 1, 0, 1, 1, 0, 1, 0, 0]
  let b_4 := Nat.ofDigits 4 [3, 1, 2, 2, 0]
  b_2 = b → b_4 = 31220 :=
by
  intros b b_2 b_4 h
  sorry

end NUMINAMATH_GPT_binary_to_base4_conversion_l237_23796


namespace NUMINAMATH_GPT_units_digit_quotient_l237_23708

theorem units_digit_quotient (n : ℕ) (h1 : n % 2 = 1): 
  (4^n + 6^n) / 10 % 10 = 1 :=
by 
  -- Given the cyclical behavior of 4^n % 10 and 6^n % 10
  -- 4^n % 10 cycles between 4 and 6, 6^n % 10 is always 6
  -- Since n is odd, 4^n % 10 = 4 and 6^n % 10 = 6
  -- Adding them gives us 4 + 6 = 10, and thus a quotient of 1
  sorry

end NUMINAMATH_GPT_units_digit_quotient_l237_23708


namespace NUMINAMATH_GPT_volume_of_pyramid_l237_23761

/--
Rectangle ABCD is the base of pyramid PABCD. Let AB = 10, BC = 6, PA is perpendicular to AB, and PB = 20. 
If PA makes an angle θ = 30° with the diagonal AC of the base, prove the volume of the pyramid PABCD is 200 cubic units.
-/
theorem volume_of_pyramid (AB BC PB : ℝ) (θ : ℝ) (hAB : AB = 10) (hBC : BC = 6)
  (hPB : PB = 20) (hθ : θ = 30) (PA_is_perpendicular_to_AB : true) (PA_makes_angle_with_AC : true) : 
  ∃ V, V = 1 / 3 * (AB * BC) * 10 ∧ V = 200 := 
by
  exists 1 / 3 * (AB * BC) * 10
  sorry

end NUMINAMATH_GPT_volume_of_pyramid_l237_23761


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l237_23795

theorem solve_system_of_inequalities (x : ℝ) :
  ( (x - 2) / (x - 1) < 1 ) ∧ ( -x^2 + x + 2 < 0 ) → x > 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l237_23795


namespace NUMINAMATH_GPT_fiona_hoodies_l237_23739

theorem fiona_hoodies (F C : ℕ) (h1 : F + C = 8) (h2 : C = F + 2) : F = 3 :=
by
  sorry

end NUMINAMATH_GPT_fiona_hoodies_l237_23739


namespace NUMINAMATH_GPT_find_m_l237_23785

theorem find_m (m : ℤ) (h1 : -180 ≤ m ∧ m ≤ 180) (h2 : Real.sin (m * Real.pi / 180) = Real.cos (810 * Real.pi / 180)) :
  m = 0 ∨ m = 180 :=
sorry

end NUMINAMATH_GPT_find_m_l237_23785


namespace NUMINAMATH_GPT_probability_five_common_correct_l237_23793

-- Define the conditions
def compulsory_subjects : ℕ := 3  -- Chinese, Mathematics, and English
def elective_from_physics_history : ℕ := 1  -- Physics and History
def elective_from_four : ℕ := 4  -- Politics, Geography, Chemistry, Biology

def chosen_subjects_by_xiaoming_xiaofang : ℕ := 2  -- two subjects from the four electives

-- Calculate total combinations
noncomputable def total_combinations : ℕ := Nat.choose 4 2 * Nat.choose 4 2

-- Calculate combinations to have exactly five subjects in common
noncomputable def combinations_five_common : ℕ := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 2 1

-- Calculate the probability
noncomputable def probability_five_common : ℚ := combinations_five_common / total_combinations

-- The theorem to be proved
theorem probability_five_common_correct : probability_five_common = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_five_common_correct_l237_23793


namespace NUMINAMATH_GPT_sum_of_roots_l237_23778

theorem sum_of_roots (a b c : ℝ) (x1 x2 x3 : ℝ) (h_eq: 6*x1^3 + 7*x2^2 - 12*x3 = 0) :
  (x1 + x2 + x3) = -1.17 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l237_23778


namespace NUMINAMATH_GPT_time_for_B_alone_l237_23701

theorem time_for_B_alone (h1 : 4 * (1/15 + 1/x) = 7/15) : x = 20 :=
sorry

end NUMINAMATH_GPT_time_for_B_alone_l237_23701


namespace NUMINAMATH_GPT_percentage_increase_l237_23730

theorem percentage_increase (lowest_price highest_price : ℝ) (h_low : lowest_price = 15) (h_high : highest_price = 25) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 66.67 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l237_23730


namespace NUMINAMATH_GPT_min_containers_needed_l237_23797

theorem min_containers_needed 
  (total_boxes1 : ℕ) 
  (weight_box1 : ℕ) 
  (total_boxes2 : ℕ) 
  (weight_box2 : ℕ) 
  (weight_limit : ℕ) :
  total_boxes1 = 90000 →
  weight_box1 = 3300 →
  total_boxes2 = 5000 →
  weight_box2 = 200 →
  weight_limit = 100000 →
  (total_boxes1 * weight_box1 + total_boxes2 * weight_box2 + weight_limit - 1) / weight_limit = 3000 :=
by
  sorry

end NUMINAMATH_GPT_min_containers_needed_l237_23797


namespace NUMINAMATH_GPT_probability_two_queens_or_at_least_one_king_l237_23780

/-- Prove that the probability of either drawing two queens or drawing at least one king 
    when 2 cards are selected randomly from a standard deck of 52 cards is 2/13. -/
theorem probability_two_queens_or_at_least_one_king :
  (∃ (kq pk pq : ℚ), kq = 4 ∧
                     pk = 4 ∧
                     pq = 52 ∧
                     (∃ (p : ℚ), p = (kq*(kq-1))/(pq*(pq-1)) + (pk/pq)*(pq-pk)/(pq-1) + (kq*(kq-1))/(pq*(pq-1)) ∧
                            p = 2/13)) :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_two_queens_or_at_least_one_king_l237_23780


namespace NUMINAMATH_GPT_election_votes_l237_23724

theorem election_votes
  (V : ℕ)  -- total number of votes
  (candidate1_votes_percent : ℕ := 80)  -- first candidate percentage
  (second_candidate_votes : ℕ := 480)  -- votes for second candidate
  (second_candidate_percent : ℕ := 20)  -- second candidate percentage
  (h : second_candidate_votes = (second_candidate_percent * V) / 100) :
  V = 2400 :=
sorry

end NUMINAMATH_GPT_election_votes_l237_23724
