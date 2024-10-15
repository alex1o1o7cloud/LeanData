import Mathlib

namespace NUMINAMATH_GPT_probability_three_cards_l1961_196127

theorem probability_three_cards (S : Type) [Fintype S]
  (deck : Finset S) (n : ℕ) (hn : n = 52)
  (hearts : Finset S) (spades : Finset S)
  (tens: Finset S)
  (hhearts_count : ∃ k, hearts.card = k ∧ k = 13)
  (hspades_count : ∃ k, spades.card = k ∧ k = 13)
  (htens_count : ∃ k, tens.card = k ∧ k = 4)
  (hdeck_partition : ∀ x ∈ deck, x ∈ hearts ∨ x ∈ spades ∨ x ∈ tens ∨ (x ∉ hearts ∧ x ∉ spades ∧ x ∉ tens)) :
  (12 / 52 * 13 / 51 * 4 / 50 + 1 / 52 * 13 / 51 * 3 / 50 = 221 / 44200) :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_three_cards_l1961_196127


namespace NUMINAMATH_GPT_garden_length_l1961_196184

theorem garden_length (w l : ℝ) (h1: l = 2 * w) (h2 : 2 * l + 2 * w = 180) : l = 60 := 
by
  sorry

end NUMINAMATH_GPT_garden_length_l1961_196184


namespace NUMINAMATH_GPT_angle_A_is_equilateral_l1961_196181

namespace TriangleProof

variables {A B C : ℝ} {a b c : ℝ}

-- Given condition (a+b+c)(a-b-c) + 3bc = 0
def condition1 (a b c : ℝ) : Prop := (a + b + c) * (a - b - c) + 3 * b * c = 0

-- Given condition a = 2c * cos B
def condition2 (a c B : ℝ) : Prop := a = 2 * c * Real.cos B

-- Prove that if (a+b+c)(a-b-c) + 3bc = 0, then A = π / 3
theorem angle_A (h1 : condition1 a b c) : A = Real.pi / 3 :=
sorry

-- Prove that if a = 2c * cos B and A = π / 3, then ∆ ABC is an equilateral triangle
theorem is_equilateral (h2 : condition2 a c B) (hA : A = Real.pi / 3) : 
  b = c ∧ a = b ∧ B = C :=
sorry

end TriangleProof

end NUMINAMATH_GPT_angle_A_is_equilateral_l1961_196181


namespace NUMINAMATH_GPT_greatest_q_minus_r_l1961_196106

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1013 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 39) := 
by
  sorry

end NUMINAMATH_GPT_greatest_q_minus_r_l1961_196106


namespace NUMINAMATH_GPT_identify_counterfeit_bag_l1961_196122

theorem identify_counterfeit_bag (n : ℕ) (w W : ℕ) (H : ∃ k : ℕ, k ≤ n ∧ W = w * (n * (n + 1) / 2) - k) : 
  ∃ bag_num, bag_num = w * (n * (n + 1) / 2) - W := by
  sorry

end NUMINAMATH_GPT_identify_counterfeit_bag_l1961_196122


namespace NUMINAMATH_GPT_distance_between_pulley_centers_l1961_196147

theorem distance_between_pulley_centers (R1 R2 CD : ℝ) (R1_pos : R1 = 10) (R2_pos : R2 = 6) (CD_pos : CD = 30) :
  ∃ AB : ℝ, AB = 2 * Real.sqrt 229 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_pulley_centers_l1961_196147


namespace NUMINAMATH_GPT_value_of_ab_l1961_196178

theorem value_of_ab (a b : ℤ) (h1 : ∀ x : ℤ, -1 < x ∧ x < 1 → (2 * x < a + 1) ∧ (x > 2 * b + 3)) :
  (a + 1) * (b - 1) = -6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_ab_l1961_196178


namespace NUMINAMATH_GPT_angle_triple_supplement_l1961_196162

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end NUMINAMATH_GPT_angle_triple_supplement_l1961_196162


namespace NUMINAMATH_GPT_exp_arbitrarily_large_l1961_196174

theorem exp_arbitrarily_large (a : ℝ) (h : a > 1) : ∀ y > 0, ∃ x > 0, a^x > y := by
  sorry

end NUMINAMATH_GPT_exp_arbitrarily_large_l1961_196174


namespace NUMINAMATH_GPT_new_job_hourly_wage_l1961_196133

def current_job_weekly_earnings : ℝ := 8 * 10
def new_job_hours_per_week : ℝ := 4
def new_job_bonus : ℝ := 35
def new_job_expected_additional_wage : ℝ := 15

theorem new_job_hourly_wage (W : ℝ) 
  (h_current_job : current_job_weekly_earnings = 80)
  (h_new_job : new_job_hours_per_week * W + new_job_bonus = current_job_weekly_earnings + new_job_expected_additional_wage) : 
  W = 15 :=
by 
  sorry

end NUMINAMATH_GPT_new_job_hourly_wage_l1961_196133


namespace NUMINAMATH_GPT_problem_equiv_none_of_these_l1961_196148

variable {x y : ℝ}

theorem problem_equiv_none_of_these (hx : x ≠ 0) (hx3 : x ≠ 3) (hy : y ≠ 0) (hy5 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) →
  ¬(3 * x + 2 * y = x * y) ∧
  ¬(y = 3 * x / (5 - y)) ∧
  ¬(x / 3 + y / 2 = 3) ∧
  ¬(3 * y / (y - 5) = x) :=
sorry

end NUMINAMATH_GPT_problem_equiv_none_of_these_l1961_196148


namespace NUMINAMATH_GPT_find_m_value_l1961_196115

-- Definitions from conditions
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (2, -4)
def OA := (A.1 - O.1, A.2 - O.2)
def AB := (B.1 - A.1, B.2 - A.2)

-- Defining the vector OP with the given expression
def OP (m : ℝ) := (2 * OA.1 + m * AB.1, 2 * OA.2 + m * AB.2)

-- The point P is on the y-axis if the x-coordinate of OP is zero
theorem find_m_value : ∃ m : ℝ, OP m = (0, (OP m).2) ∧ m = 2 / 3 :=
by { 
  -- sorry is added to skip the proof itself
  sorry 
}

end NUMINAMATH_GPT_find_m_value_l1961_196115


namespace NUMINAMATH_GPT_bill_drew_12_triangles_l1961_196136

theorem bill_drew_12_triangles 
  (T : ℕ)
  (total_lines : T * 3 + 8 * 4 + 4 * 5 = 88) : 
  T = 12 :=
sorry

end NUMINAMATH_GPT_bill_drew_12_triangles_l1961_196136


namespace NUMINAMATH_GPT_youngest_sibling_is_42_l1961_196126

-- Definitions for the problem conditions
def consecutive_even_integers (a : ℤ) := [a, a + 2, a + 4, a + 6]
def sum_of_ages_is_180 (ages : List ℤ) := ages.sum = 180

-- Main statement
theorem youngest_sibling_is_42 (a : ℤ) 
  (h1 : sum_of_ages_is_180 (consecutive_even_integers a)) :
  a = 42 := 
sorry

end NUMINAMATH_GPT_youngest_sibling_is_42_l1961_196126


namespace NUMINAMATH_GPT_fractional_part_of_blue_square_four_changes_l1961_196149

theorem fractional_part_of_blue_square_four_changes 
  (initial_area : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ (a : ℝ), f a = (8 / 9) * a) :
  (f^[4]) initial_area / initial_area = 4096 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_fractional_part_of_blue_square_four_changes_l1961_196149


namespace NUMINAMATH_GPT_solve_quadratic_l1961_196185

theorem solve_quadratic (x : ℝ) (h : x^2 - 4 = 0) : x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_l1961_196185


namespace NUMINAMATH_GPT_total_cost_to_replace_floor_l1961_196163

def removal_cost : ℝ := 50
def cost_per_sqft : ℝ := 1.25
def room_dimensions : (ℝ × ℝ) := (8, 7)

theorem total_cost_to_replace_floor :
  removal_cost + (cost_per_sqft * (room_dimensions.1 * room_dimensions.2)) = 120 := by
  sorry

end NUMINAMATH_GPT_total_cost_to_replace_floor_l1961_196163


namespace NUMINAMATH_GPT_matrix_determinant_zero_l1961_196140

theorem matrix_determinant_zero (a b : ℝ) : 
  Matrix.det ![
    ![1, Real.sin (2 * a), Real.sin a],
    ![Real.sin (2 * a), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 := 
by 
  sorry

end NUMINAMATH_GPT_matrix_determinant_zero_l1961_196140


namespace NUMINAMATH_GPT_fraction_of_trunks_l1961_196132

theorem fraction_of_trunks (h1 : 0.38 ≤ 1) (h2 : 0.63 ≤ 1) : 
  0.63 - 0.38 = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_trunks_l1961_196132


namespace NUMINAMATH_GPT_find_n_for_integer_roots_l1961_196150

theorem find_n_for_integer_roots (n : ℤ):
    (∃ x y : ℤ, x ≠ y ∧ x^2 + (n+1)*x + (2*n - 1) = 0 ∧ y^2 + (n+1)*y + (2*n - 1) = 0) →
    (n = 1 ∨ n = 5) :=
sorry

end NUMINAMATH_GPT_find_n_for_integer_roots_l1961_196150


namespace NUMINAMATH_GPT_boys_girls_relation_l1961_196131

theorem boys_girls_relation (b g : ℕ) :
  (∃ b, 3 + (b - 1) * 2 = g) → b = (g - 1) / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_boys_girls_relation_l1961_196131


namespace NUMINAMATH_GPT_agent_takes_19_percent_l1961_196155

def agentPercentage (copies_sold : ℕ) (advance_copies : ℕ) (price_per_copy : ℕ) (steve_earnings : ℕ) : ℕ :=
  let total_earnings := copies_sold * price_per_copy
  let agent_earnings := total_earnings - steve_earnings
  let percentage_agent := 100 * agent_earnings / total_earnings
  percentage_agent

theorem agent_takes_19_percent :
  agentPercentage 1000000 100000 2 1620000 = 19 :=
by 
  sorry

end NUMINAMATH_GPT_agent_takes_19_percent_l1961_196155


namespace NUMINAMATH_GPT_g_function_ratio_l1961_196161

theorem g_function_ratio (g : ℝ → ℝ) (h : ∀ c d : ℝ, c^3 * g d = d^3 * g c) (hg3 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := 
by
  sorry

end NUMINAMATH_GPT_g_function_ratio_l1961_196161


namespace NUMINAMATH_GPT_profit_function_correct_l1961_196101

-- Definitions based on Conditions
def selling_price {R : Type*} [LinearOrderedField R] : R := 45
def profit_max {R : Type*} [LinearOrderedField R] : R := 450
def price_no_sales {R : Type*} [LinearOrderedField R] : R := 60
def quadratic_profit {R : Type*} [LinearOrderedField R] (x : R) : R := -2 * (x - 30) * (x - 60)

-- The statement we need to prove.
theorem profit_function_correct {R : Type*} [LinearOrderedField R] :
  quadratic_profit (selling_price : R) = profit_max ∧ quadratic_profit (price_no_sales : R) = 0 := 
sorry

end NUMINAMATH_GPT_profit_function_correct_l1961_196101


namespace NUMINAMATH_GPT_scientific_notation_of_number_l1961_196167

theorem scientific_notation_of_number :
  ∀ (n : ℕ), n = 450000000 -> n = 45 * 10^7 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_number_l1961_196167


namespace NUMINAMATH_GPT_cuboid_volume_l1961_196177

theorem cuboid_volume (length width height : ℕ) (h_length : length = 4) (h_width : width = 4) (h_height : height = 6) : (length * width * height = 96) :=
by 
  -- Sorry places a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_cuboid_volume_l1961_196177


namespace NUMINAMATH_GPT_tree_height_increase_l1961_196179

-- Definitions given in the conditions
def h0 : ℝ := 4
def h (t : ℕ) (x : ℝ) : ℝ := h0 + t * x

-- Proof statement
theorem tree_height_increase (x : ℝ) :
  h 6 x = (4 / 3) * h 4 x + h 4 x → x = 2 :=
by
  intro h6_eq
  rw [h, h] at h6_eq
  norm_num at h6_eq
  sorry

end NUMINAMATH_GPT_tree_height_increase_l1961_196179


namespace NUMINAMATH_GPT_probability_value_expr_is_7_l1961_196176

theorem probability_value_expr_is_7 : 
  let num_ones : ℕ := 15
  let num_ops : ℕ := 14
  let target_value : ℤ := 7
  let total_ways := 2 ^ num_ops
  let favorable_ways := (Nat.choose num_ops 11)  -- Ways to choose positions for +1's
  let prob := (favorable_ways : ℝ) / total_ways
  prob = 91 / 4096 := sorry

end NUMINAMATH_GPT_probability_value_expr_is_7_l1961_196176


namespace NUMINAMATH_GPT_ab_value_l1961_196191

theorem ab_value (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 80) : a * b = 32 := by
  sorry

end NUMINAMATH_GPT_ab_value_l1961_196191


namespace NUMINAMATH_GPT_percentage_increase_l1961_196195

def initialProductivity := 120
def totalArea := 1440
def daysInitialProductivity := 2
def daysAheadOfSchedule := 2

theorem percentage_increase :
  let originalDays := totalArea / initialProductivity
  let daysWithIncrease := originalDays - daysAheadOfSchedule
  let daysWithNewProductivity := daysWithIncrease - daysInitialProductivity
  let remainingArea := totalArea - (daysInitialProductivity * initialProductivity)
  let newProductivity := remainingArea / daysWithNewProductivity
  let increase := ((newProductivity - initialProductivity) / initialProductivity) * 100
  increase = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1961_196195


namespace NUMINAMATH_GPT_butternut_wood_figurines_l1961_196188

theorem butternut_wood_figurines (B : ℕ) (basswood_blocks : ℕ) (aspen_blocks : ℕ) (butternut_blocks : ℕ) 
  (basswood_figurines_per_block : ℕ) (aspen_figurines_per_block : ℕ) (total_figurines : ℕ) 
  (h_basswood_blocks : basswood_blocks = 15)
  (h_aspen_blocks : aspen_blocks = 20)
  (h_butternut_blocks : butternut_blocks = 20)
  (h_basswood_figurines_per_block : basswood_figurines_per_block = 3)
  (h_aspen_figurines_per_block : aspen_figurines_per_block = 2 * basswood_figurines_per_block)
  (h_total_figurines : total_figurines = 245) :
  B = 4 :=
by
  -- Definitions based on the given conditions
  let basswood_figurines := basswood_blocks * basswood_figurines_per_block
  let aspen_figurines := aspen_blocks * aspen_figurines_per_block
  let figurines_from_butternut := total_figurines - basswood_figurines - aspen_figurines
  -- Calculate the number of figurines per block of butternut wood
  let butternut_figurines_per_block := figurines_from_butternut / butternut_blocks
  -- The objective is to prove that the number of figurines per block of butternut wood is 4
  exact sorry

end NUMINAMATH_GPT_butternut_wood_figurines_l1961_196188


namespace NUMINAMATH_GPT_trig_eq_solutions_l1961_196145

theorem trig_eq_solutions (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  3 * Real.sin x = 1 + Real.cos (2 * x) ↔ x = Real.pi / 6 ∨ x = 5 * Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_trig_eq_solutions_l1961_196145


namespace NUMINAMATH_GPT_binomial_identity_l1961_196109

theorem binomial_identity (n k : ℕ) (h1 : 0 < k) (h2 : k < n)
    (h3 : Nat.choose n (k-1) + Nat.choose n (k+1) = 2 * Nat.choose n k) :
  ∃ c : ℤ, k = (c^2 + c - 2) / 2 ∧ n = c^2 - 2 := sorry

end NUMINAMATH_GPT_binomial_identity_l1961_196109


namespace NUMINAMATH_GPT_shifted_function_l1961_196159

def initial_fun (x : ℝ) : ℝ := 5 * (x - 1) ^ 2 + 1

theorem shifted_function :
  (∀ x, initial_fun (x - 2) - 3 = 5 * (x + 1) ^ 2 - 2) :=
by
  intro x
  -- sorry statement to indicate proof should be here
  sorry

end NUMINAMATH_GPT_shifted_function_l1961_196159


namespace NUMINAMATH_GPT_angle_in_fourth_quadrant_l1961_196119

-- Define the main condition converting the angle to the range [0, 360)
def reducedAngle (θ : ℤ) : ℤ := (θ % 360 + 360) % 360

-- State the theorem proving the angle of -390° is in the fourth quadrant
theorem angle_in_fourth_quadrant (θ : ℤ) (h : θ = -390) : 270 ≤ reducedAngle θ ∧ reducedAngle θ < 360 := by
  sorry

end NUMINAMATH_GPT_angle_in_fourth_quadrant_l1961_196119


namespace NUMINAMATH_GPT_quadratic_solution_difference_l1961_196186

theorem quadratic_solution_difference : 
  ∃ a b : ℝ, (a^2 - 12 * a + 20 = 0) ∧ (b^2 - 12 * b + 20 = 0) ∧ (a > b) ∧ (a - b = 8) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_difference_l1961_196186


namespace NUMINAMATH_GPT_arc_length_l1961_196160

theorem arc_length 
  (a : ℝ) 
  (α β : ℝ) 
  (hα : 0 < α) 
  (hβ : 0 < β) 
  (h1 : α + β < π) 
  :  ∃ l : ℝ, l = (a * (π - α - β) * (Real.sin α) * (Real.sin β)) / (Real.sin (α + β)) :=
sorry

end NUMINAMATH_GPT_arc_length_l1961_196160


namespace NUMINAMATH_GPT_product_of_5_consecutive_numbers_not_square_l1961_196196

-- Define what it means for a product to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating the problem
theorem product_of_5_consecutive_numbers_not_square :
  ∀ (a : ℕ), 0 < a → ¬ is_perfect_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by 
  sorry

end NUMINAMATH_GPT_product_of_5_consecutive_numbers_not_square_l1961_196196


namespace NUMINAMATH_GPT_grown_ups_in_milburg_l1961_196105

def total_population : ℕ := 8243
def number_of_children : ℕ := 2987

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 :=
by {
  sorry
}

end NUMINAMATH_GPT_grown_ups_in_milburg_l1961_196105


namespace NUMINAMATH_GPT_largest_square_side_length_l1961_196187

theorem largest_square_side_length (smallest_square_side next_square_side : ℕ) (h1 : smallest_square_side = 1) 
(h2 : next_square_side = smallest_square_side + 6) :
  ∃ x : ℕ, x = 7 :=
by
  existsi 7
  sorry

end NUMINAMATH_GPT_largest_square_side_length_l1961_196187


namespace NUMINAMATH_GPT_all_items_weight_is_8040_l1961_196104

def weight_of_all_items : Real :=
  let num_tables := 15
  let settings_per_table := 8
  let backup_percentage := 0.25

  let weight_fork := 3.5
  let weight_knife := 4.0
  let weight_spoon := 4.5
  let weight_large_plate := 14.0
  let weight_small_plate := 10.0
  let weight_wine_glass := 7.0
  let weight_water_glass := 9.0
  let weight_table_decoration := 16.0

  let total_settings := (num_tables * settings_per_table) * (1 + backup_percentage)
  let weight_per_setting := (weight_fork + weight_knife + weight_spoon) + (weight_large_plate + weight_small_plate) + (weight_wine_glass + weight_water_glass)
  let total_weight_decorations := num_tables * weight_table_decoration

  let total_weight := total_settings * weight_per_setting + total_weight_decorations
  total_weight

theorem all_items_weight_is_8040 :
  weight_of_all_items = 8040 := sorry

end NUMINAMATH_GPT_all_items_weight_is_8040_l1961_196104


namespace NUMINAMATH_GPT_material_needed_for_second_type_l1961_196116

namespace CherylProject

def first_material := 5 / 9
def leftover_material := 1 / 3
def total_material_used := 5 / 9

theorem material_needed_for_second_type :
  0.8888888888888889 - (5 / 9 : ℝ) = 0.3333333333333333 := by
  sorry

end CherylProject

end NUMINAMATH_GPT_material_needed_for_second_type_l1961_196116


namespace NUMINAMATH_GPT_work_done_in_five_days_l1961_196194

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 11
def work_rate_B : ℚ := 1 / 5
def work_rate_C : ℚ := 1 / 55

-- Define the work done in a cycle of 2 days
def work_one_cycle : ℚ := (work_rate_A + work_rate_B) + (work_rate_A + work_rate_C)

-- The total work needed to be done is 1
def total_work : ℚ := 1

-- The number of days in a cycle of 2 days
def days_per_cycle : ℕ := 2

-- Proving that the work will be done in exactly 5 days
theorem work_done_in_five_days :
  ∃ n : ℕ, n = 5 →
  n * (work_rate_A + work_rate_B) + (n-1) * (work_rate_A + work_rate_C) = total_work :=
by
  -- Sorry to skip the detailed proof steps
  sorry

end NUMINAMATH_GPT_work_done_in_five_days_l1961_196194


namespace NUMINAMATH_GPT_find_y_z_l1961_196189

def abs_diff (x y : ℝ) := abs (x - y)

noncomputable def seq_stabilize (x y z : ℝ) (n : ℕ) : Prop :=
  let x1 := abs_diff x y 
  let y1 := abs_diff y z 
  let z1 := abs_diff z x
  ∃ k : ℕ, k ≥ n ∧ abs_diff x1 y1 = x ∧ abs_diff y1 z1 = y ∧ abs_diff z1 x1 = z

theorem find_y_z (x y z : ℝ) (hx : x = 1) (hstab : ∃ n : ℕ, seq_stabilize x y z n) : y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_find_y_z_l1961_196189


namespace NUMINAMATH_GPT_factorize_expression1_factorize_expression2_l1961_196107

variable {R : Type*} [CommRing R]

theorem factorize_expression1 (x y : R) : x^2 + 2 * x + 1 - y^2 = (x + y + 1) * (x - y + 1) :=
  sorry

theorem factorize_expression2 (m n p : R) : m^2 - n^2 - 2 * n * p - p^2 = (m + n + p) * (m - n - p) :=
  sorry

end NUMINAMATH_GPT_factorize_expression1_factorize_expression2_l1961_196107


namespace NUMINAMATH_GPT_minimal_time_for_horses_l1961_196166

/-- Define the individual periods of the horses' runs -/
def periods : List ℕ := [2, 3, 4, 5, 6, 7, 9, 10]

/-- Define a function to calculate the LCM of a list of numbers -/
def lcm_list (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm 1

/-- Conjecture: proving that 60 is the minimal time until at least 6 out of 8 horses meet at the starting point -/
theorem minimal_time_for_horses : lcm_list [2, 3, 4, 5, 6, 10] = 60 :=
by
  sorry

end NUMINAMATH_GPT_minimal_time_for_horses_l1961_196166


namespace NUMINAMATH_GPT_range_of_a_l1961_196142

theorem range_of_a (f : ℝ → ℝ) (a : ℝ):
  (∀ x, f x = f (-x)) →
  (∀ x y, 0 ≤ x → x < y → f x ≤ f y) →
  (∀ x, 1/2 ≤ x ∧ x ≤ 1 → f (a * x + 1) ≤ f (x - 2)) →
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1961_196142


namespace NUMINAMATH_GPT_value_2_std_devs_below_mean_l1961_196143

theorem value_2_std_devs_below_mean {μ σ : ℝ} (h_mean : μ = 10.5) (h_std_dev : σ = 1) : μ - 2 * σ = 8.5 :=
by
  sorry

end NUMINAMATH_GPT_value_2_std_devs_below_mean_l1961_196143


namespace NUMINAMATH_GPT_cancel_terms_valid_equation_l1961_196171

theorem cancel_terms_valid_equation {m n : ℕ} 
  (x : Fin n → ℕ) (y : Fin m → ℕ) 
  (h_sum_eq : (Finset.univ.sum x) = (Finset.univ.sum y))
  (h_sum_lt : (Finset.univ.sum x) < (m * n)) : 
  ∃ x' : Fin n → ℕ, ∃ y' : Fin m → ℕ, 
    (Finset.univ.sum x' = Finset.univ.sum y') ∧ x' ≠ x ∧ y' ≠ y :=
sorry

end NUMINAMATH_GPT_cancel_terms_valid_equation_l1961_196171


namespace NUMINAMATH_GPT_fraction_to_decimal_l1961_196100

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1961_196100


namespace NUMINAMATH_GPT_cubic_and_quintic_values_l1961_196193

theorem cubic_and_quintic_values (a : ℝ) (h : (a + 1/a)^2 = 11) : 
    (a^3 + 1/a^3 = 8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = 71 * Real.sqrt 11) ∨ 
    (a^3 + 1/a^3 = -8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = -71 * Real.sqrt 11) :=
by
  sorry

end NUMINAMATH_GPT_cubic_and_quintic_values_l1961_196193


namespace NUMINAMATH_GPT_mother_l1961_196134

def problem_conditions (D M : ℤ) : Prop :=
  (2 * D + M = 70) ∧ (D + 2 * M = 95)

theorem mother's_age_is_40 (D M : ℤ) (h : problem_conditions D M) : M = 40 :=
by sorry

end NUMINAMATH_GPT_mother_l1961_196134


namespace NUMINAMATH_GPT_quadratic_discriminant_l1961_196152

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/5) (1/5) = 576 / 25 := by
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_l1961_196152


namespace NUMINAMATH_GPT_adults_attended_l1961_196164

def adult_ticket_cost : ℕ := 25
def children_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400

theorem adults_attended (A C: ℕ) (h1 : adult_ticket_cost * A + children_ticket_cost * C = total_receipts)
                       (h2 : A + C = total_attendance) : A = 120 :=
by
  sorry

end NUMINAMATH_GPT_adults_attended_l1961_196164


namespace NUMINAMATH_GPT_difference_of_digits_l1961_196117

theorem difference_of_digits (A B : ℕ) (h1 : 6 * 10 + A - (B * 10 + 2) = 36) (h2 : A ≠ B) : A - B = 5 :=
sorry

end NUMINAMATH_GPT_difference_of_digits_l1961_196117


namespace NUMINAMATH_GPT_find_m_l1961_196190

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end NUMINAMATH_GPT_find_m_l1961_196190


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1961_196156

theorem necessary_but_not_sufficient (A B : Prop) (h : A → B) : ¬ (B → A) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1961_196156


namespace NUMINAMATH_GPT_unique_solution_to_equation_l1961_196130

theorem unique_solution_to_equation (x y z : ℤ) 
    (h : 5 * x^3 + 11 * y^3 + 13 * z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_to_equation_l1961_196130


namespace NUMINAMATH_GPT_grade_more_problems_l1961_196123

theorem grade_more_problems (worksheets_total problems_per_worksheet worksheets_graded: ℕ)
  (h1 : worksheets_total = 9)
  (h2 : problems_per_worksheet = 4)
  (h3 : worksheets_graded = 5):
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 16 :=
by
  sorry

end NUMINAMATH_GPT_grade_more_problems_l1961_196123


namespace NUMINAMATH_GPT_percentage_loss_15_l1961_196170

theorem percentage_loss_15
  (sold_at_loss : ℝ)
  (sold_at_profit : ℝ)
  (percentage_profit : ℝ)
  (cost_price : ℝ)
  (percentage_loss : ℝ)
  (H1 : sold_at_loss = 12)
  (H2 : sold_at_profit = 14.823529411764707)
  (H3 : percentage_profit = 5)
  (H4 : cost_price = sold_at_profit / (1 + percentage_profit / 100))
  (H5 : percentage_loss = (cost_price - sold_at_loss) / cost_price * 100) :
  percentage_loss = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_loss_15_l1961_196170


namespace NUMINAMATH_GPT_percent_problem_l1961_196138

theorem percent_problem
  (X : ℝ)
  (h1 : 0.28 * 400 = 112)
  (h2 : 0.45 * X + 112 = 224.5) :
  X = 250 := 
sorry

end NUMINAMATH_GPT_percent_problem_l1961_196138


namespace NUMINAMATH_GPT_sum_consecutive_even_l1961_196172

theorem sum_consecutive_even (m : ℤ) : m + (m + 2) + (m + 4) + (m + 6) + (m + 8) + (m + 10) = 6 * m + 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_consecutive_even_l1961_196172


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1961_196137

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^3 - 8 > 0)) ↔ (∃ x : ℝ, x > 2 ∧ x^3 - 8 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1961_196137


namespace NUMINAMATH_GPT_odd_function_a_b_l1961_196197

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end NUMINAMATH_GPT_odd_function_a_b_l1961_196197


namespace NUMINAMATH_GPT_domain_of_f_l1961_196118

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x - 1)) / Real.sqrt (x + 1)

theorem domain_of_f :
  {x : ℝ | 2 * x - 1 > 0 ∧ x + 1 ≥ 0} = {x : ℝ | x > 1/2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1961_196118


namespace NUMINAMATH_GPT_total_squares_l1961_196124

theorem total_squares (num_groups : ℕ) (squares_per_group : ℕ) (total : ℕ) 
  (h1 : num_groups = 5) (h2 : squares_per_group = 5) (h3 : total = num_groups * squares_per_group) : 
  total = 25 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_total_squares_l1961_196124


namespace NUMINAMATH_GPT_train_length_is_150_l1961_196165

noncomputable def train_length_crossing_post (t_post : ℕ := 10) : ℕ := 10
noncomputable def train_length_crossing_platform (length_platform : ℕ := 150) (t_platform : ℕ := 20) : ℕ := 20
def train_constant_speed (L v : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) : Prop :=
  v = L / t_post ∧ v = (L + length_platform) / t_platform

theorem train_length_is_150 (L : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) (H : train_constant_speed L v t_post t_platform length_platform) : 
  L = 150 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_150_l1961_196165


namespace NUMINAMATH_GPT_episodes_per_wednesday_l1961_196102

theorem episodes_per_wednesday :
  ∀ (W : ℕ), (∃ (n_episodes : ℕ) (n_mondays : ℕ) (n_weeks : ℕ), 
    n_episodes = 201 ∧ n_mondays = 67 ∧ n_weeks = 67 
    ∧ n_weeks * W + n_mondays = n_episodes) 
    → W = 2 :=
by
  intro W
  rintro ⟨n_episodes, n_mondays, n_weeks, h1, h2, h3, h4⟩
  -- proof would go here
  sorry

end NUMINAMATH_GPT_episodes_per_wednesday_l1961_196102


namespace NUMINAMATH_GPT_sum_interior_angles_l1961_196144

theorem sum_interior_angles (n : ℕ) (h : 180 * (n - 2) = 3240) : 180 * ((n + 3) - 2) = 3780 := by
  sorry

end NUMINAMATH_GPT_sum_interior_angles_l1961_196144


namespace NUMINAMATH_GPT_knights_on_red_chairs_l1961_196125

variable (K L Kr Lb : ℕ)
variable (h1 : K + L = 20)
variable (h2 : Kr + Lb = 10)
variable (h3 : Kr = L - Lb)

/-- Given the conditions:
1. There are 20 seats with knights and liars such that K + L = 20.
2. Half of the individuals claim to be sitting on blue chairs, and half on red chairs such that Kr + Lb = 10.
3. Knights on red chairs (Kr) must be equal to liars minus liars on blue chairs (Lb).
Prove that the number of knights now sitting on red chairs is 5. -/
theorem knights_on_red_chairs : Kr = 5 :=
by
  sorry

end NUMINAMATH_GPT_knights_on_red_chairs_l1961_196125


namespace NUMINAMATH_GPT_perpendicular_vectors_m_val_l1961_196157

theorem perpendicular_vectors_m_val (m : ℝ) 
  (a : ℝ × ℝ := (-1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  m = 2 := 
by 
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_m_val_l1961_196157


namespace NUMINAMATH_GPT_total_barking_dogs_eq_l1961_196110

-- Definitions
def initial_barking_dogs : ℕ := 30
def additional_barking_dogs : ℕ := 10

-- Theorem to prove the total number of barking dogs
theorem total_barking_dogs_eq :
  initial_barking_dogs + additional_barking_dogs = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_barking_dogs_eq_l1961_196110


namespace NUMINAMATH_GPT_find_abc_l1961_196111

noncomputable def log (x : ℝ) : ℝ := sorry -- Replace sorry with an actual implementation of log function if needed

theorem find_abc (a b c : ℝ) 
    (h1 : 1 ≤ a) 
    (h2 : 1 ≤ b) 
    (h3 : 1 ≤ c)
    (h4 : a * b * c = 10)
    (h5 : a^(log a) * b^(log b) * c^(log c) ≥ 10) :
    (a = 1 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 10) := 
by
  sorry

end NUMINAMATH_GPT_find_abc_l1961_196111


namespace NUMINAMATH_GPT_missing_condition_l1961_196120

theorem missing_condition (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : y = 3 * (x - 2)) :
  true := -- The equivalent mathematical statement asserts the correct missing condition.
sorry

end NUMINAMATH_GPT_missing_condition_l1961_196120


namespace NUMINAMATH_GPT_distance_from_pointM_to_xaxis_l1961_196154

-- Define the point M with coordinates (2, -3)
def pointM : ℝ × ℝ := (2, -3)

-- Define the function to compute the distance from a point to the x-axis.
def distanceToXAxis (p : ℝ × ℝ) : ℝ := |p.2|

-- Formalize the proof statement.
theorem distance_from_pointM_to_xaxis : distanceToXAxis pointM = 3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_distance_from_pointM_to_xaxis_l1961_196154


namespace NUMINAMATH_GPT_sum_outer_equal_sum_inner_l1961_196175

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

theorem sum_outer_equal_sum_inner (M N : ℕ) (a b c d : ℕ) 
  (h1 : is_four_digit M)
  (h2 : M = 1000 * a + 100 * b + 10 * c + d) 
  (h3 : N = reverse_digits M) 
  (h4 : M + N % 101 = 0) 
  (h5 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  a + d = b + c :=
  sorry

end NUMINAMATH_GPT_sum_outer_equal_sum_inner_l1961_196175


namespace NUMINAMATH_GPT_initial_investment_calculation_l1961_196198

-- Define the conditions
def r : ℝ := 0.10
def n : ℕ := 1
def t : ℕ := 2
def A : ℝ := 6050.000000000001
def one : ℝ := 1

-- The goal is to prove that the initial principal P is 5000 under these conditions
theorem initial_investment_calculation (P : ℝ) : P = 5000 :=
by
  have interest_compounded : ℝ := (one + r / n) ^ (n * t)
  have total_amount : ℝ := P * interest_compounded
  sorry

end NUMINAMATH_GPT_initial_investment_calculation_l1961_196198


namespace NUMINAMATH_GPT_symmetric_points_on_parabola_l1961_196114

theorem symmetric_points_on_parabola {a b m n : ℝ}
  (hA : m = a^2 - 2*a - 2)
  (hB : m = b^2 - 2*b - 2)
  (hP : n = (a + b)^2 - 2*(a + b) - 2)
  (h_symmetry : (a + b) / 2 = 1) :
  n = -2 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_symmetric_points_on_parabola_l1961_196114


namespace NUMINAMATH_GPT_problem1_problem2_l1961_196135

open Real

variables {α β γ : ℝ}

theorem problem1 (α β : ℝ) :
  abs (cos (α + β)) ≤ abs (cos α) + abs (sin β) ∧
  abs (sin (α + β)) ≤ abs (cos α) + abs (cos β) :=
sorry

theorem problem2 (h : α + β + γ = 0) :
  abs (cos α) + abs (cos β) + abs (cos γ) ≥ 1 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1961_196135


namespace NUMINAMATH_GPT_rain_at_least_one_day_l1961_196182

-- Define the probabilities
def P_A1 : ℝ := 0.30
def P_A2 : ℝ := 0.40
def P_A2_given_A1 : ℝ := 0.70

-- Define complementary probabilities
def P_not_A1 : ℝ := 1 - P_A1
def P_not_A2 : ℝ := 1 - P_A2
def P_not_A2_given_A1 : ℝ := 1 - P_A2_given_A1

-- Calculate probabilities of no rain on both days under different conditions
def P_no_rain_both_days_if_no_rain_first : ℝ := P_not_A1 * P_not_A2
def P_no_rain_both_days_if_rain_first : ℝ := P_A1 * P_not_A2_given_A1

-- Total probability of no rain on both days
def P_no_rain_both_days : ℝ := P_no_rain_both_days_if_no_rain_first + P_no_rain_both_days_if_rain_first

-- Probability of rain on at least one of the two days
def P_rain_one_or_more_days : ℝ := 1 - P_no_rain_both_days

-- Expressing the result as a percentage
def result_percentage : ℝ := P_rain_one_or_more_days * 100

-- Theorem statement
theorem rain_at_least_one_day : result_percentage = 49 := by
  -- We skip the proof
  sorry

end NUMINAMATH_GPT_rain_at_least_one_day_l1961_196182


namespace NUMINAMATH_GPT_odd_and_increasing_function_l1961_196146

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f (x) ≤ f (y)

def function_D (x : ℝ) : ℝ := x * abs x

theorem odd_and_increasing_function : 
  (is_odd function_D) ∧ (is_increasing function_D) :=
sorry

end NUMINAMATH_GPT_odd_and_increasing_function_l1961_196146


namespace NUMINAMATH_GPT_proof_problem_l1961_196113

-- Definitions for the conditions and the events in the problem
def P_A : ℚ := 2 / 3
def P_B : ℚ := 1 / 4
def P_not_any_module : ℚ := 1 - (P_A + P_B)

-- Definition for the binomial coefficient
def C (n k : ℕ) := Nat.choose n k

-- Definition for the event where at least 3 out of 4 students have taken "Selected Topics in Geometric Proofs"
def P_at_least_three_taken : ℚ := 
  C 4 3 * (P_A ^ 3) * ((1 - P_A) ^ 1) + C 4 4 * (P_A ^ 4)

-- The main theorem to prove
theorem proof_problem : 
  P_not_any_module = 1 / 12 ∧ P_at_least_three_taken = 16 / 27 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1961_196113


namespace NUMINAMATH_GPT_equal_intercepts_l1961_196139

theorem equal_intercepts (a : ℝ) (h : ∃p, (a * p, 0) = (0, a - 2)) : a = 1 ∨ a = 2 :=
sorry

end NUMINAMATH_GPT_equal_intercepts_l1961_196139


namespace NUMINAMATH_GPT_sum_of_surface_areas_of_two_smaller_cuboids_l1961_196173

theorem sum_of_surface_areas_of_two_smaller_cuboids
  (L W H : ℝ) (hL : L = 3) (hW : W = 2) (hH : H = 1) :
  ∃ S, (S = 26 ∨ S = 28 ∨ S = 34) ∧ (∀ l w h, (l = L / 2 ∨ w = W / 2 ∨ h = H / 2) →
  (S = 2 * 2 * (l * W + w * H + h * L))) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_surface_areas_of_two_smaller_cuboids_l1961_196173


namespace NUMINAMATH_GPT_triangle_shape_l1961_196108

-- Define the sides of the triangle and the angles
variables {a b c : ℝ}
variables {A B C : ℝ} 
-- Assume that angles are in radians and 0 < A, B, C < π
-- Also assume that the sum of angles in the triangle is π
axiom angle_sum_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : a^2 * Real.cos A * Real.sin B = b^2 * Real.sin A * Real.cos B

-- Conclusion: The shape of triangle ABC is either isosceles or right triangle
theorem triangle_shape : 
  (A = B) ∨ (A + B = (Real.pi / 2)) := 
by sorry

end NUMINAMATH_GPT_triangle_shape_l1961_196108


namespace NUMINAMATH_GPT_sin_double_angle_l1961_196169

noncomputable def r := Real.sqrt 5
noncomputable def sin_α := -2 / r
noncomputable def cos_α := 1 / r
noncomputable def sin_2α := 2 * sin_α * cos_α

theorem sin_double_angle (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (1, -2) ∧ ∃ α : ℝ, true) → sin_2α = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1961_196169


namespace NUMINAMATH_GPT_cost_of_each_teddy_bear_is_15_l1961_196121

-- Definitions
variable (number_of_toys_cost_10 : ℕ := 28)
variable (cost_per_toy : ℕ := 10)
variable (number_of_teddy_bears : ℕ := 20)
variable (total_amount_in_wallet : ℕ := 580)

-- Theorem statement
theorem cost_of_each_teddy_bear_is_15 :
  (total_amount_in_wallet - (number_of_toys_cost_10 * cost_per_toy)) / number_of_teddy_bears = 15 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cost_of_each_teddy_bear_is_15_l1961_196121


namespace NUMINAMATH_GPT_Mike_and_Sarah_missed_days_l1961_196103

theorem Mike_and_Sarah_missed_days :
  ∀ (V M S : ℕ), V + M + S = 17 → V + M = 14 → V = 5 → M + S = 12 :=
by
  intros V M S h1 h2 h3
  sorry

end NUMINAMATH_GPT_Mike_and_Sarah_missed_days_l1961_196103


namespace NUMINAMATH_GPT_total_votes_4500_l1961_196168

theorem total_votes_4500 (V : ℝ) 
  (h : 0.60 * V - 0.40 * V = 900) : V = 4500 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_4500_l1961_196168


namespace NUMINAMATH_GPT_triangle_area_via_line_eq_l1961_196128

theorem triangle_area_via_line_eq (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  area = 1 / (2 * |a * b|) :=
by
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  sorry

end NUMINAMATH_GPT_triangle_area_via_line_eq_l1961_196128


namespace NUMINAMATH_GPT_simplify_inverse_sum_l1961_196112

variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem simplify_inverse_sum :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (a * b + a * c + b * c) :=
by sorry

end NUMINAMATH_GPT_simplify_inverse_sum_l1961_196112


namespace NUMINAMATH_GPT_loss_percentage_on_first_book_l1961_196180

variable (C1 C2 SP L : ℝ)
variable (total_cost : ℝ := 540)
variable (C1_value : ℝ := 315)
variable (gain_percentage : ℝ := 0.19)
variable (common_selling_price : ℝ := 267.75)

theorem loss_percentage_on_first_book :
  C1 = C1_value →
  C2 = total_cost - C1 →
  SP = 1.19 * C2 →
  SP = C1 - (L / 100 * C1) →
  L = 15 :=
sorry

end NUMINAMATH_GPT_loss_percentage_on_first_book_l1961_196180


namespace NUMINAMATH_GPT_equation_solution_l1961_196158

theorem equation_solution (x y : ℕ) (h : x^3 - y^3 = x * y + 61) : x = 6 ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l1961_196158


namespace NUMINAMATH_GPT_tangent_line_and_area_l1961_196141

noncomputable def tangent_line_equation (t : ℝ) : String := 
  "x + e^t * y - t - 1 = 0"

noncomputable def area_triangle_MON (t : ℝ) : ℝ :=
  (t + 1)^2 / (2 * Real.exp t)

theorem tangent_line_and_area (t : ℝ) (ht : t > 0) :
  tangent_line_equation t = "x + e^t * y - t - 1 = 0" ∧
  area_triangle_MON t = (t + 1)^2 / (2 * Real.exp t) := by
  sorry

end NUMINAMATH_GPT_tangent_line_and_area_l1961_196141


namespace NUMINAMATH_GPT_total_people_after_four_years_l1961_196192

-- Define initial conditions
def initial_total_people : Nat := 9
def board_members : Nat := 3
def regular_members_initial : Nat := initial_total_people - board_members
def years : Nat := 4

-- Define the function for regular members over the years
def regular_members (n : Nat) : Nat :=
  if n = 0 then 
    regular_members_initial
  else 
    2 * regular_members (n - 1)

theorem total_people_after_four_years :
  regular_members years = 96 := 
sorry

end NUMINAMATH_GPT_total_people_after_four_years_l1961_196192


namespace NUMINAMATH_GPT_length_of_bridge_correct_l1961_196129

noncomputable def length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) : ℝ :=
  let speed_mpm := (speed_kmh * 1000) / 60  -- Convert speed from km/hr to m/min
  speed_mpm * time_min  -- Length of the bridge in meters

theorem length_of_bridge_correct :
  length_of_bridge 10 10 = 1666.7 :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_correct_l1961_196129


namespace NUMINAMATH_GPT_base12_addition_example_l1961_196151

theorem base12_addition_example : 
  (5 * 12^2 + 2 * 12^1 + 8 * 12^0) + (2 * 12^2 + 7 * 12^1 + 3 * 12^0) = (7 * 12^2 + 9 * 12^1 + 11 * 12^0) :=
by sorry

end NUMINAMATH_GPT_base12_addition_example_l1961_196151


namespace NUMINAMATH_GPT_a_values_in_terms_of_x_l1961_196183

open Real

-- Definitions for conditions
variables (a b x y : ℝ)
variables (h1 : a^3 - b^3 = 27 * x^3)
variables (h2 : a - b = y)
variables (h3 : y = 2 * x)

-- Theorem to prove
theorem a_values_in_terms_of_x : 
  (a = x + 5 * x / sqrt 6) ∨ (a = x - 5 * x / sqrt 6) :=
sorry

end NUMINAMATH_GPT_a_values_in_terms_of_x_l1961_196183


namespace NUMINAMATH_GPT_greatest_y_value_l1961_196199

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : y ≤ 1 :=
sorry

end NUMINAMATH_GPT_greatest_y_value_l1961_196199


namespace NUMINAMATH_GPT_sequence_identical_l1961_196153

noncomputable def a (n : ℕ) : ℝ :=
  (1 / (2 * Real.sqrt 3)) * ((2 + Real.sqrt 3)^n - (2 - Real.sqrt 3)^n)

theorem sequence_identical (n : ℕ) :
  a (n + 1) = (a n + a (n + 2)) / 4 :=
by
  sorry

end NUMINAMATH_GPT_sequence_identical_l1961_196153
