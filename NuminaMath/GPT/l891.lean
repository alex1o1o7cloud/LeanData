import Mathlib

namespace NUMINAMATH_GPT_part1_part2_l891_89150

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sin x + Real.cos x)

theorem part1 : f (Real.pi / 4) = 2 := sorry

theorem part2 : ∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 → 
  (2 * Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) > 0) := sorry

end NUMINAMATH_GPT_part1_part2_l891_89150


namespace NUMINAMATH_GPT_vectors_parallel_perpendicular_l891_89155

theorem vectors_parallel_perpendicular (t t1 t2 : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
    (h_a : a = (2, t)) (h_b : b = (1, 2)) :
    ((2 * 2 = t * 1) → t1 = 4) ∧ ((2 * 1 + 2 * t = 0) → t2 = -1) :=
by 
  sorry

end NUMINAMATH_GPT_vectors_parallel_perpendicular_l891_89155


namespace NUMINAMATH_GPT_henry_socks_l891_89132

theorem henry_socks : 
  ∃ a b c : ℕ, 
    a + b + c = 15 ∧ 
    2 * a + 3 * b + 5 * c = 36 ∧ 
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ 
    a = 11 :=
by
  sorry

end NUMINAMATH_GPT_henry_socks_l891_89132


namespace NUMINAMATH_GPT_product_of_repeating_decimal_l891_89105

-- Define the repeating decimal 0.3
def repeating_decimal : ℚ := 1 / 3
-- Define the question
def product (a b : ℚ) := a * b

-- State the theorem to be proved
theorem product_of_repeating_decimal :
  product repeating_decimal 8 = 8 / 3 :=
sorry

end NUMINAMATH_GPT_product_of_repeating_decimal_l891_89105


namespace NUMINAMATH_GPT_integer_solutions_l891_89173

theorem integer_solutions (t : ℤ) : 
  ∃ x y : ℤ, 5 * x - 7 * y = 3 ∧ x = 7 * t - 12 ∧ y = 5 * t - 9 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l891_89173


namespace NUMINAMATH_GPT_tangent_sum_half_angles_l891_89156

-- Lean statement for the proof problem
theorem tangent_sum_half_angles (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.tan (A / 2) * Real.tan (B / 2) + 
  Real.tan (B / 2) * Real.tan (C / 2) + 
  Real.tan (C / 2) * Real.tan (A / 2) = 1 := 
by
  sorry

end NUMINAMATH_GPT_tangent_sum_half_angles_l891_89156


namespace NUMINAMATH_GPT_maximum_area_rhombus_l891_89180

theorem maximum_area_rhombus 
    (x₀ y₀ k : ℝ)
    (h1 : 2 ≤ x₀ ∧ x₀ ≤ 4)
    (h2 : y₀ = k / x₀)
    (h3 : ∀ x > 0, ∃ y, y = k / x) :
    (∀ (x₀ : ℝ), 2 ≤ x₀ ∧ x₀ ≤ 4 → ∃ (S : ℝ), S = 3 * (Real.sqrt 2 / 2 * x₀^2) → S ≤ 24 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_rhombus_l891_89180


namespace NUMINAMATH_GPT_heavy_operators_earn_129_dollars_per_day_l891_89146

noncomputable def heavy_operator_daily_wage (H : ℕ) : Prop :=
  let laborer_wage := 82
  let total_people := 31
  let total_payroll := 3952
  let laborers_count := 1
  let heavy_operators_count := total_people - laborers_count
  let heavy_operators_payroll := total_payroll - (laborer_wage * laborers_count)
  H = heavy_operators_payroll / heavy_operators_count

theorem heavy_operators_earn_129_dollars_per_day : heavy_operator_daily_wage 129 :=
by
  unfold heavy_operator_daily_wage
  sorry

end NUMINAMATH_GPT_heavy_operators_earn_129_dollars_per_day_l891_89146


namespace NUMINAMATH_GPT_theresa_needs_15_hours_l891_89125

theorem theresa_needs_15_hours 
  (h1 : ℕ) (h2 : ℕ) (h3 : ℕ) (h4 : ℕ) (h5 : ℕ) (average : ℕ) (weeks : ℕ) (total_hours_first_5 : ℕ) :
  h1 = 10 → h2 = 13 → h3 = 9 → h4 = 14 → h5 = 11 → average = 12 → weeks = 6 → 
  total_hours_first_5 = h1 + h2 + h3 + h4 + h5 → 
  (total_hours_first_5 + x) / weeks = average → x = 15 :=
by
  intros h1_eq h2_eq h3_eq h4_eq h5_eq avg_eq weeks_eq sum_eq avg_eqn
  sorry

end NUMINAMATH_GPT_theresa_needs_15_hours_l891_89125


namespace NUMINAMATH_GPT_algebraic_expression_l891_89157

theorem algebraic_expression (m : ℝ) (hm : m^2 + m - 1 = 0) : 
  m^3 + 2 * m^2 + 2014 = 2015 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_l891_89157


namespace NUMINAMATH_GPT_charlie_share_l891_89183

theorem charlie_share (A B C D E : ℝ) (h1 : A = (1/3) * B)
  (h2 : B = (1/2) * C) (h3 : C = 0.75 * D) (h4 : D = 2 * E) 
  (h5 : A + B + C + D + E = 15000) : C = 15000 * (3 / 11) :=
by
  sorry

end NUMINAMATH_GPT_charlie_share_l891_89183


namespace NUMINAMATH_GPT_walt_total_interest_l891_89184

noncomputable def interest_8_percent (P_8 R_8 : ℝ) : ℝ :=
  P_8 * R_8

noncomputable def remaining_amount (P_total P_8 : ℝ) : ℝ :=
  P_total - P_8

noncomputable def interest_9_percent (P_9 R_9 : ℝ) : ℝ :=
  P_9 * R_9

noncomputable def total_interest (I_8 I_9 : ℝ) : ℝ :=
  I_8 + I_9

theorem walt_total_interest :
  let P_8 := 4000
  let R_8 := 0.08
  let P_total := 9000
  let R_9 := 0.09
  let I_8 := interest_8_percent P_8 R_8
  let P_9 := remaining_amount P_total P_8
  let I_9 := interest_9_percent P_9 R_9
  let I_total := total_interest I_8 I_9
  I_total = 770 := 
by
  sorry

end NUMINAMATH_GPT_walt_total_interest_l891_89184


namespace NUMINAMATH_GPT_megan_folders_l891_89182

theorem megan_folders (initial_files deleted_files files_per_folder : ℕ) (h1 : initial_files = 237)
    (h2 : deleted_files = 53) (h3 : files_per_folder = 12) :
    let remaining_files := initial_files - deleted_files
    let total_folders := (remaining_files / files_per_folder) + 1
    total_folders = 16 := 
by
  sorry

end NUMINAMATH_GPT_megan_folders_l891_89182


namespace NUMINAMATH_GPT_number_of_hens_l891_89197

theorem number_of_hens (H C : ℕ) (h1 : H + C = 44) (h2 : 2 * H + 4 * C = 128) : H = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_hens_l891_89197


namespace NUMINAMATH_GPT_sphere_volume_proof_l891_89104

noncomputable def sphereVolume (d : ℝ) (S : ℝ) : ℝ :=
  let r := Real.sqrt (S / Real.pi)
  let R := Real.sqrt (r^2 + d^2)
  (4 / 3) * Real.pi * R^3

theorem sphere_volume_proof : sphereVolume 1 (2 * Real.pi) = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_proof_l891_89104


namespace NUMINAMATH_GPT_combination_problem_l891_89171

theorem combination_problem (x : ℕ) (hx_pos : 0 < x) (h_comb : Nat.choose 9 x = Nat.choose 9 (2 * x + 3)) : x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_combination_problem_l891_89171


namespace NUMINAMATH_GPT_asterisk_replacement_l891_89109

theorem asterisk_replacement (x : ℝ) : 
  (x / 20) * (x / 80) = 1 ↔ x = 40 :=
by sorry

end NUMINAMATH_GPT_asterisk_replacement_l891_89109


namespace NUMINAMATH_GPT_number_of_routes_l891_89112

open Nat

theorem number_of_routes (south_cities north_cities : ℕ) 
  (connections : south_cities = 4 ∧ north_cities = 5) : 
  ∃ routes, routes = (factorial 3) * (5 ^ 4) := 
by
  sorry

end NUMINAMATH_GPT_number_of_routes_l891_89112


namespace NUMINAMATH_GPT_boys_at_beginning_is_15_l891_89116

noncomputable def number_of_boys_at_beginning (B : ℝ) : Prop :=
  let girls_start := 1.20 * B
  let girls_end := 2 * girls_start
  let total_students := B + girls_end
  total_students = 51 

theorem boys_at_beginning_is_15 : number_of_boys_at_beginning 15 := 
  by
  -- Sorry is added to skip the proof
  sorry

end NUMINAMATH_GPT_boys_at_beginning_is_15_l891_89116


namespace NUMINAMATH_GPT_max_value_of_trig_function_l891_89195

theorem max_value_of_trig_function : 
  ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 5 := sorry


end NUMINAMATH_GPT_max_value_of_trig_function_l891_89195


namespace NUMINAMATH_GPT_divisibility_by_six_l891_89131

theorem divisibility_by_six (a x: ℤ) : ∃ t: ℤ, x = 3 * t ∨ x = 3 * t - a^2 → 6 ∣ a * (x^3 + a^2 * x^2 + a^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_by_six_l891_89131


namespace NUMINAMATH_GPT_smallest_sum_symmetrical_dice_l891_89126

theorem smallest_sum_symmetrical_dice (p : ℝ) (N : ℕ) (h₁ : p > 0) (h₂ : 6 * N = 2022) : N = 337 := 
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_smallest_sum_symmetrical_dice_l891_89126


namespace NUMINAMATH_GPT_smallest_class_number_selected_l891_89124

theorem smallest_class_number_selected
  {n k : ℕ} (hn : n = 30) (hk : k = 5) (h_sum : ∃ x : ℕ, x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 75) :
  ∃ x : ℕ, x = 3 := 
sorry

end NUMINAMATH_GPT_smallest_class_number_selected_l891_89124


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_div4_sin2alpha_over_expr_l891_89111

variables (α : ℝ) (h : Real.tan α = 3)

-- Problem 1
theorem tan_alpha_plus_pi_div4 : Real.tan (α + π / 4) = -2 :=
by
  sorry

-- Problem 2
theorem sin2alpha_over_expr : (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_div4_sin2alpha_over_expr_l891_89111


namespace NUMINAMATH_GPT_tree_growth_factor_l891_89186

theorem tree_growth_factor 
  (initial_total : ℕ) 
  (initial_maples : ℕ) 
  (initial_lindens : ℕ) 
  (spring_total : ℕ) 
  (autumn_total : ℕ)
  (initial_maple_percentage : initial_maples = 3 * initial_total / 5)
  (spring_maple_percentage : initial_maples = spring_total / 5)
  (autumn_maple_percentage : initial_maples * 2 = autumn_total * 3 / 5) :
  autumn_total = 6 * initial_total :=
sorry

end NUMINAMATH_GPT_tree_growth_factor_l891_89186


namespace NUMINAMATH_GPT_largest_4_digit_divisible_by_35_l891_89199

theorem largest_4_digit_divisible_by_35 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 35 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 35 = 0) → m ≤ n) ∧ n = 9985 := 
by sorry

end NUMINAMATH_GPT_largest_4_digit_divisible_by_35_l891_89199


namespace NUMINAMATH_GPT_fraction_division_l891_89163

theorem fraction_division (a b c d e : ℚ)
  (h1 : a = 3 / 7)
  (h2 : b = 1 / 3)
  (h3 : d = 2 / 5)
  (h4 : c = a + b)
  (h5 : e = c / d):
  e = 40 / 21 := by
  sorry

end NUMINAMATH_GPT_fraction_division_l891_89163


namespace NUMINAMATH_GPT_ratio_meerkats_to_lion_cubs_l891_89187

-- Defining the initial conditions 
def initial_animals : ℕ := 68
def gorillas_sent : ℕ := 6
def hippo_adopted : ℕ := 1
def rhinos_rescued : ℕ := 3
def lion_cubs : ℕ := 8
def final_animal_count : ℕ := 90

-- Calculating the number of meerkats
def animals_before_meerkats : ℕ := initial_animals - gorillas_sent + hippo_adopted + rhinos_rescued + lion_cubs
def meerkats : ℕ := final_animal_count - animals_before_meerkats

-- Proving the ratio of meerkats to lion cubs is 2:1
theorem ratio_meerkats_to_lion_cubs : meerkats / lion_cubs = 2 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_meerkats_to_lion_cubs_l891_89187


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_sequence_l891_89162

-- Arithmetic sequence proof
theorem arithmetic_sequence (d n : ℕ) (a_n a_1 : ℤ) (s_n : ℤ) :
  d = 2 → n = 15 → a_n = -10 →
  a_1 = -38 ∧ s_n = -360 :=
sorry

-- Geometric sequence proof
theorem geometric_sequence (a_1 a_4 q s_3 : ℤ) :
  a_1 = -1 → a_4 = 64 →
  q = -4 ∧ s_3 = -13 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_sequence_l891_89162


namespace NUMINAMATH_GPT_list_price_of_article_l891_89188

theorem list_price_of_article (P : ℝ) 
  (first_discount second_discount final_price : ℝ)
  (h1 : first_discount = 0.10)
  (h2 : second_discount = 0.08235294117647069)
  (h3 : final_price = 56.16)
  (h4 : P * (1 - first_discount) * (1 - second_discount) = final_price) : P = 68 :=
sorry

end NUMINAMATH_GPT_list_price_of_article_l891_89188


namespace NUMINAMATH_GPT_Laura_pays_more_l891_89108

theorem Laura_pays_more 
  (slices : ℕ) 
  (cost_plain : ℝ) 
  (cost_mushrooms : ℝ) 
  (laura_mushroom_slices : ℕ) 
  (laura_plain_slices : ℕ) 
  (jessica_plain_slices: ℕ) :
  slices = 12 →
  cost_plain = 12 →
  cost_mushrooms = 3 →
  laura_mushroom_slices = 4 →
  laura_plain_slices = 2 →
  jessica_plain_slices = 6 →
  15 / 12 * (laura_mushroom_slices + laura_plain_slices) - 
  (cost_plain / 12 * jessica_plain_slices) = 1.5 :=
by
  intro slices_eq
  intro cost_plain_eq
  intro cost_mushrooms_eq
  intro laura_mushroom_slices_eq
  intro laura_plain_slices_eq
  intro jessica_plain_slices_eq
  sorry

end NUMINAMATH_GPT_Laura_pays_more_l891_89108


namespace NUMINAMATH_GPT_martha_clothes_total_l891_89135

-- Given conditions
def jackets_bought : Nat := 4
def t_shirts_bought : Nat := 9
def free_jacket_ratio : Nat := 2
def free_t_shirt_ratio : Nat := 3

-- Problem statement to prove
theorem martha_clothes_total :
  (jackets_bought + jackets_bought / free_jacket_ratio) + 
  (t_shirts_bought + t_shirts_bought / free_t_shirt_ratio) = 18 := 
by 
  sorry

end NUMINAMATH_GPT_martha_clothes_total_l891_89135


namespace NUMINAMATH_GPT_B_pow_101_l891_89164

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ]

theorem B_pow_101 :
  B ^ 101 = ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ] :=
  sorry

end NUMINAMATH_GPT_B_pow_101_l891_89164


namespace NUMINAMATH_GPT_a_minus_2_values_l891_89168

theorem a_minus_2_values (a : ℝ) (h : |a| = 3) : a - 2 = 1 ∨ a - 2 = -5 :=
by {
  -- the theorem states that given the absolute value condition, a - 2 can be 1 or -5
  sorry
}

end NUMINAMATH_GPT_a_minus_2_values_l891_89168


namespace NUMINAMATH_GPT_sum_of_quotient_and_remainder_is_184_l891_89151

theorem sum_of_quotient_and_remainder_is_184 
  (q r : ℕ)
  (h1 : 23 * 17 + 19 = q)
  (h2 : q * 10 = r)
  (h3 : r / 23 = 178)
  (h4 : r % 23 = 6) :
  178 + 6 = 184 :=
by
  -- Inform Lean that we are skipping the proof
  sorry

end NUMINAMATH_GPT_sum_of_quotient_and_remainder_is_184_l891_89151


namespace NUMINAMATH_GPT_cars_in_garage_l891_89191

theorem cars_in_garage (c : ℕ) 
  (bicycles : ℕ := 20) 
  (motorcycles : ℕ := 5) 
  (total_wheels : ℕ := 90) 
  (bicycle_wheels : ℕ := 2 * bicycles)
  (motorcycle_wheels : ℕ := 2 * motorcycles)
  (car_wheels : ℕ := 4 * c) 
  (eq : bicycle_wheels + car_wheels + motorcycle_wheels = total_wheels) : 
  c = 10 := 
by 
  sorry

end NUMINAMATH_GPT_cars_in_garage_l891_89191


namespace NUMINAMATH_GPT_binomial_10_3_l891_89110

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end NUMINAMATH_GPT_binomial_10_3_l891_89110


namespace NUMINAMATH_GPT_tip_percentage_l891_89148

theorem tip_percentage (cost_of_crown : ℕ) (total_paid : ℕ) (h1 : cost_of_crown = 20000) (h2 : total_paid = 22000) :
  (total_paid - cost_of_crown) * 100 / cost_of_crown = 10 :=
by
  sorry

end NUMINAMATH_GPT_tip_percentage_l891_89148


namespace NUMINAMATH_GPT_exponents_divisible_by_8_l891_89137

theorem exponents_divisible_by_8 (n : ℕ) : 8 ∣ (3^(4 * n + 1) + 5^(2 * n + 1)) :=
by
-- Base case and inductive step will be defined here.
sorry

end NUMINAMATH_GPT_exponents_divisible_by_8_l891_89137


namespace NUMINAMATH_GPT_mod_last_digit_l891_89174

theorem mod_last_digit (N : ℕ) (a b : ℕ) (h : N = 10 * a + b) (hb : b < 10) : 
  N % 10 = b ∧ N % 2 = b % 2 ∧ N % 5 = b % 5 :=
by
  sorry

end NUMINAMATH_GPT_mod_last_digit_l891_89174


namespace NUMINAMATH_GPT_john_initial_pairs_9_l891_89167

-- Definitions based on the conditions in the problem

def john_initial_pairs (x : ℕ) := 2 * x   -- Each pair consists of 2 socks

def john_remaining_socks (x : ℕ) := john_initial_pairs x - 5   -- John loses 5 individual socks

def john_max_pairs_left := 7
def john_minimum_socks_required := john_max_pairs_left * 2  -- 7 pairs mean he needs 14 socks

-- Theorem statement proving John initially had 9 pairs of socks
theorem john_initial_pairs_9 : 
  ∀ (x : ℕ), john_remaining_socks x ≥ john_minimum_socks_required → x = 9 := by
  sorry

end NUMINAMATH_GPT_john_initial_pairs_9_l891_89167


namespace NUMINAMATH_GPT_min_value_expr_l891_89179

-- Define the given expression
def given_expr (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x) + 200

-- Define the minimum value we need to prove
def min_value : ℝ :=
  -6290.25

-- The statement of the theorem
theorem min_value_expr :
  ∃ x : ℝ, ∀ y : ℝ, given_expr y ≥ min_value := by
  sorry

end NUMINAMATH_GPT_min_value_expr_l891_89179


namespace NUMINAMATH_GPT_inequality_abc_geq_36_l891_89138

theorem inequality_abc_geq_36 (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) (h_prod : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_geq_36_l891_89138


namespace NUMINAMATH_GPT_isosceles_triangle_sides_l891_89122

theorem isosceles_triangle_sides (a b c : ℝ) (h_iso : a = b ∨ b = c ∨ c = a) (h_perimeter : a + b + c = 14) (h_side : a = 4 ∨ b = 4 ∨ c = 4) : 
  (a = 4 ∧ b = 5 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 6) ∨ (a = 4 ∧ b = 6 ∧ c = 4) :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_l891_89122


namespace NUMINAMATH_GPT_ratio_proof_l891_89139

variables {F : Type*} [Field F] 
variables (w x y z : F)

theorem ratio_proof 
  (h1 : w / x = 4 / 3) 
  (h2 : y / z = 3 / 2) 
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_proof_l891_89139


namespace NUMINAMATH_GPT_cone_shape_in_spherical_coordinates_l891_89140

-- Define the conditions as given in the problem
def spherical_coordinates (rho theta phi c : ℝ) : Prop := 
  rho = c * Real.sin phi

-- Define the main statement to prove
theorem cone_shape_in_spherical_coordinates (rho theta phi c : ℝ) (hpos : 0 < c) :
  spherical_coordinates rho theta phi c → 
  ∃ cone : Prop, cone :=
sorry

end NUMINAMATH_GPT_cone_shape_in_spherical_coordinates_l891_89140


namespace NUMINAMATH_GPT_curve_is_line_l891_89153

theorem curve_is_line (r θ x y : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x + y = 1 := by
  sorry

end NUMINAMATH_GPT_curve_is_line_l891_89153


namespace NUMINAMATH_GPT_passenger_catches_bus_l891_89100

-- Definitions based on conditions from part a)
def P_route3 := 0.20
def P_route6 := 0.60

-- Statement to prove based on part c)
theorem passenger_catches_bus : 
  P_route3 + P_route6 = 0.80 := 
by
  sorry

end NUMINAMATH_GPT_passenger_catches_bus_l891_89100


namespace NUMINAMATH_GPT_cube_inequality_contradiction_l891_89118

theorem cube_inequality_contradiction (a b : Real) (h : a > b) : ¬(a^3 <= b^3) := by
  sorry

end NUMINAMATH_GPT_cube_inequality_contradiction_l891_89118


namespace NUMINAMATH_GPT_lena_more_than_nicole_l891_89130

theorem lena_more_than_nicole (L K N : ℕ) 
  (h1 : L = 23)
  (h2 : 4 * K = L + 7)
  (h3 : K = N - 6) : L - N = 10 := sorry

end NUMINAMATH_GPT_lena_more_than_nicole_l891_89130


namespace NUMINAMATH_GPT_transformed_parabola_is_correct_l891_89147

-- Definitions based on conditions
def original_parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 3
def shifted_left (x : ℝ) : ℝ := original_parabola (x - 2)
def shifted_up (y : ℝ) : ℝ := y + 2

-- Theorem statement
theorem transformed_parabola_is_correct :
  ∀ x : ℝ, shifted_up (shifted_left x) = 3 * x^2 + 6 * x - 1 :=
by 
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_transformed_parabola_is_correct_l891_89147


namespace NUMINAMATH_GPT_factor_polynomial_l891_89128

theorem factor_polynomial (x : ℝ) : 
    54 * x^4 - 135 * x^8 = -27 * x^4 * (5 * x^4 - 2) := 
by 
  sorry

end NUMINAMATH_GPT_factor_polynomial_l891_89128


namespace NUMINAMATH_GPT_Jane_exercises_days_per_week_l891_89142

theorem Jane_exercises_days_per_week 
  (goal_hours_per_day : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (exercise_days_per_week : ℕ) 
  (h_goal : goal_hours_per_day = 1)
  (h_weeks : weeks = 8)
  (h_total_hours : total_hours = 40)
  (h_exercise_hours_weekly : total_hours / weeks = exercise_days_per_week) :
  exercise_days_per_week = 5 :=
by
  sorry

end NUMINAMATH_GPT_Jane_exercises_days_per_week_l891_89142


namespace NUMINAMATH_GPT_focus_of_parabola_l891_89194

-- Define the equation of the given parabola
def given_parabola (x y : ℝ) : Prop := y = - (1 / 8) * x^2

-- Define the condition for the focus of the parabola
def is_focus (focus : ℝ × ℝ) : Prop := focus = (0, -2)

-- State the theorem
theorem focus_of_parabola : ∃ (focus : ℝ × ℝ), given_parabola x y → is_focus focus :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l891_89194


namespace NUMINAMATH_GPT_sum_of_first_50_digits_of_one_over_1234_l891_89101

def first_n_digits_sum (x : ℚ) (n : ℕ) : ℕ :=
  sorry  -- This function should compute the sum of the first n digits after the decimal point of x

theorem sum_of_first_50_digits_of_one_over_1234 :
  first_n_digits_sum (1/1234) 50 = 275 :=
sorry

end NUMINAMATH_GPT_sum_of_first_50_digits_of_one_over_1234_l891_89101


namespace NUMINAMATH_GPT_speed_ratio_l891_89123

theorem speed_ratio (v1 v2 : ℝ) 
  (h1 : v1 > 0) 
  (h2 : v2 > 0) 
  (h : v2 / v1 - v1 / v2 = 35 / 60) : v1 / v2 = 3 / 4 := 
sorry

end NUMINAMATH_GPT_speed_ratio_l891_89123


namespace NUMINAMATH_GPT_polynomial_roots_sum_l891_89158

theorem polynomial_roots_sum (a b c d e : ℝ) (h₀ : a ≠ 0)
  (h1 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
  (h2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h3 : a * 2^4 + b * 2^3 + c * 2^2 + d * 2 + e = 0) :
  (b + d) / a = -2677 := 
sorry

end NUMINAMATH_GPT_polynomial_roots_sum_l891_89158


namespace NUMINAMATH_GPT_find_width_l891_89190

-- Definition of the perimeter of a rectangle
def perimeter (L W : ℝ) : ℝ := 2 * (L + W)

-- The given conditions
def length := 13
def perimeter_value := 50

-- The goal to prove: if the perimeter is 50 and the length is 13, then the width must be 12
theorem find_width :
  ∃ (W : ℝ), perimeter length W = perimeter_value ∧ W = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_width_l891_89190


namespace NUMINAMATH_GPT_points_lie_on_line_l891_89175

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  x + y = 4 :=
by
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  sorry

end NUMINAMATH_GPT_points_lie_on_line_l891_89175


namespace NUMINAMATH_GPT_second_option_feasible_l891_89106

def Individual : Type := String
def M : Individual := "M"
def I : Individual := "I"
def P : Individual := "P"
def A : Individual := "A"

variable (is_sitting : Individual → Prop)

-- Given conditions
axiom fact1 : ¬ is_sitting M
axiom fact2 : ¬ is_sitting A
axiom fact3 : ¬ is_sitting M → is_sitting I
axiom fact4 : is_sitting I → is_sitting P

theorem second_option_feasible :
  is_sitting I ∧ is_sitting P ∧ ¬ is_sitting M ∧ ¬ is_sitting A :=
by
  sorry

end NUMINAMATH_GPT_second_option_feasible_l891_89106


namespace NUMINAMATH_GPT_circle_diameter_l891_89170

theorem circle_diameter (r : ℝ) (h : π * r^2 = 16 * π) : 2 * r = 8 :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_l891_89170


namespace NUMINAMATH_GPT_graph_intersects_x_axis_once_l891_89141

noncomputable def f (m x : ℝ) : ℝ := (m - 1) * x^2 - 6 * x + (3 / 2) * m

theorem graph_intersects_x_axis_once (m : ℝ) :
  (∃ x : ℝ, f m x = 0 ∧ ∀ y : ℝ, f m y = 0 → y = x) ↔ (m = 1 ∨ m = 3 ∨ m = -2) :=
by
  sorry

end NUMINAMATH_GPT_graph_intersects_x_axis_once_l891_89141


namespace NUMINAMATH_GPT_psychology_majors_percentage_in_liberal_arts_l891_89165

theorem psychology_majors_percentage_in_liberal_arts 
  (total_students : ℕ) 
  (percent_freshmen : ℝ) 
  (percent_freshmen_liberal_arts : ℝ) 
  (percent_freshmen_psych_majors_liberal_arts : ℝ) 
  (h1: percent_freshmen = 0.40) 
  (h2: percent_freshmen_liberal_arts = 0.50)
  (h3: percent_freshmen_psych_majors_liberal_arts = 0.10) :
  ((percent_freshmen_psych_majors_liberal_arts / (percent_freshmen * percent_freshmen_liberal_arts)) * 100 = 50) :=
by
  sorry

end NUMINAMATH_GPT_psychology_majors_percentage_in_liberal_arts_l891_89165


namespace NUMINAMATH_GPT_find_N_l891_89133

def consecutive_product_sum_condition (a : ℕ) : Prop :=
  a*(a + 1)*(a + 2) = 8*(a + (a + 1) + (a + 2))

theorem find_N : ∃ (N : ℕ), N = 120 ∧ ∃ (a : ℕ), a > 0 ∧ consecutive_product_sum_condition a := by
  sorry

end NUMINAMATH_GPT_find_N_l891_89133


namespace NUMINAMATH_GPT_total_cost_of_rolls_l891_89166

-- Defining the conditions
def price_per_dozen : ℕ := 5
def total_rolls_bought : ℕ := 36
def rolls_per_dozen : ℕ := 12

-- Prove the total cost calculation
theorem total_cost_of_rolls : (total_rolls_bought / rolls_per_dozen) * price_per_dozen = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_rolls_l891_89166


namespace NUMINAMATH_GPT_interest_first_year_l891_89136
-- Import the necessary math library

-- Define the conditions and proof the interest accrued in the first year
theorem interest_first_year :
  ∀ (P B₁ : ℝ) (r₂ increase_ratio: ℝ),
    P = 1000 →
    B₁ = 1100 →
    r₂ = 0.20 →
    increase_ratio = 0.32 →
    (B₁ - P) = 100 :=
by
  intros P B₁ r₂ increase_ratio P_def B₁_def r₂_def increase_ratio_def
  sorry

end NUMINAMATH_GPT_interest_first_year_l891_89136


namespace NUMINAMATH_GPT_optimal_discount_order_l891_89143

variables (p : ℝ) (d1 : ℝ) (d2 : ℝ)

-- Original price of "Stars Beyond" is 30 dollars
def original_price : ℝ := 30

-- Fixed discount is 5 dollars
def discount_5 : ℝ := 5

-- 25% discount represented as a multiplier
def discount_25 : ℝ := 0.75

-- Applying $5 discount first and then 25% discount
def price_after_5_then_25_discount := discount_25 * (original_price - discount_5)

-- Applying 25% discount first and then $5 discount
def price_after_25_then_5_discount := (discount_25 * original_price) - discount_5

-- The additional savings when applying 25% discount first
def additional_savings := price_after_5_then_25_discount - price_after_25_then_5_discount

theorem optimal_discount_order : 
  additional_savings = 1.25 :=
sorry

end NUMINAMATH_GPT_optimal_discount_order_l891_89143


namespace NUMINAMATH_GPT_evaluate_expression_l891_89107

theorem evaluate_expression : 
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l891_89107


namespace NUMINAMATH_GPT_tree_leaves_remaining_after_three_weeks_l891_89181

theorem tree_leaves_remaining_after_three_weeks :
  let initial_leaves := 1000
  let leaves_shed_first_week := (2 / 5 : ℝ) * initial_leaves
  let leaves_remaining_after_first_week := initial_leaves - leaves_shed_first_week
  let leaves_shed_second_week := (4 / 10 : ℝ) * leaves_remaining_after_first_week
  let leaves_remaining_after_second_week := leaves_remaining_after_first_week - leaves_shed_second_week
  let leaves_shed_third_week := (3 / 4 : ℝ) * leaves_shed_second_week
  let leaves_remaining_after_third_week := leaves_remaining_after_second_week - leaves_shed_third_week
  leaves_remaining_after_third_week = 180 :=
by
  sorry

end NUMINAMATH_GPT_tree_leaves_remaining_after_three_weeks_l891_89181


namespace NUMINAMATH_GPT_largest_n_unique_k_l891_89127

theorem largest_n_unique_k : ∃ n : ℕ, n = 24 ∧ (∃! k : ℕ, 
  3 / 7 < n / (n + k: ℤ) ∧ n / (n + k: ℤ) < 8 / 19) :=
by
  sorry

end NUMINAMATH_GPT_largest_n_unique_k_l891_89127


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_sum_l891_89196

theorem arithmetic_sequence_geometric_sum (a₁ a₂ d : ℕ) (h₁ : d ≠ 0) 
    (h₂ : (2 * a₁ + d)^2 = a₁ * (4 * a₁ + 6 * d)) :
    a₂ = 3 * a₁ :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_sum_l891_89196


namespace NUMINAMATH_GPT_find_x_value_l891_89192

def average_eq_condition (x : ℝ) : Prop :=
  (5050 + x) / 101 = 50 * (x + 1)

theorem find_x_value : ∃ x : ℝ, average_eq_condition x ∧ x = 0 :=
by
  use 0
  sorry

end NUMINAMATH_GPT_find_x_value_l891_89192


namespace NUMINAMATH_GPT_fraction_subtraction_l891_89178

theorem fraction_subtraction (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / y = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l891_89178


namespace NUMINAMATH_GPT_inequality_am_gm_l891_89113

variable {a b c : ℝ}

theorem inequality_am_gm (habc_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc_eq_1 : a * b * c = 1) : 
  a^3 + b^3 + c^3 + (a * b / (a^2 + b^2) + b * c / (b^2 + c^2) + c * a / (c^2 + a^2)) ≥ 9 / 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l891_89113


namespace NUMINAMATH_GPT_sum_all_possible_values_l891_89119

theorem sum_all_possible_values (x : ℝ) (h : x^2 = 16) :
  (x = 4 ∨ x = -4) → (4 + (-4) = 0) :=
by
  intro h1
  have : 4 + (-4) = 0 := by norm_num
  exact this

end NUMINAMATH_GPT_sum_all_possible_values_l891_89119


namespace NUMINAMATH_GPT_newOp_of_M_and_N_l891_89114

def newOp (A B : Set ℕ) : Set ℕ :=
  {x | x ∈ A ∨ x ∈ B ∧ x ∉ (A ∩ B)}

theorem newOp_of_M_and_N (M N : Set ℕ) :
  M = {0, 2, 4, 6, 8, 10} →
  N = {0, 3, 6, 9, 12, 15} →
  newOp (newOp M N) M = N :=
by
  intros hM hN
  sorry

end NUMINAMATH_GPT_newOp_of_M_and_N_l891_89114


namespace NUMINAMATH_GPT_scholarship_amount_l891_89198

-- Definitions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def work_hours : ℕ := 200
def hourly_wage : ℕ := 10
def work_earnings : ℕ := work_hours * hourly_wage
def remaining_tuition : ℕ := tuition_per_semester - parents_contribution - work_earnings

-- Theorem to prove the scholarship amount
theorem scholarship_amount (S : ℕ) (h : 3 * S = remaining_tuition) : S = 3000 :=
by
  sorry

end NUMINAMATH_GPT_scholarship_amount_l891_89198


namespace NUMINAMATH_GPT_maria_sister_drank_l891_89117

-- Define the conditions
def initial_bottles : ℝ := 45.0
def maria_drank : ℝ := 14.0
def remaining_bottles : ℝ := 23.0

-- Define the problem statement to prove the number of bottles Maria's sister drank
theorem maria_sister_drank (initial_bottles maria_drank remaining_bottles : ℝ) : 
    (initial_bottles - maria_drank) - remaining_bottles = 8.0 :=
by
  sorry

end NUMINAMATH_GPT_maria_sister_drank_l891_89117


namespace NUMINAMATH_GPT_determine_constants_l891_89144

theorem determine_constants :
  ∃ (a b c p : ℝ), (a = -1) ∧ (b = -1) ∧ (c = -1) ∧ (p = 3) ∧
  (∀ x : ℝ, x^3 + p*x^2 + 3*x - 10 = 0 ↔ (x = a ∨ x = b ∨ x = c)) ∧ 
  c - b = b - a ∧ c - b > 0 :=
by
  sorry

end NUMINAMATH_GPT_determine_constants_l891_89144


namespace NUMINAMATH_GPT_candle_problem_l891_89154

theorem candle_problem :
  ∃ x : ℚ,
    (1 - x / 6 = 3 * (1 - x / 5)) ∧
    x = 60 / 13 :=
by
  -- let initial_height_first_candle be 1
  -- let rate_first_burns be 1 / 6
  -- let initial_height_second_candle be 1
  -- let rate_second_burns be 1 / 5
  -- We want to prove:
  -- 1 - x / 6 = 3 * (1 - x / 5) ∧ x = 60 / 13
  sorry

end NUMINAMATH_GPT_candle_problem_l891_89154


namespace NUMINAMATH_GPT_ali_babas_cave_min_moves_l891_89185

theorem ali_babas_cave_min_moves : 
  ∀ (counters : Fin 28 → Fin 2018) (decrease_by : ℕ → Fin 28 → ℕ),
    (∀ n, n < 28 → decrease_by n ≤ 2017) → 
    (∃ (k : ℕ), k ≤ 11 ∧ 
      ∀ n, (n < 28 → decrease_by (k - n) n = 0)) :=
sorry

end NUMINAMATH_GPT_ali_babas_cave_min_moves_l891_89185


namespace NUMINAMATH_GPT_function_identity_l891_89102

-- Definitions of the problem
def f (n : ℕ) : ℕ := sorry

-- Main theorem to prove
theorem function_identity (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) * f (m - n) = f (m * m)) : 
  ∀ n : ℕ, n > 0 → f n = 1 := 
sorry

end NUMINAMATH_GPT_function_identity_l891_89102


namespace NUMINAMATH_GPT_is_factorization_l891_89120

-- given an equation A,
-- Prove A is factorization: 
-- i.e., x^3 - x = x * (x + 1) * (x - 1)

theorem is_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end NUMINAMATH_GPT_is_factorization_l891_89120


namespace NUMINAMATH_GPT_water_used_for_plates_and_clothes_is_48_l891_89152

noncomputable def waterUsedToWashPlatesAndClothes : ℕ := 
  let barrel1 := 65 
  let barrel2 := (75 * 80) / 100 
  let barrel3 := (45 * 60) / 100 
  let totalCollected := barrel1 + barrel2 + barrel3
  let usedForCars := 7 * 2
  let usedForPlants := 15
  let usedForDog := 10
  let usedForCooking := 5
  let usedForBathing := 12
  let totalUsed := usedForCars + usedForPlants + usedForDog + usedForCooking + usedForBathing
  let remainingWater := totalCollected - totalUsed
  remainingWater / 2

theorem water_used_for_plates_and_clothes_is_48 : 
  waterUsedToWashPlatesAndClothes = 48 :=
by
  sorry

end NUMINAMATH_GPT_water_used_for_plates_and_clothes_is_48_l891_89152


namespace NUMINAMATH_GPT_taxi_ride_distance_l891_89129

theorem taxi_ride_distance
  (initial_charge : ℝ) (additional_charge : ℝ) 
  (total_charge : ℝ) (initial_increment : ℝ) (distance_increment : ℝ)
  (initial_charge_eq : initial_charge = 2.10) 
  (additional_charge_eq : additional_charge = 0.40) 
  (total_charge_eq : total_charge = 17.70) 
  (initial_increment_eq : initial_increment = 1/5) 
  (distance_increment_eq : distance_increment = 1/5) : 
  (distance : ℝ) = 8 :=
by sorry

end NUMINAMATH_GPT_taxi_ride_distance_l891_89129


namespace NUMINAMATH_GPT_complement_union_eq_l891_89189

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 3, 5}

theorem complement_union_eq:
  compl A ∪ B = {0, 2, 3, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_eq_l891_89189


namespace NUMINAMATH_GPT_wall_building_time_l891_89121

variables (f b c y : ℕ) 

theorem wall_building_time :
  (y = 2 * f * c / b) 
  ↔ 
  (f > 0 ∧ b > 0 ∧ c > 0 ∧ (f * b * y = 2 * b * c)) := 
sorry

end NUMINAMATH_GPT_wall_building_time_l891_89121


namespace NUMINAMATH_GPT_marble_distribution_l891_89176

theorem marble_distribution (x : ℝ) (h : 49 = (3 * x + 2) + (x + 1) + (2 * x - 1) + x) :
  (3 * x + 2 = 22) ∧ (x + 1 = 8) ∧ (2 * x - 1 = 12) ∧ (x = 7) :=
by
  sorry

end NUMINAMATH_GPT_marble_distribution_l891_89176


namespace NUMINAMATH_GPT_min_d_value_l891_89177

noncomputable def minChordLength (a : ℝ) : ℝ :=
  let P1 := (Real.arcsin a, Real.arcsin a)
  let P2 := (Real.arccos a, -Real.arccos a)
  let d_sq := 2 * ((Real.arcsin a)^2 + (Real.arccos a)^2)
  Real.sqrt d_sq

theorem min_d_value {a : ℝ} (h₁ : a ∈ Set.Icc (-1) 1) : 
  ∃ d : ℝ, d = minChordLength a ∧ d ≥ (π / 2) :=
sorry

end NUMINAMATH_GPT_min_d_value_l891_89177


namespace NUMINAMATH_GPT_height_of_square_pyramid_is_13_l891_89160

noncomputable def square_pyramid_height (base_edge : ℝ) (adjacent_face_angle : ℝ) : ℝ :=
  let half_diagonal := base_edge * (Real.sqrt 2) / 2
  let sin_angle := Real.sin (adjacent_face_angle / 2 : ℝ)
  let opp_side := half_diagonal * sin_angle
  let height := half_diagonal * sin_angle / (Real.sqrt 3)
  height

theorem height_of_square_pyramid_is_13 :
  ∀ (base_edge : ℝ) (adjacent_face_angle : ℝ), 
  base_edge = 26 → 
  adjacent_face_angle = 120 → 
  square_pyramid_height base_edge adjacent_face_angle = 13 :=
by
  intros base_edge adjacent_face_angle h_base_edge h_adj_face_angle
  rw [h_base_edge, h_adj_face_angle]
  have half_diagonal := 26 * (Real.sqrt 2) / 2
  have sin_angle := Real.sin (120 / 2 : ℝ) -- sin 60 degrees
  have sqrt_three := Real.sqrt 3
  have height := (half_diagonal * sin_angle) / sqrt_three
  sorry

end NUMINAMATH_GPT_height_of_square_pyramid_is_13_l891_89160


namespace NUMINAMATH_GPT_equilateral_triangle_BJ_l891_89149

-- Define points G, F, H, J and their respective lengths on sides AB and BC
def equilateral_triangle_AG_GF_HJ_FC (AG GF HJ FC BJ : ℕ) : Prop :=
  AG = 3 ∧ GF = 11 ∧ HJ = 5 ∧ FC = 4 ∧ 
    (∀ (side_length : ℕ), side_length = AG + GF + HJ + FC → 
    (∀ (length_J : ℕ), length_J = side_length - (AG + HJ) → BJ = length_J))

-- Example usage statement
theorem equilateral_triangle_BJ : 
  ∃ BJ, equilateral_triangle_AG_GF_HJ_FC 3 11 5 4 BJ ∧ BJ = 15 :=
by
  use 15
  sorry

end NUMINAMATH_GPT_equilateral_triangle_BJ_l891_89149


namespace NUMINAMATH_GPT_sum_of_first_ten_nice_numbers_is_182_l891_89103

def is_proper_divisor (n d : ℕ) : Prop :=
  d > 1 ∧ d < n ∧ n % d = 0

def is_nice (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, is_proper_divisor n m → ∃ p q, n = p * q ∧ p ≠ q

def first_ten_nice_numbers : List ℕ := [6, 8, 10, 14, 15, 21, 22, 26, 27, 33]

def sum_first_ten_nice_numbers : ℕ := first_ten_nice_numbers.sum

theorem sum_of_first_ten_nice_numbers_is_182 :
  sum_first_ten_nice_numbers = 182 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_ten_nice_numbers_is_182_l891_89103


namespace NUMINAMATH_GPT_probability_of_three_white_balls_equals_8_over_65_l891_89145

noncomputable def probability_three_white_balls (n_white n_black : ℕ) (draws : ℕ) : ℚ :=
  (Nat.choose n_white draws : ℚ) / Nat.choose (n_white + n_black) draws

theorem probability_of_three_white_balls_equals_8_over_65 :
  probability_three_white_balls 8 7 3 = 8 / 65 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_three_white_balls_equals_8_over_65_l891_89145


namespace NUMINAMATH_GPT_factor_x4_minus_81_l891_89161

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intros x
  sorry

end NUMINAMATH_GPT_factor_x4_minus_81_l891_89161


namespace NUMINAMATH_GPT_f_triple_application_l891_89115

-- Define the function f : ℕ → ℕ such that f(x) = 3x + 2
def f (x : ℕ) : ℕ := 3 * x + 2

-- Theorem statement to prove f(f(f(1))) = 53
theorem f_triple_application : f (f (f 1)) = 53 := 
by 
  sorry

end NUMINAMATH_GPT_f_triple_application_l891_89115


namespace NUMINAMATH_GPT_smallest_number_in_systematic_sample_l891_89172

theorem smallest_number_in_systematic_sample (n m x : ℕ) (products : Finset ℕ) :
  n = 80 ∧ m = 5 ∧ products = Finset.range n ∧ x = 42 ∧ x ∈ products ∧ (∃ k : ℕ, x = (n / m) * k + 10) → 10 ∈ products :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_in_systematic_sample_l891_89172


namespace NUMINAMATH_GPT_mike_total_spent_l891_89193

noncomputable def total_spent_by_mike (food_cost wallet_cost shirt_cost shoes_cost belt_cost 
  discounted_shirt_cost discounted_shoes_cost discounted_belt_cost : ℝ) : ℝ :=
  food_cost + wallet_cost + discounted_shirt_cost + discounted_shoes_cost + discounted_belt_cost

theorem mike_total_spent :
  let food_cost := 30
  let wallet_cost := food_cost + 60
  let shirt_cost := wallet_cost / 3
  let shoes_cost := 2 * wallet_cost
  let belt_cost := shoes_cost - 45
  let discounted_shirt_cost := shirt_cost - (0.2 * shirt_cost)
  let discounted_shoes_cost := shoes_cost - (0.15 * shoes_cost)
  let discounted_belt_cost := belt_cost - (0.1 * belt_cost)
  total_spent_by_mike food_cost wallet_cost shirt_cost shoes_cost belt_cost
    discounted_shirt_cost discounted_shoes_cost discounted_belt_cost = 418.50 := by
  sorry

end NUMINAMATH_GPT_mike_total_spent_l891_89193


namespace NUMINAMATH_GPT_triangle_count_l891_89169

-- Define the function to compute the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the number of points on each side
def pointsAB : ℕ := 6
def pointsBC : ℕ := 7

-- Compute the number of triangles that can be formed
theorem triangle_count (h₁ : pointsAB = 6) (h₂ : pointsBC = 7) : 
  (binom pointsAB 2) * (binom pointsBC 1) + (binom pointsBC 2) * (binom pointsAB 1) = 231 := by
  sorry

end NUMINAMATH_GPT_triangle_count_l891_89169


namespace NUMINAMATH_GPT_wildcats_points_l891_89159

theorem wildcats_points (panthers_points wildcats_additional_points wildcats_points : ℕ)
  (h_panthers : panthers_points = 17)
  (h_wildcats : wildcats_additional_points = 19)
  (h_wildcats_points : wildcats_points = panthers_points + wildcats_additional_points) :
  wildcats_points = 36 :=
by
  have h1 : panthers_points = 17 := h_panthers
  have h2 : wildcats_additional_points = 19 := h_wildcats
  have h3 : wildcats_points = panthers_points + wildcats_additional_points := h_wildcats_points
  sorry

end NUMINAMATH_GPT_wildcats_points_l891_89159


namespace NUMINAMATH_GPT_find_power_l891_89134

theorem find_power (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^11) (h2 : x = 13) : y = 11 :=
sorry

end NUMINAMATH_GPT_find_power_l891_89134
