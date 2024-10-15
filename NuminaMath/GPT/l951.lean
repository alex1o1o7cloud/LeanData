import Mathlib

namespace NUMINAMATH_GPT_sub_one_inequality_l951_95109

theorem sub_one_inequality (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end NUMINAMATH_GPT_sub_one_inequality_l951_95109


namespace NUMINAMATH_GPT_find_m_l951_95106

theorem find_m (a b c m : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_m : 0 < m) (h : a * b * c * m = 1 + a^2 + b^2 + c^2) : 
  m = 4 :=
sorry

end NUMINAMATH_GPT_find_m_l951_95106


namespace NUMINAMATH_GPT_gcd_lcm_product_24_60_l951_95144

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_24_60_l951_95144


namespace NUMINAMATH_GPT_product_of_consecutive_integers_sqrt_50_l951_95160

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_sqrt_50_l951_95160


namespace NUMINAMATH_GPT_number_of_dogs_on_tuesday_l951_95185

variable (T : ℕ)
variable (H1 : 7 + T + 7 + 7 + 9 = 42)

theorem number_of_dogs_on_tuesday : T = 12 := by
  sorry

end NUMINAMATH_GPT_number_of_dogs_on_tuesday_l951_95185


namespace NUMINAMATH_GPT_min_expr_value_min_expr_value_iff_l951_95196

theorem min_expr_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 :=
by {
  sorry
}

theorem min_expr_value_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2) = 4 / 9) ↔ (x = 2.5 ∧ y = 2.5) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_expr_value_min_expr_value_iff_l951_95196


namespace NUMINAMATH_GPT_triangle_angle_inradius_l951_95130

variable (A B C : ℝ) 
variable (a b c R : ℝ)

theorem triangle_angle_inradius 
    (h1: 0 < A ∧ A < Real.pi)
    (h2: a * Real.cos C + (1/2) * c = b)
    (h3: a = 1):

    A = Real.pi / 3 ∧ R ≤ Real.sqrt 3 / 6 := 
by
  sorry

end NUMINAMATH_GPT_triangle_angle_inradius_l951_95130


namespace NUMINAMATH_GPT_no_such_function_exists_l951_95181

theorem no_such_function_exists :
  ¬ (∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l951_95181


namespace NUMINAMATH_GPT_triangle_angles_correct_l951_95147

noncomputable def triangle_angles (a b c : ℝ) : (ℝ × ℝ × ℝ) :=
by sorry

theorem triangle_angles_correct :
  triangle_angles 3 (Real.sqrt 8) (2 + Real.sqrt 2) =
    (67.5, 22.5, 90) :=
by sorry

end NUMINAMATH_GPT_triangle_angles_correct_l951_95147


namespace NUMINAMATH_GPT_evaluate_polynomial_at_3_l951_95197

def f (x : ℕ) : ℕ := 3 * x ^ 3 + x - 3

theorem evaluate_polynomial_at_3 : f 3 = 28 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_3_l951_95197


namespace NUMINAMATH_GPT_inequality_proof_l951_95158

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_inequality_proof_l951_95158


namespace NUMINAMATH_GPT_buckets_oranges_l951_95150

theorem buckets_oranges :
  ∀ (a b c : ℕ), 
  a = 22 → 
  b = a + 17 → 
  a + b + c = 89 → 
  b - c = 11 := 
by 
  intros a b c h1 h2 h3 
  sorry

end NUMINAMATH_GPT_buckets_oranges_l951_95150


namespace NUMINAMATH_GPT_total_selling_price_correct_l951_95191

noncomputable def calculateSellingPrice (price1 price2 price3 loss1 loss2 loss3 taxRate overheadCost : ℝ) : ℝ :=
  let totalPurchasePrice := price1 + price2 + price3
  let tax := taxRate * totalPurchasePrice
  let sellingPrice1 := price1 - (loss1 * price1)
  let sellingPrice2 := price2 - (loss2 * price2)
  let sellingPrice3 := price3 - (loss3 * price3)
  let totalSellingPrice := sellingPrice1 + sellingPrice2 + sellingPrice3
  totalSellingPrice + overheadCost + tax

theorem total_selling_price_correct :
  calculateSellingPrice 750 1200 500 0.10 0.15 0.05 0.05 300 = 2592.5 :=
by 
  -- The proof of this theorem is skipped.
  sorry

end NUMINAMATH_GPT_total_selling_price_correct_l951_95191


namespace NUMINAMATH_GPT_average_stamps_collected_per_day_l951_95167

theorem average_stamps_collected_per_day :
  let a := 10
  let d := 6
  let n := 6
  let total_sum := (n / 2) * (2 * a + (n - 1) * d)
  let average := total_sum / n
  average = 25 :=
by
  sorry

end NUMINAMATH_GPT_average_stamps_collected_per_day_l951_95167


namespace NUMINAMATH_GPT_mean_of_set_is_16_6_l951_95132

theorem mean_of_set_is_16_6 (m : ℝ) (h : m + 7 = 16) :
  (9 + 11 + 16 + 20 + 27) / 5 = 16.6 :=
by
  -- Proof steps would go here, but we use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_mean_of_set_is_16_6_l951_95132


namespace NUMINAMATH_GPT_five_n_plus_three_composite_l951_95104

theorem five_n_plus_three_composite (n x y : ℕ) 
  (h_pos : 0 < n)
  (h1 : 2 * n + 1 = x ^ 2)
  (h2 : 3 * n + 1 = y ^ 2) : 
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = 5 * n + 3 := 
sorry

end NUMINAMATH_GPT_five_n_plus_three_composite_l951_95104


namespace NUMINAMATH_GPT_sin_theta_value_l951_95172

theorem sin_theta_value (θ : ℝ) (h1 : 10 * (Real.tan θ) = 4 * (Real.cos θ)) (h2 : 0 < θ ∧ θ < π) : Real.sin θ = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_sin_theta_value_l951_95172


namespace NUMINAMATH_GPT_cylinder_lateral_surface_area_l951_95152

theorem cylinder_lateral_surface_area 
  (r h : ℝ) 
  (radius_eq : r = 2) 
  (height_eq : h = 5) : 
  2 * Real.pi * r * h = 62.8 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_cylinder_lateral_surface_area_l951_95152


namespace NUMINAMATH_GPT_value_of_five_minus_c_l951_95101

theorem value_of_five_minus_c (c d : ℤ) (h1 : 5 + c = 6 - d) (h2 : 7 + d = 10 + c) :
  5 - c = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_five_minus_c_l951_95101


namespace NUMINAMATH_GPT_inequality_proof_l951_95165

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + b * c) / a + (1 + c * a) / b + (1 + a * b) / c > 
  Real.sqrt (a^2 + 2) + Real.sqrt (b^2 + 2) + Real.sqrt (c^2 + 2) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l951_95165


namespace NUMINAMATH_GPT_percent_defective_units_l951_95113

theorem percent_defective_units (D : ℝ) (h1 : 0.05 * D = 0.5) : D = 10 := by
  sorry

end NUMINAMATH_GPT_percent_defective_units_l951_95113


namespace NUMINAMATH_GPT_frank_bags_on_saturday_l951_95156

def bags_filled_on_saturday (total_cans : Nat) (cans_per_bag : Nat) (bags_on_sunday : Nat) : Nat :=
  total_cans / cans_per_bag - bags_on_sunday

theorem frank_bags_on_saturday : 
  let total_cans := 40
  let cans_per_bag := 5
  let bags_on_sunday := 3
  bags_filled_on_saturday total_cans cans_per_bag bags_on_sunday = 5 :=
  by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_frank_bags_on_saturday_l951_95156


namespace NUMINAMATH_GPT_A_plus_B_eq_93_l951_95146

-- Definitions and conditions
def gcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)
def lcm (a b c : ℕ) : ℕ := a * b * c / (gcf a b c)

-- Values for A and B
def A := gcf 18 30 45
def B := lcm 18 30 45

-- Proof statement
theorem A_plus_B_eq_93 : A + B = 93 := by
  sorry

end NUMINAMATH_GPT_A_plus_B_eq_93_l951_95146


namespace NUMINAMATH_GPT_solve_trig_system_l951_95129

theorem solve_trig_system
  (k n : ℤ) :
  (∃ x y : ℝ,
    (2 * Real.sin x ^ 2 + 2 * Real.sqrt 2 * Real.sin x * Real.sin (2 * x) ^ 2 + Real.sin (2 * x) ^ 2 = 0 ∧
     Real.cos x = Real.cos y) ∧
    ((x = 2 * Real.pi * k ∧ y = 2 * Real.pi * n) ∨
     (x = Real.pi + 2 * Real.pi * k ∧ y = Real.pi + 2 * Real.pi * n) ∨
     (x = -Real.pi / 4 + 2 * Real.pi * k ∧ (y = Real.pi / 4 + 2 * Real.pi * n ∨ y = -Real.pi / 4 + 2 * Real.pi * n)) ∨
     (x = -3 * Real.pi / 4 + 2 * Real.pi * k ∧ (y = 3 * Real.pi / 4 + 2 * Real.pi * n ∨ y = -3 * Real.pi / 4 + 2 * Real.pi * n)))) :=
sorry

end NUMINAMATH_GPT_solve_trig_system_l951_95129


namespace NUMINAMATH_GPT_Yoojung_total_vehicles_l951_95154

theorem Yoojung_total_vehicles : 
  let motorcycles := 2
  let bicycles := 5
  motorcycles + bicycles = 7 := 
by
  sorry

end NUMINAMATH_GPT_Yoojung_total_vehicles_l951_95154


namespace NUMINAMATH_GPT_solve_system_exists_l951_95103

theorem solve_system_exists (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : 1 / x + 1 / y + 1 / z = 5 / 12) 
  (h3 : x^3 + y^3 + z^3 = 45) 
  : (x, y, z) = (2, -3, 4) ∨ (x, y, z) = (2, 4, -3) ∨ (x, y, z) = (-3, 2, 4) ∨ (x, y, z) = (-3, 4, 2) ∨ (x, y, z) = (4, 2, -3) ∨ (x, y, z) = (4, -3, 2) := 
sorry

end NUMINAMATH_GPT_solve_system_exists_l951_95103


namespace NUMINAMATH_GPT_cyclist_speed_ratio_l951_95121

theorem cyclist_speed_ratio
  (d : ℝ) (t₁ t₂ : ℝ) 
  (v₁ v₂ : ℝ)
  (h1 : d = 8)
  (h2 : t₁ = 4)
  (h3 : t₂ = 1)
  (h4 : d = (v₁ - v₂) * t₁)
  (h5 : d = (v₁ + v₂) * t₂) :
  v₁ / v₂ = 5 / 3 :=
sorry

end NUMINAMATH_GPT_cyclist_speed_ratio_l951_95121


namespace NUMINAMATH_GPT_emma_withdrew_amount_l951_95145

variable (W : ℝ) -- Variable representing the amount Emma withdrew

theorem emma_withdrew_amount:
  (230 - W + 2 * W = 290) →
  W = 60 :=
by
  sorry

end NUMINAMATH_GPT_emma_withdrew_amount_l951_95145


namespace NUMINAMATH_GPT_anusha_solution_l951_95123

variable (A B E : ℝ) -- Defining the variables for amounts received by Anusha, Babu, and Esha
variable (total_amount : ℝ) (h_division : 12 * A = 8 * B) (h_division2 : 8 * B = 6 * E) (h_total : A + B + E = 378)

theorem anusha_solution : A = 84 :=
by
  -- Using the given conditions and deriving the amount Anusha receives
  sorry

end NUMINAMATH_GPT_anusha_solution_l951_95123


namespace NUMINAMATH_GPT_prime_cube_solution_l951_95135

theorem prime_cube_solution (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : p^3 = p^2 + q^2 + r^2) : 
  p = 3 ∧ q = 3 ∧ r = 3 :=
by
  sorry

end NUMINAMATH_GPT_prime_cube_solution_l951_95135


namespace NUMINAMATH_GPT_log_inequality_solution_l951_95122

noncomputable def log_a (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log_a a (3 / 5) < 1) ↔ (a ∈ Set.Ioo 0 (3 / 5) ∪ Set.Ioi 1) := 
by
  sorry

end NUMINAMATH_GPT_log_inequality_solution_l951_95122


namespace NUMINAMATH_GPT_ceil_square_eq_four_l951_95153

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end NUMINAMATH_GPT_ceil_square_eq_four_l951_95153


namespace NUMINAMATH_GPT_cube_surface_area_l951_95133

-- Definitions based on conditions from the problem
def edge_length : ℕ := 7
def number_of_faces : ℕ := 6

-- Definition of the problem converted to a theorem in Lean 4
theorem cube_surface_area (edge_length : ℕ) (number_of_faces : ℕ) : 
  number_of_faces * (edge_length * edge_length) = 294 :=
by
  -- Proof steps are omitted, so we put sorry to indicate that the proof is required.
  sorry

end NUMINAMATH_GPT_cube_surface_area_l951_95133


namespace NUMINAMATH_GPT_simplify_expression_l951_95176

variable (a b c d : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h : a + b + c = d)

theorem simplify_expression :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l951_95176


namespace NUMINAMATH_GPT_log24_eq_2b_minus_a_l951_95142

variable (a b : ℝ)

-- given conditions
axiom log6_eq : Real.log 6 = a
axiom log12_eq : Real.log 12 = b

-- proof goal statement
theorem log24_eq_2b_minus_a : Real.log 24 = 2 * b - a :=
by
  sorry

end NUMINAMATH_GPT_log24_eq_2b_minus_a_l951_95142


namespace NUMINAMATH_GPT_number_of_possible_values_of_a_l951_95157

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ),
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2040 ∧
  a^2 - b^2 + c^2 - d^2 = 2040 ∧
  508 ∈ {a | ∃ b c d, a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2040 ∧ a^2 - b^2 + c^2 - d^2 = 2040}

theorem number_of_possible_values_of_a : problem_statement :=
  sorry

end NUMINAMATH_GPT_number_of_possible_values_of_a_l951_95157


namespace NUMINAMATH_GPT_flag_arrangement_division_l951_95112

noncomputable def flag_arrangement_modulo : ℕ :=
  let num_blue_flags := 9
  let num_red_flags := 8
  let num_slots := num_blue_flags + 1
  let initial_arrangements := (num_slots.choose num_red_flags) * (num_blue_flags + 1)
  let invalid_cases := (num_blue_flags.choose num_red_flags) * 2
  let M := initial_arrangements - invalid_cases
  M % 1000

theorem flag_arrangement_division (M : ℕ) (num_blue_flags num_red_flags : ℕ) :
  num_blue_flags = 9 → num_red_flags = 8 → M = flag_arrangement_modulo → M % 1000 = 432 :=
by
  intros _ _ hM
  rw [hM]
  trivial

end NUMINAMATH_GPT_flag_arrangement_division_l951_95112


namespace NUMINAMATH_GPT_proof_problem_l951_95159

variable (a b c A B C : ℝ)
variable (h_a : a = Real.sqrt 3)
variable (h_b_ge_a : b ≥ a)
variable (h_cos : Real.cos (2 * C) - Real.cos (2 * A) =
  2 * Real.sin (Real.pi / 3 + C) * Real.sin (Real.pi / 3 - C))

theorem proof_problem :
  (A = Real.pi / 3) ∧ (2 * b - c ∈ Set.Ico (Real.sqrt 3) (2 * Real.sqrt 3)) :=
  sorry

end NUMINAMATH_GPT_proof_problem_l951_95159


namespace NUMINAMATH_GPT_find_m_l951_95184

theorem find_m (m a : ℝ) (h : (2:ℝ) * 1^2 - 3 * 1 + a = 0) 
  (h_roots : ∀ x : ℝ, 2 * x^2 - 3 * x + a = 0 → (x = 1 ∨ x = m)) :
  m = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l951_95184


namespace NUMINAMATH_GPT_earrings_ratio_l951_95175

theorem earrings_ratio :
  ∃ (M R : ℕ), 10 = M / 4 ∧ 10 + M + R = 70 ∧ M / R = 2 := by
  sorry

end NUMINAMATH_GPT_earrings_ratio_l951_95175


namespace NUMINAMATH_GPT_jaime_average_speed_l951_95187

theorem jaime_average_speed :
  let start_time := 10.0 -- 10:00 AM
  let end_time := 15.5 -- 3:30 PM (in 24-hour format)
  let total_distance := 21.0 -- kilometers
  let total_time := end_time - start_time -- time in hours
  total_distance / total_time = 3.82 := 
sorry

end NUMINAMATH_GPT_jaime_average_speed_l951_95187


namespace NUMINAMATH_GPT_sequence_neither_arithmetic_nor_geometric_l951_95169

noncomputable def Sn (n : ℕ) : ℕ := 3 * n + 2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 5 else Sn n - Sn (n - 1)

def not_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ d, ∀ n, a (n + 1) = a n + d

def not_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ r, ∀ n, a (n + 1) = r * a n

theorem sequence_neither_arithmetic_nor_geometric :
  not_arithmetic_sequence a ∧ not_geometric_sequence a :=
sorry

end NUMINAMATH_GPT_sequence_neither_arithmetic_nor_geometric_l951_95169


namespace NUMINAMATH_GPT_infinite_rational_points_in_region_l951_95131

theorem infinite_rational_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), 
  (∀ p ∈ S, (p.1 ^ 2 + p.2 ^ 2 ≤ 16) ∧ (p.1 ≤ 3) ∧ (p.2 ≤ 3) ∧ (p.1 > 0) ∧ (p.2 > 0)) ∧
  Set.Infinite S :=
sorry

end NUMINAMATH_GPT_infinite_rational_points_in_region_l951_95131


namespace NUMINAMATH_GPT_find_roots_of_polynomial_l951_95163

def f (x : ℝ) := x^3 - 2*x^2 - 5*x + 6

theorem find_roots_of_polynomial :
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 3 = 0) :=
by
  -- Proof will be written here
  sorry

end NUMINAMATH_GPT_find_roots_of_polynomial_l951_95163


namespace NUMINAMATH_GPT_find_number_l951_95188

theorem find_number (number : ℤ) (h : number + 7 = 6) : number = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l951_95188


namespace NUMINAMATH_GPT_businessmen_neither_coffee_nor_tea_l951_95194

theorem businessmen_neither_coffee_nor_tea
  (total : ℕ)
  (C T : Finset ℕ)
  (hC : C.card = 15)
  (hT : T.card = 14)
  (hCT : (C ∩ T).card = 7)
  (htotal : total = 30) : 
  total - (C ∪ T).card = 8 := 
by
  sorry

end NUMINAMATH_GPT_businessmen_neither_coffee_nor_tea_l951_95194


namespace NUMINAMATH_GPT_mr_desmond_toys_l951_95162

theorem mr_desmond_toys (toys_for_elder : ℕ) (h1 : toys_for_elder = 60)
  (h2 : ∀ (toys_for_younger : ℕ), toys_for_younger = 3 * toys_for_elder) : 
  ∃ (total_toys : ℕ), total_toys = 240 :=
by {
  sorry
}

end NUMINAMATH_GPT_mr_desmond_toys_l951_95162


namespace NUMINAMATH_GPT_donna_smallest_n_l951_95136

theorem donna_smallest_n (n : ℕ) : 15 * n - 1 % 6 = 0 ↔ n % 6 = 5 := sorry

end NUMINAMATH_GPT_donna_smallest_n_l951_95136


namespace NUMINAMATH_GPT_sqrt_abc_sum_eq_162sqrt2_l951_95138

theorem sqrt_abc_sum_eq_162sqrt2 (a b c : ℝ) (h1 : b + c = 15) (h2 : c + a = 18) (h3 : a + b = 21) :
    Real.sqrt (a * b * c * (a + b + c)) = 162 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_abc_sum_eq_162sqrt2_l951_95138


namespace NUMINAMATH_GPT_smallest_sum_abc_d_l951_95148

theorem smallest_sum_abc_d (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) : a + b + c + d = 108 :=
sorry

end NUMINAMATH_GPT_smallest_sum_abc_d_l951_95148


namespace NUMINAMATH_GPT_cube_surface_area_l951_95173

theorem cube_surface_area (PQ a b : ℝ) (x : ℝ) 
  (h1 : PQ = a / 2) 
  (h2 : PQ = Real.sqrt (3 * x^2)) : 
  b = 6 * x^2 → b = a^2 / 2 := 
by
  intros h_surface
  -- sorry is added here to skip the proof step and ensure the code builds successfully.
  sorry

end NUMINAMATH_GPT_cube_surface_area_l951_95173


namespace NUMINAMATH_GPT_geometric_sequence_increasing_iff_q_gt_one_l951_95141

variables {a_n : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def is_increasing_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n (n + 1) > a_n n

theorem geometric_sequence_increasing_iff_q_gt_one 
  (h1 : ∀ n, 0 < a_n n)
  (h2 : is_geometric_sequence a_n q) :
  is_increasing_sequence a_n ↔ q > 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_increasing_iff_q_gt_one_l951_95141


namespace NUMINAMATH_GPT_sum_of_three_consecutive_odd_integers_l951_95115

theorem sum_of_three_consecutive_odd_integers (n : ℤ) 
  (h1 : n + (n + 4) = 130) 
  (h2 : n % 2 = 1) : 
  n + (n + 2) + (n + 4) = 195 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_odd_integers_l951_95115


namespace NUMINAMATH_GPT_amy_lily_tie_probability_l951_95177

theorem amy_lily_tie_probability (P_Amy P_Lily : ℚ) (hAmy : P_Amy = 4/9) (hLily : P_Lily = 1/3) :
  1 - P_Amy - (↑P_Lily : ℚ) = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_amy_lily_tie_probability_l951_95177


namespace NUMINAMATH_GPT_least_subtract_divisible_by_8_l951_95120

def least_subtracted_to_divisible_by (n : ℕ) (d : ℕ) : ℕ :=
  n % d

theorem least_subtract_divisible_by_8 (n : ℕ) (d : ℕ) (h : n = 964807) (h_d : d = 8) :
  least_subtracted_to_divisible_by n d = 7 :=
by
  sorry

end NUMINAMATH_GPT_least_subtract_divisible_by_8_l951_95120


namespace NUMINAMATH_GPT_trig_identity_75_30_15_150_l951_95137

theorem trig_identity_75_30_15_150 :
  (Real.sin (75 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - 
   Real.sin (15 * Real.pi / 180) * Real.sin (150 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_trig_identity_75_30_15_150_l951_95137


namespace NUMINAMATH_GPT_number_of_containers_needed_l951_95111

/-
  Define the parameters for the given problem
-/
def bags_suki : ℝ := 6.75
def weight_per_bag_suki : ℝ := 27

def bags_jimmy : ℝ := 4.25
def weight_per_bag_jimmy : ℝ := 23

def bags_natasha : ℝ := 3.80
def weight_per_bag_natasha : ℝ := 31

def container_capacity : ℝ := 17

/-
  The total weight bought by each person and the total combined weight
-/
def total_weight_suki : ℝ := bags_suki * weight_per_bag_suki
def total_weight_jimmy : ℝ := bags_jimmy * weight_per_bag_jimmy
def total_weight_natasha : ℝ := bags_natasha * weight_per_bag_natasha

def total_weight_combined : ℝ := total_weight_suki + total_weight_jimmy + total_weight_natasha

/-
  Prove that number of containers needed is 24
-/
theorem number_of_containers_needed : 
  Nat.ceil (total_weight_combined / container_capacity) = 24 := 
by
  sorry

end NUMINAMATH_GPT_number_of_containers_needed_l951_95111


namespace NUMINAMATH_GPT_john_cards_sum_l951_95182

theorem john_cards_sum :
  ∃ (g : ℕ → ℕ) (y : ℕ → ℕ),
    (∀ n, (g n) ∈ [1, 2, 3, 4, 5]) ∧
    (∀ n, (y n) ∈ [2, 3, 4, 5]) ∧
    (∀ n, (g n < g (n + 1))) ∧
    (∀ n, (y n < y (n + 1))) ∧
    (∀ n, (g n ∣ y (n + 1) ∨ y (n + 1) ∣ g n)) ∧
    (g 0 = 1 ∧ g 2 = 2 ∧ g 4 = 5) ∧
    ( y 1 = 2 ∧ y 3 = 3 ∧ y 5 = 4 ) →
  g 0 + g 2 + g 4 = 8 := by
sorry

end NUMINAMATH_GPT_john_cards_sum_l951_95182


namespace NUMINAMATH_GPT_smallest_n_for_three_pairs_l951_95114

theorem smallest_n_for_three_pairs :
  ∃ (n : ℕ), (0 < n) ∧
    (∀ (x y : ℕ), (x^2 - y^2 = n) → (0 < x) ∧ (0 < y)) ∧
    (∃ (a b c : ℕ), 
      (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
      (∃ (x y : ℕ), (x^2 - y^2 = n) ∧
        (((x, y) = (a, b)) ∨ ((x, y) = (b, c)) ∨ ((x, y) = (a, c))))) :=
sorry

end NUMINAMATH_GPT_smallest_n_for_three_pairs_l951_95114


namespace NUMINAMATH_GPT_general_term_of_sequence_l951_95128

theorem general_term_of_sequence (a : Nat → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (2 + a (n + 1))) :
  ∀ n : ℕ, a (n + 1) = 2 / (n + 2) := 
sorry

end NUMINAMATH_GPT_general_term_of_sequence_l951_95128


namespace NUMINAMATH_GPT_multiples_of_4_between_200_and_500_l951_95149
-- Import the necessary library

open Nat

theorem multiples_of_4_between_200_and_500 : 
  ∃ n, n = (500 / 4 - 200 / 4) :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_4_between_200_and_500_l951_95149


namespace NUMINAMATH_GPT_min_n_for_constant_term_l951_95174

theorem min_n_for_constant_term (n : ℕ) (h : n > 0) :
  ∃ (r : ℕ), (2 * n = 5 * r) → n = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_n_for_constant_term_l951_95174


namespace NUMINAMATH_GPT_sequence_formula_l951_95100

theorem sequence_formula (a : ℕ → ℤ)
  (h₁ : a 1 = 1)
  (h₂ : a 2 = -3)
  (h₃ : a 3 = 5)
  (h₄ : a 4 = -7)
  (h₅ : a 5 = 9) :
  ∀ n : ℕ, a n = (-1)^(n+1) * (2 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l951_95100


namespace NUMINAMATH_GPT_opposite_points_number_line_l951_95125

theorem opposite_points_number_line (a : ℤ) (h : a - 6 = -a) : a = 3 := by
  sorry

end NUMINAMATH_GPT_opposite_points_number_line_l951_95125


namespace NUMINAMATH_GPT_remaining_wire_length_l951_95118

theorem remaining_wire_length (total_wire_length : ℝ) (square_side_length : ℝ) 
  (h₀ : total_wire_length = 60) (h₁ : square_side_length = 9) : 
  total_wire_length - 4 * square_side_length = 24 :=
by
  sorry

end NUMINAMATH_GPT_remaining_wire_length_l951_95118


namespace NUMINAMATH_GPT_boys_in_class_l951_95164

theorem boys_in_class 
  (avg_weight_incorrect : ℝ)
  (misread_weight_diff : ℝ)
  (avg_weight_correct : ℝ) 
  (n : ℕ) 
  (h1 : avg_weight_incorrect = 58.4) 
  (h2 : misread_weight_diff = 4) 
  (h3 : avg_weight_correct = 58.6) 
  (h4 : n * avg_weight_incorrect + misread_weight_diff = n * avg_weight_correct) :
  n = 20 := 
sorry

end NUMINAMATH_GPT_boys_in_class_l951_95164


namespace NUMINAMATH_GPT_friend_redistribute_l951_95171

-- Definition and total earnings
def earnings : List Int := [30, 45, 15, 10, 60]
def total_earnings := earnings.sum

-- Number of friends
def number_of_friends : Int := 5

-- Calculate the equal share
def equal_share := total_earnings / number_of_friends

-- Calculate the amount to redistribute by the friend who earned 60
def amount_to_give := 60 - equal_share

theorem friend_redistribute :
  earnings.sum = 160 ∧ equal_share = 32 ∧ amount_to_give = 28 :=
by
  -- Proof goes here, skipped with 'sorry'
  sorry

end NUMINAMATH_GPT_friend_redistribute_l951_95171


namespace NUMINAMATH_GPT_quadratic_polynomial_coefficients_l951_95170

theorem quadratic_polynomial_coefficients (a b : ℝ)
  (h1 : 2 * a - 1 - b = 0)
  (h2 : 5 * a + b - 13 = 0) :
  a^2 + b^2 = 13 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_polynomial_coefficients_l951_95170


namespace NUMINAMATH_GPT_sum_mean_median_mode_l951_95117

theorem sum_mean_median_mode : 
  let data := [2, 5, 1, 5, 2, 6, 1, 5, 0, 2]
  let ordered_data := [0, 1, 1, 2, 2, 2, 5, 5, 5, 6]
  let mean := (0 + 1 + 1 + 2 + 2 + 2 + 5 + 5 + 5 + 6) / 10
  let median := (2 + 2) / 2
  let mode := 5
  mean + median + mode = 9.9 := by
  sorry

end NUMINAMATH_GPT_sum_mean_median_mode_l951_95117


namespace NUMINAMATH_GPT_chipmunk_acorns_l951_95102

theorem chipmunk_acorns :
  ∀ (x y : ℕ), (3 * x = 4 * y) → (y = x - 4) → (3 * x = 48) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_chipmunk_acorns_l951_95102


namespace NUMINAMATH_GPT_kristi_books_proof_l951_95183

variable (Bobby_books Kristi_books : ℕ)

def condition1 : Prop := Bobby_books = 142

def condition2 : Prop := Bobby_books = Kristi_books + 64

theorem kristi_books_proof (h1 : condition1 Bobby_books) (h2 : condition2 Bobby_books Kristi_books) : Kristi_books = 78 := 
by 
  sorry

end NUMINAMATH_GPT_kristi_books_proof_l951_95183


namespace NUMINAMATH_GPT_total_flowers_l951_95124

-- Definition of conditions
def minyoung_flowers : ℕ := 24
def yoojung_flowers (y : ℕ) : Prop := minyoung_flowers = 4 * y

-- Theorem statement
theorem total_flowers (y : ℕ) (h : yoojung_flowers y) : minyoung_flowers + y = 30 :=
by sorry

end NUMINAMATH_GPT_total_flowers_l951_95124


namespace NUMINAMATH_GPT_polynomials_with_sum_of_abs_values_and_degree_eq_4_l951_95199

-- We define the general structure and conditions of the problem.
def polynomial_count : ℕ := 
  let count_0 := 1 -- For n = 0
  let count_1 := 6 -- For n = 1
  let count_2 := 9 -- For n = 2
  let count_3 := 1 -- For n = 3
  count_0 + count_1 + count_2 + count_3

theorem polynomials_with_sum_of_abs_values_and_degree_eq_4 : polynomial_count = 17 := 
by
  unfold polynomial_count
  -- The detailed proof steps for the count would go here
  sorry

end NUMINAMATH_GPT_polynomials_with_sum_of_abs_values_and_degree_eq_4_l951_95199


namespace NUMINAMATH_GPT_sum_of_two_numbers_l951_95180

theorem sum_of_two_numbers (x y : ℕ) (h1 : 3 * x = 180) (h2 : 4 * x = y) : x + y = 420 := by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l951_95180


namespace NUMINAMATH_GPT_average_weight_of_boys_l951_95119

theorem average_weight_of_boys
  (average_weight_girls : ℕ) 
  (average_weight_students : ℕ) 
  (h_girls : average_weight_girls = 45)
  (h_students : average_weight_students = 50) : 
  ∃ average_weight_boys : ℕ, average_weight_boys = 55 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_boys_l951_95119


namespace NUMINAMATH_GPT_farmer_goats_l951_95193

theorem farmer_goats (cows sheep goats : ℕ) (extra_goats : ℕ) 
(hcows : cows = 7) (hsheep : sheep = 8) (hgoats : goats = 6) 
(h : (goats + extra_goats = (cows + sheep + goats + extra_goats) / 2)) : 
extra_goats = 9 := by
  sorry

end NUMINAMATH_GPT_farmer_goats_l951_95193


namespace NUMINAMATH_GPT_find_k_l951_95198

theorem find_k (k x y : ℝ) (h1 : x = 2) (h2 : y = -3)
    (h3 : 2 * x^2 + k * x * y = 4) : k = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l951_95198


namespace NUMINAMATH_GPT_desired_yearly_income_l951_95140

theorem desired_yearly_income (total_investment : ℝ) 
  (investment1 : ℝ) (rate1 : ℝ) 
  (investment2 : ℝ) (rate2 : ℝ) 
  (rate_remainder : ℝ) 
  (h_total : total_investment = 10000) 
  (h_invest1 : investment1 = 4000)
  (h_rate1 : rate1 = 0.05) 
  (h_invest2 : investment2 = 3500)
  (h_rate2 : rate2 = 0.04)
  (h_rate_remainder : rate_remainder = 0.064)
  : (rate1 * investment1 + rate2 * investment2 + rate_remainder * (total_investment - (investment1 + investment2))) = 500 := 
by
  sorry

end NUMINAMATH_GPT_desired_yearly_income_l951_95140


namespace NUMINAMATH_GPT_xiaolin_final_score_l951_95189

-- Define the conditions
def score_situps : ℕ := 80
def score_800m : ℕ := 90
def weight_situps : ℕ := 4
def weight_800m : ℕ := 6

-- Define the final score based on the given conditions
def final_score : ℕ :=
  (score_situps * weight_situps + score_800m * weight_800m) / (weight_situps + weight_800m)

-- Prove that the final score is 86
theorem xiaolin_final_score : final_score = 86 :=
by sorry

end NUMINAMATH_GPT_xiaolin_final_score_l951_95189


namespace NUMINAMATH_GPT_correct_options_l951_95161

open Real

def option_A (x : ℝ) : Prop :=
  x^2 - 2*x + 1 > 0

def option_B : Prop :=
  ∃ (x : ℝ), (0 < x) ∧ (x + 4 / x = 6)

def option_C (a b : ℝ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) → (b / a + a / b ≥ 2)

def option_D (x y : ℝ) : Prop :=
  (0 < x) ∧ (0 < y) ∧ (x + 2*y = 1) → (2 / x + 1 / y ≥ 8)

theorem correct_options :
  ¬(∀ (x : ℝ), option_A x) ∧ (option_B ∧ (∀ (a b : ℝ), option_C a b) = false ∧ 
  (∀ (x y : ℝ), option_D x y)) :=
by sorry

end NUMINAMATH_GPT_correct_options_l951_95161


namespace NUMINAMATH_GPT_monotonicity_and_inequality_l951_95168

noncomputable def f (x : ℝ) := 2 * Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := a * x + 2
noncomputable def F (a : ℝ) (x : ℝ) := f x - g a x

theorem monotonicity_and_inequality (a : ℝ) (x₁ x₂ : ℝ) (hF_nonneg : ∀ x, F a x ≥ 0) (h_lt : x₁ < x₂) :
  (F a x₂ - F a x₁) / (x₂ - x₁) > 2 * (Real.exp x₁ - 1) :=
sorry

end NUMINAMATH_GPT_monotonicity_and_inequality_l951_95168


namespace NUMINAMATH_GPT_length_of_bridge_l951_95139

theorem length_of_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (crossing_time_seconds : ℕ)
  (h_train_length : train_length = 125)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_crossing_time_seconds : crossing_time_seconds = 30) :
  ∃ (bridge_length : ℕ), bridge_length = 250 :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l951_95139


namespace NUMINAMATH_GPT_find_cost_price_l951_95166

variable (C : ℝ)

def profit_10_percent_selling_price := 1.10 * C

def profit_15_percent_with_150_more := 1.10 * C + 150

def profit_15_percent_selling_price := 1.15 * C

theorem find_cost_price
  (h : profit_15_percent_with_150_more C = profit_15_percent_selling_price C) :
  C = 3000 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l951_95166


namespace NUMINAMATH_GPT_coefficient_condition_l951_95192

theorem coefficient_condition (m : ℝ) (h : m^3 * Nat.choose 6 3 = -160) : m = -2 := sorry

end NUMINAMATH_GPT_coefficient_condition_l951_95192


namespace NUMINAMATH_GPT_fiona_working_hours_l951_95151

theorem fiona_working_hours (F : ℕ) 
  (John_hours_per_week : ℕ := 30) 
  (Jeremy_hours_per_week : ℕ := 25) 
  (pay_rate : ℕ := 20) 
  (monthly_total_pay : ℕ := 7600) : 
  4 * (John_hours_per_week * pay_rate + Jeremy_hours_per_week * pay_rate + F * pay_rate) = monthly_total_pay → 
  F = 40 :=
by sorry

end NUMINAMATH_GPT_fiona_working_hours_l951_95151


namespace NUMINAMATH_GPT_ratio_of_kits_to_students_l951_95179

theorem ratio_of_kits_to_students (art_kits students : ℕ) (h1 : art_kits = 20) (h2 : students = 10) : art_kits / Nat.gcd art_kits students = 2 ∧ students / Nat.gcd art_kits students = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_of_kits_to_students_l951_95179


namespace NUMINAMATH_GPT_train_length_l951_95186

theorem train_length (L : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = L / 15) 
  (h2 : V2 = (L + 800) / 45) 
  (h3 : V1 = V2) : 
  L = 400 := 
sorry

end NUMINAMATH_GPT_train_length_l951_95186


namespace NUMINAMATH_GPT_solve_for_x_l951_95105

theorem solve_for_x (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l951_95105


namespace NUMINAMATH_GPT_turner_oldest_child_age_l951_95116

theorem turner_oldest_child_age (a b c : ℕ) (avg : ℕ) :
  (a = 6) → (b = 8) → (c = 11) → (avg = 9) → 
  (4 * avg = (a + b + c + x) → x = 11) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end NUMINAMATH_GPT_turner_oldest_child_age_l951_95116


namespace NUMINAMATH_GPT_lateral_surface_area_of_frustum_l951_95190

theorem lateral_surface_area_of_frustum (slant_height : ℝ) (ratio : ℕ × ℕ) (central_angle_deg : ℝ)
  (h_slant_height : slant_height = 10) 
  (h_ratio : ratio = (2, 5)) 
  (h_central_angle_deg : central_angle_deg = 216) : 
  ∃ (area : ℝ), area = (252 * Real.pi / 5) := 
by 
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_frustum_l951_95190


namespace NUMINAMATH_GPT_solve_system_eq_l951_95108

theorem solve_system_eq (x y z : ℝ) :
  (x * y * z / (x + y) = 6 / 5) ∧
  (x * y * z / (y + z) = 2) ∧
  (x * y * z / (z + x) = 3 / 2) ↔
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨ (x = -3 ∧ y = -2 ∧ z = -1)) := 
by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_solve_system_eq_l951_95108


namespace NUMINAMATH_GPT_proof_problem_l951_95134

variable (a b c d : ℝ)
variable (ω : ℂ)

-- Conditions
def conditions : Prop :=
  a ≠ -1 ∧ b ≠ -1 ∧ c ≠ -1 ∧ d ≠ -1 ∧
  ω^4 = 1 ∧ ω ≠ 1 ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2)

theorem proof_problem (h : conditions a b c d ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 2 := 
sorry

end NUMINAMATH_GPT_proof_problem_l951_95134


namespace NUMINAMATH_GPT_rhombus_area_l951_95195

-- Define d1 and d2 as the lengths of the diagonals
def d1 : ℝ := 15
def d2 : ℝ := 17

-- The theorem to prove the area of the rhombus
theorem rhombus_area : (d1 * d2) / 2 = 127.5 := by
  sorry

end NUMINAMATH_GPT_rhombus_area_l951_95195


namespace NUMINAMATH_GPT_correct_statement_C_l951_95155

theorem correct_statement_C : (∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5)) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_C_l951_95155


namespace NUMINAMATH_GPT_sequences_count_l951_95127

open BigOperators

def consecutive_blocks (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2 - 1) - 2

theorem sequences_count {n : ℕ} (h : n = 15) :
  consecutive_blocks n = 238 :=
by
  sorry

end NUMINAMATH_GPT_sequences_count_l951_95127


namespace NUMINAMATH_GPT_bicycle_car_speed_l951_95143

theorem bicycle_car_speed (x : Real) (h1 : x > 0) :
  10 / x - 10 / (2 * x) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_car_speed_l951_95143


namespace NUMINAMATH_GPT_find_quadruples_l951_95126

open Nat

/-- Define the primality property -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Define the problem conditions -/
def valid_quadruple (p1 p2 p3 p4 : ℕ) : Prop :=
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  p1 * p2 + p2 * p3 + p3 * p4 + p4 * p1 = 882

/-- The final theorem stating the valid quadruples -/
theorem find_quadruples :
  ∀ (p1 p2 p3 p4 : ℕ), valid_quadruple p1 p2 p3 p4 ↔ 
  (p1 = 2 ∧ p2 = 5 ∧ p3 = 19 ∧ p4 = 37) ∨
  (p1 = 2 ∧ p2 = 11 ∧ p3 = 19 ∧ p4 = 31) ∨
  (p1 = 2 ∧ p2 = 13 ∧ p3 = 19 ∧ p4 = 29) :=
by
  sorry

end NUMINAMATH_GPT_find_quadruples_l951_95126


namespace NUMINAMATH_GPT_tanya_time_proof_l951_95110

noncomputable def time_sakshi : ℝ := 10
noncomputable def efficiency_increase : ℝ := 1.25
noncomputable def time_tanya (time_sakshi : ℝ) (efficiency_increase : ℝ) : ℝ := time_sakshi / efficiency_increase

theorem tanya_time_proof : time_tanya time_sakshi efficiency_increase = 8 := 
by 
  sorry

end NUMINAMATH_GPT_tanya_time_proof_l951_95110


namespace NUMINAMATH_GPT_John_is_26_l951_95107

-- Define the variables representing the ages
def John_age : ℕ := 26
def Grandmother_age : ℕ := John_age + 48

-- Conditions
def condition1 : Prop := John_age = Grandmother_age - 48
def condition2 : Prop := John_age + Grandmother_age = 100

-- Main theorem to prove: John is 26 years old
theorem John_is_26 : John_age = 26 :=
by
  have h1 : condition1 := by sorry
  have h2 : condition2 := by sorry
  -- More steps to combine the conditions and prove the theorem would go here
  -- Skipping proof steps with sorry for demonstration
  sorry

end NUMINAMATH_GPT_John_is_26_l951_95107


namespace NUMINAMATH_GPT_infinite_points_of_one_color_l951_95178

theorem infinite_points_of_one_color (colors : ℤ → Prop) (red blue : ℤ → Prop)
  (h_colors : ∀ n : ℤ, colors n → (red n ∨ blue n))
  (h_red_blue : ∀ n : ℤ, red n → ¬ blue n)
  (h_blue_red : ∀ n : ℤ, blue n → ¬ red n) :
  ∃ c : ℤ → Prop, (∀ k : ℕ, ∃ infinitely_many p : ℤ, c p ∧ p % k = 0) :=
by
  sorry

end NUMINAMATH_GPT_infinite_points_of_one_color_l951_95178
