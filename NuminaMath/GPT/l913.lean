import Mathlib

namespace NUMINAMATH_GPT_cover_with_L_shapes_l913_91305

def L_shaped (m n : ℕ) : Prop :=
  m > 1 ∧ n > 1 ∧ ∃ k, m * n = 8 * k -- Conditions and tiling pattern coverage.

-- Problem statement as a theorem
theorem cover_with_L_shapes (m n : ℕ) (h1 : m > 1) (h2 : n > 1) : (∃ k, m * n = 8 * k) ↔ L_shaped m n :=
-- Placeholder for the proof
sorry

end NUMINAMATH_GPT_cover_with_L_shapes_l913_91305


namespace NUMINAMATH_GPT_problem1_l913_91328

theorem problem1 (a b : ℤ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a - b = -8 ∨ a - b = -2 := by 
  sorry

end NUMINAMATH_GPT_problem1_l913_91328


namespace NUMINAMATH_GPT_isoland_license_plates_proof_l913_91331

def isoland_license_plates : ℕ :=
  let letters := ['A', 'B', 'D', 'E', 'I', 'L', 'N', 'O', 'R', 'U']
  let valid_letters := letters.erase 'B'
  let first_letter_choices := ['A', 'I']
  let last_letter := 'R'
  let remaining_letters:= valid_letters.erase last_letter
  (first_letter_choices.length * (remaining_letters.length - first_letter_choices.length) * (remaining_letters.length - first_letter_choices.length - 1) * (remaining_letters.length - first_letter_choices.length - 2))

theorem isoland_license_plates_proof :
  isoland_license_plates = 420 := by
  sorry

end NUMINAMATH_GPT_isoland_license_plates_proof_l913_91331


namespace NUMINAMATH_GPT_max_gcd_11n_3_6n_1_l913_91309

theorem max_gcd_11n_3_6n_1 : ∃ n : ℕ+, ∀ k : ℕ+,  11 * n + 3 = 7 * k + 1 ∧ 6 * n + 1 = 7 * k + 2 → ∃ d : ℕ, d = Nat.gcd (11 * n + 3) (6 * n + 1) ∧ d = 7 :=
by
  sorry

end NUMINAMATH_GPT_max_gcd_11n_3_6n_1_l913_91309


namespace NUMINAMATH_GPT_unique_solution_p_eq_neg8_l913_91364

theorem unique_solution_p_eq_neg8 (p : ℝ) (h : ∀ y : ℝ, 2 * y^2 - 8 * y - p = 0 → ∃! y : ℝ, 2 * y^2 - 8 * y - p = 0) : p = -8 :=
sorry

end NUMINAMATH_GPT_unique_solution_p_eq_neg8_l913_91364


namespace NUMINAMATH_GPT_algebraic_simplification_l913_91315

theorem algebraic_simplification (m x : ℝ) (h₀ : 0 < m) (h₁ : m < 10) (h₂ : m ≤ x) (h₃ : x ≤ 10) : 
  |x - m| + |x - 10| + |x - m - 10| = 20 - x :=
by
  sorry

end NUMINAMATH_GPT_algebraic_simplification_l913_91315


namespace NUMINAMATH_GPT_profit_8000_l913_91394

noncomputable def profit (selling_price increase : ℝ) : ℝ :=
  (selling_price - 40 + increase) * (500 - 10 * increase)

theorem profit_8000 (increase : ℝ) :
  profit 50 increase = 8000 →
  ((increase = 10 ∧ (50 + increase = 60) ∧ (500 - 10 * increase = 400)) ∨ 
   (increase = 30 ∧ (50 + increase = 80) ∧ (500 - 10 * increase = 200))) :=
by
  sorry

end NUMINAMATH_GPT_profit_8000_l913_91394


namespace NUMINAMATH_GPT_sum_of_first_110_terms_l913_91349

theorem sum_of_first_110_terms
  (a d : ℝ)
  (h1 : (10 : ℝ) * (2 * a + (10 - 1) * d) / 2 = 100)
  (h2 : (100 : ℝ) * (2 * a + (100 - 1) * d) / 2 = 10) :
  (110 : ℝ) * (2 * a + (110 - 1) * d) / 2 = -110 :=
  sorry

end NUMINAMATH_GPT_sum_of_first_110_terms_l913_91349


namespace NUMINAMATH_GPT_fraction_c_d_l913_91326

theorem fraction_c_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) :
  c / d = -8 / 15 :=
sorry

end NUMINAMATH_GPT_fraction_c_d_l913_91326


namespace NUMINAMATH_GPT_polygon_stats_l913_91339

-- Definitions based on the problem's conditions
def total_number_of_polygons : ℕ := 207
def median_position : ℕ := 104
def m : ℕ := 14
def sum_of_squares_of_sides : ℕ := 2860
def mean_value : ℚ := sum_of_squares_of_sides / total_number_of_polygons
def mode_median : ℚ := 11.5

-- The proof statement
theorem polygon_stats (d μ M : ℚ)
  (h₁ : μ = mean_value)
  (h₂ : d = mode_median)
  (h₃ : M = m) :
  d < μ ∧ μ < M :=
by
  rw [h₁, h₂, h₃]
  -- The exact proof steps are omitted
  sorry

end NUMINAMATH_GPT_polygon_stats_l913_91339


namespace NUMINAMATH_GPT_percent_of_l913_91385

theorem percent_of (Part Whole : ℕ) (Percent : ℕ) (hPart : Part = 120) (hWhole : Whole = 40) :
  Percent = (Part * 100) / Whole → Percent = 300 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_l913_91385


namespace NUMINAMATH_GPT_publishing_company_break_even_l913_91391

theorem publishing_company_break_even : 
  ∀ (F V P : ℝ) (x : ℝ), F = 35630 ∧ V = 11.50 ∧ P = 20.25 →
  (P * x = F + V * x) → x = 4074 :=
by
  intros F V P x h_eq h_rev
  sorry

end NUMINAMATH_GPT_publishing_company_break_even_l913_91391


namespace NUMINAMATH_GPT_train_platform_time_l913_91397

theorem train_platform_time :
  ∀ (L_train L_platform T_tree S D T_platform : ℝ),
    L_train = 1200 ∧ 
    T_tree = 120 ∧ 
    L_platform = 1100 ∧ 
    S = L_train / T_tree ∧ 
    D = L_train + L_platform ∧ 
    T_platform = D / S →
    T_platform = 230 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_platform_time_l913_91397


namespace NUMINAMATH_GPT_fraction_meaningful_iff_l913_91353

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := sorry

end NUMINAMATH_GPT_fraction_meaningful_iff_l913_91353


namespace NUMINAMATH_GPT_geometric_sequence_a3a5_l913_91318

theorem geometric_sequence_a3a5 :
  ∀ (a : ℕ → ℝ) (r : ℝ), (a 4 = 4) → (a 3 = a 0 * r ^ 3) → (a 5 = a 0 * r ^ 5) →
  a 3 * a 5 = 16 :=
by
  intros a r h1 h2 h3
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3a5_l913_91318


namespace NUMINAMATH_GPT_sin_phi_value_l913_91311

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x

theorem sin_phi_value (φ : ℝ) (h_shift : ∀ x, g x = f (x - φ)) : Real.sin φ = 24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_phi_value_l913_91311


namespace NUMINAMATH_GPT_sum_of_reciprocals_l913_91310

theorem sum_of_reciprocals {a b : ℕ} (h_sum: a + b = 55) (h_hcf: Nat.gcd a b = 5) (h_lcm: Nat.lcm a b = 120) :
  1 / (a : ℚ) + 1 / (b : ℚ) = 11 / 120 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l913_91310


namespace NUMINAMATH_GPT_latus_rectum_equation_l913_91352

theorem latus_rectum_equation (y x : ℝ) :
  y^2 = 4 * x → x = -1 :=
sorry

end NUMINAMATH_GPT_latus_rectum_equation_l913_91352


namespace NUMINAMATH_GPT_solve_for_x_l913_91367

def custom_mul (a b : ℤ) : ℤ := a * b + a + b

theorem solve_for_x (x : ℤ) :
  custom_mul 3 (3 * x - 1) = 27 → x = 7 / 3 := by
sorry

end NUMINAMATH_GPT_solve_for_x_l913_91367


namespace NUMINAMATH_GPT_correct_statements_l913_91303

theorem correct_statements :
  (20 / 100 * 40 = 8) ∧
  (2^3 = 8) ∧
  (7 - 3 * 2 ≠ 8) ∧
  (3^2 - 1^2 = 8) ∧
  (2 * (6 - 4)^2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l913_91303


namespace NUMINAMATH_GPT_least_value_of_N_l913_91330

theorem least_value_of_N : ∃ (N : ℕ), (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧ N = 59 :=
by
  sorry

end NUMINAMATH_GPT_least_value_of_N_l913_91330


namespace NUMINAMATH_GPT_product_bc_l913_91325

theorem product_bc (b c : ℤ)
    (h1 : ∀ s : ℤ, s^2 = 2 * s + 1 → s^6 - b * s - c = 0) :
    b * c = 2030 :=
sorry

end NUMINAMATH_GPT_product_bc_l913_91325


namespace NUMINAMATH_GPT_last_three_digits_of_power_l913_91312

theorem last_three_digits_of_power (h : 7^500 ≡ 1 [MOD 1250]) : 7^10000 ≡ 1 [MOD 1250] :=
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_power_l913_91312


namespace NUMINAMATH_GPT_greatest_integer_difference_l913_91333

theorem greatest_integer_difference (x y : ℤ) (h1 : 5 < x ∧ x < 8) (h2 : 8 < y ∧ y < 13)
  (h3 : x % 3 = 0) (h4 : y % 3 = 0) : y - x = 6 :=
sorry

end NUMINAMATH_GPT_greatest_integer_difference_l913_91333


namespace NUMINAMATH_GPT_natalia_total_distance_l913_91374

theorem natalia_total_distance :
  let dist_mon := 40
  let bonus_mon := 0.05 * dist_mon
  let effective_mon := dist_mon + bonus_mon
  
  let dist_tue := 50
  let bonus_tue := 0.03 * dist_tue
  let effective_tue := dist_tue + bonus_tue
  
  let dist_wed := dist_tue / 2
  let bonus_wed := 0.07 * dist_wed
  let effective_wed := dist_wed + bonus_wed
  
  let dist_thu := dist_mon + dist_wed
  let bonus_thu := 0.04 * dist_thu
  let effective_thu := dist_thu + bonus_thu
  
  let dist_fri := 1.2 * dist_thu
  let bonus_fri := 0.06 * dist_fri
  let effective_fri := dist_fri + bonus_fri
  
  let dist_sat := 0.75 * dist_fri
  let bonus_sat := 0.02 * dist_sat
  let effective_sat := dist_sat + bonus_sat
  
  let dist_sun := dist_sat - dist_wed
  let bonus_sun := 0.10 * dist_sun
  let effective_sun := dist_sun + bonus_sun
  
  effective_mon + effective_tue + effective_wed + effective_thu + effective_fri + effective_sat + effective_sun = 367.05 :=
by
  sorry

end NUMINAMATH_GPT_natalia_total_distance_l913_91374


namespace NUMINAMATH_GPT_total_revenue_correct_l913_91378

def KwikETaxCenter : Type := ℕ

noncomputable def federal_return_price : ℕ := 50
noncomputable def state_return_price : ℕ := 30
noncomputable def quarterly_business_taxes_price : ℕ := 80
noncomputable def international_return_price : ℕ := 100
noncomputable def value_added_service_price : ℕ := 75

noncomputable def federal_returns_sold : ℕ := 60
noncomputable def state_returns_sold : ℕ := 20
noncomputable def quarterly_returns_sold : ℕ := 10
noncomputable def international_returns_sold : ℕ := 13
noncomputable def value_added_services_sold : ℕ := 25

noncomputable def international_discount : ℕ := 20

noncomputable def calculate_total_revenue 
   (federal_price : ℕ) (state_price : ℕ) 
   (quarterly_price : ℕ) (international_price : ℕ) 
   (value_added_price : ℕ)
   (federal_sold : ℕ) (state_sold : ℕ) 
   (quarterly_sold : ℕ) (international_sold : ℕ) 
   (value_added_sold : ℕ)
   (discount : ℕ) : ℕ := 
    (federal_price * federal_sold) 
  + (state_price * state_sold) 
  + (quarterly_price * quarterly_sold) 
  + ((international_price - discount) * international_sold) 
  + (value_added_price * value_added_sold)

theorem total_revenue_correct :
  calculate_total_revenue federal_return_price state_return_price 
                          quarterly_business_taxes_price international_return_price 
                          value_added_service_price
                          federal_returns_sold state_returns_sold 
                          quarterly_returns_sold international_returns_sold 
                          value_added_services_sold 
                          international_discount = 7315 := 
  by sorry

end NUMINAMATH_GPT_total_revenue_correct_l913_91378


namespace NUMINAMATH_GPT_citric_acid_molecular_weight_l913_91337

def molecular_weight_citric_acid := 192.12 -- in g/mol

theorem citric_acid_molecular_weight :
  molecular_weight_citric_acid = 192.12 :=
by sorry

end NUMINAMATH_GPT_citric_acid_molecular_weight_l913_91337


namespace NUMINAMATH_GPT_probability_red_or_blue_l913_91340

theorem probability_red_or_blue :
  ∀ (total_marbles white_marbles green_marbles red_blue_marbles : ℕ),
    total_marbles = 90 →
    (white_marbles : ℝ) / total_marbles = 1 / 6 →
    (green_marbles : ℝ) / total_marbles = 1 / 5 →
    white_marbles = 15 →
    green_marbles = 18 →
    red_blue_marbles = total_marbles - (white_marbles + green_marbles) →
    (red_blue_marbles : ℝ) / total_marbles = 19 / 30 :=
by
  intros total_marbles white_marbles green_marbles red_blue_marbles
  intros h_total_marbles h_white_prob h_green_prob h_white_count h_green_count h_red_blue_count
  sorry

end NUMINAMATH_GPT_probability_red_or_blue_l913_91340


namespace NUMINAMATH_GPT_inf_many_solutions_to_ineq_l913_91345

theorem inf_many_solutions_to_ineq (x : ℕ) : (15 < 2 * x + 20) ↔ x ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inf_many_solutions_to_ineq_l913_91345


namespace NUMINAMATH_GPT_lawn_chair_original_price_l913_91332

theorem lawn_chair_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 59.95 →
  discount_percentage = 23.09 →
  original_price = sale_price / (1 - discount_percentage / 100) →
  original_price = 77.95 :=
by sorry

end NUMINAMATH_GPT_lawn_chair_original_price_l913_91332


namespace NUMINAMATH_GPT_fraction_subtraction_property_l913_91389

variable (a b c d : ℚ)

theorem fraction_subtraction_property :
  (a / b - c / d) = ((a - c) / (b + d)) → (a / c) = (b / d) ^ 2 := 
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_property_l913_91389


namespace NUMINAMATH_GPT_avg_weight_of_class_l913_91322

def A_students : Nat := 36
def B_students : Nat := 44
def C_students : Nat := 50
def D_students : Nat := 30

def A_avg_weight : ℝ := 40
def B_avg_weight : ℝ := 35
def C_avg_weight : ℝ := 42
def D_avg_weight : ℝ := 38

def A_additional_students : Nat := 5
def A_additional_weight : ℝ := 10

def B_reduced_students : Nat := 7
def B_reduced_weight : ℝ := 8

noncomputable def total_weight_class : ℝ :=
  (A_students * A_avg_weight + A_additional_students * A_additional_weight) +
  (B_students * B_avg_weight - B_reduced_students * B_reduced_weight) +
  (C_students * C_avg_weight) +
  (D_students * D_avg_weight)

noncomputable def total_students_class : Nat :=
  A_students + B_students + C_students + D_students

noncomputable def avg_weight_class : ℝ :=
  total_weight_class / total_students_class

theorem avg_weight_of_class :
  avg_weight_class = 38.84 := by
    sorry

end NUMINAMATH_GPT_avg_weight_of_class_l913_91322


namespace NUMINAMATH_GPT_equilateral_triangle_grid_l913_91354

noncomputable def number_of_triangles (n : ℕ) : ℕ :=
1 + 3 + 5 + 7 + 9 + 1 + 2 + 3 + 4 + 3 + 1 + 2 + 3 + 1 + 2 + 1

theorem equilateral_triangle_grid (n : ℕ) (h : n = 5) : number_of_triangles n = 48 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_grid_l913_91354


namespace NUMINAMATH_GPT_english_book_pages_l913_91302

def numPagesInOneEnglishBook (x y : ℕ) : Prop :=
  x = y + 12 ∧ 3 * x + 4 * y = 1275 → x = 189

-- The statement with sorry as no proof is required:
theorem english_book_pages (x y : ℕ) (h1 : x = y + 12) (h2 : 3 * x + 4 * y = 1275) : x = 189 :=
  sorry

end NUMINAMATH_GPT_english_book_pages_l913_91302


namespace NUMINAMATH_GPT_x_intercept_of_line_l2_l913_91388

theorem x_intercept_of_line_l2 :
  ∀ (l1 l2 : ℝ → ℝ),
  (∀ x y, 2 * x - y + 3 = 0 → l1 x = y) →
  (∀ x y, 2 * x - y - 6 = 0 → l2 x = y) →
  l1 0 = 6 →
  l2 0 = -6 →
  l2 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l2_l913_91388


namespace NUMINAMATH_GPT_find_a5_l913_91314

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Condition: The sum of the first n terms of the sequence {a_n} is represented by S_n = 2a_n - 1 (n ∈ ℕ)
axiom sum_of_terms (n : ℕ) : S n = 2 * (a n) - 1

-- Prove that a_5 = 16
theorem find_a5 : a 5 = 16 :=
  sorry

end NUMINAMATH_GPT_find_a5_l913_91314


namespace NUMINAMATH_GPT_find_total_cost_price_l913_91316

noncomputable def cost_prices (C1 C2 C3 : ℝ) : Prop :=
  0.85 * C1 + 72.50 = 1.125 * C1 ∧
  1.20 * C2 - 45.30 = 0.95 * C2 ∧
  0.92 * C3 + 33.60 = 1.10 * C3

theorem find_total_cost_price :
  ∃ (C1 C2 C3 : ℝ), cost_prices C1 C2 C3 ∧ C1 + C2 + C3 = 631.51 := 
by
  sorry

end NUMINAMATH_GPT_find_total_cost_price_l913_91316


namespace NUMINAMATH_GPT_total_number_of_dresses_l913_91342

theorem total_number_of_dresses (ana_dresses lisa_more_dresses : ℕ) (h_condition : ana_dresses = 15) (h_more : lisa_more_dresses = ana_dresses + 18) : ana_dresses + lisa_more_dresses = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_dresses_l913_91342


namespace NUMINAMATH_GPT_max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l913_91359

-- First proof problem
theorem max_val_xa_minus_2x (x a : ℝ) (h1 : 0 < x) (h2 : 2 * x < a) :
  ∃ y, (y = x * (a - 2 * x)) ∧ y ≤ a^2 / 8 :=
sorry

-- Second proof problem
theorem max_val_ab_plus_bc_plus_ac (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 4) :
  ab + bc + ac ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l913_91359


namespace NUMINAMATH_GPT_sequence_exists_and_unique_l913_91336

theorem sequence_exists_and_unique (a : ℕ → ℕ) :
  a 0 = 11 ∧ a 7 = 12 ∧
  (∀ n : ℕ, n < 6 → a n + a (n + 1) + a (n + 2) = 50) →
  (a 0 = 11 ∧ a 1 = 12 ∧ a 2 = 27 ∧ a 3 = 11 ∧ a 4 = 12 ∧
   a 5 = 27 ∧ a 6 = 11 ∧ a 7 = 12) :=
by
  sorry

end NUMINAMATH_GPT_sequence_exists_and_unique_l913_91336


namespace NUMINAMATH_GPT_f_ln2_add_f_ln_half_l913_91335

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_ln2_add_f_ln_half :
  f (Real.log 2) + f (Real.log (1 / 2)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_f_ln2_add_f_ln_half_l913_91335


namespace NUMINAMATH_GPT_white_tshirt_cost_l913_91395

-- Define the problem conditions
def total_tshirts : ℕ := 200
def total_minutes : ℕ := 25
def black_tshirt_cost : ℕ := 30
def revenue_per_minute : ℕ := 220

-- Prove the cost of white t-shirts given the conditions
theorem white_tshirt_cost : 
  (total_tshirts / 2) * revenue_per_minute * total_minutes 
  - (total_tshirts / 2) * black_tshirt_cost = 2500
  → 2500 / (total_tshirts / 2) = 25 :=
by
  sorry

end NUMINAMATH_GPT_white_tshirt_cost_l913_91395


namespace NUMINAMATH_GPT_eq_no_sol_l913_91301

open Nat -- Use natural number namespace

theorem eq_no_sol (k : ℤ) (x y z : ℕ) (hk1 : k ≠ 1) (hk3 : k ≠ 3) :
  ¬ (x^2 + y^2 + z^2 = k * x * y * z) := 
sorry

end NUMINAMATH_GPT_eq_no_sol_l913_91301


namespace NUMINAMATH_GPT_cube_surface_area_of_same_volume_as_prism_l913_91346

theorem cube_surface_area_of_same_volume_as_prism :
  let prism_length := 10
  let prism_width := 5
  let prism_height := 24
  let prism_volume := prism_length * prism_width * prism_height
  let cube_edge := (prism_volume : ℝ)^(1/3)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 677.76 := by
  sorry

end NUMINAMATH_GPT_cube_surface_area_of_same_volume_as_prism_l913_91346


namespace NUMINAMATH_GPT_pipe_fills_entire_cistern_in_77_minutes_l913_91356

-- Define the time taken to fill 1/11 of the cistern
def time_to_fill_one_eleven_cistern : ℕ := 7

-- Define the fraction of the cistern filled in a certain time
def fraction_filled (t : ℕ) : ℚ := t / time_to_fill_one_eleven_cistern * (1 / 11)

-- Define the problem statement
theorem pipe_fills_entire_cistern_in_77_minutes : 
  fraction_filled 77 = 1 := by
  sorry

end NUMINAMATH_GPT_pipe_fills_entire_cistern_in_77_minutes_l913_91356


namespace NUMINAMATH_GPT_goose_eggs_laied_l913_91327

theorem goose_eggs_laied (z : ℕ) (hatch_rate : ℚ := 2 / 3) (first_month_survival_rate : ℚ := 3 / 4) 
  (first_year_survival_rate : ℚ := 2 / 5) (geese_survived_first_year : ℕ := 126) :
  (hatch_rate * z) = 420 ∧ (first_month_survival_rate * 315 = 315) ∧ (first_year_survival_rate * 315 = 126) →
  z = 630 :=
by
  sorry

end NUMINAMATH_GPT_goose_eggs_laied_l913_91327


namespace NUMINAMATH_GPT_calc_value_of_ab_bc_ca_l913_91380

theorem calc_value_of_ab_bc_ca (a b c : ℝ) (h1 : a + b + c = 35) (h2 : ab + bc + ca = 320) (h3 : abc = 600) : 
  (a + b) * (b + c) * (c + a) = 10600 := 
by sorry

end NUMINAMATH_GPT_calc_value_of_ab_bc_ca_l913_91380


namespace NUMINAMATH_GPT_minimize_quadratic_l913_91308

theorem minimize_quadratic (x : ℝ) : (x = -9 / 2) → ∀ y : ℝ, y^2 + 9 * y + 7 ≥ (-9 / 2)^2 + 9 * -9 / 2 + 7 :=
by sorry

end NUMINAMATH_GPT_minimize_quadratic_l913_91308


namespace NUMINAMATH_GPT_part_a_part_b_l913_91357

-- Part (a)
theorem part_a (x : ℝ) (h : x > 0) : x^3 - 3*x ≥ -2 :=
sorry

-- Part (b)
theorem part_b (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y / z) + (y^2 * z / x) + (z^2 * x / y) + 2 * ((y / (x * z)) + (z / (x * y)) + (x / (y * z))) ≥ 9 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l913_91357


namespace NUMINAMATH_GPT_max_strong_boys_l913_91319

theorem max_strong_boys (n : ℕ) (h : n = 100) (a b : Fin n → ℕ) 
  (ha : ∀ i j : Fin n, i < j → a i > a j) 
  (hb : ∀ i j : Fin n, i < j → b i < b j) : 
  ∃ k : ℕ, k = n := 
sorry

end NUMINAMATH_GPT_max_strong_boys_l913_91319


namespace NUMINAMATH_GPT_number_of_positive_integers_with_positive_log_l913_91369

theorem number_of_positive_integers_with_positive_log (b : ℕ) (h : ∃ n : ℕ, n > 0 ∧ b ^ n = 1024) : 
  ∃ L, L = 4 :=
sorry

end NUMINAMATH_GPT_number_of_positive_integers_with_positive_log_l913_91369


namespace NUMINAMATH_GPT_polygon_sides_l913_91366

theorem polygon_sides (n : ℕ) (h : n - 1 = 2022) : n = 2023 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l913_91366


namespace NUMINAMATH_GPT_circle_eq_l913_91304

theorem circle_eq (x y : ℝ) (h k r : ℝ) (hc : h = 3) (kc : k = 1) (rc : r = 5) :
  (x - h)^2 + (y - k)^2 = r^2 ↔ (x - 3)^2 + (y - 1)^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_circle_eq_l913_91304


namespace NUMINAMATH_GPT_max_profit_l913_91399

noncomputable def profit_function (x : ℕ) : ℝ :=
  if x ≤ 400 then
    300 * x - (1 / 2) * x^2 - 20000
  else
    60000 - 100 * x

theorem max_profit : 
  (∀ x ≥ 0, profit_function x ≤ 25000) ∧ (profit_function 300 = 25000) :=
by 
  sorry

end NUMINAMATH_GPT_max_profit_l913_91399


namespace NUMINAMATH_GPT_probability_C_l913_91360

-- Definitions of probabilities
def P_A : ℚ := 3 / 8
def P_B : ℚ := 1 / 4
def P_D : ℚ := 1 / 8

-- Main proof statement
theorem probability_C :
  ∀ P_C : ℚ, P_A + P_B + P_C + P_D = 1 → P_C = 1 / 4 :=
by
  intro P_C h
  sorry

end NUMINAMATH_GPT_probability_C_l913_91360


namespace NUMINAMATH_GPT_rate_of_mixed_oil_l913_91323

theorem rate_of_mixed_oil (V1 V2 : ℝ) (P1 P2 : ℝ) : 
  (V1 = 10) → 
  (P1 = 50) → 
  (V2 = 5) → 
  (P2 = 67) → 
  ((V1 * P1 + V2 * P2) / (V1 + V2) = 55.67) :=
by
  intros V1_eq P1_eq V2_eq P2_eq
  rw [V1_eq, P1_eq, V2_eq, P2_eq]
  norm_num
  sorry

end NUMINAMATH_GPT_rate_of_mixed_oil_l913_91323


namespace NUMINAMATH_GPT_no_solution_l913_91358

theorem no_solution (n : ℕ) (k : ℕ) (hn : Prime n) (hk : 0 < k) :
  ¬ (n ≤ n.factorial - k ^ n ∧ n.factorial - k ^ n ≤ k * n) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_l913_91358


namespace NUMINAMATH_GPT_mary_income_is_128_percent_of_juan_income_l913_91306

def juan_income : ℝ := sorry
def tim_income : ℝ := 0.80 * juan_income
def mary_income : ℝ := 1.60 * tim_income

theorem mary_income_is_128_percent_of_juan_income
  (J : ℝ) : mary_income = 1.28 * J :=
by
  sorry

end NUMINAMATH_GPT_mary_income_is_128_percent_of_juan_income_l913_91306


namespace NUMINAMATH_GPT_cards_left_l913_91324
noncomputable section

def initial_cards : ℕ := 676
def bought_cards : ℕ := 224

theorem cards_left : initial_cards - bought_cards = 452 := 
by
  sorry

end NUMINAMATH_GPT_cards_left_l913_91324


namespace NUMINAMATH_GPT_expression_divisible_by_a_square_l913_91371

theorem expression_divisible_by_a_square (n : ℕ) (a : ℤ) : 
  a^2 ∣ ((a * n - 1) * (a + 1) ^ n + 1) := 
sorry

end NUMINAMATH_GPT_expression_divisible_by_a_square_l913_91371


namespace NUMINAMATH_GPT_minimum_fence_length_l913_91396

theorem minimum_fence_length {x y : ℝ} (hxy : x * y = 100) : 2 * (x + y) ≥ 40 :=
by
  sorry

end NUMINAMATH_GPT_minimum_fence_length_l913_91396


namespace NUMINAMATH_GPT_range_of_function_l913_91348

theorem range_of_function :
  (∀ y : ℝ, (∃ x : ℝ, y = (x + 1) / (x ^ 2 + 1)) ↔ 0 ≤ y ∧ y ≤ 4/3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_l913_91348


namespace NUMINAMATH_GPT_probability_in_given_interval_l913_91351

noncomputable def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval (a b c d : ℝ) : ℝ :=
  (length_interval a b) / (length_interval c d)

theorem probability_in_given_interval : 
  probability_in_interval (-1) 1 (-2) 3 = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_in_given_interval_l913_91351


namespace NUMINAMATH_GPT_distance_between_Sneezy_and_Grumpy_is_8_l913_91355

variables (DS DV SP VP: ℕ) (SV: ℕ)

theorem distance_between_Sneezy_and_Grumpy_is_8
  (hDS : DS = 5)
  (hDV : DV = 4)
  (hSP : SP = 10)
  (hVP : VP = 17)
  (hSV_condition1 : SV + SP > VP)
  (hSV_condition2 : SV < DS + DV)
  (hSV_condition3 : 7 < SV) :
  SV = 8 := 
sorry

end NUMINAMATH_GPT_distance_between_Sneezy_and_Grumpy_is_8_l913_91355


namespace NUMINAMATH_GPT_work_completion_problem_l913_91392

theorem work_completion_problem :
  (∃ x : ℕ, 9 * (1 / 45 + 1 / x) + 23 * (1 / x) = 1) → x = 40 :=
sorry

end NUMINAMATH_GPT_work_completion_problem_l913_91392


namespace NUMINAMATH_GPT_maps_skipped_l913_91365

-- Definitions based on conditions
def total_pages := 372
def pages_read := 125
def pages_left := 231

-- Statement to be proven
theorem maps_skipped : total_pages - (pages_read + pages_left) = 16 :=
by
  sorry

end NUMINAMATH_GPT_maps_skipped_l913_91365


namespace NUMINAMATH_GPT_find_b_squared_l913_91341

-- Assume a and b are real numbers and positive
variables (a b : ℝ)
-- Given conditions
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom magnitude : a^2 + b^2 = 100
axiom equidistant : 2 * a - 4 * b = 7

-- Main proof statement
theorem find_b_squared : b^2 = 287 / 17 := sorry

end NUMINAMATH_GPT_find_b_squared_l913_91341


namespace NUMINAMATH_GPT_evaluate_expression_l913_91390

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by
  -- Proof is not required, add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_evaluate_expression_l913_91390


namespace NUMINAMATH_GPT_middle_number_is_correct_l913_91382

theorem middle_number_is_correct (numbers : List ℝ) (h_length : numbers.length = 11)
  (h_avg11 : numbers.sum / 11 = 9.9)
  (first_6 : List ℝ) (h_first6_length : first_6.length = 6)
  (h_avg6_1 : first_6.sum / 6 = 10.5)
  (last_6 : List ℝ) (h_last6_length : last_6.length = 6)
  (h_avg6_2 : last_6.sum / 6 = 11.4) :
  (∃ m : ℝ, m ∈ first_6 ∧ m ∈ last_6 ∧ m = 22.5) :=
by
  sorry

end NUMINAMATH_GPT_middle_number_is_correct_l913_91382


namespace NUMINAMATH_GPT_art_museum_survey_l913_91307

theorem art_museum_survey (V E : ℕ) 
  (h1 : ∀ (x : ℕ), x = 140 → ¬ (x ≤ E))
  (h2 : E = (3 / 4) * V)
  (h3 : V = E + 140) :
  V = 560 := by
  sorry

end NUMINAMATH_GPT_art_museum_survey_l913_91307


namespace NUMINAMATH_GPT_largest_angle_of_triangle_l913_91338

theorem largest_angle_of_triangle (x : ℝ) 
  (h1 : 35 + 70 + x = 180) : 75 = max (max 35 70) x := 
sorry

end NUMINAMATH_GPT_largest_angle_of_triangle_l913_91338


namespace NUMINAMATH_GPT_tangent_line_equation_l913_91343

def f (x : ℝ) : ℝ := x^2

theorem tangent_line_equation :
  let x := (1 : ℝ)
  let y := f x
  ∃ m b : ℝ, m = 2 ∧ b = 1 ∧ (2*x - y - 1 = 0) := by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l913_91343


namespace NUMINAMATH_GPT_arithmetic_sequence_statements_l913_91347

/-- 
Given the arithmetic sequence {a_n} with first term a_1 > 0 and the sum of the first n terms denoted as S_n, 
prove the following statements based on the condition S_8 = S_16:
  1. d > 0
  2. a_{13} < 0
  3. The maximum value of S_n is S_{12}
  4. When S_n < 0, the minimum value of n is 25
--/
theorem arithmetic_sequence_statements (a_1 d : ℤ) (S : ℕ → ℤ)
  (h1 : a_1 > 0)
  (h2 : S 8 = S 16)
  (hS8 : S 8 = 8 * a_1 + 28 * d)
  (hS16 : S 16 = 16 * a_1 + 120 * d) :
  (d > 0) ∨ 
  (a_1 + 12 * d < 0) ∨ 
  (∀ n, n ≠ 12 → S n ≤ S 12) ∨ 
  (∀ n, S n < 0 → n ≥ 25) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_statements_l913_91347


namespace NUMINAMATH_GPT_opposite_neg_fraction_l913_91375

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end NUMINAMATH_GPT_opposite_neg_fraction_l913_91375


namespace NUMINAMATH_GPT_eval_at_2_l913_91379

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem eval_at_2 : f 2 = 62 := by
  sorry

end NUMINAMATH_GPT_eval_at_2_l913_91379


namespace NUMINAMATH_GPT_reduce_expression_l913_91370

-- Define the variables a, b, c as real numbers
variables (a b c : ℝ)

-- State the theorem with the given condition that expressions are defined and non-zero
theorem reduce_expression :
  (a^2 + b^2 - c^2 - 2*a*b) / (a^2 + c^2 - b^2 - 2*a*c) = ((a - b + c) * (a - b - c)) / ((a - c + b) * (a - c - b)) :=
by
  sorry

end NUMINAMATH_GPT_reduce_expression_l913_91370


namespace NUMINAMATH_GPT_rectangles_with_equal_perimeters_can_have_different_shapes_l913_91363

theorem rectangles_with_equal_perimeters_can_have_different_shapes (l₁ w₁ l₂ w₂ : ℝ) 
  (h₁ : l₁ + w₁ = l₂ + w₂) : (l₁ ≠ l₂ ∨ w₁ ≠ w₂) :=
by
  sorry

end NUMINAMATH_GPT_rectangles_with_equal_perimeters_can_have_different_shapes_l913_91363


namespace NUMINAMATH_GPT_chocolate_bar_cost_l913_91362

def total_bars := 11
def bars_left := 7
def bars_sold := total_bars - bars_left
def total_money := 16
def cost := total_money / bars_sold

theorem chocolate_bar_cost : cost = 4 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bar_cost_l913_91362


namespace NUMINAMATH_GPT_max_a_is_fractional_value_l913_91386

theorem max_a_is_fractional_value (a k : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 - (k^2 - 5 * a * k + 3) * x + 7)
  (h_k : 0 ≤ k ∧ k ≤ 2)
  (x1 x2 : ℝ)
  (h_x1 : k ≤ x1 ∧ x1 ≤ k + a)
  (h_x2 : k + 2 * a ≤ x2 ∧ x2 ≤ k + 4 * a)
  (h_fx1_fx2 : f x1 ≥ f x2) :
  a = (2 * Real.sqrt 6 - 4) / 5 :=
sorry

end NUMINAMATH_GPT_max_a_is_fractional_value_l913_91386


namespace NUMINAMATH_GPT_abe_age_sum_l913_91393

theorem abe_age_sum (h : abe_age = 29) : abe_age + (abe_age - 7) = 51 :=
by
  sorry

end NUMINAMATH_GPT_abe_age_sum_l913_91393


namespace NUMINAMATH_GPT_trig_identity_l913_91373

theorem trig_identity : 4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_trig_identity_l913_91373


namespace NUMINAMATH_GPT_seats_per_bus_l913_91300

theorem seats_per_bus (students : ℕ) (buses : ℕ) (h1 : students = 111) (h2 : buses = 37) : students / buses = 3 := by
  sorry

end NUMINAMATH_GPT_seats_per_bus_l913_91300


namespace NUMINAMATH_GPT_pounds_over_weight_limit_l913_91320

-- Definitions based on conditions

def bookcase_max_weight : ℝ := 80

def weight_hardcover_book : ℝ := 0.5
def number_hardcover_books : ℕ := 70
def total_weight_hardcover_books : ℝ := number_hardcover_books * weight_hardcover_book

def weight_textbook : ℝ := 2
def number_textbooks : ℕ := 30
def total_weight_textbooks : ℝ := number_textbooks * weight_textbook

def weight_knick_knack : ℝ := 6
def number_knick_knacks : ℕ := 3
def total_weight_knick_knacks : ℝ := number_knick_knacks * weight_knick_knack

def total_weight_items : ℝ := total_weight_hardcover_books + total_weight_textbooks + total_weight_knick_knacks

theorem pounds_over_weight_limit : total_weight_items - bookcase_max_weight = 33 := by
  sorry

end NUMINAMATH_GPT_pounds_over_weight_limit_l913_91320


namespace NUMINAMATH_GPT_scientific_notation_of_5_35_million_l913_91321

theorem scientific_notation_of_5_35_million : 
  (5.35 : ℝ) * 10^6 = 5.35 * 10^6 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_5_35_million_l913_91321


namespace NUMINAMATH_GPT_even_n_divisible_into_equal_triangles_l913_91334

theorem even_n_divisible_into_equal_triangles (n : ℕ) (hn : 3 < n) :
  (∃ (triangles : ℕ), triangles = n) ↔ (∃ (k : ℕ), n = 2 * k) := 
sorry

end NUMINAMATH_GPT_even_n_divisible_into_equal_triangles_l913_91334


namespace NUMINAMATH_GPT_common_number_of_two_sets_l913_91344

theorem common_number_of_two_sets (a b c d e f g : ℚ) :
  (a + b + c + d) / 4 = 5 →
  (d + e + f + g) / 4 = 8 →
  (a + b + c + d + e + f + g) / 7 = 46 / 7 →
  d = 6 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_common_number_of_two_sets_l913_91344


namespace NUMINAMATH_GPT_time_in_2700_minutes_is_3_am_l913_91381

def minutes_in_hour : ℕ := 60
def hours_in_day : ℕ := 24
def current_hour : ℕ := 6
def minutes_later : ℕ := 2700

-- Calculate the final hour after adding the given minutes
def final_hour (current_hour minutes_later minutes_in_hour hours_in_day: ℕ) : ℕ :=
  (current_hour + (minutes_later / minutes_in_hour) % hours_in_day) % hours_in_day

theorem time_in_2700_minutes_is_3_am :
  final_hour current_hour minutes_later minutes_in_hour hours_in_day = 3 :=
by
  sorry

end NUMINAMATH_GPT_time_in_2700_minutes_is_3_am_l913_91381


namespace NUMINAMATH_GPT_addition_correct_l913_91313

-- Define the integers involved
def num1 : ℤ := 22
def num2 : ℤ := 62
def result : ℤ := 84

-- Theorem stating the relationship between the given numbers
theorem addition_correct : num1 + num2 = result :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_addition_correct_l913_91313


namespace NUMINAMATH_GPT_evaluate_expression_l913_91361

theorem evaluate_expression : (2301 - 2222)^2 / 144 = 43 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l913_91361


namespace NUMINAMATH_GPT_polar_bear_daily_salmon_consumption_l913_91368

/-- Polar bear's fish consumption conditions and daily salmon amount calculation -/
theorem polar_bear_daily_salmon_consumption (h1: ℝ) (h2: ℝ) : 
  (h1 = 0.2) → (h2 = 0.6) → (h2 - h1 = 0.4) :=
by
  sorry

end NUMINAMATH_GPT_polar_bear_daily_salmon_consumption_l913_91368


namespace NUMINAMATH_GPT_train_speed_l913_91387

def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 120
def time_taken : ℝ := 20
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train : ℝ := total_distance / time_taken

theorem train_speed : speed_of_train = 18.5 :=
  by sorry

end NUMINAMATH_GPT_train_speed_l913_91387


namespace NUMINAMATH_GPT_one_intersection_point_two_intersection_points_l913_91376

variables (k : ℝ)

-- Condition definitions
def parabola_eq (y x : ℝ) : Prop := y^2 = -4 * x
def line_eq (x y k : ℝ) : Prop := y + 1 = k * (x - 2)
def discriminant_non_negative (a b c : ℝ) : Prop := b^2 - 4 * a * c ≥ 0

-- Mathematically equivalent proof problem 1
theorem one_intersection_point (k : ℝ) : 
  (k = 1/2 ∨ k = -1 ∨ k = 0) → 
  ∃ x y : ℝ, parabola_eq y x ∧ line_eq x y k := sorry

-- Mathematically equivalent proof problem 2
theorem two_intersection_points (k : ℝ) : 
  (-1 < k ∧ k < 1/2 ∧ k ≠ 0) → 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
  (x₁ ≠ x₂ ∧ y₁ ≠ y₂) ∧ parabola_eq y₁ x₁ ∧ parabola_eq y₂ x₂ ∧ 
  line_eq x₁ y₁ k ∧ line_eq x₂ y₂ k := sorry

end NUMINAMATH_GPT_one_intersection_point_two_intersection_points_l913_91376


namespace NUMINAMATH_GPT_find_B_value_l913_91377

theorem find_B_value (A B : ℕ) : (A * 100 + B * 10 + 2) - 41 = 591 → B = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_B_value_l913_91377


namespace NUMINAMATH_GPT_find_three_digit_number_l913_91384

noncomputable def three_digit_number := ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ 100 * x + 10 * y + z = 345 ∧
  (100 * z + 10 * y + x = 100 * x + 10 * y + z + 198) ∧
  (100 * x + 10 * z + y = 100 * x + 10 * y + z + 9) ∧
  (x^2 + y^2 + z^2 - 2 = 4 * (x + y + z))

theorem find_three_digit_number : three_digit_number :=
sorry

end NUMINAMATH_GPT_find_three_digit_number_l913_91384


namespace NUMINAMATH_GPT_find_q_l913_91383

def P (q x : ℝ) : ℝ := x^4 + 2 * q * x^3 - 3 * x^2 + 2 * q * x + 1

theorem find_q (q : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ P q x1 = 0 ∧ P q x2 = 0) → q < 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l913_91383


namespace NUMINAMATH_GPT_adult_ticket_price_l913_91329

/-- 
The community center sells 85 tickets and collects $275 in total.
35 of those tickets are adult tickets. Each child's ticket costs $2.
We want to find the price of an adult ticket.
-/
theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (adult_tickets_sold : ℕ) 
  (child_ticket_price : ℚ)
  (h1 : total_tickets = 85)
  (h2 : total_revenue = 275) 
  (h3 : adult_tickets_sold = 35) 
  (h4 : child_ticket_price = 2) 
  : ∃ A : ℚ, (35 * A + 50 * 2 = 275) ∧ (A = 5) :=
by
  sorry

end NUMINAMATH_GPT_adult_ticket_price_l913_91329


namespace NUMINAMATH_GPT_find_integer_triplets_l913_91317

theorem find_integer_triplets (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_triplets_l913_91317


namespace NUMINAMATH_GPT_rainfall_on_wednesday_l913_91372

theorem rainfall_on_wednesday 
  (rain_on_monday : ℝ)
  (rain_on_tuesday : ℝ)
  (total_rain : ℝ) 
  (hmonday : rain_on_monday = 0.16666666666666666) 
  (htuesday : rain_on_tuesday = 0.4166666666666667) 
  (htotal : total_rain = 0.6666666666666666) :
  total_rain - (rain_on_monday + rain_on_tuesday) = 0.0833333333333333 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_rainfall_on_wednesday_l913_91372


namespace NUMINAMATH_GPT_total_tickets_sold_l913_91350

def ticket_prices : Nat := 25
def senior_ticket_price : Nat := 15
def total_receipts : Nat := 9745
def senior_tickets_sold : Nat := 348
def adult_tickets_sold : Nat := (total_receipts - senior_ticket_price * senior_tickets_sold) / ticket_prices

theorem total_tickets_sold : adult_tickets_sold + senior_tickets_sold = 529 :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l913_91350


namespace NUMINAMATH_GPT_inverse_variation_l913_91398

theorem inverse_variation (a b k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 1^3 = k) : (∃ a, b = 4 → a = 1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_inverse_variation_l913_91398
