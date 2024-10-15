import Mathlib

namespace NUMINAMATH_GPT_max_dist_2_minus_2i_l32_3297

open Complex

noncomputable def max_dist (z1 : ℂ) : ℝ :=
  Complex.abs 1 + Complex.abs z1

theorem max_dist_2_minus_2i :
  max_dist (2 - 2*I) = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_max_dist_2_minus_2i_l32_3297


namespace NUMINAMATH_GPT_income_M_l32_3286

variable (M N O : ℝ)

theorem income_M (h1 : (M + N) / 2 = 5050) 
                  (h2 : (N + O) / 2 = 6250) 
                  (h3 : (M + O) / 2 = 5200) : 
                  M = 2666.67 := 
by 
  sorry

end NUMINAMATH_GPT_income_M_l32_3286


namespace NUMINAMATH_GPT_compare_exponents_l32_3211

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem compare_exponents : b < a ∧ a < c :=
by
  have h1 : a = 2^(4/3) := rfl
  have h2 : b = 4^(2/5) := rfl
  have h3 : c = 25^(1/3) := rfl
  -- These are used to indicate the definitions, not the proof steps
  sorry

end NUMINAMATH_GPT_compare_exponents_l32_3211


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l32_3230

variable (a : ℕ → ℕ)
variable (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 2 - a 1)

theorem arithmetic_sequence_sum (h : a 2 + a 8 = 6) : 
  1 / 2 * 9 * (a 1 + a 9) = 27 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l32_3230


namespace NUMINAMATH_GPT_kylie_daisies_l32_3299

theorem kylie_daisies :
  ∀ (initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies : ℕ),
    initial_daisies = 5 →
    sister_daisies = 9 →
    final_daisies = 7 →
    total_daisies = initial_daisies + sister_daisies →
    daisies_given_to_mother = total_daisies - final_daisies →
    daisies_given_to_mother * 2 = total_daisies :=
by
  intros initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_kylie_daisies_l32_3299


namespace NUMINAMATH_GPT_min_abs_y1_minus_4y2_l32_3271

-- Definitions based on conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : (ℝ × ℝ) := (1, 0)

noncomputable def equation_of_line (k y : ℝ) : ℝ := k * y + 1

-- The Lean theorem statement
theorem min_abs_y1_minus_4y2 {x1 y1 x2 y2 : ℝ} (H1 : parabola x1 y1) (H2 : parabola x2 y2)
    (A_in_first_quadrant : 0 < x1 ∧ 0 < y1)
    (line_through_focus : ∃ k : ℝ, x1 = equation_of_line k y1 ∧ x2 = equation_of_line k y2)
    : |y1 - 4 * y2| = 8 :=
sorry

end NUMINAMATH_GPT_min_abs_y1_minus_4y2_l32_3271


namespace NUMINAMATH_GPT_revenue_percentage_l32_3252

theorem revenue_percentage (R C : ℝ) (hR_pos : R > 0) (hC_pos : C > 0) :
  let projected_revenue := 1.20 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 62.5 := by
  sorry

end NUMINAMATH_GPT_revenue_percentage_l32_3252


namespace NUMINAMATH_GPT_sphere_surface_area_l32_3200

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : V = 36 * π) 
  (h2 : V = (4 / 3) * π * r^3) 
  (h3 : A = 4 * π * r^2) 
  : A = 36 * π :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l32_3200


namespace NUMINAMATH_GPT_tv_sets_sales_decrease_l32_3217

theorem tv_sets_sales_decrease
  (P Q P' Q' R R': ℝ)
  (h1 : P' = 1.6 * P)
  (h2 : R' = 1.28 * R)
  (h3 : R = P * Q)
  (h4 : R' = P' * Q')
  (h5 : Q' = Q * (1 - D / 100)) :
  D = 20 :=
by
  sorry

end NUMINAMATH_GPT_tv_sets_sales_decrease_l32_3217


namespace NUMINAMATH_GPT_coordinates_of_A_l32_3214

-- Defining the point A
def point_A : ℤ × ℤ := (1, -4)

-- Statement that needs to be proved
theorem coordinates_of_A :
  point_A = (1, -4) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_A_l32_3214


namespace NUMINAMATH_GPT_common_root_for_equations_l32_3260

theorem common_root_for_equations : 
  ∃ p x : ℤ, 3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0 ∧ p = 3 ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_common_root_for_equations_l32_3260


namespace NUMINAMATH_GPT_find_f4_l32_3242

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def equilibrium_condition (f4 : ℝ × ℝ) : Prop :=
  f1 + f2 + f3 + f4 = (0, 0)

-- Statement that needs to be proven
theorem find_f4 : ∃ (f4 : ℝ × ℝ), equilibrium_condition f4 :=
  by
  use (1, 2)
  sorry

end NUMINAMATH_GPT_find_f4_l32_3242


namespace NUMINAMATH_GPT_blocks_remaining_l32_3275

def initial_blocks : ℕ := 55
def blocks_eaten : ℕ := 29

theorem blocks_remaining : initial_blocks - blocks_eaten = 26 := by
  sorry

end NUMINAMATH_GPT_blocks_remaining_l32_3275


namespace NUMINAMATH_GPT_find_number_l32_3236

theorem find_number (x : ℚ) (h : 1 + 1 / x = 5 / 2) : x = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l32_3236


namespace NUMINAMATH_GPT_quadrilateral_area_sum_l32_3238

theorem quadrilateral_area_sum (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a^2 * b = 36) : a + b = 4 := 
sorry

end NUMINAMATH_GPT_quadrilateral_area_sum_l32_3238


namespace NUMINAMATH_GPT_grandpa_rank_l32_3268

theorem grandpa_rank (mom dad grandpa : ℕ) 
  (h1 : mom < dad) 
  (h2 : dad < grandpa) : 
  ∀ rank: ℕ, rank = 3 := 
by
  sorry

end NUMINAMATH_GPT_grandpa_rank_l32_3268


namespace NUMINAMATH_GPT_value_of_expression_l32_3276

theorem value_of_expression (a b c k : ℕ) (h_a : a = 30) (h_b : b = 25) (h_c : c = 4) (h_k : k = 3) : 
  (a - (b - k * c)) - ((a - k * b) - c) = 66 :=
by
  rw [h_a, h_b, h_c, h_k]
  simp
  sorry

end NUMINAMATH_GPT_value_of_expression_l32_3276


namespace NUMINAMATH_GPT_circle_area_x2_y2_eq_102_l32_3284

theorem circle_area_x2_y2_eq_102 :
  ∀ (x y : ℝ), (x + 9)^2 + (y - 3)^2 = 102 → π * 102 = 102 * π :=
by
  intros
  sorry

end NUMINAMATH_GPT_circle_area_x2_y2_eq_102_l32_3284


namespace NUMINAMATH_GPT_find_angle_BXY_l32_3245

noncomputable def angle_AXE (angle_CYX : ℝ) : ℝ := 3 * angle_CYX - 108

theorem find_angle_BXY
  (AB_parallel_CD : Prop)
  (h_parallel : ∀ (AXE CYX : ℝ), angle_AXE CYX = AXE)
  (x : ℝ) :
  (angle_AXE x = x) → x = 54 :=
by
  intro h₁
  unfold angle_AXE at h₁
  sorry

end NUMINAMATH_GPT_find_angle_BXY_l32_3245


namespace NUMINAMATH_GPT_total_students_l32_3270

variable (A B AB : ℕ)

-- Conditions
axiom h1 : AB = (1 / 5) * (A + AB)
axiom h2 : AB = (1 / 4) * (B + AB)
axiom h3 : A - B = 75

-- Proof problem
theorem total_students : A + B + AB = 600 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l32_3270


namespace NUMINAMATH_GPT_symmetric_points_l32_3257

-- Let points P and Q be symmetric about the origin
variables (m n : ℤ)
axiom symmetry_condition : (m, 4) = (- (-2), -n)

theorem symmetric_points :
  m = 2 ∧ n = -4 := 
  by {
    sorry
  }

end NUMINAMATH_GPT_symmetric_points_l32_3257


namespace NUMINAMATH_GPT_negation_of_p_l32_3208

variable (p : Prop) (n : ℕ)

def proposition_p := ∃ n : ℕ, n^2 > 2^n

theorem negation_of_p : ¬ proposition_p ↔ ∀ n : ℕ, n^2 <= 2^n :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l32_3208


namespace NUMINAMATH_GPT_problem_statement_l32_3296

variable (f : ℝ → ℝ)

theorem problem_statement (h : ∀ x : ℝ, 2 * (f x) + x * (deriv f x) > x^2) :
  ∀ x : ℝ, x^2 * f x ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l32_3296


namespace NUMINAMATH_GPT_num_ordered_triples_l32_3255

theorem num_ordered_triples :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ∣ b ∧ a ∣ c ∧ a + b + c = 100) :=
  sorry

end NUMINAMATH_GPT_num_ordered_triples_l32_3255


namespace NUMINAMATH_GPT_convert_base_3_to_base_10_l32_3224

theorem convert_base_3_to_base_10 : 
  (1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0) = 142 :=
by
  sorry

end NUMINAMATH_GPT_convert_base_3_to_base_10_l32_3224


namespace NUMINAMATH_GPT_cookie_price_ratio_l32_3226

theorem cookie_price_ratio (c b : ℝ) (h1 : 6 * c + 5 * b = 3 * (3 * c + 27 * b)) : c = (4 / 5) * b :=
sorry

end NUMINAMATH_GPT_cookie_price_ratio_l32_3226


namespace NUMINAMATH_GPT_total_toucans_l32_3288

def initial_toucans : Nat := 2

def new_toucans : Nat := 1

theorem total_toucans : initial_toucans + new_toucans = 3 := by
  sorry

end NUMINAMATH_GPT_total_toucans_l32_3288


namespace NUMINAMATH_GPT_greatest_term_in_expansion_l32_3253

theorem greatest_term_in_expansion :
  ∃ k : ℕ, k = 63 ∧
  (∀ n : ℕ, n ∈ (Finset.range 101) → n ≠ k → 
    (Nat.choose 100 n * (Real.sqrt 3)^n) < 
    (Nat.choose 100 k * (Real.sqrt 3)^k)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_term_in_expansion_l32_3253


namespace NUMINAMATH_GPT_correlate_height_weight_l32_3228

-- Define the problems as types
def heightWeightCorrelated : Prop := true
def distanceTimeConstantSpeed : Prop := true
def heightVisionCorrelated : Prop := false
def volumeEdgeLengthCorrelated : Prop := true

-- Define the equivalence for the problem
def correlated : Prop := heightWeightCorrelated

-- Now state that correlated == heightWeightCorrelated
theorem correlate_height_weight : correlated = heightWeightCorrelated :=
by sorry

end NUMINAMATH_GPT_correlate_height_weight_l32_3228


namespace NUMINAMATH_GPT_correct_average_l32_3283

theorem correct_average (n : ℕ) (wrong_avg : ℕ) (wrong_num correct_num : ℕ) (correct_avg : ℕ)
  (h1 : n = 10) 
  (h2 : wrong_avg = 21)
  (h3 : wrong_num = 26)
  (h4 : correct_num = 36)
  (h5 : correct_avg = 22) :
  (wrong_avg * n + (correct_num - wrong_num)) / n = correct_avg :=
by
  sorry

end NUMINAMATH_GPT_correct_average_l32_3283


namespace NUMINAMATH_GPT_solve_for_r_l32_3231

noncomputable def k (r : ℝ) : ℝ := 5 / (2 ^ r)

theorem solve_for_r (r : ℝ) :
  (5 = k r * 2 ^ r) ∧ (45 = k r * 8 ^ r) → r = (Real.log 9 / Real.log 2) / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_r_l32_3231


namespace NUMINAMATH_GPT_additional_track_length_l32_3278

theorem additional_track_length (rise : ℝ) (grade1 grade2 : ℝ) (h1 : grade1 = 0.04) (h2 : grade2 = 0.02) (h3 : rise = 800) :
  ∃ (additional_length : ℝ), additional_length = (rise / grade2 - rise / grade1) ∧ additional_length = 20000 :=
by
  sorry

end NUMINAMATH_GPT_additional_track_length_l32_3278


namespace NUMINAMATH_GPT_max_value_2x_minus_y_l32_3241

theorem max_value_2x_minus_y 
  (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  2 * x - y ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_2x_minus_y_l32_3241


namespace NUMINAMATH_GPT_max_profit_l32_3202

noncomputable def C (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then (1 / 3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def L (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then -(1 / 3) * x^2 + 40 * x - 250
  else -(x + 10000 / x) + 1200

theorem max_profit :
  ∃ x : ℝ, (L x) = 1000 ∧ x = 100 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_l32_3202


namespace NUMINAMATH_GPT_melted_mixture_weight_l32_3222

-- Let Zinc and Copper be real numbers representing their respective weights in kilograms.
variables (Zinc Copper: ℝ)
-- Assume the ratio of Zinc to Copper is 9:11.
axiom ratio_zinc_copper : Zinc / Copper = 9 / 11
-- Assume 26.1kg of Zinc has been used.
axiom zinc_value : Zinc = 26.1

-- Define the total weight of the melted mixture.
def total_weight := Zinc + Copper

-- We state the theorem to prove that the total weight of the mixture equals 58kg.
theorem melted_mixture_weight : total_weight Zinc Copper = 58 :=
by
  sorry

end NUMINAMATH_GPT_melted_mixture_weight_l32_3222


namespace NUMINAMATH_GPT_tank_ratio_l32_3223

theorem tank_ratio (V1 V2 : ℝ) (h1 : 0 < V1) (h2 : 0 < V2) (h1_full : 3 / 4 * V1 - 7 / 20 * V2 = 0) (h2_full : 1 / 4 * V2 + 7 / 20 * V2 = 3 / 5 * V2) :
  V1 / V2 = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_tank_ratio_l32_3223


namespace NUMINAMATH_GPT_sum_of_ages_l32_3205

-- Define the variables for Viggo and his younger brother's ages
variables (v y : ℕ)

-- Condition: When Viggo's younger brother was 2, Viggo's age was 10 years more than twice his brother's age
def condition1 (v y : ℕ) := (y = 2 → v = 2 * y + 10)

-- Condition: Viggo's younger brother is currently 10 years old
def condition2 (y_current : ℕ) := y_current = 10

-- Define the current age of Viggo given the conditions
def viggo_current_age (v y y_current : ℕ) := v + (y_current - y)

-- Prove that the sum of their ages is 32
theorem sum_of_ages
  (v y y_current : ℕ)
  (h1 : condition1 v y)
  (h2 : condition2 y_current) :
  viggo_current_age v y y_current + y_current = 32 :=
by
  -- Apply sorry to skip the proof
  sorry

end NUMINAMATH_GPT_sum_of_ages_l32_3205


namespace NUMINAMATH_GPT_sophie_total_spending_l32_3273

-- Definitions based on conditions
def num_cupcakes : ℕ := 5
def price_per_cupcake : ℝ := 2
def num_doughnuts : ℕ := 6
def price_per_doughnut : ℝ := 1
def num_slices_apple_pie : ℕ := 4
def price_per_slice_apple_pie : ℝ := 2
def num_cookies : ℕ := 15
def price_per_cookie : ℝ := 0.60

-- Total cost calculation
def total_cost : ℝ :=
  num_cupcakes * price_per_cupcake +
  num_doughnuts * price_per_doughnut +
  num_slices_apple_pie * price_per_slice_apple_pie +
  num_cookies * price_per_cookie

-- Theorem stating the total cost is 33
theorem sophie_total_spending : total_cost = 33 := by
  sorry

end NUMINAMATH_GPT_sophie_total_spending_l32_3273


namespace NUMINAMATH_GPT_find_k_l32_3294

noncomputable def digit_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem find_k :
  ∃ k : ℕ, digit_sum (5 * (5 * (10 ^ (k - 1) - 1) / 9)) = 600 ∧ k = 87 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l32_3294


namespace NUMINAMATH_GPT_max_disks_l32_3279

theorem max_disks (n k : ℕ) (h1: n ≥ 1) (h2: k ≥ 1) :
  (∃ (d : ℕ), d = if n > 1 ∧ k > 1 then 2 * (n + k) - 4 else max n k) ∧
  (∀ (p q : ℕ), (p <= n → q <= k → ¬∃ (x y : ℕ), x + 1 = y ∨ x - 1 = y ∨ x + 1 = p ∨ x - 1 = p)) :=
sorry

end NUMINAMATH_GPT_max_disks_l32_3279


namespace NUMINAMATH_GPT_inequality_and_equality_conditions_l32_3203

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ abc ≤ 1 ∧ ((a + b + c = 3) → (a = 1 ∧ b = 1 ∧ c = 1)) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_and_equality_conditions_l32_3203


namespace NUMINAMATH_GPT_area_of_given_trapezium_l32_3293

def area_of_trapezium (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem area_of_given_trapezium :
  area_of_trapezium 20 18 25 = 475 :=
by
  sorry

end NUMINAMATH_GPT_area_of_given_trapezium_l32_3293


namespace NUMINAMATH_GPT_medium_bed_rows_l32_3201

theorem medium_bed_rows (large_top_beds : ℕ) (large_bed_rows : ℕ) (large_bed_seeds_per_row : ℕ) 
                         (medium_beds : ℕ) (medium_bed_seeds_per_row : ℕ) (total_seeds : ℕ) :
    large_top_beds = 2 ∧ large_bed_rows = 4 ∧ large_bed_seeds_per_row = 25 ∧
    medium_beds = 2 ∧ medium_bed_seeds_per_row = 20 ∧ total_seeds = 320 →
    ((total_seeds - (large_top_beds * large_bed_rows * large_bed_seeds_per_row)) / medium_bed_seeds_per_row) = 6 :=
by
  intro conditions
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := conditions
  sorry

end NUMINAMATH_GPT_medium_bed_rows_l32_3201


namespace NUMINAMATH_GPT_part1_part2_l32_3215

open Complex

def equation (a z : ℂ) : Prop := z^2 - (a + I) * z - (I + 2) = 0

theorem part1 (m : ℝ) (a : ℝ) : equation a m → a = 1 := by
  sorry

theorem part2 (a : ℝ) : ¬ ∃ n : ℝ, equation a (n * I) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l32_3215


namespace NUMINAMATH_GPT_Q_mul_P_plus_Q_eq_one_l32_3281

noncomputable def sqrt5_plus_2_pow (n : ℕ) :=
  (Real.sqrt 5 + 2)^(2 * n + 1)

noncomputable def P (n : ℕ) :=
  Int.floor (sqrt5_plus_2_pow n)

noncomputable def Q (n : ℕ) :=
  sqrt5_plus_2_pow n - P n

theorem Q_mul_P_plus_Q_eq_one (n : ℕ) : Q n * (P n + Q n) = 1 := by
  sorry

end NUMINAMATH_GPT_Q_mul_P_plus_Q_eq_one_l32_3281


namespace NUMINAMATH_GPT_machine_production_percentage_difference_l32_3221

theorem machine_production_percentage_difference 
  (X_production_rate : ℕ := 3)
  (widgets_to_produce : ℕ := 1080)
  (difference_in_hours : ℕ := 60) :
  ((widgets_to_produce / (widgets_to_produce / X_production_rate - difference_in_hours) - 
   X_production_rate) / X_production_rate * 100) = 20 := by
  sorry

end NUMINAMATH_GPT_machine_production_percentage_difference_l32_3221


namespace NUMINAMATH_GPT_skirt_price_l32_3239

theorem skirt_price (S : ℝ) 
  (h1 : 2 * 5 = 10) 
  (h2 : 1 * 4 = 4) 
  (h3 : 6 * (5 / 2) = 15) 
  (h4 : 10 + 4 + 15 + 4 * S = 53) 
  : S = 6 :=
sorry

end NUMINAMATH_GPT_skirt_price_l32_3239


namespace NUMINAMATH_GPT_zach_fill_time_l32_3225

theorem zach_fill_time : 
  ∀ (t : ℕ), 
  (∀ (max_time max_rate zach_rate popped total : ℕ), 
    max_time = 30 → 
    max_rate = 2 → 
    zach_rate = 3 → 
    popped = 10 → 
    total = 170 → 
    (max_time * max_rate + t * zach_rate - popped = total) → 
    t = 40) := 
sorry

end NUMINAMATH_GPT_zach_fill_time_l32_3225


namespace NUMINAMATH_GPT_bob_max_candies_l32_3274

theorem bob_max_candies (b : ℕ) (h : b + 2 * b = 30) : b = 10 := 
sorry

end NUMINAMATH_GPT_bob_max_candies_l32_3274


namespace NUMINAMATH_GPT_gcd_84_210_l32_3247

theorem gcd_84_210 : Nat.gcd 84 210 = 42 :=
by {
  sorry
}

end NUMINAMATH_GPT_gcd_84_210_l32_3247


namespace NUMINAMATH_GPT_poly_coefficients_sum_l32_3263

theorem poly_coefficients_sum :
  ∀ (x A B C D : ℝ),
  (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 2 :=
by sorry

end NUMINAMATH_GPT_poly_coefficients_sum_l32_3263


namespace NUMINAMATH_GPT_stone_hitting_ground_time_l32_3262

noncomputable def equation (s : ℝ) : ℝ := -4.5 * s^2 - 12 * s + 48

theorem stone_hitting_ground_time :
  ∃ s : ℝ, equation s = 0 ∧ s = (-8 + 16 * Real.sqrt 7) / 6 :=
by
  sorry

end NUMINAMATH_GPT_stone_hitting_ground_time_l32_3262


namespace NUMINAMATH_GPT_perpendicular_bisector_correct_vertex_C_correct_l32_3289

-- Define the vertices A, B, and the coordinates of the angle bisector line
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := -1, y := -1 }

-- The angle bisector CD equation
def angle_bisector_CD (p : Point) : Prop :=
  p.x + p.y - 1 = 0

-- The perpendicular bisector equation of side AB
def perpendicular_bisector_AB (p : Point) : Prop :=
  4 * p.x + 6 * p.y - 3 = 0

-- Coordinates of vertex C
def C_coordinates (c : Point) : Prop :=
  c.x = -1 ∧ c.y = 2

theorem perpendicular_bisector_correct :
  ∀ (M : Point), M.x = 0 ∧ M.y = 1/2 →
  ∀ (p : Point), perpendicular_bisector_AB p :=
sorry

theorem vertex_C_correct :
  ∃ (C : Point), angle_bisector_CD C ∧ (C : Point) = { x := -1, y := 2 } :=
sorry

end NUMINAMATH_GPT_perpendicular_bisector_correct_vertex_C_correct_l32_3289


namespace NUMINAMATH_GPT_arithmetic_sequence_contains_term_l32_3251

theorem arithmetic_sequence_contains_term (a1 : ℤ) (d : ℤ) (k : ℕ) (h1 : a1 = 3) (h2 : d = 9) :
  ∃ n : ℕ, (a1 + (n - 1) * d) = 3 * 4 ^ k := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_contains_term_l32_3251


namespace NUMINAMATH_GPT_number_below_267_is_301_l32_3256

-- Define the row number function
def rowNumber (n : ℕ) : ℕ :=
  Nat.sqrt n + 1

-- Define the starting number of a row
def rowStart (k : ℕ) : ℕ :=
  (k - 1) * (k - 1) + 1

-- Define the number in the row below given a number and its position in the row
def numberBelow (n : ℕ) : ℕ :=
  let k := rowNumber n
  let startK := rowStart k
  let position := n - startK
  let startNext := rowStart (k + 1)
  startNext + position

-- Prove that the number below 267 is 301
theorem number_below_267_is_301 : numberBelow 267 = 301 :=
by
  -- skip proof details, just the statement is needed
  sorry

end NUMINAMATH_GPT_number_below_267_is_301_l32_3256


namespace NUMINAMATH_GPT_sufficient_not_necessary_a_equals_2_l32_3233

theorem sufficient_not_necessary_a_equals_2 {a : ℝ} :
  (∃ a : ℝ, (a = 2 ∧ 15 * a^2 = 60) → (15 * a^2 = 60) ∧ (15 * a^2 = 60 → a = 2)) → 
  (¬∀ a : ℝ, (15 * a^2 = 60) → a = 2) → 
  (a = 2 → 15 * a^2 = 60) ∧ ¬(15 * a^2 = 60 → a = 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_a_equals_2_l32_3233


namespace NUMINAMATH_GPT_unique_solution_m_n_l32_3235

theorem unique_solution_m_n (m n : ℕ) (h1 : m > 1) (h2 : (n - 1) % (m - 1) = 0) 
  (h3 : ¬ ∃ k : ℕ, n = m ^ k) :
  ∃! (a b c : ℕ), a + m * b = n ∧ a + b = m * c := 
sorry

end NUMINAMATH_GPT_unique_solution_m_n_l32_3235


namespace NUMINAMATH_GPT_min_ab_bound_l32_3229

theorem min_ab_bound (a b n : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_n : 0 < n) 
                      (h : ∀ i j, i ≤ n → j ≤ n → Nat.gcd (a + i) (b + j) > 1) :
  ∃ c > 0, min a b > c^n * n^(n/2) :=
sorry

end NUMINAMATH_GPT_min_ab_bound_l32_3229


namespace NUMINAMATH_GPT_multiple_of_6_is_multiple_of_2_and_3_l32_3298

theorem multiple_of_6_is_multiple_of_2_and_3 (n : ℕ) :
  (∃ k : ℕ, n = 6 * k) → (∃ m1 : ℕ, n = 2 * m1) ∧ (∃ m2 : ℕ, n = 3 * m2) := by
  sorry

end NUMINAMATH_GPT_multiple_of_6_is_multiple_of_2_and_3_l32_3298


namespace NUMINAMATH_GPT_mailman_junk_mail_l32_3212

theorem mailman_junk_mail (total_mail : ℕ) (magazines : ℕ) (junk_mail : ℕ) 
  (h1 : total_mail = 11) (h2 : magazines = 5) (h3 : junk_mail = total_mail - magazines) : junk_mail = 6 := by
  sorry

end NUMINAMATH_GPT_mailman_junk_mail_l32_3212


namespace NUMINAMATH_GPT_number_of_tiles_l32_3248

open Real

noncomputable def room_length : ℝ := 10
noncomputable def room_width : ℝ := 15
noncomputable def tile_length : ℝ := 5 / 12
noncomputable def tile_width : ℝ := 2 / 3

theorem number_of_tiles :
  (room_length * room_width) / (tile_length * tile_width) = 540 := by
  sorry

end NUMINAMATH_GPT_number_of_tiles_l32_3248


namespace NUMINAMATH_GPT_projectile_height_time_l32_3277

theorem projectile_height_time (h : ∀ t : ℝ, -16 * t^2 + 100 * t = 64 → t = 1) : (∃ t : ℝ, -16 * t^2 + 100 * t = 64 ∧ t = 1) :=
by sorry

end NUMINAMATH_GPT_projectile_height_time_l32_3277


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l32_3216

theorem arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (c : ℤ) :
  (∀ n : ℕ, 0 < n → S_n n = n^2 + c) →
  a_n 1 = 1 + c →
  (∀ n, 1 < n → a_n n = S_n n - S_n (n - 1)) →
  (∀ n : ℕ, 0 < n → a_n n = 1 + (n - 1) * 2) →
  c = 0 ∧ (∀ n : ℕ, 0 < n → a_n n = 2 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l32_3216


namespace NUMINAMATH_GPT_find_dividend_l32_3219

-- Given conditions as definitions
def divisor : ℕ := 16
def quotient : ℕ := 9
def remainder : ℕ := 5

-- Lean 4 statement to be proven
theorem find_dividend : divisor * quotient + remainder = 149 := by
  sorry

end NUMINAMATH_GPT_find_dividend_l32_3219


namespace NUMINAMATH_GPT_range_of_a_l32_3207

theorem range_of_a
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg_x : ∀ x, x ≤ 0 → f x = 2 * x + x^2)
  (h_three_solutions : ∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 2 * a^2 + a ∧ f x2 = 2 * a^2 + a ∧ f x3 = 2 * a^2 + a) :
  -1 < a ∧ a < 1/2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l32_3207


namespace NUMINAMATH_GPT_range_of_k_l32_3249

noncomputable def quadratic_has_real_roots (k : ℝ): Prop :=
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l32_3249


namespace NUMINAMATH_GPT_john_profit_l32_3261

theorem john_profit (cost price : ℕ) (n : ℕ) (h1 : cost = 4) (h2 : price = 8) (h3 : n = 30) : 
  n * (price - cost) = 120 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_john_profit_l32_3261


namespace NUMINAMATH_GPT_pell_eq_unique_fund_sol_l32_3254

theorem pell_eq_unique_fund_sol (x y x_0 y_0 : ℕ) 
  (h1 : x_0^2 - 2003 * y_0^2 = 1) 
  (h2 : ∀ x y, x > 0 ∧ y > 0 → x^2 - 2003 * y^2 = 1 → ∃ n : ℕ, x + Real.sqrt 2003 * y = (x_0 + Real.sqrt 2003 * y_0)^n)
  (hx_pos : x > 0) 
  (hy_pos : y > 0)
  (h_sol : x^2 - 2003 * y^2 = 1) 
  (hprime : ∀ p : ℕ, Prime p → p ∣ x → p ∣ x_0)
  : x = x_0 ∧ y = y_0 :=
sorry

end NUMINAMATH_GPT_pell_eq_unique_fund_sol_l32_3254


namespace NUMINAMATH_GPT_point_in_second_quadrant_l32_3213

def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

theorem point_in_second_quadrant : is_in_second_quadrant (-2) 3 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l32_3213


namespace NUMINAMATH_GPT_sum_max_min_values_of_g_l32_3240

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_max_min_values_of_g : (∀ x, 1 ≤ x ∧ x ≤ 7 → g x = 15 - 2 * x ∨ g x = 5) ∧ 
      (g 1 = 13 ∧ g 5 = 5)
      → (13 + 5 = 18) :=
by
  sorry

end NUMINAMATH_GPT_sum_max_min_values_of_g_l32_3240


namespace NUMINAMATH_GPT_negation_of_p_equiv_h_l32_3210

variable (p : ∀ x : ℝ, Real.sin x ≤ 1)
variable (h : ∃ x : ℝ, Real.sin x ≥ 1)

theorem negation_of_p_equiv_h : (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_equiv_h_l32_3210


namespace NUMINAMATH_GPT_no_nat_p_prime_and_p6_plus_6_prime_l32_3272

theorem no_nat_p_prime_and_p6_plus_6_prime (p : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime (p^6 + 6)) : False := 
sorry

end NUMINAMATH_GPT_no_nat_p_prime_and_p6_plus_6_prime_l32_3272


namespace NUMINAMATH_GPT_min_distinct_lines_for_polyline_l32_3258

theorem min_distinct_lines_for_polyline (n : ℕ) (h_n : n = 31) : 
  ∃ (k : ℕ), 9 ≤ k ∧ k ≤ 31 ∧ 
  (∀ (s : Fin n → Fin 31), 
     ∀ i j, i ≠ j → s i ≠ s j) := 
sorry

end NUMINAMATH_GPT_min_distinct_lines_for_polyline_l32_3258


namespace NUMINAMATH_GPT_area_of_common_part_geq_3484_l32_3295

theorem area_of_common_part_geq_3484 :
  ∀ (R : ℝ) (S T : ℝ → Prop), 
  (R = 1) →
  (∀ x y, S x ↔ (x * x + y * y = R * R) ∧ T y) →
  ∃ (S_common : ℝ) (T_common : ℝ),
    (S_common + T_common > 3.484) :=
by
  sorry

end NUMINAMATH_GPT_area_of_common_part_geq_3484_l32_3295


namespace NUMINAMATH_GPT_phase_shift_of_sine_l32_3266

theorem phase_shift_of_sine :
  let B := 5
  let C := (3 * Real.pi) / 2
  let phase_shift := C / B
  phase_shift = (3 * Real.pi) / 10 := by
    sorry

end NUMINAMATH_GPT_phase_shift_of_sine_l32_3266


namespace NUMINAMATH_GPT_ratio_size12_to_size6_l32_3285

-- Definitions based on conditions
def cheerleaders_size2 : ℕ := 4
def cheerleaders_size6 : ℕ := 10
def total_cheerleaders : ℕ := 19
def cheerleaders_size12 : ℕ := total_cheerleaders - (cheerleaders_size2 + cheerleaders_size6)

-- Proof statement
theorem ratio_size12_to_size6 : cheerleaders_size12.toFloat / cheerleaders_size6.toFloat = 1 / 2 := sorry

end NUMINAMATH_GPT_ratio_size12_to_size6_l32_3285


namespace NUMINAMATH_GPT_arithmetic_sequence_a8_l32_3218

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_a8 (h : a 7 + a 9 = 8) : a 8 = 4 := 
by 
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a8_l32_3218


namespace NUMINAMATH_GPT_incorrect_eqn_x9_y9_neg1_l32_3265

theorem incorrect_eqn_x9_y9_neg1 (x y : ℂ) 
  (hx : x = (-1 + Complex.I * Real.sqrt 3) / 2) 
  (hy : y = (-1 - Complex.I * Real.sqrt 3) / 2) : 
  x^9 + y^9 ≠ -1 :=
sorry

end NUMINAMATH_GPT_incorrect_eqn_x9_y9_neg1_l32_3265


namespace NUMINAMATH_GPT_inequality_solution_l32_3287

theorem inequality_solution (x : ℝ) :
  (6*x^2 + 24*x - 63) / ((3*x - 4)*(x + 5)) < 4 ↔ x ∈ Set.Ioo (-(5:ℝ)) (4 / 3) ∪ Set.Iio (5) ∪ Set.Ioi (4 / 3) := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l32_3287


namespace NUMINAMATH_GPT_men_in_second_group_l32_3267

theorem men_in_second_group (M : ℕ) (W : ℝ) (h1 : 15 * 25 = W) (h2 : M * 18.75 = W) : M = 20 :=
sorry

end NUMINAMATH_GPT_men_in_second_group_l32_3267


namespace NUMINAMATH_GPT_g_at_3_l32_3269

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2) : g 3 = 0 := by
  sorry

end NUMINAMATH_GPT_g_at_3_l32_3269


namespace NUMINAMATH_GPT_pyramid_cross_section_distance_l32_3246

theorem pyramid_cross_section_distance
  (area1 area2 : ℝ) (distance : ℝ)
  (h1 : area1 = 100 * Real.sqrt 3) 
  (h2 : area2 = 225 * Real.sqrt 3) 
  (h3 : distance = 5) : 
  ∃ h : ℝ, h = 15 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_cross_section_distance_l32_3246


namespace NUMINAMATH_GPT_find_z_l32_3290

noncomputable def w : ℝ := sorry
noncomputable def x : ℝ := (5 * w) / 4
noncomputable def y : ℝ := 1.40 * w

theorem find_z (z : ℝ) : x = (1 - z / 100) * y → z = 10.71 :=
by
  sorry

end NUMINAMATH_GPT_find_z_l32_3290


namespace NUMINAMATH_GPT_reciprocal_self_eq_one_or_neg_one_l32_3227

theorem reciprocal_self_eq_one_or_neg_one (x : ℝ) (h : x = 1 / x) : x = 1 ∨ x = -1 := sorry

end NUMINAMATH_GPT_reciprocal_self_eq_one_or_neg_one_l32_3227


namespace NUMINAMATH_GPT_molecular_weight_of_10_moles_of_Al2S3_l32_3220

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06

-- Define the molecular weight calculation for Al2S3
def molecular_weight_Al2S3 : ℝ :=
  (2 * atomic_weight_Al) + (3 * atomic_weight_S)

-- Define the molecular weight for 10 moles of Al2S3
def molecular_weight_10_moles_Al2S3 : ℝ :=
  10 * molecular_weight_Al2S3

-- The theorem to prove
theorem molecular_weight_of_10_moles_of_Al2S3 :
  molecular_weight_10_moles_Al2S3 = 1501.4 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_molecular_weight_of_10_moles_of_Al2S3_l32_3220


namespace NUMINAMATH_GPT_other_root_of_quadratic_l32_3209

theorem other_root_of_quadratic (a b k : ℝ) (h : 1^2 - (a+b) * 1 + ab * (1 - k) = 0) : 
  ∃ r : ℝ, r = a + b - 1 := 
sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l32_3209


namespace NUMINAMATH_GPT_find_original_amount_l32_3250

-- Let X be the original amount of money in Christina's account.
variable (X : ℝ)

-- Condition 1: Remaining balance after transferring 20% is $30,000.
def initial_transfer (X : ℝ) : Prop :=
  0.80 * X = 30000

-- Prove that the original amount before the initial transfer was $37,500.
theorem find_original_amount (h : initial_transfer X) : X = 37500 :=
  sorry

end NUMINAMATH_GPT_find_original_amount_l32_3250


namespace NUMINAMATH_GPT_cirrus_clouds_count_l32_3292

theorem cirrus_clouds_count 
  (cirrus cumulus cumulonimbus : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = 12 * cumulonimbus)
  (h3 : cumulonimbus = 3) : 
  cirrus = 144 := 
by
  sorry

end NUMINAMATH_GPT_cirrus_clouds_count_l32_3292


namespace NUMINAMATH_GPT_percentage_increase_l32_3206

theorem percentage_increase (regular_rate : ℝ) (regular_hours total_compensation total_hours_worked : ℝ)
  (h1 : regular_rate = 20)
  (h2 : regular_hours = 40)
  (h3 : total_compensation = 1000)
  (h4 : total_hours_worked = 45.714285714285715) :
  let overtime_hours := total_hours_worked - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_compensation - regular_pay
  let overtime_rate := overtime_pay / overtime_hours
  let percentage_increase := ((overtime_rate - regular_rate) / regular_rate) * 100
  percentage_increase = 75 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l32_3206


namespace NUMINAMATH_GPT_find_x_l32_3264

theorem find_x (x : ℝ) (hx : x > 0) (condition : (2 * x / 100) * x = 10) : x = 10 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l32_3264


namespace NUMINAMATH_GPT_speed_in_still_water_l32_3204

-- We define the given conditions for the man's rowing speeds
def upstream_speed : ℕ := 25
def downstream_speed : ℕ := 35

-- We want to prove that the speed in still water is 30 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 := by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l32_3204


namespace NUMINAMATH_GPT_triangle_angles_l32_3234

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180

theorem triangle_angles (x : ℝ) (hA : A = x) (hB : B = 2 * A) (hC : C + A + B = 180) :
  A = x ∧ B = 2 * x ∧ C = 180 - 3 * x := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_triangle_angles_l32_3234


namespace NUMINAMATH_GPT_zoo_total_animals_l32_3244

theorem zoo_total_animals (penguins polar_bears : ℕ)
  (h1 : penguins = 21)
  (h2 : polar_bears = 2 * penguins) :
  penguins + polar_bears = 63 := by
   sorry

end NUMINAMATH_GPT_zoo_total_animals_l32_3244


namespace NUMINAMATH_GPT_find_total_sales_l32_3243

theorem find_total_sales
  (S : ℝ)
  (h_comm1 : ∀ x, x ≤ 5000 → S = 0.9 * x → S = 16666.67 → false)
  (h_comm2 : S > 5000 → S - (500 + 0.05 * (S - 5000)) = 15000):
  S = 16052.63 :=
by
  sorry

end NUMINAMATH_GPT_find_total_sales_l32_3243


namespace NUMINAMATH_GPT_problem_statement_l32_3259

theorem problem_statement (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 ∧ x^2 + y^2 = 104 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l32_3259


namespace NUMINAMATH_GPT_average_height_is_64_l32_3232

noncomputable def Parker (H_D : ℝ) : ℝ := H_D - 4
noncomputable def Daisy (H_R : ℝ) : ℝ := H_R + 8
noncomputable def Reese : ℝ := 60

theorem average_height_is_64 :
  let H_R := Reese 
  let H_D := Daisy H_R
  let H_P := Parker H_D
  (H_P + H_D + H_R) / 3 = 64 := sorry

end NUMINAMATH_GPT_average_height_is_64_l32_3232


namespace NUMINAMATH_GPT_daily_harvest_sacks_l32_3280

theorem daily_harvest_sacks (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 65 → num_sections = 12 → total_sacks = sacks_per_section * num_sections → total_sacks = 780 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_daily_harvest_sacks_l32_3280


namespace NUMINAMATH_GPT_gcd_m_n_l32_3237

def m : ℕ := 555555555
def n : ℕ := 1111111111

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_l32_3237


namespace NUMINAMATH_GPT_technology_courses_correct_l32_3282

variable (m : ℕ)

def subject_courses := m
def arts_courses := subject_courses + 9
def technology_courses := 1 / 3 * arts_courses + 5

theorem technology_courses_correct : technology_courses = 1 / 3 * m + 8 := by
  sorry

end NUMINAMATH_GPT_technology_courses_correct_l32_3282


namespace NUMINAMATH_GPT_percentage_increase_l32_3291

theorem percentage_increase (x : ℝ) (h1 : 75 + 0.75 * x * 0.8 = 72) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l32_3291
