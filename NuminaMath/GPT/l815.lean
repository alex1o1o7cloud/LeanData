import Mathlib

namespace NUMINAMATH_GPT_find_Japanese_students_l815_81588

theorem find_Japanese_students (C K J : ℕ) (hK: K = (6 * C) / 11) (hJ: J = C / 8) (hK_value: K = 48) : J = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_Japanese_students_l815_81588


namespace NUMINAMATH_GPT_find_f_l815_81587

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := sorry

axiom f_0 : f 0 = 0
axiom f_xy (x y : ℝ) : f (x * y) = f ((x^2 + y^2) / 2) + 3 * (x - y)^2

-- Theorem to be proved
theorem find_f (x : ℝ) : f x = -6 * x + 3 :=
by sorry -- proof goes here

end NUMINAMATH_GPT_find_f_l815_81587


namespace NUMINAMATH_GPT_cost_of_candy_l815_81556

theorem cost_of_candy (initial_amount remaining_amount : ℕ) (h_init : initial_amount = 4) (h_remaining : remaining_amount = 3) : initial_amount - remaining_amount = 1 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_candy_l815_81556


namespace NUMINAMATH_GPT_imaginary_number_condition_fourth_quadrant_condition_l815_81594

-- Part 1: Prove that if \( z \) is purely imaginary, then \( m = 0 \)
theorem imaginary_number_condition (m : ℝ) :
  (m * (m + 2) = 0) ∧ (m^2 + m - 2 ≠ 0) → m = 0 :=
by
  sorry

-- Part 2: Prove that if \( z \) is in the fourth quadrant, then \( 0 < m < 1 \)
theorem fourth_quadrant_condition (m : ℝ) :
  (m * (m + 2) > 0) ∧ (m^2 + m - 2 < 0) → (0 < m ∧ m < 1) :=
by
  sorry

end NUMINAMATH_GPT_imaginary_number_condition_fourth_quadrant_condition_l815_81594


namespace NUMINAMATH_GPT_jungkook_has_smallest_collection_l815_81531

-- Define the collections
def yoongi_collection : ℕ := 7
def jungkook_collection : ℕ := 6
def yuna_collection : ℕ := 9

-- State the theorem
theorem jungkook_has_smallest_collection : 
  jungkook_collection = min yoongi_collection (min jungkook_collection yuna_collection) := 
by
  sorry

end NUMINAMATH_GPT_jungkook_has_smallest_collection_l815_81531


namespace NUMINAMATH_GPT_tax_increase_proof_l815_81526

variables (old_tax_rate new_tax_rate : ℝ) (old_income new_income : ℝ)

def old_taxes_paid (old_tax_rate old_income : ℝ) : ℝ := old_tax_rate * old_income

def new_taxes_paid (new_tax_rate new_income : ℝ) : ℝ := new_tax_rate * new_income

def increase_in_taxes (old_tax_rate new_tax_rate old_income new_income : ℝ) : ℝ :=
  new_taxes_paid new_tax_rate new_income - old_taxes_paid old_tax_rate old_income

theorem tax_increase_proof :
  increase_in_taxes 0.20 0.30 1000000 1500000 = 250000 := by
  sorry

end NUMINAMATH_GPT_tax_increase_proof_l815_81526


namespace NUMINAMATH_GPT_small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l815_81576

-- 1. Prove that the small frog can reach the 7th rung
theorem small_frog_reaches_7th_rung : ∃ (a b : ℕ), 2 * a + 3 * b = 7 :=
by sorry

-- 2. Prove that the medium frog cannot reach the 1st rung
theorem medium_frog_cannot_reach_1st_rung : ¬(∃ (a b : ℕ), 2 * a + 4 * b = 1) :=
by sorry

-- 3. Prove that the large frog can reach the 3rd rung
theorem large_frog_reaches_3rd_rung : ∃ (a b : ℕ), 6 * a + 9 * b = 3 :=
by sorry

end NUMINAMATH_GPT_small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l815_81576


namespace NUMINAMATH_GPT_triangle_BFD_ratio_l815_81598

theorem triangle_BFD_ratio (x : ℝ) : 
  let AF := 3 * x
  let FE := x
  let ED := x
  let DC := 3 * x
  let side_square := AF + FE
  let area_square := side_square^2
  let area_triangle_BFD := area_square - (1/2 * AF * side_square + 1/2 * side_square * FE + 1/2 * ED * DC)
  (area_triangle_BFD / area_square) = 7 / 16 := 
by
  sorry

end NUMINAMATH_GPT_triangle_BFD_ratio_l815_81598


namespace NUMINAMATH_GPT_frobenius_two_vars_l815_81564

theorem frobenius_two_vars (a b n : ℤ) (ha : 0 < a) (hb : 0 < b) (hgcd : Int.gcd a b = 1) (hn : n > a * b - a - b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end NUMINAMATH_GPT_frobenius_two_vars_l815_81564


namespace NUMINAMATH_GPT_population_percentage_l815_81554

theorem population_percentage (total_population : ℕ) (percentage : ℕ) (result : ℕ) :
  total_population = 25600 → percentage = 90 → result = (percentage * total_population) / 100 → result = 23040 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_population_percentage_l815_81554


namespace NUMINAMATH_GPT_even_function_coeff_l815_81510

theorem even_function_coeff (a : ℝ) (h : ∀ x : ℝ, (a-2)*x^2 + (a-1)*x + 3 = (a-2)*(-x)^2 + (a-1)*(-x) + 3) : a = 1 :=
by {
  -- Proof here
  sorry
}

end NUMINAMATH_GPT_even_function_coeff_l815_81510


namespace NUMINAMATH_GPT_room_length_l815_81516

theorem room_length (width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) (length : ℝ)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 28875)
  (h_cost_per_sqm : cost_per_sqm = 1400)
  (h_length : length = total_cost / cost_per_sqm / width) :
  length = 5.5 := by
  sorry

end NUMINAMATH_GPT_room_length_l815_81516


namespace NUMINAMATH_GPT_gas_cost_correct_l815_81577

def cost_to_fill_remaining_quarter (initial_fill : ℚ) (final_fill : ℚ) (added_gas : ℚ) (cost_per_litre : ℚ) : ℚ :=
  let tank_capacity := (added_gas * (1 / (final_fill - initial_fill)))
  let remaining_quarter_cost := (tank_capacity * (1 / 4)) * cost_per_litre
  remaining_quarter_cost

theorem gas_cost_correct :
  cost_to_fill_remaining_quarter (1/8) (3/4) 30 1.38 = 16.56 :=
by
  sorry

end NUMINAMATH_GPT_gas_cost_correct_l815_81577


namespace NUMINAMATH_GPT_no_natural_number_exists_l815_81512

theorem no_natural_number_exists 
  (n : ℕ) : ¬ ∃ x y : ℕ, (2 * n * (n + 1) * (n + 2) * (n + 3) + 12) = x^2 + y^2 := 
by sorry

end NUMINAMATH_GPT_no_natural_number_exists_l815_81512


namespace NUMINAMATH_GPT_solve_for_x_l815_81527

theorem solve_for_x (x : ℝ) (h : ⌈x⌉ * x = 156) : x = 12 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l815_81527


namespace NUMINAMATH_GPT_area_of_sector_radius_2_angle_90_l815_81584

-- Given conditions
def radius := 2
def central_angle := 90

-- Required proof: the area of the sector with given conditions equals π.
theorem area_of_sector_radius_2_angle_90 : (90 * Real.pi * (2^2) / 360) = Real.pi := 
by
  sorry

end NUMINAMATH_GPT_area_of_sector_radius_2_angle_90_l815_81584


namespace NUMINAMATH_GPT_solve_inequalities_l815_81548

theorem solve_inequalities (x : ℝ) :
  (2 * x + 1 < 3) ∧ ((x / 2) + ((1 - 3 * x) / 4) ≤ 1) → -3 ≤ x ∧ x < 1 := 
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l815_81548


namespace NUMINAMATH_GPT_numeric_value_of_BAR_l815_81552

variable (b a t c r : ℕ)

-- Conditions from the problem
axiom h1 : b + a + t = 6
axiom h2 : c + a + t = 8
axiom h3 : c + a + r = 12

-- Required to prove
theorem numeric_value_of_BAR : b + a + r = 10 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_numeric_value_of_BAR_l815_81552


namespace NUMINAMATH_GPT_fractions_sum_to_decimal_l815_81586

theorem fractions_sum_to_decimal :
  (2 / 10) + (4 / 100) + (6 / 1000) = 0.246 :=
by 
  sorry

end NUMINAMATH_GPT_fractions_sum_to_decimal_l815_81586


namespace NUMINAMATH_GPT_total_emails_in_april_is_675_l815_81519

-- Define the conditions
def daily_emails : ℕ := 20
def additional_emails : ℕ := 5
def april_days : ℕ := 30
def half_april_days : ℕ := april_days / 2

-- Define the total number of emails received
def total_emails : ℕ :=
  (daily_emails * half_april_days) +
  ((daily_emails + additional_emails) * half_april_days)

-- Define the statement to be proven
theorem total_emails_in_april_is_675 : total_emails = 675 :=
  by
  sorry

end NUMINAMATH_GPT_total_emails_in_april_is_675_l815_81519


namespace NUMINAMATH_GPT_train_cross_bridge_in_56_seconds_l815_81553

noncomputable def train_pass_time (length_train length_bridge : ℝ) (speed_train_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000 / 3600)
  total_distance / speed_train_ms

theorem train_cross_bridge_in_56_seconds :
  train_pass_time 560 140 45 = 56 :=
by
  -- The proof can be added here
  sorry

end NUMINAMATH_GPT_train_cross_bridge_in_56_seconds_l815_81553


namespace NUMINAMATH_GPT_age_ratio_proof_l815_81574

variable (j a x : ℕ)

/-- Given conditions about Jack and Alex's ages. -/
axiom h1 : j - 3 = 2 * (a - 3)
axiom h2 : j - 5 = 3 * (a - 5)

def age_ratio_in_years : Prop :=
  (3 * (a + x) = 2 * (j + x)) → (x = 1)

theorem age_ratio_proof : age_ratio_in_years j a x := by
  sorry

end NUMINAMATH_GPT_age_ratio_proof_l815_81574


namespace NUMINAMATH_GPT_selling_price_correct_l815_81515

-- Define the conditions
def purchase_price : ℝ := 12000
def repair_costs : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percentage : ℝ := 0.50

-- Calculate total cost
def total_cost : ℝ := purchase_price + repair_costs + transportation_charges

-- Define the selling price and the proof goal
def selling_price : ℝ := total_cost + (profit_percentage * total_cost)

-- Prove that the selling price equals Rs 27000
theorem selling_price_correct : selling_price = 27000 := 
by 
  -- Proof is not required, so we use sorry
  sorry

end NUMINAMATH_GPT_selling_price_correct_l815_81515


namespace NUMINAMATH_GPT_value_range_a_for_two_positive_solutions_l815_81560

theorem value_range_a_for_two_positive_solutions (a : ℝ) :
  (∃ (x : ℝ), (|2 * x - 1| - a = 0) ∧ x > 0 ∧ (0 < a ∧ a < 1)) :=
by 
  sorry

end NUMINAMATH_GPT_value_range_a_for_two_positive_solutions_l815_81560


namespace NUMINAMATH_GPT_intersection_M_N_l815_81528

open Set

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2 * x + 1}

theorem intersection_M_N : M ∩ N = {-1, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l815_81528


namespace NUMINAMATH_GPT_age_problem_l815_81558

theorem age_problem (a b c d : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : b = 3 * d)
  (h4 : a + b + c + d = 87) : 
  b = 30 :=
by sorry

end NUMINAMATH_GPT_age_problem_l815_81558


namespace NUMINAMATH_GPT_intersection_M_N_l815_81551

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l815_81551


namespace NUMINAMATH_GPT_p_q_work_l815_81557

theorem p_q_work (p_rate q_rate : ℝ) (h1: 1 / p_rate + 1 / q_rate = 1 / 6) (h2: p_rate = 15) : q_rate = 10 :=
by
  sorry

end NUMINAMATH_GPT_p_q_work_l815_81557


namespace NUMINAMATH_GPT_merchant_profit_percentage_l815_81580

noncomputable def cost_price_of_one_article (C : ℝ) : Prop := ∃ S : ℝ, 20 * C = 16 * S

theorem merchant_profit_percentage (C S : ℝ) (h : cost_price_of_one_article C) : 
  100 * ((S - C) / C) = 25 :=
by 
  sorry

end NUMINAMATH_GPT_merchant_profit_percentage_l815_81580


namespace NUMINAMATH_GPT_line_product_l815_81536

theorem line_product (b m : ℝ) (h1: b = -1) (h2: m = 2) : m * b = -2 :=
by
  rw [h1, h2]
  norm_num


end NUMINAMATH_GPT_line_product_l815_81536


namespace NUMINAMATH_GPT_neg_univ_prop_l815_81546

-- Translate the original mathematical statement to a Lean 4 statement.
theorem neg_univ_prop :
  (¬(∀ x : ℝ, x^2 ≠ x)) ↔ (∃ x : ℝ, x^2 = x) :=
by
  sorry

end NUMINAMATH_GPT_neg_univ_prop_l815_81546


namespace NUMINAMATH_GPT_polygon_sides_l815_81521

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 :=
by sorry

end NUMINAMATH_GPT_polygon_sides_l815_81521


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l815_81550

theorem number_of_men_in_first_group 
    (x : ℕ) (H1 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate x 5 = 1 / (5 * x))
    (H2 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate 15 12 = 1 / (15 * 12))
    (H3 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate x 5 = work_rate 15 12) 
    : x = 36 := 
by {
    sorry
}

end NUMINAMATH_GPT_number_of_men_in_first_group_l815_81550


namespace NUMINAMATH_GPT_quadratic_real_roots_l815_81517

theorem quadratic_real_roots (K : ℝ) :
  ∃ x : ℝ, K^2 * x^2 + (K^2 - 1) * x - 2 * K^2 = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_real_roots_l815_81517


namespace NUMINAMATH_GPT_distinct_dragons_count_l815_81599

theorem distinct_dragons_count : 
  {n : ℕ // n = 7} :=
sorry

end NUMINAMATH_GPT_distinct_dragons_count_l815_81599


namespace NUMINAMATH_GPT_mary_flour_amount_l815_81501

noncomputable def cups_of_flour_already_put_in
    (total_flour_needed : ℕ)
    (total_sugar_needed : ℕ)
    (extra_flour_needed : ℕ)
    (flour_to_be_added : ℕ) : ℕ :=
total_flour_needed - (total_sugar_needed + extra_flour_needed)

theorem mary_flour_amount
    (total_flour_needed : ℕ := 9)
    (total_sugar_needed : ℕ := 6)
    (extra_flour_needed : ℕ := 1) :
    cups_of_flour_already_put_in total_flour_needed total_sugar_needed extra_flour_needed (total_sugar_needed + extra_flour_needed) = 2 := by
  sorry

end NUMINAMATH_GPT_mary_flour_amount_l815_81501


namespace NUMINAMATH_GPT_box_volume_correct_l815_81581

-- Define the dimensions of the original sheet
def length_original : ℝ := 48
def width_original : ℝ := 36

-- Define the side length of the squares cut from each corner
def side_length_cut : ℝ := 4

-- Define the new dimensions after cutting the squares
def new_length : ℝ := length_original - 2 * side_length_cut
def new_width : ℝ := width_original - 2 * side_length_cut

-- Define the height of the box
def height_box : ℝ := side_length_cut

-- Define the expected volume of the box
def volume_box_expected : ℝ := 4480

-- Prove that the calculated volume is equal to the expected volume
theorem box_volume_correct :
  new_length * new_width * height_box = volume_box_expected := by
  sorry

end NUMINAMATH_GPT_box_volume_correct_l815_81581


namespace NUMINAMATH_GPT_interior_angle_ratio_l815_81569

variables (α β γ : ℝ)

theorem interior_angle_ratio
  (h1 : 2 * α + 3 * β = 4 * γ)
  (h2 : α = 4 * β - γ) :
  ∃ k : ℝ, k ≠ 0 ∧ 
  (α = 2 * k ∧ β = 9 * k ∧ γ = 4 * k) :=
sorry

end NUMINAMATH_GPT_interior_angle_ratio_l815_81569


namespace NUMINAMATH_GPT_find_common_difference_l815_81549

-- Definitions for arithmetic sequences and sums
def S (a1 d : ℕ) (n : ℕ) := (n * (2 * a1 + (n - 1) * d)) / 2
def a (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

theorem find_common_difference (a1 d : ℕ) :
  S a1 d 3 = 6 → a a1 d 3 = 4 → d = 2 :=
by
  intros S3_eq a3_eq
  sorry

end NUMINAMATH_GPT_find_common_difference_l815_81549


namespace NUMINAMATH_GPT_f_monotonic_non_overlapping_domains_domain_of_sum_l815_81518

axiom f : ℝ → ℝ
axiom f_decreasing : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ → x₁ ≤ 1 → -1 ≤ x₂ → x₂ ≤ 1 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

theorem f_monotonic : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ → x₁ ≤ 1 → -1 ≤ x₂ → x₂ ≤ 1 → x₁ ≤ x₂ → f x₁ ≥ f x₂ := sorry

theorem non_overlapping_domains : ∀ c : ℝ, (c - 1 > c^2 + 1 → c > 2) ∧ (c^2 - 1 > c + 1 → c < -1) := sorry

theorem domain_of_sum : 
  ∀ c : ℝ,
  -1 ≤ c ∧ c ≤ 2 →
  (∃ a b : ℝ, 
    ((-1 ≤ c ∧ c ≤ 0) ∨ (1 ≤ c ∧ c ≤ 2) → a = c^2 - 1 ∧ b = c + 1) ∧ 
    (0 < c ∧ c < 1 → a = c - 1 ∧ b = c^2 + 1)
  ) := sorry

end NUMINAMATH_GPT_f_monotonic_non_overlapping_domains_domain_of_sum_l815_81518


namespace NUMINAMATH_GPT_sum_of_two_rationals_negative_l815_81547

theorem sum_of_two_rationals_negative (a b : ℚ) (h : a + b < 0) : a < 0 ∨ b < 0 := sorry

end NUMINAMATH_GPT_sum_of_two_rationals_negative_l815_81547


namespace NUMINAMATH_GPT_sum_of_coordinates_D_l815_81504

theorem sum_of_coordinates_D
    (C N D : ℝ × ℝ) 
    (hC : C = (10, 5))
    (hN : N = (4, 9))
    (h_midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
    C.1 + D.1 + (C.2 + D.2) = 22 :=
  by sorry

end NUMINAMATH_GPT_sum_of_coordinates_D_l815_81504


namespace NUMINAMATH_GPT_constant_term_eq_160_l815_81573

-- Define the binomial coefficients and the binomial theorem
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term of (2x + 1/x)^6 expansion
def general_term_expansion (r : ℕ) : ℤ :=
  2^(6 - r) * binom 6 r

-- Define the proof statement for the required constant term
theorem constant_term_eq_160 : general_term_expansion 3 = 160 := 
by
  sorry

end NUMINAMATH_GPT_constant_term_eq_160_l815_81573


namespace NUMINAMATH_GPT_ratio_of_sum_to_first_term_l815_81591

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - (2 ^ n)) / (1 - 2)

-- Main statement to be proven
theorem ratio_of_sum_to_first_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geo : geometric_sequence a 2) (h_sum : sum_of_first_n_terms a S) :
  S 3 / a 0 = 7 :=
sorry

end NUMINAMATH_GPT_ratio_of_sum_to_first_term_l815_81591


namespace NUMINAMATH_GPT_determine_alpha_l815_81540

variables (m n : ℝ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_mn : m + n = 1)
variables (α : ℝ)

-- Defining the minimum value condition
def minimum_value_condition : Prop :=
  (1 / m + 16 / n) = 25

-- Defining the curve passing through point P
def passes_through_P : Prop :=
  (m / 5) ^ α = (m / 4)

theorem determine_alpha
  (h_min_value : minimum_value_condition m n)
  (h_passes_through : passes_through_P m α) :
  α = 1 / 2 :=
sorry

end NUMINAMATH_GPT_determine_alpha_l815_81540


namespace NUMINAMATH_GPT_cost_to_fill_pool_l815_81572

/-- Definition of the pool dimensions and constants --/
def pool_length := 20
def pool_width := 6
def pool_depth := 10
def cubic_feet_to_liters := 25
def liter_cost := 3

/-- Calculating the cost to fill the pool --/
def pool_volume := pool_length * pool_width * pool_depth
def total_liters := pool_volume * cubic_feet_to_liters
def total_cost := total_liters * liter_cost

/-- Theorem stating that the total cost to fill the pool is $90,000 --/
theorem cost_to_fill_pool : total_cost = 90000 := by
  sorry

end NUMINAMATH_GPT_cost_to_fill_pool_l815_81572


namespace NUMINAMATH_GPT_expression_value_l815_81597

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end NUMINAMATH_GPT_expression_value_l815_81597


namespace NUMINAMATH_GPT_distance_between_Jay_and_Sarah_l815_81590

theorem distance_between_Jay_and_Sarah 
  (time_in_hours : ℝ)
  (jay_speed_per_12_minutes : ℝ)
  (sarah_speed_per_36_minutes : ℝ)
  (total_distance : ℝ) :
  time_in_hours = 2 →
  jay_speed_per_12_minutes = 1 →
  sarah_speed_per_36_minutes = 3 →
  total_distance = 20 :=
by
  intros time_in_hours_eq jay_speed_eq sarah_speed_eq
  sorry

end NUMINAMATH_GPT_distance_between_Jay_and_Sarah_l815_81590


namespace NUMINAMATH_GPT_car_distance_and_velocity_l815_81500

def acceleration : ℝ := 12 -- constant acceleration in m/s^2
def time : ℝ := 36 -- time in seconds
def conversion_factor : ℝ := 3.6 -- conversion factor from m/s to km/h

theorem car_distance_and_velocity :
  (1/2 * acceleration * time^2 = 7776) ∧ (acceleration * time * conversion_factor = 1555.2) :=
by
  sorry

end NUMINAMATH_GPT_car_distance_and_velocity_l815_81500


namespace NUMINAMATH_GPT_solve_for_m_l815_81505

theorem solve_for_m (x m : ℝ) (hx : 0 < x) (h_eq : m / (x^2 - 9) + 2 / (x + 3) = 1 / (x - 3)) : m = 6 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l815_81505


namespace NUMINAMATH_GPT_temperature_on_Friday_l815_81532

variable (M T W Th F : ℝ)

def avg_M_T_W_Th := (M + T + W + Th) / 4 = 48
def avg_T_W_Th_F := (T + W + Th + F) / 4 = 46
def temp_Monday := M = 42

theorem temperature_on_Friday
  (h1 : avg_M_T_W_Th M T W Th)
  (h2 : avg_T_W_Th_F T W Th F) 
  (h3 : temp_Monday M) : F = 34 := by
  sorry

end NUMINAMATH_GPT_temperature_on_Friday_l815_81532


namespace NUMINAMATH_GPT_integer_solutions_l815_81563

-- Define the equation to be solved
def equation (x y : ℤ) : Prop := x * y + 3 * x - 5 * y + 3 = 0

-- Define the solutions
def solution_set : List (ℤ × ℤ) := 
  [(-13,-2), (-4,-1), (-1,0), (2, 3), (3, 6), (4, 15), (6, -21),
   (7, -12), (8, -9), (11, -6), (14, -5), (23, -4)]

-- The theorem stating the solutions are correct
theorem integer_solutions : ∀ (x y : ℤ), (x, y) ∈ solution_set → equation x y :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l815_81563


namespace NUMINAMATH_GPT_subtract_abs_from_local_value_l815_81567

-- Define the local value of 4 in 564823 as 4000
def local_value_of_4_in_564823 : ℕ := 4000

-- Define the absolute value of 4 as 4
def absolute_value_of_4 : ℕ := 4

-- Theorem statement: Prove that subtracting the absolute value of 4 from the local value of 4 in 564823 equals 3996
theorem subtract_abs_from_local_value : (local_value_of_4_in_564823 - absolute_value_of_4) = 3996 :=
by
  sorry

end NUMINAMATH_GPT_subtract_abs_from_local_value_l815_81567


namespace NUMINAMATH_GPT_tv_cost_l815_81593

theorem tv_cost (savings original_savings furniture_spent : ℝ) (hs : original_savings = 1000) (hf : furniture_spent = (3/4) * original_savings) (remaining_spent : savings = original_savings - furniture_spent) : savings = 250 := 
by
  sorry

end NUMINAMATH_GPT_tv_cost_l815_81593


namespace NUMINAMATH_GPT_b_alone_days_l815_81537

-- Definitions from the conditions
def work_rate_b (W_b : ℝ) : ℝ := W_b
def work_rate_a (W_b : ℝ) : ℝ := 2 * W_b
def work_rate_c (W_b : ℝ) : ℝ := 6 * W_b
def combined_work_rate (W_b : ℝ) : ℝ := work_rate_a W_b + work_rate_b W_b + work_rate_c W_b
def total_days_together : ℝ := 10
def total_work (W_b : ℝ) : ℝ := combined_work_rate W_b * total_days_together

-- The proof problem
theorem b_alone_days (W_b : ℝ) : 90 = total_work W_b / work_rate_b W_b :=
by
  sorry

end NUMINAMATH_GPT_b_alone_days_l815_81537


namespace NUMINAMATH_GPT_min_b_minus_2c_over_a_l815_81523

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (h1 : a ≤ b + c ∧ b + c ≤ 3 * a)
variable (h2 : 3 * b^2 ≤ a * (a + c) ∧ a * (a + c) ≤ 5 * b^2)

theorem min_b_minus_2c_over_a : (∃ u : ℝ, (u = (b - 2 * c) / a) ∧ (∀ v : ℝ, (v = (b - 2 * c) / a) → u ≤ v)) :=
  sorry

end NUMINAMATH_GPT_min_b_minus_2c_over_a_l815_81523


namespace NUMINAMATH_GPT_total_cost_is_83_50_l815_81571

-- Definitions according to the conditions
def cost_adult_ticket : ℝ := 5.50
def cost_child_ticket : ℝ := 3.50
def total_tickets : ℝ := 21
def adult_tickets : ℝ := 5
def child_tickets : ℝ := total_tickets - adult_tickets

-- Total cost calculation based on the conditions
def cost_adult_total : ℝ := adult_tickets * cost_adult_ticket
def cost_child_total : ℝ := child_tickets * cost_child_ticket
def total_cost : ℝ := cost_adult_total + cost_child_total

-- The theorem to prove that the total cost is $83.50
theorem total_cost_is_83_50 : total_cost = 83.50 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_83_50_l815_81571


namespace NUMINAMATH_GPT_find_v_plus_z_l815_81543

variable (x u v w z : ℂ)
variable (y : ℂ)
variable (condition1 : y = 2)
variable (condition2 : w = -x - u)
variable (condition3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I)

theorem find_v_plus_z : v + z = -4 :=
by
  have h1 : y = 2 := condition1
  have h2 : w = -x - u := condition2
  have h3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I := condition3
  sorry

end NUMINAMATH_GPT_find_v_plus_z_l815_81543


namespace NUMINAMATH_GPT_trig_identity_l815_81506

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l815_81506


namespace NUMINAMATH_GPT_missing_number_evaluation_l815_81533

theorem missing_number_evaluation (x : ℝ) (h : |4 + 9 * x| - 6 = 70) : x = 8 :=
sorry

end NUMINAMATH_GPT_missing_number_evaluation_l815_81533


namespace NUMINAMATH_GPT_decagon_adjacent_probability_l815_81585

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end NUMINAMATH_GPT_decagon_adjacent_probability_l815_81585


namespace NUMINAMATH_GPT_exclusive_movies_count_l815_81503

-- Define the conditions
def shared_movies : Nat := 15
def andrew_movies : Nat := 25
def john_movies_exclusive : Nat := 8

-- Define the result calculation
def exclusive_movies (andrew_movies shared_movies john_movies_exclusive : Nat) : Nat :=
  (andrew_movies - shared_movies) + john_movies_exclusive

-- Statement to prove
theorem exclusive_movies_count : exclusive_movies andrew_movies shared_movies john_movies_exclusive = 18 := by
  sorry

end NUMINAMATH_GPT_exclusive_movies_count_l815_81503


namespace NUMINAMATH_GPT_Robert_has_taken_more_photos_l815_81555

variables (C L R : ℕ) -- Claire's, Lisa's, and Robert's photos

-- Conditions definitions:
def ClairePhotos : Prop := C = 8
def LisaPhotos : Prop := L = 3 * C
def RobertPhotos : Prop := R > C

-- The proof problem statement:
theorem Robert_has_taken_more_photos (h1 : ClairePhotos C) (h2 : LisaPhotos C L) : RobertPhotos C R :=
by { sorry }

end NUMINAMATH_GPT_Robert_has_taken_more_photos_l815_81555


namespace NUMINAMATH_GPT_sum_bn_l815_81583

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n-1))) / 2

def geometric_sequence (a : ℕ → ℕ) :=
  ∃ r : ℕ, a 1 = a 0 * r ∧ a 2 = a 1 * r

-- Given S_5 = 35
def S5_property (S : ℕ → ℕ) := S 5 = 35

-- a_1, a_4, a_{13} is a geometric sequence
def a1_a4_a13_geometric_sequence (a : ℕ → ℕ) :=
  ∃ r : ℕ, a 4 = a 1 * r ∧ a 13 = a 4 * r

-- Define the sequence b_n and conditions
def bn_prop (a b : ℕ → ℕ) := ∀ n : ℕ, b n = a n * (2^(n-1))

-- Main theorem
theorem sum_bn {a b : ℕ → ℕ} {S T : ℕ → ℕ} (h_a : arithmetic_sequence a 2) (h_S5 : S5_property S) (h_geo : a1_a4_a13_geometric_sequence a) (h_bn : bn_prop a b)
  : ∀ n : ℕ, T n = 1 + (2 * n - 1) * 2^n := sorry

end NUMINAMATH_GPT_sum_bn_l815_81583


namespace NUMINAMATH_GPT_average_marks_of_failed_boys_l815_81538

def total_boys : ℕ := 120
def average_marks_all_boys : ℝ := 35
def number_of_passed_boys : ℕ := 100
def average_marks_passed_boys : ℝ := 39
def number_of_failed_boys : ℕ := total_boys - number_of_passed_boys

noncomputable def total_marks_all_boys : ℝ := average_marks_all_boys * total_boys
noncomputable def total_marks_passed_boys : ℝ := average_marks_passed_boys * number_of_passed_boys
noncomputable def total_marks_failed_boys : ℝ := total_marks_all_boys - total_marks_passed_boys
noncomputable def average_marks_failed_boys : ℝ := total_marks_failed_boys / number_of_failed_boys

theorem average_marks_of_failed_boys :
  average_marks_failed_boys = 15 :=
by
  -- The proof can be filled in here
  sorry

end NUMINAMATH_GPT_average_marks_of_failed_boys_l815_81538


namespace NUMINAMATH_GPT_train_cross_pole_time_l815_81545

variable (L : Real) (V : Real)

theorem train_cross_pole_time (hL : L = 110) (hV : V = 144) : 
  (110 / (144 * 1000 / 3600) = 2.75) := 
by
  sorry

end NUMINAMATH_GPT_train_cross_pole_time_l815_81545


namespace NUMINAMATH_GPT_evaluate_expression_l815_81559

variable (a b : ℝ) (h : a > b ∧ b > 0)

theorem evaluate_expression (h : a > b ∧ b > 0) : 
  (a^2 * b^3) / (b^2 * a^3) = (a / b)^(2 - 3) :=
  sorry

end NUMINAMATH_GPT_evaluate_expression_l815_81559


namespace NUMINAMATH_GPT_factorize_perfect_square_l815_81562

variable (a b : ℤ)

theorem factorize_perfect_square :
  a^2 + 6 * a * b + 9 * b^2 = (a + 3 * b)^2 := 
sorry

end NUMINAMATH_GPT_factorize_perfect_square_l815_81562


namespace NUMINAMATH_GPT_melissa_games_played_l815_81596

theorem melissa_games_played (total_points : ℕ) (points_per_game : ℕ) (num_games : ℕ) 
  (h1 : total_points = 81) 
  (h2 : points_per_game = 27) 
  (h3 : num_games = total_points / points_per_game) : 
  num_games = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_melissa_games_played_l815_81596


namespace NUMINAMATH_GPT_avg_of_7_consecutive_integers_l815_81539

theorem avg_of_7_consecutive_integers (c d : ℝ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 5.5 := by
  sorry

end NUMINAMATH_GPT_avg_of_7_consecutive_integers_l815_81539


namespace NUMINAMATH_GPT_arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l815_81513

-- Problem (a)
theorem arithmetic_sequence_a (x1 x2 x3 x4 x5 : ℕ) (h : (x1 = 2 ∧ x2 = 5 ∧ x3 = 10 ∧ x4 = 13 ∧ x5 = 15)) : 
  ∃ a b c, (a = 5 ∧ b = 10 ∧ c = 15 ∧ b - a = c - b ∧ b - a > 0) := 
sorry

-- Problem (b)
theorem find_p_q (p q : ℕ) (h : ∃ d, (7 - p = d ∧ q - 7 = d ∧ 13 - q = d)) : 
  p = 4 ∧ q = 10 :=
sorry

-- Problem (c)
theorem find_c_minus_a (a b c : ℕ) (h : ∃ d, (b - a = d ∧ c - b = d ∧ (a + 21) - c = d)) :
  c - a = 14 :=
sorry

-- Problem (d)
theorem find_y_values (y : ℤ) (h : ∃ d, ((2*y + 3) - (y - 6) = d ∧ (y*y + 2) - (2*y + 3) = d) ) :
  y = 5 ∨ y = -2 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l815_81513


namespace NUMINAMATH_GPT_average_growth_rate_le_half_sum_l815_81575

variable (p q x : ℝ)

theorem average_growth_rate_le_half_sum : 
  (1 + p) * (1 + q) = (1 + x) ^ 2 → x ≤ (p + q) / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_average_growth_rate_le_half_sum_l815_81575


namespace NUMINAMATH_GPT_sum_of_first_seven_terms_l815_81524

variable (a : ℕ → ℝ) -- a sequence of real numbers (can be adapted to other types if needed)

-- Given conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a n = a 0 + n * d

def sum_of_three_terms (a : ℕ → ℝ) (sum : ℝ) : Prop :=
  a 2 + a 3 + a 4 = sum

-- Theorem to prove
theorem sum_of_first_seven_terms (a : ℕ → ℝ) (h1 : is_arithmetic_progression a) (h2 : sum_of_three_terms a 12) :
  (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) = 28 :=
sorry

end NUMINAMATH_GPT_sum_of_first_seven_terms_l815_81524


namespace NUMINAMATH_GPT_hens_count_l815_81534

theorem hens_count (H C : ℕ) (h1 : H + C = 46) (h2 : 2 * H + 4 * C = 140) : H = 22 :=
by
  sorry

end NUMINAMATH_GPT_hens_count_l815_81534


namespace NUMINAMATH_GPT_systematic_sampling_l815_81561

theorem systematic_sampling :
  let N := 60
  let n := 5
  let k := N / n
  let initial_sample := 5
  let samples := [initial_sample, initial_sample + k, initial_sample + 2 * k, initial_sample + 3 * k, initial_sample + 4 * k] 
  samples = [5, 17, 29, 41, 53] := sorry

end NUMINAMATH_GPT_systematic_sampling_l815_81561


namespace NUMINAMATH_GPT_pyramid_on_pentagonal_prism_l815_81535

-- Define the structure of a pentagonal prism
structure PentagonalPrism where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

-- Initial pentagonal prism properties
def initialPrism : PentagonalPrism := {
  faces := 7,
  vertices := 10,
  edges := 15
}

-- Assume we add a pyramid on top of one pentagonal face
def addPyramid (prism : PentagonalPrism) : PentagonalPrism := {
  faces := prism.faces - 1 + 5, -- 1 face covered, 5 new faces
  vertices := prism.vertices + 1, -- 1 new vertex
  edges := prism.edges + 5 -- 5 new edges
}

-- The resulting shape after adding the pyramid
def resultingShape : PentagonalPrism := addPyramid initialPrism

-- Calculating the sum of faces, vertices, and edges
def sumFacesVerticesEdges (shape : PentagonalPrism) : ℕ :=
  shape.faces + shape.vertices + shape.edges

-- Statement of the problem in Lean 4
theorem pyramid_on_pentagonal_prism : sumFacesVerticesEdges resultingShape = 42 := by
  sorry

end NUMINAMATH_GPT_pyramid_on_pentagonal_prism_l815_81535


namespace NUMINAMATH_GPT_intersection_M_N_l815_81522

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | ∃ y ∈ M, |y| = x}

-- The main theorem to prove M ∩ N = {0, 1, 2}
theorem intersection_M_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l815_81522


namespace NUMINAMATH_GPT_geom_seq_prop_l815_81565

-- Definitions from the conditions
def geom_seq (a : ℕ → ℝ) := ∀ (n : ℕ), (a (n + 1)) / (a n) = (a 1) / (a 0) ∧ a n > 0

def condition (a : ℕ → ℝ) :=
  (1 / (a 2 * a 4)) + (2 / (a 4 ^ 2)) + (1 / (a 4 * a 6)) = 81

-- The statement to prove
theorem geom_seq_prop (a : ℕ → ℝ) (hgeom : geom_seq a) (hcond : condition a) :
  (1 / (a 3) + 1 / (a 5)) = 9 :=
sorry

end NUMINAMATH_GPT_geom_seq_prop_l815_81565


namespace NUMINAMATH_GPT_inverse_89_mod_90_l815_81530

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  -- Mathematical proof is skipped
  sorry

end NUMINAMATH_GPT_inverse_89_mod_90_l815_81530


namespace NUMINAMATH_GPT_haley_candy_l815_81595

theorem haley_candy (X : ℕ) (h : X - 17 + 19 = 35) : X = 33 :=
by
  sorry

end NUMINAMATH_GPT_haley_candy_l815_81595


namespace NUMINAMATH_GPT_polynomial_roots_l815_81529

theorem polynomial_roots :
  Polynomial.roots (Polynomial.C 4 * Polynomial.X ^ 5 +
                    Polynomial.C 13 * Polynomial.X ^ 4 +
                    Polynomial.C (-30) * Polynomial.X ^ 3 +
                    Polynomial.C 8 * Polynomial.X ^ 2) =
  {0, 0, 1 / 2, -2 + 2 * Real.sqrt 2, -2 - 2 * Real.sqrt 2} :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l815_81529


namespace NUMINAMATH_GPT_ending_point_divisible_by_9_l815_81578

theorem ending_point_divisible_by_9 (n : ℕ) (ending_point : ℕ) 
  (h1 : n = 11110) 
  (h2 : ∃ k : ℕ, 10 + 9 * k = ending_point) : 
  ending_point = 99999 := 
  sorry

end NUMINAMATH_GPT_ending_point_divisible_by_9_l815_81578


namespace NUMINAMATH_GPT_find_second_number_l815_81570

theorem find_second_number (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l815_81570


namespace NUMINAMATH_GPT_doubling_n_constant_C_l815_81525

theorem doubling_n_constant_C (e n R r : ℝ) (h_pos_e : 0 < e) (h_pos_n : 0 < n) (h_pos_R : 0 < R) (h_pos_r : 0 < r)
  (C : ℝ) (hC : C = e^2 * n / (R + n * r^2)) :
  C = (2 * e^2 * n) / (R + 2 * n * r^2) := 
sorry

end NUMINAMATH_GPT_doubling_n_constant_C_l815_81525


namespace NUMINAMATH_GPT_area_enclosed_curve_l815_81511

-- The proof statement
theorem area_enclosed_curve (x y : ℝ) : (x^2 + y^2 = 2 * (|x| + |y|)) → 
  (area_of_enclosed_region = 2 * π + 8) :=
sorry

end NUMINAMATH_GPT_area_enclosed_curve_l815_81511


namespace NUMINAMATH_GPT_solve_r_l815_81508

-- Definitions related to the problem
def satisfies_equation (r : ℝ) : Prop := ⌊r⌋ + 2 * r = 16

-- Theorem statement
theorem solve_r : ∃ (r : ℝ), satisfies_equation r ∧ r = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_r_l815_81508


namespace NUMINAMATH_GPT_cos_neg_1500_eq_half_l815_81592

theorem cos_neg_1500_eq_half : Real.cos (-1500 * Real.pi / 180) = 1/2 := by
  sorry

end NUMINAMATH_GPT_cos_neg_1500_eq_half_l815_81592


namespace NUMINAMATH_GPT_percentage_error_divide_instead_of_multiply_l815_81502

theorem percentage_error_divide_instead_of_multiply (x : ℝ) : 
  let correct_result := 5 * x 
  let incorrect_result := x / 10 
  let error := correct_result - incorrect_result 
  let percentage_error := (error / correct_result) * 100 
  percentage_error = 98 :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_divide_instead_of_multiply_l815_81502


namespace NUMINAMATH_GPT_combined_selling_price_l815_81541

theorem combined_selling_price 
  (cost_price1 cost_price2 cost_price3 : ℚ)
  (profit_percentage1 profit_percentage2 profit_percentage3 : ℚ)
  (h1 : cost_price1 = 1200) (h2 : profit_percentage1 = 0.4)
  (h3 : cost_price2 = 800)  (h4 : profit_percentage2 = 0.3)
  (h5 : cost_price3 = 600)  (h6 : profit_percentage3 = 0.5) : 
  cost_price1 * (1 + profit_percentage1) +
  cost_price2 * (1 + profit_percentage2) +
  cost_price3 * (1 + profit_percentage3) = 3620 := by 
  sorry

end NUMINAMATH_GPT_combined_selling_price_l815_81541


namespace NUMINAMATH_GPT_triangle_two_acute_angles_l815_81520

theorem triangle_two_acute_angles (A B C : ℝ) (h_triangle : A + B + C = 180) (h_pos : A > 0 ∧ B > 0 ∧ C > 0)
  (h_acute_triangle: A < 90 ∨ B < 90 ∨ C < 90): A < 90 ∧ B < 90 ∨ A < 90 ∧ C < 90 ∨ B < 90 ∧ C < 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_two_acute_angles_l815_81520


namespace NUMINAMATH_GPT_solve_quadratic_solve_cubic_l815_81568

theorem solve_quadratic (x : ℝ) (h : 2 * x^2 - 32 = 0) : x = 4 ∨ x = -4 := 
by sorry

theorem solve_cubic (x : ℝ) (h : (x + 4)^3 + 64 = 0) : x = -8 := 
by sorry

end NUMINAMATH_GPT_solve_quadratic_solve_cubic_l815_81568


namespace NUMINAMATH_GPT_percent_of_z_equals_120_percent_of_y_l815_81582

variable {x y z : ℝ}
variable {p : ℝ}

theorem percent_of_z_equals_120_percent_of_y
  (h1 : (p / 100) * z = 1.2 * y)
  (h2 : y = 0.75 * x)
  (h3 : z = 2 * x) :
  p = 45 :=
by sorry

end NUMINAMATH_GPT_percent_of_z_equals_120_percent_of_y_l815_81582


namespace NUMINAMATH_GPT_spaghetti_manicotti_ratio_l815_81509

-- Define the number of students who were surveyed and their preferences
def total_students := 800
def students_prefer_spaghetti := 320
def students_prefer_manicotti := 160

-- The ratio of students who prefer spaghetti to those who prefer manicotti is 2
theorem spaghetti_manicotti_ratio :
  students_prefer_spaghetti / students_prefer_manicotti = 2 :=
by
  sorry

end NUMINAMATH_GPT_spaghetti_manicotti_ratio_l815_81509


namespace NUMINAMATH_GPT_find_higher_interest_rate_l815_81579

-- Definitions and conditions based on the problem
def total_investment : ℕ := 4725
def higher_rate_investment : ℕ := 1925
def lower_rate_investment : ℕ := total_investment - higher_rate_investment
def lower_rate : ℝ := 0.08
def higher_to_lower_interest_ratio : ℝ := 2

-- The main theorem to prove the higher interest rate
theorem find_higher_interest_rate (r : ℝ) (h1 : higher_rate_investment = 1925) (h2 : lower_rate_investment = 2800) :
  1925 * r = 2 * (2800 * 0.08) → r = 448 / 1925 :=
sorry

end NUMINAMATH_GPT_find_higher_interest_rate_l815_81579


namespace NUMINAMATH_GPT_nine_x_five_y_multiple_l815_81544

theorem nine_x_five_y_multiple (x y : ℤ) (h : 2 * x + 3 * y ≡ 0 [ZMOD 17]) : 
  9 * x + 5 * y ≡ 0 [ZMOD 17] := 
by
  sorry

end NUMINAMATH_GPT_nine_x_five_y_multiple_l815_81544


namespace NUMINAMATH_GPT_find_smallest_n_l815_81589

/-- 
Define the doubling sum function D(a, n)
-/
def doubling_sum (a : ℕ) (n : ℕ) : ℕ := a * (2^n - 1)

/--
Main theorem statement that proves the smallest n for the given conditions
-/
theorem find_smallest_n :
  ∃ (n : ℕ), (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 6 → ∃ (ai : ℕ), doubling_sum ai i = n) ∧ n = 9765 := 
sorry

end NUMINAMATH_GPT_find_smallest_n_l815_81589


namespace NUMINAMATH_GPT_statement_1_statement_2_statement_3_statement_4_main_proof_l815_81566

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem statement_1 : ¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x := sorry

theorem statement_2 : ∃! x, f x - x = 0 := sorry

theorem statement_3 : ¬ ∃ k > 0, ∀ x > 0, f x > k * x := sorry

theorem statement_4 : ∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4 := sorry

theorem main_proof : (¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x) ∧ 
                     (∃! x, f x - x = 0) ∧ 
                     (¬ ∃ k > 0, ∀ x > 0, f x > k * x) ∧ 
                     (∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4) := 
by
  apply And.intro
  · exact statement_1
  · apply And.intro
    · exact statement_2
    · apply And.intro
      · exact statement_3
      · exact statement_4

end NUMINAMATH_GPT_statement_1_statement_2_statement_3_statement_4_main_proof_l815_81566


namespace NUMINAMATH_GPT_molecular_weight_N2O_correct_l815_81507

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of N2O
def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O

-- Prove the statement
theorem molecular_weight_N2O_correct : molecular_weight_N2O = 44.02 := by
  -- We leave the proof as an exercise (or assumption)
  sorry

end NUMINAMATH_GPT_molecular_weight_N2O_correct_l815_81507


namespace NUMINAMATH_GPT_repair_time_and_earnings_l815_81514

-- Definitions based on given conditions
def cars : ℕ := 10
def cars_repair_50min : ℕ := 6
def repair_time_50min : ℕ := 50 -- minutes per car
def longer_percentage : ℕ := 80 -- 80% longer for the remaining cars
def wage_per_hour : ℕ := 30 -- dollars per hour

-- Remaining cars to repair
def remaining_cars : ℕ := cars - cars_repair_50min

-- Calculate total repair time for each type of cars and total repair time
def repair_time_remaining_cars : ℕ := repair_time_50min + (repair_time_50min * longer_percentage) / 100
def total_repair_time : ℕ := (cars_repair_50min * repair_time_50min) + (remaining_cars * repair_time_remaining_cars)

-- Convert total repair time from minutes to hours
def total_repair_hours : ℕ := total_repair_time / 60

-- Calculate total earnings
def total_earnings : ℕ := wage_per_hour * total_repair_hours

-- The theorem to be proved: total_repair_time == 660 and total_earnings == 330
theorem repair_time_and_earnings :
  total_repair_time = 660 ∧ total_earnings = 330 := by
  sorry

end NUMINAMATH_GPT_repair_time_and_earnings_l815_81514


namespace NUMINAMATH_GPT_travel_ways_l815_81542

theorem travel_ways (buses : Nat) (trains : Nat) (boats : Nat) 
  (hb : buses = 5) (ht : trains = 6) (hb2 : boats = 2) : 
  buses + trains + boats = 13 := by
  sorry

end NUMINAMATH_GPT_travel_ways_l815_81542
