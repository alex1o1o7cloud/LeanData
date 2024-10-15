import Mathlib

namespace NUMINAMATH_GPT_count_ordered_triples_l1582_158221

theorem count_ordered_triples (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : 2 * a * b * c = 2 * (a * b + b * c + a * c)) : 
  ∃ n, n = 10 :=
by
  sorry

end NUMINAMATH_GPT_count_ordered_triples_l1582_158221


namespace NUMINAMATH_GPT_neg_int_solution_l1582_158282

theorem neg_int_solution (x : ℤ) : -2 * x < 4 ↔ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_neg_int_solution_l1582_158282


namespace NUMINAMATH_GPT_rational_solution_for_k_is_6_l1582_158211

theorem rational_solution_for_k_is_6 (k : ℕ) (h : 0 < k) :
  (∃ x : ℚ, k * x ^ 2 + 12 * x + k = 0) ↔ k = 6 :=
by { sorry }

end NUMINAMATH_GPT_rational_solution_for_k_is_6_l1582_158211


namespace NUMINAMATH_GPT_John_sells_each_wig_for_five_dollars_l1582_158285

theorem John_sells_each_wig_for_five_dollars
  (plays : ℕ)
  (acts_per_play : ℕ)
  (wigs_per_act : ℕ)
  (wig_cost : ℕ)
  (total_cost : ℕ)
  (sold_wigs_cost : ℕ)
  (remaining_wigs_cost : ℕ) :
  plays = 3 ∧
  acts_per_play = 5 ∧
  wigs_per_act = 2 ∧
  wig_cost = 5 ∧
  total_cost = 150 ∧
  remaining_wigs_cost = 110 ∧
  total_cost - remaining_wigs_cost = sold_wigs_cost →
  (sold_wigs_cost / (plays * acts_per_play * wigs_per_act - remaining_wigs_cost / wig_cost)) = wig_cost :=
by sorry

end NUMINAMATH_GPT_John_sells_each_wig_for_five_dollars_l1582_158285


namespace NUMINAMATH_GPT_relationship_among_abc_l1582_158238

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem relationship_among_abc :
  b > c ∧ c > a :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l1582_158238


namespace NUMINAMATH_GPT_original_price_l1582_158277

theorem original_price (x : ℝ) (h : 0.9504 * x = 108) : x = 10800 / 9504 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1582_158277


namespace NUMINAMATH_GPT_parallelepiped_surface_area_l1582_158219

theorem parallelepiped_surface_area (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 12) 
  (h2 : a * b * c = 8) : 
  6 * (a^2) = 24 :=
by
  sorry

end NUMINAMATH_GPT_parallelepiped_surface_area_l1582_158219


namespace NUMINAMATH_GPT_probability_composite_product_l1582_158229

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end NUMINAMATH_GPT_probability_composite_product_l1582_158229


namespace NUMINAMATH_GPT_tan_a2_a12_l1582_158244

noncomputable def arithmetic_term (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

theorem tan_a2_a12 (a d : ℝ) (h : a + (a + 6 * d) + (a + 12 * d) = 4 * Real.pi) :
  Real.tan (arithmetic_term a d 2 + arithmetic_term a d 12) = - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_a2_a12_l1582_158244


namespace NUMINAMATH_GPT_number_of_episodes_l1582_158234

def episode_length : ℕ := 20
def hours_per_day : ℕ := 2
def days : ℕ := 15

theorem number_of_episodes : (days * hours_per_day * 60) / episode_length = 90 :=
by
  sorry

end NUMINAMATH_GPT_number_of_episodes_l1582_158234


namespace NUMINAMATH_GPT_least_positive_number_of_24x_plus_16y_is_8_l1582_158267

theorem least_positive_number_of_24x_plus_16y_is_8 :
  ∃ (x y : ℤ), 24 * x + 16 * y = 8 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_number_of_24x_plus_16y_is_8_l1582_158267


namespace NUMINAMATH_GPT_interest_rate_l1582_158295

variable (P : ℝ) (T : ℝ) (SI : ℝ)

theorem interest_rate (h_P : P = 535.7142857142857) (h_T : T = 4) (h_SI : SI = 75) :
    (SI / (P * T)) * 100 = 3.5 := by
  sorry

end NUMINAMATH_GPT_interest_rate_l1582_158295


namespace NUMINAMATH_GPT_range_of_m_l1582_158242

theorem range_of_m {f : ℝ → ℝ} (h : ∀ x, f x = x^2 - 6*x - 16)
  {a b : ℝ} (h_domain : ∀ x, 0 ≤ x ∧ x ≤ a → ∃ y, f y ≤ b) 
  (h_range : ∀ y, -25 ≤ y ∧ y ≤ -16 → ∃ x, f x = y) : 3 ≤ a ∧ a ≤ 6 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1582_158242


namespace NUMINAMATH_GPT_age_difference_l1582_158214

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 16) : C + 16 = A := 
by
  sorry

end NUMINAMATH_GPT_age_difference_l1582_158214


namespace NUMINAMATH_GPT_intersection_points_l1582_158232

noncomputable def h (x : ℝ) : ℝ := -x^2 - 4 * x + 1
noncomputable def j (x : ℝ) : ℝ := -h x
noncomputable def k (x : ℝ) : ℝ := h (-x)

def c : ℕ := 2 -- Number of intersections of y = h(x) and y = j(x)
def d : ℕ := 1 -- Number of intersections of y = h(x) and y = k(x)

theorem intersection_points :
  10 * c + d = 21 := by
  sorry

end NUMINAMATH_GPT_intersection_points_l1582_158232


namespace NUMINAMATH_GPT_angle_2016_216_in_same_quadrant_l1582_158260

noncomputable def angle_in_same_quadrant (a b : ℝ) : Prop :=
  let normalized (x : ℝ) := x % 360
  normalized a = normalized b

theorem angle_2016_216_in_same_quadrant : angle_in_same_quadrant 2016 216 := by
  sorry

end NUMINAMATH_GPT_angle_2016_216_in_same_quadrant_l1582_158260


namespace NUMINAMATH_GPT_number_of_small_slices_l1582_158259

-- Define the given conditions
variables (S L : ℕ)
axiom total_slices : S + L = 5000
axiom total_revenue : 150 * S + 250 * L = 1050000

-- State the problem we need to prove
theorem number_of_small_slices : S = 1500 :=
by sorry

end NUMINAMATH_GPT_number_of_small_slices_l1582_158259


namespace NUMINAMATH_GPT_greatest_mean_YZ_l1582_158224

noncomputable def X_mean := 60
noncomputable def Y_mean := 70
noncomputable def XY_mean := 64
noncomputable def XZ_mean := 66

theorem greatest_mean_YZ (Xn Yn Zn : ℕ) (m : ℕ) :
  (60 * Xn + 70 * Yn) / (Xn + Yn) = 64 →
  (60 * Xn + m) / (Xn + Zn) = 66 →
  ∃ (k : ℕ), k = 69 :=
by
  intro h1 h2
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_greatest_mean_YZ_l1582_158224


namespace NUMINAMATH_GPT_find_c_value_l1582_158206

theorem find_c_value (x c : ℝ) (h₁ : 3 * x + 8 = 5) (h₂ : c * x + 15 = 3) : c = 12 :=
by
  -- This is where the proof steps would go, but we will use sorry for now.
  sorry

end NUMINAMATH_GPT_find_c_value_l1582_158206


namespace NUMINAMATH_GPT_percentage_increase_l1582_158261

theorem percentage_increase (P : ℕ) (x y : ℕ) (h1 : x = 5) (h2 : y = 7) 
    (h3 : (x * (1 + P / 100) / (y * (1 - 10 / 100))) = 20 / 21) : 
    P = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1582_158261


namespace NUMINAMATH_GPT_force_with_18_inch_crowbar_l1582_158245

noncomputable def inverseForce (L F : ℝ) : ℝ :=
  F * L

theorem force_with_18_inch_crowbar :
  ∀ (F : ℝ), (inverseForce 12 200 = inverseForce 18 F) → F = 133.333333 :=
by
  intros
  sorry

end NUMINAMATH_GPT_force_with_18_inch_crowbar_l1582_158245


namespace NUMINAMATH_GPT_area_of_region_S_is_correct_l1582_158201

noncomputable def area_of_inverted_region (d : ℝ) : ℝ :=
  if h : d = 1.5 then 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi else 0

theorem area_of_region_S_is_correct :
  area_of_inverted_region 1.5 = 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_area_of_region_S_is_correct_l1582_158201


namespace NUMINAMATH_GPT_days_considered_l1582_158288

theorem days_considered (visitors_current : ℕ) (visitors_previous : ℕ) (total_visitors : ℕ)
  (h1 : visitors_current = 132) (h2 : visitors_previous = 274) (h3 : total_visitors = 406)
  (h_total : visitors_current + visitors_previous = total_visitors) :
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_days_considered_l1582_158288


namespace NUMINAMATH_GPT_johns_average_speed_l1582_158255

-- Conditions
def biking_time_minutes : ℝ := 45
def biking_speed_mph : ℝ := 20
def walking_time_minutes : ℝ := 120
def walking_speed_mph : ℝ := 3

-- Proof statement
theorem johns_average_speed :
  let biking_time_hours := biking_time_minutes / 60
  let biking_distance := biking_speed_mph * biking_time_hours
  let walking_time_hours := walking_time_minutes / 60
  let walking_distance := walking_speed_mph * walking_time_hours
  let total_distance := biking_distance + walking_distance
  let total_time := biking_time_hours + walking_time_hours
  let average_speed := total_distance / total_time
  average_speed = 7.64 :=
by
  sorry

end NUMINAMATH_GPT_johns_average_speed_l1582_158255


namespace NUMINAMATH_GPT_determine_alpha_l1582_158294

theorem determine_alpha (α : ℝ) (y : ℝ → ℝ) (h : ∀ x, y x = x^α) (hp : y 2 = Real.sqrt 2) : α = 1 / 2 :=
sorry

end NUMINAMATH_GPT_determine_alpha_l1582_158294


namespace NUMINAMATH_GPT_quadratic_function_behavior_l1582_158202

theorem quadratic_function_behavior (x : ℝ) (h : x > 2) :
  ∃ y : ℝ, y = - (x - 2)^2 - 7 ∧ ∀ x₁ x₂, x₁ > 2 → x₂ > x₁ → (-(x₂ - 2)^2 - 7) < (-(x₁ - 2)^2 - 7) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_behavior_l1582_158202


namespace NUMINAMATH_GPT_a_perfect_square_l1582_158251

theorem a_perfect_square (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_div : 2 * a * b ∣ a^2 + b^2 - a) : ∃ k : ℕ, a = k^2 := 
sorry

end NUMINAMATH_GPT_a_perfect_square_l1582_158251


namespace NUMINAMATH_GPT_find_d_l1582_158278

variable (d x : ℕ)
axiom balls_decomposition : d = x + (x + 1) + (x + 2)
axiom probability_condition : (x : ℚ) / (d : ℚ) < 1 / 6

theorem find_d : d = 3 := sorry

end NUMINAMATH_GPT_find_d_l1582_158278


namespace NUMINAMATH_GPT_assoc_mul_l1582_158284

-- Conditions from the problem
variables (x y z : Type) [Mul x] [Mul y] [Mul z]

theorem assoc_mul (a b c : x) : (a * b) * c = a * (b * c) := by sorry

end NUMINAMATH_GPT_assoc_mul_l1582_158284


namespace NUMINAMATH_GPT_comparison_of_products_l1582_158249

def A : ℕ := 8888888888888888888 -- 19 digits, all 8's
def B : ℕ := 3333333333333333333333333333333333333333333333333333333333333333 -- 68 digits, all 3's
def C : ℕ := 4444444444444444444 -- 19 digits, all 4's
def D : ℕ := 6666666666666666666666666666666666666666666666666666666666666667 -- 68 digits, first 67 are 6's, last is 7

theorem comparison_of_products : C * D > A * B ∧ C * D - A * B = 4444444444444444444 := sorry

end NUMINAMATH_GPT_comparison_of_products_l1582_158249


namespace NUMINAMATH_GPT_evaluate_expression_l1582_158264

theorem evaluate_expression (a b : ℝ) (h : (1/2 * a * (1:ℝ)^3 - 3 * b * 1 + 4 = 9)) :
  (1/2 * a * (-1:ℝ)^3 - 3 * b * (-1) + 4 = -1) := by
sorry

end NUMINAMATH_GPT_evaluate_expression_l1582_158264


namespace NUMINAMATH_GPT_greatest_two_digit_multiple_of_7_l1582_158281

theorem greatest_two_digit_multiple_of_7 : ∃ n, 10 ≤ n ∧ n < 100 ∧ n % 7 = 0 ∧ ∀ m, 10 ≤ m ∧ m < 100 ∧ m % 7 = 0 → n ≥ m := 
by
  sorry

end NUMINAMATH_GPT_greatest_two_digit_multiple_of_7_l1582_158281


namespace NUMINAMATH_GPT_value_two_stddevs_less_l1582_158289

theorem value_two_stddevs_less (μ σ : ℝ) (hμ : μ = 16.5) (hσ : σ = 1.5) : μ - 2 * σ = 13.5 :=
by
  rw [hμ, hσ]
  norm_num

end NUMINAMATH_GPT_value_two_stddevs_less_l1582_158289


namespace NUMINAMATH_GPT_max_students_l1582_158231

def num_pens : Nat := 1204
def num_pencils : Nat := 840

theorem max_students (n_pens n_pencils : Nat) (h_pens : n_pens = num_pens) (h_pencils : n_pencils = num_pencils) :
  Nat.gcd n_pens n_pencils = 16 := by
  sorry

end NUMINAMATH_GPT_max_students_l1582_158231


namespace NUMINAMATH_GPT_meeting_time_l1582_158204

-- Variables representing the conditions
def uniform_rate_cassie := 15
def uniform_rate_brian := 18
def distance_route := 70
def cassie_start_time := 8.0
def brian_start_time := 9.25

-- The goal
theorem meeting_time : ∃ T : ℝ, (15 * T + 18 * (T - 1.25) = 70) ∧ T = 2.803 := 
by {
  sorry
}

end NUMINAMATH_GPT_meeting_time_l1582_158204


namespace NUMINAMATH_GPT_smallest_angle_of_trapezoid_l1582_158290

theorem smallest_angle_of_trapezoid 
  (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : ∀ i j k l : ℝ, i + j = k + l → i + j = 180 ∧ k + l = 180) :
  a = 40 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_of_trapezoid_l1582_158290


namespace NUMINAMATH_GPT_correct_option_b_l1582_158240

theorem correct_option_b (a : ℝ) : 
  (-2 * a) ^ 3 = -8 * a ^ 3 :=
by sorry

end NUMINAMATH_GPT_correct_option_b_l1582_158240


namespace NUMINAMATH_GPT_slope_points_eq_l1582_158287

theorem slope_points_eq (m : ℚ) (h : ((m + 2) / (3 - m) = 2)) : m = 4 / 3 :=
sorry

end NUMINAMATH_GPT_slope_points_eq_l1582_158287


namespace NUMINAMATH_GPT_janet_family_needs_91_tickets_l1582_158272

def janet_family_tickets (adults: ℕ) (children: ℕ) (roller_coaster_adult_tickets: ℕ) (roller_coaster_child_tickets: ℕ) 
  (giant_slide_adult_tickets: ℕ) (giant_slide_child_tickets: ℕ) (num_roller_coaster_rides_adult: ℕ) 
  (num_roller_coaster_rides_child: ℕ) (num_giant_slide_rides_adult: ℕ) (num_giant_slide_rides_child: ℕ) : ℕ := 
  (adults * roller_coaster_adult_tickets * num_roller_coaster_rides_adult) + 
  (children * roller_coaster_child_tickets * num_roller_coaster_rides_child) + 
  (1 * giant_slide_adult_tickets * num_giant_slide_rides_adult) + 
  (1 * giant_slide_child_tickets * num_giant_slide_rides_child)

theorem janet_family_needs_91_tickets :
  janet_family_tickets 2 2 7 5 4 3 3 2 5 3 = 91 := 
by 
  -- Calculations based on the given conditions (skipped in this statement)
  sorry

end NUMINAMATH_GPT_janet_family_needs_91_tickets_l1582_158272


namespace NUMINAMATH_GPT_triangle_is_isosceles_l1582_158271

theorem triangle_is_isosceles
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : a = 2 * c * Real.cos B)
    (h2 : b = c * Real.cos A) 
    (h3 : c = a * Real.cos C) 
    : a = b := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l1582_158271


namespace NUMINAMATH_GPT_batsman_average_after_15th_innings_l1582_158218

theorem batsman_average_after_15th_innings 
  (A : ℕ) 
  (h1 : 14 * A + 85 = 15 * (A + 3)) 
  (h2 : A = 40) : 
  (A + 3) = 43 := by 
  sorry

end NUMINAMATH_GPT_batsman_average_after_15th_innings_l1582_158218


namespace NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l1582_158223

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 10 ∧ θ = 3 * Real.pi / 4 ∧ φ = Real.pi / 6 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  :=
by
  intros ρ θ φ h
  rcases h with ⟨hρ, hθ, hφ⟩
  simp [hρ, hθ, hφ]
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l1582_158223


namespace NUMINAMATH_GPT_total_sum_l1582_158243

theorem total_sum (p q r s t : ℝ) (P : ℝ) 
  (h1 : q = 0.75 * P) 
  (h2 : r = 0.50 * P) 
  (h3 : s = 0.25 * P) 
  (h4 : t = 0.10 * P) 
  (h5 : s = 25) 
  :
  p + q + r + s + t = 260 :=
by 
  sorry

end NUMINAMATH_GPT_total_sum_l1582_158243


namespace NUMINAMATH_GPT_ratio_of_four_numbers_exists_l1582_158217

theorem ratio_of_four_numbers_exists (A B C D : ℕ) (h1 : A + B + C + D = 1344) (h2 : D = 672) : 
  ∃ rA rB rC rD, rA ≠ 0 ∧ rB ≠ 0 ∧ rC ≠ 0 ∧ rD ≠ 0 ∧ A = rA * k ∧ B = rB * k ∧ C = rC * k ∧ D = rD * k :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_four_numbers_exists_l1582_158217


namespace NUMINAMATH_GPT_sum_of_coefficients_l1582_158263

theorem sum_of_coefficients (f : ℕ → ℕ) :
  (5 * 1 + 2)^7 = 823543 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1582_158263


namespace NUMINAMATH_GPT_least_possible_z_minus_x_l1582_158253

theorem least_possible_z_minus_x (x y z : ℕ) 
  (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z)
  (hxy : x < y) (hyz : y < z) (hyx_gt_3: y - x > 3)
  (hx_even : x % 2 = 0) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1) :
  z - x = 9 :=
sorry

end NUMINAMATH_GPT_least_possible_z_minus_x_l1582_158253


namespace NUMINAMATH_GPT_value_of_b_l1582_158220

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
sorry

end NUMINAMATH_GPT_value_of_b_l1582_158220


namespace NUMINAMATH_GPT_sqrt_sum_eq_fraction_l1582_158257

-- Definitions as per conditions
def w : ℕ := 4
def x : ℕ := 9
def z : ℕ := 25

-- Main theorem statement
theorem sqrt_sum_eq_fraction : (Real.sqrt (w / x) + Real.sqrt (x / z) = 19 / 15) := by
  sorry

end NUMINAMATH_GPT_sqrt_sum_eq_fraction_l1582_158257


namespace NUMINAMATH_GPT_joe_paint_initial_amount_l1582_158216

theorem joe_paint_initial_amount (P : ℕ) (h1 : P / 6 + (5 * P / 6) / 5 = 120) :
  P = 360 := by
  sorry

end NUMINAMATH_GPT_joe_paint_initial_amount_l1582_158216


namespace NUMINAMATH_GPT_range_of_a_and_t_minimum_of_y_l1582_158226

noncomputable def minimum_value_y (a b : ℝ) (h : a + b = 1) : ℝ :=
(a + 1/a) * (b + 1/b)

theorem range_of_a_and_t (a b : ℝ) (h : a + b = 1) :
  0 < a ∧ a < 1 ∧ 0 < a * b ∧ a * b <= 1/4 :=
sorry

theorem minimum_of_y (a b : ℝ) (h : a + b = 1) :
  minimum_value_y a b h = 25/4 :=
sorry

end NUMINAMATH_GPT_range_of_a_and_t_minimum_of_y_l1582_158226


namespace NUMINAMATH_GPT_culture_medium_preparation_l1582_158237

theorem culture_medium_preparation :
  ∀ (V : ℝ), 0 < V → 
  ∃ (nutrient_broth pure_water saline_water : ℝ),
    nutrient_broth = V / 3 ∧
    pure_water = V * 0.3 ∧
    saline_water = V - (nutrient_broth + pure_water) :=
by
  sorry

end NUMINAMATH_GPT_culture_medium_preparation_l1582_158237


namespace NUMINAMATH_GPT_geometric_sequence_increasing_l1582_158258

theorem geometric_sequence_increasing {a : ℕ → ℝ} (r : ℝ) (h_pos : 0 < r) (h_geometric : ∀ n, a (n + 1) = r * a n) :
  (a 0 < a 1 ∧ a 1 < a 2) ↔ ∀ n m, n < m → a n < a m :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_increasing_l1582_158258


namespace NUMINAMATH_GPT_find_m_l1582_158228

theorem find_m (m : ℝ) (h : ∀ x : ℝ, m - |x| ≥ 0 ↔ -1 ≤ x ∧ x ≤ 1) : m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l1582_158228


namespace NUMINAMATH_GPT_equivalent_expression_l1582_158225

theorem equivalent_expression (x : ℝ) : 
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) + 1 = x^4 := 
by
  sorry

end NUMINAMATH_GPT_equivalent_expression_l1582_158225


namespace NUMINAMATH_GPT_optionA_optionB_optionC_optionD_l1582_158298

-- Statement for option A
theorem optionA : (∀ x : ℝ, x ≠ 3 → x^2 - 4 * x + 3 ≠ 0) ↔ (x^2 - 4 * x + 3 = 0 → x = 3) := sorry

-- Statement for option B
theorem optionB : (¬ (∀ x : ℝ, x^2 - x + 2 > 0) ↔ ∃ x0 : ℝ, x0^2 - x0 + 2 ≤ 0) := sorry

-- Statement for option C
theorem optionC (p q : Prop) : p ∧ q → p ∧ q := sorry

-- Statement for option D
theorem optionD (x : ℝ) : (x > -1 → x^2 + 4 * x + 3 > 0) ∧ ¬ (∀ x : ℝ, x^2 + 4 * x + 3 > 0 → x > -1) := sorry

end NUMINAMATH_GPT_optionA_optionB_optionC_optionD_l1582_158298


namespace NUMINAMATH_GPT_find_first_term_l1582_158256

variable {a r : ℚ}

theorem find_first_term (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 120) : a = 240 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_first_term_l1582_158256


namespace NUMINAMATH_GPT_not_dividable_by_wobbly_l1582_158275

-- Define a wobbly number
def is_wobbly_number (n : ℕ) : Prop :=
  n > 0 ∧ (∀ k : ℕ, k < (Nat.log 10 n) → 
    (n / (10^k) % 10 ≠ 0 → n / (10^(k+1)) % 10 = 0) ∧
    (n / (10^k) % 10 = 0 → n / (10^(k+1)) % 10 ≠ 0))

-- Define sets of multiples of 10 and 25
def multiples_of (m : ℕ) (k : ℕ): Prop :=
  ∃ q : ℕ, k = q * m

def is_multiple_of_10 (k : ℕ) : Prop := multiples_of 10 k
def is_multiple_of_25 (k : ℕ) : Prop := multiples_of 25 k

theorem not_dividable_by_wobbly (n : ℕ) : 
  ¬ ∃ w : ℕ, is_wobbly_number w ∧ n ∣ w ↔ is_multiple_of_10 n ∨ is_multiple_of_25 n :=
by
  sorry

end NUMINAMATH_GPT_not_dividable_by_wobbly_l1582_158275


namespace NUMINAMATH_GPT_tv_episode_length_l1582_158283

theorem tv_episode_length :
  ∀ (E : ℕ), 
    600 = 3 * E + 270 + 2 * 105 + 45 → 
    E = 25 :=
by
  intros E h
  sorry

end NUMINAMATH_GPT_tv_episode_length_l1582_158283


namespace NUMINAMATH_GPT_exp_13_pi_i_over_2_eq_i_l1582_158279

theorem exp_13_pi_i_over_2_eq_i : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_GPT_exp_13_pi_i_over_2_eq_i_l1582_158279


namespace NUMINAMATH_GPT_probability_of_drawing_white_ball_l1582_158270

theorem probability_of_drawing_white_ball (P_A P_B P_C : ℝ) 
    (hA : P_A = 0.4) 
    (hB : P_B = 0.25)
    (hSum : P_A + P_B + P_C = 1) : 
    P_C = 0.35 :=
by
    -- Placeholder for the proof
    sorry

end NUMINAMATH_GPT_probability_of_drawing_white_ball_l1582_158270


namespace NUMINAMATH_GPT_markup_percentage_l1582_158235

theorem markup_percentage (S M : ℝ) (h1 : S = 56 + M * S) (h2 : 0.80 * S - 56 = 8) : M = 0.30 :=
sorry

end NUMINAMATH_GPT_markup_percentage_l1582_158235


namespace NUMINAMATH_GPT_sum_of_smallest_multiples_l1582_158209

def smallest_two_digit_multiple_of_5 := 10
def smallest_three_digit_multiple_of_7 := 105

theorem sum_of_smallest_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end NUMINAMATH_GPT_sum_of_smallest_multiples_l1582_158209


namespace NUMINAMATH_GPT_original_marketing_pct_correct_l1582_158222

-- Define the initial and final percentages of finance specialization students
def initial_finance_pct := 0.88
def final_finance_pct := 0.90

-- Define the final percentage of marketing specialization students
def final_marketing_pct := 0.43333333333333335

-- Define the original percentage of marketing specialization students
def original_marketing_pct := 0.45333333333333335

-- The Lean statement to prove the original percentage of marketing students
theorem original_marketing_pct_correct :
  initial_finance_pct + (final_marketing_pct - initial_finance_pct) = original_marketing_pct := 
sorry

end NUMINAMATH_GPT_original_marketing_pct_correct_l1582_158222


namespace NUMINAMATH_GPT_total_bricks_fill_box_l1582_158205

-- Define brick and box volumes based on conditions
def volume_brick1 := 2 * 5 * 8
def volume_brick2 := 2 * 3 * 7
def volume_box := 10 * 11 * 14

-- Define the main proof problem
theorem total_bricks_fill_box (x y : ℕ) (h1 : volume_brick1 * x + volume_brick2 * y = volume_box) :
  x + y = 24 :=
by
  -- Left as an exercise (proof steps are not included per instructions)
  sorry

end NUMINAMATH_GPT_total_bricks_fill_box_l1582_158205


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_1680_l1582_158248

theorem sum_of_consecutive_integers_with_product_1680 : 
  ∃ (a b c d : ℤ), (a * b * c * d = 1680 ∧ b = a + 1 ∧ c = a + 2 ∧ d = a + 3) → (a + b + c + d = 26) := sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_1680_l1582_158248


namespace NUMINAMATH_GPT_find_x_eq_14_4_l1582_158297

theorem find_x_eq_14_4 (x : ℝ) (h : ⌈x⌉ * x = 216) : x = 14.4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_eq_14_4_l1582_158297


namespace NUMINAMATH_GPT_vector_addition_correct_l1582_158269

open Matrix

-- Define the vectors as 3x1 matrices
def v1 : Matrix (Fin 3) (Fin 1) ℤ := ![![3], ![-5], ![1]]
def v2 : Matrix (Fin 3) (Fin 1) ℤ := ![![-1], ![4], ![-2]]
def v3 : Matrix (Fin 3) (Fin 1) ℤ := ![![2], ![-1], ![3]]

-- Define the scalar multiples
def scaled_v1 := (2 : ℤ) • v1
def scaled_v2 := (3 : ℤ) • v2
def neg_v3 := (-1 : ℤ) • v3

-- Define the summation result
def result := scaled_v1 + scaled_v2 + neg_v3

-- Define the expected result for verification
def expected_result : Matrix (Fin 3) (Fin 1) ℤ := ![![1], ![3], ![-7]]

-- The proof statement (without the proof itself)
theorem vector_addition_correct :
  result = expected_result := by
  sorry

end NUMINAMATH_GPT_vector_addition_correct_l1582_158269


namespace NUMINAMATH_GPT_initial_roses_l1582_158241

theorem initial_roses {x : ℕ} (h : x + 11 = 14) : x = 3 := by
  sorry

end NUMINAMATH_GPT_initial_roses_l1582_158241


namespace NUMINAMATH_GPT_largest_interior_angle_of_triangle_l1582_158236

theorem largest_interior_angle_of_triangle (exterior_ratio_2k : ℝ) (exterior_ratio_3k : ℝ) (exterior_ratio_4k : ℝ) (sum_exterior_angles : exterior_ratio_2k + exterior_ratio_3k + exterior_ratio_4k = 360) :
  180 - exterior_ratio_2k = 100 :=
by
  sorry

end NUMINAMATH_GPT_largest_interior_angle_of_triangle_l1582_158236


namespace NUMINAMATH_GPT_distinct_intersections_count_l1582_158273

theorem distinct_intersections_count :
  (∃ (x y : ℝ), (x + 2 * y = 7 ∧ 3 * x - 4 * y + 8 = 0) ∨ (x + 2 * y = 7 ∧ 4 * x + 5 * y - 20 = 0) ∨
                (x - 2 * y - 1 = 0 ∧ 3 * x - 4 * y = 8) ∨ (x - 2 * y - 1 = 0 ∧ 4 * x + 5 * y - 20 = 0)) ∧
  ∃ count : ℕ, count = 3 :=
by sorry

end NUMINAMATH_GPT_distinct_intersections_count_l1582_158273


namespace NUMINAMATH_GPT_least_four_digit_11_heavy_l1582_158207

def is_11_heavy (n : ℕ) : Prop := (n % 11) > 7

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem least_four_digit_11_heavy : ∃ n : ℕ, is_four_digit n ∧ is_11_heavy n ∧ 
  (∀ m : ℕ, is_four_digit m ∧ is_11_heavy m → 1000 ≤ n) := 
sorry

end NUMINAMATH_GPT_least_four_digit_11_heavy_l1582_158207


namespace NUMINAMATH_GPT_point_returns_to_original_after_seven_steps_l1582_158250

-- Define a structure for a triangle and a point inside it
structure Triangle :=
  (A B C : Point)

structure Point :=
  (x y : ℝ)

-- Given a triangle and a point inside it
variable (ABC : Triangle)
variable (M : Point)

-- Define the set of movements and the intersection points
def move_parallel_to_BC (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AB (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AC (M : Point) (ABC : Triangle) : Point := sorry

-- Function to perform the stepwise movement through 7 steps
def move_M_seven_times (M : Point) (ABC : Triangle) : Point :=
  let M1 := move_parallel_to_BC M ABC
  let M2 := move_parallel_to_AB M1 ABC 
  let M3 := move_parallel_to_AC M2 ABC
  let M4 := move_parallel_to_BC M3 ABC
  let M5 := move_parallel_to_AB M4 ABC
  let M6 := move_parallel_to_AC M5 ABC
  let M7 := move_parallel_to_BC M6 ABC
  M7

-- The theorem stating that after 7 steps, point M returns to its original position
theorem point_returns_to_original_after_seven_steps :
  move_M_seven_times M ABC = M := sorry

end NUMINAMATH_GPT_point_returns_to_original_after_seven_steps_l1582_158250


namespace NUMINAMATH_GPT_gcd_71_19_l1582_158247

theorem gcd_71_19 : Int.gcd 71 19 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_71_19_l1582_158247


namespace NUMINAMATH_GPT_monotonic_range_l1582_158274

theorem monotonic_range (a : ℝ) :
  (∀ x y, 2 ≤ x ∧ x ≤ 3 ∧ 2 ≤ y ∧ y ≤ 3 ∧ x < y → (x^2 - 2*a*x + 3) < (y^2 - 2*a*y + 3))
  ∨ (∀ x y, 2 ≤ x ∧ x ≤ 3 ∧ 2 ≤ y ∧ y ≤ 3 ∧ x < y → (x^2 - 2*a*x + 3) > (y^2 - 2*a*y + 3))
  ↔ (a ≤ 2 ∨ a ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_range_l1582_158274


namespace NUMINAMATH_GPT_violet_balloons_remaining_l1582_158268

def initial_count : ℕ := 7
def lost_count : ℕ := 3

theorem violet_balloons_remaining : initial_count - lost_count = 4 :=
by sorry

end NUMINAMATH_GPT_violet_balloons_remaining_l1582_158268


namespace NUMINAMATH_GPT_A_B_distance_l1582_158239

noncomputable def distance_between_A_and_B 
  (vA: ℕ) (vB: ℕ) (vA_after_return: ℕ) 
  (meet_distance: ℕ) : ℚ := sorry

theorem A_B_distance (distance: ℚ) 
  (hA: vA = 40) (hB: vB = 60) 
  (hA_after_return: vA_after_return = 60) 
  (hmeet: meet_distance = 50) : 
  distance_between_A_and_B vA vB vA_after_return meet_distance = 1000 / 7 := sorry

end NUMINAMATH_GPT_A_B_distance_l1582_158239


namespace NUMINAMATH_GPT_birds_in_sanctuary_l1582_158266

theorem birds_in_sanctuary (x y : ℕ) 
    (h1 : x + y = 200)
    (h2 : 2 * x + 4 * y = 590) : 
    x = 105 :=
by
  sorry

end NUMINAMATH_GPT_birds_in_sanctuary_l1582_158266


namespace NUMINAMATH_GPT_necessarily_negative_expression_l1582_158291

theorem necessarily_negative_expression
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 1)
  (hy : -1 < y ∧ y < 0)
  (hz : 0 < z ∧ z < 1)
  : y - z < 0 :=
sorry

end NUMINAMATH_GPT_necessarily_negative_expression_l1582_158291


namespace NUMINAMATH_GPT_base_6_units_digit_l1582_158262

def num1 : ℕ := 217
def num2 : ℕ := 45
def base : ℕ := 6

theorem base_6_units_digit :
  (num1 % base) * (num2 % base) % base = (num1 * num2) % base :=
by
  sorry

end NUMINAMATH_GPT_base_6_units_digit_l1582_158262


namespace NUMINAMATH_GPT_austin_tax_l1582_158254

theorem austin_tax 
  (number_of_robots : ℕ)
  (cost_per_robot change_left starting_amount : ℚ) 
  (h1 : number_of_robots = 7)
  (h2 : cost_per_robot = 8.75)
  (h3 : change_left = 11.53)
  (h4 : starting_amount = 80) : 
  ∃ tax : ℚ, tax = 7.22 :=
by
  sorry

end NUMINAMATH_GPT_austin_tax_l1582_158254


namespace NUMINAMATH_GPT_fractions_addition_l1582_158299

theorem fractions_addition :
  (1 / 3) * (3 / 4) * (1 / 5) + (1 / 6) = 13 / 60 :=
by 
  sorry

end NUMINAMATH_GPT_fractions_addition_l1582_158299


namespace NUMINAMATH_GPT_time_of_same_distance_l1582_158246

theorem time_of_same_distance (m : ℝ) (h_m : 0 ≤ m ∧ m ≤ 60) : 180 - 6 * m = 90 + 0.5 * m :=
by
  sorry

end NUMINAMATH_GPT_time_of_same_distance_l1582_158246


namespace NUMINAMATH_GPT_radius_ratio_of_smaller_to_larger_l1582_158286

noncomputable def ratio_of_radii (v_large v_small : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = v_large) (h_small : v_small = 0.25 * v_large) (h_small_sphere : (4/3) * Real.pi * r^3 = v_small) : ℝ :=
  let ratio := r / R
  ratio

theorem radius_ratio_of_smaller_to_larger (v_large : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = 576 * Real.pi) (h_small_sphere : (4/3) * Real.pi * r^3 = 0.25 * 576 * Real.pi) : r / R = 1 / (2^(2/3)) :=
by
  sorry

end NUMINAMATH_GPT_radius_ratio_of_smaller_to_larger_l1582_158286


namespace NUMINAMATH_GPT_smallest_n_for_Sn_gt_10_l1582_158227

noncomputable def harmonicSeriesSum : ℕ → ℝ
| 0       => 0
| (n + 1) => harmonicSeriesSum n + 1 / (n + 1)

theorem smallest_n_for_Sn_gt_10 : ∃ n : ℕ, (harmonicSeriesSum n > 10) ∧ ∀ k < 12367, harmonicSeriesSum k ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_Sn_gt_10_l1582_158227


namespace NUMINAMATH_GPT_new_arithmetic_mean_l1582_158276

theorem new_arithmetic_mean
  (seq : List ℝ)
  (h_seq_len : seq.length = 60)
  (h_mean : (seq.sum / 60 : ℝ) = 42)
  (h_removed : ∃ a b, a ∈ seq ∧ b ∈ seq ∧ a = 50 ∧ b = 60) :
  ((seq.erase 50).erase 60).sum / 58 = 41.55 := 
sorry

end NUMINAMATH_GPT_new_arithmetic_mean_l1582_158276


namespace NUMINAMATH_GPT_binder_cost_l1582_158200

variable (B : ℕ) -- Define B as the cost of each binder

theorem binder_cost :
  let book_cost := 16
  let num_binders := 3
  let notebook_cost := 1
  let num_notebooks := 6
  let total_cost := 28
  (book_cost + num_binders * B + num_notebooks * notebook_cost = total_cost) → (B = 2) :=
by
  sorry

end NUMINAMATH_GPT_binder_cost_l1582_158200


namespace NUMINAMATH_GPT_constant_t_exists_l1582_158233

theorem constant_t_exists (c : ℝ) :
  ∃ t : ℝ, (∀ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A.2 = A.1 * c + c) ∧ (B.2 = B.1 * c + c) → (t = -2)) :=
sorry

end NUMINAMATH_GPT_constant_t_exists_l1582_158233


namespace NUMINAMATH_GPT_first_term_exceeding_10000_l1582_158212

theorem first_term_exceeding_10000 :
  ∃ (n : ℕ), (2^(n-1) > 10000) ∧ (2^(n-1) = 16384) :=
by
  sorry

end NUMINAMATH_GPT_first_term_exceeding_10000_l1582_158212


namespace NUMINAMATH_GPT_consecutive_odd_split_l1582_158296

theorem consecutive_odd_split (m : ℕ) (hm : m > 1) : (∃ n : ℕ, n = 2015 ∧ n < ((m + 2) * (m - 1)) / 2) → m = 45 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_odd_split_l1582_158296


namespace NUMINAMATH_GPT_sally_gave_joan_5_balloons_l1582_158265

theorem sally_gave_joan_5_balloons (x : ℕ) (h1 : 9 + x - 2 = 12) : x = 5 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_sally_gave_joan_5_balloons_l1582_158265


namespace NUMINAMATH_GPT_gcd_3_666666666_equals_3_l1582_158280

theorem gcd_3_666666666_equals_3 :
  Nat.gcd 33333333 666666666 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_3_666666666_equals_3_l1582_158280


namespace NUMINAMATH_GPT_find_x_l1582_158215

theorem find_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end NUMINAMATH_GPT_find_x_l1582_158215


namespace NUMINAMATH_GPT_find_y_l1582_158203

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 2 + 1 / y) 
  (h2 : y = 3 + 1 / x) : 
  y = (3/2) + (Real.sqrt 15 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1582_158203


namespace NUMINAMATH_GPT_expression_evaluation_l1582_158293

theorem expression_evaluation : abs (abs (-abs (-2 + 1) - 2) + 2) = 5 := 
by  
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1582_158293


namespace NUMINAMATH_GPT_vendor_sales_first_day_l1582_158208

theorem vendor_sales_first_day (A S: ℝ) (h1: S = S / 100) 
  (h2: 0.20 * A * (1 - S / 100) = 0.42 * A - 0.50 * A * (0.80 * (1 - S / 100)))
  (h3: 0 < S) (h4: S < 100) : 
  S = 30 := 
by
  sorry

end NUMINAMATH_GPT_vendor_sales_first_day_l1582_158208


namespace NUMINAMATH_GPT_volume_of_remaining_solid_l1582_158252

noncomputable def volume_cube_with_cylindrical_hole 
  (side_length : ℝ) (hole_diameter : ℝ) (π : ℝ := 3.141592653589793) : ℝ :=
  let V_cube := side_length^3
  let radius := hole_diameter / 2
  let height := side_length
  let V_cylinder := π * radius^2 * height
  V_cube - V_cylinder

theorem volume_of_remaining_solid 
  (side_length : ℝ)
  (hole_diameter : ℝ)
  (h₁ : side_length = 6) 
  (h₂ : hole_diameter = 3)
  (π : ℝ := 3.141592653589793) : 
  abs (volume_cube_with_cylindrical_hole side_length hole_diameter π - 173.59) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_remaining_solid_l1582_158252


namespace NUMINAMATH_GPT_frank_spend_more_l1582_158230

noncomputable def table_cost : ℝ := 140
noncomputable def chair_cost : ℝ := 100
noncomputable def joystick_cost : ℝ := 20
noncomputable def frank_joystick : ℝ := joystick_cost * (1 / 4)
noncomputable def eman_joystick : ℝ := joystick_cost - frank_joystick
noncomputable def frank_total : ℝ := table_cost + frank_joystick
noncomputable def eman_total : ℝ := chair_cost + eman_joystick

theorem frank_spend_more :
  frank_total - eman_total = 30 :=
  sorry

end NUMINAMATH_GPT_frank_spend_more_l1582_158230


namespace NUMINAMATH_GPT_coordinates_of_P_l1582_158210

def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (-9, 2)
def P : ℝ × ℝ := (-1, -2)

theorem coordinates_of_P : P = (1 / 3 • (B.1 - A.1) + 2 / 3 • A.1, 1 / 3 • (B.2 - A.2) + 2 / 3 • A.2) :=
by
    rw [A, B, P]
    sorry

end NUMINAMATH_GPT_coordinates_of_P_l1582_158210


namespace NUMINAMATH_GPT_correct_option_is_B_l1582_158292

theorem correct_option_is_B :
  (∃ (A B C D : String), A = "√49 = -7" ∧ B = "√((-3)^2) = 3" ∧ C = "-√((-5)^2) = 5" ∧ D = "√81 = ±9" ∧
    (B = "√((-3)^2) = 3")) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l1582_158292


namespace NUMINAMATH_GPT_blu_ray_movies_returned_l1582_158213

theorem blu_ray_movies_returned (D B x : ℕ)
  (h1 : D / B = 17 / 4)
  (h2 : D + B = 378)
  (h3 : D / (B - x) = 9 / 2) :
  x = 4 := by
  sorry

end NUMINAMATH_GPT_blu_ray_movies_returned_l1582_158213
