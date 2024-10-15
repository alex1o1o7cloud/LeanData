import Mathlib

namespace NUMINAMATH_GPT_area_of_mirror_l1923_192351

theorem area_of_mirror (outer_width : ℝ) (outer_height : ℝ) (frame_width : ℝ) (mirror_area : ℝ) :
  outer_width = 70 → outer_height = 100 → frame_width = 15 → mirror_area = (outer_width - 2 * frame_width) * (outer_height - 2 * frame_width) → mirror_area = 2800 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h4]
  sorry

end NUMINAMATH_GPT_area_of_mirror_l1923_192351


namespace NUMINAMATH_GPT_walls_painted_purple_l1923_192388

theorem walls_painted_purple :
  (10 - (3 * 10 / 5)) * 8 = 32 := by
  sorry

end NUMINAMATH_GPT_walls_painted_purple_l1923_192388


namespace NUMINAMATH_GPT_apples_left_l1923_192309

theorem apples_left (initial_apples : ℕ) (difference_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 46) 
  (h2 : difference_apples = 32) 
  (h3 : final_apples = initial_apples - difference_apples) : 
  final_apples = 14 := 
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_apples_left_l1923_192309


namespace NUMINAMATH_GPT_rect_solution_proof_l1923_192340

noncomputable def rect_solution_exists : Prop :=
  ∃ (l2 w2 : ℝ), 2 * (l2 + w2) = 12 ∧ l2 * w2 = 4 ∧
               l2 = 3 + Real.sqrt 5 ∧ w2 = 3 - Real.sqrt 5

theorem rect_solution_proof : rect_solution_exists :=
  by
    sorry

end NUMINAMATH_GPT_rect_solution_proof_l1923_192340


namespace NUMINAMATH_GPT_compound_percentage_increase_l1923_192357

noncomputable def weeklyEarningsAfterRaises (initial : ℝ) (raises : List ℝ) : ℝ :=
  raises.foldl (λ sal raise_rate => sal * (1 + raise_rate / 100)) initial

theorem compound_percentage_increase :
  let initial := 60
  let raises := [10, 15, 12, 8]
  weeklyEarningsAfterRaises initial raises = 91.80864 ∧
  ((weeklyEarningsAfterRaises initial raises - initial) / initial * 100 = 53.0144) :=
by
  sorry

end NUMINAMATH_GPT_compound_percentage_increase_l1923_192357


namespace NUMINAMATH_GPT_increment_in_radius_l1923_192374

theorem increment_in_radius (C1 C2 : ℝ) (hC1 : C1 = 50) (hC2 : C2 = 60) : 
  ((C2 / (2 * Real.pi)) - (C1 / (2 * Real.pi)) = (5 / Real.pi)) :=
by
  sorry

end NUMINAMATH_GPT_increment_in_radius_l1923_192374


namespace NUMINAMATH_GPT_martha_bottles_l1923_192335

def total_bottles_left (a b c d : ℕ) : ℕ :=
  a + b + c - d

theorem martha_bottles : total_bottles_left 4 4 5 3 = 10 :=
by
  sorry

end NUMINAMATH_GPT_martha_bottles_l1923_192335


namespace NUMINAMATH_GPT_box_weight_l1923_192327

theorem box_weight (total_weight : ℕ) (number_of_boxes : ℕ) (box_weight : ℕ) 
  (h1 : total_weight = 267) 
  (h2 : number_of_boxes = 3) 
  (h3 : box_weight = total_weight / number_of_boxes) : 
  box_weight = 89 := 
by 
  sorry

end NUMINAMATH_GPT_box_weight_l1923_192327


namespace NUMINAMATH_GPT_trigonometric_identity_l1923_192387

-- Define the main theorem
theorem trigonometric_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1923_192387


namespace NUMINAMATH_GPT_quadratic_real_roots_l1923_192344

theorem quadratic_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c ≥ 0) : ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_real_roots_l1923_192344


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1923_192316

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}
def intersection_M_N : Set ℕ := {0, 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1923_192316


namespace NUMINAMATH_GPT_solve_x_l1923_192396

noncomputable def x : ℝ := 4.7

theorem solve_x : (10 - x) ^ 2 = x ^ 2 + 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_l1923_192396


namespace NUMINAMATH_GPT_multiply_neg_reverse_inequality_l1923_192383

theorem multiply_neg_reverse_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end NUMINAMATH_GPT_multiply_neg_reverse_inequality_l1923_192383


namespace NUMINAMATH_GPT_hiker_final_distance_l1923_192353

-- Definitions of the movements
def northward_movement : ℤ := 20
def southward_movement : ℤ := 8
def westward_movement : ℤ := 15
def eastward_movement : ℤ := 10

-- Definitions of the net movements
def net_north_south_movement : ℤ := northward_movement - southward_movement
def net_east_west_movement : ℤ := westward_movement - eastward_movement

-- The proof statement
theorem hiker_final_distance : 
  (net_north_south_movement^2 + net_east_west_movement^2) = 13^2 := by 
    sorry

end NUMINAMATH_GPT_hiker_final_distance_l1923_192353


namespace NUMINAMATH_GPT_radius_of_circle_B_l1923_192337

theorem radius_of_circle_B (diam_A : ℝ) (factor : ℝ) (r_A r_B : ℝ) 
  (h1 : diam_A = 80) 
  (h2 : r_A = diam_A / 2) 
  (h3 : r_A = factor * r_B) 
  (h4 : factor = 4) : r_B = 10 := 
by 
  sorry

end NUMINAMATH_GPT_radius_of_circle_B_l1923_192337


namespace NUMINAMATH_GPT_percentage_reduction_correct_l1923_192307

-- Define the initial conditions
def initial_conditions (P S : ℝ) (new_sales_increase_percentage net_sale_value_increase_percentage: ℝ) :=
  new_sales_increase_percentage = 0.72 ∧ net_sale_value_increase_percentage = 0.4104

-- Define the statement for the required percentage reduction
theorem percentage_reduction_correct (P S : ℝ) (x : ℝ) 
  (h : initial_conditions P S 0.72 0.4104) : 
  (S:ℝ) * (1 - x / 100) = 1.4104 * S := 
sorry

end NUMINAMATH_GPT_percentage_reduction_correct_l1923_192307


namespace NUMINAMATH_GPT_findCostPrices_l1923_192370

def costPriceOfApple (sp_a : ℝ) (cp_a : ℝ) : Prop :=
  sp_a = (5 / 6) * cp_a

def costPriceOfOrange (sp_o : ℝ) (cp_o : ℝ) : Prop :=
  sp_o = (3 / 4) * cp_o

def costPriceOfBanana (sp_b : ℝ) (cp_b : ℝ) : Prop :=
  sp_b = (9 / 8) * cp_b

theorem findCostPrices (sp_a sp_o sp_b : ℝ) (cp_a cp_o cp_b : ℝ) :
  costPriceOfApple sp_a cp_a → 
  costPriceOfOrange sp_o cp_o → 
  costPriceOfBanana sp_b cp_b → 
  sp_a = 20 → sp_o = 15 → sp_b = 6 → 
  cp_a = 24 ∧ cp_o = 20 ∧ cp_b = 16 / 3 :=
by 
  intro h1 h2 h3 sp_a_eq sp_o_eq sp_b_eq
  -- proof goes here
  sorry

end NUMINAMATH_GPT_findCostPrices_l1923_192370


namespace NUMINAMATH_GPT_area_of_plot_l1923_192398

def central_square_area : ℕ := 64

def common_perimeter : ℕ := 32

-- This statement formalizes the proof problem: "The area of Mrs. Lígia's plot is 256 m² given the provided conditions."
theorem area_of_plot (a b : ℕ) 
  (h1 : a * a = central_square_area)
  (h2 : b = a) 
  (h3 : 4 * a = common_perimeter)  
  (h4 : ∀ (x y : ℕ), x + y = 16)
  (h5 : ∀ (x : ℕ), x + a = 16) 
  : a * 16 = 256 :=
sorry

end NUMINAMATH_GPT_area_of_plot_l1923_192398


namespace NUMINAMATH_GPT_unsold_percentage_l1923_192355

def total_harvested : ℝ := 340.2
def sold_mm : ℝ := 125.5  -- Weight sold to Mrs. Maxwell
def sold_mw : ℝ := 78.25  -- Weight sold to Mr. Wilson
def sold_mb : ℝ := 43.8   -- Weight sold to Ms. Brown
def sold_mj : ℝ := 56.65  -- Weight sold to Mr. Johnson

noncomputable def percentage_unsold (total_harvested : ℝ) 
                                   (sold_mm : ℝ) 
                                   (sold_mw : ℝ)
                                   (sold_mb : ℝ) 
                                   (sold_mj : ℝ) : ℝ :=
  let total_sold := sold_mm + sold_mw + sold_mb + sold_mj
  let unsold := total_harvested - total_sold
  (unsold / total_harvested) * 100

theorem unsold_percentage : percentage_unsold total_harvested sold_mm sold_mw sold_mb sold_mj = 10.58 :=
by
  sorry

end NUMINAMATH_GPT_unsold_percentage_l1923_192355


namespace NUMINAMATH_GPT_nancy_other_albums_count_l1923_192346

-- Definitions based on the given conditions
def total_pictures : ℕ := 51
def pics_in_first_album : ℕ := 11
def pics_per_other_album : ℕ := 5

-- Theorem to prove the question's answer
theorem nancy_other_albums_count : 
  (total_pictures - pics_in_first_album) / pics_per_other_album = 8 := by
  sorry

end NUMINAMATH_GPT_nancy_other_albums_count_l1923_192346


namespace NUMINAMATH_GPT_intersection_A_B_l1923_192392

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {-3, 1, 2, 4}

theorem intersection_A_B :
  A ∩ B = {-3, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1923_192392


namespace NUMINAMATH_GPT_total_cost_is_correct_l1923_192369

-- Definitions of the conditions given
def price_iphone12 : ℝ := 800
def price_iwatch : ℝ := 300
def discount_iphone12 : ℝ := 0.15
def discount_iwatch : ℝ := 0.1
def cashback_discount : ℝ := 0.02

-- The final total cost after applying all discounts and cashback
def total_cost_after_discounts_and_cashback : ℝ :=
  let discount_amount_iphone12 := price_iphone12 * discount_iphone12
  let new_price_iphone12 := price_iphone12 - discount_amount_iphone12
  let discount_amount_iwatch := price_iwatch * discount_iwatch
  let new_price_iwatch := price_iwatch - discount_amount_iwatch
  let initial_total_cost := new_price_iphone12 + new_price_iwatch
  let cashback_amount := initial_total_cost * cashback_discount
  initial_total_cost - cashback_amount

-- Statement to be proved
theorem total_cost_is_correct :
  total_cost_after_discounts_and_cashback = 931 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l1923_192369


namespace NUMINAMATH_GPT_runner_distance_l1923_192385

theorem runner_distance :
  ∃ x t d : ℕ,
    d = x * t ∧
    d = (x + 1) * (2 * t / 3) ∧
    d = (x - 1) * (t + 3) ∧
    d = 6 :=
by
  sorry

end NUMINAMATH_GPT_runner_distance_l1923_192385


namespace NUMINAMATH_GPT_value_of_g_neg2_l1923_192391

def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem value_of_g_neg2 : g (-2) = -1 := by
  sorry

end NUMINAMATH_GPT_value_of_g_neg2_l1923_192391


namespace NUMINAMATH_GPT_positive_integer_solution_of_inequality_l1923_192345

theorem positive_integer_solution_of_inequality :
  {x : ℕ // 0 < x ∧ x < 2} → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solution_of_inequality_l1923_192345


namespace NUMINAMATH_GPT_determine_f_2014_l1923_192338

open Function

noncomputable def f : ℕ → ℕ :=
  sorry

theorem determine_f_2014
  (h1 : f 2 = 0)
  (h2 : f 3 > 0)
  (h3 : f 6042 = 2014)
  (h4 : ∀ m n : ℕ, f (m + n) - f m - f n ∈ ({0, 1} : Set ℕ)) :
  f 2014 = 671 :=
sorry

end NUMINAMATH_GPT_determine_f_2014_l1923_192338


namespace NUMINAMATH_GPT_student_calls_out_2005th_l1923_192348

theorem student_calls_out_2005th : 
  ∀ (n : ℕ), n = 2005 → ∃ k : ℕ, k ∈ [1, 2, 3, 4, 3, 2, 1] ∧ k = 1 := 
by
  sorry

end NUMINAMATH_GPT_student_calls_out_2005th_l1923_192348


namespace NUMINAMATH_GPT_average_age_decrease_l1923_192368

theorem average_age_decrease :
  let avg_original := 40
  let new_students := 15
  let avg_new_students := 32
  let original_strength := 15
  let total_age_original := original_strength * avg_original
  let total_age_new_students := new_students * avg_new_students
  let total_strength := original_strength + new_students
  let total_age := total_age_original + total_age_new_students
  let avg_new := total_age / total_strength
  avg_original - avg_new = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_age_decrease_l1923_192368


namespace NUMINAMATH_GPT_equivalent_terminal_angle_l1923_192315

theorem equivalent_terminal_angle :
  ∃ n : ℤ, 660 = n * 360 - 420 := 
by
  sorry

end NUMINAMATH_GPT_equivalent_terminal_angle_l1923_192315


namespace NUMINAMATH_GPT_total_students_count_l1923_192301

variable (T : ℕ)
variable (J : ℕ) (S : ℕ) (F : ℕ) (Sn : ℕ)

-- Given conditions:
-- 1. 26 percent are juniors.
def percentage_juniors (T J : ℕ) : Prop := J = 26 * T / 100
-- 2. 75 percent are not sophomores.
def percentage_sophomores (T S : ℕ) : Prop := S = 25 * T / 100
-- 3. There are 160 seniors.
def seniors_count (Sn : ℕ) : Prop := Sn = 160
-- 4. There are 32 more freshmen than sophomores.
def freshmen_sophomore_relationship (F S : ℕ) : Prop := F = S + 32

-- Question: Prove the total number of students is 800.
theorem total_students_count
  (hJ : percentage_juniors T J)
  (hS : percentage_sophomores T S)
  (hSn : seniors_count Sn)
  (hF : freshmen_sophomore_relationship F S) :
  F + S + J + Sn = T → T = 800 := by
  sorry

end NUMINAMATH_GPT_total_students_count_l1923_192301


namespace NUMINAMATH_GPT_initial_pencils_sold_l1923_192361

theorem initial_pencils_sold (x : ℕ) (P : ℝ)
  (h1 : 1 = 0.9 * (x * P))
  (h2 : 1 = 1.2 * (8.25 * P))
  : x = 11 :=
by sorry

end NUMINAMATH_GPT_initial_pencils_sold_l1923_192361


namespace NUMINAMATH_GPT_computer_price_ratio_l1923_192399

theorem computer_price_ratio (d : ℝ) (h1 : d + 0.30 * d = 377) :
  ((d + 377) / d) = 2.3 := by
  sorry

end NUMINAMATH_GPT_computer_price_ratio_l1923_192399


namespace NUMINAMATH_GPT_xy_eq_119_imp_sum_values_l1923_192349

theorem xy_eq_119_imp_sum_values (x y : ℕ) (hx : x > 0) (hy : y > 0)
(hx_lt_30 : x < 30) (hy_lt_30 : y < 30) (h : x + y + x * y = 119) :
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 := 
sorry

end NUMINAMATH_GPT_xy_eq_119_imp_sum_values_l1923_192349


namespace NUMINAMATH_GPT_tan_add_pi_over_four_sin_cos_ratio_l1923_192331

-- Definition of angle α with the condition that tanα = 2
def α : ℝ := sorry -- Define α such that tan α = 2

-- The first Lean statement for proving tan(α + π/4) = -3
theorem tan_add_pi_over_four (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- The second Lean statement for proving (sinα + cosα) / (2sinα - cosα) = 1
theorem sin_cos_ratio (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1 :=
sorry

end NUMINAMATH_GPT_tan_add_pi_over_four_sin_cos_ratio_l1923_192331


namespace NUMINAMATH_GPT_christmas_sale_pricing_l1923_192373

theorem christmas_sale_pricing (a b : ℝ) : 
  (forall (c : ℝ), c = a * (3 / 5)) ∧ (forall (d : ℝ), d = b * (5 / 3)) :=
by
  sorry  -- proof goes here

end NUMINAMATH_GPT_christmas_sale_pricing_l1923_192373


namespace NUMINAMATH_GPT_prob_A_not_losing_is_correct_l1923_192320

def prob_A_wins := 0.4
def prob_draw := 0.2
def prob_A_not_losing := 0.6

theorem prob_A_not_losing_is_correct : prob_A_wins + prob_draw = prob_A_not_losing :=
by sorry

end NUMINAMATH_GPT_prob_A_not_losing_is_correct_l1923_192320


namespace NUMINAMATH_GPT_tiling_problem_l1923_192330

theorem tiling_problem (b c f : ℕ) (h : b * c = f) : c * (b^2 / f) = b :=
by 
  sorry

end NUMINAMATH_GPT_tiling_problem_l1923_192330


namespace NUMINAMATH_GPT_susans_total_chairs_l1923_192390

def number_of_red_chairs := 5
def number_of_yellow_chairs := 4 * number_of_red_chairs
def number_of_blue_chairs := number_of_yellow_chairs - 2
def total_chairs := number_of_red_chairs + number_of_yellow_chairs + number_of_blue_chairs

theorem susans_total_chairs : total_chairs = 43 :=
by
  sorry

end NUMINAMATH_GPT_susans_total_chairs_l1923_192390


namespace NUMINAMATH_GPT_largest_triangle_perimeter_l1923_192354

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem largest_triangle_perimeter : 
  ∃ (x : ℕ), x ≤ 14 ∧ 2 ≤ x ∧ is_valid_triangle 7 8 x ∧ (7 + 8 + x = 29) :=
sorry

end NUMINAMATH_GPT_largest_triangle_perimeter_l1923_192354


namespace NUMINAMATH_GPT_number_of_m_gons_proof_l1923_192334

noncomputable def number_of_m_gons_with_two_acute_angles (m n : ℕ) (h1 : 4 < m) (h2 : m < n) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem number_of_m_gons_proof {m n : ℕ} (h1 : 4 < m) (h2 : m < n) :
  number_of_m_gons_with_two_acute_angles m n h1 h2 =
  (2 * n + 1) * ((Nat.choose (n + 1) (m - 1)) + (Nat.choose n (m - 1))) :=
sorry

end NUMINAMATH_GPT_number_of_m_gons_proof_l1923_192334


namespace NUMINAMATH_GPT_find_n_l1923_192360

theorem find_n (P s k m n : ℝ) (h : P = s / (1 + k + m) ^ n) :
  n = (Real.log (s / P)) / (Real.log (1 + k + m)) :=
sorry

end NUMINAMATH_GPT_find_n_l1923_192360


namespace NUMINAMATH_GPT_largest_digit_for_divisibility_l1923_192389

theorem largest_digit_for_divisibility (N : ℕ) (h1 : N % 2 = 0) (h2 : (3 + 6 + 7 + 2 + N) % 3 = 0) : N = 6 :=
sorry

end NUMINAMATH_GPT_largest_digit_for_divisibility_l1923_192389


namespace NUMINAMATH_GPT_companyKW_price_percentage_l1923_192371

theorem companyKW_price_percentage (A B P : ℝ) (h1 : P = 1.40 * A) (h2 : P = 2.00 * B) : 
  P / ((P / 1.40) + (P / 2.00)) * 100 = 82.35 :=
by sorry

end NUMINAMATH_GPT_companyKW_price_percentage_l1923_192371


namespace NUMINAMATH_GPT_fraction_identity_l1923_192342

theorem fraction_identity (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := 
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1923_192342


namespace NUMINAMATH_GPT_hall_volume_l1923_192393

theorem hall_volume (length breadth : ℝ) (height : ℝ := 20 / 3)
  (h1 : length = 15)
  (h2 : breadth = 12)
  (h3 : 2 * (length * breadth) = 54 * height) :
  length * breadth * height = 8004 :=
by
  sorry

end NUMINAMATH_GPT_hall_volume_l1923_192393


namespace NUMINAMATH_GPT_avg_difference_l1923_192305

theorem avg_difference : 
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  avg1 - avg2 = 5 :=
by
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  show avg1 - avg2 = 5
  sorry

end NUMINAMATH_GPT_avg_difference_l1923_192305


namespace NUMINAMATH_GPT_tan_alpha_parallel_vectors_l1923_192318

theorem tan_alpha_parallel_vectors
    (α : ℝ)
    (a : ℝ × ℝ := (6, 8))
    (b : ℝ × ℝ := (Real.sin α, Real.cos α))
    (h : a.fst * b.snd = a.snd * b.fst) :
    Real.tan α = 3 / 4 := 
sorry

end NUMINAMATH_GPT_tan_alpha_parallel_vectors_l1923_192318


namespace NUMINAMATH_GPT_number_of_green_eyes_l1923_192386

-- Definitions based on conditions
def total_people : Nat := 100
def blue_eyes : Nat := 19
def brown_eyes : Nat := total_people / 2
def black_eyes : Nat := total_people / 4

-- Theorem stating the main question and its answer
theorem number_of_green_eyes : 
  (total_people - (blue_eyes + brown_eyes + black_eyes)) = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_green_eyes_l1923_192386


namespace NUMINAMATH_GPT_minimum_value_l1923_192358

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + (1 / y)) * (x + (1 / y) - 1024) +
  (y + (1 / x)) * (y + (1 / x) - 1024) ≥ -524288 :=
by sorry

end NUMINAMATH_GPT_minimum_value_l1923_192358


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1923_192377

theorem hyperbola_eccentricity (C : Type) (a b c e : ℝ)
  (h_asymptotes : ∀ x : ℝ, (∃ y : ℝ, y = x ∨ y = -x)) :
  a = b ∧ c = Real.sqrt (a^2 + b^2) ∧ e = c / a → e = Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1923_192377


namespace NUMINAMATH_GPT_find_m_l1923_192323

theorem find_m (m : ℝ) (x : ℝ) (h : 2*x + m = 1) (hx : x = -1) : m = 3 := 
by
  rw [hx] at h
  linarith

end NUMINAMATH_GPT_find_m_l1923_192323


namespace NUMINAMATH_GPT_find_a3_plus_a5_l1923_192336

variable {a : ℕ → ℝ}

-- Condition 1: The sequence {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Condition 2: All terms in the sequence are negative
def all_negative (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < 0

-- Condition 3: The given equation
def given_equation (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25

-- The problem statement
theorem find_a3_plus_a5 (h_geo : is_geometric_sequence a) (h_neg : all_negative a) (h_eq : given_equation a) :
  a 3 + a 5 = -5 :=
sorry

end NUMINAMATH_GPT_find_a3_plus_a5_l1923_192336


namespace NUMINAMATH_GPT_afb_leq_bfa_l1923_192303

open Real

variable {f : ℝ → ℝ}

theorem afb_leq_bfa
  (h_nonneg : ∀ x > 0, f x ≥ 0)
  (h_diff : ∀ x > 0, DifferentiableAt ℝ f x)
  (h_cond : ∀ x > 0, x * (deriv (deriv f) x) - f x ≤ 0)
  (a b : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_a_lt_b : a < b) :
  a * f b ≤ b * f a := 
sorry

end NUMINAMATH_GPT_afb_leq_bfa_l1923_192303


namespace NUMINAMATH_GPT_average_weight_increase_l1923_192372

theorem average_weight_increase (n : ℕ) (w_old w_new : ℝ) (h1 : n = 9) (h2 : w_old = 65) (h3 : w_new = 87.5) :
  (w_new - w_old) / n = 2.5 :=
by
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_average_weight_increase_l1923_192372


namespace NUMINAMATH_GPT_union_of_intervals_l1923_192362

theorem union_of_intervals :
  let P := {x : ℝ | -1 < x ∧ x < 1}
  let Q := {x : ℝ | -2 < x ∧ x < 0}
  P ∪ Q = {x : ℝ | -2 < x ∧ x < 1} :=
by
  let P := {x : ℝ | -1 < x ∧ x < 1}
  let Q := {x : ℝ | -2 < x ∧ x < 0}
  have h : P ∪ Q = {x : ℝ | -2 < x ∧ x < 1}
  {
     sorry
  }
  exact h

end NUMINAMATH_GPT_union_of_intervals_l1923_192362


namespace NUMINAMATH_GPT_calculate_total_bricks_l1923_192397

-- Given definitions based on the problem.
variables (a d g h : ℕ)

-- Definitions for the questions in terms of variables.
def days_to_build_bricks (a d g : ℕ) : ℕ :=
  (a * g) / d

def total_bricks_with_additional_men (a d g h : ℕ) : ℕ :=
  a + ((d + h) * a) / 2

theorem calculate_total_bricks (a d g h : ℕ)
  (h1 : 0 < d)
  (h2 : 0 < g)
  (h3 : 0 < a) :
  days_to_build_bricks a d g = a * g / d ∧
  total_bricks_with_additional_men a d g h = (3 * a + h * a) / 2 :=
  by sorry

end NUMINAMATH_GPT_calculate_total_bricks_l1923_192397


namespace NUMINAMATH_GPT_find_a_l1923_192322

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - Real.exp (1 - x) - a * x
noncomputable def g (a x : ℝ) : ℝ := Real.exp x + Real.exp (1 - x) - a

theorem find_a (x₁ x₂ a : ℝ) (h₁ : g a x₁ = 0) (h₂ : g a x₂ = 0) (hf : f a x₁ + f a x₂ = -4) : a = 4 :=
sorry

end NUMINAMATH_GPT_find_a_l1923_192322


namespace NUMINAMATH_GPT_find_triples_l1923_192364

theorem find_triples (x y z : ℝ) 
  (h1 : (1/3 : ℝ) * min x y + (2/3 : ℝ) * max x y = 2017)
  (h2 : (1/3 : ℝ) * min y z + (2/3 : ℝ) * max y z = 2018)
  (h3 : (1/3 : ℝ) * min z x + (2/3 : ℝ) * max z x = 2019) :
  (x = 2019) ∧ (y = 2016) ∧ (z = 2019) :=
sorry

end NUMINAMATH_GPT_find_triples_l1923_192364


namespace NUMINAMATH_GPT_train_meeting_distance_l1923_192312

theorem train_meeting_distance
  (d : ℝ) (tx ty: ℝ) (dx dy: ℝ)
  (hx : dx = 140) 
  (hy : dy = 140)
  (hx_speed : dx / tx = 35) 
  (hy_speed : dy / ty = 46.67) 
  (meet : tx = ty) :
  d = 60 := 
sorry

end NUMINAMATH_GPT_train_meeting_distance_l1923_192312


namespace NUMINAMATH_GPT_find_functions_l1923_192365

noncomputable def satisfies_condition (f : ℝ → ℝ) :=
  ∀ (p q r s : ℝ), p > 0 → q > 0 → r > 0 → s > 0 →
  (p * q = r * s) →
  (f p ^ 2 + f q ^ 2) / (f (r ^ 2) + f (s ^ 2)) = 
  (p ^ 2 + q ^ 2) / (r ^ 2 + s ^ 2)

theorem find_functions :
  ∀ (f : ℝ → ℝ),
  (satisfies_condition f) → 
  (∀ x : ℝ, x > 0 → f x = x ∨ f x = 1 / x) :=
by
  sorry

end NUMINAMATH_GPT_find_functions_l1923_192365


namespace NUMINAMATH_GPT_total_pages_read_l1923_192367

theorem total_pages_read (J A C D : ℝ) 
  (hJ : J = 20)
  (hA : A = 2 * J + 2)
  (hC : C = J * A - 17)
  (hD : D = (C + J) / 2) :
  J + A + C + D = 1306.5 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_read_l1923_192367


namespace NUMINAMATH_GPT_y_intercept_of_line_l1923_192381

theorem y_intercept_of_line (x y : ℝ) : x + 2 * y + 6 = 0 → x = 0 → y = -3 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1923_192381


namespace NUMINAMATH_GPT_autumn_grain_purchase_exceeds_1_8_billion_tons_l1923_192395

variable (x : ℝ)

theorem autumn_grain_purchase_exceeds_1_8_billion_tons 
  (h : x > 0.18) : 
  x > 1.8 := 
by 
  sorry

end NUMINAMATH_GPT_autumn_grain_purchase_exceeds_1_8_billion_tons_l1923_192395


namespace NUMINAMATH_GPT_trader_profit_l1923_192328

-- Definitions and conditions
def original_price (P : ℝ) := P
def discounted_price (P : ℝ) := 0.70 * P
def marked_up_price (P : ℝ) := 0.84 * P
def sale_price (P : ℝ) := 0.714 * P
def final_price (P : ℝ) := 1.2138 * P

-- Proof statement
theorem trader_profit (P : ℝ) : ((final_price P - original_price P) / original_price P) * 100 = 21.38 := by
  sorry

end NUMINAMATH_GPT_trader_profit_l1923_192328


namespace NUMINAMATH_GPT_polynomial_equality_l1923_192304

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 5 * x + 7
noncomputable def g (x : ℝ) : ℝ := 12 * x^2 - 19 * x + 25

theorem polynomial_equality :
  f 3 = g 3 ∧ f (3 - Real.sqrt 3) = g (3 - Real.sqrt 3) ∧ f (3 + Real.sqrt 3) = g (3 + Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_equality_l1923_192304


namespace NUMINAMATH_GPT_max_band_members_l1923_192325

theorem max_band_members 
  (m : ℤ)
  (h1 : 30 * m % 31 = 7)
  (h2 : 30 * m < 1500) : 
  30 * m = 720 :=
sorry

end NUMINAMATH_GPT_max_band_members_l1923_192325


namespace NUMINAMATH_GPT_value_of_a1_plus_a10_l1923_192378

noncomputable def geometric_sequence {α : Type*} [Field α] (a : ℕ → α) :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem value_of_a1_plus_a10 (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : a 4 + a 7 = 2) 
  (h3 : a 5 * a 6 = -8) 
  : a 1 + a 10 = -7 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a1_plus_a10_l1923_192378


namespace NUMINAMATH_GPT_triangle_area_l1923_192326

theorem triangle_area (AB CD : ℝ) (h₁ : 0 < AB) (h₂ : 0 < CD) (h₃ : CD = 3 * AB) :
    let trapezoid_area := 18
    let triangle_ABC_area := trapezoid_area / 4
    triangle_ABC_area = 4.5 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1923_192326


namespace NUMINAMATH_GPT_x_value_for_divisibility_l1923_192363

theorem x_value_for_divisibility (x : ℕ) (h1 : x = 0 ∨ x = 5) (h2 : (8 * 10 + x) % 4 = 0) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_x_value_for_divisibility_l1923_192363


namespace NUMINAMATH_GPT_minimize_std_deviation_l1923_192321

theorem minimize_std_deviation (m n : ℝ) (h1 : m + n = 32) 
    (h2 : 11 ≤ 12 ∧ 12 ≤ m ∧ m ≤ n ∧ n ≤ 20 ∧ 20 ≤ 27) : 
    m = 16 :=
by {
  -- No proof required, only the theorem statement as per instructions
  sorry
}

end NUMINAMATH_GPT_minimize_std_deviation_l1923_192321


namespace NUMINAMATH_GPT_area_of_garden_l1923_192343

theorem area_of_garden :
  ∃ (short_posts long_posts : ℕ), short_posts + long_posts - 4 = 24 → long_posts = 3 * short_posts →
  ∃ (short_length long_length : ℕ), short_length = (short_posts - 1) * 5 → long_length = (long_posts - 1) * 5 →
  (short_length * long_length = 3000) :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_garden_l1923_192343


namespace NUMINAMATH_GPT_gray_region_area_l1923_192382

theorem gray_region_area
  (center_C : ℝ × ℝ) (r_C : ℝ)
  (center_D : ℝ × ℝ) (r_D : ℝ)
  (C_center : center_C = (3, 5)) (C_radius : r_C = 5)
  (D_center : center_D = (13, 5)) (D_radius : r_D = 5) :
  let rect_area := 10 * 5
  let semi_circle_area := 12.5 * π
  rect_area - 2 * semi_circle_area = 50 - 25 * π := 
by 
  sorry

end NUMINAMATH_GPT_gray_region_area_l1923_192382


namespace NUMINAMATH_GPT_find_number_l1923_192384

noncomputable def number := 115.2 / 0.32

theorem find_number : number = 360 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1923_192384


namespace NUMINAMATH_GPT_sally_out_of_pocket_l1923_192339

-- Definitions based on conditions
def g : ℕ := 320 -- Amount given by the school
def c : ℕ := 12  -- Cost per book
def n : ℕ := 30  -- Number of students

-- Definition derived from conditions
def total_cost : ℕ := n * c
def out_of_pocket : ℕ := total_cost - g

-- Proof statement
theorem sally_out_of_pocket : out_of_pocket = 40 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_sally_out_of_pocket_l1923_192339


namespace NUMINAMATH_GPT_roots_cubic_identity_l1923_192306

theorem roots_cubic_identity (r s : ℚ) (h1 : 3 * r^2 + 5 * r + 2 = 0) (h2 : 3 * s^2 + 5 * s + 2 = 0) :
  (1 / r^3) + (1 / s^3) = -27 / 35 :=
sorry

end NUMINAMATH_GPT_roots_cubic_identity_l1923_192306


namespace NUMINAMATH_GPT_smallest_angle_in_triangle_l1923_192310

theorem smallest_angle_in_triangle (a b c x : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 5 * x)
  (h3 : b = 3 * x) :
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_in_triangle_l1923_192310


namespace NUMINAMATH_GPT_ant_weight_statement_l1923_192302

variable (R : ℝ) -- Rupert's weight
variable (A : ℝ) -- Antoinette's weight
variable (C : ℝ) -- Charles's weight

-- Conditions
def condition1 : Prop := A = 2 * R - 7
def condition2 : Prop := C = (A + R) / 2 + 5
def condition3 : Prop := A + R + C = 145

-- Question: Prove Antoinette's weight
def ant_weight_proof : Prop :=
  ∃ R A C, condition1 R A ∧ condition2 R A C ∧ condition3 R A C ∧ A = 79

theorem ant_weight_statement : ant_weight_proof :=
sorry

end NUMINAMATH_GPT_ant_weight_statement_l1923_192302


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l1923_192300

theorem equation1_solution (x : ℚ) : 2 * (x - 3) = 1 - 3 * (x + 1) → x = 4 / 5 :=
by sorry

theorem equation2_solution (x : ℚ) : 3 * x + (x - 1) / 2 = 3 - (x - 1) / 3 → x = 1 :=
by sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l1923_192300


namespace NUMINAMATH_GPT_rational_number_div_l1923_192380

theorem rational_number_div (x : ℚ) (h : -2 / x = 8) : x = -1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_rational_number_div_l1923_192380


namespace NUMINAMATH_GPT_collinear_points_sum_l1923_192333

theorem collinear_points_sum (x y : ℝ) : 
  (∃ a b : ℝ, a * x + b * 3 + (1 - a - b) * 2 = a * x + b * y + (1 - a - b) * y ∧ 
               a * y + b * 4 + (1 - a - b) * y = a * x + b * y + (1 - a - b) * x) → 
  x = 2 → y = 4 → x + y = 6 :=
by sorry

end NUMINAMATH_GPT_collinear_points_sum_l1923_192333


namespace NUMINAMATH_GPT_pentagon_square_ratio_l1923_192347

theorem pentagon_square_ratio (p s : ℕ) 
  (h1 : 5 * p = 20) (h2 : 4 * s = 20) : p / s = 4 / 5 :=
by sorry

end NUMINAMATH_GPT_pentagon_square_ratio_l1923_192347


namespace NUMINAMATH_GPT_number_of_non_degenerate_rectangles_excluding_center_l1923_192376

/-!
# Problem Statement
We want to find the number of non-degenerate rectangles in a 7x7 grid that do not fully cover the center point (4, 4).
-/

def num_rectangles_excluding_center : Nat :=
  let total_rectangles := (Nat.choose 7 2) * (Nat.choose 7 2)
  let rectangles_including_center := 4 * ((3 * 3 * 3) + (3 * 3))
  total_rectangles - rectangles_including_center

theorem number_of_non_degenerate_rectangles_excluding_center :
  num_rectangles_excluding_center = 297 :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_number_of_non_degenerate_rectangles_excluding_center_l1923_192376


namespace NUMINAMATH_GPT_not_cheap_is_necessary_condition_l1923_192379

-- Define propositions for "good quality" and "not cheap"
variables {P: Prop} {Q: Prop} 

-- Statement "You get what you pay for" implies "good quality is not cheap"
axiom H : P → Q 

-- The proof problem
theorem not_cheap_is_necessary_condition (H : P → Q) : Q → P :=
by sorry

end NUMINAMATH_GPT_not_cheap_is_necessary_condition_l1923_192379


namespace NUMINAMATH_GPT_find_f_value_l1923_192317

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f (x : ℝ) : f (-x) = -f x
axiom even_f_shift (x : ℝ) : f (-x + 1) = f (x + 1)
axiom f_interval (x : ℝ) (h : 2 < x ∧ x < 4) : f x = |x - 3|

theorem find_f_value : f 1 + f 2 + f 3 + f 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_f_value_l1923_192317


namespace NUMINAMATH_GPT_seating_arrangement_l1923_192324

def valid_arrangements := 6

def Alice_refusal (A B C : Prop) := (¬ (A ∧ B)) ∧ (¬ (A ∧ C))
def Derek_refusal (D E C : Prop) := (¬ (D ∧ E)) ∧ (¬ (D ∧ C))

theorem seating_arrangement (A B C D E : Prop) : 
  Alice_refusal A B C ∧ Derek_refusal D E C → valid_arrangements = 6 := 
  sorry

end NUMINAMATH_GPT_seating_arrangement_l1923_192324


namespace NUMINAMATH_GPT_perimeter_of_triangle_l1923_192359

theorem perimeter_of_triangle
  {r A P : ℝ} (hr : r = 2.5) (hA : A = 25) :
  P = 20 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l1923_192359


namespace NUMINAMATH_GPT_sqrt6_special_op_l1923_192313

-- Define the binary operation (¤) as given in the problem.
def special_op (x y : ℝ) : ℝ := (x + y) ^ 2 - (x - y) ^ 2

-- States that √6 ¤ √6 is equal to 24.
theorem sqrt6_special_op : special_op (Real.sqrt 6) (Real.sqrt 6) = 24 :=
by
  sorry

end NUMINAMATH_GPT_sqrt6_special_op_l1923_192313


namespace NUMINAMATH_GPT_smallest_N_satisfying_conditions_l1923_192319

def is_divisible (n m : ℕ) : Prop :=
  m ∣ n

def satisfies_conditions (N : ℕ) : Prop :=
  (is_divisible N 10) ∧
  (is_divisible N 5) ∧
  (N > 15)

theorem smallest_N_satisfying_conditions : ∃ N, satisfies_conditions N ∧ N = 20 := 
  sorry

end NUMINAMATH_GPT_smallest_N_satisfying_conditions_l1923_192319


namespace NUMINAMATH_GPT_gg_of_3_is_107_l1923_192375

-- Define the function g
def g (x : ℕ) : ℕ := 3 * x + 2

-- State that g(g(g(3))) equals 107
theorem gg_of_3_is_107 : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_GPT_gg_of_3_is_107_l1923_192375


namespace NUMINAMATH_GPT_intersecting_points_radius_squared_l1923_192332

noncomputable def parabola1 (x : ℝ) : ℝ := (x - 2) ^ 2
noncomputable def parabola2 (y : ℝ) : ℝ := (y - 5) ^ 2 - 1

theorem intersecting_points_radius_squared :
  ∃ (x y : ℝ), (y = parabola1 x ∧ x = parabola2 y) → (x - 2) ^ 2 + (y - 5) ^ 2 = 16 := by
sorry

end NUMINAMATH_GPT_intersecting_points_radius_squared_l1923_192332


namespace NUMINAMATH_GPT_find_f_neg_2_l1923_192341

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition 2: f is defined on ℝ
-- This is implicitly handled as f : ℝ → ℝ

-- Condition 3: f(x+2) = -f(x)
def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

theorem find_f_neg_2 (h₁ : odd_function f) (h₂ : periodic_function f) : f (-2) = 0 :=
  sorry

end NUMINAMATH_GPT_find_f_neg_2_l1923_192341


namespace NUMINAMATH_GPT_rachel_biology_homework_pages_l1923_192350

-- Declare the known quantities
def math_pages : ℕ := 8
def total_math_biology_pages : ℕ := 11

-- Define biology_pages
def biology_pages : ℕ := total_math_biology_pages - math_pages

-- Assert the main theorem
theorem rachel_biology_homework_pages : biology_pages = 3 :=
by 
  -- Proof is omitted as instructed
  sorry

end NUMINAMATH_GPT_rachel_biology_homework_pages_l1923_192350


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_root_products_eq_4_l1923_192314

theorem sum_of_reciprocals_of_root_products_eq_4
  (p q r s t : ℂ)
  (h_poly : ∀ x : ℂ, x^5 + 10*x^4 + 20*x^3 + 15*x^2 + 8*x + 5 = 0 ∨ (x - p)*(x - q)*(x - r)*(x - s)*(x - t) = 0)
  (h_vieta_2 : p*q + p*r + p*s + p*t + q*r + q*s + q*t + r*s + r*t + s*t = 20)
  (h_vieta_all : p*q*r*s*t = 5) :
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 4 := 
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_root_products_eq_4_l1923_192314


namespace NUMINAMATH_GPT_f_zero_add_f_neg_three_l1923_192329

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_add (x y : ℝ) : f x + f y = f (x + y)

axiom f_three : f 3 = 4

theorem f_zero_add_f_neg_three : f 0 + f (-3) = -4 :=
by
  sorry

end NUMINAMATH_GPT_f_zero_add_f_neg_three_l1923_192329


namespace NUMINAMATH_GPT_circle_center_l1923_192308

theorem circle_center (x y : ℝ) :
  x^2 + y^2 - 2*x + 4*y + 1 = 0 → (1, -2) = (1, -2) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_l1923_192308


namespace NUMINAMATH_GPT_smallest_counterexample_is_14_l1923_192311

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_not_prime (n : ℕ) : Prop := ¬Prime n
def smallest_counterexample (n : ℕ) : Prop :=
  is_even n ∧ is_not_prime n ∧ is_not_prime (n + 2) ∧ ∀ m, is_even m ∧ is_not_prime m ∧ is_not_prime (m + 2) → n ≤ m

theorem smallest_counterexample_is_14 : smallest_counterexample 14 :=
by
  sorry

end NUMINAMATH_GPT_smallest_counterexample_is_14_l1923_192311


namespace NUMINAMATH_GPT_small_pump_filling_time_l1923_192356

theorem small_pump_filling_time :
  ∃ S : ℝ, (L = 2) → 
         (1 / 0.4444444444444444 = S + L) → 
         (1 / S = 4) :=
by 
  sorry

end NUMINAMATH_GPT_small_pump_filling_time_l1923_192356


namespace NUMINAMATH_GPT_find_x0_l1923_192366

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^2 + c
noncomputable def int_f (a c : ℝ) : ℝ := ∫ x in (0 : ℝ)..1, f x a c

theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (hx0 : 0 ≤ x0 ∧ x0 ≤ 1)
  (h_eq : int_f a c = f x0 a c) : x0 = Real.sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_find_x0_l1923_192366


namespace NUMINAMATH_GPT_penumbra_ring_area_l1923_192394

theorem penumbra_ring_area (r_umbra r_penumbra : ℝ) (h_ratio : r_umbra / r_penumbra = 2 / 6) (h_umbra : r_umbra = 40) :
  π * (r_penumbra ^ 2 - r_umbra ^ 2) = 12800 * π := by
  sorry

end NUMINAMATH_GPT_penumbra_ring_area_l1923_192394


namespace NUMINAMATH_GPT_expression_value_l1923_192352

theorem expression_value (x y : ℤ) (h1 : x = 2) (h2 : y = 5) : 
  (x^4 + 2 * y^2) / 6 = 11 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1923_192352
