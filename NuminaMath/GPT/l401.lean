import Mathlib

namespace NUMINAMATH_GPT_liters_to_pints_conversion_l401_40147

-- Definitions based on conditions
def liters_to_pints_ratio := 0.75 / 1.575
def target_liters := 1.5
def expected_pints := 3.15

-- Lean statement
theorem liters_to_pints_conversion 
  (h_ratio : 0.75 / 1.575 = liters_to_pints_ratio)
  (h_target : 1.5 = target_liters) :
  target_liters * (1 / liters_to_pints_ratio) = expected_pints :=
by 
  sorry

end NUMINAMATH_GPT_liters_to_pints_conversion_l401_40147


namespace NUMINAMATH_GPT_mans_rate_in_still_water_l401_40174

theorem mans_rate_in_still_water
  (V_m V_s : ℝ)
  (h_with_stream : V_m + V_s = 26)
  (h_against_stream : V_m - V_s = 4) :
  V_m = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_mans_rate_in_still_water_l401_40174


namespace NUMINAMATH_GPT_find_divisor_l401_40186

def dividend := 23
def quotient := 4
def remainder := 3

theorem find_divisor (d : ℕ) (h : dividend = (d * quotient) + remainder) : d = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_divisor_l401_40186


namespace NUMINAMATH_GPT_line_intersection_l401_40179

theorem line_intersection : 
  ∃ (x y : ℚ), 
    8 * x - 5 * y = 10 ∧ 
    3 * x + 2 * y = 16 ∧ 
    x = 100 / 31 ∧ 
    y = 98 / 31 :=
by
  use 100 / 31
  use 98 / 31
  sorry

end NUMINAMATH_GPT_line_intersection_l401_40179


namespace NUMINAMATH_GPT_curve_symmetric_reflection_l401_40102

theorem curve_symmetric_reflection (f : ℝ → ℝ → ℝ) :
  (∀ x y, f x y = 0 ↔ f (y + 3) (x - 3) = 0) → 
  (∀ x y, (x - y - 3 = 0) → (f (y + 3) (x - 3) = 0)) :=
sorry

end NUMINAMATH_GPT_curve_symmetric_reflection_l401_40102


namespace NUMINAMATH_GPT_larger_number_l401_40161

theorem larger_number (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) (hcf_eq : hcf = 23) (fact1_eq : factor1 = 13) (fact2_eq : factor2 = 14) : 
  max (hcf * factor1) (hcf * factor2) = 322 := 
by
  sorry

end NUMINAMATH_GPT_larger_number_l401_40161


namespace NUMINAMATH_GPT_correct_equation_l401_40162

-- Definitions based on conditions
def total_students := 98
def transfer_students := 3
def original_students_A (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ total_students
def students_B (x : ℕ) := total_students - x

-- Equation set up based on translation of the proof problem
theorem correct_equation (x : ℕ) (h : original_students_A x) :
  students_B x + transfer_students = x - transfer_students ↔ (98 - x) + 3 = x - 3 :=
by
  sorry
  
end NUMINAMATH_GPT_correct_equation_l401_40162


namespace NUMINAMATH_GPT_sum_of_tangents_l401_40160

noncomputable def g (x : ℝ) : ℝ :=
  max (max (-7 * x - 25) (2 * x + 5)) (5 * x - 7)

theorem sum_of_tangents (a b c : ℝ) (q : ℝ → ℝ) (hq₁ : ∀ x, q x = k * (x - a) ^ 2 + (-7 * x - 25))
  (hq₂ : ∀ x, q x = k * (x - b) ^ 2 + (2 * x + 5))
  (hq₃ : ∀ x, q x = k * (x - c) ^ 2 + (5 * x - 7)) :
  a + b + c = -34 / 3 := 
sorry

end NUMINAMATH_GPT_sum_of_tangents_l401_40160


namespace NUMINAMATH_GPT_eleven_billion_in_scientific_notation_l401_40176

-- Definition: "Billion" is 10^9
def billion : ℝ := 10^9

-- Theorem: 11 billion can be represented as 1.1 * 10^10
theorem eleven_billion_in_scientific_notation : 11 * billion = 1.1 * 10^10 := by
  sorry

end NUMINAMATH_GPT_eleven_billion_in_scientific_notation_l401_40176


namespace NUMINAMATH_GPT_find_x_when_y_equals_2_l401_40192

theorem find_x_when_y_equals_2 (x : ℚ) (y : ℚ) : 
  y = (1 / (4 * x + 2)) ∧ y = 2 -> x = -3 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_when_y_equals_2_l401_40192


namespace NUMINAMATH_GPT_sequence_non_positive_l401_40119

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h₀ : a 0 = 0) 
  (hₙ : a n = 0)
  (h : ∀ k, 1 ≤ k ∧ k < n → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) : 
  ∀ k, k ≤ n → a k ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_sequence_non_positive_l401_40119


namespace NUMINAMATH_GPT_length_of_first_train_l401_40199

theorem length_of_first_train 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (length_second_train_m : ℝ) 
  (hspeed_first : speed_first_train_kmph = 120) 
  (hspeed_second : speed_second_train_kmph = 80) 
  (htime : crossing_time_s = 9) 
  (hlength_second : length_second_train_m = 320.04) :
  ∃ (length_first_train_m : ℝ), abs (length_first_train_m - 180) < 0.1 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_train_l401_40199


namespace NUMINAMATH_GPT_ab_cd_value_l401_40111

theorem ab_cd_value (a b c d: ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 5)
  (h3 : a + c + d = 14)
  (h4 : b + c + d = 9) :
  a * b + c * d = 338 / 9 := 
sorry

end NUMINAMATH_GPT_ab_cd_value_l401_40111


namespace NUMINAMATH_GPT_corrected_mean_l401_40169

theorem corrected_mean (n : ℕ) (mean old_obs new_obs : ℝ) 
    (obs_count : n = 50) (old_mean : mean = 36) (incorrect_obs : old_obs = 23) (correct_obs : new_obs = 46) :
    (mean * n - old_obs + new_obs) / n = 36.46 := by
  sorry

end NUMINAMATH_GPT_corrected_mean_l401_40169


namespace NUMINAMATH_GPT_willy_crayons_eq_l401_40107

def lucy_crayons : ℕ := 3971
def more_crayons : ℕ := 1121

theorem willy_crayons_eq : 
  ∀ willy_crayons : ℕ, willy_crayons = lucy_crayons + more_crayons → willy_crayons = 5092 :=
by
  sorry

end NUMINAMATH_GPT_willy_crayons_eq_l401_40107


namespace NUMINAMATH_GPT_central_angle_of_sector_l401_40154

-- Given conditions as hypotheses
variable (r θ : ℝ)
variable (h₁ : (1/2) * θ * r^2 = 1)
variable (h₂ : 2 * r + θ * r = 4)

-- The goal statement to be proved
theorem central_angle_of_sector :
  θ = 2 :=
by sorry

end NUMINAMATH_GPT_central_angle_of_sector_l401_40154


namespace NUMINAMATH_GPT_smallest_number_divisible_by_11_and_remainder_1_l401_40130

theorem smallest_number_divisible_by_11_and_remainder_1 {n : ℕ} :
  (n % 2 = 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 11 = 0) -> n = 121 :=
sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_11_and_remainder_1_l401_40130


namespace NUMINAMATH_GPT_probability_blue_tile_l401_40178

def is_congruent_to_3_mod_7 (n : ℕ) : Prop := n % 7 = 3

def num_blue_tiles (n : ℕ) : ℕ := (n / 7) + 1

theorem probability_blue_tile : 
  num_blue_tiles 70 / 70 = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_blue_tile_l401_40178


namespace NUMINAMATH_GPT_value_of_a_minus_b_l401_40170

theorem value_of_a_minus_b (a b : ℚ) (h1 : 3015 * a + 3021 * b = 3025) (h2 : 3017 * a + 3023 * b = 3027) : 
  a - b = - (7 / 3) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l401_40170


namespace NUMINAMATH_GPT_extended_pattern_ratio_l401_40173

def original_black_tiles : ℕ := 13
def original_white_tiles : ℕ := 12
def original_total_tiles : ℕ := 5 * 5

def new_side_length : ℕ := 7
def new_total_tiles : ℕ := new_side_length * new_side_length
def added_white_tiles : ℕ := new_total_tiles - original_total_tiles

def new_black_tiles : ℕ := original_black_tiles
def new_white_tiles : ℕ := original_white_tiles + added_white_tiles

def ratio_black_to_white : ℚ := new_black_tiles / new_white_tiles

theorem extended_pattern_ratio :
  ratio_black_to_white = 13 / 36 :=
by
  sorry

end NUMINAMATH_GPT_extended_pattern_ratio_l401_40173


namespace NUMINAMATH_GPT_chord_probability_concentric_circles_l401_40150

noncomputable def chord_intersects_inner_circle_probability : ℝ :=
  sorry

theorem chord_probability_concentric_circles :
  let r₁ := 2
  let r₂ := 3
  ∀ (P₁ P₂ : ℝ × ℝ),
    dist P₁ (0, 0) = r₂ ∧ dist P₂ (0, 0) = r₂ →
    chord_intersects_inner_circle_probability = 0.148 :=
  sorry

end NUMINAMATH_GPT_chord_probability_concentric_circles_l401_40150


namespace NUMINAMATH_GPT_sum_of_fractions_l401_40113

theorem sum_of_fractions (a b c d : ℚ) (ha : a = 2 / 5) (hb : b = 3 / 8) :
  (a + b = 31 / 40) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l401_40113


namespace NUMINAMATH_GPT_ratio_part_to_third_fraction_l401_40167

variable (P N : ℕ)

-- Definitions based on conditions
def one_fourth_one_third_P_eq_14 : Prop := (1/4 : ℚ) * (1/3 : ℚ) * (P : ℚ) = 14

def forty_percent_N_eq_168 : Prop := (40/100 : ℚ) * (N : ℚ) = 168

-- Theorem stating the required ratio
theorem ratio_part_to_third_fraction (h1 : one_fourth_one_third_P_eq_14 P) (h2 : forty_percent_N_eq_168 N) : 
  (P : ℚ) / ((1/3 : ℚ) * (N : ℚ)) = 6 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_part_to_third_fraction_l401_40167


namespace NUMINAMATH_GPT_total_files_on_flash_drive_l401_40183

theorem total_files_on_flash_drive :
  ∀ (music_files video_files picture_files : ℝ),
    music_files = 4.0 ∧ video_files = 21.0 ∧ picture_files = 23.0 →
    music_files + video_files + picture_files = 48.0 :=
by
  sorry

end NUMINAMATH_GPT_total_files_on_flash_drive_l401_40183


namespace NUMINAMATH_GPT_probability_square_or_triangle_l401_40188

theorem probability_square_or_triangle :
  let total_figures := 10
  let number_of_triangles := 4
  let number_of_squares := 3
  let number_of_favorable_outcomes := number_of_triangles + number_of_squares
  let probability := number_of_favorable_outcomes / total_figures
  probability = 7 / 10 :=
sorry

end NUMINAMATH_GPT_probability_square_or_triangle_l401_40188


namespace NUMINAMATH_GPT_ratio_of_girls_more_than_boys_l401_40121

theorem ratio_of_girls_more_than_boys 
  (B : ℕ := 50) 
  (P : ℕ := 123) 
  (driver_assistant_teacher := 3) 
  (h : P = driver_assistant_teacher + B + (P - driver_assistant_teacher - B)) : 
  (P - driver_assistant_teacher - B) - B = 21 → 
  (P - driver_assistant_teacher - B) % B = 21 / 50 := 
sorry

end NUMINAMATH_GPT_ratio_of_girls_more_than_boys_l401_40121


namespace NUMINAMATH_GPT_trigonometric_identity_l401_40181

theorem trigonometric_identity 
  (α β γ : ℝ)
  (h : (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ)) :
  (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
  abs (Real.cos α * Real.cos β * Real.cos γ) ∧
  (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ) = 
  abs (Real.cos α * Real.cos β * Real.cos γ) := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l401_40181


namespace NUMINAMATH_GPT_graph_of_equation_l401_40137

theorem graph_of_equation (x y : ℝ) :
  x^3 * (x + y + 2) = y^3 * (x + y + 2) →
  (x + y + 2 ≠ 0 ∧ (x = y ∨ x^2 + x * y + y^2 = 0)) ∨
  (x + y + 2 = 0 ∧ y = -x - 2) →
  (y = x ∨ y = -x - 2) := 
sorry

end NUMINAMATH_GPT_graph_of_equation_l401_40137


namespace NUMINAMATH_GPT_favorite_food_sandwiches_l401_40156

theorem favorite_food_sandwiches (total_students : ℕ) (cookies_percent pizza_percent pasta_percent : ℝ)
  (h_total : total_students = 200)
  (h_cookies : cookies_percent = 0.25)
  (h_pizza : pizza_percent = 0.30)
  (h_pasta : pasta_percent = 0.35) :
  let sandwiches_percent := 1 - (cookies_percent + pizza_percent + pasta_percent)
  sandwiches_percent * total_students = 20 :=
by
  sorry

end NUMINAMATH_GPT_favorite_food_sandwiches_l401_40156


namespace NUMINAMATH_GPT_intersection_empty_l401_40168

open Set

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_GPT_intersection_empty_l401_40168


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l401_40120

theorem simplify_and_evaluate_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ( (2 * x - 3) / (x - 2) - 1 ) / ( (x^2 - 2 * x + 1) / (x - 2) ) = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l401_40120


namespace NUMINAMATH_GPT_ferry_heading_to_cross_perpendicularly_l401_40118

theorem ferry_heading_to_cross_perpendicularly (river_speed ferry_speed : ℝ) (river_speed_val : river_speed = 12.5) (ferry_speed_val : ferry_speed = 25) : 
  angle_to_cross = 30 :=
by
  -- Definitions for the problem
  let river_velocity : ℝ := river_speed
  let ferry_velocity : ℝ := ferry_speed
  have river_velocity_def : river_velocity = 12.5 := river_speed_val
  have ferry_velocity_def : ferry_velocity = 25 := ferry_speed_val
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_ferry_heading_to_cross_perpendicularly_l401_40118


namespace NUMINAMATH_GPT_parabola_and_hyperbola_equation_l401_40132

theorem parabola_and_hyperbola_equation (a b c : ℝ)
    (ha : a > 0)
    (hb : b > 0)
    (hp_eq : c = 2)
    (intersect : (3 / 2, Real.sqrt 6) ∈ {p : ℝ × ℝ | p.2^2 = 4 * c * p.1}
                ∧ (3 / 2, Real.sqrt 6) ∈ {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) - (p.2 ^ 2 / b ^ 2) = 1}) :
    (∀ x y : ℝ, y^2 = 4*x ↔ c = 1)
    ∧ (∃ a', a' = 1 / 2 ∧ ∀ x y : ℝ, 4 * x^2 - (4 * y^2) / 3 = 1 ↔ a = a') := 
by 
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_parabola_and_hyperbola_equation_l401_40132


namespace NUMINAMATH_GPT_cos_identity_l401_40126

theorem cos_identity (x : ℝ) : 
  4 * Real.cos x * Real.cos (x + π / 3) * Real.cos (x - π / 3) = Real.cos (3 * x) :=
by
  sorry

end NUMINAMATH_GPT_cos_identity_l401_40126


namespace NUMINAMATH_GPT_evaluate_expression_l401_40166

theorem evaluate_expression : (24 : ℕ) = 2^3 * 3 ∧ (72 : ℕ) = 2^3 * 3^2 → (24^40 / 72^20 : ℚ) = 2^60 :=
by {
  sorry
}

end NUMINAMATH_GPT_evaluate_expression_l401_40166


namespace NUMINAMATH_GPT_not_perfect_square_l401_40133

theorem not_perfect_square (x y : ℤ) : ¬ ∃ k : ℤ, k^2 = (x^2 + x + 1)^2 + (y^2 + y + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_l401_40133


namespace NUMINAMATH_GPT_range_of_m_l401_40155

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → x < m) : m > 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l401_40155


namespace NUMINAMATH_GPT_equivalent_set_complement_intersection_l401_40117

def setM : Set ℝ := {x | -3 < x ∧ x < 1}
def setN : Set ℝ := {x | x ≤ 3}
def givenSet : Set ℝ := {x | x ≤ -3 ∨ x ≥ 1}

theorem equivalent_set_complement_intersection :
  givenSet = (setM ∩ setN)ᶜ :=
sorry

end NUMINAMATH_GPT_equivalent_set_complement_intersection_l401_40117


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l401_40142

theorem simplify_sqrt_expression (x : ℝ) : 
  Real.sqrt (x^6 + x^4 + 1) = Real.sqrt (x^6 + x^4 + 1) := by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l401_40142


namespace NUMINAMATH_GPT_percent_absent_l401_40139

-- Given conditions
def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def absent_boys_fraction : ℚ := 1 / 8
def absent_girls_fraction : ℚ := 1 / 4

-- Theorem to prove
theorem percent_absent : 100 * ((absent_boys_fraction * boys + absent_girls_fraction * girls) / total_students) = 17.5 := 
sorry

end NUMINAMATH_GPT_percent_absent_l401_40139


namespace NUMINAMATH_GPT_compute_expression_l401_40128

theorem compute_expression (x : ℝ) (h : x = 7) : (x^6 - 36*x^3 + 324) / (x^3 - 18) = 325 := 
by
  sorry

end NUMINAMATH_GPT_compute_expression_l401_40128


namespace NUMINAMATH_GPT_inscribed_circle_diameter_l401_40152

noncomputable def diameter_inscribed_circle (side_length : ℝ) : ℝ :=
  let s := (3 * side_length) / 2
  let K := (Real.sqrt 3 / 4) * (side_length ^ 2)
  let r := K / s
  2 * r

theorem inscribed_circle_diameter (side_length : ℝ) (h : side_length = 10) :
  diameter_inscribed_circle side_length = (10 * Real.sqrt 3) / 3 :=
by
  rw [h]
  simp [diameter_inscribed_circle]
  sorry

end NUMINAMATH_GPT_inscribed_circle_diameter_l401_40152


namespace NUMINAMATH_GPT_complex_product_polar_form_l401_40100

theorem complex_product_polar_form :
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 360 ∧ 
  (r = 12 ∧ θ = 245) :=
by
  sorry

end NUMINAMATH_GPT_complex_product_polar_form_l401_40100


namespace NUMINAMATH_GPT_paper_clips_distribution_l401_40131

theorem paper_clips_distribution (P c b : ℕ) (hP : P = 81) (hc : c = 9) (hb : b = P / c) : b = 9 :=
by
  rw [hP, hc] at hb
  simp at hb
  exact hb

end NUMINAMATH_GPT_paper_clips_distribution_l401_40131


namespace NUMINAMATH_GPT_arithmetic_progression_primes_l401_40125

theorem arithmetic_progression_primes (p₁ p₂ p₃ : ℕ) (d : ℕ) 
  (hp₁ : Prime p₁) (hp₁_cond : 3 < p₁) 
  (hp₂ : Prime p₂) (hp₂_cond : 3 < p₂) 
  (hp₃ : Prime p₃) (hp₃_cond : 3 < p₃) 
  (h_prog_1 : p₂ = p₁ + d) (h_prog_2 : p₃ = p₁ + 2 * d) : 
  d % 6 = 0 :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_primes_l401_40125


namespace NUMINAMATH_GPT_difference_of_squares_not_2018_l401_40138

theorem difference_of_squares_not_2018 (a b : ℕ) : a^2 - b^2 ≠ 2018 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_not_2018_l401_40138


namespace NUMINAMATH_GPT_total_enemies_l401_40134

theorem total_enemies (points_per_enemy defeated_enemies undefeated_enemies total_points total_enemies : ℕ)
  (h1 : points_per_enemy = 5) 
  (h2 : undefeated_enemies = 6) 
  (h3 : total_points = 10) :
  total_enemies = 8 := by
  sorry

end NUMINAMATH_GPT_total_enemies_l401_40134


namespace NUMINAMATH_GPT_recursive_relation_a_recursive_relation_b_recursive_relation_c_recursive_relation_d_recursive_relation_e_l401_40149

def a (n : ℕ) : ℕ := n
def b (n : ℕ) : ℕ := n^2
def c (n : ℕ) : ℕ := n^3
def d (n : ℕ) : ℕ := n^4
def e (n : ℕ) : ℕ := n^5

theorem recursive_relation_a (n : ℕ) : a (n+2) = 2 * a (n+1) - a n :=
by sorry

theorem recursive_relation_b (n : ℕ) : b (n+3) = 3 * b (n+2) - 3 * b (n+1) + b n :=
by sorry

theorem recursive_relation_c (n : ℕ) : c (n+4) = 4 * c (n+3) - 6 * c (n+2) + 4 * c (n+1) - c n :=
by sorry

theorem recursive_relation_d (n : ℕ) : d (n+5) = 5 * d (n+4) - 10 * d (n+3) + 10 * d (n+2) - 5 * d (n+1) + d n :=
by sorry

theorem recursive_relation_e (n : ℕ) : 
  e (n+6) = 6 * e (n+5) - 15 * e (n+4) + 20 * e (n+3) - 15 * e (n+2) + 6 * e (n+1) - e n :=
by sorry

end NUMINAMATH_GPT_recursive_relation_a_recursive_relation_b_recursive_relation_c_recursive_relation_d_recursive_relation_e_l401_40149


namespace NUMINAMATH_GPT_area_overlap_of_triangles_l401_40158

structure Point where
  x : ℝ
  y : ℝ

def Triangle (p1 p2 p3 : Point) : Set Point :=
  { q | ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ (a * p1.x + b * p2.x + c * p3.x = q.x) ∧ (a * p1.y + b * p2.y + c * p3.y = q.y) }

def area_of_overlap (t1 t2 : Set Point) : ℝ :=
  -- Assume we have a function that calculates the overlap area
  sorry

def point1 : Point := ⟨0, 2⟩
def point2 : Point := ⟨2, 1⟩
def point3 : Point := ⟨0, 0⟩
def point4 : Point := ⟨2, 2⟩
def point5 : Point := ⟨0, 1⟩
def point6 : Point := ⟨2, 0⟩

def triangle1 : Set Point := Triangle point1 point2 point3
def triangle2 : Set Point := Triangle point4 point5 point6

theorem area_overlap_of_triangles :
  area_of_overlap triangle1 triangle2 = 1 :=
by
  -- Proof goes here, replacing sorry with actual proof steps
  sorry

end NUMINAMATH_GPT_area_overlap_of_triangles_l401_40158


namespace NUMINAMATH_GPT_cn_geometric_seq_l401_40197

-- Given conditions
def Sn (n : ℕ) : ℚ := (3 * n^2 + 5 * n) / 2
def an (n : ℕ) : ℕ := 3 * n + 1
def bn (n : ℕ) : ℕ := 2^n

theorem cn_geometric_seq : 
  ∃ q : ℕ, ∃ (c : ℕ → ℕ), (∀ n : ℕ, c n = q^n) ∧ (∀ n : ℕ, ∃ m : ℕ, c n = an m ∧ c n = bn m) :=
sorry

end NUMINAMATH_GPT_cn_geometric_seq_l401_40197


namespace NUMINAMATH_GPT_total_earnings_l401_40136

theorem total_earnings : 
  let wage : ℕ := 10
  let hours_monday : ℕ := 7
  let tips_monday : ℕ := 18
  let hours_tuesday : ℕ := 5
  let tips_tuesday : ℕ := 12
  let hours_wednesday : ℕ := 7
  let tips_wednesday : ℕ := 20
  let total_hours : ℕ := hours_monday + hours_tuesday + hours_wednesday
  let earnings_from_wage : ℕ := total_hours * wage
  let total_tips : ℕ := tips_monday + tips_tuesday + tips_wednesday
  let total_earnings : ℕ := earnings_from_wage + total_tips
  total_earnings = 240 :=
by
  sorry

end NUMINAMATH_GPT_total_earnings_l401_40136


namespace NUMINAMATH_GPT_max_marks_l401_40198

variable (M : ℝ)

def passing_marks (M : ℝ) : ℝ := 0.45 * M

theorem max_marks (h1 : passing_marks M = 225)
  (h2 : 180 + 45 = 225) : M = 500 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l401_40198


namespace NUMINAMATH_GPT_day_crew_fraction_correct_l401_40104

-- Given conditions
variables (D W : ℕ)
def night_boxes_per_worker := (5 : ℚ) / 8 * D
def night_workers := (3 : ℚ) / 5 * W

-- Total boxes loaded
def total_day_boxes := D * W
def total_night_boxes := night_boxes_per_worker D * night_workers W

-- Fraction of boxes loaded by day crew
def fraction_loaded_by_day_crew := total_day_boxes D W / (total_day_boxes D W + total_night_boxes D W)

-- Theorem to prove
theorem day_crew_fraction_correct (D W : ℕ) : fraction_loaded_by_day_crew D W = (8 : ℚ) / 11 :=
by
  sorry

end NUMINAMATH_GPT_day_crew_fraction_correct_l401_40104


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l401_40115

-- Define the arithmetic sequence
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a_1 + (n - 1) * d

-- Define the sum of the first k terms of an arithmetic sequence
def sum_arithmetic_seq (a_1 : ℤ) (d : ℤ) (k : ℤ) : ℤ :=
  (k * (2 * a_1 + (k - 1) * d)) / 2

-- Prove that d > 0 is a necessary and sufficient condition for S_3n - S_2n > S_2n - S_n
/-- Necessary and sufficient condition for the inequality S_{3n} - S_{2n} > S_{2n} - S_n -/
theorem necessary_and_sufficient_condition {a_1 d n : ℤ} :
  d > 0 ↔ sum_arithmetic_seq a_1 d (3 * n) - sum_arithmetic_seq a_1 d (2 * n) > 
             sum_arithmetic_seq a_1 d (2 * n) - sum_arithmetic_seq a_1 d n :=
by sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l401_40115


namespace NUMINAMATH_GPT_find_expression_value_l401_40163

-- Given conditions
variables {a b : ℝ}

-- Perimeter condition
def perimeter_condition (a b : ℝ) : Prop := 2 * (a + b) = 10

-- Area condition
def area_condition (a b : ℝ) : Prop := a * b = 6

-- Goal statement
theorem find_expression_value (h1 : perimeter_condition a b) (h2 : area_condition a b) :
  a^3 * b + 2 * a^2 * b^2 + a * b^3 = 150 :=
sorry

end NUMINAMATH_GPT_find_expression_value_l401_40163


namespace NUMINAMATH_GPT_radius_of_largest_circle_correct_l401_40148

noncomputable def radius_of_largest_circle_in_quadrilateral (AB BC CD DA : ℝ) (angle_BCD : ℝ) : ℝ :=
  if AB = 10 ∧ BC = 12 ∧ CD = 8 ∧ DA = 14 ∧ angle_BCD = 90
    then Real.sqrt 210
    else 0

theorem radius_of_largest_circle_correct :
  radius_of_largest_circle_in_quadrilateral 10 12 8 14 90 = Real.sqrt 210 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_largest_circle_correct_l401_40148


namespace NUMINAMATH_GPT_tens_digit_36_pow_12_l401_40185

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def tens_digit (n : ℕ) : ℕ :=
  (last_two_digits n) / 10

theorem tens_digit_36_pow_12 : tens_digit (36^12) = 3 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_36_pow_12_l401_40185


namespace NUMINAMATH_GPT_infinite_sum_evaluation_l401_40195

theorem infinite_sum_evaluation :
  (∑' n : ℕ, (n : ℚ) / ((n^2 - 2 * n + 2) * (n^2 + 2 * n + 4))) = 5 / 24 :=
sorry

end NUMINAMATH_GPT_infinite_sum_evaluation_l401_40195


namespace NUMINAMATH_GPT_phoebe_age_l401_40172

theorem phoebe_age (P : ℕ) (h₁ : ∀ P, 60 = 4 * (P + 5)) (h₂: 55 + 5 = 60) : P = 10 := 
by
  have h₃ : 60 = 4 * (P + 5) := h₁ P
  sorry

end NUMINAMATH_GPT_phoebe_age_l401_40172


namespace NUMINAMATH_GPT_sector_area_eq_25_l401_40164

theorem sector_area_eq_25 (r θ : ℝ) (h_r : r = 5) (h_θ : θ = 2) : (1 / 2) * θ * r^2 = 25 := by
  sorry

end NUMINAMATH_GPT_sector_area_eq_25_l401_40164


namespace NUMINAMATH_GPT_original_denominator_is_two_l401_40105

theorem original_denominator_is_two (d : ℕ) : 
  (∃ d : ℕ, 2 * (d + 4) = 6) → d = 2 :=
by sorry

end NUMINAMATH_GPT_original_denominator_is_two_l401_40105


namespace NUMINAMATH_GPT_value_of_y_at_x_8_l401_40157

theorem value_of_y_at_x_8 (k : ℝ) (x y : ℝ) 
  (hx1 : y = k * x^(1/3)) 
  (hx2 : y = 4 * Real.sqrt 3) 
  (hx3 : x = 64) 
  (hx4 : 8^(1/3) = 2) : 
  (y = 2 * Real.sqrt 3) := 
by 
  sorry

end NUMINAMATH_GPT_value_of_y_at_x_8_l401_40157


namespace NUMINAMATH_GPT_sue_driving_days_l401_40108

-- Define the conditions as constants or variables
def total_cost : ℕ := 2100
def sue_payment : ℕ := 900
def sister_days : ℕ := 4
def total_days_in_week : ℕ := 7

-- Prove that the number of days Sue drives the car (x) equals 3
theorem sue_driving_days : ∃ x : ℕ, x = 3 ∧ sue_payment * sister_days = x * (total_cost - sue_payment) := 
by
  sorry

end NUMINAMATH_GPT_sue_driving_days_l401_40108


namespace NUMINAMATH_GPT_mean_score_of_students_who_failed_l401_40123

noncomputable def mean_failed_score : ℝ := sorry

theorem mean_score_of_students_who_failed (t p proportion_passed proportion_failed : ℝ) (h1 : t = 6) (h2 : p = 8) (h3 : proportion_passed = 0.6) (h4 : proportion_failed = 0.4) : mean_failed_score = 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_score_of_students_who_failed_l401_40123


namespace NUMINAMATH_GPT_parallelogram_height_l401_40114

/-- The cost of leveling a field in the form of a parallelogram is Rs. 50 per 10 sq. meter, 
    with the base being 54 m and a certain perpendicular distance from the other side. 
    The total cost is Rs. 6480. What is the perpendicular distance from the other side 
    of the parallelogram? -/
theorem parallelogram_height
  (cost_per_10_sq_meter : ℝ)
  (base_length : ℝ)
  (total_cost : ℝ)
  (height : ℝ)
  (h1 : cost_per_10_sq_meter = 50)
  (h2 : base_length = 54)
  (h3 : total_cost = 6480)
  (area : ℝ)
  (h4 : area = (total_cost / cost_per_10_sq_meter) * 10)
  (h5 : area = base_length * height) :
  height = 24 :=
by { sorry }

end NUMINAMATH_GPT_parallelogram_height_l401_40114


namespace NUMINAMATH_GPT_quadratic_function_inequality_l401_40124

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem quadratic_function_inequality (a b c : ℝ) (h_a : a > 0) (h_symm : ∀ x : ℝ, quadratic_function a b c x = quadratic_function a b c (2 - x)) :
  ∀ x : ℝ, quadratic_function a b c (2 ^ x) < quadratic_function a b c (3 ^ x) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_inequality_l401_40124


namespace NUMINAMATH_GPT_max_consecutive_integers_sum_lt_1000_l401_40109

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end NUMINAMATH_GPT_max_consecutive_integers_sum_lt_1000_l401_40109


namespace NUMINAMATH_GPT_paigeRatio_l401_40151

/-- The total number of pieces in the chocolate bar -/
def totalPieces : ℕ := 60

/-- Michael takes half of the chocolate bar -/
def michaelPieces : ℕ := totalPieces / 2

/-- Mandy gets a fixed number of pieces -/
def mandyPieces : ℕ := 15

/-- The number of pieces left after Michael takes his share -/
def remainingPiecesAfterMichael : ℕ := totalPieces - michaelPieces

/-- The number of pieces Paige takes -/
def paigePieces : ℕ := remainingPiecesAfterMichael - mandyPieces

/-- The ratio of the number of pieces Paige takes to the number of pieces left after Michael takes his share is 1:2 -/
theorem paigeRatio :
  paigePieces / (remainingPiecesAfterMichael / 15) = 1 := sorry

end NUMINAMATH_GPT_paigeRatio_l401_40151


namespace NUMINAMATH_GPT_food_price_before_tax_and_tip_l401_40103

theorem food_price_before_tax_and_tip (total_paid : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (P : ℝ) (h1 : total_paid = 198) (h2 : tax_rate = 0.10) (h3 : tip_rate = 0.20) : 
  P = 150 :=
by
  -- Given that total_paid = 198, tax_rate = 0.10, tip_rate = 0.20,
  -- we should show that the actual price of the food before tax
  -- and tip is $150.
  sorry

end NUMINAMATH_GPT_food_price_before_tax_and_tip_l401_40103


namespace NUMINAMATH_GPT_number_of_white_balls_l401_40180

theorem number_of_white_balls (r w : ℕ) (h_r : r = 8) (h_prob : (r : ℚ) / (r + w) = 2 / 5) : w = 12 :=
by sorry

end NUMINAMATH_GPT_number_of_white_balls_l401_40180


namespace NUMINAMATH_GPT_second_round_score_l401_40112

/-- 
  Given the scores in three rounds of darts, where the second round score is twice the
  first round score, and the third round score is 1.5 times the second round score,
  prove that the score in the second round is 48, given that the maximum score in the 
  third round is 72.
-/
theorem second_round_score (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 1.5 * y) (h3 : z = 72) : y = 48 :=
sorry

end NUMINAMATH_GPT_second_round_score_l401_40112


namespace NUMINAMATH_GPT_problem_statement_l401_40187

open Real

theorem problem_statement (t : ℝ) :
  cos (2 * t) ≠ 0 ∧ sin (2 * t) ≠ 0 →
  cos⁻¹ (2 * t) + sin⁻¹ (2 * t) + cos⁻¹ (2 * t) * sin⁻¹ (2 * t) = 5 →
  (∃ k : ℤ, t = arctan (1/2) + π * k) ∨ (∃ n : ℤ, t = arctan (1/3) + π * n) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l401_40187


namespace NUMINAMATH_GPT_phraseCompletion_l401_40143

-- Define the condition for the problem
def isCorrectPhrase (phrase : String) : Prop :=
  phrase = "crying"

-- State the theorem to be proven
theorem phraseCompletion : ∃ phrase, isCorrectPhrase phrase :=
by
  use "crying"
  sorry

end NUMINAMATH_GPT_phraseCompletion_l401_40143


namespace NUMINAMATH_GPT_sum_of_cubes_l401_40184

theorem sum_of_cubes (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x * y + y * z + z * x = -3) (h3 : x * y * z = 2) : 
  x^3 + y^3 + z^3 = 32 := 
sorry

end NUMINAMATH_GPT_sum_of_cubes_l401_40184


namespace NUMINAMATH_GPT_Tim_weekly_earnings_l401_40196

theorem Tim_weekly_earnings :
  let tasks_per_day := 100
  let pay_per_task := 1.2
  let days_per_week := 6
  let daily_earnings := tasks_per_day * pay_per_task
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 720 := by
  sorry

end NUMINAMATH_GPT_Tim_weekly_earnings_l401_40196


namespace NUMINAMATH_GPT_p_neither_necessary_nor_sufficient_l401_40141

def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x : ℝ) : Prop := x ≠ 0
def r (y : ℝ) : Prop := y ≠ -1

theorem p_neither_necessary_nor_sufficient (x y : ℝ) (h1: p x y) (h2: q x) (h3: r y) :
  ¬(p x y → q x) ∧ ¬(q x → p x y) := 
by 
  sorry

end NUMINAMATH_GPT_p_neither_necessary_nor_sufficient_l401_40141


namespace NUMINAMATH_GPT_rectangle_problem_l401_40194

noncomputable def calculate_width (L P : ℕ) : ℕ :=
  (P - 2 * L) / 2

theorem rectangle_problem :
  ∀ (L P : ℕ), L = 12 → P = 36 → (calculate_width L P = 6) ∧ ((calculate_width L P) / L = 1 / 2) :=
by
  intros L P hL hP
  have hw : calculate_width L P = 6 := by
    sorry
  have hr : ((calculate_width L P) / L) = 1 / 2 := by
    sorry
  exact ⟨hw, hr⟩

end NUMINAMATH_GPT_rectangle_problem_l401_40194


namespace NUMINAMATH_GPT_alt_fib_factorial_seq_last_two_digits_eq_85_l401_40106

noncomputable def alt_fib_factorial_seq_last_two_digits : ℕ :=
  let f0 := 1   -- 0!
  let f1 := 1   -- 1!
  let f2 := 2   -- 2!
  let f3 := 6   -- 3!
  let f5 := 120 -- 5! (last two digits 20)
  (f0 - f1 + f1 - f2 + f3 - (f5 % 100)) % 100

theorem alt_fib_factorial_seq_last_two_digits_eq_85 :
  alt_fib_factorial_seq_last_two_digits = 85 :=
by 
  sorry

end NUMINAMATH_GPT_alt_fib_factorial_seq_last_two_digits_eq_85_l401_40106


namespace NUMINAMATH_GPT_intersection_A_complementB_l401_40110

universe u

def R : Type := ℝ

def A (x : ℝ) : Prop := 0 < x ∧ x < 2

def B (x : ℝ) : Prop := x ≥ 1

def complement_B (x : ℝ) : Prop := x < 1

theorem intersection_A_complementB : 
  ∀ x : ℝ, (A x ∧ complement_B x) ↔ (0 < x ∧ x < 1) := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_complementB_l401_40110


namespace NUMINAMATH_GPT_solve_for_x_l401_40193

theorem solve_for_x (x y : ℕ) (h1 : x / y = 15 / 5) (h2 : y = 25) : x = 75 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l401_40193


namespace NUMINAMATH_GPT_rectangle_area_stage_8_l401_40101

def square_side_length : ℕ := 4
def stage_count : ℕ := 8

-- The function to compute the area of one square
def square_area (side_length: ℕ) : ℕ :=
  side_length * side_length

-- The function to compute the total area at a given stage
def total_area_at_stage (side_length: ℕ) (stages: ℕ) : ℕ :=
  stages * (square_area side_length)

theorem rectangle_area_stage_8 :
  total_area_at_stage square_side_length stage_count = 128 :=
  by
    sorry

end NUMINAMATH_GPT_rectangle_area_stage_8_l401_40101


namespace NUMINAMATH_GPT_CarlosAndDianaReceivedAs_l401_40122

variables (Alan Beth Carlos Diana : Prop)
variable (num_A : ℕ)

-- Condition 1: Alan => Beth
axiom AlanImpliesBeth : Alan → Beth

-- Condition 2: Beth => Carlos
axiom BethImpliesCarlos : Beth → Carlos

-- Condition 3: Carlos => Diana
axiom CarlosImpliesDiana : Carlos → Diana

-- Condition 4: Only two students received an A
axiom OnlyTwoReceivedAs : num_A = 2

-- Theorem: Carlos and Diana received A's
theorem CarlosAndDianaReceivedAs : ((Alan ∧ Beth ∧ Carlos ∧ Diana → False) ∧
                                   (Beth ∧ Carlos ∧ Diana → False) ∧
                                   (Alan ∧ Beth ∧ Diana → False) ∧
                                   (Alan ∧ Beth ∧ Carlos → False) ∧
                                   (Alan ∧ Diana → False) ∧
                                   (Beth ∧ Carlos → False) ∧
                                   (Alan ∧ Carlos → False) ∧
                                   (Beth ∧ Diana → False)) → (Carlos ∧ Diana) :=
by
  intros h
  have h1 := AlanImpliesBeth
  have h2 := BethImpliesCarlos
  have h3 := CarlosImpliesDiana
  have h4 := OnlyTwoReceivedAs
  sorry

end NUMINAMATH_GPT_CarlosAndDianaReceivedAs_l401_40122


namespace NUMINAMATH_GPT_wire_ratio_l401_40116

theorem wire_ratio (a b : ℝ) (h_eq_area : (a / 4)^2 = 2 * (b / 8)^2 * (1 + Real.sqrt 2)) :
  a / b = Real.sqrt (2 + Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_wire_ratio_l401_40116


namespace NUMINAMATH_GPT_quadratic_inequality_l401_40171

theorem quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + 4 > 0) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l401_40171


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l401_40189

theorem no_real_roots_of_quadratic 
(a b c : ℝ) 
(h1 : b + c > a)
(h2 : b + a > c)
(h3 : c + a > b) :
(b^2 + c^2 - a^2)^2 - 4 * b^2 * c^2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l401_40189


namespace NUMINAMATH_GPT_solve_inequality_l401_40146

theorem solve_inequality (x : ℝ) : (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
  sorry

end NUMINAMATH_GPT_solve_inequality_l401_40146


namespace NUMINAMATH_GPT_candles_time_l401_40135

/-- Prove that if two candles of equal length are lit at a certain time,
and by 6 PM one of the stubs is three times the length of the other,
the correct time to light the candles is 4:00 PM. -/

theorem candles_time :
  ∀ (ℓ : ℝ) (t : ℝ),
  (∀ t1 t2 : ℝ, t = t1 + t2 → 
    (180 - t1) = 3 * (300 - t2) / 3 → 
    18 <= 6 ∧ 0 <= t → ℓ / 180 * (180 - (t - 180)) = 3 * (ℓ / 300 * (300 - (6 - t))) →
    t = 4
  ) := 
by 
  sorry

end NUMINAMATH_GPT_candles_time_l401_40135


namespace NUMINAMATH_GPT_barycentric_identity_l401_40129

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def barycentric (α β γ : ℝ) (a b c : V) : V := 
  α • a + β • b + γ • c

theorem barycentric_identity 
  (A B C X : V) 
  (α β γ : ℝ)
  (h : α + β + γ = 1)
  (hXA : X = barycentric α β γ A B C) :
  X - A = β • (B - A) + γ • (C - A) :=
by
  sorry

end NUMINAMATH_GPT_barycentric_identity_l401_40129


namespace NUMINAMATH_GPT_area_sin_transformed_l401_40175

noncomputable def sin_transformed (x : ℝ) : ℝ := 4 * Real.sin (x - Real.pi)

theorem area_sin_transformed :
  ∫ x in Real.pi..3 * Real.pi, |sin_transformed x| = 16 :=
by
  sorry

end NUMINAMATH_GPT_area_sin_transformed_l401_40175


namespace NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l401_40182

open Real

/-- Prove solutions to the first equation (x + 8)(x + 1) = -12 are x = -4 and x = -5 -/
theorem solve_first_equation (x : ℝ) : (x + 8) * (x + 1) = -12 ↔ x = -4 ∨ x = -5 := by
  sorry

/-- Prove solutions to the second equation 2x^2 + 4x - 1 = 0 are x = (-2 + sqrt 6) / 2 and x = (-2 - sqrt 6) / 2 -/
theorem solve_second_equation (x : ℝ) : 2 * x^2 + 4 * x - 1 = 0 ↔ x = (-2 + sqrt 6) / 2 ∨ x = (-2 - sqrt 6) / 2 := by
  sorry

end NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l401_40182


namespace NUMINAMATH_GPT_max_student_count_l401_40144

theorem max_student_count
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : (x1 + x2 + x3 + x4 + x5) / 5 = 7)
  (h2 : ((x1 - 7) ^ 2 + (x2 - 7) ^ 2 + (x3 - 7) ^ 2 + (x4 - 7) ^ 2 + (x5 - 7) ^ 2) / 5 = 4)
  (h3 : ∀ i j, i ≠ j → List.nthLe [x1, x2, x3, x4, x5] i sorry ≠ List.nthLe [x1, x2, x3, x4, x5] j sorry) :
  max x1 (max x2 (max x3 (max x4 x5))) = 10 := 
sorry

end NUMINAMATH_GPT_max_student_count_l401_40144


namespace NUMINAMATH_GPT_parabola_intercept_sum_l401_40159

theorem parabola_intercept_sum :
  let a := 6
  let b := 1
  let c := 2
  a + b + c = 9 :=
by
  sorry

end NUMINAMATH_GPT_parabola_intercept_sum_l401_40159


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l401_40177

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_property
  (h1 : a 3 + 3 * a 8 + a 13 = 120)
  (h2 : a 3 + a 13 = 2 * a 8) :
  a 3 + a 13 - a 8 = 24 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l401_40177


namespace NUMINAMATH_GPT_jerry_needs_money_l401_40190

theorem jerry_needs_money 
  (current_count : ℕ) (total_needed : ℕ) (cost_per_action_figure : ℕ)
  (h1 : current_count = 7) 
  (h2 : total_needed = 16) 
  (h3 : cost_per_action_figure = 8) :
  (total_needed - current_count) * cost_per_action_figure = 72 :=
by sorry

end NUMINAMATH_GPT_jerry_needs_money_l401_40190


namespace NUMINAMATH_GPT_triangle_areas_l401_40140

theorem triangle_areas (S₁ S₂ : ℝ) :
  ∃ (ABC : ℝ), ABC = Real.sqrt (S₁ * S₂) :=
sorry

end NUMINAMATH_GPT_triangle_areas_l401_40140


namespace NUMINAMATH_GPT_problem_statement_equality_condition_l401_40165

theorem problem_statement (x y z : ℝ) (hx : 0 <= x) (hy : 0 <= y) (hz : 0 <= z) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) >= 2 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : 0 <= x) (hy : 0 <= y) (hz : 0 <= z) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_equality_condition_l401_40165


namespace NUMINAMATH_GPT_at_least_two_greater_than_one_l401_40191

theorem at_least_two_greater_than_one
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : a + b + c = a * b * c) : 
  1 < a ∨ 1 < b ∨ 1 < c :=
sorry

end NUMINAMATH_GPT_at_least_two_greater_than_one_l401_40191


namespace NUMINAMATH_GPT_jessica_current_age_l401_40145

theorem jessica_current_age : 
  ∃ J M_d M_c : ℕ, 
    J = (M_d / 2) ∧ 
    M_d = M_c - 10 ∧ 
    M_c = 70 ∧ 
    J + 10 = 40 := 
sorry

end NUMINAMATH_GPT_jessica_current_age_l401_40145


namespace NUMINAMATH_GPT_cuboid_diagonals_and_edges_l401_40153

theorem cuboid_diagonals_and_edges (a b c : ℝ) : 
  4 * (a^2 + b^2 + c^2) = 4 * a^2 + 4 * b^2 + 4 * c^2 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_diagonals_and_edges_l401_40153


namespace NUMINAMATH_GPT_tyler_puppies_l401_40127

theorem tyler_puppies (dogs : ℕ) (puppies_per_dog : ℕ) (total_puppies : ℕ) 
  (h1 : dogs = 15) (h2 : puppies_per_dog = 5) : total_puppies = 75 :=
by {
  sorry
}

end NUMINAMATH_GPT_tyler_puppies_l401_40127
