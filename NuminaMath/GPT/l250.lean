import Mathlib

namespace coeff_of_quadratic_term_eq_neg5_l250_25066

theorem coeff_of_quadratic_term_eq_neg5 (a b c : ℝ) (h_eq : -5 * x^2 + 5 * x + 6 = a * x^2 + b * x + c) :
  a = -5 :=
by
  sorry

end coeff_of_quadratic_term_eq_neg5_l250_25066


namespace catering_budget_l250_25053

namespace CateringProblem

variables (s c : Nat) (cost_steak cost_chicken : Nat)

def total_guests (s c : Nat) : Prop := s + c = 80

def steak_to_chicken_ratio (s c : Nat) : Prop := s = 3 * c

def total_cost (s c cost_steak cost_chicken : Nat) : Nat := s * cost_steak + c * cost_chicken

theorem catering_budget :
  ∃ (s c : Nat), (total_guests s c) ∧ (steak_to_chicken_ratio s c) ∧ (total_cost s c 25 18) = 1860 :=
by
  sorry

end CateringProblem

end catering_budget_l250_25053


namespace linear_function_no_first_quadrant_l250_25008

theorem linear_function_no_first_quadrant : 
  ¬ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = -3 * x - 2 := by
  sorry

end linear_function_no_first_quadrant_l250_25008


namespace range_of_m_l250_25046

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  x^3 - 6 * x^2 + 9 * x + m

theorem range_of_m (m : ℝ) :
  (∃ a b c : ℝ, a < b ∧ b < c ∧ f m a = 0 ∧ f m b = 0 ∧ f m c = 0) ↔ -4 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l250_25046


namespace quadratic_value_range_l250_25024

theorem quadratic_value_range (y : ℝ) (h : y^3 - 6 * y^2 + 11 * y - 6 < 0) : 
  1 ≤ y^2 - 4 * y + 5 ∧ y^2 - 4 * y + 5 ≤ 2 := 
sorry

end quadratic_value_range_l250_25024


namespace find_f_at_3_l250_25027

theorem find_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = x ^ 2 - 2 * x) : f 3 = -1 :=
by {
  -- Proof would go here.
  sorry
}

end find_f_at_3_l250_25027


namespace pick_peanut_cluster_percentage_l250_25070

def total_chocolates := 100
def typeA_caramels := 5
def typeB_caramels := 6
def typeC_caramels := 4
def typeD_nougats := 2 * typeA_caramels
def typeE_nougats := 2 * typeB_caramels
def typeF_truffles := typeA_caramels + 6
def typeG_truffles := typeB_caramels + 6
def typeH_truffles := typeC_caramels + 6

def total_non_peanut_clusters := 
  typeA_caramels + typeB_caramels + typeC_caramels + typeD_nougats + typeE_nougats + typeF_truffles + typeG_truffles + typeH_truffles

def number_peanut_clusters := total_chocolates - total_non_peanut_clusters

def percent_peanut_clusters := (number_peanut_clusters * 100) / total_chocolates

theorem pick_peanut_cluster_percentage : percent_peanut_clusters = 30 := 
by {
  sorry
}

end pick_peanut_cluster_percentage_l250_25070


namespace largest_quadrilateral_angle_l250_25083

theorem largest_quadrilateral_angle (x : ℝ)
  (h1 : 3 * x + 4 * x + 5 * x + 6 * x = 360) :
  6 * x = 120 :=
by
  sorry

end largest_quadrilateral_angle_l250_25083


namespace sum_of_parts_l250_25036

variable (x y : ℤ)
variable (h1 : x + y = 60)
variable (h2 : y = 45)

theorem sum_of_parts : 10 * x + 22 * y = 1140 :=
by
  sorry

end sum_of_parts_l250_25036


namespace max_apples_discarded_l250_25001

theorem max_apples_discarded (n : ℕ) : n % 7 ≤ 6 := by
  sorry

end max_apples_discarded_l250_25001


namespace bleach_contains_chlorine_l250_25076

noncomputable def element_in_bleach (mass_percentage : ℝ) (substance : String) : String :=
  if mass_percentage = 31.08 ∧ substance = "sodium hypochlorite" then "Chlorine"
  else "unknown"

theorem bleach_contains_chlorine : element_in_bleach 31.08 "sodium hypochlorite" = "Chlorine" :=
by
  sorry

end bleach_contains_chlorine_l250_25076


namespace calculate_large_exponent_l250_25021

theorem calculate_large_exponent : (1307 * 1307)^3 = 4984209203082045649 :=
by {
   sorry
}

end calculate_large_exponent_l250_25021


namespace factor_correct_l250_25018

noncomputable def factor_expr (x : ℝ) : ℝ :=
  75 * x^3 - 225 * x^10
  
noncomputable def factored_form (x : ℝ) : ℝ :=
  75 * x^3 * (1 - 3 * x^7)

theorem factor_correct (x : ℝ): 
  factor_expr x = factored_form x :=
by
  -- Proof omitted
  sorry

end factor_correct_l250_25018


namespace tangent_lines_diff_expected_l250_25017

noncomputable def tangent_lines_diff (a : ℝ) (k1 k2 : ℝ) : Prop :=
  let curve (x : ℝ) := a * x + 2 * Real.log (|x|)
  let deriv (x : ℝ) := a + 2 / x
  -- Tangent conditions at some x1 > 0 for k1
  (∃ x1 : ℝ, 0 < x1 ∧ k1 = deriv x1 ∧ curve x1 = k1 * x1)
  -- Tangent conditions at some x2 < 0 for k2
  ∧ (∃ x2 : ℝ, x2 < 0 ∧ k2 = deriv x2 ∧ curve x2 = k2 * x2)
  -- The lines' slopes relations
  ∧ k1 > k2

theorem tangent_lines_diff_expected (a k1 k2 : ℝ) (h : tangent_lines_diff a k1 k2) :
  k1 - k2 = 4 / Real.exp 1 :=
sorry

end tangent_lines_diff_expected_l250_25017


namespace min_distance_l250_25050

theorem min_distance (x y z : ℝ) :
  ∃ (m : ℝ), m = (Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2)) ∧ m = Real.sqrt 6 :=
by
  sorry

end min_distance_l250_25050


namespace fraction_upgraded_sensors_l250_25033

theorem fraction_upgraded_sensors (N U : ℕ) (h1 : N = U / 3) (h2 : U = 3 * N) : 
  (U : ℚ) / (24 * N + U) = 1 / 9 := by
  sorry

end fraction_upgraded_sensors_l250_25033


namespace range_of_log2_sin_squared_l250_25061

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def sin_squared_log_range (x : ℝ) : ℝ :=
  log2 ((Real.sin x) ^ 2)

theorem range_of_log2_sin_squared (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  ∃ y, y = sin_squared_log_range x ∧ y ≤ 0 :=
by
  sorry

end range_of_log2_sin_squared_l250_25061


namespace train_speed_in_km_per_hr_l250_25012

/-- Given the length of a train and a bridge, and the time taken for the train to cross the bridge, prove the speed of the train in km/hr -/
theorem train_speed_in_km_per_hr
  (train_length : ℕ)  -- 100 meters
  (bridge_length : ℕ) -- 275 meters
  (crossing_time : ℕ) -- 30 seconds
  (conversion_factor : ℝ) -- 1 m/s = 3.6 km/hr
  (h_train_length : train_length = 100)
  (h_bridge_length : bridge_length = 275)
  (h_crossing_time : crossing_time = 30)
  (h_conversion_factor : conversion_factor = 3.6) : 
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 := 
sorry

end train_speed_in_km_per_hr_l250_25012


namespace total_tiles_count_l250_25028

theorem total_tiles_count (n total_tiles: ℕ) 
  (h1: total_tiles - n^2 = 36) 
  (h2: total_tiles - (n + 1)^2 = 3) : total_tiles = 292 :=
by {
  sorry
}

end total_tiles_count_l250_25028


namespace number_of_fills_l250_25088

-- Definitions based on conditions
def needed_flour : ℚ := 4 + 3 / 4
def cup_capacity : ℚ := 1 / 3

-- The proof statement
theorem number_of_fills : (needed_flour / cup_capacity).ceil = 15 := by
  sorry

end number_of_fills_l250_25088


namespace xyz_inequality_l250_25063

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2) :=
sorry

end xyz_inequality_l250_25063


namespace evaluate_fraction_l250_25019

theorem evaluate_fraction : (35 / 0.07) = 500 := 
by
  sorry

end evaluate_fraction_l250_25019


namespace scientific_notation_189100_l250_25086

  theorem scientific_notation_189100 :
    (∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 189100 = a * 10^n) ∧ (∃ (a : ℝ) (n : ℤ), a = 1.891 ∧ n = 5) :=
  by {
    sorry
  }
  
end scientific_notation_189100_l250_25086


namespace max_satiated_pikes_l250_25010

-- Define the total number of pikes
def total_pikes : ℕ := 30

-- Define the condition for satiation
def satiated_condition (eats : ℕ) : Prop := eats ≥ 3

-- Define the number of pikes eaten by each satiated pike
def eaten_by_satiated_pike : ℕ := 3

-- Define the theorem to find the maximum number of satiated pikes
theorem max_satiated_pikes (s : ℕ) : 
  (s * eaten_by_satiated_pike < total_pikes) → s ≤ 9 :=
by
  sorry

end max_satiated_pikes_l250_25010


namespace arithmetic_sequence_third_term_l250_25085

theorem arithmetic_sequence_third_term {a d : ℝ} (h : 2 * a + 4 * d = 10) : a + 2 * d = 5 :=
sorry

end arithmetic_sequence_third_term_l250_25085


namespace fernanda_total_time_to_finish_l250_25005

noncomputable def fernanda_days_to_finish_audiobooks
  (num_audiobooks : ℕ) (hours_per_audiobook : ℕ) (hours_listened_per_day : ℕ) : ℕ :=
num_audiobooks * hours_per_audiobook / hours_listened_per_day

-- Definitions based on the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Statement to prove
theorem fernanda_total_time_to_finish :
  fernanda_days_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 := 
sorry

end fernanda_total_time_to_finish_l250_25005


namespace simplify_expression_and_evaluate_evaluate_expression_at_one_l250_25022

theorem simplify_expression_and_evaluate (x : ℝ)
  (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  ( ((x^2 - 2*x) / (x^2 - 4*x + 4) - 3 / (x - 2)) / ((x - 3) / (x^2 - 4)) ) = x + 2 :=
by {
  sorry
}

theorem evaluate_expression_at_one :
  ( ((1^2 - 2*1) / (1^2 - 4*1 + 4) - 3 / (1 - 2)) / ((1 - 3) / (1^2 - 4)) ) = 3 :=
by {
  sorry
}

end simplify_expression_and_evaluate_evaluate_expression_at_one_l250_25022


namespace hypothesis_test_l250_25057

def X : List ℕ := [3, 4, 6, 10, 13, 17]
def Y : List ℕ := [1, 2, 5, 7, 16, 20, 22]

def alpha : ℝ := 0.01
def W_lower : ℕ := 24
def W_upper : ℕ := 60
def W1 : ℕ := 41

-- stating the null hypothesis test condition
theorem hypothesis_test : (24 < 41) ∧ (41 < 60) :=
by
  sorry

end hypothesis_test_l250_25057


namespace math_proof_l250_25030

def exponentiation_result := -1 ^ 4
def negative_exponentiation_result := (-2) ^ 3
def absolute_value_result := abs (-3 - 1)
def division_result := 16 / negative_exponentiation_result
def multiplication_result := division_result * absolute_value_result
def final_result := exponentiation_result + multiplication_result

theorem math_proof : final_result = -9 := by
  -- To be proved
  sorry

end math_proof_l250_25030


namespace real_roots_of_quadratic_l250_25047

theorem real_roots_of_quadratic (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ m ≤ 4 / 3 :=
by
  sorry

end real_roots_of_quadratic_l250_25047


namespace second_smallest_three_digit_in_pascal_triangle_l250_25003

theorem second_smallest_three_digit_in_pascal_triangle (m n : ℕ) :
  (∀ k : ℕ, ∃! r c : ℕ, r ≥ c ∧ r.choose c = k) →
  (∃! r : ℕ, r ≥ 2 ∧ 100 = r.choose 1) →
  (m = 101 ∧ n = 101) :=
by
  sorry

end second_smallest_three_digit_in_pascal_triangle_l250_25003


namespace josh_remaining_marbles_l250_25074

def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7
def remaining_marbles : ℕ := 9

theorem josh_remaining_marbles : initial_marbles - lost_marbles = remaining_marbles := by
  sorry

end josh_remaining_marbles_l250_25074


namespace find_x_squared_plus_y_squared_l250_25015

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : x - y = 20) (h2 : x * y = 9) :
  x^2 + y^2 = 418 :=
sorry

end find_x_squared_plus_y_squared_l250_25015


namespace centroid_sum_of_squares_l250_25043

theorem centroid_sum_of_squares (a b c : ℝ) 
  (h : 1 / Real.sqrt (1 / a^2 + 1 / b^2 + 1 / c^2) = 2) : 
  1 / (a / 3) ^ 2 + 1 / (b / 3) ^ 2 + 1 / (c / 3) ^ 2 = 9 / 4 :=
by
  sorry

end centroid_sum_of_squares_l250_25043


namespace pieces_per_box_l250_25004

theorem pieces_per_box (total_pieces : ℕ) (boxes : ℕ) (h_total : total_pieces = 3000) (h_boxes : boxes = 6) :
  total_pieces / boxes = 500 := by
  sorry

end pieces_per_box_l250_25004


namespace intersection_complement_B_and_A_l250_25049

open Set Real

def A : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def B : Set ℝ := { x | x > 2 }
def CR_B : Set ℝ := { x | x ≤ 2 }

theorem intersection_complement_B_and_A : CR_B ∩ A = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_complement_B_and_A_l250_25049


namespace cos_of_F_in_def_l250_25082

theorem cos_of_F_in_def (E F : ℝ) (h₁ : E + F = π / 2) (h₂ : Real.sin E = 3 / 5) : Real.cos F = 3 / 5 :=
sorry

end cos_of_F_in_def_l250_25082


namespace television_price_reduction_l250_25077

variable (P : ℝ) (F : ℝ)
variable (h : F = 0.56 * P - 50)

theorem television_price_reduction :
  F / P = 0.56 - 50 / P :=
by {
  sorry
}

end television_price_reduction_l250_25077


namespace standard_eq_of_ellipse_value_of_k_l250_25091

-- Definitions and conditions
def is_ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0

def eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = (Real.sqrt 2) / 2 ∧ a^2 = b^2 + (a * e)^2

def minor_axis_length (b : ℝ) : Prop :=
  2 * b = 2

def is_tangency (k m : ℝ) : Prop := 
  m^2 = 1 + k^2

def line_intersect_ellipse (k m : ℝ) : Prop :=
  (4 * k * m)^2 - 4 * (1 + 2 * k^2) * (2 * m^2 - 2) > 0

def dot_product_condition (k m : ℝ) : Prop :=
  let x1 := -(4 * k * m) / (1 + 2 * k^2)
  let x2 := (2 * m^2 - 2) / (1 + 2 * k^2)
  let y1 := k * x1 + m
  let y2 := k * x2 + m
  x1 * x2 + y1 * y2 = 2 / 3

-- To prove the standard equation of the ellipse
theorem standard_eq_of_ellipse {a b : ℝ} (h_ellipse : is_ellipse a b)
  (h_eccentricity : eccentricity a b ((Real.sqrt 2) / 2)) 
  (h_minor_axis : minor_axis_length b) : 
  ∃ a, a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y, (x^2 / 2 + y^2 = 1)) := 
sorry

-- To prove the value of k
theorem value_of_k {k m : ℝ} (h_tangency : is_tangency k m) 
  (h_intersect : line_intersect_ellipse k m)
  (h_dot_product : dot_product_condition k m) :
  k = 1 ∨ k = -1 :=
sorry

end standard_eq_of_ellipse_value_of_k_l250_25091


namespace ott_fraction_l250_25055

/-- 
Moe, Loki, Nick, and Pat each give $2 to Ott.
Moe gave Ott one-seventh of his money.
Loki gave Ott one-fifth of his money.
Nick gave Ott one-fourth of his money.
Pat gave Ott one-sixth of his money.
-/
def fraction_of_money_ott_now_has (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : Prop :=
  A = 14 ∧ B = 10 ∧ C = 8 ∧ D = 12 ∧ (2 * (1 / 7 : ℚ)) = 2 ∧ (2 * (1 / 5 : ℚ)) = 2 ∧ (2 * (1 / 4 : ℚ)) = 2 ∧ (2 * (1 / 6 : ℚ)) = 2

theorem ott_fraction (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) (h : fraction_of_money_ott_now_has A B C D) : 
  8 = (2 / 11 : ℚ) * (A + B + C + D) :=
by sorry

end ott_fraction_l250_25055


namespace work_completion_time_of_x_l250_25025

def totalWork := 1  -- We can normalize W to 1 unit to simplify the problem

theorem work_completion_time_of_x (W : ℝ) (Wx Wy : ℝ) 
  (hx : 8 * Wx + 16 * Wy = W)
  (hy : Wy = W / 20) :
  Wx = W / 40 :=
by
  -- The proof goes here, but we just put sorry for now.
  sorry

end work_completion_time_of_x_l250_25025


namespace reflection_proof_l250_25000

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

noncomputable def initial_point : ℝ × ℝ := (3, -3)
noncomputable def reflected_over_y_axis := reflect_y initial_point
noncomputable def reflected_over_x_axis := reflect_x reflected_over_y_axis

theorem reflection_proof : reflected_over_x_axis = (-3, 3) :=
  by
    -- proof goes here
    sorry

end reflection_proof_l250_25000


namespace weight_of_new_student_l250_25026

theorem weight_of_new_student (W x y z : ℝ) (h : (W - x - y + z = W - 40)) : z = 40 - (x + y) :=
by
  sorry

end weight_of_new_student_l250_25026


namespace product_of_terms_l250_25060

variable {α : Type*} [LinearOrderedField α]

namespace GeometricSequence

def is_geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_terms (a : ℕ → α) (r : α) (h_geo : is_geometric_sequence a) :
  (a 4) * (a 8) = 16 → (a 2) * (a 10) = 16 :=
by
  intro h1
  sorry

end GeometricSequence

end product_of_terms_l250_25060


namespace smallest_positive_number_among_options_l250_25052

theorem smallest_positive_number_among_options :
  (10 > 3 * Real.sqrt 11) →
  (51 > 10 * Real.sqrt 26) →
  min (10 - 3 * Real.sqrt 11) (51 - 10 * Real.sqrt 26) = 51 - 10 * Real.sqrt 26 :=
by
  intros h1 h2
  sorry

end smallest_positive_number_among_options_l250_25052


namespace right_triangle_perimeter_l250_25059

theorem right_triangle_perimeter
  (a b : ℝ)
  (h_area : 0.5 * 30 * b = 150)
  (h_leg : a = 30) :
  a + b + Real.sqrt (a^2 + b^2) = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l250_25059


namespace max_value_of_f_l250_25068

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * (Real.cos x)

theorem max_value_of_f : ∃ x : ℝ, f x ≤ 4 :=
sorry

end max_value_of_f_l250_25068


namespace max_min_values_of_function_l250_25064

theorem max_min_values_of_function :
  ∀ (x : ℝ), -5 ≤ 4 * Real.sin x + 3 * Real.cos x ∧ 4 * Real.sin x + 3 * Real.cos x ≤ 5 :=
by
  sorry

end max_min_values_of_function_l250_25064


namespace Bo_needs_to_learn_per_day_l250_25056

theorem Bo_needs_to_learn_per_day
  (total_flashcards : ℕ)
  (known_percentage : ℚ)
  (days_to_learn : ℕ)
  (h1 : total_flashcards = 800)
  (h2 : known_percentage = 0.20)
  (h3 : days_to_learn = 40) : 
  total_flashcards * (1 - known_percentage) / days_to_learn = 16 := 
by
  sorry

end Bo_needs_to_learn_per_day_l250_25056


namespace number_and_sum_of_g3_l250_25029

-- Define the function g with its conditions
variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (x * g y - x) = 2 * x * y + g x)

-- Define the problem parameters
def n : ℕ := sorry -- Number of possible values of g(3)
def s : ℝ := sorry -- Sum of all possible values of g(3)

-- The main statement to be proved
theorem number_and_sum_of_g3 : n * s = 0 := sorry

end number_and_sum_of_g3_l250_25029


namespace cistern_water_breadth_l250_25092

theorem cistern_water_breadth 
  (length width : ℝ) (wet_surface_area : ℝ) 
  (hl : length = 9) (hw : width = 6) (hwsa : wet_surface_area = 121.5) : 
  ∃ h : ℝ, 54 + 18 * h + 12 * h = 121.5 ∧ h = 2.25 := 
by 
  sorry

end cistern_water_breadth_l250_25092


namespace geometric_series_sum_l250_25080

theorem geometric_series_sum :
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  S = 117775277204 / 30517578125 := by
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  have : S = 117775277204 / 30517578125 := sorry
  exact this

end geometric_series_sum_l250_25080


namespace seventh_grade_male_students_l250_25072

theorem seventh_grade_male_students:
  ∃ x : ℤ, (48 = x + (4*x)/5 + 3) ∧ x = 25 :=
by
  sorry

end seventh_grade_male_students_l250_25072


namespace smallest_two_digit_product_12_l250_25013

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l250_25013


namespace road_trip_ratio_l250_25069

theorem road_trip_ratio (D R: ℝ) (h1 : 1 / 2 * D = 40) (h2 : 2 * (D + R * D + 40) = 560 - (D + R * D + 40)) :
  R = 5 / 6 := by
  sorry

end road_trip_ratio_l250_25069


namespace sum_of_numbers_in_ratio_with_lcm_l250_25020

theorem sum_of_numbers_in_ratio_with_lcm (a b : ℕ) (h_lcm : Nat.lcm a b = 36) (h_ratio : a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3) : a + b = 30 :=
sorry

end sum_of_numbers_in_ratio_with_lcm_l250_25020


namespace saree_discount_l250_25087

theorem saree_discount (x : ℝ) : 
  let original_price := 495
  let final_price := 378.675
  let discounted_price := original_price * ((100 - x) / 100) * 0.9
  discounted_price = final_price -> x = 15 := 
by
  intro h
  sorry

end saree_discount_l250_25087


namespace juan_marbles_l250_25078

-- Conditions
def connie_marbles : ℕ := 39
def extra_marbles_juan : ℕ := 25

-- Theorem statement: Total marbles Juan has
theorem juan_marbles : connie_marbles + extra_marbles_juan = 64 :=
by
  sorry

end juan_marbles_l250_25078


namespace spies_denounced_each_other_l250_25089

theorem spies_denounced_each_other :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 10 ∧ 
  (∀ (u v : ℕ), (u, v) ∈ pairs → (v, u) ∈ pairs) :=
sorry

end spies_denounced_each_other_l250_25089


namespace standard_circle_equation_l250_25023

theorem standard_circle_equation (x y : ℝ) :
  ∃ (h k r : ℝ), h = 2 ∧ k = -1 ∧ r = 3 ∧ (x - h)^2 + (y - k + 1)^2 = r^2 :=
by
  use 2, -1, 3
  simp
  sorry

end standard_circle_equation_l250_25023


namespace average_marks_mathematics_chemistry_l250_25073

theorem average_marks_mathematics_chemistry (M P C B : ℕ) 
    (h1 : M + P = 80) 
    (h2 : C + B = 120) 
    (h3 : C = P + 20) 
    (h4 : B = M - 15) : 
    (M + C) / 2 = 50 :=
by
  sorry

end average_marks_mathematics_chemistry_l250_25073


namespace leading_digit_not_necessarily_one_l250_25071

-- Define a condition to check if the leading digit of a number is the same
def same_leading_digit (x: ℕ) (n: ℕ) : Prop :=
  (Nat.digits 10 x).head? = (Nat.digits 10 (x^n)).head?

-- Theorem stating the digit does not need to be 1 under given conditions
theorem leading_digit_not_necessarily_one :
  (∃ x: ℕ, x > 1 ∧ same_leading_digit x 2 ∧ same_leading_digit x 3) ∧ 
  (∃ x: ℕ, x > 1 ∧ ∀ n: ℕ, 1 ≤ n ∧ n ≤ 2015 → same_leading_digit x n) :=
sorry

end leading_digit_not_necessarily_one_l250_25071


namespace magnification_factor_l250_25084

variable (diameter_magnified : ℝ)
variable (diameter_actual : ℝ)
variable (M : ℝ)

theorem magnification_factor
    (h_magnified : diameter_magnified = 0.3)
    (h_actual : diameter_actual = 0.0003) :
    M = diameter_magnified / diameter_actual ↔ M = 1000 := by
  sorry

end magnification_factor_l250_25084


namespace cos_double_angle_l250_25040

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l250_25040


namespace probability_of_triangle_segments_from_15gon_l250_25038

/-- A proof problem that calculates the probability that three randomly selected segments 
    from a regular 15-gon inscribed in a circle form a triangle with positive area. -/
theorem probability_of_triangle_segments_from_15gon : 
  let n := 15
  let total_segments := (n * (n - 1)) / 2 
  let total_combinations := total_segments * (total_segments - 1) * (total_segments - 2) / 6 
  let valid_probability := 943 / 1365
  valid_probability = (total_combinations - count_violating_combinations) / total_combinations :=
sorry

end probability_of_triangle_segments_from_15gon_l250_25038


namespace students_need_to_walk_distance_l250_25090

-- Define distance variables and the relationships
def teacher_initial_distance : ℝ := 235
def xiao_ma_initial_distance : ℝ := 87
def xiao_lu_initial_distance : ℝ := 59
def xiao_zhou_initial_distance : ℝ := 26
def speed_ratio : ℝ := 1.5

-- Prove the distance x students need to walk
theorem students_need_to_walk_distance (x : ℝ) :
  teacher_initial_distance - speed_ratio * x =
  (xiao_ma_initial_distance - x) + (xiao_lu_initial_distance - x) + (xiao_zhou_initial_distance - x) →
  x = 42 :=
by
  sorry

end students_need_to_walk_distance_l250_25090


namespace event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l250_25007

-- Event A
def total_muffins_needed_A := 200
def arthur_muffins_A := 35
def beatrice_muffins_A := 48
def charles_muffins_A := 29
def total_muffins_baked_A := arthur_muffins_A + beatrice_muffins_A + charles_muffins_A
def additional_muffins_needed_A := total_muffins_needed_A - total_muffins_baked_A

-- Event B
def total_muffins_needed_B := 150
def arthur_muffins_B := 20
def beatrice_muffins_B := 35
def charles_muffins_B := 25
def total_muffins_baked_B := arthur_muffins_B + beatrice_muffins_B + charles_muffins_B
def additional_muffins_needed_B := total_muffins_needed_B - total_muffins_baked_B

-- Event C
def total_muffins_needed_C := 250
def arthur_muffins_C := 45
def beatrice_muffins_C := 60
def charles_muffins_C := 30
def total_muffins_baked_C := arthur_muffins_C + beatrice_muffins_C + charles_muffins_C
def additional_muffins_needed_C := total_muffins_needed_C - total_muffins_baked_C

-- Proof Statements
theorem event_A_muffins_correct : additional_muffins_needed_A = 88 := by
  sorry

theorem event_B_muffins_correct : additional_muffins_needed_B = 70 := by
  sorry

theorem event_C_muffins_correct : additional_muffins_needed_C = 115 := by
  sorry

end event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l250_25007


namespace colors_used_l250_25095

theorem colors_used (total_blocks number_per_color : ℕ) (h1 : total_blocks = 196) (h2 : number_per_color = 14) : 
  total_blocks / number_per_color = 14 :=
by
  sorry

end colors_used_l250_25095


namespace find_arithmetic_sequence_l250_25032

theorem find_arithmetic_sequence (a d : ℝ) : 
(a - d) + a + (a + d) = 6 ∧ (a - d) * a * (a + d) = -10 → 
  (a = 2 ∧ d = 3 ∨ a = 2 ∧ d = -3) :=
by
  sorry

end find_arithmetic_sequence_l250_25032


namespace positive_difference_of_squares_and_product_l250_25042

theorem positive_difference_of_squares_and_product (x y : ℕ) 
  (h1 : x + y = 60) (h2 : x - y = 16) :
  x^2 - y^2 = 960 ∧ x * y = 836 :=
by sorry

end positive_difference_of_squares_and_product_l250_25042


namespace printer_z_time_l250_25002

theorem printer_z_time (T_X T_Y T_Z : ℝ) (hZX_Y : T_X = 2.25 * (T_Y + T_Z)) 
  (hX : T_X = 15) (hY : T_Y = 10) : T_Z = 20 :=
by
  rw [hX, hY] at hZX_Y
  sorry

end printer_z_time_l250_25002


namespace expression_exists_l250_25098

theorem expression_exists (a b : ℤ) (h : 5 * a = 3125) (hb : 5 * b = 25) : b = 5 := by
  sorry

end expression_exists_l250_25098


namespace cube_greater_l250_25014

theorem cube_greater (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end cube_greater_l250_25014


namespace total_chairs_l250_25081

theorem total_chairs (living_room_chairs kitchen_chairs : ℕ) (h1 : living_room_chairs = 3) (h2 : kitchen_chairs = 6) :
  living_room_chairs + kitchen_chairs = 9 := by
  sorry

end total_chairs_l250_25081


namespace min_value_2x_minus_y_l250_25039

open Real

theorem min_value_2x_minus_y : ∀ (x y : ℝ), |x| ≤ y ∧ y ≤ 2 → ∃ (c : ℝ), c = 2 * x - y ∧ ∀ z, z = 2 * x - y → z ≥ -6 := sorry

end min_value_2x_minus_y_l250_25039


namespace arccos_cos_three_l250_25045

-- Defining the problem conditions
def three_radians : ℝ := 3

-- Main statement to prove
theorem arccos_cos_three : Real.arccos (Real.cos three_radians) = three_radians := 
sorry

end arccos_cos_three_l250_25045


namespace antonella_toonies_l250_25037

theorem antonella_toonies (L T : ℕ) (h1 : L + T = 10) (h2 : L + 2 * T = 14) : T = 4 :=
by
  sorry

end antonella_toonies_l250_25037


namespace EquivalenceStatements_l250_25062

-- Define real numbers and sets P, Q
variables {x a b c : ℝ} {P Q : Set ℝ}

-- Prove the necessary equivalences
theorem EquivalenceStatements :
  ((x > 1) → (abs x > 1)) ∧ ((∃ x, x < -1) → (abs x > 1)) ∧
  ((a ∈ P ∩ Q) ↔ (a ∈ P ∧ a ∈ Q)) ∧
  (¬ (∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0)) ∧
  (x = 1 ↔ a + b + c = 0) :=
by
  sorry

end EquivalenceStatements_l250_25062


namespace find_side2_l250_25034

-- Define the given conditions
def perimeter : ℕ := 160
def side1 : ℕ := 40
def side3 : ℕ := 70

-- Define the second side as a variable
def side2 : ℕ := perimeter - side1 - side3

-- State the theorem to be proven
theorem find_side2 : side2 = 50 := by
  -- We skip the proof here with sorry
  sorry

end find_side2_l250_25034


namespace total_income_l250_25079

-- Definitions of conditions
def charge_per_meter : ℝ := 0.2
def number_of_fences : ℝ := 50
def length_of_each_fence : ℝ := 500

-- Theorem statement
theorem total_income :
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * charge_per_meter
  total_income = 5000 := 
by
  sorry

end total_income_l250_25079


namespace quadratic_equation_transformation_l250_25048

theorem quadratic_equation_transformation (x : ℝ) :
  (-5 * x ^ 2 = 2 * x + 10) →
  (x ^ 2 + (2 / 5) * x + 2 = 0) :=
by
  intro h
  sorry

end quadratic_equation_transformation_l250_25048


namespace distinct_integers_sum_441_l250_25011

-- Define the variables and conditions
variables (a b c d : ℕ)

-- State the conditions: a, b, c, d are distinct positive integers and their product is 441
def distinct_positive_integers (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
def positive_integers (a b c d : ℕ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

-- Define the main statement to be proved
theorem distinct_integers_sum_441 (a b c d : ℕ) (h_distinct : distinct_positive_integers a b c d) 
(h_positive : positive_integers a b c d) 
(h_product : a * b * c * d = 441) : a + b + c + d = 32 :=
by
  sorry

end distinct_integers_sum_441_l250_25011


namespace reimbursement_calculation_l250_25093

variable (total_paid : ℕ) (pieces : ℕ) (cost_per_piece : ℕ)

theorem reimbursement_calculation
  (h1 : total_paid = 20700)
  (h2 : pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (pieces * cost_per_piece) = 600 := 
by
  sorry

end reimbursement_calculation_l250_25093


namespace sum_tenth_powers_l250_25044

theorem sum_tenth_powers (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : a^10 + b^10 = 123 :=
  sorry

end sum_tenth_powers_l250_25044


namespace probability_A_wins_l250_25051

variable (P_A_not_lose : ℝ) (P_draw : ℝ)
variable (h1 : P_A_not_lose = 0.8)
variable (h2 : P_draw = 0.5)

theorem probability_A_wins : P_A_not_lose - P_draw = 0.3 := by
  sorry

end probability_A_wins_l250_25051


namespace arcsin_neg_one_l250_25054

theorem arcsin_neg_one : Real.arcsin (-1) = -Real.pi / 2 := by
  sorry

end arcsin_neg_one_l250_25054


namespace root_of_quadratic_l250_25041

theorem root_of_quadratic (m : ℝ) (h : 3*1^2 - 1 + m = 0) : m = -2 :=
by {
  sorry
}

end root_of_quadratic_l250_25041


namespace find_number_l250_25031

theorem find_number (x : ℝ) (h : (3 / 4) * (1 / 2) * (2 / 5) * x = 753.0000000000001) : 
  x = 5020.000000000001 :=
by 
  sorry

end find_number_l250_25031


namespace extreme_value_h_tangent_to_both_l250_25075

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a*x + 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.log x - a
noncomputable def h (x : ℝ) : ℝ := f x 1 - g x 1

theorem extreme_value_h : h (1/2) = 11/4 + Real.log 2 := by
  sorry

theorem tangent_to_both : ∀ (a : ℝ), ∃ x₁ x₂ : ℝ, (2 * x₁ + a = 1 / x₂) ∧ 
  ((x₁ = (1 / (2 * x₂)) - (a / 2)) ∧ (a ≥ -1)) := by
  sorry

end extreme_value_h_tangent_to_both_l250_25075


namespace expression_value_l250_25016

theorem expression_value (x : ℤ) (hx : x = 1729) : abs (abs (abs x + x) + abs x) + x = 6916 :=
by
  rw [hx]
  sorry

end expression_value_l250_25016


namespace fred_earned_from_car_wash_l250_25058

def weekly_allowance : ℕ := 16
def spent_on_movies : ℕ := weekly_allowance / 2
def amount_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 14
def earned_from_car_wash : ℕ := final_amount - amount_after_movies

theorem fred_earned_from_car_wash : earned_from_car_wash = 6 := by
  sorry

end fred_earned_from_car_wash_l250_25058


namespace train_length_equals_sixty_two_point_five_l250_25006

-- Defining the conditions
noncomputable def calculate_train_length (speed_faster_train : ℝ) (speed_slower_train : ℝ) (time_seconds : ℝ) : ℝ :=
  let relative_speed_kmh := speed_faster_train - speed_slower_train
  let relative_speed_ms := (relative_speed_kmh * 5) / 18
  let distance_covered := relative_speed_ms * time_seconds
  distance_covered / 2

theorem train_length_equals_sixty_two_point_five :
  calculate_train_length 46 36 45 = 62.5 :=
sorry

end train_length_equals_sixty_two_point_five_l250_25006


namespace sufficient_but_not_necessary_condition_l250_25096

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x = 0 → (x^2 - 2 * x = 0)) ∧ (∃ y : ℝ, y ≠ 0 ∧ y ^ 2 - 2 * y = 0) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l250_25096


namespace distinct_x_intercepts_l250_25094

theorem distinct_x_intercepts : 
  let f (x : ℝ) := ((x - 8) * (x^2 + 4*x + 3))
  (∃ x1 x2 x3 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
by
  sorry

end distinct_x_intercepts_l250_25094


namespace problem_statement_l250_25067

theorem problem_statement (a b : ℝ) (h : a < b) : a - b < 0 :=
sorry

end problem_statement_l250_25067


namespace f_zero_one_and_odd_l250_25035

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (a b : ℝ) : f (a * b) = a * f b + b * f a
axiom f_not_zero : ∃ x : ℝ, f x ≠ 0

theorem f_zero_one_and_odd :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end f_zero_one_and_odd_l250_25035


namespace distance_to_city_l250_25009

variable (d : ℝ)  -- Define d as a real number

theorem distance_to_city (h1 : ¬ (d ≥ 13)) (h2 : ¬ (d ≤ 10)) :
  10 < d ∧ d < 13 :=
by
  -- Here we will formalize the proof in Lean syntax
  sorry

end distance_to_city_l250_25009


namespace initial_yellow_hard_hats_count_l250_25097

noncomputable def initial_yellow_hard_hats := 24

theorem initial_yellow_hard_hats_count
  (initial_pink: ℕ)
  (initial_green: ℕ)
  (carl_pink: ℕ)
  (john_pink: ℕ)
  (john_green: ℕ)
  (total_remaining: ℕ)
  (remaining_pink: ℕ)
  (remaining_green: ℕ)
  (initial_yellow: ℕ) :
  initial_pink = 26 →
  initial_green = 15 →
  carl_pink = 4 →
  john_pink = 6 →
  john_green = 2 * john_pink →
  total_remaining = 43 →
  remaining_pink = initial_pink - carl_pink - john_pink →
  remaining_green = initial_green - john_green →
  initial_yellow = total_remaining - remaining_pink - remaining_green →
  initial_yellow = initial_yellow_hard_hats :=
by
  intros
  sorry

end initial_yellow_hard_hats_count_l250_25097


namespace alpha_beta_inequality_l250_25065

theorem alpha_beta_inequality (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) : 
  -2 < α - β ∧ α - β < 0 := 
sorry

end alpha_beta_inequality_l250_25065


namespace henry_books_donation_l250_25099

theorem henry_books_donation
  (initial_books : ℕ := 99)
  (room_books : ℕ := 21)
  (coffee_table_books : ℕ := 4)
  (cookbook_books : ℕ := 18)
  (boxes : ℕ := 3)
  (picked_up_books : ℕ := 12)
  (final_books : ℕ := 23) :
  (initial_books - final_books + picked_up_books - (room_books + coffee_table_books + cookbook_books)) / boxes = 15 :=
by
  sorry

end henry_books_donation_l250_25099
