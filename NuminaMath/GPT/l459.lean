import Mathlib

namespace NUMINAMATH_GPT_stratified_sampling_vision_test_l459_45965

theorem stratified_sampling_vision_test 
  (n_total : ℕ) (n_HS : ℕ) (n_selected : ℕ)
  (h1 : n_total = 165)
  (h2 : n_HS = 66)
  (h3 : n_selected = 15) :
  (n_HS * n_selected / n_total) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_stratified_sampling_vision_test_l459_45965


namespace NUMINAMATH_GPT_player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l459_45924

-- Define probabilities of shots
def shooting_probability_A : ℝ := 0.5
def shooting_probability_B : ℝ := 0.6

-- Define initial points for questions
def initial_points_question_1 : ℝ := 0
def initial_points_question_2 : ℝ := 2

-- Given initial probabilities
def P_0 : ℝ := 0
def P_4 : ℝ := 1

-- Probability that player A wins after exactly 4 rounds
def probability_A_wins_after_4_rounds : ℝ :=
  let P_A := shooting_probability_A * (1 - shooting_probability_B)
  let P_B := shooting_probability_B * (1 - shooting_probability_A)
  let P_C := 1 - P_A - P_B
  P_A * P_C^2 * P_A + P_A * P_B * P_A^2

-- Define the probabilities P(i) for i=0..4
def P (i : ℕ) : ℝ := sorry -- Placeholder for the function

-- Define the proof problem
theorem player_A_wins_after_4_rounds : probability_A_wins_after_4_rounds = 0.0348 :=
sorry

theorem geometric_sequence_differences :
  ∀ i : ℕ, i < 4 → (P (i + 1) - P i) / (P (i + 2) - P (i + 1)) = 2/3 :=
sorry

theorem find_P_2 : P 2 = 4/13 :=
sorry

end NUMINAMATH_GPT_player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l459_45924


namespace NUMINAMATH_GPT_sandy_comic_books_l459_45926

-- Define Sandy's initial number of comic books
def initial_comic_books : ℕ := 14

-- Define the number of comic books Sandy sold
def sold_comic_books (n : ℕ) : ℕ := n / 2

-- Define the number of comic books Sandy bought
def bought_comic_books : ℕ := 6

-- Define the number of comic books Sandy has now
def final_comic_books (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

-- The theorem statement to prove the final number of comic books
theorem sandy_comic_books : final_comic_books initial_comic_books (sold_comic_books initial_comic_books) bought_comic_books = 13 := by
  sorry

end NUMINAMATH_GPT_sandy_comic_books_l459_45926


namespace NUMINAMATH_GPT_plane_hit_probability_l459_45985

theorem plane_hit_probability :
  let P_A : ℝ := 0.3
  let P_B : ℝ := 0.5
  let P_not_A : ℝ := 1 - P_A
  let P_not_B : ℝ := 1 - P_B
  let P_both_miss : ℝ := P_not_A * P_not_B
  let P_plane_hit : ℝ := 1 - P_both_miss
  P_plane_hit = 0.65 :=
by
  sorry

end NUMINAMATH_GPT_plane_hit_probability_l459_45985


namespace NUMINAMATH_GPT_original_avg_age_is_fifty_l459_45945

-- Definitions based on conditions
variable (N : ℕ) -- original number of students
variable (A : ℕ) -- original average age
variable (new_students : ℕ) -- number of new students
variable (new_avg_age : ℕ) -- average age of new students
variable (decreased_avg_age : ℕ) -- new average age after new students join

-- Conditions given in the problem
def original_avg_age_condition : Prop := A = 50
def new_students_condition : Prop := new_students = 12
def avg_age_new_students_condition : Prop := new_avg_age = 32
def decreased_avg_age_condition : Prop := decreased_avg_age = 46

-- Final Mathematical Equivalent Proof Problem
theorem original_avg_age_is_fifty
  (h1 : original_avg_age_condition A)
  (h2 : new_students_condition new_students)
  (h3 : avg_age_new_students_condition new_avg_age)
  (h4 : decreased_avg_age_condition decreased_avg_age) :
  A = 50 :=
by sorry

end NUMINAMATH_GPT_original_avg_age_is_fifty_l459_45945


namespace NUMINAMATH_GPT_trihedral_sphere_radius_l459_45967

noncomputable def sphere_radius 
  (α r : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  : ℝ :=
r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3)

theorem trihedral_sphere_radius 
  (α r R : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  (hR : R = sphere_radius α r hα) 
  : R = r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3) :=
by
  sorry

end NUMINAMATH_GPT_trihedral_sphere_radius_l459_45967


namespace NUMINAMATH_GPT_expected_value_of_draws_before_stopping_l459_45968

noncomputable def totalBalls := 10
noncomputable def redBalls := 2
noncomputable def whiteBalls := 8

noncomputable def prob_one_draw_white : ℚ := whiteBalls / totalBalls
noncomputable def prob_two_draws_white : ℚ := (redBalls / totalBalls) * (whiteBalls / (totalBalls - 1))
noncomputable def prob_three_draws_white : ℚ := (redBalls / (totalBalls - redBalls + 1)) * ((redBalls - 1) / (totalBalls - 1)) * (whiteBalls / (totalBalls - 2))

noncomputable def expected_draws_before_white : ℚ :=
  1 * prob_one_draw_white + 2 * prob_two_draws_white + 3 * prob_three_draws_white

theorem expected_value_of_draws_before_stopping : expected_draws_before_white = 11 / 9 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_draws_before_stopping_l459_45968


namespace NUMINAMATH_GPT_gcd_combination_l459_45989

theorem gcd_combination (a b d : ℕ) (h : d = Nat.gcd a b) : 
  Nat.gcd (5 * a + 3 * b) (13 * a + 8 * b) = d := 
by
  sorry

end NUMINAMATH_GPT_gcd_combination_l459_45989


namespace NUMINAMATH_GPT_find_h_l459_45912

theorem find_h (x : ℝ) : 
  ∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - (-3 / 2))^2 + k :=
sorry

end NUMINAMATH_GPT_find_h_l459_45912


namespace NUMINAMATH_GPT_percent_eighth_graders_combined_l459_45995

theorem percent_eighth_graders_combined (p_students : ℕ) (m_students : ℕ)
  (p_grade8_percent : ℚ) (m_grade8_percent : ℚ) :
  p_students = 160 → m_students = 250 →
  p_grade8_percent = 18 / 100 → m_grade8_percent = 22 / 100 →
  100 * (p_grade8_percent * p_students + m_grade8_percent * m_students) / (p_students + m_students) = 20 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_percent_eighth_graders_combined_l459_45995


namespace NUMINAMATH_GPT_find_a100_l459_45976

noncomputable def arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a n - a (n + 1) = 2

theorem find_a100 (a : ℕ → ℤ) (h1 : arithmetic_sequence 3 a) (h2 : a 3 = 6) :
  a 100 = -188 :=
sorry

end NUMINAMATH_GPT_find_a100_l459_45976


namespace NUMINAMATH_GPT_jill_tax_on_other_items_l459_45908

noncomputable def tax_on_other_items (total_spent clothing_tax_percent total_tax_percent : ℝ) : ℝ :=
  let clothing_spent := 0.5 * total_spent
  let food_spent := 0.25 * total_spent
  let other_spent := 0.25 * total_spent
  let clothing_tax := clothing_tax_percent * clothing_spent
  let total_tax := total_tax_percent * total_spent
  let tax_on_others := total_tax - clothing_tax
  (tax_on_others / other_spent) * 100

theorem jill_tax_on_other_items :
  let total_spent := 100
  let clothing_tax_percent := 0.1
  let total_tax_percent := 0.1
  tax_on_other_items total_spent clothing_tax_percent total_tax_percent = 20 := by
  sorry

end NUMINAMATH_GPT_jill_tax_on_other_items_l459_45908


namespace NUMINAMATH_GPT_number_534n_divisible_by_12_l459_45919

theorem number_534n_divisible_by_12 (n : ℕ) : (5340 + n) % 12 = 0 ↔ n = 0 := by sorry

end NUMINAMATH_GPT_number_534n_divisible_by_12_l459_45919


namespace NUMINAMATH_GPT_smallest_fraction_division_l459_45961

theorem smallest_fraction_division (a b : ℕ) (h_coprime : Nat.gcd a b = 1) 
(h1 : ∃ n, (25 * a = n * 21 * b)) (h2 : ∃ m, (15 * a = m * 14 * b)) : (a = 42) ∧ (b = 5) := 
sorry

end NUMINAMATH_GPT_smallest_fraction_division_l459_45961


namespace NUMINAMATH_GPT_find_value_of_N_l459_45903

theorem find_value_of_N 
  (N : ℝ) 
  (h : (20 / 100) * N = (30 / 100) * 2500) 
  : N = 3750 := 
sorry

end NUMINAMATH_GPT_find_value_of_N_l459_45903


namespace NUMINAMATH_GPT_coordinates_P_correct_l459_45952

noncomputable def coordinates_of_P : ℝ × ℝ :=
  let x_distance_to_y_axis : ℝ := 5
  let y_distance_to_x_axis : ℝ := 4
  -- x-coordinate must be negative, y-coordinate must be positive
  let x_coord : ℝ := -x_distance_to_y_axis
  let y_coord : ℝ := y_distance_to_x_axis
  (x_coord, y_coord)

theorem coordinates_P_correct:
  coordinates_of_P = (-5, 4) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_P_correct_l459_45952


namespace NUMINAMATH_GPT_exists_finite_group_with_normal_subgroup_GT_Aut_l459_45978

noncomputable def finite_group_G (n : ℕ) : Type := sorry -- Specific construction details omitted
noncomputable def normal_subgroup_H (n : ℕ) : Type := sorry -- Specific construction details omitted

def Aut_G (n : ℕ) : ℕ := sorry -- Number of automorphisms of G
def Aut_H (n : ℕ) : ℕ := sorry -- Number of automorphisms of H

theorem exists_finite_group_with_normal_subgroup_GT_Aut (n : ℕ) :
  ∃ G H, finite_group_G n = G ∧ normal_subgroup_H n = H ∧ Aut_H n > Aut_G n := sorry

end NUMINAMATH_GPT_exists_finite_group_with_normal_subgroup_GT_Aut_l459_45978


namespace NUMINAMATH_GPT_parabola_axis_of_symmetry_range_l459_45935

theorem parabola_axis_of_symmetry_range
  (a b c m n t : ℝ)
  (h₀ : 0 < a)
  (h₁ : m = a * 1^2 + b * 1 + c)
  (h₂ : n = a * 3^2 + b * 3 + c)
  (h₃ : m < n)
  (h₄ : n < c)
  (h_t : t = -b / (2 * a)) :
  (3 / 2) < t ∧ t < 2 :=
sorry

end NUMINAMATH_GPT_parabola_axis_of_symmetry_range_l459_45935


namespace NUMINAMATH_GPT_total_people_present_l459_45916

theorem total_people_present (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 14) :
  A + B = 8 :=
sorry

end NUMINAMATH_GPT_total_people_present_l459_45916


namespace NUMINAMATH_GPT_find_m_l459_45939

theorem find_m (m : ℕ) (h1 : (3 * m - 7) % 2 = 0) (h2 : 3 * m - 7 < 0) : m = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l459_45939


namespace NUMINAMATH_GPT_remainder_of_14_pow_53_mod_7_l459_45911

theorem remainder_of_14_pow_53_mod_7 : (14 ^ 53) % 7 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_14_pow_53_mod_7_l459_45911


namespace NUMINAMATH_GPT_factorize_cubic_l459_45901

theorem factorize_cubic (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  sorry

end NUMINAMATH_GPT_factorize_cubic_l459_45901


namespace NUMINAMATH_GPT_line_does_not_pass_through_fourth_quadrant_l459_45970

-- Definitions of conditions
variables {a b c x y : ℝ}

-- The mathematical statement to be proven
theorem line_does_not_pass_through_fourth_quadrant
  (h1 : a * b < 0) (h2 : b * c < 0) :
  ¬ (∃ x y, x > 0 ∧ y < 0 ∧ a * x + b * y + c = 0) :=
sorry

end NUMINAMATH_GPT_line_does_not_pass_through_fourth_quadrant_l459_45970


namespace NUMINAMATH_GPT_time_to_empty_tank_by_leakage_l459_45962

theorem time_to_empty_tank_by_leakage (R_t R_l : ℝ) (h1 : R_t = 1 / 12) (h2 : R_t - R_l = 1 / 18) :
  (1 / R_l) = 36 :=
by
  sorry

end NUMINAMATH_GPT_time_to_empty_tank_by_leakage_l459_45962


namespace NUMINAMATH_GPT_cost_per_pound_of_sausages_l459_45951

/-- Jake buys 2-pound packages of sausages. He buys 3 packages. He pays $24. 
To find the cost per pound of sausages. --/
theorem cost_per_pound_of_sausages 
  (pkg_weight : ℕ) 
  (num_pkg : ℕ) 
  (total_cost : ℕ) 
  (cost_per_pound : ℕ) 
  (h_pkg_weight : pkg_weight = 2) 
  (h_num_pkg : num_pkg = 3) 
  (h_total_cost : total_cost = 24) 
  (h_total_weight : num_pkg * pkg_weight = 6) :
  total_cost / (num_pkg * pkg_weight) = cost_per_pound :=
sorry

end NUMINAMATH_GPT_cost_per_pound_of_sausages_l459_45951


namespace NUMINAMATH_GPT_evaluate_polynomial_l459_45900

theorem evaluate_polynomial
  (x : ℝ)
  (h1 : x^2 - 3 * x - 9 = 0)
  (h2 : 0 < x)
  : x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = 8 :=
sorry

end NUMINAMATH_GPT_evaluate_polynomial_l459_45900


namespace NUMINAMATH_GPT_max_value_expression_l459_45943

theorem max_value_expression (a b c : ℝ) 
  (ha : 300 ≤ a ∧ a ≤ 500) 
  (hb : 500 ≤ b ∧ b ≤ 1500) 
  (hc : c = 100) : 
  (∃ M, M = 8 ∧ ∀ x, x = (b + c) / (a - c) → x ≤ M) := 
sorry

end NUMINAMATH_GPT_max_value_expression_l459_45943


namespace NUMINAMATH_GPT_three_bodies_with_triangle_front_view_l459_45953

def has_triangle_front_view (b : Type) : Prop :=
  -- Placeholder definition for example purposes
  sorry

theorem three_bodies_with_triangle_front_view :
  ∃ (body1 body2 body3 : Type),
  has_triangle_front_view body1 ∧
  has_triangle_front_view body2 ∧
  has_triangle_front_view body3 :=
sorry

end NUMINAMATH_GPT_three_bodies_with_triangle_front_view_l459_45953


namespace NUMINAMATH_GPT_merchant_salt_mixture_l459_45983

theorem merchant_salt_mixture (x : ℝ) (h₀ : (0.48 * (40 + x)) = 1.20 * (14 + 0.50 * x)) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_merchant_salt_mixture_l459_45983


namespace NUMINAMATH_GPT_simplify_fraction_l459_45994

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l459_45994


namespace NUMINAMATH_GPT_ratio_of_areas_of_two_concentric_circles_l459_45941

theorem ratio_of_areas_of_two_concentric_circles
  (C₁ C₂ : ℝ)
  (h1 : ∀ θ₁ θ₂, θ₁ = 30 ∧ θ₂ = 24 →
      (θ₁ / 360) * C₁ = (θ₂ / 360) * C₂):
  (C₁ / C₂) ^ 2 = (16 / 25) := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_two_concentric_circles_l459_45941


namespace NUMINAMATH_GPT_relationship_between_sets_l459_45971

def M : Set ℤ := {x | ∃ k : ℤ, x = 5 * k - 2}
def P : Set ℤ := {x | ∃ n : ℤ, x = 5 * n + 3}
def S : Set ℤ := {x | ∃ m : ℤ, x = 10 * m + 3}

theorem relationship_between_sets : S ⊆ P ∧ P = M := by
  sorry

end NUMINAMATH_GPT_relationship_between_sets_l459_45971


namespace NUMINAMATH_GPT_system_real_solution_conditions_l459_45929

theorem system_real_solution_conditions (a b c x y z : ℝ) (h1 : a * x + b * y = c * z) (h2 : a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) = c * Real.sqrt (1 - z^2)) :
  abs a ≤ abs b + abs c ∧ abs b ≤ abs a + abs c ∧ abs c ≤ abs a + abs b ∧
  (a * b >= 0 ∨ a * c >= 0 ∨ b * c >= 0) :=
sorry

end NUMINAMATH_GPT_system_real_solution_conditions_l459_45929


namespace NUMINAMATH_GPT_jen_lisa_spent_l459_45944

theorem jen_lisa_spent (J L : ℝ) 
  (h1 : L = 0.8 * J) 
  (h2 : J = L + 15) : 
  J + L = 135 := 
by
  sorry

end NUMINAMATH_GPT_jen_lisa_spent_l459_45944


namespace NUMINAMATH_GPT_shiela_used_seven_colors_l459_45925

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ)
  (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : (total_blocks / blocks_per_color) = 7 :=
by
  sorry

end NUMINAMATH_GPT_shiela_used_seven_colors_l459_45925


namespace NUMINAMATH_GPT_Danica_additional_cars_l459_45984

theorem Danica_additional_cars (n : ℕ) (row_size : ℕ) (danica_cars : ℕ) (answer : ℕ) :
  row_size = 8 →
  danica_cars = 37 →
  answer = 3 →
  ∃ k : ℕ, (k + danica_cars) % row_size = 0 ∧ k = answer :=
by
  sorry

end NUMINAMATH_GPT_Danica_additional_cars_l459_45984


namespace NUMINAMATH_GPT_minimum_value_of_f_l459_45972

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_value_of_f : ∃ x : ℝ, f x = 4 ∧ ∀ y : ℝ, f y ≥ 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_of_f_l459_45972


namespace NUMINAMATH_GPT_solve_equation_l459_45904

open Real

theorem solve_equation :
  ∀ x : ℝ, (
    (1 / ((x - 2) * (x - 3))) +
    (1 / ((x - 3) * (x - 4))) +
    (1 / ((x - 4) * (x - 5))) = (1 / 12)
  ) ↔ (x = 5 + sqrt 19 ∨ x = 5 - sqrt 19) := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l459_45904


namespace NUMINAMATH_GPT_initial_cards_collected_l459_45923

  -- Ralph collects some cards.
  variable (initial_cards: ℕ)

  -- Ralph's father gives Ralph 8 more cards.
  variable (added_cards: ℕ := 8)

  -- Now Ralph has 12 cards.
  variable (total_cards: ℕ := 12)

  -- Proof statement: Prove that the initial number of cards Ralph collected plus 8 equals 12.
  theorem initial_cards_collected: initial_cards + added_cards = total_cards := by
    sorry
  
end NUMINAMATH_GPT_initial_cards_collected_l459_45923


namespace NUMINAMATH_GPT_rhombus_area_l459_45969

theorem rhombus_area (side diagonal₁ : ℝ) (h_side : side = 20) (h_diagonal₁ : diagonal₁ = 16) : 
  ∃ (diagonal₂ : ℝ), (2 * diagonal₂ * diagonal₂ + 8 * 8 = side * side) ∧ 
  (1 / 2 * diagonal₁ * diagonal₂ = 64 * Real.sqrt 21) := by
  sorry

end NUMINAMATH_GPT_rhombus_area_l459_45969


namespace NUMINAMATH_GPT_min_sin_cos_sixth_power_l459_45996

noncomputable def min_value_sin_cos_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) / 12

theorem min_sin_cos_sixth_power :
  ∃ x : ℝ, (∀ y, (Real.sin y) ^ 6 + 2 * (Real.cos y) ^ 6 ≥ min_value_sin_cos_expr) ∧ 
            ((Real.sin x) ^ 6 + 2 * (Real.cos x) ^ 6 = min_value_sin_cos_expr) :=
sorry

end NUMINAMATH_GPT_min_sin_cos_sixth_power_l459_45996


namespace NUMINAMATH_GPT_total_cotton_yield_l459_45997

variables {m n a b : ℕ}

theorem total_cotton_yield (m n a b : ℕ) : 
  m * a + n * b = m * a + n * b := by
  sorry

end NUMINAMATH_GPT_total_cotton_yield_l459_45997


namespace NUMINAMATH_GPT_positive_integer_condition_l459_45910

theorem positive_integer_condition (x : ℝ) (hx : x ≠ 0) : 
  (∃ (n : ℤ), n > 0 ∧ (abs (x - abs x + 2) / x) = n) ↔ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_condition_l459_45910


namespace NUMINAMATH_GPT_LitterPatrol_pickup_l459_45930

theorem LitterPatrol_pickup :
  ∃ n : ℕ, n = 10 + 8 :=
sorry

end NUMINAMATH_GPT_LitterPatrol_pickup_l459_45930


namespace NUMINAMATH_GPT_latoya_call_duration_l459_45940

theorem latoya_call_duration
  (initial_credit remaining_credit : ℝ) (cost_per_minute : ℝ) (t : ℝ)
  (h1 : initial_credit = 30)
  (h2 : remaining_credit = 26.48)
  (h3 : cost_per_minute = 0.16)
  (h4 : initial_credit - remaining_credit = t * cost_per_minute) :
  t = 22 := 
sorry

end NUMINAMATH_GPT_latoya_call_duration_l459_45940


namespace NUMINAMATH_GPT_final_price_of_purchases_l459_45981

theorem final_price_of_purchases :
  let electronic_discount := 0.20
  let clothing_discount := 0.15
  let bundle_discount := 10
  let voucher_threshold := 200
  let voucher_value := 20
  let voucher_limit := 2
  let delivery_charge := 15
  let tax_rate := 0.08

  let electronic_original_price := 150
  let clothing_original_price := 80
  let num_clothing := 2

  -- Calculate discounts
  let electronic_discount_amount := electronic_original_price * electronic_discount
  let electronic_discount_price := electronic_original_price - electronic_discount_amount
  let clothing_discount_amount := clothing_original_price * clothing_discount
  let clothing_discount_price := clothing_original_price - clothing_discount_amount

  -- Sum of discounted clothing items
  let total_clothing_discount_price := clothing_discount_price * num_clothing

  -- Calculate bundle discount
  let total_before_bundle_discount := electronic_discount_price + total_clothing_discount_price
  let total_after_bundle_discount := total_before_bundle_discount - bundle_discount

  -- Calculate vouchers
  let num_vouchers := if total_after_bundle_discount >= voucher_threshold * 2 then voucher_limit else 
                      if total_after_bundle_discount >= voucher_threshold then 1 else 0
  let total_voucher_amount := num_vouchers * voucher_value
  let total_after_voucher_discount := total_after_bundle_discount - total_voucher_amount

  -- Add delivery charge
  let total_before_tax := total_after_voucher_discount + delivery_charge

  -- Calculate tax
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount

  final_price = 260.28 :=
by
  -- the actual proof will be included here
  sorry

end NUMINAMATH_GPT_final_price_of_purchases_l459_45981


namespace NUMINAMATH_GPT_total_boxes_sold_l459_45946

-- Define the number of boxes of plain cookies
def P : ℝ := 793.375

-- Define the combined value of cookies sold
def total_value : ℝ := 1586.75

-- Define the cost per box of each type of cookie
def cost_chocolate_chip : ℝ := 1.25
def cost_plain : ℝ := 0.75

-- State the theorem to prove
theorem total_boxes_sold :
  ∃ C : ℝ, cost_chocolate_chip * C + cost_plain * P = total_value ∧ C + P = 1586.75 :=
by
  sorry

end NUMINAMATH_GPT_total_boxes_sold_l459_45946


namespace NUMINAMATH_GPT_arithmetic_sequence_30th_term_l459_45959

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem arithmetic_sequence_30th_term :
  arithmetic_sequence 3 6 30 = 177 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_30th_term_l459_45959


namespace NUMINAMATH_GPT_find_circle_equation_l459_45977

noncomputable def circle_equation (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) = (-1, 3) ∨ (x, y) = (0, 0) ∨ (x, y) = (0, 2) →
  x^2 + y^2 + D * x + E * y + F = 0

theorem find_circle_equation :
  ∃ D E F : ℝ, circle_equation D E F ∧
               (∀ x y, x^2 + y^2 + D * x + E * y + F = x^2 + y^2 + 4 * x - 2 * y) :=
sorry

end NUMINAMATH_GPT_find_circle_equation_l459_45977


namespace NUMINAMATH_GPT_weather_forecast_minutes_l459_45933

theorem weather_forecast_minutes 
  (total_duration : ℕ) 
  (national_news : ℕ) 
  (international_news : ℕ) 
  (sports : ℕ) 
  (advertising : ℕ) 
  (wf : ℕ) :
  total_duration = 30 →
  national_news = 12 →
  international_news = 5 →
  sports = 5 →
  advertising = 6 →
  total_duration - (national_news + international_news + sports + advertising) = wf →
  wf = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_weather_forecast_minutes_l459_45933


namespace NUMINAMATH_GPT_adults_not_wearing_blue_l459_45937

-- Conditions
def children : ℕ := 45
def adults : ℕ := children / 3
def adults_wearing_blue : ℕ := adults / 3

-- Theorem Statement
theorem adults_not_wearing_blue :
  adults - adults_wearing_blue = 10 :=
sorry

end NUMINAMATH_GPT_adults_not_wearing_blue_l459_45937


namespace NUMINAMATH_GPT_min_squared_sum_l459_45988

theorem min_squared_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  x^2 + y^2 + z^2 ≥ 9 := 
sorry

end NUMINAMATH_GPT_min_squared_sum_l459_45988


namespace NUMINAMATH_GPT_correct_meiosis_sequence_l459_45950

-- Define the events as types
inductive Event : Type
| Replication : Event
| Synapsis : Event
| Separation : Event
| Division : Event

-- Define options as lists of events
def option_A := [Event.Replication, Event.Synapsis, Event.Separation, Event.Division]
def option_B := [Event.Synapsis, Event.Replication, Event.Separation, Event.Division]
def option_C := [Event.Synapsis, Event.Replication, Event.Division, Event.Separation]
def option_D := [Event.Replication, Event.Separation, Event.Synapsis, Event.Division]

-- Define the theorem to be proved
theorem correct_meiosis_sequence : option_A = [Event.Replication, Event.Synapsis, Event.Separation, Event.Division] :=
by
  sorry

end NUMINAMATH_GPT_correct_meiosis_sequence_l459_45950


namespace NUMINAMATH_GPT_fraction_meaningful_l459_45960

theorem fraction_meaningful (x : ℝ) : (∃ z, z = 3 / (x - 4)) ↔ x ≠ 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l459_45960


namespace NUMINAMATH_GPT_saltwater_solution_l459_45958

theorem saltwater_solution (x : ℝ) (h1 : ∃ v : ℝ, v = x ∧ v * 0.2 = 0.20 * x)
(h2 : 3 / 4 * x = 3 / 4 * x)
(h3 : ∃ v' : ℝ, v' = 3 / 4 * x + 6 + 12)
(h4 : (0.20 * x + 12) / (3 / 4 * x + 18) = 1 / 3) : x = 120 :=
by 
  sorry

end NUMINAMATH_GPT_saltwater_solution_l459_45958


namespace NUMINAMATH_GPT_savings_l459_45902

def distance_each_way : ℕ := 150
def round_trip_distance : ℕ := 2 * distance_each_way
def rental_cost_first_option : ℕ := 50
def rental_cost_second_option : ℕ := 90
def gasoline_efficiency : ℕ := 15
def gasoline_cost_per_liter : ℚ := 0.90
def gasoline_needed_for_trip : ℚ := round_trip_distance / gasoline_efficiency
def total_gasoline_cost : ℚ := gasoline_needed_for_trip * gasoline_cost_per_liter
def total_cost_first_option : ℚ := rental_cost_first_option + total_gasoline_cost
def total_cost_second_option : ℚ := rental_cost_second_option

theorem savings : total_cost_second_option - total_cost_first_option = 22 := by
  sorry

end NUMINAMATH_GPT_savings_l459_45902


namespace NUMINAMATH_GPT_divisors_of_64n4_l459_45913

theorem divisors_of_64n4 (n : ℕ) (hn : 0 < n) (hdiv : ∃ d, d = (120 * n^3) ∧ d.divisors.card = 120) : (64 * n^4).divisors.card = 375 := 
by 
  sorry

end NUMINAMATH_GPT_divisors_of_64n4_l459_45913


namespace NUMINAMATH_GPT_odd_function_periodicity_l459_45934

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_periodicity (f_odd : ∀ x, f (-x) = -f x)
  (f_periodic : ∀ x, f (x + 2) = -f x) (f_val : f 1 = 2) : f 2011 = -2 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_periodicity_l459_45934


namespace NUMINAMATH_GPT_black_to_white_ratio_l459_45990

theorem black_to_white_ratio (initial_black initial_white new_black new_white : ℕ) 
  (h1 : initial_black = 7) (h2 : initial_white = 18)
  (h3 : new_black = 31) (h4 : new_white = 18) :
  (new_black : ℚ) / new_white = 31 / 18 :=
by
  sorry

end NUMINAMATH_GPT_black_to_white_ratio_l459_45990


namespace NUMINAMATH_GPT_tenth_square_tiles_more_than_ninth_l459_45936

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := 2 * n - 1

-- Calculate the number of tiles used in the nth square
def tiles_count (n : ℕ) : ℕ := (side_length n) ^ 2

-- State the theorem that the tenth square requires 72 more tiles than the ninth square
theorem tenth_square_tiles_more_than_ninth : tiles_count 10 - tiles_count 9 = 72 :=
by
  sorry

end NUMINAMATH_GPT_tenth_square_tiles_more_than_ninth_l459_45936


namespace NUMINAMATH_GPT_contestant_wins_probability_l459_45914

section RadioProgramQuiz
  -- Defining the conditions
  def number_of_questions : ℕ := 4
  def number_of_choices_per_question : ℕ := 3
  def probability_of_correct_answer : ℚ := 1 / 3
  
  -- Defining the target probability
  def winning_probability : ℚ := 1 / 9

  -- The theorem
  theorem contestant_wins_probability :
    (let p := probability_of_correct_answer
     let p_correct_all := p^4
     let p_correct_three :=
       4 * (p^3 * (1 - p))
     p_correct_all + p_correct_three = winning_probability) :=
    sorry
end RadioProgramQuiz

end NUMINAMATH_GPT_contestant_wins_probability_l459_45914


namespace NUMINAMATH_GPT_A_share_in_profit_l459_45980

def investment_A := 6300
def investment_B := 4200
def investment_C := 10500
def total_profit := 12500

def total_investment := investment_A + investment_B + investment_C
def A_ratio := investment_A / total_investment

theorem A_share_in_profit : (total_profit * A_ratio) = 3750 := by
  sorry

end NUMINAMATH_GPT_A_share_in_profit_l459_45980


namespace NUMINAMATH_GPT_ratio_of_a_to_b_l459_45998

theorem ratio_of_a_to_b 
  (b c d a : ℚ)
  (h1 : b / c = 13 / 9)
  (h2 : c / d = 5 / 13)
  (h3 : a / d = 1 / 7.2) :
  a / b = 1 / 4 := 
by sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_l459_45998


namespace NUMINAMATH_GPT_charitable_woman_l459_45942

theorem charitable_woman (initial_pennies : ℕ) 
  (farmer_share : ℕ) (beggar_share : ℕ) (boy_share : ℕ) (left_pennies : ℕ) 
  (h1 : initial_pennies = 42)
  (h2 : farmer_share = (initial_pennies / 2 + 1))
  (h3 : beggar_share = ((initial_pennies - farmer_share) / 2 + 2))
  (h4 : boy_share = ((initial_pennies - farmer_share - beggar_share) / 2 + 3))
  (h5 : left_pennies = initial_pennies - farmer_share - beggar_share - boy_share) : 
  left_pennies = 1 :=
by
  sorry

end NUMINAMATH_GPT_charitable_woman_l459_45942


namespace NUMINAMATH_GPT_triangle_is_isosceles_l459_45938

variable (a b c : ℝ)
variable (h : a^2 - b * c = a * (b - c))

theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - b * c = a * (b - c)) : a = b ∨ b = c ∨ c = a := by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l459_45938


namespace NUMINAMATH_GPT_XiaoMingAgeWhenFathersAgeIsFiveTimes_l459_45917

-- Define the conditions
def XiaoMingAgeCurrent : ℕ := 12
def FatherAgeCurrent : ℕ := 40

-- Prove the question given the conditions
theorem XiaoMingAgeWhenFathersAgeIsFiveTimes : 
  ∃ (x : ℕ), (FatherAgeCurrent - x) = 5 * x - XiaoMingAgeCurrent ∧ x = 7 := 
by
  use 7
  sorry

end NUMINAMATH_GPT_XiaoMingAgeWhenFathersAgeIsFiveTimes_l459_45917


namespace NUMINAMATH_GPT_initial_boys_count_l459_45987

theorem initial_boys_count (b : ℕ) (h1 : b + 10 - 4 - 3 = 17) : b = 14 :=
by
  sorry

end NUMINAMATH_GPT_initial_boys_count_l459_45987


namespace NUMINAMATH_GPT_only_integer_triplet_solution_l459_45957

theorem only_integer_triplet_solution 
  (a b c : ℤ) : 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_only_integer_triplet_solution_l459_45957


namespace NUMINAMATH_GPT_product_is_approximately_9603_l459_45922

noncomputable def smaller_number : ℝ := 97.49871794028884
noncomputable def successive_number : ℝ := smaller_number + 1
noncomputable def product_of_numbers : ℝ := smaller_number * successive_number

theorem product_is_approximately_9603 : abs (product_of_numbers - 9603) < 10e-3 := 
sorry

end NUMINAMATH_GPT_product_is_approximately_9603_l459_45922


namespace NUMINAMATH_GPT_smallest_solution_l459_45907

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_smallest_solution_l459_45907


namespace NUMINAMATH_GPT_exists_sum_coprime_seventeen_not_sum_coprime_l459_45928

/-- 
 For any integer \( n \) where \( n > 17 \), there exist integers \( a \) and \( b \) 
 such that \( n = a + b \), \( a > 1 \), \( b > 1 \), and \( \gcd(a, b) = 1 \).
 Additionally, the integer 17 does not have this property.
-/
theorem exists_sum_coprime (n : ℤ) (h : n > 17) : 
  ∃ (a b : ℤ), n = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

/-- 
 The integer 17 cannot be expressed as the sum of two integers greater than 1 
 that are coprime.
-/
theorem seventeen_not_sum_coprime : 
  ¬ ∃ (a b : ℤ), 17 = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

end NUMINAMATH_GPT_exists_sum_coprime_seventeen_not_sum_coprime_l459_45928


namespace NUMINAMATH_GPT_sum_abc_l459_45973

theorem sum_abc (a b c: ℝ) 
  (h1 : ∃ x: ℝ, x^2 + a * x + 1 = 0 ∧ x^2 + b * x + c = 0)
  (h2 : ∃ x: ℝ, x^2 + x + a = 0 ∧ x^2 + c * x + b = 0) :
  a + b + c = -3 := 
sorry

end NUMINAMATH_GPT_sum_abc_l459_45973


namespace NUMINAMATH_GPT_valid_triangle_side_l459_45931

theorem valid_triangle_side (x : ℕ) (h_pos : 0 < x) (h1 : x + 6 > 15) (h2 : 21 > x) :
  10 ≤ x ∧ x ≤ 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_valid_triangle_side_l459_45931


namespace NUMINAMATH_GPT_james_hives_l459_45955

-- Define all conditions
def hive_honey : ℕ := 20  -- Each hive produces 20 liters of honey
def jar_capacity : ℕ := 1/2  -- Each jar holds 0.5 liters
def jars_needed : ℕ := 100  -- James needs 100 jars for half the honey

-- Translate to Lean statement
theorem james_hives (hive_honey jar_capacity jars_needed : ℕ) :
  (hive_honey = 20) → 
  (jar_capacity = 1 / 2) →
  (jars_needed = 100) →
  (∀ hives : ℕ, (hives * hive_honey = 200) → hives = 5) :=
by
  intros Hhoney Hjar Hjars
  intros hives Hprod
  sorry

end NUMINAMATH_GPT_james_hives_l459_45955


namespace NUMINAMATH_GPT_average_speed_for_trip_l459_45963

theorem average_speed_for_trip 
  (Speed1 Speed2 : ℝ) 
  (AverageSpeed : ℝ) 
  (h1 : Speed1 = 110) 
  (h2 : Speed2 = 72) 
  (h3 : AverageSpeed = (2 * Speed1 * Speed2) / (Speed1 + Speed2)) :
  AverageSpeed = 87 := 
by
  -- solution steps would go here
  sorry

end NUMINAMATH_GPT_average_speed_for_trip_l459_45963


namespace NUMINAMATH_GPT_hypotenuse_length_l459_45979

theorem hypotenuse_length (a b c : ℕ) (h1 : a = 12) (h2 : b = 5) (h3 : c^2 = a^2 + b^2) : c = 13 := by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l459_45979


namespace NUMINAMATH_GPT_function_identity_l459_45975

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n)
  (h3 : ∀ n : ℕ, f (n + 1) > f n) : 
  ∀ n : ℕ, f n = n :=
sorry

end NUMINAMATH_GPT_function_identity_l459_45975


namespace NUMINAMATH_GPT_percentage_four_petals_l459_45906

def total_clovers : ℝ := 200
def percentage_three_petals : ℝ := 0.75
def percentage_two_petals : ℝ := 0.24
def earnings : ℝ := 554 -- cents

theorem percentage_four_petals :
  (total_clovers - (percentage_three_petals * total_clovers + percentage_two_petals * total_clovers)) / total_clovers * 100 = 1 := 
by sorry

end NUMINAMATH_GPT_percentage_four_petals_l459_45906


namespace NUMINAMATH_GPT_dealer_purchase_fraction_l459_45920

theorem dealer_purchase_fraction (P C : ℝ) (h1 : ∃ S, S = 1.5 * P) (h2 : ∃ S, S = 2 * C) :
  C / P = 3 / 8 :=
by
  -- The statement of the theorem has been generated based on the problem conditions.
  sorry

end NUMINAMATH_GPT_dealer_purchase_fraction_l459_45920


namespace NUMINAMATH_GPT_segment_length_BD_eq_CB_l459_45986

theorem segment_length_BD_eq_CB {AC CB BD x : ℝ}
  (h1 : AC = 4 * CB)
  (h2 : BD = CB)
  (h3 : CB = x) :
  BD = CB := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_segment_length_BD_eq_CB_l459_45986


namespace NUMINAMATH_GPT_miles_traveled_correct_l459_45964

def initial_odometer_reading := 212.3
def odometer_reading_at_lunch := 372.0
def miles_traveled := odometer_reading_at_lunch - initial_odometer_reading

theorem miles_traveled_correct : miles_traveled = 159.7 :=
by
  sorry

end NUMINAMATH_GPT_miles_traveled_correct_l459_45964


namespace NUMINAMATH_GPT_probability_of_different_colors_is_correct_l459_45905

noncomputable def probability_different_colors : ℚ :=
  let total_chips := 18
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let p_blue_then_not_blue := (blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)
  let p_red_then_not_red := (red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)
  let p_yellow_then_not_yellow := (yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)
  let p_green_then_not_green := (green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips)
  p_blue_then_not_blue + p_red_then_not_red + p_yellow_then_not_yellow + p_green_then_not_green

theorem probability_of_different_colors_is_correct :
  probability_different_colors = 119 / 162 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_different_colors_is_correct_l459_45905


namespace NUMINAMATH_GPT_sequence_formula_l459_45992

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n + 1) :
  (a 1 = 3) ∧ (∀ n, n ≥ 2 → a n = 2 * n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l459_45992


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l459_45915

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (x + Real.sqrt (x^2 + 1)) / Real.log 2

theorem sufficient_and_necessary_condition {a b : ℝ} (h : a + b ≥ 0) : f a + f b ≥ 0 :=
sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l459_45915


namespace NUMINAMATH_GPT_at_least_two_pairs_in_one_drawer_l459_45974

theorem at_least_two_pairs_in_one_drawer (n : ℕ) (hn : n > 0) : 
  ∃ i j : ℕ, i ≠ j ∧ i < n ∧ j < n :=
by {
  sorry
}

end NUMINAMATH_GPT_at_least_two_pairs_in_one_drawer_l459_45974


namespace NUMINAMATH_GPT_line_relation_in_perpendicular_planes_l459_45921

-- Let's define the notions of planes and lines being perpendicular/parallel
variables {α β : Plane} {a : Line}

def plane_perpendicular (α β : Plane) : Prop := sorry -- definition of perpendicular planes
def line_perpendicular_plane (a : Line) (β : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def line_parallel_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line being parallel to a plane
def line_in_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line lying in a plane

-- The theorem stating the relationship given the conditions
theorem line_relation_in_perpendicular_planes 
  (h1 : plane_perpendicular α β) 
  (h2 : line_perpendicular_plane a β) : 
  line_parallel_plane a α ∨ line_in_plane a α :=
sorry

end NUMINAMATH_GPT_line_relation_in_perpendicular_planes_l459_45921


namespace NUMINAMATH_GPT_probability_of_X_conditioned_l459_45999

variables (P_X P_Y P_XY : ℝ)

-- Conditions
def probability_of_Y : Prop := P_Y = 2/5
def probability_of_XY : Prop := P_XY = 0.05714285714285714
def independent_selection : Prop := P_XY = P_X * P_Y

-- Theorem statement
theorem probability_of_X_conditioned (P_X P_Y P_XY : ℝ) 
  (h1 : probability_of_Y P_Y) 
  (h2 : probability_of_XY P_XY) 
  (h3 : independent_selection P_X P_Y P_XY) :
  P_X = 0.14285714285714285 := 
sorry

end NUMINAMATH_GPT_probability_of_X_conditioned_l459_45999


namespace NUMINAMATH_GPT_nate_search_time_l459_45966

def sectionG_rows : ℕ := 15
def sectionG_cars_per_row : ℕ := 10
def sectionH_rows : ℕ := 20
def sectionH_cars_per_row : ℕ := 9
def cars_per_minute : ℕ := 11

theorem nate_search_time :
  (sectionG_rows * sectionG_cars_per_row + sectionH_rows * sectionH_cars_per_row) / cars_per_minute = 30 :=
  by
    sorry

end NUMINAMATH_GPT_nate_search_time_l459_45966


namespace NUMINAMATH_GPT_circle_radius_l459_45949

theorem circle_radius (M N : ℝ) (hM : M = Real.pi * r ^ 2) (hN : N = 2 * Real.pi * r) (h : M / N = 15) : r = 30 := by
  sorry

end NUMINAMATH_GPT_circle_radius_l459_45949


namespace NUMINAMATH_GPT_adjacent_complementary_is_complementary_l459_45993

/-- Two angles are complementary if their sum is 90 degrees. -/
def complementary (α β : ℝ) : Prop :=
  α + β = 90

/-- Two angles are adjacent complementary if they are complementary and adjacent. -/
def adjacent_complementary (α β : ℝ) : Prop :=
  complementary α β ∧ α > 0 ∧ β > 0

/-- Prove that adjacent complementary angles are complementary. -/
theorem adjacent_complementary_is_complementary (α β : ℝ) : adjacent_complementary α β → complementary α β :=
by
  sorry

end NUMINAMATH_GPT_adjacent_complementary_is_complementary_l459_45993


namespace NUMINAMATH_GPT_smallest_value_expression_l459_45932

theorem smallest_value_expression
    (a b c : ℝ) 
    (h1 : c > b)
    (h2 : b > a)
    (h3 : c ≠ 0) : 
    ∃ z : ℝ, z = 0 ∧ z = (a + b)^2 / c^2 + (b - c)^2 / c^2 + (c - b)^2 / c^2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_expression_l459_45932


namespace NUMINAMATH_GPT_minimum_value_of_expression_l459_45948

theorem minimum_value_of_expression (x y z w : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h : 5 * w = 3 * x ∧ 5 * w = 4 * y ∧ 5 * w = 7 * z) : x - y + z - w = 11 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l459_45948


namespace NUMINAMATH_GPT_min_omega_condition_l459_45909

theorem min_omega_condition :
  ∃ (ω: ℝ) (k: ℤ), (ω > 0) ∧ (ω = 6 * k + 1 / 2) ∧ (∀ (ω' : ℝ), (ω' > 0) ∧ (∃ (k': ℤ), ω' = 6 * k' + 1 / 2) → ω ≤ ω') := 
sorry

end NUMINAMATH_GPT_min_omega_condition_l459_45909


namespace NUMINAMATH_GPT_initial_trucks_l459_45956

def trucks_given_to_Jeff : ℕ := 13
def trucks_left_with_Sarah : ℕ := 38

theorem initial_trucks (initial_trucks_count : ℕ) :
  initial_trucks_count = trucks_given_to_Jeff + trucks_left_with_Sarah → initial_trucks_count = 51 :=
by
  sorry

end NUMINAMATH_GPT_initial_trucks_l459_45956


namespace NUMINAMATH_GPT_is_prime_if_two_pow_n_minus_one_is_prime_is_power_of_two_if_two_pow_n_plus_one_is_prime_l459_45918

-- Problem 1: If \(2^{n} - 1\) is prime, then \(n\) is prime.
theorem is_prime_if_two_pow_n_minus_one_is_prime (n : ℕ) (hn : Prime (2^n - 1)) : Prime n :=
sorry

-- Problem 2: If \(2^{n} + 1\) is prime, then \(n\) is a power of 2.
theorem is_power_of_two_if_two_pow_n_plus_one_is_prime (n : ℕ) (hn : Prime (2^n + 1)) : ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_GPT_is_prime_if_two_pow_n_minus_one_is_prime_is_power_of_two_if_two_pow_n_plus_one_is_prime_l459_45918


namespace NUMINAMATH_GPT_calc_diagonal_of_rectangle_l459_45954

variable (a : ℕ) (A : ℕ)

theorem calc_diagonal_of_rectangle (h_a : a = 6) (h_A : A = 48) (H : a * a' = A) :
  ∃ d : ℕ, d = 10 :=
by
 sorry

end NUMINAMATH_GPT_calc_diagonal_of_rectangle_l459_45954


namespace NUMINAMATH_GPT_min_value_of_function_l459_45927

theorem min_value_of_function (h : 0 < x ∧ x < 1) : 
  ∃ (y : ℝ), (∀ z : ℝ, z = (4 / x + 1 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_function_l459_45927


namespace NUMINAMATH_GPT_part1_l459_45982

-- Define the vectors a and b
def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (2, -1)
-- Define the vectors a - x b and a - b
def vec1 (x : ℝ) : ℝ × ℝ := (a.1 - x * b.1, a.2 - x * b.2)
def vec2 : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
-- Define the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 

-- Main theorem: prove that the vectors being perpendicular implies x = -7/3
theorem part1 (x : ℝ) : dot_product (vec1 x) vec2 = 0 → x = -7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_l459_45982


namespace NUMINAMATH_GPT_fuel_tank_capacity_l459_45991

theorem fuel_tank_capacity
  (ethanol_A_ethanol : ∀ {x : Float}, x = 0.12 * 49.99999999999999)
  (ethanol_B_ethanol : ∀ {C : Float}, x = 0.16 * (C - 49.99999999999999))
  (total_ethanol : ∀ {C : Float}, 0.12 * 49.99999999999999 + 0.16 * (C - 49.99999999999999) = 30) :
  (C = 162.5) :=
sorry

end NUMINAMATH_GPT_fuel_tank_capacity_l459_45991


namespace NUMINAMATH_GPT_reduce_to_one_l459_45947

theorem reduce_to_one (n : ℕ) : ∃ k, (k = 1) :=
by
  sorry

end NUMINAMATH_GPT_reduce_to_one_l459_45947
