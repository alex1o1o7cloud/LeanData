import Mathlib

namespace minimum_value_l1451_145155

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + (1 / y)) * (x + (1 / y) - 1024) +
  (y + (1 / x)) * (y + (1 / x) - 1024) ≥ -524288 :=
by sorry

end minimum_value_l1451_145155


namespace y_intercept_of_line_l1451_145196

theorem y_intercept_of_line :
  ∃ y, (∀ x : ℝ, 2 * x - 3 * y = 6) ∧ (y = -2) :=
sorry

end y_intercept_of_line_l1451_145196


namespace y_intercept_of_line_l1451_145153

theorem y_intercept_of_line (x y : ℝ) : x + 2 * y + 6 = 0 → x = 0 → y = -3 :=
by
  sorry

end y_intercept_of_line_l1451_145153


namespace evaluate_x2_plus_y2_l1451_145190

theorem evaluate_x2_plus_y2 (x y : ℝ) (h₁ : 3 * x + 2 * y = 20) (h₂ : 4 * x + 2 * y = 26) : x^2 + y^2 = 37 := by
  sorry

end evaluate_x2_plus_y2_l1451_145190


namespace hiker_final_distance_l1451_145165

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

end hiker_final_distance_l1451_145165


namespace afb_leq_bfa_l1451_145100

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

end afb_leq_bfa_l1451_145100


namespace intersection_A_B_l1451_145179

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {-3, 1, 2, 4}

theorem intersection_A_B :
  A ∩ B = {-3, 1} := by
  sorry

end intersection_A_B_l1451_145179


namespace find_f_neg_2_l1451_145167

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

end find_f_neg_2_l1451_145167


namespace compound_percentage_increase_l1451_145154

noncomputable def weeklyEarningsAfterRaises (initial : ℝ) (raises : List ℝ) : ℝ :=
  raises.foldl (λ sal raise_rate => sal * (1 + raise_rate / 100)) initial

theorem compound_percentage_increase :
  let initial := 60
  let raises := [10, 15, 12, 8]
  weeklyEarningsAfterRaises initial raises = 91.80864 ∧
  ((weeklyEarningsAfterRaises initial raises - initial) / initial * 100 = 53.0144) :=
by
  sorry

end compound_percentage_increase_l1451_145154


namespace equivalent_terminal_angle_l1451_145132

theorem equivalent_terminal_angle :
  ∃ n : ℤ, 660 = n * 360 - 420 := 
by
  sorry

end equivalent_terminal_angle_l1451_145132


namespace shots_per_puppy_l1451_145191

-- Definitions
def num_pregnant_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def cost_per_shot : ℕ := 5
def total_shot_cost : ℕ := 120

-- Total number of puppies
def total_puppies : ℕ := num_pregnant_dogs * puppies_per_dog

-- Total number of shots
def total_shots : ℕ := total_shot_cost / cost_per_shot

-- The theorem to prove
theorem shots_per_puppy : total_shots / total_puppies = 2 :=
by
  sorry

end shots_per_puppy_l1451_145191


namespace nancy_other_albums_count_l1451_145138

-- Definitions based on the given conditions
def total_pictures : ℕ := 51
def pics_in_first_album : ℕ := 11
def pics_per_other_album : ℕ := 5

-- Theorem to prove the question's answer
theorem nancy_other_albums_count : 
  (total_pictures - pics_in_first_album) / pics_per_other_album = 8 := by
  sorry

end nancy_other_albums_count_l1451_145138


namespace f_zero_add_f_neg_three_l1451_145110

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_add (x y : ℝ) : f x + f y = f (x + y)

axiom f_three : f 3 = 4

theorem f_zero_add_f_neg_three : f 0 + f (-3) = -4 :=
by
  sorry

end f_zero_add_f_neg_three_l1451_145110


namespace susans_total_chairs_l1451_145174

def number_of_red_chairs := 5
def number_of_yellow_chairs := 4 * number_of_red_chairs
def number_of_blue_chairs := number_of_yellow_chairs - 2
def total_chairs := number_of_red_chairs + number_of_yellow_chairs + number_of_blue_chairs

theorem susans_total_chairs : total_chairs = 43 :=
by
  sorry

end susans_total_chairs_l1451_145174


namespace intersection_of_M_and_N_l1451_145129

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}
def intersection_M_N : Set ℕ := {0, 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := by
  sorry

end intersection_of_M_and_N_l1451_145129


namespace sqrt6_special_op_l1451_145111

-- Define the binary operation (¤) as given in the problem.
def special_op (x y : ℝ) : ℝ := (x + y) ^ 2 - (x - y) ^ 2

-- States that √6 ¤ √6 is equal to 24.
theorem sqrt6_special_op : special_op (Real.sqrt 6) (Real.sqrt 6) = 24 :=
by
  sorry

end sqrt6_special_op_l1451_145111


namespace find_number_l1451_145151

noncomputable def number := 115.2 / 0.32

theorem find_number : number = 360 := 
by
  sorry

end find_number_l1451_145151


namespace ant_weight_statement_l1451_145127

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

end ant_weight_statement_l1451_145127


namespace martha_bottles_l1451_145128

def total_bottles_left (a b c d : ℕ) : ℕ :=
  a + b + c - d

theorem martha_bottles : total_bottles_left 4 4 5 3 = 10 :=
by
  sorry

end martha_bottles_l1451_145128


namespace walls_painted_purple_l1451_145189

theorem walls_painted_purple :
  (10 - (3 * 10 / 5)) * 8 = 32 := by
  sorry

end walls_painted_purple_l1451_145189


namespace unsold_percentage_l1451_145177

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

end unsold_percentage_l1451_145177


namespace equation1_solution_equation2_solution_l1451_145112

theorem equation1_solution (x : ℚ) : 2 * (x - 3) = 1 - 3 * (x + 1) → x = 4 / 5 :=
by sorry

theorem equation2_solution (x : ℚ) : 3 * x + (x - 1) / 2 = 3 - (x - 1) / 3 → x = 1 :=
by sorry

end equation1_solution_equation2_solution_l1451_145112


namespace multiply_neg_reverse_inequality_l1451_145150

theorem multiply_neg_reverse_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end multiply_neg_reverse_inequality_l1451_145150


namespace tan_ratio_is_7_over_3_l1451_145103

open Real

theorem tan_ratio_is_7_over_3 (a b : ℝ) (h1 : sin (a + b) = 5 / 8) (h2 : sin (a - b) = 1 / 4) : (tan a / tan b) = 7 / 3 :=
by
  sorry

end tan_ratio_is_7_over_3_l1451_145103


namespace minimize_std_deviation_l1451_145113

theorem minimize_std_deviation (m n : ℝ) (h1 : m + n = 32) 
    (h2 : 11 ≤ 12 ∧ 12 ≤ m ∧ m ≤ n ∧ n ≤ 20 ∧ 20 ≤ 27) : 
    m = 16 :=
by {
  -- No proof required, only the theorem statement as per instructions
  sorry
}

end minimize_std_deviation_l1451_145113


namespace student_calls_out_2005th_l1451_145148

theorem student_calls_out_2005th : 
  ∀ (n : ℕ), n = 2005 → ∃ k : ℕ, k ∈ [1, 2, 3, 4, 3, 2, 1] ∧ k = 1 := 
by
  sorry

end student_calls_out_2005th_l1451_145148


namespace sum_of_reciprocals_of_root_products_eq_4_l1451_145120

theorem sum_of_reciprocals_of_root_products_eq_4
  (p q r s t : ℂ)
  (h_poly : ∀ x : ℂ, x^5 + 10*x^4 + 20*x^3 + 15*x^2 + 8*x + 5 = 0 ∨ (x - p)*(x - q)*(x - r)*(x - s)*(x - t) = 0)
  (h_vieta_2 : p*q + p*r + p*s + p*t + q*r + q*s + q*t + r*s + r*t + s*t = 20)
  (h_vieta_all : p*q*r*s*t = 5) :
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 4 := 
sorry

end sum_of_reciprocals_of_root_products_eq_4_l1451_145120


namespace value_of_a1_plus_a10_l1451_145186

noncomputable def geometric_sequence {α : Type*} [Field α] (a : ℕ → α) :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem value_of_a1_plus_a10 (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : a 4 + a 7 = 2) 
  (h3 : a 5 * a 6 = -8) 
  : a 1 + a 10 = -7 := 
by
  sorry

end value_of_a1_plus_a10_l1451_145186


namespace range_of_a_l1451_145116

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ Real.exp 1) := by
  sorry

end range_of_a_l1451_145116


namespace quadratic_real_roots_l1451_145142

theorem quadratic_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c ≥ 0) : ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end quadratic_real_roots_l1451_145142


namespace apples_left_l1451_145101

theorem apples_left (initial_apples : ℕ) (difference_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 46) 
  (h2 : difference_apples = 32) 
  (h3 : final_apples = initial_apples - difference_apples) : 
  final_apples = 14 := 
by
  rw [h1, h2] at h3
  exact h3

end apples_left_l1451_145101


namespace wayne_took_cards_l1451_145193

-- Let's define the problem context
variable (initial_cards : ℕ := 76)
variable (remaining_cards : ℕ := 17)

-- We need to show that Wayne took away 59 cards
theorem wayne_took_cards (x : ℕ) (h : x = initial_cards - remaining_cards) : x = 59 :=
by
  sorry

end wayne_took_cards_l1451_145193


namespace find_functions_l1451_145168

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

end find_functions_l1451_145168


namespace find_x0_l1451_145175

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^2 + c
noncomputable def int_f (a c : ℝ) : ℝ := ∫ x in (0 : ℝ)..1, f x a c

theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (hx0 : 0 ≤ x0 ∧ x0 ≤ 1)
  (h_eq : int_f a c = f x0 a c) : x0 = Real.sqrt 3 / 3 := sorry

end find_x0_l1451_145175


namespace runner_distance_l1451_145169

theorem runner_distance :
  ∃ x t d : ℕ,
    d = x * t ∧
    d = (x + 1) * (2 * t / 3) ∧
    d = (x - 1) * (t + 3) ∧
    d = 6 :=
by
  sorry

end runner_distance_l1451_145169


namespace christmas_sale_pricing_l1451_145160

theorem christmas_sale_pricing (a b : ℝ) : 
  (forall (c : ℝ), c = a * (3 / 5)) ∧ (forall (d : ℝ), d = b * (5 / 3)) :=
by
  sorry  -- proof goes here

end christmas_sale_pricing_l1451_145160


namespace rational_number_div_l1451_145152

theorem rational_number_div (x : ℚ) (h : -2 / x = 8) : x = -1 / 4 := 
by
  sorry

end rational_number_div_l1451_145152


namespace polynomial_equality_l1451_145135

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 5 * x + 7
noncomputable def g (x : ℝ) : ℝ := 12 * x^2 - 19 * x + 25

theorem polynomial_equality :
  f 3 = g 3 ∧ f (3 - Real.sqrt 3) = g (3 - Real.sqrt 3) ∧ f (3 + Real.sqrt 3) = g (3 + Real.sqrt 3) :=
by
  sorry

end polynomial_equality_l1451_145135


namespace small_pump_filling_time_l1451_145158

theorem small_pump_filling_time :
  ∃ S : ℝ, (L = 2) → 
         (1 / 0.4444444444444444 = S + L) → 
         (1 / S = 4) :=
by 
  sorry

end small_pump_filling_time_l1451_145158


namespace prob_A_not_losing_is_correct_l1451_145131

def prob_A_wins := 0.4
def prob_draw := 0.2
def prob_A_not_losing := 0.6

theorem prob_A_not_losing_is_correct : prob_A_wins + prob_draw = prob_A_not_losing :=
by sorry

end prob_A_not_losing_is_correct_l1451_145131


namespace count_4_digit_numbers_divisible_by_13_l1451_145198

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end count_4_digit_numbers_divisible_by_13_l1451_145198


namespace number_of_non_degenerate_rectangles_excluding_center_l1451_145164

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

end number_of_non_degenerate_rectangles_excluding_center_l1451_145164


namespace rect_solution_proof_l1451_145134

noncomputable def rect_solution_exists : Prop :=
  ∃ (l2 w2 : ℝ), 2 * (l2 + w2) = 12 ∧ l2 * w2 = 4 ∧
               l2 = 3 + Real.sqrt 5 ∧ w2 = 3 - Real.sqrt 5

theorem rect_solution_proof : rect_solution_exists :=
  by
    sorry

end rect_solution_proof_l1451_145134


namespace not_cheap_is_necessary_condition_l1451_145187

-- Define propositions for "good quality" and "not cheap"
variables {P: Prop} {Q: Prop} 

-- Statement "You get what you pay for" implies "good quality is not cheap"
axiom H : P → Q 

-- The proof problem
theorem not_cheap_is_necessary_condition (H : P → Q) : Q → P :=
by sorry

end not_cheap_is_necessary_condition_l1451_145187


namespace find_n_l1451_145143

theorem find_n (P s k m n : ℝ) (h : P = s / (1 + k + m) ^ n) :
  n = (Real.log (s / P)) / (Real.log (1 + k + m)) :=
sorry

end find_n_l1451_145143


namespace radius_of_circle_B_l1451_145123

theorem radius_of_circle_B (diam_A : ℝ) (factor : ℝ) (r_A r_B : ℝ) 
  (h1 : diam_A = 80) 
  (h2 : r_A = diam_A / 2) 
  (h3 : r_A = factor * r_B) 
  (h4 : factor = 4) : r_B = 10 := 
by 
  sorry

end radius_of_circle_B_l1451_145123


namespace increment_in_radius_l1451_145144

theorem increment_in_radius (C1 C2 : ℝ) (hC1 : C1 = 50) (hC2 : C2 = 60) : 
  ((C2 / (2 * Real.pi)) - (C1 / (2 * Real.pi)) = (5 / Real.pi)) :=
by
  sorry

end increment_in_radius_l1451_145144


namespace perimeter_of_triangle_l1451_145163

theorem perimeter_of_triangle
  {r A P : ℝ} (hr : r = 2.5) (hA : A = 25) :
  P = 20 :=
by
  sorry

end perimeter_of_triangle_l1451_145163


namespace tan_alpha_parallel_vectors_l1451_145115

theorem tan_alpha_parallel_vectors
    (α : ℝ)
    (a : ℝ × ℝ := (6, 8))
    (b : ℝ × ℝ := (Real.sin α, Real.cos α))
    (h : a.fst * b.snd = a.snd * b.fst) :
    Real.tan α = 3 / 4 := 
sorry

end tan_alpha_parallel_vectors_l1451_145115


namespace fraction_equality_l1451_145117

theorem fraction_equality (a b : ℚ) (h₁ : a = 1/2) (h₂ : b = 2/3) : 
    (6 * a + 18 * b) / (12 * a + 6 * b) = 3 / 2 := by
  sorry

end fraction_equality_l1451_145117


namespace total_pages_read_l1451_145162

theorem total_pages_read (J A C D : ℝ) 
  (hJ : J = 20)
  (hA : A = 2 * J + 2)
  (hC : C = J * A - 17)
  (hD : D = (C + J) / 2) :
  J + A + C + D = 1306.5 :=
by
  sorry

end total_pages_read_l1451_145162


namespace hyperbola_eccentricity_l1451_145166

theorem hyperbola_eccentricity (C : Type) (a b c e : ℝ)
  (h_asymptotes : ∀ x : ℝ, (∃ y : ℝ, y = x ∨ y = -x)) :
  a = b ∧ c = Real.sqrt (a^2 + b^2) ∧ e = c / a → e = Real.sqrt 2 := 
by
  sorry

end hyperbola_eccentricity_l1451_145166


namespace trader_profit_l1451_145109

-- Definitions and conditions
def original_price (P : ℝ) := P
def discounted_price (P : ℝ) := 0.70 * P
def marked_up_price (P : ℝ) := 0.84 * P
def sale_price (P : ℝ) := 0.714 * P
def final_price (P : ℝ) := 1.2138 * P

-- Proof statement
theorem trader_profit (P : ℝ) : ((final_price P - original_price P) / original_price P) * 100 = 21.38 := by
  sorry

end trader_profit_l1451_145109


namespace Kat_training_hours_l1451_145105

theorem Kat_training_hours
  (h_strength_times : ℕ)
  (h_strength_hours : ℝ)
  (h_boxing_times : ℕ)
  (h_boxing_hours : ℝ)
  (h_times : h_strength_times = 3)
  (h_strength : h_strength_hours = 1)
  (b_times : h_boxing_times = 4)
  (b_hours : h_boxing_hours = 1.5) :
  h_strength_times * h_strength_hours + h_boxing_times * h_boxing_hours = 9 :=
by
  sorry

end Kat_training_hours_l1451_145105


namespace largest_triangle_perimeter_l1451_145180

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem largest_triangle_perimeter : 
  ∃ (x : ℕ), x ≤ 14 ∧ 2 ≤ x ∧ is_valid_triangle 7 8 x ∧ (7 + 8 + x = 29) :=
sorry

end largest_triangle_perimeter_l1451_145180


namespace intersection_of_lines_l1451_145118

theorem intersection_of_lines : ∃ (x y : ℝ), 9 * x - 4 * y = 6 ∧ 7 * x + y = 17 ∧ (x, y) = (2, 3) := 
by
  sorry

end intersection_of_lines_l1451_145118


namespace total_cost_is_correct_l1451_145141

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

end total_cost_is_correct_l1451_145141


namespace circle_center_l1451_145133

theorem circle_center (x y : ℝ) :
  x^2 + y^2 - 2*x + 4*y + 1 = 0 → (1, -2) = (1, -2) :=
by
  sorry

end circle_center_l1451_145133


namespace inequality_proof_l1451_145194

theorem inequality_proof (a b c d : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a + b = 2 → c + d = 2 → 
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := 
by 
  intros ha hb hc hd hab hcd
  sorry

end inequality_proof_l1451_145194


namespace rachel_biology_homework_pages_l1451_145146

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

end rachel_biology_homework_pages_l1451_145146


namespace simplify_fraction_l1451_145119

theorem simplify_fraction (a b : ℝ) (h : a ≠ b): 
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) :=
by
  sorry

end simplify_fraction_l1451_145119


namespace tan_add_pi_over_four_sin_cos_ratio_l1451_145106

-- Definition of angle α with the condition that tanα = 2
def α : ℝ := sorry -- Define α such that tan α = 2

-- The first Lean statement for proving tan(α + π/4) = -3
theorem tan_add_pi_over_four (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- The second Lean statement for proving (sinα + cosα) / (2sinα - cosα) = 1
theorem sin_cos_ratio (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1 :=
sorry

end tan_add_pi_over_four_sin_cos_ratio_l1451_145106


namespace find_triples_l1451_145139

theorem find_triples (x y z : ℝ) 
  (h1 : (1/3 : ℝ) * min x y + (2/3 : ℝ) * max x y = 2017)
  (h2 : (1/3 : ℝ) * min y z + (2/3 : ℝ) * max y z = 2018)
  (h3 : (1/3 : ℝ) * min z x + (2/3 : ℝ) * max z x = 2019) :
  (x = 2019) ∧ (y = 2016) ∧ (z = 2019) :=
sorry

end find_triples_l1451_145139


namespace smallest_N_satisfying_conditions_l1451_145130

def is_divisible (n m : ℕ) : Prop :=
  m ∣ n

def satisfies_conditions (N : ℕ) : Prop :=
  (is_divisible N 10) ∧
  (is_divisible N 5) ∧
  (N > 15)

theorem smallest_N_satisfying_conditions : ∃ N, satisfies_conditions N ∧ N = 20 := 
  sorry

end smallest_N_satisfying_conditions_l1451_145130


namespace expression_value_l1451_145176

theorem expression_value (x y : ℤ) (h1 : x = 2) (h2 : y = 5) : 
  (x^4 + 2 * y^2) / 6 = 11 := by
  sorry

end expression_value_l1451_145176


namespace tiling_problem_l1451_145124

theorem tiling_problem (b c f : ℕ) (h : b * c = f) : c * (b^2 / f) = b :=
by 
  sorry

end tiling_problem_l1451_145124


namespace fraction_identity_l1451_145157

theorem fraction_identity (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := 
by
  sorry

end fraction_identity_l1451_145157


namespace factor_of_M_l1451_145199

theorem factor_of_M (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) : 
  1 ∣ (101010 * a + 10001 * b + 100 * c) :=
sorry

end factor_of_M_l1451_145199


namespace companyKW_price_percentage_l1451_145171

theorem companyKW_price_percentage (A B P : ℝ) (h1 : P = 1.40 * A) (h2 : P = 2.00 * B) : 
  P / ((P / 1.40) + (P / 2.00)) * 100 = 82.35 :=
by sorry

end companyKW_price_percentage_l1451_145171


namespace smallest_counterexample_is_14_l1451_145122

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_not_prime (n : ℕ) : Prop := ¬Prime n
def smallest_counterexample (n : ℕ) : Prop :=
  is_even n ∧ is_not_prime n ∧ is_not_prime (n + 2) ∧ ∀ m, is_even m ∧ is_not_prime m ∧ is_not_prime (m + 2) → n ≤ m

theorem smallest_counterexample_is_14 : smallest_counterexample 14 :=
by
  sorry

end smallest_counterexample_is_14_l1451_145122


namespace train_meeting_distance_l1451_145104

theorem train_meeting_distance
  (d : ℝ) (tx ty: ℝ) (dx dy: ℝ)
  (hx : dx = 140) 
  (hy : dy = 140)
  (hx_speed : dx / tx = 35) 
  (hy_speed : dy / ty = 46.67) 
  (meet : tx = ty) :
  d = 60 := 
sorry

end train_meeting_distance_l1451_145104


namespace george_speed_second_segment_l1451_145197

theorem george_speed_second_segment 
  (distance_total : ℝ)
  (speed_normal : ℝ)
  (distance_first : ℝ)
  (speed_first : ℝ) : 
  distance_total = 1 ∧ 
  speed_normal = 3 ∧ 
  distance_first = 0.5 ∧ 
  speed_first = 2 →
  (distance_first / speed_first + 0.5 * speed_second = 1 / speed_normal → speed_second = 6) :=
sorry

end george_speed_second_segment_l1451_145197


namespace area_of_mirror_l1451_145182

theorem area_of_mirror (outer_width : ℝ) (outer_height : ℝ) (frame_width : ℝ) (mirror_area : ℝ) :
  outer_width = 70 → outer_height = 100 → frame_width = 15 → mirror_area = (outer_width - 2 * frame_width) * (outer_height - 2 * frame_width) → mirror_area = 2800 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h4]
  sorry

end area_of_mirror_l1451_145182


namespace union_of_intervals_l1451_145184

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

end union_of_intervals_l1451_145184


namespace tank_volume_ratio_l1451_145195

variable {V1 V2 : ℝ}

theorem tank_volume_ratio
  (h1 : 3 / 4 * V1 = 5 / 8 * V2) :
  V1 / V2 = 5 / 6 :=
sorry

end tank_volume_ratio_l1451_145195


namespace intersecting_points_radius_squared_l1451_145121

noncomputable def parabola1 (x : ℝ) : ℝ := (x - 2) ^ 2
noncomputable def parabola2 (y : ℝ) : ℝ := (y - 5) ^ 2 - 1

theorem intersecting_points_radius_squared :
  ∃ (x y : ℝ), (y = parabola1 x ∧ x = parabola2 y) → (x - 2) ^ 2 + (y - 5) ^ 2 = 16 := by
sorry

end intersecting_points_radius_squared_l1451_145121


namespace max_band_members_l1451_145102

theorem max_band_members 
  (m : ℤ)
  (h1 : 30 * m % 31 = 7)
  (h2 : 30 * m < 1500) : 
  30 * m = 720 :=
sorry

end max_band_members_l1451_145102


namespace avg_difference_l1451_145136

theorem avg_difference : 
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  avg1 - avg2 = 5 :=
by
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  show avg1 - avg2 = 5
  sorry

end avg_difference_l1451_145136


namespace average_weight_increase_l1451_145172

theorem average_weight_increase (n : ℕ) (w_old w_new : ℝ) (h1 : n = 9) (h2 : w_old = 65) (h3 : w_new = 87.5) :
  (w_new - w_old) / n = 2.5 :=
by
  rw [h1, h2, h3]
  norm_num

end average_weight_increase_l1451_145172


namespace positive_integer_solution_of_inequality_l1451_145161

theorem positive_integer_solution_of_inequality :
  {x : ℕ // 0 < x ∧ x < 2} → x = 1 :=
by
  sorry

end positive_integer_solution_of_inequality_l1451_145161


namespace hall_volume_l1451_145149

theorem hall_volume (length breadth : ℝ) (height : ℝ := 20 / 3)
  (h1 : length = 15)
  (h2 : breadth = 12)
  (h3 : 2 * (length * breadth) = 54 * height) :
  length * breadth * height = 8004 :=
by
  sorry

end hall_volume_l1451_145149


namespace average_age_decrease_l1451_145173

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

end average_age_decrease_l1451_145173


namespace largest_digit_for_divisibility_l1451_145147

theorem largest_digit_for_divisibility (N : ℕ) (h1 : N % 2 = 0) (h2 : (3 + 6 + 7 + 2 + N) % 3 = 0) : N = 6 :=
sorry

end largest_digit_for_divisibility_l1451_145147


namespace find_f_value_l1451_145114

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f (x : ℝ) : f (-x) = -f x
axiom even_f_shift (x : ℝ) : f (-x + 1) = f (x + 1)
axiom f_interval (x : ℝ) (h : 2 < x ∧ x < 4) : f x = |x - 3|

theorem find_f_value : f 1 + f 2 + f 3 + f 4 = 0 :=
by
  sorry

end find_f_value_l1451_145114


namespace value_of_g_neg2_l1451_145178

def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem value_of_g_neg2 : g (-2) = -1 := by
  sorry

end value_of_g_neg2_l1451_145178


namespace decreasing_interval_l1451_145192

noncomputable def f (x : ℝ) := Real.exp (abs (x - 1))

theorem decreasing_interval : ∀ x y : ℝ, x ≤ y → y ≤ 1 → f y ≤ f x :=
by
  sorry

end decreasing_interval_l1451_145192


namespace area_of_garden_l1451_145188

theorem area_of_garden :
  ∃ (short_posts long_posts : ℕ), short_posts + long_posts - 4 = 24 → long_posts = 3 * short_posts →
  ∃ (short_length long_length : ℕ), short_length = (short_posts - 1) * 5 → long_length = (long_posts - 1) * 5 →
  (short_length * long_length = 3000) :=
by {
  sorry
}

end area_of_garden_l1451_145188


namespace box_weight_l1451_145125

theorem box_weight (total_weight : ℕ) (number_of_boxes : ℕ) (box_weight : ℕ) 
  (h1 : total_weight = 267) 
  (h2 : number_of_boxes = 3) 
  (h3 : box_weight = total_weight / number_of_boxes) : 
  box_weight = 89 := 
by 
  sorry

end box_weight_l1451_145125


namespace number_of_green_eyes_l1451_145170

-- Definitions based on conditions
def total_people : Nat := 100
def blue_eyes : Nat := 19
def brown_eyes : Nat := total_people / 2
def black_eyes : Nat := total_people / 4

-- Theorem stating the main question and its answer
theorem number_of_green_eyes : 
  (total_people - (blue_eyes + brown_eyes + black_eyes)) = 6 := by
  sorry

end number_of_green_eyes_l1451_145170


namespace xy_eq_119_imp_sum_values_l1451_145145

theorem xy_eq_119_imp_sum_values (x y : ℕ) (hx : x > 0) (hy : y > 0)
(hx_lt_30 : x < 30) (hy_lt_30 : y < 30) (h : x + y + x * y = 119) :
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 := 
sorry

end xy_eq_119_imp_sum_values_l1451_145145


namespace pentagon_square_ratio_l1451_145159

theorem pentagon_square_ratio (p s : ℕ) 
  (h1 : 5 * p = 20) (h2 : 4 * s = 20) : p / s = 4 / 5 :=
by sorry

end pentagon_square_ratio_l1451_145159


namespace x_value_for_divisibility_l1451_145185

theorem x_value_for_divisibility (x : ℕ) (h1 : x = 0 ∨ x = 5) (h2 : (8 * 10 + x) % 4 = 0) : x = 0 :=
by
  sorry

end x_value_for_divisibility_l1451_145185


namespace trigonometric_identity_l1451_145137

-- Define the main theorem
theorem trigonometric_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 :=
by
  sorry

end trigonometric_identity_l1451_145137


namespace gray_region_area_l1451_145181

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

end gray_region_area_l1451_145181


namespace gg_of_3_is_107_l1451_145156

-- Define the function g
def g (x : ℕ) : ℕ := 3 * x + 2

-- State that g(g(g(3))) equals 107
theorem gg_of_3_is_107 : g (g (g 3)) = 107 := by
  sorry

end gg_of_3_is_107_l1451_145156


namespace initial_pencils_sold_l1451_145183

theorem initial_pencils_sold (x : ℕ) (P : ℝ)
  (h1 : 1 = 0.9 * (x * P))
  (h2 : 1 = 1.2 * (8.25 * P))
  : x = 11 :=
by sorry

end initial_pencils_sold_l1451_145183


namespace find_a3_plus_a5_l1451_145107

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

end find_a3_plus_a5_l1451_145107


namespace findCostPrices_l1451_145140

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

end findCostPrices_l1451_145140


namespace total_students_count_l1451_145126

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

end total_students_count_l1451_145126


namespace triangle_area_l1451_145108

theorem triangle_area (AB CD : ℝ) (h₁ : 0 < AB) (h₂ : 0 < CD) (h₃ : CD = 3 * AB) :
    let trapezoid_area := 18
    let triangle_ABC_area := trapezoid_area / 4
    triangle_ABC_area = 4.5 := by
  sorry

end triangle_area_l1451_145108
