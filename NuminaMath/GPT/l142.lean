import Mathlib

namespace NUMINAMATH_GPT_total_spokes_in_garage_l142_14244

def bicycle1_front_spokes : ℕ := 16
def bicycle1_back_spokes : ℕ := 18
def bicycle2_front_spokes : ℕ := 20
def bicycle2_back_spokes : ℕ := 22
def bicycle3_front_spokes : ℕ := 24
def bicycle3_back_spokes : ℕ := 26
def bicycle4_front_spokes : ℕ := 28
def bicycle4_back_spokes : ℕ := 30
def tricycle_front_spokes : ℕ := 32
def tricycle_middle_spokes : ℕ := 34
def tricycle_back_spokes : ℕ := 36

theorem total_spokes_in_garage :
  bicycle1_front_spokes + bicycle1_back_spokes +
  bicycle2_front_spokes + bicycle2_back_spokes +
  bicycle3_front_spokes + bicycle3_back_spokes +
  bicycle4_front_spokes + bicycle4_back_spokes +
  tricycle_front_spokes + tricycle_middle_spokes + tricycle_back_spokes = 286 :=
by
  sorry

end NUMINAMATH_GPT_total_spokes_in_garage_l142_14244


namespace NUMINAMATH_GPT_exists_same_color_rectangle_l142_14287

variable (coloring : ℕ × ℕ → Fin 3)

theorem exists_same_color_rectangle :
  (∃ (r1 r2 r3 r4 c1 c2 c3 c4 : ℕ), 
    r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r4 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r4 ∧ 
    c1 ≠ c2 ∧ 
    coloring (4, 82) = 4 ∧ 
    coloring (r1, c1) = coloring (r1, c2) ∧ coloring (r1, c2) = coloring (r2, c1) ∧ 
    coloring (r2, c1) = coloring (r2, c2)) :=
sorry

end NUMINAMATH_GPT_exists_same_color_rectangle_l142_14287


namespace NUMINAMATH_GPT_functional_equation_solution_l142_14264

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution : 
  (∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋) 
  → ∃ c : ℝ, (c = 0 ∨ (1 ≤ c ∧ c < 2)) ∧ (∀ x : ℝ, f x = c) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l142_14264


namespace NUMINAMATH_GPT_ben_paperclip_day_l142_14221

theorem ben_paperclip_day :
  ∃ k : ℕ, k = 6 ∧ (∀ n : ℕ, n = k → 5 * 3^n > 500) :=
sorry

end NUMINAMATH_GPT_ben_paperclip_day_l142_14221


namespace NUMINAMATH_GPT_Dan_gave_Sara_limes_l142_14214

theorem Dan_gave_Sara_limes : 
  ∀ (original_limes now_limes given_limes : ℕ),
  original_limes = 9 →
  now_limes = 5 →
  given_limes = original_limes - now_limes →
  given_limes = 4 :=
by
  intros original_limes now_limes given_limes h1 h2 h3
  sorry

end NUMINAMATH_GPT_Dan_gave_Sara_limes_l142_14214


namespace NUMINAMATH_GPT_tan_alpha_value_l142_14290

open Real

theorem tan_alpha_value 
  (α : ℝ) 
  (hα_range : 0 < α ∧ α < π) 
  (h_cos_alpha : cos α = -3/5) :
  tan α = -4/3 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_value_l142_14290


namespace NUMINAMATH_GPT_solve_for_x_l142_14260

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 = 6 - x) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l142_14260


namespace NUMINAMATH_GPT_value_of_star_l142_14295

theorem value_of_star :
  ∀ x : ℕ, 45 - (28 - (37 - (15 - x))) = 55 → x = 16 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_value_of_star_l142_14295


namespace NUMINAMATH_GPT_polygon_sides_l142_14223

-- Definition of the conditions used in the problem
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Statement of the theorem
theorem polygon_sides (n : ℕ) (h : sum_of_interior_angles n = 1080) : n = 8 :=
by
  sorry  -- Proof placeholder

end NUMINAMATH_GPT_polygon_sides_l142_14223


namespace NUMINAMATH_GPT_danny_bottle_caps_l142_14250

theorem danny_bottle_caps 
  (wrappers_park : Nat := 46)
  (caps_park : Nat := 50)
  (wrappers_collection : Nat := 52)
  (more_caps_than_wrappers : Nat := 4)
  (h1 : caps_park = wrappers_park + more_caps_than_wrappers)
  (h2 : wrappers_collection = 52) : 
  (∃ initial_caps : Nat, initial_caps + caps_park = wrappers_collection + more_caps_than_wrappers) :=
by 
  use 6
  sorry

end NUMINAMATH_GPT_danny_bottle_caps_l142_14250


namespace NUMINAMATH_GPT_find_x_l142_14232

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 104) : x = 34 :=
sorry

end NUMINAMATH_GPT_find_x_l142_14232


namespace NUMINAMATH_GPT_find_a4_in_geometric_seq_l142_14293

variable {q : ℝ} -- q is the common ratio of the geometric sequence

noncomputable def geometric_seq (q : ℝ) (n : ℕ) : ℝ := 16 * q ^ (n - 1)

theorem find_a4_in_geometric_seq (h1 : geometric_seq q 1 = 16)
  (h2 : geometric_seq q 6 = 2 * geometric_seq q 5 * geometric_seq q 7) :
  geometric_seq q 4 = 2 := 
  sorry

end NUMINAMATH_GPT_find_a4_in_geometric_seq_l142_14293


namespace NUMINAMATH_GPT_paul_work_days_l142_14265

theorem paul_work_days (P : ℕ) (h : 1 / P + 1 / 120 = 1 / 48) : P = 80 := 
by 
  sorry

end NUMINAMATH_GPT_paul_work_days_l142_14265


namespace NUMINAMATH_GPT_determine_perimeter_of_fourth_shape_l142_14241

theorem determine_perimeter_of_fourth_shape
  (P_1 P_2 P_3 P_4 : ℝ)
  (h1 : P_1 = 8)
  (h2 : P_2 = 11.4)
  (h3 : P_3 = 14.7)
  (h4 : P_1 + P_2 + P_4 = 2 * P_3) :
  P_4 = 10 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_determine_perimeter_of_fourth_shape_l142_14241


namespace NUMINAMATH_GPT_alex_growth_rate_l142_14217

noncomputable def growth_rate_per_hour_hanging_upside_down
  (current_height : ℝ)
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) : Prop :=
  current_height + 12 * normal_growth_per_month + 12 * hanging_hours_per_month * answer = required_height

theorem alex_growth_rate 
  (current_height : ℝ) 
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) :
  current_height = 48 → 
  required_height = 54 → 
  normal_growth_per_month = 1/3 → 
  hanging_hours_per_month = 2 → 
  growth_rate_per_hour_hanging_upside_down current_height required_height normal_growth_per_month hanging_hours_per_month answer ↔ answer = 1/12 :=
by sorry

end NUMINAMATH_GPT_alex_growth_rate_l142_14217


namespace NUMINAMATH_GPT_required_hemispherical_containers_l142_14212

noncomputable def initial_volume : ℝ := 10940
noncomputable def initial_temperature : ℝ := 20
noncomputable def final_temperature : ℝ := 25
noncomputable def expansion_coefficient : ℝ := 0.002
noncomputable def container_volume : ℝ := 4
noncomputable def usable_capacity : ℝ := 0.8

noncomputable def volume_expansion : ℝ := initial_volume * (final_temperature - initial_temperature) * expansion_coefficient
noncomputable def final_volume : ℝ := initial_volume + volume_expansion
noncomputable def usable_volume_per_container : ℝ := container_volume * usable_capacity
noncomputable def number_of_containers_needed : ℝ := final_volume / usable_volume_per_container

theorem required_hemispherical_containers : ⌈number_of_containers_needed⌉ = 3453 :=
by 
  sorry

end NUMINAMATH_GPT_required_hemispherical_containers_l142_14212


namespace NUMINAMATH_GPT_baby_guppies_calculation_l142_14297

-- Define the problem in Lean
theorem baby_guppies_calculation :
  ∀ (initial_guppies first_sighting two_days_gups total_guppies_after_two_days : ℕ), 
  initial_guppies = 7 →
  first_sighting = 36 →
  total_guppies_after_two_days = 52 →
  total_guppies_after_two_days = initial_guppies + first_sighting + two_days_gups →
  two_days_gups = 9 :=
by
  intros initial_guppies first_sighting two_days_gups total_guppies_after_two_days
  intros h_initial h_first h_total h_eq
  sorry

end NUMINAMATH_GPT_baby_guppies_calculation_l142_14297


namespace NUMINAMATH_GPT_find_k_l142_14222

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - y = 9 * k) (h3 : x - 2 * y = 22) : k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l142_14222


namespace NUMINAMATH_GPT_price_of_first_shirt_l142_14253

theorem price_of_first_shirt
  (price1 price2 price3 : ℕ)
  (total_shirts : ℕ)
  (min_avg_price_of_remaining : ℕ)
  (total_avg_price_of_all : ℕ)
  (prices_of_first_3 : price1 = 100 ∧ price2 = 90 ∧ price3 = 82)
  (condition1 : total_shirts = 10)
  (condition2 : min_avg_price_of_remaining = 104)
  (condition3 : total_avg_price_of_all > 100) :
  price1 = 100 :=
by
  sorry

end NUMINAMATH_GPT_price_of_first_shirt_l142_14253


namespace NUMINAMATH_GPT_min_value_of_quadratic_expression_l142_14272

theorem min_value_of_quadratic_expression (a b c : ℝ) (h : a + 2 * b + 3 * c = 6) : a^2 + 4 * b^2 + 9 * c^2 ≥ 12 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_expression_l142_14272


namespace NUMINAMATH_GPT_fill_bathtub_time_l142_14288

theorem fill_bathtub_time (V : ℝ) (cold_rate hot_rate drain_rate net_rate : ℝ) 
  (hcold : cold_rate = V / 10) 
  (hhot : hot_rate = V / 15) 
  (hdrain : drain_rate = -V / 12) 
  (hnet : net_rate = cold_rate + hot_rate + drain_rate) 
  (V_eq : V = 1) : 
  1 / net_rate = 12 :=
by {
  -- placeholder for proof steps
  sorry
}

end NUMINAMATH_GPT_fill_bathtub_time_l142_14288


namespace NUMINAMATH_GPT_sum_of_largest_and_smallest_is_correct_l142_14220

-- Define the set of digits
def digits : Finset ℕ := {2, 0, 4, 1, 5, 8}

-- Define the largest possible number using the digits
def largestNumber : ℕ := 854210

-- Define the smallest possible number using the digits
def smallestNumber : ℕ := 102458

-- Define the sum of largest and smallest possible numbers
def sumOfNumbers : ℕ := largestNumber + smallestNumber

-- Main theorem to prove
theorem sum_of_largest_and_smallest_is_correct : sumOfNumbers = 956668 := by
  sorry

end NUMINAMATH_GPT_sum_of_largest_and_smallest_is_correct_l142_14220


namespace NUMINAMATH_GPT_red_ants_count_l142_14245

def total_ants : ℕ := 900
def black_ants : ℕ := 487
def red_ants (r : ℕ) : Prop := r + black_ants = total_ants

theorem red_ants_count : ∃ r : ℕ, red_ants r ∧ r = 413 := 
sorry

end NUMINAMATH_GPT_red_ants_count_l142_14245


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_polygon_l142_14252

theorem sum_of_interior_angles_of_polygon (n : ℕ) (h : n - 3 = 3) : (n - 2) * 180 = 720 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_polygon_l142_14252


namespace NUMINAMATH_GPT_sum_first_10_terms_l142_14276

def a_n (n : ℕ) : ℤ := (-1)^n * (3 * n - 2)

theorem sum_first_10_terms :
  (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) +
  (a_n 6) + (a_n 7) + (a_n 8) + (a_n 9) + (a_n 10) = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_10_terms_l142_14276


namespace NUMINAMATH_GPT_positive_sqrt_729_l142_14224

theorem positive_sqrt_729 (x : ℝ) (h_pos : 0 < x) (h_eq : x^2 = 729) : x = 27 :=
by
  sorry

end NUMINAMATH_GPT_positive_sqrt_729_l142_14224


namespace NUMINAMATH_GPT_sale_in_fifth_month_l142_14233

def sale_first_month : ℝ := 3435
def sale_second_month : ℝ := 3927
def sale_third_month : ℝ := 3855
def sale_fourth_month : ℝ := 4230
def required_avg_sale : ℝ := 3500
def sale_sixth_month : ℝ := 1991

theorem sale_in_fifth_month :
  (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + s + sale_sixth_month) / 6 = required_avg_sale ->
  s = 3562 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l142_14233


namespace NUMINAMATH_GPT_intersection_A_C_U_B_l142_14267

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | Real.log x / Real.log 2 > 0}
def C_U_B : Set ℝ := {x | ¬ (Real.log x / Real.log 2 > 0)}

theorem intersection_A_C_U_B :
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_GPT_intersection_A_C_U_B_l142_14267


namespace NUMINAMATH_GPT_fraction_inequality_l142_14292

theorem fraction_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a < b) (h2 : c < d) : (a + c) / (b + c) < (a + d) / (b + d) :=
by
  sorry

end NUMINAMATH_GPT_fraction_inequality_l142_14292


namespace NUMINAMATH_GPT_tickets_system_l142_14209

variable (x y : ℕ)

theorem tickets_system (h1 : x + y = 20) (h2 : 2800 * x + 6400 * y = 74000) :
  (x + y = 20) ∧ (2800 * x + 6400 * y = 74000) :=
by {
  exact (And.intro h1 h2)
}

end NUMINAMATH_GPT_tickets_system_l142_14209


namespace NUMINAMATH_GPT_pythagorean_triplets_l142_14280

theorem pythagorean_triplets (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ d p q : ℤ, a = 2 * d * p * q ∧ b = d * (q^2 - p^2) ∧ c = d * (p^2 + q^2) := sorry

end NUMINAMATH_GPT_pythagorean_triplets_l142_14280


namespace NUMINAMATH_GPT_pieces_present_l142_14204

-- Define the pieces and their counts in a standard chess set
def total_pieces := 32
def missing_pieces := 12
def missing_kings := 1
def missing_queens := 2
def missing_knights := 3
def missing_pawns := 6

-- The theorem statement that we need to prove
theorem pieces_present : 
  (total_pieces - (missing_kings + missing_queens + missing_knights + missing_pawns)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_pieces_present_l142_14204


namespace NUMINAMATH_GPT_sn_values_l142_14289

noncomputable def s (x1 x2 x3 : ℂ) (n : ℕ) : ℂ :=
  x1^n + x2^n + x3^n

theorem sn_values (p q x1 x2 x3 : ℂ) (h_root1 : x1^3 + p * x1 + q = 0)
                    (h_root2 : x2^3 + p * x2 + q = 0)
                    (h_root3 : x3^3 + p * x3 + q = 0) :
  s x1 x2 x3 2 = -3 * q ∧
  s x1 x2 x3 3 = 3 * q^2 ∧
  s x1 x2 x3 4 = 2 * p^2 ∧
  s x1 x2 x3 5 = 5 * p * q ∧
  s x1 x2 x3 6 = -2 * p^3 + 3 * q^2 ∧
  s x1 x2 x3 7 = -7 * p^2 * q ∧
  s x1 x2 x3 8 = 2 * p^4 - 8 * p * q^2 ∧
  s x1 x2 x3 9 = 9 * p^3 * q - 3 * q^3 ∧
  s x1 x2 x3 10 = -2 * p^5 + 15 * p^2 * q^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sn_values_l142_14289


namespace NUMINAMATH_GPT_find_b_l142_14282

noncomputable def a_and_b_integers_and_factor (a b : ℤ) : Prop :=
  ∀ (x : ℝ), (x^2 - x - 1) * (a*x^3 + b*x^2 - x + 1) = 0

theorem find_b (a b : ℤ) (h : a_and_b_integers_and_factor a b) : b = -1 :=
by 
  sorry

end NUMINAMATH_GPT_find_b_l142_14282


namespace NUMINAMATH_GPT_club_committee_selections_l142_14269

theorem club_committee_selections : (Nat.choose 18 3) = 816 := by
  sorry

end NUMINAMATH_GPT_club_committee_selections_l142_14269


namespace NUMINAMATH_GPT_Amy_crumbs_l142_14248

variable (z : ℕ)

theorem Amy_crumbs (T C : ℕ) (h1 : T * C = z)
  (h2 : ∃ T_A : ℕ, T_A = 2 * T)
  (h3 : ∃ C_A : ℕ, C_A = (3 * C) / 2) :
  ∃ z_A : ℕ, z_A = 3 * z :=
by
  sorry

end NUMINAMATH_GPT_Amy_crumbs_l142_14248


namespace NUMINAMATH_GPT_find_number_l142_14281

theorem find_number (x : ℕ) (h : x = 4) : x + 1 = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l142_14281


namespace NUMINAMATH_GPT_mean_combined_l142_14249

-- Definitions for the two sets and their properties
def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

variables (set₁ set₂ : List ℕ)
-- Conditions based on the problem
axiom h₁ : set₁.length = 7
axiom h₂ : mean set₁ = 15
axiom h₃ : set₂.length = 8
axiom h₄ : mean set₂ = 30

-- Prove that the mean of the combined set is 23
theorem mean_combined (h₁ : set₁.length = 7) (h₂ : mean set₁ = 15)
  (h₃ : set₂.length = 8) (h₄ : mean set₂ = 30) : mean (set₁ ++ set₂) = 23 := 
sorry

end NUMINAMATH_GPT_mean_combined_l142_14249


namespace NUMINAMATH_GPT_new_student_weight_l142_14294

theorem new_student_weight :
  ∀ (W : ℝ) (total_weight_19 : ℝ) (total_weight_20 : ℝ),
    total_weight_19 = 19 * 15 →
    total_weight_20 = 20 * 14.8 →
    total_weight_19 + W = total_weight_20 →
    W = 11 :=
by
  intros W total_weight_19 total_weight_20 h1 h2 h3
  -- Skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_new_student_weight_l142_14294


namespace NUMINAMATH_GPT_starting_number_of_range_l142_14298

theorem starting_number_of_range (multiples: ℕ) (end_of_range: ℕ) (span: ℕ)
  (h1: multiples = 991) (h2: end_of_range = 10000) (h3: span = multiples * 10) :
  end_of_range - span = 90 := 
by 
  sorry

end NUMINAMATH_GPT_starting_number_of_range_l142_14298


namespace NUMINAMATH_GPT_num_letters_dot_not_straight_line_l142_14268

variable (Total : ℕ)
variable (DS : ℕ)
variable (S_only : ℕ)
variable (D_only : ℕ)

theorem num_letters_dot_not_straight_line 
  (h1 : Total = 40) 
  (h2 : DS = 11) 
  (h3 : S_only = 24) 
  (h4 : Total - S_only - DS = D_only) : 
  D_only = 5 := 
by 
  sorry

end NUMINAMATH_GPT_num_letters_dot_not_straight_line_l142_14268


namespace NUMINAMATH_GPT_avg_lottery_draws_eq_5232_l142_14229

def avg_lottery_draws (n m : ℕ) : ℕ :=
  let N := 90 * 89 * 88 * 87 * 86
  let Nk := 25 * 40320
  N / Nk

theorem avg_lottery_draws_eq_5232 : avg_lottery_draws 90 5 = 5232 :=
by 
  unfold avg_lottery_draws
  sorry

end NUMINAMATH_GPT_avg_lottery_draws_eq_5232_l142_14229


namespace NUMINAMATH_GPT_largest_fraction_l142_14262

theorem largest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 5)
                                          (h2 : f2 = 3 / 6)
                                          (h3 : f3 = 5 / 10)
                                          (h4 : f4 = 7 / 15)
                                          (h5 : f5 = 8 / 20) : 
  (f2 = 1 / 2 ∨ f3 = 1 / 2) ∧ (f2 ≥ f1 ∧ f2 ≥ f4 ∧ f2 ≥ f5) ∧ (f3 ≥ f1 ∧ f3 ≥ f4 ∧ f3 ≥ f5) := 
by
  sorry

end NUMINAMATH_GPT_largest_fraction_l142_14262


namespace NUMINAMATH_GPT_apples_to_pears_l142_14299

theorem apples_to_pears (a o p : ℕ) 
  (h1 : 10 * a = 5 * o) 
  (h2 : 3 * o = 4 * p) : 
  (20 * a) = 40 / 3 * p :=
sorry

end NUMINAMATH_GPT_apples_to_pears_l142_14299


namespace NUMINAMATH_GPT_find_a_b_max_profit_allocation_l142_14291

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (a * Real.log x) / x + 5 / x - b

theorem find_a_b :
  (∃ (a b : ℝ), f 1 a b = 5 ∧ f 10 a b = 16.515) :=
sorry

noncomputable def g (x : ℝ) := 2 * Real.sqrt x / x

noncomputable def profit (x : ℝ) := x * (5 * Real.log x / x + 5 / x) + (50 - x) * (2 * Real.sqrt (50 - x) / (50 - x))

theorem max_profit_allocation :
  (∃ (x : ℝ), 10 ≤ x ∧ x ≤ 40 ∧ ∀ y, (10 ≤ y ∧ y ≤ 40) → profit x ≥ profit y)
  ∧ profit 25 = 31.09 :=
sorry

end NUMINAMATH_GPT_find_a_b_max_profit_allocation_l142_14291


namespace NUMINAMATH_GPT_combined_books_total_l142_14205

def keith_books : ℕ := 20
def jason_books : ℕ := 21
def amanda_books : ℕ := 15
def sophie_books : ℕ := 30

def total_books := keith_books + jason_books + amanda_books + sophie_books

theorem combined_books_total : total_books = 86 := 
by sorry

end NUMINAMATH_GPT_combined_books_total_l142_14205


namespace NUMINAMATH_GPT_mom_twice_alex_l142_14238

-- Definitions based on the conditions
def alex_age_in_2010 : ℕ := 10
def mom_age_in_2010 : ℕ := 5 * alex_age_in_2010
def future_years_after_2010 (x : ℕ) : ℕ := 2010 + x

-- Defining the ages in the future year
def alex_age_future (x : ℕ) : ℕ := alex_age_in_2010 + x
def mom_age_future (x : ℕ) : ℕ := mom_age_in_2010 + x

-- The theorem to prove
theorem mom_twice_alex (x : ℕ) (h : mom_age_future x = 2 * alex_age_future x) : future_years_after_2010 x = 2040 :=
  by
  sorry

end NUMINAMATH_GPT_mom_twice_alex_l142_14238


namespace NUMINAMATH_GPT_solid_is_triangular_prism_l142_14242

-- Given conditions as definitions
def front_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the front view is an isosceles triangle
  sorry

def left_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the left view is an isosceles triangle
  sorry

def top_view_is_circle (solid : Type) : Prop := 
   -- Define the property that the top view is a circle
  sorry

-- Define the property of being a triangular prism
def is_triangular_prism (solid : Type) : Prop :=
  -- Define the property that the solid is a triangular prism
  sorry

-- The main theorem: proving that given the conditions, the solid could be a triangular prism
theorem solid_is_triangular_prism (solid : Type) :
  front_view_is_isosceles_triangle solid ∧ 
  left_view_is_isosceles_triangle solid ∧ 
  top_view_is_circle solid →
  is_triangular_prism solid :=
sorry

end NUMINAMATH_GPT_solid_is_triangular_prism_l142_14242


namespace NUMINAMATH_GPT_student_incorrect_answer_l142_14228

theorem student_incorrect_answer (D I : ℕ) (h1 : D / 63 = I) (h2 : D / 36 = 42) : I = 24 := by
  sorry

end NUMINAMATH_GPT_student_incorrect_answer_l142_14228


namespace NUMINAMATH_GPT_solve_quadratic_eq_l142_14239

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2*x + 1 = 0) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l142_14239


namespace NUMINAMATH_GPT_ratio_eq_23_over_28_l142_14210

theorem ratio_eq_23_over_28 (a b : ℚ) (h : (12 * a - 5 * b) / (14 * a - 3 * b) = 4 / 7) : 
  a / b = 23 / 28 := 
sorry

end NUMINAMATH_GPT_ratio_eq_23_over_28_l142_14210


namespace NUMINAMATH_GPT_compute_multiplied_difference_l142_14237

theorem compute_multiplied_difference (a b : ℕ) (h_a : a = 25) (h_b : b = 15) :
  3 * ((a + b) ^ 2 - (a - b) ^ 2) = 4500 := by
  sorry

end NUMINAMATH_GPT_compute_multiplied_difference_l142_14237


namespace NUMINAMATH_GPT_parallelogram_base_length_l142_14203

theorem parallelogram_base_length (b : ℝ) (A : ℝ) (h : ℝ)
  (H1 : A = 288) 
  (H2 : h = 2 * b) 
  (H3 : A = b * h) : 
  b = 12 := 
by 
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l142_14203


namespace NUMINAMATH_GPT_sandy_correct_sums_l142_14270

theorem sandy_correct_sums :
  ∃ c i : ℤ,
  c + i = 40 ∧
  4 * c - 3 * i = 72 ∧
  c = 27 :=
by 
  sorry

end NUMINAMATH_GPT_sandy_correct_sums_l142_14270


namespace NUMINAMATH_GPT_probability_sibling_pair_l142_14261

-- Define the necessary constants for the problem.
def B : ℕ := 500 -- Number of business students
def L : ℕ := 800 -- Number of law students
def S : ℕ := 30  -- Number of sibling pairs

-- State the theorem representing the mathematical proof problem
theorem probability_sibling_pair :
  (S : ℝ) / (B * L) = 0.000075 := sorry

end NUMINAMATH_GPT_probability_sibling_pair_l142_14261


namespace NUMINAMATH_GPT_ratio_of_surface_areas_of_spheres_l142_14201

theorem ratio_of_surface_areas_of_spheres (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) : 
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_ratio_of_surface_areas_of_spheres_l142_14201


namespace NUMINAMATH_GPT_gcd_1911_1183_l142_14275

theorem gcd_1911_1183 : gcd 1911 1183 = 91 :=
by sorry

end NUMINAMATH_GPT_gcd_1911_1183_l142_14275


namespace NUMINAMATH_GPT_zhao_estimate_larger_l142_14200

theorem zhao_estimate_larger (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - 2 * ε) > x - y :=
by
  sorry

end NUMINAMATH_GPT_zhao_estimate_larger_l142_14200


namespace NUMINAMATH_GPT_line_intersects_plane_at_angle_l142_14208

def direction_vector : ℝ × ℝ × ℝ := (1, -1, 2)
def normal_vector : ℝ × ℝ × ℝ := (-2, 2, -4)

theorem line_intersects_plane_at_angle :
  let a := direction_vector
  let u := normal_vector
  a ≠ (0, 0, 0) → u ≠ (0, 0, 0) →
  ∃ θ : ℝ, 0 < θ ∧ θ < π :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_plane_at_angle_l142_14208


namespace NUMINAMATH_GPT_complement_union_l142_14286

variable (x : ℝ)

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 1}
def P : Set ℝ := {x | x ≥ 2}

theorem complement_union (x : ℝ) : x ∈ U → (¬ (x ∈ M ∨ x ∈ P)) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end NUMINAMATH_GPT_complement_union_l142_14286


namespace NUMINAMATH_GPT_technicians_count_l142_14246

-- Define the conditions
def avg_sal_all (total_workers : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 850

def avg_sal_technicians (teches : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 1000

def avg_sal_rest (others : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 780

-- The main theorem to prove
theorem technicians_count (total_workers : ℕ)
  (teches others : ℕ)
  (total_salary : ℕ) :
  total_workers = 22 →
  total_salary = 850 * 22 →
  avg_sal_all total_workers 850 →
  avg_sal_technicians teches 1000 →
  avg_sal_rest others 780 →
  teches + others = total_workers →
  1000 * teches + 780 * others = total_salary →
  teches = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_technicians_count_l142_14246


namespace NUMINAMATH_GPT_high_sulfur_oil_samples_l142_14219

/-- The number of high-sulfur oil samples in a container with the given conditions. -/
theorem high_sulfur_oil_samples (total_samples : ℕ) 
    (heavy_oil_freq : ℚ) (light_low_sulfur_freq : ℚ)
    (no_heavy_low_sulfur: true) (almost_full : total_samples = 198)
    (heavy_oil_freq_value : heavy_oil_freq = 1 / 9)
    (light_low_sulfur_freq_value : light_low_sulfur_freq = 11 / 18) :
    (22 + 68) = 90 := 
by
  sorry

end NUMINAMATH_GPT_high_sulfur_oil_samples_l142_14219


namespace NUMINAMATH_GPT_greatest_positive_multiple_of_4_l142_14255

theorem greatest_positive_multiple_of_4 {y : ℕ} (h1 : y % 4 = 0) (h2 : y > 0) (h3 : y^3 < 8000) : y ≤ 16 :=
by {
  -- The proof will go here
  -- Sorry is placed here to skip the proof for now
  sorry
}

end NUMINAMATH_GPT_greatest_positive_multiple_of_4_l142_14255


namespace NUMINAMATH_GPT_all_numbers_equal_l142_14215

theorem all_numbers_equal
  (n : ℕ)
  (h n_eq_20 : n = 20)
  (a : ℕ → ℝ)
  (h_avg : ∀ i : ℕ, i < n → a i = (a ((i+n-1) % n) + a ((i+1) % n)) / 2) :
  ∀ i j : ℕ, i < n → j < n → a i = a j :=
by {
  -- Proof steps go here.
  sorry
}

end NUMINAMATH_GPT_all_numbers_equal_l142_14215


namespace NUMINAMATH_GPT_subset_A_l142_14243

open Set

theorem subset_A (A : Set ℝ) (h : A = { x | x > -1 }) : {0} ⊆ A :=
by
  sorry

end NUMINAMATH_GPT_subset_A_l142_14243


namespace NUMINAMATH_GPT_fraction_of_Charlie_circumference_l142_14227

/-- Definitions for the problem conditions -/
def Jack_head_circumference : ℕ := 12
def Charlie_head_circumference : ℕ := 9 + Jack_head_circumference / 2
def Bill_head_circumference : ℕ := 10

/-- Statement of the theorem to be proved -/
theorem fraction_of_Charlie_circumference :
  Bill_head_circumference / Charlie_head_circumference = 2 / 3 :=
sorry

end NUMINAMATH_GPT_fraction_of_Charlie_circumference_l142_14227


namespace NUMINAMATH_GPT_negation_of_p_implication_q_l142_14226

noncomputable def negation_of_conditions : Prop :=
∀ (a : ℝ), (a > 0 → a^2 > a) ∧ (¬(a > 0) ↔ ¬(a^2 > a)) → ¬(a ≤ 0 → a^2 ≤ a)

theorem negation_of_p_implication_q :
  negation_of_conditions :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_p_implication_q_l142_14226


namespace NUMINAMATH_GPT_equation_of_AB_l142_14207

-- Definitions based on the conditions
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 3

def midpoint_M (p : ℝ × ℝ) : Prop :=
  p = (1, 0)

-- The theorem to be proved
theorem equation_of_AB (x y : ℝ) (M : ℝ × ℝ) :
  circle_C x y ∧ midpoint_M M → x - y = 1 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_AB_l142_14207


namespace NUMINAMATH_GPT_function_passes_through_fixed_point_l142_14257

variable (a : ℝ)

theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) : (1, 2) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1) + 1)} :=
by
  sorry

end NUMINAMATH_GPT_function_passes_through_fixed_point_l142_14257


namespace NUMINAMATH_GPT_Douglas_won_72_percent_of_votes_in_county_X_l142_14274

/-- Definition of the problem conditions and the goal -/
theorem Douglas_won_72_percent_of_votes_in_county_X
  (V : ℝ)
  (total_votes_ratio : ∀ county_X county_Y, county_X = 2 * county_Y)
  (total_votes_percentage_both_counties : 0.60 = (1.8 * V) / (2 * V + V))
  (votes_percentage_county_Y : 0.36 = (0.36 * V) / V) : 
  ∃ P : ℝ, P = 72 ∧ P = (1.44 * V) / (2 * V) * 100 :=
sorry

end NUMINAMATH_GPT_Douglas_won_72_percent_of_votes_in_county_X_l142_14274


namespace NUMINAMATH_GPT_fraction_zero_solve_l142_14279

theorem fraction_zero_solve (x : ℝ) (h : (x^2 - 49) / (x + 7) = 0) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_solve_l142_14279


namespace NUMINAMATH_GPT_triangle_altitude_l142_14256

theorem triangle_altitude (base side : ℝ) (h : ℝ) : 
  side = 6 → base = 6 → 
  (base * h) / 2 = side ^ 2 → 
  h = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_triangle_altitude_l142_14256


namespace NUMINAMATH_GPT_probability_of_sum_odd_is_correct_l142_14236

noncomputable def probability_sum_odd : ℚ :=
  let total_balls := 13
  let drawn_balls := 7
  let total_ways := Nat.choose total_balls drawn_balls
  let favorable_ways := 
    Nat.choose 7 5 * Nat.choose 6 2 + 
    Nat.choose 7 3 * Nat.choose 6 4 + 
    Nat.choose 7 1 * Nat.choose 6 6
  favorable_ways / total_ways

theorem probability_of_sum_odd_is_correct :
  probability_sum_odd = 847 / 1716 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_of_sum_odd_is_correct_l142_14236


namespace NUMINAMATH_GPT_mario_pizza_area_l142_14247

theorem mario_pizza_area
  (pizza_area : ℝ)
  (cut_distance : ℝ)
  (largest_piece : ℝ)
  (smallest_piece : ℝ)
  (total_pieces : ℕ)
  (pieces_mario_gets_area : ℝ) :
  pizza_area = 4 →
  cut_distance = 0.5 →
  total_pieces = 4 →
  pieces_mario_gets_area = (pizza_area - (largest_piece + smallest_piece)) / 2 →
  pieces_mario_gets_area = 1.5 :=
sorry

end NUMINAMATH_GPT_mario_pizza_area_l142_14247


namespace NUMINAMATH_GPT_dot_product_correct_l142_14218

-- Define the vectors as given conditions
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (1, -2)

-- State the theorem to prove the dot product
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = -4 := by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_dot_product_correct_l142_14218


namespace NUMINAMATH_GPT_polynomial_difference_of_squares_l142_14254

theorem polynomial_difference_of_squares (x y : ℤ) :
  8 * x^2 + 2 * x * y - 3 * y^2 = (3 * x - y)^2 - (x + 2 * y)^2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_difference_of_squares_l142_14254


namespace NUMINAMATH_GPT_value_of_expression_l142_14277

theorem value_of_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l142_14277


namespace NUMINAMATH_GPT_problem_circumscribing_sphere_surface_area_l142_14206

noncomputable def surface_area_of_circumscribing_sphere (a b c : ℕ) :=
  let R := (Real.sqrt (a^2 + b^2 + c^2)) / 2
  4 * Real.pi * R^2

theorem problem_circumscribing_sphere_surface_area
  (a b c : ℕ)
  (ha : (1 / 2 : ℝ) * a * b = 4)
  (hb : (1 / 2 : ℝ) * b * c = 6)
  (hc : (1 / 2: ℝ) * a * c = 12) : 
  surface_area_of_circumscribing_sphere a b c = 56 * Real.pi := 
sorry

end NUMINAMATH_GPT_problem_circumscribing_sphere_surface_area_l142_14206


namespace NUMINAMATH_GPT_john_average_speed_l142_14296

theorem john_average_speed :
  let distance_uphill := 2 -- distance in km
  let distance_downhill := 2 -- distance in km
  let time_uphill := 45 / 60 -- time in hours (45 minutes)
  let time_downhill := 15 / 60 -- time in hours (15 minutes)
  let total_distance := distance_uphill + distance_downhill -- total distance in km
  let total_time := time_uphill + time_downhill -- total time in hours
  total_distance / total_time = 4 := by
  sorry

end NUMINAMATH_GPT_john_average_speed_l142_14296


namespace NUMINAMATH_GPT_sum_greater_than_3_l142_14278

theorem sum_greater_than_3 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a > a + b + c) : a + b + c > 3 :=
sorry

end NUMINAMATH_GPT_sum_greater_than_3_l142_14278


namespace NUMINAMATH_GPT_ferry_captives_successfully_l142_14266

-- Definition of conditions
def valid_trip_conditions (trips: ℕ) (captives: ℕ) : Prop :=
  captives = 43 ∧
  (∀ k < trips, k % 2 = 0 ∨ k % 2 = 1) ∧     -- Trips done in pairs or singles
  (∀ k < captives, k > 40)                    -- At least 40 other captives known as werewolves

-- Theorem statement to be proved
theorem ferry_captives_successfully (trips : ℕ) (captives : ℕ) (result : Prop) : 
  valid_trip_conditions trips captives → result = true := by sorry

end NUMINAMATH_GPT_ferry_captives_successfully_l142_14266


namespace NUMINAMATH_GPT_a_values_l142_14273

noncomputable def A (a : ℝ) : Set ℝ := {1, a, 5}
noncomputable def B (a : ℝ) : Set ℝ := {2, a^2 + 1}

theorem a_values (a : ℝ) : A a ∩ B a = {x} → (a = 0 ∧ x = 1) ∨ (a = -2 ∧ x = 5) := sorry

end NUMINAMATH_GPT_a_values_l142_14273


namespace NUMINAMATH_GPT_product_eq_one_l142_14284

noncomputable def f (x : ℝ) : ℝ := |Real.logb 3 x|

theorem product_eq_one (a b : ℝ) (h_diff : a ≠ b) (h_eq : f a = f b) : a * b = 1 := by
  sorry

end NUMINAMATH_GPT_product_eq_one_l142_14284


namespace NUMINAMATH_GPT_ordering_of_exponentiations_l142_14258

def a : ℕ := 3 ^ 34
def b : ℕ := 2 ^ 51
def c : ℕ := 4 ^ 25

theorem ordering_of_exponentiations : c < b ∧ b < a := by
  sorry

end NUMINAMATH_GPT_ordering_of_exponentiations_l142_14258


namespace NUMINAMATH_GPT_number_of_integers_inequality_l142_14283

theorem number_of_integers_inequality : (∃ s : Finset ℤ, (∀ x ∈ s, 10 ≤ x ∧ x ≤ 24) ∧ s.card = 15) :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_inequality_l142_14283


namespace NUMINAMATH_GPT_problem_statement_l142_14240

-- Defining the sets U, M, and N
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

-- Complement of N in U
def complement_U_N : Set ℕ := U \ N

-- Problem statement
theorem problem_statement : M ∩ complement_U_N = {0, 3} :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l142_14240


namespace NUMINAMATH_GPT_remainder_not_power_of_4_l142_14202

theorem remainder_not_power_of_4 : ∃ n : ℕ, n ≥ 2 ∧ ¬ (∃ k : ℕ, (2^2^n) % (2^n - 1) = 4^k) := sorry

end NUMINAMATH_GPT_remainder_not_power_of_4_l142_14202


namespace NUMINAMATH_GPT_smallest_three_digit_number_with_property_l142_14231

theorem smallest_three_digit_number_with_property : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
  (∀ d, (1 ≤ d ∧ d ≤ 1000) → ((d = n + 1 ∨ d = n - 1) → d % 11 = 0)) ∧ 
  n = 120 :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_number_with_property_l142_14231


namespace NUMINAMATH_GPT_johnny_marbles_combination_l142_14251

theorem johnny_marbles_combination : @Nat.choose 9 4 = 126 := by
  sorry

end NUMINAMATH_GPT_johnny_marbles_combination_l142_14251


namespace NUMINAMATH_GPT_permutations_of_six_attractions_is_720_l142_14259

-- Define the number of attractions
def num_attractions : ℕ := 6

-- State the theorem to be proven
theorem permutations_of_six_attractions_is_720 : (num_attractions.factorial = 720) :=
by {
  sorry
}

end NUMINAMATH_GPT_permutations_of_six_attractions_is_720_l142_14259


namespace NUMINAMATH_GPT_garbage_classification_competition_l142_14225

theorem garbage_classification_competition :
  let boy_rate_seventh := 0.4
  let boy_rate_eighth := 0.5
  let girl_rate_seventh := 0.6
  let girl_rate_eighth := 0.7
  let combined_boy_rate := (boy_rate_seventh + boy_rate_eighth) / 2
  let combined_girl_rate := (girl_rate_seventh + girl_rate_eighth) / 2
  boy_rate_seventh < boy_rate_eighth ∧ combined_boy_rate < combined_girl_rate :=
by {
  sorry
}

end NUMINAMATH_GPT_garbage_classification_competition_l142_14225


namespace NUMINAMATH_GPT_smoothie_cost_l142_14234

-- Definitions of costs and amounts paid.
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def amount_paid : ℕ := 20
def change_received : ℕ := 11

-- Define the total cost of the order and the known costs.
def total_order_cost : ℕ := amount_paid - change_received
def known_costs : ℕ := hamburger_cost + onion_rings_cost

-- State the problem: the cost of the smoothie.
theorem smoothie_cost : total_order_cost - known_costs = 3 :=
by 
  sorry

end NUMINAMATH_GPT_smoothie_cost_l142_14234


namespace NUMINAMATH_GPT_problem_1_problem_2_l142_14211

universe u

/-- Assume the universal set U is the set of real numbers -/
def U : Set ℝ := Set.univ

/-- Define set A -/
def A : Set ℝ := {x : ℝ | x ≥ 1}

/-- Define set B -/
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

/-- Prove the intersection of A and B -/
theorem problem_1 : (A ∩ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

/-- Prove the complement of the union of A and B -/
theorem problem_2 : (U \ (A ∪ B)) = {x : ℝ | x < -1} :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l142_14211


namespace NUMINAMATH_GPT_quadratic_two_real_roots_find_m_l142_14263

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_real_roots_find_m_l142_14263


namespace NUMINAMATH_GPT_negation_correct_l142_14230

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  ∀ x > 0, x^2 - 2 * x + 1 ≥ 0

-- Define what it means to negate the proposition
def negated_proposition (x : ℝ) : Prop :=
  ∃ x > 0, x^2 - 2 * x + 1 < 0

-- Main statement: the negation of the original proposition equals the negated proposition
theorem negation_correct : (¬original_proposition x) = (negated_proposition x) :=
  sorry

end NUMINAMATH_GPT_negation_correct_l142_14230


namespace NUMINAMATH_GPT_simplify_expression_l142_14271

theorem simplify_expression : 2 - Real.sqrt 3 + 1 / (2 - Real.sqrt 3) + 1 / (Real.sqrt 3 + 2) = 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l142_14271


namespace NUMINAMATH_GPT_range_of_a_l142_14216

def f (a x : ℝ) : ℝ := x^2 - a*x + a + 3
def g (a x : ℝ) : ℝ := x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(f a x < 0 ∧ g a x < 0)) ↔ a ∈ Set.Icc (-3 : ℝ) 6 :=
sorry

end NUMINAMATH_GPT_range_of_a_l142_14216


namespace NUMINAMATH_GPT_P_gt_Q_l142_14235

variable (x : ℝ)

def P := x^2 + 2
def Q := 2 * x

theorem P_gt_Q : P x > Q x := by
  sorry

end NUMINAMATH_GPT_P_gt_Q_l142_14235


namespace NUMINAMATH_GPT_rational_xyz_squared_l142_14213

theorem rational_xyz_squared
  (x y z : ℝ)
  (hx : ∃ r1 : ℚ, x + y * z = r1)
  (hy : ∃ r2 : ℚ, y + z * x = r2)
  (hz : ∃ r3 : ℚ, z + x * y = r3)
  (hxy : x^2 + y^2 = 1) :
  ∃ r4 : ℚ, x * y * z^2 = r4 := 
sorry

end NUMINAMATH_GPT_rational_xyz_squared_l142_14213


namespace NUMINAMATH_GPT_remaining_string_length_l142_14285

theorem remaining_string_length (original_length : ℝ) (given_to_Minyoung : ℝ) (fraction_used : ℝ) :
  original_length = 70 →
  given_to_Minyoung = 27 →
  fraction_used = 7/9 →
  abs (original_length - given_to_Minyoung - fraction_used * (original_length - given_to_Minyoung) - 9.56) < 0.01 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_remaining_string_length_l142_14285
