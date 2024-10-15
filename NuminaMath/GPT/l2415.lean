import Mathlib

namespace NUMINAMATH_GPT_equal_distances_l2415_241517

def Point := ℝ × ℝ × ℝ

def dist (p1 p2 : Point) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 + (z1 - z2) ^ 2

def A : Point := (-8, 0, 0)
def B : Point := (0, 4, 0)
def C : Point := (0, 0, -6)
def D : Point := (0, 0, 0)
def P : Point := (-4, 2, -3)

theorem equal_distances : dist P A = dist P B ∧ dist P B = dist P C ∧ dist P C = dist P D :=
by
  sorry

end NUMINAMATH_GPT_equal_distances_l2415_241517


namespace NUMINAMATH_GPT_max_area_perpendicular_l2415_241543

theorem max_area_perpendicular (a b θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) : 
  ∃ θ_max, θ_max = Real.pi / 2 ∧ (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  (0 < Real.sin θ → (1 / 2) * a * b * Real.sin θ ≤ (1 / 2) * a * b * 1)) :=
sorry

end NUMINAMATH_GPT_max_area_perpendicular_l2415_241543


namespace NUMINAMATH_GPT_Wendy_age_l2415_241535

theorem Wendy_age
  (years_as_accountant : ℕ)
  (years_as_manager : ℕ)
  (percent_accounting_related : ℝ)
  (total_accounting_related : ℕ)
  (total_lifespan : ℝ) :
  years_as_accountant = 25 →
  years_as_manager = 15 →
  percent_accounting_related = 0.50 →
  total_accounting_related = years_as_accountant + years_as_manager →
  (total_accounting_related : ℝ) = percent_accounting_related * total_lifespan →
  total_lifespan = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_Wendy_age_l2415_241535


namespace NUMINAMATH_GPT_valentino_chickens_l2415_241570

variable (C : ℕ) -- Number of chickens
variable (D : ℕ) -- Number of ducks
variable (T : ℕ) -- Number of turkeys
variable (total_birds : ℕ) -- Total number of birds on the farm

theorem valentino_chickens (h1 : D = 2 * C) 
                            (h2 : T = 3 * D)
                            (h3 : total_birds = C + D + T)
                            (h4 : total_birds = 1800) :
  C = 200 := by
  sorry

end NUMINAMATH_GPT_valentino_chickens_l2415_241570


namespace NUMINAMATH_GPT_width_of_plot_is_60_l2415_241520

-- Defining the conditions
def length_of_plot := 90
def distance_between_poles := 5
def number_of_poles := 60

-- The theorem statement
theorem width_of_plot_is_60 :
  ∃ width : ℕ, 2 * (length_of_plot + width) = number_of_poles * distance_between_poles ∧ width = 60 :=
sorry

end NUMINAMATH_GPT_width_of_plot_is_60_l2415_241520


namespace NUMINAMATH_GPT_tetrahedron_edges_sum_of_squares_l2415_241540

-- Given conditions
variables {a b c d e f x y z : ℝ}

-- Mathematical statement
theorem tetrahedron_edges_sum_of_squares :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 4 * (x^2 + y^2 + z^2) :=
sorry

end NUMINAMATH_GPT_tetrahedron_edges_sum_of_squares_l2415_241540


namespace NUMINAMATH_GPT_discount_percentage_l2415_241586

theorem discount_percentage (CP MP SP D : ℝ) (cp_value : CP = 100) 
(markup : MP = CP + 0.5 * CP) (profit : SP = CP + 0.35 * CP) 
(discount : D = MP - SP) : (D / MP) * 100 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_discount_percentage_l2415_241586


namespace NUMINAMATH_GPT_secondChapterPages_is_18_l2415_241594

-- Define conditions as variables and constants
def thirdChapterPages : ℕ := 3
def additionalPages : ℕ := 15

-- The main statement to prove
theorem secondChapterPages_is_18 : (thirdChapterPages + additionalPages) = 18 := by
  -- Proof would go here, but we skip it with sorry
  sorry

end NUMINAMATH_GPT_secondChapterPages_is_18_l2415_241594


namespace NUMINAMATH_GPT_f_zero_is_two_l2415_241513

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x1 x2 x3 x4 x5 : ℝ) : 
  f (x1 + x2 + x3 + x4 + x5) = f x1 + f x2 + f x3 + f x4 + f x5 - 8

theorem f_zero_is_two : f 0 = 2 := 
by
  sorry

end NUMINAMATH_GPT_f_zero_is_two_l2415_241513


namespace NUMINAMATH_GPT_find_m_l2415_241585

variable (a : ℝ × ℝ := (2, 3))
variable (b : ℝ × ℝ := (-1, 2))

def isCollinear (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

theorem find_m (m : ℝ) (h : isCollinear (2 * m - 4, 3 * m + 8) (4, -1)) : m = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_l2415_241585


namespace NUMINAMATH_GPT_Marcy_120_votes_l2415_241507

-- Definitions based on conditions
def votes (name : String) : ℕ := sorry -- placeholder definition

-- Conditions
def Joey_votes := votes "Joey" = 8
def Jill_votes := votes "Jill" = votes "Joey" + 4
def Barry_votes := votes "Barry" = 2 * (votes "Joey" + votes "Jill")
def Marcy_votes := votes "Marcy" = 3 * votes "Barry"
def Tim_votes := votes "Tim" = votes "Marcy" / 2
def Sam_votes := votes "Sam" = votes "Tim" + 10

-- Theorem to prove
theorem Marcy_120_votes : Joey_votes → Jill_votes → Barry_votes → Marcy_votes → Tim_votes → Sam_votes → votes "Marcy" = 120 := by
  intros
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_Marcy_120_votes_l2415_241507


namespace NUMINAMATH_GPT_jessie_weight_before_jogging_l2415_241515

theorem jessie_weight_before_jogging (current_weight lost_weight : ℕ) 
(hc : current_weight = 67)
(hl : lost_weight = 7) : 
current_weight + lost_weight = 74 := 
by
  -- Here we skip the proof part
  sorry

end NUMINAMATH_GPT_jessie_weight_before_jogging_l2415_241515


namespace NUMINAMATH_GPT_original_price_of_apples_l2415_241567

-- Define variables and conditions
variables (P : ℝ)

-- The conditions of the problem
def price_increase_condition := 1.25 * P * 8 = 64

-- The theorem stating the original price per pound of apples
theorem original_price_of_apples (h : price_increase_condition P) : P = 6.40 :=
sorry

end NUMINAMATH_GPT_original_price_of_apples_l2415_241567


namespace NUMINAMATH_GPT_additional_charge_is_correct_l2415_241576

noncomputable def additional_charge_per_segment (initial_fee : ℝ) (total_distance : ℝ) (total_charge : ℝ) (segment_length : ℝ) : ℝ :=
  let segments := total_distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  charge_for_distance / segments

theorem additional_charge_is_correct :
  additional_charge_per_segment 2.0 3.6 5.15 (2/5) = 0.35 :=
by
  sorry

end NUMINAMATH_GPT_additional_charge_is_correct_l2415_241576


namespace NUMINAMATH_GPT_sandy_correct_sums_l2415_241563

variable (c i : ℕ)

theorem sandy_correct_sums (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_sandy_correct_sums_l2415_241563


namespace NUMINAMATH_GPT_total_spent_target_l2415_241593

theorem total_spent_target (face_moisturizer_cost : ℕ) (body_lotion_cost : ℕ) (face_moisturizers_bought : ℕ) (body_lotions_bought : ℕ) (christy_multiplier : ℕ) :
  face_moisturizer_cost = 50 →
  body_lotion_cost = 60 →
  face_moisturizers_bought = 2 →
  body_lotions_bought = 4 →
  christy_multiplier = 2 →
  (face_moisturizers_bought * face_moisturizer_cost + body_lotions_bought * body_lotion_cost) * (1 + christy_multiplier) = 1020 := by
  sorry

end NUMINAMATH_GPT_total_spent_target_l2415_241593


namespace NUMINAMATH_GPT_find_angle_D_l2415_241545

-- Define the given angles and conditions
def angleA := 30
def angleB (D : ℝ) := 2 * D
def angleC (D : ℝ) := D + 40
def sum_of_angles (A B C D : ℝ) := A + B + C + D = 360

theorem find_angle_D (D : ℝ) (hA : angleA = 30) (hB : angleB D = 2 * D) (hC : angleC D = D + 40) (hSum : sum_of_angles angleA (angleB D) (angleC D) D):
  D = 72.5 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_angle_D_l2415_241545


namespace NUMINAMATH_GPT_intersection_M_N_l2415_241550

open Set

def M : Set ℝ := { x | -2 ≤ x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

theorem intersection_M_N :
  M ∩ N = { x | -2 ≤ x ∧ x ≤ -1 } := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2415_241550


namespace NUMINAMATH_GPT_cylinder_radius_range_l2415_241582

theorem cylinder_radius_range :
  (V : ℝ) → (h : ℝ) → (r : ℝ) →
  V = 20 * Real.pi →
  h = 2 →
  (V = Real.pi * r^2 * h) →
  3 < r ∧ r < 4 :=
by
  -- Placeholder for the proof
  intro V h r hV hh hV_eq
  sorry

end NUMINAMATH_GPT_cylinder_radius_range_l2415_241582


namespace NUMINAMATH_GPT_triangle_angle_sine_identity_l2415_241588

theorem triangle_angle_sine_identity (A B C : ℝ) (n : ℤ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = Real.pi) :
  Real.sin (2 * n * A) + Real.sin (2 * n * B) + Real.sin (2 * n * C) = 
  (-1)^(n + 1) * 4 * Real.sin (n * A) * Real.sin (n * B) * Real.sin (n * C) :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_sine_identity_l2415_241588


namespace NUMINAMATH_GPT_gift_cost_calc_l2415_241592

theorem gift_cost_calc (C N : ℕ) (hN : N = 12)
    (h : C / (N - 4) = C / N + 10) : C = 240 := by
  sorry

end NUMINAMATH_GPT_gift_cost_calc_l2415_241592


namespace NUMINAMATH_GPT_bottle_caps_found_l2415_241500

theorem bottle_caps_found
  (caps_current : ℕ) 
  (caps_earlier : ℕ) 
  (h_current : caps_current = 32) 
  (h_earlier : caps_earlier = 25) :
  caps_current - caps_earlier = 7 :=
by 
  sorry

end NUMINAMATH_GPT_bottle_caps_found_l2415_241500


namespace NUMINAMATH_GPT_main_theorem_l2415_241538

/-- A good integer is an integer whose absolute value is not a perfect square. -/
def good (n : ℤ) : Prop := ∀ k : ℤ, k^2 ≠ |n|

/-- Integer m can be represented as a sum of three distinct good integers u, v, w whose product is the square of an odd integer. -/
def special_representation (m : ℤ) : Prop :=
  ∃ u v w : ℤ,
    good u ∧ good v ∧ good w ∧
    (u ≠ v ∧ u ≠ w ∧ v ≠ w) ∧
    (∃ k : ℤ, (u * v * w = k^2 ∧ k % 2 = 1)) ∧
    (m = u + v + w)

/-- All integers m having the property that they can be represented in infinitely many ways as a sum of three distinct good integers whose product is the square of an odd integer are those which are congruent to 3 modulo 4. -/
theorem main_theorem (m : ℤ) : special_representation m ↔ m % 4 = 3 := sorry

end NUMINAMATH_GPT_main_theorem_l2415_241538


namespace NUMINAMATH_GPT_find_principal_l2415_241580

theorem find_principal (R : ℝ) (P : ℝ) (h : ((P * (R + 5) * 10) / 100) = ((P * R * 10) / 100 + 600)) : P = 1200 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l2415_241580


namespace NUMINAMATH_GPT_three_digit_cubes_divisible_by_8_l2415_241595

theorem three_digit_cubes_divisible_by_8 : ∃ (S : Finset ℕ), S.card = 2 ∧ ∀ x ∈ S, x ^ 3 ≥ 100 ∧ x ^ 3 ≤ 999 ∧ x ^ 3 % 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_cubes_divisible_by_8_l2415_241595


namespace NUMINAMATH_GPT_geometric_sequence_second_term_l2415_241558

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_second_term_l2415_241558


namespace NUMINAMATH_GPT_repeating_decimal_division_l2415_241569

def repeating_decimal_142857 : ℚ := 1 / 7
def repeating_decimal_2_857143 : ℚ := 20 / 7

theorem repeating_decimal_division :
  (repeating_decimal_142857 / repeating_decimal_2_857143) = 1 / 20 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_division_l2415_241569


namespace NUMINAMATH_GPT_polygon_sides_l2415_241529

theorem polygon_sides {n k : ℕ} (h1 : k = n * (n - 3) / 2) (h2 : k = 3 * n / 2) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l2415_241529


namespace NUMINAMATH_GPT_number_of_girls_l2415_241526

theorem number_of_girls (total_children boys : ℕ) (h1 : total_children = 60) (h2 : boys = 16) : total_children - boys = 44 := by
  sorry

end NUMINAMATH_GPT_number_of_girls_l2415_241526


namespace NUMINAMATH_GPT_difference_mean_median_l2415_241552

theorem difference_mean_median :
  let percentage_scored_60 : ℚ := 0.20
  let percentage_scored_70 : ℚ := 0.30
  let percentage_scored_85 : ℚ := 0.25
  let percentage_scored_95 : ℚ := 1 - (percentage_scored_60 + percentage_scored_70 + percentage_scored_85)
  let score_60 : ℚ := 60
  let score_70 : ℚ := 70
  let score_85 : ℚ := 85
  let score_95 : ℚ := 95
  let mean : ℚ := percentage_scored_60 * score_60 + percentage_scored_70 * score_70 + percentage_scored_85 * score_85 + percentage_scored_95 * score_95
  let median : ℚ := 85
  (median - mean) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_difference_mean_median_l2415_241552


namespace NUMINAMATH_GPT_central_number_l2415_241554

theorem central_number (C : ℕ) (verts : Finset ℕ) (h : verts = {1, 2, 7, 8, 9, 13, 14}) :
  (∀ T ∈ {t | ∃ a b c, (a + b + c) % 3 = 0 ∧ a ∈ verts ∧ b ∈ verts ∧ c ∈ verts}, (T + C) % 3 = 0) →
  C = 9 :=
by
  sorry

end NUMINAMATH_GPT_central_number_l2415_241554


namespace NUMINAMATH_GPT_number_of_multiples_of_15_between_35_and_200_l2415_241505

theorem number_of_multiples_of_15_between_35_and_200 : ∃ n : ℕ, n = 11 ∧ ∃ k : ℕ, k ≤ 200 ∧ k ≥ 35 ∧ (∃ m : ℕ, m < n ∧ 45 + m * 15 = k) :=
by
  sorry

end NUMINAMATH_GPT_number_of_multiples_of_15_between_35_and_200_l2415_241505


namespace NUMINAMATH_GPT_largest_x_l2415_241510

theorem largest_x (x : ℝ) : 
  (∃ x, (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2) → 
  (x ≤ 1) := sorry

end NUMINAMATH_GPT_largest_x_l2415_241510


namespace NUMINAMATH_GPT_rahim_books_bought_l2415_241533

theorem rahim_books_bought (x : ℕ) 
  (first_shop_cost second_shop_cost total_books : ℕ)
  (avg_price total_spent : ℕ)
  (h1 : first_shop_cost = 1500)
  (h2 : second_shop_cost = 340)
  (h3 : total_books = x + 60)
  (h4 : avg_price = 16)
  (h5 : total_spent = first_shop_cost + second_shop_cost)
  (h6 : avg_price = total_spent / total_books) :
  x = 55 :=
by
  sorry

end NUMINAMATH_GPT_rahim_books_bought_l2415_241533


namespace NUMINAMATH_GPT_cousin_typing_time_l2415_241527

theorem cousin_typing_time (speed_ratio : ℕ) (my_time_hours : ℕ) (minutes_per_hour : ℕ) (my_time_minutes : ℕ) :
  speed_ratio = 4 →
  my_time_hours = 3 →
  minutes_per_hour = 60 →
  my_time_minutes = my_time_hours * minutes_per_hour →
  ∃ (cousin_time : ℕ), cousin_time = my_time_minutes / speed_ratio := by
  sorry

end NUMINAMATH_GPT_cousin_typing_time_l2415_241527


namespace NUMINAMATH_GPT_prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l2415_241581

def prob_has_bio_test : ℚ := 5 / 8
def prob_not_has_chem_test : ℚ := 1 / 2

theorem prob_not_has_bio_test : 1 - 5 / 8 = 3 / 8 := by
  sorry

theorem combined_prob_neither_bio_nor_chem :
  (1 - 5 / 8) * (1 / 2) = 3 / 16 := by
  sorry

end NUMINAMATH_GPT_prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l2415_241581


namespace NUMINAMATH_GPT_determine_a_l2415_241578

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

theorem determine_a (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l2415_241578


namespace NUMINAMATH_GPT_convex_m_gons_two_acute_angles_l2415_241589

noncomputable def count_convex_m_gons_with_two_acute_angles (m n : ℕ) (P : Finset ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem convex_m_gons_two_acute_angles {m n : ℕ} {P : Finset ℕ}
  (hP : P.card = 2 * n + 1)
  (hmn : 4 < m ∧ m < n) :
  count_convex_m_gons_with_two_acute_angles m n P = 
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) :=
sorry

end NUMINAMATH_GPT_convex_m_gons_two_acute_angles_l2415_241589


namespace NUMINAMATH_GPT_find_C_and_D_l2415_241598

noncomputable def C : ℚ := 15 / 8
noncomputable def D : ℚ := 17 / 8

theorem find_C_and_D (x : ℚ) (h₁ : x ≠ 9) (h₂ : x ≠ -7) :
  (4 * x - 6) / ((x - 9) * (x + 7)) = C / (x - 9) + D / (x + 7) :=
by sorry

end NUMINAMATH_GPT_find_C_and_D_l2415_241598


namespace NUMINAMATH_GPT_reciprocal_sum_l2415_241531

theorem reciprocal_sum (a b c d : ℚ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 3) (h4 : d = 4) : 
  (a / b + c / d)⁻¹ = (20 : ℚ) / 23 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_sum_l2415_241531


namespace NUMINAMATH_GPT_curves_intersect_four_points_l2415_241564

theorem curves_intersect_four_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = 4 * a^2 ∧ y = x^2 - 2 * a) → (a > 1/3)) :=
sorry

end NUMINAMATH_GPT_curves_intersect_four_points_l2415_241564


namespace NUMINAMATH_GPT_sunflower_cans_l2415_241562

theorem sunflower_cans (total_seeds seeds_per_can : ℕ) (h_total_seeds : total_seeds = 54) (h_seeds_per_can : seeds_per_can = 6) :
  total_seeds / seeds_per_can = 9 :=
by sorry

end NUMINAMATH_GPT_sunflower_cans_l2415_241562


namespace NUMINAMATH_GPT_square_floor_tiling_total_number_of_tiles_l2415_241518

theorem square_floor_tiling (s : ℕ) (h : (2 * s - 1 : ℝ) / (s ^ 2 : ℝ) = 0.41) : s = 4 :=
by
  sorry

theorem total_number_of_tiles : 4^2 = 16 := 
by
  norm_num

end NUMINAMATH_GPT_square_floor_tiling_total_number_of_tiles_l2415_241518


namespace NUMINAMATH_GPT_average_bmi_is_correct_l2415_241557

-- Define Rachel's parameters
def rachel_weight : ℕ := 75
def rachel_height : ℕ := 60  -- in inches

-- Define Jimmy's parameters based on the conditions
def jimmy_weight : ℕ := rachel_weight + 6
def jimmy_height : ℕ := rachel_height + 3

-- Define Adam's parameters based on the conditions
def adam_weight : ℕ := rachel_weight - 15
def adam_height : ℕ := rachel_height - 2

-- Define the BMI formula
def bmi (weight : ℕ) (height : ℕ) : ℚ := (weight * 703 : ℚ) / (height * height)

-- Rachel's, Jimmy's, and Adam's BMIs
def rachel_bmi : ℚ := bmi rachel_weight rachel_height
def jimmy_bmi : ℚ := bmi jimmy_weight jimmy_height
def adam_bmi : ℚ := bmi adam_weight adam_height

-- Proving the average BMI
theorem average_bmi_is_correct : 
  (rachel_bmi + jimmy_bmi + adam_bmi) / 3 = 13.85 := 
by
  sorry

end NUMINAMATH_GPT_average_bmi_is_correct_l2415_241557


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2415_241544

theorem simplify_and_evaluate :
  ∀ (x y : ℝ), x = -1/2 → y = 3 → 3 * (2 * x^2 * y - x * y^2) - 2 * (-2 * y^2 * x + x^2 * y) = -3/2 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2415_241544


namespace NUMINAMATH_GPT_triangle_congruence_example_l2415_241524

variable {A B C : Type}
variable (A' B' C' : Type)

def triangle (A B C : Type) : Prop := true

def congruent (t1 t2 : Prop) : Prop := true

variable (P : ℕ)

def perimeter (t : Prop) (p : ℕ) : Prop := true

def length (a b : Type) (l : ℕ) : Prop := true

theorem triangle_congruence_example :
  ∀ (A B C A' B' C' : Type) (h_cong : congruent (triangle A B C) (triangle A' B' C'))
    (h_perimeter : perimeter (triangle A B C) 20)
    (h_AB : length A B 8)
    (h_BC : length B C 5),
    length A C 7 :=
by sorry

end NUMINAMATH_GPT_triangle_congruence_example_l2415_241524


namespace NUMINAMATH_GPT_boat_travel_difference_l2415_241560

-- Define the speeds
variables (a b : ℝ) (ha : a > b)

-- Define the travel times
def downstream_time := 3
def upstream_time := 2

-- Define the distances
def downstream_distance := downstream_time * (a + b)
def upstream_distance := upstream_time * (a - b)

-- Prove the mathematical statement
theorem boat_travel_difference : downstream_distance a b - upstream_distance a b = a + 5 * b := by
  -- sorry can be used to skip the proof
  sorry

end NUMINAMATH_GPT_boat_travel_difference_l2415_241560


namespace NUMINAMATH_GPT_num_integer_distance_pairs_5x5_grid_l2415_241532

-- Define the problem conditions
def grid_size : ℕ := 5

-- Define a function to calculate the number of pairs of vertices with integer distances
noncomputable def count_integer_distance_pairs (n : ℕ) : ℕ := sorry

-- The theorem to prove
theorem num_integer_distance_pairs_5x5_grid : count_integer_distance_pairs grid_size = 108 :=
by
  sorry

end NUMINAMATH_GPT_num_integer_distance_pairs_5x5_grid_l2415_241532


namespace NUMINAMATH_GPT_cost_of_each_math_book_l2415_241577

-- Define the given conditions
def total_books : ℕ := 90
def math_books : ℕ := 53
def history_books : ℕ := total_books - math_books
def history_book_cost : ℕ := 5
def total_price : ℕ := 397

-- The required theorem
theorem cost_of_each_math_book (M : ℕ) (H : 53 * M + history_books * history_book_cost = total_price) : M = 4 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_math_book_l2415_241577


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2415_241522

theorem bus_speed_excluding_stoppages (v : ℝ) 
  (speed_including_stoppages : ℝ := 45) 
  (stoppage_time : ℝ := 1/6) 
  (h : v * (1 - stoppage_time) = speed_including_stoppages) : 
  v = 54 := 
by 
  sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2415_241522


namespace NUMINAMATH_GPT_cube_surface_area_726_l2415_241504

noncomputable def cubeSurfaceArea (volume : ℝ) : ℝ :=
  let side := volume^(1 / 3)
  6 * (side ^ 2)

theorem cube_surface_area_726 (h : cubeSurfaceArea 1331 = 726) : cubeSurfaceArea 1331 = 726 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_726_l2415_241504


namespace NUMINAMATH_GPT_value_of_g_neg2_l2415_241599

-- Define the function g as given in the conditions
def g (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Statement of the problem: Prove that g(-2) = 11
theorem value_of_g_neg2 : g (-2) = 11 := by
  sorry

end NUMINAMATH_GPT_value_of_g_neg2_l2415_241599


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2415_241584

theorem point_in_fourth_quadrant (x : ℝ) (y : ℝ) (hx : x = 8) (hy : y = -3) : x > 0 ∧ y < 0 :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2415_241584


namespace NUMINAMATH_GPT_ratio_of_periods_l2415_241501

variable (I_B T_B : ℝ)
variable (I_A T_A : ℝ)
variable (Profit_A Profit_B TotalProfit : ℝ)
variable (k : ℝ)

-- Define the conditions
axiom h1 : I_A = 3 * I_B
axiom h2 : T_A = k * T_B
axiom h3 : Profit_B = 4500
axiom h4 : TotalProfit = 31500
axiom h5 : Profit_A = TotalProfit - Profit_B

-- The profit shares are proportional to the product of investment and time period
axiom h6 : Profit_A = I_A * T_A
axiom h7 : Profit_B = I_B * T_B

theorem ratio_of_periods : T_A / T_B = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_periods_l2415_241501


namespace NUMINAMATH_GPT_original_deck_size_l2415_241521

/-- 
Aubrey adds 2 additional cards to a deck and then splits the deck evenly among herself and 
two other players, each player having 18 cards. 
We want to prove that the original number of cards in the deck was 52. 
-/
theorem original_deck_size :
  ∃ (n : ℕ), (n + 2) / 3 = 18 ∧ n = 52 :=
by
  sorry

end NUMINAMATH_GPT_original_deck_size_l2415_241521


namespace NUMINAMATH_GPT_arc_PQ_circumference_l2415_241575

-- Definitions based on the identified conditions
def radius : ℝ := 24
def angle_PRQ : ℝ := 90

-- The theorem to prove based on the question and correct answer
theorem arc_PQ_circumference : 
  angle_PRQ = 90 → 
  ∃ arc_length : ℝ, arc_length = (2 * Real.pi * radius) / 4 ∧ arc_length = 12 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_arc_PQ_circumference_l2415_241575


namespace NUMINAMATH_GPT_merchant_markup_percentage_l2415_241591

theorem merchant_markup_percentage (CP MP SP : ℝ) (x : ℝ) (H_CP : CP = 100)
  (H_MP : MP = CP + (x / 100 * CP)) 
  (H_SP_discount : SP = MP * 0.80) 
  (H_SP_profit : SP = CP * 1.12) : 
  x = 40 := 
by
  sorry

end NUMINAMATH_GPT_merchant_markup_percentage_l2415_241591


namespace NUMINAMATH_GPT_length_of_other_parallel_side_l2415_241566

theorem length_of_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : 323 = 1/2 * (20 + b) * 17) :
  b = 18 :=
sorry

end NUMINAMATH_GPT_length_of_other_parallel_side_l2415_241566


namespace NUMINAMATH_GPT_percentage_increase_l2415_241541

theorem percentage_increase (D1 D2 : ℕ) (total_days : ℕ) (H1 : D1 = 4) (H2 : total_days = 9) (H3 : D1 + D2 = total_days) : 
  (D2 - D1) / D1 * 100 = 25 := 
sorry

end NUMINAMATH_GPT_percentage_increase_l2415_241541


namespace NUMINAMATH_GPT_eight_digit_not_perfect_square_l2415_241556

theorem eight_digit_not_perfect_square : ∀ x : ℕ, 0 ≤ x ∧ x ≤ 9999 → ¬ ∃ y : ℤ, (99990000 + x) = y * y := 
by
  intros x hx
  intro h
  obtain ⟨y, hy⟩ := h
  sorry

end NUMINAMATH_GPT_eight_digit_not_perfect_square_l2415_241556


namespace NUMINAMATH_GPT_third_recipe_soy_sauce_l2415_241542

theorem third_recipe_soy_sauce :
  let bottle_ounces := 16
  let cup_ounces := 8
  let first_recipe_cups := 2
  let second_recipe_cups := 1
  let total_bottles := 3
  (total_bottles * bottle_ounces) / cup_ounces - (first_recipe_cups + second_recipe_cups) = 3 :=
by
  sorry

end NUMINAMATH_GPT_third_recipe_soy_sauce_l2415_241542


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_15_terms_l2415_241553

/-- An arithmetic sequence starts at 3 and has a common difference of 4.
    Prove that the sum of the first 15 terms of this sequence is 465. --/
theorem sum_of_arithmetic_sequence_15_terms :
  let a := 3
  let d := 4
  let n := 15
  let aₙ := a + (n - 1) * d
  (n / 2) * (a + aₙ) = 465 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_15_terms_l2415_241553


namespace NUMINAMATH_GPT_smallest_bisecting_segment_l2415_241574

-- Define a structure for a triangle in a plane
structure Triangle (α β γ : Type u) :=
(vertex1 : α) 
(vertex2 : β) 
(vertex3 : γ) 
(area : ℝ)

-- Define a predicate for an excellent line
def is_excellent_line {α β γ : Type u} (T : Triangle α β γ) (A : α) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line excellent here, e.g., dividing area in half
sorry

-- Define a function to get the length of a line segment within the triangle
def length_within_triangle {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : ℝ :=
-- compute the length of the segment within the triangle
sorry

-- Define predicates for triangles with specific properties like medians
def is_median {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line a median
sorry

theorem smallest_bisecting_segment {α β γ : Type u} (T : Triangle α β γ) (A : α) (median : ℝ → ℝ → ℝ) : 
  (∀ line, is_excellent_line T A line → length_within_triangle T line ≥ length_within_triangle T median) →
  median = line  := 
-- show that the median from the vertex opposite the smallest angle has the smallest segment
sorry

end NUMINAMATH_GPT_smallest_bisecting_segment_l2415_241574


namespace NUMINAMATH_GPT_angle_between_sides_of_triangle_l2415_241503

noncomputable def right_triangle_side_lengths1 : Nat × Nat × Nat := (15, 36, 39)
noncomputable def right_triangle_side_lengths2 : Nat × Nat × Nat := (40, 42, 58)

-- Assuming both triangles are right triangles
def is_right_triangle (a b c : Nat) : Prop := a^2 + b^2 = c^2

theorem angle_between_sides_of_triangle
  (h1 : is_right_triangle 15 36 39)
  (h2 : is_right_triangle 40 42 58) : 
  ∃ (θ : ℝ), θ = 90 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_sides_of_triangle_l2415_241503


namespace NUMINAMATH_GPT_multiplication_results_l2415_241547

theorem multiplication_results
  (h1 : 25 * 4 = 100) :
  25 * 8 = 200 ∧ 25 * 12 = 300 ∧ 250 * 40 = 10000 ∧ 25 * 24 = 600 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_results_l2415_241547


namespace NUMINAMATH_GPT_class_funds_l2415_241519

theorem class_funds (total_contribution : ℕ) (students : ℕ) (contribution_per_student : ℕ) (remaining_amount : ℕ) 
    (h1 : total_contribution = 90) 
    (h2 : students = 19) 
    (h3 : contribution_per_student = 4) 
    (h4 : remaining_amount = total_contribution - (students * contribution_per_student)) : 
    remaining_amount = 14 :=
sorry

end NUMINAMATH_GPT_class_funds_l2415_241519


namespace NUMINAMATH_GPT_ellipse_foci_y_axis_l2415_241516

theorem ellipse_foci_y_axis (k : ℝ) :
  (∃ a b : ℝ, a = 15 - k ∧ b = k - 9 ∧ a > 0 ∧ b > 0) ↔ (12 < k ∧ k < 15) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_y_axis_l2415_241516


namespace NUMINAMATH_GPT_find_distance_to_place_l2415_241508

noncomputable def distance_to_place (speed_boat : ℝ) (speed_stream : ℝ) (total_time : ℝ) : ℝ :=
  let downstream_speed := speed_boat + speed_stream
  let upstream_speed := speed_boat - speed_stream
  let distance := (total_time * (downstream_speed * upstream_speed)) / (downstream_speed + upstream_speed)
  distance

theorem find_distance_to_place :
  distance_to_place 16 2 937.1428571428571 = 7392.92 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_to_place_l2415_241508


namespace NUMINAMATH_GPT_arithmetic_sequence_a8_l2415_241506

theorem arithmetic_sequence_a8 (a_1 : ℕ) (S_5 : ℕ) (h_a1 : a_1 = 1) (h_S5 : S_5 = 35) : 
    ∃ a_8 : ℕ, a_8 = 22 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a8_l2415_241506


namespace NUMINAMATH_GPT_distance_between_A_and_B_is_40_l2415_241579

theorem distance_between_A_and_B_is_40
  (v1 v2 : ℝ)
  (h1 : ∃ t: ℝ, t = (40 / 2) / v1 ∧ t = (40 - 24) / v2)
  (h2 : ∃ t: ℝ, t = (40 - 15) / v1 ∧ t = 40 / (2 * v2)) :
  40 = 40 := by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_is_40_l2415_241579


namespace NUMINAMATH_GPT_cost_for_33_people_employees_for_14000_cost_l2415_241571

-- Define the conditions for pricing
def price_per_ticket (x : Nat) : Int :=
  if x ≤ 30 then 400
  else max 280 (400 - 5 * (x - 30))

def total_cost (x : Nat) : Int :=
  x * price_per_ticket x

-- Problem Part 1: Proving the total cost for 33 people
theorem cost_for_33_people :
  total_cost 33 = 12705 :=
by
  sorry

-- Problem Part 2: Given a total cost of 14000, finding the number of employees
theorem employees_for_14000_cost :
  ∃ x : Nat, total_cost x = 14000 ∧ price_per_ticket x ≥ 280 :=
by
  sorry

end NUMINAMATH_GPT_cost_for_33_people_employees_for_14000_cost_l2415_241571


namespace NUMINAMATH_GPT_cricket_innings_l2415_241511

theorem cricket_innings (n : ℕ) (h1 : (32 * n + 137) / (n + 1) = 37) : n = 20 :=
sorry

end NUMINAMATH_GPT_cricket_innings_l2415_241511


namespace NUMINAMATH_GPT_solve_trig_eq_l2415_241587

noncomputable def arccos (x : ℝ) : ℝ := sorry

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  -3 * (Real.cos x) ^ 2 + 5 * (Real.sin x) + 1 = 0 ↔
  (x = Real.arcsin (1 / 3) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (1 / 3) + 2 * k * Real.pi) :=
sorry

end NUMINAMATH_GPT_solve_trig_eq_l2415_241587


namespace NUMINAMATH_GPT_distance_between_skew_lines_l2415_241525

-- Definitions for the geometric configuration
def AB : ℝ := 4
def AA1 : ℝ := 4
def AD : ℝ := 3

-- Theorem statement to prove the distance between skew lines A1D and B1D1
theorem distance_between_skew_lines:
  ∃ d : ℝ, d = (6 * Real.sqrt 34) / 17 :=
sorry

end NUMINAMATH_GPT_distance_between_skew_lines_l2415_241525


namespace NUMINAMATH_GPT_cost_of_5_spoons_l2415_241596

theorem cost_of_5_spoons (cost_per_set : ℕ) (num_spoons_per_set : ℕ) (num_spoons_needed : ℕ)
  (h1 : cost_per_set = 21) (h2 : num_spoons_per_set = 7) (h3 : num_spoons_needed = 5) :
  (cost_per_set / num_spoons_per_set) * num_spoons_needed = 15 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_5_spoons_l2415_241596


namespace NUMINAMATH_GPT_only_one_true_l2415_241530

-- Definitions based on conditions
def line := Type
def plane := Type
def parallel (m n : line) : Prop := sorry
def perpendicular (m n : line) : Prop := sorry
def subset (m : line) (alpha : plane) : Prop := sorry

-- Propositions derived from conditions
def prop1 (m n : line) (alpha : plane) : Prop := parallel m alpha ∧ parallel n alpha → ¬ parallel m n
def prop2 (m n : line) (alpha : plane) : Prop := perpendicular m alpha ∧ perpendicular n alpha → parallel m n
def prop3 (m n : line) (alpha beta : plane) : Prop := parallel alpha beta ∧ subset m alpha ∧ subset n beta → parallel m n
def prop4 (m n : line) (alpha beta : plane) : Prop := perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular m alpha → perpendicular n beta

-- Theorem statement that only one proposition is true
theorem only_one_true (m n : line) (alpha beta : plane) :
  (prop1 m n alpha = false) ∧
  (prop2 m n alpha = true) ∧
  (prop3 m n alpha beta = false) ∧
  (prop4 m n alpha beta = false) :=
by sorry

end NUMINAMATH_GPT_only_one_true_l2415_241530


namespace NUMINAMATH_GPT_bianca_ate_candies_l2415_241561

-- Definitions based on the conditions
def total_candies : ℕ := 32
def pieces_per_pile : ℕ := 5
def number_of_piles : ℕ := 4

-- The statement to prove
theorem bianca_ate_candies : 
  total_candies - (pieces_per_pile * number_of_piles) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_bianca_ate_candies_l2415_241561


namespace NUMINAMATH_GPT_age_in_1930_l2415_241537

/-- A person's age at the time of their death (y) was one 31st of their birth year,
and we want to prove the person's age in 1930 (x). -/
theorem age_in_1930 (x y : ℕ) (h : 31 * y + x = 1930) (hx : 0 < x) (hxy : x < y) :
  x = 39 :=
sorry

end NUMINAMATH_GPT_age_in_1930_l2415_241537


namespace NUMINAMATH_GPT_number_of_tickets_bought_l2415_241539

noncomputable def ticketCost : ℕ := 5
noncomputable def popcornCost : ℕ := (80 * ticketCost) / 100
noncomputable def sodaCost : ℕ := (50 * popcornCost) / 100
noncomputable def totalSpent : ℕ := 36
noncomputable def numberOfPopcorns : ℕ := 2 
noncomputable def numberOfSodas : ℕ := 4

theorem number_of_tickets_bought : 
  (totalSpent - (numberOfPopcorns * popcornCost + numberOfSodas * sodaCost)) = 4 * ticketCost :=
by
  sorry

end NUMINAMATH_GPT_number_of_tickets_bought_l2415_241539


namespace NUMINAMATH_GPT_max_sum_a_b_l2415_241509

theorem max_sum_a_b (a b : ℝ) (h : a^2 - a*b + b^2 = 1) : a + b ≤ 2 := 
by sorry

end NUMINAMATH_GPT_max_sum_a_b_l2415_241509


namespace NUMINAMATH_GPT_robin_uploaded_pics_from_camera_l2415_241590

-- Definitions of the conditions
def pics_from_phone := 35
def albums := 5
def pics_per_album := 8

-- The statement we want to prove
theorem robin_uploaded_pics_from_camera : (albums * pics_per_album) - pics_from_phone = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_robin_uploaded_pics_from_camera_l2415_241590


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2415_241555

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
noncomputable def vector_b : ℝ × ℝ := (2, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Statement: Prove x > 0 is a necessary but not sufficient condition for the angle between vectors a and b to be acute.
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (dot_product (vector_a x) vector_b > 0) ↔ (x > 0) := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2415_241555


namespace NUMINAMATH_GPT_barbi_weight_loss_duration_l2415_241528

theorem barbi_weight_loss_duration :
  (∃ x : ℝ, 
    (∃ l_barbi l_luca : ℝ, 
      l_barbi = 1.5 * x ∧ 
      l_luca = 99 ∧ 
      l_luca = l_barbi + 81) ∧
    x = 12) :=
by
  sorry

end NUMINAMATH_GPT_barbi_weight_loss_duration_l2415_241528


namespace NUMINAMATH_GPT_percentage_of_girls_after_changes_l2415_241536

theorem percentage_of_girls_after_changes :
  let boys_classA := 15
  let girls_classA := 20
  let boys_classB := 25
  let girls_classB := 35
  let boys_transferAtoB := 3
  let girls_transferAtoB := 2
  let boys_joiningA := 4
  let girls_joiningA := 6

  let boys_classA_after := boys_classA - boys_transferAtoB + boys_joiningA
  let girls_classA_after := girls_classA - girls_transferAtoB + girls_joiningA
  let boys_classB_after := boys_classB + boys_transferAtoB
  let girls_classB_after := girls_classB + girls_transferAtoB

  let total_students := boys_classA_after + girls_classA_after + boys_classB_after + girls_classB_after
  let total_girls := girls_classA_after + girls_classB_after 

  (total_girls / total_students : ℝ) * 100 = 58.095 := by
  sorry

end NUMINAMATH_GPT_percentage_of_girls_after_changes_l2415_241536


namespace NUMINAMATH_GPT_find_c_l2415_241583

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem find_c (a b m c : ℝ) (h1 : ∀ x, f x a b ≥ 0)
  (h2 : ∀ x, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l2415_241583


namespace NUMINAMATH_GPT_find_total_people_find_children_l2415_241597

variables (x m : ℕ)

-- Given conditions translated into Lean

def group_b_more_people (x : ℕ) := x + 4
def sum_is_18_times_difference (x : ℕ) := (x + (x + 4)) = 18 * ((x + 4) - x)
def children_b_less_than_three_times (m : ℕ) := (3 * m) - 2
def adult_ticket_price := 100
def children_ticket_price := (100 * 60) / 100
def same_amount_spent (x m : ℕ) := 100 * (x - m) + (100 * 60 / 100) * m = 100 * ((group_b_more_people x) - (children_b_less_than_three_times m)) + (100 * 60 / 100) * (children_b_less_than_three_times m)

-- Proving the two propositions (question == answer given conditions)

theorem find_total_people (x : ℕ) (hx : sum_is_18_times_difference x) : x = 34 ∧ (group_b_more_people x) = 38 :=
by {
  sorry -- proof for x = 34 and group_b_people = 38 given that sum_is_18_times_difference x
}

theorem find_children (m : ℕ) (x : ℕ) (hx : sum_is_18_times_difference x) (hm : same_amount_spent x m) : m = 6 ∧ (children_b_less_than_three_times m) = 16 :=
by {
  sorry -- proof for m = 6 and children_b_people = 16 given sum_is_18_times_difference x and same_amount_spent x m
}

end NUMINAMATH_GPT_find_total_people_find_children_l2415_241597


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l2415_241502

-- Problem 1: 
theorem problem1_solution (x : ℝ) (h : 4 * x^2 = 9) : x = 3 / 2 ∨ x = - (3 / 2) := 
by sorry

-- Problem 2: 
theorem problem2_solution (x : ℝ) (h : (1 - 2 * x)^3 = 8) : x = - 1 / 2 := 
by sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l2415_241502


namespace NUMINAMATH_GPT_distance_sum_l2415_241549

theorem distance_sum (a : ℝ) (x y : ℝ) 
  (AB CD : ℝ) (A B C D P Q M N : ℝ)
  (h_AB : AB = 4) (h_CD : CD = 8) 
  (h_M_AB : M = (A + B) / 2) (h_N_CD : N = (C + D) / 2)
  (h_P_AB : P ∈ [A, B]) (h_Q_CD : Q ∈ [C, D])
  (h_x : x = dist P M) (h_y : y = dist Q N)
  (h_y_eq_2x : y = 2 * x) (h_x_eq_a : x = a) :
  x + y = 3 * a := 
by
  sorry

end NUMINAMATH_GPT_distance_sum_l2415_241549


namespace NUMINAMATH_GPT_find_9a_value_l2415_241534

theorem find_9a_value (a : ℚ) 
  (h : (4 - a) / (5 - a) = (4 / 5) ^ 2) : 9 * a = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_9a_value_l2415_241534


namespace NUMINAMATH_GPT_convex_polygon_with_tiles_l2415_241551

variable (n : ℕ)

def canFormConvexPolygon (n : ℕ) : Prop :=
  3 ≤ n ∧ n ≤ 12

theorem convex_polygon_with_tiles (n : ℕ) 
  (square_internal_angle : ℕ := 90) 
  (equilateral_triangle_internal_angle : ℕ := 60)
  (external_angle_step : ℕ := 30)
  (total_external_angle : ℕ := 360) :
  canFormConvexPolygon n :=
by 
  sorry

end NUMINAMATH_GPT_convex_polygon_with_tiles_l2415_241551


namespace NUMINAMATH_GPT_tree_height_at_3_years_l2415_241568

-- Define the conditions as Lean definitions
def tree_height (years : ℕ) : ℕ :=
  2 ^ years

-- State the theorem using the defined conditions
theorem tree_height_at_3_years : tree_height 6 = 32 → tree_height 3 = 4 := by
  intro h
  sorry

end NUMINAMATH_GPT_tree_height_at_3_years_l2415_241568


namespace NUMINAMATH_GPT_smallest_b_l2415_241565

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : a + b = 7) (h4 : 2 + a ≤ b) : b = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_l2415_241565


namespace NUMINAMATH_GPT_window_width_l2415_241514

theorem window_width (length area : ℝ) (h_length : length = 6) (h_area : area = 60) :
  area / length = 10 :=
by
  sorry

end NUMINAMATH_GPT_window_width_l2415_241514


namespace NUMINAMATH_GPT_call_cost_per_minute_l2415_241573

-- Definitions (conditions)
def initial_credit : ℝ := 30
def call_duration : ℕ := 22
def remaining_credit : ℝ := 26.48

-- The goal is to prove that the cost per minute of the call is 0.16
theorem call_cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
sorry

end NUMINAMATH_GPT_call_cost_per_minute_l2415_241573


namespace NUMINAMATH_GPT_standard_equation_of_ellipse_midpoint_of_chord_l2415_241546

variables (a b c : ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (A B : ℝ × ℝ)

axiom conditions :
  a > b ∧ b > 0 ∧
  (c / a = (Real.sqrt 6) / 3) ∧
  a = Real.sqrt 3 ∧
  a^2 = b^2 + c^2 ∧
  (A = (-1, 0)) ∧ (B = (x2, y2)) ∧
  A ≠ B ∧
  (∃ l : ℝ -> ℝ, l (-1) = 0 ∧ ∀ x, l x = x + 1) ∧
  (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = -3 / 2)

theorem standard_equation_of_ellipse :
  ∃ (e : ℝ), e = 1 ∧ (x1 / 3) + y1 = 1 := sorry

theorem midpoint_of_chord :
  ∃ (m : ℝ × ℝ), m = (-(3 / 4), 1 / 4) := sorry

end NUMINAMATH_GPT_standard_equation_of_ellipse_midpoint_of_chord_l2415_241546


namespace NUMINAMATH_GPT_range_of_m_l2415_241512

theorem range_of_m (m : ℝ) :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ 7 < m ∧ m < 24 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2415_241512


namespace NUMINAMATH_GPT_inequality_abc_l2415_241572

theorem inequality_abc (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1/a + 1/(b * c)) * (1/b + 1/(c * a)) * (1/c + 1/(a * b)) ≥ 1728 :=
by sorry

end NUMINAMATH_GPT_inequality_abc_l2415_241572


namespace NUMINAMATH_GPT_additionalPeopleNeededToMowLawn_l2415_241559

def numberOfPeopleNeeded (people : ℕ) (hours : ℕ) : ℕ :=
  (people * 8) / hours

theorem additionalPeopleNeededToMowLawn : numberOfPeopleNeeded 4 3 - 4 = 7 :=
by
  sorry

end NUMINAMATH_GPT_additionalPeopleNeededToMowLawn_l2415_241559


namespace NUMINAMATH_GPT_find_number_l2415_241548

theorem find_number (N : ℕ) (h : N / 16 = 16 * 8) : N = 2048 :=
sorry

end NUMINAMATH_GPT_find_number_l2415_241548


namespace NUMINAMATH_GPT_john_needs_more_money_l2415_241523

def total_needed : ℝ := 2.50
def current_amount : ℝ := 0.75
def remaining_amount : ℝ := 1.75

theorem john_needs_more_money : total_needed - current_amount = remaining_amount :=
by
  sorry

end NUMINAMATH_GPT_john_needs_more_money_l2415_241523
