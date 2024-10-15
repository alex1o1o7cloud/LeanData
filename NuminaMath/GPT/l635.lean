import Mathlib

namespace NUMINAMATH_GPT_pages_left_to_read_l635_63524

theorem pages_left_to_read (total_pages : ℕ) (pages_read : ℕ) (pages_skipped : ℕ) : 
  total_pages = 372 → pages_read = 125 → pages_skipped = 16 → (total_pages - (pages_read + pages_skipped)) = 231 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pages_left_to_read_l635_63524


namespace NUMINAMATH_GPT_decreases_as_x_increases_graph_passes_through_origin_l635_63529

-- Proof Problem 1: Show that y decreases as x increases if and only if k > 2
theorem decreases_as_x_increases (k : ℝ) : (∀ x1 x2 : ℝ, (x1 < x2) → ((2 - k) * x1 - k^2 + 4) > ((2 - k) * x2 - k^2 + 4)) ↔ (k > 2) := 
  sorry

-- Proof Problem 2: Show that the graph passes through the origin if and only if k = -2
theorem graph_passes_through_origin (k : ℝ) : ((2 - k) * 0 - k^2 + 4 = 0) ↔ (k = -2) :=
  sorry

end NUMINAMATH_GPT_decreases_as_x_increases_graph_passes_through_origin_l635_63529


namespace NUMINAMATH_GPT_tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l635_63578

variable (α : ℝ)
variable (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1 / 4)

theorem tan_alpha_eq_neg2 : Real.tan α = -2 :=
  sorry

theorem sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2 :
  (Real.sin (2 * α) + 1) / (1 + Real.sin (2 * α) + Real.cos (2 * α)) = -1 / 2 :=
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l635_63578


namespace NUMINAMATH_GPT_gcd_of_16_and_12_l635_63593

theorem gcd_of_16_and_12 : Nat.gcd 16 12 = 4 := by
  sorry

end NUMINAMATH_GPT_gcd_of_16_and_12_l635_63593


namespace NUMINAMATH_GPT_min_max_a_e_l635_63522

noncomputable def find_smallest_largest (a b c d e : ℝ) : ℝ × ℝ :=
  if a + b < c + d ∧ c + d < e + a ∧ e + a < b + c ∧ b + c < d + e
    then (a, e)
    else (-1, -1) -- using -1 to indicate invalid input

theorem min_max_a_e (a b c d e : ℝ) : a + b < c + d ∧ c + d < e + a ∧ e + a < b + c ∧ b + c < d + e → 
    find_smallest_largest a b c d e = (a, e) :=
  by
    -- Proof to be filled in by user
    sorry

end NUMINAMATH_GPT_min_max_a_e_l635_63522


namespace NUMINAMATH_GPT_john_total_skateboarded_miles_l635_63501

-- Definitions
def distance_skateboard_to_park := 16
def distance_walk := 8
def distance_bike := 6
def distance_skateboard_home := distance_skateboard_to_park

-- Statement to prove
theorem john_total_skateboarded_miles : 
  distance_skateboard_to_park + distance_skateboard_home = 32 := 
by
  sorry

end NUMINAMATH_GPT_john_total_skateboarded_miles_l635_63501


namespace NUMINAMATH_GPT_line_parallel_plane_l635_63555

axiom line (m : Type) : Prop
axiom plane (α : Type) : Prop
axiom has_no_common_points (m : Type) (α : Type) : Prop
axiom parallel (m : Type) (α : Type) : Prop

theorem line_parallel_plane
  (m : Type) (α : Type)
  (h : has_no_common_points m α) : parallel m α := sorry

end NUMINAMATH_GPT_line_parallel_plane_l635_63555


namespace NUMINAMATH_GPT_market_value_correct_l635_63523

noncomputable def face_value : ℝ := 100
noncomputable def dividend_per_share : ℝ := 0.14 * face_value
noncomputable def yield : ℝ := 0.08

theorem market_value_correct :
  (dividend_per_share / yield) * 100 = 175 := by
  sorry

end NUMINAMATH_GPT_market_value_correct_l635_63523


namespace NUMINAMATH_GPT_product_of_primes_l635_63540

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end NUMINAMATH_GPT_product_of_primes_l635_63540


namespace NUMINAMATH_GPT_unknown_angles_are_80_l635_63594

theorem unknown_angles_are_80 (y : ℝ) (h1 : y + y + 200 = 360) : y = 80 :=
by
  sorry

end NUMINAMATH_GPT_unknown_angles_are_80_l635_63594


namespace NUMINAMATH_GPT_binomial_20_19_eq_20_l635_63591

theorem binomial_20_19_eq_20 : Nat.choose 20 19 = 20 :=
by
  sorry

end NUMINAMATH_GPT_binomial_20_19_eq_20_l635_63591


namespace NUMINAMATH_GPT_sum_of_three_pentagons_l635_63509

variable (x y : ℚ)

axiom eq1 : 3 * x + 2 * y = 27
axiom eq2 : 2 * x + 3 * y = 25

theorem sum_of_three_pentagons : 3 * y = 63 / 5 := 
by {
  sorry -- No need to provide proof steps
}

end NUMINAMATH_GPT_sum_of_three_pentagons_l635_63509


namespace NUMINAMATH_GPT_r_plus_s_value_l635_63521

theorem r_plus_s_value :
  (∃ (r s : ℝ) (line_intercepts : ∀ x y, y = -1/2 * x + 8 ∧ ((x = 16 ∧ y = 0) ∨ (x = 0 ∧ y = 8))), 
    s = -1/2 * r + 8 ∧ (16 * 8 / 2) = 2 * (16 * s / 2) ∧ r + s = 12) :=
sorry

end NUMINAMATH_GPT_r_plus_s_value_l635_63521


namespace NUMINAMATH_GPT_golf_problem_l635_63533

variable (D : ℝ)

theorem golf_problem (h1 : D / 2 + D = 270) : D = 180 :=
by
  sorry

end NUMINAMATH_GPT_golf_problem_l635_63533


namespace NUMINAMATH_GPT_sam_annual_income_l635_63534

theorem sam_annual_income
  (q : ℝ) (I : ℝ)
  (h1 : 30000 * 0.01 * q + 15000 * 0.01 * (q + 3) + (I - 45000) * 0.01 * (q + 5) = (q + 0.35) * 0.01 * I) :
  I = 48376 := 
sorry

end NUMINAMATH_GPT_sam_annual_income_l635_63534


namespace NUMINAMATH_GPT_area_covered_three_layers_l635_63575

noncomputable def auditorium_width : ℕ := 10
noncomputable def auditorium_height : ℕ := 10

noncomputable def first_rug_width : ℕ := 6
noncomputable def first_rug_height : ℕ := 8
noncomputable def second_rug_width : ℕ := 6
noncomputable def second_rug_height : ℕ := 6
noncomputable def third_rug_width : ℕ := 5
noncomputable def third_rug_height : ℕ := 7

-- Prove that the area of part of the auditorium covered with rugs in three layers is 6 square meters.
theorem area_covered_three_layers : 
  let horizontal_overlap_second_third := 5
  let vertical_overlap_second_third := 3
  let area_overlap_second_third := horizontal_overlap_second_third * vertical_overlap_second_third
  let horizontal_overlap_all := 3
  let vertical_overlap_all := 2
  let area_overlap_all := horizontal_overlap_all * vertical_overlap_all
  area_overlap_all = 6 := 
by
  sorry

end NUMINAMATH_GPT_area_covered_three_layers_l635_63575


namespace NUMINAMATH_GPT_change_in_expression_l635_63574

theorem change_in_expression (x b : ℝ) (hb : 0 < b) : 
    (2 * (x + b) ^ 2 + 5 - (2 * x ^ 2 + 5) = 4 * x * b + 2 * b ^ 2) ∨ 
    (2 * (x - b) ^ 2 + 5 - (2 * x ^ 2 + 5) = -4 * x * b + 2 * b ^ 2) := 
by
    sorry

end NUMINAMATH_GPT_change_in_expression_l635_63574


namespace NUMINAMATH_GPT_area_relationship_l635_63590

theorem area_relationship (a b c : ℝ) (h : a^2 + b^2 = c^2) : (a + b)^2 = a^2 + 2*a*b + b^2 := 
by sorry

end NUMINAMATH_GPT_area_relationship_l635_63590


namespace NUMINAMATH_GPT_xiaoma_miscalculation_l635_63531

theorem xiaoma_miscalculation (x : ℤ) (h : 40 + x = 35) : 40 / x = -8 := by
  sorry

end NUMINAMATH_GPT_xiaoma_miscalculation_l635_63531


namespace NUMINAMATH_GPT_john_total_water_usage_l635_63597

-- Define the basic conditions
def total_days_in_weeks (weeks : ℕ) : ℕ := weeks * 7
def showers_every_other_day (days : ℕ) : ℕ := days / 2
def total_minutes_shower (showers : ℕ) (minutes_per_shower : ℕ) : ℕ := showers * minutes_per_shower
def total_water_usage (total_minutes : ℕ) (water_per_minute : ℕ) : ℕ := total_minutes * water_per_minute

-- Main statement
theorem john_total_water_usage :
  total_water_usage (total_minutes_shower (showers_every_other_day (total_days_in_weeks 4)) 10) 2 = 280 :=
by
  sorry

end NUMINAMATH_GPT_john_total_water_usage_l635_63597


namespace NUMINAMATH_GPT_smallest_z_value_l635_63543

theorem smallest_z_value :
  ∃ (w x y z : ℕ), w < x ∧ x < y ∧ y < z ∧
  w + 1 = x ∧ x + 1 = y ∧ y + 1 = z ∧
  w^3 + x^3 + y^3 = z^3 ∧ z = 6 := by
  sorry

end NUMINAMATH_GPT_smallest_z_value_l635_63543


namespace NUMINAMATH_GPT_find_remaining_score_l635_63589

-- Define the problem conditions
def student_scores : List ℕ := [70, 80, 90]
def average_score : ℕ := 70

-- Define the remaining score to prove it equals 40
def remaining_score : ℕ := 40

-- The theorem statement
theorem find_remaining_score (scores : List ℕ) (avg : ℕ) (r : ℕ) 
    (h_scores : scores = [70, 80, 90]) 
    (h_avg : avg = 70) 
    (h_length : scores.length = 3) 
    (h_avg_eq : (scores.sum + r) / (scores.length + 1) = avg) 
    : r = 40 := 
by
  sorry

end NUMINAMATH_GPT_find_remaining_score_l635_63589


namespace NUMINAMATH_GPT_least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l635_63592

def least_subtrahend (n m : ℕ) (k : ℕ) : Prop :=
  (n - k) % m = 0 ∧ ∀ k' : ℕ, k' < k → (n - k') % m ≠ 0

theorem least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22 :
  least_subtrahend 102932847 25 22 :=
sorry

end NUMINAMATH_GPT_least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l635_63592


namespace NUMINAMATH_GPT_probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l635_63546

noncomputable def total_cassettes : ℕ := 30
noncomputable def disco_cassettes : ℕ := 12
noncomputable def classical_cassettes : ℕ := 18

-- Part (a): DJ returns the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_returned :
  (disco_cassettes / total_cassettes) * (disco_cassettes / total_cassettes) = 4 / 25 :=
by
  sorry

-- Part (b): DJ does not return the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_not_returned :
  (disco_cassettes / total_cassettes) * ((disco_cassettes - 1) / (total_cassettes - 1)) = 22 / 145 :=
by
  sorry

end NUMINAMATH_GPT_probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l635_63546


namespace NUMINAMATH_GPT_product_of_solutions_l635_63580

theorem product_of_solutions : 
  ∀ x₁ x₂ : ℝ, (|6 * x₁| + 5 = 47) ∧ (|6 * x₂| + 5 = 47) → x₁ * x₂ = -49 :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_l635_63580


namespace NUMINAMATH_GPT_domain_of_f_l635_63544

-- Define the conditions
def sqrt_domain (x : ℝ) : Prop := x + 1 ≥ 0
def log_domain (x : ℝ) : Prop := 3 - x > 0

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (3 - x)

-- Statement of the theorem
theorem domain_of_f : ∀ x, sqrt_domain x ∧ log_domain x ↔ -1 ≤ x ∧ x < 3 := by
  sorry

end NUMINAMATH_GPT_domain_of_f_l635_63544


namespace NUMINAMATH_GPT_kolya_sheets_exceed_500_l635_63554

theorem kolya_sheets_exceed_500 :
  ∃ k : ℕ, (10 + k * (k + 1) / 2 > 500) :=
sorry

end NUMINAMATH_GPT_kolya_sheets_exceed_500_l635_63554


namespace NUMINAMATH_GPT_third_discount_is_five_percent_l635_63500

theorem third_discount_is_five_percent (P F : ℝ) (D : ℝ)
  (h1: P = 9356.725146198829)
  (h2: F = 6400)
  (h3: F = (1 - D / 100) * (0.9 * (0.8 * P))) : 
  D = 5 := by
  sorry

end NUMINAMATH_GPT_third_discount_is_five_percent_l635_63500


namespace NUMINAMATH_GPT_cost_of_fencing_each_side_l635_63558

theorem cost_of_fencing_each_side (total_cost : ℕ) (num_sides : ℕ) (h1 : total_cost = 288) (h2 : num_sides = 4) : (total_cost / num_sides) = 72 := by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_each_side_l635_63558


namespace NUMINAMATH_GPT_option_d_always_correct_l635_63588

variable {a b : ℝ}

theorem option_d_always_correct (h1 : a < b) (h2 : b < 0) (h3 : a < 0) :
  (a + 1 / b)^2 > (b + 1 / a)^2 :=
by
  -- Lean proof code would go here.
  sorry

end NUMINAMATH_GPT_option_d_always_correct_l635_63588


namespace NUMINAMATH_GPT_reducible_iff_form_l635_63551

def isReducible (a : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 1 ∧ d ∣ (2 * a + 5) ∧ d ∣ (3 * a + 4)

theorem reducible_iff_form (a : ℕ) : isReducible a ↔ ∃ k : ℕ, a = 7 * k + 1 := by
  sorry

end NUMINAMATH_GPT_reducible_iff_form_l635_63551


namespace NUMINAMATH_GPT_base_conversion_least_sum_l635_63542

theorem base_conversion_least_sum :
  ∃ (c d : ℕ), (5 * c + 8 = 8 * d + 5) ∧ c > 0 ∧ d > 0 ∧ (c + d = 15) := by
sorry

end NUMINAMATH_GPT_base_conversion_least_sum_l635_63542


namespace NUMINAMATH_GPT_grains_on_11th_more_than_1_to_9_l635_63518

theorem grains_on_11th_more_than_1_to_9 : 
  let grains_on_square (k : ℕ) := 3 ^ k
  let sum_first_n_squares (n : ℕ) := (3 * (3 ^ n - 1) / (3 - 1))
  grains_on_square 11 - sum_first_n_squares 9 = 147624 :=
by
  sorry

end NUMINAMATH_GPT_grains_on_11th_more_than_1_to_9_l635_63518


namespace NUMINAMATH_GPT_solve_for_n_l635_63514

theorem solve_for_n :
  ∃ n : ℤ, n + (n + 1) + (n + 2) + 3 = 15 ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l635_63514


namespace NUMINAMATH_GPT_inequality_comparison_l635_63573

theorem inequality_comparison (x y : ℝ) (h : x ≠ y) : x^4 + y^4 > x^3 * y + x * y^3 :=
  sorry

end NUMINAMATH_GPT_inequality_comparison_l635_63573


namespace NUMINAMATH_GPT_total_order_cost_is_correct_l635_63512

noncomputable def totalOrderCost : ℝ :=
  let costGeography := 35 * 10.5
  let costEnglish := 35 * 7.5
  let costMath := 20 * 12.0
  let costScience := 30 * 9.5
  let costHistory := 25 * 11.25
  let costArt := 15 * 6.75
  let discount c := c * 0.10
  let netGeography := if 35 >= 30 then costGeography - discount costGeography else costGeography
  let netEnglish := if 35 >= 30 then costEnglish - discount costEnglish else costEnglish
  let netScience := if 30 >= 30 then costScience - discount costScience else costScience
  let netMath := costMath
  let netHistory := costHistory
  let netArt := costArt
  netGeography + netEnglish + netMath + netScience + netHistory + netArt

theorem total_order_cost_is_correct : totalOrderCost = 1446.00 := by
  sorry

end NUMINAMATH_GPT_total_order_cost_is_correct_l635_63512


namespace NUMINAMATH_GPT_shared_friends_l635_63559

theorem shared_friends (crackers total_friends : ℕ) (each_friend_crackers : ℕ) 
  (h1 : crackers = 22) 
  (h2 : each_friend_crackers = 2)
  (h3 : crackers = each_friend_crackers * total_friends) 
  : total_friends = 11 := by 
  sorry

end NUMINAMATH_GPT_shared_friends_l635_63559


namespace NUMINAMATH_GPT_inequality_of_positive_numbers_l635_63513

theorem inequality_of_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end NUMINAMATH_GPT_inequality_of_positive_numbers_l635_63513


namespace NUMINAMATH_GPT_product_of_squares_of_consecutive_even_integers_l635_63587

theorem product_of_squares_of_consecutive_even_integers :
  ∃ (a : ℤ), (a - 2) * a * (a + 2) = 36 * a ∧ (a > 0) ∧ (a % 2 = 0) ∧
  ((a - 2)^2 * a^2 * (a + 2)^2) = 36864 :=
by
  sorry

end NUMINAMATH_GPT_product_of_squares_of_consecutive_even_integers_l635_63587


namespace NUMINAMATH_GPT_multiple_of_sandy_age_l635_63519

theorem multiple_of_sandy_age
    (k_age : ℕ)
    (e : ℕ) 
    (s_current_age : ℕ) 
    (h1: k_age = 10) 
    (h2: e = 340) 
    (h3: s_current_age + 2 = 3 * (k_age + 2)) :
  e / s_current_age = 10 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_sandy_age_l635_63519


namespace NUMINAMATH_GPT_intercept_sum_equation_l635_63545

theorem intercept_sum_equation (c : ℝ) (h₀ : 3 * x + 4 * y + c = 0)
  (h₁ : (-(c / 3)) + (-(c / 4)) = 28) : c = -48 := 
by
  sorry

end NUMINAMATH_GPT_intercept_sum_equation_l635_63545


namespace NUMINAMATH_GPT_average_weight_increase_per_month_l635_63577

theorem average_weight_increase_per_month (w_initial w_final : ℝ) (t : ℝ) 
  (h_initial : w_initial = 3.25) (h_final : w_final = 7) (h_time : t = 3) :
  (w_final - w_initial) / t = 1.25 := 
by 
  sorry

end NUMINAMATH_GPT_average_weight_increase_per_month_l635_63577


namespace NUMINAMATH_GPT_Mary_books_check_out_l635_63598

theorem Mary_books_check_out
  (initial_books : ℕ)
  (returned_unhelpful_books : ℕ)
  (returned_later_books : ℕ)
  (checked_out_later_books : ℕ)
  (total_books_now : ℕ)
  (h1 : initial_books = 5)
  (h2 : returned_unhelpful_books = 3)
  (h3 : returned_later_books = 2)
  (h4 : checked_out_later_books = 7)
  (h5 : total_books_now = 12) :
  ∃ (x : ℕ), (initial_books - returned_unhelpful_books + x - returned_later_books + checked_out_later_books = total_books_now) ∧ x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_Mary_books_check_out_l635_63598


namespace NUMINAMATH_GPT_count_negative_rationals_is_two_l635_63552

theorem count_negative_rationals_is_two :
  let a := (-1 : ℚ) ^ 2007
  let b := (|(-1 : ℚ)| ^ 3)
  let c := -(1 : ℚ) ^ 18
  let d := (18 : ℚ)
  (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) = 2 := by
  sorry

end NUMINAMATH_GPT_count_negative_rationals_is_two_l635_63552


namespace NUMINAMATH_GPT_inequality_proof_l635_63576

theorem inequality_proof
  (a b c d : ℝ)
  (ha : abs a > 1)
  (hb : abs b > 1)
  (hc : abs c > 1)
  (hd : abs d > 1)
  (h : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l635_63576


namespace NUMINAMATH_GPT_percentage_material_B_new_mixture_l635_63596

theorem percentage_material_B_new_mixture :
  let mixtureA := 8 -- kg of Mixture A
  let addOil := 2 -- kg of additional oil
  let addMixA := 6 -- kg of additional Mixture A
  let oil_percent := 0.20 -- 20% oil in Mixture A
  let materialB_percent := 0.80 -- 80% material B in Mixture A

  -- Initial amounts in 8 kg of Mixture A
  let initial_oil := oil_percent * mixtureA
  let initial_materialB := materialB_percent * mixtureA

  -- New mixture after adding 2 kg oil
  let new_oil := initial_oil + addOil
  let new_materialB := initial_materialB

  -- Adding 6 kg of Mixture A
  let added_oil := oil_percent * addMixA
  let added_materialB := materialB_percent * addMixA

  -- Total amounts in the new mixture
  let total_oil := new_oil + added_oil
  let total_materialB := new_materialB + added_materialB
  let total_weight := mixtureA + addOil + addMixA

  -- Percent calculation
  let percent_materialB := (total_materialB / total_weight) * 100

  percent_materialB = 70 := sorry

end NUMINAMATH_GPT_percentage_material_B_new_mixture_l635_63596


namespace NUMINAMATH_GPT_inequality_solution_l635_63539

theorem inequality_solution (x : ℝ) (h : x ≠ 4) : (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ Set.Iic (-4) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l635_63539


namespace NUMINAMATH_GPT_exists_pairs_angle_120_degrees_l635_63532

theorem exists_pairs_angle_120_degrees :
  ∃ a b : ℤ, a + b ≠ 0 ∧ a + b ≠ a ^ 2 - a * b + b ^ 2 ∧ (a + b) * 13 = 3 * (a ^ 2 - a * b + b ^ 2) :=
sorry

end NUMINAMATH_GPT_exists_pairs_angle_120_degrees_l635_63532


namespace NUMINAMATH_GPT_initial_water_percentage_l635_63504

theorem initial_water_percentage (W : ℕ) (V1 V2 V3 W3 : ℕ) (h1 : V1 = 10) (h2 : V2 = 15) (h3 : V3 = V1 + V2) (h4 : V3 = 25) (h5 : W3 = 2) (h6 : (W * V1) / 100 = (W3 * V3) / 100) : W = 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_water_percentage_l635_63504


namespace NUMINAMATH_GPT_problem1_problem2_l635_63563

open Real

noncomputable def alpha (hα : 0 < α ∧ α < π / 3) :=
  α

noncomputable def vec_a (hα : 0 < α ∧ α < π / 3) :=
  (sqrt 6 * sin (alpha hα), sqrt 2)

noncomputable def vec_b (hα : 0 < α ∧ α < π / 3) :=
  (1, cos (alpha hα) - sqrt 6 / 2)

theorem problem1 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  tan (alpha hα + π / 6) = sqrt 15 / 5 :=
sorry

theorem problem2 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  cos (2 * alpha hα + 7 * π / 12) = (sqrt 2 - sqrt 30) / 8 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l635_63563


namespace NUMINAMATH_GPT_sarah_trucks_l635_63582

-- Define the initial number of trucks denoted by T
def initial_trucks (T : ℝ) : Prop :=
  let left_after_jeff := T - 13.5
  let left_after_ashley := left_after_jeff - 0.25 * left_after_jeff
  left_after_ashley = 38

-- Theorem stating the initial number of trucks Sarah had is 64
theorem sarah_trucks : ∃ T : ℝ, initial_trucks T ∧ T = 64 :=
by
  sorry

end NUMINAMATH_GPT_sarah_trucks_l635_63582


namespace NUMINAMATH_GPT_two_distinct_real_roots_of_modified_quadratic_l635_63566

theorem two_distinct_real_roots_of_modified_quadratic (a b k : ℝ) (h1 : a^2 - b > 0) (h2 : k > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + 2 * a * x₁ + b + k * (x₁ + a)^2 = 0) ∧ (x₂^2 + 2 * a * x₂ + b + k * (x₂ + a)^2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_two_distinct_real_roots_of_modified_quadratic_l635_63566


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_l635_63548

theorem cylindrical_to_rectangular :
  ∀ (r θ z : ℝ), r = 5 → θ = (3 * Real.pi) / 4 → z = 2 →
    (r * Real.cos θ, r * Real.sin θ, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  intros r θ z hr hθ hz
  rw [hr, hθ, hz]
  -- Proof steps would go here, but are omitted as they are not required.
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_l635_63548


namespace NUMINAMATH_GPT_valentines_distribution_l635_63567

theorem valentines_distribution (valentines_initial : ℝ) (valentines_needed : ℝ) (students : ℕ) 
  (h_initial : valentines_initial = 58.0) (h_needed : valentines_needed = 16.0) (h_students : students = 74) : 
  (valentines_initial + valentines_needed) / students = 1 :=
by
  sorry

end NUMINAMATH_GPT_valentines_distribution_l635_63567


namespace NUMINAMATH_GPT_required_run_rate_equivalence_l635_63506

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.5
def overs_first_phase : ℝ := 10
def total_target_runs : ℝ := 350
def remaining_overs : ℝ := 35
def total_overs : ℝ := 45

-- Define the already scored runs
def runs_scored_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_phase

-- Define the required runs for the remaining overs
def runs_needed : ℝ := total_target_runs - runs_scored_first_10_overs

-- Theorem stating the required run rate in the remaining 35 overs
theorem required_run_rate_equivalence :
  runs_needed / remaining_overs = 9 :=
by
  sorry

end NUMINAMATH_GPT_required_run_rate_equivalence_l635_63506


namespace NUMINAMATH_GPT_range_of_a_l635_63515

-- Define the function f as given in the problem
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

-- The mathematical statement to be proven in Lean
theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, ∃ m M : ℝ, m = (f a x) ∧ M = (f a y) ∧ (∀ z : ℝ, f a z ≥ m) ∧ (∀ z : ℝ, f a z ≤ M)) ↔ 
  (a < -3 ∨ a > 6) :=
sorry

end NUMINAMATH_GPT_range_of_a_l635_63515


namespace NUMINAMATH_GPT_p_implies_q_and_not_converse_l635_63569

def p (a : ℝ) := a ≤ 1
def q (a : ℝ) := abs a ≤ 1

theorem p_implies_q_and_not_converse (a : ℝ) : (p a → q a) ∧ ¬(q a → p a) :=
by
  repeat { sorry }

end NUMINAMATH_GPT_p_implies_q_and_not_converse_l635_63569


namespace NUMINAMATH_GPT_find_divisors_l635_63565

theorem find_divisors (N : ℕ) :
  (∃ k : ℕ, 2014 = k * (N + 1) ∧ k < N) ↔ (N = 2013 ∨ N = 1006 ∨ N = 105 ∨ N = 52) := by
  sorry

end NUMINAMATH_GPT_find_divisors_l635_63565


namespace NUMINAMATH_GPT_sum_of_roots_of_cis_equation_l635_63572

theorem sum_of_roots_of_cis_equation 
  (cis : ℝ → ℂ)
  (phi : ℕ → ℝ)
  (h_conditions : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 0 ≤ phi k ∧ phi k < 360)
  (h_equation : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → (cis (phi k)) ^ 5 = (1 / Real.sqrt 2) + (Complex.I / Real.sqrt 2))
  : (phi 1 + phi 2 + phi 3 + phi 4 + phi 5) = 450 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_cis_equation_l635_63572


namespace NUMINAMATH_GPT_interest_rate_condition_l635_63557

theorem interest_rate_condition 
    (P1 P2 : ℝ) 
    (R2 : ℝ) 
    (T1 T2 : ℝ) 
    (SI500 SI160 : ℝ) 
    (H1: SI500 = (P1 * R2 * T1) / 100) 
    (H2: SI160 = (P2 * (25 / 100))):
  25 * (160 / 100) / 12.5  = 6.4 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_condition_l635_63557


namespace NUMINAMATH_GPT_product_ab_zero_l635_63510

theorem product_ab_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_ab_zero_l635_63510


namespace NUMINAMATH_GPT_eq_has_positive_integer_solution_l635_63541

theorem eq_has_positive_integer_solution (a : ℤ) :
  (∃ x : ℕ+, (x : ℤ) - 4 - 2 * (a * x - 1) = 2) → a = 0 :=
by
  sorry

end NUMINAMATH_GPT_eq_has_positive_integer_solution_l635_63541


namespace NUMINAMATH_GPT_not_exists_cube_in_sequence_l635_63599

-- Lean statement of the proof problem
theorem not_exists_cube_in_sequence : ∀ n : ℕ, ¬ ∃ k : ℤ, 2 ^ (2 ^ n) + 1 = k ^ 3 := 
by 
    intro n
    intro ⟨k, h⟩
    sorry

end NUMINAMATH_GPT_not_exists_cube_in_sequence_l635_63599


namespace NUMINAMATH_GPT_malcolm_needs_more_lights_l635_63584

def red_lights := 12
def blue_lights := 3 * red_lights
def green_lights := 6
def white_lights := 59

def colored_lights := red_lights + blue_lights + green_lights
def need_more_lights := white_lights - colored_lights

theorem malcolm_needs_more_lights :
  need_more_lights = 5 :=
by
  sorry

end NUMINAMATH_GPT_malcolm_needs_more_lights_l635_63584


namespace NUMINAMATH_GPT_solution_set_abs_ineq_l635_63586

theorem solution_set_abs_ineq (x : ℝ) : abs (2 - x) ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_GPT_solution_set_abs_ineq_l635_63586


namespace NUMINAMATH_GPT_minimum_number_of_tiles_l635_63556

def tile_width_in_inches : ℕ := 6
def tile_height_in_inches : ℕ := 4
def region_width_in_feet : ℕ := 3
def region_height_in_feet : ℕ := 8

def inches_to_feet (i : ℕ) : ℚ :=
  i / 12

def tile_width_in_feet : ℚ :=
  inches_to_feet tile_width_in_inches

def tile_height_in_feet : ℚ :=
  inches_to_feet tile_height_in_inches

def tile_area_in_square_feet : ℚ :=
  tile_width_in_feet * tile_height_in_feet

def region_area_in_square_feet : ℚ :=
  region_width_in_feet * region_height_in_feet

def number_of_tiles : ℚ :=
  region_area_in_square_feet / tile_area_in_square_feet

theorem minimum_number_of_tiles :
  number_of_tiles = 144 := by
    sorry

end NUMINAMATH_GPT_minimum_number_of_tiles_l635_63556


namespace NUMINAMATH_GPT_tom_made_washing_cars_l635_63547

-- Definitions of the conditions
def initial_amount : ℕ := 74
def final_amount : ℕ := 86

-- Statement to be proved
theorem tom_made_washing_cars : final_amount - initial_amount = 12 := by
  sorry

end NUMINAMATH_GPT_tom_made_washing_cars_l635_63547


namespace NUMINAMATH_GPT_find_m_for_parallel_lines_l635_63560

-- The given lines l1 and l2
def line1 (m: ℝ) : Prop := ∀ x y : ℝ, (3 + m) * x - 4 * y = 5 - 3 * m
def line2 : Prop := ∀ x y : ℝ, 2 * x - y = 8

-- Definition for parallel lines
def parallel_lines (l₁ l₂ : Prop) : Prop := 
  ∃ m : ℝ, (3 + m) / 4 = 2

-- The main theorem to prove
theorem find_m_for_parallel_lines (m: ℝ) (h: parallel_lines (line1 m) line2) : m = 5 :=
by sorry

end NUMINAMATH_GPT_find_m_for_parallel_lines_l635_63560


namespace NUMINAMATH_GPT_total_selling_price_correct_l635_63517

-- Definitions of initial purchase prices in different currencies
def init_price_eur : ℕ := 600
def init_price_gbp : ℕ := 450
def init_price_usd : ℕ := 750

-- Definitions of initial exchange rates
def init_exchange_rate_eur_to_usd : ℝ := 1.1
def init_exchange_rate_gbp_to_usd : ℝ := 1.3

-- Definitions of profit percentages for each article
def profit_percent_eur : ℝ := 0.08
def profit_percent_gbp : ℝ := 0.1
def profit_percent_usd : ℝ := 0.15

-- Definitions of new exchange rates at the time of selling
def new_exchange_rate_eur_to_usd : ℝ := 1.15
def new_exchange_rate_gbp_to_usd : ℝ := 1.25

-- Calculation of purchase prices in USD
def purchase_price_in_usd₁ : ℝ := init_price_eur * init_exchange_rate_eur_to_usd
def purchase_price_in_usd₂ : ℝ := init_price_gbp * init_exchange_rate_gbp_to_usd
def purchase_price_in_usd₃ : ℝ := init_price_usd

-- Calculation of selling prices including profit in USD
def selling_price_in_usd₁ : ℝ := (init_price_eur + (init_price_eur * profit_percent_eur)) * new_exchange_rate_eur_to_usd
def selling_price_in_usd₂ : ℝ := (init_price_gbp + (init_price_gbp * profit_percent_gbp)) * new_exchange_rate_gbp_to_usd
def selling_price_in_usd₃ : ℝ := init_price_usd * (1 + profit_percent_usd)

-- Total selling price in USD
def total_selling_price_in_usd : ℝ :=
  selling_price_in_usd₁ + selling_price_in_usd₂ + selling_price_in_usd₃

-- Proof goal: total selling price should equal 2225.85 USD
theorem total_selling_price_correct :
  total_selling_price_in_usd = 2225.85 :=
by
  sorry

end NUMINAMATH_GPT_total_selling_price_correct_l635_63517


namespace NUMINAMATH_GPT_fill_tank_with_leak_l635_63530

namespace TankFilling

-- Conditions
def pump_fill_rate (P : ℝ) : Prop := P = 1 / 4
def leak_drain_rate (L : ℝ) : Prop := L = 1 / 5
def net_fill_rate (P L R : ℝ) : Prop := P - L = R
def fill_time (R T : ℝ) : Prop := T = 1 / R

-- Statement
theorem fill_tank_with_leak (P L R T : ℝ) (hP : pump_fill_rate P) (hL : leak_drain_rate L) (hR : net_fill_rate P L R) (hT : fill_time R T) :
  T = 20 :=
  sorry

end TankFilling

end NUMINAMATH_GPT_fill_tank_with_leak_l635_63530


namespace NUMINAMATH_GPT_greatest_distance_between_centers_l635_63508

-- Define the conditions
noncomputable def circle_radius : ℝ := 4
noncomputable def rectangle_length : ℝ := 20
noncomputable def rectangle_width : ℝ := 16

-- Define the centers of the circles
noncomputable def circle_center1 : ℝ × ℝ := (4, circle_radius)
noncomputable def circle_center2 : ℝ × ℝ := (rectangle_length - 4, circle_radius)

-- Calculate the greatest possible distance
noncomputable def distance : ℝ := Real.sqrt ((8 ^ 2) + (rectangle_width ^ 2))

-- Statement to prove
theorem greatest_distance_between_centers :
  distance = 8 * Real.sqrt 5 :=
  sorry

end NUMINAMATH_GPT_greatest_distance_between_centers_l635_63508


namespace NUMINAMATH_GPT_intersection_A_B_l635_63538

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def B : Set ℝ := {-3, -2, -1, 0, 1, 2}

-- Define the intersection we need to prove
def A_cap_B_target : Set ℝ := {-2, -1, 0, 1}

-- Prove the intersection of A and B equals the target set
theorem intersection_A_B :
  A ∩ B = A_cap_B_target := 
sorry

end NUMINAMATH_GPT_intersection_A_B_l635_63538


namespace NUMINAMATH_GPT_initial_num_files_l635_63561

-- Define the conditions: number of files organized in the morning, files to organize in the afternoon, and missing files.
def num_files_organized_in_morning (X : ℕ) : ℕ := X / 2
def num_files_to_organize_in_afternoon : ℕ := 15
def num_files_missing : ℕ := 15

-- Theorem to prove the initial number of files is 60.
theorem initial_num_files (X : ℕ) 
  (h1 : num_files_organized_in_morning X = X / 2)
  (h2 : num_files_to_organize_in_afternoon = 15)
  (h3 : num_files_missing = 15) :
  X = 60 :=
by
  sorry

end NUMINAMATH_GPT_initial_num_files_l635_63561


namespace NUMINAMATH_GPT_wine_age_problem_l635_63520

theorem wine_age_problem
  (C F T B Bo : ℕ)
  (h1 : F = 3 * C)
  (h2 : C = 4 * T)
  (h3 : B = (1 / 2 : ℝ) * T)
  (h4 : Bo = 2 * F)
  (h5 : C = 40) :
  F = 120 ∧ T = 10 ∧ B = 5 ∧ Bo = 240 := 
  by
    sorry

end NUMINAMATH_GPT_wine_age_problem_l635_63520


namespace NUMINAMATH_GPT_seashell_count_l635_63505

variable (initial_seashells additional_seashells total_seashells : ℕ)

theorem seashell_count (h1 : initial_seashells = 19) (h2 : additional_seashells = 6) : 
  total_seashells = initial_seashells + additional_seashells → total_seashells = 25 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end NUMINAMATH_GPT_seashell_count_l635_63505


namespace NUMINAMATH_GPT_five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l635_63536

theorem five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand :
  5.8 / 0.001 = 5.8 * 1000 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l635_63536


namespace NUMINAMATH_GPT_LCM_of_apple_and_cherry_pies_l635_63595

theorem LCM_of_apple_and_cherry_pies :
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 :=
by
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  have h : (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 := sorry
  exact h

end NUMINAMATH_GPT_LCM_of_apple_and_cherry_pies_l635_63595


namespace NUMINAMATH_GPT_total_eggs_collected_l635_63568

def benjamin_collects : Nat := 6
def carla_collects := 3 * benjamin_collects
def trisha_collects := benjamin_collects - 4

theorem total_eggs_collected :
  benjamin_collects + carla_collects + trisha_collects = 26 := by
  sorry

end NUMINAMATH_GPT_total_eggs_collected_l635_63568


namespace NUMINAMATH_GPT_larry_expression_correct_l635_63503

theorem larry_expression_correct (a b c d e : ℤ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : c = 2) (h₄ : d = 5) :
  (a - b + c - d + e = a - (b + (c - (d - e)))) → e = 3 :=
by
  sorry

end NUMINAMATH_GPT_larry_expression_correct_l635_63503


namespace NUMINAMATH_GPT_solve_inequality_l635_63581

noncomputable def solution_set : Set ℝ := {x | x < -4/3 ∨ x > -13/9}

theorem solve_inequality (x : ℝ) : 
  2 - 1 / (3 * x + 4) < 5 → x ∈ solution_set :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l635_63581


namespace NUMINAMATH_GPT_perfect_square_trinomial_implies_value_of_a_l635_63537

theorem perfect_square_trinomial_implies_value_of_a (a : ℝ) :
  (∃ (b : ℝ), (∃ (x : ℝ), (x^2 - ax + 9 = 0) ∧ (x + b)^2 = x^2 - ax + 9)) ↔ a = 6 ∨ a = -6 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_implies_value_of_a_l635_63537


namespace NUMINAMATH_GPT_other_car_speed_l635_63502

-- Definitions of the conditions
def red_car_speed : ℕ := 30
def initial_gap : ℕ := 20
def overtaking_time : ℕ := 1

-- Assertion of what needs to be proved
theorem other_car_speed : (initial_gap + red_car_speed * overtaking_time) = 50 :=
  sorry

end NUMINAMATH_GPT_other_car_speed_l635_63502


namespace NUMINAMATH_GPT_strips_area_coverage_l635_63525

-- Define paper strips and their properties
def length_strip : ℕ := 8
def width_strip : ℕ := 2
def number_of_strips : ℕ := 5

-- Total area without considering overlaps
def area_one_strip : ℕ := length_strip * width_strip
def total_area_without_overlap : ℕ := number_of_strips * area_one_strip

-- Overlapping areas
def area_center_overlap : ℕ := 4 * (2 * 2)
def area_additional_overlap : ℕ := 2 * (2 * 2)
def total_overlap_area : ℕ := area_center_overlap + area_additional_overlap

-- Actual area covered
def actual_area_covered : ℕ := total_area_without_overlap - total_overlap_area

-- Theorem stating the required proof
theorem strips_area_coverage : actual_area_covered = 56 :=
by sorry

end NUMINAMATH_GPT_strips_area_coverage_l635_63525


namespace NUMINAMATH_GPT_find_function_solution_l635_63553

def satisfies_condition (f : ℝ → ℝ) :=
  ∀ (x y : ℝ), f (f (x * y)) = |x| * f y + 3 * f (x * y)

theorem find_function_solution (f : ℝ → ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 4 * |x|) ∨ (∀ x : ℝ, f x = -4 * |x|) :=
by
  sorry

end NUMINAMATH_GPT_find_function_solution_l635_63553


namespace NUMINAMATH_GPT_gumballs_initial_count_l635_63549

noncomputable def initial_gumballs := (34.3 / (0.7 ^ 3))

theorem gumballs_initial_count :
  initial_gumballs = 100 :=
sorry

end NUMINAMATH_GPT_gumballs_initial_count_l635_63549


namespace NUMINAMATH_GPT_total_cost_l635_63535

theorem total_cost (cost_pencil cost_pen : ℕ) 
(h1 : cost_pen = cost_pencil + 9) 
(h2 : cost_pencil = 2) : 
cost_pencil + cost_pen = 13 := 
by 
  -- Proof would go here 
  sorry

end NUMINAMATH_GPT_total_cost_l635_63535


namespace NUMINAMATH_GPT_b_2016_eq_neg_4_l635_63527

def b : ℕ → ℤ
| 0     => 1
| 1     => 5
| (n+2) => b (n+1) - b n

theorem b_2016_eq_neg_4 : b 2015 = -4 :=
sorry

end NUMINAMATH_GPT_b_2016_eq_neg_4_l635_63527


namespace NUMINAMATH_GPT_smallest_integer_y_l635_63511

theorem smallest_integer_y (y : ℤ) :
  (∃ y : ℤ, ((y / 4 : ℚ) + (3 / 7 : ℚ) > 2 / 3) ∧ (∀ z : ℤ, (z > 20 / 21) → y ≤ z)) :=
sorry

end NUMINAMATH_GPT_smallest_integer_y_l635_63511


namespace NUMINAMATH_GPT_james_fence_problem_l635_63526

theorem james_fence_problem (w : ℝ) (hw : 0 ≤ w) (h_area : w * (2 * w + 10) ≥ 120) : w = 5 :=
by
  sorry

end NUMINAMATH_GPT_james_fence_problem_l635_63526


namespace NUMINAMATH_GPT_jacket_initial_reduction_l635_63562

theorem jacket_initial_reduction (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.9 * 1.481481481481481 = P → x = 25 :=
by
  sorry

end NUMINAMATH_GPT_jacket_initial_reduction_l635_63562


namespace NUMINAMATH_GPT_value_decrease_proof_l635_63528

noncomputable def value_comparison (diana_usd : ℝ) (etienne_eur : ℝ) (eur_to_usd : ℝ) : ℝ :=
  let etienne_usd := etienne_eur * eur_to_usd
  let percentage_decrease := ((diana_usd - etienne_usd) / diana_usd) * 100
  percentage_decrease

theorem value_decrease_proof :
  value_comparison 700 300 1.5 = 35.71 :=
by
  sorry

end NUMINAMATH_GPT_value_decrease_proof_l635_63528


namespace NUMINAMATH_GPT_find_range_a_l635_63579

-- Define the proposition p
def p (m : ℝ) : Prop :=
1 < m ∧ m < 3 / 2

-- Define the proposition q
def q (m a : ℝ) : Prop :=
(m - a) * (m - (a + 1)) < 0

-- Define the sufficient but not necessary condition
def sufficient (a : ℝ) : Prop :=
(a ≤ 1) ∧ (3 / 2 ≤ a + 1)

theorem find_range_a (a : ℝ) :
  (∀ m, p m → q m a) → sufficient a → (1 / 2 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_GPT_find_range_a_l635_63579


namespace NUMINAMATH_GPT_range_of_f_l635_63564

noncomputable def f (x : ℤ) : ℤ := x ^ 2 + 1

def domain : Set ℤ := {-1, 0, 1, 2}

def range_f : Set ℤ := {1, 2, 5}

theorem range_of_f : Set.image f domain = range_f :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l635_63564


namespace NUMINAMATH_GPT_probability_of_square_product_l635_63583

theorem probability_of_square_product :
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9 -- (1,1), (1,4), (2,2), (4,1), (3,3), (9,1), (4,4), (5,5), (6,6)
  favorable_outcomes / total_outcomes = 1 / 8 :=
by
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9
  have h1 : favorable_outcomes / total_outcomes = 1 / 8 := sorry
  exact h1

end NUMINAMATH_GPT_probability_of_square_product_l635_63583


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l635_63571

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
(h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
(h_avg_2_1 : (a + b) / 2 = 3.4)
(h_avg_2_2 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 := 
sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l635_63571


namespace NUMINAMATH_GPT_fraction_green_after_tripling_l635_63516

theorem fraction_green_after_tripling 
  (x : ℕ)
  (h₁ : ∃ x, 0 < x) -- Total number of marbles is a positive integer
  (h₂ : ∀ g y, g + y = x ∧ g = 1/4 * x ∧ y = 3/4 * x) -- Initial distribution
  (h₃ : ∀ y : ℕ, g' = 3 * g ∧ y' = y) -- Triple the green marbles, yellow stays the same
  : (g' / (g' + y')) = 1/2 := 
sorry

end NUMINAMATH_GPT_fraction_green_after_tripling_l635_63516


namespace NUMINAMATH_GPT_avg_prime_factors_of_multiples_of_10_l635_63585

theorem avg_prime_factors_of_multiples_of_10 : 
  (2 + 5) / 2 = 3.5 :=
by
  -- The prime factors of 10 are 2 and 5.
  -- Therefore, the average of these prime factors is (2 + 5) / 2.
  sorry

end NUMINAMATH_GPT_avg_prime_factors_of_multiples_of_10_l635_63585


namespace NUMINAMATH_GPT_computation_result_l635_63507

theorem computation_result :
  let a := -6
  let b := 25
  let c := -39
  let d := 40
  9 * a + 3 * b + 6 * c + d = -173 := by
  sorry

end NUMINAMATH_GPT_computation_result_l635_63507


namespace NUMINAMATH_GPT_appropriate_sampling_method_l635_63570

def total_families := 500
def high_income_families := 125
def middle_income_families := 280
def low_income_families := 95
def sample_size := 100
def influenced_by_income := True

theorem appropriate_sampling_method
  (htotal : total_families = 500)
  (hhigh : high_income_families = 125)
  (hmiddle : middle_income_families = 280)
  (hlow : low_income_families = 95)
  (hsample : sample_size = 100)
  (hinfluence : influenced_by_income = True) :
  ∃ method, method = "Stratified sampling method" :=
sorry

end NUMINAMATH_GPT_appropriate_sampling_method_l635_63570


namespace NUMINAMATH_GPT_john_tips_problem_l635_63550

theorem john_tips_problem
  (A M : ℝ)
  (H1 : ∀ (A : ℝ), M * A = 0.5 * (6 * A + M * A)) :
  M = 6 := 
by
  sorry

end NUMINAMATH_GPT_john_tips_problem_l635_63550
