import Mathlib

namespace NUMINAMATH_GPT_cally_pants_count_l2228_222831

variable (cally_white_shirts : ℕ)
variable (cally_colored_shirts : ℕ)
variable (cally_shorts : ℕ)
variable (danny_white_shirts : ℕ)
variable (danny_colored_shirts : ℕ)
variable (danny_shorts : ℕ)
variable (danny_pants : ℕ)
variable (total_clothes_washed : ℕ)
variable (cally_pants : ℕ)

-- Given conditions
#check cally_white_shirts = 10
#check cally_colored_shirts = 5
#check cally_shorts = 7
#check danny_white_shirts = 6
#check danny_colored_shirts = 8
#check danny_shorts = 10
#check danny_pants = 6
#check total_clothes_washed = 58
#check cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed

-- Proof goal
theorem cally_pants_count (cally_white_shirts cally_colored_shirts cally_shorts danny_white_shirts danny_colored_shirts danny_shorts danny_pants cally_pants total_clothes_washed : ℕ) :
  cally_white_shirts = 10 →
  cally_colored_shirts = 5 →
  cally_shorts = 7 →
  danny_white_shirts = 6 →
  danny_colored_shirts = 8 →
  danny_shorts = 10 →
  danny_pants = 6 →
  total_clothes_washed = 58 →
  (cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed) →
  cally_pants = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_cally_pants_count_l2228_222831


namespace NUMINAMATH_GPT_dimensions_increased_three_times_l2228_222822

variables (L B H k : ℝ) (n : ℝ)
 
-- Given conditions
axiom cost_initial : 350 = k * 2 * (L + B) * H
axiom cost_increased : 3150 = k * 2 * n^2 * (L + B) * H

-- Proof statement
theorem dimensions_increased_three_times : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_dimensions_increased_three_times_l2228_222822


namespace NUMINAMATH_GPT_largest_integer_satisfying_condition_l2228_222868

-- Definition of the conditions
def has_four_digits_in_base_10 (n : ℕ) : Prop :=
  10^3 ≤ n^2 ∧ n^2 < 10^4

-- Proof statement: N is the largest integer satisfying the condition
theorem largest_integer_satisfying_condition : ∃ (N : ℕ), 
  has_four_digits_in_base_10 N ∧ (∀ (m : ℕ), has_four_digits_in_base_10 m → m ≤ N) ∧ N = 99 := 
sorry

end NUMINAMATH_GPT_largest_integer_satisfying_condition_l2228_222868


namespace NUMINAMATH_GPT_unique_solution_pairs_l2228_222895

theorem unique_solution_pairs :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 = 4 * c) ∧ (c^2 = 4 * b) :=
sorry

end NUMINAMATH_GPT_unique_solution_pairs_l2228_222895


namespace NUMINAMATH_GPT_square_area_from_inscribed_circle_l2228_222896

theorem square_area_from_inscribed_circle (r : ℝ) (π_pos : 0 < Real.pi) (circle_area : Real.pi * r^2 = 9 * Real.pi) : 
  (2 * r)^2 = 36 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_square_area_from_inscribed_circle_l2228_222896


namespace NUMINAMATH_GPT_ratio_of_side_length_to_brush_width_l2228_222884

theorem ratio_of_side_length_to_brush_width (s w : ℝ) (h : (w^2 + ((s - w)^2) / 2) = s^2 / 3) : s / w = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_side_length_to_brush_width_l2228_222884


namespace NUMINAMATH_GPT_total_seashells_found_intact_seashells_found_l2228_222867

-- Define the constants for seashells found
def tom_seashells : ℕ := 15
def fred_seashells : ℕ := 43

-- Define total_intercept
def total_intercept : ℕ := 29

-- Statement that the total seashells found by Tom and Fred is 58
theorem total_seashells_found : tom_seashells + fred_seashells = 58 := by
  sorry

-- Statement that the intact seashells are obtained by subtracting cracked ones
theorem intact_seashells_found : tom_seashells + fred_seashells - total_intercept = 29 := by
  sorry

end NUMINAMATH_GPT_total_seashells_found_intact_seashells_found_l2228_222867


namespace NUMINAMATH_GPT_range_of_x_l2228_222889

theorem range_of_x : ∀ x : ℝ, (¬ (x + 3 = 0)) ∧ (4 - x ≥ 0) ↔ x ≤ 4 ∧ x ≠ -3 := by
  sorry

end NUMINAMATH_GPT_range_of_x_l2228_222889


namespace NUMINAMATH_GPT_least_integer_value_satisfying_inequality_l2228_222883

theorem least_integer_value_satisfying_inequality : ∃ x : ℤ, 3 * |x| + 6 < 24 ∧ (∀ y : ℤ, 3 * |y| + 6 < 24 → x ≤ y) :=
  sorry

end NUMINAMATH_GPT_least_integer_value_satisfying_inequality_l2228_222883


namespace NUMINAMATH_GPT_distance_between_trees_l2228_222821

-- Definitions based on conditions
def yard_length : ℝ := 360
def number_of_trees : ℕ := 31
def number_of_gaps : ℕ := number_of_trees - 1

-- The proposition to prove
theorem distance_between_trees : yard_length / number_of_gaps = 12 := sorry

end NUMINAMATH_GPT_distance_between_trees_l2228_222821


namespace NUMINAMATH_GPT_original_flow_rate_l2228_222851

theorem original_flow_rate :
  ∃ F : ℚ, 
  (F * 0.75 * 0.4 * 0.6 - 1 = 2) ∧
  (F = 50/3) :=
by
  sorry

end NUMINAMATH_GPT_original_flow_rate_l2228_222851


namespace NUMINAMATH_GPT_quadratic_identity_l2228_222845

theorem quadratic_identity (x : ℝ) : 
  (3*x + 1)^2 + 2*(3*x + 1)*(x - 3) + (x - 3)^2 = 16*x^2 - 16*x + 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_identity_l2228_222845


namespace NUMINAMATH_GPT_present_worth_of_bill_l2228_222823

theorem present_worth_of_bill (P : ℝ) (TD BD : ℝ) 
  (hTD : TD = 36) (hBD : BD = 37.62) 
  (hFormula : BD = (TD * (P + TD)) / P) : P = 800 :=
by
  sorry

end NUMINAMATH_GPT_present_worth_of_bill_l2228_222823


namespace NUMINAMATH_GPT_base_angle_isosceles_l2228_222819

-- Define an isosceles triangle with one angle being 100 degrees
def isosceles_triangle (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
  (A = B ∨ B = C ∨ C = A) ∧ (angle_A + angle_B + angle_C = 180) ∧ (angle_A = 100)

-- The main theorem statement
theorem base_angle_isosceles {A B C : Type} (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) :
  isosceles_triangle A B C angle_A angle_B angle_C → (angle_B = 40 ∨ angle_C = 40) :=
  sorry

end NUMINAMATH_GPT_base_angle_isosceles_l2228_222819


namespace NUMINAMATH_GPT_factor_expression_l2228_222849

theorem factor_expression (x : ℝ) : 45 * x^2 + 135 * x = 45 * x * (x + 3) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2228_222849


namespace NUMINAMATH_GPT_remaining_student_number_l2228_222897

theorem remaining_student_number (s1 s2 s3 : ℕ) (h1 : s1 = 5) (h2 : s2 = 29) (h3 : s3 = 41) (N : ℕ) (hN : N = 48) :
  ∃ s4, s4 < N ∧ s4 ≠ s1 ∧ s4 ≠ s2 ∧ s4 ≠ s3 ∧ (s4 = 17) :=
by
  sorry

end NUMINAMATH_GPT_remaining_student_number_l2228_222897


namespace NUMINAMATH_GPT_max_crate_weight_on_single_trip_l2228_222812

-- Define the conditions
def trailer_capacity := {n | n = 3 ∨ n = 4 ∨ n = 5}
def min_crate_weight : ℤ := 1250

-- Define the maximum weight calculation
def max_weight (n : ℤ) (w : ℤ) : ℤ := n * w

-- Proof statement
theorem max_crate_weight_on_single_trip :
  ∃ w, (5 ∈ trailer_capacity) → max_weight 5 min_crate_weight = w ∧ w = 6250 := 
by
  sorry

end NUMINAMATH_GPT_max_crate_weight_on_single_trip_l2228_222812


namespace NUMINAMATH_GPT_age_difference_l2228_222848

theorem age_difference
  (A B C : ℕ)
  (h1 : B = 12)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 32) :
  A - B = 2 :=
sorry

end NUMINAMATH_GPT_age_difference_l2228_222848


namespace NUMINAMATH_GPT_unique_intersections_l2228_222842

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 5 * x + y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2

theorem unique_intersections :
  (∃ x1 y1, line1 x1 y1 ∧ line2 x1 y1) ∧
  (∃ x2 y2, line2 x2 y2 ∧ line3 x2 y2) ∧
  ¬ (∃ x y, line1 x y ∧ line3 x y) ∧
  (∀ x y x' y', (line1 x y ∧ line2 x y ∧ line2 x' y' ∧ line3 x' y') → (x = x' ∧ y = y')) :=
by
  sorry

end NUMINAMATH_GPT_unique_intersections_l2228_222842


namespace NUMINAMATH_GPT_backpack_price_equation_l2228_222843

-- Define the original price of the backpack
variable (x : ℝ)

-- Define the conditions
def discount1 (x : ℝ) : ℝ := 0.8 * x
def discount2 (d : ℝ) : ℝ := d - 10
def final_price (p : ℝ) : Prop := p = 90

-- Final statement to be proved
theorem backpack_price_equation : final_price (discount2 (discount1 x)) ↔ 0.8 * x - 10 = 90 := sorry

end NUMINAMATH_GPT_backpack_price_equation_l2228_222843


namespace NUMINAMATH_GPT_triangle_properties_l2228_222816

variable (a b c A B C : ℝ)
variable (CD BD : ℝ)

-- triangle properties and given conditions
variable (b_squared_eq_ac : b ^ 2 = a * c)
variable (cos_A_minus_C : Real.cos (A - C) = Real.cos B + 1 / 2)

theorem triangle_properties :
  B = π / 3 ∧ 
  A = π / 3 ∧ 
  (CD = 6 → ∃ x, x > 0 ∧ x = 4 * Real.sqrt 3 + 6) ∧
  (BD = 6 → ∀ area, area ≠ 9 / 4) :=
  by
    sorry

end NUMINAMATH_GPT_triangle_properties_l2228_222816


namespace NUMINAMATH_GPT_billion_in_scientific_notation_l2228_222834

theorem billion_in_scientific_notation :
  (10^9 = 1 * 10^9) :=
by
  sorry

end NUMINAMATH_GPT_billion_in_scientific_notation_l2228_222834


namespace NUMINAMATH_GPT_intersection_eq_l2228_222852

def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }
def B : Set ℝ := { x | Real.log (1 - x) > 0 }

theorem intersection_eq : A ∩ B = Set.Icc (-1) 0 :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l2228_222852


namespace NUMINAMATH_GPT_miles_ridden_further_l2228_222861

theorem miles_ridden_further (distance_ridden distance_walked : ℝ) (h1 : distance_ridden = 3.83) (h2 : distance_walked = 0.17) :
  distance_ridden - distance_walked = 3.66 := 
by sorry

end NUMINAMATH_GPT_miles_ridden_further_l2228_222861


namespace NUMINAMATH_GPT_exists_divisible_diff_l2228_222874

theorem exists_divisible_diff (l : List ℤ) (h_len : l.length = 2022) :
  ∃ i j, i ≠ j ∧ (l.nthLe i sorry - l.nthLe j sorry) % 2021 = 0 :=
by
  apply sorry -- Placeholder for proof

end NUMINAMATH_GPT_exists_divisible_diff_l2228_222874


namespace NUMINAMATH_GPT_relationship_between_a_b_l2228_222856

theorem relationship_between_a_b (a b x : ℝ) 
  (h₁ : x = (a + b) / 2)
  (h₂ : x^2 = (a^2 - b^2) / 2):
  a = -b ∨ a = 3 * b :=
sorry

end NUMINAMATH_GPT_relationship_between_a_b_l2228_222856


namespace NUMINAMATH_GPT_trigonometric_identity_l2228_222811

noncomputable def tan_alpha : ℝ := 4

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = tan_alpha) :
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / Real.cos (-α) = 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2228_222811


namespace NUMINAMATH_GPT_batsman_average_after_12th_inning_l2228_222893

theorem batsman_average_after_12th_inning (average_initial : ℕ) (score_12th : ℕ) (average_increase : ℕ) (total_innings : ℕ) 
    (h_avg_init : average_initial = 29) (h_score_12th : score_12th = 65) (h_avg_inc : average_increase = 3) 
    (h_total_innings : total_innings = 12) : 
    (average_initial + average_increase = 32) := 
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_12th_inning_l2228_222893


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2228_222878

variable (x y : ℤ)

def p : Prop := x ≠ 2 ∨ y ≠ 4
def q : Prop := x + y ≠ 6

theorem necessary_but_not_sufficient_condition :
  (p x y → q x y) ∧ (¬q x y → ¬p x y) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2228_222878


namespace NUMINAMATH_GPT_number_of_people_only_went_to_aquarium_is_5_l2228_222887

-- Define the conditions
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group_size : ℕ := 10
def total_earnings : ℕ := 240

-- Define the problem in Lean
theorem number_of_people_only_went_to_aquarium_is_5 :
  ∃ x : ℕ, (total_earnings - (group_size * (admission_fee + tour_fee)) = x * admission_fee) → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_only_went_to_aquarium_is_5_l2228_222887


namespace NUMINAMATH_GPT_space_shuttle_speed_l2228_222817

-- Define the conditions in Lean
def speed_kmph : ℕ := 43200 -- Speed in kilometers per hour
def seconds_per_hour : ℕ := 60 * 60 -- Number of seconds in an hour

-- Define the proof problem
theorem space_shuttle_speed :
  speed_kmph / seconds_per_hour = 12 := by
  sorry

end NUMINAMATH_GPT_space_shuttle_speed_l2228_222817


namespace NUMINAMATH_GPT_a_plus_b_l2228_222885

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem a_plus_b (a b : ℝ) (h : f (a - 1) + f b = 0) : a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_a_plus_b_l2228_222885


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2228_222872

theorem geometric_sequence_sum :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 * q + a 1 * q ^ 3 = 20 →
    a 1 * q ^ 2 + a 1 * q ^ 4 = 40 →
    a 1 * q ^ 4 + a 1 * q ^ 6 = 160 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2228_222872


namespace NUMINAMATH_GPT_a7_b7_equals_29_l2228_222894

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry

def cond1 := a + b = 1
def cond2 := a^2 + b^2 = 3
def cond3 := a^3 + b^3 = 4
def cond4 := a^4 + b^4 = 7
def cond5 := a^5 + b^5 = 11

theorem a7_b7_equals_29 : cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 → a^7 + b^7 = 29 :=
by
  sorry

end NUMINAMATH_GPT_a7_b7_equals_29_l2228_222894


namespace NUMINAMATH_GPT_inequality_change_l2228_222854

theorem inequality_change (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end NUMINAMATH_GPT_inequality_change_l2228_222854


namespace NUMINAMATH_GPT_valid_triples_count_l2228_222838

def validTriple (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 15 ∧ 
  1 ≤ b ∧ b ≤ 15 ∧ 
  1 ≤ c ∧ c ≤ 15 ∧ 
  (b % a = 0 ∨ (∃ k : ℕ, k ≤ 15 ∧ c % k = 0))

def countValidTriples : ℕ := 
  (15 + 7 + 5 + 3 + 3 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1) * 2 - 15

theorem valid_triples_count : countValidTriples = 75 :=
  by
  sorry

end NUMINAMATH_GPT_valid_triples_count_l2228_222838


namespace NUMINAMATH_GPT_minimize_abs_expression_l2228_222827

theorem minimize_abs_expression {x : ℝ} : 
  ((|x - 2|) + 3) ≥ ((|2 - 2|) + 3) := 
sorry

end NUMINAMATH_GPT_minimize_abs_expression_l2228_222827


namespace NUMINAMATH_GPT_solve_problem_l2228_222837

namespace Example

-- Definitions based on given conditions
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def condition_2 (f : ℝ → ℝ) : Prop := f 2 = -1

def condition_3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = -f (2 - x)

-- Main theorem statement
theorem solve_problem (f : ℝ → ℝ)
  (h1 : isEvenFunction f)
  (h2 : condition_2 f)
  (h3 : condition_3 f) : f 2016 = 1 :=
sorry

end Example

end NUMINAMATH_GPT_solve_problem_l2228_222837


namespace NUMINAMATH_GPT_find_all_triplets_l2228_222876

theorem find_all_triplets (a b c : ℕ)
  (h₀_a : a > 0)
  (h₀_b : b > 0)
  (h₀_c : c > 0) :
  6^a = 1 + 2^b + 3^c ↔ 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 5 ∧ c = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_all_triplets_l2228_222876


namespace NUMINAMATH_GPT_general_formula_neg_seq_l2228_222818

theorem general_formula_neg_seq (a : ℕ → ℝ) (h_neg : ∀ n, a n < 0)
  (h_recurrence : ∀ n, 2 * a n = 3 * a (n + 1))
  (h_product : a 2 * a 5 = 8 / 27) :
  ∀ n, a n = - ((2/3)^(n-2) : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_general_formula_neg_seq_l2228_222818


namespace NUMINAMATH_GPT_jonas_needs_35_pairs_of_socks_l2228_222800

def JonasWardrobeItems (socks_pairs shoes_pairs pants_items tshirts : ℕ) : ℕ :=
  2 * socks_pairs + 2 * shoes_pairs + pants_items + tshirts

def itemsNeededToDouble (initial_items : ℕ) : ℕ :=
  2 * initial_items - initial_items

theorem jonas_needs_35_pairs_of_socks (socks_pairs : ℕ) 
                                      (shoes_pairs : ℕ) 
                                      (pants_items : ℕ) 
                                      (tshirts : ℕ) 
                                      (final_socks_pairs : ℕ) 
                                      (initial_items : ℕ := JonasWardrobeItems socks_pairs shoes_pairs pants_items tshirts) 
                                      (needed_items : ℕ := itemsNeededToDouble initial_items) 
                                      (needed_pairs_of_socks := needed_items / 2) : 
                                      final_socks_pairs = 35 :=
by
  sorry

end NUMINAMATH_GPT_jonas_needs_35_pairs_of_socks_l2228_222800


namespace NUMINAMATH_GPT_find_value_of_expression_l2228_222855

theorem find_value_of_expression (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) :
  (a + b)^9 + a^6 = 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l2228_222855


namespace NUMINAMATH_GPT_sum_of_squares_l2228_222858

variables (x y z w : ℝ)

def condition1 := (x^2 / (2^2 - 1^2)) + (y^2 / (2^2 - 3^2)) + (z^2 / (2^2 - 5^2)) + (w^2 / (2^2 - 7^2)) = 1
def condition2 := (x^2 / (4^2 - 1^2)) + (y^2 / (4^2 - 3^2)) + (z^2 / (4^2 - 5^2)) + (w^2 / (4^2 - 7^2)) = 1
def condition3 := (x^2 / (6^2 - 1^2)) + (y^2 / (6^2 - 3^2)) + (z^2 / (6^2 - 5^2)) + (w^2 / (6^2 - 7^2)) = 1
def condition4 := (x^2 / (8^2 - 1^2)) + (y^2 / (8^2 - 3^2)) + (z^2 / (8^2 - 5^2)) + (w^2 / (8^2 - 7^2)) = 1

theorem sum_of_squares : condition1 x y z w → condition2 x y z w → 
                          condition3 x y z w → condition4 x y z w →
                          (x^2 + y^2 + z^2 + w^2 = 36) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2228_222858


namespace NUMINAMATH_GPT_tangent_line_eq_l2228_222836

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Define the point at which we are evaluating the tangent
def point : ℝ × ℝ := (1, -1)

-- Define the derivative of the function f(x)
def f' (x : ℝ) : ℝ := 2 * x - 3

-- The desired theorem
theorem tangent_line_eq :
  ∀ x y : ℝ, (x, y) = point → (y = -x) :=
by sorry

end NUMINAMATH_GPT_tangent_line_eq_l2228_222836


namespace NUMINAMATH_GPT_min_value_4x_plus_3y_l2228_222804

theorem min_value_4x_plus_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_4x_plus_3y_l2228_222804


namespace NUMINAMATH_GPT_arithmetic_seq_property_l2228_222866

-- Define the arithmetic sequence {a_n}
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the conditions
variable (a d : ℤ)
variable (h1 : arithmetic_seq a d 3 + arithmetic_seq a d 9 + arithmetic_seq a d 15 = 30)

-- Define the statement to be proved
theorem arithmetic_seq_property : 
  arithmetic_seq a d 17 - 2 * arithmetic_seq a d 13 = -10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_property_l2228_222866


namespace NUMINAMATH_GPT_find_common_difference_l2228_222844

variable {a : ℕ → ℝ} -- Define the sequence as a function from natural numbers to real numbers
variable {d : ℝ} -- Define the common difference as a real number

-- Sequence is arithmetic means there exists a common difference such that a_{n+1} = a_n + d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Conditions from the problem
variable (h1 : a 3 = 5)
variable (h2 : a 15 = 41)
variable (h3 : is_arithmetic_sequence a d)

-- Theorem statement
theorem find_common_difference : d = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_common_difference_l2228_222844


namespace NUMINAMATH_GPT_fourth_number_in_12th_row_is_92_l2228_222808

-- Define the number of elements per row and the row number
def elements_per_row := 8
def row_number := 12

-- Define the last number in a row function
def last_number_in_row (n : ℕ) := elements_per_row * n

-- Define the starting number in a row function
def starting_number_in_row (n : ℕ) := (elements_per_row * (n - 1)) + 1

-- Define the nth number in a specified row function
def nth_number_in_row (n : ℕ) (k : ℕ) := starting_number_in_row n + (k - 1)

-- Prove that the fourth number in the 12th row is 92
theorem fourth_number_in_12th_row_is_92 : nth_number_in_row 12 4 = 92 :=
by
  -- state the required equivalences
  sorry

end NUMINAMATH_GPT_fourth_number_in_12th_row_is_92_l2228_222808


namespace NUMINAMATH_GPT_correct_sampling_l2228_222805

-- Let n be the total number of students
def total_students : ℕ := 60

-- Define the systematic sampling function
def systematic_sampling (n m : ℕ) (start : ℕ) : List ℕ :=
  List.map (λ k => start + k * m) (List.range n)

-- Prove that the sequence generated is equal to [3, 13, 23, 33, 43, 53]
theorem correct_sampling :
  systematic_sampling 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end NUMINAMATH_GPT_correct_sampling_l2228_222805


namespace NUMINAMATH_GPT_sequence_noncongruent_modulo_l2228_222870

theorem sequence_noncongruent_modulo 
  (a : ℕ → ℕ)
  (h0 : a 1 = 1)
  (h1 : ∀ n, a (n + 1) = a n + 2^(a n)) :
  ∀ (i j : ℕ), i ≠ j → i ≤ 32021 → j ≤ 32021 →
  (a i) % (3^2021) ≠ (a j) % (3^2021) := 
by
  sorry

end NUMINAMATH_GPT_sequence_noncongruent_modulo_l2228_222870


namespace NUMINAMATH_GPT_cards_selection_count_l2228_222882

noncomputable def numberOfWaysToChooseCards : Nat :=
  (Nat.choose 4 3) * 3 * (Nat.choose 13 2) * (13 ^ 2)

theorem cards_selection_count :
  numberOfWaysToChooseCards = 158184 := by
  sorry

end NUMINAMATH_GPT_cards_selection_count_l2228_222882


namespace NUMINAMATH_GPT_borrowed_sheets_l2228_222839

-- Defining the page sum function
def sum_pages (n : ℕ) : ℕ := n * (n + 1)

-- Formulating the main theorem statement
theorem borrowed_sheets (b c : ℕ) (H : c + b ≤ 30) (H_avg : (sum_pages b + sum_pages (30 - b - c) - sum_pages (b + c)) * 2 = 25 * (60 - 2 * c)) :
  c = 10 :=
sorry

end NUMINAMATH_GPT_borrowed_sheets_l2228_222839


namespace NUMINAMATH_GPT_original_number_one_more_reciprocal_is_11_over_5_l2228_222899

theorem original_number_one_more_reciprocal_is_11_over_5 (x : ℚ) (h : 1 + 1/x = 11/5) : x = 5/6 :=
by
  sorry

end NUMINAMATH_GPT_original_number_one_more_reciprocal_is_11_over_5_l2228_222899


namespace NUMINAMATH_GPT_fencing_cost_l2228_222859

noncomputable def pi_approx : ℝ := 3.14159

theorem fencing_cost 
  (d : ℝ) (r : ℝ)
  (h_d : d = 20) 
  (h_r : r = 1.50) :
  abs (r * pi_approx * d - 94.25) < 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_fencing_cost_l2228_222859


namespace NUMINAMATH_GPT_average_marks_l2228_222814

theorem average_marks (num_students : ℕ) (marks1 marks2 marks3 : ℕ) (num1 num2 num3 : ℕ) (h1 : num_students = 50)
  (h2 : marks1 = 90) (h3 : num1 = 10) (h4 : marks2 = marks1 - 10) (h5 : num2 = 15) (h6 : marks3 = 60) 
  (h7 : num1 + num2 + num3 = 50) (h8 : num3 = num_students - (num1 + num2)) (total_marks : ℕ) 
  (h9 : total_marks = (num1 * marks1) + (num2 * marks2) + (num3 * marks3)) : 
  (total_marks / num_students = 72) :=
by
  sorry

end NUMINAMATH_GPT_average_marks_l2228_222814


namespace NUMINAMATH_GPT_maximum_rabbits_l2228_222890

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_maximum_rabbits_l2228_222890


namespace NUMINAMATH_GPT_min_max_expression_l2228_222813

variable (a b c d e : ℝ)

def expression (a b c d e : ℝ) : ℝ :=
  5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)

theorem min_max_expression :
  a + b + c + d + e = 10 →
  a^2 + b^2 + c^2 + d^2 + e^2 = 20 →
  expression a b c d e = 120 := by
  sorry

end NUMINAMATH_GPT_min_max_expression_l2228_222813


namespace NUMINAMATH_GPT_coda_password_combinations_l2228_222879

open BigOperators

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 
  ∨ n = 23 ∨ n = 29

def is_power_of_two (n : ℕ) : Prop :=
  n = 2 ∨ n = 4 ∨ n = 8 ∨ n = 16

def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n ≥ 1 ∧ n ≤ 30

def count_primes : ℕ :=
  10
def count_powers_of_two : ℕ :=
  4
def count_multiples_of_three : ℕ :=
  10

theorem coda_password_combinations : count_primes * count_powers_of_two * count_multiples_of_three = 400 := by
  sorry

end NUMINAMATH_GPT_coda_password_combinations_l2228_222879


namespace NUMINAMATH_GPT_complex_sum_l2228_222863

open Complex

theorem complex_sum (w : ℂ) (h : w^2 - w + 1 = 0) :
  w^103 + w^104 + w^105 + w^106 + w^107 = -1 :=
sorry

end NUMINAMATH_GPT_complex_sum_l2228_222863


namespace NUMINAMATH_GPT_andy_l2228_222809

theorem andy's_profit_per_cake :
  (∀ (cakes : ℕ), cakes = 2 → ∀ (ingredient_cost : ℕ), ingredient_cost = 12 →
                  ∀ (packaging_cost_per_cake : ℕ), packaging_cost_per_cake = 1 →
                  ∀ (selling_price_per_cake : ℕ), selling_price_per_cake = 15 →
                  ∀ (profit_per_cake : ℕ), profit_per_cake = selling_price_per_cake - (ingredient_cost / cakes + packaging_cost_per_cake) →
                    profit_per_cake = 8) :=
by
  sorry

end NUMINAMATH_GPT_andy_l2228_222809


namespace NUMINAMATH_GPT_complementary_angle_problem_l2228_222847

theorem complementary_angle_problem 
  (A B : ℝ) 
  (h1 : A + B = 90) 
  (h2 : A / B = 2 / 3) 
  (increase : A' = A * 1.20) 
  (new_sum : A' + B' = 90) 
  (B' : ℝ)
  (h3 : B' = B - B * 0.1333) :
  true := 
sorry

end NUMINAMATH_GPT_complementary_angle_problem_l2228_222847


namespace NUMINAMATH_GPT_dimensions_multiple_of_three_l2228_222865

theorem dimensions_multiple_of_three (a b c : ℤ) (h : a * b * c = (a + 1) * (b + 1) * (c - 2)) :
  (a % 3 = 0) ∨ (b % 3 = 0) ∨ (c % 3 = 0) :=
sorry

end NUMINAMATH_GPT_dimensions_multiple_of_three_l2228_222865


namespace NUMINAMATH_GPT_frac_not_suff_nec_l2228_222860

theorem frac_not_suff_nec {a b : ℝ} (hab : a / b > 1) : 
  ¬ ((∀ a b : ℝ, a / b > 1 → a > b) ∧ (∀ a b : ℝ, a > b → a / b > 1)) :=
sorry

end NUMINAMATH_GPT_frac_not_suff_nec_l2228_222860


namespace NUMINAMATH_GPT_first_sequence_general_term_second_sequence_general_term_l2228_222810

-- For the first sequence
def first_sequence_sum : ℕ → ℚ
| n => n^2 + 1/2 * n

theorem first_sequence_general_term (n : ℕ) : 
  (first_sequence_sum (n+1) - first_sequence_sum n) = (2 * (n+1) - 1/2) := 
sorry

-- For the second sequence
def second_sequence_sum : ℕ → ℚ
| n => 1/4 * n^2 + 2/3 * n + 3

theorem second_sequence_general_term (n : ℕ) : 
  (second_sequence_sum (n+1) - second_sequence_sum n) = 
  if n = 0 then 47/12 
  else (6 * (n+1) + 5)/12 := 
sorry

end NUMINAMATH_GPT_first_sequence_general_term_second_sequence_general_term_l2228_222810


namespace NUMINAMATH_GPT_breadth_of_plot_l2228_222871

theorem breadth_of_plot (b l : ℝ) (h1 : l * b = 18 * b) (h2 : l - b = 10) : b = 8 :=
by
  sorry

end NUMINAMATH_GPT_breadth_of_plot_l2228_222871


namespace NUMINAMATH_GPT_f_inequality_l2228_222825

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f(x+3) = -1 / f(x)
axiom f_prop1 : ∀ x : ℝ, f (x + 3) = -1 / f x

-- Condition 2: ∀ 3 ≤ x_1 < x_2 ≤ 6, f(x_1) < f(x_2)
axiom f_prop2 : ∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 6 → f x1 < f x2

-- Condition 3: The graph of y = f(x + 3) is symmetric about the y-axis
axiom f_prop3 : ∀ x : ℝ, f (3 - x) = f (3 + x)

-- Theorem: f(3) < f(4.5) < f(7)
theorem f_inequality : f 3 < f 4.5 ∧ f 4.5 < f 7 := by
  sorry

end NUMINAMATH_GPT_f_inequality_l2228_222825


namespace NUMINAMATH_GPT_valid_outfits_number_l2228_222840

def num_shirts := 7
def num_pants := 7
def num_hats := 7
def num_colors := 7

def total_outfits (num_shirts num_pants num_hats : ℕ) := num_shirts * num_pants * num_hats
def matching_color_outfits (num_colors : ℕ) := num_colors
def valid_outfits (num_shirts num_pants num_hats num_colors : ℕ) := 
  total_outfits num_shirts num_pants num_hats - matching_color_outfits num_colors

theorem valid_outfits_number : valid_outfits num_shirts num_pants num_hats num_colors = 336 := 
by
  sorry

end NUMINAMATH_GPT_valid_outfits_number_l2228_222840


namespace NUMINAMATH_GPT_Jesse_read_pages_l2228_222832

theorem Jesse_read_pages (total_pages : ℝ) (h : (2 / 3) * total_pages = 166) :
  (1 / 3) * total_pages = 83 :=
sorry

end NUMINAMATH_GPT_Jesse_read_pages_l2228_222832


namespace NUMINAMATH_GPT_additional_machines_l2228_222864

theorem additional_machines (r : ℝ) (M : ℝ) : 
  (5 * r * 20 = 1) ∧ (M * r * 10 = 1) → (M - 5 = 95) :=
by
  sorry

end NUMINAMATH_GPT_additional_machines_l2228_222864


namespace NUMINAMATH_GPT_percent_increase_correct_l2228_222828

-- Define the original and new visual ranges
def original_range : Float := 90
def new_range : Float := 150

-- Define the calculation for percent increase
def percent_increase : Float :=
  ((new_range - original_range) / original_range) * 100

-- Statement to prove
theorem percent_increase_correct : percent_increase = 66.67 :=
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_percent_increase_correct_l2228_222828


namespace NUMINAMATH_GPT_moles_of_HCl_is_one_l2228_222833

def moles_of_HCl_combined 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : ℝ := 
by 
  sorry

theorem moles_of_HCl_is_one 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : moles_of_HCl_combined moles_NaHSO3 moles_H2O_formed reaction_completes one_mole_NaHSO3_used = 1 := 
by 
  sorry

end NUMINAMATH_GPT_moles_of_HCl_is_one_l2228_222833


namespace NUMINAMATH_GPT_unique_increasing_seq_l2228_222869

noncomputable def unique_seq (a : ℕ → ℕ) (r : ℝ) : Prop :=
∀ (b : ℕ → ℕ), (∀ n, b n = 3 * n - 2 → ∑' n, r ^ (b n) = 1 / 2 ) → (∀ n, a n = b n)

theorem unique_increasing_seq {r : ℝ} 
  (hr : 0.4 < r ∧ r < 0.5) 
  (hc : r^3 + 2*r = 1):
  ∃ a : ℕ → ℕ, (∀ n, a n = 3 * n - 2) ∧ (∑'(n), r^(a n) = 1/2) ∧ unique_seq a r :=
by
  sorry

end NUMINAMATH_GPT_unique_increasing_seq_l2228_222869


namespace NUMINAMATH_GPT_triangle_ratio_l2228_222829

noncomputable def triangle_problem (BC AC : ℝ) (angleC : ℝ) : ℝ :=
  let CD := AC / 2
  let BD := BC - CD
  let HD := BD / 2
  let AD := (3^(1/2)) * CD
  let AH := AD - HD
  (AH / HD)

theorem triangle_ratio (BC AC : ℝ) (angleC : ℝ) (h1 : BC = 6) (h2 : AC = 3 * Real.sqrt 3) (h3 : angleC = Real.pi / 6) :
  triangle_problem BC AC angleC = -2 - Real.sqrt 3 :=
by
  sorry  

end NUMINAMATH_GPT_triangle_ratio_l2228_222829


namespace NUMINAMATH_GPT_washington_goats_l2228_222815

variables (W : ℕ) (P : ℕ) (total_goats : ℕ)

theorem washington_goats (W : ℕ) (h1 : P = W + 40) (h2 : total_goats = W + P) (h3 : total_goats = 320) : W = 140 :=
by
  sorry

end NUMINAMATH_GPT_washington_goats_l2228_222815


namespace NUMINAMATH_GPT_man_speed_against_current_eq_l2228_222891

-- Definitions
def downstream_speed : ℝ := 22 -- Man's speed with the current in km/hr
def current_speed : ℝ := 5 -- Speed of the current in km/hr

-- Man's speed in still water
def man_speed_in_still_water : ℝ := downstream_speed - current_speed

-- Man's speed against the current
def speed_against_current : ℝ := man_speed_in_still_water - current_speed

-- Theorem: The man's speed against the current is 12 km/hr.
theorem man_speed_against_current_eq : speed_against_current = 12 := by
  sorry

end NUMINAMATH_GPT_man_speed_against_current_eq_l2228_222891


namespace NUMINAMATH_GPT_vacation_cost_division_l2228_222846

theorem vacation_cost_division (n : ℕ) (h1 : 360 = 4 * (120 - 30)) (h2 : 360 = n * 120) : n = 3 := 
sorry

end NUMINAMATH_GPT_vacation_cost_division_l2228_222846


namespace NUMINAMATH_GPT_cost_of_each_pair_of_jeans_l2228_222881

-- Conditions
def costWallet : ℕ := 50
def costSneakers : ℕ := 100
def pairsSneakers : ℕ := 2
def costBackpack : ℕ := 100
def totalSpent : ℕ := 450
def pairsJeans : ℕ := 2

-- Definitions
def totalSpentLeonard := costWallet + pairsSneakers * costSneakers
def totalSpentMichaelWithoutJeans := costBackpack

-- Goal: Prove the cost of each pair of jeans
theorem cost_of_each_pair_of_jeans :
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  costPerPairJeans = 50 :=
by
  intros
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  show costPerPairJeans = 50
  sorry

end NUMINAMATH_GPT_cost_of_each_pair_of_jeans_l2228_222881


namespace NUMINAMATH_GPT_value_of_expression_l2228_222873

theorem value_of_expression (a b : ℤ) (h : a - b = 1) : 3 * a - 3 * b - 4 = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_expression_l2228_222873


namespace NUMINAMATH_GPT_find_v_l2228_222877

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1;
    3, 0]

noncomputable def v : Matrix (Fin 2) (Fin 1) ℝ :=
  !![0;
    1 / 30.333]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_v : 
  (A ^ 10 + A ^ 8 + A ^ 6 + A ^ 4 + A ^ 2 + I) * v = !![0; 12] :=
  sorry

end NUMINAMATH_GPT_find_v_l2228_222877


namespace NUMINAMATH_GPT_greatest_sundays_in_49_days_l2228_222806

theorem greatest_sundays_in_49_days : 
  ∀ (days : ℕ), 
    days = 49 → 
    ∀ (sundays_per_week : ℕ), 
      sundays_per_week = 1 → 
      ∀ (weeks : ℕ), 
        weeks = days / 7 → 
        weeks * sundays_per_week = 7 :=
by
  sorry

end NUMINAMATH_GPT_greatest_sundays_in_49_days_l2228_222806


namespace NUMINAMATH_GPT_pencil_probability_l2228_222875

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem pencil_probability : 
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 14 :=
by
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  have h : probability = 5 / 14 := sorry
  exact h

end NUMINAMATH_GPT_pencil_probability_l2228_222875


namespace NUMINAMATH_GPT_percentage_of_x_is_40_l2228_222820

theorem percentage_of_x_is_40 
  (x p : ℝ)
  (h1 : (1 / 2) * x = 200)
  (h2 : p * x = 160) : 
  p * 100 = 40 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_of_x_is_40_l2228_222820


namespace NUMINAMATH_GPT_find_k_l2228_222803

open Real

noncomputable def chord_intersection (k : ℝ) : Prop :=
  let R : ℝ := 3
  let d := abs (k + 1) / sqrt (1 + k^2)
  d^2 + (12 * sqrt 5 / 10)^2 = R^2

theorem find_k (k : ℝ) (h : k > 1) (h_intersect : chord_intersection k) : k = 2 := by
  sorry

end NUMINAMATH_GPT_find_k_l2228_222803


namespace NUMINAMATH_GPT_height_of_larger_box_l2228_222857

/-- Define the dimensions of the larger box and smaller boxes, 
    and show that given the constraints, the height of the larger box must be 4 meters.-/
theorem height_of_larger_box 
  (L H : ℝ) (V_small : ℝ) (N_small : ℕ) (h : ℝ) 
  (dim_large : L = 6) (width_large : H = 5)
  (vol_small : V_small = 0.6 * 0.5 * 0.4) 
  (num_boxes : N_small = 1000) 
  (vol_large : 6 * 5 * h = N_small * V_small) : 
  h = 4 :=
by 
  sorry

end NUMINAMATH_GPT_height_of_larger_box_l2228_222857


namespace NUMINAMATH_GPT_area_of_shaded_region_l2228_222801

def side_length_of_square : ℝ := 12
def radius_of_quarter_circle : ℝ := 6

theorem area_of_shaded_region :
  let area_square := side_length_of_square ^ 2
  let area_full_circle := π * radius_of_quarter_circle ^ 2
  (area_square - area_full_circle) = 144 - 36 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l2228_222801


namespace NUMINAMATH_GPT_johns_pants_cost_50_l2228_222826

variable (P : ℝ)

theorem johns_pants_cost_50 (h1 : P + 1.60 * P = 130) : P = 50 := 
by
  sorry

end NUMINAMATH_GPT_johns_pants_cost_50_l2228_222826


namespace NUMINAMATH_GPT_group_size_systematic_sampling_l2228_222880

-- Define the total number of viewers
def total_viewers : ℕ := 10000

-- Define the number of viewers to be selected
def selected_viewers : ℕ := 10

-- Lean statement to prove the group size for systematic sampling
theorem group_size_systematic_sampling (n_total n_selected : ℕ) : n_total = total_viewers → n_selected = selected_viewers → (n_total / n_selected) = 1000 :=
by
  intros h_total h_selected
  rw [h_total, h_selected]
  sorry

end NUMINAMATH_GPT_group_size_systematic_sampling_l2228_222880


namespace NUMINAMATH_GPT_arithmetic_question_l2228_222850

theorem arithmetic_question :
  ((3.25 - 1.57) * 2) = 3.36 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_question_l2228_222850


namespace NUMINAMATH_GPT_sum_digits_of_consecutive_numbers_l2228_222886

-- Define the sum of digits function
def sum_digits (n : ℕ) : ℕ := sorry -- Placeholder, define the sum of digits function

-- Given conditions
variables (N : ℕ)
axiom h1 : sum_digits N + sum_digits (N + 1) = 200
axiom h2 : sum_digits (N + 2) + sum_digits (N + 3) = 105

-- Theorem statement to be proved
theorem sum_digits_of_consecutive_numbers : 
  sum_digits (N + 1) + sum_digits (N + 2) = 103 := 
sorry  -- Proof to be provided

end NUMINAMATH_GPT_sum_digits_of_consecutive_numbers_l2228_222886


namespace NUMINAMATH_GPT_cos_beta_value_l2228_222892

noncomputable def cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) : Real :=
  Real.cos β

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) :
  Real.cos β = 56 / 65 :=
by
  sorry

end NUMINAMATH_GPT_cos_beta_value_l2228_222892


namespace NUMINAMATH_GPT_find_b_l2228_222807

theorem find_b (a b : ℝ) (x : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^5) * x^5):
  b = 40 :=
  sorry

end NUMINAMATH_GPT_find_b_l2228_222807


namespace NUMINAMATH_GPT_max_marks_equals_l2228_222862

/-
  Pradeep has to obtain 45% of the total marks to pass.
  He got 250 marks and failed by 50 marks.
  Prove that the maximum marks is 667.
-/

-- Define the passing percentage
def passing_percentage : ℝ := 0.45

-- Define Pradeep's marks and the marks he failed by
def pradeep_marks : ℝ := 250
def failed_by : ℝ := 50

-- Passing marks is the sum of Pradeep's marks and the marks he failed by
def passing_marks : ℝ := pradeep_marks + failed_by

-- Prove that the maximum marks M is 667
theorem max_marks_equals : ∃ M : ℝ, passing_percentage * M = passing_marks ∧ M = 667 :=
sorry

end NUMINAMATH_GPT_max_marks_equals_l2228_222862


namespace NUMINAMATH_GPT_total_songs_in_june_l2228_222802

-- Define the conditions
def Vivian_daily_songs : ℕ := 10
def Clara_daily_songs : ℕ := Vivian_daily_songs - 2
def Lucas_daily_songs : ℕ := Vivian_daily_songs + 5
def total_play_days_in_june : ℕ := 30 - 8 - 1

-- Total songs listened to in June
def total_songs_Vivian : ℕ := Vivian_daily_songs * total_play_days_in_june
def total_songs_Clara : ℕ := Clara_daily_songs * total_play_days_in_june
def total_songs_Lucas : ℕ := Lucas_daily_songs * total_play_days_in_june

-- The total number of songs listened to by all three
def total_songs_all_three : ℕ := total_songs_Vivian + total_songs_Clara + total_songs_Lucas

-- The proof problem
theorem total_songs_in_june : total_songs_all_three = 693 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_songs_in_june_l2228_222802


namespace NUMINAMATH_GPT_Deepak_age_l2228_222841

theorem Deepak_age : ∃ (A D : ℕ), (A / D = 4 / 3) ∧ (A + 6 = 26) ∧ (D = 15) :=
by
  sorry

end NUMINAMATH_GPT_Deepak_age_l2228_222841


namespace NUMINAMATH_GPT_opposite_of_two_is_negative_two_l2228_222888

theorem opposite_of_two_is_negative_two : -2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_two_is_negative_two_l2228_222888


namespace NUMINAMATH_GPT_power_24_eq_one_l2228_222853

theorem power_24_eq_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^24 = 1 :=
by
  sorry

end NUMINAMATH_GPT_power_24_eq_one_l2228_222853


namespace NUMINAMATH_GPT_infinite_primes_of_form_l2228_222898

theorem infinite_primes_of_form (p : ℕ) (hp : Nat.Prime p) (hpodd : p % 2 = 1) :
  ∃ᶠ n in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end NUMINAMATH_GPT_infinite_primes_of_form_l2228_222898


namespace NUMINAMATH_GPT_initial_persons_count_l2228_222830

theorem initial_persons_count (P : ℕ) (H1 : 18 * P = 1) (H2 : 6 * P = 1/3) (H3 : 9 * (P + 4) = 2/3) : P = 12 :=
by
  sorry

end NUMINAMATH_GPT_initial_persons_count_l2228_222830


namespace NUMINAMATH_GPT_a6_is_32_l2228_222824

namespace arithmetic_sequence

variables {a : ℕ → ℝ} -- {aₙ} is an arithmetic sequence with positive terms
variables (q : ℝ) -- Common ratio

-- Conditions as definitions
def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a1_is_1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2_times_a4_is_16 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 4 = 16

-- The ultimate goal is to prove a₆ = 32
theorem a6_is_32 (h_arith : is_arithmetic_sequence a q) 
  (h_a1 : a1_is_1 a) (h_product : a2_times_a4_is_16 a q) : 
  a 6 = 32 := 
sorry

end arithmetic_sequence

end NUMINAMATH_GPT_a6_is_32_l2228_222824


namespace NUMINAMATH_GPT_stretching_transformation_eq_curve_l2228_222835

variable (x y x₁ y₁ : ℝ)

theorem stretching_transformation_eq_curve :
  (x₁ = 3 * x) →
  (y₁ = y) →
  (x₁^2 + 9 * y₁^2 = 9) →
  (x^2 + y^2 = 1) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_stretching_transformation_eq_curve_l2228_222835
