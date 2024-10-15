import Mathlib

namespace NUMINAMATH_GPT_mark_ate_in_first_four_days_l1090_109037

-- Definitions based on conditions
def total_fruit : ℕ := 10
def fruit_kept : ℕ := 2
def fruit_brought_on_friday : ℕ := 3

-- Statement to be proved
theorem mark_ate_in_first_four_days : total_fruit - fruit_kept - fruit_brought_on_friday = 5 := 
by sorry

end NUMINAMATH_GPT_mark_ate_in_first_four_days_l1090_109037


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_l1090_109073

def P : Set ℤ := {-3, -2, 0, 2}
def Q : Set ℤ := {-1, -2, -3, 0, 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-3, -2, 0} := by
  sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_l1090_109073


namespace NUMINAMATH_GPT_cost_of_grapes_and_watermelon_l1090_109008

theorem cost_of_grapes_and_watermelon (p g w f : ℝ)
  (h1 : p + g + w + f = 30)
  (h2 : f = 2 * p)
  (h3 : p - g = w) :
  g + w = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_grapes_and_watermelon_l1090_109008


namespace NUMINAMATH_GPT_victor_earnings_l1090_109072

variable (wage hours_mon hours_tue : ℕ)

def hourly_wage : ℕ := 6
def hours_worked_monday : ℕ := 5
def hours_worked_tuesday : ℕ := 5

theorem victor_earnings :
  (hours_worked_monday + hours_worked_tuesday) * hourly_wage = 60 :=
by
  sorry

end NUMINAMATH_GPT_victor_earnings_l1090_109072


namespace NUMINAMATH_GPT_overall_avg_is_60_l1090_109027

-- Define the number of students and average marks for each class
def classA_students : ℕ := 30
def classA_avg_marks : ℕ := 40

def classB_students : ℕ := 50
def classB_avg_marks : ℕ := 70

def classC_students : ℕ := 25
def classC_avg_marks : ℕ := 55

def classD_students : ℕ := 45
def classD_avg_marks : ℕ := 65

-- Calculate the total number of students
def total_students : ℕ := 
  classA_students + classB_students + classC_students + classD_students

-- Calculate the total marks for each class
def total_marks_A : ℕ := classA_students * classA_avg_marks
def total_marks_B : ℕ := classB_students * classB_avg_marks
def total_marks_C : ℕ := classC_students * classC_avg_marks
def total_marks_D : ℕ := classD_students * classD_avg_marks

-- Calculate the combined total marks of all classes
def combined_total_marks : ℕ := 
  total_marks_A + total_marks_B + total_marks_C + total_marks_D

-- Calculate the overall average marks
def overall_avg_marks : ℕ := combined_total_marks / total_students

-- Prove that the overall average marks is 60
theorem overall_avg_is_60 : overall_avg_marks = 60 := by
  sorry -- Proof will be written here

end NUMINAMATH_GPT_overall_avg_is_60_l1090_109027


namespace NUMINAMATH_GPT_total_spent_is_13_l1090_109047

-- Let cost_cb represent the cost of the candy bar
def cost_cb : ℕ := 7

-- Let cost_ch represent the cost of the chocolate
def cost_ch : ℕ := 6

-- Define the total cost as the sum of cost_cb and cost_ch
def total_cost : ℕ := cost_cb + cost_ch

-- Theorem to prove the total cost equals $13
theorem total_spent_is_13 : total_cost = 13 := by
  sorry

end NUMINAMATH_GPT_total_spent_is_13_l1090_109047


namespace NUMINAMATH_GPT_greg_rolls_probability_l1090_109060

noncomputable def probability_of_more_ones_than_twos_and_threes_combined : ℚ :=
  (3046.5 : ℚ) / 7776

theorem greg_rolls_probability :
  probability_of_more_ones_than_twos_and_threes_combined = (3046.5 : ℚ) / 7776 := 
by 
  sorry

end NUMINAMATH_GPT_greg_rolls_probability_l1090_109060


namespace NUMINAMATH_GPT_number_of_boys_l1090_109075

variable {total_marbles : ℕ} (marbles_per_boy : ℕ := 10)
variable (H_total_marbles : total_marbles = 20)

theorem number_of_boys (total_marbles_marbs_eq_20 : total_marbles = 20) (marbles_per_boy_eq_10 : marbles_per_boy = 10) :
  total_marbles / marbles_per_boy = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_boys_l1090_109075


namespace NUMINAMATH_GPT_value_of_y_l1090_109026

theorem value_of_y (x y : ℝ) (cond1 : 1.5 * x = 0.75 * y) (cond2 : x = 20) : y = 40 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l1090_109026


namespace NUMINAMATH_GPT_minimum_value_expression_l1090_109065

theorem minimum_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  (x^2 + 6 * x * y + 9 * y^2 + 3/2 * z^2) ≥ 102 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l1090_109065


namespace NUMINAMATH_GPT_binom_comb_always_integer_l1090_109006

theorem binom_comb_always_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : (k + 2) ∣ n) : 
  ∃ m : ℤ, ((n - 3 * k - 2) / (k + 2)) * Nat.choose n k = m := 
sorry

end NUMINAMATH_GPT_binom_comb_always_integer_l1090_109006


namespace NUMINAMATH_GPT_ratio_p_q_l1090_109033

-- Definitions of probabilities p and q based on combinatorial choices and probabilities described.
noncomputable def p : ℚ :=
  (Nat.choose 6 1) * (Nat.choose 5 2) * (Nat.choose 24 2) * (Nat.choose 22 4) * (Nat.choose 18 4) * (Nat.choose 14 5) * (Nat.choose 9 5) * (Nat.choose 4 5) / (6 ^ 24)

noncomputable def q : ℚ :=
  (Nat.choose 6 2) * (Nat.choose 24 3) * (Nat.choose 21 3) * (Nat.choose 18 4) * (Nat.choose 14 4) * (Nat.choose 10 4) * (Nat.choose 6 4) / (6 ^ 24)

-- Lean statement to prove p / q = 6
theorem ratio_p_q : p / q = 6 := by
  sorry

end NUMINAMATH_GPT_ratio_p_q_l1090_109033


namespace NUMINAMATH_GPT_intersection_complement_l1090_109098

universe u
variable {α : Type u}

-- Define the sets I, M, N, and their complement with respect to I
def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}
def complement_I (s : Set ℕ) : Set ℕ := { x ∈ I | x ∉ s }

-- Statement of the theorem
theorem intersection_complement :
  M ∩ (complement_I N) = {1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1090_109098


namespace NUMINAMATH_GPT_emma_finishes_first_l1090_109042

noncomputable def david_lawn_area : ℝ := sorry
noncomputable def emma_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 3
noncomputable def fiona_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 4

noncomputable def david_mowing_rate : ℝ := sorry
noncomputable def fiona_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 6
noncomputable def emma_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 2

theorem emma_finishes_first (z w : ℝ) (hz : z > 0) (hw : w > 0) :
  (z / w) > (2 * z / (3 * w)) ∧ (3 * z / (2 * w)) > (2 * z / (3 * w)) :=
by
  sorry

end NUMINAMATH_GPT_emma_finishes_first_l1090_109042


namespace NUMINAMATH_GPT_range_of_a_odd_not_even_l1090_109096

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

def A : Set ℝ := Set.Ioo (-1 : ℝ) 1

def B (a : ℝ) : Set ℝ := Set.Ioo a (a + 1)

theorem range_of_a (a : ℝ) (h1 : B a ⊆ A) : -1 ≤ a ∧ a ≤ 0 := by
  sorry

theorem odd_not_even : (∀ x ∈ A, f (-x) = - f x) ∧ ¬ (∀ x ∈ A, f x = f (-x)) := by
  sorry

end NUMINAMATH_GPT_range_of_a_odd_not_even_l1090_109096


namespace NUMINAMATH_GPT_sandwich_count_l1090_109025

-- Define the given conditions
def meats : ℕ := 8
def cheeses : ℕ := 12
def cheese_combination_count : ℕ := Nat.choose cheeses 3

-- Define the total sandwich count based on the conditions
def total_sandwiches : ℕ := meats * cheese_combination_count

-- The theorem we want to prove
theorem sandwich_count : total_sandwiches = 1760 := by
  -- Mathematical steps here are omitted
  sorry

end NUMINAMATH_GPT_sandwich_count_l1090_109025


namespace NUMINAMATH_GPT_joan_remaining_kittens_l1090_109012

-- Definitions based on the given conditions
def original_kittens : Nat := 8
def kittens_given_away : Nat := 2

-- Statement to prove
theorem joan_remaining_kittens : original_kittens - kittens_given_away = 6 := 
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_joan_remaining_kittens_l1090_109012


namespace NUMINAMATH_GPT_binary_to_decimal_1100_l1090_109084

theorem binary_to_decimal_1100 : 
  (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0) = 12 := 
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_1100_l1090_109084


namespace NUMINAMATH_GPT_average_weight_all_children_l1090_109088

theorem average_weight_all_children (avg_boys_weight avg_girls_weight : ℝ) (num_boys num_girls : ℕ)
    (hb : avg_boys_weight = 155) (nb : num_boys = 8)
    (hg : avg_girls_weight = 125) (ng : num_girls = 7) :
    (num_boys + num_girls = 15) → (avg_boys_weight * num_boys + avg_girls_weight * num_girls) / (num_boys + num_girls) = 141 := by
  intro h_sum
  sorry

end NUMINAMATH_GPT_average_weight_all_children_l1090_109088


namespace NUMINAMATH_GPT_marla_errand_total_time_l1090_109078

theorem marla_errand_total_time :
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  total_time = 110 :=
by
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  show total_time = 110
  sorry

end NUMINAMATH_GPT_marla_errand_total_time_l1090_109078


namespace NUMINAMATH_GPT_circles_are_disjoint_l1090_109019

noncomputable def positional_relationship_of_circles (R₁ R₂ d : ℝ) (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : Prop :=
R₁ + R₂ = d

theorem circles_are_disjoint {R₁ R₂ d : ℝ} (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : positional_relationship_of_circles R₁ R₂ d h₁ h₂ :=
by sorry

end NUMINAMATH_GPT_circles_are_disjoint_l1090_109019


namespace NUMINAMATH_GPT_angle_B_side_b_l1090_109054

variable (A B C a b c : ℝ)
variable (S : ℝ := 5 * Real.sqrt 3)

-- Conditions
variable (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
variable (h2 : 1/2 * a * c * Real.sin B = S)
variable (h3 : a = 5)

-- The two parts to prove
theorem angle_B (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B) : 
  B = π / 3 := 
  sorry

theorem side_b (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
  (h2 : 1/2 * a * c * Real.sin B = S) (h3 : a = 5) : 
  b = Real.sqrt 21 := 
  sorry

end NUMINAMATH_GPT_angle_B_side_b_l1090_109054


namespace NUMINAMATH_GPT_m_over_n_add_one_l1090_109031

theorem m_over_n_add_one (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : (m + n : ℚ) / n = 10 / 7 :=
by
  sorry

end NUMINAMATH_GPT_m_over_n_add_one_l1090_109031


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l1090_109082

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l1090_109082


namespace NUMINAMATH_GPT_intercepts_equal_l1090_109052

theorem intercepts_equal (a : ℝ) :
  (∃ x y : ℝ, ax + y - 2 - a = 0 ∧
              y = 0 ∧ x = (a + 2) / a ∧
              x = 0 ∧ y = 2 + a) →
  (a = 1 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_intercepts_equal_l1090_109052


namespace NUMINAMATH_GPT_packs_of_blue_tshirts_l1090_109018

theorem packs_of_blue_tshirts (total_tshirts white_packs white_per_pack blue_per_pack : ℕ) 
  (h_white_packs : white_packs = 3) 
  (h_white_per_pack : white_per_pack = 6) 
  (h_blue_per_pack : blue_per_pack = 4) 
  (h_total_tshirts : total_tshirts = 26) : 
  (total_tshirts - white_packs * white_per_pack) / blue_per_pack = 2 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_packs_of_blue_tshirts_l1090_109018


namespace NUMINAMATH_GPT_peaches_per_basket_l1090_109043

-- Given conditions as definitions in Lean 4
def red_peaches : Nat := 7
def green_peaches : Nat := 3

-- The proof statement showing each basket contains 10 peaches in total.
theorem peaches_per_basket : red_peaches + green_peaches = 10 := by
  sorry

end NUMINAMATH_GPT_peaches_per_basket_l1090_109043


namespace NUMINAMATH_GPT_need_to_sell_more_rolls_l1090_109040

variable (goal sold_grandmother sold_uncle_1 sold_uncle_additional sold_neighbor_1 returned_neighbor sold_mothers_friend sold_cousin_1 sold_cousin_additional : ℕ)

theorem need_to_sell_more_rolls
  (h_goal : goal = 100)
  (h_sold_grandmother : sold_grandmother = 5)
  (h_sold_uncle_1 : sold_uncle_1 = 12)
  (h_sold_uncle_additional : sold_uncle_additional = 10)
  (h_sold_neighbor_1 : sold_neighbor_1 = 8)
  (h_returned_neighbor : returned_neighbor = 4)
  (h_sold_mothers_friend : sold_mothers_friend = 25)
  (h_sold_cousin_1 : sold_cousin_1 = 3)
  (h_sold_cousin_additional : sold_cousin_additional = 5) :
  goal - (sold_grandmother + (sold_uncle_1 + sold_uncle_additional) + (sold_neighbor_1 - returned_neighbor) + sold_mothers_friend + (sold_cousin_1 + sold_cousin_additional)) = 36 := by
  sorry

end NUMINAMATH_GPT_need_to_sell_more_rolls_l1090_109040


namespace NUMINAMATH_GPT_length_of_common_internal_tangent_l1090_109066

-- Define the conditions
def circles_centers_distance : ℝ := 50
def radius_smaller_circle : ℝ := 7
def radius_larger_circle : ℝ := 10

-- Define the statement to be proven
theorem length_of_common_internal_tangent :
  let d := circles_centers_distance
  let r₁ := radius_smaller_circle
  let r₂ := radius_larger_circle
  ∃ (length_tangent : ℝ), length_tangent = Real.sqrt (d^2 - (r₁ + r₂)^2) := by
  -- Provide the correct answer based on the conditions
  sorry

end NUMINAMATH_GPT_length_of_common_internal_tangent_l1090_109066


namespace NUMINAMATH_GPT_fruit_seller_apples_l1090_109032

theorem fruit_seller_apples (x : ℝ) (h : 0.60 * x = 420) : x = 700 :=
sorry

end NUMINAMATH_GPT_fruit_seller_apples_l1090_109032


namespace NUMINAMATH_GPT_smallest_possible_value_l1090_109099

theorem smallest_possible_value (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) : 
  ∃ (m : ℝ), m = -1/12 ∧ (∀ x y : ℝ, (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → (x + y) / (x^2) ≥ m) :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_l1090_109099


namespace NUMINAMATH_GPT_ernie_circles_l1090_109077

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes circles_ali : ℕ) 
  (h1 : boxes_per_circle_ali = 8)
  (h2 : boxes_per_circle_ernie = 10)
  (h3 : total_boxes = 80)
  (h4 : circles_ali = 5) : 
  (total_boxes - circles_ali * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end NUMINAMATH_GPT_ernie_circles_l1090_109077


namespace NUMINAMATH_GPT_total_number_of_cows_l1090_109093

variable (D C : ℕ) -- D is the number of ducks and C is the number of cows

-- Define the condition given in the problem
def legs_eq : Prop := 2 * D + 4 * C = 2 * (D + C) + 28

theorem total_number_of_cows (h : legs_eq D C) : C = 14 := by
  sorry

end NUMINAMATH_GPT_total_number_of_cows_l1090_109093


namespace NUMINAMATH_GPT_students_taking_french_l1090_109015

theorem students_taking_french 
  (Total : ℕ) (G : ℕ) (B : ℕ) (Neither : ℕ) (H_total : Total = 87)
  (H_G : G = 22) (H_B : B = 9) (H_neither : Neither = 33) : 
  ∃ F : ℕ, F = 41 := 
by
  sorry

end NUMINAMATH_GPT_students_taking_french_l1090_109015


namespace NUMINAMATH_GPT_find_a_plus_c_l1090_109000

def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_a_plus_c (a b c d : ℝ) 
  (h_vertex_f : -a / 2 = v) (h_vertex_g : -c / 2 = w)
  (h_root_v_g : g v c d = 0) (h_root_w_f : f w a b = 0)
  (h_intersect : f 50 a b = -200 ∧ g 50 c d = -200)
  (h_min_value_f : ∀ x, f (-a / 2) a b ≤ f x a b)
  (h_min_value_g : ∀ x, g (-c / 2) c d ≤ g x c d)
  (h_min_difference : f (-a / 2) a b = g (-c / 2) c d - 50) :
  a + c = sorry :=
sorry

end NUMINAMATH_GPT_find_a_plus_c_l1090_109000


namespace NUMINAMATH_GPT_hannahs_brothers_l1090_109013

theorem hannahs_brothers (B : ℕ) (h1 : ∀ (b : ℕ), b = 8) (h2 : 48 = 2 * (8 * B)) : B = 3 :=
by
  sorry

end NUMINAMATH_GPT_hannahs_brothers_l1090_109013


namespace NUMINAMATH_GPT_find_r_l1090_109030

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = Real.log 9 / Real.log 3 := by
  sorry

end NUMINAMATH_GPT_find_r_l1090_109030


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_l1090_109035

theorem arithmetic_sequence_a4 (S n : ℕ) (a : ℕ → ℕ) (h1 : S = 28) (h2 : S = 7 * a 4) : a 4 = 4 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_l1090_109035


namespace NUMINAMATH_GPT_truncated_pyramid_ratio_l1090_109055

noncomputable def volume_prism (L1 H : ℝ) : ℝ := L1^2 * H
noncomputable def volume_truncated_pyramid (L1 L2 H : ℝ) : ℝ := 
  (H / 3) * (L1^2 + L1 * L2 + L2^2)

theorem truncated_pyramid_ratio (L1 L2 H : ℝ) 
  (h_vol : volume_truncated_pyramid L1 L2 H = (2/3) * volume_prism L1 H) :
  L1 / L2 = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_truncated_pyramid_ratio_l1090_109055


namespace NUMINAMATH_GPT_positive_difference_of_squares_l1090_109080

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_squares_l1090_109080


namespace NUMINAMATH_GPT_xiaoguang_advances_l1090_109005

theorem xiaoguang_advances (x1 x2 x3 x4 : ℝ) (h1 : 96 ≤ (x1 + x2 + x3 + x4) / 4) (hx1 : x1 = 95) (hx2 : x2 = 97) (hx3 : x3 = 94) : 
  98 ≤ x4 := 
by 
  sorry

end NUMINAMATH_GPT_xiaoguang_advances_l1090_109005


namespace NUMINAMATH_GPT_cos_7theta_l1090_109086

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (7 * θ) = 49 / 2187 := 
  sorry

end NUMINAMATH_GPT_cos_7theta_l1090_109086


namespace NUMINAMATH_GPT_fourth_number_in_first_set_88_l1090_109016

theorem fourth_number_in_first_set_88 (x y : ℝ)
  (h1 : (28 + x + 70 + y + 104) / 5 = 67)
  (h2 : (50 + 62 + 97 + 124 + x) / 5 = 75.6) :
  y = 88 :=
by
  sorry

end NUMINAMATH_GPT_fourth_number_in_first_set_88_l1090_109016


namespace NUMINAMATH_GPT_oak_grove_total_books_l1090_109007

theorem oak_grove_total_books (public_library_books : ℕ) (school_library_books : ℕ)
  (h1 : public_library_books = 1986) (h2 : school_library_books = 5106) :
  public_library_books + school_library_books = 7092 := by
  sorry

end NUMINAMATH_GPT_oak_grove_total_books_l1090_109007


namespace NUMINAMATH_GPT_lcm_of_28_and_24_is_168_l1090_109028

/-- Racing car A completes the track in 28 seconds.
    Racing car B completes the track in 24 seconds.
    Both cars start at the same time.
    We want to prove that the time after which both cars will be side by side again
    (least common multiple of their lap times) is 168 seconds. -/
theorem lcm_of_28_and_24_is_168 :
  Nat.lcm 28 24 = 168 :=
sorry

end NUMINAMATH_GPT_lcm_of_28_and_24_is_168_l1090_109028


namespace NUMINAMATH_GPT_remainder_of_product_div_10_l1090_109034

theorem remainder_of_product_div_10 : 
  (3251 * 7462 * 93419) % 10 = 8 := 
sorry

end NUMINAMATH_GPT_remainder_of_product_div_10_l1090_109034


namespace NUMINAMATH_GPT_sum_of_numbers_l1090_109045

theorem sum_of_numbers : (4.75 + 0.303 + 0.432) = 5.485 := 
by  
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1090_109045


namespace NUMINAMATH_GPT_find_other_two_sides_of_isosceles_right_triangle_l1090_109039

noncomputable def is_isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  ((AB.1 ^ 2 + AB.2 ^ 2 = AC.1 ^ 2 + AC.2 ^ 2 ∧ BC.1 ^ 2 + BC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AB.1 ^ 2 + AB.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AC.1 ^ 2 + AC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AC.1 ^ 2 + AC.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AB.1 ^ 2 + AB.2 ^ 2 = 2 * (AC.1 ^ 2 + AC.2 ^ 2)))

theorem find_other_two_sides_of_isosceles_right_triangle (A B C : ℝ × ℝ)
  (h : is_isosceles_right_triangle A B C)
  (line_AB : 2 * A.1 - A.2 = 0)
  (midpoint_hypotenuse : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 2) :
  (A.1 + 2 * A.2 = 2 ∨ A.1 + 2 * A.2 = 14) ∧ 
  ((A.2 = 2 * A.1) ∨ (A.1 = 4)) :=
sorry

end NUMINAMATH_GPT_find_other_two_sides_of_isosceles_right_triangle_l1090_109039


namespace NUMINAMATH_GPT_sqrt_2x_plus_y_eq_4_l1090_109092

theorem sqrt_2x_plus_y_eq_4 (x y : ℝ) 
  (h1 : (3 * x + 1) = 4) 
  (h2 : (2 * y - 1) = 27) : 
  Real.sqrt (2 * x + y) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_2x_plus_y_eq_4_l1090_109092


namespace NUMINAMATH_GPT_trivia_team_points_l1090_109014

theorem trivia_team_points (total_members absent_members total_points : ℕ) 
    (h1 : total_members = 5) 
    (h2 : absent_members = 2) 
    (h3 : total_points = 18) 
    (h4 : total_members - absent_members = present_members) 
    (h5 : total_points = present_members * points_per_member) : 
    points_per_member = 6 :=
  sorry

end NUMINAMATH_GPT_trivia_team_points_l1090_109014


namespace NUMINAMATH_GPT_value_of_x_l1090_109024

-- Define the conditions
variable (C S x : ℝ)
variable (h1 : 20 * C = x * S)
variable (h2 : (S - C) / C * 100 = 25)

-- Define the statement to be proved
theorem value_of_x : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1090_109024


namespace NUMINAMATH_GPT_sum_of_factors_of_30_is_72_l1090_109002

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end NUMINAMATH_GPT_sum_of_factors_of_30_is_72_l1090_109002


namespace NUMINAMATH_GPT_M_inter_N_l1090_109046

def M : Set ℝ := { x | -2 < x ∧ x < 1 }
def N : Set ℤ := { x | Int.natAbs x ≤ 2 }

theorem M_inter_N : { x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) < 1 } ∩ N = { -1, 0 } :=
by
  simp [M, N]
  sorry

end NUMINAMATH_GPT_M_inter_N_l1090_109046


namespace NUMINAMATH_GPT_coefficient_of_x2_in_expansion_l1090_109020

theorem coefficient_of_x2_in_expansion :
  (x - (2 : ℤ)/x) ^ 4 = 8 * x^2 := sorry

end NUMINAMATH_GPT_coefficient_of_x2_in_expansion_l1090_109020


namespace NUMINAMATH_GPT_probability_perfect_square_l1090_109095

def is_perfect_square (n : ℕ) : Prop :=
  n = 1 ∨ n = 4

def successful_outcomes : Finset ℕ := {1, 4}

def total_possible_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_perfect_square :
  (successful_outcomes.card : ℚ) / (total_possible_outcomes.card : ℚ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_perfect_square_l1090_109095


namespace NUMINAMATH_GPT_interest_rate_is_correct_l1090_109068

variable (A P I : ℝ)
variable (T R : ℝ)

theorem interest_rate_is_correct
  (hA : A = 1232)
  (hP : P = 1100)
  (hT : T = 12 / 5)
  (hI : I = A - P) :
  R = I * 100 / (P * T) :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_is_correct_l1090_109068


namespace NUMINAMATH_GPT_complex_imaginary_axis_l1090_109004

theorem complex_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) ↔ (a = 0 ∨ a = 2) := 
by
  sorry

end NUMINAMATH_GPT_complex_imaginary_axis_l1090_109004


namespace NUMINAMATH_GPT_find_E_coordinates_l1090_109083

structure Point where
  x : ℚ
  y : ℚ

def A : Point := {x := -2, y := 1}
def B : Point := {x := 1, y := 4}
def C : Point := {x := 4, y := -3}
def D : Point := {x := (-2 * 1 + 1 * (-2)) / (1 + 2), y := (1 * 4 + 2 * 1) / (1 + 2)}

def externalDivision (P1 P2 : Point) (m n : ℚ) : Point :=
  {x := (m * P2.x - n * P1.x) / (m - n), y := (m * P2.y - n * P1.y) / (m - n)}

theorem find_E_coordinates :
  let E := externalDivision D C 1 4
  E.x = -8 / 3 ∧ E.y = 11 / 3 := 
by 
  let E := externalDivision D C 1 4
  sorry

end NUMINAMATH_GPT_find_E_coordinates_l1090_109083


namespace NUMINAMATH_GPT_fraction_sum_divided_by_2_equals_decimal_l1090_109085

theorem fraction_sum_divided_by_2_equals_decimal :
  let f1 := (3 : ℚ) / 20
  let f2 := (5 : ℚ) / 200
  let f3 := (7 : ℚ) / 2000
  let sum := f1 + f2 + f3
  let result := sum / 2
  result = 0.08925 := 
by
  sorry

end NUMINAMATH_GPT_fraction_sum_divided_by_2_equals_decimal_l1090_109085


namespace NUMINAMATH_GPT_cost_price_computer_table_l1090_109094

theorem cost_price_computer_table (C : ℝ) (S : ℝ) (H1 : S = C + 0.60 * C) (H2 : S = 2000) : C = 1250 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cost_price_computer_table_l1090_109094


namespace NUMINAMATH_GPT_Oleg_age_proof_l1090_109081

-- Defining the necessary conditions
variables (x y z : ℕ) -- defining the ages of Oleg, his father, and his grandfather

-- Stating the conditions
axiom h1 : y = x + 32
axiom h2 : z = y + 32
axiom h3 : (x - 3) + (y - 3) + (z - 3) < 100

-- Stating the proof problem
theorem Oleg_age_proof : 
  (x = 4) ∧ (y = 36) ∧ (z = 68) :=
by
  sorry

end NUMINAMATH_GPT_Oleg_age_proof_l1090_109081


namespace NUMINAMATH_GPT_maximum_b_n_T_l1090_109057

/-- Given a sequence {a_n} defined recursively and b_n = a_n / n.
   We need to prove that for all n in positive natural numbers,
   b_n is greater than or equal to T, and the maximum such T is 3. -/
theorem maximum_b_n_T (T : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  (∀ n, n ≥ 1 → b n = a n / n) →
  (∀ n, n ≥ 1 → b n ≥ T) →
  T ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_maximum_b_n_T_l1090_109057


namespace NUMINAMATH_GPT_solve_for_x_l1090_109067

theorem solve_for_x :
  ∀ x : ℚ, 10 * (5 * x + 4) - 4 = -4 * (2 - 15 * x) → x = 22 / 5 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1090_109067


namespace NUMINAMATH_GPT_contradiction_proof_l1090_109069

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) →
    false := 
by
  intros a b c d h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_contradiction_proof_l1090_109069


namespace NUMINAMATH_GPT_chord_line_equation_l1090_109079

theorem chord_line_equation 
  (x y : ℝ)
  (ellipse_eq : x^2 / 4 + y^2 / 3 = 1)
  (midpoint_condition : ∃ x1 y1 x2 y2 : ℝ, (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1
   ∧ (x1^2 / 4 + y1^2 / 3 = 1) ∧ (x2^2 / 4 + y2^2 / 3 = 1))
  : 3 * x - 4 * y + 7 = 0 :=
sorry

end NUMINAMATH_GPT_chord_line_equation_l1090_109079


namespace NUMINAMATH_GPT_crayons_per_friend_l1090_109010

theorem crayons_per_friend (total_crayons : ℕ) (num_friends : ℕ) (h1 : total_crayons = 210) (h2 : num_friends = 30) : total_crayons / num_friends = 7 :=
by
  sorry

end NUMINAMATH_GPT_crayons_per_friend_l1090_109010


namespace NUMINAMATH_GPT_find_y_when_x4_l1090_109076

theorem find_y_when_x4 : 
  (∀ x y : ℚ, 5 * y + 3 = 344 / (x ^ 3)) ∧ (5 * (8:ℚ) + 3 = 344 / (2 ^ 3)) → 
  (∃ y : ℚ, 5 * y + 3 = 344 / (4 ^ 3) ∧ y = 19 / 40) := 
by
  sorry

end NUMINAMATH_GPT_find_y_when_x4_l1090_109076


namespace NUMINAMATH_GPT_chi_squared_confidence_l1090_109059

theorem chi_squared_confidence (K_squared : ℝ) :
  (99.5 / 100 : ℝ) = 0.995 → (K_squared ≥ 7.879) :=
sorry

end NUMINAMATH_GPT_chi_squared_confidence_l1090_109059


namespace NUMINAMATH_GPT_sum_of_coefficients_3x_minus_1_pow_7_l1090_109048

theorem sum_of_coefficients_3x_minus_1_pow_7 :
  let f (x : ℕ) := (3 * x - 1) ^ 7
  (f 1) = 128 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_3x_minus_1_pow_7_l1090_109048


namespace NUMINAMATH_GPT_problem_equivalent_l1090_109053

theorem problem_equivalent (a b : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^2 * x^2)/2 + (a^3 * x^3)/6 + (a^4 * x^4)/24 + (a^5 * x^5)/120) : 
  a - b = -38 :=
sorry

end NUMINAMATH_GPT_problem_equivalent_l1090_109053


namespace NUMINAMATH_GPT_billy_has_2_cherries_left_l1090_109029

-- Define the initial number of cherries
def initialCherries : Nat := 74

-- Define the number of cherries eaten
def eatenCherries : Nat := 72

-- Define the number of remaining cherries
def remainingCherries : Nat := initialCherries - eatenCherries

-- Theorem statement: Prove that remainingCherries is equal to 2
theorem billy_has_2_cherries_left : remainingCherries = 2 := by
  sorry

end NUMINAMATH_GPT_billy_has_2_cherries_left_l1090_109029


namespace NUMINAMATH_GPT_initial_workers_l1090_109064

theorem initial_workers (M : ℝ) :
  let totalLength : ℝ := 15
  let totalDays : ℝ := 300
  let completedLength : ℝ := 2.5
  let completedDays : ℝ := 100
  let remainingLength : ℝ := totalLength - completedLength
  let remainingDays : ℝ := totalDays - completedDays
  let extraMen : ℝ := 60
  let rateWithM : ℝ := completedLength / completedDays
  let newRate : ℝ := remainingLength / remainingDays
  let newM : ℝ := M + extraMen
  (rateWithM * M = newRate * newM) → M = 100 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_initial_workers_l1090_109064


namespace NUMINAMATH_GPT_four_people_seven_chairs_l1090_109003

def num_arrangements (total_chairs : ℕ) (num_reserved : ℕ) (num_people : ℕ) : ℕ :=
  (total_chairs - num_reserved).choose num_people * (num_people.factorial)

theorem four_people_seven_chairs (total_chairs : ℕ) (chairs_occupied : ℕ) (num_people : ℕ): 
    total_chairs = 7 → chairs_occupied = 2 → num_people = 4 →
    num_arrangements total_chairs chairs_occupied num_people = 120 :=
by
  intros
  unfold num_arrangements
  sorry

end NUMINAMATH_GPT_four_people_seven_chairs_l1090_109003


namespace NUMINAMATH_GPT_train_crosses_bridge_in_30_seconds_l1090_109011

theorem train_crosses_bridge_in_30_seconds
    (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ)
    (h1 : train_length = 110)
    (h2 : train_speed_kmh = 45)
    (h3 : bridge_length = 265) : 
    (train_length + bridge_length) / (train_speed_kmh * (1000 / 3600)) = 30 := 
by
  sorry

end NUMINAMATH_GPT_train_crosses_bridge_in_30_seconds_l1090_109011


namespace NUMINAMATH_GPT_total_money_in_dollars_l1090_109050

/-- You have some amount in nickels and quarters.
    You have 40 nickels and the same number of quarters.
    Prove that the total amount of money in dollars is 12. -/
theorem total_money_in_dollars (n_nickels n_quarters : ℕ) (value_nickel value_quarter : ℕ) 
  (h1: n_nickels = 40) (h2: n_quarters = 40) (h3: value_nickel = 5) (h4: value_quarter = 25) : 
  (n_nickels * value_nickel + n_quarters * value_quarter) / 100 = 12 :=
  sorry

end NUMINAMATH_GPT_total_money_in_dollars_l1090_109050


namespace NUMINAMATH_GPT_ratio_of_part_to_whole_l1090_109044

theorem ratio_of_part_to_whole : 
  (1 / 4) * (2 / 5) * P = 15 → 
  (40 / 100) * N = 180 → 
  P / N = 1 / 6 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_ratio_of_part_to_whole_l1090_109044


namespace NUMINAMATH_GPT_woman_wait_time_for_man_to_catch_up_l1090_109051

theorem woman_wait_time_for_man_to_catch_up :
  ∀ (mans_speed womans_speed : ℕ) (time_after_passing : ℕ) (distance_up_slope : ℕ) (incline_percentage : ℕ),
  mans_speed = 5 →
  womans_speed = 25 →
  time_after_passing = 5 →
  distance_up_slope = 1 →
  incline_percentage = 5 →
  max 0 (mans_speed - incline_percentage * 1) = 0 →
  time_after_passing = 0 :=
by
  intros
  -- Insert proof here when needed
  sorry

end NUMINAMATH_GPT_woman_wait_time_for_man_to_catch_up_l1090_109051


namespace NUMINAMATH_GPT_projections_proportional_to_squares_l1090_109090

theorem projections_proportional_to_squares
  (a b c a1 b1 : ℝ)
  (h₀ : c ≠ 0)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a1 = (a^2) / c)
  (h₃ : b1 = (b^2) / c) :
  (a1 / b1) = (a^2 / b^2) :=
by sorry

end NUMINAMATH_GPT_projections_proportional_to_squares_l1090_109090


namespace NUMINAMATH_GPT_quadratic_solution_l1090_109062

theorem quadratic_solution :
  (∀ x : ℝ, (x^2 - x - 1 = 0) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = -(1 + Real.sqrt 5) / 2)) :=
by
  intro x
  rw [sub_eq_neg_add, sub_eq_neg_add]
  sorry

end NUMINAMATH_GPT_quadratic_solution_l1090_109062


namespace NUMINAMATH_GPT_parabola_directrix_eq_neg_2_l1090_109038

noncomputable def parabola_directrix (a b c : ℝ) : ℝ :=
  (b^2 - 4 * a * c) / (4 * a)

theorem parabola_directrix_eq_neg_2 (x : ℝ) :
  parabola_directrix 1 (-4) 4 = -2 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_parabola_directrix_eq_neg_2_l1090_109038


namespace NUMINAMATH_GPT_sum_of_solutions_l1090_109022

-- Given the quadratic equation: x^2 + 3x - 20 = 7x + 8
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 20 = 7*x + 8

-- Prove that the sum of the solutions to this quadratic equation is 4
theorem sum_of_solutions : 
  ∀ x1 x2 : ℝ, (quadratic_equation x1) ∧ (quadratic_equation x2) → x1 + x2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1090_109022


namespace NUMINAMATH_GPT_problem_statement_l1090_109049

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - Real.pi * x

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  ((deriv f x < 0) ∧ (f x < 0)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1090_109049


namespace NUMINAMATH_GPT_reflect_y_axis_l1090_109063

theorem reflect_y_axis (x y z : ℝ) : (x, y, z) = (1, -2, 3) → (-x, y, -z) = (-1, -2, -3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_reflect_y_axis_l1090_109063


namespace NUMINAMATH_GPT_find_a6_a7_l1090_109041

variable {a : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a n + d
axiom sum_given : a 2 + a 3 + a 10 + a 11 = 48

theorem find_a6_a7 (arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d) (h : a 2 + a 3 + a 10 + a 11 = 48) :
  a 6 + a 7 = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_a6_a7_l1090_109041


namespace NUMINAMATH_GPT_repeating_decimal_eq_l1090_109087

-- Defining the repeating decimal as a hypothesis
def repeating_decimal : ℚ := 0.7 + 3/10^2 * (1/(1 - 1/10))
-- We will prove this later by simplifying the fraction
def expected_fraction : ℚ := 11/15

theorem repeating_decimal_eq : repeating_decimal = expected_fraction := 
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_eq_l1090_109087


namespace NUMINAMATH_GPT_relationship_between_x_and_y_l1090_109021

theorem relationship_between_x_and_y (a b : ℝ) (x y : ℝ)
  (h1 : x = a^2 + b^2 + 20)
  (h2 : y = 4 * (2 * b - a)) :
  x ≥ y :=
by 
-- we need to prove x ≥ y
sorry

end NUMINAMATH_GPT_relationship_between_x_and_y_l1090_109021


namespace NUMINAMATH_GPT_investor_more_money_in_A_l1090_109061

noncomputable def investment_difference 
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ) :
  ℝ :=
investment_A * (1 + yield_A) - investment_B * (1 + yield_B)

theorem investor_more_money_in_A
  (investment_A : ℝ) 
  (investment_B : ℝ) 
  (yield_A : ℝ) 
  (yield_B : ℝ)
  (hA : investment_A = 300)
  (hB : investment_B = 200)
  (hYA : yield_A = 0.3)
  (hYB : yield_B = 0.5)
  :
  investment_difference investment_A investment_B yield_A yield_B = 90 := 
by
  sorry

end NUMINAMATH_GPT_investor_more_money_in_A_l1090_109061


namespace NUMINAMATH_GPT_Dan_age_is_28_l1090_109023

theorem Dan_age_is_28 (B D : ℕ) (h1 : B = D - 3) (h2 : B + D = 53) : D = 28 :=
by
  sorry

end NUMINAMATH_GPT_Dan_age_is_28_l1090_109023


namespace NUMINAMATH_GPT_solve_system1_solve_system2_l1090_109056

-- Define the conditions and the proof problem for System 1
theorem solve_system1 (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : 3 * x + 2 * y = 7) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- Define the conditions and the proof problem for System 2
theorem solve_system2 (x y : ℝ) (h1 : x - y = 3) (h2 : (x - y - 3) / 2 - y / 3 = -1) :
  x = 6 ∧ y = 3 := by
  sorry

end NUMINAMATH_GPT_solve_system1_solve_system2_l1090_109056


namespace NUMINAMATH_GPT_radius_of_outer_circle_l1090_109009

theorem radius_of_outer_circle (C_inner : ℝ) (width : ℝ) (h : C_inner = 880) (w : width = 25) :
  ∃ r_outer : ℝ, r_outer = 165 :=
by
  have r_inner := C_inner / (2 * Real.pi)
  have r_outer := r_inner + width
  use r_outer
  sorry

end NUMINAMATH_GPT_radius_of_outer_circle_l1090_109009


namespace NUMINAMATH_GPT_gcd_105_90_l1090_109058

theorem gcd_105_90 : Nat.gcd 105 90 = 15 :=
by
  sorry

end NUMINAMATH_GPT_gcd_105_90_l1090_109058


namespace NUMINAMATH_GPT_correct_option_l1090_109001

theorem correct_option (a b c : ℝ) : 
  (5 * a - (b + 2 * c) = 5 * a + b - 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a + b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b - 2 * c) ↔ 
  (5 * a - (b + 2 * c) = 5 * a - b - 2 * c) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l1090_109001


namespace NUMINAMATH_GPT_total_savings_during_sale_l1090_109091

theorem total_savings_during_sale :
  let regular_price_fox := 15
  let regular_price_pony := 20
  let pairs_fox := 3
  let pairs_pony := 2
  let total_discount := 22
  let discount_pony := 18.000000000000014
  let regular_total := (pairs_fox * regular_price_fox) + (pairs_pony * regular_price_pony)
  let discount_fox := total_discount - discount_pony
  (discount_fox / 100 * (pairs_fox * regular_price_fox)) + (discount_pony / 100 * (pairs_pony * regular_price_pony)) = 9 := by
  sorry

end NUMINAMATH_GPT_total_savings_during_sale_l1090_109091


namespace NUMINAMATH_GPT_abs_inequality_solution_l1090_109074

theorem abs_inequality_solution (x : ℝ) : |x - 2| < 1 ↔ 1 < x ∧ x < 3 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1090_109074


namespace NUMINAMATH_GPT_angle_bisector_eqn_l1090_109097

-- Define the vertices A, B, and C
def A : (ℝ × ℝ) := (4, 3)
def B : (ℝ × ℝ) := (-4, -1)
def C : (ℝ × ℝ) := (9, -7)

-- State the theorem with conditions and the given answer
theorem angle_bisector_eqn (A B C : (ℝ × ℝ)) (hA : A = (4, 3)) (hB : B = (-4, -1)) (hC : C = (9, -7)) :
  ∃ b c, (3:ℝ) * (3:ℝ) - b * (3:ℝ) + c = 0 ∧ b + c = -6 := 
by 
  use -1, -5
  simp
  sorry

end NUMINAMATH_GPT_angle_bisector_eqn_l1090_109097


namespace NUMINAMATH_GPT_sum_of_intervals_length_l1090_109089

theorem sum_of_intervals_length (m : ℝ) (h : m ≠ 0) (h_pos : m > 0) :
  (∃ l : ℝ, ∀ x : ℝ, (1 < x ∧ x ≤ x₁) ∨ (2 < x ∧ x ≤ x₂) → 
  l = x₁ - 1 + x₂ - 2) → 
  l = 3 / m :=
sorry

end NUMINAMATH_GPT_sum_of_intervals_length_l1090_109089


namespace NUMINAMATH_GPT_f_1987_eq_5_l1090_109036

noncomputable def f : ℕ → ℝ := sorry

axiom f_def : ∀ x : ℕ, x ≥ 0 → ∃ y : ℝ, f x = y
axiom f_one : f 1 = 2
axiom functional_eq : ∀ a b : ℕ, a ≥ 0 → b ≥ 0 → f (a + b) = f a + f b - 3 * f (a * b) + 1

theorem f_1987_eq_5 : f 1987 = 5 := sorry

end NUMINAMATH_GPT_f_1987_eq_5_l1090_109036


namespace NUMINAMATH_GPT_gcd_40304_30203_eq_1_l1090_109070

theorem gcd_40304_30203_eq_1 : Nat.gcd 40304 30203 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_40304_30203_eq_1_l1090_109070


namespace NUMINAMATH_GPT_characterize_functional_equation_l1090_109071

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

theorem characterize_functional_equation (f : ℝ → ℝ) (h : satisfies_condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_GPT_characterize_functional_equation_l1090_109071


namespace NUMINAMATH_GPT_range_of_a_l1090_109017

def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, operation (x - a) (x + 1) < 1) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1090_109017
