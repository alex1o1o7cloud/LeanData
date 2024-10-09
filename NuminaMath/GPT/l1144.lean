import Mathlib

namespace time_for_B_is_24_days_l1144_114498

noncomputable def A_work : ℝ := (1 / 2) / (3 / 4)
noncomputable def B_work : ℝ := 1 -- assume B does 1 unit of work in 1 day
noncomputable def total_work : ℝ := (A_work + B_work) * 18

theorem time_for_B_is_24_days : 
  ((A_work + B_work) * 18) / B_work = 24 := by
  sorry

end time_for_B_is_24_days_l1144_114498


namespace compare_neg_rational_l1144_114436

def neg_one_third : ℚ := -1 / 3
def neg_one_half : ℚ := -1 / 2

theorem compare_neg_rational : neg_one_third > neg_one_half :=
by sorry

end compare_neg_rational_l1144_114436


namespace percentage_of_liquid_X_in_solution_A_l1144_114492

theorem percentage_of_liquid_X_in_solution_A (P : ℝ) :
  (0.018 * 700 / 1200 + P * 500 / 1200) = 0.0166 → P = 0.01464 :=
by 
  sorry

end percentage_of_liquid_X_in_solution_A_l1144_114492


namespace complement_A_complement_B_intersection_A_B_complement_union_A_B_l1144_114425

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def set_U : Set ℝ := {x | true}  -- This represents U = ℝ
def set_A : Set ℝ := {x | x < -2 ∨ x > 5}
def set_B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

theorem complement_A :
  ∀ x : ℝ, x ∈ set_U \ set_A ↔ -2 ≤ x ∧ x ≤ 5 :=
by
  intro x
  sorry

theorem complement_B :
  ∀ x : ℝ, x ∉ set_B ↔ x < 4 ∨ x > 6 :=
by
  intro x
  sorry

theorem intersection_A_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ 5 < x ∧ x ≤ 6 :=
by
  intro x
  sorry

theorem complement_union_A_B :
  ∀ x : ℝ, x ∈ set_U \ (set_A ∪ set_B) ↔ -2 ≤ x ∧ x < 4 :=
by
  intro x
  sorry

end complement_A_complement_B_intersection_A_B_complement_union_A_B_l1144_114425


namespace least_prime_factor_of_11_pow4_minus_11_pow3_l1144_114468

open Nat

theorem least_prime_factor_of_11_pow4_minus_11_pow3 : 
  Nat.minFac (11^4 - 11^3) = 2 :=
  sorry

end least_prime_factor_of_11_pow4_minus_11_pow3_l1144_114468


namespace divisor_condition_l1144_114433

def M (n : ℤ) : Set ℤ := {n, n+1, n+2, n+3, n+4}

def S (n : ℤ) : ℤ := 5*n^2 + 20*n + 30

def P (n : ℤ) : ℤ := (n * (n+1) * (n+2) * (n+3) * (n+4))^2

theorem divisor_condition (n : ℤ) : S n ∣ P n ↔ n = 3 := 
by
  sorry

end divisor_condition_l1144_114433


namespace independent_trials_probability_l1144_114496

theorem independent_trials_probability (p : ℝ) (q : ℝ) (ε : ℝ) (desired_prob : ℝ) 
    (h_p : p = 0.7) (h_q : q = 0.3) (h_ε : ε = 0.2) (h_desired_prob : desired_prob = 0.96) :
    ∃ n : ℕ, n > (p * q) / (desired_prob * ε^2) ∧ n = 132 :=
by
  sorry

end independent_trials_probability_l1144_114496


namespace sum_of_integers_70_to_85_l1144_114448

theorem sum_of_integers_70_to_85 :
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sum = 1240 :=
by
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sorry

end sum_of_integers_70_to_85_l1144_114448


namespace distance_between_intersection_points_l1144_114452

noncomputable def C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def l (t : ℝ) : ℝ × ℝ :=
  (-2 * t + 2, 3 * t)

theorem distance_between_intersection_points :
  ∃ (A B : ℝ × ℝ), 
    (∃ θ : ℝ, C θ = A) ∧
    (∃ t : ℝ, l t = A) ∧
    (∃ θ : ℝ, C θ = B) ∧
    (∃ t : ℝ, l t = B) ∧
    dist A B = Real.sqrt 13 / 2 :=
sorry

end distance_between_intersection_points_l1144_114452


namespace largest_angle_measure_l1144_114457

noncomputable def measure_largest_angle (x : ℚ) : Prop :=
  let a1 := 2 * x + 2
  let a2 := 3 * x
  let a3 := 4 * x + 3
  let a4 := 5 * x
  let a5 := 6 * x - 1
  let a6 := 7 * x
  a1 + a2 + a3 + a4 + a5 + a6 = 720 ∧ a6 = 5012 / 27

theorem largest_angle_measure : ∃ x : ℚ, measure_largest_angle x := by
  sorry

end largest_angle_measure_l1144_114457


namespace proof_of_independence_l1144_114461

/-- A line passing through the plane of two parallel lines and intersecting one of them also intersects the other. -/
def independent_of_parallel_postulate (statement : String) : Prop :=
  statement = "A line passing through the plane of two parallel lines and intersecting one of them also intersects the other."

theorem proof_of_independence :
  independent_of_parallel_postulate "A line passing through the plane of two parallel lines and intersecting one of them also intersects the other." :=
sorry

end proof_of_independence_l1144_114461


namespace train_speed_is_100_kmph_l1144_114478

noncomputable def speed_of_train (length_of_train : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  (length_of_train / time_to_cross_pole) * 3.6

theorem train_speed_is_100_kmph :
  speed_of_train 100 3.6 = 100 :=
by
  sorry

end train_speed_is_100_kmph_l1144_114478


namespace train_length_is_correct_l1144_114400

noncomputable def train_length (speed_kmph : ℝ) (crossing_time_s : ℝ) (platform_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * crossing_time_s
  total_distance - platform_length_m

theorem train_length_is_correct :
  train_length 60 14.998800095992321 150 = 100 := by
  sorry

end train_length_is_correct_l1144_114400


namespace right_triangle_other_side_l1144_114409

theorem right_triangle_other_side (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 17) (h_a : a = 15) : b = 8 := 
by
  sorry

end right_triangle_other_side_l1144_114409


namespace equilateral_given_inequality_l1144_114438

open Real

-- Define the primary condition to be used in the theorem
def inequality (a b c : ℝ) : Prop :=
  (1 / a * sqrt (1 / b + 1 / c) + 1 / b * sqrt (1 / c + 1 / a) + 1 / c * sqrt (1 / a + 1 / b)) ≥
  (3 / 2 * sqrt ((1 / a + 1 / b) * (1 / b + 1 / c) * (1 / c + 1 / a)))

-- Define the theorem that states the sides form an equilateral triangle under the given condition
theorem equilateral_given_inequality (a b c : ℝ) (habc : inequality a b c) (htriangle : a > 0 ∧ b > 0 ∧ c > 0):
  a = b ∧ b = c ∧ c = a := 
sorry

end equilateral_given_inequality_l1144_114438


namespace first_triangular_number_year_in_21st_century_l1144_114466

theorem first_triangular_number_year_in_21st_century :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2016 ∧ 2000 ≤ 2016 ∧ 2016 < 2100 :=
by
  sorry

end first_triangular_number_year_in_21st_century_l1144_114466


namespace subcommittees_with_at_least_one_teacher_l1144_114413

theorem subcommittees_with_at_least_one_teacher
  (total_members teachers : ℕ)
  (total_members_eq : total_members = 12)
  (teachers_eq : teachers = 5)
  (subcommittee_size : ℕ)
  (subcommittee_size_eq : subcommittee_size = 5) :
  ∃ (n : ℕ), n = 771 :=
by
  sorry

end subcommittees_with_at_least_one_teacher_l1144_114413


namespace pin_probability_l1144_114469

theorem pin_probability :
  let total_pins := 9 * 10^5
  let valid_pins := 10^4
  ∃ p : ℚ, p = valid_pins / total_pins ∧ p = 1 / 90 := by
  sorry

end pin_probability_l1144_114469


namespace gondor_total_earnings_l1144_114424

-- Defining the earnings from repairing a phone and a laptop
def phone_earning : ℕ := 10
def laptop_earning : ℕ := 20

-- Defining the number of repairs
def monday_phone_repairs : ℕ := 3
def tuesday_phone_repairs : ℕ := 5
def wednesday_laptop_repairs : ℕ := 2
def thursday_laptop_repairs : ℕ := 4

-- Calculating total earnings
def monday_earnings : ℕ := monday_phone_repairs * phone_earning
def tuesday_earnings : ℕ := tuesday_phone_repairs * phone_earning
def wednesday_earnings : ℕ := wednesday_laptop_repairs * laptop_earning
def thursday_earnings : ℕ := thursday_laptop_repairs * laptop_earning

def total_earnings : ℕ := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings

-- The theorem to be proven
theorem gondor_total_earnings : total_earnings = 200 := by
  sorry

end gondor_total_earnings_l1144_114424


namespace halloween_candy_l1144_114441

theorem halloween_candy (katie_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) (total_candy : ℕ) (eaten_candy : ℕ)
  (h1 : katie_candy = 10) 
  (h2 : sister_candy = 6) 
  (h3 : remaining_candy = 7) 
  (h4 : total_candy = katie_candy + sister_candy) 
  (h5 : eaten_candy = total_candy - remaining_candy) : 
  eaten_candy = 9 :=
by sorry

end halloween_candy_l1144_114441


namespace angle_triple_supplement_l1144_114489

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l1144_114489


namespace ratio_gluten_free_l1144_114412

theorem ratio_gluten_free (total_cupcakes vegan_cupcakes non_vegan_gluten cupcakes_gluten_free : ℕ)
    (H1 : total_cupcakes = 80)
    (H2 : vegan_cupcakes = 24)
    (H3 : non_vegan_gluten = 28)
    (H4 : cupcakes_gluten_free = vegan_cupcakes / 2) :
    (cupcakes_gluten_free : ℚ) / (total_cupcakes : ℚ) = 3 / 20 :=
by 
  -- Proof goes here
  sorry

end ratio_gluten_free_l1144_114412


namespace problem_l1144_114485

def remainder_when_divided_by_20 (a b : ℕ) : ℕ := (a + b) % 20

theorem problem (a b : ℕ) (n m : ℤ) (h1 : a = 60 * n + 53) (h2 : b = 50 * m + 24) : 
  remainder_when_divided_by_20 a b = 17 := 
by
  -- Proof would go here
  sorry

end problem_l1144_114485


namespace interior_points_in_divided_square_l1144_114490

theorem interior_points_in_divided_square :
  ∀ (n : ℕ), 
  (n = 2016) →
  ∃ (k : ℕ), 
  (∀ (t : ℕ), t = 180 * n) → 
  k = 1007 :=
by
  intros n hn
  use 1007
  sorry

end interior_points_in_divided_square_l1144_114490


namespace one_half_of_scientific_notation_l1144_114419

theorem one_half_of_scientific_notation :
  (1 / 2) * (1.2 * 10 ^ 30) = 6.0 * 10 ^ 29 :=
by
  sorry

end one_half_of_scientific_notation_l1144_114419


namespace find_length_of_polaroid_l1144_114410

theorem find_length_of_polaroid 
  (C : ℝ) (W : ℝ) (L : ℝ)
  (hC : C = 40) (hW : W = 8) 
  (hFormula : C = 2 * (L + W)) : 
  L = 12 :=
by
  sorry

end find_length_of_polaroid_l1144_114410


namespace Nina_money_before_tax_l1144_114491

theorem Nina_money_before_tax :
  ∃ (M P : ℝ), M = 6 * P ∧ M = 8 * 0.9 * P ∧ M = 5 :=
by 
  sorry

end Nina_money_before_tax_l1144_114491


namespace geometric_sequence_common_ratio_and_general_formula_l1144_114402

variable (a : ℕ → ℝ)

theorem geometric_sequence_common_ratio_and_general_formula (h₁ : a 1 = 1) (h₃ : a 3 = 4) : 
  (∃ q : ℝ, q = 2 ∨ q = -2 ∧ (∀ n : ℕ, a n = 2^(n-1) ∨ a n = (-2)^(n-1))) := 
by
  sorry

end geometric_sequence_common_ratio_and_general_formula_l1144_114402


namespace quadratic_two_equal_real_roots_l1144_114449

theorem quadratic_two_equal_real_roots (m : ℝ) :
  (∃ (x : ℝ), x^2 + m * x + m = 0 ∧ ∀ (y : ℝ), x = y → x^2 + m * y + m = 0) →
  (m = 0 ∨ m = 4) :=
by {
  sorry
}

end quadratic_two_equal_real_roots_l1144_114449


namespace find_x_when_areas_equal_l1144_114487

-- Definitions based on the problem conditions
def glass_area : ℕ := 4 * (30 * 20)
def window_area (x : ℕ) : ℕ := (60 + 3 * x) * (40 + 3 * x)
def total_area_of_glass : ℕ := glass_area
def total_area_of_wood (x : ℕ) : ℕ := window_area x - glass_area

-- Proof problem, proving x == 20 / 3 when total area of glass equals total area of wood
theorem find_x_when_areas_equal : 
  ∃ x : ℕ, (total_area_of_glass = total_area_of_wood x) ∧ x = 20 / 3 :=
sorry

end find_x_when_areas_equal_l1144_114487


namespace probability_not_all_dice_same_l1144_114474

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l1144_114474


namespace find_missing_number_l1144_114458

theorem find_missing_number (x : ℝ) :
  ((20 + 40 + 60) / 3) = ((10 + 70 + x) / 3) + 8 → x = 16 :=
by
  intro h
  sorry

end find_missing_number_l1144_114458


namespace percent_gain_correct_l1144_114481

theorem percent_gain_correct :
  ∀ (x : ℝ), (900 * x + 50 * (900 * x / 850) - 900 * x) / (900 * x) * 100 = 58.82 :=
by sorry

end percent_gain_correct_l1144_114481


namespace range_of_square_root_l1144_114429

theorem range_of_square_root (x : ℝ) : x + 4 ≥ 0 → x ≥ -4 :=
by
  intro h
  linarith

end range_of_square_root_l1144_114429


namespace weather_conclusion_l1144_114446

variables (T C : ℝ) (visitors : ℕ)

def condition1 : Prop :=
  (T ≥ 75.0 ∧ C < 10) → visitors > 100

def condition2 : Prop :=
  visitors ≤ 100

theorem weather_conclusion (h1 : condition1 T C visitors) (h2 : condition2 visitors) : 
  T < 75.0 ∨ C ≥ 10 :=
by 
  sorry

end weather_conclusion_l1144_114446


namespace sufficient_condition_range_k_l1144_114475

theorem sufficient_condition_range_k {x k : ℝ} (h : ∀ x, x > k → (3 / (x + 1) < 1)) : k ≥ 2 :=
sorry

end sufficient_condition_range_k_l1144_114475


namespace sum_reciprocals_seven_l1144_114427

variable (x y : ℝ)

theorem sum_reciprocals_seven (h : x + y = 7 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / x) + (1 / y) = 7 := 
sorry

end sum_reciprocals_seven_l1144_114427


namespace range_of_a_l1144_114455

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l1144_114455


namespace product_of_two_numbers_in_ratio_l1144_114464

theorem product_of_two_numbers_in_ratio (x y : ℚ) 
  (h1 : x - y = d)
  (h2 : x + y = 8 * d)
  (h3 : x * y = 15 * d) :
  x * y = 100 / 7 :=
by
  sorry

end product_of_two_numbers_in_ratio_l1144_114464


namespace average_test_score_l1144_114482

theorem average_test_score (x : ℝ) :
  (0.45 * 95 + 0.50 * x + 0.05 * 60 = 84.75) → x = 78 :=
by
  sorry

end average_test_score_l1144_114482


namespace additional_days_use_l1144_114473

variable (m a : ℝ)

theorem additional_days_use (hm : m > 0) (ha : a > 1) : 
  (m / (a - 1) - m / a) = m / (a * (a - 1)) :=
sorry

end additional_days_use_l1144_114473


namespace marks_per_correct_answer_l1144_114411

-- Definitions based on the conditions
def total_questions : ℕ := 60
def total_marks : ℕ := 160
def correct_questions : ℕ := 44
def wrong_mark_loss : ℕ := 1

-- The number of correct answers multiplies the marks per correct answer,
-- minus the loss from wrong answers, equals the total marks.
theorem marks_per_correct_answer (x : ℕ) :
  correct_questions * x - (total_questions - correct_questions) * wrong_mark_loss = total_marks → x = 4 := by
sorry

end marks_per_correct_answer_l1144_114411


namespace carpet_area_in_yards_l1144_114403

def main_length_feet : ℕ := 15
def main_width_feet : ℕ := 12
def extension_length_feet : ℕ := 6
def extension_width_feet : ℕ := 5
def feet_per_yard : ℕ := 3

def main_length_yards : ℕ := main_length_feet / feet_per_yard
def main_width_yards : ℕ := main_width_feet / feet_per_yard
def extension_length_yards : ℕ := extension_length_feet / feet_per_yard
def extension_width_yards : ℕ := extension_width_feet / feet_per_yard

def main_area_yards : ℕ := main_length_yards * main_width_yards
def extension_area_yards : ℕ := extension_length_yards * extension_width_yards

theorem carpet_area_in_yards : (main_area_yards : ℚ) + (extension_area_yards : ℚ) = 23.33 := 
by
  apply sorry

end carpet_area_in_yards_l1144_114403


namespace fourth_term_sum_eq_40_l1144_114444

theorem fourth_term_sum_eq_40 : 3^0 + 3^1 + 3^2 + 3^3 = 40 := by
  sorry

end fourth_term_sum_eq_40_l1144_114444


namespace find_x_if_perpendicular_l1144_114465

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (2 * x - 1, 3)
def vec_n : ℝ × ℝ := (1, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_x_if_perpendicular (x : ℝ) : 
  dot_product (vec_m x) vec_n = 0 ↔ x = 2 :=
by
  sorry

end find_x_if_perpendicular_l1144_114465


namespace xy_minimization_l1144_114454

theorem xy_minimization (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : (1 / (x : ℝ)) + 1 / (3 * y) = 1 / 11) : x * y = 176 ∧ x + y = 30 :=
by
  sorry

end xy_minimization_l1144_114454


namespace smallest_integer_five_consecutive_sum_2025_l1144_114476

theorem smallest_integer_five_consecutive_sum_2025 :
  ∃ n : ℤ, 5 * n + 10 = 2025 ∧ n = 403 :=
by
  sorry

end smallest_integer_five_consecutive_sum_2025_l1144_114476


namespace mart_income_percentage_l1144_114477

theorem mart_income_percentage 
  (J T M : ℝ)
  (h1 : M = 1.60 * T)
  (h2 : T = 0.60 * J) :
  M = 0.96 * J :=
sorry

end mart_income_percentage_l1144_114477


namespace negation_of_p_l1144_114460

def p : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - x ≥ 0

theorem negation_of_p : ¬ p ↔ ∃ x : ℝ, x ≥ 0 ∧ x^2 - x < 0 :=
by
  sorry

end negation_of_p_l1144_114460


namespace instantaneous_acceleration_at_3_l1144_114467

def v (t : ℝ) : ℝ := t^2 + 3

theorem instantaneous_acceleration_at_3 :
  deriv v 3 = 6 :=
by
  sorry

end instantaneous_acceleration_at_3_l1144_114467


namespace circles_intersection_distance_squared_l1144_114437

open Real

-- Definitions of circles
def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 25

def circle2 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 6)^2 = 9

-- Theorem to prove
theorem circles_intersection_distance_squared :
  ∃ A B : (ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B) ∧
  (dist A B)^2 = 675 / 49 :=
sorry

end circles_intersection_distance_squared_l1144_114437


namespace solve_system_of_equations_l1144_114450

theorem solve_system_of_equations:
  (∀ (x y : ℝ), 2 * y - x - 2 * x * y = -1 ∧ 4 * x ^ 2 * y ^ 2 + x ^ 2 + 4 * y ^ 2 - 4 * x * y = 61 →
  (x, y) = (-6, -1/2) ∨ (x, y) = (1, 3) ∨ (x, y) = (1, -5/2) ∨ (x, y) = (5, -1/2)) :=
by
  sorry

end solve_system_of_equations_l1144_114450


namespace kittens_given_is_two_l1144_114470

-- Definitions of the conditions
def original_kittens : Nat := 8
def current_kittens : Nat := 6

-- Statement of the proof problem
theorem kittens_given_is_two : (original_kittens - current_kittens) = 2 := 
by
  sorry

end kittens_given_is_two_l1144_114470


namespace largest_value_a_plus_b_plus_c_l1144_114432

open Nat
open Function

def sum_of_digits (n : ℕ) : ℕ :=
  (digits 10 n).sum

theorem largest_value_a_plus_b_plus_c :
  ∃ (a b c : ℕ),
    10 ≤ a ∧ a < 100 ∧
    100 ≤ b ∧ b < 1000 ∧
    1000 ≤ c ∧ c < 10000 ∧
    sum_of_digits (a + b) = 2 ∧
    sum_of_digits (b + c) = 2 ∧
    (a + b + c = 10199) := sorry

end largest_value_a_plus_b_plus_c_l1144_114432


namespace area_of_rectangle_inscribed_in_triangle_l1144_114416

theorem area_of_rectangle_inscribed_in_triangle :
  ∀ (E F G A B C D : ℝ) (EG altitude_ABCD : ℝ),
    E < F ∧ F < G ∧ A < B ∧ B < C ∧ C < D ∧ A < D ∧ D < G ∧ A < G ∧
    EG = 10 ∧ 
    altitude_ABCD = 7 ∧ 
    B = C ∧ 
    A + D = EG ∧ 
    A + 2 * B = EG →
    ((A * B) = (1225 / 72)) :=
by
  intros E F G A B C D EG altitude_ABCD
  intro h
  sorry

end area_of_rectangle_inscribed_in_triangle_l1144_114416


namespace border_area_correct_l1144_114421

theorem border_area_correct :
  let photo_height := 9
  let photo_width := 12
  let border_width := 3
  let photo_area := photo_height * photo_width
  let framed_height := photo_height + 2 * border_width
  let framed_width := photo_width + 2 * border_width
  let framed_area := framed_height * framed_width
  let border_area := framed_area - photo_area
  border_area = 162 :=
by sorry

end border_area_correct_l1144_114421


namespace good_numbers_count_1_to_50_l1144_114405

def is_good_number (n : ℕ) : Prop :=
  ∃ (k l : ℕ), k ≠ 0 ∧ l ≠ 0 ∧ n = k * l + l - k

theorem good_numbers_count_1_to_50 : ∃ cnt, cnt = 49 ∧ (∀ n, n ∈ (Finset.range 51).erase 0 → is_good_number n) :=
  sorry

end good_numbers_count_1_to_50_l1144_114405


namespace quadratic_root_d_value_l1144_114486

theorem quadratic_root_d_value :
  (∃ d : ℝ, ∀ x : ℝ, (2 * x^2 + 8 * x + d = 0) ↔ (x = (-8 + Real.sqrt 12) / 4) ∨ (x = (-8 - Real.sqrt 12) / 4)) → 
  d = 6.5 :=
by
  sorry

end quadratic_root_d_value_l1144_114486


namespace inequality_proof_l1144_114426

theorem inequality_proof (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  (a^2 - b * c) / (2 * a^2 + b * c) + (b^2 - c * a) / (2 * b^2 + c * a) + (c^2 - a * b) / (2 * c^2 + a * b) ≤ 0 :=
sorry

end inequality_proof_l1144_114426


namespace particular_solution_satisfies_initial_conditions_l1144_114420

noncomputable def x_solution : ℝ → ℝ := λ t => (-4/3) * Real.exp t + (7/3) * Real.exp (-2 * t)
noncomputable def y_solution : ℝ → ℝ := λ t => (-1/3) * Real.exp t + (7/3) * Real.exp (-2 * t)

def x_prime (x y : ℝ) := 2 * x - 4 * y
def y_prime (x y : ℝ) := x - 3 * y

theorem particular_solution_satisfies_initial_conditions :
  (∀ t, deriv x_solution t = x_prime (x_solution t) (y_solution t)) ∧
  (∀ t, deriv y_solution t = y_prime (x_solution t) (y_solution t)) ∧
  (x_solution 0 = 1) ∧
  (y_solution 0 = 2) := by
  sorry

end particular_solution_satisfies_initial_conditions_l1144_114420


namespace largest_rectangle_area_l1144_114463

theorem largest_rectangle_area (l w : ℕ) (hl : l > 0) (hw : w > 0) (hperimeter : 2 * l + 2 * w = 42)
  (harea_diff : ∃ (l1 w1 l2 w2 : ℕ), l1 > 0 ∧ w1 > 0 ∧ l2 > 0 ∧ w2 > 0 ∧ 2 * l1 + 2 * w1 = 42 
  ∧ 2 * l2 + 2 * w2 = 42 ∧ (l1 * w1) - (l2 * w2) = 90) : (l * w ≤ 110) :=
sorry

end largest_rectangle_area_l1144_114463


namespace line_perpendicular_to_plane_l1144_114443

-- Define a structure for vectors in 3D
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define line l with the given direction vector
def direction_vector_l : Vector3D := ⟨1, -1, -2⟩

-- Define plane α with the given normal vector
def normal_vector_alpha : Vector3D := ⟨2, -2, -4⟩

-- Prove that line l is perpendicular to plane α
theorem line_perpendicular_to_plane :
  let a := direction_vector_l
  let b := normal_vector_alpha
  (b.x = 2 * a.x) ∧ (b.y = 2 * a.y) ∧ (b.z = 2 * a.z) → 
  (a.x * b.x + a.y * b.y + a.z * b.z = 0) :=
by
  intro a b h
  sorry

end line_perpendicular_to_plane_l1144_114443


namespace find_range_of_m_l1144_114445

-- Define propositions p and q based on the problem description
def p (m : ℝ) : Prop :=
  ∀ x y : ℝ, m ≠ 0 → (x - 2 * y + 3 = 0 ∧ y * y ≠ m * x)

def q (m : ℝ) : Prop :=
  5 - 2 * m ≠ 0 ∧ m ≠ 0 ∧ (∃ x y : ℝ, (x * x) / (5 - 2 * m) + (y * y) / m = 1)

-- Given conditions
def condition1 (m : ℝ) : Prop := p m ∨ q m
def condition2 (m : ℝ) : Prop := ¬ (p m ∧ q m)

-- The range of m that satisfies the given problem
def valid_m (m : ℝ) : Prop :=
  (m ≥ 3) ∨ (m < 0) ∨ (0 < m ∧ m ≤ 2.5)

theorem find_range_of_m (m : ℝ) : condition1 m → condition2 m → valid_m m := 
  sorry

end find_range_of_m_l1144_114445


namespace vector_c_equals_combination_l1144_114451

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)
def vector_c : ℝ × ℝ := (-2, 4)

theorem vector_c_equals_combination : vector_c = (vector_a.1 - 3 * vector_b.1, vector_a.2 - 3 * vector_b.2) :=
sorry

end vector_c_equals_combination_l1144_114451


namespace simplify_polynomial_l1144_114408

theorem simplify_polynomial (q : ℚ) :
  (4 * q^3 - 7 * q^2 + 3 * q - 2) + (5 * q^2 - 9 * q + 8) = 4 * q^3 - 2 * q^2 - 6 * q + 6 := 
by 
  sorry

end simplify_polynomial_l1144_114408


namespace tangent_line_to_curve_l1144_114406

theorem tangent_line_to_curve (a : ℝ) : (∀ (x : ℝ), y = x → y = a + Real.log x) → a = 1 := 
sorry

end tangent_line_to_curve_l1144_114406


namespace max_contribution_l1144_114480

theorem max_contribution (total_contribution : ℝ) (num_people : ℕ) (min_contribution_each : ℝ) (h1 : total_contribution = 45.00) (h2 : num_people = 25) (h3 : min_contribution_each = 1.00) : 
  ∃ max_cont : ℝ, max_cont = 21.00 :=
by
  sorry

end max_contribution_l1144_114480


namespace range_of_k_l1144_114483

noncomputable def circle_equation (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem range_of_k (k : ℝ) :
  circle_equation k →
  k ∈ (Set.Iio (-1) ∪ Set.Ioi 4) :=
sorry

end range_of_k_l1144_114483


namespace percentage_change_in_receipts_l1144_114484

theorem percentage_change_in_receipts
  (P S : ℝ) -- Original price and sales
  (hP : P > 0)
  (hS : S > 0)
  (new_P : ℝ := 0.70 * P) -- Price after 30% reduction
  (new_S : ℝ := 1.50 * S) -- Sales after 50% increase
  :
  (new_P * new_S - P * S) / (P * S) * 100 = 5 :=
by
  sorry

end percentage_change_in_receipts_l1144_114484


namespace books_left_unchanged_l1144_114407

theorem books_left_unchanged (initial_books : ℕ) (initial_pens : ℕ) (pens_sold : ℕ) (pens_left : ℕ) :
  initial_books = 51 → initial_pens = 106 → pens_sold = 92 → pens_left = 14 → initial_books = 51 := 
by
  intros h_books h_pens h_sold h_left
  exact h_books

end books_left_unchanged_l1144_114407


namespace area_of_right_triangle_l1144_114488

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end area_of_right_triangle_l1144_114488


namespace find_n_values_l1144_114442

theorem find_n_values (n : ℚ) :
  ( 4 * n ^ 2 + 3 * n + 2 = 2 * n + 2 ∨ 4 * n ^ 2 + 3 * n + 2 = 5 * n + 4 ) →
  ( n = 0 ∨ n = 1 ) :=
by
  sorry

end find_n_values_l1144_114442


namespace total_bricks_l1144_114453

theorem total_bricks (n1 n2 r1 r2 : ℕ) (w1 w2 : ℕ)
  (h1 : n1 = 60) (h2 : r1 = 100) (h3 : n2 = 80) (h4 : r2 = 120)
  (h5 : w1 = 5) (h6 : w2 = 5) :
  (w1 * (n1 * r1) + w2 * (n2 * r2)) = 78000 :=
by sorry

end total_bricks_l1144_114453


namespace calculate_glass_area_l1144_114439

-- Given conditions as definitions
def long_wall_length : ℕ := 30
def long_wall_height : ℕ := 12
def short_wall_length : ℕ := 20

-- Total area of glass required (what we want to prove)
def total_glass_area : ℕ := 960

-- The theorem to prove
theorem calculate_glass_area
  (a1 : long_wall_length = 30)
  (a2 : long_wall_height = 12)
  (a3 : short_wall_length = 20) :
  2 * (long_wall_length * long_wall_height) + (short_wall_length * long_wall_height) = total_glass_area :=
by
  -- The proof is omitted
  sorry

end calculate_glass_area_l1144_114439


namespace election_majority_l1144_114494

theorem election_majority (total_votes : ℕ) (winning_percentage : ℝ) (losing_percentage : ℝ)
  (h_total_votes : total_votes = 700)
  (h_winning_percentage : winning_percentage = 0.70)
  (h_losing_percentage : losing_percentage = 0.30) :
  (winning_percentage * total_votes - losing_percentage * total_votes) = 280 :=
by
  sorry

end election_majority_l1144_114494


namespace range_of_a_l1144_114459

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ax^2 + 3 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4 / 9) :=
sorry

end range_of_a_l1144_114459


namespace problem_equivalent_l1144_114417

theorem problem_equivalent :
  2^1998 - 2^1997 - 2^1996 + 2^1995 = 3 * 2^1995 :=
by
  sorry

end problem_equivalent_l1144_114417


namespace one_quarters_in_one_eighth_l1144_114414

theorem one_quarters_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 :=
by sorry

end one_quarters_in_one_eighth_l1144_114414


namespace quadratic_inequality_min_value_l1144_114434

noncomputable def min_value (a b: ℝ) : ℝ := 2 * a^2 + b^2

theorem quadratic_inequality_min_value
  (a b: ℝ) (hx: ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (x0: ℝ) (hx0: a * x0^2 + 2 * x0 + b = 0) :
  a > b → min_value a b = 2 * Real.sqrt 2 := 
sorry

end quadratic_inequality_min_value_l1144_114434


namespace jump_difference_l1144_114423

def frog_jump := 39
def grasshopper_jump := 17

theorem jump_difference :
  frog_jump - grasshopper_jump = 22 := by
  sorry

end jump_difference_l1144_114423


namespace cost_of_paint_per_quart_l1144_114499

/-- Tommy has a flag that is 5 feet wide and 4 feet tall. 
He needs to paint both sides of the flag. 
A quart of paint covers 4 square feet. 
He spends $20 on paint. 
Prove that the cost of paint per quart is $2. --/
theorem cost_of_paint_per_quart
  (width height : ℕ) (paint_area_per_quart : ℕ) (total_cost : ℕ) (total_area : ℕ) (quarts_needed : ℕ) :
  width = 5 →
  height = 4 →
  paint_area_per_quart = 4 →
  total_cost = 20 →
  total_area = 2 * (width * height) →
  quarts_needed = total_area / paint_area_per_quart →
  total_cost / quarts_needed = 2 := 
by
  intros h_w h_h h_papq h_tc h_ta h_qn
  sorry

end cost_of_paint_per_quart_l1144_114499


namespace find_m_l1144_114471

/-- 
If the function y=x + m/(x-1) defined for x > 1 attains its minimum value at x = 3,
then the positive number m is 4.
-/
theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x -> x + m / (x - 1) ≥ 3 + m / 2):
  m = 4 :=
sorry

end find_m_l1144_114471


namespace g_of_50_eq_zero_l1144_114447

theorem g_of_50_eq_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - 3 * y * g x = g (x / y)) : g 50 = 0 :=
sorry

end g_of_50_eq_zero_l1144_114447


namespace volume_correctness_l1144_114456

noncomputable def volume_of_regular_triangular_pyramid (d : ℝ) : ℝ :=
  1/3 * d^2 * d * Real.sqrt 2

theorem volume_correctness (d : ℝ) : 
  volume_of_regular_triangular_pyramid d = 1/3 * d^3 * Real.sqrt 2 :=
by
  sorry

end volume_correctness_l1144_114456


namespace lcm_48_75_l1144_114404

theorem lcm_48_75 : Nat.lcm 48 75 = 1200 := by
  sorry

end lcm_48_75_l1144_114404


namespace pell_infinite_solutions_l1144_114415

theorem pell_infinite_solutions : ∃ m : ℕ, ∃ a b c : ℕ, 
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (∀ n : ℕ, ∃ an bn cn : ℕ, 
    (1 / an + 1 / bn + 1 / cn + 1 / (an * bn * cn) = m / (an + bn + cn))) := 
sorry

end pell_infinite_solutions_l1144_114415


namespace probability_one_defective_l1144_114430

theorem probability_one_defective (g d : Nat) (h1 : g = 3) (h2 : d = 1) : 
  let total_combinations := (g + d).choose 2
  let favorable_outcomes := g * d
  favorable_outcomes / total_combinations = 1 / 2 := by
sorry

end probability_one_defective_l1144_114430


namespace wheel_radius_increase_l1144_114479

theorem wheel_radius_increase :
  let r := 18
  let distance_AB := 600   -- distance from A to B in miles
  let distance_BA := 582   -- distance from B to A in miles
  let circumference_orig := 2 * Real.pi * r
  let dist_per_rotation_orig := circumference_orig / 63360
  let rotations_orig := distance_AB / dist_per_rotation_orig
  let r' := ((distance_BA * dist_per_rotation_orig * 63360) / (2 * Real.pi * rotations_orig))
  ((r' - r) : ℝ) = 0.34 := by
  sorry

end wheel_radius_increase_l1144_114479


namespace sum_of_coefficients_l1144_114440

noncomputable def expand_and_sum_coefficients (d : ℝ) : ℝ :=
  let poly := -2 * (4 - d) * (d + 3 * (4 - d))
  let expanded := -4 * d^2 + 40 * d - 96
  let sum_coefficients := (-4) + 40 + (-96)
  sum_coefficients

theorem sum_of_coefficients (d : ℝ) : expand_and_sum_coefficients d = -60 := by
  sorry

end sum_of_coefficients_l1144_114440


namespace water_for_bathing_per_horse_per_day_l1144_114431

-- Definitions of the given conditions
def initial_horses : ℕ := 3
def additional_horses : ℕ := 5
def total_horses : ℕ := initial_horses + additional_horses
def drink_water_per_horse_per_day : ℕ := 5
def total_days : ℕ := 28
def total_water_needed : ℕ := 1568

-- The proven statement
theorem water_for_bathing_per_horse_per_day :
  ((total_water_needed - (total_horses * drink_water_per_horse_per_day * total_days)) / (total_horses * total_days)) = 2 :=
by
  sorry

end water_for_bathing_per_horse_per_day_l1144_114431


namespace oranges_and_apples_l1144_114497

theorem oranges_and_apples (O A : ℕ) (h₁ : 7 * O = 5 * A) (h₂ : O = 28) : A = 20 :=
by {
  sorry
}

end oranges_and_apples_l1144_114497


namespace sqrt_9_is_rational_l1144_114418

theorem sqrt_9_is_rational : ∃ q : ℚ, (q : ℝ) = 3 := by
  sorry

end sqrt_9_is_rational_l1144_114418


namespace polygon_has_area_144_l1144_114472

noncomputable def polygonArea (n_sides : ℕ) (perimeter : ℕ) (n_squares : ℕ) : ℕ :=
  let s := perimeter / n_sides
  let square_area := s * s
  square_area * n_squares

theorem polygon_has_area_144 :
  polygonArea 32 64 36 = 144 :=
by
  sorry

end polygon_has_area_144_l1144_114472


namespace strawberry_quality_meets_standard_l1144_114435

def acceptable_weight_range (w : ℝ) : Prop :=
  4.97 ≤ w ∧ w ≤ 5.03

theorem strawberry_quality_meets_standard :
  acceptable_weight_range 4.98 :=
by
  sorry

end strawberry_quality_meets_standard_l1144_114435


namespace find_number_l1144_114495

-- Define the conditions
def number_times_x_eq_165 (number x : ℕ) : Prop :=
  number * x = 165

def x_eq_11 (x : ℕ) : Prop :=
  x = 11

-- The proof problem statement
theorem find_number (number x : ℕ) (h1 : number_times_x_eq_165 number x) (h2 : x_eq_11 x) : number = 15 :=
by
  sorry

end find_number_l1144_114495


namespace correct_option_c_l1144_114493

-- Definitions for the problem context
noncomputable def qualification_rate : ℝ := 0.99
noncomputable def picking_probability := qualification_rate

-- The theorem statement that needs to be proven
theorem correct_option_c : picking_probability = 0.99 :=
sorry

end correct_option_c_l1144_114493


namespace percentage_of_students_in_band_l1144_114428

theorem percentage_of_students_in_band 
  (students_in_band : ℕ)
  (total_students : ℕ)
  (students_in_band_eq : students_in_band = 168)
  (total_students_eq : total_students = 840) :
  (students_in_band / total_students : ℚ) * 100 = 20 :=
by
  sorry

end percentage_of_students_in_band_l1144_114428


namespace pants_price_100_l1144_114462

-- Define the variables and conditions
variables (x y : ℕ)

-- Define the prices according to the conditions
def coat_price_pants := x + 340
def coat_price_shoes_pants := y + x + 180
def total_price := (coat_price_pants x) + x + y

-- The theorem to prove
theorem pants_price_100 (h1: coat_price_pants x = coat_price_shoes_pants x y) (h2: total_price x y = 700) : x = 100 :=
sorry

end pants_price_100_l1144_114462


namespace find_cubic_expression_l1144_114422

theorem find_cubic_expression (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end find_cubic_expression_l1144_114422


namespace consecutive_integers_no_two_l1144_114401

theorem consecutive_integers_no_two (a n : ℕ) : 
  ¬(∃ (b : ℤ), (b : ℤ) = 2) :=
sorry

end consecutive_integers_no_two_l1144_114401
