import Mathlib

namespace NUMINAMATH_GPT_cannot_sum_85_with_five_coins_l844_84489

def coin_value (c : Nat) : Prop :=
  c = 1 ∨ c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50

theorem cannot_sum_85_with_five_coins : 
  ¬ ∃ (a b c d e : Nat), 
    coin_value a ∧ 
    coin_value b ∧ 
    coin_value c ∧ 
    coin_value d ∧ 
    coin_value e ∧ 
    a + b + c + d + e = 85 :=
by
  sorry

end NUMINAMATH_GPT_cannot_sum_85_with_five_coins_l844_84489


namespace NUMINAMATH_GPT_sum_remainder_l844_84464

theorem sum_remainder (n : ℤ) : ((9 - n) + (n + 4)) % 9 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_sum_remainder_l844_84464


namespace NUMINAMATH_GPT_angle_A_is_pi_over_3_l844_84461

theorem angle_A_is_pi_over_3 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C)
  (h2 : a ^ 2 = b ^ 2 + c ^ 2 - bc * (2 * Real.cos A))
  (triangle_ABC : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ A + B + C = π) :
  A = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_A_is_pi_over_3_l844_84461


namespace NUMINAMATH_GPT_ratio_of_hypotenuse_segments_l844_84499

theorem ratio_of_hypotenuse_segments (a b c d : ℝ) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : b = (3/4) * a)
  (h3 : d^2 = (c - d)^2 + b^2) :
  (d / (c - d)) = (4 / 3) :=
sorry

end NUMINAMATH_GPT_ratio_of_hypotenuse_segments_l844_84499


namespace NUMINAMATH_GPT_cos_seven_theta_l844_84498

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end NUMINAMATH_GPT_cos_seven_theta_l844_84498


namespace NUMINAMATH_GPT_solve_inequality_l844_84400

theorem solve_inequality (x : ℝ) : 2 * x + 4 > 0 ↔ x > -2 := sorry

end NUMINAMATH_GPT_solve_inequality_l844_84400


namespace NUMINAMATH_GPT_donut_cubes_eaten_l844_84476

def cube_dimensions := 5

def total_cubes_in_cube : ℕ := cube_dimensions ^ 3

def even_neighbors (faces_sharing_cubes : ℕ) : Prop :=
  faces_sharing_cubes % 2 = 0

/-- A corner cube in a 5x5x5 cube has 3 neighbors. --/
def corner_cube_neighbors := 3

/-- An edge cube in a 5x5x5 cube (excluding corners) has 4 neighbors. --/
def edge_cube_neighbors := 4

/-- A face center cube in a 5x5x5 cube has 5 neighbors. --/
def face_center_cube_neighbors := 5

/-- An inner cube in a 5x5x5 cube has 6 neighbors. --/
def inner_cube_neighbors := 6

/-- Count of edge cubes that share 4 neighbors in a 5x5x5 cube. --/
def edge_cubes_count := 12 * (cube_dimensions - 2)

def inner_cubes_count := (cube_dimensions - 2) ^ 3

theorem donut_cubes_eaten :
  (edge_cubes_count + inner_cubes_count) = 63 := by
  sorry

end NUMINAMATH_GPT_donut_cubes_eaten_l844_84476


namespace NUMINAMATH_GPT_find_number_l844_84475

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l844_84475


namespace NUMINAMATH_GPT_ben_examined_7_trays_l844_84434

open Int

def trays_of_eggs (total_eggs : ℕ) (eggs_per_tray : ℕ) : ℕ := total_eggs / eggs_per_tray

theorem ben_examined_7_trays : trays_of_eggs 70 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_ben_examined_7_trays_l844_84434


namespace NUMINAMATH_GPT_average_hit_targets_value_average_hit_targets_ge_half_l844_84483

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end NUMINAMATH_GPT_average_hit_targets_value_average_hit_targets_ge_half_l844_84483


namespace NUMINAMATH_GPT_fraction_of_menu_items_my_friend_can_eat_l844_84463

theorem fraction_of_menu_items_my_friend_can_eat {menu_size vegan_dishes nut_free_vegan_dishes : ℕ}
    (h1 : vegan_dishes = 6)
    (h2 : vegan_dishes = menu_size / 6)
    (h3 : nut_free_vegan_dishes = vegan_dishes - 5) :
    (nut_free_vegan_dishes : ℚ) / menu_size = 1 / 36 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_menu_items_my_friend_can_eat_l844_84463


namespace NUMINAMATH_GPT_simplify_expression_l844_84426

variable (x y : ℝ)

theorem simplify_expression (A B : ℝ) (hA : A = x^2) (hB : B = y^2) :
  (A + B) / (A - B) + (A - B) / (A + B) = 2 * (x^4 + y^4) / (x^4 - y^4) :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l844_84426


namespace NUMINAMATH_GPT_billy_weight_l844_84414

variable (B Bd C D : ℝ)

theorem billy_weight
  (h1 : B = Bd + 9)
  (h2 : Bd = C + 5)
  (h3 : C = D - 8)
  (h4 : C = 145)
  (h5 : D = 2 * Bd) :
  B = 85.5 :=
by
  sorry

end NUMINAMATH_GPT_billy_weight_l844_84414


namespace NUMINAMATH_GPT_increasing_iff_range_a_three_distinct_real_roots_l844_84442

noncomputable def f (a x : ℝ) : ℝ :=
  if x >= 2 * a then x^2 + (2 - 2 * a) * x else - x^2 + (2 + 2 * a) * x

theorem increasing_iff_range_a (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem three_distinct_real_roots (a t : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2)
  (h_roots : ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧
                           f a x₁ = t * f a (2 * a) ∧
                           f a x₂ = t * f a (2 * a) ∧
                           f a x₃ = t * f a (2 * a)) :
  1 < t ∧ t < 9 / 8 :=
sorry

end NUMINAMATH_GPT_increasing_iff_range_a_three_distinct_real_roots_l844_84442


namespace NUMINAMATH_GPT_arithmetic_mean_of_integers_from_neg3_to_6_l844_84479

def integer_range := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

noncomputable def arithmetic_mean : ℚ :=
  (integer_range.sum : ℚ) / (integer_range.length : ℚ)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  arithmetic_mean = 1.5 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_integers_from_neg3_to_6_l844_84479


namespace NUMINAMATH_GPT_find_range_of_m_l844_84430

noncomputable def p (m : ℝ) : Prop := 1 - Real.sqrt 2 < m ∧ m < 1 + Real.sqrt 2
noncomputable def q (m : ℝ) : Prop := 0 < m ∧ m < 4

theorem find_range_of_m (m : ℝ) (hpq : p m ∨ q m) (hnp : ¬ p m) : 1 + Real.sqrt 2 ≤ m ∧ m < 4 :=
sorry

end NUMINAMATH_GPT_find_range_of_m_l844_84430


namespace NUMINAMATH_GPT_union_M_N_eq_l844_84481

open Set

-- Define M according to the condition x^2 < 15 for x in ℕ
def M : Set ℕ := {x | x^2 < 15}

-- Define N according to the correct answer
def N : Set ℕ := {x | 0 < x ∧ x < 5}

-- Prove that M ∪ N = {x | 0 ≤ x ∧ x < 5}
theorem union_M_N_eq : M ∪ N = {x : ℕ | 0 ≤ x ∧ x < 5} :=
sorry

end NUMINAMATH_GPT_union_M_N_eq_l844_84481


namespace NUMINAMATH_GPT_luca_lost_more_weight_l844_84401

theorem luca_lost_more_weight (barbi_kg_month : ℝ) (luca_kg_year : ℝ) (months_in_year : ℕ) (years : ℕ) 
(h_barbi : barbi_kg_month = 1.5) (h_luca : luca_kg_year = 9) (h_months_in_year : months_in_year = 12) (h_years : years = 11) : 
  (luca_kg_year * years) - (barbi_kg_month * months_in_year * (years / 11)) = 81 := 
by 
  sorry

end NUMINAMATH_GPT_luca_lost_more_weight_l844_84401


namespace NUMINAMATH_GPT_number_of_students_in_class_l844_84410

theorem number_of_students_in_class :
  ∃ n : ℕ, n > 0 ∧ (∀ avg_age teacher_age total_avg_age, avg_age = 26 ∧ teacher_age = 52 ∧ total_avg_age = 27 →
    (∃ total_student_age total_age_with_teacher, 
      total_student_age = n * avg_age ∧ 
      total_age_with_teacher = total_student_age + teacher_age ∧ 
      (total_age_with_teacher / (n + 1) = total_avg_age) → n = 25)) :=
sorry

end NUMINAMATH_GPT_number_of_students_in_class_l844_84410


namespace NUMINAMATH_GPT_tap_filling_time_l844_84452

theorem tap_filling_time
  (T : ℝ)
  (H1 : 10 > 0) -- Second tap can empty the cistern in 10 hours
  (H2 : T > 0)  -- First tap's time must be positive
  (H3 : (1 / T) - (1 / 10) = (3 / 20))  -- Both taps together fill the cistern in 6.666... hours
  : T = 4 := sorry

end NUMINAMATH_GPT_tap_filling_time_l844_84452


namespace NUMINAMATH_GPT_unit_prices_min_chess_sets_l844_84416

-- Define the conditions and prove the unit prices.
theorem unit_prices (x y : ℝ) 
  (h1 : 6 * x + 5 * y = 190)
  (h2 : 8 * x + 10 * y = 320) : 
  x = 15 ∧ y = 20 :=
by
  sorry

-- Define the conditions for the budget and prove the minimum number of chess sets.
theorem min_chess_sets (x y : ℝ) (m : ℕ)
  (hx : x = 15)
  (hy : y = 20)
  (number_sets : m + (100 - m) = 100)
  (budget : 15 * ↑m + 20 * ↑(100 - m) ≤ 1800) :
  m ≥ 40 :=
by
  sorry

end NUMINAMATH_GPT_unit_prices_min_chess_sets_l844_84416


namespace NUMINAMATH_GPT_sufficient_condition_x_gt_2_l844_84413

theorem sufficient_condition_x_gt_2 (x : ℝ) (h : x > 2) : x^2 - 2 * x > 0 := by
  sorry

end NUMINAMATH_GPT_sufficient_condition_x_gt_2_l844_84413


namespace NUMINAMATH_GPT_new_ratio_is_three_half_l844_84484

theorem new_ratio_is_three_half (F J : ℕ) (h1 : F * 4 = J * 5) (h2 : J = 120) :
  ((F + 30) : ℚ) / J = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_new_ratio_is_three_half_l844_84484


namespace NUMINAMATH_GPT_simplify_fraction_l844_84417

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l844_84417


namespace NUMINAMATH_GPT_kelseys_sisters_age_l844_84467

theorem kelseys_sisters_age :
  ∀ (current_year : ℕ) (kelsey_birth_year : ℕ)
    (kelsey_sister_birth_year : ℕ),
    kelsey_birth_year = 1999 - 25 →
    kelsey_sister_birth_year = kelsey_birth_year - 3 →
    current_year = 2021 →
    current_year - kelsey_sister_birth_year = 50 :=
by
  intros current_year kelsey_birth_year kelsey_sister_birth_year h1 h2 h3
  sorry

end NUMINAMATH_GPT_kelseys_sisters_age_l844_84467


namespace NUMINAMATH_GPT_min_value_S_l844_84471

theorem min_value_S (a b c : ℤ) (h1 : a + b + c = 2) (h2 : (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) > 200) :
  ∃ a b c : ℤ, a + b + c = 2 ∧ (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) = 256 :=
sorry

end NUMINAMATH_GPT_min_value_S_l844_84471


namespace NUMINAMATH_GPT_side_length_of_square_base_l844_84440

theorem side_length_of_square_base (area : ℝ) (slant_height : ℝ) (s : ℝ) (h : slant_height = 40) (a : area = 160) : s = 8 :=
by sorry

end NUMINAMATH_GPT_side_length_of_square_base_l844_84440


namespace NUMINAMATH_GPT_perfect_square_form_l844_84491

theorem perfect_square_form (N : ℕ) (hN : 0 < N) : 
  ∃ x : ℤ, 2^N - 2 * (N : ℤ) = x^2 ↔ N = 1 ∨ N = 2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_form_l844_84491


namespace NUMINAMATH_GPT_part1_part2_l844_84447

-- Definitions of sets A and B
def A (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < a + 1 }
def B : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 ≥ 0 }

-- Proving the first condition
theorem part1 (a : ℝ) : (A a ∩ B = ∅) ∧ (A a ∪ B = Set.univ) ↔ a = 2 :=
by
  sorry

-- Proving the second condition
theorem part2 (a : ℝ) : (A a ⊆ B) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l844_84447


namespace NUMINAMATH_GPT_certain_number_of_tenths_l844_84487

theorem certain_number_of_tenths (n : ℝ) (h : n = 375 * (1/10)) : n = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_of_tenths_l844_84487


namespace NUMINAMATH_GPT_max_vec_diff_magnitude_l844_84497

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ := (1, sin θ)
noncomputable def vec_b (θ : ℝ) : ℝ × ℝ := (1, cos θ)

noncomputable def vec_diff_magnitude (θ : ℝ) : ℝ :=
  let a := vec_a θ
  let b := vec_b θ
  abs ((a.1 - b.1)^2 + (a.2 - b.2)^2)^(1/2)

theorem max_vec_diff_magnitude : ∀ θ : ℝ, vec_diff_magnitude θ ≤ sqrt 2 :=
by
  intro θ
  sorry

end NUMINAMATH_GPT_max_vec_diff_magnitude_l844_84497


namespace NUMINAMATH_GPT_pyramid_height_l844_84486

noncomputable def height_pyramid (perimeter_base : ℝ) (distance_apex_vertex : ℝ) : ℝ :=
  let side_length := perimeter_base / 4
  let half_diagonal := (side_length * Real.sqrt 2) / 2
  Real.sqrt (distance_apex_vertex ^ 2 - half_diagonal ^ 2)

theorem pyramid_height
  (perimeter_base: ℝ)
  (h_perimeter : perimeter_base = 32)
  (distance_apex_vertex: ℝ)
  (h_distance : distance_apex_vertex = 10) :
  height_pyramid perimeter_base distance_apex_vertex = 2 * Real.sqrt 17 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_height_l844_84486


namespace NUMINAMATH_GPT_option_d_is_pythagorean_triple_l844_84411

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem option_d_is_pythagorean_triple : is_pythagorean_triple 5 12 13 :=
by
  -- This will be the proof part, which is omitted as per the problem's instructions.
  sorry

end NUMINAMATH_GPT_option_d_is_pythagorean_triple_l844_84411


namespace NUMINAMATH_GPT_complex_multiplication_l844_84477

-- Definition of the imaginary unit
def is_imaginary_unit (i : ℂ) : Prop := i * i = -1

theorem complex_multiplication (i : ℂ) (h : is_imaginary_unit i) : (1 + i) * (1 - i) = 2 :=
by
  -- Given that i is the imaginary unit satisfying i^2 = -1
  -- We need to show that (1 + i) * (1 - i) = 2
  sorry

end NUMINAMATH_GPT_complex_multiplication_l844_84477


namespace NUMINAMATH_GPT_number_in_tenth_group_l844_84403

-- Number of students
def students : ℕ := 1000

-- Number of groups
def groups : ℕ := 100

-- Interval between groups
def interval : ℕ := students / groups

-- First number drawn
def first_number : ℕ := 6

-- Number drawn from n-th group given first_number and interval
def number_in_group (n : ℕ) : ℕ := first_number + interval * (n - 1)

-- Statement to prove
theorem number_in_tenth_group :
  number_in_group 10 = 96 :=
by
  sorry

end NUMINAMATH_GPT_number_in_tenth_group_l844_84403


namespace NUMINAMATH_GPT_mean_of_other_two_numbers_l844_84462

theorem mean_of_other_two_numbers (a b c d e f g h : ℕ)
  (h_tuple : a = 1871 ∧ b = 2011 ∧ c = 2059 ∧ d = 2084 ∧ e = 2113 ∧ f = 2167 ∧ g = 2198 ∧ h = 2210)
  (h_mean : (a + b + c + d + e + f) / 6 = 2100) :
  ((g + h) / 2 : ℚ) = 2056.5 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_other_two_numbers_l844_84462


namespace NUMINAMATH_GPT_minimum_bailing_rate_l844_84409

theorem minimum_bailing_rate
  (distance_from_shore : Real := 1.5)
  (rowing_speed : Real := 3)
  (water_intake_rate : Real := 12)
  (max_water : Real := 45) :
  (distance_from_shore / rowing_speed) * 60 * water_intake_rate - max_water / ((distance_from_shore / rowing_speed) * 60) >= 10.5 :=
by
  -- Provide the units are consistent and the calculations agree with the given numerical data
  sorry

end NUMINAMATH_GPT_minimum_bailing_rate_l844_84409


namespace NUMINAMATH_GPT_no_first_or_fourth_quadrant_l844_84456

theorem no_first_or_fourth_quadrant (a b : ℝ) (h : a * b > 0) : 
  ¬ ((∃ x, a * x + b = 0 ∧ x > 0) ∧ (∃ x, b * x + a = 0 ∧ x > 0)) 
  ∧ ¬ ((∃ x, a * x + b = 0 ∧ x < 0) ∧ (∃ x, b * x + a = 0 ∧ x < 0)) := sorry

end NUMINAMATH_GPT_no_first_or_fourth_quadrant_l844_84456


namespace NUMINAMATH_GPT_find_quaterns_l844_84433

theorem find_quaterns {
  x y z w : ℝ
} : 
  (x + y = z^2 + w^2 + 6 * z * w) → 
  (x + z = y^2 + w^2 + 6 * y * w) → 
  (x + w = y^2 + z^2 + 6 * y * z) → 
  (y + z = x^2 + w^2 + 6 * x * w) → 
  (y + w = x^2 + z^2 + 6 * x * z) → 
  (z + w = x^2 + y^2 + 6 * x * y) → 
  ( (x, y, z, w) = (0, 0, 0, 0) 
    ∨ (x, y, z, w) = (1/4, 1/4, 1/4, 1/4) 
    ∨ (x, y, z, w) = (-1/4, -1/4, 3/4, -1/4) 
    ∨ (x, y, z, w) = (-1/2, -1/2, 5/2, -1/2)
  ) :=
  sorry

end NUMINAMATH_GPT_find_quaterns_l844_84433


namespace NUMINAMATH_GPT_fraction_of_b_equals_4_15_of_a_is_0_4_l844_84422

variable (A B : ℤ)
variable (X : ℚ)

def a_and_b_together_have_1210 : Prop := A + B = 1210
def b_has_484 : Prop := B = 484
def fraction_of_b_equals_4_15_of_a : Prop := (4 / 15 : ℚ) * A = X * B

theorem fraction_of_b_equals_4_15_of_a_is_0_4
  (h1 : a_and_b_together_have_1210 A B)
  (h2 : b_has_484 B)
  (h3 : fraction_of_b_equals_4_15_of_a A B X) :
  X = 0.4 := sorry

end NUMINAMATH_GPT_fraction_of_b_equals_4_15_of_a_is_0_4_l844_84422


namespace NUMINAMATH_GPT_number_of_distinct_sentences_l844_84448

noncomputable def count_distinct_sentences (phrase : String) : Nat :=
  let I_options := 3 -- absent, partially present, fully present
  let II_options := 2 -- absent, present
  let IV_options := 2 -- incomplete or absent
  let III_mandatory := 1 -- always present
  (III_mandatory * IV_options * I_options * II_options) - 1 -- subtract the original sentence

theorem number_of_distinct_sentences :
  count_distinct_sentences "ранним утром на рыбалку улыбающийся Игорь мчался босиком" = 23 :=
by
  sorry

end NUMINAMATH_GPT_number_of_distinct_sentences_l844_84448


namespace NUMINAMATH_GPT_faster_train_passes_slower_in_54_seconds_l844_84469

-- Definitions of the conditions.
def length_of_train := 75 -- Length of each train in meters.
def speed_faster_train := 46 * 1000 / 3600 -- Speed of the faster train in m/s.
def speed_slower_train := 36 * 1000 / 3600 -- Speed of the slower train in m/s.
def relative_speed := speed_faster_train - speed_slower_train -- Relative speed in m/s.
def total_distance := 2 * length_of_train -- Total distance to cover to pass the slower train.

-- The proof statement.
theorem faster_train_passes_slower_in_54_seconds : total_distance / relative_speed = 54 := by
  sorry

end NUMINAMATH_GPT_faster_train_passes_slower_in_54_seconds_l844_84469


namespace NUMINAMATH_GPT_tenth_day_of_month_is_monday_l844_84438

theorem tenth_day_of_month_is_monday (Sundays_on_even_dates : ℕ → Prop)
  (h1: Sundays_on_even_dates 2)
  (h2: Sundays_on_even_dates 16)
  (h3: Sundays_on_even_dates 30) :
  ∃ k : ℕ, 10 = k + 2 + 7 * 1 ∧ k.succ.succ.succ.succ.succ.succ.succ.succ.succ.succ = 1 :=
by sorry

end NUMINAMATH_GPT_tenth_day_of_month_is_monday_l844_84438


namespace NUMINAMATH_GPT_train_speed_l844_84429

/--A train leaves Delhi at 9 a.m. at a speed of 30 kmph.
Another train leaves at 3 p.m. on the same day and in the same direction.
The two trains meet 720 km away from Delhi.
Prove that the speed of the second train is 120 kmph.-/
theorem train_speed
  (speed_first_train speed_first_kmph : 30 = 30)
  (leave_first_train : Nat)
  (leave_first_9am : 9 = 9)
  (leave_second_train : Nat)
  (leave_second_3pm : 3 = 3)
  (distance_meeting_km : Nat)
  (distance_meeting_720km : 720 = 720) :
  ∃ speed_second_train, speed_second_train = 120 := 
sorry

end NUMINAMATH_GPT_train_speed_l844_84429


namespace NUMINAMATH_GPT_calculate_revolutions_l844_84488

noncomputable def number_of_revolutions (diameter distance: ℝ) : ℝ :=
  distance / (Real.pi * diameter)

theorem calculate_revolutions :
  number_of_revolutions 10 5280 = 528 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_calculate_revolutions_l844_84488


namespace NUMINAMATH_GPT_profit_difference_l844_84415

-- Define the initial capitals of A, B, and C
def capital_A := 8000
def capital_B := 10000
def capital_C := 12000

-- Define B's profit share
def profit_share_B := 3500

-- Define the total number of parts
def total_parts := 15

-- Define the number of parts for each person
def parts_A := 4
def parts_B := 5
def parts_C := 6

-- Define the total profit
noncomputable def total_profit := profit_share_B * (total_parts / parts_B)

-- Define the profit shares of A and C
noncomputable def profit_share_A := (parts_A / total_parts) * total_profit
noncomputable def profit_share_C := (parts_C / total_parts) * total_profit

-- Define the difference between the profit shares of A and C
noncomputable def profit_share_difference := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_difference :
  profit_share_difference = 1400 := by
  sorry

end NUMINAMATH_GPT_profit_difference_l844_84415


namespace NUMINAMATH_GPT_rhombus_has_perpendicular_diagonals_and_rectangle_not_l844_84412

-- Definitions based on conditions (a))
def rhombus (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_perpendicular : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_perpendicular

def rectangle (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_equal : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_equal

-- Theorem to prove (c))
theorem rhombus_has_perpendicular_diagonals_and_rectangle_not 
  (rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular : Prop)
  (rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal : Prop) :
  rhombus rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular → 
  rectangle rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal → 
  rhombus_diagonals_perpendicular ∧ ¬(rectangle (rectangle_sides_equal) (rectangle_diagonals_bisect) (rhombus_diagonals_perpendicular)) :=
sorry

end NUMINAMATH_GPT_rhombus_has_perpendicular_diagonals_and_rectangle_not_l844_84412


namespace NUMINAMATH_GPT_sequence_all_integers_l844_84439

open Nat

def a : ℕ → ℤ
| 0 => 1
| 1 => 1
| n+2 => (a (n+1))^2 + 2 / a n

theorem sequence_all_integers :
  ∀ n : ℕ, ∃ k : ℤ, a n = k :=
by
  sorry

end NUMINAMATH_GPT_sequence_all_integers_l844_84439


namespace NUMINAMATH_GPT_number_minus_45_l844_84427

theorem number_minus_45 (x : ℕ) (h1 : (x / 2) / 2 = 85 + 45) : x - 45 = 475 := by
  sorry

end NUMINAMATH_GPT_number_minus_45_l844_84427


namespace NUMINAMATH_GPT_average_speed_bike_l844_84423

theorem average_speed_bike (t_goal : ℚ) (d_swim r_swim : ℚ) (d_run r_run : ℚ) (d_bike r_bike : ℚ) :
  t_goal = 1.75 →
  d_swim = 1 / 3 ∧ r_swim = 1.5 →
  d_run = 2.5 ∧ r_run = 8 →
  d_bike = 12 →
  r_bike = 1728 / 175 :=
by
  intros h_goal h_swim h_run h_bike
  sorry

end NUMINAMATH_GPT_average_speed_bike_l844_84423


namespace NUMINAMATH_GPT_find_box_value_l844_84465

theorem find_box_value (r x : ℕ) 
  (h1 : x + r = 75)
  (h2 : (x + r) + 2 * r = 143) : 
  x = 41 := 
by
  sorry

end NUMINAMATH_GPT_find_box_value_l844_84465


namespace NUMINAMATH_GPT_recruits_total_l844_84459

theorem recruits_total (x y z : ℕ) (total_people : ℕ) :
  (x = total_people - 51) ∧
  (y = total_people - 101) ∧
  (z = total_people - 171) ∧
  (x = 4 * y ∨ y = 4 * z ∨ x = 4 * z) ∧
  (∃ total_people, total_people = 211) :=
sorry

end NUMINAMATH_GPT_recruits_total_l844_84459


namespace NUMINAMATH_GPT_quadratic_solution_condition_sufficient_but_not_necessary_l844_84480

theorem quadratic_solution_condition_sufficient_but_not_necessary (m : ℝ) :
  (m < -2) → (∃ x : ℝ, x^2 + m * x + 1 = 0) ∧ ¬(∀ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0 → m < -2) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_solution_condition_sufficient_but_not_necessary_l844_84480


namespace NUMINAMATH_GPT_complex_number_coordinates_l844_84421

-- Define i as the imaginary unit
def i := Complex.I

-- State the theorem
theorem complex_number_coordinates : (i * (1 - i)).re = 1 ∧ (i * (1 - i)).im = 1 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_complex_number_coordinates_l844_84421


namespace NUMINAMATH_GPT_remainder_eq_six_l844_84492

theorem remainder_eq_six
  (Dividend : ℕ) (Divisor : ℕ) (Quotient : ℕ) (Remainder : ℕ)
  (h1 : Dividend = 139)
  (h2 : Divisor = 19)
  (h3 : Quotient = 7)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Remainder = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_eq_six_l844_84492


namespace NUMINAMATH_GPT_gcd_lcm_mul_l844_84453

theorem gcd_lcm_mul (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := 
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_mul_l844_84453


namespace NUMINAMATH_GPT_solution_set_contains_0_and_2_l844_84472

theorem solution_set_contains_0_and_2 (k : ℝ) : 
  ∀ x, ((1 + k^2) * x ≤ k^4 + 4) → (x = 0 ∨ x = 2) :=
by {
  sorry -- Proof is omitted
}

end NUMINAMATH_GPT_solution_set_contains_0_and_2_l844_84472


namespace NUMINAMATH_GPT_cost_of_each_box_of_pencils_l844_84404

-- Definitions based on conditions
def cartons_of_pencils : ℕ := 20
def boxes_per_carton_of_pencils : ℕ := 10
def cartons_of_markers : ℕ := 10
def boxes_per_carton_of_markers : ℕ := 5
def cost_per_carton_of_markers : ℕ := 4
def total_spent : ℕ := 600

-- Variable to define cost per box of pencils
variable (P : ℝ)

-- Main theorem to prove
theorem cost_of_each_box_of_pencils :
  cartons_of_pencils * boxes_per_carton_of_pencils * P + 
  cartons_of_markers * cost_per_carton_of_markers = total_spent → 
  P = 2.80 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_box_of_pencils_l844_84404


namespace NUMINAMATH_GPT_students_not_picked_l844_84494

theorem students_not_picked (total_students groups group_size : ℕ) (h1 : total_students = 64)
(h2 : groups = 4) (h3 : group_size = 7) :
total_students - groups * group_size = 36 :=
by
  sorry

end NUMINAMATH_GPT_students_not_picked_l844_84494


namespace NUMINAMATH_GPT_surface_area_invisible_block_l844_84478

-- Define the given areas of the seven blocks
def A1 := 148
def A2 := 46
def A3 := 72
def A4 := 28
def A5 := 88
def A6 := 126
def A7 := 58

-- Define total surface areas of the black and white blocks
def S_black := A1 + A2 + A3 + A4
def S_white := A5 + A6 + A7

-- Define the proof problem
theorem surface_area_invisible_block : S_black - S_white = 22 :=
by
  -- This sorry allows the Lean statement to build successfully
  sorry

end NUMINAMATH_GPT_surface_area_invisible_block_l844_84478


namespace NUMINAMATH_GPT_sam_current_yellow_marbles_l844_84436

theorem sam_current_yellow_marbles (original_yellow : ℕ) (taken_yellow : ℕ) (current_yellow : ℕ) :
  original_yellow = 86 → 
  taken_yellow = 25 → 
  current_yellow = original_yellow - taken_yellow → 
  current_yellow = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_sam_current_yellow_marbles_l844_84436


namespace NUMINAMATH_GPT_solve_ineq_l844_84443

noncomputable def inequality (x : ℝ) : Prop :=
  (x^2 / (x+1)) ≥ (3 / (x+1) + 3)

theorem solve_ineq :
  { x : ℝ | inequality x } = { x : ℝ | x ≤ -6 ∨ (-1 < x ∧ x ≤ 3) } := sorry

end NUMINAMATH_GPT_solve_ineq_l844_84443


namespace NUMINAMATH_GPT_value_of_a1_plus_a3_l844_84405

theorem value_of_a1_plus_a3 (a a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, (1 + x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) →
  a1 + a3 = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a1_plus_a3_l844_84405


namespace NUMINAMATH_GPT_solve_adult_tickets_l844_84419

theorem solve_adult_tickets (A C : ℕ) (h1 : 8 * A + 5 * C = 236) (h2 : A + C = 34) : A = 22 :=
sorry

end NUMINAMATH_GPT_solve_adult_tickets_l844_84419


namespace NUMINAMATH_GPT_smallest_n_for_abc_factorials_l844_84450

theorem smallest_n_for_abc_factorials (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b + c = 2006) :
  ∃ m n : ℕ, (¬ ∃ k : ℕ, m = 10 * k) ∧ a.factorial * b.factorial * c.factorial = m * 10^n ∧ n = 492 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_abc_factorials_l844_84450


namespace NUMINAMATH_GPT_new_machine_rate_l844_84457

def old_machine_rate : ℕ := 100
def total_bolts : ℕ := 500
def time_hours : ℕ := 2

theorem new_machine_rate (R : ℕ) : 
  (old_machine_rate * time_hours + R * time_hours = total_bolts) → 
  R = 150 := 
by
  sorry

end NUMINAMATH_GPT_new_machine_rate_l844_84457


namespace NUMINAMATH_GPT_christina_speed_limit_l844_84408

theorem christina_speed_limit :
  ∀ (D total_distance friend_distance : ℝ), 
  total_distance = 210 → 
  friend_distance = 3 * 40 → 
  D = total_distance - friend_distance → 
  D / 3 = 30 :=
by
  intros D total_distance friend_distance 
  intros h1 h2 h3 
  sorry

end NUMINAMATH_GPT_christina_speed_limit_l844_84408


namespace NUMINAMATH_GPT_rectangle_area_l844_84431

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 56) :
  l * b = 147 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l844_84431


namespace NUMINAMATH_GPT_roots_square_sum_l844_84406

theorem roots_square_sum (r s p q : ℝ) 
  (root_cond : ∀ x : ℝ, x^2 - 2 * p * x + 3 * q = 0 → (x = r ∨ x = s)) :
  r^2 + s^2 = 4 * p^2 - 6 * q :=
by
  sorry

end NUMINAMATH_GPT_roots_square_sum_l844_84406


namespace NUMINAMATH_GPT_combinatorial_identity_l844_84470

theorem combinatorial_identity :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 :=
sorry

end NUMINAMATH_GPT_combinatorial_identity_l844_84470


namespace NUMINAMATH_GPT_sum_of_first_13_terms_is_39_l844_84444

-- Definition of arithmetic sequence and the given condition
def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

-- Given condition
axiom given_condition {a : ℕ → ℤ} (h : arithmetic_sequence a) : a 5 + a 6 + a 7 = 9

-- The main theorem
theorem sum_of_first_13_terms_is_39 {a : ℕ → ℤ} (h : arithmetic_sequence a) (h9 : a 5 + a 6 + a 7 = 9) : sum_of_first_n_terms a 12 = 39 :=
sorry

end NUMINAMATH_GPT_sum_of_first_13_terms_is_39_l844_84444


namespace NUMINAMATH_GPT_smallest_n_for_polygon_cutting_l844_84482

theorem smallest_n_for_polygon_cutting : 
  ∃ n : ℕ, (∃ k : ℕ, n - 2 = k * 31) ∧ (∃ k' : ℕ, n - 2 = k' * 65) ∧ n = 2017 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_polygon_cutting_l844_84482


namespace NUMINAMATH_GPT_medium_ceiling_lights_count_l844_84460

theorem medium_ceiling_lights_count (S M L : ℕ) 
  (h1 : L = 2 * M) 
  (h2 : S = M + 10) 
  (h_bulbs : S + 2 * M + 3 * L = 118) : M = 12 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_medium_ceiling_lights_count_l844_84460


namespace NUMINAMATH_GPT_correct_completion_of_sentence_l844_84418

def committee_discussing_problem : Prop := True -- Placeholder for the condition
def problem_expected_to_be_solved_next_week : Prop := True -- Placeholder for the condition

theorem correct_completion_of_sentence 
  (h1 : committee_discussing_problem) 
  (h2 : problem_expected_to_be_solved_next_week) 
  : "hopefully" = "hopefully" :=
by 
  sorry

end NUMINAMATH_GPT_correct_completion_of_sentence_l844_84418


namespace NUMINAMATH_GPT_ratio_area_III_IV_l844_84473

theorem ratio_area_III_IV 
  (perimeter_I : ℤ)
  (perimeter_II : ℤ)
  (perimeter_IV : ℤ)
  (side_III_is_three_times_side_I : ℤ)
  (h1 : perimeter_I = 16)
  (h2 : perimeter_II = 20)
  (h3 : perimeter_IV = 32)
  (h4 : side_III_is_three_times_side_I = 3 * (perimeter_I / 4)) :
  (3 * (perimeter_I / 4))^2 / (perimeter_IV / 4)^2 = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_area_III_IV_l844_84473


namespace NUMINAMATH_GPT_part1_part2_l844_84490

open Real

def f (x a : ℝ) := abs (x + 2 * a) + abs (x - 1)

section part1

variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

end part1

section part2

noncomputable def g (a : ℝ) := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part2 {a : ℝ} (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end part2

end NUMINAMATH_GPT_part1_part2_l844_84490


namespace NUMINAMATH_GPT_minute_hand_distance_l844_84445

noncomputable def distance_traveled (length_of_minute_hand : ℝ) (time_duration : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * length_of_minute_hand
  let revolutions := time_duration / 60
  circumference * revolutions

theorem minute_hand_distance :
  distance_traveled 8 45 = 12 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_minute_hand_distance_l844_84445


namespace NUMINAMATH_GPT_smallest_k_for_positive_roots_5_l844_84402

noncomputable def smallest_k_for_positive_roots : ℕ := 5

theorem smallest_k_for_positive_roots_5
  (k p q : ℕ) 
  (hk : k = smallest_k_for_positive_roots)
  (hq_pos : 0 < q)
  (h_distinct_pos_roots : ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
    k * x₁ * x₂ = q ∧ k * x₁ + k * x₂ > p ∧ k * x₁ * x₂ < q * ( 1 / (x₁*(1 - x₁) * x₂ * (1 - x₂)))) :
  k = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_for_positive_roots_5_l844_84402


namespace NUMINAMATH_GPT_most_balls_l844_84449

def soccerballs : ℕ := 50
def basketballs : ℕ := 26
def baseballs : ℕ := basketballs + 8

theorem most_balls :
  max (max soccerballs basketballs) baseballs = soccerballs := by
  sorry

end NUMINAMATH_GPT_most_balls_l844_84449


namespace NUMINAMATH_GPT_find_a12_l844_84466

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- The Lean statement for the problem
theorem find_a12 (h_seq : arithmetic_sequence a d)
  (h_cond1 : a 7 + a 9 = 16) (h_cond2 : a 4 = 1) : 
  a 12 = 15 :=
sorry

end NUMINAMATH_GPT_find_a12_l844_84466


namespace NUMINAMATH_GPT_difference_of_digits_is_six_l844_84454

theorem difference_of_digits_is_six (a b : ℕ) (h_sum : a + b = 10) (h_number : 10 * a + b = 82) : a - b = 6 :=
sorry

end NUMINAMATH_GPT_difference_of_digits_is_six_l844_84454


namespace NUMINAMATH_GPT_additional_girls_needed_l844_84446

theorem additional_girls_needed (initial_girls initial_boys additional_girls : ℕ)
  (h_initial_girls : initial_girls = 2)
  (h_initial_boys : initial_boys = 6)
  (h_fraction_goal : (initial_girls + additional_girls) = (5 * (initial_girls + initial_boys + additional_girls)) / 8) :
  additional_girls = 8 :=
by
  -- A placeholder for the proof
  sorry

end NUMINAMATH_GPT_additional_girls_needed_l844_84446


namespace NUMINAMATH_GPT_GregPPO_reward_correct_l844_84435

-- Define the maximum ProcGen reward
def maxProcGenReward : ℕ := 240

-- Define the maximum CoinRun reward in the more challenging version
def maxCoinRunReward : ℕ := maxProcGenReward / 2

-- Define the percentage reward obtained by Greg's PPO algorithm
def percentageRewardObtained : ℝ := 0.9

-- Calculate the reward obtained by Greg's PPO algorithm
def rewardGregPPO : ℝ := percentageRewardObtained * maxCoinRunReward

-- The theorem to prove the correct answer
theorem GregPPO_reward_correct : rewardGregPPO = 108 := by
  sorry

end NUMINAMATH_GPT_GregPPO_reward_correct_l844_84435


namespace NUMINAMATH_GPT_condition_of_A_with_respect_to_D_l844_84458

variables {A B C D : Prop}

theorem condition_of_A_with_respect_to_D (h1 : A → B) (h2 : ¬ (B → A)) (h3 : B ↔ C) (h4 : C → D) (h5 : ¬ (D → C)) :
  (D → A) ∧ ¬ (A → D) :=
by
  sorry

end NUMINAMATH_GPT_condition_of_A_with_respect_to_D_l844_84458


namespace NUMINAMATH_GPT_seventh_triangular_number_eq_28_l844_84441

noncomputable def triangular_number (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem seventh_triangular_number_eq_28 :
  triangular_number 7 = 28 :=
by
  sorry

end NUMINAMATH_GPT_seventh_triangular_number_eq_28_l844_84441


namespace NUMINAMATH_GPT_speed_conversion_l844_84493

theorem speed_conversion (speed_mps : ℝ) (conversion_factor : ℝ) (speed_kmph_expected : ℝ) :
  speed_mps = 35.0028 →
  conversion_factor = 3.6 →
  speed_kmph_expected = 126.01008 →
  speed_mps * conversion_factor = speed_kmph_expected :=
by
  intros h_mps h_cf h_kmph
  rw [h_mps, h_cf, h_kmph]
  sorry

end NUMINAMATH_GPT_speed_conversion_l844_84493


namespace NUMINAMATH_GPT_calculate_lower_profit_percentage_l844_84424

theorem calculate_lower_profit_percentage 
  (CP : ℕ) 
  (profitAt18Percent : ℕ) 
  (additionalProfit : ℕ)
  (hCP : CP = 800) 
  (hProfitAt18Percent : profitAt18Percent = 144) 
  (hAdditionalProfit : additionalProfit = 72) 
  (hProfitRelation : profitAt18Percent = additionalProfit + ((9 * CP) / 100)) :
  9 = ((9 * CP) / 100) :=
by
  sorry

end NUMINAMATH_GPT_calculate_lower_profit_percentage_l844_84424


namespace NUMINAMATH_GPT_a₁₀_greater_than_500_l844_84485

variables (a : ℕ → ℕ) (b : ℕ → ℕ)

-- Conditions
def strictly_increasing (a : ℕ → ℕ) : Prop := ∀ n, a n < a (n + 1)

def largest_divisor (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ n, b n < a n ∧ ∃ d > 1, d ∣ a n ∧ b n = a n / d

def greater_sequence (b : ℕ → ℕ) : Prop := ∀ n, b n > b (n + 1)

-- Statement to prove
theorem a₁₀_greater_than_500
  (h1 : strictly_increasing a)
  (h2 : largest_divisor a b)
  (h3 : greater_sequence b) :
  a 10 > 500 :=
sorry

end NUMINAMATH_GPT_a₁₀_greater_than_500_l844_84485


namespace NUMINAMATH_GPT_paint_liters_needed_l844_84495

theorem paint_liters_needed :
  let cost_brushes : ℕ := 20
  let cost_canvas : ℕ := 3 * cost_brushes
  let cost_paint_per_liter : ℕ := 8
  let total_costs : ℕ := 120
  ∃ (liters_of_paint : ℕ), cost_brushes + cost_canvas + cost_paint_per_liter * liters_of_paint = total_costs ∧ liters_of_paint = 5 :=
by
  sorry

end NUMINAMATH_GPT_paint_liters_needed_l844_84495


namespace NUMINAMATH_GPT_determine_y_l844_84451

theorem determine_y : 
  ∀ y : ℝ, 
    (2 * Real.arctan (1 / 5) + Real.arctan (1 / 25) + Real.arctan (1 / y) = Real.pi / 4) -> 
    y = -121 / 60 :=
by
  sorry

end NUMINAMATH_GPT_determine_y_l844_84451


namespace NUMINAMATH_GPT_not_all_ten_on_boundary_of_same_square_l844_84420

open Function

variable (points : Fin 10 → ℝ × ℝ)

def four_points_on_square (A B C D : ℝ × ℝ) : Prop :=
  -- Define your own predicate to check if 4 points A, B, C, D are on the boundary of some square
  sorry 

theorem not_all_ten_on_boundary_of_same_square :
  (∀ A B C D : Fin 10, four_points_on_square (points A) (points B) (points C) (points D)) →
  ¬ (∃ square : ℝ × ℝ → Prop, ∀ i : Fin 10, square (points i)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_not_all_ten_on_boundary_of_same_square_l844_84420


namespace NUMINAMATH_GPT_S_not_eq_T_l844_84432

def S := {x : ℤ | ∃ n : ℤ, x = 2 * n}
def T := {x : ℤ | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}

theorem S_not_eq_T : S ≠ T := by
  sorry

end NUMINAMATH_GPT_S_not_eq_T_l844_84432


namespace NUMINAMATH_GPT_david_marks_in_biology_l844_84428

theorem david_marks_in_biology (english: ℕ) (math: ℕ) (physics: ℕ) (chemistry: ℕ) (average: ℕ) (biology: ℕ) :
  english = 81 ∧ math = 65 ∧ physics = 82 ∧ chemistry = 67 ∧ average = 76 → (biology = 85) :=
by
  sorry

end NUMINAMATH_GPT_david_marks_in_biology_l844_84428


namespace NUMINAMATH_GPT_fraction_raised_to_zero_l844_84455

theorem fraction_raised_to_zero:
  (↑(-4305835) / ↑1092370457 : ℚ)^0 = 1 := 
by
  sorry

end NUMINAMATH_GPT_fraction_raised_to_zero_l844_84455


namespace NUMINAMATH_GPT_problem_expression_value_l844_84425

theorem problem_expression_value :
  (100 - (3010 - 301)) + (3010 - (301 - 100)) = 200 :=
by
  sorry

end NUMINAMATH_GPT_problem_expression_value_l844_84425


namespace NUMINAMATH_GPT_combined_weight_of_daughter_and_child_l844_84474

variables (M D C : ℝ)
axiom mother_daughter_grandchild_weight : M + D + C = 120
axiom daughter_weight : D = 48
axiom child_weight_fraction_of_grandmother : C = (1 / 5) * M

theorem combined_weight_of_daughter_and_child : D + C = 60 :=
  sorry

end NUMINAMATH_GPT_combined_weight_of_daughter_and_child_l844_84474


namespace NUMINAMATH_GPT_intersection_of_diagonals_l844_84496

-- Define the four lines based on the given conditions
def line1 (k b x : ℝ) : ℝ := k*x + b
def line2 (k b x : ℝ) : ℝ := k*x - b
def line3 (m b x : ℝ) : ℝ := m*x + b
def line4 (m b x : ℝ) : ℝ := m*x - b

-- Define a function to represent the problem
noncomputable def point_of_intersection_of_diagonals (k m b : ℝ) : ℝ × ℝ :=
(0, 0)

-- State the theorem to be proved
theorem intersection_of_diagonals (k m b : ℝ) :
  point_of_intersection_of_diagonals k m b = (0, 0) :=
sorry

end NUMINAMATH_GPT_intersection_of_diagonals_l844_84496


namespace NUMINAMATH_GPT_toys_per_rabbit_l844_84437

theorem toys_per_rabbit 
  (rabbits toys_mon toys_wed toys_fri toys_sat : ℕ) 
  (hrabbits : rabbits = 16) 
  (htoys_mon : toys_mon = 6)
  (htoys_wed : toys_wed = 2 * toys_mon)
  (htoys_fri : toys_fri = 4 * toys_mon)
  (htoys_sat : toys_sat = toys_wed / 2) :
  (toys_mon + toys_wed + toys_fri + toys_sat) / rabbits = 3 :=
by 
  sorry

end NUMINAMATH_GPT_toys_per_rabbit_l844_84437


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l844_84407

-- Definition for question 1:
def gcd_21n_4_14n_3 (n : ℕ) : Prop := (Nat.gcd (21 * n + 4) (14 * n + 3)) = 1

-- Definition for question 2:
def gcd_n_factorial_plus_1 (n : ℕ) : Prop := (Nat.gcd (Nat.factorial n + 1) (Nat.factorial (n + 1) + 1)) = 1

-- Definition for question 3:
def fermat_number (k : ℕ) : ℕ := 2^(2^k) + 1
def gcd_fermat_numbers (m n : ℕ) (h : m ≠ n) : Prop := (Nat.gcd (fermat_number m) (fermat_number n)) = 1

-- Theorem statements
theorem problem_1 (n : ℕ) (h_pos : 0 < n) : gcd_21n_4_14n_3 n := sorry

theorem problem_2 (n : ℕ) (h_pos : 0 < n) : gcd_n_factorial_plus_1 n := sorry

theorem problem_3 (m n : ℕ) (h_pos1 : 0 ≠ m) (h_pos2 : 0 ≠ n) (h_neq : m ≠ n) : gcd_fermat_numbers m n h_neq := sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l844_84407


namespace NUMINAMATH_GPT_rain_at_least_one_day_probability_l844_84468

-- Definitions based on given conditions
def P_rain_Friday : ℝ := 0.30
def P_rain_Monday : ℝ := 0.20

-- Events probabilities based on independence
def P_no_rain_Friday := 1 - P_rain_Friday
def P_no_rain_Monday := 1 - P_rain_Monday
def P_no_rain_both := P_no_rain_Friday * P_no_rain_Monday

-- The probability of raining at least one day
def P_rain_at_least_one_day := 1 - P_no_rain_both

-- Expected probability
def expected_probability : ℝ := 0.44

theorem rain_at_least_one_day_probability : 
  P_rain_at_least_one_day = expected_probability := by
  sorry

end NUMINAMATH_GPT_rain_at_least_one_day_probability_l844_84468
