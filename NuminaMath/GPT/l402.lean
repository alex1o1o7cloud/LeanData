import Mathlib

namespace NUMINAMATH_GPT_divide_composite_products_l402_40252

theorem divide_composite_products :
  let first_three := [4, 6, 8]
  let next_three := [9, 10, 12]
  let prod_first_three := first_three.prod
  let prod_next_three := next_three.prod
  (prod_first_three : ℚ) / prod_next_three = 8 / 45 :=
by
  sorry

end NUMINAMATH_GPT_divide_composite_products_l402_40252


namespace NUMINAMATH_GPT_find_biology_marks_l402_40220

variables (e m p c b : ℕ)
variable (a : ℝ)

def david_marks_in_biology : Prop :=
  e = 72 ∧
  m = 45 ∧
  p = 72 ∧
  c = 77 ∧
  a = 68.2 ∧
  (e + m + p + c + b) / 5 = a

theorem find_biology_marks (h : david_marks_in_biology e m p c b a) : b = 75 :=
sorry

end NUMINAMATH_GPT_find_biology_marks_l402_40220


namespace NUMINAMATH_GPT_distance_relation_possible_l402_40218

-- Define a structure representing points in 2D space
structure Point where
  x : ℤ
  y : ℤ

-- Define the artificial geometry distance function (Euclidean distance)
def varrho (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

-- Define the non-collinearity condition for points A, B, and C
def non_collinear (A B C : Point) : Prop :=
  ¬(A.x = B.x ∧ B.x = C.x) ∧ ¬(A.y = B.y ∧ B.y = C.y)

theorem distance_relation_possible :
  ∃ (A B C : Point), non_collinear A B C ∧ varrho A C ^ 2 + varrho B C ^ 2 = varrho A B ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_relation_possible_l402_40218


namespace NUMINAMATH_GPT_milkman_profit_percentage_l402_40232

noncomputable def profit_percentage (x : ℝ) : ℝ :=
  let cp_per_litre := x
  let sp_per_litre := 2 * x
  let mixture_litres := 8
  let milk_litres := 6
  let cost_price := milk_litres * cp_per_litre
  let selling_price := mixture_litres * sp_per_litre
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

theorem milkman_profit_percentage (x : ℝ) 
  (h : x > 0) : 
  profit_percentage x = 166.67 :=
by
  sorry

end NUMINAMATH_GPT_milkman_profit_percentage_l402_40232


namespace NUMINAMATH_GPT_find_angle_C_find_sin_A_plus_sin_B_l402_40292

open Real

namespace TriangleProblem

variables (a b c : ℝ) (A B C : ℝ)

def sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  c^2 = a^2 + b^2 + a * b

def given_c (c : ℝ) : Prop :=
  c = 4 * sqrt 7

def perimeter (a b c : ℝ) : Prop :=
  a + b + c = 12 + 4 * sqrt 7

theorem find_angle_C (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C) : 
  C = 2 * pi / 3 :=
sorry

theorem find_sin_A_plus_sin_B (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C)
  (h2 : given_c c)
  (h3 : perimeter a b c) : 
  sin A + sin B = 3 * sqrt 21 / 28 :=
sorry

end TriangleProblem

end NUMINAMATH_GPT_find_angle_C_find_sin_A_plus_sin_B_l402_40292


namespace NUMINAMATH_GPT_husband_weekly_saving_l402_40294

variable (H : ℕ)

-- conditions
def weekly_wife : ℕ := 225
def months : ℕ := 6
def weeks_per_month : ℕ := 4
def weeks := months * weeks_per_month
def amount_per_child : ℕ := 1680
def num_children : ℕ := 4

-- total savings calculation
def total_saving : ℕ := weeks * H + weeks * weekly_wife

-- half of total savings divided among children
def half_savings_div_by_children : ℕ := num_children * amount_per_child

-- proof statement
theorem husband_weekly_saving : H = 335 :=
by
  let total_children_saving := half_savings_div_by_children
  have half_saving : ℕ := total_children_saving 
  have total_saving_eq : total_saving = 2 * total_children_saving := sorry
  have total_saving_eq_simplified : weeks * H + weeks * weekly_wife = 13440 := sorry
  have H_eq : H = 335 := sorry
  exact H_eq

end NUMINAMATH_GPT_husband_weekly_saving_l402_40294


namespace NUMINAMATH_GPT_find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l402_40273

noncomputable def board : Type := (Fin 5) × (Fin 5)

def is_counterfeit (c1 : board) (c2 : board) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 + 1 = c2.2)) ∨
  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 + 1 = c2.1))

theorem find_13_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 13 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem find_15_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 15 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem cannot_find_17_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ¬ (∃ C : Finset board, C.card = 17 ∧ ∀ c ∈ C, coins c = coins (0,0)) :=
sorry

end NUMINAMATH_GPT_find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l402_40273


namespace NUMINAMATH_GPT_bakery_problem_l402_40226

theorem bakery_problem :
  let chocolate_chip := 154
  let oatmeal_raisin := 86
  let sugar := 52
  let capacity := 16
  let needed_chocolate_chip := capacity - (chocolate_chip % capacity)
  let needed_oatmeal_raisin := capacity - (oatmeal_raisin % capacity)
  let needed_sugar := capacity - (sugar % capacity)
  (needed_chocolate_chip = 6) ∧ (needed_oatmeal_raisin = 10) ∧ (needed_sugar = 12) :=
by
  sorry

end NUMINAMATH_GPT_bakery_problem_l402_40226


namespace NUMINAMATH_GPT_problem_π_digit_sequence_l402_40258

def f (n : ℕ) : ℕ :=
  match n with
  | 1  => 1
  | 2  => 4
  | 3  => 1
  | 4  => 5
  | 5  => 9
  | 6  => 2
  | 7  => 6
  | 8  => 5
  | 9  => 3
  | 10 => 5
  | _  => 0  -- for simplicity we define other cases arbitrarily

theorem problem_π_digit_sequence :
  ∃ n : ℕ, n > 0 ∧ f (f (f (f (f 10)))) = 1 := by
  sorry

end NUMINAMATH_GPT_problem_π_digit_sequence_l402_40258


namespace NUMINAMATH_GPT_acute_angle_sum_l402_40237

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = (2 * Real.sqrt 5) / 5) (h2 : Real.sin β = (3 * Real.sqrt 10) / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_acute_angle_sum_l402_40237


namespace NUMINAMATH_GPT_line_equation_l402_40270

theorem line_equation {k b : ℝ} 
  (h1 : (∀ x : ℝ, k * x + b = -4 * x + 2023 → k = -4))
  (h2 : b = -5) :
  ∀ x : ℝ, k * x + b = -4 * x - 5 := by
sorry

end NUMINAMATH_GPT_line_equation_l402_40270


namespace NUMINAMATH_GPT_shelves_used_l402_40254

-- Definitions from conditions
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Theorem statement
theorem shelves_used : (initial_bears + shipment_bears) / bears_per_shelf = 4 := by
  sorry

end NUMINAMATH_GPT_shelves_used_l402_40254


namespace NUMINAMATH_GPT_lowest_score_within_two_std_devs_l402_40284

variable (mean : ℝ) (std_dev : ℝ) (jack_score : ℝ)

def within_two_std_devs (mean : ℝ) (std_dev : ℝ) (score : ℝ) : Prop :=
  score >= mean - 2 * std_dev

theorem lowest_score_within_two_std_devs :
  mean = 60 → std_dev = 10 → within_two_std_devs mean std_dev jack_score → (40 ≤ jack_score) :=
by
  intros h1 h2 h3
  change mean = 60 at h1
  change std_dev = 10 at h2
  sorry

end NUMINAMATH_GPT_lowest_score_within_two_std_devs_l402_40284


namespace NUMINAMATH_GPT_seating_5_out_of_6_around_circle_l402_40279

def number_of_ways_to_seat_5_out_of_6_in_circle : Nat :=
  Nat.factorial 4

theorem seating_5_out_of_6_around_circle : number_of_ways_to_seat_5_out_of_6_in_circle = 24 :=
by {
  -- proof would be here
  sorry
}

end NUMINAMATH_GPT_seating_5_out_of_6_around_circle_l402_40279


namespace NUMINAMATH_GPT_center_distance_correct_l402_40287

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R₁ : ℝ := 150
noncomputable def R₂ : ℝ := 50
noncomputable def R₃ : ℝ := 90
noncomputable def R₄ : ℝ := 120
noncomputable def elevation : ℝ := 4

noncomputable def adjusted_R₁ : ℝ := R₁ - ball_radius
noncomputable def adjusted_R₂ : ℝ := R₂ + ball_radius + elevation
noncomputable def adjusted_R₃ : ℝ := R₃ - ball_radius
noncomputable def adjusted_R₄ : ℝ := R₄ - ball_radius

noncomputable def distance_R₁ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₁
noncomputable def distance_R₂ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₂
noncomputable def distance_R₃ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₃
noncomputable def distance_R₄ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₄

noncomputable def total_distance : ℝ := distance_R₁ + distance_R₂ + distance_R₃ + distance_R₄

theorem center_distance_correct : total_distance = 408 * Real.pi := 
  by
  sorry

end NUMINAMATH_GPT_center_distance_correct_l402_40287


namespace NUMINAMATH_GPT_grandma_Olga_grandchildren_l402_40297

def daughters : Nat := 3
def sons : Nat := 3
def sons_per_daughter : Nat := 6
def daughters_per_son : Nat := 5

theorem grandma_Olga_grandchildren : 
  (daughters * sons_per_daughter) + (sons * daughters_per_son) = 33 := by
  sorry

end NUMINAMATH_GPT_grandma_Olga_grandchildren_l402_40297


namespace NUMINAMATH_GPT_determine_values_of_a_and_b_l402_40290

def ab_product_eq_one (a b : ℝ) : Prop := a * b = 1

def given_equation (a b : ℝ) : Prop :=
  (a + b + 2) / 4 = (1 / (a + 1)) + (1 / (b + 1))

theorem determine_values_of_a_and_b (a b : ℝ) (h1 : ab_product_eq_one a b) (h2 : given_equation a b) :
  a = 1 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_values_of_a_and_b_l402_40290


namespace NUMINAMATH_GPT_remainder_zero_division_l402_40215

theorem remainder_zero_division :
  ∀ x : ℂ, (x^2 - x + 1 = 0) →
    ((x^5 + x^4 - x^3 - x^2 + 1) * (x^3 - 1)) % (x^2 - x + 1) = 0 :=
by sorry

end NUMINAMATH_GPT_remainder_zero_division_l402_40215


namespace NUMINAMATH_GPT_students_in_class_l402_40204

theorem students_in_class (S : ℕ) (h1 : S / 3 + 2 * S / 5 + 12 = S) : S = 45 :=
sorry

end NUMINAMATH_GPT_students_in_class_l402_40204


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l402_40271

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x < 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l402_40271


namespace NUMINAMATH_GPT_total_savings_l402_40239

-- Define the given conditions
def number_of_tires : ℕ := 4
def sale_price : ℕ := 75
def original_price : ℕ := 84

-- State the proof problem
theorem total_savings : (original_price - sale_price) * number_of_tires = 36 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_savings_l402_40239


namespace NUMINAMATH_GPT_infinitely_many_squares_of_form_l402_40259

theorem infinitely_many_squares_of_form (k : ℕ) (hk : 0 < k) : 
  ∃ (n : ℕ), ∀ m : ℕ, ∃ n' > n, 2 * k * n' - 7 = m^2 :=
sorry

end NUMINAMATH_GPT_infinitely_many_squares_of_form_l402_40259


namespace NUMINAMATH_GPT_problem_statement_l402_40280

theorem problem_statement : (1021 ^ 1022) % 1023 = 4 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l402_40280


namespace NUMINAMATH_GPT_jason_cutting_grass_time_l402_40260

-- Conditions
def time_to_cut_one_lawn : ℕ := 30 -- in minutes
def lawns_cut_each_day : ℕ := 8
def days : ℕ := 2
def minutes_in_an_hour : ℕ := 60

-- Proof that the number of hours Jason spends cutting grass over the weekend is 8
theorem jason_cutting_grass_time:
  ((lawns_cut_each_day * days) * time_to_cut_one_lawn) / minutes_in_an_hour = 8 :=
by
  sorry

end NUMINAMATH_GPT_jason_cutting_grass_time_l402_40260


namespace NUMINAMATH_GPT_fraction_of_cracked_pots_is_2_over_5_l402_40283

-- Definitions for the problem conditions
def total_pots : ℕ := 80
def price_per_pot : ℕ := 40
def total_revenue : ℕ := 1920

-- Statement to prove the fraction of cracked pots
theorem fraction_of_cracked_pots_is_2_over_5 
  (C : ℕ) 
  (h1 : (total_pots - C) * price_per_pot = total_revenue) : 
  C / total_pots = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_cracked_pots_is_2_over_5_l402_40283


namespace NUMINAMATH_GPT_son_time_to_complete_work_l402_40222

noncomputable def man_work_rate : ℚ := 1 / 6
noncomputable def combined_work_rate : ℚ := 1 / 3

theorem son_time_to_complete_work :
  (1 / (combined_work_rate - man_work_rate)) = 6 := by
  sorry

end NUMINAMATH_GPT_son_time_to_complete_work_l402_40222


namespace NUMINAMATH_GPT_sum_of_coeffs_eq_92_l402_40276

noncomputable def sum_of_integer_coeffs_in_factorization (x y : ℝ) : ℝ :=
  let f := 27 * (x ^ 6) - 512 * (y ^ 6)
  3 - 8 + 9 + 24 + 64  -- Sum of integer coefficients

theorem sum_of_coeffs_eq_92 (x y : ℝ) : sum_of_integer_coeffs_in_factorization x y = 92 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_eq_92_l402_40276


namespace NUMINAMATH_GPT_sum_smallest_largest_l402_40289

theorem sum_smallest_largest (n a : ℕ) (h_even_n : n % 2 = 0) (y x : ℕ)
  (h_y : y = a + n - 1)
  (h_x : x = (a + 3 * (n / 3 - 1)) * (n / 3)) : 
  2 * y = a + (a + 2 * (n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_sum_smallest_largest_l402_40289


namespace NUMINAMATH_GPT_rectangular_to_cylindrical_l402_40288

theorem rectangular_to_cylindrical (x y z : ℝ) (r θ : ℝ) (h1 : x = -3) (h2 : y = 4) (h3 : z = 5) (h4 : r = 5) (h5 : θ = Real.pi - Real.arctan (4 / 3)) :
  (r, θ, z) = (5, Real.pi - Real.arctan (4 / 3), 5) :=
by
  sorry

end NUMINAMATH_GPT_rectangular_to_cylindrical_l402_40288


namespace NUMINAMATH_GPT_length_of_first_train_is_correct_l402_40231

noncomputable def length_of_first_train 
  (speed_first_train_kmph : ℝ)
  (length_second_train_m : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_crossing_s : ℝ) : ℝ :=
  let speed_first_train_mps := (speed_first_train_kmph * 1000) / 3600
  let speed_second_train_mps := (speed_second_train_kmph * 1000) / 3600
  let relative_speed_mps := speed_first_train_mps + speed_second_train_mps
  let total_distance_m := relative_speed_mps * time_crossing_s
  total_distance_m - length_second_train_m

theorem length_of_first_train_is_correct :
  length_of_first_train 50 112 82 6 = 108.02 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_train_is_correct_l402_40231


namespace NUMINAMATH_GPT_count_squares_with_center_55_25_l402_40201

noncomputable def number_of_squares_with_natural_number_coordinates : ℕ :=
  600

theorem count_squares_with_center_55_25 :
  ∀ (x y : ℕ), (x = 55) ∧ (y = 25) → number_of_squares_with_natural_number_coordinates = 600 :=
by
  intros x y h
  cases h
  sorry

end NUMINAMATH_GPT_count_squares_with_center_55_25_l402_40201


namespace NUMINAMATH_GPT_cube_edge_factor_l402_40208

theorem cube_edge_factor (e f : ℝ) (h₁ : e > 0) (h₂ : (f * e) ^ 3 = 8 * e ^ 3) : f = 2 :=
by
  sorry

end NUMINAMATH_GPT_cube_edge_factor_l402_40208


namespace NUMINAMATH_GPT_age_of_twin_brothers_l402_40205

theorem age_of_twin_brothers (x : Nat) : (x + 1) * (x + 1) = x * x + 11 ↔ x = 5 :=
by
  sorry  -- Proof omitted.

end NUMINAMATH_GPT_age_of_twin_brothers_l402_40205


namespace NUMINAMATH_GPT_multiplication_pattern_correct_l402_40233

theorem multiplication_pattern_correct :
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  123456 * 9 + 7 = 1111111 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_pattern_correct_l402_40233


namespace NUMINAMATH_GPT_Keenan_essay_length_l402_40241

-- Given conditions
def words_per_hour_first_two_hours : ℕ := 400
def first_two_hours : ℕ := 2
def words_per_hour_later : ℕ := 200
def later_hours : ℕ := 2

-- Total words written in 4 hours
def total_words : ℕ := words_per_hour_first_two_hours * first_two_hours + words_per_hour_later * later_hours

-- Theorem statement
theorem Keenan_essay_length : total_words = 1200 := by
  sorry

end NUMINAMATH_GPT_Keenan_essay_length_l402_40241


namespace NUMINAMATH_GPT_seven_segments_impossible_l402_40213

theorem seven_segments_impossible :
  ¬(∃(segments : Fin 7 → Set (Fin 7)), (∀i, ∃ (S : Finset (Fin 7)), S.card = 3 ∧ ∀ j ∈ S, i ≠ j ∧ segments i j) ∧ (∀ i j, i ≠ j → segments i j → segments j i)) :=
sorry

end NUMINAMATH_GPT_seven_segments_impossible_l402_40213


namespace NUMINAMATH_GPT_checkerboard_probability_not_on_perimeter_l402_40268

def total_squares : ℕ := 81

def perimeter_squares : ℕ := 32

def non_perimeter_squares : ℕ := total_squares - perimeter_squares

noncomputable def probability_not_on_perimeter : ℚ := non_perimeter_squares / total_squares

theorem checkerboard_probability_not_on_perimeter :
  probability_not_on_perimeter = 49 / 81 :=
by
  sorry

end NUMINAMATH_GPT_checkerboard_probability_not_on_perimeter_l402_40268


namespace NUMINAMATH_GPT_nested_fraction_simplifies_l402_40265

theorem nested_fraction_simplifies : 
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 := 
by 
  sorry

end NUMINAMATH_GPT_nested_fraction_simplifies_l402_40265


namespace NUMINAMATH_GPT_total_weight_tommy_ordered_l402_40282

theorem total_weight_tommy_ordered :
  let apples := 3
  let oranges := 1
  let grapes := 3
  let strawberries := 3
  apples + oranges + grapes + strawberries = 10 := by
  sorry

end NUMINAMATH_GPT_total_weight_tommy_ordered_l402_40282


namespace NUMINAMATH_GPT_max_diameters_l402_40210

theorem max_diameters (n : ℕ) (points : Finset (ℝ × ℝ)) (h : n ≥ 3) (hn : points.card = n)
  (d : ℝ) (h_d_max : ∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q ≤ d) :
  ∃ m : ℕ, m ≤ n ∧ (∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q = d → m ≤ n) := 
sorry

end NUMINAMATH_GPT_max_diameters_l402_40210


namespace NUMINAMATH_GPT_range_of_a_l402_40203

noncomputable def f (x a : ℝ) := Real.log x + 1 / 2 * x^2 + a * x

theorem range_of_a
  (a : ℝ)
  (h : ∃ x : ℝ, x > 0 ∧ (1/x + x + a = 3)) :
  a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l402_40203


namespace NUMINAMATH_GPT_players_taking_physics_l402_40278

-- Definitions based on the conditions
def total_players : ℕ := 30
def players_taking_math : ℕ := 15
def players_taking_both : ℕ := 6

-- The main theorem to prove
theorem players_taking_physics : total_players - players_taking_math + players_taking_both = 21 := by
  sorry

end NUMINAMATH_GPT_players_taking_physics_l402_40278


namespace NUMINAMATH_GPT_peanut_butter_revenue_l402_40272

theorem peanut_butter_revenue :
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  plantation_length * plantation_width * peanuts_per_sqft * butter_from_peanuts_ratio / 1000 * butter_price_per_kg = 31250 := 
by
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  sorry

end NUMINAMATH_GPT_peanut_butter_revenue_l402_40272


namespace NUMINAMATH_GPT_second_player_cannot_prevent_first_l402_40275

noncomputable def player_choice (set_x2_coeff_to_zero : Prop) (first_player_sets : Prop) (second_player_cannot_prevent : Prop) : Prop :=
  ∀ (b : ℝ) (c : ℝ), (set_x2_coeff_to_zero ∧ first_player_sets ∧ second_player_cannot_prevent) → 
  (∀ x : ℝ, x^3 + b * x + c = 0 → ∃! x : ℝ, x^3 + b * x + c = 0)

theorem second_player_cannot_prevent_first (b c : ℝ) :
  player_choice (set_x2_coeff_to_zero := true)
                (first_player_sets := true)
                (second_player_cannot_prevent := true) :=
sorry

end NUMINAMATH_GPT_second_player_cannot_prevent_first_l402_40275


namespace NUMINAMATH_GPT_felipe_total_time_l402_40240

-- Given definitions
def combined_time_without_breaks := 126
def combined_time_with_breaks := 150
def felipe_break := 6
def emilio_break := 2 * felipe_break
def carlos_break := emilio_break / 2

theorem felipe_total_time (F E C : ℕ) 
(h1 : F = E / 2) 
(h2 : C = F + E)
(h3 : (F + E + C) = combined_time_without_breaks)
(h4 : (F + felipe_break) + (E + emilio_break) + (C + carlos_break) = combined_time_with_breaks) : 
F + felipe_break = 27 := 
sorry

end NUMINAMATH_GPT_felipe_total_time_l402_40240


namespace NUMINAMATH_GPT_student_second_subject_percentage_l402_40216

theorem student_second_subject_percentage (x : ℝ) (h : (50 + x + 90) / 3 = 70) : x = 70 :=
by { sorry }

end NUMINAMATH_GPT_student_second_subject_percentage_l402_40216


namespace NUMINAMATH_GPT_exists_horizontal_chord_l402_40274

theorem exists_horizontal_chord (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_eq : f 0 = f 1) : ∃ n : ℕ, n ≥ 1 ∧ ∃ x : ℝ, 0 ≤ x ∧ x + 1/n ≤ 1 ∧ f x = f (x + 1/n) :=
by
  sorry

end NUMINAMATH_GPT_exists_horizontal_chord_l402_40274


namespace NUMINAMATH_GPT_find_value_at_frac_one_third_l402_40225

theorem find_value_at_frac_one_third
  (f : ℝ → ℝ) 
  (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = 1 / 4) :
  f (1 / 3) = 9 := 
  sorry

end NUMINAMATH_GPT_find_value_at_frac_one_third_l402_40225


namespace NUMINAMATH_GPT_line_and_circle_separate_l402_40257

theorem line_and_circle_separate
  (θ : ℝ) (hθ : ¬ ∃ k : ℤ, θ = k * Real.pi) :
  ¬ ∃ (x y : ℝ), (x^2 + y^2 = 1 / 2) ∧ (x * Real.cos θ + y - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_and_circle_separate_l402_40257


namespace NUMINAMATH_GPT_decreased_cost_l402_40238

theorem decreased_cost (original_cost : ℝ) (decrease_percentage : ℝ) (h1 : original_cost = 200) (h2 : decrease_percentage = 0.50) : 
  (original_cost - original_cost * decrease_percentage) = 100 :=
by
  -- This is the proof placeholder
  sorry

end NUMINAMATH_GPT_decreased_cost_l402_40238


namespace NUMINAMATH_GPT_center_of_conic_l402_40224

-- Define the conic equation
def conic_equation (p q r α β γ : ℝ) : Prop :=
  p * α * β + q * α * γ + r * β * γ = 0

-- Define the barycentric coordinates of the center
def center_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (r * (p + q - r), q * (p + r - q), p * (r + q - p))

-- Theorem to prove that the barycentric coordinates of the center are as expected
theorem center_of_conic (p q r α β γ : ℝ) (h : conic_equation p q r α β γ) :
  center_coordinates p q r = (r * (p + q - r), q * (p + r - q), p * (r + q - p)) := 
sorry

end NUMINAMATH_GPT_center_of_conic_l402_40224


namespace NUMINAMATH_GPT_math_problem_l402_40246

noncomputable def parametric_equation_line (x y t : ℝ) : Prop :=
  x = 1 + (1/2) * t ∧ y = -5 + (Real.sqrt 3 / 2) * t

noncomputable def polar_equation_circle (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.sin θ

noncomputable def line_disjoint_circle (sqrt3 x y d : ℝ) : Prop :=
  sqrt3 = Real.sqrt 3 ∧ x = 0 ∧ y = 4 ∧ d = (9 + sqrt3) / 2 ∧ d > 4

theorem math_problem 
  (t θ x y ρ sqrt3 d : ℝ) :
  parametric_equation_line x y t ∧
  polar_equation_circle ρ θ ∧
  line_disjoint_circle sqrt3 x y d :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l402_40246


namespace NUMINAMATH_GPT_lucas_fence_painting_l402_40293

-- Define the conditions
def total_time := 60
def time_painting := 12
def rate_per_minute := 1 / total_time

-- State the theorem
theorem lucas_fence_painting :
  let work_done := rate_per_minute * time_painting
  work_done = 1 / 5 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_lucas_fence_painting_l402_40293


namespace NUMINAMATH_GPT_mod_remainder_l402_40229

theorem mod_remainder (a b c x: ℤ):
    a = 9 → b = 5 → c = 3 → x = 7 →
    (a^6 + b^7 + c^8) % x = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mod_remainder_l402_40229


namespace NUMINAMATH_GPT_boys_of_other_communities_l402_40253

axiom total_boys : ℕ
axiom muslim_percentage : ℝ
axiom hindu_percentage : ℝ
axiom sikh_percentage : ℝ

noncomputable def other_boy_count (total_boys : ℕ) 
                                   (muslim_percentage : ℝ) 
                                   (hindu_percentage : ℝ) 
                                   (sikh_percentage : ℝ) : ℝ :=
  let total_percentage := muslim_percentage + hindu_percentage + sikh_percentage
  let other_percentage := 1 - total_percentage
  other_percentage * total_boys

theorem boys_of_other_communities : 
    other_boy_count 850 0.44 0.32 0.10 = 119 :=
  by 
    sorry

end NUMINAMATH_GPT_boys_of_other_communities_l402_40253


namespace NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l402_40296

-- Define the propositions P and Q
def P (x : ℝ) : Prop := |x - 2| ≤ 3
def Q (x : ℝ) : Prop := x ≥ -1 ∨ x ≤ 5

-- Define the statement to prove
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) :=
by
  sorry

end NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l402_40296


namespace NUMINAMATH_GPT_regular_polygon_sides_l402_40249

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l402_40249


namespace NUMINAMATH_GPT_ways_A_not_head_is_600_l402_40244

-- Definitions for the problem conditions
def num_people : ℕ := 6
def valid_positions_for_A : ℕ := 5
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The total number of ways person A can be placed in any position except the first
def num_ways_A_not_head : ℕ := valid_positions_for_A * factorial (num_people - 1)

-- The theorem to prove
theorem ways_A_not_head_is_600 : num_ways_A_not_head = 600 := by
  sorry

end NUMINAMATH_GPT_ways_A_not_head_is_600_l402_40244


namespace NUMINAMATH_GPT_equation_no_solution_for_k_7_l402_40277

theorem equation_no_solution_for_k_7 :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ 5) → ¬ (x ^ 2 - 1) / (x - 3) = (x ^ 2 - 7) / (x - 5) :=
by
  intro x h
  have h1 : x ≠ 3 := h.1
  have h2 : x ≠ 5 := h.2
  sorry

end NUMINAMATH_GPT_equation_no_solution_for_k_7_l402_40277


namespace NUMINAMATH_GPT_nth_odd_positive_integer_is_199_l402_40217

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end NUMINAMATH_GPT_nth_odd_positive_integer_is_199_l402_40217


namespace NUMINAMATH_GPT_units_digit_of_product_l402_40286

theorem units_digit_of_product : 
  (3 ^ 401 * 7 ^ 402 * 23 ^ 403) % 10 = 9 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_of_product_l402_40286


namespace NUMINAMATH_GPT_number_of_solutions_sine_quadratic_l402_40255

theorem number_of_solutions_sine_quadratic :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → 3 * (Real.sin x) ^ 2 - 5 * (Real.sin x) + 2 = 0 →
  ∃ a b c, x = a ∨ x = b ∨ x = c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c :=
sorry

end NUMINAMATH_GPT_number_of_solutions_sine_quadratic_l402_40255


namespace NUMINAMATH_GPT_number_of_principals_in_oxford_high_school_l402_40209

-- Define the conditions
def numberOfTeachers : ℕ := 48
def numberOfClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def totalStudents : ℕ := numberOfClasses * studentsPerClass
def totalPeople : ℕ := 349
def numberOfPrincipals : ℕ := totalPeople - (numberOfTeachers + totalStudents)

-- Proposition: Prove the number of principals in Oxford High School
theorem number_of_principals_in_oxford_high_school :
  numberOfPrincipals = 1 := by sorry

end NUMINAMATH_GPT_number_of_principals_in_oxford_high_school_l402_40209


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l402_40269

def point : Type := (ℝ × ℝ)
def vec (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

noncomputable def A : point := (-2, 4)
noncomputable def B : point := (3, -1)
noncomputable def C : point := (-3, -4)

noncomputable def a : point := vec A B
noncomputable def b : point := vec B C
noncomputable def c : point := vec C A

-- Problem 1
theorem problem1 : (3 * a.1 + b.1 - 3 * c.1, 3 * a.2 + b.2 - 3 * c.2) = (6, -42) :=
sorry

-- Problem 2
theorem problem2 : ∃ m n : ℝ, a = (m * b.1 + n * c.1, m * b.2 + n * c.2) ∧ m = -1 ∧ n = -1 :=
sorry

-- Helper function for point addition
def add_point (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)
def scale_point (k : ℝ) (p : point) : point := (k * p.1, k * p.2)

-- problem 3
noncomputable def M : point := add_point (scale_point 3 c) C
noncomputable def N : point := add_point (scale_point (-2) b) C

theorem problem3 : M = (0, 20) ∧ N = (9, 2) ∧ vec M N = (9, -18) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l402_40269


namespace NUMINAMATH_GPT_f_monotone_f_inequality_solution_l402_40295

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x > 0 → ∃ y, f y = x
axiom f_at_2: f 2 = 1
axiom f_mul : ∀ x y, f (x * y) = f x + f y
axiom f_positive : ∀ x, x > 1 → f x > 0

theorem f_monotone (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

theorem f_inequality_solution (x : ℝ) (hx : x > 2 ∧ x ≤ 4) : f x + f (x - 2) ≤ 3 :=
sorry

end NUMINAMATH_GPT_f_monotone_f_inequality_solution_l402_40295


namespace NUMINAMATH_GPT_addition_terms_correct_l402_40266

def first_seq (n : ℕ) : ℕ := 2 * n + 1
def second_seq (n : ℕ) : ℕ := 5 * n - 1

theorem addition_terms_correct :
  first_seq 10 = 21 ∧ second_seq 10 = 49 ∧
  first_seq 80 = 161 ∧ second_seq 80 = 399 :=
by
  sorry

end NUMINAMATH_GPT_addition_terms_correct_l402_40266


namespace NUMINAMATH_GPT_remainder_of_division_l402_40242

theorem remainder_of_division :
  ∀ (L S R : ℕ), 
  L = 1575 → 
  L - S = 1365 → 
  S * 7 + R = L → 
  R = 105 :=
by
  intros L S R h1 h2 h3
  sorry

end NUMINAMATH_GPT_remainder_of_division_l402_40242


namespace NUMINAMATH_GPT_ashok_average_marks_l402_40285

theorem ashok_average_marks (avg_6 : ℝ) (marks_6 : ℝ) (total_sub : ℕ) (sub_6 : ℕ)
  (h1 : avg_6 = 75) (h2 : marks_6 = 80) (h3 : total_sub = 6) (h4 : sub_6 = 5) :
  (avg_6 * total_sub - marks_6) / sub_6 = 74 :=
by
  sorry

end NUMINAMATH_GPT_ashok_average_marks_l402_40285


namespace NUMINAMATH_GPT_impossible_to_transport_50_stones_l402_40236

def arithmetic_sequence (a d n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

def can_transport (weights : List ℕ) (k : ℕ) (max_weight : ℕ) : Prop :=
  ∃ partition : List (List ℕ), partition.length = k ∧
    (∀ part ∈ partition, (part.sum ≤ max_weight))

theorem impossible_to_transport_50_stones :
  ¬ can_transport (arithmetic_sequence 370 2 50) 7 3000 :=
by
  sorry

end NUMINAMATH_GPT_impossible_to_transport_50_stones_l402_40236


namespace NUMINAMATH_GPT_arithmetic_sequence_a17_l402_40247

theorem arithmetic_sequence_a17 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : S 13 = 78)
  (h2 : a 7 + a 12 = 10)
  (h_sum : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 1 + (a 2 - a 1) / (2 - 1)))
  (h_term : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1) / (2 - 1)) :
  a 17 = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a17_l402_40247


namespace NUMINAMATH_GPT_race_head_start_l402_40230

/-- A's speed is 22/19 times that of B. If A and B run a race, A should give B a head start of (3 / 22) of the race length so the race ends in a dead heat. -/
theorem race_head_start {Va Vb L H : ℝ} (hVa : Va = (22 / 19) * Vb) (hL_Va : L / Va = (L - H) / Vb) : 
  H = (3 / 22) * L :=
by
  sorry

end NUMINAMATH_GPT_race_head_start_l402_40230


namespace NUMINAMATH_GPT_least_n_divisibility_condition_l402_40234

theorem least_n_divisibility_condition :
  ∃ n : ℕ, 0 < n ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k ∣ (n^2 - n + 1) ↔ (n = 5 ∧ k = 3)) := 
sorry

end NUMINAMATH_GPT_least_n_divisibility_condition_l402_40234


namespace NUMINAMATH_GPT_total_respondents_l402_40227

theorem total_respondents (X Y : ℕ) 
  (hX : X = 60) 
  (hRatio : 3 * Y = X) : 
  X + Y = 80 := 
by
  sorry

end NUMINAMATH_GPT_total_respondents_l402_40227


namespace NUMINAMATH_GPT_product_calc_l402_40298

theorem product_calc : (16 * 0.5 * 4 * 0.125 = 4) :=
by
  sorry

end NUMINAMATH_GPT_product_calc_l402_40298


namespace NUMINAMATH_GPT_cobbler_works_fri_hours_l402_40243

-- Conditions
def mending_rate : ℕ := 3  -- Pairs of shoes per hour
def mon_to_thu_days : ℕ := 4
def hours_per_day : ℕ := 8
def weekly_mended_pairs : ℕ := 105

-- Translate the conditions
def hours_mended_mon_to_thu : ℕ := mon_to_thu_days * hours_per_day
def pairs_mended_mon_to_thu : ℕ := mending_rate * hours_mended_mon_to_thu
def pairs_mended_fri : ℕ := weekly_mended_pairs - pairs_mended_mon_to_thu

-- Theorem statement to prove the desired question
theorem cobbler_works_fri_hours : (pairs_mended_fri / mending_rate) = 3 := by
  sorry

end NUMINAMATH_GPT_cobbler_works_fri_hours_l402_40243


namespace NUMINAMATH_GPT_fruit_eating_problem_l402_40214

theorem fruit_eating_problem (a₀ p₀ o₀ : ℕ) (h₀ : a₀ = 5) (h₁ : p₀ = 8) (h₂ : o₀ = 11) :
  ¬ ∃ (d : ℕ), (a₀ - d) = (p₀ - d) ∧ (p₀ - d) = (o₀ - d) ∧ ∀ k, k ≤ d → ((a₀ - k) + (p₀ - k) + (o₀ - k) = 24 - 2 * k ∧ a₀ - k ≥ 0 ∧ p₀ - k ≥ 0 ∧ o₀ - k ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_fruit_eating_problem_l402_40214


namespace NUMINAMATH_GPT_d_in_N_l402_40264

def M := {x : ℤ | ∃ n : ℤ, x = 3 * n}
def N := {x : ℤ | ∃ n : ℤ, x = 3 * n + 1}
def P := {x : ℤ | ∃ n : ℤ, x = 3 * n - 1}

theorem d_in_N (a b c d : ℤ) (ha : a ∈ M) (hb : b ∈ N) (hc : c ∈ P) (hd : d = a - b + c) : d ∈ N :=
by sorry

end NUMINAMATH_GPT_d_in_N_l402_40264


namespace NUMINAMATH_GPT_harriet_return_speed_l402_40223

/-- Harriet's trip details: 
  - speed from A-ville to B-town is 100 km/h
  - the entire trip took 5 hours
  - time to drive from A-ville to B-town is 180 minutes (3 hours) 
  Prove the speed while driving back to A-ville is 150 km/h
--/
theorem harriet_return_speed:
  ∀ (t₁ t₂ : ℝ),
  (t₁ = 3) ∧ 
  (100 * t₁ = d) ∧ 
  (t₁ + t₂ = 5) ∧ 
  (t₂ = 2) →
  (d / t₂ = 150) :=
by
  intros t₁ t₂ h
  sorry

end NUMINAMATH_GPT_harriet_return_speed_l402_40223


namespace NUMINAMATH_GPT_matrix_not_invertible_l402_40207

noncomputable def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem matrix_not_invertible (x : ℝ) :
  determinant (2*x + 1) 9 (4 - x) 10 = 0 ↔ x = 26/29 := by
  sorry

end NUMINAMATH_GPT_matrix_not_invertible_l402_40207


namespace NUMINAMATH_GPT_cycling_sequences_reappear_after_28_cycles_l402_40250

/-- Cycling pattern of letters and digits. Letter cycle length is 7; digit cycle length is 4.
Prove that the LCM of 7 and 4 is 28, which is the first line on which both sequences will reappear -/
theorem cycling_sequences_reappear_after_28_cycles 
  (letters_cycle_length : ℕ) (digits_cycle_length : ℕ) 
  (h_letters : letters_cycle_length = 7) 
  (h_digits : digits_cycle_length = 4) 
  : Nat.lcm letters_cycle_length digits_cycle_length = 28 :=
by
  rw [h_letters, h_digits]
  sorry

end NUMINAMATH_GPT_cycling_sequences_reappear_after_28_cycles_l402_40250


namespace NUMINAMATH_GPT_geometric_sequence_increasing_iff_l402_40248

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem geometric_sequence_increasing_iff 
  (ha : is_geometric_sequence a q) 
  (h : a 0 < a 1 ∧ a 1 < a 2) : 
  is_increasing_sequence a ↔ (a 0 < a 1 ∧ a 1 < a 2) := 
sorry

end NUMINAMATH_GPT_geometric_sequence_increasing_iff_l402_40248


namespace NUMINAMATH_GPT_bottles_difference_l402_40206

noncomputable def Donald_drinks_bottles (P: ℕ): ℕ := 2 * P + 3
noncomputable def Paul_drinks_bottles: ℕ := 3
noncomputable def actual_Donald_bottles: ℕ := 9

theorem bottles_difference:
  actual_Donald_bottles - 2 * Paul_drinks_bottles = 3 :=
by 
  sorry

end NUMINAMATH_GPT_bottles_difference_l402_40206


namespace NUMINAMATH_GPT_find_f_l402_40228

def f (x : ℝ) : ℝ := 3 * x + 2

theorem find_f (x : ℝ) : f x = 3 * x + 2 :=
  sorry

end NUMINAMATH_GPT_find_f_l402_40228


namespace NUMINAMATH_GPT_cost_of_three_stamps_is_correct_l402_40251

-- Define the cost of one stamp
def cost_of_one_stamp : ℝ := 0.34

-- Define the number of stamps
def number_of_stamps : ℕ := 3

-- Define the expected total cost for three stamps
def expected_cost : ℝ := 1.02

-- Prove that the cost of three stamps is equal to the expected cost
theorem cost_of_three_stamps_is_correct : cost_of_one_stamp * number_of_stamps = expected_cost :=
by
  sorry

end NUMINAMATH_GPT_cost_of_three_stamps_is_correct_l402_40251


namespace NUMINAMATH_GPT_children_distribution_l402_40261

theorem children_distribution (a b c d N : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : a + b + c + d < 18) 
  (h5 : a * b * c * d = N) : 
  N = 120 ∧ a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 := 
by 
  sorry

end NUMINAMATH_GPT_children_distribution_l402_40261


namespace NUMINAMATH_GPT_repeated_1991_mod_13_l402_40211

theorem repeated_1991_mod_13 (k : ℕ) : 
  ((10^4 - 9) * (1991 * (10^(4*k) - 1)) / 9) % 13 = 8 :=
by
  sorry

end NUMINAMATH_GPT_repeated_1991_mod_13_l402_40211


namespace NUMINAMATH_GPT_units_digit_sum_l402_40291

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum
  (h1 : units_digit 13 = 3)
  (h2 : units_digit 41 = 1)
  (h3 : units_digit 27 = 7)
  (h4 : units_digit 34 = 4) :
  units_digit ((13 * 41) + (27 * 34)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_sum_l402_40291


namespace NUMINAMATH_GPT_square_area_l402_40256

/- Given: 
    1. The area of the isosceles right triangle ΔAEF is 1 cm².
    2. The area of the rectangle EFGH is 10 cm².
- To prove: 
    The area of the square ABCD is 24.5 cm².
-/

theorem square_area
  (h1 : ∃ a : ℝ, (0 < a) ∧ (a * a / 2 = 1))  -- Area of isosceles right triangle ΔAEF is 1 cm²
  (h2 : ∃ w l : ℝ, (w = 2) ∧ (l * w = 10))  -- Area of rectangle EFGH is 10 cm²
  : ∃ s : ℝ, (s * s = 24.5) := -- Area of the square ABCD is 24.5 cm²
sorry

end NUMINAMATH_GPT_square_area_l402_40256


namespace NUMINAMATH_GPT_least_number_to_subtract_l402_40202

theorem least_number_to_subtract (n : ℕ) (h : n = 13294) : ∃ k : ℕ, n - 1 = k * 97 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l402_40202


namespace NUMINAMATH_GPT_GPA_of_rest_of_classroom_l402_40219

variable (n : ℕ) (x : ℝ)
variable (H1 : ∀ n, n > 0)
variable (H2 : (15 * n + 2 * n * x) / (3 * n) = 17)

theorem GPA_of_rest_of_classroom (n : ℕ) (H1 : ∀ n, n > 0) (H2 : (15 * n + 2 * n * x) / (3 * n) = 17) : x = 18 := by
  sorry

end NUMINAMATH_GPT_GPA_of_rest_of_classroom_l402_40219


namespace NUMINAMATH_GPT_rational_ordering_l402_40200

theorem rational_ordering :
  (-3:ℚ)^2 < -1/3 ∧ (-1/3 < ((-3):ℚ)^2 ∧ ((-3:ℚ)^2 = |((-3:ℚ))^2|)) := 
by 
  sorry

end NUMINAMATH_GPT_rational_ordering_l402_40200


namespace NUMINAMATH_GPT_length_of_faster_train_is_380_meters_l402_40262

-- Defining the conditions
def speed_faster_train_kmph := 144
def speed_slower_train_kmph := 72
def time_seconds := 19

-- Conversion factor
def kmph_to_mps (speed : Nat) : Nat := speed * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps : Nat := kmph_to_mps (speed_faster_train_kmph - speed_slower_train_kmph)

-- Problem statement: Prove that the length of the faster train is 380 meters
theorem length_of_faster_train_is_380_meters :
  relative_speed_mps * time_seconds = 380 :=
sorry

end NUMINAMATH_GPT_length_of_faster_train_is_380_meters_l402_40262


namespace NUMINAMATH_GPT_fill_tank_with_leak_l402_40235

theorem fill_tank_with_leak (A L : ℝ) (h1 : A = 1 / 6) (h2 : L = 1 / 18) : (1 / (A - L)) = 9 :=
by
  sorry

end NUMINAMATH_GPT_fill_tank_with_leak_l402_40235


namespace NUMINAMATH_GPT_triangle_side_lengths_exist_l402_40281

theorem triangle_side_lengths_exist 
  (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  ∃ (x y z : ℝ), 
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ 
  (a = y + z) ∧ (b = x + z) ∧ (c = x + y) :=
by
  let x := (a - b + c) / 2
  let y := (a + b - c) / 2
  let z := (-a + b + c) / 2
  have hx : x > 0 := sorry
  have hy : y > 0 := sorry
  have hz : z > 0 := sorry
  have ha : a = y + z := sorry
  have hb : b = x + z := sorry
  have hc : c = x + y := sorry
  exact ⟨x, y, z, hx, hy, hz, ha, hb, hc⟩

end NUMINAMATH_GPT_triangle_side_lengths_exist_l402_40281


namespace NUMINAMATH_GPT_number_of_red_balls_l402_40221

-- Initial conditions
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 5
def freq_red_ball : ℝ := 0.4

-- Proving the number of red balls
theorem number_of_red_balls (total_balls : ℕ) (num_red_balls : ℕ) :
  total_balls = num_black_balls + num_white_balls + num_red_balls ∧
  (num_red_balls : ℝ) / total_balls = freq_red_ball →
  num_red_balls = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_balls_l402_40221


namespace NUMINAMATH_GPT_max_tiles_to_spell_CMWMC_l402_40299

theorem max_tiles_to_spell_CMWMC {Cs Ms Ws : ℕ} (hC : Cs = 8) (hM : Ms = 8) (hW : Ws = 8) : 
  ∃ (max_draws : ℕ), max_draws = 18 :=
by
  -- Assuming we have 8 C's, 8 M's, and 8 W's in the bag
  sorry

end NUMINAMATH_GPT_max_tiles_to_spell_CMWMC_l402_40299


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l402_40212

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 - 2 * x - 8 ≤ 0 ↔ -4/3 ≤ x ∧ x ≤ 2 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l402_40212


namespace NUMINAMATH_GPT_candy_eaten_l402_40267

/--
Given:
- Faye initially had 47 pieces of candy
- Faye ate x pieces the first night
- Faye's sister gave her 40 more pieces
- Now Faye has 62 pieces of candy

We need to prove:
- Faye ate 25 pieces of candy the first night.
-/
theorem candy_eaten (x : ℕ) (h1 : 47 - x + 40 = 62) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_candy_eaten_l402_40267


namespace NUMINAMATH_GPT_minimum_value_inverse_sum_l402_40245

variables {m n : ℝ}

theorem minimum_value_inverse_sum 
  (hm : m > 0) 
  (hn : n > 0) 
  (hline : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1)
  (hchord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ m * x2 + n * y2 + 2 = 0 → 
    (x1 - x2)^2 + (y1 - y2)^2 = 4) : 
  ∃ m n : ℝ, 3 * m + n = 2 ∧ m > 0 ∧ n > 0 ∧ 
    (∀ m' n' : ℝ, 3 * m' + n' = 2 → m' > 0 → n' > 0 → 
      (1 / m' + 3 / n' ≥ 6)) :=
sorry

end NUMINAMATH_GPT_minimum_value_inverse_sum_l402_40245


namespace NUMINAMATH_GPT_smallest_number_of_eggs_proof_l402_40263

noncomputable def smallest_number_of_eggs (c : ℕ) : ℕ := 15 * c - 3

theorem smallest_number_of_eggs_proof :
  ∃ c : ℕ, c ≥ 11 ∧ smallest_number_of_eggs c = 162 ∧ smallest_number_of_eggs c > 150 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_eggs_proof_l402_40263
