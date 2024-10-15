import Mathlib

namespace NUMINAMATH_GPT_city_population_correct_l2323_232339

variable (C G : ℕ)

theorem city_population_correct :
  (C - G = 119666) ∧ (C + G = 845640) → (C = 482653) := by
  intro h
  have h1 : C - G = 119666 := h.1
  have h2 : C + G = 845640 := h.2
  sorry

end NUMINAMATH_GPT_city_population_correct_l2323_232339


namespace NUMINAMATH_GPT_winning_strategy_l2323_232377

/-- Given a square table n x n, two players A and B are playing the following game: 
  - At the beginning, all cells of the table are empty.
  - Player A has the first move, and in each of their moves, a player will put a coin on some cell 
    that doesn't contain a coin and is not adjacent to any of the cells that already contain a coin. 
  - The player who makes the last move wins. 

  Cells are adjacent if they share an edge.

  - If n is even, player B has the winning strategy.
  - If n is odd, player A has the winning strategy.
-/
theorem winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ (B_strat : winning_strategy_for_B), True) ∧ (n % 2 = 1 → ∃ (A_strat : winning_strategy_for_A), True) :=
by {
  admit
}

end NUMINAMATH_GPT_winning_strategy_l2323_232377


namespace NUMINAMATH_GPT_initial_students_count_l2323_232334

theorem initial_students_count (n W : ℝ)
    (h1 : W = n * 28)
    (h2 : W + 10 = (n + 1) * 27.4) :
    n = 29 :=
by
  sorry

end NUMINAMATH_GPT_initial_students_count_l2323_232334


namespace NUMINAMATH_GPT_set_in_proportion_l2323_232341

theorem set_in_proportion : 
  let a1 := 3
  let a2 := 9
  let b1 := 10
  let b2 := 30
  (a1 * b2 = a2 * b1) := 
by {
  sorry
}

end NUMINAMATH_GPT_set_in_proportion_l2323_232341


namespace NUMINAMATH_GPT_consecutive_integer_sets_l2323_232360

-- Define the problem
def sum_consecutive_integers (n a : ℕ) : ℕ :=
  (n * (2 * a + n - 1)) / 2

def is_valid_sequence (n a S : ℕ) : Prop :=
  n ≥ 2 ∧ sum_consecutive_integers n a = S

-- Lean 4 theorem statement
theorem consecutive_integer_sets (S : ℕ) (h : S = 180) :
  (∃ (n a : ℕ), is_valid_sequence n a S) →
  (∃ (n1 n2 n3 : ℕ) (a1 a2 a3 : ℕ), 
    is_valid_sequence n1 a1 S ∧ 
    is_valid_sequence n2 a2 S ∧ 
    is_valid_sequence n3 a3 S ∧
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integer_sets_l2323_232360


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2323_232327

noncomputable def min_value (p q r s t u : ℝ) : ℝ :=
  (1 / p) + (9 / q) + (25 / r) + (49 / s) + (81 / t) + (121 / u)

theorem minimum_value_of_expression (p q r s t u : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hs : 0 < s) (ht : 0 < t) (hu : 0 < u) (h_sum : p + q + r + s + t + u = 11) :
  min_value p q r s t u ≥ 1296 / 11 :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2323_232327


namespace NUMINAMATH_GPT_taxi_fare_distance_l2323_232376

-- Define the fare calculation and distance function
def fare (x : ℕ) : ℝ :=
  if x ≤ 4 then 10
  else 10 + (x - 4) * 1.5

-- Proof statement
theorem taxi_fare_distance (x : ℕ) : fare x = 16 → x = 8 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_taxi_fare_distance_l2323_232376


namespace NUMINAMATH_GPT_perfect_cubes_l2323_232358

theorem perfect_cubes (n : ℕ) (h : n > 0) : 
  (n = 7 ∨ n = 11 ∨ n = 12 ∨ n = 25) ↔ ∃ k : ℤ, (n^3 - 18*n^2 + 115*n - 391) = k^3 :=
by exact sorry

end NUMINAMATH_GPT_perfect_cubes_l2323_232358


namespace NUMINAMATH_GPT_increasing_interval_f_l2323_232391

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (Real.pi / 6))

theorem increasing_interval_f : ∃ a b : ℝ, a < b ∧ 
  (∀ x y : ℝ, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y) ∧
  (a = - (Real.pi / 6)) ∧ (b = (Real.pi / 3)) :=
by
  sorry

end NUMINAMATH_GPT_increasing_interval_f_l2323_232391


namespace NUMINAMATH_GPT_solve_for_y_l2323_232323

theorem solve_for_y (x y : ℝ) (h : 5 * x + 3 * y = 1) : y = (1 - 5 * x) / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2323_232323


namespace NUMINAMATH_GPT_compare_abc_l2323_232338

theorem compare_abc :
  let a := Real.log 17
  let b := 3
  let c := Real.exp (Real.sqrt 2)
  a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_compare_abc_l2323_232338


namespace NUMINAMATH_GPT_velvet_needed_for_box_l2323_232337

theorem velvet_needed_for_box : 
  let area_long_side := 8 * 6
  let area_short_side := 5 * 6
  let area_top_bottom := 40
  let total_area := (2 * area_long_side) + (2 * area_short_side) + (2 * area_top_bottom)
  total_area = 236 :=
by
  sorry

end NUMINAMATH_GPT_velvet_needed_for_box_l2323_232337


namespace NUMINAMATH_GPT_sum_of_valid_two_digit_numbers_l2323_232308

theorem sum_of_valid_two_digit_numbers
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (a - b) ∣ (10 * a + b))
  (h4 : (a * b) ∣ (10 * a + b)) :
  (10 * a + b = 21) → (21 = 21) :=
sorry

end NUMINAMATH_GPT_sum_of_valid_two_digit_numbers_l2323_232308


namespace NUMINAMATH_GPT_find_primes_l2323_232313

theorem find_primes (p : ℕ) (hp : Nat.Prime p) :
  (∃ a b c k : ℤ, a^2 + b^2 + c^2 = p ∧ a^4 + b^4 + c^4 = k * p) ↔ (p = 2 ∨ p = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_primes_l2323_232313


namespace NUMINAMATH_GPT_find_m_range_l2323_232329

-- Definitions for the conditions and the required proof
def condition_alpha (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m + 7
def condition_beta (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

-- Proof problem translated to Lean 4 statement
theorem find_m_range (m : ℝ) :
  (∀ x, condition_beta x → condition_alpha m x) → (-2 ≤ m ∧ m ≤ 0) :=
by sorry

end NUMINAMATH_GPT_find_m_range_l2323_232329


namespace NUMINAMATH_GPT_determine_a_value_l2323_232352

theorem determine_a_value :
  ∀ (a b c d : ℕ), 
  (a = b + 3) →
  (b = c + 6) →
  (c = d + 15) →
  (d = 50) →
  a = 74 :=
by
  intros a b c d h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_determine_a_value_l2323_232352


namespace NUMINAMATH_GPT_sum_base8_to_decimal_l2323_232345

theorem sum_base8_to_decimal (a b : ℕ) (ha : a = 5) (hb : b = 0o17)
  (h_sum_base8 : a + b = 0o24) : (a + b) = 20 := by
  sorry

end NUMINAMATH_GPT_sum_base8_to_decimal_l2323_232345


namespace NUMINAMATH_GPT_corrected_mean_l2323_232333

/-- The original mean of 20 observations is 36, an observation of 25 was wrongly recorded as 40.
    The correct mean is 35.25. -/
theorem corrected_mean 
  (Mean : ℝ)
  (Observations : ℕ)
  (IncorrectObservation : ℝ)
  (CorrectObservation : ℝ)
  (h1 : Mean = 36)
  (h2 : Observations = 20)
  (h3 : IncorrectObservation = 40)
  (h4 : CorrectObservation = 25) :
  (Mean * Observations - (IncorrectObservation - CorrectObservation)) / Observations = 35.25 :=
sorry

end NUMINAMATH_GPT_corrected_mean_l2323_232333


namespace NUMINAMATH_GPT_remainder_of_polynomial_l2323_232317

theorem remainder_of_polynomial 
  (P : ℝ → ℝ) 
  (h₁ : P 15 = 16)
  (h₂ : P 10 = 4) :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 10) * (x - 15) * Q x + (12 / 5 * x - 20) :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_l2323_232317


namespace NUMINAMATH_GPT_gray_percentage_correct_l2323_232330

-- Define the conditions
def total_squares := 25
def type_I_triangle_equivalent_squares := 8 * (1 / 2)
def type_II_triangle_equivalent_squares := 8 * (1 / 4)
def full_gray_squares := 4

-- Calculate the gray component
def gray_squares := type_I_triangle_equivalent_squares + type_II_triangle_equivalent_squares + full_gray_squares

-- Fraction representing the gray part of the quilt
def gray_fraction := gray_squares / total_squares

-- Translate fraction to percentage
def gray_percentage := gray_fraction * 100

theorem gray_percentage_correct : gray_percentage = 40 := by
  simp [total_squares, type_I_triangle_equivalent_squares, type_II_triangle_equivalent_squares, full_gray_squares, gray_squares, gray_fraction, gray_percentage]
  sorry -- You could expand this to a detailed proof if needed.

end NUMINAMATH_GPT_gray_percentage_correct_l2323_232330


namespace NUMINAMATH_GPT_maria_min_score_fourth_quarter_l2323_232320

theorem maria_min_score_fourth_quarter (x : ℝ) :
  (82 + 77 + 78 + x) / 4 ≥ 85 ↔ x ≥ 103 :=
by
  sorry

end NUMINAMATH_GPT_maria_min_score_fourth_quarter_l2323_232320


namespace NUMINAMATH_GPT_jill_food_percentage_l2323_232332

theorem jill_food_percentage (total_amount : ℝ) (tax_rate_clothing tax_rate_other_items spent_clothing_rate spent_other_rate spent_total_tax_rate : ℝ) : 
  spent_clothing_rate = 0.5 →
  spent_other_rate = 0.25 →
  tax_rate_clothing = 0.1 →
  tax_rate_other_items = 0.2 →
  spent_total_tax_rate = 0.1 →
  (spent_clothing_rate * tax_rate_clothing * total_amount) + (spent_other_rate * tax_rate_other_items * total_amount) = spent_total_tax_rate * total_amount →
  (1 - spent_clothing_rate - spent_other_rate) * total_amount / total_amount = 0.25 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_jill_food_percentage_l2323_232332


namespace NUMINAMATH_GPT_fixed_point_graph_l2323_232316

theorem fixed_point_graph (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) : ∃ x y : ℝ, (x = 2 ∧ y = 2 ∧ y = a^(x-2) + 1) :=
by
  use 2
  use 2
  sorry

end NUMINAMATH_GPT_fixed_point_graph_l2323_232316


namespace NUMINAMATH_GPT_find_modulus_l2323_232305

open Complex -- Open the Complex namespace for convenience

noncomputable def modulus_of_z (a : ℝ) (h : (1 + 2 * Complex.I) * (a + Complex.I : ℂ) = Complex.re ((1 + 2 * Complex.I) * (a + Complex.I)) + Complex.im ((1 + 2 * Complex.I) * (a + Complex.I)) * Complex.I) : ℝ :=
  Complex.abs ((1 + 2 * Complex.I) * (a + Complex.I))

theorem find_modulus : modulus_of_z (-3) (by {
  -- Provide the condition that real part equals imaginary part
  admit -- This 'admit' serves as a placeholder for the proof of the condition 
}) = 5 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_find_modulus_l2323_232305


namespace NUMINAMATH_GPT_male_to_female_cat_weight_ratio_l2323_232373

variable (w_f w_m w_t : ℕ)

def female_cat_weight : Prop := w_f = 2
def total_weight : Prop := w_t = 6
def male_cat_heavier : Prop := w_m > w_f

theorem male_to_female_cat_weight_ratio
  (h_female_cat_weight : female_cat_weight w_f)
  (h_total_weight : total_weight w_t)
  (h_male_cat_heavier : male_cat_heavier w_m w_f) :
  w_m = 4 ∧ w_t = w_f + w_m ∧ (w_m / w_f) = 2 :=
by
  sorry

end NUMINAMATH_GPT_male_to_female_cat_weight_ratio_l2323_232373


namespace NUMINAMATH_GPT_orange_orchard_land_l2323_232384

theorem orange_orchard_land (F H : ℕ) 
  (h1 : F + H = 120) 
  (h2 : ∃ x : ℕ, x + (2 * x + 1) = 10) 
  (h3 : ∃ x : ℕ, 2 * x + 1 = H)
  (h4 : ∃ x : ℕ, F = x) 
  (h5 : ∃ y : ℕ, H = 2 * y + 1) :
  F = 36 ∧ H = 84 :=
by
  sorry

end NUMINAMATH_GPT_orange_orchard_land_l2323_232384


namespace NUMINAMATH_GPT_other_root_l2323_232363

open Complex

-- Defining the conditions that are given in the problem
def quadratic_equation (x : ℂ) (m : ℝ) : Prop :=
  x^2 + (1 - 2 * I) * x + (3 * m - I) = 0

def has_real_root (x : ℂ) : Prop :=
  ∃ α : ℝ, x = α

-- The main theorem statement we need to prove
theorem other_root (m : ℝ) (α : ℝ) (α_real_root : quadratic_equation α m) :
  quadratic_equation (-1/2 + 2 * I) m :=
sorry

end NUMINAMATH_GPT_other_root_l2323_232363


namespace NUMINAMATH_GPT_avg_new_students_l2323_232394

-- Definitions for conditions
def orig_strength : ℕ := 17
def orig_avg_age : ℕ := 40
def new_students_count : ℕ := 17
def decreased_avg_age : ℕ := 36 -- given that average decreases by 4 years, i.e., 40 - 4

-- Definition for the original total age
def total_age_orig : ℕ := orig_strength * orig_avg_age

-- Definition for the total number of students after new students join
def total_students : ℕ := orig_strength + new_students_count

-- Definition for the total age after new students join
def total_age_new : ℕ := total_students * decreased_avg_age

-- Definition for the total age of new students
def total_age_new_students : ℕ := total_age_new - total_age_orig

-- Definition for the average age of new students
def avg_age_new_students : ℕ := total_age_new_students / new_students_count

-- Lean theorem stating the proof problem
theorem avg_new_students : 
  avg_age_new_students = 32 := 
by sorry

end NUMINAMATH_GPT_avg_new_students_l2323_232394


namespace NUMINAMATH_GPT_number_of_subsets_B_l2323_232306

def A : Set ℕ := {1, 3}
def C : Set ℕ := {1, 3, 5}

theorem number_of_subsets_B : ∃ (n : ℕ), (∀ B : Set ℕ, A ∪ B = C → n = 4) :=
sorry

end NUMINAMATH_GPT_number_of_subsets_B_l2323_232306


namespace NUMINAMATH_GPT_max_value_expr_l2323_232362

theorem max_value_expr : ∃ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) = 85 :=
by sorry

end NUMINAMATH_GPT_max_value_expr_l2323_232362


namespace NUMINAMATH_GPT_original_square_area_l2323_232325

theorem original_square_area {x y : ℕ} (h1 : y ≠ 1)
  (h2 : x^2 = 24 + y^2) : x^2 = 49 :=
sorry

end NUMINAMATH_GPT_original_square_area_l2323_232325


namespace NUMINAMATH_GPT_intersection_M_N_l2323_232350

def M := { x : ℝ | x^2 - 2 * x < 0 }
def N := { x : ℝ | abs x < 1 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2323_232350


namespace NUMINAMATH_GPT_sqrt_multiplication_l2323_232389

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_multiplication_l2323_232389


namespace NUMINAMATH_GPT_find_wanderer_in_8th_bar_l2323_232328

noncomputable def wanderer_probability : ℚ := 1 / 3

theorem find_wanderer_in_8th_bar
    (total_bars : ℕ)
    (initial_prob_in_any_bar : ℚ)
    (prob_not_in_specific_bar : ℚ)
    (prob_not_in_first_seven : ℚ)
    (posterior_prob : ℚ)
    (h1 : total_bars = 8)
    (h2 : initial_prob_in_any_bar = 4 / 5)
    (h3 : prob_not_in_specific_bar = 1 - (initial_prob_in_any_bar / total_bars))
    (h4 : prob_not_in_first_seven = prob_not_in_specific_bar ^ 7)
    (h5 : posterior_prob = initial_prob_in_any_bar / prob_not_in_first_seven) :
    posterior_prob = wanderer_probability := 
sorry

end NUMINAMATH_GPT_find_wanderer_in_8th_bar_l2323_232328


namespace NUMINAMATH_GPT_simplify_trig_expr_l2323_232370

theorem simplify_trig_expr : 
  let θ := 60
  let tan_θ := Real.sqrt 3
  let cot_θ := (Real.sqrt 3)⁻¹
  (tan_θ^3 + cot_θ^3) / (tan_θ + cot_θ) = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_trig_expr_l2323_232370


namespace NUMINAMATH_GPT_parallel_lines_slope_eq_l2323_232319

theorem parallel_lines_slope_eq (a : ℝ) : (∀ x y : ℝ, 3 * y - 4 * a = 8 * x) ∧ (∀ x y : ℝ, y - 2 = (a + 4) * x) → a = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_eq_l2323_232319


namespace NUMINAMATH_GPT_abs_of_sub_sqrt_l2323_232354

theorem abs_of_sub_sqrt (h : 2 > Real.sqrt 3) : |2 - Real.sqrt 3| = 2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_abs_of_sub_sqrt_l2323_232354


namespace NUMINAMATH_GPT_largest_good_number_smallest_bad_number_l2323_232312

def is_good_number (M : ℕ) : Prop :=
  ∃ a b c d : ℤ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

theorem largest_good_number :
  ∀ M : ℕ, is_good_number M ↔ M ≤ 576 :=
by sorry

theorem smallest_bad_number :
  ∀ M : ℕ, ¬ is_good_number M ↔ M ≥ 443 :=
by sorry

end NUMINAMATH_GPT_largest_good_number_smallest_bad_number_l2323_232312


namespace NUMINAMATH_GPT_find_QE_l2323_232356

noncomputable def QE (QD DE : ℝ) : ℝ :=
  QD + DE

theorem find_QE :
  ∀ (Q C R D E : Type) (QR QD DE QE : ℝ), 
  QD = 5 →
  QE = QD + DE →
  QR = DE - QD →
  QR^2 = QD * QE →
  QE = (QD + 5 + 5 * Real.sqrt 5) / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_QE_l2323_232356


namespace NUMINAMATH_GPT_larger_number_is_eight_l2323_232383

theorem larger_number_is_eight (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_eight_l2323_232383


namespace NUMINAMATH_GPT_calculate_percentage_increase_l2323_232397

variable (fish_first_round : ℕ) (fish_second_round : ℕ) (fish_total : ℕ) (fish_last_round : ℕ) (increase : ℚ) (percentage_increase : ℚ)

theorem calculate_percentage_increase
  (h1 : fish_first_round = 8)
  (h2 : fish_second_round = fish_first_round + 12)
  (h3 : fish_total = 60)
  (h4 : fish_last_round = fish_total - (fish_first_round + fish_second_round))
  (h5 : increase = fish_last_round - fish_second_round)
  (h6 : percentage_increase = (increase / fish_second_round) * 100) :
  percentage_increase = 60 := by
  sorry

end NUMINAMATH_GPT_calculate_percentage_increase_l2323_232397


namespace NUMINAMATH_GPT_no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l2323_232388

theorem no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square :
  ∀ b : ℤ, ¬ ∃ k : ℤ, b^2 + 3*b + 1 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l2323_232388


namespace NUMINAMATH_GPT_sum_of_products_non_positive_l2323_232321

theorem sum_of_products_non_positive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end NUMINAMATH_GPT_sum_of_products_non_positive_l2323_232321


namespace NUMINAMATH_GPT_cost_to_replace_is_800_l2323_232369

-- Definitions based on conditions
def trade_in_value (num_movies : ℕ) (trade_in_price : ℕ) : ℕ :=
  num_movies * trade_in_price

def dvd_cost (num_movies : ℕ) (dvd_price : ℕ) : ℕ :=
  num_movies * dvd_price

def replacement_cost (num_movies : ℕ) (trade_in_price : ℕ) (dvd_price : ℕ) : ℕ :=
  dvd_cost num_movies dvd_price - trade_in_value num_movies trade_in_price

-- Problem statement: it costs John $800 to replace his movies
theorem cost_to_replace_is_800 (num_movies trade_in_price dvd_price : ℕ)
  (h1 : num_movies = 100) (h2 : trade_in_price = 2) (h3 : dvd_price = 10) :
  replacement_cost num_movies trade_in_price dvd_price = 800 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_cost_to_replace_is_800_l2323_232369


namespace NUMINAMATH_GPT_find_x_minus_4y_l2323_232395

theorem find_x_minus_4y (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - 3 * y = 10) : x - 4 * y = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_minus_4y_l2323_232395


namespace NUMINAMATH_GPT_pyramid_lateral_surface_area_l2323_232355

noncomputable def lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) : ℝ :=
  n * S

theorem pyramid_lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) (A : ℝ) :
  A = n * S * (Real.cos α) →
  lateral_surface_area S n α = A / (Real.cos α) :=
by
  sorry

end NUMINAMATH_GPT_pyramid_lateral_surface_area_l2323_232355


namespace NUMINAMATH_GPT_seashells_after_giving_cannot_determine_starfish_l2323_232375

-- Define the given conditions
def initial_seashells : Nat := 66
def seashells_given : Nat := 52
def seashells_left : Nat := 14

-- The main theorem to prove
theorem seashells_after_giving (initial : Nat) (given : Nat) (left : Nat) :
  initial = 66 -> given = 52 -> left = 14 -> initial - given = left :=
by 
  intros 
  sorry

-- The starfish count question
def starfish (count: Option Nat) : Prop :=
  count = none

-- Prove that we cannot determine the number of starfish Benny found
theorem cannot_determine_starfish (count: Option Nat) :
  count = none :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_seashells_after_giving_cannot_determine_starfish_l2323_232375


namespace NUMINAMATH_GPT_probability_same_color_white_l2323_232367

/--
Given a box with 6 white balls and 5 black balls, if 3 balls are drawn such that all drawn balls have the same color,
prove that the probability that these balls are white is 2/3.
-/
theorem probability_same_color_white :
  (∃ (n_white n_black drawn_white drawn_black total_same_color : ℕ),
    n_white = 6 ∧ n_black = 5 ∧
    drawn_white = Nat.choose n_white 3 ∧ drawn_black = Nat.choose n_black 3 ∧
    total_same_color = drawn_white + drawn_black ∧
    (drawn_white:ℚ) / total_same_color = 2 / 3) :=
sorry

end NUMINAMATH_GPT_probability_same_color_white_l2323_232367


namespace NUMINAMATH_GPT_length_of_one_side_nonagon_l2323_232387

def total_perimeter (n : ℕ) (side_length : ℝ) : ℝ := n * side_length

theorem length_of_one_side_nonagon (total_perimeter : ℝ) (n : ℕ) (side_length : ℝ) (h1 : n = 9) (h2 : total_perimeter = 171) : side_length = 19 :=
by
  sorry

end NUMINAMATH_GPT_length_of_one_side_nonagon_l2323_232387


namespace NUMINAMATH_GPT_students_from_other_communities_eq_90_l2323_232386

theorem students_from_other_communities_eq_90 {total_students : ℕ} 
  (muslims_percentage : ℕ)
  (hindus_percentage : ℕ)
  (sikhs_percentage : ℕ)
  (christians_percentage : ℕ)
  (buddhists_percentage : ℕ)
  : total_students = 1000 →
    muslims_percentage = 36 →
    hindus_percentage = 24 →
    sikhs_percentage = 15 →
    christians_percentage = 10 →
    buddhists_percentage = 6 →
    (total_students * (100 - (muslims_percentage + hindus_percentage + sikhs_percentage + christians_percentage + buddhists_percentage))) / 100 = 90 :=
by
  intros h_total h_muslims h_hindus h_sikhs h_christians h_buddhists
  -- Proof can be omitted as indicated
  sorry

end NUMINAMATH_GPT_students_from_other_communities_eq_90_l2323_232386


namespace NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l2323_232382

theorem value_of_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : (x + y)^2 = 4)
  (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l2323_232382


namespace NUMINAMATH_GPT_initially_caught_and_tagged_is_30_l2323_232368

open Real

-- Define conditions
def total_second_catch : ℕ := 50
def tagged_second_catch : ℕ := 2
def total_pond_fish : ℕ := 750

-- Define ratio condition
def ratio_condition (T : ℕ) : Prop :=
  (T : ℝ) / (total_pond_fish : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ)

-- Prove the number of fish initially caught and tagged is 30
theorem initially_caught_and_tagged_is_30 :
  ∃ T : ℕ, ratio_condition T ∧ T = 30 :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_initially_caught_and_tagged_is_30_l2323_232368


namespace NUMINAMATH_GPT_complement_of_M_in_U_l2323_232398

def universal_set : Set ℝ := {x | x > 0}
def set_M : Set ℝ := {x | x > 1}
def complement (U M : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ M}

theorem complement_of_M_in_U :
  complement universal_set set_M = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l2323_232398


namespace NUMINAMATH_GPT_radius_of_bicycle_wheel_is_13_l2323_232393

-- Define the problem conditions
def diameter_cm : ℕ := 26

-- Define the function to calculate radius from diameter
def radius (d : ℕ) : ℕ := d / 2

-- Prove that the radius is 13 cm when diameter is 26 cm
theorem radius_of_bicycle_wheel_is_13 :
  radius diameter_cm = 13 := 
sorry

end NUMINAMATH_GPT_radius_of_bicycle_wheel_is_13_l2323_232393


namespace NUMINAMATH_GPT_largest_possible_dividend_l2323_232340

theorem largest_possible_dividend (divisor quotient : ℕ) (remainder : ℕ) 
  (h_divisor : divisor = 18)
  (h_quotient : quotient = 32)
  (h_remainder : remainder < divisor) :
  quotient * divisor + remainder = 593 :=
by
  -- No proof here, add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_largest_possible_dividend_l2323_232340


namespace NUMINAMATH_GPT_line_equation_l2323_232366

theorem line_equation (a b : ℝ)
(h1 : a * -1 + b * 2 = 0) 
(h2 : a = b) :
((a = 1 ∧ b = -1) ∨ (a = 2 ∧ b = -1)) := 
by
  sorry

end NUMINAMATH_GPT_line_equation_l2323_232366


namespace NUMINAMATH_GPT_time_to_finish_work_with_both_tractors_l2323_232371

-- Definitions of given conditions
def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 15
def time_A_worked : ℚ := 13
def remaining_work : ℚ := 1 - (work_rate_A * time_A_worked)
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Statement that needs to be proven
theorem time_to_finish_work_with_both_tractors : 
  remaining_work / combined_work_rate = 3 :=
by
  sorry

end NUMINAMATH_GPT_time_to_finish_work_with_both_tractors_l2323_232371


namespace NUMINAMATH_GPT_earnings_last_friday_l2323_232385

theorem earnings_last_friday 
  (price_per_kg : ℕ := 2)
  (earnings_wednesday : ℕ := 30)
  (earnings_today : ℕ := 42)
  (total_kg_sold : ℕ := 48)
  (total_earnings : ℕ := total_kg_sold * price_per_kg) 
  (F : ℕ) :
  earnings_wednesday + F + earnings_today = total_earnings → F = 24 := by
  sorry

end NUMINAMATH_GPT_earnings_last_friday_l2323_232385


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2323_232372

namespace QuadraticInequality

variables {a b : ℝ}

def hasRoots (a b : ℝ) : Prop :=
  let x1 := -1 / 2
  let x2 := 1 / 3
  (- x1 + x2 = - b / a) ∧ (-x1 * x2 = 2 / a)

theorem solution_set_of_quadratic_inequality (h : hasRoots a b) : a + b = -14 :=
sorry

end QuadraticInequality

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2323_232372


namespace NUMINAMATH_GPT_problem_l2323_232311

variable {x y : ℝ}

theorem problem (hx : 0 < x) (hy : 0 < y) (h : x^2 - y^2 = 3 * x * y) :
  (x^2 / y^2) + (y^2 / x^2) - 2 = 9 :=
sorry

end NUMINAMATH_GPT_problem_l2323_232311


namespace NUMINAMATH_GPT_stan_run_duration_l2323_232304

def run_duration : ℕ := 100

def num_3_min_songs : ℕ := 10
def num_2_min_songs : ℕ := 15
def time_per_3_min_song : ℕ := 3
def time_per_2_min_song : ℕ := 2
def additional_time_needed : ℕ := 40

theorem stan_run_duration :
  (num_3_min_songs * time_per_3_min_song) + (num_2_min_songs * time_per_2_min_song) + additional_time_needed = run_duration := by
  sorry

end NUMINAMATH_GPT_stan_run_duration_l2323_232304


namespace NUMINAMATH_GPT_find_k_l2323_232380

-- Definitions for arithmetic sequence properties
noncomputable def sum_arith_seq (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n-1) / 2) * d

noncomputable def term_arith_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Given Conditions
variables (a₁ d : ℝ)
variables (k : ℕ)

axiom sum_condition : sum_arith_seq a₁ d 9 = sum_arith_seq a₁ d 4
axiom term_condition : term_arith_seq a₁ d 4 + term_arith_seq a₁ d k = 0

-- Prove k = 10
theorem find_k : k = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2323_232380


namespace NUMINAMATH_GPT_avg_age_family_now_l2323_232303

namespace average_age_family

-- Define initial conditions
def avg_age_husband_wife_marriage := 23
def years_since_marriage := 5
def age_child := 1
def number_of_family_members := 3

-- Prove that the average age of the family now is 19 years
theorem avg_age_family_now :
  (2 * avg_age_husband_wife_marriage + 2 * years_since_marriage + age_child) / number_of_family_members = 19 := by
  sorry

end average_age_family

end NUMINAMATH_GPT_avg_age_family_now_l2323_232303


namespace NUMINAMATH_GPT_athlete_speed_l2323_232326

theorem athlete_speed (d t : ℝ) (H_d : d = 200) (H_t : t = 40) : (d / t) = 5 := by
  sorry

end NUMINAMATH_GPT_athlete_speed_l2323_232326


namespace NUMINAMATH_GPT_minerals_now_l2323_232300

def minerals_yesterday (M : ℕ) : Prop := (M / 2 = 21)

theorem minerals_now (M : ℕ) (H : minerals_yesterday M) : (M + 6 = 48) :=
by 
  unfold minerals_yesterday at H
  sorry

end NUMINAMATH_GPT_minerals_now_l2323_232300


namespace NUMINAMATH_GPT_ArithmeticSequenceSum_l2323_232318

theorem ArithmeticSequenceSum (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 + a 2 = 10) 
  (h2 : a 4 = a 3 + 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  a 3 + a 4 = 18 :=
by
  sorry

end NUMINAMATH_GPT_ArithmeticSequenceSum_l2323_232318


namespace NUMINAMATH_GPT_no_arithmetic_progression_in_squares_l2323_232324

theorem no_arithmetic_progression_in_squares :
  ∀ (a d : ℕ), d > 0 → ¬ (∃ (f : ℕ → ℕ), 
    (∀ n, f n = a + n * d) ∧ 
    (∀ n, ∃ m, n ^ 2 = f m)) :=
by
  sorry

end NUMINAMATH_GPT_no_arithmetic_progression_in_squares_l2323_232324


namespace NUMINAMATH_GPT_correct_answer_l2323_232310

-- Define the problem conditions and question
def equation (y : ℤ) : Prop := y + 2 = -3

-- Prove that the correct answer is y = -5
theorem correct_answer : ∀ y : ℤ, equation y → y = -5 :=
by
  intros y h
  unfold equation at h
  linarith

end NUMINAMATH_GPT_correct_answer_l2323_232310


namespace NUMINAMATH_GPT_correct_answer_is_A_l2323_232361

-- Definitions derived from problem conditions
def algorithm := Type
def has_sequential_structure (alg : algorithm) : Prop := sorry -- Actual definition should define what a sequential structure is for an algorithm

-- Given: An algorithm must contain a sequential structure.
theorem correct_answer_is_A (alg : algorithm) : has_sequential_structure alg :=
sorry

end NUMINAMATH_GPT_correct_answer_is_A_l2323_232361


namespace NUMINAMATH_GPT_value_of_a7_l2323_232365

-- Let \( \{a_n\} \) be a sequence such that \( S_n \) denotes the sum of the first \( n \) terms.
-- Given \( S_{n+1}, S_{n+2}, S_{n+3} \) form an arithmetic sequence and \( a_2 = -2 \),
-- prove that \( a_7 = 64 \).

theorem value_of_a7 (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 2) + S (n + 1) = 2 * S n) →
  a 2 = -2 →
  (∀ n : ℕ, a (n + 2) = -2 * a (n + 1)) →
  a 7 = 64 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_value_of_a7_l2323_232365


namespace NUMINAMATH_GPT_wanda_crayons_l2323_232342

variable (Dina Jacob Wanda : ℕ)

theorem wanda_crayons : Dina = 28 ∧ Jacob = Dina - 2 ∧ Dina + Jacob + Wanda = 116 → Wanda = 62 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_wanda_crayons_l2323_232342


namespace NUMINAMATH_GPT_conference_fraction_married_men_l2323_232396

theorem conference_fraction_married_men 
  (total_women : ℕ) 
  (single_probability : ℚ) 
  (h_single_prob : single_probability = 3/7) 
  (h_total_women : total_women = 7) : 
  (4 : ℚ) / (11 : ℚ) = 4 / 11 := 
by
  sorry

end NUMINAMATH_GPT_conference_fraction_married_men_l2323_232396


namespace NUMINAMATH_GPT_money_split_l2323_232344

theorem money_split (donna_share friend_share : ℝ) (h1 : donna_share = 32.50) (h2 : friend_share = 32.50) :
  donna_share + friend_share = 65 :=
by
  sorry

end NUMINAMATH_GPT_money_split_l2323_232344


namespace NUMINAMATH_GPT_cylinder_height_l2323_232301

theorem cylinder_height (h : ℝ)
  (circumference : ℝ)
  (rectangle_diagonal : ℝ)
  (C_eq : circumference = 12)
  (d_eq : rectangle_diagonal = 20) :
  h = 16 :=
by
  -- We derive the result based on the given conditions and calculations
  sorry -- Skipping the proof part

end NUMINAMATH_GPT_cylinder_height_l2323_232301


namespace NUMINAMATH_GPT_length_of_bridge_l2323_232343

theorem length_of_bridge
  (length_train : ℕ) (speed_train_kmhr : ℕ) (crossing_time : ℕ)
  (speed_conversion_factor : ℝ) (m_per_s_kmhr : ℝ) 
  (speed_train_ms : ℝ) (total_distance : ℝ) (length_bridge : ℝ)
  (h1 : length_train = 155)
  (h2 : speed_train_kmhr = 45)
  (h3 : crossing_time = 30)
  (h4 : speed_conversion_factor = 1000 / 3600)
  (h5 : m_per_s_kmhr = speed_train_kmhr * speed_conversion_factor)
  (h6 : speed_train_ms = 45 * (5 / 18))
  (h7 : total_distance = speed_train_ms * crossing_time)
  (h8 : length_bridge = total_distance - length_train):
  length_bridge = 220 :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l2323_232343


namespace NUMINAMATH_GPT_sugar_already_put_in_l2323_232302

-- Definitions based on conditions
def required_sugar : ℕ := 13
def additional_sugar_needed : ℕ := 11

-- Theorem to be proven
theorem sugar_already_put_in :
  required_sugar - additional_sugar_needed = 2 := by
  sorry

end NUMINAMATH_GPT_sugar_already_put_in_l2323_232302


namespace NUMINAMATH_GPT_max_possible_value_of_k_l2323_232309

noncomputable def max_knights_saying_less : Nat :=
  let n := 2015
  let k := n - 2
  k

theorem max_possible_value_of_k : max_knights_saying_less = 2013 :=
by
  sorry

end NUMINAMATH_GPT_max_possible_value_of_k_l2323_232309


namespace NUMINAMATH_GPT_problem_a_problem_b_l2323_232322

-- Problem (a)
theorem problem_a (n : Nat) : Nat.mod (7 ^ (2 * n) - 4 ^ (2 * n)) 33 = 0 := sorry

-- Problem (b)
theorem problem_b (n : Nat) : Nat.mod (3 ^ (6 * n) - 2 ^ (6 * n)) 35 = 0 := sorry

end NUMINAMATH_GPT_problem_a_problem_b_l2323_232322


namespace NUMINAMATH_GPT_integer_pairs_perfect_squares_l2323_232314

theorem integer_pairs_perfect_squares (a b : ℤ) :
  (∃ k : ℤ, (a, b) = (k^2, 0) ∨ (a, b) = (0, k^2) ∨ (a, b) = (k, 1-k) ∨ (a, b) = (-6, -5) ∨ (a, b) = (-5, -6) ∨ (a, b) = (-4, -4))
  ↔ 
  (∃ x1 x2 : ℤ, a^2 + 4*b = x1^2 ∧ b^2 + 4*a = x2^2) :=
sorry

end NUMINAMATH_GPT_integer_pairs_perfect_squares_l2323_232314


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l2323_232357

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

def condition (S : ℕ → ℝ) : Prop :=
  (S 8 - S 5) * (S 8 - S 4) < 0

-- Theorem to prove
theorem arithmetic_sequence_property {a : ℕ → ℝ} {S : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_cond : condition S) :
  |a 5| > |a 6| := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l2323_232357


namespace NUMINAMATH_GPT_cone_height_l2323_232335

theorem cone_height
  (V1 V2 V : ℝ)
  (h1 h2 : ℝ)
  (fact1 : h1 = 10)
  (fact2 : h2 = 2)
  (h : ∀ m : ℝ, V1 = V * (10 ^ 3) / (m ^ 3) ∧ V2 = V * ((m - 2) ^ 3) / (m ^ 3))
  (equal_volumes : V1 + V2 = V) :
  (∃ m : ℝ, m = 13.897) :=
by
  sorry

end NUMINAMATH_GPT_cone_height_l2323_232335


namespace NUMINAMATH_GPT_xy_relationship_l2323_232348

theorem xy_relationship :
  let x := 123456789 * 123456786
  let y := 123456788 * 123456787
  x < y := 
by
  sorry

end NUMINAMATH_GPT_xy_relationship_l2323_232348


namespace NUMINAMATH_GPT_find_y_value_l2323_232390

theorem find_y_value (y : ℕ) (h1 : y ≤ 150)
  (h2 : (45 + 76 + 123 + y + y + y) / 6 = 2 * y) :
  y = 27 :=
sorry

end NUMINAMATH_GPT_find_y_value_l2323_232390


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_iff_m_eq_1_l2323_232381

theorem inequality_holds_for_all_x_iff_m_eq_1 (m : ℝ) (h_m : m ≠ 0) :
  (∀ x > 0, x^2 - 2 * m * Real.log x ≥ 1) ↔ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_iff_m_eq_1_l2323_232381


namespace NUMINAMATH_GPT_expected_value_of_biased_coin_l2323_232379

noncomputable def expected_value : ℚ :=
  (2 / 3) * 5 + (1 / 3) * -6

theorem expected_value_of_biased_coin :
  expected_value = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_biased_coin_l2323_232379


namespace NUMINAMATH_GPT_value_of_x_l2323_232378

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x_l2323_232378


namespace NUMINAMATH_GPT_hairstylist_monthly_earnings_l2323_232399

noncomputable def hairstylist_earnings_per_month : ℕ :=
  let monday_wednesday_friday_earnings : ℕ := (4 * 10) + (3 * 15) + (1 * 22);
  let tuesday_thursday_earnings : ℕ := (6 * 10) + (2 * 15) + (3 * 30);
  let weekend_earnings : ℕ := (10 * 22) + (5 * 30);
  let weekly_earnings : ℕ :=
    (monday_wednesday_friday_earnings * 3) +
    (tuesday_thursday_earnings * 2) +
    (weekend_earnings * 2);
  weekly_earnings * 4

theorem hairstylist_monthly_earnings : hairstylist_earnings_per_month = 5684 := by
  -- Assertion based on the provided problem conditions
  sorry

end NUMINAMATH_GPT_hairstylist_monthly_earnings_l2323_232399


namespace NUMINAMATH_GPT_bill_experience_l2323_232331

theorem bill_experience (j b : ℕ) 
  (h₁ : j - 5 = 3 * (b - 5)) 
  (h₂ : j = 2 * b) : b = 10 :=
sorry

end NUMINAMATH_GPT_bill_experience_l2323_232331


namespace NUMINAMATH_GPT_sector_perimeter_l2323_232347

theorem sector_perimeter (r : ℝ) (c : ℝ) (angle_deg : ℝ) (angle_rad := angle_deg * Real.pi / 180) 
  (arc_length := r * angle_rad) (P := arc_length + c)
  (h1 : r = 10) (h2 : c = 10) (h3 : angle_deg = 120) :
  P = 20 * Real.pi / 3 + 10 :=
by
  sorry

end NUMINAMATH_GPT_sector_perimeter_l2323_232347


namespace NUMINAMATH_GPT_average_price_of_initial_fruit_l2323_232336

theorem average_price_of_initial_fruit (A O : ℕ) (h1 : A + O = 10) (h2 : (40 * A + 60 * (O - 6)) / (A + O - 6) = 45) : 
  (40 * A + 60 * O) / 10 = 54 :=
by 
  sorry

end NUMINAMATH_GPT_average_price_of_initial_fruit_l2323_232336


namespace NUMINAMATH_GPT_numerator_of_first_fraction_l2323_232315

theorem numerator_of_first_fraction (y : ℝ) (h : y > 0) (x : ℝ) 
  (h_eq : (x / y) * y + (3 * y) / 10 = 0.35 * y) : x = 32 := 
by
  sorry

end NUMINAMATH_GPT_numerator_of_first_fraction_l2323_232315


namespace NUMINAMATH_GPT_gcd_markers_l2323_232307

variable (n1 n2 n3 : ℕ)

-- Let the markers Mary, Luis, and Ali bought be represented by n1, n2, and n3
def MaryMarkers : ℕ := 36
def LuisMarkers : ℕ := 45
def AliMarkers : ℕ := 75

theorem gcd_markers : Nat.gcd (Nat.gcd MaryMarkers LuisMarkers) AliMarkers = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_markers_l2323_232307


namespace NUMINAMATH_GPT_polynomial_identity_l2323_232364

theorem polynomial_identity (x : ℝ) (h₁ : x^5 - 3*x + 2 = 0) (h₂ : x ≠ 1) : 
  x^4 + x^3 + x^2 + x + 1 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_identity_l2323_232364


namespace NUMINAMATH_GPT_combined_average_age_of_fifth_graders_teachers_and_parents_l2323_232351

theorem combined_average_age_of_fifth_graders_teachers_and_parents
  (num_fifth_graders : ℕ) (avg_age_fifth_graders : ℕ)
  (num_teachers : ℕ) (avg_age_teachers : ℕ)
  (num_parents : ℕ) (avg_age_parents : ℕ)
  (h1 : num_fifth_graders = 40) (h2 : avg_age_fifth_graders = 10)
  (h3 : num_teachers = 4) (h4 : avg_age_teachers = 40)
  (h5 : num_parents = 60) (h6 : avg_age_parents = 34)
  : (num_fifth_graders * avg_age_fifth_graders + num_teachers * avg_age_teachers + num_parents * avg_age_parents) /
    (num_fifth_graders + num_teachers + num_parents) = 25 :=
by sorry

end NUMINAMATH_GPT_combined_average_age_of_fifth_graders_teachers_and_parents_l2323_232351


namespace NUMINAMATH_GPT_find_x_l2323_232346

-- Define the functions δ (delta) and φ (phi)
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- State the theorem with conditions and question, and assert the answer
theorem find_x :
  (delta ∘ phi) x = 11 → x = -5/6 := by
  intros
  sorry

end NUMINAMATH_GPT_find_x_l2323_232346


namespace NUMINAMATH_GPT_dmitriev_older_by_10_l2323_232374

-- Define the ages of each of the elders
variables (A B C D E F : ℕ)

-- The conditions provided in the problem
axiom hAlyosha : A > (A - 1)
axiom hBorya : B > (B - 2)
axiom hVasya : C > (C - 3)
axiom hGrisha : D > (D - 4)

-- Establishing an equation for the age differences leading to the proof
axiom age_sum_relation : A + B + C + D + E = (A - 1) + (B - 2) + (C - 3) + (D - 4) + F

-- We state that Dmitriev is older than Dima by 10 years
theorem dmitriev_older_by_10 : F = E + 10 :=
by
  -- sorry replaces the proof
  sorry

end NUMINAMATH_GPT_dmitriev_older_by_10_l2323_232374


namespace NUMINAMATH_GPT_ceil_minus_x_of_fractional_part_half_l2323_232392

theorem ceil_minus_x_of_fractional_part_half (x : ℝ) (hx : x - ⌊x⌋ = 1 / 2) : ⌈x⌉ - x = 1 / 2 :=
by
 sorry

end NUMINAMATH_GPT_ceil_minus_x_of_fractional_part_half_l2323_232392


namespace NUMINAMATH_GPT_problem_solution_l2323_232353

variables {a b c : ℝ}

theorem problem_solution
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) :
  (a + b) * (b + c) * (a + c) = 0 := 
sorry

end NUMINAMATH_GPT_problem_solution_l2323_232353


namespace NUMINAMATH_GPT_hours_per_day_l2323_232359

variable (m w : ℝ)
variable (h : ℕ)

-- Assume the equivalence of work done by women and men
axiom work_equiv : 3 * w = 2 * m

-- Total work done by men
def work_men := 15 * m * 21 * h
-- Total work done by women
def work_women := 21 * w * 36 * 5

-- The total work done by men and women is equal
theorem hours_per_day (h : ℕ) (w m : ℝ) (work_equiv : 3 * w = 2 * m) :
  15 * m * 21 * h = 21 * w * 36 * 5 → h = 8 :=
by
  intro H
  sorry

end NUMINAMATH_GPT_hours_per_day_l2323_232359


namespace NUMINAMATH_GPT_solve_f_log2_20_l2323_232349

noncomputable def f (x : ℝ) : ℝ :=
if -1 ≤ x ∧ x < 0 then 2^x else 0 -- Placeholder for other values

theorem solve_f_log2_20 :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 4) = f x) →
  (∀ x, -1 ≤ x ∧ x < 0 → f x = 2^x) →
  f (Real.log 20 / Real.log 2) = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_f_log2_20_l2323_232349
