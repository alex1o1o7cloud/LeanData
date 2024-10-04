import Mathlib

namespace fx_le_1_l18_18673

-- Statement
theorem fx_le_1 (x : ℝ) (h : x > 0) : (1 + Real.log x) / x ≤ 1 := 
sorry

end fx_le_1_l18_18673


namespace min_value_of_n_l18_18337

def is_prime (p : ℕ) : Prop := p ≥ 2 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_not_prime (n : ℕ) : Prop := ¬ is_prime n

def decomposable_into_primes_leq_10 (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≤ 10 ∧ q ≤ 10 ∧ n = p * q

theorem min_value_of_n : ∃ n : ℕ, is_not_prime n ∧ decomposable_into_primes_leq_10 n ∧ n = 6 :=
by
  -- The proof would go here.
  sorry

end min_value_of_n_l18_18337


namespace largest_prime_factor_1001_l18_18324

theorem largest_prime_factor_1001 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧
  (∀ q : ℕ, nat.prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l18_18324


namespace binom_9_5_eq_126_l18_18972

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l18_18972


namespace probability_no_3x3_red_square_l18_18229

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l18_18229


namespace binom_9_5_eq_126_l18_18986

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l18_18986


namespace domain_of_c_l18_18059

theorem domain_of_c (m : ℝ) :
  (∀ x : ℝ, 7*x^2 - 6*x + m ≠ 0) ↔ (m > (9 / 7)) :=
by
  -- you would typically put the proof here, but we use sorry to skip it
  sorry

end domain_of_c_l18_18059


namespace determinant_zero_l18_18966

open Matrix

variables {R : Type*} [Field R] {a b : R}

def M : Matrix (Fin 3) (Fin 3) R :=
  ![
    ![1, sin (a - b), sin a],
    ![sin (a - b), 1, sin b],
    ![sin a, sin b, 1]
  ]

theorem determinant_zero : det M = 0 :=
by
  sorry

end determinant_zero_l18_18966


namespace total_length_of_figure_2_segments_l18_18638

-- Definitions based on conditions
def rectangle_length : ℕ := 10
def rectangle_breadth : ℕ := 6
def square_side : ℕ := 4
def interior_segment : ℕ := rectangle_breadth / 2

-- Summing up the lengths of segments in Figure 2
def total_length_of_segments : ℕ :=
  square_side + 2 * rectangle_length + interior_segment

-- Mathematical proof problem statement
theorem total_length_of_figure_2_segments :
  total_length_of_segments = 27 :=
sorry

end total_length_of_figure_2_segments_l18_18638


namespace even_positive_factors_count_l18_18260

theorem even_positive_factors_count (n : ℕ) (h : n = 2^4 * 3^3 * 7) : 
  ∃ k : ℕ, k = 32 := 
by
  sorry

end even_positive_factors_count_l18_18260


namespace combined_weight_of_candles_l18_18805

theorem combined_weight_of_candles 
  (beeswax_weight_per_candle : ℕ)
  (coconut_oil_weight_per_candle : ℕ)
  (total_candles : ℕ)
  (candles_made : ℕ) 
  (total_weight: ℕ) 
  : 
  beeswax_weight_per_candle = 8 → 
  coconut_oil_weight_per_candle = 1 → 
  total_candles = 10 → 
  candles_made = total_candles - 3 →
  total_weight = candles_made * (beeswax_weight_per_candle + coconut_oil_weight_per_candle) →
  total_weight = 63 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end combined_weight_of_candles_l18_18805


namespace quadratic_roots_distinct_l18_18378

variable (a b c : ℤ)

theorem quadratic_roots_distinct (h_eq : 3 * a^2 - 3 * a - 4 = 0) : ∃ (x y : ℝ), x ≠ y ∧ (3 * x^2 - 3 * x - 4 = 0) ∧ (3 * y^2 - 3 * y - 4 = 0) := 
  sorry

end quadratic_roots_distinct_l18_18378


namespace fraction_of_students_l18_18701

theorem fraction_of_students {G B T : ℕ} (h1 : B = 2 * G) (h2 : T = G + B) (h3 : (1 / 2) * (G : ℝ) = (x : ℝ) * (T : ℝ)) : x = (1 / 6) :=
by sorry

end fraction_of_students_l18_18701


namespace jake_weight_loss_l18_18262

variable {J K L : Nat}

theorem jake_weight_loss
  (h1 : J + K = 290)
  (h2 : J = 196)
  (h3 : J - L = 2 * K) : L = 8 :=
by
  sorry

end jake_weight_loss_l18_18262


namespace value_a_plus_c_l18_18867

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c
noncomputable def g (a b c : ℝ) (x : ℝ) := c * x^2 + b * x + a

theorem value_a_plus_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c (g a b c x) = x) : a + c = -1 :=
by
  sorry

end value_a_plus_c_l18_18867


namespace binom_1294_2_l18_18364

def combination (n k : Nat) := n.choose k

theorem binom_1294_2 : combination 1294 2 = 836161 := by
  sorry

end binom_1294_2_l18_18364


namespace binomial_coefficient_9_5_l18_18996

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l18_18996


namespace initial_books_count_l18_18878

-- Definitions in conditions
def books_sold : ℕ := 42
def books_left : ℕ := 66

-- The theorem to prove the initial books count
theorem initial_books_count (initial_books : ℕ) : initial_books = books_sold + books_left :=
  by sorry

end initial_books_count_l18_18878


namespace parking_lot_problem_l18_18011

theorem parking_lot_problem :
  let total_spaces := 50
  let cars := 2
  let total_ways := total_spaces * (total_spaces - 1)
  let adjacent_ways := (total_spaces - 1) * 2
  let valid_ways := total_ways - adjacent_ways
  valid_ways = 2352 :=
by
  sorry

end parking_lot_problem_l18_18011


namespace cake_has_more_calories_l18_18477

-- Define the conditions
def cake_slices : Nat := 8
def cake_calories_per_slice : Nat := 347
def brownie_count : Nat := 6
def brownie_calories_per_brownie : Nat := 375

-- Define the total calories for the cake and the brownies
def total_cake_calories : Nat := cake_slices * cake_calories_per_slice
def total_brownie_calories : Nat := brownie_count * brownie_calories_per_brownie

-- Prove the difference in calories
theorem cake_has_more_calories : 
  total_cake_calories - total_brownie_calories = 526 :=
by
  sorry

end cake_has_more_calories_l18_18477


namespace speed_of_second_car_l18_18759

theorem speed_of_second_car
  (t : ℝ) (d : ℝ) (d1 : ℝ) (d2 : ℝ) (v : ℝ)
  (h1 : t = 2.5)
  (h2 : d = 175)
  (h3 : d1 = 25 * t)
  (h4 : d2 = v * t)
  (h5 : d1 + d2 = d) :
  v = 45 := by sorry

end speed_of_second_car_l18_18759


namespace geometric_sequence_a4_l18_18136

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ (n : ℕ), a (n + 1) = a n * r

def a_3a_5_is_64 (a : ℕ → ℝ) : Prop :=
  a 3 * a 5 = 64

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h1 : is_geometric_sequence a) (h2 : a_3a_5_is_64 a) : a 4 = 8 ∨ a 4 = -8 :=
by
  sorry

end geometric_sequence_a4_l18_18136


namespace num_arrangements_l18_18317

def volunteers : Finset ℕ := {1, 2, 3, 4}
def counties : Finset ℕ := {1, 2, 3}

noncomputable def count_arrangements : ℕ :=
  (volunteers.card.choose 2) * (volunteers.card - 2).choose 1 * 1 / 2 * county_permutations.count

theorem num_arrangements :
  count_arrangements = 36 := 
by 
  sorry

end num_arrangements_l18_18317


namespace extreme_points_of_f_range_of_a_l18_18725

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≥ -1 then Real.log (x + 1) + a * (x^2 - x) 
  else 0

theorem extreme_points_of_f (a : ℝ) :
  (a < 0 → ∃ x, f a x = 0) ∧
  (0 ≤ a ∧ a ≤ 8/9 → ∃! x, f a x = 0) ∧
  (a > 8/9 → ∃ x₁ x₂, x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end extreme_points_of_f_range_of_a_l18_18725


namespace symmetric_points_x_axis_l18_18395

theorem symmetric_points_x_axis (m n : ℤ) :
  (-4, m - 3) = (2 * n, -1) → (m = 2 ∧ n = -2) :=
by
  sorry

end symmetric_points_x_axis_l18_18395


namespace length_BC_value_tan_2B_l18_18399

noncomputable def length {α : Type*} [inner_product_space ℝ α] (a b : α) : ℝ := real.sqrt (inner_product (a - b) (a - b))

-- Given Conditions
variables 
  (A B C : ℝ × ℝ)
  (AB_length : ℝ := 6)
  (AC_length : ℝ := 3 * real.sqrt 2)
  (dot_product_AB_AC : ℝ := -18)

-- Definitions based on the given conditions
def AB : ℝ := AB_length
def AC : ℝ := AC_length
def dot_prod : ℝ := dot_product_AB_AC

-- Main Statements
theorem length_BC (BC_length : ℝ) : length A B C = 3 * real.sqrt 10 :=
begin
  sorry
end

theorem value_tan_2B (tan_2B : ℝ) : tan 2 * (real.atan (length B C / length A B)) = 3 / 4 :=
begin
  sorry
end

end length_BC_value_tan_2B_l18_18399


namespace person_age_is_30_l18_18771

-- Definitions based on the conditions
def age (x : ℕ) := x
def age_5_years_hence (x : ℕ) := x + 5
def age_5_years_ago (x : ℕ) := x - 5

-- The main theorem to prove
theorem person_age_is_30 (x : ℕ) (h : 3 * age_5_years_hence x - 3 * age_5_years_ago x = age x) : x = 30 :=
by
  sorry

end person_age_is_30_l18_18771


namespace simplify_polynomial_l18_18762

variable (x : ℝ)

theorem simplify_polynomial :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 6*x^3 =
  6*x^3 - x^2 + 23*x - 3 :=
by
  sorry

end simplify_polynomial_l18_18762


namespace binom_9_5_eq_126_l18_18990

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l18_18990


namespace notepad_duration_l18_18641

theorem notepad_duration (a8_papers_per_a4 : ℕ)
  (a4_papers : ℕ)
  (notes_per_day : ℕ)
  (notes_per_side : ℕ) :
  a8_papers_per_a4 = 16 →
  a4_papers = 8 →
  notes_per_day = 15 →
  notes_per_side = 2 →
  (a4_papers * a8_papers_per_a4 * notes_per_side) / notes_per_day = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end notepad_duration_l18_18641


namespace sum_of_real_y_values_l18_18079

theorem sum_of_real_y_values :
  (∀ (x y : ℝ), x^2 + x^2 * y^2 + x^2 * y^4 = 525 ∧ x + x * y + x * y^2 = 35 → y = 1 / 2 ∨ y = 2) →
    (1 / 2 + 2 = 5 / 2) :=
by
  intro h
  have := h (1 / 2)
  have := h 2
  sorry  -- Proof steps showing 1/2 and 2 are the solutions, leading to the sum 5/2

end sum_of_real_y_values_l18_18079


namespace find_numbers_satisfying_conditions_l18_18604

theorem find_numbers_satisfying_conditions (x y z : ℝ)
(h1 : x + y + z = 11 / 18)
(h2 : 1 / x + 1 / y + 1 / z = 18)
(h3 : 2 / y = 1 / x + 1 / z) :
x = 1 / 9 ∧ y = 1 / 6 ∧ z = 1 / 3 :=
sorry

end find_numbers_satisfying_conditions_l18_18604


namespace maurice_needs_7_letters_l18_18606
noncomputable def prob_no_job (n : ℕ) : ℝ := (4 / 5) ^ n

theorem maurice_needs_7_letters :
  ∃ n : ℕ, (prob_no_job n) ≤ 1 / 4 ∧ n = 7 :=
by
  sorry

end maurice_needs_7_letters_l18_18606


namespace ratio_a_to_d_l18_18090

theorem ratio_a_to_d (a b c d : ℚ) 
  (h1 : a / b = 5 / 4) 
  (h2 : b / c = 2 / 3) 
  (h3 : c / d = 3 / 5) : 
  a / d = 1 / 2 :=
sorry

end ratio_a_to_d_l18_18090


namespace triangle_inequality_inequality_equality_condition_l18_18565

variable (a b c : ℝ)

-- indicating triangle inequality conditions
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end triangle_inequality_inequality_equality_condition_l18_18565


namespace correct_conclusions_l18_18019

-- Definitions based on conditions
def condition_1 (x : ℝ) : Prop := x ≠ 0 → x + |x| > 0
def condition_3 (a b c : ℝ) (Δ : ℝ) : Prop := a > 0 ∧ Δ ≤ 0 ∧ Δ = b^2 - 4*a*c → 
  ∀ x, a*x^2 + b*x + c ≥ 0

-- Stating the proof problem
theorem correct_conclusions (x a b c Δ : ℝ) :
  (condition_1 x) ∧ (condition_3 a b c Δ) :=
sorry

end correct_conclusions_l18_18019


namespace tan_five_pi_over_four_l18_18029

-- Define the question to prove
theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l18_18029


namespace determinant_simplifies_to_zero_l18_18968

theorem determinant_simplifies_to_zero (a b : ℝ) :
  matrix.det ![
    ![1, real.sin (a - b), real.sin a],
    ![real.sin (a - b), 1, real.sin b],
    ![real.sin a, real.sin b, 1]
  ] = 0 := 
by
  sorry

end determinant_simplifies_to_zero_l18_18968


namespace solve_cubic_equation_l18_18431

theorem solve_cubic_equation (x : ℝ) (h : 4 * x^(1/3) - 2 * (x / x^(2/3)) = 7 + x^(1/3)) : x = 343 := by
  sorry

end solve_cubic_equation_l18_18431


namespace ribbon_per_gift_l18_18861

theorem ribbon_per_gift
  (total_ribbon : ℕ)
  (number_of_gifts : ℕ)
  (ribbon_left : ℕ)
  (used_ribbon := total_ribbon - ribbon_left)
  (ribbon_per_gift := used_ribbon / number_of_gifts)
  (h_total : total_ribbon = 18)
  (h_gifts : number_of_gifts = 6)
  (h_left : ribbon_left = 6) :
  ribbon_per_gift = 2 := by
  sorry

end ribbon_per_gift_l18_18861


namespace units_digit_product_composites_l18_18610

theorem units_digit_product_composites :
  (4 * 6 * 8 * 9 * 10) % 10 = 0 :=
sorry

end units_digit_product_composites_l18_18610


namespace simplify_and_evaluate_l18_18572

-- Problem statement with conditions translated into Lean
theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 5 + 1) :
  (a / (a^2 - 2*a + 1)) / (1 + 1 / (a - 1)) = Real.sqrt 5 / 5 := sorry

end simplify_and_evaluate_l18_18572


namespace project_completion_days_l18_18316

theorem project_completion_days (A B C : ℝ) (h1 : 1/A + 1/B = 1/2) (h2 : 1/B + 1/C = 1/4) (h3 : 1/C + 1/A = 1/2.4) : A = 3 :=
by
sorry

end project_completion_days_l18_18316


namespace complement_A_in_U_l18_18389

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}

theorem complement_A_in_U : (U \ A) = {x | -1 <= x ∧ x <= 3} :=
by
  sorry

end complement_A_in_U_l18_18389


namespace sri_lanka_population_problem_l18_18338

theorem sri_lanka_population_problem
  (P : ℝ)
  (h1 : 0.85 * (0.9 * P) = 3213) :
  P = 4200 :=
sorry

end sri_lanka_population_problem_l18_18338


namespace remainder_103_107_div_11_l18_18766

theorem remainder_103_107_div_11 :
  (103 * 107) % 11 = 10 :=
by
  sorry

end remainder_103_107_div_11_l18_18766


namespace cistern_water_depth_l18_18343

theorem cistern_water_depth 
  (l w a : ℝ)
  (hl : l = 8)
  (hw : w = 6)
  (ha : a = 83) :
  ∃ d : ℝ, 48 + 28 * d = 83 :=
by
  use 1.25
  sorry

end cistern_water_depth_l18_18343


namespace hamburgers_sold_last_week_l18_18945

theorem hamburgers_sold_last_week (avg_hamburgers_per_day : ℕ) (days_in_week : ℕ) 
    (h_avg : avg_hamburgers_per_day = 9) (h_days : days_in_week = 7) : 
    avg_hamburgers_per_day * days_in_week = 63 :=
by
  -- Avg hamburgers per day times number of days
  sorry

end hamburgers_sold_last_week_l18_18945


namespace count_friendly_sets_with_max_8_l18_18178

def cards : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

def is_friendly_set (S : Finset ℕ) : Prop :=
  S.card = 4 ∧ S.max' (by simp [Finset.nonempty_of_card_eq_four]) ≤ 8 ∧ S.sum id = 24

def friendly_sets : Finset (Finset ℕ) :=
  (cards.powerset.filter is_friendly_set)

theorem count_friendly_sets_with_max_8 : friendly_sets.card = 8 :=
sorry

end count_friendly_sets_with_max_8_l18_18178


namespace initial_distance_is_18_l18_18482

-- Step a) Conditions and Definitions
def distance_covered (v t d : ℝ) : Prop := 
  d = v * t

def increased_speed_time (v t d : ℝ) : Prop := 
  d = (v + 1) * (3 / 4 * t)

def decreased_speed_time (v t d : ℝ) : Prop := 
  d = (v - 1) * (t + 3)

-- Step c) Mathematically Equivalent Proof Problem
theorem initial_distance_is_18 (v t d : ℝ) 
  (h1 : distance_covered v t d) 
  (h2 : increased_speed_time v t d) 
  (h3 : decreased_speed_time v t d) : 
  d = 18 :=
sorry

end initial_distance_is_18_l18_18482


namespace initial_orchid_bushes_l18_18312

def final_orchid_bushes : ℕ := 35
def orchid_bushes_to_be_planted : ℕ := 13

theorem initial_orchid_bushes :
  final_orchid_bushes - orchid_bushes_to_be_planted = 22 :=
by
  sorry

end initial_orchid_bushes_l18_18312


namespace hausdorff_space_with_sigma_compact_subspaces_is_countable_l18_18154

noncomputable def is_sigma_compact (X : Type*) [topological_space X] :=
∃ (A : ℕ → set X), (∀ n, is_compact (A n)) ∧ (X = ⋃ n, A n)

theorem hausdorff_space_with_sigma_compact_subspaces_is_countable 
  {X : Type*} [topological_space X] [T2_space X]
  (h : ∀ (Y : set X), is_sigma_compact Y) : countable (set.univ : set X) :=
sorry

end hausdorff_space_with_sigma_compact_subspaces_is_countable_l18_18154


namespace days_kept_first_book_l18_18033

def cost_per_day : ℝ := 0.50
def total_days_in_may : ℝ := 31
def total_cost_paid : ℝ := 41

theorem days_kept_first_book (x : ℝ) : 0.50 * x + 2 * (0.50 * 31) = 41 → x = 20 :=
by sorry

end days_kept_first_book_l18_18033


namespace imaginary_part_of_1_minus_2i_l18_18720

def i := Complex.I

theorem imaginary_part_of_1_minus_2i : Complex.im (1 - 2 * i) = -2 :=
by
  sorry

end imaginary_part_of_1_minus_2i_l18_18720


namespace count_primes_with_digit_three_l18_18682

def is_digit_three (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := Prime n

def primes_with_digit_three_count (lim : ℕ) (count : ℕ) : Prop :=
  ∀ n < lim, is_digit_three n → is_prime n → count = 9

theorem count_primes_with_digit_three (lim : ℕ) (count : ℕ) :
  primes_with_digit_three_count 150 9 := 
by
  sorry

end count_primes_with_digit_three_l18_18682


namespace probability_no_3x3_red_square_l18_18221

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l18_18221


namespace scenario1_winner_scenario2_winner_l18_18025

def optimal_play_winner1 (n : ℕ) (start_by_anna : Bool := true) : String :=
  if n % 6 = 0 then "Balázs"
  else "Anna"

def optimal_play_winner2 (n : ℕ) (start_by_anna : Bool := true) : String :=
  if n % 4 = 0 then "Balázs"
  else "Anna"

theorem scenario1_winner:
  optimal_play_winner1 39 true = "Balázs" :=
by 
  sorry

theorem scenario2_winner:
  optimal_play_winner2 39 true = "Anna" :=
by
  sorry

end scenario1_winner_scenario2_winner_l18_18025


namespace count_five_digit_multiples_of_5_l18_18096

-- Define the range of five-digit positive integers
def lower_bound : ℕ := 10000
def upper_bound : ℕ := 99999

-- Define the divisor
def divisor : ℕ := 5

-- Define the count of multiples of 5 in the range
def count_multiples_of_5 : ℕ :=
  (upper_bound / divisor) - (lower_bound / divisor) + 1

-- The main statement: The number of five-digit multiples of 5 is 18000
theorem count_five_digit_multiples_of_5 : count_multiples_of_5 = 18000 :=
  sorry

end count_five_digit_multiples_of_5_l18_18096


namespace function_solution_l18_18672

theorem function_solution (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = sorry) → f a = sorry → (a = 1 ∨ a = -1) :=
by
  intros hfa hfb
  sorry

end function_solution_l18_18672


namespace quadratic_inequality_solution_l18_18837

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l18_18837


namespace mass_of_man_l18_18938

-- Definitions based on problem conditions
def boat_length : ℝ := 8
def boat_breadth : ℝ := 3
def sinking_height : ℝ := 0.01
def water_density : ℝ := 1000

-- Mass of the man to be proven
theorem mass_of_man : boat_density * (boat_length * boat_breadth * sinking_height) = 240 :=
by
  sorry

end mass_of_man_l18_18938


namespace length_of_bridge_l18_18044

theorem length_of_bridge (speed : ℝ) (time_min : ℝ) (length : ℝ)
  (h_speed : speed = 5) (h_time : time_min = 15) :
  length = 1250 :=
sorry

end length_of_bridge_l18_18044


namespace sam_total_yellow_marbles_l18_18882

def sam_original_yellow_marbles : Float := 86.0
def sam_yellow_marbles_given_by_joan : Float := 25.0

theorem sam_total_yellow_marbles : sam_original_yellow_marbles + sam_yellow_marbles_given_by_joan = 111.0 := by
  sorry

end sam_total_yellow_marbles_l18_18882


namespace smallest_five_angles_sum_l18_18502

-- Definition of Q(x) and related conditions
def Q (x : ℂ) : ℂ := (x^20 - 1)^2 / (x - 1)^2 - x^19

-- Definition of angles and their sum
def angle_sum (angles: Fin 5 → ℝ) : ℝ :=
  angles 0 + angles 1 + angles 2 + angles 3 + angles 4

-- The proof goal
theorem smallest_five_angles_sum :
  ∃ (β : Fin 5 → ℝ), 
    (∀ i : Fin 5, 0 < β i ∧ β i < 1) ∧
    angle_sum β = 183/399 :=
by
  sorry

end smallest_five_angles_sum_l18_18502


namespace correct_fraction_simplification_l18_18686

theorem correct_fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (∀ (c d : ℝ), (c ≠ d) → (a+2 = c → b+2 = d → (a+2)/d ≠ a/b))
  ∧ (∀ (e f : ℝ), (e ≠ f) → (a-2 = e → b-2 = f → (a-2)/f ≠ a/b))
  ∧ (∀ (g h : ℝ), (g ≠ h) → (a^2 = g → b^2 = h → a^2/h ≠ a/b))
  ∧ (a / b = ( (1/2)*a / (1/2)*b )) := 
sorry

end correct_fraction_simplification_l18_18686


namespace zoo_ticket_sales_l18_18777

theorem zoo_ticket_sales (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : K = 202 :=
by {
  sorry
}

end zoo_ticket_sales_l18_18777


namespace solve_equation_l18_18585

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l18_18585


namespace ratio_area_circle_to_triangle_l18_18890

theorem ratio_area_circle_to_triangle (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
    (π * r) / (h + r) = (π * r ^ 2) / (r * (h + r)) := sorry

end ratio_area_circle_to_triangle_l18_18890


namespace count_digit_9_l18_18116

theorem count_digit_9 (n : ℕ) (h : n = 1000) : 
  (Finset.range n).sum (λ x, (Nat.digits 10 x).count 9) = 300 :=
by
  sorry

end count_digit_9_l18_18116


namespace quadratic_inequality_solution_l18_18840

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l18_18840


namespace curve_distance_bound_l18_18084

/--
Given the point A on the curve y = e^x and point B on the curve y = ln(x),
prove that |AB| >= a always holds if and only if a <= sqrt(2).
-/
theorem curve_distance_bound {A B : ℝ × ℝ} (a : ℝ)
  (hA : A.2 = Real.exp A.1) (hB : B.2 = Real.log B.1) :
  (dist A B ≥ a) ↔ (a ≤ Real.sqrt 2) :=
sorry

end curve_distance_bound_l18_18084


namespace candy_selection_l18_18339

theorem candy_selection (m n : ℕ) (h1 : Nat.gcd m n = 1) (h2 : m = 1) (h3 : n = 5) :
  m + n = 6 := by
  sorry

end candy_selection_l18_18339


namespace simplify_expression_l18_18737

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
    (Real.sqrt (4 + ( (x^3 - 2) / (3 * x) ) ^ 2)) = 
    (Real.sqrt (x^6 - 4 * x^3 + 36 * x^2 + 4) / (3 * x)) :=
by sorry

end simplify_expression_l18_18737


namespace product_of_integers_P_Q_R_S_l18_18517

theorem product_of_integers_P_Q_R_S (P Q R S : ℤ)
  (h1 : 0 < P) (h2 : 0 < Q) (h3 : 0 < R) (h4 : 0 < S)
  (h_sum : P + Q + R + S = 50)
  (h_rel : P + 4 = Q - 4 ∧ P + 4 = R * 3 ∧ P + 4 = S / 3) :
  P * Q * R * S = 43 * 107 * 75 * 225 / 1536 := 
by { sorry }

end product_of_integers_P_Q_R_S_l18_18517


namespace solve_for_q_l18_18119

theorem solve_for_q :
  ∀ (q : ℕ), 16^15 = 4^q → q = 30 :=
by
  intro q
  intro h
  sorry

end solve_for_q_l18_18119


namespace compare_A_B_C_l18_18612

-- Define the expressions A, B, and C
def A : ℚ := (2010 / 2009) + (2010 / 2011)
def B : ℚ := (2010 / 2011) + (2012 / 2011)
def C : ℚ := (2011 / 2010) + (2011 / 2012)

-- The statement asserting A is the greatest
theorem compare_A_B_C : A > B ∧ A > C := by
  sorry

end compare_A_B_C_l18_18612


namespace complement_A_intersection_B_l18_18823

open Set

variable (α : Type*) [LinearOrder α] [TopologicalSpace α]

def A : Set α := { x : α | x > 3 }
def B : Set α := { x : α | 2 < x ∧ x < 4 }

theorem complement_A_intersection_B :
  (compl A) ∩ B = { x : α | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end complement_A_intersection_B_l18_18823


namespace perpendicular_lines_slope_l18_18129

theorem perpendicular_lines_slope {a : ℝ} :
  (∃ (a : ℝ), (∀ x y : ℝ, x + 2 * y - 1 = 0 → a * x - y - 1 = 0) ∧ (a * (-1 / 2)) = -1) → a = 2 :=
by sorry

end perpendicular_lines_slope_l18_18129


namespace incorrect_statement_D_l18_18870

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2

theorem incorrect_statement_D :
  (∃ T : ℝ, ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x : ℝ, f (π / 2 + x) = f (π / 2 - x)) ∧
  (f (π / 2 + π / 4) = 0) ∧ ¬(∀ x : ℝ, (π / 2 < x ∧ x < π) → f x < f (x - 0.1)) := by
  sorry

end incorrect_statement_D_l18_18870


namespace power_function_m_value_l18_18441

theorem power_function_m_value (m : ℝ) (h : m^2 - 2 * m + 2 = 1) : m = 1 :=
by
  sorry

end power_function_m_value_l18_18441


namespace time_to_reach_ticket_window_l18_18925

-- Define the conditions as per the problem
def rate_kit : ℕ := 2 -- feet per minute (rate)
def remaining_distance : ℕ := 210 -- feet

-- Goal: To prove the time required to reach the ticket window is 105 minutes
theorem time_to_reach_ticket_window : remaining_distance / rate_kit = 105 :=
by sorry

end time_to_reach_ticket_window_l18_18925


namespace apple_price_36_kgs_l18_18050

theorem apple_price_36_kgs (l q : ℕ) 
  (H1 : ∀ n, n ≤ 30 → ∀ n', n' ≤ 30 → l * n' = 100)
  (H2 : 30 * l + 3 * q = 168) : 
  30 * l + 6 * q = 186 :=
by {
  sorry
}

end apple_price_36_kgs_l18_18050


namespace solve_fractional_eq_l18_18593

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l18_18593


namespace inequality_solution_addition_eq_seven_l18_18164

theorem inequality_solution_addition_eq_seven (b c : ℝ) :
  (∀ x : ℝ, -5 < 2 * x - 3 ∧ 2 * x - 3 < 5 → -1 < x ∧ x < 4) →
  (∀ x : ℝ, -x^2 + b * x + c = 0 ↔ (x = -1 ∨ x = 4)) →
  b + c = 7 :=
by
  intro h1 h2
  sorry

end inequality_solution_addition_eq_seven_l18_18164


namespace find_number_l18_18918

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end find_number_l18_18918


namespace slope_of_line_passes_through_points_l18_18126

theorem slope_of_line_passes_through_points :
  let k := (2 + Real.sqrt 3 - 2) / (4 - 1)
  k = Real.sqrt 3 / 3 :=
by
  sorry

end slope_of_line_passes_through_points_l18_18126


namespace solve_equation_l18_18600

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l18_18600


namespace additional_charge_is_correct_l18_18714

noncomputable def additional_charge_per_segment (initial_fee : ℝ) (total_distance : ℝ) (total_charge : ℝ) (segment_length : ℝ) : ℝ :=
  let segments := total_distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  charge_for_distance / segments

theorem additional_charge_is_correct :
  additional_charge_per_segment 2.0 3.6 5.15 (2/5) = 0.35 :=
by
  sorry

end additional_charge_is_correct_l18_18714


namespace jesse_money_left_after_mall_l18_18553

theorem jesse_money_left_after_mall :
  ∀ (initial_amount novel_cost lunch_cost total_spent remaining_amount : ℕ),
    initial_amount = 50 →
    novel_cost = 7 →
    lunch_cost = 2 * novel_cost →
    total_spent = novel_cost + lunch_cost →
    remaining_amount = initial_amount - total_spent →
    remaining_amount = 29 :=
by
  intros initial_amount novel_cost lunch_cost total_spent remaining_amount
  sorry

end jesse_money_left_after_mall_l18_18553


namespace part_a_l18_18618

theorem part_a (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x + y = 2) : 
  (1 / x + 1 / y) ≤ (1 / x^2 + 1 / y^2) := 
sorry

end part_a_l18_18618


namespace calculation_result_l18_18204

theorem calculation_result : (18 * 23 - 24 * 17) / 3 + 5 = 7 :=
by
  sorry

end calculation_result_l18_18204


namespace total_cats_l18_18876

theorem total_cats (current_cats : ℕ) (additional_cats : ℕ) (h1 : current_cats = 11) (h2 : additional_cats = 32):
  current_cats + additional_cats = 43 :=
by
  -- We state the given conditions:
  -- current_cats = 11
  -- additional_cats = 32
  -- We need to prove:
  -- current_cats + additional_cats = 43
  sorry

end total_cats_l18_18876


namespace solve_fractional_eq_l18_18594

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l18_18594


namespace find_fraction_l18_18373

-- Definition of the fractions and the given condition
def certain_fraction : ℚ := 1 / 2
def given_ratio : ℚ := 2 / 6
def target_fraction : ℚ := 1 / 3

-- The proof problem to verify
theorem find_fraction (X : ℚ) : (X / given_ratio) = 1 ↔ X = target_fraction :=
by
  sorry

end find_fraction_l18_18373


namespace tangent_and_parallel_l18_18745

noncomputable def parabola1 (x : ℝ) (b1 c1 : ℝ) : ℝ := -x^2 + b1 * x + c1
noncomputable def parabola2 (x : ℝ) (b2 c2 : ℝ) : ℝ := -x^2 + b2 * x + c2
noncomputable def parabola3 (x : ℝ) (b3 c3 : ℝ) : ℝ := x^2 + b3 * x + c3

theorem tangent_and_parallel (b1 b2 b3 c1 c2 c3 : ℝ) :
  (b3 - b1)^2 = 8 * (c3 - c1) → (b3 - b2)^2 = 8 * (c3 - c2) →
  ((b2^2 - b1^2 + 2 * b3 * (b2 - b1)) / (4 * (b2 - b1))) = 
  ((4 * (c1 - c2) - 2 * b3 * (b1 - b2)) / (2 * (b2 - b1))) :=
by
  intros h1 h2
  sorry

end tangent_and_parallel_l18_18745


namespace digit_9_occurrences_1_to_1000_l18_18105

theorem digit_9_occurrences_1_to_1000 : 
  (finset.range 1000).sum (λ n, (n.digits 10).count 9) = 300 := 
by
  sorry

end digit_9_occurrences_1_to_1000_l18_18105


namespace final_price_is_correct_l18_18446

-- Define the original price and percentages as constants
def original_price : ℝ := 160
def increase_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25

-- Calculate increased price
def increased_price : ℝ := original_price * (1 + increase_percentage)
-- Calculate the discount on the increased price
def discount_amount : ℝ := increased_price * discount_percentage
-- Calculate final price after discount
def final_price : ℝ := increased_price - discount_amount

-- Statement of the theorem: prove final price is $150
theorem final_price_is_correct : final_price = 150 :=
by
  -- Proof would go here
  sorry

end final_price_is_correct_l18_18446


namespace pete_backward_speed_l18_18425

variable (p b t s : ℝ)  -- speeds of Pete, backward walk, Tracy, and Susan respectively

-- Given conditions
axiom h1 : p / t = 1 / 4      -- Pete walks on his hands at a quarter speed of Tracy's cartwheeling
axiom h2 : t = 2 * s          -- Tracy cartwheels twice as fast as Susan walks
axiom h3 : b = 3 * s          -- Pete walks backwards three times faster than Susan
axiom h4 : p = 2              -- Pete walks on his hands at 2 miles per hour

-- Prove Pete's backward walking speed is 12 miles per hour
theorem pete_backward_speed : b = 12 :=
by
  sorry

end pete_backward_speed_l18_18425


namespace Billy_age_l18_18643

-- Defining the ages of Billy, Joe, and Sam
variable (B J S : ℕ)

-- Conditions given in the problem
axiom Billy_twice_Joe : B = 2 * J
axiom sum_BJ_three_times_S : B + J = 3 * S
axiom Sam_age : S = 27

-- Statement to prove
theorem Billy_age : B = 54 :=
by
  sorry

end Billy_age_l18_18643


namespace calculate_ratio_l18_18562

theorem calculate_ratio (l m n : ℝ) :
  let D := (l + 1, 1, 1)
  let E := (1, m + 1, 1)
  let F := (1, 1, n + 1)
  let AB_sq := 4 * ((n - m) ^ 2)
  let AC_sq := 4 * ((l - n) ^ 2)
  let BC_sq := 4 * ((m - l) ^ 2)
  (AB_sq + AC_sq + BC_sq + 3) / (l^2 + m^2 + n^2 + 3) = 8 := by
  sorry

end calculate_ratio_l18_18562


namespace no_b_gt_4_such_that_143b_is_square_l18_18060

theorem no_b_gt_4_such_that_143b_is_square :
  ∀ (b : ℕ), 4 < b → ¬ ∃ (n : ℕ), b^2 + 4 * b + 3 = n^2 :=
by sorry

end no_b_gt_4_such_that_143b_is_square_l18_18060


namespace cost_of_banana_l18_18285

theorem cost_of_banana (B : ℝ) (apples bananas oranges total_pieces total_cost : ℝ) 
  (h1 : apples = 12) (h2 : bananas = 4) (h3 : oranges = 4) 
  (h4 : total_pieces = 20) (h5 : total_cost = 40)
  (h6 : 2 * apples + 3 * oranges + bananas * B = total_cost)
  : B = 1 :=
by
  sorry

end cost_of_banana_l18_18285


namespace nth_term_correct_l18_18356

noncomputable def nth_term (a b : ℝ) (n : ℕ) : ℝ :=
  (-1 : ℝ)^n * (2 * n - 1) * b / a^n

theorem nth_term_correct (a b : ℝ) (n : ℕ) (h : 0 < a) : 
  nth_term a b n = (-1 : ℝ)^↑n * (2 * n - 1) * b / a^n :=
by sorry

end nth_term_correct_l18_18356


namespace rebs_bank_save_correct_gamma_bank_save_correct_tisi_bank_save_correct_btv_bank_save_correct_l18_18027

-- Define the available rubles
def available_funds : ℝ := 150000

-- Define the total expenses for the vacation
def total_expenses : ℝ := 201200

-- Define interest rates and compounding formulas for each bank
def rebs_bank_annual_rate : ℝ := 0.036
def rebs_bank_monthly_rate : ℝ := rebs_bank_annual_rate / 12
def rebs_bank_amount (initial : ℝ) (months : ℕ) : ℝ :=
  initial * (1 + rebs_bank_monthly_rate) ^ months

def gamma_bank_annual_rate : ℝ := 0.045
def gamma_bank_amount (initial : ℝ) (months : ℕ) : ℝ :=
  initial * (1 + gamma_bank_annual_rate * (months / 12))

def tisi_bank_annual_rate : ℝ := 0.0312
def tisi_bank_quarterly_rate : ℝ := tisi_bank_annual_rate / 4
def tisi_bank_amount (initial : ℝ) (quarters : ℕ) : ℝ :=
  initial * (1 + tisi_bank_quarterly_rate) ^ quarters

def btv_bank_monthly_rate : ℝ := 0.0025
def btv_bank_amount (initial : ℝ) (months : ℕ) : ℝ :=
  initial * (1 + btv_bank_monthly_rate) ^ months

-- Calculate the interest earned for each bank
def rebs_bank_interest : ℝ := rebs_bank_amount available_funds 6 - available_funds
def gamma_bank_interest : ℝ := gamma_bank_amount available_funds 6 - available_funds
def tisi_bank_interest : ℝ := tisi_bank_amount available_funds 2 - available_funds
def btv_bank_interest : ℝ := btv_bank_amount available_funds 6 - available_funds

-- Calculate the remaining amount to be saved from salary for each bank
def rebs_bank_save : ℝ := total_expenses - available_funds - rebs_bank_interest
def gamma_bank_save : ℝ := total_expenses - available_funds - gamma_bank_interest
def tisi_bank_save : ℝ := total_expenses - available_funds - tisi_bank_interest
def btv_bank_save : ℝ := total_expenses - available_funds - btv_bank_interest

-- Prove the calculated save amounts
theorem rebs_bank_save_correct : rebs_bank_save = 48479.67 := by sorry
theorem gamma_bank_save_correct : gamma_bank_save = 47825.00 := by sorry
theorem tisi_bank_save_correct : tisi_bank_save = 48850.87 := by sorry
theorem btv_bank_save_correct : btv_bank_save = 48935.89 := by sorry

end rebs_bank_save_correct_gamma_bank_save_correct_tisi_bank_save_correct_btv_bank_save_correct_l18_18027


namespace prob_144_eq_1_div_72_l18_18465

open Probability
open MeasureTheory
open Classical

noncomputable def probability_abc_144 : ℝ :=
  let outcomes := ({1, 2, 3, 4, 5, 6} : Finset ℕ)
  let prod := {d : Finset ℕ × Finset ℕ × Finset ℕ | 
    (d.1.1 * d.1.2 * d.2) = 144} 
  (∑ in prod, (1 / 6 : ℝ) ^ 3)

theorem prob_144_eq_1_div_72 : probability_abc_144 = 1 / 72 :=
sorry

end prob_144_eq_1_div_72_l18_18465


namespace fraction_increases_by_3_l18_18127

-- Define initial fraction
def initial_fraction (x y : ℕ) : ℕ :=
  2 * x * y / (3 * x - y)

-- Define modified fraction
def modified_fraction (x y : ℕ) (m : ℕ) : ℕ :=
  2 * (m * x) * (m * y) / (m * (3 * x) - (m * y))

-- State the theorem to prove the value of modified fraction is 3 times the initial fraction
theorem fraction_increases_by_3 (x y : ℕ) : modified_fraction x y 3 = 3 * initial_fraction x y :=
by sorry

end fraction_increases_by_3_l18_18127


namespace candy_left_l18_18075

theorem candy_left (d : ℕ) (s : ℕ) (ate : ℕ) (h_d : d = 32) (h_s : s = 42) (h_ate : ate = 35) : d + s - ate = 39 :=
by
  -- d, s, and ate are given as natural numbers
  -- h_d, h_s, and h_ate are the provided conditions
  -- The goal is to prove d + s - ate = 39
  sorry

end candy_left_l18_18075


namespace find_savings_l18_18469

-- Definitions of given conditions
def income : ℕ := 10000
def ratio_income_expenditure : ℕ × ℕ := (10, 8)

-- Proving the savings based on given conditions
theorem find_savings (income : ℕ) (ratio_income_expenditure : ℕ × ℕ) :
  let expenditure := (ratio_income_expenditure.2 * income) / ratio_income_expenditure.1
  let savings := income - expenditure
  savings = 2000 :=
by
  sorry

end find_savings_l18_18469


namespace evaluate_expression_l18_18061

theorem evaluate_expression :
  (3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^7) = (6^5 + 3^7) :=
sorry

end evaluate_expression_l18_18061


namespace count_nine_in_1_to_1000_l18_18108

def count_nine_units : ℕ := 100
def count_nine_tens : ℕ := 100
def count_nine_hundreds : ℕ := 100

theorem count_nine_in_1_to_1000 :
  count_nine_units + count_nine_tens + count_nine_hundreds = 300 :=
by
  sorry

end count_nine_in_1_to_1000_l18_18108


namespace value_of_abc_l18_18691

noncomputable def f (a b c x : ℝ) := a * x^2 + b * x + c
noncomputable def f_inv (a b c x : ℝ) := c * x^2 + b * x + a

-- The main theorem statement
theorem value_of_abc (a b c : ℝ) (h : ∀ x : ℝ, f a b c (f_inv a b c x) = x) : a + b + c = 1 :=
sorry

end value_of_abc_l18_18691


namespace algebraic_expression_value_l18_18698

theorem algebraic_expression_value (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 - 5 * a + 2 = 0) (h3 : b^2 - 5 * b + 2 = 0) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -13 / 2 := by
  sorry

end algebraic_expression_value_l18_18698


namespace range_of_alpha_plus_beta_l18_18243

theorem range_of_alpha_plus_beta (α β : ℝ) (h1 : 0 < α - β) (h2 : α - β < π) (h3 : 0 < α + 2 * β) (h4 : α + 2 * β < π) :
  0 < α + β ∧ α + β < π :=
sorry

end range_of_alpha_plus_beta_l18_18243


namespace dice_probability_theorem_l18_18513

def at_least_three_same_value_probability (num_dice : ℕ) (num_sides : ℕ) : ℚ :=
  if num_dice = 5 ∧ num_sides = 10 then
    -- Calculating the probability
    (81 / 10000) + (9 / 20000) + (1 / 10000)
  else
    0

theorem dice_probability_theorem :
  at_least_three_same_value_probability 5 10 = 173 / 20000 :=
by
  sorry

end dice_probability_theorem_l18_18513


namespace Wayne_blocks_count_l18_18015

-- Statement of the proof problem
theorem Wayne_blocks_count (initial_blocks additional_blocks total_blocks : ℕ) 
  (h1 : initial_blocks = 9) 
  (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 := 
by 
  -- proof would go here, but we will use sorry for now
  sorry

end Wayne_blocks_count_l18_18015


namespace average_speed_correct_l18_18334

-- Define the conditions as constants
def distance (D : ℝ) := D
def first_segment_speed := 60 -- km/h
def second_segment_speed := 24 -- km/h
def third_segment_speed := 48 -- km/h

-- Define the function that calculates average speed
noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / first_segment_speed
  let t2 := (D / 3) / second_segment_speed
  let t3 := (D / 3) / third_segment_speed
  let total_time := t1 + t2 + t3
  let total_distance := D
  total_distance / total_time

-- Prove that the average speed is 720 / 19 km/h
theorem average_speed_correct (D : ℝ) (hD : D > 0) : 
  average_speed D = 720 / 19 :=
by
  sorry

end average_speed_correct_l18_18334


namespace find_a_l18_18397

theorem find_a (a : ℝ) (h1 : 1 < a) (h2 : 1 + a = 3) : a = 2 :=
sorry

end find_a_l18_18397


namespace division_of_decimals_l18_18018

theorem division_of_decimals : 0.25 / 0.005 = 50 := 
by
  sorry

end division_of_decimals_l18_18018


namespace number_of_connections_l18_18065

-- Definitions based on conditions
def switches : ℕ := 15
def connections_per_switch : ℕ := 4

-- Theorem statement proving the correct number of connections
theorem number_of_connections : switches * connections_per_switch / 2 = 30 := by
  sorry

end number_of_connections_l18_18065


namespace find_teachers_and_students_l18_18481

-- Mathematical statements corresponding to the problem conditions
def teachers_and_students_found (x y : ℕ) : Prop :=
  (y = 30 * x + 7) ∧ (31 * x = y + 1)

-- The theorem we need to prove
theorem find_teachers_and_students : ∃ x y, teachers_and_students_found x y ∧ x = 8 ∧ y = 247 :=
  by
    sorry

end find_teachers_and_students_l18_18481


namespace probability_of_selecting_meiqi_l18_18626

def four_red_bases : List String := ["Meiqi", "Wangcunkou", "Zhulong", "Xiaoshun"]

theorem probability_of_selecting_meiqi :
  (1 / 4 : ℝ) = 1 / (four_red_bases.length : ℝ) :=
  by sorry

end probability_of_selecting_meiqi_l18_18626


namespace coin_toss_sequences_count_l18_18360

theorem coin_toss_sequences_count :
  ∃ (seqs : ℕ), 
    (seqs = 27720) ∧ 
    (∃ n, n = 20) ∧
    (∃ hh, hh = 3) ∧
    (∃ ht, ht = 4) ∧
    (∃ th, th = 5) ∧
    (∃ tt, tt = 7) ∧
    (seqs = (Nat.choose (3 + 5 - 1) (5 - 1)) * (Nat.choose (7 + 6 - 1) (6 - 1))) :=
begin
  sorry
end

end coin_toss_sequences_count_l18_18360


namespace quadratic_root_range_l18_18741

/-- 
  Define the quadratic function y = ax^2 + bx + c for given values.
  Show that there exists x_1 in the interval (-1, 0) such that y = 0.
-/
theorem quadratic_root_range {a b c : ℝ} (h : a ≠ 0) 
  (h_minus3 : a * (-3)^2 + b * (-3) + c = -11)
  (h_minus2 : a * (-2)^2 + b * (-2) + c = -5)
  (h_minus1 : a * (-1)^2 + b * (-1) + c = -1)
  (h_0 : a * 0^2 + b * 0 + c = 1)
  (h_1 : a * 1^2 + b * 1 + c = 1) : 
  ∃ x1 : ℝ, -1 < x1 ∧ x1 < 0 ∧ a * x1^2 + b * x1 + c = 0 :=
sorry

end quadratic_root_range_l18_18741


namespace solve_equation_l18_18586

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l18_18586


namespace triangle_perimeter_correct_l18_18670

noncomputable def triangle_perimeter (a b x : ℕ) : ℕ := a + b + x

theorem triangle_perimeter_correct :
  ∀ (x : ℕ), (2 + 4 + x = 10) → 2 < x → x < 6 → (∀ k : ℕ, k = x → k % 2 = 0) → triangle_perimeter 2 4 x = 10 :=
by
  intros x h1 h2 h3
  rw [triangle_perimeter, h1]
  sorry

end triangle_perimeter_correct_l18_18670


namespace cross_section_area_parallel_l18_18665

-- Definitions
def regular_tetrahedron (S A B C : Point) (SO height : ℝ) (BC length : ℝ) : Prop :=
  (SO = 3) ∧ (BC = 6) -- This captures the height and the base side length conditions

def perpendicular (A S B C O' : Point) (segment_length: (A - O').length): Prop :=
  (S.is_perpendicular_to_plane (A, segment_length))

-- Given constants for the problem
noncomputable def S : Point := sorry
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def O' : Point := sorry
noncomputable def P : Point := sorry

-- Representation of the length ratio
def length_ratio (AP PO' : ℝ) : Prop :=
  (AP / PO' = 8)

-- Main theorem stating the area of the cross-section parallel to base through P
theorem cross_section_area_parallel (S A B C O' P : Point) (height side_length : ℝ) (AP PO' : ℝ)
  (h1 : regular_tetrahedron S A B C height side_length)
  (h2 : perpendicular A S B C O' AP)
  (h3 : length_ratio AP PO') :
  cross_section_area_parallel_to_base_through (S A B C O' P) = real.sqrt 3 := 
  sorry

end cross_section_area_parallel_l18_18665


namespace ticket_savings_percentage_l18_18202

theorem ticket_savings_percentage:
  ∀ (P : ℝ), 9 * P - 6 * P = (1 / 3) * (9 * P) ∧ (33 + 1/3) = 100 * (3 * P / (9 * P)) := 
by
  intros P
  sorry

end ticket_savings_percentage_l18_18202


namespace seashells_solution_l18_18754

def seashells_problem (T : ℕ) : Prop :=
  T + 13 = 50 → T = 37

theorem seashells_solution : seashells_problem 37 :=
by
  intro h
  sorry

end seashells_solution_l18_18754


namespace billy_tickets_used_l18_18497

-- Definitions for the number of rides and cost per ride
def ferris_wheel_rides : Nat := 7
def bumper_car_rides : Nat := 3
def ticket_per_ride : Nat := 5

-- Total number of rides
def total_rides : Nat := ferris_wheel_rides + bumper_car_rides

-- Total tickets used
def total_tickets : Nat := total_rides * ticket_per_ride

-- Theorem stating the number of tickets Billy used in total
theorem billy_tickets_used : total_tickets = 50 := by
  sorry

end billy_tickets_used_l18_18497


namespace geometric_series_sum_l18_18791

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1 / 3) : 
  (∑' n : ℕ, a * r ^ n) = 3 / 2 := 
by
  sorry

end geometric_series_sum_l18_18791


namespace find_original_price_of_petrol_l18_18784

open Real

noncomputable def original_price_of_petrol (P : ℝ) : Prop :=
  ∀ G : ℝ, 
  (G * P = 300) ∧ 
  ((G + 7) * 0.85 * P = 300) → 
  P = 7.56

-- Theorems should ideally be defined within certain scopes or namespaces
theorem find_original_price_of_petrol (P : ℝ) : original_price_of_petrol P :=
  sorry

end find_original_price_of_petrol_l18_18784


namespace minimum_value_f_l18_18660

theorem minimum_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : (∃ x y : ℝ, (x + y + x * y) ≥ - (9 / 8) ∧ 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :=
sorry

end minimum_value_f_l18_18660


namespace number_of_pencils_l18_18024

theorem number_of_pencils (P L : ℕ) (h1 : (P : ℚ) / L = 5 / 6) (h2 : L = P + 6) : L = 36 :=
sorry

end number_of_pencils_l18_18024


namespace ratio_of_potatoes_l18_18712

def total_potatoes : ℕ := 24
def number_of_people : ℕ := 3
def potatoes_per_person : ℕ := 8
def total_each_person : ℕ := potatoes_per_person * number_of_people

theorem ratio_of_potatoes :
  total_potatoes = total_each_person → (potatoes_per_person : ℚ) / (potatoes_per_person : ℚ) = 1 :=
by
  sorry

end ratio_of_potatoes_l18_18712


namespace total_books_in_week_l18_18729

def books_read (n : ℕ) : ℕ :=
  if n = 0 then 2 -- day 1 (indexed by 0)
  else if n = 1 then 2 -- day 2
  else 2 + n -- starting from day 3 (indexed by 2)

-- Summing the books read from day 1 to day 7 (indexed from 0 to 6)
theorem total_books_in_week : (List.sum (List.map books_read [0, 1, 2, 3, 4, 5, 6])) = 29 := by
  sorry

end total_books_in_week_l18_18729


namespace max_bag_weight_is_50_l18_18751

noncomputable def max_weight_allowed (people bags_per_person more_bags_allowed total_weight : ℕ) : ℝ := 
  total_weight / ((people * bags_per_person) + more_bags_allowed)

theorem max_bag_weight_is_50 : ∀ (people bags_per_person more_bags_allowed total_weight : ℕ), 
  people = 6 → 
  bags_per_person = 5 → 
  more_bags_allowed = 90 → 
  total_weight = 6000 →
  max_weight_allowed people bags_per_person more_bags_allowed total_weight = 50 := 
by 
  sorry

end max_bag_weight_is_50_l18_18751


namespace measure_C_in_triangle_ABC_l18_18401

noncomputable def measure_angle_C (A B : ℝ) : ℝ := 180 - (A + B)

theorem measure_C_in_triangle_ABC (A B : ℝ) (h : A + B = 150) : measure_angle_C A B = 30 := 
by
  unfold measure_angle_C
  rw h
  norm_num
  sorry

end measure_C_in_triangle_ABC_l18_18401


namespace find_f_neg_five_half_l18_18865

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
else if 0 ≤ x + 2 ∧ x + 2 ≤ 1 then 2 * (x + 2) * (1 - (x + 2))
     else -2 * abs x * (1 - abs x)

theorem find_f_neg_five_half (x : ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 2) = f x)
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)) : 
  f (-5 / 2) = -1 / 2 :=
  by sorry

end find_f_neg_five_half_l18_18865


namespace count_five_digit_multiples_of_five_l18_18094

theorem count_five_digit_multiples_of_five : 
  ∃ (n : ℕ), n = 18000 ∧ (∀ x, 10000 ≤ x ∧ x ≤ 99999 ∧ x % 5 = 0 ↔ ∃ k, 10000 ≤ 5 * k ∧ 5 * k ≤ 99999) :=
by
  sorry

end count_five_digit_multiples_of_five_l18_18094


namespace soaking_time_l18_18498

theorem soaking_time : 
  let grass_stain_time := 4 
  let marinara_stain_time := 7 
  let num_grass_stains := 3 
  let num_marinara_stains := 1 
  in 
  num_grass_stains * grass_stain_time + num_marinara_stains * marinara_stain_time = 19 := 
by 
  sorry

end soaking_time_l18_18498


namespace intersection_of_A_B_C_l18_18415

-- Define the sets A, B, and C as given conditions:
def A : Set ℕ := { x | ∃ n : ℕ, x = 2 * n }
def B : Set ℕ := { x | ∃ n : ℕ, x = 3 * n }
def C : Set ℕ := { x | ∃ n : ℕ, x = n ^ 2 }

-- Prove that A ∩ B ∩ C = { x | ∃ n : ℕ, x = 36 * n ^ 2 }
theorem intersection_of_A_B_C :
  (A ∩ B ∩ C) = { x | ∃ n : ℕ, x = 36 * n ^ 2 } :=
sorry

end intersection_of_A_B_C_l18_18415


namespace solution_set_for_fractional_inequality_l18_18072

theorem solution_set_for_fractional_inequality :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} :=
by
  sorry

end solution_set_for_fractional_inequality_l18_18072


namespace small_ball_rubber_bands_l18_18290

theorem small_ball_rubber_bands (S : ℕ) 
    (large_ball : ℕ := 300) 
    (initial_rubber_bands : ℕ := 5000) 
    (small_balls : ℕ := 22) 
    (large_balls : ℕ := 13) :
  (small_balls * S + large_balls * large_ball = initial_rubber_bands) → S = 50 := by
    sorry

end small_ball_rubber_bands_l18_18290


namespace solve_equation_l18_18885

theorem solve_equation (x : ℝ) : (x + 1) * (x - 3) = 5 ↔ (x = 4 ∨ x = -2) :=
by
  sorry

end solve_equation_l18_18885


namespace problem_statement_l18_18387

-- Definitions based on conditions
def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem problem_statement : f (g 3) = 120 ∧ f 3 = 8 :=
by sorry

end problem_statement_l18_18387


namespace min_value_x_plus_y_l18_18521

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 4 / x + 9 / y = 1) : x + y = 25 :=
sorry

end min_value_x_plus_y_l18_18521


namespace probability_no_3by3_red_grid_correct_l18_18218

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l18_18218


namespace lighting_effect_improves_l18_18139

theorem lighting_effect_improves (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
    (a + m) / (b + m) > a / b := 
sorry

end lighting_effect_improves_l18_18139


namespace binom_9_5_eq_126_l18_18989

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l18_18989


namespace sandbox_sand_weight_l18_18629

theorem sandbox_sand_weight
  (side_len : ℝ) 
  (side_len_eq : side_len = 40)
  (bag_weight : ℝ)
  (bag_weight_eq : bag_weight = 30)
  (bag_coverage : ℝ)
  (bag_coverage_eq : bag_coverage = 80) :
  let area := side_len * side_len in
  let num_bags := area / bag_coverage in
  let total_weight := num_bags * bag_weight in
  total_weight = 600 := by
  sorry

end sandbox_sand_weight_l18_18629


namespace digit_9_appears_301_times_l18_18103

def countDigitOccurrences (d : ℕ) (a b : ℕ) : ℕ :=
  list.count d (list.join (list.map toDigits (list.range' a (b - a + 1))))

theorem digit_9_appears_301_times :
  countDigitOccurrences 9 1 1000 = 301 := by
  sorry

end digit_9_appears_301_times_l18_18103


namespace prove_a_pow_minus_b_l18_18583

-- Definitions of conditions
variables (x a b : ℝ)

def condition_1 : Prop := x - a > 2
def condition_2 : Prop := 2 * x - b < 0
def solution_set_condition : Prop := -1 < x ∧ x < 1
def derived_a : Prop := a + 2 = -1
def derived_b : Prop := b / 2 = 1

-- The main theorem to prove
theorem prove_a_pow_minus_b (h1 : condition_1 x a) (h2 : condition_2 x b) (h3 : solution_set_condition x) (ha : derived_a a) (hb : derived_b b) : a^(-b) = (1 / 9) :=
by
  sorry

end prove_a_pow_minus_b_l18_18583


namespace geometric_sequence_common_ratio_l18_18078

theorem geometric_sequence_common_ratio 
  (a1 q : ℝ) 
  (h : (a1 * (1 - q^3) / (1 - q)) + 3 * (a1 * (1 - q^2) / (1 - q)) = 0) : 
  q = -1 :=
sorry

end geometric_sequence_common_ratio_l18_18078


namespace quadratic_positive_difference_l18_18299
open Real

theorem quadratic_positive_difference :
  ∀ (x : ℝ), (2*x^2 - 7*x + 1 = x + 31) →
    (abs ((2 + sqrt 19) - (2 - sqrt 19)) = 2 * sqrt 19) :=
by intros x h
   sorry

end quadratic_positive_difference_l18_18299


namespace number_of_teachers_students_possible_rental_plans_economical_plan_l18_18479

-- Definitions of conditions

def condition1 (x y : ℕ) : Prop := y - 30 * x = 7
def condition2 (x y : ℕ) : Prop := 31 * x - y = 1
def capacity_condition (m : ℕ) : Prop := 35 * m + 30 * (8 - m) ≥ 255
def rental_fee_condition (m : ℕ) : Prop := 400 * m + 320 * (8 - m) ≤ 3000

-- Problems to prove

theorem number_of_teachers_students (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 8 ∧ y = 247 := 
by sorry

theorem possible_rental_plans (m : ℕ) (h_cap : capacity_condition m) (h_fee : rental_fee_condition m) : m = 3 ∨ m = 4 ∨ m = 5 := 
by sorry

theorem economical_plan (m : ℕ) (h_fee : rental_fee_condition 3) (h_fee_alt1 : rental_fee_condition 4) (h_fee_alt2 : rental_fee_condition 5) : m = 3 := 
by sorry

end number_of_teachers_students_possible_rental_plans_economical_plan_l18_18479


namespace avg_speed_is_65_l18_18471

theorem avg_speed_is_65
  (speed1: ℕ) (speed2: ℕ) (time1: ℕ) (time2: ℕ)
  (h_speed1: speed1 = 85)
  (h_speed2: speed2 = 45)
  (h_time1: time1 = 1)
  (h_time2: time2 = 1) :
  (speed1 + speed2) / (time1 + time2) = 65 := by
  sorry

end avg_speed_is_65_l18_18471


namespace mrs_hilt_hot_dogs_l18_18728

theorem mrs_hilt_hot_dogs (cost_per_hotdog total_cost : ℕ) (h1 : cost_per_hotdog = 50) (h2 : total_cost = 300) :
  total_cost / cost_per_hotdog = 6 := by
  sorry

end mrs_hilt_hot_dogs_l18_18728


namespace exists_v_satisfying_equation_l18_18657

noncomputable def custom_operation (v : ℝ) : ℝ :=
  v - (v / 3) + Real.sin v

theorem exists_v_satisfying_equation :
  ∃ v : ℝ, custom_operation (custom_operation v) = 24 := 
sorry

end exists_v_satisfying_equation_l18_18657


namespace sum_of_coordinates_l18_18537

theorem sum_of_coordinates {g h : ℝ → ℝ} 
  (h₁ : g 4 = 5)
  (h₂ : ∀ x, h x = (g x)^2) :
  4 + h 4 = 29 := by
  sorry

end sum_of_coordinates_l18_18537


namespace prize_expectation_l18_18855

theorem prize_expectation :
  let total_people := 100
  let envelope_percentage := 0.4
  let grand_prize_prob := 0.1
  let second_prize_prob := 0.2
  let consolation_prize_prob := 0.3
  let people_with_envelopes := total_people * envelope_percentage
  let grand_prize_winners := people_with_envelopes * grand_prize_prob
  let second_prize_winners := people_with_envelopes * second_prize_prob
  let consolation_prize_winners := people_with_envelopes * consolation_prize_prob
  let empty_envelopes := people_with_envelopes - (grand_prize_winners + second_prize_winners + consolation_prize_winners)
  grand_prize_winners = 4 ∧
  second_prize_winners = 8 ∧
  consolation_prize_winners = 12 ∧
  empty_envelopes = 16 := by
  sorry

end prize_expectation_l18_18855


namespace nathan_paintable_area_l18_18422

def total_paintable_area (rooms : ℕ) (length width height : ℕ) (non_paintable_area : ℕ) : ℕ :=
  let wall_area := 2 * (length * height + width * height)
  rooms * (wall_area - non_paintable_area)

theorem nathan_paintable_area :
  total_paintable_area 4 15 12 9 75 = 1644 :=
by sorry

end nathan_paintable_area_l18_18422


namespace sodas_purchasable_l18_18435

namespace SodaPurchase

variable {D C : ℕ}

theorem sodas_purchasable (D C : ℕ) : (3 * (4 * D) / 5 + 5 * C / 15) = (36 * D + 5 * C) / 15 := 
  sorry

end SodaPurchase

end sodas_purchasable_l18_18435


namespace man_l18_18185

theorem man's_speed_against_current :
  ∀ (V_down V_c V_m V_up : ℝ),
    (V_down = 15) →
    (V_c = 2.8) →
    (V_m = V_down - V_c) →
    (V_up = V_m - V_c) →
    V_up = 9.4 :=
by
  intros V_down V_c V_m V_up
  intros hV_down hV_c hV_m hV_up
  sorry

end man_l18_18185


namespace jeremy_remaining_money_l18_18274

-- Conditions as definitions
def computer_cost : ℝ := 3000
def accessories_cost : ℝ := 0.1 * computer_cost
def initial_money : ℝ := 2 * computer_cost

-- Theorem statement for the proof problem
theorem jeremy_remaining_money : initial_money - computer_cost - accessories_cost = 2700 := by
  -- Proof will be added here
  sorry

end jeremy_remaining_money_l18_18274


namespace fraction_value_l18_18887

variable {x y : ℝ}

theorem fraction_value (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x - 3 * y) / (x + 2 * y) = 3) :
  (x - 2 * y) / (2 * x + 3 * y) = 11 / 15 :=
  sorry

end fraction_value_l18_18887


namespace largest_of_choices_l18_18020

theorem largest_of_choices :
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  A < D ∧ B < D ∧ C < D ∧ E < D :=
by
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  sorry

end largest_of_choices_l18_18020


namespace parallelogram_area_l18_18773

-- Definitions
def base_cm : ℕ := 22
def height_cm : ℕ := 21

-- Theorem statement
theorem parallelogram_area : base_cm * height_cm = 462 := by
  sorry

end parallelogram_area_l18_18773


namespace probability_no_3x3_red_square_l18_18228

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l18_18228


namespace probability_no_3x3_red_square_l18_18214

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l18_18214


namespace part_I_part_II_l18_18132

namespace VectorProblems

def vector_a : ℝ × ℝ := (3, 2)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (4, 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem part_I (m : ℝ) :
  let u := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
  let v := (4 * m + vector_b.1, m + vector_b.2)
  dot_product u v > 0 →
  m ≠ 4 / 7 →
  m > -1 / 2 :=
sorry

theorem part_II (k : ℝ) :
  let u := (vector_a.1 + 4 * k, vector_a.2 + k)
  let v := (2 * vector_b.1 - vector_a.1, 2 * vector_b.2 - vector_a.2)
  dot_product u v = 0 →
  k = -11 / 18 :=
sorry

end VectorProblems

end part_I_part_II_l18_18132


namespace sqrt_computation_l18_18795

theorem sqrt_computation : 
  Real.sqrt ((35 * 34 * 33 * 32) + Nat.factorial 4) = 1114 := by
sorry

end sqrt_computation_l18_18795


namespace haley_number_of_shirts_l18_18092

-- Define the given information
def washing_machine_capacity : ℕ := 7
def total_loads : ℕ := 5
def number_of_sweaters : ℕ := 33
def number_of_shirts := total_loads * washing_machine_capacity - number_of_sweaters

-- The statement that needs to be proven
theorem haley_number_of_shirts : number_of_shirts = 2 := by
  sorry

end haley_number_of_shirts_l18_18092


namespace initial_percentage_of_water_l18_18348

variable (P : ℚ) -- Initial percentage of water

theorem initial_percentage_of_water (h : P / 100 * 40 + 5 = 9) : P = 10 := 
  sorry

end initial_percentage_of_water_l18_18348


namespace perfect_square_of_expression_l18_18904

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end perfect_square_of_expression_l18_18904


namespace total_hamburgers_sold_is_63_l18_18947

-- Define the average number of hamburgers sold each day
def avg_hamburgers_per_day : ℕ := 9

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total hamburgers sold in a week
def total_hamburgers_sold : ℕ := avg_hamburgers_per_day * days_in_week

-- Prove that the total hamburgers sold in a week is 63
theorem total_hamburgers_sold_is_63 : total_hamburgers_sold = 63 :=
by
  sorry

end total_hamburgers_sold_is_63_l18_18947


namespace total_profit_l18_18175

theorem total_profit (a_cap b_cap : ℝ) (a_profit : ℝ) (a_share b_share : ℝ) (P : ℝ) :
  a_cap = 15000 ∧ b_cap = 25000 ∧ a_share = 0.10 ∧ a_profit = 4200 →
  a_share * P + (a_cap / (a_cap + b_cap)) * (1 - a_share) * P = a_profit →
  P = 9600 :=
by
  intros h1 h2
  have h3 : a_share * P + (a_cap / (a_cap + b_cap)) * (1 - a_share) * P = a_profit := h2
  sorry

end total_profit_l18_18175


namespace tim_morning_running_hours_l18_18315

theorem tim_morning_running_hours 
  (runs_per_week : ℕ) 
  (total_hours_per_week : ℕ) 
  (runs_per_day : ℕ → ℕ) 
  (hrs_per_day_morning_evening_equal : ∀ (d : ℕ), runs_per_day d = runs_per_week * total_hours_per_week / runs_per_week) 
  (hrs_per_day : ℕ) 
  (hrs_per_morning : ℕ) 
  (hrs_per_evening : ℕ) 
  : hrs_per_morning = 1 :=
by 
  -- Given conditions
  have hrs_per_day := total_hours_per_week / runs_per_week
  have hrs_per_morning_evening := hrs_per_day / 2
  -- Conclusion
  sorry

end tim_morning_running_hours_l18_18315


namespace maximum_value_conditions_l18_18245

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem maximum_value_conditions (x_0 : ℝ) (h_max : ∀ x : ℝ, f x ≤ f x_0) :
    f x_0 = x_0 ∧ f x_0 < 1 / 2 :=
by
  sorry

end maximum_value_conditions_l18_18245


namespace arithmetic_sequence_a10_l18_18145

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : (S 9) / 9 - (S 5) / 5 = 4)
  (hSn : ∀ n, S n = n * (2 + (n - 1) / 2 * (a 2 - a 1) )) : 
  a 10 = 20 := 
sorry

end arithmetic_sequence_a10_l18_18145


namespace percentage_of_other_investment_l18_18955

theorem percentage_of_other_investment (investment total_interest interest_5 interest_other percentage_other : ℝ) 
  (h1 : investment = 18000)
  (h2 : interest_5 = 6000 * 0.05)
  (h3 : total_interest = 660)
  (h4 : percentage_other / 100 * (investment - 6000) = 360) : 
  percentage_other = 3 :=
by
  sorry

end percentage_of_other_investment_l18_18955


namespace number_division_l18_18920

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l18_18920


namespace count_digit_9_in_range_l18_18117

theorem count_digit_9_in_range :
  (∑ n in List.finRange 1000, (List.some (Nat.digits 10 (n+1)).count 9) 0) = 300 := by
  sorry

end count_digit_9_in_range_l18_18117


namespace largest_angle_convex_hexagon_l18_18894

theorem largest_angle_convex_hexagon : 
  ∃ x : ℝ, (x-3) + (x-2) + (x-1) + x + (x+1) + (x+2) = 720 → (x + 2) = 122.5 :=
by 
  intros,
  sorry

end largest_angle_convex_hexagon_l18_18894


namespace probability_no_3x3_red_square_l18_18215

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l18_18215


namespace number_division_l18_18921

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l18_18921


namespace range_of_a_l18_18850

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - a| < 4) → -1 < a ∧ a < 7 :=
  sorry

end range_of_a_l18_18850


namespace larger_integer_is_24_l18_18898

theorem larger_integer_is_24 {x : ℤ} (h1 : ∃ x, 4 * x = x + 6) :
  ∃ y, y = 4 * x ∧ y = 24 := by
  sorry

end larger_integer_is_24_l18_18898


namespace ice_cream_volume_l18_18000

-- Definitions based on Conditions
def radius_cone : Real := 3 -- radius at the opening of the cone
def height_cone : Real := 12 -- height of the cone

-- The proof statement
theorem ice_cream_volume :
  (1 / 3 * Real.pi * radius_cone^2 * height_cone) + (4 / 3 * Real.pi * radius_cone^3) = 72 * Real.pi := by
  sorry

end ice_cream_volume_l18_18000


namespace C_plus_D_l18_18649

theorem C_plus_D (D C : ℚ) (h1 : ∀ x : ℚ, (Dx - 17) / ((x - 2) * (x - 4)) = C / (x - 2) + 4 / (x - 4))
  (h2 : ∀ x : ℚ, (x - 2) * (x - 4) = x^2 - 6 * x + 8) :
  C + D = 8.5 := sorry

end C_plus_D_l18_18649


namespace Jesse_remaining_money_l18_18550

-- Define the conditions
def initial_money := 50
def novel_cost := 7
def lunch_cost := 2 * novel_cost
def total_spent := novel_cost + lunch_cost

-- Define the remaining money after spending
def remaining_money := initial_money - total_spent

-- Prove that the remaining money is $29
theorem Jesse_remaining_money : remaining_money = 29 := 
by
  sorry

end Jesse_remaining_money_l18_18550


namespace compute_combination_l18_18984

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l18_18984


namespace stuffed_animal_tickets_correct_l18_18957

-- Define the total tickets spent
def total_tickets : ℕ := 14

-- Define the tickets spent on the hat
def hat_tickets : ℕ := 2

-- Define the tickets spent on the yoyo
def yoyo_tickets : ℕ := 2

-- Define the tickets spent on the stuffed animal
def stuffed_animal_tickets : ℕ := total_tickets - (hat_tickets + yoyo_tickets)

-- The theorem we want to prove.
theorem stuffed_animal_tickets_correct :
  stuffed_animal_tickets = 10 :=
by
  sorry

end stuffed_animal_tickets_correct_l18_18957


namespace expenditure_increase_l18_18580

theorem expenditure_increase
  (current_expenditure : ℝ)
  (future_expenditure : ℝ)
  (years : ℕ)
  (r : ℝ)
  (h₁ : current_expenditure = 1000)
  (h₂ : future_expenditure = 2197)
  (h₃ : years = 3)
  (h₄ : future_expenditure = current_expenditure * (1 + r / 100) ^ years) :
  r = 30 :=
sorry

end expenditure_increase_l18_18580


namespace nines_appear_600_times_l18_18111

-- Define a function to count the number of nines in a single digit place
def count_nines_in_place (place_value : Nat) (range_start : Nat) (range_end : Nat) : Nat :=
  (range_end - range_start + 1) / 10 * place_value

-- Define a function to count the total number of nines in all places from 1 to 1000
def total_nines_1_to_1000 : Nat :=
  let counts_per_place := 3 * count_nines_in_place 100 1 1000
  counts_per_place

theorem nines_appear_600_times :
  total_nines_1_to_1000 = 600 :=
by
  -- Using specific step counts calculated in the problem
  have units_place := count_nines_in_place 100 1 1000
  have tens_place := units_place
  have hundreds_place := units_place
  have thousands_place := 0
  have total_nines := units_place + tens_place + hundreds_place + thousands_place
  sorry

end nines_appear_600_times_l18_18111


namespace probability_of_collinear_dots_l18_18544

theorem probability_of_collinear_dots (dots : ℕ) (rows : ℕ) (columns : ℕ) (choose : ℕ → ℕ → ℕ) :
  dots = 20 ∧ rows = 5 ∧ columns = 4 ∧ choose 20 4 = 4845 → 
  (∃ sets_of_collinear_dots : ℕ, sets_of_collinear_dots = 20 ∧ 
   ∃ probability : ℚ,  probability = 4 / 969) :=
by
  sorry

end probability_of_collinear_dots_l18_18544


namespace alice_bob_meet_l18_18355

/--
Alice and Bob play a game on a circle divided into 18 equally-spaced points.
Alice moves 7 points clockwise per turn, and Bob moves 13 points counterclockwise.
Prove that they will meet at the same point after 9 turns.
-/
theorem alice_bob_meet : ∃ k : ℕ, k = 9 ∧ (7 * k) % 18 = (18 - 13 * k) % 18 :=
by
  sorry

end alice_bob_meet_l18_18355


namespace correct_calculation_result_l18_18329

-- Define the conditions in Lean
variable (num : ℤ) (mistake_mult : ℤ) (result : ℤ)
variable (h_mistake : mistake_mult = num * 10) (h_result : result = 50)

-- The statement we want to prove
theorem correct_calculation_result 
  (h_mistake : mistake_mult = num * 10) 
  (h_result : result = 50) 
  (h_num_correct : num = result / 10) :
  (20 / num = 4) := sorry

end correct_calculation_result_l18_18329


namespace original_number_of_people_l18_18423

theorem original_number_of_people (x : ℕ) (h1 : x - x / 3 + (x / 3) * 3/4 = x * 1/4 + 15) : x = 30 :=
sorry

end original_number_of_people_l18_18423


namespace circumscribed_circle_area_l18_18746

theorem circumscribed_circle_area (x y c : ℝ)
  (h1 : x + y + c = 24)
  (h2 : x * y = 48)
  (h3 : x^2 + y^2 = c^2) :
  ∃ R : ℝ, (x + y + 2 * R = 24) ∧ (π * R^2 = 25 * π) := 
sorry

end circumscribed_circle_area_l18_18746


namespace find_original_petrol_price_l18_18783

noncomputable def original_petrol_price (P : ℝ) : Prop :=
  let original_amount := 300 / P in
  let reduced_price := 0.85 * P in
  let new_amount := 300 / reduced_price in
  new_amount = original_amount + 7

theorem find_original_petrol_price (P : ℝ) (h : original_petrol_price P) : P ≈ 45 / 5.95 :=
by {
  sorry
}

end find_original_petrol_price_l18_18783


namespace leak_empties_tank_in_30_hours_l18_18470

-- Define the known rates based on the problem conditions
def rate_pipe_a : ℚ := 1 / 12
def combined_rate : ℚ := 1 / 20

-- Define the rate at which the leak empties the tank
def rate_leak : ℚ := rate_pipe_a - combined_rate

-- Define the time it takes for the leak to empty the tank
def time_to_empty_tank : ℚ := 1 / rate_leak

-- The theorem that needs to be proved
theorem leak_empties_tank_in_30_hours : time_to_empty_tank = 30 :=
sorry

end leak_empties_tank_in_30_hours_l18_18470


namespace Jesse_remaining_money_l18_18549

-- Define the conditions
def initial_money := 50
def novel_cost := 7
def lunch_cost := 2 * novel_cost
def total_spent := novel_cost + lunch_cost

-- Define the remaining money after spending
def remaining_money := initial_money - total_spent

-- Prove that the remaining money is $29
theorem Jesse_remaining_money : remaining_money = 29 := 
by
  sorry

end Jesse_remaining_money_l18_18549


namespace iter_f_eq_l18_18278

namespace IteratedFunction

def f (n : ℕ) (x : ℕ) : ℕ :=
  if 2 * x <= n then
    2 * x
  else
    2 * n - 2 * x + 1

def iter_f (n m : ℕ) (x : ℕ) : ℕ :=
  (Nat.iterate (f n) m) x

variables (n m : ℕ) (S : Fin n.succ → Fin n.succ)

theorem iter_f_eq (h : iter_f n m 1 = 1) (k : Fin n.succ) :
  iter_f n m k = k := by
  sorry

end IteratedFunction

end iter_f_eq_l18_18278


namespace missing_number_l18_18068

theorem missing_number (x : ℝ) (h : 0.72 * 0.43 + x * 0.34 = 0.3504) : x = 0.12 :=
by sorry

end missing_number_l18_18068


namespace count_nine_in_1_to_1000_l18_18107

def count_nine_units : ℕ := 100
def count_nine_tens : ℕ := 100
def count_nine_hundreds : ℕ := 100

theorem count_nine_in_1_to_1000 :
  count_nine_units + count_nine_tens + count_nine_hundreds = 300 :=
by
  sorry

end count_nine_in_1_to_1000_l18_18107


namespace shortest_paths_ratio_l18_18484

theorem shortest_paths_ratio (k n : ℕ) (h : k > 0):
  let paths_along_AB := Nat.choose (k * n + n - 1) (n - 1)
  let paths_along_AD := Nat.choose (k * n + n - 1) k * n - 1
  paths_along_AD = k * paths_along_AB :=
by sorry

end shortest_paths_ratio_l18_18484


namespace binom_9_5_eq_126_l18_18971

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l18_18971


namespace slices_left_for_Era_l18_18800

def total_burgers : ℕ := 5
def slices_per_burger : ℕ := 8

def first_friend_slices : ℕ := 3
def second_friend_slices : ℕ := 8
def third_friend_slices : ℕ := 5
def fourth_friend_slices : ℕ := 11
def fifth_friend_slices : ℕ := 6

def total_slices : ℕ := total_burgers * slices_per_burger
def slices_given_to_friends : ℕ := first_friend_slices + second_friend_slices + third_friend_slices + fourth_friend_slices + fifth_friend_slices

theorem slices_left_for_Era : total_slices - slices_given_to_friends = 7 :=
by
  rw [total_slices, slices_given_to_friends]
  exact Eq.refl 7

#reduce slices_left_for_Era

end slices_left_for_Era_l18_18800


namespace choose_correct_graph_l18_18453

noncomputable def appropriate_graph : String :=
  let bar_graph := "Bar graph"
  let pie_chart := "Pie chart"
  let line_graph := "Line graph"
  let freq_dist_graph := "Frequency distribution graph"
  
  if (bar_graph = "Bar graph") ∧ (pie_chart = "Pie chart") ∧ (line_graph = "Line graph") ∧ (freq_dist_graph = "Frequency distribution graph") then
    "Line graph"
  else
    sorry

theorem choose_correct_graph :
  appropriate_graph = "Line graph" :=
by
  sorry

end choose_correct_graph_l18_18453


namespace identify_INPUT_statement_l18_18048

/-- Definition of the PRINT statement --/
def is_PRINT_statement (s : String) : Prop := s = "PRINT"

/-- Definition of the INPUT statement --/
def is_INPUT_statement (s : String) : Prop := s = "INPUT"

/-- Definition of the IF statement --/
def is_IF_statement (s : String) : Prop := s = "IF"

/-- Definition of the WHILE statement --/
def is_WHILE_statement (s : String) : Prop := s = "WHILE"

/-- Proof statement that the INPUT statement is the one for input --/
theorem identify_INPUT_statement (s : String) (h1 : is_PRINT_statement "PRINT") (h2: is_INPUT_statement "INPUT") (h3 : is_IF_statement "IF") (h4 : is_WHILE_statement "WHILE") : s = "INPUT" :=
sorry

end identify_INPUT_statement_l18_18048


namespace twenty_twenty_third_term_l18_18733

def sequence_denominator (n : ℕ) : ℕ :=
  2 * n - 1

def sequence_numerator_pos (n : ℕ) : ℕ :=
  (n + 1) / 2

def sequence_numerator_neg (n : ℕ) : ℤ :=
  -((n + 1) / 2 : ℤ)

def sequence_term (n : ℕ) : ℚ :=
  if n % 2 = 1 then 
    (sequence_numerator_pos n) / (sequence_denominator n) 
  else 
    (sequence_numerator_neg n : ℚ) / (sequence_denominator n)

theorem twenty_twenty_third_term :
  sequence_term 2023 = 1012 / 4045 := 
sorry

end twenty_twenty_third_term_l18_18733


namespace teacher_problems_remaining_l18_18623

theorem teacher_problems_remaining (problems_per_worksheet : Nat) 
                                   (total_worksheets : Nat) 
                                   (graded_worksheets : Nat) 
                                   (remaining_problems : Nat)
  (h1 : problems_per_worksheet = 4)
  (h2 : total_worksheets = 9)
  (h3 : graded_worksheets = 5)
  (h4 : remaining_problems = total_worksheets * problems_per_worksheet - graded_worksheets * problems_per_worksheet) :
  remaining_problems = 16 :=
sorry

end teacher_problems_remaining_l18_18623


namespace cameron_speed_ratio_l18_18362

variables (C Ch : ℝ)
-- Danielle's speed is three times Cameron's speed
def Danielle_speed := 3 * C
-- Danielle's travel time from Granville to Salisbury is 30 minutes
def Danielle_time := 30
-- Chase's travel time from Granville to Salisbury is 180 minutes
def Chase_time := 180

-- Prove the ratio of Cameron's speed to Chase's speed is 2
theorem cameron_speed_ratio :
  (Danielle_speed C / Ch) = (Chase_time / Danielle_time) → (C / Ch) = 2 :=
by {
  sorry
}

end cameron_speed_ratio_l18_18362


namespace smallest_integer_to_perfect_cube_l18_18767

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem smallest_integer_to_perfect_cube :
  ∃ n : ℕ, 
    n > 0 ∧ 
    is_perfect_cube (45216 * n) ∧ 
    (∀ m : ℕ, m > 0 ∧ is_perfect_cube (45216 * m) → n ≤ m) ∧ 
    n = 7 := sorry

end smallest_integer_to_perfect_cube_l18_18767


namespace original_integer_is_26_l18_18375

theorem original_integer_is_26 (x y z w : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : 0 < w)
(h₅ : x ≠ y) (h₆ : x ≠ z) (h₇ : x ≠ w) (h₈ : y ≠ z) (h₉ : y ≠ w) (h₁₀ : z ≠ w)
(h₁₁ : (x + y + z) / 3 + w = 34)
(h₁₂ : (x + y + w) / 3 + z = 22)
(h₁₃ : (x + z + w) / 3 + y = 26)
(h₁₄ : (y + z + w) / 3 + x = 18) :
    w = 26 := 
sorry

end original_integer_is_26_l18_18375


namespace odd_expression_divisible_by_48_l18_18294

theorem odd_expression_divisible_by_48 (x : ℤ) (h : Odd x) : 48 ∣ (x^3 + 3*x^2 - x - 3) :=
  sorry

end odd_expression_divisible_by_48_l18_18294


namespace fraction_equality_l18_18257

theorem fraction_equality (a b : ℝ) (h : a / 4 = b / 3) : b / (a - b) = 3 :=
sorry

end fraction_equality_l18_18257


namespace intersection_of_A_and_B_l18_18381

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B_l18_18381


namespace smallest_integer_is_17_l18_18518

theorem smallest_integer_is_17
  (a b c d : ℕ)
  (h1 : b = 33)
  (h2 : d = b + 3)
  (h3 : (a + b + c + d) = 120)
  (h4 : a ≤ b)
  (h5 : c > b)
  : a = 17 :=
sorry

end smallest_integer_is_17_l18_18518


namespace sum_difference_even_odd_l18_18023

-- Define the sum of even integers from 2 to 100
def sum_even (n : ℕ) : ℕ := (n / 2) * (2 + n)

-- Define the sum of odd integers from 1 to 99
def sum_odd (n : ℕ) : ℕ := (n / 2) * (1 + n)

theorem sum_difference_even_odd:
  let a := sum_even 100
  let b := sum_odd 99
  a - b = 50 :=
by
  sorry

end sum_difference_even_odd_l18_18023


namespace quadratic_one_real_root_positive_m_l18_18506

theorem quadratic_one_real_root_positive_m (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ((6 * m)^2 - 4 * 1 * (2 * m) = 0)) → m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_positive_m_l18_18506


namespace solve_equation_l18_18601

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l18_18601


namespace composite_for_all_n_greater_than_one_l18_18152

theorem composite_for_all_n_greater_than_one (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
by
  sorry

end composite_for_all_n_greater_than_one_l18_18152


namespace perfect_square_l18_18906

variable (n k l : ℕ)

theorem perfect_square (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m ^ 2 := by
  have h1 : (2 * l - n - k) * (2 * l - n + k) = (2 * l - n) ^ 2 - k ^ 2 := by sorry
  have h2 : k ^ 2 = 2 * l ^ 2 - n ^ 2 := by sorry
  have h3 : (2 * l - n) ^ 2 - (2 * l ^ 2 - n ^ 2) = 2 * (l - n) ^ 2 := by sorry
  have h4 : (2 * (l - n) ^ 2) / 2 = (l - n) ^ 2 := by sorry
  use (l - n)
  rw [h4]
  sorry

end perfect_square_l18_18906


namespace imaginary_part_of_l18_18180

theorem imaginary_part_of (i : ℂ) (h : i.im = 1) : (1 + i) ^ 5 = -14 - 4 * i := by
  sorry

end imaginary_part_of_l18_18180


namespace nelly_bid_l18_18732

theorem nelly_bid (joe_bid sarah_bid : ℕ) (h1 : joe_bid = 160000) (h2 : sarah_bid = 50000)
  (h3 : ∀ nelly_bid, nelly_bid = 3 * joe_bid + 2000) (h4 : ∀ nelly_bid, nelly_bid = 4 * sarah_bid + 1500) :
  ∃ nelly_bid, nelly_bid = 482000 :=
by
  -- Skipping the proof with sorry
  sorry

end nelly_bid_l18_18732


namespace intersection_line_exists_unique_l18_18170

universe u

noncomputable section

structure Point (α : Type u) :=
(x y z : α)

structure Line (α : Type u) :=
(dir point : Point α)

variables {α : Type u} [Field α]

-- Define skew lines conditions
def skew_lines (l1 l2 : Line α) : Prop :=
¬ ∃ p : Point α, ∃ t1 t2 : α, 
  l1.point = p ∧ l1.dir ≠ (Point.mk 0 0 0) ∧ l2.point = p ∧ l2.dir ≠ (Point.mk 0 0 0) ∧
  l1.dir.x * t1 = l2.dir.x * t2 ∧
  l1.dir.y * t1 = l2.dir.y * t2 ∧
  l1.dir.z * t1 = l2.dir.z * t2

-- Define a point not on the lines
def point_not_on_lines (p : Point α) (l1 l2 : Line α) : Prop :=
  (∀ t1 : α, p ≠ Point.mk (l1.point.x + l1.dir.x * t1) (l1.point.y + l1.dir.y * t1) (l1.point.z + l1.dir.z * t1))
  ∧
  (∀ t2 : α, p ≠ Point.mk (l2.point.x + l2.dir.x * t2) (l2.point.y + l2.dir.y * t2) (l2.point.z + l2.dir.z * t2))

-- Main theorem: existence and typical uniqueness of the intersection line
theorem intersection_line_exists_unique {l1 l2 : Line α} {O : Point α}
  (h_skew : skew_lines l1 l2) (h_point_not_on_lines : point_not_on_lines O l1 l2) :
  ∃! l : Line α, l.point = O ∧ (
    ∃ t1 : α, ∃ t2 : α,
    Point.mk (O.x + l.dir.x * t1) (O.y + l.dir.y * t1) (O.z + l.dir.z * t1) = 
    Point.mk (l1.point.x + l1.dir.x * t1) (l1.point.y + l1.dir.y * t1) (l1.point.z + l1.dir.z * t1) ∧
    Point.mk (O.x + l.dir.x * t2) (O.y + l.dir.x * t2) (O.z + l.dir.z * t2) = 
    Point.mk (l2.point.x + l2.dir.x * t2) (l2.point.y + l2.dir.y * t2) (l2.point.z + l2.dir.z * t2)
  ) :=
by
  sorry

end intersection_line_exists_unique_l18_18170


namespace sum_of_numbers_l18_18462

theorem sum_of_numbers : 72.52 + 12.23 + 5.21 = 89.96 :=
by sorry

end sum_of_numbers_l18_18462


namespace swimming_pool_radius_l18_18929

theorem swimming_pool_radius 
  (r : ℝ)
  (h1 : ∀ (r : ℝ), r > 0)
  (h2 : π * (r + 4)^2 - π * r^2 = (11 / 25) * π * r^2) :
  r = 20 := 
sorry

end swimming_pool_radius_l18_18929


namespace proof_by_contradiction_l18_18743

-- Definitions for the conditions
inductive ContradictionType
| known          -- ① Contradictory to what is known
| assumption     -- ② Contradictory to the assumption
| definitions    -- ③ Contradictory to definitions, theorems, axioms, laws
| facts          -- ④ Contradictory to facts

open ContradictionType

-- Proving that in proof by contradiction, a contradiction can be of type 1, 2, 3, or 4
theorem proof_by_contradiction :
  (∃ ct : ContradictionType, 
    ct = known ∨ 
    ct = assumption ∨ 
    ct = definitions ∨ 
    ct = facts) :=
by
  sorry

end proof_by_contradiction_l18_18743


namespace parabola_distance_to_y_axis_l18_18128

theorem parabola_distance_to_y_axis :
  ∀ (M : ℝ × ℝ), (M.2 ^ 2 = 4 * M.1) → 
  dist (M, (1, 0)) = 10 →
  abs (M.1) = 9 :=
by
  intros M hParabola hDist
  sorry

end parabola_distance_to_y_axis_l18_18128


namespace comics_in_box_l18_18196

def comics_per_comic := 25
def total_pages := 150
def existing_comics := 5

def torn_comics := total_pages / comics_per_comic
def total_comics := torn_comics + existing_comics

theorem comics_in_box : total_comics = 11 := by
  sorry

end comics_in_box_l18_18196


namespace suraya_picked_more_apples_l18_18159

theorem suraya_picked_more_apples (k c s : ℕ)
  (h_kayla : k = 20)
  (h_caleb : c = k - 5)
  (h_suraya : s = k + 7) :
  s - c = 12 :=
by
  -- Mark this as a place where the proof can be provided
  sorry

end suraya_picked_more_apples_l18_18159


namespace lucas_total_assignments_l18_18487

theorem lucas_total_assignments : 
  ∃ (total_assignments : ℕ), 
  (∀ (points : ℕ), 
    (points ≤ 10 → total_assignments = points * 1) ∧
    (10 < points ∧ points ≤ 20 → total_assignments = 10 * 1 + (points - 10) * 2) ∧
    (20 < points ∧ points ≤ 30 → total_assignments = 10 * 1 + 10 * 2 + (points - 20) * 3)
  ) ∧
  total_assignments = 60 :=
by
  sorry

end lucas_total_assignments_l18_18487


namespace solve_equation_l18_18597

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l18_18597


namespace graph_single_point_l18_18303

theorem graph_single_point (d : ℝ) :
  (∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0 -> (x = -1 ∧ y = 3)) ↔ d = 12 :=
by 
  sorry

end graph_single_point_l18_18303


namespace reservoir_fullness_before_storm_l18_18931

-- Definition of the conditions as Lean definitions
def storm_deposits : ℝ := 120 -- in billion gallons
def reservoir_percentage_after_storm : ℝ := 85 -- percentage
def original_contents : ℝ := 220 -- in billion gallons

-- The proof statement
theorem reservoir_fullness_before_storm (storm_deposits reservoir_percentage_after_storm original_contents : ℝ) : 
    (169 / 340) * 100 = 49.7 := 
  sorry

end reservoir_fullness_before_storm_l18_18931


namespace find_c_l18_18818

-- Definitions
def is_root (x c : ℝ) : Prop := x^2 - 3*x + c = 0

-- Main statement
theorem find_c (c : ℝ) (h : is_root 1 c) : c = 2 :=
sorry

end find_c_l18_18818


namespace annual_expenditure_l18_18569

theorem annual_expenditure (x y : ℝ) (h1 : y = 0.8 * x + 0.1) (h2 : x = 15) : y = 12.1 :=
by
  sorry

end annual_expenditure_l18_18569


namespace find_x_l18_18828

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Definition of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem statement
theorem find_x (x : ℝ) (h_parallel : parallel a (b x)) : x = 6 :=
sorry

end find_x_l18_18828


namespace number_of_teams_l18_18700

theorem number_of_teams (n : ℕ) 
  (h1 : ∀ (i j : ℕ), i ≠ j → (games_played : ℕ) = 4) 
  (h2 : ∀ (i j : ℕ), i ≠ j → (count : ℕ) = 760) : 
  n = 20 := 
by 
  sorry

end number_of_teams_l18_18700


namespace min_value_expression_l18_18080

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ a b, a > 0 ∧ b > 0 ∧ (∀ x y, x > 0 ∧ y > 0 → (1 / x + x / y^2 + y ≥ 2 * Real.sqrt 2)) := 
sorry

end min_value_expression_l18_18080


namespace imaginary_part_of_conjugate_z_mul_i_l18_18667

variable (i : ℂ) 

theorem imaginary_part_of_conjugate_z_mul_i (z : ℂ) (hz : z = 2 + i) : 
  Complex.imag (Complex.conj z * i) = 2 :=
by
  sorry

end imaginary_part_of_conjugate_z_mul_i_l18_18667


namespace find_a_l18_18664

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x + 3 else 4 / x

theorem find_a (a : ℝ) (h : f a = 2) : a = -1 ∨ a = 2 :=
sorry

end find_a_l18_18664


namespace smallest_number_of_slices_l18_18877

def cheddar_slices : ℕ := 12
def swiss_slices : ℕ := 28
def gouda_slices : ℕ := 18

theorem smallest_number_of_slices : Nat.lcm (Nat.lcm cheddar_slices swiss_slices) gouda_slices = 252 :=
by 
  sorry

end smallest_number_of_slices_l18_18877


namespace perfect_square_of_expression_l18_18902

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end perfect_square_of_expression_l18_18902


namespace max_sin_a_l18_18280

theorem max_sin_a (a b c : ℝ) (h1 : Real.cos a = Real.tan b) 
                                  (h2 : Real.cos b = Real.tan c) 
                                  (h3 : Real.cos c = Real.tan a) : 
  Real.sin a ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) := 
by
  sorry

end max_sin_a_l18_18280


namespace intersection_M_N_l18_18528

def set_M : Set ℝ := { x : ℝ | -3 ≤ x ∧ x < 4 }
def set_N : Set ℝ := { x : ℝ | x^2 - 2 * x - 8 ≤ 0 }

theorem intersection_M_N : (set_M ∩ set_N) = { x : ℝ | -2 ≤ x ∧ x < 4 } :=
sorry

end intersection_M_N_l18_18528


namespace inequality_proof_l18_18153

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 := 
by 
  sorry

end inequality_proof_l18_18153


namespace number_division_l18_18922

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l18_18922


namespace perimeter_to_side_ratio_l18_18880

variable (a b c h_a r : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < h_a ∧ 0 < r ∧ a + b > c ∧ a + c > b ∧ b + c > a)

theorem perimeter_to_side_ratio (P : ℝ) (hP : P = a + b + c) :
  P / a = h_a / r := by
  sorry

end perimeter_to_side_ratio_l18_18880


namespace algebraic_identity_l18_18523

theorem algebraic_identity (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2001 = -2000 :=
by
  sorry

end algebraic_identity_l18_18523


namespace cost_two_enchiladas_two_tacos_three_burritos_l18_18881

variables (e t b : ℝ)

theorem cost_two_enchiladas_two_tacos_three_burritos 
  (h1 : 2 * e + 3 * t + b = 5.00)
  (h2 : 3 * e + 2 * t + 2 * b = 7.50) : 
  2 * e + 2 * t + 3 * b = 10.625 :=
sorry

end cost_two_enchiladas_two_tacos_three_burritos_l18_18881


namespace find_multiplier_l18_18326

theorem find_multiplier (x : ℕ) (h₁ : 3 * x = (26 - x) + 26) (h₂ : x = 13) : 3 = 3 := 
by 
  sorry

end find_multiplier_l18_18326


namespace correct_polynomial_l18_18567

noncomputable def p : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X^6 - Polynomial.C 8 * Polynomial.X^4 - Polynomial.C 2 * Polynomial.X^3 + Polynomial.C 13 * Polynomial.X^2 - Polynomial.C 10 * Polynomial.X - Polynomial.C 1

theorem correct_polynomial (r t : ℝ) :
  (r^3 - r - 1 = 0) → (t = r + Real.sqrt 2) → Polynomial.aeval t p = 0 :=
by
  sorry

end correct_polynomial_l18_18567


namespace quadratic_inequality_l18_18847

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l18_18847


namespace ruby_siblings_l18_18853

structure Child :=
  (name : String)
  (eye_color : String)
  (hair_color : String)

def children : List Child :=
[
  {name := "Mason", eye_color := "Green", hair_color := "Red"},
  {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"},
  {name := "Fiona", eye_color := "Brown", hair_color := "Red"},
  {name := "Leo", eye_color := "Green", hair_color := "Blonde"},
  {name := "Ivy", eye_color := "Green", hair_color := "Red"},
  {name := "Carlos", eye_color := "Green", hair_color := "Blonde"}
]

def is_sibling_group (c1 c2 c3 : Child) : Prop :=
  (c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color) ∧
  (c2.eye_color = c3.eye_color ∨ c2.hair_color = c3.hair_color) ∧
  (c1.eye_color = c3.eye_color ∨ c1.hair_color = c3.hair_color)

theorem ruby_siblings :
  ∃ (c1 c2 : Child), 
    c1.name ≠ "Ruby" ∧ c2.name ≠ "Ruby" ∧
    c1 ≠ c2 ∧
    is_sibling_group {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"} c1 c2 ∧
    ((c1.name = "Leo" ∧ c2.name = "Carlos") ∨ (c1.name = "Carlos" ∧ c2.name = "Leo")) :=
by
  sorry

end ruby_siblings_l18_18853


namespace find_lambda_l18_18526

variables {R : Type*} [field R]

def A (λ : R) : matrix (fin 3) (fin 3) R :=
  ![
    ![1, 2, -2],
    ![2, -1, λ],
    ![3, 1, -1]
  ]

theorem find_lambda (λ : R) (h : det (A λ) = 0) : λ = 7 / 5 :=
by
  sorry

end find_lambda_l18_18526


namespace algebraic_expression_value_l18_18677

open Real

theorem algebraic_expression_value
  (θ : ℝ)
  (a := (cos θ, sin θ))
  (b := (1, -2))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (2 * sin θ - cos θ) / (sin θ + cos θ) = 5 :=
by
  sorry

end algebraic_expression_value_l18_18677


namespace expenditure_on_concrete_blocks_l18_18141

def blocks_per_section : ℕ := 30
def cost_per_block : ℕ := 2
def number_of_sections : ℕ := 8

theorem expenditure_on_concrete_blocks : 
  (number_of_sections * blocks_per_section) * cost_per_block = 480 := 
by 
  sorry

end expenditure_on_concrete_blocks_l18_18141


namespace scientific_notation_of_8_5_million_l18_18491

theorem scientific_notation_of_8_5_million :
  (8.5 * 10^6) = 8500000 :=
by sorry

end scientific_notation_of_8_5_million_l18_18491


namespace lowest_sale_price_is_30_percent_l18_18952

-- Definitions and conditions
def list_price : ℝ := 80
def max_initial_discount : ℝ := 0.50
def additional_sale_discount : ℝ := 0.20

-- Calculations
def initial_discount_amount : ℝ := list_price * max_initial_discount
def initial_discounted_price : ℝ := list_price - initial_discount_amount
def additional_discount_amount : ℝ := list_price * additional_sale_discount
def lowest_sale_price : ℝ := initial_discounted_price - additional_discount_amount

-- Proof statement (with correct answer)
theorem lowest_sale_price_is_30_percent :
  lowest_sale_price = 0.30 * list_price := 
by
  sorry

end lowest_sale_price_is_30_percent_l18_18952


namespace sum_of_ages_l18_18954

theorem sum_of_ages (a b c : ℕ) (h₁ : a = 20 + b + c) (h₂ : a^2 = 2050 + (b + c)^2) : a + b + c = 80 :=
sorry

end sum_of_ages_l18_18954


namespace xiao_ming_total_evaluation_score_l18_18770

theorem xiao_ming_total_evaluation_score 
  (regular midterm final : ℤ) (weight_regular weight_midterm weight_final : ℕ)
  (h1 : regular = 80)
  (h2 : midterm = 90)
  (h3 : final = 85)
  (h_weight_regular : weight_regular = 3)
  (h_weight_midterm : weight_midterm = 3)
  (h_weight_final : weight_final = 4) :
  (regular * weight_regular + midterm * weight_midterm + final * weight_final) /
    (weight_regular + weight_midterm + weight_final) = 85 :=
by
  sorry

end xiao_ming_total_evaluation_score_l18_18770


namespace cevian_concurrency_l18_18935

-- Definitions for the acute triangle and the angles
structure AcuteTriangle (α β γ : ℝ) :=
  (A B C : ℝ)
  (acute_α : α > 0 ∧ α < π / 2)
  (acute_β : β > 0 ∧ β < π / 2)
  (acute_γ : γ > 0 ∧ γ < π / 2)
  (triangle_sum : α + β + γ = π)

-- Definition for the concurrency of cevians
def cevians_concurrent (α β γ : ℝ) (t : AcuteTriangle α β γ) :=
  ∀ (A₁ B₁ C₁ : ℝ), sorry -- placeholder

-- The main theorem with the proof of concurrency
theorem cevian_concurrency (α β γ : ℝ) (t : AcuteTriangle α β γ) :
  ∃ (A₁ B₁ C₁ : ℝ), cevians_concurrent α β γ t :=
  sorry -- proof to be provided


end cevian_concurrency_l18_18935


namespace percentage_of_seeds_germinated_l18_18374

theorem percentage_of_seeds_germinated (P1 P2 : ℕ) (GP1 GP2 : ℕ) (SP1 SP2 TotalGerminated TotalPlanted : ℕ) (PG : ℕ) 
  (h1 : P1 = 300) (h2 : P2 = 200) (h3 : GP1 = 60) (h4 : GP2 = 70) (h5 : SP1 = P1) (h6 : SP2 = P2)
  (h7 : TotalGerminated = GP1 + GP2) (h8 : TotalPlanted = SP1 + SP2) : 
  PG = (TotalGerminated * 100) / TotalPlanted :=
sorry

end percentage_of_seeds_germinated_l18_18374


namespace solve_equation_l18_18589

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l18_18589


namespace probability_meeting_semifinals_probability_meeting_final_l18_18702

-- Definitions used in the problem
variable {teams : Fin 8 → Prop}

-- Placeholder for main theorem - Part (a)
theorem probability_meeting_semifinals (A B : Fin 8) : 
  (probability (λ s, teams s) (λ s, s = A ∧ s ≠ B ∧ sorry)) = 1 / 14 :=
sorry

-- Placeholder for main theorem - Part (b)
theorem probability_meeting_final (A B : Fin 8) : 
  (probability (λ s, teams s) (λ s, s = A ∧ s ≠ B ∧ sorry)) = 1 / 28 :=
sorry

end probability_meeting_semifinals_probability_meeting_final_l18_18702


namespace div_of_abs_values_l18_18247

theorem div_of_abs_values (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x < y) : x / y = -2 := 
by
  sorry

end div_of_abs_values_l18_18247


namespace total_value_of_bills_in_cash_drawer_l18_18340

-- Definitions based on conditions
def total_bills := 54
def five_dollar_bills := 20
def twenty_dollar_bills := total_bills - five_dollar_bills
def value_of_five_dollar_bills := 5
def value_of_twenty_dollar_bills := 20
def total_value_of_five_dollar_bills := five_dollar_bills * value_of_five_dollar_bills
def total_value_of_twenty_dollar_bills := twenty_dollar_bills * value_of_twenty_dollar_bills

-- Statement to prove
theorem total_value_of_bills_in_cash_drawer :
  total_value_of_five_dollar_bills + total_value_of_twenty_dollar_bills = 780 :=
by
  -- Proof goes here
  sorry

end total_value_of_bills_in_cash_drawer_l18_18340


namespace cylinder_height_l18_18046

theorem cylinder_height {D r : ℝ} (hD : D = 10) (hr : r = 3) : 
  ∃ h : ℝ, h = 8 :=
by
  -- hD -> Diameter of hemisphere = 10
  -- hr -> Radius of cylinder's base = 3
  sorry

end cylinder_height_l18_18046


namespace man_receives_total_amount_l18_18781
noncomputable def total_amount_received : ℝ := 
  let itemA_price := 1300
  let itemB_price := 750
  let itemC_price := 1800
  
  let itemA_loss := 0.20 * itemA_price
  let itemB_loss := 0.15 * itemB_price
  let itemC_loss := 0.10 * itemC_price

  let itemA_selling_price := itemA_price - itemA_loss
  let itemB_selling_price := itemB_price - itemB_loss
  let itemC_selling_price := itemC_price - itemC_loss

  let vat_rate := 0.12
  let itemA_vat := vat_rate * itemA_selling_price
  let itemB_vat := vat_rate * itemB_selling_price
  let itemC_vat := vat_rate * itemC_selling_price

  let final_itemA := itemA_selling_price + itemA_vat
  let final_itemB := itemB_selling_price + itemB_vat
  let final_itemC := itemC_selling_price + itemC_vat

  final_itemA + final_itemB + final_itemC

theorem man_receives_total_amount :
  total_amount_received = 3693.2 := by
  sorry

end man_receives_total_amount_l18_18781


namespace coeff_x2_term_l18_18808

theorem coeff_x2_term (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : d = 6) (h5 : e = 7) (h6 : f = 8) :
    (a * f + b * e * 1 + c * d) = 82 := 
by
    sorry

end coeff_x2_term_l18_18808


namespace football_total_points_l18_18707

theorem football_total_points :
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  Zach_points + Ben_points + Sarah_points + Emily_points = 109.0 :=
by
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  have h : Zach_points + Ben_points + Sarah_points + Emily_points = 42.0 + 21.0 + 18.5 + 27.5 := by rfl
  have total_points := 42.0 + 21.0 + 18.5 + 27.5
  have result := 109.0
  sorry

end football_total_points_l18_18707


namespace interest_rate_l18_18730

theorem interest_rate (P1 P2 I T1 T2 total_amount : ℝ) (r : ℝ) :
  P1 = 10000 →
  P2 = 22000 →
  T1 = 2 →
  T2 = 3 →
  total_amount = 27160 →
  (I = P1 * r * T1 / 100 + P2 * r * T2 / 100) →
  P1 + P2 = 22000 →
  (P1 + I = total_amount) →
  r = 6 :=
by
  intros hP1 hP2 hT1 hT2 htotal_amount hI hP_total hP1_I_total
  -- Actual proof would go here
  sorry

end interest_rate_l18_18730


namespace no_such_n_exists_l18_18651

noncomputable def is_partitionable (s : Finset ℕ) : Prop :=
  ∃ (A B : Finset ℕ), A ∪ B = s ∧ A ∩ B = ∅ ∧ (A.prod id = B.prod id)

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, n > 0 ∧ is_partitionable {n, n+1, n+2, n+3, n+4, n+5} :=
by
  sorry

end no_such_n_exists_l18_18651


namespace perfect_square_l18_18907

variable (n k l : ℕ)

theorem perfect_square (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m ^ 2 := by
  have h1 : (2 * l - n - k) * (2 * l - n + k) = (2 * l - n) ^ 2 - k ^ 2 := by sorry
  have h2 : k ^ 2 = 2 * l ^ 2 - n ^ 2 := by sorry
  have h3 : (2 * l - n) ^ 2 - (2 * l ^ 2 - n ^ 2) = 2 * (l - n) ^ 2 := by sorry
  have h4 : (2 * (l - n) ^ 2) / 2 = (l - n) ^ 2 := by sorry
  use (l - n)
  rw [h4]
  sorry

end perfect_square_l18_18907


namespace probability_no_3by3_red_grid_correct_l18_18216

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l18_18216


namespace nancy_pictures_left_l18_18936

-- Given conditions stated in the problem
def picturesZoo : Nat := 49
def picturesMuseum : Nat := 8
def picturesDeleted : Nat := 38

-- The statement of the problem, proving Nancy still has 19 pictures after deletions
theorem nancy_pictures_left : (picturesZoo + picturesMuseum) - picturesDeleted = 19 := by
  sorry

end nancy_pictures_left_l18_18936


namespace problem_statement_l18_18030

open Real

theorem problem_statement :
  (10 * 1.8 - 2 * 1.5) / 0.3 + 3^(2/3) - Real.log 4 = 50.6938 :=
by
  sorry

end problem_statement_l18_18030


namespace combined_weight_of_candles_l18_18803

theorem combined_weight_of_candles :
  (∀ (candles: ℕ), ∀ (weight_per_candle: ℕ),
    (weight_per_candle = 8 + 1) →
    (candles = 10 - 3) →
    (candles * weight_per_candle = 63)) :=
by
  intros
  sorry

end combined_weight_of_candles_l18_18803


namespace chocolates_initially_initial_chocolates_eq_60_chocolates_ate_pre_rearrange_l18_18035

theorem chocolates_initially (M : ℕ) (R1 R2 C1 C2 : ℕ) (third : ℕ) 
  (h1 : M = 60) (h2 : R1 = 3 * 12 - 1)
  (h3 : R2 = 3 * 12) (h4 : C1 = 5 * (third - 1))
  (h5 : C2 = 5 * third) (h6 : third = M / 3) :
  (M = 60 ∧ (M - (3 * 12 - 1)) = 25) :=
by
  split
  case left => exact h1
  case right => rw [h1]; exact Nat.sub_eq_of_eq_add h2

-- Theorem for the initial chocolates in the entire box
theorem initial_chocolates_eq_60 (N remaining : ℕ) 
  (h1 : remaining = N / 3) :
  N = 60 →
  remaining = 20 :=
by 
  intro h2
  rw [h2] at h1
  exact h1

-- Theorem for the number of chocolates Míša ate before the first rearrangement
theorem chocolates_ate_pre_rearrange (N eaten row_rearranged : ℕ) 
  (h1 : row_rearranged = 3 * 12 - 1)
  (h2 : eaten = N - row_rearranged)
  (h3 : N = 60) :
  eaten = 25 :=
by
  rw [h3] at *
  rw [h2]
  exact Nat.sub_eq_of_eq_add h1 h2

end chocolates_initially_initial_chocolates_eq_60_chocolates_ate_pre_rearrange_l18_18035


namespace number_of_permutations_l18_18409

noncomputable def num_satisfying_permutations : ℕ :=
  Nat.choose 15 7

theorem number_of_permutations : num_satisfying_permutations = 6435 := by
  sorry

end number_of_permutations_l18_18409


namespace Woojin_harvested_weight_l18_18616

-- Definitions based on conditions
def younger_brother_harvest : Float := 3.8
def older_sister_harvest : Float := younger_brother_harvest + 8.4
def one_tenth_older_sister : Float := older_sister_harvest / 10
def woojin_extra_g : Float := 3720

-- Convert grams to kilograms
def grams_to_kg (g : Float) : Float := g / 1000

-- Theorem to be proven
theorem Woojin_harvested_weight :
  grams_to_kg (one_tenth_older_sister * 1000 + woojin_extra_g) = 4.94 :=
by
  sorry

end Woojin_harvested_weight_l18_18616


namespace closest_point_on_line_is_correct_l18_18654

theorem closest_point_on_line_is_correct :
  ∃ (p : ℝ × ℝ), p = (-0.04, -0.28) ∧
  ∃ x : ℝ, p = (x, (3 * x - 1) / 4) ∧
  ∀ q : ℝ × ℝ, (q = (x, (3 * x - 1) / 4) → 
  (dist (2, -3) p) ≤ (dist (2, -3) q)) :=
sorry

end closest_point_on_line_is_correct_l18_18654


namespace compute_expression_l18_18970

theorem compute_expression : 2 * ((3 + 7) ^ 2 + (3 ^ 2 + 7 ^ 2)) = 316 := 
by
  sorry

end compute_expression_l18_18970


namespace aston_comics_l18_18198

theorem aston_comics (total_pages_on_floor : ℕ) (pages_per_comic : ℕ) (untorn_comics_in_box : ℕ) :
  total_pages_on_floor = 150 →
  pages_per_comic = 25 →
  untorn_comics_in_box = 5 →
  (total_pages_on_floor / pages_per_comic + untorn_comics_in_box) = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end aston_comics_l18_18198


namespace problem_a_l18_18776

theorem problem_a (x a : ℝ) (h : (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) = 3 * a^4) :
  x = (-5 * a + a * Real.sqrt 37) / 2 ∨ x = (-5 * a - a * Real.sqrt 37) / 2 :=
by
  sorry

end problem_a_l18_18776


namespace smallest_among_given_numbers_l18_18642

theorem smallest_among_given_numbers :
  let a := abs (-3)
  let b := -2
  let c := 0
  let d := Real.pi
  b < a ∧ b < c ∧ b < d := by
  sorry

end smallest_among_given_numbers_l18_18642


namespace find_d_for_single_point_l18_18300

/--
  Suppose that the graph of \(3x^2 + y^2 + 6x - 6y + d = 0\) consists of a single point.
  Prove that \(d = 12\).
-/
theorem find_d_for_single_point : 
  ∀ (d : ℝ), (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0) ∧
              (∀ (x1 y1 x2 y2 : ℝ), 
                (3 * x1^2 + y1^2 + 6 * x1 - 6 * y1 + d = 0 ∧ 
                 3 * x2^2 + y2^2 + 6 * x2 - 6 * y2 + d = 0 → 
                 x1 = x2 ∧ y1 = y2)) ↔ d = 12 := 
by 
  sorry

end find_d_for_single_point_l18_18300


namespace solve_equation_l18_18588

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l18_18588


namespace coeff_x5_in_expansion_l18_18541

theorem coeff_x5_in_expansion :
  (x : ℝ) → polynomial.eval x ((x^2 + 1)^2 * (x - 1)^6) = -52 * x^5 + (other_terms : polynomial ℝ) := by
  sorry

end coeff_x5_in_expansion_l18_18541


namespace cos_sq_sub_sin_sq_l18_18376

noncomputable def cos_sq_sub_sin_sq_eq := 
  ∀ (α : ℝ), α ∈ Set.Ioo 0 Real.pi → (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  (Real.cos α) ^ 2 - (Real.sin α) ^ 2 = -Real.sqrt 5 / 3

theorem cos_sq_sub_sin_sq :
  cos_sq_sub_sin_sq_eq := 
by
  intros α hα h_eq
  sorry

end cos_sq_sub_sin_sq_l18_18376


namespace number_of_rows_is_ten_l18_18630

-- Definition of the arithmetic sequence
def arithmetic_sequence_sum (n : ℕ) : ℕ :=
  n * (3 * n + 1) / 2

-- The main theorem to prove
theorem number_of_rows_is_ten :
  (∃ n : ℕ, arithmetic_sequence_sum n = 145) ↔ n = 10 :=
by
  sorry

end number_of_rows_is_ten_l18_18630


namespace number_is_2250_l18_18914

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l18_18914


namespace binom_9_5_eq_126_l18_18987

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l18_18987


namespace binomial_coefficient_9_5_l18_18998

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l18_18998


namespace count_digit_9_from_1_to_1000_l18_18114

theorem count_digit_9_from_1_to_1000 : (list.range 1000).countp (λ n, n.digits 10).contains 9 = 301 := 
sorry

end count_digit_9_from_1_to_1000_l18_18114


namespace conditional_probability_P_B_given_A_l18_18456

-- Let E be an enumeration type with exactly five values, each representing one attraction.
inductive Attraction : Type
| dayu_yashan : Attraction
| qiyunshan : Attraction
| tianlongshan : Attraction
| jiulianshan : Attraction
| sanbaishan : Attraction

open Attraction

-- Define A and B's choices as random variables.
axiom A_choice : Attraction
axiom B_choice : Attraction

-- Event A is that A and B choose different attractions.
def event_A : Prop := A_choice ≠ B_choice

-- Event B is that A and B each choose Chongyi Qiyunshan.
def event_B : Prop := A_choice = qiyunshan ∧ B_choice = qiyunshan

-- Calculate the conditional probability P(B|A)
theorem conditional_probability_P_B_given_A : 
  (1 - (1 / 5)) * (1 - (1 / 5)) = 2 / 5 :=
sorry

end conditional_probability_P_B_given_A_l18_18456


namespace smaller_angle_between_east_and_northwest_l18_18941

theorem smaller_angle_between_east_and_northwest
  (rays : ℕ)
  (each_angle : ℕ)
  (direction : ℕ → ℝ)
  (h1 : rays = 10)
  (h2 : each_angle = 36)
  (h3 : direction 0 = 0) -- ray at due North
  (h4 : direction 3 = 90) -- ray at due East
  (h5 : direction 5 = 135) -- ray at due Northwest
  : direction 5 - direction 3 = each_angle :=
by
  -- to be proved
  sorry

end smaller_angle_between_east_and_northwest_l18_18941


namespace is_divisible_by_N2_l18_18571

def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def eulers_totient (n : ℕ) : ℕ :=
  Nat.totient n

theorem is_divisible_by_N2 (N1 N2 : ℕ) (h_coprime : are_coprime N1 N2) 
  (k := eulers_totient N2) : 
  (N1 ^ k - 1) % N2 = 0 :=
by
  sorry

end is_divisible_by_N2_l18_18571


namespace pipe_fill_time_without_leak_l18_18187

theorem pipe_fill_time_without_leak (T : ℕ) :
  let pipe_with_leak_time := 10
  let leak_empty_time := 10
  ((1 / T : ℚ) - (1 / leak_empty_time) = (1 / pipe_with_leak_time)) →
  T = 5 := 
sorry

end pipe_fill_time_without_leak_l18_18187


namespace gcd_of_B_l18_18410

def B : Set ℕ := {n | ∃ x : ℕ, n = 5 * x}

theorem gcd_of_B : Int.gcd_range (B.toList) = 5 := by 
  sorry

end gcd_of_B_l18_18410


namespace smallest_odd_m_satisfying_inequality_l18_18241

theorem smallest_odd_m_satisfying_inequality : ∃ m : ℤ, m^2 - 11 * m + 24 ≥ 0 ∧ (m % 2 = 1) ∧ ∀ n : ℤ, n^2 - 11 * n + 24 ≥ 0 ∧ (n % 2 = 1) → m ≤ n → m = 3 :=
by
  sorry

end smallest_odd_m_satisfying_inequality_l18_18241


namespace max_shortest_side_decagon_inscribed_circle_l18_18892

noncomputable def shortest_side_decagon : ℝ :=
  2 * Real.sin (36 * Real.pi / 180 / 2)

theorem max_shortest_side_decagon_inscribed_circle :
  shortest_side_decagon = (Real.sqrt 5 - 1) / 2 :=
by {
  -- Proof details here
  sorry
}

end max_shortest_side_decagon_inscribed_circle_l18_18892


namespace arithmetic_sequence_identification_l18_18259

variable (a : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_identification (h : is_arithmetic a d) :
  (is_arithmetic (fun n => a n + 3) d) ∧
  ¬ (is_arithmetic (fun n => a n ^ 2) d) ∧
  (is_arithmetic (fun n => a (n + 1) - a n) d) ∧
  (is_arithmetic (fun n => 2 * a n) (2 * d)) ∧
  (is_arithmetic (fun n => 2 * a n + n) (2 * d + 1)) :=
by
  sorry

end arithmetic_sequence_identification_l18_18259


namespace dot_product_value_l18_18254

variables (a b : ℝ × ℝ)

theorem dot_product_value
  (h1 : a + b = (1, -3))
  (h2 : a - b = (3, 7)) :
  a.1 * b.1 + a.2 * b.2 = -12 :=
sorry

end dot_product_value_l18_18254


namespace ellipse_equation_1_ellipse_equation_2_l18_18811

-- Proof Problem 1
theorem ellipse_equation_1 (x y : ℝ) 
  (foci_condition : (x+2) * (x+2) + y*y + (x-2) * (x-2) + y*y = 36) :
  x^2 / 9 + y^2 / 5 = 1 :=
sorry

-- Proof Problem 2
theorem ellipse_equation_2 (x y : ℝ)
  (foci_condition : (x^2 + (y+5)^2 = 0) ∧ (x^2 + (y-5)^2 = 0))
  (point_on_ellipse : 3^2 / 15 + 4^2 / (15 + 25) = 1) :
  y^2 / 40 + x^2 / 15 = 1 :=
sorry

end ellipse_equation_1_ellipse_equation_2_l18_18811


namespace number_division_l18_18923

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l18_18923


namespace find_m_plus_n_l18_18227

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l18_18227


namespace sum_m_n_l18_18233

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l18_18233


namespace basis_group1_basis_group2_basis_group3_basis_l18_18192

def vector (α : Type*) := α × α

def is_collinear (v1 v2: vector ℝ) : Prop :=
  v1.1 * v2.2 - v2.1 * v1.2 = 0

def group1_v1 : vector ℝ := (-1, 2)
def group1_v2 : vector ℝ := (5, 7)

def group2_v1 : vector ℝ := (3, 5)
def group2_v2 : vector ℝ := (6, 10)

def group3_v1 : vector ℝ := (2, -3)
def group3_v2 : vector ℝ := (0.5, 0.75)

theorem basis_group1 : ¬ is_collinear group1_v1 group1_v2 :=
by sorry

theorem basis_group2 : is_collinear group2_v1 group2_v2 :=
by sorry

theorem basis_group3 : ¬ is_collinear group3_v1 group3_v2 :=
by sorry

theorem basis : (¬ is_collinear group1_v1 group1_v2) ∧ (is_collinear group2_v1 group2_v2) ∧ (¬ is_collinear group3_v1 group3_v2) :=
by sorry

end basis_group1_basis_group2_basis_group3_basis_l18_18192


namespace find_a_purely_imaginary_l18_18120

noncomputable def purely_imaginary_condition (a : ℝ) : Prop :=
    (2 * a - 1) / (a^2 + 1) = 0 ∧ (a + 2) / (a^2 + 1) ≠ 0

theorem find_a_purely_imaginary :
    ∀ (a : ℝ), purely_imaginary_condition a ↔ a = 1/2 := 
by
  sorry

end find_a_purely_imaginary_l18_18120


namespace arithmetic_sequence_sum_l18_18965

open Nat

theorem arithmetic_sequence_sum :
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  3 * S = 3774 := 
by
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end arithmetic_sequence_sum_l18_18965


namespace proof_expr_28_times_35_1003_l18_18238

theorem proof_expr_28_times_35_1003 :
  (5^1003 + 7^1004)^2 - (5^1003 - 7^1004)^2 = 28 * 35^1003 :=
by
  sorry

end proof_expr_28_times_35_1003_l18_18238


namespace amount_collected_from_ii_and_iii_class_l18_18177

theorem amount_collected_from_ii_and_iii_class
  (P1 P2 P3 : ℕ) (F1 F2 F3 : ℕ) (total_amount amount_ii_iii : ℕ)
  (H1 : P1 / P2 = 1 / 50)
  (H2 : P1 / P3 = 1 / 100)
  (H3 : F1 / F2 = 5 / 2)
  (H4 : F1 / F3 = 5 / 1)
  (H5 : total_amount = 3575)
  (H6 : total_amount = (P1 * F1) + (P2 * F2) + (P3 * F3))
  (H7 : amount_ii_iii = (P2 * F2) + (P3 * F3)) :
  amount_ii_iii = 3488 := sorry

end amount_collected_from_ii_and_iii_class_l18_18177


namespace range_of_a_l18_18083

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x) ∧ (∃ x : ℝ, x^2 - 4 * x + a ≤ 0) →
  a ∈ Set.Icc (Real.exp 1) 4 :=
by
  sorry

end range_of_a_l18_18083


namespace probability_no_3x3_red_square_l18_18213

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l18_18213


namespace negation_of_universal_proposition_l18_18893

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^3 - x^2 + 1 ≤ 0) ↔ ∃ (x₀ : ℝ), x₀^3 - x₀^2 + 1 > 0 :=
by {
  sorry
}

end negation_of_universal_proposition_l18_18893


namespace perfect_square_expression_l18_18901

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end perfect_square_expression_l18_18901


namespace isosceles_triangle_sin_vertex_angle_l18_18385

theorem isosceles_triangle_sin_vertex_angle (A : ℝ) (hA : 0 < A ∧ A < π / 2) 
  (hSinA : Real.sin A = 5 / 13) : 
  Real.sin (2 * A) = 120 / 169 :=
by 
  -- This placeholder indicates where the proof would go
  sorry

end isosceles_triangle_sin_vertex_angle_l18_18385


namespace determine_dimensions_l18_18752

theorem determine_dimensions (a b : ℕ) (h : a < b) 
    (h1 : ∃ (m n : ℕ), 49 * 51 = (m * a) * (n * b))
    (h2 : ∃ (p q : ℕ), 99 * 101 = (p * a) * (q * b)) : 
    a = 1 ∧ b = 3 :=
  by 
  sorry

end determine_dimensions_l18_18752


namespace simplify_expression_l18_18738

theorem simplify_expression (x : ℝ) : 
  x - 2 * (1 + x) + 3 * (1 - x) - 4 * (1 + 2 * x) = -12 * x - 3 := 
by 
  -- Proof goes here
  sorry

end simplify_expression_l18_18738


namespace find_a_l18_18089

variable (a : ℝ)

def f (x : ℝ) := a * x^3 + 3 * x^2 + 2

theorem find_a (h : deriv (deriv (f a)) (-1) = 4) : a = 10 / 3 :=
by
  sorry

end find_a_l18_18089


namespace number_of_players_taking_mathematics_l18_18495

-- Define the conditions
def total_players := 15
def players_physics := 10
def players_both := 4

-- Define the conclusion to be proven
theorem number_of_players_taking_mathematics : (total_players - players_physics + players_both) = 9 :=
by
  -- Placeholder for proof
  sorry

end number_of_players_taking_mathematics_l18_18495


namespace triangle_perimeter_l18_18443

theorem triangle_perimeter
  (a b : ℕ) (c : ℕ) 
  (h_side1 : a = 3)
  (h_side2 : b = 4)
  (h_third_side : c^2 - 13 * c + 40 = 0)
  (h_valid_triangle : (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :
  a + b + c = 12 :=
by {
  sorry
}

end triangle_perimeter_l18_18443


namespace first_candidate_percentage_l18_18856

theorem first_candidate_percentage (P : ℝ) 
    (total_votes : ℝ) (votes_second : ℝ)
    (h_total_votes : total_votes = 1200)
    (h_votes_second : votes_second = 480) :
    (P / 100) * total_votes + votes_second = total_votes → P = 60 := 
by
  intro h
  rw [h_total_votes, h_votes_second] at h
  sorry

end first_candidate_percentage_l18_18856


namespace sasha_took_right_triangle_l18_18010

-- Define types of triangles
inductive Triangle
| acute
| right
| obtuse

open Triangle

-- Define the function that determines if Borya can form a triangle identical to Sasha's
def can_form_identical_triangle (t1 t2 t3: Triangle) : Bool :=
match t1, t2, t3 with
| right, acute, obtuse => true
| _ , _ , _ => false

-- Define the main theorem
theorem sasha_took_right_triangle : 
  ∀ (sasha_takes borya_takes1 borya_takes2 : Triangle),
  (sasha_takes ≠ borya_takes1 ∧ sasha_takes ≠ borya_takes2 ∧ borya_takes1 ≠ borya_takes2) →
  can_form_identical_triangle sasha_takes borya_takes1 borya_takes2 →
  sasha_takes = right :=
by sorry

end sasha_took_right_triangle_l18_18010


namespace solve_MQ_above_A_l18_18706

-- Definitions of the given conditions
def ABCD_side := 8
def MNPQ_length := 16
def MNPQ_width := 8
def area_outer_inner_ratio := 1 / 3

-- Definition to prove
def length_MQ_above_A := 8 / 3

-- The area calculations
def area_MNPQ := MNPQ_length * MNPQ_width
def area_ABCD := ABCD_side * ABCD_side
def area_outer := (area_outer_inner_ratio * area_MNPQ)
def MQ_above_A_calculated := area_outer / MNPQ_length

theorem solve_MQ_above_A :
  MQ_above_A_calculated = length_MQ_above_A := by sorry

end solve_MQ_above_A_l18_18706


namespace JimSiblings_l18_18135

-- Define the students and their characteristics.
structure Student :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (wearsGlasses : Bool)

def Benjamin : Student := ⟨"Benjamin", "Blue", "Blond", true⟩
def Jim : Student := ⟨"Jim", "Brown", "Blond", false⟩
def Nadeen : Student := ⟨"Nadeen", "Brown", "Black", true⟩
def Austin : Student := ⟨"Austin", "Blue", "Black", false⟩
def Tevyn : Student := ⟨"Tevyn", "Blue", "Blond", true⟩
def Sue : Student := ⟨"Sue", "Brown", "Blond", false⟩

-- Define the condition that students from the same family share at least one characteristic.
def shareCharacteristic (s1 s2 : Student) : Bool :=
  (s1.eyeColor = s2.eyeColor) ∨
  (s1.hairColor = s2.hairColor) ∨
  (s1.wearsGlasses = s2.wearsGlasses)

-- Define what it means to be siblings of a student.
def areSiblings (s1 s2 s3 : Student) : Bool :=
  shareCharacteristic s1 s2 ∧
  shareCharacteristic s1 s3 ∧
  shareCharacteristic s2 s3

-- The theorem we are trying to prove.
theorem JimSiblings : areSiblings Jim Sue Benjamin = true := 
  by sorry

end JimSiblings_l18_18135


namespace triangle_inequality_l18_18647

theorem triangle_inequality (a b c : ℝ) (h : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end triangle_inequality_l18_18647


namespace isosceles_triangle_angle_B_l18_18137

theorem isosceles_triangle_angle_B (A B C : ℝ)
  (h_triangle : (A + B + C = 180))
  (h_exterior_A : 180 - A = 110)
  (h_sum_angles : A + B + C = 180) :
  B = 70 ∨ B = 55 ∨ B = 40 :=
by
  sorry

end isosceles_triangle_angle_B_l18_18137


namespace fraction_simplifies_correctly_l18_18688

variable (a b : ℕ)

theorem fraction_simplifies_correctly (h : a ≠ b) : (1/2 * a) / (1/2 * b) = a / b := 
by sorry

end fraction_simplifies_correctly_l18_18688


namespace initial_percentage_is_30_l18_18476

def percentage_alcohol (P : ℝ) : Prop :=
  let initial_alcohol := (P / 100) * 50
  let mixed_solution_volume := 50 + 30
  let final_percentage_alcohol := 18.75
  let final_alcohol := (final_percentage_alcohol / 100) * mixed_solution_volume
  initial_alcohol = final_alcohol

theorem initial_percentage_is_30 :
  percentage_alcohol 30 :=
by
  unfold percentage_alcohol
  sorry

end initial_percentage_is_30_l18_18476


namespace new_train_travel_distance_l18_18186

-- Definitions of the trains' travel distances
def older_train_distance : ℝ := 180
def new_train_additional_distance_ratio : ℝ := 0.50

-- Proof that the new train can travel 270 miles
theorem new_train_travel_distance
: new_train_additional_distance_ratio * older_train_distance + older_train_distance = 270 := 
by
  sorry

end new_train_travel_distance_l18_18186


namespace find_length_of_EF_l18_18250

-- Definitions based on conditions
noncomputable def AB : ℝ := 300
noncomputable def DC : ℝ := 180
noncomputable def BC : ℝ := 200
noncomputable def E_as_fraction_of_BC : ℝ := (3 / 5)

-- Derived definition based on given conditions
noncomputable def EB : ℝ := E_as_fraction_of_BC * BC
noncomputable def EC : ℝ := BC - EB
noncomputable def EF : ℝ := (EC / BC) * DC

-- The theorem we need to prove
theorem find_length_of_EF : EF = 72 := by
  sorry

end find_length_of_EF_l18_18250


namespace guess_probability_greater_than_two_thirds_l18_18293

theorem guess_probability_greater_than_two_thirds :
  (1335 : ℝ) / 2002 > 2 / 3 :=
by {
  -- Placeholder for proof
  sorry
}

end guess_probability_greater_than_two_thirds_l18_18293


namespace distinct_real_roots_c_l18_18570

theorem distinct_real_roots_c (c : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + c = 0 ∧ x₂^2 - 4*x₂ + c = 0) ↔ c < 4 := by
  sorry

end distinct_real_roots_c_l18_18570


namespace number_is_2250_l18_18913

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l18_18913


namespace infinite_series_sum_l18_18796

theorem infinite_series_sum : (∑' n : ℕ, if n % 3 = 0 then 1 / (3 * 2^(((n - n % 3) / 3) + 1)) 
                                 else if n % 3 = 1 then -1 / (6 * 2^(((n - n % 3) / 3)))
                                 else -1 / (12 * 2^(((n - n % 3) / 3)))) = 1 / 72 :=
by
  sorry

end infinite_series_sum_l18_18796


namespace playdough_cost_l18_18716

-- Definitions of the costs and quantities
def lego_cost := 250
def sword_cost := 120
def playdough_quantity := 10
def total_paid := 1940

-- Variables representing the quantities bought
def lego_quantity := 3
def sword_quantity := 7

-- Function to calculate the total cost for lego and sword
def total_lego_cost := lego_quantity * lego_cost
def total_sword_cost := sword_quantity * sword_cost

-- Variable representing the cost of playdough
variable (P : ℝ)

-- The main statement to prove
theorem playdough_cost :
  total_lego_cost + total_sword_cost + playdough_quantity * P = total_paid → P = 35 :=
by
  sorry

end playdough_cost_l18_18716


namespace madeline_refills_l18_18419

theorem madeline_refills :
  let total_water := 100
  let bottle_capacity := 12
  let remaining_to_drink := 16
  let already_drank := total_water - remaining_to_drink
  let initial_refills := already_drank / bottle_capacity
  let refills := initial_refills + 1
  refills = 8 :=
by
  sorry

end madeline_refills_l18_18419


namespace Mary_bought_stickers_initially_l18_18289

variable (S A M : ℕ) -- Define S, A, and M as natural numbers

-- Given conditions in the problem
def condition1 : Prop := S = A
def condition2 : Prop := M = 3 * A
def condition3 : Prop := A + (2 / 3) * M = 900

-- The theorem we need to prove
theorem Mary_bought_stickers_initially
  (h1 : condition1 S A)
  (h2 : condition2 A M)
  (h3 : condition3 A M)
  : S + A + M = 1500 :=
sorry -- Proof

end Mary_bought_stickers_initially_l18_18289


namespace basketball_surface_area_l18_18625

theorem basketball_surface_area (C : ℝ) (r : ℝ) (A : ℝ) (π : ℝ) 
  (h1 : C = 30) 
  (h2 : C = 2 * π * r) 
  (h3 : A = 4 * π * r^2) 
  : A = 900 / π := by
  sorry

end basketball_surface_area_l18_18625


namespace final_cost_is_30_l18_18757

-- Define conditions as constants
def cost_of_repair : ℝ := 7
def sales_tax : ℝ := 0.50
def number_of_tires : ℕ := 4

-- Define the cost for one tire repair
def cost_one_tire : ℝ := cost_of_repair + sales_tax

-- Define the cost for all tires
def total_cost : ℝ := cost_one_tire * number_of_tires

-- Theorem stating that the total cost is $30
theorem final_cost_is_30 : total_cost = 30 :=
by
  sorry

end final_cost_is_30_l18_18757


namespace fraction_div_addition_l18_18237

theorem fraction_div_addition : ( (3 / 7 : ℚ) / 4) + (1 / 28) = (1 / 7) :=
  sorry

end fraction_div_addition_l18_18237


namespace solve_equation_l18_18886

theorem solve_equation : ∀ (x : ℝ), (2 * x + 5 = 3 * x - 2) → (x = 7) :=
by
  intro x
  intro h
  sorry

end solve_equation_l18_18886


namespace spinner_three_digit_prob_div_by_4_l18_18486

theorem spinner_three_digit_prob_div_by_4 :
  (({x // x ∈ {1, 2, 4, 8} }) × ({y // y ∈ {1, 2, 4, 8} }) × ({z // z ∈ {1, 2, 4, 8} })).cardinal.to_rat ≠ 0 →
  (∃! p : ℚ, p = 11/16) :=
by
   intro h
   have total := Nat.cast 64 
   have favorable := Nat.cast 44
   have prob := favorable / total
   exists (11/16 : ℚ)
   split
   · exact prob
   · intro q hq
     simp [*] at *
#align spinner_three_digit_prob_div_by_4 spinner_three_digit_prob_div_by_4

end spinner_three_digit_prob_div_by_4_l18_18486


namespace payment_plan_months_l18_18953

theorem payment_plan_months 
  (M T : ℝ) (r : ℝ) 
  (hM : M = 100)
  (hT : T = 1320)
  (hr : r = 0.10)
  : ∃ t : ℕ, t = 12 ∧ T = (M * t) + (M * t * r) :=
by
  sorry

end payment_plan_months_l18_18953


namespace largest_prime_factor_1001_l18_18322

theorem largest_prime_factor_1001 : ∃ p, Nat.Prime p ∧ Nat.dvd p 1001 ∧ p = 13 :=
by
  sorry

end largest_prime_factor_1001_l18_18322


namespace sum_of_two_digit_divisors_l18_18501

theorem sum_of_two_digit_divisors (d : ℕ) (h1 : 145 % d = 4) (h2 : 10 ≤ d ∧ d < 100) :
  d = 47 :=
by
  have hd : d ∣ 141 := sorry
  exact sorry

end sum_of_two_digit_divisors_l18_18501


namespace smallest_positive_integer_a_l18_18810

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem smallest_positive_integer_a :
  ∃ (a : ℕ), 0 < a ∧ (isPerfectSquare (10 + a)) ∧ (isPerfectSquare (10 * a)) ∧ 
  ∀ b : ℕ, 0 < b ∧ (isPerfectSquare (10 + b)) ∧ (isPerfectSquare (10 * b)) → a ≤ b :=
sorry

end smallest_positive_integer_a_l18_18810


namespace evaluate_expression_l18_18650

theorem evaluate_expression : 
  let a := 45
  let b := 15
  (a + b)^2 - (a^2 + b^2 + 2 * a * 5) = 900 :=
by
  let a := 45
  let b := 15
  sorry

end evaluate_expression_l18_18650


namespace student_question_choice_l18_18930

/-- A student needs to choose 8 questions from part A and 5 questions from part B. Both parts contain 10 questions each.
   This Lean statement proves that the student can choose the questions in 11340 different ways. -/
theorem student_question_choice : (Nat.choose 10 8) * (Nat.choose 10 5) = 11340 := by
  sorry

end student_question_choice_l18_18930


namespace twenty_three_percent_of_number_is_forty_six_l18_18455

theorem twenty_three_percent_of_number_is_forty_six (x : ℝ) (h : (23 / 100) * x = 46) : x = 200 :=
sorry

end twenty_three_percent_of_number_is_forty_six_l18_18455


namespace largest_prime_factor_of_1001_l18_18320

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n then
    Classical.some (Nat.largest_prime_divisor n)
  else
    1

theorem largest_prime_factor_of_1001 : largest_prime_factor 1001 = 13 :=
by sorry

end largest_prime_factor_of_1001_l18_18320


namespace cubesWithTwoColoredFaces_l18_18644

structure CuboidDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

def numberOfSmallerCubes (d : CuboidDimensions) : ℕ :=
  d.length * d.width * d.height

def numberOfCubesWithTwoColoredFaces (d : CuboidDimensions) : ℕ :=
  2 * (d.length - 2) * 2 + 2 * (d.width - 2) * 2 + 2 * (d.height - 2) * 2

theorem cubesWithTwoColoredFaces :
  numberOfCubesWithTwoColoredFaces { length := 4, width := 3, height := 3 } = 16 := by
  sorry

end cubesWithTwoColoredFaces_l18_18644


namespace product_of_roots_l18_18809

theorem product_of_roots (x : ℝ) (h : x + 16 / x = 12) : (8 : ℝ) * (4 : ℝ) = 32 :=
by
  -- Your proof would go here
  sorry

end product_of_roots_l18_18809


namespace neznaika_mistake_l18_18292

-- Let's define the conditions
variables {X A Y M E O U : ℕ} -- Represents distinct digits

-- Ascending order of the numbers
variables (XA AY AX OY EM EY MU : ℕ)
  (h1 : XA < AY)
  (h2 : AY < AX)
  (h3 : AX < OY)
  (h4 : OY < EM)
  (h5 : EM < EY)
  (h6 : EY < MU)

-- Identical digits replaced with the same letters
variables (h7 : XA = 10 * X + A)
  (h8 : AY = 10 * A + Y)
  (h9 : AX = 10 * A + X)
  (h10 : OY = 10 * O + Y)
  (h11 : EM = 10 * E + M)
  (h12 : EY = 10 * E + Y)
  (h13 : MU = 10 * M + U)

-- Each letter represents a different digit
variables (h_distinct : X ≠ A ∧ X ≠ Y ∧ X ≠ M ∧ X ≠ E ∧ X ≠ O ∧ X ≠ U ∧
                       A ≠ Y ∧ A ≠ M ∧ A ≠ E ∧ A ≠ O ∧ A ≠ U ∧
                       Y ≠ M ∧ Y ≠ E ∧ Y ≠ O ∧ Y ≠ U ∧
                       M ≠ E ∧ M ≠ O ∧ M ≠ U ∧
                       E ≠ O ∧ E ≠ U ∧
                       O ≠ U)

-- Prove Neznaika made a mistake
theorem neznaika_mistake : false :=
by
  -- Here we'll reach a contradiction, proving false.
  sorry

end neznaika_mistake_l18_18292


namespace quadratic_inequality_solution_l18_18842

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l18_18842


namespace perfect_square_expression_l18_18909

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 :=
by
  sorry

end perfect_square_expression_l18_18909


namespace son_distance_from_father_is_correct_l18_18331

noncomputable def distance_between_son_and_father 
  (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1) 
  (incident_point_condition : F / d = L / (d + x) ∧ S / x = F / (d + x)) : ℝ :=
  4.9

theorem son_distance_from_father_is_correct (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1)
  (incident_point_condition : F / d = L / (d + 4.9) ∧ S / 4.9 = F / (d + 4.9)) : 
  distance_between_son_and_father L F S d h_L h_F h_S h_d incident_point_condition = 4.9 :=
sorry

end son_distance_from_father_is_correct_l18_18331


namespace volume_in_30_minutes_l18_18576

-- Define the conditions
def rate_of_pumping := 540 -- gallons per hour
def time_in_hours := 30 / 60 -- 30 minutes as a fraction of an hour

-- Define the volume pumped in 30 minutes
def volume_pumped := rate_of_pumping * time_in_hours

-- State the theorem
theorem volume_in_30_minutes : volume_pumped = 270 := by
  sorry

end volume_in_30_minutes_l18_18576


namespace competition_score_difference_l18_18854

theorem competition_score_difference :
  let perc_60 := 0.20
  let perc_75 := 0.25
  let perc_85 := 0.15
  let perc_90 := 0.30
  let perc_95 := 0.10
  let mean := (perc_60 * 60) + (perc_75 * 75) + (perc_85 * 85) + (perc_90 * 90) + (perc_95 * 95)
  let median := 85
  (median - mean = 5) := by
sorry

end competition_score_difference_l18_18854


namespace number_of_sets_satisfying_union_l18_18863

open Set

theorem number_of_sets_satisfying_union (M : Set ℤ) (hM : M = { x | -1 ≤ x ∧ x < 2 }) :
  {P : Set ℤ | P ⊆ M ∧ P ∪ M = M}.Finite.toFinset.card = 8 := by
{
  sorry
}

end number_of_sets_satisfying_union_l18_18863


namespace laura_owes_amount_l18_18932

-- Define the given conditions as variables
def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the interest calculation
def interest : ℝ := principal * rate * time

-- Define the final amount owed calculation
def amount_owed : ℝ := principal + interest

-- State the theorem we want to prove
theorem laura_owes_amount
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (interest : ℝ := principal * rate * time)
  (amount_owed : ℝ := principal + interest) :
  amount_owed = 36.75 := 
by 
  -- proof would go here
  sorry

end laura_owes_amount_l18_18932


namespace geometric_sequence_sum_l18_18859

theorem geometric_sequence_sum (S : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n : ℕ, n > 0 → S n = 2^n + a) →
  (S 1 = 2 + a) →
  (∀ n ≥ 2, a_n n = S n - S (n - 1)) →
  (a_n 1 = 1) →
  a = -1 :=
by
  sorry

end geometric_sequence_sum_l18_18859


namespace cost_of_weed_eater_string_l18_18631

-- Definitions
def num_blades := 4
def cost_per_blade := 8
def total_spent := 39
def total_cost_of_blades := num_blades * cost_per_blade
def cost_of_string := total_spent - total_cost_of_blades

-- The theorem statement
theorem cost_of_weed_eater_string : cost_of_string = 7 :=
by {
  -- The proof would go here
  sorry
}

end cost_of_weed_eater_string_l18_18631


namespace yearly_return_500_correct_l18_18031

noncomputable def yearly_return_500_investment : ℝ :=
  let total_investment : ℝ := 500 + 1500
  let combined_yearly_return : ℝ := 0.10 * total_investment
  let yearly_return_1500 : ℝ := 0.11 * 1500
  let yearly_return_500 : ℝ := combined_yearly_return - yearly_return_1500
  (yearly_return_500 / 500) * 100

theorem yearly_return_500_correct : yearly_return_500_investment = 7 :=
by
  sorry

end yearly_return_500_correct_l18_18031


namespace solve_equation_l18_18603

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l18_18603


namespace coaches_together_next_l18_18236

theorem coaches_together_next (a b c d : ℕ) (h_a : a = 5) (h_b : b = 9) (h_c : c = 8) (h_d : d = 11) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 3960 :=
by 
  rw [h_a, h_b, h_c, h_d]
  sorry

end coaches_together_next_l18_18236


namespace inequality_proof_l18_18816

open Real

theorem inequality_proof (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
    ( (2 * a + b + c)^2 / (2 * a^2 + (b + c)^2) ) +
    ( (2 * b + c + a)^2 / (2 * b^2 + (c + a)^2) ) +
    ( (2 * c + a + b)^2 / (2 * c^2 + (a + b)^2) ) ≤ 8 :=
by
  sorry

end inequality_proof_l18_18816


namespace minimum_value_proof_l18_18668

noncomputable def minValue (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : ℝ := 
  (x + 8 * y) / (x * y)

theorem minimum_value_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : 
  minValue x y hx hy h = 9 := 
by
  sorry

end minimum_value_proof_l18_18668


namespace xy_difference_l18_18130

theorem xy_difference (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) (h3 : x = 15) : x - y = 10 :=
by
  sorry

end xy_difference_l18_18130


namespace snail_stops_at_25_26_l18_18780

def grid_width : ℕ := 300
def grid_height : ℕ := 50

def initial_position : ℕ × ℕ := (1, 1)

def snail_moves_in_spiral (w h : ℕ) (initial : ℕ × ℕ) : ℕ × ℕ := (25, 26)

theorem snail_stops_at_25_26 :
  snail_moves_in_spiral grid_width grid_height initial_position = (25, 26) :=
sorry

end snail_stops_at_25_26_l18_18780


namespace trapezoid_area_l18_18190

theorem trapezoid_area (h : ℝ) : 
  let b1 : ℝ := 4 * h + 2
  let b2 : ℝ := 5 * h
  (b1 + b2) / 2 * h = (9 * h ^ 2 + 2 * h) / 2 :=
by 
  let b1 := 4 * h + 2
  let b2 := 5 * h
  sorry

end trapezoid_area_l18_18190


namespace part_a_part_b_l18_18056

-- Definition for bishops not attacking each other
def bishops_safe (positions : List (ℕ × ℕ)) : Prop :=
  ∀ (b1 b2 : ℕ × ℕ), b1 ∈ positions → b2 ∈ positions → b1 ≠ b2 → 
    (b1.1 + b1.2 ≠ b2.1 + b2.2) ∧ (b1.1 - b1.2 ≠ b2.1 - b2.2)

-- Part (a): 14 bishops on an 8x8 chessboard such that no two attack each other
theorem part_a : ∃ (positions : List (ℕ × ℕ)), positions.length = 14 ∧ bishops_safe positions := 
by
  sorry

-- Part (b): It is impossible to place 15 bishops on an 8x8 chessboard without them attacking each other
theorem part_b : ¬ ∃ (positions : List (ℕ × ℕ)), positions.length = 15 ∧ bishops_safe positions :=
by 
  sorry

end part_a_part_b_l18_18056


namespace total_flowers_eaten_l18_18291

theorem total_flowers_eaten :
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  f1 + f2 + f3 + f4 + f5 + f6 + f7 = 16.5 :=
by
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  sorry

end total_flowers_eaten_l18_18291


namespace count_digit_9_from_1_to_1000_l18_18113

theorem count_digit_9_from_1_to_1000 : (list.range 1000).countp (λ n, n.digits 10).contains 9 = 301 := 
sorry

end count_digit_9_from_1_to_1000_l18_18113


namespace part_I_part_II_l18_18276

noncomputable def f (x b c : ℝ) := x^2 + b*x + c

theorem part_I (x_1 x_2 b c : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) :
  b^2 > 2 * (b + 2 * c) :=
sorry

theorem part_II (x_1 x_2 b c t : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) (h5 : 0 < t ∧ t < x_1) :
  f t b c > x_1 :=
sorry

end part_I_part_II_l18_18276


namespace value_of_f_15_l18_18690

def f (n : ℕ) : ℕ := n^2 + 2*n + 19

theorem value_of_f_15 : f 15 = 274 := 
by 
  -- Add proof here
  sorry

end value_of_f_15_l18_18690


namespace no_difference_of_squares_equals_222_l18_18205

theorem no_difference_of_squares_equals_222 (a b : ℤ) : a^2 - b^2 ≠ 222 := 
  sorry

end no_difference_of_squares_equals_222_l18_18205


namespace fare_range_l18_18267

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 8 else 8 + 1.5 * (x - 3)

theorem fare_range (x : ℝ) (hx : fare x = 16) : 8 ≤ x ∧ x < 9 :=
by
  sorry

end fare_range_l18_18267


namespace uniq_increasing_seq_l18_18652

noncomputable def a (n : ℕ) : ℕ := n -- The correct sequence a_n = n

theorem uniq_increasing_seq (a : ℕ → ℕ)
  (h1 : a 2 = 2)
  (h2 : ∀ n m : ℕ, a (n * m) = a n * a m)
  (h_inc : ∀ n m : ℕ, n < m → a n < a m) : ∀ n : ℕ, a n = n := by
  -- Here we would place the proof, skipping it for now with sorry
  sorry

end uniq_increasing_seq_l18_18652


namespace solve_equation_l18_18602

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l18_18602


namespace ages_proof_l18_18451

noncomputable def A : ℝ := 12.1
noncomputable def B : ℝ := 6.1
noncomputable def C : ℝ := 11.3

-- Conditions extracted from the problem
def sum_of_ages (A B C : ℝ) : Prop := A + B + C = 29.5
def specific_age (C : ℝ) : Prop := C = 11.3
def twice_as_old (A B : ℝ) : Prop := A = 2 * B

theorem ages_proof : 
  ∃ (A B C : ℝ), 
    specific_age C ∧ twice_as_old A B ∧ sum_of_ages A B C :=
by
  exists 12.1, 6.1, 11.3
  sorry

end ages_proof_l18_18451


namespace largest_prime_factor_1001_l18_18321

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l18_18321


namespace solve_equation_l18_18587

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l18_18587


namespace table_capacity_l18_18038

def invited_people : Nat := 18
def no_show_people : Nat := 12
def number_of_tables : Nat := 2
def attendees := invited_people - no_show_people
def people_per_table : Nat := attendees / number_of_tables

theorem table_capacity : people_per_table = 3 :=
by
  sorry

end table_capacity_l18_18038


namespace finalCostCalculation_l18_18755

-- Define the inputs
def tireRepairCost : ℝ := 7
def salesTaxPerTire : ℝ := 0.50
def numberOfTires : ℕ := 4

-- The total cost should be $30
theorem finalCostCalculation : 
  let repairTotal := tireRepairCost * numberOfTires
  let salesTaxTotal := salesTaxPerTire * numberOfTires
  repairTotal + salesTaxTotal = 30 := 
by {
  sorry
}

end finalCostCalculation_l18_18755


namespace prime_solution_exists_l18_18505

theorem prime_solution_exists :
  ∃ (p q r : ℕ), p.Prime ∧ q.Prime ∧ r.Prime ∧ (p + q^2 = r^4) ∧ (p = 7) ∧ (q = 3) ∧ (r = 2) := 
by
  sorry

end prime_solution_exists_l18_18505


namespace solve_k_values_l18_18775

def has_positive_integer_solution (k : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = k * a * b * c

def infinitely_many_solutions (k : ℕ) : Prop :=
  ∃ (a b c : ℕ → ℕ), (∀ n, a n > 0 ∧ b n > 0 ∧ c n > 0 ∧ a n^2 + b n^2 + c n^2 = k * a n * b n * c n) ∧
  (∀ n, ∃ x y: ℤ, x^2 + y^2 = (a n * b n))

theorem solve_k_values :
  ∃ k : ℕ, (k = 1 ∨ k = 3) ∧ has_positive_integer_solution k ∧ infinitely_many_solutions k :=
sorry

end solve_k_values_l18_18775


namespace fraction_bad_teams_leq_l18_18543

variable (teams total_teams : ℕ) (b : ℝ)

-- Given conditions
variable (cond₁ : total_teams = 18)
variable (cond₂ : teams = total_teams / 2)
variable (cond₃ : ∀ (rb_teams : ℕ), rb_teams ≠ 10 → rb_teams ≤ teams)

theorem fraction_bad_teams_leq (H : 18 * b ≤ teams) : b ≤ 1 / 2 :=
sorry

end fraction_bad_teams_leq_l18_18543


namespace problem_1_problem_2_problem_3_l18_18558

def M := {n : ℕ | 0 < n ∧ n < 1000}

def circ (a b : ℕ) : ℕ :=
  if a * b < 1000 then a * b
  else 
    let k := (a * b) / 1000
    let r := (a * b) % 1000
    if k + r < 1000 then k + r
    else (k + r) % 1000 + 1

theorem problem_1 : circ 559 758 = 146 := 
by
  sorry

theorem problem_2 : ∃ (x : ℕ) (h : x ∈ M), circ 559 x = 1 ∧ x = 361 :=
by
  sorry

theorem problem_3 : ∀ (a b c : ℕ) (h₁ : a ∈ M) (h₂ : b ∈ M) (h₃ : c ∈ M), circ a (circ b c) = circ (circ a b) c :=
by
  sorry

end problem_1_problem_2_problem_3_l18_18558


namespace sum_of_f_greater_than_zero_l18_18252

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem sum_of_f_greater_than_zero 
  (a b c : ℝ) 
  (h1 : a + b > 0) 
  (h2 : b + c > 0) 
  (h3 : c + a > 0) : 
  f a + f b + f c > 0 := 
by 
  sorry

end sum_of_f_greater_than_zero_l18_18252


namespace quadratic_inequality_l18_18848

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l18_18848


namespace chocolates_problem_l18_18034

theorem chocolates_problem 
  (N : ℕ)
  (h1 : ∃ R C : ℕ, N = R * C)
  (h2 : ∃ r1, r1 = 3 * C - 1)
  (h3 : ∃ c1, c1 = R * 5 - 1)
  (h4 : ∃ r2 c2 : ℕ, r2 = 3 ∧ c2 = 5 ∧ r2 * c2 = r1 - 1)
  (h5 : N = 3 * (3 * C - 1))
  (h6 : N / 3 = (12 * 5) / 3) : 
  N = 60 ∧ (N - (3 * C - 1)) = 25 :=
by 
  sorry

end chocolates_problem_l18_18034


namespace barkley_total_net_buried_bones_l18_18053

def monthly_bones_received (months : ℕ) : (ℕ × ℕ × ℕ) := (10 * months, 6 * months, 4 * months)

def burying_pattern_A (months : ℕ) : ℕ := 6 * months
def eating_pattern_A (months : ℕ) : ℕ := if months > 2 then 3 else 1

def burying_pattern_B (months : ℕ) : ℕ := if months = 5 then 0 else 4 * (months - 1)
def eating_pattern_B (months : ℕ) : ℕ := 2

def burying_pattern_C (months : ℕ) : ℕ := 2 * months
def eating_pattern_C (months : ℕ) : ℕ := 2

def total_net_buried_bones (months : ℕ) : ℕ :=
  let (received_A, received_B, received_C) := monthly_bones_received months
  let net_A := burying_pattern_A months - eating_pattern_A months
  let net_B := burying_pattern_B months - eating_pattern_B months
  let net_C := burying_pattern_C months - eating_pattern_C months
  net_A + net_B + net_C

theorem barkley_total_net_buried_bones : total_net_buried_bones 5 = 49 := by
  sorry

end barkley_total_net_buried_bones_l18_18053


namespace binom_9_5_eq_126_l18_18988

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l18_18988


namespace axis_center_symmetry_sine_shifted_l18_18209
  noncomputable def axis_of_symmetry (k : ℤ) : ℝ := 3 * Real.pi / 4 + k * Real.pi

  noncomputable def center_of_symmetry (k : ℤ) : ℝ × ℝ := (Real.pi / 4 + k * Real.pi, 0)

  theorem axis_center_symmetry_sine_shifted :
    ∀ (k : ℤ),
    ∃ x y : ℝ,
      (x = axis_of_symmetry k) ∧ (y = 0) ∧ (y, 0) = center_of_symmetry k := 
  sorry
  
end axis_center_symmetry_sine_shifted_l18_18209


namespace range_of_a_l18_18173

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → (x - 1) ^ 2 < Real.log x / Real.log a) → a ∈ Set.Ioc 1 2 :=
by
  sorry

end range_of_a_l18_18173


namespace calculate_exponent_product_l18_18203

theorem calculate_exponent_product : (3^3 * 5^3) * (3^8 * 5^8) = 15^11 := by
  sorry

end calculate_exponent_product_l18_18203


namespace abby_bridget_chris_probability_l18_18297

noncomputable def seatingProbability : ℚ :=
  let totalArrangements := 720
  let favorableArrangements := 114
  favorableArrangements / totalArrangements

theorem abby_bridget_chris_probability :
  seatingProbability = 19 / 120 :=
by
  simp [seatingProbability]
  sorry

end abby_bridget_chris_probability_l18_18297


namespace binom_9_5_l18_18992

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l18_18992


namespace find_x_l18_18372

-- Let's define the constants and the condition
def a : ℝ := 2.12
def b : ℝ := 0.345
def c : ℝ := 2.4690000000000003

-- We need to prove that there exists a number x such that
def x : ℝ := 0.0040000000000003

-- Formal statement
theorem find_x : a + b + x = c :=
by
  -- Proof skipped
  sorry
 
end find_x_l18_18372


namespace division_of_decimals_l18_18017

theorem division_of_decimals : 0.25 / 0.005 = 50 := 
by
  sorry

end division_of_decimals_l18_18017


namespace sum_multiple_of_three_l18_18157

theorem sum_multiple_of_three (a b : ℤ) (h₁ : ∃ m, a = 6 * m) (h₂ : ∃ n, b = 9 * n) : ∃ k, (a + b) = 3 * k :=
by
  sorry

end sum_multiple_of_three_l18_18157


namespace find_additional_discount_l18_18350

noncomputable def calculate_additional_discount (msrp : ℝ) (regular_discount_percent : ℝ) (final_price : ℝ) : ℝ :=
  let regular_discounted_price := msrp * (1 - regular_discount_percent / 100)
  let additional_discount_percent := ((regular_discounted_price - final_price) / regular_discounted_price) * 100
  additional_discount_percent

theorem find_additional_discount :
  calculate_additional_discount 35 30 19.6 = 20 :=
by
  sorry

end find_additional_discount_l18_18350


namespace problem_statement_l18_18122

variable {x a : Real}

theorem problem_statement (h1 : x < a) (h2 : a < 0) : x^2 > a * x ∧ a * x > a^2 := 
sorry

end problem_statement_l18_18122


namespace find_number_l18_18620

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 150) : N = 288 := by
  sorry

end find_number_l18_18620


namespace greg_attendance_probability_l18_18678

theorem greg_attendance_probability :
  let P_Rain := 0.5
  let P_Attend_if_Rain := 0.3
  let P_Sunny := 1 - P_Rain
  let P_Attend_if_Sunny := 0.9
  let P_Attend := P_Rain * P_Attend_if_Rain + P_Sunny * P_Attend_if_Sunny
  in P_Attend = 0.6 :=
by
  sorry

end greg_attendance_probability_l18_18678


namespace rectangular_prism_parallel_edges_l18_18681

theorem rectangular_prism_parallel_edges (length width height : ℕ) (h1 : length ≠ width) (h2 : width ≠ height) (h3 : length ≠ height) : 
  ∃ pairs : ℕ, pairs = 6 := by
  sorry

end rectangular_prism_parallel_edges_l18_18681


namespace total_amount_paid_is_correct_l18_18659

-- Define the initial conditions
def tireA_price : ℕ := 75
def tireA_discount : ℕ := 20
def tireB_price : ℕ := 90
def tireB_discount : ℕ := 30
def tireC_price : ℕ := 120
def tireC_discount : ℕ := 45
def tireD_price : ℕ := 150
def tireD_discount : ℕ := 60
def installation_fee : ℕ := 15
def disposal_fee : ℕ := 5

-- Calculate the total amount paid
def total_paid : ℕ :=
  let tireA_total := (tireA_price - tireA_discount) + installation_fee + disposal_fee
  let tireB_total := (tireB_price - tireB_discount) + installation_fee + disposal_fee
  let tireC_total := (tireC_price - tireC_discount) + installation_fee + disposal_fee
  let tireD_total := (tireD_price - tireD_discount) + installation_fee + disposal_fee
  tireA_total + tireB_total + tireC_total + tireD_total

-- Statement of the theorem
theorem total_amount_paid_is_correct :
  total_paid = 360 :=
by
  -- proof goes here
  sorry

end total_amount_paid_is_correct_l18_18659


namespace sum_of_units_digits_eq_0_l18_18655

-- Units digit function definition
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement in Lean 
theorem sum_of_units_digits_eq_0 :
  units_digit (units_digit (17 * 34) + units_digit (19 * 28)) = 0 :=
by
  sorry

end sum_of_units_digits_eq_0_l18_18655


namespace crackers_per_friend_l18_18874

theorem crackers_per_friend (Total_crackers Left_crackers Friends : ℕ) (h1 : Total_crackers = 23) (h2 : Left_crackers = 11) (h3 : Friends = 2):
  (Total_crackers - Left_crackers) / Friends = 6 :=
by
  sorry

end crackers_per_friend_l18_18874


namespace sample_size_calculation_l18_18342

theorem sample_size_calculation 
    (total_teachers : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) 
    (sample_size_female_students : ℕ) 
    (H1 : total_teachers = 100) (H2 : total_male_students = 600) 
    (H3 : total_female_students = 500) (H4 : sample_size_female_students = 40)
    : (sample_size_female_students * (total_teachers + total_male_students + total_female_students) / total_female_students) = 96 := 
by
  /- sorry, proof omitted -/
  sorry
  
end sample_size_calculation_l18_18342


namespace domain_of_f_l18_18821

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.log (x + 2)

theorem domain_of_f :
  {x : ℝ | (x ≠ 0) ∧ (x > -2)} = {x : ℝ | (-2 < x ∧ x < 0) ∨ (0 < x)} :=
by
  sorry

end domain_of_f_l18_18821


namespace smallest_m_inequality_l18_18934

theorem smallest_m_inequality (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end smallest_m_inequality_l18_18934


namespace min_value_f_l18_18662

theorem min_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : 
  ∃ z : ℝ, z = x + y + x * y ∧ z = -9/8 :=
by 
  sorry

end min_value_f_l18_18662


namespace price_of_large_slice_is_250_l18_18357

noncomputable def priceOfLargeSlice (totalSlices soldSmallSlices totalRevenue smallSlicePrice: ℕ) : ℕ :=
  let totalRevenueSmallSlices := soldSmallSlices * smallSlicePrice
  let totalRevenueLargeSlices := totalRevenue - totalRevenueSmallSlices
  let soldLargeSlices := totalSlices - soldSmallSlices
  totalRevenueLargeSlices / soldLargeSlices

theorem price_of_large_slice_is_250 :
  priceOfLargeSlice 5000 2000 1050000 150 = 250 :=
by
  sorry

end price_of_large_slice_is_250_l18_18357


namespace largest_prime_factor_1001_l18_18323

theorem largest_prime_factor_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_1001_l18_18323


namespace lincoln_county_houses_l18_18314

theorem lincoln_county_houses (original_houses : ℕ) (built_houses : ℕ) (total_houses : ℕ) 
(h1 : original_houses = 20817) 
(h2 : built_houses = 97741) 
(h3 : total_houses = original_houses + built_houses) : 
total_houses = 118558 :=
by
  -- proof omitted
  sorry

end lincoln_county_houses_l18_18314


namespace monkey_reach_top_in_20_hours_l18_18349

-- Defining the conditions
def tree_height : ℕ := 21
def hop_distance : ℕ := 3
def slip_distance : ℕ := 2

-- Defining the net distance gain per hour
def net_gain_per_hour : ℕ := hop_distance - slip_distance

-- Proof statement
theorem monkey_reach_top_in_20_hours :
  ∃ t : ℕ, t = 20 ∧ 20 * net_gain_per_hour + hop_distance = tree_height :=
by
  sorry

end monkey_reach_top_in_20_hours_l18_18349


namespace S_4n_l18_18514

variable {a : ℕ → ℕ}
variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (r : ℝ)
variable (a1 : ℝ)

-- Conditions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * r
axiom positive_terms : ∀ n, 0 < a n
axiom sum_n : S n = a1 * (1 - r^n) / (1 - r)
axiom sum_3n : S (3 * n) = 14
axiom sum_n_value : S n = 2

-- Theorem
theorem S_4n : S (4 * n) = 30 :=
sorry

end S_4n_l18_18514


namespace range_of_a_no_fixed_points_l18_18077

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1

theorem range_of_a_no_fixed_points : 
  ∀ a : ℝ, ¬∃ x : ℝ, f x a = x ↔ -1 < a ∧ a < 3 :=
by sorry

end range_of_a_no_fixed_points_l18_18077


namespace sin_of_angle_in_first_quadrant_l18_18383

theorem sin_of_angle_in_first_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 3 / 4) : Real.sin α = 3 / 5 :=
by
  sorry

end sin_of_angle_in_first_quadrant_l18_18383


namespace ab_minus_a_plus_b_eq_two_l18_18669

theorem ab_minus_a_plus_b_eq_two
  (a b : ℝ)
  (h1 : a + 1 ≠ 0)
  (h2 : b - 1 ≠ 0)
  (h3 : a + (1 / (a + 1)) = b + (1 / (b - 1)) - 2)
  (h4 : a - b + 2 ≠ 0)
: ab - a + b = 2 :=
sorry

end ab_minus_a_plus_b_eq_two_l18_18669


namespace converse_of_prop1_true_l18_18251

theorem converse_of_prop1_true
  (h1 : ∀ {x : ℝ}, x^2 - 3 * x + 2 = 0 → x = 1 ∨ x = 2)
  (h2 : ∀ {x : ℝ}, -2 ≤ x ∧ x < 3 → (x - 2) * (x - 3) ≤ 0)
  (h3 : ∀ {x y : ℝ}, x = 0 ∧ y = 0 → x^2 + y^2 = 0)
  (h4 : ∀ {x y : ℕ}, x > 0 ∧ y > 0 ∧ (x + y) % 2 = 1 → (x % 2 = 1 ∧ y % 2 = 0) ∨ (x % 2 = 0 ∧ y % 2 = 1)) :
  (∀ {x : ℝ}, x = 1 ∨ x = 2 → x^2 - 3 * x + 2 = 0) :=
by
  sorry

end converse_of_prop1_true_l18_18251


namespace initial_value_divisible_by_456_l18_18172

def initial_value := 374
def to_add := 82
def divisor := 456

theorem initial_value_divisible_by_456 : (initial_value + to_add) % divisor = 0 := by
  sorry

end initial_value_divisible_by_456_l18_18172


namespace min_value_f_l18_18663

theorem min_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : 
  ∃ z : ℝ, z = x + y + x * y ∧ z = -9/8 :=
by 
  sorry

end min_value_f_l18_18663


namespace quadratic_inequality_solution_l18_18838

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l18_18838


namespace compute_fraction_l18_18366

theorem compute_fraction :
  ((1/3)^4 * (1/5) = (1/405)) :=
by
  sorry

end compute_fraction_l18_18366


namespace binom_9_5_l18_18994

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l18_18994


namespace perfect_square_of_expression_l18_18903

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end perfect_square_of_expression_l18_18903


namespace probability_no_3x3_red_square_l18_18222

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l18_18222


namespace width_at_bottom_of_stream_l18_18933

theorem width_at_bottom_of_stream 
    (top_width : ℝ) (area : ℝ) (height : ℝ) (bottom_width : ℝ) :
    top_width = 10 → area = 640 → height = 80 → 
    2 * area = height * (top_width + bottom_width) → 
    bottom_width = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Finding bottom width
  have h5 : 2 * 640 = 80 * (10 + bottom_width) := h4
  norm_num at h5
  linarith [h5]

#check width_at_bottom_of_stream

end width_at_bottom_of_stream_l18_18933


namespace simplify_one_simplify_two_simplify_three_simplify_four_l18_18430

-- (1) Prove that (1 / 2) * sqrt(4 / 7) = sqrt(7) / 7
theorem simplify_one : (1 / 2) * Real.sqrt (4 / 7) = Real.sqrt 7 / 7 := sorry

-- (2) Prove that sqrt(20 ^ 2 - 15 ^ 2) = 5 * sqrt(7)
theorem simplify_two : Real.sqrt (20 ^ 2 - 15 ^ 2) = 5 * Real.sqrt 7 := sorry

-- (3) Prove that sqrt((32 * 9) / 25) = (12 * sqrt(2)) / 5
theorem simplify_three : Real.sqrt ((32 * 9) / 25) = (12 * Real.sqrt 2) / 5 := sorry

-- (4) Prove that sqrt(22.5) = (3 * sqrt(10)) / 2
theorem simplify_four : Real.sqrt 22.5 = (3 * Real.sqrt 10) / 2 := sorry

end simplify_one_simplify_two_simplify_three_simplify_four_l18_18430


namespace fraction_simplifies_correctly_l18_18689

variable (a b : ℕ)

theorem fraction_simplifies_correctly (h : a ≠ b) : (1/2 * a) / (1/2 * b) = a / b := 
by sorry

end fraction_simplifies_correctly_l18_18689


namespace translated_line_expression_l18_18265

theorem translated_line_expression (x y : ℝ) (b : ℝ) :
  (∀ x y, y = 2 * x + 3 ∧ (5, 1).2 = 2 * (5, 1).1 + b) → y = 2 * x - 9 :=
by
  sorry

end translated_line_expression_l18_18265


namespace total_hamburgers_sold_is_63_l18_18946

-- Define the average number of hamburgers sold each day
def avg_hamburgers_per_day : ℕ := 9

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total hamburgers sold in a week
def total_hamburgers_sold : ℕ := avg_hamburgers_per_day * days_in_week

-- Prove that the total hamburgers sold in a week is 63
theorem total_hamburgers_sold_is_63 : total_hamburgers_sold = 63 :=
by
  sorry

end total_hamburgers_sold_is_63_l18_18946


namespace capacity_of_new_vessel_is_10_l18_18637

-- Define the conditions
def first_vessel_capacity : ℕ := 2
def first_vessel_concentration : ℚ := 0.25
def second_vessel_capacity : ℕ := 6
def second_vessel_concentration : ℚ := 0.40
def total_liquid_combined : ℕ := 8
def new_mixture_concentration : ℚ := 0.29
def total_alcohol_content : ℚ := (first_vessel_capacity * first_vessel_concentration) + (second_vessel_capacity * second_vessel_concentration)
def desired_vessel_capacity : ℚ := total_alcohol_content / new_mixture_concentration

-- The theorem we want to prove
theorem capacity_of_new_vessel_is_10 : desired_vessel_capacity = 10 := by
  sorry

end capacity_of_new_vessel_is_10_l18_18637


namespace equilateral_triangle_area_decrease_l18_18049

theorem equilateral_triangle_area_decrease (s : ℝ) (A : ℝ) (s_new : ℝ) (A_new : ℝ)
    (hA : A = 100 * Real.sqrt 3)
    (hs : s^2 = 400)
    (hs_new : s_new = s - 6)
    (hA_new : A_new = (Real.sqrt 3 / 4) * s_new^2) :
    (A - A_new) / A * 100 = 51 := by
  sorry

end equilateral_triangle_area_decrease_l18_18049


namespace find_multiplier_l18_18483

theorem find_multiplier (A N : ℕ) (h : A = 32) (eqn : N * (A + 4) - 4 * (A - 4) = A) : N = 4 :=
sorry

end find_multiplier_l18_18483


namespace beef_weight_after_processing_l18_18485

def original_weight : ℝ := 861.54
def weight_loss_percentage : ℝ := 0.35
def retained_percentage : ℝ := 1 - weight_loss_percentage
def weight_after_processing (w : ℝ) := retained_percentage * w

theorem beef_weight_after_processing :
  weight_after_processing original_weight = 560.001 :=
by
  sorry

end beef_weight_after_processing_l18_18485


namespace rational_expr_evaluation_l18_18248

theorem rational_expr_evaluation (a b c : ℚ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) (h2 : a + b + c = a * b * c) :
  (a / b + a / c + b / a + b / c + c / a + c / b - a * b - b * c - c * a) = -3 :=
by
  sorry

end rational_expr_evaluation_l18_18248


namespace angle_between_vectors_is_90_degrees_l18_18380

noncomputable def vec_angle (v₁ v₂ : ℝ × ℝ) : ℝ :=
sorry -- This would be the implementation that calculates the angle between two vectors

theorem angle_between_vectors_is_90_degrees
  (A B C O : ℝ × ℝ)
  (h1 : dist O A = dist O B)
  (h2 : dist O A = dist O C)
  (h3 : dist O B = dist O C)
  (h4 : 2 • (A - O) = (B - O) + (C - O)) :
  vec_angle (B - A) (C - A) = 90 :=
sorry

end angle_between_vectors_is_90_degrees_l18_18380


namespace remainder_when_divided_by_2_is_0_l18_18464

theorem remainder_when_divided_by_2_is_0 (n : ℕ)
  (h1 : ∃ r, n % 2 = r)
  (h2 : n % 7 = 5)
  (h3 : ∃ p, p = 5 ∧ (n + p) % 10 = 0) :
  n % 2 = 0 :=
by
  -- skipping the proof steps; hence adding sorry
  sorry

end remainder_when_divided_by_2_is_0_l18_18464


namespace apple_price_equals_oranges_l18_18448

theorem apple_price_equals_oranges (A O : ℝ) (H1 : A = 28 * O) (H2 : 45 * A + 60 * O = 1350) (H3 : 30 * A + 40 * O = 900) : A = 28 * O :=
by
  sorry

end apple_price_equals_oranges_l18_18448


namespace readers_all_three_l18_18402

def total_readers : ℕ := 500
def readers_science_fiction : ℕ := 320
def readers_literary_works : ℕ := 200
def readers_non_fiction : ℕ := 150
def readers_sf_and_lw : ℕ := 120
def readers_sf_and_nf : ℕ := 80
def readers_lw_and_nf : ℕ := 60

theorem readers_all_three :
  total_readers = readers_science_fiction + readers_literary_works + readers_non_fiction - (readers_sf_and_lw + readers_sf_and_nf + readers_lw_and_nf) + 90 :=
by
  sorry

end readers_all_three_l18_18402


namespace count_integers_with_same_remainder_l18_18101

theorem count_integers_with_same_remainder (n : ℤ) : 
  (150 < n ∧ n < 250) ∧ 
  (∃ r : ℤ, 0 ≤ r ∧ r ≤ 6 ∧ ∃ a b : ℤ, n = 7 * a + r ∧ n = 9 * b + r) ↔ n = 7 :=
sorry

end count_integers_with_same_remainder_l18_18101


namespace measure_of_angle_C_l18_18400

theorem measure_of_angle_C (A B C : ℕ) (h1 : A + B = 150) (h2 : A + B + C = 180) : C = 30 := 
by
  sorry

end measure_of_angle_C_l18_18400


namespace distance_to_grandmas_house_is_78_l18_18151

-- Define the conditions
def miles_to_pie_shop : ℕ := 35
def miles_to_gas_station : ℕ := 18
def miles_remaining : ℕ := 25

-- Define the mathematical claim
def total_distance_to_grandmas_house : ℕ :=
  miles_to_pie_shop + miles_to_gas_station + miles_remaining

-- Prove the claim
theorem distance_to_grandmas_house_is_78 :
  total_distance_to_grandmas_house = 78 :=
by
  sorry

end distance_to_grandmas_house_is_78_l18_18151


namespace smallest_n_boxes_l18_18370

theorem smallest_n_boxes (n : ℕ) : (15 * n - 1) % 11 = 0 ↔ n = 3 :=
by
  sorry

end smallest_n_boxes_l18_18370


namespace tom_buys_papayas_l18_18454

-- Defining constants for the costs of each fruit
def lemon_cost : ℕ := 2
def papaya_cost : ℕ := 1
def mango_cost : ℕ := 4

-- Defining the number of each fruit Tom buys
def lemons_bought : ℕ := 6
def mangos_bought : ℕ := 2
def total_paid : ℕ := 21

-- Defining the function to calculate the total cost 
def total_cost (P : ℕ) : ℕ := (lemons_bought * lemon_cost) + (mangos_bought * mango_cost) + (P * papaya_cost)

-- Defining the function to calculate the discount based on the total number of fruits
def discount (P : ℕ) : ℕ := (lemons_bought + mangos_bought + P) / 4

-- Main theorem to prove
theorem tom_buys_papayas (P : ℕ) : total_cost P - discount P = total_paid → P = 4 := 
by
  intro h
  sorry

end tom_buys_papayas_l18_18454


namespace solve_for_c_l18_18263

theorem solve_for_c (a b c : ℝ) (h : 1/a - 1/b = 2/c) : c = (a * b * (b - a)) / 2 := by
  sorry

end solve_for_c_l18_18263


namespace soccer_team_lineups_l18_18731

-- Define the number of players in the team
def numPlayers : Nat := 16

-- Define the number of regular players to choose (excluding the goalie)
def numRegularPlayers : Nat := 10

-- Define the total number of starting lineups, considering the goalie and the combination of regular players
def totalStartingLineups : Nat :=
  numPlayers * Nat.choose (numPlayers - 1) numRegularPlayers

-- The theorem to prove
theorem soccer_team_lineups : totalStartingLineups = 48048 := by
  sorry

end soccer_team_lineups_l18_18731


namespace medal_awarding_ways_l18_18310

def num_sprinters := 10
def num_americans := 4
def num_kenyans := 2
def medal_positions := 3 -- gold, silver, bronze

-- The main statement to be proven
theorem medal_awarding_ways :
  let ways_case1 := 2 * 3 * 5 * 4
  let ways_case2 := 4 * 3 * 2 * 2 * 5
  ways_case1 + ways_case2 = 360 :=
by
  sorry

end medal_awarding_ways_l18_18310


namespace compute_combination_l18_18981

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l18_18981


namespace maximum_distance_value_of_m_l18_18388

-- Define the line equation
def line_eq (m x y : ℝ) : Prop := y = m * x - m - 1

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the problem statement
theorem maximum_distance_value_of_m :
  ∃ (m : ℝ), (∀ x y : ℝ, circle_eq x y → ∃ P : ℝ × ℝ, line_eq m P.fst P.snd) →
  m = -0.5 :=
sorry

end maximum_distance_value_of_m_l18_18388


namespace time_upstream_equal_nine_hours_l18_18162

noncomputable def distance : ℝ := 126
noncomputable def time_downstream : ℝ := 7
noncomputable def current_speed : ℝ := 2
noncomputable def downstream_speed := distance / time_downstream
noncomputable def boat_speed := downstream_speed - current_speed
noncomputable def upstream_speed := boat_speed - current_speed

theorem time_upstream_equal_nine_hours : (distance / upstream_speed) = 9 := by
  sorry

end time_upstream_equal_nine_hours_l18_18162


namespace probability_at_least_one_hit_l18_18529

theorem probability_at_least_one_hit (pA pB pC : ℝ) (hA : pA = 0.7) (hB : pB = 0.5) (hC : pC = 0.4) : 
  (1 - ((1 - pA) * (1 - pB) * (1 - pC))) = 0.91 :=
by
  sorry

end probability_at_least_one_hit_l18_18529


namespace solution_of_inequality_l18_18685

theorem solution_of_inequality (a x : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  (x - a) * (x - a⁻¹) < 0 ↔ a < x ∧ x < a⁻¹ :=
by sorry

end solution_of_inequality_l18_18685


namespace probability_no_3x3_red_square_l18_18231

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l18_18231


namespace binom_9_5_eq_126_l18_18975

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l18_18975


namespace customer_paid_amount_l18_18744

def cost_price : Real := 7239.13
def percentage_increase : Real := 0.15
def selling_price := (1 + percentage_increase) * cost_price

theorem customer_paid_amount :
  selling_price = 8325.00 :=
by
  sorry

end customer_paid_amount_l18_18744


namespace compute_combination_l18_18983

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l18_18983


namespace composite_quadratic_l18_18004

theorem composite_quadratic (a b : Int) (x1 x2 : Int)
  (h1 : x1 + x2 = -a)
  (h2 : x1 * x2 = b)
  (h3 : abs x1 > 2)
  (h4 : abs x2 > 2) :
  ∃ m n : Int, a + b + 1 = m * n ∧ m > 1 ∧ n > 1 :=
by
  sorry

end composite_quadratic_l18_18004


namespace essay_body_section_length_l18_18363

theorem essay_body_section_length :
  ∀ (intro_length conclusion_multiplier : ℕ) (body_sections total_words : ℕ),
  intro_length = 450 →
  conclusion_multiplier = 3 →
  body_sections = 4 →
  total_words = 5000 →
  let conclusion_length := conclusion_multiplier * intro_length in
  let total_intro_conclusion := intro_length + conclusion_length in
  let remaining_words := total_words - total_intro_conclusion in
  let section_length := remaining_words / body_sections in
  section_length = 800 :=
by 
  intros intro_length conclusion_multiplier body_sections total_words 
         h_intro_len h_concl_mul h_body_sec h_total_words;
  dsimp only;
  rw [h_intro_len, h_concl_mul, h_body_sec, h_total_words];
  let conclusion_length := 3 * 450;
  let total_intro_conclusion := 450 + 1350;
  let remaining_words := 5000 - 1800;
  let section_length := 3200 / 4;
  exact sorry

end essay_body_section_length_l18_18363


namespace division_of_decimals_l18_18016

theorem division_of_decimals : 0.25 / 0.005 = 50 := 
by
  sorry

end division_of_decimals_l18_18016


namespace total_number_of_girls_l18_18134

-- Define the given initial number of girls and the number of girls joining the school
def initial_girls : Nat := 732
def girls_joined : Nat := 682
def total_girls : Nat := 1414

-- Formalize the problem
theorem total_number_of_girls :
  initial_girls + girls_joined = total_girls :=
by
  -- placeholder for proof
  sorry

end total_number_of_girls_l18_18134


namespace ball_count_in_box_eq_57_l18_18341

theorem ball_count_in_box_eq_57 (N : ℕ) (h : N - 44 = 70 - N) : N = 57 :=
sorry

end ball_count_in_box_eq_57_l18_18341


namespace jean_pages_written_l18_18713

theorem jean_pages_written:
  (∀ d : ℕ, 150 * d = 900 → d * 2 = 12) :=
by
  sorry

end jean_pages_written_l18_18713


namespace sequence_sum_equality_l18_18091

theorem sequence_sum_equality {a_n : ℕ → ℕ} (S_n : ℕ → ℕ) (n : ℕ) (h : n > 0) 
  (h1 : ∀ n, 3 * a_n n = 2 * S_n n + n) : 
  S_n n = (3^((n:ℕ)+1) - 2 * n) / 4 := 
sorry

end sequence_sum_equality_l18_18091


namespace elastic_band_radius_increase_l18_18174

theorem elastic_band_radius_increase 
  (C1 C2 : ℝ) 
  (hC1 : C1 = 40) 
  (hC2 : C2 = 80) 
  (hC1_def : C1 = 2 * π * r1) 
  (hC2_def : C2 = 2 * π * r2) :
  r2 - r1 = 20 / π :=
by
  sorry

end elastic_band_radius_increase_l18_18174


namespace compute_fraction_l18_18365

theorem compute_fraction :
  ((1/3)^4 * (1/5) = (1/405)) :=
by
  sorry

end compute_fraction_l18_18365


namespace separate_curves_l18_18253

variable {A : Type} [CommRing A]

def crossing_characteristic (ε : A → ℤ) (A1 A2 A3 A4 : A) : Prop :=
  ε A1 + ε A2 + ε A3 + ε A4 = 0

theorem separate_curves {A : Type} [CommRing A]
  {ε : A → ℤ} {A1 A2 A3 A4 : A} 
  (h : ε A1 + ε A2 + ε A3 + ε A4 = 0)
  (h1 : ε A1 = 1 ∨ ε A1 = -1)
  (h2 : ε A2 = 1 ∨ ε A2 = -1)
  (h3 : ε A3 = 1 ∨ ε A3 = -1)
  (h4 : ε A4 = 1 ∨ ε A4 = -1) :
  (∃ B1 B2 : A, B1 ≠ B2 ∧  ∀ (A : A), ((ε A = 1) → (A = B1)) ∨ ((ε A = -1) → (A = B2))) :=
  sorry

end separate_curves_l18_18253


namespace gcd_of_set_B_l18_18411

-- Let B be the set of all numbers which can be represented as the sum of five consecutive positive integers
def B : Set ℕ := {n | ∃ y : ℕ, n = (y - 2) + (y - 1) + y + (y + 1) + (y + 2)}

-- State the problem
theorem gcd_of_set_B : ∀ k ∈ B, ∃ d : ℕ, is_gcd 5 k d → d = 5 := by
sorry

end gcd_of_set_B_l18_18411


namespace rate_in_still_water_l18_18942

-- Definitions of given conditions
def downstream_speed : ℝ := 26
def upstream_speed : ℝ := 12

-- The statement we need to prove
theorem rate_in_still_water : (downstream_speed + upstream_speed) / 2 = 19 := by
  sorry

end rate_in_still_water_l18_18942


namespace number_of_triangles_in_polygon_l18_18684

theorem number_of_triangles_in_polygon {n : ℕ} (h : n > 0) :
  let vertices := (2 * n + 1)
  ∃ triangles_containing_center : ℕ, triangles_containing_center = (n * (n + 1) * (2 * n + 1)) / 6 :=
sorry

end number_of_triangles_in_polygon_l18_18684


namespace lowest_sale_price_is_30_percent_l18_18951

-- Definitions and conditions
def list_price : ℝ := 80
def max_initial_discount : ℝ := 0.50
def additional_sale_discount : ℝ := 0.20

-- Calculations
def initial_discount_amount : ℝ := list_price * max_initial_discount
def initial_discounted_price : ℝ := list_price - initial_discount_amount
def additional_discount_amount : ℝ := list_price * additional_sale_discount
def lowest_sale_price : ℝ := initial_discounted_price - additional_discount_amount

-- Proof statement (with correct answer)
theorem lowest_sale_price_is_30_percent :
  lowest_sale_price = 0.30 * list_price := 
by
  sorry

end lowest_sale_price_is_30_percent_l18_18951


namespace complement_union_l18_18825

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {3, 4, 5}

theorem complement_union :
  ((U \ A) ∪ B) = {1, 3, 4, 5, 6} :=
by
  sorry

end complement_union_l18_18825


namespace final_cost_is_30_l18_18758

-- Define conditions as constants
def cost_of_repair : ℝ := 7
def sales_tax : ℝ := 0.50
def number_of_tires : ℕ := 4

-- Define the cost for one tire repair
def cost_one_tire : ℝ := cost_of_repair + sales_tax

-- Define the cost for all tires
def total_cost : ℝ := cost_one_tire * number_of_tires

-- Theorem stating that the total cost is $30
theorem final_cost_is_30 : total_cost = 30 :=
by
  sorry

end final_cost_is_30_l18_18758


namespace binary_predecessor_l18_18536

def M : ℕ := 84
def N : ℕ := 83
def M_bin : ℕ := 0b1010100
def N_bin : ℕ := 0b1010011

theorem binary_predecessor (H : M = M_bin ∧ N = M - 1) : N = N_bin := by
  sorry

end binary_predecessor_l18_18536


namespace mark_first_vaccine_wait_time_l18_18873

-- Define the variables and conditions
variable (x : ℕ)
variable (total_wait_time : ℕ)
variable (second_appointment_wait : ℕ)
variable (effectiveness_wait : ℕ)

-- Given conditions
axiom h1 : second_appointment_wait = 20
axiom h2 : effectiveness_wait = 14
axiom h3 : total_wait_time = 38

-- The statement to be proven
theorem mark_first_vaccine_wait_time
  (h4 : x + second_appointment_wait + effectiveness_wait = total_wait_time) :
  x = 4 := by
  sorry

end mark_first_vaccine_wait_time_l18_18873


namespace max_value_a_plus_b_l18_18559

theorem max_value_a_plus_b
  (a b : ℝ)
  (h1 : 4 * a + 3 * b ≤ 10)
  (h2 : 3 * a + 5 * b ≤ 11) :
  a + b ≤ 156 / 55 :=
sorry

end max_value_a_plus_b_l18_18559


namespace probability_no_3x3_red_square_l18_18220

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l18_18220


namespace pete_backward_speed_l18_18424

variable (p b t s : ℝ)  -- speeds of Pete, backward walk, Tracy, and Susan respectively

-- Given conditions
axiom h1 : p / t = 1 / 4      -- Pete walks on his hands at a quarter speed of Tracy's cartwheeling
axiom h2 : t = 2 * s          -- Tracy cartwheels twice as fast as Susan walks
axiom h3 : b = 3 * s          -- Pete walks backwards three times faster than Susan
axiom h4 : p = 2              -- Pete walks on his hands at 2 miles per hour

-- Prove Pete's backward walking speed is 12 miles per hour
theorem pete_backward_speed : b = 12 :=
by
  sorry

end pete_backward_speed_l18_18424


namespace polynomial_division_quotient_correct_l18_18071

open Polynomial

noncomputable def proof_problem : Prop :=
  let dividend := 8 * (X ^ 4) - 4 * (X ^ 3) + 3 * (X ^ 2) - 5 * X - 10
  let divisor := (X ^ 2) + 3 * X + 2
  let quotient := 8 * (X ^ 2) - 28 * X + 89
  dividend /ₚ divisor = quotient

theorem polynomial_division_quotient_correct : proof_problem :=
  sorry

end polynomial_division_quotient_correct_l18_18071


namespace comics_in_box_l18_18195

def comics_per_comic := 25
def total_pages := 150
def existing_comics := 5

def torn_comics := total_pages / comics_per_comic
def total_comics := torn_comics + existing_comics

theorem comics_in_box : total_comics = 11 := by
  sorry

end comics_in_box_l18_18195


namespace tax_calculation_l18_18615

variable (winnings : ℝ) (processing_fee : ℝ) (take_home : ℝ)
variable (tax_percentage : ℝ)

def given_conditions : Prop :=
  winnings = 50 ∧ processing_fee = 5 ∧ take_home = 35

def to_prove : Prop :=
  tax_percentage = 20

theorem tax_calculation (h : given_conditions winnings processing_fee take_home) : to_prove tax_percentage :=
by
  sorry

end tax_calculation_l18_18615


namespace remove_toothpicks_l18_18658

-- Definitions based on problem conditions
def toothpicks := 40
def triangles := 40
def initial_triangulation := True
def additional_condition := True

-- Statement to be proved
theorem remove_toothpicks :
  initial_triangulation ∧ additional_condition ∧ (triangles > 40) → ∃ (t: ℕ), t = 15 :=
by
  sorry

end remove_toothpicks_l18_18658


namespace binom_9_5_l18_18993

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l18_18993


namespace correct_equations_l18_18007

-- Defining the problem statement
theorem correct_equations (m n : ℕ) :
  (∀ (m n : ℕ), 40 * m + 10 = 43 * m + 1 ∧ 
   (n - 10) / 40 = (n - 1) / 43) :=
by
  sorry

end correct_equations_l18_18007


namespace frank_initial_money_l18_18519

theorem frank_initial_money (X : ℝ) (h1 : X * (4 / 5) * (3 / 4) * (6 / 7) * (2 / 3) = 600) : X = 2333.33 :=
sorry

end frank_initial_money_l18_18519


namespace intersection_S_T_l18_18813

def S : Set ℝ := { x | 2 * x + 1 > 0 }
def T : Set ℝ := { x | 3 * x - 5 < 0 }

theorem intersection_S_T :
  S ∩ T = { x | -1/2 < x ∧ x < 5/3 } := by
  sorry

end intersection_S_T_l18_18813


namespace arithmetic_sequence_sum_of_bn_l18_18082

variable (a : ℕ → ℕ) (b : ℕ → ℚ) (S : ℕ → ℚ)

theorem arithmetic_sequence (h1 : a 1 + a 4 = 10) (h2 : a 3 = 6) :
  (∀ n, a n = 2 * n) :=
by sorry

theorem sum_of_bn (h1 : a 1 + a 4 = 10) (h2 : a 3 = 6)
                  (h3 : ∀ n, a n = 2 * n)
                  (h4 : ∀ n, b n = 4 / (a n * a (n + 1))) :
  (∀ n, S n = n / (n + 1)) :=
by sorry

end arithmetic_sequence_sum_of_bn_l18_18082


namespace evaluate_expression_at_2_l18_18866

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := 2 * x - 3

theorem evaluate_expression_at_2 : f (g 2) + g (f 2) = 331 / 20 :=
by
  sorry

end evaluate_expression_at_2_l18_18866


namespace find_a_and_tangent_line_l18_18531

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2 * x^2 + a * x - 1

theorem find_a_and_tangent_line (a x y : ℝ) (h1 : f a 1 = x^3 - 2 * x^2 + 2 * x - 1)
  (h2 : (deriv (f a) 1 = 1)) :
  a = 2 ∧ (exists y, (y + 6 = 9 * (x + 1)))) :=
by
sry

end find_a_and_tangent_line_l18_18531


namespace stapler_problem_l18_18450

noncomputable def staplesLeft (initial_staples : ℕ) (dozens : ℕ) (staples_per_report : ℝ) : ℝ :=
  initial_staples - (dozens * 12) * staples_per_report

theorem stapler_problem : staplesLeft 200 7 0.75 = 137 := 
by
  sorry

end stapler_problem_l18_18450


namespace zach_saved_money_l18_18330

-- Definitions of known quantities
def cost_of_bike : ℝ := 100
def weekly_allowance : ℝ := 5
def mowing_earnings : ℝ := 10
def babysitting_rate : ℝ := 7
def babysitting_hours : ℝ := 2
def additional_earnings_needed : ℝ := 6

-- Calculate total earnings for this week
def total_earnings_this_week : ℝ := weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)

-- Prove that Zach has already saved $65
theorem zach_saved_money : (cost_of_bike - total_earnings_this_week - additional_earnings_needed) = 65 :=
by
  -- Sorry used as placeholder to skip the proof
  sorry

end zach_saved_money_l18_18330


namespace geometric_sequence_fifth_term_l18_18708

theorem geometric_sequence_fifth_term (α : ℕ → ℝ) (h : α 4 * α 5 * α 6 = 27) : α 5 = 3 :=
sorry

end geometric_sequence_fifth_term_l18_18708


namespace find_a_tangent_line_at_minus_one_l18_18530

-- Define the function f with variable a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f with variable a
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Given conditions
def condition_1 : Prop := f' 1 = 1
def condition_2 : Prop := f' 2 (1 : ℝ) = 1

-- Prove that a = 2 given f'(1) = 1
theorem find_a : f' 2 (1 : ℝ) = 1 → 2 = 2 := by
  sorry

-- Given a = 2, find the tangent line equation at x = -1
def tangent_line_equation (x y : ℝ) : Prop := 9*x - y + 3 = 0

-- Define the coordinates of the point on the curve at x = -1
def point_on_curve : Prop := f 2 (-1) = -6

-- Prove the tangent line equation at x = -1 given a = 2
theorem tangent_line_at_minus_one (h : true) : tangent_line_equation 9 (f' 2 (-1)) := by
  sorry

end find_a_tangent_line_at_minus_one_l18_18530


namespace binomial_coefficient_9_5_l18_18997

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l18_18997


namespace number_of_six_digit_with_sum_51_l18_18833

open Finset

/-- A digit is a number between 0 and 9 inclusive.-/
def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

/-- Friendly notation for digit sums -/
def digit_sum (n : Fin 6 → ℕ) : ℕ :=
  (Finset.univ.sum n)

def is_six_digit_with_sum_51 (n : Fin 6 → ℕ) : Prop :=
  (∀ i, is_digit (n i)) ∧ (digit_sum n = 51)

/-- There are exactly 56 six-digit numbers such that the sum of their digits is 51. -/
theorem number_of_six_digit_with_sum_51 : 
  card {n : Fin 6 → ℕ // is_six_digit_with_sum_51 n} = 56 :=
by
  sorry

end number_of_six_digit_with_sum_51_l18_18833


namespace length_AB_of_parabola_l18_18043

theorem length_AB_of_parabola (x1 x2 : ℝ)
  (h : x1 + x2 = 6) :
  abs (x1 + x2 + 2) = 8 := by
  sorry

end length_AB_of_parabola_l18_18043


namespace fundraiser_price_per_item_l18_18076

theorem fundraiser_price_per_item
  (students_brownies : ℕ)
  (brownies_per_student : ℕ)
  (students_cookies : ℕ)
  (cookies_per_student : ℕ)
  (students_donuts : ℕ)
  (donuts_per_student : ℕ)
  (total_amount_raised : ℕ)
  (total_brownies : ℕ := students_brownies * brownies_per_student)
  (total_cookies : ℕ := students_cookies * cookies_per_student)
  (total_donuts : ℕ := students_donuts * donuts_per_student)
  (total_items : ℕ := total_brownies + total_cookies + total_donuts)
  (price_per_item : ℕ := total_amount_raised / total_items) :
  students_brownies = 30 →
  brownies_per_student = 12 →
  students_cookies = 20 →
  cookies_per_student = 24 →
  students_donuts = 15 →
  donuts_per_student = 12 →
  total_amount_raised = 2040 →
  price_per_item = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end fundraiser_price_per_item_l18_18076


namespace sufficient_but_not_necessary_l18_18282

variable (x : ℝ)

theorem sufficient_but_not_necessary : (x = 1) → (x^3 = x) ∧ (∀ y, y^3 = y → y = 1 → x ≠ y) :=
by
  sorry

end sufficient_but_not_necessary_l18_18282


namespace product_of_roots_l18_18396

theorem product_of_roots :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) ∧ (x₁ ^ 2 + 2 * x₁ - 4 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 4 = 0) := by
  sorry

end product_of_roots_l18_18396


namespace ants_meet_probability_l18_18753

-- Definitions based on the conditions
variable (V : Type)
variable [Fintype V]
variable [DecidableEq V]

constant tetrahedron : SimpleGraph V
constant ants_initial_vertices : Fin 3 → V
constant move_probability : (v₁ v₂ : V) → ℝ

-- Conditions for the problem
axiom ants_start_on_different_vertices (x y : Fin 3) : x ≠ y → ants_initial_vertices x ≠ ants_initial_vertices y
axiom movement_probability (v : V) (u ∈ tetrahedron.adj v) : move_probability v u = 1 / 3

-- Main theorem statement
theorem ants_meet_probability :
  (Probability (at_stop_same_vertex ants_initial_vertices move_probability) = 1 / 16) :=
sorry

end ants_meet_probability_l18_18753


namespace fraction_is_one_fourth_l18_18125

theorem fraction_is_one_fourth
  (f : ℚ)
  (m : ℕ)
  (h1 : (1 / 5) ^ m * f^2 = 1 / (10 ^ 4))
  (h2 : m = 4) : f = 1 / 4 := by
  sorry

end fraction_is_one_fourth_l18_18125


namespace sum_of_three_numbers_l18_18309

theorem sum_of_three_numbers :
  ∃ (S1 S2 S3 : ℕ), 
    S2 = 72 ∧
    S1 = 2 * S2 ∧
    S3 = S1 / 3 ∧
    S1 + S2 + S3 = 264 := 
by
  sorry

end sum_of_three_numbers_l18_18309


namespace total_amount_divided_l18_18036

theorem total_amount_divided (B_amount A_amount C_amount: ℝ) (h1 : A_amount = (1/3) * B_amount)
    (h2 : B_amount = 270) (h3 : B_amount = (1/4) * C_amount) :
    A_amount + B_amount + C_amount = 1440 :=
by
  sorry

end total_amount_divided_l18_18036


namespace gcd_of_set_B_is_five_l18_18413

def is_in_set_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

theorem gcd_of_set_B_is_five : ∃ d, (∀ n, is_in_set_B n → d ∣ n) ∧ d = 5 :=
by 
  sorry

end gcd_of_set_B_is_five_l18_18413


namespace elizabeth_husband_weight_l18_18452

-- Defining the variables for weights of the three wives
variable (s : ℝ) -- Weight of Simona
def elizabeta_weight : ℝ := s + 5
def georgetta_weight : ℝ := s + 10

-- Condition: The total weight of all wives
def total_wives_weight : ℝ := s + elizabeta_weight s + georgetta_weight s

-- Given: The total weight of all wives is 171 kg
def total_wives_weight_cond : Prop := total_wives_weight s = 171

-- Given:
-- Leon weighs the same as his wife.
-- Victor weighs one and a half times more than his wife.
-- Maurice weighs twice as much as his wife.

-- Given: Elizabeth's weight relationship
def elizabeth_weight_cond : Prop := (s + 5 * 1.5) = 85.5

-- Main proof problem:
theorem elizabeth_husband_weight (s : ℝ) (h1: total_wives_weight_cond s) : elizabeth_weight_cond s :=
by
  sorry

end elizabeth_husband_weight_l18_18452


namespace degree_sum_twice_edges_even_number_of_odd_degree_vertices_l18_18155

variables {V : Type*} (G : SimpleGraph V)

open SimpleGraph

-- Part (a)
theorem degree_sum_twice_edges (G : SimpleGraph V) : 
  (∑ v in G.vertices, G.degree v) = 2 * G.edge_count := 
sorry

-- Part (b)
theorem even_number_of_odd_degree_vertices (G : SimpleGraph V) : 
  even (card {v ∈ G.vertices | odd (G.degree v)}) :=
sorry

end degree_sum_twice_edges_even_number_of_odd_degree_vertices_l18_18155


namespace minimal_time_for_horses_l18_18133

/-- Define the individual periods of the horses' runs -/
def periods : List ℕ := [2, 3, 4, 5, 6, 7, 9, 10]

/-- Define a function to calculate the LCM of a list of numbers -/
def lcm_list (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm 1

/-- Conjecture: proving that 60 is the minimal time until at least 6 out of 8 horses meet at the starting point -/
theorem minimal_time_for_horses : lcm_list [2, 3, 4, 5, 6, 10] = 60 :=
by
  sorry

end minimal_time_for_horses_l18_18133


namespace break_even_shirts_needed_l18_18158

-- Define the conditions
def initialInvestment : ℕ := 1500
def costPerShirt : ℕ := 3
def sellingPricePerShirt : ℕ := 20

-- Define the profit per T-shirt and the number of T-shirts to break even
def profitPerShirt (sellingPrice costPrice : ℕ) : ℕ := sellingPrice - costPrice

def shirtsToBreakEven (investment profit : ℕ) : ℕ :=
  (investment + profit - 1) / profit -- ceil division

-- The theorem to prove
theorem break_even_shirts_needed :
  shirtsToBreakEven initialInvestment (profitPerShirt sellingPricePerShirt costPerShirt) = 89 :=
by
  -- Calculation
  sorry

end break_even_shirts_needed_l18_18158


namespace solve_equation_l18_18584

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l18_18584


namespace hamburger_per_meatball_l18_18956

theorem hamburger_per_meatball (family_members : ℕ) (total_hamburger : ℕ) (antonio_meatballs : ℕ) 
    (hmembers : family_members = 8)
    (hhamburger : total_hamburger = 4)
    (hantonio : antonio_meatballs = 4) : 
    (total_hamburger : ℝ) / (family_members * antonio_meatballs) = 0.125 := 
by
  sorry

end hamburger_per_meatball_l18_18956


namespace vasilyev_max_car_loan_l18_18160

def vasilyev_income := 71000 + 11000 + 2600
def vasilyev_expenses := 8400 + 18000 + 3200 + 2200 + 18000
def remaining_income := vasilyev_income - vasilyev_expenses
def financial_security_cushion := 0.1 * remaining_income
def max_car_loan := remaining_income - financial_security_cushion

theorem vasilyev_max_car_loan : max_car_loan = 31320 := by
  -- Definitions to set up the problem conditions
  have h_income : vasilyev_income = 84600 := rfl
  have h_expenses : vasilyev_expenses = 49800 := rfl
  have h_remaining_income : remaining_income = 34800 := by
    rw [←h_income, ←h_expenses]
    exact rfl
  have h_security_cushion : financial_security_cushion = 3480 := by
    rw [←h_remaining_income]
    exact (mul_comm 0.1 34800).symm
  have h_max_loan : max_car_loan = 31320 := by
    rw [←h_remaining_income, ←h_security_cushion]
    exact rfl
  -- Conclusion of the theorem proof
  exact h_max_loan

end vasilyev_max_car_loan_l18_18160


namespace combined_weight_of_candles_l18_18804

theorem combined_weight_of_candles 
  (beeswax_weight_per_candle : ℕ)
  (coconut_oil_weight_per_candle : ℕ)
  (total_candles : ℕ)
  (candles_made : ℕ) 
  (total_weight: ℕ) 
  : 
  beeswax_weight_per_candle = 8 → 
  coconut_oil_weight_per_candle = 1 → 
  total_candles = 10 → 
  candles_made = total_candles - 3 →
  total_weight = candles_made * (beeswax_weight_per_candle + coconut_oil_weight_per_candle) →
  total_weight = 63 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end combined_weight_of_candles_l18_18804


namespace min_value_f_when_a_is_zero_inequality_holds_for_f_l18_18386

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x

-- Problem (1): Prove the minimum value of f(x) when a = 0 is 2 - 2 * ln 2.
theorem min_value_f_when_a_is_zero : 
  (∃ x : ℝ, f x 0 = 2 - 2 * Real.log 2) :=
sorry

-- Problem (2): Prove that for a < (exp(1) / 2) - 1, f(x) > (exp(1) / 2) - 1 for all x in (0, +∞).
theorem inequality_holds_for_f :
  ∀ a : ℝ, a < (Real.exp 1) / 2 - 1 → 
  ∀ x : ℝ, 0 < x → f x a > (Real.exp 1) / 2 - 1 :=
sorry

end min_value_f_when_a_is_zero_inequality_holds_for_f_l18_18386


namespace count_digit_9_in_range_l18_18118

theorem count_digit_9_in_range :
  (∑ n in List.finRange 1000, (List.some (Nat.digits 10 (n+1)).count 9) 0) = 300 := by
  sorry

end count_digit_9_in_range_l18_18118


namespace find_x_minus_y_l18_18693

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  -- Proof omitted
  sorry

end find_x_minus_y_l18_18693


namespace monotonicity_of_f_range_of_a_l18_18666

noncomputable def f (a x : ℝ) : ℝ := -2 * x^3 + 6 * a * x^2 - 1
noncomputable def g (a x : ℝ) : ℝ := Real.exp x - 2 * a * x - 1

theorem monotonicity_of_f (a : ℝ) :
  (a < 0 → (∀ x, f a x ≥ f a (2 * a) → x ≤ 0 ∨ x ≤ 2 * a)) ∧
  (a = 0 → ∀ x y, x ≤ y → f a x ≥ f a y) ∧
  (a > 0 → (∀ x, f a x ≤ f a 0 → x ≤ 0) ∧
           (∀ x, 0 < x ∧ x < 2 * a → f a x ≥ f a 2 * a) ∧
           (∀ x, 2 * a < x → f a x ≤ f a (2 * a))) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, a ≥ 1 / 2 →
  ∃ x1 : ℝ, x1 > 0 ∧ ∃ x2 : ℝ, f a x1 ≥ g a x2 :=
sorry

end monotonicity_of_f_range_of_a_l18_18666


namespace coprime_lcm_inequality_l18_18826

theorem coprime_lcm_inequality
  (p q : ℕ)
  (hpq_coprime : Nat.gcd p q = 1)
  (hp_gt_1 : p > 1)
  (hq_gt_1 : q > 1)
  (hpq_diff_gt_1 : abs (p - q) > 1) :
  ∃ n : ℕ, Nat.lcm (p + n) (q + n) < Nat.lcm p q :=
by
  sorry

end coprime_lcm_inequality_l18_18826


namespace second_train_speed_is_correct_l18_18778

noncomputable def speed_of_second_train (length_first : ℝ) (speed_first : ℝ) (time_cross : ℝ) (length_second : ℝ) : ℝ :=
let total_distance := length_first + length_second
let relative_speed := total_distance / time_cross
let relative_speed_kmph := relative_speed * 3.6
relative_speed_kmph - speed_first

theorem second_train_speed_is_correct :
  speed_of_second_train 270 120 9 230.04 = 80.016 :=
by
  sorry

end second_train_speed_is_correct_l18_18778


namespace find_number_l18_18919

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end find_number_l18_18919


namespace lamps_purchased_min_type_B_lamps_l18_18037

variables (x y m : ℕ)

def total_lamps := x + y = 50
def total_cost := 40 * x + 65 * y = 2500
def profit_type_A := 60 - 40 = 20
def profit_type_B := 100 - 65 = 35
def profit_requirement := 20 * (50 - m) + 35 * m ≥ 1400

theorem lamps_purchased (h₁ : total_lamps x y) (h₂ : total_cost x y) : 
  x = 30 ∧ y = 20 :=
  sorry

theorem min_type_B_lamps (h₃ : profit_type_A) (h₄ : profit_type_B) (h₅ : profit_requirement m) : 
  m ≥ 27 :=
  sorry

end lamps_purchased_min_type_B_lamps_l18_18037


namespace binomial_pmf_value_l18_18417

open ProbabilityTheory

noncomputable def binomial_pmf 
  (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem binomial_pmf_value :
  binomial_pmf 6 (1 / 2) 3 = 5 / 16 :=
by
  sorry

end binomial_pmf_value_l18_18417


namespace Jason_spent_on_jacket_l18_18548

/-
Given:
- Amount_spent_on_shorts: ℝ := 14.28
- Total_spent_on_clothing: ℝ := 19.02

Prove:
- Amount_spent_on_jacket = 4.74
-/
def Amount_spent_on_shorts : ℝ := 14.28
def Total_spent_on_clothing : ℝ := 19.02

-- We need to prove:
def Amount_spent_on_jacket : ℝ := Total_spent_on_clothing - Amount_spent_on_shorts 

theorem Jason_spent_on_jacket : Amount_spent_on_jacket = 4.74 := by
  sorry

end Jason_spent_on_jacket_l18_18548


namespace even_function_l18_18524

-- Definition of f and F with the given conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Condition that x is in the interval (-a, a)
def in_interval (a x : ℝ) : Prop := x > -a ∧ x < a

-- Definition of F(x)
def F (x : ℝ) : ℝ := f x + f (-x)

-- The proposition that we want to prove
theorem even_function (h : in_interval a x) : F f x = F f (-x) :=
by
  unfold F
  sorry

end even_function_l18_18524


namespace gcd_36_48_72_l18_18764

theorem gcd_36_48_72 : Int.gcd (Int.gcd 36 48) 72 = 12 := by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 48 = 2^4 * 3 := by norm_num
  have h3 : 72 = 2^3 * 3^2 := by norm_num
  sorry

end gcd_36_48_72_l18_18764


namespace ratio_proof_l18_18055

-- Define Clara's initial number of stickers and the stickers given at each step
def initial_stickers : ℕ := 100
def stickers_given_to_boy : ℕ := 10
def stickers_left_after_friends : ℕ := 45

-- Define the number of stickers after giving to the boy and the number of stickers given to friends
def stickers_after_boy := initial_stickers - stickers_given_to_boy
def stickers_given_to_friends := stickers_after_boy - stickers_left_after_friends

-- Define the ratio to be proven
def ratio := stickers_given_to_friends : ℚ / stickers_after_boy

-- Statement to prove
theorem ratio_proof : ratio = (1 : ℚ) / 2 :=
by
  -- initial number of stickers
  -- initial number of stickers given to boy
  -- number of stickers left after sharing with friends
  -- calculate number of stickers given to friends
  -- define the ratio
  -- prove the ratio
  sorry

end ratio_proof_l18_18055


namespace quadratic_inequality_solution_l18_18841

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l18_18841


namespace solve_a_plus_b_l18_18835

theorem solve_a_plus_b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 7 * a + 2 * b = 54) : a + b = -103 / 31 :=
by
  sorry

end solve_a_plus_b_l18_18835


namespace perfect_square_expression_l18_18899

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end perfect_square_expression_l18_18899


namespace option_c_l18_18384

theorem option_c (a b : ℝ) (h : a > |b|) : a^2 > b^2 := sorry

end option_c_l18_18384


namespace bills_average_speed_l18_18468

theorem bills_average_speed :
  ∃ v t : ℝ, 
      (v + 5) * (t + 2) + v * t = 680 ∧ 
      (t + 2) + t = 18 ∧ 
      v = 35 :=
by
  sorry

end bills_average_speed_l18_18468


namespace find_number_l18_18917

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end find_number_l18_18917


namespace binom_9_5_eq_126_l18_18979

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l18_18979


namespace ratio_x_y_l18_18211

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 7) : x / y = 29 / 64 :=
by
  sorry

end ratio_x_y_l18_18211


namespace distance_traveled_by_second_hand_l18_18749

def second_hand_length : ℝ := 8
def time_period_minutes : ℝ := 45
def rotations_per_minute : ℝ := 1

theorem distance_traveled_by_second_hand :
  let circumference := 2 * Real.pi * second_hand_length
  let rotations := time_period_minutes * rotations_per_minute
  let total_distance := rotations * circumference
  total_distance = 720 * Real.pi := by
  sorry

end distance_traveled_by_second_hand_l18_18749


namespace marks_fathers_gift_l18_18568

noncomputable def total_spent (books : ℕ) (cost_per_book : ℕ) : ℕ :=
  books * cost_per_book

noncomputable def total_money_given (spent : ℕ) (left_over : ℕ) : ℕ :=
  spent + left_over

theorem marks_fathers_gift :
  total_money_given (total_spent 10 5) 35 = 85 := by
  sorry

end marks_fathers_gift_l18_18568


namespace kennedy_lost_pawns_l18_18862

-- Definitions based on conditions
def initial_pawns_per_player := 8
def total_pawns := 2 * initial_pawns_per_player -- Total pawns in the game initially
def pawns_lost_by_Riley := 1 -- Riley lost 1 pawn
def pawns_remaining := 11 -- 11 pawns left in the game

-- Translations of conditions to Lean
theorem kennedy_lost_pawns : 
  initial_pawns_per_player - (pawns_remaining - (initial_pawns_per_player - pawns_lost_by_Riley)) = 4 := 
by 
  sorry

end kennedy_lost_pawns_l18_18862


namespace who_is_who_l18_18328

-- Define the types for inhabitants
inductive Inhabitant
| A : Inhabitant
| B : Inhabitant

-- Define the property of being a liar
def is_liar (x : Inhabitant) : Prop := 
  match x with
  | Inhabitant.A  => false -- Initial assumption, to be refined
  | Inhabitant.B  => false -- Initial assumption, to be refined

-- Define the statement made by A
def statement_by_A : Prop :=
  (is_liar Inhabitant.A ∧ ¬ is_liar Inhabitant.B)

-- The main theorem to prove
theorem who_is_who (h : ¬statement_by_A) :
  is_liar Inhabitant.A ∧ is_liar Inhabitant.B :=
by
  -- Proof goes here
  sorry

end who_is_who_l18_18328


namespace solve_fractional_eq_l18_18592

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l18_18592


namespace problem1_problem2_l18_18298

theorem problem1 (x : ℝ) : (x + 3) * (x - 1) ≤ 0 ↔ -3 ≤ x ∧ x ≤ 1 :=
sorry

theorem problem2 (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 :=
sorry

end problem1_problem2_l18_18298


namespace digit_9_occurrences_1_to_1000_l18_18106

theorem digit_9_occurrences_1_to_1000 : 
  (finset.range 1000).sum (λ n, (n.digits 10).count 9) = 300 := 
by
  sorry

end digit_9_occurrences_1_to_1000_l18_18106


namespace largest_prime_factor_of_1001_l18_18325

theorem largest_prime_factor_of_1001 : 
  ∃ p : ℕ, prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1001 → q ≤ p) :=
sorry

end largest_prime_factor_of_1001_l18_18325


namespace plane_coloring_l18_18608

-- Define a type for colors to represent red and blue
inductive Color
| red
| blue

-- The main statement
theorem plane_coloring (x : ℝ) (h_pos : 0 < x) (coloring : ℝ × ℝ → Color) :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ coloring p1 = coloring p2 ∧ dist p1 p2 = x :=
sorry

end plane_coloring_l18_18608


namespace unique_n_in_range_satisfying_remainders_l18_18097

theorem unique_n_in_range_satisfying_remainders : 
  ∃! (n : ℤ) (k r : ℤ), 150 < n ∧ n < 250 ∧ 0 <= r ∧ r <= 6 ∧ n = 7 * k + r ∧ n = 9 * r + r :=
sorry

end unique_n_in_range_satisfying_remainders_l18_18097


namespace rectangle_area_l18_18634

theorem rectangle_area (y : ℝ) (w : ℝ) : 
  (3 * w) ^ 2 + w ^ 2 = y ^ 2 → 
  3 * w * w = (3 / 10) * y ^ 2 :=
by
  intro h
  sorry

end rectangle_area_l18_18634


namespace calculate_expression_l18_18964

theorem calculate_expression :
  (-0.125) ^ 2009 * (8 : ℝ) ^ 2009 = -1 :=
sorry

end calculate_expression_l18_18964


namespace binom_9_5_eq_126_l18_18977

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l18_18977


namespace number_of_primes_under_150_with_ones_digit_3_l18_18683

noncomputable def primes_under_150_with_ones_digit_3 : Finset ℕ :=
  Finset.filter (λ n, Nat.Prime n) (Finset.filter (λ n, n % 10 = 3) (Finset.range 150))

theorem number_of_primes_under_150_with_ones_digit_3 :
  Finset.card primes_under_150_with_ones_digit_3 = 9 :=
by
  sorry

end number_of_primes_under_150_with_ones_digit_3_l18_18683


namespace parabola_shifted_l18_18888

-- Define the original parabola
def originalParabola (x : ℝ) : ℝ := (x + 2)^2 + 3

-- Shift the parabola by 3 units to the right
def shiftedRight (x : ℝ) : ℝ := originalParabola (x - 3)

-- Then shift the parabola 2 units down
def shiftedRightThenDown (x : ℝ) : ℝ := shiftedRight x - 2

-- The problem asks to prove that the final expression is equal to (x - 1)^2 + 1
theorem parabola_shifted (x : ℝ) : shiftedRightThenDown x = (x - 1)^2 + 1 :=
by
  sorry

end parabola_shifted_l18_18888


namespace total_students_in_class_l18_18852

theorem total_students_in_class (F G B N T : ℕ)
  (hF : F = 41)
  (hG : G = 22)
  (hB : B = 9)
  (hN : N = 15)
  (hT : T = (F + G - B) + N) :
  T = 69 :=
by
  -- This is a theorem statement, proof is intentionally omitted.
  sorry

end total_students_in_class_l18_18852


namespace Isabel_subtasks_remaining_l18_18138

-- Definition of the known quantities
def Total_problems : ℕ := 72
def Completed_problems : ℕ := 32
def Subtasks_per_problem : ℕ := 5

-- Definition of the calculations
def Total_subtasks : ℕ := Total_problems * Subtasks_per_problem
def Completed_subtasks : ℕ := Completed_problems * Subtasks_per_problem
def Remaining_subtasks : ℕ := Total_subtasks - Completed_subtasks

-- The theorem we need to prove
theorem Isabel_subtasks_remaining : Remaining_subtasks = 200 := by
  -- Proof would go here, but we'll use sorry to indicate it's omitted
  sorry

end Isabel_subtasks_remaining_l18_18138


namespace menkara_index_card_area_l18_18150

theorem menkara_index_card_area :
  ∀ (length width: ℕ), 
  length = 5 → width = 7 → (length - 2) * width = 21 → 
  (length * (width - 2) = 25) :=
by
  intros length width h_length h_width h_area
  sorry

end menkara_index_card_area_l18_18150


namespace trapezium_area_proof_l18_18542

-- Define the coordinates and values
variables (E B C A D : Point)
variables (R S : ℝ)

-- Geometry conditions
def equilateral_triangle (E B C : Point) :=
(E.dist(B) = E.dist(C)) ∧ (B.dist(C) = E.dist(B))

def on_line (P1 P2 P : Point) :=
∃ k : ℝ, P = P1 + k • (P2 - P1)

def parallel_lines (l1 l2 : Line) :=
l1.direction ∥ l2.direction

def perpendicular_lines (l1 l2 : Line) :=
l1.direction ⟂ l2.direction

def trapezium_area (A B C D : Point) : ℝ :=
1/2 * (A.dist(C) + B.dist(D)) * (perp_dist_line_point (line(A B)) C)

-- Area of trapezium
def area_ABCD := 9408

-- Problem statement
theorem trapezium_area_proof (H1 : equilateral_triangle E B C) (H2 : on_line E B A)
    (H3 : on_line E C D) (H4 : parallel_lines (line(A D)) (line(B C)))
    (H5 : A.dist B = R) (H6 : C.dist D = R) (H7 : perpendicular_lines (line(A C)) (line(B D))) :
    trapezium_area A B C D = area_ABCD :=
sorry

end trapezium_area_proof_l18_18542


namespace meals_for_children_l18_18345

theorem meals_for_children (C : ℕ)
  (H1 : 70 * C = 70 * 45)
  (H2 : 70 * 45 = 2 * 45 * 35) :
  C = 90 :=
by
  sorry

end meals_for_children_l18_18345


namespace remainder_is_nine_l18_18460

-- Define the dividend and divisor
def n : ℕ := 4039
def d : ℕ := 31

-- Prove that n mod d equals 9
theorem remainder_is_nine : n % d = 9 := by
  sorry

end remainder_is_nine_l18_18460


namespace cookies_left_after_week_l18_18168

theorem cookies_left_after_week (cookies_in_jar : ℕ) (total_taken_out_in_4_days : ℕ) (same_amount_each_day : Prop)
  (h1 : cookies_in_jar = 70) (h2 : total_taken_out_in_4_days = 24) :
  ∃ (cookies_left : ℕ), cookies_left = 28 :=
by
  sorry

end cookies_left_after_week_l18_18168


namespace correct_fraction_simplification_l18_18687

theorem correct_fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (∀ (c d : ℝ), (c ≠ d) → (a+2 = c → b+2 = d → (a+2)/d ≠ a/b))
  ∧ (∀ (e f : ℝ), (e ≠ f) → (a-2 = e → b-2 = f → (a-2)/f ≠ a/b))
  ∧ (∀ (g h : ℝ), (g ≠ h) → (a^2 = g → b^2 = h → a^2/h ≠ a/b))
  ∧ (a / b = ( (1/2)*a / (1/2)*b )) := 
sorry

end correct_fraction_simplification_l18_18687


namespace total_plants_in_garden_l18_18961

-- Definitions based on conditions
def basil_plants : ℕ := 5
def oregano_plants : ℕ := 2 + 2 * basil_plants

-- Theorem statement
theorem total_plants_in_garden : basil_plants + oregano_plants = 17 := by
  -- Skipping the proof with sorry
  sorry

end total_plants_in_garden_l18_18961


namespace find_b_l18_18579

theorem find_b (x : ℝ) (b : ℝ) :
  (3 * x + 9 = 0) → (2 * b * x - 15 = -5) → b = -5 / 3 :=
by
  intros h1 h2
  sorry

end find_b_l18_18579


namespace simple_interest_rate_l18_18622

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 800) (hSI : SI = 128) (hT : T = 4) : 
  (SI = P * (R : ℝ) * T / 100) → R = 4 := 
by {
  -- Proof goes here.
  sorry
}

end simple_interest_rate_l18_18622


namespace quadratic_has_two_roots_l18_18822

variables {R : Type*} [LinearOrderedField R]

theorem quadratic_has_two_roots (a1 a2 a3 b1 b2 b3 : R) 
  (h1 : a1 * a2 * a3 = b1 * b2 * b3) (h2 : a1 * a2 * a3 > 1) : 
  (4 * a1^2 - 4 * b1 > 0) ∨ (4 * a2^2 - 4 * b2 > 0) ∨ (4 * a3^2 - 4 * b3 > 0) :=
sorry

end quadratic_has_two_roots_l18_18822


namespace student_correct_answers_l18_18785

-- Definitions based on the conditions
def total_questions : ℕ := 100
def score (correct incorrect : ℕ) : ℕ := correct - 2 * incorrect
def studentScore : ℕ := 73

-- Main theorem to prove
theorem student_correct_answers (C I : ℕ) (h1 : C + I = total_questions) (h2 : score C I = studentScore) : C = 91 :=
by
  sorry

end student_correct_answers_l18_18785


namespace fourth_training_session_end_time_l18_18958

theorem fourth_training_session_end_time :
  (let start_time := ⟨8, 0⟩ : Time -- 8:00 AM
   let session_duration := 40 -- minutes
   let break_duration := 15 -- minutes
   let num_sessions := 4
   let total_training_duration := num_sessions * session_duration
   let num_breaks := num_sessions - 1
   let total_break_duration := num_breaks * break_duration
   let total_duration := total_training_duration + total_break_duration
   let hours := total_duration / 60
   let minutes := total_duration % 60
   let end_time := Time.mk (start_time.hour + hours) (start_time.minute + minutes)
   end_time = ⟨11, 25⟩
  ) := sorry

end fourth_training_session_end_time_l18_18958


namespace triangle_area_example_l18_18763

def point : Type := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_example : 
  triangle_area (0, 0) (0, 6) (8, 10) = 24 :=
by
  sorry

end triangle_area_example_l18_18763


namespace inequality_holds_l18_18428

theorem inequality_holds (x : ℝ) (hx : x ≥ 0) : 3 * x^3 - 6 * x^2 + 4 ≥ 0 := 
  sorry

end inequality_holds_l18_18428


namespace ratio_of_autobiographies_to_fiction_l18_18311

theorem ratio_of_autobiographies_to_fiction (total_books fiction_books non_fiction_books picture_books autobiographies: ℕ) 
  (h1 : total_books = 35) 
  (h2 : fiction_books = 5) 
  (h3 : non_fiction_books = fiction_books + 4) 
  (h4 : picture_books = 11) 
  (h5 : autobiographies = total_books - (fiction_books + non_fiction_books + picture_books)) :
  autobiographies / fiction_books = 2 :=
by sorry

end ratio_of_autobiographies_to_fiction_l18_18311


namespace quadratic_root_four_times_another_l18_18648

theorem quadratic_root_four_times_another (a : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + a * x + 2 * a = 0 ∧ x2 = 4 * x1) → a = 25 / 2 :=
by
  sorry

end quadratic_root_four_times_another_l18_18648


namespace remainder_when_divided_l18_18240

open Polynomial

noncomputable def poly : Polynomial ℚ := X^6 + X^5 + 2*X^3 - X^2 + 3
noncomputable def divisor : Polynomial ℚ := (X + 2) * (X - 1)
noncomputable def remainder : Polynomial ℚ := -X + 5

theorem remainder_when_divided :
  ∃ q : Polynomial ℚ, poly = divisor * q + remainder :=
sorry

end remainder_when_divided_l18_18240


namespace Marie_speed_l18_18286

theorem Marie_speed (distance time : ℕ) (h1 : distance = 372) (h2 : time = 31) : distance / time = 12 :=
by
  have h3 : distance = 372 := h1
  have h4 : time = 31 := h2
  sorry

end Marie_speed_l18_18286


namespace largest_prime_factor_of_1001_l18_18319

theorem largest_prime_factor_of_1001 :
  (∃ p : ℕ, nat.prime p ∧ p ∣ 1001 ∧ (∀ q : ℕ, nat.prime q → q ∣ 1001 → q ≤ p) ∧ p = 13) :=
begin
  sorry
end

end largest_prime_factor_of_1001_l18_18319


namespace hyperbola_center_l18_18806

theorem hyperbola_center :
  ∃ c : ℝ × ℝ, (c = (3, 4) ∧ ∀ x y : ℝ, 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0 ↔ (x - 3)^2 / 4 - (y - 4)^2 / 1 = 1) :=
sorry

end hyperbola_center_l18_18806


namespace son_l18_18632

-- Define the context of the problem with conditions
variables (S M : ℕ)

-- Condition 1: The man is 28 years older than his son
def condition1 : Prop := M = S + 28

-- Condition 2: In two years, the man's age will be twice the son's age
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The final statement to prove the son's present age
theorem son's_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 26 :=
by
  sorry

end son_l18_18632


namespace total_bottle_caps_in_collection_l18_18789

-- Statements of given conditions
def small_box_caps : ℕ := 35
def large_box_caps : ℕ := 75
def num_small_boxes : ℕ := 7
def num_large_boxes : ℕ := 3
def individual_caps : ℕ := 23

-- Theorem statement that needs to be proved
theorem total_bottle_caps_in_collection :
  small_box_caps * num_small_boxes + large_box_caps * num_large_boxes + individual_caps = 493 :=
by sorry

end total_bottle_caps_in_collection_l18_18789


namespace number_of_teachers_students_possible_rental_plans_economical_plan_l18_18478

-- Definitions of conditions

def condition1 (x y : ℕ) : Prop := y - 30 * x = 7
def condition2 (x y : ℕ) : Prop := 31 * x - y = 1
def capacity_condition (m : ℕ) : Prop := 35 * m + 30 * (8 - m) ≥ 255
def rental_fee_condition (m : ℕ) : Prop := 400 * m + 320 * (8 - m) ≤ 3000

-- Problems to prove

theorem number_of_teachers_students (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 8 ∧ y = 247 := 
by sorry

theorem possible_rental_plans (m : ℕ) (h_cap : capacity_condition m) (h_fee : rental_fee_condition m) : m = 3 ∨ m = 4 ∨ m = 5 := 
by sorry

theorem economical_plan (m : ℕ) (h_fee : rental_fee_condition 3) (h_fee_alt1 : rental_fee_condition 4) (h_fee_alt2 : rental_fee_condition 5) : m = 3 := 
by sorry

end number_of_teachers_students_possible_rental_plans_economical_plan_l18_18478


namespace pete_backwards_speed_l18_18427

variable (speed_pete_hands : ℕ) (speed_tracy_cartwheel : ℕ) (speed_susan_walk : ℕ) (speed_pete_backwards : ℕ)

axiom pete_hands_speed : speed_pete_hands = 2
axiom pete_hands_speed_quarter_tracy_cartwheel : speed_pete_hands = speed_tracy_cartwheel / 4
axiom tracy_cartwheel_twice_susan_walk : speed_tracy_cartwheel = 2 * speed_susan_walk
axiom pete_backwards_three_times_susan_walk : speed_pete_backwards = 3 * speed_susan_walk

theorem pete_backwards_speed : 
  speed_pete_backwards = 12 :=
by
  sorry

end pete_backwards_speed_l18_18427


namespace point_in_third_quadrant_l18_18709

theorem point_in_third_quadrant
  (a b : ℝ)
  (hne : a ≠ 0)
  (y_increase : ∀ x1 x2, x1 < x2 → -5 * a * x1 + b < -5 * a * x2 + b)
  (ab_pos : a * b > 0) : 
  a < 0 ∧ b < 0 :=
by
  sorry

end point_in_third_quadrant_l18_18709


namespace two_colonies_limit_l18_18928

def doubles_each_day (size: ℕ) (day: ℕ) : ℕ := size * 2 ^ day

theorem two_colonies_limit (habitat_limit: ℕ) (initial_size: ℕ) : 
  (∀ t, doubles_each_day initial_size t = habitat_limit → t = 20) → 
  initial_size > 0 →
  ∀ t, doubles_each_day (2 * initial_size) t = habitat_limit → t = 20 :=
by
  sorry

end two_colonies_limit_l18_18928


namespace symmetric_difference_card_l18_18022

variable (x y : Finset ℤ)
variable (h1 : x.card = 16)
variable (h2 : y.card = 18)
variable (h3 : (x ∩ y).card = 6)

theorem symmetric_difference_card :
  (x \ y ∪ y \ x).card = 22 := by sorry

end symmetric_difference_card_l18_18022


namespace total_volume_of_barrel_l18_18179

-- Define the total volume of the barrel and relevant conditions.
variable (x : ℝ) -- total volume of the barrel

-- State the given condition about the barrel's honey content.
def condition := (0.7 * x - 0.3 * x = 30)

-- Goal to prove:
theorem total_volume_of_barrel : condition x → x = 75 :=
by
  sorry

end total_volume_of_barrel_l18_18179


namespace problem1_problem2_problem3_l18_18869

-- Definition of sets A, B, and U
def A : Set ℤ := {1, 2, 3, 4, 5}
def B : Set ℤ := {-1, 1, 2, 3}
def U : Set ℤ := {x | -1 ≤ x ∧ x < 6}

-- The complement of B in U
def C_U (B : Set ℤ) : Set ℤ := {x ∈ U | x ∉ B}

-- Problem statements
theorem problem1 : A ∩ B = {1, 2, 3} := by sorry
theorem problem2 : A ∪ B = {-1, 1, 2, 3, 4, 5} := by sorry
theorem problem3 : (C_U B) ∩ A = {4, 5} := by sorry

end problem1_problem2_problem3_l18_18869


namespace reciprocal_of_5_is_1_div_5_l18_18581

-- Define the concept of reciprocal
def is_reciprocal (a b : ℚ) : Prop := a * b = 1

-- The problem statement: Prove that the reciprocal of 5 is 1/5
theorem reciprocal_of_5_is_1_div_5 : is_reciprocal 5 (1 / 5) :=
by
  sorry

end reciprocal_of_5_is_1_div_5_l18_18581


namespace number_of_methods_l18_18009

def doctors : ℕ := 6
def days : ℕ := 3

theorem number_of_methods : (days^doctors) = 729 := 
by sorry

end number_of_methods_l18_18009


namespace total_time_equals_l18_18354

-- Define the distances and speeds
def first_segment_distance : ℝ := 50
def first_segment_speed : ℝ := 30
def second_segment_distance (b : ℝ) : ℝ := b
def second_segment_speed : ℝ := 80

-- Prove that the total time is equal to (400 + 3b) / 240 hours
theorem total_time_equals (b : ℝ) : 
  (first_segment_distance / first_segment_speed) + (second_segment_distance b / second_segment_speed) 
  = (400 + 3 * b) / 240 := 
by
  sorry

end total_time_equals_l18_18354


namespace num_girls_l18_18005

-- Define conditions as constants
def ratio (B G : ℕ) : Prop := B = (5 * G) / 8
def total (B G : ℕ) : Prop := B + G = 260

-- State the proof problem
theorem num_girls (B G : ℕ) (h1 : ratio B G) (h2 : total B G) : G = 160 :=
by {
  -- actual proof omitted
  sorry
}

end num_girls_l18_18005


namespace equivalent_trigonometric_identity_l18_18695

variable (α : ℝ)

theorem equivalent_trigonometric_identity
  (h1 : α ∈ Set.Ioo (-(Real.pi/2)) 0)
  (h2 : Real.sin (α + (Real.pi/4)) = -1/3) :
  (Real.sin (2*α) / Real.cos ((Real.pi/4) - α)) = 7/3 := 
by
  sorry

end equivalent_trigonometric_identity_l18_18695


namespace find_f_2018_l18_18087

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom f_zero : f 0 = -1
axiom functional_equation (x : ℝ) : f x = -f (2 - x)

theorem find_f_2018 : f 2018 = 1 := 
by 
  sorry

end find_f_2018_l18_18087


namespace green_beads_in_pattern_l18_18304

noncomputable def G : ℕ := 3
def P : ℕ := 5
def R (G : ℕ) : ℕ := 2 * G
def total_beads (G : ℕ) (P : ℕ) (R : ℕ) : ℕ := 3 * (G + P + R) + 10 * 5 * (G + P + R)

theorem green_beads_in_pattern :
  total_beads 3 5 (R 3) = 742 :=
by
  sorry

end green_beads_in_pattern_l18_18304


namespace triangle_inequality_l18_18717

open Real

variables {a b c S : ℝ}

-- Assuming a, b, c are the sides of a triangle
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
-- Assuming S is the area of the triangle
axiom Herons_area : S = sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_inequality : 
  a^2 + b^2 + c^2 ≥ 4 * S * sqrt 3 ∧ (a^2 + b^2 + c^2 = 4 * S * sqrt 3 ↔ a = b ∧ b = c) := sorry

end triangle_inequality_l18_18717


namespace total_soccer_games_l18_18605

theorem total_soccer_games (months : ℕ) (games_per_month : ℕ) (h_months : months = 3) (h_games_per_month : games_per_month = 9) : months * games_per_month = 27 :=
by
  sorry

end total_soccer_games_l18_18605


namespace train_crossing_time_l18_18318

theorem train_crossing_time
  (L1 L2 : ℝ) (v : ℝ) 
  (t1 t2 t : ℝ) 
  (h_t1 : t1 = 27)
  (h_t2 : t2 = 17)
  (hv_ratio : v = v)
  (h_L1 : L1 = v * t1)
  (h_L2 : L2 = v * t2)
  (h_t12 : t = (L1 + L2) / (v + v)) :
  t = 22 :=
by
  sorry

end train_crossing_time_l18_18318


namespace possible_values_of_a_l18_18724

theorem possible_values_of_a (x y a : ℝ) (h1 : x + y = a) (h2 : x^3 + y^3 = a) (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 :=
by sorry

end possible_values_of_a_l18_18724


namespace Martha_needs_54_cakes_l18_18288

theorem Martha_needs_54_cakes :
  let n_children := 3
  let n_cakes_per_child := 18
  let n_cakes_total := 54
  n_cakes_total = n_children * n_cakes_per_child :=
by
  sorry

end Martha_needs_54_cakes_l18_18288


namespace field_dimension_area_l18_18039

theorem field_dimension_area (m : ℝ) : (3 * m + 8) * (m - 3) = 120 → m = 7 :=
by
  sorry

end field_dimension_area_l18_18039


namespace more_people_joined_l18_18012

def initial_people : Nat := 61
def final_people : Nat := 83

theorem more_people_joined :
  final_people - initial_people = 22 := by
  sorry

end more_people_joined_l18_18012


namespace cube_prism_surface_area_l18_18768

theorem cube_prism_surface_area (a : ℝ) (h : a > 0) :
  2 * (6 * a^2) > 4 * a^2 + 2 * (2 * a * a) :=
by sorry

end cube_prism_surface_area_l18_18768


namespace profit_percentage_is_20_l18_18188

variable (C : ℝ) -- Assuming the cost price C is a real number.

theorem profit_percentage_is_20 
  (h1 : 10 * 1 = 12 * (C / 1)) :  -- Shopkeeper sold 10 articles at the cost price of 12 articles.
  ((12 * C - 10 * C) / (10 * C)) * 100 = 20 := 
by
  sorry

end profit_percentage_is_20_l18_18188


namespace QT_value_l18_18269

noncomputable def find_QT (PQ RS PT : ℝ) : ℝ :=
  let tan_gamma := (RS / PQ)
  let QT := (RS / tan_gamma) - PT
  QT

theorem QT_value :
  let PQ := 45
  let RS := 75
  let PT := 15
  find_QT PQ RS PT = 210 := by
  sorry

end QT_value_l18_18269


namespace box_height_l18_18546

theorem box_height (x h : ℕ) 
  (h1 : h = x + 5) 
  (h2 : 6 * x^2 + 20 * x ≥ 150) 
  (h3 : 5 * x + 5 ≥ 25) 
  : h = 9 :=
by 
  sorry

end box_height_l18_18546


namespace hexagon_perimeter_of_intersecting_triangles_l18_18449

/-- Given two equilateral triangles with parallel sides, where the perimeter of the blue triangle 
    is 4 and the perimeter of the green triangle is 5, prove that the perimeter of the hexagon 
    formed by their intersection is 3. -/
theorem hexagon_perimeter_of_intersecting_triangles 
    (P_blue P_green P_hexagon : ℝ)
    (h_blue : P_blue = 4)
    (h_green : P_green = 5) :
    P_hexagon = 3 := 
sorry

end hexagon_perimeter_of_intersecting_triangles_l18_18449


namespace ant_minimum_distance_l18_18353

section
variables (x y z w u : ℝ)

-- Given conditions
axiom h1 : x + y + z = 22
axiom h2 : w + y + z = 29
axiom h3 : x + y + u = 30

-- Prove the ant crawls at least 47 cm to cover all paths
theorem ant_minimum_distance : x + y + z + w ≥ 47 :=
sorry
end

end ant_minimum_distance_l18_18353


namespace binom_9_5_l18_18991

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l18_18991


namespace rectangle_area_l18_18439

theorem rectangle_area :
  ∃ (L B : ℝ), (L - B = 23) ∧ (2 * (L + B) = 206) ∧ (L * B = 2520) :=
sorry

end rectangle_area_l18_18439


namespace find_y_value_l18_18440

noncomputable def y_value (y : ℝ) : Prop :=
  let side1_sq_area := 9 * y^2
  let side2_sq_area := 36 * y^2
  let triangle_area := 9 * y^2
  (side1_sq_area + side2_sq_area + triangle_area = 1000)

theorem find_y_value (y : ℝ) : y_value y → y = 10 * Real.sqrt 3 / 3 :=
sorry

end find_y_value_l18_18440


namespace car_city_mileage_l18_18182

theorem car_city_mileage (h c t : ℝ) 
  (h_eq : h * t = 462)
  (c_eq : (h - 15) * t = 336) 
  (c_def : c = h - 15) : 
  c = 40 := 
by 
  sorry

end car_city_mileage_l18_18182


namespace number_of_flute_players_l18_18308

theorem number_of_flute_players (F T B D C H : ℕ)
  (hT : T = 3 * F)
  (hB : B = T - 8)
  (hD : D = B + 11)
  (hC : C = 2 * F)
  (hH : H = B + 3)
  (h_total : F + T + B + D + C + H = 65) :
  F = 6 :=
by
  sorry

end number_of_flute_players_l18_18308


namespace log_579_between_consec_ints_l18_18750

theorem log_579_between_consec_ints (a b : ℤ) (h₁ : 2 < Real.log 579 / Real.log 10) (h₂ : Real.log 579 / Real.log 10 < 3) : a + b = 5 :=
sorry

end log_579_between_consec_ints_l18_18750


namespace company_fund_amount_l18_18444

theorem company_fund_amount (n : ℕ) (h : 70 * n + 160 = 80 * n - 8) : 
  80 * n - 8 = 1352 :=
sorry

end company_fund_amount_l18_18444


namespace exists_five_positive_integers_sum_20_product_420_l18_18726
-- Import the entirety of Mathlib to ensure all necessary definitions are available

-- Lean statement for the proof problem
theorem exists_five_positive_integers_sum_20_product_420 :
  ∃ (a b c d e : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ a + b + c + d + e = 20 ∧ a * b * c * d * e = 420 :=
sorry

end exists_five_positive_integers_sum_20_product_420_l18_18726


namespace jeremy_money_left_l18_18273

theorem jeremy_money_left (computer_cost : ℕ) (accessories_percentage : ℕ) (factor : ℕ)
  (h1 : computer_cost = 3000)
  (h2 : accessories_percentage = 10)
  (h3 : factor = 2) :
  let accessories_cost := (accessories_percentage * computer_cost) / 100 in
  let total_money_before := factor * computer_cost in
  let total_spent := computer_cost + accessories_cost in
  let money_left := total_money_before - total_spent in
  money_left = 2700 :=
by
  sorry

end jeremy_money_left_l18_18273


namespace rate_of_second_batch_l18_18194

-- Define the problem statement
theorem rate_of_second_batch
  (rate_first : ℝ)
  (weight_first weight_second weight_total : ℝ)
  (rate_mixture : ℝ)
  (profit_multiplier : ℝ) 
  (total_selling_price : ℝ) :
  rate_first = 11.5 →
  weight_first = 30 →
  weight_second = 20 →
  weight_total = weight_first + weight_second →
  rate_mixture = 15.12 →
  profit_multiplier = 1.20 →
  total_selling_price = weight_total * rate_mixture →
  (rate_first * weight_first + (weight_second * x) * profit_multiplier = total_selling_price) →
  x = 14.25 :=
by
  intros
  sorry

end rate_of_second_batch_l18_18194


namespace solve_equation_l18_18598

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l18_18598


namespace weight_of_replaced_person_l18_18305

-- Define the conditions
variables (W : ℝ) (new_person_weight : ℝ) (avg_weight_increase : ℝ)
#check ℝ

def initial_group_size := 10

-- Define the conditions as hypothesis statements
axiom weight_increase_eq : avg_weight_increase = 3.5
axiom new_person_weight_eq : new_person_weight = 100

-- Define the result to be proved
theorem weight_of_replaced_person (W : ℝ) : 
  ∀ (avg_weight_increase : ℝ) (new_person_weight : ℝ),
    avg_weight_increase = 3.5 ∧ new_person_weight = 100 → 
    (new_person_weight - (avg_weight_increase * initial_group_size)) = 65 := 
by
  sorry

end weight_of_replaced_person_l18_18305


namespace least_multiple_greater_than_500_l18_18765

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 500 ∧ n % 32 = 0 := by
  let n := 512
  have h1 : n > 500 := by 
    -- proof omitted, as we're not solving the problem here
    sorry
  have h2 : n % 32 = 0 := by 
    -- proof omitted
    sorry
  exact ⟨n, h1, h2⟩

end least_multiple_greater_than_500_l18_18765


namespace annual_percentage_increase_l18_18307

theorem annual_percentage_increase (present_value future_value : ℝ) (years: ℝ) (r : ℝ) 
  (h1 : present_value = 20000)
  (h2 : future_value = 24200)
  (h3 : years = 2) : 
  future_value = present_value * (1 + r)^years → r = 0.1 :=
sorry

end annual_percentage_increase_l18_18307


namespace solve_equation_l18_18596

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l18_18596


namespace union_of_sets_l18_18527

def A : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }
def C : Set ℝ := { x : ℝ | -1 ≤ x ∧ x < 1 }

theorem union_of_sets :
  A ∪ B = C := 
sorry

end union_of_sets_l18_18527


namespace part1_part2_part3_l18_18081

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  abs (x^2 - 1) + x^2 + k * x

theorem part1 (h : 2 = 2) :
  (f (- (1 + Real.sqrt 3) /2) 2 = 0) ∧ (f (-1/2) 2 = 0) := by
  sorry

theorem part2 (h_alpha : 0 < α) (h_beta : α < β) (h_beta2 : β < 2) (h_f_alpha : f α k = 0) (h_f_beta : f β k = 0) :
  -7/2 < k ∧ k < -1 := by
  sorry

theorem part3 (h_alpha : 0 < α) (h_alpha1 : α ≤ 1) (h_beta1 : 1 < β) (h_beta2 : β < 2) (h1 : k = - 1 / α) (h2 : 2 * β^2 + k * β - 1 = 0) :
  1/α + 1/β < 4 := by
  sorry

end part1_part2_part3_l18_18081


namespace number_is_2250_l18_18915

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l18_18915


namespace celery_cost_l18_18872

noncomputable def supermarket_problem
  (total_money : ℕ)
  (price_cereal discount_cereal price_bread : ℕ)
  (price_milk discount_milk price_potato num_potatoes : ℕ)
  (leftover_money : ℕ) 
  (total_cost : ℕ) 
  (cost_of_celery : ℕ) :=
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) + 
   leftover_money = total_money ∧
  total_cost = total_money - leftover_money ∧
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) = total_cost - cost_of_celery

theorem celery_cost (total_money : ℕ := 60) 
  (price_cereal : ℕ := 12) 
  (discount_cereal : ℕ := 50) 
  (price_bread : ℕ := 8) 
  (price_milk : ℕ := 10) 
  (discount_milk : ℕ := 90) 
  (price_potato : ℕ := 1) 
  (num_potatoes : ℕ := 6) 
  (leftover_money : ℕ := 26) 
  (total_cost : ℕ := 34) :
  supermarket_problem total_money price_cereal discount_cereal price_bread price_milk discount_milk price_potato num_potatoes leftover_money total_cost 5 :=
by
  sorry

end celery_cost_l18_18872


namespace digit_9_appears_301_times_l18_18104

def countDigitOccurrences (d : ℕ) (a b : ℕ) : ℕ :=
  list.count d (list.join (list.map toDigits (list.range' a (b - a + 1))))

theorem digit_9_appears_301_times :
  countDigitOccurrences 9 1 1000 = 301 := by
  sorry

end digit_9_appears_301_times_l18_18104


namespace equilateral_triangle_side_length_l18_18635

theorem equilateral_triangle_side_length (perimeter : ℕ) (h_perimeter : perimeter = 69) : 
  ∃ (side_length : ℕ), side_length = perimeter / 3 := 
by
  sorry

end equilateral_triangle_side_length_l18_18635


namespace W_k_two_lower_bound_l18_18574

-- Define W(k, 2)
def W (k : ℕ) (c : ℕ) : ℕ := -- smallest number such that for every n >= W(k, 2), 
  -- any 2-coloring of the set {1, 2, ..., n} contains a monochromatic arithmetic progression of length k
  sorry 

-- Define the statement to prove
theorem W_k_two_lower_bound (k : ℕ) : ∃ C > 0, W k 2 ≥ C * 2^(k / 2) :=
by
  sorry

end W_k_two_lower_bound_l18_18574


namespace nine_appears_300_times_l18_18110

-- We will define a function that counts the digit 9 in the range from 1 to 1000
-- and prove that it indeed equals 300.

def count_digit_9 (n : ℕ) : ℕ :=
  (n.to_digits 10).count 9

def count_digit_9_range (start : ℕ) (end : ℕ) : ℕ :=
  (list.range' start (end - start + 1)).sum count_digit_9

theorem nine_appears_300_times : count_digit_9_range 1 1000 = 300 :=
by
  sorry -- Proof to be filled in

end nine_appears_300_times_l18_18110


namespace female_rainbow_trout_l18_18131

-- Define the conditions given in the problem
variables (F_s M_s M_r F_r T : ℕ)
variables (h1 : F_s + M_s = 645)
variables (h2 : M_s = 2 * F_s + 45)
variables (h3 : 4 * M_r = 3 * F_s)
variables (h4 : 20 * M_r = 3 * T)
variables (h5 : T = 645 + F_r + M_r)

theorem female_rainbow_trout :
  F_r = 205 :=
by
  sorry

end female_rainbow_trout_l18_18131


namespace sum_h_k_a_b_l18_18306

-- Defining h, k, a, and b with their respective given values
def h : Int := -4
def k : Int := 2
def a : Int := 5
def b : Int := 3

-- Stating the theorem to prove \( h + k + a + b = 6 \)
theorem sum_h_k_a_b : h + k + a + b = 6 := by
  /- Proof omitted as per instructions -/
  sorry

end sum_h_k_a_b_l18_18306


namespace square_area_from_diagonal_l18_18609

theorem square_area_from_diagonal
  (d : ℝ) (h : d = 10) : ∃ (A : ℝ), A = 50 :=
by {
  -- here goes the proof
  sorry
}

end square_area_from_diagonal_l18_18609


namespace min_value_of_expression_l18_18147

open Real

theorem min_value_of_expression (x y z : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : 0 < z) (h₃ : x * y * z = 18) :
  x^2 + 4*x*y + y^2 + 3*z^2 ≥ 63 :=
sorry

end min_value_of_expression_l18_18147


namespace f_17_l18_18332

def f : ℕ → ℤ := sorry

axiom f_prop1 : f 1 = 0
axiom f_prop2 : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) = f m + f n + 4 * (9 * m * n - 1)

theorem f_17 : f 17 = 1052 := by
  sorry

end f_17_l18_18332


namespace minimize_quadratic_l18_18611

theorem minimize_quadratic (x : ℝ) : (x = -9 / 2) → ∀ y : ℝ, y^2 + 9 * y + 7 ≥ (-9 / 2)^2 + 9 * -9 / 2 + 7 :=
by sorry

end minimize_quadratic_l18_18611


namespace valid_values_for_D_l18_18705

-- Definitions for the distinct digits and the non-zero condition
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9
def distinct_nonzero_digits (A B C D : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Condition for the carry situation
def carry_in_addition (A B C D : ℕ) : Prop :=
  ∃ carry1 carry2 carry3 carry4 : ℕ,
  (A + B + carry1) % 10 = D ∧ (B + C + carry2) % 10 = A ∧
  (C + C + carry3) % 10 = B ∧ (A + B + carry4) % 10 = C ∧
  (carry1 = 1 ∨ carry2 = 1 ∨ carry3 = 1 ∨ carry4 = 1)

-- Main statement
theorem valid_values_for_D (A B C D : ℕ) :
  distinct_nonzero_digits A B C D →
  carry_in_addition A B C D →
  ∃ n, n = 5 :=
sorry

end valid_values_for_D_l18_18705


namespace total_time_before_main_game_l18_18711

-- Define the time spent on each activity according to the conditions
def download_time := 10
def install_time := download_time / 2
def update_time := 2 * download_time
def account_time := 5
def internet_issues_time := 15
def discussion_time := 20
def video_time := 8

-- Define the total preparation time
def preparation_time := download_time + install_time + update_time + account_time + internet_issues_time + discussion_time + video_time

-- Define the in-game tutorial time
def tutorial_time := 3 * preparation_time

-- Prove that the total time before playing the main game is 332 minutes
theorem total_time_before_main_game : preparation_time + tutorial_time = 332 := by
  -- Provide a detailed proof here
  sorry

end total_time_before_main_game_l18_18711


namespace sequence_divisible_by_11_l18_18675

theorem sequence_divisible_by_11 
  (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
  (∀ n, n = 4 ∨ n = 8 ∨ n ≥ 10 → 11 ∣ a n) := sorry

end sequence_divisible_by_11_l18_18675


namespace problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l18_18924

-- Define Lean goals for the true statements
theorem problem_part_a (x : ℝ) (h : x < 0) : x^3 < x := sorry
theorem problem_part_b (x : ℝ) (h : x^3 > 0) : x > 0 := sorry
theorem problem_part_e (x : ℝ) (h : x > 1) : x^3 > x := sorry

-- Disprove the false statements by showing the negation
theorem problem_part_c_disproof (x : ℝ) (h : x^3 < x) : ¬ (|x| > 1) := sorry
theorem problem_part_d_disproof (x : ℝ) (h : x^3 > x) : ¬ (x > 1) := sorry

end problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l18_18924


namespace one_number_greater_than_one_l18_18295

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1)
  (h_sum : a + b + c > 1/a + 1/b + 1/c) :
  ((1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ 1 < b ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ 1 < c)) 
  ∧ (¬ ((1 < a ∧ 1 < b) ∨ (1 < b ∧ 1 < c) ∨ (1 < a ∧ 1 < c))) :=
sorry

end one_number_greater_than_one_l18_18295


namespace unique_n_in_range_satisfying_remainders_l18_18098

theorem unique_n_in_range_satisfying_remainders : 
  ∃! (n : ℤ) (k r : ℤ), 150 < n ∧ n < 250 ∧ 0 <= r ∧ r <= 6 ∧ n = 7 * k + r ∧ n = 9 * r + r :=
sorry

end unique_n_in_range_satisfying_remainders_l18_18098


namespace max_profit_l18_18779

noncomputable def profit (x : ℝ) : ℝ :=
  20 * x - 3 * x^2 + 96 * Real.log x - 90

theorem max_profit :
  ∃ x : ℝ, 4 ≤ x ∧ x ≤ 12 ∧ 
  (∀ y : ℝ, 4 ≤ y ∧ y ≤ 12 → profit y ≤ profit x) ∧ profit x = 96 * Real.log 6 - 78 :=
by
  sorry

end max_profit_l18_18779


namespace total_comics_in_box_l18_18200

theorem total_comics_in_box 
  (pages_per_comic : ℕ)
  (total_pages_found : ℕ)
  (untorn_comics : ℕ)
  (comics_fixed : ℕ := total_pages_found / pages_per_comic)
  (total_comics : ℕ := comics_fixed + untorn_comics)
  (h_pages_per_comic : pages_per_comic = 25)
  (h_total_pages_found : total_pages_found = 150)
  (h_untorn_comics : untorn_comics = 5) :
  total_comics = 11 :=
by
  sorry

end total_comics_in_box_l18_18200


namespace find_cost_of_book_sold_at_loss_l18_18333

-- Definitions from the conditions
def total_cost (C1 C2 : ℝ) : Prop := C1 + C2 = 540
def selling_price_loss (C1 : ℝ) : ℝ := 0.85 * C1
def selling_price_gain (C2 : ℝ) : ℝ := 1.19 * C2
def same_selling_price (SP1 SP2 : ℝ) : Prop := SP1 = SP2

theorem find_cost_of_book_sold_at_loss (C1 C2 : ℝ) 
  (h1 : total_cost C1 C2) 
  (h2 : same_selling_price (selling_price_loss C1) (selling_price_gain C2)) :
  C1 = 315 :=
by {
   sorry
}

end find_cost_of_book_sold_at_loss_l18_18333


namespace total_fencing_cost_l18_18697

theorem total_fencing_cost
  (park_is_square : true)
  (cost_per_side : ℕ)
  (h1 : cost_per_side = 43) :
  4 * cost_per_side = 172 :=
by
  sorry

end total_fencing_cost_l18_18697


namespace total_earrings_after_one_year_l18_18959

theorem total_earrings_after_one_year :
  let bella_earrings := 10
  let monica_earrings := 10 / 0.25
  let rachel_earrings := monica_earrings / 2
  let initial_total := bella_earrings + monica_earrings + rachel_earrings
  let olivia_earrings_initial := initial_total + 5
  let olivia_earrings_after := olivia_earrings_initial * 1.2
  let total_earrings := bella_earrings + monica_earrings + rachel_earrings + olivia_earrings_after
  total_earrings = 160 :=
by
  sorry

end total_earrings_after_one_year_l18_18959


namespace interest_years_proof_l18_18403

theorem interest_years_proof :
  let interest_r800_first_2_years := 800 * 0.05 * 2
  let interest_r800_next_3_years := 800 * 0.12 * 3
  let total_interest_r800 := interest_r800_first_2_years + interest_r800_next_3_years
  let interest_r600_first_3_years := 600 * 0.07 * 3
  let interest_r600_next_n_years := 600 * 0.10 * n
  (interest_r600_first_3_years + interest_r600_next_n_years = total_interest_r800) ->
  n = 5 →
  3 + n = 8 :=
by
  sorry

end interest_years_proof_l18_18403


namespace wristwatch_cost_proof_l18_18359

-- Definition of the problem conditions
def allowance_per_week : ℕ := 5
def initial_weeks : ℕ := 10
def initial_savings : ℕ := 20
def additional_weeks : ℕ := 16

-- The total cost of the wristwatch
def wristwatch_cost : ℕ := 100

-- Let's state the proof problem
theorem wristwatch_cost_proof :
  (initial_savings + additional_weeks * allowance_per_week) = wristwatch_cost :=
by
  sorry

end wristwatch_cost_proof_l18_18359


namespace perfect_square_expression_l18_18910

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 :=
by
  sorry

end perfect_square_expression_l18_18910


namespace compute_combination_l18_18982

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l18_18982


namespace which_set_forms_triangle_l18_18614

def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem which_set_forms_triangle : 
  satisfies_triangle_inequality 4 3 6 ∧ 
  ¬ satisfies_triangle_inequality 1 2 3 ∧ 
  ¬ satisfies_triangle_inequality 7 8 16 ∧ 
  ¬ satisfies_triangle_inequality 9 10 20 :=
by
  sorry

end which_set_forms_triangle_l18_18614


namespace max_second_smallest_l18_18433

noncomputable def f (M : ℕ) : ℕ :=
  (M - 1) * (90 - M) * (89 - M) * (88 - M)

theorem max_second_smallest (M : ℕ) (cond : 1 ≤ M ∧ M ≤ 89) : M = 23 ↔ (∀ N : ℕ, f M ≥ f N) :=
by
  sorry

end max_second_smallest_l18_18433


namespace seats_with_middle_empty_l18_18857

-- Define the parameters
def chairs := 5
def people := 4
def middle_empty := 3

-- Define the function to calculate seating arrangements
def number_of_ways (people : ℕ) (chairs : ℕ) (middle_empty : ℕ) : ℕ := 
  if chairs < people + 1 then 0
  else (chairs - 1) * (chairs - 2) * (chairs - 3) * (chairs - 4)

-- The theorem to prove the number of ways given the conditions
theorem seats_with_middle_empty : number_of_ways 4 5 3 = 24 := by
  sorry

end seats_with_middle_empty_l18_18857


namespace min_sum_abc_l18_18408

theorem min_sum_abc (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a * b * c + b * c + c = 2014) : a + b + c = 40 :=
sorry

end min_sum_abc_l18_18408


namespace ralph_total_cost_correct_l18_18156

noncomputable def calculate_total_cost : ℝ :=
  let original_cart_cost := 54.00
  let small_issue_item_original := 20.00
  let additional_item_original := 15.00
  let small_issue_discount := 0.20
  let additional_item_discount := 0.25
  let coupon_discount := 0.10
  let sales_tax := 0.07

  -- Calculate the discounted prices
  let small_issue_discounted := small_issue_item_original * (1 - small_issue_discount)
  let additional_item_discounted := additional_item_original * (1 - additional_item_discount)

  -- Total cost before the coupon and tax
  let total_before_coupon := original_cart_cost + small_issue_discounted + additional_item_discounted

  -- Apply the coupon discount
  let total_after_coupon := total_before_coupon * (1 - coupon_discount)

  -- Apply the sales tax
  total_after_coupon * (1 + sales_tax)

-- Define the problem statement
theorem ralph_total_cost_correct : calculate_total_cost = 78.24 :=
by sorry

end ralph_total_cost_correct_l18_18156


namespace largest_A_l18_18613

def A : ℝ := (2010 / 2009) + (2010 / 2011)
def B : ℝ := (2010 / 2011) + (2012 / 2011)
def C : ℝ := (2011 / 2010) + (2011 / 2012)

theorem largest_A : A > B ∧ A > C := by sorry

end largest_A_l18_18613


namespace valid_base6_number_2015_l18_18493

def is_valid_base6_digit (d : Nat) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def is_base6_number (n : Nat) : Prop :=
  ∀ (digit : Nat), digit ∈ (n.digits 10) → is_valid_base6_digit digit

theorem valid_base6_number_2015 : is_base6_number 2015 := by
  sorry

end valid_base6_number_2015_l18_18493


namespace algebraic_expression_evaluation_l18_18538

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + 3 * x - 5 = 2) : 2 * x^2 + 6 * x - 3 = 11 :=
sorry

end algebraic_expression_evaluation_l18_18538


namespace quadratic_real_roots_m_l18_18699

theorem quadratic_real_roots_m (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 + 4 * x1 + m = 0 ∧ x2 * x2 + 4 * x2 + m = 0) →
  m ≤ 4 :=
by
  sorry

end quadratic_real_roots_m_l18_18699


namespace units_digit_of_p_is_6_l18_18819

theorem units_digit_of_p_is_6 (p : ℕ) (h_even : Even p) (h_units_p_plus_1 : (p + 1) % 10 = 7) (h_units_p3_minus_p2 : ((p^3) % 10 - (p^2) % 10) % 10 = 0) : p % 10 = 6 := 
by 
  -- proof steps go here
  sorry

end units_digit_of_p_is_6_l18_18819


namespace parabola_points_count_l18_18508

theorem parabola_points_count :
  ∃ n : ℕ, n = 8 ∧ 
    (∀ x y : ℕ, (y = -((x^2 : ℤ) / 3) + 7 * (x : ℤ) + 54) → 1 ≤ x ∧ x ≤ 26 ∧ x % 3 = 0) :=
by
  sorry

end parabola_points_count_l18_18508


namespace arith_seq_sum_l18_18270

-- We start by defining what it means for a sequence to be arithmetic
def is_arith_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

-- We are given that a_2 = 5 and a_6 = 33 for an arithmetic sequence
variable (a : ℕ → ℤ)
variable (h_arith : is_arith_seq a)
variable (h1 : a 2 = 5)
variable (h2 : a 6 = 33)

-- The statement we want to prove
theorem arith_seq_sum (a : ℕ → ℤ) (h_arith : is_arith_seq a) (h1 : a 2 = 5) (h2 : a 6 = 33) :
  (a 3 + a 5) = 38 :=
  sorry

end arith_seq_sum_l18_18270


namespace total_students_l18_18619

-- Definition of the problem conditions
def ratio_boys_girls : ℕ := 8
def ratio_girls : ℕ := 5
def number_girls : ℕ := 160

-- The main theorem statement
theorem total_students (b g : ℕ) (h1 : b * ratio_girls = g * ratio_boys_girls) (h2 : g = number_girls) :
  b + g = 416 :=
sorry

end total_students_l18_18619


namespace Sue_necklace_total_beads_l18_18047

theorem Sue_necklace_total_beads :
  ∃ (purple blue green red total : ℕ),
  purple = 7 ∧
  blue = 2 * purple ∧
  green = blue + 11 ∧
  (red : ℕ) = green / 2 ∧
  total = purple + blue + green + red ∧
  total % 2 = 0 ∧
  total = 58 := by
    sorry

end Sue_necklace_total_beads_l18_18047


namespace solution_set_for_inequality_l18_18511

open Set Real

theorem solution_set_for_inequality : 
  { x : ℝ | (2 * x) / (x + 1) ≤ 1 } = Ioc (-1 : ℝ) 1 := 
sorry

end solution_set_for_inequality_l18_18511


namespace perfect_square_expression_l18_18908

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 :=
by
  sorry

end perfect_square_expression_l18_18908


namespace car_distance_problem_l18_18939

-- A definition for the initial conditions.
def initial_conditions (D : ℝ) (S : ℝ) (T : ℝ) : Prop :=
  T = 6 ∧ S = 50 ∧ (3/2 * T = 9)

-- The statement corresponding to the given problem.
theorem car_distance_problem (D : ℝ) (S : ℝ) (T : ℝ) :
  initial_conditions D S T → D = 450 :=
by
  -- leave the proof as an exercise.
  sorry

end car_distance_problem_l18_18939


namespace geometric_series_sum_l18_18790

noncomputable def geometric_sum (a r : ℚ) (h : |r| < 1) : ℚ :=
a / (1 - r)

theorem geometric_series_sum :
  geometric_sum 1 (1/3) (by norm_num) = 3/2 :=
by
  sorry

end geometric_series_sum_l18_18790


namespace log_comparisons_l18_18561

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 3 / (2 * Real.log 2)
noncomputable def c := 1 / 2

theorem log_comparisons : c < b ∧ b < a := 
by
  sorry

end log_comparisons_l18_18561


namespace nine_appears_300_times_l18_18109

-- We will define a function that counts the digit 9 in the range from 1 to 1000
-- and prove that it indeed equals 300.

def count_digit_9 (n : ℕ) : ℕ :=
  (n.to_digits 10).count 9

def count_digit_9_range (start : ℕ) (end : ℕ) : ℕ :=
  (list.range' start (end - start + 1)).sum count_digit_9

theorem nine_appears_300_times : count_digit_9_range 1 1000 = 300 :=
by
  sorry -- Proof to be filled in

end nine_appears_300_times_l18_18109


namespace solve_equation_l18_18591

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l18_18591


namespace probability_no_3by3_red_grid_correct_l18_18219

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l18_18219


namespace not_divides_l18_18566

theorem not_divides (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) : ¬ d ∣ a^(2^n) + 1 := 
sorry

end not_divides_l18_18566


namespace express_function_as_chain_of_equalities_l18_18064

theorem express_function_as_chain_of_equalities (x : ℝ) : 
  ∃ (u : ℝ), (u = 2 * x - 5) ∧ ((2 * x - 5) ^ 10 = u ^ 10) :=
by 
  sorry

end express_function_as_chain_of_equalities_l18_18064


namespace total_comics_in_box_l18_18199

theorem total_comics_in_box 
  (pages_per_comic : ℕ)
  (total_pages_found : ℕ)
  (untorn_comics : ℕ)
  (comics_fixed : ℕ := total_pages_found / pages_per_comic)
  (total_comics : ℕ := comics_fixed + untorn_comics)
  (h_pages_per_comic : pages_per_comic = 25)
  (h_total_pages_found : total_pages_found = 150)
  (h_untorn_comics : untorn_comics = 5) :
  total_comics = 11 :=
by
  sorry

end total_comics_in_box_l18_18199


namespace scientific_notation_of_coronavirus_diameter_l18_18578

theorem scientific_notation_of_coronavirus_diameter:
  0.000000907 = 9.07 * 10^(-7) :=
sorry

end scientific_notation_of_coronavirus_diameter_l18_18578


namespace age_difference_l18_18166

theorem age_difference (A B C : ℕ) (h1 : A + B > B + C) (h2 : C = A - 17) : (A + B) - (B + C) = 17 :=
by
  sorry

end age_difference_l18_18166


namespace find_m_plus_n_l18_18224

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l18_18224


namespace area_of_triangle_BFE_l18_18429

theorem area_of_triangle_BFE (A B C D E F : ℝ × ℝ) (u v : ℝ) 
  (h_rectangle : (0, 0) = A ∧ (3 * u, 0) = B ∧ (3 * u, 3 * v) = C ∧ (0, 3 * v) = D)
  (h_E : E = (0, 2 * v))
  (h_F : F = (2 * u, 0))
  (h_area_rectangle : 3 * u * 3 * v = 48) :
  ∃ (area : ℝ), area = |3 * u * 2 * v| / 2 ∧ area = 24 :=
by 
  sorry

end area_of_triangle_BFE_l18_18429


namespace solve_equation_l18_18599

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l18_18599


namespace time_comparison_l18_18142

-- Definitions from the conditions
def speed_first_trip (v : ℝ) : ℝ := v
def distance_first_trip : ℝ := 80
def distance_second_trip : ℝ := 240
def speed_second_trip (v : ℝ) : ℝ := 4 * v

-- Theorem to prove
theorem time_comparison (v : ℝ) (hv : v > 0) :
  (distance_second_trip / speed_second_trip v) = (3 / 4) * (distance_first_trip / speed_first_trip v) :=
by
  -- Outline of the proof, we skip the actual steps
  sorry

end time_comparison_l18_18142


namespace phase_shift_cos_l18_18509

theorem phase_shift_cos (b c : ℝ) (h_b : b = 2) (h_c : c = π / 2) :
  (-c / b) = -π / 4 :=
by
  rw [h_b, h_c]
  sorry

end phase_shift_cos_l18_18509


namespace quadratic_inequality_solution_l18_18839

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l18_18839


namespace count_digit_9_l18_18115

theorem count_digit_9 (n : ℕ) (h : n = 1000) : 
  (Finset.range n).sum (λ x, (Nat.digits 10 x).count 9) = 300 :=
by
  sorry

end count_digit_9_l18_18115


namespace evaluate_square_difference_l18_18063

theorem evaluate_square_difference:
  let a := 70
  let b := 30
  (a^2 - b^2) = 4000 :=
by
  sorry

end evaluate_square_difference_l18_18063


namespace balloon_rearrangements_l18_18679

-- Define the letters involved: vowels and consonants
def vowels := ['A', 'O', 'O', 'O']
def consonants := ['B', 'L', 'L', 'N']

-- State the problem in Lean 4:
theorem balloon_rearrangements : 
  ∃ n : ℕ, 
  (∀ (vowels := ['A', 'O', 'O', 'O']) 
     (consonants := ['B', 'L', 'L', 'N']), 
     n = 32) := sorry  -- we state that the number of rearrangements is 32 but do not provide the proof itself.

end balloon_rearrangements_l18_18679


namespace conference_handshakes_l18_18937

theorem conference_handshakes (n : ℕ) (h : n = 10) :
  (n * (n - 1)) / 2 = 45 :=
by
  sorry

end conference_handshakes_l18_18937


namespace jesse_money_left_after_mall_l18_18554

theorem jesse_money_left_after_mall :
  ∀ (initial_amount novel_cost lunch_cost total_spent remaining_amount : ℕ),
    initial_amount = 50 →
    novel_cost = 7 →
    lunch_cost = 2 * novel_cost →
    total_spent = novel_cost + lunch_cost →
    remaining_amount = initial_amount - total_spent →
    remaining_amount = 29 :=
by
  intros initial_amount novel_cost lunch_cost total_spent remaining_amount
  sorry

end jesse_money_left_after_mall_l18_18554


namespace sum_twice_father_age_plus_son_age_l18_18398

/-- 
  Given:
  1. Twice the son's age plus the father's age equals 70.
  2. Father's age is 40.
  3. Son's age is 15.

  Prove:
  The sum when twice the father's age is added to the son's age is 95.
-/
theorem sum_twice_father_age_plus_son_age :
  ∀ (father_age son_age : ℕ), 
    2 * son_age + father_age = 70 → 
    father_age = 40 → 
    son_age = 15 → 
    2 * father_age + son_age = 95 := by
  intros
  sorry

end sum_twice_father_age_plus_son_age_l18_18398


namespace seven_horses_meet_at_same_time_l18_18167

theorem seven_horses_meet_at_same_time:
  ∃ T, (T > 0 ∧ (∃ seven_horses: Finset ℕ, seven_horses.card = 7 ∧ 
    All (λ k, k ∈ seven_horses → T % (k + 1) = 0) (Finset.range 12)) ∧ 
    Nat.digits 10 T |>.sum = 6) :=
begin
  sorry
end

end seven_horses_meet_at_same_time_l18_18167


namespace percentage_discount_is_12_l18_18032

noncomputable def cost_price : ℝ := 47.50
noncomputable def list_price : ℝ := 67.47
noncomputable def desired_selling_price : ℝ := cost_price + 0.25 * cost_price
noncomputable def actual_selling_price : ℝ := 59.375

theorem percentage_discount_is_12 :
  ∃ D : ℝ, desired_selling_price = list_price - (list_price * D) ∧ D = 0.12 := 
by 
  sorry

end percentage_discount_is_12_l18_18032


namespace exists_lcm_lt_l18_18827

theorem exists_lcm_lt (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1) (hp_gt_one : p > 1) (hq_gt_one : q > 1) (hpq_diff_gt_one : (p < q ∧ q - p > 1) ∨ (p > q ∧ p - q > 1)) :
  ∃ n : ℕ, Nat.lcm (p + n) (q + n) < Nat.lcm p q := by
  sorry

end exists_lcm_lt_l18_18827


namespace nasadkas_in_barrel_l18_18858

def capacity (B N V : ℚ) :=
  (B + 20 * V = 3 * B) ∧ (19 * B + N + 15.5 * V = 20 * B + 8 * V)

theorem nasadkas_in_barrel (B N V : ℚ) (h : capacity B N V) : B / N = 4 :=
by
  sorry

end nasadkas_in_barrel_l18_18858


namespace Jesse_remaining_money_l18_18551

-- Define the conditions
def initial_money := 50
def novel_cost := 7
def lunch_cost := 2 * novel_cost
def total_spent := novel_cost + lunch_cost

-- Define the remaining money after spending
def remaining_money := initial_money - total_spent

-- Prove that the remaining money is $29
theorem Jesse_remaining_money : remaining_money = 29 := 
by
  sorry

end Jesse_remaining_money_l18_18551


namespace smallest_lambda_inequality_l18_18510

theorem smallest_lambda_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * y * (x^2 + y^2) + y * z * (y^2 + z^2) + z * x * (z^2 + x^2) ≤ (1 / 8) * (x + y + z)^4 :=
sorry

end smallest_lambda_inequality_l18_18510


namespace servings_per_day_l18_18239

-- Conditions
def week_servings := 21
def days_per_week := 7

-- Question and Answer
theorem servings_per_day : week_servings / days_per_week = 3 := 
by
  sorry

end servings_per_day_l18_18239


namespace John_l18_18532

/-- Assume Grant scored 10 points higher on his math test than John.
John received a certain ratio of points as Hunter who scored 45 points on his math test.
Grant's test score was 100. -/
theorem John's_points_to_Hunter's_points_ratio 
  (Grant John Hunter : ℕ) 
  (h1 : Grant = John + 10)
  (h2 : Hunter = 45)
  (h_grant_score : Grant = 100) : 
  (John : ℚ) / (Hunter : ℚ) = 2 / 1 :=
sorry

end John_l18_18532


namespace sum_m_n_l18_18232

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l18_18232


namespace required_hemispherical_containers_l18_18490

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

end required_hemispherical_containers_l18_18490


namespace fraction_meaningful_range_l18_18896

theorem fraction_meaningful_range (x : ℝ) : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_range_l18_18896


namespace min_value_xyz_l18_18563

theorem min_value_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_prod : x * y * z = 8) : 
  x + 3 * y + 6 * z ≥ 18 :=
sorry

end min_value_xyz_l18_18563


namespace positive_number_solution_exists_l18_18761

theorem positive_number_solution_exists (x : ℝ) (h₁ : 0 < x) (h₂ : (2 / 3) * x = (64 / 216) * (1 / x)) : x = 2 / 3 :=
by sorry

end positive_number_solution_exists_l18_18761


namespace log_expression_value_l18_18361

theorem log_expression_value : 
  let log4_3 := (Real.log 3) / (Real.log 4)
  let log8_3 := (Real.log 3) / (Real.log 8)
  let log3_2 := (Real.log 2) / (Real.log 3)
  let log9_2 := (Real.log 2) / (Real.log 9)
  (log4_3 + log8_3) * (log3_2 + log9_2) = 5 / 4 := 
by
  sorry

end log_expression_value_l18_18361


namespace solution_set_fractional_inequality_l18_18073

theorem solution_set_fractional_inequality (x : ℝ) (h : x ≠ -2) :
  (x + 1) / (x + 2) < 0 ↔ x ∈ Ioo (-2 : ℝ) (-1 : ℝ) := sorry

end solution_set_fractional_inequality_l18_18073


namespace minimum_value_f_l18_18661

theorem minimum_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : (∃ x y : ℝ, (x + y + x * y) ≥ - (9 / 8) ∧ 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :=
sorry

end minimum_value_f_l18_18661


namespace find_n_l18_18074

-- Definitions based on conditions
variables (x n y : ℕ)
variable (h1 : x / n = 3 / 2)
variable (h2 : (7 * x + n * y) / (x - n * y) = 23)

-- Proof that n is equivalent to 1 given the conditions.
theorem find_n : n = 1 :=
sorry

end find_n_l18_18074


namespace triangle_inequality_equality_iff_equilateral_l18_18149

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

theorem equality_iff_equilateral (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c := 
sorry

end triangle_inequality_equality_iff_equilateral_l18_18149


namespace four_digit_multiples_of_13_and_7_l18_18832

theorem four_digit_multiples_of_13_and_7 : 
  (∃ n : ℕ, 
    (∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧ k % 91 = 0 → k = 1001 + 91 * (n - 11)) 
    ∧ n - 11 + 1 = 99) :=
by
  sorry

end four_digit_multiples_of_13_and_7_l18_18832


namespace plane_divided_into_four_regions_l18_18503

-- Definition of the conditions
def line1 (x y : ℝ) : Prop := y = 3 * x
def line2 (x y : ℝ) : Prop := y = (1 / 3) * x

-- Proof statement
theorem plane_divided_into_four_regions :
  (∃ f g : ℝ → ℝ, ∀ x, line1 x (f x) ∧ line2 x (g x)) →
  ∃ n : ℕ, n = 4 :=
by sorry

end plane_divided_into_four_regions_l18_18503


namespace maximum_value_expr_l18_18067

theorem maximum_value_expr :
  ∀ (a b c d : ℝ), (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) ∧ (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1) →
  a + b + c + d - a * b - b * c - c * d - d * a ≤ 2 :=
by
  intros a b c d h
  sorry

end maximum_value_expr_l18_18067


namespace remainder_of_470521_div_5_l18_18461

theorem remainder_of_470521_div_5 : 470521 % 5 = 1 := 
by sorry

end remainder_of_470521_div_5_l18_18461


namespace measure_15_minutes_with_hourglasses_l18_18255

theorem measure_15_minutes_with_hourglasses (h7 h11 : ℕ) (h7_eq : h7 = 7) (h11_eq : h11 = 11) : ∃ t : ℕ, t = 15 :=
by
  let t := 15
  have h7 : ℕ := 7
  have h11 : ℕ := 11
  exact ⟨t, by norm_num⟩

end measure_15_minutes_with_hourglasses_l18_18255


namespace correct_calculation_l18_18466

theorem correct_calculation (m n : ℝ) : -m^2 * n - 2 * m^2 * n = -3 * m^2 * n :=
by
  sorry

end correct_calculation_l18_18466


namespace none_of_the_above_option_l18_18144

-- Define integers m and n
variables (m n: ℕ)

-- Define P and R in terms of m and n
def P : ℕ := 2^m
def R : ℕ := 5^n

-- Define the statement to prove
theorem none_of_the_above_option : ∀ (m n : ℕ), 15^(m + n) ≠ P^(m + n) * R ∧ 15^(m + n) ≠ (3^m * 3^n * 5^m) ∧ 15^(m + n) ≠ (3^m * P^n) ∧ 15^(m + n) ≠ (2^m * 5^n * 5^m) :=
by sorry

end none_of_the_above_option_l18_18144


namespace closest_number_to_fraction_l18_18801

theorem closest_number_to_fraction (x : ℝ) : 
  (abs (x - 2000) < abs (x - 1500)) ∧ 
  (abs (x - 2000) < abs (x - 2500)) ∧ 
  (abs (x - 2000) < abs (x - 3000)) ∧ 
  (abs (x - 2000) < abs (x - 3500)) :=
by
  let x := 504 / 0.252
  sorry

end closest_number_to_fraction_l18_18801


namespace circle_projections_l18_18774

open Real

def is_prime (n : ℕ) : Prop := ∃ p : ℕ, nat.prime p ∧ p = n

theorem circle_projections :
  ∀ (r : ℝ) (p q : ℕ) (m n : ℕ),
  r ∈ set.Ioo 0 ⊤ ∧
  r ≠ 0 ∧
  r % 2 = 1 ∧
  is_prime p ∧
  is_prime q ∧
  0 < m ∧ 0 < n ∧
  let u := p^m in
  let v := q^n in
  u^2 + v^2 = r^2 ∧
  u > v →
  let A := (r, 0) in
  let B := (-r, 0) in
  let C := (0, -r) in
  let D := (0, r) in
  let P := (u, v) in
  let M := (u, 0) in
  let N := (0, v) in
  dist A M = 1 ∧
  dist B M = 9 ∧
  dist C N = 8 ∧
  dist D N = 2 :=
begin
  intro r p q m n,
  rintro ⟨hr_pos, hr_nonzero, hr_odd, prime_p, prime_q, hm_pos, hn_pos, hu, hv, huv_eq, h_u_gt_v⟩,
  let u := p^m,
  let v := q^n,
  simp only [u, v] at huv_eq h_u_gt_v,
  have h_hyp := dist_eq.users P M A u v r huv_eq p q m n hr_nonzero hr_pos prime_p prime_q hm_pos hn_pos,
  let A := (r, 0),
  let B := (-r, 0),
  let C := (0, -r),
  let D := (0, r),
  let P := (u, v),
  let M := (u, 0),
  let N := (0, v),
  suffices : dist A M = 1 ∧ dist B M = 9 ∧ dist C N = 8 ∧ dist D N = 2,
  exact this,
  exact sorry
end

end circle_projections_l18_18774


namespace number_of_new_students_l18_18703

theorem number_of_new_students (initial_students left_students final_students new_students : ℕ) 
  (h_initial : initial_students = 4) 
  (h_left : left_students = 3) 
  (h_final : final_students = 43) : 
  new_students = final_students - (initial_students - left_students) :=
by 
  sorry

end number_of_new_students_l18_18703


namespace diff_set_Q_minus_P_l18_18277

def P (x : ℝ) : Prop := 1 - (2 / x) < 0
def Q (x : ℝ) : Prop := |x - 2| < 1
def diff_set (P Q : ℝ → Prop) (x : ℝ) : Prop := Q x ∧ ¬ P x

theorem diff_set_Q_minus_P :
  ∀ x : ℝ, diff_set Q P x ↔ (2 ≤ x ∧ x < 3) :=
by
  sorry

end diff_set_Q_minus_P_l18_18277


namespace ab_cd_zero_l18_18674

theorem ab_cd_zero {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) (h3 : ac + bd = 0) : ab + cd = 0 :=
sorry

end ab_cd_zero_l18_18674


namespace find_teachers_and_students_l18_18480

-- Mathematical statements corresponding to the problem conditions
def teachers_and_students_found (x y : ℕ) : Prop :=
  (y = 30 * x + 7) ∧ (31 * x = y + 1)

-- The theorem we need to prove
theorem find_teachers_and_students : ∃ x y, teachers_and_students_found x y ∧ x = 8 ∧ y = 247 :=
  by
    sorry

end find_teachers_and_students_l18_18480


namespace add_salt_solution_l18_18680

theorem add_salt_solution
  (initial_amount : ℕ) (added_concentration : ℕ) (desired_concentration : ℕ)
  (initial_concentration : ℝ) :
  initial_amount = 50 ∧ initial_concentration = 0.4 ∧ added_concentration = 10 ∧ desired_concentration = 25 →
  (∃ (x : ℕ), x = 50 ∧ 
    (initial_concentration * initial_amount + 0.1 * x) / (initial_amount + x) = 0.25) :=
by
  sorry

end add_salt_solution_l18_18680


namespace dave_spent_on_books_l18_18368

-- Define the cost of books in each category without any discounts or taxes
def cost_animal_books : ℝ := 8 * 10
def cost_outer_space_books : ℝ := 6 * 12
def cost_train_books : ℝ := 9 * 8
def cost_history_books : ℝ := 4 * 15
def cost_science_books : ℝ := 5 * 18

-- Define the discount and tax rates
def discount_animal_books : ℝ := 0.10
def tax_science_books : ℝ := 0.15

-- Apply the discount to animal books
def discounted_cost_animal_books : ℝ := cost_animal_books * (1 - discount_animal_books)

-- Apply the tax to science books
def final_cost_science_books : ℝ := cost_science_books * (1 + tax_science_books)

-- Calculate the total cost of all books after discounts and taxes
def total_cost : ℝ := discounted_cost_animal_books 
                  + cost_outer_space_books
                  + cost_train_books
                  + cost_history_books
                  + final_cost_science_books

theorem dave_spent_on_books : total_cost = 379.5 := by
  sorry

end dave_spent_on_books_l18_18368


namespace time_between_train_arrivals_l18_18052

-- Define the conditions as given in the problem statement
def passengers_per_train : ℕ := 320 + 200
def total_passengers_per_hour : ℕ := 6240
def minutes_per_hour : ℕ := 60

-- Declare the statement to be proven
theorem time_between_train_arrivals: 
  (total_passengers_per_hour / passengers_per_train) = (minutes_per_hour / 5) := by 
  sorry

end time_between_train_arrivals_l18_18052


namespace quadratic_function_expression_l18_18820

theorem quadratic_function_expression : 
  ∃ (a : ℝ), (a ≠ 0) ∧ (∀ x : ℝ, x = -1 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ x : ℝ, x = 2 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ y : ℝ, ∃ x : ℝ, x = 0 ∧ y = -2 → y = a * (x + 1) * (x - 2)) 
  → (∀ x : ℝ, ∃ y : ℝ, y = x^2 - x - 2) := 
sorry

end quadratic_function_expression_l18_18820


namespace trapezoid_area_pqrs_l18_18459

theorem trapezoid_area_pqrs :
  let P := (1, 1)
  let Q := (1, 4)
  let R := (6, 4)
  let S := (7, 1)
  let parallelogram := true -- indicates that PQ and RS are parallel
  let PQ := abs (Q.2 - P.2)
  let RS := abs (S.1 - R.1)
  let height := abs (R.1 - P.1)
  (1 / 2 : ℚ) * (PQ + RS) * height = 10 := by
  sorry

end trapezoid_area_pqrs_l18_18459


namespace train_passing_time_l18_18488

def train_distance_km : ℝ := 10
def train_time_min : ℝ := 15
def train_length_m : ℝ := 111.11111111111111

theorem train_passing_time : 
  let time_to_pass_signal_post := train_length_m / ((train_distance_km * 1000) / (train_time_min * 60))
  time_to_pass_signal_post = 10 :=
by
  sorry

end train_passing_time_l18_18488


namespace math_proof_problem_l18_18382

-- Defining the problem condition
def condition (x y z : ℝ) := 
  x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0

-- Adding constraints to x, y, z
def constraints (x y z : ℝ) :=
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x ≠ y ∨ y ≠ z ∨ z ≠ x)

-- Stating the main theorem
theorem math_proof_problem (x y z : ℝ) (h_condition : condition x y z) (h_constraints : constraints x y z) :
  x + y + z = 3 ∧ x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6 := 
sorry

end math_proof_problem_l18_18382


namespace joey_needs_figures_to_cover_cost_l18_18715

-- Definitions based on conditions
def cost_sneakers : ℕ := 92
def earnings_per_lawn : ℕ := 8
def lawns : ℕ := 3
def earnings_per_hour : ℕ := 5
def work_hours : ℕ := 10
def price_per_figure : ℕ := 9

-- Total earnings from mowing lawns
def earnings_lawns := lawns * earnings_per_lawn
-- Total earnings from job
def earnings_job := work_hours * earnings_per_hour
-- Total earnings from both
def total_earnings := earnings_lawns + earnings_job
-- Remaining amount to cover the cost
def remaining_amount := cost_sneakers - total_earnings

-- Correct answer based on the problem statement
def collectible_figures_needed := remaining_amount / price_per_figure

-- Lean 4 statement to prove the requirement
theorem joey_needs_figures_to_cover_cost :
  collectible_figures_needed = 2 := by
  sorry

end joey_needs_figures_to_cover_cost_l18_18715


namespace num_valid_k_l18_18070

/--
The number of natural numbers \( k \), not exceeding 485000, 
such that \( k^2 - 1 \) is divisible by 485 is 4000.
-/
theorem num_valid_k (n : ℕ) (h₁ : n ≤ 485000) (h₂ : 485 ∣ (n^2 - 1)) : 
  (∃ k : ℕ, k = 4000) :=
sorry

end num_valid_k_l18_18070


namespace consecutive_odd_integers_sum_l18_18911

theorem consecutive_odd_integers_sum (x : ℤ) (h : x + (x + 4) = 134) : x + (x + 2) + (x + 4) = 201 := 
by sorry

end consecutive_odd_integers_sum_l18_18911


namespace union_M_N_l18_18824

-- Define the set M
def M : Set ℤ := {x | x^2 - x = 0}

-- Define the set N
def N : Set ℤ := {y | y^2 + y = 0}

-- Prove that the union of M and N is {-1, 0, 1}
theorem union_M_N :
  M ∪ N = {-1, 0, 1} :=
by
  sorry

end union_M_N_l18_18824


namespace empty_rooms_le_1000_l18_18539

/--
In a 50x50 grid where each cell can contain at most one tree, 
with the following rules: 
1. A pomegranate tree has at least one apple neighbor
2. A peach tree has at least one apple neighbor and one pomegranate neighbor
3. An empty room has at least one apple neighbor, one pomegranate neighbor, and one peach neighbor
Show that the number of empty rooms is not greater than 1000.
-/
theorem empty_rooms_le_1000 (apple pomegranate peach : ℕ) (empty : ℕ)
  (h1 : apple + pomegranate + peach + empty = 2500)
  (h2 : ∀ p, pomegranate ≥ p → apple ≥ 1)
  (h3 : ∀ p, peach ≥ p → apple ≥ 1 ∧ pomegranate ≥ 1)
  (h4 : ∀ e, empty ≥ e → apple ≥ 1 ∧ pomegranate ≥ 1 ∧ peach ≥ 1) :
  empty ≤ 1000 :=
sorry

end empty_rooms_le_1000_l18_18539


namespace inequality_proof_l18_18416

noncomputable def sum_expression (a b c : ℝ) : ℝ :=
  (1 / (b * c + a + 1 / a)) + (1 / (c * a + b + 1 / b)) + (1 / (a * b + c + 1 / c))

theorem inequality_proof (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  sum_expression a b c ≤ 27 / 31 :=
by
  sorry

end inequality_proof_l18_18416


namespace add_to_divisible_l18_18327

theorem add_to_divisible (n d x : ℕ) (h : n = 987654) (h1 : d = 456) (h2 : x = 222) : 
  (n + x) % d = 0 := 
by {
  sorry
}

end add_to_divisible_l18_18327


namespace scholars_number_l18_18739

theorem scholars_number (n : ℕ) : n < 600 ∧ n % 15 = 14 ∧ n % 19 = 13 → n = 509 :=
by
  intro h
  sorry

end scholars_number_l18_18739


namespace difference_of_numbers_l18_18165

variable (x y d : ℝ)

theorem difference_of_numbers
  (h1 : x + y = 5)
  (h2 : x - y = d)
  (h3 : x^2 - y^2 = 50) :
  d = 10 :=
by
  sorry

end difference_of_numbers_l18_18165


namespace sign_of_ac_l18_18516

theorem sign_of_ac (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b) + (c / d) = (a + c) / (b + d)) : a * c < 0 :=
by
  sorry

end sign_of_ac_l18_18516


namespace count_five_digit_multiples_of_5_l18_18095

-- Define the range of five-digit positive integers
def lower_bound : ℕ := 10000
def upper_bound : ℕ := 99999

-- Define the divisor
def divisor : ℕ := 5

-- Define the count of multiples of 5 in the range
def count_multiples_of_5 : ℕ :=
  (upper_bound / divisor) - (lower_bound / divisor) + 1

-- The main statement: The number of five-digit multiples of 5 is 18000
theorem count_five_digit_multiples_of_5 : count_multiples_of_5 = 18000 :=
  sorry

end count_five_digit_multiples_of_5_l18_18095


namespace add_base7_l18_18492

-- Define the two numbers in base 7 to be added.
def number1 : ℕ := 2 * 7 + 5
def number2 : ℕ := 5 * 7 + 4

-- Define the expected result in base 7.
def expected_sum : ℕ := 1 * 7^2 + 1 * 7 + 2

theorem add_base7 :
  let sum : ℕ := number1 + number2
  sum = expected_sum := sorry

end add_base7_l18_18492


namespace sequence_expression_l18_18814

theorem sequence_expression (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) - 2 * a n = 2^n) :
  ∀ n, a n = n * 2^(n - 1) :=
by
  sorry

end sequence_expression_l18_18814


namespace probability_age_21_to_30_l18_18268

theorem probability_age_21_to_30 : 
  let total_people := 160 
  let people_10_to_20 := 40
  let people_21_to_30 := 70
  let people_31_to_40 := 30
  let people_41_to_50 := 20
  (people_21_to_30 / total_people : ℚ) = 7 / 16 := by
  sorry

end probability_age_21_to_30_l18_18268


namespace find_m_plus_n_l18_18225

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l18_18225


namespace determine_x_l18_18121

-- Definitions for given conditions
variables (x y z a b c : ℝ)
variables (h₁ : xy / (x - y) = a) (h₂ : xz / (x - z) = b) (h₃ : yz / (y - z) = c)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- Main statement to prove
theorem determine_x :
  x = (2 * a * b * c) / (a * b + b * c + c * a) :=
sorry

end determine_x_l18_18121


namespace polygon_diagonals_150_sides_l18_18054

-- Define the function to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The theorem to state what we want to prove
theorem polygon_diagonals_150_sides : num_diagonals 150 = 11025 :=
by sorry

end polygon_diagonals_150_sides_l18_18054


namespace polygon_sum_13th_position_l18_18636

theorem polygon_sum_13th_position :
  let sum_n : ℕ := (100 * 101) / 2;
  2 * sum_n = 10100 :=
by
  sorry

end polygon_sum_13th_position_l18_18636


namespace num_satisfying_n_conditions_l18_18099

theorem num_satisfying_n_conditions :
  let count := {n : ℤ | 150 < n ∧ n < 250 ∧ (n % 7 = n % 9) }.toFinset.card
  count = 7 :=
by
  sorry

end num_satisfying_n_conditions_l18_18099


namespace problem1_l18_18474

theorem problem1 (x : ℝ) : abs (2 * x - 3) < 1 ↔ 1 < x ∧ x < 2 := sorry

end problem1_l18_18474


namespace polynomial_value_l18_18671

variable (x : ℝ)

theorem polynomial_value (h : x^2 - 2*x + 6 = 9) : 2*x^2 - 4*x + 6 = 12 :=
by
  sorry

end polynomial_value_l18_18671


namespace value_of_f_5_l18_18244

variable (a b c m : ℝ)

-- Conditions: definition of f and given value of f(-5)
def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2
axiom H1 : f a b c (-5) = m

-- Question: Prove that f(5) = -m + 4
theorem value_of_f_5 : f a b c 5 = -m + 4 :=
by
  sorry

end value_of_f_5_l18_18244


namespace find_d_for_single_point_l18_18301

/--
  Suppose that the graph of \(3x^2 + y^2 + 6x - 6y + d = 0\) consists of a single point.
  Prove that \(d = 12\).
-/
theorem find_d_for_single_point : 
  ∀ (d : ℝ), (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0) ∧
              (∀ (x1 y1 x2 y2 : ℝ), 
                (3 * x1^2 + y1^2 + 6 * x1 - 6 * y1 + d = 0 ∧ 
                 3 * x2^2 + y2^2 + 6 * x2 - 6 * y2 + d = 0 → 
                 x1 = x2 ∧ y1 = y2)) ↔ d = 12 := 
by 
  sorry

end find_d_for_single_point_l18_18301


namespace monica_problem_l18_18727

open Real

noncomputable def completingSquare : Prop :=
  ∃ (b c : ℤ), (∀ x : ℝ, (x - 4) ^ 2 = x^2 - 8 * x + 16) ∧ b = -4 ∧ c = 8 ∧ (b + c = 4)

theorem monica_problem : completingSquare := by
  sorry

end monica_problem_l18_18727


namespace alice_expected_games_l18_18008

-- Defining the initial conditions
def skill_levels := Fin 21

def initial_active_player := 0

-- Defining Alice's skill level
def Alice_skill_level := 11

-- Define the tournament structure and conditions
def tournament_round (active: skill_levels) (inactive: Set skill_levels) : skill_levels :=
  sorry

-- Define the expected number of games Alice plays
noncomputable def expected_games_Alice_plays : ℚ :=
  sorry

-- Statement of the problem proving the expected number of games Alice plays
theorem alice_expected_games : expected_games_Alice_plays = 47 / 42 :=
sorry

end alice_expected_games_l18_18008


namespace sequence_a_n_term_l18_18710

theorem sequence_a_n_term :
  ∃ a : ℕ → ℕ, 
  a 1 = 1 ∧
  (∀ n : ℕ, a (n+1) = 2 * a n + 1) ∧
  a 10 = 1023 := by
  sorry

end sequence_a_n_term_l18_18710


namespace pythagorean_theorem_sets_l18_18467

theorem pythagorean_theorem_sets :
  ¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2) ∧
  (1 ^ 2 + (Real.sqrt 3) ^ 2 = 2 ^ 2) ∧
  ¬ (5 ^ 2 + 6 ^ 2 = 7 ^ 2) ∧
  ¬ (1 ^ 2 + (Real.sqrt 2) ^ 2 = 3 ^ 2) :=
by {
  sorry
}

end pythagorean_theorem_sets_l18_18467


namespace binomial_coefficient_9_5_l18_18999

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l18_18999


namespace three_friends_at_least_50_mushrooms_l18_18883

theorem three_friends_at_least_50_mushrooms (a : Fin 7 → ℕ) (h_sum : (Finset.univ.sum a) = 100) (h_different : Function.Injective a) :
  ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
by
  sorry

end three_friends_at_least_50_mushrooms_l18_18883


namespace vasilyev_max_car_loan_l18_18161

-- Define the incomes
def parents_salary := 71000
def rental_income := 11000
def scholarship := 2600

-- Define the expenses
def utility_payments := 8400
def food_expenses := 18000
def transportation_expenses := 3200
def tutor_fees := 2200
def miscellaneous_expenses := 18000

-- Define the emergency fund percentage
def emergency_fund_percentage := 0.1

-- Theorem to prove the maximum car loan payment
theorem vasilyev_max_car_loan : 
  let total_income := parents_salary + rental_income + scholarship,
      total_expenses := utility_payments + food_expenses + transportation_expenses + tutor_fees + miscellaneous_expenses,
      remaining_income := total_income - total_expenses,
      emergency_fund := emergency_fund_percentage * remaining_income,
      max_car_loan := remaining_income - emergency_fund in
  max_car_loan = 31320 := by
  sorry

end vasilyev_max_car_loan_l18_18161


namespace project_completion_days_l18_18351

theorem project_completion_days 
  (total_mandays : ℕ)
  (initial_workers : ℕ)
  (leaving_workers : ℕ)
  (remaining_workers : ℕ)
  (days_total : ℕ) :
  total_mandays = 200 →
  initial_workers = 10 →
  leaving_workers = 4 →
  remaining_workers = 6 →
  days_total = 40 :=
by
  intros h0 h1 h2 h3
  sorry

end project_completion_days_l18_18351


namespace angle_B_in_parallelogram_l18_18405

variable (A B : ℝ)

theorem angle_B_in_parallelogram (h_parallelogram : ∀ {A B C D : ℝ}, A + B = 180 ↔ A = B) 
  (h_A : A = 50) : B = 130 := by
  sorry

end angle_B_in_parallelogram_l18_18405


namespace solution_interval_l18_18507

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x + x - 2

theorem solution_interval :
  ∃ x, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end solution_interval_l18_18507


namespace fraction_of_beans_remaining_l18_18335

variables (J B R : ℝ)

-- Given conditions
def condition1 : Prop := J = 0.10 * (J + B)
def condition2 : Prop := J + R = 0.60 * (J + B)

theorem fraction_of_beans_remaining (h1 : condition1 J B) (h2 : condition2 J B R) :
  R / B = 5 / 9 :=
  sorry

end fraction_of_beans_remaining_l18_18335


namespace probability_no_3x3_red_square_l18_18223

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l18_18223


namespace num_four_digit_divisibles_l18_18831

theorem num_four_digit_divisibles : 
  ∃ n, n = 99 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 91 = 0 ↔ x ∈ (set.Ico 1001 9999) :=
by
  sorry

end num_four_digit_divisibles_l18_18831


namespace feathers_before_crossing_road_l18_18889

theorem feathers_before_crossing_road : 
  ∀ (F : ℕ), 
  (F - (2 * 23) = 5217) → 
  F = 5263 :=
by
  intros F h
  sorry

end feathers_before_crossing_road_l18_18889


namespace line_through_points_a_minus_b_l18_18264

theorem line_through_points_a_minus_b :
  ∃ a b : ℝ, 
  (∀ x, (x = 3 → 7 = a * 3 + b) ∧ (x = 6 → 19 = a * 6 + b)) → 
  a - b = 9 :=
by
  sorry

end line_through_points_a_minus_b_l18_18264


namespace determinant_sin_eq_zero_l18_18967

theorem determinant_sin_eq_zero (a b : ℝ) : 
  matrix.det ![
    ![1, Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by
  sorry

end determinant_sin_eq_zero_l18_18967


namespace min_sum_is_11_over_28_l18_18718

-- Definition of the problem
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the minimum sum problem
def min_sum (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧ D ∈ digits →
  ((A : ℚ) / B + (C : ℚ) / D) = (11 : ℚ) / 28

-- The theorem statement
theorem min_sum_is_11_over_28 :
  ∃ A B C D : ℕ, min_sum A B C D :=
sorry

end min_sum_is_11_over_28_l18_18718


namespace coprime_divisibility_by_240_l18_18879

theorem coprime_divisibility_by_240 (n : ℤ) (h : Int.gcd n 30 = 1) : 240 ∣ (-n^4 - 1) := sorry

end coprime_divisibility_by_240_l18_18879


namespace race_result_l18_18927

theorem race_result
    (distance_race : ℕ)
    (distance_diff : ℕ)
    (distance_second_start_diff : ℕ)
    (speed_xm speed_xl : ℕ)
    (h1 : distance_race = 100)
    (h2 : distance_diff = 20)
    (h3 : distance_second_start_diff = 20)
    (xm_wins_first_race : speed_xm * distance_race >= speed_xl * (distance_race - distance_diff)) :
    speed_xm * (distance_race + distance_second_start_diff) >= speed_xl * (distance_race + distance_diff) :=
by
  sorry

end race_result_l18_18927


namespace tom_books_problem_l18_18607

theorem tom_books_problem 
  (original_books : ℕ)
  (books_sold : ℕ)
  (books_bought : ℕ)
  (h1 : original_books = 5)
  (h2 : books_sold = 4)
  (h3 : books_bought = 38) : 
  original_books - books_sold + books_bought = 39 :=
by
  sorry

end tom_books_problem_l18_18607


namespace rectangle_max_area_l18_18352

theorem rectangle_max_area (w : ℝ) (h : ℝ) (hw : h = 2 * w) (perimeter : 2 * (w + h) = 40) :
  w * h = 800 / 9 := 
by
  -- Given: h = 2w and 2(w + h) = 40
  -- We need to prove that the area A = wh = 800/9
  sorry

end rectangle_max_area_l18_18352


namespace figure4_total_length_l18_18639

-- Define the conditions
def top_segments_sum := 3 + 1 + 1  -- Sum of top segments in Figure 3
def bottom_segment := top_segments_sum -- Bottom segment length in Figure 3
def vertical_segment1 := 10  -- First vertical segment length
def vertical_segment2 := 9  -- Second vertical segment length
def remaining_segment := 1  -- The remaining horizontal segment

-- Total length of remaining segments in Figure 4
theorem figure4_total_length : 
  bottom_segment + vertical_segment1 + vertical_segment2 + remaining_segment = 25 := by
  sorry

end figure4_total_length_l18_18639


namespace math_proof_problem_l18_18851

noncomputable def problem1 (A : ℝ) : Prop :=
  (1/2) * real.cos (2 * A) = real.cos A ^ 2 - real.cos A 

noncomputable def problem2 (a b c : ℝ) (A B C : ℝ) : Prop := 
  a = 3 ∧ real.sin B = 2 * real.sin C → 
  1/2 * b * c * real.sin A = (3 * real.sqrt 3) / 2

theorem math_proof_problem (A B C a b c : ℝ) :
  problem1 A → 0 < A → A < real.pi → A = real.pi / 3 ∧
  problem2 a b c A B C := 
  sorry

end math_proof_problem_l18_18851


namespace standard_normal_symmetric_l18_18418

noncomputable def standard_normal_distribution : ProbabilityDistribution ℝ :=
{ density := λ x, (1 / (Mathlib.Real.pi.sqrt * 2)) * Mathlib.Real.exp (-(x^2)/2),
  density_integrable := sorry,
  density_nonneg := sorry }

theorem standard_normal_symmetric (p : ℝ) :
  (P (λ ξ, ξ ~ standard_normal_distribution) (Set.Ici 1) = p) →
  (P (λ ξ, ξ ~ standard_normal_distribution) (Set.Ioo (-1) 0) = (1 / 2) - p) := 
sorry

end standard_normal_symmetric_l18_18418


namespace find_a_l18_18002

theorem find_a (x y z a : ℝ) (k : ℝ) (h1 : x = 2 * k) (h2 : y = 3 * k) (h3 : z = 5 * k)
    (h4 : x + y + z = 100) (h5 : y = a * x - 10) : a = 2 :=
  sorry

end find_a_l18_18002


namespace find_x_l18_18849

theorem find_x (x y : ℝ) (h1 : y = 1) (h2 : 4 * x - 2 * y + 3 = 3 * x + 3 * y) : x = 2 :=
by
  sorry

end find_x_l18_18849


namespace asymptotes_and_foci_of_hyperbola_l18_18346

def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

theorem asymptotes_and_foci_of_hyperbola :
  (∀ x y : ℝ, hyperbola x y → y = x * (3 / 4) ∨ y = x * -(3 / 4)) ∧
  (∃ x y : ℝ, (x, y) = (15, 0) ∨ (x, y) = (-15, 0)) :=
by {
  -- prove these conditions here
  sorry 
}

end asymptotes_and_foci_of_hyperbola_l18_18346


namespace degrees_to_radians_750_l18_18057

theorem degrees_to_radians_750 (π : ℝ) (deg_750 : ℝ) 
  (h : 180 = π) : 
  750 * (π / 180) = 25 / 6 * π :=
by
  sorry

end degrees_to_radians_750_l18_18057


namespace minimum_dimes_l18_18793

-- Given amounts in dollars
def value_of_dimes (n : ℕ) : ℝ := 0.10 * n
def value_of_nickels : ℝ := 0.50
def value_of_one_dollar_bill : ℝ := 1.0
def value_of_four_tens : ℝ := 40.0
def price_of_scarf : ℝ := 42.85

-- Prove the total value of the money is at least the price of the scarf implies n >= 14
theorem minimum_dimes (n : ℕ) :
  value_of_four_tens + value_of_one_dollar_bill + value_of_nickels + value_of_dimes n ≥ price_of_scarf → n ≥ 14 :=
by
  sorry

end minimum_dimes_l18_18793


namespace shared_property_l18_18193

-- Definitions of the shapes
structure Parallelogram where
  sides_equal    : Bool -- Parallelograms have opposite sides equal but not necessarily all four.

structure Rectangle where
  sides_equal    : Bool -- Rectangles have opposite sides equal.
  diagonals_equal: Bool

structure Rhombus where
  sides_equal: Bool -- Rhombuses have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a rhombus are perpendicular.

structure Square where
  sides_equal: Bool -- Squares have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a square are perpendicular.
  diagonals_equal: Bool -- Diagonals of a square are equal in length.

-- Definitions of properties
def all_sides_equal (p1 p2 p3 p4 : Parallelogram) := p1.sides_equal ∧ p2.sides_equal ∧ p3.sides_equal ∧ p4.sides_equal
def diagonals_equal (r1 r2 r3 : Rectangle) (s1 s2 : Square) := r1.diagonals_equal ∧ r2.diagonals_equal ∧ s1.diagonals_equal ∧ s2.diagonals_equal
def diagonals_perpendicular (r1 : Rhombus) (s1 s2 : Square) := r1.diagonals_perpendicular ∧ s1.diagonals_perpendicular ∧ s2.diagonals_perpendicular
def diagonals_bisect_each_other (p1 p2 p3 p4 : Parallelogram) (r1 : Rectangle) (r2 : Rhombus) (s1 s2 : Square) := True -- All these shapes have diagonals that bisect each other.

-- The statement we need to prove
theorem shared_property (p1 p2 p3 p4 : Parallelogram) (r1 r2 : Rectangle) (r3 : Rhombus) (s1 s2 : Square) : 
  (diagonals_bisect_each_other p1 p2 p3 p4 r1 r3 s1 s2) :=
by
  sorry

end shared_property_l18_18193


namespace identity_is_only_sum_free_preserving_surjection_l18_18336

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, f m = n

def is_sum_free (A : Set ℕ) : Prop :=
  ∀ x y : ℕ, x ∈ A → y ∈ A → x + y ∉ A

noncomputable def identity_function_property : Prop :=
  ∀ f : ℕ → ℕ, is_surjective f →
  (∀ A : Set ℕ, is_sum_free A → is_sum_free (Set.image f A)) →
  ∀ n : ℕ, f n = n

theorem identity_is_only_sum_free_preserving_surjection : identity_function_property := sorry

end identity_is_only_sum_free_preserving_surjection_l18_18336


namespace macaroon_count_l18_18242

def baked_red_macaroons : ℕ := 50
def baked_green_macaroons : ℕ := 40
def ate_green_macaroons : ℕ := 15
def ate_red_macaroons := 2 * ate_green_macaroons

def remaining_macaroons : ℕ := (baked_red_macaroons - ate_red_macaroons) + (baked_green_macaroons - ate_green_macaroons)

theorem macaroon_count : remaining_macaroons = 45 := by
  sorry

end macaroon_count_l18_18242


namespace a_must_not_be_zero_l18_18434

theorem a_must_not_be_zero (a b c d : ℝ) (h₁ : a / b < -3 * (c / d)) (h₂ : b ≠ 0) (h₃ : d ≠ 0) (h₄ : c = 2 * a) : a ≠ 0 :=
sorry

end a_must_not_be_zero_l18_18434


namespace min_S_l18_18540

-- Define the arithmetic sequence
def a (n : ℕ) (a1 d : ℤ) : ℤ :=
  a1 + (n - 1) * d

-- Define the sum of the first n terms
def S (n : ℕ) (a1 : ℤ) (d : ℤ) : ℤ :=
  (n * (a1 + a n a1 d)) / 2

-- Conditions
def a4 : ℤ := -15
def d : ℤ := 3

-- Found a1 from a4 and d
def a1 : ℤ := -24

-- Theorem stating the minimum value of the sum
theorem min_S : ∃ n, S n a1 d = -108 :=
  sorry

end min_S_l18_18540


namespace largest_angle_of_convex_hexagon_l18_18895

noncomputable def largest_angle (angles : Fin 6 → ℝ) (consecutive : ∀ i : Fin 5, angles i + 1 = angles (i + 1)) (sum_eq_720 : (∑ i, angles i) = 720) : ℝ :=
  sorry

theorem largest_angle_of_convex_hexagon (angles : Fin 6 → ℝ) (consecutive : ∀ i : Fin 5, angles i + 1 = angles (i + 1)) (sum_eq_720 : (∑ i, angles i) = 720) :
  largest_angle angles consecutive sum_eq_720 = 122.5 :=
  sorry

end largest_angle_of_convex_hexagon_l18_18895


namespace solve_for_x_l18_18463

theorem solve_for_x (x : ℚ) (h : 3 / x - 3 / x / (9 / x) = 0.5) : x = 6 / 5 :=
sorry

end solve_for_x_l18_18463


namespace inequality_solution_l18_18573

theorem inequality_solution {x : ℝ} :
  (12 * x^2 + 24 * x - 75) / ((3 * x - 5) * (x + 5)) < 4 ↔ -5 < x ∧ x < 5 / 3 := by
  sorry

end inequality_solution_l18_18573


namespace non_negative_solution_count_positive_solution_count_l18_18646

open Nat

-- Problem 1: Non-negative integer solutions
theorem non_negative_solution_count (N n : ℕ) (hN : N ≥ 1) (hn : n ≥ 1) :
    (∑ (i : Fin N), n.toNat) ≤ n → (Nat.choose (n + N) n) = (Nat.coe choose n n hN hn) :=
sorry

-- Problem 2: Positive integer solutions
theorem positive_solution_count (N n : ℕ) (hN : N ≥ 1) (hn : n ≥ 1) :
    (∑ (i : Fin N), (n.toNat + 1)) ≤ n → (Nat.choose (n - 1) (N - 1)) = (Nat.coe choose n n hn hN) :=
sorry

end non_negative_solution_count_positive_solution_count_l18_18646


namespace solve_equation_l18_18590

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l18_18590


namespace geometric_sequence_div_sum_l18_18564

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable def S (n : ℕ) : ℝ := sorry

theorem geometric_sequence_div_sum 
  (h₁ : S 3 = (1 - (2 : ℝ) ^ 3) / (1 - (2 : ℝ) ^ 2) * a 1)
  (h₂ : S 2 = (1 - (2 : ℝ) ^ 2) / (1 - 2) * a 1)
  (h₃ : 8 * a 2 = a 5) : 
  S 3 / S 2 = 7 / 3 := 
by
  sorry

end geometric_sequence_div_sum_l18_18564


namespace six_digit_number_condition_l18_18860

theorem six_digit_number_condition :
  ∃ A B : ℕ, 100 ≤ A ∧ A < 1000 ∧ 100 ≤ B ∧ B < 1000 ∧
            1000 * B + A = 6 * (1000 * A + B) :=
by
  sorry

end six_digit_number_condition_l18_18860


namespace nines_appear_600_times_l18_18112

-- Define a function to count the number of nines in a single digit place
def count_nines_in_place (place_value : Nat) (range_start : Nat) (range_end : Nat) : Nat :=
  (range_end - range_start + 1) / 10 * place_value

-- Define a function to count the total number of nines in all places from 1 to 1000
def total_nines_1_to_1000 : Nat :=
  let counts_per_place := 3 * count_nines_in_place 100 1 1000
  counts_per_place

theorem nines_appear_600_times :
  total_nines_1_to_1000 = 600 :=
by
  -- Using specific step counts calculated in the problem
  have units_place := count_nines_in_place 100 1 1000
  have tens_place := units_place
  have hundreds_place := units_place
  have thousands_place := 0
  have total_nines := units_place + tens_place + hundreds_place + thousands_place
  sorry

end nines_appear_600_times_l18_18112


namespace alpha_in_third_quadrant_l18_18258

theorem alpha_in_third_quadrant (α : ℝ)
 (h₁ : Real.tan (α - 3 * Real.pi) > 0)
 (h₂ : Real.sin (-α + Real.pi) < 0) :
 (0 < α % (2 * Real.pi) ∧ α % (2 * Real.pi) < Real.pi) := 
sorry

end alpha_in_third_quadrant_l18_18258


namespace count_integers_with_same_remainder_l18_18102

theorem count_integers_with_same_remainder (n : ℤ) : 
  (150 < n ∧ n < 250) ∧ 
  (∃ r : ℤ, 0 ≤ r ∧ r ≤ 6 ∧ ∃ a b : ℤ, n = 7 * a + r ∧ n = 9 * b + r) ↔ n = 7 :=
sorry

end count_integers_with_same_remainder_l18_18102


namespace arithmetic_sequence_abs_sum_l18_18379

theorem arithmetic_sequence_abs_sum :
  ∀ (a : ℕ → ℤ), (∀ n, a (n + 1) - a n = 2) → a 1 = -5 → 
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 18) :=
by
  sorry

end arithmetic_sequence_abs_sum_l18_18379


namespace olympic_high_school_amc10_l18_18496

/-- At Olympic High School, 2/5 of the freshmen and 4/5 of the sophomores took the AMC-10.
    Given that the number of freshmen and sophomore contestants was the same, there are twice as many freshmen as sophomores. -/
theorem olympic_high_school_amc10 (f s : ℕ) (hf : f > 0) (hs : s > 0)
  (contest_equal : (2 / 5 : ℚ)*f = (4 / 5 : ℚ)*s) : f = 2 * s :=
by
  sorry

end olympic_high_school_amc10_l18_18496


namespace inequality_proof_l18_18696

theorem inequality_proof (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = x^2 + 3 * x + 2) →
  a > 0 →
  b > 0 →
  b ≤ a / 7 →
  (∀ x : ℝ, |x + 2| < b → |f x + 4| < a) :=
by
  sorry

end inequality_proof_l18_18696


namespace count_five_digit_multiples_of_five_l18_18093

theorem count_five_digit_multiples_of_five : 
  ∃ (n : ℕ), n = 18000 ∧ (∀ x, 10000 ≤ x ∧ x ≤ 99999 ∧ x % 5 = 0 ↔ ∃ k, 10000 ≤ 5 * k ∧ 5 * k ≤ 99999) :=
by
  sorry

end count_five_digit_multiples_of_five_l18_18093


namespace sum_of_positive_integers_l18_18760

theorem sum_of_positive_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 272) : x + y = 32 := 
by 
  sorry

end sum_of_positive_integers_l18_18760


namespace smallest_k_divides_ab_l18_18722

theorem smallest_k_divides_ab (S : Finset ℕ) (hS : S = Finset.range 51)
  (k : ℕ) : (∀ T : Finset ℕ, T ⊆ S → T.card = k → ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b)) ↔ k = 39 :=
by
  sorry

end smallest_k_divides_ab_l18_18722


namespace probability_not_expired_l18_18617

theorem probability_not_expired (total_bottles expired_bottles not_expired_bottles : ℕ) 
  (h1 : total_bottles = 5) 
  (h2 : expired_bottles = 1) 
  (h3 : not_expired_bottles = total_bottles - expired_bottles) :
  (not_expired_bottles / total_bottles : ℚ) = 4 / 5 := 
by
  sorry

end probability_not_expired_l18_18617


namespace lowest_possible_sale_price_is_30_percent_l18_18950

noncomputable def list_price : ℝ := 80
noncomputable def discount_factor : ℝ := 0.5
noncomputable def additional_discount : ℝ := 0.2
noncomputable def lowest_price := (list_price - discount_factor * list_price) - additional_discount * list_price
noncomputable def percent_of_list_price := (lowest_price / list_price) * 100

theorem lowest_possible_sale_price_is_30_percent :
  percent_of_list_price = 30 := by
  sorry

end lowest_possible_sale_price_is_30_percent_l18_18950


namespace combinations_with_common_subjects_l18_18404

-- Conditions and known facts
def subjects : Finset String := {"politics", "history", "geography", "physics", "chemistry", "biology", "technology"}
def personA_must_choose : Finset String := {"physics", "politics"}
def personB_cannot_choose : String := "technology"
def total_combinations : Nat := Nat.choose 7 3
def valid_combinations : Nat := Nat.choose 5 1 * Nat.choose 6 3
def non_common_subject_combinations : Nat := 4 + 4

-- We need to prove this statement
theorem combinations_with_common_subjects : valid_combinations - non_common_subject_combinations = 92 := by
  sorry

end combinations_with_common_subjects_l18_18404


namespace pen_average_price_l18_18181

theorem pen_average_price (pens_purchased pencils_purchased : ℕ) (total_cost pencil_avg_price : ℝ)
  (H0 : pens_purchased = 30) (H1 : pencils_purchased = 75) 
  (H2 : total_cost = 690) (H3 : pencil_avg_price = 2) :
  (total_cost - (pencils_purchased * pencil_avg_price)) / pens_purchased = 18 :=
by
  rw [H0, H1, H2, H3]
  sorry

end pen_average_price_l18_18181


namespace num_satisfying_n_conditions_l18_18100

theorem num_satisfying_n_conditions :
  let count := {n : ℤ | 150 < n ∧ n < 250 ∧ (n % 7 = n % 9) }.toFinset.card
  count = 7 :=
by
  sorry

end num_satisfying_n_conditions_l18_18100


namespace coffee_vacation_days_l18_18884

theorem coffee_vacation_days 
  (pods_per_day : ℕ := 3)
  (pods_per_box : ℕ := 30)
  (box_cost : ℝ := 8.00)
  (total_spent : ℝ := 32) :
  (total_spent / box_cost) * pods_per_box / pods_per_day = 40 := 
by 
  sorry

end coffee_vacation_days_l18_18884


namespace first_term_geometric_l18_18163

-- Definition: geometric sequence properties
variables (a r : ℚ) -- sequence terms are rational numbers
variables (n : ℕ)

-- Conditions: fifth and sixth terms of a geometric sequence
def fifth_term_geometric (a r : ℚ) : ℚ := a * r^4
def sixth_term_geometric (a r : ℚ) : ℚ := a * r^5

-- Proof: given conditions
theorem first_term_geometric (a r : ℚ) (h1 : fifth_term_geometric a r = 48) 
  (h2 : sixth_term_geometric a r = 72) : a = 768 / 81 :=
by {
  sorry
}

end first_term_geometric_l18_18163


namespace book_price_l18_18534

theorem book_price (x : ℕ) (h1 : x - 1 = 1 + (x - 1)) : x = 2 :=
by
  sorry

end book_price_l18_18534


namespace solve_for_q_l18_18393

-- Define the conditions
variables (p q : ℝ)
axiom condition1 : 3 * p + 4 * q = 8
axiom condition2 : 4 * p + 3 * q = 13

-- State the goal to prove q = -1
theorem solve_for_q : q = -1 :=
by
  sorry

end solve_for_q_l18_18393


namespace part_I_part_II_l18_18525

-- Define the function f
def f (x a : ℝ) := |x - a| + |x - 2|

-- Statement for part (I)
theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ a) : a ≥ 1 := sorry

-- Statement for part (II)
theorem part_II (m n p : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : p > 0) (h4 : m + 2 * n + 3 * p = 1) : 
  (3 / m) + (2 / n) + (1 / p) ≥ 6 + 2 * Real.sqrt 6 + 2 * Real.sqrt 2 := sorry

end part_I_part_II_l18_18525


namespace symmetrical_character_l18_18769

def is_symmetrical (char : String) : Prop := 
  sorry  -- Here the definition for symmetry will be elaborated

theorem symmetrical_character : 
  let A : String := "坡"
  let B : String := "上"
  let C : String := "草"
  let D : String := "原"
  is_symmetrical C := 
  sorry

end symmetrical_character_l18_18769


namespace finalCostCalculation_l18_18756

-- Define the inputs
def tireRepairCost : ℝ := 7
def salesTaxPerTire : ℝ := 0.50
def numberOfTires : ℕ := 4

-- The total cost should be $30
theorem finalCostCalculation : 
  let repairTotal := tireRepairCost * numberOfTires
  let salesTaxTotal := salesTaxPerTire * numberOfTires
  repairTotal + salesTaxTotal = 30 := 
by {
  sorry
}

end finalCostCalculation_l18_18756


namespace find_m_plus_n_l18_18226

def m_n_sum (p : ℚ) : ℕ :=
  let m := p.num.natAbs
  let n := p.denom
  m + n

noncomputable def prob_3x3_red_square_free : ℚ :=
  let totalWays := 2^16
  let redSquareWays := totalWays - 511
  redSquareWays / totalWays

theorem find_m_plus_n :
  m_n_sum prob_3x3_red_square_free = 130561 :=
by
  sorry

end find_m_plus_n_l18_18226


namespace monotonicity_and_extremum_of_f_l18_18088

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem monotonicity_and_extremum_of_f :
  (∀ x, 1 < x → ∀ y, x < y → f x < f y) ∧
  (∀ x, 0 < x → x < 1 → ∀ y, x < y → y < 1 → f x > f y) ∧
  (f 1 = -1) :=
by
  sorry

end monotonicity_and_extremum_of_f_l18_18088


namespace Marie_speed_l18_18287

theorem Marie_speed (distance time : ℕ) (h1 : distance = 372) (h2 : time = 31) : distance / time = 12 :=
by
  have h3 : distance = 372 := h1
  have h4 : time = 31 := h2
  sorry

end Marie_speed_l18_18287


namespace hamburgers_sold_last_week_l18_18944

theorem hamburgers_sold_last_week (avg_hamburgers_per_day : ℕ) (days_in_week : ℕ) 
    (h_avg : avg_hamburgers_per_day = 9) (h_days : days_in_week = 7) : 
    avg_hamburgers_per_day * days_in_week = 63 :=
by
  -- Avg hamburgers per day times number of days
  sorry

end hamburgers_sold_last_week_l18_18944


namespace determinant_of_non_right_triangle_l18_18279

theorem determinant_of_non_right_triangle (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
(h_sum_ABC : A + B + C = π) :
  Matrix.det ![
    ![2 * Real.sin A, 1, 1],
    ![1, 2 * Real.sin B, 1],
    ![1, 1, 2 * Real.sin C]
  ] = 2 := by
  sorry

end determinant_of_non_right_triangle_l18_18279


namespace pete_backwards_speed_l18_18426

variable (speed_pete_hands : ℕ) (speed_tracy_cartwheel : ℕ) (speed_susan_walk : ℕ) (speed_pete_backwards : ℕ)

axiom pete_hands_speed : speed_pete_hands = 2
axiom pete_hands_speed_quarter_tracy_cartwheel : speed_pete_hands = speed_tracy_cartwheel / 4
axiom tracy_cartwheel_twice_susan_walk : speed_tracy_cartwheel = 2 * speed_susan_walk
axiom pete_backwards_three_times_susan_walk : speed_pete_backwards = 3 * speed_susan_walk

theorem pete_backwards_speed : 
  speed_pete_backwards = 12 :=
by
  sorry

end pete_backwards_speed_l18_18426


namespace abc_sitting_together_probability_l18_18021

-- Definitions and conditions
def favourable_arrangements : ℕ := Nat.factorial 6 * Nat.factorial 3
def total_arrangements : ℕ := Nat.factorial 8
def probability : ℚ := favourable_arrangements / total_arrangements

-- Theorem: Prove that the probability of a, b, c sitting together is 1/9.375
theorem abc_sitting_together_probability : probability = 1 / 9.375 := by
  unfold favourable_arrangements
  unfold total_arrangements
  unfold probability
  sorry

end abc_sitting_together_probability_l18_18021


namespace length_of_DE_l18_18437

-- Given conditions
variables (AB DE : ℝ) (area_projected area_ABC : ℝ)

-- Hypotheses
def base_length (AB : ℝ) : Prop := AB = 15
def projected_area_ratio (area_projected area_ABC : ℝ) : Prop := area_projected = 0.25 * area_ABC
def parallel_lines (DE AB : ℝ) : Prop := ∀ x : ℝ, DE = 0.5 * AB

-- The theorem to prove
theorem length_of_DE (h1 : base_length AB) (h2 : projected_area_ratio area_projected area_ABC) (h3 : parallel_lines DE AB) : DE = 7.5 :=
by
  sorry

end length_of_DE_l18_18437


namespace min_sets_bound_l18_18556

theorem min_sets_bound (A : Type) (n k : ℕ) (S : Finset (Finset A))
  (h₁ : S.card = k)
  (h₂ : ∀ x y : A, x ≠ y → ∃ B ∈ S, (x ∈ B ∧ y ∉ B) ∨ (y ∈ B ∧ x ∉ B)) :
  2^k ≥ n :=
sorry

end min_sets_bound_l18_18556


namespace number_of_six_digit_with_sum_51_l18_18834

open Finset

/-- A digit is a number between 0 and 9 inclusive.-/
def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

/-- Friendly notation for digit sums -/
def digit_sum (n : Fin 6 → ℕ) : ℕ :=
  (Finset.univ.sum n)

def is_six_digit_with_sum_51 (n : Fin 6 → ℕ) : Prop :=
  (∀ i, is_digit (n i)) ∧ (digit_sum n = 51)

/-- There are exactly 56 six-digit numbers such that the sum of their digits is 51. -/
theorem number_of_six_digit_with_sum_51 : 
  card {n : Fin 6 → ℕ // is_six_digit_with_sum_51 n} = 56 :=
by
  sorry

end number_of_six_digit_with_sum_51_l18_18834


namespace car_price_is_5_l18_18520

variable (numCars : ℕ) (totalEarnings legoCost carCost : ℕ)

-- Conditions
axiom h1 : numCars = 3
axiom h2 : totalEarnings = 45
axiom h3 : legoCost = 30
axiom h4 : totalEarnings - legoCost = 15
axiom h5 : (totalEarnings - legoCost) / numCars = carCost

-- The proof problem statement
theorem car_price_is_5 : carCost = 5 :=
  by
    -- Here the proof steps would be filled in, but are not required for this task.
    sorry

end car_price_is_5_l18_18520


namespace four_roots_sum_eq_neg8_l18_18058

def op (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := op x 2

theorem four_roots_sum_eq_neg8 :
  ∃ (x1 x2 x3 x4 : ℝ), 
  (x1 ≠ -2) ∧ (x2 ≠ -2) ∧ (x3 ≠ -2) ∧ (x4 ≠ -2) ∧
  (f x1 = Real.log (abs (x1 + 2))) ∧ 
  (f x2 = Real.log (abs (x2 + 2))) ∧ 
  (f x3 = Real.log (abs (x3 + 2))) ∧ 
  (f x4 = Real.log (abs (x4 + 2))) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ 
  x2 ≠ x3 ∧ x2 ≠ x4 ∧ 
  x3 ≠ x4 ∧ 
  x1 + x2 + x3 + x4 = -8 :=
by 
  sorry

end four_roots_sum_eq_neg8_l18_18058


namespace original_time_to_complete_book_l18_18171

-- Define the problem based on the given conditions
variables (n : ℕ) (T : ℚ)

-- Define the conditions
def condition1 : Prop := 
  ∃ (n T : ℚ), 
  n / T = (n + 3) / (0.75 * T) ∧
  n / T = (n - 3) / (T + 5 / 6)

-- State the theorem with the correct answer
theorem original_time_to_complete_book : condition1 → T = 5 / 3 :=
by sorry

end original_time_to_complete_book_l18_18171


namespace sum_m_n_l18_18234

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l18_18234


namespace arithmetic_expression_evaluation_l18_18500

theorem arithmetic_expression_evaluation : 5 * (9 / 3) + 7 * 4 - 36 / 4 = 34 := 
by
  sorry

end arithmetic_expression_evaluation_l18_18500


namespace find_y_l18_18515

theorem find_y (y: ℕ) (h1: y > 0) (h2: y ≤ 100)
  (h3: (43 + 69 + 87 + y + y) / 5 = 2 * y): 
  y = 25 :=
sorry

end find_y_l18_18515


namespace min_sum_y1_y2_l18_18815

theorem min_sum_y1_y2 (y : ℕ → ℕ) (h_seq : ∀ n ≥ 1, y (n+2) = (y n + 2013)/(1 + y (n+1))) : 
  ∃ y1 y2, y1 + y2 = 94 ∧ (∀ n, y n > 0) ∧ (y 1 = y1) ∧ (y 2 = y2) := 
sorry

end min_sum_y1_y2_l18_18815


namespace evaluate_expression_l18_18062

theorem evaluate_expression : (18 * 3 + 6) / (6 - 3) = 20 := by
  sorry

end evaluate_expression_l18_18062


namespace marbles_given_l18_18207

theorem marbles_given (initial remaining given : ℕ) (h_initial : initial = 143) (h_remaining : remaining = 70) :
    given = initial - remaining → given = 73 :=
by
  intros
  sorry

end marbles_given_l18_18207


namespace simplify_expression_l18_18864

noncomputable def p (x a b c : ℝ) :=
  (x + 2 * a)^2 / ((a - b) * (a - c)) +
  (x + 2 * b)^2 / ((b - a) * (b - c)) +
  (x + 2 * c)^2 / ((c - a) * (c - b))

theorem simplify_expression (a b c x : ℝ) (h : a ≠ b ∧ a ≠ c ∧ b ≠ c) :
  p x a b c = 4 :=
by
  sorry

end simplify_expression_l18_18864


namespace combined_weight_of_candles_l18_18802

theorem combined_weight_of_candles :
  (∀ (candles: ℕ), ∀ (weight_per_candle: ℕ),
    (weight_per_candle = 8 + 1) →
    (candles = 10 - 3) →
    (candles * weight_per_candle = 63)) :=
by
  intros
  sorry

end combined_weight_of_candles_l18_18802


namespace alcohol_solution_contradiction_l18_18535

theorem alcohol_solution_contradiction (initial_volume : ℕ) (added_water : ℕ) 
                                        (final_volume : ℕ) (final_concentration : ℕ) 
                                        (initial_concentration : ℕ) : 
                                        initial_volume = 75 → added_water = 50 → 
                                        final_volume = initial_volume + added_water → 
                                        final_concentration = 45 → 
                                        ¬ (initial_concentration * initial_volume = final_concentration * final_volume) :=
by 
  intro h_initial_volume h_added_water h_final_volume h_final_concentration
  sorry

end alcohol_solution_contradiction_l18_18535


namespace probability_two_red_marbles_drawn_l18_18042

/-- A jar contains two red marbles, three green marbles, and ten white marbles and no other marbles.
Two marbles are randomly drawn from this jar without replacement. -/
theorem probability_two_red_marbles_drawn (total_marbles red_marbles green_marbles white_marbles : ℕ)
    (draw_without_replacement : Bool) :
    total_marbles = 15 ∧ red_marbles = 2 ∧ green_marbles = 3 ∧ white_marbles = 10 ∧ draw_without_replacement = true →
    (2 / 15) * (1 / 14) = 1 / 105 :=
by
  intro h
  sorry

end probability_two_red_marbles_drawn_l18_18042


namespace binom_9_5_eq_126_l18_18978

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l18_18978


namespace mangoes_ratio_l18_18786

theorem mangoes_ratio (a d_a : ℕ)
  (h1 : a = 60)
  (h2 : a + d_a = 75) : a / (75 - a) = 4 := by
  sorry

end mangoes_ratio_l18_18786


namespace train_crossing_time_l18_18189

theorem train_crossing_time
  (length_train : ℕ)
  (speed_train_kmph : ℕ)
  (total_length : ℕ)
  (htotal_length : total_length = 225)
  (hlength_train : length_train = 150)
  (hspeed_train_kmph : speed_train_kmph = 45) : 
  (total_length / (speed_train_kmph * 1000 / 3600)) = 18 := by 
  sorry

end train_crossing_time_l18_18189


namespace grass_coverage_day_l18_18940

theorem grass_coverage_day (coverage : ℕ → ℚ) : 
  (∀ n : ℕ, coverage (n + 1) = 2 * coverage n) → 
  coverage 24 = 1 → 
  coverage 21 = 1 / 8 := 
by
  sorry

end grass_coverage_day_l18_18940


namespace first_tier_tax_rate_l18_18504

theorem first_tier_tax_rate (price : ℕ) (total_tax : ℕ) (tier1_limit : ℕ) (tier2_rate : ℝ) (tier1_tax_rate : ℝ) :
  price = 18000 →
  total_tax = 1950 →
  tier1_limit = 11000 →
  tier2_rate = 0.09 →
  ((price - tier1_limit) * tier2_rate + tier1_tax_rate * tier1_limit = total_tax) →
  tier1_tax_rate = 0.12 :=
by
  intros hprice htotal htier1 hrate htax_eq
  sorry

end first_tier_tax_rate_l18_18504


namespace parabola_y_intercepts_l18_18830

theorem parabola_y_intercepts : ∃ y1 y2 : ℝ, (3 * y1^2 - 6 * y1 + 2 = 0) ∧ (3 * y2^2 - 6 * y2 + 2 = 0) ∧ (y1 ≠ y2) :=
by 
  sorry

end parabola_y_intercepts_l18_18830


namespace cost_per_item_l18_18545

theorem cost_per_item (total_profit : ℝ) (total_customers : ℕ) (purchase_percentage : ℝ) (pays_advertising : ℝ)
    (H1: total_profit = 1000)
    (H2: total_customers = 100)
    (H3: purchase_percentage = 0.80)
    (H4: pays_advertising = 1000)
    : (total_profit / (total_customers * purchase_percentage)) = 12.50 :=
by
  sorry

end cost_per_item_l18_18545


namespace sufficient_but_not_necessary_l18_18868

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧ ∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_l18_18868


namespace simplify_expression_l18_18369

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 3) :
  (3 * x ^ 2 - 2 * x - 4) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3)) =
  3 * (x ^ 2 - x - 3) / ((x + 2) * (x - 3)) :=
by
  sorry

end simplify_expression_l18_18369


namespace total_worth_of_stock_l18_18948

noncomputable def shop_equation (X : ℝ) : Prop :=
  0.04 * X - 0.02 * X = 400

theorem total_worth_of_stock :
  ∃ (X : ℝ), shop_equation X ∧ X = 20000 :=
by
  use 20000
  have h : shop_equation 20000 := by
    unfold shop_equation
    norm_num
  exact ⟨h, rfl⟩

end total_worth_of_stock_l18_18948


namespace quadratic_inequality_l18_18846

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l18_18846


namespace graph_single_point_l18_18302

theorem graph_single_point (d : ℝ) :
  (∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0 -> (x = -1 ∧ y = 3)) ↔ d = 12 :=
by 
  sorry

end graph_single_point_l18_18302


namespace polar_to_cartesian_l18_18406

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.sin θ) : 
  ∀ (x y : ℝ) (h₁ : x = ρ * Real.cos θ) (h₂ : y = ρ * Real.sin θ), 
    x^2 + (y - 1)^2 = 1 :=
by
  sorry

end polar_to_cartesian_l18_18406


namespace biology_marks_l18_18208

theorem biology_marks (E M P C: ℝ) (A: ℝ) (N: ℕ) 
  (hE: E = 96) (hM: M = 98) (hP: P = 99) (hC: C = 100) (hA: A = 98.2) (hN: N = 5):
  (E + M + P + C + B) / N = A → B = 98 :=
by
  intro h
  sorry

end biology_marks_l18_18208


namespace problem1_l18_18028

theorem problem1 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : 
  (x * y = 5) ∧ ((x - y)^2 = 5) :=
by
  sorry

end problem1_l18_18028


namespace distance_last_day_l18_18436

theorem distance_last_day
  (total_distance : ℕ)
  (days : ℕ)
  (initial_distance : ℕ)
  (common_ratio : ℚ)
  (sum_geometric : initial_distance * (1 - common_ratio^days) / (1 - common_ratio) = total_distance) :
  total_distance = 378 → days = 6 → common_ratio = 1/2 → 
  initial_distance = 192 → initial_distance * common_ratio^(days - 1) = 6 := 
by
  intros h1 h2 h3 h4
  sorry

end distance_last_day_l18_18436


namespace perfect_square_l18_18905

variable (n k l : ℕ)

theorem perfect_square (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m ^ 2 := by
  have h1 : (2 * l - n - k) * (2 * l - n + k) = (2 * l - n) ^ 2 - k ^ 2 := by sorry
  have h2 : k ^ 2 = 2 * l ^ 2 - n ^ 2 := by sorry
  have h3 : (2 * l - n) ^ 2 - (2 * l ^ 2 - n ^ 2) = 2 * (l - n) ^ 2 := by sorry
  have h4 : (2 * (l - n) ^ 2) / 2 = (l - n) ^ 2 := by sorry
  use (l - n)
  rw [h4]
  sorry

end perfect_square_l18_18905


namespace quadratic_root_iff_l18_18624

theorem quadratic_root_iff (a b c : ℝ) :
  (∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0) ↔ (a + b + c = 0) :=
by
  sorry

end quadratic_root_iff_l18_18624


namespace min_value_expression_l18_18246

theorem min_value_expression (x y z : ℝ) (h : x - 2 * y + 2 * z = 5) : (x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2 ≥ 36 :=
by
  sorry

end min_value_expression_l18_18246


namespace total_snakes_seen_l18_18420

-- Define the number of snakes in each breeding ball
def snakes_in_first_breeding_ball : Nat := 15
def snakes_in_second_breeding_ball : Nat := 20
def snakes_in_third_breeding_ball : Nat := 25
def snakes_in_fourth_breeding_ball : Nat := 30
def snakes_in_fifth_breeding_ball : Nat := 35
def snakes_in_sixth_breeding_ball : Nat := 40
def snakes_in_seventh_breeding_ball : Nat := 45

-- Define the number of pairs of extra snakes
def extra_pairs_of_snakes : Nat := 23

-- Define the total number of snakes observed
def total_snakes_observed : Nat :=
  snakes_in_first_breeding_ball +
  snakes_in_second_breeding_ball +
  snakes_in_third_breeding_ball +
  snakes_in_fourth_breeding_ball +
  snakes_in_fifth_breeding_ball +
  snakes_in_sixth_breeding_ball +
  snakes_in_seventh_breeding_ball +
  (extra_pairs_of_snakes * 2)

theorem total_snakes_seen : total_snakes_observed = 256 := by
  sorry

end total_snakes_seen_l18_18420


namespace number_is_2250_l18_18912

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l18_18912


namespace roots_sum_prod_eq_l18_18582

theorem roots_sum_prod_eq (p q : ℤ) (h1 : p / 3 = 9) (h2 : q / 3 = 20) : p + q = 87 :=
by
  sorry

end roots_sum_prod_eq_l18_18582


namespace employees_both_fraction_l18_18201

-- Define the total number of employees as a variable
variable {x : ℚ}

-- Define the fractions representing employees with cell phones, pagers, and neither
def cell_phone_fraction : ℚ := 2/3
def pager_fraction : ℚ := 2/5
def neither_fraction : ℚ := 1/3

-- Define a fraction representing the employees with both devices
def both_fraction (x : ℚ) := cell_phone_fraction + pager_fraction - 1 + neither_fraction

-- Statement to prove
theorem employees_both_fraction : both_fraction x = 2/5 :=
by
  sorry

end employees_both_fraction_l18_18201


namespace valid_parametrizations_l18_18891

-- Definitions for the given points and directions
def pointA := (0, 4)
def dirA := (3, -1)

def pointB := (4/3, 0)
def dirB := (1, -3)

def pointC := (-2, 10)
def dirC := (-3, 9)

-- Line equation definition
def line (x y : ℝ) : Prop := y = -3 * x + 4

-- Proof statement
theorem valid_parametrizations :
  (line pointB.1 pointB.2 ∧ dirB.2 = -3 * dirB.1) ∧
  (line pointC.1 pointC.2 ∧ dirC.2 / dirC.1 = 3) :=
by
  sorry

end valid_parametrizations_l18_18891


namespace A_alone_completes_one_work_in_32_days_l18_18394

def amount_of_work_per_day_by_B : ℝ := sorry
def amount_of_work_per_day_by_A : ℝ := 3 * amount_of_work_per_day_by_B
def total_work : ℝ := (amount_of_work_per_day_by_A + amount_of_work_per_day_by_B) * 24

theorem A_alone_completes_one_work_in_32_days :
  total_work = amount_of_work_per_day_by_A * 32 :=
by
  sorry

end A_alone_completes_one_work_in_32_days_l18_18394


namespace complement_of_A_in_U_l18_18390

-- Given definitions from the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}

-- The theorem to be proven
theorem complement_of_A_in_U : U \ A = {1, 3, 6, 7} :=
by
  sorry

end complement_of_A_in_U_l18_18390


namespace ratio_hours_per_day_l18_18494

theorem ratio_hours_per_day 
  (h₁ : ∀ h : ℕ, h * 30 = 1200 + (h - 40) * 45 → 40 ≤ h ∧ 6 * 3 ≤ 40)
  (h₂ : 6 * 3 + (x - 6 * 3) / 2 = 24)
  (h₃ : x = 1290) :
  (24 / 2) / 6 = 2 := 
by
  sorry

end ratio_hours_per_day_l18_18494


namespace grace_clyde_ratio_l18_18794

theorem grace_clyde_ratio (C G : ℕ) (h1 : G = C + 35) (h2 : G = 40) : G / C = 8 :=
by sorry

end grace_clyde_ratio_l18_18794


namespace sum_of_roots_l18_18281

theorem sum_of_roots 
  (a b c : ℝ)
  (h1 : 1^2 + a * 1 + 2 = 0)
  (h2 : (∀ x : ℝ, x^2 + 5 * x + c = 0 → (x = a ∨ x = b))) :
  a + b + c = 1 :=
by
  sorry

end sum_of_roots_l18_18281


namespace bounded_fx_range_a_l18_18344

-- Part (1)
theorem bounded_fx :
  ∃ M > 0, ∀ x ∈ Set.Icc (-(1/2):ℝ) (1/2), abs (x / (x + 1)) ≤ M :=
by
  sorry

-- Part (2)
theorem range_a (a : ℝ) :
  (∀ x ≥ 0, abs (1 + a * (1/2)^x + (1/4)^x) ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by
  sorry

end bounded_fx_range_a_l18_18344


namespace a2020_lt_inv_2020_l18_18148

theorem a2020_lt_inv_2020 (a : ℕ → ℝ) (ha0 : a 0 > 0) 
    (h_rec : ∀ n, a (n + 1) = a n / Real.sqrt (1 + 2020 * a n ^ 2)) :
    a 2020 < 1 / 2020 :=
sorry

end a2020_lt_inv_2020_l18_18148


namespace ceil_minus_floor_eq_one_implies_ceil_minus_y_l18_18721

noncomputable def fractional_part (y : ℝ) : ℝ := y - ⌊y⌋

theorem ceil_minus_floor_eq_one_implies_ceil_minus_y (y : ℝ) (h : ⌈y⌉ - ⌊y⌋ = 1) : ⌈y⌉ - y = 1 - fractional_part y :=
by
  sorry

end ceil_minus_floor_eq_one_implies_ceil_minus_y_l18_18721


namespace sum_of_numbers_is_919_l18_18734

-- Problem Conditions
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def is_three_digit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999
def satisfies_equation (x y : ℕ) : Prop := 1000 * x + y = 11 * x * y

-- Main Statement
theorem sum_of_numbers_is_919 (x y : ℕ) 
  (h1 : is_two_digit x) 
  (h2 : is_three_digit y) 
  (h3 : satisfies_equation x y) : 
  x + y = 919 := 
sorry

end sum_of_numbers_is_919_l18_18734


namespace aurelia_percentage_l18_18191

variables (P : ℝ)

theorem aurelia_percentage (h1 : 2000 + (P / 100) * 2000 = 3400) : 
  P = 70 :=
by
  sorry

end aurelia_percentage_l18_18191


namespace number_of_sweet_potatoes_sold_to_mrs_adams_l18_18875

def sweet_potatoes_harvested := 80
def sweet_potatoes_sold_to_mr_lenon := 15
def sweet_potatoes_unsold := 45

def sweet_potatoes_sold_to_mrs_adams :=
  sweet_potatoes_harvested - sweet_potatoes_sold_to_mr_lenon - sweet_potatoes_unsold

theorem number_of_sweet_potatoes_sold_to_mrs_adams :
  sweet_potatoes_sold_to_mrs_adams = 20 := by
  sorry

end number_of_sweet_potatoes_sold_to_mrs_adams_l18_18875


namespace aston_comics_l18_18197

theorem aston_comics (total_pages_on_floor : ℕ) (pages_per_comic : ℕ) (untorn_comics_in_box : ℕ) :
  total_pages_on_floor = 150 →
  pages_per_comic = 25 →
  untorn_comics_in_box = 5 →
  (total_pages_on_floor / pages_per_comic + untorn_comics_in_box) = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end aston_comics_l18_18197


namespace betty_garden_total_plants_l18_18962

theorem betty_garden_total_plants (B O : ℕ) 
  (h1 : B = 5) 
  (h2 : O = 2 + 2 * B) : 
  B + O = 17 := by
  sorry

end betty_garden_total_plants_l18_18962


namespace binom_9_5_eq_126_l18_18976

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l18_18976


namespace solve_equation_l18_18432

theorem solve_equation : ∃ x : ℚ, 3 * (x - 2) = x - (2 * x - 1) ∧ x = 7/4 := by
  sorry

end solve_equation_l18_18432


namespace price_of_chips_l18_18442

theorem price_of_chips (P : ℝ) (h1 : 1.5 = 1.5) (h2 : 45 = 45) (h3 : 15 = 15) (h4 : 10 = 10) :
  15 * P + 10 * 1.5 = 45 → P = 2 :=
by
  sorry

end price_of_chips_l18_18442


namespace quintuplets_babies_l18_18358

theorem quintuplets_babies (a b c d : ℕ) 
  (h1 : d = 2 * c) 
  (h2 : c = 3 * b) 
  (h3 : b = 2 * a) 
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1200) : 
  5 * d = 18000 / 23 :=
by 
  sorry

end quintuplets_babies_l18_18358


namespace min_value_of_expr_l18_18692

theorem min_value_of_expr (x : ℝ) (h : x > 2) : ∃ y, (y = x + 4 / (x - 2)) ∧ y ≥ 6 :=
by
  sorry

end min_value_of_expr_l18_18692


namespace sand_weight_proof_l18_18628

-- Definitions for the given conditions
def side_length : ℕ := 40
def bag_weight : ℕ := 30
def area_per_bag : ℕ := 80

-- Total area of the sandbox
def total_area := side_length * side_length

-- Number of bags needed
def number_of_bags := total_area / area_per_bag

-- Total weight of sand needed
def total_weight := number_of_bags * bag_weight

-- The proof statement
theorem sand_weight_proof :
  total_weight = 600 :=
by
  sorry

end sand_weight_proof_l18_18628


namespace total_cost_of_concrete_blocks_l18_18140

theorem total_cost_of_concrete_blocks
  (sections : ℕ) (blocks_per_section : ℕ) (cost_per_block : ℕ)
  (h_sections : sections = 8)
  (h_blocks_per_section : blocks_per_section = 30)
  (h_cost_per_block : cost_per_block = 2) :
  sections * blocks_per_section * cost_per_block = 480 :=
by
  rw [h_sections, h_blocks_per_section, h_cost_per_block]
  sorry

end total_cost_of_concrete_blocks_l18_18140


namespace find_x_l18_18371

theorem find_x (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1)
  (geom_seq : (x - ⌊x⌋) * x = ⌊x⌋^2) : x = 1.618 :=
by
  sorry

end find_x_l18_18371


namespace lowest_possible_sale_price_is_30_percent_l18_18949

noncomputable def list_price : ℝ := 80
noncomputable def discount_factor : ℝ := 0.5
noncomputable def additional_discount : ℝ := 0.2
noncomputable def lowest_price := (list_price - discount_factor * list_price) - additional_discount * list_price
noncomputable def percent_of_list_price := (lowest_price / list_price) * 100

theorem lowest_possible_sale_price_is_30_percent :
  percent_of_list_price = 30 := by
  sorry

end lowest_possible_sale_price_is_30_percent_l18_18949


namespace soaking_time_l18_18499

theorem soaking_time (time_per_grass_stain : ℕ) (time_per_marinara_stain : ℕ) 
    (number_of_grass_stains : ℕ) (number_of_marinara_stains : ℕ) : 
    time_per_grass_stain = 4 ∧ time_per_marinara_stain = 7 ∧ 
    number_of_grass_stains = 3 ∧ number_of_marinara_stains = 1 →
    (time_per_grass_stain * number_of_grass_stains + time_per_marinara_stain * number_of_marinara_stains) = 19 :=
by
  sorry

end soaking_time_l18_18499


namespace wyatt_headmaster_duration_l18_18266

def Wyatt_start_month : Nat := 3 -- March
def Wyatt_break_start_month : Nat := 7 -- July
def Wyatt_break_end_month : Nat := 12 -- December
def Wyatt_end_year : Nat := 2011

def months_worked_before_break : Nat := Wyatt_break_start_month - Wyatt_start_month -- March to June (inclusive, hence -1)
def break_duration : Nat := 6
def months_worked_after_break : Nat := 12 -- January to December 2011

def total_months_worked : Nat := months_worked_before_break + months_worked_after_break
theorem wyatt_headmaster_duration : total_months_worked = 16 :=
by
  sorry

end wyatt_headmaster_duration_l18_18266


namespace range_of_fraction_l18_18747

theorem range_of_fraction : 
  ∀ y ∈ set.range (λ x : ℝ, (x + 3) / (x + 1)), 
  5 / 3 ≤ y ∧ y ≤ 3 := by
    sorry

end range_of_fraction_l18_18747


namespace total_chocolate_bars_l18_18041

theorem total_chocolate_bars :
  let num_large_boxes := 45
  let num_small_boxes_per_large_box := 36
  let num_chocolate_bars_per_small_box := 72
  num_large_boxes * num_small_boxes_per_large_box * num_chocolate_bars_per_small_box = 116640 :=
by
  sorry

end total_chocolate_bars_l18_18041


namespace zero_pow_2014_l18_18792

-- Define the condition that zero raised to any positive power is zero
def zero_pow_pos {n : ℕ} (h : 0 < n) : (0 : ℝ)^n = 0 := by
  sorry

-- Use this definition to prove the specific case of 0 ^ 2014 = 0
theorem zero_pow_2014 : (0 : ℝ)^(2014) = 0 := by
  have h : 0 < 2014 := by decide
  exact zero_pow_pos h

end zero_pow_2014_l18_18792


namespace inequality_correct_l18_18560

theorem inequality_correct (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1 / a) < (1 / b) :=
sorry

end inequality_correct_l18_18560


namespace quadratic_inequality_solution_l18_18843

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l18_18843


namespace final_cards_l18_18275

def initial_cards : ℝ := 47.0
def lost_cards : ℝ := 7.0

theorem final_cards : (initial_cards - lost_cards) = 40.0 :=
by
  sorry

end final_cards_l18_18275


namespace perfect_square_expression_l18_18900

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end perfect_square_expression_l18_18900


namespace quadratic_inequality_l18_18845

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l18_18845


namespace six_digit_palindromes_count_l18_18533

open Nat

theorem six_digit_palindromes_count :
  let digits := {d | 0 ≤ d ∧ d ≤ 9}
  let a_digits := {a | 1 ≤ a ∧ a ≤ 9}
  let b_digits := digits
  let c_digits := digits
  ∃ (total : ℕ), (∀ a ∈ a_digits, ∀ b ∈ b_digits, ∀ c ∈ c_digits, True) → total = 900 :=
by
  sorry

end six_digit_palindromes_count_l18_18533


namespace new_students_admitted_l18_18313

-- Definitions of the conditions
def original_students := 35
def increase_in_expenses := 42
def decrease_in_average_expense := 1
def original_expenditure := 420

-- Main statement: proving the number of new students admitted
theorem new_students_admitted : ∃ x : ℕ, 
  (original_expenditure + increase_in_expenses = 11 * (original_students + x)) ∧ 
  (x = 7) := 
sorry

end new_students_admitted_l18_18313


namespace sum_of_cubes_l18_18575

theorem sum_of_cubes {a b c : ℝ} (h1 : a + b + c = 5) (h2 : a * b + a * c + b * c = 7) (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 29 :=
by
  -- The proof part is intentionally left out.
  sorry

end sum_of_cubes_l18_18575


namespace binom_9_5_eq_126_l18_18974

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l18_18974


namespace perfect_square_a_value_l18_18694

theorem perfect_square_a_value (x y a : ℝ) :
  (∃ k : ℝ, x^2 + 2 * x * y + y^2 - a * (x + y) + 25 = k^2) →
  a = 10 ∨ a = -10 :=
sorry

end perfect_square_a_value_l18_18694


namespace ten_sided_polygon_diagonals_l18_18627

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem ten_sided_polygon_diagonals :
  number_of_diagonals 10 = 35 :=
by sorry

end ten_sided_polygon_diagonals_l18_18627


namespace nature_of_roots_l18_18210

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 + 3 * x^2 - 8 * x + 16

theorem nature_of_roots : (∀ x : ℝ, x < 0 → P x > 0) ∧ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ P x = 0 := 
by
  sorry

end nature_of_roots_l18_18210


namespace odds_against_C_l18_18829

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C (pA pB pC : ℚ) (hA : pA = 1 / 3) (hB : pB = 1 / 5) (hC : pC = 7 / 15) :
  odds_against_winning pC = 8 / 7 :=
by
  -- Definitions based on the conditions provided in a)
  have h1 : odds_against_winning (1/3) = 2 := by sorry
  have h2 : odds_against_winning (1/5) = 4 := by sorry

  -- Odds against C
  have h3 : 1 - (pA + pB) = pC := by sorry
  have h4 : pA + pB = 8 / 15 := by sorry

  -- Show that odds against C winning is 8/7
  have h5 : odds_against_winning pC = 8 / 7 := by sorry
  exact h5

end odds_against_C_l18_18829


namespace intersection_of_A_and_B_l18_18812

def A : Set ℝ := {y | y > 1}
def B : Set ℝ := {x | Real.log x ≥ 0}
def Intersect : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = Intersect :=
by
  sorry

end intersection_of_A_and_B_l18_18812


namespace probability_no_3by3_red_grid_correct_l18_18217

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l18_18217


namespace betty_garden_total_plants_l18_18963

theorem betty_garden_total_plants (B O : ℕ) 
  (h1 : B = 5) 
  (h2 : O = 2 + 2 * B) : 
  B + O = 17 := by
  sorry

end betty_garden_total_plants_l18_18963


namespace vasya_faster_than_petya_l18_18457

theorem vasya_faster_than_petya 
  (L : ℝ) (v : ℝ) (x : ℝ) (t : ℝ) 
  (meeting_condition : (v + x * v) * t = L)
  (petya_lap : v * t = L)
  (vasya_meet_petya_after_lap : x * v * t = 2 * L) :
  x = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end vasya_faster_than_petya_l18_18457


namespace range_of_a_l18_18817

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + 2 * x0 + a ≤ 0
def q (a : ℝ) : Prop := ∀ x > 0, x + 1/x > a

-- The theorem statement: if p is false and q is true, then 1 < a < 2
theorem range_of_a (a : ℝ) (h1 : ¬ p a) (h2 : q a) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l18_18817


namespace coefficient_x2y2_l18_18740

theorem coefficient_x2y2 : 
  let expr1 := (1 + x) ^ 3
  let expr2 := (1 + y) ^ 4
  let C3_2 := Nat.choose 3 2
  let C4_2 := Nat.choose 4 2
  (C3_2 * C4_2 = 18) := by
    sorry

end coefficient_x2y2_l18_18740


namespace prime_divisor_greater_than_p_l18_18414

theorem prime_divisor_greater_than_p (p q : ℕ) (hp : Prime p) 
    (hq : Prime q) (hdiv : q ∣ 2^p - 1) : p < q := 
by
  sorry

end prime_divisor_greater_than_p_l18_18414


namespace min_value_l18_18656

variable {α : Type*} [LinearOrderedField α]

-- Define a geometric sequence with strictly positive terms
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ (q : α), q > 0 ∧ ∀ n, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → α) (S : ℕ → α)
variables (h_geom : is_geometric_sequence a)
variables (h_pos : ∀ n, a n > 0)
variables (h_a23 : a 2 * a 6 = 4) (h_a3 : a 3 = 1)

-- Sum of the first n terms of a geometric sequence
def sum_first_n (a : ℕ → α) (n : ℕ) : α :=
  if n = 0 then 0
  else a 0 * ((1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0)))

-- Statement of the theorem
theorem min_value (a : ℕ → α) (S : ℕ → α) 
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a23 : a 2 * a 6 = 4)
  (h_a3 : a 3 = 1)
  (h_Sn : ∀ n, S n = sum_first_n a n) :
  ∃ n, n = 3 ∧ (S n + 9 / 4) ^ 2 / (2 * a n) = 8 :=
sorry

end min_value_l18_18656


namespace divisibility_properties_l18_18475

theorem divisibility_properties (a b : ℤ) (k : ℕ) :
  (¬(a + b ∣ a^(2*k) + b^(2*k)) ∧ ¬(a - b ∣ a^(2*k) + b^(2*k))) ∧ 
  ((a + b ∣ a^(2*k) - b^(2*k)) ∧ (a - b ∣ a^(2*k) - b^(2*k))) ∧ 
  (a + b ∣ a^(2*k + 1) + b^(2*k + 1)) ∧ 
  (a - b ∣ a^(2*k + 1) - b^(2*k + 1)) := 
by sorry

end divisibility_properties_l18_18475


namespace find_number_l18_18916

theorem find_number (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by
  sorry

end find_number_l18_18916


namespace hiker_speeds_l18_18013

theorem hiker_speeds:
  ∃ (d : ℝ), 
  (d > 5) ∧ ((70 / (d - 5)) = (110 / d)) ∧ (d - 5 = 8.75) :=
by
  sorry

end hiker_speeds_l18_18013


namespace total_cost_eq_57_l18_18003

namespace CandyCost

-- Conditions
def cost_of_caramel : ℕ := 3
def cost_of_candy_bar : ℕ := 2 * cost_of_caramel
def cost_of_cotton_candy : ℕ := (4 * cost_of_candy_bar) / 2

-- Define the total cost calculation
def total_cost : ℕ :=
  (6 * cost_of_candy_bar) + (3 * cost_of_caramel) + cost_of_cotton_candy

-- Theorem we want to prove
theorem total_cost_eq_57 : total_cost = 57 :=
by
  sorry  -- Proof to be provided

end CandyCost

end total_cost_eq_57_l18_18003


namespace negation_of_universal_l18_18001

theorem negation_of_universal :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end negation_of_universal_l18_18001


namespace frog_ends_on_horizontal_side_l18_18040

-- Definitions for the problem conditions
def frog_jump_probability (x y : ℤ) : ℚ := sorry

-- Main theorem statement based on the identified question and correct answer
theorem frog_ends_on_horizontal_side :
  frog_jump_probability 2 3 = 13 / 14 :=
sorry

end frog_ends_on_horizontal_side_l18_18040


namespace number_of_students_l18_18577

theorem number_of_students (n : ℕ)
  (h1 : ∃ n, (175 * n) / n = 175)
  (h2 : 175 * n - 40 = 173 * n) :
  n = 20 :=
sorry

end number_of_students_l18_18577


namespace length_of_segment_correct_l18_18256

noncomputable def length_of_segment (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem length_of_segment_correct :
  length_of_segment 5 (-1) 13 11 = 4 * Real.sqrt 13 := by
  sorry

end length_of_segment_correct_l18_18256


namespace sum_m_n_l18_18235

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l18_18235


namespace weight_of_new_person_l18_18621

theorem weight_of_new_person {avg_increase : ℝ} (n : ℕ) (p : ℝ) (w : ℝ) (h : n = 8) (h1 : avg_increase = 2.5) (h2 : w = 67):
  p = 87 :=
by
  sorry

end weight_of_new_person_l18_18621


namespace relay_race_time_l18_18736

-- Define the time it takes for each runner.
def Rhonda_time : ℕ := 24
def Sally_time : ℕ := Rhonda_time + 2
def Diane_time : ℕ := Rhonda_time - 3

-- Define the total time for the relay race.
def total_relay_time : ℕ := Rhonda_time + Sally_time + Diane_time

-- State the theorem we want to prove: the total relay time is 71 seconds.
theorem relay_race_time : total_relay_time = 71 := 
by 
  -- The following "sorry" indicates a step where the proof would be completed.
  sorry

end relay_race_time_l18_18736


namespace binomial_expansion_problem_l18_18249

theorem binomial_expansion_problem :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ),
    (1 + 2 * x) ^ 11 =
      a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 +
      a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 +
      a_9 * x^9 + a_10 * x^10 + a_11 * x^11 →
    a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 + 5 * a_5 - 6 * a_6 +
    7 * a_7 - 8 * a_8 + 9 * a_9 - 10 * a_10 + 11 * a_11 = 22 :=
by
  -- The proof is omitted for this exercise
  sorry

end binomial_expansion_problem_l18_18249


namespace common_noninteger_root_eq_coeffs_l18_18735

theorem common_noninteger_root_eq_coeffs (p1 p2 q1 q2 : ℤ) (α : ℝ) :
  (α^2 + (p1: ℝ) * α + (q1: ℝ) = 0) ∧ (α^2 + (p2: ℝ) * α + (q2: ℝ) = 0) ∧ ¬(∃ (k : ℤ), α = k) → p1 = p2 ∧ q1 = q2 :=
by {
  sorry
}

end common_noninteger_root_eq_coeffs_l18_18735


namespace roots_square_sum_l18_18261

theorem roots_square_sum (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) : 
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by 
  -- proof skipped
  sorry

end roots_square_sum_l18_18261


namespace cone_volume_calc_l18_18392

noncomputable def cone_volume (diameter slant_height: ℝ) : ℝ :=
  let r := diameter / 2
  let h := Real.sqrt (slant_height^2 - r^2)
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_calc :
  cone_volume 12 10 = 96 * Real.pi :=
by
  sorry

end cone_volume_calc_l18_18392


namespace problem1_solution_correct_problem2_solution_correct_l18_18473

def problem1 (x : ℤ) : Prop := (x - 1) ∣ (x + 3)
def problem2 (x : ℤ) : Prop := (x + 2) ∣ (x^2 + 2)
def solution1 (x : ℤ) : Prop := x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5
def solution2 (x : ℤ) : Prop := x = -8 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 4

theorem problem1_solution_correct : ∀ x: ℤ, problem1 x ↔ solution1 x := by
  sorry

theorem problem2_solution_correct : ∀ x: ℤ, problem2 x ↔ solution2 x := by
  sorry

end problem1_solution_correct_problem2_solution_correct_l18_18473


namespace ellie_runs_8_miles_in_24_minutes_l18_18799

theorem ellie_runs_8_miles_in_24_minutes (time_max : ℝ) (distance_max : ℝ) 
  (time_ellie_fraction : ℝ) (distance_ellie : ℝ) (distance_ellie_final : ℝ)
  (h1 : distance_max = 6) 
  (h2 : time_max = 36) 
  (h3 : time_ellie_fraction = 1/3) 
  (h4 : distance_ellie = 4) 
  (h5 : distance_ellie_final = 8) :
  ((time_ellie_fraction * time_max) / distance_ellie) * distance_ellie_final = 24 :=
by
  sorry

end ellie_runs_8_miles_in_24_minutes_l18_18799


namespace probability_no_3x3_red_square_l18_18212

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l18_18212


namespace jesse_money_left_after_mall_l18_18552

theorem jesse_money_left_after_mall :
  ∀ (initial_amount novel_cost lunch_cost total_spent remaining_amount : ℕ),
    initial_amount = 50 →
    novel_cost = 7 →
    lunch_cost = 2 * novel_cost →
    total_spent = novel_cost + lunch_cost →
    remaining_amount = initial_amount - total_spent →
    remaining_amount = 29 :=
by
  intros initial_amount novel_cost lunch_cost total_spent remaining_amount
  sorry

end jesse_money_left_after_mall_l18_18552


namespace simplify_expression_l18_18723

noncomputable def p (a b c x k : ℝ) := 
  k * (((x + a) ^ 2 / ((a - b) * (a - c))) +
       ((x + b) ^ 2 / ((b - a) * (b - c))) +
       ((x + c) ^ 2 / ((c - a) * (c - b))))

theorem simplify_expression (a b c k : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : b ≠ c) (h₃ : k ≠ 0) :
  p a b c x k = k :=
sorry

end simplify_expression_l18_18723


namespace coeff_x_squared_l18_18807

def P (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5
def Q (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 8

theorem coeff_x_squared :
  let coeff : ℝ := 82 in
  ∀ x : ℝ, (P x * Q x).coeff 2 = coeff :=
sorry

end coeff_x_squared_l18_18807


namespace determinant_of_A_l18_18645

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 0, -2],
  ![8, 5, -4],
  ![3, 3, 6]
]

theorem determinant_of_A : A.det = 108 := by
  sorry

end determinant_of_A_l18_18645


namespace chocolate_bars_in_large_box_l18_18184

theorem chocolate_bars_in_large_box : 
  let small_boxes := 19 
  let bars_per_small_box := 25 
  let total_bars := small_boxes * bars_per_small_box 
  total_bars = 475 := by 
  -- declarations and assumptions
  let small_boxes : ℕ := 19 
  let bars_per_small_box : ℕ := 25 
  let total_bars : ℕ := small_boxes * bars_per_small_box 
  sorry

end chocolate_bars_in_large_box_l18_18184


namespace solve_fractional_eq_l18_18595

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l18_18595


namespace bowling_average_before_last_match_l18_18347

theorem bowling_average_before_last_match
  (wickets_before_last : ℕ)
  (wickets_last_match : ℕ)
  (runs_last_match : ℕ)
  (decrease_in_average : ℝ)
  (average_before_last : ℝ) :

  wickets_before_last = 115 →
  wickets_last_match = 6 →
  runs_last_match = 26 →
  decrease_in_average = 0.4 →
  (average_before_last - decrease_in_average) = 
  ((wickets_before_last * average_before_last + runs_last_match) / 
  (wickets_before_last + wickets_last_match)) →
  average_before_last = 12.4 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end bowling_average_before_last_match_l18_18347


namespace quadratic_inequality_solution_l18_18844

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l18_18844


namespace integer_solution_unique_l18_18653

variable (x y : ℤ)

def nested_sqrt_1964_times (x : ℤ) : ℤ := 
  sorry -- (This should define the function for nested sqrt 1964 times, but we'll use sorry to skip the proof)

theorem integer_solution_unique : 
  nested_sqrt_1964_times x = y → x = 0 ∧ y = 0 :=
by
  intros h
  sorry -- Proof of the theorem goes here

end integer_solution_unique_l18_18653


namespace value_of_expression_l18_18719

theorem value_of_expression {a b : ℝ} (h1 : 2 * a^2 + 6 * a - 14 = 0) (h2 : 2 * b^2 + 6 * b - 14 = 0) :
  (2 * a - 3) * (4 * b - 6) = -2 :=
by
  sorry

end value_of_expression_l18_18719


namespace least_distance_on_cone_l18_18045

noncomputable def least_distance_fly_could_crawl_cone (R C : ℝ) (slant_height : ℝ) (start_dist vertex_dist : ℝ) : ℝ :=
  if start_dist = 150 ∧ vertex_dist = 450 ∧ R = 500 ∧ C = 800 * Real.pi ∧ slant_height = R ∧ 
     (500 * (8 * Real.pi / 5) = 800 * Real.pi) then 600 else 0

theorem least_distance_on_cone : least_distance_fly_could_crawl_cone 500 (800 * Real.pi) 500 150 450 = 600 :=
by
  sorry

end least_distance_on_cone_l18_18045


namespace negation_proposition_l18_18445

open Classical

variable (x : ℝ)

theorem negation_proposition :
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end negation_proposition_l18_18445


namespace complement_union_l18_18391

open Set

-- Definitions from the given conditions
def U : Set ℕ := {x | x ≤ 9}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- Statement of the proof problem
theorem complement_union :
  compl (A ∪ B) = {7, 8, 9} :=
sorry

end complement_union_l18_18391


namespace range_of_p_l18_18742

def p (x : ℝ) : ℝ := (x^3 + 3)^2

theorem range_of_p :
  (∀ y, ∃ x ∈ Set.Ici (-1 : ℝ), p x = y) ↔ y ∈ Set.Ici (4 : ℝ) :=
by
  sorry

end range_of_p_l18_18742


namespace value_of_a3_a6_a9_l18_18284

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The common difference is 2
axiom common_difference : d = 2

-- Condition: a_1 + a_4 + a_7 = -50
axiom sum_a1_a4_a7 : a 1 + a 4 + a 7 = -50

-- The goal: a_3 + a_6 + a_9 = -38
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = -38 := 
by 
  sorry

end value_of_a3_a6_a9_l18_18284


namespace binom_9_5_l18_18995

theorem binom_9_5 : binomial 9 5 = 756 := 
by 
  sorry

end binom_9_5_l18_18995


namespace solve_for_y_l18_18797

theorem solve_for_y : ∀ (y : ℚ), 
  (y + 4 / 5 = 2 / 3 + y / 6) → y = -4 / 25 :=
by
  sorry

end solve_for_y_l18_18797


namespace spherical_to_rectangular_conversion_l18_18367

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular 4 (Real.pi / 2) (Real.pi / 4) = (0, 2 * Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l18_18367


namespace mother_hubbard_children_l18_18787

theorem mother_hubbard_children :
  (∃ c : ℕ, (2 / 3 : ℚ) = c * (1 / 12 : ℚ)) → c = 8 :=
by
  sorry

end mother_hubbard_children_l18_18787


namespace degree_greater_than_2_l18_18555

variable (P Q : ℤ[X]) -- P and Q are polynomials with integer coefficients

theorem degree_greater_than_2 (P_nonconstant : ¬(P.degree = 0))
  (Q_nonconstant : ¬(Q.degree = 0))
  (h : ∃ S : Finset ℤ, S.card ≥ 25 ∧ ∀ x ∈ S, (P.eval x) * (Q.eval x) = 2009) :
  P.degree > 2 ∧ Q.degree > 2 :=
by
  sorry

end degree_greater_than_2_l18_18555


namespace swimming_lane_length_l18_18798

-- Conditions
def num_round_trips : ℕ := 3
def total_distance : ℕ := 600

-- Hypothesis that 1 round trip is equivalent to 2 lengths of the lane
def lengths_per_round_trip : ℕ := 2

-- Statement to prove
theorem swimming_lane_length :
  (total_distance / (num_round_trips * lengths_per_round_trip) = 100) := by
  sorry

end swimming_lane_length_l18_18798


namespace students_in_school_B_l18_18169

theorem students_in_school_B 
    (A B C : ℕ) 
    (h1 : A + C = 210) 
    (h2 : A = 4 * B) 
    (h3 : C = 3 * B) : 
    B = 30 := 
by 
    sorry

end students_in_school_B_l18_18169


namespace binom_9_5_eq_126_l18_18973

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := 
by
  sorry

end binom_9_5_eq_126_l18_18973


namespace max_ab_sum_l18_18146

theorem max_ab_sum (a b: ℤ) (h1: a ≠ b) (h2: a * b = -132) (h3: a ≤ b): a + b = -1 :=
sorry

end max_ab_sum_l18_18146


namespace photos_difference_is_120_l18_18871

theorem photos_difference_is_120 (initial_photos : ℕ) (final_photos : ℕ) (first_day_factor : ℕ) (first_day_photos : ℕ) (second_day_photos : ℕ) : 
  initial_photos = 400 → 
  final_photos = 920 → 
  first_day_factor = 2 →
  first_day_photos = initial_photos / first_day_factor →
  final_photos = initial_photos + first_day_photos + second_day_photos →
  second_day_photos - first_day_photos = 120 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end photos_difference_is_120_l18_18871


namespace find_sum_lent_l18_18633

theorem find_sum_lent (r t : ℝ) (I : ℝ) (P : ℝ) (h1: r = 0.06) (h2 : t = 8) (h3 : I = P - 520) (h4: I = P * r * t) : P = 1000 := by
  sorry

end find_sum_lent_l18_18633


namespace words_per_page_large_font_l18_18407

theorem words_per_page_large_font
    (total_words : ℕ)
    (large_font_pages : ℕ)
    (small_font_pages : ℕ)
    (small_font_words_per_page : ℕ)
    (total_pages : ℕ)
    (words_in_large_font : ℕ) :
    total_words = 48000 →
    total_pages = 21 →
    large_font_pages = 4 →
    small_font_words_per_page = 2400 →
    words_in_large_font = total_words - (small_font_pages * small_font_words_per_page) →
    small_font_pages = total_pages - large_font_pages →
    (words_in_large_font = large_font_pages * 1800) :=
by 
    sorry

end words_per_page_large_font_l18_18407


namespace magnitude_2a_minus_b_l18_18086

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (θ : ℝ) (h_angle : θ = 5 * Real.pi / 6)
variables (h_mag_a : ‖a‖ = 4) (h_mag_b : ‖b‖ = Real.sqrt 3)

theorem magnitude_2a_minus_b :
  ‖2 • a - b‖ = Real.sqrt 91 := by
  -- Proof goes here.
  sorry

end magnitude_2a_minus_b_l18_18086


namespace journey_time_l18_18547

theorem journey_time 
  (d1 d2 T : ℝ)
  (h1 : d1 / 30 + (150 - d1) / 10 = T)
  (h2 : d1 / 30 + d2 / 30 + (150 - (d1 - d2)) / 30 = T)
  (h3 : (d1 - d2) / 10 + (150 - (d1 - d2)) / 30 = T) :
  T = 5 := 
sorry

end journey_time_l18_18547


namespace quadratic_inequality_solution_l18_18006

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 12 ≥ 0} = {x : ℝ | x ≤ -4 ∨ x ≥ 3} :=
sorry

end quadratic_inequality_solution_l18_18006


namespace anne_speed_l18_18788

-- Conditions
def time_hours : ℝ := 3
def distance_miles : ℝ := 6

-- Question with correct answer
theorem anne_speed : distance_miles / time_hours = 2 := by 
  sorry

end anne_speed_l18_18788


namespace find_maximum_k_l18_18748

theorem find_maximum_k {k : ℝ} 
  (h_eq : ∀ x, x^2 + k * x + 8 = 0)
  (h_roots_diff : ∀ x₁ x₂, x₁ - x₂ = 10) :
  k = 2 * Real.sqrt 33 := 
sorry

end find_maximum_k_l18_18748


namespace binom_9_5_eq_126_l18_18980

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end binom_9_5_eq_126_l18_18980


namespace greatest_common_divisor_B_l18_18412

def sum_of_five_consecutive_integers (x : ℕ) : ℕ :=
  (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

def B : set ℕ := {n | ∃ x : ℕ, n = sum_of_five_consecutive_integers x}

theorem greatest_common_divisor_B : gcd (set.to_finset B).min' (set.to_finset B).max' = 5 :=
by
  sorry

end greatest_common_divisor_B_l18_18412


namespace find_d_l18_18123

theorem find_d (d : ℤ) :
  (∀ x : ℤ, (4 * x^3 + 13 * x^2 + d * x + 18 = 0 ↔ x = -3)) →
  d = 9 :=
by
  sorry

end find_d_l18_18123


namespace compute_combination_l18_18985

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l18_18985


namespace cube_edge_length_l18_18512

def radius := 2
def edge_length (r : ℕ) := 4 + 2 * r

theorem cube_edge_length :
  ∀ r : ℕ, r = radius → edge_length r = 8 :=
by
  intros r h
  rw [h, edge_length]
  rfl

end cube_edge_length_l18_18512


namespace probability_no_3x3_red_square_l18_18230

theorem probability_no_3x3_red_square (p : ℚ) : 
  (∀ (grid : Fin 4 × Fin 4 → bool), 
    (∀ i j : Fin 4, (grid (i, j) = tt ∨ grid (i, j) = ff)) → 
    p = 65410 / 65536) :=
by sorry

end probability_no_3x3_red_square_l18_18230


namespace isabella_hair_length_l18_18272

theorem isabella_hair_length (original : ℝ) (increase_percent : ℝ) (new_length : ℝ) 
    (h1 : original = 18) (h2 : increase_percent = 0.75) 
    (h3 : new_length = original + increase_percent * original) : 
    new_length = 31.5 := by sorry

end isabella_hair_length_l18_18272


namespace coeff_x3_in_expansion_l18_18438

theorem coeff_x3_in_expansion : (Polynomial.coeff ((Polynomial.C 1 - Polynomial.C 2 * Polynomial.X)^6) 3) = -160 := 
by 
  sorry

end coeff_x3_in_expansion_l18_18438


namespace trigonometric_signs_l18_18085

noncomputable def terminal_side (θ α : ℝ) : Prop :=
  ∃ k : ℤ, θ = α + 2 * k * Real.pi

theorem trigonometric_signs :
  ∀ (α θ : ℝ), 
    (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 5) ∧ terminal_side θ α →
    (Real.sin θ < 0) ∧ (Real.cos θ > 0) ∧ (Real.tan θ < 0) →
    (Real.sin θ / abs (Real.sin θ) + Real.cos θ / abs (Real.cos θ) + Real.tan θ / abs (Real.tan θ) = -1) :=
by intros
   sorry

end trigonometric_signs_l18_18085


namespace students_in_each_class_l18_18421

-- Define the conditions
def sheets_per_student : ℕ := 5
def total_sheets : ℕ := 400
def number_of_classes : ℕ := 4

-- Define the main proof theorem
theorem students_in_each_class : (total_sheets / sheets_per_student) / number_of_classes = 20 := by
  sorry -- Proof goes here

end students_in_each_class_l18_18421


namespace cost_price_of_book_l18_18772

-- Define the variables and conditions
variable (C : ℝ)
variable (P : ℝ)
variable (S : ℝ)

-- State the conditions given in the problem
def conditions := S = 260 ∧ P = 0.20 * C ∧ S = C + P

-- State the theorem
theorem cost_price_of_book (h : conditions C P S) : C = 216.67 :=
sorry

end cost_price_of_book_l18_18772


namespace find_c_l18_18066

theorem find_c (c : ℝ) 
  (h1 : ∃ x : ℝ, 3 * x^2 + 23 * x - 75 = 0 ∧ x = ⌊c⌋) 
  (h2 : ∃ y : ℝ, 4 * y^2 - 19 * y + 3 = 0 ∧ y = c - ⌊c⌋) : 
  c = -11.84 :=
by
  sorry

end find_c_l18_18066


namespace find_n_l18_18472

theorem find_n (n : ℕ) (h : 1 < n) :
  (∀ a b : ℕ, Nat.gcd a b = 1 → (a % n = b % n ↔ (a * b) % n = 1)) →
  (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 ∨ n = 24) :=
by
  sorry

end find_n_l18_18472


namespace find_width_of_rectangle_l18_18447

variable (w : ℝ) (l : ℝ) (P : ℝ)

def width_correct (h1 : P = 150) (h2 : l = w + 15) : Prop :=
  w = 30

-- Theorem statement in Lean
theorem find_width_of_rectangle (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : width_correct w l P h1 h2 :=
by
  sorry

end find_width_of_rectangle_l18_18447


namespace integer_modulo_problem_l18_18458

theorem integer_modulo_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ (-250 % 23 = n) := 
  sorry

end integer_modulo_problem_l18_18458


namespace fraction_of_A_eq_l18_18183

noncomputable def fraction_A (A B C T : ℕ) : ℚ :=
  A / (T - A)

theorem fraction_of_A_eq :
  ∃ (A B C T : ℕ), T = 360 ∧ A = B + 10 ∧ B = 2 * (A + C) / 7 ∧ T = A + B + C ∧ fraction_A A B C T = 1 / 3 :=
by
  sorry

end fraction_of_A_eq_l18_18183


namespace single_fraction_l18_18926

theorem single_fraction (c : ℕ) : (6 + 5 * c) / 5 + 3 = (21 + 5 * c) / 5 :=
by sorry

end single_fraction_l18_18926


namespace problem_statement_l18_18676

open Finset

variable (E : Finset ℕ) (G : Finset ℕ)

theorem problem_statement (hE : E = Finset.range 200 \ {0})
  (hG : ∀ x ∈ G, x ∈ E) 
  (h_size : G.card = 100)
  (h_sum : ∑ i in G, i = 10080)
  (h_pair_sum : ∀ i j ∈ G, i ≠ j → i + j ≠ 201) :
  (∑ i in G, (i^2) = 2686700) ∧ (G.filter (λ x, odd x)).card % 4 = 0 := sorry

end problem_statement_l18_18676


namespace custom_op_equality_l18_18124

def custom_op (x y : Int) : Int :=
  x * y - 2 * x

theorem custom_op_equality : custom_op 5 3 - custom_op 3 5 = -4 := by
  sorry

end custom_op_equality_l18_18124


namespace rectangle_area_l18_18943

theorem rectangle_area (x : ℝ) (w l : ℝ) (h₁ : l = 3 * w) (h₂ : l^2 + w^2 = x^2) :
    l * w = (3 / 10) * x^2 :=
by
  sorry

end rectangle_area_l18_18943


namespace determinant_identity_l18_18969

variable (a b : ℝ)

theorem determinant_identity :
  Matrix.det ![
      ![1, Real.sin (a - b), Real.sin a],
      ![Real.sin (a - b), 1, Real.sin b],
      ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by sorry

end determinant_identity_l18_18969


namespace min_value_l18_18522

-- Given points A, B, and C and their specific coordinates
def A : (ℝ × ℝ) := (1, 3)
def B (a : ℝ) : (ℝ × ℝ) := (a, 1)
def C (b : ℝ) : (ℝ × ℝ) := (-b, 0)

-- Conditions
axiom a_pos (a : ℝ) : a > 0
axiom b_pos (b : ℝ) : b > 0
axiom collinear (a b : ℝ) : 3 * a + 2 * b = 1

-- The theorem to prove
theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hcollinear : 3 * a + 2 * b = 1) : 
  ∃ z, z = 11 + 6 * Real.sqrt 2 ∧ ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 1) -> (3 / x + 1 / y) ≥ z :=
by sorry -- Proof to be provided

end min_value_l18_18522


namespace find_pq_l18_18704

theorem find_pq :
    ∃ (p q : ℕ), 
        (280 + q : ℚ) / (400 + p + q) = 4 / 7 ∧
        (p : ℚ) / (p + 120) = 3 / 5 ∧
        p = 180 ∧ 
        q = 120 :=
by
    sorry

end find_pq_l18_18704


namespace binomial_coefficient_plus_ten_l18_18206

theorem binomial_coefficient_plus_ten :
  Nat.choose 9 5 + 10 = 136 := 
by
  sorry

end binomial_coefficient_plus_ten_l18_18206


namespace test_point_selection_0618_method_l18_18271

theorem test_point_selection_0618_method :
  ∀ (x1 x2 x3 : ℝ),
    1000 + 0.618 * (2000 - 1000) = x1 →
    1000 + (2000 - x1) = x2 →
    x2 < x1 →
    (∀ (f : ℝ → ℝ), f x2 < f x1) →
    x1 + (1000 - x2) = x3 →
    x3 = 1236 :=
by
  intros x1 x2 x3 h1 h2 h3 h4 h5
  sorry

end test_point_selection_0618_method_l18_18271


namespace abs_diff_x_y_l18_18557

variables {x y : ℝ}

noncomputable def floor (z : ℝ) : ℤ := Int.floor z
noncomputable def fract (z : ℝ) : ℝ := z - floor z

theorem abs_diff_x_y 
  (h1 : floor x + fract y = 3.7) 
  (h2 : fract x + floor y = 4.6) : 
  |x - y| = 1.1 :=
by
  sorry

end abs_diff_x_y_l18_18557


namespace no_integer_solution_l18_18143

theorem no_integer_solution (a b c d : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) :
  ¬ ∃ n : ℤ, n^4 - (a : ℤ)*n^3 - (b : ℤ)*n^2 - (c : ℤ)*n - (d : ℤ) = 0 :=
sorry

end no_integer_solution_l18_18143


namespace arthur_walks_total_distance_l18_18051

theorem arthur_walks_total_distance :
  let east_blocks := 8
  let north_blocks := 10
  let west_blocks := 3
  let block_distance := 1 / 3
  let total_blocks := east_blocks + north_blocks + west_blocks
  let total_miles := total_blocks * block_distance
  total_miles = 7 :=
by
  sorry

end arthur_walks_total_distance_l18_18051


namespace distance_between_centers_eq_l18_18489

theorem distance_between_centers_eq (r1 r2 : ℝ) : ∃ d : ℝ, (d = r1 * Real.sqrt 2) := by
  sorry

end distance_between_centers_eq_l18_18489


namespace relation_correct_l18_18283

def M := {x : ℝ | x < 2}
def N := {x : ℝ | 0 < x ∧ x < 1}
def CR (S : Set ℝ) := {x : ℝ | x ∈ (Set.univ : Set ℝ) \ S}

theorem relation_correct : M ∪ CR N = (Set.univ : Set ℝ) :=
by sorry

end relation_correct_l18_18283


namespace gamma_bank_min_savings_l18_18026

def total_airfare_cost : ℕ := 10200 * 2 * 3
def total_hotel_cost : ℕ := 6500 * 12
def total_food_cost : ℕ := 1000 * 14 * 3
def total_excursion_cost : ℕ := 20000
def total_expenses : ℕ := total_airfare_cost + total_hotel_cost + total_food_cost + total_excursion_cost
def initial_amount_available : ℕ := 150000

def annual_rate_rebs : ℝ := 0.036
def annual_rate_gamma : ℝ := 0.045
def annual_rate_tisi : ℝ := 0.0312
def monthly_rate_btv : ℝ := 0.0025

noncomputable def compounded_amount_rebs : ℝ :=
  initial_amount_available * (1 + annual_rate_rebs / 12) ^ 6

noncomputable def compounded_amount_gamma : ℝ :=
  initial_amount_available * (1 + annual_rate_gamma / 2)

noncomputable def compounded_amount_tisi : ℝ :=
  initial_amount_available * (1 + annual_rate_tisi / 4) ^ 2

noncomputable def compounded_amount_btv : ℝ :=
  initial_amount_available * (1 + monthly_rate_btv) ^ 6

noncomputable def interest_rebs : ℝ := compounded_amount_rebs - initial_amount_available
noncomputable def interest_gamma : ℝ := compounded_amount_gamma - initial_amount_available
noncomputable def interest_tisi : ℝ := compounded_amount_tisi - initial_amount_available
noncomputable def interest_btv : ℝ := compounded_amount_btv - initial_amount_available

noncomputable def amount_needed_from_salary_rebs : ℝ :=
  total_expenses - initial_amount_available - interest_rebs

noncomputable def amount_needed_from_salary_gamma : ℝ :=
  total_expenses - initial_amount_available - interest_gamma

noncomputable def amount_needed_from_salary_tisi : ℝ :=
  total_expenses - initial_amount_available - interest_tisi

noncomputable def amount_needed_from_salary_btv : ℝ :=
  total_expenses - initial_amount_available - interest_btv

theorem gamma_bank_min_savings :
  amount_needed_from_salary_gamma = 47825.00 ∧
  amount_needed_from_salary_rebs = 48479.67 ∧
  amount_needed_from_salary_tisi = 48850.87 ∧
  amount_needed_from_salary_btv = 48935.89 :=
by sorry

end gamma_bank_min_savings_l18_18026


namespace f_even_f_increasing_f_range_l18_18377

variables {R : Type*} [OrderedRing R] (f : R → R)

-- Conditions
axiom f_mul : ∀ x y : R, f (x * y) = f x * f y
axiom f_neg1 : f (-1) = 1
axiom f_27 : f 27 = 9
axiom f_lt_1 : ∀ x : R, 0 ≤ x → x < 1 → 0 ≤ f x ∧ f x < 1

-- Questions
theorem f_even (x : R) : f x = f (-x) :=
by sorry

theorem f_increasing (x1 x2 : R) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 < x2) : f x1 < f x2 :=
by sorry

theorem f_range (a : R) (h1 : 0 ≤ a) (h2 : f (a + 1) ≤ 39) : 0 ≤ a ∧ a ≤ 2 :=
by sorry

end f_even_f_increasing_f_range_l18_18377


namespace min_denominator_of_sum_600_700_l18_18014

def is_irreducible_fraction (a : ℕ) (b : ℕ) : Prop := 
  Nat.gcd a b = 1

def min_denominator_of_sum (d1 d2 : ℕ) (a b : ℕ) : ℕ :=
  let lcm := Nat.lcm d1 d2
  let sum_numerator := a * (lcm / d1) + b * (lcm / d2)
  Nat.gcd sum_numerator lcm

theorem min_denominator_of_sum_600_700 (a b : ℕ) (h1 : is_irreducible_fraction a 600) (h2 : is_irreducible_fraction b 700) :
  min_denominator_of_sum 600 700 a b = 168 := sorry

end min_denominator_of_sum_600_700_l18_18014


namespace count_k_square_modulo_485_l18_18069

theorem count_k_square_modulo_485
    (d : ℕ := 485)
    (m : ℕ := 485000)
    (count : ℕ := 2000) :
    ∃ c, c = count ∧ (∀ k : ℕ, k ≤ m → (k^2 - 1) % d = 0 ↔ k ∈ {1..m} ∧ k^2 % d = 1) :=
by
  sorry

end count_k_square_modulo_485_l18_18069


namespace total_plants_in_garden_l18_18960

-- Definitions based on conditions
def basil_plants : ℕ := 5
def oregano_plants : ℕ := 2 + 2 * basil_plants

-- Theorem statement
theorem total_plants_in_garden : basil_plants + oregano_plants = 17 := by
  -- Skipping the proof with sorry
  sorry

end total_plants_in_garden_l18_18960


namespace sam_paid_amount_l18_18296

theorem sam_paid_amount (F : ℝ) (Joe Peter Sam : ℝ) 
  (h1 : Joe = (1/4)*F + 7) 
  (h2 : Peter = (1/3)*F - 7) 
  (h3 : Sam = (1/2)*F - 12)
  (h4 : Joe + Peter + Sam = F) : 
  Sam = 60 := 
by 
  sorry

end sam_paid_amount_l18_18296


namespace find_y_of_equations_l18_18836

theorem find_y_of_equations (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 1 + 1 / y) (h2 : y = 2 + 1 / x) : 
  y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3 :=
by
  sorry

end find_y_of_equations_l18_18836


namespace min_pos_int_k_l18_18897

noncomputable def minimum_k (x0 : ℝ) : ℝ := (x0 * (Real.log x0 + 1)) / (x0 - 2)

theorem min_pos_int_k : ∃ k : ℝ, (∀ x0 : ℝ, x0 > 2 → k > minimum_k x0) ∧ k = 5 := 
by
  sorry

end min_pos_int_k_l18_18897


namespace problem_a_l18_18176

theorem problem_a (nums : Fin 101 → ℤ) : ∃ i j : Fin 101, i ≠ j ∧ (nums i - nums j) % 100 = 0 := sorry

end problem_a_l18_18176


namespace rectangle_area_l18_18782

theorem rectangle_area (w d : ℝ) 
  (h1 : d = (w^2 + (3 * w)^2) ^ (1/2))
  (h2 : ∃ A : ℝ, A = w * 3 * w) :
  ∃ A : ℝ, A = 3 * (d^2 / 10) := 
by {
  sorry
}

end rectangle_area_l18_18782


namespace aitana_fraction_more_than_jayda_l18_18640

theorem aitana_fraction_more_than_jayda (jayda_spending total_spending aitana_spending : ℚ)
  (h_jayda : jayda_spending = 400)
  (h_total : total_spending = 960)
  (h_aitana : aitana_spending = total_spending - jayda_spending) :
  (aitana_spending - jayda_spending) / jayda_spending = 2 / 5 := by
  sorry

end aitana_fraction_more_than_jayda_l18_18640
