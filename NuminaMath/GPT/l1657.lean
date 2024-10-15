import Mathlib

namespace NUMINAMATH_GPT_min_three_beverages_overlap_l1657_165768

variable (a b c d : ℝ)
variable (ha : a = 0.9)
variable (hb : b = 0.8)
variable (hc : c = 0.7)

theorem min_three_beverages_overlap : d = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_min_three_beverages_overlap_l1657_165768


namespace NUMINAMATH_GPT_semicircle_problem_l1657_165739

theorem semicircle_problem (N : ℕ) (r : ℝ) (π : ℝ) (hπ : 0 < π) 
  (h1 : ∀ (r : ℝ), ∃ (A B : ℝ), A = N * (π * r^2 / 2) ∧ B = (π * (N^2 * r^2 / 2) - N * (π * r^2 / 2)) ∧ A / B = 1 / 3) :
  N = 4 :=
by
  sorry

end NUMINAMATH_GPT_semicircle_problem_l1657_165739


namespace NUMINAMATH_GPT_exists_real_x_for_sequence_floor_l1657_165729

open Real

theorem exists_real_x_for_sequence_floor (a : Fin 1998 → ℕ)
  (h1 : ∀ n : Fin 1998, 0 ≤ a n)
  (h2 : ∀ (i j : Fin 1998), (i.val + j.val ≤ 1997) → (a i + a j ≤ a ⟨i.val + j.val, sorry⟩ ∧ a ⟨i.val + j.val, sorry⟩ ≤ a i + a j + 1)) :
  ∃ x : ℝ, ∀ n : Fin 1998, a n = ⌊(n.val + 1) * x⌋ :=
sorry

end NUMINAMATH_GPT_exists_real_x_for_sequence_floor_l1657_165729


namespace NUMINAMATH_GPT_correctness_of_statements_l1657_165742

theorem correctness_of_statements 
  (A B C D : Prop)
  (h1 : A → B) (h2 : ¬(B → A))
  (h3 : C → B) (h4 : B → C)
  (h5 : D → C) (h6 : ¬(C → D)) : 
  (A → (C ∧ ¬(C → A))) ∧ (¬(A → D) ∧ ¬(D → A)) := 
by
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_correctness_of_statements_l1657_165742


namespace NUMINAMATH_GPT_ay_bz_cx_lt_S_squared_l1657_165731

theorem ay_bz_cx_lt_S_squared 
  (S : ℝ) (a b c x y z : ℝ) 
  (hS : 0 < S) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h1 : a + x = S) 
  (h2 : b + y = S) 
  (h3 : c + z = S) : 
  a * y + b * z + c * x < S^2 := 
sorry

end NUMINAMATH_GPT_ay_bz_cx_lt_S_squared_l1657_165731


namespace NUMINAMATH_GPT_compound_interest_rate_l1657_165747

theorem compound_interest_rate (
  P : ℝ) (r : ℝ)  (A : ℕ → ℝ) :
  A 2 = 2420 ∧ A 3 = 3025 ∧ 
  (∀ n : ℕ, A n = P * (1 + r / 100)^n) → r = 25 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1657_165747


namespace NUMINAMATH_GPT_drawing_probability_consecutive_order_l1657_165782

theorem drawing_probability_consecutive_order :
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  probability = 1 / 665280 :=
by
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  sorry

end NUMINAMATH_GPT_drawing_probability_consecutive_order_l1657_165782


namespace NUMINAMATH_GPT_find_k_l1657_165714

theorem find_k (n m : ℕ) (hn : n > 0) (hm : m > 0) (h : (1 : ℚ) / n^2 + 1 / m^2 = k / (n^2 + m^2)) : k = 4 :=
sorry

end NUMINAMATH_GPT_find_k_l1657_165714


namespace NUMINAMATH_GPT_gray_region_area_l1657_165772

theorem gray_region_area (d_small r_large r_small π : ℝ) (h1 : d_small = 6)
    (h2 : r_large = 3 * r_small) (h3 : r_small = d_small / 2) :
    (π * r_large ^ 2 - π * r_small ^ 2) = 72 * π := 
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_gray_region_area_l1657_165772


namespace NUMINAMATH_GPT_geometric_series_sum_l1657_165703

theorem geometric_series_sum :
  let a := 2 / 3
  let r := 1 / 3
  a / (1 - r) = 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1657_165703


namespace NUMINAMATH_GPT_men_in_first_group_l1657_165719

theorem men_in_first_group (M : ℕ) (h1 : (M * 7 * 18) = (12 * 7 * 12)) : M = 8 :=
by sorry

end NUMINAMATH_GPT_men_in_first_group_l1657_165719


namespace NUMINAMATH_GPT_largest_digit_divisible_by_4_l1657_165732

theorem largest_digit_divisible_by_4 :
  ∃ (A : ℕ), A ≤ 9 ∧ (∃ n : ℕ, 100000 * 4 + 10000 * A + 67994 = n * 4) ∧ 
  (∀ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (∃ m : ℕ, 100000 * 4 + 10000 * B + 67994 = m * 4) → B ≤ A) :=
sorry

end NUMINAMATH_GPT_largest_digit_divisible_by_4_l1657_165732


namespace NUMINAMATH_GPT_problem_l1657_165774

theorem problem (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : a^2 + b^2 = 29 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1657_165774


namespace NUMINAMATH_GPT_g_at_neg1_l1657_165744

-- Defining even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

-- Given functions f and g
variables (f g : ℝ → ℝ)

-- Given conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom fg_relation : ∀ x : ℝ, f x - g x = 2^(1 - x)

-- Proof statement
theorem g_at_neg1 : g (-1) = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_g_at_neg1_l1657_165744


namespace NUMINAMATH_GPT_seq_a5_eq_one_ninth_l1657_165796

theorem seq_a5_eq_one_ninth (a : ℕ → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  a 5 = 1 / 9 :=
sorry

end NUMINAMATH_GPT_seq_a5_eq_one_ninth_l1657_165796


namespace NUMINAMATH_GPT_incorrect_statement_d_l1657_165792

-- Definitions based on the problem's conditions
def is_acute (θ : ℝ) := 0 < θ ∧ θ < 90

def is_complementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 90

def is_supplementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 180

-- Statement D from the problem
def statement_d (θ : ℝ) := is_acute θ → ∀ θc, is_complementary θ θc → θ > θc

-- The theorem we want to prove
theorem incorrect_statement_d : ¬(∀ θ : ℝ, statement_d θ) := 
by sorry

end NUMINAMATH_GPT_incorrect_statement_d_l1657_165792


namespace NUMINAMATH_GPT_soccer_field_kids_l1657_165785

def a := 14
def b := 22
def c := a + b

theorem soccer_field_kids : c = 36 :=
by
    sorry

end NUMINAMATH_GPT_soccer_field_kids_l1657_165785


namespace NUMINAMATH_GPT_num_impossible_events_l1657_165779

def water_boils_at_90C := false
def iron_melts_at_room_temp := false
def coin_flip_results_heads := true
def abs_value_not_less_than_zero := true

theorem num_impossible_events :
  water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
  coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true →
  (if ¬water_boils_at_90C then 1 else 0) + (if ¬iron_melts_at_room_temp then 1 else 0) +
  (if ¬coin_flip_results_heads then 1 else 0) + (if ¬abs_value_not_less_than_zero then 1 else 0) = 2
:= by
  intro h
  have : 
    water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
    coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true := h
  sorry

end NUMINAMATH_GPT_num_impossible_events_l1657_165779


namespace NUMINAMATH_GPT_total_pens_bought_l1657_165743

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_pens_bought_l1657_165743


namespace NUMINAMATH_GPT_circle_symmetry_l1657_165765

def initial_circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 1 = 0

def standard_form_eq (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

def symmetric_circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

theorem circle_symmetry :
  (∀ x y : ℝ, initial_circle_eq x y ↔ standard_form_eq x y) →
  (∀ x y : ℝ, standard_form_eq x y → symmetric_circle_eq (-x) (-y)) →
  ∀ x y : ℝ, initial_circle_eq x y → symmetric_circle_eq x y :=
by
  intros h1 h2 x y hxy
  sorry

end NUMINAMATH_GPT_circle_symmetry_l1657_165765


namespace NUMINAMATH_GPT_zeoland_speeding_fine_l1657_165728

-- Define the conditions
def fine_per_mph (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ) : ℕ :=
  total_fine / (actual_speed - speed_limit)

-- Variables for the given problem
variables (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ)
variable (fine_per_mph_over_limit : ℕ)

-- Theorem statement
theorem zeoland_speeding_fine :
  total_fine = 256 ∧ speed_limit = 50 ∧ actual_speed = 66 →
  fine_per_mph total_fine actual_speed speed_limit = 16 :=
by
  sorry

end NUMINAMATH_GPT_zeoland_speeding_fine_l1657_165728


namespace NUMINAMATH_GPT_lcm_of_18_and_30_l1657_165748

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end NUMINAMATH_GPT_lcm_of_18_and_30_l1657_165748


namespace NUMINAMATH_GPT_find_a_l1657_165733

def f(x : ℚ) : ℚ := x / 3 + 2
def g(x : ℚ) : ℚ := 5 - 2 * x

theorem find_a (a : ℚ) (h : f (g a) = 4) : a = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1657_165733


namespace NUMINAMATH_GPT_complement_union_eq_l1657_165709

-- Define the sets U, M, and N
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

-- Define the complement of a set within another set
def complement (S T : Set ℕ) : Set ℕ := { x | x ∈ S ∧ x ∉ T }

-- Define the union of M and N
def union_M_N : Set ℕ := {x | x ∈ M ∨ x ∈ N}

-- State the theorem
theorem complement_union_eq :
  complement U union_M_N = {4} :=
sorry

end NUMINAMATH_GPT_complement_union_eq_l1657_165709


namespace NUMINAMATH_GPT_lowest_score_l1657_165740

theorem lowest_score (score1 score2 : ℕ) (max_score : ℕ) (desired_mean : ℕ) (lowest_possible_score : ℕ) 
  (h_score1 : score1 = 82) (h_score2 : score2 = 75) (h_max_score : max_score = 100) (h_desired_mean : desired_mean = 85)
  (h_lowest_possible_score : lowest_possible_score = 83) : 
  ∃ x1 x2 : ℕ, x1 = max_score ∧ x2 = lowest_possible_score ∧ (score1 + score2 + x1 + x2) / 4 = desired_mean := by
  sorry

end NUMINAMATH_GPT_lowest_score_l1657_165740


namespace NUMINAMATH_GPT_digit_difference_l1657_165798

variable (X Y : ℕ)

theorem digit_difference (h : 10 * X + Y - (10 * Y + X) = 27) : X - Y = 3 :=
by
  sorry

end NUMINAMATH_GPT_digit_difference_l1657_165798


namespace NUMINAMATH_GPT_mary_baseball_cards_l1657_165735

theorem mary_baseball_cards :
  let initial_cards := 18
  let torn_cards := 8
  let fred_gifted_cards := 26
  let bought_cards := 40
  let exchanged_cards := 10
  let lost_cards := 5
  
  let remaining_cards := initial_cards - torn_cards
  let after_gift := remaining_cards + fred_gifted_cards
  let after_buy := after_gift + bought_cards
  let after_exchange := after_buy - exchanged_cards + exchanged_cards
  let final_count := after_exchange - lost_cards
  
  final_count = 71 :=
by
  sorry

end NUMINAMATH_GPT_mary_baseball_cards_l1657_165735


namespace NUMINAMATH_GPT_elena_savings_l1657_165775

theorem elena_savings :
  let original_cost := 7 * 3
  let discount_rate := 0.25
  let rebate := 5
  let disc_amount := original_cost * discount_rate
  let price_after_discount := original_cost - disc_amount
  let final_price := price_after_discount - rebate
  original_cost - final_price = 10.25 :=
by
  sorry

end NUMINAMATH_GPT_elena_savings_l1657_165775


namespace NUMINAMATH_GPT_shaded_areas_I_and_III_equal_l1657_165793

def area_shaded_square_I : ℚ := 1 / 4
def area_shaded_square_II : ℚ := 1 / 2
def area_shaded_square_III : ℚ := 1 / 4

theorem shaded_areas_I_and_III_equal :
  area_shaded_square_I = area_shaded_square_III ∧
   area_shaded_square_I ≠ area_shaded_square_II ∧
   area_shaded_square_III ≠ area_shaded_square_II :=
by {
  sorry
}

end NUMINAMATH_GPT_shaded_areas_I_and_III_equal_l1657_165793


namespace NUMINAMATH_GPT_greatest_b_value_for_integer_solution_eq_l1657_165736

theorem greatest_b_value_for_integer_solution_eq : ∀ (b : ℤ), (∃ (x : ℤ), x^2 + b * x = -20) → b > 0 → b ≤ 21 :=
by
  sorry

end NUMINAMATH_GPT_greatest_b_value_for_integer_solution_eq_l1657_165736


namespace NUMINAMATH_GPT_nested_radical_solution_l1657_165707

theorem nested_radical_solution :
  (∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x ≥ 0) ∧ ∀ x : ℝ, x = Real.sqrt (18 + x) → x ≥ 0 → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_nested_radical_solution_l1657_165707


namespace NUMINAMATH_GPT_acronym_XYZ_length_l1657_165769

theorem acronym_XYZ_length :
  let X_length := 2 * Real.sqrt 2
  let Y_length := 1 + 2 * Real.sqrt 2
  let Z_length := 4 + Real.sqrt 5
  X_length + Y_length + Z_length = 5 + 4 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_acronym_XYZ_length_l1657_165769


namespace NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l1657_165723

theorem smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits : 
  ∃ n : ℕ, n < 10000 ∧ 1000 ≤ n ∧ (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ 
    (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1 ∧ d % 2 = 0) ∧ 
    (n % 11 = 0)) ∧ n = 1056 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l1657_165723


namespace NUMINAMATH_GPT_solve_equation_l1657_165778

theorem solve_equation (x : ℝ) (h_eq : 1 / (x - 2) = 3 / (x - 5)) : 
  x = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_solve_equation_l1657_165778


namespace NUMINAMATH_GPT_sad_children_count_l1657_165758

-- Definitions of conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 18
def girls : ℕ := 42
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- Calculate the number of children who are either happy or sad
def happy_or_sad_children : ℕ := total_children - neither_happy_nor_sad_children

-- Prove that the number of sad children is 10
theorem sad_children_count : happy_or_sad_children - happy_children = 10 := by
  sorry

end NUMINAMATH_GPT_sad_children_count_l1657_165758


namespace NUMINAMATH_GPT_max_side_of_triangle_with_perimeter_30_l1657_165749

theorem max_side_of_triangle_with_perimeter_30 
  (a b c : ℕ) 
  (h1 : a + b + c = 30) 
  (h2 : a ≥ b) 
  (h3 : b ≥ c) 
  (h4 : a < b + c) 
  (h5 : b < a + c) 
  (h6 : c < a + b) 
  : a ≤ 14 :=
sorry

end NUMINAMATH_GPT_max_side_of_triangle_with_perimeter_30_l1657_165749


namespace NUMINAMATH_GPT_square_side_length_l1657_165746

theorem square_side_length (x y : ℕ) (h_gcd : Nat.gcd x y = 5) (h_area : ∃ a : ℝ, a^2 = (169 / 6) * ↑(Nat.lcm x y)) : ∃ a : ℝ, a = 65 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_l1657_165746


namespace NUMINAMATH_GPT_min_workers_needed_to_make_profit_l1657_165787

def wage_per_worker_per_hour := 20
def fixed_cost := 800
def units_per_worker_per_hour := 6
def price_per_unit := 4.5
def hours_per_workday := 9

theorem min_workers_needed_to_make_profit : ∃ (n : ℕ), 243 * n > 800 + 180 * n ∧ n ≥ 13 :=
by
  sorry

end NUMINAMATH_GPT_min_workers_needed_to_make_profit_l1657_165787


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1657_165730

theorem solution_set_of_inequality (x : ℝ) :
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1657_165730


namespace NUMINAMATH_GPT_jen_total_birds_l1657_165708

-- Define the number of chickens and ducks
variables (C D : ℕ)

-- Define the conditions
def ducks_condition (C D : ℕ) : Prop := D = 4 * C + 10
def num_ducks (D : ℕ) : Prop := D = 150

-- Define the total number of birds
def total_birds (C D : ℕ) : ℕ := C + D

-- Prove that the total number of birds is 185 given the conditions
theorem jen_total_birds (C D : ℕ) (h1 : ducks_condition C D) (h2 : num_ducks D) : total_birds C D = 185 :=
by
  sorry

end NUMINAMATH_GPT_jen_total_birds_l1657_165708


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_geometric_sequence_sum_l1657_165724

section ArithmeticSequence

variable {a_n : ℕ → ℤ} {d : ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a_n (n + 1) - a_n n = d

theorem arithmetic_sequence_general_term (h : is_arithmetic_sequence a_n 2) :
  ∃ a1 : ℤ, ∀ n, a_n n = 2 * n + a1 :=
sorry

end ArithmeticSequence

section GeometricSequence

variable {b_n : ℕ → ℤ} {a_n : ℕ → ℤ}

def is_geometric_sequence_with_reference (b_n : ℕ → ℤ) (a_n : ℕ → ℤ) :=
  b_n 1 = a_n 1 ∧ b_n 2 = a_n 4 ∧ b_n 3 = a_n 13

theorem geometric_sequence_sum (h : is_geometric_sequence_with_reference b_n a_n)
  (h_arith : is_arithmetic_sequence a_n 2) :
  ∃ b1 : ℤ, ∀ n, b_n n = b1 * 3^(n - 1) ∧
                (∃ Sn : ℕ → ℤ, Sn n = (3 * (3^n - 1)) / 2) :=
sorry

end GeometricSequence

end NUMINAMATH_GPT_arithmetic_sequence_general_term_geometric_sequence_sum_l1657_165724


namespace NUMINAMATH_GPT_cards_traded_between_Padma_and_Robert_l1657_165712

def total_cards_traded (padma_first_trade padma_second_trade robert_first_trade robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + padma_second_trade + robert_first_trade + robert_second_trade

theorem cards_traded_between_Padma_and_Robert (h1 : padma_first_trade = 2) 
                                            (h2 : robert_first_trade = 10)
                                            (h3 : padma_second_trade = 15)
                                            (h4 : robert_second_trade = 8) :
                                            total_cards_traded 2 15 10 8 = 35 := 
by 
  sorry

end NUMINAMATH_GPT_cards_traded_between_Padma_and_Robert_l1657_165712


namespace NUMINAMATH_GPT_sugar_per_chocolate_bar_l1657_165780

-- Definitions from conditions
def total_sugar : ℕ := 177
def lollipop_sugar : ℕ := 37
def chocolate_bar_count : ℕ := 14

-- Proof problem statement
theorem sugar_per_chocolate_bar : 
  (total_sugar - lollipop_sugar) / chocolate_bar_count = 10 := 
by 
  sorry

end NUMINAMATH_GPT_sugar_per_chocolate_bar_l1657_165780


namespace NUMINAMATH_GPT_stickers_on_fifth_page_l1657_165790

theorem stickers_on_fifth_page :
  ∀ (stickers : ℕ → ℕ),
    stickers 1 = 8 →
    stickers 2 = 16 →
    stickers 3 = 24 →
    stickers 4 = 32 →
    (∀ n, stickers (n + 1) = stickers n + 8) →
    stickers 5 = 40 :=
by
  intros stickers h1 h2 h3 h4 pattern
  apply sorry

end NUMINAMATH_GPT_stickers_on_fifth_page_l1657_165790


namespace NUMINAMATH_GPT_expenditure_representation_correct_l1657_165715

-- Define the representation of income
def income_representation (income : ℝ) : ℝ :=
  income

-- Define the representation of expenditure
def expenditure_representation (expenditure : ℝ) : ℝ :=
  -expenditure

-- Condition: an income of 10.5 yuan is represented as +10.5 yuan.
-- We need to prove: an expenditure of 6 yuan is represented as -6 yuan.
theorem expenditure_representation_correct (h : income_representation 10.5 = 10.5) : 
  expenditure_representation 6 = -6 :=
by
  sorry

end NUMINAMATH_GPT_expenditure_representation_correct_l1657_165715


namespace NUMINAMATH_GPT_find_center_of_circle_l1657_165726

theorem find_center_of_circle (x y : ℝ) :
  4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 16 = 0 →
  (x + 1)^2 + (y - 3)^2 = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_center_of_circle_l1657_165726


namespace NUMINAMATH_GPT_train_length_is_549_95_l1657_165738

noncomputable def length_of_train 
(speed_of_train : ℝ) -- 63 km/hr
(speed_of_man : ℝ) -- 3 km/hr
(time_to_cross : ℝ) -- 32.997 seconds
: ℝ := 
(speed_of_train - speed_of_man) * (5 / 18) * time_to_cross

theorem train_length_is_549_95 (speed_of_train : ℝ) (speed_of_man : ℝ) (time_to_cross : ℝ) :
    speed_of_train = 63 → speed_of_man = 3 → time_to_cross = 32.997 →
    length_of_train speed_of_train speed_of_man time_to_cross = 549.95 :=
by
  intros h_train h_man h_time
  rw [h_train, h_man, h_time]
  norm_num
  sorry

end NUMINAMATH_GPT_train_length_is_549_95_l1657_165738


namespace NUMINAMATH_GPT_radius_of_fourth_circle_is_12_l1657_165710

theorem radius_of_fourth_circle_is_12 (r : ℝ) (radii : Fin 7 → ℝ) 
  (h_geometric : ∀ i, radii (Fin.succ i) = r * radii i) 
  (h_smallest : radii 0 = 6)
  (h_largest : radii 6 = 24) :
  radii 3 = 12 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_fourth_circle_is_12_l1657_165710


namespace NUMINAMATH_GPT_joseph_total_payment_l1657_165760
-- Importing necessary libraries

-- Defining the variables and conditions
variables (W : ℝ) -- The cost for the water heater

-- Conditions
def condition1 := 3 * W -- The cost for the refrigerator
def condition2 := 2 * W = 500 -- The electric oven
def condition3 := 300 -- The cost for the air conditioner
def condition4 := 100 -- The cost for the washing machine

-- Calculate total cost
def total_cost := (3 * W) + W + 500 + 300 + 100

-- The theorem stating the total amount Joseph pays
theorem joseph_total_payment : total_cost = 1900 :=
by 
  have hW := condition2;
  sorry

end NUMINAMATH_GPT_joseph_total_payment_l1657_165760


namespace NUMINAMATH_GPT_total_pay_XY_l1657_165764

-- Assuming X's pay is 120% of Y's pay and Y's pay is 268.1818181818182,
-- Prove that the total pay to X and Y is 590.00.
theorem total_pay_XY (Y_pay : ℝ) (X_pay : ℝ) (total_pay : ℝ) :
  Y_pay = 268.1818181818182 →
  X_pay = 1.2 * Y_pay →
  total_pay = X_pay + Y_pay →
  total_pay = 590.00 :=
by
  intros hY hX hT
  sorry

end NUMINAMATH_GPT_total_pay_XY_l1657_165764


namespace NUMINAMATH_GPT_ab_equals_six_l1657_165704

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end NUMINAMATH_GPT_ab_equals_six_l1657_165704


namespace NUMINAMATH_GPT_system_of_equations_solution_l1657_165755

theorem system_of_equations_solution (x y : ℤ) (h1 : 2 * x + 5 * y = 26) (h2 : 4 * x - 2 * y = 4) : 
    x = 3 ∧ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1657_165755


namespace NUMINAMATH_GPT_maggie_remaining_goldfish_l1657_165786

theorem maggie_remaining_goldfish
  (total_goldfish : ℕ)
  (allowed_fraction : ℕ → ℕ)
  (caught_fraction : ℕ → ℕ)
  (halfsies : ℕ)
  (remaining_goldfish : ℕ)
  (h1 : total_goldfish = 100)
  (h2 : allowed_fraction total_goldfish = total_goldfish / 2)
  (h3 : caught_fraction (allowed_fraction total_goldfish) = (3 * allowed_fraction total_goldfish) / 5)
  (h4 : halfsies = allowed_fraction total_goldfish)
  (h5 : remaining_goldfish = halfsies - caught_fraction halfsies) :
  remaining_goldfish = 20 :=
sorry

end NUMINAMATH_GPT_maggie_remaining_goldfish_l1657_165786


namespace NUMINAMATH_GPT_field_ratio_l1657_165750

theorem field_ratio (w l: ℕ) (h: l = 36)
  (h_area_ratio: 81 = (1/8) * l * w)
  (h_multiple: ∃ k : ℕ, l = k * w) :
  l / w = 2 :=
by 
  sorry

end NUMINAMATH_GPT_field_ratio_l1657_165750


namespace NUMINAMATH_GPT_polynomial_value_l1657_165711

theorem polynomial_value (x y : ℝ) (h : x + 2 * y = 6) : 2 * x + 4 * y - 5 = 7 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l1657_165711


namespace NUMINAMATH_GPT_find_months_contributed_l1657_165706

theorem find_months_contributed (x : ℕ) (profit_A profit_total : ℝ)
  (contrib_A : ℝ) (contrib_B : ℝ) (months_B : ℕ) :
  profit_A / profit_total = (contrib_A * x) / (contrib_A * x + contrib_B * months_B) →
  profit_A = 4800 →
  profit_total = 8400 →
  contrib_A = 5000 →
  contrib_B = 6000 →
  months_B = 5 →
  x = 8 :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end NUMINAMATH_GPT_find_months_contributed_l1657_165706


namespace NUMINAMATH_GPT_probability_green_light_is_8_over_15_l1657_165789

def total_cycle_duration (red yellow green : ℕ) : ℕ :=
  red + yellow + green

def probability_green_light (red yellow green : ℕ) : ℚ :=
  green / (total_cycle_duration red yellow green : ℚ)

theorem probability_green_light_is_8_over_15 :
  probability_green_light 30 5 40 = 8 / 15 := by
  sorry

end NUMINAMATH_GPT_probability_green_light_is_8_over_15_l1657_165789


namespace NUMINAMATH_GPT_count_valid_b_values_l1657_165781

-- Definitions of the inequalities and the condition
def inequality1 (x : ℤ) : Prop := 3 * x > 4 * x - 4
def inequality2 (x b: ℤ) : Prop := 4 * x - b > -8

-- The main statement proving that the count of valid b values is 4
theorem count_valid_b_values (x b : ℤ) (h1 : inequality1 x) (h2 : inequality2 x b) :
  ∃ (b_values : Finset ℤ), 
    ((∀ b' ∈ b_values, ∀ x' : ℤ, inequality2 x' b' → x' ≠ 3) ∧ 
     (∀ b' ∈ b_values, 16 ≤ b' ∧ b' < 20) ∧ 
     b_values.card = 4) := by
  sorry

end NUMINAMATH_GPT_count_valid_b_values_l1657_165781


namespace NUMINAMATH_GPT_paul_sandwiches_in_6_days_l1657_165784

def sandwiches_eaten_in_n_days (n : ℕ) : ℕ :=
  let day1 := 2
  let day2 := 2 * day1
  let day3 := 2 * day2
  let three_day_total := day1 + day2 + day3
  three_day_total * (n / 3)

theorem paul_sandwiches_in_6_days : sandwiches_eaten_in_n_days 6 = 28 :=
by
  sorry

end NUMINAMATH_GPT_paul_sandwiches_in_6_days_l1657_165784


namespace NUMINAMATH_GPT_serpent_ridge_trail_length_l1657_165751

/-- Phoenix hiked the Serpent Ridge Trail last week. It took her five days to complete the trip.
The first two days she hiked a total of 28 miles. The second and fourth days she averaged 15 miles per day.
The last three days she hiked a total of 42 miles. The total hike for the first and third days was 30 miles.
How many miles long was the trail? -/
theorem serpent_ridge_trail_length
  (a b c d e : ℕ)
  (h1 : a + b = 28)
  (h2 : b + d = 30)
  (h3 : c + d + e = 42)
  (h4 : a + c = 30) :
  a + b + c + d + e = 70 :=
sorry

end NUMINAMATH_GPT_serpent_ridge_trail_length_l1657_165751


namespace NUMINAMATH_GPT_constants_unique_l1657_165756

theorem constants_unique (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → (5 * x) / ((x - 4) * (x - 2) ^ 2) = A / (x - 4) + B / (x - 2) + C / (x - 2) ^ 2) ↔
  A = 5 ∧ B = -5 ∧ C = -5 :=
by
  sorry

end NUMINAMATH_GPT_constants_unique_l1657_165756


namespace NUMINAMATH_GPT_no_two_perfect_cubes_between_two_perfect_squares_l1657_165783

theorem no_two_perfect_cubes_between_two_perfect_squares :
  ∀ n a b : ℤ, n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2 → False :=
by 
  sorry

end NUMINAMATH_GPT_no_two_perfect_cubes_between_two_perfect_squares_l1657_165783


namespace NUMINAMATH_GPT_general_term_arithmetic_sum_first_n_terms_geometric_l1657_165716

-- Definitions and assumptions based on given conditions
def a (n : ℕ) : ℤ := 2 * n + 1

-- Given conditions
def initial_a1 : ℤ := 3
def common_difference : ℤ := 2

-- Validate the general formula for the arithmetic sequence
theorem general_term_arithmetic : ∀ n : ℕ, a n = 2 * n + 1 := 
by sorry

-- Definitions and assumptions for geometric sequence
def b (n : ℕ) : ℤ := 3^n

-- Sum of the first n terms of the geometric sequence
def Sn (n : ℕ) : ℤ := 3 / 2 * (3^n - 1)

-- Validate the sum formula for the geometric sequence
theorem sum_first_n_terms_geometric (n : ℕ) : Sn n = 3 / 2 * (3^n - 1) := 
by sorry

end NUMINAMATH_GPT_general_term_arithmetic_sum_first_n_terms_geometric_l1657_165716


namespace NUMINAMATH_GPT_solve_equation_l1657_165753

theorem solve_equation (x : ℝ) : 
  (x - 4)^6 + (x - 6)^6 = 64 → x = 4 ∨ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1657_165753


namespace NUMINAMATH_GPT_completing_the_square_equation_l1657_165717

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_completing_the_square_equation_l1657_165717


namespace NUMINAMATH_GPT_fixed_point_of_function_l1657_165700

theorem fixed_point_of_function (a : ℝ) : 
  (a - 1) * 2^1 - 2 * a = -2 := by
  sorry

end NUMINAMATH_GPT_fixed_point_of_function_l1657_165700


namespace NUMINAMATH_GPT_snack_bar_training_count_l1657_165762

noncomputable def num_trained_in_snack_bar 
  (total_employees : ℕ) 
  (trained_in_buffet : ℕ) 
  (trained_in_dining_room : ℕ) 
  (trained_in_two_restaurants : ℕ) 
  (trained_in_three_restaurants : ℕ) : ℕ :=
  total_employees - trained_in_buffet - trained_in_dining_room + 
  trained_in_two_restaurants + trained_in_three_restaurants

theorem snack_bar_training_count : 
  num_trained_in_snack_bar 39 17 18 4 2 = 8 :=
sorry

end NUMINAMATH_GPT_snack_bar_training_count_l1657_165762


namespace NUMINAMATH_GPT_adjusted_volume_bowling_ball_l1657_165767

noncomputable def bowling_ball_diameter : ℝ := 40
noncomputable def hole1_diameter : ℝ := 5
noncomputable def hole1_depth : ℝ := 10
noncomputable def hole2_diameter : ℝ := 4
noncomputable def hole2_depth : ℝ := 12
noncomputable def expected_adjusted_volume : ℝ := 10556.17 * Real.pi

theorem adjusted_volume_bowling_ball :
  let radius := bowling_ball_diameter / 2
  let volume_ball := (4 / 3) * Real.pi * radius^3
  let hole1_radius := hole1_diameter / 2
  let hole1_volume := Real.pi * hole1_radius^2 * hole1_depth
  let hole2_radius := hole2_diameter / 2
  let hole2_volume := Real.pi * hole2_radius^2 * hole2_depth
  let adjusted_volume := volume_ball - hole1_volume - hole2_volume
  adjusted_volume = expected_adjusted_volume :=
by
  sorry

end NUMINAMATH_GPT_adjusted_volume_bowling_ball_l1657_165767


namespace NUMINAMATH_GPT_common_elements_count_l1657_165771

theorem common_elements_count (S T : Set ℕ) (hS : S = {n | ∃ k : ℕ, k < 3000 ∧ n = 5 * (k + 1)})
    (hT : T = {n | ∃ k : ℕ, k < 3000 ∧ n = 8 * (k + 1)}) :
    S ∩ T = {n | ∃ m : ℕ, m < 375 ∧ n = 40 * (m + 1)} :=
by {
  sorry
}

end NUMINAMATH_GPT_common_elements_count_l1657_165771


namespace NUMINAMATH_GPT_range_of_a_l1657_165763

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (2 * x - 3) - 2 * a ≥ abs (x + a)) ↔ ( -3/2 ≤ a ∧ a < -1/2) := 
by sorry

end NUMINAMATH_GPT_range_of_a_l1657_165763


namespace NUMINAMATH_GPT_magnitude_diff_is_correct_l1657_165777

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b : ℝ × ℝ × ℝ := (-1, 1, -4)

theorem magnitude_diff_is_correct : 
  ‖(2, -3, 1) - (-1, 1, -4)‖ = 5 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_magnitude_diff_is_correct_l1657_165777


namespace NUMINAMATH_GPT_M_eq_N_l1657_165788

def M : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = (5/6) * Real.pi + 2 * k * Real.pi}
def N : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = -(7/6) * Real.pi + 2 * k * Real.pi}

theorem M_eq_N : M = N := 
by {
  sorry
}

end NUMINAMATH_GPT_M_eq_N_l1657_165788


namespace NUMINAMATH_GPT_jerry_more_votes_l1657_165791

-- Definitions based on conditions
def total_votes : ℕ := 196554
def jerry_votes : ℕ := 108375
def john_votes : ℕ := total_votes - jerry_votes

-- Theorem to prove the number of more votes Jerry received than John Pavich
theorem jerry_more_votes : jerry_votes - john_votes = 20196 :=
by
  -- Definitions and proof can be filled out here as required.
  sorry

end NUMINAMATH_GPT_jerry_more_votes_l1657_165791


namespace NUMINAMATH_GPT_product_of_smallest_primes_l1657_165713

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end NUMINAMATH_GPT_product_of_smallest_primes_l1657_165713


namespace NUMINAMATH_GPT_f_nonnegative_when_a_ge_one_l1657_165754

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

noncomputable def h (a : ℝ) : ℝ := Real.log a + 1 - (1 / a)

theorem f_nonnegative_when_a_ge_one (a : ℝ) (x : ℝ) (h_a : a ≥ 1) : f a x ≥ 0 := by
  sorry  -- Placeholder for the proof.

end NUMINAMATH_GPT_f_nonnegative_when_a_ge_one_l1657_165754


namespace NUMINAMATH_GPT_height_on_hypotenuse_correct_l1657_165770

noncomputable def height_on_hypotenuse (a b : ℝ) (ha : a = 3) (hb : b = 4) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let area := (a * b) / 2
  (2 * area) / c

theorem height_on_hypotenuse_correct (h : ℝ) : 
  height_on_hypotenuse 3 4 rfl rfl = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_height_on_hypotenuse_correct_l1657_165770


namespace NUMINAMATH_GPT_nina_money_l1657_165718

theorem nina_money :
  ∃ (m C : ℝ), 
    m = 6 * C ∧ 
    m = 8 * (C - 1) ∧ 
    m = 24 :=
by
  sorry

end NUMINAMATH_GPT_nina_money_l1657_165718


namespace NUMINAMATH_GPT_no_t_for_xyz_equal_l1657_165797

theorem no_t_for_xyz_equal (t : ℝ) (x y z : ℝ) : 
  (x = 1 - 3 * t) → 
  (y = 2 * t - 3) → 
  (z = 4 * t^2 - 5 * t + 1) → 
  ¬ (x = y ∧ y = z) := 
by
  intro h1 h2 h3 h4
  have h5 : t = 4 / 5 := 
    by linarith [h1, h2, h4]
  rw [h5] at h3
  sorry

end NUMINAMATH_GPT_no_t_for_xyz_equal_l1657_165797


namespace NUMINAMATH_GPT_original_denominator_is_15_l1657_165725

theorem original_denominator_is_15
  (d : ℕ)
  (h1 : (6 : ℚ) / (d + 3) = 1 / 3)
  (h2 : 3 = 3) : d = 15 := -- h2 is trivial but included according to the problem condition
by
  sorry

end NUMINAMATH_GPT_original_denominator_is_15_l1657_165725


namespace NUMINAMATH_GPT_expressible_numbers_count_l1657_165721

theorem expressible_numbers_count : ∃ k : ℕ, k = 2222 ∧ ∀ n : ℕ, n ≤ 2000 → ∃ x : ℝ, n = Int.floor x + Int.floor (3 * x) + Int.floor (5 * x) :=
by sorry

end NUMINAMATH_GPT_expressible_numbers_count_l1657_165721


namespace NUMINAMATH_GPT_k_value_l1657_165773

theorem k_value (m n k : ℤ) (h₁ : m + 2 * n + 5 = 0) (h₂ : (m + 2) + 2 * (n + k) + 5 = 0) : k = -1 :=
by sorry

end NUMINAMATH_GPT_k_value_l1657_165773


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1657_165741

theorem necessary_and_sufficient_condition (a b : ℝ) : a^2 * b > a * b^2 ↔ 1/a < 1/b := 
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1657_165741


namespace NUMINAMATH_GPT_inverse_function_value_l1657_165799

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^3 + 4) :
  f⁻¹ 58 = 3 :=
by sorry

end NUMINAMATH_GPT_inverse_function_value_l1657_165799


namespace NUMINAMATH_GPT_average_operating_time_l1657_165757

-- Definition of problem conditions
def cond1 : Nat := 5 -- originally had 5 air conditioners
def cond2 : Nat := 6 -- after installing 1 more
def total_hours : Nat := 24 * 5 -- total operating hours allowable in 24 hours

-- Formalize the average operating time calculation
theorem average_operating_time : (total_hours / cond2) = 20 := by
  sorry

end NUMINAMATH_GPT_average_operating_time_l1657_165757


namespace NUMINAMATH_GPT_target_runs_correct_l1657_165734

noncomputable def target_runs (run_rate1 : ℝ) (ovs1 : ℕ) (run_rate2 : ℝ) (ovs2 : ℕ) : ℝ :=
  (run_rate1 * ovs1) + (run_rate2 * ovs2)

theorem target_runs_correct : target_runs 4.5 12 8.052631578947368 38 = 360 :=
by
  sorry

end NUMINAMATH_GPT_target_runs_correct_l1657_165734


namespace NUMINAMATH_GPT_nat_solution_unique_l1657_165705

theorem nat_solution_unique (n : ℕ) (h : 2 * n - 1 / n^5 = 3 - 2 / n) : 
  n = 1 :=
sorry

end NUMINAMATH_GPT_nat_solution_unique_l1657_165705


namespace NUMINAMATH_GPT_test_total_questions_l1657_165761

theorem test_total_questions (total_points : ℕ) (num_5_point_questions : ℕ) (points_per_5_point_question : ℕ) (points_per_10_point_question : ℕ) : 
  total_points = 200 → 
  num_5_point_questions = 20 → 
  points_per_5_point_question = 5 → 
  points_per_10_point_question = 10 → 
  (total_points = (num_5_point_questions * points_per_5_point_question) + 
    ((total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) * points_per_10_point_question) →
  (num_5_point_questions + (total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_test_total_questions_l1657_165761


namespace NUMINAMATH_GPT_four_digit_palindromic_squares_with_different_middle_digits_are_zero_l1657_165776

theorem four_digit_palindromic_squares_with_different_middle_digits_are_zero :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ k, k * k = n) ∧ (∃ a b, n = 1001 * a + 110 * b) → a ≠ b → false :=
by sorry

end NUMINAMATH_GPT_four_digit_palindromic_squares_with_different_middle_digits_are_zero_l1657_165776


namespace NUMINAMATH_GPT_mul_mod_l1657_165766

theorem mul_mod (n1 n2 n3 : ℤ) (h1 : n1 = 2011) (h2 : n2 = 1537) (h3 : n3 = 450) : 
  (2011 * 1537) % 450 = 307 := by
  sorry

end NUMINAMATH_GPT_mul_mod_l1657_165766


namespace NUMINAMATH_GPT_minimum_trucks_needed_l1657_165701

theorem minimum_trucks_needed {n : ℕ} (total_weight : ℕ) (box_weight : ℕ → ℕ) (truck_capacity : ℕ) :
  (total_weight = 10 ∧ truck_capacity = 3 ∧ (∀ b, box_weight b ≤ 1) ∧ (∃ n, 3 * n ≥ total_weight)) → n ≥ 5 :=
by
  -- We need to prove the statement based on the given conditions.
  sorry

end NUMINAMATH_GPT_minimum_trucks_needed_l1657_165701


namespace NUMINAMATH_GPT_greatest_value_k_l1657_165752

theorem greatest_value_k (k : ℝ) (h : ∀ x : ℝ, (x - 1) ∣ (x^2 + 2*k*x - 3*k^2)) : k ≤ 1 :=
  by
  sorry

end NUMINAMATH_GPT_greatest_value_k_l1657_165752


namespace NUMINAMATH_GPT_factor_expression_l1657_165722

theorem factor_expression (b : ℚ) : 
  294 * b^3 + 63 * b^2 - 21 * b = 21 * b * (14 * b^2 + 3 * b - 1) :=
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l1657_165722


namespace NUMINAMATH_GPT_fraction_of_tips_l1657_165727

variable (S T : ℝ) -- assuming S is salary and T is tips
variable (h : T / (S + T) = 0.7142857142857143)

/-- 
If the fraction of the waiter's income from tips is 0.7142857142857143,
then the fraction of his salary that were his tips is 2.5.
-/
theorem fraction_of_tips (h : T / (S + T) = 0.7142857142857143) : T / S = 2.5 :=
sorry

end NUMINAMATH_GPT_fraction_of_tips_l1657_165727


namespace NUMINAMATH_GPT_shorter_side_length_l1657_165795

theorem shorter_side_length 
  (L W : ℝ) 
  (h1 : L * W = 117) 
  (h2 : 2 * L + 2 * W = 44) :
  L = 9 ∨ W = 9 :=
by
  sorry

end NUMINAMATH_GPT_shorter_side_length_l1657_165795


namespace NUMINAMATH_GPT_area_of_region_a_area_of_region_b_area_of_region_c_l1657_165737

-- Definition of regions and their areas
def area_of_square : Real := sorry
def area_of_diamond : Real := sorry
def area_of_hexagon : Real := sorry

-- Define the conditions for the regions
def region_a (x y : ℝ) := abs x ≤ 1 ∧ abs y ≤ 1
def region_b (x y : ℝ) := abs x + abs y ≤ 10
def region_c (x y : ℝ) := abs x + abs y + abs (x + y) ≤ 2020

-- Prove that the areas match the calculated solutions
theorem area_of_region_a : area_of_square = 4 := 
by sorry

theorem area_of_region_b : area_of_diamond = 200 := 
by sorry

theorem area_of_region_c : area_of_hexagon = 3060300 := 
by sorry

end NUMINAMATH_GPT_area_of_region_a_area_of_region_b_area_of_region_c_l1657_165737


namespace NUMINAMATH_GPT_chocolate_bar_min_breaks_l1657_165745

theorem chocolate_bar_min_breaks (n : ℕ) (h : n = 40) : ∃ k : ℕ, k = n - 1 := 
by 
  sorry

end NUMINAMATH_GPT_chocolate_bar_min_breaks_l1657_165745


namespace NUMINAMATH_GPT_tim_words_per_day_l1657_165794

variable (original_words : ℕ)
variable (years : ℕ)
variable (increase_percent : ℚ)

noncomputable def words_per_day (original_words : ℕ) (years : ℕ) (increase_percent : ℚ) : ℚ :=
  let increase_words := original_words * increase_percent
  let total_days := years * 365
  increase_words / total_days

theorem tim_words_per_day :
    words_per_day 14600 2 (50 / 100) = 10 := by
  sorry

end NUMINAMATH_GPT_tim_words_per_day_l1657_165794


namespace NUMINAMATH_GPT_inequality_proof_l1657_165702

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c)) ≥ 1 / 3 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1657_165702


namespace NUMINAMATH_GPT_fraction_of_6_l1657_165759

theorem fraction_of_6 (x y : ℕ) (h : (x / y : ℚ) * 6 + 6 = 10) : (x / y : ℚ) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_6_l1657_165759


namespace NUMINAMATH_GPT_rolling_cube_dot_path_l1657_165720

theorem rolling_cube_dot_path (a b c : ℝ) (h_edge : a = 1) (h_dot_top : True):
  c = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_GPT_rolling_cube_dot_path_l1657_165720
