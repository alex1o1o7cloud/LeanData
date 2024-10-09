import Mathlib

namespace scientific_notation_of_probe_unit_area_l893_89311

def probe_unit_area : ℝ := 0.0000064

theorem scientific_notation_of_probe_unit_area :
  ∃ (mantissa : ℝ) (exponent : ℤ), probe_unit_area = mantissa * 10^exponent ∧ mantissa = 6.4 ∧ exponent = -6 :=
by
  sorry

end scientific_notation_of_probe_unit_area_l893_89311


namespace complex_number_is_real_implies_m_eq_3_l893_89362

open Complex

theorem complex_number_is_real_implies_m_eq_3 (m : ℝ) :
  (∃ (z : ℂ), z = (1 / (m + 5) : ℝ) + (m^2 + 2 * m - 15) * I ∧ z.im = 0) →
  m = 3 :=
by
  sorry

end complex_number_is_real_implies_m_eq_3_l893_89362


namespace floss_leftover_l893_89374

noncomputable def leftover_floss
    (students : ℕ)
    (floss_per_student : ℚ)
    (floss_per_packet : ℚ) :
    ℚ :=
  let total_needed := students * floss_per_student
  let packets_needed := (total_needed / floss_per_packet).ceil
  let total_floss := packets_needed * floss_per_packet
  total_floss - total_needed

theorem floss_leftover {students : ℕ} {floss_per_student floss_per_packet : ℚ}
    (h_students : students = 20)
    (h_floss_per_student : floss_per_student = 3 / 2)
    (h_floss_per_packet : floss_per_packet = 35) :
    leftover_floss students floss_per_student floss_per_packet = 5 :=
by
  rw [h_students, h_floss_per_student, h_floss_per_packet]
  simp only [leftover_floss]
  norm_num
  sorry

end floss_leftover_l893_89374


namespace tangent_line_eq_l893_89366

theorem tangent_line_eq (f : ℝ → ℝ) (f' : ℝ → ℝ) (x y : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, f' x = Real.exp x) →
  f 0 = 1 →
  f' 0 = 1 →
  x = 0 →
  y = 1 →
  x - y + 1 = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end tangent_line_eq_l893_89366


namespace mark_second_part_playtime_l893_89353

theorem mark_second_part_playtime (total_time initial_time sideline_time : ℕ) 
  (h1 : total_time = 90) (h2 : initial_time = 20) (h3 : sideline_time = 35) :
  total_time - initial_time - sideline_time = 35 :=
sorry

end mark_second_part_playtime_l893_89353


namespace not_equivalent_expression_l893_89397

/--
Let A, B, C, D be expressions defined as follows:
A := 3 * (x + 2)
B := (-9 * x - 18) / -3
C := (1/3) * (3 * x) + (2/3) * 9
D := (1/3) * (9 * x + 18)

Prove that only C is not equivalent to 3 * x + 6.
-/
theorem not_equivalent_expression (x : ℝ) :
  let A := 3 * (x + 2)
  let B := (-9 * x - 18) / -3
  let C := (1/3) * (3 * x) + (2/3) * 9
  let D := (1/3) * (9 * x + 18)
  C ≠ 3 * x + 6 :=
by
  intros A B C D
  sorry

end not_equivalent_expression_l893_89397


namespace profit_when_sold_at_double_price_l893_89396

-- Define the problem parameters

-- Assume cost price (CP)
def CP : ℕ := 100

-- Define initial selling price (SP) with 50% profit
def SP : ℕ := CP + (CP / 2)

-- Define new selling price when sold at double the initial selling price
def SP2 : ℕ := 2 * SP

-- Define profit when sold at SP2
def profit : ℕ := SP2 - CP

-- Define the percentage profit
def profit_percentage : ℕ := (profit * 100) / CP

-- The proof goal: if selling at double the price, percentage profit is 200%
theorem profit_when_sold_at_double_price : profit_percentage = 200 :=
by {sorry}

end profit_when_sold_at_double_price_l893_89396


namespace greatest_four_digit_divisible_by_3_5_6_l893_89325

theorem greatest_four_digit_divisible_by_3_5_6 : 
  ∃ n, n ≤ 9999 ∧ n ≥ 1000 ∧ (∀ m, m ≤ 9999 ∧ m ≥ 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n) ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n = 9990 :=
by 
  sorry

end greatest_four_digit_divisible_by_3_5_6_l893_89325


namespace samuel_faster_than_sarah_l893_89300

theorem samuel_faster_than_sarah
  (efficiency_samuel : ℝ := 0.90)
  (efficiency_sarah : ℝ := 0.75)
  (efficiency_tim : ℝ := 0.80)
  (time_tim : ℝ := 45)
  : (time_tim * efficiency_tim / efficiency_sarah) - (time_tim * efficiency_tim / efficiency_samuel) = 8 :=
by
  sorry

end samuel_faster_than_sarah_l893_89300


namespace g_neither_even_nor_odd_l893_89367

noncomputable def g (x : ℝ) : ℝ := ⌊x⌋ + 1/2 + Real.sin x

theorem g_neither_even_nor_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) := sorry

end g_neither_even_nor_odd_l893_89367


namespace select_pencils_l893_89345

theorem select_pencils (boxes : Fin 10 → ℕ) (colors : ∀ (i : Fin 10), Fin (boxes i) → Fin 10) :
  (∀ i : Fin 10, 1 ≤ boxes i) → -- Each box is non-empty
  (∀ i j : Fin 10, i ≠ j → boxes i ≠ boxes j) → -- Different number of pencils in each box
  ∃ (selection : Fin 10 → Fin 10), -- Function to select a pencil color from each box
  Function.Injective selection := -- All selected pencils have different colors
sorry

end select_pencils_l893_89345


namespace trigonometric_identities_l893_89323

theorem trigonometric_identities (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : Real.sin α = 4 / 5) :
    (Real.tan α = 4 / 3) ∧ 
    ((Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) :=
by
  sorry

end trigonometric_identities_l893_89323


namespace probability_at_least_two_meters_l893_89338

def rope_length : ℝ := 6
def num_nodes : ℕ := 5
def equal_parts : ℕ := 6
def min_length : ℝ := 2

theorem probability_at_least_two_meters (h_rope_division : rope_length / equal_parts = 1) :
  let favorable_cuts := 3
  let total_cuts := num_nodes
  (favorable_cuts : ℝ) / total_cuts = 3 / 5 :=
by
  sorry

end probability_at_least_two_meters_l893_89338


namespace find_fraction_l893_89357

theorem find_fraction (F N : ℝ) 
  (h1 : F * (1 / 4 * N) = 15)
  (h2 : (3 / 10) * N = 54) : 
  F = 1 / 3 := 
by
  sorry

end find_fraction_l893_89357


namespace range_of_k_l893_89313

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 3| + |x - 1| > k) ↔ k < 4 :=
by sorry

end range_of_k_l893_89313


namespace alice_oranges_l893_89319

theorem alice_oranges (E A : ℕ) 
  (h1 : A = 2 * E) 
  (h2 : E + A = 180) : 
  A = 120 :=
by
  sorry

end alice_oranges_l893_89319


namespace remainder_of_division_l893_89315

noncomputable def dividend : Polynomial ℤ := Polynomial.C 1 * Polynomial.X^4 +
                                             Polynomial.C 3 * Polynomial.X^2 + 
                                             Polynomial.C (-4)

noncomputable def divisor : Polynomial ℤ := Polynomial.C 1 * Polynomial.X^3 +
                                            Polynomial.C (-3)

theorem remainder_of_division :
  Polynomial.modByMonic dividend divisor = Polynomial.C 3 * Polynomial.X^2 +
                                            Polynomial.C 3 * Polynomial.X +
                                            Polynomial.C (-4) :=
by
  sorry

end remainder_of_division_l893_89315


namespace value_of_expression_l893_89350

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 = 1) 
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 = 12) 
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 = 123) 
  : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 = 334 :=
by
  sorry

end value_of_expression_l893_89350


namespace find_x_for_vectors_l893_89351

theorem find_x_for_vectors
  (x : ℝ)
  (h1 : x ∈ Set.Icc 0 Real.pi)
  (a : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2)))
  (b : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2)))
  (h2 : (a.1 + b.1)^2 + (a.2 + b.2)^2 = 1) :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end find_x_for_vectors_l893_89351


namespace john_got_80_percent_of_value_l893_89373

noncomputable def percentage_of_value (P : ℝ) : Prop :=
  let old_system_cost := 250
  let new_system_cost := 600
  let discount_percentage := 0.25
  let pocket_spent := 250
  let discount_amount := discount_percentage * new_system_cost
  let price_after_discount := new_system_cost - discount_amount
  let value_for_old_system := (P / 100) * old_system_cost
  value_for_old_system + pocket_spent = price_after_discount

theorem john_got_80_percent_of_value : percentage_of_value 80 :=
by
  sorry

end john_got_80_percent_of_value_l893_89373


namespace F_of_3153_max_value_of_N_l893_89303

-- Define friendly number predicate
def is_friendly (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  a - b = c - d

-- Define F(M)
def F (M : ℕ) : ℕ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let s := M / 10
  let t := M % 1000
  s - t - 10 * b

-- Prove F(3153) = 152
theorem F_of_3153 : F 3153 = 152 :=
by sorry

-- Define the given predicate for N
def is_k_special (N : ℕ) : Prop :=
  let x := N / 1000
  let y := (N / 100) % 10
  let m := (N / 30) % 10
  let n := N % 10
  (N % 5 = 1) ∧ (1000 * x + 100 * y + 30 * m + n + 1001 = N) ∧
  (0 ≤ y ∧ y < x ∧ x ≤ 8) ∧ (0 ≤ m ∧ m ≤ 3) ∧ (0 ≤ n ∧ n ≤ 8) ∧ 
  is_friendly N

-- Prove the maximum value satisfying the given constraints
theorem max_value_of_N : ∀ N, is_k_special N → N ≤ 9696 :=
by sorry

end F_of_3153_max_value_of_N_l893_89303


namespace amount_needed_for_free_delivery_l893_89321

theorem amount_needed_for_free_delivery :
  let chicken_cost := 1.5 * 6.00
  let lettuce_cost := 3.00
  let tomatoes_cost := 2.50
  let sweet_potatoes_cost := 4 * 0.75
  let broccoli_cost := 2 * 2.00
  let brussel_sprouts_cost := 2.50
  let total_cost := chicken_cost + lettuce_cost + tomatoes_cost + sweet_potatoes_cost + broccoli_cost + brussel_sprouts_cost
  let min_spend_for_free_delivery := 35.00
  min_spend_for_free_delivery - total_cost = 11.00 := sorry

end amount_needed_for_free_delivery_l893_89321


namespace no_integer_pair_satisfies_conditions_l893_89318

theorem no_integer_pair_satisfies_conditions :
  ¬ ∃ (x y : ℤ), x = x^2 + y^2 + 1 ∧ y = 3 * x * y := 
by
  sorry

end no_integer_pair_satisfies_conditions_l893_89318


namespace y_intercept_of_line_l893_89339

theorem y_intercept_of_line (m x1 y1 : ℝ) (x_intercept : x1 = 4) (y_intercept_at_x1_zero : y1 = 0) (m_value : m = -3) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ∧ x = 0 → y = b) ∧ b = 12 :=
by
  sorry

end y_intercept_of_line_l893_89339


namespace number_of_boys_took_exam_l893_89309

theorem number_of_boys_took_exam (T F : ℕ) (h_avg_all : 35 * T = 39 * 100 + 15 * F)
                                (h_total_boys : T = 100 + F) : T = 120 :=
sorry

end number_of_boys_took_exam_l893_89309


namespace petya_coloring_l893_89334

theorem petya_coloring (k : ℕ) : k = 1 :=
  sorry

end petya_coloring_l893_89334


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l893_89388

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l893_89388


namespace log_sum_eval_l893_89308

theorem log_sum_eval :
  (Real.logb 5 625 + Real.logb 5 5 - Real.logb 5 (1 / 25)) = 7 :=
by
  have h1 : Real.logb 5 625 = 4 := by sorry
  have h2 : Real.logb 5 5 = 1 := by sorry
  have h3 : Real.logb 5 (1 / 25) = -2 := by sorry
  rw [h1, h2, h3]
  norm_num

end log_sum_eval_l893_89308


namespace first_term_of_arithmetic_series_l893_89352

theorem first_term_of_arithmetic_series 
  (a d : ℝ)
  (h1 : 20 * (2 * a + 39 * d) = 600)
  (h2 : 20 * (2 * a + 119 * d) = 1800) :
  a = 0.375 :=
by
  sorry

end first_term_of_arithmetic_series_l893_89352


namespace distinct_cubes_meet_condition_l893_89370

theorem distinct_cubes_meet_condition :
  ∃ (a b c d e f : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a + b + c + d + e + f = 60) ∧
    ∃ (k : ℕ), 
        ((a = k) ∧ (b = k) ∧ (c = k) ∧ (d = k) ∧ (e = k) ∧ (f = k)) ∧
        -- Number of distinct ways
        (∃ (num_ways : ℕ), num_ways = 84) :=
sorry

end distinct_cubes_meet_condition_l893_89370


namespace five_million_squared_l893_89342

theorem five_million_squared : (5 * 10^6)^2 = 25 * 10^12 := by
  sorry

end five_million_squared_l893_89342


namespace ratio_pentagon_rectangle_l893_89363

theorem ratio_pentagon_rectangle (P: ℝ) (a w: ℝ) (h1: 5 * a = P) (h2: 6 * w = P) (h3: P = 75) : a / w = 6 / 5 := 
by 
  -- Proof steps will be provided to conclude this result 
  sorry

end ratio_pentagon_rectangle_l893_89363


namespace positive_integer_cases_l893_89394

theorem positive_integer_cases (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℤ, (abs (x^2 - abs x)) / x = n ∧ n > 0) ↔ (∃ m : ℤ, (x = m) ∧ (m > 1 ∨ m < -1)) :=
by
  sorry

end positive_integer_cases_l893_89394


namespace real_number_a_l893_89304

theorem real_number_a (a : ℝ) (ha : ∃ b : ℝ, z = 0 + bi) : a = 1 :=
sorry

end real_number_a_l893_89304


namespace constant_two_l893_89344

theorem constant_two (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) (c : ℕ) (n : ℕ) (h_n : n = c * p) (h_even_divisors : ∀ d : ℕ, d ∣ n → (d % 2 = 0) → d = 2) : c = 2 := by
  sorry

end constant_two_l893_89344


namespace chocolate_syrup_amount_l893_89364

theorem chocolate_syrup_amount (x : ℝ) (H1 : 2 * x + 6 = 14) : x = 4 :=
by
  sorry

end chocolate_syrup_amount_l893_89364


namespace matches_length_l893_89387

-- Definitions and conditions
def area_shaded_figure : ℝ := 300 -- given in cm^2
def num_small_squares : ℕ := 8
def large_square_area_coefficient : ℕ := 4
def area_small_square (a : ℝ) : ℝ := num_small_squares * a + large_square_area_coefficient * a

-- Question and answer to be proven
theorem matches_length (a : ℝ) (side_length: ℝ) :
  area_shaded_figure = 300 → 
  area_small_square a = area_shaded_figure →
  (a = 25) →
  (side_length = 5) →
  4 * 7 * side_length = 140 :=
by
  intros h1 h2 h3 h4
  sorry

end matches_length_l893_89387


namespace total_colors_needed_l893_89332

def num_planets : ℕ := 8
def num_people : ℕ := 3

theorem total_colors_needed : num_people * num_planets = 24 := by
  sorry

end total_colors_needed_l893_89332


namespace solution_to_diff_eq_l893_89348

def y (x C : ℝ) : ℝ := x^2 + x + C

theorem solution_to_diff_eq (C : ℝ) : ∀ x : ℝ, 
  (dy = (2 * x + 1) * dx) :=
by
  sorry

end solution_to_diff_eq_l893_89348


namespace parabola_tangent_circle_l893_89368

noncomputable def parabola_focus (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p / 2, 0)

theorem parabola_tangent_circle (p : ℝ) (hp : p > 0)
  (x0 : ℝ) (hx0 : x0 = p)
  (M : ℝ × ℝ) (hM : M = (x0, 2 * (Real.sqrt 2)))
  (MA AF : ℝ) (h_ratio : MA / AF = 2) :
  p = 2 :=
by
  sorry

end parabola_tangent_circle_l893_89368


namespace vacation_cost_l893_89393

theorem vacation_cost (n : ℕ) (h : 480 / n + 40 = 120) : n = 6 :=
sorry

end vacation_cost_l893_89393


namespace custom_operation_example_l893_89375

-- Define the custom operation
def custom_operation (a b : ℕ) : ℕ := a * b + (a - b)

-- State the theorem
theorem custom_operation_example : custom_operation (custom_operation 3 2) 4 = 31 :=
by
  -- the proof will go here, but we skip it for now
  sorry

end custom_operation_example_l893_89375


namespace length_of_string_C_l893_89355

theorem length_of_string_C (A B C : ℕ) (h1 : A = 6 * C) (h2 : A = 5 * B) (h3 : B = 12) : C = 10 :=
sorry

end length_of_string_C_l893_89355


namespace find_z_l893_89340

theorem find_z (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : z * i = 2 - i) : z = -1 - 2 * i := 
by
  sorry

end find_z_l893_89340


namespace range_a_l893_89359

theorem range_a (a : ℝ) (x : ℝ) : 
    (∀ x, (x = 1 → x - a ≥ 1) ∧ (x = -1 → ¬(x - a ≥ 1))) ↔ (-2 < a ∧ a ≤ 0) :=
by
  sorry

end range_a_l893_89359


namespace largest_even_two_digit_largest_odd_two_digit_l893_89329

-- Define conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Theorem statements
theorem largest_even_two_digit : ∃ n, is_two_digit n ∧ is_even n ∧ ∀ m, is_two_digit m ∧ is_even m → m ≤ n := 
sorry

theorem largest_odd_two_digit : ∃ n, is_two_digit n ∧ is_odd n ∧ ∀ m, is_two_digit m ∧ is_odd m → m ≤ n := 
sorry

end largest_even_two_digit_largest_odd_two_digit_l893_89329


namespace batsman_average_increase_l893_89337

theorem batsman_average_increase 
    (A : ℝ) 
    (h1 : 11 * A + 80 = 12 * 47) : 
    47 - A = 3 := 
by 
  -- Proof goes here
  sorry

end batsman_average_increase_l893_89337


namespace not_diff_of_squares_count_l893_89326

theorem not_diff_of_squares_count :
  ∃ n : ℕ, n ≤ 1000 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ 1000 → (∃ a b : ℤ, k = a^2 - b^2 ↔ k % 4 ≠ 2)) ∧
  n = 250 :=
sorry

end not_diff_of_squares_count_l893_89326


namespace mk97_x_eq_one_l893_89377

noncomputable def mk97_initial_number (x : ℝ) : Prop := 
  x ≠ 0 ∧ 4 * (x^2 - x) = 0

theorem mk97_x_eq_one (x : ℝ) (h : mk97_initial_number x) : x = 1 := by
  sorry

end mk97_x_eq_one_l893_89377


namespace ratio_of_x_intercepts_l893_89302

theorem ratio_of_x_intercepts (b s t : ℝ) (hb : b ≠ 0) (h1 : s = -b / 8) (h2 : t = -b / 4) : s / t = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l893_89302


namespace ratio_of_areas_l893_89310

variable (s' : ℝ) -- Let s' be the side length of square S'

def area_square : ℝ := s' ^ 2
def length_longer_side_rectangle : ℝ := 1.15 * s'
def length_shorter_side_rectangle : ℝ := 0.95 * s'
def area_rectangle : ℝ := length_longer_side_rectangle s' * length_shorter_side_rectangle s'

theorem ratio_of_areas :
  (area_rectangle s') / (area_square s') = (10925 / 10000) :=
by
  -- skip the proof for now
  sorry

end ratio_of_areas_l893_89310


namespace additional_savings_is_300_l893_89343

-- Define constants
def price_per_window : ℕ := 120
def discount_threshold : ℕ := 10
def discount_per_window : ℕ := 10
def free_window_threshold : ℕ := 5

-- Define the number of windows Alice needs
def alice_windows : ℕ := 9

-- Define the number of windows Bob needs
def bob_windows : ℕ := 12

-- Define the function to calculate total cost without discount
def cost_without_discount (n : ℕ) : ℕ := n * price_per_window

-- Define the function to calculate cost with discount
def cost_with_discount (n : ℕ) : ℕ :=
  let full_windows := n - n / free_window_threshold
  let discounted_price := if n > discount_threshold then price_per_window - discount_per_window else price_per_window
  full_windows * discounted_price

-- Define the function to calculate savings when windows are bought separately
def savings_separately : ℕ :=
  (cost_without_discount alice_windows + cost_without_discount bob_windows) 
  - (cost_with_discount alice_windows + cost_with_discount bob_windows)

-- Define the function to calculate savings when windows are bought together
def savings_together : ℕ :=
  let combined_windows := alice_windows + bob_windows
  cost_without_discount combined_windows - cost_with_discount combined_windows

-- Prove that the additional savings when buying together is $300
theorem additional_savings_is_300 : savings_together - savings_separately = 300 := by
  -- missing proof
  sorry

end additional_savings_is_300_l893_89343


namespace circle_radius_increase_l893_89378

theorem circle_radius_increase (r r' : ℝ) (h : π * r'^2 = (25.44 / 100 + 1) * π * r^2) : 
  (r' - r) / r * 100 = 12 :=
by sorry

end circle_radius_increase_l893_89378


namespace inequality_proof_l893_89382

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^4 / (y-1)^2) + (y^4 / (z-1)^2) + (z^4 / (x-1)^2) ≥ 48 := 
by
  sorry -- The actual proof is omitted

end inequality_proof_l893_89382


namespace domain_of_sqrt_function_l893_89399

theorem domain_of_sqrt_function (f : ℝ → ℝ) (x : ℝ) 
  (h1 : ∀ x, (1 / (Real.log x) - 2) ≥ 0) 
  (h2 : ∀ x, Real.log x ≠ 0) : 
  (1 < x ∧ x ≤ Real.sqrt 10) ↔ (∀ x, 0 < Real.log x ∧ Real.log x ≤ 1 / 2) := 
  sorry

end domain_of_sqrt_function_l893_89399


namespace number_of_shirts_l893_89395

theorem number_of_shirts (ratio_pants_shirts: ℕ) (num_pants: ℕ) (S: ℕ) : 
  ratio_pants_shirts = 7 ∧ num_pants = 14 → S = 20 :=
by
  sorry

end number_of_shirts_l893_89395


namespace boy_running_time_l893_89391

theorem boy_running_time :
  let side_length := 60
  let speed1 := 9 * 1000 / 3600       -- 9 km/h to m/s
  let speed2 := 6 * 1000 / 3600       -- 6 km/h to m/s
  let speed3 := 8 * 1000 / 3600       -- 8 km/h to m/s
  let speed4 := 7 * 1000 / 3600       -- 7 km/h to m/s
  let hurdle_time := 5 * 3 * 4        -- 3 hurdles per side, 4 sides
  let time1 := side_length / speed1
  let time2 := side_length / speed2
  let time3 := side_length / speed3
  let time4 := side_length / speed4
  let total_time := time1 + time2 + time3 + time4 + hurdle_time
  total_time = 177.86 := by
{
  -- actual proof would be provided here
  sorry
}

end boy_running_time_l893_89391


namespace normal_line_equation_at_x0_l893_89349

def curve (x : ℝ) : ℝ := x - x^3
noncomputable def x0 : ℝ := -1
noncomputable def y0 : ℝ := curve x0

theorem normal_line_equation_at_x0 :
  ∀ (y : ℝ), y = (1/2 : ℝ) * x + 1/2 ↔ (∃ (x : ℝ), y = curve x ∧ x = x0) :=
by
  sorry

end normal_line_equation_at_x0_l893_89349


namespace cab_income_third_day_l893_89307

noncomputable def cab_driver_income (day1 day2 day3 day4 day5 : ℕ) : ℕ := 
day1 + day2 + day3 + day4 + day5

theorem cab_income_third_day 
  (day1 day2 day4 day5 avg_income total_income day3 : ℕ)
  (h1 : day1 = 45)
  (h2 : day2 = 50)
  (h3 : day4 = 65)
  (h4 : day5 = 70)
  (h_avg : avg_income = 58)
  (h_total : total_income = 5 * avg_income)
  (h_day_sum : day1 + day2 + day4 + day5 = 230) :
  total_income - 230 = 60 :=
sorry

end cab_income_third_day_l893_89307


namespace prime_condition_l893_89381

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_condition (p : ℕ) (h1 : is_prime p) (h2 : is_prime (8 * p^2 + 1)) : 
  p = 3 ∧ is_prime (8 * p^2 - p + 2) :=
by
  sorry

end prime_condition_l893_89381


namespace roots_of_quadratic_sum_cube_l893_89336

noncomputable def quadratic_roots (a b c : ℤ) (p q : ℤ) : Prop :=
  p^2 - b * p + c = 0 ∧ q^2 - b * q + c = 0

theorem roots_of_quadratic_sum_cube (p q : ℤ) :
  quadratic_roots 1 (-5) 6 p q →
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 503 :=
by
  sorry

end roots_of_quadratic_sum_cube_l893_89336


namespace evaluate_expression_l893_89384

-- Definition of the function f
def f (x : ℤ) : ℤ := 3 * x^2 - 5 * x + 8

-- Theorems and lemmas
theorem evaluate_expression : 3 * f 4 + 2 * f (-4) = 260 := by
  sorry

end evaluate_expression_l893_89384


namespace c_share_l893_89386

theorem c_share (a b c : ℕ) (k : ℕ) 
    (h1 : a + b + c = 1010)
    (h2 : a - 25 = 3 * k) 
    (h3 : b - 10 = 2 * k) 
    (h4 : c - 15 = 5 * k) 
    : c = 495 := 
sorry

end c_share_l893_89386


namespace ripe_mangoes_remaining_l893_89327

theorem ripe_mangoes_remaining
  (initial_mangoes : ℕ)
  (ripe_fraction : ℚ)
  (consume_fraction : ℚ)
  (initial_total : initial_mangoes = 400)
  (ripe_ratio : ripe_fraction = 3 / 5)
  (consume_ratio : consume_fraction = 60 / 100) :
  (initial_mangoes * ripe_fraction - initial_mangoes * ripe_fraction * consume_fraction) = 96 :=
by
  sorry

end ripe_mangoes_remaining_l893_89327


namespace sin_gamma_delta_l893_89317

theorem sin_gamma_delta (γ δ : ℝ)
  (hγ : Complex.exp (Complex.I * γ) = Complex.ofReal 4 / 5 + Complex.I * (3 / 5))
  (hδ : Complex.exp (Complex.I * δ) = Complex.ofReal (-5 / 13) + Complex.I * (12 / 13)) :
  Real.sin (γ + δ) = 21 / 65 :=
by
  sorry

end sin_gamma_delta_l893_89317


namespace amc12a_2006_p24_l893_89347

theorem amc12a_2006_p24 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
by
  sorry

end amc12a_2006_p24_l893_89347


namespace correct_statements_l893_89392

variables {n : ℕ}
noncomputable def S (n : ℕ) : ℝ := (n + 1) / n
noncomputable def T (n : ℕ) : ℝ := (n + 1)
noncomputable def a (n : ℕ) : ℝ := if n = 1 then 2 else (-(1:ℝ)) / (n * (n - 1))

theorem correct_statements (n : ℕ) (hn : n ≠ 0) :
  (S n + T n = S n * T n) ∧ (a 1 = 2) ∧ (∀ n, ∃ d, ∀ m, T (n + m) - T n = m * d) ∧ (S n = (n + 1) / n) :=
by
  sorry

end correct_statements_l893_89392


namespace find_RS_length_l893_89360

-- Define the given conditions
def tetrahedron_edges (a b c d e f : ℕ) : Prop :=
  (a = 8 ∨ a = 14 ∨ a = 19 ∨ a = 28 ∨ a = 37 ∨ a = 42) ∧
  (b = 8 ∨ b = 14 ∨ b = 19 ∨ b = 28 ∨ b = 37 ∨ b = 42) ∧
  (c = 8 ∨ c = 14 ∨ c = 19 ∨ c = 28 ∨ c = 37 ∨ c = 42) ∧
  (d = 8 ∨ d = 14 ∨ d = 19 ∨ d = 28 ∨ d = 37 ∨ d = 42) ∧
  (e = 8 ∨ e = 14 ∨ e = 19 ∨ e = 28 ∨ e = 37 ∨ e = 42) ∧
  (f = 8 ∨ f = 14 ∨ f = 19 ∨ f = 28 ∨ f = 37 ∨ f = 42) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def length_of_PQ (pq : ℕ) : Prop := pq = 42

def length_of_RS (rs : ℕ) (a b c d e f pq : ℕ) : Prop :=
  tetrahedron_edges a b c d e f ∧ length_of_PQ pq →
  (rs = 14)

-- The theorem statement
theorem find_RS_length (a b c d e f pq rs : ℕ) :
  tetrahedron_edges a b c d e f ∧ length_of_PQ pq →
  length_of_RS rs a b c d e f pq :=
by sorry

end find_RS_length_l893_89360


namespace don_can_have_more_rum_l893_89316

-- Definitions based on conditions:
def given_rum : ℕ := 10
def max_consumption_rate : ℕ := 3
def already_had : ℕ := 12

-- Maximum allowed consumption calculation:
def max_allowed_rum : ℕ := max_consumption_rate * given_rum

-- Remaining rum calculation:
def remaining_rum : ℕ := max_allowed_rum - already_had

-- Proof statement of the problem:
theorem don_can_have_more_rum : remaining_rum = 18 := by
  -- Let's compute directly:
  have h1 : max_allowed_rum = 30 := by
    simp [max_allowed_rum, max_consumption_rate, given_rum]

  have h2 : remaining_rum = 18 := by
    simp [remaining_rum, h1, already_had]

  exact h2

end don_can_have_more_rum_l893_89316


namespace simplify_expression_l893_89301

theorem simplify_expression (a : ℝ) (h : a ≠ 0) : (a^9 * a^15) / a^3 = a^21 :=
by sorry

end simplify_expression_l893_89301


namespace perpendicular_vector_x_value_l893_89341

-- Definitions based on the given problem conditions
def dot_product_perpendicular (a1 a2 b1 b2 x : ℝ) : Prop :=
  (a1 * b1 + a2 * b2 = 0)

-- Statement to be proved
theorem perpendicular_vector_x_value (x : ℝ) :
  dot_product_perpendicular 4 x 2 4 x → x = -2 :=
by
  intros h
  sorry

end perpendicular_vector_x_value_l893_89341


namespace net_rate_of_pay_is_25_l893_89324

-- Define the conditions 
variables (hours : ℕ) (speed : ℕ) (efficiency : ℕ)
variables (pay_per_mile : ℝ) (cost_per_gallon : ℝ)
variables (total_distance : ℕ) (gas_used : ℕ)
variables (total_earnings : ℝ) (total_cost : ℝ) (net_earnings : ℝ) (net_rate_of_pay : ℝ)

-- Assume the given conditions are as stated in the problem
axiom hrs : hours = 3
axiom spd : speed = 50
axiom eff : efficiency = 25
axiom ppm : pay_per_mile = 0.60
axiom cpg : cost_per_gallon = 2.50

-- Assuming intermediate computations
axiom distance_calc : total_distance = speed * hours
axiom gas_calc : gas_used = total_distance / efficiency
axiom earnings_calc : total_earnings = pay_per_mile * total_distance
axiom cost_calc : total_cost = cost_per_gallon * gas_used
axiom net_earnings_calc : net_earnings = total_earnings - total_cost
axiom pay_rate_calc : net_rate_of_pay = net_earnings / hours

-- Proving the final result
theorem net_rate_of_pay_is_25 :
  net_rate_of_pay = 25 :=
by
  -- Proof goes here
  sorry

end net_rate_of_pay_is_25_l893_89324


namespace math_problem_l893_89371

theorem math_problem :
  (- (1 / 8)) ^ 2007 * (- 8) ^ 2008 = -8 :=
by
  sorry

end math_problem_l893_89371


namespace minimum_value_of_quad_func_l893_89320

def quad_func (x : ℝ) : ℝ :=
  2 * x^2 - 8 * x + 15

theorem minimum_value_of_quad_func :
  (∀ x : ℝ, quad_func 2 ≤ quad_func x) ∧ (quad_func 2 = 7) :=
by
  -- sorry to skip proof
  sorry

end minimum_value_of_quad_func_l893_89320


namespace new_person_weight_l893_89398

-- The conditions from part (a)
variables (average_increase: ℝ) (num_people: ℕ) (weight_lost_person: ℝ)
variables (total_increase: ℝ) (new_weight: ℝ)

-- Assigning the given conditions
axiom h1 : average_increase = 2.5
axiom h2 : num_people = 8
axiom h3 : weight_lost_person = 45
axiom h4 : total_increase = num_people * average_increase
axiom h5 : new_weight = weight_lost_person + total_increase

-- The proof goal: proving that the new person's weight is 65 kg
theorem new_person_weight : new_weight = 65 :=
by
  -- Proof steps go here
  sorry

end new_person_weight_l893_89398


namespace Lisa_types_correctly_l893_89356

-- Given conditions
def Rudy_wpm : ℕ := 64
def Joyce_wpm : ℕ := 76
def Gladys_wpm : ℕ := 91
def Mike_wpm : ℕ := 89
def avg_wpm : ℕ := 80
def num_employees : ℕ := 5

-- Define the hypothesis about Lisa's typing speaking
def Lisa_wpm : ℕ := (num_employees * avg_wpm) - Rudy_wpm - Joyce_wpm - Gladys_wpm - Mike_wpm

-- The statement to prove
theorem Lisa_types_correctly :
  Lisa_wpm = 140 := by
  sorry

end Lisa_types_correctly_l893_89356


namespace recreation_percentage_l893_89361

def wages_last_week (W : ℝ) : ℝ := W
def spent_on_recreation_last_week (W : ℝ) : ℝ := 0.15 * W
def wages_this_week (W : ℝ) : ℝ := 0.90 * W
def spent_on_recreation_this_week (W : ℝ) : ℝ := 0.30 * (wages_this_week W)

theorem recreation_percentage (W : ℝ) (hW: W > 0) :
  (spent_on_recreation_this_week W) / (spent_on_recreation_last_week W) * 100 = 180 := by
  sorry

end recreation_percentage_l893_89361


namespace checker_moves_10_cells_l893_89322

theorem checker_moves_10_cells :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ a 2 = 2 ∧ (∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) ∧ a 10 = 89 :=
by
  -- mathematical proof goes here
  sorry

end checker_moves_10_cells_l893_89322


namespace homework_points_l893_89335

variable (H Q T : ℕ)

theorem homework_points (h1 : T = 4 * Q)
                        (h2 : Q = H + 5)
                        (h3 : H + Q + T = 265) : 
  H = 40 :=
sorry

end homework_points_l893_89335


namespace union_A_B_l893_89314

open Set

-- Define the sets A and B
def setA : Set ℝ := { x | abs x < 3 }
def setB : Set ℝ := { x | x - 1 ≤ 0 }

-- State the theorem we want to prove
theorem union_A_B : setA ∪ setB = { x : ℝ | x < 3 } :=
by
  -- Skip the proof
  sorry

end union_A_B_l893_89314


namespace proposition_1_proposition_4_l893_89331

-- Definitions
variable {a b c : Type} (Line : Type) (Plane : Type)
variable (a b c : Line) (γ : Plane)

-- Given conditions
variable (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Propositions to prove
theorem proposition_1 (H1 : parallel a b) (H2 : parallel b c) : parallel a c := sorry

theorem proposition_4 (H3 : perpendicular a γ) (H4 : perpendicular b γ) : parallel a b := sorry

end proposition_1_proposition_4_l893_89331


namespace complex_omega_sum_l893_89312

open Complex

theorem complex_omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 + ω^48 + ω^51 + ω^54 + ω^57 + ω^60 + ω^63 = 1 := 
by
  sorry

end complex_omega_sum_l893_89312


namespace long_letter_time_ratio_l893_89333

-- Definitions based on conditions
def letters_per_month := (30 / 3 : Nat)
def regular_letter_pages := (20 / 10 : Nat)
def total_regular_pages := letters_per_month * regular_letter_pages
def long_letter_pages := 24 - total_regular_pages

-- Define the times and calculate the ratios
def time_spent_per_page_regular := (20 / regular_letter_pages : Nat)
def time_spent_per_page_long := (80 / long_letter_pages : Nat)
def time_ratio := time_spent_per_page_long / time_spent_per_page_regular

-- Theorem to prove the ratio
theorem long_letter_time_ratio : time_ratio = 2 := by
  sorry

end long_letter_time_ratio_l893_89333


namespace smallest_x_value_l893_89354

-- Definitions based on given problem conditions
def is_solution (x y : ℕ) : Prop :=
  0 < x ∧ 0 < y ∧ (3 : ℝ) / 4 = y / (252 + x)

theorem smallest_x_value : ∃ x : ℕ, ∀ y : ℕ, is_solution x y → x = 0 :=
by
  sorry

end smallest_x_value_l893_89354


namespace smallest_n_reducible_fraction_l893_89328

theorem smallest_n_reducible_fraction : ∀ (n : ℕ), (∃ (k : ℕ), gcd (n - 13) (5 * n + 6) = k ∧ k > 1) ↔ n = 84 := by
  sorry

end smallest_n_reducible_fraction_l893_89328


namespace correct_statement_C_l893_89383

theorem correct_statement_C : ∀ y : ℝ, y^2 = 25 ↔ (y = 5 ∨ y = -5) := 
by sorry

end correct_statement_C_l893_89383


namespace evaluate_polynomial_at_2_l893_89306

def polynomial (x : ℕ) : ℕ := 3 * x^4 + x^3 + 2 * x^2 + x + 4

def horner_method (x : ℕ) : ℕ :=
  let v_0 := x
  let v_1 := 3 * v_0 + 1
  let v_2 := v_1 * v_0 + 2
  v_2

theorem evaluate_polynomial_at_2 :
  horner_method 2 = 16 :=
by
  sorry

end evaluate_polynomial_at_2_l893_89306


namespace number_of_true_propositions_l893_89385

-- Define the original proposition
def prop (x: Real) : Prop := x^2 > 1 → x > 1

-- Define converse, inverse, contrapositive
def converse (x: Real) : Prop := x > 1 → x^2 > 1
def inverse (x: Real) : Prop := x^2 ≤ 1 → x ≤ 1
def contrapositive (x: Real) : Prop := x ≤ 1 → x^2 ≤ 1

-- Define the proposition we want to prove: the number of true propositions among them
theorem number_of_true_propositions :
  (converse 2 = True) ∧ (inverse 2 = True) ∧ (contrapositive 2 = False) → 2 = 2 :=
by sorry

end number_of_true_propositions_l893_89385


namespace tangent_circle_locus_l893_89369

-- Definitions for circle C1 and circle C2
def Circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Definition of being tangent to a circle
def ExternallyTangent (cx cy cr : ℝ) : Prop := (cx - 0)^2 + (cy - 0)^2 = (cr + 1)^2
def InternallyTangent (cx cy cr : ℝ) : Prop := (cx - 3)^2 + (cy - 0)^2 = (3 - cr)^2

-- Definition of locus L where (a,b) are centers of circles tangent to both C1 and C2
def Locus (a b : ℝ) : Prop := 28 * a^2 + 64 * b^2 - 84 * a - 49 = 0

-- The theorem to be proved
theorem tangent_circle_locus (a b r : ℝ) :
  (ExternallyTangent a b r) → (InternallyTangent a b r) → Locus a b :=
by {
  sorry
}

end tangent_circle_locus_l893_89369


namespace arrangements_three_events_l893_89376

theorem arrangements_three_events (volunteers : ℕ) (events : ℕ) (h_vol : volunteers = 5) (h_events : events = 3) : 
  ∃ n : ℕ, n = (events^volunteers - events * 2^volunteers + events * 1^volunteers) ∧ n = 150 := 
by
  sorry

end arrangements_three_events_l893_89376


namespace salon_visitors_l893_89305

noncomputable def total_customers (x : ℕ) : ℕ :=
  let revenue_customers_with_one_visit := 10 * x
  let revenue_customers_with_two_visits := 30 * 18
  let revenue_customers_with_three_visits := 10 * 26
  let total_revenue := revenue_customers_with_one_visit + revenue_customers_with_two_visits + revenue_customers_with_three_visits
  if total_revenue = 1240 then
    x + 30 + 10
  else
    0

theorem salon_visitors : 
  ∃ x, total_customers x = 84 :=
by
  use 44
  sorry

end salon_visitors_l893_89305


namespace solve_equation_l893_89372

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * (x^2020)^(1/202) - 1 = 2020 * x → x = 1 :=
by
  sorry

end solve_equation_l893_89372


namespace product_p_yi_eq_neg26_l893_89390

-- Definitions of the polynomials h and p.
def h (y : ℂ) : ℂ := y^3 - 3 * y + 1
def p (y : ℂ) : ℂ := y^3 + 2

-- Given that y1, y2, y3 are roots of h(y)
variables (y1 y2 y3 : ℂ) (H1 : h y1 = 0) (H2 : h y2 = 0) (H3 : h y3 = 0)

-- State the theorem to show p(y1) * p(y2) * p(y3) = -26
theorem product_p_yi_eq_neg26 : p y1 * p y2 * p y3 = -26 :=
sorry

end product_p_yi_eq_neg26_l893_89390


namespace total_students_count_l893_89389

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) : Prop := g * 4 = b * 3
def boys_count : ℕ := 28

-- Theorem to prove the total number of students
theorem total_students_count {g : ℕ} (h : ratio_girls_to_boys g boys_count) : g + boys_count = 49 :=
sorry

end total_students_count_l893_89389


namespace binomial_coeff_coprime_l893_89365

def binom (a b : ℕ) : ℕ := Nat.factorial a / (Nat.factorial b * Nat.factorial (a - b))

theorem binomial_coeff_coprime (p a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hp : Nat.Prime p) 
  (hbase_p_a : ∀ i, (a / p^i % p) ≥ (b / p^i % p)) 
  : Nat.gcd (binom a b) p = 1 :=
by sorry

end binomial_coeff_coprime_l893_89365


namespace problem1_solution_set_problem2_a_range_l893_89380

-- Define the function f
def f (x a : ℝ) := |x - a| + 5 * x

-- Problem Part 1: Prove for a = -1, the solution set for f(x) ≤ 5x + 3 is [-4, 2]
theorem problem1_solution_set :
  ∀ (x : ℝ), f x (-1) ≤ 5 * x + 3 ↔ -4 ≤ x ∧ x ≤ 2 := by
  sorry

-- Problem Part 2: Prove that if f(x) ≥ 0 for all x ≥ -1, then a ≥ 4 or a ≤ -6
theorem problem2_a_range (a : ℝ) :
  (∀ (x : ℝ), x ≥ -1 → f x a ≥ 0) ↔ a ≥ 4 ∨ a ≤ -6 := by
  sorry

end problem1_solution_set_problem2_a_range_l893_89380


namespace evaluate_expression_l893_89379

-- Definitions based on conditions
def a : ℤ := 5
def b : ℤ := -3
def c : ℤ := 2

-- Theorem to be proved: evaluate the expression
theorem evaluate_expression : (3 : ℚ) / (a + b + c) = 3 / 4 := by
  sorry

end evaluate_expression_l893_89379


namespace probability_calculation_l893_89346

open Classical

def probability_odd_sum_given_even_product :=
  let num_even := 4  -- even numbers: 2, 4, 6, 8
  let num_odd := 4   -- odd numbers: 1, 3, 5, 7
  let total_outcomes := 8^5
  let prob_all_odd := (num_odd / 8)^5
  let prob_even_product := 1 - prob_all_odd

  let ways_one_odd := 5 * num_odd * num_even^4
  let ways_three_odd := Nat.choose 5 3 * num_odd^3 * num_even^2
  let ways_five_odd := num_odd^5

  let favorable_outcomes := ways_one_odd + ways_three_odd + ways_five_odd
  let total_even_product_outcomes := total_outcomes * prob_even_product

  favorable_outcomes / total_even_product_outcomes

theorem probability_calculation :
  probability_odd_sum_given_even_product = rational_result := sorry

end probability_calculation_l893_89346


namespace evaluate_nav_expression_l893_89330
noncomputable def nav (k m : ℕ) := k * (k - m)

theorem evaluate_nav_expression : (nav 5 1) + (nav 4 1) = 32 :=
by
  -- Skipping the proof as instructed
  sorry

end evaluate_nav_expression_l893_89330


namespace problem_am_hm_l893_89358

open Real

theorem problem_am_hm (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 2) :
  ∃ S : Set ℝ, (∀ s ∈ S, (2 ≤ s)) ∧ (∀ z, (2 ≤ z) → (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 ∧ z = 1/x + 1/y))
  ∧ (S = {z | 2 ≤ z}) := sorry

end problem_am_hm_l893_89358
