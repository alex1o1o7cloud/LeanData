import Mathlib

namespace distance_between_foci_l476_47656

-- Define the ellipse
def ellipse_eq (x y : ℝ) := 9 * x^2 + 36 * y^2 = 1296

-- Define the semi-major and semi-minor axes
def semi_major_axis := 12
def semi_minor_axis := 6

-- Distance between the foci of the ellipse
theorem distance_between_foci : 
  (∃ x y : ℝ, ellipse_eq x y) → 2 * Real.sqrt (semi_major_axis^2 - semi_minor_axis^2) = 12 * Real.sqrt 3 :=
by
  sorry

end distance_between_foci_l476_47656


namespace intersection_lines_l476_47601

theorem intersection_lines (a b : ℝ) (h1 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → x = 1/3 * y + a)
                          (h2 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → y = 1/3 * x + b) :
  a + b = 8 / 3 :=
sorry

end intersection_lines_l476_47601


namespace cube_sum_gt_l476_47646

variable (a b c d : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
variable (h1 : a + b = c + d)
variable (h2 : a^2 + b^2 > c^2 + d^2)

theorem cube_sum_gt : a^3 + b^3 > c^3 + d^3 := by
  sorry

end cube_sum_gt_l476_47646


namespace slope_of_given_line_l476_47643

theorem slope_of_given_line : ∀ (x y : ℝ), (4 / x + 5 / y = 0) → (y = (-5 / 4) * x) := 
by 
  intros x y h
  sorry

end slope_of_given_line_l476_47643


namespace math_problem_l476_47698

theorem math_problem (a b : ℝ) (h : a * b < 0) : a^2 * |b| - b^2 * |a| + a * b * (|a| - |b|) = 0 :=
sorry

end math_problem_l476_47698


namespace ratio_of_percent_increase_to_decrease_l476_47610

variable (P U V : ℝ)
variable (h1 : P * U = 0.25 * P * V)
variable (h2 : P ≠ 0)

theorem ratio_of_percent_increase_to_decrease (h : U = 0.25 * V) :
  ((V - U) / U) * 100 / 75 = 4 :=
by
  sorry

end ratio_of_percent_increase_to_decrease_l476_47610


namespace length_of_train_is_correct_l476_47663

noncomputable def speed_in_m_per_s (speed_in_km_per_hr : ℝ) : ℝ := speed_in_km_per_hr * 1000 / 3600

noncomputable def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

noncomputable def length_of_train (total_distance : ℝ) (length_of_bridge : ℝ) : ℝ := total_distance - length_of_bridge

theorem length_of_train_is_correct :
  ∀ (speed_in_km_per_hr : ℝ) (time_to_cross_bridge : ℝ) (length_of_bridge : ℝ),
  speed_in_km_per_hr = 72 →
  time_to_cross_bridge = 12.199024078073753 →
  length_of_bridge = 134 →
  length_of_train (total_distance (speed_in_m_per_s speed_in_km_per_hr) time_to_cross_bridge) length_of_bridge = 110.98048156147506 :=
by 
  intros speed_in_km_per_hr time_to_cross_bridge length_of_bridge hs ht hl;
  rw [hs, ht, hl];
  sorry

end length_of_train_is_correct_l476_47663


namespace rachel_math_homework_l476_47659

def rachel_homework (M : ℕ) (reading : ℕ) (biology : ℕ) (total : ℕ) : Prop :=
reading = 3 ∧ biology = 10 ∧ total = 15 ∧ reading + biology + M = total

theorem rachel_math_homework: ∃ M : ℕ, rachel_homework M 3 10 15 ∧ M = 2 := 
by 
  sorry

end rachel_math_homework_l476_47659


namespace not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l476_47658

theorem not_divisible_by_5_square_plus_or_minus_1_divisible_by_5 (a : ℤ) (h : a % 5 ≠ 0) :
  (a^2 + 1) % 5 = 0 ∨ (a^2 - 1) % 5 = 0 :=
by
  sorry

end not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l476_47658


namespace integer_base10_from_bases_l476_47667

theorem integer_base10_from_bases (C D : ℕ) (hC : 0 ≤ C ∧ C ≤ 7) (hD : 0 ≤ D ∧ D ≤ 5)
    (h : 8 * C + D = 6 * D + C) : C = 0 ∧ D = 0 ∧ (8 * C + D = 0) := by
  sorry

end integer_base10_from_bases_l476_47667


namespace domain_of_g_l476_47657

theorem domain_of_g : ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 1 ≠ 0 :=
by
  intro t
  sorry

end domain_of_g_l476_47657


namespace unique_cell_distance_50_l476_47695

noncomputable def king_dist (A B: ℤ × ℤ) : ℤ :=
  max (abs (A.1 - B.1)) (abs (A.2 - B.2))

theorem unique_cell_distance_50
  (A B C: ℤ × ℤ)
  (hAB: king_dist A B = 100)
  (hBC: king_dist B C = 100)
  (hCA: king_dist C A = 100) :
  ∃! (X: ℤ × ℤ), king_dist X A = 50 ∧ king_dist X B = 50 ∧ king_dist X C = 50 :=
sorry

end unique_cell_distance_50_l476_47695


namespace value_of_x_l476_47686

theorem value_of_x 
    (r : ℝ) (a : ℝ) (x : ℝ) (shaded_area : ℝ)
    (h1 : r = 2)
    (h2 : a = 2)
    (h3 : shaded_area = 2) :
  x = (Real.pi / 3) + (Real.sqrt 3 / 2) - 1 :=
sorry

end value_of_x_l476_47686


namespace profit_percentage_l476_47684

variable {C S : ℝ}

theorem profit_percentage (h : 19 * C = 16 * S) :
  ((S - C) / C) * 100 = 18.75 := by
  sorry

end profit_percentage_l476_47684


namespace value_of_expression_l476_47694

def expr : ℕ :=
  8 + 2 * (3^2)

theorem value_of_expression : expr = 26 :=
  by
  sorry

end value_of_expression_l476_47694


namespace ganesh_average_speed_l476_47691

variable (D : ℝ) (hD : D > 0)

/-- Ganesh's average speed over the entire journey is 45 km/hr.
    Given:
    - Speed from X to Y is 60 km/hr
    - Speed from Y to X is 36 km/hr
--/
theorem ganesh_average_speed :
  let T1 := D / 60
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  (total_distance / total_time) = 45 :=
by
  sorry

end ganesh_average_speed_l476_47691


namespace spoons_needed_to_fill_cup_l476_47665

-- Define necessary conditions
def spoon_capacity : Nat := 5
def liter_to_milliliters : Nat := 1000

-- State the problem
theorem spoons_needed_to_fill_cup : liter_to_milliliters / spoon_capacity = 200 := 
by 
  -- Skip the actual proof
  sorry

end spoons_needed_to_fill_cup_l476_47665


namespace part_a_part_b_l476_47647

theorem part_a (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 → (3^m - 1) % (2^m) = 0 := by
  sorry

theorem part_b (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 ∨ m = 6 ∨ m = 8 → (31^m - 1) % (2^m) = 0 := by
  sorry

end part_a_part_b_l476_47647


namespace sum_of_palindromes_l476_47632

theorem sum_of_palindromes (a b : ℕ) (ha : a > 99) (ha' : a < 1000) (hb : b > 99) (hb' : b < 1000) 
  (hpal_a : ∀ i j k, a = 100*i + 10*j + k → a = 100*k + 10*j + i) 
  (hpal_b : ∀ i j k, b = 100*i + 10*j + k → b = 100*k + 10*j + i) 
  (hprod : a * b = 589185) : a + b = 1534 :=
sorry

end sum_of_palindromes_l476_47632


namespace fishing_problem_l476_47607

theorem fishing_problem :
  ∃ (x y : ℕ), 
    (x + y = 70) ∧ 
    (∃ k : ℕ, x = 9 * k) ∧ 
    (∃ m : ℕ, y = 17 * m) ∧ 
    x = 36 ∧ 
    y = 34 := 
by
  sorry

end fishing_problem_l476_47607


namespace geom_prog_235_l476_47668

theorem geom_prog_235 (q : ℝ) (k n : ℕ) (hk : 1 < k) (hn : k < n) : 
  ¬ (q > 0 ∧ q ≠ 1 ∧ 3 = 2 * q^(k - 1) ∧ 5 = 2 * q^(n - 1)) := 
by 
  sorry

end geom_prog_235_l476_47668


namespace candy_total_l476_47649

theorem candy_total (n m : ℕ) (h1 : n = 2) (h2 : m = 8) : n * m = 16 :=
by
  -- This will contain the proof
  sorry

end candy_total_l476_47649


namespace inequality_solution_l476_47617

theorem inequality_solution (x : ℝ) : 3 * x + 2 ≥ 5 ↔ x ≥ 1 :=
by sorry

end inequality_solution_l476_47617


namespace maximum_value_l476_47692

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_value (a b c : ℝ) (h_a : 1 ≤ a ∧ a ≤ 2)
  (h_f1 : f a b c 1 ≤ 1) (h_f2 : f a b c 2 ≤ 1) :
  7 * b + 5 * c ≤ -6 :=
sorry

end maximum_value_l476_47692


namespace malcolm_initial_white_lights_l476_47683

theorem malcolm_initial_white_lights :
  let red_lights := 12
  let blue_lights := 3 * red_lights
  let green_lights := 6
  let bought_lights := red_lights + blue_lights + green_lights
  let remaining_lights := 5
  let total_needed_lights := bought_lights + remaining_lights
  W = total_needed_lights :=
by
  sorry

end malcolm_initial_white_lights_l476_47683


namespace find_side_length_l476_47693

theorem find_side_length
  (a b c : ℝ) 
  (cosine_diff_angle : ℝ) 
  (h_b : b = 5)
  (h_c : c = 4)
  (h_cosine_diff_angle : cosine_diff_angle = 31 / 32) :
  a = 6 := 
sorry

end find_side_length_l476_47693


namespace no_solution_system_l476_47661

noncomputable def system_inconsistent : Prop :=
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 8 ∧ 6 * x - 8 * y = 12)

theorem no_solution_system : system_inconsistent :=
by
  sorry

end no_solution_system_l476_47661


namespace cat_mouse_position_after_299_moves_l476_47609

-- Definitions based on conditions
def cat_position (move : Nat) : Nat :=
  let active_moves := move - (move / 100)
  active_moves % 4

def mouse_position (move : Nat) : Nat :=
  move % 8

-- Main theorem
theorem cat_mouse_position_after_299_moves :
  cat_position 299 = 0 ∧ mouse_position 299 = 3 :=
by
  sorry

end cat_mouse_position_after_299_moves_l476_47609


namespace tangent_slope_is_four_l476_47680

-- Define the given curve and point
def curve (x : ℝ) : ℝ := 2 * x^2
def point : ℝ × ℝ := (1, 2)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the tangent slope at the given point
def tangent_slope_at_point : ℝ := curve_derivative 1

-- Prove that the tangent slope at point (1, 2) is 4
theorem tangent_slope_is_four : tangent_slope_at_point = 4 :=
by
  -- We state that the slope at x = 1 is 4
  sorry

end tangent_slope_is_four_l476_47680


namespace digit_expression_equals_2021_l476_47675

theorem digit_expression_equals_2021 :
  ∃ (f : ℕ → ℕ), 
  (f 0 = 0 ∧
   f 1 = 1 ∧
   f 2 = 2 ∧
   f 3 = 3 ∧
   f 4 = 4 ∧
   f 5 = 5 ∧
   f 6 = 6 ∧
   f 7 = 7 ∧
   f 8 = 8 ∧
   f 9 = 9 ∧
   43 * (8 * 5 + 7) + 0 * 1 * 2 * 6 * 9 = 2021) :=
sorry

end digit_expression_equals_2021_l476_47675


namespace protein_percentage_in_mixture_l476_47653

theorem protein_percentage_in_mixture :
  let soybean_meal_weight := 240
  let cornmeal_weight := 40
  let mixture_weight := 280
  let soybean_protein_content := 0.14
  let cornmeal_protein_content := 0.07
  let total_protein := soybean_meal_weight * soybean_protein_content + cornmeal_weight * cornmeal_protein_content
  let protein_percentage := (total_protein / mixture_weight) * 100
  protein_percentage = 13 :=
by
  sorry

end protein_percentage_in_mixture_l476_47653


namespace evaluate_expression_evaluate_fraction_l476_47618

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  3 * x^3 + 4 * y^3 = 337 :=
by
  sorry

theorem evaluate_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) 
  (h : 3 * x^3 + 4 * y^3 = 337) :
  (3 * x^3 + 4 * y^3) / 9 = 37 + 4/9 :=
by
  sorry

end evaluate_expression_evaluate_fraction_l476_47618


namespace compute_a_l476_47604

theorem compute_a (a b : ℚ) 
  (h_root1 : (-1:ℚ) - 5 * (Real.sqrt 3) = -1 - 5 * (Real.sqrt 3))
  (h_rational1 : (-1:ℚ) + 5 * (Real.sqrt 3) = -1 + 5 * (Real.sqrt 3))
  (h_poly : ∀ x, x^3 + a*x^2 + b*x + 48 = 0) :
  a = 50 / 37 :=
by
  sorry

end compute_a_l476_47604


namespace solve_problem_l476_47674

noncomputable def solution_set : Set ℤ := {x | abs (7 * x - 5) ≤ 9}

theorem solve_problem : solution_set = {0, 1, 2} := by
  sorry

end solve_problem_l476_47674


namespace mistaken_divisor_l476_47671

theorem mistaken_divisor (x : ℕ) (h : 49 * x = 28 * 21) : x = 12 :=
sorry

end mistaken_divisor_l476_47671


namespace square_division_l476_47628

theorem square_division (n k : ℕ) (m : ℕ) (h : n * k = m * m) :
  ∃ u v d : ℕ, (gcd u v = 1) ∧ (n = d * u * u) ∧ (k = d * v * v) ∧ (m = d * u * v) :=
by sorry

end square_division_l476_47628


namespace n_prime_of_divisors_l476_47654

theorem n_prime_of_divisors (n k : ℕ) (h₁ : n > 1) 
  (h₂ : ∀ d : ℕ, d ∣ n → (d + k ∣ n) ∨ (d - k ∣ n)) : Prime n :=
  sorry

end n_prime_of_divisors_l476_47654


namespace cubic_polynomial_coefficients_l476_47672

theorem cubic_polynomial_coefficients (f g : Polynomial ℂ) (b c d : ℂ) :
  f = Polynomial.C 4 + Polynomial.X * (Polynomial.C 3 + Polynomial.X * (Polynomial.C 2 + Polynomial.X)) →
  (∀ x, Polynomial.eval x f = 0 → Polynomial.eval (x^2) g = 0) →
  g = Polynomial.C d + Polynomial.X * (Polynomial.C c + Polynomial.X * (Polynomial.C b + Polynomial.X)) →
  (b, c, d) = (4, -15, -32) :=
by
  intro h1 h2 h3
  sorry

end cubic_polynomial_coefficients_l476_47672


namespace system1_solution_system2_solution_l476_47648

-- Define the first system of equations and its solution
theorem system1_solution (x y : ℝ) : 
    (3 * (x - 1) = y + 5 ∧ 5 * (y - 1) = 3 * (x + 5)) ↔ (x = 5 ∧ y = 7) :=
sorry

-- Define the second system of equations and its solution
theorem system2_solution (x y a : ℝ) :
    (2 * x + 4 * y = a ∧ 7 * x - 2 * y = 3 * a) ↔ 
    (x = (7 / 16) * a ∧ y = (1 / 32) * a) :=
sorry

end system1_solution_system2_solution_l476_47648


namespace mrs_hilt_initial_marbles_l476_47620

theorem mrs_hilt_initial_marbles (lost_marble : ℕ) (remaining_marble : ℕ) (h1 : lost_marble = 15) (h2 : remaining_marble = 23) : 
    (remaining_marble + lost_marble) = 38 :=
by
  sorry

end mrs_hilt_initial_marbles_l476_47620


namespace fraction_subtraction_l476_47681

theorem fraction_subtraction :
  (12 / 30) - (1 / 7) = 9 / 35 :=
by sorry

end fraction_subtraction_l476_47681


namespace no_positive_integer_n_for_perfect_squares_l476_47689

theorem no_positive_integer_n_for_perfect_squares :
  ∀ (n : ℕ), 0 < n → ¬ (∃ a b : ℤ, (n + 1) * 2^n = a^2 ∧ (n + 3) * 2^(n + 2) = b^2) :=
by
  sorry

end no_positive_integer_n_for_perfect_squares_l476_47689


namespace find_a_minus_b_l476_47676

theorem find_a_minus_b (a b : ℚ) (h_eq : ∀ x : ℚ, (a * (-5 * x + 3) + b) = x - 9) : 
  a - b = 41 / 5 := 
by {
  sorry
}

end find_a_minus_b_l476_47676


namespace new_area_eq_1_12_original_area_l476_47697

variable (L W : ℝ)
def increased_length (L : ℝ) : ℝ := 1.40 * L
def decreased_width (W : ℝ) : ℝ := 0.80 * W
def original_area (L W : ℝ) : ℝ := L * W
def new_area (L W : ℝ) : ℝ := (increased_length L) * (decreased_width W)

theorem new_area_eq_1_12_original_area (L W : ℝ) :
  new_area L W = 1.12 * (original_area L W) :=
by
  sorry

end new_area_eq_1_12_original_area_l476_47697


namespace array_sum_remainder_l476_47633

def entry_value (r c : ℕ) : ℚ :=
  (1 / (2 * 1013) ^ r) * (1 / 1013 ^ c)

def array_sum : ℚ :=
  (1 / (2 * 1013 - 1)) * (1 / (1013 - 1))

def m : ℤ := 1
def n : ℤ := 2046300
def mn_sum : ℤ := m + n

theorem array_sum_remainder :
  (mn_sum % 1013) = 442 :=
by
  sorry

end array_sum_remainder_l476_47633


namespace part1_solution_part2_solution_l476_47662

noncomputable def find_prices (price_peanuts price_tea : ℝ) : Prop :=
price_peanuts + 40 = price_tea ∧
50 * price_peanuts = 10 * price_tea

theorem part1_solution :
  ∃ (price_peanuts price_tea : ℝ), find_prices price_peanuts price_tea :=
by
  sorry

def cost_function (m : ℝ) : ℝ :=
6 * m + 36 * (60 - m)

def profit_function (m : ℝ) : ℝ :=
(10 - 6) * m + (50 - 36) * (60 - m)

noncomputable def max_profit := 540

theorem part2_solution :
  ∃ (m t : ℝ), 30 ≤ m ∧ m ≤ 40 ∧ cost_function m ≤ 1260 ∧ profit_function m = max_profit :=
by
  sorry

end part1_solution_part2_solution_l476_47662


namespace one_less_than_neg_one_is_neg_two_l476_47625

theorem one_less_than_neg_one_is_neg_two : (-1 - 1 = -2) :=
by
  sorry

end one_less_than_neg_one_is_neg_two_l476_47625


namespace condition1_condition2_condition3_l476_47621

-- Condition 1 statement
theorem condition1: (number_of_ways_condition1 : ℕ) = 5520 := by
  -- Expected proof that number_of_ways_condition1 = 5520
  sorry

-- Condition 2 statement
theorem condition2: (number_of_ways_condition2 : ℕ) = 3360 := by
  -- Expected proof that number_of_ways_condition2 = 3360
  sorry

-- Condition 3 statement
theorem condition3: (number_of_ways_condition3 : ℕ) = 360 := by
  -- Expected proof that number_of_ways_condition3 = 360
  sorry

end condition1_condition2_condition3_l476_47621


namespace selena_ran_24_miles_l476_47666

theorem selena_ran_24_miles (S J : ℝ) (h1 : S + J = 36) (h2 : J = S / 2) : S = 24 := 
sorry

end selena_ran_24_miles_l476_47666


namespace find_a6_l476_47623

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem find_a6 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h2 : a 2 = 4) (h4 : a 4 = 2) : a 6 = 0 :=
by sorry

end find_a6_l476_47623


namespace initial_bottle_count_l476_47641

variable (B: ℕ)

-- Conditions: Each bottle holds 15 stars, bought 3 more bottles, total 75 stars to fill
def bottle_capacity := 15
def additional_bottles := 3
def total_stars := 75

-- The main statement we want to prove
theorem initial_bottle_count (h : (B + additional_bottles) * bottle_capacity = total_stars) : 
    B = 2 :=
by sorry

end initial_bottle_count_l476_47641


namespace find_abc_sum_l476_47685

theorem find_abc_sum (a b c : ℕ) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
sorry

end find_abc_sum_l476_47685


namespace max_sum_unique_digits_expression_equivalent_l476_47642

theorem max_sum_unique_digits_expression_equivalent :
  ∃ (a b c d e : ℕ), (2 * 19 * 53 = 2014) ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (2 * (b + c) * (d + e) = 2014) ∧
    (a + b + c + d + e = 35) ∧ 
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) :=
by
  sorry

end max_sum_unique_digits_expression_equivalent_l476_47642


namespace quadratic_root_zero_l476_47688

theorem quadratic_root_zero (a : ℝ) : 
  ((a-1) * 0^2 + 0 + a^2 - 1 = 0) 
  → a ≠ 1 
  → a = -1 := 
by
  intro h1 h2
  sorry

end quadratic_root_zero_l476_47688


namespace first_reduction_percentage_l476_47615

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.6 = P * 0.45 → x = 25 :=
by
  sorry

end first_reduction_percentage_l476_47615


namespace choice_of_b_l476_47650

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x - 2)
noncomputable def g (x : ℝ) : ℝ := f (x + 3)

theorem choice_of_b (b : ℝ) :
  (g (g x) = x) ↔ (b = -4) :=
sorry

end choice_of_b_l476_47650


namespace circle_radius_squared_l476_47605

open Real

/-- Prove that the square of the radius of a circle is 200 given the conditions provided. -/

theorem circle_radius_squared {r : ℝ}
  (AB CD : ℝ)
  (BP : ℝ) 
  (APD : ℝ) 
  (hAB : AB = 12)
  (hCD : CD = 9)
  (hBP : BP = 10)
  (hAPD : APD = 45) :
  r^2 = 200 := 
sorry

end circle_radius_squared_l476_47605


namespace smallest_possible_sum_l476_47624

theorem smallest_possible_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Nat.gcd (a + b) 330 = 1) (h4 : b ^ b ∣ a ^ a) (h5 : ¬ b ∣ a) :
  a + b = 147 :=
sorry

end smallest_possible_sum_l476_47624


namespace max_largest_int_of_avg_and_diff_l476_47652

theorem max_largest_int_of_avg_and_diff (A B C D E : ℕ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : C ≤ D) (h4 : D ≤ E) 
  (h_avg : (A + B + C + D + E) / 5 = 70) (h_diff : E - A = 10) : E = 340 :=
by
  sorry

end max_largest_int_of_avg_and_diff_l476_47652


namespace solve_for_n_l476_47616

theorem solve_for_n (n : ℕ) (h : (16^n) * (16^n) * (16^n) * (16^n) * (16^n) = 256^5) : n = 2 := by
  sorry

end solve_for_n_l476_47616


namespace Alex_dimes_l476_47699

theorem Alex_dimes : 
    ∃ (d q : ℕ), 10 * d + 25 * q = 635 ∧ d = q + 5 ∧ d = 22 :=
by sorry

end Alex_dimes_l476_47699


namespace original_ratio_l476_47631

theorem original_ratio (x y : ℤ) (h₁ : y = 72) (h₂ : (x + 6) / y = 1 / 3) : y / x = 4 := 
by
  sorry

end original_ratio_l476_47631


namespace inequality_proof_l476_47606

theorem inequality_proof (a b : Real) (h1 : (1 / a) < (1 / b)) (h2 : (1 / b) < 0) : 
  (b / a) + (a / b) > 2 :=
by
  sorry

end inequality_proof_l476_47606


namespace numberOfBoys_playground_boys_count_l476_47651

-- Definitions and conditions
def numberOfGirls : ℕ := 28
def totalNumberOfChildren : ℕ := 63

-- Theorem statement
theorem numberOfBoys (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) : ℕ :=
  totalNumberOfChildren - numberOfGirls

-- Proof statement
theorem playground_boys_count (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) (boysOnPlayground : ℕ) : 
  numberOfGirls = 28 → 
  totalNumberOfChildren = 63 → 
  boysOnPlayground = totalNumberOfChildren - numberOfGirls →
  boysOnPlayground = 35 :=
by
  intros
  -- since no proof is required, we use sorry here
  exact sorry

end numberOfBoys_playground_boys_count_l476_47651


namespace relationship_among_neg_a_neg_a3_a2_l476_47613

theorem relationship_among_neg_a_neg_a3_a2 (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 :=
by sorry

end relationship_among_neg_a_neg_a3_a2_l476_47613


namespace solve_for_q_l476_47608

theorem solve_for_q
  (n m q : ℚ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m - n) / 66)
  (h3 : 5 / 6 = (q - m) / 150) :
  q = 230 :=
by
  sorry

end solve_for_q_l476_47608


namespace leak_takes_3_hours_to_empty_l476_47627

noncomputable def leak_emptying_time (inlet_rate_per_minute: ℕ) (tank_empty_time_with_inlet: ℕ) (tank_capacity: ℕ) : ℕ :=
  let inlet_rate_per_hour := inlet_rate_per_minute * 60
  let effective_empty_rate := tank_capacity / tank_empty_time_with_inlet
  let leak_rate := inlet_rate_per_hour + effective_empty_rate
  tank_capacity / leak_rate

theorem leak_takes_3_hours_to_empty:
  leak_emptying_time 6 12 1440 = 3 := 
sorry

end leak_takes_3_hours_to_empty_l476_47627


namespace phi_value_for_unique_symmetry_center_l476_47696

theorem phi_value_for_unique_symmetry_center :
  ∃ (φ : ℝ), (0 < φ ∧ φ < π / 2) ∧
  (φ = π / 12 ∨ φ = π / 6 ∨ φ = π / 3 ∨ φ = 5 * π / 12) ∧
  ((∃ x : ℝ, 2 * x + φ = π ∧ π / 6 < x ∧ x < π / 3) ↔ φ = 5 * π / 12) :=
  sorry

end phi_value_for_unique_symmetry_center_l476_47696


namespace find_red_chairs_l476_47660

noncomputable def red_chairs := Nat
noncomputable def yellow_chairs := Nat
noncomputable def blue_chairs := Nat

theorem find_red_chairs
    (R Y B : Nat)
    (h1 : Y = 2 * R)
    (h2 : B = Y - 2)
    (h3 : R + Y + B = 18) :
    R = 4 := by
  sorry

end find_red_chairs_l476_47660


namespace correct_statement_is_B_l476_47690

-- Define integers and zero
def is_integer (n : ℤ) : Prop := True
def is_zero (n : ℤ) : Prop := n = 0

-- Define rational numbers
def is_rational (q : ℚ) : Prop := True

-- Positive and negative zero cannot co-exist
def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0

-- Statement A: Integers and negative integers are collectively referred to as integers.
def statement_A : Prop :=
  ∀ n : ℤ, (is_positive n ∨ is_negative n) ↔ is_integer n

-- Statement B: Integers and fractions are collectively referred to as rational numbers.
def statement_B : Prop :=
  ∀ q : ℚ, is_rational q

-- Statement C: Zero can be either a positive integer or a negative integer.
def statement_C : Prop :=
  ∀ n : ℤ, is_zero n → (is_positive n ∨ is_negative n)

-- Statement D: A rational number is either a positive number or a negative number.
def statement_D : Prop :=
  ∀ q : ℚ, (q ≠ 0 → (is_positive q.num ∨ is_negative q.num))

-- The problem is to prove that statement B is the only correct statement.
theorem correct_statement_is_B : statement_B ∧ ¬statement_A ∧ ¬statement_C ∧ ¬statement_D :=
by sorry

end correct_statement_is_B_l476_47690


namespace sum_of_six_consecutive_integers_l476_47637

theorem sum_of_six_consecutive_integers (m : ℤ) : 
  (m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)) = 6 * m + 15 :=
by
  sorry

end sum_of_six_consecutive_integers_l476_47637


namespace range_of_a_l476_47645

noncomputable def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a * x^2 - x - 1

theorem range_of_a {a : ℝ} : is_monotonic (f a) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end range_of_a_l476_47645


namespace either_d_or_2d_is_perfect_square_l476_47644

theorem either_d_or_2d_is_perfect_square
  (a c d : ℕ) (hrel_prime : Nat.gcd a c = 1) (hd : ∃ D : ℝ, D = d ∧ (D:ℝ) > 0)
  (hdiam : d^2 = 2 * a^2 + c^2) :
  ∃ m : ℕ, m^2 = d ∨ m^2 = 2 * d :=
by
  sorry

end either_d_or_2d_is_perfect_square_l476_47644


namespace average_gpa_of_whole_class_l476_47629

-- Define the conditions
variables (n : ℕ)
def num_students_in_group1 := n / 3
def num_students_in_group2 := 2 * n / 3

def gpa_group1 := 15
def gpa_group2 := 18

-- Lean statement for the proof problem
theorem average_gpa_of_whole_class (hn_pos : 0 < n):
  ((num_students_in_group1 * gpa_group1) + (num_students_in_group2 * gpa_group2)) / n = 17 :=
sorry

end average_gpa_of_whole_class_l476_47629


namespace quadratic_root_sum_eight_l476_47635

theorem quadratic_root_sum_eight (p r : ℝ) (hp : p > 0) (hr : r > 0) 
  (h : ∀ (x₁ x₂ : ℝ), (x₁ + x₂ = p) -> (x₁ * x₂ = r) -> (x₁ + x₂ = 8)) : r = 8 :=
sorry

end quadratic_root_sum_eight_l476_47635


namespace three_x_squared_y_squared_eq_588_l476_47679

theorem three_x_squared_y_squared_eq_588 (x y : ℤ) 
  (h : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 :=
sorry

end three_x_squared_y_squared_eq_588_l476_47679


namespace tory_earned_more_than_bert_l476_47678

open Real

noncomputable def bert_day1_earnings : ℝ :=
  let initial_sales := 12 * 18
  let discounted_sales := 3 * (18 - 0.15 * 18)
  let total_sales := initial_sales - 3 * 18 + discounted_sales
  total_sales * 0.95

noncomputable def tory_day1_earnings : ℝ :=
  let initial_sales := 15 * 20
  let discounted_sales := 5 * (20 - 0.10 * 20)
  let total_sales := initial_sales - 5 * 20 + discounted_sales
  total_sales * 0.95

noncomputable def bert_day2_earnings : ℝ :=
  let sales := 10 * 15
  (sales * 0.95) * 1.4

noncomputable def tory_day2_earnings : ℝ :=
  let sales := 8 * 18
  (sales * 0.95) * 1.4

noncomputable def bert_total_earnings : ℝ := bert_day1_earnings + bert_day2_earnings

noncomputable def tory_total_earnings : ℝ := tory_day1_earnings + tory_day2_earnings

noncomputable def earnings_difference : ℝ := tory_total_earnings - bert_total_earnings

theorem tory_earned_more_than_bert :
  earnings_difference = 71.82 := by
  sorry

end tory_earned_more_than_bert_l476_47678


namespace find_numbers_l476_47636

def is_7_digit (n : ℕ) : Prop := n ≥ 1000000 ∧ n < 10000000
def is_14_digit (n : ℕ) : Prop := n >= 10^13 ∧ n < 10^14

theorem find_numbers (x y z : ℕ) (hx7 : is_7_digit x) (hy7 : is_7_digit y) (hz14 : is_14_digit z) :
  3 * x * y = z ∧ z = 10^7 * x + y → 
  x = 1666667 ∧ y = 3333334 ∧ z = 16666673333334 := 
by
  sorry

end find_numbers_l476_47636


namespace largest_number_not_sum_of_two_composites_l476_47611

-- Define composite numbers
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Define the problem predicate
def cannot_be_expressed_as_sum_of_two_composites (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_number_not_sum_of_two_composites : 
  ∃ n, cannot_be_expressed_as_sum_of_two_composites n ∧ 
  (∀ m, cannot_be_expressed_as_sum_of_two_composites m → m ≤ n) ∧ 
  n = 11 :=
by {
  sorry
}

end largest_number_not_sum_of_two_composites_l476_47611


namespace problem_proof_l476_47600

variable {α : Type*}
noncomputable def op (a b : ℝ) : ℝ := 1/a + 1/b
theorem problem_proof (a b : ℝ) (h : op a (-b) = 2) : (3 * a * b) / (2 * a - 2 * b) = -3/4 :=
by
  sorry

end problem_proof_l476_47600


namespace solve_cubic_equation_l476_47612

theorem solve_cubic_equation :
  ∀ x : ℝ, x^3 = 13 * x + 12 ↔ x = 4 ∨ x = -1 ∨ x = -3 :=
by
  sorry

end solve_cubic_equation_l476_47612


namespace work_completion_time_l476_47655

-- Define the rate of work done by a, b, and c.
def rate_a := 1 / 4
def rate_b := 1 / 12
def rate_c := 1 / 6

-- Define the time each person starts working and the cycle pattern.
def start_time : ℕ := 6 -- in hours
def cycle_pattern := [rate_a, rate_b, rate_c]

-- Calculate the total amount of work done in one cycle of 3 hours.
def work_per_cycle := (rate_a + rate_b + rate_c)

-- Calculate the total time to complete the work.
def total_time_to_complete_work := 2 * 3 -- number of cycles times 3 hours per cycle

-- Calculate the time of completion.
def completion_time := start_time + total_time_to_complete_work

-- Theorem to prove the work completion time.
theorem work_completion_time : completion_time = 12 := 
by
  -- Proof can be filled in here
  sorry

end work_completion_time_l476_47655


namespace ratio_of_runs_l476_47603

theorem ratio_of_runs (A B C : ℕ) (h1 : B = C / 5) (h2 : A + B + C = 95) (h3 : C = 75) :
  A / B = 1 / 3 :=
by sorry

end ratio_of_runs_l476_47603


namespace kayak_manufacture_total_l476_47673

theorem kayak_manufacture_total :
  let feb : ℕ := 5
  let mar : ℕ := 3 * feb
  let apr : ℕ := 3 * mar
  let may : ℕ := 3 * apr
  feb + mar + apr + may = 200 := by
  sorry

end kayak_manufacture_total_l476_47673


namespace remainder_when_divided_by_8_l476_47614

theorem remainder_when_divided_by_8 (x : ℤ) (k : ℤ) (h : x = 72 * k + 19) : x % 8 = 3 :=
by sorry

end remainder_when_divided_by_8_l476_47614


namespace determine_b_l476_47677

noncomputable def f (x b : ℝ) : ℝ := x^3 - b * x^2 + 1/2

theorem determine_b (b : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 b = 0 ∧ f x2 b = 0) → b = 3/2 :=
by
  sorry

end determine_b_l476_47677


namespace percentage_difference_between_chef_and_dishwasher_l476_47622

theorem percentage_difference_between_chef_and_dishwasher
    (manager_wage : ℝ)
    (dishwasher_wage : ℝ)
    (chef_wage : ℝ)
    (h1 : manager_wage = 6.50)
    (h2 : dishwasher_wage = manager_wage / 2)
    (h3 : chef_wage = manager_wage - 2.60) :
    (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 :=
by
  -- The proof would go here
  sorry

end percentage_difference_between_chef_and_dishwasher_l476_47622


namespace minor_axis_of_ellipse_l476_47669

noncomputable def length_minor_axis 
    (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (p3 : ℝ × ℝ) (p4 : ℝ × ℝ) (p5 : ℝ × ℝ) : ℝ :=
if h : (p1, p2, p3, p4, p5) = ((1, 0), (1, 3), (4, 0), (4, 3), (6, 1.5)) then 3 else 0

theorem minor_axis_of_ellipse (p1 p2 p3 p4 p5 : ℝ × ℝ) :
  p1 = (1, 0) → p2 = (1, 3) → p3 = (4, 0) → p4 = (4, 3) → p5 = (6, 1.5) →
  length_minor_axis p1 p2 p3 p4 p5 = 3 :=
by sorry

end minor_axis_of_ellipse_l476_47669


namespace effect_on_revenue_decrease_l476_47640

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q

def new_price (P : ℝ) : ℝ := P * 1.40

def new_quantity (Q : ℝ) : ℝ := Q * 0.65

def new_revenue (P Q : ℝ) : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue_decrease :
  new_revenue P Q = original_revenue P Q * 0.91 →
  new_revenue P Q - original_revenue P Q = original_revenue P Q * -0.09 :=
by
  sorry

end effect_on_revenue_decrease_l476_47640


namespace count_special_digits_base7_l476_47682

theorem count_special_digits_base7 : 
  let n := 2401
  let total_valid_numbers := n - 4^4
  total_valid_numbers = 2145 :=
by
  sorry

end count_special_digits_base7_l476_47682


namespace central_angle_unchanged_l476_47670

theorem central_angle_unchanged (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0) :
  (s / r) = (2 * s / (2 * r)) :=
by
  sorry

end central_angle_unchanged_l476_47670


namespace find_c_l476_47634

noncomputable def P (c : ℝ) (x : ℝ) : ℝ := x^3 - 3 * x^2 + c * x - 8

theorem find_c (c : ℝ) : (∀ x, P c (x + 2) = 0) → c = -14 :=
sorry

end find_c_l476_47634


namespace maximize_S_n_l476_47602

variable {a : ℕ → ℝ} -- Sequence term definition
variable {S : ℕ → ℝ} -- Sum of first n terms

-- Definitions based on conditions
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  n * a 1 + (n * (n - 1) / 2) * ((a 2) - (a 1))

axiom a1_positive (a1 : ℝ) : 0 < a1 -- given a1 > 0
axiom S3_eq_S16 (a1 d : ℝ) : sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16

-- Problem Statement
theorem maximize_S_n (a : ℕ → ℝ) (d : ℝ) : is_arithmetic_sequence a d →
  a 1 > 0 →
  sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16 →
  (∀ n, sum_of_first_n_terms a n = sum_of_first_n_terms a 9 ∨ sum_of_first_n_terms a n = sum_of_first_n_terms a 10) :=
by
  sorry

end maximize_S_n_l476_47602


namespace typist_original_salary_l476_47638

theorem typist_original_salary (S : ℝ) :
  (1.10 * S * 0.95 * 1.07 * 0.97 = 2090) → (S = 2090 / (1.10 * 0.95 * 1.07 * 0.97)) :=
by
  intro h
  sorry

end typist_original_salary_l476_47638


namespace proportional_x_y2_y_z2_l476_47626

variable {x y z k m c : ℝ}

theorem proportional_x_y2_y_z2 (h1 : x = k * y^2) (h2 : y = m / z^2) (h3 : x = 2) (hz4 : z = 4) (hz16 : z = 16):
  x = 1/128 :=
by
  sorry

end proportional_x_y2_y_z2_l476_47626


namespace braxton_total_earnings_l476_47687

-- Definitions of the given problem conditions
def students_ashwood : ℕ := 9
def days_ashwood : ℕ := 4
def students_braxton : ℕ := 6
def days_braxton : ℕ := 7
def students_cedar : ℕ := 8
def days_cedar : ℕ := 6

def total_payment : ℕ := 1080
def daily_wage_per_student : ℚ := total_payment / ((students_ashwood * days_ashwood) + 
                                                   (students_braxton * days_braxton) + 
                                                   (students_cedar * days_cedar))

-- The statement to be proven
theorem braxton_total_earnings :
  (students_braxton * days_braxton * daily_wage_per_student) = 360 := 
by
  sorry -- proof goes here

end braxton_total_earnings_l476_47687


namespace inequality_proof_l476_47630

variables {a b : ℝ}

theorem inequality_proof :
  a^2 + b^2 - 1 - a^2 * b^2 <= 0 ↔ (a^2 - 1) * (b^2 - 1) >= 0 :=
by sorry

end inequality_proof_l476_47630


namespace sqrt_3_between_neg_1_and_2_l476_47619

theorem sqrt_3_between_neg_1_and_2 : -1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by
  sorry

end sqrt_3_between_neg_1_and_2_l476_47619


namespace slope_of_line_l476_47639

-- Define the point and the line equation with a generic slope
def point : ℝ × ℝ := (-1, 2)

def line (a : ℝ) := a * (point.fst) + (point.snd) - 4 = 0

-- The main theorem statement
theorem slope_of_line (a : ℝ) (h : line a) : ∃ m : ℝ, m = 2 :=
by
  -- The slope of the line derived from the equation and condition
  sorry

end slope_of_line_l476_47639


namespace find_number_l476_47664

theorem find_number (x : ℝ) (h : (168 / 100) * x / 6 = 354.2) : x = 1265 := 
by
  sorry

end find_number_l476_47664
