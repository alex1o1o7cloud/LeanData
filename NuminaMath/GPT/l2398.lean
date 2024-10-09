import Mathlib

namespace solution_l2398_239876

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), (x > 0 ∧ y > 0) ∧ (6 * x^2 + 18 * x * y = 2 * x^3 + 3 * x^2 * y^2) ∧ x = (3 + Real.sqrt 153) / 4

theorem solution : problem_statement :=
by
  sorry

end solution_l2398_239876


namespace trapezium_area_example_l2398_239875

noncomputable def trapezium_area (a b h : ℝ) : ℝ := 1/2 * (a + b) * h

theorem trapezium_area_example :
  trapezium_area 20 18 16 = 304 :=
by
  -- The proof steps would go here, but we're skipping them.
  sorry

end trapezium_area_example_l2398_239875


namespace cannot_all_white_without_diagonals_cannot_all_white_with_diagonals_l2398_239862

/-- A 4x4 chessboard is entirely white except for one square which is black.
The allowed operations are flipping the colors of all squares in a column or in a row.
Prove that it is impossible to have all the squares the same color regardless of the position of the black square. -/
theorem cannot_all_white_without_diagonals :
  ∀ (i j : Fin 4), False :=
by sorry

/-- If diagonal flips are also allowed, prove that 
it is impossible to have all squares the same color if the black square is at certain positions. -/
theorem cannot_all_white_with_diagonals :
  ∀ (i j : Fin 4), (i, j) ≠ (0, 1) ∧ (i, j) ≠ (0, 2) ∧
                   (i, j) ≠ (1, 0) ∧ (i, j) ≠ (1, 3) ∧
                   (i, j) ≠ (2, 0) ∧ (i, j) ≠ (2, 3) ∧
                   (i, j) ≠ (3, 1) ∧ (i, j) ≠ (3, 2) → False :=
by sorry

end cannot_all_white_without_diagonals_cannot_all_white_with_diagonals_l2398_239862


namespace braiding_time_l2398_239815

variables (n_dancers : ℕ) (b_braids_per_dancer : ℕ) (t_seconds_per_braid : ℕ)

theorem braiding_time : n_dancers = 8 → b_braids_per_dancer = 5 → t_seconds_per_braid = 30 → 
  (n_dancers * b_braids_per_dancer * t_seconds_per_braid) / 60 = 20 :=
by
  intros
  sorry

end braiding_time_l2398_239815


namespace valid_numbers_count_l2398_239895

def count_valid_numbers (n : ℕ) : ℕ := 1 / 4 * (5^n + 2 * 3^n + 1)

theorem valid_numbers_count (n : ℕ) : count_valid_numbers n = (1 / 4) * (5^n + 2 * 3^n + 1) :=
by sorry

end valid_numbers_count_l2398_239895


namespace weight_of_new_student_l2398_239899

-- Definitions from conditions
def total_weight_19 : ℝ := 19 * 15
def total_weight_20 : ℝ := 20 * 14.9

-- Theorem to prove the weight of the new student
theorem weight_of_new_student : (total_weight_20 - total_weight_19) = 13 := by
  sorry

end weight_of_new_student_l2398_239899


namespace op_dot_of_10_5_l2398_239897

-- Define the operation \odot
def op_dot (a b : ℕ) : ℕ := a + (2 * a) / b

-- Theorem stating that 10 \odot 5 = 14
theorem op_dot_of_10_5 : op_dot 10 5 = 14 :=
by
  sorry

end op_dot_of_10_5_l2398_239897


namespace range_of_b_l2398_239836

-- Define the conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16
def line_eq (x y b : ℝ) : Prop := y = x + b
def distance_point_line_eq (x y b d : ℝ) : Prop := 
  d = abs (b) / (Real.sqrt 2)
def at_least_three_points_on_circle_at_distance_one (b : ℝ) : Prop := 
  ∃ p1 p2 p3 : ℝ × ℝ, circle_eq p1.1 p1.2 ∧ circle_eq p2.1 p2.2 ∧ circle_eq p3.1 p3.2 ∧ 
  distance_point_line_eq p1.1 p1.2 b 1 ∧ distance_point_line_eq p2.1 p2.2 b 1 ∧ distance_point_line_eq p3.1 p3.2 b 1

-- The theorem statement to prove
theorem range_of_b (b : ℝ) (h : at_least_three_points_on_circle_at_distance_one b) : 
  -3 * Real.sqrt 2 ≤ b ∧ b ≤ 3 * Real.sqrt 2 := 
sorry

end range_of_b_l2398_239836


namespace cube_volume_l2398_239869

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V, V = 125 := 
by
  sorry

end cube_volume_l2398_239869


namespace eddie_weekly_earnings_l2398_239854

theorem eddie_weekly_earnings :
  let mon_hours := 2.5
  let tue_hours := 7 / 6
  let wed_hours := 7 / 4
  let sat_hours := 3 / 4
  let weekday_rate := 4
  let saturday_rate := 6
  let mon_earnings := mon_hours * weekday_rate
  let tue_earnings := tue_hours * weekday_rate
  let wed_earnings := wed_hours * weekday_rate
  let sat_earnings := sat_hours * saturday_rate
  let total_earnings := mon_earnings + tue_earnings + wed_earnings + sat_earnings
  total_earnings = 26.17 := by
  simp only
  norm_num
  sorry

end eddie_weekly_earnings_l2398_239854


namespace joe_total_time_to_school_l2398_239867

theorem joe_total_time_to_school:
  ∀ (d r_w: ℝ), (1 / 3) * d = r_w * 9 →
                  4 * r_w * (2 * (r_w * 9) / (3 * (4 * r_w))) = (2 / 3) * d →
                  (1 / 3) * d / r_w + (2 / 3) * d / (4 * r_w) = 13.5 :=
by
  intros d r_w h1 h2
  sorry

end joe_total_time_to_school_l2398_239867


namespace solutions_to_equation_l2398_239822

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 10*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 12*x - 8)) = 0

theorem solutions_to_equation :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = -19 ∨ x = (5 + Real.sqrt 57) / 2 ∨ x = (5 - Real.sqrt 57) / 2) :=
sorry

end solutions_to_equation_l2398_239822


namespace verify_compound_interest_rate_l2398_239826

noncomputable def compound_interest_rate
  (P A : ℝ) (t n : ℕ) : ℝ :=
  let r := (A / P) ^ (1 / (n * t)) - 1
  n * r

theorem verify_compound_interest_rate :
  let P := 5000
  let A := 6800
  let t := 4
  let n := 1
  compound_interest_rate P A t n = 8.02 / 100 :=
by
  sorry

end verify_compound_interest_rate_l2398_239826


namespace local_minimum_condition_l2398_239818

-- Define the function f(x)
def f (x b : ℝ) : ℝ := x ^ 3 - 3 * b * x + 3 * b

-- Define the first derivative of f(x)
def f_prime (x b : ℝ) : ℝ := 3 * x ^ 2 - 3 * b

-- Define the second derivative of f(x)
def f_double_prime (x b : ℝ) : ℝ := 6 * x

-- Theorem stating that f(x) has a local minimum if and only if b > 0
theorem local_minimum_condition (b : ℝ) (x : ℝ) (h : f_prime x b = 0) : f_double_prime x b > 0 ↔ b > 0 :=
by sorry

end local_minimum_condition_l2398_239818


namespace fixed_point_of_function_l2398_239857

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : (2, 3) ∈ { (x, y) | y = 2 + a^(x-2) } :=
sorry

end fixed_point_of_function_l2398_239857


namespace min_value_of_f_l2398_239802

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  2 * x^3 - 6 * x^2 + m

theorem min_value_of_f :
  ∀ (m : ℝ),
    f 0 m = 3 →
    ∃ x min, x ∈ Set.Icc (-2:ℝ) (2:ℝ) ∧ min = f x m ∧ min = -37 :=
by
  intros m h
  have h' : f 0 m = 3 := h
  -- Proof omitted.
  sorry

end min_value_of_f_l2398_239802


namespace smallest_positive_integer_l2398_239877

def is_prime_gt_60 (n : ℕ) : Prop :=
  n > 60 ∧ Prime n

def smallest_integer_condition (k : ℕ) : Prop :=
  ¬ Prime k ∧ ¬ (∃ m : ℕ, m * m = k) ∧ 
  ∀ p : ℕ, Prime p → p ∣ k → p > 60

theorem smallest_positive_integer : ∃ k : ℕ, k = 4087 ∧ smallest_integer_condition k := by
  sorry

end smallest_positive_integer_l2398_239877


namespace no_solution_frac_eq_l2398_239833

theorem no_solution_frac_eq (k : ℝ) : (∀ x : ℝ, ¬(1 / (x + 1) = 3 * k / x)) ↔ (k = 0 ∨ k = 1 / 3) :=
by
  sorry

end no_solution_frac_eq_l2398_239833


namespace factorization_a_minus_b_l2398_239812

theorem factorization_a_minus_b (a b : ℤ) (h1 : 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) : a - b = -7 :=
by
  sorry

end factorization_a_minus_b_l2398_239812


namespace extremum_range_k_l2398_239886

noncomputable def f (x k : Real) : Real :=
  Real.exp x / x + k * (Real.log x - x)

/-- 
For the function f(x) = (exp(x) / x) + k * (log(x) - x), if x = 1 is the only extremum point, 
then k is in the interval (-∞, e].
-/
theorem extremum_range_k (k : Real) : 
  (∀ x : Real, (0 < x) → (f x k ≤ f 1 k)) → 
  k ≤ Real.exp 1 :=
sorry

end extremum_range_k_l2398_239886


namespace exists_perfect_square_subtraction_l2398_239865

theorem exists_perfect_square_subtraction {k : ℕ} (hk : k > 0) : 
  ∃ (n : ℕ), n > 0 ∧ ∃ m : ℕ, n * 2^k - 7 = m^2 := 
  sorry

end exists_perfect_square_subtraction_l2398_239865


namespace smallest_perimeter_of_acute_triangle_with_consecutive_sides_l2398_239813

theorem smallest_perimeter_of_acute_triangle_with_consecutive_sides :
  ∃ (a : ℕ), (a > 1) ∧ (∃ (b c : ℕ), b = a + 1 ∧ c = a + 2 ∧ (∃ (C : ℝ), a^2 + b^2 - c^2 < 0 ∧ c = 4)) ∧ (a + (a + 1) + (a + 2) = 9) :=
by {
  sorry
}

end smallest_perimeter_of_acute_triangle_with_consecutive_sides_l2398_239813


namespace paint_mixer_days_l2398_239896

/-- Making an equal number of drums of paint each day, a paint mixer takes three days to make 18 drums of paint.
    We want to determine how many days it will take for him to make 360 drums of paint. -/
theorem paint_mixer_days (n : ℕ) (h1 : n > 0) 
  (h2 : 3 * n = 18) : 
  360 / n = 60 := by
  sorry

end paint_mixer_days_l2398_239896


namespace prime_p_prime_p₁₀_prime_p₁₄_l2398_239889

theorem prime_p_prime_p₁₀_prime_p₁₄ (p : ℕ) (h₀p : Nat.Prime p) 
  (h₁ : Nat.Prime (p + 10)) (h₂ : Nat.Prime (p + 14)) : p = 3 := by
  sorry

end prime_p_prime_p₁₀_prime_p₁₄_l2398_239889


namespace july_percentage_is_correct_l2398_239839

def total_scientists : ℕ := 120
def july_scientists : ℕ := 16
def july_percentage : ℚ := (july_scientists : ℚ) / (total_scientists : ℚ) * 100

theorem july_percentage_is_correct : july_percentage = 13.33 := 
by 
  -- Provides the proof directly as a statement
  sorry

end july_percentage_is_correct_l2398_239839


namespace matrix_exponentiation_l2398_239864

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2],
    ![2, -1]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-4, 6],
    ![-6, 5]]

theorem matrix_exponentiation :
  A^4 = B :=
by
  sorry

end matrix_exponentiation_l2398_239864


namespace solution_set_abs_inequality_l2398_239884

theorem solution_set_abs_inequality : {x : ℝ | |x - 2| < 1} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end solution_set_abs_inequality_l2398_239884


namespace sqrt_expression_meaningful_l2398_239801

theorem sqrt_expression_meaningful {x : ℝ} : (2 * x - 4) ≥ 0 → x ≥ 2 :=
by
  intro h
  sorry

end sqrt_expression_meaningful_l2398_239801


namespace additional_hours_equal_five_l2398_239832

-- The total hovering time constraint over two days
def total_time : ℕ := 24

-- Hovering times for each zone on the first day
def day1_mountain_time : ℕ := 3
def day1_central_time : ℕ := 4
def day1_eastern_time : ℕ := 2

-- Additional hours on the second day (variables M, C, E)
variables (M C E : ℕ)

-- The main proof statement
theorem additional_hours_equal_five 
  (h : day1_mountain_time + M + day1_central_time + C + day1_eastern_time + E = total_time) :
  M = 5 ∧ C = 5 ∧ E = 5 :=
by
  sorry

end additional_hours_equal_five_l2398_239832


namespace tara_additional_stamps_l2398_239898

def stamps_needed (current_stamps total_stamps : Nat) : Nat :=
  if total_stamps % 9 == 0 then 0 else 9 - (total_stamps % 9)

theorem tara_additional_stamps :
  stamps_needed 38 45 = 7 := by
  sorry

end tara_additional_stamps_l2398_239898


namespace loaves_at_start_l2398_239806

variable (X : ℕ) -- X represents the number of loaves at the start of the day.

-- Conditions given in the problem:
def final_loaves (X : ℕ) : Prop := X - 629 + 489 = 2215

-- The theorem to be proved:
theorem loaves_at_start (h : final_loaves X) : X = 2355 :=
by sorry

end loaves_at_start_l2398_239806


namespace car_distance_l2398_239882

theorem car_distance (t : ℚ) (s : ℚ) (d : ℚ) 
(h1 : t = 2 + 2 / 5) 
(h2 : s = 260) 
(h3 : d = s * t) : 
d = 624 := by
  sorry

end car_distance_l2398_239882


namespace min_max_product_l2398_239853

noncomputable def min_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the minimum value of 3x^2 + 4xy + 3y^2
  sorry

noncomputable def max_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the maximum value of 3x^2 + 4xy + 3y^2
  sorry

theorem min_max_product (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) :
  min_value x y h * max_value x y h = 7 / 16 :=
sorry

end min_max_product_l2398_239853


namespace count_valid_subsets_l2398_239803

theorem count_valid_subsets : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ A ∈ S, A ⊆ {1, 2, 3, 4, 5} ∧ 
    (∀ a ∈ A, 6 - a ∈ A)) ∧ 
    S.card = 7 := 
sorry

end count_valid_subsets_l2398_239803


namespace melanie_bought_books_l2398_239894

-- Defining the initial number of books and final number of books
def initial_books : ℕ := 41
def final_books : ℕ := 87

-- Theorem stating that Melanie bought 46 books at the yard sale
theorem melanie_bought_books : (final_books - initial_books) = 46 := by
  sorry

end melanie_bought_books_l2398_239894


namespace num_dress_designs_l2398_239834

-- Define the number of fabric colors and patterns
def fabric_colors : ℕ := 4
def patterns : ℕ := 5

-- Define the number of possible dress designs
def total_dress_designs : ℕ := fabric_colors * patterns

-- State the theorem that needs to be proved
theorem num_dress_designs : total_dress_designs = 20 := by
  sorry

end num_dress_designs_l2398_239834


namespace find_initial_oranges_l2398_239859

variable (O : ℕ)
variable (reserved_fraction : ℚ := 1 / 4)
variable (sold_fraction : ℚ := 3 / 7)
variable (rotten_oranges : ℕ := 4)
variable (good_oranges_today : ℕ := 32)

-- Define the total oranges before finding the rotten oranges
def oranges_before_rotten := good_oranges_today + rotten_oranges

-- Define the remaining fraction of oranges after reserving for friends and selling some
def remaining_fraction := (1 - reserved_fraction) * (1 - sold_fraction)

-- State the theorem to be proven
theorem find_initial_oranges (h : remaining_fraction * O = oranges_before_rotten) : O = 84 :=
sorry

end find_initial_oranges_l2398_239859


namespace fixed_point_exists_l2398_239838

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y : ℝ, (x = 2 ∧ y = -2 ∧ (ax - 5 = y)) :=
by
  sorry

end fixed_point_exists_l2398_239838


namespace probability_defective_unit_l2398_239885

theorem probability_defective_unit 
  (T : ℝ)
  (machine_a_output : ℝ := 0.4 * T)
  (machine_b_output : ℝ := 0.6 * T)
  (machine_a_defective_rate : ℝ := 9 / 1000)
  (machine_b_defective_rate : ℝ := 1 / 50)
  (total_defective_units : ℝ := (machine_a_output * machine_a_defective_rate) + (machine_b_output * machine_b_defective_rate))
  (probability_defective : ℝ := total_defective_units / T) :
  probability_defective = 0.0156 :=
by
  sorry

end probability_defective_unit_l2398_239885


namespace train_length_l2398_239866

theorem train_length (S L : ℝ)
  (h1 : L = S * 11)
  (h2 : L + 120 = S * 22) : 
  L = 120 := 
by
  -- proof goes here
  sorry

end train_length_l2398_239866


namespace abs_diff_of_numbers_l2398_239840

theorem abs_diff_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 :=
by
  sorry

end abs_diff_of_numbers_l2398_239840


namespace inverse_proportion_quadrant_l2398_239827

theorem inverse_proportion_quadrant (k : ℝ) (h : k < 0) : 
  ∀ x : ℝ, (0 < x → y = k / x → y < 0) ∧ (x < 0 → y = k / x → 0 < y) :=
by
  sorry

end inverse_proportion_quadrant_l2398_239827


namespace value_of_x_l2398_239829

def is_whole_number (n : ℝ) : Prop := ∃ (k : ℤ), n = k

theorem value_of_x (n : ℝ) (x : ℝ) :
  n = 1728 →
  is_whole_number (Real.log n / Real.log x + Real.log n / Real.log 12) →
  x = 12 :=
by
  intro h₁ h₂
  sorry

end value_of_x_l2398_239829


namespace article_word_limit_l2398_239817

theorem article_word_limit 
  (total_pages : ℕ) (large_font_pages : ℕ) (words_per_large_page : ℕ) 
  (words_per_small_page : ℕ) (remaining_pages : ℕ) (total_words : ℕ)
  (h1 : total_pages = 21) 
  (h2 : large_font_pages = 4) 
  (h3 : words_per_large_page = 1800) 
  (h4 : words_per_small_page = 2400) 
  (h5 : remaining_pages = total_pages - large_font_pages) 
  (h6 : total_words = large_font_pages * words_per_large_page + remaining_pages * words_per_small_page) :
  total_words = 48000 := 
by
  sorry

end article_word_limit_l2398_239817


namespace shoe_length_increase_l2398_239871

theorem shoe_length_increase
  (L : ℝ)
  (x : ℝ)
  (h1 : L + 9*x = L * 1.2)
  (h2 : L + 7*x = 10.4) :
  x = 0.2 :=
by
  sorry

end shoe_length_increase_l2398_239871


namespace white_chocolate_bars_sold_l2398_239874

theorem white_chocolate_bars_sold (W D : ℕ) (h1 : D = 15) (h2 : W / D = 4 / 3) : W = 20 :=
by
  -- This is where the proof would go.
  sorry

end white_chocolate_bars_sold_l2398_239874


namespace julia_monday_kids_l2398_239843

theorem julia_monday_kids (x : ℕ) (h1 : x + 14 = 16) : x = 2 := 
by
  sorry

end julia_monday_kids_l2398_239843


namespace largest_root_in_range_l2398_239848

-- Define the conditions for the equation parameters
variables (a0 a1 a2 : ℝ)
-- Define the conditions for the absolute value constraints
variables (h0 : |a0| < 2) (h1 : |a1| < 2) (h2 : |a2| < 2)

-- Define the equation
def cubic_equation (x : ℝ) : ℝ := x^3 + a2 * x^2 + a1 * x + a0

-- Define the property we want to prove about the largest positive root r
theorem largest_root_in_range :
  ∃ r > 0, (∃ x, cubic_equation a0 a1 a2 x = 0 ∧ r = x) ∧ (5 / 2 < r ∧ r < 3) :=
by sorry

end largest_root_in_range_l2398_239848


namespace square_neg_2x_squared_l2398_239847

theorem square_neg_2x_squared (x : ℝ) : (-2 * x ^ 2) ^ 2 = 4 * x ^ 4 :=
by
  sorry

end square_neg_2x_squared_l2398_239847


namespace time_for_A_and_C_l2398_239842

variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := A + B = 1 / 8
def condition2 : Prop := B + C = 1 / 12
def condition3 : Prop := A + B + C = 1 / 6

theorem time_for_A_and_C (h1 : condition1 A B)
                        (h2 : condition2 B C)
                        (h3 : condition3 A B C) :
  1 / (A + C) = 8 :=
sorry

end time_for_A_and_C_l2398_239842


namespace sin_600_eq_neg_sqrt_3_div_2_l2398_239879

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * (Real.pi / 180)) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l2398_239879


namespace no_more_than_four_intersection_points_l2398_239873

noncomputable def conic1 (a b c d e f : ℝ) (x y : ℝ) : Prop := 
  a * x^2 + 2 * b * x * y + c * y^2 + 2 * d * x + 2 * e * y = f

noncomputable def conic2_param (P Q A : ℝ → ℝ) (t : ℝ) : ℝ × ℝ :=
  (P t / A t, Q t / A t)

theorem no_more_than_four_intersection_points (a b c d e f : ℝ)
  (P Q A : ℝ → ℝ) :
  (∃ t1 t2 t3 t4 t5,
    conic1 a b c d e f (P t1 / A t1) (Q t1 / A t1) ∧
    conic1 a b c d e f (P t2 / A t2) (Q t2 / A t2) ∧
    conic1 a b c d e f (P t3 / A t3) (Q t3 / A t3) ∧
    conic1 a b c d e f (P t4 / A t4) (Q t4 / A t4) ∧
    conic1 a b c d e f (P t5 / A t5) (Q t5 / A t5)) → false :=
sorry

end no_more_than_four_intersection_points_l2398_239873


namespace complement_A_is_closed_interval_l2398_239850

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A with the given condition
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U
def complement_A : Set ℝ := Set.compl A

theorem complement_A_is_closed_interval :
  complement_A = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry  -- Proof to be inserted

end complement_A_is_closed_interval_l2398_239850


namespace sum_fifth_powers_l2398_239893

theorem sum_fifth_powers (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^5 + b^5 + c^5 = 98 / 6 := 
by 
  sorry

end sum_fifth_powers_l2398_239893


namespace red_balls_count_l2398_239878

theorem red_balls_count (w r : ℕ) (h1 : w = 16) (h2 : 4 * r = 3 * w) : r = 12 :=
by
  sorry

end red_balls_count_l2398_239878


namespace james_older_brother_is_16_l2398_239872

variables (John James James_older_brother : ℕ)

-- Given conditions
def current_age_john : ℕ := 39
def three_years_ago_john (caj : ℕ) : ℕ := caj - 3
def twice_as_old_condition (ja : ℕ) (james_age_in_6_years : ℕ) : Prop :=
  ja = 2 * james_age_in_6_years
def james_age_in_6_years (jc : ℕ) : ℕ := jc + 6
def james_older_brother_age (jc : ℕ) : ℕ := jc + 4

-- Theorem to be proved
theorem james_older_brother_is_16
  (H1 : current_age_john = John)
  (H2 : three_years_ago_john current_age_john = 36)
  (H3 : twice_as_old_condition 36 (james_age_in_6_years James))
  (H4 : james_older_brother_age James = James_older_brother) :
  James_older_brother = 16 := sorry

end james_older_brother_is_16_l2398_239872


namespace machines_make_2550_copies_l2398_239844

def total_copies (rate1 rate2 : ℕ) (time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

theorem machines_make_2550_copies :
  total_copies 30 55 30 = 2550 :=
by
  unfold total_copies
  decide

end machines_make_2550_copies_l2398_239844


namespace finite_perfect_squares_l2398_239845

noncomputable def finite_squares (a b : ℕ) : Prop :=
  ∃ (f : Finset ℕ), ∀ n, n ∈ f ↔ 
    ∃ (x y : ℕ), a * n ^ 2 + b = x ^ 2 ∧ a * (n + 1) ^ 2 + b = y ^ 2

theorem finite_perfect_squares (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  finite_squares a b :=
sorry

end finite_perfect_squares_l2398_239845


namespace find_n_l2398_239825

theorem find_n (x n : ℝ) (h1 : ((x / n) * 5) + 10 - 12 = 48) (h2 : x = 40) : n = 4 :=
sorry

end find_n_l2398_239825


namespace no_integral_value_2001_l2398_239849

noncomputable def P (x : ℤ) : ℤ := sorry -- Polynomial definition needs to be filled in

theorem no_integral_value_2001 (a0 a1 a2 a3 a4 : ℤ) (x1 x2 x3 x4 : ℤ) :
  (P x1 = 2020) ∧ (P x2 = 2020) ∧ (P x3 = 2020) ∧ (P x4 = 2020) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  ¬ ∃ x : ℤ, P x = 2001 :=
sorry

end no_integral_value_2001_l2398_239849


namespace quadratic_discriminant_l2398_239880

-- Define the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -11
def c : ℤ := 2

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- assert the discriminant for given coefficients
theorem quadratic_discriminant : discriminant a b c = 81 :=
by
  sorry

end quadratic_discriminant_l2398_239880


namespace part1_minimum_value_part2_max_k_l2398_239808

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x : ℝ) : ℝ := (x + x * Real.log x) / (x - 1)

theorem part1_minimum_value : ∃ x₀ : ℝ, x₀ = Real.exp (-2) ∧ f x₀ = -Real.exp (-2) := 
by
  use Real.exp (-2)
  sorry

theorem part2_max_k (k : ℤ) : (∀ x > 1, f x > k * (x - 1)) → k ≤ 3 := 
by
  sorry

end part1_minimum_value_part2_max_k_l2398_239808


namespace reaction_completion_l2398_239819

-- Definitions from conditions
def NaOH_moles : ℕ := 2
def H2O_moles : ℕ := 2

-- Given the balanced equation
-- 2 NaOH + H2SO4 → Na2SO4 + 2 H2O

theorem reaction_completion (H2SO4_moles : ℕ) :
  (2 * (NaOH_moles / 2)) = H2O_moles → H2SO4_moles = 1 :=
by 
  -- Skip proof
  sorry

end reaction_completion_l2398_239819


namespace moles_of_Cl2_required_l2398_239816

theorem moles_of_Cl2_required (n_C2H6 n_HCl : ℕ) (balance : n_C2H6 = 3) (HCl_needed : n_HCl = 6) :
  ∃ n_Cl2 : ℕ, n_Cl2 = 9 :=
by
  sorry

end moles_of_Cl2_required_l2398_239816


namespace perfect_square_sequence_l2398_239890

theorem perfect_square_sequence (k : ℤ) (y : ℕ → ℤ) :
  (y 1 = 1) ∧ (y 2 = 1) ∧
  (∀ n : ℕ, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) →
  (∀ n ≥ 1, ∃ m : ℤ, y n = m^2) ↔ (k = 1 ∨ k = 3) :=
sorry

end perfect_square_sequence_l2398_239890


namespace six_star_three_l2398_239831

def binary_op (x y : ℕ) : ℕ := 4 * x + 5 * y - x * y

theorem six_star_three : binary_op 6 3 = 21 := by
  sorry

end six_star_three_l2398_239831


namespace prove_f_f_x_eq_4_prove_f_f_x_eq_5_l2398_239841

variable (f : ℝ → ℝ)

-- Conditions
axiom f_of_4 : f (-2) = 4 ∧ f 2 = 4 ∧ f 6 = 4
axiom f_of_5 : f (-4) = 5 ∧ f 4 = 5

-- Intermediate Values
axiom f_inv_of_4 : f 0 = -2 ∧ f (-1) = 2 ∧ f 3 = 6
axiom f_inv_of_5 : f 2 = 4

theorem prove_f_f_x_eq_4 :
  {x : ℝ | f (f x) = 4} = {0, -1, 3} :=
by
  sorry

theorem prove_f_f_x_eq_5 :
  {x : ℝ | f (f x) = 5} = {2} :=
by
  sorry

end prove_f_f_x_eq_4_prove_f_f_x_eq_5_l2398_239841


namespace circles_intersect_at_2_points_l2398_239814

theorem circles_intersect_at_2_points :
  let circle1 := { p : ℝ × ℝ | (p.1 - 5 / 2) ^ 2 + p.2 ^ 2 = 25 / 4 }
  let circle2 := { p : ℝ × ℝ | p.1 ^ 2 + (p.2 - 7 / 2) ^ 2 = 49 / 4 }
  ∃ (P1 P2 : ℝ × ℝ), P1 ∈ circle1 ∧ P1 ∈ circle2 ∧
                     P2 ∈ circle1 ∧ P2 ∈ circle2 ∧
                     P1 ≠ P2 ∧ ∀ (P : ℝ × ℝ), P ∈ circle1 ∧ P ∈ circle2 → P = P1 ∨ P = P2 := 
by 
  sorry

end circles_intersect_at_2_points_l2398_239814


namespace isosceles_triangle_perimeter_l2398_239861

variable (a b c : ℝ)
variable (h1 : a = 4 ∨ a = 8)
variable (h2 : b = 4 ∨ b = 8)
variable (h3 : a = b ∨ c = 8)

theorem isosceles_triangle_perimeter (h : a + b + c = 20) : a = b ∨ b = 8 ∧ (a = 8 ∧ c = 4 ∨ b = c) := 
  by
  sorry

end isosceles_triangle_perimeter_l2398_239861


namespace function_domain_length_correct_l2398_239856

noncomputable def function_domain_length : ℕ :=
  let p : ℕ := 240 
  let q : ℕ := 1
  p + q

theorem function_domain_length_correct : function_domain_length = 241 := by
  sorry

end function_domain_length_correct_l2398_239856


namespace days_worked_per_week_l2398_239805

theorem days_worked_per_week (toys_per_week toys_per_day : ℕ) (h1 : toys_per_week = 5500) (h2 : toys_per_day = 1375) : toys_per_week / toys_per_day = 4 := by
  sorry

end days_worked_per_week_l2398_239805


namespace find_first_term_and_ratio_l2398_239858

variable (b1 q : ℝ)

-- Conditions
def infinite_geometric_series (q : ℝ) : Prop := |q| < 1

def sum_odd_even_difference (b1 q : ℝ) : Prop := 
  b1 / (1 - q^2) = 2 + (b1 * q) / (1 - q^2)

def sum_square_odd_even_difference (b1 q : ℝ) : Prop :=
  b1^2 / (1 - q^4) - (b1^2 * q^2) / (1 - q^4) = 36 / 5

-- Proof problem
theorem find_first_term_and_ratio (b1 q : ℝ) 
  (h1 : infinite_geometric_series q) 
  (h2 : sum_odd_even_difference b1 q)
  (h3 : sum_square_odd_even_difference b1 q) : 
  b1 = 3 ∧ q = 1 / 2 := by
  sorry

end find_first_term_and_ratio_l2398_239858


namespace pies_sold_in_a_week_l2398_239881

theorem pies_sold_in_a_week : 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 114 :=
by 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  have h1 : Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 8 + 12 + 14 + 20 + 20 + 20 + 20 := by rfl
  have h2 : 8 + 12 + 14 + 20 + 20 + 20 + 20 = 114 := by norm_num
  exact h1.trans h2

end pies_sold_in_a_week_l2398_239881


namespace proposition_does_not_hold_6_l2398_239846

-- Define P as a proposition over positive integers
variable (P : ℕ → Prop)

-- Assumptions
variables (h1 : ∀ k : ℕ, P k → P (k + 1))  
variable (h2 : ¬ P 7)

-- Statement of the Problem
theorem proposition_does_not_hold_6 : ¬ P 6 :=
sorry

end proposition_does_not_hold_6_l2398_239846


namespace total_cats_and_kittens_received_l2398_239855

theorem total_cats_and_kittens_received 
  (adult_cats : ℕ) 
  (perc_female : ℕ) 
  (frac_litters : ℚ) 
  (kittens_per_litter : ℕ)
  (rescued_cats : ℕ) 
  (total_received : ℕ)
  (h1 : adult_cats = 120)
  (h2 : perc_female = 60)
  (h3 : frac_litters = 2/3)
  (h4 : kittens_per_litter = 3)
  (h5 : rescued_cats = 30)
  (h6 : total_received = 294) :
  adult_cats + rescued_cats + (frac_litters * (perc_female * adult_cats / 100) * kittens_per_litter) = total_received := 
sorry

end total_cats_and_kittens_received_l2398_239855


namespace min_value_max_value_l2398_239870

theorem min_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 0) := sorry

theorem max_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 44) := sorry

end min_value_max_value_l2398_239870


namespace find_difference_l2398_239804

noncomputable def expression (x y : ℝ) : ℝ :=
  (|x + y| / (|x| + |y|))^2

theorem find_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  let m := 0
  let M := 1
  M - m = 1 :=
by
  -- Please note that the proof is omitted and replaced with sorry
  sorry

end find_difference_l2398_239804


namespace triangle_area_six_parts_l2398_239811

theorem triangle_area_six_parts (S S₁ S₂ S₃ : ℝ) (h₁ : S₁ ≥ 0) (h₂ : S₂ ≥ 0) (h₃ : S₃ ≥ 0) :
  S = (Real.sqrt S₁ + Real.sqrt S₂ + Real.sqrt S₃) ^ 2 := 
sorry

end triangle_area_six_parts_l2398_239811


namespace power_zero_equals_one_specific_case_l2398_239807

theorem power_zero_equals_one 
    (a b : ℤ) 
    (h : a ≠ 0)
    (h2 : b ≠ 0) : 
    (a / b : ℚ) ^ 0 = 1 := 
by {
  sorry
}

-- Specific case
theorem specific_case : 
  ( ( (-123456789 : ℤ) / (9876543210 : ℤ) : ℚ ) ^ 0 = 1 ) := 
by {
  apply power_zero_equals_one;
  norm_num;
  sorry
}

end power_zero_equals_one_specific_case_l2398_239807


namespace maryann_work_time_l2398_239837

variables (C A R : ℕ)

theorem maryann_work_time
  (h1 : A = 2 * C)
  (h2 : R = 6 * C)
  (h3 : C + A + R = 1440) :
  C = 160 ∧ A = 320 ∧ R = 960 :=
by
  sorry

end maryann_work_time_l2398_239837


namespace product_ab_l2398_239892

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l2398_239892


namespace printer_z_time_l2398_239823

theorem printer_z_time (t_z : ℝ)
  (hx : (∀ (p : ℝ), p = 16))
  (hy : (∀ (q : ℝ), q = 12))
  (ratio : (16 / (1 /  ((1 / 12) + (1 / t_z)))) = 10 / 3) :
  t_z = 8 := by
  sorry

end printer_z_time_l2398_239823


namespace customer_count_l2398_239891

theorem customer_count :
  let initial_customers := 13
  let customers_after_first_leave := initial_customers - 5
  let customers_after_new_arrival := customers_after_first_leave + 4
  let customers_after_group_join := customers_after_new_arrival + 8
  let final_customers := customers_after_group_join - 6
  final_customers = 14 :=
by
  sorry

end customer_count_l2398_239891


namespace initial_number_of_cards_l2398_239830

theorem initial_number_of_cards (x : ℕ) (h : x + 76 = 79) : x = 3 :=
by
  sorry

end initial_number_of_cards_l2398_239830


namespace triangle_type_and_area_l2398_239821

theorem triangle_type_and_area (x : ℝ) (hpos : 0 < x) (h : 3 * x + 4 * x + 5 * x = 36) :
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  a^2 + b^2 = c^2 ∧ (1 / 2) * a * b = 54 :=
by {
  sorry
}

end triangle_type_and_area_l2398_239821


namespace trapezoid_CD_length_l2398_239863

/-- In trapezoid ABCD with AD parallel to BC and diagonals intersecting:
  - BD = 2
  - ∠DBC = 36°
  - ∠BDA = 72°
  - The ratio BC : AD = 5 : 3

We are to show that the length of CD is 4/3. --/
theorem trapezoid_CD_length
  {A B C D : Type}
  (BD : ℝ) (DBC : ℝ) (BDA : ℝ) (BC_over_AD : ℝ)
  (AD_parallel_BC : Prop) (diagonals_intersect : Prop)
  (hBD : BD = 2) 
  (hDBC : DBC = 36) 
  (hBDA : BDA = 72)
  (hBC_over_AD : BC_over_AD = 5 / 3) 
  :  CD = 4 / 3 :=
by
  sorry

end trapezoid_CD_length_l2398_239863


namespace negation_of_exactly_one_is_even_l2398_239828

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_is_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ ¬ is_even b ∧ is_even c))

def at_least_two_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ is_even b) ∨ (is_even b ∧ is_even c) ∨ (is_even a ∧ is_even c))

def all_are_odd (a b c : ℕ) : Prop := ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c 

theorem negation_of_exactly_one_is_even (a b c : ℕ) :
  ¬ exactly_one_is_even a b c ↔ at_least_two_even a b c ∨ all_are_odd a b c := by
  sorry

end negation_of_exactly_one_is_even_l2398_239828


namespace time_to_fill_is_correct_l2398_239810

-- Definitions of rates
variable (R_1 : ℚ) (R_2 : ℚ)

-- Conditions given in the problem
def rate1 := (1 : ℚ) / 8
def rate2 := (1 : ℚ) / 12

-- The resultant rate when both pipes work together
def combined_rate := rate1 + rate2

-- Calculate the time taken to fill the tank
def time_to_fill_tank := 1 / combined_rate

theorem time_to_fill_is_correct (h1 : R_1 = rate1) (h2 : R_2 = rate2) :
  time_to_fill_tank = 24 / 5 := by
  sorry

end time_to_fill_is_correct_l2398_239810


namespace investment_doubling_time_l2398_239883

theorem investment_doubling_time :
  ∀ (r : ℝ) (initial_investment future_investment : ℝ),
  r = 8 →
  initial_investment = 5000 →
  future_investment = 20000 →
  (future_investment = initial_investment * 2 ^ (70 / r * 2)) →
  70 / r * 2 = 17.5 :=
by
  intros r initial_investment future_investment h_r h_initial h_future h_double
  sorry

end investment_doubling_time_l2398_239883


namespace remainder_when_divided_by_6_eq_5_l2398_239860

theorem remainder_when_divided_by_6_eq_5 (k : ℕ) (hk1 : k % 5 = 2) (hk2 : k < 41) (hk3 : k % 7 = 3) : k % 6 = 5 :=
sorry

end remainder_when_divided_by_6_eq_5_l2398_239860


namespace find_f_4_l2398_239852

-- Lean code to encapsulate the conditions and the goal
theorem find_f_4 (f : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), x * f y = y * f x)
  (h2 : f 12 = 24) : 
  f 4 = 8 :=
sorry

end find_f_4_l2398_239852


namespace orchard_apples_relation_l2398_239851

/-- 
A certain orchard has 10 apple trees, and on average each tree can produce 200 apples. 
Based on experience, for each additional tree planted, the average number of apples produced per tree decreases by 5. 
We are to show that if the orchard has planted x additional apple trees and the total number of apples is y, then the relationship between y and x is:
y = (10 + x) * (200 - 5x)
-/
theorem orchard_apples_relation (x : ℕ) (y : ℕ) 
    (initial_trees : ℕ := 10)
    (initial_apples : ℕ := 200)
    (decrease_per_tree : ℕ := 5)
    (total_trees := initial_trees + x)
    (average_apples := initial_apples - decrease_per_tree * x)
    (total_apples := total_trees * average_apples) :
    y = total_trees * average_apples := 
  by 
    sorry

end orchard_apples_relation_l2398_239851


namespace find_integer_x_l2398_239868

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧
  2 < x ∧ x < 15 ∧
  -1 < x ∧ x < 7 ∧
  0 < x ∧ x < 4 ∧
  x + 1 < 5 → 
  x = 3 :=
by
  sorry

end find_integer_x_l2398_239868


namespace M_eq_N_l2398_239820

-- Define the sets M and N
def M : Set ℤ := {u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r}

-- Prove that M equals N
theorem M_eq_N : M = N := 
by {
  sorry
}

end M_eq_N_l2398_239820


namespace number_of_sets_l2398_239809

theorem number_of_sets (a n : ℕ) (M : Finset ℕ) (h_consecutive : ∀ x ∈ M, ∃ k, x = a + k ∧ k < n) (h_card : M.card ≥ 2) (h_sum : M.sum id = 2002) : n = 7 :=
sorry

end number_of_sets_l2398_239809


namespace find_third_number_l2398_239800

theorem find_third_number (N : ℤ) :
  (1274 % 12 = 2) ∧ (1275 % 12 = 3) ∧ (1285 % 12 = 1) ∧ ((1274 * 1275 * N * 1285) % 12 = 6) →
  N % 12 = 1 :=
by
  sorry

end find_third_number_l2398_239800


namespace Julio_current_age_l2398_239835

theorem Julio_current_age (J : ℕ) (James_current_age : ℕ) (h1 : James_current_age = 11)
    (h2 : J + 14 = 2 * (James_current_age + 14)) : 
    J = 36 := 
by 
  sorry

end Julio_current_age_l2398_239835


namespace fractional_eq_nonneg_solution_l2398_239888

theorem fractional_eq_nonneg_solution 
  (m x : ℝ)
  (h1 : x ≠ 2)
  (h2 : x ≥ 0)
  (eq_fractional : m / (x - 2) + 1 = x / (2 - x)) :
  m ≤ 2 ∧ m ≠ -2 := 
  sorry

end fractional_eq_nonneg_solution_l2398_239888


namespace at_least_one_alarm_rings_on_time_l2398_239887

-- Definitions for the problem
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.6

def prob_not_A : ℝ := 1 - prob_A
def prob_not_B : ℝ := 1 - prob_B
def prob_neither_A_nor_B : ℝ := prob_not_A * prob_not_B
def prob_at_least_one : ℝ := 1 - prob_neither_A_nor_B

-- Final statement
theorem at_least_one_alarm_rings_on_time : prob_at_least_one = 0.8 :=
by sorry

end at_least_one_alarm_rings_on_time_l2398_239887


namespace solve_for_diamond_l2398_239824

-- Define what it means for a digit to represent a base-9 number and base-10 number
noncomputable def fromBase (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * b + d) 0

-- The theorem we want to prove
theorem solve_for_diamond (diamond : ℕ) (h_digit : diamond < 10) :
  fromBase 9 [diamond, 3] = fromBase 10 [diamond, 2] → diamond = 1 :=
by 
  sorry

end solve_for_diamond_l2398_239824
