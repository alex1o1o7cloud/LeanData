import Mathlib

namespace NUMINAMATH_GPT_inequality_x2_gt_y2_plus_6_l38_3889

theorem inequality_x2_gt_y2_plus_6 (x y : ℝ) (h1 : x > y) (h2 : y > 3 / (x - y)) : x^2 > y^2 + 6 :=
sorry

end NUMINAMATH_GPT_inequality_x2_gt_y2_plus_6_l38_3889


namespace NUMINAMATH_GPT_tan_sum_l38_3869

open Real

theorem tan_sum 
  (α β γ θ φ : ℝ)
  (h1 : tan θ = (sin α * cos γ - sin β * sin γ) / (cos α * cos γ - cos β * sin γ))
  (h2 : tan φ = (sin α * sin γ - sin β * cos γ) / (cos α * sin γ - cos β * cos γ)) : 
  tan (θ + φ) = tan (α + β) :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_l38_3869


namespace NUMINAMATH_GPT_men_in_second_group_l38_3861

theorem men_in_second_group (m w : ℝ) (x : ℝ) 
  (h1 : 3 * m + 8 * w = x * m + 2 * w) 
  (h2 : 2 * m + 2 * w = (3 / 7) * (3 * m + 8 * w)) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_men_in_second_group_l38_3861


namespace NUMINAMATH_GPT_purely_imaginary_complex_l38_3810

theorem purely_imaginary_complex :
  ∀ (x y : ℤ), (x - 4) ≠ 0 → (y^2 - 3*y - 4) ≠ 0 → (∃ (z : ℂ), z = ⟨0, x^2 + 3*x - 4⟩) → 
    (x = 4 ∧ y ≠ 4 ∧ y ≠ -1) :=
by
  intro x y hx hy hz
  sorry

end NUMINAMATH_GPT_purely_imaginary_complex_l38_3810


namespace NUMINAMATH_GPT_average_weight_a_b_l38_3848

variables (A B C : ℝ)

theorem average_weight_a_b (h1 : (A + B + C) / 3 = 43)
                          (h2 : (B + C) / 2 = 43)
                          (h3 : B = 37) :
                          (A + B) / 2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_a_b_l38_3848


namespace NUMINAMATH_GPT_sales_on_third_day_l38_3888

variable (a m : ℕ)

def first_day_sales : ℕ := a
def second_day_sales : ℕ := 3 * a - 3 * m
def third_day_sales : ℕ := (3 * a - 3 * m) + m

theorem sales_on_third_day 
  (a m : ℕ) : third_day_sales a m = 3 * a - 2 * m :=
by
  -- Assuming the conditions as our definitions:
  let fds := first_day_sales a
  let sds := second_day_sales a m
  let tds := third_day_sales a m

  -- Proof direction:
  show tds = 3 * a - 2 * m
  sorry

end NUMINAMATH_GPT_sales_on_third_day_l38_3888


namespace NUMINAMATH_GPT_error_in_step_one_l38_3824

theorem error_in_step_one : 
  ∃ a b c d : ℝ, 
    (a * (x + 1) - b = c * (x - 2)) = (3 * (x + 1) - 6 = 2 * (x - 2)) → 
    a ≠ 3 ∨ b ≠ 6 ∨ c ≠ 2 := 
by
  sorry

end NUMINAMATH_GPT_error_in_step_one_l38_3824


namespace NUMINAMATH_GPT_class_size_l38_3863

theorem class_size
  (S_society : ℕ) (S_music : ℕ) (S_both : ℕ) (S : ℕ)
  (h_society : S_society = 25)
  (h_music : S_music = 32)
  (h_both : S_both = 27)
  (h_total : S = S_society + S_music - S_both) :
  S = 30 :=
by
  rw [h_society, h_music, h_both] at h_total
  exact h_total

end NUMINAMATH_GPT_class_size_l38_3863


namespace NUMINAMATH_GPT_union_of_sets_l38_3878

open Set

variable (a b : ℕ)

noncomputable def M : Set ℕ := {3, 2 * a}
noncomputable def N : Set ℕ := {a, b}

theorem union_of_sets (h : M a ∩ N a b = {2}) : M a ∪ N a b = {1, 2, 3} :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_union_of_sets_l38_3878


namespace NUMINAMATH_GPT_friendly_number_pair_a_equals_negative_three_fourths_l38_3818

theorem friendly_number_pair_a_equals_negative_three_fourths (a : ℚ) (h : (a / 2) + (3 / 4) = (a + 3) / 6) : 
  a = -3 / 4 :=
sorry

end NUMINAMATH_GPT_friendly_number_pair_a_equals_negative_three_fourths_l38_3818


namespace NUMINAMATH_GPT_points_on_line_with_slope_l38_3838

theorem points_on_line_with_slope :
  ∃ a b : ℝ, 
  (a - 3) ≠ 0 ∧ (b - 5) ≠ 0 ∧
  (7 - 5) / (a - 3) = 4 ∧ (b - 5) / (-1 - 3) = 4 ∧
  a = 7 / 2 ∧ b = -11 := 
by
  existsi 7 / 2
  existsi -11
  repeat {split}
  all_goals { sorry }

end NUMINAMATH_GPT_points_on_line_with_slope_l38_3838


namespace NUMINAMATH_GPT_right_triangle_345_l38_3898

def is_right_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

theorem right_triangle_345 : is_right_triangle 3 4 5 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_345_l38_3898


namespace NUMINAMATH_GPT_positive_number_square_roots_l38_3816

theorem positive_number_square_roots (a : ℝ) 
  (h1 : (2 * a - 1) ^ 2 = (a - 2) ^ 2) 
  (h2 : ∃ b : ℝ, b > 0 ∧ ((2 * a - 1) = b ∨ (a - 2) = b)) : 
  ∃ n : ℝ, n = 1 :=
by
  sorry

end NUMINAMATH_GPT_positive_number_square_roots_l38_3816


namespace NUMINAMATH_GPT_square_non_negative_is_universal_l38_3846

/-- The square of any real number is non-negative, which is a universal proposition. -/
theorem square_non_negative_is_universal : 
  ∀ x : ℝ, x^2 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_square_non_negative_is_universal_l38_3846


namespace NUMINAMATH_GPT_t_shirt_sale_revenue_per_minute_l38_3886

theorem t_shirt_sale_revenue_per_minute (total_tshirts : ℕ) (total_minutes : ℕ)
  (black_tshirts white_tshirts : ℕ) (cost_black cost_white : ℕ) 
  (half_total_tshirts : total_tshirts = black_tshirts + white_tshirts)
  (equal_halves : black_tshirts = white_tshirts)
  (black_price : cost_black = 30) (white_price : cost_white = 25)
  (total_time : total_minutes = 25)
  (total_sold : total_tshirts = 200) :
  ((black_tshirts * cost_black) + (white_tshirts * cost_white)) / total_minutes = 220 := by
  sorry

end NUMINAMATH_GPT_t_shirt_sale_revenue_per_minute_l38_3886


namespace NUMINAMATH_GPT_no_positive_real_solutions_l38_3871

theorem no_positive_real_solutions 
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^3 + y^3 + z^3 = x + y + z) (h2 : x^2 + y^2 + z^2 = x * y * z) :
  false :=
by sorry

end NUMINAMATH_GPT_no_positive_real_solutions_l38_3871


namespace NUMINAMATH_GPT_integers_square_less_than_three_times_l38_3817

theorem integers_square_less_than_three_times (x : ℤ) : x^2 < 3 * x ↔ x = 1 ∨ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_integers_square_less_than_three_times_l38_3817


namespace NUMINAMATH_GPT_dot_product_two_a_plus_b_with_a_l38_3801

-- Define vector a
def a : ℝ × ℝ := (2, -1)

-- Define vector b
def b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication of vector a by 2
def two_a : ℝ × ℝ := (2 * a.1, 2 * a.2)

-- Define the vector addition of 2a and b
def two_a_plus_b : ℝ × ℝ := (two_a.1 + b.1, two_a.2 + b.2)

-- Define dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that the dot product of (2 * a + b) and a equals 6
theorem dot_product_two_a_plus_b_with_a :
  dot_product two_a_plus_b a = 6 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_two_a_plus_b_with_a_l38_3801


namespace NUMINAMATH_GPT_cos_sum_is_zero_l38_3893

theorem cos_sum_is_zero (x y z : ℝ) 
  (h1: Real.cos (2 * x) + 2 * Real.cos (2 * y) + 3 * Real.cos (2 * z) = 0) 
  (h2: Real.sin (2 * x) + 2 * Real.sin (2 * y) + 3 * Real.sin (2 * z) = 0) : 
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_cos_sum_is_zero_l38_3893


namespace NUMINAMATH_GPT_find_function_l38_3809

theorem find_function (f : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, f n < f (n + 1)) →
  (∀ n : ℕ, f (f n) = n + 2 * k) →
  ∀ n : ℕ, f n = n + k := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_function_l38_3809


namespace NUMINAMATH_GPT_paper_stars_per_bottle_l38_3836

theorem paper_stars_per_bottle (a b total_bottles : ℕ) (h1 : a = 33) (h2 : b = 307) (h3 : total_bottles = 4) :
  (a + b) / total_bottles = 85 :=
by
  sorry

end NUMINAMATH_GPT_paper_stars_per_bottle_l38_3836


namespace NUMINAMATH_GPT_solution_set_of_inequality_l38_3862

def fraction_inequality_solution : Set ℝ := {x : ℝ | -4 < x ∧ x < -1}

theorem solution_set_of_inequality (x : ℝ) :
  (2 - x) / (x + 4) > 1 ↔ -4 < x ∧ x < -1 := by
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l38_3862


namespace NUMINAMATH_GPT_calories_difference_l38_3860

def calories_burnt (hours : ℕ) : ℕ := 30 * hours

theorem calories_difference :
  calories_burnt 5 - calories_burnt 2 = 90 :=
by
  sorry

end NUMINAMATH_GPT_calories_difference_l38_3860


namespace NUMINAMATH_GPT_greatest_x_value_l38_3837

theorem greatest_x_value :
  ∃ x, (∀ y, (y ≠ 6) → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧ (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧ x ≠ 6 ∧ x = -3 :=
sorry

end NUMINAMATH_GPT_greatest_x_value_l38_3837


namespace NUMINAMATH_GPT_neg_power_of_square_l38_3897

theorem neg_power_of_square (a : ℝ) : (-a^2)^3 = -a^6 :=
by sorry

end NUMINAMATH_GPT_neg_power_of_square_l38_3897


namespace NUMINAMATH_GPT_shifted_parabola_passes_through_neg1_1_l38_3854

def original_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 4

def shifted_parabola (x : ℝ) : ℝ := -(x - 1)^2 + 2

theorem shifted_parabola_passes_through_neg1_1 :
  shifted_parabola (-1) = 1 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_shifted_parabola_passes_through_neg1_1_l38_3854


namespace NUMINAMATH_GPT_heads_at_least_twice_in_5_tosses_l38_3875

noncomputable def probability_at_least_two_heads (n : ℕ) (p : ℚ) : ℚ :=
1 - (n : ℚ) * p^(n : ℕ)

theorem heads_at_least_twice_in_5_tosses :
  probability_at_least_two_heads 5 (1/2) = 13/16 :=
by
  sorry

end NUMINAMATH_GPT_heads_at_least_twice_in_5_tosses_l38_3875


namespace NUMINAMATH_GPT_triangle_existence_l38_3822

theorem triangle_existence 
  (h_a h_b m_a : ℝ) :
  (m_a ≥ h_a) → 
  ((h_a > 1/2 * h_b ∧ m_a > h_a → true ∨ false) ∧ 
  (m_a = h_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b < m_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b = m_a → false ∨ true) ∧ 
  (1/2 * h_b > m_a → false)) :=
by
  intro
  sorry

end NUMINAMATH_GPT_triangle_existence_l38_3822


namespace NUMINAMATH_GPT_leap_year_53_sundays_and_february_5_sundays_l38_3866

theorem leap_year_53_sundays_and_february_5_sundays :
  let Y := 366
  let W := 52
  ∃ (p : ℚ), p = (2/7) * (1/7) → p = 2/49
:=
by
  sorry

end NUMINAMATH_GPT_leap_year_53_sundays_and_february_5_sundays_l38_3866


namespace NUMINAMATH_GPT_number_of_students_with_type_B_l38_3865

theorem number_of_students_with_type_B
  (total_students : ℕ)
  (students_with_type_A : total_students ≠ 0 ∧ total_students ≠ 0 → 2 * total_students = 90)
  (students_with_type_B : 2 * total_students = 90) :
  2/5 * total_students = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_with_type_B_l38_3865


namespace NUMINAMATH_GPT_smallest_n_value_l38_3870

theorem smallest_n_value (n : ℕ) (h : 15 * n - 2 ≡ 0 [MOD 11]) : n = 6 :=
sorry

end NUMINAMATH_GPT_smallest_n_value_l38_3870


namespace NUMINAMATH_GPT_kite_diagonal_ratio_l38_3834

theorem kite_diagonal_ratio (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx1 : 0 ≤ x) (hx2 : x < a) (hy1 : 0 ≤ y) (hy2 : y < b)
  (orthogonal_diagonals : a^2 + y^2 = b^2 + x^2) :
  (a / b)^2 = 4 / 3 := 
sorry

end NUMINAMATH_GPT_kite_diagonal_ratio_l38_3834


namespace NUMINAMATH_GPT_min_frac_sum_l38_3856

noncomputable def min_value (x y : ℝ) : ℝ :=
  if (x + y = 1 ∧ x > 0 ∧ y > 0) then 1/x + 4/y else 0

theorem min_frac_sum (x y : ℝ) (h₁ : x + y = 1) (h₂: x > 0) (h₃: y > 0) : 
  min_value x y = 9 :=
sorry

end NUMINAMATH_GPT_min_frac_sum_l38_3856


namespace NUMINAMATH_GPT_sum_roots_of_quadratic_eq_l38_3899

theorem sum_roots_of_quadratic_eq (a b c: ℝ) (x: ℝ) :
    (a = 1) →
    (b = -7) →
    (c = -9) →
    (x ^ 2 - 7 * x + 2 = 11) →
    (∃ r1 r2 : ℝ, x ^ 2 - 7 * x - 9 = 0 ∧ r1 + r2 = 7) :=
by
  sorry

end NUMINAMATH_GPT_sum_roots_of_quadratic_eq_l38_3899


namespace NUMINAMATH_GPT_no_nat_fun_satisfying_property_l38_3895

theorem no_nat_fun_satisfying_property :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_no_nat_fun_satisfying_property_l38_3895


namespace NUMINAMATH_GPT_fraction_of_25_exact_value_l38_3835

-- Define the conditions
def eighty_percent_of_sixty : ℝ := 0.80 * 60
def smaller_by_twenty_eight (x : ℝ) : Prop := x * 25 = eighty_percent_of_sixty - 28

-- The proof problem
theorem fraction_of_25_exact_value (x : ℝ) : smaller_by_twenty_eight x → x = 4 / 5 := by
  intro h
  sorry

end NUMINAMATH_GPT_fraction_of_25_exact_value_l38_3835


namespace NUMINAMATH_GPT_sufficient_condition_implies_range_l38_3840

theorem sufficient_condition_implies_range {x m : ℝ} : (∀ x, 1 ≤ x ∧ x < 4 → x < m) → 4 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_implies_range_l38_3840


namespace NUMINAMATH_GPT_prime_divides_expression_l38_3857

theorem prime_divides_expression (p : ℕ) (hp : p > 5 ∧ Prime p) : 
  ∃ n : ℕ, p ∣ (20^n + 15^n - 12^n) := 
  by
  use (p - 3)
  sorry

end NUMINAMATH_GPT_prime_divides_expression_l38_3857


namespace NUMINAMATH_GPT_geometric_sequence_sum_terms_l38_3850

noncomputable def geom_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_terms
  (a : ℕ → ℝ) (q : ℝ)
  (h_q_nonzero : q ≠ 1)
  (S3_eq : geom_sequence_sum a q 3 = 8)
  (S6_eq : geom_sequence_sum a q 6 = 7)
  : a 6 * q ^ 6 + a 7 * q ^ 7 + a 8 * q ^ 8 = 1 / 8 := sorry

end NUMINAMATH_GPT_geometric_sequence_sum_terms_l38_3850


namespace NUMINAMATH_GPT_amy_books_l38_3867

theorem amy_books (maddie_books : ℕ) (luisa_books : ℕ) (amy_luisa_more_than_maddie : ℕ) (h1 : maddie_books = 15) (h2 : luisa_books = 18) (h3 : amy_luisa_more_than_maddie = maddie_books + 9) : ∃ (amy_books : ℕ), amy_books = amy_luisa_more_than_maddie - luisa_books ∧ amy_books = 6 :=
by
  have total_books := 24
  sorry

end NUMINAMATH_GPT_amy_books_l38_3867


namespace NUMINAMATH_GPT_digit_sum_of_product_l38_3851

def digits_after_multiplication (a b : ℕ) : ℕ :=
  let product := a * b
  let units_digit := product % 10
  let tens_digit := (product / 10) % 10
  tens_digit + units_digit

theorem digit_sum_of_product :
  digits_after_multiplication 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909 = 9 :=
by 
  -- proof goes here
sorry

end NUMINAMATH_GPT_digit_sum_of_product_l38_3851


namespace NUMINAMATH_GPT_additional_distance_if_faster_speed_l38_3839

-- Conditions
def speed_slow := 10 -- km/hr
def speed_fast := 15 -- km/hr
def actual_distance := 30 -- km

-- Question and answer
theorem additional_distance_if_faster_speed : (speed_fast * (actual_distance / speed_slow) - actual_distance) = 15 := by
  sorry

end NUMINAMATH_GPT_additional_distance_if_faster_speed_l38_3839


namespace NUMINAMATH_GPT_Jane_possible_numbers_l38_3804

def is_factor (a b : ℕ) : Prop := b % a = 0
def in_range (n : ℕ) : Prop := 500 ≤ n ∧ n ≤ 4000

def Jane_number (m : ℕ) : Prop :=
  is_factor 180 m ∧
  is_factor 42 m ∧
  in_range m

theorem Jane_possible_numbers :
  Jane_number 1260 ∧ Jane_number 2520 ∧ Jane_number 3780 :=
by
  sorry

end NUMINAMATH_GPT_Jane_possible_numbers_l38_3804


namespace NUMINAMATH_GPT_find_y_l38_3872

-- Definitions of angles and the given problem.
def angle_ABC : ℝ := 90
def angle_ABD (y : ℝ) : ℝ := 3 * y
def angle_DBC (y : ℝ) : ℝ := 2 * y

-- The theorem stating the problem
theorem find_y (y : ℝ) (h1 : angle_ABC = 90) (h2 : angle_ABD y + angle_DBC y = angle_ABC) : y = 18 :=
  by 
  sorry

end NUMINAMATH_GPT_find_y_l38_3872


namespace NUMINAMATH_GPT_coefficient_x3_in_expansion_l38_3882

theorem coefficient_x3_in_expansion : 
  (∃ (r : ℕ), 5 - r / 2 = 3 ∧ 2 * Nat.choose 5 r = 10) :=
by 
  sorry

end NUMINAMATH_GPT_coefficient_x3_in_expansion_l38_3882


namespace NUMINAMATH_GPT_tom_sells_games_for_225_42_usd_l38_3873

theorem tom_sells_games_for_225_42_usd :
  let initial_usd := 200
  let usd_to_eur := 0.85
  let tripled_usd := initial_usd * 3
  let eur_value := tripled_usd * usd_to_eur
  let eur_to_jpy := 130
  let jpy_value := eur_value * eur_to_jpy
  let percent_sold := 0.40
  let sold_jpy_value := jpy_value * percent_sold
  let jpy_to_usd := 0.0085
  let sold_usd_value := sold_jpy_value * jpy_to_usd
  sold_usd_value = 225.42 :=
by
  sorry

end NUMINAMATH_GPT_tom_sells_games_for_225_42_usd_l38_3873


namespace NUMINAMATH_GPT_find_bicycle_speed_l38_3826

-- Let's define the conditions first
def distance := 10  -- Distance in km
def time_diff := 1 / 3  -- Time difference in hours
def speed_of_bicycle (x : ℝ) := x
def speed_of_car (x : ℝ) := 2 * x

-- Prove the equation using the given conditions
theorem find_bicycle_speed (x : ℝ) (h : x ≠ 0) :
  (distance / speed_of_bicycle x) = (distance / speed_of_car x) + time_diff :=
by {
  sorry
}

end NUMINAMATH_GPT_find_bicycle_speed_l38_3826


namespace NUMINAMATH_GPT_tom_total_spent_on_video_games_l38_3829

-- Conditions
def batman_game_cost : ℝ := 13.6
def superman_game_cost : ℝ := 5.06

-- Statement to be proved
theorem tom_total_spent_on_video_games : batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end NUMINAMATH_GPT_tom_total_spent_on_video_games_l38_3829


namespace NUMINAMATH_GPT_average_temperature_Robertson_l38_3859

def temperatures : List ℝ := [18, 21, 19, 22, 20]

noncomputable def average (temps : List ℝ) : ℝ :=
  (temps.sum) / (temps.length)

theorem average_temperature_Robertson :
  average temperatures = 20.0 :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_Robertson_l38_3859


namespace NUMINAMATH_GPT_bryan_total_books_and_magazines_l38_3855

-- Define the conditions
def books_per_shelf : ℕ := 23
def magazines_per_shelf : ℕ := 61
def bookshelves : ℕ := 29

-- Define the total books and magazines
def total_books : ℕ := books_per_shelf * bookshelves
def total_magazines : ℕ := magazines_per_shelf * bookshelves
def total_books_and_magazines : ℕ := total_books + total_magazines

-- The proof problem statement
theorem bryan_total_books_and_magazines : total_books_and_magazines = 2436 := 
by
  sorry

end NUMINAMATH_GPT_bryan_total_books_and_magazines_l38_3855


namespace NUMINAMATH_GPT_kyle_paper_delivery_l38_3841

-- Define the number of houses Kyle delivers to from Monday through Saturday
def housesDaily : ℕ := 100

-- Define the number of days from Monday to Saturday
def daysWeek : ℕ := 6

-- Define the adjustment for Sunday (10 fewer, 30 additional)
def sundayAdjust : ℕ := 30 - 10 + 100

-- Define the total number of papers delivered from Monday to Saturday
def papersMonToSat : ℕ := daysWeek * housesDaily

-- Define the total number of papers delivered on Sunday
def papersSunday : ℕ := sundayAdjust

-- Define the total number of papers delivered each week
def totalPapers : ℕ := papersMonToSat + papersSunday

-- The theorem we want to prove
theorem kyle_paper_delivery : totalPapers = 720 := by
  -- We are focusing only on the statement here.
  sorry

end NUMINAMATH_GPT_kyle_paper_delivery_l38_3841


namespace NUMINAMATH_GPT_angle_A_measure_l38_3833

variable {a b c A : ℝ}

def vector_m (b c a : ℝ) : ℝ × ℝ := (b, c - a)
def vector_n (b c a : ℝ) : ℝ × ℝ := (b - c, c + a)

theorem angle_A_measure (h_perpendicular : (vector_m b c a).1 * (vector_n b c a).1 + (vector_m b c a).2 * (vector_n b c a).2 = 0) :
  A = 2 * π / 3 := sorry

end NUMINAMATH_GPT_angle_A_measure_l38_3833


namespace NUMINAMATH_GPT_sequence_fifth_term_l38_3808

theorem sequence_fifth_term (a b c : ℕ) :
  (a = (2 + b) / 3) →
  (b = (a + 34) / 3) →
  (34 = (b + c) / 3) →
  c = 89 :=
by
  intros ha hb hc
  sorry

end NUMINAMATH_GPT_sequence_fifth_term_l38_3808


namespace NUMINAMATH_GPT_age_problem_solution_l38_3853

namespace AgeProblem

variables (S M : ℕ) (k : ℕ)

-- Condition: The present age of the son is 22
def son_age (S : ℕ) := S = 22

-- Condition: The man is 24 years older than his son
def man_age (M S : ℕ) := M = S + 24

-- Condition: In two years, man's age will be a certain multiple of son's age
def age_multiple (M S k : ℕ) := M + 2 = k * (S + 2)

-- Question: The ratio of man's age to son's age in two years
def age_ratio (M S : ℕ) := (M + 2) / (S + 2)

theorem age_problem_solution (S M : ℕ) (k : ℕ) 
  (h1 : son_age S)
  (h2 : man_age M S)
  (h3 : age_multiple M S k)
  : age_ratio M S = 2 :=
by
  rw [son_age, man_age, age_multiple, age_ratio] at *
  sorry

end AgeProblem

end NUMINAMATH_GPT_age_problem_solution_l38_3853


namespace NUMINAMATH_GPT_inequality_proof_l38_3876

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b+c-a)^2 / ((b+c)^2+a^2) + (c+a-b)^2 / ((c+a)^2+b^2) + (a+b-c)^2 / ((a+b)^2+c^2) ≥ 3 / 5 :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l38_3876


namespace NUMINAMATH_GPT_first_trial_addition_amounts_l38_3843

-- Define the range and conditions for the biological agent addition amount.
def lower_bound : ℝ := 20
def upper_bound : ℝ := 30
def golden_ratio_method : ℝ := 0.618
def first_trial_addition_amount_1 : ℝ := lower_bound + (upper_bound - lower_bound) * golden_ratio_method
def first_trial_addition_amount_2 : ℝ := upper_bound - (upper_bound - lower_bound) * golden_ratio_method

-- Prove that the possible addition amounts for the first trial are 26.18g or 23.82g.
theorem first_trial_addition_amounts :
  (first_trial_addition_amount_1 = 26.18 ∨ first_trial_addition_amount_2 = 23.82) :=
by
  -- Placeholder for the proof.
  sorry

end NUMINAMATH_GPT_first_trial_addition_amounts_l38_3843


namespace NUMINAMATH_GPT_find_pos_real_nums_l38_3845

theorem find_pos_real_nums (x y z a b c : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z):
  (x + y + z = a + b + c) ∧ (4 * x * y * z = a^2 * x + b^2 * y + c^2 * z + a * b * c) →
  (a = y + z - x ∧ b = z + x - y ∧ c = x + y - z) :=
by
  sorry

end NUMINAMATH_GPT_find_pos_real_nums_l38_3845


namespace NUMINAMATH_GPT_find_a_for_square_binomial_l38_3820

theorem find_a_for_square_binomial (a : ℚ) (h: ∃ (b : ℚ), ∀ (x : ℚ), 9 * x^2 + 21 * x + a = (3 * x + b)^2) : a = 49 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_for_square_binomial_l38_3820


namespace NUMINAMATH_GPT_Michelle_silver_beads_count_l38_3828

theorem Michelle_silver_beads_count 
  (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)
  (h1 : total_beads = 40)
  (h2 : blue_beads = 5)
  (h3 : red_beads = 2 * blue_beads)
  (h4 : white_beads = blue_beads + red_beads)
  (h5 : silver_beads = total_beads - (blue_beads + red_beads + white_beads)) :
  silver_beads = 10 :=
by
  sorry

end NUMINAMATH_GPT_Michelle_silver_beads_count_l38_3828


namespace NUMINAMATH_GPT_value_of_p_l38_3830

theorem value_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (x1 x2 : ℕ), x1 > 0 ∧ x2 > 0 ∧ 67 * x1^2 + p * x1 + q = 0 ∧ 67 * x2^2 + p * x2 + q = 0) : p = -2278 :=
by
  sorry

end NUMINAMATH_GPT_value_of_p_l38_3830


namespace NUMINAMATH_GPT_total_books_to_read_l38_3884

theorem total_books_to_read (books_per_week : ℕ) (weeks : ℕ) (total_books : ℕ) 
  (h1 : books_per_week = 6) 
  (h2 : weeks = 5) 
  (h3 : total_books = books_per_week * weeks) : 
  total_books = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_total_books_to_read_l38_3884


namespace NUMINAMATH_GPT_cost_of_7_cubic_yards_of_topsoil_is_1512_l38_3890

-- Definition of the given conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yards : ℕ := 7
def cubic_yards_to_cubic_feet : ℕ := 27

-- Problem definition
def cost_of_topsoil (cubic_yards : ℕ) (cost_per_cubic_foot : ℕ) (cubic_yards_to_cubic_feet : ℕ) : ℕ :=
  cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot

-- The proof statement
theorem cost_of_7_cubic_yards_of_topsoil_is_1512 :
  cost_of_topsoil cubic_yards cost_per_cubic_foot cubic_yards_to_cubic_feet = 1512 := by
  sorry

end NUMINAMATH_GPT_cost_of_7_cubic_yards_of_topsoil_is_1512_l38_3890


namespace NUMINAMATH_GPT_car_and_cyclist_speeds_and_meeting_point_l38_3802

/-- 
(1) Distance between points $A$ and $B$ is $80 \mathrm{~km}$.
(2) After one hour, the distance between them reduces to $24 \mathrm{~km}$.
(3) The cyclist takes a 1-hour rest but they meet $90$ minutes after their departure.
-/
def initial_distance : ℝ := 80 -- km
def distance_after_one_hour : ℝ := 24 -- km apart after 1 hour
def cyclist_rest_duration : ℝ := 1 -- hour
def meeting_time : ℝ := 1.5 -- hours (90 minutes after departure)

def car_speed : ℝ := 40 -- km/hr
def cyclist_speed : ℝ := 16 -- km/hr

theorem car_and_cyclist_speeds_and_meeting_point :
  initial_distance = 80 → 
  distance_after_one_hour = 24 → 
  cyclist_rest_duration = 1 → 
  meeting_time = 1.5 → 
  car_speed = 40 ∧ cyclist_speed = 16 ∧ meeting_point_from_A = 60 ∧ meeting_point_from_B = 20 :=
by
  sorry

end NUMINAMATH_GPT_car_and_cyclist_speeds_and_meeting_point_l38_3802


namespace NUMINAMATH_GPT_inequality_problem_l38_3832

noncomputable def nonneg_real := {x : ℝ // 0 ≤ x}

theorem inequality_problem (x y z : nonneg_real) (h : x.val * y.val + y.val * z.val + z.val * x.val = 1) :
  1 / (x.val + y.val) + 1 / (y.val + z.val) + 1 / (z.val + x.val) ≥ 5 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_problem_l38_3832


namespace NUMINAMATH_GPT_baking_dish_to_recipe_book_ratio_is_2_l38_3812

-- Definitions of costs
def cost_recipe_book : ℕ := 6
def cost_ingredient : ℕ := 3
def num_ingredients : ℕ := 5
def cost_apron : ℕ := cost_recipe_book + 1
def total_spent : ℕ := 40

-- Definition to calculate the total cost excluding the baking dish
def cost_excluding_baking_dish : ℕ :=
  cost_recipe_book + cost_apron + cost_ingredient * num_ingredients

-- Definition of cost of baking dish
def cost_baking_dish : ℕ := total_spent - cost_excluding_baking_dish

-- Definition of the ratio
def ratio_baking_dish_to_recipe_book : ℕ := cost_baking_dish / cost_recipe_book

-- Theorem stating that the ratio is 2
theorem baking_dish_to_recipe_book_ratio_is_2 :
  ratio_baking_dish_to_recipe_book = 2 :=
sorry

end NUMINAMATH_GPT_baking_dish_to_recipe_book_ratio_is_2_l38_3812


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l38_3877

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 0 = 2) 
  (h2 : ∀ n, a (n+1) = a n + d)
  (h3 : a 9 = 20): 
  d = 2 := 
by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l38_3877


namespace NUMINAMATH_GPT_minutes_in_3_5_hours_l38_3874

theorem minutes_in_3_5_hours : 3.5 * 60 = 210 := 
by
  sorry

end NUMINAMATH_GPT_minutes_in_3_5_hours_l38_3874


namespace NUMINAMATH_GPT_ratio_of_coeffs_l38_3842

theorem ratio_of_coeffs
  (a b c d e : ℝ) 
  (h_poly : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) : 
  d / e = 25 / 12 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_coeffs_l38_3842


namespace NUMINAMATH_GPT_MrJones_pants_count_l38_3887

theorem MrJones_pants_count (P : ℕ) (h1 : 6 * P + P = 280) : P = 40 := by
  sorry

end NUMINAMATH_GPT_MrJones_pants_count_l38_3887


namespace NUMINAMATH_GPT_parabola_sum_l38_3811

variables (a b c x y : ℝ)

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_sum (h1 : ∀ x, quadratic a b c x = -(x - 3)^2 + 4)
    (h2 : quadratic a b c 1 = 0)
    (h3 : quadratic a b c 5 = 0) :
    a + b + c = 0 :=
by
  -- We assume quadratic(a, b, c, x) = a * x^2 + b * x + c
  -- We assume quadratic(a, b, c, 1) = 0 and quadratic(a, b, c, 5) = 0
  -- We need to prove a + b + c = 0
  sorry

end NUMINAMATH_GPT_parabola_sum_l38_3811


namespace NUMINAMATH_GPT_integer_solutions_of_inequality_l38_3825

theorem integer_solutions_of_inequality (x : ℤ) : 
  (-4 < 1 - 3 * (x: ℤ) ∧ 1 - 3 * (x: ℤ) ≤ 4) ↔ (x = -1 ∨ x = 0 ∨ x = 1) := 
by 
  sorry

end NUMINAMATH_GPT_integer_solutions_of_inequality_l38_3825


namespace NUMINAMATH_GPT_smallest_nat_mod_5_6_7_l38_3864

theorem smallest_nat_mod_5_6_7 (n : ℕ) :
  n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 → n = 209 :=
sorry

end NUMINAMATH_GPT_smallest_nat_mod_5_6_7_l38_3864


namespace NUMINAMATH_GPT_evaluate_expression_l38_3847

variable (a : ℝ)
variable (x : ℝ)

theorem evaluate_expression (h : x = a + 9) : x - a + 6 = 15 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l38_3847


namespace NUMINAMATH_GPT_intersection_points_l38_3803

theorem intersection_points : 
  (∃ x : ℝ, y = -2 * x + 4 ∧ y = 0 ∧ (x, y) = (2, 0)) ∧
  (∃ y : ℝ, y = -2 * 0 + 4 ∧ (0, y) = (0, 4)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l38_3803


namespace NUMINAMATH_GPT_book_pairs_count_l38_3805

theorem book_pairs_count :
  let mystery_count := 3
  let fantasy_count := 4
  let biography_count := 3
  mystery_count * fantasy_count + mystery_count * biography_count + fantasy_count * biography_count = 33 :=
by 
  sorry

end NUMINAMATH_GPT_book_pairs_count_l38_3805


namespace NUMINAMATH_GPT_range_of_x_l38_3880

theorem range_of_x (x : ℝ) : (abs (x + 1) + abs (x - 5) = 6) ↔ (-1 ≤ x ∧ x ≤ 5) :=
by sorry

end NUMINAMATH_GPT_range_of_x_l38_3880


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l38_3881

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom a2_a5 : a 2 + a 5 = 4
axiom a6_a9 : a 6 + a 9 = 20

theorem arithmetic_sequence_sum : a 4 + a 7 = 12 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l38_3881


namespace NUMINAMATH_GPT_calculate_fraction_pow_l38_3831

theorem calculate_fraction_pow :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
  sorry

end NUMINAMATH_GPT_calculate_fraction_pow_l38_3831


namespace NUMINAMATH_GPT_vanya_scores_not_100_l38_3879

-- Definitions for initial conditions
def score_r (M : ℕ) := M - 14
def score_p (M : ℕ) := M - 9
def score_m (M : ℕ) := M

-- Define the maximum score constraint
def max_score := 100

-- Main statement to be proved
theorem vanya_scores_not_100 (M : ℕ) 
  (hr : score_r M ≤ max_score) 
  (hp : score_p M ≤ max_score) 
  (hm : score_m M ≤ max_score) : 
  ¬(score_r M = max_score ∧ (score_p M = max_score ∨ score_m M = max_score)) ∧
  ¬(score_r M = max_score ∧ score_p M = max_score ∧ score_m M = max_score) :=
sorry

end NUMINAMATH_GPT_vanya_scores_not_100_l38_3879


namespace NUMINAMATH_GPT_infinite_triangles_with_conditions_l38_3844

theorem infinite_triangles_with_conditions :
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), 
  (A > 0) ∧ (B > 0) ∧ (C > 0) ∧ (B - A = 2) ∧ (C = 4) ∧ 
  (Δ > 0) := sorry

end NUMINAMATH_GPT_infinite_triangles_with_conditions_l38_3844


namespace NUMINAMATH_GPT_find_nat_numbers_l38_3814

theorem find_nat_numbers (a b : ℕ) (c : ℕ) (h : ∀ n : ℕ, a^n + b^n = c^(n+1)) : a = 2 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_nat_numbers_l38_3814


namespace NUMINAMATH_GPT_find_number_l38_3806

theorem find_number (x : ℝ) (h : x / 14.5 = 171) : x = 2479.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l38_3806


namespace NUMINAMATH_GPT_dot_product_range_l38_3894

theorem dot_product_range (a b : ℝ) (θ : ℝ) (h1 : a = 8) (h2 : b = 12)
  (h3 : 30 * (Real.pi / 180) ≤ θ ∧ θ ≤ 60 * (Real.pi / 180)) :
  48 * Real.sqrt 3 ≤ a * b * Real.cos θ ∧ a * b * Real.cos θ ≤ 48 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_range_l38_3894


namespace NUMINAMATH_GPT_jordan_rectangle_width_l38_3815

theorem jordan_rectangle_width
  (carol_length : ℕ) (carol_width : ℕ) (jordan_length : ℕ) (jordan_width : ℕ)
  (h_carol_dims : carol_length = 12) (h_carol_dims2 : carol_width = 15)
  (h_jordan_length : jordan_length = 6)
  (h_area_eq : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 30 := 
sorry

end NUMINAMATH_GPT_jordan_rectangle_width_l38_3815


namespace NUMINAMATH_GPT_remainder_of_x50_div_x_minus_1_cubed_l38_3868

theorem remainder_of_x50_div_x_minus_1_cubed :
  (x : ℝ) → (x ^ 50) % ((x - 1) ^ 3) = 1225 * x ^ 2 - 2400 * x + 1176 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_x50_div_x_minus_1_cubed_l38_3868


namespace NUMINAMATH_GPT_sequence_general_term_l38_3892

theorem sequence_general_term (n : ℕ) : 
  (2 * n - 1) / (2 ^ n) = a_n := 
sorry

end NUMINAMATH_GPT_sequence_general_term_l38_3892


namespace NUMINAMATH_GPT_min_value_of_x_plus_y_l38_3852

theorem min_value_of_x_plus_y (x y : ℝ) (h1: y ≠ 0) (h2: 1 / y = (x - 1) / 2) : x + y ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_min_value_of_x_plus_y_l38_3852


namespace NUMINAMATH_GPT_binomial_expansion_sum_l38_3823

theorem binomial_expansion_sum (a : ℝ) (a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h₁ : (a * x - 1)^5 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5)
  (h₂ : a₃ = 80) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 :=
sorry

end NUMINAMATH_GPT_binomial_expansion_sum_l38_3823


namespace NUMINAMATH_GPT_cos_angle_identity_l38_3896

theorem cos_angle_identity (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) :
  Real.cos (π - 2 * α) = - (5 / 9) := by
sorry

end NUMINAMATH_GPT_cos_angle_identity_l38_3896


namespace NUMINAMATH_GPT_total_students_in_class_l38_3807

theorem total_students_in_class 
  (b : ℕ)
  (boys_jelly_beans : ℕ := b * b)
  (girls_jelly_beans : ℕ := (b + 1) * (b + 1))
  (total_jelly_beans : ℕ := 432) 
  (condition : boys_jelly_beans + girls_jelly_beans = total_jelly_beans) :
  (b + b + 1 = 29) :=
sorry

end NUMINAMATH_GPT_total_students_in_class_l38_3807


namespace NUMINAMATH_GPT_sufficient_conditions_for_x_sq_lt_one_l38_3891

theorem sufficient_conditions_for_x_sq_lt_one
  (x : ℝ) :
  (0 < x ∧ x < 1) ∨ (-1 < x ∧ x < 0) ∨ (-1 < x ∧ x < 1) → x^2 < 1 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_conditions_for_x_sq_lt_one_l38_3891


namespace NUMINAMATH_GPT_range_alpha_minus_beta_l38_3858

theorem range_alpha_minus_beta (α β : ℝ) (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π / 2) :
  - (3 * π) / 2 ≤ α - β ∧ α - β ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_alpha_minus_beta_l38_3858


namespace NUMINAMATH_GPT_smallest_possible_difference_after_101_years_l38_3883

theorem smallest_possible_difference_after_101_years {D E : ℤ} 
  (init_dollar : D = 6) 
  (init_euro : E = 7)
  (transformations : ∀ D E : ℤ, 
    (D', E') = (D + E, 2 * D + 1) ∨ (D', E') = (D + E, 2 * D - 1) ∨ 
    (D', E') = (D + E, 2 * E + 1) ∨ (D', E') = (D + E, 2 * E - 1)) :
  ∃ n_diff : ℤ, 101 = 2 * n_diff ∧ n_diff = 2 :=
sorry

end NUMINAMATH_GPT_smallest_possible_difference_after_101_years_l38_3883


namespace NUMINAMATH_GPT_find_n_l38_3827

theorem find_n 
  (n : ℕ) 
  (h_lcm : Nat.lcm n 16 = 48) 
  (h_gcf : Nat.gcd n 16 = 18) : 
  n = 54 := 
sorry

end NUMINAMATH_GPT_find_n_l38_3827


namespace NUMINAMATH_GPT_general_term_of_sequence_l38_3813

-- Definition of arithmetic sequence with positive common difference
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℤ} {d : ℤ}
axiom positive_common_difference : d > 0
axiom cond1 : a 3 * a 4 = 117
axiom cond2 : a 2 + a 5 = 22

-- Target statement to prove
theorem general_term_of_sequence : is_arithmetic_sequence a d → a n = 4 * n - 3 :=
by sorry

end NUMINAMATH_GPT_general_term_of_sequence_l38_3813


namespace NUMINAMATH_GPT_scientific_notation_of_great_wall_l38_3885

theorem scientific_notation_of_great_wall : 
  ∀ n : ℕ, (6700010 : ℝ) = 6.7 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_great_wall_l38_3885


namespace NUMINAMATH_GPT_ratio_arithmetic_sequence_triangle_l38_3821

theorem ratio_arithmetic_sequence_triangle (a b c : ℝ) 
  (h_triangle : a^2 + b^2 = c^2)
  (h_arith_seq : ∃ d, b = a + d ∧ c = a + 2 * d) :
  a / b = 3 / 4 ∧ b / c = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_arithmetic_sequence_triangle_l38_3821


namespace NUMINAMATH_GPT_number_verification_l38_3849

def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ a : ℕ, n = a * (a + 1) * (a + 2) * (a + 3)

theorem number_verification (h1 : 1680 % 3 = 0) (h2 : ∃ a : ℕ, 1680 = a * (a + 1) * (a + 2) * (a + 3)) : 
  is_product_of_four_consecutive 1680 :=
by
  sorry

end NUMINAMATH_GPT_number_verification_l38_3849


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l38_3800

variable {x : ℝ}

theorem necessary_but_not_sufficient_condition 
    (h : -1 ≤ x ∧ x < 2) : 
    (-1 ≤ x ∧ x < 3) ∧ ¬(((-1 ≤ x ∧ x < 3) → (-1 ≤ x ∧ x < 2))) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l38_3800


namespace NUMINAMATH_GPT_neg_power_identity_l38_3819

variable (m : ℝ)

theorem neg_power_identity : (-m^2)^3 = -m^6 :=
sorry

end NUMINAMATH_GPT_neg_power_identity_l38_3819
