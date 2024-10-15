import Mathlib

namespace NUMINAMATH_GPT_pink_cookies_eq_fifty_l621_62195

-- Define the total number of cookies
def total_cookies : ℕ := 86

-- Define the number of red cookies
def red_cookies : ℕ := 36

-- The property we want to prove
theorem pink_cookies_eq_fifty : (total_cookies - red_cookies = 50) :=
by
  sorry

end NUMINAMATH_GPT_pink_cookies_eq_fifty_l621_62195


namespace NUMINAMATH_GPT_statues_created_first_year_l621_62116

-- Definition of the initial conditions and the variable representing the number of statues created in the first year.
variables (S : ℕ)

-- Condition 1: In the second year, statues are quadrupled.
def second_year_statues : ℕ := 4 * S

-- Condition 2: In the third year, 12 statues are added, and 3 statues are broken.
def third_year_statues : ℕ := second_year_statues S + 12 - 3

-- Condition 3: In the fourth year, twice as many new statues are added as had been broken the previous year (2 * 3).
def fourth_year_added_statues : ℕ := 2 * 3
def fourth_year_statues : ℕ := third_year_statues S + fourth_year_added_statues

-- Condition 4: Total number of statues at the end of four years is 31.
def total_statues : ℕ := fourth_year_statues S

theorem statues_created_first_year : total_statues S = 31 → S = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_statues_created_first_year_l621_62116


namespace NUMINAMATH_GPT_a_plus_b_l621_62138

open Complex

theorem a_plus_b (a b : ℝ) (h : (a - I) * I = -b + 2 * I) : a + b = 1 := by
  sorry

end NUMINAMATH_GPT_a_plus_b_l621_62138


namespace NUMINAMATH_GPT_sequence_starting_point_l621_62113

theorem sequence_starting_point
  (n : ℕ) 
  (k : ℕ) 
  (h₁ : n * 9 ≤ 100000)
  (h₂ : k = 11110)
  (h₃ : 9 * (n + k - 1) = 99999) : 
  9 * n = 88890 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_starting_point_l621_62113


namespace NUMINAMATH_GPT_common_root_cubic_polynomials_l621_62161

open Real

theorem common_root_cubic_polynomials (a b c : ℝ)
  (h1 : ∃ α : ℝ, α^3 - a * α^2 + b = 0 ∧ α^3 - b * α^2 + c = 0)
  (h2 : ∃ β : ℝ, β^3 - b * β^2 + c = 0 ∧ β^3 - c * β^2 + a = 0)
  (h3 : ∃ γ : ℝ, γ^3 - c * γ^2 + a = 0 ∧ γ^3 - a * γ^2 + b = 0)
  : a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_common_root_cubic_polynomials_l621_62161


namespace NUMINAMATH_GPT_kylie_total_apples_l621_62171

-- Define the conditions as given in the problem.
def first_hour_apples : ℕ := 66
def second_hour_apples : ℕ := 2 * first_hour_apples
def third_hour_apples : ℕ := first_hour_apples / 3

-- Define the mathematical proof problem.
theorem kylie_total_apples : 
  first_hour_apples + second_hour_apples + third_hour_apples = 220 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_kylie_total_apples_l621_62171


namespace NUMINAMATH_GPT_shaded_area_in_6x6_grid_l621_62177

def total_shaded_area (grid_size : ℕ) (triangle_squares : ℕ) (num_triangles : ℕ)
  (trapezoid_squares : ℕ) (num_trapezoids : ℕ) : ℕ :=
  (triangle_squares * num_triangles) + (trapezoid_squares * num_trapezoids)

theorem shaded_area_in_6x6_grid :
  total_shaded_area 6 2 2 3 4 = 16 :=
by
  -- Proof omitted for demonstration purposes
  sorry

end NUMINAMATH_GPT_shaded_area_in_6x6_grid_l621_62177


namespace NUMINAMATH_GPT_find_product_of_roots_l621_62191

noncomputable def equation (x : ℝ) : ℝ := (Real.sqrt 2023) * x^3 - 4047 * x^2 + 3

theorem find_product_of_roots (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) 
  (h3 : equation x1 = 0) (h4 : equation x2 = 0) (h5 : equation x3 = 0) :
  x2 * (x1 + x3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_product_of_roots_l621_62191


namespace NUMINAMATH_GPT_sum_of_z_values_l621_62157

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem sum_of_z_values (z1 z2 : ℝ) (hz1 : f (3 * z1) = 11) (hz2 : f (3 * z2) = 11) :
  z1 + z2 = - (2 / 9) :=
sorry

end NUMINAMATH_GPT_sum_of_z_values_l621_62157


namespace NUMINAMATH_GPT_total_cans_collected_l621_62165

theorem total_cans_collected 
  (bags_saturday : ℕ) 
  (bags_sunday : ℕ) 
  (cans_per_bag : ℕ) 
  (h1 : bags_saturday = 6) 
  (h2 : bags_sunday = 3) 
  (h3 : cans_per_bag = 8) : 
  bags_saturday + bags_sunday * cans_per_bag = 72 := 
by 
  simp [h1, h2, h3]; -- Simplify using the given conditions
  sorry -- Placeholder for the computation proof

end NUMINAMATH_GPT_total_cans_collected_l621_62165


namespace NUMINAMATH_GPT_fraction_meaningful_domain_l621_62180

theorem fraction_meaningful_domain (x : ℝ) : (∃ f : ℝ, f = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_domain_l621_62180


namespace NUMINAMATH_GPT_find_third_root_l621_62107

-- Define the polynomial
def poly (a b x : ℚ) : ℚ := a * x^3 + 2 * (a + b) * x^2 + (b - 2 * a) * x + (10 - a)

-- Define the roots condition
def is_root (a b x : ℚ) : Prop := poly a b x = 0

-- Given conditions and required proof
theorem find_third_root (a b : ℚ) (ha : a = 350 / 13) (hb : b = -1180 / 13) :
  is_root a b (-1) ∧ is_root a b 4 → 
  ∃ r : ℚ, is_root a b r ∧ r ≠ -1 ∧ r ≠ 4 ∧ r = 61 / 35 :=
by sorry

end NUMINAMATH_GPT_find_third_root_l621_62107


namespace NUMINAMATH_GPT_period_in_years_proof_l621_62105

-- Definitions
def marbles (P : ℕ) : ℕ := P

def remaining_marbles (M : ℕ) : ℕ := (M / 4)

def doubled_remaining_marbles (M : ℕ) : ℕ := 2 * (M / 4)

def age_in_five_years (current_age : ℕ) : ℕ := current_age + 5

-- Given Conditions
variables (P : ℕ) (current_age : ℕ) (H1 : marbles P = P) (H2 : current_age = 45)

-- Final Proof Goal
theorem period_in_years_proof (H3 : doubled_remaining_marbles P = age_in_five_years current_age) : P = 100 :=
sorry

end NUMINAMATH_GPT_period_in_years_proof_l621_62105


namespace NUMINAMATH_GPT_chameleons_to_blue_l621_62136

-- Define a function that simulates the biting between chameleons and their resulting color changes
def color_transition (color_biter : ℕ) (color_bitten : ℕ) : ℕ :=
  if color_bitten = 1 then color_biter + 1
  else if color_bitten = 2 then color_biter + 2
  else if color_bitten = 3 then color_biter + 3
  else if color_bitten = 4 then color_biter + 4
  else 5  -- Once it reaches color 5 (blue), it remains blue.

-- Define the main theorem statement that given 5 red chameleons, all can be turned to blue.
theorem chameleons_to_blue : ∀ (red_chameleons : ℕ), red_chameleons = 5 → 
  ∃ (sequence_of_bites : ℕ → (ℕ × ℕ)), (∀ (c : ℕ), c < 5 → color_transition c (sequence_of_bites c).fst = 5) :=
by sorry

end NUMINAMATH_GPT_chameleons_to_blue_l621_62136


namespace NUMINAMATH_GPT_arithmetic_seq_sum_mod_9_l621_62139

def sum_arithmetic_seq := 88230 + 88231 + 88232 + 88233 + 88234 + 88235 + 88236 + 88237 + 88238 + 88239 + 88240

theorem arithmetic_seq_sum_mod_9 : 
  sum_arithmetic_seq % 9 = 0 :=
by
-- proof will be provided here
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_mod_9_l621_62139


namespace NUMINAMATH_GPT_find_x_when_y_neg4_l621_62130

variable {x y : ℝ}
variable (k : ℝ)

-- Condition: x is inversely proportional to y
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop :=
  x * y = k

theorem find_x_when_y_neg4 (h : inversely_proportional 5 10 50) :
  inversely_proportional x (-4) 50 → x = -25 / 2 :=
by sorry

end NUMINAMATH_GPT_find_x_when_y_neg4_l621_62130


namespace NUMINAMATH_GPT_range_of_x_add_y_l621_62133

noncomputable def floor_not_exceeding (z : ℝ) : ℤ := ⌊z⌋

theorem range_of_x_add_y (x y : ℝ) (h1 : y = 3 * floor_not_exceeding x + 4) 
    (h2 : y = 4 * floor_not_exceeding (x - 3) + 7) (h3 : ¬ ∃ n : ℤ, x = n) : 
    40 < x + y ∧ x + y < 41 :=
by 
  sorry 

end NUMINAMATH_GPT_range_of_x_add_y_l621_62133


namespace NUMINAMATH_GPT_eval_sin_570_l621_62190

theorem eval_sin_570:
  2 * Real.sin (570 * Real.pi / 180) = -1 := 
by sorry

end NUMINAMATH_GPT_eval_sin_570_l621_62190


namespace NUMINAMATH_GPT_asymptote_of_hyperbola_l621_62118

theorem asymptote_of_hyperbola (x y : ℝ) :
  (x^2 - (y^2 / 4) = 1) → (y = 2 * x ∨ y = -2 * x) := sorry

end NUMINAMATH_GPT_asymptote_of_hyperbola_l621_62118


namespace NUMINAMATH_GPT_shares_difference_l621_62182

-- conditions: the ratio is 3:7:12, and the difference between q and r's share is Rs. 3000
theorem shares_difference (x : ℕ) (h : 12 * x - 7 * x = 3000) : 7 * x - 3 * x = 2400 :=
by
  -- simply skip the proof since it's not required in the prompt
  sorry

end NUMINAMATH_GPT_shares_difference_l621_62182


namespace NUMINAMATH_GPT_points_6_units_away_from_neg1_l621_62163

theorem points_6_units_away_from_neg1 (A : ℝ) (h : A = -1) :
  { x : ℝ | abs (x - A) = 6 } = { -7, 5 } :=
by
  sorry

end NUMINAMATH_GPT_points_6_units_away_from_neg1_l621_62163


namespace NUMINAMATH_GPT_unique_wxyz_solution_l621_62124

theorem unique_wxyz_solution (w x y z : ℕ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : w.factorial = x.factorial + y.factorial + z.factorial) : (w, x, y, z) = (3, 2, 2, 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_wxyz_solution_l621_62124


namespace NUMINAMATH_GPT_fg_of_2_eq_225_l621_62111

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem fg_of_2_eq_225 : f (g 2) = 225 := by
  sorry

end NUMINAMATH_GPT_fg_of_2_eq_225_l621_62111


namespace NUMINAMATH_GPT_largest_real_number_mu_l621_62186

noncomputable def largest_mu : ℝ := 13 / 2

theorem largest_real_number_mu (
  a b c d : ℝ
) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) :
  (a^2 + b^2 + c^2 + d^2) ≥ (largest_mu * a * b + b * c + 2 * c * d) :=
sorry

end NUMINAMATH_GPT_largest_real_number_mu_l621_62186


namespace NUMINAMATH_GPT_common_tangent_curves_l621_62137

theorem common_tangent_curves (s t a : ℝ) (e : ℝ) (he : e > 0) :
  (t = (1 / (2 * e)) * s^2) →
  (t = a * Real.log s) →
  (s / e = a / s) →
  a = 1 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_common_tangent_curves_l621_62137


namespace NUMINAMATH_GPT_cost_of_four_books_l621_62132

theorem cost_of_four_books
  (H : 2 * book_cost = 36) :
  4 * book_cost = 72 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_four_books_l621_62132


namespace NUMINAMATH_GPT_ted_age_solution_l621_62148

theorem ted_age_solution (t s : ℝ) (h1 : t = 3 * s - 10) (h2 : t + s = 60) : t = 42.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_ted_age_solution_l621_62148


namespace NUMINAMATH_GPT_geometric_sequence_value_l621_62167

theorem geometric_sequence_value (a : ℝ) (h_pos : 0 < a) 
    (h_geom1 : ∃ r, 25 * r = a)
    (h_geom2 : ∃ r, a * r = 7 / 9) : 
    a = 5 * Real.sqrt 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_value_l621_62167


namespace NUMINAMATH_GPT_area_of_square_l621_62109

theorem area_of_square (r s L B: ℕ) (h1 : r = s) (h2 : L = 5 * r) (h3 : B = 11) (h4 : 220 = L * B) : s^2 = 16 := by
  sorry

end NUMINAMATH_GPT_area_of_square_l621_62109


namespace NUMINAMATH_GPT_cos_product_equals_one_over_128_l621_62120

theorem cos_product_equals_one_over_128 :
  (Real.cos (Real.pi / 15)) *
  (Real.cos (2 * Real.pi / 15)) *
  (Real.cos (3 * Real.pi / 15)) *
  (Real.cos (4 * Real.pi / 15)) *
  (Real.cos (5 * Real.pi / 15)) *
  (Real.cos (6 * Real.pi / 15)) *
  (Real.cos (7 * Real.pi / 15))
  = 1 / 128 := 
sorry

end NUMINAMATH_GPT_cos_product_equals_one_over_128_l621_62120


namespace NUMINAMATH_GPT_tenth_term_is_correct_l621_62158

-- Define the conditions
def first_term : ℚ := 3
def last_term : ℚ := 88
def num_terms : ℕ := 30
def common_difference : ℚ := (last_term - first_term) / (num_terms - 1)

-- Define the function for the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℚ := first_term + (n - 1) * common_difference

-- Prove that the 10th term is 852/29
theorem tenth_term_is_correct : nth_term 10 = 852 / 29 := 
by 
  -- Add the proof later, the statement includes the setup and conditions
  sorry

end NUMINAMATH_GPT_tenth_term_is_correct_l621_62158


namespace NUMINAMATH_GPT_king_total_payment_l621_62145

/-- 
A king gets a crown made that costs $20,000. He tips the person 10%. Prove that the total amount the king paid after the tip is $22,000.
-/
theorem king_total_payment (C : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h1 : C = 20000) 
  (h2 : tip_percentage = 0.1) 
  (h3 : total_paid = C + C * tip_percentage) : 
  total_paid = 22000 := 
by 
  sorry

end NUMINAMATH_GPT_king_total_payment_l621_62145


namespace NUMINAMATH_GPT_dice_sum_prob_l621_62123

theorem dice_sum_prob :
  (3 / 6) * (3 / 6) * (2 / 5) * (1 / 6) * 2 = 13 / 216 :=
by sorry

end NUMINAMATH_GPT_dice_sum_prob_l621_62123


namespace NUMINAMATH_GPT_final_S_is_correct_l621_62135

/-- Define a function to compute the final value of S --/
def final_value_of_S : ℕ :=
  let S := 0
  let I_values := List.range' 1 27 3 -- generate list [1, 4, 7, ..., 28]
  I_values.foldl (fun S I => S + I) 0  -- compute the sum of the list

/-- Theorem stating the final value of S is 145 --/
theorem final_S_is_correct : final_value_of_S = 145 := by
  sorry

end NUMINAMATH_GPT_final_S_is_correct_l621_62135


namespace NUMINAMATH_GPT_average_speed_l621_62160

-- Define the speeds and times
def speed1 : ℝ := 120 -- km/h
def time1 : ℝ := 1 -- hour

def speed2 : ℝ := 150 -- km/h
def time2 : ℝ := 2 -- hours

def speed3 : ℝ := 80 -- km/h
def time3 : ℝ := 0.5 -- hour

-- Define the conversion factor
def km_to_miles : ℝ := 0.62

-- Calculate total distance (in kilometers)
def distance1 : ℝ := speed1 * time1
def distance2 : ℝ := speed2 * time2
def distance3 : ℝ := speed3 * time3

def total_distance_km : ℝ := distance1 + distance2 + distance3

-- Convert total distance to miles
def total_distance_miles : ℝ := total_distance_km * km_to_miles

-- Calculate total time (in hours)
def total_time : ℝ := time1 + time2 + time3

-- Final proof statement for average speed
theorem average_speed : total_distance_miles / total_time = 81.49 := by {
  sorry
}

end NUMINAMATH_GPT_average_speed_l621_62160


namespace NUMINAMATH_GPT_Thabo_books_l621_62155

theorem Thabo_books :
  ∃ (H : ℕ), ∃ (P : ℕ), ∃ (F : ℕ), 
  (H + P + F = 220) ∧ 
  (P = H + 20) ∧ 
  (F = 2 * P) ∧ 
  (H = 40) :=
by
  -- Here will be the formal proof, which is not required for this task.
  sorry

end NUMINAMATH_GPT_Thabo_books_l621_62155


namespace NUMINAMATH_GPT_simplify_expression_l621_62153

theorem simplify_expression : 1 + 3 / (2 + 5 / 6) = 35 / 17 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l621_62153


namespace NUMINAMATH_GPT_num_boys_l621_62168

-- Definitions as per the conditions
def boys (d : ℕ) := 2 * d
def reducedGirls (d : ℕ) := d - 1

-- Lean statement for the proof problem
theorem num_boys (d b : ℕ) 
  (h1 : b = boys d)
  (h2 : b = reducedGirls d + 8) : b = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_num_boys_l621_62168


namespace NUMINAMATH_GPT_value_of_a5_l621_62142

variable (a_n : ℕ → ℝ)
variable (a1 a9 a5 : ℝ)

-- Given conditions
axiom a1_plus_a9_eq_10 : a1 + a9 = 10
axiom arithmetic_sequence : ∀ n, a_n n = a1 + (n - 1) * (a_n 2 - a1)

-- Prove that a5 = 5
theorem value_of_a5 : a5 = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a5_l621_62142


namespace NUMINAMATH_GPT_slices_dinner_l621_62128

variable (lunch_slices : ℕ) (total_slices : ℕ)
variable (h1 : lunch_slices = 7) (h2 : total_slices = 12)

theorem slices_dinner : total_slices - lunch_slices = 5 :=
by sorry

end NUMINAMATH_GPT_slices_dinner_l621_62128


namespace NUMINAMATH_GPT_problem_statement_l621_62100

-- Define the odd function and the conditions given
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Main theorem statement
theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (3 - x))
  (h_f1 : f 1 = -2) :
  2012 * f 2012 - 2013 * f 2013 = -4026 := 
sorry

end NUMINAMATH_GPT_problem_statement_l621_62100


namespace NUMINAMATH_GPT_inverse_36_mod_53_l621_62199

theorem inverse_36_mod_53 (h : 17 * 26 ≡ 1 [MOD 53]) : 36 * 27 ≡ 1 [MOD 53] :=
sorry

end NUMINAMATH_GPT_inverse_36_mod_53_l621_62199


namespace NUMINAMATH_GPT_y_equals_px_div_5x_p_l621_62198

variable (p x y : ℝ)

theorem y_equals_px_div_5x_p (h : p = 5 * x * y / (x - y)) : y = p * x / (5 * x + p) :=
sorry

end NUMINAMATH_GPT_y_equals_px_div_5x_p_l621_62198


namespace NUMINAMATH_GPT_polynomial_real_root_l621_62114

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 + a * x^3 - x^2 + a^2 * x + 1 = 0) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_real_root_l621_62114


namespace NUMINAMATH_GPT_cases_in_1990_is_correct_l621_62166

-- Define the initial and final number of cases.
def initial_cases : ℕ := 600000
def final_cases : ℕ := 200

-- Define the years and time spans.
def year_1970 : ℕ := 1970
def year_1985 : ℕ := 1985
def year_2000 : ℕ := 2000

def span_1970_to_1985 : ℕ := year_1985 - year_1970 -- 15 years
def span_1985_to_2000 : ℕ := year_2000 - year_1985 -- 15 years

-- Define the rate of decrease from 1970 to 1985 as r cases per year.
-- Define the rate of decrease from 1985 to 2000 as (r / 2) cases per year.
def rate_of_decrease_1 (r : ℕ) := r
def rate_of_decrease_2 (r : ℕ) := r / 2

-- Define the intermediate number of cases in 1985.
def cases_in_1985 (r : ℕ) : ℕ := initial_cases - (span_1970_to_1985 * rate_of_decrease_1 r)

-- Define the number of cases in 1990.
def cases_in_1990 (r : ℕ) : ℕ := cases_in_1985 r - (5 * rate_of_decrease_2 r) -- 5 years from 1985 to 1990

-- Total decrease in cases over 30 years.
def total_decrease : ℕ := initial_cases - final_cases

-- Formalize the proof that the number of cases in 1990 is 133,450.
theorem cases_in_1990_is_correct : 
  ∃ (r : ℕ), 15 * rate_of_decrease_1 r + 15 * rate_of_decrease_2 r = total_decrease ∧ cases_in_1990 r = 133450 := 
by {
  sorry
}

end NUMINAMATH_GPT_cases_in_1990_is_correct_l621_62166


namespace NUMINAMATH_GPT_weeks_saved_l621_62129

theorem weeks_saved (w : ℕ) :
  (10 * w / 2) - ((10 * w / 2) / 4) = 15 → 
  w = 4 := 
by
  sorry

end NUMINAMATH_GPT_weeks_saved_l621_62129


namespace NUMINAMATH_GPT_calculation_correct_l621_62103

theorem calculation_correct :
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 4^128 - 3^128 :=
by
  sorry

end NUMINAMATH_GPT_calculation_correct_l621_62103


namespace NUMINAMATH_GPT_binomial_product_l621_62104

open Nat

theorem binomial_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end NUMINAMATH_GPT_binomial_product_l621_62104


namespace NUMINAMATH_GPT_longer_side_of_rectangle_l621_62197

noncomputable def circle_radius : ℝ := 6
noncomputable def circle_area : ℝ := Real.pi * circle_radius^2
noncomputable def rectangle_area : ℝ := 3 * circle_area
noncomputable def shorter_side : ℝ := 2 * circle_radius

theorem longer_side_of_rectangle :
    ∃ (l : ℝ), l = rectangle_area / shorter_side ∧ l = 9 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_longer_side_of_rectangle_l621_62197


namespace NUMINAMATH_GPT_abs_diff_of_two_numbers_l621_62193

theorem abs_diff_of_two_numbers (x y : ℝ) (h_sum : x + y = 42) (h_prod : x * y = 437) : |x - y| = 4 :=
sorry

end NUMINAMATH_GPT_abs_diff_of_two_numbers_l621_62193


namespace NUMINAMATH_GPT_professors_women_tenured_or_both_l621_62101

variable (professors : ℝ) -- Total number of professors as percentage
variable (women tenured men_tenured tenured_women : ℝ) -- Given percentages

-- Conditions
variables (hw : women = 0.69 * professors) 
          (ht : tenured = 0.7 * professors)
          (hm_t : men_tenured = 0.52 * (1 - women) * professors)
          (htw : tenured_women = tenured - men_tenured)
          
-- The statement to prove
theorem professors_women_tenured_or_both :
  women + tenured - tenured_women = 0.8512 * professors :=
by
  sorry

end NUMINAMATH_GPT_professors_women_tenured_or_both_l621_62101


namespace NUMINAMATH_GPT_probability_male_female_ratio_l621_62146

theorem probability_male_female_ratio :
  let total_possibilities := Nat.choose 9 5
  let specific_scenarios := Nat.choose 5 2 * Nat.choose 4 3 + Nat.choose 5 3 * Nat.choose 4 2
  let probability := specific_scenarios / (total_possibilities : ℚ)
  probability = 50 / 63 :=
by 
  sorry

end NUMINAMATH_GPT_probability_male_female_ratio_l621_62146


namespace NUMINAMATH_GPT_quilt_shading_fraction_l621_62149

/-- 
Statement:
Given a quilt block made from nine unit squares, where two unit squares are divided diagonally into triangles, 
and one unit square is divided into four smaller equal squares with one of the smaller squares shaded, 
the fraction of the quilt that is shaded is \( \frac{5}{36} \).
-/
theorem quilt_shading_fraction : 
  let total_area := 9 
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2 
  shaded_area / total_area = 5 / 36 :=
by
  -- Definitions based on conditions
  let total_area := 9
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2
  -- The proof statement (fraction of shaded area)
  have h : shaded_area / total_area = 5 / 36 := sorry
  exact h

end NUMINAMATH_GPT_quilt_shading_fraction_l621_62149


namespace NUMINAMATH_GPT_quadratic_smallest_root_a_quadratic_smallest_root_b_l621_62175

-- For Part (a)
theorem quadratic_smallest_root_a (a : ℝ) 
  (h : a^2 - 9 * a - 10 = 0 ∧ ∀ x, x^2 - 9 * x - 10 = 0 → x ≥ a) : 
  a^4 - 909 * a = 910 :=
by sorry

-- For Part (b)
theorem quadratic_smallest_root_b (b : ℝ) 
  (h : b^2 - 9 * b + 10 = 0 ∧ ∀ x, x^2 - 9 * x + 10 = 0 → x ≥ b) : 
  b^4 - 549 * b = -710 :=
by sorry

end NUMINAMATH_GPT_quadratic_smallest_root_a_quadratic_smallest_root_b_l621_62175


namespace NUMINAMATH_GPT_find_2alpha_minus_beta_l621_62150

theorem find_2alpha_minus_beta (α β : ℝ) (tan_diff : Real.tan (α - β) = 1 / 2) 
  (cos_β : Real.cos β = -7 * Real.sqrt 2 / 10) (α_range : 0 < α ∧ α < Real.pi) 
  (β_range : 0 < β ∧ β < Real.pi) : 2 * α - β = -3 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_find_2alpha_minus_beta_l621_62150


namespace NUMINAMATH_GPT_village_current_population_l621_62102

theorem village_current_population (initial_population : ℕ) (ten_percent_die : ℕ)
  (twenty_percent_leave : ℕ) : 
  initial_population = 4399 →
  ten_percent_die = initial_population / 10 →
  twenty_percent_leave = (initial_population - ten_percent_die) / 5 →
  (initial_population - ten_percent_die) - twenty_percent_leave = 3167 :=
sorry

end NUMINAMATH_GPT_village_current_population_l621_62102


namespace NUMINAMATH_GPT_sampling_methods_match_l621_62162

inductive SamplingMethod
| simple_random
| stratified
| systematic

open SamplingMethod

def commonly_used_sampling_methods : List SamplingMethod := 
  [simple_random, stratified, systematic]

def option_C : List SamplingMethod := 
  [simple_random, stratified, systematic]

theorem sampling_methods_match : commonly_used_sampling_methods = option_C := by
  sorry

end NUMINAMATH_GPT_sampling_methods_match_l621_62162


namespace NUMINAMATH_GPT_maximum_distance_value_of_m_l621_62183

-- Define the line equation
def line_eq (m x y : ℝ) : Prop := y = m * x - m - 1

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the problem statement
theorem maximum_distance_value_of_m :
  ∃ (m : ℝ), (∀ x y : ℝ, circle_eq x y → ∃ P : ℝ × ℝ, line_eq m P.fst P.snd) →
  m = -0.5 :=
sorry

end NUMINAMATH_GPT_maximum_distance_value_of_m_l621_62183


namespace NUMINAMATH_GPT_find_original_number_l621_62189

theorem find_original_number (x : ℕ) 
    (h1 : (73 * x - 17) / 5 - (61 * x + 23) / 7 = 183) : x = 32 := 
by
  sorry

end NUMINAMATH_GPT_find_original_number_l621_62189


namespace NUMINAMATH_GPT_range_of_r_l621_62115

theorem range_of_r (r : ℝ) (h_r : r > 0) :
  let M := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}
  let N := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}
  (∀ p, p ∈ N → p ∈ M) → 0 < r ∧ r ≤ 2 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_r_l621_62115


namespace NUMINAMATH_GPT_subset_condition_l621_62110

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_condition (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_subset_condition_l621_62110


namespace NUMINAMATH_GPT_root_triple_condition_l621_62154

theorem root_triple_condition (a b c α β : ℝ)
  (h_eq : a * α^2 + b * α + c = 0)
  (h_β_eq : β = 3 * α)
  (h_vieta_sum : α + β = -b / a)
  (h_vieta_product : α * β = c / a) :
  3 * b^2 = 16 * a * c :=
by
  sorry

end NUMINAMATH_GPT_root_triple_condition_l621_62154


namespace NUMINAMATH_GPT_max_deflection_angle_l621_62117

variable (M m : ℝ)
variable (h : M > m)

theorem max_deflection_angle :
  ∃ α : ℝ, α = Real.arcsin (m / M) := by
  sorry

end NUMINAMATH_GPT_max_deflection_angle_l621_62117


namespace NUMINAMATH_GPT_roots_of_quadratic_range_k_l621_62108

theorem roots_of_quadratic_range_k :
  (∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ 
    x1 ≠ x2 ∧ 
    (x1 ≠ 1 ∧ x2 ≠ 1) ∧
    ∀ k : ℝ, x1 ^ 2 + (k - 3) * x1 + k ^ 2 = 0 ∧ x2 ^ 2 + (k - 3) * x2 + k ^ 2 = 0) ↔
  ((k : ℝ) < 1 ∧ k > -2) :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_range_k_l621_62108


namespace NUMINAMATH_GPT_simple_interest_years_l621_62122

theorem simple_interest_years (P : ℝ) (difference : ℝ) (N : ℝ) : 
  P = 2300 → difference = 69 → (23 * N = 69) → N = 3 :=
by
  intros hP hdifference heq
  sorry

end NUMINAMATH_GPT_simple_interest_years_l621_62122


namespace NUMINAMATH_GPT_sequence_m_value_l621_62126

theorem sequence_m_value (m : ℕ) (a : ℕ → ℝ) (h₀ : a 0 = 37) (h₁ : a 1 = 72)
  (hm : a m = 0) (h_rec : ∀ k, 1 ≤ k ∧ k < m → a (k + 1) = a (k - 1) - 3 / a k) : m = 889 :=
sorry

end NUMINAMATH_GPT_sequence_m_value_l621_62126


namespace NUMINAMATH_GPT_circle_positions_n_l621_62192

theorem circle_positions_n (n : ℕ) (h1 : n ≥ 23) (h2 : (23 - 7) * 2 + 2 = n) : n = 32 :=
sorry

end NUMINAMATH_GPT_circle_positions_n_l621_62192


namespace NUMINAMATH_GPT_find_a_b_tangent_line_at_zero_l621_62173

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3
noncomputable def f' (a b x : ℝ) : ℝ := 2 * a * x + b

theorem find_a_b :
  ∃ a b : ℝ, (a ≠ 0) ∧ (∀ x, f' a b x = 2 * x - 8) := 
sorry

noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.sin x + x^2 - 8 * x + 3
noncomputable def g' (x : ℝ) : ℝ := Real.exp x * Real.sin x + Real.exp x * Real.cos x + 2 * x - 8

theorem tangent_line_at_zero :
  g' 0 = -7 ∧ g 0 = 3 ∧ (∀ y, y = 3 + (-7) * x) := 
sorry

end NUMINAMATH_GPT_find_a_b_tangent_line_at_zero_l621_62173


namespace NUMINAMATH_GPT_mother_hen_heavier_l621_62176

-- Define the weights in kilograms
def weight_mother_hen : ℝ := 2.3
def weight_baby_chick : ℝ := 0.4

-- State the theorem with the final correct answer
theorem mother_hen_heavier :
  weight_mother_hen - weight_baby_chick = 1.9 :=
by
  sorry

end NUMINAMATH_GPT_mother_hen_heavier_l621_62176


namespace NUMINAMATH_GPT_proof_problem_l621_62134

def problem : Prop :=
  ∃ (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004

theorem proof_problem : 
  problem → 
  ∃! (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004 :=
sorry

end NUMINAMATH_GPT_proof_problem_l621_62134


namespace NUMINAMATH_GPT_find_m_l621_62181

theorem find_m (x : ℝ) (m : ℝ) (h1 : x > 2) (h2 : x - 3 * m + 1 > 0) : m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l621_62181


namespace NUMINAMATH_GPT_geom_progression_n_eq_6_l621_62178

theorem geom_progression_n_eq_6
  (a r : ℝ)
  (h_r : r = 6)
  (h_ratio : (a * (1 - r^n) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217) :
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_geom_progression_n_eq_6_l621_62178


namespace NUMINAMATH_GPT_triangle_side_length_l621_62121

theorem triangle_side_length {A B C : Type*} 
  (a b : ℝ) (S : ℝ) (ha : a = 4) (hb : b = 5) (hS : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l621_62121


namespace NUMINAMATH_GPT_sector_area_l621_62194

theorem sector_area
  (r : ℝ) (s : ℝ) (h_r : r = 1) (h_s : s = 1) : 
  (1 / 2) * r * s = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sector_area_l621_62194


namespace NUMINAMATH_GPT_ratio_of_cost_to_selling_price_l621_62184

-- Define the given conditions
def cost_price (CP : ℝ) := CP
def selling_price (CP : ℝ) : ℝ := CP + 0.25 * CP

-- Lean statement for the problem
theorem ratio_of_cost_to_selling_price (CP SP : ℝ) (h1 : SP = selling_price CP) : CP / SP = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_cost_to_selling_price_l621_62184


namespace NUMINAMATH_GPT_problem_DE_length_l621_62196

theorem problem_DE_length
  (AB AD : ℝ)
  (AB_eq : AB = 7)
  (AD_eq : AD = 10)
  (area_eq : 7 * CE = 140)
  (DC CE DE : ℝ)
  (DC_eq : DC = 7)
  (CE_eq : CE = 20)
  : DE = Real.sqrt 449 :=
by
  sorry

end NUMINAMATH_GPT_problem_DE_length_l621_62196


namespace NUMINAMATH_GPT_no_unique_p_l621_62159

-- Define the probabilities P_1 and P_2 given p
def P1 (p : ℝ) : ℝ := 3 * p^2 - 2 * p^3
def P2 (p : ℝ) : ℝ := 3 * p^2 - 3 * p^3

-- Define the expected value E(xi)
def E_xi (p : ℝ) : ℝ := P1 p + P2 p

-- Prove that there does not exist a unique p in (0, 1) such that E(xi) = 1.5
theorem no_unique_p (p : ℝ) (h : 0 < p ∧ p < 1) : E_xi p ≠ 1.5 := by
  sorry

end NUMINAMATH_GPT_no_unique_p_l621_62159


namespace NUMINAMATH_GPT_solution_l621_62119

noncomputable def problem : Prop := 
  - (Real.sin (133 * Real.pi / 180)) * (Real.cos (197 * Real.pi / 180)) -
  (Real.cos (47 * Real.pi / 180)) * (Real.cos (73 * Real.pi / 180)) = 1 / 2

theorem solution : problem :=
by
  sorry

end NUMINAMATH_GPT_solution_l621_62119


namespace NUMINAMATH_GPT_greatest_possible_value_l621_62112

theorem greatest_possible_value (x y : ℝ) (h1 : -4 ≤ x) (h2 : x ≤ -2) (h3 : 2 ≤ y) (h4 : y ≤ 4) : 
  ∃ z: ℝ, z = (x + y) / x ∧ (∀ z', z' = (x' + y') / x' ∧ -4 ≤ x' ∧ x' ≤ -2 ∧ 2 ≤ y' ∧ y' ≤ 4 → z' ≤ z) ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_value_l621_62112


namespace NUMINAMATH_GPT_contrapositive_l621_62144

theorem contrapositive (q p : Prop) (h : q → p) : ¬p → ¬q :=
by
  -- Proof will be filled in later.
  sorry

end NUMINAMATH_GPT_contrapositive_l621_62144


namespace NUMINAMATH_GPT_total_packages_of_gum_l621_62188

theorem total_packages_of_gum (R_total R_extra R_per_package A_total A_extra A_per_package : ℕ) 
  (hR1 : R_total = 41) (hR2 : R_extra = 6) (hR3 : R_per_package = 7)
  (hA1 : A_total = 23) (hA2 : A_extra = 3) (hA3 : A_per_package = 5) :
  (R_total - R_extra) / R_per_package + (A_total - A_extra) / A_per_package = 9 :=
by
  sorry

end NUMINAMATH_GPT_total_packages_of_gum_l621_62188


namespace NUMINAMATH_GPT_value_of_a_l621_62151

theorem value_of_a (a : ℝ) (h₁ : ∀ x : ℝ, (2 * x - (1/3) * a ≤ 0) → (x ≤ 2)) : a = 12 :=
sorry

end NUMINAMATH_GPT_value_of_a_l621_62151


namespace NUMINAMATH_GPT_max_minus_min_on_interval_l621_62106

def f (x a : ℝ) : ℝ := x^3 - 3 * x - a

theorem max_minus_min_on_interval (a : ℝ) :
  let M := max (f 0 a) (f 3 a)
  let N := f 1 a
  M - N = 20 :=
by
  sorry

end NUMINAMATH_GPT_max_minus_min_on_interval_l621_62106


namespace NUMINAMATH_GPT_john_swimming_improvement_l621_62143

theorem john_swimming_improvement :
  let initial_lap_time := 35 / 15 -- initial lap time in minutes per lap
  let current_lap_time := 33 / 18 -- current lap time in minutes per lap
  initial_lap_time - current_lap_time = 1 / 9 := 
by
  -- Definition of initial and current lap times are implied in Lean.
  sorry

end NUMINAMATH_GPT_john_swimming_improvement_l621_62143


namespace NUMINAMATH_GPT_woman_working_days_l621_62164

-- Define the conditions
def man_work_rate := 1 / 6
def boy_work_rate := 1 / 18
def combined_work_rate := 1 / 4

-- Question statement in Lean 4
theorem woman_working_days :
  ∃ W : ℚ, (man_work_rate + W + boy_work_rate = combined_work_rate) ∧ (1 / W = 1296) :=
sorry

end NUMINAMATH_GPT_woman_working_days_l621_62164


namespace NUMINAMATH_GPT_roots_quadratic_sum_squares_l621_62140

theorem roots_quadratic_sum_squares :
  (∃ a b : ℝ, (∀ x : ℝ, x^2 - 4 * x + 4 = 0 → (x = a ∨ x = b)) ∧ a^2 + b^2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_sum_squares_l621_62140


namespace NUMINAMATH_GPT_conditional_probability_l621_62127

theorem conditional_probability :
  let P_B : ℝ := 0.15
  let P_A : ℝ := 0.05
  let P_A_and_B : ℝ := 0.03
  let P_B_given_A := P_A_and_B / P_A
  P_B_given_A = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_conditional_probability_l621_62127


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l621_62141

theorem intersection_of_A_and_B :
  let A := {0, 1, 2, 3, 4}
  let B := {x | ∃ n ∈ A, x = 2 * n}
  A ∩ B = {0, 2, 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l621_62141


namespace NUMINAMATH_GPT_average_of_remaining_numbers_l621_62169

theorem average_of_remaining_numbers (S : ℕ) (h1 : S = 12 * 90) :
  ((S - 65 - 75 - 85) / 9) = 95 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_numbers_l621_62169


namespace NUMINAMATH_GPT_optimal_optimism_coefficient_l621_62152

theorem optimal_optimism_coefficient (a b : ℝ) (x : ℝ) (h_b_gt_a : b > a) (h_x : 0 < x ∧ x < 1) 
  (h_c : ∀ (c : ℝ), c = a + x * (b - a) → (c - a) * (c - a) = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_optimal_optimism_coefficient_l621_62152


namespace NUMINAMATH_GPT_number_of_five_dollar_bills_l621_62125

theorem number_of_five_dollar_bills (total_money denomination expected_bills : ℕ) 
  (h1 : total_money = 45) 
  (h2 : denomination = 5) 
  (h3 : expected_bills = total_money / denomination) : 
  expected_bills = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_five_dollar_bills_l621_62125


namespace NUMINAMATH_GPT_solve_for_x_l621_62185

theorem solve_for_x (x y : ℝ) (h : (x + 1) / (x - 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 5)) : 
  x = (y^2 + 3 * y - 1) / 7 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l621_62185


namespace NUMINAMATH_GPT_num_people_for_new_avg_l621_62170

def avg_salary := 430
def old_supervisor_salary := 870
def new_supervisor_salary := 870
def num_workers := 8
def total_people_before := num_workers + 1
def total_salary_before := total_people_before * avg_salary
def workers_salary := total_salary_before - old_supervisor_salary
def total_salary_after := workers_salary + new_supervisor_salary

theorem num_people_for_new_avg :
    ∃ (x : ℕ), x * avg_salary = total_salary_after ∧ x = 9 :=
by
  use 9
  field_simp
  sorry

end NUMINAMATH_GPT_num_people_for_new_avg_l621_62170


namespace NUMINAMATH_GPT_vector_dot_product_parallel_l621_62172

theorem vector_dot_product_parallel (m : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (m, -4))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (a.1 * b.1 + a.2 * b.2) = -10 := by
  sorry

end NUMINAMATH_GPT_vector_dot_product_parallel_l621_62172


namespace NUMINAMATH_GPT_geometric_sequence_product_l621_62187

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_product (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
  (h : a 3 = -1) : a 1 * a 2 * a 3 * a 4 * a 5 = -1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l621_62187


namespace NUMINAMATH_GPT_no_infinite_arithmetic_progression_divisible_l621_62147

-- Definitions based on the given condition
def is_arithmetic_progression (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

def product_divisible_by_sum (a : ℕ → ℕ) (n : ℕ) : Prop :=
(a n * a (n+1) * a (n+2) * a (n+3) * a (n+4) * a (n+5) * a (n+6) * a (n+7) * a (n+8) * a (n+9)) %
(a n + a (n+1) + a (n+2) + a (n+3) + a (n+4) + a (n+5) + a (n+6) + a (n+7) + a (n+8) + a (n+9)) = 0

-- Final statement to be proven
theorem no_infinite_arithmetic_progression_divisible :
  ¬ ∃ (a : ℕ → ℕ), is_arithmetic_progression a ∧ ∀ n : ℕ, product_divisible_by_sum a n := 
sorry

end NUMINAMATH_GPT_no_infinite_arithmetic_progression_divisible_l621_62147


namespace NUMINAMATH_GPT_trig_expression_simplify_l621_62131

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end NUMINAMATH_GPT_trig_expression_simplify_l621_62131


namespace NUMINAMATH_GPT_mia_spent_total_l621_62179

theorem mia_spent_total (sibling_cost parent_cost : ℕ) (num_siblings num_parents : ℕ)
    (h1 : sibling_cost = 30)
    (h2 : parent_cost = 30)
    (h3 : num_siblings = 3)
    (h4 : num_parents = 2) :
    sibling_cost * num_siblings + parent_cost * num_parents = 150 :=
by
  sorry

end NUMINAMATH_GPT_mia_spent_total_l621_62179


namespace NUMINAMATH_GPT_part1_69_part1_97_not_part2_difference_numbers_in_range_l621_62174

def is_difference_number (n : ℕ) : Prop :=
  (n % 7 = 6) ∧ (n % 5 = 4)

theorem part1_69 : is_difference_number 69 :=
sorry

theorem part1_97_not : ¬is_difference_number 97 :=
sorry

theorem part2_difference_numbers_in_range :
  {n : ℕ | is_difference_number n ∧ 500 < n ∧ n < 600} = {524, 559, 594} :=
sorry

end NUMINAMATH_GPT_part1_69_part1_97_not_part2_difference_numbers_in_range_l621_62174


namespace NUMINAMATH_GPT_cylinder_lateral_surface_area_l621_62156

theorem cylinder_lateral_surface_area 
  (diameter height : ℝ) 
  (h1 : diameter = 2) 
  (h2 : height = 2) : 
  2 * Real.pi * (diameter / 2) * height = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cylinder_lateral_surface_area_l621_62156
