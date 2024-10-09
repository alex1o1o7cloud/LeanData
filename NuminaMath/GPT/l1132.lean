import Mathlib

namespace sequence_uniquely_determined_l1132_113270

theorem sequence_uniquely_determined (a : ℕ → ℝ) (p q : ℝ) (a0 a1 : ℝ)
  (h : ∀ n, a (n + 2) = p * a (n + 1) + q * a n)
  (h0 : a 0 = a0)
  (h1 : a 1 = a1) :
  ∀ n, ∃! a_n, a n = a_n :=
sorry

end sequence_uniquely_determined_l1132_113270


namespace necessary_but_not_sufficient_l1132_113295

-- Define \(\frac{1}{x} < 2\) and \(x > \frac{1}{2}\)
def condition1 (x : ℝ) : Prop := 1 / x < 2
def condition2 (x : ℝ) : Prop := x > 1 / 2

-- Theorem stating that condition1 is necessary but not sufficient for condition2
theorem necessary_but_not_sufficient (x : ℝ) : condition1 x → condition2 x ↔ true :=
sorry

end necessary_but_not_sufficient_l1132_113295


namespace solve_quadratic_l1132_113224

theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
sorry

end solve_quadratic_l1132_113224


namespace ariel_years_fencing_l1132_113248

-- Definitions based on given conditions
def fencing_start_year := 2006
def birth_year := 1992
def current_age := 30

-- To find: The number of years Ariel has been fencing
def current_year : ℕ := birth_year + current_age
def years_fencing : ℕ := current_year - fencing_start_year

-- Proof statement
theorem ariel_years_fencing : years_fencing = 16 := by
  sorry

end ariel_years_fencing_l1132_113248


namespace only_solution_2_pow_eq_y_sq_plus_y_plus_1_l1132_113201

theorem only_solution_2_pow_eq_y_sq_plus_y_plus_1 {x y : ℕ} (h1 : 2^x = y^2 + y + 1) : x = 0 ∧ y = 0 := 
by {
  sorry -- proof goes here
}

end only_solution_2_pow_eq_y_sq_plus_y_plus_1_l1132_113201


namespace max_hardcover_books_l1132_113200

-- Define the conditions as provided in the problem
def total_books : ℕ := 36
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ a * b = n

-- The logical statement we need to prove
theorem max_hardcover_books :
  ∃ h : ℕ, (∃ c : ℕ, is_composite c ∧ 2 * h + c = total_books) ∧ 
  ∀ h' c', is_composite c' ∧ 2 * h' + c' = total_books → h' ≤ h :=
sorry

end max_hardcover_books_l1132_113200


namespace integer_solution_l1132_113275

theorem integer_solution (n m : ℤ) (h : (n + 2)^4 - n^4 = m^3) : (n = -1 ∧ m = 0) :=
by
  sorry

end integer_solution_l1132_113275


namespace find_water_bottles_l1132_113260

def water_bottles (W A : ℕ) :=
  A = W + 6 ∧ W + A = 54 → W = 24

theorem find_water_bottles (W A : ℕ) (h1 : A = W + 6) (h2 : W + A = 54) : W = 24 :=
by sorry

end find_water_bottles_l1132_113260


namespace largest_and_smallest_multiples_of_12_l1132_113251

theorem largest_and_smallest_multiples_of_12 (k : ℤ) (n₁ n₂ : ℤ) (h₁ : k = -150) (h₂ : n₁ = -156) (h₃ : n₂ = -144) :
  (∃ m1 : ℤ, m1 * 12 = n₁ ∧ n₁ < k) ∧ (¬ (∃ m2 : ℤ, m2 * 12 = n₂ ∧ n₂ > k ∧ ∃ m2' : ℤ, m2' * 12 > k ∧ m2' * 12 < n₂)) :=
by
  sorry

end largest_and_smallest_multiples_of_12_l1132_113251


namespace sum_of_arithmetic_sequence_l1132_113209

variable (S : ℕ → ℝ)

def arithmetic_seq_property (S : ℕ → ℝ) : Prop :=
  S 4 = 4 ∧ S 8 = 12

theorem sum_of_arithmetic_sequence (h : arithmetic_seq_property S) : S 12 = 24 :=
by
  sorry

end sum_of_arithmetic_sequence_l1132_113209


namespace expression_value_l1132_113205

theorem expression_value (a b c d m : ℚ) (h1 : a + b = 0) (h2 : a ≠ 0) (h3 : c * d = 1) (h4 : m = -5 ∨ m = 1) :
  |m| - (a / b) + ((a + b) / 2020) - (c * d) = 1 ∨ |m| - (a / b) + ((a + b) / 2020) - (c * d) = 5 :=
by sorry

end expression_value_l1132_113205


namespace find_percentage_l1132_113206

variable (P : ℝ)

def percentage_condition (P : ℝ) : Prop :=
  P * 30 = (0.25 * 16) + 2

theorem find_percentage : percentage_condition P → P = 0.2 :=
by
  intro h
  -- Proof steps go here
  sorry

end find_percentage_l1132_113206


namespace solve_k_values_l1132_113269

def has_positive_integer_solution (k : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = k * a * b * c

def infinitely_many_solutions (k : ℕ) : Prop :=
  ∃ (a b c : ℕ → ℕ), (∀ n, a n > 0 ∧ b n > 0 ∧ c n > 0 ∧ a n^2 + b n^2 + c n^2 = k * a n * b n * c n) ∧
  (∀ n, ∃ x y: ℤ, x^2 + y^2 = (a n * b n))

theorem solve_k_values :
  ∃ k : ℕ, (k = 1 ∨ k = 3) ∧ has_positive_integer_solution k ∧ infinitely_many_solutions k :=
sorry

end solve_k_values_l1132_113269


namespace find_x_l1132_113292

-- Definitions based on the problem conditions
def angle_CDE : ℝ := 90 -- angle CDE in degrees
def angle_ECB : ℝ := 68 -- angle ECB in degrees

-- Theorem statement
theorem find_x (x : ℝ) 
  (h1 : angle_CDE = 90) 
  (h2 : angle_ECB = 68) 
  (h3 : angle_CDE + x + angle_ECB = 180) : 
  x = 22 := 
by
  sorry

end find_x_l1132_113292


namespace sum_of_polynomial_roots_l1132_113257

theorem sum_of_polynomial_roots:
  ∀ (a b : ℝ),
  (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) →
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b^3 + b * a^3 = 683 := by
  intros a b h
  sorry

end sum_of_polynomial_roots_l1132_113257


namespace treaty_signed_on_friday_l1132_113291

def days_between (start_date : Nat) (end_date : Nat) : Nat := sorry

def day_of_week (start_day : Nat) (days_elapsed : Nat) : Nat :=
  (start_day + days_elapsed) % 7

def is_leap_year (year : Nat) : Bool :=
  if year % 4 = 0 then
    if year % 100 = 0 then
      if year % 400 = 0 then true else false
    else true
  else false

noncomputable def days_from_1802_to_1814 : Nat :=
  let leap_years := [1804, 1808, 1812]
  let normal_year_days := 365 * 9
  let leap_year_days := 366 * 3
  normal_year_days + leap_year_days

noncomputable def days_from_feb_5_to_apr_11_1814 : Nat :=
  24 + 31 + 11 -- days in February, March, and April 11

noncomputable def total_days_elapsed : Nat :=
  days_from_1802_to_1814 + days_from_feb_5_to_apr_11_1814

noncomputable def start_day : Nat := 5 -- Friday (0 = Sunday, ..., 5 = Friday, 6 = Saturday)

theorem treaty_signed_on_friday : day_of_week start_day total_days_elapsed = 5 := sorry

end treaty_signed_on_friday_l1132_113291


namespace min_speed_to_arrive_before_cara_l1132_113256

theorem min_speed_to_arrive_before_cara (d : ℕ) (sc : ℕ) (tc : ℕ) (sd : ℕ) (td : ℕ) (hd : ℕ) :
  d = 180 ∧ sc = 30 ∧ tc = d / sc ∧ hd = 1 ∧ td = tc - hd ∧ sd = d / td ∧ (36 < sd) :=
sorry

end min_speed_to_arrive_before_cara_l1132_113256


namespace cos_sq_minus_sin_sq_l1132_113245

noncomputable def alpha : ℝ := sorry

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem cos_sq_minus_sin_sq : Real.cos alpha ^ 2 - Real.sin alpha ^ 2 = -3/5 := by
  sorry

end cos_sq_minus_sin_sq_l1132_113245


namespace find_initial_balance_l1132_113250

-- Define the initial balance
variable (X : ℝ)

-- Conditions
def balance_tripled (X : ℝ) : ℝ := 3 * X
def balance_after_withdrawal (X : ℝ) : ℝ := balance_tripled X - 250

-- The problem statement to prove
theorem find_initial_balance (h : balance_after_withdrawal X = 950) : X = 400 :=
by
  sorry

end find_initial_balance_l1132_113250


namespace integer_squares_l1132_113266

theorem integer_squares (x y : ℤ) 
  (hx : ∃ a : ℤ, x + y = a^2)
  (h2x3y : ∃ b : ℤ, 2 * x + 3 * y = b^2)
  (h3xy : ∃ c : ℤ, 3 * x + y = c^2) : 
  x = 0 ∧ y = 0 := 
by { sorry }

end integer_squares_l1132_113266


namespace smaller_angle_at_8_15_pm_l1132_113262

noncomputable def smaller_angle_between_clock_hands (minute_hand_degrees_per_min: ℝ) (hour_hand_degrees_per_min: ℝ) (time_in_minutes: ℝ) : ℝ := sorry

theorem smaller_angle_at_8_15_pm :
  smaller_angle_between_clock_hands 6 0.5 495 = 157.5 :=
sorry

end smaller_angle_at_8_15_pm_l1132_113262


namespace solve_trigonometric_equation_l1132_113276

theorem solve_trigonometric_equation :
  ∃ (S : Finset ℝ), (∀ X ∈ S, 0 < X ∧ X < 360 ∧ 1 + 2 * Real.sin (X * Real.pi / 180) - 4 * (Real.sin (X * Real.pi / 180))^2 - 8 * (Real.sin (X * Real.pi / 180))^3 = 0) ∧ S.card = 4 :=
by
  sorry

end solve_trigonometric_equation_l1132_113276


namespace coefficient_x2_in_expansion_l1132_113213

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Statement to prove the coefficient of the x^2 term in (x + 1)^42 is 861
theorem coefficient_x2_in_expansion :
  (binomial 42 2) = 861 := by
  sorry

end coefficient_x2_in_expansion_l1132_113213


namespace simple_annual_interest_rate_l1132_113281

-- Given definitions and conditions
def monthly_interest_payment := 225
def principal_amount := 30000
def annual_interest_payment := monthly_interest_payment * 12
def annual_interest_rate := annual_interest_payment / principal_amount

-- Theorem statement
theorem simple_annual_interest_rate :
  annual_interest_rate * 100 = 9 := by
sorry

end simple_annual_interest_rate_l1132_113281


namespace real_ratio_sum_values_l1132_113218

variables (a b c d : ℝ)

theorem real_ratio_sum_values :
  (a / b + b / c + c / d + d / a = 6) ∧
  (a / c + b / d + c / a + d / b = 8) →
  (a / b + c / d = 2 ∨ a / b + c / d = 4) :=
by
  sorry

end real_ratio_sum_values_l1132_113218


namespace original_cookies_l1132_113232

noncomputable def initial_cookies (final_cookies : ℝ) (ratio : ℝ) (days : ℕ) : ℝ :=
  final_cookies / ratio^days

theorem original_cookies :
  ∀ (final_cookies : ℝ) (ratio : ℝ) (days : ℕ),
  final_cookies = 28 →
  ratio = 0.7 →
  days = 3 →
  initial_cookies final_cookies ratio days = 82 :=
by
  intros final_cookies ratio days h_final h_ratio h_days
  rw [initial_cookies, h_final, h_ratio, h_days]
  norm_num
  sorry

end original_cookies_l1132_113232


namespace quadratic_inequality_solution_l1132_113237

theorem quadratic_inequality_solution (x : ℝ) :
  (-3 * x^2 + 8 * x + 3 > 0) ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end quadratic_inequality_solution_l1132_113237


namespace weaving_sequence_l1132_113210

-- Define the arithmetic sequence conditions
def day1_weaving := 5
def total_cloth := 390
def days := 30

-- Mathematical statement to be proved
theorem weaving_sequence : 
    ∃ d : ℚ, 30 * day1_weaving + (days * (days - 1) / 2) * d = total_cloth ∧ d = 16 / 29 :=
by 
  sorry

end weaving_sequence_l1132_113210


namespace marla_errand_time_l1132_113285

theorem marla_errand_time :
  let drive_time_one_way := 20
  let parent_teacher_night := 70
  let drive_time_total := drive_time_one_way * 2
  let total_time := drive_time_total + parent_teacher_night
  total_time = 110 := by
  let drive_time_one_way := 20
  let parent_teacher_night := 70
  let drive_time_total := drive_time_one_way * 2
  let total_time := drive_time_total + parent_teacher_night
  sorry

end marla_errand_time_l1132_113285


namespace total_birds_times_types_l1132_113290

-- Defining the number of adults and offspring for each type of bird.
def num_ducks1 : ℕ := 2
def num_ducklings1 : ℕ := 5
def num_ducks2 : ℕ := 6
def num_ducklings2 : ℕ := 3
def num_ducks3 : ℕ := 9
def num_ducklings3 : ℕ := 6

def num_geese : ℕ := 4
def num_goslings : ℕ := 7

def num_swans : ℕ := 3
def num_cygnets : ℕ := 4

-- Calculate total number of birds
def total_ducks := (num_ducks1 * num_ducklings1 + num_ducks1) + (num_ducks2 * num_ducklings2 + num_ducks2) +
                      (num_ducks3 * num_ducklings3 + num_ducks3)

def total_geese := num_geese * num_goslings + num_geese
def total_swans := num_swans * num_cygnets + num_swans

def total_birds := total_ducks + total_geese + total_swans

-- Calculate the number of different types of birds
def num_types_of_birds : ℕ := 3 -- ducks, geese, swans

-- The final Lean statement to be proven
theorem total_birds_times_types :
  total_birds * num_types_of_birds = 438 :=
  by sorry

end total_birds_times_types_l1132_113290


namespace income_percentage_increase_l1132_113272

theorem income_percentage_increase (b : ℝ) (a : ℝ) (h : a = b * 0.75) :
  (b - a) / a * 100 = 33.33 :=
by
  sorry

end income_percentage_increase_l1132_113272


namespace solve_for_x_l1132_113215

theorem solve_for_x (x : ℝ) (h : 9 / x^2 = x / 81) : x = 9 := 
  sorry

end solve_for_x_l1132_113215


namespace find_a_if_odd_l1132_113298

theorem find_a_if_odd :
  ∀ (a : ℝ), (∀ x : ℝ, (a * (-x)^3 + (a - 1) * (-x)^2 + (-x) = -(a * x^3 + (a - 1) * x^2 + x))) → 
  a = 1 :=
by
  sorry

end find_a_if_odd_l1132_113298


namespace ratio_of_sums_l1132_113211

theorem ratio_of_sums (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49) 
  (h2 : x^2 + y^2 + z^2 = 64) 
  (h3 : a * x + b * y + c * z = 56) : 
  (a + b + c) / (x + y + z) = 7/8 := 
by 
  sorry

end ratio_of_sums_l1132_113211


namespace square_paper_side_length_l1132_113282

theorem square_paper_side_length :
  ∀ (edge_length : ℝ) (num_pieces : ℕ) (side_length : ℝ),
  edge_length = 12 ∧ num_pieces = 54 ∧ 6 * (edge_length ^ 2) = num_pieces * (side_length ^ 2)
  → side_length = 4 :=
by
  intros edge_length num_pieces side_length h
  sorry

end square_paper_side_length_l1132_113282


namespace one_number_greater_than_one_l1132_113253

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_prod: a * b * c = 1)
  (h_sum: a + b + c > 1 / a + 1 / b + 1 / c) :
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
by
  sorry

end one_number_greater_than_one_l1132_113253


namespace investment_of_D_l1132_113287

/--
Given C and D started a business where C invested Rs. 1000 and D invested some amount.
They made a total profit of Rs. 500, and D's share of the profit is Rs. 100.
So, how much did D invest in the business?
-/
theorem investment_of_D 
  (C_invested : ℕ) (D_share : ℕ) (total_profit : ℕ) 
  (H1 : C_invested = 1000) 
  (H2 : D_share = 100) 
  (H3 : total_profit = 500) 
  : ∃ D : ℕ, D = 250 :=
by
  sorry

end investment_of_D_l1132_113287


namespace angle_CBD_is_10_degrees_l1132_113223

theorem angle_CBD_is_10_degrees (angle_ABC angle_ABD : ℝ) (h1 : angle_ABC = 40) (h2 : angle_ABD = 30) :
  angle_ABC - angle_ABD = 10 :=
by
  sorry

end angle_CBD_is_10_degrees_l1132_113223


namespace relationship_among_abc_l1132_113236

noncomputable def a : ℝ := 2 ^ (3 / 2)
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2
noncomputable def c : ℝ := 0.8 ^ 2

theorem relationship_among_abc : b < c ∧ c < a := 
by
  -- these are conditions directly derived from the problem
  let h1 : a = 2 ^ (3 / 2) := rfl
  let h2 : b = Real.log 0.3 / Real.log 2 := rfl
  let h3 : c = 0.8 ^ 2 := rfl
  sorry

end relationship_among_abc_l1132_113236


namespace no_solution_for_equation_l1132_113299

theorem no_solution_for_equation (x y z : ℤ) : x^3 + y^3 ≠ 9 * z + 5 := 
by
  sorry

end no_solution_for_equation_l1132_113299


namespace no_ordered_triples_exist_l1132_113212

theorem no_ordered_triples_exist :
  ¬ ∃ (x y z : ℤ), 
    (x^2 - 3 * x * y + 2 * y^2 - z^2 = 39) ∧
    (-x^2 + 6 * y * z + 2 * z^2 = 40) ∧
    (x^2 + x * y + 8 * z^2 = 96) :=
sorry

end no_ordered_triples_exist_l1132_113212


namespace fireworks_number_l1132_113229

variable (x : ℕ)
variable (fireworks_total : ℕ := 484)
variable (happy_new_year_fireworks : ℕ := 12 * 5)
variable (boxes_of_fireworks : ℕ := 50 * 8)
variable (year_fireworks : ℕ := 4 * x)

theorem fireworks_number :
    4 * x + happy_new_year_fireworks + boxes_of_fireworks = fireworks_total →
    x = 6 := 
by
  sorry

end fireworks_number_l1132_113229


namespace find_k_for_perpendicular_lines_l1132_113249

theorem find_k_for_perpendicular_lines (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (5 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 1 ∨ k = 4) :=
by
  sorry

end find_k_for_perpendicular_lines_l1132_113249


namespace quadratic_has_distinct_real_roots_l1132_113240

theorem quadratic_has_distinct_real_roots (a : ℝ) (h : a = -2) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 3 = 0 ∧ a * x2^2 + 2 * x2 + 3 = 0) :=
by
  sorry

end quadratic_has_distinct_real_roots_l1132_113240


namespace exterior_angle_polygon_l1132_113286

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end exterior_angle_polygon_l1132_113286


namespace minimize_abs_a_n_l1132_113214

noncomputable def a_n (n : ℕ) : ℝ :=
  14 - (3 / 4) * (n - 1)

theorem minimize_abs_a_n : ∃ n : ℕ, n = 20 ∧ ∀ m : ℕ, |a_n n| ≤ |a_n m| := by
  sorry

end minimize_abs_a_n_l1132_113214


namespace triangle_largest_angle_l1132_113263

theorem triangle_largest_angle (x : ℝ) (hx : x + 2 * x + 3 * x = 180) :
  3 * x = 90 :=
by
  sorry

end triangle_largest_angle_l1132_113263


namespace jane_bought_two_bagels_l1132_113207

variable (b m d k : ℕ)

def problem_conditions : Prop :=
  b + m + d = 6 ∧ 
  (60 * b + 45 * m + 30 * d) = 100 * k

theorem jane_bought_two_bagels (hb : problem_conditions b m d k) : b = 2 :=
  sorry

end jane_bought_two_bagels_l1132_113207


namespace area_of_rectangle_l1132_113219

theorem area_of_rectangle (AB AC : ℝ) (angle_ABC : ℝ) (h_AB : AB = 15) (h_AC : AC = 17) (h_angle_ABC : angle_ABC = 90) :
  ∃ BC : ℝ, (BC = 8) ∧ (AB * BC = 120) :=
by
  sorry

end area_of_rectangle_l1132_113219


namespace part_I_part_II_part_III_no_zeros_part_III_one_zero_part_III_two_zeros_l1132_113222

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a / x + Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 1 - a / (x^2) + 1 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f' x a - x

-- Problem (I)
theorem part_I (a : ℝ) : f' 1 a = 0 → a = 2 := sorry

-- Problem (II)
theorem part_II (a : ℝ) : (∀ x, 1 < x ∧ x < 2 → f' x a ≥ 0) → a ≤ 2 := sorry

-- Problem (III)
theorem part_III_no_zeros (a : ℝ) : a > 1 → ∀ x, g x a ≠ 0 := sorry
theorem part_III_one_zero (a : ℝ) : (a = 1 ∨ a ≤ 0) → ∃! x, g x a = 0 := sorry
theorem part_III_two_zeros (a : ℝ) : 0 < a ∧ a < 1 → ∃ x1 x2, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0 := sorry

end part_I_part_II_part_III_no_zeros_part_III_one_zero_part_III_two_zeros_l1132_113222


namespace nina_math_homework_l1132_113216

theorem nina_math_homework (x : ℕ) :
  let ruby_math := 6
  let ruby_read := 2
  let nina_math := x * ruby_math
  let nina_read := 8 * ruby_read
  let nina_total := nina_math + nina_read
  nina_total = 48 → x = 5 :=
by
  intros
  sorry

end nina_math_homework_l1132_113216


namespace solve_for_real_a_l1132_113243

theorem solve_for_real_a (a : ℝ) (i : ℂ) (h : i^2 = -1) (h1 : (a - i)^2 = 2 * i) : a = -1 :=
by sorry

end solve_for_real_a_l1132_113243


namespace trigonometric_identity_proof_l1132_113234

theorem trigonometric_identity_proof :
  3.438 * (Real.sin (84 * Real.pi / 180)) * (Real.sin (24 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * (Real.sin (12 * Real.pi / 180)) = 1 / 16 :=
  sorry

end trigonometric_identity_proof_l1132_113234


namespace cubes_squares_problem_l1132_113239

theorem cubes_squares_problem (h1 : 2^3 - 7^2 = 1) (h2 : 3^3 - 6^2 = 9) (h3 : 5^3 - 9^2 = 16) : 4^3 - 8^2 = 0 := 
by
  sorry

end cubes_squares_problem_l1132_113239


namespace savannah_wraps_4_with_third_roll_l1132_113226

variable (gifts total_rolls : ℕ)
variable (wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ)
variable (no_leftover : Prop)

def savannah_wrapping_presents (gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ) (no_leftover : Prop) : Prop :=
  gifts = 12 ∧
  total_rolls = 3 ∧
  wrap_with_roll1 = 3 ∧
  wrap_with_roll2 = 5 ∧
  remaining_wrap_with_roll3 = gifts - (wrap_with_roll1 + wrap_with_roll2) ∧
  no_leftover = (total_rolls = 3) ∧ (wrap_with_roll1 + wrap_with_roll2 + remaining_wrap_with_roll3 = gifts)

theorem savannah_wraps_4_with_third_roll
  (h : savannah_wrapping_presents gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 no_leftover) :
  remaining_wrap_with_roll3 = 4 :=
by
  sorry

end savannah_wraps_4_with_third_roll_l1132_113226


namespace taxi_speed_l1132_113289

theorem taxi_speed (v : ℝ) (hA : ∀ v : ℝ, 3 * v = 6 * (v - 30)) : v = 60 :=
by
  sorry

end taxi_speed_l1132_113289


namespace arithmetic_sequence_geometric_term_ratio_l1132_113297

theorem arithmetic_sequence_geometric_term_ratio (a : ℕ → ℤ) (d : ℤ) (h₀ : d ≠ 0)
  (h₁ : a 1 = a 1)
  (h₂ : a 3 = a 1 + 2 * d)
  (h₃ : a 4 = a 1 + 3 * d)
  (h_geom : (a 1 + 2 * d)^2 = a 1 * (a 1 + 3 * d)) :
  (a 1 + a 5 + a 17) / (a 2 + a 6 + a 18) = 8 / 11 :=
by
  sorry

end arithmetic_sequence_geometric_term_ratio_l1132_113297


namespace unique_solution_eq_condition_l1132_113204

theorem unique_solution_eq_condition (p q : ℝ) :
  (∃! x : ℝ, (2 * x - 2 * p + q) / (2 * x - 2 * p - q) = (2 * q + p + x) / (2 * q - p - x)) ↔ (p = 3 * q / 4 ∧ q ≠ 0) :=
  sorry

end unique_solution_eq_condition_l1132_113204


namespace largest_n_value_l1132_113217

theorem largest_n_value (n : ℕ) (h : (1 / 5 : ℝ) + (n / 8 : ℝ) + 1 < 2) : n ≤ 6 :=
by
  sorry

end largest_n_value_l1132_113217


namespace time_after_3577_minutes_l1132_113284

-- Definitions
def startingTime : Nat := 6 * 60 -- 6:00 PM in minutes
def startDate : String := "2020-12-31"
def durationMinutes : Nat := 3577
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24

-- Theorem to prove that 3577 minutes after 6:00 PM on December 31, 2020 is January 3 at 5:37 AM
theorem time_after_3577_minutes : 
  (durationMinutes + startingTime) % (hoursInDay * minutesInHour) = 5 * minutesInHour + 37 :=
  by
  sorry -- proof goes here

end time_after_3577_minutes_l1132_113284


namespace determine_g_two_l1132_113230

variables (a b c d p q r s : ℝ) -- Define variables a, b, c, d, p, q, r, s as real numbers
variables (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) -- The conditions a < b < c < d

noncomputable def f (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d
noncomputable def g (x : ℝ) : ℝ := (x - 1/p) * (x - 1/q) * (x - 1/r) * (x - 1/s)

noncomputable def g_two := g 2
noncomputable def f_two := f 2

theorem determine_g_two :
  g_two a b c d = (16 + 8*a + 4*b + 2*c + d) / (p*q*r*s) :=
sorry

end determine_g_two_l1132_113230


namespace bus_seating_capacity_l1132_113233

-- Conditions
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seat_capacity : ℕ := 3
def back_seat_capacity : ℕ := 9
def total_seats : ℕ := left_side_seats + right_side_seats

-- Proof problem statement
theorem bus_seating_capacity :
  (total_seats * seat_capacity) + back_seat_capacity = 90 := by
  sorry

end bus_seating_capacity_l1132_113233


namespace solve_inequality_l1132_113227

theorem solve_inequality (x : ℝ) : x + 1 > 3 → x > 2 := 
sorry

end solve_inequality_l1132_113227


namespace john_finish_work_alone_in_48_days_l1132_113293

variable {J R : ℝ}

theorem john_finish_work_alone_in_48_days
  (h1 : J + R = 1 / 24)
  (h2 : 16 * (J + R) = 2 / 3)
  (h3 : 16 * J = 1 / 3) :
  1 / J = 48 := 
by
  sorry

end john_finish_work_alone_in_48_days_l1132_113293


namespace range_a_l1132_113228

theorem range_a (a : ℝ) :
  (∀ x : ℝ, a ≤ x ∧ x ≤ 3 → -1 ≤ -x^2 + 2 * x + 2 ∧ -x^2 + 2 * x + 2 ≤ 3) →
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l1132_113228


namespace arithmetic_sequence_ninth_term_l1132_113221

theorem arithmetic_sequence_ninth_term 
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29)
  : a + 8 * d = 35 :=
by
  sorry

end arithmetic_sequence_ninth_term_l1132_113221


namespace P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l1132_113294

-- Define the problem conditions and questions
def P_1 (n : ℕ) : ℚ := sorry
def P_2 (n : ℕ) : ℚ := sorry

-- Part (a)
theorem P2_3_eq_2_3 : P_2 3 = 2 / 3 := sorry

-- Part (b)
theorem P1_n_eq_1_n (n : ℕ) (h : n ≥ 1): P_1 n = 1 / n := sorry

-- Part (c)
theorem P2_recurrence (n : ℕ) (h : n ≥ 2) : 
  P_2 n = (2 / n) * P_1 (n-1) + ((n-2) / n) * P_2 (n-1) := sorry

-- Part (d)
theorem P2_n_eq_2_n (n : ℕ) (h : n ≥ 1): P_2 n = 2 / n := sorry

end P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l1132_113294


namespace percentage_of_teachers_without_issues_l1132_113267

theorem percentage_of_teachers_without_issues (total_teachers : ℕ) 
    (high_bp_teachers : ℕ) (heart_issue_teachers : ℕ) 
    (both_issues_teachers : ℕ) (h1 : total_teachers = 150) 
    (h2 : high_bp_teachers = 90) 
    (h3 : heart_issue_teachers = 60) 
    (h4 : both_issues_teachers = 30) : 
    (total_teachers - (high_bp_teachers + heart_issue_teachers - both_issues_teachers)) / total_teachers * 100 = 20 :=
by sorry

end percentage_of_teachers_without_issues_l1132_113267


namespace distance_focus_asymptote_l1132_113283

noncomputable def focus := (Real.sqrt 6 / 2, 0)
def asymptote (x y : ℝ) := x - Real.sqrt 2 * y = 0
def hyperbola (x y : ℝ) := x^2 - 2 * y^2 = 1

theorem distance_focus_asymptote :
  let d := (Real.sqrt 6 / 2, 0)
  let A := 1
  let B := -Real.sqrt 2
  let C := 0
  let numerator := abs (A * d.1 + B * d.2 + C)
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator = Real.sqrt 2 / 2 :=
sorry

end distance_focus_asymptote_l1132_113283


namespace max_composite_numbers_l1132_113279

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l1132_113279


namespace num_factors_of_M_l1132_113225

theorem num_factors_of_M (M : ℕ) 
  (hM : M = (2^5) * (3^4) * (5^3) * (11^2)) : ∃ n : ℕ, n = 360 ∧ M = (2^5) * (3^4) * (5^3) * (11^2) := 
by
  sorry

end num_factors_of_M_l1132_113225


namespace rectangle_area_l1132_113247

theorem rectangle_area
  (b : ℝ)
  (l : ℝ)
  (P : ℝ)
  (h1 : l = 3 * b)
  (h2 : P = 2 * (l + b))
  (h3 : P = 112) :
  l * b = 588 := by
  sorry

end rectangle_area_l1132_113247


namespace largest_neg_int_solution_l1132_113288

theorem largest_neg_int_solution :
  ∃ x : ℤ, 26 * x + 8 ≡ 4 [ZMOD 18] ∧ ∀ y : ℤ, 26 * y + 8 ≡ 4 [ZMOD 18] → y < -14 → false :=
by
  sorry

end largest_neg_int_solution_l1132_113288


namespace feet_count_l1132_113235

-- We define the basic quantities
def total_heads : ℕ := 50
def num_hens : ℕ := 30
def num_cows : ℕ := total_heads - num_hens
def hens_feet : ℕ := num_hens * 2
def cows_feet : ℕ := num_cows * 4
def total_feet : ℕ := hens_feet + cows_feet

-- The theorem we want to prove
theorem feet_count : total_feet = 140 :=
  by
  sorry

end feet_count_l1132_113235


namespace average_of_numbers_is_correct_l1132_113244

theorem average_of_numbers_is_correct :
  let nums := [12, 13, 14, 510, 520, 530, 1120, 1, 1252140, 2345]
  let sum_nums := 1253205
  let count_nums := 10
  (sum_nums / count_nums.toFloat) = 125320.5 :=
by {
  sorry
}

end average_of_numbers_is_correct_l1132_113244


namespace office_light_ratio_l1132_113258

theorem office_light_ratio (bedroom_light: ℕ) (living_room_factor: ℕ) (total_energy: ℕ) 
  (time: ℕ) (ratio: ℕ) (office_light: ℕ) :
  bedroom_light = 6 →
  living_room_factor = 4 →
  total_energy = 96 →
  time = 2 →
  ratio = 3 →
  total_energy = (bedroom_light * time) + (office_light * time) + ((bedroom_light * living_room_factor) * time) →
  (office_light / bedroom_light) = ratio :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  -- The actual solution steps would go here
  sorry

end office_light_ratio_l1132_113258


namespace customers_tried_sample_l1132_113252

theorem customers_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left_over : ℕ)
  (samples_per_customer : ℕ := 1)
  (h_samples_per_box : samples_per_box = 20)
  (h_boxes_opened : boxes_opened = 12)
  (h_samples_left_over : samples_left_over = 5) :
  (samples_per_box * boxes_opened - samples_left_over) / samples_per_customer = 235 :=
by
  sorry

end customers_tried_sample_l1132_113252


namespace max_value_f_l1132_113203

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem max_value_f : ∃ x ∈ (Set.Icc 0 (Real.pi / 2)), f x = Real.pi / 12 + Real.sqrt 3 / 2 :=
by
  sorry

end max_value_f_l1132_113203


namespace sum_of_transformed_parabolas_is_non_horizontal_line_l1132_113273

theorem sum_of_transformed_parabolas_is_non_horizontal_line
    (a b c : ℝ)
    (f g : ℝ → ℝ)
    (hf : ∀ x, f x = a * (x - 8)^2 + b * (x - 8) + c)
    (hg : ∀ x, g x = -a * (x + 8)^2 - b * (x + 8) - (c - 3)) :
    ∃ m q : ℝ, ∀ x : ℝ, (f x + g x) = m * x + q ∧ m ≠ 0 :=
by sorry

end sum_of_transformed_parabolas_is_non_horizontal_line_l1132_113273


namespace initial_typists_count_l1132_113254

theorem initial_typists_count
  (letters_per_20_min : Nat)
  (letters_total_1_hour : Nat)
  (letters_typists_count : Nat)
  (n_typists_init : Nat)
  (h1 : letters_per_20_min = 46)
  (h2 : letters_typists_count = 30)
  (h3 : letters_total_1_hour = 207) :
  n_typists_init = 20 :=
by {
  sorry
}

end initial_typists_count_l1132_113254


namespace least_8_heavy_three_digit_l1132_113277

def is_8_heavy (n : ℕ) : Prop :=
  n % 8 > 6

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem least_8_heavy_three_digit : ∃ n : ℕ, is_three_digit n ∧ is_8_heavy n ∧ ∀ m : ℕ, is_three_digit m ∧ is_8_heavy m → n ≤ m := 
sorry

end least_8_heavy_three_digit_l1132_113277


namespace sum_of_money_l1132_113264

theorem sum_of_money (P R : ℝ) (h : (P * 2 * (R + 3) / 100) = (P * 2 * R / 100) + 300) : P = 5000 :=
by
    -- We are given that the sum of money put at 2 years SI rate is Rs. 300 more when rate is increased by 3%.
    sorry

end sum_of_money_l1132_113264


namespace smallest_n_divides_l1132_113280

theorem smallest_n_divides (m : ℕ) (h1 : m % 2 = 1) (h2 : m > 2) :
  ∃ n : ℕ, 2^(1988) = n ∧ 2^1989 ∣ m^n - 1 :=
by
  sorry

end smallest_n_divides_l1132_113280


namespace quadratic_inequality_empty_solution_range_l1132_113241

theorem quadratic_inequality_empty_solution_range (b : ℝ) :
  (∀ x : ℝ, ¬ (x^2 + b * x + 1 ≤ 0)) ↔ -2 < b ∧ b < 2 :=
by
  sorry

end quadratic_inequality_empty_solution_range_l1132_113241


namespace smallest_natural_number_l1132_113202

theorem smallest_natural_number (n : ℕ) (h : 2006 ^ 1003 < n ^ 2006) : n ≥ 45 := 
by {
    sorry
}

end smallest_natural_number_l1132_113202


namespace bruce_initial_money_l1132_113246

-- Definitions of the conditions
def cost_crayons : ℕ := 5 * 5
def cost_books : ℕ := 10 * 5
def cost_calculators : ℕ := 3 * 5
def total_spent : ℕ := cost_crayons + cost_books + cost_calculators
def cost_bags : ℕ := 11 * 10
def initial_money : ℕ := total_spent + cost_bags

-- Theorem statement
theorem bruce_initial_money :
  initial_money = 200 := by
  sorry

end bruce_initial_money_l1132_113246


namespace find_A_from_equation_and_conditions_l1132_113261

theorem find_A_from_equation_and_conditions 
  (A B C D : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : 10 * A + B ≠ 0)
  (h8 : 10 * 10 * 10 * A + 10 * 10 * B + 8 * 10 + 2 - (900 + C * 10 + 9) = 490 + 3 * 10 + D) :
  A = 5 :=
by
  sorry

end find_A_from_equation_and_conditions_l1132_113261


namespace solve_inequality_l1132_113259

theorem solve_inequality (x : ℝ) : 
  (x / (x^2 + x - 6) ≥ 0) ↔ (x < -3) ∨ (x = 0) ∨ (0 < x ∧ x < 2) :=
by 
  sorry 

end solve_inequality_l1132_113259


namespace time_to_school_gate_l1132_113220

theorem time_to_school_gate (total_time gate_to_building building_to_room time_to_gate : ℕ) 
                            (h1 : total_time = 30)
                            (h2 : gate_to_building = 6)
                            (h3 : building_to_room = 9)
                            (h4 : total_time = time_to_gate + gate_to_building + building_to_room) :
  time_to_gate = 15 :=
  sorry

end time_to_school_gate_l1132_113220


namespace negation_of_exists_lt_zero_l1132_113231

theorem negation_of_exists_lt_zero (m : ℝ) :
  ¬ (∃ x : ℝ, x < 0 ∧ x^2 + 2 * x - m > 0) ↔ ∀ x : ℝ, x < 0 → x^2 + 2 * x - m ≤ 0 :=
by sorry

end negation_of_exists_lt_zero_l1132_113231


namespace j_at_4_l1132_113238

noncomputable def h (x : ℚ) : ℚ := 5 / (3 - x)

noncomputable def h_inv (x : ℚ) : ℚ := (3 * x - 5) / x

noncomputable def j (x : ℚ) : ℚ := (1 / h_inv x) + 7

theorem j_at_4 : j 4 = 53 / 7 :=
by
  -- Proof steps would be inserted here.
  sorry

end j_at_4_l1132_113238


namespace initial_girls_count_l1132_113208

variable (p : ℕ) -- total number of people initially in the group
variable (initial_girls : ℕ) -- number of girls initially

-- Condition 1: Initially, 50% of the group are girls
def initially_fifty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop := initial_girls = p / 2

-- Condition 2: Three girls leave and three boys arrive
def after_girls_leave_and_boys_arrive (initial_girls : ℕ) : ℕ := initial_girls - 3

-- Condition 3: After the change, 40% of the group are girls
def after_the_change_forty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop :=
  (after_girls_leave_and_boys_arrive initial_girls) = 2 * (p / 5)

theorem initial_girls_count (p : ℕ) (initial_girls : ℕ) :
  initially_fifty_percent_girls p initial_girls →
  after_the_change_forty_percent_girls p initial_girls →
  initial_girls = 15 := by
  sorry

end initial_girls_count_l1132_113208


namespace arithmetic_sequence_properties_l1132_113265

theorem arithmetic_sequence_properties (a : ℕ → ℤ) (T : ℕ → ℤ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) (h2 : a 4 = a 2 + 4) (h3 : a 3 = 6) :
  (∀ n, a n = 2 * n) ∧ (∀ n, T n = (4 / 3 * (4^n - 1))) :=
by
  sorry

end arithmetic_sequence_properties_l1132_113265


namespace find_total_results_l1132_113268

noncomputable def total_results (S : ℕ) (n : ℕ) (sum_first6 sum_last6 sixth_result : ℕ) :=
  (S = 52 * n) ∧ (sum_first6 = 6 * 49) ∧ (sum_last6 = 6 * 52) ∧ (sixth_result = 34)

theorem find_total_results {S n sum_first6 sum_last6 sixth_result : ℕ} :
  total_results S n sum_first6 sum_last6 sixth_result → n = 11 :=
by
  intros h
  sorry

end find_total_results_l1132_113268


namespace megan_bottles_l1132_113242

theorem megan_bottles (initial_bottles drank gave_away remaining_bottles : ℕ) 
  (h1 : initial_bottles = 45)
  (h2 : drank = 8)
  (h3 : gave_away = 12) :
  remaining_bottles = initial_bottles - (drank + gave_away) :=
by 
  sorry

end megan_bottles_l1132_113242


namespace sum_of_leading_digits_l1132_113271

def leading_digit (n : ℕ) (x : ℝ) : ℕ := sorry

def M := 10^500 - 1

def g (r : ℕ) : ℕ := leading_digit r (M^(1 / r))

theorem sum_of_leading_digits :
  g 3 + g 4 + g 5 + g 7 + g 8 = 10 := sorry

end sum_of_leading_digits_l1132_113271


namespace divides_x_by_5_l1132_113296

theorem divides_x_by_5 (x y : ℤ) (hx1 : 1 < x) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : 2 * x^2 - 1 = y^15) : 5 ∣ x := by
  sorry

end divides_x_by_5_l1132_113296


namespace cat_weight_problem_l1132_113274

variable (female_cat_weight male_cat_weight : ℕ)

theorem cat_weight_problem
  (h1 : male_cat_weight = 2 * female_cat_weight)
  (h2 : female_cat_weight + male_cat_weight = 6) :
  female_cat_weight = 2 :=
by
  sorry

end cat_weight_problem_l1132_113274


namespace S6_value_l1132_113278

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ := x^m + (1/x)^m

theorem S6_value (x : ℝ) (h : x + 1/x = 4) : S_m x 6 = 2700 :=
by
  -- Skipping proof
  sorry

end S6_value_l1132_113278


namespace donna_has_40_bananas_l1132_113255

-- Define the number of bananas each person has
variables (dawn lydia donna total : ℕ)

-- State the conditions
axiom h1 : dawn + lydia + donna = total
axiom h2 : dawn = lydia + 40
axiom h3 : lydia = 60
axiom h4 : total = 200

-- State the theorem to be proved
theorem donna_has_40_bananas : donna = 40 :=
by {
  sorry -- Placeholder for the proof
}

end donna_has_40_bananas_l1132_113255
