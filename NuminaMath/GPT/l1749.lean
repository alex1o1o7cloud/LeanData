import Mathlib

namespace sqrt_calc_l1749_174914

theorem sqrt_calc : Real.sqrt (Real.sqrt (0.00032 ^ (1 / 5))) = 0.669 := by
  sorry

end sqrt_calc_l1749_174914


namespace integer_solution_unique_l1749_174948

theorem integer_solution_unique (x y : ℝ) (h : -1 < (y - x) / (x + y) ∧ (y - x) / (x + y) < 2) (hyx : ∃ n : ℤ, y = n * x) : y = x :=
by
  sorry

end integer_solution_unique_l1749_174948


namespace remainder_x150_l1749_174984

theorem remainder_x150 (x : ℝ) : 
  ∃ r : ℝ, ∃ q : ℝ, x^150 = q * (x - 1)^3 + 11175*x^2 - 22200*x + 11026 := 
by
  sorry

end remainder_x150_l1749_174984


namespace not_even_or_odd_l1749_174977

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem not_even_or_odd : ¬(∀ x : ℝ, f (-x) = f x) ∧ ¬(∀ x : ℝ, f (-x) = -f x) := by
  sorry

end not_even_or_odd_l1749_174977


namespace right_triangle_third_side_square_l1749_174981

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) 
  (h₁ : a = 3) (h₂ : b = 4) (h₃ : a^2 + b^2 = c^2) :
  c^2 = 25 ∨ a^2 + c^2 = b^2 ∨ a^2 + b^2 = 7 :=
by
  sorry

end right_triangle_third_side_square_l1749_174981


namespace min_value_of_u_l1749_174944

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : a^2 - b + 4 ≤ 0)

theorem min_value_of_u : (∃ (u : ℝ), u = (2*a + 3*b) / (a + b) ∧ u ≥ 14/5) :=
sorry

end min_value_of_u_l1749_174944


namespace trig_problem_l1749_174965

variable (α : ℝ)

theorem trig_problem
  (h1 : Real.sin (Real.pi + α) = -1 / 3) :
  Real.cos (α - 3 * Real.pi / 2) = -1 / 3 ∧
  (Real.sin (Real.pi / 2 + α) = 2 * Real.sqrt 2 / 3 ∨ Real.sin (Real.pi / 2 + α) = -2 * Real.sqrt 2 / 3) ∧
  (Real.tan (5 * Real.pi - α) = -Real.sqrt 2 / 4 ∨ Real.tan (5 * Real.pi - α) = Real.sqrt 2 / 4) :=
sorry

end trig_problem_l1749_174965


namespace pure_imaginary_product_imaginary_part_fraction_l1749_174942

-- Part 1
theorem pure_imaginary_product (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i) :
  (z1 * z2).re = 0 ↔ m = 0 := 
sorry

-- Part 2
theorem imaginary_part_fraction (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i)
  (h4 : z1^2 - 2 * z1 + 2 = 0) :
  (z2 / z1).im = -1 / 2 :=
sorry

end pure_imaginary_product_imaginary_part_fraction_l1749_174942


namespace books_sum_l1749_174988

theorem books_sum (darryl_books lamont_books loris_books danielle_books : ℕ) 
  (h1 : darryl_books = 20)
  (h2 : lamont_books = 2 * darryl_books)
  (h3 : lamont_books = loris_books + 3)
  (h4 : danielle_books = lamont_books + darryl_books + 10) : 
  darryl_books + lamont_books + loris_books + danielle_books = 167 := 
by
  sorry

end books_sum_l1749_174988


namespace complex_number_condition_l1749_174998

theorem complex_number_condition (b : ℝ) :
  (2 + b) / 5 = (2 * b - 1) / 5 → b = 3 :=
by
  sorry

end complex_number_condition_l1749_174998


namespace female_employees_count_l1749_174926

theorem female_employees_count (E Male_E Female_E M : ℕ)
  (h1: M = (2 / 5) * E)
  (h2: 200 = (E - Male_E) * (2 / 5))
  (h3: M = (2 / 5) * Male_E + 200) :
  Female_E = 500 := by
{
  sorry
}

end female_employees_count_l1749_174926


namespace ratio_of_steps_l1749_174921

-- Defining the conditions of the problem
def andrew_steps : ℕ := 150
def jeffrey_steps : ℕ := 200

-- Stating the theorem that we need to prove
theorem ratio_of_steps : andrew_steps / Nat.gcd andrew_steps jeffrey_steps = 3 ∧ jeffrey_steps / Nat.gcd andrew_steps jeffrey_steps = 4 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_steps_l1749_174921


namespace determine_a_b_l1749_174997

-- Step d) The Lean 4 statement for the transformed problem
theorem determine_a_b (a b : ℝ) (h : ∀ t : ℝ, (t^2 + t + 1) * 1^2 - 2 * (a + t)^2 * 1 + t^2 + 3 * a * t + b = 0) : 
  a = 1 ∧ b = 1 := 
sorry

end determine_a_b_l1749_174997


namespace father_son_speed_ratio_l1749_174915

theorem father_son_speed_ratio
  (F S t : ℝ)
  (distance_hallway : ℝ)
  (distance_meet_from_father : ℝ)
  (H1 : distance_hallway = 16)
  (H2 : distance_meet_from_father = 12)
  (H3 : 12 = F * t)
  (H4 : 4 = S * t)
  : F / S = 3 := by
  sorry

end father_son_speed_ratio_l1749_174915


namespace find_t_l1749_174910

theorem find_t (s t : ℤ) (h1 : 12 * s + 7 * t = 173) (h2 : s = t - 3) : t = 11 :=
by
  sorry

end find_t_l1749_174910


namespace initial_population_of_first_village_l1749_174968

theorem initial_population_of_first_village (P : ℕ) :
  (P - 1200 * 18) = (42000 + 800 * 18) → P = 78000 :=
by
  sorry

end initial_population_of_first_village_l1749_174968


namespace soccer_team_points_l1749_174973

theorem soccer_team_points 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (points_per_win : ℕ) 
  (points_per_draw : ℕ) 
  (points_per_loss : ℕ) 
  (draws : ℕ := total_games - (wins + losses)) : 
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  46 = (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) :=
by sorry

end soccer_team_points_l1749_174973


namespace largest_prime_divisor_l1749_174964

-- Let n be a positive integer
def is_positive_integer (n : ℕ) : Prop :=
  n > 0

-- Define that n equals the sum of the squares of its four smallest positive divisors
def is_sum_of_squares_of_smallest_divisors (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a = 2 ∧ b = 5 ∧ c = 10 ∧ n = 1 + a^2 + b^2 + c^2

-- Prove that the largest prime divisor of n is 13
theorem largest_prime_divisor (n : ℕ) (h1 : is_positive_integer n) (h2 : is_sum_of_squares_of_smallest_divisors n) :
  ∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Prime q ∧ q ∣ n → q ≤ p ∧ p = 13 :=
by
  sorry

end largest_prime_divisor_l1749_174964


namespace minimum_value_l1749_174903

noncomputable def problem_statement (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 4 * c^2

theorem minimum_value : ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27), 
  problem_statement a b c h = 180 :=
sorry

end minimum_value_l1749_174903


namespace area_of_remaining_shape_l1749_174969

/-- Define the initial 6x6 square grid with each cell of size 1 cm. -/
def initial_square_area : ℝ := 6 * 6

/-- Define the area of the combined dark gray triangles forming a 1x3 rectangle. -/
def dark_gray_area : ℝ := 1 * 3

/-- Define the area of the combined light gray triangles forming a 2x3 rectangle. -/
def light_gray_area : ℝ := 2 * 3

/-- Define the total area of the gray triangles cut out. -/
def total_gray_area : ℝ := dark_gray_area + light_gray_area

/-- Calculate the area of the remaining figure after cutting out the gray triangles. -/
def remaining_area : ℝ := initial_square_area - total_gray_area

/-- Proof that the area of the remaining shape is 27 square centimeters. -/
theorem area_of_remaining_shape : remaining_area = 27 := by
  sorry

end area_of_remaining_shape_l1749_174969


namespace sum_zero_of_absolute_inequalities_l1749_174932

theorem sum_zero_of_absolute_inequalities 
  (a b c : ℝ) 
  (h1 : |a| ≥ |b + c|) 
  (h2 : |b| ≥ |c + a|) 
  (h3 : |c| ≥ |a + b|) :
  a + b + c = 0 := 
  by
    sorry

end sum_zero_of_absolute_inequalities_l1749_174932


namespace sin_240_eq_neg_sqrt3_div_2_l1749_174957

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l1749_174957


namespace polygon_sides_l1749_174947

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l1749_174947


namespace part1_proof_l1749_174962

variable (α β t x1 x2 : ℝ)

-- Conditions
def quadratic_roots := 2 * α ^ 2 - t * α - 2 = 0 ∧ 2 * β ^ 2 - t * β - 2 = 0
def roots_relation := α + β = t / 2 ∧ α * β = -1
def points_in_interval := α < β ∧ α ≤ x1 ∧ x1 ≤ β ∧ α ≤ x2 ∧ x2 ≤ β ∧ x1 ≠ x2

-- Proof of Part 1
theorem part1_proof (h1 : quadratic_roots α β t) (h2 : roots_relation α β t)
                    (h3 : points_in_interval α β x1 x2) : 
                    4 * x1 * x2 - t * (x1 + x2) - 4 < 0 := 
sorry

end part1_proof_l1749_174962


namespace modulus_of_z_l1749_174940

-- Define the given condition
def condition (z : ℂ) : Prop := (z - 3) * (1 - 3 * Complex.I) = 10

-- State the main theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = 5 :=
sorry

end modulus_of_z_l1749_174940


namespace number_of_zeros_l1749_174924

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then |x| - 2 else 2 * x - 6 + Real.log x

theorem number_of_zeros :
  (∃ x : ℝ, f x = 0) ∧ (∃ y : ℝ, f y = 0) ∧ (∀ z : ℝ, f z = 0 → z = x ∨ z = y) :=
by
  sorry

end number_of_zeros_l1749_174924


namespace beautiful_fold_probability_l1749_174970

noncomputable def probability_beautiful_fold (a : ℝ) : ℝ := 1 / 2

theorem beautiful_fold_probability 
  (A B C D F : ℝ × ℝ) 
  (ABCD_square : (A.1 = 0) ∧ (A.2 = 0) ∧ 
                 (B.1 = a) ∧ (B.2 = 0) ∧ 
                 (C.1 = a) ∧ (C.2 = a) ∧ 
                 (D.1 = 0) ∧ (D.2 = a))
  (F_in_square : 0 ≤ F.1 ∧ F.1 ≤ a ∧ 0 ≤ F.2 ∧ F.2 ≤ a):
  probability_beautiful_fold a = 1 / 2 :=
sorry

end beautiful_fold_probability_l1749_174970


namespace pears_morning_sales_l1749_174971

theorem pears_morning_sales (morning afternoon : ℕ) 
  (h1 : afternoon = 2 * morning)
  (h2 : morning + afternoon = 360) : 
  morning = 120 := 
sorry

end pears_morning_sales_l1749_174971


namespace simplify_expression_l1749_174949

noncomputable def expression : ℝ :=
  (4 * (Real.sqrt 3 + Real.sqrt 7)) / (5 * Real.sqrt (3 + (1 / 2)))

theorem simplify_expression : expression = (16 + 8 * Real.sqrt 21) / 35 := by
  sorry

end simplify_expression_l1749_174949


namespace max_weight_American_l1749_174902

noncomputable def max_weight_of_American_swallow (A E : ℕ) : Prop :=
A = 5 ∧ 2 * E + E = 90 ∧ 60 * A + 60 * 2 * A = 600

theorem max_weight_American (A E : ℕ) : max_weight_of_American_swallow A E :=
by
  sorry

end max_weight_American_l1749_174902


namespace eval_polynomial_at_neg2_l1749_174943

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := x^4 + x^3 + x^2 + x + 1

-- Statement of the problem, proving that the polynomial equals 11 when x = -2
theorem eval_polynomial_at_neg2 : polynomial (-2) = 11 := by
  sorry

end eval_polynomial_at_neg2_l1749_174943


namespace range_of_m_l1749_174954

theorem range_of_m (m : ℝ) (h₁ : ∀ x : ℝ, -x^2 + 7*x + 8 ≥ 0 → x^2 - 7*x - 8 ≤ 0)
  (h₂ : ∀ x : ℝ, x^2 - 2*x + 1 - 4*m^2 ≤ 0 → 1 - 2*m ≤ x ∧ x ≤ 1 + 2*m)
  (not_p_sufficient_for_not_q : ∀ x : ℝ, ¬(-x^2 + 7*x + 8 ≥ 0) → ¬(x^2 - 2*x + 1 - 4*m^2 ≤ 0))
  (suff_non_necess : ∀ x : ℝ, (x^2 - 2*x + 1 - 4*m^2 ≤ 0) → ¬(x^2 - 7*x - 8 ≤ 0))
  : 0 < m ∧ m ≤ 1 := sorry

end range_of_m_l1749_174954


namespace reduced_expression_none_of_these_l1749_174952

theorem reduced_expression_none_of_these (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : b ≠ a^2) (h4 : ab ≠ a^3) :
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 1 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (b^2 + b) / (b - a^2) ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 0 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (a^2 + b) / (a^2 - b) :=
by
  sorry

end reduced_expression_none_of_these_l1749_174952


namespace pyramid_sphere_proof_l1749_174925

theorem pyramid_sphere_proof
  (h R_1 R_2 : ℝ) 
  (O_1 O_2 T_1 T_2 : ℝ) 
  (inscription: h > 0 ∧ R_1 > 0 ∧ R_2 > 0) :
  R_1 * R_2 * h^2 = (R_1^2 - O_1 * T_1^2) * (R_2^2 - O_2 * T_2^2) :=
by
  sorry

end pyramid_sphere_proof_l1749_174925


namespace smallest_integer_is_10_l1749_174919

noncomputable def smallest_integer (a b c : ℕ) : ℕ :=
  if h : (a + b + c = 90) ∧ (2 * b = 3 * a) ∧ (5 * a = 2 * c)
  then a
  else 0

theorem smallest_integer_is_10 (a b c : ℕ) (h₁ : a + b + c = 90) (h₂ : 2 * b = 3 * a) (h₃ : 5 * a = 2 * c) : 
  smallest_integer a b c = 10 :=
sorry

end smallest_integer_is_10_l1749_174919


namespace probability_sum_of_five_l1749_174930

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 4

theorem probability_sum_of_five :
  favorable_outcomes / total_outcomes = 1 / 9 := 
by
  sorry

end probability_sum_of_five_l1749_174930


namespace inscribed_circle_radius_eq_l1749_174992

noncomputable def inscribedCircleRadius :=
  let AB := 6
  let AC := 7
  let BC := 8
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  let r := K / s
  r

theorem inscribed_circle_radius_eq :
  inscribedCircleRadius = Real.sqrt 413.4375 / 10.5 := by
  sorry

end inscribed_circle_radius_eq_l1749_174992


namespace max_valid_n_eq_3210_l1749_174941

-- Define the digit sum function S
def digit_sum (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- The condition S(3n) = 3S(n) and all digits of n are distinct
def valid_n (n : ℕ) : Prop :=
  digit_sum (3 * n) = 3 * digit_sum n ∧ (Nat.digits 10 n).Nodup

-- Prove that the maximum value of such n is 3210
theorem max_valid_n_eq_3210 : ∃ n : ℕ, valid_n n ∧ n = 3210 :=
by
  existsi 3210
  sorry

end max_valid_n_eq_3210_l1749_174941


namespace no_possible_arrangement_of_balloons_l1749_174982

/-- 
  There are 10 balloons hanging in a row: blue and green. This statement proves that it is impossible 
  to arrange 10 balloons such that between every two blue balloons, there is an even number of 
  balloons and between every two green balloons, there is an odd number of balloons.
--/

theorem no_possible_arrangement_of_balloons :
  ¬ (∃ (color : Fin 10 → Bool), 
    (∀ i j, i < j ∧ color i = color j ∧ color i = tt → (j - i - 1) % 2 = 0) ∧
    (∀ i j, i < j ∧ color i = color j ∧ color i = ff → (j - i - 1) % 2 = 1)) :=
by
  sorry

end no_possible_arrangement_of_balloons_l1749_174982


namespace distance_to_city_hall_l1749_174994

variable (d : ℝ) (t : ℝ)

-- Conditions
def condition1 : Prop := d = 45 * (t + 1.5)
def condition2 : Prop := d - 45 = 65 * (t - 1.25)
def condition3 : Prop := t > 0

theorem distance_to_city_hall
  (h1 : condition1 d t)
  (h2 : condition2 d t)
  (h3 : condition3 t)
  : d = 300 := by
  sorry

end distance_to_city_hall_l1749_174994


namespace deepak_profit_share_l1749_174931

theorem deepak_profit_share (anand_investment : ℕ) (deepak_investment : ℕ) (total_profit : ℕ) 
  (h₁ : anand_investment = 22500) 
  (h₂ : deepak_investment = 35000) 
  (h₃ : total_profit = 13800) : 
  (14 * total_profit / (9 + 14)) = 8400 := 
by
  sorry

end deepak_profit_share_l1749_174931


namespace non_mobile_payment_probability_40_60_l1749_174986

variable (total_customers : ℕ)
variable (num_non_mobile_40_50 : ℕ)
variable (num_non_mobile_50_60 : ℕ)

theorem non_mobile_payment_probability_40_60 
  (h_total_customers: total_customers = 100)
  (h_num_non_mobile_40_50: num_non_mobile_40_50 = 9)
  (h_num_non_mobile_50_60: num_non_mobile_50_60 = 5) : 
  (num_non_mobile_40_50 + num_non_mobile_50_60 : ℚ) / total_customers = 7 / 50 :=
by
  -- Placeholder for the actual proof
  sorry

end non_mobile_payment_probability_40_60_l1749_174986


namespace sum_of_six_selected_primes_is_even_l1749_174946

noncomputable def prob_sum_even_when_selecting_six_primes : ℚ := 
  let first_twenty_primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
  let num_ways_to_choose_6_without_even_sum := Nat.choose 19 6
  let total_num_ways_to_choose_6 := Nat.choose 20 6
  num_ways_to_choose_6_without_even_sum / total_num_ways_to_choose_6

theorem sum_of_six_selected_primes_is_even : 
  prob_sum_even_when_selecting_six_primes = 354 / 505 := 
sorry

end sum_of_six_selected_primes_is_even_l1749_174946


namespace correct_result_l1749_174995

-- Definitions to capture the problem conditions:
def cond1 (a b : ℤ) : Prop := 5 * a^2 * b - 2 * a^2 * b = 3 * a^2 * b
def cond2 (x : ℤ) : Prop := x^6 / x^2 = x^4
def cond3 (a b : ℤ) : Prop := (a - b)^2 = a^2 - b^2

-- Proof statement to verify the correct answer
theorem correct_result (x : ℤ) : (2 * x^2)^3 = 8 * x^6 :=
  by sorry

-- Note that cond1, cond2, and cond3 are intended to capture the erroneous conditions mentioned for completeness.

end correct_result_l1749_174995


namespace no_valid_pairs_of_real_numbers_l1749_174906

theorem no_valid_pairs_of_real_numbers :
  ∀ (a b : ℝ), ¬ (∃ (x y : ℤ), 3 * a * x + 7 * b * y = 3 ∧ x^2 + y^2 = 85 ∧ (x % 5 = 0 ∨ y % 5 = 0)) :=
by
  sorry

end no_valid_pairs_of_real_numbers_l1749_174906


namespace Buratino_can_solve_l1749_174939

theorem Buratino_can_solve :
  ∃ (MA TE TI KA : ℕ), MA ≠ TE ∧ MA ≠ TI ∧ MA ≠ KA ∧ TE ≠ TI ∧ TE ≠ KA ∧ TI ≠ KA ∧
  MA * TE * MA * TI * KA = 2016000 :=
by
  -- skip the proof using sorry
  sorry

end Buratino_can_solve_l1749_174939


namespace find_c_plus_1_over_b_l1749_174972

theorem find_c_plus_1_over_b (a b c : ℝ) (h1: a * b * c = 1) 
    (h2: a + 1 / c = 7) (h3: b + 1 / a = 12) : c + 1 / b = 21 / 83 := 
by 
    sorry

end find_c_plus_1_over_b_l1749_174972


namespace isosceles_triangle_base_length_l1749_174983

theorem isosceles_triangle_base_length (x : ℝ) (h1 : 2 * x + 2 * x + x = 20) : x = 4 :=
sorry

end isosceles_triangle_base_length_l1749_174983


namespace chuck_team_score_proof_chuck_team_score_l1749_174960

-- Define the conditions
def yellow_team_score : ℕ := 55
def lead : ℕ := 17

-- State the main proposition
theorem chuck_team_score (yellow_team_score : ℕ) (lead : ℕ) : ℕ :=
yellow_team_score + lead

-- Formulate the final proof goal
theorem proof_chuck_team_score : chuck_team_score yellow_team_score lead = 72 :=
by {
  -- This is the place where the proof should go
  sorry
}

end chuck_team_score_proof_chuck_team_score_l1749_174960


namespace pebbles_game_invariant_l1749_174928

/-- 
The game of pebbles is played on an infinite board of lattice points (i, j).
Initially, there is a pebble at (0, 0).
A move consists of removing a pebble from point (i, j) and placing a pebble at each of the points (i+1, j) and (i, j+1) provided both are vacant.
Show that at any stage of the game there is a pebble at some lattice point (a, b) with 0 ≤ a + b ≤ 3. 
-/
theorem pebbles_game_invariant :
  ∀ (board : ℕ × ℕ → Prop) (initial_state : board (0, 0)) (move : (ℕ × ℕ) → Prop → Prop → Prop),
  (∀ (i j : ℕ), board (i, j) → ¬ board (i+1, j) ∧ ¬ board (i, j+1) → board (i+1, j) ∧ board (i, j+1)) →
  ∃ (a b : ℕ), (0 ≤ a + b ∧ a + b ≤ 3) ∧ board (a, b) :=
by
  intros board initial_state move move_rule
  sorry 

end pebbles_game_invariant_l1749_174928


namespace trigonometric_identity_l1749_174993

theorem trigonometric_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (70 * Real.pi / 180) +
   Real.sin (10 * Real.pi / 180) * Real.sin (50 * Real.pi / 180)) = 1 / 4 :=
by sorry

end trigonometric_identity_l1749_174993


namespace problem_proof_l1749_174978

variables {m n : ℝ}

-- Line definitions
def l1 (m n x y : ℝ) : Prop := m * x + 8 * y + n = 0
def l2 (m x y : ℝ) : Prop := 2 * x + m * y - 1 = 0

-- Conditions
def intersects_at (m n : ℝ) : Prop :=
  l1 m n m (-1) ∧ l2 m m (-1)

def parallel (m n : ℝ) : Prop :=
  (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2)

def perpendicular (m n : ℝ) : Prop :=
  m = 0 ∧ n = 8

theorem problem_proof :
  intersects_at m n → (m = 1 ∧ n = 7) ∧
  parallel m n → (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2) ∧
  perpendicular m n → (m = 0 ∧ n = 8) :=
by
  sorry

end problem_proof_l1749_174978


namespace followers_after_one_year_l1749_174900

theorem followers_after_one_year :
  let initial_followers := 100000
  let daily_new_followers := 1000
  let unfollowers_per_year := 20000
  let days_per_year := 365
  initial_followers + (daily_new_followers * days_per_year - unfollowers_per_year) = 445000 :=
by
  sorry

end followers_after_one_year_l1749_174900


namespace man_double_son_age_in_2_years_l1749_174927

def present_age_son : ℕ := 25
def age_difference : ℕ := 27
def years_to_double_age : ℕ := 2

theorem man_double_son_age_in_2_years 
  (S : ℕ := present_age_son)
  (M : ℕ := S + age_difference)
  (Y : ℕ := years_to_double_age) : 
  M + Y = 2 * (S + Y) :=
by sorry

end man_double_son_age_in_2_years_l1749_174927


namespace resting_time_is_thirty_l1749_174990

-- Defining the conditions as Lean 4 definitions
def speed := 10 -- miles per hour
def time_first_part := 30 -- minutes
def distance_second_part := 15 -- miles
def distance_third_part := 20 -- miles
def total_time := 270 -- minutes

-- Function to convert hours to minutes
def hours_to_minutes (h : ℕ) : ℕ := h * 60

-- Problem statement in Lean 4: Proving the resting time is 30 minutes
theorem resting_time_is_thirty :
  let distance_first := speed * (time_first_part / 60)
  let time_second_part := (distance_second_part / speed) * 60
  let time_third_part := (distance_third_part / speed) * 60
  let times_sum := time_first_part + time_second_part + time_third_part
  total_time = times_sum + 30 := 
  sorry

end resting_time_is_thirty_l1749_174990


namespace installation_time_l1749_174923

-- Definitions (based on conditions)
def total_windows := 14
def installed_windows := 8
def hours_per_window := 8

-- Define what we need to prove
def remaining_windows := total_windows - installed_windows
def total_install_hours := remaining_windows * hours_per_window

theorem installation_time : total_install_hours = 48 := by
  sorry

end installation_time_l1749_174923


namespace percentage_of_copper_first_alloy_l1749_174938

theorem percentage_of_copper_first_alloy :
  ∃ x : ℝ, 
  (66 * x / 100) + (55 * 21 / 100) = 121 * 15 / 100 ∧
  x = 10 := 
sorry

end percentage_of_copper_first_alloy_l1749_174938


namespace initial_weights_of_apples_l1749_174958

variables {A B : ℕ}

theorem initial_weights_of_apples (h₁ : A + B = 75) (h₂ : A - 5 = (B + 5) + 7) :
  A = 46 ∧ B = 29 :=
by
  sorry

end initial_weights_of_apples_l1749_174958


namespace Eddy_travel_time_l1749_174937

theorem Eddy_travel_time :
  ∀ (T_F D_F D_E : ℕ) (S_ratio : ℝ),
    T_F = 4 →
    D_F = 360 →
    D_E = 600 →
    S_ratio = 2.2222222222222223 →
    ((D_F / T_F : ℝ) * S_ratio ≠ 0) →
    D_E / ((D_F / T_F) * S_ratio) = 3 :=
by
  intros T_F D_F D_E S_ratio ht hf hd hs hratio
  sorry  -- Proof to be provided

end Eddy_travel_time_l1749_174937


namespace least_value_difference_l1749_174935

noncomputable def least_difference (x : ℝ) : ℝ := 6 - 13/5

theorem least_value_difference (x n m : ℝ) (h1 : 2*x + 5 + 4*x - 3 > x + 15)
                               (h2 : 2*x + 5 + x + 15 > 4*x - 3)
                               (h3 : 4*x - 3 + x + 15 > 2*x + 5)
                               (h4 : x + 15 > 2*x + 5)
                               (h5 : x + 15 > 4*x - 3)
                               (h_m : m = 13/5) (h_n : n = 6)
                               (hx : m < x ∧ x < n) :
  n - m = 17 / 5 :=
  by sorry

end least_value_difference_l1749_174935


namespace min_value_of_function_l1749_174975

noncomputable def y (θ : ℝ) : ℝ := (2 - Real.sin θ) / (1 - Real.cos θ)

theorem min_value_of_function : ∃ θ : ℝ, y θ = 3 / 4 :=
sorry

end min_value_of_function_l1749_174975


namespace problem_l1749_174987

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l1749_174987


namespace arithmetical_puzzle_l1749_174999

theorem arithmetical_puzzle (S I X T W E N : ℕ) 
  (h1 : S = 1) 
  (h2 : N % 2 = 0) 
  (h3 : (1 * 100 + I * 10 + X) * 3 = T * 1000 + W * 100 + E * 10 + N) 
  (h4 : ∀ (a b c d e f : ℕ), 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f) :
  T = 5 := sorry

end arithmetical_puzzle_l1749_174999


namespace total_number_of_animals_l1749_174936

-- Definitions for the animal types
def heads_per_hen := 2
def legs_per_hen := 8
def heads_per_peacock := 3
def legs_per_peacock := 9
def heads_per_zombie_hen := 6
def legs_per_zombie_hen := 12

-- Given total heads and legs
def total_heads := 800
def total_legs := 2018

-- Proof that the total number of animals is 203
theorem total_number_of_animals : 
  ∀ (H P Z : ℕ), 
    heads_per_hen * H + heads_per_peacock * P + heads_per_zombie_hen * Z = total_heads
    ∧ legs_per_hen * H + legs_per_peacock * P + legs_per_zombie_hen * Z = total_legs 
    → H + P + Z = 203 :=
by
  sorry

end total_number_of_animals_l1749_174936


namespace calculate_f_at_8_l1749_174920

def f (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 27 * x^2 - 24 * x - 72

theorem calculate_f_at_8 : f 8 = 952 :=
by sorry

end calculate_f_at_8_l1749_174920


namespace probability_answered_within_first_four_rings_l1749_174966

theorem probability_answered_within_first_four_rings 
  (P1 P2 P3 P4 : ℝ) (h1 : P1 = 0.1) (h2 : P2 = 0.3) (h3 : P3 = 0.4) (h4 : P4 = 0.1) :
  (1 - ((1 - P1) * (1 - P2) * (1 - P3) * (1 - P4))) = 0.9 := 
sorry

end probability_answered_within_first_four_rings_l1749_174966


namespace f_2002_eq_0_l1749_174980

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_2_eq_0 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem f_2002_eq_0 : f 2002 = 0 :=
by
  sorry

end f_2002_eq_0_l1749_174980


namespace multiplication_of_powers_of_10_l1749_174974

theorem multiplication_of_powers_of_10 : (10 : ℝ) ^ 65 * (10 : ℝ) ^ 64 = (10 : ℝ) ^ 129 := by
  sorry

end multiplication_of_powers_of_10_l1749_174974


namespace sequence_general_formula_l1749_174953

theorem sequence_general_formula :
  ∀ (a : ℕ → ℝ),
  (a 1 = 1) →
  (∀ n : ℕ, n > 0 → a n - a (n + 1) = 2 * a n * a (n + 1) / (n * (n + 1))) →
  ∀ n : ℕ, n > 0 → a n = n / (3 * n - 2) :=
by
  intros a h1 h_rec n hn
  sorry

end sequence_general_formula_l1749_174953


namespace solution_set_inequality_l1749_174950

theorem solution_set_inequality (x : ℝ) : (x^2 - 2*x - 8 ≥ 0) ↔ (x ≤ -2 ∨ x ≥ 4) := 
sorry

end solution_set_inequality_l1749_174950


namespace option_b_is_factorization_l1749_174934

theorem option_b_is_factorization (m : ℝ) :
  m^2 - 1 = (m + 1) * (m - 1) :=
sorry

end option_b_is_factorization_l1749_174934


namespace inradius_of_triangle_l1749_174979

theorem inradius_of_triangle (A p s r : ℝ) 
  (h1 : A = (1/2) * p) 
  (h2 : p = 2 * s) 
  (h3 : A = r * s) : 
  r = 1 :=
by
  sorry

end inradius_of_triangle_l1749_174979


namespace white_cannot_lose_l1749_174918

-- Define a type to represent the game state
structure Game :=
  (state : Type)
  (white_move : state → state)
  (black_move : state → state)
  (initial : state)

-- Define a type to represent the double chess game conditions
structure DoubleChess extends Game :=
  (double_white_move : state → state)
  (double_black_move : state → state)

-- Define the hypothesis based on the conditions
noncomputable def white_has_no_losing_strategy (g : DoubleChess) : Prop :=
  ∃ s, g.double_white_move (g.double_white_move s) = g.initial

theorem white_cannot_lose (g : DoubleChess) :
  white_has_no_losing_strategy g :=
sorry

end white_cannot_lose_l1749_174918


namespace train_length_l1749_174922

theorem train_length (L : ℝ) (h1 : ∀ t1 : ℝ, t1 = 15 → ∀ p1 : ℝ, p1 = 180 → (L + p1) / t1 = v)
(h2 : ∀ t2 : ℝ, t2 = 20 → ∀ p2 : ℝ, p2 = 250 → (L + p2) / t2 = v) : 
L = 30 :=
by
  have h1 := h1 15 rfl 180 rfl
  have h2 := h2 20 rfl 250 rfl
  sorry

end train_length_l1749_174922


namespace gross_profit_value_l1749_174991

theorem gross_profit_value (C GP : ℝ) (h1 : GP = 1.6 * C) (h2 : 91 = C + GP) : GP = 56 :=
by
  sorry

end gross_profit_value_l1749_174991


namespace parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l1749_174961

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 2 * x

theorem parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2 :
  parabola_equation 1 (-Real.sqrt 2) :=
by
  sorry

end parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l1749_174961


namespace Craig_bench_press_percentage_l1749_174976

theorem Craig_bench_press_percentage {Dave_weight : ℕ} (h1 : Dave_weight = 175) (h2 : ∀ w : ℕ, Dave_bench_press = 3 * Dave_weight) 
(Craig_bench_press Mark_bench_press : ℕ) (h3 : Mark_bench_press = 55) (h4 : Mark_bench_press = Craig_bench_press - 50) : 
(Craig_bench_press / (3 * Dave_weight) * 100) = 20 := by
  sorry

end Craig_bench_press_percentage_l1749_174976


namespace product_abc_l1749_174929

theorem product_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_eqn : a * b * c = a * b^3) (h_c_eq_1 : c = 1) :
  a * b * c = a :=
by
  sorry

end product_abc_l1749_174929


namespace difference_between_numbers_l1749_174909

theorem difference_between_numbers (x y d : ℝ) (h1 : x + y = 10) (h2 : x - y = d) (h3 : x^2 - y^2 = 80) : d = 8 :=
by {
  sorry
}

end difference_between_numbers_l1749_174909


namespace number_of_small_gardens_l1749_174913

def totalSeeds : ℕ := 85
def tomatoSeeds : ℕ := 42
def capsicumSeeds : ℕ := 26
def cucumberSeeds : ℕ := 17

def plantedTomatoSeeds : ℕ := 24
def plantedCucumberSeeds : ℕ := 17

def remainingTomatoSeeds : ℕ := tomatoSeeds - plantedTomatoSeeds
def remainingCapsicumSeeds : ℕ := capsicumSeeds
def remainingCucumberSeeds : ℕ := cucumberSeeds - plantedCucumberSeeds

def seedsInSmallGardenTomato : ℕ := 2
def seedsInSmallGardenCapsicum : ℕ := 1
def seedsInSmallGardenCucumber : ℕ := 1

theorem number_of_small_gardens : (remainingTomatoSeeds / seedsInSmallGardenTomato = 9) :=
by 
  sorry

end number_of_small_gardens_l1749_174913


namespace vanessa_deleted_30_files_l1749_174917

-- Define the initial conditions
def original_files : Nat := 16 + 48
def files_left : Nat := 34

-- Define the number of files deleted
def files_deleted : Nat := original_files - files_left

-- The theorem to prove the number of files deleted
theorem vanessa_deleted_30_files : files_deleted = 30 := by
  sorry

end vanessa_deleted_30_files_l1749_174917


namespace least_sum_of_exponents_l1749_174955

theorem least_sum_of_exponents {n : ℕ} (h : n = 520) (h_exp : ∃ (a b : ℕ), 2^a + 2^b = n ∧ a ≠ b ∧ a = 9 ∧ b = 3) : 
    (∃ (s : ℕ), s = 9 + 3) :=
by
  sorry

end least_sum_of_exponents_l1749_174955


namespace sin_cos_value_l1749_174985

variable (α : ℝ) (a b : ℝ × ℝ)
def vectors_parallel : Prop := b = (Real.sin α, Real.cos α) ∧
a = (4, 3) ∧ (∃ k : ℝ, a = (k * (Real.sin α), k * (Real.cos α)))

theorem sin_cos_value (h : vectors_parallel α a b) : ((Real.sin α) * (Real.cos α)) = 12 / 25 :=
by
  sorry

end sin_cos_value_l1749_174985


namespace average_marks_passed_l1749_174945

noncomputable def total_candidates := 120
noncomputable def total_average_marks := 35
noncomputable def passed_candidates := 100
noncomputable def failed_candidates := total_candidates - passed_candidates
noncomputable def average_marks_failed := 15
noncomputable def total_marks := total_average_marks * total_candidates
noncomputable def total_marks_failed := average_marks_failed * failed_candidates

theorem average_marks_passed :
  ∃ P, P * passed_candidates + total_marks_failed = total_marks ∧ P = 39 := by
  sorry

end average_marks_passed_l1749_174945


namespace second_cannibal_wins_l1749_174959

/-- Define a data structure for the position on the chessboard -/
structure Position where
  x : Nat
  y : Nat
  deriving Inhabited, DecidableEq

/-- Check if two positions are adjacent in a legal move (vertical or horizontal) -/
def isAdjacent (p1 p2 : Position) : Bool :=
  (p1.x = p2.x ∧ (p1.y = p2.y + 1 ∨ p1.y = p2.y - 1)) ∨
  (p1.y = p2.y ∧ (p1.x = p2.x + 1 ∨ p1.x = p2.x - 1))

/-- Define the initial positions of the cannibals -/
def initialPositionFirstCannibal : Position := ⟨1, 1⟩
def initialPositionSecondCannibal : Position := ⟨8, 8⟩

/-- Define a move function for a cannibal (a valid move should keep it on the board) -/
def move (p : Position) (direction : String) : Position :=
  match direction with
  | "up"     => if p.y < 8 then ⟨p.x, p.y + 1⟩ else p
  | "down"   => if p.y > 1 then ⟨p.x, p.y - 1⟩ else p
  | "left"   => if p.x > 1 then ⟨p.x - 1, p.y⟩ else p
  | "right"  => if p.x < 8 then ⟨p.x + 1, p.y⟩ else p
  | _        => p

/-- Predicate determining if a cannibal can eat the other by moving to its position -/
def canEat (p1 p2 : Position) : Bool :=
  p1 = p2

/-- 
  Prove that the second cannibal will eat the first cannibal with the correct strategy. 
  We formalize the fact that with correct play, starting from the initial positions, 
  the second cannibal (initially at ⟨8, 8⟩) can always force a win.
-/
theorem second_cannibal_wins :
  ∀ (p1 p2 : Position), 
  p1 = initialPositionFirstCannibal →
  p2 = initialPositionSecondCannibal →
  (∃ strategy : (Position → String), ∀ positionFirstCannibal : Position, canEat (move p2 (strategy p2)) positionFirstCannibal) :=
by
  sorry

end second_cannibal_wins_l1749_174959


namespace george_initial_candy_l1749_174989

theorem george_initial_candy (number_of_bags : ℕ) (pieces_per_bag : ℕ) 
  (h1 : number_of_bags = 8) (h2 : pieces_per_bag = 81) : 
  number_of_bags * pieces_per_bag = 648 := 
by 
  sorry

end george_initial_candy_l1749_174989


namespace perimeter_F_is_18_l1749_174956

-- Define the dimensions of the rectangles.
def vertical_rectangle : ℤ × ℤ := (3, 5)
def horizontal_rectangle : ℤ × ℤ := (1, 5)

-- Define the perimeter calculation for a single rectangle.
def perimeter (width_height : ℤ × ℤ) : ℤ :=
  2 * width_height.1 + 2 * width_height.2

-- The overlapping width and height.
def overlap_width : ℤ := 5
def overlap_height : ℤ := 1

-- Perimeter of the letter F.
def perimeter_F : ℤ :=
  perimeter vertical_rectangle + perimeter horizontal_rectangle - 2 * overlap_width

-- Statement to prove.
theorem perimeter_F_is_18 : perimeter_F = 18 := by sorry

end perimeter_F_is_18_l1749_174956


namespace num_solutions_eq_3_l1749_174933

theorem num_solutions_eq_3 : 
  ∃ (x1 x2 x3 : ℝ), (∀ x : ℝ, 2^x - 2 * (⌊x⌋:ℝ) - 1 = 0 → x = x1 ∨ x = x2 ∨ x = x3) 
  ∧ ¬ ∃ x4, (2^x4 - 2 * (⌊x4⌋:ℝ) - 1 = 0 ∧ x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3) :=
sorry

end num_solutions_eq_3_l1749_174933


namespace xyz_plus_54_l1749_174905

theorem xyz_plus_54 (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y + z = 53) (h2 : y * z + x = 53) (h3 : z * x + y = 53) : 
  x + y + z = 54 := by
  sorry

end xyz_plus_54_l1749_174905


namespace smallest_b_base_45b_perfect_square_l1749_174963

theorem smallest_b_base_45b_perfect_square : ∃ b : ℕ, b > 3 ∧ (∃ n : ℕ, n^2 = 4 * b + 5) ∧ ∀ b' : ℕ, b' > 3 ∧ (∃ n' : ℕ, n'^2 = 4 * b' + 5) → b ≤ b' := 
sorry

end smallest_b_base_45b_perfect_square_l1749_174963


namespace probability_red_ball_first_occurrence_l1749_174904

theorem probability_red_ball_first_occurrence 
  (P : ℕ → ℝ) : 
  ∃ (P1 P2 P3 P4 : ℝ),
    P 1 = 0.4 ∧ P 2 = 0.3 ∧ P 3 = 0.2 ∧ P 4 = 0.1 :=
  sorry

end probability_red_ball_first_occurrence_l1749_174904


namespace simplify_fraction_l1749_174996

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l1749_174996


namespace union_of_A_and_B_intersection_of_A_and_B_l1749_174907

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 4 }
noncomputable def B : Set ℝ := { x | x > 3 ∨ x < 1 }

theorem union_of_A_and_B : A ∪ B = Set.univ :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = { x | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4) } :=
by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_l1749_174907


namespace total_weight_of_courtney_marble_collection_l1749_174967

def marble_weight_first_jar : ℝ := 80 * 0.35
def marble_weight_second_jar : ℝ := 160 * 0.45
def marble_weight_third_jar : ℝ := 20 * 0.25

/-- The total weight of Courtney's marble collection -/
theorem total_weight_of_courtney_marble_collection :
    marble_weight_first_jar + marble_weight_second_jar + marble_weight_third_jar = 105 := by
  sorry

end total_weight_of_courtney_marble_collection_l1749_174967


namespace parallelogram_height_l1749_174916

variable (base height area : ℝ)
variable (h_eq_diag : base = 30)
variable (h_eq_area : area = 600)

theorem parallelogram_height :
  (height = 20) ↔ (base * height = area) :=
by
  sorry

end parallelogram_height_l1749_174916


namespace third_side_length_not_12_l1749_174951

theorem third_side_length_not_12 (x : ℕ) (h1 : x % 2 = 0) (h2 : 5 < x) (h3 : x < 11) : x ≠ 12 := 
sorry

end third_side_length_not_12_l1749_174951


namespace length_of_bridge_l1749_174911

noncomputable def convert_speed (km_per_hour : ℝ) : ℝ := km_per_hour * (1000 / 3600)

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (passing_time : ℝ)
  (total_distance_covered : ℝ)
  (bridge_length : ℝ) :
  train_length = 120 →
  train_speed_kmh = 40 →
  passing_time = 25.2 →
  total_distance_covered = convert_speed train_speed_kmh * passing_time →
  bridge_length = total_distance_covered - train_length →
  bridge_length = 160 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end length_of_bridge_l1749_174911


namespace coin_problem_l1749_174908

variable (x y S k : ℕ)

theorem coin_problem
  (h1 : x + y = 14)
  (h2 : 2 * x + 5 * y = S)
  (h3 : S = k + 2 * k)
  (h4 : k * 4 = S) :
  y = 4 ∨ y = 8 ∨ y = 12 :=
by
  sorry

end coin_problem_l1749_174908


namespace geometric_progression_first_term_and_ratio_l1749_174912

theorem geometric_progression_first_term_and_ratio (
  b_1 q : ℝ
) :
  b_1 * (1 + q + q^2) = 21 →
  b_1^2 * (1 + q^2 + q^4) = 189 →
  (b_1 = 12 ∧ q = 1/2) ∨ (b_1 = 3 ∧ q = 2) :=
by
  intros hsum hsumsq
  sorry

end geometric_progression_first_term_and_ratio_l1749_174912


namespace girls_in_class_l1749_174901

theorem girls_in_class :
  ∀ (x : ℕ), (12 * 84 + 92 * x = 86 * (12 + x)) → x = 4 :=
by
  sorry

end girls_in_class_l1749_174901
