import Mathlib

namespace radius_of_first_cylinder_l24_24431

theorem radius_of_first_cylinder :
  ∀ (rounds1 rounds2 : ℕ) (r2 r1 : ℝ), rounds1 = 70 → rounds2 = 49 → r2 = 20 → 
  (2 * Real.pi * r1 * rounds1 = 2 * Real.pi * r2 * rounds2) → r1 = 14 :=
by
  sorry

end radius_of_first_cylinder_l24_24431


namespace matrix_equation_l24_24537

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 5], ![-6, -2]]
def p : ℤ := 2
def q : ℤ := -18

theorem matrix_equation :
  M * M = p • M + q • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  sorry

end matrix_equation_l24_24537


namespace algebraic_expression_value_l24_24962

theorem algebraic_expression_value (x : ℝ) (h : x = 2 * Real.sqrt 3 - 1) : x^2 + 2 * x - 3 = 8 :=
by 
  sorry

end algebraic_expression_value_l24_24962


namespace unique_b_for_quadratic_l24_24256

theorem unique_b_for_quadratic (c : ℝ) (h_c : c ≠ 0) : (∃! b : ℝ, b > 0 ∧ (2*b + 2/b)^2 - 4*c = 0) → c = 4 :=
by
  sorry

end unique_b_for_quadratic_l24_24256


namespace age_ratio_in_two_years_l24_24121

-- Definitions of conditions
def son_present_age : ℕ := 26
def age_difference : ℕ := 28
def man_present_age : ℕ := son_present_age + age_difference

-- Future ages after 2 years
def son_future_age : ℕ := son_present_age + 2
def man_future_age : ℕ := man_present_age + 2

-- The theorem to prove
theorem age_ratio_in_two_years : (man_future_age / son_future_age) = 2 := 
by
  -- Step-by-Step proof would go here
  sorry

end age_ratio_in_two_years_l24_24121


namespace avg_temp_correct_l24_24365

-- Defining the temperatures for each day from March 1st to March 5th
def day_1_temp := 55.0
def day_2_temp := 59.0
def day_3_temp := 60.0
def day_4_temp := 57.0
def day_5_temp := 64.0

-- Calculating the average temperature
def avg_temp := (day_1_temp + day_2_temp + day_3_temp + day_4_temp + day_5_temp) / 5.0

-- Proving that the average temperature equals 59.0°F
theorem avg_temp_correct : avg_temp = 59.0 := sorry

end avg_temp_correct_l24_24365


namespace correct_option_is_D_l24_24071

-- Define the expressions to be checked
def exprA (x : ℝ) := 3 * x + 2 * x = 5 * x^2
def exprB (x : ℝ) := -2 * x^2 * x^3 = 2 * x^5
def exprC (x y : ℝ) := (y + 3 * x) * (3 * x - y) = y^2 - 9 * x^2
def exprD (x y : ℝ) := (-2 * x^2 * y)^3 = -8 * x^6 * y^3

theorem correct_option_is_D (x y : ℝ) : 
  ¬ exprA x ∧ 
  ¬ exprB x ∧ 
  ¬ exprC x y ∧ 
  exprD x y := by
  -- The proof would be provided here
  sorry

end correct_option_is_D_l24_24071


namespace find_larger_box_ounces_l24_24410

-- Define the conditions
def ounces_smaller_box : ℕ := 20
def cost_smaller_box : ℝ := 3.40
def cost_larger_box : ℝ := 4.80
def best_value_price_per_ounce : ℝ := 0.16

-- Define the question and its expected answer
def expected_ounces_larger_box : ℕ := 30

-- Proof statement
theorem find_larger_box_ounces :
  (cost_larger_box / best_value_price_per_ounce = expected_ounces_larger_box) :=
by
  sorry

end find_larger_box_ounces_l24_24410


namespace contrapositive_example_l24_24244

variable (a b : ℝ)

theorem contrapositive_example
  (h₁ : a > 0)
  (h₃ : a + b < 0) :
  b < 0 := 
sorry

end contrapositive_example_l24_24244


namespace georges_final_score_l24_24694

theorem georges_final_score :
  (6 + 4) * 3 = 30 := 
by
  sorry

end georges_final_score_l24_24694


namespace sample_size_student_congress_l24_24447

-- Definitions based on the conditions provided in the problem
def num_classes := 40
def students_per_class := 3

-- Theorem statement for the mathematically equivalent proof problem
theorem sample_size_student_congress : 
  (num_classes * students_per_class) = 120 := 
by 
  sorry

end sample_size_student_congress_l24_24447


namespace tree_heights_l24_24989

theorem tree_heights :
  let Tree_A := 150
  let Tree_B := (2/3 : ℝ) * Tree_A
  let Tree_C := (1/2 : ℝ) * Tree_B
  let Tree_D := Tree_C + 25
  let Tree_E := 0.40 * Tree_A
  let Tree_F := (Tree_B + Tree_D) / 2
  let Tree_G := (3/8 : ℝ) * Tree_A
  let Tree_H := 1.25 * Tree_F
  let Tree_I := 0.60 * (Tree_E + Tree_G)
  let total_height := Tree_A + Tree_B + Tree_C + Tree_D + Tree_E + Tree_F + Tree_G + Tree_H + Tree_I
  Tree_A = 150 ∧
  Tree_B = 100 ∧
  Tree_C = 50 ∧
  Tree_D = 75 ∧
  Tree_E = 60 ∧
  Tree_F = 87.5 ∧
  Tree_G = 56.25 ∧
  Tree_H = 109.375 ∧
  Tree_I = 69.75 ∧
  total_height = 758.125 :=
by
  sorry

end tree_heights_l24_24989


namespace worker_savings_fraction_l24_24350

theorem worker_savings_fraction (P : ℝ) (F : ℝ) (h1 : P > 0) (h2 : 12 * F * P = 5 * (1 - F) * P) : F = 5 / 17 :=
by
  sorry

end worker_savings_fraction_l24_24350


namespace third_motorcyclist_speed_l24_24837

theorem third_motorcyclist_speed 
  (t₁ t₂ : ℝ)
  (x : ℝ)
  (h1 : t₁ - t₂ = 1.25)
  (h2 : 80 * t₁ = x * (t₁ - 0.5))
  (h3 : 60 * t₂ = x * (t₂ - 0.5))
  (h4 : x ≠ 60)
  (h5 : x ≠ 80):
  x = 100 :=
by
  sorry

end third_motorcyclist_speed_l24_24837


namespace find_a_10_l24_24516

/-- 
a_n is an arithmetic sequence
-/
def a (n : ℕ) : ℝ := sorry

/-- 
Given conditions:
- Condition 1: a_2 + a_5 = 19
- Condition 2: S_5 = 40, where S_5 is the sum of the first five terms
-/
axiom condition1 : a 2 + a 5 = 19
axiom condition2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 40

noncomputable def a_10 : ℝ := a 10

theorem find_a_10 : a_10 = 29 :=
by
  sorry

end find_a_10_l24_24516


namespace prove_d_value_l24_24391

-- Definitions of the conditions
def d (x : ℝ) : ℝ := x^4 - 2*x^3 + x^2 - 12*x - 5

-- The statement to prove
theorem prove_d_value (x : ℝ) (h : x^2 - 2*x - 5 = 0) : d x = 25 :=
sorry

end prove_d_value_l24_24391


namespace rectangle_with_perpendicular_diagonals_is_square_l24_24366

-- Define rectangle and its properties
structure Rectangle where
  length : ℝ
  width : ℝ
  opposite_sides_equal : length = width

-- Define the condition that the diagonals of the rectangle are perpendicular
axiom perpendicular_diagonals {r : Rectangle} : r.length = r.width → True

-- Define the square property that a rectangle with all sides equal is a square
structure Square extends Rectangle where
  all_sides_equal : length = width

-- The main theorem to be proven
theorem rectangle_with_perpendicular_diagonals_is_square (r : Rectangle) (h : r.length = r.width) : Square := by
  sorry

end rectangle_with_perpendicular_diagonals_is_square_l24_24366


namespace diff_of_squares_div_l24_24545

theorem diff_of_squares_div (a b : ℤ) (h1 : a = 121) (h2 : b = 112) : 
  (a^2 - b^2) / (a - b) = a + b :=
by
  rw [h1, h2]
  rw [sub_eq_add_neg, add_comm]
  exact sorry

end diff_of_squares_div_l24_24545


namespace polynomial_multiple_of_six_l24_24456

theorem polynomial_multiple_of_six 
  (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) :
  6 ∣ P.eval 5 :=
sorry

end polynomial_multiple_of_six_l24_24456


namespace inequality_abc_l24_24042

theorem inequality_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) :=
sorry

end inequality_abc_l24_24042


namespace shanghai_team_score_l24_24853

variables (S B : ℕ)

-- Conditions
def yao_ming_points : ℕ := 30
def point_margin : ℕ := 10
def total_points_minus_10 : ℕ := 5 * yao_ming_points - 10
def combined_total_points : ℕ := total_points_minus_10

-- The system of equations as conditions
axiom condition1 : S - B = point_margin
axiom condition2 : S + B = combined_total_points

-- The proof statement
theorem shanghai_team_score : S = 75 :=
by
  sorry

end shanghai_team_score_l24_24853


namespace arithmetic_sequence_ratio_l24_24463

-- Definitions based on conditions
variables {a_n b_n : ℕ → ℕ} -- Arithmetic sequences
variables {A_n B_n : ℕ → ℕ} -- Sums of the first n terms

-- Given condition
axiom sums_of_arithmetic_sequences (n : ℕ) : A_n n / B_n n = (7 * n + 1) / (4 * n + 27)

-- Theorem to prove
theorem arithmetic_sequence_ratio :
  ∀ (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℕ), 
    (∀ n, A_n n / B_n n = (7 * n + 1) / (4 * n + 27)) → 
    a_6 / b_6 = 78 / 71 := 
by {
  sorry
}

end arithmetic_sequence_ratio_l24_24463


namespace solve_for_y_l24_24193

theorem solve_for_y : ∀ (y : ℝ), 4 + 2.3 * y = 1.7 * y - 20 → y = -40 :=
by
  sorry

end solve_for_y_l24_24193


namespace rectangle_length_l24_24150

theorem rectangle_length :
  ∀ (side : ℕ) (width : ℕ) (length : ℕ), 
  side = 4 → 
  width = 8 → 
  side * side = width * length → 
  length = 2 := 
by
  -- sorry to skip the proof
  intros side width length h1 h2 h3
  sorry

end rectangle_length_l24_24150


namespace brownies_per_person_l24_24214

-- Define the conditions as constants
def columns : ℕ := 6
def rows : ℕ := 3
def people : ℕ := 6

-- Define the total number of brownies
def total_brownies : ℕ := columns * rows

-- Define the theorem to be proved
theorem brownies_per_person : total_brownies / people = 3 :=
by sorry

end brownies_per_person_l24_24214


namespace g_triple_application_l24_24610

def g (x : ℕ) : ℕ := 7 * x + 3

theorem g_triple_application : g (g (g 3)) = 1200 :=
by
  sorry

end g_triple_application_l24_24610


namespace work_days_for_A_and_B_l24_24680

theorem work_days_for_A_and_B (W_A W_B : ℝ) (h1 : W_A = (1/2) * W_B) (h2 : W_B = 1/21) : 
  1 / (W_A + W_B) = 14 := by 
  sorry

end work_days_for_A_and_B_l24_24680


namespace probability_fewer_heads_than_tails_is_793_over_2048_l24_24106

noncomputable def probability_fewer_heads_than_tails (n : ℕ) : ℝ :=
(793 / 2048 : ℚ)

theorem probability_fewer_heads_than_tails_is_793_over_2048 :
  probability_fewer_heads_than_tails 12 = (793 / 2048 : ℚ) :=
sorry

end probability_fewer_heads_than_tails_is_793_over_2048_l24_24106


namespace benny_birthday_money_l24_24177

-- Define conditions
def spent_on_gear : ℕ := 47
def left_over : ℕ := 32

-- Define the total amount Benny received
def total_money_received : ℕ := 79

-- Theorem statement
theorem benny_birthday_money (spent_on_gear : ℕ) (left_over : ℕ) : spent_on_gear + left_over = total_money_received :=
by
  sorry

end benny_birthday_money_l24_24177


namespace marble_prob_l24_24326

theorem marble_prob
  (a b x y m n : ℕ)
  (h1 : a + b = 30)
  (h2 : (x : ℚ) / a * (y : ℚ) / b = 4 / 9)
  (h3 : x * y = 36)
  (h4 : Nat.gcd m n = 1)
  (h5 : (a - x : ℚ) / a * (b - y) / b = m / n) :
  m + n = 29 := 
sorry

end marble_prob_l24_24326


namespace point_line_real_assoc_l24_24107

theorem point_line_real_assoc : 
  ∀ (p : ℝ), ∃! (r : ℝ), p = r := 
by 
  sorry

end point_line_real_assoc_l24_24107


namespace general_formula_sum_of_first_10_terms_l24_24821

variable (a : ℕ → ℝ) (d : ℝ) (S_10 : ℝ)
variable (h1 : a 5 = 11) (h2 : a 8 = 5)

theorem general_formula (n : ℕ) : a n = -2 * n + 21 :=
sorry

theorem sum_of_first_10_terms : S_10 = 100 :=
sorry

end general_formula_sum_of_first_10_terms_l24_24821


namespace solve_for_y_l24_24523

theorem solve_for_y : ∀ y : ℝ, (y - 5)^3 = (1 / 27)⁻¹ → y = 8 :=
by
  intro y
  intro h
  sorry

end solve_for_y_l24_24523


namespace gcd_168_54_264_l24_24862

theorem gcd_168_54_264 : Nat.gcd (Nat.gcd 168 54) 264 = 6 :=
by
  -- proof goes here and ends with sorry for now
  sorry

end gcd_168_54_264_l24_24862


namespace calculation_l24_24204

theorem calculation : 
  ((18 ^ 13 * 18 ^ 11) ^ 2 / 6 ^ 8) * 3 ^ 4 = 2 ^ 40 * 3 ^ 92 :=
by sorry

end calculation_l24_24204


namespace greatest_divisor_of_remainders_l24_24651

theorem greatest_divisor_of_remainders (x : ℕ) :
  (1442 % x = 12) ∧ (1816 % x = 6) ↔ x = 10 :=
by
  sorry

end greatest_divisor_of_remainders_l24_24651


namespace weeks_project_lasts_l24_24746

-- Definition of the conditions
def meal_cost : ℤ := 4
def people : ℤ := 4
def days_per_week : ℤ := 5
def total_spent : ℤ := 1280
def weekly_cost : ℤ := meal_cost * people * days_per_week

-- Problem statement: prove that the number of weeks the project will last equals 16 weeks.
theorem weeks_project_lasts : total_spent / weekly_cost = 16 := by 
  sorry

end weeks_project_lasts_l24_24746


namespace red_shirts_count_l24_24830

theorem red_shirts_count :
  ∀ (total blue_fraction green_fraction : ℕ),
    total = 60 →
    blue_fraction = total / 3 →
    green_fraction = total / 4 →
    (total - (blue_fraction + green_fraction)) = 25 :=
by
  intros total blue_fraction green_fraction h_total h_blue h_green
  rw [h_total, h_blue, h_green]
  norm_num
  sorry

end red_shirts_count_l24_24830


namespace find_m_l24_24578

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (m : ℕ)

theorem find_m (h1 : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
               (h2 : S (2 * m - 1) = 39)       -- sum of first (2m-1) terms
               (h3 : a (m - 1) + a (m + 1) - a m - 1 = 0)
               (h4 : m > 1) : 
               m = 20 :=
   sorry

end find_m_l24_24578


namespace complex_fraction_l24_24852

open Complex

/-- The given complex fraction \(\frac{5 - i}{1 - i}\) evaluates to \(3 + 2i\). -/
theorem complex_fraction : (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = ⟨3, 2⟩ :=
  by
  sorry

end complex_fraction_l24_24852


namespace no_real_roots_of_x_squared_plus_5_l24_24159

theorem no_real_roots_of_x_squared_plus_5 : ¬ ∃ (x : ℝ), x^2 + 5 = 0 :=
by
  sorry

end no_real_roots_of_x_squared_plus_5_l24_24159


namespace fraction_greater_than_decimal_l24_24176

theorem fraction_greater_than_decimal :
  (1 / 4 : ℝ) > (24999999 / (10^8 : ℝ)) + (1 / (4 * (10^8 : ℝ))) :=
by
  sorry

end fraction_greater_than_decimal_l24_24176


namespace find_real_solutions_l24_24968

noncomputable def cubic_eq_solutions (x : ℝ) : Prop := 
  x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3

theorem find_real_solutions : {x : ℝ | cubic_eq_solutions x} = {6} :=
by
  sorry

end find_real_solutions_l24_24968


namespace time_difference_l24_24307

def joey_time : ℕ :=
  let uphill := 12 / 6 * 60
  let downhill := 10 / 25 * 60
  let flat := 20 / 15 * 60
  uphill + downhill + flat

def sue_time : ℕ :=
  let downhill := 10 / 35 * 60
  let uphill := 12 / 12 * 60
  let flat := 20 / 25 * 60
  downhill + uphill + flat

theorem time_difference : joey_time - sue_time = 99 := by
  -- calculation steps skipped
  sorry

end time_difference_l24_24307


namespace count_triangles_in_figure_l24_24291

noncomputable def triangles_in_figure : ℕ := 53

theorem count_triangles_in_figure : triangles_in_figure = 53 := 
by sorry

end count_triangles_in_figure_l24_24291


namespace zeros_in_square_of_999_999_999_l24_24455

noncomputable def number_of_zeros_in_square (n : ℕ) : ℕ :=
  if n ≥ 1 then n - 1 else 0

theorem zeros_in_square_of_999_999_999 :
  number_of_zeros_in_square 9 = 8 :=
sorry

end zeros_in_square_of_999_999_999_l24_24455


namespace only_integer_solution_is_trivial_l24_24028

theorem only_integer_solution_is_trivial (a b c : ℤ) (h : 5 * a^2 + 9 * b^2 = 13 * c^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end only_integer_solution_is_trivial_l24_24028


namespace cost_of_one_pencil_and_one_pen_l24_24705

variables (x y : ℝ)

def eq1 := 4 * x + 3 * y = 3.70
def eq2 := 3 * x + 4 * y = 4.20

theorem cost_of_one_pencil_and_one_pen (h₁ : eq1 x y) (h₂ : eq2 x y) :
  x + y = 1.1286 :=
sorry

end cost_of_one_pencil_and_one_pen_l24_24705


namespace total_sweaters_knit_l24_24525

-- Definitions from condition a)
def monday_sweaters : ℕ := 8
def tuesday_sweaters : ℕ := monday_sweaters + 2
def wednesday_sweaters : ℕ := tuesday_sweaters - 4
def thursday_sweaters : ℕ := wednesday_sweaters
def friday_sweaters : ℕ := monday_sweaters / 2

-- Theorem statement
theorem total_sweaters_knit : 
  monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters = 34 :=
  by
    sorry

end total_sweaters_knit_l24_24525


namespace quarters_total_l24_24882

def initial_quarters : ℕ := 21
def additional_quarters : ℕ := 49
def total_quarters : ℕ := initial_quarters + additional_quarters

theorem quarters_total : total_quarters = 70 := by
  sorry

end quarters_total_l24_24882


namespace fractional_exponent_calculation_l24_24754

variables (a b : ℝ) -- Define a and b as real numbers
variable (ha : a > 0) -- Condition a > 0
variable (hb : b > 0) -- Condition b > 0

theorem fractional_exponent_calculation :
  (a^(2 * b^(1/4)) / (a * b^(1/2))^(1/2)) = a^(1/2) :=
by
  sorry -- Proof is not required, skip with sorry

end fractional_exponent_calculation_l24_24754


namespace two_largest_divisors_difference_l24_24389

theorem two_largest_divisors_difference (N : ℕ) (h : N > 1) (a : ℕ) (ha : a ∣ N) (h6a : 6 * a ∣ N) :
  (N / 2 : ℚ) / (N / 3 : ℚ) = 1.5 := by
  sorry

end two_largest_divisors_difference_l24_24389


namespace right_triangle_of_altitude_ratios_l24_24752

theorem right_triangle_of_altitude_ratios
  (h1 h2 h3 : ℝ) 
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0) 
  (h3_pos : h3 > 0) 
  (H : (h1 / h2)^2 + (h1 / h3)^2 = 1) : 
  ∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ h1 = 1 / a ∧ h2 = 1 / b ∧ h3 = 1 / c :=
sorry

end right_triangle_of_altitude_ratios_l24_24752


namespace perpendicularity_condition_l24_24201

theorem perpendicularity_condition 
  (A B C D E F k b : ℝ) 
  (h1 : b ≠ 0)
  (line : ∀ (x : ℝ), y = k * x + b)
  (curve : ∀ (x y : ℝ), A * x^2 + 2 * B * x * y + C * y^2 + 2 * D * x + 2 * E * y + F = 0):
  A * b^2 - 2 * D * k * b + F * k^2 + C * b^2 + 2 * E * b + F = 0 :=
sorry

end perpendicularity_condition_l24_24201


namespace ratio_first_part_l24_24759

theorem ratio_first_part (x : ℕ) (h1 : x / 3 = 2) : x = 6 :=
by
  sorry

end ratio_first_part_l24_24759


namespace peregrine_falcon_dive_time_l24_24604

theorem peregrine_falcon_dive_time 
  (bald_eagle_speed : ℝ := 100) 
  (peregrine_falcon_speed : ℝ := 2 * bald_eagle_speed) 
  (bald_eagle_time : ℝ := 30) : 
  peregrine_falcon_speed = 2 * bald_eagle_speed ∧ peregrine_falcon_speed / bald_eagle_speed = 2 →
  ∃ peregrine_falcon_time : ℝ, peregrine_falcon_time = 15 :=
by
  intro h
  use (bald_eagle_time / 2)
  sorry

end peregrine_falcon_dive_time_l24_24604


namespace max_abs_value_of_quadratic_function_l24_24049

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def point_in_band_region (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

theorem max_abs_value_of_quadratic_function (a b c t : ℝ) (h1 : point_in_band_region (quadratic_function a b c (-2) + 2) 0 4)
                                             (h2 : point_in_band_region (quadratic_function a b c 0 + 2) 0 4)
                                             (h3 : point_in_band_region (quadratic_function a b c 2 + 2) 0 4)
                                             (h4 : point_in_band_region (t + 1) (-1) 3) :
  |quadratic_function a b c t| ≤ 5 / 2 :=
sorry

end max_abs_value_of_quadratic_function_l24_24049


namespace problem1_problem2_l24_24215

def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x | x > 1 ∨ x < -6}

theorem problem1 (a : ℝ) : (setA a ∩ setB = ∅) → (-6 ≤ a ∧ a ≤ -2) := by
  intro h
  sorry

theorem problem2 (a : ℝ) : (setA a ∪ setB = setB) → (a < -9 ∨ a > 1) := by
  intro h
  sorry

end problem1_problem2_l24_24215


namespace mass_percentage_K_l24_24798

theorem mass_percentage_K (compound : Type) (m : ℝ) (mass_percentage : ℝ) (h : mass_percentage = 23.81) : mass_percentage = 23.81 :=
by
  sorry

end mass_percentage_K_l24_24798


namespace total_chairs_l24_24675

/-- Susan loves chairs. In her house, there are red chairs, yellow chairs, blue chairs, and green chairs.
    There are 5 red chairs. There are 4 times as many yellow chairs as red chairs.
    There are 2 fewer blue chairs than yellow chairs. The number of green chairs is half the sum of the number of red chairs and blue chairs (rounded down).
    We want to determine the total number of chairs in Susan's house. -/
theorem total_chairs (r y b g : ℕ) 
  (hr : r = 5)
  (hy : y = 4 * r) 
  (hb : b = y - 2) 
  (hg : g = (r + b) / 2) :
  r + y + b + g = 54 := 
sorry

end total_chairs_l24_24675


namespace geom_seq_product_l24_24730

noncomputable def geom_seq (a : ℕ → ℝ) := 
∀ n m: ℕ, ∃ r : ℝ, a (n + m) = a n * r ^ m

theorem geom_seq_product (a : ℕ → ℝ) 
  (h_seq : geom_seq a) 
  (h_pos : ∀ n, 0 < a n) 
  (h_log_sum : Real.log (a 3) + Real.log (a 6) + Real.log (a 9) = 3) : 
  a 1 * a 11 = 100 := 
sorry

end geom_seq_product_l24_24730


namespace sequence_satisfies_conditions_l24_24703

theorem sequence_satisfies_conditions : 
  let seq1 := [4, 1, 3, 1, 2, 4, 3, 2]
  let seq2 := [2, 3, 4, 2, 1, 3, 1, 4]
  (seq1[0] = 4 ∧ seq1[1] = 1 ∧ seq1[2] = 3 ∧ seq1[3] = 1 ∧ seq1[4] = 2 ∧ seq1[5] = 4 ∧ seq1[6] = 3 ∧ seq1[7] = 2)
  ∨ (seq2[0] = 2 ∧ seq2[1] = 3 ∧ seq2[2] = 4 ∧ seq2[3] = 2 ∧ seq2[4] = 1 ∧ seq2[5] = 3 ∧ seq2[6] = 1 ∧ seq2[7] = 4)
  ∧ (seq1[1] = 1 ∧ seq1[3] - seq1[1] = 2 ∧ seq1[4] - seq1[2] = 3 ∧ seq1[5] - seq1[2] = 4) := 
  sorry

end sequence_satisfies_conditions_l24_24703


namespace fraction_spent_first_week_l24_24979

theorem fraction_spent_first_week
  (S : ℝ) (F : ℝ)
  (h1 : S > 0)
  (h2 : F * S + 3 * (0.20 * S) + 0.15 * S = S) : 
  F = 0.25 := 
sorry

end fraction_spent_first_week_l24_24979


namespace fathers_age_l24_24161

variable (S F : ℕ)
variable (h1 : F = 3 * S)
variable (h2 : F + 15 = 2 * (S + 15))

theorem fathers_age : F = 45 :=
by
  -- the proof steps would go here
  sorry

end fathers_age_l24_24161


namespace largest_multiple_of_9_lt_120_is_117_l24_24229

theorem largest_multiple_of_9_lt_120_is_117 : ∃ k : ℕ, 9 * k < 120 ∧ (∀ m : ℕ, 9 * m < 120 → 9 * m ≤ 9 * k) ∧ 9 * k = 117 := 
by 
  sorry

end largest_multiple_of_9_lt_120_is_117_l24_24229


namespace construct_segment_AB_l24_24876

-- Define the two points A and B and assume the distance between them is greater than 1 meter
variables {A B : Point} (dist_AB_gt_1m : Distance A B > 1)

-- Define the ruler length as 10 cm
def ruler_length : ℝ := 0.1

theorem construct_segment_AB 
  (h : dist_AB_gt_1m) 
  (ruler : ℝ := ruler_length) : ∃ (AB : Segment), Distance A B = AB.length ∧ AB.length > 1 :=
sorry

end construct_segment_AB_l24_24876


namespace money_conditions_l24_24889

theorem money_conditions (a b : ℝ) (h1 : 4 * a - b > 32) (h2 : 2 * a + b = 26) : 
  a > 9.67 ∧ b < 6.66 := 
sorry

end money_conditions_l24_24889


namespace brownies_pieces_l24_24346

theorem brownies_pieces (tray_length tray_width piece_length piece_width : ℕ) 
  (h1 : tray_length = 24) 
  (h2 : tray_width = 16) 
  (h3 : piece_length = 2) 
  (h4 : piece_width = 2) : 
  tray_length * tray_width / (piece_length * piece_width) = 96 :=
by sorry

end brownies_pieces_l24_24346


namespace joan_jogged_3563_miles_l24_24734

noncomputable def steps_per_mile : ℕ := 1200

noncomputable def flips_per_year : ℕ := 28

noncomputable def steps_per_full_flip : ℕ := 150000

noncomputable def final_day_steps : ℕ := 75000

noncomputable def total_steps_in_year := flips_per_year * steps_per_full_flip + final_day_steps

noncomputable def miles_jogged := total_steps_in_year / steps_per_mile

theorem joan_jogged_3563_miles :
  miles_jogged = 3563 :=
by
  sorry

end joan_jogged_3563_miles_l24_24734


namespace directrix_of_parabola_l24_24064

theorem directrix_of_parabola : 
  (∀ (y x: ℝ), y^2 = 12 * x → x = -3) :=
sorry

end directrix_of_parabola_l24_24064


namespace complement_of_A_relative_to_U_l24_24611

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {x | 1 ≤ x ∧ x ≤ 3}

theorem complement_of_A_relative_to_U : (U \ A) = {4, 5, 6} := 
by
  sorry

end complement_of_A_relative_to_U_l24_24611


namespace non_negative_integer_solutions_l24_24867

theorem non_negative_integer_solutions (x : ℕ) : 3 * x - 2 < 7 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
  sorry

end non_negative_integer_solutions_l24_24867


namespace face_value_of_share_l24_24939

theorem face_value_of_share 
  (F : ℝ)
  (dividend_rate : ℝ := 0.09)
  (desired_return_rate : ℝ := 0.12)
  (market_value : ℝ := 15) 
  (h_eq : (dividend_rate * F) / market_value = desired_return_rate) : 
  F = 20 := 
by 
  sorry

end face_value_of_share_l24_24939


namespace apple_trees_count_l24_24066

-- Conditions
def num_peach_trees : ℕ := 45
def kg_per_peach_tree : ℕ := 65
def total_mass_fruit : ℕ := 7425
def kg_per_apple_tree : ℕ := 150
variable (A : ℕ)

-- Proof goal
theorem apple_trees_count (h : A * kg_per_apple_tree + num_peach_trees * kg_per_peach_tree = total_mass_fruit) : A = 30 := 
sorry

end apple_trees_count_l24_24066


namespace pears_in_basket_l24_24909

def TaniaFruits (b1 b2 b3 b4 b5 : ℕ) : Prop :=
  b1 = 18 ∧ b2 = 12 ∧ b3 = 9 ∧ b4 = b3 ∧ b5 + b1 + b2 + b3 + b4 = 58

theorem pears_in_basket {b1 b2 b3 b4 b5 : ℕ} (h : TaniaFruits b1 b2 b3 b4 b5) : b5 = 10 :=
by 
  sorry

end pears_in_basket_l24_24909


namespace rona_age_l24_24341

theorem rona_age (R : ℕ) (hR1 : ∀ Rachel Collete : ℕ, Rachel = 2 * R ∧ Collete = R / 2 ∧ Rachel - Collete = 12) : R = 12 :=
sorry

end rona_age_l24_24341


namespace find_range_of_a_l24_24699

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^3) / 3 - (a / 2) * x^2 + x + 1

def is_monotonically_decreasing_in (a : ℝ) (x : ℝ) : Prop := 
  ∀ s t : ℝ, (s ∈ Set.Ioo (3 / 2) 4) ∧ (t ∈ Set.Ioo (3 / 2) 4) ∧ s < t → 
  f a t ≤ f a s

theorem find_range_of_a :
  ∀ a : ℝ, is_monotonically_decreasing_in a x → 
  a ∈ Set.Ici (17/4)
:= sorry

end find_range_of_a_l24_24699


namespace triangle_is_isosceles_l24_24149

open Real

-- Define the basic setup of the triangle and the variables involved
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to A, B, and C respectively
variables (h1 : a * cos B = b * cos A) -- Given condition: a * cos B = b * cos A

-- The theorem stating that the given condition implies the triangle is isosceles
theorem triangle_is_isosceles (h1 : a * cos B = b * cos A) : A = B :=
sorry

end triangle_is_isosceles_l24_24149


namespace total_monthly_feed_l24_24333

def daily_feed (pounds_per_pig_per_day : ℕ) (number_of_pigs : ℕ) : ℕ :=
  pounds_per_pig_per_day * number_of_pigs

def monthly_feed (daily_feed : ℕ) (days_per_month : ℕ) : ℕ :=
  daily_feed * days_per_month

theorem total_monthly_feed :
  let pounds_per_pig_per_day := 15
  let number_of_pigs := 4
  let days_per_month := 30
  monthly_feed (daily_feed pounds_per_pig_per_day number_of_pigs) days_per_month = 1800 :=
by
  sorry

end total_monthly_feed_l24_24333


namespace geometric_sequence_second_term_l24_24349

theorem geometric_sequence_second_term (b : ℝ) (hb : b > 0) 
  (h1 : ∃ r : ℝ, 210 * r = b) 
  (h2 : ∃ r : ℝ, b * r = 135 / 56) : 
  b = 22.5 := 
sorry

end geometric_sequence_second_term_l24_24349


namespace Maria_drove_approximately_517_miles_l24_24226

noncomputable def carRentalMaria (daily_rate per_mile_charge discount_rate insurance_rate rental_duration total_invoice : ℝ) (discount_threshold : ℕ) : ℝ :=
  let total_rental_cost := rental_duration * daily_rate
  let discount := if rental_duration ≥ discount_threshold then discount_rate * total_rental_cost else 0
  let discounted_cost := total_rental_cost - discount
  let insurance_cost := rental_duration * insurance_rate
  let cost_without_mileage := discounted_cost + insurance_cost
  let mileage_cost := total_invoice - cost_without_mileage
  mileage_cost / per_mile_charge

noncomputable def approx_equal (a b : ℝ) (epsilon : ℝ := 1) : Prop :=
  abs (a - b) < epsilon

theorem Maria_drove_approximately_517_miles :
  approx_equal (carRentalMaria 35 0.09 0.10 5 4 192.50 3) 517 :=
by
  sorry

end Maria_drove_approximately_517_miles_l24_24226


namespace inequality_not_always_hold_l24_24704

theorem inequality_not_always_hold (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) : ¬(∀ a b, a^3 + b^3 ≥ 2 * a * b^2) :=
sorry

end inequality_not_always_hold_l24_24704


namespace multiple_of_75_with_36_divisors_l24_24009

theorem multiple_of_75_with_36_divisors (n : ℕ) (h1 : n % 75 = 0) (h2 : ∃ (a b c : ℕ), a ≥ 1 ∧ b ≥ 2 ∧ n = 3^a * 5^b * (2^c) ∧ (a+1)*(b+1)*(c+1) = 36) : n / 75 = 24 := 
sorry

end multiple_of_75_with_36_divisors_l24_24009


namespace total_cost_charlotte_l24_24728

noncomputable def regular_rate : ℝ := 40.00
noncomputable def discount_rate : ℝ := 0.25
noncomputable def number_of_people : ℕ := 5

theorem total_cost_charlotte :
  number_of_people * (regular_rate * (1 - discount_rate)) = 150.00 := by
  sorry

end total_cost_charlotte_l24_24728


namespace drowning_ratio_l24_24999

variable (total_sheep total_cows total_dogs drowned_sheep drowned_cows total_animals : ℕ)

-- Conditions provided
variable (initial_conditions : total_sheep = 20 ∧ total_cows = 10 ∧ total_dogs = 14)
variable (sheep_drowned_condition : drowned_sheep = 3)
variable (dogs_shore_condition : total_dogs = 14)
variable (total_made_it_shore : total_animals = 35)

theorem drowning_ratio (h1 : total_sheep = 20) (h2 : total_cows = 10) (h3 : total_dogs = 14) 
    (h4 : drowned_sheep = 3) (h5 : total_animals = 35) 
    : (drowned_cows = 2 * drowned_sheep) :=
by
  sorry

end drowning_ratio_l24_24999


namespace ratio_triangle_square_l24_24880

noncomputable def square_area (s : ℝ) : ℝ := s * s

noncomputable def triangle_PTU_area (s : ℝ) : ℝ := 1 / 2 * (s / 2) * (s / 2)

theorem ratio_triangle_square (s : ℝ) (h : s > 0) : 
  triangle_PTU_area s / square_area s = 1 / 8 := 
sorry

end ratio_triangle_square_l24_24880


namespace vanya_number_l24_24114

theorem vanya_number (m n : ℕ) (hm : m < 10) (hn : n < 10) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 
  10 * m + n = 81 :=
by sorry

end vanya_number_l24_24114


namespace value_of_b_l24_24626

-- Defining the number sum in circles and overlap
def circle_sum := 21
def num_circles := 5
def total_sum := 69

-- Overlapping numbers
def overlap_1 := 2
def overlap_2 := 8
def overlap_3 := 9
variable (b d : ℕ)

-- Circle equation containing d
def circle_with_d := d + 5 + 9

-- Prove b = 10 given the conditions
theorem value_of_b (h₁ : num_circles * circle_sum = 105)
    (h₂ : 105 - (overlap_1 + overlap_2 + overlap_3 + b + d) = total_sum)
    (h₃ : circle_with_d d = 21) : b = 10 :=
by sorry

end value_of_b_l24_24626


namespace more_orange_pages_read_l24_24047

-- Define the conditions
def purple_pages_per_book : Nat := 230
def orange_pages_per_book : Nat := 510
def purple_books_read : Nat := 5
def orange_books_read : Nat := 4

-- Calculate the total pages read from purple and orange books respectively
def total_purple_pages_read : Nat := purple_pages_per_book * purple_books_read
def total_orange_pages_read : Nat := orange_pages_per_book * orange_books_read

-- State the theorem to be proved
theorem more_orange_pages_read : total_orange_pages_read - total_purple_pages_read = 890 :=
by
  -- This is where the proof steps would go, but we'll leave it as sorry to indicate the proof is not provided
  sorry

end more_orange_pages_read_l24_24047


namespace value_of_a3_l24_24090

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem value_of_a3 (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 + a 4 = 20) :
  a 2 = 4 :=
sorry

end value_of_a3_l24_24090


namespace luke_games_l24_24569

theorem luke_games (F G : ℕ) (H1 : G = 2) (H2 : F + G - 2 = 2) : F = 2 := by
  sorry

end luke_games_l24_24569


namespace product_of_solutions_of_t_squared_eq_49_l24_24210

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l24_24210


namespace jake_comic_books_l24_24782

variables (J : ℕ)

def brother_comic_books := J + 15
def total_comic_books := J + brother_comic_books

theorem jake_comic_books : total_comic_books = 87 → J = 36 :=
by
  sorry

end jake_comic_books_l24_24782


namespace range_of_a_l24_24212

def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

theorem range_of_a (a : ℝ) : (-2 / 3 : ℝ) ≤ a ∧ a < 0 := sorry

end range_of_a_l24_24212


namespace Jack_can_form_rectangle_l24_24521

theorem Jack_can_form_rectangle : 
  ∃ (a b : ℕ), 
  3 * a = 2016 ∧ 
  4 * a = 2016 ∧ 
  4 * b = 2016 ∧ 
  3 * b = 2016 ∧ 
  (503 * 4 + 3 * 9 = 2021) ∧ 
  (2 * 3 = 4) :=
by 
  sorry

end Jack_can_form_rectangle_l24_24521


namespace third_candidate_votes_l24_24997

theorem third_candidate_votes (V A B W: ℕ) (hA : A = 2500) (hB : B = 15000) 
  (hW : W = (2 * V) / 3) (hV : V = W + A + B) : (V - (A + B)) = 35000 := by
  sorry

end third_candidate_votes_l24_24997


namespace part1_solution_part2_solution_l24_24179

-- Define the inequality for part (1)
def ineq_part1 (x : ℝ) : Prop := 1 - (4 / (x + 1)) < 0

-- Define the solution set P for part (1)
def P (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Prove that the solution set for the inequality is P
theorem part1_solution :
  ∀ (x : ℝ), ineq_part1 x ↔ P x :=
by
  -- proof omitted
  sorry

-- Define the inequality for part (2)
def ineq_part2 (x : ℝ) : Prop := abs (x + 2) < 3

-- Define the solution set Q for part (2)
def Q (x : ℝ) : Prop := -5 < x ∧ x < 1

-- Define P as depending on some parameter a
def P_param (a : ℝ) (x : ℝ) : Prop := -1 < x ∧ x < a

-- Prove the range of a given P ∪ Q = Q 
theorem part2_solution :
  ∀ a : ℝ, (∀ x : ℝ, (P_param a x ∨ Q x) ↔ Q x) → 
    (0 < a ∧ a ≤ 1) :=
by
  -- proof omitted
  sorry

end part1_solution_part2_solution_l24_24179


namespace ratio_of_ages_l24_24014

theorem ratio_of_ages (S F : Nat) 
  (h1 : F = 3 * S) 
  (h2 : (S + 6) + (F + 6) = 156) : 
  (F + 6) / (S + 6) = 19 / 7 := 
by 
  sorry

end ratio_of_ages_l24_24014


namespace compare_f_l24_24252

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem compare_f (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x1 + x2 = 0) : 
  f x1 < f x2 :=
by sorry

end compare_f_l24_24252


namespace average_of_middle_three_l24_24077

theorem average_of_middle_three
  (a b c d e : ℕ)
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_sum : a + b + c + d + e = 25)
  (h_max_diff : ∀ x y : ℕ, x + y = 24 → (e - a) ≥ (y - x)) :
  (b + c + d) / 3 = 3 :=
by
  sorry

end average_of_middle_three_l24_24077


namespace train_a_distance_at_meeting_l24_24093

noncomputable def train_a_speed : ℝ := 75 / 3
noncomputable def train_b_speed : ℝ := 75 / 2
noncomputable def relative_speed : ℝ := train_a_speed + train_b_speed
noncomputable def time_until_meet : ℝ := 75 / relative_speed
noncomputable def distance_traveled_by_train_a : ℝ := train_a_speed * time_until_meet

theorem train_a_distance_at_meeting : distance_traveled_by_train_a = 30 := by
  sorry

end train_a_distance_at_meeting_l24_24093


namespace remainder_N_div_5_is_1_l24_24380

-- The statement proving the remainder of N when divided by 5 is 1
theorem remainder_N_div_5_is_1 (N : ℕ) (h1 : N % 2 = 1) (h2 : N % 35 = 1) : N % 5 = 1 :=
sorry

end remainder_N_div_5_is_1_l24_24380


namespace gcd_binom_integer_l24_24208

theorem gcd_binom_integer (n m : ℕ) (hnm : n ≥ m) (hm : m ≥ 1) :
  (Nat.gcd m n) * Nat.choose n m % n = 0 := sorry

end gcd_binom_integer_l24_24208


namespace exists_sol_in_naturals_l24_24562

theorem exists_sol_in_naturals : ∃ (x y : ℕ), x^2 + y^2 = 61^3 := 
sorry

end exists_sol_in_naturals_l24_24562


namespace number_of_apples_remaining_l24_24931

def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples_before_giving_away : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℕ := total_apples_before_giving_away / 5
def apples_remaining : ℕ := total_apples_before_giving_away - apples_given_to_son

theorem number_of_apples_remaining : apples_remaining = 12 :=
by
  sorry

end number_of_apples_remaining_l24_24931


namespace sum_integer_solutions_l24_24474

theorem sum_integer_solutions (n : ℤ) (h1 : |n^2| < |n - 5|^2) (h2 : |n - 5|^2 < 16) : n = 2 := 
sorry

end sum_integer_solutions_l24_24474


namespace smallest_B_to_divisible_3_l24_24309

-- Define the problem
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define the digits in the integer
def digit_sum (B : ℕ) : ℕ := 8 + B + 4 + 6 + 3 + 5

-- Prove that the smallest digit B that makes 8B4,635 divisible by 3 is 1
theorem smallest_B_to_divisible_3 : ∃ B : ℕ, B ≥ 0 ∧ B ≤ 9 ∧ is_divisible_by_3 (digit_sum B) ∧ ∀ B' : ℕ, B' < B → ¬ is_divisible_by_3 (digit_sum B') ∧ B = 1 :=
sorry

end smallest_B_to_divisible_3_l24_24309


namespace max_sum_of_segments_l24_24104

theorem max_sum_of_segments (A B C D : ℝ × ℝ × ℝ)
    (h : (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D ≤ 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1)
      ∨ (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D > 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1))
    : dist A B + dist A C + dist A D + dist B C + dist B D + dist C D ≤ 5 + Real.sqrt 3 := sorry

end max_sum_of_segments_l24_24104


namespace project_hours_l24_24559

variable (K P M : ℕ)

theorem project_hours
  (h1 : P + K + M = 144)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 80 :=
sorry

end project_hours_l24_24559


namespace range_of_a_l24_24944

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * x

theorem range_of_a (a : ℝ) (h : f (2 * a) < f (a - 1)) : a < -1 :=
by
  -- Steps of the proof would be placed here, but we're skipping them for now
  sorry

end range_of_a_l24_24944


namespace area_of_white_square_l24_24023

theorem area_of_white_square
  (face_area : ℕ)
  (total_surface_area : ℕ)
  (blue_paint_area : ℕ)
  (faces : ℕ)
  (area_of_white_square : ℕ) :
  face_area = 12 * 12 →
  total_surface_area = 6 * face_area →
  blue_paint_area = 432 →
  faces = 6 →
  area_of_white_square = face_area - (blue_paint_area / faces) →
  area_of_white_square = 72 :=
by
  sorry

end area_of_white_square_l24_24023


namespace nested_sqrt_simplification_l24_24973

theorem nested_sqrt_simplification (y : ℝ) (hy : y ≥ 0) : 
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt y)) = y^(9/4) := 
sorry

end nested_sqrt_simplification_l24_24973


namespace triangle_area_solution_l24_24977

noncomputable def triangle_area_problem 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  : ℝ := (1 / 2) * a * c * Real.sin B

theorem triangle_area_solution 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  (ha : a = 12)
  (hb : b = 6 * Real.sin (π / 3))
  (hA : A = π / 2)
  (hB : B = π / 3)
  (hC : C = π / 6) 
  : triangle_area_problem a b c A B C h1 h2 h3 = 18 * Real.sqrt 3 := by
  sorry

end triangle_area_solution_l24_24977


namespace find_least_positive_x_l24_24583

theorem find_least_positive_x :
  ∃ x : ℕ, x + 5419 ≡ 3789 [MOD 15] ∧ x = 5 :=
by
  use 5
  constructor
  · sorry
  · rfl

end find_least_positive_x_l24_24583


namespace Turner_Catapult_rides_l24_24286

def tickets_needed (rollercoaster_rides Ferris_wheel_rides Catapult_rides : ℕ) : ℕ :=
  4 * rollercoaster_rides + 1 * Ferris_wheel_rides + 4 * Catapult_rides

theorem Turner_Catapult_rides :
  ∀ (x : ℕ), tickets_needed 3 1 x = 21 → x = 2 := by
  intros x h
  sorry

end Turner_Catapult_rides_l24_24286


namespace algebra_expression_value_l24_24390

theorem algebra_expression_value (a b : ℝ) 
  (h₁ : a - b = 5) 
  (h₂ : a * b = -1) : 
  (2 * a + 3 * b - 2 * a * b) 
  - (a + 4 * b + a * b) 
  - (3 * a * b + 2 * b - 2 * a) = 21 := 
by
  sorry

end algebra_expression_value_l24_24390


namespace find_xy_l24_24026

theorem find_xy (x y : ℝ) (h : (x^2 + 6 * x + 12) * (5 * y^2 + 2 * y + 1) = 12 / 5) : 
    x * y = 3 / 5 :=
sorry

end find_xy_l24_24026


namespace random_event_proof_l24_24088

-- Definitions based on conditions
def event1 := "Tossing a coin twice in a row, and both times it lands heads up."
def event2 := "Opposite charges attract each other."
def event3 := "Water freezes at 1℃ under standard atmospheric pressure."

def is_random_event (event: String) : Prop :=
  event = event1 ∨ event = event2 ∨ event = event3 → event = event1

theorem random_event_proof : is_random_event event1 ∧ ¬is_random_event event2 ∧ ¬is_random_event event3 :=
by
  -- Proof goes here
  sorry

end random_event_proof_l24_24088


namespace diff_g_eq_l24_24312

def g (n : ℤ) : ℚ := (1/6) * n * (n+1) * (n+3)

theorem diff_g_eq :
  ∀ (r : ℤ), g r - g (r - 1) = (3/2) * r^2 + (5/2) * r :=
by
  intro r
  sorry

end diff_g_eq_l24_24312


namespace min_value_of_reciprocals_l24_24216

open Real

theorem min_value_of_reciprocals (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) :
  (1 / a) + (1 / (b + 1)) ≥ 2 :=
sorry

end min_value_of_reciprocals_l24_24216


namespace max_square_side_length_l24_24809

-- Given: distances between consecutive lines in L and P
def distances_L : List ℕ := [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6, 2]
def distances_P : List ℕ := [3, 1, 2, 6, 3, 1, 2, 6, 3, 1, 2, 6, 3, 1]

-- Theorem: Maximum possible side length of a square with sides on lines L and P
theorem max_square_side_length : ∀ (L P : List ℕ), L = distances_L → P = distances_P → ∃ s, s = 40 :=
by
  intros L P hL hP
  sorry

end max_square_side_length_l24_24809


namespace simplify_and_evaluate_expression_l24_24444

variable (a b : ℤ)

theorem simplify_and_evaluate_expression (h1 : a = 1) (h2 : b = -1) :
  (3 * a^2 * b - 2 * (a * b - (3/2) * a^2 * b) + a * b - 2 * a^2 * b) = -3 := by
  sorry

end simplify_and_evaluate_expression_l24_24444


namespace range_of_x_l24_24560

variable (x : ℝ)

def p := x^2 - 4 * x + 3 < 0
def q := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

theorem range_of_x : ¬ (p x ∧ q x) ∧ (p x ∨ q x) → (1 < x ∧ x ≤ 2) ∨ x = 3 :=
by 
  sorry

end range_of_x_l24_24560


namespace range_of_a_l24_24135

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x < 2 → (x + a < 0))) → (a ≤ -2) :=
sorry

end range_of_a_l24_24135


namespace necessary_not_sufficient_l24_24230

theorem necessary_not_sufficient (x : ℝ) : (x > 5) → (x > 2) ∧ ¬((x > 2) → (x > 5)) :=
by
  sorry

end necessary_not_sufficient_l24_24230


namespace least_number_divisible_by_13_l24_24006

theorem least_number_divisible_by_13 (n : ℕ) :
  (∀ m : ℕ, 2 ≤ m ∧ m ≤ 7 → n % m = 2) ∧ (n % 13 = 0) → n = 1262 :=
by sorry

end least_number_divisible_by_13_l24_24006


namespace shem_wage_multiple_kem_l24_24834

-- Define the hourly wages and conditions
def kem_hourly_wage : ℝ := 4
def shem_daily_wage : ℝ := 80
def shem_workday_hours : ℝ := 8

-- Prove the multiple of Shem's hourly wage compared to Kem's hourly wage
theorem shem_wage_multiple_kem : (shem_daily_wage / shem_workday_hours) / kem_hourly_wage = 2.5 := by
  sorry

end shem_wage_multiple_kem_l24_24834


namespace solution_set_inequality_l24_24192

variable {x : ℝ}
variable {a b : ℝ}

theorem solution_set_inequality (h₁ : ∀ x : ℝ, (ax^2 + bx - 1 > 0) ↔ (-1/2 < x ∧ x < -1/3)) :
  ∀ x : ℝ, (x^2 - bx - a ≥ 0) ↔ (x ≤ -3 ∨ x ≥ -2) := 
sorry

end solution_set_inequality_l24_24192


namespace range_of_m_l24_24321

noncomputable def f (x m : ℝ) : ℝ :=
  x^2 - 2 * m * x + m + 2

theorem range_of_m
  (m : ℝ)
  (h1 : ∃ a b : ℝ, f a m = 0 ∧ f b m = 0 ∧ a ≠ b)
  (h2 : ∀ x : ℝ, x ≥ 1 → 2*x - 2*m ≥ 0) :
  m < -1 :=
sorry

end range_of_m_l24_24321


namespace fruit_seller_l24_24838

theorem fruit_seller (A P : ℝ) (h1 : A = 700) (h2 : A * (100 - P) / 100 = 420) : P = 40 :=
sorry

end fruit_seller_l24_24838


namespace max_value_npk_l24_24378

theorem max_value_npk : 
  ∃ (M K : ℕ), 
    (M ≠ K) ∧ (1 ≤ M ∧ M ≤ 9) ∧ (1 ≤ K ∧ K ≤ 9) ∧ 
    (NPK = 11 * M * K ∧ 100 ≤ NPK ∧ NPK < 1000 ∧ NPK = 891) :=
sorry

end max_value_npk_l24_24378


namespace points_three_units_away_from_neg_two_on_number_line_l24_24423

theorem points_three_units_away_from_neg_two_on_number_line :
  ∃! p1 p2 : ℤ, |p1 + 2| = 3 ∧ |p2 + 2| = 3 ∧ p1 ≠ p2 ∧ (p1 = -5 ∨ p2 = -5) ∧ (p1 = 1 ∨ p2 = 1) :=
sorry

end points_three_units_away_from_neg_two_on_number_line_l24_24423


namespace cost_price_computer_table_l24_24907

theorem cost_price_computer_table (CP SP : ℝ) (h1 : SP = 1.15 * CP) (h2 : SP = 6400) : CP = 5565.22 :=
by sorry

end cost_price_computer_table_l24_24907


namespace percentage_passed_both_subjects_l24_24197

def failed_H : ℝ := 0.35
def failed_E : ℝ := 0.45
def failed_HE : ℝ := 0.20

theorem percentage_passed_both_subjects :
  (100 - (failed_H * 100 + failed_E * 100 - failed_HE * 100)) = 40 := 
by
  sorry

end percentage_passed_both_subjects_l24_24197


namespace no_valid_x_l24_24368

-- Definitions based on given conditions
variables {m n x : ℝ}
variables (hm : m > 0) (hn : n < 0)

-- Theorem statement
theorem no_valid_x (hm : m > 0) (hn : n < 0) :
  ¬ ∃ x, (x - m)^2 - (x - n)^2 = (m - n)^2 :=
by
  sorry

end no_valid_x_l24_24368


namespace apple_tree_fruits_production_l24_24375

def apple_production (first_season : ℕ) (second_season : ℕ) (third_season : ℕ): ℕ :=
  first_season + second_season + third_season

theorem apple_tree_fruits_production :
  let first_season := 200
  let second_season := 160    -- 200 - 20% of 200
  let third_season := 320     -- 2 * 160
  apple_production first_season second_season third_season = 680 := by
  -- This is where the proof would go
  sorry

end apple_tree_fruits_production_l24_24375


namespace unique_digit_solution_l24_24996

-- Define the constraints as Lean predicates.
def sum_top_less_7 (A B C D E : ℕ) := A + B = (C + D + E) / 7
def sum_left_less_5 (A B C D E : ℕ) := A + C = (B + D + E) / 5

-- The main theorem statement asserting there is a unique solution.
theorem unique_digit_solution :
  ∃! (A B C D E : ℕ), 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 
  sum_top_less_7 A B C D E ∧ sum_left_less_5 A B C D E ∧
  (A, B, C, D, E) = (1, 2, 3, 4, 6) := sorry

end unique_digit_solution_l24_24996


namespace remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l24_24459

structure ArtCollection where
  medieval : ℕ
  renaissance : ℕ
  modern : ℕ

def AliciaArtCollection : ArtCollection := {
  medieval := 70,
  renaissance := 120,
  modern := 150
}

def donationPercentages : ArtCollection := {
  medieval := 65,
  renaissance := 30,
  modern := 45
}

def remainingArtPieces (initial : ℕ) (percent : ℕ) : ℕ :=
  initial - ((percent * initial) / 100)

theorem remaining_medieval_art_pieces :
  remainingArtPieces AliciaArtCollection.medieval donationPercentages.medieval = 25 := by
  sorry

theorem remaining_renaissance_art_pieces :
  remainingArtPieces AliciaArtCollection.renaissance donationPercentages.renaissance = 84 := by
  sorry

theorem remaining_modern_art_pieces :
  remainingArtPieces AliciaArtCollection.modern donationPercentages.modern = 83 := by
  sorry

end remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l24_24459


namespace sum_of_interior_diagonals_of_box_l24_24702

theorem sum_of_interior_diagonals_of_box (a b c : ℝ) 
  (h_edges : 4 * (a + b + c) = 60)
  (h_surface_area : 2 * (a * b + b * c + c * a) = 150) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 3 := 
by
  sorry

end sum_of_interior_diagonals_of_box_l24_24702


namespace probability_of_picking_letter_in_mathematics_l24_24535

-- Definitions and conditions
def total_letters : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Theorem to be proven
theorem probability_of_picking_letter_in_mathematics :
  probability unique_letters_in_mathematics total_letters = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l24_24535


namespace annalise_total_cost_l24_24221

/-- 
Given conditions:
- 25 boxes of tissues.
- Each box contains 18 packs.
- Each pack contains 150 tissues.
- Each tissue costs $0.06.
- A 10% discount on the total price of the packs in each box.

Prove:
The total amount of money Annalise spent is $3645.
-/
theorem annalise_total_cost :
  let boxes := 25
  let packs_per_box := 18
  let tissues_per_pack := 150
  let cost_per_tissue := 0.06
  let discount_rate := 0.10
  let price_per_box := (packs_per_box * tissues_per_pack * cost_per_tissue)
  let discount_per_box := discount_rate * price_per_box
  let discounted_price_per_box := price_per_box - discount_per_box
  let total_cost := discounted_price_per_box * boxes
  total_cost = 3645 :=
by
  sorry

end annalise_total_cost_l24_24221


namespace partial_fraction_sum_zero_l24_24982

theorem partial_fraction_sum_zero (A B C D E F : ℚ) :
  (∀ x : ℚ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 → x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l24_24982


namespace david_bike_distance_l24_24162

noncomputable def david_time_hours : ℝ := 2 + 1 / 3
noncomputable def david_speed_mph : ℝ := 6.998571428571427
noncomputable def david_distance : ℝ := 16.33

theorem david_bike_distance :
  david_speed_mph * david_time_hours = david_distance :=
by
  sorry

end david_bike_distance_l24_24162


namespace inequality_true_l24_24898

theorem inequality_true (a b : ℝ) (h : a^2 + b^2 > 1) : |a| + |b| > 1 :=
sorry

end inequality_true_l24_24898


namespace race_distance_l24_24822

def race_distance_problem (V_A V_B T : ℝ) : Prop :=
  V_A * T = 218.75 ∧
  V_B * T = 193.75 ∧
  V_B * (T + 10) = 218.75 ∧
  T = 77.5

theorem race_distance (D : ℝ) (V_A V_B T : ℝ) 
  (h1 : V_A * T = D) 
  (h2 : V_B * T = D - 25) 
  (h3 : V_B * (T + 10) = D) 
  (h4 : V_A * T = 218.75) 
  (h5 : T = 77.5) 
  : D = 218.75 := 
by 
  sorry

end race_distance_l24_24822


namespace intersection_M_N_l24_24404

def M : Set ℝ := { x | x^2 - x - 2 = 0 }
def N : Set ℝ := { -1, 0 }

theorem intersection_M_N : M ∩ N = {-1} :=
by
  sorry

end intersection_M_N_l24_24404


namespace erica_earnings_l24_24472

def price_per_kg : ℝ := 20
def past_catch : ℝ := 80
def catch_today := 2 * past_catch
def total_catch := past_catch + catch_today
def total_earnings := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end erica_earnings_l24_24472


namespace smallest_value_of_a_for_polynomial_l24_24739

theorem smallest_value_of_a_for_polynomial (r1 r2 r3 : ℕ) (h_prod : r1 * r2 * r3 = 30030) :
  (r1 + r2 + r3 = 54) ∧ (r1 * r2 * r3 = 30030) → 
  (∀ a, a = r1 + r2 + r3 → a ≥ 54) :=
by
  sorry

end smallest_value_of_a_for_polynomial_l24_24739


namespace ducks_problem_l24_24330

theorem ducks_problem :
  ∃ (adelaide ephraim kolton : ℕ),
    adelaide = 30 ∧
    adelaide = 2 * ephraim ∧
    ephraim + 45 = kolton ∧
    (adelaide + ephraim + kolton) % 9 = 0 ∧
    1 ≤ adelaide ∧
    1 ≤ ephraim ∧
    1 ≤ kolton ∧
    adelaide + ephraim + kolton = 108 ∧
    (adelaide + ephraim + kolton) / 3 = 36 :=
by
  sorry

end ducks_problem_l24_24330


namespace butterfat_in_final_mixture_l24_24094

noncomputable def final_butterfat_percentage (gallons_of_35_percentage : ℕ) 
                                             (percentage_of_35_butterfat : ℝ) 
                                             (total_gallons : ℕ)
                                             (percentage_of_10_butterfat : ℝ) : ℝ :=
  let gallons_of_10 := total_gallons - gallons_of_35_percentage
  let butterfat_35 := gallons_of_35_percentage * percentage_of_35_butterfat
  let butterfat_10 := gallons_of_10 * percentage_of_10_butterfat
  let total_butterfat := butterfat_35 + butterfat_10
  (total_butterfat / total_gallons) * 100

theorem butterfat_in_final_mixture : 
  final_butterfat_percentage 8 0.35 12 0.10 = 26.67 :=
sorry

end butterfat_in_final_mixture_l24_24094


namespace distance_between_A_and_B_l24_24240

-- Define speeds, times, and distances as real numbers
def speed_A_to_B := 42.5
def time_travelled := 1.5
def remaining_to_midpoint := 26.0

-- Define the total distance between A and B as a variable
def distance_A_to_B : ℝ := 179.5

-- Prove that the distance between locations A and B is 179.5 kilometers given the conditions
theorem distance_between_A_and_B : (42.5 * 1.5 + 26) * 2 = 179.5 :=
by 
  sorry

end distance_between_A_and_B_l24_24240


namespace intersection_A_B_l24_24755

def A (x : ℝ) : Prop := 0 < x ∧ x < 2
def B (x : ℝ) : Prop := -1 < x ∧ x < 1
def C (x : ℝ) : Prop := 0 < x ∧ x < 1

theorem intersection_A_B : ∀ x, A x ∧ B x ↔ C x := by
  sorry

end intersection_A_B_l24_24755


namespace range_of_m_l24_24308

theorem range_of_m (m : ℝ) : (∀ x > 1, 2*x + m + 8/(x-1) > 0) → m > -10 := 
by
  -- The formal proof will be completed here.
  sorry

end range_of_m_l24_24308


namespace sequence_general_term_l24_24513

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 3 * a n - 2 * n ^ 2 + 4 * n + 4) :
  ∀ n, a n = 3^n + n^2 - n - 2 :=
sorry

end sequence_general_term_l24_24513


namespace smallest_n_l24_24360

theorem smallest_n (n : ℕ) : 
  (25 * n = (Nat.lcm 10 (Nat.lcm 16 18)) → n = 29) :=
by sorry

end smallest_n_l24_24360


namespace multiply_3_6_and_0_3_l24_24706

theorem multiply_3_6_and_0_3 : 3.6 * 0.3 = 1.08 :=
by
  sorry

end multiply_3_6_and_0_3_l24_24706


namespace possible_value_of_a_eq_neg1_l24_24933

theorem possible_value_of_a_eq_neg1 (a : ℝ) : (-6 * a ^ 2 = 3 * (4 * a + 2)) → (a = -1) :=
by
  intro h
  have H : a^2 + 2*a + 1 = 0
  · sorry
  show a = -1
  · sorry

end possible_value_of_a_eq_neg1_l24_24933


namespace parallel_lines_condition_l24_24655

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y: ℝ, (x + a * y + 6 = 0) ↔ ((a - 2) * x + 3 * y + 2 * a = 0)) ↔ a = -1 :=
by
  sorry

end parallel_lines_condition_l24_24655


namespace double_root_divisors_l24_24985

theorem double_root_divisors (b3 b2 b1 s : ℤ) (h : 0 = (s^2) • (x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 50)) : 
  s = -5 ∨ s = -1 ∨ s = 1 ∨ s = 5 :=
by
  sorry

end double_root_divisors_l24_24985


namespace solid_is_frustum_l24_24652

-- Definitions for views
def front_view_is_isosceles_trapezoid (S : Type) : Prop := sorry
def side_view_is_isosceles_trapezoid (S : Type) : Prop := sorry
def top_view_is_concentric_circles (S : Type) : Prop := sorry

-- Define the target solid as a frustum
def is_frustum (S : Type) : Prop := sorry

-- The theorem statement
theorem solid_is_frustum
  (S : Type) 
  (h1 : front_view_is_isosceles_trapezoid S)
  (h2 : side_view_is_isosceles_trapezoid S)
  (h3 : top_view_is_concentric_circles S) :
  is_frustum S :=
sorry

end solid_is_frustum_l24_24652


namespace square_area_given_equal_perimeters_l24_24519

theorem square_area_given_equal_perimeters 
  (a b c : ℝ) (a_eq : a = 7.5) (b_eq : b = 9.5) (c_eq : c = 12) 
  (sq_perimeter_eq_tri : 4 * s = a + b + c) : 
  s^2 = 52.5625 :=
by
  sorry

end square_area_given_equal_perimeters_l24_24519


namespace find_percentage_l24_24405

variable (P : ℝ)

/-- A number P% that satisfies the condition is 65. -/
theorem find_percentage (h : ((P / 100) * 40 = ((5 / 100) * 60) + 23)) : P = 65 :=
sorry

end find_percentage_l24_24405


namespace cube_vertex_adjacency_l24_24485

noncomputable def beautiful_face (a b c d : ℕ) : Prop :=
  a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c

theorem cube_vertex_adjacency :
  ∀ (v1 v2 v3 v4 v5 v6 v7 v8 : ℕ), 
  v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ v1 ≠ v6 ∧ v1 ≠ v7 ∧ v1 ≠ v8 ∧
  v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ v2 ≠ v6 ∧ v2 ≠ v7 ∧ v2 ≠ v8 ∧
  v3 ≠ v4 ∧ v3 ≠ v5 ∧ v3 ≠ v6 ∧ v3 ≠ v7 ∧ v3 ≠ v8 ∧
  v4 ≠ v5 ∧ v4 ≠ v6 ∧ v4 ≠ v7 ∧ v4 ≠ v8 ∧
  v5 ≠ v6 ∧ v5 ≠ v7 ∧ v5 ≠ v8 ∧
  v6 ≠ v7 ∧ v6 ≠ v8 ∧
  v7 ≠ v8 ∧
  beautiful_face v1 v2 v3 v4 ∧ beautiful_face v5 v6 v7 v8 ∧
  beautiful_face v1 v3 v5 v7 ∧ beautiful_face v2 v4 v6 v8 ∧
  beautiful_face v1 v2 v5 v6 ∧ beautiful_face v3 v4 v7 v8 →
  (v6 = 6 → (v1 = 2 ∧ v2 = 3 ∧ v3 = 5) ∨ 
   (v1 = 3 ∧ v2 = 5 ∧ v3 = 7) ∨ 
   (v1 = 2 ∧ v2 = 3 ∧ v3 = 7)) :=
sorry

end cube_vertex_adjacency_l24_24485


namespace domino_swap_correct_multiplication_l24_24840

theorem domino_swap_correct_multiplication :
  ∃ (a b c d e f : ℕ), 
    a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 3 ∧ e = 12 ∧ f = 3 ∧ 
    a * b = 6 ∧ c * d = 3 ∧ e * f = 36 ∧
    ∃ (x y : ℕ), x * y = 36 := sorry

end domino_swap_correct_multiplication_l24_24840


namespace math_problem_l24_24057

-- Arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 2 = 8 ∧ a 3 + a 5 = 4 * a 2

-- General term of the arithmetic sequence {a_n}
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = 4 * n

-- Geometric sequence {b_n}
def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 4 = a 1 ∧ b 6 = a 4

-- The sum S_n of the first n terms of the sequence {b_n - a_n}
def sum_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (2 ^ (n - 1) - 1 / 2 - 2 * n ^ 2 - 2 * n)

-- Full proof statement
theorem math_problem (a : ℕ → ℕ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  general_term a →
  ∀ a_n : ℕ → ℝ, a_n 1 = 4 ∧ a_n 4 = 16 →
  geometric_sequence b a_n →
  sum_sequence b a_n S :=
by
  intros h_arith_seq h_gen_term h_a_n h_geom_seq
  sorry

end math_problem_l24_24057


namespace round_2748397_542_nearest_integer_l24_24476

theorem round_2748397_542_nearest_integer :
  let n := 2748397.542
  let int_part := 2748397
  let decimal_part := 0.542
  (n.round = 2748398) :=
by
  sorry

end round_2748397_542_nearest_integer_l24_24476


namespace total_sentence_l24_24534

theorem total_sentence (base_rate : ℝ) (value_stolen : ℝ) (third_offense_increase : ℝ) (additional_years : ℕ) : 
  base_rate = 1 / 5000 → 
  value_stolen = 40000 → 
  third_offense_increase = 0.25 → 
  additional_years = 2 →
  (value_stolen * base_rate * (1 + third_offense_increase) + additional_years) = 12 := 
by
  intros
  sorry

end total_sentence_l24_24534


namespace units_digit_G_n_for_n_eq_3_l24_24255

def G (n : ℕ) : ℕ := 2 ^ 2 ^ 2 ^ n + 1

theorem units_digit_G_n_for_n_eq_3 : (G 3) % 10 = 7 := 
by 
  sorry

end units_digit_G_n_for_n_eq_3_l24_24255


namespace dave_age_l24_24934

theorem dave_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end dave_age_l24_24934


namespace geometric_sequence_proof_l24_24672

theorem geometric_sequence_proof (a : ℕ → ℝ) (q : ℝ) (h1 : q > 1) (h2 : a 1 > 0)
    (h3 : a 2 * a 4 + a 4 * a 10 - a 4 * a 6 - (a 5)^2 = 9) :
  a 3 - a 7 = -3 :=
by sorry

end geometric_sequence_proof_l24_24672


namespace weierstrass_limit_l24_24826

theorem weierstrass_limit (a_n : ℕ → ℝ) (M : ℝ) :
  (∀ n m, n ≤ m → a_n n ≤ a_n m) → 
  (∀ n, a_n n ≤ M ) → 
  ∃ c, ∀ ε > 0, ∃ N, ∀ n ≥ N, |a_n n - c| < ε :=
by
  sorry

end weierstrass_limit_l24_24826


namespace simplify_expression_l24_24678

variable {x y : ℝ}

theorem simplify_expression : (x^5 * x^3 * y^2 * y^4) = (x^8 * y^6) := by
  sorry

end simplify_expression_l24_24678


namespace minimum_value_x_l24_24134

theorem minimum_value_x (a b x : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
    (H : 4 * a + b * (1 - a) = 0) 
    (Hinequality : ∀ (a b : ℝ), a > 0 → b > 0 → 
        (4 * a + b * (1 - a) = 0 → 
        (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))) : 
    x >= 1 := 
sorry

end minimum_value_x_l24_24134


namespace cos_product_value_l24_24170

open Real

theorem cos_product_value (α : ℝ) (h : sin α = 1 / 3) : 
  cos (π / 4 + α) * cos (π / 4 - α) = 7 / 18 :=
by
  sorry

end cos_product_value_l24_24170


namespace right_triangle_sum_of_legs_l24_24906

theorem right_triangle_sum_of_legs (a b : ℝ) (h₁ : a^2 + b^2 = 2500) (h₂ : (1 / 2) * a * b = 600) : a + b = 70 :=
sorry

end right_triangle_sum_of_legs_l24_24906


namespace line_bisects_circle_perpendicular_l24_24130

theorem line_bisects_circle_perpendicular :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, x^2 + y^2 + x - 2*y + 1 = 0 → l x = y)
               ∧ (∀ x y : ℝ, x + 2*y + 3 = 0 → x ∈ { x | ∃ k:ℝ, y = -1/2 * k + l x})
               ∧ (∀ x y : ℝ, l x = 2 * x - 2)) :=
sorry

end line_bisects_circle_perpendicular_l24_24130


namespace sequence_an_l24_24248

theorem sequence_an (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * (a n - 1)) : a 2 = 4 := 
by
  sorry

end sequence_an_l24_24248


namespace range_of_a_l24_24673

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x - 1/2

noncomputable def g (x a : ℝ) : ℝ := x^2 + Real.log (x + a)

theorem range_of_a : 
  (∀ x ∈ Set.Iio 0, ∃ y, f x = g y a ∧ y = -x) →
  a < Real.sqrt (Real.exp 1) :=
  sorry

end range_of_a_l24_24673


namespace num_C_atoms_in_compound_l24_24581

def num_H_atoms := 6
def num_O_atoms := 1
def molecular_weight := 58
def atomic_weight_C := 12
def atomic_weight_H := 1
def atomic_weight_O := 16

theorem num_C_atoms_in_compound : 
  ∃ (num_C_atoms : ℕ), 
    molecular_weight = (num_C_atoms * atomic_weight_C) + (num_H_atoms * atomic_weight_H) + (num_O_atoms * atomic_weight_O) ∧ 
    num_C_atoms = 3 :=
by
  -- To be proven
  sorry

end num_C_atoms_in_compound_l24_24581


namespace correct_product_l24_24498

theorem correct_product (a b : ℚ) (calc_incorrect : a = 52 ∧ b = 735)
                        (incorrect_product : a * b = 38220) :
  (0.52 * 7.35 = 3.822) :=
by
  sorry

end correct_product_l24_24498


namespace bus_students_after_fifth_stop_l24_24482

theorem bus_students_after_fifth_stop :
  let initial := 72
  let firstStop := (2 / 3 : ℚ) * initial
  let secondStop := (2 / 3 : ℚ) * firstStop
  let thirdStop := (2 / 3 : ℚ) * secondStop
  let fourthStop := (2 / 3 : ℚ) * thirdStop
  let fifthStop := fourthStop + 12
  fifthStop = 236 / 9 :=
by
  sorry

end bus_students_after_fifth_stop_l24_24482


namespace hyperbola_asymptotes_l24_24462

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 16 * x^2 - 9 * y^2 = -144 → (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intros x y h1
  sorry

end hyperbola_asymptotes_l24_24462


namespace baguettes_leftover_l24_24811

-- Definitions based on conditions
def batches_per_day := 3
def baguettes_per_batch := 48
def sold_after_first_batch := 37
def sold_after_second_batch := 52
def sold_after_third_batch := 49

-- Prove the question equals the answer
theorem baguettes_leftover : 
  (batches_per_day * baguettes_per_batch - (sold_after_first_batch + sold_after_second_batch + sold_after_third_batch)) = 6 := 
by 
  sorry

end baguettes_leftover_l24_24811


namespace total_shirts_made_l24_24700

def shirtsPerMinute := 6
def minutesWorkedYesterday := 12
def shirtsMadeToday := 14

theorem total_shirts_made : shirtsPerMinute * minutesWorkedYesterday + shirtsMadeToday = 86 := by
  sorry

end total_shirts_made_l24_24700


namespace point_not_on_line_pq_neg_l24_24576

theorem point_not_on_line_pq_neg (p q : ℝ) (h : p * q < 0) : ¬ (21 * p + q = -101) := 
by sorry

end point_not_on_line_pq_neg_l24_24576


namespace sum_f_neg12_to_13_l24_24021

noncomputable def f (x : ℝ) := 1 / (3^x + Real.sqrt 3)

theorem sum_f_neg12_to_13 : 
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6)
  + f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0
  + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10
  + f 11 + f 12 + f 13) = (13 * Real.sqrt 3 / 3) :=
sorry

end sum_f_neg12_to_13_l24_24021


namespace solve_for_x_l24_24502

theorem solve_for_x :
  ∃ x : ℝ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 4.5 :=
by
  use 4.5
  sorry

end solve_for_x_l24_24502


namespace probability_green_dinosaur_or_blue_robot_l24_24674

theorem probability_green_dinosaur_or_blue_robot (t: ℕ) (blue_dinosaurs green_robots blue_robots: ℕ) 
(h1: blue_dinosaurs = 16) (h2: green_robots = 14) (h3: blue_robots = 36) (h4: t = 93):
  t = 93 → (blue_dinosaurs = 16) → (green_robots = 14) → (blue_robots = 36) → 
  (∃ green_dinosaurs: ℕ, t = blue_dinosaurs + green_robots + blue_robots + green_dinosaurs ∧ 
    (∃ k: ℕ, k = (green_dinosaurs + blue_robots) / (t / 31) ∧ k = 21 / 31)) := sorry

end probability_green_dinosaur_or_blue_robot_l24_24674


namespace monotone_f_solve_inequality_range_of_a_l24_24648

noncomputable def e := Real.exp 1
noncomputable def f (x : ℝ) : ℝ := e^x + 1/(e^x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.log ((3 - a) * (f x - 1/e^x) + 1) - Real.log (3 * a) - 2 * x

-- Part 1: Monotonicity of f(x)
theorem monotone_f : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by sorry

-- Part 2: Solving the inequality f(2x) ≥ f(x + 1)
theorem solve_inequality : ∀ x : ℝ, f (2 * x) ≥ f (x + 1) ↔ x ≥ 1 ∨ x ≤ -1 / 3 :=
by sorry

-- Part 3: Finding the range of a
theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x → g x a ≤ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
by sorry

end monotone_f_solve_inequality_range_of_a_l24_24648


namespace find_possible_values_of_y_l24_24910

theorem find_possible_values_of_y (x : ℝ) (h : x^2 + 9 * (3 * x / (x - 3))^2 = 90) :
  y = (x - 3)^3 * (x + 2) / (2 * x - 4) → y = 28 / 3 ∨ y = 169 :=
by
  sorry

end find_possible_values_of_y_l24_24910


namespace real_and_imaginary_parts_of_z_l24_24864

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i^2 + i

-- State the theorem
theorem real_and_imaginary_parts_of_z :
  z.re = -1 ∧ z.im = 1 :=
by
  -- Provide the proof or placeholder
  sorry

end real_and_imaginary_parts_of_z_l24_24864


namespace vector_identity_l24_24992

def vec_a : ℝ × ℝ := (2, 2)
def vec_b : ℝ × ℝ := (-1, 3)

theorem vector_identity : 2 • vec_a - vec_b = (5, 1) := by
  sorry

end vector_identity_l24_24992


namespace central_angle_of_sector_l24_24407

noncomputable def central_angle (l S r : ℝ) : ℝ :=
  2 * S / r^2

theorem central_angle_of_sector (r : ℝ) (h₁ : 4 * r / 2 = 4) (h₂ : r = 2) : central_angle 4 4 r = 2 :=
by
  sorry

end central_angle_of_sector_l24_24407


namespace carpets_triple_overlap_area_l24_24490

theorem carpets_triple_overlap_area {W H : ℕ} (hW : W = 10) (hH : H = 10) 
    {w1 h1 w2 h2 w3 h3 : ℕ} 
    (h1_w1 : w1 = 6) (h1_h1 : h1 = 8)
    (h2_w2 : w2 = 6) (h2_h2 : h2 = 6)
    (h3_w3 : w3 = 5) (h3_h3 : h3 = 7) :
    ∃ (area : ℕ), area = 6 := by
  sorry

end carpets_triple_overlap_area_l24_24490


namespace train_crosses_bridge_in_approximately_21_seconds_l24_24847

noncomputable def length_of_train : ℝ := 110  -- meters
noncomputable def speed_of_train_kmph : ℝ := 60  -- kilometers per hour
noncomputable def length_of_bridge : ℝ := 240  -- meters

noncomputable def speed_of_train_mps : ℝ := (speed_of_train_kmph * 1000) / 3600

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge

noncomputable def required_time : ℝ := total_distance / speed_of_train_mps

theorem train_crosses_bridge_in_approximately_21_seconds :
  |required_time - 21| < 1 :=
by sorry

end train_crosses_bridge_in_approximately_21_seconds_l24_24847


namespace value_of_expression_l24_24669

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by
  sorry

end value_of_expression_l24_24669


namespace austin_needs_six_weeks_l24_24372

theorem austin_needs_six_weeks
  (work_rate: ℕ) (hours_monday hours_wednesday hours_friday: ℕ) (bicycle_cost: ℕ) 
  (weekly_hours: ℕ := hours_monday + hours_wednesday + hours_friday) 
  (weekly_earnings: ℕ := weekly_hours * work_rate) 
  (weeks_needed: ℕ := bicycle_cost / weekly_earnings):
  work_rate = 5 ∧ hours_monday = 2 ∧ hours_wednesday = 1 ∧ hours_friday = 3 ∧ bicycle_cost = 180 ∧ weeks_needed = 6 :=
by {
  sorry
}

end austin_needs_six_weeks_l24_24372


namespace total_kids_on_soccer_field_l24_24301

theorem total_kids_on_soccer_field (initial_kids : ℕ) (joining_kids : ℕ) (total_kids : ℕ)
  (h₁ : initial_kids = 14)
  (h₂ : joining_kids = 22)
  (h₃ : total_kids = initial_kids + joining_kids) :
  total_kids = 36 :=
by
  sorry

end total_kids_on_soccer_field_l24_24301


namespace solution_set_condition_l24_24614

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, x * (x - a + 1) > a ↔ (x < -1 ∨ x > a)) → a > -1 :=
sorry

end solution_set_condition_l24_24614


namespace find_smaller_number_l24_24287

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end find_smaller_number_l24_24287


namespace other_number_is_36_l24_24932

theorem other_number_is_36 (hcf lcm given_number other_number : ℕ) 
  (hcf_val : hcf = 16) (lcm_val : lcm = 396) (given_number_val : given_number = 176) 
  (relation : hcf * lcm = given_number * other_number) : 
  other_number = 36 := 
by 
  sorry

end other_number_is_36_l24_24932


namespace sum_of_three_numbers_l24_24137

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l24_24137


namespace decimal_to_fraction_equiv_l24_24711

theorem decimal_to_fraction_equiv : (0.38 : ℝ) = 19 / 50 :=
by
  sorry

end decimal_to_fraction_equiv_l24_24711


namespace marks_in_biology_l24_24222

theorem marks_in_biology (E M P C : ℝ) (A B : ℝ)
  (h1 : E = 90)
  (h2 : M = 92)
  (h3 : P = 85)
  (h4 : C = 87)
  (h5 : A = 87.8) 
  (h6 : (E + M + P + C + B) / 5 = A) : 
  B = 85 := 
by
  -- Placeholder for the proof
  sorry

end marks_in_biology_l24_24222


namespace greatest_possible_sum_of_consecutive_integers_prod_lt_200_l24_24512

theorem greatest_possible_sum_of_consecutive_integers_prod_lt_200 :
  ∃ n : ℤ, (n * (n + 1) < 200) ∧ ( ∀ m : ℤ, (m * (m + 1) < 200) → m ≤ n) ∧ (n + (n + 1) = 27) :=
by
  sorry

end greatest_possible_sum_of_consecutive_integers_prod_lt_200_l24_24512


namespace no_integer_solutions_l24_24438

theorem no_integer_solutions :
  ∀ n m : ℤ, (n^2 + (n+1)^2 + (n+2)^2) ≠ m^2 :=
by
  intro n m
  sorry

end no_integer_solutions_l24_24438


namespace expand_product_l24_24538

variable (x : ℝ)

theorem expand_product :
  (x + 3) * (x^2 + 4 * x + 6) = x^3 + 7 * x^2 + 18 * x + 18 := 
  sorry

end expand_product_l24_24538


namespace solve_for_x_l24_24387

theorem solve_for_x (x : ℝ) (h : (2 / 7) * (1 / 3) * x = 14) : x = 147 :=
sorry

end solve_for_x_l24_24387


namespace total_fruit_count_l24_24515

theorem total_fruit_count :
  let gerald_apple_bags := 5
  let gerald_orange_bags := 4
  let apples_per_gerald_bag := 30
  let oranges_per_gerald_bag := 25
  let pam_apple_bags := 6
  let pam_orange_bags := 4
  let sue_apple_bags := 2 * gerald_apple_bags
  let sue_orange_bags := gerald_orange_bags / 2
  let apples_per_sue_bag := apples_per_gerald_bag - 10
  let oranges_per_sue_bag := oranges_per_gerald_bag + 5
  
  let gerald_apples := gerald_apple_bags * apples_per_gerald_bag
  let gerald_oranges := gerald_orange_bags * oranges_per_gerald_bag
  
  let pam_apples := pam_apple_bags * (3 * apples_per_gerald_bag)
  let pam_oranges := pam_orange_bags * (2 * oranges_per_gerald_bag)
  
  let sue_apples := sue_apple_bags * apples_per_sue_bag
  let sue_oranges := sue_orange_bags * oranges_per_sue_bag

  let total_apples := gerald_apples + pam_apples + sue_apples
  let total_oranges := gerald_oranges + pam_oranges + sue_oranges
  total_apples + total_oranges = 1250 :=

by
  sorry

end total_fruit_count_l24_24515


namespace cylinder_volume_l24_24056

theorem cylinder_volume (r h : ℝ) (radius_is_2 : r = 2) (height_is_3 : h = 3) :
  π * r^2 * h = 12 * π :=
by
  rw [radius_is_2, height_is_3]
  sorry

end cylinder_volume_l24_24056


namespace proj_eq_line_eqn_l24_24594

theorem proj_eq_line_eqn (x y : ℝ)
  (h : (6 * x + 3 * y) * 6 / 45 = -3 ∧ (6 * x + 3 * y) * 3 / 45 = -3 / 2) :
  y = -2 * x - 15 / 2 :=
by
  sorry

end proj_eq_line_eqn_l24_24594


namespace henry_added_water_l24_24347

theorem henry_added_water (F : ℕ) (h2 : F = 32) (α β : ℚ) (h3 : α = 3/4) (h4 : β = 7/8) :
  (F * β) - (F * α) = 4 := by
  sorry

end henry_added_water_l24_24347


namespace log_base_250_2662sqrt10_l24_24547

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variables (a b : ℝ)
variables (h1 : log_base 50 55 = a) (h2 : log_base 55 20 = b)

theorem log_base_250_2662sqrt10 : log_base 250 (2662 * Real.sqrt 10) = (18 * a + 11 * a * b - 13) / (10 - 2 * a * b) :=
by
  sorry

end log_base_250_2662sqrt10_l24_24547


namespace molecular_weight_K3AlC2O4_3_l24_24903

noncomputable def molecularWeightOfCompound : ℝ :=
  let potassium_weight : ℝ := 39.10
  let aluminum_weight  : ℝ := 26.98
  let carbon_weight    : ℝ := 12.01
  let oxygen_weight    : ℝ := 16.00
  let total_potassium_weight : ℝ := 3 * potassium_weight
  let total_aluminum_weight  : ℝ := aluminum_weight
  let total_carbon_weight    : ℝ := 3 * 2 * carbon_weight
  let total_oxygen_weight    : ℝ := 3 * 4 * oxygen_weight
  total_potassium_weight + total_aluminum_weight + total_carbon_weight + total_oxygen_weight

theorem molecular_weight_K3AlC2O4_3 : molecularWeightOfCompound = 408.34 := by
  sorry

end molecular_weight_K3AlC2O4_3_l24_24903


namespace lcm_144_132_eq_1584_l24_24924

theorem lcm_144_132_eq_1584 :
  Nat.lcm 144 132 = 1584 :=
by
  sorry

end lcm_144_132_eq_1584_l24_24924


namespace polyhedron_edges_faces_vertices_l24_24839

theorem polyhedron_edges_faces_vertices
  (E F V n m : ℕ)
  (h1 : n * F = 2 * E)
  (h2 : m * V = 2 * E)
  (h3 : V + F = E + 2) :
  ¬(m * F = 2 * E) :=
sorry

end polyhedron_edges_faces_vertices_l24_24839


namespace range_g_l24_24355

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x + 1
noncomputable def g (a x : ℝ) : ℝ := x^2 + a * x + 1

theorem range_g (a : ℝ) (h : Set.range (λ x => f a x) = Set.univ) : Set.range (λ x => g a x) = { y : ℝ | 1 ≤ y } := by
  sorry

end range_g_l24_24355


namespace problem_statement_l24_24774

variable (F : ℕ → Prop)

theorem problem_statement (h1 : ∀ k : ℕ, F k → F (k + 1)) (h2 : ¬F 7) : ¬F 6 ∧ ¬F 5 := by
  sorry

end problem_statement_l24_24774


namespace smallest_positive_phi_l24_24849

open Real

theorem smallest_positive_phi :
  (∃ k : ℤ, (2 * φ + π / 4 = π / 2 + k * π)) →
  (∀ k, φ = π / 8 + k * π / 2) → 
  0 < φ → 
  φ = π / 8 :=
by
  sorry

end smallest_positive_phi_l24_24849


namespace bullet_train_pass_time_l24_24089

noncomputable def time_to_pass (length_train : ℕ) (speed_train_kmph : ℕ) (speed_man_kmph : ℕ) : ℝ := 
  let relative_speed_kmph := speed_train_kmph + speed_man_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * 1000 / 3600
  length_train / relative_speed_mps

def length_train := 350
def speed_train_kmph := 75
def speed_man_kmph := 12

theorem bullet_train_pass_time : 
  abs (time_to_pass length_train speed_train_kmph speed_man_kmph - 14.47) < 0.01 :=
by
  sorry

end bullet_train_pass_time_l24_24089


namespace decreasing_f_range_l24_24532

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log x - 2 * k * x - 1

theorem decreasing_f_range (k : ℝ) (x₁ x₂ : ℝ) (h₁ : 2 ≤ x₁) (h₂ : x₁ < x₂) (h₃ : x₂ ≤ 4) :
  k ≥ 1 / 4 → (x₁ - x₂) * (f x₁ k - f x₂ k) < 0 :=
sorry

end decreasing_f_range_l24_24532


namespace pq_sum_l24_24190

theorem pq_sum {p q : ℤ}
  (h : ∀ x : ℤ, 36 * x^2 - 4 * (p^2 + 11) * x + 135 * (p + q) + 576 = 0) :
  p + q = 20 :=
sorry

end pq_sum_l24_24190


namespace quotient_calculation_l24_24118

theorem quotient_calculation
  (dividend : ℕ)
  (divisor : ℕ)
  (remainder : ℕ)
  (h_dividend : dividend = 176)
  (h_divisor : divisor = 14)
  (h_remainder : remainder = 8) :
  ∃ q, dividend = divisor * q + remainder ∧ q = 12 :=
by
  sorry

end quotient_calculation_l24_24118


namespace remainder_when_divided_by_24_l24_24896

theorem remainder_when_divided_by_24 (m k : ℤ) (h : m = 288 * k + 47) : m % 24 = 23 :=
by
  sorry

end remainder_when_divided_by_24_l24_24896


namespace obtain_any_natural_from_4_l24_24132

/-- Definitions of allowed operations:
  - Append the digit 4.
  - Append the digit 0.
  - Divide by 2, if the number is even.
--/
def append4 (n : ℕ) : ℕ := 10 * n + 4
def append0 (n : ℕ) : ℕ := 10 * n
def divide2 (n : ℕ) : ℕ := n / 2

/-- We'll also define if a number is even --/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Define the set of operations applied on a number --/
inductive operations : ℕ → ℕ → Prop
| initial : operations 4 4
| append4_step (n m : ℕ) : operations n m → operations n (append4 m)
| append0_step (n m : ℕ) : operations n m → operations n (append0 m)
| divide2_step (n m : ℕ) : is_even m → operations n m → operations n (divide2 m)

/-- The main theorem proving that any natural number can be obtained from 4 using the allowed operations --/
theorem obtain_any_natural_from_4 (n : ℕ) : ∃ m, operations 4 m ∧ m = n :=
by sorry

end obtain_any_natural_from_4_l24_24132


namespace pen_cost_l24_24987

variable (p i : ℝ)

theorem pen_cost (h1 : p + i = 1.10) (h2 : p = 1 + i) : p = 1.05 :=
by 
  -- proof steps here
  sorry

end pen_cost_l24_24987


namespace equal_shipments_by_truck_l24_24414

theorem equal_shipments_by_truck (T : ℕ) (hT1 : 120 % T = 0) (hT2 : T ≠ 5) : T = 2 :=
by
  sorry

end equal_shipments_by_truck_l24_24414


namespace f_2009_value_l24_24416

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_function (f : ℝ → ℝ) : ∀ x, f x = f (-x)
axiom odd_function (g : ℝ → ℝ) : ∀ x, g x = -g (-x)
axiom f_value : f 1 = 0
axiom g_def : ∀ x, g x = f (x - 1)

theorem f_2009_value : f 2009 = 0 :=
by
  sorry

end f_2009_value_l24_24416


namespace alcohol_percentage_in_new_solution_l24_24784

theorem alcohol_percentage_in_new_solution :
  let original_volume := 40 -- liters
  let original_percentage_alcohol := 0.05
  let added_alcohol := 5.5 -- liters
  let added_water := 4.5 -- liters
  let original_alcohol := original_percentage_alcohol * original_volume
  let new_alcohol := original_alcohol + added_alcohol
  let new_volume := original_volume + added_alcohol + added_water
  (new_alcohol / new_volume) * 100 = 15 := by
  sorry

end alcohol_percentage_in_new_solution_l24_24784


namespace candy_pieces_total_l24_24096

def number_of_packages_of_candy := 45
def pieces_per_package := 9

theorem candy_pieces_total : number_of_packages_of_candy * pieces_per_package = 405 :=
by
  sorry

end candy_pieces_total_l24_24096


namespace manufacturer_cost_price_l24_24160

theorem manufacturer_cost_price
    (C : ℝ)
    (h1 : C > 0)
    (h2 : 1.18 * 1.20 * 1.25 * C = 30.09) :
    |C - 17| < 0.01 :=
by
    sorry

end manufacturer_cost_price_l24_24160


namespace radian_measure_15_degrees_l24_24820

theorem radian_measure_15_degrees : (15 * (Real.pi / 180)) = (Real.pi / 12) :=
by
  sorry

end radian_measure_15_degrees_l24_24820


namespace each_half_month_has_15_days_l24_24619

noncomputable def days_in_each_half (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) : ℕ :=
  let first_half_days := total_days / 2
  let second_half_days := total_days - first_half_days
  first_half_days

theorem each_half_month_has_15_days (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) :
  total_days = 30 → mean_profit_total = 350 → mean_profit_first_half = 275 → mean_profit_last_half = 425 → 
  days_in_each_half total_days mean_profit_total mean_profit_first_half mean_profit_last_half = 15 :=
by
  intros h_days h_total h_first h_last
  sorry

end each_half_month_has_15_days_l24_24619


namespace toys_cost_price_gain_l24_24310

theorem toys_cost_price_gain (selling_price : ℕ) (cost_price_per_toy : ℕ) (num_toys : ℕ)
    (total_cost_price : ℕ) (gain : ℕ) (x : ℕ) :
    selling_price = 21000 →
    cost_price_per_toy = 1000 →
    num_toys = 18 →
    total_cost_price = num_toys * cost_price_per_toy →
    gain = selling_price - total_cost_price →
    x = gain / cost_price_per_toy →
    x = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3] at *
  sorry

end toys_cost_price_gain_l24_24310


namespace men_women_arrangement_l24_24930

theorem men_women_arrangement :
  let men := 2
  let women := 4
  let slots := 5
  (Nat.choose slots women) * women.factorial * men.factorial = 240 :=
by
  sorry

end men_women_arrangement_l24_24930


namespace power_function_increasing_iff_l24_24155

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_increasing_iff (a : ℝ) : 
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → power_function a x1 < power_function a x2) ↔ a > 0 := 
by
  sorry

end power_function_increasing_iff_l24_24155


namespace number_of_students_l24_24409

theorem number_of_students (n T : ℕ) (h1 : T = n * 90) 
(h2 : T - 120 = (n - 3) * 95) : n = 33 := 
by
sorry

end number_of_students_l24_24409


namespace hyperbola_eccentricity_l24_24259

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), (b = 5) → (c = 3) → (c^2 = a^2 + b) → (a > 0) →
  (a + c = 3) → (e = c / a) → (e = 3 / 2) :=
by
  intros a b c hb hc hc2 ha hac he
  sorry

end hyperbola_eccentricity_l24_24259


namespace max_combinations_for_n_20_l24_24539

def num_combinations (s n k : ℕ) : ℕ :=
if n = 0 then if s = 0 then 1 else 0
else if s < n then 0
else if k = 0 then 0
else num_combinations (s - k) (n - 1) (k - 1) + num_combinations s n (k - 1)

theorem max_combinations_for_n_20 : ∀ s k, s = 20 ∧ k = 9 → num_combinations s 4 k = 12 :=
by
  intros s k h
  cases h
  sorry

end max_combinations_for_n_20_l24_24539


namespace value_of_g_at_3_l24_24317

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_g_at_3 : g 3 = 3 := by
  sorry

end value_of_g_at_3_l24_24317


namespace train_length_proof_l24_24367

def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  speed_kmph * 5 / 18

theorem train_length_proof (speed_kmph : ℕ) (platform_length_m : ℕ) (crossing_time_s : ℕ) (speed_mps : ℕ) (distance_covered_m : ℕ) (train_length_m : ℕ) :
  speed_kmph = 72 →
  platform_length_m = 270 →
  crossing_time_s = 26 →
  speed_mps = convert_kmph_to_mps speed_kmph →
  distance_covered_m = speed_mps * crossing_time_s →
  train_length_m = distance_covered_m - platform_length_m →
  train_length_m = 250 :=
by
  intros h_speed h_platform h_time h_conv h_dist h_train_length
  sorry

end train_length_proof_l24_24367


namespace qingyang_2015_mock_exam_l24_24993

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def problem :=
  U = {1, 2, 3, 4, 5} ∧ A = {2, 3, 4} ∧ B = {2, 5} →
  B ∪ (U \ A) = {1, 2, 5}

theorem qingyang_2015_mock_exam (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) : problem U A B :=
by
  intros
  sorry

end qingyang_2015_mock_exam_l24_24993


namespace GCF_LCM_18_30_10_45_eq_90_l24_24600

-- Define LCM and GCF functions
def LCM (a b : ℕ) := a / Nat.gcd a b * b
def GCF (a b : ℕ) := Nat.gcd a b

-- Define the problem
theorem GCF_LCM_18_30_10_45_eq_90 : 
  GCF (LCM 18 30) (LCM 10 45) = 90 := by
sorry

end GCF_LCM_18_30_10_45_eq_90_l24_24600


namespace monotonic_intervals_extreme_value_closer_l24_24268

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * (x - 1)

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧
  (a > 0 → (∀ x : ℝ, x < Real.log a → f x a > f (x + 1) a) ∧ (∀ x : ℝ, x > Real.log a → f x a < f (x + 1) a)) :=
sorry

theorem extreme_value_closer (a : ℝ) :
  a > e - 1 →
  ∀ x : ℝ, x ≥ 1 → |Real.exp 1/x - Real.log x| < |Real.exp (x - 1) + a - Real.log x| :=
sorry

end monotonic_intervals_extreme_value_closer_l24_24268


namespace unread_pages_when_a_is_11_l24_24072

variable (a : ℕ)

def total_pages : ℕ := 250
def pages_per_day : ℕ := 15

def unread_pages_after_a_days (a : ℕ) : ℕ := total_pages - pages_per_day * a

theorem unread_pages_when_a_is_11 : unread_pages_after_a_days 11 = 85 :=
by
  sorry

end unread_pages_when_a_is_11_l24_24072


namespace vector_b_value_l24_24757

theorem vector_b_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, -2)
  2 • a + b = (3, 2) → b = (1, -2) :=
by
  intros
  sorry

end vector_b_value_l24_24757


namespace circle_center_and_radius_l24_24725

-- Define a circle in the plane according to the given equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

-- Define the center of the circle
def center (x : ℝ) (y : ℝ) : Prop := x = -2 ∧ y = 0

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 2

-- The theorem statement
theorem circle_center_and_radius :
  (∀ x y, circle_eq x y → center x y) ∧ radius 2 :=
sorry

end circle_center_and_radius_l24_24725


namespace perpendicular_lines_with_foot_l24_24717

theorem perpendicular_lines_with_foot (n : ℝ) : 
  (∀ x y, 10 * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + n = 0) ∧
  (2 * 1 - 5 * (-2) + n = 0) → n = -12 := 
by sorry

end perpendicular_lines_with_foot_l24_24717


namespace solution_set_inequality_l24_24298

theorem solution_set_inequality (x : ℝ) : (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := 
sorry

end solution_set_inequality_l24_24298


namespace parabola_focus_l24_24771

theorem parabola_focus :
  ∀ (x y : ℝ), x^2 = 4 * y → (0, 1) = (0, (2 / 2)) :=
by
  intros x y h
  sorry

end parabola_focus_l24_24771


namespace final_value_of_x_l24_24925

noncomputable def initial_x : ℝ := 52 * 1.2
noncomputable def decreased_x : ℝ := initial_x * 0.9
noncomputable def final_x : ℝ := decreased_x * 1.15

theorem final_value_of_x : final_x = 64.584 := by
  sorry

end final_value_of_x_l24_24925


namespace find_a_inverse_function_l24_24133

theorem find_a_inverse_function
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x y, y = f x ↔ x = a * y)
  (h2 : f 4 = 2) :
  a = 2 := 
sorry

end find_a_inverse_function_l24_24133


namespace asymptotes_of_hyperbola_l24_24676

theorem asymptotes_of_hyperbola (a b : ℝ) (h_cond1 : a > b) (h_cond2 : b > 0) 
  (h_eq_ell : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h_eq_hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h_product : ∀ e1 e2 : ℝ, (e1 = Real.sqrt (1 - (b^2 / a^2))) → 
                (e2 = Real.sqrt (1 + (b^2 / a^2))) → 
                (e1 * e2 = Real.sqrt 3 / 2)) :
  ∀ x y : ℝ, x + Real.sqrt 2 * y = 0 ∨ x - Real.sqrt 2 * y = 0 :=
sorry

end asymptotes_of_hyperbola_l24_24676


namespace g_2002_eq_1_l24_24217

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ := λ x => f x + 1 - x)

axiom f_one : f 1 = 1
axiom f_inequality_1 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

theorem g_2002_eq_1 : g 2002 = 1 := by
  sorry

end g_2002_eq_1_l24_24217


namespace solve_equation1_solve_equation2_l24_24318

-- Problem for Equation (1)
theorem solve_equation1 (x : ℝ) : x * (x - 6) = 2 * (x - 8) → x = 4 := by
  sorry

-- Problem for Equation (2)
theorem solve_equation2 (x : ℝ) : (2 * x - 1)^2 + 3 * (2 * x - 1) + 2 = 0 → x = 0 ∨ x = -1 / 2 := by
  sorry

end solve_equation1_solve_equation2_l24_24318


namespace proof_a_eq_neg2x_or_3x_l24_24080

theorem proof_a_eq_neg2x_or_3x (a b x : ℝ) (h1 : a - b = x) (h2 : a^3 - b^3 = 19 * x^3) (h3 : x ≠ 0) : 
  a = -2 * x ∨ a = 3 * x :=
  sorry

end proof_a_eq_neg2x_or_3x_l24_24080


namespace find_x_l24_24829

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_x (x : ℝ) : 
  (sqrt x / sqrt 0.81 + sqrt 1.44 / sqrt 0.49 = 3.0751133491652576) → 
  x = 1.5 :=
by { sorry }

end find_x_l24_24829


namespace can_divide_cube_into_71_l24_24353

theorem can_divide_cube_into_71 : 
  ∃ (n : ℕ), n = 71 ∧ 
  (∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = f k + 7) ∧ f n = 71) :=
by
  sorry

end can_divide_cube_into_71_l24_24353


namespace compute_division_l24_24921

theorem compute_division : 0.182 / 0.0021 = 86 + 14 / 21 :=
by
  sorry

end compute_division_l24_24921


namespace total_tiles_is_1352_l24_24475

noncomputable def side_length_of_floor := 39

noncomputable def total_tiles_covering_floor (n : ℕ) : ℕ :=
  (n ^ 2) - ((n / 3) ^ 2)

theorem total_tiles_is_1352 :
  total_tiles_covering_floor side_length_of_floor = 1352 := by
  sorry

end total_tiles_is_1352_l24_24475


namespace initial_investment_l24_24846

noncomputable def doubling_period (r : ℝ) : ℝ := 70 / r
noncomputable def investment_after_doubling (P : ℝ) (n : ℝ) : ℝ := P * (2 ^ n)

theorem initial_investment (total_amount : ℝ) (years : ℝ) (rate : ℝ) (initial : ℝ) :
  rate = 8 → total_amount = 28000 → years = 18 → 
  initial = total_amount / (2 ^ (years / (doubling_period rate))) :=
by
  intros hrate htotal hyears
  simp [doubling_period, investment_after_doubling] at *
  rw [hrate, htotal, hyears]
  norm_num
  sorry

end initial_investment_l24_24846


namespace calculate_p_p_l24_24324

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 2*y
  else if x ≥ 0 ∧ y < 0 then x^2 + y^2
  else 3*x + y

theorem calculate_p_p : p (p 2 (-3)) (p (-4) 1) = 290 :=
by {
  -- required statement of proof problem
  sorry
}

end calculate_p_p_l24_24324


namespace train_lengths_combined_l24_24542

noncomputable def speed_to_mps (kmph : ℤ) : ℚ := (kmph : ℚ) * 5 / 18

def length_of_train (speed : ℚ) (time : ℚ) : ℚ := speed * time

theorem train_lengths_combined :
  let speed1_kmph := 100
  let speed2_kmph := 120
  let time1_sec := 9
  let time2_sec := 8
  let speed1_mps := speed_to_mps speed1_kmph
  let speed2_mps := speed_to_mps speed2_kmph
  let length1 := length_of_train speed1_mps time1_sec
  let length2 := length_of_train speed2_mps time2_sec
  length1 + length2 = 516.66 :=
by
  sorry

end train_lengths_combined_l24_24542


namespace find_P_l24_24388

noncomputable def parabola_vertex : ℝ × ℝ := (0, 0)
noncomputable def parabola_focus : ℝ × ℝ := (0, -1)
noncomputable def point_P : ℝ × ℝ := (20 * Real.sqrt 6, -120)
noncomputable def PF_distance : ℝ := 121

def parabola_equation (x y : ℝ) : Prop :=
  x^2 = -4 * y

def parabola_condition (x y : ℝ) : Prop :=
  (parabola_equation x y) ∧ 
  (Real.sqrt (x^2 + (y + 1)^2) = PF_distance)

theorem find_P : parabola_condition (point_P.1) (point_P.2) :=
by
  sorry

end find_P_l24_24388


namespace soccer_balls_per_class_l24_24011

-- Definitions for all conditions in the problem
def elementary_classes_per_school : ℕ := 4
def middle_school_classes_per_school : ℕ := 5
def number_of_schools : ℕ := 2
def total_soccer_balls_donated : ℕ := 90

-- The total number of classes in one school
def classes_per_school : ℕ := elementary_classes_per_school + middle_school_classes_per_school

-- The total number of classes in both schools
def total_classes : ℕ := classes_per_school * number_of_schools

-- Prove that the number of soccer balls donated per class is 5
theorem soccer_balls_per_class : total_soccer_balls_donated / total_classes = 5 :=
  by sorry

end soccer_balls_per_class_l24_24011


namespace regular_polygon_perimeter_l24_24887

theorem regular_polygon_perimeter
  (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : side_length = 8)
  (h2 : exterior_angle = 90)
  (h3 : n = 360 / exterior_angle) :
  n * side_length = 32 := by
  sorry

end regular_polygon_perimeter_l24_24887


namespace orange_cost_l24_24622

-- Definitions based on the conditions
def dollar_per_pound := 5 / 6
def pounds : ℕ := 18
def total_cost := pounds * dollar_per_pound

-- The statement to be proven
theorem orange_cost : total_cost = 15 :=
by
  sorry

end orange_cost_l24_24622


namespace calculator_to_protractors_l24_24344

def calculator_to_rulers (c: ℕ) : ℕ := 100 * c
def rulers_to_compasses (r: ℕ) : ℕ := (r * 30) / 10
def compasses_to_protractors (p: ℕ) : ℕ := (p * 50) / 25

theorem calculator_to_protractors (c: ℕ) : compasses_to_protractors (rulers_to_compasses (calculator_to_rulers c)) = 600 * c :=
by
  sorry

end calculator_to_protractors_l24_24344


namespace loop_until_correct_l24_24911

-- Define the conditions
def num_iterations := 20

-- Define the loop condition
def loop_condition (i : Nat) : Prop := i > num_iterations

-- Theorem: Proof that the loop should continue until the counter i exceeds 20
theorem loop_until_correct (i : Nat) : loop_condition i := by
  sorry

end loop_until_correct_l24_24911


namespace cos_theta_minus_pi_six_l24_24998

theorem cos_theta_minus_pi_six (θ : ℝ) (h : Real.sin (θ + π / 3) = 2 / 3) : 
  Real.cos (θ - π / 6) = 2 / 3 :=
sorry

end cos_theta_minus_pi_six_l24_24998


namespace find_QS_l24_24029

theorem find_QS (cosR : ℝ) (RS QR QS : ℝ) (h1 : cosR = 3 / 5) (h2 : RS = 10) (h3 : cosR = QR / RS) (h4: QR ^ 2 + QS ^ 2 = RS ^ 2) : QS = 8 :=
by 
  sorry

end find_QS_l24_24029


namespace buses_needed_l24_24693

def total_students : ℕ := 111
def seats_per_bus : ℕ := 3

theorem buses_needed : total_students / seats_per_bus = 37 :=
by
  sorry

end buses_needed_l24_24693


namespace f_19_eq_2017_l24_24870

noncomputable def f : ℤ → ℤ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ m n : ℤ, f (m + n) = f m + f n + 3 * (4 * m * n - 1)

theorem f_19_eq_2017 : f 19 = 2017 := by
  sorry

end f_19_eq_2017_l24_24870


namespace percent_savings_12_roll_package_l24_24789

def percent_savings_per_roll (package_cost : ℕ) (individual_cost : ℕ) (num_rolls : ℕ) : ℚ :=
  let individual_total := num_rolls * individual_cost
  let package_total := package_cost
  let per_roll_package_cost := package_total / num_rolls
  let savings_per_roll := individual_cost - per_roll_package_cost
  (savings_per_roll / individual_cost) * 100

theorem percent_savings_12_roll_package :
  percent_savings_per_roll 9 1 12 = 25 := 
sorry

end percent_savings_12_roll_package_l24_24789


namespace equation_solutions_l24_24483

noncomputable def solve_equation (x : ℝ) : Prop :=
  x - 3 = 4 * (x - 3)^2

theorem equation_solutions :
  ∀ x : ℝ, solve_equation x ↔ x = 3 ∨ x = 3.25 :=
by sorry

end equation_solutions_l24_24483


namespace find_n_l24_24439

theorem find_n (n : ℕ) (h_pos : 0 < n) (h_prime : Nat.Prime (n^4 - 16 * n^2 + 100)) : n = 3 := 
sorry

end find_n_l24_24439


namespace range_of_a_l24_24203

theorem range_of_a (x y a : ℝ) (h1 : x - y = 2) (h2 : x + y = a) (h3 : x > -1) (h4 : y < 0) : -4 < a ∧ a < 2 :=
sorry

end range_of_a_l24_24203


namespace range_alpha_div_three_l24_24891

open Real

theorem range_alpha_div_three (α : ℝ) (k : ℤ) :
  sin α > 0 → cos α < 0 → sin (α / 3) > cos (α / 3) →
  ∃ k : ℤ,
    (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) ∨
    (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
by
  intros
  sorry

end range_alpha_div_three_l24_24891


namespace possible_to_fill_grid_l24_24218

/-- Define the grid as a 2D array where each cell contains either 0 or 1. --/
def grid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), i < 5 → j < 5 → f i j = 0 ∨ f i j = 1

/-- Ensure the sum of every 2x2 subgrid is divisible by 3. --/
def divisible_by_3_in_subgrid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < 4 → j < 4 → (f i j + f (i+1) j + f i (j+1) + f (i+1) (j+1)) % 3 = 0

/-- Ensure both 0 and 1 are present in the grid. --/
def contains_0_and_1 (f : ℕ → ℕ → ℕ) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 0) ∧ (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 1)

/-- The main theorem stating the possibility of such a grid. --/
theorem possible_to_fill_grid :
  ∃ f, grid f ∧ divisible_by_3_in_subgrid f ∧ contains_0_and_1 f :=
sorry

end possible_to_fill_grid_l24_24218


namespace sqrt_sq_eq_abs_l24_24151

theorem sqrt_sq_eq_abs (a : ℝ) : Real.sqrt (a^2) = |a| :=
sorry

end sqrt_sq_eq_abs_l24_24151


namespace fraction_of_second_year_students_l24_24045

-- Define the fractions of first-year and second-year students
variables (F S f s: ℝ)

-- Conditions
axiom h1 : F + S = 1
axiom h2 : f = (1 / 5) * F
axiom h3 : s = 4 * f
axiom h4 : S - s = 0.2

-- The theorem statement to prove the fraction of second-year students is 2 / 3
theorem fraction_of_second_year_students (F S f s: ℝ) 
    (h1: F + S = 1) 
    (h2: f = (1 / 5) * F) 
    (h3: s = 4 * f) 
    (h4: S - s = 0.2) : 
    S = 2 / 3 :=
by 
    sorry

end fraction_of_second_year_students_l24_24045


namespace BC_at_least_17_l24_24383

-- Given conditions
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
-- Distances given
variables (AB AC EC BD BC : ℝ)
variables (AB_pos : AB = 7)
variables (AC_pos : AC = 15)
variables (EC_pos : EC = 9)
variables (BD_pos : BD = 26)
-- Triangle Inequalities
variables (triangle_ABC : ∀ {x y z : Type} [MetricSpace x] [MetricSpace y] [MetricSpace z], AC - AB < BC)
variables (triangle_DEC : ∀ {x y z : Type} [MetricSpace x] [MetricSpace y] [MetricSpace z], BD - EC < BC)

-- Proof statement
theorem BC_at_least_17 : BC ≥ 17 := by
  sorry

end BC_at_least_17_l24_24383


namespace max_marks_l24_24915

theorem max_marks (M : ℝ) (score passing shortfall : ℝ)
  (h_score : score = 212)
  (h_shortfall : shortfall = 44)
  (h_passing : passing = score + shortfall)
  (h_pass_cond : passing = 0.4 * M) :
  M = 640 :=
by
  sorry

end max_marks_l24_24915


namespace dress_designs_count_l24_24250

inductive Color
| red | green | blue | yellow

inductive Pattern
| stripes | polka_dots | floral | geometric | plain

def patterns_for_color (c : Color) : List Pattern :=
  match c with
  | Color.red    => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.green  => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]
  | Color.blue   => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.yellow => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]

noncomputable def number_of_dress_designs : ℕ :=
  (patterns_for_color Color.red).length +
  (patterns_for_color Color.green).length +
  (patterns_for_color Color.blue).length +
  (patterns_for_color Color.yellow).length

theorem dress_designs_count : number_of_dress_designs = 18 :=
  by
  sorry

end dress_designs_count_l24_24250


namespace Mike_got_18_cards_l24_24553

theorem Mike_got_18_cards (original_cards : ℕ) (total_cards : ℕ) : 
  original_cards = 64 → total_cards = 82 → total_cards - original_cards = 18 :=
by
  intros h1 h2
  sorry

end Mike_got_18_cards_l24_24553


namespace find_f_2006_l24_24772

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x
def g_def (f : ℝ → ℝ) (g : ℝ → ℝ) := ∀ x : ℝ, g x = f (x - 1)
def f_at_2 (f : ℝ → ℝ) := f 2 = 2

-- The theorem to prove
theorem find_f_2006 (f g : ℝ → ℝ) 
  (even_f : is_even f) 
  (odd_g : is_odd g) 
  (g_eq_f_shift : g_def f g) 
  (f_eq_2 : f_at_2 f) : 
  f 2006 = 2 := 
sorry

end find_f_2006_l24_24772


namespace quentavious_gum_count_l24_24511

def initial_nickels : Nat := 5
def remaining_nickels : Nat := 2
def gum_per_nickel : Nat := 2
def traded_nickels (initial remaining : Nat) : Nat := initial - remaining
def total_gum (trade_n gum_per_n : Nat) : Nat := trade_n * gum_per_n

theorem quentavious_gum_count : total_gum (traded_nickels initial_nickels remaining_nickels) gum_per_nickel = 6 := by
  sorry

end quentavious_gum_count_l24_24511


namespace mary_regular_hours_l24_24140

theorem mary_regular_hours (x y : ℕ) :
  8 * x + 10 * y = 760 ∧ x + y = 80 → x = 20 :=
by
  intro h
  sorry

end mary_regular_hours_l24_24140


namespace smallest_n_equal_sums_l24_24018

def sum_first_n_arithmetic (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_n_equal_sums : ∀ (n : ℕ), 
  sum_first_n_arithmetic 7 4 n = sum_first_n_arithmetic 15 3 n → n ≠ 0 → n = 7 := by
  intros n h1 h2
  sorry

end smallest_n_equal_sums_l24_24018


namespace bread_problem_l24_24095

variable (x : ℝ)

theorem bread_problem (h1 : x > 0) :
  (15 / x) - 1 = 14 / (x + 2) :=
sorry

end bread_problem_l24_24095


namespace simplify_expression_l24_24357

variable (a b : ℚ)

theorem simplify_expression (ha : a = -2) (hb : b = 1/5) :
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -2 := by
  -- Proof can be filled here
  sorry

end simplify_expression_l24_24357


namespace smallest_positive_integer_k_l24_24904

-- Define the conditions
def y : ℕ := 2^3 * 3^4 * (2^2)^5 * 5^6 * (2*3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Define the question statement
theorem smallest_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (y * k) = m^2) ∧ k = 30 :=
by
  sorry

end smallest_positive_integer_k_l24_24904


namespace value_of_4x_l24_24607

variable (x : ℤ)

theorem value_of_4x (h : 2 * x - 3 = 10) : 4 * x = 26 := 
by
  sorry

end value_of_4x_l24_24607


namespace problem1_l24_24552

theorem problem1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

end problem1_l24_24552


namespace range_of_m_l24_24738

theorem range_of_m (m n : ℝ) (h1 : n = 2 / m) (h2 : n ≥ -1) :
  m ≤ -2 ∨ m > 0 := 
sorry

end range_of_m_l24_24738


namespace spending_spring_months_l24_24546

variable (s_feb s_may : ℝ)

theorem spending_spring_months (h1 : s_feb = 2.8) (h2 : s_may = 5.6) : s_may - s_feb = 2.8 := 
by
  sorry

end spending_spring_months_l24_24546


namespace sum_solutions_eq_16_l24_24715

theorem sum_solutions_eq_16 (x y : ℝ) 
  (h1 : |x - 5| = |y - 11|)
  (h2 : |x - 11| = 2 * |y - 5|)
  (h3 : x + y = 16) :
  x + y = 16 :=
by
  sorry

end sum_solutions_eq_16_l24_24715


namespace negation_of_exists_l24_24844

theorem negation_of_exists (h : ¬ (∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0)) : ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
by
  sorry

end negation_of_exists_l24_24844


namespace triangle_inequality_cubed_l24_24733

theorem triangle_inequality_cubed
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a^3 / c^3) + (b^3 / c^3) + (3 * a * b / c^2) > 1 := 
sorry

end triangle_inequality_cubed_l24_24733


namespace find_ratio_l24_24695

variable (x y : ℝ)

-- Hypotheses: x and y are distinct real numbers and the given equation holds
variable (h₁ : x ≠ y)
variable (h₂ : x / y + (x + 15 * y) / (y + 15 * x) = 3)

-- We aim to prove that x / y = 0.8
theorem find_ratio (h₁ : x ≠ y) (h₂ : x / y + (x + 15 * y) / (y + 15 * x) = 3) : x / y = 0.8 :=
sorry

end find_ratio_l24_24695


namespace max_difference_intersection_ycoords_l24_24043

theorem max_difference_intersection_ycoords :
  let f₁ (x : ℝ) := 5 - 2 * x^2 + x^3
  let f₂ (x : ℝ) := 1 + x^2 + x^3
  let x1 := (2 : ℝ) / Real.sqrt 3
  let x2 := - (2 : ℝ) / Real.sqrt 3
  let y1 := f₁ x1
  let y2 := f₂ x2
  (f₁ = f₂)
  → abs (y1 - y2) = (16 * Real.sqrt 3 / 9) :=
by
  sorry

end max_difference_intersection_ycoords_l24_24043


namespace solve_for_r_l24_24953

theorem solve_for_r (r : ℝ) : 
  (r^2 - 3) / 3 = (5 - r) / 2 ↔ 
  r = (-3 + Real.sqrt 177) / 4 ∨ r = (-3 - Real.sqrt 177) / 4 :=
by
  sorry

end solve_for_r_l24_24953


namespace difference_of_squares_l24_24479

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 := 
sorry

end difference_of_squares_l24_24479


namespace max_daily_sales_revenue_l24_24315

noncomputable def f (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t < 15 
  then (1 / 3) * t + 8
  else if 15 ≤ t ∧ t < 30 
  then -(1 / 3) * t + 18
  else 0

noncomputable def g (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t ≤ 30
  then -t + 30
  else 0

noncomputable def W (t : ℕ) : ℝ :=
  f t * g t

theorem max_daily_sales_revenue : ∃ t : ℕ, W t = 243 :=
by
  existsi 3
  sorry

end max_daily_sales_revenue_l24_24315


namespace integer_ratio_value_l24_24922

theorem integer_ratio_value {x y : ℝ} (h1 : 3 < (x^2 - y^2) / (x^2 + y^2)) (h2 : (x^2 - y^2) / (x^2 + y^2) < 4) (h3 : ∃ t : ℤ, x = t * y) : ∃ t : ℤ, t = 2 :=
by
  sorry

end integer_ratio_value_l24_24922


namespace students_in_second_class_l24_24793

variable (x : ℕ)

theorem students_in_second_class :
  (∃ x, 30 * 40 + 70 * x = (30 + x) * 58.75) → x = 50 :=
by
  sorry

end students_in_second_class_l24_24793


namespace f_zero_is_one_l24_24171

def f (n : ℕ) : ℕ := sorry

theorem f_zero_is_one (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (f n) + f n = 2 * n + 3)
  (h2 : f 2015 = 2016) : f 0 = 1 := 
by {
  -- proof not required
  sorry
}

end f_zero_is_one_l24_24171


namespace cause_of_polarization_by_electronegativity_l24_24732

-- Definition of the problem conditions as hypotheses
def strong_polarization_of_CH_bond (C_H_bond : Prop) (electronegativity : Prop) : Prop 
  := C_H_bond ∧ electronegativity

-- Given conditions: Carbon atom is in sp hybridization and C-H bond shows strong polarization
axiom carbon_sp_hybridized : Prop
axiom CH_bond_strong_polarization : Prop

-- Question: The cause of strong polarization of the C-H bond at the carbon atom in sp hybridization in alkynes
def cause_of_strong_polarization (sp_hybridization : Prop) : Prop 
  := true  -- This definition will hold as a placeholder, to indicate there is a causal connection

-- Correct answer: high electronegativity of the carbon atom in sp-hybrid state causes strong polarization
theorem cause_of_polarization_by_electronegativity 
  (high_electronegativity : Prop) 
  (sp_hybridized : Prop) 
  (polarized : Prop) 
  (H : strong_polarization_of_CH_bond polarized high_electronegativity) 
  : sp_hybridized ∧ polarized := 
  sorry

end cause_of_polarization_by_electronegativity_l24_24732


namespace area_ratio_is_four_l24_24297

-- Definitions based on the given conditions
variables (k a b c d : ℝ)
variables (ka kb kc kd : ℝ)

-- Equations from the conditions
def eq1 : a = k * ka := sorry
def eq2 : b = k * kb := sorry
def eq3 : c = k * kc := sorry
def eq4 : d = k * kd := sorry

-- Ratios provided in the problem
def ratio1 : ka / kc = 2 / 5 := sorry
def ratio2 : kb / kd = 2 / 5 := sorry

-- The theorem to prove the ratio of areas is 4:1
theorem area_ratio_is_four : (k * ka * k * kb) / (k * kc * k * kd) = 4 :=
by sorry

end area_ratio_is_four_l24_24297


namespace find_x_l24_24908

theorem find_x (x y : ℝ) :
  (x / (x - 1) = (y^2 + 3*y + 2) / (y^2 + 3*y - 1)) →
  x = (y^2 + 3*y + 2) / 3 :=
by
  intro h
  sorry

end find_x_l24_24908


namespace largest_possible_s_l24_24191

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) (h3 : (r - 2) * 180 * s = (s - 2) * 180 * r * 61 / 60) : s = 118 :=
sorry

end largest_possible_s_l24_24191


namespace total_balls_l24_24037

theorem total_balls (S V B Total : ℕ) (hS : S = 68) (hV : S = V - 12) (hB : S = B + 23) : 
  Total = S + V + B := by
  sorry

end total_balls_l24_24037


namespace maximize_profit_l24_24393

theorem maximize_profit : 
  ∃ (a b : ℕ), 
  a ≤ 8 ∧ 
  b ≤ 7 ∧ 
  2 * a + b ≤ 19 ∧ 
  a + b ≤ 12 ∧ 
  10 * a + 6 * b ≥ 72 ∧ 
  (a * 450 + b * 350) = 4900 :=
by
  sorry

end maximize_profit_l24_24393


namespace h_has_only_one_zero_C2_below_C1_l24_24189

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := 1 - 1/x
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem h_has_only_one_zero (x : ℝ) (hx : x > 0) : 
  ∃! (x0 : ℝ), x0 > 0 ∧ h x0 = 0 := sorry

theorem C2_below_C1 (x : ℝ) (hx : x > 0) (hx1 : x ≠ 1) : 
  g x < f x := sorry

end h_has_only_one_zero_C2_below_C1_l24_24189


namespace numerical_puzzle_solution_l24_24740

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l24_24740


namespace find_c_for_minimum_value_l24_24440

-- Definitions based on the conditions
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Main statement to be proved
theorem find_c_for_minimum_value (c : ℝ) : (∀ x, (3*x^2 - 4*c*x + c^2) = 0) → c = 3 :=
by
  sorry

end find_c_for_minimum_value_l24_24440


namespace percent_motorists_receive_tickets_l24_24231

theorem percent_motorists_receive_tickets (n : ℕ) (h1 : (25 : ℕ) % 100 = 25) (h2 : (20 : ℕ) % 100 = 20) :
  (75 * n / 100) = (20 * n / 100) :=
by
  sorry

end percent_motorists_receive_tickets_l24_24231


namespace geometric_sum_l24_24427

theorem geometric_sum 
  (a : ℕ → ℝ) (q : ℝ) (h1 : a 2 + a 4 = 32) (h2 : a 6 + a 8 = 16) 
  (h_seq : ∀ n, a (n+2) = a n * q ^ 2):
  a 10 + a 12 + a 14 + a 16 = 12 :=
by
  -- Proof needs to be written here
  sorry

end geometric_sum_l24_24427


namespace calc_sqrt_expr_l24_24051

theorem calc_sqrt_expr : (Real.sqrt 2 + 1) ^ 2 - Real.sqrt 18 + 2 * Real.sqrt (1 / 2) = 3 := by
  sorry

end calc_sqrt_expr_l24_24051


namespace canoe_upstream_speed_l24_24258

theorem canoe_upstream_speed (C : ℝ) (stream_speed downstream_speed : ℝ) 
  (h_stream : stream_speed = 2) (h_downstream : downstream_speed = 12) 
  (h_equation : C + stream_speed = downstream_speed) :
  C - stream_speed = 8 := 
by 
  sorry

end canoe_upstream_speed_l24_24258


namespace jake_sausages_cost_l24_24110

theorem jake_sausages_cost :
  let package_weight := 2
  let num_packages := 3
  let cost_per_pound := 4
  let total_weight := package_weight * num_packages
  let total_cost := total_weight * cost_per_pound
  total_cost = 24 := by
  sorry

end jake_sausages_cost_l24_24110


namespace total_area_of_figure_l24_24687

theorem total_area_of_figure :
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  total_area = 89 := by
  -- Definitions
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  -- Proof
  sorry

end total_area_of_figure_l24_24687


namespace line_tangent_constant_sum_l24_24131

noncomputable def parabolaEquation (x y : ℝ) : Prop :=
  y ^ 2 = 4 * x

noncomputable def circleEquation (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

noncomputable def isTangent (l : ℝ → ℝ) (x y : ℝ) : Prop :=
  l x = y ∧ ((x - 2) ^ 2 + y ^ 2 = 4)

theorem line_tangent_constant_sum (l : ℝ → ℝ) (A B P : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabolaEquation x₁ y₁ →
  parabolaEquation x₂ y₂ →
  isTangent l (4 / 5) (8 / 5) →
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  let F := (1, 0)
  let distance (p1 p2 : ℝ × ℝ) : ℝ := (Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2))
  (distance F A) + (distance F B) - (distance A B) = 2 :=
sorry

end line_tangent_constant_sum_l24_24131


namespace number_of_cars_in_second_box_is_31_l24_24927

-- Define the total number of toy cars, and the number of toy cars in the first and third boxes
def total_toy_cars : ℕ := 71
def cars_in_first_box : ℕ := 21
def cars_in_third_box : ℕ := 19

-- Define the number of toy cars in the second box
def cars_in_second_box : ℕ := total_toy_cars - cars_in_first_box - cars_in_third_box

-- Theorem stating that the number of toy cars in the second box is 31
theorem number_of_cars_in_second_box_is_31 : cars_in_second_box = 31 :=
by
  sorry

end number_of_cars_in_second_box_is_31_l24_24927


namespace sum_of_three_largest_of_consecutive_numbers_l24_24067

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l24_24067


namespace total_dots_not_visible_l24_24936

theorem total_dots_not_visible
    (num_dice : ℕ)
    (dots_per_die : ℕ)
    (visible_faces : ℕ → ℕ)
    (visible_faces_count : ℕ)
    (total_dots : ℕ)
    (dots_visible : ℕ) :
    num_dice = 4 →
    dots_per_die = 21 →
    visible_faces 0 = 1 →
    visible_faces 1 = 2 →
    visible_faces 2 = 2 →
    visible_faces 3 = 3 →
    visible_faces 4 = 4 →
    visible_faces 5 = 5 →
    visible_faces 6 = 6 →
    visible_faces 7 = 6 →
    visible_faces_count = 8 →
    total_dots = num_dice * dots_per_die →
    dots_visible = visible_faces 0 + visible_faces 1 + visible_faces 2 + visible_faces 3 + visible_faces 4 + visible_faces 5 + visible_faces 6 + visible_faces 7 →
    total_dots - dots_visible = 55 := by
  sorry

end total_dots_not_visible_l24_24936


namespace avg_age_l24_24300

-- Given conditions
variables (A B C : ℕ)
variable (h1 : (A + C) / 2 = 29)
variable (h2 : B = 20)

-- to prove
theorem avg_age (A B C : ℕ) (h1 : (A + C) / 2 = 29) (h2 : B = 20) : (A + B + C) / 3 = 26 :=
sorry

end avg_age_l24_24300


namespace AB_ratio_CD_l24_24473

variable (AB CD : ℝ)
variable (h : ℝ)
variable (O : Point)
variable (ABCD_isosceles : IsIsoscelesTrapezoid AB CD)
variable (areas_condition : List ℝ) 
-- where the list areas_condition represents: [S_OCD, S_OBC, S_OAB, S_ODA]

theorem AB_ratio_CD : 
  ABCD_isosceles ∧ areas_condition = [2, 3, 4, 5] → AB = 2 * CD :=
by
  sorry

end AB_ratio_CD_l24_24473


namespace reciprocal_geometric_sum_l24_24742

/-- The sum of the new geometric progression formed by taking the reciprocal of each term in the original progression,
    where the original progression has 10 terms, the first term is 2, and the common ratio is 3, is \( \frac{29524}{59049} \). -/
theorem reciprocal_geometric_sum :
  let a := 2
  let r := 3
  let n := 10
  let sn := (2 * (1 - r^n)) / (1 - r)
  let sn_reciprocal := (1 / a) * (1 - (1/r)^n) / (1 - 1/r)
  (sn_reciprocal = 29524 / 59049) :=
by
  sorry

end reciprocal_geometric_sum_l24_24742


namespace fraction_value_l24_24168

theorem fraction_value : (2020 / (20 * 20 : ℝ)) = 5.05 := by
  sorry

end fraction_value_l24_24168


namespace find_original_comic_books_l24_24751

def comic_books (X : ℕ) : Prop :=
  X / 2 + 6 = 13

theorem find_original_comic_books (X : ℕ) (h : comic_books X) : X = 14 :=
by
  sorry

end find_original_comic_books_l24_24751


namespace complex_expression_power_48_l24_24123

open Complex

noncomputable def complex_expression := (1 + I) / Real.sqrt 2

theorem complex_expression_power_48 : complex_expression ^ 48 = 1 := by
  sorry

end complex_expression_power_48_l24_24123


namespace g_odd_l24_24851

def g (x : ℝ) : ℝ := x^3 - 2*x

theorem g_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_odd_l24_24851


namespace difference_of_numbers_l24_24900

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : abs (x - y) = 7 :=
sorry

end difference_of_numbers_l24_24900


namespace proof_neg_q_l24_24165

variable (f : ℝ → ℝ)
variable (x : ℝ)

def proposition_p (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

def proposition_q : Prop := ∃ x : ℝ, (deriv fun y => 1 / y) x > 0

theorem proof_neg_q : ¬ proposition_q := 
by
  intro h
  -- proof omitted for brevity
  sorry

end proof_neg_q_l24_24165


namespace greatest_three_digit_divisible_by_3_5_6_l24_24413

theorem greatest_three_digit_divisible_by_3_5_6 : 
    ∃ n : ℕ, 
        (100 ≤ n ∧ n ≤ 999) ∧ 
        (∃ k₃ : ℕ, n = 3 * k₃) ∧ 
        (∃ k₅ : ℕ, n = 5 * k₅) ∧ 
        (∃ k₆ : ℕ, n = 6 * k₆) ∧ 
        (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) ∧ (∃ k₃ : ℕ, m = 3 * k₃) ∧ (∃ k₅ : ℕ, m = 5 * k₅) ∧ (∃ k₆ : ℕ, m = 6 * k₆) → m ≤ 990) := by
  sorry

end greatest_three_digit_divisible_by_3_5_6_l24_24413


namespace omega_value_l24_24092

theorem omega_value (ω : ℕ) (h : ω > 0) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = Real.sin (ω * x + Real.pi / 4)) 
  (h2 : ∀ x y, (Real.pi / 6 < x ∧ x < Real.pi / 3) → (Real.pi / 6 < y ∧ y < Real.pi / 3) → x < y → f y < f x) :
    ω = 2 ∨ ω = 3 := 
sorry

end omega_value_l24_24092


namespace equilateral_triangle_l24_24188

variable {a b c : ℝ}

-- Conditions
def condition1 (a b c : ℝ) : Prop :=
  (a + b + c) * (b + c - a) = 3 * b * c

def condition2 (a b c : ℝ) (cos_B cos_C : ℝ) : Prop :=
  c * cos_B = b * cos_C

-- Theorem statement
theorem equilateral_triangle (a b c : ℝ) (cos_B cos_C : ℝ)
  (h1 : condition1 a b c)
  (h2 : condition2 a b c cos_B cos_C) :
  a = b ∧ b = c :=
sorry

end equilateral_triangle_l24_24188


namespace inconsistent_intercepts_l24_24053

-- Define the ellipse equation
def ellipse (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m + y^2 / 4 = 1

-- Define the line equations
def line1 (x k : ℝ) : ℝ := k * x + 1
def line2 (x : ℝ) (k : ℝ) : ℝ := - k * x - 2

-- Disc calculation for line1
def disc1 (m k : ℝ) : ℝ :=
  let a := 4 + m * k^2
  let b := 2 * m * k
  let c := -3 * m
  b^2 - 4 * a * c

-- Disc calculation for line2
def disc2 (m k : ℝ) : ℝ :=
  let bb := 4 * m * k
  bb^2

-- Statement of the problem
theorem inconsistent_intercepts (m k : ℝ) (hm_pos : 0 < m) :
  disc1 m k ≠ disc2 m k :=
by
  sorry

end inconsistent_intercepts_l24_24053


namespace cistern_height_l24_24920

theorem cistern_height (l w A : ℝ) (h : ℝ) (hl : l = 8) (hw : w = 6) (hA : 48 + 2 * (l * h) + 2 * (w * h) = 99.8) : h = 1.85 := by
  sorry

end cistern_height_l24_24920


namespace elephants_at_WePreserveForFuture_l24_24561

theorem elephants_at_WePreserveForFuture (E : ℕ) 
  (h1 : ∀ gest : ℕ, gest = 3 * E)
  (h2 : ∀ total : ℕ, total = E + 3 * E) 
  (h3 : total = 280) : 
  E = 70 := 
by
  sorry

end elephants_at_WePreserveForFuture_l24_24561


namespace recurring_division_l24_24496

def recurring_36_as_fraction : ℚ := 36 / 99
def recurring_12_as_fraction : ℚ := 12 / 99

theorem recurring_division :
  recurring_36_as_fraction / recurring_12_as_fraction = 3 := 
sorry

end recurring_division_l24_24496


namespace giant_spider_weight_ratio_l24_24275

theorem giant_spider_weight_ratio 
    (W_previous : ℝ)
    (A_leg : ℝ)
    (P : ℝ)
    (n : ℕ)
    (W_previous_eq : W_previous = 6.4)
    (A_leg_eq : A_leg = 0.5)
    (P_eq : P = 4)
    (n_eq : n = 8):
    (P * A_leg * n) / W_previous = 2.5 := by
  sorry

end giant_spider_weight_ratio_l24_24275


namespace total_snakes_l24_24384

/-
  Problem: Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes.
           How many snakes did she see total?
  Conditions:
    - There are 3 breeding balls.
    - Each breeding ball has 8 snakes.
    - There are 6 additional pairs of snakes.
  Answer: 36 snakes
-/

theorem total_snakes (balls : ℕ) (snakes_per_ball : ℕ) (pairs : ℕ) (snakes_per_pair : ℕ) :
    balls = 3 → snakes_per_ball = 8 → pairs = 6 → snakes_per_pair = 2 →
    (balls * snakes_per_ball) + (pairs * snakes_per_pair) = 36 :=
  by 
    intros hb hspb hp hsp
    sorry

end total_snakes_l24_24384


namespace remainder_of_sum_mod_13_l24_24941

theorem remainder_of_sum_mod_13 (a b c d e : ℕ) 
  (h1: a % 13 = 3) (h2: b % 13 = 5) (h3: c % 13 = 7) (h4: d % 13 = 9) (h5: e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := 
by 
  sorry

end remainder_of_sum_mod_13_l24_24941


namespace triangle_c_and_area_l24_24556

theorem triangle_c_and_area
  (a b : ℝ) (C : ℝ)
  (h_a : a = 1)
  (h_b : b = 2)
  (h_C : C = Real.pi / 3) :
  ∃ (c S : ℝ), c = Real.sqrt 3 ∧ S = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_c_and_area_l24_24556


namespace number_of_men_in_group_l24_24178

-- Define the conditions
variable (n : ℕ) -- number of men in the group
variable (A : ℝ) -- original average age of the group
variable (increase_in_years : ℝ := 2) -- the increase in the average age
variable (ages_before_replacement : ℝ := 21 + 23) -- total age of the men replaced
variable (ages_after_replacement : ℝ := 2 * 37) -- total age of the new men

-- Define the theorem using the conditions
theorem number_of_men_in_group 
  (h1 : n * increase_in_years = ages_after_replacement - ages_before_replacement) :
  n = 15 :=
sorry

end number_of_men_in_group_l24_24178


namespace least_n_l24_24563

theorem least_n (n : ℕ) (h : (1 : ℝ) / n - (1 / (n + 1)) < (1 / 15)) : n = 4 :=
sorry

end least_n_l24_24563


namespace part_a_l24_24412

-- Part (a)
theorem part_a (x : ℕ)  : (x^2 - x + 2) % 7 = 0 → x % 7 = 4 := by 
  sorry

end part_a_l24_24412


namespace alt_rep_of_set_l24_24024

def NatPos (x : ℕ) := x > 0

theorem alt_rep_of_set : {x : ℕ | NatPos x ∧ x - 3 < 2} = {1, 2, 3, 4} := by
  sorry

end alt_rep_of_set_l24_24024


namespace geometric_body_is_cylinder_l24_24142

def top_view_is_circle : Prop := sorry

def is_prism_or_cylinder : Prop := sorry

theorem geometric_body_is_cylinder 
  (h1 : top_view_is_circle) 
  (h2 : is_prism_or_cylinder) 
  : Cylinder := 
sorry

end geometric_body_is_cylinder_l24_24142


namespace age_of_15th_student_is_15_l24_24854

-- Define the total number of students
def total_students : Nat := 15

-- Define the average age of all 15 students together
def avg_age_all_students : Nat := 15

-- Define the average age of the first group of 7 students
def avg_age_first_group : Nat := 14

-- Define the average age of the second group of 7 students
def avg_age_second_group : Nat := 16

-- Define the total age based on the average age and number of students
def total_age_all_students : Nat := total_students * avg_age_all_students
def total_age_first_group : Nat := 7 * avg_age_first_group
def total_age_second_group : Nat := 7 * avg_age_second_group

-- Define the age of the 15th student
def age_of_15th_student : Nat := total_age_all_students - (total_age_first_group + total_age_second_group)

-- Theorem: prove that the age of the 15th student is 15 years
theorem age_of_15th_student_is_15 : age_of_15th_student = 15 := by
  -- The proof will go here
  sorry

end age_of_15th_student_is_15_l24_24854


namespace possible_permutations_100_l24_24198

def tasty_permutations (n : ℕ) : ℕ := sorry

theorem possible_permutations_100 :
  2^100 ≤ tasty_permutations 100 ∧ tasty_permutations 100 ≤ 4^100 :=
sorry

end possible_permutations_100_l24_24198


namespace club_co_presidents_l24_24954

def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem club_co_presidents : choose 18 3 = 816 := by
  sorry

end club_co_presidents_l24_24954


namespace area_of_original_rectangle_l24_24153

theorem area_of_original_rectangle 
  (L W : ℝ)
  (h1 : 2 * L * (3 * W) = 1800) :
  L * W = 300 :=
by
  sorry

end area_of_original_rectangle_l24_24153


namespace cargo_per_truck_is_2_5_l24_24122

-- Define our instance conditions
variables (x : ℝ) (n : ℕ)

-- Conditions extracted from the problem
def truck_capacity_change : Prop :=
  55 ≤ x ∧ x ≤ 64 ∧
  (x = (x / n - 0.5) * (n + 4))

-- Objective based on these conditions
theorem cargo_per_truck_is_2_5 :
  truck_capacity_change x n → (x = 60) → (n + 4 = 24) → (x / 24 = 2.5) :=
by 
  sorry

end cargo_per_truck_is_2_5_l24_24122


namespace tickets_per_friend_l24_24225

-- Defining the conditions
def initial_tickets := 11
def remaining_tickets := 3
def friends := 4

-- Statement to prove
theorem tickets_per_friend (h_tickets_given : initial_tickets - remaining_tickets = 8) : (initial_tickets - remaining_tickets) / friends = 2 :=
by
  sorry

end tickets_per_friend_l24_24225


namespace evaluate_polynomial_l24_24791

theorem evaluate_polynomial (x : ℝ) : x * (x * (x * (x - 3) - 5) + 9) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 9 * x + 2 := by
  sorry

end evaluate_polynomial_l24_24791


namespace ratio_c_d_l24_24720

theorem ratio_c_d (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -4 / 5 :=
by
  sorry

end ratio_c_d_l24_24720


namespace probability_of_at_least_six_heads_is_correct_l24_24599

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l24_24599


namespace transformed_system_solution_l24_24100

theorem transformed_system_solution :
  (∀ (a b : ℝ), 2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9 → a = 8.3 ∧ b = 1.2) →
  (∀ (x y : ℝ), 2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9 →
    x = 6.3 ∧ y = 2.2) :=
by
  intro h1
  intro x y
  intro hy
  sorry

end transformed_system_solution_l24_24100


namespace x_gt_1_sufficient_not_necessary_x_squared_gt_1_l24_24642

variable {x : ℝ}

-- Condition: $x > 1$
def condition_x_gt_1 (x : ℝ) : Prop := x > 1

-- Condition: $x^2 > 1$
def condition_x_squared_gt_1 (x : ℝ) : Prop := x^2 > 1

-- Theorem: Prove that $x > 1$ is a sufficient but not necessary condition for $x^2 > 1$
theorem x_gt_1_sufficient_not_necessary_x_squared_gt_1 :
  (condition_x_gt_1 x → condition_x_squared_gt_1 x) ∧ (¬ ∀ x, condition_x_squared_gt_1 x → condition_x_gt_1 x) :=
sorry

end x_gt_1_sufficient_not_necessary_x_squared_gt_1_l24_24642


namespace ellipse_domain_l24_24032

theorem ellipse_domain (m : ℝ) :
  (-1 < m ∧ m < 2 ∧ m ≠ 1 / 2) -> 
  ∃ a b : ℝ, (a = 2 - m) ∧ (b = m + 1) ∧ a > 0 ∧ b > 0 ∧ a ≠ b :=
by
  sorry

end ellipse_domain_l24_24032


namespace total_legs_in_christophers_room_l24_24601

def total_legs (num_spiders num_legs_per_spider num_ants num_butterflies num_beetles num_legs_per_insect : ℕ) : ℕ :=
  let spider_legs := num_spiders * num_legs_per_spider
  let ant_legs := num_ants * num_legs_per_insect
  let butterfly_legs := num_butterflies * num_legs_per_insect
  let beetle_legs := num_beetles * num_legs_per_insect
  spider_legs + ant_legs + butterfly_legs + beetle_legs

theorem total_legs_in_christophers_room : total_legs 12 8 10 5 5 6 = 216 := by
  -- Calculation and reasoning omitted
  sorry

end total_legs_in_christophers_room_l24_24601


namespace find_N_l24_24069

theorem find_N (x N : ℝ) (h1 : x + 1 / x = N) (h2 : x^2 + 1 / x^2 = 2) : N = 2 :=
sorry

end find_N_l24_24069


namespace find_larger_number_l24_24379

theorem find_larger_number (x y : ℝ) (h1 : x - y = 5) (h2 : 2 * (x + y) = 40) : x = 12.5 :=
by 
  sorry

end find_larger_number_l24_24379


namespace flight_time_is_10_hours_l24_24541

def time_watching_TV_episodes : ℕ := 3 * 25
def time_sleeping : ℕ := 4 * 60 + 30
def time_watching_movies : ℕ := 2 * (1 * 60 + 45)
def remaining_flight_time : ℕ := 45

def total_flight_time : ℕ := (time_watching_TV_episodes + time_sleeping + time_watching_movies + remaining_flight_time) / 60

theorem flight_time_is_10_hours : total_flight_time = 10 := by
  sorry

end flight_time_is_10_hours_l24_24541


namespace solve_for_k_l24_24781

theorem solve_for_k (k x : ℝ) (h₁ : 4 * k - 3 * x = 2) (h₂ : x = -1) : 
  k = -1 / 4 := 
by sorry

end solve_for_k_l24_24781


namespace product_of_repeating145_and_11_equals_1595_over_999_l24_24506

-- Defining the repeating decimal as a fraction
def repeating145_as_fraction : ℚ :=
  145 / 999

-- Stating the main theorem
theorem product_of_repeating145_and_11_equals_1595_over_999 :
  11 * repeating145_as_fraction = 1595 / 999 :=
by
  sorry

end product_of_repeating145_and_11_equals_1595_over_999_l24_24506


namespace find_ordered_pair_l24_24575

theorem find_ordered_pair (a b : ℚ) :
  a • (⟨2, 3⟩ : ℚ × ℚ) + b • (⟨-2, 5⟩ : ℚ × ℚ) = (⟨10, -8⟩ : ℚ × ℚ) →
  (a, b) = (17 / 8, -23 / 8) :=
by
  intro h
  sorry

end find_ordered_pair_l24_24575


namespace range_of_f_l24_24279

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2

theorem range_of_f : Set.range f = Set.Ici 3 :=
by
  sorry

end range_of_f_l24_24279


namespace train_crossing_time_l24_24745

theorem train_crossing_time (length_of_train : ℝ) (speed_kmh : ℝ) :
  length_of_train = 180 →
  speed_kmh = 72 →
  (180 / (72 * (1000 / 3600))) = 9 :=
by 
  intros h1 h2
  sorry

end train_crossing_time_l24_24745


namespace coeff_comparison_l24_24983

def a_k (k : ℕ) : ℕ := (2 ^ k) * Nat.choose 100 k

theorem coeff_comparison :
  (Finset.filter (fun r => a_k r < a_k (r + 1)) (Finset.range 100)).card = 67 :=
by
  sorry

end coeff_comparison_l24_24983


namespace probability_yellow_chalk_is_three_fifths_l24_24806

open Nat

theorem probability_yellow_chalk_is_three_fifths
  (yellow_chalks : ℕ) (red_chalks : ℕ) (total_chalks : ℕ)
  (h_yellow : yellow_chalks = 3) (h_red : red_chalks = 2) (h_total : total_chalks = yellow_chalks + red_chalks) :
  (yellow_chalks : ℚ) / (total_chalks : ℚ) = 3 / 5 := by
  sorry

end probability_yellow_chalk_is_three_fifths_l24_24806


namespace angle_addition_l24_24249

open Real

theorem angle_addition (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : tan α = 1 / 3) (h₄ : cos β = 3 / 5) : α + 3 * β = 3 * π / 4 :=
by
  sorry

end angle_addition_l24_24249


namespace joggers_meet_again_at_correct_time_l24_24434

-- Define the joggers and their lap times
def bob_lap_time := 3
def carol_lap_time := 5
def ted_lap_time := 8

-- Calculate the Least Common Multiple (LCM) of their lap times
def lcm_joggers := Nat.lcm (Nat.lcm bob_lap_time carol_lap_time) ted_lap_time

-- Start time is 9:00 AM
def start_time := 9 * 60  -- in minutes

-- The time (in minutes) we get back together is start_time plus the LCM
def earliest_meeting_time := start_time + lcm_joggers

-- Convert the meeting time to hours and minutes
def hours := earliest_meeting_time / 60
def minutes := earliest_meeting_time % 60

-- Define an expected result
def expected_meeting_hour := 11
def expected_meeting_minute := 0

-- Prove that all joggers will meet again at the correct time
theorem joggers_meet_again_at_correct_time :
  hours = expected_meeting_hour ∧ minutes = expected_meeting_minute :=
by
  -- Here you would provide the proof, but we'll use sorry for brevity
  sorry

end joggers_meet_again_at_correct_time_l24_24434


namespace boiling_point_C_l24_24938

-- Water boils at 212 °F
def water_boiling_point_F : ℝ := 212
-- Ice melts at 32 °F
def ice_melting_point_F : ℝ := 32
-- Ice melts at 0 °C
def ice_melting_point_C : ℝ := 0
-- The temperature of a pot of water in °C
def pot_water_temp_C : ℝ := 40
-- The temperature of the pot of water in °F
def pot_water_temp_F : ℝ := 104

-- The boiling point of water in Celsius is 100 °C.
theorem boiling_point_C : water_boiling_point_F = 212 ∧ ice_melting_point_F = 32 ∧ ice_melting_point_C = 0 ∧ pot_water_temp_C = 40 ∧ pot_water_temp_F = 104 → exists bp_C : ℝ, bp_C = 100 :=
by
  sorry

end boiling_point_C_l24_24938


namespace single_discount_equivalence_l24_24467

theorem single_discount_equivalence (original_price : ℝ) (first_discount second_discount : ℝ) (final_price : ℝ) :
  original_price = 50 →
  first_discount = 0.30 →
  second_discount = 0.10 →
  final_price = original_price * (1 - first_discount) * (1 - second_discount) →
  ((original_price - final_price) / original_price) * 100 = 37 := by
  sorry

end single_discount_equivalence_l24_24467


namespace angle_halving_quadrant_l24_24744

theorem angle_halving_quadrant (k : ℤ) (α : ℝ) 
  (h : k * 360 + 180 < α ∧ α < k * 360 + 270) : 
  k * 180 + 90 < α / 2 ∧ α / 2 < k * 180 + 135 :=
sorry

end angle_halving_quadrant_l24_24744


namespace minimize_expression_l24_24555

theorem minimize_expression (a b c d e f : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h_sum : a + b + c + d + e + f = 10) :
  (1 / a + 9 / b + 25 / c + 49 / d + 81 / e + 121 / f) ≥ 129.6 :=
by
  sorry

end minimize_expression_l24_24555


namespace point_in_fourth_quadrant_l24_24480

theorem point_in_fourth_quadrant (m : ℝ) : 0 < m ∧ 2 - m < 0 ↔ m > 2 := 
by 
  sorry

end point_in_fourth_quadrant_l24_24480


namespace perfect_square_trinomial_k_l24_24079

theorem perfect_square_trinomial_k (k : ℤ) : (∃ a b : ℤ, (a*x + b)^2 = x^2 + k*x + 9) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_k_l24_24079


namespace correct_option_l24_24828

-- Conditions
def option_A (a : ℝ) : Prop := a^3 + a^3 = a^6
def option_B (a : ℝ) : Prop := (a^3)^2 = a^9
def option_C (a : ℝ) : Prop := a^6 / a^3 = a^2
def option_D (a b : ℝ) : Prop := (a * b)^2 = a^2 * b^2

-- Proof Problem Statement
theorem correct_option (a b : ℝ) : option_D a b ↔ ¬option_A a ∧ ¬option_B a ∧ ¬option_C a :=
by
  sorry

end correct_option_l24_24828


namespace value_of_m_minus_n_l24_24074

theorem value_of_m_minus_n (m n : ℝ) (h : (-3)^2 + m * (-3) + 3 * n = 0) : m - n = 3 :=
sorry

end value_of_m_minus_n_l24_24074


namespace pirates_total_coins_l24_24005

theorem pirates_total_coins :
  ∀ (x : ℕ), (∃ (paul_coins pete_coins : ℕ), 
  paul_coins = x ∧ pete_coins = 5 * x ∧ pete_coins = (x * (x + 1)) / 2) → x + 5 * x = 54 := by
  sorry

end pirates_total_coins_l24_24005


namespace range_a_monotonically_increasing_l24_24167

def g (a x : ℝ) : ℝ := a * x^3 + a * x^2 + x

theorem range_a_monotonically_increasing (a : ℝ) : 
  (∀ x : ℝ, 3 * a * x^2 + 2 * a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 3) := 
sorry

end range_a_monotonically_increasing_l24_24167


namespace sample_size_is_150_l24_24335

-- Define the conditions
def total_parents : ℕ := 823
def sampled_parents : ℕ := 150
def negative_attitude_parents : ℕ := 136

-- State the theorem
theorem sample_size_is_150 : sampled_parents = 150 := 
by
  sorry

end sample_size_is_150_l24_24335


namespace gasoline_price_increase_l24_24653

theorem gasoline_price_increase 
  (P Q : ℝ)
  (h_intends_to_spend : ∃ M, M = P * Q * 1.15)
  (h_reduction : ∃ N, N = Q * (1 - 0.08))
  (h_equation : P * Q * 1.15 = P * (1 + x) * Q * (1 - 0.08)) :
  x = 0.25 :=
by
  sorry

end gasoline_price_increase_l24_24653


namespace num_cars_can_be_parked_l24_24749

theorem num_cars_can_be_parked (length width : ℝ) (useable_percentage : ℝ) (area_per_car : ℝ) 
  (h_length : length = 400) (h_width : width = 500) (h_useable_percentage : useable_percentage = 0.80) 
  (h_area_per_car : area_per_car = 10) : 
  length * width * useable_percentage / area_per_car = 16000 := 
by 
  sorry

end num_cars_can_be_parked_l24_24749


namespace nurses_count_l24_24679

theorem nurses_count (total_medical_staff : ℕ) (ratio_doctors : ℕ) (ratio_nurses : ℕ) 
  (total_ratio_parts : ℕ) (h1 : total_medical_staff = 200) 
  (h2 : ratio_doctors = 4) (h3 : ratio_nurses = 6) (h4 : total_ratio_parts = ratio_doctors + ratio_nurses) :
  (ratio_nurses * total_medical_staff) / total_ratio_parts = 120 :=
by
  sorry

end nurses_count_l24_24679


namespace sequence_properties_l24_24328

variable {a : ℕ → ℤ}

-- Conditions
axiom seq_add : ∀ (p q : ℕ), 1 ≤ p → 1 ≤ q → a (p + q) = a p + a q
axiom a2_neg4 : a 2 = -4

-- Theorem statement: We need to prove a6 = -12 and a_n = -2n for all n
theorem sequence_properties :
  (a 6 = -12) ∧ ∀ n : ℕ, 1 ≤ n → a n = -2 * n :=
by
  sorry

end sequence_properties_l24_24328


namespace inequality_350_l24_24735

theorem inequality_350 (a b c d : ℝ) : 
  (a - b) * (b - c) * (c - d) * (d - a) + (a - c)^2 * (b - d)^2 ≥ 0 :=
by
  sorry

end inequality_350_l24_24735


namespace range_a_l24_24778

noncomputable def f (x : ℝ) : ℝ := -(1 / 3) * x^3 + x

theorem range_a (a : ℝ) (h1 : a < 1) (h2 : 1 < 10 - a^2) (h3 : f a ≤ f 1) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_a_l24_24778


namespace nest_building_twig_count_l24_24630

theorem nest_building_twig_count
    (total_twigs_to_weave : ℕ)
    (found_twigs : ℕ)
    (remaining_twigs : ℕ)
    (n : ℕ)
    (x : ℕ)
    (h1 : total_twigs_to_weave = 12 * x)
    (h2 : found_twigs = (total_twigs_to_weave) / 3)
    (h3 : remaining_twigs = 48)
    (h4 : found_twigs + remaining_twigs = total_twigs_to_weave) :
    x = 18 := 
by
  sorry

end nest_building_twig_count_l24_24630


namespace total_area_rectABCD_l24_24060

theorem total_area_rectABCD (BF CF : ℝ) (X Y : ℝ)
  (h1 : BF = 3 * CF)
  (h2 : 3 * X - Y - (X - Y) = 96)
  (h3 : X + 3 * X = 192) :
  X + 3 * X = 192 :=
by
  sorry

end total_area_rectABCD_l24_24060


namespace minimum_value_expression_l24_24916

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ k, k = 729 ∧ ∀ x y z, 0 < x → 0 < y → 0 < z → k ≤ (x^2 + 4*x + 4) * (y^2 + 4*y + 4) * (z^2 + 4*z + 4) / (x * y * z) :=
by 
  use 729
  sorry

end minimum_value_expression_l24_24916


namespace mrs_peterson_change_l24_24591

def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

theorem mrs_peterson_change : 
  (num_bills * value_per_bill) - (num_tumblers * cost_per_tumbler) = 50 :=
by
  sorry

end mrs_peterson_change_l24_24591


namespace volume_of_rectangular_solid_l24_24902

theorem volume_of_rectangular_solid (x y z : ℝ) 
  (h1 : x * y = 18) 
  (h2 : y * z = 15) 
  (h3 : z * x = 10) : 
  x * y * z = 30 * Real.sqrt 3 := 
sorry

end volume_of_rectangular_solid_l24_24902


namespace medium_pizza_slices_l24_24602

theorem medium_pizza_slices (M : ℕ) 
  (small_pizza_slices : ℕ := 6)
  (large_pizza_slices : ℕ := 12)
  (total_pizzas : ℕ := 15)
  (small_pizzas : ℕ := 4)
  (medium_pizzas : ℕ := 5)
  (total_slices : ℕ := 136) :
  (small_pizzas * small_pizza_slices) + (medium_pizzas * M) + ((total_pizzas - small_pizzas - medium_pizzas) * large_pizza_slices) = total_slices → 
  M = 8 :=
by
  intro h
  sorry

end medium_pizza_slices_l24_24602


namespace solution_set_inequality_l24_24370

theorem solution_set_inequality (x : ℝ) : |5 - x| < |x - 2| + |7 - 2 * x| ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 3.5 :=
by
  sorry

end solution_set_inequality_l24_24370


namespace eval_expression_l24_24665

theorem eval_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ x) (hz' : z ≠ -x) :
  ((x / (x + z) + z / (x - z)) / (z / (x + z) - x / (x - z)) = -1) :=
by
  sorry

end eval_expression_l24_24665


namespace probability_non_adjacent_l24_24124

def total_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n m 

def non_adjacent_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n (m - 1)

def probability_zeros_non_adjacent (n m : ℕ) : ℚ :=
  (non_adjacent_arrangements n m : ℚ) / (total_arrangements n m : ℚ)

theorem probability_non_adjacent (a b : ℕ) (h₁ : a = 4) (h₂ : b = 2) :
  probability_zeros_non_adjacent 5 2 = 2 / 3 := 
by 
  rw [probability_zeros_non_adjacent]
  rw [non_adjacent_arrangements, total_arrangements]
  sorry

end probability_non_adjacent_l24_24124


namespace new_quadratic_equation_has_square_roots_l24_24608

theorem new_quadratic_equation_has_square_roots (p q : ℝ) (x : ℝ) :
  (x^2 + px + q = 0 → ∃ x1 x2 : ℝ, x^2 - (p^2 - 2 * q) * x + q^2 = 0 ∧ (x1^2 = x ∨ x2^2 = x)) :=
by sorry

end new_quadratic_equation_has_square_roots_l24_24608


namespace solve_for_a_l24_24397

theorem solve_for_a (a x : ℝ) (h : x^2 + a * x + 4 = (x + 2)^2) : a = 4 :=
by sorry

end solve_for_a_l24_24397


namespace magician_card_trick_l24_24292

-- Definitions and proof goal
theorem magician_card_trick :
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 :=
by
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  have h : (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 := sorry
  exact h

end magician_card_trick_l24_24292


namespace range_of_a_l24_24995

def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, 1 ≤ x ∧ x ≤ y → quadratic_function a x ≤ quadratic_function a y) : a ≤ 1 :=
sorry

end range_of_a_l24_24995


namespace solve_inequality_l24_24175

theorem solve_inequality (x : ℝ) :
  (x - 2) / (x + 5) ≤ 1 / 2 ↔ x ∈ Set.Ioc (-5 : ℝ) 9 :=
by
  sorry

end solve_inequality_l24_24175


namespace equivalent_solution_eq1_eqC_l24_24127

-- Define the given equation
def eq1 (x y : ℝ) : Prop := 4 * x - 8 * y - 5 = 0

-- Define the candidate equations
def eqA (x y : ℝ) : Prop := 8 * x - 8 * y - 10 = 0
def eqB (x y : ℝ) : Prop := 8 * x - 16 * y - 5 = 0
def eqC (x y : ℝ) : Prop := 8 * x - 16 * y - 10 = 0
def eqD (x y : ℝ) : Prop := 12 * x - 24 * y - 10 = 0

-- The theorem that we need to prove
theorem equivalent_solution_eq1_eqC : ∀ x y, eq1 x y ↔ eqC x y :=
by
  sorry

end equivalent_solution_eq1_eqC_l24_24127


namespace evaluate_expression_l24_24396

theorem evaluate_expression :
  2^4 - 4 * 2^3 + 6 * 2^2 - 4 * 2 + 1 = 1 :=
by
  sorry

end evaluate_expression_l24_24396


namespace find_x2_plus_y2_l24_24722

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + 2 * x + 2 * y = 88) :
    x^2 + y^2 = 304 / 9 := sorry

end find_x2_plus_y2_l24_24722


namespace find_digits_l24_24875

def five_digit_subtraction (a b c d e : ℕ) : Prop :=
    let n1 := 10000 * a + 1000 * b + 100 * c + 10 * d + e
    let n2 := 10000 * e + 1000 * d + 100 * c + 10 * b + a
    (n1 - n2) % 10 = 2 ∧ (((n1 - n2) / 10) % 10) = 7 ∧ a > e ∧ a - e = 2 ∧ b - a = 7

theorem find_digits 
    (a b c d e : ℕ) 
    (h : five_digit_subtraction a b c d e) :
    a = 9 ∧ e = 7 :=
by 
    sorry

end find_digits_l24_24875


namespace intersection_point_exists_l24_24918

theorem intersection_point_exists :
  ∃ t u x y : ℚ,
    (x = 2 + 3 * t) ∧ (y = 3 - 4 * t) ∧
    (x = 4 + 5 * u) ∧ (y = -6 + u) ∧
    (x = 175 / 23) ∧ (y = 19 / 23) :=
by
  sorry

end intersection_point_exists_l24_24918


namespace ellipse_equation_and_fixed_point_proof_l24_24445

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l24_24445


namespace hyperbola_eccentricity_l24_24723

-- Define the conditions given in the problem
def asymptote_equation_related (a b : ℝ) : Prop := a / b = 3 / 4
def hyperbola_eccentricity_relation (a c : ℝ) : Prop := c^2 / a^2 = 25 / 9

-- Define the proof problem
theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : asymptote_equation_related a b)
  (h2 : hyperbola_eccentricity_relation a c)
  (he : e = c / a) :
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l24_24723


namespace counterexample_statement_l24_24644

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n

theorem counterexample_statement (n : ℕ) : is_composite n ∧ (is_prime (n - 3) ∨ is_prime (n - 2)) ↔ n = 22 :=
by
  sorry

end counterexample_statement_l24_24644


namespace triangle_side_length_l24_24270

theorem triangle_side_length 
  (r : ℝ)                    -- radius of the inscribed circle
  (h_cos_ABC : ℝ)            -- cosine of angle ABC
  (h_midline : Bool)         -- the circle touches the midline parallel to AC
  (h_r : r = 1)              -- given radius is 1
  (h_cos : h_cos_ABC = 0.8)  -- given cos(ABC) = 0.8
  (h_touch : h_midline = true)  -- given that circle touches the midline
  : AC = 3 := 
sorry

end triangle_side_length_l24_24270


namespace cone_base_radius_l24_24374

open Real

theorem cone_base_radius (r_sector : ℝ) (θ_sector : ℝ) : 
    r_sector = 6 ∧ θ_sector = 120 → (∃ r : ℝ, 2 * π * r = θ_sector * π * r_sector / 180 ∧ r = 2) :=
by
  sorry

end cone_base_radius_l24_24374


namespace max_value_m_n_squared_sum_l24_24174

theorem max_value_m_n_squared_sum (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m * n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end max_value_m_n_squared_sum_l24_24174


namespace compute_expression_l24_24340

theorem compute_expression : 3 * 3^3 - 9^50 / 9^48 = 0 := by
  sorry

end compute_expression_l24_24340


namespace remainder_when_divided_by_9_l24_24144

theorem remainder_when_divided_by_9 (x : ℕ) (h : 4 * x % 9 = 2) : x % 9 = 5 :=
by sorry

end remainder_when_divided_by_9_l24_24144


namespace sugar_solution_sweeter_l24_24196

variables (a b m : ℝ)

theorem sugar_solution_sweeter (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : 
  (a / b < (a + m) / (b + m)) :=
sorry

end sugar_solution_sweeter_l24_24196


namespace price_of_turban_l24_24354

theorem price_of_turban : 
  ∃ T : ℝ, (9 / 12) * (90 + T) = 40 + T ↔ T = 110 :=
by
  sorry

end price_of_turban_l24_24354


namespace relationship_among_a_b_c_l24_24986

theorem relationship_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (4 : ℝ) ^ (1 / 2))
  (hb : b = (2 : ℝ) ^ (1 / 3))
  (hc : c = (5 : ℝ) ^ (1 / 2))
: b < a ∧ a < c := 
sorry

end relationship_among_a_b_c_l24_24986


namespace solve_inequality_l24_24494

theorem solve_inequality : {x : ℝ | 3 * x ^ 2 - 7 * x - 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
sorry

end solve_inequality_l24_24494


namespace inequality_solution_l24_24527

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) → a ≥ 2 :=
by
  sorry

end inequality_solution_l24_24527


namespace machine_does_not_require_repair_l24_24813

-- Define the conditions.

def max_deviation := 37

def nominal_portion_max_deviation_percentage := 0.10

def deviation_within_limit (M : ℝ) : Prop :=
  37 ≤ 0.10 * M

def unreadable_measurements_deviation (deviation : ℝ) : Prop :=
  deviation < 37

-- Define the theorem we want to prove

theorem machine_does_not_require_repair (M : ℝ)
  (h1 : deviation_within_limit M)
  (h2 : ∀ deviation, unreadable_measurements_deviation deviation) :
  true := 
sorry

end machine_does_not_require_repair_l24_24813


namespace part_a_roots_part_b_sum_l24_24386

theorem part_a_roots : ∀ x : ℝ, 2^x = x + 1 ↔ x = 0 ∨ x = 1 :=
by 
  intros x
  sorry

theorem part_b_sum (f : ℝ → ℝ) (h : ∀ x : ℝ, (f ∘ f) x = 2^x - 1) : f 0 + f 1 = 1 :=
by 
  sorry

end part_a_roots_part_b_sum_l24_24386


namespace total_cookies_in_box_l24_24422

-- Definitions from the conditions
def oldest_son_cookies : ℕ := 4
def youngest_son_cookies : ℕ := 2
def days_box_lasts : ℕ := 9

-- Total cookies consumed per day
def daily_cookies_consumption : ℕ := oldest_son_cookies + youngest_son_cookies

-- Theorem statement: total number of cookies in the box
theorem total_cookies_in_box : (daily_cookies_consumption * days_box_lasts) = 54 := by
  sorry

end total_cookies_in_box_l24_24422


namespace ratio_of_a_to_b_l24_24293

theorem ratio_of_a_to_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
    (h_x : x = 1.25 * a) (h_m : m = 0.40 * b) (h_ratio : m / x = 0.4) 
    : (a / b) = 4 / 5 := by
  sorry

end ratio_of_a_to_b_l24_24293


namespace largest_value_f12_l24_24554

theorem largest_value_f12 (f : ℝ → ℝ) (hf_poly : ∀ x, f x ≥ 0) 
  (hf_6 : f 6 = 24) (hf_24 : f 24 = 1536) :
  f 12 ≤ 192 :=
sorry

end largest_value_f12_l24_24554


namespace total_spots_l24_24507

-- Define the variables
variables (R C G S B : ℕ)

-- State the problem conditions
def conditions : Prop :=
  R = 46 ∧
  C = R / 2 - 5 ∧
  G = 5 * C ∧
  S = 3 * R ∧
  B = 2 * (G + S)

-- State the proof problem
theorem total_spots : conditions R C G S B → G + C + S + B = 702 :=
by
  intro h
  obtain ⟨hR, hC, hG, hS, hB⟩ := h
  -- The proof steps would go here
  sorry

end total_spots_l24_24507


namespace correct_operation_l24_24634

theorem correct_operation (a b : ℝ) :
  (a + b) * (b - a) = b^2 - a^2 :=
by
  sorry

end correct_operation_l24_24634


namespace correct_commutative_property_usage_l24_24264

-- Definitions for the transformations
def transformA := 3 + (-2) = 2 + 3
def transformB := 4 + (-6) + 3 = (-6) + 4 + 3
def transformC := (5 + (-2)) + 4 = (5 + (-4)) + 2
def transformD := (1 / 6) + (-1) + (5 / 6) = ((1 / 6) + (5 / 6)) + 1

-- The theorem stating that transformB uses the commutative property correctly
theorem correct_commutative_property_usage : transformB :=
by
  sorry

end correct_commutative_property_usage_l24_24264


namespace inverse_g_of_87_l24_24497

noncomputable def g (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_g_of_87 : (g x = 87) → (x = 3) :=
by
  intro h
  sorry

end inverse_g_of_87_l24_24497


namespace economical_club_l24_24097

-- Definitions of cost functions for Club A and Club B
def f (x : ℕ) : ℕ := 5 * x

def g (x : ℕ) : ℕ := if x ≤ 30 then 90 else 2 * x + 30

-- Theorem to determine the more economical club
theorem economical_club (x : ℕ) (hx : 15 ≤ x ∧ x ≤ 40) :
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 30 → f x > g x) ∧
  (30 < x ∧ x ≤ 40 → f x > g x) :=
sorry

end economical_club_l24_24097


namespace Serezha_puts_more_berries_l24_24294

theorem Serezha_puts_more_berries (berries : ℕ) 
    (Serezha_puts : ℕ) (Serezha_eats : ℕ)
    (Dima_puts : ℕ) (Dima_eats : ℕ)
    (Serezha_rate : ℕ) (Dima_rate : ℕ)
    (total_berries : berries = 450)
    (Serezha_pattern : Serezha_puts = 1 ∧ Serezha_eats = 1)
    (Dima_pattern : Dima_puts = 2 ∧ Dima_eats = 1)
    (Serezha_faster : Serezha_rate = 2 * Dima_rate) : 
    ∃ (Serezha_in_basket : ℕ) (Dima_in_basket : ℕ),
      Serezha_in_basket > Dima_in_basket ∧ Serezha_in_basket - Dima_in_basket = 50 :=
by
  sorry -- Skip the proof

end Serezha_puts_more_berries_l24_24294


namespace product_of_real_roots_l24_24139

theorem product_of_real_roots : 
  (∃ x y : ℝ, (x ^ Real.log x = Real.exp 1) ∧ (y ^ Real.log y = Real.exp 1) ∧ x ≠ y ∧ x * y = 1) :=
by
  sorry

end product_of_real_roots_l24_24139


namespace reciprocal_of_2_l24_24103

theorem reciprocal_of_2 : 1 / 2 = 1 / (2 : ℝ) := by
  sorry

end reciprocal_of_2_l24_24103


namespace no_pos_int_sol_l24_24194

theorem no_pos_int_sol (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : ¬ ∃ (k : ℕ), (15 * a + b) * (a + 15 * b) = 3^k := 
sorry

end no_pos_int_sol_l24_24194


namespace justin_current_age_l24_24492

theorem justin_current_age (angelina_future_age : ℕ) (years_until_future : ℕ) (age_difference : ℕ)
  (h_future_age : angelina_future_age = 40) (h_years_until_future : years_until_future = 5)
  (h_age_difference : age_difference = 4) : 
  (angelina_future_age - years_until_future) - age_difference = 31 :=
by
  -- This is where the proof would go.
  sorry

end justin_current_age_l24_24492


namespace second_pump_drain_time_l24_24637

-- Definitions of the rates R1 and R2
def R1 : ℚ := 1 / 12  -- Rate of the first pump
def R2 : ℚ := 1 - R1  -- Rate of the second pump (from the combined rate equation)

-- The time it takes the second pump alone to drain the pond
def time_to_drain_second_pump := 1 / R2

-- The goal is to prove that this value is 12/11
theorem second_pump_drain_time : time_to_drain_second_pump = 12 / 11 := by
  -- The proof is omitted
  sorry

end second_pump_drain_time_l24_24637


namespace students_answered_both_correctly_l24_24976

theorem students_answered_both_correctly 
  (total_students : ℕ) (took_test : ℕ) 
  (q1_correct : ℕ) (q2_correct : ℕ)
  (did_not_take_test : ℕ)
  (h1 : total_students = 25)
  (h2 : q1_correct = 22)
  (h3 : q2_correct = 20)
  (h4 : did_not_take_test = 3)
  (h5 : took_test = total_students - did_not_take_test) :
  (q1_correct + q2_correct) - took_test = 20 := 
by 
  -- Proof skipped.
  sorry

end students_answered_both_correctly_l24_24976


namespace solve_for_x_l24_24254

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
sorry

end solve_for_x_l24_24254


namespace solve_inequality_l24_24919

open Real

noncomputable def expression (x : ℝ) : ℝ :=
  (sqrt (x^2 - 4*x + 3) + 1) * log x / (log 2 * 5) + (1 / x) * (sqrt (8 * x - 2 * x^2 - 6) + 1)

theorem solve_inequality :
  ∃ x : ℝ, x = 1 ∧
    (x > 0) ∧
    (x^2 - 4 * x + 3 ≥ 0) ∧
    (8 * x - 2 * x^2 - 6 ≥ 0) ∧
    expression x ≤ 0 :=
by
  sorry

end solve_inequality_l24_24919


namespace line_equation_l24_24337

variable (θ : ℝ) (b : ℝ) (y x : ℝ)

-- Conditions: 
-- Slope angle θ = 45°
def slope_angle_condition : θ = 45 := by
  sorry

-- Y-intercept b = 2
def y_intercept_condition : b = 2 := by
  sorry

-- Given these conditions, we want to prove the line equation
theorem line_equation (x : ℝ) (θ : ℝ) (b : ℝ) :
  θ = 45 → b = 2 → y = x + 2 := by
  sorry

end line_equation_l24_24337


namespace negation_correct_l24_24075

variable {α : Type*} (A B : Set α)

-- Define the original proposition
def original_proposition : Prop := A ∪ B = A → A ∩ B = B

-- Define the negation of the original proposition
def negation_proposition : Prop := A ∪ B ≠ A → A ∩ B ≠ B

-- State that the negation of the original proposition is equivalent to the negation proposition
theorem negation_correct : ¬(original_proposition A B) ↔ negation_proposition A B := by sorry

end negation_correct_l24_24075


namespace series_sum_l24_24195

open BigOperators

theorem series_sum :
  (∑ n in Finset.range 99, (1 : ℝ) / ((n + 1) * (n + 2))) = 99 / 100 :=
by
  sorry

end series_sum_l24_24195


namespace greatest_five_digit_number_sum_of_digits_l24_24988

def is_five_digit_number (n : ℕ) : Prop :=
  10000 <= n ∧ n < 100000

def digits_product (n : ℕ) : ℕ :=
  (n % 10) * ((n / 10) % 10) * ((n / 100) % 10) * ((n / 1000) % 10) * (n / 10000)

def digits_sum (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10) + (n / 10000)

theorem greatest_five_digit_number_sum_of_digits (M : ℕ) 
  (h1 : is_five_digit_number M) 
  (h2 : digits_product M = 210) :
  digits_sum M = 20 := 
sorry

end greatest_five_digit_number_sum_of_digits_l24_24988


namespace three_lines_l24_24348

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem three_lines (x y : ℝ) : (diamond x y = diamond y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) := 
by sorry

end three_lines_l24_24348


namespace darcy_commute_l24_24468

theorem darcy_commute (d w r t x time_walk train_time : ℝ) 
  (h1 : d = 1.5)
  (h2 : w = 3)
  (h3 : r = 20)
  (h4 : train_time = t + x)
  (h5 : time_walk = 15 + train_time)
  (h6 : time_walk = d / w * 60)  -- Time taken to walk in minutes
  (h7 : t = d / r * 60)  -- Time taken on train in minutes
  : x = 10.5 :=
sorry

end darcy_commute_l24_24468


namespace unique_intersection_point_l24_24760

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (Real.log 3 / Real.log x)
noncomputable def h (x : ℝ) : ℝ := 3 - (1 / Real.sqrt (Real.log 3 / Real.log x))

theorem unique_intersection_point : (∃! (x : ℝ), (x > 0) ∧ (f x = g x ∨ f x = h x ∨ g x = h x)) :=
sorry

end unique_intersection_point_l24_24760


namespace minimum_b_l24_24082

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

noncomputable def g (a b x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ a then f a b x else f a b (f a b x)

theorem minimum_b {a b : ℝ} (ha : 0 < a) :
  (∀ x : ℝ, 0 ≤ x → g a b x > g a b (x - 1)) → b ≥ 1 / 4 :=
sorry

end minimum_b_l24_24082


namespace only_one_passes_prob_l24_24913

variable (P_A P_B P_C : ℚ)
variable (only_one_passes : ℚ)

def prob_A := 4 / 5 
def prob_B := 3 / 5
def prob_C := 7 / 10

def prob_only_A := prob_A * (1 - prob_B) * (1 - prob_C)
def prob_only_B := (1 - prob_A) * prob_B * (1 - prob_C)
def prob_only_C := (1 - prob_A) * (1 - prob_B) * prob_C

def prob_sum : ℚ := prob_only_A + prob_only_B + prob_only_C

theorem only_one_passes_prob : prob_sum = 47 / 250 := 
by sorry

end only_one_passes_prob_l24_24913


namespace contrapositive_equiv_l24_24342

variable (a b : ℝ)

def original_proposition : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

def contrapositive_proposition : Prop := a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0

theorem contrapositive_equiv : original_proposition a b ↔ contrapositive_proposition a b :=
by
  sorry

end contrapositive_equiv_l24_24342


namespace complex_number_in_first_quadrant_l24_24799

open Complex

theorem complex_number_in_first_quadrant 
    (z : ℂ) 
    (h : z = (3 + I) / (1 - 3 * I) + 2) : 
    0 < z.re ∧ 0 < z.im :=
by
  sorry

end complex_number_in_first_quadrant_l24_24799


namespace find_algebraic_expression_value_l24_24609

theorem find_algebraic_expression_value (x : ℝ) (h : 3 * x^2 + 5 * x + 1 = 0) : 
  (x + 2) ^ 2 + x * (2 * x + 1) = 3 := 
by 
  -- Proof steps go here
  sorry

end find_algebraic_expression_value_l24_24609


namespace binary_multiplication_correct_l24_24579

theorem binary_multiplication_correct:
  let n1 := 29 -- binary 11101 is decimal 29
  let n2 := 13 -- binary 1101 is decimal 13
  let result := 303 -- binary 100101111 is decimal 303
  n1 * n2 = result :=
by
  -- Proof goes here
  sorry

end binary_multiplication_correct_l24_24579


namespace baez_marble_loss_l24_24495

theorem baez_marble_loss :
  ∃ p : ℚ, (p > 0 ∧ (p / 100) * 25 * 2 = 60) ∧ p = 20 :=
by
  sorry

end baez_marble_loss_l24_24495


namespace tan_5105_eq_tan_85_l24_24804

noncomputable def tan_deg (d : ℝ) := Real.tan (d * Real.pi / 180)

theorem tan_5105_eq_tan_85 :
  tan_deg 5105 = tan_deg 85 := by
  have eq_265 : tan_deg 5105 = tan_deg 265 := by sorry
  have eq_neg : tan_deg 265 = tan_deg 85 := by sorry
  exact Eq.trans eq_265 eq_neg

end tan_5105_eq_tan_85_l24_24804


namespace find_y_l24_24274

-- Definitions for the given conditions
variable (p y : ℕ) (h : p > 30)  -- Natural numbers, noting p > 30 condition

-- The initial amount of acid in ounces
def initial_acid_amount : ℕ := p * p / 100

-- The amount of acid after adding y ounces of water
def final_acid_amount : ℕ := (p - 15) * (p + y) / 100

-- Lean statement to prove y = 15p/(p-15)
theorem find_y (h1 : p > 30) (h2 : initial_acid_amount p = final_acid_amount p y) :
  y = 15 * p / (p - 15) :=
sorry

end find_y_l24_24274


namespace train_speed_is_54_kmh_l24_24696

noncomputable def train_length_m : ℝ := 285
noncomputable def train_length_km : ℝ := train_length_m / 1000
noncomputable def time_seconds : ℝ := 19
noncomputable def time_hours : ℝ := time_seconds / 3600
noncomputable def speed : ℝ := train_length_km / time_hours

theorem train_speed_is_54_kmh :
  speed = 54 := by
sorry

end train_speed_is_54_kmh_l24_24696


namespace balloons_left_l24_24247

theorem balloons_left (yellow blue pink violet friends : ℕ) (total_balloons remainder : ℕ) 
  (hy : yellow = 20) (hb : blue = 24) (hp : pink = 50) (hv : violet = 102) (hf : friends = 9)
  (ht : total_balloons = yellow + blue + pink + violet) (hr : total_balloons % friends = remainder) : 
  remainder = 7 :=
by
  sorry

end balloons_left_l24_24247


namespace wire_not_used_l24_24421

variable (total_wire length_cut_parts parts_used : ℕ)

theorem wire_not_used (h1 : total_wire = 50) (h2 : length_cut_parts = 5) (h3 : parts_used = 3) : 
  total_wire - (parts_used * (total_wire / length_cut_parts)) = 20 := 
  sorry

end wire_not_used_l24_24421


namespace prop_logic_example_l24_24408

theorem prop_logic_example (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end prop_logic_example_l24_24408


namespace num_ways_two_different_colors_l24_24981

theorem num_ways_two_different_colors 
  (red white blue : ℕ) 
  (total_balls : ℕ) 
  (choose : ℕ → ℕ → ℕ) 
  (h_red : red = 2) 
  (h_white : white = 3) 
  (h_blue : blue = 1) 
  (h_total : total_balls = red + white + blue) 
  (h_choose_total : choose total_balls 3 = 20)
  (h_choose_three_diff_colors : 2 * 3 * 1 = 6)
  (h_one_color : 1 = 1) :
  choose total_balls 3 - 6 - 1 = 13 := 
by
  sorry

end num_ways_two_different_colors_l24_24981


namespace circle_condition_l24_24261

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - x + y + m = 0 → (m < 1 / 2)) :=
by {
-- Skipping the proof here
sorry
}

end circle_condition_l24_24261


namespace oil_leakage_during_repair_l24_24276

variables (initial_leak: ℚ) (initial_hours: ℚ) (repair_hours: ℚ) (reduction: ℚ) (total_leak: ℚ)

theorem oil_leakage_during_repair
    (h1 : initial_leak = 2475)
    (h2 : initial_hours = 7)
    (h3 : repair_hours = 5)
    (h4 : reduction = 0.75)
    (h5 : total_leak = 6206) :
    (total_leak - initial_leak = 3731) :=
by
  sorry

end oil_leakage_during_repair_l24_24276


namespace triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l24_24801

/-- Points \(P, Q, R, S\) are distinct, collinear, and ordered on a line with line segment lengths \( a, b, c \)
    such that \(a = PQ\), \(b = PR\), \(c = PS\). After rotating \(PQ\) and \(RS\) to make \( P \) and \( S \) coincide
    and form a triangle with a positive area, we must show:
    \(I. a < \frac{c}{3}\) must be satisfied in accordance to the triangle inequality revelations -/
theorem triangle_inequality_necessary_conditions (a b c : ℝ)
  (h_abc1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : b > c - b ∧ c > a ∧ c > b - a) :
  a < c / 3 :=
sorry

theorem triangle_inequality_sufficient_conditions (a b c : ℝ)
  (h_abc2 : b ≥ c / 3 ∧ a < c ∧ 2 * b ≤ c) :
  ¬ b < c / 3 :=
sorry

end triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l24_24801


namespace johns_profit_l24_24658

def profit (n : ℕ) (p c : ℕ) : ℕ :=
  n * p - c

theorem johns_profit :
  profit 20 15 100 = 200 :=
by
  sorry

end johns_profit_l24_24658


namespace tarun_garden_area_l24_24156

theorem tarun_garden_area :
  ∀ (side : ℝ), 
  (1500 / 8 = 4 * side) → 
  (30 * side = 1500) → 
  side^2 = 2197.265625 :=
by
  sorry

end tarun_garden_area_l24_24156


namespace hyperbola_focus_coordinates_l24_24257

theorem hyperbola_focus_coordinates : 
  ∃ (x y : ℝ), -2 * x^2 + 3 * y^2 + 8 * x - 18 * y - 8 = 0 ∧ (x, y) = (2, 7.5) :=
sorry

end hyperbola_focus_coordinates_l24_24257


namespace reflect_center_is_image_center_l24_24399

def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem reflect_center_is_image_center : 
  reflect_over_y_eq_neg_x (3, -4) = (4, -3) :=
by
  -- Proof is omitted as per instructions.
  -- This proof would show the reflection of the point (3, -4) over the line y = -x resulting in (4, -3).
  sorry

end reflect_center_is_image_center_l24_24399


namespace find_k_for_parallel_vectors_l24_24957

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem find_k_for_parallel_vectors 
  (h_a : a = (2, -1)) 
  (h_b : b = (1, 1)) 
  (h_c : c = (-5, 1)) 
  (h_parallel : vector_parallel (a.1 + k * b.1, a.2 + k * b.2) c) : 
  k = 1 / 2 :=
by
  unfold vector_parallel at h_parallel
  simp at h_parallel
  sorry

end find_k_for_parallel_vectors_l24_24957


namespace problem_conditions_l24_24158

theorem problem_conditions (a : ℕ → ℤ) :
  (1 + x)^6 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 →
  a 6 = 1 ∧ a 1 + a 3 + a 5 = -364 :=
by sorry

end problem_conditions_l24_24158


namespace slant_asymptote_and_sum_of_slope_and_intercept_l24_24964

noncomputable def f (x : ℚ) : ℚ := (3 * x^2 + 5 * x + 1) / (x + 2)

theorem slant_asymptote_and_sum_of_slope_and_intercept :
  (∀ x : ℚ, ∃ (m b : ℚ), (∃ r : ℚ, (r = f x ∧ (r + (m * x + b)) = f x)) ∧ m = 3 ∧ b = -1) →
  3 - 1 = 2 :=
by
  sorry

end slant_asymptote_and_sum_of_slope_and_intercept_l24_24964


namespace John_age_l24_24568

theorem John_age (Drew Maya Peter John Jacob : ℕ)
  (h1 : Drew = Maya + 5)
  (h2 : Peter = Drew + 4)
  (h3 : John = 2 * Maya)
  (h4 : (Jacob + 2) * 2 = Peter + 2)
  (h5 : Jacob = 11) : John = 30 :=
by 
  sorry

end John_age_l24_24568


namespace combined_mpg_l24_24429

-- Definitions based on the conditions
def ray_miles : ℕ := 150
def tom_miles : ℕ := 100
def ray_mpg : ℕ := 30
def tom_mpg : ℕ := 20

-- Theorem statement
theorem combined_mpg : (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 25 := by
  sorry

end combined_mpg_l24_24429


namespace no_real_roots_iff_range_m_l24_24290

open Real

theorem no_real_roots_iff_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + (m + 3) ≠ 0) ↔ (-2 < m ∧ m < 6) :=
by
  sorry

end no_real_roots_iff_range_m_l24_24290


namespace loaned_out_books_is_50_l24_24905

-- Define the conditions
def initial_books : ℕ := 75
def end_books : ℕ := 60
def percent_returned : ℝ := 0.70

-- Define the variable to represent the number of books loaned out
noncomputable def loaned_out_books := (15:ℝ) / (1 - percent_returned)

-- The target theorem statement we need to prove
theorem loaned_out_books_is_50 : loaned_out_books = 50 :=
by
  sorry

end loaned_out_books_is_50_l24_24905


namespace total_cows_in_ranch_l24_24736

theorem total_cows_in_ranch :
  ∀ (WTP_cows : ℕ) (HGHF_cows : ℕ), WTP_cows = 17 → HGHF_cows = 3 * WTP_cows + 2 → (HGHF_cows + WTP_cows) = 70 :=
by 
  intros WTP_cows HGHF_cows WTP_cows_def HGHF_cows_def
  rw [WTP_cows_def, HGHF_cows_def]
  sorry

end total_cows_in_ranch_l24_24736


namespace hyperbola_vertex_distance_l24_24517

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  4 * x^2 + 24 * x - 4 * y^2 + 16 * y + 44 = 0 →
  2 = 2 :=
by
  intros x y h
  sorry

end hyperbola_vertex_distance_l24_24517


namespace profit_percentage_approx_l24_24200

-- Define the cost price of the first item
def CP1 (S1 : ℚ) : ℚ := 0.81 * S1

-- Define the selling price of the second item as 10% less than the first
def S2 (S1 : ℚ) : ℚ := 0.90 * S1

-- Define the cost price of the second item as 81% of its selling price
def CP2 (S1 : ℚ) : ℚ := 0.81 * (S2 S1)

-- Define the total selling price before tax
def TSP (S1 : ℚ) : ℚ := S1 + S2 S1

-- Define the total amount received after a 5% tax
def TAR (S1 : ℚ) : ℚ := TSP S1 * 0.95

-- Define the total cost price of both items
def TCP (S1 : ℚ) : ℚ := CP1 S1 + CP2 S1

-- Define the profit
def P (S1 : ℚ) : ℚ := TAR S1 - TCP S1

-- Define the profit percentage
def ProfitPercentage (S1 : ℚ) : ℚ := (P S1 / TCP S1) * 100

-- Prove the profit percentage is approximately 17.28%
theorem profit_percentage_approx (S1 : ℚ) : abs (ProfitPercentage S1 - 17.28) < 0.01 :=
by
  sorry

end profit_percentage_approx_l24_24200


namespace max_frac_a_S_l24_24518

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else S n - S (n - 1)

theorem max_frac_a_S (n : ℕ) (h : S n = 2^n - 1) : 
  let frac := (a n) / (a n * S n + a 6)
  ∃ N : ℕ, N > 0 ∧ (frac ≤ 1 / 15) := by
  sorry

end max_frac_a_S_l24_24518


namespace eraser_ratio_l24_24529

-- Define the variables and conditions
variables (c j g : ℕ)
variables (total : ℕ := 35)
variables (c_erasers : ℕ := 10)
variables (gabriel_erasers : ℕ := c_erasers / 2)
variables (julian_erasers : ℕ := c_erasers)

-- The proof statement
theorem eraser_ratio (hc : c_erasers = 10)
                      (h1 : c_erasers = 2 * gabriel_erasers)
                      (h2 : julian_erasers = c_erasers)
                      (h3 : c_erasers + gabriel_erasers + julian_erasers = total) :
                      julian_erasers / c_erasers = 1 :=
by
  sorry

end eraser_ratio_l24_24529


namespace common_pts_above_curve_l24_24138

open Real

theorem common_pts_above_curve {x y t : ℝ} (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : 0 ≤ y ∧ y ≤ 1) (h3 : 0 < t ∧ t < 1) :
  (∀ t, y ≥ (t-1)/t * x + 1 - t) ↔ (sqrt x + sqrt y ≥ 1) := 
by
  sorry

end common_pts_above_curve_l24_24138


namespace train_crosses_signal_pole_in_18_seconds_l24_24530

-- Define the given conditions
def train_length := 300  -- meters
def platform_length := 450  -- meters
def time_to_cross_platform := 45  -- seconds

-- Define the question and the correct answer
def time_to_cross_signal_pole := 18  -- seconds (this is what we need to prove)

-- Define the total distance the train covers when crossing the platform
def total_distance_crossing_platform := train_length + platform_length  -- meters

-- Define the speed of the train
def train_speed := total_distance_crossing_platform / time_to_cross_platform  -- meters per second

theorem train_crosses_signal_pole_in_18_seconds :
  300 / train_speed = time_to_cross_signal_pole :=
by
  -- train_speed is defined directly in terms of the given conditions
  unfold train_speed total_distance_crossing_platform train_length platform_length time_to_cross_platform
  sorry

end train_crosses_signal_pole_in_18_seconds_l24_24530


namespace pool_full_capacity_is_2000_l24_24960

-- Definitions based on the conditions given
def water_loss_per_jump : ℕ := 400 -- in ml
def jumps_before_cleaning : ℕ := 1000
def cleaning_threshold : ℚ := 0.80 -- 80%
def total_water_loss : ℕ := water_loss_per_jump * jumps_before_cleaning -- in ml
def water_loss_liters : ℚ := total_water_loss / 1000 -- converting ml to liters
def cleaning_loss_fraction : ℚ := 1 - cleaning_threshold -- 20% loss

-- The actual proof statement
theorem pool_full_capacity_is_2000 :
  (water_loss_liters : ℚ) / cleaning_loss_fraction = 2000 :=
by
  sorry

end pool_full_capacity_is_2000_l24_24960


namespace max_candies_theorem_l24_24068

-- Defining constants: the number of students and the total number of candies.
def n : ℕ := 40
def T : ℕ := 200

-- Defining the condition that each student takes at least 2 candies.
def min_candies_per_student : ℕ := 2

-- Calculating the minimum total number of candies taken by 39 students.
def min_total_for_39_students := min_candies_per_student * (n - 1)

-- The maximum number of candies one student can take.
def max_candies_one_student_can_take := T - min_total_for_39_students

-- The statement to prove.
theorem max_candies_theorem : max_candies_one_student_can_take = 122 :=
by
  sorry

end max_candies_theorem_l24_24068


namespace tan_product_in_triangle_l24_24083

theorem tan_product_in_triangle (A B C : ℝ) (h1 : A + B + C = Real.pi)
  (h2 : Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = Real.sin B ^ 2) :
  Real.tan A * Real.tan C = 1 :=
sorry

end tan_product_in_triangle_l24_24083


namespace reservoir_water_level_at_6_pm_l24_24369

/-
  Initial conditions:
  - initial_water_level: Water level at 8 a.m.
  - increase_rate: Rate of increase in water level from 8 a.m. to 12 p.m.
  - decrease_rate: Rate of decrease in water level from 12 p.m. to 6 p.m.
  - start_increase_time: Starting time of increase (in hours from 8 a.m.)
  - end_increase_time: Ending time of increase (in hours from 8 a.m.)
  - start_decrease_time: Starting time of decrease (in hours from 12 p.m.)
  - end_decrease_time: Ending time of decrease (in hours from 12 p.m.)
-/
def initial_water_level : ℝ := 45
def increase_rate : ℝ := 0.6
def decrease_rate : ℝ := 0.3
def start_increase_time : ℝ := 0 -- 8 a.m. in hours from 8 a.m.
def end_increase_time : ℝ := 4 -- 12 p.m. in hours from 8 a.m.
def start_decrease_time : ℝ := 0 -- 12 p.m. in hours from 12 p.m.
def end_decrease_time : ℝ := 6 -- 6 p.m. in hours from 12 p.m.

theorem reservoir_water_level_at_6_pm :
  initial_water_level
  + (end_increase_time - start_increase_time) * increase_rate
  - (end_decrease_time - start_decrease_time) * decrease_rate
  = 45.6 :=
by
  sorry

end reservoir_water_level_at_6_pm_l24_24369


namespace gray_region_area_l24_24458

theorem gray_region_area 
  (r : ℝ) 
  (h1 : ∀ r : ℝ, (3 * r) - r = 3) 
  (h2 : r = 1.5) 
  (inner_circle_area : ℝ := π * r * r) 
  (outer_circle_area : ℝ := π * (3 * r) * (3 * r)) : 
  outer_circle_area - inner_circle_area = 18 * π := 
by
  sorry

end gray_region_area_l24_24458


namespace find_x_l24_24305

theorem find_x (x : ℝ) (h : 2 * x = 26 - x + 19) : x = 15 :=
by
  sorry

end find_x_l24_24305


namespace total_number_of_toy_cars_l24_24807

-- Definitions based on conditions
def numCarsBox1 : ℕ := 21
def numCarsBox2 : ℕ := 31
def numCarsBox3 : ℕ := 19

-- The proof statement
theorem total_number_of_toy_cars : numCarsBox1 + numCarsBox2 + numCarsBox3 = 71 := by
  sorry

end total_number_of_toy_cars_l24_24807


namespace quad_area_l24_24322

theorem quad_area (a b : Int) (h1 : a > b) (h2 : b > 0) (h3 : 2 * |a - b| * |a + b| = 50) : a + b = 15 :=
by
  sorry

end quad_area_l24_24322


namespace second_term_of_geometric_series_l24_24073

theorem second_term_of_geometric_series 
  (a : ℝ) (r : ℝ) (S : ℝ) :
  r = 1 / 4 → S = 40 → S = a / (1 - r) → a * r = 7.5 :=
by
  intros hr hS hSum
  sorry

end second_term_of_geometric_series_l24_24073


namespace simple_interest_rate_l24_24147

theorem simple_interest_rate (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : SI = P / 5)
  (h2 : SI = P * R * T / 100)
  (h3 : T = 7) : 
  R = 20 / 7 :=
by 
  sorry

end simple_interest_rate_l24_24147


namespace max_value_of_a2b3c2_l24_24020

theorem max_value_of_a2b3c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) :
  a^2 * b^3 * c^2 ≤ 81 / 262144 :=
sorry

end max_value_of_a2b3c2_l24_24020


namespace initial_amount_is_1875_l24_24613

-- Defining the conditions as given in the problem
def initial_amount : ℝ := sorry
def spent_on_clothes : ℝ := 250
def spent_on_food (remaining : ℝ) : ℝ := 0.35 * remaining
def spent_on_electronics (remaining : ℝ) : ℝ := 0.50 * remaining

-- Given conditions
axiom condition1 : initial_amount - spent_on_clothes = sorry
axiom condition2 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) = sorry
axiom condition3 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) - spent_on_electronics (initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes)) = 200

-- Prove that initial amount is $1875
theorem initial_amount_is_1875 : initial_amount = 1875 :=
sorry

end initial_amount_is_1875_l24_24613


namespace find_value_of_expression_l24_24544

theorem find_value_of_expression (x y z : ℚ)
  (h1 : 2 * x + y + z = 14)
  (h2 : 2 * x + y = 7)
  (h3 : x + 2 * y = 10) : (x + y - z) / 3 = -4 / 9 :=
by sorry

end find_value_of_expression_l24_24544


namespace factorize_x4_minus_81_l24_24358

theorem factorize_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factorize_x4_minus_81_l24_24358


namespace ab_value_l24_24779

variable (a b : ℝ)

theorem ab_value (h1 : a^5 * b^8 = 12) (h2 : a^8 * b^13 = 18) : a * b = 128 / 3 := 
by 
  sorry

end ab_value_l24_24779


namespace shaded_area_triangle_l24_24866

theorem shaded_area_triangle (a b : ℝ) (h1 : a = 5) (h2 : b = 15) :
  let area_shaded : ℝ := (5^2) - (1/2 * ((15 / 4) * 5))
  area_shaded = 175 / 8 := 
by
  sorry

end shaded_area_triangle_l24_24866


namespace chickens_in_coop_l24_24296

theorem chickens_in_coop (C : ℕ)
  (H1 : ∃ C : ℕ, ∀ R : ℕ, R = 2 * C)
  (H2 : ∃ R : ℕ, ∀ F : ℕ, F = 2 * R - 4)
  (H3 : ∃ F : ℕ, F = 52) :
  C = 14 :=
by sorry

end chickens_in_coop_l24_24296


namespace distance_planes_A_B_l24_24631

noncomputable def distance_between_planes : ℝ :=
  let d1 := 1
  let d2 := 2
  let a := 1
  let b := 1
  let c := 1
  (|d2 - d1|) / (Real.sqrt (a^2 + b^2 + c^2))

theorem distance_planes_A_B :
  let A := fun (x y z : ℝ) => x + y + z = 1
  let B := fun (x y z : ℝ) => x + y + z = 2
  distance_between_planes = 1 / Real.sqrt 3 :=
  by
    -- Proof steps will be here
    sorry

end distance_planes_A_B_l24_24631


namespace minimum_value_of_expression_l24_24050

theorem minimum_value_of_expression (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : a * b > 0) : 
  (1 / a) + (2 / b) = 5 :=
sorry

end minimum_value_of_expression_l24_24050


namespace consecutive_even_numbers_l24_24044

theorem consecutive_even_numbers (n m : ℕ) (h : 52 * (2 * n - 1) = 100 * n) : n = 13 :=
by
  sorry

end consecutive_even_numbers_l24_24044


namespace express_train_leaves_6_hours_later_l24_24590

theorem express_train_leaves_6_hours_later
  (V_g V_e : ℕ) (t : ℕ) (catch_up_time : ℕ)
  (goods_train_speed : V_g = 36)
  (express_train_speed : V_e = 90)
  (catch_up_in_4_hours : catch_up_time = 4)
  (distance_e : V_e * catch_up_time = 360)
  (distance_g : V_g * (t + catch_up_time) = 360) :
  t = 6 := by
  sorry

end express_train_leaves_6_hours_later_l24_24590


namespace abs_expression_value_l24_24842

theorem abs_expression_value (x : ℤ) (h : x = -2023) :
  abs (2 * abs (abs x - x) - abs x) - x = 8092 :=
by {
  -- Proof will be provided here
  sorry
}

end abs_expression_value_l24_24842


namespace selling_price_correct_l24_24363

-- Define the conditions
def cost_price : ℝ := 900
def gain_percentage : ℝ := 0.2222222222222222

-- Define the selling price calculation
def profit := cost_price * gain_percentage
def selling_price := cost_price + profit

-- The problem statement in Lean 4
theorem selling_price_correct : selling_price = 1100 := 
by
  -- Proof to be filled in later
  sorry

end selling_price_correct_l24_24363


namespace pipe_fill_time_without_leak_l24_24376

theorem pipe_fill_time_without_leak (T : ℕ) :
  let pipe_with_leak_time := 10
  let leak_empty_time := 10
  ((1 / T : ℚ) - (1 / leak_empty_time) = (1 / pipe_with_leak_time)) →
  T = 5 := 
sorry

end pipe_fill_time_without_leak_l24_24376


namespace div_mult_result_l24_24316

theorem div_mult_result : 150 / (30 / 3) * 2 = 30 :=
by sorry

end div_mult_result_l24_24316


namespace findFirstCarSpeed_l24_24615

noncomputable def firstCarSpeed (v : ℝ) (blackCarSpeed : ℝ) (initialGap : ℝ) (timeToCatchUp : ℝ) : Prop :=
  blackCarSpeed * timeToCatchUp = initialGap + v * timeToCatchUp → v = 30

theorem findFirstCarSpeed :
  firstCarSpeed 30 50 20 1 :=
by
  sorry

end findFirstCarSpeed_l24_24615


namespace dealers_profit_percentage_l24_24339

theorem dealers_profit_percentage 
  (articles_purchased : ℕ)
  (total_cost_price : ℝ)
  (articles_sold : ℕ)
  (total_selling_price : ℝ)
  (CP_per_article : ℝ := total_cost_price / articles_purchased)
  (SP_per_article : ℝ := total_selling_price / articles_sold)
  (profit_per_article : ℝ := SP_per_article - CP_per_article)
  (profit_percentage : ℝ := (profit_per_article / CP_per_article) * 100) :
  articles_purchased = 15 →
  total_cost_price = 25 →
  articles_sold = 12 →
  total_selling_price = 32 →
  profit_percentage = 60 :=
by
  intros h1 h2 h3 h4
  sorry

end dealers_profit_percentage_l24_24339


namespace problem_statement_l24_24632

variable {a b c x y z : ℝ}
variable (h1 : 17 * x + b * y + c * z = 0)
variable (h2 : a * x + 29 * y + c * z = 0)
variable (h3 : a * x + b * y + 53 * z = 0)
variable (ha : a ≠ 17)
variable (hx : x ≠ 0)

theorem problem_statement : 
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
sorry

end problem_statement_l24_24632


namespace magnitude_of_a_l24_24747

open Real

-- Assuming the standard inner product space for vectors in Euclidean space

variables (a b : ℝ) -- Vectors in R^n (could be general but simplified to real numbers for this example)
variable (θ : ℝ)    -- Angle between vectors
axiom angle_ab : θ = 60 -- Given angle between vectors

-- Conditions:
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom norm_b : abs b = 1
axiom norm_2a_minus_b : abs (2 * a - b) = 1

-- To prove:
theorem magnitude_of_a : abs a = 1 / 2 :=
sorry

end magnitude_of_a_l24_24747


namespace angle_B_in_equilateral_triangle_l24_24078

theorem angle_B_in_equilateral_triangle (A B C : ℝ) (h_angle_sum : A + B + C = 180) (h_A : A = 80) (h_BC : B = C) :
  B = 50 :=
by
  -- Conditions
  have h1 : A = 80 := by exact h_A
  have h2 : B = C := by exact h_BC
  have h3 : A + B + C = 180 := by exact h_angle_sum

  sorry -- completing the proof is not required

end angle_B_in_equilateral_triangle_l24_24078


namespace polygon_n_sides_l24_24272

theorem polygon_n_sides (n : ℕ) (h : (n - 2) * 180 - x = 2000) : n = 14 :=
sorry

end polygon_n_sides_l24_24272


namespace inequality_solution_l24_24949

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (x^2 > x^(1 / 2)) ↔ (x > 1) :=
by
  sorry

end inequality_solution_l24_24949


namespace theater_ticket_sales_l24_24449

theorem theater_ticket_sales (O B : ℕ) 
  (h1 : O + B = 370) 
  (h2 : 12 * O + 8 * B = 3320) : 
  B - O = 190 := 
sorry

end theater_ticket_sales_l24_24449


namespace odd_number_as_diff_of_squares_l24_24033

theorem odd_number_as_diff_of_squares (n : ℤ) : ∃ a b : ℤ, a^2 - b^2 = 2 * n + 1 :=
by
  use (n + 1), n
  sorry

end odd_number_as_diff_of_squares_l24_24033


namespace number_of_chocolate_boxes_l24_24850

theorem number_of_chocolate_boxes
  (x y p : ℕ)
  (pieces_per_box : ℕ)
  (total_candies : ℕ)
  (h_y : y = 4)
  (h_pieces : pieces_per_box = 9)
  (h_total : total_candies = 90) :
  x = 6 :=
by
  -- Definitions of the conditions
  let caramel_candies := y * pieces_per_box
  let total_chocolate_candies := total_candies - caramel_candies
  let x := total_chocolate_candies / pieces_per_box
  
  -- Main theorem statement: x = 6
  sorry

end number_of_chocolate_boxes_l24_24850


namespace solve_equation_l24_24017

theorem solve_equation : ∃ x : ℝ, (2 * x - 1) / 3 - (x - 2) / 6 = 2 ∧ x = 4 :=
by
  sorry

end solve_equation_l24_24017


namespace solve_for_x_l24_24690

theorem solve_for_x (x : ℝ) (h : (x - 5)^4 = (1 / 16)⁻¹) : x = 7 :=
by
  sorry

end solve_for_x_l24_24690


namespace coach_A_spent_less_l24_24329

-- Definitions of costs and discounts for coaches purchases
def total_cost_before_discount_A : ℝ := 10 * 29 + 5 * 15
def total_cost_before_discount_B : ℝ := 14 * 2.50 + 1 * 18 + 4 * 25 + 1 * 72
def total_cost_before_discount_C : ℝ := 8 * 32 + 12 * 12

def discount_A : ℝ := 0.05 * total_cost_before_discount_A
def discount_B : ℝ := 0.10 * total_cost_before_discount_B
def discount_C : ℝ := 0.07 * total_cost_before_discount_C

def total_cost_after_discount_A : ℝ := total_cost_before_discount_A - discount_A
def total_cost_after_discount_B : ℝ := total_cost_before_discount_B - discount_B
def total_cost_after_discount_C : ℝ := total_cost_before_discount_C - discount_C

def combined_cost_B_C : ℝ := total_cost_after_discount_B + total_cost_after_discount_C
def difference_A_BC : ℝ := total_cost_after_discount_A - combined_cost_B_C

theorem coach_A_spent_less : difference_A_BC = -227.75 := by
  sorry

end coach_A_spent_less_l24_24329


namespace no_possible_values_of_k_l24_24873

theorem no_possible_values_of_k :
  ¬(∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p + q = 65) :=
by
  sorry

end no_possible_values_of_k_l24_24873


namespace number_is_16_l24_24635

theorem number_is_16 (n : ℝ) (h : (1/2) * n + 5 = 13) : n = 16 :=
sorry

end number_is_16_l24_24635


namespace probability_at_least_one_defective_probability_at_most_one_defective_l24_24792

noncomputable def machine_a_defect_rate : ℝ := 0.05
noncomputable def machine_b_defect_rate : ℝ := 0.1

/-- 
Prove the probability that there is at least one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_least_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - (1 - pA) * (1 - pB)) = 0.145 :=
  sorry

/-- 
Prove the probability that there is at most one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_most_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - pA * pB) = 0.995 :=
  sorry

end probability_at_least_one_defective_probability_at_most_one_defective_l24_24792


namespace remainder_when_n_plus_2947_divided_by_7_l24_24003

theorem remainder_when_n_plus_2947_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (n + 2947) % 7 = 3 :=
by
  sorry

end remainder_when_n_plus_2947_divided_by_7_l24_24003


namespace malcolm_social_media_followers_l24_24762

theorem malcolm_social_media_followers :
  let instagram_initial := 240
  let facebook_initial := 500
  let twitter_initial := (instagram_initial + facebook_initial) / 2
  let tiktok_initial := 3 * twitter_initial
  let youtube_initial := tiktok_initial + 510
  let pinterest_initial := 120
  let snapchat_initial := pinterest_initial / 2

  let instagram_after := instagram_initial + (15 * instagram_initial / 100)
  let facebook_after := facebook_initial + (20 * facebook_initial / 100)
  let twitter_after := twitter_initial - 12
  let tiktok_after := tiktok_initial + (10 * tiktok_initial / 100)
  let youtube_after := youtube_initial + (8 * youtube_initial / 100)
  let pinterest_after := pinterest_initial + 20
  let snapchat_after := snapchat_initial - (5 * snapchat_initial / 100)

  instagram_after + facebook_after + twitter_after + tiktok_after + youtube_after + pinterest_after + snapchat_after = 4402 := sorry

end malcolm_social_media_followers_l24_24762


namespace find_a_l24_24790

-- Define the sets A and B and the condition that A union B is a subset of A intersect B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) :
  A ∪ B a ⊆ A ∩ B a → a = 1 :=
sorry

end find_a_l24_24790


namespace cheaper_to_buy_more_cheaper_2_values_l24_24812

def cost_function (n : ℕ) : ℕ :=
  if (1 ≤ n ∧ n ≤ 30) then 15 * n - 20
  else if (31 ≤ n ∧ n ≤ 55) then 14 * n
  else if (56 ≤ n) then 13 * n + 10
  else 0  -- Assuming 0 for n < 1 as it shouldn't happen in this context

theorem cheaper_to_buy_more_cheaper_2_values : 
  ∃ n1 n2 : ℕ, n1 < n2 ∧ cost_function (n1 + 1) < cost_function n1 ∧ cost_function (n2 + 1) < cost_function n2 ∧
  ∀ n : ℕ, (cost_function (n + 1) < cost_function n → n = n1 ∨ n = n2) := 
sorry

end cheaper_to_buy_more_cheaper_2_values_l24_24812


namespace boxes_neither_markers_nor_crayons_l24_24484

theorem boxes_neither_markers_nor_crayons (total boxes_markers boxes_crayons boxes_both: ℕ)
  (htotal : total = 15)
  (hmarkers : boxes_markers = 9)
  (hcrayons : boxes_crayons = 4)
  (hboth : boxes_both = 5) :
  total - (boxes_markers + boxes_crayons - boxes_both) = 7 := by
  sorry

end boxes_neither_markers_nor_crayons_l24_24484


namespace cornbread_pieces_l24_24334

theorem cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ)
  (h₁ : pan_length = 24) (h₂ : pan_width = 20) 
  (h₃ : piece_length = 3) (h₄ : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 80 := by
  sorry

end cornbread_pieces_l24_24334


namespace simplify_expression_l24_24540

theorem simplify_expression :
  (1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 6 - 2)))) =
  ((3 * Real.sqrt 5 + 2 * Real.sqrt 6 + 2) / 29) :=
  sorry

end simplify_expression_l24_24540


namespace solve_for_b_l24_24419

theorem solve_for_b (b : ℚ) : 
  (∃ m1 m2 : ℚ, 3 * m1 - 2 * 1 + 4 = 0 ∧ 5 * m2 + b * 1 - 1 = 0 ∧ m1 * m2 = -1) → b = 15 / 2 :=
by
  sorry

end solve_for_b_l24_24419


namespace sequence_a_n_l24_24800

theorem sequence_a_n (a : ℕ → ℚ)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 1) * (a n + 1) = a n) :
  a 6 = 1 / 6 :=
  sorry

end sequence_a_n_l24_24800


namespace cubic_inches_in_two_cubic_feet_l24_24588

-- Define the conversion factor between feet and inches
def foot_to_inch : ℕ := 12
-- Define the conversion factor between cubic feet and cubic inches
def cubic_foot_to_cubic_inch : ℕ := foot_to_inch ^ 3

-- State the theorem to be proved
theorem cubic_inches_in_two_cubic_feet : 2 * cubic_foot_to_cubic_inch = 3456 :=
by
  -- Proof steps go here
  sorry

end cubic_inches_in_two_cubic_feet_l24_24588


namespace increase_in_avg_commission_l24_24303

def new_avg_commission := 250
def num_sales := 6
def big_sale_commission := 1000

theorem increase_in_avg_commission :
  (new_avg_commission - (500 / (num_sales - 1))) = 150 := by
  sorry

end increase_in_avg_commission_l24_24303


namespace shopkeeper_total_cards_l24_24942

-- Definition of the number of cards in standard, Uno, and tarot decks.
def std_deck := 52
def uno_deck := 108
def tarot_deck := 78

-- Number of complete decks and additional cards.
def std_decks := 4
def uno_decks := 3
def tarot_decks := 5
def additional_std := 12
def additional_uno := 7
def additional_tarot := 9

-- Calculate the total number of cards.
def total_standard_cards := (std_decks * std_deck) + additional_std
def total_uno_cards := (uno_decks * uno_deck) + additional_uno
def total_tarot_cards := (tarot_decks * tarot_deck) + additional_tarot

def total_cards := total_standard_cards + total_uno_cards + total_tarot_cards

theorem shopkeeper_total_cards : total_cards = 950 := by
  sorry

end shopkeeper_total_cards_l24_24942


namespace part1_part2_l24_24141

variable {x m : ℝ}

def P (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def S (x : ℝ) (m : ℝ) : Prop := -m + 1 ≤ x ∧ x ≤ m + 1

theorem part1 (h : ∀ x, P x → P x ∨ S x m) : m ≤ 0 :=
sorry

theorem part2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (P x ↔ S x m) :=
sorry

end part1_part2_l24_24141


namespace nonneg_integer_solutions_otimes_l24_24331

noncomputable def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

theorem nonneg_integer_solutions_otimes :
  {x : ℕ | otimes 2 x ≥ 3} = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_otimes_l24_24331


namespace eccentricity_of_hyperbola_l24_24879

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  let c := 2 * b
  let e := c / a
  e

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_cond : hyperbola_eccentricity a b h_a h_b = 2 * (b / a)) :
  hyperbola_eccentricity a b h_a h_b = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l24_24879


namespace fewest_cookies_l24_24428

theorem fewest_cookies
  (area_art_cookies : ℝ)
  (area_roger_cookies : ℝ)
  (area_paul_cookies : ℝ)
  (area_trisha_cookies : ℝ)
  (h_art : area_art_cookies = 12)
  (h_roger : area_roger_cookies = 8)
  (h_paul : area_paul_cookies = 6)
  (h_trisha : area_trisha_cookies = 6)
  (dough : ℝ) :
  (dough / area_art_cookies) < (dough / area_roger_cookies) ∧
  (dough / area_art_cookies) < (dough / area_paul_cookies) ∧
  (dough / area_art_cookies) < (dough / area_trisha_cookies) := by
  sorry

end fewest_cookies_l24_24428


namespace P_sufficient_but_not_necessary_for_Q_l24_24091

-- Definitions based on given conditions
def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

-- The theorem to prove that P is sufficient but not necessary for Q
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l24_24091


namespace find_m_l24_24617

theorem find_m (x y m : ℝ)
  (h1 : 2 * x + y = 6 * m)
  (h2 : 3 * x - 2 * y = 2 * m)
  (h3 : x / 3 - y / 5 = 4) :
  m = 15 :=
by
  sorry

end find_m_l24_24617


namespace remainder_of_82460_div_8_l24_24926

theorem remainder_of_82460_div_8 :
  82460 % 8 = 4 :=
sorry

end remainder_of_82460_div_8_l24_24926


namespace new_prism_volume_l24_24593

theorem new_prism_volume (L W H : ℝ) 
  (h_volume : L * W * H = 54)
  (L_new : ℝ := 2 * L)
  (W_new : ℝ := 3 * W)
  (H_new : ℝ := 1.5 * H) :
  L_new * W_new * H_new = 486 := 
by
  sorry

end new_prism_volume_l24_24593


namespace number_of_other_workers_l24_24713

theorem number_of_other_workers (N : ℕ) (h1 : N ≥ 2) (h2 : 1 / ((N * (N - 1)) / 2) = 1 / 6) : N - 2 = 2 :=
by
  sorry

end number_of_other_workers_l24_24713


namespace maci_red_pens_l24_24503

def cost_blue_pens (b : ℕ) (cost_blue : ℕ) : ℕ := b * cost_blue

def cost_red_pen (cost_blue : ℕ) : ℕ := 2 * cost_blue

def total_cost (cost_blue : ℕ) (n_blue : ℕ) (n_red : ℕ) : ℕ := 
  n_blue * cost_blue + n_red * (2 * cost_blue)

theorem maci_red_pens :
  ∀ (n_blue cost_blue n_red total : ℕ),
  n_blue = 10 →
  cost_blue = 10 →
  total = 400 →
  total_cost cost_blue n_blue n_red = total →
  n_red = 15 := 
by
  intros n_blue cost_blue n_red total h1 h2 h3 h4
  sorry

end maci_red_pens_l24_24503


namespace range_of_a_l24_24766

noncomputable def M : Set ℝ := {2, 0, -1}
noncomputable def N (a : ℝ) : Set ℝ := {x | abs (x - a) < 1}

theorem range_of_a (a : ℝ) : (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 3) ↔ M ∩ N a = {x} :=
by
  sorry

end range_of_a_l24_24766


namespace geometric_series_sum_l24_24466

theorem geometric_series_sum (a r : ℝ)
  (h₁ : a / (1 - r) = 15)
  (h₂ : a / (1 - r^4) = 9) :
  r = 1 / 3 :=
sorry

end geometric_series_sum_l24_24466


namespace playground_perimeter_l24_24470

-- Defining the conditions
def length : ℕ := 100
def breadth : ℕ := 500
def perimeter (L B : ℕ) : ℕ := 2 * (L + B)

-- The theorem to prove
theorem playground_perimeter : perimeter length breadth = 1200 := 
by
  -- The actual proof will be filled later
  sorry

end playground_perimeter_l24_24470


namespace remainder_equality_l24_24441

theorem remainder_equality (a b s t d : ℕ) (h1 : a > b) (h2 : a % d = s % d) (h3 : b % d = t % d) :
  ((a + 1) * (b + 1)) % d = ((s + 1) * (t + 1)) % d :=
by
  sorry

end remainder_equality_l24_24441


namespace Peter_speed_is_correct_l24_24278

variable (Peter_speed : ℝ)

def Juan_speed : ℝ := Peter_speed + 3

def distance_Peter_in_1_5_hours : ℝ := 1.5 * Peter_speed

def distance_Juan_in_1_5_hours : ℝ := 1.5 * Juan_speed Peter_speed

theorem Peter_speed_is_correct (h : distance_Peter_in_1_5_hours Peter_speed + distance_Juan_in_1_5_hours Peter_speed = 19.5) : Peter_speed = 5 :=
by
  sorry

end Peter_speed_is_correct_l24_24278


namespace cornelia_european_countries_l24_24426

def total_countries : Nat := 42
def south_american_countries : Nat := 10
def asian_countries : Nat := 6

def non_european_countries : Nat :=
  south_american_countries + 2 * asian_countries

def european_countries : Nat :=
  total_countries - non_european_countries

theorem cornelia_european_countries :
  european_countries = 20 := by
  sorry

end cornelia_european_countries_l24_24426


namespace abs_neg_two_eq_two_l24_24885

theorem abs_neg_two_eq_two : abs (-2) = 2 :=
sorry

end abs_neg_two_eq_two_l24_24885


namespace min_value_of_function_l24_24002

theorem min_value_of_function (x : ℝ) (h: x > 1) :
  ∃ t > 0, x = t + 1 ∧ (t + 3 / t + 3) = 3 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_l24_24002


namespace greatest_three_digit_multiple_of_23_l24_24430

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of_23 (n : ℕ) : Prop :=
  n % 23 = 0

theorem greatest_three_digit_multiple_of_23 :
  ∀ n, is_three_digit n ∧ is_multiple_of_23 n → n ≤ 989 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l24_24430


namespace pagoda_lights_l24_24452

/-- From afar, the magnificent pagoda has seven layers, with red lights doubling on each
ascending floor, totaling 381 lights. How many lights are there at the very top? -/
theorem pagoda_lights :
  ∃ x, (1 + 2 + 4 + 8 + 16 + 32 + 64) * x = 381 ∧ x = 3 :=
by
  sorry

end pagoda_lights_l24_24452


namespace plane_ticket_price_l24_24263

theorem plane_ticket_price :
  ∀ (P : ℕ),
  (20 * 155) + 2900 = 30 * P →
  P = 200 := 
by
  sorry

end plane_ticket_price_l24_24263


namespace sum_smallest_largest_l24_24461

theorem sum_smallest_largest (z b : ℤ) (n : ℤ) (h_even_n : (n % 2 = 0)) (h_mean : z = (n * b + ((n - 1) * n) / 2) / n) : 
  (2 * (z - (n - 1) / 2) + n - 1) = 2 * z := by
  sorry

end sum_smallest_largest_l24_24461


namespace Oliver_ferris_wheel_rides_l24_24945

theorem Oliver_ferris_wheel_rides :
  ∃ (F : ℕ), (4 * 7 + F * 7 = 63) ∧ (F = 5) :=
by
  sorry

end Oliver_ferris_wheel_rides_l24_24945


namespace number_of_rows_containing_53_l24_24708

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l24_24708


namespace maximum_marks_l24_24285

theorem maximum_marks (M : ℝ) (h1 : 0.45 * M = 180) : M = 400 := 
by sorry

end maximum_marks_l24_24285


namespace min_heaviest_weight_l24_24063

theorem min_heaviest_weight : 
  ∃ (w : ℕ), ∀ (weights : Fin 8 → ℕ),
    (∀ i j, i ≠ j → weights i ≠ weights j) ∧
    (∀ (a b c d : Fin 8),
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
      (weights a + weights b) ≠ (weights c + weights d) ∧ 
      max (max (weights a) (weights b)) (max (weights c) (weights d)) >= w) 
  → w = 34 := 
by
  sorry

end min_heaviest_weight_l24_24063


namespace triangle_side_eq_median_l24_24612

theorem triangle_side_eq_median (A B C : Type) (a b c : ℝ) (hAB : a = 2) (hAC : b = 3) (hBC_eq_median : c = (2 * (Real.sqrt (13 / 10)))) :
  c = (Real.sqrt 130) / 5 := by
  sorry

end triangle_side_eq_median_l24_24612


namespace rectangle_area_stage4_l24_24325

-- Define the condition: area of one square
def square_area : ℕ := 25

-- Define the condition: number of squares at Stage 4
def num_squares_stage4 : ℕ := 4

-- Define the total area of rectangle at Stage 4
def total_area_stage4 : ℕ := num_squares_stage4 * square_area

-- Prove that total_area_stage4 equals 100 square inches
theorem rectangle_area_stage4 : total_area_stage4 = 100 :=
by
  sorry

end rectangle_area_stage4_l24_24325


namespace find_principal_l24_24001

-- Definitions based on conditions
def simple_interest (P R T : ℚ) : ℚ := (P * R * T) / 100

-- Given conditions
def SI : ℚ := 6016.75
def R : ℚ := 8
def T : ℚ := 5

-- Stating the proof problem
theorem find_principal : 
  ∃ P : ℚ, simple_interest P R T = SI ∧ P = 15041.875 :=
by {
  sorry
}

end find_principal_l24_24001


namespace initial_boys_count_l24_24232

theorem initial_boys_count (B : ℕ) (boys girls : ℕ)
  (h1 : boys = 3 * B)                             -- The ratio of boys to girls is 3:4
  (h2 : girls = 4 * B)                            -- The ratio of boys to girls is 3:4
  (h3 : boys - 10 = 4 * (girls - 20))             -- The final ratio after transfer is 4:5
  : boys = 90 :=                                  -- Prove initial boys count was 90
by 
  sorry

end initial_boys_count_l24_24232


namespace average_score_l24_24451

-- Definitions from conditions
def June_score := 97
def Patty_score := 85
def Josh_score := 100
def Henry_score := 94
def total_children := 4
def total_score := June_score + Patty_score + Josh_score + Henry_score

-- Prove the average score
theorem average_score : (total_score / total_children) = 94 :=
by
  sorry

end average_score_l24_24451


namespace range_of_m_l24_24662

noncomputable def quadratic_function (m x : ℝ) : ℝ := (m-2) * x^2 + 2 * m * x - (3 - m)

theorem range_of_m (m : ℝ) (h_vertex_third_quadrant : (-(m) / (m-2) < 0) ∧ ((-5)*m + 6) / (m-2) < 0)
                   (h_parabola_opens_upwards : m - 2 > 0)
                   (h_intersects_negative_y_axis : m < 3) : 2 < m ∧ m < 3 :=
by {
    sorry
}

end range_of_m_l24_24662


namespace minimum_value_expression_l24_24224

-- Define the conditions for positive real numbers
variables (a b c : ℝ)
variable (h_a : 0 < a)
variable (h_b : 0 < b)
variable (h_c : 0 < c)

-- State the theorem to prove the minimum value of the expression
theorem minimum_value_expression (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : 
  (a / b) + (b / c) + (c / a) ≥ 3 := 
sorry

end minimum_value_expression_l24_24224


namespace no_such_n_l24_24381

theorem no_such_n (n : ℕ) (h_pos : 0 < n) :
  ¬ ∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
sorry

end no_such_n_l24_24381


namespace area_of_yard_l24_24677

theorem area_of_yard (L W : ℕ) (h1 : L = 40) (h2 : L + 2 * W = 64) : L * W = 480 := by
  sorry

end area_of_yard_l24_24677


namespace factorize_difference_of_squares_l24_24169

theorem factorize_difference_of_squares (x : ℝ) :
  4 * x^2 - 1 = (2 * x + 1) * (2 * x - 1) :=
sorry

end factorize_difference_of_squares_l24_24169


namespace parts_of_a_number_l24_24564

theorem parts_of_a_number 
  (a p q : ℝ) 
  (x y z : ℝ)
  (h1 : y + z = p * x)
  (h2 : x + y = q * z)
  (h3 : x + y + z = a) :
  x = a / (1 + p) ∧ y = a * (p * q - 1) / ((p + 1) * (q + 1)) ∧ z = a / (1 + q) := 
by 
  sorry

end parts_of_a_number_l24_24564


namespace problem_l24_24465

theorem problem (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 16) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 4) : 
  x * y * z = 4 := 
sorry

end problem_l24_24465


namespace circle_inside_triangle_l24_24522

-- Define the problem conditions
def triangle_sides : ℕ × ℕ × ℕ := (3, 4, 5)
def circle_area : ℚ := 25 / 8

-- Define the problem statement
theorem circle_inside_triangle (a b c : ℕ) (area : ℚ)
    (h1 : (a, b, c) = triangle_sides)
    (h2 : area = circle_area) :
    ∃ r R : ℚ, R < r ∧ 2 * r = a + b - c ∧ R^2 = area / π := sorry

end circle_inside_triangle_l24_24522


namespace fill_tank_time_l24_24477

-- Define the rates at which the pipes fill or empty the tank
def rateA : ℚ := 1 / 16
def rateB : ℚ := - (1 / 24)  -- Since pipe B empties the tank, it's negative.

-- Define the time after which pipe B is closed
def timeBClosed : ℚ := 21

-- Define the initial combined rate of both pipes
def combinedRate : ℚ := rateA + rateB

-- Define the proportion of the tank filled in the initial 21 minutes
def filledIn21Minutes : ℚ := combinedRate * timeBClosed

-- Define the remaining tank to be filled after pipe B is closed
def remainingTank : ℚ := 1 - filledIn21Minutes

-- Define the additional time required to fill the remaining part of the tank with only pipe A
def additionalTime : ℚ := remainingTank / rateA

-- Total time is the sum of the initial time and additional time
def totalTime : ℚ := timeBClosed + additionalTime

theorem fill_tank_time : totalTime = 30 :=
by
  -- Proof omitted
  sorry

end fill_tank_time_l24_24477


namespace solve_for_y_l24_24007

theorem solve_for_y (y : ℝ) : (5:ℝ)^(2*y + 3) = (625:ℝ)^y → y = 3/2 :=
by
  intro h
  sorry

end solve_for_y_l24_24007


namespace derivative_of_y_is_correct_l24_24054

noncomputable def y (x : ℝ) := x^2 * Real.sin x

theorem derivative_of_y_is_correct : (deriv y x = 2 * x * Real.sin x + x^2 * Real.cos x) :=
by
  sorry

end derivative_of_y_is_correct_l24_24054


namespace blocks_differs_in_exactly_two_ways_correct_l24_24016

structure Block where
  material : Bool       -- material: false for plastic, true for wood
  size : Fin 3          -- sizes: 0 for small, 1 for medium, 2 for large
  color : Fin 4         -- colors: 0 for blue, 1 for green, 2 for red, 3 for yellow
  shape : Fin 4         -- shapes: 0 for circle, 1 for hexagon, 2 for square, 3 for triangle
  finish : Bool         -- finish: false for glossy, true for matte

def originalBlock : Block :=
  { material := false, size := 1, color := 2, shape := 0, finish := false }

def differsInExactlyTwoWays (b1 b2 : Block) : Bool :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0) +
  (if b1.finish ≠ b2.finish then 1 else 0) == 2

def countBlocksDifferingInTwoWays : Nat :=
  let allBlocks := List.product
                  (List.product
                    (List.product
                      (List.product
                        [false, true]
                        ([0, 1, 2] : List (Fin 3)))
                      ([0, 1, 2, 3] : List (Fin 4)))
                    ([0, 1, 2, 3] : List (Fin 4)))
                  [false, true]
  (allBlocks.filter
    (λ b => differsInExactlyTwoWays originalBlock
                { material := b.1.1.1.1, size := b.1.1.1.2, color := b.1.1.2, shape := b.1.2, finish := b.2 })).length

theorem blocks_differs_in_exactly_two_ways_correct :
  countBlocksDifferingInTwoWays = 51 :=
  by
    sorry

end blocks_differs_in_exactly_two_ways_correct_l24_24016


namespace set_intersection_set_union_set_complement_l24_24266

open Set

variable (U : Set ℝ) (A B : Set ℝ)
noncomputable def setA : Set ℝ := {x | x^2 - 3*x - 4 ≥ 0}
noncomputable def setB : Set ℝ := {x | x < 5}

theorem set_intersection : (U = univ) -> (A = setA) -> (B = setB) -> A ∩ B = Ico 4 5 := by
  intros
  sorry

theorem set_union : (U = univ) -> (A = setA) -> (B = setB) -> A ∪ B = univ := by
  intros
  sorry

theorem set_complement : (U = univ) -> (A = setA) -> U \ A = Ioo (-1 : ℝ) 4 := by
  intros
  sorry

end set_intersection_set_union_set_complement_l24_24266


namespace dinner_cost_l24_24815

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tip_rate : ℝ)
variable (pre_tax_cost : ℝ)
variable (tip : ℝ)
variable (tax : ℝ)
variable (final_cost : ℝ)

axiom h1 : total_cost = 27.50
axiom h2 : tax_rate = 0.10
axiom h3 : tip_rate = 0.15
axiom h4 : tax = tax_rate * pre_tax_cost
axiom h5 : tip = tip_rate * pre_tax_cost
axiom h6 : final_cost = pre_tax_cost + tax + tip

theorem dinner_cost : pre_tax_cost = 22 := by sorry

end dinner_cost_l24_24815


namespace sum_of_pqrstu_eq_22_l24_24639

theorem sum_of_pqrstu_eq_22 (p q r s t : ℤ) 
  (h : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -48) : 
  p + q + r + s + t = 22 :=
sorry

end sum_of_pqrstu_eq_22_l24_24639


namespace complex_exp_l24_24173

theorem complex_exp {i : ℂ} (h : i^2 = -1) : (1 + i)^30 + (1 - i)^30 = 0 := by
  sorry

end complex_exp_l24_24173


namespace jar_a_marbles_l24_24874

theorem jar_a_marbles : ∃ A : ℕ, (∃ B : ℕ, B = A + 12) ∧ (∃ C : ℕ, C = 2 * (A + 12)) ∧ (A + (A + 12) + 2 * (A + 12) = 148) ∧ (A = 28) :=
by
sorry

end jar_a_marbles_l24_24874


namespace prob_A_two_qualified_l24_24894

noncomputable def prob_qualified (p : ℝ) : ℝ := p * p

def qualified_rate : ℝ := 0.8

theorem prob_A_two_qualified : prob_qualified qualified_rate = 0.64 :=
by
  sorry

end prob_A_two_qualified_l24_24894


namespace least_number_to_divisible_sum_l24_24765

-- Define the conditions and variables
def initial_number : ℕ := 1100
def divisor : ℕ := 23
def least_number_to_add : ℕ := 4

-- Statement to prove
theorem least_number_to_divisible_sum :
  ∃ least_n, least_n + initial_number % divisor = divisor ∧ least_n = least_number_to_add :=
  by
    sorry

end least_number_to_divisible_sum_l24_24765


namespace tanker_filling_rate_l24_24265

theorem tanker_filling_rate :
  let barrels_per_minute := 5
  let liters_per_barrel := 159
  let minutes_per_hour := 60
  let liters_per_cubic_meter := 1000
  (barrels_per_minute * liters_per_barrel * minutes_per_hour) / 
  liters_per_cubic_meter = 47.7 :=
by
  sorry

end tanker_filling_rate_l24_24265


namespace scale_division_remainder_l24_24863

theorem scale_division_remainder (a b c r : ℕ) (h1 : a = b * c + r) (h2 : 0 ≤ r) (h3 : r < b) :
  (3 * a) % (3 * b) = 3 * r :=
sorry

end scale_division_remainder_l24_24863


namespace product_of_solutions_eq_neg_35_l24_24737

theorem product_of_solutions_eq_neg_35 :
  ∀ (x : ℝ), -35 = -x^2 - 2 * x → ∃ (p : ℝ), p = -35 :=
by
  intro x h
  sorry

end product_of_solutions_eq_neg_35_l24_24737


namespace motorcycle_tire_max_distance_l24_24884

theorem motorcycle_tire_max_distance :
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  let s := 18750
  wear_front * (s / 2) + wear_rear * (s / 2) = 1 :=
by 
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  sorry

end motorcycle_tire_max_distance_l24_24884


namespace usual_eggs_accepted_l24_24606

theorem usual_eggs_accepted (A R : ℝ) (h1 : A / R = 1 / 4) (h2 : (A + 12) / (R - 4) = 99 / 1) (h3 : A + R = 400) :
  A = 392 :=
by
  sorry

end usual_eggs_accepted_l24_24606


namespace distance_travelled_downstream_l24_24649

theorem distance_travelled_downstream :
  let speed_boat_still_water := 42 -- km/hr
  let rate_current := 7 -- km/hr
  let time_travelled_min := 44 -- minutes
  let time_travelled_hrs := time_travelled_min / 60.0 -- converting minutes to hours
  let effective_speed_downstream := speed_boat_still_water + rate_current -- km/hr
  let distance_downstream := effective_speed_downstream * time_travelled_hrs
  distance_downstream = 35.93 :=
by
  -- Proof will go here
  sorry

end distance_travelled_downstream_l24_24649


namespace parallel_lines_of_equation_l24_24872

theorem parallel_lines_of_equation (y : Real) :
  (y - 2) * (y + 3) = 0 → (y = 2 ∨ y = -3) :=
by
  sorry

end parallel_lines_of_equation_l24_24872


namespace find_m_of_quadratic_root_l24_24242

theorem find_m_of_quadratic_root
  (m : ℤ) 
  (h : ∃ x : ℤ, x^2 - (m+3)*x + m + 2 = 0 ∧ x = 81) : 
  m = 79 :=
by
  sorry

end find_m_of_quadratic_root_l24_24242


namespace sin_neg_30_eq_neg_half_l24_24565

/-- Prove that the sine of -30 degrees is -1/2 -/
theorem sin_neg_30_eq_neg_half : Real.sin (-(30 * Real.pi / 180)) = -1 / 2 :=
by
  sorry

end sin_neg_30_eq_neg_half_l24_24565


namespace percentage_of_engineers_from_university_A_l24_24401

theorem percentage_of_engineers_from_university_A :
  let original_engineers := 20
  let new_hired_engineers := 8
  let percentage_original_from_A := 0.65
  let original_from_A := percentage_original_from_A * original_engineers
  let total_engineers := original_engineers + new_hired_engineers
  let total_from_A := original_from_A + new_hired_engineers
  (total_from_A / total_engineers) * 100 = 75 :=
by
  sorry

end percentage_of_engineers_from_university_A_l24_24401


namespace ratio_of_buckets_l24_24338

theorem ratio_of_buckets 
  (shark_feed_per_day : ℕ := 4)
  (dolphin_feed_per_day : ℕ := shark_feed_per_day / 2)
  (total_buckets : ℕ := 546)
  (days_in_weeks : ℕ := 3 * 7)
  (ratio_R : ℕ) :
  (total_buckets = days_in_weeks * (shark_feed_per_day + dolphin_feed_per_day + (ratio_R * shark_feed_per_day)) → ratio_R = 5) := sorry

end ratio_of_buckets_l24_24338


namespace smallest_x_for_multiple_of_450_and_648_l24_24394

theorem smallest_x_for_multiple_of_450_and_648 (x : ℕ) (hx : x > 0) :
  ∃ (y : ℕ), (450 * 36) = y ∧ (450 * 36) % 648 = 0 :=
by
  use (450 / gcd 450 648 * 648 / gcd 450 648)
  sorry

end smallest_x_for_multiple_of_450_and_648_l24_24394


namespace range_of_a_minus_abs_b_l24_24897

theorem range_of_a_minus_abs_b (a b : ℝ) (h1 : 1 < a ∧ a < 8) (h2 : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 8 :=
sorry

end range_of_a_minus_abs_b_l24_24897


namespace fraction_eq_l24_24180

theorem fraction_eq (x : ℝ) (h1 : x * 180 = 24) (h2 : x < 20 / 100) : x = 2 / 15 :=
sorry

end fraction_eq_l24_24180


namespace luke_bike_vs_bus_slowness_l24_24770

theorem luke_bike_vs_bus_slowness
  (luke_bus_time : ℕ)
  (paula_ratio : ℚ)
  (total_travel_time : ℕ)
  (paula_total_bus_time : ℕ)
  (luke_total_travel_time_lhs : ℕ)
  (luke_total_travel_time_rhs : ℕ)
  (bike_time : ℕ)
  (ratio : ℚ) :
  luke_bus_time = 70 ∧
  paula_ratio = 3 / 5 ∧
  total_travel_time = 504 ∧
  paula_total_bus_time = 2 * (paula_ratio * luke_bus_time) ∧
  luke_total_travel_time_lhs = luke_bus_time + bike_time ∧
  luke_total_travel_time_rhs + paula_total_bus_time = total_travel_time ∧
  bike_time = ratio * luke_bus_time ∧
  ratio = bike_time / luke_bus_time →
  ratio = 5 :=
sorry

end luke_bike_vs_bus_slowness_l24_24770


namespace tabletop_qualification_l24_24656

theorem tabletop_qualification (length width diagonal : ℕ) :
  length = 60 → width = 32 → diagonal = 68 → (diagonal * diagonal = length * length + width * width) :=
by
  intros
  sorry

end tabletop_qualification_l24_24656


namespace derivative_y_l24_24914

noncomputable def y (a α x : ℝ) :=
  (Real.exp (a * x)) * (3 * Real.sin (3 * x) - α * Real.cos (3 * x)) / (a ^ 2 + 9)

theorem derivative_y (a α x : ℝ) :
  (deriv (y a α) x) =
    (Real.exp (a * x)) * ((3 * a + 3 * α) * Real.sin (3 * x) + (9 - a * α) * Real.cos (3 * x)) / (a ^ 2 + 9) := 
sorry

end derivative_y_l24_24914


namespace total_rainfall_2004_l24_24629

theorem total_rainfall_2004 (avg_2003 : ℝ) (increment : ℝ) (months : ℕ) (total_2004 : ℝ) 
  (h1 : avg_2003 = 41.5) 
  (h2 : increment = 2) 
  (h3 : months = 12) 
  (h4 : total_2004 = avg_2003 + increment * months) :
  total_2004 = 522 :=
by 
  sorry

end total_rainfall_2004_l24_24629


namespace daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l24_24306

-- Given conditions
def cost_price : ℝ := 80
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 320
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * daily_sales_quantity x

-- Part 1: Functional relationship
theorem daily_profit_functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) : daily_profit x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Part 2: Maximizing daily profit
theorem daily_profit_maximizes_at_120 (hx : 80 ≤ 120 ∧ 120 ≤ 160) : daily_profit 120 = 3200 :=
by sorry

-- Part 3: Selling price for a daily profit of $2400
theorem selling_price_for_2400_profit (hx : 80 ≤ 100 ∧ 100 ≤ 160) : daily_profit 100 = 2400 :=
by sorry

end daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l24_24306


namespace young_people_sampled_l24_24623

def num_young_people := 800
def num_middle_aged_people := 1600
def num_elderly_people := 1400
def sampled_elderly_people := 70

-- Lean statement to prove the number of young people sampled
theorem young_people_sampled : 
  (sampled_elderly_people:ℝ) / num_elderly_people = (1 / 20:ℝ) ->
  num_young_people * (1 / 20:ℝ) = 40 := by
  sorry

end young_people_sampled_l24_24623


namespace relationship_xyz_w_l24_24460

theorem relationship_xyz_w (x y z w : ℝ) (h : (x + y) / (y + z) = (2 * z + w) / (w + x)) :
  x = 2 * z - w := 
sorry

end relationship_xyz_w_l24_24460


namespace cone_lateral_surface_area_l24_24984

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l24_24984


namespace red_button_probability_l24_24489

-- Definitions of the initial state
def initial_red_buttons : ℕ := 8
def initial_blue_buttons : ℕ := 12
def total_buttons := initial_red_buttons + initial_blue_buttons

-- Condition of removal and remaining buttons
def removed_buttons := total_buttons - (5 / 8 : ℚ) * total_buttons

-- Equal number of red and blue buttons removed
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

-- State after removal
def remaining_red_buttons := initial_red_buttons - removed_red_buttons
def remaining_blue_buttons := initial_blue_buttons - removed_blue_buttons

-- Jars after removal
def jar_X := remaining_red_buttons + remaining_blue_buttons
def jar_Y := removed_red_buttons + removed_blue_buttons

-- Probability calculations
def probability_red_X : ℚ := remaining_red_buttons / jar_X
def probability_red_Y : ℚ := removed_red_buttons / jar_Y

-- Final probability
def final_probability : ℚ := probability_red_X * probability_red_Y

theorem red_button_probability :
  final_probability = 4 / 25 := 
  sorry

end red_button_probability_l24_24489


namespace length_of_segment_AB_l24_24682

-- Define the parabola and its properties
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ C.1 = 3

-- Main statement of the problem
theorem length_of_segment_AB
  (A B : ℝ × ℝ)
  (hA : parabola_equation A.1 A.2)
  (hB : parabola_equation B.1 B.2)
  (C : ℝ × ℝ)
  (hfoc : focus (1, 0))
  (hm : midpoint_condition A B C) :
  dist A B = 8 :=
by sorry

end length_of_segment_AB_l24_24682


namespace closest_point_on_line_l24_24320

theorem closest_point_on_line 
  (t : ℚ)
  (x y z : ℚ)
  (x_eq : x = 3 + t)
  (y_eq : y = 2 - 3 * t)
  (z_eq : z = -1 + 2 * t)
  (x_ortho_eq : (1 + t) = 0)
  (y_ortho_eq : (3 - 3 * t) = 0)
  (z_ortho_eq : (-3 + 2 * t) = 0) :
  (45/14, 16/14, -1/7) = (x, y, z) := by
  sorry

end closest_point_on_line_l24_24320


namespace square_paintings_size_l24_24510

theorem square_paintings_size (total_area : ℝ) (small_paintings_count : ℕ) (small_painting_area : ℝ) 
                              (large_painting_area : ℝ) (square_paintings_count : ℕ) (square_paintings_total_area : ℝ) : 
  total_area = small_paintings_count * small_painting_area + large_painting_area + square_paintings_total_area → 
  square_paintings_count = 3 → 
  small_paintings_count = 4 → 
  small_painting_area = 2 * 3 → 
  large_painting_area = 10 * 15 → 
  square_paintings_total_area = 3 * 6^2 → 
  ∃ side_length, side_length^2 = (square_paintings_total_area / square_paintings_count) ∧ side_length = 6 := 
by
  intro h_total h_square_count h_small_count h_small_area h_large_area h_square_total 
  use 6
  sorry

end square_paintings_size_l24_24510


namespace volunteers_meet_again_in_360_days_l24_24531

-- Definitions of the given values for the problem
def ella_days := 5
def fiona_days := 6
def george_days := 8
def harry_days := 9

-- Statement of the problem in Lean 4
theorem volunteers_meet_again_in_360_days :
  Nat.lcm (Nat.lcm (Nat.lcm ella_days fiona_days) george_days) harry_days = 360 :=
by
  sorry

end volunteers_meet_again_in_360_days_l24_24531


namespace overall_percentage_support_l24_24937

theorem overall_percentage_support (p_men : ℕ) (p_women : ℕ) (n_men : ℕ) (n_women : ℕ) : 
  (p_men = 55) → (p_women = 80) → (n_men = 200) → (n_women = 800) → 
  (p_men * n_men + p_women * n_women) / (n_men + n_women) = 75 :=
by
  sorry

end overall_percentage_support_l24_24937


namespace initial_goal_proof_l24_24036

def marys_collection (k : ℕ) : ℕ := 5 * k
def scotts_collection (m : ℕ) : ℕ := m / 3
def total_collected (k : ℕ) (m : ℕ) (s : ℕ) : ℕ := k + m + s
def initial_goal (total : ℕ) (excess : ℕ) : ℕ := total - excess

theorem initial_goal_proof : 
  initial_goal (total_collected 600 (marys_collection 600) (scotts_collection (marys_collection 600))) 600 = 4000 :=
by
  sorry

end initial_goal_proof_l24_24036


namespace quadratic_inequality_solution_l24_24573

theorem quadratic_inequality_solution (x : ℝ) :
  (x < -7 ∨ x > 3) → x^2 + 4 * x - 21 > 0 :=
by
  -- The proof will go here
  sorry

end quadratic_inequality_solution_l24_24573


namespace line_through_point_equal_intercepts_l24_24756

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (hP : P = (1, 1)) :
  (∀ x y : ℝ, (x - y = 0 ∨ x + y - 2 = 0) → ∃ k : ℝ, k = 1 ∧ k = 2) :=
by
  sorry

end line_through_point_equal_intercepts_l24_24756


namespace find_ratio_l24_24269

def given_conditions (a b c x y z : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 25 ∧ x^2 + y^2 + z^2 = 36 ∧ a * x + b * y + c * z = 30

theorem find_ratio (a b c x y z : ℝ)
  (h : given_conditions a b c x y z) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
sorry

end find_ratio_l24_24269


namespace value_of_f_at_2_l24_24031

def f (x : ℝ) : ℝ := x^3 - x^2 - 1

theorem value_of_f_at_2 : f 2 = 3 := by
  sorry

end value_of_f_at_2_l24_24031


namespace min_sum_x_y_condition_l24_24154

theorem min_sum_x_y_condition {x y : ℝ} (h₁ : x > 0) (h₂ : y > 0) (h₃ : 1 / x + 9 / y = 1) : x + y = 16 :=
by
  sorry -- proof skipped

end min_sum_x_y_condition_l24_24154


namespace Bernoulli_inequality_l24_24457

theorem Bernoulli_inequality (n : ℕ) (a : ℝ) (h : a > -1) : (1 + a)^n ≥ n * a + 1 := 
sorry

end Bernoulli_inequality_l24_24457


namespace max_u_plus_2v_l24_24848

theorem max_u_plus_2v (u v : ℝ) (h1 : 2 * u + 3 * v ≤ 10) (h2 : 4 * u + v ≤ 9) : u + 2 * v ≤ 6.1 :=
sorry

end max_u_plus_2v_l24_24848


namespace second_consecutive_odd_integer_l24_24443

theorem second_consecutive_odd_integer (x : ℤ) 
  (h1 : ∃ x, x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1) 
  (h2 : (x + 2) + (x + 4) = x + 17) : 
  (x + 2) = 13 :=
by
  sorry

end second_consecutive_odd_integer_l24_24443


namespace exists_nonneg_coefs_some_n_l24_24351

-- Let p(x) be a polynomial with real coefficients
variable (p : Polynomial ℝ)

-- Assumption: p(x) > 0 for all x >= 0
axiom positive_poly : ∀ x : ℝ, x ≥ 0 → p.eval x > 0 

theorem exists_nonneg_coefs_some_n :
  ∃ n : ℕ, ∀ k : ℕ, Polynomial.coeff ((1 + Polynomial.X)^n * p) k ≥ 0 :=
sorry

end exists_nonneg_coefs_some_n_l24_24351


namespace double_angle_value_l24_24260

theorem double_angle_value : 2 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 2 := 
sorry

end double_angle_value_l24_24260


namespace general_formula_a_sum_T_max_k_value_l24_24345

-- Given conditions
noncomputable def S (n : ℕ) : ℚ := (1/2 : ℚ) * n^2 + (11/2 : ℚ) * n
noncomputable def a (n : ℕ) : ℚ := if n = 1 then 6 else n + 5
noncomputable def b (n : ℕ) : ℚ := 3 / ((2 * a n - 11) * (2 * a (n + 1) - 11))
noncomputable def T (n : ℕ) : ℚ := (3 * n) / (2 * n + 1)

-- Proof statements
theorem general_formula_a (n : ℕ) : a n = if n = 1 then 6 else n + 5 :=
by sorry

theorem sum_T (n : ℕ) : T n = (3 * n) / (2 * n + 1) :=
by sorry

theorem max_k_value (k : ℕ) : k = 19 → ∀ n : ℕ, T n > k / 20 :=
by sorry

end general_formula_a_sum_T_max_k_value_l24_24345


namespace tan_alpha_value_l24_24584

theorem tan_alpha_value (α β : ℝ) (h₁ : Real.tan (α + β) = 3) (h₂ : Real.tan β = 2) : 
  Real.tan α = 1 / 7 := 
by 
  sorry

end tan_alpha_value_l24_24584


namespace intersection_of_S_and_complement_of_T_in_U_l24_24883

def U : Set ℕ := { x | 0 ≤ x ∧ x ≤ 8 }
def S : Set ℕ := { 1, 2, 4, 5 }
def T : Set ℕ := { 3, 5, 7 }
def C_U_T : Set ℕ := { x | x ∈ U ∧ x ∉ T }

theorem intersection_of_S_and_complement_of_T_in_U :
  S ∩ C_U_T = { 1, 2, 4 } :=
by
  sorry

end intersection_of_S_and_complement_of_T_in_U_l24_24883


namespace correct_calculation_l24_24505

theorem correct_calculation (x : ℝ) (h : 3 * x - 12 = 60) : (x / 3) + 12 = 20 :=
by 
  sorry

end correct_calculation_l24_24505


namespace andrew_made_35_sandwiches_l24_24688

-- Define the number of friends and sandwiches per friend
def num_friends : ℕ := 7
def sandwiches_per_friend : ℕ := 5

-- Define the total number of sandwiches and prove it equals 35
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

theorem andrew_made_35_sandwiches : total_sandwiches = 35 := by
  sorry

end andrew_made_35_sandwiches_l24_24688


namespace players_at_least_two_sciences_l24_24415

-- Define the conditions of the problem
def total_players : Nat := 30
def players_biology : Nat := 15
def players_chemistry : Nat := 10
def players_physics : Nat := 5
def players_all_three : Nat := 3

-- Define the main theorem we want to prove
theorem players_at_least_two_sciences :
  (players_biology + players_chemistry + players_physics 
    - players_all_three - total_players) = 9 :=
sorry

end players_at_least_two_sciences_l24_24415


namespace isosceles_triangle_angles_l24_24912

theorem isosceles_triangle_angles (a b : ℝ) (h₁ : a = 80 ∨ b = 80) (h₂ : a + b + c = 180) (h_iso : a = b ∨ a = c ∨ b = c) :
  (a = 80 ∧ b = 20 ∧ c = 80)
  ∨ (a = 80 ∧ b = 80 ∧ c = 20)
  ∨ (a = 50 ∧ b = 50 ∧ c = 80) :=
by sorry

end isosceles_triangle_angles_l24_24912


namespace line_equation_slope_intercept_l24_24098

theorem line_equation_slope_intercept (m b : ℝ) (h1 : m = -1) (h2 : b = -1) :
  ∀ x y : ℝ, y = m * x + b → x + y + 1 = 0 :=
by
  intros x y h
  sorry

end line_equation_slope_intercept_l24_24098


namespace parabola_example_l24_24246

theorem parabola_example (p : ℝ) (hp : p > 0)
    (h_intersect : ∀ x y : ℝ, y = x - p / 2 ∧ y^2 = 2 * p * x → ((x - p / 2)^2 = 2 * p * x))
    (h_AB : ∀ A B : ℝ × ℝ, A.2 = A.1 - p / 2 ∧ B.2 = B.1 - p / 2 ∧ |A.1 - B.1| = 8) :
    p = 2 := 
sorry

end parabola_example_l24_24246


namespace range_of_a_l24_24783

variable {f : ℝ → ℝ} {a : ℝ}
open Real

-- Conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def f_positive_at_2 (f : ℝ → ℝ) : Prop := f 2 > 1
def f_value_at_2014 (f : ℝ → ℝ) (a : ℝ) : Prop := f 2014 = (a + 3) / (a - 3)

-- Proof Problem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : odd_function f)
  (h2 : periodic_function f 7)
  (h3 : f_positive_at_2 f)
  (h4 : f_value_at_2014 f a) :
  0 < a ∧ a < 3 :=
sorry

end range_of_a_l24_24783


namespace domain_of_tan_l24_24128

noncomputable def is_excluded_from_domain (x : ℝ) : Prop :=
  ∃ k : ℤ, x = 1 + 6 * k

theorem domain_of_tan {x : ℝ} :
  ∀ x, ¬ is_excluded_from_domain x ↔ ¬ ∃ k : ℤ, x = 1 + 6 * k := 
by 
  sorry

end domain_of_tan_l24_24128


namespace chicken_price_per_pound_l24_24145

theorem chicken_price_per_pound (beef_pounds chicken_pounds : ℕ) (beef_price chicken_price : ℕ)
    (total_amount : ℕ)
    (h_beef_quantity : beef_pounds = 1000)
    (h_beef_cost : beef_price = 8)
    (h_chicken_quantity : chicken_pounds = 2 * beef_pounds)
    (h_total_price : 1000 * beef_price + chicken_pounds * chicken_price = total_amount)
    (h_total_amount : total_amount = 14000) : chicken_price = 3 :=
by
  sorry

end chicken_price_per_pound_l24_24145


namespace find_angle_B_l24_24802

-- Definitions and conditions
variables (α β γ δ : ℝ) -- representing angles ∠A, ∠B, ∠C, and ∠D

-- Given Condition: it's a parallelogram and sum of angles A and C
def quadrilateral_parallelogram (A B C D : ℝ) : Prop :=
  A + C = 200 ∧ A = C ∧ A + B = 180

-- Theorem: Degree of angle B is 80°
theorem find_angle_B (A B C D : ℝ) (h : quadrilateral_parallelogram A B C D) : B = 80 := 
  by sorry

end find_angle_B_l24_24802


namespace intersection_points_count_l24_24726

variables {R : Type*} [LinearOrderedField R]

def line1 (x y : R) : Prop := 3 * y - 2 * x = 1
def line2 (x y : R) : Prop := x + 2 * y = 2
def line3 (x y : R) : Prop := 4 * x - 6 * y = 5

theorem intersection_points_count : 
  ∃ p1 p2 : R × R, 
   (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
   (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
   p1 ≠ p2 ∧ 
   (∀ p : R × R, (line1 p.1 p.2 ∧ line3 p.1 p.2) → False) := 
sorry

end intersection_points_count_l24_24726


namespace white_to_brown_eggs_ratio_l24_24184

-- Define variables W and B (the initial numbers of white and brown eggs respectively)
variable (W B : ℕ)

-- Conditions: 
-- 1. All 5 brown eggs survived.
-- 2. Total number of eggs after dropping is 12.
def egg_conditions : Prop :=
  B = 5 ∧ (W + B) = 12

-- Prove the ratio of white eggs to brown eggs is 7/5 given these conditions.
theorem white_to_brown_eggs_ratio (h : egg_conditions W B) : W / B = 7 / 5 :=
by 
  sorry

end white_to_brown_eggs_ratio_l24_24184


namespace sphere_radius_l24_24058

/-- Given the curved surface area (CSA) of a sphere and its formula, 
    prove that the radius of the sphere is 4 cm.
    Conditions:
    - CSA = 4πr²
    - Curved surface area is 64π cm²
-/
theorem sphere_radius (r : ℝ) (h : 4 * Real.pi * r^2 = 64 * Real.pi) : r = 4 := by
  sorry

end sphere_radius_l24_24058


namespace petya_vasya_three_numbers_equal_l24_24220

theorem petya_vasya_three_numbers_equal (a b c : ℕ) :
  gcd a b = lcm a b ∧ gcd b c = lcm b c ∧ gcd a c = lcm a c → a = b ∧ b = c :=
by
  sorry

end petya_vasya_three_numbers_equal_l24_24220


namespace min_floor_sum_l24_24418

-- Definitions of the conditions
variables (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24)

-- Our main theorem statement
theorem min_floor_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24) :
  (Nat.floor ((a+b) / c) + Nat.floor ((b+c) / a) + Nat.floor ((c+a) / b)) = 6 := 
sorry

end min_floor_sum_l24_24418


namespace deepak_age_l24_24395

theorem deepak_age (A D : ℕ)
  (h1 : A / D = 2 / 3)
  (h2 : A + 5 = 25) :
  D = 30 := 
by
  sorry

end deepak_age_l24_24395


namespace jason_cards_l24_24892

theorem jason_cards :
  (initial_cards - bought_cards = remaining_cards) →
  initial_cards = 676 →
  bought_cards = 224 →
  remaining_cards = 452 :=
by
  intros h1 h2 h3
  sorry

end jason_cards_l24_24892


namespace age_of_child_l24_24767

theorem age_of_child 
  (avg_age_3_years_ago : ℕ)
  (family_size_3_years_ago : ℕ)
  (current_family_size : ℕ)
  (current_avg_age : ℕ)
  (h1 : avg_age_3_years_ago = 17)
  (h2 : family_size_3_years_ago = 5)
  (h3 : current_family_size = 6)
  (h4 : current_avg_age = 17)
  : ∃ age_of_baby : ℕ, age_of_baby = 2 := 
by
  sorry

end age_of_child_l24_24767


namespace initial_number_of_fruits_l24_24526

theorem initial_number_of_fruits (oranges apples limes : ℕ) (h_oranges : oranges = 50)
  (h_apples : apples = 72) (h_oranges_limes : oranges = 2 * limes) (h_apples_limes : apples = 3 * limes) :
  (oranges + apples + limes) * 2 = 288 :=
by
  sorry

end initial_number_of_fruits_l24_24526


namespace compare_values_l24_24943

-- Define that f(x) is an even function, periodic and satisfies decrease and increase conditions as given
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

noncomputable def f : ℝ → ℝ := sorry -- the exact definition of f is unknown, so we use sorry for now

-- The conditions of the problem
axiom f_even : is_even_function f
axiom f_period : periodic_function f 2
axiom f_decreasing : decreasing_on_interval f (-1) 0
axiom f_transformation : ∀ x, f (x + 1) = 1 / f x

-- Prove the comparison between a, b, and c under the given conditions
theorem compare_values (a b c : ℝ) (h1 : a = f (Real.log 2 / Real.log 5)) (h2 : b = f (Real.log 4 / Real.log 2)) (h3 : c = f (Real.sqrt 2)) :
  a > c ∧ c > b :=
by
  sorry

end compare_values_l24_24943


namespace solve_for_q_l24_24836

theorem solve_for_q :
  ∀ (q : ℕ), 16^15 = 4^q → q = 30 :=
by
  intro q
  intro h
  sorry

end solve_for_q_l24_24836


namespace number_chosen_div_8_sub_100_eq_6_l24_24794

variable (n : ℤ)

theorem number_chosen_div_8_sub_100_eq_6 (h : (n / 8) - 100 = 6) : n = 848 := 
by
  sorry

end number_chosen_div_8_sub_100_eq_6_l24_24794


namespace largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l24_24105

noncomputable def largest_integral_x_in_ineq (x : ℤ) : Prop :=
  (2 / 5 : ℚ) < (x / 7 : ℚ) ∧ (x / 7 : ℚ) < (8 / 11 : ℚ)

theorem largest_integral_x_satisfies_ineq : largest_integral_x_in_ineq 5 :=
sorry

theorem largest_integral_x_is_5 (x : ℤ) (h : largest_integral_x_in_ineq x) : x ≤ 5 :=
sorry

end largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l24_24105


namespace c_horses_months_l24_24643

theorem c_horses_months (cost_total Rs_a Rs_b num_horses_a num_months_a num_horses_b num_months_b num_horses_c amount_paid_b : ℕ) (x : ℕ) 
  (h1 : cost_total = 841) 
  (h2 : Rs_a = 12 * 8)
  (h3 : Rs_b = 16 * 9)
  (h4 : amount_paid_b = 348)
  (h5 : 96 * (amount_paid_b / Rs_b) + (18 * x) * (amount_paid_b / Rs_b) = cost_total - amount_paid_b) :
  x = 11 :=
sorry

end c_horses_months_l24_24643


namespace garden_fencing_needed_l24_24478

/-- Given a rectangular garden where the length is 300 yards and the length is twice the width,
prove that the total amount of fencing needed to enclose the garden is 900 yards. -/
theorem garden_fencing_needed :
  ∃ (W L P : ℝ), L = 300 ∧ L = 2 * W ∧ P = 2 * (L + W) ∧ P = 900 :=
by
  sorry

end garden_fencing_needed_l24_24478


namespace Sara_taller_than_Joe_l24_24284

noncomputable def Roy_height := 36

noncomputable def Joe_height := Roy_height + 3

noncomputable def Sara_height := 45

theorem Sara_taller_than_Joe : Sara_height - Joe_height = 6 :=
by
  sorry

end Sara_taller_than_Joe_l24_24284


namespace number_of_pencil_boxes_l24_24636

open Nat

def books_per_box : Nat := 46
def num_book_boxes : Nat := 19
def pencils_per_box : Nat := 170
def total_books_and_pencils : Nat := 1894

theorem number_of_pencil_boxes :
  (total_books_and_pencils - (num_book_boxes * books_per_box)) / pencils_per_box = 6 := 
by
  sorry

end number_of_pencil_boxes_l24_24636


namespace find_number_l24_24816

theorem find_number (x : ℝ) (h : 0.62 * x - 50 = 43) : x = 150 :=
sorry

end find_number_l24_24816


namespace find_z_percentage_of_1000_l24_24302

noncomputable def x := (3 / 5) * 4864
noncomputable def y := (2 / 3) * 9720
noncomputable def z := (1 / 4) * 800

theorem find_z_percentage_of_1000 :
  (2 / 3) * x + (1 / 2) * y = z → (z / 1000) * 100 = 20 :=
by
  sorry

end find_z_percentage_of_1000_l24_24302


namespace right_triangle_legs_l24_24433

theorem right_triangle_legs (a b : ℕ) (h : a^2 + b^2 = 100) (h_r: a + b - 10 = 4) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
sorry

end right_triangle_legs_l24_24433


namespace range_of_a3_l24_24753

theorem range_of_a3 (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) + a n = 4 * n + 3)
  (h2 : ∀ n : ℕ, n > 0 → a n + 2 * n^2 ≥ 0) 
  : 2 ≤ a 3 ∧ a 3 ≤ 19 := 
sorry

end range_of_a3_l24_24753


namespace longest_line_segment_l24_24282

theorem longest_line_segment (total_length_cm : ℕ) (h : total_length_cm = 3000) :
  ∃ n : ℕ, 2 * (n * (n + 1) / 2) ≤ total_length_cm ∧ n = 54 :=
by
  use 54
  sorry

end longest_line_segment_l24_24282


namespace yogurt_count_l24_24223

theorem yogurt_count (Y : ℕ) 
  (ice_cream_cartons : ℕ := 20)
  (cost_ice_cream_per_carton : ℕ := 6)
  (cost_yogurt_per_carton : ℕ := 1)
  (spent_more_on_ice_cream : ℕ := 118)
  (total_cost_ice_cream : ℕ := ice_cream_cartons * cost_ice_cream_per_carton)
  (total_cost_yogurt : ℕ := Y * cost_yogurt_per_carton)
  (expenditure_condition : total_cost_ice_cream = total_cost_yogurt + spent_more_on_ice_cream) :
  Y = 2 :=
by {
  sorry
}

end yogurt_count_l24_24223


namespace total_cost_8_dozen_pencils_2_dozen_notebooks_l24_24550

variable (P N : ℝ)

def eq1 : Prop := 3 * P + 4 * N = 60
def eq2 : Prop := P + N = 15.512820512820513

theorem total_cost_8_dozen_pencils_2_dozen_notebooks :
  eq1 P N ∧ eq2 P N → (96 * P + 24 * N = 520) :=
by
  sorry

end total_cost_8_dozen_pencils_2_dozen_notebooks_l24_24550


namespace sum_of_two_numbers_l24_24566

variable {x y : ℝ}

theorem sum_of_two_numbers (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l24_24566


namespace sin_alpha_value_l24_24618

-- Define the given conditions
def α : ℝ := sorry -- α is an acute angle
def β : ℝ := sorry -- β has an unspecified value

-- Given conditions translated to Lean
def condition1 : Prop := 2 * Real.tan (Real.pi - α) - 3 * Real.cos (Real.pi / 2 + β) + 5 = 0
def condition2 : Prop := Real.tan (Real.pi + α) + 6 * Real.sin (Real.pi + β) = 1

-- Acute angle condition
def α_acute : Prop := 0 < α ∧ α < Real.pi / 2

-- The proof statement
theorem sin_alpha_value (h1 : condition1) (h2 : condition2) (h3 : α_acute) : Real.sin α = 3 * Real.sqrt 10 / 10 :=
by sorry

end sin_alpha_value_l24_24618


namespace condition_on_a_and_b_l24_24471

theorem condition_on_a_and_b (a b p q : ℝ) 
    (h1 : (∀ x : ℝ, (x + a) * (x + b) = x^2 + p * x + q))
    (h2 : p > 0)
    (h3 : q < 0) :
    (a < 0 ∧ b > 0 ∧ b > -a) ∨ (a > 0 ∧ b < 0 ∧ a > -b) :=
by
  sorry

end condition_on_a_and_b_l24_24471


namespace second_discount_percentage_is_20_l24_24039

theorem second_discount_percentage_is_20 
    (normal_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (first_discount_percentage : ℝ)
    (h1 : normal_price = 149.99999999999997)
    (h2 : final_price = 108)
    (h3 : first_discount_percentage = 10)
    (h4 : first_discount = normal_price * (first_discount_percentage / 100)) :
    (((normal_price - first_discount) - final_price) / (normal_price - first_discount)) * 100 = 20 := by
  sorry

end second_discount_percentage_is_20_l24_24039


namespace total_handshakes_l24_24577

theorem total_handshakes (twins_num : ℕ) (triplets_num : ℕ) (twins_sets : ℕ) (triplets_sets : ℕ) (h_twins : twins_sets = 9) (h_triplets : triplets_sets = 6) (h_twins_num : twins_num = 2 * twins_sets) (h_triplets_num: triplets_num = 3 * triplets_sets) (h_handshakes : twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2) = 882): 
  (twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2)) / 2 = 441 :=
by
  sorry

end total_handshakes_l24_24577


namespace range_of_a_l24_24598

def A (x : ℝ) : Prop := (x - 1) * (x - 2) ≥ 0
def B (a x : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, A x ∨ B a x) ↔ a ≤ 1 :=
sorry

end range_of_a_l24_24598


namespace large_number_divisible_by_12_l24_24664

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end large_number_divisible_by_12_l24_24664


namespace average_pastries_per_day_l24_24796

def monday_sales : ℕ := 2
def increment_weekday : ℕ := 2
def increment_weekend : ℕ := 3

def tuesday_sales : ℕ := monday_sales + increment_weekday
def wednesday_sales : ℕ := tuesday_sales + increment_weekday
def thursday_sales : ℕ := wednesday_sales + increment_weekday
def friday_sales : ℕ := thursday_sales + increment_weekday
def saturday_sales : ℕ := friday_sales + increment_weekend
def sunday_sales : ℕ := saturday_sales + increment_weekend

def total_sales_week : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
def average_sales_per_day : ℚ := total_sales_week / 7

theorem average_pastries_per_day : average_sales_per_day = 59 / 7 := by
  sorry

end average_pastries_per_day_l24_24796


namespace hyperbola_equation_focus_and_eccentricity_l24_24605

theorem hyperbola_equation_focus_and_eccentricity (a b : ℝ)
  (h_focus : ∃ c : ℝ, c = 1 ∧ (∃ c_squared : ℝ, c_squared = c ^ 2))
  (h_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 ∧ e = c / a)
  (h_b : b ^ 2 = c ^ 2 - a ^ 2) :
  5 * x^2 - (5 / 4) * y^2 = 1 :=
sorry

end hyperbola_equation_focus_and_eccentricity_l24_24605


namespace unvisited_planet_exists_l24_24818

theorem unvisited_planet_exists (n : ℕ) (h : 1 ≤ n)
  (planets : Fin (2 * n + 1) → ℝ) 
  (distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → planets i ≠ planets j) 
  (expeditions : Fin (2 * n + 1) → Fin (2 * n + 1))
  (closest : ∀ i : Fin (2 * n + 1), expeditions i = i ↔ False) :
  ∃ p : Fin (2 * n + 1), ∀ q : Fin (2 * n + 1), expeditions q ≠ p := sorry

end unvisited_planet_exists_l24_24818


namespace jackson_points_l24_24965

theorem jackson_points (team_total_points : ℕ)
                       (num_other_players : ℕ)
                       (average_points_other_players : ℕ)
                       (points_other_players: ℕ)
                       (points_jackson: ℕ)
                       (h_team_total_points : team_total_points = 65)
                       (h_num_other_players : num_other_players = 5)
                       (h_average_points_other_players : average_points_other_players = 6)
                       (h_points_other_players : points_other_players = num_other_players * average_points_other_players)
                       (h_points_total: points_jackson + points_other_players = team_total_points) :
  points_jackson = 35 :=
by
  -- proof will be done here
  sorry

end jackson_points_l24_24965


namespace original_cost_price_of_car_l24_24486

theorem original_cost_price_of_car (x : ℝ) (y : ℝ) (h1 : y = 0.87 * x) (h2 : 1.20 * y = 54000) :
  x = 54000 / 1.044 :=
by
  sorry

end original_cost_price_of_car_l24_24486


namespace calculate_expression_l24_24136

theorem calculate_expression : 3 * ((-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4) = 81 := 
by sorry

end calculate_expression_l24_24136


namespace carmen_sold_1_box_of_fudge_delights_l24_24508

noncomputable def boxes_of_fudge_delights (total_earned: ℝ) (samoas_price: ℝ) (thin_mints_price: ℝ) (fudge_delights_price: ℝ) (sugar_cookies_price: ℝ) (samoas_sold: ℝ) (thin_mints_sold: ℝ) (sugar_cookies_sold: ℝ): ℝ :=
  let samoas_total := samoas_price * samoas_sold
  let thin_mints_total := thin_mints_price * thin_mints_sold
  let sugar_cookies_total := sugar_cookies_price * sugar_cookies_sold
  let other_cookies_total := samoas_total + thin_mints_total + sugar_cookies_total
  (total_earned - other_cookies_total) / fudge_delights_price

theorem carmen_sold_1_box_of_fudge_delights: boxes_of_fudge_delights 42 4 3.5 5 2 3 2 9 = 1 :=
by
  sorry

end carmen_sold_1_box_of_fudge_delights_l24_24508


namespace problem_statement_l24_24185

noncomputable def f (x : ℝ) := 3 * x ^ 5 + 4 * x ^ 4 - 5 * x ^ 3 + 2 * x ^ 2 + x + 6
noncomputable def d (x : ℝ) := x ^ 3 + 2 * x ^ 2 - x - 3
noncomputable def q (x : ℝ) := 3 * x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) := 19 * x ^ 2 - 11 * x - 57

theorem problem_statement : (f 1 = q 1 * d 1 + r 1) ∧ q 1 + r 1 = -47 := by
  sorry

end problem_statement_l24_24185


namespace student_correct_answers_l24_24570

theorem student_correct_answers (C W : ℕ) (h₁ : C + W = 50) (h₂ : 4 * C - W = 130) : C = 36 := 
by
  sorry

end student_correct_answers_l24_24570


namespace find_f_zero_l24_24035

variable (f : ℝ → ℝ)
variable (hf : ∀ x y : ℝ, f (x + y) = f x + f y + 1 / 2)

theorem find_f_zero : f 0 = -1 / 2 :=
by
  sorry

end find_f_zero_l24_24035


namespace range_a_inequality_l24_24596

theorem range_a_inequality (a : ℝ) : (∀ x : ℝ, (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ 1 < a ∧ a ≤ 2 :=
by {
    sorry
}

end range_a_inequality_l24_24596


namespace tree_height_at_end_of_4_years_l24_24963

theorem tree_height_at_end_of_4_years 
  (initial_growth : ℕ → ℕ)
  (height_7_years : initial_growth 7 = 64)
  (growth_pattern : ∀ n, initial_growth (n + 1) = 2 * initial_growth n) :
  initial_growth 4 = 8 :=
by
  sorry

end tree_height_at_end_of_4_years_l24_24963


namespace taxi_fare_relationship_taxi_fare_relationship_simplified_l24_24085

variable (x : ℝ) (y : ℝ)

-- Conditions
def starting_fare : ℝ := 14
def additional_fare_per_km : ℝ := 2.4
def initial_distance : ℝ := 3
def total_distance (x : ℝ) := x
def total_fare (x : ℝ) (y : ℝ) := y
def distance_condition (x : ℝ) := x > 3

-- Theorem Statement
theorem taxi_fare_relationship (h : distance_condition x) :
  total_fare x y = additional_fare_per_km * (total_distance x - initial_distance) + starting_fare :=
by
  sorry

-- Simplified Theorem Statement
theorem taxi_fare_relationship_simplified (h : distance_condition x) :
  y = 2.4 * x + 6.8 :=
by
  sorry

end taxi_fare_relationship_taxi_fare_relationship_simplified_l24_24085


namespace chairs_per_row_l24_24861

-- Definition of the given conditions
def rows : ℕ := 20
def people_per_chair : ℕ := 5
def total_people : ℕ := 600

-- The statement to be proven
theorem chairs_per_row (x : ℕ) (h : rows * (x * people_per_chair) = total_people) : x = 6 := 
by sorry

end chairs_per_row_l24_24861


namespace sum_of_thetas_l24_24437

noncomputable def theta (k : ℕ) : ℝ := (54 + 72 * k) % 360

theorem sum_of_thetas : (theta 0 + theta 1 + theta 2 + theta 3 + theta 4) = 990 :=
by
  -- proof goes here
  sorry

end sum_of_thetas_l24_24437


namespace correlation_is_1_3_4_l24_24731

def relationship1 := "The relationship between a person's age and their wealth"
def relationship2 := "The relationship between a point on a curve and its coordinates"
def relationship3 := "The relationship between apple production and climate"
def relationship4 := "The relationship between the diameter of the cross-section and the height of the same type of tree in a forest"

def isCorrelation (rel: String) : Bool :=
  if rel == relationship1 ∨ rel == relationship3 ∨ rel == relationship4 then true else false

theorem correlation_is_1_3_4 :
  {relationship1, relationship3, relationship4} = {r | isCorrelation r = true} := 
by
  sorry

end correlation_is_1_3_4_l24_24731


namespace circle_radius_l24_24281

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end circle_radius_l24_24281


namespace monotone_increasing_interval_l24_24432

def f (x : ℝ) := x^2 - 2

theorem monotone_increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y :=
by
  sorry

end monotone_increasing_interval_l24_24432


namespace find_partition_l24_24211

open Nat

def isBad (S : Finset ℕ) : Prop :=
  ∃ T : Finset ℕ, T ⊆ S ∧ T.sum id = 2012

def partition_not_bad (S : Finset ℕ) (n : ℕ) : Prop :=
  ∃ (P : Finset (Finset ℕ)), P.card = n ∧ (∀ p ∈ P, isBad p = false) ∧ (S = P.sup id)

theorem find_partition :
  ∃ n : ℕ, n = 2 ∧ partition_not_bad (Finset.range (2012 - 503) \ Finset.range 503) n :=
by
  sorry

end find_partition_l24_24211


namespace framing_required_l24_24163

/- 
  Problem: A 5-inch by 7-inch picture is enlarged by quadrupling its dimensions.
  A 3-inch-wide border is then placed around each side of the enlarged picture.
  What is the minimum number of linear feet of framing that must be purchased
  to go around the perimeter of the border?
-/
def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

theorem framing_required : (2 * ((original_width * enlargement_factor + 2 * border_width) + (original_height * enlargement_factor + 2 * border_width))) / 12 = 10 :=
by
  sorry

end framing_required_l24_24163


namespace casper_initial_candies_l24_24582

theorem casper_initial_candies 
  (x : ℚ)
  (h1 : ∃ y : ℚ, y = x - (1/4) * x - 3) 
  (h2 : ∃ z : ℚ, z = y - (1/5) * y - 5) 
  (h3 : z - 10 = 10) : x = 224 / 3 :=
by
  sorry

end casper_initial_candies_l24_24582


namespace sum_of_divisors_143_l24_24283

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l24_24283


namespace find_distance_between_B_and_C_l24_24788

def problem_statement : Prop :=
  ∃ (x y : ℝ),
  (y / 75 + x / 145 = 4.8) ∧ 
  ((x + y) / 100 = 2 + y / 70) ∧ 
  x = 290

theorem find_distance_between_B_and_C : problem_statement :=
by
  sorry

end find_distance_between_B_and_C_l24_24788


namespace jesse_money_left_after_mall_l24_24234

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

end jesse_money_left_after_mall_l24_24234


namespace angle_in_first_quadrant_l24_24235

def angle := -999 - 30 / 60 -- defining the angle as -999°30'
def coterminal (θ : Real) : Real := θ + 3 * 360 -- function to compute a coterminal angle

theorem angle_in_first_quadrant : 
  let θ := coterminal angle
  0 <= θ ∧ θ < 90 :=
by
  -- Exact proof steps would go here, but they are omitted as per instructions.
  sorry

end angle_in_first_quadrant_l24_24235


namespace isosceles_triangle_perimeter_l24_24970

theorem isosceles_triangle_perimeter :
  ∀ x y : ℝ, x^2 - 7*x + 10 = 0 → y^2 - 7*y + 10 = 0 → x ≠ y → x + x + y = 12 :=
by
  intros x y hx hy hxy
  -- Place for proof
  sorry

end isosceles_triangle_perimeter_l24_24970


namespace tan_of_11pi_over_4_l24_24729

theorem tan_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 := by
  sorry

end tan_of_11pi_over_4_l24_24729


namespace prob_business_less25_correct_l24_24589

def prob_male : ℝ := 0.4
def prob_female : ℝ := 0.6

def prob_science : ℝ := 0.3
def prob_arts : ℝ := 0.45
def prob_business : ℝ := 0.25

def prob_male_science_25plus : ℝ := 0.4
def prob_male_arts_25plus : ℝ := 0.5
def prob_male_business_25plus : ℝ := 0.35

def prob_female_science_25plus : ℝ := 0.3
def prob_female_arts_25plus : ℝ := 0.45
def prob_female_business_25plus : ℝ := 0.2

def prob_male_science_less25 : ℝ := 1 - prob_male_science_25plus
def prob_male_arts_less25 : ℝ := 1 - prob_male_arts_25plus
def prob_male_business_less25 : ℝ := 1 - prob_male_business_25plus

def prob_female_science_less25 : ℝ := 1 - prob_female_science_25plus
def prob_female_arts_less25 : ℝ := 1 - prob_female_arts_25plus
def prob_female_business_less25 : ℝ := 1 - prob_female_business_25plus

def prob_science_less25 : ℝ := prob_male * prob_science * prob_male_science_less25 + prob_female * prob_science * prob_female_science_less25
def prob_arts_less25 : ℝ := prob_male * prob_arts * prob_male_arts_less25 + prob_female * prob_arts * prob_female_arts_less25
def prob_business_less25 : ℝ := prob_male * prob_business * prob_male_business_less25 + prob_female * prob_business * prob_female_business_less25

theorem prob_business_less25_correct :
    prob_business_less25 = 0.185 :=
by
  -- Theorem statement to be proved (proof omitted)
  sorry

end prob_business_less25_correct_l24_24589


namespace height_of_tower_l24_24647

-- Definitions for points and distances
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 0, y := 0, z := 0 }
def C : Point := { x := 0, y := 0, z := 129 }
def D : Point := { x := 0, y := 0, z := 258 }
def B : Point  := { x := 0, y := 305, z := 305 }

-- Given conditions
def angle_elevation_A_to_B : ℝ := 45 -- degrees
def angle_elevation_D_to_B : ℝ := 60 -- degrees
def distance_A_to_D : ℝ := 258 -- meters

-- The problem is to prove the height of the tower is 305 meters given the conditions
theorem height_of_tower : B.y = 305 :=
by
  -- This spot would contain the actual proof
  sorry

end height_of_tower_l24_24647


namespace smaller_triangle_perimeter_l24_24860

theorem smaller_triangle_perimeter (p : ℕ) (h : p * 3 = 120) : p = 40 :=
sorry

end smaller_triangle_perimeter_l24_24860


namespace hamburgers_sold_last_week_l24_24233

theorem hamburgers_sold_last_week (avg_hamburgers_per_day : ℕ) (days_in_week : ℕ) 
    (h_avg : avg_hamburgers_per_day = 9) (h_days : days_in_week = 7) : 
    avg_hamburgers_per_day * days_in_week = 63 :=
by
  -- Avg hamburgers per day times number of days
  sorry

end hamburgers_sold_last_week_l24_24233


namespace boys_from_other_communities_l24_24743

theorem boys_from_other_communities :
  ∀ (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℕ),
  total_boys = 400 →
  percentage_muslims = 44 →
  percentage_hindus = 28 →
  percentage_sikhs = 10 →
  (total_boys * (100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 72 := 
by
  intros total_boys percentage_muslims percentage_hindus percentage_sikhs h1 h2 h3 h4
  sorry

end boys_from_other_communities_l24_24743


namespace not_both_hit_prob_l24_24543

-- Defining the probabilities
def prob_archer_A_hits : ℚ := 1 / 3
def prob_archer_B_hits : ℚ := 1 / 2

-- Defining event B as both hit the bullseye
def prob_both_hit : ℚ := prob_archer_A_hits * prob_archer_B_hits

-- Defining the complementary event of not both hitting the bullseye
def prob_not_both_hit : ℚ := 1 - prob_both_hit

theorem not_both_hit_prob : prob_not_both_hit = 5 / 6 := by
  -- This is the statement we are trying to prove.
  sorry

end not_both_hit_prob_l24_24543


namespace grandma_gave_each_l24_24975

-- Define the conditions
def gasoline: ℝ := 8
def lunch: ℝ := 15.65
def gifts: ℝ := 5 * 2  -- $5 each for two persons
def total_spent: ℝ := gasoline + lunch + gifts
def initial_amount: ℝ := 50
def amount_left: ℝ := 36.35

-- Define the proof problem
theorem grandma_gave_each :
  (amount_left - (initial_amount - total_spent)) / 2 = 10 :=
by
  sorry

end grandma_gave_each_l24_24975


namespace fence_width_l24_24022

theorem fence_width (L W : ℝ) 
  (circumference_eq : 2 * (L + W) = 30)
  (width_eq : W = 2 * L) : 
  W = 10 :=
by 
  sorry

end fence_width_l24_24022


namespace determine_irrational_option_l24_24865

def is_irrational (x : ℝ) : Prop := ¬ ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

def option_A : ℝ := 7
def option_B : ℝ := 0.5
def option_C : ℝ := abs (3 / 20 : ℚ)
def option_D : ℝ := 0.5151151115 -- Assume notation describes the stated behavior

theorem determine_irrational_option :
  is_irrational option_D ∧
  ¬ is_irrational option_A ∧
  ¬ is_irrational option_B ∧
  ¬ is_irrational option_C := 
by
  sorry

end determine_irrational_option_l24_24865


namespace part1_part2_l24_24187

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l24_24187


namespace simplify_fraction_l24_24514

theorem simplify_fraction (a b : ℕ) (h : a = 2020) (h2 : b = 2018) :
  (2 ^ a - 2 ^ b) / (2 ^ a + 2 ^ b) = 3 / 5 := by
  sorry

end simplify_fraction_l24_24514


namespace range_of_a_l24_24727

open Classical

noncomputable def parabola_line_common_point_range (a : ℝ) : Prop :=
  ∃ (k : ℝ), ∃ (x : ℝ), ∃ (y : ℝ), 
  (y = a * x ^ 2) ∧ ((y + 2 = k * (x - 1)) ∨ (y + 2 = - (1 / k) * (x - 1)))

theorem range_of_a (a : ℝ) : 
  (∃ k : ℝ, ∃ x : ℝ, ∃ y : ℝ, 
    y = a * x ^ 2 ∧ (y + 2 = k * (x - 1) ∨ y + 2 = - (1 / k) * (x - 1))) ↔ 
  0 < a ∧ a <= 1 / 8 :=
sorry

end range_of_a_l24_24727


namespace ian_money_left_l24_24052

theorem ian_money_left
  (hours_worked : ℕ)
  (hourly_rate : ℕ)
  (spending_percentage : ℚ)
  (total_earnings : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (h_worked : hours_worked = 8)
  (h_rate : hourly_rate = 18)
  (h_spending : spending_percentage = 0.5)
  (h_earnings : total_earnings = hours_worked * hourly_rate)
  (h_spent : amount_spent = total_earnings * spending_percentage)
  (h_left : amount_left = total_earnings - amount_spent) :
  amount_left = 72 := 
  sorry

end ian_money_left_l24_24052


namespace mike_total_cards_l24_24055

-- Given conditions
def mike_original_cards : ℕ := 87
def sam_given_cards : ℕ := 13

-- Question equivalence in Lean: Prove that Mike has 100 baseball cards now
theorem mike_total_cards : mike_original_cards + sam_given_cards = 100 :=
by 
  sorry

end mike_total_cards_l24_24055


namespace probability_jqk_3_13_l24_24392

def probability_jack_queen_king (total_cards jacks queens kings : ℕ) : ℚ :=
  (jacks + queens + kings) / total_cards

theorem probability_jqk_3_13 :
  probability_jack_queen_king 52 4 4 4 = 3 / 13 := by
  sorry

end probability_jqk_3_13_l24_24392


namespace greatest_number_of_groups_l24_24493

theorem greatest_number_of_groups (s a t b n : ℕ) (hs : s = 10) (ha : a = 15) (ht : t = 12) (hb : b = 18) :
  (∀ n, n ≤ n ∧ n ∣ s ∧ n ∣ a ∧ n ∣ t ∧ n ∣ b ∧ n > 1 → 
  (s / n < (a / n) + (t / n) + (b / n))
  ∧ (∃ groups, groups = n)) → n = 3 :=
sorry

end greatest_number_of_groups_l24_24493


namespace mary_carrots_correct_l24_24991

def sandy_carrots := 8
def total_carrots := 14

def mary_carrots := total_carrots - sandy_carrots

theorem mary_carrots_correct : mary_carrots = 6 := by
  unfold mary_carrots
  unfold total_carrots
  unfold sandy_carrots
  sorry

end mary_carrots_correct_l24_24991


namespace common_ratio_infinite_geometric_series_l24_24406

theorem common_ratio_infinite_geometric_series :
  let a₁ := (4 : ℚ) / 7
  let a₂ := (16 : ℚ) / 49
  let a₃ := (64 : ℚ) / 343
  let r := a₂ / a₁
  r = 4 / 7 :=
by
  sorry

end common_ratio_infinite_geometric_series_l24_24406


namespace solve_for_y_l24_24025

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/4) - 3 * (y / y^(3/4)) = 12 + y^(1/4)) : y = 1296 := by
  sorry

end solve_for_y_l24_24025


namespace parabola_vertex_l24_24228

-- Definition of the quadratic function representing the parabola
def parabola (x : ℝ) : ℝ := (3 * x - 1) ^ 2 + 2

-- Statement asserting the coordinates of the vertex of the given parabola
theorem parabola_vertex :
  ∃ h k : ℝ, ∀ x : ℝ, parabola x = 9 * (x - h) ^ 2 + k ∧ h = 1/3 ∧ k = 2 :=
by
  sorry

end parabola_vertex_l24_24228


namespace min_ge_n_l24_24271

theorem min_ge_n (x y z n : ℕ) (h : x^n + y^n = z^n) : min x y ≥ n :=
sorry

end min_ge_n_l24_24271


namespace probability_third_winning_l24_24425

-- Definitions based on the conditions provided
def num_tickets : ℕ := 10
def num_winning_tickets : ℕ := 3
def num_non_winning_tickets : ℕ := num_tickets - num_winning_tickets

-- Define the probability function
def probability_of_third_draw_winning : ℚ :=
  (num_non_winning_tickets / num_tickets) * 
  ((num_non_winning_tickets - 1) / (num_tickets - 1)) * 
  (num_winning_tickets / (num_tickets - 2))

-- The theorem to prove
theorem probability_third_winning : probability_of_third_draw_winning = 7 / 40 :=
  by sorry

end probability_third_winning_l24_24425


namespace max_grapes_leftover_l24_24371

-- Define variables and conditions
def total_grapes (n : ℕ) : ℕ := n
def kids : ℕ := 5
def grapes_leftover (n : ℕ) : ℕ := n % kids

-- The proposition we need to prove
theorem max_grapes_leftover (n : ℕ) (h : n ≥ 5) : grapes_leftover n = 4 :=
sorry

end max_grapes_leftover_l24_24371


namespace price_of_peaches_is_2_l24_24327

noncomputable def price_per_pound_peaches (total_spent: ℝ) (price_per_pound_other: ℝ) (total_weight_peaches: ℝ) (total_weight_apples: ℝ) (total_weight_blueberries: ℝ) : ℝ :=
  (total_spent - (total_weight_apples + total_weight_blueberries) * price_per_pound_other) / total_weight_peaches

theorem price_of_peaches_is_2 
  (total_spent: ℝ := 51)
  (price_per_pound_other: ℝ := 1)
  (num_peach_pies: ℕ := 5)
  (num_apple_pies: ℕ := 4)
  (num_blueberry_pies: ℕ := 3)
  (weight_per_pie: ℝ := 3):
  price_per_pound_peaches total_spent price_per_pound_other 
                          (num_peach_pies * weight_per_pie) 
                          (num_apple_pies * weight_per_pie) 
                          (num_blueberry_pies * weight_per_pie) = 2 := 
by
  sorry

end price_of_peaches_is_2_l24_24327


namespace ratio_of_birds_to_trees_and_stones_l24_24956

theorem ratio_of_birds_to_trees_and_stones (stones birds : ℕ) (h_stones : stones = 40)
  (h_birds : birds = 400) (trees : ℕ) (h_trees : trees = 3 * stones + stones) :
  (birds : ℚ) / (trees + stones) = 2 :=
by
  -- The actual proof steps would go here.
  sorry

end ratio_of_birds_to_trees_and_stones_l24_24956


namespace find_AD_length_l24_24382

variables (A B C D O : Point)
variables (BO OD AO OC AB AD : ℝ)

def quadrilateral_properties (BO OD AO OC AB : ℝ) (O : Point) : Prop :=
  BO = 3 ∧ OD = 9 ∧ AO = 5 ∧ OC = 2 ∧ AB = 7

theorem find_AD_length (h : quadrilateral_properties BO OD AO OC AB O) : AD = Real.sqrt 151 :=
by
  sorry

end find_AD_length_l24_24382


namespace inequality_holds_l24_24890

theorem inequality_holds 
  (a b c : ℝ) 
  (h1 : a > 0)
  (h2 : b < 0) 
  (h3 : b > c) : 
  (a / (c^2)) > (b / (c^2)) :=
by
  sorry

end inequality_holds_l24_24890


namespace coords_A_l24_24061

def A : ℝ × ℝ := (1, -2)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

def A' : ℝ × ℝ := reflect_y_axis A

def A'' : ℝ × ℝ := move_up A' 3

theorem coords_A'' : A'' = (-1, 1) := by
  sorry

end coords_A_l24_24061


namespace sum_x_y_650_l24_24721

theorem sum_x_y_650 (x y : ℤ) (h1 : x - y = 200) (h2 : y = 225) : x + y = 650 :=
by
  sorry

end sum_x_y_650_l24_24721


namespace compute_operation_value_l24_24585

def operation (a b c : ℝ) : ℝ := b^3 - 3 * a * b * c - 4 * a * c^2

theorem compute_operation_value : operation 2 (-1) 4 = -105 :=
by
  sorry

end compute_operation_value_l24_24585


namespace number_of_non_officers_l24_24785

theorem number_of_non_officers 
  (avg_salary_employees: ℝ) (avg_salary_officers: ℝ) (avg_salary_nonofficers: ℝ) 
  (num_officers: ℕ) (num_nonofficers: ℕ):
  avg_salary_employees = 120 ∧ avg_salary_officers = 440 ∧ avg_salary_nonofficers = 110 ∧
  num_officers = 15 ∧ 
  (15 * 440 + num_nonofficers * 110 = (15 + num_nonofficers) * 120)  → 
  num_nonofficers = 480 := 
by 
sorry

end number_of_non_officers_l24_24785


namespace average_s_t_l24_24817

theorem average_s_t (s t : ℝ) 
  (h : (1 + 3 + 7 + s + t) / 5 = 12) : 
  (s + t) / 2 = 24.5 :=
by
  sorry

end average_s_t_l24_24817


namespace greatest_multiple_of_30_less_than_1000_l24_24000

theorem greatest_multiple_of_30_less_than_1000 : ∃ (n : ℕ), n < 1000 ∧ n % 30 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 30 = 0 → m ≤ n := 
by 
  use 990
  sorry

end greatest_multiple_of_30_less_than_1000_l24_24000


namespace john_payment_l24_24833

noncomputable def amount_paid_by_john := (3 * 12) / 2

theorem john_payment : amount_paid_by_john = 18 :=
by
  sorry

end john_payment_l24_24833


namespace peter_candles_l24_24030

theorem peter_candles (candles_rupert : ℕ) (ratio : ℝ) (candles_peter : ℕ) 
  (h1 : ratio = 3.5) (h2 : candles_rupert = 35) (h3 : candles_peter = candles_rupert / ratio) : 
  candles_peter = 10 := 
sorry

end peter_candles_l24_24030


namespace find_constant_e_l24_24797

theorem find_constant_e {x y e : ℝ} : (x / (2 * y) = 3 / e) → ((7 * x + 4 * y) / (x - 2 * y) = 25) → (e = 2) :=
by
  intro h1 h2
  sorry

end find_constant_e_l24_24797


namespace find_c_l24_24487

-- Define the problem
def parabola (x y : ℝ) (a : ℝ) : Prop := 
  x = a * (y - 3) ^ 2 + 5

def point (x y : ℝ) (a : ℝ) : Prop := 
  7 = a * (6 - 3) ^ 2 + 5

-- Theorem to be proved
theorem find_c (a : ℝ) (c : ℝ) (h1 : parabola 7 6 a) (h2 : point 7 6 a) : c = 7 :=
by
  sorry

end find_c_l24_24487


namespace valid_words_count_l24_24572

noncomputable def count_valid_words : Nat :=
  let total_possible_words : Nat := ((25^1) + (25^2) + (25^3) + (25^4) + (25^5))
  let total_possible_words_without_B : Nat := ((24^1) + (24^2) + (24^3) + (24^4) + (24^5))
  total_possible_words - total_possible_words_without_B

theorem valid_words_count : count_valid_words = 1864701 :=
by
  let total_1_letter_words := 25^1
  let total_2_letter_words := 25^2
  let total_3_letter_words := 25^3
  let total_4_letter_words := 25^4
  let total_5_letter_words := 25^5

  let total_words_without_B_1_letter := 24^1
  let total_words_without_B_2_letter := 24^2
  let total_words_without_B_3_letter := 24^3
  let total_words_without_B_4_letter := 24^4
  let total_words_without_B_5_letter := 24^5

  let valid_1_letter_words := total_1_letter_words - total_words_without_B_1_letter
  let valid_2_letter_words := total_2_letter_words - total_words_without_B_2_letter
  let valid_3_letter_words := total_3_letter_words - total_words_without_B_3_letter
  let valid_4_letter_words := total_4_letter_words - total_words_without_B_4_letter
  let valid_5_letter_words := total_5_letter_words - total_words_without_B_5_letter

  let valid_words := valid_1_letter_words + valid_2_letter_words + valid_3_letter_words + valid_4_letter_words + valid_5_letter_words
  sorry

end valid_words_count_l24_24572


namespace limit_of_f_at_infinity_l24_24972

open Filter
open Topology

variable (f : ℝ → ℝ)
variable (h_continuous : Continuous f)
variable (h_seq_limit : ∀ α > 0, Tendsto (fun n : ℕ => f (n * α)) atTop (nhds 0))

theorem limit_of_f_at_infinity : Tendsto f atTop (nhds 0) := by
  sorry

end limit_of_f_at_infinity_l24_24972


namespace flowers_per_row_correct_l24_24241

/-- Definition for the number of each type of flower. -/
def num_yellow_flowers : ℕ := 12
def num_green_flowers : ℕ := 2 * num_yellow_flowers -- Given that green flowers are twice the yellow flowers.
def num_red_flowers : ℕ := 42

/-- Total number of flowers. -/
def total_flowers : ℕ := num_yellow_flowers + num_green_flowers + num_red_flowers

/-- Number of rows in the garden. -/
def num_rows : ℕ := 6

/-- The number of flowers per row Wilma's garden has. -/
def flowers_per_row : ℕ := total_flowers / num_rows

/-- Proof statement: flowers per row should be 13. -/
theorem flowers_per_row_correct : flowers_per_row = 13 :=
by
  -- The proof will go here.
  sorry

end flowers_per_row_correct_l24_24241


namespace company_pays_per_box_per_month_l24_24685

/-
  Given:
  - The dimensions of each box are 15 inches by 12 inches by 10 inches
  - The total volume occupied by all boxes is 1,080,000 cubic inches
  - The total cost for record storage per month is $480

  Prove:
  - The company pays $0.80 per box per month for record storage
-/

theorem company_pays_per_box_per_month :
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  cost_per_box_per_month = 0.80 :=
by
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  sorry

end company_pays_per_box_per_month_l24_24685


namespace volume_of_hall_l24_24616

-- Define the dimensions and areas conditions
def length_hall : ℝ := 15
def breadth_hall : ℝ := 12
def area_floor_ceiling : ℝ := 2 * (length_hall * breadth_hall)
def area_walls (h : ℝ) : ℝ := 2 * (length_hall * h) + 2 * (breadth_hall * h)

-- Given condition: The sum of the areas of the floor and ceiling is equal to the sum of the areas of the four walls
def condition (h : ℝ) : Prop := area_floor_ceiling = area_walls h

-- Define the volume of the hall
def volume_hall (h : ℝ) : ℝ := length_hall * breadth_hall * h

-- The theorem to be proven: given the condition, the volume equals 8004
theorem volume_of_hall : ∃ h : ℝ, condition h ∧ volume_hall h = 8004 := by
  sorry

end volume_of_hall_l24_24616


namespace total_worth_of_travelers_checks_l24_24787

variable (x y : ℕ)

theorem total_worth_of_travelers_checks
  (h1 : x + y = 30)
  (h2 : 50 * (x - 15) + 100 * y = 1050) :
  50 * x + 100 * y = 1800 :=
sorry

end total_worth_of_travelers_checks_l24_24787


namespace smaller_angle_at_6_30_l24_24841
-- Import the Mathlib library

-- Define the conditions as a structure
structure ClockAngleConditions where
  hours_on_clock : ℕ
  degrees_per_hour : ℕ
  minute_hand_position : ℕ
  hour_hand_position : ℕ

-- Initialize the conditions for 6:30
def conditions : ClockAngleConditions := {
  hours_on_clock := 12,
  degrees_per_hour := 30,
  minute_hand_position := 180,
  hour_hand_position := 195
}

-- Define the theorem to be proven
theorem smaller_angle_at_6_30 (c : ClockAngleConditions) : 
  c.hour_hand_position - c.minute_hand_position = 15 :=
by
  -- Skip the proof
  sorry

end smaller_angle_at_6_30_l24_24841


namespace digit_makes_5678d_multiple_of_9_l24_24099

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

theorem digit_makes_5678d_multiple_of_9 (d : Nat) (h : d ≥ 0 ∧ d < 10) :
  is_multiple_of_9 (5 * 10000 + 6 * 1000 + 7 * 100 + 8 * 10 + d) ↔ d = 1 := 
by
  sorry

end digit_makes_5678d_multiple_of_9_l24_24099


namespace negation_proof_l24_24881

theorem negation_proof :
  ¬(∀ x : ℝ, x > 0 → Real.exp x > x + 1) ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x ≤ x + 1 :=
by sorry

end negation_proof_l24_24881


namespace sub_base8_l24_24551

theorem sub_base8 : (1352 - 674) == 1456 :=
by sorry

end sub_base8_l24_24551


namespace isosceles_triangle_exists_l24_24741

-- Definitions for a triangle vertex and side lengths
structure Triangle :=
  (A B C : ℝ × ℝ) -- Vertices A, B, C
  (AB AC BC : ℝ)  -- Sides AB, AC, BC

-- Definition for all sides being less than 1 unit
def sides_less_than_one (T : Triangle) : Prop :=
  T.AB < 1 ∧ T.AC < 1 ∧ T.BC < 1

-- Definition for isosceles triangle containing the original one
def exists_isosceles_containing (T : Triangle) : Prop :=
  ∃ (T' : Triangle), 
    (T'.AB = T'.AC ∨ T'.AB = T'.BC ∨ T'.AC = T'.BC) ∧
    T'.A = T.A ∧ -- T'.A vertex is same as T.A
    (T'.AB < 1 ∧ T'.AC < 1 ∧ T'.BC < 1) ∧
    (∃ (B1 : ℝ × ℝ), -- There exists point B1 such that new triangle T' incorporates B1
      T'.B = B1 ∧
      T'.C = T.C) -- T' also has vertex C of original triangle

-- Complete theorem statement
theorem isosceles_triangle_exists (T : Triangle) (hT : sides_less_than_one T) : exists_isosceles_containing T :=
by 
  sorry

end isosceles_triangle_exists_l24_24741


namespace max_lattice_points_in_unit_circle_l24_24019

-- Define a point with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the condition for a lattice point to be strictly inside a given circle
def strictly_inside_circle (p : LatticePoint) (center : Prod ℤ ℤ) (r : ℝ) : Prop :=
  let dx := (p.x - center.fst : ℝ)
  let dy := (p.y - center.snd : ℝ)
  dx^2 + dy^2 < r^2

-- Define the problem statement
theorem max_lattice_points_in_unit_circle : ∀ (center : Prod ℤ ℤ) (r : ℝ),
  r = 1 → 
  ∃ (ps : Finset LatticePoint), 
    (∀ p ∈ ps, strictly_inside_circle p center r) ∧ 
    ps.card = 4 :=
by
  sorry

end max_lattice_points_in_unit_circle_l24_24019


namespace sample_size_l24_24536

theorem sample_size (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : 8 = n * 2 / 10) : n = 40 :=
by
  sorry

end sample_size_l24_24536


namespace tigers_wins_l24_24659

def totalGames : ℕ := 56
def losses : ℕ := 12
def ties : ℕ := losses / 2

theorem tigers_wins : totalGames - losses - ties = 38 := by
  sorry

end tigers_wins_l24_24659


namespace terminating_decimal_expansion_of_17_div_625_l24_24684

theorem terminating_decimal_expansion_of_17_div_625 : 
  ∃ d : ℚ, d = 17 / 625 ∧ d = 0.0272 :=
by
  sorry

end terminating_decimal_expansion_of_17_div_625_l24_24684


namespace min_value_expression_l24_24869

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    9 ≤ (5 * z / (2 * x + y) + 5 * x / (y + 2 * z) + 2 * y / (x + z) + (x + y + z) / (x * y + y * z + z * x)) :=
sorry

end min_value_expression_l24_24869


namespace circle_sum_value_l24_24500

-- Define the problem
theorem circle_sum_value (a b x : ℕ) (h1 : a = 35) (h2 : b = 47) : x = a + b :=
by
  -- Given conditions
  have ha : a = 35 := h1
  have hb : b = 47 := h2
  -- Prove that the value of x is the sum of a and b
  have h_sum : x = a + b := sorry
  -- Assert the value of x is 82 based on given a and b
  exact h_sum

end circle_sum_value_l24_24500


namespace max_area_central_angle_l24_24805

theorem max_area_central_angle (r l : ℝ) (S α : ℝ) (h1 : 2 * r + l = 4)
  (h2 : S = (1 / 2) * l * r) : (∀ x y : ℝ, (1 / 2) * x * y ≤ (1 / 4) * ((x + y) / 2) ^ 2) → α = l / r → α = 2 :=
by
  sorry

end max_area_central_angle_l24_24805


namespace value_of_alpha_beta_l24_24481

variable (α β : ℝ)

-- Conditions
def quadratic_eq (x: ℝ) : Prop := x^2 + 2*x - 2005 = 0

-- Lean 4 statement
theorem value_of_alpha_beta 
  (hα : quadratic_eq α) 
  (hβ : quadratic_eq β)
  (sum_roots : α + β = -2) :
  α^2 + 3*α + β = 2003 :=
sorry

end value_of_alpha_beta_l24_24481


namespace find_integers_l24_24509

theorem find_integers (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) : 
  (a, b, c, d) = (1, 2, 3, 4) ∨ (a, b, c, d) = (1, 2, 4, 3) ∨ (a, b, c, d) = (1, 3, 2, 4) ∨ (a, b, c, d) = (1, 3, 4, 2) ∨ (a, b, c, d) = (1, 4, 2, 3) ∨ (a, b, c, d) = (1, 4, 3, 2) ∨ (a, b, c, d) = (2, 1, 3, 4) ∨ (a, b, c, d) = (2, 1, 4, 3) ∨ (a, b, c, d) = (2, 3, 1, 4) ∨ (a, b, c, d) = (2, 3, 4, 1) ∨ (a, b, c, d) = (2, 4, 1, 3) ∨ (a, b, c, d) = (2, 4, 3, 1) ∨ (a, b, c, d) = (3, 1, 2, 4) ∨ (a, b, c, d) = (3, 1, 4, 2) ∨ (a, b, c, d) = (3, 2, 1, 4) ∨ (a, b, c, d) = (3, 2, 4, 1) ∨ (a, b, c, d) = (3, 4, 1, 2) ∨ (a, b, c, d) = (3, 4, 2, 1) ∨ (a, b, c, d) = (4, 1, 2, 3) ∨ (a, b, c, d) = (4, 1, 3, 2) ∨ (a, b, c, d) = (4, 2, 1, 3) ∨ (a, b, c, d) = (4, 2, 3, 1) ∨ (a, b, c, d) = (4, 3, 1, 2) ∨ (a, b, c, d) = (4, 3, 2, 1) :=
sorry

end find_integers_l24_24509


namespace color_cube_color_octahedron_l24_24657

theorem color_cube (colors : Fin 6) : ∃ (ways : Nat), ways = 30 :=
  sorry

theorem color_octahedron (colors : Fin 8) : ∃ (ways : Nat), ways = 1680 :=
  sorry

end color_cube_color_octahedron_l24_24657


namespace find_a_l24_24524

theorem find_a (a b c : ℝ) (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
                 (h2 : a * 15 * 7 = 1.5) : a = 6 :=
sorry

end find_a_l24_24524


namespace maria_change_l24_24048

def cost_per_apple : ℝ := 0.75
def number_of_apples : ℕ := 5
def amount_paid : ℝ := 10.0
def total_cost := number_of_apples * cost_per_apple
def change_received := amount_paid - total_cost

theorem maria_change :
  change_received = 6.25 :=
sorry

end maria_change_l24_24048


namespace compound_carbon_atoms_l24_24448

-- Definition of data given in the problem.
def molecular_weight : ℕ := 60
def hydrogen_atoms : ℕ := 4
def oxygen_atoms : ℕ := 2
def carbon_atomic_weight : ℕ := 12
def hydrogen_atomic_weight : ℕ := 1
def oxygen_atomic_weight : ℕ := 16

-- Statement to prove the number of carbon atoms in the compound.
theorem compound_carbon_atoms : 
  (molecular_weight - (hydrogen_atoms * hydrogen_atomic_weight + oxygen_atoms * oxygen_atomic_weight)) / carbon_atomic_weight = 2 := 
by
  sorry

end compound_carbon_atoms_l24_24448


namespace find_smallest_N_l24_24115

-- Define the sum of digits functions as described
def sum_of_digits_base (n : ℕ) (b : ℕ) : ℕ :=
  n.digits b |>.sum

-- Define f(n) which is the sum of digits in base-five representation of n
def f (n : ℕ) : ℕ :=
  sum_of_digits_base n 5

-- Define g(n) which is the sum of digits in base-seven representation of f(n)
def g (n : ℕ) : ℕ :=
  sum_of_digits_base (f n) 7

-- The statement of the problem: find the smallest N such that 
-- g(N) in base-sixteen cannot be represented using only digits 0 to 9
theorem find_smallest_N : ∃ N : ℕ, (g N ≥ 10) ∧ (N % 1000 = 610) :=
by
  sorry

end find_smallest_N_l24_24115


namespace no_integer_roots_l24_24182

theorem no_integer_roots : ∀ x : ℤ, x^3 - 3 * x^2 - 16 * x + 20 ≠ 0 := by
  intro x
  sorry

end no_integer_roots_l24_24182


namespace mean_of_five_numbers_l24_24453

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l24_24453


namespace system1_solution_system2_solution_l24_24768

theorem system1_solution (x y : ℚ) :
  x + y = 4 ∧ 5 * (x - y) - 2 * (x + y) = -1 →
  x = 27 / 10 ∧ y = 13 / 10 := by
  sorry

theorem system2_solution (x y : ℚ) :
  (2 * (x - y) / 3) - ((x + y) / 4) = -1 / 12 ∧ 3 * (x + y) - 2 * (2 * x - y) = 3 →
  x = 2 ∧ y = 1 := by
  sorry

end system1_solution_system2_solution_l24_24768


namespace projectiles_initial_distance_l24_24638

theorem projectiles_initial_distance (Projectile1_speed Projectile2_speed Time_to_meet : ℕ) 
  (h1 : Projectile1_speed = 444)
  (h2 : Projectile2_speed = 555)
  (h3 : Time_to_meet = 2) : 
  (Projectile1_speed + Projectile2_speed) * Time_to_meet = 1998 := by
  sorry

end projectiles_initial_distance_l24_24638


namespace peter_fraction_is_1_8_l24_24304

-- Define the total number of slices, slices Peter ate alone, and slices Peter shared with Paul
def total_slices := 16
def peter_alone_slices := 1
def shared_slices := 2

-- Define the fraction of the pizza Peter ate alone
def peter_fraction_alone := peter_alone_slices / total_slices

-- Define the fraction of the pizza Peter ate from the shared slices
def shared_fraction := shared_slices * (1 / 2) / total_slices

-- Define the total fraction of the pizza Peter ate
def total_fraction_peter_ate := peter_fraction_alone + shared_fraction

-- Prove that the total fraction of the pizza Peter ate is 1/8
theorem peter_fraction_is_1_8 : total_fraction_peter_ate = 1/8 := by
  sorry

end peter_fraction_is_1_8_l24_24304


namespace sticks_form_equilateral_triangle_l24_24488

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end sticks_form_equilateral_triangle_l24_24488


namespace common_points_count_l24_24994

noncomputable def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
noncomputable def eq2 (x y : ℝ) : Prop := (x + 2 * y - 5) * (3 * x - 4 * y + 6) = 0

theorem common_points_count : 
  (∃ x1 y1 : ℝ, eq1 x1 y1 ∧ eq2 x1 y1) ∧
  (∃ x2 y2 : ℝ, eq1 x2 y2 ∧ eq2 x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2)) ∧
  (∃ x3 y3 : ℝ, eq1 x3 y3 ∧ eq2 x3 y3 ∧ (x3 ≠ x1 ∧ x3 ≠ x2 ∧ y3 ≠ y1 ∧ y3 ≠ y2)) ∧ 
  (∃ x4 y4 : ℝ, eq1 x4 y4 ∧ eq2 x4 y4 ∧ (x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ y4 ≠ y1 ∧ y4 ≠ y2 ∧ y4 ≠ y3)) ∧ 
  ∀ x y : ℝ, (eq1 x y ∧ eq2 x y) → (((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4))) :=
by
  sorry

end common_points_count_l24_24994


namespace fg_of_3_l24_24620

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem fg_of_3 : f (g 3) = 25 := by
  sorry

end fg_of_3_l24_24620


namespace spacy_subsets_15_l24_24352

def spacy_subsets_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | (k + 5) => spacy_subsets_count (k + 4) + spacy_subsets_count k

theorem spacy_subsets_15 : spacy_subsets_count 15 = 181 :=
sorry

end spacy_subsets_15_l24_24352


namespace toothpicks_at_200th_stage_l24_24081

-- Define initial number of toothpicks at stage 1
def a_1 : ℕ := 4

-- Define the function to compute the number of toothpicks at stage n, taking into account the changing common difference
def a (n : ℕ) : ℕ :=
  if n = 1 then 4
  else if n <= 49 then 4 + 4 * (n - 1)
  else if n <= 99 then 200 + 5 * (n - 50)
  else if n <= 149 then 445 + 6 * (n - 100)
  else if n <= 199 then 739 + 7 * (n - 150)
  else 0  -- This covers cases not considered in the problem for clarity

-- State the theorem to check the number of toothpicks at stage 200
theorem toothpicks_at_200th_stage : a 200 = 1082 :=
  sorry

end toothpicks_at_200th_stage_l24_24081


namespace all_increased_quadratics_have_integer_roots_l24_24769

def original_quadratic (p q : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -p ∧ α * β = q

def increased_quadratic (p q n : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -(p + n) ∧ α * β = (q + n)

theorem all_increased_quadratics_have_integer_roots (p q : ℤ) :
  original_quadratic p q →
  (∀ n, 0 ≤ n ∧ n ≤ 9 → increased_quadratic p q n) :=
sorry

end all_increased_quadratics_have_integer_roots_l24_24769


namespace scooter_gain_percent_l24_24923

theorem scooter_gain_percent 
  (purchase_price : ℕ) 
  (repair_costs : ℕ) 
  (selling_price : ℕ)
  (h1 : purchase_price = 900)
  (h2 : repair_costs = 300)
  (h3 : selling_price = 1320) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 10 :=
by
  sorry

end scooter_gain_percent_l24_24923


namespace quadratic_equation_solution_unique_l24_24845

noncomputable def b_solution := (-3 + 3 * Real.sqrt 21) / 2
noncomputable def c_solution := (33 - 3 * Real.sqrt 21) / 2

theorem quadratic_equation_solution_unique :
  (∃ (b c : ℝ), 
     (∀ (x : ℝ), 3 * x^2 + b * x + c = 0 → x = b_solution) ∧ 
     b + c = 15 ∧ 3 * c = b^2 ∧
     b = b_solution ∧ c = c_solution) :=
by { sorry }

end quadratic_equation_solution_unique_l24_24845


namespace bekahs_reading_l24_24183

def pages_per_day (total_pages read_pages days_left : ℕ) : ℕ :=
  (total_pages - read_pages) / days_left

theorem bekahs_reading :
  pages_per_day 408 113 5 = 59 := by
  sorry

end bekahs_reading_l24_24183


namespace solution_set_for_f_geq_zero_l24_24446

theorem solution_set_for_f_geq_zero (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f3 : f 3 = 0) (h_cond : ∀ x : ℝ, x < 0 → x * (deriv f x) < f x) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | -3 < x ∧ x < 0} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end solution_set_for_f_geq_zero_l24_24446


namespace probability_one_each_item_l24_24714

theorem probability_one_each_item :
  let num_items := 32
  let total_ways := Nat.choose num_items 4
  let favorable_outcomes := 8 * 8 * 8 * 8
  total_ways = 35960 →
  let probability := favorable_outcomes / total_ways
  probability = (128 : ℚ) / 1125 :=
by
  sorry

end probability_one_each_item_l24_24714


namespace isabel_remaining_pages_l24_24450

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def problems_per_page : ℕ := 8

theorem isabel_remaining_pages :
  (total_problems - finished_problems) / problems_per_page = 5 := 
sorry

end isabel_remaining_pages_l24_24450


namespace area_of_triangle_ACD_l24_24814

theorem area_of_triangle_ACD (p : ℝ) (y1 y2 x1 x2 : ℝ)
  (h1 : y1^2 = 2 * p * x1)
  (h2 : y2^2 = 2 * p * x2)
  (h3 : y1 + y2 = 4 * p)
  (h4 : y2 - y1 = p)
  (h5 : 2 * y1 + 2 * y2 = 8 * p^2 / (x2 - x1))
  (h6 : x2 - x1 = 2 * p)
  (h7 : 8 * p^2 = (y1 + y2) * 2 * p) :
  1 / 2 * (y1 * (x1 - (x2 + x1) / 2) + y2 * (x2 - (x2 + x1) / 2)) = 15 / 2 * p^2 :=
by
  sorry

end area_of_triangle_ACD_l24_24814


namespace females_over_30_prefer_webstream_l24_24070

-- Define the total number of survey participants
def total_participants : ℕ := 420

-- Define the number of participants who prefer WebStream
def prefer_webstream : ℕ := 200

-- Define the number of participants who do not prefer WebStream
def not_prefer_webstream : ℕ := 220

-- Define the number of males who prefer WebStream
def males_prefer : ℕ := 80

-- Define the number of females under 30 who do not prefer WebStream
def females_under_30_not_prefer : ℕ := 90

-- Define the number of females over 30 who do not prefer WebStream
def females_over_30_not_prefer : ℕ := 70

-- Define the total number of females under 30 who do not prefer WebStream
def females_not_prefer : ℕ := females_under_30_not_prefer + females_over_30_not_prefer

-- Define the total number of participants who do not prefer WebStream
def total_not_prefer : ℕ := 220

-- Define the number of males who do not prefer WebStream
def males_not_prefer : ℕ := total_not_prefer - females_not_prefer

-- Define the number of females who prefer WebStream
def females_prefer : ℕ := prefer_webstream - males_prefer

-- Define the total number of females under 30 who prefer WebStream
def females_under_30_prefer : ℕ := total_participants - prefer_webstream - females_under_30_not_prefer

-- Define the remaining females over 30 who prefer WebStream
def females_over_30_prefer : ℕ := females_prefer - females_under_30_prefer

-- The Lean statement to prove
theorem females_over_30_prefer_webstream : females_over_30_prefer = 110 := by
  sorry

end females_over_30_prefer_webstream_l24_24070


namespace count_positive_multiples_of_7_ending_in_5_below_1500_l24_24823

theorem count_positive_multiples_of_7_ending_in_5_below_1500 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (k < 1500) → ((k % 7 = 0) ∧ (k % 10 = 5) → (∃ m : ℕ, k = 35 + 70 * m) ∧ (0 ≤ m) ∧ (m < 21))) :=
sorry

end count_positive_multiples_of_7_ending_in_5_below_1500_l24_24823


namespace a1_is_1_l24_24403

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = (2^n - 1)

theorem a1_is_1 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : sequence_sum a S) : 
  a 1 = 1 :=
by 
  sorry

end a1_is_1_l24_24403


namespace Diego_half_block_time_l24_24251

def problem_conditions_and_solution : Prop :=
  ∃ (D : ℕ), (3 * 60 + D * 60) / 2 = 240 ∧ D = 5

theorem Diego_half_block_time :
  problem_conditions_and_solution :=
by
  sorry

end Diego_half_block_time_l24_24251


namespace complete_square_result_l24_24013

theorem complete_square_result (x : ℝ) :
  (∃ r s : ℝ, (16 * x ^ 2 + 32 * x - 1280 = 0) → ((x + r) ^ 2 = s) ∧ s = 81) :=
by
  sorry

end complete_square_result_l24_24013


namespace transform_fraction_l24_24955

theorem transform_fraction (x : ℝ) (h : x ≠ 1) : - (1 / (1 - x)) = 1 / (x - 1) :=
by
  sorry

end transform_fraction_l24_24955


namespace percentage_of_l_equals_150_percent_k_l24_24442

section

variables (j k l m : ℝ) (x : ℝ)

-- Given conditions
axiom cond1 : 1.25 * j = 0.25 * k
axiom cond2 : 1.50 * k = x / 100 * l
axiom cond3 : 1.75 * l = 0.75 * m
axiom cond4 : 0.20 * m = 7.00 * j

-- Proof statement
theorem percentage_of_l_equals_150_percent_k : x = 50 :=
sorry

end

end percentage_of_l_equals_150_percent_k_l24_24442


namespace area_triangle_PTS_l24_24586

theorem area_triangle_PTS {PQ QR PS QT PT TS : ℝ} 
  (hPQ : PQ = 4) 
  (hQR : QR = 6) 
  (hPS : PS = 2 * Real.sqrt 13) 
  (hQT : QT = 12 * Real.sqrt 13 / 13) 
  (hPT : PT = 4) 
  (hTS : TS = (2 * Real.sqrt 13) - 4) : 
  (1 / 2) * PT * QT = 24 * Real.sqrt 13 / 13 := 
by 
  sorry

end area_triangle_PTS_l24_24586


namespace seeds_per_can_l24_24012

theorem seeds_per_can (total_seeds : Float) (cans : Float) (h1 : total_seeds = 54.0) (h2 : cans = 9.0) : total_seeds / cans = 6.0 :=
by
  sorry

end seeds_per_can_l24_24012


namespace find_smallest_k_satisfying_cos_square_l24_24065

theorem find_smallest_k_satisfying_cos_square (k : ℕ) (h : ∃ n : ℕ, k^2 = 180 * n - 64):
  k = 48 ∨ k = 53 :=
by sorry

end find_smallest_k_satisfying_cos_square_l24_24065


namespace john_total_money_l24_24499

-- Variables representing the prices and quantities.
def chip_price : ℝ := 2
def corn_chip_price : ℝ := 1.5
def chips_quantity : ℕ := 15
def corn_chips_quantity : ℕ := 10

-- Hypothesis representing the total money John has.
theorem john_total_money : 
    (chips_quantity * chip_price + corn_chips_quantity * corn_chip_price) = 45 := by
  sorry

end john_total_money_l24_24499


namespace geometric_sequence_problem_l24_24420

-- Step d) Rewrite the problem in Lean 4 statement
theorem geometric_sequence_problem 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (b_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (q : ℝ) 
  (h1 : ∀ n, n > 0 → a_n n = 1 * q^(n-1)) 
  (h2 : 1 + q + q^2 = 7)
  (h3 : 6 * 1 * q = 1 + 3 + 1 * q^2 + 4)
  :
  (∀ n, a_n n = 2^(n-1)) ∧ 
  (∀ n, T_n n = 4 - (n+2) / 2^(n-1)) :=
  sorry

end geometric_sequence_problem_l24_24420


namespace pizzeria_large_pizzas_l24_24299

theorem pizzeria_large_pizzas (price_small : ℕ) (price_large : ℕ) (total_revenue : ℕ) (small_pizzas_sold : ℕ) (L : ℕ) 
    (h1 : price_small = 2) 
    (h2 : price_large = 8) 
    (h3 : total_revenue = 40) 
    (h4 : small_pizzas_sold = 8) 
    (h5 : price_small * small_pizzas_sold + price_large * L = total_revenue) :
    L = 3 := 
by 
  -- Lean will expect a proof here; add sorry for now
  sorry

end pizzeria_large_pizzas_l24_24299


namespace tangent_line_perpendicular_l24_24267

noncomputable def f (x k : ℝ) : ℝ := x^3 - (k^2 - 1) * x^2 - k^2 + 2

theorem tangent_line_perpendicular (k : ℝ) (b : ℝ) (a : ℝ)
  (h1 : ∀ (x : ℝ), f x k = x^3 - (k^2 - 1) * x^2 - k^2 + 2)
  (h2 : (3 - 2 * (k^2 - 1)) = -1) :
  a = -2 := sorry

end tangent_line_perpendicular_l24_24267


namespace total_weight_of_ripe_apples_is_1200_l24_24143

def total_apples : Nat := 14
def weight_ripe_apple : Nat := 150
def weight_unripe_apple : Nat := 120
def unripe_apples : Nat := 6
def ripe_apples : Nat := total_apples - unripe_apples
def total_weight_ripe_apples : Nat := ripe_apples * weight_ripe_apple

theorem total_weight_of_ripe_apples_is_1200 :
  total_weight_ripe_apples = 1200 := by
  sorry

end total_weight_of_ripe_apples_is_1200_l24_24143


namespace largest_lucky_number_l24_24243

theorem largest_lucky_number (n : ℕ) (h₀ : n = 160) (h₁ : ∀ k, 160 > k → k > 0) (h₂ : ∀ k, k ≡ 7 [MOD 16] → k ≤ 160) : 
  ∃ k, k = 151 := 
sorry

end largest_lucky_number_l24_24243


namespace original_price_is_125_l24_24558

noncomputable def original_price (sold_price : ℝ) (discount_percent : ℝ) : ℝ :=
  sold_price / ((100 - discount_percent) / 100)

theorem original_price_is_125 : original_price 120 4 = 125 :=
by
  sorry

end original_price_is_125_l24_24558


namespace value_diff_l24_24501

theorem value_diff (a b : ℕ) (h1 : a * b = 2 * (a + b) + 14) (h2 : b = 8) : b - a = 3 :=
by
  sorry

end value_diff_l24_24501


namespace base_rate_of_first_company_is_7_l24_24398

noncomputable def telephone_company_base_rate_proof : Prop :=
  ∃ (base_rate1 base_rate2 charge_per_minute1 charge_per_minute2 minutes : ℝ),
  base_rate1 = 7 ∧
  charge_per_minute1 = 0.25 ∧
  base_rate2 = 12 ∧
  charge_per_minute2 = 0.20 ∧
  minutes = 100 ∧
  (base_rate1 + charge_per_minute1 * minutes) =
  (base_rate2 + charge_per_minute2 * minutes) ∧
  base_rate1 = 7

theorem base_rate_of_first_company_is_7 :
  telephone_company_base_rate_proof :=
by
  -- The proof step will go here
  sorry

end base_rate_of_first_company_is_7_l24_24398


namespace log5_x_l24_24557

theorem log5_x (x : ℝ) (h : x = (Real.log 2 / Real.log 4) ^ (Real.log 16 / Real.log 2) ^ 2) :
    Real.log x / Real.log 5 = -16 / (Real.log 2 / Real.log 5) := by
  sorry

end log5_x_l24_24557


namespace meat_sales_beyond_plan_l24_24336

-- Define the constants for each day's sales
def sales_thursday := 210
def sales_friday := 2 * sales_thursday
def sales_saturday := 130
def sales_sunday := sales_saturday / 2
def original_plan := 500

-- Define the total sales
def total_sales := sales_thursday + sales_friday + sales_saturday + sales_sunday

-- Prove that they sold 325kg beyond their original plan
theorem meat_sales_beyond_plan : total_sales - original_plan = 325 :=
by
  -- The proof is not included, so we add sorry to skip the proof
  sorry

end meat_sales_beyond_plan_l24_24336


namespace find_three_numbers_l24_24206

theorem find_three_numbers (x y z : ℝ) 
  (h1 : x - y = 12) 
  (h2 : (x + y) / 4 = 7) 
  (h3 : z = 2 * y) 
  (h4 : x + z = 24) : 
  x = 20 ∧ y = 8 ∧ z = 16 := by
  sorry

end find_three_numbers_l24_24206


namespace value_added_to_number_l24_24825

theorem value_added_to_number (x : ℤ) : 
  (150 - 109 = 109 + x) → (x = -68) :=
by
  sorry

end value_added_to_number_l24_24825


namespace min_digits_decimal_correct_l24_24084

noncomputable def min_digits_decimal : ℕ := 
  let n : ℕ := 123456789
  let d : ℕ := 2^26 * 5^4
  26 -- As per the problem statement

theorem min_digits_decimal_correct :
  let n := 123456789
  let d := 2^26 * 5^4
  ∀ x:ℕ, (∃ k:ℕ, n = k * 10^x) → x ≥ min_digits_decimal := 
by
  sorry

end min_digits_decimal_correct_l24_24084


namespace george_hours_tuesday_l24_24650

def wage_per_hour := 5
def hours_monday := 7
def total_earnings := 45

theorem george_hours_tuesday : ∃ (hours_tuesday : ℕ), 
  hours_tuesday = (total_earnings - (hours_monday * wage_per_hour)) / wage_per_hour := 
by
  sorry

end george_hours_tuesday_l24_24650


namespace extremum_of_function_l24_24172

theorem extremum_of_function (k : ℝ) (h₀ : k ≠ 1) :
  (k > 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≤ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) ∧
  (k < 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≥ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) :=
by
  sorry

end extremum_of_function_l24_24172


namespace degree_reduction_l24_24827

theorem degree_reduction (x : ℝ) (h1 : x^2 = x + 1) (h2 : 0 < x) : x^4 - 2 * x^3 + 3 * x = 1 + Real.sqrt 5 :=
by
  sorry

end degree_reduction_l24_24827


namespace find_a_l24_24858

theorem find_a (a : ℝ) : (∃ x y : ℝ, y = 4 - 3 * x ∧ y = 2 * x - 1 ∧ y = a * x + 7) → a = 6 := 
by
  sorry

end find_a_l24_24858


namespace intersection_eq_l24_24803

def setM (x : ℝ) : Prop := x > -1
def setN (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem intersection_eq : {x : ℝ | setM x} ∩ {x | setN x} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end intersection_eq_l24_24803


namespace quadrilateral_inequality_l24_24709

variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2)^2 + (y1 - y2)^2

-- Given a convex quadrilateral ABCD with vertices at (x1, y1), (x2, y2), (x3, y3), and (x4, y4), prove:
theorem quadrilateral_inequality :
  (distance_squared x1 y1 x3 y3 + distance_squared x2 y2 x4 y4) ≤
  (distance_squared x1 y1 x2 y2 + distance_squared x2 y2 x3 y3 +
   distance_squared x3 y3 x4 y4 + distance_squared x4 y4 x1 y1) :=
sorry

end quadrilateral_inequality_l24_24709


namespace part1_part2_l24_24571

noncomputable def Sn (a : ℕ → ℚ) (n : ℕ) (p : ℚ) : ℚ :=
4 * a n - p

theorem part1 (a : ℕ → ℚ) (S : ℕ → ℚ) (p : ℚ) (hp : p ≠ 0)
  (hS : ∀ n, S n = Sn a n p) : 
  ∃ q, ∀ n, a (n + 1) = q * a n :=
sorry

noncomputable def an_formula (n : ℕ) : ℚ := (4/3)^(n - 1)

theorem part2 (b : ℕ → ℚ) (a : ℕ → ℚ)
  (p : ℚ) (hp : p = 3)
  (hb : b 1 = 2)
  (ha1 : a 1 = 1) 
  (h_rec : ∀ n, b (n + 1) = b n + a n) :
  ∀ n, b n = 3 * ((4/3)^(n - 1)) - 1 :=
sorry

end part1_part2_l24_24571


namespace steel_parts_count_l24_24990

-- Definitions for conditions
variables (a b : ℕ)

-- Conditions provided in the problem
axiom machines_count : a + b = 21
axiom chrome_parts : 2 * a + 4 * b = 66

-- Statement to prove
theorem steel_parts_count : 3 * a + 2 * b = 51 :=
by
  sorry

end steel_parts_count_l24_24990


namespace evaluate_expression_l24_24661

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 5 * x + 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = -31 := by
  sorry

end evaluate_expression_l24_24661


namespace statue_original_cost_l24_24780

noncomputable def original_cost (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

theorem statue_original_cost :
  original_cost 660 0.20 = 550 := 
by
  sorry

end statue_original_cost_l24_24780


namespace price_of_coffee_increased_by_300_percent_l24_24795

theorem price_of_coffee_increased_by_300_percent
  (P : ℝ) -- cost per pound of milk powder and coffee in June
  (h1 : 0.20 * P = 0.20) -- price of a pound of milk powder in July
  (h2 : 1.5 * 0.20 = 0.30) -- cost of 1.5 lbs of milk powder in July
  (h3 : 6.30 - 0.30 = 6.00) -- cost of 1.5 lbs of coffee in July
  (h4 : 6.00 / 1.5 = 4.00) -- price per pound of coffee in July
  : ((4.00 - 1.00) / 1.00) * 100 = 300 := 
by 
  sorry

end price_of_coffee_increased_by_300_percent_l24_24795


namespace find_common_ratio_geometric_l24_24895

variable {α : Type*} [Field α] {a : ℕ → α} {S : ℕ → α} {q : α} (h₁ : a 3 = 2 * S 2 + 1) (h₂ : a 4 = 2 * S 3 + 1)

def common_ratio_geometric : α := 3

theorem find_common_ratio_geometric (ha₃ : a 3 = 2 * S 2 + 1) (ha₄ : a 4 = 2 * S 3 + 1) :
  q = common_ratio_geometric := 
  sorry

end find_common_ratio_geometric_l24_24895


namespace fraction_remain_unchanged_l24_24980

theorem fraction_remain_unchanged (m n a b : ℚ) (h : n ≠ 0 ∧ b ≠ 0) : 
  (a / b = (a + m) / (b + n)) ↔ (a / b = m / n) :=
sorry

end fraction_remain_unchanged_l24_24980


namespace exactly_one_germinates_l24_24010

theorem exactly_one_germinates (pA pB : ℝ) (hA : pA = 0.8) (hB : pB = 0.9) : 
  (pA * (1 - pB) + (1 - pA) * pB) = 0.26 :=
by
  sorry

end exactly_one_germinates_l24_24010


namespace smallest_prime_x_l24_24683

-- Define prime number checker
def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem conditions and proof goal
theorem smallest_prime_x 
  (x y z : ℕ) 
  (hx : is_prime x)
  (hy : is_prime y)
  (hz : is_prime z)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hyz : y ≠ z)
  (hd : ∀ d : ℕ, d ∣ (x * x * y * z) ↔ (d = 1 ∨ d = x ∨ d = x * x ∨ d = y ∨ d = x * y ∨ d = x * x * y ∨ d = z ∨ d = x * z ∨ d = x * x * z ∨ d = y * z ∨ d = x * y * z ∨ d = x * x * y * z)) 
  : x = 2 := 
sorry

end smallest_prime_x_l24_24683


namespace no_prime_number_between_30_and_40_mod_9_eq_7_l24_24273

theorem no_prime_number_between_30_and_40_mod_9_eq_7 : ¬ ∃ n : ℕ, 30 ≤ n ∧ n ≤ 40 ∧ Nat.Prime n ∧ n % 9 = 7 :=
by
  sorry

end no_prime_number_between_30_and_40_mod_9_eq_7_l24_24273


namespace digit_B_divisibility_l24_24710

theorem digit_B_divisibility (B : ℕ) (h : 4 * 1000 + B * 100 + B * 10 + 6 % 11 = 0) : B = 5 :=
sorry

end digit_B_divisibility_l24_24710


namespace polynomial_remainder_l24_24567

theorem polynomial_remainder :
  ∀ (x : ℂ), (x^1010 % (x^4 - 1)) = x^2 :=
sorry

end polynomial_remainder_l24_24567


namespace parameter_values_for_three_distinct_roots_l24_24111

theorem parameter_values_for_three_distinct_roots (a : ℝ) :
  (∀ x : ℝ, (|x^3 - a^3| = x - a) → (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ↔ 
  (-2 / Real.sqrt 3 < a ∧ a < -1 / Real.sqrt 3) :=
sorry

end parameter_values_for_three_distinct_roots_l24_24111


namespace radius_of_circle_l24_24832

open Complex

theorem radius_of_circle (z : ℂ) (h : (z + 2)^4 = 16 * z^4) : abs z = 2 / Real.sqrt 3 :=
sorry

end radius_of_circle_l24_24832


namespace sequence_property_l24_24946

def Sn (n : ℕ) (a : ℕ → ℕ) : ℕ := (Finset.range (n + 1)).sum a

theorem sequence_property (a : ℕ → ℕ) (h : ∀ n : ℕ, Sn (n + 1) a = 2 * a n + 1) : a 3 = 2 :=
sorry

end sequence_property_l24_24946


namespace sector_area_l24_24580

theorem sector_area (r : ℝ) (alpha : ℝ) (h : r = 2) (h2 : alpha = π / 3) : 
  1/2 * alpha * r^2 = (2 * π) / 3 := by
  sorry

end sector_area_l24_24580


namespace find_number_l24_24855

theorem find_number (x : ℝ) 
  (h1 : 0.15 * 40 = 6) 
  (h2 : 6 = 0.25 * x + 2) : 
  x = 16 := 
sorry

end find_number_l24_24855


namespace ivanov_family_net_worth_l24_24958

-- Define the financial values
def value_of_apartment := 3000000
def market_value_of_car := 900000
def bank_savings := 300000
def value_of_securities := 200000
def liquid_cash := 100000
def remaining_mortgage := 1500000
def car_loan := 500000
def debt_to_relatives := 200000

-- Calculate total assets and total liabilities
def total_assets := value_of_apartment + market_value_of_car + bank_savings + value_of_securities + liquid_cash
def total_liabilities := remaining_mortgage + car_loan + debt_to_relatives

-- Define the hypothesis and the final result of the net worth calculation
theorem ivanov_family_net_worth : total_assets - total_liabilities = 2300000 := by
  sorry

end ivanov_family_net_worth_l24_24958


namespace range_of_m_l24_24592

noncomputable def f (x m : ℝ) := Real.exp x + x^2 / m^2 - x

theorem range_of_m (m : ℝ) (hm : m ≠ 0) :
  (∀ a b : ℝ, a ∈ Set.Icc (-1) 1 -> b ∈ Set.Icc (-1) 1 -> |f a m - f b m| ≤ Real.exp 1) ↔
  (m ∈ Set.Iic (-Real.sqrt 2 / 2) ∪ Set.Ici (Real.sqrt 2 / 2)) :=
by
  sorry

end range_of_m_l24_24592


namespace horatio_sonnets_count_l24_24947

-- Each sonnet consists of 14 lines
def lines_per_sonnet : ℕ := 14

-- The number of sonnets his lady fair heard
def heard_sonnets : ℕ := 7

-- The total number of unheard lines
def unheard_lines : ℕ := 70

-- Calculate sonnets Horatio wrote by the heard and unheard components
def total_sonnets : ℕ := heard_sonnets + (unheard_lines / lines_per_sonnet)

-- Prove the total number of sonnets horatio wrote
theorem horatio_sonnets_count : total_sonnets = 12 := 
by sorry

end horatio_sonnets_count_l24_24947


namespace xyz_value_l24_24213

noncomputable def positive (x : ℝ) : Prop := 0 < x

theorem xyz_value (x y z : ℝ) (hx : positive x) (hy : positive y) (hz : positive z): 
  (x + 1/y = 5) → (y + 1/z = 2) → (z + 1/x = 8/3) → x * y * z = (17 + Real.sqrt 285) / 2 :=
by
  sorry

end xyz_value_l24_24213


namespace distinct_natural_numbers_l24_24763

theorem distinct_natural_numbers (n : ℕ) (h : n = 100) : 
  ∃ (nums : Fin n → ℕ), 
    (∀ i j, i ≠ j → nums i ≠ nums j) ∧
    (∀ (a b c d e : Fin n), 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
     c ≠ d ∧ c ≠ e ∧ 
     d ≠ e →
      (nums a) * (nums b) * (nums c) * (nums d) * (nums e) % ((nums a) + (nums b) + (nums c) + (nums d) + (nums e)) = 0) :=
by
  sorry

end distinct_natural_numbers_l24_24763


namespace doses_A_correct_doses_B_correct_doses_C_correct_l24_24238

def days_in_july : ℕ := 31

def daily_dose_A : ℕ := 1
def daily_dose_B : ℕ := 2
def daily_dose_C : ℕ := 3

def missed_days_A : ℕ := 3
def missed_days_B_morning : ℕ := 5
def missed_days_C_all : ℕ := 2

def total_doses_A : ℕ := days_in_july * daily_dose_A
def total_doses_B : ℕ := days_in_july * daily_dose_B
def total_doses_C : ℕ := days_in_july * daily_dose_C

def missed_doses_A : ℕ := missed_days_A * daily_dose_A
def missed_doses_B : ℕ := missed_days_B_morning
def missed_doses_C : ℕ := missed_days_C_all * daily_dose_C

def doses_consumed_A := total_doses_A - missed_doses_A
def doses_consumed_B := total_doses_B - missed_doses_B
def doses_consumed_C := total_doses_C - missed_doses_C

theorem doses_A_correct : doses_consumed_A = 28 := by sorry
theorem doses_B_correct : doses_consumed_B = 57 := by sorry
theorem doses_C_correct : doses_consumed_C = 87 := by sorry

end doses_A_correct_doses_B_correct_doses_C_correct_l24_24238


namespace cube_side_length_l24_24689

theorem cube_side_length (n : ℕ) (h : n^3 - (n-2)^3 = 98) : n = 5 :=
by sorry

end cube_side_length_l24_24689


namespace calc1_calc2_calc3_l24_24628

theorem calc1 : -4 - 4 = -8 := by
  sorry

theorem calc2 : (-32) / 4 = -8 := by
  sorry

theorem calc3 : -(-2)^3 = 8 := by
  sorry

end calc1_calc2_calc3_l24_24628


namespace halfway_between_one_fourth_and_one_seventh_l24_24288

theorem halfway_between_one_fourth_and_one_seventh : (1 / 4 + 1 / 7) / 2 = 11 / 56 := by
  sorry

end halfway_between_one_fourth_and_one_seventh_l24_24288


namespace paperback_copies_sold_l24_24961

theorem paperback_copies_sold 
(H : ℕ)
(hardback_sold : H = 36000)
(P : ℕ)
(paperback_relation : P = 9 * H)
(total_copies : H + P = 440000) :
P = 324000 :=
sorry

end paperback_copies_sold_l24_24961


namespace insulin_pills_per_day_l24_24199

def conditions (I B A : ℕ) : Prop := 
  B = 3 ∧ A = 2 * B ∧ 7 * (I + B + A) = 77

theorem insulin_pills_per_day : ∃ (I : ℕ), ∀ (B A : ℕ), conditions I B A → I = 2 := by
  sorry

end insulin_pills_per_day_l24_24199


namespace max_rectangle_area_l24_24237

-- Definitions based on conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_rectangle_area
  (l w : ℕ)
  (h_perim : perimeter l w = 50)
  (h_prime : is_prime l)
  (h_composite : is_composite w) :
  l * w = 156 :=
sorry

end max_rectangle_area_l24_24237


namespace repeating_decimal_conversion_l24_24108

-- Definition of 0.\overline{23} as a rational number
def repeating_decimal_fraction : ℚ := 23 / 99

-- The main statement to prove
theorem repeating_decimal_conversion : (3 / 10) + (repeating_decimal_fraction) = 527 / 990 := 
by
  -- Placeholder for proof steps
  sorry

end repeating_decimal_conversion_l24_24108


namespace problem_statement_l24_24701

theorem problem_statement (x y z t : ℝ) (h : (x + y) / (y + z) = (z + t) / (t + x)) : x * (z + t + y) = z * (x + y + t) :=
sorry

end problem_statement_l24_24701


namespace process_cannot_continue_indefinitely_l24_24819

theorem process_cannot_continue_indefinitely (n : ℕ) (hn : 2018 ∣ n) :
  ¬(∀ m, ∃ k, (10*m + k) % 11 = 0 ∧ (10*m + k) / 11 ∣ n) :=
sorry

end process_cannot_continue_indefinitely_l24_24819


namespace arithmetic_identity_l24_24364

theorem arithmetic_identity : 15 * 30 + 45 * 15 - 15 * 10 = 975 :=
by
  sorry

end arithmetic_identity_l24_24364


namespace find_a_l24_24008

theorem find_a (m c a b : ℝ) (h_m : m < 0) (h_radius : (m^2 + 3) = 4) 
  (h_c : c = 1 ∨ c = -3) (h_focus : c > 0) (h_ellipse : b^2 = 3) 
  (h_focus_eq : c^2 = a^2 - b^2) : a = 2 :=
by
  sorry

end find_a_l24_24008


namespace reflection_xy_plane_reflection_across_point_l24_24773

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def reflect_across_xy_plane (p : Point3D) : Point3D :=
  {x := p.x, y := p.y, z := -p.z}

def reflect_across_point (a p : Point3D) : Point3D :=
  {x := 2 * a.x - p.x, y := 2 * a.y - p.y, z := 2 * a.z - p.z}

theorem reflection_xy_plane :
  reflect_across_xy_plane {x := -2, y := 1, z := 4} = {x := -2, y := 1, z := -4} :=
by sorry

theorem reflection_across_point :
  reflect_across_point {x := 1, y := 0, z := 2} {x := -2, y := 1, z := 4} = {x := -5, y := -1, z := 0} :=
by sorry

end reflection_xy_plane_reflection_across_point_l24_24773


namespace m_above_x_axis_m_on_line_l24_24603

namespace ComplexNumberProblem

def above_x_axis (m : ℝ) : Prop :=
  m^2 - 2 * m - 15 > 0

def on_line (m : ℝ) : Prop :=
  2 * m^2 + 3 * m - 4 = 0

theorem m_above_x_axis (m : ℝ) : above_x_axis m → (m < -3 ∨ m > 5) :=
  sorry

theorem m_on_line (m : ℝ) : on_line m → 
  (m = (-3 + Real.sqrt 41) / 4) ∨ (m = (-3 - Real.sqrt 41) / 4) :=
  sorry

end ComplexNumberProblem

end m_above_x_axis_m_on_line_l24_24603


namespace solution_in_quadrant_IV_l24_24698

theorem solution_in_quadrant_IV (k : ℝ) :
  (∃ x y : ℝ, x + 2 * y = 4 ∧ k * x - y = 1 ∧ x > 0 ∧ y < 0) ↔ (-1 / 2 < k ∧ k < 2) :=
by
  sorry

end solution_in_quadrant_IV_l24_24698


namespace Dan_has_five_limes_l24_24253

-- Define the initial condition of limes Dan had
def initial_limes : Nat := 9

-- Define the limes Dan gave to Sara
def limes_given : Nat := 4

-- Define the remaining limes Dan has
def remaining_limes : Nat := initial_limes - limes_given

-- The theorem we need to prove, i.e., the remaining limes Dan has is 5
theorem Dan_has_five_limes : remaining_limes = 5 := by
  sorry

end Dan_has_five_limes_l24_24253


namespace ferry_speed_difference_l24_24966

open Nat

-- Define the time and speed of ferry P
def timeP := 3 -- hours
def speedP := 8 -- kilometers per hour

-- Define the distance of ferry P
def distanceP := speedP * timeP -- kilometers

-- Define the distance of ferry Q
def distanceQ := 3 * distanceP -- kilometers

-- Define the time of ferry Q
def timeQ := timeP + 5 -- hours

-- Define the speed of ferry Q
def speedQ := distanceQ / timeQ -- kilometers per hour

-- Define the speed difference
def speedDifference := speedQ - speedP -- kilometers per hour

-- The target theorem to prove
theorem ferry_speed_difference : speedDifference = 1 := by
  sorry

end ferry_speed_difference_l24_24966


namespace solution_set_l24_24831

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := x^2 + 3 * x - 4

-- Define the inequality
def inequality (x : ℝ) : Prop := quadratic_expr x > 0

-- State the theorem
theorem solution_set : ∀ x : ℝ, inequality x ↔ (x > 1 ∨ x < -4) :=
by
  sorry

end solution_set_l24_24831


namespace min_f_on_interval_l24_24323

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x + 1) * (cos x + 1) / (sin x * cos x)

theorem min_f_on_interval : 
  ∀ x, 0 < x ∧ x < π / 2 → f x ≥ 3 + 2 * sqrt 2 :=
sorry

end min_f_on_interval_l24_24323


namespace prove_remainder_l24_24666

def problem_statement : Prop := (33333332 % 8 = 4)

theorem prove_remainder : problem_statement := 
by
  sorry

end prove_remainder_l24_24666


namespace age_difference_l24_24621

variables (P M Mo : ℕ)

def patrick_michael_ratio (P M : ℕ) : Prop := (P * 5 = M * 3)
def michael_monica_ratio (M Mo : ℕ) : Prop := (M * 4 = Mo * 3)
def sum_of_ages (P M Mo : ℕ) : Prop := (P + M + Mo = 88)

theorem age_difference (P M Mo : ℕ) : 
  patrick_michael_ratio P M → 
  michael_monica_ratio M Mo → 
  sum_of_ages P M Mo → 
  (Mo - P = 22) :=
by
  sorry

end age_difference_l24_24621


namespace problem1_problem2_l24_24181

theorem problem1 : (Real.sqrt 2) * (Real.sqrt 6) + (Real.sqrt 3) = 3 * (Real.sqrt 3) :=
  sorry

theorem problem2 : (1 - Real.sqrt 2) * (2 - Real.sqrt 2) = 4 - 3 * (Real.sqrt 2) :=
  sorry

end problem1_problem2_l24_24181


namespace meadow_area_l24_24385

theorem meadow_area (x : ℝ) (h1 : ∀ y : ℝ, y = x / 2 + 3) (h2 : ∀ z : ℝ, z = 1 / 3 * (x / 2 - 3) + 6) :
  (x / 2 + 3) + (1 / 3 * (x / 2 - 3) + 6) = x → x = 24 := by
  sorry

end meadow_area_l24_24385


namespace plane_divided_into_four_regions_l24_24697

-- Definition of the conditions
def line1 (x y : ℝ) : Prop := y = 3 * x
def line2 (x y : ℝ) : Prop := y = (1 / 3) * x

-- Proof statement
theorem plane_divided_into_four_regions :
  (∃ f g : ℝ → ℝ, ∀ x, line1 x (f x) ∧ line2 x (g x)) →
  ∃ n : ℕ, n = 4 :=
by sorry

end plane_divided_into_four_regions_l24_24697


namespace min_AP_BP_l24_24059

-- Definitions based on conditions in the problem
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 6)
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- The theorem to prove the minimum value of AP + BP
theorem min_AP_BP
  (P : ℝ × ℝ)
  (hP_parabola : parabola P.1 P.2) :
  dist P A + dist P B ≥ 9 :=
sorry

end min_AP_BP_l24_24059


namespace range_of_a_l24_24888

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by {
  sorry
}

end range_of_a_l24_24888


namespace length_of_hypotenuse_l24_24952

/-- Define the problem's parameters -/
def perimeter : ℝ := 34
def area : ℝ := 24
def length_hypotenuse (a b c : ℝ) : Prop := a + b + c = perimeter 
  ∧ (1/2) * a * b = area
  ∧ a^2 + b^2 = c^2

/- Lean statement for the proof problem -/
theorem length_of_hypotenuse (a b c : ℝ) 
  (h1: a + b + c = 34)
  (h2: (1/2) * a * b = 24)
  (h3: a^2 + b^2 = c^2)
  : c = 62 / 4 := sorry

end length_of_hypotenuse_l24_24952


namespace train_speed_l24_24681

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) (h_train_length : train_length = 100) (h_bridge_length : bridge_length = 300) (h_crossing_time : crossing_time = 12) : 
  (train_length + bridge_length) / crossing_time = 33.33 := 
by 
  -- sorry allows us to skip the proof
  sorry

end train_speed_l24_24681


namespace abs_diff_probs_l24_24761

def numRedMarbles := 1000
def numBlackMarbles := 1002
def totalMarbles := numRedMarbles + numBlackMarbles

def probSame : ℚ := 
  ((numRedMarbles * (numRedMarbles - 1)) / 2 + (numBlackMarbles * (numBlackMarbles - 1)) / 2) / (totalMarbles * (totalMarbles - 1) / 2)

def probDiff : ℚ :=
  (numRedMarbles * numBlackMarbles) / (totalMarbles * (totalMarbles - 1) / 2)

theorem abs_diff_probs : |probSame - probDiff| = 999 / 2003001 := 
by {
  sorry
}

end abs_diff_probs_l24_24761


namespace least_element_of_S_is_4_l24_24280

theorem least_element_of_S_is_4 :
  ∃ S : Finset ℕ, S.card = 7 ∧ (S ⊆ Finset.range 16) ∧
  (∀ {a b : ℕ}, a ∈ S → b ∈ S → a < b → ¬ (b % a = 0)) ∧
  (∀ T : Finset ℕ, T.card = 7 → (T ⊆ Finset.range 16) →
  (∀ {a b : ℕ}, a ∈ T → b ∈ T → a < b → ¬ (b % a = 0)) →
  ∃ x : ℕ, x ∈ T ∧ x = 4) :=
by
  sorry

end least_element_of_S_is_4_l24_24280


namespace proportion_solution_l24_24686

theorem proportion_solution (x : ℝ) (h : 0.6 / x = 5 / 8) : x = 0.96 :=
by 
  -- The proof will go here
  sorry

end proportion_solution_l24_24686


namespace principal_amount_borrowed_l24_24692

theorem principal_amount_borrowed (P R T SI : ℕ) (h₀ : SI = (P * R * T) / 100) (h₁ : SI = 5400) (h₂ : R = 12) (h₃ : T = 3) : P = 15000 :=
by
  sorry

end principal_amount_borrowed_l24_24692


namespace jake_time_to_row_lake_l24_24041

noncomputable def time_to_row_lake (side_length miles_per_side : ℝ) (swim_time_per_mile minutes_per_mile : ℝ) : ℝ :=
  let swim_speed := 60 / swim_time_per_mile -- miles per hour
  let row_speed := 2 * swim_speed          -- miles per hour
  let total_distance := 4 * side_length    -- miles
  total_distance / row_speed               -- hours

theorem jake_time_to_row_lake :
  time_to_row_lake 15 20 = 10 := sorry

end jake_time_to_row_lake_l24_24041


namespace sum_of_c_and_d_l24_24899

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x - 3) / (x^2 + c * x + d)

theorem sum_of_c_and_d (c d : ℝ) (h_asymptote1 : (2:ℝ)^2 + c * 2 + d = 0) (h_asymptote2 : (-1:ℝ)^2 - c + d = 0) :
  c + d = -3 :=
by
-- theorem body (proof omitted)
sorry

end sum_of_c_and_d_l24_24899


namespace tank_full_capacity_l24_24670

variable (T : ℝ) -- Define T as a real number representing the total capacity of the tank.

-- The main condition: (3 / 4) * T + 5 = (7 / 8) * T
axiom condition : (3 / 4) * T + 5 = (7 / 8) * T

-- Proof statement: Prove that T = 40
theorem tank_full_capacity : T = 40 :=
by
  -- Using the given condition to derive that T = 40.
  sorry

end tank_full_capacity_l24_24670


namespace opposite_negative_nine_l24_24718

theorem opposite_negative_nine : 
  (∃ (y : ℤ), -9 + y = 0 ∧ y = 9) :=
by sorry

end opposite_negative_nine_l24_24718


namespace student_rank_left_l24_24548

theorem student_rank_left {n m : ℕ} (h1 : n = 10) (h2 : m = 6) : (n - m + 1) = 5 := by
  sorry

end student_rank_left_l24_24548


namespace acute_triangle_l24_24843

theorem acute_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
                       (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a)
                       (h7 : a^3 + b^3 = c^3) :
                       c^2 < a^2 + b^2 :=
by {
  sorry
}

end acute_triangle_l24_24843


namespace win_probability_l24_24361

theorem win_probability (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose = 3 / 8) :=
by
  -- Provide the proof here if needed, but skip it
  sorry

end win_probability_l24_24361


namespace line_repr_exists_same_line_iff_scalar_multiple_l24_24109

-- Given that D is a line in 3D space, there exist a, b, c not all zero
theorem line_repr_exists
  (D : Set (ℝ × ℝ × ℝ)) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ 
  (D = {p | ∃ (u v w : ℝ), p = (u, v, w) ∧ a * u + b * v + c * w = 0}) :=
sorry

-- Given two lines represented by different coefficients being the same
-- Prove that the coefficients are scalar multiples of each other
theorem same_line_iff_scalar_multiple
  (α1 β1 γ1 α2 β2 γ2 : ℝ) :
  (∀ (u v w : ℝ), α1 * u + β1 * v + γ1 * w = 0 ↔ α2 * u + β2 * v + γ2 * w = 0) ↔
  (∃ k : ℝ, k ≠ 0 ∧ α2 = k * α1 ∧ β2 = k * β1 ∧ γ2 = k * γ1) :=
sorry

end line_repr_exists_same_line_iff_scalar_multiple_l24_24109


namespace quadratic_eq_mn_sum_l24_24436

theorem quadratic_eq_mn_sum (m n : ℤ) 
  (h1 : m - 1 = 2) 
  (h2 : 16 + 4 * n = 0) 
  : m + n = -1 :=
by
  sorry

end quadratic_eq_mn_sum_l24_24436


namespace set_1234_excellent_no_proper_subset_excellent_l24_24295

open Set

namespace StepLength

def excellent_set (D : Set ℤ) : Prop :=
∀ A : Set ℤ, ∃ a d : ℤ, d ∈ D → ({a - d, a, a + d} ⊆ A ∨ {a - d, a, a + d} ⊆ (univ \ A))

noncomputable def S : Set (Set ℤ) := {{1}, {2}, {3}, {4}}

theorem set_1234_excellent : excellent_set {1, 2, 3, 4} := sorry

theorem no_proper_subset_excellent :
  ¬ (excellent_set {1, 3, 4} ∨ excellent_set {1, 2, 3} ∨ excellent_set {1, 2, 4} ∨ excellent_set {2, 3, 4}) := sorry

end StepLength

end set_1234_excellent_no_proper_subset_excellent_l24_24295


namespace heating_time_correct_l24_24424

structure HeatingProblem where
  initial_temp : ℕ
  final_temp : ℕ
  heating_rate : ℕ

def time_to_heat (hp : HeatingProblem) : ℕ :=
  (hp.final_temp - hp.initial_temp) / hp.heating_rate

theorem heating_time_correct (hp : HeatingProblem) (h1 : hp.initial_temp = 20) (h2 : hp.final_temp = 100) (h3 : hp.heating_rate = 5) :
  time_to_heat hp = 16 :=
by
  sorry

end heating_time_correct_l24_24424


namespace solve_ordered_pair_l24_24758

theorem solve_ordered_pair (x y : ℝ) (h1 : x + y = (5 - x) + (5 - y)) (h2 : x - y = (x - 1) + (y - 1)) : (x, y) = (4, 1) :=
by
  sorry

end solve_ordered_pair_l24_24758


namespace largest_possible_d_l24_24027

theorem largest_possible_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : 
    d ≤ (5 + Real.sqrt 105) / 2 :=
by
  sorry

end largest_possible_d_l24_24027


namespace factor_correct_l24_24164

def factor_expression (x : ℝ) : Prop :=
  x * (x - 3) - 5 * (x - 3) = (x - 5) * (x - 3)

theorem factor_correct (x : ℝ) : factor_expression x :=
  by sorry

end factor_correct_l24_24164


namespace square_area_l24_24660

theorem square_area (x : ℚ) (side_length : ℚ) 
  (h1 : side_length = 3 * x - 12) 
  (h2 : side_length = 24 - 2 * x) : 
  side_length ^ 2 = 92.16 := 
by 
  sorry

end square_area_l24_24660


namespace find_x_perpendicular_l24_24034

-- Definitions used in the conditions
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Condition: vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- The theorem we want to prove
theorem find_x_perpendicular : ∀ x : ℝ, perpendicular a (b x) → x = -8 / 3 :=
by
  intros x h
  sorry

end find_x_perpendicular_l24_24034


namespace find_m_l24_24967

theorem find_m 
  (x1 x2 : ℝ) 
  (m : ℝ)
  (h1 : x1 + x2 = m)
  (h2 : x1 * x2 = 2 * m - 1)
  (h3 : x1^2 + x2^2 = 7) : 
  m = 5 :=
by
  sorry

end find_m_l24_24967


namespace lemango_eating_mangos_l24_24152

theorem lemango_eating_mangos :
  ∃ (mangos_eaten : ℕ → ℕ), 
    (mangos_eaten 1 * (2^6 - 1) = 364 * (2 - 1)) ∧
    (mangos_eaten 6 = 128) :=
by
  sorry

end lemango_eating_mangos_l24_24152


namespace right_triangle_count_l24_24871

theorem right_triangle_count (a b : ℕ) (h1 : b < 100) (h2 : a^2 + b^2 = (b + 2)^2) : 
∃ n, n = 10 :=
by sorry

end right_triangle_count_l24_24871


namespace sin_neg_30_eq_neg_one_half_l24_24454

theorem sin_neg_30_eq_neg_one_half :
  Real.sin (-30 * Real.pi / 180) = -1 / 2 := 
by
  sorry -- Proof is skipped

end sin_neg_30_eq_neg_one_half_l24_24454


namespace confidence_of_independence_test_l24_24625

-- Define the observed value of K^2
def K2_obs : ℝ := 5

-- Define the critical value(s) of K^2 for different confidence levels
def K2_critical_0_05 : ℝ := 3.841
def K2_critical_0_01 : ℝ := 6.635

-- Define the confidence levels corresponding to the critical values
def P_K2_ge_3_841 : ℝ := 0.05
def P_K2_ge_6_635 : ℝ := 0.01

-- Define the statement to be proved: there is 95% confidence that "X and Y are related".
theorem confidence_of_independence_test
  (K2_obs K2_critical_0_05 P_K2_ge_3_841 : ℝ)
  (hK2_obs_gt_critical : K2_obs > K2_critical_0_05)
  (hP : P_K2_ge_3_841 = 0.05) :
  1 - P_K2_ge_3_841 = 0.95 :=
by
  -- The proof is omitted
  sorry

end confidence_of_independence_test_l24_24625


namespace find_a_l24_24978

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.sin x

theorem find_a (a : ℝ) : (∀ f', f' = (fun x => a * Real.exp x - Real.cos x) → f' 0 = 0) → a = 1 :=
by
  intros h
  specialize h (fun x => a * Real.exp x - Real.cos x) rfl
  sorry  -- proof is omitted

end find_a_l24_24978


namespace total_water_hold_l24_24940

variables
  (first : ℕ := 100)
  (second : ℕ := 150)
  (third : ℕ := 75)
  (total : ℕ := 325)

theorem total_water_hold :
  first + second + third = total := by
  sorry

end total_water_hold_l24_24940


namespace sum_congruence_example_l24_24157

theorem sum_congruence_example (a b c : ℤ) (h1 : a % 15 = 7) (h2 : b % 15 = 3) (h3 : c % 15 = 9) : 
  (a + b + c) % 15 = 4 :=
by 
  sorry

end sum_congruence_example_l24_24157


namespace number_of_students_l24_24719

variables (m d r : ℕ) (k : ℕ)

theorem number_of_students :
  (30 < m + d ∧ m + d < 40) → (r = 3 * m) → (r = 5 * d) → m + d = 32 :=
by 
  -- The proof body is not necessary here according to instructions.
  sorry

end number_of_students_l24_24719


namespace min_value_l24_24186

theorem min_value (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/29 :=
sorry

end min_value_l24_24186


namespace function_identity_l24_24668

variables {R : Type*} [LinearOrderedField R]

-- Define real-valued functions f, g, h
variables (f g h : R → R)

-- Define function composition and multiplication
def comp (f g : R → R) (x : R) := f (g x)
def mul (f g : R → R) (x : R) := f x * g x

-- The statement to prove
theorem function_identity (x : R) : 
  comp (mul f g) h x = mul (comp f h) (comp g h) x :=
sorry

end function_identity_l24_24668


namespace value_of_expression_l24_24886

variables (m n c d : ℝ)
variables (h1 : m = -n) (h2 : c * d = 1)

theorem value_of_expression : m + n + 3 * c * d - 10 = -7 :=
by sorry

end value_of_expression_l24_24886


namespace total_students_in_class_l24_24332

theorem total_students_in_class
    (students_in_front : ℕ)
    (students_in_back : ℕ)
    (lines : ℕ)
    (total_students_line : ℕ)
    (total_class : ℕ)
    (h_front: students_in_front = 2)
    (h_back: students_in_back = 5)
    (h_lines: lines = 3)
    (h_students_line : total_students_line = students_in_front + 1 + students_in_back)
    (h_total_class : total_class = lines * total_students_line) :
  total_class = 24 := by
  sorry

end total_students_in_class_l24_24332


namespace find_expression_for_f_l24_24646

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem find_expression_for_f (x : ℝ) (h : x ≠ -1) 
    (hf : f ((1 - x) / (1 + x)) = x) : 
    f x = (1 - x) / (1 + x) :=
sorry

end find_expression_for_f_l24_24646


namespace initial_buckets_correct_l24_24314

-- Define the conditions as variables
def total_buckets : ℝ := 9.8
def added_buckets : ℝ := 8.8
def initial_buckets : ℝ := total_buckets - added_buckets

-- The theorem to prove the initial amount of water is 1 bucket
theorem initial_buckets_correct : initial_buckets = 1 := 
by
  sorry

end initial_buckets_correct_l24_24314


namespace conic_sections_with_foci_at_F2_zero_l24_24959

theorem conic_sections_with_foci_at_F2_zero (a b m n: ℝ) (h1 : a > b) (h2: b > 0) (h3: m > 0) (h4: n > 0) (h5: a^2 - b^2 = 4) (h6: m^2 + n^2 = 4):
  (∀ x y: ℝ, x^2 / (a^2) + y^2 / (b^2) = 1) ∧ (∀ x y: ℝ, x^2 / (11/60) + y^2 / (11/16) = 1) ∧ 
  ∀ x y: ℝ, x^2 / (m^2) - y^2 / (n^2) = 1 ∧ ∀ x y: ℝ, 5*x^2 / 4 - 5*y^2 / 16 = 1 := 
sorry

end conic_sections_with_foci_at_F2_zero_l24_24959


namespace percentage_of_first_to_second_l24_24239

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) (h1 : first = (7 / 100) * X) (h2 : second = (14 / 100) * X) : 
(first / second) * 100 = 50 := by
  sorry

end percentage_of_first_to_second_l24_24239


namespace fractional_eq_solution_l24_24549

theorem fractional_eq_solution : ∀ x : ℝ, (x ≠ 3) → ((2 - x) / (x - 3) + 1 / (3 - x) = 1) → (x = 2) :=
by
  intros x h_cond h_eq
  sorry

end fractional_eq_solution_l24_24549


namespace oranges_for_juice_l24_24469

theorem oranges_for_juice (total_oranges : ℝ) (exported_percentage : ℝ) (juice_percentage : ℝ) :
  total_oranges = 7 →
  exported_percentage = 0.30 →
  juice_percentage = 0.60 →
  (total_oranges * (1 - exported_percentage) * juice_percentage) = 2.9 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end oranges_for_juice_l24_24469


namespace johns_previous_earnings_l24_24101

theorem johns_previous_earnings (new_earnings raise_percentage old_earnings : ℝ) 
  (h1 : new_earnings = 68) (h2 : raise_percentage = 0.1333333333333334)
  (h3 : new_earnings = old_earnings * (1 + raise_percentage)) : old_earnings = 60 :=
sorry

end johns_previous_earnings_l24_24101


namespace original_number_l24_24595

theorem original_number (x : ℝ) (h : 1.4 * x = 700) : x = 500 :=
sorry

end original_number_l24_24595


namespace total_distance_fourth_time_l24_24764

/-- 
A super ball is dropped from a height of 100 feet and rebounds half the distance it falls each time.
We need to prove that the total distance the ball travels when it hits the ground
the fourth time is 275 feet.
-/
noncomputable def total_distance : ℝ :=
  let first_descent := 100
  let second_descent := first_descent / 2
  let third_descent := second_descent / 2
  let fourth_descent := third_descent / 2
  let first_ascent := second_descent
  let second_ascent := third_descent
  let third_ascent := fourth_descent
  first_descent + second_descent + third_descent + fourth_descent +
  first_ascent + second_ascent + third_ascent

theorem total_distance_fourth_time : total_distance = 275 := 
  by
  sorry

end total_distance_fourth_time_l24_24764


namespace slower_speed_for_on_time_arrival_l24_24087

variable (distance : ℝ) (actual_speed : ℝ) (time_early : ℝ)

theorem slower_speed_for_on_time_arrival 
(h1 : distance = 20)
(h2 : actual_speed = 40)
(h3 : time_early = 1 / 15) :
  actual_speed - (600 / 17) = 4.71 :=
by 
  sorry

end slower_speed_for_on_time_arrival_l24_24087


namespace quadratic_has_real_roots_l24_24859

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, x^2 + x - 4 * m = 0) ↔ m ≥ -1 / 16 :=
by
  sorry

end quadratic_has_real_roots_l24_24859


namespace original_price_before_discounts_l24_24319

theorem original_price_before_discounts (P : ℝ) (h : 0.684 * P = 6840) : P = 10000 :=
by
  sorry

end original_price_before_discounts_l24_24319


namespace intersection_range_l24_24712

noncomputable def function1 (x : ℝ) : ℝ := abs (x^2 - 1) / (x - 1)
noncomputable def function2 (k x : ℝ) : ℝ := k * x - 2

theorem intersection_range (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ function1 x₁ = function2 k x₁ ∧ function1 x₂ = function2 k x₂) ↔ 
  (0 < k ∧ k < 1) ∨ (1 < k ∧ k < 4) := 
sorry

end intersection_range_l24_24712


namespace captain_co_captain_selection_l24_24236

theorem captain_co_captain_selection 
  (men women : ℕ)
  (h_men : men = 12) 
  (h_women : women = 12) : 
  (men * (men - 1) + women * (women - 1)) = 264 := 
by
  -- Since we are skipping the proof here, we use sorry.
  sorry

end captain_co_captain_selection_l24_24236


namespace sum_of_integers_l24_24464

theorem sum_of_integers (a b : ℤ) (h : (Int.sqrt (a - 2023) + |b + 2023| = 1)) : a + b = 1 ∨ a + b = -1 :=
by
  sorry

end sum_of_integers_l24_24464


namespace kristy_baked_cookies_l24_24777

theorem kristy_baked_cookies (C : ℕ) :
  (C - 3) - 8 - 12 - 16 - 6 - 14 = 10 ↔ C = 69 := by
  sorry

end kristy_baked_cookies_l24_24777


namespace value_of_f_neg_a_l24_24641

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + x^3 + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 3) : f (-a) = -1 := 
by
  sorry

end value_of_f_neg_a_l24_24641


namespace find_c_of_triangle_area_l24_24076

-- Define the problem in Lean 4 statement.
theorem find_c_of_triangle_area (A : ℝ) (b c : ℝ) (area : ℝ)
  (hA : A = 60 * Real.pi / 180)  -- Converting degrees to radians
  (hb : b = 1)
  (hArea : area = Real.sqrt 3) :
  c = 4 :=
by 
  -- Lean proof goes here (we include sorry to skip)
  sorry

end find_c_of_triangle_area_l24_24076


namespace problem_C_l24_24857

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def is_obtuse_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0

theorem problem_C (A B C : ℝ × ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0 :=
by
  sorry

end problem_C_l24_24857


namespace exists_nat_pair_l24_24533

theorem exists_nat_pair 
  (k : ℕ) : 
  let a := 2 * k
  let b := 2 * k * k + 2 * k + 1
  (b - 1) % (a + 1) = 0 ∧ (a * a + a + 2) % b = 0 := by
  sorry

end exists_nat_pair_l24_24533


namespace overtime_hours_l24_24435

theorem overtime_hours
  (regularPayPerHour : ℝ)
  (regularHours : ℝ)
  (totalPay : ℝ)
  (overtimeRate : ℝ) 
  (h1 : regularPayPerHour = 3)
  (h2 : regularHours = 40)
  (h3 : totalPay = 168)
  (h4 : overtimeRate = 2 * regularPayPerHour) :
  (totalPay - (regularPayPerHour * regularHours)) / overtimeRate = 8 :=
by
  sorry

end overtime_hours_l24_24435


namespace old_camera_model_cost_l24_24528

theorem old_camera_model_cost (C new_model_cost discounted_lens_cost : ℝ)
  (h1 : new_model_cost = 1.30 * C)
  (h2 : discounted_lens_cost = 200)
  (h3 : new_model_cost + discounted_lens_cost = 5400)
  : C = 4000 := by
sorry

end old_camera_model_cost_l24_24528


namespace truck_capacity_l24_24750

theorem truck_capacity (x y : ℝ)
  (h1 : 3 * x + 4 * y = 22)
  (h2 : 5 * x + 2 * y = 25) :
  4 * x + 3 * y = 23.5 :=
sorry

end truck_capacity_l24_24750


namespace exists_same_color_ratios_l24_24207

-- Definition of coloring function.
def coloring : ℕ → Fin 2 := sorry

-- Definition of the problem: there exist A, B, C such that A : C = C : B,
-- and A, B, C are of same color.
theorem exists_same_color_ratios :
  ∃ A B C : ℕ, coloring A = coloring B ∧ coloring B = coloring C ∧ 
  (A : ℚ) / C = (C : ℚ) / B := 
sorry

end exists_same_color_ratios_l24_24207


namespace calc_x_equals_condition_l24_24724

theorem calc_x_equals_condition (m n p q x : ℝ) :
  x^2 + (2 * m * p + 2 * n * q) ^ 2 + (2 * m * q - 2 * n * p) ^ 2 = (m ^ 2 + n ^ 2 + p ^ 2 + q ^ 2) ^ 2 →
  x = m ^ 2 + n ^ 2 - p ^ 2 - q ^ 2 ∨ x = - m ^ 2 - n ^ 2 + p ^ 2 + q ^ 2 :=
by
  sorry

end calc_x_equals_condition_l24_24724


namespace supplement_of_angle_l24_24587

-- Condition: The complement of angle α is 54 degrees 32 minutes
theorem supplement_of_angle (α : ℝ) (h : α = 90 - (54 + 32 / 60)) :
  180 - α = 144 + 32 / 60 := by
sorry

end supplement_of_angle_l24_24587


namespace area_of_ABCD_l24_24504

noncomputable def AB := 6
noncomputable def BC := 8
noncomputable def CD := 15
noncomputable def DA := 17
def right_angle_BCD := true
def convex_ABCD := true

theorem area_of_ABCD : ∃ area : ℝ, area = 110 := by
  -- Given conditions
  have hAB : AB = 6 := rfl
  have hBC : BC = 8 := rfl
  have hCD : CD = 15 := rfl
  have hDA : DA = 17 := rfl
  have hAngle : right_angle_BCD = true := rfl
  have hConvex : convex_ABCD = true := rfl

  -- skip the proof
  sorry

end area_of_ABCD_l24_24504


namespace k_times_a_plus_b_l24_24362

/-- Given a quadrilateral with vertices P(ka, kb), Q(kb, ka), R(-ka, -kb), and S(-kb, -ka),
where a and b are consecutive integers with a > b > 0, and k is an odd integer.
It is given that the area of PQRS is 50.
Prove that k(a + b) = 5. -/
theorem k_times_a_plus_b (a b k : ℤ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a = b + 1)
  (h4 : Odd k)
  (h5 : 2 * k^2 * (a - b) * (a + b) = 50) :
  k * (a + b) = 5 := by
  sorry

end k_times_a_plus_b_l24_24362


namespace friend1_reading_time_friend2_reading_time_l24_24402

theorem friend1_reading_time (my_reading_time : ℕ) (h1 : my_reading_time = 180) (h2 : ∀ t : ℕ, t = my_reading_time / 2) : 
  ∃ t1 : ℕ, t1 = 90 := by
  sorry

theorem friend2_reading_time (my_reading_time : ℕ) (h1 : my_reading_time = 180) (h2 : ∀ t : ℕ, t = my_reading_time * 2) : 
  ∃ t2 : ℕ, t2 = 360 := by
  sorry

end friend1_reading_time_friend2_reading_time_l24_24402


namespace Jennifer_has_24_dollars_left_l24_24202

def remaining_money (initial amount: ℕ) (spent_sandwich spent_museum_ticket spent_book: ℕ) : ℕ :=
  initial - (spent_sandwich + spent_museum_ticket + spent_book)

theorem Jennifer_has_24_dollars_left :
  remaining_money 180 (1/5*180) (1/6*180) (1/2*180) = 24 :=
by
  sorry

end Jennifer_has_24_dollars_left_l24_24202


namespace library_fiction_percentage_l24_24311

theorem library_fiction_percentage:
  let original_volumes := 18360
  let fiction_percentage := 0.30
  let fraction_transferred := 1/3
  let fraction_fiction_transferred := 1/5
  let initial_fiction := fiction_percentage * original_volumes
  let transferred_volumes := fraction_transferred * original_volumes
  let transferred_fiction := fraction_fiction_transferred * transferred_volumes
  let remaining_fiction := initial_fiction - transferred_fiction
  let remaining_volumes := original_volumes - transferred_volumes
  let remaining_fiction_percentage := (remaining_fiction / remaining_volumes) * 100
  remaining_fiction_percentage = 35 := 
by
  sorry

end library_fiction_percentage_l24_24311


namespace mashed_potatoes_count_l24_24062

theorem mashed_potatoes_count :
  ∀ (b s : ℕ), b = 489 → b = s + 10 → s = 479 :=
by
  intros b s h₁ h₂
  sorry

end mashed_potatoes_count_l24_24062


namespace geometric_sequence_sum_8_l24_24146

variable {a : ℝ} 

-- conditions
def geometric_series_sum_4 (r : ℝ) (a : ℝ) : ℝ :=
  a + a * r + a * r^2 + a * r^3

def geometric_series_sum_8 (r : ℝ) (a : ℝ) : ℝ :=
  a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 + a * r^6 + a * r^7

theorem geometric_sequence_sum_8 (r : ℝ) (S4 : ℝ) (S8 : ℝ) (hr : r = 2) (hS4 : S4 = 1) :
  (∃ a : ℝ, geometric_series_sum_4 r a = S4 ∧ geometric_series_sum_8 r a = S8) → S8 = 17 :=
by
  sorry

end geometric_sequence_sum_8_l24_24146


namespace call_cost_inequalities_min_call_cost_correct_l24_24245

noncomputable def call_cost_before (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2 else 0.4

noncomputable def call_cost_after (x : ℝ) : ℝ :=
  if x ≤ 3 then 0.2
  else if x ≤ 4 then 0.2 + 0.1 * (x - 3)
  else 0.3 + 0.1 * (x - 4)

theorem call_cost_inequalities : 
  (call_cost_before 4 = 0.4 ∧ call_cost_after 4 = 0.3) ∧
  (call_cost_before 4.3 = 0.4 ∧ call_cost_after 4.3 = 0.4) ∧
  (call_cost_before 5.8 = 0.4 ∧ call_cost_after 5.8 = 0.5) ∧
  (∀ x, (0 < x ∧ x ≤ 3) ∨ x > 4 → call_cost_before x ≤ call_cost_after x) :=
by
  sorry

noncomputable def min_call_cost_plan (m : ℝ) (n : ℕ) : ℝ :=
  if 3 * n - 1 < m ∧ m ≤ 3 * n then 0.2 * n
  else if 3 * n < m ∧ m ≤ 3 * n + 1 then 0.2 * n + 0.1
  else if 3 * n + 1 < m ∧ m ≤ 3 * n + 2 then 0.2 * n + 0.2
  else 0.0  -- Fallback, though not necessary as per the conditions

theorem min_call_cost_correct (m : ℝ) (n : ℕ) (h : m > 5) :
  (3 * n - 1 < m ∧ m ≤ 3 * n → min_call_cost_plan m n = 0.2 * n) ∧
  (3 * n < m ∧ m ≤ 3 * n + 1 → min_call_cost_plan m n = 0.2 * n + 0.1) ∧
  (3 * n + 1 < m ∧ m ≤ 3 * n + 2 → min_call_cost_plan m n = 0.2 * n + 0.2) :=
by
  sorry

end call_cost_inequalities_min_call_cost_correct_l24_24245


namespace point_2000_coordinates_l24_24116

-- Definition to describe the spiral numbering system in the first quadrant
def spiral_number (n : ℕ) : ℕ × ℕ := sorry

-- The task is to prove that the coordinates of the 2000th point are (44, 25).
theorem point_2000_coordinates : spiral_number 2000 = (44, 25) :=
by
  sorry

end point_2000_coordinates_l24_24116


namespace Jake_watched_hours_on_Friday_l24_24707

theorem Jake_watched_hours_on_Friday :
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  total_show_hours - total_hours_before_Friday = 19 :=
by
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  sorry

end Jake_watched_hours_on_Friday_l24_24707


namespace exists_special_N_l24_24356

open Nat

theorem exists_special_N :
  ∃ N : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 150 → N % i = 0 ∨ i = 127 ∨ i = 128) ∧ 
  ¬ (N % 127 = 0) ∧ ¬ (N % 128 = 0) :=
by
  sorry

end exists_special_N_l24_24356


namespace thirteenth_term_is_correct_l24_24102

noncomputable def third_term : ℚ := 2 / 11
noncomputable def twenty_third_term : ℚ := 3 / 7

theorem thirteenth_term_is_correct : 
  (third_term + twenty_third_term) / 2 = 47 / 154 := sorry

end thirteenth_term_is_correct_l24_24102


namespace triangle_acute_l24_24951

theorem triangle_acute (A B C : ℝ) (h1 : A = 2 * (180 / 9)) (h2 : B = 3 * (180 / 9)) (h3 : C = 4 * (180 / 9)) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  sorry

end triangle_acute_l24_24951


namespace reducible_fraction_l24_24877

theorem reducible_fraction (l : ℤ) : ∃ k : ℤ, l = 13 * k + 4 ↔ (∃ d > 1, d ∣ (5 * l + 6) ∧ d ∣ (8 * l + 7)) :=
sorry

end reducible_fraction_l24_24877


namespace square_line_product_l24_24856

theorem square_line_product (b : ℝ) 
  (h1 : ∃ y1 y2, y1 = -1 ∧ y2 = 4) 
  (h2 : ∃ x1, x1 = 3) 
  (h3 : (4 - (-1)) = (5 : ℝ)) 
  (h4 : ((∃ b1, b1 = 3 + 5 ∨ b1 = 3 - 5) → b = b1)) :
  b = -2 ∨ b = 8 → b * 8 = -16 :=
by sorry

end square_line_product_l24_24856


namespace expected_value_is_20_point_5_l24_24377

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

def coin_heads_probability : ℚ := 1 / 2

noncomputable def expected_value : ℚ :=
  coin_heads_probability * (penny_value + nickel_value + dime_value + quarter_value)

theorem expected_value_is_20_point_5 :
  expected_value = 20.5 := by
  sorry

end expected_value_is_20_point_5_l24_24377


namespace remove_one_to_get_average_of_75_l24_24775

theorem remove_one_to_get_average_of_75 : 
  ∃ l : List ℕ, l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] ∧ 
  (∃ m : ℕ, List.erase l m = ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] : List ℕ) ∧ 
  (12 : ℕ) = List.length (List.erase l m) ∧
  7.5 = ((List.sum (List.erase l m) : ℚ) / 12)) :=
sorry

end remove_one_to_get_average_of_75_l24_24775


namespace angle_conversion_l24_24046

/--
 Given an angle in degrees, express it in degrees, minutes, and seconds.
 Theorem: 20.23 degrees can be converted to 20 degrees, 13 minutes, and 48 seconds.
-/
theorem angle_conversion : (20.23:ℝ) = 20 + (13/60 : ℝ) + (48/3600 : ℝ) :=
by
  sorry

end angle_conversion_l24_24046


namespace cos_pi_plus_2alpha_l24_24627

theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin ((Real.pi / 2) + α) = 1 / 3) : Real.cos (Real.pi + 2 * α) = 7 / 9 :=
by
  sorry

end cos_pi_plus_2alpha_l24_24627


namespace find_a_l24_24004

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

theorem find_a (a : ℝ) (h_intersect : ∃ x₀, f a x₀ = g x₀) (h_tangent : ∃ x₀, (f a x₀) = g x₀ ∧ (1/x₀ * a = 1/ (2 * Real.sqrt x₀))):
  a = Real.exp 1 / 2 :=
by
  sorry

end find_a_l24_24004


namespace diameter_of_outer_circle_l24_24901

theorem diameter_of_outer_circle (D d : ℝ) 
  (h1 : d = 24) 
  (h2 : π * (D / 2) ^ 2 - π * (d / 2) ^ 2 = 0.36 * π * (D / 2) ^ 2) : D = 30 := 
by 
  sorry

end diameter_of_outer_circle_l24_24901


namespace problem1_problem2_l24_24125

-- Problem 1: Calculation Proof
theorem problem1 : (3 - Real.pi)^0 - Real.sqrt 4 + 4 * Real.sin (Real.pi * 60 / 180) + |Real.sqrt 3 - 3| = 2 + Real.sqrt 3 :=
by
  sorry

-- Problem 2: Inequality Systems Proof
theorem problem2 (x : ℝ) :
  (5 * (x + 3) > 4 * x + 8) ∧ (x / 6 - 1 < (x - 2) / 3) → x > -2 :=
by
  sorry

end problem1_problem2_l24_24125


namespace two_digit_number_condition_l24_24663

theorem two_digit_number_condition :
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 2 = 0 ∧ (n + 1) % 3 = 0 ∧ (n + 2) % 4 = 0 ∧ (n + 3) % 5 = 0 ∧ n = 62 :=
by
  sorry

end two_digit_number_condition_l24_24663


namespace range_of_m_l24_24411

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h1 : 1/x + 1/y = 1) (h2 : x + y > m) : m < 4 := 
sorry

end range_of_m_l24_24411


namespace system1_solution_system2_solution_l24_24748

-- Problem 1
theorem system1_solution (x z : ℤ) (h1 : 3 * x - 5 * z = 6) (h2 : x + 4 * z = -15) : x = -3 ∧ z = -3 :=
by
  sorry

-- Problem 2
theorem system2_solution (x y : ℚ) 
 (h1 : ((2 * x - 1) / 5) + ((3 * y - 2) / 4) = 2) 
 (h2 : ((3 * x + 1) / 5) - ((3 * y + 2) / 4) = 0) : x = 3 ∧ y = 2 :=
by
  sorry

end system1_solution_system2_solution_l24_24748


namespace simplify_expression_l24_24948

theorem simplify_expression (w : ℕ) : 
  4 * w + 6 * w + 8 * w + 10 * w + 12 * w + 14 * w + 16 = 54 * w + 16 :=
by 
  sorry

end simplify_expression_l24_24948


namespace min_spiders_sufficient_spiders_l24_24574

def grid_size : ℕ := 2019

noncomputable def min_k_catch (k : ℕ) : Prop :=
∀ (fly spider1 spider2 : ℕ × ℕ) (fly_move spider1_move spider2_move: ℕ × ℕ → ℕ × ℕ), 
  (fly_move fly = fly ∨ fly_move fly = (fly.1 + 1, fly.2) ∨ fly_move fly = (fly.1 - 1, fly.2)
  ∨ fly_move fly = (fly.1, fly.2 + 1) ∨ fly_move fly = (fly.1, fly.2 - 1))
  ∧ (spider1_move spider1 = spider1 ∨ spider1_move spider1 = (spider1.1 + 1, spider1.2) ∨ spider1_move spider1 = (spider1.1 - 1, spider1.2)
  ∨ spider1_move spider1 = (spider1.1, spider1.2 + 1) ∨ spider1_move spider1 = (spider1.1, spider1.2 - 1))
  ∧ (spider2_move spider2 = spider2 ∨ spider2_move spider2 = (spider2.1 + 1, spider2.2) ∨ spider2_move spider2 = (spider2.1 - 1, spider2.2)
  ∨ spider2_move spider2 = (spider2.1, spider2.2 + 1) ∨ spider2_move spider2 = (spider2.1, spider2.2 - 1))
  → (spider1 = fly ∨ spider2 = fly)

theorem min_spiders (k : ℕ) : min_k_catch k → k ≥ 2 :=
sorry

theorem sufficient_spiders : min_k_catch 2 :=
sorry

end min_spiders_sufficient_spiders_l24_24574


namespace find_m_and_n_l24_24624

theorem find_m_and_n (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
                    (h2 : a + b + c + d = m^2) 
                    (h3 : max a (max b (max c d)) = n^2) : 
                    m = 9 ∧ n = 6 := 
sorry

end find_m_and_n_l24_24624


namespace prove_ratio_l24_24810

variable (a b c d : ℚ)

-- Conditions
def cond1 : a / b = 5 := sorry
def cond2 : b / c = 1 / 4 := sorry
def cond3 : c / d = 7 := sorry

-- Theorem to prove the final result
theorem prove_ratio (h1 : a / b = 5) (h2 : b / c = 1 / 4) (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end prove_ratio_l24_24810


namespace red_balls_in_box_l24_24935

theorem red_balls_in_box (initial_red_balls added_red_balls : ℕ) (initial_blue_balls : ℕ) 
  (h_initial : initial_red_balls = 5) (h_added : added_red_balls = 2) : 
  initial_red_balls + added_red_balls = 7 :=
by {
  sorry
}

end red_balls_in_box_l24_24935


namespace number_of_apple_trees_l24_24113

variable (T : ℕ) -- Declare the number of apple trees as a natural number

-- Define the conditions
def picked_apples := 8 * T
def remaining_apples := 9
def initial_apples := 33

-- The statement to prove Rachel has 3 apple trees
theorem number_of_apple_trees :
  initial_apples - picked_apples + remaining_apples = initial_apples → T = 3 := 
by
  sorry

end number_of_apple_trees_l24_24113


namespace count_even_factors_is_correct_l24_24808

def prime_factors_444_533_72 := (2^8 * 5^3 * 7^2)

def range_a := {a : ℕ | 0 ≤ a ∧ a ≤ 8}
def range_b := {b : ℕ | 0 ≤ b ∧ b ≤ 3}
def range_c := {c : ℕ | 0 ≤ c ∧ c ≤ 2}

def even_factors_count : ℕ :=
  (8 - 1 + 1) * (3 - 0 + 1) * (2 - 0 + 1)

theorem count_even_factors_is_correct :
  even_factors_count = 96 := by
  sorry

end count_even_factors_is_correct_l24_24808


namespace significant_digits_of_side_length_l24_24835

noncomputable def num_significant_digits (n : Float) : Nat :=
  -- This is a placeholder function to determine the number of significant digits
  sorry

theorem significant_digits_of_side_length :
  ∀ (A : Float), A = 3.2400 → num_significant_digits (Float.sqrt A) = 5 :=
by
  intro A h
  -- Proof would go here
  sorry

end significant_digits_of_side_length_l24_24835


namespace factor_expression_l24_24119

theorem factor_expression (y : ℝ) : 
  5 * y * (y - 2) + 11 * (y - 2) = (y - 2) * (5 * y + 11) :=
by
  sorry

end factor_expression_l24_24119


namespace find_f_2012_l24_24776

noncomputable def f : ℤ → ℤ := sorry

axiom even_function : ∀ x : ℤ, f (-x) = f x
axiom f_1 : f 1 = 1
axiom f_2011_ne_1 : f 2011 ≠ 1
axiom max_property : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)

theorem find_f_2012 : f 2012 = 1 := sorry

end find_f_2012_l24_24776


namespace solve_system_of_equations_l24_24974

theorem solve_system_of_equations (x y : ℝ) :
  (1 / 2 * x - 3 / 2 * y = -1) ∧ (2 * x + y = 3) → 
  (x = 1) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l24_24974


namespace eval_f_pi_over_8_l24_24313

noncomputable def f (θ : ℝ) : ℝ :=
(2 * (Real.sin (θ / 2)) ^ 2 - 1) / (Real.sin (θ / 2) * Real.cos (θ / 2)) + 2 * Real.tan θ

theorem eval_f_pi_over_8 : f (π / 8) = -4 :=
sorry

end eval_f_pi_over_8_l24_24313


namespace factorize_expression_l24_24786

theorem factorize_expression (x : ℝ) :
  9 * x^2 - 6 * x + 1 = (3 * x - 1)^2 := 
by sorry

end factorize_expression_l24_24786


namespace max_value_of_f_l24_24343

-- Define the function f(x) = 5x - x^2
def f (x : ℝ) : ℝ := 5 * x - x^2

-- The theorem we want to prove, stating the maximum value of f(x) is 6.25
theorem max_value_of_f : ∃ x, f x = 6.25 :=
by
  -- Placeholder proof, to be completed
  sorry

end max_value_of_f_l24_24343


namespace cube_volume_surface_area_l24_24262

-- Define volume and surface area conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 3 * x
def surface_area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x

-- The main theorem statement
theorem cube_volume_surface_area (x : ℝ) (s : ℝ) :
  volume_condition x s → surface_area_condition x s → x = 5832 :=
by
  intros h_volume h_area
  sorry

end cube_volume_surface_area_l24_24262


namespace geometric_seq_a5_value_l24_24166

theorem geometric_seq_a5_value 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : ∀ n : ℕ, a (n+1) = a n * q)
  (h_pos : ∀ n : ℕ, a n > 0) 
  (h1 : a 1 * a 8 = 4 * a 5)
  (h2 : (a 4 + 2 * a 6) / 2 = 18) 
  : a 5 = 16 := 
sorry

end geometric_seq_a5_value_l24_24166


namespace michael_initial_money_l24_24917

theorem michael_initial_money 
  (M B_initial B_left B_spent : ℕ) 
  (h_split : M / 2 = B_initial - B_left + B_spent): 
  (M / 2 + B_left = 17 + 35) → M = 152 :=
by
  sorry

end michael_initial_money_l24_24917


namespace optionA_optionC_l24_24520

noncomputable def f (x : ℝ) : ℝ := Real.log (|x - 2| + 1)

theorem optionA : ∀ x : ℝ, f (x + 2) = f (-x + 2) := 
by sorry

theorem optionC : (∀ x : ℝ, x < 2 → f x > f (x + 0.01)) ∧ (∀ x : ℝ, x > 2 → f x < f (x - 0.01)) := 
by sorry

end optionA_optionC_l24_24520


namespace probability_event_occurs_l24_24120

def in_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 * Real.pi

def event_occurs (x : ℝ) : Prop :=
  Real.cos (x + Real.pi / 3) + Real.sqrt 3 * Real.sin (x + Real.pi / 3) ≥ 1

theorem probability_event_occurs : 
  (∀ x, in_interval x → event_occurs x) → 
  (∃ p, p = 1/3) :=
by
  intros h
  sorry

end probability_event_occurs_l24_24120


namespace remainder_of_x_div_9_l24_24359

theorem remainder_of_x_div_9 (x : ℕ) (hx_pos : 0 < x) (h : (6 * x) % 9 = 3) : x % 9 = 5 :=
by {
  sorry
}

end remainder_of_x_div_9_l24_24359


namespace find_c_plus_d_l24_24277

-- Conditions as definitions
variables {P A C : Point }
variables {O₁ O₂ : Point}
variables {AB AP CP : ℝ}
variables {c d : ℕ}

-- Given conditions
def Point_on_diagonal (P A C : Point) : Prop := true -- We need to code the detailed properties of being on the diagonal
def circumcenter_of_triangle (P Q R O : Point) : Prop := true -- We need to code the properties of being a circumcenter
def AP_greater_than_CP (AP CP : ℝ) : Prop := AP > CP
def angle_right (A B O : Point) : Prop := true -- Define the right angle property

-- Main statement to prove
theorem find_c_plus_d : 
  Point_on_diagonal P A C ∧
  circumcenter_of_triangle A B P O₁ ∧ 
  circumcenter_of_triangle C D P O₂ ∧ 
  AP_greater_than_CP AP CP ∧
  AB = 10 ∧
  angle_right O₁ P O₂ ∧
  (AP = Real.sqrt c + Real.sqrt d) →
  (c + d = 100) :=
by
  sorry

end find_c_plus_d_l24_24277


namespace correct_assignment_statements_l24_24015

-- Defining what constitutes an assignment statement in this context.
def is_assignment_statement (s : String) : Prop :=
  s ∈ ["x ← 1", "y ← 2", "z ← 3", "i ← i + 2"]

-- Given statements
def statements : List String :=
  ["x ← 1, y ← 2, z ← 3", "S^2 ← 4", "i ← i + 2", "x + 1 ← x"]

-- The Lean Theorem statement that these are correct assignment statements.
theorem correct_assignment_statements (s₁ s₃ : String) (h₁ : s₁ = "x ← 1, y ← 2, z ← 3") (h₃ : s₃ = "i ← i + 2") :
  is_assignment_statement s₁ ∧ is_assignment_statement s₃ :=
by
  sorry

end correct_assignment_statements_l24_24015


namespace supplementary_angle_proof_l24_24148

noncomputable def complementary_angle (α : ℝ) : ℝ := 125 + 12 / 60

noncomputable def calculate_angle (c : ℝ) := 180 - c

noncomputable def supplementary_angle (α : ℝ) := 90 - α

theorem supplementary_angle_proof :
    let α := calculate_angle (complementary_angle α)
    supplementary_angle α = 35 + 12 / 60 := 
by
  sorry

end supplementary_angle_proof_l24_24148


namespace spilled_wax_amount_l24_24893

-- Definitions based on conditions
def car_wax := 3
def suv_wax := 4
def total_wax := 11
def remaining_wax := 2

-- The theorem to be proved
theorem spilled_wax_amount : car_wax + suv_wax + (total_wax - remaining_wax - (car_wax + suv_wax)) = total_wax - remaining_wax :=
by
  sorry


end spilled_wax_amount_l24_24893


namespace dawson_failed_by_36_l24_24117

-- Define the constants and conditions
def max_marks : ℕ := 220
def passing_percentage : ℝ := 0.3
def marks_obtained : ℕ := 30

-- Calculate the minimum passing marks
noncomputable def min_passing_marks : ℝ :=
  passing_percentage * max_marks

-- Calculate the marks Dawson failed by
noncomputable def marks_failed_by : ℝ :=
  min_passing_marks - marks_obtained

-- State the theorem
theorem dawson_failed_by_36 :
  marks_failed_by = 36 := by
  -- Proof is omitted
  sorry

end dawson_failed_by_36_l24_24117


namespace smallest_k_DIVISIBLE_by_3_67_l24_24691

theorem smallest_k_DIVISIBLE_by_3_67 :
  ∃ k : ℕ, (∀ n : ℕ, (2016^k % 3^67 = 0 ∧ (2016^n % 3^67 = 0 → k ≤ n)) ∧ k = 34) := by
  sorry

end smallest_k_DIVISIBLE_by_3_67_l24_24691


namespace garden_fencing_l24_24969

theorem garden_fencing (length width : ℕ) (h1 : length = 80) (h2 : width = length / 2) : 2 * (length + width) = 240 :=
by
  sorry

end garden_fencing_l24_24969


namespace min_ratio_of_cylinder_cone_l24_24971

open Real

noncomputable def V1 (r : ℝ) : ℝ := 2 * π * r^3
noncomputable def V2 (R m r : ℝ) : ℝ := (1 / 3) * π * R^2 * m
noncomputable def geometric_constraint (R m r : ℝ) : Prop :=
  R / m = r / (sqrt ((m - r)^2 - r^2))

theorem min_ratio_of_cylinder_cone (r : ℝ) (hr : r > 0) : 
  ∃ R m, geometric_constraint R m r ∧ (V2 R m r) / (V1 r) = 4 / 3 := 
sorry

end min_ratio_of_cylinder_cone_l24_24971


namespace simplify_sum_l24_24633

theorem simplify_sum :
  -2^2004 + (-2)^2005 + 2^2006 - 2^2007 = -2^2004 - 2^2005 + 2^2006 - 2^2007 :=
by
  sorry

end simplify_sum_l24_24633


namespace f_iterate_result_l24_24878

def f (n : ℕ) : ℕ :=
if n < 3 then n^2 + 1 else 4*n - 3

theorem f_iterate_result : f (f (f 1)) = 17 :=
by
  sorry

end f_iterate_result_l24_24878


namespace plane_passes_through_line_l24_24373

-- Definition for a plane α and a line l
variable {α : Set Point} -- α represents the set of points in plane α
variable {l : Set Point} -- l represents the set of points in line l

-- The condition given
def passes_through (α : Set Point) (l : Set Point) : Prop :=
  l ⊆ α

-- The theorem statement
theorem plane_passes_through_line (α : Set Point) (l : Set Point) :
  passes_through α l = (l ⊆ α) :=
by
  sorry

end plane_passes_through_line_l24_24373


namespace english_only_students_l24_24597

theorem english_only_students (T B G_total : ℕ) (hT : T = 40) (hB : B = 12) (hG_total : G_total = 22) :
  (T - (G_total - B) - B) = 18 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end english_only_students_l24_24597


namespace domain_and_range_of_f_l24_24667

noncomputable def f (a x : ℝ) : ℝ := Real.log (a - a * x) / Real.log a

theorem domain_and_range_of_f (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, a - a * x > 0 → x < 1) ∧ 
  (∀ t : ℝ, 0 < t ∧ t < a → ∃ x : ℝ, t = a - a * x) :=
by
  sorry

end domain_and_range_of_f_l24_24667


namespace difference_digits_in_base2_l24_24086

def binaryDigitCount (n : Nat) : Nat := Nat.log2 n + 1

theorem difference_digits_in_base2 : binaryDigitCount 1400 - binaryDigitCount 300 = 2 :=
by
  sorry

end difference_digits_in_base2_l24_24086


namespace family_members_count_l24_24950

-- Defining the conditions given in the problem
variables (cyrus_bites_arms_legs : ℕ) (cyrus_bites_body : ℕ) (total_bites_family : ℕ)
variables (family_bites_per_person : ℕ) (cyrus_total_bites : ℕ)

-- Given conditions
def condition1 : cyrus_bites_arms_legs = 14 := sorry
def condition2 : cyrus_bites_body = 10 := sorry
def condition3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body := sorry
def condition4 : total_bites_family = cyrus_total_bites / 2 := sorry
def condition5 : ∀ n : ℕ, total_bites_family = n * family_bites_per_person := sorry

-- The theorem to prove: The number of people in the rest of Cyrus' family is 12
theorem family_members_count (n : ℕ) (h1 : cyrus_bites_arms_legs = 14)
    (h2 : cyrus_bites_body = 10) (h3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body)
    (h4 : total_bites_family = cyrus_total_bites / 2)
    (h5 : ∀ n, total_bites_family = n * family_bites_per_person) : n = 12 :=
sorry

end family_members_count_l24_24950


namespace part1_part2_l24_24491

open Real

variable {x y a: ℝ}

-- Condition for the second proof to avoid division by zero
variable (h1 : a ≠ 1) (h2 : a ≠ 4) (h3 : a ≠ -4)

theorem part1 : (x + y)^2 + y * (3 * x - y) = x^2 + 5 * (x * y) := 
by sorry

theorem part2 (h1: a ≠ 1) (h2: a ≠ 4) (h3: a ≠ -4) : 
  ((4 - a^2) / (a - 1) + a) / ((a^2 - 16) / (a - 1)) = -1 / (a + 4) := 
by sorry

end part1_part2_l24_24491


namespace total_trees_planted_total_trees_when_a_100_l24_24716

-- Define the number of trees planted by each team based on 'a'
def trees_first_team (a : ℕ) : ℕ := a
def trees_second_team (a : ℕ) : ℕ := 2 * a + 8
def trees_third_team (a : ℕ) : ℕ := (2 * a + 8) / 2 - 6

-- Define the total number of trees
def total_trees (a : ℕ) : ℕ := 
  trees_first_team a + trees_second_team a + trees_third_team a

-- The main theorem
theorem total_trees_planted (a : ℕ) : total_trees a = 4 * a + 6 :=
by
  sorry

-- The specific calculation when a = 100
theorem total_trees_when_a_100 : total_trees 100 = 406 :=
by
  sorry

end total_trees_planted_total_trees_when_a_100_l24_24716


namespace father_children_age_l24_24824

theorem father_children_age (F C n : Nat) (h1 : F = C) (h2 : F = 75) (h3 : C + 5 * n = 2 * (F + n)) : 
  n = 25 :=
by
  sorry

end father_children_age_l24_24824


namespace triangle_area_ratio_l24_24112

/-
In triangle XYZ, XY=12, YZ=16, and XZ=20. Point D is on XY,
E is on YZ, and F is on XZ. Let XD=p*XY, YE=q*YZ, and ZF=r*XZ,
where p, q, r are positive and satisfy p+q+r=0.9 and p^2+q^2+r^2=0.29.
Prove that the ratio of the area of triangle DEF to the area of triangle XYZ 
can be written in the form m/n where m, n are relatively prime positive 
integers and m+n=137.
-/

theorem triangle_area_ratio :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m + n = 137 ∧ 
  ∃ (p q r : ℝ), p + q + r = 0.9 ∧ p^2 + q^2 + r^2 = 0.29 ∧ 
                  ∀ (XY YZ XZ : ℝ), XY = 12 ∧ YZ = 16 ∧ XZ = 20 → 
                  (1 - (p * (1 - r) + q * (1 - p) + r * (1 - q))) = (37 / 100) :=
by
   sorry

end triangle_area_ratio_l24_24112


namespace count_valid_abcd_is_zero_l24_24126

def valid_digits := {a // 1 ≤ a ∧ a ≤ 9} 
def zero_to_nine := {n // 0 ≤ n ∧ n ≤ 9}

noncomputable def increasing_arithmetic_sequence_with_difference_5 (a b c d : ℕ) : Prop := 
  10 * a + b + 5 = 10 * b + c ∧ 
  10 * b + c + 5 = 10 * c + d

theorem count_valid_abcd_is_zero :
  ∀ (a : valid_digits) (b c d : zero_to_nine),
    ¬ increasing_arithmetic_sequence_with_difference_5 a.val b.val c.val d.val := 
sorry

end count_valid_abcd_is_zero_l24_24126


namespace distance_of_route_l24_24654

theorem distance_of_route (Vq : ℝ) (Vy : ℝ) (D : ℝ) (h1 : Vy = 1.5 * Vq) (h2 : D = Vq * 2) (h3 : D = Vy * 1.3333333333333333) : D = 1.5 :=
by
  sorry

end distance_of_route_l24_24654


namespace fraction_multiplication_l24_24671

theorem fraction_multiplication (x : ℚ) (h : x = 236 / 100) : x * 3 = 177 / 25 :=
by
  sorry

end fraction_multiplication_l24_24671


namespace part1_part2_l24_24219

-- Part 1
theorem part1 (x : ℝ) (h1 : 2 * x = 3 * x - 1) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h2 : x < 0) (h3 : |2 * x| + |3 * x - 1| = 16) : x = -3 :=
by
  sorry

end part1_part2_l24_24219


namespace students_present_in_class_l24_24205

theorem students_present_in_class :
  ∀ (total_students absent_percentage : ℕ), 
    total_students = 50 → absent_percentage = 12 → 
    (88 * total_students / 100) = 44 :=
by
  intros total_students absent_percentage h1 h2
  sorry

end students_present_in_class_l24_24205


namespace simplify_and_rationalize_l24_24289

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l24_24289


namespace clara_age_l24_24868

theorem clara_age (x : ℕ) (n m : ℕ) (h1 : x - 2 = n^2) (h2 : x + 3 = m^3) : x = 123 :=
by sorry

end clara_age_l24_24868


namespace power_mod_equiv_l24_24040

-- Define the main theorem
theorem power_mod_equiv {a n k : ℕ} (h₁ : a ≥ 2) (h₂ : n ≥ 1) :
  (a^k ≡ 1 [MOD (a^n - 1)]) ↔ (k % n = 0) :=
by sorry

end power_mod_equiv_l24_24040


namespace distance_to_Rock_Mist_Mountains_l24_24640

theorem distance_to_Rock_Mist_Mountains
  (d_Sky_Falls : ℕ) (d_Sky_Falls_eq : d_Sky_Falls = 8)
  (d_Rock_Mist : ℕ) (d_Rock_Mist_eq : d_Rock_Mist = 50 * d_Sky_Falls)
  (detour_Thunder_Pass : ℕ) (detour_Thunder_Pass_eq : detour_Thunder_Pass = 25) :
  d_Rock_Mist + detour_Thunder_Pass = 425 := by
  sorry

end distance_to_Rock_Mist_Mountains_l24_24640


namespace cos_angle_difference_l24_24129

theorem cos_angle_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1): 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_angle_difference_l24_24129


namespace isosceles_triangle_base_length_l24_24400

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l24_24400


namespace equal_area_bisecting_line_slope_l24_24928

theorem equal_area_bisecting_line_slope 
  (circle1_center circle2_center : ℝ × ℝ) 
  (radius : ℝ) 
  (line_point : ℝ × ℝ) 
  (h1 : circle1_center = (20, 100))
  (h2 : circle2_center = (25, 90))
  (h3 : radius = 4)
  (h4 : line_point = (20, 90))
  : ∃ (m : ℝ), |m| = 2 :=
by
  sorry

end equal_area_bisecting_line_slope_l24_24928


namespace solve_for_x_l24_24417

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end solve_for_x_l24_24417


namespace central_angle_remains_unchanged_l24_24645

theorem central_angle_remains_unchanged
  (r l : ℝ)
  (h_r : r > 0)
  (h_l : l > 0) :
  (l / r) = (2 * l) / (2 * r) :=
by
  sorry

end central_angle_remains_unchanged_l24_24645


namespace gondor_repaired_3_phones_on_monday_l24_24209

theorem gondor_repaired_3_phones_on_monday :
  ∃ P : ℕ, 
    (10 * P + 10 * 5 + 20 * 2 + 20 * 4 = 200) ∧
    P = 3 :=
by
  sorry

end gondor_repaired_3_phones_on_monday_l24_24209


namespace polynomial_value_sum_l24_24227

theorem polynomial_value_sum
  (a b c d : ℝ)
  (f : ℝ → ℝ)
  (Hf : ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d)
  (H1 : f 1 = 1) (H2 : f 2 = 2) (H3 : f 3 = 3) :
  f 0 + f 4 = 28 :=
sorry

end polynomial_value_sum_l24_24227


namespace problem_solution_l24_24929

-- Define the arithmetic sequence and its sum
def arith_seq_sum (n : ℕ) (a1 d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Define the specific condition for our problem
def a1_a5_equal_six (a1 d : ℕ) : Prop :=
  a1 + (a1 + 4 * d) = 6

-- The target value of S5 that we want to prove
def S5 (a1 d : ℕ) : ℕ :=
  arith_seq_sum 5 a1 d

theorem problem_solution (a1 d : ℕ) (h : a1_a5_equal_six a1 d) : S5 a1 d = 15 :=
by
  sorry

end problem_solution_l24_24929


namespace decimal_properties_l24_24038

theorem decimal_properties :
  (3.00 : ℝ) = (3 : ℝ) :=
by sorry

end decimal_properties_l24_24038
