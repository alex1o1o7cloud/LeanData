import Mathlib

namespace NUMINAMATH_GPT_remainder_of_x_squared_div_20_l2068_206884

theorem remainder_of_x_squared_div_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 4 * x ≡ 12 [ZMOD 20]) :
  (x * x) % 20 = 4 :=
sorry

end NUMINAMATH_GPT_remainder_of_x_squared_div_20_l2068_206884


namespace NUMINAMATH_GPT_num_multiples_6_not_12_lt_300_l2068_206840

theorem num_multiples_6_not_12_lt_300 : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k < 300 ∧ k % 6 = 0 ∧ k % 12 ≠ 0 → ∃ m : ℕ, k = 6 * (2 * m - 1) ∧ 1 ≤ m ∧ m ≤ 25 := 
by
  sorry

end NUMINAMATH_GPT_num_multiples_6_not_12_lt_300_l2068_206840


namespace NUMINAMATH_GPT_number_of_sophomores_l2068_206899

theorem number_of_sophomores (n x : ℕ) (freshmen seniors selected freshmen_selected : ℕ)
  (h_freshmen : freshmen = 450)
  (h_seniors : seniors = 250)
  (h_selected : selected = 60)
  (h_freshmen_selected : freshmen_selected = 27)
  (h_eq : selected / (freshmen + seniors + x) = freshmen_selected / freshmen) :
  x = 300 := by
  sorry

end NUMINAMATH_GPT_number_of_sophomores_l2068_206899


namespace NUMINAMATH_GPT_combined_time_alligators_walked_l2068_206872

-- Define the conditions
def original_time : ℕ := 4
def return_time := original_time + 2 * Int.sqrt original_time

-- State the theorem to be proven
theorem combined_time_alligators_walked : original_time + return_time = 12 := by
  sorry

end NUMINAMATH_GPT_combined_time_alligators_walked_l2068_206872


namespace NUMINAMATH_GPT_ammonium_iodide_required_l2068_206808

theorem ammonium_iodide_required
  (KOH_moles NH3_moles KI_moles H2O_moles : ℕ)
  (hn : NH3_moles = 3) (hk : KOH_moles = 3) (hi : KI_moles = 3) (hw : H2O_moles = 3) :
  ∃ NH4I_moles, NH3_moles = 3 ∧ KI_moles = 3 ∧ H2O_moles = 3 ∧ KOH_moles = 3 ∧ NH4I_moles = 3 :=
by
  sorry

end NUMINAMATH_GPT_ammonium_iodide_required_l2068_206808


namespace NUMINAMATH_GPT_average_test_score_before_dropping_l2068_206842

theorem average_test_score_before_dropping (A B C : ℝ) :
  (A + B + C) / 3 = 40 → (A + B + C + 20) / 4 = 35 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_average_test_score_before_dropping_l2068_206842


namespace NUMINAMATH_GPT_solve_system_of_equations_l2068_206880

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 := 
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2068_206880


namespace NUMINAMATH_GPT_squares_of_natural_numbers_l2068_206852

theorem squares_of_natural_numbers (x y z : ℕ) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) : ∃ a b c : ℕ, x = a^2 ∧ y = b^2 ∧ z = c^2 := 
by
  sorry

end NUMINAMATH_GPT_squares_of_natural_numbers_l2068_206852


namespace NUMINAMATH_GPT_age_difference_of_declans_sons_l2068_206876

theorem age_difference_of_declans_sons 
  (current_age_elder_son : ℕ) 
  (future_age_younger_son : ℕ) 
  (years_until_future : ℕ) 
  (current_age_elder_son_eq : current_age_elder_son = 40) 
  (future_age_younger_son_eq : future_age_younger_son = 60) 
  (years_until_future_eq : years_until_future = 30) :
  (current_age_elder_son - (future_age_younger_son - years_until_future)) = 10 := by
  sorry

end NUMINAMATH_GPT_age_difference_of_declans_sons_l2068_206876


namespace NUMINAMATH_GPT_percentage_of_absent_students_l2068_206828

theorem percentage_of_absent_students (total_students boys girls : ℕ) (fraction_boys_absent fraction_girls_absent : ℚ)
  (total_students_eq : total_students = 180)
  (boys_eq : boys = 120)
  (girls_eq : girls = 60)
  (fraction_boys_absent_eq : fraction_boys_absent = 1/6)
  (fraction_girls_absent_eq : fraction_girls_absent = 1/4) :
  let boys_absent := fraction_boys_absent * boys
  let girls_absent := fraction_girls_absent * girls
  let total_absent := boys_absent + girls_absent
  let absent_percentage := (total_absent / total_students) * 100
  abs (absent_percentage - 19) < 1 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_absent_students_l2068_206828


namespace NUMINAMATH_GPT_question1_question2_l2068_206875

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem question1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ 1 := sorry

theorem question2 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  x1 * Real.log x1 - x1 * Real.log x2 > x1 - x2 := sorry

end NUMINAMATH_GPT_question1_question2_l2068_206875


namespace NUMINAMATH_GPT_family_percentage_eaten_after_dinner_l2068_206821

theorem family_percentage_eaten_after_dinner
  (total_brownies : ℕ)
  (children_percentage : ℚ)
  (left_over_brownies : ℕ)
  (lorraine_extra_brownie : ℕ)
  (remaining_percentage : ℚ) :
  total_brownies = 16 →
  children_percentage = 0.25 →
  lorraine_extra_brownie = 1 →
  left_over_brownies = 5 →
  remaining_percentage = 50 := by
  sorry

end NUMINAMATH_GPT_family_percentage_eaten_after_dinner_l2068_206821


namespace NUMINAMATH_GPT_student_courses_last_year_l2068_206826

variable (x : ℕ)
variable (courses_last_year : ℕ := x)
variable (avg_grade_last_year : ℕ := 100)
variable (courses_year_before : ℕ := 5)
variable (avg_grade_year_before : ℕ := 60)
variable (avg_grade_two_years : ℕ := 81)

theorem student_courses_last_year (h1 : avg_grade_last_year = 100)
                                   (h2 : courses_year_before = 5)
                                   (h3 : avg_grade_year_before = 60)
                                   (h4 : avg_grade_two_years = 81)
                                   (hc : ((5 * avg_grade_year_before) + (courses_last_year * avg_grade_last_year)) / (courses_year_before + courses_last_year) = avg_grade_two_years) :
                                   courses_last_year = 6 := by
  sorry

end NUMINAMATH_GPT_student_courses_last_year_l2068_206826


namespace NUMINAMATH_GPT_value_of_m_l2068_206888

theorem value_of_m (m x : ℝ) (h : x = 3) (h_eq : 3 * m - 2 * x = 6) : m = 4 := by
  -- Given x = 3
  subst h
  -- Now we have to show m = 4
  sorry

end NUMINAMATH_GPT_value_of_m_l2068_206888


namespace NUMINAMATH_GPT_min_k_value_l2068_206823

-- Definition of the problem's conditions
def remainder_condition (n k : ℕ) : Prop :=
  ∀ i, 2 ≤ i → i ≤ k → n % i = i - 1

def in_range (x a b : ℕ) : Prop :=
  a < x ∧ x < b

-- The statement of the proof problem in Lean 4
theorem min_k_value (n k : ℕ) (h1 : remainder_condition n k) (hn_range : in_range n 2000 3000) :
  k = 9 :=
sorry

end NUMINAMATH_GPT_min_k_value_l2068_206823


namespace NUMINAMATH_GPT_greatest_fraction_lt_17_l2068_206896

theorem greatest_fraction_lt_17 :
  ∃ (x : ℚ), x = 15 / 4 ∧ x^2 < 17 ∧ ∀ y : ℚ, y < 4 → y^2 < 17 → y ≤ 15 / 4 := 
by
  use 15 / 4
  sorry

end NUMINAMATH_GPT_greatest_fraction_lt_17_l2068_206896


namespace NUMINAMATH_GPT_find_smallest_c_l2068_206836

/-- Let a₀, a₁, ... and b₀, b₁, ... be geometric sequences with common ratios rₐ and r_b, 
respectively, such that ∑ i=0 ∞ aᵢ = ∑ i=0 ∞ bᵢ = 1 and 
(∑ i=0 ∞ aᵢ²)(∑ i=0 ∞ bᵢ²) = ∑ i=0 ∞ aᵢbᵢ. Prove that a₀ < 4/3 -/
theorem find_smallest_c (r_a r_b : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∑' n, a n = 1)
  (h2 : ∑' n, b n = 1)
  (h3 : (∑' n, (a n)^2) * (∑' n, (b n)^2) = ∑' n, (a n) * (b n)) :
  a 0 < 4 / 3 := by
  sorry

end NUMINAMATH_GPT_find_smallest_c_l2068_206836


namespace NUMINAMATH_GPT_age_problem_l2068_206839

variables (K M A B : ℕ)

theorem age_problem
  (h1 : K + 7 = 3 * M)
  (h2 : M = 5)
  (h3 : A + B = 2 * M + 4)
  (h4 : A = B - 3)
  (h5 : K + B = M + 9) :
  K = 8 ∧ M = 5 ∧ B = 6 ∧ A = 3 :=
sorry

end NUMINAMATH_GPT_age_problem_l2068_206839


namespace NUMINAMATH_GPT_longest_tape_length_l2068_206894

/-!
  Problem: Find the length of the longest tape that can exactly measure the lengths 
  24 m, 36 m, and 54 m in cm.
  
  Solution: Convert the given lengths to the same unit (cm), then find their GCD.
  
  Given: Lengths are 2400 cm, 3600 cm, and 5400 cm.
  To Prove: gcd(2400, 3600, 5400) = 300.
-/

theorem longest_tape_length (a b c : ℕ) : a = 2400 → b = 3600 → c = 5400 → Nat.gcd (Nat.gcd a b) c = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- omitted proof steps
  sorry

end NUMINAMATH_GPT_longest_tape_length_l2068_206894


namespace NUMINAMATH_GPT_correct_equation_l2068_206801

def initial_investment : ℝ := 2500
def expected_investment : ℝ := 6600
def growth_rate (x : ℝ) : ℝ := x

theorem correct_equation (x : ℝ) : 
  initial_investment * (1 + growth_rate x) + initial_investment * (1 + growth_rate x)^2 = expected_investment :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l2068_206801


namespace NUMINAMATH_GPT_a_range_l2068_206830

variables {x a : ℝ}

def p (x : ℝ) : Prop := (4 * x - 3) ^ 2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem a_range (h : ∀ x, ¬p x → ¬q x a ∧ (∃ x, q x a ∧ ¬p x)) :
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_GPT_a_range_l2068_206830


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l2068_206897

theorem repeating_decimal_to_fraction : 
∀ (x : ℝ), x = 4 + (0.0036 / (1 - 0.01)) → x = 144/33 :=
by
  intro x hx
  -- This is a placeholder where the conversion proof would go.
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l2068_206897


namespace NUMINAMATH_GPT_winner_beats_by_16_secons_l2068_206870

-- Definitions of the times for mathematician and physicist
variables (x y : ℕ)

-- Conditions based on the given problem
def condition1 := 2 * y - x = 24
def condition2 := 2 * x - y = 72

-- The statement to prove
theorem winner_beats_by_16_secons (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x - 2 * y = 16 := 
sorry

end NUMINAMATH_GPT_winner_beats_by_16_secons_l2068_206870


namespace NUMINAMATH_GPT_solve_log_eq_l2068_206871

theorem solve_log_eq (x : ℝ) (h : 0 < x) :
  (1 / (Real.sqrt (Real.logb 5 (5 * x)) + Real.sqrt (Real.logb 5 x)) + Real.sqrt (Real.logb 5 x) = 2) ↔ x = 125 := 
  sorry

end NUMINAMATH_GPT_solve_log_eq_l2068_206871


namespace NUMINAMATH_GPT_initial_action_figures_l2068_206885

theorem initial_action_figures (x : ℕ) (h : x + 2 - 7 = 10) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_initial_action_figures_l2068_206885


namespace NUMINAMATH_GPT_combined_annual_income_after_expenses_l2068_206819

noncomputable def brady_monthly_incomes : List ℕ := [150, 200, 250, 300, 200, 150, 180, 220, 240, 270, 300, 350]
noncomputable def dwayne_monthly_incomes : List ℕ := [100, 150, 200, 250, 150, 120, 140, 190, 180, 230, 260, 300]
def brady_annual_expense : ℕ := 450
def dwayne_annual_expense : ℕ := 300

def annual_income (monthly_incomes : List ℕ) : ℕ :=
  monthly_incomes.foldr (· + ·) 0

theorem combined_annual_income_after_expenses :
  (annual_income brady_monthly_incomes - brady_annual_expense) +
  (annual_income dwayne_monthly_incomes - dwayne_annual_expense) = 3930 :=
by
  sorry

end NUMINAMATH_GPT_combined_annual_income_after_expenses_l2068_206819


namespace NUMINAMATH_GPT_fractions_integer_or_fractional_distinct_l2068_206817

theorem fractions_integer_or_fractional_distinct (a b : Fin 6 → ℕ) (h_pos : ∀ i, 0 < a i ∧ 0 < b i)
  (h_irreducible : ∀ i, Nat.gcd (a i) (b i) = 1)
  (h_sum_eq : (Finset.univ : Finset (Fin 6)).sum a = (Finset.univ : Finset (Fin 6)).sum b) :
  ¬ ∀ i j : Fin 6, i ≠ j → ((a i / b i = a j / b j) ∨ (a i % b i / b i = a j % b j / b j)) :=
sorry

end NUMINAMATH_GPT_fractions_integer_or_fractional_distinct_l2068_206817


namespace NUMINAMATH_GPT_rewrite_expression_l2068_206809

theorem rewrite_expression (k : ℝ) :
  ∃ d r s : ℝ, (8 * k^2 - 12 * k + 20 = d * (k + r)^2 + s) ∧ (r + s = 14.75) := 
sorry

end NUMINAMATH_GPT_rewrite_expression_l2068_206809


namespace NUMINAMATH_GPT_total_buyers_l2068_206866

-- Definitions based on conditions
def C : ℕ := 50
def M : ℕ := 40
def B : ℕ := 19
def pN : ℝ := 0.29  -- Probability that a random buyer purchases neither

-- The theorem statement
theorem total_buyers :
  ∃ T : ℝ, (T = (C + M - B) + pN * T) ∧ T = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_buyers_l2068_206866


namespace NUMINAMATH_GPT_greatest_perimeter_isosceles_triangle_l2068_206887

theorem greatest_perimeter_isosceles_triangle :
  let base := 12
  let height := 15
  let segments := 6
  let max_perimeter := 32.97
  -- Assuming division such that each of the 6 pieces is of equal area,
  -- the greatest perimeter among these pieces to the nearest hundredth is:
  (∀ (base height segments : ℝ), base = 12 ∧ height = 15 ∧ segments = 6 → 
   max_perimeter = 32.97) :=
by
  sorry

end NUMINAMATH_GPT_greatest_perimeter_isosceles_triangle_l2068_206887


namespace NUMINAMATH_GPT_point_symmetric_second_quadrant_l2068_206806

theorem point_symmetric_second_quadrant (m : ℝ) 
  (symmetry : ∃ x y : ℝ, P = (-m, m-3) ∧ (-x, -y) = (x, y)) 
  (second_quadrant : ∃ x y : ℝ, P = (-m, m-3) ∧ x < 0 ∧ y > 0) : 
  m < 0 := 
sorry

end NUMINAMATH_GPT_point_symmetric_second_quadrant_l2068_206806


namespace NUMINAMATH_GPT_birdhouse_flight_distance_l2068_206856

variable (car_distance : ℕ)
variable (lawn_chair_distance : ℕ)
variable (birdhouse_distance : ℕ)

def problem_condition1 := car_distance = 200
def problem_condition2 := lawn_chair_distance = 2 * car_distance
def problem_condition3 := birdhouse_distance = 3 * lawn_chair_distance

theorem birdhouse_flight_distance
  (h1 : car_distance = 200)
  (h2 : lawn_chair_distance = 2 * car_distance)
  (h3 : birdhouse_distance = 3 * lawn_chair_distance) :
  birdhouse_distance = 1200 := by
  sorry

end NUMINAMATH_GPT_birdhouse_flight_distance_l2068_206856


namespace NUMINAMATH_GPT_total_clothes_washed_l2068_206835

theorem total_clothes_washed (cally_white_shirts : ℕ) (cally_colored_shirts : ℕ) (cally_shorts : ℕ) (cally_pants : ℕ) 
                             (danny_white_shirts : ℕ) (danny_colored_shirts : ℕ) (danny_shorts : ℕ) (danny_pants : ℕ) 
                             (total_clothes : ℕ)
                             (hcally : cally_white_shirts = 10 ∧ cally_colored_shirts = 5 ∧ cally_shorts = 7 ∧ cally_pants = 6)
                             (hdanny : danny_white_shirts = 6 ∧ danny_colored_shirts = 8 ∧ danny_shorts = 10 ∧ danny_pants = 6)
                             (htotal : total_clothes = 58) : 
  cally_white_shirts + cally_colored_shirts + cally_shorts + cally_pants + 
  danny_white_shirts + danny_colored_shirts + danny_shorts + danny_pants = total_clothes := 
by {
  sorry
}

end NUMINAMATH_GPT_total_clothes_washed_l2068_206835


namespace NUMINAMATH_GPT_range_of_c_l2068_206815

def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (1 - 2 * c < 0)

theorem range_of_c (c : ℝ) : (p c ∨ q c) ∧ ¬ (p c ∧ q c) ↔ (0 < c ∧ c ≤ 1/2) ∨ (1 < c) :=
by sorry

end NUMINAMATH_GPT_range_of_c_l2068_206815


namespace NUMINAMATH_GPT_roberto_valid_outfits_l2068_206827

-- Definitions based on the conditions
def total_trousers : ℕ := 6
def total_shirts : ℕ := 8
def total_jackets : ℕ := 4
def restricted_jacket : ℕ := 1
def restricted_shirts : ℕ := 2

-- Theorem statement
theorem roberto_valid_outfits : 
  total_trousers * total_shirts * total_jackets - total_trousers * restricted_shirts * restricted_jacket = 180 := 
by
  sorry

end NUMINAMATH_GPT_roberto_valid_outfits_l2068_206827


namespace NUMINAMATH_GPT_eliminate_xy_l2068_206890

variable {R : Type*} [Ring R]

theorem eliminate_xy
  (x y a b c : R)
  (h1 : a = x + y)
  (h2 : b = x^3 + y^3)
  (h3 : c = x^5 + y^5) :
  5 * b * (a^3 + b) = a * (a^5 + 9 * c) :=
sorry

end NUMINAMATH_GPT_eliminate_xy_l2068_206890


namespace NUMINAMATH_GPT_expand_and_simplify_l2068_206847

-- Define the two polynomials P and Q.
def P (x : ℝ) := 5 * x + 3
def Q (x : ℝ) := 2 * x^2 - x + 4

-- State the theorem we want to prove.
theorem expand_and_simplify (x : ℝ) : (P x * Q x) = 10 * x^3 + x^2 + 17 * x + 12 := 
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l2068_206847


namespace NUMINAMATH_GPT_custom_op_neg2_neg3_l2068_206825

  def custom_op (a b : ℤ) : ℤ := b^2 - a

  theorem custom_op_neg2_neg3 : custom_op (-2) (-3) = 11 :=
  by
    sorry
  
end NUMINAMATH_GPT_custom_op_neg2_neg3_l2068_206825


namespace NUMINAMATH_GPT_carlos_picks_24_integers_l2068_206834

def is_divisor (n m : ℕ) : Prop := m % n = 0

theorem carlos_picks_24_integers :
  ∃ (s : Finset ℕ), s.card = 24 ∧ ∀ n ∈ s, is_divisor n 4500 ∧ 1 ≤ n ∧ n ≤ 4500 ∧ n % 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_carlos_picks_24_integers_l2068_206834


namespace NUMINAMATH_GPT_boat_stream_speed_l2068_206879

theorem boat_stream_speed (v : ℝ) (h : (60 / (15 - v)) - (60 / (15 + v)) = 2) : v = 3.5 := 
by 
  sorry
 
end NUMINAMATH_GPT_boat_stream_speed_l2068_206879


namespace NUMINAMATH_GPT_solve_for_A_plus_B_l2068_206858

-- Definition of the problem conditions
def T := 7 -- The common total sum for rows and columns

-- Summing the rows and columns in the partially filled table
variable (A B : ℕ)
def table_condition :=
  4 + 1 + 2 = T ∧
  2 + A + B = T ∧
  4 + 2 + B = T ∧
  1 + A + B = T

-- Statement to prove
theorem solve_for_A_plus_B (A B : ℕ) (h : table_condition A B) : A + B = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_A_plus_B_l2068_206858


namespace NUMINAMATH_GPT_find_ages_l2068_206802

theorem find_ages (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 5) : x = 2 := 
sorry

end NUMINAMATH_GPT_find_ages_l2068_206802


namespace NUMINAMATH_GPT_largest_vertex_sum_of_parabola_l2068_206800

theorem largest_vertex_sum_of_parabola 
  (a T : ℤ)
  (hT : T ≠ 0)
  (h1 : 0 = a * 0^2 + b * 0 + c)
  (h2 : 0 = a * (2 * T) ^ 2 + b * (2 * T) + c)
  (h3 : 36 = a * (2 * T + 2) ^ 2 + b * (2 * T + 2) + c) :
  ∃ N : ℚ, N = -5 / 4 :=
sorry

end NUMINAMATH_GPT_largest_vertex_sum_of_parabola_l2068_206800


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l2068_206843

-- Definitions of the coefficients
def a : ℝ := 1
def b : ℝ := -1
def c : ℝ := -2

-- Definition of the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The theorem stating the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots :
  discriminant a b c > 0 :=
by
  -- Coefficients specific to the problem
  unfold a b c
  -- Calculate the discriminant
  unfold discriminant
  -- Substitute the values and compute
  sorry -- Skipping the actual proof as per instructions

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l2068_206843


namespace NUMINAMATH_GPT_sum_of_squares_ge_sum_of_products_l2068_206860

theorem sum_of_squares_ge_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_ge_sum_of_products_l2068_206860


namespace NUMINAMATH_GPT_initial_orchid_bushes_l2068_206874

def final_orchid_bushes : ℕ := 35
def orchid_bushes_to_be_planted : ℕ := 13

theorem initial_orchid_bushes :
  final_orchid_bushes - orchid_bushes_to_be_planted = 22 :=
by
  sorry

end NUMINAMATH_GPT_initial_orchid_bushes_l2068_206874


namespace NUMINAMATH_GPT_rate_of_current_is_8_5_l2068_206865

-- Define the constants for the problem
def downstream_speed : ℝ := 24
def upstream_speed : ℝ := 7
def rate_still_water : ℝ := 15.5

-- Define the rate of the current calculation
def rate_of_current : ℝ := downstream_speed - rate_still_water

-- Define the rate of the current proof statement
theorem rate_of_current_is_8_5 :
  rate_of_current = 8.5 :=
by
  -- This skip the actual proof
  sorry

end NUMINAMATH_GPT_rate_of_current_is_8_5_l2068_206865


namespace NUMINAMATH_GPT_cary_initial_wage_l2068_206820

noncomputable def initial_hourly_wage (x : ℝ) : Prop :=
  let first_year_wage := 1.20 * x
  let second_year_wage := 0.75 * first_year_wage
  second_year_wage = 9

theorem cary_initial_wage : ∃ x : ℝ, initial_hourly_wage x ∧ x = 10 := 
by
  use 10
  unfold initial_hourly_wage
  simp
  sorry

end NUMINAMATH_GPT_cary_initial_wage_l2068_206820


namespace NUMINAMATH_GPT_P_investment_calculation_l2068_206804

variable {P_investment : ℝ}
variable (Q_investment : ℝ := 36000)
variable (total_profit : ℝ := 18000)
variable (Q_profit : ℝ := 6001.89)

def P_profit : ℝ := total_profit - Q_profit

theorem P_investment_calculation :
  P_investment = (P_profit * Q_investment) / Q_profit :=
by
  sorry

end NUMINAMATH_GPT_P_investment_calculation_l2068_206804


namespace NUMINAMATH_GPT_camp_problem_l2068_206862

variable (x : ℕ) -- number of girls
variable (y : ℕ) -- number of boys
variable (total_children : ℕ) -- total number of children
variable (girls_cannot_swim : ℕ) -- number of girls who cannot swim
variable (boys_cannot_swim : ℕ) -- number of boys who cannot swim
variable (children_can_swim : ℕ) -- total number of children who can swim
variable (children_cannot_swim : ℕ) -- total number of children who cannot swim
variable (o_six_girls : ℕ) -- one-sixth of the total number of girls
variable (o_eight_boys : ℕ) -- one-eighth of the total number of boys

theorem camp_problem 
    (hc1 : total_children = 50)
    (hc2 : girls_cannot_swim = x / 6)
    (hc3 : boys_cannot_swim = y / 8)
    (hc4 : children_can_swim = 43)
    (hc5 : children_cannot_swim = total_children - children_can_swim)
    (h_total : x + y = total_children)
    (h_swim : children_cannot_swim = girls_cannot_swim + boys_cannot_swim) :
    x = 18 :=
  by
    have hc6 : children_cannot_swim = 7 := by sorry -- from hc4 and hc5
    have h_eq : x / 6 + (50 - x) / 8 = 7 := by sorry -- from hc2, hc3, hc6
    -- solving for x
    sorry

end NUMINAMATH_GPT_camp_problem_l2068_206862


namespace NUMINAMATH_GPT_cube_surface_divisible_into_12_squares_l2068_206867

theorem cube_surface_divisible_into_12_squares (a : ℝ) :
  (∃ b : ℝ, b = a / Real.sqrt 2 ∧
  ∀ cube_surface_area: ℝ, cube_surface_area = 6 * a^2 →
  ∀ smaller_square_area: ℝ, smaller_square_area = b^2 →
  12 * smaller_square_area = cube_surface_area) :=
sorry

end NUMINAMATH_GPT_cube_surface_divisible_into_12_squares_l2068_206867


namespace NUMINAMATH_GPT_passengers_got_on_in_Texas_l2068_206805

theorem passengers_got_on_in_Texas (start_pax : ℕ) 
  (texas_depart_pax : ℕ) 
  (nc_depart_pax : ℕ) 
  (nc_board_pax : ℕ) 
  (virginia_total_people : ℕ) 
  (crew_members : ℕ) 
  (final_pax_virginia : ℕ) 
  (X : ℕ) :
  start_pax = 124 →
  texas_depart_pax = 58 →
  nc_depart_pax = 47 →
  nc_board_pax = 14 →
  virginia_total_people = 67 →
  crew_members = 10 →
  final_pax_virginia = virginia_total_people - crew_members →
  X + 33 = final_pax_virginia →
  X = 24 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_passengers_got_on_in_Texas_l2068_206805


namespace NUMINAMATH_GPT_find_n_times_s_l2068_206829

noncomputable def g (x : ℝ) : ℝ :=
  if x = 1 then 2011
  else if x = 2 then (1 / 2 + 2010)
  else 0 /- For purposes of the problem -/

theorem find_n_times_s :
  (∀ x y : ℝ, x > 0 → y > 0 → g x * g y = g (x * y) + 2010 * (1 / x + 1 / y + 2010)) →
  ∃ n s : ℝ, n = 1 ∧ s = (4021 / 2) ∧ n * s = 4021 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_times_s_l2068_206829


namespace NUMINAMATH_GPT_jess_remaining_blocks_l2068_206814

-- Define the number of blocks for each segment of Jess's errands
def blocks_to_post_office : Nat := 24
def blocks_to_store : Nat := 18
def blocks_to_gallery : Nat := 15
def blocks_to_library : Nat := 14
def blocks_to_work : Nat := 22
def blocks_already_walked : Nat := 9

-- Calculate the total blocks to be walked
def total_blocks : Nat :=
  blocks_to_post_office + blocks_to_store + blocks_to_gallery + blocks_to_library + blocks_to_work

-- The remaining blocks Jess needs to walk
def blocks_remaining : Nat :=
  total_blocks - blocks_already_walked

-- The statement to be proved
theorem jess_remaining_blocks : blocks_remaining = 84 :=
by
  sorry

end NUMINAMATH_GPT_jess_remaining_blocks_l2068_206814


namespace NUMINAMATH_GPT_inverse_sum_l2068_206831

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x^2

theorem inverse_sum :
  (∃ x₁, g x₁ = -2 ∧ x₁ ≠ 5) ∨ (∃ x₂, g x₂ = 0 ∧ x₂ = 3) ∨ (∃ x₃, g x₃ = 4 ∧ x₃ = -1) → 
  g⁻¹ (-2) + g⁻¹ (0) + g⁻¹ (4) = 6 :=
by
  sorry

end NUMINAMATH_GPT_inverse_sum_l2068_206831


namespace NUMINAMATH_GPT_lucas_total_assignments_l2068_206863

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

end NUMINAMATH_GPT_lucas_total_assignments_l2068_206863


namespace NUMINAMATH_GPT_find_number_l2068_206889

theorem find_number (x : ℝ) (h : 0.36 * x = 129.6) : x = 360 :=
by sorry

end NUMINAMATH_GPT_find_number_l2068_206889


namespace NUMINAMATH_GPT_Enid_made_8_sweaters_l2068_206873

def scarves : ℕ := 10
def sweaters_Aaron : ℕ := 5
def wool_per_scarf : ℕ := 3
def wool_per_sweater : ℕ := 4
def total_wool_used : ℕ := 82
def Enid_sweaters : ℕ := 8

theorem Enid_made_8_sweaters
  (scarves : ℕ)
  (sweaters_Aaron : ℕ)
  (wool_per_scarf : ℕ)
  (wool_per_sweater : ℕ)
  (total_wool_used : ℕ)
  (Enid_sweaters : ℕ)
  : Enid_sweaters = 8 :=
by
  sorry

end NUMINAMATH_GPT_Enid_made_8_sweaters_l2068_206873


namespace NUMINAMATH_GPT_lcm_of_5_6_8_9_l2068_206893

theorem lcm_of_5_6_8_9 : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 := 
by 
  sorry

end NUMINAMATH_GPT_lcm_of_5_6_8_9_l2068_206893


namespace NUMINAMATH_GPT_original_cost_price_l2068_206849

theorem original_cost_price (C : ℝ) 
  (h1 : 0.87 * C > 0) 
  (h2 : 1.2 * (0.87 * C) = 54000) : 
  C = 51724.14 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_price_l2068_206849


namespace NUMINAMATH_GPT_find_difference_l2068_206803

-- Define the problem conditions in Lean
theorem find_difference (a b : ℕ) (hrelprime : Nat.gcd a b = 1)
                        (hpos : a > b) 
                        (hfrac : (a^3 - b^3) / (a - b)^3 = 73 / 3) :
    a - b = 3 :=
by
    sorry

end NUMINAMATH_GPT_find_difference_l2068_206803


namespace NUMINAMATH_GPT_quadratic_expression_factorization_l2068_206861

theorem quadratic_expression_factorization :
  ∃ c d : ℕ, (c > d) ∧ (x^2 - 18*x + 72 = (x - c) * (x - d)) ∧ (4*d - c = 12) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_expression_factorization_l2068_206861


namespace NUMINAMATH_GPT_value_of_expression_l2068_206850

variables (x y z : ℝ)

axiom eq1 : 3 * x - 4 * y - 2 * z = 0
axiom eq2 : 2 * x + 6 * y - 21 * z = 0
axiom z_ne_zero : z ≠ 0

theorem value_of_expression : (x^2 + 4 * x * y) / (y^2 + z^2) = 7 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l2068_206850


namespace NUMINAMATH_GPT_problem_solution_l2068_206857

theorem problem_solution (m : ℤ) (x : ℤ) (h : 4 * x + 2 * m = 14) : x = 2 → m = 3 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l2068_206857


namespace NUMINAMATH_GPT_tens_digit_23_1987_l2068_206812

theorem tens_digit_23_1987 : (23 ^ 1987 % 100) / 10 % 10 = 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_tens_digit_23_1987_l2068_206812


namespace NUMINAMATH_GPT_specific_value_of_n_l2068_206864

theorem specific_value_of_n (n : ℕ) 
  (A_n : ℕ → ℕ)
  (C_n : ℕ → ℕ → ℕ)
  (h1 : A_n n ^ 2 = C_n n (n-3)) :
  n = 8 :=
sorry

end NUMINAMATH_GPT_specific_value_of_n_l2068_206864


namespace NUMINAMATH_GPT_triangle_angle_A_l2068_206881

theorem triangle_angle_A (a c : ℝ) (C A : ℝ) 
  (h1 : a = 4 * Real.sqrt 3)
  (h2 : c = 12)
  (h3 : C = Real.pi / 3)
  (h4 : a < c) :
  A = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_triangle_angle_A_l2068_206881


namespace NUMINAMATH_GPT_min_y_value_l2068_206818

noncomputable def y (a x : ℝ) : ℝ := (Real.exp x - a)^2 + (Real.exp (-x) - a)^2

theorem min_y_value (a : ℝ) (h : a ≠ 0) : 
  (a ≥ 2 → ∃ x, y a x = a^2 - 2) ∧ (a < 2 → ∃ x, y a x = 2*(a-1)^2) :=
sorry

end NUMINAMATH_GPT_min_y_value_l2068_206818


namespace NUMINAMATH_GPT_nonneg_int_solutions_eq_l2068_206855

theorem nonneg_int_solutions_eq (a b : ℕ) : a^2 + b^2 = 841 * (a * b + 1) ↔ (a = 0 ∧ b = 29) ∨ (a = 29 ∧ b = 0) :=
by {
  sorry -- Proof omitted
}

end NUMINAMATH_GPT_nonneg_int_solutions_eq_l2068_206855


namespace NUMINAMATH_GPT_problem_I_inequality_solution_problem_II_condition_on_b_l2068_206853

-- Define the function f(x).
def f (x : ℝ) : ℝ := |x - 2|

-- Problem (I): Proving the solution set to the given inequality.
theorem problem_I_inequality_solution (x : ℝ) : 
  f x + f (x + 1) ≥ 5 ↔ (x ≥ 4 ∨ x ≤ -1) :=
sorry

-- Problem (II): Proving the condition on |b|.
theorem problem_II_condition_on_b (a b : ℝ) (ha : |a| > 1) (h : f (a * b) > |a| * f (b / a)) :
  |b| > 2 :=
sorry

end NUMINAMATH_GPT_problem_I_inequality_solution_problem_II_condition_on_b_l2068_206853


namespace NUMINAMATH_GPT_maryville_population_increase_l2068_206811

def average_people_added_per_year (P2000 P2005 : ℕ) (period : ℕ) : ℕ :=
  (P2005 - P2000) / period
  
theorem maryville_population_increase :
  let P2000 := 450000
  let P2005 := 467000
  let period := 5
  average_people_added_per_year P2000 P2005 period = 3400 :=
by
  sorry

end NUMINAMATH_GPT_maryville_population_increase_l2068_206811


namespace NUMINAMATH_GPT_binary_rep_of_21_l2068_206883

theorem binary_rep_of_21 : 
  (Nat.digits 2 21) = [1, 0, 1, 0, 1] := 
by 
  sorry

end NUMINAMATH_GPT_binary_rep_of_21_l2068_206883


namespace NUMINAMATH_GPT_diet_soda_count_l2068_206846

theorem diet_soda_count (D : ℕ) (h1 : 81 = D + 21) : D = 60 := by
  sorry

end NUMINAMATH_GPT_diet_soda_count_l2068_206846


namespace NUMINAMATH_GPT_repeating_decimals_product_l2068_206848

-- Definitions to represent the conditions
def repeating_decimal_03_as_frac : ℚ := 1 / 33
def repeating_decimal_36_as_frac : ℚ := 4 / 11

-- The statement to be proven
theorem repeating_decimals_product : (repeating_decimal_03_as_frac * repeating_decimal_36_as_frac) = (4 / 363) :=
by {
  sorry
}

end NUMINAMATH_GPT_repeating_decimals_product_l2068_206848


namespace NUMINAMATH_GPT_max_books_borrowed_l2068_206841

theorem max_books_borrowed (students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ) (more_books : ℕ)
  (h_students : students = 30)
  (h_no_books : no_books = 5)
  (h_one_book : one_book = 12)
  (h_two_books : two_books = 8)
  (h_more_books : more_books = students - no_books - one_book - two_books)
  (avg_books : ℕ)
  (h_avg_books : avg_books = 2) :
  ∃ max_books : ℕ, max_books = 20 := 
by 
  sorry

end NUMINAMATH_GPT_max_books_borrowed_l2068_206841


namespace NUMINAMATH_GPT_a_8_value_l2068_206822

variable {n : ℕ}
def S (n : ℕ) : ℕ := n^2
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_8_value : a 8 = 15 := by
  sorry

end NUMINAMATH_GPT_a_8_value_l2068_206822


namespace NUMINAMATH_GPT_clock_tick_intervals_l2068_206837

theorem clock_tick_intervals (intervals_6: ℕ) (intervals_12: ℕ) (total_time_12: ℕ) (interval_time: ℕ):
  intervals_6 = 5 →
  intervals_12 = 11 →
  total_time_12 = 88 →
  interval_time = total_time_12 / intervals_12 →
  intervals_6 * interval_time = 40 :=
by
  intros h1 h2 h3 h4
  -- will continue proof here
  sorry

end NUMINAMATH_GPT_clock_tick_intervals_l2068_206837


namespace NUMINAMATH_GPT_smallest_percentage_increase_l2068_206844

theorem smallest_percentage_increase :
  let n2005 := 75
  let n2006 := 85
  let n2007 := 88
  let n2008 := 94
  let n2009 := 96
  let n2010 := 102
  let perc_increase (a b : ℕ) := ((b - a) : ℚ) / a * 100
  perc_increase n2008 n2009 < perc_increase n2006 n2007 ∧
  perc_increase n2008 n2009 < perc_increase n2007 n2008 ∧
  perc_increase n2008 n2009 < perc_increase n2009 n2010 ∧
  perc_increase n2008 n2009 < perc_increase n2005 n2006
:= sorry

end NUMINAMATH_GPT_smallest_percentage_increase_l2068_206844


namespace NUMINAMATH_GPT_total_men_wages_l2068_206854

-- Define our variables and parameters
variable (M W B : ℝ)
variable (W_women : ℝ)

-- Conditions from the problem:
-- 1. 12M = WW (where WW is W_women)
-- 2. WW = 20B
-- 3. 12M + WW + 20B = 450
axiom eq_12M_WW : 12 * M = W_women
axiom eq_WW_20B : W_women = 20 * B
axiom eq_total_earnings : 12 * M + W_women + 20 * B = 450

-- Prove total wages of the men is Rs. 150
theorem total_men_wages : 12 * M = 150 := by
  sorry

end NUMINAMATH_GPT_total_men_wages_l2068_206854


namespace NUMINAMATH_GPT_parallelogram_height_l2068_206845

theorem parallelogram_height (b A : ℝ) (h : ℝ) (h_base : b = 28) (h_area : A = 896) : h = A / b := by
  simp [h_base, h_area]
  norm_num
  sorry

end NUMINAMATH_GPT_parallelogram_height_l2068_206845


namespace NUMINAMATH_GPT_triangle_inequality_problem_l2068_206898

-- Define the problem statement: Given the specified conditions, prove the interval length and sum
theorem triangle_inequality_problem :
  ∀ (A B C D : Type) (AB AC BC BD CD AD AO : ℝ),
  AB = 12 ∧ CD = 4 →
  (∃ x : ℝ, (4 < x ∧ x < 24) ∧ (AC = x ∧ m = 4 ∧ n = 24 ∧ m + n = 28)) :=
by
  intro A B C D AB AC BC BD CD AD AO h
  sorry

end NUMINAMATH_GPT_triangle_inequality_problem_l2068_206898


namespace NUMINAMATH_GPT_smallest_value_of_x_l2068_206832

theorem smallest_value_of_x :
  ∀ x : ℚ, ( ( (5 * x - 20) / (4 * x - 5) ) ^ 3
           + ( (5 * x - 20) / (4 * x - 5) ) ^ 2
           - ( (5 * x - 20) / (4 * x - 5) )
           - 15 = 0 ) → x = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_x_l2068_206832


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2068_206895

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 1 = 13) (h4 : a_n 4 = 1) : 
  ∃ d : ℤ, d = -4 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2068_206895


namespace NUMINAMATH_GPT_union_is_real_l2068_206851

-- Definitions of sets A and B
def setA : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def setB : Set ℝ := {x | x > -1}

-- Theorem to prove
theorem union_is_real :
  setA ∪ setB = Set.univ :=
by
  sorry

end NUMINAMATH_GPT_union_is_real_l2068_206851


namespace NUMINAMATH_GPT_real_solutions_equation_l2068_206807

theorem real_solutions_equation :
  ∃! x : ℝ, 9 * x^2 - 90 * ⌊ x ⌋ + 99 = 0 :=
sorry

end NUMINAMATH_GPT_real_solutions_equation_l2068_206807


namespace NUMINAMATH_GPT_height_of_stack_of_pots_l2068_206813

-- Definitions corresponding to problem conditions
def pot_thickness : ℕ := 1

def top_pot_diameter : ℕ := 16

def bottom_pot_diameter : ℕ := 4

def diameter_decrement : ℕ := 2

-- Number of pots calculation
def num_pots : ℕ := (top_pot_diameter - bottom_pot_diameter) / diameter_decrement + 1

-- The total vertical distance from the bottom of the lowest pot to the top of the highest pot
def total_vertical_distance : ℕ := 
  let inner_heights := num_pots * (top_pot_diameter - pot_thickness + bottom_pot_diameter - pot_thickness) / 2
  let total_thickness := num_pots * pot_thickness
  inner_heights + total_thickness

theorem height_of_stack_of_pots : total_vertical_distance = 65 := 
sorry

end NUMINAMATH_GPT_height_of_stack_of_pots_l2068_206813


namespace NUMINAMATH_GPT_total_revenue_correct_l2068_206859

def price_per_book : ℝ := 25
def revenue_monday : ℝ := 60 * ((price_per_book * 0.9) * 1.05)
def revenue_tuesday : ℝ := 10 * (price_per_book * 1.03)
def revenue_wednesday : ℝ := 20 * ((price_per_book * 0.95) * 1.02)
def revenue_thursday : ℝ := 44 * ((price_per_book * 0.85) * 1.04)
def revenue_friday : ℝ := 66 * (price_per_book * 0.8)

def total_revenue : ℝ :=
  revenue_monday + revenue_tuesday + revenue_wednesday +
  revenue_thursday + revenue_friday

theorem total_revenue_correct :
  total_revenue = 4452.4 :=
by
  rw [total_revenue, revenue_monday, revenue_tuesday, revenue_wednesday, 
      revenue_thursday, revenue_friday]
  -- Verification steps would continue by calculating each term.
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l2068_206859


namespace NUMINAMATH_GPT_solve_system_of_equations_l2068_206824

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + 2 * x * y + x^2 - 6 * y - 6 * x + 5 = 0)
  (h2 : y - x + 1 = x^2 - 3 * x) : 
  ((x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 2) ∨ (x = -2 ∧ y = 7)) ∧ x ≠ 0 ∧ x ≠ 3 :=
by 
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2068_206824


namespace NUMINAMATH_GPT_finger_cycle_2004th_l2068_206891

def finger_sequence : List String :=
  ["Little finger", "Ring finger", "Middle finger", "Index finger", "Thumb", "Index finger", "Middle finger", "Ring finger"]

theorem finger_cycle_2004th : 
  finger_sequence.get! ((2004 - 1) % finger_sequence.length) = "Index finger" :=
by
  -- The proof is not required, so we use sorry
  sorry

end NUMINAMATH_GPT_finger_cycle_2004th_l2068_206891


namespace NUMINAMATH_GPT_employed_females_part_time_percentage_l2068_206868

theorem employed_females_part_time_percentage (P : ℕ) (hP1 : 0 < P)
  (h1 : ∀ x : ℕ, x = P * 6 / 10) -- 60% of P are employed
  (h2 : ∀ e : ℕ, e = P * 6 / 10) -- e is the number of employed individuals
  (h3 : ∀ f : ℕ, f = e * 4 / 10) -- 40% of employed are females
  (h4 : ∀ pt : ℕ, pt = f * 6 / 10) -- 60% of employed females are part-time
  (h5 : ∀ m : ℕ, m = P * 48 / 100) -- 48% of P are employed males
  (h6 : e = f + m) -- Employed individuals are either males or females
  : f * 6 / f * 10 = 60 := sorry

end NUMINAMATH_GPT_employed_females_part_time_percentage_l2068_206868


namespace NUMINAMATH_GPT_isosceles_triangle_large_angles_l2068_206810

theorem isosceles_triangle_large_angles (y : ℝ) (h : 2 * y + 40 = 180) : y = 70 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_large_angles_l2068_206810


namespace NUMINAMATH_GPT_problem_3_at_7_hash_4_l2068_206878

def oper_at (a b : ℕ) : ℚ := (a * b) / (a + b)
def oper_hash (c d : ℚ) : ℚ := c + d

theorem problem_3_at_7_hash_4 :
  oper_hash (oper_at 3 7) 4 = 61 / 10 := by
  sorry

end NUMINAMATH_GPT_problem_3_at_7_hash_4_l2068_206878


namespace NUMINAMATH_GPT_xy_value_is_one_l2068_206886

open Complex

theorem xy_value_is_one (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : x * y = 1 :=
by
  sorry

end NUMINAMATH_GPT_xy_value_is_one_l2068_206886


namespace NUMINAMATH_GPT_least_number_to_add_l2068_206882

theorem least_number_to_add (n : ℕ) :
  (exists n, 1202 + n % 4 = 0 ∧ (∀ m, (1202 + m) % 4 = 0 → n ≤ m)) → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l2068_206882


namespace NUMINAMATH_GPT_problem1_l2068_206816

theorem problem1 :
  (Real.sqrt (3/2)) * (Real.sqrt (21/4)) / (Real.sqrt (7/2)) = 3/2 :=
sorry

end NUMINAMATH_GPT_problem1_l2068_206816


namespace NUMINAMATH_GPT_triangle_perimeter_l2068_206833

theorem triangle_perimeter (P₁ P₂ P₃ : ℝ) (hP₁ : P₁ = 12) (hP₂ : P₂ = 14) (hP₃ : P₃ = 16) : 
  P₁ + P₂ + P₃ = 42 := by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l2068_206833


namespace NUMINAMATH_GPT_product_of_fraction_l2068_206892

-- Define the repeating decimal as given in the problem
def repeating_decimal : Rat := 0.018 -- represents 0.\overline{018}

-- Define the given fraction obtained by simplifying
def simplified_fraction : Rat := 2 / 111

-- The goal is to prove that the product of the numerator and denominator of 
-- the simplified fraction of the repeating decimal is 222
theorem product_of_fraction (y : Rat) (hy : y = 0.018) (fraction_eq : y = 18 / 999) : 
  (2:ℕ) * (111:ℕ) = 222 :=
by
  sorry

end NUMINAMATH_GPT_product_of_fraction_l2068_206892


namespace NUMINAMATH_GPT_sum_ages_of_brothers_l2068_206877

theorem sum_ages_of_brothers (x : ℝ) (ages : List ℝ) 
  (h1 : ages = [x, x + 1.5, x + 3, x + 4.5, x + 6, x + 7.5, x + 9])
  (h2 : x + 9 = 4 * x) : 
    List.sum ages = 52.5 := 
  sorry

end NUMINAMATH_GPT_sum_ages_of_brothers_l2068_206877


namespace NUMINAMATH_GPT_simplify_fraction_l2068_206869

theorem simplify_fraction (a b c d k : ℕ) (h₁ : a = 123) (h₂ : b = 9999) (h₃ : k = 41)
                           (h₄ : c = a / 3) (h₅ : d = b / 3)
                           (h₆ : c = k) (h₇ : d = 3333) :
  (a * k) / b = (k^2) / d :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2068_206869


namespace NUMINAMATH_GPT_exists_xyz_t_l2068_206838

theorem exists_xyz_t (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0) (h5 : x + y + z + t = 15) : ∃ y, y = 12 :=
by
  sorry

end NUMINAMATH_GPT_exists_xyz_t_l2068_206838
