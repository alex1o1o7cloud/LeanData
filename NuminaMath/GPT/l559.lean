import Mathlib

namespace dig_site_date_l559_55941

theorem dig_site_date (S1 S2 S3 S4 : ℕ) (S2_bc : S2 = 852) 
  (h1 : S1 = S2 - 352) 
  (h2 : S3 = S1 + 3700) 
  (h3 : S4 = 2 * S3) : 
  S4 = 6400 :=
by sorry

end dig_site_date_l559_55941


namespace fraction_subtraction_l559_55935

theorem fraction_subtraction (a b : ℝ) (h1 : 2 * b = 1 + a * b) (h2 : a ≠ 1) (h3 : b ≠ 1) :
  (a + 1) / (a - 1) - (b + 1) / (b - 1) = 2 :=
by
  sorry

end fraction_subtraction_l559_55935


namespace triangle_inequality_l559_55907

theorem triangle_inequality 
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  2 * (a + b + c) * (a * b + b * c + c * a) ≤ (a + b + c) * (a^2 + b^2 + c^2) + 9 * a * b * c :=
by
  sorry

end triangle_inequality_l559_55907


namespace sum_last_two_digits_is_correct_l559_55911

def fibs : List Nat := [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

def factorial_last_two_digits (n : Nat) : Nat :=
  (Nat.factorial n) % 100

def modified_fib_factorial_series : List Nat :=
  fibs.map (λ k => (factorial_last_two_digits k + 2) % 100)

def sum_last_two_digits : Nat :=
  (modified_fib_factorial_series.sum) % 100

theorem sum_last_two_digits_is_correct :
  sum_last_two_digits = 14 :=
sorry

end sum_last_two_digits_is_correct_l559_55911


namespace part1_part2_l559_55972

theorem part1 (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 :=
sorry

theorem part2 (a b : ℝ) (n : ℝ) :
  n = 6 → (a > 0 ∧ b > 0 ∧ (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = 1) → (4 * a + 7 * b) ≥ 9 :=
sorry

end part1_part2_l559_55972


namespace eval_expression_l559_55956

theorem eval_expression : (825 * 825) - (824 * 826) = 1 := by
  sorry

end eval_expression_l559_55956


namespace prob_snow_both_days_l559_55988

-- Definitions for the conditions
def prob_snow_monday : ℚ := 40 / 100
def prob_snow_tuesday : ℚ := 30 / 100

def independent_events (A B : Prop) : Prop := true  -- A placeholder definition of independence

-- The proof problem: 
theorem prob_snow_both_days : 
  independent_events (prob_snow_monday = 0.40) (prob_snow_tuesday = 0.30) →
  prob_snow_monday * prob_snow_tuesday = 0.12 := 
by 
  sorry

end prob_snow_both_days_l559_55988


namespace total_hours_l559_55957

variable (K : ℕ) (P : ℕ) (M : ℕ)

-- Conditions:
axiom h1 : P = 2 * K
axiom h2 : P = (1 / 3 : ℝ) * M
axiom h3 : M = K + 105

-- Goal: Proving the total number of hours is 189
theorem total_hours : K + P + M = 189 := by
  sorry

end total_hours_l559_55957


namespace min_distance_between_graphs_l559_55950

noncomputable def minimum_distance (a : ℝ) (h : 1 < a) : ℝ :=
  if h1 : a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a)

theorem min_distance_between_graphs (a : ℝ) (h1 : 1 < a) :
  minimum_distance a h1 = 
  if a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a) :=
by
  intros
  sorry

end min_distance_between_graphs_l559_55950


namespace exists_n_prime_divides_exp_sum_l559_55959

theorem exists_n_prime_divides_exp_sum (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) :=
by
  sorry

end exists_n_prime_divides_exp_sum_l559_55959


namespace average_last_three_l559_55992

theorem average_last_three (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d + e + f + g) / 7 = 65) 
  (h2 : (a + b + c + d) / 4 = 60) : 
  (e + f + g) / 3 = 71.67 :=
by
  sorry

end average_last_three_l559_55992


namespace second_planner_cheaper_l559_55944

theorem second_planner_cheaper (x : ℕ) :
  (∀ x, 250 + 15 * x < 150 + 18 * x → x ≥ 34) :=
by
  intros x h
  sorry

end second_planner_cheaper_l559_55944


namespace find_matrix_N_l559_55906

theorem find_matrix_N (N : Matrix (Fin 4) (Fin 4) ℤ)
  (hi : N.mulVec ![1, 0, 0, 0] = ![3, 4, -9, 1])
  (hj : N.mulVec ![0, 1, 0, 0] = ![-1, 6, -3, 2])
  (hk : N.mulVec ![0, 0, 1, 0] = ![8, -2, 5, 0])
  (hl : N.mulVec ![0, 0, 0, 1] = ![1, 0, 7, -1]) :
  N = ![![3, -1, 8, 1],
         ![4, 6, -2, 0],
         ![-9, -3, 5, 7],
         ![1, 2, 0, -1]] := by
  sorry

end find_matrix_N_l559_55906


namespace units_digit_smallest_n_l559_55984

theorem units_digit_smallest_n (n : ℕ) (h1 : 7 * n ≥ 10^2015) (h2 : 7 * (n - 1) < 10^2015) : (n % 10) = 6 :=
sorry

end units_digit_smallest_n_l559_55984


namespace max_y_value_l559_55918

theorem max_y_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = (x - y) / (x + 3 * y)) : y ≤ 1 / 3 :=
by
  sorry

end max_y_value_l559_55918


namespace smallest_three_digit_solution_l559_55947

theorem smallest_three_digit_solution :
  ∃ n : ℕ, 70 * n ≡ 210 [MOD 350] ∧ 100 ≤ n ∧ n = 103 :=
by
  sorry

end smallest_three_digit_solution_l559_55947


namespace hyperbola_equation_l559_55946

theorem hyperbola_equation 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_asymptote : (b / a) = (Real.sqrt 3 / 2))
  (c : ℝ) (hc : c = Real.sqrt 7)
  (foci_directrix_condition : a^2 + b^2 = c^2) :
  (∀ x y : ℝ, (x^2 / 4 - y^2 / 3 = 1)) :=
by
  -- We do not provide the proof as per instructions
  sorry

end hyperbola_equation_l559_55946


namespace lisa_marbles_l559_55908

def ConnieMarbles : ℕ := 323
def JuanMarbles (ConnieMarbles : ℕ) : ℕ := ConnieMarbles + 175
def MarkMarbles (JuanMarbles : ℕ) : ℕ := 3 * JuanMarbles
def LisaMarbles (MarkMarbles : ℕ) : ℕ := MarkMarbles / 2 - 200

theorem lisa_marbles :
  LisaMarbles (MarkMarbles (JuanMarbles ConnieMarbles)) = 547 := by
  sorry

end lisa_marbles_l559_55908


namespace max_students_for_distribution_l559_55919

theorem max_students_for_distribution : 
  ∃ (n : Nat), (∀ k, k ∣ 1048 ∧ k ∣ 828 → k ≤ n) ∧ 
               (n ∣ 1048 ∧ n ∣ 828) ∧ 
               n = 4 :=
by
  sorry

end max_students_for_distribution_l559_55919


namespace no_valid_n_values_l559_55965

theorem no_valid_n_values :
  ¬ ∃ n : ℕ, (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end no_valid_n_values_l559_55965


namespace minimum_value_of_F_l559_55902

noncomputable def F (m n : ℝ) : ℝ := (m - n)^2 + (m^2 - n + 1)^2

theorem minimum_value_of_F : 
  (∀ m n : ℝ, F m n ≥ 9 / 32) ∧ (∃ m n : ℝ, F m n = 9 / 32) :=
by
  sorry

end minimum_value_of_F_l559_55902


namespace complementary_implies_right_triangle_l559_55975

theorem complementary_implies_right_triangle (A B C : ℝ) (h : A + B = 90 ∧ A + B + C = 180) :
  C = 90 :=
by
  sorry

end complementary_implies_right_triangle_l559_55975


namespace max_reflections_max_reflections_example_l559_55955

-- Definition of the conditions
def angle_cda := 10  -- angle in degrees
def max_angle := 90  -- practical limit for angle of reflections

-- Given that the angle of incidence after n reflections is 10n degrees,
-- prove that the largest possible n is 9 before exceeding practical limits.
theorem max_reflections (n : ℕ) (h₁ : angle_cda = 10) (h₂ : max_angle = 90) :
  10 * n ≤ 90 :=
by sorry

-- Specific case instantiating n = 9
theorem max_reflections_example : (10 : ℕ) * 9 ≤ 90 := max_reflections 9 rfl rfl

end max_reflections_max_reflections_example_l559_55955


namespace log_simplification_l559_55979

open Real

theorem log_simplification (a b d e z y : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (ha : a ≠ 0) (hz : z ≠ 0) (hy : y ≠ 0) :
  log (a / b) + log (b / e) + log (e / d) - log (az / dy) = log (dy / z) :=
by
  sorry

end log_simplification_l559_55979


namespace train_lengths_equal_l559_55985

theorem train_lengths_equal (v_fast v_slow : ℝ) (t : ℝ) (L : ℝ)  
  (h1 : v_fast = 46) 
  (h2 : v_slow = 36) 
  (h3 : t = 36.00001) : 
  2 * L = (v_fast - v_slow) / 3600 * t → L = 1800.0005 := 
by
  sorry

end train_lengths_equal_l559_55985


namespace symmetric_circle_eq_l559_55994

theorem symmetric_circle_eq :
  (∃ f : ℝ → ℝ → Prop, (∀ x y, f x y ↔ (x - 2)^2 + (y + 1)^2 = 1)) →
  (∃ line : ℝ → ℝ → Prop, (∀ x y, line x y ↔ x - y + 3 = 0)) →
  (∃ eq : ℝ → ℝ → Prop, (∀ x y, eq x y ↔ (x - 4)^2 + (y - 5)^2 = 1)) :=
by
  sorry

end symmetric_circle_eq_l559_55994


namespace integral_x_squared_l559_55980

theorem integral_x_squared:
  ∫ x in (0:ℝ)..(1:ℝ), x^2 = 1/3 :=
by
  sorry

end integral_x_squared_l559_55980


namespace simplify_and_evaluate_expression_l559_55971

-- Define the conditions
def a := 2
def b := -1

-- State the theorem
theorem simplify_and_evaluate_expression : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 3 * a * b) / (-b) = -12 := by
  -- Placeholder for the proof
  sorry

end simplify_and_evaluate_expression_l559_55971


namespace incorrect_statement_A_l559_55912

theorem incorrect_statement_A (x_1 x_2 y_1 y_2 : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2*x - 4*y - 4 = 0) ∧
  x_1 = 1 - Real.sqrt 5 ∧
  x_2 = 1 + Real.sqrt 5 ∧
  y_1 = 2 - 2 * Real.sqrt 2 ∧
  y_2 = 2 + 2 * Real.sqrt 2 →
  x_1 + x_2 ≠ -2 := by
  intro h
  sorry

end incorrect_statement_A_l559_55912


namespace investment_return_l559_55940

theorem investment_return (y_r : ℝ) :
  (500 + 1500) * 0.085 = 500 * 0.07 + 1500 * y_r → y_r = 0.09 :=
by
  sorry

end investment_return_l559_55940


namespace sum_of_segments_AK_KB_eq_AB_l559_55916

-- Given conditions: length of segment AB is 9 cm
def length_AB : ℝ := 9

-- For any point K on segment AB, prove that AK + KB = AB
theorem sum_of_segments_AK_KB_eq_AB (K : ℝ) (h : 0 ≤ K ∧ K ≤ length_AB) : 
  K + (length_AB - K) = length_AB := by
  sorry

end sum_of_segments_AK_KB_eq_AB_l559_55916


namespace max_m_for_inequality_min_4a2_9b2_c2_l559_55922

theorem max_m_for_inequality (m : ℝ) : (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 := 
sorry

theorem min_4a2_9b2_c2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (4 * a^2 + 9 * b^2 + c^2) = 36 / 49 ∧ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49 :=
sorry

end max_m_for_inequality_min_4a2_9b2_c2_l559_55922


namespace scientific_notation_of_56_point_5_million_l559_55981

-- Definitions based on conditions
def million : ℝ := 10^6
def number_in_millions : ℝ := 56.5 * million

-- Statement to be proved
theorem scientific_notation_of_56_point_5_million : 
  number_in_millions = 5.65 * 10^7 :=
sorry

end scientific_notation_of_56_point_5_million_l559_55981


namespace ratio_unit_price_brand_x_to_brand_y_l559_55993

-- Definitions based on the conditions in the problem
def volume_brand_y (v : ℝ) := v
def price_brand_y (p : ℝ) := p
def volume_brand_x (v : ℝ) := 1.3 * v
def price_brand_x (p : ℝ) := 0.85 * p
noncomputable def unit_price (volume : ℝ) (price : ℝ) := price / volume

-- Theorems to prove the ratio of unit price of Brand X to Brand Y is 17/26
theorem ratio_unit_price_brand_x_to_brand_y (v p : ℝ) (hv : v ≠ 0) (hp : p ≠ 0) : 
  (unit_price (volume_brand_x v) (price_brand_x p)) / (unit_price (volume_brand_y v) (price_brand_y p)) = 17 / 26 := by
  sorry

end ratio_unit_price_brand_x_to_brand_y_l559_55993


namespace squirrels_cannot_divide_equally_l559_55938

theorem squirrels_cannot_divide_equally
    (n : ℕ) : ¬ (∃ k, 2022 + n * (n + 1) = 5 * k) :=
by
sorry

end squirrels_cannot_divide_equally_l559_55938


namespace shirts_sold_l559_55958

theorem shirts_sold (S : ℕ) (H_total : 69 = 7 * 7 + 5 * S) : S = 4 :=
by
  sorry -- Placeholder for the proof

end shirts_sold_l559_55958


namespace number_of_pages_in_chunk_l559_55999

-- Conditions
def first_page : Nat := 213
def last_page : Nat := 312

-- Define the property we need to prove
theorem number_of_pages_in_chunk : last_page - first_page + 1 = 100 := by
  -- skipping the proof
  sorry

end number_of_pages_in_chunk_l559_55999


namespace find_constants_l559_55987

-- Define constants and the problem
variables (C D Q : Type) [AddCommGroup Q] [Module ℝ Q]
variables (CQ QD : ℝ) (h_ratio : CQ = 3 * QD / 5)

-- Define the conjecture we want to prove
theorem find_constants (t u : ℝ) (h_t : t = 5 / (3 + 5)) (h_u : u = 3 / (3 + 5)) :
  (CQ = 3 * QD / 5) → 
  (t * CQ + u * QD = (5 / 8) * CQ + (3 / 8) * QD) :=
sorry

end find_constants_l559_55987


namespace arithmetic_sequence_a4_is_5_l559_55969

variable (a : ℕ → ℕ)

-- Arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m k : ℕ, n < m ∧ m < k → 2 * a m = a n + a k

-- Given condition
axiom sum_third_and_fifth : a 3 + a 5 = 10

-- Prove that a_4 = 5
theorem arithmetic_sequence_a4_is_5
  (h : is_arithmetic_sequence a) : a 4 = 5 := by
  sorry

end arithmetic_sequence_a4_is_5_l559_55969


namespace number_of_months_to_fully_pay_off_car_l559_55905

def total_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem number_of_months_to_fully_pay_off_car :
  (total_price - initial_payment) / monthly_payment = 19 :=
by
  sorry

end number_of_months_to_fully_pay_off_car_l559_55905


namespace bob_calories_l559_55967

-- conditions
def slices : ℕ := 8
def half_slices (slices : ℕ) : ℕ := slices / 2
def calories_per_slice : ℕ := 300
def total_calories (half_slices : ℕ) (calories_per_slice : ℕ) : ℕ := half_slices * calories_per_slice

-- proof problem
theorem bob_calories : total_calories (half_slices slices) calories_per_slice = 1200 := by
  sorry

end bob_calories_l559_55967


namespace sum_of_five_consecutive_odd_integers_l559_55913

theorem sum_of_five_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 8) = 156) :
  n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 390 :=
by
  sorry

end sum_of_five_consecutive_odd_integers_l559_55913


namespace solution_set_of_inequality_l559_55939

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 4 * x - 3 > 0 } = { x : ℝ | 1 < x ∧ x < 3 } := sorry

end solution_set_of_inequality_l559_55939


namespace total_cost_of_books_l559_55966

theorem total_cost_of_books (C1 C2 : ℝ) 
  (hC1 : C1 = 268.33)
  (h_selling_prices_equal : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 459.15 :=
by
  -- placeholder for the proof
  sorry

end total_cost_of_books_l559_55966


namespace right_triangle_third_angle_l559_55964

-- Define the problem
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the given angles
def is_right_angle (a : ℝ) : Prop := a = 90
def given_angle (b : ℝ) : Prop := b = 25

-- Define the third angle
def third_angle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove 
theorem right_triangle_third_angle : ∀ (a b c : ℝ), 
  is_right_angle a → given_angle b → third_angle a b c → c = 65 :=
by
  intros a b c ha hb h_triangle
  sorry

end right_triangle_third_angle_l559_55964


namespace law_firm_associates_l559_55961

def percentage (total: ℕ) (part: ℕ): ℕ := part * 100 / total

theorem law_firm_associates (total: ℕ) (second_year: ℕ) (first_year: ℕ) (more_than_two_years: ℕ):
  percentage total more_than_two_years = 50 →
  percentage total second_year = 25 →
  first_year = more_than_two_years - second_year →
  percentage total first_year = 25 →
  percentage total (total - first_year) = 75 :=
by
  intros h1 h2 h3 h4
  sorry

end law_firm_associates_l559_55961


namespace percentage_of_invalid_votes_l559_55953

theorem percentage_of_invalid_votes:
  ∃ (A B V I VV : ℕ), 
    V = 5720 ∧
    B = 1859 ∧
    A = B + 15 / 100 * V ∧
    VV = A + B ∧
    V = VV + I ∧
    (I: ℚ) / V * 100 = 20 :=
by
  sorry

end percentage_of_invalid_votes_l559_55953


namespace range_of_a_minimize_S_l559_55924

open Real

-- Problem 1: Prove the range of a 
theorem range_of_a (a : ℝ) : (∃ x ≠ 0, x^3 - 3*x^2 + (2 - a)*x = 0) ↔ a > -1 / 4 := sorry

-- Problem 2: Prove the minimizing value of a for the area function S(a)
noncomputable def S (a : ℝ) : ℝ := 
  let α := sorry -- α is the root depending on a (to be determined from the context)
  let β := sorry -- β is the root depending on a (to be determined from the context)
  (1/4 * α^4 - α^3 + (1/2) * (2-a) * α^2) + (1/4 * β^4 - β^3 + (1/2) * (2-a) * β^2)

theorem minimize_S (a : ℝ) : a = 38 - 27 * sqrt 2 → S a = S (38 - 27 * sqrt 2) := sorry

end range_of_a_minimize_S_l559_55924


namespace lesser_number_l559_55927

theorem lesser_number (x y : ℕ) (h1: x + y = 60) (h2: x - y = 10) : y = 25 :=
sorry

end lesser_number_l559_55927


namespace gain_percent_correct_l559_55931

noncomputable def cycleCP : ℝ := 900
noncomputable def cycleSP : ℝ := 1180
noncomputable def gainPercent : ℝ := (cycleSP - cycleCP) / cycleCP * 100

theorem gain_percent_correct :
  gainPercent = 31.11 := by
  sorry

end gain_percent_correct_l559_55931


namespace hours_per_toy_l559_55963

-- Defining the conditions
def toys_produced (hours: ℕ) : ℕ := 40 
def hours_worked : ℕ := 80

-- Theorem: If a worker makes 40 toys in 80 hours, then it takes 2 hours to make one toy.
theorem hours_per_toy : (hours_worked / toys_produced hours_worked) = 2 :=
by
  sorry

end hours_per_toy_l559_55963


namespace deficit_percentage_l559_55934

variable (A B : ℝ) -- Actual lengths of the sides of the rectangle
variable (x : ℝ) -- Percentage in deficit
variable (measuredA := A * 1.06) -- One side measured 6% in excess
variable (errorPercent := 0.7) -- Error percent in area
variable (measuredB := B * (1 - x / 100)) -- Other side measured x% in deficit
variable (actualArea := A * B) -- Actual area of the rectangle
variable (calculatedArea := (A * 1.06) * (B * (1 - x / 100))) -- Calculated area with measurement errors
variable (correctArea := actualArea * (1 + errorPercent / 100)) -- Correct area considering the error

theorem deficit_percentage : 
  calculatedArea = correctArea → 
  x = 5 :=
by
  sorry

end deficit_percentage_l559_55934


namespace divisibility_equiv_l559_55928

-- Definition of the functions a(n) and b(n)
def a (n : ℕ) := n^5 + 5^n
def b (n : ℕ) := n^5 * 5^n + 1

-- Define a positive integer
variables (n : ℕ) (hn : n > 0)

-- The theorem stating the equivalence
theorem divisibility_equiv : (a n) % 11 = 0 ↔ (b n) % 11 = 0 :=
sorry
 
end divisibility_equiv_l559_55928


namespace hypotenuse_min_length_l559_55937

theorem hypotenuse_min_length
  (a b l : ℝ)
  (h_area : (1/2) * a * b = 8)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = l)
  (h_min_l : l = 8 + 4 * Real.sqrt 2) :
  Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_min_length_l559_55937


namespace sum_of_cubes_application_l559_55986

theorem sum_of_cubes_application : 
  ¬ ((a+1) * (a^2 - a + 1) = a^3 + 1) :=
by
  sorry

end sum_of_cubes_application_l559_55986


namespace factor_polynomial_l559_55915

def A (x : ℝ) : ℝ := x^2 + 5 * x + 3
def B (x : ℝ) : ℝ := x^2 + 9 * x + 20
def C (x : ℝ) : ℝ := x^2 + 7 * x - 8

theorem factor_polynomial (x : ℝ) :
  (A x) * (B x) + (C x) = (x^2 + 7 * x + 8) * (x^2 + 7 * x + 14) :=
by
  sorry

end factor_polynomial_l559_55915


namespace payment_denotation_is_correct_l559_55949

-- Define the initial condition of receiving money
def received_amount : ℤ := 120

-- Define the payment amount
def payment_amount : ℤ := 85

-- The expected payoff
def expected_payment_denotation : ℤ := -85

-- Theorem stating that the payment should be denoted as -85 yuan
theorem payment_denotation_is_correct : (payment_amount = -expected_payment_denotation) :=
by
  sorry

end payment_denotation_is_correct_l559_55949


namespace pet_store_animals_left_l559_55978

theorem pet_store_animals_left (initial_birds initial_puppies initial_cats initial_spiders initial_snakes : ℕ)
  (donation_fraction snakes_share_sold birds_sold puppies_adopted cats_transferred kittens_brought : ℕ)
  (spiders_loose spiders_captured : ℕ)
  (H_initial_birds : initial_birds = 12)
  (H_initial_puppies : initial_puppies = 9)
  (H_initial_cats : initial_cats = 5)
  (H_initial_spiders : initial_spiders = 15)
  (H_initial_snakes : initial_snakes = 8)
  (H_donation_fraction : donation_fraction = 25)
  (H_snakes_share_sold : snakes_share_sold = (donation_fraction * initial_snakes) / 100)
  (H_birds_sold : birds_sold = initial_birds / 2)
  (H_puppies_adopted : puppies_adopted = 3)
  (H_cats_transferred : cats_transferred = 4)
  (H_kittens_brought : kittens_brought = 2)
  (H_spiders_loose : spiders_loose = 7)
  (H_spiders_captured : spiders_captured = 5) :
  (initial_snakes - snakes_share_sold) + (initial_birds - birds_sold) + 
  (initial_puppies - puppies_adopted) + (initial_cats - cats_transferred + kittens_brought) + 
  (initial_spiders - (spiders_loose - spiders_captured)) = 34 := 
by 
  sorry

end pet_store_animals_left_l559_55978


namespace pizza_order_cost_l559_55917

def base_cost_per_pizza : ℕ := 10
def cost_per_topping : ℕ := 1
def topping_count_pepperoni : ℕ := 1
def topping_count_sausage : ℕ := 1
def topping_count_black_olive_and_mushroom : ℕ := 2
def tip : ℕ := 5

theorem pizza_order_cost :
  3 * base_cost_per_pizza + (topping_count_pepperoni * cost_per_topping) + (topping_count_sausage * cost_per_topping) + (topping_count_black_olive_and_mushroom * cost_per_topping) + tip = 39 := by
  sorry

end pizza_order_cost_l559_55917


namespace min_deliveries_to_cover_cost_l559_55982

theorem min_deliveries_to_cover_cost (cost_per_van earnings_per_delivery gasoline_cost_per_delivery : ℕ) (h1 : cost_per_van = 4500) (h2 : earnings_per_delivery = 15 ) (h3 : gasoline_cost_per_delivery = 5) : 
  ∃ d : ℕ, 10 * d ≥ cost_per_van ∧ ∀ x : ℕ, x < d → 10 * x < cost_per_van :=
by
  use 450
  sorry

end min_deliveries_to_cover_cost_l559_55982


namespace bamboo_middle_node_capacity_l559_55970

def capacities_form_arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem bamboo_middle_node_capacity :
  ∃ (a : ℕ → ℚ) (d : ℚ), 
    capacities_form_arithmetic_sequence a d ∧ 
    (a 1 + a 2 + a 3 = 4) ∧
    (a 6 + a 7 + a 8 + a 9 = 3) ∧
    (a 5 = 67 / 66) :=
  sorry

end bamboo_middle_node_capacity_l559_55970


namespace maximum_value_of_m_solve_inequality_l559_55995

theorem maximum_value_of_m (a b : ℝ) (h : a ≠ 0) : 
  ∃ m : ℝ, (∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ m * |a|) ∧ (m = 2) :=
by
  use 2
  sorry

theorem solve_inequality (x : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| ≤ 2 → (1/2 ≤ x ∧ x ≤ 5/2)) :=
by
  sorry

end maximum_value_of_m_solve_inequality_l559_55995


namespace area_of_fourth_rectangle_l559_55921

theorem area_of_fourth_rectangle (A B C D E F G H I J K L : Type) 
  (x y z w : ℕ) (a1 : x * y = 20) (a2 : x * w = 12) (a3 : z * w = 16) : 
  y * w = 16 :=
by sorry

end area_of_fourth_rectangle_l559_55921


namespace problem_solution_l559_55948

variable (x : ℝ)

-- Given condition
def condition1 : Prop := (7 / 8) * x = 28

-- The main statement to prove
theorem problem_solution (h : condition1 x) : (x + 16) * (5 / 16) = 15 := by
  sorry

end problem_solution_l559_55948


namespace value_of_f_l559_55976

def f (x z : ℕ) (y : ℕ) : ℕ := 2 * x^2 + y - z

theorem value_of_f (y : ℕ) (h1 : f 2 3 y = 100) : f 5 7 y = 138 := by
  sorry

end value_of_f_l559_55976


namespace find_difference_l559_55983

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end find_difference_l559_55983


namespace expression_constant_for_large_x_l559_55991

theorem expression_constant_for_large_x (x : ℝ) (h : x ≥ 4 / 7) : 
  -4 * x + |4 - 7 * x| - |1 - 3 * x| + 4 = 1 :=
by
  sorry

end expression_constant_for_large_x_l559_55991


namespace calculate_flat_tax_l559_55920

open Real

def price_per_sq_ft (property: String) : Real :=
  if property = "Condo" then 98
  else if property = "BarnHouse" then 84
  else if property = "DetachedHouse" then 102
  else if property = "Townhouse" then 96
  else if property = "Garage" then 60
  else if property = "PoolArea" then 50
  else 0

def area_in_sq_ft (property: String) : Real :=
  if property = "Condo" then 2400
  else if property = "BarnHouse" then 1200
  else if property = "DetachedHouse" then 3500
  else if property = "Townhouse" then 2750
  else if property = "Garage" then 480
  else if property = "PoolArea" then 600
  else 0

def total_value : Real :=
  (price_per_sq_ft "Condo" * area_in_sq_ft "Condo") +
  (price_per_sq_ft "BarnHouse" * area_in_sq_ft "BarnHouse") +
  (price_per_sq_ft "DetachedHouse" * area_in_sq_ft "DetachedHouse") +
  (price_per_sq_ft "Townhouse" * area_in_sq_ft "Townhouse") +
  (price_per_sq_ft "Garage" * area_in_sq_ft "Garage") +
  (price_per_sq_ft "PoolArea" * area_in_sq_ft "PoolArea")

def tax_rate : Real := 0.0125

theorem calculate_flat_tax : total_value * tax_rate = 12697.50 := by
  sorry

end calculate_flat_tax_l559_55920


namespace total_time_is_correct_l559_55929

-- Defining the number of items
def chairs : ℕ := 7
def tables : ℕ := 3
def bookshelves : ℕ := 2
def lamps : ℕ := 4

-- Defining the time spent on each type of furniture
def time_per_chair : ℕ := 4
def time_per_table : ℕ := 8
def time_per_bookshelf : ℕ := 12
def time_per_lamp : ℕ := 2

-- Defining the total time calculation
def total_time : ℕ :=
  (chairs * time_per_chair) + 
  (tables * time_per_table) +
  (bookshelves * time_per_bookshelf) +
  (lamps * time_per_lamp)

-- Theorem stating the total time
theorem total_time_is_correct : total_time = 84 :=
by
  -- Skipping the proof details
  sorry

end total_time_is_correct_l559_55929


namespace find_p_q_sum_p_plus_q_l559_55973

noncomputable def probability_third_six : ℚ :=
  have fair_die_prob_two_sixes := (1 / 6) * (1 / 6)
  have biased_die_prob_two_sixes := (2 / 3) * (2 / 3)
  have total_prob_two_sixes := (1 / 2) * fair_die_prob_two_sixes + (1 / 2) * biased_die_prob_two_sixes
  have prob_fair_given_two_sixes := fair_die_prob_two_sixes / total_prob_two_sixes
  have prob_biased_given_two_sixes := biased_die_prob_two_sixes / total_prob_two_sixes
  let prob_third_six :=
    prob_fair_given_two_sixes * (1 / 6) +
    prob_biased_given_two_sixes * (2 / 3)
  prob_third_six

theorem find_p_q_sum : 
  probability_third_six = 65 / 102 :=
by sorry

theorem p_plus_q : 
  65 + 102 = 167 :=
by sorry

end find_p_q_sum_p_plus_q_l559_55973


namespace value_of_leftover_coins_l559_55962

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 40

def ana_quarters : ℕ := 95
def ana_dimes : ℕ := 183

def ben_quarters : ℕ := 104
def ben_dimes : ℕ := 219

def leftover_quarters : ℕ := (ana_quarters + ben_quarters) % quarters_per_roll
def leftover_dimes : ℕ := (ana_dimes + ben_dimes) % dimes_per_roll

def dollar_value (quarters dimes : ℕ) : ℝ := quarters * 0.25 + dimes * 0.10

theorem value_of_leftover_coins : 
  dollar_value leftover_quarters leftover_dimes = 6.95 := 
  sorry

end value_of_leftover_coins_l559_55962


namespace find_C_plus_D_l559_55989

theorem find_C_plus_D (C D : ℝ) (h : ∀ x : ℝ, (Cx - 20) / (x^2 - 3 * x - 10) = D / (x + 2) + 4 / (x - 5)) :
  C + D = 4.7 :=
sorry

end find_C_plus_D_l559_55989


namespace hyperbola_sufficient_not_necessary_condition_l559_55925

-- Define the equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 16 = 1

-- Define the asymptotic line equations of the hyperbola
def asymptotes_eq (x y : ℝ) : Prop :=
  y = 2 * x ∨ y = -2 * x

-- Prove that the equation of the hyperbola is a sufficient but not necessary condition for the asymptotic lines
theorem hyperbola_sufficient_not_necessary_condition :
  (∀ x y : ℝ, hyperbola_eq x y → asymptotes_eq x y) ∧ ¬ (∀ x y : ℝ, asymptotes_eq x y → hyperbola_eq x y) :=
by
  sorry

end hyperbola_sufficient_not_necessary_condition_l559_55925


namespace Sue_waited_in_NY_l559_55954

-- Define the conditions as constants and assumptions
def T_NY_SF : ℕ := 24
def T_total : ℕ := 58
def T_NO_NY : ℕ := (3 * T_NY_SF) / 4

-- Define the waiting time
def T_wait : ℕ := T_total - T_NO_NY - T_NY_SF

-- Theorem stating the problem
theorem Sue_waited_in_NY :
  T_wait = 16 :=
by
  -- Implicitly using the given conditions
  sorry

end Sue_waited_in_NY_l559_55954


namespace simplify_and_evaluate_expression_l559_55901

theorem simplify_and_evaluate_expression (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 2) : 
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) :=
by
  sorry

example : (∃ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ (x = 3) ∧ ((x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = 5)) :=
  ⟨3, by norm_num, by norm_num, rfl, by norm_num⟩

end simplify_and_evaluate_expression_l559_55901


namespace factorization_correct_l559_55900

theorem factorization_correct (x : ℝ) : 2 * x^2 - 6 * x - 8 = 2 * (x - 4) * (x + 1) :=
by
  sorry

end factorization_correct_l559_55900


namespace find_value_of_fraction_l559_55998

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l559_55998


namespace laps_run_l559_55951

theorem laps_run (x : ℕ) (total_distance required_distance lap_length extra_laps : ℕ) (h1 : total_distance = 2400) (h2 : lap_length = 150) (h3 : extra_laps = 4) (h4 : total_distance = lap_length * (x + extra_laps)) : x = 12 :=
by {
  sorry
}

end laps_run_l559_55951


namespace sum_50th_set_l559_55914

-- Definition of the sequence repeating pattern
def repeating_sequence : List (List Nat) :=
  [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]

-- Definition to get the nth set in the repeating sequence
def nth_set (n : Nat) : List Nat :=
  repeating_sequence.get! ((n - 1) % 4)

-- Definition to sum the elements of a list
def sum_list (l : List Nat) : Nat :=
  l.sum

-- Proposition to prove that the sum of the 50th set is 4
theorem sum_50th_set : sum_list (nth_set 50) = 4 :=
by
  sorry

end sum_50th_set_l559_55914


namespace range_of_a_min_value_ab_range_of_y_l559_55903
-- Import the necessary Lean library 

-- Problem 1
theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x - 3| ≥ a^2 + a) → (-2 ≤ a ∧ a ≤ 1) := 
sorry

-- Problem 2
theorem min_value_ab (a b : ℝ) (h₁ : a + b = 1) : 
  (∀ x, |x - 1| + |x - 3| ≥ a^2 + a) → 
  (min ((1 : ℝ) / (4 * |b|) + |b| / a) = 3 / 4 ∧ (a = 2)) :=
sorry

-- Problem 3
theorem range_of_y (a : ℝ) (y : ℝ) (h₁ : a ∈ Set.Ici (2 : ℝ)) : 
  y = (2 * a) / (a^2 + 1) → 0 < y ∧ y ≤ (4 / 5) :=
sorry

end range_of_a_min_value_ab_range_of_y_l559_55903


namespace vector_dot_product_zero_implies_orthogonal_l559_55909

theorem vector_dot_product_zero_implies_orthogonal
  (a b : ℝ → ℝ)
  (h0 : ∀ (x y : ℝ), a x * b y = 0) :
  ¬(a = 0 ∨ b = 0) := 
sorry

end vector_dot_product_zero_implies_orthogonal_l559_55909


namespace supply_without_leak_last_for_20_days_l559_55943

variable (C V : ℝ)

-- Condition 1: if there is a 10-liter leak per day, the supply lasts for 15 days
axiom h1 : C = 15 * (V + 10)

-- Condition 2: if there is a 20-liter leak per day, the supply lasts for 12 days
axiom h2 : C = 12 * (V + 20)

-- The problem to prove: without any leak, the tank can supply water to the village for 20 days
theorem supply_without_leak_last_for_20_days (C V : ℝ) (h1 : C = 15 * (V + 10)) (h2 : C = 12 * (V + 20)) : C / V = 20 := 
by 
  sorry

end supply_without_leak_last_for_20_days_l559_55943


namespace intersecting_lines_l559_55996

theorem intersecting_lines (x y : ℝ) : x ^ 2 - y ^ 2 = 0 ↔ (y = x ∨ y = -x) := by
  sorry

end intersecting_lines_l559_55996


namespace solve_for_x_l559_55942

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l559_55942


namespace simplify_expression_l559_55932

theorem simplify_expression : 
  (1 / ((1 / ((1 / 2)^1)) + (1 / ((1 / 2)^3)) + (1 / ((1 / 2)^4)))) = (1 / 26) := 
by 
  sorry

end simplify_expression_l559_55932


namespace cooling_time_condition_l559_55936

theorem cooling_time_condition :
  ∀ (θ0 θ1 θ1' θ0' : ℝ) (t : ℝ), 
    θ0 = 20 → θ1 = 100 → θ1' = 60 → θ0' = 20 →
    let θ := θ0 + (θ1 - θ0) * Real.exp (-t / 4)
    let θ' := θ0' + (θ1' - θ0') * Real.exp (-t / 4)
    (θ - θ' ≤ 10) → (t ≥ 5.52) :=
sorry

end cooling_time_condition_l559_55936


namespace area_of_vegetable_patch_l559_55974

theorem area_of_vegetable_patch : ∃ (a b : ℕ), 
  (2 * (a + b) = 24 ∧ b = 3 * a + 2 ∧ (6 * (a + 1)) * (6 * (b + 1)) = 576) :=
sorry

end area_of_vegetable_patch_l559_55974


namespace polynomial_root_problem_l559_55904

theorem polynomial_root_problem (a b c d : ℤ) (r1 r2 r3 r4 : ℕ)
  (h_roots : ∀ x, x^4 + a * x^3 + b * x^2 + c * x + d = (x + r1) * (x + r2) * (x + r3) * (x + r4))
  (h_sum : a + b + c + d = 2009) :
  d = 528 := 
by
  sorry

end polynomial_root_problem_l559_55904


namespace problem1_problem2_l559_55977

section problems

variables (m n a b : ℕ)
variables (h1 : 4 ^ m = a) (h2 : 8 ^ n = b)

theorem problem1 : 2 ^ (2 * m + 3 * n) = a * b :=
sorry

theorem problem2 : 2 ^ (4 * m - 6 * n) = a ^ 2 / b ^ 2 :=
sorry

end problems

end problem1_problem2_l559_55977


namespace max_m_plus_n_l559_55997

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

theorem max_m_plus_n (m n : ℝ) (h : n = quadratic_function m) : m + n ≤ 13/4 :=
sorry

end max_m_plus_n_l559_55997


namespace product_ABCD_is_9_l559_55952

noncomputable def A : ℝ := Real.sqrt 2018 + Real.sqrt 2019 + 1
noncomputable def B : ℝ := -Real.sqrt 2018 - Real.sqrt 2019 - 1
noncomputable def C : ℝ := Real.sqrt 2018 - Real.sqrt 2019 + 1
noncomputable def D : ℝ := Real.sqrt 2019 - Real.sqrt 2018 + 1

theorem product_ABCD_is_9 : A * B * C * D = 9 :=
by sorry

end product_ABCD_is_9_l559_55952


namespace sabrina_fraction_books_second_month_l559_55990

theorem sabrina_fraction_books_second_month (total_books : ℕ) (pages_per_book : ℕ) (books_first_month : ℕ) (pages_total_read : ℕ)
  (h_total_books : total_books = 14)
  (h_pages_per_book : pages_per_book = 200)
  (h_books_first_month : books_first_month = 4)
  (h_pages_total_read : pages_total_read = 1000) :
  let total_pages := total_books * pages_per_book
  let pages_first_month := books_first_month * pages_per_book
  let pages_remaining := total_pages - pages_first_month
  let books_remaining := total_books - books_first_month
  let pages_read_first_month := total_pages - pages_total_read
  let pages_read_second_month := pages_read_first_month - pages_first_month
  let books_second_month := pages_read_second_month / pages_per_book
  let fraction_books := books_second_month / books_remaining
  fraction_books = 1 / 2 :=
by
  sorry

end sabrina_fraction_books_second_month_l559_55990


namespace fraction_sum_reciprocal_ge_two_l559_55945

theorem fraction_sum_reciprocal_ge_two (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a / b) + (b / a) ≥ 2 :=
sorry

end fraction_sum_reciprocal_ge_two_l559_55945


namespace range_of_m_l559_55910

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, abs (x - 3) + abs (x - m) < 5) : -2 < m ∧ m < 8 :=
  sorry

end range_of_m_l559_55910


namespace least_number_to_add_l559_55926

theorem least_number_to_add (k : ℕ) (h : 1019 % 25 = 19) : (1019 + k) % 25 = 0 ↔ k = 6 :=
by
  sorry

end least_number_to_add_l559_55926


namespace number_of_zeros_l559_55930

noncomputable def f (x : Real) : Real :=
if x > 0 then -1 + Real.log x
else 3 * x + 4

theorem number_of_zeros : (∃ a b : Real, f a = 0 ∧ f b = 0 ∧ a ≠ b) := 
sorry

end number_of_zeros_l559_55930


namespace math_problem_l559_55960

variable {a b c : ℝ}

theorem math_problem
  (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0)
  (h : a + b + c = -a * b * c) :
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) +
  a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) +
  b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 :=
by
  sorry

end math_problem_l559_55960


namespace arithmetic_sequence_20th_term_l559_55923

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 5
  let n := 20
  let a_n := a + (n - 1) * d
  a_n = 97 := by
  sorry

end arithmetic_sequence_20th_term_l559_55923


namespace first_month_sale_l559_55933

def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029
def sale6 : ℕ := 4937
def average_sale : ℕ := 5600

theorem first_month_sale :
  let total_sales := average_sale * 6
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  let sale1 := total_sales - known_sales
  sale1 = 5266 :=
by
  sorry

end first_month_sale_l559_55933


namespace factorial_fraction_integer_l559_55968

open Nat

theorem factorial_fraction_integer (m n : ℕ) : 
  ∃ k : ℕ, k = (2 * m).factorial * (2 * n).factorial / (m.factorial * n.factorial * (m + n).factorial) := 
sorry

end factorial_fraction_integer_l559_55968
