import Mathlib

namespace NUMINAMATH_GPT_solve_for_x_l928_92882

theorem solve_for_x (x : ℝ) : 
  2.5 * ((3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002 → 
  x = 2.5 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l928_92882


namespace NUMINAMATH_GPT_average_of_numbers_not_1380_l928_92846

def numbers : List ℤ := [1200, 1300, 1400, 1520, 1530, 1200]

theorem average_of_numbers_not_1380 :
  let s := numbers.sum
  let n := numbers.length
  n > 0 → (s / n : ℚ) ≠ 1380 := by
  sorry

end NUMINAMATH_GPT_average_of_numbers_not_1380_l928_92846


namespace NUMINAMATH_GPT_find_m_l928_92822

theorem find_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 ^ a = m) (h4 : 3 ^ b = m) (h5 : 2 * a * b = a + b) : m = Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_find_m_l928_92822


namespace NUMINAMATH_GPT_diamond_value_l928_92810

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem diamond_value : diamond 3 4 = 36 := by
  -- Given condition: x ♢ y = 4x + 6y
  -- To prove: (diamond 3 4) = 36
  sorry

end NUMINAMATH_GPT_diamond_value_l928_92810


namespace NUMINAMATH_GPT_tap_B_time_l928_92802

-- Define the capacities and time variables
variable (A_rate B_rate : ℝ) -- rates in percentage per hour
variable (T_A T_B : ℝ) -- time in hours

-- Define the conditions as hypotheses
def conditions : Prop :=
  (4 * (A_rate + B_rate) = 50) ∧ (2 * A_rate = 15)

-- Define the question and the target time
def target_time := 7

-- Define the goal to prove
theorem tap_B_time (h : conditions A_rate B_rate) : T_B = target_time := by
  sorry

end NUMINAMATH_GPT_tap_B_time_l928_92802


namespace NUMINAMATH_GPT_lines_per_page_l928_92885

theorem lines_per_page
  (total_words : ℕ)
  (words_per_line : ℕ)
  (words_left : ℕ)
  (pages_filled : ℚ) :
  total_words = 400 →
  words_per_line = 10 →
  words_left = 100 →
  pages_filled = 1.5 →
  (total_words - words_left) / words_per_line / pages_filled = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_lines_per_page_l928_92885


namespace NUMINAMATH_GPT_value_of_a8_l928_92871

variable (a : ℕ → ℝ) (a_1 : a 1 = 2) (common_sum : ℝ) (h_sum : common_sum = 5)
variable (equal_sum_sequence : ∀ n, a (n + 1) + a n = common_sum)

theorem value_of_a8 : a 8 = 3 :=
sorry

end NUMINAMATH_GPT_value_of_a8_l928_92871


namespace NUMINAMATH_GPT_find_m_l928_92892

def vector (α : Type) := α × α

noncomputable def dot_product {α} [Add α] [Mul α] (a b : vector α) : α :=
a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) (a : vector ℝ) (b : vector ℝ) (h₁ : a = (1, 2)) (h₂ : b = (m, 1)) (h₃ : dot_product a b = 0) : 
m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l928_92892


namespace NUMINAMATH_GPT_lcm_36_105_l928_92843

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_GPT_lcm_36_105_l928_92843


namespace NUMINAMATH_GPT_recurring_decimal_as_fraction_l928_92862

theorem recurring_decimal_as_fraction :
  0.53 + (247 / 999) * 0.001 = 53171 / 99900 :=
by
  sorry

end NUMINAMATH_GPT_recurring_decimal_as_fraction_l928_92862


namespace NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_l928_92852

variable (a b : ℝ)
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0
def proposition_p (a : ℝ) : Prop := a = 0

theorem necessary_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : isPureImaginary z → proposition_p a := sorry

theorem not_sufficient_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : proposition_p a → ¬isPureImaginary z := sorry

end NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_l928_92852


namespace NUMINAMATH_GPT_length_of_AB_l928_92829

theorem length_of_AB
  (height h : ℝ)
  (AB CD : ℝ)
  (ratio_AB_ADC : (1/2 * AB * h) / (1/2 * CD * h) = 5/4)
  (sum_AB_CD : AB + CD = 300) :
  AB = 166.67 :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_length_of_AB_l928_92829


namespace NUMINAMATH_GPT_elastic_band_radius_increase_l928_92813

theorem elastic_band_radius_increase 
  (C1 C2 : ℝ) 
  (hC1 : C1 = 40) 
  (hC2 : C2 = 80) 
  (hC1_def : C1 = 2 * π * r1) 
  (hC2_def : C2 = 2 * π * r2) :
  r2 - r1 = 20 / π :=
by
  sorry

end NUMINAMATH_GPT_elastic_band_radius_increase_l928_92813


namespace NUMINAMATH_GPT_number_is_165_l928_92886

def is_between (n a b : ℕ) : Prop := a ≤ n ∧ n ≤ b
def is_odd (n : ℕ) : Prop := n % 2 = 1
def contains_digit_5 (n : ℕ) : Prop := ∃ k : ℕ, 10^k * 5 ≤ n ∧ n < 10^(k+1) * 5
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem number_is_165 : 
  (is_between 165 144 169) ∧ 
  (is_odd 165) ∧ 
  (contains_digit_5 165) ∧ 
  (is_divisible_by_3 165) :=
by 
  sorry 

end NUMINAMATH_GPT_number_is_165_l928_92886


namespace NUMINAMATH_GPT_fraction_unseated_l928_92800

theorem fraction_unseated :
  ∀ (tables seats_per_table seats_taken : ℕ),
  tables = 15 →
  seats_per_table = 10 →
  seats_taken = 135 →
  ((tables * seats_per_table - seats_taken : ℕ) / (tables * seats_per_table : ℕ) : ℚ) = 1 / 10 :=
by
  intros tables seats_per_table seats_taken h_tables h_seats_per_table h_seats_taken
  sorry

end NUMINAMATH_GPT_fraction_unseated_l928_92800


namespace NUMINAMATH_GPT_find_quadratic_function_l928_92888

theorem find_quadratic_function (g : ℝ → ℝ) 
  (h1 : g 0 = 0) 
  (h2 : g 1 = 1) 
  (h3 : g (-1) = 5) 
  (h_quadratic : ∃ a b, ∀ x, g x = a * x^2 + b * x) : 
  g = fun x => 3 * x^2 - 2 * x := 
by
  sorry

end NUMINAMATH_GPT_find_quadratic_function_l928_92888


namespace NUMINAMATH_GPT_positive_integer_pairs_l928_92804

theorem positive_integer_pairs (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) :
  (∃ k : ℕ, k > 0 ∧ a^2 = k * (2 * a * b^2 - b^3 + 1)) ↔ 
  ∃ l : ℕ, 0 < l ∧ ((a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l)) :=
by 
  sorry

end NUMINAMATH_GPT_positive_integer_pairs_l928_92804


namespace NUMINAMATH_GPT_product_mod_25_l928_92840

theorem product_mod_25 (m : ℕ) (h : 0 ≤ m ∧ m < 25) : 
  43 * 67 * 92 % 25 = 2 :=
by
  sorry

end NUMINAMATH_GPT_product_mod_25_l928_92840


namespace NUMINAMATH_GPT_number_modulo_conditions_l928_92872

theorem number_modulo_conditions : 
  ∃ n : ℕ, 
  (n % 10 = 9) ∧ 
  (n % 9 = 8) ∧ 
  (n % 8 = 7) ∧ 
  (n % 7 = 6) ∧ 
  (n % 6 = 5) ∧ 
  (n % 5 = 4) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (n % 2 = 1) ∧ 
  (n = 2519) :=
by
  sorry

end NUMINAMATH_GPT_number_modulo_conditions_l928_92872


namespace NUMINAMATH_GPT_A_and_C_together_2_hours_l928_92838

theorem A_and_C_together_2_hours (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1 / 5)
  (hBC : B_rate + C_rate = 1 / 3) (hB : B_rate = 1 / 30) : A_rate + C_rate = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_A_and_C_together_2_hours_l928_92838


namespace NUMINAMATH_GPT_arithmetic_sequence_third_term_l928_92849

theorem arithmetic_sequence_third_term
  (a d : ℤ)
  (h_fifteenth_term : a + 14 * d = 15)
  (h_sixteenth_term : a + 15 * d = 21) :
  a + 2 * d = -57 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_third_term_l928_92849


namespace NUMINAMATH_GPT_union_P_Q_l928_92866

noncomputable def P : Set ℤ := {x | x^2 - x = 0}
noncomputable def Q : Set ℤ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

theorem union_P_Q : P ∪ Q = {-1, 0, 1} :=
by 
  sorry

end NUMINAMATH_GPT_union_P_Q_l928_92866


namespace NUMINAMATH_GPT_orange_ribbons_count_l928_92868

variable (total_ribbons : ℕ)
variable (orange_ribbons : ℚ)

-- Definitions of the given conditions
def yellow_fraction := (1 : ℚ) / 4
def purple_fraction := (1 : ℚ) / 3
def orange_fraction := (1 : ℚ) / 6
def black_ribbons := 40
def black_fraction := (1 : ℚ) / 4

-- Using the given and derived conditions
theorem orange_ribbons_count
  (hy : yellow_fraction = 1 / 4)
  (hp : purple_fraction = 1 / 3)
  (ho : orange_fraction = 1 / 6)
  (hb : black_ribbons = 40)
  (hbf : black_fraction = 1 / 4)
  (total_eq : total_ribbons = black_ribbons * 4) :
  orange_ribbons = total_ribbons * orange_fraction := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_orange_ribbons_count_l928_92868


namespace NUMINAMATH_GPT_average_of_four_l928_92842

variable {r s t u : ℝ}

theorem average_of_four (h : (5 / 2) * (r + s + t + u) = 20) : (r + s + t + u) / 4 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_average_of_four_l928_92842


namespace NUMINAMATH_GPT_Vanya_433_sum_l928_92861

theorem Vanya_433_sum : 
  ∃ (A B : ℕ), 
  A + B = 91 
  ∧ (3 * A + 7 * B = 433) 
  ∧ (∃ (subsetA subsetB : Finset ℕ),
      (∀ x ∈ subsetA, x ∈ Finset.range (13 + 1))
      ∧ (∀ x ∈ subsetB, x ∈ Finset.range (13 + 1))
      ∧ subsetA ∩ subsetB = ∅
      ∧ subsetA ∪ subsetB = Finset.range (13 + 1)
      ∧ subsetA.card = 5
      ∧ subsetA.sum id = A
      ∧ subsetB.sum id = B) :=
by
  sorry

end NUMINAMATH_GPT_Vanya_433_sum_l928_92861


namespace NUMINAMATH_GPT_determine_exponent_l928_92895

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x ^ a

theorem determine_exponent (a : ℝ) (hf : power_function a 4 = 8) : power_function (3/2) = power_function a := by
  sorry

end NUMINAMATH_GPT_determine_exponent_l928_92895


namespace NUMINAMATH_GPT_time_for_nth_mile_l928_92891

noncomputable def speed (k : ℝ) (d : ℝ) : ℝ := k / (d * d)

noncomputable def time_for_mile (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then 2
  else 2 * (n - 1) * (n - 1)

theorem time_for_nth_mile (n : ℕ) (h₁ : ∀ d : ℝ, d ≥ 1 → speed (1/2) d = 1 / (2 * d * d))
  (h₂ : time_for_mile 1 = 1)
  (h₃ : time_for_mile 2 = 2) :
  time_for_mile n = 2 * (n - 1) * (n - 1) := sorry

end NUMINAMATH_GPT_time_for_nth_mile_l928_92891


namespace NUMINAMATH_GPT_simple_interest_l928_92858

/-- Given:
    - Principal (P) = Rs. 80325
    - Rate (R) = 1% per annum
    - Time (T) = 5 years
    Prove that the total simple interest earned (SI) is Rs. 4016.25.
-/
theorem simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ)
  (hP : P = 80325)
  (hR : R = 1)
  (hT : T = 5)
  (hSI : SI = P * R * T / 100) :
  SI = 4016.25 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_l928_92858


namespace NUMINAMATH_GPT_determinant_matrix_3x3_l928_92816

theorem determinant_matrix_3x3 :
  Matrix.det ![![3, 1, -2], ![8, 5, -4], ![1, 3, 6]] = 140 :=
by
  sorry

end NUMINAMATH_GPT_determinant_matrix_3x3_l928_92816


namespace NUMINAMATH_GPT_plates_not_adj_l928_92826

def num_ways_arrange_plates (blue red green orange : ℕ) (no_adj : Bool) : ℕ :=
  -- assuming this function calculates the desired number of arrangements
  sorry

theorem plates_not_adj (h : num_ways_arrange_plates 6 2 2 1 true = 1568) : 
  num_ways_arrange_plates 6 2 2 1 true = 1568 :=
  by exact h -- using the hypothesis directly for the theorem statement

end NUMINAMATH_GPT_plates_not_adj_l928_92826


namespace NUMINAMATH_GPT_petya_wrong_l928_92879

theorem petya_wrong : ∃ (a b : ℕ), b^2 ∣ a^5 ∧ ¬ (b ∣ a^2) :=
by
  use 4
  use 32
  sorry

end NUMINAMATH_GPT_petya_wrong_l928_92879


namespace NUMINAMATH_GPT_nearest_edge_of_picture_l928_92856

theorem nearest_edge_of_picture
    (wall_width : ℝ) (picture_width : ℝ) (offset : ℝ) (x : ℝ)
    (hw : wall_width = 25) (hp : picture_width = 5) (ho : offset = 2) :
    x + (picture_width / 2) + offset = wall_width / 2 →
    x = 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_nearest_edge_of_picture_l928_92856


namespace NUMINAMATH_GPT_units_digit_of_sum_is_three_l928_92814

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_of_factorials : ℕ :=
  (List.range 10).map factorial |>.sum

def power_of_ten (n : ℕ) : ℕ :=
  10^n

theorem units_digit_of_sum_is_three : 
  units_digit (sum_of_factorials + power_of_ten 3) = 3 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_sum_is_three_l928_92814


namespace NUMINAMATH_GPT_female_voters_percentage_is_correct_l928_92896

def percentage_of_population_that_are_female_voters
  (female_percentage : ℝ)
  (voter_percentage_of_females : ℝ) : ℝ :=
  female_percentage * voter_percentage_of_females * 100

theorem female_voters_percentage_is_correct :
  percentage_of_population_that_are_female_voters 0.52 0.4 = 20.8 := by
  sorry

end NUMINAMATH_GPT_female_voters_percentage_is_correct_l928_92896


namespace NUMINAMATH_GPT_quadrilateral_trapezium_l928_92876

theorem quadrilateral_trapezium (a b c d : ℝ) 
  (h1 : a / 6 = b / 7) 
  (h2 : b / 7 = c / 8) 
  (h3 : c / 8 = d / 9) 
  (h4 : a + b + c + d = 360) : 
  ((a + c = 180) ∨ (b + d = 180)) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_trapezium_l928_92876


namespace NUMINAMATH_GPT_round_155_628_l928_92874

theorem round_155_628 :
  round (155.628 : Real) = 156 := by
  sorry

end NUMINAMATH_GPT_round_155_628_l928_92874


namespace NUMINAMATH_GPT_sqrt_factorial_product_l928_92860

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_factorial_product_l928_92860


namespace NUMINAMATH_GPT_power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l928_92854

theorem power_function_condition (m : ℝ) : m^2 + 2 * m = 1 ↔ m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2 :=
by sorry

theorem direct_proportionality_condition (m : ℝ) : (m^2 + m - 1 = 1 ∧ m^2 + 3 * m ≠ 0) ↔ m = 1 :=
by sorry

theorem inverse_proportionality_condition (m : ℝ) : (m^2 + m - 1 = -1 ∧ m^2 + 3 * m ≠ 0) ↔ m = -1 :=
by sorry

end NUMINAMATH_GPT_power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l928_92854


namespace NUMINAMATH_GPT_expected_number_of_heads_after_flips_l928_92836

theorem expected_number_of_heads_after_flips :
  let p_heads_after_tosses : ℚ := (1/3) + (2/9) + (4/27) + (8/81)
  let expected_heads : ℚ := 100 * p_heads_after_tosses
  expected_heads = 6500 / 81 :=
by
  let p_heads_after_tosses : ℚ := (1/3) + (2/9) + (4/27) + (8/81)
  let expected_heads : ℚ := 100 * p_heads_after_tosses
  show expected_heads = (6500 / 81)
  sorry

end NUMINAMATH_GPT_expected_number_of_heads_after_flips_l928_92836


namespace NUMINAMATH_GPT_payment_amount_l928_92832

/-- 
A certain debt will be paid in 52 installments from January 1 to December 31 of a certain year.
Each of the first 25 payments is to be a certain amount; each of the remaining payments is to be $100 more than each of the first payments.
The average (arithmetic mean) payment that will be made on the debt for the year is $551.9230769230769.
Prove that the amount of each of the first 25 payments is $500.
-/
theorem payment_amount (X : ℝ) 
  (h1 : 25 * X + 27 * (X + 100) = 52 * 551.9230769230769) :
  X = 500 :=
sorry

end NUMINAMATH_GPT_payment_amount_l928_92832


namespace NUMINAMATH_GPT_polyhedron_equation_l928_92807

variables (V E F H T : ℕ)

-- Euler's formula for convex polyhedra
axiom euler_formula : V - E + F = 2
-- Number of faces is 50, and each face is either a triangle or a hexagon
axiom faces_count : F = 50
-- At each vertex, 3 triangles and 2 hexagons meet
axiom triangles_meeting : T = 3
axiom hexagons_meeting : H = 2

-- Prove that 100H + 10T + V = 230
theorem polyhedron_equation : 100 * H + 10 * T + V = 230 :=
  sorry

end NUMINAMATH_GPT_polyhedron_equation_l928_92807


namespace NUMINAMATH_GPT_ice_cream_amount_l928_92812

/-- Given: 
    Amount of ice cream eaten on Friday night: 3.25 pints
    Total amount of ice cream eaten over both nights: 3.5 pints
    Prove: 
    Amount of ice cream eaten on Saturday night = 0.25 pints -/
theorem ice_cream_amount (friday_night saturday_night total : ℝ) (h_friday : friday_night = 3.25) (h_total : total = 3.5) : 
  saturday_night = total - friday_night → saturday_night = 0.25 :=
by
  intro h
  rw [h_total, h_friday] at h
  simp [h]
  sorry

end NUMINAMATH_GPT_ice_cream_amount_l928_92812


namespace NUMINAMATH_GPT_slope_of_parallel_line_l928_92898

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l928_92898


namespace NUMINAMATH_GPT_greatest_negative_root_l928_92841

noncomputable def sine (x : ℝ) : ℝ := Real.sin (Real.pi * x)
noncomputable def cosine (x : ℝ) : ℝ := Real.cos (2 * Real.pi * x)

theorem greatest_negative_root :
  ∀ (x : ℝ), (x < 0 ∧ (sine x - cosine x) / ((sine x + 1)^2 + (Real.cos (Real.pi * x))^2) = 0) → 
    x ≤ -7/6 :=
by
  sorry

end NUMINAMATH_GPT_greatest_negative_root_l928_92841


namespace NUMINAMATH_GPT_B_contribution_l928_92815

theorem B_contribution (A_capital : ℝ) (A_time : ℝ) (B_time : ℝ) (total_profit : ℝ) (A_profit_share : ℝ) (B_contributed : ℝ) :
  A_capital * A_time / (A_capital * A_time + B_contributed * B_time) = A_profit_share / total_profit →
  B_contributed = 6000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_B_contribution_l928_92815


namespace NUMINAMATH_GPT_proof_problem_l928_92819

def a : ℕ := 5^2
def b : ℕ := a^4

theorem proof_problem : b = 390625 := 
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l928_92819


namespace NUMINAMATH_GPT_herring_invariant_l928_92850

/--
A circle is divided into six sectors. Each sector contains one herring. 
In one move, you can move any two herrings in adjacent sectors moving them in opposite directions.
Prove that it is impossible to gather all herrings into one sector using these operations.
-/
theorem herring_invariant (herring : Fin 6 → Bool) :
  ¬ ∃ i : Fin 6, ∀ j : Fin 6, herring j = herring i := 
sorry

end NUMINAMATH_GPT_herring_invariant_l928_92850


namespace NUMINAMATH_GPT_probability_prime_or_odd_ball_l928_92823

def isPrime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isPrimeOrOdd (n : ℕ) : Prop :=
  isPrime n ∨ isOdd n

theorem probability_prime_or_odd_ball :
  (1+2+3+5+7)/8 = 5/8 := by
  sorry

end NUMINAMATH_GPT_probability_prime_or_odd_ball_l928_92823


namespace NUMINAMATH_GPT_evaluate_expression_l928_92828

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : (6 * a^2 - 8 * a + 3) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l928_92828


namespace NUMINAMATH_GPT_sum_series_eq_260_l928_92873

theorem sum_series_eq_260 : (2 + 12 + 22 + 32 + 42) + (10 + 20 + 30 + 40 + 50) = 260 := by
  sorry

end NUMINAMATH_GPT_sum_series_eq_260_l928_92873


namespace NUMINAMATH_GPT_DanAgeIs12_l928_92848

def DanPresentAge (x : ℕ) : Prop :=
  (x + 18 = 5 * (x - 6))

theorem DanAgeIs12 : ∃ x : ℕ, DanPresentAge x ∧ x = 12 :=
by
  use 12
  unfold DanPresentAge
  sorry

end NUMINAMATH_GPT_DanAgeIs12_l928_92848


namespace NUMINAMATH_GPT_hens_on_farm_l928_92897

theorem hens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H + R = 75) : H = 67 :=
by
  sorry

end NUMINAMATH_GPT_hens_on_farm_l928_92897


namespace NUMINAMATH_GPT_deductive_reasoning_correctness_l928_92847

theorem deductive_reasoning_correctness (major_premise minor_premise form_of_reasoning correct : Prop) 
  (h : major_premise ∧ minor_premise ∧ form_of_reasoning) : correct :=
  sorry

end NUMINAMATH_GPT_deductive_reasoning_correctness_l928_92847


namespace NUMINAMATH_GPT_tailoring_business_days_l928_92831

theorem tailoring_business_days
  (shirts_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (pants_per_day : ℕ)
  (fabric_per_pant : ℕ)
  (total_fabric : ℕ)
  (h1 : shirts_per_day = 3)
  (h2 : fabric_per_shirt = 2)
  (h3 : pants_per_day = 5)
  (h4 : fabric_per_pant = 5)
  (h5 : total_fabric = 93) :
  (total_fabric / (shirts_per_day * fabric_per_shirt + pants_per_day * fabric_per_pant)) = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_tailoring_business_days_l928_92831


namespace NUMINAMATH_GPT_solution1_solution2_l928_92883

open Complex

noncomputable def problem1 : Prop := 
  ((3 - I) / (1 + I)) ^ 2 = -3 - 4 * I

noncomputable def problem2 (z : ℂ) : Prop := 
  z = 1 + I → (2 / z - z = -2 * I)

theorem solution1 : problem1 := 
  by sorry

theorem solution2 : problem2 (1 + I) :=
  by sorry

end NUMINAMATH_GPT_solution1_solution2_l928_92883


namespace NUMINAMATH_GPT_smallest_m_for_integral_solutions_l928_92890

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), (∀ (p q : ℤ), (15 * (p * p) - m * p + 630 = 0 ∧ 15 * (q * q) - m * q + 630 = 0) → (m = 195)) :=
sorry

end NUMINAMATH_GPT_smallest_m_for_integral_solutions_l928_92890


namespace NUMINAMATH_GPT_area_of_ABCD_l928_92806

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end NUMINAMATH_GPT_area_of_ABCD_l928_92806


namespace NUMINAMATH_GPT_find_a_l928_92845

def operation (a b : ℤ) : ℤ := 2 * a - b * b

theorem find_a (a : ℤ) : operation a 3 = 15 → a = 12 := by
  sorry

end NUMINAMATH_GPT_find_a_l928_92845


namespace NUMINAMATH_GPT_geometric_sequence_a3_l928_92809

theorem geometric_sequence_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 1)
  (h5 : a 5 = 4)
  (geo_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l928_92809


namespace NUMINAMATH_GPT_product_sum_correct_l928_92827

def product_sum_eq : Prop :=
  let a := 4 * 10^6
  let b := 8 * 10^6
  (a * b + 2 * 10^13) = 5.2 * 10^13

theorem product_sum_correct : product_sum_eq :=
by
  sorry

end NUMINAMATH_GPT_product_sum_correct_l928_92827


namespace NUMINAMATH_GPT_max_b_c_value_l928_92875

theorem max_b_c_value (a b c : ℕ) (h1 : a > b) (h2 : a + b = 18) (h3 : c - b = 2) : b + c = 18 :=
sorry

end NUMINAMATH_GPT_max_b_c_value_l928_92875


namespace NUMINAMATH_GPT_even_marked_squares_9x9_l928_92808

open Nat

theorem even_marked_squares_9x9 :
  let n := 9
  let total_squares := n * n
  let odd_rows_columns := [1, 3, 5, 7, 9]
  let odd_squares := odd_rows_columns.length * odd_rows_columns.length
  total_squares - odd_squares = 56 :=
by
  sorry

end NUMINAMATH_GPT_even_marked_squares_9x9_l928_92808


namespace NUMINAMATH_GPT_crayons_per_pack_l928_92880

theorem crayons_per_pack (total_crayons : ℕ) (num_packs : ℕ) (crayons_per_pack : ℕ) 
  (h1 : total_crayons = 615) (h2 : num_packs = 41) : crayons_per_pack = 15 := by
sorry

end NUMINAMATH_GPT_crayons_per_pack_l928_92880


namespace NUMINAMATH_GPT_alex_needs_packs_of_buns_l928_92864

-- Definitions (conditions)
def guests : ℕ := 10
def burgers_per_guest : ℕ := 3
def meat_eating_guests : ℕ := guests - 1
def bread_eating_ratios : ℕ := meat_eating_guests - 1
def buns_per_pack : ℕ := 8

-- Theorem (question == answer)
theorem alex_needs_packs_of_buns : 
  (burgers_per_guest * meat_eating_guests - burgers_per_guest) / buns_per_pack = 3 := by
  sorry

end NUMINAMATH_GPT_alex_needs_packs_of_buns_l928_92864


namespace NUMINAMATH_GPT_smallest_denominator_fraction_l928_92839

theorem smallest_denominator_fraction 
  (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (h1 : 99 / 100 < p / q) 
  (h2 : p / q < 100 / 101) :
  p = 199 ∧ q = 201 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_denominator_fraction_l928_92839


namespace NUMINAMATH_GPT_find_a_for_symmetric_and_parallel_lines_l928_92818

theorem find_a_for_symmetric_and_parallel_lines :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * x + 3 ↔ x = a * y + 3) ∧ (∀ (x y : ℝ), x + 2 * y - 1 = 0 ↔ x = a * y + 3) ∧ ∃ (a : ℝ), a = -2 := 
sorry

end NUMINAMATH_GPT_find_a_for_symmetric_and_parallel_lines_l928_92818


namespace NUMINAMATH_GPT_taxi_charge_l928_92881

theorem taxi_charge :
  ∀ (initial_fee additional_charge_per_segment total_distance total_charge : ℝ),
  initial_fee = 2.05 →
  total_distance = 3.6 →
  total_charge = 5.20 →
  (total_charge - initial_fee) / (5/2 * total_distance) = 0.35 :=
by
  intros initial_fee additional_charge_per_segment total_distance total_charge
  intros h_initial_fee h_total_distance h_total_charge
  -- Proof here
  sorry

end NUMINAMATH_GPT_taxi_charge_l928_92881


namespace NUMINAMATH_GPT_inverse_44_mod_53_l928_92878

theorem inverse_44_mod_53 : (44 * 22) % 53 = 1 :=
by
-- Given condition: 19's inverse modulo 53 is 31
have h: (19 * 31) % 53 = 1 := by sorry
-- We should prove the required statement using the given condition.
sorry

end NUMINAMATH_GPT_inverse_44_mod_53_l928_92878


namespace NUMINAMATH_GPT_multiple_of_5_multiple_of_10_not_multiple_of_20_not_multiple_of_40_l928_92869

def x : ℤ := 50 + 100 + 140 + 180 + 320 + 400 + 5000

theorem multiple_of_5 : x % 5 = 0 := by 
  sorry

theorem multiple_of_10 : x % 10 = 0 := by 
  sorry

theorem not_multiple_of_20 : x % 20 ≠ 0 := by 
  sorry

theorem not_multiple_of_40 : x % 40 ≠ 0 := by 
  sorry

end NUMINAMATH_GPT_multiple_of_5_multiple_of_10_not_multiple_of_20_not_multiple_of_40_l928_92869


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l928_92801

theorem problem_part_1 (a b : ℝ) (h1 : a * 1^2 - 3 * 1 + 2 = 0) (h2 : a * b^2 - 3 * b + 2 = 0) (h3 : 1 + b = 3 / a) (h4 : 1 * b = 2 / a) : a = 1 ∧ b = 2 :=
sorry

theorem problem_part_2 (m : ℝ) (h5 : a = 1) (h6 : b = 2) : 
  (m = 2 → ∀ x, ¬ (x^2 - (m + 2) * x + 2 * m < 0)) ∧
  (m < 2 → ∀ x, x ∈ Set.Ioo m 2 ↔ x^2 - (m + 2) * x + 2 * m < 0) ∧
  (m > 2 → ∀ x, x ∈ Set.Ioo 2 m ↔ x^2 - (m + 2) * x + 2 * m < 0) :=
sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l928_92801


namespace NUMINAMATH_GPT_find_m_value_l928_92859

noncomputable def is_solution (p q m : ℝ) : Prop :=
  ∃ x : ℝ, (x^2 + p*x + q = 0) ∧ (x^2 - m*x + m^2 - 19 = 0)

theorem find_m_value :
  let A := { x : ℝ | x^2 + 2 * x - 8 = 0 }
  let B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
  ∀ (C : ℝ → Prop), 
  (∃ x, B x ∧ C x) ∧ (¬ ∃ x, A x ∧ C x) → 
  (∃ m, C = { x : ℝ | x^2 - m * x + m^2 - 19 = 0 } ∧ m = -2) :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l928_92859


namespace NUMINAMATH_GPT_factorize_problem1_factorize_problem2_l928_92821

-- Problem 1: Prove that 6p^3q - 10p^2 == 2p^2 * (3pq - 5)
theorem factorize_problem1 (p q : ℝ) : 
    6 * p^3 * q - 10 * p^2 = 2 * p^2 * (3 * p * q - 5) := 
by 
    sorry

-- Problem 2: Prove that a^4 - 8a^2 + 16 == (a-2)^2 * (a+2)^2
theorem factorize_problem2 (a : ℝ) : 
    a^4 - 8 * a^2 + 16 = (a - 2)^2 * (a + 2)^2 := 
by 
    sorry

end NUMINAMATH_GPT_factorize_problem1_factorize_problem2_l928_92821


namespace NUMINAMATH_GPT_solve_inequality_l928_92824

theorem solve_inequality (x : ℝ) :
  (x - 5) / (x - 3)^2 < 0 ↔ x ∈ (Set.Iio 3 ∪ Set.Ioo 3 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l928_92824


namespace NUMINAMATH_GPT_adult_tickets_count_l928_92825

theorem adult_tickets_count (A C : ℕ) (h1 : A + C = 7) (h2 : 21 * A + 14 * C = 119) : A = 3 :=
sorry

end NUMINAMATH_GPT_adult_tickets_count_l928_92825


namespace NUMINAMATH_GPT_clock_ticks_12_times_l928_92865

theorem clock_ticks_12_times (t1 t2 : ℕ) (d1 d2 : ℕ) (h1 : t1 = 6) (h2 : d1 = 40) (h3 : d2 = 88) : t2 = 12 := by
  sorry

end NUMINAMATH_GPT_clock_ticks_12_times_l928_92865


namespace NUMINAMATH_GPT_compute_expression_l928_92833

section
variable (a : ℝ)

theorem compute_expression :
  (-a^2)^3 * a^3 = -a^9 :=
sorry
end

end NUMINAMATH_GPT_compute_expression_l928_92833


namespace NUMINAMATH_GPT_g_13_equals_236_l928_92894

def g (n : ℕ) : ℕ := n^2 + 2 * n + 41

theorem g_13_equals_236 : g 13 = 236 := sorry

end NUMINAMATH_GPT_g_13_equals_236_l928_92894


namespace NUMINAMATH_GPT_tan_of_cos_alpha_l928_92853

open Real

theorem tan_of_cos_alpha (α : ℝ) (h1 : cos α = 3 / 5) (h2 : -π < α ∧ α < 0) : tan α = -4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_of_cos_alpha_l928_92853


namespace NUMINAMATH_GPT_labor_cost_per_hour_l928_92805

theorem labor_cost_per_hour (total_repair_cost part_cost labor_hours : ℕ)
    (h1 : total_repair_cost = 2400)
    (h2 : part_cost = 1200)
    (h3 : labor_hours = 16) :
    (total_repair_cost - part_cost) / labor_hours = 75 := by
  sorry

end NUMINAMATH_GPT_labor_cost_per_hour_l928_92805


namespace NUMINAMATH_GPT_part_I_solution_set_part_II_range_of_a_l928_92830

-- Definitions
def f (x : ℝ) (a : ℝ) := |x - 1| + |a * x + 1|
def g (x : ℝ) := |x + 1| + 2

-- Part I: Prove the solution set of the inequality f(x) < 2 when a = 1/2
theorem part_I_solution_set (x : ℝ) : f x (1/2 : ℝ) < 2 ↔ 0 < x ∧ x < (4/3 : ℝ) :=
sorry
  
-- Part II: Prove the range of a such that (0, 1] ⊆ {x | f x a ≤ g x}
theorem part_II_range_of_a (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 1 → f x a ≤ g x) ↔ -5 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_GPT_part_I_solution_set_part_II_range_of_a_l928_92830


namespace NUMINAMATH_GPT_faster_current_takes_more_time_l928_92893

theorem faster_current_takes_more_time (v v1 v2 S : ℝ) (h_v1_gt_v2 : v1 > v2) :
  let t1 := (2 * S * v) / (v^2 - v1^2)
  let t2 := (2 * S * v) / (v^2 - v2^2)
  t1 > t2 :=
by
  sorry

end NUMINAMATH_GPT_faster_current_takes_more_time_l928_92893


namespace NUMINAMATH_GPT_total_students_l928_92817

theorem total_students (m d : ℕ) 
  (H1: 30 < m + d ∧ m + d < 40)
  (H2: ∃ r, r = 3 * m ∧ r = 5 * d) : 
  m + d = 32 := 
by
  sorry

end NUMINAMATH_GPT_total_students_l928_92817


namespace NUMINAMATH_GPT_sum_of_powers_twice_square_l928_92837

theorem sum_of_powers_twice_square (x y : ℤ) : 
  ∃ z : ℤ, x^4 + y^4 + (x + y)^4 = 2 * z^2 := by
  let z := x^2 + x * y + y^2
  use z
  sorry

end NUMINAMATH_GPT_sum_of_powers_twice_square_l928_92837


namespace NUMINAMATH_GPT_second_trial_addition_amount_l928_92834

variable (optimal_min optimal_max: ℝ) (phi: ℝ)

def method_618 (optimal_min optimal_max phi: ℝ) :=
  let x1 := optimal_min + (optimal_max - optimal_min) * phi
  let x2 := optimal_max + optimal_min - x1
  x2

theorem second_trial_addition_amount:
  optimal_min = 10 ∧ optimal_max = 110 ∧ phi = 0.618 →
  method_618 10 110 0.618 = 48.2 :=
by
  intro h
  simp [method_618, h]
  sorry

end NUMINAMATH_GPT_second_trial_addition_amount_l928_92834


namespace NUMINAMATH_GPT_green_apples_more_than_red_apples_l928_92820

theorem green_apples_more_than_red_apples 
    (total_apples : ℕ)
    (red_apples : ℕ)
    (total_apples_eq : total_apples = 44)
    (red_apples_eq : red_apples = 16) :
    (total_apples - red_apples) - red_apples = 12 :=
by
  sorry

end NUMINAMATH_GPT_green_apples_more_than_red_apples_l928_92820


namespace NUMINAMATH_GPT_first_machine_copies_per_minute_l928_92844

theorem first_machine_copies_per_minute
    (x : ℕ)
    (h1 : ∀ (x : ℕ), 30 * x + 30 * 55 = 2850) :
  x = 40 :=
by
  sorry

end NUMINAMATH_GPT_first_machine_copies_per_minute_l928_92844


namespace NUMINAMATH_GPT_abs_neg_five_l928_92867

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_five_l928_92867


namespace NUMINAMATH_GPT_smallest_prime_factor_of_1917_l928_92899

theorem smallest_prime_factor_of_1917 : ∃ p : ℕ, Prime p ∧ (p ∣ 1917) ∧ (∀ q : ℕ, Prime q ∧ (q ∣ 1917) → q ≥ p) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_factor_of_1917_l928_92899


namespace NUMINAMATH_GPT_find_m_l928_92887

-- Definitions for the system of equations and the condition
def system_of_equations (x y m : ℝ) :=
  2 * x + 6 * y = 25 ∧ 6 * x + 2 * y = -11 ∧ x - y = m - 1

-- Statement to prove
theorem find_m (x y m : ℝ) (h : system_of_equations x y m) : m = -8 :=
  sorry

end NUMINAMATH_GPT_find_m_l928_92887


namespace NUMINAMATH_GPT_perpendicular_lines_m_l928_92863

theorem perpendicular_lines_m (m : ℝ) : 
    (∀ x y : ℝ, x - 2 * y + 5 = 0 → 
                2 * x + m * y - 6 = 0 → 
                (1 / 2) * (-2 / m) = -1) → 
    m = 1 :=
by
  intros
  -- proof goes here
  sorry

end NUMINAMATH_GPT_perpendicular_lines_m_l928_92863


namespace NUMINAMATH_GPT_proof_problem_l928_92857

-- Define the operation
def star (a b : ℝ) : ℝ := (a - b) ^ 2

-- The proof problem as a Lean statement
theorem proof_problem (x y : ℝ) : star ((x - y) ^ 2 + 1) ((y - x) ^ 2 + 1) = 0 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l928_92857


namespace NUMINAMATH_GPT_zamena_inequalities_l928_92855

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end NUMINAMATH_GPT_zamena_inequalities_l928_92855


namespace NUMINAMATH_GPT_only_C_forms_triangle_l928_92803

def triangle_sides (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_C_forms_triangle :
  ¬ triangle_sides 3 4 8 ∧
  ¬ triangle_sides 2 5 2 ∧
  triangle_sides 3 5 6 ∧
  ¬ triangle_sides 5 6 11 :=
by
  sorry

end NUMINAMATH_GPT_only_C_forms_triangle_l928_92803


namespace NUMINAMATH_GPT_value_of_a_l928_92835

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem value_of_a : ∃ a : ℤ, star a 3 = 63 ∧ a = 30 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l928_92835


namespace NUMINAMATH_GPT_abs_difference_equality_l928_92889

theorem abs_difference_equality : (abs (3 - Real.sqrt 2) - abs (Real.sqrt 2 - 2) = 1) :=
  by
    -- Define our conditions as hypotheses
    have h1 : 3 > Real.sqrt 2 := sorry
    have h2 : Real.sqrt 2 < 2 := sorry
    -- The proof itself is skipped in this step
    sorry

end NUMINAMATH_GPT_abs_difference_equality_l928_92889


namespace NUMINAMATH_GPT_binom_28_7_l928_92811

theorem binom_28_7 (h1 : Nat.choose 26 3 = 2600) (h2 : Nat.choose 26 4 = 14950) (h3 : Nat.choose 26 5 = 65780) : 
  Nat.choose 28 7 = 197340 :=
by
  sorry

end NUMINAMATH_GPT_binom_28_7_l928_92811


namespace NUMINAMATH_GPT_greatest_possible_y_l928_92877

theorem greatest_possible_y (y : ℕ) (h1 : (y^4 / y^2) < 18) : y ≤ 4 := 
  sorry -- Proof to be filled in later

end NUMINAMATH_GPT_greatest_possible_y_l928_92877


namespace NUMINAMATH_GPT_joel_average_speed_l928_92870

theorem joel_average_speed :
  let start_time := (8, 50)
  let end_time := (14, 35)
  let total_distance := 234
  let total_time := (14 - 8) + (35 - 50) / 60
  ∀ start_time end_time total_distance,
    (start_time = (8, 50)) →
    (end_time = (14, 35)) →
    total_distance = 234 →
    (total_time = (14 - 8) + (35 - 50) / 60) →
    total_distance / total_time = 41 :=
by
  sorry

end NUMINAMATH_GPT_joel_average_speed_l928_92870


namespace NUMINAMATH_GPT_solve_inequality_l928_92884

theorem solve_inequality (x : ℝ) :
  (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 6) ↔ (2 < x) ∧ (x < 3) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l928_92884


namespace NUMINAMATH_GPT_product_of_last_two_digits_l928_92851

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 14) (h2 : B = 0 ∨ B = 5) : A * B = 45 :=
sorry

end NUMINAMATH_GPT_product_of_last_two_digits_l928_92851
