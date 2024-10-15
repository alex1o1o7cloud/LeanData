import Mathlib

namespace NUMINAMATH_GPT_trick_proof_l1919_191938

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end NUMINAMATH_GPT_trick_proof_l1919_191938


namespace NUMINAMATH_GPT_ashley_percentage_secured_l1919_191927

noncomputable def marks_secured : ℕ := 332
noncomputable def max_marks : ℕ := 400
noncomputable def percentage_secured : ℕ := (marks_secured * 100) / max_marks

theorem ashley_percentage_secured 
    (h₁ : marks_secured = 332)
    (h₂ : max_marks = 400) :
    percentage_secured = 83 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ashley_percentage_secured_l1919_191927


namespace NUMINAMATH_GPT_reduced_price_per_kg_l1919_191907

-- Define the conditions
def reduction_factor : ℝ := 0.80
def extra_kg : ℝ := 4
def total_cost : ℝ := 684

-- Assume the original price P and reduced price R
variables (P R : ℝ)

-- Define the equations derived from the conditions
def original_cost_eq := (P * 16 = total_cost)
def reduced_cost_eq := (0.80 * P * (16 + extra_kg) = total_cost)

-- The final theorem stating the reduced price per kg of oil is 34.20 Rs
theorem reduced_price_per_kg : R = 34.20 :=
by
  have h1: P * 16 = total_cost := sorry -- This will establish the original cost
  have h2: 0.80 * P * (16 + extra_kg) = total_cost := sorry -- This will establish the reduced cost
  have Q: 16 = 16 := sorry -- Calculation of Q (original quantity)
  have h3: P = 42.75 := sorry -- Calculation of original price
  have h4: R = 0.80 * P := sorry -- Calculation of reduced price
  have h5: R = 34.20 := sorry -- Final calculation matching the required answer
  exact h5

end NUMINAMATH_GPT_reduced_price_per_kg_l1919_191907


namespace NUMINAMATH_GPT_trains_cross_in_9_seconds_l1919_191950

noncomputable def time_to_cross (length1 length2 : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (speed1 + speed2)

theorem trains_cross_in_9_seconds :
  time_to_cross 240 260.04 (120 * (5 / 18)) (80 * (5 / 18)) = 9 := 
by
  sorry

end NUMINAMATH_GPT_trains_cross_in_9_seconds_l1919_191950


namespace NUMINAMATH_GPT_curve_B_is_not_good_l1919_191943

-- Define the points A and B
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define the condition for being a "good curve"
def is_good_curve (C : ℝ × ℝ → Prop) : Prop :=
  ∃ M : ℝ × ℝ, C M ∧ abs (dist M A - dist M B) = 8

-- Define the curves
def curve_A (p : ℝ × ℝ) : Prop := p.1 + p.2 = 5
def curve_B (p : ℝ × ℝ) : Prop := p.1 ^ 2 + p.2 ^ 2 = 9
def curve_C (p : ℝ × ℝ) : Prop := (p.1 ^ 2) / 25 + (p.2 ^ 2) / 9 = 1
def curve_D (p : ℝ × ℝ) : Prop := p.1 ^ 2 = 16 * p.2

-- Prove that curve_B is not a "good curve"
theorem curve_B_is_not_good : ¬ is_good_curve curve_B := by
  sorry

end NUMINAMATH_GPT_curve_B_is_not_good_l1919_191943


namespace NUMINAMATH_GPT_part_one_part_two_l1919_191964

-- Part (1)
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

-- Part (2)
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : 
  2 * a + b = 8 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1919_191964


namespace NUMINAMATH_GPT_three_zeros_condition_l1919_191973

noncomputable def f (ω : ℝ) (x : ℝ) := Real.sin (ω * x) + Real.cos (ω * x)

theorem three_zeros_condition (ω : ℝ) (hω : ω > 0) :
  (∃ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ 2 * Real.pi ∧
  f ω x1 = 0 ∧ f ω x2 = 0 ∧ f ω x3 = 0) →
  (∀ ω, (11 / 8 : ℝ) ≤ ω ∧ ω < (15 / 8 : ℝ) ∧
  (∀ x, f ω x = 0 ↔ x = (5 * Real.pi) / (4 * ω))) :=
sorry

end NUMINAMATH_GPT_three_zeros_condition_l1919_191973


namespace NUMINAMATH_GPT_tan_seven_pi_over_six_l1919_191936
  
theorem tan_seven_pi_over_six :
  Real.tan (7 * Real.pi / 6) = 1 / Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_seven_pi_over_six_l1919_191936


namespace NUMINAMATH_GPT_tan_angle_sum_l1919_191929

theorem tan_angle_sum
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
by
  sorry

end NUMINAMATH_GPT_tan_angle_sum_l1919_191929


namespace NUMINAMATH_GPT_students_not_coming_l1919_191963

-- Define the conditions
def pieces_per_student : ℕ := 4
def pieces_made_last_monday : ℕ := 40
def pieces_made_upcoming_monday : ℕ := 28

-- Define the number of students not coming to class
theorem students_not_coming :
  (pieces_made_last_monday / pieces_per_student) - 
  (pieces_made_upcoming_monday / pieces_per_student) = 3 :=
by sorry

end NUMINAMATH_GPT_students_not_coming_l1919_191963


namespace NUMINAMATH_GPT_uncle_ben_eggs_l1919_191982

noncomputable def total_eggs (total_chickens : ℕ) (roosters : ℕ) (non_egg_laying_hens : ℕ) (eggs_per_hen : ℕ) : ℕ :=
  let total_hens := total_chickens - roosters
  let egg_laying_hens := total_hens - non_egg_laying_hens
  egg_laying_hens * eggs_per_hen

theorem uncle_ben_eggs :
  total_eggs 440 39 15 3 = 1158 :=
by
  unfold total_eggs
  -- Correct steps to prove the theorem can be skipped with sorry
  sorry

end NUMINAMATH_GPT_uncle_ben_eggs_l1919_191982


namespace NUMINAMATH_GPT_speed_ratio_l1919_191947

variable (v1 v2 : ℝ) -- Speeds of A and B respectively
variable (dA dB : ℝ) -- Distances to destinations A and B respectively

-- Conditions:
-- 1. Both reach their destinations in 1 hour
def condition_1 : Prop := dA = v1 ∧ dB = v2

-- 2. When they swap destinations, A takes 35 minutes more to reach B's destination
def condition_2 : Prop := dB / v1 = dA / v2 + 35 / 60

-- Given these conditions, prove that the ratio of v1 to v2 is 3
theorem speed_ratio (h1 : condition_1 v1 v2 dA dB) (h2 : condition_2 v1 v2 dA dB) : v1 = 3 * v2 :=
sorry

end NUMINAMATH_GPT_speed_ratio_l1919_191947


namespace NUMINAMATH_GPT_expression_evaluation_l1919_191941

theorem expression_evaluation : |1 - Real.sqrt 3| + 2 * Real.cos (Real.pi / 6) - Real.sqrt 12 - 2023 = -2024 := 
by {
    sorry
}

end NUMINAMATH_GPT_expression_evaluation_l1919_191941


namespace NUMINAMATH_GPT_Mirella_read_purple_books_l1919_191971

theorem Mirella_read_purple_books (P : ℕ) 
  (pages_per_purple_book : ℕ := 230)
  (pages_per_orange_book : ℕ := 510)
  (orange_books_read : ℕ := 4)
  (extra_orange_pages : ℕ := 890)
  (total_orange_pages : ℕ := orange_books_read * pages_per_orange_book)
  (total_purple_pages : ℕ := P * pages_per_purple_book)
  (condition : total_orange_pages - total_purple_pages = extra_orange_pages) :
  P = 5 := 
by 
  sorry

end NUMINAMATH_GPT_Mirella_read_purple_books_l1919_191971


namespace NUMINAMATH_GPT_cos_pi_minus_2alpha_l1919_191968

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (Real.pi / 2 - α) = Real.sqrt 2 / 3) : 
  Real.cos (Real.pi - 2 * α) = -5 / 9 := by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_2alpha_l1919_191968


namespace NUMINAMATH_GPT_Tia_drove_192_more_miles_l1919_191969

noncomputable def calculate_additional_miles (s_C t_C : ℝ) : ℝ :=
  let d_C := s_C * t_C
  let d_M := (s_C + 8) * (t_C + 3)
  let d_T := (s_C + 12) * (t_C + 4)
  d_T - d_C

theorem Tia_drove_192_more_miles (s_C t_C : ℝ) (h1 : d_M = d_C + 120) (h2 : d_M = (s_C + 8) * (t_C + 3)) : calculate_additional_miles s_C t_C = 192 :=
by {
  sorry
}

end NUMINAMATH_GPT_Tia_drove_192_more_miles_l1919_191969


namespace NUMINAMATH_GPT_find_C_l1919_191911

noncomputable def A_annual_income : ℝ := 403200.0000000001
noncomputable def A_monthly_income : ℝ := A_annual_income / 12 -- 33600.00000000001

noncomputable def x : ℝ := A_monthly_income / 5 -- 6720.000000000002

noncomputable def C : ℝ := (2 * x) / 1.12 -- should be 12000.000000000004

theorem find_C : C = 12000.000000000004 := 
by sorry

end NUMINAMATH_GPT_find_C_l1919_191911


namespace NUMINAMATH_GPT_interior_triangle_area_l1919_191961

theorem interior_triangle_area (a b c : ℝ)
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (hpythagorean : a^2 + b^2 = c^2) :
  1/2 * a * b = 24 :=
by
  sorry

end NUMINAMATH_GPT_interior_triangle_area_l1919_191961


namespace NUMINAMATH_GPT_george_blocks_l1919_191946

theorem george_blocks (num_boxes : ℕ) (blocks_per_box : ℕ) (total_blocks : ℕ) :
  num_boxes = 2 → blocks_per_box = 6 → total_blocks = num_boxes * blocks_per_box → total_blocks = 12 := by
  intros h_num_boxes h_blocks_per_box h_blocks_equal
  rw [h_num_boxes, h_blocks_per_box] at h_blocks_equal
  exact h_blocks_equal

end NUMINAMATH_GPT_george_blocks_l1919_191946


namespace NUMINAMATH_GPT_prime_has_two_square_numbers_l1919_191986

noncomputable def isSquareNumber (p q : ℕ) : Prop :=
  p > q ∧ Nat.Prime p ∧ Nat.Prime q ∧ ¬ p^2 ∣ (q^(p-1) - 1)

theorem prime_has_two_square_numbers (p : ℕ) (hp : Nat.Prime p) (h5 : p ≥ 5) :
  ∃ q1 q2 : ℕ, isSquareNumber p q1 ∧ isSquareNumber p q2 ∧ q1 ≠ q2 :=
by 
  sorry

end NUMINAMATH_GPT_prime_has_two_square_numbers_l1919_191986


namespace NUMINAMATH_GPT_value_of_q_l1919_191914

theorem value_of_q (m p q a b : ℝ) 
  (h₁ : a * b = 6) 
  (h₂ : (a + 1 / b) * (b + 1 / a) = q): 
  q = 49 / 6 := 
sorry

end NUMINAMATH_GPT_value_of_q_l1919_191914


namespace NUMINAMATH_GPT_ratio_of_sums_l1919_191988

theorem ratio_of_sums (a b c : ℚ) (h1 : b / a = 2) (h2 : c / b = 3) : (a + b) / (b + c) = 3 / 8 := 
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l1919_191988


namespace NUMINAMATH_GPT_dyslexian_alphabet_size_l1919_191930

theorem dyslexian_alphabet_size (c v : ℕ) (h1 : (c * v * c * v * c + v * c * v * c * v) = 4800) : c + v = 12 :=
by
  sorry

end NUMINAMATH_GPT_dyslexian_alphabet_size_l1919_191930


namespace NUMINAMATH_GPT_percent_of_men_tenured_l1919_191924

theorem percent_of_men_tenured (total_professors : ℕ) (women_percent tenured_percent women_tenured_or_both_percent men_percent tenured_men_percent : ℝ)
  (h1 : women_percent = 70 / 100)
  (h2 : tenured_percent = 70 / 100)
  (h3 : women_tenured_or_both_percent = 90 / 100)
  (h4 : men_percent = 30 / 100)
  (h5 : total_professors > 0)
  (h6 : tenured_men_percent = (2/3)) :
  tenured_men_percent * 100 = 66.67 :=
by sorry

end NUMINAMATH_GPT_percent_of_men_tenured_l1919_191924


namespace NUMINAMATH_GPT_hypotenuse_unique_l1919_191945

theorem hypotenuse_unique (a b : ℝ) (h: ∃ x : ℝ, x^2 = a^2 + b^2 ∧ x > 0) : 
  ∃! c : ℝ, c^2 = a^2 + b^2 :=
sorry

end NUMINAMATH_GPT_hypotenuse_unique_l1919_191945


namespace NUMINAMATH_GPT_paint_per_door_l1919_191955

variable (cost_per_pint : ℕ) (cost_per_gallon : ℕ) (num_doors : ℕ) (pints_per_gallon : ℕ) (savings : ℕ)

theorem paint_per_door :
  cost_per_pint = 8 →
  cost_per_gallon = 55 →
  num_doors = 8 →
  pints_per_gallon = 8 →
  savings = 9 →
  (pints_per_gallon / num_doors = 1) :=
by
  intros h_cpint h_cgallon h_nd h_pgallon h_savings
  sorry

end NUMINAMATH_GPT_paint_per_door_l1919_191955


namespace NUMINAMATH_GPT_sum_of_digits_of_product_in_base9_l1919_191977

def base9_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  d1 * 9 + d0

def base10_to_base9 (n : ℕ) : ℕ :=
  let d0 := n % 9
  let d1 := (n / 9) % 9
  let d2 := (n / 81) % 9
  d2 * 100 + d1 * 10 + d0

def sum_of_digits_base9 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 + d1 + d0

theorem sum_of_digits_of_product_in_base9 :
  let n1 := base9_to_decimal 36
  let n2 := base9_to_decimal 21
  let product := n1 * n2
  let base9_product := base10_to_base9 product
  sum_of_digits_base9 base9_product = 19 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_product_in_base9_l1919_191977


namespace NUMINAMATH_GPT_fg_difference_l1919_191906

def f (x : ℝ) : ℝ := x^2 - 4 * x + 7
def g (x : ℝ) : ℝ := x + 4

theorem fg_difference : f (g 3) - g (f 3) = 20 :=
by
  sorry

end NUMINAMATH_GPT_fg_difference_l1919_191906


namespace NUMINAMATH_GPT_Q_joined_after_4_months_l1919_191912

namespace Business

-- Definitions
def P_cap := 4000
def Q_cap := 9000
def P_time := 12
def profit_ratio := (2 : ℚ) / 3

-- Statement to prove
theorem Q_joined_after_4_months (x : ℕ) (h : P_cap * P_time / (Q_cap * (12 - x)) = profit_ratio) :
  x = 4 := 
sorry

end Business

end NUMINAMATH_GPT_Q_joined_after_4_months_l1919_191912


namespace NUMINAMATH_GPT_find_triples_l1919_191931

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_triples (a b c : ℕ) :
  is_prime (a^2 + 1) ∧
  is_prime (b^2 + 1) ∧
  (a^2 + 1) * (b^2 + 1) = c^2 + 1 →
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_triples_l1919_191931


namespace NUMINAMATH_GPT_arc_length_correct_l1919_191913

noncomputable def arcLengthOfCurve : ℝ :=
  ∫ φ in (0 : ℝ)..(5 * Real.pi / 12), (2 : ℝ) * (Real.sqrt (φ ^ 2 + 1))

theorem arc_length_correct :
  arcLengthOfCurve = (65 / 144) + Real.log (3 / 2) := by
  sorry

end NUMINAMATH_GPT_arc_length_correct_l1919_191913


namespace NUMINAMATH_GPT_total_vegetables_correct_l1919_191935

def cucumbers : ℕ := 70
def tomatoes : ℕ := 3 * cucumbers
def total_vegetables : ℕ := cucumbers + tomatoes

theorem total_vegetables_correct : total_vegetables = 280 :=
by
  sorry

end NUMINAMATH_GPT_total_vegetables_correct_l1919_191935


namespace NUMINAMATH_GPT_man_speed_l1919_191954

theorem man_speed (L T V_t V_m : ℝ) (hL : L = 400) (hT : T = 35.99712023038157) (hVt : V_t = 46 * 1000 / 3600) (hE : L = (V_t - V_m) * T) : V_m = 1.666666666666684 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_l1919_191954


namespace NUMINAMATH_GPT_number_of_winning_scores_l1919_191932

-- Define the problem conditions
variable (n : ℕ) (team1_scores team2_scores : Finset ℕ)

-- Define the total number of runners
def total_runners := 12

-- Define the sum of placements
def sum_placements : ℕ := (total_runners * (total_runners + 1)) / 2

-- Define the threshold for the winning score
def winning_threshold : ℕ := sum_placements / 2

-- Define the minimum score for a team
def min_score : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Prove that the number of different possible winning scores is 19
theorem number_of_winning_scores : 
  Finset.card (Finset.range (winning_threshold + 1) \ Finset.range min_score) = 19 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_number_of_winning_scores_l1919_191932


namespace NUMINAMATH_GPT_soda_cost_l1919_191966

-- Definitions based on conditions of the problem
variable (b s : ℤ)
variable (h1 : 4 * b + 3 * s = 540)
variable (h2 : 3 * b + 2 * s = 390)

-- The theorem to prove the cost of a soda
theorem soda_cost : s = 60 := by
  sorry

end NUMINAMATH_GPT_soda_cost_l1919_191966


namespace NUMINAMATH_GPT_participation_arrangements_l1919_191901

def num_students : ℕ := 5
def num_competitions : ℕ := 3
def eligible_dance_students : ℕ := 4

def arrangements_singing : ℕ := num_students
def arrangements_chess : ℕ := num_students
def arrangements_dance : ℕ := eligible_dance_students

def total_arrangements : ℕ := arrangements_singing * arrangements_chess * arrangements_dance

theorem participation_arrangements :
  total_arrangements = 100 := by
  sorry

end NUMINAMATH_GPT_participation_arrangements_l1919_191901


namespace NUMINAMATH_GPT_evan_amount_l1919_191952

def adrian : ℤ := sorry
def brenda : ℤ := sorry
def charlie : ℤ := sorry
def dana : ℤ := sorry
def evan : ℤ := sorry

def amounts_sum : Prop := adrian + brenda + charlie + dana + evan = 72
def abs_diff_1 : Prop := abs (adrian - brenda) = 21
def abs_diff_2 : Prop := abs (brenda - charlie) = 8
def abs_diff_3 : Prop := abs (charlie - dana) = 6
def abs_diff_4 : Prop := abs (dana - evan) = 5
def abs_diff_5 : Prop := abs (evan - adrian) = 14

theorem evan_amount
  (h_sum : amounts_sum)
  (h_diff1 : abs_diff_1)
  (h_diff2 : abs_diff_2)
  (h_diff3 : abs_diff_3)
  (h_diff4 : abs_diff_4)
  (h_diff5 : abs_diff_5) :
  evan = 21 := sorry

end NUMINAMATH_GPT_evan_amount_l1919_191952


namespace NUMINAMATH_GPT_cooper_pies_days_l1919_191970

theorem cooper_pies_days :
  ∃ d : ℕ, 7 * d - 50 = 34 ∧ d = 12 :=
by
  sorry

end NUMINAMATH_GPT_cooper_pies_days_l1919_191970


namespace NUMINAMATH_GPT_new_average_age_l1919_191920

theorem new_average_age (n_students : ℕ) (average_student_age : ℕ) (teacher_age : ℕ)
  (h_students : n_students = 50)
  (h_average_student_age : average_student_age = 14)
  (h_teacher_age : teacher_age = 65) :
  (n_students * average_student_age + teacher_age) / (n_students + 1) = 15 :=
by
  sorry

end NUMINAMATH_GPT_new_average_age_l1919_191920


namespace NUMINAMATH_GPT_dragon_2023_first_reappearance_l1919_191949

theorem dragon_2023_first_reappearance :
  let cycle_letters := 6
  let cycle_digits := 4
  Nat.lcm cycle_letters cycle_digits = 12 :=
by
  rfl -- since LCM of 6 and 4 directly calculates to 12

end NUMINAMATH_GPT_dragon_2023_first_reappearance_l1919_191949


namespace NUMINAMATH_GPT_derivative_y_l1919_191933

open Real

noncomputable def y (x : ℝ) : ℝ :=
  log (2 * x - 3 + sqrt (4 * x ^ 2 - 12 * x + 10)) -
  sqrt (4 * x ^ 2 - 12 * x + 10) * arctan (2 * x - 3)

theorem derivative_y (x : ℝ) : 
  (deriv y x) = - arctan (2 * x - 3) / sqrt (4 * x ^ 2 - 12 * x + 10) :=
by
  sorry

end NUMINAMATH_GPT_derivative_y_l1919_191933


namespace NUMINAMATH_GPT_diff_of_squares_l1919_191975

theorem diff_of_squares : (1001^2 - 999^2 = 4000) :=
by
  sorry

end NUMINAMATH_GPT_diff_of_squares_l1919_191975


namespace NUMINAMATH_GPT_total_profit_l1919_191923

theorem total_profit (investment_B : ℝ) (period_B : ℝ) (profit_B : ℝ) (investment_A : ℝ) (period_A : ℝ) (total_profit : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : period_A = 2 * period_B)
  (h3 : profit_B = 6000)
  (h4 : profit_B / (profit_A * 6 + profit_B) = profit_B) : total_profit = 7 * 6000 :=
by 
  sorry

#print axioms total_profit

end NUMINAMATH_GPT_total_profit_l1919_191923


namespace NUMINAMATH_GPT_inequality_problem_l1919_191951

-- Given a < b < 0, we want to prove a^2 > ab > b^2
theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
sorry

end NUMINAMATH_GPT_inequality_problem_l1919_191951


namespace NUMINAMATH_GPT_number_of_girls_l1919_191991

theorem number_of_girls (B G : ℕ) (ratio_condition : B = G / 2) (total_condition : B + G = 90) : 
  G = 60 := 
by
  -- This is the problem statement, with conditions and required result.
  sorry

end NUMINAMATH_GPT_number_of_girls_l1919_191991


namespace NUMINAMATH_GPT_range_of_x_l1919_191958

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) 
  (ineq : |a + b| + |a - b| ≥ |a| * |x - 2|) : 
  0 ≤ x ∧ x ≤ 4 :=
  sorry

end NUMINAMATH_GPT_range_of_x_l1919_191958


namespace NUMINAMATH_GPT_prank_combinations_l1919_191917

theorem prank_combinations :
  let monday := 1
  let tuesday := 4
  let wednesday := 7
  let thursday := 5
  let friday := 1
  (monday * tuesday * wednesday * thursday * friday) = 140 :=
by
  sorry

end NUMINAMATH_GPT_prank_combinations_l1919_191917


namespace NUMINAMATH_GPT_derivative_of_f_l1919_191904

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((2 * x + 1) / Real.sqrt 2) + (2 * x + 1) / (4 * x^2 + 4 * x + 3)

theorem derivative_of_f (x : ℝ) : deriv f x = 8 / (4 * x^2 + 4 * x + 3)^2 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_derivative_of_f_l1919_191904


namespace NUMINAMATH_GPT_crushing_load_calculation_l1919_191956

theorem crushing_load_calculation (T H : ℝ) (L : ℝ) 
  (h1 : L = 40 * T^5 / H^3) 
  (h2 : T = 3) 
  (h3 : H = 6) : 
  L = 45 := 
by sorry

end NUMINAMATH_GPT_crushing_load_calculation_l1919_191956


namespace NUMINAMATH_GPT_clarence_oranges_left_l1919_191981

-- Definitions based on the conditions in the problem
def initial_oranges : ℕ := 5
def oranges_from_joyce : ℕ := 3
def total_oranges_after_joyce : ℕ := initial_oranges + oranges_from_joyce
def oranges_given_to_bob : ℕ := total_oranges_after_joyce / 2
def oranges_left : ℕ := total_oranges_after_joyce - oranges_given_to_bob

-- Proof statement that needs to be proven
theorem clarence_oranges_left : oranges_left = 4 :=
by
  sorry

end NUMINAMATH_GPT_clarence_oranges_left_l1919_191981


namespace NUMINAMATH_GPT_simplify_expression_l1919_191997

theorem simplify_expression (a b : ℝ) : 
  (2 * a^2 * b - 5 * a * b) - 2 * (-a * b + a^2 * b) = -3 * a * b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1919_191997


namespace NUMINAMATH_GPT_solution_l1919_191959

theorem solution (y : ℚ) (h : (1/3 : ℚ) + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end NUMINAMATH_GPT_solution_l1919_191959


namespace NUMINAMATH_GPT_smallest_side_is_10_l1919_191999

noncomputable def smallest_side_of_triangle (x : ℝ) : ℝ :=
    let side1 := 10
    let side2 := 3 * x + 6
    let side3 := x + 5
    min side1 (min side2 side3)

theorem smallest_side_is_10 (x : ℝ) (h : 10 + (3 * x + 6) + (x + 5) = 60) : 
    smallest_side_of_triangle x = 10 :=
by
    sorry

end NUMINAMATH_GPT_smallest_side_is_10_l1919_191999


namespace NUMINAMATH_GPT_landscape_length_l1919_191995

theorem landscape_length (b length : ℕ) (A_playground : ℕ) (h1 : length = 4 * b) (h2 : A_playground = 1200) (h3 : A_playground = (1 / 3 : ℚ) * (length * b)) :
  length = 120 :=
by
  sorry

end NUMINAMATH_GPT_landscape_length_l1919_191995


namespace NUMINAMATH_GPT_complete_square_eq_l1919_191989

theorem complete_square_eq (x : ℝ) : x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  have : x^2 - 2 * x = 5 := by linarith
  have : x^2 - 2 * x + 1 = 6 := by linarith
  exact eq_of_sub_eq_zero (by linarith)

end NUMINAMATH_GPT_complete_square_eq_l1919_191989


namespace NUMINAMATH_GPT_wolf_does_not_catch_hare_l1919_191918

-- Define the distance the hare needs to cover
def distanceHare := 250 -- meters

-- Define the initial separation between the wolf and the hare
def separation := 30 -- meters

-- Define the speed of the hare
def speedHare := 550 -- meters per minute

-- Define the speed of the wolf
def speedWolf := 600 -- meters per minute

-- Define the time it takes for the hare to reach the refuge
def tHare := (distanceHare : ℚ) / speedHare

-- Define the total distance the wolf needs to cover
def totalDistanceWolf := distanceHare + separation

-- Define the time it takes for the wolf to cover the total distance
def tWolf := (totalDistanceWolf : ℚ) / speedWolf

-- Final proposition to be proven
theorem wolf_does_not_catch_hare : tHare < tWolf :=
by
  sorry

end NUMINAMATH_GPT_wolf_does_not_catch_hare_l1919_191918


namespace NUMINAMATH_GPT_increasing_exponential_function_l1919_191962

theorem increasing_exponential_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → (a ^ x) < (a ^ y)) → (1 < a) :=
by
  sorry

end NUMINAMATH_GPT_increasing_exponential_function_l1919_191962


namespace NUMINAMATH_GPT_g_negative_example1_g_negative_example2_g_negative_example3_l1919_191978

noncomputable def g (a : ℚ) : ℚ := sorry

axiom g_mul (a b : ℚ) (ha : 0 < a) (hb : 0 < b) : g (a * b) = g a + g b
axiom g_prime (p : ℕ) (hp : Nat.Prime p) : g (p * p) = p

theorem g_negative_example1 : g (8/81) < 0 := sorry
theorem g_negative_example2 : g (25/72) < 0 := sorry
theorem g_negative_example3 : g (49/18) < 0 := sorry

end NUMINAMATH_GPT_g_negative_example1_g_negative_example2_g_negative_example3_l1919_191978


namespace NUMINAMATH_GPT_axis_of_symmetry_cosine_l1919_191974

theorem axis_of_symmetry_cosine (x : ℝ) : 
  (∃ k : ℤ, 2 * x + π / 3 = k * π) → x = -π / 6 :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_cosine_l1919_191974


namespace NUMINAMATH_GPT_min_reciprocal_sum_l1919_191921

theorem min_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : 
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_reciprocal_sum_l1919_191921


namespace NUMINAMATH_GPT_total_interest_at_tenth_year_l1919_191993

-- Define the conditions for the simple interest problem
variables (P R T : ℝ)

-- Given conditions in the problem
def initial_condition : Prop := (P * R * 10) / 100 = 800
def trebled_principal_condition : Prop := (3 * P * R * 5) / 100 = 1200

-- Statement to prove
theorem total_interest_at_tenth_year (h1 : initial_condition P R) (h2 : trebled_principal_condition P R) :
  (800 + 1200) = 2000 := by
  sorry

end NUMINAMATH_GPT_total_interest_at_tenth_year_l1919_191993


namespace NUMINAMATH_GPT_failed_in_hindi_percentage_l1919_191979

/-- In an examination, a specific percentage of students failed in Hindi (H%), 
45% failed in English, and 20% failed in both. We know that 40% passed in both subjects. 
Prove that 35% students failed in Hindi. --/
theorem failed_in_hindi_percentage : 
  ∀ (H E B P : ℕ),
    (E = 45) → (B = 20) → (P = 40) → (100 - P = H + E - B) → H = 35 := by
  intros H E B P hE hB hP h
  sorry

end NUMINAMATH_GPT_failed_in_hindi_percentage_l1919_191979


namespace NUMINAMATH_GPT_tunnel_length_proof_l1919_191985

variable (train_length : ℝ) (train_speed : ℝ) (time_in_tunnel : ℝ)

noncomputable def tunnel_length (train_length train_speed time_in_tunnel : ℝ) : ℝ :=
  (train_speed / 60) * time_in_tunnel - train_length

theorem tunnel_length_proof 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 30) 
  (h_time_in_tunnel : time_in_tunnel = 4) : 
  tunnel_length 2 30 4 = 2 := by
    simp [tunnel_length, h_train_length, h_train_speed, h_time_in_tunnel]
    norm_num
    sorry

end NUMINAMATH_GPT_tunnel_length_proof_l1919_191985


namespace NUMINAMATH_GPT_people_on_train_after_third_stop_l1919_191960

variable (initial_people : ℕ) (off_1 boarded_1 off_2 boarded_2 off_3 boarded_3 : ℕ)

def people_after_first_stop (initial : ℕ) (off_1 boarded_1 : ℕ) : ℕ :=
  initial - off_1 + boarded_1

def people_after_second_stop (first_stop : ℕ) (off_2 boarded_2 : ℕ) : ℕ :=
  first_stop - off_2 + boarded_2

def people_after_third_stop (second_stop : ℕ) (off_3 boarded_3 : ℕ) : ℕ :=
  second_stop - off_3 + boarded_3

theorem people_on_train_after_third_stop :
  people_after_third_stop (people_after_second_stop (people_after_first_stop initial_people off_1 boarded_1) off_2 boarded_2) off_3 boarded_3 = 42 :=
  by
    have initial_people := 48
    have off_1 := 12
    have boarded_1 := 7
    have off_2 := 15
    have boarded_2 := 9
    have off_3 := 6
    have boarded_3 := 11
    sorry

end NUMINAMATH_GPT_people_on_train_after_third_stop_l1919_191960


namespace NUMINAMATH_GPT_consecutive_ints_product_div_6_l1919_191998

theorem consecutive_ints_product_div_6 (n : ℤ) : (n * (n + 1) * (n + 2)) % 6 = 0 := 
sorry

end NUMINAMATH_GPT_consecutive_ints_product_div_6_l1919_191998


namespace NUMINAMATH_GPT_triangle_area_inscribed_rectangle_area_l1919_191967

theorem triangle_area (m n : ℝ) : ∃ (S : ℝ), S = m * n := 
sorry

theorem inscribed_rectangle_area (m n : ℝ) : ∃ (A : ℝ), A = (2 * m^2 * n^2) / (m + n)^2 :=
sorry

end NUMINAMATH_GPT_triangle_area_inscribed_rectangle_area_l1919_191967


namespace NUMINAMATH_GPT_monotonicity_of_f_l1919_191926

noncomputable def f (x : ℝ) : ℝ := - (2 * x) / (1 + x^2)

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y ∧ (y < -1 ∨ x > 1) → f x < f y) ∧
  (∀ x y : ℝ, x < y ∧ -1 < x ∧ y < 1 → f y < f x) := sorry

end NUMINAMATH_GPT_monotonicity_of_f_l1919_191926


namespace NUMINAMATH_GPT_inequality_proved_l1919_191905

theorem inequality_proved (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proved_l1919_191905


namespace NUMINAMATH_GPT_total_wheels_at_park_l1919_191992

-- Define the problem based on the given conditions
def num_bicycles : ℕ := 6
def num_tricycles : ℕ := 15
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

-- Statement to prove the total number of wheels is 57
theorem total_wheels_at_park : (num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle) = 57 := by
  -- This will be filled in with the proof.
  sorry

end NUMINAMATH_GPT_total_wheels_at_park_l1919_191992


namespace NUMINAMATH_GPT_equal_wear_tires_l1919_191980

theorem equal_wear_tires (t D d : ℕ) (h1 : t = 7) (h2 : D = 42000) (h3 : t * d = 6 * D) : d = 36000 :=
by
  sorry

end NUMINAMATH_GPT_equal_wear_tires_l1919_191980


namespace NUMINAMATH_GPT_line_through_P_origin_line_through_P_perpendicular_to_l3_l1919_191922

-- Define lines l1, l2, l3
def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x + y + 2 = 0
def l3 (x y : ℝ) := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Prove the equations of the lines passing through P
theorem line_through_P_origin : ∃ A B C : ℝ, A * -2 + B * 2 + C = 0 ∧ A * 0 + B * 0 + C = 0 ∧ A = 1 ∧ B = 1 ∧ C = 0 :=
by sorry

theorem line_through_P_perpendicular_to_l3 : ∃ A B C : ℝ, A * -2 + B * 2 + C = 0 ∧ A * P.1 + B * P.2 + C = 0 ∧ A = 2 ∧ B = 1 ∧ C = 2 :=
by sorry

end NUMINAMATH_GPT_line_through_P_origin_line_through_P_perpendicular_to_l3_l1919_191922


namespace NUMINAMATH_GPT_gcd_equation_solutions_l1919_191990

theorem gcd_equation_solutions:
  ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y^2 + Nat.gcd x y ^ 3 = x * y * Nat.gcd x y 
  → (x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = 6) ∨ (x = 5 ∧ y = 2) ∨ (x = 5 ∧ y = 3) := 
by
  intros x y h
  sorry

end NUMINAMATH_GPT_gcd_equation_solutions_l1919_191990


namespace NUMINAMATH_GPT_train_B_speed_l1919_191903

noncomputable def train_speed_B (V_A : ℕ) (T_A : ℕ) (T_B : ℕ) : ℕ :=
  V_A * T_A / T_B

theorem train_B_speed
  (V_A : ℕ := 60)
  (T_A : ℕ := 9)
  (T_B : ℕ := 4) :
  train_speed_B V_A T_A T_B = 135 := 
by
  sorry

end NUMINAMATH_GPT_train_B_speed_l1919_191903


namespace NUMINAMATH_GPT_no_integer_solution_for_z_l1919_191934

theorem no_integer_solution_for_z (z : ℤ) (h : 2 / z = 2 / (z + 1) + 2 / (z + 25)) : false :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_for_z_l1919_191934


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1919_191996

/--
Given an arithmetic sequence {a_n}, the sum of the first n terms is S_n,
a_3 and a_7 are the two roots of the equation 2x^2 - 12x + c = 0,
and S_{13} = c.
Prove that the common difference of the sequence {a_n} satisfies d = -3/2 or d = -7/4.
-/
theorem common_difference_of_arithmetic_sequence 
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (c : ℚ)
  (h1 : ∃ a_3 a_7, (2 * a_3^2 - 12 * a_3 + c = 0) ∧ (2 * a_7^2 - 12 * a_7 + c = 0))
  (h2 : S 13 = c) :
  ∃ d : ℚ, d = -3/2 ∨ d = -7/4 :=
sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1919_191996


namespace NUMINAMATH_GPT_find_value_of_x_cubed_plus_y_cubed_l1919_191953

-- Definitions based on the conditions provided
variables (x y : ℝ)
variables (h1 : y + 3 = (x - 3)^2) (h2 : x + 3 = (y - 3)^2) (h3 : x ≠ y)

theorem find_value_of_x_cubed_plus_y_cubed :
  x^3 + y^3 = 217 :=
sorry

end NUMINAMATH_GPT_find_value_of_x_cubed_plus_y_cubed_l1919_191953


namespace NUMINAMATH_GPT_c_plus_d_l1919_191942

theorem c_plus_d (c d : ℝ)
  (h1 : c^3 - 12 * c^2 + 15 * c - 36 = 0)
  (h2 : 6 * d^3 - 36 * d^2 - 150 * d + 1350 = 0) :
  c + d = 7 := 
  sorry

end NUMINAMATH_GPT_c_plus_d_l1919_191942


namespace NUMINAMATH_GPT_a_is_constant_l1919_191987

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, a n ≥ (a (n+2) + a (n+1) + a (n-1) + a (n-2)) / 4)

theorem a_is_constant : ∀ n m, a n = a m :=
by
  sorry

end NUMINAMATH_GPT_a_is_constant_l1919_191987


namespace NUMINAMATH_GPT_satisfies_negative_inverse_l1919_191994

noncomputable def f1 (x : ℝ) : ℝ := x - 1/x
noncomputable def f2 (x : ℝ) : ℝ := x + 1/x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x
noncomputable def f4 (x : ℝ) : ℝ :=
  if x < 1 then x
  else if x = 1 then 0
  else -1/x

theorem satisfies_negative_inverse :
  { f | (∀ x : ℝ, f (1 / x) = -f x) } = {f1, f3, f4} :=
sorry

end NUMINAMATH_GPT_satisfies_negative_inverse_l1919_191994


namespace NUMINAMATH_GPT_four_g_users_scientific_notation_l1919_191984

-- Condition for scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- Given problem in scientific notation form
theorem four_g_users_scientific_notation :
  ∃ a n, is_scientific_notation a n 1030000000 ∧ a = 1.03 ∧ n = 9 :=
sorry

end NUMINAMATH_GPT_four_g_users_scientific_notation_l1919_191984


namespace NUMINAMATH_GPT_box_dimensions_l1919_191944

theorem box_dimensions {a b c : ℕ} (h1 : a + c = 17) (h2 : a + b = 13) (h3 : 2 * (b + c) = 40) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_box_dimensions_l1919_191944


namespace NUMINAMATH_GPT_simplify_fraction_subtraction_l1919_191983

theorem simplify_fraction_subtraction :
  (5 / 15 : ℚ) - (2 / 45) = 13 / 45 :=
by
  -- (The proof will go here)
  sorry

end NUMINAMATH_GPT_simplify_fraction_subtraction_l1919_191983


namespace NUMINAMATH_GPT_count_whole_numbers_in_interval_l1919_191909

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end NUMINAMATH_GPT_count_whole_numbers_in_interval_l1919_191909


namespace NUMINAMATH_GPT_k_9_pow_4_eq_81_l1919_191965

theorem k_9_pow_4_eq_81 
  (h k : ℝ → ℝ) 
  (hk1 : ∀ (x : ℝ), x ≥ 1 → h (k x) = x^3) 
  (hk2 : ∀ (x : ℝ), x ≥ 1 → k (h x) = x^4) 
  (k81_eq_9 : k 81 = 9) :
  (k 9)^4 = 81 :=
by
  sorry

end NUMINAMATH_GPT_k_9_pow_4_eq_81_l1919_191965


namespace NUMINAMATH_GPT_linda_age_l1919_191948

variable (s j l : ℕ)

theorem linda_age (h1 : (s + j + l) / 3 = 11) 
                  (h2 : l - 5 = s) 
                  (h3 : j + 4 = 3 * (s + 4) / 4) :
                  l = 14 := by
  sorry

end NUMINAMATH_GPT_linda_age_l1919_191948


namespace NUMINAMATH_GPT_ratio_of_scores_l1919_191908

theorem ratio_of_scores (Lizzie Nathalie Aimee teammates : ℕ) (combinedLN : ℕ)
    (team_total : ℕ) (m : ℕ) :
    Lizzie = 4 →
    Nathalie = Lizzie + 3 →
    combinedLN = Lizzie + Nathalie →
    Aimee = m * combinedLN →
    teammates = 17 →
    team_total = Lizzie + Nathalie + Aimee + teammates →
    team_total = 50 →
    (Aimee / combinedLN) = 2 :=
by 
    sorry

end NUMINAMATH_GPT_ratio_of_scores_l1919_191908


namespace NUMINAMATH_GPT_symmetry_about_origin_l1919_191910

def f (x : ℝ) : ℝ := x^3 - x

theorem symmetry_about_origin : 
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_GPT_symmetry_about_origin_l1919_191910


namespace NUMINAMATH_GPT_cos_pi_over_2_minus_2alpha_l1919_191919

theorem cos_pi_over_2_minus_2alpha (α : ℝ) (h : Real.tan α = 2) : Real.cos (Real.pi / 2 - 2 * α) = 4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_cos_pi_over_2_minus_2alpha_l1919_191919


namespace NUMINAMATH_GPT_trapezium_distance_parallel_sides_l1919_191937

theorem trapezium_distance_parallel_sides
  (l1 l2 area : ℝ) (h : ℝ)
  (h_area : area = (1 / 2) * (l1 + l2) * h)
  (hl1 : l1 = 30)
  (hl2 : l2 = 12)
  (h_area_val : area = 336) :
  h = 16 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_distance_parallel_sides_l1919_191937


namespace NUMINAMATH_GPT_sum_largest_smallest_gx_l1919_191957

noncomputable def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_largest_smallest_gx : (∀ x, 1 ≤ x ∧ x ≤ 10 → True) → ∀ (a b : ℝ), (∃ x, 1 ≤ x ∧ x ≤ 10 ∧ g x = a) → (∃ y, 1 ≤ y ∧ y ≤ 10 ∧ g y = b) → a + b = -1 :=
by
  intro h x y hx hy
  sorry

end NUMINAMATH_GPT_sum_largest_smallest_gx_l1919_191957


namespace NUMINAMATH_GPT_lcm_5_711_is_3555_l1919_191976

theorem lcm_5_711_is_3555 : Nat.lcm 5 711 = 3555 := by
  sorry

end NUMINAMATH_GPT_lcm_5_711_is_3555_l1919_191976


namespace NUMINAMATH_GPT_find_M_l1919_191972

theorem find_M : 995 + 997 + 999 + 1001 + 1003 = 5100 - 104 :=
by 
  sorry

end NUMINAMATH_GPT_find_M_l1919_191972


namespace NUMINAMATH_GPT_sum_of_three_eq_six_l1919_191916

theorem sum_of_three_eq_six
  (a b c : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 150) :
  a + b + c = 6 :=
sorry

end NUMINAMATH_GPT_sum_of_three_eq_six_l1919_191916


namespace NUMINAMATH_GPT_correct_calculation_l1919_191902

theorem correct_calculation (a b : ℝ) :
  (6 * a - 5 * a ≠ 1) ∧
  (a + 2 * a^2 ≠ 3 * a^3) ∧
  (- (a - b) = -a + b) ∧
  (2 * (a + b) ≠ 2 * a + b) :=
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_l1919_191902


namespace NUMINAMATH_GPT_find_eighth_number_l1919_191900

theorem find_eighth_number (x : ℕ) (h1 : (1 + 2 + 4 + 5 + 6 + 9 + 9 + x + 12) / 9 = 7) : x = 27 :=
sorry

end NUMINAMATH_GPT_find_eighth_number_l1919_191900


namespace NUMINAMATH_GPT_right_triangle_median_to_hypotenuse_l1919_191925

theorem right_triangle_median_to_hypotenuse 
    {DEF : Type} [MetricSpace DEF] 
    (D E F M : DEF) 
    (h_triangle : dist D E = 15 ∧ dist D F = 20 ∧ dist E F = 25) 
    (h_midpoint : dist D M = dist E M ∧ dist D E = 2 * dist D M ∧ dist E F * dist E F = dist E D * dist E D + dist D F * dist D F) :
    dist F M = 12.5 :=
by sorry

end NUMINAMATH_GPT_right_triangle_median_to_hypotenuse_l1919_191925


namespace NUMINAMATH_GPT_non_congruent_rectangles_with_even_dimensions_l1919_191915

/-- Given a rectangle with perimeter 120 inches and even integer dimensions,
    prove that there are 15 non-congruent rectangles that meet these criteria. -/
theorem non_congruent_rectangles_with_even_dimensions (h w : ℕ) (h_even : h % 2 = 0) (w_even : w % 2 = 0) (perimeter_condition : 2 * (h + w) = 120) :
  ∃ n : ℕ, n = 15 := sorry

end NUMINAMATH_GPT_non_congruent_rectangles_with_even_dimensions_l1919_191915


namespace NUMINAMATH_GPT_measure_of_angle_y_l1919_191939

def is_straight_angle (a : ℝ) := a = 180

theorem measure_of_angle_y (angle_ABC angle_ADB angle_BDA y : ℝ) 
  (h1 : angle_ABC = 117)
  (h2 : angle_ADB = 31)
  (h3 : angle_BDA = 28)
  (h4 : is_straight_angle (angle_ABC + (180 - angle_ABC)))
  : y = 86 := 
by 
  sorry

end NUMINAMATH_GPT_measure_of_angle_y_l1919_191939


namespace NUMINAMATH_GPT_fraction_simplification_l1919_191928

theorem fraction_simplification (x y : ℚ) (hx : x = 4 / 6) (hy : y = 5 / 8) :
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1919_191928


namespace NUMINAMATH_GPT_simplified_radical_formula_l1919_191940

theorem simplified_radical_formula (y : ℝ) (hy : 0 ≤ y):
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) :=
by
  sorry

end NUMINAMATH_GPT_simplified_radical_formula_l1919_191940
