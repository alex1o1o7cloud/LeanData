import Mathlib

namespace NUMINAMATH_GPT_ron_spending_increase_l812_81236

variable (P Q : ℝ) -- initial price and quantity
variable (X : ℝ)   -- intended percentage increase in spending

theorem ron_spending_increase :
  (1 + X / 100) * P * Q = 1.25 * P * (0.92 * Q) →
  X = 15 := 
by
  sorry

end NUMINAMATH_GPT_ron_spending_increase_l812_81236


namespace NUMINAMATH_GPT_negation_proposition_l812_81258

theorem negation_proposition :
  (∀ x : ℝ, 0 < x → x^2 + 1 ≥ 2 * x) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + 1 < 2 * x) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l812_81258


namespace NUMINAMATH_GPT_original_average_l812_81297

theorem original_average (A : ℝ) (h : 5 * A = 130) : A = 26 :=
by
  have h1 : 5 * A / 5 = 130 / 5 := by sorry
  sorry

end NUMINAMATH_GPT_original_average_l812_81297


namespace NUMINAMATH_GPT_sum_digits_largest_N_l812_81281

-- Define the conditions
def is_multiple_of_six (N : ℕ) : Prop := N % 6 = 0

def P (N : ℕ) : ℚ := 
  let favorable_positions := (N + 1) *
    (⌊(1:ℚ) / 3 * N⌋ + 1 + (N - ⌈(2:ℚ) / 3 * N⌉ + 1))
  favorable_positions / (N + 1)

axiom P_6_equals_1 : P 6 = 1
axiom P_large_N : ∀ ε > 0, ∃ N > 0, is_multiple_of_six N ∧ P N ≥ (5/6) - ε

-- Main theorem statement
theorem sum_digits_largest_N : 
  ∃ N : ℕ, is_multiple_of_six N ∧ P N > 3/4 ∧ (N.digits 10).sum = 6 :=
sorry

end NUMINAMATH_GPT_sum_digits_largest_N_l812_81281


namespace NUMINAMATH_GPT_arithmetic_sequence_values_l812_81259

theorem arithmetic_sequence_values (a b c : ℤ) 
  (h1 : 2 * b = a + c)
  (h2 : 2 * a = b + 1)
  (h3 : 2 * c = b + 9) 
  (h4 : a + b + c = -15) :
  b = -5 ∧ a * c = 21 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_values_l812_81259


namespace NUMINAMATH_GPT_quad_to_square_l812_81208

theorem quad_to_square (a b z : ℝ)
  (h_dim : a = 9) 
  (h_dim2 : b = 16) 
  (h_area : a * b = z * z) :
  z = 12 :=
by
  -- Proof outline would go here, but let's skip the actual proof for this definition.
  sorry

end NUMINAMATH_GPT_quad_to_square_l812_81208


namespace NUMINAMATH_GPT_angle_measure_F_l812_81251

theorem angle_measure_F (D E F : ℝ) 
  (h1 : D = 75) 
  (h2 : E = 4 * F - 15) 
  (h3 : D + E + F = 180) : 
  F = 24 := 
sorry

end NUMINAMATH_GPT_angle_measure_F_l812_81251


namespace NUMINAMATH_GPT_payment_ratio_l812_81238

theorem payment_ratio (m p t : ℕ) (hm : m = 14) (hp : p = 84) (ht : t = m * 12) :
  (p : ℚ) / ((t : ℚ) - p) = 1 :=
by
  sorry

end NUMINAMATH_GPT_payment_ratio_l812_81238


namespace NUMINAMATH_GPT_average_of_combined_results_l812_81266

theorem average_of_combined_results {avg1 avg2 n1 n2 : ℝ} (h1 : avg1 = 28) (h2 : avg2 = 55) (h3 : n1 = 55) (h4 : n2 = 28) :
  ((n1 * avg1) + (n2 * avg2)) / (n1 + n2) = 37.11 :=
by sorry

end NUMINAMATH_GPT_average_of_combined_results_l812_81266


namespace NUMINAMATH_GPT_min_sum_of_factors_of_144_is_neg_145_l812_81241

theorem min_sum_of_factors_of_144_is_neg_145 
  (a b : ℤ) 
  (h : a * b = 144) : 
  a + b ≥ -145 := 
sorry

end NUMINAMATH_GPT_min_sum_of_factors_of_144_is_neg_145_l812_81241


namespace NUMINAMATH_GPT_proof_problem_l812_81220

theorem proof_problem (p q : Prop) : (p ∧ q) ↔ ¬ (¬ p ∨ ¬ q) :=
sorry

end NUMINAMATH_GPT_proof_problem_l812_81220


namespace NUMINAMATH_GPT_min_red_hair_students_l812_81234

theorem min_red_hair_students (B N R : ℕ) 
  (h1 : B + N + R = 50)
  (h2 : N ≥ B - 1)
  (h3 : R ≥ N - 1) :
  R = 17 := sorry

end NUMINAMATH_GPT_min_red_hair_students_l812_81234


namespace NUMINAMATH_GPT_symmetry_origin_points_l812_81292

theorem symmetry_origin_points (x y : ℝ) (h₁ : (x, -2) = (-3, -y)) : x + y = -1 :=
sorry

end NUMINAMATH_GPT_symmetry_origin_points_l812_81292


namespace NUMINAMATH_GPT_original_salary_l812_81271

theorem original_salary (S : ℝ) (h : (1.12) * (0.93) * (1.09) * (0.94) * S = 1212) : 
  S = 1212 / ((1.12) * (0.93) * (1.09) * (0.94)) :=
by
  sorry

end NUMINAMATH_GPT_original_salary_l812_81271


namespace NUMINAMATH_GPT_Marissa_sunflower_height_l812_81225

-- Define the necessary conditions
def sister_height_feet : ℕ := 4
def sister_height_inches : ℕ := 3
def extra_sunflower_height : ℕ := 21
def inches_per_foot : ℕ := 12

-- Calculate the total height of the sister in inches
def sister_total_height_inch : ℕ := (sister_height_feet * inches_per_foot) + sister_height_inches

-- Calculate the sunflower height in inches
def sunflower_height_inch : ℕ := sister_total_height_inch + extra_sunflower_height

-- Convert the sunflower height to feet
def sunflower_height_feet : ℕ := sunflower_height_inch / inches_per_foot

-- The theorem we want to prove
theorem Marissa_sunflower_height : sunflower_height_feet = 6 := by
  sorry

end NUMINAMATH_GPT_Marissa_sunflower_height_l812_81225


namespace NUMINAMATH_GPT_total_candy_bars_correct_l812_81298

-- Define the number of each type of candy bar.
def snickers : Nat := 3
def marsBars : Nat := 2
def butterfingers : Nat := 7

-- Define the total number of candy bars.
def totalCandyBars : Nat := snickers + marsBars + butterfingers

-- Formulate the theorem about the total number of candy bars.
theorem total_candy_bars_correct : totalCandyBars = 12 :=
sorry

end NUMINAMATH_GPT_total_candy_bars_correct_l812_81298


namespace NUMINAMATH_GPT_interior_angle_regular_octagon_l812_81296

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end NUMINAMATH_GPT_interior_angle_regular_octagon_l812_81296


namespace NUMINAMATH_GPT_train_length_l812_81274

theorem train_length (L : ℝ) (h1 : 46 - 36 = 10) (h2 : 45 * (10 / 3600) = 1 / 8) : L = 62.5 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l812_81274


namespace NUMINAMATH_GPT_total_sweaters_l812_81210

-- Define the conditions
def washes_per_load : ℕ := 9
def total_shirts : ℕ := 19
def total_loads : ℕ := 3

-- Define the total_sweaters theorem to prove Nancy had to wash 9 sweaters
theorem total_sweaters {n : ℕ} (h1 : washes_per_load = 9) (h2 : total_shirts = 19) (h3 : total_loads = 3) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_total_sweaters_l812_81210


namespace NUMINAMATH_GPT_sum_of_coordinates_l812_81288

theorem sum_of_coordinates (x : ℚ) : (0, 0) = (0, 0) ∧ (x, -3) = (x, -3) ∧ ((-3 - 0) / (x - 0) = 4 / 5) → x - 3 = -27 / 4 := 
sorry

end NUMINAMATH_GPT_sum_of_coordinates_l812_81288


namespace NUMINAMATH_GPT_unit_prices_l812_81269

theorem unit_prices (x y : ℕ) (h1 : 5 * x + 4 * y = 139) (h2 : 4 * x + 5 * y = 140) :
  x = 15 ∧ y = 16 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_unit_prices_l812_81269


namespace NUMINAMATH_GPT_total_savings_percentage_l812_81231

theorem total_savings_percentage :
  let coat_price := 100
  let hat_price := 50
  let shoes_price := 75
  let coat_discount := 0.30
  let hat_discount := 0.40
  let shoes_discount := 0.25
  let original_total := coat_price + hat_price + shoes_price
  let coat_savings := coat_price * coat_discount
  let hat_savings := hat_price * hat_discount
  let shoes_savings := shoes_price * shoes_discount
  let total_savings := coat_savings + hat_savings + shoes_savings
  let savings_percentage := (total_savings / original_total) * 100
  savings_percentage = 30.556 :=
by
  sorry

end NUMINAMATH_GPT_total_savings_percentage_l812_81231


namespace NUMINAMATH_GPT_sin_seven_pi_over_six_l812_81233

theorem sin_seven_pi_over_six :
  Real.sin (7 * Real.pi / 6) = - 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_seven_pi_over_six_l812_81233


namespace NUMINAMATH_GPT_area_triangle_BFC_l812_81278

-- Definitions based on conditions
def Rectangle (A B C D : Type) (AB BC CD DA : ℝ) := AB = 5 ∧ BC = 12 ∧ CD = 5 ∧ DA = 12

def PointOnDiagonal (F A C : Type) := True  -- Simplified definition as being on the diagonal
def Perpendicular (B F A C : Type) := True  -- Simplified definition as being perpendicular

-- Main theorem statement
theorem area_triangle_BFC 
  (A B C D F : Type)
  (rectangle_ABCD : Rectangle A B C D 5 12 5 12)
  (F_on_AC : PointOnDiagonal F A C)
  (BF_perpendicular_AC : Perpendicular B F A C) :
  ∃ (area : ℝ), area = 30 :=
sorry

end NUMINAMATH_GPT_area_triangle_BFC_l812_81278


namespace NUMINAMATH_GPT_time_needed_n_l812_81263

variable (n : Nat)
variable (d : Nat := n - 1)
variable (s : ℚ := 2 / 3 * (d))
variable (time_third_mile : ℚ := 3)
noncomputable def time_needed (n : Nat) : ℚ := (3 * (n - 1)) / 2

theorem time_needed_n: 
  (∀ (n : Nat), n > 2 → time_needed n = (3 * (n - 1)) / 2) :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_time_needed_n_l812_81263


namespace NUMINAMATH_GPT_money_equations_l812_81229

theorem money_equations (x y : ℝ) (h1 : x + (1 / 2) * y = 50) (h2 : y + (2 / 3) * x = 50) :
  x + (1 / 2) * y = 50 ∧ y + (2 / 3) * x = 50 :=
by
  exact ⟨h1, h2⟩

-- Please note that by stating the theorem this way, we have restated the conditions and conclusion
-- in Lean 4. The proof uses the given conditions directly without the need for intermediate steps.

end NUMINAMATH_GPT_money_equations_l812_81229


namespace NUMINAMATH_GPT_part_1_part_2_l812_81204

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part_1 (a : ℝ) (h : ∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) : a = 2 :=
sorry

theorem part_2 (a : ℝ) (h : a = 2) : ∀ m, (∀ x, f (3 * x) a + f (x + 3) a ≥ m) ↔ m ≤ 5 / 3 :=
sorry

end NUMINAMATH_GPT_part_1_part_2_l812_81204


namespace NUMINAMATH_GPT_sequence_property_l812_81286

theorem sequence_property :
  ∃ (a_0 a_1 a_2 a_3 : ℕ),
    a_0 + a_1 + a_2 + a_3 = 4 ∧
    (a_0 = ([a_0, a_1, a_2, a_3].count 0)) ∧
    (a_1 = ([a_0, a_1, a_2, a_3].count 1)) ∧
    (a_2 = ([a_0, a_1, a_2, a_3].count 2)) ∧
    (a_3 = ([a_0, a_1, a_2, a_3].count 3)) :=
sorry

end NUMINAMATH_GPT_sequence_property_l812_81286


namespace NUMINAMATH_GPT_bruce_paid_amount_l812_81277

def kg_of_grapes : ℕ := 8
def rate_per_kg_grapes : ℕ := 70
def kg_of_mangoes : ℕ := 10
def rate_per_kg_mangoes : ℕ := 55

def total_amount_paid : ℕ := (kg_of_grapes * rate_per_kg_grapes) + (kg_of_mangoes * rate_per_kg_mangoes)

theorem bruce_paid_amount : total_amount_paid = 1110 :=
by sorry

end NUMINAMATH_GPT_bruce_paid_amount_l812_81277


namespace NUMINAMATH_GPT_find_x_l812_81247

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Lean statement asserting that if a is parallel to b for some x, then x = 2
theorem find_x (x : ℝ) (h : parallel a (b x)) : x = 2 := 
by sorry

end NUMINAMATH_GPT_find_x_l812_81247


namespace NUMINAMATH_GPT_find_x_for_abs_expression_zero_l812_81201

theorem find_x_for_abs_expression_zero (x : ℚ) : |5 * x - 2| = 0 → x = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_find_x_for_abs_expression_zero_l812_81201


namespace NUMINAMATH_GPT_find_y_of_equations_l812_81245

theorem find_y_of_equations (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 1 + 1 / y) (h2 : y = 2 + 1 / x) : 
  y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_y_of_equations_l812_81245


namespace NUMINAMATH_GPT_three_students_with_A_l812_81219

-- Define the statements of the students
variables (Eliza Fiona George Harry : Prop)

-- Conditions based on the problem statement
axiom Fiona_implies_Eliza : Fiona → Eliza
axiom George_implies_Fiona : George → Fiona
axiom Harry_implies_George : Harry → George

-- There are exactly three students who scored an A
theorem three_students_with_A (hE : Bool) : 
  (Eliza = false) → (Fiona = true) → (George = true) → (Harry = true) :=
by
  sorry

end NUMINAMATH_GPT_three_students_with_A_l812_81219


namespace NUMINAMATH_GPT_remainder_43_pow_43_plus_43_mod_44_l812_81206

theorem remainder_43_pow_43_plus_43_mod_44 :
  let n := 43
  let m := 44
  (n^43 + n) % m = 42 :=
by 
  let n := 43
  let m := 44
  sorry

end NUMINAMATH_GPT_remainder_43_pow_43_plus_43_mod_44_l812_81206


namespace NUMINAMATH_GPT_apple_harvest_l812_81282

theorem apple_harvest (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 →
  num_sections = 8 →
  total_sacks = sacks_per_section * num_sections →
  total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_apple_harvest_l812_81282


namespace NUMINAMATH_GPT_total_gold_value_l812_81211

def legacy_bars : ℕ := 5
def aleena_bars : ℕ := legacy_bars - 2
def value_per_bar : ℕ := 2200
def total_bars : ℕ := legacy_bars + aleena_bars
def total_value : ℕ := total_bars * value_per_bar

theorem total_gold_value : total_value = 17600 :=
by
  -- Begin proof
  sorry

end NUMINAMATH_GPT_total_gold_value_l812_81211


namespace NUMINAMATH_GPT_sum_of_coordinates_of_B_l812_81264

theorem sum_of_coordinates_of_B (x : ℝ) (y : ℝ) 
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (0,0)) 
  (hB : B = (x, 3))
  (hslope : (3 - 0) / (x - 0) = 4 / 5) :
  x + 3 = 6.75 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_B_l812_81264


namespace NUMINAMATH_GPT_unique_solution_c_eq_one_l812_81237

theorem unique_solution_c_eq_one (b c : ℝ) (hb : b > 0) 
  (h_unique_solution : ∃ x : ℝ, x^2 + (b + 1/b) * x + c = 0 ∧ 
  ∀ y : ℝ, y^2 + (b + 1/b) * y + c = 0 → y = x) : c = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_c_eq_one_l812_81237


namespace NUMINAMATH_GPT_initial_percent_l812_81215

theorem initial_percent (x : ℝ) :
  (x / 100) * (5 / 100) = 60 / 100 → x = 1200 := 
by 
  sorry

end NUMINAMATH_GPT_initial_percent_l812_81215


namespace NUMINAMATH_GPT_simplify_abs_value_l812_81214

theorem simplify_abs_value : abs (- 5 ^ 2 + 6) = 19 := by
  sorry

end NUMINAMATH_GPT_simplify_abs_value_l812_81214


namespace NUMINAMATH_GPT_candle_height_comparison_l812_81217

def first_candle_height (t : ℝ) : ℝ := 10 - 2 * t
def second_candle_height (t : ℝ) : ℝ := 8 - 2 * t

theorem candle_height_comparison (t : ℝ) :
  first_candle_height t = 3 * second_candle_height t → t = 3.5 :=
by
  -- the main proof steps would be here
  sorry

end NUMINAMATH_GPT_candle_height_comparison_l812_81217


namespace NUMINAMATH_GPT_price_of_first_oil_l812_81242

variable {x : ℝ}
variable {price1 volume1 price2 volume2 mix_price mix_volume : ℝ}

theorem price_of_first_oil:
  volume1 = 10 →
  price2 = 68 →
  volume2 = 5 →
  mix_volume = 15 →
  mix_price = 56 →
  (volume1 * x + volume2 * price2 = mix_volume * mix_price) →
  x = 50 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h1 : volume1 = 10 := h1
  have h2 : price2 = 68 := h2
  have h3 : volume2 = 5 := h3
  have h4 : mix_volume = 15 := h4
  have h5 : mix_price = 56 := h5
  have h6 : volume1 * x + volume2 * price2 = mix_volume * mix_price := h6
  sorry

end NUMINAMATH_GPT_price_of_first_oil_l812_81242


namespace NUMINAMATH_GPT_roger_coins_left_l812_81240

theorem roger_coins_left {pennies nickels dimes donated_coins initial_coins remaining_coins : ℕ} 
    (h1 : pennies = 42) 
    (h2 : nickels = 36) 
    (h3 : dimes = 15) 
    (h4 : donated_coins = 66) 
    (h5 : initial_coins = pennies + nickels + dimes) 
    (h6 : remaining_coins = initial_coins - donated_coins) : 
    remaining_coins = 27 := 
sorry

end NUMINAMATH_GPT_roger_coins_left_l812_81240


namespace NUMINAMATH_GPT_no_perfect_square_l812_81284

theorem no_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = 2 * 13^n + 5 * 7^n + 26 :=
sorry

end NUMINAMATH_GPT_no_perfect_square_l812_81284


namespace NUMINAMATH_GPT_student_failed_by_l812_81272

-- Definitions based on the problem conditions
def total_marks : ℕ := 500
def passing_percentage : ℕ := 40
def marks_obtained : ℕ := 150
def passing_marks : ℕ := (passing_percentage * total_marks) / 100

-- The theorem statement
theorem student_failed_by :
  (passing_marks - marks_obtained) = 50 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_student_failed_by_l812_81272


namespace NUMINAMATH_GPT_find_first_term_of_geometric_progression_l812_81227

theorem find_first_term_of_geometric_progression
  (a_2 : ℝ) (a_3 : ℝ) (a_1 : ℝ) (q : ℝ)
  (h1 : a_2 = a_1 * q)
  (h2 : a_3 = a_1 * q^2)
  (h3 : a_2 = 5)
  (h4 : a_3 = 1) : a_1 = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_first_term_of_geometric_progression_l812_81227


namespace NUMINAMATH_GPT_simplify_radicals_l812_81294

open Real

theorem simplify_radicals : sqrt 72 + sqrt 32 = 10 * sqrt 2 := by
  sorry

end NUMINAMATH_GPT_simplify_radicals_l812_81294


namespace NUMINAMATH_GPT_planA_charge_for_8_minutes_eq_48_cents_l812_81254

theorem planA_charge_for_8_minutes_eq_48_cents
  (X : ℝ)
  (hA : ∀ t : ℝ, t ≤ 8 → X = X)
  (hB : ∀ t : ℝ, 6 * 0.08 = 0.48)
  (hEqual : 6 * 0.08 = X) :
  X = 0.48 := by
  sorry

end NUMINAMATH_GPT_planA_charge_for_8_minutes_eq_48_cents_l812_81254


namespace NUMINAMATH_GPT_average_ducks_l812_81209

theorem average_ducks (a e k : ℕ) 
  (h1 : a = 2 * e) 
  (h2 : e = k - 45) 
  (h3 : a = 30) :
  (a + e + k) / 3 = 35 :=
by
  sorry

end NUMINAMATH_GPT_average_ducks_l812_81209


namespace NUMINAMATH_GPT_boys_in_class_l812_81203

theorem boys_in_class (total_students : ℕ) (fraction_girls : ℝ) (fraction_girls_eq : fraction_girls = 1 / 4) (total_students_eq : total_students = 160) :
  (total_students - fraction_girls * total_students = 120) :=
by
  rw [fraction_girls_eq, total_students_eq]
  -- Here, additional lines proving the steps would follow, but we use sorry for completeness.
  sorry

end NUMINAMATH_GPT_boys_in_class_l812_81203


namespace NUMINAMATH_GPT_probability_queen_of_diamonds_l812_81285

/-- 
A standard deck of 52 cards consists of 13 ranks and 4 suits.
We want to prove that the probability the top card is the Queen of Diamonds is 1/52.
-/
theorem probability_queen_of_diamonds 
  (total_cards : ℕ) 
  (queen_of_diamonds : ℕ)
  (h1 : total_cards = 52)
  (h2 : queen_of_diamonds = 1) : 
  (queen_of_diamonds : ℚ) / (total_cards : ℚ) = 1 / 52 := 
by 
  sorry

end NUMINAMATH_GPT_probability_queen_of_diamonds_l812_81285


namespace NUMINAMATH_GPT_gcd_binom_is_integer_l812_81279

theorem gcd_binom_is_integer 
  (m n : ℤ) 
  (hm : m ≥ 1) 
  (hn : n ≥ m)
  (gcd_mn : ℤ := Int.gcd m n)
  (binom_nm : ℤ := Nat.choose n.toNat m.toNat) :
  (gcd_mn * binom_nm) % n.toNat = 0 := by
  sorry

end NUMINAMATH_GPT_gcd_binom_is_integer_l812_81279


namespace NUMINAMATH_GPT_function_characterization_l812_81244

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization (f : ℝ → ℝ) (k : ℝ) :
  (∀ x y : ℝ, f (x^2 + 2*x*y + y^2) = (x + y) * (f x + f y)) →
  (∀ x : ℝ, |f x - k * x| ≤ |x^2 - x|) →
  ∀ x : ℝ, f x = k * x :=
by
  sorry

end NUMINAMATH_GPT_function_characterization_l812_81244


namespace NUMINAMATH_GPT_minimum_cost_l812_81289

noncomputable def volume : ℝ := 4800
noncomputable def depth : ℝ := 3
noncomputable def base_cost_per_sqm : ℝ := 150
noncomputable def wall_cost_per_sqm : ℝ := 120
noncomputable def base_area (volume depth : ℝ) : ℝ := volume / depth
noncomputable def wall_surface_area (x : ℝ) : ℝ :=
  6 * x + (2 * (volume * depth / x))

noncomputable def construction_cost (x : ℝ) : ℝ :=
  wall_surface_area x * wall_cost_per_sqm + base_area volume depth * base_cost_per_sqm

theorem minimum_cost :
  ∃(x : ℝ), x = 40 ∧ construction_cost x = 297600 := by
  sorry

end NUMINAMATH_GPT_minimum_cost_l812_81289


namespace NUMINAMATH_GPT_line_equation_l812_81290

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4))
  (sum_intercepts_zero : ∃ a b : ℝ, (a + b = 0) ∧ (A.1 * b + A.2 * a = a * b)) :
  (∀ x y : ℝ, x - A.1 = (y - A.2) * 4 → 4 * x - y = 0) ∨
  (∀ x y : ℝ, (x / (-3)) + (y / 3) = 1 → x - y + 3 = 0) :=
sorry

end NUMINAMATH_GPT_line_equation_l812_81290


namespace NUMINAMATH_GPT_randy_blocks_left_l812_81207

theorem randy_blocks_left 
  (initial_blocks : ℕ := 78)
  (used_blocks : ℕ := 19)
  (given_blocks : ℕ := 25)
  (bought_blocks : ℕ := 36)
  (sets_from_sister : ℕ := 3)
  (blocks_per_set : ℕ := 12) :
  (initial_blocks - used_blocks - given_blocks + bought_blocks + (sets_from_sister * blocks_per_set)) / 2 = 53 := 
by
  sorry

end NUMINAMATH_GPT_randy_blocks_left_l812_81207


namespace NUMINAMATH_GPT_find_m_from_power_function_l812_81224

theorem find_m_from_power_function :
  (∃ a : ℝ, (2 : ℝ) ^ a = (Real.sqrt 2) / 2) →
  (∃ m : ℝ, (m : ℝ) ^ (-1 / 2 : ℝ) = 2) →
  ∃ m : ℝ, m = 1 / 4 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_m_from_power_function_l812_81224


namespace NUMINAMATH_GPT_intersection_eq_l812_81221

noncomputable def A : Set ℕ := {1, 2, 3, 4}
noncomputable def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_eq : A ∩ B = {2, 3, 4} := 
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l812_81221


namespace NUMINAMATH_GPT_problem_statement_l812_81252

theorem problem_statement
  (a b c : ℝ)
  (h1 : a + 2 * b + 3 * c = 12)
  (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 := 
sorry

end NUMINAMATH_GPT_problem_statement_l812_81252


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l812_81291

def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := 
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l812_81291


namespace NUMINAMATH_GPT_prove_angle_C_prove_max_area_l812_81299

open Real

variables {A B C : ℝ} {a b c : ℝ} (abc_is_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (R : ℝ) (circumradius_is_sqrt2 : R = sqrt 2)
variables (H : 2 * sqrt 2 * (sin A ^ 2 - sin C ^ 2) = (a - b) * sin B)
variables (law_of_sines : a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C)

-- Part 1: Prove that angle C = π / 3
theorem prove_angle_C : C = π / 3 :=
sorry

-- Part 2: Prove that the maximum value of the area S of triangle ABC is (3 * sqrt 3) / 2
theorem prove_max_area : (1 / 2) * a * b * sin C ≤ (3 * sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_prove_angle_C_prove_max_area_l812_81299


namespace NUMINAMATH_GPT_ratio_shortest_to_middle_tree_l812_81239

theorem ratio_shortest_to_middle_tree (height_tallest : ℕ) 
  (height_middle : ℕ) (height_shortest : ℕ)
  (h1 : height_tallest = 150) 
  (h2 : height_middle = (2 * height_tallest) / 3) 
  (h3 : height_shortest = 50) : 
  height_shortest / height_middle = 1 / 2 := by sorry

end NUMINAMATH_GPT_ratio_shortest_to_middle_tree_l812_81239


namespace NUMINAMATH_GPT_Oshea_needs_30_small_planters_l812_81295

theorem Oshea_needs_30_small_planters 
  (total_seeds : ℕ) 
  (large_planters : ℕ) 
  (capacity_large : ℕ) 
  (capacity_small : ℕ)
  (h1: total_seeds = 200) 
  (h2: large_planters = 4) 
  (h3: capacity_large = 20) 
  (h4: capacity_small = 4) : 
  (total_seeds - large_planters * capacity_large) / capacity_small = 30 :=
by 
  sorry

end NUMINAMATH_GPT_Oshea_needs_30_small_planters_l812_81295


namespace NUMINAMATH_GPT_number_of_multiples_of_10_lt_200_l812_81260

theorem number_of_multiples_of_10_lt_200 : 
  ∃ n, (∀ k, (1 ≤ k) → (k < 20) → k * 10 < 200) ∧ n = 19 := 
by
  sorry

end NUMINAMATH_GPT_number_of_multiples_of_10_lt_200_l812_81260


namespace NUMINAMATH_GPT_A_investment_l812_81256

theorem A_investment (x : ℝ) (hx : 0 < x) :
  (∃ a b c d e : ℝ,
    a = x ∧ b = 12 ∧ c = 200 ∧ d = 6 ∧ e = 60 ∧ 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    ((a * b) / (a * b + c * d)) * 100 = e)
  → x = 150 :=
by
  sorry

end NUMINAMATH_GPT_A_investment_l812_81256


namespace NUMINAMATH_GPT_ratio_shorter_longer_l812_81293

theorem ratio_shorter_longer (total_length shorter_length longer_length : ℝ)
  (h1 : total_length = 21) 
  (h2 : shorter_length = 6) 
  (h3 : longer_length = total_length - shorter_length) 
  (h4 : shorter_length / longer_length = 2 / 5) : 
  shorter_length / longer_length = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_shorter_longer_l812_81293


namespace NUMINAMATH_GPT_n_squared_plus_n_plus_1_is_perfect_square_l812_81205

theorem n_squared_plus_n_plus_1_is_perfect_square (n : ℕ) :
  (∃ k : ℕ, n^2 + n + 1 = k^2) ↔ n = 0 :=
by
  sorry

end NUMINAMATH_GPT_n_squared_plus_n_plus_1_is_perfect_square_l812_81205


namespace NUMINAMATH_GPT_xyz_value_l812_81246

theorem xyz_value (x y z : ℝ) (h1 : y = x + 1) (h2 : x + y = 2 * z) (h3 : x = 3) : x * y * z = 42 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_xyz_value_l812_81246


namespace NUMINAMATH_GPT_yellow_crayons_count_l812_81273

def red_crayons := 14
def blue_crayons := red_crayons + 5
def yellow_crayons := 2 * blue_crayons - 6

theorem yellow_crayons_count : yellow_crayons = 32 := by
  sorry

end NUMINAMATH_GPT_yellow_crayons_count_l812_81273


namespace NUMINAMATH_GPT_andre_flowers_given_l812_81216

variable (initialFlowers totalFlowers flowersGiven : ℕ)

theorem andre_flowers_given (h1 : initialFlowers = 67) (h2 : totalFlowers = 90) :
  flowersGiven = totalFlowers - initialFlowers → flowersGiven = 23 :=
by
  intro h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end NUMINAMATH_GPT_andre_flowers_given_l812_81216


namespace NUMINAMATH_GPT_slope_angle_of_line_l812_81223

theorem slope_angle_of_line (α : ℝ) (hα : 0 ≤ α ∧ α < 180) 
    (slope_eq_tan : Real.tan α = 1) : α = 45 :=
by
  sorry

end NUMINAMATH_GPT_slope_angle_of_line_l812_81223


namespace NUMINAMATH_GPT_num_integers_condition_l812_81287

theorem num_integers_condition : 
  (∃ (n1 n2 n3 : ℤ), 0 < n1 ∧ n1 < 30 ∧ (∃ k1 : ℤ, (30 - n1) / n1 = k1 ^ 2) ∧
                     0 < n2 ∧ n2 < 30 ∧ (∃ k2 : ℤ, (30 - n2) / n2 = k2 ^ 2) ∧
                     0 < n3 ∧ n3 < 30 ∧ (∃ k3 : ℤ, (30 - n3) / n3 = k3 ^ 2) ∧
                     ∀ n : ℤ, 0 < n ∧ n < 30 ∧ (∃ k : ℤ, (30 - n) / n = k ^ 2) → 
                              (n = n1 ∨ n = n2 ∨ n = n3)) :=
sorry

end NUMINAMATH_GPT_num_integers_condition_l812_81287


namespace NUMINAMATH_GPT_trajectory_and_min_area_l812_81268

theorem trajectory_and_min_area (C : ℝ → ℝ → Prop) (P : ℝ × ℝ → Prop)
  (l : ℝ → ℝ) (F : ℝ × ℝ) (M : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ)
  (k : ℝ) : 
  (∀ x y, P (x, y) ↔ x ^ 2 = 4 * y) → 
  P (0, 1) →
  (∀ y, l y = -1) →
  F = (0, 1) →
  (∀ x1 y1 x2 y2, x1 + x2 = 4 * k → x1 * x2 = -4 →
    M (x1, y1) (x2, y2) = (2 * k, -1)) →
  (min_area : ℝ) → 
  min_area = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_trajectory_and_min_area_l812_81268


namespace NUMINAMATH_GPT_exists_special_number_divisible_by_1991_l812_81212

theorem exists_special_number_divisible_by_1991 :
  ∃ (N : ℤ) (n : ℕ), n > 2 ∧ (N % 1991 = 0) ∧ 
  (∃ a b x : ℕ, N = 10 ^ (n + 1) * a + 10 ^ n * x + 9 * 10 ^ (n - 1) + b) :=
sorry

end NUMINAMATH_GPT_exists_special_number_divisible_by_1991_l812_81212


namespace NUMINAMATH_GPT_total_supermarkets_FGH_chain_l812_81283

def supermarkets_us : ℕ := 47
def supermarkets_difference : ℕ := 10
def supermarkets_canada : ℕ := supermarkets_us - supermarkets_difference
def total_supermarkets : ℕ := supermarkets_us + supermarkets_canada

theorem total_supermarkets_FGH_chain : total_supermarkets = 84 :=
by 
  sorry

end NUMINAMATH_GPT_total_supermarkets_FGH_chain_l812_81283


namespace NUMINAMATH_GPT_minimum_toys_to_add_l812_81232

theorem minimum_toys_to_add {T : ℤ} (k m n : ℤ) (h1 : T = 12 * k + 3) (h2 : T = 18 * m + 3) 
  (h3 : T = 36 * n + 3) : 
  ∃ x : ℤ, (T + x) % 7 = 0 ∧ x = 4 :=
sorry

end NUMINAMATH_GPT_minimum_toys_to_add_l812_81232


namespace NUMINAMATH_GPT_hank_newspaper_reading_time_l812_81250

theorem hank_newspaper_reading_time
  (n_days_weekday : ℕ := 5)
  (novel_reading_time_weekday : ℕ := 60)
  (n_days_weekend : ℕ := 2)
  (total_weekly_reading_time : ℕ := 810)
  (x : ℕ)
  (h1 : n_days_weekday * x + n_days_weekday * novel_reading_time_weekday +
        n_days_weekend * 2 * x + n_days_weekend * 2 * novel_reading_time_weekday = total_weekly_reading_time) :
  x = 30 := 
by {
  sorry -- Proof would go here
}

end NUMINAMATH_GPT_hank_newspaper_reading_time_l812_81250


namespace NUMINAMATH_GPT_fraction_halfway_l812_81243

theorem fraction_halfway (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  (1 / 2) * ((a / b) + (c / d)) = 19 / 24 := 
by
  sorry

end NUMINAMATH_GPT_fraction_halfway_l812_81243


namespace NUMINAMATH_GPT_domain_of_f_l812_81230

noncomputable def f (x : ℝ) : ℝ := (5 * x - 2) / Real.sqrt (x^2 - 3 * x - 4)

theorem domain_of_f :
  {x : ℝ | ∃ (f_x : ℝ), f x = f_x} = {x : ℝ | (x < -1) ∨ (x > 4)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l812_81230


namespace NUMINAMATH_GPT_real_roots_quadratic_iff_l812_81253

theorem real_roots_quadratic_iff (a : ℝ) : (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := 
sorry

end NUMINAMATH_GPT_real_roots_quadratic_iff_l812_81253


namespace NUMINAMATH_GPT_partial_fraction_sum_l812_81262

theorem partial_fraction_sum :
  ∃ P Q R : ℚ, 
    P * ((-1 : ℚ) * (-2 : ℚ)) + Q * ((-3 : ℚ) * (-2 : ℚ)) + R * ((-3 : ℚ) * (1 : ℚ))
    = 14 ∧ 
    R * (1 : ℚ) * (3 : ℚ) + Q * ((-4 : ℚ) * (-3 : ℚ)) + P * ((3 : ℚ) * (1 : ℚ)) 
      = 12 ∧ 
    P + Q + R = 115 / 30 := by
  sorry

end NUMINAMATH_GPT_partial_fraction_sum_l812_81262


namespace NUMINAMATH_GPT_students_water_count_l812_81275

-- Define the given conditions
def pct_students_juice (total_students : ℕ) : ℕ := 70 * total_students / 100
def pct_students_water (total_students : ℕ) : ℕ := 30 * total_students / 100
def students_juice (total_students : ℕ) : Prop := pct_students_juice total_students = 140

-- Define the proposition that needs to be proven
theorem students_water_count (total_students : ℕ) (h1 : students_juice total_students) : 
  pct_students_water total_students = 60 := 
by
  sorry


end NUMINAMATH_GPT_students_water_count_l812_81275


namespace NUMINAMATH_GPT_calculate_area_bounded_figure_l812_81255

noncomputable def area_of_bounded_figure (R : ℝ) : ℝ :=
  (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi)

theorem calculate_area_bounded_figure (R : ℝ) :
  ∀ r, r = (R / 3) → area_of_bounded_figure R = (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi) :=
by
  intros r hr
  subst hr
  exact rfl

end NUMINAMATH_GPT_calculate_area_bounded_figure_l812_81255


namespace NUMINAMATH_GPT_range_of_x_l812_81218

noncomputable def function_domain (x : ℝ) : Prop :=
x + 2 > 0 ∧ x ≠ 1

theorem range_of_x {x : ℝ} (h : function_domain x) : x > -2 ∧ x ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l812_81218


namespace NUMINAMATH_GPT_frames_per_page_l812_81235

theorem frames_per_page (total_frames : ℕ) (total_pages : ℝ) (h1 : total_frames = 1573) (h2 : total_pages = 11.0) : total_frames / total_pages = 143 := by
  sorry

end NUMINAMATH_GPT_frames_per_page_l812_81235


namespace NUMINAMATH_GPT_frac_div_l812_81270

theorem frac_div : (3 / 7) / (4 / 5) = 15 / 28 := by
  sorry

end NUMINAMATH_GPT_frac_div_l812_81270


namespace NUMINAMATH_GPT_find_physics_marks_l812_81280

theorem find_physics_marks (P C M : ℕ) (h1 : P + C + M = 210) (h2 : P + M = 180) (h3 : P + C = 140) : P = 110 :=
sorry

end NUMINAMATH_GPT_find_physics_marks_l812_81280


namespace NUMINAMATH_GPT_hannah_age_l812_81276

-- Define the constants and conditions
variables (E F G H : ℕ)
axiom h₁ : E = F - 4
axiom h₂ : F = G + 6
axiom h₃ : H = G + 2
axiom h₄ : E = 15

-- Prove that Hannah is 15 years old
theorem hannah_age : H = 15 :=
by sorry

end NUMINAMATH_GPT_hannah_age_l812_81276


namespace NUMINAMATH_GPT_system_equivalence_l812_81265

theorem system_equivalence (f g : ℝ → ℝ) (x : ℝ) (h1 : f x > 0) (h2 : g x > 0) : f x + g x > 0 :=
sorry

end NUMINAMATH_GPT_system_equivalence_l812_81265


namespace NUMINAMATH_GPT_min_value_inequality_l812_81200

theorem min_value_inequality (a b c : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h : a + b + c = 2) : 
  (1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a)) ≥ 27 / 8 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l812_81200


namespace NUMINAMATH_GPT_least_m_for_no_real_roots_l812_81202

theorem least_m_for_no_real_roots : ∃ (m : ℤ), (∀ (x : ℝ), 3 * x * (m * x + 6) - 2 * x^2 + 8 ≠ 0) ∧ m = 4 := 
sorry

end NUMINAMATH_GPT_least_m_for_no_real_roots_l812_81202


namespace NUMINAMATH_GPT_cost_per_square_meter_of_mat_l812_81249

theorem cost_per_square_meter_of_mat {L W E : ℝ} : 
  L = 20 → W = 15 → E = 57000 → (E / (L * W)) = 190 :=
by
  intros hL hW hE
  rw [hL, hW, hE]
  sorry

end NUMINAMATH_GPT_cost_per_square_meter_of_mat_l812_81249


namespace NUMINAMATH_GPT_roberto_outfits_l812_81248

-- Define the conditions
def trousers := 5
def shirts := 8
def jackets := 4

-- Define the total number of outfits
def total_outfits : ℕ := trousers * shirts * jackets

-- The theorem stating the actual problem and answer
theorem roberto_outfits : total_outfits = 160 :=
by
  -- skip the proof for now
  sorry

end NUMINAMATH_GPT_roberto_outfits_l812_81248


namespace NUMINAMATH_GPT_nina_money_l812_81226

theorem nina_money (W : ℝ) (h1 : W > 0) (h2 : 10 * W = 14 * (W - 1)) : 10 * W = 35 := by
  sorry

end NUMINAMATH_GPT_nina_money_l812_81226


namespace NUMINAMATH_GPT_intersection_complement_l812_81213

def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}
def compl_U_N : Set ℕ := {x ∈ U | x ∉ N}

theorem intersection_complement :
  M ∩ compl_U_N = {4} :=
by
  have h1 : compl_U_N = {2, 4, 8} := by sorry
  have h2 : M ∩ compl_U_N = {4} := by sorry
  exact h2

end NUMINAMATH_GPT_intersection_complement_l812_81213


namespace NUMINAMATH_GPT_find_monic_cubic_polynomial_with_root_l812_81261

-- Define the monic cubic polynomial
def Q (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 6

-- Define the root condition we need to prove
theorem find_monic_cubic_polynomial_with_root (a : ℝ) (ha : a = (5 : ℝ)^(1/3) + 1) : Q a = 0 :=
by
  -- Proof goes here (omitted)
  sorry

end NUMINAMATH_GPT_find_monic_cubic_polynomial_with_root_l812_81261


namespace NUMINAMATH_GPT_james_gave_away_one_bag_l812_81222

theorem james_gave_away_one_bag (initial_marbles : ℕ) (bags : ℕ) (marbles_left : ℕ) (h1 : initial_marbles = 28) (h2 : bags = 4) (h3 : marbles_left = 21) : (initial_marbles / bags) = (initial_marbles - marbles_left) / (initial_marbles / bags) :=
by
  sorry

end NUMINAMATH_GPT_james_gave_away_one_bag_l812_81222


namespace NUMINAMATH_GPT_find_x4_plus_y4_l812_81267

theorem find_x4_plus_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^4 + y^4 = 135.5 :=
by
  sorry

end NUMINAMATH_GPT_find_x4_plus_y4_l812_81267


namespace NUMINAMATH_GPT_arithmetic_calculation_l812_81257

theorem arithmetic_calculation : 3.5 * 0.3 + 1.2 * 0.4 = 1.53 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_calculation_l812_81257


namespace NUMINAMATH_GPT_minji_total_water_intake_l812_81228

variable (morning_water : ℝ)
variable (afternoon_water : ℝ)

theorem minji_total_water_intake (h_morning : morning_water = 0.26) (h_afternoon : afternoon_water = 0.37):
  morning_water + afternoon_water = 0.63 :=
sorry

end NUMINAMATH_GPT_minji_total_water_intake_l812_81228
