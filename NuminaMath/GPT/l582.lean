import Mathlib

namespace NUMINAMATH_GPT_rhombus_perimeter_l582_58289

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
    let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
    (4 * s) = 52 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_l582_58289


namespace NUMINAMATH_GPT_square_side_is_8_l582_58229

-- Definitions based on problem conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 16
def rectangle_area : ℝ := rectangle_width * rectangle_length

def square_side_length (s : ℝ) : Prop := s^2 = rectangle_area

-- The theorem we need to prove
theorem square_side_is_8 (s : ℝ) : square_side_length s → s = 8 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_square_side_is_8_l582_58229


namespace NUMINAMATH_GPT_volume_of_tetrahedron_l582_58251

theorem volume_of_tetrahedron 
(angle_ABC_BCD : Real := 45 * Real.pi / 180)
(area_ABC : Real := 150)
(area_BCD : Real := 90)
(length_BC : Real := 10) :
  let h := 2 * area_BCD / length_BC
  let height_perpendicular := h * Real.sin angle_ABC_BCD
  let volume := (1 / 3 : Real) * area_ABC * height_perpendicular
  volume = 450 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_tetrahedron_l582_58251


namespace NUMINAMATH_GPT_sum_exponents_binary_3400_l582_58257

theorem sum_exponents_binary_3400 : 
  ∃ (a b c d e : ℕ), 
    3400 = 2^a + 2^b + 2^c + 2^d + 2^e ∧ 
    a > b ∧ b > c ∧ c > d ∧ d > e ∧ 
    a + b + c + d + e = 38 :=
sorry

end NUMINAMATH_GPT_sum_exponents_binary_3400_l582_58257


namespace NUMINAMATH_GPT_inverse_proposition_is_false_l582_58256

theorem inverse_proposition_is_false (a : ℤ) (h : a = 6) : ¬ (|a| = 6 → a = 6) :=
sorry

end NUMINAMATH_GPT_inverse_proposition_is_false_l582_58256


namespace NUMINAMATH_GPT_two_absent_one_present_probability_l582_58248

-- Define the probabilities
def probability_absent_normal : ℚ := 1 / 15

-- Given that the absence rate on Monday increases by 10%
def monday_increase_factor : ℚ := 1.1

-- Calculate the probability of being absent on Monday
def probability_absent_monday : ℚ := probability_absent_normal * monday_increase_factor

-- Calculate the probability of being present on Monday
def probability_present_monday : ℚ := 1 - probability_absent_monday

-- Define the probability that exactly two students are absent and one present
def probability_two_absent_one_present : ℚ :=
  3 * (probability_absent_monday ^ 2) * probability_present_monday

-- Convert the probability to a percentage and round to the nearest tenth
def probability_as_percent : ℚ := round (probability_two_absent_one_present * 100 * 10) / 10

theorem two_absent_one_present_probability : probability_as_percent = 1.5 := by sorry

end NUMINAMATH_GPT_two_absent_one_present_probability_l582_58248


namespace NUMINAMATH_GPT_factorize_expression_l582_58208

theorem factorize_expression (x y : ℝ) : 4 * x^2 - 2 * x * y = 2 * x * (2 * x - y) := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l582_58208


namespace NUMINAMATH_GPT_minimize_f_l582_58220

noncomputable def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f (a : ℝ) : a = 82 / 43 :=
by
  sorry

end NUMINAMATH_GPT_minimize_f_l582_58220


namespace NUMINAMATH_GPT_john_tour_days_l582_58283

noncomputable def numberOfDaysInTourProgram (d e : ℕ) : Prop :=
  d * e = 800 ∧ (d + 7) * (e - 5) = 800

theorem john_tour_days :
  ∃ (d e : ℕ), numberOfDaysInTourProgram d e ∧ d = 28 :=
by
  sorry

end NUMINAMATH_GPT_john_tour_days_l582_58283


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_l582_58247

theorem arithmetic_sequence_a1 (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_inc : d > 0)
  (h_a3 : a 3 = 1)
  (h_a2a4 : a 2 * a 4 = 3 / 4) : 
  a 1 = 0 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1_l582_58247


namespace NUMINAMATH_GPT_linda_original_savings_l582_58228

theorem linda_original_savings (S : ℝ) (h1 : (2 / 3) * S + (1 / 3) * S = S) 
  (h2 : (1 / 3) * S = 250) : S = 750 :=
by sorry

end NUMINAMATH_GPT_linda_original_savings_l582_58228


namespace NUMINAMATH_GPT_evaluate_expression_l582_58235

theorem evaluate_expression : ((5^2 + 3)^2 - (5^2 - 3)^2)^3 = 27000000 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l582_58235


namespace NUMINAMATH_GPT_ellipse_with_foci_on_y_axis_l582_58221

theorem ellipse_with_foci_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1) ↔ (m > n ∧ n > 0) := 
sorry

end NUMINAMATH_GPT_ellipse_with_foci_on_y_axis_l582_58221


namespace NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l582_58265

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 4 then 7 else
  if n = 5 then 16 - 7 else sorry

-- Define the arithmetic sequence and the given conditions
theorem find_n_in_arithmetic_sequence (n : ℕ) (a : ℕ → ℕ) 
  (h1 : a 4 = 7) 
  (h2 : a 3 + a 6 = 16) 
  (h3 : a n = 31) :
  n = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l582_58265


namespace NUMINAMATH_GPT_smallest_units_C_union_D_l582_58230

-- Definitions for the sets C and D and their sizes
def C_units : ℝ := 25.5
def D_units : ℝ := 18.0

-- Definition stating the inclusion-exclusion principle for sets C and D
def C_union_D (C_units D_units C_intersection_units : ℝ) : ℝ :=
  C_units + D_units - C_intersection_units

-- Statement to prove the minimum units in C union D
theorem smallest_units_C_union_D : ∃ h, h ≤ C_union_D C_units D_units D_units ∧ h = 25.5 := by
  sorry

end NUMINAMATH_GPT_smallest_units_C_union_D_l582_58230


namespace NUMINAMATH_GPT_part1_part2_l582_58278

noncomputable def is_monotonically_increasing (f' : ℝ → ℝ) := ∀ x, f' x ≥ 0

noncomputable def is_monotonically_decreasing (f' : ℝ → ℝ) (I : Set ℝ) := ∀ x ∈ I, f' x ≤ 0

def f' (a x : ℝ) : ℝ := 3 * x ^ 2 - a

theorem part1 (a : ℝ) : 
  is_monotonically_increasing (f' a) ↔ a ≤ 0 := sorry

theorem part2 (a : ℝ) : 
  is_monotonically_decreasing (f' a) (Set.Ioo (-1 : ℝ) (1 : ℝ)) ↔ a ≥ 3 := sorry

end NUMINAMATH_GPT_part1_part2_l582_58278


namespace NUMINAMATH_GPT_cindy_added_pens_l582_58299

-- Definitions based on conditions:
def initial_pens : ℕ := 20
def mike_pens : ℕ := 22
def sharon_pens : ℕ := 19
def final_pens : ℕ := 65

-- Intermediate calculations:
def pens_after_mike : ℕ := initial_pens + mike_pens
def pens_after_sharon : ℕ := pens_after_mike - sharon_pens

-- Proof statement:
theorem cindy_added_pens : pens_after_sharon + 42 = final_pens :=
by
  sorry

end NUMINAMATH_GPT_cindy_added_pens_l582_58299


namespace NUMINAMATH_GPT_downstream_rate_l582_58227

/--  
A man's rowing conditions and rates:
- The man's upstream rate is U = 12 kmph.
- The man's rate in still water is S = 7 kmph.
- We need to prove that the man's downstream rate D is 14 kmph.
-/
theorem downstream_rate (U S D : ℝ) (hU : U = 12) (hS : S = 7) : D = 14 :=
by
  -- Proof to be filled here
  sorry

end NUMINAMATH_GPT_downstream_rate_l582_58227


namespace NUMINAMATH_GPT_isosceles_triangle_area_l582_58204

theorem isosceles_triangle_area 
  (x y : ℝ)
  (h_perimeter : 2*y + 2*x = 32)
  (h_height : ∃ h : ℝ, h = 8 ∧ y^2 = x^2 + h^2) :
  ∃ area : ℝ, area = 48 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l582_58204


namespace NUMINAMATH_GPT_length_of_purple_part_l582_58239

theorem length_of_purple_part (p : ℕ) (black : ℕ) (blue : ℕ) (total : ℕ) 
  (h1 : black = 2) (h2 : blue = 1) (h3 : total = 6) (h4 : p + black + blue = total) : 
  p = 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_purple_part_l582_58239


namespace NUMINAMATH_GPT_cost_of_remaining_ingredients_l582_58254

theorem cost_of_remaining_ingredients :
  let cocoa_required := 0.4
  let sugar_required := 0.6
  let cake_weight := 450
  let given_cocoa := 259
  let cost_per_lb_cocoa := 3.50
  let cost_per_lb_sugar := 0.80
  let total_cocoa_needed := cake_weight * cocoa_required
  let total_sugar_needed := cake_weight * sugar_required
  let remaining_cocoa := max 0 (total_cocoa_needed - given_cocoa)
  let remaining_sugar := total_sugar_needed
  let total_cost := remaining_cocoa * cost_per_lb_cocoa + remaining_sugar * cost_per_lb_sugar
  total_cost = 216 := by
  sorry

end NUMINAMATH_GPT_cost_of_remaining_ingredients_l582_58254


namespace NUMINAMATH_GPT_ratio_of_a_to_b_l582_58297

theorem ratio_of_a_to_b 
  (b c a : ℝ)
  (h1 : b / c = 1 / 5) 
  (h2 : a / c = 1 / 7.5) : 
  a / b = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_l582_58297


namespace NUMINAMATH_GPT_greater_savings_on_hat_l582_58233

theorem greater_savings_on_hat (savings_shoes spent_shoes savings_hat sale_price_hat : ℝ) 
  (h1 : savings_shoes = 3.75)
  (h2 : spent_shoes = 42.25)
  (h3 : savings_hat = 1.80)
  (h4 : sale_price_hat = 18.20) :
  ((savings_hat / (sale_price_hat + savings_hat)) * 100) > ((savings_shoes / (spent_shoes + savings_shoes)) * 100) :=
by
  sorry

end NUMINAMATH_GPT_greater_savings_on_hat_l582_58233


namespace NUMINAMATH_GPT_length_of_platform_l582_58215

theorem length_of_platform
  (length_train : ℝ)
  (speed_train_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_covered : ℝ)
  (conversion_factor : ℝ) :
  length_train = 250 →
  speed_train_kmph = 90 →
  time_seconds = 20 →
  distance_covered = (speed_train_kmph * 1000 / 3600) * time_seconds →
  conversion_factor = 1000 / 3600 →
  ∃ P : ℝ, distance_covered = length_train + P ∧ P = 250 :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l582_58215


namespace NUMINAMATH_GPT_complement_of_A_in_U_l582_58223

open Set

def U : Set ℕ := {x | x < 8}
def A : Set ℕ := {x | (x - 1) * (x - 3) * (x - 4) * (x - 7) = 0}

theorem complement_of_A_in_U : (U \ A) = {0, 2, 5, 6} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l582_58223


namespace NUMINAMATH_GPT_find_b_l582_58214

-- Variables representing the terms in the equations
variables (a b t : ℝ)

-- Conditions given in the problem
def cond1 : Prop := a - (t / 6) * b = 20
def cond2 : Prop := a - (t / 5) * b = -10
def t_value : Prop := t = 60

-- The theorem we need to prove
theorem find_b (H1 : cond1 a b t) (H2 : cond2 a b t) (H3 : t_value t) : b = 15 :=
by {
  -- Assuming the conditions are true
  sorry
}

end NUMINAMATH_GPT_find_b_l582_58214


namespace NUMINAMATH_GPT_anne_already_made_8_drawings_l582_58296

-- Define the conditions as Lean definitions
def num_markers : ℕ := 12
def drawings_per_marker : ℚ := 3 / 2 -- Equivalent to 1.5
def remaining_drawings : ℕ := 10

-- Calculate the total number of drawings Anne can make with her markers
def total_drawings : ℚ := num_markers * drawings_per_marker

-- Calculate the already made drawings
def already_made_drawings : ℚ := total_drawings - remaining_drawings

-- The theorem to prove
theorem anne_already_made_8_drawings : already_made_drawings = 8 := 
by 
  have h1 : total_drawings = 18 := by sorry -- Calculating total drawings as 18
  have h2 : already_made_drawings = 8 := by sorry -- Calculating already made drawings as total drawings minus remaining drawings
  exact h2

end NUMINAMATH_GPT_anne_already_made_8_drawings_l582_58296


namespace NUMINAMATH_GPT_sum_sequences_l582_58259

theorem sum_sequences : 
  (1 + 12 + 23 + 34 + 45) + (10 + 20 + 30 + 40 + 50) = 265 := by
  sorry

end NUMINAMATH_GPT_sum_sequences_l582_58259


namespace NUMINAMATH_GPT_proof_statement_l582_58238

def convert_base_9_to_10 (n : Nat) : Nat :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def convert_base_6_to_10 (n : Nat) : Nat :=
  2 * 6^2 + 2 * 6^1 + 1 * 6^0

def problem_statement : Prop :=
  convert_base_9_to_10 324 - convert_base_6_to_10 221 = 180

theorem proof_statement : problem_statement := 
  by
    sorry

end NUMINAMATH_GPT_proof_statement_l582_58238


namespace NUMINAMATH_GPT_not_prime_3999991_l582_58246

   theorem not_prime_3999991 : ¬ Nat.Prime 3999991 :=
   by
     -- Provide the factorization proof
     sorry
   
end NUMINAMATH_GPT_not_prime_3999991_l582_58246


namespace NUMINAMATH_GPT_bottles_per_case_l582_58224

theorem bottles_per_case (days: ℕ) (daily_intake: ℚ) (total_spent: ℚ) (case_cost: ℚ) (total_cases: ℕ) (total_bottles: ℕ) (B: ℕ) 
    (H1 : days = 240)
    (H2 : daily_intake = 1/2)
    (H3 : total_spent = 60)
    (H4 : case_cost = 12)
    (H5 : total_cases = total_spent / case_cost)
    (H6 : total_bottles = days * daily_intake)
    (H7 : B = total_bottles / total_cases) :
    B = 24 :=
by
    sorry

end NUMINAMATH_GPT_bottles_per_case_l582_58224


namespace NUMINAMATH_GPT_arithmetic_sequence_nth_term_639_l582_58234

theorem arithmetic_sequence_nth_term_639 :
  ∀ (x n : ℕ) (a₁ a₂ a₃ aₙ : ℤ),
  a₁ = 3 * x - 5 →
  a₂ = 7 * x - 17 →
  a₃ = 4 * x + 3 →
  aₙ = a₁ + (n - 1) * (a₂ - a₁) →
  aₙ = 4018 →
  n = 639 :=
by
  intros x n a₁ a₂ a₃ aₙ h₁ h₂ h₃ hₙ hₙ_eq
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_nth_term_639_l582_58234


namespace NUMINAMATH_GPT_consecutive_coeff_sum_l582_58282

theorem consecutive_coeff_sum (P : Polynomial ℕ) (hdeg : P.degree = 699)
  (hP : P.eval 1 ≤ 2022) :
  ∃ k : ℕ, k < 700 ∧ (P.coeff (k + 1) + P.coeff k) = 22 ∨
                    (P.coeff (k + 1) + P.coeff k) = 55 ∨
                    (P.coeff (k + 1) + P.coeff k) = 77 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_coeff_sum_l582_58282


namespace NUMINAMATH_GPT_einstein_needs_more_money_l582_58211

-- Definitions based on conditions
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.3
def soda_price : ℝ := 2
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def goal : ℝ := 500

-- Total amount raised calculation
def total_raised : ℝ :=
  (pizzas_sold * pizza_price) +
  (fries_sold * fries_price) +
  (sodas_sold * soda_price)

-- Proof statement
theorem einstein_needs_more_money : goal - total_raised = 258 :=
by
  sorry

end NUMINAMATH_GPT_einstein_needs_more_money_l582_58211


namespace NUMINAMATH_GPT_y_in_terms_of_x_l582_58213

theorem y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 4) : y = 4 - 3 * x := 
by
  sorry

end NUMINAMATH_GPT_y_in_terms_of_x_l582_58213


namespace NUMINAMATH_GPT_cost_for_paving_is_486_l582_58292

-- Definitions and conditions
def ratio_longer_side : ℝ := 4
def ratio_shorter_side : ℝ := 3
def diagonal : ℝ := 45
def cost_per_sqm : ℝ := 0.5 -- converting pence to pounds

-- Mathematical formulation
def longer_side (x : ℝ) : ℝ := ratio_longer_side * x
def shorter_side (x : ℝ) : ℝ := ratio_shorter_side * x
def area_of_rectangle (l w : ℝ) : ℝ := l * w
def cost_paving (area : ℝ) (cost_per_sqm : ℝ) : ℝ := area * cost_per_sqm

-- Main problem: given the conditions, prove that the cost is £486.
theorem cost_for_paving_is_486 (x : ℝ) 
  (h1 : (ratio_longer_side^2 + ratio_shorter_side^2) * x^2 = diagonal^2) :
  cost_paving (area_of_rectangle (longer_side x) (shorter_side x)) cost_per_sqm = 486 :=
by
  sorry

end NUMINAMATH_GPT_cost_for_paving_is_486_l582_58292


namespace NUMINAMATH_GPT_work_completion_l582_58268

theorem work_completion (a b : ℝ) 
  (h1 : a + b = 6) 
  (h2 : a = 10) : 
  a + b = 6 :=
by sorry

end NUMINAMATH_GPT_work_completion_l582_58268


namespace NUMINAMATH_GPT_ratio_area_rect_sq_l582_58209

/-- 
  Given:
  1. The longer side of rectangle R is 1.2 times the length of a side of square S.
  2. The shorter side of rectangle R is 0.85 times the length of a side of square S.
  Prove that the ratio of the area of rectangle R to the area of square S is 51/50.
-/
theorem ratio_area_rect_sq (s : ℝ) 
  (h1 : ∃ r1, r1 = 1.2 * s) 
  (h2 : ∃ r2, r2 = 0.85 * s) : 
  (1.2 * s * 0.85 * s) / (s * s) = 51 / 50 := 
by
  sorry

end NUMINAMATH_GPT_ratio_area_rect_sq_l582_58209


namespace NUMINAMATH_GPT_average_of_pqrs_l582_58250

variable (p q r s : ℝ)

theorem average_of_pqrs
  (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_of_pqrs_l582_58250


namespace NUMINAMATH_GPT_range_of_a12_l582_58222

variable (a : ℕ → ℝ)
variable (a1 d : ℝ)

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

variable (h_arith_seq : arithmetic_seq a a1 d)
variable (h_a8 : a 7 ≥ 15)
variable (h_a9 : a 8 ≤ 13)

theorem range_of_a12 : ∀ a1 d, (arithmetic_seq a a1 d) → (a 7 ≥ 15) → (a 8 ≤ 13) → (a 11 ≤ 7) :=
by
  intro a1 d h_arith_seq h_a8 h_a9
  sorry

end NUMINAMATH_GPT_range_of_a12_l582_58222


namespace NUMINAMATH_GPT_range_of_a_l582_58266

-- Lean statement that represents the proof problem
theorem range_of_a 
  (h1 : ∀ x y : ℝ, x^2 - 2 * x + Real.log (2 * y^2 - y) = 0 → x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0)
  (h2 : ∀ b : ℝ, 2 * b^2 - b > 0) :
  (∀ a : ℝ, x^2 - 2 * x + Real.log (2 * a^2 - a) = 0 → (- (1:ℝ) / 2) < a ∧ a < 0 ∨ (1 / 2) < a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l582_58266


namespace NUMINAMATH_GPT_people_after_second_turn_l582_58200

noncomputable def number_of_people_in_front_after_second_turn (formation_size : ℕ) (initial_people : ℕ) (first_turn_people : ℕ) : ℕ := 
  if formation_size = 9 ∧ initial_people = 2 ∧ first_turn_people = 4 then 6 else 0

theorem people_after_second_turn :
  number_of_people_in_front_after_second_turn 9 2 4 = 6 :=
by
  -- Prove the theorem using the conditions and given data
  sorry

end NUMINAMATH_GPT_people_after_second_turn_l582_58200


namespace NUMINAMATH_GPT_problem_solution_l582_58219

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n % 100) / 10) * 8 + (n % 10)

def base3_to_base10 (n : ℕ) : ℕ :=
  (n / 10) * 3 + (n % 10)

def base7_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 49 + ((n % 100) / 10) * 7 + (n % 10)

def base5_to_base10 (n : ℕ) : ℕ :=
  (n / 10) * 5 + (n % 10)

def expression_in_base10 : ℕ :=
  (base8_to_base10 254) / (base3_to_base10 13) + (base7_to_base10 232) / (base5_to_base10 32)

theorem problem_solution : expression_in_base10 = 35 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l582_58219


namespace NUMINAMATH_GPT_floor_e_equals_two_l582_58287

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end NUMINAMATH_GPT_floor_e_equals_two_l582_58287


namespace NUMINAMATH_GPT_platform_length_is_350_l582_58275

variables (L : ℕ)

def train_length := 300
def time_to_cross_pole := 18
def time_to_cross_platform := 39

-- Speed of the train when crossing the pole
def speed_cross_pole : ℚ := train_length / time_to_cross_pole

-- Speed of the train when crossing the platform
def speed_cross_platform (L : ℕ) : ℚ := (train_length + L) / time_to_cross_platform

-- The main goal is to prove that the length of the platform is 350 meters
theorem platform_length_is_350 (L : ℕ) (h : speed_cross_pole = speed_cross_platform L) : L = 350 := sorry

end NUMINAMATH_GPT_platform_length_is_350_l582_58275


namespace NUMINAMATH_GPT_find_x_value_l582_58203

theorem find_x_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 5 * x^2 + 15 * x * y = x^3 + 2 * x^2 * y + 3 * x * y^2) : x = 5 :=
sorry

end NUMINAMATH_GPT_find_x_value_l582_58203


namespace NUMINAMATH_GPT_drink_cost_l582_58269

/-- Wade has called into a rest stop and decides to get food for the road. 
  He buys a sandwich to eat now, one for the road, and one for the evening. 
  He also buys 2 drinks. Wade spends a total of $26 and the sandwiches 
  each cost $6. Prove that the drinks each cost $4. -/
theorem drink_cost (cost_sandwich : ℕ) (num_sandwiches : ℕ) (cost_total : ℕ) (num_drinks : ℕ) :
  cost_sandwich = 6 → num_sandwiches = 3 → cost_total = 26 → num_drinks = 2 → 
  ∃ (cost_drink : ℕ), cost_drink = 4 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_drink_cost_l582_58269


namespace NUMINAMATH_GPT_find_x_l582_58293

-- Define the conditions as variables and the target equation
variable (x : ℝ)

theorem find_x : 67 * x - 59 * x = 4828 → x = 603.5 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l582_58293


namespace NUMINAMATH_GPT_geometric_sequence_value_of_b_l582_58244

-- Definitions
def is_geometric_sequence (a b c : ℝ) := 
  ∃ r : ℝ, a * r = b ∧ b * r = c

-- Theorem statement
theorem geometric_sequence_value_of_b (b : ℝ) (h : b > 0) 
  (h_seq : is_geometric_sequence 15 b 1) : b = Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_value_of_b_l582_58244


namespace NUMINAMATH_GPT_least_alpha_prime_l582_58298

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_distinct_prime (α β : ℕ) : Prop :=
  α ≠ β ∧ is_prime α ∧ is_prime β

theorem least_alpha_prime (α : ℕ) :
  is_distinct_prime α (180 - 2 * α) → α ≥ 41 :=
sorry

end NUMINAMATH_GPT_least_alpha_prime_l582_58298


namespace NUMINAMATH_GPT_new_paint_intensity_l582_58280

def I1 : ℝ := 0.50
def I2 : ℝ := 0.25
def F : ℝ := 0.2

theorem new_paint_intensity : (1 - F) * I1 + F * I2 = 0.45 := by
  sorry

end NUMINAMATH_GPT_new_paint_intensity_l582_58280


namespace NUMINAMATH_GPT_handshakes_meeting_l582_58249

theorem handshakes_meeting (x : ℕ) (h : x * (x - 1) / 2 = 66) : x = 12 := 
by 
  sorry

end NUMINAMATH_GPT_handshakes_meeting_l582_58249


namespace NUMINAMATH_GPT_book_total_pages_l582_58212

theorem book_total_pages (P : ℕ) (days_read : ℕ) (pages_per_day : ℕ) (fraction_read : ℚ) 
  (total_pages_read : ℕ) :
  (days_read = 15 ∧ pages_per_day = 12 ∧ fraction_read = 3 / 4 ∧ total_pages_read = 180 ∧ 
    total_pages_read = days_read * pages_per_day ∧ total_pages_read = fraction_read * P) → 
    P = 240 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_book_total_pages_l582_58212


namespace NUMINAMATH_GPT_trig_identity_A_trig_identity_D_l582_58294

theorem trig_identity_A : 
  (Real.tan (25 * Real.pi / 180) + Real.tan (20 * Real.pi / 180) + Real.tan (25 * Real.pi / 180) * Real.tan (20 * Real.pi / 180) = 1) :=
by sorry

theorem trig_identity_D : 
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180) = 4) :=
by sorry

end NUMINAMATH_GPT_trig_identity_A_trig_identity_D_l582_58294


namespace NUMINAMATH_GPT_tanya_efficiency_greater_sakshi_l582_58286

theorem tanya_efficiency_greater_sakshi (S_e T_e : ℝ) (h1 : S_e = 1 / 20) (h2 : T_e = 1 / 16) :
  ((T_e - S_e) / S_e) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_tanya_efficiency_greater_sakshi_l582_58286


namespace NUMINAMATH_GPT_find_fraction_l582_58274

noncomputable def fraction_of_third (F N : ℝ) : Prop := F * (1 / 3 * N) = 30

noncomputable def fraction_of_number (G N : ℝ) : Prop := G * N = 75

noncomputable def product_is_90 (F N : ℝ) : Prop := F * N = 90

theorem find_fraction (F G N : ℝ) (h1 : fraction_of_third F N) (h2 : fraction_of_number G N) (h3 : product_is_90 F N) :
  G = 5 / 6 :=
sorry

end NUMINAMATH_GPT_find_fraction_l582_58274


namespace NUMINAMATH_GPT_class_average_score_l582_58264

theorem class_average_score :
  let total_students := 40
  let absent_students := 2
  let present_students := total_students - absent_students
  let initial_avg := 92
  let absent_scores := [100, 100]
  let initial_total_score := initial_avg * present_students
  let total_final_score := initial_total_score + absent_scores.sum
  let final_avg := total_final_score / total_students
  final_avg = 92.4 := by
  sorry

end NUMINAMATH_GPT_class_average_score_l582_58264


namespace NUMINAMATH_GPT_find_C_and_D_l582_58276

theorem find_C_and_D (C D : ℚ) (h1 : 5 * C + 3 * D - 4 = 47) (h2 : C = D + 2) : 
  C = 57 / 8 ∧ D = 41 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_find_C_and_D_l582_58276


namespace NUMINAMATH_GPT_pattern_E_cannot_be_formed_l582_58263

-- Define the basic properties of the tile and the patterns
inductive Tile
| rhombus (diag_coloring : Bool) -- representing black-and-white diagonals

inductive Pattern
| optionA
| optionB
| optionC
| optionD
| optionE

-- The given tile is a rhombus with a certain coloring scheme
def given_tile : Tile := Tile.rhombus true

-- The statement to prove
theorem pattern_E_cannot_be_formed : 
  ¬ (∃ f : Pattern → Tile, f Pattern.optionE = given_tile) :=
sorry

end NUMINAMATH_GPT_pattern_E_cannot_be_formed_l582_58263


namespace NUMINAMATH_GPT_train_stops_time_l582_58210

/-- Given the speeds of a train excluding and including stoppages, 
calculate the stopping time in minutes per hour. --/
theorem train_stops_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 48)
  (h2 : speed_including_stoppages = 40) :
  ∃ minutes_stopped : ℝ, minutes_stopped = 10 :=
by
  sorry

end NUMINAMATH_GPT_train_stops_time_l582_58210


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_l582_58226

-- Definitions based on the conditions
variables (a b : ℝ)
def roots_real (a b : ℝ) : Prop := a^2 ≥ 8 * b ∧ b^2 ≥ a
def positive_vars (a b : ℝ) : Prop := a > 0 ∧ b > 0
def min_a_plus_b (a b : ℝ) : Prop := a + b = 6

-- Lean theorem statement
theorem min_value_of_a_plus_b (a b : ℝ) (hr : roots_real a b) (pv : positive_vars a b) : min_a_plus_b a b :=
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_b_l582_58226


namespace NUMINAMATH_GPT_fill_grid_power_of_two_l582_58260

theorem fill_grid_power_of_two (n : ℕ) (h : ∃ m : ℕ, n = 2^m) :
  ∃ f : ℕ → ℕ → ℕ, 
    (∀ i j : ℕ, i < n → j < n → 1 ≤ f i j ∧ f i j ≤ 2 * n - 1) ∧
    (∀ k, 1 ≤ k ∧ k ≤ n → (∀ i, i < n → ∀ j, j < n → i ≠ j → f i k ≠ f j k))
:= by
  sorry

end NUMINAMATH_GPT_fill_grid_power_of_two_l582_58260


namespace NUMINAMATH_GPT_stella_weeks_l582_58243

-- Define the constants used in the conditions
def rolls_per_bathroom_per_day : ℕ := 1
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def rolls_per_pack : ℕ := 12
def packs_bought : ℕ := 14

-- Define the total number of rolls Stella uses per day and per week
def rolls_per_day := rolls_per_bathroom_per_day * bathrooms
def rolls_per_week := rolls_per_day * days_per_week

-- Calculate the total number of rolls bought
def total_rolls_bought := packs_bought * rolls_per_pack

-- Calculate the number of weeks Stella bought toilet paper for
def weeks := total_rolls_bought / rolls_per_week

theorem stella_weeks : weeks = 4 := by
  sorry

end NUMINAMATH_GPT_stella_weeks_l582_58243


namespace NUMINAMATH_GPT_dan_present_age_l582_58261

-- Let x be Dan's present age
variable (x : ℤ)

-- Condition: Dan's age after 18 years will be 8 times his age 3 years ago
def condition (x : ℤ) : Prop :=
  x + 18 = 8 * (x - 3)

-- The goal is to prove that Dan's present age is 6
theorem dan_present_age (x : ℤ) (h : condition x) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_dan_present_age_l582_58261


namespace NUMINAMATH_GPT_probability_three_aligned_l582_58232

theorem probability_three_aligned (total_arrangements favorable_arrangements : ℕ) 
  (h1 : total_arrangements = 126)
  (h2 : favorable_arrangements = 48) :
  (favorable_arrangements : ℚ) / total_arrangements = 8 / 21 :=
by sorry

end NUMINAMATH_GPT_probability_three_aligned_l582_58232


namespace NUMINAMATH_GPT_incorrect_counting_of_students_l582_58245

open Set

theorem incorrect_counting_of_students
  (total_students : ℕ)
  (english_only : ℕ)
  (german_only : ℕ)
  (french_only : ℕ)
  (english_german : ℕ)
  (english_french : ℕ)
  (german_french : ℕ)
  (all_three : ℕ)
  (reported_total : ℕ)
  (h_total_students : total_students = 100)
  (h_english_only : english_only = 30)
  (h_german_only : german_only = 23)
  (h_french_only : french_only = 50)
  (h_english_german : english_german = 10)
  (h_english_french : english_french = 8)
  (h_german_french : german_french = 20)
  (h_all_three : all_three = 5)
  (h_reported_total : reported_total = 100) :
  (english_only + german_only + french_only + english_german +
   english_french + german_french - 2 * all_three) ≠ reported_total :=
by
  sorry

end NUMINAMATH_GPT_incorrect_counting_of_students_l582_58245


namespace NUMINAMATH_GPT_cole_time_to_work_is_90_minutes_l582_58206

noncomputable def cole_drive_time_to_work (D : ℝ) : ℝ := D / 30

def cole_trip_proof : Prop :=
  ∃ (D : ℝ), (D / 30) + (D / 90) = 2 ∧ cole_drive_time_to_work D * 60 = 90

theorem cole_time_to_work_is_90_minutes : cole_trip_proof :=
  sorry

end NUMINAMATH_GPT_cole_time_to_work_is_90_minutes_l582_58206


namespace NUMINAMATH_GPT_sequence_general_term_l582_58295

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n / (a n + 3)) :
  ∀ n : ℕ, n > 0 → a n = 3 / (n + 2) := 
by 
  sorry

end NUMINAMATH_GPT_sequence_general_term_l582_58295


namespace NUMINAMATH_GPT_min_button_presses_l582_58258

theorem min_button_presses :
  ∃ (a b : ℤ), 9 * a - 20 * b = 13 ∧  a + b = 24 := 
by
  sorry

end NUMINAMATH_GPT_min_button_presses_l582_58258


namespace NUMINAMATH_GPT_ratio_equality_l582_58279

def op_def (a b : ℕ) : ℕ := a * b + b^2
def ot_def (a b : ℕ) : ℕ := a - b + a * b^2

theorem ratio_equality : (op_def 8 3 : ℚ) / (ot_def 8 3 : ℚ) = (33 : ℚ) / 77 := by
  sorry

end NUMINAMATH_GPT_ratio_equality_l582_58279


namespace NUMINAMATH_GPT_sum_of_ages_l582_58202

theorem sum_of_ages (A B C : ℕ)
  (h1 : A = C + 8)
  (h2 : A + 10 = 3 * (C - 6))
  (h3 : B = 2 * C) :
  A + B + C = 80 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_ages_l582_58202


namespace NUMINAMATH_GPT_sum_max_min_ratios_l582_58205

theorem sum_max_min_ratios
  (c d : ℚ)
  (h1 : ∀ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 → y / x = c ∨ y / x = d)
  (h2 : ∀ r : ℚ, (∃ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 ∧ y / x = r) → (r = c ∨ r = d))
  : c + d = 63 / 43 :=
sorry

end NUMINAMATH_GPT_sum_max_min_ratios_l582_58205


namespace NUMINAMATH_GPT_crow_eats_nuts_l582_58241

theorem crow_eats_nuts (time_fifth_nuts : ℕ) (time_quarter_nuts : ℕ) (h : time_fifth_nuts = 8) :
  time_quarter_nuts = 10 :=
sorry

end NUMINAMATH_GPT_crow_eats_nuts_l582_58241


namespace NUMINAMATH_GPT_triangle_XYZ_ratio_l582_58291

theorem triangle_XYZ_ratio (XZ YZ : ℝ)
  (hXZ : XZ = 9) (hYZ : YZ = 40)
  (XY : ℝ) (hXY : XY = Real.sqrt (XZ ^ 2 + YZ ^ 2))
  (ZD : ℝ) (hZD : ZD = Real.sqrt (XZ * YZ))
  (XJ YJ : ℝ) (hXJ : XJ = Real.sqrt (XZ * (XZ + 2 * ZD)))
  (hYJ : YJ = Real.sqrt (YZ * (YZ + 2 * ZD)))
  (ratio : ℝ) (h_ratio : ratio = (XJ + YJ + XY) / XY) :
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ ratio = p / q ∧ p + q = 203 := sorry

end NUMINAMATH_GPT_triangle_XYZ_ratio_l582_58291


namespace NUMINAMATH_GPT_palace_to_airport_distance_l582_58231

-- Let I be the distance from the palace to the airport
-- Let v be the speed of the Emir's car
-- Let t be the time taken to travel from the palace to the airport

theorem palace_to_airport_distance (v t I : ℝ) 
    (h1 : v = I / t) 
    (h2 : v + 20 = I / (t - 2 / 60)) 
    (h3 : v - 20 = I / (t + 3 / 60)) : 
    I = 20 := by
  sorry

end NUMINAMATH_GPT_palace_to_airport_distance_l582_58231


namespace NUMINAMATH_GPT_negation_example_l582_58237

theorem negation_example :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l582_58237


namespace NUMINAMATH_GPT_speed_in_still_water_l582_58281

-- Definitions for the conditions
def upstream_speed : ℕ := 30
def downstream_speed : ℕ := 60

-- Prove that the speed of the man in still water is 45 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 45 := by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l582_58281


namespace NUMINAMATH_GPT_fraction_of_short_students_l582_58252

theorem fraction_of_short_students 
  (total_students tall_students average_students : ℕ) 
  (htotal : total_students = 400) 
  (htall : tall_students = 90) 
  (haverage : average_students = 150) : 
  (total_students - (tall_students + average_students)) / total_students = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_short_students_l582_58252


namespace NUMINAMATH_GPT_every_positive_integer_sum_of_distinct_powers_of_3_4_7_l582_58240

theorem every_positive_integer_sum_of_distinct_powers_of_3_4_7 :
  ∀ n : ℕ, n > 0 →
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  ∃ (i j k : ℕ), n = 3^i + 4^j + 7^k :=
by
  sorry

end NUMINAMATH_GPT_every_positive_integer_sum_of_distinct_powers_of_3_4_7_l582_58240


namespace NUMINAMATH_GPT_find_base_tax_rate_l582_58216

noncomputable def income : ℝ := 10550
noncomputable def tax_paid : ℝ := 950
noncomputable def base_income : ℝ := 5000
noncomputable def excess_income : ℝ := income - base_income
noncomputable def excess_tax_rate : ℝ := 0.10

theorem find_base_tax_rate (base_tax_rate: ℝ) :
  base_tax_rate * base_income + excess_tax_rate * excess_income = tax_paid -> 
  base_tax_rate = 7.9 / 100 :=
by sorry

end NUMINAMATH_GPT_find_base_tax_rate_l582_58216


namespace NUMINAMATH_GPT_oil_bill_january_l582_58270

-- Define the problem in Lean
theorem oil_bill_january (F J : ℝ) (h1 : F / J = 3 / 2) (h2 : (F + 30) / J = 5 / 3) : J = 180 := 
sorry

end NUMINAMATH_GPT_oil_bill_january_l582_58270


namespace NUMINAMATH_GPT_expressions_equal_iff_conditions_l582_58273

theorem expressions_equal_iff_conditions (a b c : ℝ) :
  (2 * a + 3 * b * c = (a + 2 * b) * (2 * a + 3 * c)) ↔ (a = 0 ∨ a + 2 * b + 1.5 * c = 0) :=
by
  sorry

end NUMINAMATH_GPT_expressions_equal_iff_conditions_l582_58273


namespace NUMINAMATH_GPT_janice_walk_dog_more_than_homework_l582_58217

theorem janice_walk_dog_more_than_homework 
  (H C T: Nat) 
  (W: Nat) 
  (total_time remaining_time spent_time: Nat) 
  (hw_time room_time trash_time extra_time: Nat)
  (H_eq : H = 30)
  (C_eq : C = H / 2)
  (T_eq : T = H / 6)
  (remaining_time_eq : remaining_time = 35)
  (total_time_eq : total_time = 120)
  (spent_time_eq : spent_time = total_time - remaining_time)
  (task_time_sum_eq : task_time_sum = H + C + T)
  (W_eq : W = spent_time - task_time_sum)
  : W - H = 5 := 
sorry

end NUMINAMATH_GPT_janice_walk_dog_more_than_homework_l582_58217


namespace NUMINAMATH_GPT_unattainable_y_l582_58255

theorem unattainable_y (x : ℝ) (hx : x ≠ -2 / 3) : ¬ (∃ x, y = (x - 3) / (3 * x + 2) ∧ y = 1 / 3) := by
  sorry

end NUMINAMATH_GPT_unattainable_y_l582_58255


namespace NUMINAMATH_GPT_perpendicular_angles_l582_58218

theorem perpendicular_angles (α : ℝ) 
  (h1 : 4 * Real.pi < α) 
  (h2 : α < 6 * Real.pi)
  (h3 : ∃ (k : ℤ), α = -2 * Real.pi / 3 + Real.pi / 2 + k * Real.pi) :
  α = 29 * Real.pi / 6 ∨ α = 35 * Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_angles_l582_58218


namespace NUMINAMATH_GPT_minimum_height_l582_58267

theorem minimum_height (x : ℝ) (h : ℝ) (A : ℝ) :
  (h = x + 4) →
  (A = 6*x^2 + 16*x) →
  (A ≥ 120) →
  (x ≥ 2) →
  h = 6 :=
by
  intros h_def A_def A_geq min_x
  sorry

end NUMINAMATH_GPT_minimum_height_l582_58267


namespace NUMINAMATH_GPT_difference_between_median_and_mean_is_five_l582_58284

noncomputable def mean_score : ℝ :=
  0.20 * 60 + 0.20 * 75 + 0.40 * 85 + 0.20 * 95

noncomputable def median_score : ℝ := 85

theorem difference_between_median_and_mean_is_five :
  abs (median_score - mean_score) = 5 :=
by
  unfold mean_score median_score
  -- median_score - mean_score = 85 - 80
  -- thus the absolute value of the difference is 5
  sorry

end NUMINAMATH_GPT_difference_between_median_and_mean_is_five_l582_58284


namespace NUMINAMATH_GPT_sam_initial_balloons_l582_58262

theorem sam_initial_balloons:
  ∀ (S : ℝ), (S - 5.0 + 7.0 = 8) → S = 6.0 :=
by
  intro S h
  sorry

end NUMINAMATH_GPT_sam_initial_balloons_l582_58262


namespace NUMINAMATH_GPT_expression_evaluate_l582_58207

theorem expression_evaluate :
  50 * (50 - 5) - (50 * 50 - 5) = -245 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluate_l582_58207


namespace NUMINAMATH_GPT_pentagon_ABEDF_area_l582_58272

theorem pentagon_ABEDF_area (BD_diagonal : ∀ (ABCD : Nat) (BD : Nat),
                            ABCD = BD^2 / 2 → BD = 20) 
                            (BDFE_is_rectangle : ∀ (BDFE : Nat), BDFE = 2 * BD) 
                            : ∃ (area : Nat), area = 300 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_pentagon_ABEDF_area_l582_58272


namespace NUMINAMATH_GPT_sequence_formula_l582_58236

noncomputable def seq (a : ℕ+ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, (a n - 3) * a (n + 1) - a n + 4 = 0

theorem sequence_formula (a : ℕ+ → ℚ) (h : seq a) :
  ∀ n : ℕ+, a n = (2 * n - 1) / n :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l582_58236


namespace NUMINAMATH_GPT_smartphone_price_l582_58253

/-
Question: What is the sticker price of the smartphone, given the following conditions?
Conditions:
1: Store A offers a 20% discount on the sticker price, followed by a $120 rebate. Prices include an 8% sales tax applied after all discounts and fees.
2: Store B offers a 30% discount on the sticker price but adds a $50 handling fee. Prices include an 8% sales tax applied after all discounts and fees.
3: Natalie saves $27 by purchasing the smartphone at store A instead of store B.

Proof Problem:
Prove that given the above conditions, the sticker price of the smartphone is $1450.
-/

theorem smartphone_price (p : ℝ) :
  (1.08 * (0.7 * p + 50) - 1.08 * (0.8 * p - 120)) = 27 ->
  p = 1450 :=
by
  sorry

end NUMINAMATH_GPT_smartphone_price_l582_58253


namespace NUMINAMATH_GPT_determine_d_l582_58225

theorem determine_d (u v d c : ℝ) (p q : ℝ → ℝ)
  (hp : ∀ x, p x = x^3 + c * x + d)
  (hq : ∀ x, q x = x^3 + c * x + d + 300)
  (huv : p u = 0 ∧ p v = 0)  
  (hu5_v4 : q (u + 5) = 0 ∧ q (v - 4) = 0)
  (sum_roots_p : u + v + (-u - v) = 0)
  (sum_roots_q : (u + 5) + (v - 4) + (-u - v - 1) = 0)
  : d = -4 ∨ d = 6 :=
sorry

end NUMINAMATH_GPT_determine_d_l582_58225


namespace NUMINAMATH_GPT_contrapositive_inequality_l582_58285

theorem contrapositive_inequality (a b : ℝ) :
  (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) := by
sorry

end NUMINAMATH_GPT_contrapositive_inequality_l582_58285


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_n_squared_l582_58201

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_mean (x y z : ℝ) : Prop :=
(y * y = x * z)

def is_strictly_increasing (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) > a n

theorem arithmetic_sequence_sum_n_squared
  (a : ℕ → ℝ)
  (h₁ : is_arithmetic_sequence a)
  (h₂ : a 1 = 1)
  (h₃ : is_geometric_mean (a 1) (a 2) (a 5))
  (h₄ : is_strictly_increasing a) :
  ∃ S : ℕ → ℝ, ∀ n : ℕ, S n = n ^ 2 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_n_squared_l582_58201


namespace NUMINAMATH_GPT_quadratic_complete_square_l582_58290

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 2 * x + 3 = (x - 1)^2 + 2) := 
by
  intro x
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l582_58290


namespace NUMINAMATH_GPT_find_number_l582_58288

theorem find_number (x : ℝ) : 0.5 * 56 = 0.3 * x + 13 ↔ x = 50 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_number_l582_58288


namespace NUMINAMATH_GPT_proportion_is_equation_l582_58277

/-- A proportion containing unknowns is an equation -/
theorem proportion_is_equation (P : Prop) (contains_equality_sign: Prop)
  (indicates_equality : Prop)
  (contains_unknowns : Prop) : (contains_equality_sign ∧ indicates_equality ∧ contains_unknowns ↔ True) := by
  sorry

end NUMINAMATH_GPT_proportion_is_equation_l582_58277


namespace NUMINAMATH_GPT_number_of_elements_l582_58271

noncomputable def set_mean (S : Set ℝ) : ℝ := sorry

theorem number_of_elements (S : Set ℝ) (M : ℝ)
  (h1 : set_mean (S ∪ {15}) = M + 2)
  (h2 : set_mean (S ∪ {15, 1}) = M + 1) :
  ∃ k : ℕ, (M * k + 15 = (M + 2) * (k + 1)) ∧ (M * k + 16 = (M + 1) * (k + 2)) ∧ k = 4 := sorry

end NUMINAMATH_GPT_number_of_elements_l582_58271


namespace NUMINAMATH_GPT_fourth_game_water_correct_fourth_game_sports_drink_l582_58242

noncomputable def total_bottled_water_cases : ℕ := 10
noncomputable def total_sports_drink_cases : ℕ := 5
noncomputable def bottles_per_case_water : ℕ := 20
noncomputable def bottles_per_case_sports_drink : ℕ := 15
noncomputable def initial_bottled_water : ℕ := total_bottled_water_cases * bottles_per_case_water
noncomputable def initial_sports_drinks : ℕ := total_sports_drink_cases * bottles_per_case_sports_drink

noncomputable def first_game_water : ℕ := 70
noncomputable def first_game_sports_drink : ℕ := 30
noncomputable def second_game_water : ℕ := 40
noncomputable def second_game_sports_drink : ℕ := 20
noncomputable def third_game_water : ℕ := 50
noncomputable def third_game_sports_drink : ℕ := 25

noncomputable def total_consumed_water : ℕ := first_game_water + second_game_water + third_game_water
noncomputable def total_consumed_sports_drink : ℕ := first_game_sports_drink + second_game_sports_drink + third_game_sports_drink

noncomputable def remaining_water_before_fourth_game : ℕ := initial_bottled_water - total_consumed_water
noncomputable def remaining_sports_drink_before_fourth_game : ℕ := initial_sports_drinks - total_consumed_sports_drink

noncomputable def remaining_water_after_fourth_game : ℕ := 20
noncomputable def remaining_sports_drink_after_fourth_game : ℕ := 10

noncomputable def fourth_game_water_consumed : ℕ := remaining_water_before_fourth_game - remaining_water_after_fourth_game

theorem fourth_game_water_correct : fourth_game_water_consumed = 20 :=
by
  unfold fourth_game_water_consumed remaining_water_before_fourth_game
  sorry

theorem fourth_game_sports_drink : false :=
by
  sorry

end NUMINAMATH_GPT_fourth_game_water_correct_fourth_game_sports_drink_l582_58242
