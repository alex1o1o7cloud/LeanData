import Mathlib

namespace NUMINAMATH_GPT_system_of_linear_equations_l1192_119263

-- Define the system of linear equations and a lemma stating the given conditions and the proof goals.
theorem system_of_linear_equations (x y m : ℚ) :
  (x + 3 * y = 7) ∧ (2 * x - 3 * y = 2) ∧ (x - 3 * y + m * x + 3 = 0) ↔ 
  (x = 4 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ m = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_system_of_linear_equations_l1192_119263


namespace NUMINAMATH_GPT_two_pow_a_add_three_pow_b_eq_square_l1192_119218

theorem two_pow_a_add_three_pow_b_eq_square (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) 
(h : 2 ^ a + 3 ^ b = n ^ 2) : (a = 4 ∧ b = 2) :=
sorry

end NUMINAMATH_GPT_two_pow_a_add_three_pow_b_eq_square_l1192_119218


namespace NUMINAMATH_GPT_union_A_B_equals_C_l1192_119209

-- Define Set A
def A : Set ℝ := {x : ℝ | 3 - 2 * x > 0}

-- Define Set B
def B : Set ℝ := {x : ℝ | x^2 ≤ 4}

-- Define the target set C which is supposed to be A ∪ B
def C : Set ℝ := {x : ℝ | x ≤ 2}

theorem union_A_B_equals_C : A ∪ B = C := by 
  -- Proof is omitted here
  sorry

end NUMINAMATH_GPT_union_A_B_equals_C_l1192_119209


namespace NUMINAMATH_GPT_greatest_possible_NPMPP_l1192_119281

theorem greatest_possible_NPMPP :
  ∃ (M N P PP : ℕ),
    0 ≤ M ∧ M ≤ 9 ∧
    M^2 % 10 = M ∧
    NPMPP = M * (1111 * M) ∧
    NPMPP = 89991 := by
  sorry

end NUMINAMATH_GPT_greatest_possible_NPMPP_l1192_119281


namespace NUMINAMATH_GPT_acute_angled_triangle_count_l1192_119204

def num_vertices := 8

def total_triangles := Nat.choose num_vertices 3

def right_angled_triangles := 8 * 6

def acute_angled_triangles := total_triangles - right_angled_triangles

theorem acute_angled_triangle_count : acute_angled_triangles = 8 :=
by
  sorry

end NUMINAMATH_GPT_acute_angled_triangle_count_l1192_119204


namespace NUMINAMATH_GPT_oil_bill_additional_amount_l1192_119233

variables (F JanuaryBill : ℝ) (x : ℝ)

-- Given conditions
def condition1 : Prop := F / JanuaryBill = 5 / 4
def condition2 : Prop := (F + x) / JanuaryBill = 3 / 2
def JanuaryBillVal : Prop := JanuaryBill = 180

-- The theorem to prove
theorem oil_bill_additional_amount
  (h1 : condition1 F JanuaryBill)
  (h2 : condition2 F JanuaryBill x)
  (h3 : JanuaryBillVal JanuaryBill) :
  x = 45 := 
  sorry

end NUMINAMATH_GPT_oil_bill_additional_amount_l1192_119233


namespace NUMINAMATH_GPT_find_k_value_l1192_119275

theorem find_k_value (k : ℝ) : (∃ k, ∀ x y, y = k * x + 3 ∧ (x, y) = (1, 2)) → k = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l1192_119275


namespace NUMINAMATH_GPT_a_put_his_oxen_for_grazing_for_7_months_l1192_119215

theorem a_put_his_oxen_for_grazing_for_7_months
  (x : ℕ)
  (a_oxen : ℕ := 10)
  (b_oxen : ℕ := 12)
  (b_months : ℕ := 5)
  (c_oxen : ℕ := 15)
  (c_months : ℕ := 3)
  (total_rent : ℝ := 105)
  (c_share : ℝ := 27) :
  (c_share / total_rent = (c_oxen * c_months) / ((a_oxen * x) + (b_oxen * b_months) + (c_oxen * c_months))) → (x = 7) :=
by
  sorry

end NUMINAMATH_GPT_a_put_his_oxen_for_grazing_for_7_months_l1192_119215


namespace NUMINAMATH_GPT_num_people_end_race_l1192_119289

-- Define the conditions
def num_cars : ℕ := 20
def initial_passengers_per_car : ℕ := 2
def drivers_per_car : ℕ := 1
def additional_passengers_per_car : ℕ := 1

-- Define the total number of people in a car at the start
def total_people_per_car_initial := initial_passengers_per_car + drivers_per_car

-- Define the total number of people in a car after halfway point
def total_people_per_car_end := total_people_per_car_initial + additional_passengers_per_car

-- Define the total number of people in all cars at the end
def total_people_end := num_cars * total_people_per_car_end

-- Theorem statement
theorem num_people_end_race : total_people_end = 80 := by
  sorry

end NUMINAMATH_GPT_num_people_end_race_l1192_119289


namespace NUMINAMATH_GPT_positive_function_characterization_l1192_119244

theorem positive_function_characterization (f : ℝ → ℝ) (h₁ : ∀ x, x > 0 → f x > 0) (h₂ : ∀ a b : ℝ, a > 0 → b > 0 → a * b ≤ 0.5 * (a * f a + b * (f b)⁻¹)) :
  ∃ C > 0, ∀ x > 0, f x = C * x :=
sorry

end NUMINAMATH_GPT_positive_function_characterization_l1192_119244


namespace NUMINAMATH_GPT_gear_teeth_count_l1192_119253

theorem gear_teeth_count 
  (x y z: ℕ) 
  (h1: x + y + z = 60) 
  (h2: 4 * x - 20 = 5 * y) 
  (h3: 5 * y = 10 * z):
  x = 30 ∧ y = 20 ∧ z = 10 :=
by
  sorry

end NUMINAMATH_GPT_gear_teeth_count_l1192_119253


namespace NUMINAMATH_GPT_difference_of_squirrels_and_nuts_l1192_119220

-- Definitions
def number_of_squirrels : ℕ := 4
def number_of_nuts : ℕ := 2

-- Theorem statement with conditions and conclusion
theorem difference_of_squirrels_and_nuts : number_of_squirrels - number_of_nuts = 2 := by
  sorry

end NUMINAMATH_GPT_difference_of_squirrels_and_nuts_l1192_119220


namespace NUMINAMATH_GPT_calculate_expression_l1192_119250

theorem calculate_expression (x : ℤ) (h : x^2 = 1681) : (x + 3) * (x - 3) = 1672 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1192_119250


namespace NUMINAMATH_GPT_relationship_between_y1_y2_l1192_119299

theorem relationship_between_y1_y2 (b y1 y2 : ℝ) 
  (h₁ : y1 = -(-1) + b) 
  (h₂ : y2 = -(2) + b) : 
  y1 > y2 := 
by 
  sorry

end NUMINAMATH_GPT_relationship_between_y1_y2_l1192_119299


namespace NUMINAMATH_GPT_triangle_inequality_l1192_119228

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b - c) * (a - b + c) * (-a + b + c) ≤ a * b * c := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l1192_119228


namespace NUMINAMATH_GPT_log_expression_eq_zero_l1192_119229

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_eq_zero : 2 * log_base 5 10 + log_base 5 0.25 = 0 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_eq_zero_l1192_119229


namespace NUMINAMATH_GPT_min_possible_frac_l1192_119278

theorem min_possible_frac (x A C : ℝ) (hx : x ≠ 0) (hC_pos : 0 < C) (hA_pos : 0 < A)
  (h1 : x^2 + (1/x)^2 = A)
  (h2 : x - 1/x = C)
  (hC : C = Real.sqrt 3):
  A / C = (5 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_GPT_min_possible_frac_l1192_119278


namespace NUMINAMATH_GPT_initial_girls_l1192_119292

theorem initial_girls (G : ℕ) 
  (h1 : G + 7 + (15 - 4) = 36) : G = 18 :=
by
  sorry

end NUMINAMATH_GPT_initial_girls_l1192_119292


namespace NUMINAMATH_GPT_determine_m_value_l1192_119259

theorem determine_m_value 
  (a b m : ℝ)
  (h1 : 2^a = m)
  (h2 : 5^b = m)
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := 
sorry

end NUMINAMATH_GPT_determine_m_value_l1192_119259


namespace NUMINAMATH_GPT_esperanza_savings_l1192_119214

-- Define the conditions as constants
def rent := 600
def gross_salary := 4840
def food_cost := (3 / 5) * rent
def mortgage_bill := 3 * food_cost
def total_expenses := rent + food_cost + mortgage_bill
def savings := gross_salary - total_expenses
def taxes := (2 / 5) * savings
def actual_savings := savings - taxes

theorem esperanza_savings : actual_savings = 1680 := by
  sorry

end NUMINAMATH_GPT_esperanza_savings_l1192_119214


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1192_119283

-- Step 1: Define the imaginary unit.
def i : ℂ := Complex.I  -- ℂ represents complex numbers in Lean and Complex.I is the imaginary unit.

-- Step 2: Define the complex number z.
noncomputable def z : ℂ := (4 - 3 * i) / i

-- Step 3: State the theorem.
theorem imaginary_part_of_z : Complex.im z = -4 :=
by 
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l1192_119283


namespace NUMINAMATH_GPT_sophie_marble_exchange_l1192_119237

theorem sophie_marble_exchange (sophie_initial_marbles joe_initial_marbles : ℕ) 
  (final_ratio : ℕ) (sophie_gives_joe : ℕ) : 
  sophie_initial_marbles = 120 → joe_initial_marbles = 19 → final_ratio = 3 → 
  (120 - sophie_gives_joe = 3 * (19 + sophie_gives_joe)) → sophie_gives_joe = 16 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sophie_marble_exchange_l1192_119237


namespace NUMINAMATH_GPT_third_bowler_points_162_l1192_119290

variable (x : ℕ)

def total_score (x : ℕ) : Prop :=
  let first_bowler_points := x
  let second_bowler_points := 3 * x
  let third_bowler_points := x
  first_bowler_points + second_bowler_points + third_bowler_points = 810

theorem third_bowler_points_162 (x : ℕ) (h : total_score x) : x = 162 := by
  sorry

end NUMINAMATH_GPT_third_bowler_points_162_l1192_119290


namespace NUMINAMATH_GPT_maximum_value_N_27_l1192_119293

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end NUMINAMATH_GPT_maximum_value_N_27_l1192_119293


namespace NUMINAMATH_GPT_eliza_is_shorter_by_2_inch_l1192_119231

theorem eliza_is_shorter_by_2_inch
  (total_height : ℕ)
  (height_sibling1 height_sibling2 height_sibling3 height_eliza : ℕ) :
  total_height = 330 →
  height_sibling1 = 66 →
  height_sibling2 = 66 →
  height_sibling3 = 60 →
  height_eliza = 68 →
  total_height - (height_sibling1 + height_sibling2 + height_sibling3 + height_eliza) - height_eliza = 2 :=
by
  sorry

end NUMINAMATH_GPT_eliza_is_shorter_by_2_inch_l1192_119231


namespace NUMINAMATH_GPT_symmetry_of_F_l1192_119246

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
|f x| + f (|x|)

theorem symmetry_of_F (f : ℝ → ℝ) (h : is_odd_function f) :
    ∀ x : ℝ, F f x = F f (-x) :=
by
  sorry

end NUMINAMATH_GPT_symmetry_of_F_l1192_119246


namespace NUMINAMATH_GPT_probability_factor_24_l1192_119249

theorem probability_factor_24 : 
  (∃ (k : ℚ), k = 1 / 3 ∧ 
  ∀ (n : ℕ), n ≤ 24 ∧ n > 0 → 
  (∃ (m : ℕ), 24 = m * n)) := sorry

end NUMINAMATH_GPT_probability_factor_24_l1192_119249


namespace NUMINAMATH_GPT_comparison_arctan_l1192_119230

theorem comparison_arctan (a b c : ℝ) (h : Real.arctan a + Real.arctan b + Real.arctan c + Real.pi / 2 = 0) :
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_comparison_arctan_l1192_119230


namespace NUMINAMATH_GPT_find_integer_solutions_l1192_119298

theorem find_integer_solutions :
  ∀ (x y : ℕ), 0 < x → 0 < y → (2 * x^2 + 5 * x * y + 2 * y^2 = 2006 ↔ (x = 28 ∧ y = 3) ∨ (x = 3 ∧ y = 28)) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_solutions_l1192_119298


namespace NUMINAMATH_GPT_total_cost_of_vacation_l1192_119226

noncomputable def total_cost (C : ℝ) : Prop :=
  let cost_per_person_three := C / 3
  let cost_per_person_four := C / 4
  cost_per_person_three - cost_per_person_four = 60

theorem total_cost_of_vacation (C : ℝ) (h : total_cost C) : C = 720 :=
  sorry

end NUMINAMATH_GPT_total_cost_of_vacation_l1192_119226


namespace NUMINAMATH_GPT_system1_solution_l1192_119252

theorem system1_solution (x y : ℝ) (h₁ : x = 2 * y) (h₂ : 3 * x - 2 * y = 8) : x = 4 ∧ y = 2 := 
by admit

end NUMINAMATH_GPT_system1_solution_l1192_119252


namespace NUMINAMATH_GPT_inv_g_inv_5_l1192_119206

noncomputable def g (x : ℝ) : ℝ := 25 / (2 + 5 * x)
noncomputable def g_inv (y : ℝ) : ℝ := (15 - 10) / 25  -- g^{-1}(5) as shown in the derivation above

theorem inv_g_inv_5 : (g_inv 5)⁻¹ = 5 / 3 := by
  have h_g_inv_5 : g_inv 5 = 3 / 5 := by sorry
  rw [h_g_inv_5]
  exact inv_div 3 5

end NUMINAMATH_GPT_inv_g_inv_5_l1192_119206


namespace NUMINAMATH_GPT_find_square_side_length_l1192_119267

/-- Define the side lengths of the rectangle and the square --/
def rectangle_side_lengths (k : ℕ) (n : ℕ) : Prop := 
  k ≥ 7 ∧ n = 12 ∧ k * (k - 7) = n * n

theorem find_square_side_length (k n : ℕ) : rectangle_side_lengths k n → n = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_square_side_length_l1192_119267


namespace NUMINAMATH_GPT_proof_tan_alpha_proof_exp_l1192_119294

-- Given conditions
variables (α : ℝ) (h_condition1 : Real.tan (α + Real.pi / 4) = - 1 / 2) (h_condition2 : Real.pi / 2 < α ∧ α < Real.pi)

-- To prove
theorem proof_tan_alpha :
  Real.tan α = -3 :=
sorry -- proof goes here

theorem proof_exp :
  (Real.sin (2 * α) - 2 * Real.cos α ^ 2) / Real.sin (α - Real.pi / 4) = - 2 * Real.sqrt 5 / 5 :=
sorry -- proof goes here

end NUMINAMATH_GPT_proof_tan_alpha_proof_exp_l1192_119294


namespace NUMINAMATH_GPT_average_age_after_swap_l1192_119280

theorem average_age_after_swap :
  let initial_average_age := 28
  let num_people_initial := 8
  let person_leaving_age := 20
  let person_entering_age := 25
  let initial_total_age := initial_average_age * num_people_initial
  let total_age_after_leaving := initial_total_age - person_leaving_age
  let total_age_final := total_age_after_leaving + person_entering_age
  let num_people_final := 8
  initial_average_age / num_people_initial = 28 ->
  total_age_final / num_people_final = 28.625 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_age_after_swap_l1192_119280


namespace NUMINAMATH_GPT_toothpicks_pattern_100th_stage_l1192_119266

theorem toothpicks_pattern_100th_stage :
  let a_1 := 5
  let d := 4
  let n := 100
  (a_1 + (n - 1) * d) = 401 := by
  sorry

end NUMINAMATH_GPT_toothpicks_pattern_100th_stage_l1192_119266


namespace NUMINAMATH_GPT_factorial_fraction_eq_seven_l1192_119235

theorem factorial_fraction_eq_seven : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_factorial_fraction_eq_seven_l1192_119235


namespace NUMINAMATH_GPT_equal_candy_distribution_l1192_119256

theorem equal_candy_distribution :
  ∀ (candies friends : ℕ), candies = 30 → friends = 4 → candies % friends = 2 :=
by
  sorry

end NUMINAMATH_GPT_equal_candy_distribution_l1192_119256


namespace NUMINAMATH_GPT_martin_and_martina_ages_l1192_119225

-- Conditions
def martin_statement (x y : ℕ) : Prop := x = 3 * (2 * y - x)
def martina_statement (x y : ℕ) : Prop := 3 * x - y = 77

-- Proof problem
theorem martin_and_martina_ages :
  ∃ (x y : ℕ), martin_statement x y ∧ martina_statement x y ∧ x = 33 ∧ y = 22 :=
by {
  -- No proof required, just the statement
  sorry
}

end NUMINAMATH_GPT_martin_and_martina_ages_l1192_119225


namespace NUMINAMATH_GPT_remainder_product_mod_5_l1192_119234

theorem remainder_product_mod_5 
  (a b c : ℕ) 
  (ha : a % 5 = 1) 
  (hb : b % 5 = 2) 
  (hc : c % 5 = 3) : 
  (a * b * c) % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_product_mod_5_l1192_119234


namespace NUMINAMATH_GPT_find_other_number_l1192_119279

theorem find_other_number (HCF LCM num1 num2 : ℕ) (h1 : HCF = 16) (h2 : LCM = 396) (h3 : num1 = 36) (h4 : HCF * LCM = num1 * num2) : num2 = 176 :=
sorry

end NUMINAMATH_GPT_find_other_number_l1192_119279


namespace NUMINAMATH_GPT_seventyFifthTermInSequence_l1192_119210

/-- Given a sequence that starts at 2 and increases by 4 each term, 
prove that the 75th term in this sequence is 298. -/
theorem seventyFifthTermInSequence : 
  (∃ a : ℕ → ℤ, (∀ n : ℕ, a n = 2 + 4 * n) ∧ a 74 = 298) :=
by
  sorry

end NUMINAMATH_GPT_seventyFifthTermInSequence_l1192_119210


namespace NUMINAMATH_GPT_base_digits_equality_l1192_119254

theorem base_digits_equality (b : ℕ) (h_condition : b^5 ≤ 200 ∧ 200 < b^6) : b = 2 := 
by {
  sorry -- proof not required as per the instructions
}

end NUMINAMATH_GPT_base_digits_equality_l1192_119254


namespace NUMINAMATH_GPT_lives_lost_l1192_119217

-- Conditions given in the problem
def initial_lives : ℕ := 83
def current_lives : ℕ := 70

-- Prove the number of lives lost
theorem lives_lost : initial_lives - current_lives = 13 :=
by
  sorry

end NUMINAMATH_GPT_lives_lost_l1192_119217


namespace NUMINAMATH_GPT_calc_g_3_l1192_119258

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x - 1

theorem calc_g_3 : g (g (g (g 3))) = 1 := by
  sorry

end NUMINAMATH_GPT_calc_g_3_l1192_119258


namespace NUMINAMATH_GPT_remainder_modulo_9_l1192_119264

noncomputable def power10 := 10^15
noncomputable def power3  := 3^15

theorem remainder_modulo_9 : (7 * power10 + power3) % 9 = 7 := by
  -- Define the conditions given in the problem
  have h1 : (10 % 9 = 1) := by 
    norm_num
  have h2 : (3^2 % 9 = 0) := by 
    norm_num
  
  -- Utilize these conditions to prove the statement
  sorry

end NUMINAMATH_GPT_remainder_modulo_9_l1192_119264


namespace NUMINAMATH_GPT_sqrt_neg4_squared_l1192_119269

theorem sqrt_neg4_squared : Real.sqrt ((-4 : ℝ) ^ 2) = 4 := 
by 
-- add proof here
sorry

end NUMINAMATH_GPT_sqrt_neg4_squared_l1192_119269


namespace NUMINAMATH_GPT_percentage_short_l1192_119208

def cost_of_goldfish : ℝ := 0.25
def sale_price_of_goldfish : ℝ := 0.75
def tank_price : ℝ := 100
def goldfish_sold : ℕ := 110

theorem percentage_short : ((tank_price - (sale_price_of_goldfish - cost_of_goldfish) * goldfish_sold) / tank_price) * 100 = 45 := 
by
  sorry

end NUMINAMATH_GPT_percentage_short_l1192_119208


namespace NUMINAMATH_GPT_product_remainder_mod_7_l1192_119282

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end NUMINAMATH_GPT_product_remainder_mod_7_l1192_119282


namespace NUMINAMATH_GPT_find_c_l1192_119272

theorem find_c (c : ℝ) 
  (h1 : ∃ x : ℝ, 3 * x^2 + 23 * x - 75 = 0 ∧ x = ⌊c⌋) 
  (h2 : ∃ y : ℝ, 4 * y^2 - 19 * y + 3 = 0 ∧ y = c - ⌊c⌋) : 
  c = -11.84 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1192_119272


namespace NUMINAMATH_GPT_mike_unbroken_seashells_l1192_119243

-- Define the conditions from the problem
def totalSeashells : ℕ := 6
def brokenSeashells : ℕ := 4
def unbrokenSeashells : ℕ := totalSeashells - brokenSeashells

-- Statement to prove
theorem mike_unbroken_seashells : unbrokenSeashells = 2 := by
  sorry

end NUMINAMATH_GPT_mike_unbroken_seashells_l1192_119243


namespace NUMINAMATH_GPT_units_digit_5_pow_17_mul_4_l1192_119240

theorem units_digit_5_pow_17_mul_4 : ((5 ^ 17) * 4) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_5_pow_17_mul_4_l1192_119240


namespace NUMINAMATH_GPT_files_remaining_l1192_119286

theorem files_remaining 
(h_music_files : ℕ := 16) 
(h_video_files : ℕ := 48) 
(h_files_deleted : ℕ := 30) :
(h_music_files + h_video_files - h_files_deleted = 34) := 
by sorry

end NUMINAMATH_GPT_files_remaining_l1192_119286


namespace NUMINAMATH_GPT_algebraic_expression_value_l1192_119295

theorem algebraic_expression_value (a b : ℝ) (h : a = b + 1) : a^2 - 2 * a * b + b^2 + 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1192_119295


namespace NUMINAMATH_GPT_melanie_has_4_plums_l1192_119239

theorem melanie_has_4_plums (initial_plums : ℕ) (given_plums : ℕ) :
  initial_plums = 7 ∧ given_plums = 3 → initial_plums - given_plums = 4 :=
by
  sorry

end NUMINAMATH_GPT_melanie_has_4_plums_l1192_119239


namespace NUMINAMATH_GPT_calculate_fraction_l1192_119221

theorem calculate_fraction :
  let a := 7
  let b := 5
  let c := -2
  (a^3 + b^3 + c^3) / (a^2 - a * b + b^2 + c^2) = 460 / 43 :=
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l1192_119221


namespace NUMINAMATH_GPT_range_of_a_l1192_119255

-- Define propositions p and q
def p := { x : ℝ | (4 * x - 3) ^ 2 ≤ 1 }
def q (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 1 }

-- Define sets A and B
def A := { x : ℝ | 1 / 2 ≤ x ∧ x ≤ 1 }
def B (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 1 }

-- negation of p (p' is a necessary but not sufficient condition for q')
def p_neg := { x : ℝ | ¬ ((4 * x - 3) ^ 2 ≤ 1) }
def q_neg (a : ℝ) := { x : ℝ | ¬ (a ≤ x ∧ x ≤ a + 1) }

-- range of real number a
theorem range_of_a (a : ℝ) : (A ⊆ B a ∧ A ≠ B a) → 0 ≤ a ∧ a ≤ 1 / 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1192_119255


namespace NUMINAMATH_GPT_find_number_l1192_119248

-- Define the condition
def exceeds_by_30 (x : ℝ) : Prop :=
  x = (3/8) * x + 30

-- Prove the main statement
theorem find_number : ∃ x : ℝ, exceeds_by_30 x ∧ x = 48 := by
  sorry

end NUMINAMATH_GPT_find_number_l1192_119248


namespace NUMINAMATH_GPT_zinc_copper_mixture_weight_l1192_119260

theorem zinc_copper_mixture_weight (Z C : ℝ) (h1 : Z / C = 9 / 11) (h2 : Z = 31.5) : Z + C = 70 := by
  sorry

end NUMINAMATH_GPT_zinc_copper_mixture_weight_l1192_119260


namespace NUMINAMATH_GPT_generalized_schur_inequality_l1192_119200

theorem generalized_schur_inequality (t : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^t * (a - b) * (a - c) + b^t * (b - c) * (b - a) + c^t * (c - a) * (c - b) ≥ 0 :=
sorry

end NUMINAMATH_GPT_generalized_schur_inequality_l1192_119200


namespace NUMINAMATH_GPT_largest_of_eight_consecutive_l1192_119242

theorem largest_of_eight_consecutive (n : ℕ) (h : 8 * n + 28 = 2024) : n + 7 = 256 := by
  -- This means you need to solve for n first, then add 7 to get the largest number
  sorry

end NUMINAMATH_GPT_largest_of_eight_consecutive_l1192_119242


namespace NUMINAMATH_GPT_change_calculation_l1192_119224

def cost_of_apple : ℝ := 0.75
def amount_paid : ℝ := 5.00

theorem change_calculation : (amount_paid - cost_of_apple) = 4.25 := by
  sorry

end NUMINAMATH_GPT_change_calculation_l1192_119224


namespace NUMINAMATH_GPT_total_calculators_sold_l1192_119201

theorem total_calculators_sold 
    (x y : ℕ)
    (h₁ : y = 35)
    (h₂ : 15 * x + 67 * y = 3875) :
    x + y = 137 :=
by 
  -- We will insert the proof here
  sorry

end NUMINAMATH_GPT_total_calculators_sold_l1192_119201


namespace NUMINAMATH_GPT_max_m_sufficient_min_m_necessary_l1192_119213

-- Define variables and conditions
variables (x m : ℝ) (p : Prop := abs x ≤ m) (q : Prop := -1 ≤ x ∧ x ≤ 4) 

-- Problem 1: Maximum value of m for sufficient condition
theorem max_m_sufficient : (∀ x, abs x ≤ m → (-1 ≤ x ∧ x ≤ 4)) → m = 4 := sorry

-- Problem 2: Minimum value of m for necessary condition
theorem min_m_necessary : (∀ x, (-1 ≤ x ∧ x ≤ 4) → abs x ≤ m) → m = 4 := sorry

end NUMINAMATH_GPT_max_m_sufficient_min_m_necessary_l1192_119213


namespace NUMINAMATH_GPT_line_equation_final_equation_l1192_119241

theorem line_equation (k : ℝ) : 
  (∀ x y, y = k * (x - 1) + 1 ↔ 
  ∀ x y, y = k * ((x + 2) - 1) + 1 - 1) → 
  k = 1 / 2 :=
by
  sorry

theorem final_equation : 
  ∃ k : ℝ, k = 1 / 2 ∧ (∀ x y, y = k * (x - 1) + 1) → 
  ∀ x y, x - 2 * y + 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_final_equation_l1192_119241


namespace NUMINAMATH_GPT_total_fruits_sum_l1192_119277

theorem total_fruits_sum (Mike_oranges Matt_apples Mark_bananas Mary_grapes : ℕ)
  (hMike : Mike_oranges = 3)
  (hMatt : Matt_apples = 2 * Mike_oranges)
  (hMark : Mark_bananas = Mike_oranges + Matt_apples)
  (hMary : Mary_grapes = Mike_oranges + Matt_apples + Mark_bananas + 5) :
  Mike_oranges + Matt_apples + Mark_bananas + Mary_grapes = 41 :=
by
  sorry

end NUMINAMATH_GPT_total_fruits_sum_l1192_119277


namespace NUMINAMATH_GPT_range_of_a_l1192_119207

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → (f a x) * (f a (1 - x)) ≥ 1) ↔ (1 ≤ a) ∨ (a ≤ - (1/4)) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1192_119207


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l1192_119211

-- Definitions of given conditions
def d1 : ℝ := 10
def Area : ℝ := 60

-- Proof of desired condition
theorem rhombus_diagonal_length (d2 : ℝ) : 
  (Area = d1 * d2 / 2) → d2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l1192_119211


namespace NUMINAMATH_GPT_probability_three_green_is_14_over_99_l1192_119247

noncomputable def probability_three_green :=
  let total_combinations := Nat.choose 12 4
  let successful_outcomes := (Nat.choose 5 3) * (Nat.choose 7 1)
  (successful_outcomes : ℚ) / total_combinations

theorem probability_three_green_is_14_over_99 :
  probability_three_green = 14 / 99 :=
by
  sorry

end NUMINAMATH_GPT_probability_three_green_is_14_over_99_l1192_119247


namespace NUMINAMATH_GPT_percentage_of_50_of_125_l1192_119265

theorem percentage_of_50_of_125 : (50 / 125) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_50_of_125_l1192_119265


namespace NUMINAMATH_GPT_probability_same_color_correct_l1192_119273

-- conditions
def sides := ["maroon", "teal", "cyan", "sparkly"]
def die : Type := {v // v ∈ sides}
def maroon_count := 6
def teal_count := 9
def cyan_count := 10
def sparkly_count := 5
def total_sides := 30

-- calculate probabilities
def prob (count : ℕ) : ℚ := (count ^ 2) / (total_sides ^ 2)
def prob_same_color : ℚ :=
  prob maroon_count +
  prob teal_count +
  prob cyan_count +
  prob sparkly_count

-- statement
theorem probability_same_color_correct :
  prob_same_color = 121 / 450 :=
sorry

end NUMINAMATH_GPT_probability_same_color_correct_l1192_119273


namespace NUMINAMATH_GPT_mean_difference_l1192_119268

variable (a1 a2 a3 a4 a5 a6 A : ℝ)

-- Arithmetic mean of six numbers is A
axiom mean_six_numbers : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

-- Arithmetic mean of the first four numbers is A + 10
axiom mean_first_four : (a1 + a2 + a3 + a4) / 4 = A + 10

-- Arithmetic mean of the last four numbers is A - 7
axiom mean_last_four : (a3 + a4 + a5 + a6) / 4 = A - 7

-- Prove the arithmetic mean of the first, second, fifth, and sixth numbers differs from A by 3
theorem mean_difference :
  (a1 + a2 + a5 + a6) / 4 = A - 3 := 
sorry

end NUMINAMATH_GPT_mean_difference_l1192_119268


namespace NUMINAMATH_GPT_polygons_sides_l1192_119257

theorem polygons_sides 
  (n1 n2 : ℕ)
  (h1 : n1 * (n1 - 3) / 2 + n2 * (n2 - 3) / 2 = 158)
  (h2 : 180 * (n1 + n2 - 4) = 4320) :
  (n1 = 16 ∧ n2 = 12) ∨ (n1 = 12 ∧ n2 = 16) :=
sorry

end NUMINAMATH_GPT_polygons_sides_l1192_119257


namespace NUMINAMATH_GPT_sufficient_not_necessary_not_necessary_l1192_119262

theorem sufficient_not_necessary (x : ℝ) (h1: x > 2) : x^2 - 3 * x + 2 > 0 :=
sorry

theorem not_necessary (x : ℝ) (h2: x^2 - 3 * x + 2 > 0) : (x > 2 ∨ x < 1) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_not_necessary_l1192_119262


namespace NUMINAMATH_GPT_weight_of_b_l1192_119232

theorem weight_of_b (a b c d : ℝ)
  (h1 : a + b + c + d = 160)
  (h2 : a + b = 50)
  (h3 : b + c = 56)
  (h4 : c + d = 64) :
  b = 46 :=
by sorry

end NUMINAMATH_GPT_weight_of_b_l1192_119232


namespace NUMINAMATH_GPT_real_roots_exist_for_nonzero_K_l1192_119284

theorem real_roots_exist_for_nonzero_K (K : ℝ) (hK : K ≠ 0) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by
  sorry

end NUMINAMATH_GPT_real_roots_exist_for_nonzero_K_l1192_119284


namespace NUMINAMATH_GPT_xyz_div_by_27_l1192_119276

theorem xyz_div_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  27 ∣ (x + y + z) :=
sorry

end NUMINAMATH_GPT_xyz_div_by_27_l1192_119276


namespace NUMINAMATH_GPT_most_accurate_reading_l1192_119251

def temperature_reading (temp: ℝ) : Prop := 
  98.6 ≤ temp ∧ temp ≤ 99.1 ∧ temp ≠ 98.85 ∧ temp > 98.85

theorem most_accurate_reading (temp: ℝ) : temperature_reading temp → temp = 99.1 :=
by
  intros h
  sorry 

end NUMINAMATH_GPT_most_accurate_reading_l1192_119251


namespace NUMINAMATH_GPT_rows_count_mod_pascals_triangle_l1192_119227

-- Define the modified Pascal's triangle function that counts the required rows.
def modified_pascals_triangle_satisfying_rows (n : ℕ) : ℕ := sorry

-- Statement of the problem
theorem rows_count_mod_pascals_triangle :
  modified_pascals_triangle_satisfying_rows 30 = 4 :=
sorry

end NUMINAMATH_GPT_rows_count_mod_pascals_triangle_l1192_119227


namespace NUMINAMATH_GPT_total_students_l1192_119288

theorem total_students (T : ℕ)
  (A_cond : (2/9 : ℚ) * T = (a_real : ℚ))
  (B_cond : (1/3 : ℚ) * T = (b_real : ℚ))
  (C_cond : (2/9 : ℚ) * T = (c_real : ℚ))
  (D_cond : (1/9 : ℚ) * T = (d_real : ℚ))
  (E_cond : 15 = e_real) :
  (2/9 : ℚ) * T + (1/3 : ℚ) * T + (2/9 : ℚ) * T + (1/9 : ℚ) * T + 15 = T → T = 135 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l1192_119288


namespace NUMINAMATH_GPT_total_distance_journey_l1192_119212

theorem total_distance_journey :
  let south := 40
  let east := south + 20
  let north := 2 * east
  (south + east + north) = 220 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_journey_l1192_119212


namespace NUMINAMATH_GPT_abs_diff_x_plus_1_x_minus_2_l1192_119297

theorem abs_diff_x_plus_1_x_minus_2 (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : |x + 1| - |x - 2| = -3 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_x_plus_1_x_minus_2_l1192_119297


namespace NUMINAMATH_GPT_ages_of_sons_l1192_119202

variable (x y z : ℕ)

def father_age_current : ℕ := 33

def youngest_age_current : ℕ := 2

def father_age_in_12_years : ℕ := father_age_current + 12

def sum_of_ages_in_12_years : ℕ := youngest_age_current + 12 + y + 12 + z + 12

theorem ages_of_sons (x y z : ℕ) 
  (h1 : x = 2)
  (h2 : father_age_current = 33)
  (h3 : father_age_in_12_years = 45)
  (h4 : sum_of_ages_in_12_years = 45) :
  x = 2 ∧ y + z = 7 ∧ ((y = 3 ∧ z = 4) ∨ (y = 4 ∧ z = 3)) :=
by
  sorry

end NUMINAMATH_GPT_ages_of_sons_l1192_119202


namespace NUMINAMATH_GPT_largest_of_three_consecutive_odds_l1192_119223

theorem largest_of_three_consecutive_odds (n : ℤ) (h_sum : n + (n + 2) + (n + 4) = -147) : n + 4 = -47 :=
by {
  -- Proof steps here, but we're skipping for this exercise
  sorry
}

end NUMINAMATH_GPT_largest_of_three_consecutive_odds_l1192_119223


namespace NUMINAMATH_GPT_trees_in_park_l1192_119216

variable (W O T : Nat)

theorem trees_in_park (h1 : W = 36) (h2 : O = W + 11) (h3 : T = W + O) : T = 83 := by
  sorry

end NUMINAMATH_GPT_trees_in_park_l1192_119216


namespace NUMINAMATH_GPT_largest_divisible_n_l1192_119245

theorem largest_divisible_n (n : ℕ) :
  (n^3 + 2006) % (n + 26) = 0 → n = 15544 :=
sorry

end NUMINAMATH_GPT_largest_divisible_n_l1192_119245


namespace NUMINAMATH_GPT_fare_calculation_l1192_119261

-- Definitions for given conditions
def initial_mile_fare : ℝ := 3.00
def additional_rate : ℝ := 0.30
def initial_miles : ℝ := 0.5
def available_fare : ℝ := 15 - 3  -- Total minus tip

-- Proof statement
theorem fare_calculation (miles : ℝ) : initial_mile_fare + additional_rate * (miles - initial_miles) / 0.10 = available_fare ↔ miles = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_fare_calculation_l1192_119261


namespace NUMINAMATH_GPT_females_dont_listen_correct_l1192_119205

/-- Number of males who listen to the station -/
def males_listen : ℕ := 45

/-- Number of females who don't listen to the station -/
def females_dont_listen : ℕ := 87

/-- Total number of people who listen to the station -/
def total_listen : ℕ := 120

/-- Total number of people who don't listen to the station -/
def total_dont_listen : ℕ := 135

/-- Number of females surveyed based on the problem description -/
def total_females_surveyed (total_peoples_total : ℕ) (males_dont_listen : ℕ) : ℕ := 
  total_peoples_total - (males_listen + males_dont_listen)

/-- Number of females who listen to the station -/
def females_listen (total_females : ℕ) : ℕ := total_females - females_dont_listen

/-- Proof that the number of females who do not listen to the station is 87 -/
theorem females_dont_listen_correct 
  (total_peoples_total : ℕ)
  (males_dont_listen : ℕ)
  (total_females := total_females_surveyed total_peoples_total males_dont_listen)
  (females_listen := females_listen total_females) :
  females_dont_listen = 87 :=
sorry

end NUMINAMATH_GPT_females_dont_listen_correct_l1192_119205


namespace NUMINAMATH_GPT_determine_angles_l1192_119296

theorem determine_angles 
  (small_angle1 : ℝ) 
  (small_angle2 : ℝ) 
  (large_angle1 : ℝ) 
  (large_angle2 : ℝ) 
  (triangle_sum_property : ∀ a b c : ℝ, a + b + c = 180) 
  (exterior_angle_property : ∀ a c : ℝ, a + c = 180) :
  (small_angle1 = 70) → 
  (small_angle2 = 180 - 130) → 
  (large_angle1 = 45) → 
  (large_angle2 = 50) → 
  ∃ α β : ℝ, α = 120 ∧ β = 85 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_determine_angles_l1192_119296


namespace NUMINAMATH_GPT_find_a_l1192_119271

open Set

def set_A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def set_B : Set ℝ := {2, 3}
def set_C : Set ℝ := {2, -4}

theorem find_a (a : ℝ) (haB : (set_A a) ∩ set_B ≠ ∅) (haC : (set_A a) ∩ set_C = ∅) : a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_l1192_119271


namespace NUMINAMATH_GPT_number_of_sequences_l1192_119270

-- Define the number of targets and their columns
def targetSequence := ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']

-- Define our problem statement
theorem number_of_sequences :
  (List.permutations targetSequence).length = 4200 := by
  sorry

end NUMINAMATH_GPT_number_of_sequences_l1192_119270


namespace NUMINAMATH_GPT_max_xy_min_x2y2_l1192_119274

open Real

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x * y ≤ 1 / 8) :=
sorry

theorem min_x2y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x ^ 2 + y ^ 2 ≥ 1 / 5) :=
sorry


end NUMINAMATH_GPT_max_xy_min_x2y2_l1192_119274


namespace NUMINAMATH_GPT_arith_seq_general_formula_l1192_119219

noncomputable def increasing_arith_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arith_seq_general_formula (a : ℕ → ℤ) (d : ℤ)
  (h_arith : increasing_arith_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = (a 2)^2 - 4) :
  ∀ n, a n = 3 * n - 2 :=
sorry

end NUMINAMATH_GPT_arith_seq_general_formula_l1192_119219


namespace NUMINAMATH_GPT_original_price_of_car_l1192_119203

theorem original_price_of_car (spent price_percent original_price : ℝ) (h1 : spent = 15000) (h2 : price_percent = 0.40) (h3 : spent = price_percent * original_price) : original_price = 37500 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_car_l1192_119203


namespace NUMINAMATH_GPT_draw_balls_equiv_l1192_119222

noncomputable def number_of_ways_to_draw_balls (n : ℕ) (k : ℕ) (ball1 : ℕ) (ball2 : ℕ) : ℕ :=
  if n = 15 ∧ k = 4 ∧ ball1 = 1 ∧ ball2 = 15 then
    4 * (Nat.choose 14 3 * Nat.factorial 3) * 2
  else
    0

theorem draw_balls_equiv : number_of_ways_to_draw_balls 15 4 1 15 = 17472 :=
by
  dsimp [number_of_ways_to_draw_balls]
  rw [Nat.choose, Nat.factorial]
  norm_num
  sorry

end NUMINAMATH_GPT_draw_balls_equiv_l1192_119222


namespace NUMINAMATH_GPT_evaluate_F_2_f_3_l1192_119236

def f (a : ℤ) : ℤ := a^2 - 2
def F (a b : ℤ) : ℤ := b^3 - a

theorem evaluate_F_2_f_3 : F 2 (f 3) = 341 := by
  sorry

end NUMINAMATH_GPT_evaluate_F_2_f_3_l1192_119236


namespace NUMINAMATH_GPT_max_intersections_three_circles_two_lines_l1192_119287

noncomputable def max_intersections_3_circles_2_lines : ℕ :=
  3 * 2 * 1 + 2 * 3 * 2 + 1

theorem max_intersections_three_circles_two_lines :
  max_intersections_3_circles_2_lines = 19 :=
by
  sorry

end NUMINAMATH_GPT_max_intersections_three_circles_two_lines_l1192_119287


namespace NUMINAMATH_GPT_tan_sum_simplification_l1192_119285
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_simplification_l1192_119285


namespace NUMINAMATH_GPT_original_deck_card_count_l1192_119238

theorem original_deck_card_count (r b u : ℕ)
  (h1 : r / (r + b + u) = 1 / 5)
  (h2 : r / (r + b + u + 3) = 1 / 6) :
  r + b + u = 15 := by
  sorry

end NUMINAMATH_GPT_original_deck_card_count_l1192_119238


namespace NUMINAMATH_GPT_inverse_h_l1192_119291

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := 3 * x + 2
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_h (x : ℝ) : h⁻¹ (x) = (x - 7) / 12 :=
sorry

end NUMINAMATH_GPT_inverse_h_l1192_119291
