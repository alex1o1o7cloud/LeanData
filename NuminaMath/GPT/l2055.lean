import Mathlib

namespace NUMINAMATH_GPT_ian_number_is_1021_l2055_205545

-- Define the sequences each student skips
def alice_skips (n : ℕ) := ∃ k : ℕ, n = 4 * k
def barbara_skips (n : ℕ) := ∃ k : ℕ, n = 16 * (k + 1)
def candice_skips (n : ℕ) := ∃ k : ℕ, n = 64 * (k + 1)
-- Similar definitions for Debbie, Eliza, Fatima, Greg, and Helen

-- Define the condition under which Ian says a number
def ian_says (n : ℕ) :=
  ¬(alice_skips n) ∧ ¬(barbara_skips n) ∧ ¬(candice_skips n) -- and so on for Debbie, Eliza, Fatima, Greg, Helen

theorem ian_number_is_1021 : ian_says 1021 :=
by
  sorry

end NUMINAMATH_GPT_ian_number_is_1021_l2055_205545


namespace NUMINAMATH_GPT_base_b_digits_l2055_205598

theorem base_b_digits (b : ℕ) : b^4 ≤ 500 ∧ 500 < b^5 → b = 4 := by
  intro h
  sorry

end NUMINAMATH_GPT_base_b_digits_l2055_205598


namespace NUMINAMATH_GPT_option_B_more_cost_effective_l2055_205527

def cost_option_A (x : ℕ) : ℕ := 60 + 18 * x
def cost_option_B (x : ℕ) : ℕ := 150 + 15 * x
def x : ℕ := 40

theorem option_B_more_cost_effective : cost_option_B x < cost_option_A x := by
  -- Placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_option_B_more_cost_effective_l2055_205527


namespace NUMINAMATH_GPT_sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l2055_205543

noncomputable def sin_110_degrees : ℝ := Real.sin (110 * Real.pi / 180)
noncomputable def tan_945_degrees_reduction : ℝ := Real.tan (945 * Real.pi / 180 - 5 * Real.pi)
noncomputable def cos_25pi_over_4_reduction : ℝ := Real.cos (25 * Real.pi / 4 - 6 * 2 * Real.pi)

theorem sin_110_correct : sin_110_degrees = Real.sin (110 * Real.pi / 180) :=
by
  sorry

theorem tan_945_correct : tan_945_degrees_reduction = 1 :=
by 
  sorry

theorem cos_25pi_over_4_correct : cos_25pi_over_4_reduction = Real.cos (Real.pi / 4) :=
by 
  sorry

end NUMINAMATH_GPT_sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l2055_205543


namespace NUMINAMATH_GPT_exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l2055_205533

theorem exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987 :
  ∃ n : ℕ, n ^ n + (n + 1) ^ n ≡ 0 [MOD 1987] := sorry

end NUMINAMATH_GPT_exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l2055_205533


namespace NUMINAMATH_GPT_platform_length_150_l2055_205523

def speed_kmph : ℕ := 54  -- Speed in km/hr

def speed_mps : ℚ := speed_kmph * 1000 / 3600  -- Speed in m/s

def time_pass_man : ℕ := 20  -- Time to pass a man in seconds
def time_pass_platform : ℕ := 30  -- Time to pass a platform in seconds

def length_train : ℚ := speed_mps * time_pass_man  -- Length of the train in meters

def length_platform (P : ℚ) : Prop :=
  length_train + P = speed_mps * time_pass_platform  -- The condition involving platform length

theorem platform_length_150 :
  length_platform 150 := by
  -- We would provide a proof here.
  sorry

end NUMINAMATH_GPT_platform_length_150_l2055_205523


namespace NUMINAMATH_GPT_longest_boat_length_l2055_205568

theorem longest_boat_length (a : ℝ) (c : ℝ) 
  (parallel_banks : ∀ x y : ℝ, (x = y) ∨ (x = -y)) 
  (right_angle_bend : ∃ b : ℝ, b = a) :
  c = 2 * a * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_longest_boat_length_l2055_205568


namespace NUMINAMATH_GPT_monomial_addition_l2055_205589

-- Definition of a monomial in Lean
def isMonomial (p : ℕ → ℝ) : Prop := ∃ c n, ∀ x, p x = c * x^n

theorem monomial_addition (A : ℕ → ℝ) :
  (isMonomial (fun x => -3 * x + A x)) → isMonomial A :=
sorry

end NUMINAMATH_GPT_monomial_addition_l2055_205589


namespace NUMINAMATH_GPT_greater_number_is_64_l2055_205504

-- Proof statement: The greater number (y) is 64 given the conditions
theorem greater_number_is_64 (x y : ℕ) 
    (h1 : y = 2 * x) 
    (h2 : x + y = 96) : 
    y = 64 := 
sorry

end NUMINAMATH_GPT_greater_number_is_64_l2055_205504


namespace NUMINAMATH_GPT_charlie_more_apples_than_bella_l2055_205530

variable (D : ℝ) 

theorem charlie_more_apples_than_bella 
    (hC : C = 1.75 * D)
    (hB : B = 1.50 * D) :
    (C - B) / B = 0.1667 := 
by
  sorry

end NUMINAMATH_GPT_charlie_more_apples_than_bella_l2055_205530


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l2055_205538

theorem arithmetic_sequence_product 
  (a d : ℤ)
  (h1 : a + 6 * d = 20)
  (h2 : d = 2) : 
  a * (a + d) * (a + 2 * d) = 960 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l2055_205538


namespace NUMINAMATH_GPT_waynes_son_time_to_shovel_l2055_205582

-- Definitions based on the conditions
variables (S W : ℝ) (son_rate : S = 1 / 21) (wayne_rate : W = 6 * S) (together_rate : 3 * (S + W) = 1)

theorem waynes_son_time_to_shovel : 
  1 / S = 21 :=
by
  -- Proof will be provided later
  sorry

end NUMINAMATH_GPT_waynes_son_time_to_shovel_l2055_205582


namespace NUMINAMATH_GPT_factor_expression_l2055_205505

theorem factor_expression (x : ℝ) :
  (3*x^3 + 48*x^2 - 14) - (-9*x^3 + 2*x^2 - 14) =
  2*x^2 * (6*x + 23) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2055_205505


namespace NUMINAMATH_GPT_restaurant_problem_l2055_205557

theorem restaurant_problem (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_problem_l2055_205557


namespace NUMINAMATH_GPT_total_weight_is_40_l2055_205593

def marco_strawberries_weight : ℕ := 8
def dad_strawberries_weight : ℕ := 32
def total_strawberries_weight := marco_strawberries_weight + dad_strawberries_weight

theorem total_weight_is_40 : total_strawberries_weight = 40 := by
  sorry

end NUMINAMATH_GPT_total_weight_is_40_l2055_205593


namespace NUMINAMATH_GPT_third_side_length_l2055_205532

/-- Given two sides of a triangle with lengths 4cm and 9cm, prove that the valid length of the third side must be 9cm. -/
theorem third_side_length (a b c : ℝ) (h₀ : a = 4) (h₁ : b = 9) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → (c = 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_third_side_length_l2055_205532


namespace NUMINAMATH_GPT_chain_of_inequalities_l2055_205526

theorem chain_of_inequalities (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  9 / (a + b + c) ≤ (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ∧ 
  (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ≤ (1 / a + 1 / b + 1 / c) := 
by 
  sorry

end NUMINAMATH_GPT_chain_of_inequalities_l2055_205526


namespace NUMINAMATH_GPT_value_of_otimes_difference_l2055_205588

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem value_of_otimes_difference :
  otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = - 1184 / 243 := 
by
  sorry

end NUMINAMATH_GPT_value_of_otimes_difference_l2055_205588


namespace NUMINAMATH_GPT_triangle_area_is_correct_l2055_205518

noncomputable def triangle_area : ℝ :=
  let A := (3, 3)
  let B := (4.5, 7.5)
  let C := (7.5, 4.5)
  1 / 2 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℝ)|

theorem triangle_area_is_correct : triangle_area = 9 := by
  sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l2055_205518


namespace NUMINAMATH_GPT_all_integers_appear_exactly_once_l2055_205555

noncomputable def sequence_of_integers (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ m : ℕ, a m > 0 ∧ ∃ m' : ℕ, a m' < 0

noncomputable def distinct_modulo_n (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, (∀ i j : ℕ, i < j ∧ j < n → a i % n ≠ a j % n)

theorem all_integers_appear_exactly_once
  (a : ℕ → ℤ)
  (h_seq : sequence_of_integers a)
  (h_distinct : distinct_modulo_n a) :
  ∀ x : ℤ, ∃! i : ℕ, a i = x := 
sorry

end NUMINAMATH_GPT_all_integers_appear_exactly_once_l2055_205555


namespace NUMINAMATH_GPT_remainder_of_product_l2055_205556

theorem remainder_of_product (a b n : ℕ) (h1 : a = 2431) (h2 : b = 1587) (h3 : n = 800) : 
  (a * b) % n = 397 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_product_l2055_205556


namespace NUMINAMATH_GPT_value_of_place_ratio_l2055_205586

theorem value_of_place_ratio :
  let d8_pos := 10000
  let d6_pos := 0.1
  d8_pos = 100000 * d6_pos :=
by
  let d8_pos := 10000
  let d6_pos := 0.1
  sorry

end NUMINAMATH_GPT_value_of_place_ratio_l2055_205586


namespace NUMINAMATH_GPT_parallel_lines_eq_l2055_205559

theorem parallel_lines_eq {a x y : ℝ} :
  (∀ x y : ℝ, x + a * y = 2 * a + 2) ∧ (∀ x y : ℝ, a * x + y = a + 1) →
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_eq_l2055_205559


namespace NUMINAMATH_GPT_triangle_side_y_values_l2055_205539

theorem triangle_side_y_values (y : ℕ) : (4 < y^2 ∧ y^2 < 20) ↔ (y = 3 ∨ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_y_values_l2055_205539


namespace NUMINAMATH_GPT_complement_union_eq_l2055_205534

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end NUMINAMATH_GPT_complement_union_eq_l2055_205534


namespace NUMINAMATH_GPT_cube_surface_area_l2055_205558

theorem cube_surface_area (s : ℝ) (h : s = 8) : 6 * s^2 = 384 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l2055_205558


namespace NUMINAMATH_GPT_company_bought_gravel_l2055_205576

def weight_of_gravel (total_weight_of_materials : ℝ) (weight_of_sand : ℝ) : ℝ :=
  total_weight_of_materials - weight_of_sand

theorem company_bought_gravel :
  weight_of_gravel 14.02 8.11 = 5.91 := 
by
  sorry

end NUMINAMATH_GPT_company_bought_gravel_l2055_205576


namespace NUMINAMATH_GPT_negative_expression_b_negative_expression_c_negative_expression_e_l2055_205580

theorem negative_expression_b:
  3 * Real.sqrt 11 - 10 < 0 := 
sorry

theorem negative_expression_c:
  18 - 5 * Real.sqrt 13 < 0 := 
sorry

theorem negative_expression_e:
  10 * Real.sqrt 26 - 51 < 0 := 
sorry

end NUMINAMATH_GPT_negative_expression_b_negative_expression_c_negative_expression_e_l2055_205580


namespace NUMINAMATH_GPT_polygon_sides_eq_eight_l2055_205510

theorem polygon_sides_eq_eight (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
sorry

end NUMINAMATH_GPT_polygon_sides_eq_eight_l2055_205510


namespace NUMINAMATH_GPT_abs_fraction_inequality_solution_l2055_205502

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end NUMINAMATH_GPT_abs_fraction_inequality_solution_l2055_205502


namespace NUMINAMATH_GPT_melissa_gave_x_books_l2055_205535

-- Define the initial conditions as constants
def initial_melissa_books : ℝ := 123
def initial_jordan_books : ℝ := 27
def final_melissa_books (x : ℝ) : ℝ := initial_melissa_books - x
def final_jordan_books (x : ℝ) : ℝ := initial_jordan_books + x

-- The main theorem to prove how many books Melissa gave to Jordan
theorem melissa_gave_x_books : ∃ x : ℝ, final_melissa_books x = 3 * final_jordan_books x ∧ x = 10.5 :=
sorry

end NUMINAMATH_GPT_melissa_gave_x_books_l2055_205535


namespace NUMINAMATH_GPT_Anna_needs_308_tulips_l2055_205546

-- Define conditions as assertions or definitions
def number_of_eyes := 2
def red_tulips_per_eye := 8 
def number_of_eyebrows := 2
def purple_tulips_per_eyebrow := 5
def red_tulips_for_nose := 12
def red_tulips_for_smile := 18
def yellow_tulips_background := 9 * red_tulips_for_smile
def additional_purple_tulips_eyebrows := 4 * number_of_eyes * red_tulips_per_eye - number_of_eyebrows * purple_tulips_per_eyebrow
def yellow_tulips_for_nose := 3 * red_tulips_for_nose

-- Define total number of tulips for each color
def total_red_tulips := number_of_eyes * red_tulips_per_eye + red_tulips_for_nose + red_tulips_for_smile
def total_purple_tulips := number_of_eyebrows * purple_tulips_per_eyebrow + additional_purple_tulips_eyebrows
def total_yellow_tulips := yellow_tulips_background + yellow_tulips_for_nose

-- Define the total number of tulips
def total_tulips := total_red_tulips + total_purple_tulips + total_yellow_tulips

theorem Anna_needs_308_tulips :
  total_tulips = 308 :=
sorry

end NUMINAMATH_GPT_Anna_needs_308_tulips_l2055_205546


namespace NUMINAMATH_GPT_combined_loss_l2055_205564

variable (initial : ℕ) (donation : ℕ) (prize : ℕ) (final : ℕ) (lottery_winning : ℕ) (X : ℕ)

theorem combined_loss (h1 : initial = 10) (h2 : donation = 4) (h3 : prize = 90) 
                      (h4 : final = 94) (h5 : lottery_winning = 65) :
                      (initial - donation + prize - X + lottery_winning = final) ↔ (X = 67) :=
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_combined_loss_l2055_205564


namespace NUMINAMATH_GPT_usual_time_is_49_l2055_205506

variable (R T : ℝ)
variable (h1 : R > 0) -- Usual rate is positive
variable (h2 : T > 0) -- Usual time is positive
variable (condition : T * R = (T - 7) * (7 / 6 * R)) -- Main condition derived from the problem

theorem usual_time_is_49 (h1 : R > 0) (h2 : T > 0) (condition : T * R = (T - 7) * (7 / 6 * R)) : T = 49 := by
  sorry -- Proof goes here

end NUMINAMATH_GPT_usual_time_is_49_l2055_205506


namespace NUMINAMATH_GPT_pants_cost_l2055_205511

/-- Given:
- 3 skirts with each costing $20.00
- 5 blouses with each costing $15.00
- The total spending is $180.00
- A discount on pants: buy 1 pair get 1 pair 1/2 off

Prove that each pair of pants costs $30.00 before the discount. --/
theorem pants_cost (cost_skirt cost_blouse total_amount : ℤ) (pants_discount: ℚ) (total_cost: ℤ) :
  cost_skirt = 20 ∧ cost_blouse = 15 ∧ total_amount = 180 
  ∧ pants_discount * 2 = 1 
  ∧ total_cost = 3 * cost_skirt + 5 * cost_blouse + 3/2 * pants_discount → 
  pants_discount = 30 := by
  sorry

end NUMINAMATH_GPT_pants_cost_l2055_205511


namespace NUMINAMATH_GPT_problem_solution_l2055_205507

noncomputable def circles_intersect (m : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), (A ∈ { p | p.1^2 + p.2^2 = 1 }) ∧ (B ∈ { p | p.1^2 + p.2^2 = 1 }) ∧
  (A ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ (B ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ 
  (dist A B = (4 * Real.sqrt 5) / 5)

theorem problem_solution (m : ℝ) : circles_intersect m ↔ (m = 1 ∨ m = -3) := by
  sorry

end NUMINAMATH_GPT_problem_solution_l2055_205507


namespace NUMINAMATH_GPT_best_choice_for_square_formula_l2055_205571

theorem best_choice_for_square_formula : 
  (89.8^2 = (90 - 0.2)^2) :=
by sorry

end NUMINAMATH_GPT_best_choice_for_square_formula_l2055_205571


namespace NUMINAMATH_GPT_DVDs_sold_168_l2055_205503

-- Definitions of the conditions
def CDs_sold := ℤ
def DVDs_sold := ℤ

def ratio_condition (C D : ℤ) : Prop := D = 16 * C / 10
def total_condition (C D : ℤ) : Prop := D + C = 273

-- The main statement to prove
theorem DVDs_sold_168 (C D : ℤ) 
  (h1 : ratio_condition C D) 
  (h2 : total_condition C D) : D = 168 :=
sorry

end NUMINAMATH_GPT_DVDs_sold_168_l2055_205503


namespace NUMINAMATH_GPT_sum_of_three_numbers_l2055_205573

theorem sum_of_three_numbers :
  ((3 : ℝ) / 8) + 0.125 + 9.51 = 10.01 :=
sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l2055_205573


namespace NUMINAMATH_GPT_x_eq_sum_of_squares_of_two_consecutive_integers_l2055_205540

noncomputable def x_seq (n : ℕ) : ℝ :=
  1 / 4 * ((2 + Real.sqrt 3) ^ (2 * n - 1) + (2 - Real.sqrt 3) ^ (2 * n - 1))

theorem x_eq_sum_of_squares_of_two_consecutive_integers (n : ℕ) : 
  ∃ y : ℤ, x_seq n = (y:ℝ)^2 + (y + 1)^2 :=
sorry

end NUMINAMATH_GPT_x_eq_sum_of_squares_of_two_consecutive_integers_l2055_205540


namespace NUMINAMATH_GPT_sum_of_variables_is_233_l2055_205561

-- Define A, B, C, D, E, F with their corresponding values.
def A : ℤ := 13
def B : ℤ := 9
def C : ℤ := -3
def D : ℤ := -2
def E : ℕ := 165
def F : ℕ := 51

-- Define the main theorem to prove the sum of A, B, C, D, E, F equals 233.
theorem sum_of_variables_is_233 : A + B + C + D + E + F = 233 := 
by {
  -- Proof is not required according to problem statement, hence using sorry.
  sorry
}

end NUMINAMATH_GPT_sum_of_variables_is_233_l2055_205561


namespace NUMINAMATH_GPT_divisibility_l2055_205521

def Q (X : ℤ) := (X - 1) ^ 3

def P_n (n : ℕ) (X : ℤ) : ℤ :=
  n * X ^ (n + 2) - (n + 2) * X ^ (n + 1) + (n + 2) * X - n

theorem divisibility (n : ℕ) (h : n > 0) : ∀ X : ℤ, Q X ∣ P_n n X :=
by
  sorry

end NUMINAMATH_GPT_divisibility_l2055_205521


namespace NUMINAMATH_GPT_vertical_asymptote_unique_d_values_l2055_205515

theorem vertical_asymptote_unique_d_values (d : ℝ) :
  (∃! x : ℝ, ∃ c : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x^2 - 2*x + d) = 0) ↔ (d = 0 ∨ d = -3) := 
sorry

end NUMINAMATH_GPT_vertical_asymptote_unique_d_values_l2055_205515


namespace NUMINAMATH_GPT_probability_AC_adjacent_l2055_205581

noncomputable def probability_AC_adjacent_given_AB_adjacent : ℚ :=
  let total_permutations_with_AB_adjacent := 48
  let permutations_with_ABC_adjacent := 12
  permutations_with_ABC_adjacent / total_permutations_with_AB_adjacent

theorem probability_AC_adjacent :  
  probability_AC_adjacent_given_AB_adjacent = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_AC_adjacent_l2055_205581


namespace NUMINAMATH_GPT_evaluate_expression_l2055_205550

theorem evaluate_expression :
  let a := Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let b := - Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let c := Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  let d := - Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  (1/a + 1/b + 1/c + 1/d)^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2055_205550


namespace NUMINAMATH_GPT_necessary_condition_for_abs_ab_l2055_205595

theorem necessary_condition_for_abs_ab {a b : ℝ} (h : |a - b| = |a| - |b|) : ab ≥ 0 :=
sorry

end NUMINAMATH_GPT_necessary_condition_for_abs_ab_l2055_205595


namespace NUMINAMATH_GPT_sheets_in_stack_l2055_205574

theorem sheets_in_stack (sheets : ℕ) (thickness : ℝ) (h1 : sheets = 400) (h2 : thickness = 4) :
    let thickness_per_sheet := thickness / sheets
    let stack_height := 6
    (stack_height / thickness_per_sheet = 600) :=
by
  sorry

end NUMINAMATH_GPT_sheets_in_stack_l2055_205574


namespace NUMINAMATH_GPT_infinite_natural_numbers_with_factored_polynomial_l2055_205597

theorem infinite_natural_numbers_with_factored_polynomial :
  ∃ (N : ℕ), ∀ k : ℤ, ∃ (A B: Polynomial ℤ),
  (Polynomial.X ^ 8 + Polynomial.C (N : ℤ) * Polynomial.X ^ 4 + 1) = A * B :=
sorry

end NUMINAMATH_GPT_infinite_natural_numbers_with_factored_polynomial_l2055_205597


namespace NUMINAMATH_GPT_find_value_of_expression_l2055_205500

theorem find_value_of_expression (m n : ℝ) 
  (h1 : m^2 + 2 * m * n = 3) 
  (h2 : m * n + n^2 = 4) : 
  m^2 + 3 * m * n + n^2 = 7 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l2055_205500


namespace NUMINAMATH_GPT_maximize_farmer_profit_l2055_205551

theorem maximize_farmer_profit :
  ∃ x y : ℝ, x + y ≤ 2 ∧ 3 * x + y ≤ 5 ∧ x ≥ 0 ∧ y ≥ 0 ∧ x = 1.5 ∧ y = 0.5 ∧ 
  (∀ x' y' : ℝ, x' + y' ≤ 2 ∧ 3 * x' + y' ≤ 5 ∧ x' ≥ 0 ∧ y' ≥ 0 → 14400 * x + 6300 * y ≥ 14400 * x' + 6300 * y') :=
by
  sorry

end NUMINAMATH_GPT_maximize_farmer_profit_l2055_205551


namespace NUMINAMATH_GPT_max_tickets_l2055_205584

theorem max_tickets (ticket_price normal_discounted_price budget : ℕ) (h1 : ticket_price = 15) (h2 : normal_discounted_price = 13) (h3 : budget = 180) :
  ∃ n : ℕ, ((n ≤ 10 → ticket_price * n ≤ budget) ∧ (n > 10 → normal_discounted_price * n ≤ budget)) ∧ ∀ m : ℕ, ((m ≤ 10 → ticket_price * m ≤ budget) ∧ (m > 10 → normal_discounted_price * m ≤ budget)) → m ≤ 13 :=
by
  sorry

end NUMINAMATH_GPT_max_tickets_l2055_205584


namespace NUMINAMATH_GPT_no_xy_term_implies_k_eq_4_l2055_205585

theorem no_xy_term_implies_k_eq_4 (k : ℝ) :
  (∀ x y : ℝ, (x + 2 * y) * (2 * x - k * y - 1) = 2 * x^2 + (4 - k) * x * y - x - 2 * k * y^2 - 2 * y) →
  ((4 - k) = 0) →
  k = 4 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_no_xy_term_implies_k_eq_4_l2055_205585


namespace NUMINAMATH_GPT_brooke_initial_l2055_205536

variable (B : ℕ)

def brooke_balloons_initially (B : ℕ) :=
  let brooke_balloons := B + 8
  let tracy_balloons_initial := 6
  let tracy_added_balloons := 24
  let tracy_balloons := tracy_balloons_initial + tracy_added_balloons
  let tracy_popped_balloons := tracy_balloons / 2 -- Tracy having half her balloons popped.
  (brooke_balloons + tracy_popped_balloons = 35)

theorem brooke_initial (h : brooke_balloons_initially B) : B = 12 :=
  sorry

end NUMINAMATH_GPT_brooke_initial_l2055_205536


namespace NUMINAMATH_GPT_carA_travel_time_l2055_205513

theorem carA_travel_time 
    (speedA speedB distanceB : ℕ)
    (ratio : ℕ)
    (timeB : ℕ)
    (h_speedA : speedA = 50)
    (h_speedB : speedB = 100)
    (h_distanceB : distanceB = speedB * timeB)
    (h_ratio : distanceA / distanceB = ratio)
    (h_ratio_value : ratio = 3)
    (h_timeB : timeB = 1)
  : distanceA / speedA = 6 :=
by sorry

end NUMINAMATH_GPT_carA_travel_time_l2055_205513


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2055_205566

theorem simplify_and_evaluate (x : ℝ) (h₁ : x ≠ 0) (h₂ : x = 2) : 
  (1 + 1 / x) / ((x^2 - 1) / x) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2055_205566


namespace NUMINAMATH_GPT_proof1_proof2_proof3_l2055_205562

variables (x m n : ℝ)

theorem proof1 (x : ℝ) : (-3 * x - 5) * (5 - 3 * x) = 9 * x^2 - 25 :=
sorry

theorem proof2 (x : ℝ) : (-3 * x - 5) * (5 + 3 * x) = - (3 * x + 5) ^ 2 :=
sorry

theorem proof3 (m n : ℝ) : (2 * m - 3 * n + 1) * (2 * m + 1 + 3 * n) = (2 * m + 1) ^ 2 - (3 * n) ^ 2 :=
sorry

end NUMINAMATH_GPT_proof1_proof2_proof3_l2055_205562


namespace NUMINAMATH_GPT_max_levels_passable_prob_pass_three_levels_l2055_205517

-- Define the condition for passing a level
def passes_level (n : ℕ) (sum : ℕ) : Prop :=
  sum > 2^n

-- Define the maximum sum possible for n dice rolls
def max_sum (n : ℕ) : ℕ :=
  6 * n

-- Define the probability of passing the n-th level
def prob_passing_level (n : ℕ) : ℚ :=
  if n = 1 then 2/3
  else if n = 2 then 5/6
  else if n = 3 then 20/27
  else 0 

-- Combine probabilities for passing the first three levels
def prob_passing_three_levels : ℚ :=
  (2/3) * (5/6) * (20/27)

-- Theorem statement for the maximum number of levels passable
theorem max_levels_passable : 4 = 4 :=
sorry

-- Theorem statement for the probability of passing the first three levels
theorem prob_pass_three_levels : prob_passing_three_levels = 100 / 243 :=
sorry

end NUMINAMATH_GPT_max_levels_passable_prob_pass_three_levels_l2055_205517


namespace NUMINAMATH_GPT_attendees_on_monday_is_10_l2055_205554

-- Define the given conditions
def attendees_tuesday : ℕ := 15
def attendees_wed_thru_fri : ℕ := 10
def days_wed_thru_fri : ℕ := 3
def average_attendance : ℕ := 11
def total_days : ℕ := 5

-- Define the number of people who attended class on Monday
def attendees_tuesday_to_friday : ℕ := attendees_tuesday + attendees_wed_thru_fri * days_wed_thru_fri
def total_attendance : ℕ := average_attendance * total_days
def attendees_monday : ℕ := total_attendance - attendees_tuesday_to_friday

-- State the theorem
theorem attendees_on_monday_is_10 : attendees_monday = 10 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_attendees_on_monday_is_10_l2055_205554


namespace NUMINAMATH_GPT_work_done_in_11_days_l2055_205594

-- Given conditions as definitions
def a_days := 24
def b_days := 30
def c_days := 40
def combined_work_rate := (1 / a_days) + (1 / b_days) + (1 / c_days)
def days_c_leaves_before_completion := 4

-- Statement of the problem to be proved
theorem work_done_in_11_days :
  ∃ (D : ℕ), D = 11 ∧ ((D - days_c_leaves_before_completion) * combined_work_rate) + 
  (days_c_leaves_before_completion * ((1 / a_days) + (1 / b_days))) = 1 :=
sorry

end NUMINAMATH_GPT_work_done_in_11_days_l2055_205594


namespace NUMINAMATH_GPT_factor_1024_into_three_factors_l2055_205587

theorem factor_1024_into_three_factors :
  ∃ (factors : Finset (Finset ℕ)), factors.card = 14 ∧
  ∀ f ∈ factors, ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ (2 ^ a) * (2 ^ b) * (2 ^ c) = 1024 :=
sorry

end NUMINAMATH_GPT_factor_1024_into_three_factors_l2055_205587


namespace NUMINAMATH_GPT_total_time_watching_videos_l2055_205590

theorem total_time_watching_videos 
  (cat_video_length : ℕ)
  (dog_video_length : ℕ)
  (gorilla_video_length : ℕ)
  (h1 : cat_video_length = 4)
  (h2 : dog_video_length = 2 * cat_video_length)
  (h3 : gorilla_video_length = 2 * (cat_video_length + dog_video_length)) :
  cat_video_length + dog_video_length + gorilla_video_length = 36 :=
  by
  sorry

end NUMINAMATH_GPT_total_time_watching_videos_l2055_205590


namespace NUMINAMATH_GPT_length_of_AC_l2055_205520

theorem length_of_AC (AB : ℝ) (C : ℝ) (h1 : AB = 4) (h2 : 0 < C) (h3 : C < AB) (mean_proportional : C * C = AB * (AB - C)) :
  C = 2 * Real.sqrt 5 - 2 := 
sorry

end NUMINAMATH_GPT_length_of_AC_l2055_205520


namespace NUMINAMATH_GPT_minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l2055_205524

theorem minimum_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

theorem minimum_value_x_add_2y_achieved (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  ∃ x y, 0 < x ∧ 0 < y ∧ 1/x + 9/y = 1 ∧ x + 2 * y = 19 + 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l2055_205524


namespace NUMINAMATH_GPT_KarenEggRolls_l2055_205548

-- Definitions based on conditions
def OmarEggRolls : ℕ := 219
def TotalEggRolls : ℕ := 448

-- The statement to be proved
theorem KarenEggRolls : (TotalEggRolls - OmarEggRolls = 229) :=
by {
    -- Proof step goes here
    sorry
}

end NUMINAMATH_GPT_KarenEggRolls_l2055_205548


namespace NUMINAMATH_GPT_yolanda_walking_rate_l2055_205579

theorem yolanda_walking_rate 
  (d_xy : ℕ) (bob_start_after_yolanda : ℕ) (bob_distance_walked : ℕ) 
  (bob_rate : ℕ) (y : ℕ) 
  (bob_distance_to_time : bob_rate ≠ 0 ∧ bob_distance_walked / bob_rate = 2) 
  (yolanda_distance_walked : d_xy - bob_distance_walked = 9 ∧ y = 9 / 3) : 
  y = 3 :=
by 
  sorry

end NUMINAMATH_GPT_yolanda_walking_rate_l2055_205579


namespace NUMINAMATH_GPT_container_capacity_l2055_205549

variable (C : ℝ)
variable (h1 : 0.30 * C + 27 = (3/4) * C)

theorem container_capacity : C = 60 := by
  sorry

end NUMINAMATH_GPT_container_capacity_l2055_205549


namespace NUMINAMATH_GPT_polynomial_no_strictly_positive_roots_l2055_205522

-- Define the necessary conditions and prove the main statement

variables (n : ℕ)
variables (a : Fin n → ℕ) (k : ℕ) (M : ℕ)

-- Axioms/Conditions
axiom pos_a (i : Fin n) : 0 < a i
axiom pos_k : 0 < k
axiom pos_M : 0 < M
axiom M_gt_1 : M > 1

axiom sum_reciprocals : (Finset.univ.sum (λ i => (1 : ℚ) / a i)) = k
axiom product_a : (Finset.univ.prod a) = M

noncomputable def polynomial_has_no_positive_roots : Prop :=
  ∀ x : ℝ, 0 < x →
    M * (1 + x)^k > (Finset.univ.prod (λ i => x + a i))

theorem polynomial_no_strictly_positive_roots (h : polynomial_has_no_positive_roots n a k M) : 
  ∀ x : ℝ, 0 < x → (M * (1 + x)^k - (Finset.univ.prod (λ i => x + a i)) ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_no_strictly_positive_roots_l2055_205522


namespace NUMINAMATH_GPT_find_value_of_p_l2055_205591

-- Definition of the parabola and ellipse
def parabola (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 = 2 * p * xy.2}
def ellipse : Set (ℝ × ℝ) := {xy | xy.1 ^ 2 / 6 + xy.2 ^ 2 / 4 = 1}

-- Hypotheses
variables (p : ℝ) (h_pos : p > 0)

-- Latus rectum tangent to the ellipse
theorem find_value_of_p (h_tangent : ∃ (x y : ℝ),
  (parabola p (x, y) ∧ ellipse (x, y) ∧ y = -p / 2)) : p = 4 := sorry

end NUMINAMATH_GPT_find_value_of_p_l2055_205591


namespace NUMINAMATH_GPT_find_y_value_l2055_205553

theorem find_y_value : (15^3 * 7^4) / 5670 = 1428.75 := by
  sorry

end NUMINAMATH_GPT_find_y_value_l2055_205553


namespace NUMINAMATH_GPT_pens_to_sell_to_make_profit_l2055_205508

theorem pens_to_sell_to_make_profit (initial_pens : ℕ) (purchase_price selling_price profit : ℝ) :
  initial_pens = 2000 →
  purchase_price = 0.15 →
  selling_price = 0.30 →
  profit = 150 →
  (initial_pens * selling_price - initial_pens * purchase_price = profit) →
  initial_pens * profit / (selling_price - purchase_price) = 1500 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_pens_to_sell_to_make_profit_l2055_205508


namespace NUMINAMATH_GPT_count_two_digit_perfect_squares_divisible_by_4_l2055_205519

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end NUMINAMATH_GPT_count_two_digit_perfect_squares_divisible_by_4_l2055_205519


namespace NUMINAMATH_GPT_calculate_unshaded_perimeter_l2055_205516

-- Defining the problem's conditions and results.
def total_length : ℕ := 20
def total_width : ℕ := 12
def shaded_area : ℕ := 65
def inner_shaded_width : ℕ := 5
def total_area : ℕ := total_length * total_width
def unshaded_area : ℕ := total_area - shaded_area

-- Define dimensions for the unshaded region based on the problem conditions.
def unshaded_width : ℕ := total_width - inner_shaded_width
def unshaded_height : ℕ := unshaded_area / unshaded_width

-- Calculate perimeter of the unshaded region.
def unshaded_perimeter : ℕ := 2 * (unshaded_width + unshaded_height)

-- Stating the theorem to be proved.
theorem calculate_unshaded_perimeter : unshaded_perimeter = 64 := 
sorry

end NUMINAMATH_GPT_calculate_unshaded_perimeter_l2055_205516


namespace NUMINAMATH_GPT_sahil_purchase_price_l2055_205569

def purchase_price (P : ℝ) : Prop :=
  let repair_cost := 5000
  let transportation_charges := 1000
  let total_cost := repair_cost + transportation_charges
  let selling_price := 27000
  let profit_factor := 1.5
  profit_factor * (P + total_cost) = selling_price

theorem sahil_purchase_price : ∃ P : ℝ, purchase_price P ∧ P = 12000 :=
by
  use 12000
  unfold purchase_price
  simp
  sorry

end NUMINAMATH_GPT_sahil_purchase_price_l2055_205569


namespace NUMINAMATH_GPT_option_C_correct_l2055_205578

theorem option_C_correct (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := 
by 
  sorry

end NUMINAMATH_GPT_option_C_correct_l2055_205578


namespace NUMINAMATH_GPT_math_problem_l2055_205528

noncomputable def proof_statement : Prop :=
  ∃ (a b m : ℝ),
    0 < a ∧ 0 < b ∧ 0 < m ∧
    (5 = m^2 * ((a^2 / b^2) + (b^2 / a^2)) + m * (a/b + b/a)) ∧
    m = (-1 + Real.sqrt 21) / 2

theorem math_problem : proof_statement :=
  sorry

end NUMINAMATH_GPT_math_problem_l2055_205528


namespace NUMINAMATH_GPT_ratio_problem_l2055_205570

theorem ratio_problem 
  (a b c d : ℚ)
  (h₁ : a / b = 8)
  (h₂ : c / b = 5)
  (h₃ : c / d = 1 / 3) : 
  d / a = 15 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_problem_l2055_205570


namespace NUMINAMATH_GPT_root_equation_value_l2055_205563

theorem root_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2026 - m^2 + 2 * m = 2023 :=
sorry

end NUMINAMATH_GPT_root_equation_value_l2055_205563


namespace NUMINAMATH_GPT_curve_distance_bound_l2055_205509

/--
Given the point A on the curve y = e^x and point B on the curve y = ln(x),
prove that |AB| >= a always holds if and only if a <= sqrt(2).
-/
theorem curve_distance_bound {A B : ℝ × ℝ} (a : ℝ)
  (hA : A.2 = Real.exp A.1) (hB : B.2 = Real.log B.1) :
  (dist A B ≥ a) ↔ (a ≤ Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_curve_distance_bound_l2055_205509


namespace NUMINAMATH_GPT_students_chose_apples_l2055_205529

theorem students_chose_apples (total students choosing_bananas : ℕ) (h1 : students_choosing_bananas = 168) 
  (h2 : 3 * total = 4 * students_choosing_bananas) : (total / 4) = 56 :=
  by
  sorry

end NUMINAMATH_GPT_students_chose_apples_l2055_205529


namespace NUMINAMATH_GPT_sum_of_coefficients_256_l2055_205542

theorem sum_of_coefficients_256 (n : ℕ) (h : (3 + 1)^n = 256) : n = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_256_l2055_205542


namespace NUMINAMATH_GPT_intersection_eq_l2055_205575

-- defining the set A
def A := {x : ℝ | x^2 + 2*x - 3 ≤ 0}

-- defining the set B
def B := {y : ℝ | ∃ x ∈ A, y = x^2 + 4*x + 3}

-- The proof problem statement: prove that A ∩ B = [-1, 1]
theorem intersection_eq : A ∩ B = {y : ℝ | -1 ≤ y ∧ y ≤ 1} :=
by sorry

end NUMINAMATH_GPT_intersection_eq_l2055_205575


namespace NUMINAMATH_GPT_ratio_evaluation_l2055_205596

theorem ratio_evaluation :
  (10 ^ 2003 + 10 ^ 2001) / (2 * 10 ^ 2002) = 101 / 20 := 
by sorry

end NUMINAMATH_GPT_ratio_evaluation_l2055_205596


namespace NUMINAMATH_GPT_dog_paws_ground_l2055_205514

theorem dog_paws_ground (total_dogs : ℕ) (two_thirds_back_legs : ℕ) (remaining_dogs_four_legs : ℕ) (two_paws_per_back_leg_dog : ℕ) (four_paws_per_four_leg_dog : ℕ) :
  total_dogs = 24 →
  two_thirds_back_legs = 2 * total_dogs / 3 →
  remaining_dogs_four_legs = total_dogs - two_thirds_back_legs →
  two_paws_per_back_leg_dog = 2 →
  four_paws_per_four_leg_dog = 4 →
  (two_thirds_back_legs * two_paws_per_back_leg_dog + remaining_dogs_four_legs * four_paws_per_four_leg_dog) = 64 := 
by 
  sorry

end NUMINAMATH_GPT_dog_paws_ground_l2055_205514


namespace NUMINAMATH_GPT_determine_var_phi_l2055_205599

open Real

theorem determine_var_phi (φ : ℝ) (h₀ : 0 ≤ φ ∧ φ ≤ 2 * π) :
  (∀ x, sin (x + φ) = sin (x - π / 6)) → φ = 11 * π / 6 :=
by
  sorry

end NUMINAMATH_GPT_determine_var_phi_l2055_205599


namespace NUMINAMATH_GPT_total_pages_in_book_l2055_205560

theorem total_pages_in_book (P : ℕ) 
  (h1 : 7 / 13 * P = P - 96 - 5 / 9 * (P - 7 / 13 * P))
  (h2 : 96 = 4 / 9 * (P - 7 / 13 * P)) : 
  P = 468 :=
 by 
    sorry

end NUMINAMATH_GPT_total_pages_in_book_l2055_205560


namespace NUMINAMATH_GPT_scientific_notation_proof_l2055_205577

-- Given number is 657,000
def number : ℕ := 657000

-- Scientific notation of the given number
def scientific_notation (n : ℕ) : Prop :=
    n = 657000 ∧ (6.57 : ℝ) * (10 : ℝ)^5 = 657000

theorem scientific_notation_proof : scientific_notation number :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_proof_l2055_205577


namespace NUMINAMATH_GPT_range_func_l2055_205525

noncomputable def func (x : ℝ) : ℝ := x + 4 / x

theorem range_func (x : ℝ) (hx : x ≠ 0) : func x ≤ -4 ∨ func x ≥ 4 := by
  sorry

end NUMINAMATH_GPT_range_func_l2055_205525


namespace NUMINAMATH_GPT_triple_comp_g_of_2_l2055_205501

def g (n : ℕ) : ℕ :=
  if n ≤ 3 then n^3 - 2 else 4 * n + 1

theorem triple_comp_g_of_2 : g (g (g 2)) = 101 := by
  sorry

end NUMINAMATH_GPT_triple_comp_g_of_2_l2055_205501


namespace NUMINAMATH_GPT_instructors_teach_together_in_360_days_l2055_205572

def Felicia_teaches_every := 5
def Greg_teaches_every := 3
def Hannah_teaches_every := 9
def Ian_teaches_every := 2
def Joy_teaches_every := 8

def lcm_multiple (a b c d e : ℕ) : ℕ := Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e)))

theorem instructors_teach_together_in_360_days :
  lcm_multiple Felicia_teaches_every
               Greg_teaches_every
               Hannah_teaches_every
               Ian_teaches_every
               Joy_teaches_every = 360 :=
by
  -- Since the real proof is omitted, we close with sorry
  sorry

end NUMINAMATH_GPT_instructors_teach_together_in_360_days_l2055_205572


namespace NUMINAMATH_GPT_mask_distribution_l2055_205537

theorem mask_distribution (x : ℕ) (total_masks_3 : ℕ) (total_masks_4 : ℕ)
    (h1 : total_masks_3 = 3 * x + 20)
    (h2 : total_masks_4 = 4 * x - 25) :
    3 * x + 20 = 4 * x - 25 :=
by
  sorry

end NUMINAMATH_GPT_mask_distribution_l2055_205537


namespace NUMINAMATH_GPT_my_op_identity_l2055_205544

def my_op (a b : ℕ) : ℕ := a + b + a * b

theorem my_op_identity (a : ℕ) : my_op (my_op a 1) 2 = 6 * a + 5 :=
by
  sorry

end NUMINAMATH_GPT_my_op_identity_l2055_205544


namespace NUMINAMATH_GPT_largest_power_dividing_factorial_l2055_205531

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2015) : ∃ k : ℕ, (2015^k ∣ n!) ∧ k = 67 :=
by
  sorry

end NUMINAMATH_GPT_largest_power_dividing_factorial_l2055_205531


namespace NUMINAMATH_GPT_sam_investment_time_l2055_205565

theorem sam_investment_time (P r : ℝ) (n A t : ℕ) (hP : P = 8000) (hr : r = 0.10) (hn : n = 2) (hA : A = 8820) :
  A = P * (1 + r / n) ^ (n * t) → t = 1 :=
by
  sorry

end NUMINAMATH_GPT_sam_investment_time_l2055_205565


namespace NUMINAMATH_GPT_problem1_problem2_l2055_205552

-- Problem 1: Prove that 2023 * 2023 - 2024 * 2022 = 1
theorem problem1 : 2023 * 2023 - 2024 * 2022 = 1 := 
by 
  sorry

-- Problem 2: Prove that (-4 * x * y^3) * (1/2 * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4
theorem problem2 (x y : ℝ) : (-4 * x * y^3) * ((1/2) * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2055_205552


namespace NUMINAMATH_GPT_smallest_b_value_is_6_l2055_205541

noncomputable def smallest_b_value (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : ℝ :=
b

theorem smallest_b_value_is_6 (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : 
  smallest_b_value a b c h_arith h_pos h_prod = 6 :=
sorry

end NUMINAMATH_GPT_smallest_b_value_is_6_l2055_205541


namespace NUMINAMATH_GPT_log_inequality_l2055_205512

theorem log_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
    (Real.log (c ^ 2) / Real.log (a + b) + Real.log (a ^ 2) / Real.log (b + c) + Real.log (b ^ 2) / Real.log (c + a)) ≥ 3 :=
sorry

end NUMINAMATH_GPT_log_inequality_l2055_205512


namespace NUMINAMATH_GPT_n_square_divisible_by_144_l2055_205583

theorem n_square_divisible_by_144 (n : ℤ) (hn : n > 0)
  (hw : ∃ k : ℤ, n = 12 * k) : ∃ m : ℤ, n^2 = 144 * m :=
by {
  sorry
}

end NUMINAMATH_GPT_n_square_divisible_by_144_l2055_205583


namespace NUMINAMATH_GPT_gcd_of_g_y_and_y_l2055_205567

theorem gcd_of_g_y_and_y (y : ℤ) (h : 9240 ∣ y) : Int.gcd ((5 * y + 3) * (11 * y + 2) * (17 * y + 8) * (4 * y + 7)) y = 168 := by
  sorry

end NUMINAMATH_GPT_gcd_of_g_y_and_y_l2055_205567


namespace NUMINAMATH_GPT_gcd_7392_15015_l2055_205547

-- Define the two numbers
def num1 : ℕ := 7392
def num2 : ℕ := 15015

-- State the theorem and use sorry to omit the proof
theorem gcd_7392_15015 : Nat.gcd num1 num2 = 1 := 
  by sorry

end NUMINAMATH_GPT_gcd_7392_15015_l2055_205547


namespace NUMINAMATH_GPT_inequality_solution_set_l2055_205592

theorem inequality_solution_set (a : ℤ) : 
  (∀ x : ℤ, (1 + a) * x > 1 + a → x < 1) → a < -1 :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l2055_205592
