import Mathlib

namespace NUMINAMATH_GPT_unique_solution_j_l2002_200271

theorem unique_solution_j (j : ℝ) : (∀ x : ℝ, (2 * x + 7) * (x - 5) = -43 + j * x) → (j = 5 ∨ j = -11) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_j_l2002_200271


namespace NUMINAMATH_GPT_strawberries_left_l2002_200224

theorem strawberries_left (initial : ℝ) (eaten : ℝ) (remaining : ℝ) : initial = 78.0 → eaten = 42.0 → remaining = 36.0 → initial - eaten = remaining :=
by
  sorry

end NUMINAMATH_GPT_strawberries_left_l2002_200224


namespace NUMINAMATH_GPT_find_a7_l2002_200201

theorem find_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ)
  (h : x^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 +
            a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 +
            a_7 * (x + 1)^7 + a_8 * (x + 1)^8) : 
  a_7 = -8 := 
sorry

end NUMINAMATH_GPT_find_a7_l2002_200201


namespace NUMINAMATH_GPT_Z_divisible_by_10001_l2002_200293

def is_eight_digit_integer (Z : Nat) : Prop :=
  (10^7 ≤ Z) ∧ (Z < 10^8)

def first_four_equal_last_four (Z : Nat) : Prop :=
  ∃ (a b c d : Nat), a ≠ 0 ∧ (Z = 1001 * (1000 * a + 100 * b + 10 * c + d))

theorem Z_divisible_by_10001 (Z : Nat) (h1 : is_eight_digit_integer Z) (h2 : first_four_equal_last_four Z) : 
  10001 ∣ Z :=
sorry

end NUMINAMATH_GPT_Z_divisible_by_10001_l2002_200293


namespace NUMINAMATH_GPT_student_ratio_l2002_200299

theorem student_ratio (total_students below_eight eight_years above_eight : ℕ) 
  (h1 : below_eight = total_students * 20 / 100) 
  (h2 : eight_years = 72) 
  (h3 : total_students = 150) 
  (h4 : total_students = below_eight + eight_years + above_eight) :
  (above_eight / eight_years) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_student_ratio_l2002_200299


namespace NUMINAMATH_GPT_equation_infinitely_many_solutions_iff_b_eq_neg9_l2002_200222

theorem equation_infinitely_many_solutions_iff_b_eq_neg9 (b : ℤ) :
  (∀ x : ℤ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  sorry

end NUMINAMATH_GPT_equation_infinitely_many_solutions_iff_b_eq_neg9_l2002_200222


namespace NUMINAMATH_GPT_only_set_C_is_pythagorean_triple_l2002_200265

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem only_set_C_is_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 7 ∧
  ¬ is_pythagorean_triple 15 20 25 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 1 3 5 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_only_set_C_is_pythagorean_triple_l2002_200265


namespace NUMINAMATH_GPT_estimated_white_balls_is_correct_l2002_200248

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of trials
def trials : ℕ := 100

-- Define the number of times a red ball is drawn
def red_draws : ℕ := 80

-- Define the function to estimate the number of red balls based on the frequency
def estimated_red_balls (total_balls : ℕ) (red_draws : ℕ) (trials : ℕ) : ℕ :=
  total_balls * red_draws / trials

-- Define the function to estimate the number of white balls
def estimated_white_balls (total_balls : ℕ) (estimated_red_balls : ℕ) : ℕ :=
  total_balls - estimated_red_balls

-- State the theorem to prove the estimated number of white balls
theorem estimated_white_balls_is_correct : 
  estimated_white_balls total_balls (estimated_red_balls total_balls red_draws trials) = 2 :=
by
  sorry

end NUMINAMATH_GPT_estimated_white_balls_is_correct_l2002_200248


namespace NUMINAMATH_GPT_days_to_finish_together_l2002_200244

-- Define the work rate of B
def work_rate_B : ℚ := 1 / 12

-- Define the work rate of A
def work_rate_A : ℚ := 2 * work_rate_B

-- Combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Prove that the number of days required for A and B to finish the work together is 4
theorem days_to_finish_together : (1 / combined_work_rate) = 4 := 
by
  sorry

end NUMINAMATH_GPT_days_to_finish_together_l2002_200244


namespace NUMINAMATH_GPT_partners_count_l2002_200245

theorem partners_count (P A : ℕ) (h1 : P / A = 2 / 63) (h2 : P / (A + 50) = 1 / 34) : P = 20 :=
sorry

end NUMINAMATH_GPT_partners_count_l2002_200245


namespace NUMINAMATH_GPT_giant_lollipop_calories_l2002_200241

-- Definitions based on the conditions
def sugar_per_chocolate_bar := 10
def chocolate_bars_bought := 14
def sugar_in_giant_lollipop := 37
def total_sugar := 177
def calories_per_gram_of_sugar := 4

-- Prove that the number of calories in the giant lollipop is 148 given the conditions
theorem giant_lollipop_calories : (sugar_in_giant_lollipop * calories_per_gram_of_sugar) = 148 := by
  sorry

end NUMINAMATH_GPT_giant_lollipop_calories_l2002_200241


namespace NUMINAMATH_GPT_oak_trees_remaining_l2002_200250

theorem oak_trees_remaining (initial_trees cut_down_trees remaining_trees : ℕ)
  (h1 : initial_trees = 9)
  (h2 : cut_down_trees = 2)
  (h3 : remaining_trees = initial_trees - cut_down_trees) :
  remaining_trees = 7 :=
by 
  sorry

end NUMINAMATH_GPT_oak_trees_remaining_l2002_200250


namespace NUMINAMATH_GPT_base7_to_base10_l2002_200223

theorem base7_to_base10 : 6 * 7^3 + 4 * 7^2 + 2 * 7^1 + 3 * 7^0 = 2271 := by
  sorry

end NUMINAMATH_GPT_base7_to_base10_l2002_200223


namespace NUMINAMATH_GPT_assign_questions_to_students_l2002_200263

theorem assign_questions_to_students:
  ∃ (assignment : Fin 20 → Fin 20), 
  (∀ s : Fin 20, ∃ q1 q2 : Fin 20, (assignment s = q1 ∨ assignment s = q2) ∧ q1 ≠ q2 ∧ ∀ q : Fin 20, ∃ s1 s2 : Fin 20, (assignment s1 = q ∧ assignment s2 = q) ∧ s1 ≠ s2) :=
by
  sorry

end NUMINAMATH_GPT_assign_questions_to_students_l2002_200263


namespace NUMINAMATH_GPT_center_of_circle_l2002_200231

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 4 * x^2 - 8 * x + 4 * y^2 - 24 * y - 36 = 0

-- Define what it means to be the center of the circle, which is (h, k)
def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 1

-- The statement that we need to prove
theorem center_of_circle : is_center 1 3 :=
sorry

end NUMINAMATH_GPT_center_of_circle_l2002_200231


namespace NUMINAMATH_GPT_acme_vowel_soup_sequences_l2002_200246

-- Define the vowels and their frequencies
def vowels : List (Char × ℕ) := [('A', 6), ('E', 6), ('I', 6), ('O', 4), ('U', 4)]

-- Noncomputable definition to calculate the total number of sequences
noncomputable def number_of_sequences : ℕ :=
  let single_vowel_choices := 6 + 6 + 6 + 4 + 4
  single_vowel_choices^5

-- Theorem stating the number of five-letter sequences
theorem acme_vowel_soup_sequences : number_of_sequences = 11881376 := by
  sorry

end NUMINAMATH_GPT_acme_vowel_soup_sequences_l2002_200246


namespace NUMINAMATH_GPT_total_amount_l2002_200298

theorem total_amount (a b c : ℕ) (h1 : a * 5 = b * 3) (h2 : c * 5 = b * 9) (h3 : b = 50) :
  a + b + c = 170 := by
  sorry

end NUMINAMATH_GPT_total_amount_l2002_200298


namespace NUMINAMATH_GPT_value_of_m_l2002_200237

theorem value_of_m (m : ℝ) :
  let A := {2, 3}
  let B := {x : ℝ | m * x - 6 = 0}
  (B ⊆ A) → (m = 0 ∨ m = 2 ∨ m = 3) :=
by
  intros A B h
  sorry

end NUMINAMATH_GPT_value_of_m_l2002_200237


namespace NUMINAMATH_GPT_abs_neg_seventeen_l2002_200297

theorem abs_neg_seventeen : |(-17 : ℤ)| = 17 := by
  sorry

end NUMINAMATH_GPT_abs_neg_seventeen_l2002_200297


namespace NUMINAMATH_GPT_inequality_holds_l2002_200226

theorem inequality_holds (x : ℝ) (hx : 0 < x ∧ x < 4) :
  ∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y) :=
by
  intros y hy_gt_zero
  sorry

end NUMINAMATH_GPT_inequality_holds_l2002_200226


namespace NUMINAMATH_GPT_range_of_a_l2002_200269

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else a * x^2 - x + 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ -1) ↔ (a ≥ 1/12) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2002_200269


namespace NUMINAMATH_GPT_calc_result_l2002_200260

theorem calc_result : (377 / 13 / 29 * 1 / 4 / 2) = 0.125 := 
by sorry

end NUMINAMATH_GPT_calc_result_l2002_200260


namespace NUMINAMATH_GPT_set_equality_example_l2002_200292

theorem set_equality_example : {x : ℕ | 2 * x + 3 ≥ 3 * x} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_set_equality_example_l2002_200292


namespace NUMINAMATH_GPT_a_minus_b_perfect_square_l2002_200276

theorem a_minus_b_perfect_square (a b : ℕ) (h : 2 * a^2 + a = 3 * b^2 + b) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, a - b = k^2 :=
by sorry

end NUMINAMATH_GPT_a_minus_b_perfect_square_l2002_200276


namespace NUMINAMATH_GPT_truth_of_q_l2002_200280

variable {p q : Prop}

theorem truth_of_q (hnp : ¬ p) (hpq : p ∨ q) : q :=
  by
  sorry

end NUMINAMATH_GPT_truth_of_q_l2002_200280


namespace NUMINAMATH_GPT_sin_double_angle_l2002_200227

-- Define the conditions and the goal
theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_l2002_200227


namespace NUMINAMATH_GPT_solve_for_diamond_l2002_200257

theorem solve_for_diamond (d : ℕ) (h1 : d * 9 + 6 = d * 10 + 3) (h2 : d < 10) : d = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_diamond_l2002_200257


namespace NUMINAMATH_GPT_probability_rain_at_most_3_days_in_july_l2002_200273

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_rain_at_most_3_days_in_july :
  let p := 1 / 5
  let n := 31
  let sum_prob := binomial_probability n 0 p + binomial_probability n 1 p + binomial_probability n 2 p + binomial_probability n 3 p
  abs (sum_prob - 0.125) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_probability_rain_at_most_3_days_in_july_l2002_200273


namespace NUMINAMATH_GPT_nine_fifths_sum_l2002_200274

open Real

theorem nine_fifths_sum (a b: ℝ) (ha: a > 0) (hb: b > 0)
    (h1: a * (sqrt a) + b * (sqrt b) = 183) 
    (h2: a * (sqrt b) + b * (sqrt a) = 182) : 
    9 / 5 * (a + b) = 657 := 
by 
    sorry

end NUMINAMATH_GPT_nine_fifths_sum_l2002_200274


namespace NUMINAMATH_GPT_circle_radius_squared_l2002_200278

-- Let r be the radius of the circle.
-- Let AB and CD be chords of the circle with lengths 10 and 7 respectively.
-- Let the extensions of AB and CD intersect at a point P outside the circle.
-- Let ∠APD be 60 degrees.
-- Let BP be 8.

theorem circle_radius_squared
  (r : ℝ)       -- radius of the circle
  (AB : ℝ)     -- length of chord AB
  (CD : ℝ)     -- length of chord CD
  (APD : ℝ)    -- angle APD
  (BP : ℝ)     -- length of segment BP
  (hAB : AB = 10)
  (hCD : CD = 7)
  (hAPD : APD = 60)
  (hBP : BP = 8)
  : r^2 = 73 := 
  sorry

end NUMINAMATH_GPT_circle_radius_squared_l2002_200278


namespace NUMINAMATH_GPT_triangle_bisector_ratio_l2002_200264

theorem triangle_bisector_ratio (AB BC CA : ℝ) (h_AB_pos : 0 < AB) (h_BC_pos : 0 < BC) (h_CA_pos : 0 < CA)
  (AA1_bisector : True) (BB1_bisector : True) (O_intersection : True) : 
  AA1 / OA1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_bisector_ratio_l2002_200264


namespace NUMINAMATH_GPT_average_employees_per_week_l2002_200284

theorem average_employees_per_week (x : ℝ)
  (h1 : ∀ (x : ℝ), ∃ y : ℝ, y = x + 200)
  (h2 : ∀ (x : ℝ), ∃ z : ℝ, z = x + 150)
  (h3 : ∀ (x : ℝ), ∃ w : ℝ, w = 2 * (x + 150))
  (h4 : ∀ (w : ℝ), w = 400) :
  (250 + 50 + 200 + 400) / 4 = 225 :=
by 
  sorry

end NUMINAMATH_GPT_average_employees_per_week_l2002_200284


namespace NUMINAMATH_GPT_intersection_of_sets_l2002_200238

noncomputable def setA : Set ℝ := {x | 1 / (x - 1) ≤ 1}
def setB : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_sets : setA ∩ setB = {-1, 0, 2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l2002_200238


namespace NUMINAMATH_GPT_sin_2pi_minus_alpha_l2002_200252

theorem sin_2pi_minus_alpha (α : ℝ) (h₁ : Real.cos (α + Real.pi) = Real.sqrt 3 / 2) (h₂ : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
    Real.sin (2 * Real.pi - α) = -1 / 2 := 
sorry

end NUMINAMATH_GPT_sin_2pi_minus_alpha_l2002_200252


namespace NUMINAMATH_GPT_parallel_lines_slope_l2002_200242

theorem parallel_lines_slope (a : ℝ) :
  (∃ (a : ℝ), ∀ x y, (3 * y - a = 9 * x + 1) ∧ (y - 2 = (2 * a - 3) * x)) → a = 3 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l2002_200242


namespace NUMINAMATH_GPT_sets_equal_sufficient_condition_l2002_200220

variable (a : ℝ)

-- Define sets A and B
def A (x : ℝ) : Prop := 0 < a * x + 1 ∧ a * x + 1 ≤ 5
def B (x : ℝ) : Prop := -1/2 < x ∧ x ≤ 2

-- Statement for Part 1: Sets A and B can be equal if and only if a = 2
theorem sets_equal (h : ∀ x, A a x ↔ B x) : a = 2 :=
sorry

-- Statement for Part 2: Proposition p ⇒ q holds if and only if a > 2 or a < -8
theorem sufficient_condition (h : ∀ x, A a x → B x) (h_neq : ∃ x, B x ∧ ¬A a x) : a > 2 ∨ a < -8 :=
sorry

end NUMINAMATH_GPT_sets_equal_sufficient_condition_l2002_200220


namespace NUMINAMATH_GPT_equation_solution_l2002_200287

theorem equation_solution:
  ∀ (x y : ℝ), 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) := 
by
  sorry

end NUMINAMATH_GPT_equation_solution_l2002_200287


namespace NUMINAMATH_GPT_ken_gets_back_16_dollars_l2002_200290

-- Given constants and conditions
def price_per_pound_steak : ℕ := 7
def pounds_of_steak : ℕ := 2
def price_carton_eggs : ℕ := 3
def price_gallon_milk : ℕ := 4
def price_pack_bagels : ℕ := 6
def bill_20_dollar : ℕ := 20
def bill_10_dollar : ℕ := 10
def bill_5_dollar_count : ℕ := 2
def coin_1_dollar_count : ℕ := 3

-- Calculate total cost of items
def total_cost_items : ℕ :=
  (pounds_of_steak * price_per_pound_steak) +
  price_carton_eggs +
  price_gallon_milk +
  price_pack_bagels

-- Calculate total amount paid
def total_amount_paid : ℕ :=
  bill_20_dollar +
  bill_10_dollar +
  (bill_5_dollar_count * 5) +
  (coin_1_dollar_count * 1)

-- Theorem statement to be proved
theorem ken_gets_back_16_dollars :
  total_amount_paid - total_cost_items = 16 := by
  sorry

end NUMINAMATH_GPT_ken_gets_back_16_dollars_l2002_200290


namespace NUMINAMATH_GPT_correct_order_shopping_process_l2002_200219

/-- Definition of each step --/
def step1 : String := "The buyer logs into the Taobao website to select products."
def step2 : String := "The buyer selects the product, clicks the buy button, and pays through Alipay."
def step3 : String := "Upon receiving the purchase information, the seller ships the goods to the buyer through a logistics company."
def step4 : String := "The buyer receives the goods, inspects them for any issues, and confirms receipt online."
def step5 : String := "After receiving the buyer's confirmation of receipt, the Taobao website transfers the payment from Alipay to the seller."

/-- The correct sequence of steps --/
def correct_sequence : List String := [
  "The buyer logs into the Taobao website to select products.",
  "The buyer selects the product, clicks the buy button, and pays through Alipay.",
  "Upon receiving the purchase information, the seller ships the goods to the buyer through a logistics company.",
  "The buyer receives the goods, inspects them for any issues, and confirms receipt online.",
  "After receiving the buyer's confirmation of receipt, the Taobao website transfers the payment from Alipay to the seller."
]

theorem correct_order_shopping_process :
  [step1, step2, step3, step4, step5] = correct_sequence :=
by
  sorry

end NUMINAMATH_GPT_correct_order_shopping_process_l2002_200219


namespace NUMINAMATH_GPT_greatest_of_consecutive_integers_with_sum_39_l2002_200234

theorem greatest_of_consecutive_integers_with_sum_39 :
  ∃ x : ℤ, x + (x + 1) + (x + 2) = 39 ∧ max (max x (x + 1)) (x + 2) = 14 :=
by
  sorry

end NUMINAMATH_GPT_greatest_of_consecutive_integers_with_sum_39_l2002_200234


namespace NUMINAMATH_GPT_student_number_in_eighth_group_l2002_200225

-- Definitions corresponding to each condition
def students : ℕ := 50
def group_size : ℕ := 5
def third_group_student_number : ℕ := 12
def kth_group_number (k : ℕ) (n : ℕ) : ℕ := n + (k - 3) * group_size

-- Main statement to prove
theorem student_number_in_eighth_group :
  kth_group_number 8 third_group_student_number = 37 :=
  by
  sorry

end NUMINAMATH_GPT_student_number_in_eighth_group_l2002_200225


namespace NUMINAMATH_GPT_same_school_probability_l2002_200289

theorem same_school_probability :
  let total_teachers : ℕ := 6
  let teachers_from_school_A : ℕ := 3
  let teachers_from_school_B : ℕ := 3
  let ways_to_choose_2_from_6 : ℕ := Nat.choose total_teachers 2
  let ways_to_choose_2_from_A := Nat.choose teachers_from_school_A 2
  let ways_to_choose_2_from_B := Nat.choose teachers_from_school_B 2
  let same_school_ways : ℕ := ways_to_choose_2_from_A + ways_to_choose_2_from_B
  let probability := (same_school_ways : ℚ) / ways_to_choose_2_from_6 
  probability = (2 : ℚ) / (5 : ℚ) := by sorry

end NUMINAMATH_GPT_same_school_probability_l2002_200289


namespace NUMINAMATH_GPT_area_of_triangle_AMN_is_correct_l2002_200272

noncomputable def area_triangle_AMN : ℝ :=
  let A := (120 + 56 * Real.sqrt 3) / 3
  let M := (12 + 20 * Real.sqrt 3) / 3
  let N := 4 * Real.sqrt 3 + 20
  (A * N) / 2

theorem area_of_triangle_AMN_is_correct :
  area_triangle_AMN = (224 * Real.sqrt 3 + 240) / 3 := sorry

end NUMINAMATH_GPT_area_of_triangle_AMN_is_correct_l2002_200272


namespace NUMINAMATH_GPT_triathlon_bike_speed_l2002_200277

theorem triathlon_bike_speed :
  ∀ (t_total t_swim t_run t_bike : ℚ) (d_swim d_run d_bike : ℚ)
    (v_swim v_run r_bike : ℚ),
  t_total = 3 →
  d_swim = 1 / 2 →
  v_swim = 1 →
  d_run = 4 →
  v_run = 5 →
  d_bike = 10 →
  t_swim = d_swim / v_swim →
  t_run = d_run / v_run →
  t_bike = t_total - (t_swim + t_run) →
  r_bike = d_bike / t_bike →
  r_bike = 100 / 17 :=
by
  intros t_total t_swim t_run t_bike d_swim d_run d_bike v_swim v_run r_bike
         h_total h_d_swim h_v_swim h_d_run h_v_run h_d_bike h_t_swim h_t_run h_t_bike h_r_bike
  sorry

end NUMINAMATH_GPT_triathlon_bike_speed_l2002_200277


namespace NUMINAMATH_GPT_min_m_min_expression_l2002_200253

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Part (Ⅰ)
theorem min_m (m : ℝ) (h : ∃ x₀ : ℝ, f x₀ ≤ m) : m ≥ 2 := sorry

-- Part (Ⅱ)
theorem min_expression (a b : ℝ) (h1 : 3 * a + b = 2) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / (2 * a) + 1 / (a + b)) ≥ 2 := sorry

end NUMINAMATH_GPT_min_m_min_expression_l2002_200253


namespace NUMINAMATH_GPT_problem_statement_l2002_200214

variable (p q r s : ℝ) (ω : ℂ)

theorem problem_statement (hp : p ≠ -1) (hq : q ≠ -1) (hr : r ≠ -1) (hs : s ≠ -1) 
  (hω : ω ^ 4 = 1) (hω_ne : ω ≠ 1)
  (h_eq : (1 / (p + ω) + 1 / (q + ω) + 1 / (r + ω) + 1 / (s + ω)) = 3 / ω^2) :
  1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1) + 1 / (s + 1) = 3 := 
by sorry

end NUMINAMATH_GPT_problem_statement_l2002_200214


namespace NUMINAMATH_GPT_length_of_AB_l2002_200235

-- Define the parabola and the line passing through the focus F
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line (x y : ℝ) : Prop := y = x - 1

theorem length_of_AB : 
  (∃ F : ℝ × ℝ, F = (1, 0) ∧ line F.1 F.2) →
  (∃ A B : ℝ × ℝ, parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A ≠ B ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 64)) :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l2002_200235


namespace NUMINAMATH_GPT_minimize_m_at_l2002_200281

noncomputable def m (x y : ℝ) : ℝ := 4 * x ^ 2 - 12 * x * y + 10 * y ^ 2 + 4 * y + 9

theorem minimize_m_at (x y : ℝ) : m x y = 5 ↔ (x = -3 ∧ y = -2) := 
sorry

end NUMINAMATH_GPT_minimize_m_at_l2002_200281


namespace NUMINAMATH_GPT_circle_center_sum_l2002_200230

/-- Given the equation of a circle, prove that the sum of the x and y coordinates of the center is -1. -/
theorem circle_center_sum (x y : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by 
  sorry

end NUMINAMATH_GPT_circle_center_sum_l2002_200230


namespace NUMINAMATH_GPT_coordinates_of_point_in_fourth_quadrant_l2002_200282

-- Define the conditions as separate hypotheses
def point_in_fourth_quadrant (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- State the main theorem
theorem coordinates_of_point_in_fourth_quadrant
  (x y : ℝ) (h1 : point_in_fourth_quadrant x y) (h2 : |x| = 3) (h3 : |y| = 5) :
  (x = 3) ∧ (y = -5) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_in_fourth_quadrant_l2002_200282


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2002_200233

noncomputable def min_value_expression (x y : ℝ) : ℝ := 
  (x + 1)^2 / (x + 2) + 3 / (x + 2) + y^2 / (y + 1)

theorem minimum_value_of_expression :
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → x + y = 2 → min_value_expression x y = 14 / 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2002_200233


namespace NUMINAMATH_GPT_tangent_line_y_intercept_at_P_1_12_is_9_l2002_200288

noncomputable def curve (x : ℝ) : ℝ := x^3 + 11

noncomputable def tangent_slope_at (x : ℝ) : ℝ := 3 * x^2

noncomputable def tangent_line_y_intercept : ℝ :=
  let P : ℝ × ℝ := (1, curve 1)
  let slope := tangent_slope_at 1
  P.snd - slope * P.fst

theorem tangent_line_y_intercept_at_P_1_12_is_9 :
  tangent_line_y_intercept = 9 :=
sorry

end NUMINAMATH_GPT_tangent_line_y_intercept_at_P_1_12_is_9_l2002_200288


namespace NUMINAMATH_GPT_isosceles_trapezoid_l2002_200279

-- Define a type for geometric points
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define structures for geometric properties
structure Trapezoid :=
  (A B C D M N : Point)
  (is_midpoint_M : 2 * M.x = A.x + B.x ∧ 2 * M.y = A.y + B.y)
  (is_midpoint_N : 2 * N.x = C.x + D.x ∧ 2 * N.y = C.y + D.y)
  (AB_parallel_CD : (B.y - A.y) * (D.x - C.x) = (B.x - A.x) * (D.y - C.y)) -- AB || CD
  (MN_perpendicular_AB_CD : (N.y - M.y) * (B.y - A.y) + (N.x - M.x) * (B.x - A.x) = 0 ∧
                            (N.y - M.y) * (D.y - C.y) + (N.x - M.x) * (D.x - C.x) = 0) -- MN ⊥ AB && MN ⊥ CD

-- The isosceles condition
def is_isosceles (T : Trapezoid) : Prop :=
  ((T.A.x - T.D.x) ^ 2 + (T.A.y - T.D.y) ^ 2) = ((T.B.x - T.C.x) ^ 2 + (T.B.y - T.C.y) ^ 2)

-- The theorem statement
theorem isosceles_trapezoid (T : Trapezoid) : is_isosceles T :=
by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_l2002_200279


namespace NUMINAMATH_GPT_tangent_parallel_and_point_P_l2002_200275

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_parallel_and_point_P (P : ℝ × ℝ) (hP1 : P = (1, f 1)) (hP2 : P = (-1, f (-1))) :
  (f 1 = 3 ∧ f (-1) = 3) ∧ (deriv f 1 = 2 ∧ deriv f (-1) = 2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_parallel_and_point_P_l2002_200275


namespace NUMINAMATH_GPT_remaining_volume_is_21_l2002_200283

-- Definitions of edge lengths and volumes
def edge_length_original : ℕ := 3
def edge_length_small : ℕ := 1
def volume (a : ℕ) : ℕ := a ^ 3

-- Volumes of the original cube and the small cubes
def volume_original : ℕ := volume edge_length_original
def volume_small : ℕ := volume edge_length_small
def number_of_faces : ℕ := 6
def total_volume_cut : ℕ := number_of_faces * volume_small

-- Volume of the remaining part
def volume_remaining : ℕ := volume_original - total_volume_cut

-- Proof statement
theorem remaining_volume_is_21 : volume_remaining = 21 := by
  sorry

end NUMINAMATH_GPT_remaining_volume_is_21_l2002_200283


namespace NUMINAMATH_GPT_product_of_two_numbers_l2002_200258

theorem product_of_two_numbers (a b : ℕ) (H1 : Nat.gcd a b = 20) (H2 : Nat.lcm a b = 128) : a * b = 2560 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2002_200258


namespace NUMINAMATH_GPT_change_after_buying_tickets_l2002_200211

def cost_per_ticket := 8
def number_of_tickets := 2
def total_money := 25

theorem change_after_buying_tickets :
  total_money - number_of_tickets * cost_per_ticket = 9 := by
  sorry

end NUMINAMATH_GPT_change_after_buying_tickets_l2002_200211


namespace NUMINAMATH_GPT_CarmenBrushLengthInCentimeters_l2002_200205

-- Given conditions
def CarlaBrushLengthInInches : ℝ := 12
def CarmenBrushPercentIncrease : ℝ := 0.5
def InchToCentimeterConversionFactor : ℝ := 2.5

-- Question: What is Carmen's brush length in centimeters?
-- Proof Goal: Prove that Carmen's brush length in centimeters is 45 cm.
theorem CarmenBrushLengthInCentimeters :
  let CarmenBrushLengthInInches := CarlaBrushLengthInInches * (1 + CarmenBrushPercentIncrease)
  CarmenBrushLengthInInches * InchToCentimeterConversionFactor = 45 := by
  -- sorry is used as a placeholder for the completed proof
  sorry

end NUMINAMATH_GPT_CarmenBrushLengthInCentimeters_l2002_200205


namespace NUMINAMATH_GPT_gray_area_l2002_200259

def center_C : (ℝ × ℝ) := (6, 5)
def center_D : (ℝ × ℝ) := (14, 5)
def radius_C : ℝ := 3
def radius_D : ℝ := 3

theorem gray_area :
  let area_rectangle := 8 * 5
  let area_sector_C := (1 / 2) * π * radius_C^2
  let area_sector_D := (1 / 2) * π * radius_D^2
  area_rectangle - (area_sector_C + area_sector_D) = 40 - 9 * π :=
by
  sorry

end NUMINAMATH_GPT_gray_area_l2002_200259


namespace NUMINAMATH_GPT_total_cost_correct_l2002_200236

-- Definitions of the constants based on given problem conditions
def cost_burger : ℕ := 5
def cost_pack_of_fries : ℕ := 2
def num_packs_of_fries : ℕ := 2
def cost_salad : ℕ := 3 * cost_pack_of_fries

-- The total cost calculation based on the conditions
def total_cost : ℕ := cost_burger + num_packs_of_fries * cost_pack_of_fries + cost_salad

-- The statement to prove that the total cost Benjamin paid is $15
theorem total_cost_correct : total_cost = 15 := by
  -- This is where the proof would go, but we're omitting it for now.
  sorry

end NUMINAMATH_GPT_total_cost_correct_l2002_200236


namespace NUMINAMATH_GPT_no_boys_love_cards_l2002_200207

def boys_love_marbles := 13
def total_marbles := 26
def marbles_per_boy := 2

theorem no_boys_love_cards (boys_love_marbles total_marbles marbles_per_boy : ℕ)
  (h1 : boys_love_marbles * marbles_per_boy = total_marbles) : 
  ∃ no_boys_love_cards : ℕ, no_boys_love_cards = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_boys_love_cards_l2002_200207


namespace NUMINAMATH_GPT_coeff_b_l2002_200240

noncomputable def g (a b c d e : ℝ) (x : ℝ) :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem coeff_b (a b c d e : ℝ):
  -- The function g(x) has roots at x = -1, 0, 1, 2
  (g a b c d e (-1) = 0) →
  (g a b c d e 0 = 0) →
  (g a b c d e 1 = 0) →
  (g a b c d e 2 = 0) →
  -- The function passes through the point (0, 3)
  (g a b c d e 0 = 3) →
  -- Assuming a = 1
  (a = 1) →
  -- Prove that b = -2
  b = -2 :=
by
  intros _ _ _ _ _ a_eq_1
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_coeff_b_l2002_200240


namespace NUMINAMATH_GPT_infinitely_many_c_exist_l2002_200206

theorem infinitely_many_c_exist :
  ∃ c: ℕ, ∃ x y z: ℕ, (x^2 - c) * (y^2 - c) = z^2 - c ∧ (x^2 + c) * (y^2 - c) = z^2 - c :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_c_exist_l2002_200206


namespace NUMINAMATH_GPT_julia_height_is_172_7_cm_l2002_200202

def julia_height_in_cm (height_in_inches : ℝ) (conversion_factor : ℝ) : ℝ :=
  height_in_inches * conversion_factor

theorem julia_height_is_172_7_cm :
  julia_height_in_cm 68 2.54 = 172.7 :=
by
  sorry

end NUMINAMATH_GPT_julia_height_is_172_7_cm_l2002_200202


namespace NUMINAMATH_GPT_solution1_solution2_solution3_solution4_solution5_l2002_200229

noncomputable def problem1 : ℤ :=
  -3 + 8 - 15 - 6

theorem solution1 : problem1 = -16 := by
  sorry

noncomputable def problem2 : ℚ :=
  -35 / -7 * (-1 / 7)

theorem solution2 : problem2 = -(5 / 7) := by
  sorry

noncomputable def problem3 : ℤ :=
  -2^2 - |2 - 5| / -3

theorem solution3 : problem3 = -3 := by
  sorry

noncomputable def problem4 : ℚ :=
  (1 / 2 + 5 / 6 - 7 / 12) * -24 

theorem solution4 : problem4 = -18 := by
  sorry

noncomputable def problem5 : ℚ :=
  (-99 - 6 / 11) * 22

theorem solution5 : problem5 = -2190 := by
  sorry

end NUMINAMATH_GPT_solution1_solution2_solution3_solution4_solution5_l2002_200229


namespace NUMINAMATH_GPT_total_cost_at_discount_l2002_200204

-- Definitions for conditions
def original_price_notebook : ℕ := 15
def original_price_planner : ℕ := 10
def discount_rate : ℕ := 20
def number_of_notebooks : ℕ := 4
def number_of_planners : ℕ := 8

-- Theorem statement for the proof
theorem total_cost_at_discount :
  let discounted_price_notebook := original_price_notebook - (original_price_notebook * discount_rate / 100)
  let discounted_price_planner := original_price_planner - (original_price_planner * discount_rate / 100)
  let total_cost := (number_of_notebooks * discounted_price_notebook) + (number_of_planners * discounted_price_planner)
  total_cost = 112 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_at_discount_l2002_200204


namespace NUMINAMATH_GPT_hannah_stocking_stuffers_l2002_200268

theorem hannah_stocking_stuffers (candy_caness : ℕ) (beanie_babies : ℕ) (books : ℕ) (kids : ℕ) : 
  candy_caness = 4 → 
  beanie_babies = 2 → 
  books = 1 → 
  kids = 3 → 
  candy_caness + beanie_babies + books = 7 → 
  7 * kids = 21 := 
by sorry

end NUMINAMATH_GPT_hannah_stocking_stuffers_l2002_200268


namespace NUMINAMATH_GPT_remainder_of_37_div_8_is_5_l2002_200251

theorem remainder_of_37_div_8_is_5 : ∃ A B : ℤ, 37 = 8 * A + B ∧ 0 ≤ B ∧ B < 8 ∧ B = 5 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_37_div_8_is_5_l2002_200251


namespace NUMINAMATH_GPT_orthogonal_vectors_y_value_l2002_200262

theorem orthogonal_vectors_y_value (y : ℝ) :
  (3 : ℝ) * (-1) + (4 : ℝ) * y = 0 → y = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_orthogonal_vectors_y_value_l2002_200262


namespace NUMINAMATH_GPT_minimum_value_amgm_l2002_200254

theorem minimum_value_amgm (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 27) : (a + 3 * b + 9 * c) ≥ 27 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_amgm_l2002_200254


namespace NUMINAMATH_GPT_general_term_formula_not_arithmetic_sequence_l2002_200249

noncomputable def geometric_sequence (n : ℕ) : ℕ := 2^n

theorem general_term_formula :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    (∃ (q : ℕ),
      ∀ n, a n = 2^n) :=
by
  sorry

theorem not_arithmetic_sequence :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    ¬(∃ m n p : ℕ, m < n ∧ n < p ∧ (2 * a n = a m + a p)) :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_not_arithmetic_sequence_l2002_200249


namespace NUMINAMATH_GPT_union_sets_l2002_200213

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

theorem union_sets : A ∪ B = {x | x ≥ 1} :=
  by
    sorry

end NUMINAMATH_GPT_union_sets_l2002_200213


namespace NUMINAMATH_GPT_restaurant_A2_probability_l2002_200243

noncomputable def prob_A2 (P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 : ℝ) : ℝ :=
  P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1

theorem restaurant_A2_probability :
  let P_A1 := 0.4
  let P_B1 := 0.6
  let P_A2_given_A1 := 0.6
  let P_A2_given_B1 := 0.5
  prob_A2 P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 = 0.54 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_A2_probability_l2002_200243


namespace NUMINAMATH_GPT_pacific_ocean_area_rounded_l2002_200210

def pacific_ocean_area : ℕ := 19996800

def ten_thousand : ℕ := 10000

noncomputable def pacific_ocean_area_in_ten_thousands (area : ℕ) : ℕ :=
  (area / ten_thousand + if (area % ten_thousand) >= (ten_thousand / 2) then 1 else 0)

theorem pacific_ocean_area_rounded :
  pacific_ocean_area_in_ten_thousands pacific_ocean_area = 2000 :=
by
  sorry

end NUMINAMATH_GPT_pacific_ocean_area_rounded_l2002_200210


namespace NUMINAMATH_GPT_union_of_A_B_l2002_200209

open Set

def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | -3 < x ∧ x ≤ 1}

theorem union_of_A_B : A ∪ B = {x : ℝ | -3 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_union_of_A_B_l2002_200209


namespace NUMINAMATH_GPT_neighbors_receive_mangoes_l2002_200215

-- Definitions of the conditions
def harvested_mangoes : ℕ := 560
def sold_mangoes : ℕ := harvested_mangoes / 2
def given_to_family : ℕ := 50
def num_neighbors : ℕ := 12

-- Calculation of mangoes left
def mangoes_left : ℕ := harvested_mangoes - sold_mangoes - given_to_family

-- The statement we want to prove
theorem neighbors_receive_mangoes : mangoes_left / num_neighbors = 19 := by
  sorry

end NUMINAMATH_GPT_neighbors_receive_mangoes_l2002_200215


namespace NUMINAMATH_GPT_gcd_90_252_eq_18_l2002_200270

theorem gcd_90_252_eq_18 : Nat.gcd 90 252 = 18 := 
sorry

end NUMINAMATH_GPT_gcd_90_252_eq_18_l2002_200270


namespace NUMINAMATH_GPT_solution_set_abs_inequality_l2002_200261

theorem solution_set_abs_inequality : {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_solution_set_abs_inequality_l2002_200261


namespace NUMINAMATH_GPT_square_side_increase_l2002_200208

theorem square_side_increase (p : ℝ) (h : (1 + p / 100)^2 = 1.69) : p = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_square_side_increase_l2002_200208


namespace NUMINAMATH_GPT_cat_finishes_food_on_tuesday_second_week_l2002_200218

def initial_cans : ℚ := 8
def extra_treat : ℚ := 1 / 6
def morning_diet : ℚ := 1 / 4
def evening_diet : ℚ := 1 / 5

def daily_consumption (morning_diet evening_diet : ℚ) : ℚ :=
  morning_diet + evening_diet

def first_day_consumption (daily_consumption extra_treat : ℚ) : ℚ :=
  daily_consumption + extra_treat

theorem cat_finishes_food_on_tuesday_second_week 
  (initial_cans extra_treat morning_diet evening_diet : ℚ)
  (h1 : initial_cans = 8)
  (h2 : extra_treat = 1 / 6)
  (h3 : morning_diet = 1 / 4)
  (h4 : evening_diet = 1 / 5) :
  -- The computation must be performed here or defined previously
  -- The proof of this theorem is the task, the result is postulated as a theorem
  final_day = "Tuesday (second week)" :=
sorry

end NUMINAMATH_GPT_cat_finishes_food_on_tuesday_second_week_l2002_200218


namespace NUMINAMATH_GPT_A_n_plus_B_n_eq_2n_cubed_l2002_200247

-- Definition of A_n given the grouping of positive integers
def A_n (n : ℕ) : ℕ :=
  let sum_first_n_squared := n * n * (n * n + 1) / 2
  let sum_first_n_minus_1_squared := (n - 1) * (n - 1) * ((n - 1) * (n - 1) + 1) / 2
  sum_first_n_squared - sum_first_n_minus_1_squared

-- Definition of B_n given the array of cubes of natural numbers
def B_n (n : ℕ) : ℕ := n * n * n - (n - 1) * (n - 1) * (n - 1)

-- The theorem to prove that A_n + B_n = 2n^3
theorem A_n_plus_B_n_eq_2n_cubed (n : ℕ) : A_n n + B_n n = 2 * n^3 := by
  sorry

end NUMINAMATH_GPT_A_n_plus_B_n_eq_2n_cubed_l2002_200247


namespace NUMINAMATH_GPT_value_of_x_is_two_l2002_200285

theorem value_of_x_is_two (x : ℝ) (h : x + x^3 = 10) : x = 2 :=
sorry

end NUMINAMATH_GPT_value_of_x_is_two_l2002_200285


namespace NUMINAMATH_GPT_probability_different_colors_l2002_200239

def total_chips : ℕ := 12

def blue_chips : ℕ := 5
def yellow_chips : ℕ := 3
def red_chips : ℕ := 4

def prob_diff_color (x y : ℕ) : ℚ :=
(x / total_chips) * (y / total_chips) + (y / total_chips) * (x / total_chips)

theorem probability_different_colors :
  prob_diff_color blue_chips yellow_chips +
  prob_diff_color blue_chips red_chips +
  prob_diff_color yellow_chips red_chips = 47 / 72 := by
sorry

end NUMINAMATH_GPT_probability_different_colors_l2002_200239


namespace NUMINAMATH_GPT_prime_dates_in_2008_l2002_200295

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_prime_date (month day : ℕ) : Prop := is_prime month ∧ is_prime day

noncomputable def prime_dates_2008 : ℕ :=
  let prime_months := [2, 3, 5, 7, 11]
  let prime_days_31 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let prime_days_30 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_days_29 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  prime_months.foldl (λ acc month => 
    acc + match month with
      | 2 => List.length prime_days_29
      | 3 | 5 | 7 => List.length prime_days_31
      | 11 => List.length prime_days_30
      | _ => 0
    ) 0

theorem prime_dates_in_2008 : 
  prime_dates_2008 = 53 :=
  sorry

end NUMINAMATH_GPT_prime_dates_in_2008_l2002_200295


namespace NUMINAMATH_GPT_fraction_distinctly_marked_l2002_200291

theorem fraction_distinctly_marked 
  (area_large_rectangle : ℕ)
  (fraction_shaded : ℚ)
  (fraction_further_marked : ℚ)
  (h_area_large_rectangle : area_large_rectangle = 15 * 24)
  (h_fraction_shaded : fraction_shaded = 1/3)
  (h_fraction_further_marked : fraction_further_marked = 1/2) :
  (fraction_further_marked * fraction_shaded = 1/6) :=
by
  sorry

end NUMINAMATH_GPT_fraction_distinctly_marked_l2002_200291


namespace NUMINAMATH_GPT_lucille_total_revenue_l2002_200255

theorem lucille_total_revenue (salary_ratio stock_ratio : ℕ) (salary_amount : ℝ) (h_ratio : salary_ratio / stock_ratio = 4 / 11) (h_salary : salary_amount = 800) : 
  ∃ total_revenue : ℝ, total_revenue = 3000 :=
by
  sorry

end NUMINAMATH_GPT_lucille_total_revenue_l2002_200255


namespace NUMINAMATH_GPT_area_comparison_perimeter_comparison_l2002_200216

-- Define side length of square and transformation to sides of the rectangle
variable (a : ℝ)

-- Conditions: side lengths of the rectangle relative to the square
def long_side : ℝ := 1.11 * a
def short_side : ℝ := 0.9 * a

-- Area calculations and comparison
def square_area : ℝ := a^2
def rectangle_area : ℝ := long_side a * short_side a

theorem area_comparison : (rectangle_area a / square_area a) = 0.999 := by
  sorry

-- Perimeter calculations and comparison
def square_perimeter : ℝ := 4 * a
def rectangle_perimeter : ℝ := 2 * (long_side a + short_side a)

theorem perimeter_comparison : (rectangle_perimeter a / square_perimeter a) = 1.005 := by
  sorry

end NUMINAMATH_GPT_area_comparison_perimeter_comparison_l2002_200216


namespace NUMINAMATH_GPT_train_B_time_to_destination_l2002_200212

theorem train_B_time_to_destination (speed_A : ℕ) (time_A : ℕ) (speed_B : ℕ) (dA : ℕ) :
  speed_A = 100 ∧ time_A = 9 ∧ speed_B = 150 ∧ dA = speed_A * time_A →
  dA / speed_B = 6 := 
by
  sorry

end NUMINAMATH_GPT_train_B_time_to_destination_l2002_200212


namespace NUMINAMATH_GPT_find_x_sq_add_y_sq_l2002_200217

theorem find_x_sq_add_y_sq (x y : ℝ) (h1 : (x + y) ^ 2 = 36) (h2 : x * y = 10) : x ^ 2 + y ^ 2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_x_sq_add_y_sq_l2002_200217


namespace NUMINAMATH_GPT_cone_lateral_surface_area_ratio_l2002_200203

/-- Let a be the side length of the equilateral triangle front view of a cone.
    The base area of the cone is (π * (a / 2)^2).
    The lateral surface area of the cone is (π * (a / 2) * a).
    We want to show that the ratio of the lateral surface area to the base area is 2.
 -/
theorem cone_lateral_surface_area_ratio 
  (a : ℝ) 
  (base_area : ℝ := π * (a / 2)^2) 
  (lateral_surface_area : ℝ := π * (a / 2) * a) 
  : lateral_surface_area / base_area = 2 :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_ratio_l2002_200203


namespace NUMINAMATH_GPT_correct_operation_l2002_200266

variable (a : ℕ)

theorem correct_operation :
  (3 * a + 2 * a ≠ 5 * a^2) ∧
  (3 * a - 2 * a ≠ 1) ∧
  a^2 * a^3 = a^5 ∧
  (a / a^2 ≠ a) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2002_200266


namespace NUMINAMATH_GPT_train_cross_first_platform_in_15_seconds_l2002_200232

noncomputable def length_of_train : ℝ := 100
noncomputable def length_of_second_platform : ℝ := 500
noncomputable def time_to_cross_second_platform : ℝ := 20
noncomputable def length_of_first_platform : ℝ := 350
noncomputable def speed_of_train := (length_of_train + length_of_second_platform) / time_to_cross_second_platform
noncomputable def time_to_cross_first_platform := (length_of_train + length_of_first_platform) / speed_of_train

theorem train_cross_first_platform_in_15_seconds : time_to_cross_first_platform = 15 := by
  sorry

end NUMINAMATH_GPT_train_cross_first_platform_in_15_seconds_l2002_200232


namespace NUMINAMATH_GPT_hypotenuse_intersection_incircle_diameter_l2002_200286

/-- Let \( a \) and \( b \) be the legs of a right triangle with hypotenuse \( c \). 
    Let two circles be centered at the endpoints of the hypotenuse, with radii \( a \) and \( b \). 
    Prove that the segment of the hypotenuse that lies in the intersection of the two circles is equal in length to the diameter of the incircle of the triangle. -/
theorem hypotenuse_intersection_incircle_diameter (a b : ℝ) :
    let c := Real.sqrt (a^2 + b^2)
    let x := a + b - c
    let r := (a + b - c) / 2
    x = 2 * r :=
by
  let c := Real.sqrt (a^2 + b^2)
  let x := a + b - c
  let r := (a + b - c) / 2
  show x = 2 * r
  sorry

end NUMINAMATH_GPT_hypotenuse_intersection_incircle_diameter_l2002_200286


namespace NUMINAMATH_GPT_school_boys_count_l2002_200296

theorem school_boys_count (B G : ℕ) (h1 : B + G = 1150) (h2 : G = (B / 1150) * 100) : B = 1058 := 
by 
  sorry

end NUMINAMATH_GPT_school_boys_count_l2002_200296


namespace NUMINAMATH_GPT_distribute_7_balls_into_4_boxes_l2002_200228

-- Define the problem conditions
def number_of_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  if balls < boxes then 0 else Nat.choose (balls - 1) (boxes - 1)

-- Prove the specific case
theorem distribute_7_balls_into_4_boxes : number_of_ways_to_distribute_balls 7 4 = 20 :=
by
  -- Definition and proof to be filled
  sorry

end NUMINAMATH_GPT_distribute_7_balls_into_4_boxes_l2002_200228


namespace NUMINAMATH_GPT_aaron_earnings_l2002_200267

def time_worked_monday := 75 -- in minutes
def time_worked_tuesday := 50 -- in minutes
def time_worked_wednesday := 145 -- in minutes
def time_worked_friday := 30 -- in minutes
def hourly_rate := 3 -- dollars per hour

def total_minutes_worked := 
  time_worked_monday + time_worked_tuesday + 
  time_worked_wednesday + time_worked_friday

def total_hours_worked := total_minutes_worked / 60

def total_earnings := total_hours_worked * hourly_rate

theorem aaron_earnings :
  total_earnings = 15 := by
  sorry

end NUMINAMATH_GPT_aaron_earnings_l2002_200267


namespace NUMINAMATH_GPT_sweaters_to_wash_l2002_200221

theorem sweaters_to_wash (pieces_per_load : ℕ) (total_loads : ℕ) (shirts_to_wash : ℕ) 
  (h1 : pieces_per_load = 5) (h2 : total_loads = 9) (h3 : shirts_to_wash = 43) : ℕ :=
  if total_loads * pieces_per_load - shirts_to_wash = 2 then 2 else 0

end NUMINAMATH_GPT_sweaters_to_wash_l2002_200221


namespace NUMINAMATH_GPT_ratio_is_7_to_10_l2002_200294

-- Given conditions in the problem translated to Lean definitions
def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 10 * leopards
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := 670
def other_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + alligators
def cheetahs : ℕ := total_animals - other_animals

-- The ratio of cheetahs to snakes to be proven
def ratio_cheetahs_to_snakes (cheetahs snakes : ℕ) : ℚ := cheetahs / snakes

theorem ratio_is_7_to_10 : ratio_cheetahs_to_snakes cheetahs snakes = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_ratio_is_7_to_10_l2002_200294


namespace NUMINAMATH_GPT_range_of_m_l2002_200200

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x < y → y < 0 → f y < f x) (h_cond : ∀ m : ℝ, f (1 - m) < f (m - 3)) : ∀ m, 1 < m ∧ m < 2 :=
by
  intros m
  sorry

end NUMINAMATH_GPT_range_of_m_l2002_200200


namespace NUMINAMATH_GPT_no_monochromatic_10_term_progression_l2002_200256

def can_color_without_monochromatic_progression (n k : ℕ) (c : Fin n → Fin k) : Prop :=
  ∀ (a d : ℕ), (a < n) → (a + (9 * d) < n) → (∀ i : ℕ, i < 10 → c ⟨a + (i * d), sorry⟩ = c ⟨a, sorry⟩) → 
    (∃ j i : ℕ, j < 10 ∧ i < 10 ∧ c ⟨a + (i * d), sorry⟩ ≠ c ⟨a + (j * d), sorry⟩)

theorem no_monochromatic_10_term_progression :
  ∃ c : Fin 2008 → Fin 4, can_color_without_monochromatic_progression 2008 4 c :=
sorry

end NUMINAMATH_GPT_no_monochromatic_10_term_progression_l2002_200256
