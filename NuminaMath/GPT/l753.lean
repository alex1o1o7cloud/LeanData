import Mathlib

namespace NUMINAMATH_GPT_range_of_x_for_obtuse_angle_l753_75329

def vectors_are_obtuse (a b : ℝ × ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product < 0

theorem range_of_x_for_obtuse_angle :
  ∀ (x : ℝ), vectors_are_obtuse (1, 3) (x, -1) ↔ (x < -1/3 ∨ (-1/3 < x ∧ x < 3)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_for_obtuse_angle_l753_75329


namespace NUMINAMATH_GPT_sum_of_first_17_terms_arithmetic_sequence_l753_75341

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem sum_of_first_17_terms_arithmetic_sequence
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 3 + a 9 + a 15 = 9) :
  sum_of_first_n_terms a 17 = 51 :=
sorry

end NUMINAMATH_GPT_sum_of_first_17_terms_arithmetic_sequence_l753_75341


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l753_75317

theorem isosceles_triangle_base_angle (a b c : ℝ) (h : a + b + c = 180) (h_isosceles : b = c) (h_angle_a : a = 120) : b = 30 := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l753_75317


namespace NUMINAMATH_GPT_part_I_part_II_l753_75376

open Real

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 4)

theorem part_I (x : ℝ) : f x > 0 ↔ (x > 1 ∨ x < -5) := 
sorry

theorem part_II (m : ℝ) : (∀ x : ℝ, f x + 3 * abs (x - 4) > m) ↔ (m < 9) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l753_75376


namespace NUMINAMATH_GPT_middle_digit_base_5_reversed_in_base_8_l753_75387

theorem middle_digit_base_5_reversed_in_base_8 (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 4) (h₂ : 0 ≤ b ∧ b ≤ 4) 
  (h₃ : 0 ≤ c ∧ c ≤ 4) (h₄ : 25 * a + 5 * b + c = 64 * c + 8 * b + a) : b = 3 := 
by 
  sorry

end NUMINAMATH_GPT_middle_digit_base_5_reversed_in_base_8_l753_75387


namespace NUMINAMATH_GPT_find_eccentricity_l753_75350

variables {a b x_N x_M : ℝ}
variable {e : ℝ}

-- Conditions
def line_passes_through_N (x_N : ℝ) (x_M : ℝ) : Prop :=
x_N ≠ 0 ∧ x_N = 4 * x_M

def hyperbola (x y a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def midpoint_x_M (x_M : ℝ) : Prop :=
∃ (x1 x2 y1 y2 : ℝ), (x1 + x2) / 2 = x_M

-- Proof Problem
theorem find_eccentricity
  (hN : line_passes_through_N x_N x_M)
  (hC : hyperbola x_N 0 a b)
  (hM : midpoint_x_M x_M) :
  e = 2 :=
sorry

end NUMINAMATH_GPT_find_eccentricity_l753_75350


namespace NUMINAMATH_GPT_tv_selection_l753_75358

theorem tv_selection (A B : ℕ) (hA : A = 4) (hB : B = 5) : 
  ∃ n, n = 3 ∧ (∃ k, k = 70 ∧ 
    (n = 1 ∧ k = A * (B * (B - 1) / 2) + A * (A - 1) / 2 * B)) :=
sorry

end NUMINAMATH_GPT_tv_selection_l753_75358


namespace NUMINAMATH_GPT_find_smaller_number_l753_75394

theorem find_smaller_number (a b : ℤ) (h₁ : a + b = 8) (h₂ : a - b = 4) : b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l753_75394


namespace NUMINAMATH_GPT_chris_dana_shared_rest_days_l753_75366

/-- Chris's and Dana's working schedules -/
structure work_schedule where
  work_days : ℕ
  rest_days : ℕ

/-- Define Chris's and Dana's schedules -/
def Chris_schedule : work_schedule := { work_days := 5, rest_days := 2 }
def Dana_schedule : work_schedule := { work_days := 6, rest_days := 1 }

/-- Number of days to consider -/
def total_days : ℕ := 1200

/-- Combinatorial function to calculate the number of coinciding rest-days -/
noncomputable def coinciding_rest_days (schedule1 schedule2 : work_schedule) (days : ℕ) : ℕ :=
  (days / (Nat.lcm (schedule1.work_days + schedule1.rest_days) (schedule2.work_days + schedule2.rest_days)))

/-- The proof problem statement -/
theorem chris_dana_shared_rest_days : 
coinciding_rest_days Chris_schedule Dana_schedule total_days = 171 :=
by sorry

end NUMINAMATH_GPT_chris_dana_shared_rest_days_l753_75366


namespace NUMINAMATH_GPT_probability_of_rolling_four_threes_l753_75388
open BigOperators

def probability_four_threes (n : ℕ) (k : ℕ) (p : ℚ) (q : ℚ) : ℚ := 
  (n.choose k) * (p ^ k) * (q ^ (n - k))

theorem probability_of_rolling_four_threes : 
  probability_four_threes 5 4 (1 / 10) (9 / 10) = 9 / 20000 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_rolling_four_threes_l753_75388


namespace NUMINAMATH_GPT_find_speeds_l753_75364

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end NUMINAMATH_GPT_find_speeds_l753_75364


namespace NUMINAMATH_GPT_employee_saves_l753_75316

-- Given conditions
def cost_price : ℝ := 500
def markup_percentage : ℝ := 0.15
def employee_discount_percentage : ℝ := 0.15

-- Definitions
def final_retail_price : ℝ := cost_price * (1 + markup_percentage)
def employee_discount_amount : ℝ := final_retail_price * employee_discount_percentage

-- Assertion
theorem employee_saves :
  employee_discount_amount = 86.25 := by
  sorry

end NUMINAMATH_GPT_employee_saves_l753_75316


namespace NUMINAMATH_GPT_minimum_inlets_needed_l753_75396

noncomputable def waterInflow (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := x * a - b

theorem minimum_inlets_needed (a b : ℝ) (ha : a = b)
  (h1 : (4 * a - b) * 5 = (2 * a - b) * 15)
  (h2 : (a * 9 - b) * 2 ≥ 1) : 
  ∃ n : ℕ, 2 * (a * n - b) ≥ (4 * a - b) * 5 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_inlets_needed_l753_75396


namespace NUMINAMATH_GPT_min_output_to_avoid_losses_l753_75332

theorem min_output_to_avoid_losses (x : ℝ) (y : ℝ) (h : y = 0.1 * x - 150) : y ≥ 0 → x ≥ 1500 :=
sorry

end NUMINAMATH_GPT_min_output_to_avoid_losses_l753_75332


namespace NUMINAMATH_GPT_solution_set_l753_75302

theorem solution_set (x : ℝ) : 
  (x * (x + 2) > 0 ∧ |x| < 1) ↔ (0 < x ∧ x < 1) := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_l753_75302


namespace NUMINAMATH_GPT_continuous_function_fixed_point_l753_75363

variable (f : ℝ → ℝ)
variable (h_cont : Continuous f)
variable (h_comp : ∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ (f^[n] x = 1))

theorem continuous_function_fixed_point : f 1 = 1 := 
by
  sorry

end NUMINAMATH_GPT_continuous_function_fixed_point_l753_75363


namespace NUMINAMATH_GPT_geometric_progression_condition_l753_75389

variables (a b c : ℝ) (k n p : ℕ)

theorem geometric_progression_condition :
  (a / b) ^ (k - p) = (a / c) ^ (k - n) :=
sorry

end NUMINAMATH_GPT_geometric_progression_condition_l753_75389


namespace NUMINAMATH_GPT_average_headcount_is_11033_l753_75326

def average_headcount (count1 count2 count3 : ℕ) : ℕ :=
  (count1 + count2 + count3) / 3

theorem average_headcount_is_11033 :
  average_headcount 10900 11500 10700 = 11033 :=
by
  sorry

end NUMINAMATH_GPT_average_headcount_is_11033_l753_75326


namespace NUMINAMATH_GPT_algebraic_expression_value_l753_75368

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -4) :
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 23 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l753_75368


namespace NUMINAMATH_GPT_last_digit_2008_pow_2005_l753_75307

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_2008_pow_2005 : last_digit (2008 ^ 2005) = 8 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_2008_pow_2005_l753_75307


namespace NUMINAMATH_GPT_side_length_of_cloth_l753_75303

namespace ClothProblem

def original_side_length (trimming_x_sides trimming_y_sides remaining_area : ℤ) :=
  let x : ℤ := 12
  x

theorem side_length_of_cloth (x_trim y_trim remaining_area : ℤ) (h_trim_x : x_trim = 4) 
                             (h_trim_y : y_trim = 3) (h_area : remaining_area = 120) :
  original_side_length x_trim y_trim remaining_area = 12 :=
by
  sorry

end ClothProblem

end NUMINAMATH_GPT_side_length_of_cloth_l753_75303


namespace NUMINAMATH_GPT_meet_at_centroid_l753_75378

-- Definitions of positions
def Harry : ℝ × ℝ := (10, -3)
def Sandy : ℝ × ℝ := (2, 7)
def Ron : ℝ × ℝ := (6, 1)

-- Mathematical proof problem statement
theorem meet_at_centroid : 
    (Harry.1 + Sandy.1 + Ron.1) / 3 = 6 ∧ (Harry.2 + Sandy.2 + Ron.2) / 3 = 5 / 3 := 
by
  sorry

end NUMINAMATH_GPT_meet_at_centroid_l753_75378


namespace NUMINAMATH_GPT_employee_payments_l753_75367

noncomputable def amount_paid_to_Y : ℝ := 934 / 3
noncomputable def amount_paid_to_X : ℝ := 1.20 * amount_paid_to_Y
noncomputable def amount_paid_to_Z : ℝ := 0.80 * amount_paid_to_Y

theorem employee_payments :
  amount_paid_to_X + amount_paid_to_Y + amount_paid_to_Z = 934 :=
by
  sorry

end NUMINAMATH_GPT_employee_payments_l753_75367


namespace NUMINAMATH_GPT_cost_of_shirt_l753_75349

theorem cost_of_shirt (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 71) : S = 15 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_shirt_l753_75349


namespace NUMINAMATH_GPT_number_of_parallelograms_l753_75300

-- Given conditions
def num_horizontal_lines : ℕ := 4
def num_vertical_lines : ℕ := 4

-- Mathematical function for combinations
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Proof statement
theorem number_of_parallelograms :
  binom num_horizontal_lines 2 * binom num_vertical_lines 2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_parallelograms_l753_75300


namespace NUMINAMATH_GPT_euler_polyhedron_problem_l753_75340

theorem euler_polyhedron_problem : 
  ( ∀ (V E F T S : ℕ), F = 42 → (T = 2 ∧ S = 3) → V - E + F = 2 → 100 * S + 10 * T + V = 337 ) := 
by sorry

end NUMINAMATH_GPT_euler_polyhedron_problem_l753_75340


namespace NUMINAMATH_GPT_women_more_than_men_l753_75380

def men (W : ℕ) : ℕ := (5 * W) / 11

theorem women_more_than_men (M W : ℕ) (h1 : M + W = 16) (h2 : M = (5 * W) / 11) : W - M = 6 :=
by
  sorry

end NUMINAMATH_GPT_women_more_than_men_l753_75380


namespace NUMINAMATH_GPT_solve_inequality_l753_75355

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - (a + 2) * x + 2 < 0

-- Prove the solution sets for different values of a
theorem solve_inequality :
  ∀ (a : ℝ),
    (a = -1 → {x : ℝ | inequality a x} = {x | x < -2 ∨ x > 1}) ∧
    (a = 0 → {x : ℝ | inequality a x} = {x | x > 1}) ∧
    (a < 0 → {x : ℝ | inequality a x} = {x | x < 2 / a ∨ x > 1}) ∧
    (0 < a ∧ a < 2 → {x : ℝ | inequality a x} = {x | 1 < x ∧ x < 2 / a}) ∧
    (a = 2 → {x : ℝ | inequality a x} = ∅) ∧
    (a > 2 → {x : ℝ | inequality a x} = {x | 2 / a < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l753_75355


namespace NUMINAMATH_GPT_part1_part2_1_part2_2_l753_75371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x : ℝ, x > 0 → (2 * a * x - Real.log x - 1) ≥ 0) ↔ a ≥ 0.5 := 
sorry

theorem part2_1 (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 = x1 ∧ f a x2 = x2) :
  0 < a ∧ a < 1 := 
sorry

theorem part2_2 (a x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f a x1 = x1) (h3 : f a x2 = x2) (h4 : x2 ≥ 3 * x1) :
  x1 * x2 ≥ 9 / Real.exp 2 := 
sorry

end NUMINAMATH_GPT_part1_part2_1_part2_2_l753_75371


namespace NUMINAMATH_GPT_a1_geq_2_pow_k_l753_75345

-- Definitions of the problem conditions in Lean 4
def conditions (a : ℕ → ℕ) (n k : ℕ) : Prop :=
  (∀ i, 1 ≤ i ∧ i ≤ n → a i < 2 * n) ∧
  (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j → ¬(a i ∣ a j)) ∧
  (3^k < 2 * n ∧ 2 * n < 3^(k+1))

-- The main theorem to be proven
theorem a1_geq_2_pow_k (a : ℕ → ℕ) (n k : ℕ) (h : conditions a n k) : 
  a 1 ≥ 2^k :=
sorry

end NUMINAMATH_GPT_a1_geq_2_pow_k_l753_75345


namespace NUMINAMATH_GPT_find_wrongly_written_height_l753_75328

variable (n : ℕ := 35)
variable (average_height_incorrect : ℚ := 184)
variable (actual_height_one_boy : ℚ := 106)
variable (actual_average_height : ℚ := 182)
variable (x : ℚ)

theorem find_wrongly_written_height
  (h_incorrect_total : n * average_height_incorrect = 6440)
  (h_correct_total : n * actual_average_height = 6370) :
  6440 - x + actual_height_one_boy = 6370 ↔ x = 176 := by
  sorry

end NUMINAMATH_GPT_find_wrongly_written_height_l753_75328


namespace NUMINAMATH_GPT_proof_problem_l753_75327

variable {R : Type} [LinearOrderedField R]

def is_increasing (f : R → R) : Prop :=
  ∀ x y : R, x < y → f x < f y

theorem proof_problem (f : R → R) (a b : R) 
  (inc_f : is_increasing f) 
  (h : f a + f b > f (-a) + f (-b)) : 
  a + b > 0 := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l753_75327


namespace NUMINAMATH_GPT_perpendicular_condition_l753_75324

def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_condition (a : ℝ) :
  is_perpendicular (a^2) (1/a) ↔ a = -1 :=
sorry

end NUMINAMATH_GPT_perpendicular_condition_l753_75324


namespace NUMINAMATH_GPT_joe_spent_on_fruits_l753_75323

theorem joe_spent_on_fruits (total_money amount_left : ℝ) (spent_on_chocolates : ℝ)
  (h1 : total_money = 450)
  (h2 : spent_on_chocolates = (1/9) * total_money)
  (h3 : amount_left = 220)
  : (total_money - spent_on_chocolates - amount_left) / total_money = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_joe_spent_on_fruits_l753_75323


namespace NUMINAMATH_GPT_money_spent_correct_l753_75315

-- Define the number of plays, acts per play, wigs per act, and the cost of each wig
def num_plays := 3
def acts_per_play := 5
def wigs_per_act := 2
def wig_cost := 5
def sell_price := 4

-- Given the total number of wigs he drops and sells from one play
def dropped_plays := 1
def total_wigs_dropped := dropped_plays * acts_per_play * wigs_per_act
def money_from_selling_dropped_wigs := total_wigs_dropped * sell_price

-- Calculate the initial cost
def total_wigs := num_plays * acts_per_play * wigs_per_act
def initial_cost := total_wigs * wig_cost

-- The final spent money should be calculated by subtracting money made from selling the wigs of the dropped play
def final_spent_money := initial_cost - money_from_selling_dropped_wigs

-- Specify the expected amount of money John spent
def expected_final_spent_money := 110

theorem money_spent_correct :
  final_spent_money = expected_final_spent_money := by
  sorry

end NUMINAMATH_GPT_money_spent_correct_l753_75315


namespace NUMINAMATH_GPT_translation_vector_condition_l753_75339

theorem translation_vector_condition (m n : ℝ) :
  (∀ x : ℝ, 2 * (x - m) + n = 2 * x + 5) → n = 2 * m + 5 :=
by
  intro h
  -- proof can be filled here
  sorry

end NUMINAMATH_GPT_translation_vector_condition_l753_75339


namespace NUMINAMATH_GPT_monomial_degree_and_coefficient_l753_75334

theorem monomial_degree_and_coefficient (a b : ℤ) (h1 : -a = 7) (h2 : 1 + b = 4) : a + b = -4 :=
by
  sorry

end NUMINAMATH_GPT_monomial_degree_and_coefficient_l753_75334


namespace NUMINAMATH_GPT_find_x_l753_75304

def hash_op (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_x (x : ℕ) (h : hash_op x 6 = 48) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l753_75304


namespace NUMINAMATH_GPT_lucy_total_cost_for_lamp_and_table_l753_75393

noncomputable def original_price_lamp : ℝ := 200 / 1.2

noncomputable def table_price : ℝ := 2 * original_price_lamp

noncomputable def total_cost_paid (lamp_cost discounted_price table_price: ℝ) :=
  lamp_cost + table_price

theorem lucy_total_cost_for_lamp_and_table :
  total_cost_paid 20 (original_price_lamp * 0.6) table_price = 353.34 :=
by
  let lamp_original_price := original_price_lamp
  have h1 : original_price_lamp * (0.6 * (1 / 5)) = 20 := by sorry
  have h2 : table_price = 2 * original_price_lamp := by sorry
  have h3 : total_cost_paid 20 (original_price_lamp * 0.6) table_price = 20 + table_price := by sorry
  have h4 : table_price = 2 * (200 / 1.2) := by sorry
  have h5 : 20 + table_price = 353.34 := by sorry
  exact h5

end NUMINAMATH_GPT_lucy_total_cost_for_lamp_and_table_l753_75393


namespace NUMINAMATH_GPT_remaining_people_statement_l753_75352

-- Definitions of conditions
def number_of_people : Nat := 10
def number_of_knights (K : Nat) : Prop := K ≤ number_of_people
def number_of_liars (L : Nat) : Prop := L ≤ number_of_people
def statement (s : String) : Prop := s = "There are more liars" ∨ s = "There are equal numbers"

-- Main theorem
theorem remaining_people_statement (K L : Nat) (h_total : K + L = number_of_people) 
  (h_knights_behavior : ∀ k, k < K → statement "There are equal numbers") 
  (h_liars_behavior : ∀ l, l < L → statement "There are more liars") :
  K = 5 → L = 5 → ∀ i, i < number_of_people → (i < 5 → statement "There are more liars") ∧ (i >= 5 → statement "There are equal numbers") := 
by
  sorry

end NUMINAMATH_GPT_remaining_people_statement_l753_75352


namespace NUMINAMATH_GPT_complement_intersect_l753_75370

def U : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}
def A : Set ℤ := {x | x^2 - 1 ≤ 0}
def B : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def C : Set ℤ := {x | x ∉ A ∧ x ∈ U} -- complement of A in U

theorem complement_intersect (U A B : Set ℤ) :
  (C ∩ B) = {2, 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersect_l753_75370


namespace NUMINAMATH_GPT_right_to_left_evaluation_l753_75337

variable (a b c d : ℝ)

theorem right_to_left_evaluation :
  a / b - c + d = a / (b - c - d) :=
sorry

end NUMINAMATH_GPT_right_to_left_evaluation_l753_75337


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l753_75343

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := -(k + 3)
  let c := 2 * k + 1
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l753_75343


namespace NUMINAMATH_GPT_sum_base_49_l753_75360

-- Definitions of base b numbers and their base 10 conversion
def num_14_in_base (b : ℕ) : ℕ := b + 4
def num_17_in_base (b : ℕ) : ℕ := b + 7
def num_18_in_base (b : ℕ) : ℕ := b + 8
def num_6274_in_base (b : ℕ) : ℕ := 6 * b^3 + 2 * b^2 + 7 * b + 4

-- The question: Compute 14 + 17 + 18 in base b
def sum_in_base (b : ℕ) : ℕ := 14 + 17 + 18

-- The main statement to prove
theorem sum_base_49 (b : ℕ) (h : (num_14_in_base b) * (num_17_in_base b) * (num_18_in_base b) = num_6274_in_base (b)) :
  sum_in_base b = 49 :=
by sorry

end NUMINAMATH_GPT_sum_base_49_l753_75360


namespace NUMINAMATH_GPT_waiting_time_probability_l753_75310

-- Given conditions
def dep1 := 7 * 60 -- 7:00 in minutes
def dep2 := 7 * 60 + 30 -- 7:30 in minutes
def dep3 := 8 * 60 -- 8:00 in minutes

def arrival_start := 7 * 60 + 25 -- 7:25 in minutes
def arrival_end := 8 * 60 -- 8:00 in minutes
def total_time_window := arrival_end - arrival_start -- 35 minutes

def favorable_window1_start := 7 * 60 + 25 -- 7:25 in minutes
def favorable_window1_end := 7 * 60 + 30 -- 7:30 in minutes
def favorable_window2_start := 8 * 60 -- 8:00 in minutes
def favorable_window2_end := 8 * 60 + 10 -- 8:10 in minutes

def favorable_time_window := 
  (favorable_window1_end - favorable_window1_start) + 
  (favorable_window2_end - favorable_window2_start) -- 15 minutes

-- Probability calculation
theorem waiting_time_probability : 
  (favorable_time_window : ℚ) / (total_time_window : ℚ) = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_waiting_time_probability_l753_75310


namespace NUMINAMATH_GPT_cos_alpha_plus_pi_over_4_l753_75362

theorem cos_alpha_plus_pi_over_4
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = 3 / 5)
  (h4 : Real.sin (β - π / 4) = 5 / 13) : 
  Real.cos (α + π / 4) = 56 / 65 :=
by
  sorry 

end NUMINAMATH_GPT_cos_alpha_plus_pi_over_4_l753_75362


namespace NUMINAMATH_GPT_xiaomin_house_position_l753_75383

-- Define the initial position of the school at the origin
def school_pos : ℝ × ℝ := (0, 0)

-- Define the movement east and south from the school's position
def xiaomin_house_pos (east_distance south_distance : ℝ) : ℝ × ℝ :=
  (school_pos.1 + east_distance, school_pos.2 - south_distance)

-- The given conditions
def east_distance := 200
def south_distance := 150

-- The theorem stating Xiaomin's house position
theorem xiaomin_house_position :
  xiaomin_house_pos east_distance south_distance = (200, -150) :=
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_xiaomin_house_position_l753_75383


namespace NUMINAMATH_GPT_correct_factorization_l753_75321

-- Define the polynomial expressions
def polyA (x : ℝ) := x^3 - x
def factorA1 (x : ℝ) := x * (x^2 - 1)
def factorA2 (x : ℝ) := x * (x + 1) * (x - 1)

def polyB (a : ℝ) := 4 * a^2 - 4 * a + 1
def factorB (a : ℝ) := 4 * a * (a - 1) + 1

def polyC (x y : ℝ) := x^2 + y^2
def factorC (x y : ℝ) := (x + y)^2

def polyD (x : ℝ) := -3 * x + 6 * x^2 - 3 * x^3
def factorD (x : ℝ) := -3 * x * (x - 1)^2

-- Statement of the correctness of factorization D
theorem correct_factorization : ∀ (x : ℝ), polyD x = factorD x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_correct_factorization_l753_75321


namespace NUMINAMATH_GPT_parallel_line_slope_l753_75357

theorem parallel_line_slope (x y : ℝ) :
  ∃ m b : ℝ, (3 * x - 6 * y = 21) → ∀ (x₁ y₁ : ℝ), (3 * x₁ - 6 * y₁ = 21) → m = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_slope_l753_75357


namespace NUMINAMATH_GPT_linear_regression_neg_corr_l753_75399

-- Given variables x and y with certain properties
variables (x y : ℝ)

-- Conditions provided in the problem
def neg_corr (x y : ℝ) : Prop := ∀ a b : ℝ, (a < b → x * y < 0)
def sample_mean_x := (2 : ℝ)
def sample_mean_y := (1.5 : ℝ)

-- Statement to prove the linear regression equation
theorem linear_regression_neg_corr (h1 : neg_corr x y)
    (hx : sample_mean_x = 2)
    (hy : sample_mean_y = 1.5) : 
    ∃ b0 b1 : ℝ, b0 = 5.5 ∧ b1 = -2 ∧ y = b0 + b1 * x :=
sorry

end NUMINAMATH_GPT_linear_regression_neg_corr_l753_75399


namespace NUMINAMATH_GPT_randy_trip_length_l753_75373

theorem randy_trip_length (x : ℝ) (h : x / 2 + 30 + x / 4 = x) : x = 120 :=
by
  sorry

end NUMINAMATH_GPT_randy_trip_length_l753_75373


namespace NUMINAMATH_GPT_smallest_solution_fraction_eq_l753_75392

theorem smallest_solution_fraction_eq (x : ℝ) (h : x ≠ 3) :
    3 * x / (x - 3) + (3 * x^2 - 27) / x = 16 ↔ x = (2 - Real.sqrt 31) / 3 := 
sorry

end NUMINAMATH_GPT_smallest_solution_fraction_eq_l753_75392


namespace NUMINAMATH_GPT_polar_to_rectangular_coordinates_l753_75347

theorem polar_to_rectangular_coordinates :
  let r := 2
  let θ := Real.pi / 3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (1, Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_coordinates_l753_75347


namespace NUMINAMATH_GPT_Petya_wrong_example_l753_75381

def a := 8
def b := 128

theorem Petya_wrong_example : (a^7 ∣ b^3) ∧ ¬ (a^2 ∣ b) :=
by {
  -- Prove the divisibility conditions and the counterexample
  sorry
}

end NUMINAMATH_GPT_Petya_wrong_example_l753_75381


namespace NUMINAMATH_GPT_area_of_triangle_BCD_l753_75385

-- Define the points A, B, C, D
variables {A B C D : Type} 

-- Define the lengths of segments AC and CD
variables (AC CD : ℝ)
-- Define the area of triangle ABC
variables (area_ABC : ℝ)

-- Define height h
variables (h : ℝ)

-- Initial conditions
axiom length_AC : AC = 9
axiom length_CD : CD = 39
axiom area_ABC_is_36 : area_ABC = 36
axiom height_is_8 : h = (2 * area_ABC) / AC

-- Define the area of triangle BCD
def area_BCD (CD h : ℝ) : ℝ := 0.5 * CD * h

-- The theorem that we want to prove
theorem area_of_triangle_BCD : area_BCD 39 8 = 156 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_BCD_l753_75385


namespace NUMINAMATH_GPT_monotonically_increasing_condition_l753_75386

theorem monotonically_increasing_condition 
  (a b c d : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ 3 * a * x ^ 2 + 2 * b * x + c) ↔ (b^2 - 3 * a * c ≤ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_monotonically_increasing_condition_l753_75386


namespace NUMINAMATH_GPT_sequence_properties_l753_75320

-- Define the sequence according to the problem
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ (∀ n : ℕ, n ≥ 2 → a n = (n * a (n - 1)) / (n - 1))

-- State the theorem to be proved
theorem sequence_properties :
  ∃ (a : ℕ → ℕ), 
    seq a ∧ a 2 = 6 ∧ a 3 = 9 ∧ (∀ n : ℕ, n ≥ 1 → a n = 3 * n) :=
by
  -- Existence quantifier and properties (sequence definition, first three terms, and general term)
  sorry

end NUMINAMATH_GPT_sequence_properties_l753_75320


namespace NUMINAMATH_GPT_minimum_value_l753_75309

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : 
  (∃ x, (∀ y, y = (1 / a) + (4 / b) → y ≥ x) ∧ x = 9 / 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l753_75309


namespace NUMINAMATH_GPT_randy_total_trees_l753_75348

theorem randy_total_trees (mango_trees : ℕ) (coconut_trees : ℕ) 
  (h1 : mango_trees = 60) 
  (h2 : coconut_trees = (mango_trees / 2) - 5) : 
  mango_trees + coconut_trees = 85 :=
by
  sorry

end NUMINAMATH_GPT_randy_total_trees_l753_75348


namespace NUMINAMATH_GPT_fish_upstream_speed_l753_75375

def Vs : ℝ := 45
def Vdownstream : ℝ := 55

def Vupstream (Vs Vw : ℝ) : ℝ := Vs - Vw
def Vstream (Vs Vdownstream : ℝ) : ℝ := Vdownstream - Vs

theorem fish_upstream_speed :
  Vupstream Vs (Vstream Vs Vdownstream) = 35 := by
  sorry

end NUMINAMATH_GPT_fish_upstream_speed_l753_75375


namespace NUMINAMATH_GPT_cost_of_camel_l753_75374

variables (C H O E G Z L : ℕ)

theorem cost_of_camel :
  (10 * C = 24 * H) →
  (16 * H = 4 * O) →
  (6 * O = 4 * E) →
  (3 * E = 5 * G) →
  (8 * G = 12 * Z) →
  (20 * Z = 7 * L) →
  (10 * E = 120000) →
  C = 4800 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_camel_l753_75374


namespace NUMINAMATH_GPT_select_rows_and_columns_l753_75312

theorem select_rows_and_columns (n : Nat) (pieces : Fin (2 * n) × Fin (2 * n) → Bool) :
  (∃ rows cols : Finset (Fin (2 * n)),
    rows.card = n ∧ cols.card = n ∧
    (∀ r c, r ∈ rows → c ∈ cols → pieces (r, c))) :=
sorry

end NUMINAMATH_GPT_select_rows_and_columns_l753_75312


namespace NUMINAMATH_GPT_no_simultaneous_squares_l753_75318

theorem no_simultaneous_squares (x y : ℕ) :
  ¬ (∃ a b : ℤ, x^2 + 2 * y = a^2 ∧ y^2 + 2 * x = b^2) :=
by
  sorry

end NUMINAMATH_GPT_no_simultaneous_squares_l753_75318


namespace NUMINAMATH_GPT_rowing_time_one_hour_l753_75301

noncomputable def total_time_to_travel (Vm Vr distance : ℝ) : ℝ :=
  let upstream_speed := Vm - Vr
  let downstream_speed := Vm + Vr
  let one_way_distance := distance / 2
  let time_upstream := one_way_distance / upstream_speed
  let time_downstream := one_way_distance / downstream_speed
  time_upstream + time_downstream

theorem rowing_time_one_hour : 
  total_time_to_travel 8 1.8 7.595 = 1 := 
sorry

end NUMINAMATH_GPT_rowing_time_one_hour_l753_75301


namespace NUMINAMATH_GPT_perpendicular_lines_k_value_l753_75384

theorem perpendicular_lines_k_value (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_k_value_l753_75384


namespace NUMINAMATH_GPT_susan_avg_speed_l753_75344

variable (d1 d2 : ℕ) (s1 s2 : ℕ)

def time (d s : ℕ) : ℚ := d / s

theorem susan_avg_speed 
  (h1 : d1 = 40) 
  (h2 : s1 = 30) 
  (h3 : d2 = 40) 
  (h4 : s2 = 15) : 
  (d1 + d2) / (time d1 s1 + time d2 s2) = 20 := 
by 
  -- Sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_susan_avg_speed_l753_75344


namespace NUMINAMATH_GPT_simplify_expression_l753_75361

theorem simplify_expression (x y : ℤ) (h1 : x = 1) (h2 : y = -2) :
  2 * x ^ 2 - (3 * (-5 / 3 * x ^ 2 + 2 / 3 * x * y) - (x * y - 3 * x ^ 2)) + 2 * x * y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l753_75361


namespace NUMINAMATH_GPT_not_in_sequence_l753_75398

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence property
def sequence_property (a b : ℕ) : Prop :=
  b = a + sum_of_digits a

-- Main theorem
theorem not_in_sequence (n : ℕ) (h : n = 793210041) : 
  ¬ (∃ a : ℕ, sequence_property a n) :=
by
  sorry

end NUMINAMATH_GPT_not_in_sequence_l753_75398


namespace NUMINAMATH_GPT_arrangements_with_gap_l753_75305

theorem arrangements_with_gap :
  ∃ (arrangements : ℕ), arrangements = 36 :=
by
  sorry

end NUMINAMATH_GPT_arrangements_with_gap_l753_75305


namespace NUMINAMATH_GPT_fraction_of_capital_subscribed_l753_75351

theorem fraction_of_capital_subscribed (T : ℝ) (x : ℝ) :
  let B_capital := (1 / 4) * T
  let C_capital := (1 / 5) * T
  let Total_profit := 2445
  let A_profit := 815
  A_profit / Total_profit = x → x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_capital_subscribed_l753_75351


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l753_75382

variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)

theorem negation_of_universal_proposition :
  (∀ x1 x2 : R, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : R, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l753_75382


namespace NUMINAMATH_GPT_divisibility_by_9_l753_75391

theorem divisibility_by_9 (x y z : ℕ) (h1 : 9 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) (h3 : 0 ≤ z ∧ z ≤ 9) :
  (100 * x + 10 * y + z) % 9 = 0 ↔ (x + y + z) % 9 = 0 := by
  sorry

end NUMINAMATH_GPT_divisibility_by_9_l753_75391


namespace NUMINAMATH_GPT_area_of_sector_l753_75335

-- Given conditions
def central_angle : ℝ := 2
def perimeter : ℝ := 8

-- Define variables and expressions
variable (r l : ℝ)

-- Equations based on the conditions
def eq1 := l + 2 * r = perimeter
def eq2 := l = central_angle * r

-- Assertion of the correct answer
theorem area_of_sector : ∃ r l : ℝ, eq1 r l ∧ eq2 r l ∧ (1 / 2 * l * r = 4) := by
  sorry

end NUMINAMATH_GPT_area_of_sector_l753_75335


namespace NUMINAMATH_GPT_sum_is_zero_l753_75377

variable (a b c x y : ℝ)

theorem sum_is_zero (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
(h4 : a^3 + a * x + y = 0)
(h5 : b^3 + b * x + y = 0)
(h6 : c^3 + c * x + y = 0) : a + b + c = 0 :=
sorry

end NUMINAMATH_GPT_sum_is_zero_l753_75377


namespace NUMINAMATH_GPT_tangent_line_at_x_5_l753_75397

noncomputable def f : ℝ → ℝ := sorry

theorem tangent_line_at_x_5 :
  (∀ x, f x = -x + 8 → f 5 + deriv f 5 = 2) := sorry

end NUMINAMATH_GPT_tangent_line_at_x_5_l753_75397


namespace NUMINAMATH_GPT_Malcom_has_more_cards_l753_75353

-- Define the number of cards Brandon has
def Brandon_cards : ℕ := 20

-- Define the number of cards Malcom has initially, to be found
def Malcom_initial_cards (n : ℕ) := n

-- Define the given condition: Malcom has 14 cards left after giving away half of his cards
def Malcom_half_condition (n : ℕ) := n / 2 = 14

-- Prove that Malcom had 8 more cards than Brandon initially
theorem Malcom_has_more_cards (n : ℕ) (h : Malcom_half_condition n) :
  Malcom_initial_cards n - Brandon_cards = 8 :=
by
  sorry

end NUMINAMATH_GPT_Malcom_has_more_cards_l753_75353


namespace NUMINAMATH_GPT_sum_of_youngest_and_oldest_l753_75325

-- Let a1, a2, a3, a4 be the ages of Janet's 4 children arranged in non-decreasing order.
-- Given conditions:
variable (a₁ a₂ a₃ a₄ : ℕ)
variable (h_mean : (a₁ + a₂ + a₃ + a₄) / 4 = 10)
variable (h_median : (a₂ + a₃) / 2 = 7)

-- Proof problem:
theorem sum_of_youngest_and_oldest :
  a₁ + a₄ = 26 :=
sorry

end NUMINAMATH_GPT_sum_of_youngest_and_oldest_l753_75325


namespace NUMINAMATH_GPT_x_cubed_plus_y_cubed_l753_75313

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^3 + y^3 = 85 / 2 :=
by
  sorry

end NUMINAMATH_GPT_x_cubed_plus_y_cubed_l753_75313


namespace NUMINAMATH_GPT_cos_B_value_l753_75359

-- Define the sides of the triangle
def AB : ℝ := 8
def AC : ℝ := 10
def right_angle_at_A : Prop := true

-- Define the cosine function within the context of the given triangle
noncomputable def cos_B : ℝ := AB / AC

-- The proof statement asserting the condition
theorem cos_B_value : cos_B = 4 / 5 :=
by
  -- Given conditions
  have h1 : AB = 8 := rfl
  have h2 : AC = 10 := rfl
  -- Direct computation
  sorry

end NUMINAMATH_GPT_cos_B_value_l753_75359


namespace NUMINAMATH_GPT_solve_complex_eq_l753_75365

theorem solve_complex_eq (z : ℂ) (h : z^2 = -100 - 64 * I) : z = 3.06 - 10.46 * I ∨ z = -3.06 + 10.46 * I :=
by
  sorry

end NUMINAMATH_GPT_solve_complex_eq_l753_75365


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l753_75330

open Real

theorem cyclic_sum_inequality
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / c + b^2 / a + c^2 / b) + (b^2 / c + c^2 / a + a^2 / b) + (c^2 / a + a^2 / b + b^2 / c) + 
  7 * (a + b + c) 
  ≥ ((a + b + c)^3) / (a * b + b * c + c * a) + (2 * (a * b + b * c + c * a)^2) / (a * b * c) := 
sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l753_75330


namespace NUMINAMATH_GPT_complement_U_A_l753_75314

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | |x - 1| > 1 }

theorem complement_U_A : (U \ A) = { x | 0 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_complement_U_A_l753_75314


namespace NUMINAMATH_GPT_combined_average_speed_l753_75379

-- Definitions based on conditions
def distance_A : ℕ := 250
def time_A : ℕ := 4

def distance_B : ℕ := 480
def time_B : ℕ := 6

def distance_C : ℕ := 390
def time_C : ℕ := 5

def total_distance : ℕ := distance_A + distance_B + distance_C
def total_time : ℕ := time_A + time_B + time_C

-- Prove combined average speed
theorem combined_average_speed : (total_distance : ℚ) / (total_time : ℚ) = 74.67 :=
  by
    sorry

end NUMINAMATH_GPT_combined_average_speed_l753_75379


namespace NUMINAMATH_GPT_work_problem_l753_75336

theorem work_problem (W : ℕ) (h1: ∀ w, w = W → (24 * w + 1 = 73)) : W = 3 :=
by {
  -- Insert proof here
  sorry
}

end NUMINAMATH_GPT_work_problem_l753_75336


namespace NUMINAMATH_GPT_carB_speed_l753_75331

variable (distance : ℝ) (time : ℝ) (ratio : ℝ) (speedB : ℝ)

theorem carB_speed (h1 : distance = 240) (h2 : time = 1.5) (h3 : ratio = 3 / 5) 
(h4 : (speedB + ratio * speedB) * time = distance) : speedB = 100 := 
by 
  sorry

end NUMINAMATH_GPT_carB_speed_l753_75331


namespace NUMINAMATH_GPT_decagon_area_l753_75308

theorem decagon_area (perimeter : ℝ) (n : ℕ) (side_length : ℝ)
  (segments : ℕ) (area : ℝ) :
  perimeter = 200 ∧ n = 4 ∧ side_length = perimeter / n ∧ segments = 5 ∧ 
  area = (side_length / segments)^2 * (1 - (1/2)) * 4 * segments  →
  area = 2300 := 
by
  sorry

end NUMINAMATH_GPT_decagon_area_l753_75308


namespace NUMINAMATH_GPT_functional_equation_solution_l753_75395

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (y * f (x + y) + f x) = 4 * x + 2 * y * f (x + y)) →
  (∀ x : ℝ, f x = 2 * x) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l753_75395


namespace NUMINAMATH_GPT_women_doubles_tournament_handshakes_l753_75342

theorem women_doubles_tournament_handshakes :
  ∀ (teams : List (List Prop)), List.length teams = 4 → (∀ t ∈ teams, List.length t = 2) →
  (∃ (handshakes : ℕ), handshakes = 24) :=
by
  intro teams h1 h2
  -- Assume teams are disjoint and participants shake hands meeting problem conditions
  -- The lean proof will follow the logical structure used for the mathematical solution
  -- We'll now formalize the conditions and the handshake calculation
  sorry

end NUMINAMATH_GPT_women_doubles_tournament_handshakes_l753_75342


namespace NUMINAMATH_GPT_sqrt_expression_eq_two_l753_75333

theorem sqrt_expression_eq_two : 
  (Real.sqrt 3) * (Real.sqrt 3 - 1 / (Real.sqrt 3)) = 2 := 
  sorry

end NUMINAMATH_GPT_sqrt_expression_eq_two_l753_75333


namespace NUMINAMATH_GPT_find_number_l753_75319

theorem find_number (N : ℝ)
  (h1 : 5 / 6 * N = 5 / 16 * N + 250) :
  N = 480 :=
sorry

end NUMINAMATH_GPT_find_number_l753_75319


namespace NUMINAMATH_GPT_max_pages_copied_l753_75354

-- Definitions based on conditions
def cents_per_page := 7 / 4
def budget_cents := 1500

-- The theorem to prove
theorem max_pages_copied (c : ℝ) (budget : ℝ) (h₁ : c = cents_per_page) (h₂ : budget = budget_cents) : 
  ⌊(budget / c)⌋ = 857 :=
sorry

end NUMINAMATH_GPT_max_pages_copied_l753_75354


namespace NUMINAMATH_GPT_pine_sample_count_l753_75356

variable (total_saplings : ℕ)
variable (pine_saplings : ℕ)
variable (sample_size : ℕ)

theorem pine_sample_count (h1 : total_saplings = 30000) (h2 : pine_saplings = 4000) (h3 : sample_size = 150) :
  pine_saplings * sample_size / total_saplings = 20 := 
sorry

end NUMINAMATH_GPT_pine_sample_count_l753_75356


namespace NUMINAMATH_GPT_intersection_x_value_l753_75390

/-- Prove that the x-value at the point of intersection of the lines
    y = 5x - 28 and 3x + y = 120 is 18.5 -/
theorem intersection_x_value :
  ∃ x y : ℝ, (y = 5 * x - 28) ∧ (3 * x + y = 120) ∧ (x = 18.5) :=
by
  sorry

end NUMINAMATH_GPT_intersection_x_value_l753_75390


namespace NUMINAMATH_GPT_purely_imaginary_a_l753_75306

theorem purely_imaginary_a (a : ℝ) (h : (a^3 - a) = 0) (h2 : (a / (1 - a)) ≠ 0) : a = -1 := 
sorry

end NUMINAMATH_GPT_purely_imaginary_a_l753_75306


namespace NUMINAMATH_GPT_Q_investment_time_l753_75338

theorem Q_investment_time  
  (P Q x t : ℝ)
  (h_ratio_investments : P = 7 * x ∧ Q = 5 * x)
  (h_ratio_profits : (7 * x * 10) / (5 * x * t) = 7 / 10) :
  t = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_Q_investment_time_l753_75338


namespace NUMINAMATH_GPT_find_a_from_limit_l753_75311

theorem find_a_from_limit (a : ℝ) (h : (Filter.Tendsto (fun n : ℕ => (a * n - 2) / (n + 1)) Filter.atTop (Filter.principal {1}))) :
    a = 1 := 
sorry

end NUMINAMATH_GPT_find_a_from_limit_l753_75311


namespace NUMINAMATH_GPT_total_selling_price_is_18000_l753_75346

def cost_price_per_meter : ℕ := 50
def loss_per_meter : ℕ := 5
def meters_sold : ℕ := 400

def selling_price_per_meter := cost_price_per_meter - loss_per_meter

def total_selling_price := selling_price_per_meter * meters_sold

theorem total_selling_price_is_18000 :
  total_selling_price = 18000 :=
sorry

end NUMINAMATH_GPT_total_selling_price_is_18000_l753_75346


namespace NUMINAMATH_GPT_value_of_f_ln6_l753_75372

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x + Real.exp x else -(x + Real.exp (-x))

theorem value_of_f_ln6 : (f (Real.log 6)) = Real.log 6 - (1/6) :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_ln6_l753_75372


namespace NUMINAMATH_GPT_prob_both_even_correct_l753_75369

-- Define the dice and verify their properties
def die1 := {n : ℕ // n ≥ 1 ∧ n ≤ 6}
def die2 := {n : ℕ // n ≥ 1 ∧ n ≤ 7}

-- Define the sets of even numbers for both dice
def even_die1 (n : die1) : Prop := n.1 % 2 = 0
def even_die2 (n : die2) : Prop := n.1 % 2 = 0

-- Define the probabilities of rolling an even number on each die
def prob_even_die1 := 3 / 6
def prob_even_die2 := 3 / 7

-- Calculate the combined probability
def prob_both_even := prob_even_die1 * prob_even_die2

-- The theorem stating the probability of both dice rolling even is 3/14
theorem prob_both_even_correct : prob_both_even = 3 / 14 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_prob_both_even_correct_l753_75369


namespace NUMINAMATH_GPT_remainder_of_n_l753_75322

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_n_l753_75322
