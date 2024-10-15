import Mathlib

namespace NUMINAMATH_GPT_equally_spaced_markings_number_line_l1251_125195

theorem equally_spaced_markings_number_line 
  (steps : ℕ) (distance : ℝ) (z_steps : ℕ) (z : ℝ)
  (h1 : steps = 4)
  (h2 : distance = 16)
  (h3 : z_steps = 2) :
  z = (distance / steps) * z_steps :=
by
  sorry

end NUMINAMATH_GPT_equally_spaced_markings_number_line_l1251_125195


namespace NUMINAMATH_GPT_robin_gum_pieces_l1251_125110

-- Defining the conditions
def packages : ℕ := 9
def pieces_per_package : ℕ := 15
def total_pieces : ℕ := 135

-- Theorem statement
theorem robin_gum_pieces (h1 : packages = 9) (h2 : pieces_per_package = 15) : packages * pieces_per_package = total_pieces := by
  -- According to the problem, the correct answer is 135 pieces
  have h: 9 * 15 = 135 := by norm_num
  rw [h1, h2]
  exact h

end NUMINAMATH_GPT_robin_gum_pieces_l1251_125110


namespace NUMINAMATH_GPT_new_car_distance_in_same_time_l1251_125167

-- Define the given conditions and the distances
variable (older_car_distance : ℝ := 150)
variable (new_car_speed_factor : ℝ := 1.30)  -- Since the new car is 30% faster, its speed factor is 1.30
variable (time : ℝ)

-- Define the older car's distance as a function of time and speed
def older_car_distance_covered (t : ℝ) (distance : ℝ) : ℝ := distance

-- Define the new car's distance as a function of time and speed factor
def new_car_distance_covered (t : ℝ) (distance : ℝ) (speed_factor : ℝ) : ℝ := speed_factor * distance

theorem new_car_distance_in_same_time
  (older_car_distance : ℝ)
  (new_car_speed_factor : ℝ)
  (time : ℝ)
  (h1 : older_car_distance = 150)
  (h2 : new_car_speed_factor = 1.30) :
  new_car_distance_covered time older_car_distance new_car_speed_factor = 195 := by
  sorry

end NUMINAMATH_GPT_new_car_distance_in_same_time_l1251_125167


namespace NUMINAMATH_GPT_monotonicity_and_extremes_l1251_125186

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonicity_and_extremes :
  (∀ x, f x > f (-3) ∨ f x < f (-3)) ∧
  (∀ x, f x > f 1 ∨ f x < f 1) ∧
  (∀ x, (x < -3 → (∀ y, y < x → f y < f x)) ∧ (x > 1 → (∀ y, y > x → f y < f x))) ∧
  f (-3) = 10 ∧ f 1 = -(2 / 3) :=
sorry

end NUMINAMATH_GPT_monotonicity_and_extremes_l1251_125186


namespace NUMINAMATH_GPT_sum_prime_numbers_l1251_125104

theorem sum_prime_numbers (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c) (hEqn : a * b * c + a = 851) : 
  a + b + c = 50 :=
sorry

end NUMINAMATH_GPT_sum_prime_numbers_l1251_125104


namespace NUMINAMATH_GPT_total_distance_l1251_125124

theorem total_distance (D : ℝ) (h_walk : ∀ d t, d = 4 * t) 
                       (h_run : ∀ d t, d = 8 * t) 
                       (h_time : ∀ t_walk t_run, t_walk + t_run = 0.75) 
                       (h_half : D / 2 = d_walk ∧ D / 2 = d_run) :
                       D = 8 := 
by
  sorry

end NUMINAMATH_GPT_total_distance_l1251_125124


namespace NUMINAMATH_GPT_find_value_l1251_125111

variable (x y : ℝ)

def conditions (x y : ℝ) :=
  y > 2 * x ∧ 2 * x > 0 ∧ (x / y + y / x = 8)

theorem find_value (h : conditions x y) : (x + y) / (x - y) = -Real.sqrt (5 / 3) :=
sorry

end NUMINAMATH_GPT_find_value_l1251_125111


namespace NUMINAMATH_GPT_bracelets_count_l1251_125114

-- Define the conditions
def stones_total : Nat := 36
def stones_per_bracelet : Nat := 12

-- Define the theorem statement
theorem bracelets_count : stones_total / stones_per_bracelet = 3 := by
  sorry

end NUMINAMATH_GPT_bracelets_count_l1251_125114


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1251_125102

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}

-- Conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def S9_is_90 (S : ℕ → ℝ) := S 9 = 90

-- The proof goal
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : is_arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : S9_is_90 S) :
  a 3 + a 5 + a 7 = 30 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1251_125102


namespace NUMINAMATH_GPT_find_x_l1251_125136

-- Definitions of the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Inner product definition
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Perpendicular condition
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_x (x : ℝ) (h : is_perpendicular vector_a (vector_b x)) : x = 4 :=
  sorry

end NUMINAMATH_GPT_find_x_l1251_125136


namespace NUMINAMATH_GPT_find_x_l1251_125137

/-!
# Problem Statement
Given that the segment with endpoints (-8, 0) and (32, 0) is the diameter of a circle,
and the point (x, 20) lies on the circle, prove that x = 12.
-/

def point_on_circle (x y : ℝ) (center_x center_y radius : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2

theorem find_x : 
  let center_x := (32 + (-8)) / 2
  let center_y := (0 + 0) / 2
  let radius := (32 - (-8)) / 2
  ∃ x : ℝ, point_on_circle x 20 center_x center_y radius → x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1251_125137


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1251_125177

-- Define the parameters for m and n.
def m : ℚ := -1 / 3
def n : ℚ := 1 / 2

-- Define the expression to simplify and evaluate.
def complex_expr (m n : ℚ) : ℚ :=
  -2 * (m * n - 3 * m^2) + 3 * (2 * m * n - 5 * m^2)

-- State the theorem that proves the expression equals -5/3.
theorem simplify_and_evaluate_expression :
  complex_expr m n = -5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1251_125177


namespace NUMINAMATH_GPT_quotient_when_m_divided_by_11_is_2_l1251_125196

theorem quotient_when_m_divided_by_11_is_2 :
  let n_values := [1, 2, 3, 4, 5]
  let squares := n_values.map (λ n => n^2)
  let remainders := List.eraseDup (squares.map (λ x => x % 11))
  let m := remainders.sum
  m / 11 = 2 :=
by
  sorry

end NUMINAMATH_GPT_quotient_when_m_divided_by_11_is_2_l1251_125196


namespace NUMINAMATH_GPT_prize_winners_l1251_125199

variable (Elaine Frank George Hannah : Prop)

axiom ElaineImpliesFrank : Elaine → Frank
axiom FrankImpliesGeorge : Frank → George
axiom GeorgeImpliesHannah : George → Hannah
axiom OnlyTwoWinners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah)

theorem prize_winners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) → (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) :=
by
  sorry

end NUMINAMATH_GPT_prize_winners_l1251_125199


namespace NUMINAMATH_GPT_time_for_one_paragraph_l1251_125183

-- Definitions for the given conditions
def short_answer_time := 3 -- minutes
def essay_time := 60 -- minutes
def total_homework_time := 240 -- minutes
def essays_assigned := 2
def paragraphs_assigned := 5
def short_answers_assigned := 15

-- Function to calculate total time from given conditions
def total_time_for_essays (essays : ℕ) : ℕ :=
  essays * essay_time

def total_time_for_short_answers (short_answers : ℕ) : ℕ :=
  short_answers * short_answer_time

def total_time_for_paragraphs (paragraphs : ℕ) : ℕ :=
  total_homework_time - (total_time_for_essays essays_assigned + total_time_for_short_answers short_answers_assigned)

def time_per_paragraph (paragraphs : ℕ) : ℕ :=
  total_time_for_paragraphs paragraphs / paragraphs_assigned

-- Proving the question part
theorem time_for_one_paragraph : 
  time_per_paragraph paragraphs_assigned = 15 := by
  sorry

end NUMINAMATH_GPT_time_for_one_paragraph_l1251_125183


namespace NUMINAMATH_GPT_smallest_integer_n_l1251_125107

theorem smallest_integer_n (n : ℕ) (h : Nat.lcm 60 n / Nat.gcd 60 n = 75) : n = 500 :=
sorry

end NUMINAMATH_GPT_smallest_integer_n_l1251_125107


namespace NUMINAMATH_GPT_Andre_final_price_l1251_125191

theorem Andre_final_price :
  let treadmill_price := 1350
  let treadmill_discount_rate := 0.30
  let plate_price := 60
  let num_of_plates := 2
  let plate_discount_rate := 0.15
  let sales_tax_rate := 0.07
  let treadmill_discount := treadmill_price * treadmill_discount_rate
  let treadmill_discounted_price := treadmill_price - treadmill_discount
  let total_plate_price := plate_price * num_of_plates
  let plate_discount := total_plate_price * plate_discount_rate
  let plate_discounted_price := total_plate_price - plate_discount
  let total_price_before_tax := treadmill_discounted_price + plate_discounted_price
  let sales_tax := total_price_before_tax * sales_tax_rate
  let final_price := total_price_before_tax + sales_tax
  final_price = 1120.29 := 
by
  repeat { 
    sorry 
  }

end NUMINAMATH_GPT_Andre_final_price_l1251_125191


namespace NUMINAMATH_GPT_largest_n_for_factoring_polynomial_l1251_125190

theorem largest_n_for_factoring_polynomial :
  ∃ A B : ℤ, A * B = 120 ∧ (∀ n, (5 * 120 + 1 ≤ n → n ≤ 601)) := sorry

end NUMINAMATH_GPT_largest_n_for_factoring_polynomial_l1251_125190


namespace NUMINAMATH_GPT_trig_identity_l1251_125172

open Real

theorem trig_identity : sin (20 * (π / 180)) * cos (10 * (π / 180)) - cos (200 * (π / 180)) * sin (10 * (π / 180)) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1251_125172


namespace NUMINAMATH_GPT_find_number_of_persons_l1251_125159

-- Definitions of the given conditions
def total_amount : ℕ := 42900
def amount_per_person : ℕ := 1950

-- The statement to prove
theorem find_number_of_persons (n : ℕ) (h : total_amount = n * amount_per_person) : n = 22 :=
sorry

end NUMINAMATH_GPT_find_number_of_persons_l1251_125159


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1251_125170

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (x y : ℝ), x = -2 ∧ y = 1 ∧ y = k * x + 2 * k + 1 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1251_125170


namespace NUMINAMATH_GPT_sugar_in_first_combination_l1251_125130

def cost_per_pound : ℝ := 0.45
def cost_combination_1 (S : ℝ) : ℝ := cost_per_pound * S + cost_per_pound * 16
def cost_combination_2 : ℝ := cost_per_pound * 30 + cost_per_pound * 25
def total_weight_combination_2 : ℕ := 30 + 25
def total_weight_combination_1 (S : ℕ) : ℕ := S + 16

theorem sugar_in_first_combination :
  ∀ (S : ℕ), cost_combination_1 S = 26 ∧ cost_combination_2 = 26 → total_weight_combination_1 S = total_weight_combination_2 → S = 39 :=
by sorry

end NUMINAMATH_GPT_sugar_in_first_combination_l1251_125130


namespace NUMINAMATH_GPT_jimmy_max_loss_l1251_125133

-- Definition of the conditions
def exam_points : ℕ := 20
def number_of_exams : ℕ := 3
def points_lost_for_behavior : ℕ := 5
def passing_score : ℕ := 50

-- Total points Jimmy has earned and lost
def total_points : ℕ := (number_of_exams * exam_points) - points_lost_for_behavior

-- The maximum points Jimmy can lose and still pass
def max_points_jimmy_can_lose : ℕ := total_points - passing_score

-- Statement to prove
theorem jimmy_max_loss : max_points_jimmy_can_lose = 5 := 
by
  sorry

end NUMINAMATH_GPT_jimmy_max_loss_l1251_125133


namespace NUMINAMATH_GPT_range_of_a_l1251_125171

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ f a 0) : 0 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1251_125171


namespace NUMINAMATH_GPT_Markus_bags_count_l1251_125151

-- Definitions of the conditions
def Mara_bags : ℕ := 12
def Mara_marbles_per_bag : ℕ := 2
def Markus_marbles_per_bag : ℕ := 13
def marbles_difference : ℕ := 2

-- Derived conditions
def Mara_total_marbles : ℕ := Mara_bags * Mara_marbles_per_bag
def Markus_total_marbles : ℕ := Mara_total_marbles + marbles_difference

-- Statement to prove
theorem Markus_bags_count : Markus_total_marbles / Markus_marbles_per_bag = 2 :=
by
  -- Skip the proof, leaving it as a task for the prover
  sorry

end NUMINAMATH_GPT_Markus_bags_count_l1251_125151


namespace NUMINAMATH_GPT_bicycles_sold_saturday_l1251_125165

variable (S : ℕ)

theorem bicycles_sold_saturday :
  let net_increase_friday := 15 - 10
  let net_increase_saturday := 8 - S
  let net_increase_sunday := 11 - 9
  (net_increase_friday + net_increase_saturday + net_increase_sunday = 3) → 
  S = 12 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_bicycles_sold_saturday_l1251_125165


namespace NUMINAMATH_GPT_box_base_length_max_l1251_125162

noncomputable def V (x : ℝ) := x^2 * ((60 - x) / 2)

theorem box_base_length_max 
  (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 60)
  (h3 : ∀ y : ℝ, 0 < y ∧ y < 60 → V x ≥ V y)
  : x = 40 :=
sorry

end NUMINAMATH_GPT_box_base_length_max_l1251_125162


namespace NUMINAMATH_GPT_man_son_age_ratio_is_two_to_one_l1251_125105

-- Define the present age of the son
def son_present_age := 33

-- Define the present age of the man
def man_present_age := son_present_age + 35

-- Define the son's age in two years
def son_age_in_two_years := son_present_age + 2

-- Define the man's age in two years
def man_age_in_two_years := man_present_age + 2

-- Define the expected ratio of the man's age to son's age in two years
def ratio := man_age_in_two_years / son_age_in_two_years

-- Theorem statement verifying the ratio
theorem man_son_age_ratio_is_two_to_one : ratio = 2 := by
  -- Note: Proof not required, so we use sorry to denote the missing proof
  sorry

end NUMINAMATH_GPT_man_son_age_ratio_is_two_to_one_l1251_125105


namespace NUMINAMATH_GPT_jessica_money_left_l1251_125152

theorem jessica_money_left : 
  let initial_amount := 11.73
  let amount_spent := 10.22
  initial_amount - amount_spent = 1.51 :=
by
  sorry

end NUMINAMATH_GPT_jessica_money_left_l1251_125152


namespace NUMINAMATH_GPT_savings_per_month_l1251_125139

noncomputable def annual_salary : ℝ := 48000
noncomputable def monthly_payments : ℝ := 12
noncomputable def savings_percentage : ℝ := 0.10

theorem savings_per_month :
  (annual_salary / monthly_payments) * savings_percentage = 400 :=
by
  sorry

end NUMINAMATH_GPT_savings_per_month_l1251_125139


namespace NUMINAMATH_GPT_mul_101_eq_10201_l1251_125123

theorem mul_101_eq_10201 : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_GPT_mul_101_eq_10201_l1251_125123


namespace NUMINAMATH_GPT_number_of_real_solutions_l1251_125182

noncomputable def system_of_equations_solutions_count (x : ℝ) : Prop :=
  3 * x^2 - 45 * (⌊x⌋:ℝ) + 60 = 0 ∧ 2 * x - 3 * (⌊x⌋:ℝ) + 1 = 0

theorem number_of_real_solutions : ∃ (x₁ x₂ x₃ : ℝ), system_of_equations_solutions_count x₁ ∧ system_of_equations_solutions_count x₂ ∧ system_of_equations_solutions_count x₃ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ :=
sorry

end NUMINAMATH_GPT_number_of_real_solutions_l1251_125182


namespace NUMINAMATH_GPT_part_I_part_II_l1251_125157

noncomputable section

def f (x a : ℝ) : ℝ := |x + a| + |x - (1 / a)|

theorem part_I (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -5/2 ∨ x ≥ 5/2 := by
  sorry

theorem part_II (a m : ℝ) (h : ∀ x : ℝ, f x a ≥ |m - 1|) : -1 ≤ m ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1251_125157


namespace NUMINAMATH_GPT_planes_parallel_or_coincide_l1251_125174

-- Define normal vectors
def normal_vector_u : ℝ × ℝ × ℝ := (1, 2, -2)
def normal_vector_v : ℝ × ℝ × ℝ := (-3, -6, 6)

-- The theorem states that planes defined by these normal vectors are either 
-- parallel or coincide if their normal vectors are collinear.
theorem planes_parallel_or_coincide (u v : ℝ × ℝ × ℝ) 
  (h_u : u = normal_vector_u) 
  (h_v : v = normal_vector_v) 
  (h_collinear : v = (-3) • u) : 
    ∃ k : ℝ, v = k • u := 
by
  sorry

end NUMINAMATH_GPT_planes_parallel_or_coincide_l1251_125174


namespace NUMINAMATH_GPT_union_A_B_l1251_125156

noncomputable def A : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def B : Set ℝ := {x | x^2 - 4 = 0}

theorem union_A_B : A ∪ B = {-3, -2, 2} := by
  sorry

end NUMINAMATH_GPT_union_A_B_l1251_125156


namespace NUMINAMATH_GPT_slope_of_line_l1251_125122

theorem slope_of_line (x1 y1 x2 y2 : ℝ)
  (h1 : 4 * y1 + 6 * x1 = 0)
  (h2 : 4 * y2 + 6 * x2 = 0)
  (h1x2 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by sorry

end NUMINAMATH_GPT_slope_of_line_l1251_125122


namespace NUMINAMATH_GPT_lake_circumference_ratio_l1251_125187

theorem lake_circumference_ratio 
    (D C : ℝ) 
    (hD : D = 100) 
    (hC : C = 314) : 
    C / D = 3.14 := 
sorry

end NUMINAMATH_GPT_lake_circumference_ratio_l1251_125187


namespace NUMINAMATH_GPT_probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l1251_125198

-- Probability for different numbers facing up when die is thrown twice
theorem probability_different_numbers :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := n_faces * (n_faces - 1)
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry -- Proof to be filled

-- Probability for sum of numbers being 6 when die is thrown twice
theorem probability_sum_six :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := 5
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 36 :=
by
  sorry -- Proof to be filled

-- Probability for exactly three outcomes being odd when die is thrown five times
theorem probability_three_odds_in_five_throws :
  let n_faces := 6
  let n_throws := 5
  let p_odd := 3 / n_faces
  let p_even := 1 - p_odd
  let binomial_coeff := Nat.choose n_throws 3
  let p_three_odds := (binomial_coeff : ℚ) * (p_odd ^ 3) * (p_even ^ 2)
  p_three_odds = 5 / 16 :=
by
  sorry -- Proof to be filled

end NUMINAMATH_GPT_probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l1251_125198


namespace NUMINAMATH_GPT_initial_number_2008_l1251_125103

theorem initial_number_2008 (x : ℕ) (h : x = 2008 ∨ (∃ y: ℕ, (x = 2*y + 1 ∨ (x = y / (y + 2))))): x = 2008 :=
by
  cases h with
  | inl h2008 => exact h2008
  | inr hexists => cases hexists with
    | intro y hy =>
        cases hy
        case inl h2y => sorry
        case inr hdiv => sorry

end NUMINAMATH_GPT_initial_number_2008_l1251_125103


namespace NUMINAMATH_GPT_slope_angle_of_line_l1251_125119

theorem slope_angle_of_line (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : (m^2 + n^2) / m^2 = 4) :
  ∃ θ : ℝ, θ = π / 6 ∨ θ = 5 * π / 6 :=
by
  sorry

end NUMINAMATH_GPT_slope_angle_of_line_l1251_125119


namespace NUMINAMATH_GPT_find_d_l1251_125155

theorem find_d (a b c d x : ℝ)
  (h1 : ∀ x, 2 ≤ a * (Real.cos (b * x + c)) + d ∧ a * (Real.cos (b * x + c)) + d ≤ 4)
  (h2 : Real.cos (b * 0 + c) = 1) :
  d = 3 :=
sorry

end NUMINAMATH_GPT_find_d_l1251_125155


namespace NUMINAMATH_GPT_age_of_replaced_person_l1251_125108

theorem age_of_replaced_person
    (T : ℕ) -- total age of the original group of 10 persons
    (age_person_replaced : ℕ) -- age of the person who was replaced
    (age_new_person : ℕ) -- age of the new person
    (h1 : age_new_person = 15)
    (h2 : (T / 10) - 3 = (T - age_person_replaced + age_new_person) / 10) :
    age_person_replaced = 45 :=
by
  sorry

end NUMINAMATH_GPT_age_of_replaced_person_l1251_125108


namespace NUMINAMATH_GPT_no_positive_integer_satisfies_conditions_l1251_125192

theorem no_positive_integer_satisfies_conditions :
  ¬∃ (n : ℕ), n > 1 ∧ (∃ (p1 : ℕ), Prime p1 ∧ n = p1^2) ∧ (∃ (p2 : ℕ), Prime p2 ∧ 3 * n + 16 = p2^2) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_satisfies_conditions_l1251_125192


namespace NUMINAMATH_GPT_problem1_l1251_125145

theorem problem1 : 3 * 403 + 5 * 403 + 2 * 403 + 401 = 4431 := by 
  sorry

end NUMINAMATH_GPT_problem1_l1251_125145


namespace NUMINAMATH_GPT_sum_of_digits_succ_2080_l1251_125164

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_succ_2080 (m : ℕ) (h : sum_of_digits m = 2080) :
  sum_of_digits (m + 1) = 2081 ∨ sum_of_digits (m + 1) = 2090 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_succ_2080_l1251_125164


namespace NUMINAMATH_GPT_f_2020_eq_neg_1_l1251_125169

noncomputable def f: ℝ → ℝ :=
sorry

axiom f_2_x_eq_neg_f_x : ∀ x: ℝ, f (2 - x) = -f x
axiom f_x_minus_2_eq_f_neg_x : ∀ x: ℝ, f (x - 2) = f (-x)
axiom f_specific : ∀ x : ℝ, -1 < x ∧ x < 1 -> f x = x^2 + 1

theorem f_2020_eq_neg_1 : f 2020 = -1 :=
sorry

end NUMINAMATH_GPT_f_2020_eq_neg_1_l1251_125169


namespace NUMINAMATH_GPT_present_population_l1251_125116

theorem present_population (P : ℝ) (h1 : (P : ℝ) * (1 + 0.1) ^ 2 = 14520) : P = 12000 :=
sorry

end NUMINAMATH_GPT_present_population_l1251_125116


namespace NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1251_125189

theorem arithmetic_sequence_15th_term :
  let first_term := 3
  let second_term := 8
  let third_term := 13
  let common_difference := second_term - first_term
  (first_term + (15 - 1) * common_difference) = 73 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1251_125189


namespace NUMINAMATH_GPT_variation_of_powers_l1251_125144

theorem variation_of_powers (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by
  sorry

end NUMINAMATH_GPT_variation_of_powers_l1251_125144


namespace NUMINAMATH_GPT_total_necklaces_made_l1251_125150

-- Definitions based on conditions
def first_machine_necklaces : ℝ := 45
def second_machine_necklaces : ℝ := 2.4 * first_machine_necklaces

-- Proof statement
theorem total_necklaces_made : (first_machine_necklaces + second_machine_necklaces) = 153 := by
  sorry

end NUMINAMATH_GPT_total_necklaces_made_l1251_125150


namespace NUMINAMATH_GPT_total_corn_yield_l1251_125100

/-- 
The total corn yield in centners, harvested from a certain field area, is expressed 
as a four-digit number composed of the digits 0, 2, 3, and 5. When the average 
yield per hectare was calculated, it was found to be the same number of centners 
as the number of hectares of the field area. 
This statement proves that the total corn yield is 3025. 
-/
theorem total_corn_yield : ∃ (Y A : ℕ), (Y = A^2) ∧ (A >= 10 ∧ A < 100) ∧ 
  (Y / 1000 != 0) ∧ (Y / 1000 != 1) ∧ (Y / 10 % 10 != 4) ∧ 
  (Y % 10 != 1) ∧ (Y % 10 = 0 ∨ Y % 10 = 5) ∧ 
  (Y / 100 % 10 == 0 ∨ Y / 100 % 10 == 2 ∨ Y / 100 % 10 == 3 ∨ Y / 100 % 10 == 5) ∧ 
  Y = 3025 := 
by 
  sorry

end NUMINAMATH_GPT_total_corn_yield_l1251_125100


namespace NUMINAMATH_GPT_investment_time_period_l1251_125160

variable (P : ℝ) (r15 r12 : ℝ) (T : ℝ)
variable (hP : P = 15000)
variable (hr15 : r15 = 0.15)
variable (hr12 : r12 = 0.12)
variable (diff : 2250 * T - 1800 * T = 900)

theorem investment_time_period :
  T = 2 := by
  sorry

end NUMINAMATH_GPT_investment_time_period_l1251_125160


namespace NUMINAMATH_GPT_hyperbola_eccentricity_proof_l1251_125149

noncomputable def hyperbola_eccentricity (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) : 
    ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_proof (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) :    
    hyperbola_eccentricity a b k1 k2 ha hb C_on_hyperbola slope_condition minimized_expr = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_proof_l1251_125149


namespace NUMINAMATH_GPT_perpendicular_lines_a_eq_3_l1251_125117

theorem perpendicular_lines_a_eq_3 (a : ℝ) :
  let l₁ := (a + 1) * x + 2 * y + 6
  let l₂ := x + (a - 5) * y + a^2 - 1
  (a ≠ 5 → -((a + 1) / 2) * (1 / (5 - a)) = -1) → a = 3 := by
  intro l₁ l₂ h
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_eq_3_l1251_125117


namespace NUMINAMATH_GPT_johns_contribution_l1251_125181

-- Definitions
variables (A J : ℝ)
axiom h1 : 1.5 * A = 75
axiom h2 : (2 * A + J) / 3 = 75

-- Statement of the proof problem
theorem johns_contribution : J = 125 :=
by
  sorry

end NUMINAMATH_GPT_johns_contribution_l1251_125181


namespace NUMINAMATH_GPT_blue_pens_removed_l1251_125127

def initial_blue_pens := 9
def initial_black_pens := 21
def initial_red_pens := 6
def removed_black_pens := 7
def pens_left := 25

theorem blue_pens_removed (x : ℕ) :
  initial_blue_pens - x + (initial_black_pens - removed_black_pens) + initial_red_pens = pens_left ↔ x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_blue_pens_removed_l1251_125127


namespace NUMINAMATH_GPT_solution_set_inequality_l1251_125173

theorem solution_set_inequality (m : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.exp x + Real.exp (-x))
  (h2 : ∀ x, f (-x) = f x) (h3 : ∀ x, 0 ≤ x → ∀ y, 0 ≤ y → x ≤ y → f x ≤ f y) :
  (f (2 * m) > f (m - 2)) ↔ (m > (2 / 3) ∨ m < -2) :=
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1251_125173


namespace NUMINAMATH_GPT_combined_cost_price_l1251_125158

theorem combined_cost_price :
  let face_value_A : ℝ := 100
  let discount_A : ℝ := 2
  let purchase_price_A := face_value_A - (discount_A / 100 * face_value_A)
  let brokerage_A := 0.2 / 100 * purchase_price_A
  let total_cost_price_A := purchase_price_A + brokerage_A

  let face_value_B : ℝ := 100
  let premium_B : ℝ := 1.5
  let purchase_price_B := face_value_B + (premium_B / 100 * face_value_B)
  let brokerage_B := 0.2 / 100 * purchase_price_B
  let total_cost_price_B := purchase_price_B + brokerage_B

  let combined_cost_price := total_cost_price_A + total_cost_price_B

  combined_cost_price = 199.899 := by
  sorry

end NUMINAMATH_GPT_combined_cost_price_l1251_125158


namespace NUMINAMATH_GPT_problem_statement_l1251_125166

open Real

theorem problem_statement (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : A ≠ 0)
    (h3 : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) : 
    |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| := sorry

end NUMINAMATH_GPT_problem_statement_l1251_125166


namespace NUMINAMATH_GPT_bounded_g_of_f_l1251_125180

theorem bounded_g_of_f
  (f g : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := 
sorry

end NUMINAMATH_GPT_bounded_g_of_f_l1251_125180


namespace NUMINAMATH_GPT_relationship_between_abcd_l1251_125178

theorem relationship_between_abcd (a b c d : ℝ) (h : d ≠ 0) :
  (∀ x : ℝ, (a * x + c) / (b * x + d) = (a + c * x) / (b + d * x)) ↔ a / b = c / d :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_abcd_l1251_125178


namespace NUMINAMATH_GPT_bill_steps_l1251_125161

theorem bill_steps (step_length : ℝ) (total_distance : ℝ) (n_steps : ℕ) 
  (h_step_length : step_length = 1 / 2) 
  (h_total_distance : total_distance = 12) 
  (h_n_steps : n_steps = total_distance / step_length) : 
  n_steps = 24 :=
by sorry

end NUMINAMATH_GPT_bill_steps_l1251_125161


namespace NUMINAMATH_GPT_simplify_expression_l1251_125163

theorem simplify_expression : 
  let a := (3 + 2 : ℚ)
  let b := a⁻¹ + 2
  let c := b⁻¹ + 2
  let d := c⁻¹ + 2
  d = 65 / 27 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1251_125163


namespace NUMINAMATH_GPT_correct_answer_statement_l1251_125138

theorem correct_answer_statement
  (A := "In order to understand the situation of extracurricular reading among middle school students in China, a comprehensive survey should be conducted.")
  (B := "The median and mode of a set of data 1, 2, 5, 5, 5, 3, 3 are both 5.")
  (C := "When flipping a coin 200 times, there will definitely be 100 times when it lands 'heads up.'")
  (D := "If the variance of data set A is 0.03 and the variance of data set B is 0.1, then data set A is more stable than data set B.")
  (correct_answer := "D") : 
  correct_answer = "D" :=
  by sorry

end NUMINAMATH_GPT_correct_answer_statement_l1251_125138


namespace NUMINAMATH_GPT_sharon_trip_distance_l1251_125140

theorem sharon_trip_distance
  (h1 : ∀ (d : ℝ), (180 * d) = 1 ∨ (d = 0))  -- Any distance traveled in 180 minutes follows 180d=1 (usual speed)
  (h2 : ∀ (d : ℝ), (276 * (d - 20 / 60)) = 1 ∨ (d = 0))  -- With reduction in speed due to snowstorm too follows a similar relation
  (h3: ∀ (total_time : ℝ), total_time = 276 ∨ total_time = 0)  -- Total time is 276 minutes
  : ∃ (x : ℝ), x = 135 := sorry

end NUMINAMATH_GPT_sharon_trip_distance_l1251_125140


namespace NUMINAMATH_GPT_vertical_asymptote_l1251_125154

theorem vertical_asymptote (x : ℝ) : (4 * x + 6 = 0) -> x = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptote_l1251_125154


namespace NUMINAMATH_GPT_ganesh_speed_x_to_y_l1251_125112

-- Define the conditions
variables (D : ℝ) (V : ℝ)

-- Theorem statement: Prove that Ganesh's average speed from x to y is 44 km/hr
theorem ganesh_speed_x_to_y
  (H1 : 39.6 = 2 * D / (D / V + D / 36))
  (H2 : V = 44) :
  true :=
sorry

end NUMINAMATH_GPT_ganesh_speed_x_to_y_l1251_125112


namespace NUMINAMATH_GPT_cricket_team_initial_games_l1251_125128

theorem cricket_team_initial_games
  (initial_games : ℕ)
  (won_30_percent_initially : ℕ)
  (additional_wins : ℕ)
  (final_win_rate : ℚ) :
  won_30_percent_initially = initial_games * 30 / 100 →
  final_win_rate = (won_30_percent_initially + additional_wins) / (initial_games + additional_wins) →
  additional_wins = 55 →
  final_win_rate = 52 / 100 →
  initial_games = 120 := by sorry

end NUMINAMATH_GPT_cricket_team_initial_games_l1251_125128


namespace NUMINAMATH_GPT_units_digit_of_k3_plus_5k_l1251_125106

def k : ℕ := 2024^2 + 3^2024

theorem units_digit_of_k3_plus_5k (k := 2024^2 + 3^2024) : 
  ((k^3 + 5^k) % 10) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_of_k3_plus_5k_l1251_125106


namespace NUMINAMATH_GPT_find_product_of_constants_l1251_125175

theorem find_product_of_constants
  (M1 M2 : ℝ)
  (h : ∀ x : ℝ, (x - 1) * (x - 2) ≠ 0 → (45 * x - 31) / (x * x - 3 * x + 2) = M1 / (x - 1) + M2 / (x - 2)) :
  M1 * M2 = -826 :=
sorry

end NUMINAMATH_GPT_find_product_of_constants_l1251_125175


namespace NUMINAMATH_GPT_count_even_three_digit_numbers_less_than_800_l1251_125194

def even_three_digit_numbers_less_than_800 : Nat :=
  let hundreds_choices := 7
  let tens_choices := 8
  let units_choices := 4
  hundreds_choices * tens_choices * units_choices

theorem count_even_three_digit_numbers_less_than_800 :
  even_three_digit_numbers_less_than_800 = 224 := 
by 
  unfold even_three_digit_numbers_less_than_800
  rfl

end NUMINAMATH_GPT_count_even_three_digit_numbers_less_than_800_l1251_125194


namespace NUMINAMATH_GPT_min_value_of_function_l1251_125131

noncomputable def f (a x : ℝ) : ℝ := (a^x - a)^2 + (a^(-x) - a)^2

theorem min_value_of_function (a : ℝ) (h : a > 0) : ∃ x : ℝ, f a x = 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_function_l1251_125131


namespace NUMINAMATH_GPT_ratio_is_five_over_twelve_l1251_125109

theorem ratio_is_five_over_twelve (a b c d : ℚ) (h1 : b = 4 * a) (h2 : d = 2 * c) :
    (a + b) / (c + d) = 5 / 12 :=
sorry

end NUMINAMATH_GPT_ratio_is_five_over_twelve_l1251_125109


namespace NUMINAMATH_GPT_solve_pair_l1251_125118

theorem solve_pair (x y : ℕ) (h₁ : x = 12785 ∧ y = 12768 ∨ x = 11888 ∧ y = 11893 ∨ x = 12784 ∧ y = 12770 ∨ x = 1947 ∧ y = 1945) :
  1983 = 1982 * 11888 - 1981 * 11893 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_pair_l1251_125118


namespace NUMINAMATH_GPT_claire_needs_80_tiles_l1251_125126

def room_length : ℕ := 14
def room_width : ℕ := 18
def border_width : ℕ := 2
def small_tile_side : ℕ := 1
def large_tile_side : ℕ := 3

def num_small_tiles : ℕ :=
  let perimeter_length := (2 * (room_width - 2 * border_width))
  let perimeter_width := (2 * (room_length - 2 * border_width))
  let corner_tiles := (2 * border_width) * 4
  perimeter_length + perimeter_width + corner_tiles

def num_large_tiles : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  Nat.ceil (inner_area / (large_tile_side * large_tile_side))

theorem claire_needs_80_tiles : num_small_tiles + num_large_tiles = 80 :=
by sorry

end NUMINAMATH_GPT_claire_needs_80_tiles_l1251_125126


namespace NUMINAMATH_GPT_bubble_bath_amount_l1251_125176

noncomputable def total_bubble_bath_needed 
  (couple_rooms : ℕ) (single_rooms : ℕ) (people_per_couple_room : ℕ) (people_per_single_room : ℕ) (ml_per_bath : ℕ) : ℕ :=
  couple_rooms * people_per_couple_room * ml_per_bath + single_rooms * people_per_single_room * ml_per_bath

theorem bubble_bath_amount :
  total_bubble_bath_needed 13 14 2 1 10 = 400 := by 
  sorry

end NUMINAMATH_GPT_bubble_bath_amount_l1251_125176


namespace NUMINAMATH_GPT_melanie_trout_catch_l1251_125115

def trout_caught_sara : ℕ := 5
def trout_caught_melanie (sara_trout : ℕ) : ℕ := 2 * sara_trout

theorem melanie_trout_catch :
  trout_caught_melanie trout_caught_sara = 10 :=
by
  sorry

end NUMINAMATH_GPT_melanie_trout_catch_l1251_125115


namespace NUMINAMATH_GPT_sandy_has_four_times_more_marbles_l1251_125143

-- Definitions based on conditions
def jessica_red_marbles : ℕ := 3 * 12
def sandy_red_marbles : ℕ := 144

-- The theorem to prove
theorem sandy_has_four_times_more_marbles : sandy_red_marbles = 4 * jessica_red_marbles :=
by
  sorry

end NUMINAMATH_GPT_sandy_has_four_times_more_marbles_l1251_125143


namespace NUMINAMATH_GPT_decrease_percent_revenue_l1251_125129

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.10 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 12 := by
  sorry

end NUMINAMATH_GPT_decrease_percent_revenue_l1251_125129


namespace NUMINAMATH_GPT_Luke_trips_l1251_125184

variable (carries : Nat) (table1 : Nat) (table2 : Nat)

theorem Luke_trips (h1 : carries = 4) (h2 : table1 = 20) (h3 : table2 = 16) : 
  (table1 / carries + table2 / carries) = 9 :=
by
  sorry

end NUMINAMATH_GPT_Luke_trips_l1251_125184


namespace NUMINAMATH_GPT_minimum_races_to_find_top3_l1251_125188

-- Define a constant to represent the number of horses and maximum horses per race
def total_horses : ℕ := 25
def max_horses_per_race : ℕ := 5

-- Define the problem statement as a theorem
theorem minimum_races_to_find_top3 (total_horses : ℕ) (max_horses_per_race : ℕ) : ℕ :=
  if total_horses = 25 ∧ max_horses_per_race = 5 then 7 else sorry

end NUMINAMATH_GPT_minimum_races_to_find_top3_l1251_125188


namespace NUMINAMATH_GPT_unique_real_value_for_equal_roots_l1251_125193

-- Definitions of conditions
def quadratic_eq (p : ℝ) : Prop := 
  ∀ x : ℝ, x^2 - (p + 1) * x + p = 0

-- Statement of the problem
theorem unique_real_value_for_equal_roots :
  ∃! p : ℝ, ∀ x y : ℝ, (x^2 - (p+1)*x + p = 0) ∧ (y^2 - (p+1)*y + p = 0) → x = y := 
sorry

end NUMINAMATH_GPT_unique_real_value_for_equal_roots_l1251_125193


namespace NUMINAMATH_GPT_banker_l1251_125125

-- Define the given conditions
def present_worth : ℝ := 400
def interest_rate : ℝ := 0.10
def time_period : ℕ := 3

-- Define the amount due in the future
def amount_due (PW : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  PW * (1 + r) ^ n

-- Define the banker's gain
def bankers_gain (A PW : ℝ) : ℝ :=
  A - PW

-- State the theorem we need to prove
theorem banker's_gain_is_correct :
  bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 132.4 :=
by sorry

end NUMINAMATH_GPT_banker_l1251_125125


namespace NUMINAMATH_GPT_distance_between_planes_l1251_125153

def plane1 (x y z : ℝ) := 3 * x - y + z - 3 = 0
def plane2 (x y z : ℝ) := 6 * x - 2 * y + 2 * z + 4 = 0

theorem distance_between_planes :
  ∃ d : ℝ, d = (5 * Real.sqrt 11) / 11 ∧ 
            ∀ x y z : ℝ, plane1 x y z → plane2 x y z → d = (5 * Real.sqrt 11) / 11 :=
sorry

end NUMINAMATH_GPT_distance_between_planes_l1251_125153


namespace NUMINAMATH_GPT_math_problem_l1251_125113

theorem math_problem :
  (∃ n : ℕ, 28 = 4 * n) ∧
  ((∃ n1 : ℕ, 361 = 19 * n1) ∧ ¬(∃ n2 : ℕ, 63 = 19 * n2)) ∧
  (¬((∃ n3 : ℕ, 90 = 30 * n3) ∧ ¬(∃ n4 : ℕ, 65 = 30 * n4))) ∧
  ((∃ n5 : ℕ, 45 = 15 * n5) ∧ (∃ n6 : ℕ, 30 = 15 * n6)) ∧
  (∃ n7 : ℕ, 144 = 12 * n7) :=
by {
  -- We need to prove each condition to be true and then prove the statements A, B, D, E are true.
  sorry
}

end NUMINAMATH_GPT_math_problem_l1251_125113


namespace NUMINAMATH_GPT_power_subtraction_l1251_125185

variable {a m n : ℝ}

theorem power_subtraction (hm : a^m = 8) (hn : a^n = 2) : a^(m - 3 * n) = 1 := by
  sorry

end NUMINAMATH_GPT_power_subtraction_l1251_125185


namespace NUMINAMATH_GPT_seq_a6_l1251_125168

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 3 * a n - 2

theorem seq_a6 (a : ℕ → ℕ) (h : seq a) : a 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_seq_a6_l1251_125168


namespace NUMINAMATH_GPT_most_frequent_data_is_mode_l1251_125101

-- Define the options
inductive Options where
  | Mean
  | Mode
  | Median
  | Frequency

-- Define the problem statement
def mostFrequentDataTerm (freqMost : String) : Options :=
  if freqMost == "Mode" then 
    Options.Mode
  else if freqMost == "Mean" then 
    Options.Mean
  else if freqMost == "Median" then 
    Options.Median
  else 
    Options.Frequency

-- Statement of the problem as a theorem
theorem most_frequent_data_is_mode (freqMost : String) :
  mostFrequentDataTerm freqMost = Options.Mode :=
by
  sorry

end NUMINAMATH_GPT_most_frequent_data_is_mode_l1251_125101


namespace NUMINAMATH_GPT_degree_to_radian_conversion_l1251_125121

theorem degree_to_radian_conversion : (1440 * (Real.pi / 180) = 8 * Real.pi) := 
by
  sorry

end NUMINAMATH_GPT_degree_to_radian_conversion_l1251_125121


namespace NUMINAMATH_GPT_min_initial_seeds_l1251_125120

/-- Given conditions:
  - The farmer needs to sell at least 10,000 watermelons each year.
  - Each watermelon produces 250 seeds when used for seeds but cannot be sold if used for seeds.
  - We need to find the minimum number of initial seeds S the farmer must buy to never buy seeds again.
-/
theorem min_initial_seeds : ∃ (S : ℕ), S = 10041 ∧ ∀ (yearly_sales : ℕ), yearly_sales = 10000 →
  ∀ (seed_yield : ℕ), seed_yield = 250 →
  ∃ (x : ℕ), S = yearly_sales + x ∧ x * seed_yield ≥ S :=
sorry

end NUMINAMATH_GPT_min_initial_seeds_l1251_125120


namespace NUMINAMATH_GPT_segments_count_bound_l1251_125148

-- Define the overall setup of the problem
variable (n : ℕ) (points : Finset ℕ)

-- The main hypothesis and goal
theorem segments_count_bound (hn : n ≥ 2) (hpoints : points.card = 3 * n) :
  ∃ A B : Finset (ℕ × ℕ), (∀ (i j : ℕ), i ∈ points → j ∈ points → i ≠ j → ((i, j) ∈ A ↔ (i, j) ∉ B)) ∧
  ∀ (X : Finset ℕ) (hX : X.card = n), ∃ C : Finset (ℕ × ℕ), (C ⊆ A) ∧ (X ⊆ points) ∧
  (∃ count : ℕ, count ≥ (n - 1) / 6 ∧ count = C.card ∧ ∀ (a b : ℕ), (a, b) ∈ C → a ∈ X ∧ b ∈ points \ X) := sorry

end NUMINAMATH_GPT_segments_count_bound_l1251_125148


namespace NUMINAMATH_GPT_compare_abc_l1251_125147

noncomputable def a : ℝ := Real.exp 0.25
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := -4 * Real.log 0.75

theorem compare_abc : b < c ∧ c < a := by
  -- Additional proof steps would follow here
  sorry

end NUMINAMATH_GPT_compare_abc_l1251_125147


namespace NUMINAMATH_GPT_choose_athlete_B_l1251_125135

variable (SA2 : ℝ) (SB2 : ℝ)
variable (num_shots : ℕ) (avg_rings : ℝ)

-- Conditions
def athlete_A_variance := SA2 = 3.5
def athlete_B_variance := SB2 = 2.8
def same_number_of_shots := true -- Implicit condition, doesn't need proof
def same_average_rings := true -- Implicit condition, doesn't need proof

-- Question: prove Athlete B should be chosen
theorem choose_athlete_B 
  (hA_var : athlete_A_variance SA2)
  (hB_var : athlete_B_variance SB2)
  (same_shots : same_number_of_shots)
  (same_avg : same_average_rings) :
  "B" = "B" :=
by 
  sorry

end NUMINAMATH_GPT_choose_athlete_B_l1251_125135


namespace NUMINAMATH_GPT_time_spent_on_spelling_l1251_125134

-- Define the given conditions
def total_time : Nat := 60
def math_time : Nat := 15
def reading_time : Nat := 27

-- Define the question as a Lean theorem statement
theorem time_spent_on_spelling : total_time - math_time - reading_time = 18 := sorry

end NUMINAMATH_GPT_time_spent_on_spelling_l1251_125134


namespace NUMINAMATH_GPT_Zixuan_amount_l1251_125141

noncomputable def amounts (X Y Z : ℕ) : Prop := 
  (X + Y + Z = 50) ∧
  (X = 3 * (Y + Z) / 2) ∧
  (Y = Z + 4)

theorem Zixuan_amount : ∃ Z : ℕ, ∃ X Y : ℕ, amounts X Y Z ∧ Z = 8 :=
by
  sorry

end NUMINAMATH_GPT_Zixuan_amount_l1251_125141


namespace NUMINAMATH_GPT_value_of_a_minus_b_l1251_125142

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a - b| = a - b) : a - b = 7 ∨ a - b = 3 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l1251_125142


namespace NUMINAMATH_GPT_average_weight_of_three_l1251_125179

theorem average_weight_of_three
  (rachel_weight jimmy_weight adam_weight : ℝ)
  (h1 : rachel_weight = 75)
  (h2 : jimmy_weight = rachel_weight + 6)
  (h3 : adam_weight = rachel_weight - 15) :
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_three_l1251_125179


namespace NUMINAMATH_GPT_total_oranges_picked_l1251_125197

theorem total_oranges_picked :
  let Mary_oranges := 14
  let Jason_oranges := 41
  let Amanda_oranges := 56
  Mary_oranges + Jason_oranges + Amanda_oranges = 111 := by
    sorry

end NUMINAMATH_GPT_total_oranges_picked_l1251_125197


namespace NUMINAMATH_GPT_cos_double_angle_of_parallel_vectors_l1251_125146

theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (h_parallel : (1 / 3, Real.tan α) = (Real.cos α, 1)) : 
  Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_of_parallel_vectors_l1251_125146


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1251_125132

-- Step d: Lean 4 statement
theorem sufficient_but_not_necessary_condition 
  (m n : ℕ) (e : ℚ) (h₁ : m = 5) (h₂ : n = 4) (h₃ : e = 3 / 5)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) :
  (m = 5 ∧ n = 4) → (e = 3 / 5) ∧ (¬(e = 3 / 5 → m = 5 ∧ n = 4)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1251_125132
