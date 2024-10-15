import Mathlib

namespace NUMINAMATH_GPT_pure_imaginary_solution_l731_73153

theorem pure_imaginary_solution (m : ℝ) (z : ℂ)
  (h1 : z = (m^2 - 1) + (m - 1) * I)
  (h2 : z.re = 0) : m = -1 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_solution_l731_73153


namespace NUMINAMATH_GPT_open_box_volume_l731_73164

-- Define the initial conditions
def length_of_sheet := 100
def width_of_sheet := 50
def height_of_parallelogram := 10
def base_of_parallelogram := 10

-- Define the expected dimensions of the box after cutting
def length_of_box := length_of_sheet - 2 * base_of_parallelogram
def width_of_box := width_of_sheet - 2 * base_of_parallelogram
def height_of_box := height_of_parallelogram

-- Define the expected volume of the box
def volume_of_box := length_of_box * width_of_box * height_of_box

-- Theorem to prove the correct volume of the box based on the given dimensions
theorem open_box_volume : volume_of_box = 24000 := by
  -- The proof will be included here
  sorry

end NUMINAMATH_GPT_open_box_volume_l731_73164


namespace NUMINAMATH_GPT_geo_seq_arith_seq_l731_73184

theorem geo_seq_arith_seq (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_gp : ∀ n, a_n (n+1) = a_n n * q)
  (h_pos : ∀ n, a_n n > 0) (h_arith : a_n 4 - a_n 3 = a_n 5 - a_n 4) 
  (hq_pos : q > 0) (hq_neq1 : q ≠ 1) :
  S 6 / S 3 = 2 := by
  sorry

end NUMINAMATH_GPT_geo_seq_arith_seq_l731_73184


namespace NUMINAMATH_GPT_product_of_integers_l731_73139

theorem product_of_integers
  (A B C D : ℕ)
  (hA : A > 0)
  (hB : B > 0)
  (hC : C > 0)
  (hD : D > 0)
  (h_sum : A + B + C + D = 72)
  (h_eq : A + 3 = B - 3 ∧ B - 3 = C * 3 ∧ C * 3 = D / 2) :
  A * B * C * D = 68040 := 
by
  sorry

end NUMINAMATH_GPT_product_of_integers_l731_73139


namespace NUMINAMATH_GPT_egg_rolls_total_l731_73170

def total_egg_rolls (omar_rolls : ℕ) (karen_rolls : ℕ) : ℕ :=
  omar_rolls + karen_rolls

theorem egg_rolls_total :
  total_egg_rolls 219 229 = 448 :=
by
  sorry

end NUMINAMATH_GPT_egg_rolls_total_l731_73170


namespace NUMINAMATH_GPT_polynomial_remainder_l731_73128

theorem polynomial_remainder (P : Polynomial ℝ) (H1 : P.eval 1 = 2) (H2 : P.eval 2 = 1) :
  ∃ Q : Polynomial ℝ, P = Q * (Polynomial.X - 1) * (Polynomial.X - 2) + (3 - Polynomial.X) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l731_73128


namespace NUMINAMATH_GPT_jason_additional_manager_months_l731_73112

def additional_manager_months (bartender_years manager_years total_exp_months : ℕ) : ℕ :=
  let bartender_months := bartender_years * 12
  let manager_months := manager_years * 12
  total_exp_months - (bartender_months + manager_months)

theorem jason_additional_manager_months : 
  additional_manager_months 9 3 150 = 6 := 
by 
  sorry

end NUMINAMATH_GPT_jason_additional_manager_months_l731_73112


namespace NUMINAMATH_GPT_brick_length_proof_l731_73172

-- Defining relevant parameters and conditions
def width_of_brick : ℝ := 10 -- width in cm
def height_of_brick : ℝ := 7.5 -- height in cm
def wall_length : ℝ := 26 -- length in m
def wall_width : ℝ := 2 -- width in m
def wall_height : ℝ := 0.75 -- height in m
def num_bricks : ℝ := 26000 

-- Defining known volumes for conversion
def volume_of_wall_m3 : ℝ := wall_length * wall_width * wall_height
def volume_of_wall_cm3 : ℝ := volume_of_wall_m3 * 1000000 -- converting m³ to cm³

-- Volume of one brick given the unknown length L
def volume_of_one_brick (L : ℝ) : ℝ := L * width_of_brick * height_of_brick

-- Total volume of bricks is the volume of one brick times the number of bricks
def total_volume_of_bricks (L : ℝ) : ℝ := volume_of_one_brick L * num_bricks

-- The length of the brick is found by equating the total volume of bricks to the volume of the wall
theorem brick_length_proof : ∃ L : ℝ, total_volume_of_bricks L = volume_of_wall_cm3 ∧ L = 20 :=
by
  existsi 20
  sorry

end NUMINAMATH_GPT_brick_length_proof_l731_73172


namespace NUMINAMATH_GPT_C_pow_50_l731_73109

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_pow_50 :
  C ^ 50 = !![-299, -100; 800, 249] := by
  sorry

end NUMINAMATH_GPT_C_pow_50_l731_73109


namespace NUMINAMATH_GPT_sequence_formula_l731_73175

-- Definitions of the sequence and conditions
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) a

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S a n + a n = 2 * n + 1

-- Proposition to prove
theorem sequence_formula (a : ℕ → ℝ) (h : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 - 1 / 2^n := sorry

end NUMINAMATH_GPT_sequence_formula_l731_73175


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l731_73149

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (A : ℝ) (G : ℝ)
  (hA : A = (a + b) / 2) (hG : G = Real.sqrt (a * b)) : A ≥ G :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l731_73149


namespace NUMINAMATH_GPT_quadrilateral_perimeter_l731_73134

theorem quadrilateral_perimeter (a b : ℝ) (h₁ : a = 10) (h₂ : b = 15)
  (h₃ : ∀ (ABD BCD ABC ACD : ℝ), ABD = BCD ∧ ABC = ACD) : a + a + b + b = 50 :=
by
  rw [h₁, h₂]
  linarith


end NUMINAMATH_GPT_quadrilateral_perimeter_l731_73134


namespace NUMINAMATH_GPT_points_subtracted_per_wrong_answer_l731_73197

theorem points_subtracted_per_wrong_answer 
  (total_problems : ℕ) 
  (wrong_answers : ℕ) 
  (score : ℕ) 
  (points_per_right_answer : ℕ) 
  (correct_answers : ℕ)
  (subtracted_points : ℕ) 
  (expected_points : ℕ) 
  (points_subtracted : ℕ) :
  total_problems = 25 → 
  wrong_answers = 3 → 
  score = 85 → 
  points_per_right_answer = 4 → 
  correct_answers = total_problems - wrong_answers → 
  expected_points = correct_answers * points_per_right_answer → 
  subtracted_points = expected_points - score → 
  points_subtracted = subtracted_points / wrong_answers → 
  points_subtracted = 1 := 
by
  intros;
  sorry

end NUMINAMATH_GPT_points_subtracted_per_wrong_answer_l731_73197


namespace NUMINAMATH_GPT_evaluate_expression_l731_73104

theorem evaluate_expression : -(16 / 4 * 11 - 70 + 5 * 11) = -29 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l731_73104


namespace NUMINAMATH_GPT_find_f_zero_forall_x_f_pos_solve_inequality_l731_73162

variable {f : ℝ → ℝ}

-- Conditions
axiom condition_1 : ∀ x, x > 0 → f x > 1
axiom condition_2 : ∀ x y, f (x + y) = f x * f y
axiom condition_3 : f 2 = 3

-- Questions rewritten as Lean theorems

theorem find_f_zero : f 0 = 1 := sorry

theorem forall_x_f_pos : ∀ x, f x > 0 := sorry

theorem solve_inequality : ∀ x, f (7 + 2 * x) > 9 ↔ x > -3 / 2 := sorry

end NUMINAMATH_GPT_find_f_zero_forall_x_f_pos_solve_inequality_l731_73162


namespace NUMINAMATH_GPT_tan_of_acute_angle_and_cos_pi_add_alpha_l731_73173

theorem tan_of_acute_angle_and_cos_pi_add_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2)
  (h2 : Real.cos (π + α) = -Real.sqrt (3) / 2) : 
  Real.tan α = Real.sqrt (3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_of_acute_angle_and_cos_pi_add_alpha_l731_73173


namespace NUMINAMATH_GPT_ellipse_ratio_squared_l731_73126

theorem ellipse_ratio_squared (a b c : ℝ) 
    (h1 : b / a = a / c) 
    (h2 : c^2 = a^2 - b^2) : (b / a)^2 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_ratio_squared_l731_73126


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l731_73117

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ) -- define the arithmetic sequence
  (h_arith : ∀ n : ℕ, a n = a 0 + n * 4) -- condition of arithmetic sequence
  (h_a5 : a 4 = 8) -- given a_5 = 8
  (h_a9 : a 8 = 24) -- given a_9 = 24
  : 4 = 4 := -- statement to be proven
by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l731_73117


namespace NUMINAMATH_GPT_find_a_l731_73106

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2 : ℝ) * a * x^3 - (3 / 2 : ℝ) * x^2 + (3 / 2 : ℝ) * a^2 * x

theorem find_a (a : ℝ) (h_max : ∀ x : ℝ, f a x ≤ f a 1) : a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_l731_73106


namespace NUMINAMATH_GPT_prime_quadratic_root_range_l731_73116

theorem prime_quadratic_root_range (p : ℕ) (hprime : Prime p) 
  (hroots : ∃ x1 x2 : ℤ, x1 * x2 = -580 * p ∧ x1 + x2 = p) : 20 < p ∧ p < 30 :=
by
  sorry

end NUMINAMATH_GPT_prime_quadratic_root_range_l731_73116


namespace NUMINAMATH_GPT_math_problem_l731_73196

-- Definitions for increasing function and periodic function
def increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x ≤ f y
def periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x : ℝ, f (x + T) = f x

-- The main theorem statement
theorem math_problem (f g h : ℝ → ℝ) (T : ℝ) :
  (∀ x y : ℝ, x < y → f x + g x ≤ f y + g y) ∧ (∀ x y : ℝ, x < y → f x + h x ≤ f y + h y) ∧ (∀ x y : ℝ, x < y → g x + h x ≤ g y + h y) → 
  ¬(increasing g) ∧
  (∀ x : ℝ, f (x + T) + g (x + T) = f x + g x ∧ f (x + T) + h (x + T) = f x + h x ∧ g (x + T) + h (x + T) = g x + h x) → 
  increasing f ∧ increasing g ∧ increasing h :=
sorry

end NUMINAMATH_GPT_math_problem_l731_73196


namespace NUMINAMATH_GPT_second_cyclist_speed_l731_73166

-- Definitions of the given conditions
def total_course_length : ℝ := 45
def first_cyclist_speed : ℝ := 14
def meeting_time : ℝ := 1.5

-- Lean 4 statement for the proof problem
theorem second_cyclist_speed : 
  ∃ v : ℝ, first_cyclist_speed * meeting_time + v * meeting_time = total_course_length → v = 16 := 
by 
  sorry

end NUMINAMATH_GPT_second_cyclist_speed_l731_73166


namespace NUMINAMATH_GPT_horse_saddle_ratio_l731_73152

variable (H S : ℝ)
variable (m : ℝ)
variable (total_value saddle_value : ℝ)

theorem horse_saddle_ratio :
  total_value = 100 ∧ saddle_value = 12.5 ∧ H = m * saddle_value ∧ H + saddle_value = total_value → m = 7 :=
by
  sorry

end NUMINAMATH_GPT_horse_saddle_ratio_l731_73152


namespace NUMINAMATH_GPT_desserts_brought_by_mom_l731_73127

-- Definitions for the number of each type of dessert
def num_coconut := 1
def num_meringues := 2
def num_caramel := 7

-- Conditions from the problem as definitions
def total_desserts := num_coconut + num_meringues + num_caramel = 10
def fewer_coconut_than_meringues := num_coconut < num_meringues
def most_caramel := num_caramel > num_meringues
def josef_jakub_condition := (num_coconut + num_meringues + num_caramel) - (4 * 2) = 1

-- We need to prove the answer based on these conditions
theorem desserts_brought_by_mom :
  total_desserts ∧ fewer_coconut_than_meringues ∧ most_caramel ∧ josef_jakub_condition → 
  num_coconut = 1 ∧ num_meringues = 2 ∧ num_caramel = 7 :=
by sorry

end NUMINAMATH_GPT_desserts_brought_by_mom_l731_73127


namespace NUMINAMATH_GPT_evaluate_expression_l731_73114

theorem evaluate_expression (x : Int) (h : x = -2023) : abs (abs (abs x - x) + abs x) + x = 4046 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l731_73114


namespace NUMINAMATH_GPT_multiple_of_spending_on_wednesday_l731_73190

-- Definitions based on the conditions
def monday_spending : ℤ := 60
def tuesday_spending : ℤ := 4 * monday_spending
def total_spending : ℤ := 600

-- Problem to prove
theorem multiple_of_spending_on_wednesday (x : ℤ) : 
  monday_spending + tuesday_spending + x * monday_spending = total_spending → 
  x = 5 := by
  sorry

end NUMINAMATH_GPT_multiple_of_spending_on_wednesday_l731_73190


namespace NUMINAMATH_GPT_monotonic_increasing_iff_monotonic_decreasing_on_interval_l731_73108

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 - a * x - 1

theorem monotonic_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ a ≤ 0 :=
by 
  sorry

theorem monotonic_decreasing_on_interval (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → ∀ y : ℝ, -1 < y ∧ y < 1 → x < y → f y a < f x a) ↔ 3 ≤ a :=
by 
  sorry

end NUMINAMATH_GPT_monotonic_increasing_iff_monotonic_decreasing_on_interval_l731_73108


namespace NUMINAMATH_GPT_prob_two_red_balls_l731_73140

-- Define the initial conditions for the balls in the bag
def red_balls : ℕ := 5
def blue_balls : ℕ := 6
def green_balls : ℕ := 2
def total_balls : ℕ := red_balls + blue_balls + green_balls

-- Define the probability of picking a red ball first
def prob_red1 : ℚ := red_balls / total_balls

-- Define the remaining number of balls and the probability of picking a red ball second
def remaining_red_balls : ℕ := red_balls - 1
def remaining_total_balls : ℕ := total_balls - 1
def prob_red2 : ℚ := remaining_red_balls / remaining_total_balls

-- Define the combined probability of both events
def prob_both_red : ℚ := prob_red1 * prob_red2

-- Statement of the theorem to be proved
theorem prob_two_red_balls : prob_both_red = 5 / 39 := by
  sorry

end NUMINAMATH_GPT_prob_two_red_balls_l731_73140


namespace NUMINAMATH_GPT_cos_five_pi_over_three_l731_73105

theorem cos_five_pi_over_three : Real.cos (5 * Real.pi / 3) = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_cos_five_pi_over_three_l731_73105


namespace NUMINAMATH_GPT_am_gm_inequality_l731_73122

theorem am_gm_inequality (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) : 
  8 * a * b * c ≤ (a + b) * (b + c) * (c + a) := 
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l731_73122


namespace NUMINAMATH_GPT_sum_of_max_marks_l731_73165

theorem sum_of_max_marks :
  ∀ (M S E : ℝ),
  (30 / 100 * M = 180) ∧
  (50 / 100 * S = 200) ∧
  (40 / 100 * E = 120) →
  M + S + E = 1300 :=
by
  intros M S E h
  sorry

end NUMINAMATH_GPT_sum_of_max_marks_l731_73165


namespace NUMINAMATH_GPT_prize_winner_is_B_l731_73154

-- Define the possible entries winning the prize
inductive Prize
| A
| B
| C
| D

open Prize

-- Define each student's predictions
def A_pred (prize : Prize) : Prop := prize = C ∨ prize = D
def B_pred (prize : Prize) : Prop := prize = B
def C_pred (prize : Prize) : Prop := prize ≠ A ∧ prize ≠ D
def D_pred (prize : Prize) : Prop := prize = C

-- Define the main theorem to prove
theorem prize_winner_is_B (prize : Prize) :
  (A_pred prize ∧ B_pred prize ∧ ¬C_pred prize ∧ ¬D_pred prize) ∨
  (A_pred prize ∧ ¬B_pred prize ∧ C_pred prize ∧ ¬D_pred prize) ∨
  (¬A_pred prize ∧ B_pred prize ∧ C_pred prize ∧ ¬D_pred prize) ∨
  (¬A_pred prize ∧ ¬B_pred prize ∧ C_pred prize ∧ D_pred prize) →
  prize = B :=
sorry

end NUMINAMATH_GPT_prize_winner_is_B_l731_73154


namespace NUMINAMATH_GPT_eval_imaginary_expression_l731_73182

theorem eval_imaginary_expression :
  ∀ (i : ℂ), i^2 = -1 → i^2022 + i^2023 + i^2024 + i^2025 = 0 :=
by
  sorry

end NUMINAMATH_GPT_eval_imaginary_expression_l731_73182


namespace NUMINAMATH_GPT_baseball_game_earnings_l731_73111

theorem baseball_game_earnings
  (S : ℝ) (W : ℝ)
  (h1 : S = 2662.50)
  (h2 : W + S = 5182.50) :
  S - W = 142.50 :=
by
  sorry

end NUMINAMATH_GPT_baseball_game_earnings_l731_73111


namespace NUMINAMATH_GPT_linear_dependency_k_val_l731_73180

theorem linear_dependency_k_val (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 2 * c1 + 4 * c2 = 0 ∧ 3 * c1 + k * c2 = 0) ↔ k = 6 :=
by sorry

end NUMINAMATH_GPT_linear_dependency_k_val_l731_73180


namespace NUMINAMATH_GPT_equation_c_is_linear_l731_73187

-- Define the condition for being a linear equation with one variable
def is_linear_equation_with_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x + b = 0)

-- The given equation to check is (x - 1) / 2 = 1, which simplifies to x = 3
def equation_c (x : ℝ) : Prop := (x - 1) / 2 = 1

-- Prove that the given equation is a linear equation with one variable
theorem equation_c_is_linear :
  is_linear_equation_with_one_variable equation_c :=
sorry

end NUMINAMATH_GPT_equation_c_is_linear_l731_73187


namespace NUMINAMATH_GPT_geometric_seq_a7_l731_73125

theorem geometric_seq_a7 (a : ℕ → ℝ) (r : ℝ) (h1 : a 3 = 16) (h2 : a 5 = 4) (h_geom : ∀ n, a (n + 1) = a n * r) : a 7 = 1 := by
  sorry

end NUMINAMATH_GPT_geometric_seq_a7_l731_73125


namespace NUMINAMATH_GPT_avg_difference_even_avg_difference_odd_l731_73189

noncomputable def avg (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

def even_ints_20_to_60 := [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
def even_ints_10_to_140 := [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140]

def odd_ints_21_to_59 := [21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
def odd_ints_11_to_139 := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139]

theorem avg_difference_even :
  avg even_ints_20_to_60 - avg even_ints_10_to_140 = -35 := sorry

theorem avg_difference_odd :
  avg odd_ints_21_to_59 - avg odd_ints_11_to_139 = -35 := sorry

end NUMINAMATH_GPT_avg_difference_even_avg_difference_odd_l731_73189


namespace NUMINAMATH_GPT_quadratic_function_coefficient_nonzero_l731_73177

theorem quadratic_function_coefficient_nonzero (m : ℝ) :
  (y = (m + 2) * x * x + m) ↔ (m ≠ -2 ∧ (m^2 + m - 2 = 0) → m = 1) := by
  sorry

end NUMINAMATH_GPT_quadratic_function_coefficient_nonzero_l731_73177


namespace NUMINAMATH_GPT_principal_amount_l731_73145

theorem principal_amount (P R : ℝ) : 
  (P + P * R * 2 / 100 = 850) ∧ (P + P * R * 7 / 100 = 1020) → P = 782 :=
by
  sorry

end NUMINAMATH_GPT_principal_amount_l731_73145


namespace NUMINAMATH_GPT_remainder_when_divided_by_30_l731_73146

theorem remainder_when_divided_by_30 (x : ℤ) : 
  (4 + x) % 8 = 9 % 8 ∧
  (6 + x) % 27 = 4 % 27 ∧
  (8 + x) % 125 = 49 % 125 
  → x % 30 = 1 % 30 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_30_l731_73146


namespace NUMINAMATH_GPT_volume_hemisphere_from_sphere_l731_73185

theorem volume_hemisphere_from_sphere (r : ℝ) (V_sphere : ℝ) (V_hemisphere : ℝ) 
  (h1 : V_sphere = 150 * Real.pi) 
  (h2 : V_sphere = (4 / 3) * Real.pi * r^3) : 
  V_hemisphere = 75 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_hemisphere_from_sphere_l731_73185


namespace NUMINAMATH_GPT_athena_total_spent_l731_73156

noncomputable def cost_sandwiches := 4 * 3.25
noncomputable def cost_fruit_drinks := 3 * 2.75
noncomputable def cost_cookies := 6 * 1.50
noncomputable def cost_chips := 2 * 1.85

noncomputable def total_cost := cost_sandwiches + cost_fruit_drinks + cost_cookies + cost_chips

theorem athena_total_spent : total_cost = 33.95 := 
by 
  simp [cost_sandwiches, cost_fruit_drinks, cost_cookies, cost_chips, total_cost]
  sorry

end NUMINAMATH_GPT_athena_total_spent_l731_73156


namespace NUMINAMATH_GPT_find_a_l731_73161

theorem find_a (a : ℝ) (x : ℝ) (h : ∀ (x : ℝ), 2 * x - a ≤ -1 ↔ x ≤ 1) : a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_l731_73161


namespace NUMINAMATH_GPT_smallest_k_l731_73193

theorem smallest_k (k : ℕ) (h₁ : k > 1) (h₂ : k % 17 = 1) (h₃ : k % 6 = 1) (h₄ : k % 2 = 1) : k = 103 :=
by sorry

end NUMINAMATH_GPT_smallest_k_l731_73193


namespace NUMINAMATH_GPT_Cheerful_snakes_not_Green_l731_73160

variables {Snake : Type} (snakes : Finset Snake)
variable (Cheerful Green CanSing CanMultiply : Snake → Prop)

-- Conditions
axiom Cheerful_impl_CanSing : ∀ s, Cheerful s → CanSing s
axiom Green_impl_not_CanMultiply : ∀ s, Green s → ¬ CanMultiply s
axiom not_CanMultiply_impl_not_CanSing : ∀ s, ¬ CanMultiply s → ¬ CanSing s

-- Question
theorem Cheerful_snakes_not_Green : ∀ s, Cheerful s → ¬ Green s :=
by sorry

end NUMINAMATH_GPT_Cheerful_snakes_not_Green_l731_73160


namespace NUMINAMATH_GPT_inequality_condition_l731_73191

noncomputable def inequality_holds_for_all (a b c : ℝ) : Prop :=
  ∀ (x : ℝ), a * Real.sin x + b * Real.cos x + c > 0

theorem inequality_condition (a b c : ℝ) :
  inequality_holds_for_all a b c ↔ Real.sqrt (a^2 + b^2) < c :=
by sorry

end NUMINAMATH_GPT_inequality_condition_l731_73191


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l731_73142

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = -4) : 
  q = -2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l731_73142


namespace NUMINAMATH_GPT_correct_answers_count_l731_73129

theorem correct_answers_count (total_questions correct_pts incorrect_pts final_score : ℤ)
  (h1 : total_questions = 26)
  (h2 : correct_pts = 8)
  (h3 : incorrect_pts = -5)
  (h4 : final_score = 0) :
  ∃ c i : ℤ, c + i = total_questions ∧ correct_pts * c + incorrect_pts * i = final_score ∧ c = 10 :=
by
  use 10, (26 - 10)
  simp
  sorry

end NUMINAMATH_GPT_correct_answers_count_l731_73129


namespace NUMINAMATH_GPT_list_price_of_article_l731_73167

theorem list_price_of_article 
(paid_price : ℝ) 
(first_discount second_discount : ℝ)
(list_price : ℝ)
(h_paid_price : paid_price = 59.22)
(h_first_discount : first_discount = 0.10)
(h_second_discount : second_discount = 0.06000000000000002)
(h_final_price : paid_price = (1 - first_discount) * (1 - second_discount) * list_price) :
  list_price = 70 := 
by
  sorry

end NUMINAMATH_GPT_list_price_of_article_l731_73167


namespace NUMINAMATH_GPT_simplify_expression_l731_73101

theorem simplify_expression (x : ℝ) (h : x = 1) : (x - 1)^2 + (x + 1) * (x - 1) - 2 * x^2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l731_73101


namespace NUMINAMATH_GPT_opposite_of_2_is_minus_2_l731_73183

-- Define the opposite function
def opposite (x : ℤ) : ℤ := -x

-- Assert the theorem to prove that the opposite of 2 is -2
theorem opposite_of_2_is_minus_2 : opposite 2 = -2 := by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_opposite_of_2_is_minus_2_l731_73183


namespace NUMINAMATH_GPT_other_number_is_twelve_l731_73107

variable (x certain_number : ℕ)
variable (h1: certain_number = 60)
variable (h2: certain_number = 5 * x)

theorem other_number_is_twelve :
  x = 12 :=
by
  sorry

end NUMINAMATH_GPT_other_number_is_twelve_l731_73107


namespace NUMINAMATH_GPT_single_cone_scoops_l731_73135

theorem single_cone_scoops (banana_split_scoops : ℕ) (waffle_bowl_scoops : ℕ) (single_cone_scoops : ℕ) (double_cone_scoops : ℕ)
  (h1 : banana_split_scoops = 3 * single_cone_scoops)
  (h2 : waffle_bowl_scoops = banana_split_scoops + 1)
  (h3 : double_cone_scoops = 2 * single_cone_scoops)
  (h4 : single_cone_scoops + banana_split_scoops + waffle_bowl_scoops + double_cone_scoops = 10) :
  single_cone_scoops = 1 :=
by
  sorry

end NUMINAMATH_GPT_single_cone_scoops_l731_73135


namespace NUMINAMATH_GPT_prime_ge_5_divisible_by_12_l731_73147

theorem prime_ge_5_divisible_by_12 (p : ℕ) (hp1 : p ≥ 5) (hp2 : Nat.Prime p) : 12 ∣ p^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_prime_ge_5_divisible_by_12_l731_73147


namespace NUMINAMATH_GPT_number_of_solutions_l731_73144

theorem number_of_solutions :
  ∃ sols: Finset (ℕ × ℕ), (∀ (x y : ℕ), (x, y) ∈ sols ↔ x^2 + y^2 + 2*x*y - 1988*x - 1988*y = 1989 ∧ x > 0 ∧ y > 0)
  ∧ sols.card = 1988 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l731_73144


namespace NUMINAMATH_GPT_inequality_nonneg_ab_l731_73120

theorem inequality_nonneg_ab (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) :
  (1 + a)^4 * (1 + b)^4 ≥ 64 * a * b * (a + b)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_nonneg_ab_l731_73120


namespace NUMINAMATH_GPT_max_books_borrowed_l731_73158

theorem max_books_borrowed (students_total : ℕ) (students_no_books : ℕ) 
  (students_1_book : ℕ) (students_2_books : ℕ) (students_at_least_3_books : ℕ) 
  (average_books_per_student : ℝ) (H1 : students_total = 60) 
  (H2 : students_no_books = 4) 
  (H3 : students_1_book = 18) 
  (H4 : students_2_books = 20) 
  (H5 : students_at_least_3_books = students_total - (students_no_books + students_1_book + students_2_books)) 
  (H6 : average_books_per_student = 2.5) : 
  ∃ max_books : ℕ, max_books = 41 :=
by
  sorry

end NUMINAMATH_GPT_max_books_borrowed_l731_73158


namespace NUMINAMATH_GPT_sum_of_remainders_l731_73123

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l731_73123


namespace NUMINAMATH_GPT_islander_C_response_l731_73157

-- Define the types and assumptions
variables {Person : Type} (is_knight : Person → Prop) (is_liar : Person → Prop)
variables (A B C : Person)

-- Conditions from the problem
axiom A_statement : (is_liar A) ↔ (is_knight B = false ∧ is_knight C = false)
axiom B_statement : (is_knight B) ↔ (is_knight A ↔ ¬ is_knight C)

-- Conclusion we want to prove
theorem islander_C_response : is_knight C → (is_knight A ↔ ¬ is_knight C) := sorry

end NUMINAMATH_GPT_islander_C_response_l731_73157


namespace NUMINAMATH_GPT_shift_parabola_l731_73133

theorem shift_parabola (x : ℝ) : 
  let y := -x^2
  let y_shifted_left := -((x + 3)^2)
  let y_shifted := y_shifted_left + 5
  y_shifted = -(x + 3)^2 + 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_shift_parabola_l731_73133


namespace NUMINAMATH_GPT_both_firms_participate_l731_73171

-- Definitions based on the conditions
variable (V IC : ℝ) (α : ℝ)
-- Assumptions
variable (hα : 0 < α ∧ α < 1)
-- Part (a) condition transformation
def participation_condition := α * (1 - α) * V + 0.5 * α^2 * V ≥ IC

-- Given values for part (b)
def V_value : ℝ := 24
def α_value : ℝ := 0.5
def IC_value : ℝ := 7

-- New definitions for given values
def part_b_condition := (α_value * (1 - α_value) * V_value + 0.5 * α_value^2 * V_value) ≥ IC_value

-- Profits for part (c) comparison
def profit_when_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
def profit_when_one := α * V - IC

-- Proof problem statement in Lean 4
theorem both_firms_participate (hV : V = 24) (hα : α = 0.5) (hIC : IC = 7) :
    (α * (1 - α) * V + 0.5 * α^2 * V) ≥ IC ∧ profit_when_both V alpha IC > profit_when_one V α IC := by
  sorry

end NUMINAMATH_GPT_both_firms_participate_l731_73171


namespace NUMINAMATH_GPT_probability_at_least_eight_stayed_correct_l731_73130

noncomputable def probability_at_least_eight_stayed (n : ℕ) (c : ℕ) (p : ℚ) : ℚ :=
  let certain_count := c
  let unsure_count := n - c
  let k := 3
  let prob_eight := 
    (Nat.choose unsure_count k : ℚ) * (p^k) * ((1 - p)^(unsure_count - k))
  let prob_nine := p^unsure_count
  prob_eight + prob_nine

theorem probability_at_least_eight_stayed_correct :
  probability_at_least_eight_stayed 9 5 (3/7) = 513 / 2401 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_eight_stayed_correct_l731_73130


namespace NUMINAMATH_GPT_variance_of_scores_l731_73148

-- Define the list of scores
def scores : List ℕ := [110, 114, 121, 119, 126]

-- Define the formula for variance calculation
def variance (l : List ℕ) : ℚ :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  (l.map (λ x => ((x : ℚ) - mean) ^ 2)).sum / n

-- The main theorem to be proved
theorem variance_of_scores :
  variance scores = 30.8 := 
  by
    sorry

end NUMINAMATH_GPT_variance_of_scores_l731_73148


namespace NUMINAMATH_GPT_solve_equation_one_solve_equation_two_l731_73138

theorem solve_equation_one (x : ℝ) : (x - 3) ^ 2 - 4 = 0 ↔ x = 5 ∨ x = 1 := sorry

theorem solve_equation_two (x : ℝ) : (x + 2) ^ 2 - 2 * (x + 2) = 3 ↔ x = 1 ∨ x = -1 := sorry

end NUMINAMATH_GPT_solve_equation_one_solve_equation_two_l731_73138


namespace NUMINAMATH_GPT_simplify_expression_l731_73151

theorem simplify_expression (x y : ℝ) :
  (2 * x^3 * y^2 - 3 * x^2 * y^3) / (1 / 2 * x * y)^2 = 8 * x - 12 * y := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l731_73151


namespace NUMINAMATH_GPT_point_value_of_other_questions_l731_73141

theorem point_value_of_other_questions (x y p : ℕ) 
  (h1 : x = 10) 
  (h2 : x + y = 40) 
  (h3 : 40 + 30 * p = 100) : 
  p = 2 := 
  sorry

end NUMINAMATH_GPT_point_value_of_other_questions_l731_73141


namespace NUMINAMATH_GPT_resulting_total_mass_l731_73192

-- Define initial conditions
def initial_total_mass : ℝ := 12
def initial_white_paint_mass : ℝ := 0.8 * initial_total_mass
def initial_black_paint_mass : ℝ := initial_total_mass - initial_white_paint_mass

-- Required condition for the new mixture
def final_white_paint_percentage : ℝ := 0.9

-- Prove that the resulting total mass of paint is 24 kg
theorem resulting_total_mass (x : ℝ) (h1 : initial_total_mass = 12) 
                            (h2 : initial_white_paint_mass = 0.8 * initial_total_mass)
                            (h3 : initial_black_paint_mass = initial_total_mass - initial_white_paint_mass)
                            (h4 : final_white_paint_percentage = 0.9) 
                            (h5 : (initial_white_paint_mass + x) / (initial_total_mass + x) = final_white_paint_percentage) : 
                            initial_total_mass + x = 24 :=
by 
  -- Temporarily assume the proof without detailing the solution steps
  sorry

end NUMINAMATH_GPT_resulting_total_mass_l731_73192


namespace NUMINAMATH_GPT_six_digit_palindromes_count_l731_73188

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end NUMINAMATH_GPT_six_digit_palindromes_count_l731_73188


namespace NUMINAMATH_GPT_count_ways_to_sum_2020_as_1s_and_2s_l731_73198

theorem count_ways_to_sum_2020_as_1s_and_2s : ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 2020 → x + y = n) → n = 102 :=
by
-- Mathematics proof needed.
sorry

end NUMINAMATH_GPT_count_ways_to_sum_2020_as_1s_and_2s_l731_73198


namespace NUMINAMATH_GPT_topics_assignment_l731_73163

theorem topics_assignment (students groups arrangements : ℕ) (h1 : students = 6) (h2 : groups = 3) (h3 : arrangements = 90) :
  let T := arrangements / (students * (students - 1) / 2 * (4 * 3 / 2 * 1))
  T = 1 :=
by
  sorry

end NUMINAMATH_GPT_topics_assignment_l731_73163


namespace NUMINAMATH_GPT_magician_earnings_at_least_l731_73113

def magician_starting_decks := 15
def magician_remaining_decks := 3
def decks_sold := magician_starting_decks - magician_remaining_decks
def standard_price_per_deck := 3
def discount := 1
def discounted_price_per_deck := standard_price_per_deck - discount
def min_earnings := decks_sold * discounted_price_per_deck

theorem magician_earnings_at_least :
  min_earnings ≥ 24 :=
by sorry

end NUMINAMATH_GPT_magician_earnings_at_least_l731_73113


namespace NUMINAMATH_GPT_max_xy_l731_73176

theorem max_xy (x y : ℝ) (hxy_pos : x > 0 ∧ y > 0) (h : 5 * x + 8 * y = 65) : 
  xy ≤ 25 :=
by
  sorry

end NUMINAMATH_GPT_max_xy_l731_73176


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l731_73115

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (∃ m0 : ℝ, m0 > 0 ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m0 x1 ≤ f m0 x2)) ∧ 
  ¬ (∀ m : ℝ, (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m x1 ≤ f m x2) → m > 0) :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l731_73115


namespace NUMINAMATH_GPT_ernie_circles_l731_73143

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes ali_circles : ℕ)
  (h1: boxes_per_circle_ali = 8)
  (h2: boxes_per_circle_ernie = 10)
  (h3: total_boxes = 80)
  (h4: ali_circles = 5) : 
  (total_boxes - ali_circles * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end NUMINAMATH_GPT_ernie_circles_l731_73143


namespace NUMINAMATH_GPT_find_dolls_l731_73137

namespace DollsProblem

variables (S D : ℕ) -- Define S and D as natural numbers

-- Conditions as per the problem
def cond1 : Prop := 4 * S + 3 = D
def cond2 : Prop := 5 * S = D + 6

-- Theorem stating the problem
theorem find_dolls (h1 : cond1 S D) (h2 : cond2 S D) : D = 39 :=
by
  sorry

end DollsProblem

end NUMINAMATH_GPT_find_dolls_l731_73137


namespace NUMINAMATH_GPT_triangle_shape_l731_73159

theorem triangle_shape (a b c : ℝ) (h : a^4 - b^4 + (b^2 * c^2 - a^2 * c^2) = 0) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end NUMINAMATH_GPT_triangle_shape_l731_73159


namespace NUMINAMATH_GPT_population_increase_20th_century_l731_73181

theorem population_increase_20th_century (P : ℕ) :
  let population_mid_century := 3 * P
  let population_end_century := 12 * P
  (population_end_century - P) / P * 100 = 1100 :=
by
  sorry

end NUMINAMATH_GPT_population_increase_20th_century_l731_73181


namespace NUMINAMATH_GPT_g_max_value_l731_73131

def g (n : ℕ) : ℕ :=
if n < 15 then n + 15 else g (n - 7)

theorem g_max_value : ∃ N : ℕ, ∀ n : ℕ, g n ≤ N ∧ N = 29 := 
by 
  sorry

end NUMINAMATH_GPT_g_max_value_l731_73131


namespace NUMINAMATH_GPT_smallest_a_l731_73110

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem smallest_a (a : ℕ) (h1 : 5880 = 2^3 * 3^1 * 5^1 * 7^2)
                    (h2 : ∀ b : ℕ, b < a → ¬ is_perfect_square (5880 * b))
                    : a = 15 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_l731_73110


namespace NUMINAMATH_GPT_fraction_identity_l731_73103

theorem fraction_identity (a b : ℝ) (h : a / b = 3 / 4) : a / (a + b) = 3 / 7 := 
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l731_73103


namespace NUMINAMATH_GPT_denomination_other_currency_notes_l731_73119

noncomputable def denomination_proof : Prop :=
  ∃ D x y : ℕ, 
  (x + y = 85) ∧
  (100 * x + D * y = 5000) ∧
  (D * y = 3500) ∧
  (D = 50)

theorem denomination_other_currency_notes :
  denomination_proof :=
sorry

end NUMINAMATH_GPT_denomination_other_currency_notes_l731_73119


namespace NUMINAMATH_GPT_problem1_problem2_l731_73178

variable (α : ℝ) (tan_alpha_eq_one_over_three : Real.tan α = 1 / 3)

-- For the first proof problem
theorem problem1 : (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by sorry

-- For the second proof problem
theorem problem2 : Real.cos α ^ 2 - Real.sin (2 * α) = 3 / 10 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l731_73178


namespace NUMINAMATH_GPT_inequality_condition_necessary_not_sufficient_l731_73136

theorem inequality_condition (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  (1 / a > 1 / b) :=
by
  sorry

theorem necessary_not_sufficient (a b : ℝ) :
  (1 / a > 1 / b → 0 < a ∧ a < b) ∧ ¬ (0 < a ∧ a < b → 1 / a > 1 / b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_condition_necessary_not_sufficient_l731_73136


namespace NUMINAMATH_GPT_gcd_of_45_135_225_is_45_l731_73194

theorem gcd_of_45_135_225_is_45 : Nat.gcd (Nat.gcd 45 135) 225 = 45 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_45_135_225_is_45_l731_73194


namespace NUMINAMATH_GPT_solve_quadratic_l731_73155

-- Problem Definition
def quadratic_equation (x : ℝ) : Prop :=
  2 * x^2 - 6 * x + 3 = 0

-- Solution Definition
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2

-- Lean Theorem Statement
theorem solve_quadratic : ∀ x : ℝ, quadratic_equation x ↔ solution1 x :=
sorry

end NUMINAMATH_GPT_solve_quadratic_l731_73155


namespace NUMINAMATH_GPT_equal_distribution_l731_73121

theorem equal_distribution (k : ℤ) : ∃ n : ℤ, n = 81 + 95 * k ∧ ∃ b : ℤ, (19 + 6 * n) = 95 * b :=
by
  -- to be proved
  sorry

end NUMINAMATH_GPT_equal_distribution_l731_73121


namespace NUMINAMATH_GPT_find_d_k_l731_73199

open Matrix

noncomputable def matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![6, d]]

noncomputable def inv_matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let detA := 3 * d - 24
  (1 / detA) • ![![d, -4], ![-6, 3]]

theorem find_d_k (d k : ℝ) (h : inv_matrix_A d = k • matrix_A d) :
    (d, k) = (-3, 1/33) := by
  sorry

end NUMINAMATH_GPT_find_d_k_l731_73199


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l731_73124

noncomputable def f (b a : ℝ) (x : ℝ) := (b - 2^x) / (2^x + a) 

-- (1) Prove values of a and b
theorem problem1 (a b : ℝ) : 
  (f b a 0 = 0) ∧ (f b a (-1) = -f b a 1) → (a = 1 ∧ b = 1) :=
sorry

-- (2) Prove f is decreasing function
theorem problem2 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f b a x₁ - f b a x₂ > 0 :=
sorry

-- (3) Find range of k such that inequality always holds
theorem problem3 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) (k : ℝ) : 
  (∀ t : ℝ, f b a (t^2 - 2*t) + f b a (2*t^2 - k) < 0) → k < -(1/3) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l731_73124


namespace NUMINAMATH_GPT_cost_of_4_stamps_l731_73100

theorem cost_of_4_stamps (cost_per_stamp : ℕ) (h : cost_per_stamp = 34) : 4 * cost_per_stamp = 136 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_4_stamps_l731_73100


namespace NUMINAMATH_GPT_oranges_in_bin_after_changes_l731_73174

-- Define the initial number of oranges
def initial_oranges : ℕ := 34

-- Define the number of oranges thrown away
def oranges_thrown_away : ℕ := 20

-- Define the number of new oranges added
def new_oranges_added : ℕ := 13

-- Theorem statement to prove the final number of oranges in the bin
theorem oranges_in_bin_after_changes :
  initial_oranges - oranges_thrown_away + new_oranges_added = 27 := by
  sorry

end NUMINAMATH_GPT_oranges_in_bin_after_changes_l731_73174


namespace NUMINAMATH_GPT_negation_of_divisible_by_2_even_l731_73168

theorem negation_of_divisible_by_2_even :
  (¬ ∀ n : ℤ, (∃ k, n = 2 * k) → (∃ k, n = 2 * k ∧ n % 2 = 0)) ↔
  ∃ n : ℤ, (∃ k, n = 2 * k) ∧ ¬ (n % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_divisible_by_2_even_l731_73168


namespace NUMINAMATH_GPT_arithmetic_progression_sum_l731_73150

noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_sum
    (a1 d : ℤ)
    (h : a 9 a1 d = a 12 a1 d / 2 + 3) :
  S 11 a1 d = 66 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_l731_73150


namespace NUMINAMATH_GPT_rectangle_diagonal_length_l731_73169

theorem rectangle_diagonal_length (L W : ℝ) (h1 : L * W = 20) (h2 : L + W = 9) :
  (L^2 + W^2) = 41 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_length_l731_73169


namespace NUMINAMATH_GPT_compute_pqr_l731_73132

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 30) 
  (h_equation : 1 / p + 1 / q + 1 / r + 420 / (p * q * r) = 1) : 
  p * q * r = 576 :=
sorry

end NUMINAMATH_GPT_compute_pqr_l731_73132


namespace NUMINAMATH_GPT_fred_gave_cards_l731_73102

theorem fred_gave_cards (initial_cards : ℕ) (torn_cards : ℕ) 
  (bought_cards : ℕ) (total_cards : ℕ) (fred_cards : ℕ) : 
  initial_cards = 18 → torn_cards = 8 → bought_cards = 40 → total_cards = 84 →
  fred_cards = total_cards - (initial_cards - torn_cards + bought_cards) →
  fred_cards = 34 :=
by
  intros h_initial h_torn h_bought h_total h_fred
  sorry

end NUMINAMATH_GPT_fred_gave_cards_l731_73102


namespace NUMINAMATH_GPT_domain_of_function_l731_73195

theorem domain_of_function :
  {x : ℝ | x ≥ -1} \ {0} = {x : ℝ | (x ≥ -1 ∧ x < 0) ∨ x > 0} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l731_73195


namespace NUMINAMATH_GPT_total_profit_calculation_l731_73118

-- Definitions based on conditions
def initial_investment_A := 5000
def initial_investment_B := 8000
def initial_investment_C := 9000
def initial_investment_D := 7000

def investment_A_after_4_months := initial_investment_A + 2000
def investment_B_after_4_months := initial_investment_B - 1000

def investment_C_after_6_months := initial_investment_C + 3000
def investment_D_after_6_months := initial_investment_D + 5000

def profit_A_percentage := 20
def profit_B_percentage := 30
def profit_C_percentage := 25
def profit_D_percentage := 25

def profit_C := 60000

-- Total profit is what we need to determine
def total_profit := 240000

-- The proof statement
theorem total_profit_calculation :
  total_profit = (profit_C * 100) / profit_C_percentage := 
by 
  sorry

end NUMINAMATH_GPT_total_profit_calculation_l731_73118


namespace NUMINAMATH_GPT_red_tile_probability_l731_73186

def is_red_tile (n : ℕ) : Prop := n % 7 = 3

noncomputable def red_tiles_count : ℕ :=
  Nat.card {n : ℕ | n ≤ 70 ∧ is_red_tile n}

noncomputable def total_tiles_count : ℕ := 70

theorem red_tile_probability :
  (red_tiles_count : ℤ) / (total_tiles_count : ℤ) = (1 : ℤ) / 7 :=
sorry

end NUMINAMATH_GPT_red_tile_probability_l731_73186


namespace NUMINAMATH_GPT_baker_additional_cakes_l731_73179

theorem baker_additional_cakes (X : ℕ) : 
  (62 + X) - 144 = 67 → X = 149 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_baker_additional_cakes_l731_73179
