import Mathlib

namespace NUMINAMATH_GPT_sqrt_nat_or_irrational_l477_47765

theorem sqrt_nat_or_irrational {n : ℕ} : 
  (∃ m : ℕ, m^2 = n) ∨ (¬ ∃ q r : ℕ, r ≠ 0 ∧ (q^2 = n * r^2 ∧ r * r ≠ n * n)) :=
sorry

end NUMINAMATH_GPT_sqrt_nat_or_irrational_l477_47765


namespace NUMINAMATH_GPT_plum_cost_l477_47789

theorem plum_cost
  (total_fruits : ℕ)
  (total_cost : ℕ)
  (peach_cost : ℕ)
  (plums_bought : ℕ)
  (peaches_bought : ℕ)
  (P : ℕ) :
  total_fruits = 32 →
  total_cost = 52 →
  peach_cost = 1 →
  plums_bought = 20 →
  peaches_bought = total_fruits - plums_bought →
  total_cost = 20 * P + peaches_bought * peach_cost →
  P = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_plum_cost_l477_47789


namespace NUMINAMATH_GPT_count_shapes_in_figure_l477_47737

-- Definitions based on the conditions
def firstLayerTriangles : Nat := 3
def secondLayerSquares : Nat := 2
def thirdLayerLargeTriangle : Nat := 1
def totalSmallTriangles := firstLayerTriangles
def totalLargeTriangles := thirdLayerLargeTriangle
def totalTriangles := totalSmallTriangles + totalLargeTriangles
def totalSquares := secondLayerSquares

-- Lean 4 statement to prove the problem
theorem count_shapes_in_figure : totalTriangles = 4 ∧ totalSquares = 2 :=
by {
  -- The proof is not required, so we use sorry to skip it.
  sorry
}

end NUMINAMATH_GPT_count_shapes_in_figure_l477_47737


namespace NUMINAMATH_GPT_max_students_distribute_eq_pens_pencils_l477_47706

theorem max_students_distribute_eq_pens_pencils (n_pens n_pencils n : ℕ) (h_pens : n_pens = 890) (h_pencils : n_pencils = 630) :
  (∀ k : ℕ, k > n → (n_pens % k ≠ 0 ∨ n_pencils % k ≠ 0)) → (n = Nat.gcd n_pens n_pencils) := by
  sorry

end NUMINAMATH_GPT_max_students_distribute_eq_pens_pencils_l477_47706


namespace NUMINAMATH_GPT_inequality_proof_l477_47792

theorem inequality_proof (a b c : ℝ) (h : a > b) : a / (c ^ 2 + 1) > b / (c ^ 2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l477_47792


namespace NUMINAMATH_GPT_fencing_rate_l477_47701

noncomputable def rate_per_meter (d : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := Real.pi * d
  total_cost / circumference

theorem fencing_rate (diameter cost : ℝ) (h₀ : diameter = 34) (h₁ : cost = 213.63) :
  rate_per_meter diameter cost = 2 := by
  sorry

end NUMINAMATH_GPT_fencing_rate_l477_47701


namespace NUMINAMATH_GPT_blue_marbles_in_bag_l477_47721

theorem blue_marbles_in_bag
  (total_marbles : ℕ)
  (red_marbles : ℕ)
  (prob_red_white : ℚ)
  (number_red_marbles: red_marbles = 9) 
  (total_marbles_eq: total_marbles = 30) 
  (prob_red_white_eq: prob_red_white = 5/6): 
  ∃ (blue_marbles : ℕ), blue_marbles = 5 :=
by
  have W := 16        -- This is from (9 + W)/30 = 5/6 which gives W = 16
  let B := total_marbles - red_marbles - W
  use B
  have h : B = 30 - 9 - 16 := by
    -- Remaining calculations
    sorry
  exact h

end NUMINAMATH_GPT_blue_marbles_in_bag_l477_47721


namespace NUMINAMATH_GPT_solve_equation_l477_47743

theorem solve_equation : ∀ x : ℝ, (x - (x + 2) / 2 = (2 * x - 1) / 3 - 1) → (x = 2) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_equation_l477_47743


namespace NUMINAMATH_GPT_ab_value_l477_47784

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l477_47784


namespace NUMINAMATH_GPT_pairs_satisfying_condition_l477_47798

theorem pairs_satisfying_condition :
  (∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 1000 ∧ 1 ≤ y ∧ y ≤ 1000 ∧ (x^2 + y^2) % 7 = 0) → 
  (∃ n : ℕ, n = 20164) :=
sorry

end NUMINAMATH_GPT_pairs_satisfying_condition_l477_47798


namespace NUMINAMATH_GPT_isosceles_trapezoid_height_l477_47742

theorem isosceles_trapezoid_height (S h : ℝ) (h_nonneg : 0 ≤ h) 
  (diag_perpendicular : S = (1 / 2) * h^2) : h = Real.sqrt S :=
by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_height_l477_47742


namespace NUMINAMATH_GPT_research_question_correct_survey_method_correct_l477_47708

-- Define the conditions.
def total_students : Nat := 400
def sampled_students : Nat := 80

-- Define the research question.
def research_question : String := "To understand the vision conditions of 400 eighth-grade students in a certain school."

-- Define the survey method.
def survey_method : String := "A sampling survey method was used."

-- Prove the research_question matches the expected question given the conditions.
theorem research_question_correct :
  research_question = "To understand the vision conditions of 400 eighth-grade students in a certain school" := by
  sorry

-- Prove the survey method used matches the expected method given the conditions.
theorem survey_method_correct :
  survey_method = "A sampling survey method was used" := by
  sorry

end NUMINAMATH_GPT_research_question_correct_survey_method_correct_l477_47708


namespace NUMINAMATH_GPT_clock_hands_straight_twenty_four_hours_l477_47770

noncomputable def hands_straight_per_day : ℕ :=
  2 * 22

theorem clock_hands_straight_twenty_four_hours :
  hands_straight_per_day = 44 :=
by
  sorry

end NUMINAMATH_GPT_clock_hands_straight_twenty_four_hours_l477_47770


namespace NUMINAMATH_GPT_minimum_value_expr_l477_47709

theorem minimum_value_expr (x y : ℝ) : 
  ∃ (a b : ℝ), 2 * x^2 + 3 * y^2 - 12 * x + 6 * y + 25 = 2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ∧ 
  2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ≥ 4 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_expr_l477_47709


namespace NUMINAMATH_GPT_initial_amount_of_liquid_A_l477_47732

theorem initial_amount_of_liquid_A (A B : ℕ) (x : ℕ) (h1 : 4 * x = A) (h2 : x = B) (h3 : 4 * x + x = 5 * x)
    (h4 : 4 * x - 8 = 3 * (x + 8) / 2) : A = 16 :=
  by
  sorry

end NUMINAMATH_GPT_initial_amount_of_liquid_A_l477_47732


namespace NUMINAMATH_GPT_count_3_digit_numbers_divisible_by_5_l477_47786

theorem count_3_digit_numbers_divisible_by_5 :
  let a := 100
  let l := 995
  let d := 5
  let n := (l - a) / d + 1
  n = 180 :=
by
  sorry

end NUMINAMATH_GPT_count_3_digit_numbers_divisible_by_5_l477_47786


namespace NUMINAMATH_GPT_repay_loan_with_interest_l477_47779

theorem repay_loan_with_interest (amount_borrowed : ℝ) (interest_rate : ℝ) (total_payment : ℝ) 
  (h1 : amount_borrowed = 100) (h2 : interest_rate = 0.10) :
  total_payment = amount_borrowed + (amount_borrowed * interest_rate) :=
by sorry

end NUMINAMATH_GPT_repay_loan_with_interest_l477_47779


namespace NUMINAMATH_GPT_pow_fraction_eq_l477_47734

theorem pow_fraction_eq : (4:ℕ) = 2^2 ∧ (8:ℕ) = 2^3 → (4^800 / 8^400 = 2^400) :=
by
  -- proof steps should go here, but they are omitted as per the instruction
  sorry

end NUMINAMATH_GPT_pow_fraction_eq_l477_47734


namespace NUMINAMATH_GPT_find_a_l477_47720

theorem find_a (a: ℕ) : (2000 + 100 * a + 17) % 19 = 0 ↔ a = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l477_47720


namespace NUMINAMATH_GPT_fraction_expression_eq_l477_47718

theorem fraction_expression_eq (x y : ℕ) (hx : x = 4) (hy : y = 5) : 
  ((1 / y) + (1 / x)) / (1 / x) = 9 / 5 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_fraction_expression_eq_l477_47718


namespace NUMINAMATH_GPT_product_evaluation_l477_47736

theorem product_evaluation (a b c : ℕ) (h : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) :
  6 * 15 * 2 = 4 := by
  sorry

end NUMINAMATH_GPT_product_evaluation_l477_47736


namespace NUMINAMATH_GPT_remainder_division_l477_47761

def f (x : ℝ) : ℝ := x^3 - 4 * x + 7

theorem remainder_division (x : ℝ) : f 3 = 22 := by
  sorry

end NUMINAMATH_GPT_remainder_division_l477_47761


namespace NUMINAMATH_GPT_tens_digit_of_7_pow_35_l477_47730

theorem tens_digit_of_7_pow_35 : 
  (7 ^ 35) % 100 / 10 % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_7_pow_35_l477_47730


namespace NUMINAMATH_GPT_find_original_price_l477_47793

-- Define the conditions for the problem
def original_price (P : ℝ) : Prop :=
  0.90 * P = 1620

-- Prove the original price P
theorem find_original_price (P : ℝ) (h : original_price P) : P = 1800 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_find_original_price_l477_47793


namespace NUMINAMATH_GPT_slope_of_parallel_line_l477_47788

theorem slope_of_parallel_line (m : ℚ) (b : ℚ) :
  (∀ x y : ℚ, 5 * x - 3 * y = 21 → y = (5 / 3) * x + b) →
  m = 5 / 3 :=
by
  intros hyp
  sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l477_47788


namespace NUMINAMATH_GPT_parabola_directrix_l477_47733

theorem parabola_directrix (y : ℝ) (x : ℝ) (h : y = 8 * x^2) : 
  y = -1 / 32 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l477_47733


namespace NUMINAMATH_GPT_circle_area_l477_47785

/-
Circle A has a diameter equal to the radius of circle B.
The area of circle A is 16π square units.
Prove the area of circle B is 64π square units.
-/

theorem circle_area (rA dA rB : ℝ) (h1 : dA = 2 * rA) (h2 : rB = dA) (h3 : π * rA ^ 2 = 16 * π) : π * rB ^ 2 = 64 * π :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l477_47785


namespace NUMINAMATH_GPT_largest_number_with_two_moves_l477_47776

theorem largest_number_with_two_moves (n : Nat) (matches_limit : Nat) (initial_number : Nat)
  (h_n : initial_number = 1405) (h_limit: matches_limit = 2) : n = 7705 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_with_two_moves_l477_47776


namespace NUMINAMATH_GPT_Jason_cards_l477_47714

theorem Jason_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 3) (h2 : cards_bought = 2) : remaining_cards = 1 :=
by
  sorry

end NUMINAMATH_GPT_Jason_cards_l477_47714


namespace NUMINAMATH_GPT_grims_groks_zeets_l477_47741

variable {T : Type}
variable (Groks Zeets Grims Snarks : Set T)

-- Given conditions as definitions in Lean 4
variable (h1 : Groks ⊆ Zeets)
variable (h2 : Grims ⊆ Zeets)
variable (h3 : Snarks ⊆ Groks)
variable (h4 : Grims ⊆ Snarks)

-- The statement to be proved
theorem grims_groks_zeets : Grims ⊆ Groks ∧ Grims ⊆ Zeets := by
  sorry

end NUMINAMATH_GPT_grims_groks_zeets_l477_47741


namespace NUMINAMATH_GPT_sine_thirteen_pi_over_six_l477_47755

theorem sine_thirteen_pi_over_six : Real.sin ((13 * Real.pi) / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sine_thirteen_pi_over_six_l477_47755


namespace NUMINAMATH_GPT_factorize_difference_of_squares_factorize_cubic_l477_47750

-- Problem 1: Prove that 4x^2 - 36 = 4(x + 3)(x - 3)
theorem factorize_difference_of_squares (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) := 
  sorry

-- Problem 2: Prove that x^3 - 2x^2y + xy^2 = x(x - y)^2
theorem factorize_cubic (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_factorize_cubic_l477_47750


namespace NUMINAMATH_GPT_average_salary_l477_47702

theorem average_salary (total_workers technicians other_workers technicians_avg_salary other_workers_avg_salary total_salary : ℝ)
  (h_workers : total_workers = 21)
  (h_technicians : technicians = 7)
  (h_other_workers : other_workers = total_workers - technicians)
  (h_technicians_avg_salary : technicians_avg_salary = 12000)
  (h_other_workers_avg_salary : other_workers_avg_salary = 6000)
  (h_total_technicians_salary : total_salary = (technicians * technicians_avg_salary + other_workers * other_workers_avg_salary))
  (h_total_other_salary : total_salary = 168000) :
  total_salary / total_workers = 8000 := by
    sorry

end NUMINAMATH_GPT_average_salary_l477_47702


namespace NUMINAMATH_GPT_triangle_area_l477_47753

theorem triangle_area (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10)
  (right_triangle : a^2 + b^2 = c^2) : (1 / 2 : ℝ) * (a * b) = 24 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l477_47753


namespace NUMINAMATH_GPT_fifth_term_sum_of_powers_of_4_l477_47764

theorem fifth_term_sum_of_powers_of_4 :
  (4^0 + 4^1 + 4^2 + 4^3 + 4^4) = 341 := 
by
  sorry

end NUMINAMATH_GPT_fifth_term_sum_of_powers_of_4_l477_47764


namespace NUMINAMATH_GPT_difference_30th_28th_triangular_l477_47751

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_30th_28th_triangular :
  triangular_number 30 - triangular_number 28 = 59 :=
by
  sorry

end NUMINAMATH_GPT_difference_30th_28th_triangular_l477_47751


namespace NUMINAMATH_GPT_geometric_sequence_sum_l477_47724

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 3)
  (h_sum : a 1 + a 3 + a 5 = 21) : 
  a 3 + a 5 + a 7 = 42 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l477_47724


namespace NUMINAMATH_GPT_dot_product_square_ABCD_l477_47767

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end NUMINAMATH_GPT_dot_product_square_ABCD_l477_47767


namespace NUMINAMATH_GPT_points_in_quadrants_l477_47787

theorem points_in_quadrants (x y : ℝ) (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : |x| = |y|) : 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end NUMINAMATH_GPT_points_in_quadrants_l477_47787


namespace NUMINAMATH_GPT_odd_function_solution_l477_47771

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_solution (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_solution_l477_47771


namespace NUMINAMATH_GPT_problem_c_l477_47778

noncomputable def M (a b : ℝ) := (a^4 + b^4) * (a^2 + b^2)
noncomputable def N (a b : ℝ) := (a^3 + b^3) ^ 2

theorem problem_c (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_neq : a ≠ b) : M a b > N a b := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_problem_c_l477_47778


namespace NUMINAMATH_GPT_find_percentage_decrease_l477_47762

noncomputable def initialPrice : ℝ := 100
noncomputable def priceAfterJanuary : ℝ := initialPrice * 1.30
noncomputable def priceAfterFebruary : ℝ := priceAfterJanuary * 0.85
noncomputable def priceAfterMarch : ℝ := priceAfterFebruary * 1.10

theorem find_percentage_decrease :
  ∃ (y : ℝ), (priceAfterMarch * (1 - y / 100) = initialPrice) ∧ abs (y - 18) < 1 := 
sorry

end NUMINAMATH_GPT_find_percentage_decrease_l477_47762


namespace NUMINAMATH_GPT_fraction_simplification_l477_47772

theorem fraction_simplification : (98 / 210 : ℚ) = 7 / 15 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_simplification_l477_47772


namespace NUMINAMATH_GPT_males_in_sample_l477_47731

theorem males_in_sample (total_employees female_employees sample_size : ℕ) 
  (h1 : total_employees = 300)
  (h2 : female_employees = 160)
  (h3 : sample_size = 15)
  (h4 : (female_employees * sample_size) / total_employees = 8) :
  sample_size - ((female_employees * sample_size) / total_employees) = 7 :=
by
  sorry

end NUMINAMATH_GPT_males_in_sample_l477_47731


namespace NUMINAMATH_GPT_volume_increase_is_79_4_percent_l477_47757

noncomputable def original_volume (L B H : ℝ) : ℝ := L * B * H

noncomputable def new_volume (L B H : ℝ) : ℝ :=
  (L * 1.15) * (B * 1.30) * (H * 1.20)

noncomputable def volume_increase (L B H : ℝ) : ℝ :=
  new_volume L B H - original_volume L B H

theorem volume_increase_is_79_4_percent (L B H : ℝ) :
  volume_increase L B H = 0.794 * original_volume L B H := by
  sorry

end NUMINAMATH_GPT_volume_increase_is_79_4_percent_l477_47757


namespace NUMINAMATH_GPT_complex_pow_difference_l477_47705

theorem complex_pow_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 12 - (1 - i) ^ 12 = 0 :=
  sorry

end NUMINAMATH_GPT_complex_pow_difference_l477_47705


namespace NUMINAMATH_GPT_faster_train_speed_l477_47775

theorem faster_train_speed (v : ℝ) (h_total_length : 100 + 100 = 200) 
  (h_cross_time : 8 = 8) (h_speeds : 3 * v = 200 / 8) : 2 * v = 50 / 3 :=
sorry

end NUMINAMATH_GPT_faster_train_speed_l477_47775


namespace NUMINAMATH_GPT_gcd_1617_1225_gcd_2023_111_gcd_589_6479_l477_47719

theorem gcd_1617_1225 : Nat.gcd 1617 1225 = 49 :=
by
  sorry

theorem gcd_2023_111 : Nat.gcd 2023 111 = 1 :=
by
  sorry

theorem gcd_589_6479 : Nat.gcd 589 6479 = 589 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1617_1225_gcd_2023_111_gcd_589_6479_l477_47719


namespace NUMINAMATH_GPT_statement_a_statement_b_statement_c_statement_d_l477_47780

open Real

-- Statement A (incorrect)
theorem statement_a (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a*c > b*d) := sorry

-- Statement B (correct)
theorem statement_b (a b : ℝ) (h1 : b < a) (h2 : a < 0) : (1 / a < 1 / b) := sorry

-- Statement C (incorrect)
theorem statement_c (a b : ℝ) (h : 1 / (a^2) < 1 / (b^2)) : ¬ (a > abs b) := sorry

-- Statement D (correct)
theorem statement_d (a b m : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : (a + m) / (b + m) > a / b := sorry

end NUMINAMATH_GPT_statement_a_statement_b_statement_c_statement_d_l477_47780


namespace NUMINAMATH_GPT_infinite_cube_volume_sum_l477_47763

noncomputable def sum_of_volumes_of_infinite_cubes (a : ℝ) : ℝ :=
  ∑' n, (((a / (3 ^ n))^3))

theorem infinite_cube_volume_sum (a : ℝ) : sum_of_volumes_of_infinite_cubes a = (27 / 26) * a^3 :=
sorry

end NUMINAMATH_GPT_infinite_cube_volume_sum_l477_47763


namespace NUMINAMATH_GPT_square_of_radius_l477_47735

-- Definitions based on conditions
def ER := 24
def RF := 31
def GS := 40
def SH := 29

-- The goal is to find square of radius r such that r^2 = 841
theorem square_of_radius (r : ℝ) :
  let R := ER
  let F := RF
  let G := GS
  let S := SH
  (∀ r : ℝ, (R + F) * (G + S) = r^2) → r^2 = 841 :=
sorry

end NUMINAMATH_GPT_square_of_radius_l477_47735


namespace NUMINAMATH_GPT_price_reduction_l477_47774

theorem price_reduction (C : ℝ) (h1 : C > 0) :
  let first_discounted_price := 0.7 * C
  let final_discounted_price := 0.8 * first_discounted_price
  let reduction := 1 - final_discounted_price / C
  reduction = 0.44 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_l477_47774


namespace NUMINAMATH_GPT_problem_solutions_l477_47725

theorem problem_solutions (a b c : ℝ) (h : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -4 ∨ x ≥ 3) :
  (a + b + c > 0) ∧ (∀ x, bx + c > 0 ↔ x < 12) :=
by
  -- The following proof steps are not needed as per the instructions provided
  sorry

end NUMINAMATH_GPT_problem_solutions_l477_47725


namespace NUMINAMATH_GPT_number_of_ways_to_write_528_as_sum_of_consecutive_integers_l477_47711

theorem number_of_ways_to_write_528_as_sum_of_consecutive_integers : 
  ∃ (n : ℕ), (2 ≤ n ∧ ∃ k : ℕ, n * (2 * k + n - 1) = 1056) ∧ n = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_write_528_as_sum_of_consecutive_integers_l477_47711


namespace NUMINAMATH_GPT_price_of_each_toy_l477_47729

variables (T : ℝ)

-- Given conditions
def total_cost (T : ℝ) : ℝ := 3 * T + 2 * 5 + 5 * 6

theorem price_of_each_toy :
  total_cost T = 70 → T = 10 :=
sorry

end NUMINAMATH_GPT_price_of_each_toy_l477_47729


namespace NUMINAMATH_GPT_find_piles_l477_47704

theorem find_piles :
  ∃ N : ℕ, 
  (1000 < N ∧ N < 2000) ∧ 
  (N % 2 = 1) ∧ (N % 3 = 1) ∧ (N % 4 = 1) ∧ 
  (N % 5 = 1) ∧ (N % 6 = 1) ∧ (N % 7 = 1) ∧ (N % 8 = 1) ∧ 
  (∃ p : ℕ, p = 41 ∧ p > 1 ∧ p < N ∧ N % p = 0) :=
sorry

end NUMINAMATH_GPT_find_piles_l477_47704


namespace NUMINAMATH_GPT_find_number_l477_47782

theorem find_number (x : ℤ) (h : 2 * x - 8 = -12) : x = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l477_47782


namespace NUMINAMATH_GPT_isosceles_triangle_smallest_angle_l477_47744

def is_isosceles (angle_A angle_B angle_C : ℝ) : Prop := 
(angle_A = angle_B) ∨ (angle_B = angle_C) ∨ (angle_C = angle_A)

theorem isosceles_triangle_smallest_angle
  (angle_A angle_B angle_C : ℝ)
  (h_isosceles : is_isosceles angle_A angle_B angle_C)
  (h_angle_162 : angle_A = 162) :
  angle_B = 9 ∧ angle_C = 9 ∨ angle_A = 9 ∧ (angle_B = 9 ∨ angle_C = 9) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_smallest_angle_l477_47744


namespace NUMINAMATH_GPT_find_speed_of_man_in_still_water_l477_47715

def speed_of_man_in_still_water (t1 t2 d1 d2: ℝ) (v_m v_s: ℝ) : Prop :=
  d1 / t1 = v_m + v_s ∧ d2 / t2 = v_m - v_s

theorem find_speed_of_man_in_still_water :
  ∃ v_m : ℝ, ∃ v_s : ℝ, speed_of_man_in_still_water 2 2 16 10 v_m v_s ∧ v_m = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_of_man_in_still_water_l477_47715


namespace NUMINAMATH_GPT_parabola_midpoint_length_squared_l477_47796

theorem parabola_midpoint_length_squared :
  ∀ (A B : ℝ × ℝ), 
  (∃ (x y : ℝ), A = (x, 3*x^2 + 4*x + 2) ∧ B = (-x, -(3*x^2 + 4*x + 2)) ∧ ((A.1 + B.1) / 2 = 0) ∧ ((A.2 + B.2) / 2 = 0)) →
  dist A B^2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_midpoint_length_squared_l477_47796


namespace NUMINAMATH_GPT_find_a_l477_47760

noncomputable def exists_nonconstant_function (a : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 ≠ f x2) ∧ 
  (∀ x : ℝ, f (a * x) = a^2 * f x) ∧
  (∀ x : ℝ, f (f x) = a * f x)

theorem find_a :
  ∀ (a : ℝ), exists_nonconstant_function a → (a = 0 ∨ a = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l477_47760


namespace NUMINAMATH_GPT_fifth_equation_l477_47758

noncomputable def equation_1 : Prop := 2 * 1 = 2
noncomputable def equation_2 : Prop := 2 ^ 2 * 1 * 3 = 3 * 4
noncomputable def equation_3 : Prop := 2 ^ 3 * 1 * 3 * 5 = 4 * 5 * 6

theorem fifth_equation
  (h1 : equation_1)
  (h2 : equation_2)
  (h3 : equation_3) :
  2 ^ 5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_fifth_equation_l477_47758


namespace NUMINAMATH_GPT_coefficients_divisible_by_seven_l477_47713

theorem coefficients_divisible_by_seven {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  a % 7 = 0 ∧ b % 7 = 0 ∧ c % 7 = 0 ∧ d % 7 = 0 ∧ e % 7 = 0 := 
  sorry

end NUMINAMATH_GPT_coefficients_divisible_by_seven_l477_47713


namespace NUMINAMATH_GPT_find_m_l477_47739

theorem find_m (m : ℕ) (h₁ : 256 = 4^4) : (256 : ℝ)^(1/4) = (4 : ℝ)^m ↔ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l477_47739


namespace NUMINAMATH_GPT_sum_of_numbers_ge_1_1_l477_47777

theorem sum_of_numbers_ge_1_1 :
  let numbers := [1.4, 0.9, 1.2, 0.5, 1.3]
  let threshold := 1.1
  let filtered_numbers := numbers.filter (fun x => x >= threshold)
  let sum_filtered := filtered_numbers.sum
  sum_filtered = 3.9 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_numbers_ge_1_1_l477_47777


namespace NUMINAMATH_GPT_geometric_sequence_problem_l477_47795

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : a 1 = 2)
  (h2 : a 1 + a 3 + a 5 = 14) (h_seq : geometric_sequence a) :
  (1 / a 1) + (1 / a 3) + (1 / a 5) = 7 / 8 := sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l477_47795


namespace NUMINAMATH_GPT_integer_satisfying_conditions_l477_47738

theorem integer_satisfying_conditions :
  {a : ℤ | 1 ≤ a ∧ a ≤ 105 ∧ 35 ∣ (a^3 - 1)} = {1, 11, 16, 36, 46, 51, 71, 81, 86} :=
by
  sorry

end NUMINAMATH_GPT_integer_satisfying_conditions_l477_47738


namespace NUMINAMATH_GPT_ordered_pair_A_B_l477_47700

noncomputable def cubic_function (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 6
noncomputable def linear_function (x : ℝ) : ℝ := -2 / 3 * x + 2

noncomputable def points_intersect (x1 x2 x3 y1 y2 y3 : ℝ) : Prop :=
  cubic_function x1 = y1 ∧ cubic_function x2 = y2 ∧ cubic_function x3 = y3 ∧
  2 * x1 + 3 * y1 = 6 ∧ 2 * x2 + 3 * y2 = 6 ∧ 2 * x3 + 3 * y3 = 6

theorem ordered_pair_A_B (x1 x2 x3 y1 y2 y3 A B : ℝ)
  (h_intersect : points_intersect x1 x2 x3 y1 y2 y3) 
  (h_sum_x : x1 + x2 + x3 = A)
  (h_sum_y : y1 + y2 + y3 = B) :
  (A, B) = (2, 14 / 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_ordered_pair_A_B_l477_47700


namespace NUMINAMATH_GPT_find_a10_of_arithmetic_sequence_l477_47766

theorem find_a10_of_arithmetic_sequence (a : ℕ → ℚ)
  (h_seq : ∀ n : ℕ, ∃ d : ℚ, ∀ m : ℕ, a (n + m + 1) = a (n + m) + d)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 4) :
  a 10 = -4 / 5 :=
sorry

end NUMINAMATH_GPT_find_a10_of_arithmetic_sequence_l477_47766


namespace NUMINAMATH_GPT_zog_words_count_l477_47783

-- Defining the number of letters in the Zoggian alphabet
def num_letters : ℕ := 6

-- Function to calculate the number of words with n letters
def words_with_n_letters (n : ℕ) : ℕ := num_letters ^ n

-- Definition to calculate the total number of words with at most 4 letters
def total_words : ℕ :=
  (words_with_n_letters 1) +
  (words_with_n_letters 2) +
  (words_with_n_letters 3) +
  (words_with_n_letters 4)

-- Theorem statement
theorem zog_words_count : total_words = 1554 := by
  sorry

end NUMINAMATH_GPT_zog_words_count_l477_47783


namespace NUMINAMATH_GPT_evaluate_expression_l477_47726

theorem evaluate_expression (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l477_47726


namespace NUMINAMATH_GPT_evaluate_expression_l477_47710

theorem evaluate_expression (x y : ℚ) (hx : x = 4 / 3) (hy : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 13 / 40 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l477_47710


namespace NUMINAMATH_GPT_stock_percentage_l477_47747

theorem stock_percentage (investment income : ℝ) (investment total : ℝ) (P : ℝ) : 
  (income = 3800) → (total = 15200) → (income = (total * P) / 100) → P = 25 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_stock_percentage_l477_47747


namespace NUMINAMATH_GPT_probability_all_red_is_correct_l477_47712

def total_marbles (R W B : Nat) : Nat := R + W + B

def first_red_probability (R W B : Nat) : Rat := R / total_marbles R W B
def second_red_probability (R W B : Nat) : Rat := (R - 1) / (total_marbles R W B - 1)
def third_red_probability (R W B : Nat) : Rat := (R - 2) / (total_marbles R W B - 2)

def all_red_probability (R W B : Nat) : Rat := 
  first_red_probability R W B * 
  second_red_probability R W B * 
  third_red_probability R W B

theorem probability_all_red_is_correct 
  (R W B : Nat) (hR : R = 5) (hW : W = 6) (hB : B = 7) :
  all_red_probability R W B = 5 / 408 := by
  sorry

end NUMINAMATH_GPT_probability_all_red_is_correct_l477_47712


namespace NUMINAMATH_GPT_colorful_family_children_count_l477_47790

theorem colorful_family_children_count 
    (B W S x : ℕ)
    (h1 : B = W) (h2 : W = S)
    (h3 : (B - x) + W = 10)
    (h4 : W + (S + x) = 18) :
    B + W + S = 21 :=
by
  sorry

end NUMINAMATH_GPT_colorful_family_children_count_l477_47790


namespace NUMINAMATH_GPT_can_form_triangle_l477_47768

theorem can_form_triangle (a b c : ℕ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 10) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  rw [h1, h2, h3]
  repeat {sorry}

end NUMINAMATH_GPT_can_form_triangle_l477_47768


namespace NUMINAMATH_GPT_Erik_ate_pie_l477_47746

theorem Erik_ate_pie (Frank_ate Erik_ate more_than: ℝ) (h1: Frank_ate = 0.3333333333333333)
(h2: more_than = 0.3333333333333333)
(h3: Erik_ate = Frank_ate + more_than) : Erik_ate = 0.6666666666666666 :=
by
  sorry

end NUMINAMATH_GPT_Erik_ate_pie_l477_47746


namespace NUMINAMATH_GPT_problem_l477_47723

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}
def C : Set ℕ := {1, 3}

theorem problem : A ∩ (U \ B) = C := by
  sorry

end NUMINAMATH_GPT_problem_l477_47723


namespace NUMINAMATH_GPT_negativity_of_c_plus_b_l477_47781

variable (a b c : ℝ)

def isWithinBounds : Prop := (1 < a ∧ a < 2) ∧ (0 < b ∧ b < 1) ∧ (-2 < c ∧ c < -1)

theorem negativity_of_c_plus_b (h : isWithinBounds a b c) : c + b < 0 :=
sorry

end NUMINAMATH_GPT_negativity_of_c_plus_b_l477_47781


namespace NUMINAMATH_GPT_four_digit_swap_square_l477_47728

theorem four_digit_swap_square (a b : ℤ) (N M : ℤ) : 
  N = 1111 * a + 123 ∧ 
  M = 1111 * a + 1023 ∧ 
  M = b ^ 2 → 
  N = 3456 := 
by sorry

end NUMINAMATH_GPT_four_digit_swap_square_l477_47728


namespace NUMINAMATH_GPT_garden_area_l477_47794

theorem garden_area 
  (property_width : ℕ)
  (property_length : ℕ)
  (garden_width_ratio : ℚ)
  (garden_length_ratio : ℚ)
  (width_ratio_eq : garden_width_ratio = (1 : ℚ) / 8)
  (length_ratio_eq : garden_length_ratio = (1 : ℚ) / 10)
  (property_width_eq : property_width = 1000)
  (property_length_eq : property_length = 2250) :
  (property_width * garden_width_ratio * property_length * garden_length_ratio = 28125) :=
  sorry

end NUMINAMATH_GPT_garden_area_l477_47794


namespace NUMINAMATH_GPT_relationship_of_inequalities_l477_47752

theorem relationship_of_inequalities (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a > b) → (a^2 > b^2)) ∧ 
  ¬ (∀ a b : ℝ, (a^2 > b^2) → (a > b)) := 
by 
  sorry

end NUMINAMATH_GPT_relationship_of_inequalities_l477_47752


namespace NUMINAMATH_GPT_find_multiple_of_number_l477_47716

theorem find_multiple_of_number (n : ℝ) (m : ℝ) (h1 : n ≠ 0) (h2 : n = 9) (h3 : (n + n^2) / 2 = m * n) : m = 5 :=
sorry

end NUMINAMATH_GPT_find_multiple_of_number_l477_47716


namespace NUMINAMATH_GPT_initial_mixtureA_amount_l477_47722

-- Condition 1: Mixture A is 20% oil and 80% material B by weight.
def oil_content (x : ℝ) : ℝ := 0.20 * x
def materialB_content (x : ℝ) : ℝ := 0.80 * x

-- Condition 2: 2 more kilograms of oil are added to a certain amount of mixture A
def oil_added := 2

-- Condition 3: 6 kilograms of mixture A must be added to make a 70% material B in the new mixture.
def mixture_added := 6

-- The total weight of the new mixture
def total_weight (x : ℝ) : ℝ := x + mixture_added + oil_added

-- The total amount of material B in the new mixture
def total_materialB (x : ℝ) : ℝ := 0.80 * x + 0.80 * mixture_added

-- The new mixture is supposed to be 70% material B.
def is_70_percent_materialB (x : ℝ) : Prop := total_materialB x = 0.70 * total_weight x

-- Proving x == 8 given the conditions
theorem initial_mixtureA_amount : ∃ x : ℝ, is_70_percent_materialB x ∧ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_mixtureA_amount_l477_47722


namespace NUMINAMATH_GPT_sum_of_reciprocal_transformed_roots_l477_47773

theorem sum_of_reciprocal_transformed_roots :
  ∀ (a b c : ℝ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    -1 < a ∧ a < 1 ∧
    -1 < b ∧ b < 1 ∧
    -1 < c ∧ c < 1 ∧
    (45 * a ^ 3 - 70 * a ^ 2 + 28 * a - 2 = 0) ∧
    (45 * b ^ 3 - 70 * b ^ 2 + 28 * b - 2 = 0) ∧
    (45 * c ^ 3 - 70 * c ^ 2 + 28 * c - 2 = 0)
  → (1 - a)⁻¹ + (1 - b)⁻¹ + (1 - c)⁻¹ = 13 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocal_transformed_roots_l477_47773


namespace NUMINAMATH_GPT_divisibility_1989_l477_47703

theorem divisibility_1989 (n : ℕ) (h1 : n ≥ 3) :
  1989 ∣ n^(n^(n^n)) - n^(n^n) :=
sorry

end NUMINAMATH_GPT_divisibility_1989_l477_47703


namespace NUMINAMATH_GPT_minimum_value_expression_l477_47769

theorem minimum_value_expression (x y : ℝ) : ∃ (m : ℝ), ∀ x y : ℝ, x^2 + 3 * x * y + y^2 ≥ m ∧ m = 0 :=
by
  use 0
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l477_47769


namespace NUMINAMATH_GPT_min_major_axis_l477_47717

theorem min_major_axis (a b c : ℝ) (h1 : b * c = 1) (h2 : a = Real.sqrt (b^2 + c^2)) : 2 * a ≥ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_major_axis_l477_47717


namespace NUMINAMATH_GPT_problem_l477_47754

variable (a : Int)
variable (h : -a = 1)

theorem problem : 3 * a - 2 = -5 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_problem_l477_47754


namespace NUMINAMATH_GPT_sum_of_cubes_is_zero_l477_47791

theorem sum_of_cubes_is_zero 
  (a b : ℝ) 
  (h1 : a + b = 0) 
  (h2 : a * b = -1) : 
  a^3 + b^3 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_is_zero_l477_47791


namespace NUMINAMATH_GPT_reciprocal_of_sum_l477_47749

theorem reciprocal_of_sum :
  (1 / ((3 : ℚ) / 4 + (5 : ℚ) / 6)) = (12 / 19) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_sum_l477_47749


namespace NUMINAMATH_GPT_no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l477_47797

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n - 2) ^ 2 + (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2

def is_divisible_by (a b : ℤ) : Prop := b ≠ 0 ∧ a % b = 0

theorem no_member_of_T_divisible_by_9_but_some_member_divisible_by_4 :
  ¬ (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 9) ∧
  (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 4) :=
by 
  sorry

end NUMINAMATH_GPT_no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l477_47797


namespace NUMINAMATH_GPT_dig_days_l477_47727

theorem dig_days (m1 m2 : ℕ) (d1 d2 : ℚ) (k : ℚ) 
  (h1 : m1 * d1 = k) (h2 : m2 * d2 = k) : 
  m1 = 30 ∧ d1 = 6 ∧ m2 = 40 → d2 = 4.5 := 
by sorry

end NUMINAMATH_GPT_dig_days_l477_47727


namespace NUMINAMATH_GPT_an_expression_l477_47745

-- Given conditions
def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - n

-- The statement to be proved
theorem an_expression (a : ℕ → ℕ) (n : ℕ) (h_Sn : ∀ n, Sn a n = 2 * a n - n) :
  a n = 2^n - 1 :=
sorry

end NUMINAMATH_GPT_an_expression_l477_47745


namespace NUMINAMATH_GPT_number_machine_output_l477_47759

def machine (x : ℕ) : ℕ := x + 15 - 6

theorem number_machine_output : machine 68 = 77 := by
  sorry

end NUMINAMATH_GPT_number_machine_output_l477_47759


namespace NUMINAMATH_GPT_cantor_length_formula_l477_47756

noncomputable def cantor_length : ℕ → ℚ
| 0 => 1
| (n+1) => 2/3 * cantor_length n

theorem cantor_length_formula (n : ℕ) : cantor_length n = (2/3 : ℚ)^(n-1) :=
  sorry

end NUMINAMATH_GPT_cantor_length_formula_l477_47756


namespace NUMINAMATH_GPT_sum_of_roots_quadratic_l477_47740

theorem sum_of_roots_quadratic :
  ∀ (a b : ℝ), (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a + b = 1) :=
by
  intro a b
  intros
  sorry

end NUMINAMATH_GPT_sum_of_roots_quadratic_l477_47740


namespace NUMINAMATH_GPT_net_change_salary_l477_47799

/-- Given an initial salary S and a series of percentage changes:
    20% increase, 10% decrease, 15% increase, and 5% decrease,
    prove that the net change in salary is 17.99%. -/
theorem net_change_salary (S : ℝ) :
  (1.20 * 0.90 * 1.15 * 0.95 - 1) * S = 0.1799 * S :=
sorry

end NUMINAMATH_GPT_net_change_salary_l477_47799


namespace NUMINAMATH_GPT_find_C_line_MN_l477_47748

def point := (ℝ × ℝ)

-- Given points A and B
def A : point := (5, -2)
def B : point := (7, 3)

-- Conditions: M is the midpoint of AC and is on the y-axis
def M_on_y_axis (M : point) (A C : point) : Prop :=
  M.1 = 0 ∧ M.2 = (A.2 + C.2) / 2

-- Conditions: N is the midpoint of BC and is on the x-axis
def N_on_x_axis (N : point) (B C : point) : Prop :=
  N.1 = (B.1 + C.1) / 2 ∧ N.2 = 0

-- Coordinates of point C
theorem find_C (C : point)
  (M : point) (N : point)
  (hM : M_on_y_axis M A C)
  (hN : N_on_x_axis N B C) : C = (-5, -8) := sorry

-- Equation of line MN
theorem line_MN (M N : point)
  (MN_eq : M_on_y_axis M A (-5, -8) ∧ N_on_x_axis N B (-5, -8)) :
   ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ ((y = M.2) ∧ (x = M.1)) ∨ ((y = N.2) ∧ (x = N.1))) ∧ m = (3/2) ∧ b = 0 := sorry

end NUMINAMATH_GPT_find_C_line_MN_l477_47748


namespace NUMINAMATH_GPT_ordering_of_xyz_l477_47707

theorem ordering_of_xyz :
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  z < y ∧ y < x :=
by
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  sorry

end NUMINAMATH_GPT_ordering_of_xyz_l477_47707
