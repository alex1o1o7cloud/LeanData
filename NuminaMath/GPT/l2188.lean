import Mathlib

namespace NUMINAMATH_GPT_number_of_questionnaires_drawn_from_15_to_16_is_120_l2188_218871

variable (x : ℕ)
variable (H1 : 120 + 180 + 240 + x = 900)
variable (H2 : 60 = (bit0 90) / 180)
variable (H3 : (bit0 (bit0 (bit0 15))) = (bit0 (bit0 (bit0 15))) * (900 / 300))

theorem number_of_questionnaires_drawn_from_15_to_16_is_120 :
  ((900 - 120 - 180 - 240) * (300 / 900)) = 120 :=
sorry

end NUMINAMATH_GPT_number_of_questionnaires_drawn_from_15_to_16_is_120_l2188_218871


namespace NUMINAMATH_GPT_fraction_of_planted_area_l2188_218873

-- Definitions of the conditions
def right_triangle (a b : ℕ) : Prop :=
  a * a + b * b = (Int.sqrt (a ^ 2 + b ^ 2))^2

def unplanted_square_distance (dist : ℕ) : Prop :=
  dist = 3

-- The main theorem to be proved
theorem fraction_of_planted_area (a b : ℕ) (dist : ℕ) (h_triangle : right_triangle a b) (h_square_dist : unplanted_square_distance dist) :
  (a = 5) → (b = 12) → ((a * b - dist ^ 2) / (a * b) = 412 / 1000) :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_planted_area_l2188_218873


namespace NUMINAMATH_GPT_exponentiation_rule_l2188_218875

theorem exponentiation_rule (a m : ℕ) (h : (a^2)^m = a^6) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_exponentiation_rule_l2188_218875


namespace NUMINAMATH_GPT_count_total_shells_l2188_218846

theorem count_total_shells 
  (purple_shells : ℕ := 13)
  (pink_shells : ℕ := 8)
  (yellow_shells : ℕ := 18)
  (blue_shells : ℕ := 12)
  (orange_shells : ℕ := 14) :
  purple_shells + pink_shells + yellow_shells + blue_shells + orange_shells = 65 :=
by
  -- Calculation
  sorry

end NUMINAMATH_GPT_count_total_shells_l2188_218846


namespace NUMINAMATH_GPT_nearly_tricky_7_tiny_count_l2188_218819

-- Define a tricky polynomial
def is_tricky (P : Polynomial ℤ) : Prop :=
  Polynomial.eval 4 P = 0

-- Define a k-tiny polynomial
def is_k_tiny (k : ℤ) (P : Polynomial ℤ) : Prop :=
  P.degree ≤ 7 ∧ ∀ i, abs (Polynomial.coeff P i) ≤ k

-- Define a 1-tiny polynomial
def is_1_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 1 P

-- Define a nearly tricky polynomial as the sum of a tricky polynomial and a 1-tiny polynomial
def is_nearly_tricky (P : Polynomial ℤ) : Prop :=
  ∃ Q T : Polynomial ℤ, is_tricky Q ∧ is_1_tiny T ∧ P = Q + T

-- Define a 7-tiny polynomial
def is_7_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 7 P

-- Count the number of nearly tricky 7-tiny polynomials
def count_nearly_tricky_7_tiny : ℕ :=
  -- Simplification: hypothetical function counting the number of polynomials
  sorry

-- The main theorem statement
theorem nearly_tricky_7_tiny_count :
  count_nearly_tricky_7_tiny = 64912347 :=
sorry

end NUMINAMATH_GPT_nearly_tricky_7_tiny_count_l2188_218819


namespace NUMINAMATH_GPT_print_colored_pages_l2188_218822

theorem print_colored_pages (cost_per_page : ℕ) (dollars : ℕ) (conversion_rate : ℕ) 
    (h_cost : cost_per_page = 4) (h_dollars : dollars = 30) (h_conversion : conversion_rate = 100) :
    (dollars * conversion_rate) / cost_per_page = 750 := 
by
  sorry

end NUMINAMATH_GPT_print_colored_pages_l2188_218822


namespace NUMINAMATH_GPT_math_problem_l2188_218884

/-- The proof problem: Calculate -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11. -/
theorem math_problem : -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2188_218884


namespace NUMINAMATH_GPT_triangles_with_positive_integer_area_count_l2188_218839

theorem triangles_with_positive_integer_area_count :
  let points := { p : (ℕ × ℕ) // 41 * p.1 + p.2 = 2017 }
  ∃ count, count = 600 ∧ ∀ (P Q : points), P ≠ Q →
    let area := (P.val.1 * Q.val.2 - Q.val.1 * P.val.2 : ℤ)
    0 < area ∧ (area % 2 = 0) := sorry

end NUMINAMATH_GPT_triangles_with_positive_integer_area_count_l2188_218839


namespace NUMINAMATH_GPT_find_Q_digit_l2188_218854

theorem find_Q_digit (P Q R S T U : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S)
  (h4 : P ≠ T) (h5 : P ≠ U) (h6 : Q ≠ R) (h7 : Q ≠ S) (h8 : Q ≠ T)
  (h9 : Q ≠ U) (h10 : R ≠ S) (h11 : R ≠ T) (h12 : R ≠ U) (h13 : S ≠ T)
  (h14 : S ≠ U) (h15 : T ≠ U) (h_range_P : 4 ≤ P ∧ P ≤ 9)
  (h_range_Q : 4 ≤ Q ∧ Q ≤ 9) (h_range_R : 4 ≤ R ∧ R ≤ 9)
  (h_range_S : 4 ≤ S ∧ S ≤ 9) (h_range_T : 4 ≤ T ∧ T ≤ 9)
  (h_range_U : 4 ≤ U ∧ U ≤ 9) 
  (h_sum_lines : 3 * P + 2 * Q + 3 * S + R + T + 2 * U = 100)
  (h_sum_digits : P + Q + S + R + T + U = 39) : Q = 6 :=
sorry  -- proof to be provided

end NUMINAMATH_GPT_find_Q_digit_l2188_218854


namespace NUMINAMATH_GPT_nonagon_diagonals_l2188_218860

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nonagon_diagonals : number_of_diagonals 9 = 27 := 
by
  sorry

end NUMINAMATH_GPT_nonagon_diagonals_l2188_218860


namespace NUMINAMATH_GPT_domain_of_function_l2188_218848

theorem domain_of_function :
  { x : ℝ | x + 2 ≥ 0 ∧ x - 1 ≠ 0 } = { x : ℝ | x ≥ -2 ∧ x ≠ 1 } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2188_218848


namespace NUMINAMATH_GPT_radius_of_circle_l2188_218829

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

-- Prove that given the circle's equation, the radius is 1
theorem radius_of_circle (x y : ℝ) :
  circle_equation x y → ∃ (r : ℝ), r = 1 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l2188_218829


namespace NUMINAMATH_GPT_area_of_quadrilateral_l2188_218879

theorem area_of_quadrilateral (d h1 h2 : ℝ) (h1_pos : h1 = 9) (h2_pos : h2 = 6) (d_pos : d = 30) : 
  let area1 := (1/2 : ℝ) * d * h1
  let area2 := (1/2 : ℝ) * d * h2
  (area1 + area2) = 225 :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l2188_218879


namespace NUMINAMATH_GPT_total_seats_theater_l2188_218845

theorem total_seats_theater (a1 an d n Sn : ℕ) 
    (h1 : a1 = 12) 
    (h2 : d = 2) 
    (h3 : an = 48) 
    (h4 : an = a1 + (n - 1) * d) 
    (h5 : Sn = n * (a1 + an) / 2) : 
    Sn = 570 := 
sorry

end NUMINAMATH_GPT_total_seats_theater_l2188_218845


namespace NUMINAMATH_GPT_range_of_a1_l2188_218866

theorem range_of_a1 (a : ℕ → ℕ) (S : ℕ → ℕ) (h_seq : ∀ n, 12 * S n = 4 * a (n + 1) + 5^n - 13)
  (h_S4 : ∀ n, S n ≤ S 4):
  13 / 48 ≤ a 1 ∧ a 1 ≤ 59 / 64 :=
sorry

end NUMINAMATH_GPT_range_of_a1_l2188_218866


namespace NUMINAMATH_GPT_thirty_times_multiple_of_every_integer_is_zero_l2188_218800

theorem thirty_times_multiple_of_every_integer_is_zero (n : ℤ) (h : ∀ x : ℤ, n = 30 * x ∧ x = 0 → n = 0) : n = 0 :=
by
  sorry

end NUMINAMATH_GPT_thirty_times_multiple_of_every_integer_is_zero_l2188_218800


namespace NUMINAMATH_GPT_adam_earnings_l2188_218883

theorem adam_earnings
  (earn_per_lawn : ℕ) (total_lawns : ℕ) (forgot_lawns : ℕ)
  (h1 : earn_per_lawn = 9) (h2 : total_lawns = 12) (h3 : forgot_lawns = 8) :
  (total_lawns - forgot_lawns) * earn_per_lawn = 36 :=
by
  sorry

end NUMINAMATH_GPT_adam_earnings_l2188_218883


namespace NUMINAMATH_GPT_correct_operation_l2188_218823

-- Define the conditions
def cond1 (m : ℝ) : Prop := m^2 + m^3 ≠ m^5
def cond2 (m : ℝ) : Prop := m^2 * m^3 = m^5
def cond3 (m : ℝ) : Prop := (m^2)^3 = m^6

-- Main statement that checks the correct operation
theorem correct_operation (m : ℝ) : cond1 m → cond2 m → cond3 m → (m^2 * m^3 = m^5) :=
by
  intros h1 h2 h3
  exact h2

end NUMINAMATH_GPT_correct_operation_l2188_218823


namespace NUMINAMATH_GPT_M_inter_N_is_empty_l2188_218834

-- Definition conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | (x - 1) / x < 0}

-- Theorem statement
theorem M_inter_N_is_empty : M ∩ N = ∅ := by
  sorry

end NUMINAMATH_GPT_M_inter_N_is_empty_l2188_218834


namespace NUMINAMATH_GPT_problem1_problem2_l2188_218840
noncomputable section

-- Problem (1) Lean Statement
theorem problem1 : |-4| - (2021 - Real.pi)^0 + (Real.cos (Real.pi / 3))⁻¹ - (-Real.sqrt 3)^2 = 2 :=
by 
  sorry

-- Problem (2) Lean Statement
theorem problem2 (a : ℝ) (h : a ≠ 2 ∧ a ≠ -2) : 
  (1 + 4 / (a^2 - 4)) / (a / (a + 2)) = a / (a - 2) := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2188_218840


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_l2188_218865

theorem general_term_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (a1 : a 1 = -1) 
  (d : ℤ) 
  (h : d = 4) : 
  ∀ n : ℕ, a n = 4 * n - 5 :=
by
  sorry

end NUMINAMATH_GPT_general_term_arithmetic_sequence_l2188_218865


namespace NUMINAMATH_GPT_find_y_l2188_218841

theorem find_y (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := 
by sorry

end NUMINAMATH_GPT_find_y_l2188_218841


namespace NUMINAMATH_GPT_integer_solutions_zero_l2188_218898

theorem integer_solutions_zero (x y u t : ℤ) :
  x^2 + y^2 = 1974 * (u^2 + t^2) → 
  x = 0 ∧ y = 0 ∧ u = 0 ∧ t = 0 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_zero_l2188_218898


namespace NUMINAMATH_GPT_largest_even_sum_1988_is_290_l2188_218818

theorem largest_even_sum_1988_is_290 (n : ℕ) 
  (h : 14 * n = 1988) : 2 * n + 6 = 290 :=
sorry

end NUMINAMATH_GPT_largest_even_sum_1988_is_290_l2188_218818


namespace NUMINAMATH_GPT_square_area_l2188_218821

theorem square_area (side_length : ℝ) (h : side_length = 11) : side_length * side_length = 121 := 
by 
  simp [h]
  sorry

end NUMINAMATH_GPT_square_area_l2188_218821


namespace NUMINAMATH_GPT_returning_players_count_l2188_218826

def total_players_in_team (groups : ℕ) (players_per_group : ℕ): ℕ := groups * players_per_group
def returning_players (total_players : ℕ) (new_players : ℕ): ℕ := total_players - new_players

theorem returning_players_count
    (new_players : ℕ)
    (groups : ℕ)
    (players_per_group : ℕ)
    (total_players : ℕ := total_players_in_team groups players_per_group)
    (returning_players_count : ℕ := returning_players total_players new_players):
    new_players = 4 ∧
    groups = 2 ∧
    players_per_group = 5 → 
    returning_players_count = 6 := by
    intros h
    sorry

end NUMINAMATH_GPT_returning_players_count_l2188_218826


namespace NUMINAMATH_GPT_total_amount_of_money_if_all_cookies_sold_equals_1255_50_l2188_218803

-- Define the conditions
def number_cookies_Clementine : ℕ := 72
def number_cookies_Jake : ℕ := 5 * number_cookies_Clementine / 2
def number_cookies_Tory : ℕ := (number_cookies_Jake + number_cookies_Clementine) / 2
def number_cookies_Spencer : ℕ := 3 * (number_cookies_Jake + number_cookies_Tory) / 2
def price_per_cookie : ℝ := 1.50

-- Total number of cookies
def total_cookies : ℕ :=
  number_cookies_Clementine + number_cookies_Jake + number_cookies_Tory + number_cookies_Spencer

-- Proof statement
theorem total_amount_of_money_if_all_cookies_sold_equals_1255_50 :
  (total_cookies * price_per_cookie : ℝ) = 1255.50 := by
  sorry

end NUMINAMATH_GPT_total_amount_of_money_if_all_cookies_sold_equals_1255_50_l2188_218803


namespace NUMINAMATH_GPT_problem_solution_l2188_218893

theorem problem_solution (x : ℝ) (h : x * Real.log 4 / Real.log 3 = 1) : 
  2^x + 4^(-x) = 1 / 3 + Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l2188_218893


namespace NUMINAMATH_GPT_game_show_prizes_count_l2188_218872

theorem game_show_prizes_count:
  let digits := [1, 1, 1, 1, 3, 3, 3, 3]
  let is_valid_prize (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9999
  let is_three_digit_or_more (n : ℕ) : Prop := 100 ≤ n
  ∃ (A B C : ℕ), 
    is_valid_prize A ∧ is_valid_prize B ∧ is_valid_prize C ∧
    is_three_digit_or_more C ∧
    (A + B + C = digits.sum) ∧
    (A + B + C = 1260) := sorry

end NUMINAMATH_GPT_game_show_prizes_count_l2188_218872


namespace NUMINAMATH_GPT_barefoot_kids_count_l2188_218817

def kidsInClassroom : Nat := 35
def kidsWearingSocks : Nat := 18
def kidsWearingShoes : Nat := 15
def kidsWearingBoth : Nat := 8

def barefootKids : Nat := kidsInClassroom - (kidsWearingSocks - kidsWearingBoth + kidsWearingShoes - kidsWearingBoth + kidsWearingBoth)

theorem barefoot_kids_count : barefootKids = 10 := by
  sorry

end NUMINAMATH_GPT_barefoot_kids_count_l2188_218817


namespace NUMINAMATH_GPT_hermione_utility_l2188_218897

theorem hermione_utility (h : ℕ) : (h * (10 - h) = (4 - h) * (h + 2)) ↔ h = 4 := by
  sorry

end NUMINAMATH_GPT_hermione_utility_l2188_218897


namespace NUMINAMATH_GPT_range_f_does_not_include_zero_l2188_218877

noncomputable def f (x : ℝ) : ℤ :=
if x > 0 then ⌈1 / (x + 1)⌉ else if x < 0 then ⌈1 / (x - 1)⌉ else 0 -- this will be used only as a formal definition

theorem range_f_does_not_include_zero : ¬ (0 ∈ {y : ℤ | ∃ x : ℝ, x ≠ 0 ∧ y = f x}) :=
by sorry

end NUMINAMATH_GPT_range_f_does_not_include_zero_l2188_218877


namespace NUMINAMATH_GPT_investments_ratio_l2188_218806

theorem investments_ratio (P Q : ℝ) (hpq : 7 / 10 = (P * 2) / (Q * 4)) : P / Q = 7 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_investments_ratio_l2188_218806


namespace NUMINAMATH_GPT_number_of_solutions_l2188_218874

theorem number_of_solutions : 
  ∃ n : ℕ, n = 5 ∧ (∃ (x y : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ 4 * x + 5 * y = 98) :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l2188_218874


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l2188_218828

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l2188_218828


namespace NUMINAMATH_GPT_inequality_proof_l2188_218868

variable (a b c d e f : Real)

theorem inequality_proof (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2188_218868


namespace NUMINAMATH_GPT_calculate_product_l2188_218861

theorem calculate_product : 3^6 * 4^3 = 46656 := by
  sorry

end NUMINAMATH_GPT_calculate_product_l2188_218861


namespace NUMINAMATH_GPT_factor_quadratic_l2188_218851

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 16 * x^2 - 56 * x + 49

-- The goal is to prove that the quadratic expression is equal to (4x - 7)^2
theorem factor_quadratic (x : ℝ) : quadratic_expr x = (4 * x - 7)^2 :=
by
  sorry

end NUMINAMATH_GPT_factor_quadratic_l2188_218851


namespace NUMINAMATH_GPT_exists_prime_q_l2188_218862

theorem exists_prime_q (p : ℕ) (hp : Nat.Prime p) :
  ∃ q, Nat.Prime q ∧ ∀ n, ¬ (q ∣ n^p - p) := by
  sorry

end NUMINAMATH_GPT_exists_prime_q_l2188_218862


namespace NUMINAMATH_GPT_problem_l2188_218809

def f (x: ℝ) := 3 * x - 4
def g (x: ℝ) := 2 * x + 3

theorem problem (x : ℝ) : f (2 + g 3) = 29 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2188_218809


namespace NUMINAMATH_GPT_train_A_reaches_destination_in_6_hours_l2188_218870

noncomputable def t : ℕ := 
  let tA := 110
  let tB := 165
  let tB_time := 4
  (tB * tB_time) / tA

theorem train_A_reaches_destination_in_6_hours :
  t = 6 := by
  sorry

end NUMINAMATH_GPT_train_A_reaches_destination_in_6_hours_l2188_218870


namespace NUMINAMATH_GPT_fraction_difference_l2188_218859

theorem fraction_difference:
  let f1 := 2 / 3
  let f2 := 3 / 4
  let f3 := 4 / 5
  let f4 := 5 / 7
  (max f1 (max f2 (max f3 f4)) - min f1 (min f2 (min f3 f4))) = 2 / 15 :=
by
  sorry

end NUMINAMATH_GPT_fraction_difference_l2188_218859


namespace NUMINAMATH_GPT_prove_math_problem_l2188_218842

noncomputable def ellipse_foci : Prop := 
  ∃ (a b : ℝ), 
  a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ),
  (x^2 / a^2 + y^2 / b^2 = 1) → 
  a = 2 ∧ b^2 = 3)

noncomputable def intersect_and_rhombus : Prop :=
  ∃ (m : ℝ) (t : ℝ),
  (3 * m^2 + 4) > 0 ∧ 
  t = 1 / (3 * m^2 + 4) ∧ 
  0 < t ∧ t < 1 / 4

theorem prove_math_problem : ellipse_foci ∧ intersect_and_rhombus :=
by sorry

end NUMINAMATH_GPT_prove_math_problem_l2188_218842


namespace NUMINAMATH_GPT_Murtha_pebbles_l2188_218857

-- Definition of the geometric series sum formula
noncomputable def sum_geometric_series (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Constants for the problem
def a : ℕ := 1
def r : ℕ := 2
def n : ℕ := 10

-- The theorem to be proven
theorem Murtha_pebbles : sum_geometric_series a r n = 1023 :=
by
  -- Our condition setup implies the formula
  sorry

end NUMINAMATH_GPT_Murtha_pebbles_l2188_218857


namespace NUMINAMATH_GPT_new_rectangle_area_l2188_218869

theorem new_rectangle_area (a b : ℝ) : 
  let base := b + 2 * a
  let height := b - a
  let area := base * height
  area = b^2 + b * a - 2 * a^2 :=
by
  let base := b + 2 * a
  let height := b - a
  let area := base * height
  show area = b^2 + b * a - 2 * a^2
  sorry

end NUMINAMATH_GPT_new_rectangle_area_l2188_218869


namespace NUMINAMATH_GPT_height_of_fourth_person_l2188_218852

theorem height_of_fourth_person
  (h : ℝ)
  (H1 : h + (h + 2) + (h + 4) + (h + 10) = 4 * 79) :
  h + 10 = 85 :=
by
  have H2 : h + 4 = 79 := by linarith
  linarith


end NUMINAMATH_GPT_height_of_fourth_person_l2188_218852


namespace NUMINAMATH_GPT_negation_proposition_false_l2188_218899

theorem negation_proposition_false : 
  (¬ ∃ x : ℝ, x^2 + 2 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negation_proposition_false_l2188_218899


namespace NUMINAMATH_GPT_total_bottle_caps_in_collection_l2188_218882

-- Statements of given conditions
def small_box_caps : ℕ := 35
def large_box_caps : ℕ := 75
def num_small_boxes : ℕ := 7
def num_large_boxes : ℕ := 3
def individual_caps : ℕ := 23

-- Theorem statement that needs to be proved
theorem total_bottle_caps_in_collection :
  small_box_caps * num_small_boxes + large_box_caps * num_large_boxes + individual_caps = 493 :=
by sorry

end NUMINAMATH_GPT_total_bottle_caps_in_collection_l2188_218882


namespace NUMINAMATH_GPT_trig_expression_evaluation_l2188_218886

theorem trig_expression_evaluation
  (α : ℝ)
  (h : Real.tan α = 2) :
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_trig_expression_evaluation_l2188_218886


namespace NUMINAMATH_GPT_g_one_fourth_l2188_218830

noncomputable def g : ℝ → ℝ := sorry

theorem g_one_fourth :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1) ∧  -- g(x) is defined for 0 ≤ x ≤ 1
  g 0 = 0 ∧                                    -- g(0) = 0
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧ -- g is non-decreasing
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧ -- symmetric property
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2)   -- scaling property
  → g (1/4) = 1/2 :=
sorry

end NUMINAMATH_GPT_g_one_fourth_l2188_218830


namespace NUMINAMATH_GPT_w12_plus_inv_w12_l2188_218820

open Complex

-- Given conditions
def w_plus_inv_w_eq_two_cos_45 (w : ℂ) : Prop :=
  w + (1 / w) = 2 * Real.cos (Real.pi / 4)

-- Statement of the theorem to prove
theorem w12_plus_inv_w12 {w : ℂ} (h : w_plus_inv_w_eq_two_cos_45 w) : 
  w^12 + (1 / (w^12)) = -2 :=
sorry

end NUMINAMATH_GPT_w12_plus_inv_w12_l2188_218820


namespace NUMINAMATH_GPT_geometric_sequence_condition_l2188_218813

variable {a : ℕ → ℝ}

-- Definitions based on conditions in the problem
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The statement translating the problem
theorem geometric_sequence_condition (q : ℝ) (a : ℕ → ℝ) (h : is_geometric_sequence a q) : ¬((q > 1) ↔ is_increasing_sequence a) :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l2188_218813


namespace NUMINAMATH_GPT_odd_function_ln_negx_l2188_218867

theorem odd_function_ln_negx (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_positive : ∀ x, x > 0 → f x = Real.log x) :
  ∀ x, x < 0 → f x = -Real.log (-x) :=
by 
  intros x hx_neg
  have hx_pos : -x > 0 := by linarith
  rw [← h_positive (-x) hx_pos, h_odd x]
  sorry

end NUMINAMATH_GPT_odd_function_ln_negx_l2188_218867


namespace NUMINAMATH_GPT_ThreeDigitEvenNumbersCount_l2188_218881

theorem ThreeDigitEvenNumbersCount : 
  let a := 100
  let max := 998
  let d := 2
  let n := (max - a) / d + 1
  100 < 999 ∧ 100 % 2 = 0 ∧ max % 2 = 0 
  → d > 0 
  → n = 450 :=
by
  sorry

end NUMINAMATH_GPT_ThreeDigitEvenNumbersCount_l2188_218881


namespace NUMINAMATH_GPT_simplify_expression_l2188_218891

theorem simplify_expression (n : ℕ) : 
  (3^(n + 3) - 3 * 3^n) / (3 * 3^(n + 2)) = 8 / 3 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l2188_218891


namespace NUMINAMATH_GPT_pages_in_first_issue_l2188_218801

-- Define variables for the number of pages in the issues and total pages
variables (P : ℕ) (total_pages : ℕ) (eqn : total_pages = 3 * P + 4)

-- State the theorem using the given conditions and question
theorem pages_in_first_issue (h : total_pages = 220) : P = 72 :=
by
  -- Use the given equation
  have h_eqn : total_pages = 3 * P + 4 := eqn
  sorry

end NUMINAMATH_GPT_pages_in_first_issue_l2188_218801


namespace NUMINAMATH_GPT_notebooks_have_50_pages_l2188_218810

theorem notebooks_have_50_pages (notebooks : ℕ) (total_dollars : ℕ) (page_cost_cents : ℕ) 
  (total_cents : ℕ) (total_pages : ℕ) (pages_per_notebook : ℕ)
  (h1 : notebooks = 2) 
  (h2 : total_dollars = 5) 
  (h3 : page_cost_cents = 5) 
  (h4 : total_cents = total_dollars * 100) 
  (h5 : total_pages = total_cents / page_cost_cents) 
  (h6 : pages_per_notebook = total_pages / notebooks) 
  : pages_per_notebook = 50 :=
by
  sorry

end NUMINAMATH_GPT_notebooks_have_50_pages_l2188_218810


namespace NUMINAMATH_GPT_greatest_possible_value_of_n_l2188_218844

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_value_of_n_l2188_218844


namespace NUMINAMATH_GPT_remainder_of_division_l2188_218880

noncomputable def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 1
noncomputable def g (x : ℝ) : ℝ := (x - 3) ^ 2
noncomputable def remainder (x : ℝ) : ℝ := 324 * x - 488

theorem remainder_of_division :
  ∀ (x : ℝ), (f x) % (g x) = remainder x :=
sorry

end NUMINAMATH_GPT_remainder_of_division_l2188_218880


namespace NUMINAMATH_GPT_saree_sale_price_l2188_218894

def initial_price : Real := 150
def discount1 : Real := 0.20
def tax1 : Real := 0.05
def discount2 : Real := 0.15
def tax2 : Real := 0.04
def discount3 : Real := 0.10
def tax3 : Real := 0.03
def final_price : Real := 103.25

theorem saree_sale_price :
  let price_after_discount1 : Real := initial_price * (1 - discount1)
  let price_after_tax1 : Real := price_after_discount1 * (1 + tax1)
  let price_after_discount2 : Real := price_after_tax1 * (1 - discount2)
  let price_after_tax2 : Real := price_after_discount2 * (1 + tax2)
  let price_after_discount3 : Real := price_after_tax2 * (1 - discount3)
  let price_after_tax3 : Real := price_after_discount3 * (1 + tax3)
  abs (price_after_tax3 - final_price) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_saree_sale_price_l2188_218894


namespace NUMINAMATH_GPT_inheritance_problem_l2188_218812

theorem inheritance_problem
    (A B C : ℕ)
    (h1 : A + B + C = 30000)
    (h2 : A - B = B - C)
    (h3 : A = B + C) :
    A = 15000 ∧ B = 10000 ∧ C = 5000 := by
  sorry

end NUMINAMATH_GPT_inheritance_problem_l2188_218812


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l2188_218885

-- Define the isosceles triangle problem
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  perimeter : ℝ
  isIsosceles : (side1 = side2 ∨ side1 = base ∨ side2 = base)
  sideLengthCondition : (side1 = 3 ∨ side2 = 3 ∨ base = 3)
  perimeterCondition : side1 + side2 + base = 13
  triangleInequality1 : side1 + side2 > base
  triangleInequality2 : side1 + base > side2
  triangleInequality3 : side2 + base > side1

-- Define the theorem to prove
theorem isosceles_triangle_base_length (T : IsoscelesTriangle) :
  T.base = 3 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l2188_218885


namespace NUMINAMATH_GPT_remainder_of_4000th_term_l2188_218815

def sequence_term_position (n : ℕ) : ℕ :=
  n^2

def sum_of_squares_up_to (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem remainder_of_4000th_term : 
  ∃ n : ℕ, sum_of_squares_up_to n ≥ 4000 ∧ (n-1) * n * (2 * (n-1) + 1) / 6 < 4000 ∧ (n % 7) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_of_4000th_term_l2188_218815


namespace NUMINAMATH_GPT_nina_earnings_l2188_218890

/-- 
Problem: Calculate the total earnings from selling various types of jewelry.
Conditions:
- Necklace price: $25 each
- Bracelet price: $15 each
- Earring price: $10 per pair
- Complete jewelry ensemble price: $45 each
- Number of necklaces sold: 5
- Number of bracelets sold: 10
- Number of earrings sold: 20
- Number of complete jewelry ensembles sold: 2
Question: How much money did Nina make over the weekend?
Answer: Nina made $565.00
-/
theorem nina_earnings
  (necklace_price : ℕ)
  (bracelet_price : ℕ)
  (earring_price : ℕ)
  (ensemble_price : ℕ)
  (necklaces_sold : ℕ)
  (bracelets_sold : ℕ)
  (earrings_sold : ℕ)
  (ensembles_sold : ℕ) :
  necklace_price = 25 → 
  bracelet_price = 15 → 
  earring_price = 10 → 
  ensemble_price = 45 → 
  necklaces_sold = 5 → 
  bracelets_sold = 10 → 
  earrings_sold = 20 → 
  ensembles_sold = 2 →
  (necklace_price * necklaces_sold) + 
  (bracelet_price * bracelets_sold) + 
  (earring_price * earrings_sold) +
  (ensemble_price * ensembles_sold) = 565 := by
  sorry

end NUMINAMATH_GPT_nina_earnings_l2188_218890


namespace NUMINAMATH_GPT_soap_last_duration_l2188_218816

-- Definitions of the given conditions
def cost_per_bar := 8 -- cost in dollars
def total_spent := 48 -- total spent in dollars
def months_in_year := 12

-- Definition of the query statement/proof goal
theorem soap_last_duration (h₁ : total_spent = 48) (h₂ : cost_per_bar = 8) (h₃ : months_in_year = 12) : months_in_year / (total_spent / cost_per_bar) = 2 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_soap_last_duration_l2188_218816


namespace NUMINAMATH_GPT_expression_not_defined_l2188_218858

theorem expression_not_defined (x : ℝ) :
    ¬(x^2 - 22*x + 121 = 0) ↔ ¬(x - 11 = 0) :=
by sorry

end NUMINAMATH_GPT_expression_not_defined_l2188_218858


namespace NUMINAMATH_GPT_numerator_of_fraction_l2188_218895

-- Define the conditions
def y_pos (y : ℝ) : Prop := y > 0

-- Define the equation
def equation (x y : ℝ) : Prop := x + (3 * y) / 10 = (1 / 2) * y

-- Prove that x = (1/5) * y given the conditions
theorem numerator_of_fraction {y x : ℝ} (h1 : y_pos y) (h2 : equation x y) : x = (1/5) * y :=
  sorry

end NUMINAMATH_GPT_numerator_of_fraction_l2188_218895


namespace NUMINAMATH_GPT_value_of_m_l2188_218892

theorem value_of_m (m : ℤ) (h : m + 1 = - (-2)) : m = 1 :=
sorry

end NUMINAMATH_GPT_value_of_m_l2188_218892


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l2188_218833

theorem common_ratio_of_geometric_series (a r : ℝ) (h1 : (a / (1 - r)) = 81 * (a * (r^4) / (1 - r))) :
  r = 1 / 3 :=
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l2188_218833


namespace NUMINAMATH_GPT_monotonically_increasing_function_l2188_218836

open Function

theorem monotonically_increasing_function (f : ℝ → ℝ) (h_mono : ∀ x y, x < y → f x < f y) (t : ℝ) (h_t : t ≠ 0) :
    f (t^2 + t) > f t :=
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_function_l2188_218836


namespace NUMINAMATH_GPT_shifted_parabola_sum_l2188_218824

theorem shifted_parabola_sum (a b c : ℝ) :
  (∃ (a b c : ℝ), ∀ x : ℝ, 3 * x^2 + 2 * x - 5 = 3 * (x - 6)^2 + 2 * (x - 6) - 5 → y = a * x^2 + b * x + c) → a + b + c = 60 :=
sorry

end NUMINAMATH_GPT_shifted_parabola_sum_l2188_218824


namespace NUMINAMATH_GPT_concert_cost_l2188_218831

noncomputable def ticket_price : ℝ := 50.0
noncomputable def processing_fee_rate : ℝ := 0.15
noncomputable def parking_fee : ℝ := 10.0
noncomputable def entrance_fee : ℝ := 5.0
def number_of_people : ℕ := 2

noncomputable def processing_fee_per_ticket : ℝ := processing_fee_rate * ticket_price
noncomputable def total_cost_per_ticket : ℝ := ticket_price + processing_fee_per_ticket
noncomputable def total_ticket_cost : ℝ := number_of_people * total_cost_per_ticket
noncomputable def total_cost_with_parking : ℝ := total_ticket_cost + parking_fee
noncomputable def total_entrance_fee : ℝ := number_of_people * entrance_fee
noncomputable def total_cost : ℝ := total_cost_with_parking + total_entrance_fee

theorem concert_cost : total_cost = 135.0 := by
  sorry

end NUMINAMATH_GPT_concert_cost_l2188_218831


namespace NUMINAMATH_GPT_value_of_g_at_neg3_l2188_218888

def g (x : ℚ) : ℚ := (6 * x + 2) / (x - 2)

theorem value_of_g_at_neg3 : g (-3) = 16 / 5 := by
  sorry

end NUMINAMATH_GPT_value_of_g_at_neg3_l2188_218888


namespace NUMINAMATH_GPT_remainder_when_160_divided_by_k_l2188_218811

-- Define k to be a positive integer
def positive_integer (n : ℕ) := n > 0

-- Given conditions in the problem
def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def problem_condition (k : ℕ) := positive_integer k ∧ (120 % (k * k) = 12)

-- Prove the main statement
theorem remainder_when_160_divided_by_k (k : ℕ) (h : problem_condition k) : 160 % k = 4 := 
sorry  -- Proof here

end NUMINAMATH_GPT_remainder_when_160_divided_by_k_l2188_218811


namespace NUMINAMATH_GPT_total_distance_is_27_l2188_218808

-- Condition: Renaldo drove 15 kilometers
def renaldo_distance : ℕ := 15

-- Condition: Ernesto drove 7 kilometers more than one-third of Renaldo's distance
def ernesto_distance := (1 / 3 : ℚ) * renaldo_distance + 7

-- Theorem to prove that total distance driven by both men is 27 kilometers
theorem total_distance_is_27 : renaldo_distance + ernesto_distance = 27 := by
  sorry

end NUMINAMATH_GPT_total_distance_is_27_l2188_218808


namespace NUMINAMATH_GPT_infinite_equal_pairs_l2188_218838

theorem infinite_equal_pairs
  (a : ℤ → ℝ)
  (h : ∀ k : ℤ, a k = 1/4 * (a (k - 1) + a (k + 1)))
  (k p : ℤ) (hne : k ≠ p) (heq : a k = a p) :
  ∃ infinite_pairs : ℕ → (ℤ × ℤ), 
  (∀ n : ℕ, (infinite_pairs n).1 ≠ (infinite_pairs n).2) ∧
  (∀ n : ℕ, a (infinite_pairs n).1 = a (infinite_pairs n).2) :=
sorry

end NUMINAMATH_GPT_infinite_equal_pairs_l2188_218838


namespace NUMINAMATH_GPT_tetrahedron_pairs_l2188_218864

theorem tetrahedron_pairs (tetra_edges : ℕ) (h_tetra : tetra_edges = 6) :
  ∀ (num_pairs : ℕ), num_pairs = (tetra_edges * (tetra_edges - 1)) / 2 → num_pairs = 15 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_pairs_l2188_218864


namespace NUMINAMATH_GPT_range_of_a_l2188_218853

variable (x a : ℝ)

-- Definition of α: x > a
def α : Prop := x > a

-- Definition of β: (x - 1) / x > 0
def β : Prop := (x - 1) / x > 0

-- Theorem to prove the range of a
theorem range_of_a (h : α x a → β x) : 1 ≤ a :=
  sorry

end NUMINAMATH_GPT_range_of_a_l2188_218853


namespace NUMINAMATH_GPT_pupils_like_both_l2188_218827

theorem pupils_like_both (total_pupils : ℕ) (likes_pizza : ℕ) (likes_burgers : ℕ)
  (total := 200) (P := 125) (B := 115) :
  (P + B - total_pupils) = 40 :=
by
  sorry

end NUMINAMATH_GPT_pupils_like_both_l2188_218827


namespace NUMINAMATH_GPT_farmer_tomatoes_l2188_218856

theorem farmer_tomatoes (t p l : ℕ) (H1 : t = 97) (H2 : p = 83) : l = t - p → l = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_farmer_tomatoes_l2188_218856


namespace NUMINAMATH_GPT_ratio_sum_2_or_4_l2188_218814

theorem ratio_sum_2_or_4 (a b c d : ℝ) 
  (h1 : a / b + b / c + c / d + d / a = 6)
  (h2 : a / c + b / d + c / a + d / b = 8) : 
  (a / b + c / d = 2) ∨ (a / b + c / d = 4) :=
sorry

end NUMINAMATH_GPT_ratio_sum_2_or_4_l2188_218814


namespace NUMINAMATH_GPT_external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l2188_218863

variables {O₁ O₂ : ℝ} {r R : ℝ}

-- External tangency implies sum of radii equals distance between centers
theorem external_tangency_sum {O₁ O₂ r R : ℝ} (h1 : O₁ ≠ O₂) (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = r + R) : 
  dist O₁ O₂ = r + R :=
sorry

-- Internal tangency implies difference of radii equals distance between centers
theorem internal_tangency_diff {O₁ O₂ r R : ℝ} 
  (h1 : O₁ ≠ O₂) 
  (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = abs (R - r)) : 
  dist O₁ O₂ = abs (R - r) :=
sorry

-- Converse for sum of radii equals distance between centers
theorem converse_sum_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = r + R) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = r + R) :=
sorry

-- Converse for difference of radii equals distance between centers
theorem converse_diff_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = abs (R - r)) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = abs (R - r)) :=
sorry

end NUMINAMATH_GPT_external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l2188_218863


namespace NUMINAMATH_GPT_a_5_value_l2188_218847

noncomputable def seq : ℕ → ℤ
| 0       => 1
| (n + 1) => (seq n) ^ 2 - 1

theorem a_5_value : seq 4 = -1 :=
by
  sorry

end NUMINAMATH_GPT_a_5_value_l2188_218847


namespace NUMINAMATH_GPT_original_triangle_angles_determined_l2188_218805

-- Define the angles of the formed triangle
def formed_triangle_angles : Prop := 
  52 + 61 + 67 = 180

-- Define the angles of the original triangle
def original_triangle_angles (α β γ : ℝ) : Prop := 
  α + β + γ = 180

theorem original_triangle_angles_determined :
  formed_triangle_angles → 
  ∃ α β γ : ℝ, 
    original_triangle_angles α β γ ∧
    α = 76 ∧ β = 58 ∧ γ = 46 :=
by
  sorry

end NUMINAMATH_GPT_original_triangle_angles_determined_l2188_218805


namespace NUMINAMATH_GPT_rectangle_area_with_inscribed_circle_l2188_218896

theorem rectangle_area_with_inscribed_circle (w h r : ℝ)
  (hw : ∀ O : ℝ × ℝ, dist O (w/2, h/2) = r)
  (hw_eq_h : w = h) :
  w * h = 2 * r^2 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_with_inscribed_circle_l2188_218896


namespace NUMINAMATH_GPT_jake_first_test_score_l2188_218876

theorem jake_first_test_score 
  (avg_score : ℕ)
  (n_tests : ℕ)
  (second_test_extra : ℕ)
  (third_test_score : ℕ)
  (x : ℕ) : 
  avg_score = 75 → 
  n_tests = 4 → 
  second_test_extra = 10 → 
  third_test_score = 65 →
  (x + (x + second_test_extra) + third_test_score + third_test_score) / n_tests = avg_score →
  x = 80 := by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_jake_first_test_score_l2188_218876


namespace NUMINAMATH_GPT_find_side_b_of_triangle_l2188_218843

theorem find_side_b_of_triangle
  (A B : Real) (a b : Real)
  (hA : A = Real.pi / 6)
  (hB : B = Real.pi / 4)
  (ha : a = 2) :
  b = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_side_b_of_triangle_l2188_218843


namespace NUMINAMATH_GPT_triangle_type_is_isosceles_l2188_218825

theorem triangle_type_is_isosceles {A B C : ℝ}
  (h1 : A + B + C = π)
  (h2 : ∀ x : ℝ, x^2 - x * (Real.cos A * Real.cos B) + 2 * Real.sin (C / 2)^2 = 0)
  (h3 : ∃ x1 x2 : ℝ, x1 + x2 = Real.cos A * Real.cos B ∧ x1 * x2 = 2 * Real.sin (C / 2)^2 ∧ (x1 + x2 = (x1 * x2) / 2)) :
  A = B ∨ B = C ∨ C = A := 
sorry

end NUMINAMATH_GPT_triangle_type_is_isosceles_l2188_218825


namespace NUMINAMATH_GPT_elgin_money_l2188_218887

theorem elgin_money {A B C D E : ℤ} 
  (h1 : |A - B| = 19) 
  (h2 : |B - C| = 9) 
  (h3 : |C - D| = 5) 
  (h4 : |D - E| = 4) 
  (h5 : |E - A| = 11) 
  (h6 : A + B + C + D + E = 60) : 
  E = 10 := 
sorry

end NUMINAMATH_GPT_elgin_money_l2188_218887


namespace NUMINAMATH_GPT_triangle_area_is_180_l2188_218878

theorem triangle_area_is_180 {a b c : ℕ} (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) 
  (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 : ℚ) * a * b = 180 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_180_l2188_218878


namespace NUMINAMATH_GPT_maximize_volume_l2188_218855

-- Define the given dimensions
def length := 90
def width := 48

-- Define the volume function based on the height h
def volume (h : ℝ) : ℝ := h * (length - 2 * h) * (width - 2 * h)

-- Define the height that maximizes the volume
def optimal_height := 10

-- Define the maximum volume obtained at the optimal height
def max_volume := 19600

-- State the proof problem
theorem maximize_volume : 
  (∃ h : ℝ, volume h ≤ volume optimal_height) ∧
  volume optimal_height = max_volume := 
by
  sorry

end NUMINAMATH_GPT_maximize_volume_l2188_218855


namespace NUMINAMATH_GPT_maximum_sum_S6_l2188_218802

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + (n - 1) * d

def sum_arithmetic_sequence (a d : α) (n : ℕ) : α :=
  (n : α) / 2 * (2 * a + (n - 1) * d)

theorem maximum_sum_S6 (a d : α)
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 10 < 0)
  (h2 : sum_arithmetic_sequence a d 11 > 0) :
  ∀ n : ℕ, sum_arithmetic_sequence a d n ≤ sum_arithmetic_sequence a d 6 :=
by sorry

end NUMINAMATH_GPT_maximum_sum_S6_l2188_218802


namespace NUMINAMATH_GPT_percent_within_one_standard_deviation_l2188_218837

variable (m d : ℝ)
variable (distribution : ℝ → ℝ)
variable (symmetric_about_mean : ∀ x, distribution (m + x) = distribution (m - x))
variable (percent_less_than_m_plus_d : distribution (m + d) = 0.84)

theorem percent_within_one_standard_deviation :
  distribution (m + d) - distribution (m - d) = 0.68 :=
sorry

end NUMINAMATH_GPT_percent_within_one_standard_deviation_l2188_218837


namespace NUMINAMATH_GPT_positive_integer_a_l2188_218849

theorem positive_integer_a (a : ℕ) (h1 : 0 < a) (h2 : ∃ (k : ℤ), (2 * a + 8) = k * (a + 1)) :
  a = 1 ∨ a = 2 ∨ a = 5 :=
by sorry

end NUMINAMATH_GPT_positive_integer_a_l2188_218849


namespace NUMINAMATH_GPT_man_is_older_by_16_l2188_218804

variable (M S : ℕ)

-- Condition: The present age of the son is 14.
def son_age := S = 14

-- Condition: In two years, the man's age will be twice the son's age.
def age_relation := M + 2 = 2 * (S + 2)

-- Theorem: Prove that the man is 16 years older than his son.
theorem man_is_older_by_16 (h1 : son_age S) (h2 : age_relation M S) : M - S = 16 := 
sorry

end NUMINAMATH_GPT_man_is_older_by_16_l2188_218804


namespace NUMINAMATH_GPT_initial_percentage_proof_l2188_218850

-- Defining the initial percentage of water filled in the container
def initial_percentage (capacity add amount_filled : ℕ) : ℕ :=
  (amount_filled * 100) / capacity

-- The problem constraints
theorem initial_percentage_proof : initial_percentage 120 48 (3 * 120 / 4 - 48) = 35 := by
  -- We need to show that the initial percentage is 35%
  sorry

end NUMINAMATH_GPT_initial_percentage_proof_l2188_218850


namespace NUMINAMATH_GPT_fencing_rate_l2188_218835

/-- Given a circular field of diameter 20 meters and a total cost of fencing of Rs. 94.24777960769379,
    prove that the rate per meter for the fencing is Rs. 1.5. -/
theorem fencing_rate 
  (d : ℝ) (cost : ℝ) (π : ℝ) (rate : ℝ)
  (hd : d = 20)
  (hcost : cost = 94.24777960769379)
  (hπ : π = 3.14159)
  (Circumference : ℝ := π * d)
  (Rate : ℝ := cost / Circumference) : 
  rate = 1.5 :=
sorry

end NUMINAMATH_GPT_fencing_rate_l2188_218835


namespace NUMINAMATH_GPT_johns_quarters_l2188_218889

variable (x : ℕ)  -- Number of quarters John has

def number_of_dimes : ℕ := x + 3  -- Number of dimes
def number_of_nickels : ℕ := x - 6  -- Number of nickels

theorem johns_quarters (h : x + (x + 3) + (x - 6) = 63) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_johns_quarters_l2188_218889


namespace NUMINAMATH_GPT_sum_of_squares_of_sides_l2188_218832

-- Definition: A cyclic quadrilateral with perpendicular diagonals inscribed in a circle
structure CyclicQuadrilateral (R : ℝ) :=
  (m n k t : ℝ) -- sides of the quadrilateral
  (perpendicular_diagonals : true) -- diagonals are perpendicular (trivial placeholder)
  (radius : ℝ := R) -- Radius of the circumscribed circle

-- The theorem to prove: The sum of the squares of the sides of the quadrilateral is 8R^2
theorem sum_of_squares_of_sides (R : ℝ) (quad : CyclicQuadrilateral R) :
  quad.m ^ 2 + quad.n ^ 2 + quad.k ^ 2 + quad.t ^ 2 = 8 * R^2 := 
by sorry

end NUMINAMATH_GPT_sum_of_squares_of_sides_l2188_218832


namespace NUMINAMATH_GPT_part1_part2_l2188_218807

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : f x ≤ x^2 :=
sorry

theorem part2 (x : ℝ) (hx : x > 0) (c : ℝ) (hc : c ≥ -1) : f x ≤ 2 * x + c :=
sorry

end NUMINAMATH_GPT_part1_part2_l2188_218807
