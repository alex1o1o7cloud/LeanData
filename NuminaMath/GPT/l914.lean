import Mathlib

namespace NUMINAMATH_GPT_correct_calculation_l914_91452

theorem correct_calculation (a b : ℝ) :
  2 * a^2 * b - 3 * a^2 * b = -a^2 * b ∧
  ¬ (a^3 * a^4 = a^12) ∧
  ¬ ((-2 * a^2 * b)^3 = -6 * a^6 * b^3) ∧
  ¬ ((a + b)^2 = a^2 + b^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l914_91452


namespace NUMINAMATH_GPT_danielles_rooms_l914_91499

variable (rooms_heidi rooms_danielle : ℕ)

theorem danielles_rooms 
  (h1 : rooms_heidi = 3 * rooms_danielle)
  (h2 : 2 = 1 / 9 * rooms_heidi) :
  rooms_danielle = 6 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_danielles_rooms_l914_91499


namespace NUMINAMATH_GPT_filling_tank_ratio_l914_91456

theorem filling_tank_ratio :
  ∀ (t : ℝ),
    (1 / 40) * t + (1 / 24) * (29.999999999999993 - t) = 1 →
    t / 29.999999999999993 = 1 / 2 :=
by
  intro t
  intro H
  sorry

end NUMINAMATH_GPT_filling_tank_ratio_l914_91456


namespace NUMINAMATH_GPT_jenny_ate_65_chocolates_l914_91425

noncomputable def chocolates_eaten_by_Jenny : ℕ :=
  let chocolates_mike := 20
  let chocolates_john := chocolates_mike / 2
  let combined_chocolates := chocolates_mike + chocolates_john
  let twice_combined_chocolates := 2 * combined_chocolates
  5 + twice_combined_chocolates

theorem jenny_ate_65_chocolates :
  chocolates_eaten_by_Jenny = 65 :=
by
  -- Skipping the proof details
  sorry

end NUMINAMATH_GPT_jenny_ate_65_chocolates_l914_91425


namespace NUMINAMATH_GPT_average_weight_of_all_children_l914_91488

theorem average_weight_of_all_children 
  (Boys: ℕ) (Girls: ℕ) (Additional: ℕ)
  (avgWeightBoys: ℚ) (avgWeightGirls: ℚ) (avgWeightAdditional: ℚ) :
  Boys = 8 ∧ Girls = 5 ∧ Additional = 3 ∧ 
  avgWeightBoys = 160 ∧ avgWeightGirls = 130 ∧ avgWeightAdditional = 145 →
  ((Boys * avgWeightBoys + Girls * avgWeightGirls + Additional * avgWeightAdditional) / (Boys + Girls + Additional) = 148) :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_weight_of_all_children_l914_91488


namespace NUMINAMATH_GPT_theater_ticket_sales_l914_91424

-- Definitions of the given constants and initialization
def R : ℕ := 25

-- Conditions based on the problem statement
def condition_horror (H : ℕ) := H = 3 * R + 18
def condition_action (A : ℕ) := A = 2 * R
def condition_comedy (C H : ℕ) := 4 * H = 5 * C

-- Desired outcomes based on the solutions
def desired_horror := 93
def desired_action := 50
def desired_comedy := 74

theorem theater_ticket_sales
  (H A C : ℕ)
  (h1 : condition_horror H)
  (h2 : condition_action A)
  (h3 : condition_comedy C H)
  : H = desired_horror ∧ A = desired_action ∧ C = desired_comedy :=
by {
    sorry
}

end NUMINAMATH_GPT_theater_ticket_sales_l914_91424


namespace NUMINAMATH_GPT_inequality_solution_range_l914_91449

theorem inequality_solution_range (a : ℝ) :
  (∃ (x : ℝ), |x + 1| - |x - 2| < a^2 - 4 * a) → (a > 3 ∨ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_range_l914_91449


namespace NUMINAMATH_GPT_total_length_infinite_sum_l914_91473

-- Define the infinite sums
noncomputable def S1 : ℝ := ∑' n : ℕ, (1 / (3^n))
noncomputable def S2 : ℝ := (∑' n : ℕ, (1 / (5^n))) * Real.sqrt 3
noncomputable def S3 : ℝ := (∑' n : ℕ, (1 / (7^n))) * Real.sqrt 5

-- Define the total length
noncomputable def total_length : ℝ := S1 + S2 + S3

-- The statement of the theorem
theorem total_length_infinite_sum : total_length = (3 / 2) + (Real.sqrt 3 / 4) + (Real.sqrt 5 / 6) :=
by
  sorry

end NUMINAMATH_GPT_total_length_infinite_sum_l914_91473


namespace NUMINAMATH_GPT_paint_coverage_is_10_l914_91450

noncomputable def paintCoverage (cost_per_quart : ℝ) (cube_edge_length : ℝ) (total_cost : ℝ) : ℝ :=
  let total_surface_area := 6 * (cube_edge_length ^ 2)
  let number_of_quarts := total_cost / cost_per_quart
  total_surface_area / number_of_quarts

theorem paint_coverage_is_10 :
  paintCoverage 3.2 10 192 = 10 :=
by
  sorry

end NUMINAMATH_GPT_paint_coverage_is_10_l914_91450


namespace NUMINAMATH_GPT_angle_A_value_sin_BC_value_l914_91402

open Real

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ 
  A + B + C = π 

theorem angle_A_value (A B C : ℝ) (h : triangleABC a b c A B C) (h1 : cos 2 * A - 3 * cos (B + C) = 1) : 
  A = π / 3 :=
sorry

theorem sin_BC_value (A B C S b c : ℝ) (h : triangleABC a b c A B C)
  (hA : A = π / 3) (hS : S = 5 * sqrt 3) (hb : b = 5) : 
  sin B * sin C = 5 / 7 :=
sorry

end NUMINAMATH_GPT_angle_A_value_sin_BC_value_l914_91402


namespace NUMINAMATH_GPT_staircase_perimeter_l914_91441

theorem staircase_perimeter (area : ℝ) (side_length : ℝ) (num_sides : ℕ) (right_angles : Prop) :
  area = 85 ∧ side_length = 1 ∧ num_sides = 10 ∧ right_angles → 
  ∃ perimeter : ℝ, perimeter = 30.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_staircase_perimeter_l914_91441


namespace NUMINAMATH_GPT_proof_problem_l914_91443

noncomputable def a : ℝ := 2 - 0.5
noncomputable def b : ℝ := Real.log (Real.pi) / Real.log 3
noncomputable def c : ℝ := Real.log 2 / Real.log 4

theorem proof_problem : b > a ∧ a > c := 
by
sorry

end NUMINAMATH_GPT_proof_problem_l914_91443


namespace NUMINAMATH_GPT_common_difference_arithmetic_progression_l914_91410

theorem common_difference_arithmetic_progression {n : ℕ} (x y : ℝ) (a : ℕ → ℝ) 
  (h : ∀ k : ℕ, k ≤ n → a (k+1) = a k + (y - x) / (n + 1)) 
  : (∃ d : ℝ, ∀ i : ℕ, i ≤ n + 1 → a (i+1) = x + i * d) ∧ d = (y - x) / (n + 1) := 
by
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_progression_l914_91410


namespace NUMINAMATH_GPT_henry_has_30_more_lollipops_than_alison_l914_91486

noncomputable def num_lollipops_alison : ℕ := 60
noncomputable def num_lollipops_diane : ℕ := 2 * num_lollipops_alison
noncomputable def total_num_days : ℕ := 6
noncomputable def num_lollipops_per_day : ℕ := 45
noncomputable def total_lollipops : ℕ := total_num_days * num_lollipops_per_day
noncomputable def num_lollipops_total_ad : ℕ := num_lollipops_alison + num_lollipops_diane
noncomputable def num_lollipops_henry : ℕ := total_lollipops - num_lollipops_total_ad
noncomputable def lollipops_diff_henry_alison : ℕ := num_lollipops_henry - num_lollipops_alison

theorem henry_has_30_more_lollipops_than_alison :
  lollipops_diff_henry_alison = 30 :=
by
  unfold lollipops_diff_henry_alison
  unfold num_lollipops_henry
  unfold num_lollipops_total_ad
  unfold total_lollipops
  sorry

end NUMINAMATH_GPT_henry_has_30_more_lollipops_than_alison_l914_91486


namespace NUMINAMATH_GPT_rectangle_area_from_square_l914_91491

theorem rectangle_area_from_square 
  (square_area : ℕ) 
  (width_rect : ℕ) 
  (length_rect : ℕ) 
  (h_square_area : square_area = 36)
  (h_width_rect : width_rect * width_rect = square_area)
  (h_length_rect : length_rect = 3 * width_rect) :
  width_rect * length_rect = 108 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_from_square_l914_91491


namespace NUMINAMATH_GPT_remove_candies_even_distribution_l914_91474

theorem remove_candies_even_distribution (candies friends : ℕ) (h_candies : candies = 30) (h_friends : friends = 4) :
  ∃ k, candies - k % friends = 0 ∧ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_remove_candies_even_distribution_l914_91474


namespace NUMINAMATH_GPT_relationship_among_three_numbers_l914_91440

noncomputable def M (a b : ℝ) : ℝ := a^b
noncomputable def N (a b : ℝ) : ℝ := Real.log a / Real.log b
noncomputable def P (a b : ℝ) : ℝ := b^a

theorem relationship_among_three_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : N a b < M a b ∧ M a b < P a b := 
by
  sorry

end NUMINAMATH_GPT_relationship_among_three_numbers_l914_91440


namespace NUMINAMATH_GPT_smallest_positive_number_div_conditions_is_perfect_square_l914_91465

theorem smallest_positive_number_div_conditions_is_perfect_square :
  ∃ n : ℕ,
    (n % 11 = 10) ∧
    (n % 10 = 9) ∧
    (n % 9 = 8) ∧
    (n % 8 = 7) ∧
    (n % 7 = 6) ∧
    (n % 6 = 5) ∧
    (n % 5 = 4) ∧
    (n % 4 = 3) ∧
    (n % 3 = 2) ∧
    (n % 2 = 1) ∧
    (∃ k : ℕ, n = k * k) ∧
    n = 2782559 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_number_div_conditions_is_perfect_square_l914_91465


namespace NUMINAMATH_GPT_proof_y_minus_x_l914_91470

theorem proof_y_minus_x (x y : ℤ) (h1 : x + y = 540) (h2 : x = (4 * y) / 5) : y - x = 60 :=
sorry

end NUMINAMATH_GPT_proof_y_minus_x_l914_91470


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l914_91438

theorem geometric_sequence_first_term (S_3 S_6 : ℝ) (a_1 q : ℝ)
  (hS3 : S_3 = 6) (hS6 : S_6 = 54)
  (hS3_def : S_3 = a_1 * (1 - q^3) / (1 - q))
  (hS6_def : S_6 = a_1 * (1 - q^6) / (1 - q)) :
  a_1 = 6 / 7 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l914_91438


namespace NUMINAMATH_GPT_investment_B_l914_91420

theorem investment_B {x : ℝ} :
  let a_investment := 6300
  let c_investment := 10500
  let total_profit := 12100
  let a_share_profit := 3630
  (6300 / (6300 + x + 10500) = 3630 / 12100) →
  x = 13650 :=
by { sorry }

end NUMINAMATH_GPT_investment_B_l914_91420


namespace NUMINAMATH_GPT_find_second_largest_element_l914_91457

open List

theorem find_second_largest_element 
(a1 a2 a3 a4 a5 : ℕ) 
(h_pos : 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4 ∧ 0 < a5) 
(h_sorted : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) 
(h_mean : (a1 + a2 + a3 + a4 + a5) / 5 = 15) 
(h_range : a5 - a1 = 24) 
(h_mode : a2 = 10 ∧ a3 = 10) 
(h_median : a3 = 10) 
(h_three_diff : (a1 ≠ a2 ∨ a1 ≠ a3 ∨ a1 ≠ a4 ∨ a1 ≠ a5) ∧ (a4 ≠ a5)) :
a4 = 11 :=
sorry

end NUMINAMATH_GPT_find_second_largest_element_l914_91457


namespace NUMINAMATH_GPT_dad_strawberries_weight_l914_91417

-- Definitions for the problem
def weight_marco := 15
def total_weight := 37

-- Theorem statement
theorem dad_strawberries_weight :
  (total_weight - weight_marco = 22) :=
by
  sorry

end NUMINAMATH_GPT_dad_strawberries_weight_l914_91417


namespace NUMINAMATH_GPT_quadratic_real_roots_iff_l914_91469

/-- For the quadratic equation x^2 + 3x + m = 0 to have two real roots,
    the value of m must satisfy m ≤ 9/4. -/
theorem quadratic_real_roots_iff (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x2 = m ∧ x1 + x2 = -3) ↔ m ≤ 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_iff_l914_91469


namespace NUMINAMATH_GPT_Euclid1976_PartA_Problem8_l914_91442

theorem  Euclid1976_PartA_Problem8 (a b c m n : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h4 : Polynomial.eval (-a) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0)
  (h5 : Polynomial.eval (-b) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0) :
  n = -1 :=
sorry

end NUMINAMATH_GPT_Euclid1976_PartA_Problem8_l914_91442


namespace NUMINAMATH_GPT_meaningful_expression_iff_l914_91430

theorem meaningful_expression_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 3))) ↔ x > 3 := by
  sorry

end NUMINAMATH_GPT_meaningful_expression_iff_l914_91430


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l914_91497

open Real

theorem relationship_between_a_and_b
   (a b : ℝ)
   (ha : 0 < a ∧ a < 1)
   (hb : 0 < b ∧ b < 1)
   (hab : (1 - a) * b > 1 / 4) :
   a < b := 
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l914_91497


namespace NUMINAMATH_GPT_price_of_36kgs_l914_91408

namespace Apples

-- Define the parameters l and q
variables (l q : ℕ)

-- Define the conditions
def cost_first_30kgs (l : ℕ) : ℕ := 30 * l
def cost_first_15kgs : ℕ := 150
def cost_33kgs (l q : ℕ) : ℕ := (30 * l) + (3 * q)
def cost_36kgs (l q : ℕ) : ℕ := (30 * l) + (6 * q)

-- Define the hypothesis for l and q based on given conditions
axiom l_value (h1 : cost_first_15kgs = 150) : l = 10
axiom q_value (h2 : cost_33kgs l q = 333) : q = 11

-- Prove the price of 36 kilograms of apples
theorem price_of_36kgs (h1 : cost_first_15kgs = 150) (h2 : cost_33kgs l q = 333) : cost_36kgs l q = 366 :=
sorry

end Apples

end NUMINAMATH_GPT_price_of_36kgs_l914_91408


namespace NUMINAMATH_GPT_find_rate_per_kg_grapes_l914_91495

-- Define the main conditions
def rate_per_kg_mango := 55
def total_payment := 985
def kg_grapes := 7
def kg_mangoes := 9

-- Define the problem statement
theorem find_rate_per_kg_grapes (G : ℝ) : 
  (kg_grapes * G + kg_mangoes * rate_per_kg_mango = total_payment) → 
  G = 70 :=
by
  sorry

end NUMINAMATH_GPT_find_rate_per_kg_grapes_l914_91495


namespace NUMINAMATH_GPT_g_at_0_eq_1_l914_91453

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x * g y
axiom g_deriv_at_0 : deriv g 0 = 2

theorem g_at_0_eq_1 : g 0 = 1 :=
by
  sorry

end NUMINAMATH_GPT_g_at_0_eq_1_l914_91453


namespace NUMINAMATH_GPT_age_problem_l914_91421

-- Define the conditions
variables (a b c : ℕ)

-- Assumptions based on conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 37) : b = 14 :=
by {
  sorry   -- Placeholder for the detailed proof
}

end NUMINAMATH_GPT_age_problem_l914_91421


namespace NUMINAMATH_GPT_smallest_square_area_l914_91489

theorem smallest_square_area (n : ℕ) (h : ∃ m : ℕ, 14 * n = m ^ 2) : n = 14 :=
sorry

end NUMINAMATH_GPT_smallest_square_area_l914_91489


namespace NUMINAMATH_GPT_integer_solutions_l914_91427

theorem integer_solutions :
  ∃ (a b c : ℤ), a + b + c = 24 ∧ a^2 + b^2 + c^2 = 210 ∧ a * b * c = 440 ∧
    (a = 5 ∧ b = 8 ∧ c = 11) ∨ (a = 5 ∧ b = 11 ∧ c = 8) ∨ 
    (a = 8 ∧ b = 5 ∧ c = 11) ∨ (a = 8 ∧ b = 11 ∧ c = 5) ∨
    (a = 11 ∧ b = 5 ∧ c = 8) ∨ (a = 11 ∧ b = 8 ∧ c = 5) :=
sorry

end NUMINAMATH_GPT_integer_solutions_l914_91427


namespace NUMINAMATH_GPT_ada_original_seat_l914_91455

-- Define the problem conditions
def initial_seats : List ℕ := [1, 2, 3, 4, 5]  -- seat numbers

def bea_move (seat : ℕ) : ℕ := seat + 2  -- Bea moves 2 seats to the right
def ceci_move (seat : ℕ) : ℕ := seat - 1  -- Ceci moves 1 seat to the left
def switch (seats : (ℕ × ℕ)) : (ℕ × ℕ) := (seats.2, seats.1)  -- Dee and Edie switch seats

-- The final seating positions (end seats are 1 or 5 for Ada)
axiom ada_end_seat : ∃ final_seat : ℕ, final_seat ∈ [1, 5]  -- Ada returns to an end seat

-- Prove Ada was originally sitting in seat 2
theorem ada_original_seat (final_seat : ℕ) (h₁ : ∃ (s₁ s₂ : ℕ), s₁ ≠ s₂ ∧ bea_move s₁ ≠ final_seat ∧ ceci_move s₂ ≠ final_seat ∧ switch (s₁, s₂).2 ≠ final_seat) : 2 ∈ initial_seats :=
by
  sorry

end NUMINAMATH_GPT_ada_original_seat_l914_91455


namespace NUMINAMATH_GPT_range_of_2a_plus_b_l914_91472

variable {a b c A B C : Real}
variable {sin cos : Real → Real}

theorem range_of_2a_plus_b (h1 : a^2 + b^2 + ab = 4) (h2 : c = 2) (h3 : a = c * sin A / sin C) (h4 : b = c * sin B / sin C) :
  2 < 2 * a + b ∧ 2 * a + b < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_2a_plus_b_l914_91472


namespace NUMINAMATH_GPT_average_is_correct_l914_91419

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

def sum_of_numbers : ℕ := numbers.sum
def count_of_numbers : ℕ := numbers.length
def average_of_numbers : ℚ := sum_of_numbers / count_of_numbers

theorem average_is_correct : average_of_numbers = 1380 := 
by 
  -- Here, you would normally put the proof steps.
  sorry

end NUMINAMATH_GPT_average_is_correct_l914_91419


namespace NUMINAMATH_GPT_lea_total_cost_example_l914_91485

/-- Léa bought one book for $16, three binders for $2 each, and six notebooks for $1 each. -/
def total_cost (book_cost binders_cost notebooks_cost : ℕ) : ℕ :=
  book_cost + binders_cost + notebooks_cost

/-- Given the individual costs, prove the total cost of Léa's purchases is $28. -/
theorem lea_total_cost_example : total_cost 16 (3 * 2) (6 * 1) = 28 := by
  sorry

end NUMINAMATH_GPT_lea_total_cost_example_l914_91485


namespace NUMINAMATH_GPT_altitude_product_difference_eq_zero_l914_91437

variables (A B C P Q H : Type*) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited H]
variable {HP HQ BP PC AQ QC AH BH : ℝ}

-- Given conditions
axiom altitude_intersects_at_H : true
axiom HP_val : HP = 3
axiom HQ_val : HQ = 7

-- Statement to prove
theorem altitude_product_difference_eq_zero (h_BP_PC : BP * PC = 3 / (AH + 3))
                                           (h_AQ_QC : AQ * QC = 7 / (BH + 7))
                                           (h_AH_BQ_ratio : AH / BH = 3 / 7) :
  (BP * PC) - (AQ * QC) = 0 :=
by sorry

end NUMINAMATH_GPT_altitude_product_difference_eq_zero_l914_91437


namespace NUMINAMATH_GPT_geometric_sequence_product_l914_91483

theorem geometric_sequence_product (a₁ aₙ : ℝ) (n : ℕ) (hn : n > 0) (number_of_terms : n ≥ 1) :
  -- Conditions: First term, last term, number of terms
  ∃ P : ℝ, P = (a₁ * aₙ) ^ (n / 2) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l914_91483


namespace NUMINAMATH_GPT_amount_charged_for_kids_l914_91481

theorem amount_charged_for_kids (K A: ℝ) (H1: A = 2 * K) (H2: 8 * K + 10 * A = 84) : K = 3 :=
by
  sorry

end NUMINAMATH_GPT_amount_charged_for_kids_l914_91481


namespace NUMINAMATH_GPT_James_watch_time_l914_91403

def Jeopardy_length : ℕ := 20
def Wheel_of_Fortune_length : ℕ := Jeopardy_length * 2
def Jeopardy_episodes : ℕ := 2
def Wheel_of_Fortune_episodes : ℕ := 2

theorem James_watch_time :
  (Jeopardy_episodes * Jeopardy_length + Wheel_of_Fortune_episodes * Wheel_of_Fortune_length) / 60 = 2 :=
by
  sorry

end NUMINAMATH_GPT_James_watch_time_l914_91403


namespace NUMINAMATH_GPT_scientific_notation_of_935million_l914_91404

theorem scientific_notation_of_935million :
  935000000 = 9.35 * 10 ^ 8 :=
  sorry

end NUMINAMATH_GPT_scientific_notation_of_935million_l914_91404


namespace NUMINAMATH_GPT_husband_additional_payment_l914_91475

theorem husband_additional_payment (total_medical_cost : ℝ) (total_salary : ℝ) 
                                  (half_medical_cost : ℝ) (deduction_from_salary : ℝ) 
                                  (remaining_salary : ℝ) (total_payment : ℝ)
                                  (each_share : ℝ) (amount_paid_by_husband : ℝ) : 
                                  
                                  total_medical_cost = 128 →
                                  total_salary = 160 →
                                  half_medical_cost = total_medical_cost / 2 →
                                  deduction_from_salary = half_medical_cost →
                                  remaining_salary = total_salary - deduction_from_salary →
                                  total_payment = remaining_salary + half_medical_cost →
                                  each_share = total_payment / 2 →
                                  amount_paid_by_husband = 64 →
                                  (each_share - amount_paid_by_husband) = 16 := by
  sorry

end NUMINAMATH_GPT_husband_additional_payment_l914_91475


namespace NUMINAMATH_GPT_total_green_ducks_percentage_l914_91418

def ducks_in_park_A : ℕ := 200
def green_percentage_A : ℕ := 25

def ducks_in_park_B : ℕ := 350
def green_percentage_B : ℕ := 20

def ducks_in_park_C : ℕ := 120
def green_percentage_C : ℕ := 50

def ducks_in_park_D : ℕ := 60
def green_percentage_D : ℕ := 25

def ducks_in_park_E : ℕ := 500
def green_percentage_E : ℕ := 30

theorem total_green_ducks_percentage (green_ducks_A green_ducks_B green_ducks_C green_ducks_D green_ducks_E total_ducks : ℕ)
  (h_A : green_ducks_A = ducks_in_park_A * green_percentage_A / 100)
  (h_B : green_ducks_B = ducks_in_park_B * green_percentage_B / 100)
  (h_C : green_ducks_C = ducks_in_park_C * green_percentage_C / 100)
  (h_D : green_ducks_D = ducks_in_park_D * green_percentage_D / 100)
  (h_E : green_ducks_E = ducks_in_park_E * green_percentage_E / 100)
  (h_total_ducks : total_ducks = ducks_in_park_A + ducks_in_park_B + ducks_in_park_C + ducks_in_park_D + ducks_in_park_E) :
  (green_ducks_A + green_ducks_B + green_ducks_C + green_ducks_D + green_ducks_E) * 100 / total_ducks = 2805 / 100 :=
by sorry

end NUMINAMATH_GPT_total_green_ducks_percentage_l914_91418


namespace NUMINAMATH_GPT_price_reduction_correct_l914_91487

noncomputable def percentage_reduction (x : ℝ) : Prop :=
  (5000 * (1 - x)^2 = 4050)

theorem price_reduction_correct {x : ℝ} (h : percentage_reduction x) : x = 0.1 :=
by
  -- proof is omitted, so we use sorry
  sorry

end NUMINAMATH_GPT_price_reduction_correct_l914_91487


namespace NUMINAMATH_GPT_incorrect_statement_D_l914_91478

-- Definitions based on conditions
def length_of_spring (x : ℝ) : ℝ := 8 + 0.5 * x

-- Incorrect Statement (to be proved as incorrect)
def statement_D_incorrect : Prop :=
  ¬ (length_of_spring 30 = 23)

-- Main theorem statement
theorem incorrect_statement_D : statement_D_incorrect :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_D_l914_91478


namespace NUMINAMATH_GPT_find_LP_l914_91498

variables (A B C K L P M : Type) 
variables {AC BC AK CK CL AM LP : ℕ}

-- Defining the given conditions
def conditions (AC BC AK CK : ℕ) (AM : ℕ) :=
  AC = 360 ∧ BC = 240 ∧ AK = CK ∧ AK = 180 ∧ AM = 144

-- The theorem statement: proving LP equals 57.6
theorem find_LP (h : conditions 360 240 180 180 144) : LP = 576 / 10 := 
by sorry

end NUMINAMATH_GPT_find_LP_l914_91498


namespace NUMINAMATH_GPT_inverse_proposition_false_l914_91496

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a = b → abs a = abs b

-- Define the inverse proposition
def inverse_proposition (a b : ℝ) : Prop :=
  abs a = abs b → a = b

-- The theorem to prove
theorem inverse_proposition_false : ∃ (a b : ℝ), abs a = abs b ∧ a ≠ b :=
sorry

end NUMINAMATH_GPT_inverse_proposition_false_l914_91496


namespace NUMINAMATH_GPT_press_t_denomination_l914_91405

def press_f_rate_per_minute := 1000
def press_t_rate_per_minute := 200
def time_in_seconds := 3
def f_denomination := 5
def additional_amount := 50

theorem press_t_denomination : 
  ∃ (x : ℝ), 
  (3 * (5 * (1000 / 60))) = (3 * (x * (200 / 60)) + 50) → 
  x = 20 := 
by 
  -- Proof logic here
  sorry

end NUMINAMATH_GPT_press_t_denomination_l914_91405


namespace NUMINAMATH_GPT_minimum_value_condition_l914_91409

def f (a x : ℝ) : ℝ := -x^3 + 0.5 * (a + 3) * x^2 - a * x - 1

theorem minimum_value_condition (a : ℝ) (h : a ≥ 3) : 
  (∃ x₀ : ℝ, f a x₀ < f a 1) ∨ (f a 1 > f a ((a/3))) := 
sorry

end NUMINAMATH_GPT_minimum_value_condition_l914_91409


namespace NUMINAMATH_GPT_cos_alpha_l914_91401

-- Define the conditions
variable (α : Real)
variable (x y r : Real)
-- Given the point (-3, 4)
def point_condition (x : Real) (y : Real) : Prop := x = -3 ∧ y = 4

-- Define r as the distance
def radius_condition (x y r : Real) : Prop := r = Real.sqrt (x ^ 2 + y ^ 2)

-- Prove that cos α and cos 2α are the given values
theorem cos_alpha (α : Real) (x y r : Real) (h1 : point_condition x y) (h2 : radius_condition x y r) :
  Real.cos α = -3 / 5 ∧ Real.cos (2 * α) = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_l914_91401


namespace NUMINAMATH_GPT_ratio_of_circle_areas_l914_91494

variable (S L A : ℝ)

theorem ratio_of_circle_areas 
  (h1 : A = (3 / 5) * S)
  (h2 : A = (6 / 25) * L)
  : S / L = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_circle_areas_l914_91494


namespace NUMINAMATH_GPT_arithmetic_sequence_10th_term_l914_91423

theorem arithmetic_sequence_10th_term (a d : ℤ) :
    (a + 4 * d = 26) →
    (a + 7 * d = 50) →
    (a + 9 * d = 66) := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_10th_term_l914_91423


namespace NUMINAMATH_GPT_find_missing_fraction_l914_91447

def f1 := 1/3
def f2 := 1/2
def f3 := 1/5
def f4 := 1/4
def f5 := -9/20
def f6 := -9/20
def total_sum := 45/100
def missing_fraction := 1/15

theorem find_missing_fraction : f1 + f2 + f3 + f4 + f5 + f6 + missing_fraction = total_sum :=
by
  sorry

end NUMINAMATH_GPT_find_missing_fraction_l914_91447


namespace NUMINAMATH_GPT_sum_of_fourth_powers_eq_square_of_sum_of_squares_l914_91426

theorem sum_of_fourth_powers_eq_square_of_sum_of_squares 
  (x1 x2 x3 : ℝ) (p q n : ℝ)
  (h1 : x1^3 + p*x1^2 + q*x1 + n = 0)
  (h2 : x2^3 + p*x2^2 + q*x2 + n = 0)
  (h3 : x3^3 + p*x3^2 + q*x3 + n = 0)
  (h_rel : q^2 = 2 * n * p) :
  x1^4 + x2^4 + x3^4 = (x1^2 + x2^2 + x3^2)^2 := 
sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_eq_square_of_sum_of_squares_l914_91426


namespace NUMINAMATH_GPT_average_of_remaining_two_nums_l914_91454

theorem average_of_remaining_two_nums (S S4 : ℕ) (h1 : S / 6 = 8) (h2 : S4 / 4 = 5) :
  ((S - S4) / 2 = 14) :=
by 
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_nums_l914_91454


namespace NUMINAMATH_GPT_winner_exceeds_second_opponent_l914_91431

theorem winner_exceeds_second_opponent
  (total_votes : ℕ)
  (votes_winner : ℕ)
  (votes_second : ℕ)
  (votes_third : ℕ)
  (votes_fourth : ℕ) 
  (h_votes_sum : total_votes = votes_winner + votes_second + votes_third + votes_fourth)
  (h_total_votes : total_votes = 963) 
  (h_winner_votes : votes_winner = 195) 
  (h_second_votes : votes_second = 142) 
  (h_third_votes : votes_third = 116) 
  (h_fourth_votes : votes_fourth = 90) :
  votes_winner - votes_second = 53 := by
  sorry

end NUMINAMATH_GPT_winner_exceeds_second_opponent_l914_91431


namespace NUMINAMATH_GPT_molecular_weight_of_BaBr2_l914_91492

theorem molecular_weight_of_BaBr2 
    (atomic_weight_Ba : ℝ)
    (atomic_weight_Br : ℝ)
    (moles : ℝ)
    (hBa : atomic_weight_Ba = 137.33)
    (hBr : atomic_weight_Br = 79.90) 
    (hmol : moles = 8) :
    (atomic_weight_Ba + 2 * atomic_weight_Br) * moles = 2377.04 :=
by 
  sorry

end NUMINAMATH_GPT_molecular_weight_of_BaBr2_l914_91492


namespace NUMINAMATH_GPT_maximal_value_of_product_l914_91468

theorem maximal_value_of_product (m n : ℤ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (1 < x1 ∧ x1 < 3) ∧ (1 < x2 ∧ x2 < 3) ∧ 
    ∀ x : ℝ, (10 * x^2 + m * x + n) = 10 * (x - x1) * (x - x2)) :
  (∃ f1 f3 : ℝ, f1 = 10 * (1 - x1) * (1 - x2) ∧ f3 = 10 * (3 - x1) * (3 - x2) ∧ (f1 * f3 = 99)) := 
sorry

end NUMINAMATH_GPT_maximal_value_of_product_l914_91468


namespace NUMINAMATH_GPT_lines_intersect_l914_91429

structure Point where
  x : ℝ
  y : ℝ

def line1 (t : ℝ) : Point :=
  ⟨1 + 2 * t, 4 - 3 * t⟩

def line2 (u : ℝ) : Point :=
  ⟨5 + 4 * u, -2 - 5 * u⟩

theorem lines_intersect (x y t u : ℝ) 
  (h1 : x = 1 + 2 * t)
  (h2 : y = 4 - 3 * t)
  (h3 : x = 5 + 4 * u)
  (h4 : y = -2 - 5 * u) :
  x = 5 ∧ y = -2 := 
sorry

end NUMINAMATH_GPT_lines_intersect_l914_91429


namespace NUMINAMATH_GPT_nesbitt_inequality_l914_91493

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nesbitt_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_nesbitt_inequality_l914_91493


namespace NUMINAMATH_GPT_problem_statement_l914_91477

noncomputable def C_points_count (A B : (ℝ × ℝ)) : ℕ :=
  if A = (0, 0) ∧ B = (12, 0) then 4 else 0

theorem problem_statement :
  let A := (0, 0)
  let B := (12, 0)
  C_points_count A B = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l914_91477


namespace NUMINAMATH_GPT_find_a_value_l914_91436

theorem find_a_value :
  let center := (0.5, Real.sqrt 2)
  let line_dist (a : ℝ) := (abs (0.5 * a + Real.sqrt 2 - Real.sqrt 2)) / Real.sqrt (a^2 + 1)
  line_dist a = Real.sqrt 2 / 4 ↔ (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l914_91436


namespace NUMINAMATH_GPT_negation_of_exists_l914_91480

open Set Real

theorem negation_of_exists (x : Real) :
  ¬ (∃ x ∈ Icc 0 1, x^3 + x^2 > 1) ↔ ∀ x ∈ Icc 0 1, x^3 + x^2 ≤ 1 := 
by sorry

end NUMINAMATH_GPT_negation_of_exists_l914_91480


namespace NUMINAMATH_GPT_coffee_mug_cost_l914_91451

theorem coffee_mug_cost (bracelet_cost gold_heart_necklace_cost total_change total_money_spent : ℤ)
    (bracelets_count gold_heart_necklace_count mugs_count : ℤ)
    (h_bracelet_cost : bracelet_cost = 15)
    (h_gold_heart_necklace_cost : gold_heart_necklace_cost = 10)
    (h_total_change : total_change = 15)
    (h_total_money_spent : total_money_spent = 100)
    (h_bracelets_count : bracelets_count = 3)
    (h_gold_heart_necklace_count : gold_heart_necklace_count = 2)
    (h_mugs_count : mugs_count = 1) :
    mugs_count * ((total_money_spent - total_change) - (bracelets_count * bracelet_cost + gold_heart_necklace_count * gold_heart_necklace_cost)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_coffee_mug_cost_l914_91451


namespace NUMINAMATH_GPT_area_ratio_l914_91458

theorem area_ratio
  (a b c : ℕ)
  (h1 : 2 * (a + c) = 2 * 2 * (b + c))
  (h2 : a = 2 * b)
  (h3 : c = c) :
  (a * c) = 2 * (b * c) :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_l914_91458


namespace NUMINAMATH_GPT_height_to_top_floor_l914_91459

def total_height : ℕ := 1454
def antenna_spire_height : ℕ := 204

theorem height_to_top_floor : (total_height - antenna_spire_height) = 1250 := by
  sorry

end NUMINAMATH_GPT_height_to_top_floor_l914_91459


namespace NUMINAMATH_GPT_gcd_36745_59858_l914_91416

theorem gcd_36745_59858 : Nat.gcd 36745 59858 = 7 :=
sorry

end NUMINAMATH_GPT_gcd_36745_59858_l914_91416


namespace NUMINAMATH_GPT_base_five_of_156_is_1111_l914_91490

def base_five_equivalent (n : ℕ) : ℕ := sorry

theorem base_five_of_156_is_1111 :
  base_five_equivalent 156 = 1111 :=
sorry

end NUMINAMATH_GPT_base_five_of_156_is_1111_l914_91490


namespace NUMINAMATH_GPT_solution_set_inequality_l914_91400

theorem solution_set_inequality (a : ℝ) (x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - 1 / a) > 0) ↔ (a < x ∧ x < 1 / a) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l914_91400


namespace NUMINAMATH_GPT_fill_tank_time_l914_91435

theorem fill_tank_time :
  ∀ (rate_fill rate_empty : ℝ), 
    rate_fill = 1 / 25 → 
    rate_empty = 1 / 50 → 
    (1/2) / (rate_fill - rate_empty) = 25 :=
by
  intros rate_fill rate_empty h_fill h_empty
  sorry

end NUMINAMATH_GPT_fill_tank_time_l914_91435


namespace NUMINAMATH_GPT_area_of_quadrilateral_l914_91446

theorem area_of_quadrilateral (A B C D H : Type) (AB BC : Real)
    (angle_ABC angle_ADC : Real) (BH h : Real)
    (H1 : AB = BC) (H2 : angle_ABC = 90 ∧ angle_ADC = 90)
    (H3 : BH = h) :
    (∃ area : Real, area = h^2) :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l914_91446


namespace NUMINAMATH_GPT_equal_utilities_l914_91484

-- Conditions
def utility (juggling coding : ℕ) : ℕ := juggling * coding

def wednesday_utility (s : ℕ) : ℕ := utility s (12 - s)
def thursday_utility (s : ℕ) : ℕ := utility (6 - s) (s + 4)

-- Theorem
theorem equal_utilities (s : ℕ) (h : wednesday_utility s = thursday_utility s) : s = 12 / 5 := 
by sorry

end NUMINAMATH_GPT_equal_utilities_l914_91484


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l914_91476

noncomputable def a := Real.log 2 / 2
noncomputable def b := Real.log 3 / 3
noncomputable def c := Real.log 5 / 5

theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l914_91476


namespace NUMINAMATH_GPT_original_price_of_cycle_l914_91463

theorem original_price_of_cycle (selling_price : ℝ) (loss_percentage : ℝ) (original_price : ℝ) 
  (h1 : selling_price = 1610)
  (h2 : loss_percentage = 30) 
  (h3 : selling_price = original_price * (1 - loss_percentage / 100)) : 
  original_price = 2300 := 
by 
  sorry

end NUMINAMATH_GPT_original_price_of_cycle_l914_91463


namespace NUMINAMATH_GPT_base_angle_isosceles_triangle_l914_91444

theorem base_angle_isosceles_triangle (α : ℝ) (hα : α = 108) (isosceles : ∀ (a b c : ℝ), a = b ∨ b = c ∨ c = a) : α = 108 →
  α + β + β = 180 → β = 36 :=
by
  sorry

end NUMINAMATH_GPT_base_angle_isosceles_triangle_l914_91444


namespace NUMINAMATH_GPT_chalk_pieces_l914_91415

theorem chalk_pieces (boxes: ℕ) (pieces_per_box: ℕ) (total_chalk: ℕ) 
  (hb: boxes = 194) (hp: pieces_per_box = 18) : 
  total_chalk = 194 * 18 :=
by 
  sorry

end NUMINAMATH_GPT_chalk_pieces_l914_91415


namespace NUMINAMATH_GPT_inequality_problem_l914_91433

theorem inequality_problem (a b : ℝ) (h₁ : 1/a < 1/b) (h₂ : 1/b < 0) :
  (∃ (p q : Prop), 
    (p ∧ q) ∧ 
    ((p ↔ (a + b < a * b)) ∧ 
    (¬q ↔ |a| ≤ |b|) ∧ 
    (¬q ↔ a > b) ∧ 
    (q ↔ (b / a + a / b > 2)))) :=
sorry

end NUMINAMATH_GPT_inequality_problem_l914_91433


namespace NUMINAMATH_GPT_sin_cos_pi_12_eq_l914_91464

theorem sin_cos_pi_12_eq:
  (Real.sin (Real.pi / 12)) * (Real.cos (Real.pi / 12)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_pi_12_eq_l914_91464


namespace NUMINAMATH_GPT_strawberry_growth_rate_l914_91445

theorem strawberry_growth_rate
  (initial_plants : ℕ)
  (months : ℕ)
  (plants_given_away : ℕ)
  (total_plants_after : ℕ)
  (growth_rate : ℕ)
  (h_initial : initial_plants = 3)
  (h_months : months = 3)
  (h_given_away : plants_given_away = 4)
  (h_total_after : total_plants_after = 20)
  (h_equation : initial_plants + growth_rate * months - plants_given_away = total_plants_after) :
  growth_rate = 7 :=
sorry

end NUMINAMATH_GPT_strawberry_growth_rate_l914_91445


namespace NUMINAMATH_GPT_muffins_per_person_l914_91448

-- Definitions based on conditions
def total_friends : ℕ := 4
def total_people : ℕ := 1 + total_friends
def total_muffins : ℕ := 20

-- Theorem statement for the proof
theorem muffins_per_person : total_muffins / total_people = 4 := by
  sorry

end NUMINAMATH_GPT_muffins_per_person_l914_91448


namespace NUMINAMATH_GPT_range_of_x_l914_91414

theorem range_of_x (x : ℝ) : (6 - 2 * x) ≠ 0 ↔ x ≠ 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_range_of_x_l914_91414


namespace NUMINAMATH_GPT_cost_of_tea_l914_91460

theorem cost_of_tea (x : ℕ) (h1 : 9 * x < 1000) (h2 : 10 * x > 1100) : x = 111 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_tea_l914_91460


namespace NUMINAMATH_GPT_train_length_proof_l914_91412

-- Define the conditions
def time_to_cross := 12 -- Time in seconds
def speed_km_per_h := 75 -- Speed in km/h

-- Convert the speed to m/s
def speed_m_per_s := speed_km_per_h * (5 / 18 : ℚ)

-- The length of the train using the formula: length = speed * time
def length_of_train := speed_m_per_s * (time_to_cross : ℚ)

-- The theorem to prove
theorem train_length_proof : length_of_train = 250 := by
  sorry

end NUMINAMATH_GPT_train_length_proof_l914_91412


namespace NUMINAMATH_GPT_solve_fractional_equation_l914_91439

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 4) : 
  (3 - x) / (x - 4) + 1 / (4 - x) = 1 → x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_fractional_equation_l914_91439


namespace NUMINAMATH_GPT_math_problem_l914_91466

theorem math_problem (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
  a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
  b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l914_91466


namespace NUMINAMATH_GPT_count_two_digit_integers_with_perfect_square_sum_l914_91467

def valid_pairs : List (ℕ × ℕ) :=
[(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def reversed_sum_is_perfect_square (n : ℕ) : Prop :=
  ∃ t u, n = 10 * t + u ∧ t + u = 11

theorem count_two_digit_integers_with_perfect_square_sum :
  Nat.card { n : ℕ // is_two_digit n ∧ reversed_sum_is_perfect_square n } = 8 := 
sorry

end NUMINAMATH_GPT_count_two_digit_integers_with_perfect_square_sum_l914_91467


namespace NUMINAMATH_GPT_max_elements_X_l914_91479

structure GameState where
  fire : Nat
  stone : Nat
  metal : Nat

def canCreateX (state : GameState) (x : Nat) : Bool :=
  state.metal >= x ∧ state.fire >= 2 * x ∧ state.stone >= 3 * x

def maxCreateX (state : GameState) : Nat :=
  if h : canCreateX state 14 then 14 else 0 -- we would need to show how to actually maximizing the value

theorem max_elements_X : maxCreateX ⟨50, 50, 0⟩ = 14 := 
by 
  -- Proof would go here, showing via the conditions given above
  -- We would need to show no more than 14 can be created given the initial resources
  sorry

end NUMINAMATH_GPT_max_elements_X_l914_91479


namespace NUMINAMATH_GPT_trucks_after_redistribution_l914_91406

/-- Problem Statement:
   Prove that the total number of trucks after redistribution is 10.
-/

theorem trucks_after_redistribution
    (num_trucks1 : ℕ)
    (boxes_per_truck1 : ℕ)
    (num_trucks2 : ℕ)
    (boxes_per_truck2 : ℕ)
    (containers_per_box : ℕ)
    (containers_per_truck_after : ℕ)
    (h1 : num_trucks1 = 7)
    (h2 : boxes_per_truck1 = 20)
    (h3 : num_trucks2 = 5)
    (h4 : boxes_per_truck2 = 12)
    (h5 : containers_per_box = 8)
    (h6 : containers_per_truck_after = 160) :
  (num_trucks1 * boxes_per_truck1 + num_trucks2 * boxes_per_truck2) * containers_per_box / containers_per_truck_after = 10 := by
  sorry

end NUMINAMATH_GPT_trucks_after_redistribution_l914_91406


namespace NUMINAMATH_GPT_find_z_l914_91482

/-- x and y are positive integers. When x is divided by 9, the remainder is 2, 
and when x is divided by 7, the remainder is 4. When y is divided by 13, 
the remainder is 12. The least possible value of y - x is 14. 
Prove that the number that y is divided by to get a remainder of 3 is 22. -/
theorem find_z (x y z : ℕ) (hx9 : x % 9 = 2) (hx7 : x % 7 = 4) (hy13 : y % 13 = 12) (hyx : y = x + 14) 
: y % z = 3 → z = 22 := 
by 
  sorry

end NUMINAMATH_GPT_find_z_l914_91482


namespace NUMINAMATH_GPT_fourth_quadrant_negative_half_x_axis_upper_half_plane_l914_91428

theorem fourth_quadrant (m : ℝ) : ((-7 < m ∧ m < 3) ↔ ((m^2 - 8 * m + 15 > 0) ∧ (m^2 + 3 * m - 28 < 0))) :=
sorry

theorem negative_half_x_axis (m : ℝ) : (m = 4 ↔ ((m^2 - 8 * m + 15 < 0) ∧ (m^2 + 3 * m - 28 = 0))) :=
sorry

theorem upper_half_plane (m : ℝ) : ((m ≥ 4 ∨ m ≤ -7) ↔ (m^2 + 3 * m - 28 ≥ 0)) :=
sorry

end NUMINAMATH_GPT_fourth_quadrant_negative_half_x_axis_upper_half_plane_l914_91428


namespace NUMINAMATH_GPT_triangle_to_rectangle_ratio_l914_91411

def triangle_perimeter := 60
def rectangle_perimeter := 60

def is_equilateral_triangle (side_length: ℝ) : Prop :=
  3 * side_length = triangle_perimeter

def is_valid_rectangle (length width: ℝ) : Prop :=
  2 * (length + width) = rectangle_perimeter ∧ length = 2 * width

theorem triangle_to_rectangle_ratio (s l w: ℝ) 
  (ht: is_equilateral_triangle s) 
  (hr: is_valid_rectangle l w) : 
  s / w = 2 := by
  sorry

end NUMINAMATH_GPT_triangle_to_rectangle_ratio_l914_91411


namespace NUMINAMATH_GPT_max_value_y_l914_91434

noncomputable def y (x : ℝ) : ℝ := 3 - 3*x - 1/x

theorem max_value_y : (∃ x > 0, ∀ x' > 0, y x' ≤ y x) ∧ (y (1 / Real.sqrt 3) = 3 - 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_max_value_y_l914_91434


namespace NUMINAMATH_GPT_symmetric_about_origin_l914_91413

theorem symmetric_about_origin (x y : ℝ) :
  (∀ (x y : ℝ), (x*y - x^2 = 1) → ((-x)*(-y) - (-x)^2 = 1)) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_symmetric_about_origin_l914_91413


namespace NUMINAMATH_GPT_arithmetic_seq_common_diff_l914_91461

theorem arithmetic_seq_common_diff (a b : ℕ) (d : ℕ) (a1 a2 a8 a9 : ℕ) 
  (h1 : a1 + a8 = 10)
  (h2 : a2 + a9 = 18)
  (h3 : a2 = a1 + d)
  (h4 : a8 = a1 + 7 * d)
  (h5 : a9 = a1 + 8 * d)
  : d = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_common_diff_l914_91461


namespace NUMINAMATH_GPT_percent_change_area_decrease_l914_91432

theorem percent_change_area_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
    let A_initial := L * W
    let L_new := 1.60 * L
    let W_new := 0.40 * W
    let A_new := L_new * W_new
    let percent_change := (A_new - A_initial) / A_initial * 100
    percent_change = -36 :=
by
  sorry

end NUMINAMATH_GPT_percent_change_area_decrease_l914_91432


namespace NUMINAMATH_GPT_select_2n_comparable_rectangles_l914_91422

def comparable (A B : Rectangle) : Prop :=
  -- A can be placed into B by translation and rotation
  exists f : Rectangle → Rectangle, f A = B

theorem select_2n_comparable_rectangles (n : ℕ) (h : n > 1) :
  ∃ (rectangles : List Rectangle), rectangles.length = 2 * n ∧
  ∀ (a b : Rectangle), a ∈ rectangles → b ∈ rectangles → comparable a b :=
sorry

end NUMINAMATH_GPT_select_2n_comparable_rectangles_l914_91422


namespace NUMINAMATH_GPT_average_difference_l914_91462

theorem average_difference (t : ℚ) (ht : t = 4) :
  let m := (13 + 16 + 10 + 15 + 11) / 5
  let n := (16 + t + 3 + 13) / 4
  m - n = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_difference_l914_91462


namespace NUMINAMATH_GPT_problem_l914_91471

theorem problem (h : (0.00027 : ℝ) = 27 / 100000) : (10^5 - 10^3) * 0.00027 = 26.73 := by
  sorry

end NUMINAMATH_GPT_problem_l914_91471


namespace NUMINAMATH_GPT_cube_volume_edge_length_range_l914_91407

theorem cube_volume_edge_length_range (a : ℝ) (h : a^3 = 9) : 2 < a ∧ a < 2.5 :=
by {
    -- proof will go here
    sorry
}

end NUMINAMATH_GPT_cube_volume_edge_length_range_l914_91407
