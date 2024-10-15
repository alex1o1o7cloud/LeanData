import Mathlib

namespace NUMINAMATH_GPT_number_of_tables_l1019_101903

-- Define conditions
def chairs_in_base5 : ℕ := 310  -- chairs in base-5
def chairs_base10 : ℕ := 3 * 5^2 + 1 * 5^1 + 0 * 5^0  -- conversion to base-10
def people_per_table : ℕ := 3

-- The theorem to prove
theorem number_of_tables : chairs_base10 / people_per_table = 26 := by
  -- include the automatic proof here
  sorry

end NUMINAMATH_GPT_number_of_tables_l1019_101903


namespace NUMINAMATH_GPT_arith_seq_general_term_sum_b_n_l1019_101907

-- Definitions and conditions
structure ArithSeq (f : ℕ → ℕ) :=
  (d : ℕ)
  (d_ne_zero : d ≠ 0)
  (Sn : ℕ → ℕ)
  (a3_plus_S5 : f 3 + Sn 5 = 42)
  (geom_seq : (f 4)^2 = (f 1) * (f 13))

-- Given the definitions and conditions, prove the general term formula of the sequence
theorem arith_seq_general_term (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℕ) 
  (d_ne_zero : d ≠ 0) (a3_plus_S5 : a_n 3 + S_n 5 = 42)
  (geom_seq : (a_n 4)^2 = (a_n 1) * (a_n 13)) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- Prove the sum of the first n terms of the sequence b_n
theorem sum_b_n (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (T_n : ℕ → ℕ) (n : ℕ):
  b_n n = 1 / (a_n (n - 1) * a_n n) →
  T_n n = (1 / 2) * (1 - (1 / (2 * n - 1))) →
  T_n n = (n - 1) / (2 * n - 1) :=
sorry

end NUMINAMATH_GPT_arith_seq_general_term_sum_b_n_l1019_101907


namespace NUMINAMATH_GPT_average_greater_median_l1019_101946

theorem average_greater_median :
  let h : ℝ := 120
  let s1 : ℝ := 4
  let s2 : ℝ := 4
  let s3 : ℝ := 5
  let s4 : ℝ := 7
  let s5 : ℝ := 9
  let median : ℝ := (s3 + s4) / 2
  let average : ℝ := (h + s1 + s2 + s3 + s4 + s5) / 6
  average - median = 18.8333 := by
    sorry

end NUMINAMATH_GPT_average_greater_median_l1019_101946


namespace NUMINAMATH_GPT_correct_system_of_equations_l1019_101948

theorem correct_system_of_equations :
  ∃ (x y : ℕ), 
    x + y = 38 
    ∧ 26 * x + 20 * y = 952 := 
by
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l1019_101948


namespace NUMINAMATH_GPT_evaluate_expression_l1019_101992

theorem evaluate_expression :
  2003^3 - 2002 * 2003^2 - 2002^2 * 2003 + 2002^3 = 4005 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1019_101992


namespace NUMINAMATH_GPT_find_missing_number_l1019_101925

theorem find_missing_number 
  (x : ℕ) 
  (avg : (744 + 745 + 747 + 748 + 749 + some_num + 753 + 755 + x) / 9 = 750)
  (hx : x = 755) : 
  some_num = 804 := 
  sorry

end NUMINAMATH_GPT_find_missing_number_l1019_101925


namespace NUMINAMATH_GPT_more_candidates_selected_l1019_101971

theorem more_candidates_selected (total_a total_b selected_a selected_b : ℕ)
  (h1 : total_a = 8000)
  (h2 : total_b = 8000)
  (h3 : selected_a = 6 * total_a / 100)
  (h4 : selected_b = 7 * total_b / 100) :
  selected_b - selected_a = 80 :=
  sorry

end NUMINAMATH_GPT_more_candidates_selected_l1019_101971


namespace NUMINAMATH_GPT_equalize_costs_l1019_101932

theorem equalize_costs (X Y Z : ℝ) (h1 : Y > X) (h2 : Z > Y) : 
  (Y + (Z - (X + Z - 2 * Y) / 3) = Z) → 
   (Y - (Y + Z - (X + Z - 2 * Y)) / 3 = (X + Z - 2 * Y) / 3) := sorry

end NUMINAMATH_GPT_equalize_costs_l1019_101932


namespace NUMINAMATH_GPT_hyperbola_equation_is_correct_l1019_101930

-- Given Conditions
def hyperbola_eq (x y : ℝ) (a : ℝ) : Prop := (x^2) / (a^2) - (y^2) / 4 = 1
def asymptote_eq (x y : ℝ) : Prop := y = (1 / 2) * x

-- Correct answer to be proven
def hyperbola_correct (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 4 = 1

theorem hyperbola_equation_is_correct (x y : ℝ) (a : ℝ) :
  (hyperbola_eq x y a) → (asymptote_eq x y) → (a = 4) → hyperbola_correct x y :=
by 
  intros h_hyperbola h_asymptote h_a
  sorry

end NUMINAMATH_GPT_hyperbola_equation_is_correct_l1019_101930


namespace NUMINAMATH_GPT_largest_val_is_E_l1019_101936

noncomputable def A : ℚ := 4 / (2 - 1/4)
noncomputable def B : ℚ := 4 / (2 + 1/4)
noncomputable def C : ℚ := 4 / (2 - 1/3)
noncomputable def D : ℚ := 4 / (2 + 1/3)
noncomputable def E : ℚ := 4 / (2 - 1/2)

theorem largest_val_is_E : E > A ∧ E > B ∧ E > C ∧ E > D := 
by sorry

end NUMINAMATH_GPT_largest_val_is_E_l1019_101936


namespace NUMINAMATH_GPT_triangle_inequality_l1019_101904

variables {R : Type*} [LinearOrderedField R]

theorem triangle_inequality 
  (a b c u v w : R)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (a + b + c) * (1 / u + 1 / v + 1 / w) ≤ 3 * (a / u + b / v + c / w) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1019_101904


namespace NUMINAMATH_GPT_equation_B_no_solution_l1019_101943

theorem equation_B_no_solution : ¬ ∃ x : ℝ, |-2 * x| + 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_equation_B_no_solution_l1019_101943


namespace NUMINAMATH_GPT_unique_solution_exists_q_l1019_101998

theorem unique_solution_exists_q :
  (∃ q : ℝ, q ≠ 0 ∧ (∀ x y : ℝ, (2 * q * x^2 - 20 * x + 5 = 0) ∧ (2 * q * y^2 - 20 * y + 5 = 0) → x = y)) ↔ q = 10 := 
sorry

end NUMINAMATH_GPT_unique_solution_exists_q_l1019_101998


namespace NUMINAMATH_GPT_percentage_of_fish_gone_bad_l1019_101915

-- Definitions based on conditions
def fish_per_roll : ℕ := 40
def total_fish_bought : ℕ := 400
def sushi_rolls_made : ℕ := 8

-- Definition of fish calculations
def total_fish_used (rolls: ℕ) (per_roll: ℕ) : ℕ := rolls * per_roll
def fish_gone_bad (total : ℕ) (used : ℕ) : ℕ := total - used
def percentage (part : ℕ) (whole : ℕ) : ℚ := (part : ℚ) / (whole : ℚ) * 100

-- Theorem to prove the percentage of bad fish
theorem percentage_of_fish_gone_bad :
  percentage (fish_gone_bad total_fish_bought (total_fish_used sushi_rolls_made fish_per_roll)) total_fish_bought = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_of_fish_gone_bad_l1019_101915


namespace NUMINAMATH_GPT_soda_cost_l1019_101927

variable (b s f : ℝ)

noncomputable def keegan_equation : Prop :=
  3 * b + 2 * s + f = 975

noncomputable def alex_equation : Prop :=
  2 * b + 3 * s + f = 900

theorem soda_cost (h1 : keegan_equation b s f) (h2 : alex_equation b s f) : s = 18.75 :=
by
  sorry

end NUMINAMATH_GPT_soda_cost_l1019_101927


namespace NUMINAMATH_GPT_value_of_expression_l1019_101909

theorem value_of_expression : 48^2 - 2 * 48 * 3 + 3^2 = 2025 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1019_101909


namespace NUMINAMATH_GPT_find_value_l1019_101963

open Classical

variables (a b c : ℝ)

-- Assume a, b, c are roots of the polynomial x^3 - 24x^2 + 50x - 42
def is_root (x : ℝ) : Prop := x^3 - 24*x^2 + 50*x - 42 = 0

-- Vieta's formulas for the given polynomial
axiom h1 : is_root a
axiom h2 : is_root b
axiom h3 : is_root c
axiom h4 : a + b + c = 24
axiom h5 : a * b + b * c + c * a = 50
axiom h6 : a * b * c = 42

-- We want to prove the given expression equals 476/43
theorem find_value : 
  (a/(1/a + b*c) + b/(1/b + c*a) + c/(1/c + a*b) = 476/43) :=
sorry

end NUMINAMATH_GPT_find_value_l1019_101963


namespace NUMINAMATH_GPT_real_number_representation_l1019_101994

theorem real_number_representation (x : ℝ) 
  (h₀ : 0 < x) (h₁ : x ≤ 1) :
  ∃ (n : ℕ → ℕ), (∀ k, n k > 0) ∧ (∀ k, n (k + 1) = n k * 2 ∨ n (k + 1) = n k * 3 ∨ n (k + 1) = n k * 4) ∧ 
  (x = ∑' k, 1 / (n k)) :=
sorry

end NUMINAMATH_GPT_real_number_representation_l1019_101994


namespace NUMINAMATH_GPT_only_number_smaller_than_zero_l1019_101985

theorem only_number_smaller_than_zero : ∀ (x : ℝ), (x = 5 ∨ x = 2 ∨ x = 0 ∨ x = -Real.sqrt 2) → x < 0 → x = -Real.sqrt 2 :=
by
  intro x hx h
  sorry

end NUMINAMATH_GPT_only_number_smaller_than_zero_l1019_101985


namespace NUMINAMATH_GPT_problem_statement_l1019_101954

-- Define the variables
variables (S T Tie : ℝ)

-- Define the given conditions
def condition1 : Prop := 6 * S + 4 * T + 2 * Tie = 80
def condition2 : Prop := 5 * S + 3 * T + 2 * Tie = 110

-- Define the question to be proved
def target : Prop := 4 * S + 2 * T + 2 * Tie = 50

-- Lean theorem statement
theorem problem_statement (h1 : condition1 S T Tie) (h2 : condition2 S T Tie) : target S T Tie :=
  sorry

end NUMINAMATH_GPT_problem_statement_l1019_101954


namespace NUMINAMATH_GPT_height_of_sky_island_l1019_101978

theorem height_of_sky_island (day_climb : ℕ) (night_slide : ℕ) (days : ℕ) (final_day_climb : ℕ) :
  day_climb = 25 →
  night_slide = 3 →
  days = 64 →
  final_day_climb = 25 →
  (days - 1) * (day_climb - night_slide) + final_day_climb = 1411 :=
by
  -- Add the formal proof here
  sorry

end NUMINAMATH_GPT_height_of_sky_island_l1019_101978


namespace NUMINAMATH_GPT_sin_330_eq_neg_half_l1019_101982

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_330_eq_neg_half_l1019_101982


namespace NUMINAMATH_GPT_probability_no_order_l1019_101908

theorem probability_no_order (P : ℕ) 
  (h1 : 60 ≤ 100) (h2 : 10 ≤ 100) (h3 : 15 ≤ 100) 
  (h4 : 5 ≤ 100) (h5 : 3 ≤ 100) (h6 : 2 ≤ 100) :
  P = 100 - (60 + 10 + 15 + 5 + 3 + 2) :=
by 
  sorry

end NUMINAMATH_GPT_probability_no_order_l1019_101908


namespace NUMINAMATH_GPT_inequality_nonnegative_reals_l1019_101966

theorem inequality_nonnegative_reals (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  x^2 * y^2 + x^2 * y + x * y^2 ≤ x^4 * y + x + y^4 :=
sorry

end NUMINAMATH_GPT_inequality_nonnegative_reals_l1019_101966


namespace NUMINAMATH_GPT_at_least_one_heart_or_king_l1019_101949

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_heart_or_king_l1019_101949


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1019_101960

-- Define the sets A and B
def setA : Set ℝ := { x | -1 < x ∧ x ≤ 4 }
def setB : Set ℝ := { x | 2 < x ∧ x ≤ 5 }

-- The intersection of sets A and B
def intersectAB : Set ℝ := { x | 2 < x ∧ x ≤ 4 }

-- The theorem statement to be proved
theorem intersection_of_A_and_B : ∀ x, x ∈ setA ∩ setB ↔ x ∈ intersectAB := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1019_101960


namespace NUMINAMATH_GPT_cost_per_bag_proof_minimize_total_cost_l1019_101917

-- Definitions of given conditions
variable (x y : ℕ) -- cost per bag for brands A and B respectively
variable (m : ℕ) -- number of bags of brand B

def first_purchase_eq := 100 * x + 150 * y = 7000
def second_purchase_eq := 180 * x + 120 * y = 8100
def cost_per_bag_A : ℕ := 25
def cost_per_bag_B : ℕ := 30
def total_bags := 300
def constraint := (300 - m) ≤ 2 * m

-- Prove the costs per bag
theorem cost_per_bag_proof (h1 : first_purchase_eq x y)
                           (h2 : second_purchase_eq x y) :
  x = cost_per_bag_A ∧ y = cost_per_bag_B :=
sorry

-- Define the cost function and prove the purchase strategy
def total_cost (m : ℕ) : ℕ := 25 * (300 - m) + 30 * m

theorem minimize_total_cost (h : constraint m) :
  m = 100 ∧ total_cost 100 = 8000 :=
sorry

end NUMINAMATH_GPT_cost_per_bag_proof_minimize_total_cost_l1019_101917


namespace NUMINAMATH_GPT_opponent_score_l1019_101923

theorem opponent_score (s g c total opponent : ℕ)
  (h1 : s = 20)
  (h2 : g = 2 * s)
  (h3 : c = 2 * g)
  (h4 : total = s + g + c)
  (h5 : total - 55 = opponent) :
  opponent = 85 := by
  sorry

end NUMINAMATH_GPT_opponent_score_l1019_101923


namespace NUMINAMATH_GPT_linear_function_properties_l1019_101976

def linear_function (x : ℝ) : ℝ := -2 * x + 1

theorem linear_function_properties :
  (∀ x, linear_function x = -2 * x + 1) ∧
  (∀ x₁ x₂, x₁ < x₂ → linear_function x₁ > linear_function x₂) ∧
  (linear_function 0 = 1) ∧
  ((∃ x, x > 0 ∧ linear_function x > 0) ∧ (∃ x, x < 0 ∧ linear_function x > 0) ∧ (∃ x, x > 0 ∧ linear_function x < 0))
  :=
by
  sorry

end NUMINAMATH_GPT_linear_function_properties_l1019_101976


namespace NUMINAMATH_GPT_kite_area_is_192_l1019_101913

-- Define the points with doubled dimensions
def A : (ℝ × ℝ) := (0, 16)
def B : (ℝ × ℝ) := (8, 24)
def C : (ℝ × ℝ) := (16, 16)
def D : (ℝ × ℝ) := (8, 0)

-- Calculate the area of the kite
noncomputable def kiteArea (A B C D : ℝ × ℝ) : ℝ :=
  let baseUpper := abs (C.1 - A.1)
  let heightUpper := abs (B.2 - A.2)
  let areaUpper := 1 / 2 * baseUpper * heightUpper
  let baseLower := baseUpper
  let heightLower := abs (B.2 - D.2)
  let areaLower := 1 / 2 * baseLower * heightLower
  areaUpper + areaLower

-- State the theorem to prove the kite area is 192 square inches
theorem kite_area_is_192 : kiteArea A B C D = 192 := 
  sorry

end NUMINAMATH_GPT_kite_area_is_192_l1019_101913


namespace NUMINAMATH_GPT_middle_card_is_four_l1019_101947

theorem middle_card_is_four (a b c : ℕ) (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                            (h2 : a + b + c = 15)
                            (h3 : a < b ∧ b < c)
                            (h_casey : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_tracy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_stacy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            : b = 4 := 
sorry

end NUMINAMATH_GPT_middle_card_is_four_l1019_101947


namespace NUMINAMATH_GPT_line_equation_l1019_101901

theorem line_equation
  (P : ℝ × ℝ) (hP : P = (1, -1))
  (h_perp : ∀ x y : ℝ, 3 * x - 2 * y = 0 → 2 * x + 3 * y = 0):
  ∃ m : ℝ, (2 * P.1 + 3 * P.2 + m = 0) ∧ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l1019_101901


namespace NUMINAMATH_GPT_tank_capacity_percentage_l1019_101950

noncomputable def radius (C : ℝ) := C / (2 * Real.pi)
noncomputable def volume (r h : ℝ) := Real.pi * r^2 * h

theorem tank_capacity_percentage :
  let r_M := radius 8
  let r_B := radius 10
  let V_M := volume r_M 10
  let V_B := volume r_B 8
  (V_M / V_B * 100) = 80 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_percentage_l1019_101950


namespace NUMINAMATH_GPT_cannot_determine_remaining_pictures_l1019_101957

theorem cannot_determine_remaining_pictures (taken_pics : ℕ) (dolphin_show_pics : ℕ) (total_pics : ℕ) :
  taken_pics = 28 → dolphin_show_pics = 16 → total_pics = 44 → 
  (∀ capacity : ℕ, ¬ (total_pics + x = capacity)) → 
  ¬ ∃ remaining_pics : ℕ, remaining_pics = capacity - total_pics :=
by {
  sorry
}

end NUMINAMATH_GPT_cannot_determine_remaining_pictures_l1019_101957


namespace NUMINAMATH_GPT_inequality_solution_l1019_101973

theorem inequality_solution (x : ℝ) : x^2 + x - 12 ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3 := sorry

end NUMINAMATH_GPT_inequality_solution_l1019_101973


namespace NUMINAMATH_GPT_simple_annual_interest_rate_l1019_101999

-- Given definitions and conditions
def monthly_interest_payment := 225
def principal_amount := 30000
def annual_interest_payment := monthly_interest_payment * 12
def annual_interest_rate := annual_interest_payment / principal_amount

-- Theorem statement
theorem simple_annual_interest_rate :
  annual_interest_rate * 100 = 9 := by
sorry

end NUMINAMATH_GPT_simple_annual_interest_rate_l1019_101999


namespace NUMINAMATH_GPT_savings_after_increase_l1019_101912

theorem savings_after_increase (salary savings_rate increase_rate : ℝ) (old_savings old_expenses new_expenses new_savings : ℝ)
  (h_salary : salary = 6000)
  (h_savings_rate : savings_rate = 0.2)
  (h_increase_rate : increase_rate = 0.2)
  (h_old_savings : old_savings = savings_rate * salary)
  (h_old_expenses : old_expenses = salary - old_savings)
  (h_new_expenses : new_expenses = old_expenses * (1 + increase_rate))
  (h_new_savings : new_savings = salary - new_expenses) :
  new_savings = 240 :=
by sorry

end NUMINAMATH_GPT_savings_after_increase_l1019_101912


namespace NUMINAMATH_GPT_find_a_l1019_101937

theorem find_a (a b x : ℝ) (h1 : a ≠ b)
  (h2 : a^3 + b^3 = 35 * x^3)
  (h3 : a^2 - b^2 = 4 * x^2) : a = 2 * x ∨ a = -2 * x :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1019_101937


namespace NUMINAMATH_GPT_minimal_sum_of_squares_l1019_101942

theorem minimal_sum_of_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ p q r : ℕ, a + b = p^2 ∧ b + c = q^2 ∧ a + c = r^2) ∧
  a + b + c = 55 := 
by sorry

end NUMINAMATH_GPT_minimal_sum_of_squares_l1019_101942


namespace NUMINAMATH_GPT_tetrahedron_volume_l1019_101916

variable {R : ℝ}
variable {S1 S2 S3 S4 : ℝ}
variable {V : ℝ}

theorem tetrahedron_volume (R : ℝ) (S1 S2 S3 S4 V : ℝ) :
  V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_l1019_101916


namespace NUMINAMATH_GPT_triangle_inequality_l1019_101900

theorem triangle_inequality (A B C : ℝ) (k : ℝ) (hABC : A + B + C = π) (h1 : 1 ≤ k) (h2 : k ≤ 2) :
  (1 / (k - Real.cos A)) + (1 / (k - Real.cos B)) + (1 / (k - Real.cos C)) ≥ 6 / (2 * k - 1) := 
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1019_101900


namespace NUMINAMATH_GPT_number_of_functions_with_given_range_l1019_101920

theorem number_of_functions_with_given_range : 
  let S := {2, 5, 10}
  let R (x : ℤ) := x^2 + 1
  ∃ f : ℤ → ℤ, (∀ y ∈ S, ∃ x : ℤ, f x = y) ∧ (f '' {x | R x ∈ S} = S) :=
    sorry

end NUMINAMATH_GPT_number_of_functions_with_given_range_l1019_101920


namespace NUMINAMATH_GPT_find_a_l1019_101990

noncomputable def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x * a = 1}
axiom A_is_B (a : ℝ) : A ∩ B a = B a → (a = 0) ∨ (a = 1/3) ∨ (a = 1/5)

-- statement to prove
theorem find_a (a : ℝ) (h : A ∩ B a = B a) : (a = 0) ∨ (a = 1/3) ∨ (a = 1/5) :=
by 
  apply A_is_B
  assumption

end NUMINAMATH_GPT_find_a_l1019_101990


namespace NUMINAMATH_GPT_range_of_function_l1019_101996

theorem range_of_function (y : ℝ) (t: ℝ) (x : ℝ) (h_t : t = x^2 - 1) (h_domain : t ∈ Set.Ici (-1)) :
  ∃ (y_set : Set ℝ), ∀ y ∈ y_set, y = (1/3)^t ∧ y_set = Set.Ioo 0 3 ∨ y_set = Set.Icc 0 3 := by
  sorry

end NUMINAMATH_GPT_range_of_function_l1019_101996


namespace NUMINAMATH_GPT_intersection_points_l1019_101988

variables {α β : Type*} [DecidableEq α] {f : α → β} {x m : α}

theorem intersection_points (dom : α → Prop) (h : dom x → ∃! y, f x = y) : 
  (∃ y, f m = y) ∨ ¬ ∃ y, f m = y :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l1019_101988


namespace NUMINAMATH_GPT_no_real_roots_in_interval_l1019_101924

variable {a b c : ℝ}

theorem no_real_roots_in_interval (ha : 0 < a) (h : 12 * a + 5 * b + 2 * c > 0) :
  ¬ ∃ α β, (2 < α ∧ α < 3) ∧ (2 < β ∧ β < 3) ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0 := by
  sorry

end NUMINAMATH_GPT_no_real_roots_in_interval_l1019_101924


namespace NUMINAMATH_GPT_solve_inequality_l1019_101922

noncomputable def g (x : ℝ) := Real.arcsin x + x^3

theorem solve_inequality (x : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1)
    (h2 : Real.arcsin (x^2) + Real.arcsin x + x^6 + x^3 > 0) :
    0 < x ∧ x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1019_101922


namespace NUMINAMATH_GPT_find_number_and_n_l1019_101918

def original_number (x y z n : ℕ) : Prop := 
  n = 2 ∧ 100 * x + 10 * y + z = 178

theorem find_number_and_n (x y z n : ℕ) :
  (∀ x y z n, original_number x y z n) ↔ (n = 2 ∧ 100 * x + 10 * y + z = 178) := 
sorry

end NUMINAMATH_GPT_find_number_and_n_l1019_101918


namespace NUMINAMATH_GPT_y_square_range_l1019_101934

theorem y_square_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 16) ^ (1/3) = 4) : 
  230 ≤ y^2 ∧ y^2 < 240 :=
sorry

end NUMINAMATH_GPT_y_square_range_l1019_101934


namespace NUMINAMATH_GPT_hexagon_sum_balanced_assignment_exists_l1019_101965

-- Definitions based on the conditions
def is_valid_assignment (a b c d e f g : ℕ) : Prop :=
a + b + g = a + c + g ∧ a + b + g = a + d + g ∧ a + b + g = a + e + g ∧
a + b + g = b + c + g ∧ a + b + g = b + d + g ∧ a + b + g = b + e + g ∧
a + b + g = c + d + g ∧ a + b + g = c + e + g ∧ a + b + g = d + e + g

-- The theorem we want to prove
theorem hexagon_sum_balanced_assignment_exists :
  ∃ (a b c d e f g : ℕ), 
  (a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 2 ∨ b = 3 ∨ b = 5) ∧ 
  (c = 2 ∨ c = 3 ∨ c = 5) ∧ 
  (d = 2 ∨ d = 3 ∨ d = 5) ∧ 
  (e = 2 ∨ e = 3 ∨ e = 5) ∧
  (f = 2 ∨ f = 3 ∨ f = 5) ∧
  (g = 2 ∨ g = 3 ∨ g = 5) ∧
  is_valid_assignment a b c d e f g :=
sorry

end NUMINAMATH_GPT_hexagon_sum_balanced_assignment_exists_l1019_101965


namespace NUMINAMATH_GPT_star_three_and_four_l1019_101980

def star (a b : ℝ) : ℝ := 4 * a + 5 * b - 2 * a * b

theorem star_three_and_four : star 3 4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_star_three_and_four_l1019_101980


namespace NUMINAMATH_GPT_option_b_option_c_option_d_l1019_101972

theorem option_b (x : ℝ) (h : x > 1) : (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4*Real.sqrt 2 + 1) :=
by
  sorry

theorem option_c (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3 * x * y) : 2*x + y ≥ 3 :=
by
  sorry

theorem option_d (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) : 3*x + y ≤ 2*Real.sqrt 21 / 7 :=
by
  sorry

end NUMINAMATH_GPT_option_b_option_c_option_d_l1019_101972


namespace NUMINAMATH_GPT_train_speed_l1019_101931

theorem train_speed (length : Nat) (time_sec : Nat) (length_km : length = 200)
  (time_hr : time_sec = 12) : (200 : ℝ) / (12 / 3600 : ℝ) = 60 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_train_speed_l1019_101931


namespace NUMINAMATH_GPT_number_of_sides_l1019_101951

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end NUMINAMATH_GPT_number_of_sides_l1019_101951


namespace NUMINAMATH_GPT_max_n_value_is_9_l1019_101938

variable (a b c d n : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : c > d)
variable (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d)))

theorem max_n_value_is_9 (h1 : a > b) (h2 : b > c) (h3 : c > d)
    (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_n_value_is_9_l1019_101938


namespace NUMINAMATH_GPT_remainder_when_divided_l1019_101974

theorem remainder_when_divided (x : ℤ) (k : ℤ) (h: x = 82 * k + 5) : 
  ((x + 17) % 41) = 22 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l1019_101974


namespace NUMINAMATH_GPT_beef_weight_loss_percentage_l1019_101905

noncomputable def weight_after_processing : ℝ := 570
noncomputable def weight_before_processing : ℝ := 876.9230769230769

theorem beef_weight_loss_percentage :
  (weight_before_processing - weight_after_processing) / weight_before_processing * 100 = 35 :=
by
  sorry

end NUMINAMATH_GPT_beef_weight_loss_percentage_l1019_101905


namespace NUMINAMATH_GPT_least_f_e_l1019_101964

theorem least_f_e (e : ℝ) (he : e > 0) : 
  ∃ f, (∀ (a b c d : ℝ), a^3 + b^3 + c^3 + d^3 ≤ e^2 * (a^2 + b^2 + c^2 + d^2) + f * (a^4 + b^4 + c^4 + d^4)) ∧ f = 1 / (4 * e^2) :=
sorry

end NUMINAMATH_GPT_least_f_e_l1019_101964


namespace NUMINAMATH_GPT_probability_of_selecting_letter_a_l1019_101902

def total_ways := Nat.choose 5 2
def ways_to_select_a := 4
def probability_of_selecting_a := (ways_to_select_a : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_letter_a :
  probability_of_selecting_a = 2 / 5 :=
by
  -- proof steps will be filled in here
  sorry

end NUMINAMATH_GPT_probability_of_selecting_letter_a_l1019_101902


namespace NUMINAMATH_GPT_exists_group_of_four_l1019_101944

-- Assuming 21 students, and any three have done homework together exactly once in either mathematics or Russian.
-- We aim to prove there exists a group of four students such that any three of them have done homework together in the same subject.
noncomputable def students : Type := Fin 21

-- Define a predicate to show that three students have done homework together.
-- We use "math" and "russian" to denote the subjects.
inductive Subject
| math
| russian

-- Define a relation expressing that any three students have done exactly one subject homework together.
axiom homework_done (s1 s2 s3 : students) : Subject 

theorem exists_group_of_four :
  ∃ (a b c d : students), 
    (homework_done a b c = homework_done a b d) ∧
    (homework_done a b c = homework_done a c d) ∧
    (homework_done a b c = homework_done b c d) ∧
    (homework_done a b d = homework_done a c d) ∧
    (homework_done a b d = homework_done b c d) ∧
    (homework_done a c d = homework_done b c d) :=
sorry

end NUMINAMATH_GPT_exists_group_of_four_l1019_101944


namespace NUMINAMATH_GPT_average_number_of_glasses_per_box_l1019_101929

-- Definitions and conditions
variables (S L : ℕ) -- S is the number of smaller boxes, L is the number of larger boxes

-- Condition 1: One box contains 12 glasses, and the other contains 16 glasses.
-- (This is implicitly understood in the equation for total glasses)

-- Condition 3: There are 16 more larger boxes than smaller smaller boxes
def condition_3 := L = S + 16

-- Condition 4: The total number of glasses is 480.
def condition_4 := 12 * S + 16 * L = 480

-- Proving the average number of glasses per box is 15
theorem average_number_of_glasses_per_box (h1 : condition_3 S L) (h2 : condition_4 S L) :
  (480 : ℝ) / (S + L) = 15 :=
by 
  -- Assuming S and L are natural numbers 
  sorry

end NUMINAMATH_GPT_average_number_of_glasses_per_box_l1019_101929


namespace NUMINAMATH_GPT_solve_equation_l1019_101991

theorem solve_equation {x : ℂ} : (x - 2)^4 + (x - 6)^4 = 272 →
  x = 6 ∨ x = 2 ∨ x = 4 + 2 * Complex.I ∨ x = 4 - 2 * Complex.I :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l1019_101991


namespace NUMINAMATH_GPT_speed_of_second_train_40_kmph_l1019_101921

noncomputable def length_train_1 : ℝ := 140
noncomputable def length_train_2 : ℝ := 160
noncomputable def crossing_time : ℝ := 10.799136069114471
noncomputable def speed_train_1 : ℝ := 60

theorem speed_of_second_train_40_kmph :
  let total_distance := length_train_1 + length_train_2
  let relative_speed_mps := total_distance / crossing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  let speed_train_2 := relative_speed_kmph - speed_train_1
  speed_train_2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_second_train_40_kmph_l1019_101921


namespace NUMINAMATH_GPT_total_pages_in_book_is_250_l1019_101940

-- Definitions
def avg_pages_first_part := 36
def days_first_part := 3
def avg_pages_second_part := 44
def days_second_part := 3
def pages_last_day := 10

-- Calculate total pages
def total_pages := (days_first_part * avg_pages_first_part) + (days_second_part * avg_pages_second_part) + pages_last_day

-- Theorem statement
theorem total_pages_in_book_is_250 : total_pages = 250 := by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_is_250_l1019_101940


namespace NUMINAMATH_GPT_find_multiple_l1019_101970

-- Definitions of the conditions
def is_positive (x : ℝ) : Prop := x > 0

-- Main statement
theorem find_multiple (x : ℝ) (h : is_positive x) (hx : x = 8) : ∃ k : ℝ, x + 8 = k * (1 / x) ∧ k = 128 :=
by
  use 128
  sorry

end NUMINAMATH_GPT_find_multiple_l1019_101970


namespace NUMINAMATH_GPT_x_intercept_of_line_l1019_101935

theorem x_intercept_of_line (x y : ℚ) (h_eq : 4 * x + 7 * y = 28) (h_y : y = 0) : (x, y) = (7, 0) := 
by 
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l1019_101935


namespace NUMINAMATH_GPT_circumference_of_tank_B_l1019_101989

noncomputable def radius_of_tank (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def volume_of_tank (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem circumference_of_tank_B 
  (h_A : ℝ) (C_A : ℝ) (h_B : ℝ) (volume_ratio : ℝ)
  (hA_pos : 0 < h_A) (CA_pos : 0 < C_A) (hB_pos : 0 < h_B) (vr_pos : 0 < volume_ratio) :
  2 * Real.pi * (radius_of_tank (volume_of_tank (radius_of_tank C_A) h_A / (volume_ratio * Real.pi * h_B))) = 17.7245 :=
by 
  sorry

end NUMINAMATH_GPT_circumference_of_tank_B_l1019_101989


namespace NUMINAMATH_GPT_parallelogram_area_l1019_101961

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) 
  (h_b : b = 7) (h_h : h = 2 * b) (h_A : A = b * h) : A = 98 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallelogram_area_l1019_101961


namespace NUMINAMATH_GPT_factorization_correct_l1019_101955

theorem factorization_correct :
  ∀ (m a b x y : ℝ), 
    (m^2 - 4 = (m + 2) * (m - 2)) ∧
    ((a + 3) * (a - 3) = a^2 - 9) ∧
    (a^2 - b^2 + 1 = (a + b) * (a - b) + 1) ∧
    (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3) →
    (m^2 - 4 = (m + 2) * (m - 2)) :=
by
  intros m a b x y h
  have ⟨hA, hB, hC, hD⟩ := h
  exact hA

end NUMINAMATH_GPT_factorization_correct_l1019_101955


namespace NUMINAMATH_GPT_real_b_values_for_non_real_roots_l1019_101993

theorem real_b_values_for_non_real_roots (b : ℝ) :
  let discriminant := b^2 - 4 * 1 * 16
  discriminant < 0 ↔ -8 < b ∧ b < 8 := 
sorry

end NUMINAMATH_GPT_real_b_values_for_non_real_roots_l1019_101993


namespace NUMINAMATH_GPT_option_B_can_be_factored_l1019_101956

theorem option_B_can_be_factored (a b : ℝ) : 
  (-a^2 + b^2) = (b+a)*(b-a) := 
by
  sorry

end NUMINAMATH_GPT_option_B_can_be_factored_l1019_101956


namespace NUMINAMATH_GPT_jellybean_total_l1019_101995

theorem jellybean_total (large_jellybeans_per_glass : ℕ) 
  (small_jellybeans_per_glass : ℕ) 
  (num_large_glasses : ℕ) 
  (num_small_glasses : ℕ) 
  (h1 : large_jellybeans_per_glass = 50) 
  (h2 : small_jellybeans_per_glass = large_jellybeans_per_glass / 2) 
  (h3 : num_large_glasses = 5) 
  (h4 : num_small_glasses = 3) : 
  (num_large_glasses * large_jellybeans_per_glass + num_small_glasses * small_jellybeans_per_glass) = 325 :=
by
  sorry

end NUMINAMATH_GPT_jellybean_total_l1019_101995


namespace NUMINAMATH_GPT_river_trip_longer_than_lake_trip_l1019_101968

theorem river_trip_longer_than_lake_trip (v w : ℝ) (h1 : v > w) : 
  (20 * v) / (v^2 - w^2) > 20 / v :=
by {
  sorry
}

end NUMINAMATH_GPT_river_trip_longer_than_lake_trip_l1019_101968


namespace NUMINAMATH_GPT_quadratic_extreme_values_l1019_101939

theorem quadratic_extreme_values (y1 y2 y3 y4 : ℝ) 
  (h1 : y2 < y3) 
  (h2 : y3 = y4) 
  (h3 : ∀ x, ∃ (a b c : ℝ), ∀ y, y = a * x * x + b * x + c) :
  (y1 < y2) ∧ (y2 < y3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_extreme_values_l1019_101939


namespace NUMINAMATH_GPT_number_of_jump_sequences_l1019_101911

def jump_sequences (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (a 3 = 3) ∧
  (∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 2))

theorem number_of_jump_sequences :
  ∃ a : ℕ → ℕ, jump_sequences a ∧ a 11 = 60 :=
by
  sorry

end NUMINAMATH_GPT_number_of_jump_sequences_l1019_101911


namespace NUMINAMATH_GPT_find_normal_price_l1019_101984

theorem find_normal_price (P : ℝ) (S : ℝ) (d1 d2 d3 : ℝ) : 
  (P * (1 - d1) * (1 - d2) * (1 - d3) = S) → S = 144 → d1 = 0.12 → d2 = 0.22 → d3 = 0.15 → P = 246.81 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_find_normal_price_l1019_101984


namespace NUMINAMATH_GPT_complex_exponential_to_rectangular_form_l1019_101958

theorem complex_exponential_to_rectangular_form :
  Real.sqrt 2 * Complex.exp (13 * Real.pi * Complex.I / 4) = -1 - Complex.I := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_complex_exponential_to_rectangular_form_l1019_101958


namespace NUMINAMATH_GPT_isosceles_triangle_angle_split_l1019_101914

theorem isosceles_triangle_angle_split (A B C1 C2 : ℝ)
  (h_isosceles : A = B)
  (h_greater_than_third : A > C1)
  (h_split : C1 + C2 = C) :
  C1 = C2 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_split_l1019_101914


namespace NUMINAMATH_GPT_palindrome_probability_divisible_by_7_l1019_101953

-- Define the conditions
def is_four_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 1001 * a + 110 * b

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Define the proof problem
theorem palindrome_probability_divisible_by_7 : 
  (∃ (n : ℕ), is_four_digit_palindrome n ∧ is_divisible_by_7 n) →
  ∃ p : ℚ, p = 1/5 :=
sorry

end NUMINAMATH_GPT_palindrome_probability_divisible_by_7_l1019_101953


namespace NUMINAMATH_GPT_smallest_possible_norm_l1019_101906

-- Defining the vector \begin{pmatrix} -2 \\ 4 \end{pmatrix}
def vec_a : ℝ × ℝ := (-2, 4)

-- Condition: the norm of \mathbf{v} + \begin{pmatrix} -2 \\ 4 \end{pmatrix} = 10
def satisfies_condition (v : ℝ × ℝ) : Prop :=
  (Real.sqrt ((v.1 + vec_a.1) ^ 2 + (v.2 + vec_a.2) ^ 2)) = 10

noncomputable def smallest_norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_possible_norm (v : ℝ × ℝ) (h : satisfies_condition v) : smallest_norm v = 10 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_smallest_possible_norm_l1019_101906


namespace NUMINAMATH_GPT_find_triples_l1019_101997

-- Defining the conditions
def divides (x y : ℕ) : Prop := ∃ k, y = k * x

-- The main Lean statement
theorem find_triples (a b c : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  divides a (b * c - 1) → divides b (a * c - 1) → divides c (a * b - 1) →
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 2 ∧ b = 5 ∧ c = 3) ∨
  (a = 3 ∧ b = 2 ∧ c = 5) ∨ (a = 3 ∧ b = 5 ∧ c = 2) ∨
  (a = 5 ∧ b = 2 ∧ c = 3) ∨ (a = 5 ∧ b = 3 ∧ c = 2) :=
sorry

end NUMINAMATH_GPT_find_triples_l1019_101997


namespace NUMINAMATH_GPT_power_eq_45_l1019_101928

theorem power_eq_45 (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 5) : a^(2*m + n) = 45 := by
  sorry

end NUMINAMATH_GPT_power_eq_45_l1019_101928


namespace NUMINAMATH_GPT_image_relative_velocity_l1019_101967

-- Definitions of the constants
def f : ℝ := 0.2
def x : ℝ := 0.5
def vt : ℝ := 3

-- Lens equation
def lens_equation (f x y : ℝ) : Prop :=
  (1 / x) + (1 / y) = 1 / f

-- Image distance
noncomputable def y (f x : ℝ) : ℝ :=
  1 / (1 / f - 1 / x)

-- Derivative of y with respect to x
noncomputable def dy_dx (f x : ℝ) : ℝ :=
  (f^2) / (x - f)^2

-- Image velocity
noncomputable def vk (vt dy_dx : ℝ) : ℝ :=
  vt * dy_dx

-- Relative velocity
noncomputable def v_rel (vt vk : ℝ) : ℝ :=
  vk - vt

-- Theorem to prove the relative velocity
theorem image_relative_velocity : v_rel vt (vk vt (dy_dx f x)) = -5 / 3 := 
by
  sorry

end NUMINAMATH_GPT_image_relative_velocity_l1019_101967


namespace NUMINAMATH_GPT_sculptures_not_on_display_count_l1019_101981

noncomputable def total_art_pieces : ℕ := 1800
noncomputable def pieces_on_display : ℕ := total_art_pieces / 3
noncomputable def pieces_not_on_display : ℕ := total_art_pieces - pieces_on_display
noncomputable def sculptures_on_display : ℕ := pieces_on_display / 6
noncomputable def sculptures_not_on_display : ℕ := pieces_not_on_display * 2 / 3

theorem sculptures_not_on_display_count : sculptures_not_on_display = 800 :=
by {
  -- Since this is a statement only as requested, we use sorry to skip the proof
  sorry
}

end NUMINAMATH_GPT_sculptures_not_on_display_count_l1019_101981


namespace NUMINAMATH_GPT_function_passes_through_fixed_point_l1019_101945

noncomputable def f (a x : ℝ) := a^(x+1) - 1

theorem function_passes_through_fixed_point (a : ℝ) (h_pos : 0 < a) (h_not_one : a ≠ 1) :
  f a (-1) = 0 := by
  sorry

end NUMINAMATH_GPT_function_passes_through_fixed_point_l1019_101945


namespace NUMINAMATH_GPT_value_of_x_add_y_l1019_101926

theorem value_of_x_add_y (x y : ℝ) 
  (h1 : x + Real.sin y = 2023)
  (h2 : x + 2023 * Real.cos y = 2021)
  (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (3 * Real.pi / 4)) : 
  x + y = 2023 - (Real.sqrt 2) / 2 + (3 * Real.pi) / 4 := 
sorry

end NUMINAMATH_GPT_value_of_x_add_y_l1019_101926


namespace NUMINAMATH_GPT_triangles_intersection_area_is_zero_l1019_101941

-- Define the vertices of the two triangles
def vertex_triangle_1 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (0, 2)
| ⟨1, _⟩ => (2, 1)
| ⟨2, _⟩ => (0, 0)

def vertex_triangle_2 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (2, 2)
| ⟨1, _⟩ => (0, 1)
| ⟨2, _⟩ => (2, 0)

-- The area of the intersection of the two triangles
def area_intersection (v1 v2 : Fin 3 → (ℝ × ℝ)) : ℝ :=
  0

-- The theorem to prove
theorem triangles_intersection_area_is_zero :
  area_intersection vertex_triangle_1 vertex_triangle_2 = 0 :=
by
  -- Proof is omitted here.
  sorry

end NUMINAMATH_GPT_triangles_intersection_area_is_zero_l1019_101941


namespace NUMINAMATH_GPT_handshake_count_l1019_101987

def total_handshakes (men women : ℕ) := 
  (men * (men - 1)) / 2 + men * (women - 1)

theorem handshake_count :
  let men := 13
  let women := 13
  total_handshakes men women = 234 :=
by
  sorry

end NUMINAMATH_GPT_handshake_count_l1019_101987


namespace NUMINAMATH_GPT_mixed_groups_count_l1019_101986

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end NUMINAMATH_GPT_mixed_groups_count_l1019_101986


namespace NUMINAMATH_GPT_equation_parallel_equation_perpendicular_l1019_101959

variables {x y : ℝ}

def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x - 5 * y + 14 = 0
def l3 (x y : ℝ) := 2 * x - y + 7 = 0

theorem equation_parallel {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : 2 * x - y + 6 = 0 :=
sorry

theorem equation_perpendicular {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : x + 2 * y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_equation_parallel_equation_perpendicular_l1019_101959


namespace NUMINAMATH_GPT_age_of_new_person_l1019_101952

theorem age_of_new_person (T : ℝ) (A : ℝ) (h : T / 20 - 4 = (T - 60 + A) / 20) : A = 40 :=
sorry

end NUMINAMATH_GPT_age_of_new_person_l1019_101952


namespace NUMINAMATH_GPT_James_balloons_l1019_101983

theorem James_balloons (A J : ℕ) (h1 : A = 513) (h2 : J = A + 208) : J = 721 :=
by {
  sorry
}

end NUMINAMATH_GPT_James_balloons_l1019_101983


namespace NUMINAMATH_GPT_solution_ne_zero_l1019_101910

theorem solution_ne_zero (a x : ℝ) (h : x = a * x + 1) : x ≠ 0 := sorry

end NUMINAMATH_GPT_solution_ne_zero_l1019_101910


namespace NUMINAMATH_GPT_divides_expression_l1019_101919

theorem divides_expression (n : ℕ) : 7 ∣ (3^(12 * n^2 + 1) + 2^(6 * n + 2)) := sorry

end NUMINAMATH_GPT_divides_expression_l1019_101919


namespace NUMINAMATH_GPT_find_x_l1019_101979

theorem find_x (x : ℝ) (h : 2 * x - 1 = -( -x + 5 )) : x = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1019_101979


namespace NUMINAMATH_GPT_remainder_of_sum_l1019_101933

theorem remainder_of_sum (c d : ℤ) (p q : ℤ) (h1 : c = 60 * p + 53) (h2 : d = 45 * q + 28) : 
  (c + d) % 15 = 6 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l1019_101933


namespace NUMINAMATH_GPT_algebraic_expression_value_l1019_101977

theorem algebraic_expression_value (x : ℝ) (h : x^2 - x - 1 = 0) : x^3 - 2*x + 1 = 2 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1019_101977


namespace NUMINAMATH_GPT_needed_adjustment_l1019_101975

def price_adjustment (P : ℝ) : ℝ :=
  let P_reduced := P - 0.20 * P
  let P_raised := P_reduced + 0.10 * P_reduced
  let P_target := P - 0.10 * P
  P_target - P_raised

theorem needed_adjustment (P : ℝ) : price_adjustment P = 2 * (P / 100) := sorry

end NUMINAMATH_GPT_needed_adjustment_l1019_101975


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_506_l1019_101962

theorem sum_of_consecutive_integers_with_product_506 :
  ∃ x : ℕ, (x * (x + 1) = 506) → (x + (x + 1) = 45) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_506_l1019_101962


namespace NUMINAMATH_GPT_line_intersects_semicircle_at_two_points_l1019_101969

theorem line_intersects_semicircle_at_two_points
  (m : ℝ) :
  (3 ≤ m ∧ m < 3 * Real.sqrt 2) ↔ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ (y₁ = -x₁ + m ∧ y₁ = Real.sqrt (9 - x₁^2)) ∧ (y₂ = -x₂ + m ∧ y₂ = Real.sqrt (9 - x₂^2))) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_line_intersects_semicircle_at_two_points_l1019_101969
