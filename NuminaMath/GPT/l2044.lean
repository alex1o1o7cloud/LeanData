import Mathlib

namespace NUMINAMATH_GPT_sum_ef_l2044_204468

variables (a b c d e f : ℝ)

-- Definitions based on conditions
def avg_ab : Prop := (a + b) / 2 = 5.2
def avg_cd : Prop := (c + d) / 2 = 5.8
def overall_avg : Prop := (a + b + c + d + e + f) / 6 = 5.4

-- Main theorem to prove
theorem sum_ef (h1 : avg_ab a b) (h2 : avg_cd c d) (h3 : overall_avg a b c d e f) : e + f = 10.4 :=
sorry

end NUMINAMATH_GPT_sum_ef_l2044_204468


namespace NUMINAMATH_GPT_islanders_liars_l2044_204498

inductive Person
| A
| B

open Person

def is_liar (p : Person) : Prop :=
  sorry -- placeholder for the actual definition

def makes_statement (p : Person) (statement : Prop) : Prop :=
  sorry -- placeholder for the actual definition

theorem islanders_liars :
  makes_statement A (is_liar A ∧ ¬ is_liar B) →
  is_liar A ∧ is_liar B :=
by
  sorry

end NUMINAMATH_GPT_islanders_liars_l2044_204498


namespace NUMINAMATH_GPT_number_of_students_who_went_to_church_l2044_204444

-- Define the number of chairs and the number of students.
variables (C S : ℕ)

-- Define the first condition: 9 students per chair with one student left.
def condition1 := S = 9 * C + 1

-- Define the second condition: 10 students per chair with one chair vacant.
def condition2 := S = 10 * C - 10

-- The theorem to be proved.
theorem number_of_students_who_went_to_church (h1 : condition1 C S) (h2 : condition2 C S) : S = 100 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_students_who_went_to_church_l2044_204444


namespace NUMINAMATH_GPT_find_B_investment_l2044_204436

def A_investment : ℝ := 24000
def C_investment : ℝ := 36000
def C_profit : ℝ := 36000
def total_profit : ℝ := 92000
def B_investment := 32000

theorem find_B_investment (B_investment_unknown : ℝ) :
  (C_investment / C_profit) = ((A_investment + B_investment_unknown + C_investment) / total_profit) →
  B_investment_unknown = B_investment := 
by 
  -- Mathematical equivalence to the given problem
  -- Proof omitted since only the statement is required
  sorry

end NUMINAMATH_GPT_find_B_investment_l2044_204436


namespace NUMINAMATH_GPT_distinct_prime_factors_count_l2044_204407

theorem distinct_prime_factors_count :
  ∀ (a b c d : ℕ),
  (a = 79) → (b = 3^4) → (c = 5 * 17) → (d = 3 * 29) →
  (∃ s : Finset ℕ, ∀ n ∈ s, Nat.Prime n ∧ 79 * 81 * 85 * 87 = s.prod id) :=
sorry

end NUMINAMATH_GPT_distinct_prime_factors_count_l2044_204407


namespace NUMINAMATH_GPT_drums_of_grapes_per_day_l2044_204499

-- Definitions derived from conditions
def pickers := 235
def raspberry_drums_per_day := 100
def total_days := 77
def total_drums := 17017

-- Prove the main theorem
theorem drums_of_grapes_per_day : (total_drums - total_days * raspberry_drums_per_day) / total_days = 121 := by
  sorry

end NUMINAMATH_GPT_drums_of_grapes_per_day_l2044_204499


namespace NUMINAMATH_GPT_bug_total_distance_l2044_204420

def total_distance_bug (start : ℤ) (pos1 : ℤ) (pos2 : ℤ) (pos3 : ℤ) : ℤ :=
  abs (pos1 - start) + abs (pos2 - pos1) + abs (pos3 - pos2)

theorem bug_total_distance :
  total_distance_bug 3 (-4) 6 2 = 21 :=
by
  -- We insert a sorry here to indicate the proof is skipped.
  sorry

end NUMINAMATH_GPT_bug_total_distance_l2044_204420


namespace NUMINAMATH_GPT_ratio_of_areas_l2044_204473

theorem ratio_of_areas (s : ℝ) : (s^2) / ((3 * s)^2) = 1 / 9 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l2044_204473


namespace NUMINAMATH_GPT_delores_money_left_l2044_204478

theorem delores_money_left (initial_amount spent_computer spent_printer : ℝ) 
    (h1 : initial_amount = 450) 
    (h2 : spent_computer = 400) 
    (h3 : spent_printer = 40) : 
    initial_amount - (spent_computer + spent_printer) = 10 := 
by 
    sorry

end NUMINAMATH_GPT_delores_money_left_l2044_204478


namespace NUMINAMATH_GPT_lisa_minimum_fifth_term_score_l2044_204481

theorem lisa_minimum_fifth_term_score :
  ∀ (score1 score2 score3 score4 average_needed total_terms : ℕ),
  score1 = 84 →
  score2 = 80 →
  score3 = 82 →
  score4 = 87 →
  average_needed = 85 →
  total_terms = 5 →
  (∃ (score5 : ℕ), 
     (score1 + score2 + score3 + score4 + score5) / total_terms ≥ average_needed ∧ 
     score5 = 92) :=
by
  sorry

end NUMINAMATH_GPT_lisa_minimum_fifth_term_score_l2044_204481


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2044_204427

theorem problem1 : (-23 + 13 - 12) = -22 := 
by sorry

theorem problem2 : ((-2)^3 / 4 + 3 * (-5)) = -17 := 
by sorry

theorem problem3 : (-24 * (1/2 - 3/4 - 1/8)) = 9 := 
by sorry

theorem problem4 : ((2 - 7) / 5^2 + (-1)^2023 * (1/10)) = -3/10 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2044_204427


namespace NUMINAMATH_GPT_range_of_a_l2044_204424

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) ↔ a ∈ Set.union (Set.Iio 1) (Set.Ioi 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2044_204424


namespace NUMINAMATH_GPT_find_a8_l2044_204430

theorem find_a8 (a : ℕ → ℝ) (h1 : ∀ n ≥ 2, (2 * a n - 3) / (a n - 1) = 2) (h2 : a 2 = 1) : a 8 = 16 := 
sorry

end NUMINAMATH_GPT_find_a8_l2044_204430


namespace NUMINAMATH_GPT_problem_from_conditions_l2044_204463

theorem problem_from_conditions 
  (x y : ℝ)
  (h1 : 3 * x * (2 * x + y) = 14)
  (h2 : y * (2 * x + y) = 35) :
  (2 * x + y)^2 = 49 := 
by 
  sorry

end NUMINAMATH_GPT_problem_from_conditions_l2044_204463


namespace NUMINAMATH_GPT_calc_expression_l2044_204455

theorem calc_expression : (4 + 6 + 10) / 3 - 2 / 3 = 6 := by
  sorry

end NUMINAMATH_GPT_calc_expression_l2044_204455


namespace NUMINAMATH_GPT_distance_not_six_l2044_204445

theorem distance_not_six (x : ℝ) : 
  (x = 6 → 10 + (x - 3) * 1.8 ≠ 17.2) ∧ 
  (10 + (x - 3) * 1.8 = 17.2 → x ≠ 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_not_six_l2044_204445


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2044_204497

theorem simplify_and_evaluate_expression 
  (x y : ℤ) (hx : x = -3) (hy : y = -2) :
  3 * x^2 * y - (2 * x^2 * y - (2 * x * y - x^2 * y) - 4 * x^2 * y) - x * y = -66 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2044_204497


namespace NUMINAMATH_GPT_bricks_required_to_pave_courtyard_l2044_204485

theorem bricks_required_to_pave_courtyard :
  let courtyard_length : ℝ := 25
  let courtyard_width : ℝ := 16
  let brick_length : ℝ := 0.20
  let brick_width : ℝ := 0.10
  let area_courtyard := courtyard_length * courtyard_width
  let area_brick := brick_length * brick_width
  let number_of_bricks := area_courtyard / area_brick
  number_of_bricks = 20000 := by
    let courtyard_length : ℝ := 25
    let courtyard_width : ℝ := 16
    let brick_length : ℝ := 0.20
    let brick_width : ℝ := 0.10
    let area_courtyard := courtyard_length * courtyard_width
    let area_brick := brick_length * brick_width
    let number_of_bricks := area_courtyard / area_brick
    sorry

end NUMINAMATH_GPT_bricks_required_to_pave_courtyard_l2044_204485


namespace NUMINAMATH_GPT_range_of_k_l2044_204451

theorem range_of_k (k x y : ℝ) 
  (h₁ : 2 * x - y = k + 1) 
  (h₂ : x - y = -3) 
  (h₃ : x + y > 2) : k > -4.5 :=
sorry

end NUMINAMATH_GPT_range_of_k_l2044_204451


namespace NUMINAMATH_GPT_cost_scheme_1_cost_scheme_2_cost_comparison_scheme_more_cost_effective_combined_plan_l2044_204449

variable (x : ℕ) (x_ge_4 : x ≥ 4)

-- Total cost under scheme ①
def scheme_1_cost (x : ℕ) : ℕ := 5 * x + 60

-- Total cost under scheme ②
def scheme_2_cost (x : ℕ) : ℕ := 9 * (80 + 5 * x) / 10

theorem cost_scheme_1 (x : ℕ) (x_ge_4 : x ≥ 4) : 
  scheme_1_cost x = 5 * x + 60 :=  
sorry

theorem cost_scheme_2 (x : ℕ) (x_ge_4 : x ≥ 4) : 
  scheme_2_cost x = (80 + 5 * x) * 9 / 10 := 
sorry

-- When x = 30, compare which scheme is more cost-effective
variable (x_eq_30 : x = 30)
theorem cost_comparison_scheme (x_eq_30 : x = 30) : 
  scheme_1_cost 30 > scheme_2_cost 30 := 
sorry

-- When x = 30, a more cost-effective combined purchasing plan
def combined_scheme_cost : ℕ := scheme_1_cost 4 + scheme_2_cost (30 - 4)

theorem more_cost_effective_combined_plan (x_eq_30 : x = 30) : 
  combined_scheme_cost < scheme_1_cost 30 ∧ combined_scheme_cost < scheme_2_cost 30 := 
sorry

end NUMINAMATH_GPT_cost_scheme_1_cost_scheme_2_cost_comparison_scheme_more_cost_effective_combined_plan_l2044_204449


namespace NUMINAMATH_GPT_rooster_stamps_eq_two_l2044_204440

variable (r d : ℕ) -- r is the number of rooster stamps, d is the number of daffodil stamps

theorem rooster_stamps_eq_two (h1 : d = 2) (h2 : r - d = 0) : r = 2 := by
  sorry

end NUMINAMATH_GPT_rooster_stamps_eq_two_l2044_204440


namespace NUMINAMATH_GPT_simple_interest_time_l2044_204423

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r/n)^(n*t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem simple_interest_time (SI CI : ℝ) (SI_given CI_given P_simp P_comp r_simp r_comp t_comp : ℝ) :
  SI = CI / 2 →
  CI = compound_interest P_comp r_comp 1 t_comp - P_comp →
  SI = simple_interest P_simp r_simp t_comp →
  P_simp = 1272 →
  r_simp = 0.10 →
  P_comp = 5000 →
  r_comp = 0.12 →
  t_comp = 2 →
  t_comp = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_simple_interest_time_l2044_204423


namespace NUMINAMATH_GPT_valid_exponent_rule_l2044_204411

theorem valid_exponent_rule (a : ℝ) : (a^3)^2 = a^6 :=
by
  sorry

end NUMINAMATH_GPT_valid_exponent_rule_l2044_204411


namespace NUMINAMATH_GPT_find_y_intercept_l2044_204447

-- Conditions
def line_equation (x y : ℝ) : Prop := 4 * x + 7 * y - 3 * x * y = 28

-- Statement (Proof Problem)
theorem find_y_intercept : ∃ y : ℝ, line_equation 0 y ∧ (0, y) = (0, 4) := by
  sorry

end NUMINAMATH_GPT_find_y_intercept_l2044_204447


namespace NUMINAMATH_GPT_certain_number_any_number_l2044_204432

theorem certain_number_any_number (k : ℕ) (n : ℕ) (h1 : 5^k - k^5 = 1) (h2 : 15^k ∣ n) : true :=
by
  sorry

end NUMINAMATH_GPT_certain_number_any_number_l2044_204432


namespace NUMINAMATH_GPT_distance_center_to_line_l2044_204482

noncomputable def circle_center : ℝ × ℝ :=
  let b := 2
  let c := -4
  (1, -2)

noncomputable def distance_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / (Real.sqrt (a^2 + b^2))

theorem distance_center_to_line : distance_point_to_line circle_center 3 4 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_distance_center_to_line_l2044_204482


namespace NUMINAMATH_GPT_person_speed_kmh_l2044_204471

-- Given conditions
def distance_meters : ℝ := 1000
def time_minutes : ℝ := 10

-- Proving the speed in km/h
theorem person_speed_kmh :
  (distance_meters / 1000) / (time_minutes / 60) = 6 :=
  sorry

end NUMINAMATH_GPT_person_speed_kmh_l2044_204471


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2044_204414

theorem sufficient_but_not_necessary {a b : ℝ} (h₁ : a < b) (h₂ : b < 0) : 
  (a^2 > b^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2044_204414


namespace NUMINAMATH_GPT_expected_value_decagonal_die_l2044_204457

-- Given conditions
def decagonal_die_faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def probability (n : ℕ) : ℚ := 1 / 10

-- The mathematical proof problem (statement only, no proof required)
theorem expected_value_decagonal_die : 
  (List.sum decagonal_die_faces : ℚ) / List.length decagonal_die_faces = 5.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_decagonal_die_l2044_204457


namespace NUMINAMATH_GPT_triangles_not_necessarily_symmetric_l2044_204491

open Real

structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A1 : Point)
(A2 : Point)
(A3 : Point)

structure Ellipse :=
(a : ℝ) -- semi-major axis
(b : ℝ) -- semi-minor axis

def inscribed_in (T : Triangle) (E : Ellipse) : Prop :=
  -- Assuming the definition of the inscribed, can be encoded based on the ellipse equation: x^2/a^2 + y^2/b^2 <= 1 for each vertex.
  sorry

def symmetric_wrt_axis (T₁ T₂ : Triangle) : Prop :=
  -- Definition of symmetry with respect to an axis (to be defined)
  sorry

def symmetric_wrt_center (T₁ T₂ : Triangle) : Prop :=
  -- Definition of symmetry with respect to the center (to be defined)
  sorry

theorem triangles_not_necessarily_symmetric {E : Ellipse} {T₁ T₂ : Triangle}
  (h₁ : inscribed_in T₁ E) (h₂ : inscribed_in T₂ E) (heq : T₁ = T₂) :
  ¬ symmetric_wrt_axis T₁ T₂ ∧ ¬ symmetric_wrt_center T₁ T₂ :=
sorry

end NUMINAMATH_GPT_triangles_not_necessarily_symmetric_l2044_204491


namespace NUMINAMATH_GPT_find_number_l2044_204477

theorem find_number (number : ℝ) (h1 : 213 * number = 3408) (h2 : 0.16 * 2.13 = 0.3408) : number = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2044_204477


namespace NUMINAMATH_GPT_three_obtuse_impossible_l2044_204492

-- Define the type for obtuse angle
def is_obtuse (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

-- Define the main theorem stating the problem
theorem three_obtuse_impossible 
  (A B C D O : Type) 
  (angle_AOB angle_COD angle_AOD angle_COB
   angle_OAB angle_OBA angle_OBC angle_OCB
   angle_OAD angle_ODA angle_ODC angle_OCC : ℝ)
  (h1 : angle_AOB = angle_COD)
  (h2 : angle_AOD = angle_COB)
  (h_sum : angle_AOB + angle_COD + angle_AOD + angle_COB = 360)
  : ¬ (is_obtuse angle_OAB ∧ is_obtuse angle_OBC ∧ is_obtuse angle_ODA) := 
sorry

end NUMINAMATH_GPT_three_obtuse_impossible_l2044_204492


namespace NUMINAMATH_GPT_no_real_roots_for_pair_2_2_3_l2044_204412

noncomputable def discriminant (A B : ℝ) : ℝ :=
  let a := 1 - 2 * B
  let b := -B
  let c := -A + A * B
  b ^ 2 - 4 * a * c

theorem no_real_roots_for_pair_2_2_3 : discriminant 2 (2 / 3) < 0 := by
  sorry

end NUMINAMATH_GPT_no_real_roots_for_pair_2_2_3_l2044_204412


namespace NUMINAMATH_GPT_find_n_l2044_204461

def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Given conditions
variable (n : ℕ)
variable (coef : ℕ)
variable (h : coef = binomial_coeff n 2 * 9)

-- Proof target
theorem find_n (h : coef = 54) : n = 4 :=
  sorry

end NUMINAMATH_GPT_find_n_l2044_204461


namespace NUMINAMATH_GPT_log_sum_range_l2044_204446

theorem log_sum_range {x y : ℝ} (hx : 0 < x) (hy : 0 < y)
  (h : Real.log (x + y) / Real.log 2 = Real.log x / Real.log 2 + Real.log y / Real.log 2) :
  4 ≤ x + y :=
by
  sorry

end NUMINAMATH_GPT_log_sum_range_l2044_204446


namespace NUMINAMATH_GPT_total_students_in_school_l2044_204474

theorem total_students_in_school
  (students_per_group : ℕ) (groups_per_class : ℕ) (number_of_classes : ℕ)
  (h1 : students_per_group = 7) (h2 : groups_per_class = 9) (h3 : number_of_classes = 13) :
  students_per_group * groups_per_class * number_of_classes = 819 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_total_students_in_school_l2044_204474


namespace NUMINAMATH_GPT_inclination_angle_of_line_l2044_204458

theorem inclination_angle_of_line (θ : ℝ) : 
  (∃ m : ℝ, ∀ x y : ℝ, 2 * x - y + 1 = 0 → m = 2) → θ = Real.arctan 2 :=
by
  sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l2044_204458


namespace NUMINAMATH_GPT_range_of_m_l2044_204462

-- Definitions based on given conditions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * x + m ≠ 0
def q (m : ℝ) : Prop := m > 1 ∧ m - 1 > 1

-- The mathematically equivalent proof problem
theorem range_of_m (m : ℝ) (hnp : ¬p m) (hapq : ¬ (p m ∧ q m)) : 1 < m ∧ m ≤ 2 :=
  by sorry

end NUMINAMATH_GPT_range_of_m_l2044_204462


namespace NUMINAMATH_GPT_total_boxes_count_l2044_204404

theorem total_boxes_count
  (initial_boxes : ℕ := 2013)
  (boxes_per_operation : ℕ := 13)
  (operations : ℕ := 2013)
  (non_empty_boxes : ℕ := 2013)
  (total_boxes : ℕ := initial_boxes + boxes_per_operation * operations) :
  non_empty_boxes = operations → total_boxes = 28182 :=
by
  sorry

end NUMINAMATH_GPT_total_boxes_count_l2044_204404


namespace NUMINAMATH_GPT_range_of_m_for_point_in_second_quadrant_l2044_204410

theorem range_of_m_for_point_in_second_quadrant (m : ℝ) :
  (m - 3 < 0) ∧ (m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  -- The proof will be inserted here.
  sorry

end NUMINAMATH_GPT_range_of_m_for_point_in_second_quadrant_l2044_204410


namespace NUMINAMATH_GPT_kelly_points_l2044_204495

theorem kelly_points (K : ℕ) 
  (h1 : 12 + 2 * 12 + K + 2 * K + 12 / 2 = 69) : K = 9 := by
  sorry

end NUMINAMATH_GPT_kelly_points_l2044_204495


namespace NUMINAMATH_GPT_smallest_number_of_oranges_l2044_204453

theorem smallest_number_of_oranges (n : ℕ) (total_oranges : ℕ) :
  (total_oranges > 200) ∧ total_oranges = 15 * n - 6 ∧ n ≥ 14 → total_oranges = 204 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_oranges_l2044_204453


namespace NUMINAMATH_GPT_find_quarters_l2044_204425

-- Define the conditions
def quarters_bounds (q : ℕ) : Prop :=
  8 < q ∧ q < 80

def stacks_mod4 (q : ℕ) : Prop :=
  q % 4 = 2

def stacks_mod6 (q : ℕ) : Prop :=
  q % 6 = 2

def stacks_mod8 (q : ℕ) : Prop :=
  q % 8 = 2

-- The theorem to prove
theorem find_quarters (q : ℕ) (h_bounds : quarters_bounds q) (h4 : stacks_mod4 q) (h6 : stacks_mod6 q) (h8 : stacks_mod8 q) : 
  q = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_quarters_l2044_204425


namespace NUMINAMATH_GPT_difference_of_numbers_l2044_204418

theorem difference_of_numbers :
  ∃ (a b : ℕ), a + b = 36400 ∧ b = 100 * a ∧ b - a = 35640 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l2044_204418


namespace NUMINAMATH_GPT_smallest_non_unit_digit_multiple_of_five_l2044_204496

theorem smallest_non_unit_digit_multiple_of_five :
  ∀ (d : ℕ), ((d = 0) ∨ (d = 5)) → (d ≠ 1 ∧ d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 6 ∧ d ≠ 7 ∧ d ≠ 8 ∧ d ≠ 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_non_unit_digit_multiple_of_five_l2044_204496


namespace NUMINAMATH_GPT_fit_jack_apples_into_jill_basket_l2044_204426

-- Conditions:
def jack_basket_full : ℕ := 12
def jack_basket_space : ℕ := 4
def jack_current_apples : ℕ := jack_basket_full - jack_basket_space
def jill_basket_capacity : ℕ := 2 * jack_basket_full

-- Proof statement:
theorem fit_jack_apples_into_jill_basket : jill_basket_capacity / jack_current_apples = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_fit_jack_apples_into_jill_basket_l2044_204426


namespace NUMINAMATH_GPT_loaned_books_count_l2044_204416

variable (x : ℕ) -- x is the number of books loaned out during the month

theorem loaned_books_count 
  (initial_books : ℕ) (returned_percentage : ℚ) (remaining_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : returned_percentage = 0.80)
  (h3 : remaining_books = 66) :
  x = 45 :=
by
  -- Proof can be inserted here
  sorry

end NUMINAMATH_GPT_loaned_books_count_l2044_204416


namespace NUMINAMATH_GPT_sum_of_digits_N_l2044_204472

-- Define the main problem conditions and the result statement
theorem sum_of_digits_N {N : ℕ} 
  (h₁ : (N * (N + 1)) / 2 = 5103) : 
  (N.digits 10).sum = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_N_l2044_204472


namespace NUMINAMATH_GPT_total_numbers_l2044_204419

theorem total_numbers (m j c : ℕ) (h1 : m = j + 20) (h2 : j = c - 40) (h3 : c = 80) : m + j + c = 180 := 
by sorry

end NUMINAMATH_GPT_total_numbers_l2044_204419


namespace NUMINAMATH_GPT_find_element_in_A_l2044_204417

def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

def f (p : A) : B := (p.1 + 2 * p.2, 2 * p.1 - p.2)

theorem find_element_in_A : ∃ p : A, f p = (3, 1) ∧ p = (1, 1) := by
  sorry

end NUMINAMATH_GPT_find_element_in_A_l2044_204417


namespace NUMINAMATH_GPT_heath_average_carrots_per_hour_l2044_204402

theorem heath_average_carrots_per_hour 
  (rows1 rows2 : ℕ)
  (plants_per_row1 plants_per_row2 : ℕ)
  (hours1 hours2 : ℕ)
  (h1 : rows1 = 200)
  (h2 : rows2 = 200)
  (h3 : plants_per_row1 = 275)
  (h4 : plants_per_row2 = 325)
  (h5 : hours1 = 15)
  (h6 : hours2 = 25) :
  ((rows1 * plants_per_row1 + rows2 * plants_per_row2) / (hours1 + hours2) = 3000) :=
  by
  sorry

end NUMINAMATH_GPT_heath_average_carrots_per_hour_l2044_204402


namespace NUMINAMATH_GPT_value_of_one_stamp_l2044_204488

theorem value_of_one_stamp (matches_per_book : ℕ) (initial_stamps : ℕ) (trade_matchbooks : ℕ) (stamps_left : ℕ) :
  matches_per_book = 24 → initial_stamps = 13 → trade_matchbooks = 5 → stamps_left = 3 →
  (trade_matchbooks * matches_per_book) / (initial_stamps - stamps_left) = 12 :=
by
  intros h1 h2 h3 h4
  -- Insert the logical connection assertions here, concluding with the final proof step.
  sorry

end NUMINAMATH_GPT_value_of_one_stamp_l2044_204488


namespace NUMINAMATH_GPT_cary_earnings_l2044_204429

variable (shoe_cost : ℕ) (saved_amount : ℕ)
variable (lawns_per_weekend : ℕ) (weeks_needed : ℕ)
variable (total_cost_needed : ℕ) (total_lawns : ℕ) (earn_per_lawn : ℕ)
variable (h1 : shoe_cost = 120)
variable (h2 : saved_amount = 30)
variable (h3 : lawns_per_weekend = 3)
variable (h4 : weeks_needed = 6)
variable (h5 : total_cost_needed = shoe_cost - saved_amount)
variable (h6 : total_lawns = lawns_per_weekend * weeks_needed)
variable (h7 : earn_per_lawn = total_cost_needed / total_lawns)

theorem cary_earnings :
  earn_per_lawn = 5 :=
by 
  sorry

end NUMINAMATH_GPT_cary_earnings_l2044_204429


namespace NUMINAMATH_GPT_greatest_possible_perimeter_l2044_204476

theorem greatest_possible_perimeter (a b c : ℕ) 
    (h₁ : a = 4 * b ∨ b = 4 * a ∨ c = 4 * a ∨ c = 4 * b)
    (h₂ : a = 18 ∨ b = 18 ∨ c = 18)
    (triangle_ineq : a + b > c ∧ a + c > b ∧ b + c > a) :
    a + b + c = 43 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_possible_perimeter_l2044_204476


namespace NUMINAMATH_GPT_tom_has_1_dollar_left_l2044_204470

/-- Tom has $19 and each folder costs $2. After buying as many folders as possible,
Tom will have $1 left. -/
theorem tom_has_1_dollar_left (initial_money : ℕ) (folder_cost : ℕ) (folders_bought : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 19)
  (h2 : folder_cost = 2)
  (h3 : folders_bought = initial_money / folder_cost)
  (h4 : money_left = initial_money - folders_bought * folder_cost) :
  money_left = 1 :=
by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_tom_has_1_dollar_left_l2044_204470


namespace NUMINAMATH_GPT_ribbon_left_l2044_204443

-- Define the variables
def T : ℕ := 18 -- Total ribbon in yards
def G : ℕ := 6  -- Number of gifts
def P : ℕ := 2  -- Ribbon per gift in yards

-- Statement of the theorem
theorem ribbon_left (T G P : ℕ) : (T - G * P) = 6 :=
by
  -- Add conditions as Lean assumptions
  have hT : T = 18 := sorry
  have hG : G = 6 := sorry
  have hP : P = 2 := sorry
  -- Now prove the final result
  sorry

end NUMINAMATH_GPT_ribbon_left_l2044_204443


namespace NUMINAMATH_GPT_number_of_people_with_cards_greater_than_0p3_l2044_204406

theorem number_of_people_with_cards_greater_than_0p3 :
  (∃ (number_of_people : ℕ),
     number_of_people = (if 0.3 < 0.8 then 1 else 0) +
                        (if 0.3 < (1 / 2) then 1 else 0) +
                        (if 0.3 < 0.9 then 1 else 0) +
                        (if 0.3 < (1 / 3) then 1 else 0)) →
  number_of_people = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_with_cards_greater_than_0p3_l2044_204406


namespace NUMINAMATH_GPT_smallest_prime_after_seven_non_primes_l2044_204437

-- Define the property of being non-prime
def non_prime (n : ℕ) : Prop :=
¬Nat.Prime n

-- Statement of the proof problem
theorem smallest_prime_after_seven_non_primes :
  ∃ m : ℕ, (∀ i : ℕ, (m - 7 ≤ i ∧ i < m) → non_prime i) ∧ Nat.Prime m ∧
  (∀ p : ℕ, (∀ i : ℕ, (p - 7 ≤ i ∧ i < p) → non_prime i) → Nat.Prime p → m ≤ p) :=
sorry

end NUMINAMATH_GPT_smallest_prime_after_seven_non_primes_l2044_204437


namespace NUMINAMATH_GPT_largest_number_is_D_l2044_204484

noncomputable def A : ℝ := 15467 + 3 / 5791
noncomputable def B : ℝ := 15467 - 3 / 5791
noncomputable def C : ℝ := 15467 * (3 / 5791)
noncomputable def D : ℝ := 15467 / (3 / 5791)
noncomputable def E : ℝ := 15467.5791

theorem largest_number_is_D :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end NUMINAMATH_GPT_largest_number_is_D_l2044_204484


namespace NUMINAMATH_GPT_find_x_minus_y_l2044_204460

/-
Given that:
  2 * x + y = 7
  x + 2 * y = 8
We want to prove:
  x - y = -1
-/

theorem find_x_minus_y (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : x - y = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l2044_204460


namespace NUMINAMATH_GPT_max_marks_paper_I_l2044_204467

-- Definitions based on the problem conditions
def percent_to_pass : ℝ := 0.35
def secured_marks : ℝ := 42
def failed_by : ℝ := 23

-- The calculated passing marks
def passing_marks : ℝ := secured_marks + failed_by

-- The theorem statement that needs to be proved
theorem max_marks_paper_I : ∀ (M : ℝ), (percent_to_pass * M = passing_marks) → M = 186 :=
by
  intros M h
  have h1 : M = passing_marks / percent_to_pass := by sorry
  have h2 : M = 186 := by sorry
  exact h2

end NUMINAMATH_GPT_max_marks_paper_I_l2044_204467


namespace NUMINAMATH_GPT_polynomial_roots_problem_l2044_204494

theorem polynomial_roots_problem (a b c d e : ℝ) (h1 : a ≠ 0) 
    (h2 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
    (h3 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
    (h4 : a + b + c + d + e = 0) :
    (b + c + d) / a = -7 := 
sorry

end NUMINAMATH_GPT_polynomial_roots_problem_l2044_204494


namespace NUMINAMATH_GPT_farmer_profit_l2044_204479

-- Define the conditions and relevant information
def feeding_cost_per_month_per_piglet : ℕ := 12
def number_of_piglets : ℕ := 8

def selling_details : List (ℕ × ℕ × ℕ) :=
[
  (2, 350, 12),
  (3, 400, 15),
  (2, 450, 18),
  (1, 500, 21)
]

-- Calculate total revenue
def total_revenue : ℕ :=
selling_details.foldl (λ acc (piglets, price, _) => acc + piglets * price) 0

-- Calculate total feeding cost
def total_feeding_cost : ℕ :=
selling_details.foldl (λ acc (piglets, _, months) => acc + piglets * feeding_cost_per_month_per_piglet * months) 0

-- Calculate profit
def profit : ℕ := total_revenue - total_feeding_cost

-- Statement of the theorem
theorem farmer_profit : profit = 1788 := by
  sorry

end NUMINAMATH_GPT_farmer_profit_l2044_204479


namespace NUMINAMATH_GPT_ramu_paid_for_old_car_l2044_204428

theorem ramu_paid_for_old_car (repairs : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (P : ℝ) :
    repairs = 12000 ∧ selling_price = 64900 ∧ profit_percent = 20.185185185185187 → 
    selling_price = P + repairs + (P + repairs) * (profit_percent / 100) → 
    P = 42000 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_ramu_paid_for_old_car_l2044_204428


namespace NUMINAMATH_GPT_area_of_triangle_BEF_l2044_204448

open Real

theorem area_of_triangle_BEF (a b x y : ℝ) (h1 : a * b = 30) (h2 : (1/2) * abs (x * (b - y) + a * b - a * y) = 2) (h3 : (1/2) * abs (x * (-y) + a * y - x * b) = 3) :
  (1/2) * abs (x * y) = 35 / 8 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_BEF_l2044_204448


namespace NUMINAMATH_GPT_brown_stripes_l2044_204400

theorem brown_stripes (B G Bl : ℕ) (h1 : G = 3 * B) (h2 : Bl = 5 * G) (h3 : Bl = 60) : B = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_brown_stripes_l2044_204400


namespace NUMINAMATH_GPT_frank_composes_problems_l2044_204456

theorem frank_composes_problems (bill_problems : ℕ) (ryan_problems : ℕ) (frank_problems : ℕ) 
  (h1 : bill_problems = 20)
  (h2 : ryan_problems = 2 * bill_problems)
  (h3 : frank_problems = 3 * ryan_problems)
  : frank_problems / 4 = 30 :=
by
  sorry

end NUMINAMATH_GPT_frank_composes_problems_l2044_204456


namespace NUMINAMATH_GPT_martha_points_calculation_l2044_204452

theorem martha_points_calculation :
  let beef_cost := 3 * 11
  let beef_discount := 0.10 * beef_cost
  let total_beef_cost := beef_cost - beef_discount

  let fv_cost := 8 * 4
  let fv_discount := 0.05 * fv_cost
  let total_fv_cost := fv_cost - fv_discount

  let spices_cost := 2 * 6

  let other_groceries_cost := 37 - 3

  let total_cost := total_beef_cost + total_fv_cost + spices_cost + other_groceries_cost

  let spending_points := (total_cost / 10).floor * 50

  let bonus_points_over_100 := if total_cost > 100 then 250 else 0

  let loyalty_points := 100
  
  spending_points + bonus_points_over_100 + loyalty_points = 850 := by
    sorry

end NUMINAMATH_GPT_martha_points_calculation_l2044_204452


namespace NUMINAMATH_GPT_ratio_equation_solution_l2044_204459

variable (x y z : ℝ)
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)
variables (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)

theorem ratio_equation_solution
  (h : y / (2 * x - z) = (x + y) / (2 * z) ∧ (x + y) / (2 * z) = x / y) :
  x / y = 3 :=
sorry

end NUMINAMATH_GPT_ratio_equation_solution_l2044_204459


namespace NUMINAMATH_GPT_find_x3_l2044_204465

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1)

theorem find_x3
  (x1 x2 : ℝ)
  (h1 : 0 < x1)
  (h2 : x1 < x2)
  (h1_eq : x1 = 1)
  (h2_eq : x2 = Real.exp 3)
  : ∃ x3 : ℝ, x3 = Real.log (2 / 3 + 1 / 3 * Real.exp (Real.exp 3 - 1)) + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x3_l2044_204465


namespace NUMINAMATH_GPT_missy_total_patients_l2044_204493

theorem missy_total_patients 
  (P : ℕ)
  (h1 : ∀ x, (∃ y, y = ↑(1/3) * ↑x) → ∃ z, z = y * (120/100))
  (h2 : ∀ x, 5 * x = 5 * (x - ↑(1/3) * ↑x) + (120/100) * 5 * (↑(1/3) * ↑x))
  (h3 : 64 = 5 * (2/3) * (P : ℕ) + 6 * (1/3) * (P : ℕ)) :
  P = 12 :=
by
  sorry

end NUMINAMATH_GPT_missy_total_patients_l2044_204493


namespace NUMINAMATH_GPT_range_of_a_l2044_204435

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2044_204435


namespace NUMINAMATH_GPT_geometric_sequence_expression_l2044_204490

theorem geometric_sequence_expression (a : ℝ) (a_n: ℕ → ℝ)
  (h1 : a_n 1 = a - 1)
  (h2 : a_n 2 = a + 1)
  (h3 : a_n 3 = a + 4)
  (hn : ∀ n, a_n (n + 1) = a_n n * (a_n 2 / a_n 1)) :
  a_n n = 4 * (3/2)^(n-1) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_expression_l2044_204490


namespace NUMINAMATH_GPT_max_value_of_k_l2044_204487

theorem max_value_of_k (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : ∀ x y : ℝ, 0 < x → 0 < y → (x + 2 * y) / (x * y) ≥ k / (2 * x + y)) :
  k ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_k_l2044_204487


namespace NUMINAMATH_GPT_find_sin_2alpha_l2044_204486

theorem find_sin_2alpha (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) 
    (h2 : 3 * Real.cos (2 * α) = Real.sqrt 2 * Real.sin (π / 4 - α)) : 
  Real.sin (2 * α) = -8 / 9 := 
sorry

end NUMINAMATH_GPT_find_sin_2alpha_l2044_204486


namespace NUMINAMATH_GPT_sector_area_l2044_204439

theorem sector_area (r : ℝ) : (2 * r + 2 * r = 16) → (1/2 * r^2 * 2 = 16) :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_sector_area_l2044_204439


namespace NUMINAMATH_GPT_gcd_1248_1001_l2044_204469

theorem gcd_1248_1001 : Nat.gcd 1248 1001 = 13 := by
  sorry

end NUMINAMATH_GPT_gcd_1248_1001_l2044_204469


namespace NUMINAMATH_GPT_find_constants_l2044_204409

def equation1 (x p q : ℝ) : Prop := (x + p) * (x + q) * (x + 5) = 0
def equation2 (x p q : ℝ) : Prop := (x + 2 * p) * (x + 2) * (x + 3) = 0

def valid_roots1 (p q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ p q ∧ equation1 x₂ p q ∧
  x₁ = -5 ∨ x₁ = -q ∨ x₁ = -p

def valid_roots2 (p q : ℝ) : Prop :=
  ∃ x₃ x₄ : ℝ, x₃ ≠ x₄ ∧ equation2 x₃ p q ∧ equation2 x₄ p q ∧
  (x₃ = -2 * p ∨ x₃ = -2 ∨ x₃ = -3)

theorem find_constants (p q : ℝ) (h1 : valid_roots1 p q) (h2 : valid_roots2 p q) : 100 * p + q = 502 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l2044_204409


namespace NUMINAMATH_GPT_lambs_traded_for_goat_l2044_204442

-- Definitions for the given conditions
def initial_lambs : ℕ := 6
def babies_per_lamb : ℕ := 2 -- each of 2 lambs had 2 babies
def extra_babies : ℕ := 2 * babies_per_lamb
def extra_lambs : ℕ := 7
def current_lambs : ℕ := 14

-- Proof statement for the number of lambs traded
theorem lambs_traded_for_goat : initial_lambs + extra_babies + extra_lambs - current_lambs = 3 :=
by
  sorry

end NUMINAMATH_GPT_lambs_traded_for_goat_l2044_204442


namespace NUMINAMATH_GPT_polynomial_identity_l2044_204441

theorem polynomial_identity (x : ℝ) (hx : x^2 + x - 1 = 0) : x^4 + 2*x^3 - 3*x^2 - 4*x + 5 = 2 :=
sorry

end NUMINAMATH_GPT_polynomial_identity_l2044_204441


namespace NUMINAMATH_GPT_constant_ratio_of_arithmetic_sequence_l2044_204401

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ a₁ d : ℝ, ∀ n : ℕ, a n = a₁ + (n-1) * d

-- The main theorem stating the result
theorem constant_ratio_of_arithmetic_sequence 
  (a : ℕ → ℝ) (c : ℝ) (h_seq : arithmetic_sequence a)
  (h_const : ∀ n : ℕ, a n ≠ 0 ∧ a (2 * n) ≠ 0 ∧ a n / a (2 * n) = c) :
  c = 1 ∨ c = 1 / 2 :=
sorry

end NUMINAMATH_GPT_constant_ratio_of_arithmetic_sequence_l2044_204401


namespace NUMINAMATH_GPT_initial_glass_bottles_count_l2044_204405

namespace Bottles

variable (G P : ℕ)

/-- The weight of some glass bottles is 600 g. 
    The total weight of 4 glass bottles and 5 plastic bottles is 1050 g.
    A glass bottle is 150 g heavier than a plastic bottle.
    Prove that the number of glass bottles initially weighed is 3. -/
theorem initial_glass_bottles_count (h1 : G * (P + 150) = 600)
  (h2 : 4 * (P + 150) + 5 * P = 1050)
  (h3 : P + 150 > P) :
  G = 3 :=
  by sorry

end Bottles

end NUMINAMATH_GPT_initial_glass_bottles_count_l2044_204405


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2044_204431

theorem quadratic_has_two_distinct_real_roots (k : ℝ) (hk1 : k ≠ 0) (hk2 : k < 0) : (5 - 4 * k) > 0 :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2044_204431


namespace NUMINAMATH_GPT_perimeter_square_C_l2044_204466

theorem perimeter_square_C 
  (a b c : ℝ) 
  (ha : 4 * a = 16) 
  (hb : 4 * b = 28) 
  (hc : c = |a - b|) : 
  4 * c = 12 := 
sorry

end NUMINAMATH_GPT_perimeter_square_C_l2044_204466


namespace NUMINAMATH_GPT_intersection_A_B_l2044_204480

def A : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x^2) }
def B : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 1 } := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2044_204480


namespace NUMINAMATH_GPT_unique_point_on_circle_conditions_l2044_204434

noncomputable def point : Type := ℝ × ℝ

-- Define points A and B
def A : point := (-1, 4)
def B : point := (2, 1)

def PA_squared (P : point) : ℝ :=
  let (x, y) := P
  (x + 1) ^ 2 + (y - 4) ^ 2

def PB_squared (P : point) : ℝ :=
  let (x, y) := P
  (x - 2) ^ 2 + (y - 1) ^ 2

-- Define circle C
def on_circle (a : ℝ) (P : point) : Prop :=
  let (x, y) := P
  (x - a) ^ 2 + (y - 2) ^ 2 = 16

-- Define the condition PA² + 2PB² = 24
def condition (P : point) : Prop :=
  PA_squared P + 2 * PB_squared P = 24

-- The main theorem stating the possible values of a
theorem unique_point_on_circle_conditions :
  ∃ (a : ℝ), ∀ (P : point), on_circle a P → condition P → (a = -1 ∨ a = 3) :=
sorry

end NUMINAMATH_GPT_unique_point_on_circle_conditions_l2044_204434


namespace NUMINAMATH_GPT_union_sets_intersection_complement_sets_l2044_204438

universe u
variable {U A B : Set ℝ}

def universal_set : Set ℝ := {x | x ≤ 4}
def set_A : Set ℝ := {x | -2 < x ∧ x < 3}
def set_B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

theorem union_sets : set_A ∪ set_B = {x | -3 ≤ x ∧ x < 3} := by
  sorry

theorem intersection_complement_sets :
  set_A ∩ (universal_set \ set_B) = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_union_sets_intersection_complement_sets_l2044_204438


namespace NUMINAMATH_GPT_Jackson_money_is_125_l2044_204422

-- Definitions of given conditions
def Williams_money : ℕ := sorry
def Jackson_money : ℕ := 5 * Williams_money

-- Given condition: together they have $150
def total_money_condition : Prop := 
  Jackson_money + Williams_money = 150

-- Proof statement
theorem Jackson_money_is_125 
  (h1 : total_money_condition) : 
  Jackson_money = 125 := 
by
  sorry

end NUMINAMATH_GPT_Jackson_money_is_125_l2044_204422


namespace NUMINAMATH_GPT_Evelyn_bottle_caps_l2044_204433

theorem Evelyn_bottle_caps (initial_caps found_caps total_caps : ℕ)
  (h1 : initial_caps = 18)
  (h2 : found_caps = 63) :
  total_caps = 81 :=
by
  sorry

end NUMINAMATH_GPT_Evelyn_bottle_caps_l2044_204433


namespace NUMINAMATH_GPT_max_value_of_abs_asinx_plus_b_l2044_204464

theorem max_value_of_abs_asinx_plus_b 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, |a * (Real.cos x)^2 + b * Real.sin x + c| ≤ 1) : 
  ∃ M, M = 2 ∧ ∀ x : ℝ, |a * Real.sin x + b| ≤ M :=
by
  use 2
  sorry

end NUMINAMATH_GPT_max_value_of_abs_asinx_plus_b_l2044_204464


namespace NUMINAMATH_GPT_time_for_plastic_foam_drift_l2044_204421

def boat_speed_in_still_water : ℝ := sorry
def speed_of_water_flow : ℝ := sorry
def distance_between_docks : ℝ := sorry

theorem time_for_plastic_foam_drift (x y s t : ℝ) 
(hx : 6 * (x + y) = s)
(hy : 8 * (x - y) = s)
(t_eq : t = s / y) : 
t = 48 := 
sorry

end NUMINAMATH_GPT_time_for_plastic_foam_drift_l2044_204421


namespace NUMINAMATH_GPT_P_subset_M_l2044_204403

def P : Set ℝ := {x | x^2 - 6 * x + 9 = 0}
def M : Set ℝ := {x | x > 1}

theorem P_subset_M : P ⊂ M := by sorry

end NUMINAMATH_GPT_P_subset_M_l2044_204403


namespace NUMINAMATH_GPT_recurring_decimal_to_fraction_l2044_204413

theorem recurring_decimal_to_fraction : ∀ x : ℝ, (x = 7 + (1/3 : ℝ)) → x = (22/3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_recurring_decimal_to_fraction_l2044_204413


namespace NUMINAMATH_GPT_average_salary_all_workers_l2044_204415

-- Definitions based on the conditions
def technicians_avg_salary := 16000
def rest_avg_salary := 6000
def total_workers := 35
def technicians := 7
def rest_workers := total_workers - technicians

-- Prove that the average salary of all workers is 8000
theorem average_salary_all_workers :
  (technicians * technicians_avg_salary + rest_workers * rest_avg_salary) / total_workers = 8000 := by
  sorry

end NUMINAMATH_GPT_average_salary_all_workers_l2044_204415


namespace NUMINAMATH_GPT_distance_A_B_l2044_204454

theorem distance_A_B (d : ℝ)
  (speed_A : ℝ := 100) (speed_B : ℝ := 90) (speed_C : ℝ := 75)
  (location_A location_B : point) (is_at_A : location_A = point_A) (is_at_B : location_B = point_B)
  (t_meet_AB : ℝ := d / (speed_A + speed_B))
  (t_meet_AC : ℝ := t_meet_AB + 3)
  (distance_AC : ℝ := speed_A * 3)
  (distance_C : ℝ := speed_C * t_meet_AC) :
  d = 650 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_A_B_l2044_204454


namespace NUMINAMATH_GPT_systematic_sampling_first_group_l2044_204475

/-- 
    In a systematic sampling of size 20 from 160 students,
    where students are divided into 20 groups evenly,
    if the number drawn from the 15th group is 116,
    then the number drawn from the first group is 4.
-/
theorem systematic_sampling_first_group (groups : ℕ) (students : ℕ) (interval : ℕ)
  (number_from_15th : ℕ) (number_from_first : ℕ) :
  groups = 20 →
  students = 160 →
  interval = 8 →
  number_from_15th = 116 →
  number_from_first = number_from_15th - interval * 14 →
  number_from_first = 4 :=
by
  intros hgroups hstudents hinterval hnumber_from_15th hequation
  sorry

end NUMINAMATH_GPT_systematic_sampling_first_group_l2044_204475


namespace NUMINAMATH_GPT_quadratic_solution_identity_l2044_204483

theorem quadratic_solution_identity {a b c : ℝ} (h1 : a ≠ 0) (h2 : a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) : 
  a + b + c = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_identity_l2044_204483


namespace NUMINAMATH_GPT_grade12_sample_size_correct_l2044_204450

-- Given conditions
def grade10_students : ℕ := 1200
def grade11_students : ℕ := 900
def grade12_students : ℕ := 1500
def total_sample_size : ℕ := 720
def total_students : ℕ := grade10_students + grade11_students + grade12_students

-- Stratified sampling calculation
def fraction_grade12 : ℚ := grade12_students / total_students
def number_grade12_in_sample : ℚ := fraction_grade12 * total_sample_size

-- Main theorem
theorem grade12_sample_size_correct :
  number_grade12_in_sample = 300 := by
  sorry

end NUMINAMATH_GPT_grade12_sample_size_correct_l2044_204450


namespace NUMINAMATH_GPT_angle_B_shape_triangle_l2044_204489

variable {a b c R : ℝ} 

theorem angle_B_shape_triangle 
  (h1 : c > a ∧ c > b)
  (h2 : b = Real.sqrt 3 * R)
  (h3 : b * Real.sin (Real.arcsin (b / (2 * R))) = (a + c) * Real.sin (Real.arcsin (a / (2 * R)))) :
  (Real.arcsin (b / (2 * R)) = Real.pi / 3 ∧ a = c / 2 ∧ Real.arcsin (a / (2 * R)) = Real.pi / 6 ∧ Real.arcsin (c / (2 * R)) = Real.pi / 2) :=
by
  sorry

end NUMINAMATH_GPT_angle_B_shape_triangle_l2044_204489


namespace NUMINAMATH_GPT_sequence_sqrt_l2044_204408

theorem sequence_sqrt (a : ℕ → ℝ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a n > 0)
  (h₃ : ∀ n, a (n+1 - 1) ^ 2 = a (n+1) ^ 2 + 4) :
  ∀ n, a n = Real.sqrt (4 * n - 3) :=
by
  sorry

end NUMINAMATH_GPT_sequence_sqrt_l2044_204408
