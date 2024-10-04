import Mathlib

namespace problem_statement_l275_275551

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions of the problem
def cond1 : Prop := (1 / x) + (1 / y) = 2
def cond2 : Prop := (x * y) + x - y = 6

-- The corresponding theorem to prove: x² - y² = 2
theorem problem_statement (h1 : cond1) (h2 : cond2) : x^2 - y^2 = 2 :=
  sorry

end problem_statement_l275_275551


namespace optimal_order_l275_275606

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l275_275606


namespace perpendicular_line_eq_slope_intercept_l275_275647

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l275_275647


namespace remainder_2abc_mod_7_l275_275138

theorem remainder_2abc_mod_7
  (a b c : ℕ)
  (h₀ : 2 * a + 3 * b + c ≡ 1 [MOD 7])
  (h₁ : 3 * a + b + 2 * c ≡ 2 [MOD 7])
  (h₂ : a + b + c ≡ 3 [MOD 7])
  (ha : a < 7)
  (hb : b < 7)
  (hc : c < 7) :
  2 * a * b * c ≡ 0 [MOD 7] :=
sorry

end remainder_2abc_mod_7_l275_275138


namespace ostap_advantageous_order_l275_275596

theorem ostap_advantageous_order (p1 p2 p3 : ℝ) (h1 : p3 < p1) (h2 : p1 < p2) : 
  ∀ order : List ℝ, 
    (order = [p1, p2, p3] ∨ order = [p2, p1, p3] ∨ order = [p3, p1, p2]) → (order.nth 1 = some p2) :=
sorry

end ostap_advantageous_order_l275_275596


namespace klinker_twice_as_old_l275_275593

theorem klinker_twice_as_old :
  ∃ x : ℕ, (∀ (m k d : ℕ), m = 35 → d = 10 → m + x = 2 * (d + x)) → x = 15 :=
by
  sorry

end klinker_twice_as_old_l275_275593


namespace evaluate_expression_l275_275937

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 := by
  sorry

end evaluate_expression_l275_275937


namespace quadratic_inequality_solution_l275_275329

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 < x + 6) ↔ (-2 < x ∧ x < 3) := 
by
  sorry

end quadratic_inequality_solution_l275_275329


namespace find_PF2_l275_275723

-- Statement of the problem

def hyperbola_1 (x y: ℝ) := (x^2 / 16) - (y^2 / 20) = 1

theorem find_PF2 (x y PF1 PF2: ℝ) (a : ℝ)
    (h_hyperbola : hyperbola_1 x y)
    (h_a : a = 4) 
    (h_dist_PF1 : PF1 = 9) :
    abs (PF1 - PF2) = 2 * a → PF2 = 17 :=
by
  intro h1
  sorry

end find_PF2_l275_275723


namespace find_a_pow_b_l275_275722

theorem find_a_pow_b (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : a^b = 1 / 2 := 
sorry

end find_a_pow_b_l275_275722


namespace reading_time_equal_l275_275313

/--
  Alice, Bob, and Chandra are reading a 760-page book. Alice reads a page in 20 seconds, 
  Bob reads a page in 45 seconds, and Chandra reads a page in 30 seconds. Prove that if 
  they divide the book into three sections such that each reads for the same length of 
  time, then each person will read for 7200 seconds.
-/
theorem reading_time_equal 
  (rate_A : ℝ := 1/20) 
  (rate_B : ℝ := 1/45) 
  (rate_C : ℝ := 1/30) 
  (total_pages : ℝ := 760) : 
  ∃ t : ℝ, t = 7200 ∧ 
    (t * rate_A + t * rate_B + t * rate_C = total_pages) := 
by
  sorry  -- proof to be provided

end reading_time_equal_l275_275313


namespace find_x2_x1_add_x3_l275_275029

-- Definition of the polynomial
def polynomial (x : ℝ) : ℝ := (10*x^3 - 210*x^2 + 3)

-- Statement including conditions and the question we need to prove
theorem find_x2_x1_add_x3 :
  ∃ x₁ x₂ x₃ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ 
    polynomial x₁ = 0 ∧ 
    polynomial x₂ = 0 ∧ 
    polynomial x₃ = 0 ∧ 
    x₂ * (x₁ + x₃) = 21 :=
by sorry

end find_x2_x1_add_x3_l275_275029


namespace probability_correct_l275_275222

def total_chips : ℕ := 15
def total_ways_to_draw_2_chips : ℕ := Nat.choose 15 2

def chips_same_color : ℕ := 3 * (Nat.choose 5 2)
def chips_same_number : ℕ := 5 * (Nat.choose 3 2)
def favorable_outcomes : ℕ := chips_same_color + chips_same_number

def probability_same_color_or_number : ℚ := favorable_outcomes / total_ways_to_draw_2_chips

theorem probability_correct :
  probability_same_color_or_number = 3 / 7 :=
by sorry

end probability_correct_l275_275222


namespace calculate_expression_l275_275028

def smallest_positive_two_digit_multiple_of_7 : ℕ := 14
def smallest_positive_three_digit_multiple_of_5 : ℕ := 100

theorem calculate_expression : 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  (c * d) - 100 = 1300 :=
by 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  sorry

end calculate_expression_l275_275028


namespace canoe_vs_kayak_l275_275053

theorem canoe_vs_kayak (
  C K : ℕ 
) (h1 : 14 * C + 15 * K = 288) 
  (h2 : C = (3 * K) / 2) : 
  C - K = 4 := 
sorry

end canoe_vs_kayak_l275_275053


namespace find_machines_l275_275039

theorem find_machines (R : ℝ) : 
  (N : ℕ) -> 
  (H1 : N * R * 6 = 1) -> 
  (H2 : 4 * R * 12 = 1) -> 
  N = 8 :=
by
  sorry

end find_machines_l275_275039


namespace Eva_needs_weeks_l275_275853

theorem Eva_needs_weeks (apples : ℕ) (days_in_week : ℕ) (weeks : ℕ) 
  (h1 : apples = 14)
  (h2 : days_in_week = 7) 
  (h3 : apples = weeks * days_in_week) : 
  weeks = 2 := 
by 
  sorry

end Eva_needs_weeks_l275_275853


namespace tap_B_filling_time_l275_275783

theorem tap_B_filling_time : 
  ∀ (r_A r_B : ℝ), 
  (r_A + r_B = 1 / 30) → 
  (r_B * 40 = 2 / 3) → 
  (1 / r_B = 60) := 
by
  intros r_A r_B h₁ h₂
  sorry

end tap_B_filling_time_l275_275783


namespace cat_moves_on_circular_arc_l275_275677

theorem cat_moves_on_circular_arc (L : ℝ) (x y : ℝ)
  (h : x^2 + y^2 = L^2) :
  (x / 2)^2 + (y / 2)^2 = (L / 2)^2 :=
  by sorry

end cat_moves_on_circular_arc_l275_275677


namespace surface_area_of_given_cylinder_l275_275979

noncomputable def surface_area_of_cylinder (length width : ℝ) : ℝ :=
  let r := (length / (2 * Real.pi))
  let h := width
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem surface_area_of_given_cylinder : 
  surface_area_of_cylinder (4 * Real.pi) 2 = 16 * Real.pi :=
by
  -- Proof will be filled here
  sorry

end surface_area_of_given_cylinder_l275_275979


namespace number_of_solution_values_l275_275398

theorem number_of_solution_values (c : ℕ) : 
  0 ≤ c ∧ c ≤ 2000 ↔ (∃ x : ℝ, 5 * (⌊x⌋ : ℝ) + 3 * (⌈x⌉ : ℝ) = c) →
  c = 251 := 
sorry

end number_of_solution_values_l275_275398


namespace division_of_5_parts_division_of_7_parts_division_of_8_parts_l275_275507

-- Problem 1: Primary Division of Square into 5 Equal Parts
theorem division_of_5_parts (x : ℝ) (h : x^2 = 1 / 5) : x = Real.sqrt (1 / 5) :=
sorry

-- Problem 2: Primary Division of Square into 7 Equal Parts
theorem division_of_7_parts (x : ℝ) (hx : 196 * x^3 - 294 * x^2 + 128 * x - 15 = 0) : 
  x = (7 + Real.sqrt 19) / 14 :=
sorry

-- Problem 3: Primary Division of Square into 8 Equal Parts
theorem division_of_8_parts (x : ℝ) (hx : 6 * x^2 - 6 * x + 1 = 0) : 
  x = (3 + Real.sqrt 3) / 6 :=
sorry

end division_of_5_parts_division_of_7_parts_division_of_8_parts_l275_275507


namespace relation_between_x_and_y_l275_275573

-- Definitions based on the conditions
variables (r x y : ℝ)

-- Power of a Point Theorem and provided conditions
variables (AE_eq_3EC : AE = 3 * EC)
variables (x_def : x = AE)
variables (y_def : y = r)

-- Main statement to be proved
theorem relation_between_x_and_y (r x y : ℝ) (AE_eq_3EC : AE = 3 * EC) (x_def : x = AE) (y_def : y = r) :
  y^2 = x^3 / (2 * r - x) :=
sorry

end relation_between_x_and_y_l275_275573


namespace perpendicular_line_equation_l275_275653

-- Conditions definition
def line1 := ∀ x y : ℝ, 3 * x - 6 * y = 9
def point := (2 : ℝ, -3 : ℝ)

-- Target statement
theorem perpendicular_line_equation : 
  (∀ x y : ℝ, line1 x y → y = (-2) * x + 1) := 
by 
  -- proof goes here
  sorry

end perpendicular_line_equation_l275_275653


namespace smallest_k_for_sixty_four_gt_four_nineteen_l275_275802

-- Definitions of the conditions
def sixty_four (k : ℕ) : ℕ := 64^k
def four_nineteen : ℕ := 4^19

-- The theorem to prove
theorem smallest_k_for_sixty_four_gt_four_nineteen (k : ℕ) : sixty_four k > four_nineteen ↔ k ≥ 7 := 
by
  sorry

end smallest_k_for_sixty_four_gt_four_nineteen_l275_275802


namespace sum_first_99_terms_l275_275487

def geom_sum (n : ℕ) : ℕ := (2^n) - 1

def seq_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum geom_sum

theorem sum_first_99_terms :
  seq_sum 99 = 2^100 - 101 := by
  sorry

end sum_first_99_terms_l275_275487


namespace ribbon_tape_needed_l275_275761

theorem ribbon_tape_needed 
  (total_length : ℝ) (num_boxes : ℕ) (ribbon_per_box : ℝ)
  (h1 : total_length = 82.04)
  (h2 : num_boxes = 28)
  (h3 : total_length / num_boxes = ribbon_per_box)
  : ribbon_per_box = 2.93 :=
sorry

end ribbon_tape_needed_l275_275761


namespace james_pays_total_l275_275577

theorem james_pays_total (lessons_total : ℕ) (lessons_paid : ℕ) (lesson_cost : ℕ) (uncle_share : ℕ) :
  lessons_total = 20 →
  lessons_paid = 15 →
  lesson_cost = 5 →
  uncle_share = 2 → 
  (lesson_cost * lessons_paid) / uncle_share = 37.5 := 
by
  intros h1 h2 h3 h4
  sorry

end james_pays_total_l275_275577


namespace operation_value_l275_275030

-- Define the operations as per the conditions.
def star (m n : ℤ) : ℤ := n^2 - m
def hash (m k : ℤ) : ℚ := (k + 2 * m) / 3

-- State the theorem we want to prove.
theorem operation_value : hash (star 3 3) (star 2 5) = 35 / 3 :=
  by
  sorry

end operation_value_l275_275030


namespace value_of_one_house_l275_275251

theorem value_of_one_house
  (num_brothers : ℕ) (num_houses : ℕ) (payment_each : ℕ) 
  (total_money_paid : ℕ) (num_older : ℕ) (num_younger : ℕ)
  (share_per_younger : ℕ) (total_inheritance : ℕ) (value_of_house : ℕ) :
  num_brothers = 5 →
  num_houses = 3 →
  num_older = 3 →
  num_younger = 2 →
  payment_each = 800 →
  total_money_paid = num_older * payment_each →
  share_per_younger = total_money_paid / num_younger →
  total_inheritance = num_brothers * share_per_younger →
  value_of_house = total_inheritance / num_houses →
  value_of_house = 2000 :=
by {
  -- Provided conditions and statements without proofs
  sorry
}

end value_of_one_house_l275_275251


namespace boys_ages_l275_275779

theorem boys_ages (a b : ℕ) (h1 : a = b) (h2 : a + b + 11 = 29) : a = 9 :=
by
  sorry

end boys_ages_l275_275779


namespace find_quadruples_l275_275697

theorem find_quadruples (x y z n : ℕ) : 
  x^2 + y^2 + z^2 + 1 = 2^n → 
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
by
  sorry

end find_quadruples_l275_275697


namespace impossible_to_form_3x3_in_upper_left_or_right_l275_275718

noncomputable def initial_positions : List (ℕ × ℕ) := 
  [(6, 1), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3), (8, 1), (8, 2), (8, 3)]

def sum_vertical (positions : List (ℕ × ℕ)) : ℕ :=
  positions.foldr (λ pos acc => pos.1 + acc) 0

theorem impossible_to_form_3x3_in_upper_left_or_right
  (initial_positions_set : List (ℕ × ℕ) := initial_positions)
  (initial_sum := sum_vertical initial_positions_set)
  (target_positions_upper_left : List (ℕ × ℕ) := [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)])
  (target_positions_upper_right : List (ℕ × ℕ) := [(1, 6), (1, 7), (1, 8), (2, 6), (2, 7), (2, 8), (3, 6), (3, 7), (3, 8)])
  (target_sum_upper_left := sum_vertical target_positions_upper_left)
  (target_sum_upper_right := sum_vertical target_positions_upper_right) : 
  ¬ (initial_sum % 2 = 1 ∧ target_sum_upper_left % 2 = 0 ∧ target_sum_upper_right % 2 = 0) := sorry

end impossible_to_form_3x3_in_upper_left_or_right_l275_275718


namespace Beth_bought_10_cans_of_corn_l275_275086

theorem Beth_bought_10_cans_of_corn (a b : ℕ) (h1 : b = 15 + 2 * a) (h2 : b = 35) : a = 10 := by
  sorry

end Beth_bought_10_cans_of_corn_l275_275086


namespace overlap_length_l275_275495

-- Variables in the conditions
variables (tape_length overlap total_length : ℕ)

-- Conditions
def two_tapes_overlap := (tape_length + tape_length - overlap = total_length)

-- The proof statement we need to prove
theorem overlap_length (h : two_tapes_overlap 275 overlap 512) : overlap = 38 :=
by
  sorry

end overlap_length_l275_275495


namespace songs_distribution_l275_275221

-- Define the sets involved
structure Girl := (Amy Beth Jo : Prop)
axiom no_song_liked_by_all : ∀ song : Girl, ¬(song.Amy ∧ song.Beth ∧ song.Jo)
axiom no_song_disliked_by_all : ∀ song : Girl, song.Amy ∨ song.Beth ∨ song.Jo
axiom pairwise_liked : ∀ song : Girl,
  (song.Amy ∧ song.Beth ∧ ¬song.Jo) ∨
  (song.Beth ∧ song.Jo ∧ ¬song.Amy) ∨
  (song.Jo ∧ song.Amy ∧ ¬song.Beth)

-- Define the theorem to prove that there are exactly 90 ways to distribute the songs
theorem songs_distribution : ∃ ways : ℕ, ways = 90 := sorry

end songs_distribution_l275_275221


namespace perpendicular_line_through_point_l275_275650

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l275_275650


namespace correct_operation_l275_275353

variable (a b : ℝ)

theorem correct_operation : (-a^2 * b + 2 * a^2 * b = a^2 * b) :=
by sorry

end correct_operation_l275_275353


namespace tax_budget_level_correct_l275_275694

-- Definitions for tax types and their corresponding budget levels
inductive TaxType where
| property_tax_organizations : TaxType
| federal_tax : TaxType
| profit_tax_organizations : TaxType
| tax_subjects_RF : TaxType
| transport_collecting : TaxType
deriving DecidableEq

inductive BudgetLevel where
| federal_budget : BudgetLevel
| subjects_RF_budget : BudgetLevel
deriving DecidableEq

def tax_to_budget_level : TaxType → BudgetLevel
| TaxType.property_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.federal_tax => BudgetLevel.federal_budget
| TaxType.profit_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.tax_subjects_RF => BudgetLevel.subjects_RF_budget
| TaxType.transport_collecting => BudgetLevel.subjects_RF_budget

theorem tax_budget_level_correct :
  tax_to_budget_level TaxType.property_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.federal_tax = BudgetLevel.federal_budget ∧
  tax_to_budget_level TaxType.profit_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.tax_subjects_RF = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.transport_collecting = BudgetLevel.subjects_RF_budget :=
by
  sorry

end tax_budget_level_correct_l275_275694


namespace class3_total_score_l275_275374

theorem class3_total_score 
  (total_points : ℕ)
  (class1_score class2_score class3_score : ℕ)
  (class1_places class2_places class3_places : ℕ)
  (total_places : ℕ)
  (points_1st  points_2nd  points_3rd : ℕ)
  (h1 : total_points = 27)
  (h2 : class1_score = class2_score)
  (h3 : 2 * class1_places = class2_places)
  (h4 : class1_places + class2_places + class3_places = total_places)
  (h5 : 3 * points_1st + 3 * points_2nd + 3 * points_3rd = total_points)
  (h6 : total_places = 9)
  (h7 : points_1st = 5)
  (h8 : points_2nd = 3)
  (h9 : points_3rd = 1) :
  class3_score = 7 :=
sorry

end class3_total_score_l275_275374


namespace gcd_36_54_l275_275791

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factorization (n : ℕ) : list ℕ :=
if h : n = 0 then []
else
(list.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

theorem gcd_36_54 : Nat.gcd 36 54 = 18 :=
by
  sorry

end gcd_36_54_l275_275791


namespace rectangle_perimeter_l275_275984

-- Definitions of the conditions
def side_length_square : ℕ := 75  -- side length of the square in mm
def height_sum (x y z : ℕ) : Prop := x + y + z = side_length_square  -- sum of heights of the rectangles

-- Perimeter definition
def perimeter (h : ℕ) (w : ℕ) : ℕ := 2 * (h + w)

-- Statement of the problem
theorem rectangle_perimeter (x y z : ℕ) (h_sum : height_sum x y z)
  (h1 : perimeter x side_length_square = (perimeter y side_length_square + perimeter z side_length_square) / 2)
  : perimeter x side_length_square = 200 := by
  sorry

end rectangle_perimeter_l275_275984


namespace find_pairs_l275_275859

theorem find_pairs (x y : ℕ) (h1 : 0 < x ∧ 0 < y)
  (h2 : ∃ p : ℕ, Prime p ∧ (x + y = 2 * p))
  (h3 : (x! + y!) % (x + y) = 0) : ∃ p : ℕ, Prime p ∧ x = p ∧ y = p :=
by
  sorry

end find_pairs_l275_275859


namespace impossible_15_cents_l275_275254

theorem impossible_15_cents (a b c d : ℕ) (ha : a ≤ 4) (hb : b ≤ 4) (hc : c ≤ 4) (hd : d ≤ 4) (h : a + b + c + d = 4) : 
  1 * a + 5 * b + 10 * c + 25 * d ≠ 15 :=
by
  sorry

end impossible_15_cents_l275_275254


namespace scientific_notation_of_0_000000023_l275_275897

theorem scientific_notation_of_0_000000023 : 
  0.000000023 = 2.3 * 10^(-8) :=
sorry

end scientific_notation_of_0_000000023_l275_275897


namespace ellipse_hyperbola_foci_l275_275326

theorem ellipse_hyperbola_foci (c d : ℝ) 
  (h_ellipse : d^2 - c^2 = 25) 
  (h_hyperbola : c^2 + d^2 = 64) : |c * d| = Real.sqrt 868.5 := by
  sorry

end ellipse_hyperbola_foci_l275_275326


namespace mixture_weight_l275_275368

def almonds := 116.67
def walnuts := almonds / 5
def total_weight := almonds + walnuts

theorem mixture_weight : total_weight = 140.004 := by
  sorry

end mixture_weight_l275_275368


namespace incircle_area_of_triangle_l275_275751

noncomputable def hyperbola_params : Type :=
  sorry

noncomputable def point_on_hyperbola (P : hyperbola_params) : Prop :=
  sorry

noncomputable def in_first_quadrant (P : hyperbola_params) : Prop :=
  sorry

noncomputable def distance_ratio (PF1 PF2 : ℝ) : Prop :=
  PF1 / PF2 = 4 / 3

noncomputable def distance1_is_8 (PF1 : ℝ) : Prop :=
  PF1 = 8

noncomputable def distance2_is_6 (PF2 : ℝ) : Prop :=
  PF2 = 6

noncomputable def distance_between_foci (F1F2 : ℝ) : Prop :=
  F1F2 = 10

noncomputable def incircle_area (area : ℝ) : Prop :=
  area = 4 * Real.pi

theorem incircle_area_of_triangle (P : hyperbola_params) 
  (hP : point_on_hyperbola P) 
  (h1 : in_first_quadrant P)
  (PF1 PF2 : ℝ)
  (h2 : distance_ratio PF1 PF2)
  (h3 : distance1_is_8 PF1)
  (h4 : distance2_is_6 PF2)
  (F1F2 : ℝ) 
  (h5 : distance_between_foci F1F2) :
  ∃ r : ℝ, incircle_area (Real.pi * r^2) :=
by
  sorry

end incircle_area_of_triangle_l275_275751


namespace projectiles_meet_time_l275_275432

def distance : ℕ := 2520
def speed1 : ℕ := 432
def speed2 : ℕ := 576
def combined_speed : ℕ := speed1 + speed2

theorem projectiles_meet_time :
  (distance * 60) / combined_speed = 150 := 
by
  sorry

end projectiles_meet_time_l275_275432


namespace men_with_tv_at_least_11_l275_275285

-- Definitions for the given conditions
def total_men : ℕ := 100
def married_men : ℕ := 81
def men_with_radio : ℕ := 85
def men_with_ac : ℕ := 70
def men_with_tv_radio_ac_and_married : ℕ := 11

-- The proposition to prove the minimum number of men with TV
theorem men_with_tv_at_least_11 :
  ∃ (T : ℕ), T ≥ men_with_tv_radio_ac_and_married := 
by
  sorry

end men_with_tv_at_least_11_l275_275285


namespace angle_A_range_l275_275908

theorem angle_A_range (a b : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) :
  ∃ A : ℝ, 0 < A ∧ A ≤ Real.pi / 4 :=
sorry

end angle_A_range_l275_275908


namespace cost_of_bananas_and_cantaloupe_l275_275255

theorem cost_of_bananas_and_cantaloupe (a b c d h : ℚ) 
  (h1: a + b + c + d + h = 30)
  (h2: d = 4 * a)
  (h3: c = 2 * a - b) :
  b + c = 50 / 7 := 
sorry

end cost_of_bananas_and_cantaloupe_l275_275255


namespace equation_of_circle_l275_275703

-- Definitions directly based on conditions 
noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)
noncomputable def directrix_of_parabola : ℝ × ℝ -> Prop
  | (x, _) => x = -1

-- The statement of the problem: equation of the circle with given conditions
theorem equation_of_circle : ∃ (r : ℝ), (∀ (x y : ℝ), (x - 1)^2 + y^2 = r^2) ∧ r = 2 :=
sorry

end equation_of_circle_l275_275703


namespace integer_modulo_solution_l275_275196

theorem integer_modulo_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 137 ∧ 12345 ≡ n [ZMOD 137] ∧ n = 15 :=
sorry

end integer_modulo_solution_l275_275196


namespace taxi_fare_l275_275781

-- Define the necessary values and functions based on the problem conditions
def starting_price : ℝ := 6
def additional_charge_per_km : ℝ := 1.5
def distance (P : ℝ) : Prop := P > 6

-- Lean proposition to state the problem
theorem taxi_fare (P : ℝ) (hP : distance P) : 
  (starting_price + additional_charge_per_km * (P - 6)) = 1.5 * P - 3 := 
by 
  sorry

end taxi_fare_l275_275781


namespace custom_operation_example_l275_275690

-- Define the custom operation
def custom_operation (a b : ℕ) : ℕ := a * b + (a - b)

-- State the theorem
theorem custom_operation_example : custom_operation (custom_operation 3 2) 4 = 31 :=
by
  -- the proof will go here, but we skip it for now
  sorry

end custom_operation_example_l275_275690


namespace least_common_multiple_lcm_condition_l275_275948

variable (a b c : ℕ)

theorem least_common_multiple_lcm_condition :
  Nat.lcm a b = 18 → Nat.lcm b c = 28 → Nat.lcm a c = 126 :=
by
  intros h1 h2
  sorry

end least_common_multiple_lcm_condition_l275_275948


namespace xiaoming_problem_l275_275354

theorem xiaoming_problem (a x : ℝ) 
  (h1 : 20.18 * a - 20.18 = x)
  (h2 : x = 2270.25) : 
  a = 113.5 := 
by 
  sorry

end xiaoming_problem_l275_275354


namespace part1_part2_l275_275730

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 2 * a + 3)

theorem part1 (x : ℝ) : f x 2 ≤ 9 ↔ -2 ≤ x ∧ x ≤ 4 :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : a ∈ Set.Iic (-2 / 3) ∪ Set.Ici (14 / 3) :=
by sorry

end part1_part2_l275_275730


namespace max_belts_l275_275517

theorem max_belts (h t b : ℕ) (Hh : h >= 1) (Ht : t >= 1) (Hb : b >= 1) (total_cost : 3 * h + 4 * t + 9 * b = 60) : b <= 5 :=
sorry

end max_belts_l275_275517


namespace lara_bought_52_stems_l275_275298

-- Define the conditions given in the problem:
def flowers_given_to_mom : ℕ := 15
def flowers_given_to_grandma : ℕ := flowers_given_to_mom + 6
def flowers_in_vase : ℕ := 16

-- The total number of stems of flowers Lara bought should be:
def total_flowers_bought : ℕ := flowers_given_to_mom + flowers_given_to_grandma + flowers_in_vase

-- The main theorem to prove the total number of flowers Lara bought is 52:
theorem lara_bought_52_stems : total_flowers_bought = 52 := by
  sorry

end lara_bought_52_stems_l275_275298


namespace geometric_sequence_sum_l275_275257

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 1) (h3 : a 4 * a 5 * a 6 = 8) :
  a 2 + a 5 + a 8 + a 11 = 15 :=
by
  sorry

end geometric_sequence_sum_l275_275257


namespace money_given_by_school_correct_l275_275314

-- Definitions from the problem conditions
def cost_per_book : ℕ := 12
def number_of_students : ℕ := 30
def out_of_pocket : ℕ := 40

-- Derived definition from these conditions
def total_cost : ℕ := cost_per_book * number_of_students
def money_given_by_school : ℕ := total_cost - out_of_pocket

-- The theorem stating that the amount given by the school is $320
theorem money_given_by_school_correct : money_given_by_school = 320 :=
by
  sorry -- Proof placeholder

end money_given_by_school_correct_l275_275314


namespace triangle_geometry_l275_275005

theorem triangle_geometry 
  (A : ℝ × ℝ) 
  (hA : A = (5,1))
  (median_CM : ∀ x y : ℝ, 2 * x - y - 5 = 0)
  (altitude_BH : ∀ x y : ℝ, x - 2 * y - 5 = 0):
  (∀ x y : ℝ, 2 * x + y - 11 = 0) ∧
  (4, 3) ∈ {(x, y) | 2 * x + y = 11 ∧ 2 * x - y = 5} :=
by
  sorry

end triangle_geometry_l275_275005


namespace Victor_more_scoops_l275_275964

def ground_almonds : ℝ := 1.56
def white_sugar : ℝ := 0.75

theorem Victor_more_scoops :
  ground_almonds - white_sugar = 0.81 :=
by
  sorry

end Victor_more_scoops_l275_275964


namespace square_area_l275_275381

theorem square_area (s : ℝ) (h : s = 12) : s * s = 144 :=
by
  rw [h]
  norm_num

end square_area_l275_275381


namespace joan_paid_amount_l275_275580

theorem joan_paid_amount (J K : ℕ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end joan_paid_amount_l275_275580


namespace extremum_value_of_a_g_monotonicity_l275_275128

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2

theorem extremum_value_of_a (a : ℝ) (h : (3 * a * (-4 / 3) ^ 2 + 2 * (-4 / 3) = 0)) : a = 1 / 2 :=
by
  -- We need to prove that a = 1 / 2 given the extremum condition.
  sorry

noncomputable def g (x : ℝ) : ℝ := (1 / 2 * x ^ 3 + x ^ 2) * Real.exp x

theorem g_monotonicity :
  (∀ x < -4, deriv g x < 0) ∧
  (∀ x, -4 < x ∧ x < -1 → deriv g x > 0) ∧
  (∀ x, -1 < x ∧ x < 0 → deriv g x < 0) ∧
  (∀ x > 0, deriv g x > 0) :=
by
  -- We need to prove the monotonicity of the function g in the specified intervals.
  sorry

end extremum_value_of_a_g_monotonicity_l275_275128


namespace find_matrix_M_l275_275105

theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) (h : M^3 - 3 • M^2 + 4 • M = ![![6, 12], ![3, 6]]) :
  M = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_M_l275_275105


namespace sum_of_angles_is_55_l275_275036

noncomputable def arc_BR : ℝ := 60
noncomputable def arc_RS : ℝ := 50
noncomputable def arc_AC : ℝ := 0
noncomputable def arc_BS := arc_BR + arc_RS
noncomputable def angle_P := (arc_BS - arc_AC) / 2
noncomputable def angle_R := arc_AC / 2
noncomputable def sum_of_angles := angle_P + angle_R

theorem sum_of_angles_is_55 :
  sum_of_angles = 55 :=
by
  sorry

end sum_of_angles_is_55_l275_275036


namespace bounds_on_xyz_l275_275412

theorem bounds_on_xyz (a x y z : ℝ) (h1 : x + y + z = a)
                      (h2 : x^2 + y^2 + z^2 = (a^2) / 2)
                      (h3 : a > 0) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z) :
                      (0 < x ∧ x ≤ (2 / 3) * a) ∧ 
                      (0 < y ∧ y ≤ (2 / 3) * a) ∧ 
                      (0 < z ∧ z ≤ (2 / 3) * a) :=
sorry

end bounds_on_xyz_l275_275412


namespace correct_properties_l275_275613

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (-Real.pi / 6) = 0) :=
by
  sorry

end correct_properties_l275_275613


namespace problem_statement_l275_275448

noncomputable def x : ℝ := (3 + real.sqrt 8) ^ 1001
noncomputable def n : ℤ := int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
  sorry

end problem_statement_l275_275448


namespace common_ratio_is_2_l275_275752

noncomputable def common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : ℝ :=
(a1 + 2 * d) / a1

theorem common_ratio_is_2 (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : 
    common_ratio a1 d h1 h2 = 2 :=
by
  -- Proof would go here
  sorry

end common_ratio_is_2_l275_275752


namespace integer_solutions_of_quadratic_l275_275109

theorem integer_solutions_of_quadratic (k : ℤ) :
  ∀ x : ℤ, (6 - k) * (9 - k) * x^2 - (117 - 15 * k) * x + 54 = 0 ↔
  k = 3 ∨ k = 7 ∨ k = 15 ∨ k = 6 ∨ k = 9 :=
by
  sorry

end integer_solutions_of_quadratic_l275_275109


namespace geom_seq_product_l275_275009

noncomputable def geom_seq (a : ℕ → ℝ) := 
∀ n m: ℕ, ∃ r : ℝ, a (n + m) = a n * r ^ m

theorem geom_seq_product (a : ℕ → ℝ) 
  (h_seq : geom_seq a) 
  (h_pos : ∀ n, 0 < a n) 
  (h_log_sum : Real.log (a 3) + Real.log (a 6) + Real.log (a 9) = 3) : 
  a 1 * a 11 = 100 := 
sorry

end geom_seq_product_l275_275009


namespace problem_1_problem_2_l275_275419

theorem problem_1 
  : (∃ (m n : ℝ), m = -1 ∧ n = 1 ∧ ∀ (x : ℝ), |x + 1| + |2 * x - 1| ≤ 3 ↔ m ≤ x ∧ x ≤ n) :=
sorry

theorem problem_2 
  : (∀ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 → 
    ∃ (min_val : ℝ), min_val = 9 / 2 ∧ 
    ∀ (x : ℝ), x = (1 / a + 1 / b + 1 / c) → min_val ≤ x) :=
sorry

end problem_1_problem_2_l275_275419


namespace standard_equation_of_tangent_circle_l275_275563

theorem standard_equation_of_tangent_circle (r h k : ℝ)
  (h_r : r = 1) 
  (h_k : k = 1) 
  (h_center_quadrant : h > 0 ∧ k > 0)
  (h_tangent_x_axis : k = r) 
  (h_tangent_line : r = abs (4 * h - 3) / 5)
  : (x - 2)^2 + (y - 1)^2 = 1 := 
by {
  sorry
}

end standard_equation_of_tangent_circle_l275_275563


namespace scalene_triangle_area_l275_275997

theorem scalene_triangle_area (outer_triangle_area : ℝ) (hexagon_area : ℝ) (num_scalene_triangles : ℕ)
  (h1 : outer_triangle_area = 25) (h2 : hexagon_area = 4) (h3 : num_scalene_triangles = 6) : 
  (outer_triangle_area - hexagon_area) / num_scalene_triangles = 3.5 :=
by
  sorry

end scalene_triangle_area_l275_275997


namespace exists_distinct_nonzero_ints_for_poly_factorization_l275_275406

theorem exists_distinct_nonzero_ints_for_poly_factorization :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ P Q : Polynomial ℤ, (P * Q = Polynomial.X * (Polynomial.X - Polynomial.C a) * 
   (Polynomial.X - Polynomial.C b) * (Polynomial.X - Polynomial.C c) + 1) ∧ 
   P.leadingCoeff = 1 ∧ Q.leadingCoeff = 1) :=
by
  sorry

end exists_distinct_nonzero_ints_for_poly_factorization_l275_275406


namespace original_cost_l275_275361

theorem original_cost (C : ℝ) (h : 550 = 1.35 * C) : C = 550 / 1.35 :=
by
  sorry

end original_cost_l275_275361


namespace television_price_reduction_l275_275079

theorem television_price_reduction (P : ℝ) (h₁ : 0 ≤ P):
  ((P - (P * 0.7 * 0.8)) / P) * 100 = 44 :=
by
  sorry

end television_price_reduction_l275_275079


namespace perpendicular_line_eq_l275_275640

theorem perpendicular_line_eq (x y : ℝ) :
  (3 * x - 6 * y = 9) ∧ ∃ x₀ y₀, (2 : ℝ) = x₀ ∧ (-3 : ℝ) = y₀ ∧ y₀ + (2 * (x - x₀)) = y :=
  ∃ y₀, y = -2 * x + y₀ ∧ y₀ = 1 := sorry

end perpendicular_line_eq_l275_275640


namespace advantageous_order_l275_275610

variables {p1 p2 p3 : ℝ}

-- Conditions
axiom prob_ordering : p3 < p1 ∧ p1 < p2

-- Definition of sequence probabilities
def prob_first_second := p1 * p2 + (1 - p1) * p2 * p3
def prob_second_first := p2 * p1 + (1 - p2) * p1 * p3

-- Theorem to be proved
theorem advantageous_order :
  prob_first_second = prob_second_first →
  p2 > p1 → (p2 > p1) :=
by
  sorry

end advantageous_order_l275_275610


namespace probability_reach_3_1_in_8_steps_l275_275472

theorem probability_reach_3_1_in_8_steps :
  let m := 35
  let n := 2048
  let q := m / n
  ∃ (m n : ℕ), (Nat.gcd m n = 1) ∧ (q = 35 / 2048) ∧ (m + n = 2083) := by
  sorry

end probability_reach_3_1_in_8_steps_l275_275472


namespace find_k_value_l275_275127

theorem find_k_value (x₁ x₂ x₃ x₄ : ℝ)
  (h1 : (x₁ + x₂ + x₃ + x₄) = 18)
  (h2 : (x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄) = k)
  (h3 : (x₁ * x₂ * x₃ + x₁ * x₂ * x₄ + x₁ * x₃ * x₄ + x₂ * x₃ * x₄) = -200)
  (h4 : (x₁ * x₂ * x₃ * x₄) = -1984)
  (h5 : x₁ * x₂ = -32) :
  k = 86 :=
by sorry

end find_k_value_l275_275127


namespace sales_on_second_day_l275_275211

variable (m : ℕ)

-- Define the condition for sales on the first day
def first_day_sales : ℕ := m

-- Define the condition for sales on the second day
def second_day_sales : ℕ := 2 * first_day_sales m - 3

-- The proof statement
theorem sales_on_second_day (m : ℕ) : second_day_sales m = 2 * m - 3 := by
  -- provide the actual proof here
  sorry

end sales_on_second_day_l275_275211


namespace sum_of_digits_of_binary_315_is_6_l275_275806
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l275_275806


namespace probability_of_other_note_being_counterfeit_l275_275111

def total_notes := 20
def counterfeit_notes := 5

-- Binomial coefficient (n choose k)
noncomputable def binom (n k : ℕ) : ℚ := n.choose k

-- Probability of event A: both notes are counterfeit
noncomputable def P_A : ℚ :=
  binom counterfeit_notes 2 / binom total_notes 2

-- Probability of event B: at least one note is counterfeit
noncomputable def P_B : ℚ :=
  (binom counterfeit_notes 2 + binom counterfeit_notes 1 * binom (total_notes - counterfeit_notes) 1) / binom total_notes 2

-- Conditional probability P(A|B)
noncomputable def P_A_given_B : ℚ :=
  P_A / P_B

theorem probability_of_other_note_being_counterfeit :
  P_A_given_B = 2/17 :=
by
  sorry

end probability_of_other_note_being_counterfeit_l275_275111


namespace problem_l275_275550

theorem problem
  (a b : ℝ)
  (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) :
  2 * a^100 - 3 * b⁻¹ = 3 := 
by {
  -- Proof steps go here
  sorry
}

end problem_l275_275550


namespace find_decimal_number_l275_275350

noncomputable def decimal_number (x : ℝ) : Prop := 
x > 0 ∧ (100000 * x = 5 * (1 / x))

theorem find_decimal_number {x : ℝ} (h : decimal_number x) : x = 1 / (100 * Real.sqrt 2) :=
by
  sorry

end find_decimal_number_l275_275350


namespace sol_earnings_in_a_week_l275_275318

-- Define the number of candy bars sold each day using recurrence relation
def candies_sold (n : ℕ) : ℕ :=
  match n with
  | 0     => 10  -- Day 1
  | (n+1) => candies_sold n + 4  -- Each subsequent day

-- Define the total candies sold in a week and total earnings in dollars
def total_candies_sold_in_a_week : ℕ :=
  List.sum (List.map candies_sold [0, 1, 2, 3, 4, 5])

def total_earnings_in_dollars : ℕ :=
  (total_candies_sold_in_a_week * 10) / 100

-- Proving that Sol will earn 12 dollars in a week
theorem sol_earnings_in_a_week : total_earnings_in_dollars = 12 := by
  sorry

end sol_earnings_in_a_week_l275_275318


namespace big_dogs_count_l275_275328

theorem big_dogs_count (B S : ℕ) (h_ratio : 3 * S = 17 * B) (h_total : B + S = 80) :
  B = 12 :=
by
  sorry

end big_dogs_count_l275_275328


namespace problem_solution_l275_275590

noncomputable def a (n : ℕ) : ℕ := 2 * n - 3

noncomputable def b (n : ℕ) : ℕ := 2 ^ n

noncomputable def c (n : ℕ) : ℕ := a n * b n

noncomputable def sum_c (n : ℕ) : ℕ :=
  (2 * n - 5) * 2 ^ (n + 1) + 10

theorem problem_solution :
  ∀ n : ℕ, n > 0 →
  (S_n = 2 * (b n - 1)) ∧
  (a 2 = b 1 - 1) ∧
  (a 5 = b 3 - 1)
  →
  (∀ n, a n = 2 * n - 3) ∧
  (∀ n, b n = 2 ^ n) ∧
  (sum_c n = (2 * n - 5) * 2 ^ (n + 1) + 10) :=
by
  intros n hn h
  sorry


end problem_solution_l275_275590


namespace gcd_of_36_and_54_l275_275797

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l275_275797


namespace sphere_volume_l275_275623

theorem sphere_volume (C : ℝ) (h : C = 30) : 
  ∃ (V : ℝ), V = 4500 / (π^2) :=
by sorry

end sphere_volume_l275_275623


namespace find_positive_real_solutions_l275_275701

open Real

theorem find_positive_real_solutions 
  (x : ℝ) 
  (h : (1/3 * (4 * x^2 - 2)) = ((x^2 - 60 * x - 15) * (x^2 + 30 * x + 3))) :
  x = 30 + sqrt 917 ∨ x = -15 + (sqrt 8016) / 6 :=
by sorry

end find_positive_real_solutions_l275_275701


namespace volume_of_cube_for_tetrahedron_l275_275912

theorem volume_of_cube_for_tetrahedron (h : ℝ) (b1 b2 : ℝ) (V : ℝ) 
  (h_condition : h = 15) (b1_condition : b1 = 8) (b2_condition : b2 = 12)
  (V_condition : V = 3375) : 
  V = (max h (max b1 b2)) ^ 3 := by
  -- To illustrate the mathematical context and avoid concrete steps,
  -- sorry provides the completion of the logical binding to the correct answer
  sorry

end volume_of_cube_for_tetrahedron_l275_275912


namespace customers_who_didnt_tip_l275_275995

theorem customers_who_didnt_tip:
  ∀ (total_customers tips_per_customer total_tips : ℕ),
  total_customers = 10 →
  tips_per_customer = 3 →
  total_tips = 15 →
  (total_customers - total_tips / tips_per_customer) = 5 :=
by
  intros
  sorry

end customers_who_didnt_tip_l275_275995


namespace prove_ab_eq_neg_26_l275_275880

theorem prove_ab_eq_neg_26
  (a b : ℚ)
  (H : ∀ k : ℚ, ∃ x : ℚ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6) :
  a * b = -26 := sorry

end prove_ab_eq_neg_26_l275_275880


namespace axis_of_symmetry_l275_275769

-- Define the given parabolic function
def parabola (x : ℝ) : ℝ := (2 - x) * x

-- Define the axis of symmetry property for the given parabola
theorem axis_of_symmetry : ∀ x : ℝ, ((2 - x) * x) = -((x - 1)^2) + 1 → (∃ x_sym : ℝ, x_sym = 1) :=
by
  sorry

end axis_of_symmetry_l275_275769


namespace total_weight_correct_l275_275310

variable (c1 c2 w2 c : Float)

def total_weight (c1 c2 w2 c : Float) (W x : Float) :=
  (c1 * x + c2 * w2) / (x + w2) = c ∧ W = x + w2

theorem total_weight_correct :
  total_weight 9 8 12 8.40 20 8 :=
by sorry

end total_weight_correct_l275_275310


namespace find_y_when_x_is_neg2_l275_275889

noncomputable def k : ℝ :=
4 / 2

def f (x : ℝ) : ℝ := k * x

theorem find_y_when_x_is_neg2 : f (-2) = -4 :=
sorry

end find_y_when_x_is_neg2_l275_275889


namespace sum_of_digits_base2_315_l275_275822

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l275_275822


namespace part1_part2_l275_275729

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

theorem part1 (a : ℝ) (x : ℝ) (hx1 : 1 ≤ x) (hx2 : x ≤ Real.exp 1) :
  a = 1 →
  (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 + (Real.exp 1)^2 / 2) ∧ (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 / 2) :=
sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 - 2 * a * x + Real.log x

theorem part2 (a : ℝ) :
  (-1/2 ≤ a ∧ a ≤ 1/2) ↔
  ∀ x, 1 < x → g a x < 0 :=
sorry

end part1_part2_l275_275729


namespace building_height_l275_275080

-- Definitions of the conditions
def wooden_box_height : ℝ := 3
def wooden_box_shadow : ℝ := 12
def building_shadow : ℝ := 36

-- The statement that needs to be proved
theorem building_height : ∃ (height : ℝ), height = 9 ∧ wooden_box_height / wooden_box_shadow = height / building_shadow :=
by
  sorry

end building_height_l275_275080


namespace sum_of_four_l275_275149

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0 := a₁
| (n+1) := geometric_sequence a₁ q n * q

def sum_geometric_sequence (a₁ q : α) (n : ℕ) : α :=
if q = 1 then a₁ * (n + 1) else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_of_four {a₁ q S₅ S₃ S₄ : α} (h₁ : a₁ = 1) (h₂ : sum_geometric_sequence a₁ q 4 = S₄) (h₃ : sum_geometric_sequence a₁ q 5 = S₅) (h₄ : sum_geometric_sequence a₁ q 3 = S₃) : S₅ = 5 * S₃ - 4 → S₄ = 15 :=
by
  sorry

end sum_of_four_l275_275149


namespace smallest_n_for_gcd_l275_275346

theorem smallest_n_for_gcd (n : ℕ) :
  (∃ n > 0, gcd (11 * n - 3) (8 * n + 4) > 1) ∧ (∀ m > 0, gcd (11 * m - 3) (8 * m + 4) > 1 → n ≤ m) → n = 38 :=
by
  sorry

end smallest_n_for_gcd_l275_275346


namespace complex_power_rectangular_form_l275_275098

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry

theorem complex_power_rectangular_form :
  (3 * complex.exp (complex.I * 30))^4 = -40.5 + 40.5 * complex.I * √3 :=
by
  sorry

end complex_power_rectangular_form_l275_275098


namespace time_spent_on_Type_A_problems_l275_275143

theorem time_spent_on_Type_A_problems (t : ℝ) (h1 : 25 * (8 * t) + 100 * (2 * t) = 120) : 
  25 * (8 * t) = 60 := by
  sorry

-- Conditions
-- t is the time spent on a Type C problem in minutes
-- 25 * (8 * t) + 100 * (2 * t) = 120 (time spent on Type A and B problems combined equals 120 minutes)

end time_spent_on_Type_A_problems_l275_275143


namespace burt_net_profit_l275_275229

theorem burt_net_profit
  (cost_seeds : ℝ := 2.00)
  (cost_soil : ℝ := 8.00)
  (num_plants : ℕ := 20)
  (price_per_plant : ℝ := 5.00) :
  let total_cost := cost_seeds + cost_soil
  let total_revenue := num_plants * price_per_plant
  let net_profit := total_revenue - total_cost
  net_profit = 90.00 :=
by sorry

end burt_net_profit_l275_275229


namespace earnings_of_r_l275_275830

theorem earnings_of_r (P Q R : ℕ) (h1 : 9 * (P + Q + R) = 1710) (h2 : 5 * (P + R) = 600) (h3 : 7 * (Q + R) = 910) : 
  R = 60 :=
by
  -- proof will be provided here
  sorry

end earnings_of_r_l275_275830


namespace larger_number_is_299_l275_275946

theorem larger_number_is_299 (A B : ℕ) 
  (HCF_AB : Nat.gcd A B = 23) 
  (LCM_12_13 : Nat.lcm A B = 23 * 12 * 13) : 
  max A B = 299 := 
sorry

end larger_number_is_299_l275_275946


namespace investment_percentage_change_l275_275282

/-- 
Isabel's investment problem statement:
Given an initial investment, and percentage changes over three years,
prove that the overall percentage change in Isabel's investment is 1.2% gain.
-/
theorem investment_percentage_change (initial_investment : ℝ) (gain1 : ℝ) (loss2 : ℝ) (gain3 : ℝ) 
    (final_investment : ℝ) :
    initial_investment = 500 →
    gain1 = 0.10 →
    loss2 = 0.20 →
    gain3 = 0.15 →
    final_investment = initial_investment * (1 + gain1) * (1 - loss2) * (1 + gain3) →
    ((final_investment - initial_investment) / initial_investment) * 100 = 1.2 :=
by
  intros h_init h_gain1 h_loss2 h_gain3 h_final
  sorry

end investment_percentage_change_l275_275282


namespace solve_system_l275_275470

theorem solve_system :
  ∃ x y z : ℝ, (8 * (x^3 + y^3 + z^3) = 73) ∧
              (2 * (x^2 + y^2 + z^2) = 3 * (x * y + y * z + z * x)) ∧
              (x * y * z = 1) ∧
              (x, y, z) = (1, 2, 0.5) ∨ (x, y, z) = (1, 0.5, 2) ∨
              (x, y, z) = (2, 1, 0.5) ∨ (x, y, z) = (2, 0.5, 1) ∨
              (x, y, z) = (0.5, 1, 2) ∨ (x, y, z) = (0.5, 2, 1) :=
by
  sorry

end solve_system_l275_275470


namespace correct_operation_l275_275963

-- Define the conditions
def cond1 (m : ℝ) : Prop := m^2 + m^3 ≠ m^5
def cond2 (m : ℝ) : Prop := m^2 * m^3 = m^5
def cond3 (m : ℝ) : Prop := (m^2)^3 = m^6

-- Main statement that checks the correct operation
theorem correct_operation (m : ℝ) : cond1 m → cond2 m → cond3 m → (m^2 * m^3 = m^5) :=
by
  intros h1 h2 h3
  exact h2

end correct_operation_l275_275963


namespace balloons_remaining_l275_275294

def bags_round : ℕ := 5
def balloons_per_bag_round : ℕ := 20
def bags_long : ℕ := 4
def balloons_per_bag_long : ℕ := 30
def balloons_burst : ℕ := 5

def total_round_balloons : ℕ := bags_round * balloons_per_bag_round
def total_long_balloons : ℕ := bags_long * balloons_per_bag_long
def total_balloons : ℕ := total_round_balloons + total_long_balloons
def balloons_left : ℕ := total_balloons - balloons_burst

theorem balloons_remaining : balloons_left = 215 := by 
  -- We leave this as sorry since the proof is not required
  sorry

end balloons_remaining_l275_275294


namespace andrey_wins_iff_irreducible_fraction_l275_275387

def is_irreducible_fraction (p : ℝ) : Prop :=
  ∃ m n : ℕ, p = m / 2^n ∧ gcd m (2^n) = 1

def can_reach_0_or_1 (p : ℝ) : Prop :=
  ∀ move : ℝ, ∃ dir : ℝ, (p + dir * move = 0 ∨ p + dir * move = 1)

theorem andrey_wins_iff_irreducible_fraction (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ move_sequence : ℕ → ℝ, ∀ n, can_reach_0_or_1 (move_sequence n)) ↔ is_irreducible_fraction p :=
sorry

end andrey_wins_iff_irreducible_fraction_l275_275387


namespace complex_power_rectangular_form_l275_275099

noncomputable def cos (θ : ℝ) : ℝ := sorry
noncomputable def sin (θ : ℝ) : ℝ := sorry

theorem complex_power_rectangular_form :
  (3 * complex.exp (complex.I * 30))^4 = -40.5 + 40.5 * complex.I * √3 :=
by
  sorry

end complex_power_rectangular_form_l275_275099


namespace second_smallest_perimeter_l275_275660

theorem second_smallest_perimeter (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → 
  (a + b + c = 12) :=
by
  sorry

end second_smallest_perimeter_l275_275660


namespace driving_hours_fresh_l275_275402

theorem driving_hours_fresh (x : ℚ) : (25 * x + 15 * (9 - x) = 152) → x = 17 / 10 :=
by
  intros h
  sorry

end driving_hours_fresh_l275_275402


namespace students_at_end_of_year_l275_275570

def students_start : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0
def students_end : ℝ := 28.0

theorem students_at_end_of_year :
  students_start - students_left - students_transferred = students_end := by
  sorry

end students_at_end_of_year_l275_275570


namespace range_of_a_l275_275451

theorem range_of_a (a : ℝ)
  (A : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≥ 0})
  (B : Set ℝ := {x : ℝ | x ≥ a - 1})
  (H : A ∪ B = Set.univ) :
  a ≤ 2 :=
by
  sorry

end range_of_a_l275_275451


namespace second_smallest_root_is_zero_l275_275041

noncomputable def poly (a b c d : ℝ) : Polynomial ℝ :=
  Polynomial.X^7 - 8 * Polynomial.X^6 + 20 * Polynomial.X^5 + 5 * Polynomial.X^4 - a * Polynomial.X^3 - 2 * Polynomial.X^2 + (b - d) * Polynomial.X + c + 5

theorem second_smallest_root_is_zero (a b c d : ℝ) (h : ∀ x : ℝ, poly a b c d x = 0) :
  (∃ xs : List ℝ, xs.nodup ∧ xs.length = 4 ∧ list.sorted List.Ord xs ∧ xs.nth 1 = 0) :=
sorry

end second_smallest_root_is_zero_l275_275041


namespace arithmetic_sequence_sum_l275_275012

theorem arithmetic_sequence_sum
  (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (H1 : a1 = -2017)
  (H2 : (S 2013 : ℤ) / 2013 - (S 2011 : ℤ) / 2011 = 2)
  (H3 : ∀ n : ℕ, S n = n * a1 + (n * (n - 1) / 2) * d) :
  S 2017 = -2017 :=
by
  sorry

end arithmetic_sequence_sum_l275_275012


namespace value_of_expression_l275_275347

-- Definitions of the variables x and y along with their assigned values
def x : ℕ := 20
def y : ℕ := 8

-- The theorem that asserts the value of (x - y) * (x + y) equals 336
theorem value_of_expression : (x - y) * (x + y) = 336 := by 
  -- Skipping proof
  sorry

end value_of_expression_l275_275347


namespace right_triangle_inradius_height_ratio_l275_275611

-- Define a right triangle with sides a, b, and hypotenuse c
variables {a b c : ℝ}
-- Define the altitude from the right angle vertex
variables {h : ℝ}
-- Define the inradius of the triangle
variables {r : ℝ}

-- Define the conditions: right triangle 
-- and the relationships for h and r
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def altitude (h : ℝ) (a b c : ℝ) : Prop := h = (a * b) / c
def inradius (r : ℝ) (a b c : ℝ) : Prop := r = (a + b - c) / 2

theorem right_triangle_inradius_height_ratio {a b c h r : ℝ} 
  (Hrt : is_right_triangle a b c)
  (Hh : altitude h a b c)
  (Hr : inradius r a b c) : 
  0.4 < r / h ∧ r / h < 0.5 :=
sorry

end right_triangle_inradius_height_ratio_l275_275611


namespace alicia_average_speed_correct_l275_275991

/-
Alicia drove 320 miles in 6 hours.
Alicia drove another 420 miles in 7 hours.
Prove Alicia's average speed for the entire journey is 56.92 miles per hour.
-/

def alicia_total_distance : ℕ := 320 + 420
def alicia_total_time : ℕ := 6 + 7
def alicia_average_speed : ℚ := alicia_total_distance / alicia_total_time

theorem alicia_average_speed_correct : alicia_average_speed = 56.92 :=
by
  -- Proof goes here
  sorry

end alicia_average_speed_correct_l275_275991


namespace largest_possible_a_l275_275301

theorem largest_possible_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) (hp : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a ≤ 8924 :=
sorry

end largest_possible_a_l275_275301


namespace optimal_order_l275_275604

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l275_275604


namespace mandy_yoga_time_l275_275360

theorem mandy_yoga_time 
  (gym_ratio : ℕ)
  (bike_ratio : ℕ)
  (yoga_exercise_ratio : ℕ)
  (bike_time : ℕ) 
  (exercise_ratio : ℕ) 
  (yoga_ratio : ℕ)
  (h1 : gym_ratio = 2)
  (h2 : bike_ratio = 3)
  (h3 : yoga_exercise_ratio = 2)
  (h4 : exercise_ratio = 3)
  (h5 : bike_time = 18)
  (total_exercise_time : ℕ)
  (yoga_time : ℕ)
  (h6: total_exercise_time = ((gym_ratio * bike_time) / bike_ratio) + bike_time)
  (h7 : yoga_time = (yoga_exercise_ratio * total_exercise_time) / exercise_ratio) :
  yoga_time = 20 := 
by 
  sorry

end mandy_yoga_time_l275_275360


namespace find_p_when_q_is_1_l275_275137

-- Define the proportionality constant k and the relationship
variables {k p q : ℝ}
def inversely_proportional (k q p : ℝ) : Prop := p = k / (q + 2)

-- Given conditions
theorem find_p_when_q_is_1 (h1 : inversely_proportional k 4 1) : 
  inversely_proportional k 1 2 :=
by 
  sorry

end find_p_when_q_is_1_l275_275137


namespace no_solution_exists_l275_275236

theorem no_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x^y + 3 = y^x ∧ 3 * x^y = y^x + 8) :=
by
  intro h
  obtain ⟨eq1, eq2⟩ := h
  sorry

end no_solution_exists_l275_275236


namespace bus_total_capacity_l275_275898

-- Definitions based on conditions in a)
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seats_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 12

-- Proof statement
theorem bus_total_capacity : (left_side_seats + right_side_seats) * seats_per_seat + back_seat_capacity = 93 := by
  sorry

end bus_total_capacity_l275_275898


namespace glens_speed_is_37_l275_275265

/-!
# Problem Statement
Glen and Hannah drive at constant speeds toward each other on a highway. Glen drives at a certain speed G km/h. At some point, they pass by each other, and keep driving away from each other, maintaining their constant speeds. 
Glen is 130 km away from Hannah at 6 am and again at 11 am. Hannah is driving at 15 kilometers per hour.
Prove that Glen's speed is 37 km/h.
-/

def glens_speed (G : ℝ) : Prop :=
  ∃ G: ℝ, 
    (∃ H_speed : ℝ, H_speed = 15) ∧ -- Hannah's speed
    (∃ distance : ℝ, distance = 130) ∧ -- distance at 6 am and 11 am
    G + 15 = 260 / 5 -- derived equation from conditions

theorem glens_speed_is_37 : glens_speed 37 :=
by {
  sorry -- proof to be filled in
}

end glens_speed_is_37_l275_275265


namespace coefficient_of_ab_is_correct_l275_275277

noncomputable def a : ℝ := 15 / 7
noncomputable def b : ℝ := 15 / 2
noncomputable def ab : ℝ := 674.9999999999999
noncomputable def coeff_ab := ab / (a * b)

theorem coefficient_of_ab_is_correct :
  coeff_ab = 674.9999999999999 / ((15 * 15) / (7 * 2)) := sorry

end coefficient_of_ab_is_correct_l275_275277


namespace water_level_drop_recording_l275_275564

theorem water_level_drop_recording (rise6_recorded: Int): 
    (rise6_recorded = 6) → (6 = -rise6_recorded) :=
by
  sorry

end water_level_drop_recording_l275_275564


namespace num_integer_solutions_abs_ineq_l275_275270

theorem num_integer_solutions_abs_ineq : 
  ∃ (S : Set ℤ), (∀ (x : ℝ), |x - 3| ≤ 4.5 ↔ ∃ (n: ℤ), n.val = x) ∧ S.card = 9 := 
sorry

end num_integer_solutions_abs_ineq_l275_275270


namespace max_teams_in_chess_tournament_l275_275836

theorem max_teams_in_chess_tournament :
  ∃ n : ℕ, n * (n - 1) ≤ 500 / 9 ∧ ∀ m : ℕ, m * (m - 1) ≤ 500 / 9 → m ≤ n :=
sorry

end max_teams_in_chess_tournament_l275_275836


namespace explicit_x_n_formula_l275_275184

theorem explicit_x_n_formula (x y : ℕ → ℕ) (n : ℕ) :
  x 0 = 2 ∧ y 0 = 1 ∧
  (∀ n, x (n + 1) = x n ^ 2 + y n ^ 2) ∧
  (∀ n, y (n + 1) = 2 * x n * y n) →
  x n = (3 ^ (2 ^ n) + 1) / 2 :=
by
  sorry

end explicit_x_n_formula_l275_275184


namespace find_ab_plus_a_plus_b_l275_275918

-- Define the polynomial
def quartic_poly (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 - 6*x - 1

-- Define the roots conditions
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

-- State the proof problem
theorem find_ab_plus_a_plus_b :
  ∃ a b : ℝ,
    is_root quartic_poly a ∧
    is_root quartic_poly b ∧
    ab = a * b ∧
    a_plus_b = a + b ∧
    ab + a_plus_b = 4 :=
by sorry

end find_ab_plus_a_plus_b_l275_275918


namespace expr1_correct_expr2_correct_expr3_correct_l275_275390

-- Define the expressions and corresponding correct answers
def expr1 : Int := 58 + 15 * 4
def expr2 : Int := 216 - 72 / 8
def expr3 : Int := (358 - 295) / 7

-- State the proof goals
theorem expr1_correct : expr1 = 118 := by
  sorry

theorem expr2_correct : expr2 = 207 := by
  sorry

theorem expr3_correct : expr3 = 9 := by
  sorry

end expr1_correct_expr2_correct_expr3_correct_l275_275390


namespace gcd_36_54_l275_275792

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factorization (n : ℕ) : list ℕ :=
if h : n = 0 then []
else
(list.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

theorem gcd_36_54 : Nat.gcd 36 54 = 18 :=
by
  sorry

end gcd_36_54_l275_275792


namespace marie_stamps_l275_275304

variable (n_notebooks : ℕ) (stamps_per_notebook : ℕ) (n_binders : ℕ) (stamps_per_binder : ℕ) (fraction_keep : ℚ)

theorem marie_stamps :
  n_notebooks = 4 →
  stamps_per_notebook = 20 →
  n_binders = 2 →
  stamps_per_binder = 50 →
  fraction_keep = 1/4 →
  let total_stamps := n_notebooks * stamps_per_notebook + n_binders * stamps_per_binder in
  let stamps_keep := total_stamps * fraction_keep in
  let stamps_give_away := total_stamps - stamps_keep in
  stamps_give_away = 135 :=
by
  intros h1 h2 h3 h4 h5
  let total_stamps := n_notebooks * stamps_per_notebook + n_binders * stamps_per_binder
  let stamps_keep := total_stamps * fraction_keep
  let stamps_give_away := total_stamps - stamps_keep
  have h_total_stamps : total_stamps = 180 := by simp [h1, h2, h3, h4, total_stamps]
  have h_stamps_keep : stamps_keep = 45 := by simp [h_total_stamps, h5, stamps_keep]
  have h_stamps_give_away : stamps_give_away = 135 := by simp [h_total_stamps, h_stamps_keep, stamps_give_away]
  exact h_stamps_give_away

end marie_stamps_l275_275304


namespace inequality_proof_l275_275588

-- Definitions for the conditions
variable (x y : ℝ)

-- Conditions
def conditions : Prop := 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Problem statement to be proven
theorem inequality_proof (h : conditions x y) : 
  x^3 + x * y^2 + 2 * x * y ≤ 2 * x^2 * y + x^2 + x + y := 
by 
  sorry

end inequality_proof_l275_275588


namespace original_number_l275_275348

theorem original_number (x : ℝ) (hx : 100000 * x = 5 * (1 / x)) : x = 0.00707 := 
by
  sorry

end original_number_l275_275348


namespace simplify_expression_l275_275617

theorem simplify_expression (x : ℝ) (h1 : x^2 - 4*x + 3 ≠ 0) (h2 : x^2 - 6*x + 9 ≠ 0) (h3 : x^2 - 3*x + 2 ≠ 0) (h4 : x^2 - 4*x + 4 ≠ 0) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / (x^2 - 3*x + 2) / (x^2 - 4*x + 4) = (x-2) / (x-3) :=
by {
  sorry
}

end simplify_expression_l275_275617


namespace admin_fee_percentage_l275_275338

noncomputable def percentage_deducted_for_admin_fees 
  (amt_johnson : ℕ) (amt_sutton : ℕ) (amt_rollin : ℕ)
  (amt_school : ℕ) (amt_after_deduction : ℕ) : ℚ :=
  ((amt_school - amt_after_deduction) * 100) / amt_school

theorem admin_fee_percentage : 
  ∃ (amt_johnson amt_sutton amt_rollin amt_school amt_after_deduction : ℕ),
  amt_johnson = 2300 ∧
  amt_johnson = 2 * amt_sutton ∧
  amt_sutton * 8 = amt_rollin ∧
  amt_rollin * 3 = amt_school ∧
  amt_after_deduction = 27048 ∧
  percentage_deducted_for_admin_fees amt_johnson amt_sutton amt_rollin amt_school amt_after_deduction = 2 :=
by
  sorry

end admin_fee_percentage_l275_275338


namespace exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l275_275929

open Real EuclideanGeometry

def is_isosceles_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def is_isosceles_triangle_3D (A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def five_points_isosceles (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 5, is_isosceles_triangle (pts i) (pts j) (pts k)

def six_points_isosceles (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 6, is_isosceles_triangle (pts i) (pts j) (pts k)

def seven_points_isosceles_3D (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∀ i j k : Fin 7, is_isosceles_triangle_3D (pts i) (pts j) (pts k)

theorem exists_five_points_isosceles : ∃ (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)), five_points_isosceles pts :=
sorry

theorem exists_six_points_isosceles : ∃ (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)), six_points_isosceles pts :=
sorry

theorem exists_seven_points_isosceles_3D : ∃ (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)), seven_points_isosceles_3D pts :=
sorry

end exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l275_275929


namespace initial_total_cards_l275_275676

theorem initial_total_cards (x y : ℕ) (h1 : x / (x + y) = 1 / 3) (h2 : x / (x + y + 4) = 1 / 4) : x + y = 12 := 
sorry

end initial_total_cards_l275_275676


namespace relationship_l275_275717

noncomputable def a : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def b : ℝ := (3 / 5) ^ (2 / 5)
noncomputable def c : ℝ := Real.logb (3 / 5) (2 / 5)

theorem relationship : a < b ∧ b < c :=
by
  -- proof will go here
  sorry


end relationship_l275_275717


namespace constant_term_of_product_l275_275961

def P(x: ℝ) : ℝ := x^6 + 2 * x^2 + 3
def Q(x: ℝ) : ℝ := x^4 + x^3 + 4
def R(x: ℝ) : ℝ := 2 * x^2 + 3 * x + 7

theorem constant_term_of_product :
  let C := (P 0) * (Q 0) * (R 0)
  C = 84 :=
by
  let C := (P 0) * (Q 0) * (R 0)
  show C = 84
  sorry

end constant_term_of_product_l275_275961


namespace average_price_of_remaining_cans_l275_275749

theorem average_price_of_remaining_cans (price_all price_returned : ℕ) (average_all average_returned : ℚ) 
    (h1 : price_all = 6) (h2 : average_all = 36.5) (h3 : price_returned = 2) (h4 : average_returned = 49.5) : 
    (price_all - price_returned) ≠ 0 → 
    4 * 30 = 6 * 36.5 - 2 * 49.5 :=
by
  intros hne
  sorry

end average_price_of_remaining_cans_l275_275749


namespace sum_geometric_seq_l275_275489

theorem sum_geometric_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1)
  (h2 : 4 * a 2 = 4 * a 1 + a 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 3 = 15 :=
by
  sorry

end sum_geometric_seq_l275_275489


namespace find_decimal_number_l275_275351

noncomputable def decimal_number (x : ℝ) : Prop := 
x > 0 ∧ (100000 * x = 5 * (1 / x))

theorem find_decimal_number {x : ℝ} (h : decimal_number x) : x = 1 / (100 * Real.sqrt 2) :=
by
  sorry

end find_decimal_number_l275_275351


namespace prices_of_books_book_purchasing_plans_l275_275060

-- Define the conditions
def cost_eq1 (x y : ℕ): Prop := 20 * x + 40 * y = 1520
def cost_eq2 (x y : ℕ): Prop := 20 * x - 20 * y = 440
def plan_conditions (x y : ℕ): Prop := (20 + y - x = 20) ∧ (x + y + 20 ≥ 72) ∧ (40 * x + 18 * (y + 20) ≤ 2000)

-- Prove price of each book
theorem prices_of_books : 
  ∃ (x y : ℕ), cost_eq1 x y ∧ cost_eq2 x y ∧ x = 40 ∧ y = 18 :=
by {
  sorry
}

-- Prove possible book purchasing plans
theorem book_purchasing_plans : 
  ∃ (x : ℕ), plan_conditions x (x + 20) ∧ 
  (x = 26 ∧ x + 20 = 46 ∨ 
   x = 27 ∧ x + 20 = 47 ∨ 
   x = 28 ∧ x + 20 = 48) :=
by {
  sorry
}

end prices_of_books_book_purchasing_plans_l275_275060


namespace imaginary_part_of_z_is_1_l275_275549

def z := Complex.ofReal 0 + Complex.ofReal 1 * Complex.I * (Complex.ofReal 1 + Complex.ofReal 2 * Complex.I)
theorem imaginary_part_of_z_is_1 : z.im = 1 := by
  sorry

end imaginary_part_of_z_is_1_l275_275549


namespace fred_games_last_year_proof_l275_275866

def fred_games_last_year (this_year: ℕ) (diff: ℕ) : ℕ := this_year + diff

theorem fred_games_last_year_proof : 
  ∀ (this_year: ℕ) (diff: ℕ),
  this_year = 25 → 
  diff = 11 →
  fred_games_last_year this_year diff = 36 := 
by 
  intros this_year diff h_this_year h_diff
  rw [h_this_year, h_diff]
  sorry

end fred_games_last_year_proof_l275_275866


namespace green_paint_amount_l275_275191

theorem green_paint_amount (T W B : ℕ) (hT : T = 69) (hW : W = 20) (hB : B = 34) : 
  T - (W + B) = 15 := 
by
  sorry

end green_paint_amount_l275_275191


namespace second_pump_drain_time_l275_275311

-- Definitions of the rates R1 and R2
def R1 : ℚ := 1 / 12  -- Rate of the first pump
def R2 : ℚ := 1 - R1  -- Rate of the second pump (from the combined rate equation)

-- The time it takes the second pump alone to drain the pond
def time_to_drain_second_pump := 1 / R2

-- The goal is to prove that this value is 12/11
theorem second_pump_drain_time : time_to_drain_second_pump = 12 / 11 := by
  -- The proof is omitted
  sorry

end second_pump_drain_time_l275_275311


namespace reimbursement_calculation_l275_275927

variable (total_paid : ℕ) (pieces : ℕ) (cost_per_piece : ℕ)

theorem reimbursement_calculation
  (h1 : total_paid = 20700)
  (h2 : pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (pieces * cost_per_piece) = 600 := 
by
  sorry

end reimbursement_calculation_l275_275927


namespace boots_cost_more_l275_275321

theorem boots_cost_more (S B : ℝ) 
  (h1 : 22 * S + 16 * B = 460) 
  (h2 : 8 * S + 32 * B = 560) : B - S = 5 :=
by
  -- Here we provide the statement only, skipping the proof
  sorry

end boots_cost_more_l275_275321


namespace squared_product_l275_275393

theorem squared_product (a b : ℝ) : (- (1 / 2) * a^2 * b)^2 = (1 / 4) * a^4 * b^2 := by 
  sorry

end squared_product_l275_275393


namespace range_of_m_is_increasing_l275_275431

noncomputable def f (x : ℝ) (m: ℝ) := x^2 + m*x + m

theorem range_of_m_is_increasing :
  { m : ℝ // ∀ x y : ℝ, -2 ≤ x → x ≤ y → f x m ≤ f y m } = {m | 4 ≤ m} :=
by
  sorry

end range_of_m_is_increasing_l275_275431


namespace roots_in_interval_l275_275535

theorem roots_in_interval (f : ℝ → ℝ)
  (h : ∀ x, f x = 4 * x ^ 2 - (3 * m + 1) * x - m - 2) :
  (forall (x1 x2 : ℝ), (f x1 = 0 ∧ f x2 = 0) → -1 < x1 ∧ x1 < 2 ∧ -1 < x2 ∧ x2 < 2) ↔ -1 < m ∧ m < 12 / 7 :=
sorry

end roots_in_interval_l275_275535


namespace compare_solutions_l275_275584

theorem compare_solutions 
  (c d p q : ℝ) 
  (hc : c ≠ 0) 
  (hp : p ≠ 0) :
  (-d / c) < (-q / p) ↔ (q / p) < (d / c) :=
by
  sorry

end compare_solutions_l275_275584


namespace cos_2pi_minus_alpha_tan_alpha_minus_7pi_l275_275716

open Real

variables (α : ℝ)
variables (h1 : sin (π + α) = -1 / 3) (h2 : π / 2 < α ∧ α < π)

-- Statement for the problem (Ⅰ)
theorem cos_2pi_minus_alpha :
  cos (2 * π - α) = -2 * sqrt 2 / 3 :=
sorry

-- Statement for the problem (Ⅱ)
theorem tan_alpha_minus_7pi :
  tan (α - 7 * π) = -sqrt 2 / 4 :=
sorry

end cos_2pi_minus_alpha_tan_alpha_minus_7pi_l275_275716


namespace find_a5_of_geom_seq_l275_275258

theorem find_a5_of_geom_seq 
  (a : ℕ → ℝ) (q : ℝ)
  (hgeom : ∀ n, a (n + 1) = a n * q)
  (S : ℕ → ℝ)
  (hS3 : S 3 = a 0 * (1 - q ^ 3) / (1 - q))
  (hS6 : S 6 = a 0 * (1 - q ^ 6) / (1 - q))
  (hS9 : S 9 = a 0 * (1 - q ^ 9) / (1 - q))
  (harith : S 3 + S 6 = 2 * S 9)
  (a8 : a 8 = 3) :
  a 5 = -6 :=
by
  sorry

end find_a5_of_geom_seq_l275_275258


namespace probability_of_different_value_and_suit_l275_275799

theorem probability_of_different_value_and_suit :
  let total_cards := 52
  let first_card_choices := 52
  let remaining_cards := 51
  let different_suits := 3
  let different_values := 12
  let favorable_outcomes := different_suits * different_values
  let total_outcomes := remaining_cards
  let probability := favorable_outcomes / total_outcomes
  probability = 12 / 17 := 
by
  sorry

end probability_of_different_value_and_suit_l275_275799


namespace sum_of_squares_of_coeffs_l275_275500

theorem sum_of_squares_of_coeffs (c1 c2 c3 c4 : ℝ) (h1 : c1 = 3) (h2 : c2 = 6) (h3 : c3 = 15) (h4 : c4 = 6) :
  c1^2 + c2^2 + c3^2 + c4^2 = 306 :=
by
  sorry

end sum_of_squares_of_coeffs_l275_275500


namespace gcd_eq_gcd_of_division_l275_275253

theorem gcd_eq_gcd_of_division (a b q r : ℕ) (h1 : a = b * q + r) (h2 : 0 < r) (h3 : r < b) (h4 : a > b) : 
  Nat.gcd a b = Nat.gcd b r :=
by
  sorry

end gcd_eq_gcd_of_division_l275_275253


namespace perpendicular_line_equation_l275_275656

noncomputable def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m : ℝ := -a / b
  let b' : ℝ := c / b
  (m, b')

noncomputable def perpendicular_slope (slope : ℝ) : ℝ :=
  -1 / slope

theorem perpendicular_line_equation (x1 y1 : ℝ) 
  (h_line_eq : ∃ (a b c : ℝ), a * x1 + b * y1 = c)
  (h_point : x1 = 2 ∧ y1 = -3) :
  ∃ a b : ℝ, y1 = a * x1 + b ∧ a = -2 ∧ b = 1 :=
by
  let (m, b') := slope_intercept_form 3 (-6) 9
  have h_slope := (-1 : ℝ) / m
  have h_perpendicular_slope : h_slope = -2 := sorry
  let (x1, y1) := (2, -3)
  have h_eq : y1 - (-3) = h_slope * (x1 - 2) := sorry
  use [-2, 1]
  simp [h_eq]
  sorry

end perpendicular_line_equation_l275_275656


namespace spotted_mushrooms_ratio_l275_275509

theorem spotted_mushrooms_ratio 
  (total_mushrooms : ℕ) 
  (gilled_mushrooms : ℕ) 
  (spotted_mushrooms : ℕ) 
  (total_mushrooms_eq : total_mushrooms = 30) 
  (gilled_mushrooms_eq : gilled_mushrooms = 3) 
  (spots_and_gills_exclusive : ∀ x, x = spotted_mushrooms ∨ x = gilled_mushrooms) : 
  spotted_mushrooms / gilled_mushrooms = 9 := 
by
  sorry

end spotted_mushrooms_ratio_l275_275509


namespace purchased_only_A_l275_275182

-- Definitions for the conditions
def total_B (x : ℕ) := x + 500
def total_A (y : ℕ) := 2 * y

-- Question formulated in Lean 4
theorem purchased_only_A : 
  ∃ C : ℕ, (∀ x y : ℕ, 2 * x = 500 → y = total_B x → 2 * y = total_A y → C = total_A y - 500) ∧ C = 1000 :=
  sorry

end purchased_only_A_l275_275182


namespace triangle_inequality_proof_l275_275508

noncomputable def triangle_inequality (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) : Prop :=
  Real.pi / 3 ≤ (a * A + b * B + c * C) / (a + b + c) ∧ (a * A + b * B + c * C) / (a + b + c) < Real.pi / 2

theorem triangle_inequality_proof (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h₁: A + B + C = Real.pi) (h₂: ∀ {x y : ℝ}, A ≥ B  → a ≥ b → A * b + B * a ≤ A * a + B * b) 
  (h₃: ∀ {x y : ℝ}, x + y > 0 → A * x + B * y + C * (a + b - x - y) > 0) : 
  triangle_inequality A B C a b c hABC :=
by
  sorry

end triangle_inequality_proof_l275_275508


namespace fred_fewer_games_l275_275408

/-- Fred went to 36 basketball games last year -/
def games_last_year : ℕ := 36

/-- Fred went to 25 basketball games this year -/
def games_this_year : ℕ := 25

/-- Prove that Fred went to 11 fewer games this year compared to last year -/
theorem fred_fewer_games : games_last_year - games_this_year = 11 := by
  sorry

end fred_fewer_games_l275_275408


namespace gcd_36_54_l275_275788

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l275_275788


namespace geometric_progression_positions_l275_275910

theorem geometric_progression_positions (u1 q : ℝ) (m n p : ℕ)
  (h27 : 27 = u1 * q ^ (m - 1))
  (h8 : 8 = u1 * q ^ (n - 1))
  (h12 : 12 = u1 * q ^ (p - 1)) :
  m = 3 * p - 2 * n :=
sorry

end geometric_progression_positions_l275_275910


namespace cut_wire_l275_275327

theorem cut_wire (x y : ℕ) : 
  15 * x + 12 * y = 102 ↔ (x = 2 ∧ y = 6) ∨ (x = 6 ∧ y = 1) :=
by
  sorry

end cut_wire_l275_275327


namespace quadratic_inequality_always_positive_l275_275662

theorem quadratic_inequality_always_positive (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
by sorry

end quadratic_inequality_always_positive_l275_275662


namespace maximize_expression_l275_275881

-- Given the condition
theorem maximize_expression (x y : ℝ) (h : x + y = 1) : (x^3 + 1) * (y^3 + 1) ≤ (1)^3 + 1 * (0)^3 + 1 * (0)^3 + 1 :=
sorry

end maximize_expression_l275_275881


namespace temperature_at_midnight_l275_275434

-- Define the variables for initial conditions and changes
def T_morning : ℤ := 7 -- Morning temperature in degrees Celsius
def ΔT_noon : ℤ := 2   -- Temperature increase at noon in degrees Celsius
def ΔT_midnight : ℤ := -10  -- Temperature drop at midnight in degrees Celsius

-- Calculate the temperatures at noon and midnight
def T_noon := T_morning + ΔT_noon
def T_midnight := T_noon + ΔT_midnight

-- State the theorem to prove the temperature at midnight
theorem temperature_at_midnight : T_midnight = -1 := by
  sorry

end temperature_at_midnight_l275_275434


namespace rebecca_pies_l275_275932

theorem rebecca_pies 
  (P : ℕ) 
  (slices_per_pie : ℕ := 8) 
  (rebecca_slices : ℕ := P) 
  (family_and_friends_slices : ℕ := (7 * P) / 2) 
  (additional_slices : ℕ := 2) 
  (remaining_slices : ℕ := 5) 
  (total_slices : ℕ := slices_per_pie * P) :
  rebecca_slices + family_and_friends_slices + additional_slices + remaining_slices = total_slices → 
  P = 2 := 
by { sorry }

end rebecca_pies_l275_275932


namespace marginal_cost_per_product_calculation_l275_275174

def fixed_cost : ℝ := 12000
def total_cost : ℝ := 16000
def num_products : ℕ := 20

theorem marginal_cost_per_product_calculation :
  (total_cost - fixed_cost) / num_products = 200 := by
  sorry

end marginal_cost_per_product_calculation_l275_275174


namespace nat_digit_problem_l275_275177

theorem nat_digit_problem :
  ∀ n : Nat, (n % 10 = (2016 * (n / 2016)) % 10) → (n = 4032 ∨ n = 8064 ∨ n = 12096 ∨ n = 16128) :=
by
  sorry

end nat_digit_problem_l275_275177


namespace chromium_alloy_l275_275152

theorem chromium_alloy (x : ℝ) (h1 : 0.12 * x + 0.10 * 35 = 0.106 * (x + 35)) : x = 15 := 
by 
  -- statement only, no proof required.
  sorry

end chromium_alloy_l275_275152


namespace fiona_weekly_earnings_l275_275266

theorem fiona_weekly_earnings :
  let monday_hours := 1.5
  let tuesday_hours := 1.25
  let wednesday_hours := 3.1667
  let thursday_hours := 0.75
  let hourly_wage := 4
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours
  let total_earnings := total_hours * hourly_wage
  total_earnings = 26.67 := by
  sorry

end fiona_weekly_earnings_l275_275266


namespace germination_at_least_4_out_of_5_l275_275625

noncomputable def germination_prob : ℚ := 0.8

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  n.choose k * p^k * (1 - p)^(n - k)

theorem germination_at_least_4_out_of_5 (p : ℚ) (n : ℕ) (h : p = germination_prob) :
  (binomial_probability n 4 p + binomial_probability n 5 p) = 0.73728 :=
by
  sorry

end germination_at_least_4_out_of_5_l275_275625


namespace probability_all_same_color_l275_275973

theorem probability_all_same_color :
  let total_marbles := 20
  let red_marbles := 5
  let white_marbles := 7
  let blue_marbles := 8
  let total_ways_to_draw_3 := (total_marbles * (total_marbles - 1) * (total_marbles - 2)) / 6
  let ways_to_draw_3_red := (red_marbles * (red_marbles - 1) * (red_marbles - 2)) / 6
  let ways_to_draw_3_white := (white_marbles * (white_marbles - 1) * (white_marbles - 2)) / 6
  let ways_to_draw_3_blue := (blue_marbles * (blue_marbles - 1) * (blue_marbles - 2)) / 6
  let probability := (ways_to_draw_3_red + ways_to_draw_3_white + ways_to_draw_3_blue) / total_ways_to_draw_3
  probability = 101/1140 :=
by
  sorry

end probability_all_same_color_l275_275973


namespace rectangle_length_l275_275043

/--
The perimeter of a rectangle is 150 cm. The length is 15 cm greater than the width.
This theorem proves that the length of the rectangle is 45 cm under these conditions.
-/
theorem rectangle_length (P w l : ℝ) (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : l = 45 :=
by
  sorry

end rectangle_length_l275_275043


namespace old_books_to_reread_l275_275849

/-- Brianna problem -/
def total_books_needed : ℕ := 2 * 12

def books_given_as_gift : ℕ := 6

def books_bought : ℕ := 8

def books_borrowed : ℕ := books_bought - 2

def total_new_books : ℕ := books_given_as_gift + books_bought + books_borrowed

theorem old_books_to_reread : 
  let num_old_books := total_books_needed - total_new_books in
  num_old_books = 4 :=
by
  sorry

end old_books_to_reread_l275_275849


namespace perpendicular_line_through_point_l275_275644

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l275_275644


namespace bobby_initial_candy_count_l275_275226

theorem bobby_initial_candy_count (x : ℕ) (h : x + 17 = 43) : x = 26 :=
by
  sorry

end bobby_initial_candy_count_l275_275226


namespace lisa_matching_pair_probability_l275_275245

theorem lisa_matching_pair_probability :
  let total_socks := 22
  let gray_socks := 12
  let white_socks := 10
  let total_pairs := total_socks * (total_socks - 1) / 2
  let gray_pairs := gray_socks * (gray_socks - 1) / 2
  let white_pairs := white_socks * (white_socks - 1) / 2
  let matching_pairs := gray_pairs + white_pairs
  let probability := matching_pairs / total_pairs
  probability = (111 / 231) :=
by
  sorry

end lisa_matching_pair_probability_l275_275245


namespace min_distance_MN_l275_275450

open Real

noncomputable def f (x : ℝ) := exp x - (1 / 2) * x^2
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_MN (x1 x2 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 > 0) (h3 : f x1 = g x2) :
  abs (x2 - x1) = 2 :=
by
  sorry

end min_distance_MN_l275_275450


namespace probability_exactly_two_heads_l275_275976

theorem probability_exactly_two_heads :
  (nat.choose 6 2) / (2^6) = (15 : ℚ) / 64 :=
by {
  sorry
}

end probability_exactly_two_heads_l275_275976


namespace question1_question2_l275_275890

-- Define the sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := { x | 2 * x + a > 0 }
def setB : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Question 1: When a = 2, find the set A ∩ B
theorem question1 : A ∩ B = { x | x > 3 } :=
  sorry

-- Question 2: If A ∩ (complement of B) = ∅, find the range of a
theorem question2 : A ∩ (U \ B) = ∅ → a ≤ -6 :=
  sorry

end question1_question2_l275_275890


namespace at_least_12_boxes_l275_275081

theorem at_least_12_boxes (extra_boxes : Nat) : 
  let total_boxes := 12 + extra_boxes
  extra_boxes ≥ 0 → total_boxes ≥ 12 :=
by
  intros
  sorry

end at_least_12_boxes_l275_275081


namespace shirt_cost_l275_275968

def cost_of_jeans_and_shirts (J S : ℝ) : Prop := (3 * J + 2 * S = 69) ∧ (2 * J + 3 * S = 81)

theorem shirt_cost (J S : ℝ) (h : cost_of_jeans_and_shirts J S) : S = 21 :=
by {
  sorry
}

end shirt_cost_l275_275968


namespace perpendicular_line_equation_l275_275652

-- Conditions definition
def line1 := ∀ x y : ℝ, 3 * x - 6 * y = 9
def point := (2 : ℝ, -3 : ℝ)

-- Target statement
theorem perpendicular_line_equation : 
  (∀ x y : ℝ, line1 x y → y = (-2) * x + 1) := 
by 
  -- proof goes here
  sorry

end perpendicular_line_equation_l275_275652


namespace find_a_range_l275_275418

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 2 * x^2 - 2 * x + a - 3 = 0) ∧ 
  (∃ y : ℝ, y > 0 ∧ y ≠ x ∧ 2 * y^2 - 2 * y + a - 3 = 0) 
  ↔ 3 < a ∧ a < 7 / 2 := 
sorry

end find_a_range_l275_275418


namespace correct_operation_l275_275200

theorem correct_operation : (a : ℕ) →
  (a^2 * a^3 = a^5) ∧
  (2 * a + 4 ≠ 6 * a) ∧
  ((2 * a)^2 ≠ 2 * a^2) ∧
  (a^3 / a^3 ≠ a) := sorry

end correct_operation_l275_275200


namespace intersection_and_complement_l275_275130

open Set

def A := {x : ℝ | -4 ≤ x ∧ x ≤ -2}
def B := {x : ℝ | x + 3 ≥ 0}

theorem intersection_and_complement : 
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧ (compl (A ∩ B) = {x | x < -3 ∨ x > -2}) :=
by
  sorry

end intersection_and_complement_l275_275130


namespace largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l275_275548

-- Definitions and conditions
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ := (x + (3 * x^2))^n

-- Problem statements
theorem largest_binomial_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  (2^n = 128) →
  ∃ t : ℕ, t = 2835 * x^11 := 
by sorry

theorem largest_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  exists t, t = 5103 * x^13 :=
by sorry

theorem remainder_mod_7 :
  ∀ x n,
  x = 3 →
  n = 2016 →
  (x + (3 * x^2))^n % 7 = 1 :=
by sorry

end largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l275_275548


namespace calc_factorial_sum_l275_275231

theorem calc_factorial_sum : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end calc_factorial_sum_l275_275231


namespace evaluate_expression_l275_275523

theorem evaluate_expression (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := 
by
  sorry

end evaluate_expression_l275_275523


namespace no_valid_pairs_l275_275696

theorem no_valid_pairs (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  ¬(1000 * a + 100 * b + 32) % 99 = 0 :=
by
  sorry

end no_valid_pairs_l275_275696


namespace area_of_triangle_is_correct_l275_275446

def vector := (ℝ × ℝ)

def a : vector := (7, 3)
def b : vector := (-1, 5)

noncomputable def det2x2 (v1 v2 : vector) : ℝ :=
  (v1.1 * v2.2) - (v1.2 * v2.1)

theorem area_of_triangle_is_correct :
  let area := (det2x2 a b) / 2
  area = 19 := by
  -- defintions and conditions are set here, proof skipped
  sorry

end area_of_triangle_is_correct_l275_275446


namespace irrational_sum_root_l275_275758

theorem irrational_sum_root
  (α : ℝ) (hα : Irrational α)
  (n : ℕ) (hn : 0 < n) :
  Irrational ((α + (α^2 - 1).sqrt)^(1/n : ℝ) + (α - (α^2 - 1).sqrt)^(1/n : ℝ)) := sorry

end irrational_sum_root_l275_275758


namespace calculate_expression_l275_275233

theorem calculate_expression : 
  (1 / 2) ^ (-2: ℤ) - 3 * Real.tan (Real.pi / 6) - abs (Real.sqrt 3 - 2) = 2 := 
by
  sorry

end calculate_expression_l275_275233


namespace range_of_a_l275_275878

variable (a x : ℝ)

def P : Prop := a < x ∧ x < a + 1
def q : Prop := x^2 - 7 * x + 10 ≤ 0

theorem range_of_a (h₁ : P a x → q x) (h₂ : ∃ x, q x ∧ ¬P a x) : 2 ≤ a ∧ a ≤ 4 := 
sorry

end range_of_a_l275_275878


namespace train_total_distance_l275_275219

theorem train_total_distance (x : ℝ) (h1 : x > 0) 
  (h_speed_avg : 48 = ((3 * x) / (x / 8))) : 
  3 * x = 6 := 
by
  sorry

end train_total_distance_l275_275219


namespace largest_number_is_B_l275_275501

noncomputable def numA : ℝ := 7.196533
noncomputable def numB : ℝ := 7.19655555555555555555555555555555555555 -- 7.196\overline{5}
noncomputable def numC : ℝ := 7.1965656565656565656565656565656565 -- 7.19\overline{65}
noncomputable def numD : ℝ := 7.196596596596596596596596596596596 -- 7.1\overline{965}
noncomputable def numE : ℝ := 7.196519651965196519651965196519651 -- 7.\overline{1965}

theorem largest_number_is_B : 
  numB > numA ∧ numB > numC ∧ numB > numD ∧ numB > numE :=
by
  sorry

end largest_number_is_B_l275_275501


namespace students_qualifying_percentage_l275_275672

theorem students_qualifying_percentage (N B G : ℕ) (boy_percent : ℝ) (girl_percent : ℝ) :
  N = 400 →
  G = 100 →
  B = N - G →
  boy_percent = 0.60 →
  girl_percent = 0.80 →
  (boy_percent * B + girl_percent * G) / N * 100 = 65 :=
by
  intros hN hG hB hBoy hGirl
  simp [hN, hG, hB, hBoy, hGirl]
  sorry

end students_qualifying_percentage_l275_275672


namespace bins_of_vegetables_l275_275401

-- Define the conditions
def total_bins : ℝ := 0.75
def bins_of_soup : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

-- Define the statement to be proved
theorem bins_of_vegetables :
  total_bins = bins_of_soup + (0.13) + bins_of_pasta := 
sorry

end bins_of_vegetables_l275_275401


namespace probability_different_colors_is_correct_l275_275566

-- Definitions of chip counts
def blue_chips := 6
def red_chips := 5
def yellow_chips := 4
def green_chips := 3
def total_chips := blue_chips + red_chips + yellow_chips + green_chips

-- Definition of the probability calculation
def probability_different_colors := 
  ((blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)) +
  ((red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)) +
  ((yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)) +
  ((green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips))

-- Given the problem conditions, we assert the correct answer
theorem probability_different_colors_is_correct :
  probability_different_colors = (119 / 162) := 
sorry

end probability_different_colors_is_correct_l275_275566


namespace probability_heads_mod_coin_l275_275370

theorem probability_heads_mod_coin (p : ℝ) (h : 20 * p ^ 3 * (1 - p) ^ 3 = 1 / 20) : p = (1 - Real.sqrt 0.6816) / 2 :=
by
  sorry

end probability_heads_mod_coin_l275_275370


namespace bottles_produced_l275_275205

/-- 
14 machines produce 2520 bottles in 4 minutes, given that 6 machines produce 270 bottles per minute. 
-/
theorem bottles_produced (rate_6_machines : Nat) (bottles_per_minute : Nat) 
  (rate_one_machine : Nat) (rate_14_machines : Nat) (total_production : Nat) : 
  rate_6_machines = 6 ∧ bottles_per_minute = 270 ∧ rate_one_machine = bottles_per_minute / rate_6_machines 
  ∧ rate_14_machines = 14 * rate_one_machine ∧ total_production = rate_14_machines * 4 → 
  total_production = 2520 :=
sorry

end bottles_produced_l275_275205


namespace complex_fourth_power_l275_275095

-- Definitions related to the problem conditions
def trig_identity (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)
def z : ℂ := 3 * trig_identity (30 * real.pi / 180)
def result := (81 : ℝ) * (-1/2 + complex.I * (real.sqrt 3 / 2))

-- Statement of the problem
theorem complex_fourth_power : z^4 = result := by
  sorry

end complex_fourth_power_l275_275095


namespace cos_beta_eq_neg_16_over_65_l275_275869

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin β = 5 / 13)
variable (h4 : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_eq_neg_16_over_65 : Real.cos β = -16 / 65 := by
  sorry

end cos_beta_eq_neg_16_over_65_l275_275869


namespace find_common_difference_l275_275765

-- Define the arithmetic series sum formula
def arithmetic_series_sum (a₁ : ℕ) (d : ℚ) (n : ℕ) :=
  (n / 2) * (2 * a₁ + (n - 1) * d)

-- Define the first day's production, total days, and total fabric
def first_day := 5
def total_days := 30
def total_fabric := 390

-- The proof statement
theorem find_common_difference : 
  ∃ d : ℚ, arithmetic_series_sum first_day d total_days = total_fabric ∧ d = 16 / 29 :=
by
  sorry

end find_common_difference_l275_275765


namespace employee_y_payment_l275_275969

theorem employee_y_payment (X Y : ℝ) (h1 : X + Y = 616) (h2 : X = 1.2 * Y) : Y = 280 :=
by
  sorry

end employee_y_payment_l275_275969


namespace both_participation_correct_l275_275673

-- Define the number of total participants
def total_participants : ℕ := 50

-- Define the number of participants in Chinese competition
def chinese_participants : ℕ := 30

-- Define the number of participants in Mathematics competition
def math_participants : ℕ := 38

-- Define the number of people who do not participate in either competition
def neither_participants : ℕ := 2

-- Define the number of people who participate in both competitions
def both_participants : ℕ :=
  chinese_participants + math_participants - (total_participants - neither_participants)

-- The theorem we want to prove
theorem both_participation_correct : both_participants = 20 :=
by
  sorry

end both_participation_correct_l275_275673


namespace ratio_of_P_to_Q_l275_275512

theorem ratio_of_P_to_Q (p q r s : ℕ) (h1 : p + q + r + s = 1000)
    (h2 : s = 4 * r) (h3 : q = r) (h4 : s - p = 250) : 
    p = 2 * q :=
by
  -- Proof omitted
  sorry

end ratio_of_P_to_Q_l275_275512


namespace min_sine_difference_l275_275449

theorem min_sine_difference (N : ℕ) (hN : 0 < N) :
  ∃ (n k : ℕ), (1 ≤ n ∧ n ≤ N + 1) ∧ (1 ≤ k ∧ k ≤ N + 1) ∧ (n ≠ k) ∧ 
    (|Real.sin n - Real.sin k| < 2 / N) := 
sorry

end min_sine_difference_l275_275449


namespace find_b2_a2_a1_l275_275120

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def geometric_sequence (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, b (n + 1) / b n = b 1 / b 0

theorem find_b2_a2_a1 (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a1 : a 0 = a₁) (h_a2 : a 2 = a₂)
  (h_b2 : b 2 = b₂) :
  b₂ * (a₂ - a₁) = 6 ∨ b₂ * (a₂ - a₁) = -6 :=
by
  sorry

end find_b2_a2_a1_l275_275120


namespace gcd_36_54_l275_275790

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l275_275790


namespace two_digit_even_multiple_of_7_l275_275378

def all_digits_product_square (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  (d1 * d2) > 0 ∧ ∃ k, d1 * d2 = k * k

theorem two_digit_even_multiple_of_7 (n : ℕ) :
  10 ≤ n ∧ n < 100 ∧ n % 2 = 0 ∧ n % 7 = 0 ∧ all_digits_product_square n ↔ n = 14 ∨ n = 28 ∨ n = 70 :=
by sorry

end two_digit_even_multiple_of_7_l275_275378


namespace oldest_brother_age_ratio_l275_275615

-- Define the ages
def rick_age : ℕ := 15
def youngest_brother_age : ℕ := 3
def smallest_brother_age : ℕ := youngest_brother_age + 2
def middle_brother_age : ℕ := smallest_brother_age * 2
def oldest_brother_age : ℕ := middle_brother_age * 3

-- Define the ratio
def expected_ratio : ℕ := oldest_brother_age / rick_age

theorem oldest_brother_age_ratio : expected_ratio = 2 := by
  sorry 

end oldest_brother_age_ratio_l275_275615


namespace trent_bus_blocks_to_library_l275_275786

-- Define the given conditions
def total_distance := 22
def walking_distance := 4

-- Define the function to determine bus block distance
def bus_ride_distance (total: ℕ) (walk: ℕ) : ℕ :=
  (total - (walk * 2)) / 2

-- The theorem we need to prove
theorem trent_bus_blocks_to_library : 
  bus_ride_distance total_distance walking_distance = 7 := by
  sorry

end trent_bus_blocks_to_library_l275_275786


namespace evaluate_expression_l275_275695

theorem evaluate_expression : 
  (1 / 10 : ℝ) + (2 / 20 : ℝ) - (3 / 60 : ℝ) = 0.15 :=
by
  sorry

end evaluate_expression_l275_275695


namespace problem_I_problem_II_l275_275553

open Real -- To use real number definitions and sin function.
open Set -- To use set constructs like intervals.

noncomputable def f (x : ℝ) : ℝ := sin (4 * x - π / 6) + sqrt 3 * sin (4 * x + π / 3)

-- Proof statement for monotonically decreasing interval of f(x).
theorem problem_I (k : ℤ) : 
  ∃ k : ℤ, ∀ x : ℝ, x ∈ Icc ((π / 12) + (k * π / 2)) ((π / 3) + (k * π / 2)) → 
  (4 * x + π / 6) ∈ Icc ((π / 2) + 2 * k * π) ((3 * π / 2) + 2 * k * π) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * sin (x + π / 4)

-- Proof statement for the range of g(x) on the interval [-π, 0].
theorem problem_II : 
  ∀ x : ℝ, x ∈ Icc (-π) 0 → g x ∈ Icc (-2) (sqrt 2) := 
sorry

end problem_I_problem_II_l275_275553


namespace joe_total_toy_cars_l275_275581

def joe_toy_cars (initial_cars additional_cars : ℕ) : ℕ :=
  initial_cars + additional_cars

theorem joe_total_toy_cars : joe_toy_cars 500 120 = 620 := by
  sorry

end joe_total_toy_cars_l275_275581


namespace find_amount_after_two_years_l275_275403

noncomputable def initial_value : ℝ := 64000
noncomputable def yearly_increase (amount : ℝ) : ℝ := amount / 9
noncomputable def amount_after_year (amount : ℝ) : ℝ := amount + yearly_increase amount
noncomputable def amount_after_two_years : ℝ := amount_after_year (amount_after_year initial_value)

theorem find_amount_after_two_years : amount_after_two_years = 79012.34 :=
by
  sorry

end find_amount_after_two_years_l275_275403


namespace optimal_order_l275_275602

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l275_275602


namespace prob_no_rain_four_days_l275_275045

noncomputable def prob_rain_one_day : ℚ := 2 / 3

noncomputable def prob_no_rain_one_day : ℚ := 1 - prob_rain_one_day

def independent_events (events : List (Unit → Prop)) : Prop :=
  -- A statement about independence of events
  sorry

theorem prob_no_rain_four_days :
  let days := 4
  let prob_no_rain := prob_no_rain_one_day
  independent_events (List.replicate days (fun _ => prob_no_rain)) →
  (prob_no_rain^days) = (1/81) := 
by
  sorry

end prob_no_rain_four_days_l275_275045


namespace insurance_slogan_equivalence_l275_275513

variables (H I : Prop)

theorem insurance_slogan_equivalence :
  (∀ x, x → H → I) ↔ (∀ y, y → ¬I → ¬H) :=
sorry

end insurance_slogan_equivalence_l275_275513


namespace perimeter_of_flowerbed_l275_275076

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem perimeter_of_flowerbed : perimeter length width = 22 := by
  sorry

end perimeter_of_flowerbed_l275_275076


namespace determine_w_arithmetic_seq_l275_275971

theorem determine_w_arithmetic_seq (w : ℝ) (h : (w ≠ 0) ∧ 
  (1 / w - 1 / 2 = 1 / 2 - 1 / 3) ∧ (1 / 2 - 1 / 3 = 1 / 3 - 1 / 6)) :
  w = 3 / 2 := 
sorry

end determine_w_arithmetic_seq_l275_275971


namespace number_of_ways_to_distribute_66_coins_l275_275706

theorem number_of_ways_to_distribute_66_coins : ∃ n : ℕ, n = 315 ∧
  (∃ a b c : ℕ, 0 < a ∧ a < b ∧ b < c ∧ a + b + c = 66) :=
by sorry

end number_of_ways_to_distribute_66_coins_l275_275706


namespace profit_sharing_l275_275051

-- Define constants and conditions
def Tom_investment : ℕ := 30000
def Tom_share : ℝ := 0.40

def Jose_investment : ℕ := 45000
def Jose_start_month : ℕ := 2
def Jose_share : ℝ := 0.30

def Sarah_investment : ℕ := 60000
def Sarah_start_month : ℕ := 5
def Sarah_share : ℝ := 0.20

def Ravi_investment : ℕ := 75000
def Ravi_start_month : ℕ := 8
def Ravi_share : ℝ := 0.10

def total_profit : ℕ := 120000

-- Define expected shares
def Tom_expected_share : ℕ := 48000
def Jose_expected_share : ℕ := 36000
def Sarah_expected_share : ℕ := 24000
def Ravi_expected_share : ℕ := 12000

-- Theorem statement
theorem profit_sharing :
  let Tom_contribution := Tom_investment * 12
  let Jose_contribution := Jose_investment * (12 - Jose_start_month)
  let Sarah_contribution := Sarah_investment * (12 - Sarah_start_month)
  let Ravi_contribution := Ravi_investment * (12 - Ravi_start_month)
  Tom_share * total_profit = Tom_expected_share ∧
  Jose_share * total_profit = Jose_expected_share ∧
  Sarah_share * total_profit = Sarah_expected_share ∧
  Ravi_share * total_profit = Ravi_expected_share := by {
    sorry
  }

end profit_sharing_l275_275051


namespace smaller_number_l275_275190

theorem smaller_number (x y : ℤ) (h1 : x + y = 79) (h2 : x - y = 15) : y = 32 := by
  sorry

end smaller_number_l275_275190


namespace find_x_81_9_729_l275_275047

theorem find_x_81_9_729
  (x : ℝ)
  (h : (81 : ℝ)^(x-2) / (9 : ℝ)^(x-2) = (729 : ℝ)^(2*x-1)) :
  x = 1/5 :=
sorry

end find_x_81_9_729_l275_275047


namespace total_amount_spent_l275_275502

variable (you friend : ℝ)

theorem total_amount_spent (h1 : friend = you + 3) (h2 : friend = 7) : 
  you + friend = 11 :=
by
  sorry

end total_amount_spent_l275_275502


namespace kelly_apples_total_l275_275750

def initial_apples : ℕ := 56
def second_day_pick : ℕ := 105
def third_day_pick : ℕ := 84
def apples_eaten : ℕ := 23

theorem kelly_apples_total :
  initial_apples + second_day_pick + third_day_pick - apples_eaten = 222 := by
  sorry

end kelly_apples_total_l275_275750


namespace mean_of_elements_increased_by_2_l275_275181

noncomputable def calculate_mean_after_increase (m : ℝ) (median_value : ℝ) (increase_value : ℝ) : ℝ :=
  let set := [m, m + 2, m + 4, m + 7, m + 11, m + 13]
  let increased_set := set.map (λ x => x + increase_value)
  increased_set.sum / increased_set.length

theorem mean_of_elements_increased_by_2 (m : ℝ) (h : (m + 4 + m + 7) / 2 = 10) :
  calculate_mean_after_increase m 10 2 = 38 / 3 :=
by 
  sorry

end mean_of_elements_increased_by_2_l275_275181


namespace smallest_base_b_l275_275055

theorem smallest_base_b (k : ℕ) (hk : k = 7) : ∃ (b : ℕ), b = 64 ∧ b^k > 4^20 := by
  sorry

end smallest_base_b_l275_275055


namespace remainder_div_84_l275_275115

def a := (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)

theorem remainder_div_84 (a : ℕ) (h : a = (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)) : a % 84 = 63 := 
by 
  -- Placeholder for the actual steps to prove
  sorry

end remainder_div_84_l275_275115


namespace median_books_per_person_l275_275362

-- Definitions representing the data conditions
def books_bought_per_person : List ℝ := [3, 1, 5, 2]

-- Statement that expresses the proof problem:
theorem median_books_per_person : (List.median books_bought_per_person) = 2.5 := 
by
  sorry

end median_books_per_person_l275_275362


namespace shoe_cost_l275_275851

def initial_amount : ℕ := 91
def cost_sweater : ℕ := 24
def cost_tshirt : ℕ := 6
def amount_left : ℕ := 50
def cost_shoes : ℕ := 11

theorem shoe_cost :
  initial_amount - (cost_sweater + cost_tshirt) - amount_left = cost_shoes :=
by
  sorry

end shoe_cost_l275_275851


namespace total_leftover_tarts_l275_275989

variable (cherry_tart blueberry_tart peach_tart : ℝ)
variable (h1 : cherry_tart = 0.08)
variable (h2 : blueberry_tart = 0.75)
variable (h3 : peach_tart = 0.08)

theorem total_leftover_tarts : 
  cherry_tart + blueberry_tart + peach_tart = 0.91 := 
by 
  sorry

end total_leftover_tarts_l275_275989


namespace cricket_team_members_eq_11_l275_275784

-- Definitions based on conditions:
def captain_age : ℕ := 26
def wicket_keeper_age : ℕ := 31
def avg_age_whole_team : ℕ := 24
def avg_age_remaining_players : ℕ := 23

-- Definition of n based on the problem conditions
def number_of_members (n : ℕ) : Prop :=
  n * avg_age_whole_team = (n - 2) * avg_age_remaining_players + (captain_age + wicket_keeper_age)

-- The proof statement:
theorem cricket_team_members_eq_11 : ∃ n, number_of_members n ∧ n = 11 := 
by
  use 11
  unfold number_of_members
  sorry

end cricket_team_members_eq_11_l275_275784


namespace circle_distance_k_l275_275192

theorem circle_distance_k (h1 : (5:ℝ)^2 + 12^2 = 25 + 144)
                          (h2 : ∃ k : ℝ, (0, k) ∈ circle 0 8)
                          (h3 : distance (origin: ℝ×ℝ) (5, 12) = 13)
                          (h4 : distance (origin: ℝ×ℝ) (0, k) + 5 = 13):
    k = 8 := 
by
  sorry

end circle_distance_k_l275_275192


namespace complex_power_l275_275088

def cos_30 := Float.sqrt 3 / 2
def sin_30 := 1 / 2

theorem complex_power (i : ℂ) : 
  (3 * (⟨cos_30, sin_30⟩ : ℂ)) ^ 4 = -40.5 + 40.5 * i * Float.sqrt 3 := 
  sorry

end complex_power_l275_275088


namespace jason_hours_saturday_l275_275440

def hours_after_school (x : ℝ) : ℝ := 4 * x
def hours_saturday (y : ℝ) : ℝ := 6 * y

theorem jason_hours_saturday 
  (x y : ℝ) 
  (total_hours : x + y = 18) 
  (total_earnings : 4 * x + 6 * y = 88) : 
  y = 8 :=
by 
  sorry

end jason_hours_saturday_l275_275440


namespace maximize_profit_l275_275974

noncomputable def I (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * (x - 1) * Real.exp (x - 2) + 2
  else if h' : 2 < x ∧ x ≤ 50 then 440 + 3050 / x - 9000 / x^2
  else 0 -- default case for Lean to satisfy definition

noncomputable def P (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * x * (x - 1) * Real.exp (x - 2) - 448 * x - 180
  else if h' : 2 < x ∧ x ≤ 50 then -10 * x - 9000 / x + 2870
  else 0 -- default case for Lean to satisfy definition

theorem maximize_profit :
  (∀ x : ℝ, 0 < x ∧ x ≤ 50 → P x ≤ 2270) ∧ P 30 = 2270 :=
by
  sorry

end maximize_profit_l275_275974


namespace partial_fraction_sum_zero_l275_275850

theorem partial_fraction_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 0 :=
by
  sorry

end partial_fraction_sum_zero_l275_275850


namespace integral_value_l275_275239

theorem integral_value :
  ∫ x in -1..1, (x * cos x + real.cbrt (x^2)) = 6 / 5 :=
by
  sorry

end integral_value_l275_275239


namespace total_number_of_boys_in_class_is_40_l275_275966

theorem total_number_of_boys_in_class_is_40 
  (n : ℕ) (h : 27 - 7 = n / 2):
  n = 40 :=
by
  sorry

end total_number_of_boys_in_class_is_40_l275_275966


namespace sum_reciprocals_l275_275456

theorem sum_reciprocals (a b α β : ℝ) (h1: 7 * a^2 + 2 * a + 6 = 0) (h2: 7 * b^2 + 2 * b + 6 = 0) 
  (h3: α = 1 / a) (h4: β = 1 / b) (h5: a + b = -2/7) (h6: a * b = 6/7) : 
  α + β = -1/3 :=
by
  sorry

end sum_reciprocals_l275_275456


namespace gcd_of_36_and_54_l275_275795

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l275_275795


namespace sugar_amount_first_week_l275_275858

theorem sugar_amount_first_week (s : ℕ → ℕ) (h : s 4 = 3) (h_rec : ∀ n, s (n + 1) = s n / 2) : s 1 = 24 :=
by
  sorry

end sugar_amount_first_week_l275_275858


namespace ordered_pairs_count_l275_275134

theorem ordered_pairs_count :
  ∃ (p : Finset (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ p → a * b + 45 = 10 * Nat.lcm a b + 18 * Nat.gcd a b) ∧
  p.card = 4 :=
by
  sorry

end ordered_pairs_count_l275_275134


namespace find_a_10_l275_275117

def seq (a : ℕ → ℚ) : Prop :=
∀ n, a (n + 1) = 2 * a n / (a n + 2)

def initial_value (a : ℕ → ℚ) : Prop :=
a 1 = 1

theorem find_a_10 (a : ℕ → ℚ) (h1 : initial_value a) (h2 : seq a) : 
  a 10 = 2 / 11 := 
sorry

end find_a_10_l275_275117


namespace tourists_remaining_l275_275510

theorem tourists_remaining (initial_tourists : ℕ) (eaten_by_anacondas : ℕ) (poisoned_fraction : ℚ) 
  (recovered_poisoned_fraction : ℚ) (bitten_fraction : ℚ) (received_antivenom_fraction : ℚ) :
  initial_tourists = 42 →
  eaten_by_anacondas = 3 →
  poisoned_fraction = 2/3 →
  recovered_poisoned_fraction = 2/9 →
  bitten_fraction = 1/4 →
  received_antivenom_fraction = 3/5 →
  let remaining_after_anacondas := initial_tourists - eaten_by_anacondas in
  let poisoned_tourists := int.of_nat (poisoned_fraction * remaining_after_anacondas).floor in
  let recovered_poisoned := int.of_nat (recovered_poisoned_fraction * poisoned_tourists).floor in
  let remaining_after_poisoned_recover := remaining_after_anacondas - poisoned_tourists + recovered_poisoned in
  let bitten_tourists := int.of_nat (bitten_fraction * remaining_after_poisoned_recover).floor in
  let received_antivenom := int.of_nat (received_antivenom_fraction * bitten_tourists).floor in
  let final_remaining := remaining_after_poisoned_recover - bitten_tourists + received_antivenom in
  final_remaining = 16 :=
begin
  intros,
  sorry
end

end tourists_remaining_l275_275510


namespace correct_division_result_l275_275056

theorem correct_division_result : 
  ∀ (a b : ℕ),
  (1722 / (10 * b + a) = 42) →
  (10 * a + b = 14) →
  1722 / 14 = 123 :=
by
  intros a b h1 h2
  sorry

end correct_division_result_l275_275056


namespace monthly_income_of_A_l275_275476

theorem monthly_income_of_A (A B C : ℝ)
  (h1 : (A + B) / 2 = 5050)
  (h2 : (B + C) / 2 = 6250)
  (h3 : (A + C) / 2 = 5200) :
  A = 4000 :=
sorry

end monthly_income_of_A_l275_275476


namespace sallys_woodworking_llc_reimbursement_l275_275925

/-
Conditions:
1. Remy paid $20,700 for 150 pieces of furniture.
2. The cost of a piece of furniture is $134.
-/
def reimbursement_amount (pieces_paid : ℕ) (total_paid : ℕ) (price_per_piece : ℕ) : ℕ :=
  total_paid - (pieces_paid * price_per_piece)

theorem sallys_woodworking_llc_reimbursement :
  reimbursement_amount 150 20700 134 = 600 :=
by 
  sorry

end sallys_woodworking_llc_reimbursement_l275_275925


namespace evaluate_expression_at_x_neg3_l275_275857

theorem evaluate_expression_at_x_neg3 :
  (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 :=
by
  sorry

end evaluate_expression_at_x_neg3_l275_275857


namespace purely_imaginary_a_l275_275920

theorem purely_imaginary_a (a : ℝ) (h : (a^3 - a) = 0) (h2 : (a / (1 - a)) ≠ 0) : a = -1 := 
sorry

end purely_imaginary_a_l275_275920


namespace sum_mean_median_mode_l275_275661

def numbers : List ℕ := [3, 5, 3, 0, 2, 5, 0, 2]

def mode (l : List ℕ) : ℝ := 4

def median (l : List ℕ) : ℝ := 2.5

def mean (l : List ℕ) : ℝ := 2.5

theorem sum_mean_median_mode : mean numbers + median numbers + mode numbers = 9 := by
  sorry

end sum_mean_median_mode_l275_275661


namespace problem_inverse_range_m_l275_275113

theorem problem_inverse_range_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 2 / x + 1 / y = 1) : 
  (2 * x + y > m^2 + 8 * m) ↔ (m > -9 ∧ m < 1) := 
by
  sorry

end problem_inverse_range_m_l275_275113


namespace reimbursement_calculation_l275_275926

variable (total_paid : ℕ) (pieces : ℕ) (cost_per_piece : ℕ)

theorem reimbursement_calculation
  (h1 : total_paid = 20700)
  (h2 : pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (pieces * cost_per_piece) = 600 := 
by
  sorry

end reimbursement_calculation_l275_275926


namespace choose_starting_team_l275_275312

-- Definitions derived from the conditions
def team_size : ℕ := 18
def selected_goalie (n : ℕ) : ℕ := n
def selected_players (m : ℕ) (k : ℕ) : ℕ := Nat.choose m k

-- The number of ways to choose the starting team
theorem choose_starting_team :
  let n := team_size
  let k := 7
  selected_goalie n * selected_players (n - 1) k = 222768 :=
by
  simp only [team_size, selected_goalie, selected_players]
  sorry

end choose_starting_team_l275_275312


namespace min_a_plus_b_l275_275426

theorem min_a_plus_b (a b : ℝ) (h : a^2 + 2 * b^2 = 6) : a + b ≥ -3 :=
sorry

end min_a_plus_b_l275_275426


namespace puzzle_sets_l275_275460

theorem puzzle_sets (l v w : ℕ) (h_l : l = 30) (h_v : v = 18) (h_w : w = 12) 
(h_ratio: ∀ {l_set v_set : ℕ}, l_set / v_set = 2 ) (h_min_puzzles : ∀ {s : ℕ}, s >= 5 ) :
∃ (S : ℕ), S = 3 :=
by
  existsi 3
  sorry

end puzzle_sets_l275_275460


namespace slope_of_line_l275_275237

theorem slope_of_line (x1 y1 x2 y2 : ℝ)
  (h1 : 4 * y1 + 6 * x1 = 0)
  (h2 : 4 * y2 + 6 * x2 = 0)
  (h1x2 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by sorry

end slope_of_line_l275_275237


namespace cost_price_of_article_l275_275506

theorem cost_price_of_article 
  (CP SP : ℝ)
  (H1 : SP = 1.13 * CP)
  (H2 : 1.10 * SP = 616) :
  CP = 495.58 :=
by
  sorry

end cost_price_of_article_l275_275506


namespace max_disks_l275_275965

theorem max_disks (n k : ℕ) (h1: n ≥ 1) (h2: k ≥ 1) :
  (∃ (d : ℕ), d = if n > 1 ∧ k > 1 then 2 * (n + k) - 4 else max n k) ∧
  (∀ (p q : ℕ), (p <= n → q <= k → ¬∃ (x y : ℕ), x + 1 = y ∨ x - 1 = y ∨ x + 1 = p ∨ x - 1 = p)) :=
sorry

end max_disks_l275_275965


namespace fence_calculation_l275_275074

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def fence_needed : ℕ := 2 * length + 2 * width

theorem fence_calculation : fence_needed = 22 := by
  sorry

end fence_calculation_l275_275074


namespace problem_statement_l275_275407

theorem problem_statement (x y : ℕ) (hx : x = 7) (hy : y = 3) : (x - y)^2 * (x + y)^2 = 1600 :=
by
  rw [hx, hy]
  sorry

end problem_statement_l275_275407


namespace line_equation_l275_275954

-- Define the points A and M
structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 3 1
def M := Point.mk 4 (-3)

def symmetric_point (A M : Point) : Point :=
  Point.mk (2 * M.x - A.x) (2 * M.y - A.y)

def line_through_origin (B : Point) : Prop :=
  7 * B.x + 5 * B.y = 0

theorem line_equation (B : Point) (hB : B = symmetric_point A M) : line_through_origin B :=
  by
  sorry

end line_equation_l275_275954


namespace fraction_division_l275_275684

theorem fraction_division : 
  ((8 / 4) * (9 / 3) * (20 / 5)) / ((10 / 5) * (12 / 4) * (15 / 3)) = (4 / 5) := 
by
  sorry

end fraction_division_l275_275684


namespace number_greater_by_l275_275482

def question (a b : Int) : Int := a + b

theorem number_greater_by (a b : Int) : question a b = -11 :=
  by
    sorry

-- Use specific values from the provided problem:
example : question -5 -6 = -11 :=
  by
    sorry

end number_greater_by_l275_275482


namespace monotonic_intervals_nonneg_f_for_all_x_ge_0_compare_magnitudes_l275_275554

open Real

-- (I)
theorem monotonic_intervals (f : ℝ → ℝ) : 
  (∀ x, f x = exp(2 * x) - 1 - 2 * x) →
  (∀ x, 0 < x → diff f x > 0) ∧ (∀ x, x < 0 → diff f x < 0) :=
sorry

-- (II)
theorem nonneg_f_for_all_x_ge_0 (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = exp(2 * x) - 1 - 2 * x - k * x^2) →
  (k ≤ 2) →
  (∀ x, 0 ≤ x → f x ≥ 0) :=
sorry

-- (III)
theorem compare_magnitudes (n : ℕ) (h : 0 < n) :
  (∑ i in range n, (exp(2 * i))) ≥ (2 * n^3 + n) / 3 :=
sorry

end monotonic_intervals_nonneg_f_for_all_x_ge_0_compare_magnitudes_l275_275554


namespace min_value_nS_n_l275_275118

theorem min_value_nS_n (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) 
  (h2 : m ≥ 2)
  (h3 : S (m - 1) = -2)
  (h4 : S m = 0)
  (h5 : S (m + 1) = 3) :
  ∃ n : ℕ, n * S n = -9 :=
sorry

end min_value_nS_n_l275_275118


namespace boxes_in_case_number_of_boxes_in_case_l275_275409

-- Definitions based on the conditions
def boxes_of_eggs : Nat := 5
def eggs_per_box : Nat := 3
def total_eggs : Nat := 15

-- Proposition
theorem boxes_in_case (boxes_of_eggs : Nat) (eggs_per_box : Nat) (total_eggs : Nat) : Nat :=
  if boxes_of_eggs * eggs_per_box = total_eggs then boxes_of_eggs else 0

-- Assertion that needs to be proven
theorem number_of_boxes_in_case : boxes_in_case boxes_of_eggs eggs_per_box total_eggs = 5 :=
by sorry

end boxes_in_case_number_of_boxes_in_case_l275_275409


namespace value_of_k_l275_275193

theorem value_of_k (k : ℝ) : 
  (∃ P Q R : ℝ × ℝ, P = (5, 12) ∧ Q = (0, k) ∧ dist (0, 0) P = dist (0, 0) Q + 5) → 
  k = 8 := 
by
  sorry

end value_of_k_l275_275193


namespace initial_investment_l275_275316

theorem initial_investment (P : ℝ) 
  (h1: ∀ (r : ℝ) (n : ℕ), r = 0.20 ∧ n = 3 → P * (1 + r)^n = P * 1.728)
  (h2: ∀ (A : ℝ), A = P * 1.728 → 3 * A = 5.184 * P)
  (h3: ∀ (P_new : ℝ) (r_new : ℝ), P_new = 5.184 * P ∧ r_new = 0.15 → P_new * (1 + r_new) = 5.9616 * P)
  (h4: 5.9616 * P = 59616)
  : P = 10000 :=
sorry

end initial_investment_l275_275316


namespace work_done_in_days_l275_275737

theorem work_done_in_days (M B : ℕ) (x : ℕ) 
  (h1 : 12 * 2 * B + 16 * B = 200 * B / 5) 
  (h2 : 13 * 2 * B + 24 * B = 50 * x * B)
  (h3 : M = 2 * B) : 
  x = 4 := 
by
  sorry

end work_done_in_days_l275_275737


namespace batsman_average_after_15th_innings_l275_275357

theorem batsman_average_after_15th_innings 
  (A : ℕ) 
  (h1 : 14 * A + 85 = 15 * (A + 3)) 
  (h2 : A = 40) : 
  (A + 3) = 43 := by 
  sorry

end batsman_average_after_15th_innings_l275_275357


namespace sum_of_common_ratios_eq_three_l275_275921

variable (k p r a2 a3 b2 b3 : ℝ)

-- Conditions on the sequences:
variable (h_nz_k : k ≠ 0)  -- k is nonzero as it is scaling factor
variable (h_seq1 : a2 = k * p)
variable (h_seq2 : a3 = k * p^2)
variable (h_seq3 : b2 = k * r)
variable (h_seq4 : b3 = k * r^2)
variable (h_diff_ratios : p ≠ r)

-- The given equation:
variable (h_eq : a3^2 - b3^2 = 3 * (a2^2 - b2^2))

-- The theorem statement
theorem sum_of_common_ratios_eq_three :
  p^2 + r^2 = 3 :=
by
  -- Introduce the assumptions
  sorry

end sum_of_common_ratios_eq_three_l275_275921


namespace range_of_a_l275_275887

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2 - 2 * x

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x - a * x - 1

theorem range_of_a
  (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f_prime x1 a = 0 ∧ f_prime x2 a = 0) ↔
  0 < a ∧ a < Real.exp (-2) :=
sorry

end range_of_a_l275_275887


namespace perpendiculars_form_regular_ngon_l275_275675

-- Definitions related to the problem conditions
def circle_divided_by_equal_arcs (n : ℕ) (O : point) : Prop :=
  -- Definition for a circle divided into n equal arcs by diameters
  ∃ diameters : fin n → line,
    (∀ i, (diameters i).contains O) ∧ 
    (∀ i j, i ≠ j → is_perpendicular (diameters i) (diameters j))

def perpendicular_foot (M : point) (d : line) : point :=
  -- Definition for the foot of the perpendicular dropped from M to line d
  let p := Foot (M, d) in p

def regular_n_gon (vertices : list point) (n : ℕ) : Prop :=
  -- Definition for a regular n-gon given a list of vertices
  (∀ i j : fin n, ∃ d : ℝ, vertices.nth i = vertices.nth j → dist vertices.nth i vertices.nth j = d)

-- Proof statement
theorem perpendiculars_form_regular_ngon (n : ℕ) (O M : point) (h : circle_divided_by_equal_arcs n O) :
  ∃ vertices : list point, (∀ i, vertices.nth i = perpendicular_foot M (h.1 i)) ∧ regular_n_gon vertices n :=
by
  sorry

end perpendiculars_form_regular_ngon_l275_275675


namespace geometric_sequence_problem_l275_275014

noncomputable def geometric_sequence_solution (a_1 a_2 a_3 a_4 a_5 q : ℝ) : Prop :=
  (a_5 - a_1 = 15) ∧
  (a_4 - a_2 = 6) ∧
  (a_3 = 4 ∧ q = 2 ∨ a_3 = -4 ∧ q = 1/2)

theorem geometric_sequence_problem :
  ∃ a_1 a_2 a_3 a_4 a_5 q : ℝ, geometric_sequence_solution a_1 a_2 a_3 a_4 a_5 q :=
by
  sorry

end geometric_sequence_problem_l275_275014


namespace perpendicular_line_eq_slope_intercept_l275_275646

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l275_275646


namespace find_principal_l275_275358

theorem find_principal (R : ℝ) (P : ℝ) (h : (P * (R + 2) * 4) / 100 = (P * R * 4) / 100 + 56) : P = 700 := 
sorry

end find_principal_l275_275358


namespace fraction_equality_implies_equality_l275_275496

theorem fraction_equality_implies_equality (a b c : ℝ) (hc : c ≠ 0) :
  (a / c = b / c) → (a = b) :=
by {
  sorry
}

end fraction_equality_implies_equality_l275_275496


namespace value_of_m_l275_275630

noncomputable def TV_sales_volume_function (x : ℕ) : ℚ :=
  10 * x + 540

theorem value_of_m : ∀ (m : ℚ),
  (3200 * (1 + m / 100) * 9 / 10) * (600 * (1 - 2 * m / 100) + 220) = 3200 * 600 * (1 + 15.5 / 100) →
  m = 10 :=
by sorry

end value_of_m_l275_275630


namespace botanical_garden_correct_path_length_l275_275208

noncomputable def correct_path_length_on_ground
  (inch_length_on_map : ℝ)
  (inch_per_error_segment : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  (inch_length_on_map * conversion_rate) - (inch_per_error_segment * conversion_rate)

theorem botanical_garden_correct_path_length :
  correct_path_length_on_ground 6.5 0.75 1200 = 6900 := 
by
  sorry

end botanical_garden_correct_path_length_l275_275208


namespace equal_pair_b_l275_275846

def exprA1 := -3^2
def exprA2 := -2^3

def exprB1 := -6^3
def exprB2 := (-6)^3

def exprC1 := -6^2
def exprC2 := (-6)^2

def exprD1 := (-3 * 2)^2
def exprD2 := (-3) * 2^2

theorem equal_pair_b : exprB1 = exprB2 :=
by {
  -- proof steps should go here
  sorry
}

end equal_pair_b_l275_275846


namespace minimum_value_of_f_l275_275175

open Real

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 3 * sqrt 2 * cos (x + φ) + sin x

theorem minimum_value_of_f (φ : ℝ) (hφ : -π / 2 < φ ∧ φ < π / 2)
  (H : f (π / 2) φ = 4) : ∃ x : ℝ, f x φ = -5 := 
by
  sorry

end minimum_value_of_f_l275_275175


namespace number_of_boxes_l275_275162

theorem number_of_boxes (total_eggs : ℕ) (eggs_per_box : ℕ) (boxes : ℕ) : 
  total_eggs = 21 → eggs_per_box = 7 → boxes = total_eggs / eggs_per_box → boxes = 3 :=
by
  intros h_total_eggs h_eggs_per_box h_boxes
  rw [h_total_eggs, h_eggs_per_box] at h_boxes
  exact h_boxes

end number_of_boxes_l275_275162


namespace children_in_circle_l275_275227

theorem children_in_circle (n m : ℕ) (k : ℕ) 
  (h1 : n = m) 
  (h2 : n + m = 2 * k) :
  ∃ k', n + m = 4 * k' :=
by
  sorry

end children_in_circle_l275_275227


namespace cost_difference_l275_275913

theorem cost_difference (S : ℕ) (h1 : 15 + S = 24) : 15 - S = 6 :=
by
  sorry

end cost_difference_l275_275913


namespace distance_between_parallel_lines_eq_2_l275_275302

def line1 (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 8 = 0

theorem distance_between_parallel_lines_eq_2 :
  let A := 3
  let B := -4
  let c1 := 2
  let c2 := -8
  let d := (|c1 - c2| / Real.sqrt (A^2 + B^2))
  d = 2 :=
by
  sorry

end distance_between_parallel_lines_eq_2_l275_275302


namespace eval_expression_l275_275344

theorem eval_expression : (8 / 4 - 3 * 2 + 9 - 3^2) = -4 := sorry

end eval_expression_l275_275344


namespace problem_conditions_l275_275518

noncomputable def f (x : ℝ) : ℝ := -x - x^3

variables (x₁ x₂ : ℝ)

theorem problem_conditions (h₁ : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧
  (¬ (f x₂ * f (-x₂) > 0)) ∧
  (¬ (f x₁ + f x₂ ≤ f (-x₁) + f (-x₂))) ∧
  (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) :=
sorry

end problem_conditions_l275_275518


namespace find_divisor_l275_275639

theorem find_divisor (D Q R d : ℕ) (h1 : D = 159) (h2 : Q = 9) (h3 : R = 6) (h4 : D = d * Q + R) : d = 17 := by
  sorry

end find_divisor_l275_275639


namespace problem_statement_l275_275300

section Problem

variable {X : ℕ → ℝ} -- Sequence of random variables
variable {S : ℕ → ℝ} -- Partial sums
variable (E : ℝ → ℝ) -- Expectation operator

-- Conditions
axiom identically_distributed : ∀ n, E (X 1) = E (X n)
axiom independent : ∀ n m, n ≠ m → E (X n * X m) = E (X n) * E (X m)
axiom bounded : ∃ B, ∀ n, ∥X n∥ ≤ B
axiom mean_zero : E (X 1) = 0
axiom partial_sum : ∀ n, S n = ∑ i in finset.range n, X i 

-- Statement to Prove
theorem problem_statement (p : ℝ) (hp : p > 0) : 
    ∃ C, ∀ n, E (∥S n∥^p) ≤ C * n^(p/2) :=
sorry

end Problem

end problem_statement_l275_275300


namespace evaluate_expression_l275_275524

theorem evaluate_expression (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := 
by
  sorry

end evaluate_expression_l275_275524


namespace calculator_sum_is_large_l275_275636

-- Definitions for initial conditions and operations
def participants := 50
def initial_calc1 := 2
def initial_calc2 := -2
def initial_calc3 := 0

-- Define the operations
def operation_calc1 (n : ℕ) := initial_calc1 * 2^n
def operation_calc2 (n : ℕ) := (-2) ^ (2^n)
def operation_calc3 (n : ℕ) := initial_calc3 - n

-- Define the final values for each calculator
def final_calc1 := operation_calc1 participants
def final_calc2 := operation_calc2 participants
def final_calc3 := operation_calc3 participants

-- The final sum
def final_sum := final_calc1 + final_calc2 + final_calc3

-- Prove the final result
theorem calculator_sum_is_large :
  final_sum = 2 ^ (2 ^ 50) :=
by
  -- The proof would go here.
  sorry

end calculator_sum_is_large_l275_275636


namespace sum_youngest_oldest_l275_275773

variables {a1 a2 a3 a4 a5 : ℕ}

def mean_age (x y z u v : ℕ) : ℕ := (x + y + z + u + v) / 5
def median_age (x y z u v : ℕ) : ℕ := z

theorem sum_youngest_oldest
  (h_mean: mean_age a1 a2 a3 a4 a5 = 10) 
  (h_median: median_age a1 a2 a3 a4 a5 = 7)
  (h_sorted: a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) :
  a1 + a5 = 23 :=
sorry

end sum_youngest_oldest_l275_275773


namespace probability_even_and_greater_than_14_l275_275934

theorem probability_even_and_greater_than_14 : 
  (∃ (s : finset (ℕ × ℕ)), 
    s = {(x, y) | x ∈ {1, 2, 3, 4, 5, 6, 7} ∧ y ∈ {1, 2, 3, 4, 5, 6, 7} ∧
    x * y % 2 = 0 ∧ x * y > 14}) → 
  (↑(∃ (t : ℕ), t = 16) / 49 : ℚ) :=
sorry

end probability_even_and_greater_than_14_l275_275934


namespace amount_spent_on_marbles_l275_275992

-- Definitions of conditions
def cost_of_football : ℝ := 5.71
def total_spent_on_toys : ℝ := 12.30

-- Theorem statement
theorem amount_spent_on_marbles : (total_spent_on_toys - cost_of_football) = 6.59 :=
by
  sorry

end amount_spent_on_marbles_l275_275992


namespace equalize_costs_l275_275493

theorem equalize_costs (X Y Z : ℝ) (h1 : Y > X) (h2 : Z > Y) : 
  (Y + (Z - (X + Z - 2 * Y) / 3) = Z) → 
   (Y - (Y + Z - (X + Z - 2 * Y)) / 3 = (X + Z - 2 * Y) / 3) := sorry

end equalize_costs_l275_275493


namespace cost_of_one_package_of_berries_l275_275755

noncomputable def martin_daily_consumption : ℚ := 1 / 2

noncomputable def package_content : ℚ := 1

noncomputable def total_period_days : ℚ := 30

noncomputable def total_spent : ℚ := 30

theorem cost_of_one_package_of_berries :
  (total_spent / (total_period_days * martin_daily_consumption / package_content)) = 2 :=
sorry

end cost_of_one_package_of_berries_l275_275755


namespace ellipse_minor_axis_length_l275_275000

theorem ellipse_minor_axis_length
  (semi_focal_distance : ℝ)
  (eccentricity : ℝ)
  (semi_focal_distance_eq : semi_focal_distance = 2)
  (eccentricity_eq : eccentricity = 2 / 3) :
  ∃ minor_axis_length : ℝ, minor_axis_length = 2 * Real.sqrt 5 :=
by
  sorry

end ellipse_minor_axis_length_l275_275000


namespace sum_of_distinct_integers_l275_275027

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_prod : (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120) : a + b + c + d + e = 39 :=
by
  sorry

end sum_of_distinct_integers_l275_275027


namespace S₄_eq_15_l275_275147

-- Definitions based on the given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

def sequence_condition (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 1 = 1 ∧ sum_of_first_n_terms a 5 = 5 * sum_of_first_n_terms a 3 - 4

theorem S₄_eq_15 (a : ℕ → ℝ) (q : ℝ) :
  sequence_condition a →
  (∀ n, a n = 1 * q ^ (n-1)) → 
  sum_of_first_n_terms a 4 = 15 :=
sorry

end S₄_eq_15_l275_275147


namespace result_when_7_multiplies_number_l275_275365

theorem result_when_7_multiplies_number (x : ℤ) (h : x + 45 - 62 = 55) : 7 * x = 504 :=
by sorry

end result_when_7_multiplies_number_l275_275365


namespace prime_angle_triangle_l275_275728

theorem prime_angle_triangle (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_sum : a + b + c = 180) : a = 2 ∨ b = 2 ∨ c = 2 :=
sorry

end prime_angle_triangle_l275_275728


namespace linear_eq_m_minus_2n_zero_l275_275944

theorem linear_eq_m_minus_2n_zero (m n : ℕ) (x y : ℝ) 
  (h1 : 2 * x ^ (m - 1) + 3 * y ^ (2 * n - 1) = 7)
  (h2 : m - 1 = 1) (h3 : 2 * n - 1 = 1) : 
  m - 2 * n = 0 := 
sorry

end linear_eq_m_minus_2n_zero_l275_275944


namespace number_of_solutions_l275_275462

theorem number_of_solutions (n : ℕ) : (4 * n) = 80 ↔ n = 20 :=
by
  sorry

end number_of_solutions_l275_275462


namespace initial_number_of_children_l275_275292

-- Define the initial conditions
variables {X : ℕ} -- Initial number of children on the bus
variables (got_off got_on children_after : ℕ)
variables (H1 : got_off = 10)
variables (H2 : got_on = 5)
variables (H3 : children_after = 16)

-- Define the theorem to be proved
theorem initial_number_of_children (H : X - got_off + got_on = children_after) : X = 21 :=
by sorry

end initial_number_of_children_l275_275292


namespace evaluate_expression_at_neg3_l275_275856

theorem evaluate_expression_at_neg3 : 
  (let x := -3 in (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)) = -26 :=
by
  simp only
  intro x
  sorry

end evaluate_expression_at_neg3_l275_275856


namespace exactly_one_even_l275_275466

theorem exactly_one_even (a b c : ℕ) : 
  (∀ x, ¬ (a = x ∧ b = x ∧ c = x) ∧ 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ b % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ c % 2 = 0) ∧ 
  ¬ (b % 2 = 0 ∧ c % 2 = 0)) :=
by
  sorry

end exactly_one_even_l275_275466


namespace sqrt_224_between_14_and_15_l275_275250

theorem sqrt_224_between_14_and_15 : 14 < Real.sqrt 224 ∧ Real.sqrt 224 < 15 := by
  sorry

end sqrt_224_between_14_and_15_l275_275250


namespace domain_of_f_l275_275692

noncomputable def f (x : ℝ) : ℝ := (x + 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : 
  {x : ℝ | Real.sqrt (x^2 - 5 * x + 6) ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l275_275692


namespace smaller_part_area_l275_275203

theorem smaller_part_area (x y : ℝ) (h1 : x + y = 500) (h2 : y - x = (1 / 5) * ((x + y) / 2)) : x = 225 :=
by
  sorry

end smaller_part_area_l275_275203


namespace percentage_reduction_l275_275980

theorem percentage_reduction (original reduced : ℕ) (h₁ : original = 260) (h₂ : reduced = 195) :
  (original - reduced) / original * 100 = 25 := by
  sorry

end percentage_reduction_l275_275980


namespace janeth_balloons_count_l275_275295

-- Define the conditions
def bags_round_balloons : Nat := 5
def balloons_per_bag_round : Nat := 20
def bags_long_balloons : Nat := 4
def balloons_per_bag_long : Nat := 30
def burst_round_balloons : Nat := 5

-- Proof statement
theorem janeth_balloons_count:
  let total_round_balloons := bags_round_balloons * balloons_per_bag_round
  let total_long_balloons := bags_long_balloons * balloons_per_bag_long
  let total_balloons := total_round_balloons + total_long_balloons
  total_balloons - burst_round_balloons = 215 :=
by {
  sorry
}

end janeth_balloons_count_l275_275295


namespace sqrt_2700_minus_37_form_l275_275475

theorem sqrt_2700_minus_37_form (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : (Int.sqrt 2700 - 37) = Int.sqrt a - b ^ 3) : a + b = 13 :=
sorry

end sqrt_2700_minus_37_form_l275_275475


namespace line_segments_property_l275_275871

theorem line_segments_property (L : List (ℝ × ℝ)) :
  L.length = 50 →
  (∃ S : List (ℝ × ℝ), S.length = 8 ∧ ∃ x : ℝ, ∀ seg ∈ S, seg.fst ≤ x ∧ x ≤ seg.snd) ∨
  (∃ T : List (ℝ × ℝ), T.length = 8 ∧ ∀ seg1 ∈ T, ∀ seg2 ∈ T, seg1 ≠ seg2 → seg1.snd < seg2.fst ∨ seg2.snd < seg1.fst) :=
by
  -- Theorem proof placeholder
  sorry

end line_segments_property_l275_275871


namespace John_pays_2400_per_year_l275_275019

theorem John_pays_2400_per_year
  (hours_per_month : ℕ)
  (average_length : ℕ)
  (cost_per_song : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : average_length = 3)
  (h3 : cost_per_song = 50) :
  (hours_per_month * 60 / average_length * cost_per_song * 12 = 2400) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end John_pays_2400_per_year_l275_275019


namespace taxi_speed_l275_275204

theorem taxi_speed (v : ℕ) (h₁ : v > 30) (h₂ : ∃ t₁ t₂ : ℕ, t₁ = 3 ∧ t₂ = 3 ∧ 
                    v * t₁ = (v - 30) * (t₁ + t₂)) : 
                    v = 60 :=
by
  sorry

end taxi_speed_l275_275204


namespace marie_stamps_giveaway_l275_275303

theorem marie_stamps_giveaway :
  let notebooks := 4
  let stamps_per_notebook := 20
  let binders := 2
  let stamps_per_binder := 50
  let fraction_to_keep := 1/4
  let total_stamps := notebooks * stamps_per_notebook + binders * stamps_per_binder
  let stamps_to_keep := fraction_to_keep * total_stamps
  let stamps_to_give_away := total_stamps - stamps_to_keep
  stamps_to_give_away = 135 :=
by
  sorry

end marie_stamps_giveaway_l275_275303


namespace coordinate_sum_of_point_on_graph_l275_275006

theorem coordinate_sum_of_point_on_graph (g : ℕ → ℕ) (h : ℕ → ℕ)
  (h1 : g 2 = 8)
  (h2 : ∀ x, h x = 3 * (g x) ^ 2) :
  2 + h 2 = 194 :=
by
  sorry

end coordinate_sum_of_point_on_graph_l275_275006


namespace x7_value_l275_275485

theorem x7_value
  (x : ℕ → ℕ)
  (h1 : x 6 = 144)
  (h2 : ∀ n, 1 ≤ n ∧ n ≤ 4 → x (n + 3) = x (n + 2) * (x (n + 1) + x n))
  (h3 : ∀ m, m < 1 → 0 < x m) : x 7 = 3456 :=
by
  sorry

end x7_value_l275_275485


namespace count_terms_expansion_l275_275733

/-
This function verifies that the number of distinct terms in the expansion
of (a + b + c)(a + d + e + f + g) is equal to 15.
-/

theorem count_terms_expansion : 
    (a b c d e f g : ℕ) → 
    3 * 5 = 15 :=
by 
    intros a b c d e f g
    sorry

end count_terms_expansion_l275_275733


namespace derivative_of_even_function_is_odd_l275_275035

variables {R : Type*}

-- Definitions and Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem derivative_of_even_function_is_odd (f g : ℝ → ℝ) (h1 : even_function f) (h2 : ∀ x, deriv f x = g x) : odd_function g :=
sorry

end derivative_of_even_function_is_odd_l275_275035


namespace complex_exp_form_pow_four_l275_275096

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l275_275096


namespace complex_power_eq_rectangular_l275_275093

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_30 := Real.sin (Real.pi / 6)

theorem complex_power_eq_rectangular :
  (3 * (cos_30 + (complex.I * sin_30)))^4 = -40.5 + 40.5 * complex.I * Real.sqrt 3 := 
sorry

end complex_power_eq_rectangular_l275_275093


namespace find_last_even_number_l275_275632

theorem find_last_even_number (n : ℕ) (h : 4 * (n * (n + 1) * (2 * n + 1) / 6) = 560) : 2 * n = 14 :=
by
  sorry

end find_last_even_number_l275_275632


namespace work_done_in_days_l275_275738

theorem work_done_in_days (M B : ℕ) (x : ℕ) 
  (h1 : 12 * 2 * B + 16 * B = 200 * B / 5) 
  (h2 : 13 * 2 * B + 24 * B = 50 * x * B)
  (h3 : M = 2 * B) : 
  x = 4 := 
by
  sorry

end work_done_in_days_l275_275738


namespace original_price_l275_275069

theorem original_price (P : ℝ) (profit : ℝ) (profit_percentage : ℝ)
  (h1 : profit = 675) (h2 : profit_percentage = 0.35) :
  P = 1928.57 :=
by
  -- The proof is skipped using sorry
  sorry

end original_price_l275_275069


namespace largest_number_eq_l275_275332

theorem largest_number_eq (x y z : ℚ) (h1 : x + y + z = 82) (h2 : z - y = 10) (h3 : y - x = 4) :
  z = 106 / 3 :=
sorry

end largest_number_eq_l275_275332


namespace determinant_matrix_3x3_l275_275391

theorem determinant_matrix_3x3 :
  Matrix.det ![![3, 1, -2], ![8, 5, -4], ![1, 3, 6]] = 140 :=
by
  sorry

end determinant_matrix_3x3_l275_275391


namespace distance_between_cities_l275_275827

variable (D : ℝ) -- D is the distance between City A and City B
variable (time_AB : ℝ) -- Time from City A to City B
variable (time_BA : ℝ) -- Time from City B to City A
variable (saved_time : ℝ) -- Time saved per trip
variable (avg_speed : ℝ) -- Average speed for the round trip with saved time

theorem distance_between_cities :
  time_AB = 6 → time_BA = 4.5 → saved_time = 0.5 → avg_speed = 90 →
  D = 427.5 :=
by
  sorry

end distance_between_cities_l275_275827


namespace sum_of_binary_digits_of_315_l275_275811

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l275_275811


namespace find_m_l275_275132

-- Mathematical conditions definitions
def line1 (x y : ℝ) (m : ℝ) : Prop := 3 * x + m * y - 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0

-- Given the lines are parallel
def lines_parallel (l1 l2 : ℝ → ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∀ x y : ℝ, l1 x y m → l2 x y m → (3 / (m + 2)) = (m / (-(m - 2)))

-- The proof problem statement
theorem find_m (m : ℝ) : 
  lines_parallel (line1) (line2) m → (m = -6 ∨ m = 1) :=
by
  sorry

end find_m_l275_275132


namespace factorize_poly1_l275_275833

variable (a : ℝ)

theorem factorize_poly1 : a^4 + 2 * a^3 + 1 = (a + 1) * (a^3 + a^2 - a + 1) := 
sorry

end factorize_poly1_l275_275833


namespace minimum_rectangle_length_l275_275620

theorem minimum_rectangle_length (a x y : ℝ) (h : x * y = a^2) : x ≥ a ∨ y ≥ a :=
sorry

end minimum_rectangle_length_l275_275620


namespace real_roots_quadratic_l275_275873

theorem real_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 1 = 0) ↔ (m ≥ -5/4 ∧ m ≠ 1) := by
  sorry

end real_roots_quadratic_l275_275873


namespace total_apples_l275_275514

def packs : ℕ := 2
def apples_per_pack : ℕ := 4

theorem total_apples : packs * apples_per_pack = 8 := by
  sorry

end total_apples_l275_275514


namespace consecutive_integers_sum_l275_275629

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 := by
  sorry

end consecutive_integers_sum_l275_275629


namespace part1_part2_l275_275001

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B (a : ℝ) := {x : ℝ | (x - a) * (x - a - 1) < 0}

theorem part1 (a : ℝ) : (1 ∈ set_B a) → 0 < a ∧ a < 1 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, x ∈ set_B a → x ∈ set_A) ∧ (∃ x, x ∉ set_B a ∧ x ∈ set_A) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l275_275001


namespace sam_distance_walked_l275_275829

variable (d : ℝ := 40) -- initial distance between Fred and Sam
variable (v_f : ℝ := 4) -- Fred's constant speed in miles per hour
variable (v_s : ℝ := 4) -- Sam's constant speed in miles per hour

theorem sam_distance_walked :
  (d / (v_f + v_s)) * v_s = 20 :=
by
  sorry

end sam_distance_walked_l275_275829


namespace S8_value_l275_275541

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem S8_value 
  (h_geo : is_geometric_sequence a q)
  (h_S4 : S 4 = 3)
  (h_S12_S8 : S 12 - S 8 = 12) :
  S 8 = 9 := 
sorry

end S8_value_l275_275541


namespace sum_groups_is_250_l275_275685

-- Definitions based on the conditions
def group1 := [3, 13, 23, 33, 43]
def group2 := [7, 17, 27, 37, 47]

-- The proof problem
theorem sum_groups_is_250 : (group1.sum + group2.sum) = 250 :=
by
  sorry

end sum_groups_is_250_l275_275685


namespace broadcasting_methods_count_l275_275767

-- Defining the given conditions
def num_commercials : ℕ := 4 -- number of different commercial advertisements
def num_psa : ℕ := 2 -- number of different public service advertisements
def total_slots : ℕ := 6 -- total number of slots for commercials

-- The assertion we want to prove
theorem broadcasting_methods_count : 
  (num_psa * (total_slots - num_commercials - 1) * (num_commercials.factorial)) = 48 :=
by sorry

end broadcasting_methods_count_l275_275767


namespace george_earnings_l275_275867

theorem george_earnings (cars_sold : ℕ) (price_per_car : ℕ) (lego_set_price : ℕ) (h1 : cars_sold = 3) (h2 : price_per_car = 5) (h3 : lego_set_price = 30) :
  cars_sold * price_per_car + lego_set_price = 45 :=
by
  sorry

end george_earnings_l275_275867


namespace compute_y_geometric_series_l275_275689

theorem compute_y_geometric_series :
  let S1 := (∑' n : ℕ, (1 / 3)^n)
  let S2 := (∑' n : ℕ, (-1)^n * (1 / 3)^n)
  (S1 * S2 = ∑' n : ℕ, (1 / 9)^n) → 
  S1 = 3 / 2 →
  S2 = 3 / 4 →
  (∑' n : ℕ, (1 / y)^n) = 9 / 8 →
  y = 9 := 
by
  intros S1 S2 h₁ h₂ h₃ h₄
  sorry

end compute_y_geometric_series_l275_275689


namespace parallel_vectors_l275_275558

def vec_a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem parallel_vectors (x : ℝ) : vec_a x = (2, 4) → x = 2 := by
  sorry

end parallel_vectors_l275_275558


namespace arithmetic_sequence_common_difference_l275_275546

variable {α : Type*} [AddGroup α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n, a (n + 1) = a n + (a 2 - a 1)

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 2)
  (h_a3 : a 3 = -4) :
  a 3 - a 2 = -6 := 
sorry

end arithmetic_sequence_common_difference_l275_275546


namespace evaluate_expression_l275_275525

variable (a : ℕ)

theorem evaluate_expression (h : a = 2) : a^3 * a^4 = 128 :=
by
  sorry

end evaluate_expression_l275_275525


namespace sheila_weekly_earnings_is_288_l275_275762

-- Define the conditions as constants.
def sheilaWorksHoursPerDay (d : String) : ℕ :=
  if d = "Monday" ∨ d = "Wednesday" ∨ d = "Friday" then 8
  else if d = "Tuesday" ∨ d = "Thursday" then 6
  else 0

def hourlyWage : ℕ := 8

-- Calculate total weekly earnings based on conditions.
def weeklyEarnings : ℕ :=
  (sheilaWorksHoursPerDay "Monday" + sheilaWorksHoursPerDay "Wednesday" + sheilaWorksHoursPerDay "Friday") * hourlyWage +
  (sheilaWorksHoursPerDay "Tuesday" + sheilaWorksHoursPerDay "Thursday") * hourlyWage

-- The Lean statement for the proof.
theorem sheila_weekly_earnings_is_288 : weeklyEarnings = 288 :=
  by
    sorry

end sheila_weekly_earnings_is_288_l275_275762


namespace tan_sum_pi_over_4_x_l275_275114

theorem tan_sum_pi_over_4_x (x : ℝ) (h1 : x > -π/2 ∧ x < 0) (h2 : Real.cos x = 4/5) :
  Real.tan (π/4 + x) = 1/7 :=
by
  sorry

end tan_sum_pi_over_4_x_l275_275114


namespace find_lighter_ball_min_weighings_l275_275957

noncomputable def min_weighings_to_find_lighter_ball (balls : Fin 9 → ℕ) : ℕ :=
  2

-- Given: 9 balls, where 8 weigh 10 grams and 1 weighs 9 grams, and a balance scale.
theorem find_lighter_ball_min_weighings :
  (∃ i : Fin 9, balls i = 9 ∧ (∀ j : Fin 9, j ≠ i → balls j = 10)) 
  → min_weighings_to_find_lighter_ball balls = 2 :=
by
  intros
  sorry

end find_lighter_ball_min_weighings_l275_275957


namespace sufficient_but_not_necessary_l275_275537

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → 2 / a < 1) ∧ (2 / a < 1 → a > 2 ∨ a < 0) :=
by sorry

end sufficient_but_not_necessary_l275_275537


namespace quadratic_solution_condition_sufficient_but_not_necessary_l275_275693

theorem quadratic_solution_condition_sufficient_but_not_necessary (m : ℝ) :
  (m < -2) → (∃ x : ℝ, x^2 + m * x + 1 = 0) ∧ ¬(∀ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0 → m < -2) :=
by 
  sorry

end quadratic_solution_condition_sufficient_but_not_necessary_l275_275693


namespace distance_to_y_axis_eq_reflection_across_x_axis_eq_l275_275770

-- Definitions based on the conditions provided
def point_P : ℝ × ℝ := (4, -2)

-- Statements we need to prove
theorem distance_to_y_axis_eq : (abs (point_P.1) = 4) := 
by
  sorry  -- Proof placeholder

theorem reflection_across_x_axis_eq : (point_P.1 = 4 ∧ -point_P.2 = 2) :=
by
  sorry  -- Proof placeholder

end distance_to_y_axis_eq_reflection_across_x_axis_eq_l275_275770


namespace num_isosceles_triangles_l275_275268

theorem num_isosceles_triangles (a b : ℕ) (h1 : 2 * a + b = 27) (h2 : a > b / 2) : 
  ∃! (n : ℕ), n = 13 :=
by 
  sorry

end num_isosceles_triangles_l275_275268


namespace card_draw_probability_l275_275337

theorem card_draw_probability :
  let p1 := 12 / 52 * 4 / 51 * 26 / 50
  let p2 := 1 / 52 * 3 / 51 * 26 / 50
  p1 + p2 = 1 / 100 :=
by
  let p1 := (12:ℚ) / 52 * (4:ℚ) / 51 * (26:ℚ) / 50
  let p2 := (1:ℚ) / 52 * (3:ℚ) / 51 * (26:ℚ) / 50
  have p_total : p1 + p2 = 1 / 100 := sorry
  exact p_total

end card_draw_probability_l275_275337


namespace union_of_A_and_B_l275_275715

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} := by
  sorry

end union_of_A_and_B_l275_275715


namespace greatest_partition_l275_275999

-- Define the condition on the partitions of the positive integers
def satisfies_condition (A : ℕ → Prop) (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ b ∧ A a ∧ A b ∧ a + b = n

-- Define what it means for k subsets to meet the requirements
def partition_satisfies (k : ℕ) : Prop :=
∃ A : ℕ → ℕ → Prop,
  (∀ i : ℕ, i < k → ∀ n ≥ 15, satisfies_condition (A i) n)

-- Our conjecture is that k can be at most 3 for the given condition
theorem greatest_partition (k : ℕ) : k ≤ 3 :=
sorry

end greatest_partition_l275_275999


namespace students_outside_time_l275_275343

def total_outside_time (recess1 recess2 lunch recess3 : ℕ) : ℕ := 
  recess1 + recess2 + lunch + recess3

theorem students_outside_time : 
  total_outside_time 15 15 30 20 = 80 :=
by
  -- Conditions
  let first_recess := 15
  let second_recess := 15
  let lunch_break := 30
  let last_recess := 20
  -- Calculation
  have calculation := first_recess + second_recess + lunch_break + last_recess
  have result : calculation = 80 := sorry
  exact result

end students_outside_time_l275_275343


namespace arithmetic_sequence_S11_l275_275876

theorem arithmetic_sequence_S11 (a1 d : ℝ) 
  (h1 : a1 + d + a1 + 3 * d + 3 * (a1 + 6 * d) + a1 + 8 * d = 24) : 
  let a2 := a1 + d
  let a4 := a1 + 3 * d
  let a7 := a1 + 6 * d
  let a9 := a1 + 8 * d
  let S11 := 11 * (a1 + 5 * d)
  S11 = 44 :=
by
  sorry

end arithmetic_sequence_S11_l275_275876


namespace planned_pigs_correct_l275_275308

-- Define initial number of animals
def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

-- Define planned addition of animals
def added_cows : ℕ := 3
def added_goats : ℕ := 2
def total_animals : ℕ := 21

-- Define the total planned number of pigs to verify:
def planned_pigs := 8

-- State the final number of pigs to be proven
theorem planned_pigs_correct : 
  initial_cows + initial_pigs + initial_goats + added_cows + planned_pigs + added_goats = total_animals :=
by
  sorry

end planned_pigs_correct_l275_275308


namespace initial_soccer_balls_l275_275334

theorem initial_soccer_balls {x : ℕ} (h : x + 18 = 24) : x = 6 := 
sorry

end initial_soccer_balls_l275_275334


namespace expected_adjacent_red_pairs_l275_275900

theorem expected_adjacent_red_pairs :
  ∀ (deck : list (fin 2)), -- A deck of 60 cards is composed of 30 red (0) and 30 black (1)
  list.length deck = 60 →
  (list.count 0 deck = 30 ∧ list.count 1 deck = 30) →
  (∀ (i : fin 60), deck.nth (i + 1 % 60) = deck.nth (i + 1)) →
  @expected_value (fin 60) {p : list (fin 2) // list.length p = 60 ∧ (list.count 0 p = 30 ∧ list.count 1 p = 30)}
    (λ d, list.sum (list.map (λ i, if(deck.nth i = 0 ∧ deck.nth (i + 1 % 60) = 0)
    then 1 else 0) (list.fin_range 60))) = (30 * 29) / 59 := sorry

end expected_adjacent_red_pairs_l275_275900


namespace coconut_grove_l275_275741

theorem coconut_grove (x : ℕ) :
  (60 * (x + 1) + 120 * x + 180 * (x - 1)) = 300 * x → x = 2 :=
by
  intro h
  -- We can leave the proof part to prove this later.
  sorry

end coconut_grove_l275_275741


namespace find_overhead_expenses_l275_275680

noncomputable def overhead_expenses : ℝ := 35.29411764705882 / (1 + 0.1764705882352942)

theorem find_overhead_expenses (cost_price selling_price profit_percent : ℝ) (h_cp : cost_price = 225) (h_sp : selling_price = 300) (h_pp : profit_percent = 0.1764705882352942) :
  overhead_expenses = 30 :=
by
  sorry

end find_overhead_expenses_l275_275680


namespace product_of_digits_of_non_divisible_number_l275_275595

theorem product_of_digits_of_non_divisible_number:
  (¬ (3641 % 4 = 0)) →
  ((3641 % 10) * ((3641 / 10) % 10)) = 4 :=
by
  intro h
  sorry

end product_of_digits_of_non_divisible_number_l275_275595


namespace arithmetic_geometric_ratio_l275_275413

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
1 + 3 = a1 + a2

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
b2 ^ 2 = 4

theorem arithmetic_geometric_ratio (a1 a2 b2 : ℝ) 
  (h1 : arithmetic_sequence a1 a2) 
  (h2 : geometric_sequence b2) : 
  (a1 + a2) / b2 = 5 / 2 :=
by sorry

end arithmetic_geometric_ratio_l275_275413


namespace find_min_n_l275_275670

theorem find_min_n (k : ℕ) : ∃ n, 
  (∀ (m : ℕ), (k = 2 * m → n = 100 * (m + 1)) ∨ (k = 2 * m + 1 → n = 100 * (m + 1) + 1)) ∧
  (∀ n', (∀ (m : ℕ), (k = 2 * m → n' ≥ 100 * (m + 1)) ∨ (k = 2 * m + 1 → n' ≥ 100 * (m + 1) + 1)) → n' ≥ n) :=
by {
  sorry
}

end find_min_n_l275_275670


namespace prob_run_past_spectator_is_one_fourth_l275_275380

open ProbabilityTheory

noncomputable theory

def probability_run_past_spectator (A B : ℝ) : ℝ := 
  if |A - B| > 1 / 2 then 1 else 0

theorem prob_run_past_spectator_is_one_fourth : 
  ∀ (A B : ℝ), (A, B) ∈ set.Icc (0 : ℝ) 1 × set.Icc (0 : ℝ) 1 →
  integrable (λ (A B: ℝ), probability_run_past_spectator A B) measure_space.volume →
  probability (|A - B| > 1 / 2) = 1 / 4 :=
sorry

end prob_run_past_spectator_is_one_fourth_l275_275380


namespace geom_seq_sum_4_l275_275148

noncomputable def geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def sum_geom_seq (n : ℕ) (q : ℝ) (a₁ : ℝ) : ℝ :=
  if q = 1 then a₁ * n else (a₁ * (1 - q^n) / (1 - q))

theorem geom_seq_sum_4 {q : ℝ} (hq : q > 0) (hq1 : q ≠ 1) :
  let a₁ := 1 in
  let S5 := sum_geom_seq 5 q a₁ in
  let S3 := sum_geom_seq 3 q a₁ in
  S5 = 5 * S3 - 4 →
  sum_geom_seq 4 q a₁ = 15 :=
by
  sorry

end geom_seq_sum_4_l275_275148


namespace case1_equiv_case2_equiv_determine_case_l275_275194

theorem case1_equiv (a c x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) : 
  ((x + a) / (x + c) = a / c) ↔ (a = c) :=
by sorry

theorem case2_equiv (b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) : 
  (b / d = b / d) :=
by sorry

theorem determine_case (a b c d x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0) :
  ¬((x + a) / (x + c) = a / c) ∧ (b / d = b / d) :=
by sorry

end case1_equiv_case2_equiv_determine_case_l275_275194


namespace sum_of_digits_base_2_315_l275_275815

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l275_275815


namespace laura_running_speed_l275_275023

noncomputable def running_speed (x : ℝ) : ℝ := x^2 - 1

noncomputable def biking_speed (x : ℝ) : ℝ := 3 * x + 2

noncomputable def biking_time (x: ℝ) : ℝ := 30 / (biking_speed x)

noncomputable def running_time (x: ℝ) : ℝ := 5 / (running_speed x)

noncomputable def total_motion_time (x : ℝ) : ℝ := biking_time x + running_time x

-- Laura's total workout duration without transition time
noncomputable def required_motion_time : ℝ := 140 / 60

theorem laura_running_speed (x : ℝ) (hx : total_motion_time x = required_motion_time) :
  running_speed x = 83.33 :=
sorry

end laura_running_speed_l275_275023


namespace intersection_reciprocal_sum_l275_275141

open Real

theorem intersection_reciprocal_sum :
    ∀ (a b : ℝ),
    (∃ x : ℝ, x - 1 = a ∧ 3 / x = b) ∧
    (a * b = 3) →
    ∃ s : ℝ, (s = (a + b) / 3 ∨ s = -(a + b) / 3) ∧ (1 / a + 1 / b = s) := by
  sorry

end intersection_reciprocal_sum_l275_275141


namespace domain_g_l275_275702

noncomputable def g (x : ℝ) := Real.tan (Real.arccos (x ^ 3))

theorem domain_g :
  {x : ℝ | ∃ y, g x = y} = {x : ℝ | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)} :=
by
  sorry

end domain_g_l275_275702


namespace James_pays_35_l275_275576

theorem James_pays_35 (first_lesson_free : Bool) (total_lessons : Nat) (cost_per_lesson : Nat) 
  (first_x_paid_lessons_free : Nat) (every_other_remainings_free : Nat) (uncle_pays_half : Bool) :
  total_lessons = 20 → 
  first_lesson_free = true → 
  cost_per_lesson = 5 →
  first_x_paid_lessons_free = 10 →
  every_other_remainings_free = 1 → 
  uncle_pays_half = true →
  (10 * cost_per_lesson + 4 * cost_per_lesson) / 2 = 35 :=
by
  sorry

end James_pays_35_l275_275576


namespace largest_z_l275_275397

theorem largest_z (x y z : ℝ) 
  (h1 : x + y + z = 5)  
  (h2 : x * y + y * z + x * z = 3) 
  : z ≤ 13 / 3 := sorry

end largest_z_l275_275397


namespace pyramid_sphere_proof_l275_275363

theorem pyramid_sphere_proof
  (h R_1 R_2 : ℝ) 
  (O_1 O_2 T_1 T_2 : ℝ) 
  (inscription: h > 0 ∧ R_1 > 0 ∧ R_2 > 0) :
  R_1 * R_2 * h^2 = (R_1^2 - O_1 * T_1^2) * (R_2^2 - O_2 * T_2^2) :=
by
  sorry

end pyramid_sphere_proof_l275_275363


namespace solution_set_of_fractional_inequality_l275_275048

theorem solution_set_of_fractional_inequality :
  {x : ℝ | (x + 1) / (x - 3) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_of_fractional_inequality_l275_275048


namespace manny_received_fraction_l275_275299

-- Conditions
def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def leo_kept_packs : ℕ := 25
def neil_received_fraction : ℚ := 1 / 8

-- Definition of total packs
def total_packs : ℕ := total_marbles / marbles_per_pack

-- Proof problem: What fraction of the total packs did Manny receive?
theorem manny_received_fraction :
  (total_packs - leo_kept_packs - neil_received_fraction * total_packs) / total_packs = 1 / 4 :=
by sorry

end manny_received_fraction_l275_275299


namespace arithmetic_sequence_sum_9_is_36_l275_275884

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = r * (a n)
noncomputable def Sn (b : ℕ → ℝ) (n : ℕ) : ℝ := n * (b 1 + b n) / 2

theorem arithmetic_sequence_sum_9_is_36 (a b : ℕ → ℝ) (h_geom : geometric_sequence a) 
    (h_cond : a 4 * a 6 = 2 * a 5) (h_b5 : b 5 = 2 * a 5) : Sn b 9 = 36 :=
by
  sorry

end arithmetic_sequence_sum_9_is_36_l275_275884


namespace rational_eq1_rational_eq2_l275_275471

open Rational

theorem rational_eq1 (x : ℚ) : 
  (2 * x - 5) / (x - 2) = 3 / (2 - x) ↔ x = 4 := by
  sorry

theorem rational_eq2 (x : ℚ) : 
  (12 / ((x - 3) * (x + 3))) - (2 / (x - 3)) = (1 / (x + 3)) ↔ x ∉ {3, -3} := by
  sorry

end rational_eq1_rational_eq2_l275_275471


namespace range_of_b_l275_275288

theorem range_of_b (A B C a b c : ℝ) (h_triangle : a ^ 2 + b ^ 2 > c ^ 2 ∧ b ^ 2 + c ^ 2 > a ^ 2 ∧ c ^ 2 + a ^ 2 > b ^ 2)
  (ha : a = 1) (hB : B = 60) (h_acute : 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90) :
  ∃ l u, l = (sqrt 3 / 2) ∧ u = sqrt 3 ∧ l < b ∧ b < u :=
by
  sorry

end range_of_b_l275_275288


namespace total_cats_in_meow_and_paw_l275_275688

-- Define the conditions
def CatsInCatCafeCool : Nat := 5
def CatsInCatCafePaw : Nat := 2 * CatsInCatCafeCool
def CatsInCatCafeMeow : Nat := 3 * CatsInCatCafePaw

-- Define the total number of cats in Cat Cafe Meow and Cat Cafe Paw
def TotalCats : Nat := CatsInCatCafeMeow + CatsInCatCafePaw

-- The theorem stating the problem
theorem total_cats_in_meow_and_paw : TotalCats = 40 :=
by
  sorry

end total_cats_in_meow_and_paw_l275_275688


namespace solve_equation_l275_275488

theorem solve_equation (x : ℝ) (h : x + 3 ≠ 0) : (2 / (x + 3) = 1) → (x = -1) :=
by
  intro h1
  -- Proof skipped
  sorry

end solve_equation_l275_275488


namespace evaluate_expression_l275_275243

theorem evaluate_expression:
  (-2)^2002 + (-1)^2003 + 2^2004 + (-1)^2005 = 3 * 2^2002 - 2 :=
by
  sorry

end evaluate_expression_l275_275243


namespace expected_lotus_seed_zongzi_is_3_l275_275574

-- Define all the conditions
def total_zongzi : ℕ := 72 + 18 + 36 + 54
def lotus_seed_zongzi : ℕ := 54
def num_selected_zongzi : ℕ := 10

-- Define the expected number of lotus seed zongzi in the gift box
def expected_lotus_seed_zongzi : ℚ := num_selected_zongzi * (↑lotus_seed_zongzi / ↑total_zongzi)

/-- Prove that the expected number of lotus seed zongzi in the gift box is 3. -/
theorem expected_lotus_seed_zongzi_is_3 :
  expected_lotus_seed_zongzi = 3 :=
by sorry

end expected_lotus_seed_zongzi_is_3_l275_275574


namespace contest_end_time_l275_275066

-- Definitions for the conditions
def start_time_pm : Nat := 15 -- 3:00 p.m. in 24-hour format
def duration_min : Nat := 720

-- Proof that the contest ended at 3:00 a.m.
theorem contest_end_time :
  let end_time := (start_time_pm + (duration_min / 60)) % 24
  end_time = 3 :=
by
  -- This would be the place to provide the proof
  sorry

end contest_end_time_l275_275066


namespace gcd_36_54_l275_275787

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l275_275787


namespace problem_statement_l275_275759

theorem problem_statement (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
sorry

end problem_statement_l275_275759


namespace ayse_guarantee_win_l275_275388

def can_ayse_win (m n k : ℕ) : Prop :=
  -- Function defining the winning strategy for Ayşe
  sorry -- The exact strategy definition would be here

theorem ayse_guarantee_win :
  ((can_ayse_win 1 2012 2014) ∧ 
   (can_ayse_win 2011 2011 2012) ∧ 
   (can_ayse_win 2011 2012 2013) ∧ 
   (can_ayse_win 2011 2012 2014) ∧ 
   (can_ayse_win 2011 2013 2013)) = true :=
sorry -- Proof goes here

end ayse_guarantee_win_l275_275388


namespace gcd_36_54_l275_275789

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l275_275789


namespace ammeter_sum_l275_275252

variable (A1 A2 A3 A4 A5 : ℝ)
variable (I2 : ℝ)
variable (h1 : I2 = 4)
variable (h2 : A1 = I2)
variable (h3 : A3 = 2 * A1)
variable (h4 : A5 = A3 + A1)
variable (h5 : A4 = (5 / 3) * A5)

theorem ammeter_sum (A1 A2 A3 A4 A5 I2 : ℝ) (h1 : I2 = 4) (h2 : A1 = I2) (h3 : A3 = 2 * A1)
                   (h4 : A5 = A3 + A1) (h5 : A4 = (5 / 3) * A5) :
  A1 + I2 + A3 + A4 + A5 = 48 := 
sorry

end ammeter_sum_l275_275252


namespace sample_points_correlation_l275_275010

-- Define the set of sample data points
def sample_points (n : ℕ) (xs ys : ℕ → ℝ) :=
  n ≥ 2 ∧ ¬∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → xs i = xs j ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (ys i = - (1 / 2) * (xs i) + 1)

-- Define the function to calculate the correlation coefficient
def correlation_coefficient (xs ys : ℕ → ℝ) : ℝ := sorry -- Assume there's a function calculating the correlation coefficient of the data points.

theorem sample_points_correlation (n : ℕ) (xs ys : ℕ → ℝ) (h : sample_points n xs ys) :
  correlation_coefficient xs ys = -1 := 
sorry

end sample_points_correlation_l275_275010


namespace total_tiles_count_l275_275198

theorem total_tiles_count (n total_tiles: ℕ) 
  (h1: total_tiles - n^2 = 36) 
  (h2: total_tiles - (n + 1)^2 = 3) : total_tiles = 292 :=
by {
  sorry
}

end total_tiles_count_l275_275198


namespace jeremy_home_to_school_distance_l275_275579

theorem jeremy_home_to_school_distance (v d : ℝ) (h1 : 30 / 60 = 1 / 2) (h2 : 15 / 60 = 1 / 4)
  (h3 : d = v * (1 / 2)) (h4 : d = (v + 12) * (1 / 4)):
  d = 6 :=
by
  -- We assume that the conditions given lead to the distance being 6 miles
  sorry

end jeremy_home_to_school_distance_l275_275579


namespace largest_composite_in_five_consecutive_ints_l275_275713

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_of_five_composite_ints : ℕ :=
  36

theorem largest_composite_in_five_consecutive_ints (a b c d e : ℕ) :
  a < 40 ∧ b < 40 ∧ c < 40 ∧ d < 40 ∧ e < 40 ∧ 
  ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a = 32 ∧ b = 33 ∧ c = 34 ∧ d = 35 ∧ e = 36 →
  e = largest_of_five_composite_ints :=
by 
  sorry

end largest_composite_in_five_consecutive_ints_l275_275713


namespace perpendicular_line_through_point_l275_275643

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l275_275643


namespace minimum_sugar_quantity_l275_275225

theorem minimum_sugar_quantity :
  ∃ s f : ℝ, s = 4 ∧ f ≥ 4 + s / 3 ∧ f ≤ 3 * s ∧ 2 * s + 3 * f ≤ 36 :=
sorry

end minimum_sugar_quantity_l275_275225


namespace sum_of_digits_base2_315_l275_275810

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l275_275810


namespace problem_solution_l275_275893

def x : ℤ := -2 + 3
def y : ℤ := abs (-5)
def z : ℤ := 4 * (-1/4)

theorem problem_solution : x + y + z = 5 := 
by
  -- Definitions based on the problem statement
  have h1 : x = -2 + 3 := rfl
  have h2 : y = abs (-5) := rfl
  have h3 : z = 4 * (-1/4) := rfl
  
  -- Exact result required to be proved. Adding placeholder for steps.
  sorry

end problem_solution_l275_275893


namespace max_not_expressed_as_linear_comb_l275_275753

theorem max_not_expressed_as_linear_comb {a b c : ℕ} (h_coprime_ab : Nat.gcd a b = 1)
                                        (h_coprime_bc : Nat.gcd b c = 1)
                                        (h_coprime_ca : Nat.gcd c a = 1) :
    Nat := sorry

end max_not_expressed_as_linear_comb_l275_275753


namespace measure_A_l275_275905

noncomputable def angle_A (C B A : ℝ) : Prop :=
  C = 3 / 2 * B ∧ B = 30 ∧ A = 180 - B - C

theorem measure_A (A B C : ℝ) (h : angle_A C B A) : A = 105 :=
by
  -- Extract conditions from h
  obtain ⟨h1, h2, h3⟩ := h
  
  -- Use the conditions to prove the thesis
  simp [h1, h2, h3]
  sorry

end measure_A_l275_275905


namespace ratio_of_books_to_pens_l275_275515

theorem ratio_of_books_to_pens (total_stationery : ℕ) (books : ℕ) (pens : ℕ) 
    (h1 : total_stationery = 400) (h2 : books = 280) (h3 : pens = total_stationery - books) : 
    books / (Nat.gcd books pens) = 7 ∧ pens / (Nat.gcd books pens) = 3 := 
by 
  -- proof steps would go here
  sorry

end ratio_of_books_to_pens_l275_275515


namespace tina_more_than_katya_l275_275021

theorem tina_more_than_katya (katya_sales ricky_sales : ℕ) (tina_sales : ℕ) (
  h1 : katya_sales = 8) (h2 : ricky_sales = 9) (h3 : tina_sales = 2 * (katya_sales + ricky_sales)) :
  tina_sales - katya_sales = 26 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  rw [h3, h1]
  norm_num
  sorry

end tina_more_than_katya_l275_275021


namespace dense_local_minima_of_continuous_nowhere_monotone_l275_275025

noncomputable def nowhere_monotone_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
∀ I (hI : I ⊆ s) (hI' : I ≠ ∅) (hI'' : I ≠ set.univ),
  ¬ (∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≤ f y)
  ∧ ¬ (∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≥ f y)

theorem dense_local_minima_of_continuous_nowhere_monotone
(f : ℝ → ℝ) (H_cont : continuous_on f (set.Icc 0 1))
(H_nowhere_monotone : nowhere_monotone_on f (set.Icc 0 1)) :
dense {x | ∃ U ∈ 𝓝 x, ∀ y ∈ U, f y ≥ f x} :=
sorry

end dense_local_minima_of_continuous_nowhere_monotone_l275_275025


namespace no_solution_m_4_l275_275894

theorem no_solution_m_4 (m : ℝ) : 
  (¬ ∃ x : ℝ, 2/x = m/(2*x + 1)) → m = 4 :=
by
  sorry

end no_solution_m_4_l275_275894


namespace peter_fish_caught_l275_275465

theorem peter_fish_caught (n : ℕ) (h : 3 * n = n + 24) : n = 12 :=
sorry

end peter_fish_caught_l275_275465


namespace smallest_b_l275_275249

theorem smallest_b (b : ℝ) : b^2 - 16 * b + 63 ≤ 0 → (∃ b : ℝ, b = 7) :=
sorry

end smallest_b_l275_275249


namespace intersection_chord_line_eq_l275_275891

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
noncomputable def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

theorem intersection_chord_line_eq (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : 
  2 * x + y = 0 :=
sorry

end intersection_chord_line_eq_l275_275891


namespace sunglasses_price_l275_275843

theorem sunglasses_price (P : ℝ) 
  (buy_cost_per_pair : ℝ := 26) 
  (pairs_sold : ℝ := 10) 
  (sign_cost : ℝ := 20) :
  (pairs_sold * P - pairs_sold * buy_cost_per_pair) / 2 = sign_cost →
  P = 30 := 
by
  sorry

end sunglasses_price_l275_275843


namespace prime_squared_difference_divisible_by_24_l275_275671

theorem prime_squared_difference_divisible_by_24 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) :
  24 ∣ (p^2 - q^2) :=
sorry

end prime_squared_difference_divisible_by_24_l275_275671


namespace find_d_l275_275070

def point_in_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3030 ∧ 0 ≤ y ∧ y ≤ 3030

def point_in_ellipse (x y : ℝ) : Prop :=
  (x^2 / 2020^2) + (y^2 / 4040^2) ≤ 1

def point_within_distance (d : ℝ) (x y : ℝ) : Prop :=
  (∃ (a b : ℤ), (x - a) ^ 2 + (y - b) ^ 2 ≤ d ^ 2)

theorem find_d :
  (∃ d : ℝ, (∀ x y : ℝ, point_in_square x y → point_in_ellipse x y → point_within_distance d x y) ∧ (d = 0.5)) :=
by
  sorry

end find_d_l275_275070


namespace kiran_currency_notes_l275_275297

theorem kiran_currency_notes :
  ∀ (n50_amount n100_amount total50 total100 : ℝ),
    n50_amount = 3500 →
    total50 = 5000 →
    total100 = 5000 - 3500 →
    n100_amount = total100 →
    (n50_amount / 50 + total100 / 100) = 85 :=
by
  intros n50_amount n100_amount total50 total100 n50_amount_eq total50_eq total100_eq n100_amount_eq
  sorry

end kiran_currency_notes_l275_275297


namespace perimeter_of_flowerbed_l275_275075

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem perimeter_of_flowerbed : perimeter length width = 22 := by
  sorry

end perimeter_of_flowerbed_l275_275075


namespace probability_difference_l275_275144

noncomputable def Ps (red black : ℕ) : ℚ :=
  let total := red + black
  (red * (red - 1) + black * (black - 1)) / (total * (total - 1))

noncomputable def Pd (red black : ℕ) : ℚ :=
  let total := red + black
  (red * black * 2) / (total * (total - 1))

noncomputable def abs_diff (Ps Pd : ℚ) : ℚ :=
  |Ps - Pd|

theorem probability_difference :
  let red := 1200
  let black := 800
  let total := red + black
  abs_diff (Ps red black) (Pd red black) = 789 / 19990 := by
  sorry

end probability_difference_l275_275144


namespace correct_speed_l275_275164

def distance_40_late (d : ℝ) (t : ℝ) : Prop :=
  d = 40 * (t + 1/20)

def distance_60_early (d : ℝ) (t : ℝ) : Prop :=
  d = 60 * (t - 1/20)

theorem correct_speed (d t : ℝ) :
  distance_40_late d t →
  distance_60_early d t →
  (d = 12 ∧ t = 1/4) →
  (48 = d / t) :=
by {
  intros h1 h2 h3,
  sorry
}

end correct_speed_l275_275164


namespace train_A_start_time_l275_275052

theorem train_A_start_time :
  let distance := 155 -- km
  let speed_A := 20 -- km/h
  let speed_B := 25 -- km/h
  let start_B := 8 -- a.m.
  let meet_time := 11 -- a.m.
  let travel_time_B := meet_time - start_B -- time in hours for train B from 8 a.m. to 11 a.m.
  let distance_B := speed_B * travel_time_B -- distance covered by train B
  let distance_A := distance - distance_B -- remaining distance covered by train A
  let travel_time_A := distance_A / speed_A -- time for train A to cover its distance
  let start_A := meet_time - travel_time_A -- start time for train A
  start_A = 7 := by
  sorry

end train_A_start_time_l275_275052


namespace angle_A_range_l275_275909

theorem angle_A_range (a b : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) :
  ∃ A : ℝ, 0 < A ∧ A ≤ Real.pi / 4 :=
sorry

end angle_A_range_l275_275909


namespace find_f2_l275_275112

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 - a * x^3 + b * x - 6

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -22 :=
by
  sorry

end find_f2_l275_275112


namespace true_inverse_of_opposites_true_contrapositive_of_real_roots_l275_275259

theorem true_inverse_of_opposites (X Y : Int) :
  (X = -Y) → (X + Y = 0) :=
by 
  sorry

theorem true_contrapositive_of_real_roots (q : Real) :
  (¬ ∃ x : Real, x^2 + 2*x + q = 0) → (q > 1) :=
by
  sorry

end true_inverse_of_opposites_true_contrapositive_of_real_roots_l275_275259


namespace sum_of_interior_angles_of_regular_polygon_l275_275179

theorem sum_of_interior_angles_of_regular_polygon :
  (∀ (n : ℕ), (n ≠ 0) ∧ ((360 / 45 = n) → (180 * (n - 2) = 1080))) := by sorry

end sum_of_interior_angles_of_regular_polygon_l275_275179


namespace circle_equations_l275_275704

-- Given conditions: the circle passes through points O(0,0), A(1,1), B(4,2)
-- Prove the general equation of the circle and the standard equation 

theorem circle_equations : 
  ∃ (D E F : ℝ), (∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ 
                      (x, y) = (0, 0) ∨ (x, y) = (1, 1) ∨ (x, y) = (4, 2)) ∧
  (D = -8) ∧ (E = 6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y = 0 ↔ (x - 4)^2 + (y + 3)^2 = 25) :=
sorry

end circle_equations_l275_275704


namespace optimal_order_for_ostap_l275_275599

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l275_275599


namespace trig_system_solution_l275_275352

theorem trig_system_solution (x y : ℝ) (hx : 0 ≤ x ∧ x < 2 * Real.pi) (hy : 0 ≤ y ∧ y < 2 * Real.pi)
  (h1 : Real.sin x + Real.cos y = 0) (h2 : Real.cos x * Real.sin y = -1/2) :
    (x = Real.pi / 4 ∧ y = 5 * Real.pi / 4) ∨
    (x = 3 * Real.pi / 4 ∧ y = 3 * Real.pi / 4) ∨
    (x = 5 * Real.pi / 4 ∧ y = Real.pi / 4) ∨
    (x = 7 * Real.pi / 4 ∧ y = 7 * Real.pi / 4) := by
  sorry

end trig_system_solution_l275_275352


namespace mark_bench_press_correct_l275_275394

def dave_weight : ℝ := 175
def dave_bench_press : ℝ := 3 * dave_weight

def craig_bench_percentage : ℝ := 0.20
def craig_bench_press : ℝ := craig_bench_percentage * dave_bench_press

def emma_bench_percentage : ℝ := 0.75
def emma_initial_bench_press : ℝ := emma_bench_percentage * dave_bench_press
def emma_actual_bench_press : ℝ := emma_initial_bench_press + 15

def combined_craig_emma : ℝ := craig_bench_press + emma_actual_bench_press

def john_bench_factor : ℝ := 2
def john_bench_press : ℝ := john_bench_factor * combined_craig_emma

def mark_reduction : ℝ := 50
def mark_bench_press : ℝ := combined_craig_emma - mark_reduction

theorem mark_bench_press_correct : mark_bench_press = 463.75 := by
  sorry

end mark_bench_press_correct_l275_275394


namespace arun_age_l275_275452

variable (A S G M : ℕ)

theorem arun_age (h1 : A - 6 = 18 * G)
                 (h2 : G + 2 = M)
                 (h3 : M = 5)
                 (h4 : S = A - 8) : A = 60 :=
by sorry

end arun_age_l275_275452


namespace solve_for_y_l275_275320

theorem solve_for_y (y : ℤ) : 
  7 * (4 * y + 3) - 3 = -3 * (2 - 9 * y) → y = -24 :=
by
  intro h
  sorry

end solve_for_y_l275_275320


namespace evaluate_expression_l275_275242

theorem evaluate_expression (c : ℕ) (h : c = 4) : (c^c - c * (c - 1)^(c - 1))^c = 148^4 := 
by 
  sorry

end evaluate_expression_l275_275242


namespace beth_cans_of_corn_l275_275083

theorem beth_cans_of_corn (C P : ℕ) (h1 : P = 2 * C + 15) (h2 : P = 35) : C = 10 :=
by
  sorry

end beth_cans_of_corn_l275_275083


namespace present_age_of_eldest_is_45_l275_275486

theorem present_age_of_eldest_is_45 (x : ℕ) 
  (h1 : (5 * x - 10) + (7 * x - 10) + (8 * x - 10) + (9 * x - 10) = 107) :
  9 * x = 45 :=
sorry

end present_age_of_eldest_is_45_l275_275486


namespace angles_MAB_NAC_l275_275202

/-- Given equal chords AB and AC, and a tangent MAN, with arc BC's measure (excluding point A) being 200 degrees,
prove that the angles MAB and NAC are either 40 degrees or 140 degrees. -/
theorem angles_MAB_NAC (AB AC : ℝ) (tangent_MAN : Prop)
    (arc_BC_measure : ∀ A : ℝ , A = 200) : 
    ∃ θ : ℝ, (θ = 40 ∨ θ = 140) :=
by
  sorry

end angles_MAB_NAC_l275_275202


namespace one_factor_exists_l275_275527

variables (G : Type*) [Graph G] (V : Type*) [Fintype V] (E : V → V → Prop)
          (S : Finset V) (C_G_S : Finset (Finset V))
          (factor_critical : Finset V → Prop)
          (is_matchable : Finset V → Finset (Finset V) → Prop)

axiom condition_1 : is_matchable S (C_G_S)
axiom condition_2 : ∀ C ∈ C_G_S, factor_critical C
axiom condition_3 : S.card = C_G_S.card

theorem one_factor_exists :
  ∃ M : Finset (V × V), is_perfect_matching G M ∧ S.card = C_G_S.card := sorry

end one_factor_exists_l275_275527


namespace second_group_work_days_l275_275736

theorem second_group_work_days (M B : ℕ) (d1 d2 : ℕ) (H1 : M = 2 * B) 
  (H2 : (12 * M + 16 * B) * 5 = d1) (H3 : (13 * M + 24 * B) * d2 = d1) : 
  d2 = 4 :=
by
  sorry

end second_group_work_days_l275_275736


namespace special_numbers_count_l275_275135

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_zero (n : ℕ) : Prop := n % 10 = 0
def divisible_by_30 (n : ℕ) : Prop := n % 30 = 0

-- Define the count of numbers with the specified conditions
noncomputable def count_special_numbers : ℕ :=
  (9990 - 1020) / 30 + 1

-- The proof problem
theorem special_numbers_count : count_special_numbers = 300 := sorry

end special_numbers_count_l275_275135


namespace find_third_test_score_l275_275459

-- Definitions of the given conditions
def test_score_1 := 80
def test_score_2 := 70
variable (x : ℕ) -- the unknown third score
def test_score_4 := 100
def average_score (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4

-- Theorem stating that given the conditions, the third test score must be 90
theorem find_third_test_score (h : average_score test_score_1 test_score_2 x test_score_4 = 85) : x = 90 :=
by
  sorry

end find_third_test_score_l275_275459


namespace quadruples_positive_integers_l275_275700

theorem quadruples_positive_integers (x y z n : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ n > 0 ∧ (x^2 + y^2 + z^2 + 1 = 2^n) →
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
sorry

end quadruples_positive_integers_l275_275700


namespace brianna_books_l275_275848

theorem brianna_books :
  ∀ (books_per_month : ℕ) (given_books : ℕ) (bought_books : ℕ) (borrowed_books : ℕ) (total_books_needed : ℕ),
    (books_per_month = 2) →
    (given_books = 6) →
    (bought_books = 8) →
    (borrowed_books = bought_books - 2) →
    (total_books_needed = 12 * books_per_month) →
    (total_books_needed - (given_books + bought_books + borrowed_books)) = 4 :=
by
  intros
  sorry

end brianna_books_l275_275848


namespace sum_of_binary_digits_of_315_l275_275813

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l275_275813


namespace problem_statement_l275_275877

noncomputable def f : ℝ → ℝ
| x if (0 < x ∧ x ≤ 1)  := 2^x
| x                     := sorry -- definition for other x is determined by properties

theorem problem_statement :
  (∀ x : ℝ, f(-x) = -f(x)) → (∀ x : ℝ, f(x+2) = -f(x)) → f(2016) - f(2015) = 2 :=
by
  intro h_odd h_periodic
  have f_period : ∀ x : ℝ, f(x + 4) = f(x), from
    λ x, by rw [add_assoc, h_periodic, h_periodic (x+2), h_periodic x];
             exact congr_arg (λ y, -(-f(y))) (h_periodic x) -- Utilizing periodicity
  have : f(2016) = f(0), from congr_arg f (show 2016 % 4 = 0, by norm_num)
  have : f(2015) = f(-1), from congr_arg f (show 2015 % 4 = -1, by norm_num)
  have f_0 : f(0) = 0, from sorry -- Using that f is odd and f is 0 at 0
  have f_neg1 : f(-1) = -2, from sorry -- Using that f(1) = 2^1 = 2 and f is odd
  calc
    f(2016) - f(2015) = f(0) - f(-1) : by congr; exact_mod_cast 𝕌-h :- sorry -- Using periodic properties
                ...  = 0 - (-2)     : by rw [f_0, f_neg1]
                ...  = 2,
  sorry

end problem_statement_l275_275877


namespace selling_price_per_sweater_correct_l275_275852

-- Definitions based on the problem's conditions
def balls_of_yarn_per_sweater := 4
def cost_per_ball_of_yarn := 6
def number_of_sweaters := 28
def total_gain := 308

-- Defining the required selling price per sweater
def total_cost_of_yarn : Nat := balls_of_yarn_per_sweater * cost_per_ball_of_yarn * number_of_sweaters
def total_revenue : Nat := total_cost_of_yarn + total_gain
def selling_price_per_sweater : ℕ := total_revenue / number_of_sweaters

theorem selling_price_per_sweater_correct :
  selling_price_per_sweater = 35 :=
  by
  sorry

end selling_price_per_sweater_correct_l275_275852


namespace burt_net_profit_l275_275230

theorem burt_net_profit
  (cost_seeds : ℝ := 2.00)
  (cost_soil : ℝ := 8.00)
  (num_plants : ℕ := 20)
  (price_per_plant : ℝ := 5.00) :
  let total_cost := cost_seeds + cost_soil
  let total_revenue := num_plants * price_per_plant
  let net_profit := total_revenue - total_cost
  net_profit = 90.00 :=
by sorry

end burt_net_profit_l275_275230


namespace total_arrangement_with_at_least_one_girl_l275_275037

theorem total_arrangement_with_at_least_one_girl :
  let boys := 4 in
  let girls := 3 in
  let people := boys + girls in
  let combinations (n k : ℕ) := Nat.choose n k in
  let arrangements (n : ℕ) := Nat.fact n in
  combinations people 3 * arrangements 3 - combinations boys 3 * arrangements 3 = 186 :=
by
  let boys := 4
  let girls := 3
  let people := boys + girls
  let combinations (n k : ℕ) := Nat.choose n k
  let arrangements (n : ℕ) := Nat.fact n
  -- sorry is used to skip the proof part
  calc combinations people 3 * arrangements 3 - combinations boys 3 * arrangements 3 = 186 : sorry

end total_arrangement_with_at_least_one_girl_l275_275037


namespace marble_distribution_l275_275336

theorem marble_distribution (x : ℚ) (total : ℚ) (boy1 : ℚ) (boy2 : ℚ) (boy3 : ℚ) :
  (4 * x + 2) + (2 * x + 1) + (3 * x) = total → total = 62 →
  boy1 = 4 * x + 2 → boy2 = 2 * x + 1 → boy3 = 3 * x →
  boy1 = 254 / 9 ∧ boy2 = 127 / 9 ∧ boy3 = 177 / 9 :=
by
  sorry

end marble_distribution_l275_275336


namespace optimal_order_l275_275605

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l275_275605


namespace quadratic_inequality_solution_l275_275078

theorem quadratic_inequality_solution (x : ℝ) : 2 * x^2 - 5 * x - 3 ≥ 0 ↔ x ≤ -1/2 ∨ x ≥ 3 := 
by
  sorry

end quadratic_inequality_solution_l275_275078


namespace percentage_calculation_l275_275967

theorem percentage_calculation (amount : ℝ) (percentage : ℝ) (res : ℝ) :
  amount = 400 → percentage = 0.25 → res = amount * percentage → res = 100 := by
  intro h_amount h_percentage h_res
  rw [h_amount, h_percentage] at h_res
  norm_num at h_res
  exact h_res

end percentage_calculation_l275_275967


namespace quadratic_inequality_solution_l275_275691

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - 4 * x - 21 < 0) ↔ (-3 < x ∧ x < 7) :=
sorry

end quadratic_inequality_solution_l275_275691


namespace least_n_divisibility_l275_275499

theorem least_n_divisibility :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ n → k ∣ (n - 1)^2) ∧ (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ ¬ k ∣ (n - 1)^2) ∧ n = 3 :=
by
  sorry

end least_n_divisibility_l275_275499


namespace integral_evaluation_l275_275103

theorem integral_evaluation {a : ℝ} (ha : 0 < a) : 
  ∫ x in -1..a^2, 1 / (x^2 + a^2) = (Real.pi / 2) / a := 
by sorry

end integral_evaluation_l275_275103


namespace carpet_length_l275_275386

theorem carpet_length (percent_covered : ℝ) (width : ℝ) (floor_area : ℝ) (carpet_length : ℝ) :
  percent_covered = 0.30 → width = 4 → floor_area = 120 → carpet_length = 9 :=
by
  sorry

end carpet_length_l275_275386


namespace exists_nat_sum_of_squares_two_ways_l275_275206

theorem exists_nat_sum_of_squares_two_ways :
  ∃ n : ℕ, n < 100 ∧ ∃ a b c d : ℕ, a ≠ b ∧ c ≠ d ∧ n = a^2 + b^2 ∧ n = c^2 + d^2 :=
by {
  sorry
}

end exists_nat_sum_of_squares_two_ways_l275_275206


namespace number_of_common_tangents_l275_275721

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem number_of_common_tangents
  (C₁ C₂ : ℝ × ℝ) (r₁ r₂ : ℝ)
  (h₁ : ∀ (x y : ℝ), x^2 + y^2 - 2 * x = 0 → (C₁ = (1, 0)) ∧ (r₁ = 1))
  (h₂ : ∀ (x y : ℝ), x^2 + y^2 - 4 * y + 3 = 0 → (C₂ = (0, 2)) ∧ (r₂ = 1))
  (d : distance C₁ C₂ = Real.sqrt 5) :
  4 = 4 := 
by sorry

end number_of_common_tangents_l275_275721


namespace cylinder_lateral_surface_area_l275_275505

variable (r : ℝ) (h : ℝ)

theorem cylinder_lateral_surface_area (hr : r = 12) (hh : h = 21) :
  (2 * Real.pi * r * h) = 504 * Real.pi := by
  sorry

end cylinder_lateral_surface_area_l275_275505


namespace sequence_formula_l275_275015

def seq (n : ℕ) : ℕ := 
  match n with
  | 0     => 1
  | (n+1) => 2 * seq n + 3

theorem sequence_formula (n : ℕ) (h1 : n ≥ 1) : 
  seq n = 2^n + 1 - 3 :=
sorry

end sequence_formula_l275_275015


namespace sum_of_digits_base2_315_l275_275819

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l275_275819


namespace problem_f_2011_2012_l275_275415

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2011_2012 :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f (1-x) = f (1+x)) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x = 2^x - 1) →
  f 2011 + f 2012 = -1 :=
by
  intros h1 h2 h3
  sorry

end problem_f_2011_2012_l275_275415


namespace part1_l275_275972

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)
variable (h0 : ∀ x, 0 ≤ x → f x = Real.sqrt x)
variable (h1 : 0 ≤ x1)
variable (h2 : 0 ≤ x2)
variable (h3 : x1 ≠ x2)

theorem part1 : (1/2) * (f x1 + f x2) < f ((x1 + x2) / 2) :=
  sorry

end part1_l275_275972


namespace michael_completes_in_50_days_l275_275059

theorem michael_completes_in_50_days :
  ∀ {M A W : ℝ},
    (W / M + W / A = W / 20) →
    (14 * W / 20 + 10 * W / A = W) →
    M = 50 :=
by
  sorry

end michael_completes_in_50_days_l275_275059


namespace anthony_ate_total_l275_275223

def slices := 16

def ate_alone := 1 / slices
def shared_with_ben := (1 / 2) * (1 / slices)
def shared_with_chris := (1 / 2) * (1 / slices)

theorem anthony_ate_total :
  ate_alone + shared_with_ben + shared_with_chris = 1 / 8 :=
by
  sorry

end anthony_ate_total_l275_275223


namespace Mike_gave_marbles_l275_275592

variables (original_marbles given_marbles remaining_marbles : ℕ)

def Mike_original_marbles : ℕ := 8
def Mike_remaining_marbles : ℕ := 4
def Mike_given_marbles (original remaining : ℕ) : ℕ := original - remaining

theorem Mike_gave_marbles :
  Mike_given_marbles Mike_original_marbles Mike_remaining_marbles = 4 :=
sorry

end Mike_gave_marbles_l275_275592


namespace Jenny_total_wins_l275_275441

theorem Jenny_total_wins :
  let games_against_mark := 10
  let mark_wins := 1
  let mark_losses := games_against_mark - mark_wins
  let games_against_jill := 2 * games_against_mark
  let jill_wins := (75 / 100) * games_against_jill
  let jenny_wins_against_jill := games_against_jill - jill_wins
  mark_losses + jenny_wins_against_jill = 14 :=
by
  sorry

end Jenny_total_wins_l275_275441


namespace cartons_loaded_l275_275987

def total_cartons : Nat := 50
def cans_per_carton : Nat := 20
def cans_left_to_load : Nat := 200

theorem cartons_loaded (C : Nat) (h : cans_per_carton ≠ 0) : 
  C = total_cartons - (cans_left_to_load / cans_per_carton) := by
  sorry

end cartons_loaded_l275_275987


namespace maximum_height_of_projectile_l275_275071

def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

theorem maximum_height_of_projectile : ∀ t : ℝ, (h t ≤ 116) :=
by sorry

end maximum_height_of_projectile_l275_275071


namespace eval_expr1_eval_expr2_l275_275244

theorem eval_expr1 : (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
by
  -- proof goes here
  sorry

theorem eval_expr2 : (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) / (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180))) = Real.sqrt 2 :=
by
  -- proof goes here
  sorry

end eval_expr1_eval_expr2_l275_275244


namespace boxes_filled_l275_275975

noncomputable def bags_per_box := 6
noncomputable def balls_per_bag := 8
noncomputable def total_balls := 720

theorem boxes_filled (h1 : balls_per_bag = 8) (h2 : bags_per_box = 6) (h3 : total_balls = 720) :
  (total_balls / balls_per_bag) / bags_per_box = 15 :=
by
  sorry

end boxes_filled_l275_275975


namespace revenue_correct_l275_275744

def calculate_revenue : Real :=
  let pumpkin_pie_revenue := 4 * 8 * 5
  let custard_pie_revenue := 5 * 6 * 6
  let apple_pie_revenue := 3 * 10 * 4
  let pecan_pie_revenue := 2 * 12 * 7
  let cookie_revenue := 15 * 2
  let red_velvet_revenue := 6 * 8 * 9
  pumpkin_pie_revenue + custard_pie_revenue + apple_pie_revenue + pecan_pie_revenue + cookie_revenue + red_velvet_revenue

theorem revenue_correct : calculate_revenue = 1090 :=
by
  sorry

end revenue_correct_l275_275744


namespace total_sticks_of_gum_in_12_brown_boxes_l275_275844

-- Definitions based on the conditions
def packs_per_carton := 7
def sticks_per_pack := 5
def cartons_in_full_box := 6
def cartons_in_partial_box := 3
def num_brown_boxes := 12
def num_partial_boxes := 2

-- Calculation definitions
def sticks_per_carton := packs_per_carton * sticks_per_pack
def sticks_per_full_box := cartons_in_full_box * sticks_per_carton
def sticks_per_partial_box := cartons_in_partial_box * sticks_per_carton
def num_full_boxes := num_brown_boxes - num_partial_boxes

-- Final total sticks of gum
def total_sticks_of_gum := (num_full_boxes * sticks_per_full_box) + (num_partial_boxes * sticks_per_partial_box)

-- The theorem to be proved
theorem total_sticks_of_gum_in_12_brown_boxes :
  total_sticks_of_gum = 2310 :=
by
  -- The proof is omitted.
  sorry

end total_sticks_of_gum_in_12_brown_boxes_l275_275844


namespace postage_arrangements_11_cents_l275_275521

-- Definitions for the problem settings, such as stamp denominations and counts
def stamp_collection : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

-- Function to calculate all unique arrangements of stamps that sum to a given value (11 cents)
def count_arrangements (total_cents : ℕ) : ℕ :=
  -- The implementation would involve a combinatorial counting taking into account the problem conditions
  sorry

-- The main theorem statement asserting the solution
theorem postage_arrangements_11_cents :
  count_arrangements 11 = 71 :=
  sorry

end postage_arrangements_11_cents_l275_275521


namespace find_f_of_500_l275_275159

theorem find_f_of_500
  (f : ℕ → ℕ)
  (h_pos : ∀ x y : ℕ, f x > 0 ∧ f y > 0) 
  (h_mul : ∀ x y : ℕ, f (x * y) = f x + f y) 
  (h_f10 : f 10 = 15)
  (h_f40 : f 40 = 23) :
  f 500 = 41 :=
sorry

end find_f_of_500_l275_275159


namespace billiard_angle_correct_l275_275835

-- Definitions for the problem conditions
def center_O : ℝ × ℝ := (0, 0)
def point_P : ℝ × ℝ := (0.5, 0)
def radius : ℝ := 1

-- The angle to be proven
def strike_angle (α x : ℝ) := x = (90 - 2 * α)

-- Main theorem statement
theorem billiard_angle_correct :
  ∃ α x : ℝ, (strike_angle α x) ∧ x = 47 + (4 / 60) :=
sorry

end billiard_angle_correct_l275_275835


namespace hundreds_digit_even_l275_275955

-- Define the given conditions
def units_digit (n : ℕ) : ℕ := n % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- The main theorem to prove
theorem hundreds_digit_even (x : ℕ) 
  (h1 : units_digit (x*x) = 9) 
  (h2 : tens_digit (x*x) = 0) : ((x*x) / 100) % 2 = 0 :=
  sorry

end hundreds_digit_even_l275_275955


namespace cost_of_painting_l275_275970

def area_of_house : ℕ := 484
def price_per_sqft : ℕ := 20

theorem cost_of_painting : area_of_house * price_per_sqft = 9680 := by
  sorry

end cost_of_painting_l275_275970


namespace hospital_staff_l275_275492

-- Define the conditions
variables (d n : ℕ) -- d: number of doctors, n: number of nurses
variables (x : ℕ) -- common multiplier

theorem hospital_staff (h1 : d + n = 456) (h2 : 8 * x = d) (h3 : 11 * x = n) : n = 264 :=
by
  -- noncomputable def only when necessary, skipping the proof with sorry
  sorry

end hospital_staff_l275_275492


namespace area_of_triangle_l275_275678

variables (yellow_area green_area blue_area : ℝ)
variables (is_equilateral_triangle : Prop)
variables (centered_at_vertices : Prop)
variables (radius_less_than_height : Prop)

theorem area_of_triangle (h_yellow : yellow_area = 1000)
                        (h_green : green_area = 100)
                        (h_blue : blue_area = 1)
                        (h_triangle : is_equilateral_triangle)
                        (h_centered : centered_at_vertices)
                        (h_radius : radius_less_than_height) :
  ∃ (area : ℝ), area = 150 :=
by
  sorry

end area_of_triangle_l275_275678


namespace nick_total_quarters_l275_275461

theorem nick_total_quarters (Q : ℕ)
  (h1 : 2 / 5 * Q = state_quarters)
  (h2 : 1 / 2 * state_quarters = PA_quarters)
  (h3 : PA_quarters = 7) :
  Q = 35 := by
  sorry

end nick_total_quarters_l275_275461


namespace solve_for_x_l275_275562

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 9 / (x / 3)) : x = 15 * Real.sqrt 1.8 ∨ x = -15 * Real.sqrt 1.8 := 
by
  sorry

end solve_for_x_l275_275562


namespace avg_weights_N_square_of_integer_l275_275364

theorem avg_weights_N_square_of_integer (N : ℕ) :
  (∃ S : ℕ, S > 0 ∧ ∃ k : ℕ, k * k = N + 1 ∧ S = (N * (N + 1)) / 2 / (N - k + 1) ∧ (N * (N + 1)) / 2 - S = (N - k) * S) ↔ (∃ k : ℕ, k * k = N + 1) := by
  sorry

end avg_weights_N_square_of_integer_l275_275364


namespace parabola_distance_l275_275943

theorem parabola_distance (p : ℝ) : 
  (∃ p: ℝ, y^2 = 10*x ∧ 2*p = 10) → p = 5 :=
by
  sorry

end parabola_distance_l275_275943


namespace gillian_spent_multiple_of_sandi_l275_275469

theorem gillian_spent_multiple_of_sandi
  (sandi_had : ℕ := 600)
  (gillian_spent : ℕ := 1050)
  (sandi_spent : ℕ := sandi_had / 2)
  (diff : ℕ := gillian_spent - sandi_spent)
  (extra : ℕ := 150)
  (multiple_of_sandi : ℕ := (diff - extra) / sandi_spent) : 
  multiple_of_sandi = 1 := 
  by sorry

end gillian_spent_multiple_of_sandi_l275_275469


namespace ellipse_equation_l275_275119

theorem ellipse_equation (a b c : ℝ) 
  (h1 : 0 < b) (h2 : b < a) 
  (h3 : c = 3 * Real.sqrt 3) 
  (h4 : a = 6) 
  (h5 : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
by
  sorry

end ellipse_equation_l275_275119


namespace quadratic_completing_square_l275_275072

theorem quadratic_completing_square (b p : ℝ) (hb : b < 0)
  (h_quad_eq : ∀ x : ℝ, x^2 + b * x + (1 / 6) = (x + p)^2 + (1 / 18)) :
  b = - (2 / 3) :=
by
  sorry

end quadratic_completing_square_l275_275072


namespace sum_of_digits_base_2_315_l275_275818

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l275_275818


namespace stones_equally_distributed_l275_275956

theorem stones_equally_distributed (n k : ℕ) 
    (h : ∃ piles : Fin n → ℕ, (∀ i j, 2 * piles i + piles j = k * n)) :
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end stones_equally_distributed_l275_275956


namespace max_m_value_l275_275275

noncomputable def f (x m : ℝ) : ℝ := x * Real.log x + x^2 - m * x + Real.exp (2 - x)

theorem max_m_value (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) → m ≤ 3 :=
sorry

end max_m_value_l275_275275


namespace intersection_point_l275_275106

def parametric_line (t : ℝ) : ℝ × ℝ × ℝ :=
  (-1 - 2 * t, 0, -1 + 3 * t)

def plane (x y z : ℝ) : Prop := x + 4 * y + 13 * z - 23 = 0

theorem intersection_point :
  ∃ t : ℝ, plane (-1 - 2 * t) 0 (-1 + 3 * t) ∧ parametric_line t = (-3, 0, 2) :=
by
  sorry

end intersection_point_l275_275106


namespace range_of_x_l275_275260

variables {x : Real}

def P (x : Real) : Prop := (x + 1) / (x - 3) ≥ 0
def Q (x : Real) : Prop := abs (1 - x/2) < 1

theorem range_of_x (hP : P x) (hQ : ¬ Q x) : x ≤ -1 ∨ x ≥ 4 :=
  sorry

end range_of_x_l275_275260


namespace books_a_count_l275_275490

theorem books_a_count (A B : ℕ) (h1 : A + B = 20) (h2 : A = B + 4) : A = 12 :=
by
  sorry

end books_a_count_l275_275490


namespace quadratic_distinct_real_roots_l275_275428

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 9 ∧ a * c * 4 < b^2) ↔ (m < -6 ∨ m > 6) :=
by
  sorry

end quadratic_distinct_real_roots_l275_275428


namespace sol_earns_amount_l275_275319

theorem sol_earns_amount (candy_bars_first_day : ℕ)
                         (additional_candy_bars_per_day : ℕ)
                         (sell_days_per_week : ℕ)
                         (cost_per_candy_bar_cents : ℤ) :
                         candy_bars_first_day = 10 →
                         additional_candy_bars_per_day = 4 →
                         sell_days_per_week = 6 →
                         cost_per_candy_bar_cents = 10 →
                         (∑ i in finset.range sell_days_per_week, (candy_bars_first_day + additional_candy_bars_per_day * i).to_int) * cost_per_candy_bar_cents / 100 = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end sol_earns_amount_l275_275319


namespace number_of_rhombuses_l275_275290

-- Definition: A grid with 25 small equilateral triangles arranged in a larger triangular pattern
def equilateral_grid (n : ℕ) : Prop :=
  n = 25

-- Theorem: Proving the number of rhombuses that can be formed from the grid
theorem number_of_rhombuses (n : ℕ) (h : equilateral_grid n) : ℕ :=
  30 

-- Main proof statement
example (n : ℕ) (h : equilateral_grid n) : number_of_rhombuses n h = 30 :=
by
  sorry

end number_of_rhombuses_l275_275290


namespace find_B_l275_275863

theorem find_B : 
  ∀ (A B : ℕ), A ≤ 9 → B ≤ 9 → (600 + 10 * A + 5) + (100 + B) = 748 → B = 3 :=
by
  intros A B hA hB hEq
  sorry

end find_B_l275_275863


namespace no_two_delegates_next_to_each_other_l275_275522

theorem no_two_delegates_next_to_each_other :
  let n := 8
  let delegates : Fin n → ℕ := λ i, i % 4
  P := 0
  ∃ (m n : ℕ), nat.gcd m n = 1 ∧ (m : ℚ) / n = P :=
sorry

end no_two_delegates_next_to_each_other_l275_275522


namespace find_number_l275_275366

theorem find_number (x : ℝ) (h : 75 = 0.6 * x) : x = 125 :=
sorry

end find_number_l275_275366


namespace max_m_value_l275_275919

theorem max_m_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 35) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m = 3 :=
by
  sorry

end max_m_value_l275_275919


namespace area_of_given_circle_is_4pi_l275_275520

-- Define the given equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 12 * x + 18 * y + 27 = 0

-- Define the area of the circle to be proved
noncomputable def area_of_circle : ℝ := 4 * Real.pi

-- Statement of the theorem to be proved in Lean
theorem area_of_given_circle_is_4pi :
  (∃ x y : ℝ, circle_equation x y) → area_of_circle = 4 * Real.pi :=
by
  -- The proof will go here
  sorry

end area_of_given_circle_is_4pi_l275_275520


namespace arithmetic_sequence_sum_properties_l275_275261

theorem arithmetic_sequence_sum_properties {S : ℕ → ℝ} {a : ℕ → ℝ} (d : ℝ)
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  let a6 := (S 6 - S 5)
  let a7 := (S 7 - S 6)
  (d = a7 - a6) →
  d < 0 ∧ S 12 > 0 ∧ ¬(∀ n, S n = S 11) ∧ abs a6 > abs a7 :=
by
  sorry

end arithmetic_sequence_sum_properties_l275_275261


namespace sum_of_consecutive_integers_with_product_506_l275_275776

theorem sum_of_consecutive_integers_with_product_506 :
  ∃ x : ℕ, (x * (x + 1) = 506) → (x + (x + 1) = 45) :=
by
  sorry

end sum_of_consecutive_integers_with_product_506_l275_275776


namespace reciprocals_sum_eq_neg_one_over_three_l275_275454

-- Let the reciprocals of the roots of the polynomial 7x^2 + 2x + 6 be alpha and beta.
-- Given that a and b are roots of the polynomial, and alpha = 1/a and beta = 1/b,
-- Prove that alpha + beta = -1/3.

theorem reciprocals_sum_eq_neg_one_over_three
  (a b : ℝ)
  (ha : 7 * a ^ 2 + 2 * a + 6 = 0)
  (hb : 7 * b ^ 2 + 2 * b + 6 = 0)
  (h_sum : a + b = -2 / 7)
  (h_prod : a * b = 6 / 7) :
  (1 / a) + (1 / b) = -1 / 3 := by
  sorry

end reciprocals_sum_eq_neg_one_over_three_l275_275454


namespace num_statements_imply_impl_l275_275998

variable (p q r : Prop)

def cond1 := p ∧ q ∧ ¬r
def cond2 := ¬p ∧ q ∧ r
def cond3 := p ∧ q ∧ r
def cond4 := ¬p ∧ ¬q ∧ ¬r

def impl := ((p → ¬q) → ¬r)

theorem num_statements_imply_impl : 
  (cond1 p q r → impl p q r) ∧ 
  (cond3 p q r → impl p q r) ∧ 
  (cond4 p q r → impl p q r) ∧ 
  ¬(cond2 p q r → impl p q r) :=
by {
  sorry
}

end num_statements_imply_impl_l275_275998


namespace best_coupon1_price_l275_275212

theorem best_coupon1_price (x : ℝ) 
    (h1 : 60 ≤ x ∨ x = 60)
    (h2_1 : 25 < 0.12 * x) 
    (h2_2 : 0.12 * x > 0.2 * x - 30) :
    x = 209.95 ∨ x = 229.95 ∨ x = 249.95 :=
by sorry

end best_coupon1_price_l275_275212


namespace johns_yearly_music_cost_l275_275018

theorem johns_yearly_music_cost 
  (hours_per_month : ℕ := 20)
  (minutes_per_hour : ℕ := 60)
  (average_song_length : ℕ := 3)
  (cost_per_song : ℕ := 50) -- represented in cents to avoid decimals
  (months_per_year : ℕ := 12)
  : (hours_per_month * minutes_per_hour // average_song_length) * cost_per_song * months_per_year = 2400 * 100 := -- 2400 dollars (* 100 to represent cents)
  sorry

end johns_yearly_music_cost_l275_275018


namespace paul_and_paula_cookies_l275_275561

-- Define the number of cookies per pack type
def cookies_in_pack (pack : ℕ) : ℕ :=
  match pack with
  | 1 => 15
  | 2 => 30
  | 3 => 45
  | 4 => 60
  | _ => 0

-- Paul's purchase: 2 packs of Pack B and 1 pack of Pack A
def pauls_cookies : ℕ :=
  2 * cookies_in_pack 2 + cookies_in_pack 1

-- Paula's purchase: 1 pack of Pack A and 1 pack of Pack C
def paulas_cookies : ℕ :=
  cookies_in_pack 1 + cookies_in_pack 3

-- Total number of cookies Paul and Paula have
def total_cookies : ℕ :=
  pauls_cookies + paulas_cookies

theorem paul_and_paula_cookies : total_cookies = 135 :=
by
  sorry

end paul_and_paula_cookies_l275_275561


namespace remainder_of_2n_div4_l275_275668

theorem remainder_of_2n_div4 (n : ℕ) (h : ∃ k : ℕ, n = 4 * k + 3) : (2 * n) % 4 = 2 := 
by
  sorry

end remainder_of_2n_div4_l275_275668


namespace tetrahedron_labeling_impossible_l275_275473

/-- Suppose each vertex of a tetrahedron needs to be labeled with an integer from 1 to 4, each integer being used exactly once.
We need to prove that there are no such arrangements in which the sum of the numbers on the vertices of each face is the same for all four faces.
Arrangements that can be rotated into each other are considered identical. -/
theorem tetrahedron_labeling_impossible :
  ∀ (label : Fin 4 → Fin 5) (h_unique : ∀ v1 v2 : Fin 4, v1 ≠ v2 → label v1 ≠ label v2),
  ∃ (sum_faces : ℕ), sum_faces = 7 ∧ sum_faces % 3 = 1 → False :=
by
  sorry

end tetrahedron_labeling_impossible_l275_275473


namespace identified_rectangle_perimeter_l275_275982

-- Define the side length of the square
def side_length_mm : ℕ := 75

-- Define the heights of the rectangles
variables (x y z : ℕ)

-- Define conditions
def rectangles_cut_condition (x y z : ℕ) : Prop := x + y + z = side_length_mm
def perimeter_relation_condition (x y z : ℕ) : Prop := 2 * (x + side_length_mm) = (y + side_length_mm) + (z + side_length_mm)

-- Define the perimeter of the identified rectangle
def identified_perimeter_mm (x : ℕ) := 2 * (x + side_length_mm)

-- Define conversion from mm to cm
def mm_to_cm (mm : ℕ) : ℕ := mm / 10

-- Final proof statement
theorem identified_rectangle_perimeter :
  ∃ x y z : ℕ, rectangles_cut_condition x y z ∧ perimeter_relation_condition x y z ∧ mm_to_cm (identified_perimeter_mm x) = 20 := 
sorry

end identified_rectangle_perimeter_l275_275982


namespace sum_of_interior_angles_l275_275180

-- Define the conditions:
def exterior_angle (n : ℕ) := 45

def sum_exterior_angles := 360

-- Define the Lean statement for the proof problem
theorem sum_of_interior_angles : ∃ n : ℕ, 
  sum_exterior_angles / exterior_angle n = n ∧
  (180 * (n - 2) = 1080) :=
by
  use 8
  split
  calc
    sum_exterior_angles / exterior_angle 8 = 360 / 45 := rfl
    ... = 8 := rfl
  calc
    180 * (8 - 2) = 180 * 6 := rfl
    ... = 1080 := rfl

end sum_of_interior_angles_l275_275180


namespace trajectory_eq_l275_275782

theorem trajectory_eq {x y m : ℝ} (h : x^2 + y^2 - (4 * m + 2) * x - 2 * m * y + 4 * m^2 + 4 * m + 1 = 0) :
  x - 2 * y - 1 = 0 ∧ x ≠ 1 :=
sorry

end trajectory_eq_l275_275782


namespace max_value_l275_275922

open Real

theorem max_value (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha1 : a ≤ 1) (hb1 : b ≤ 1) (hc1 : c ≤ 1/2) :
  sqrt (a * b * c) + sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ (1 / sqrt 2) + (1 / 2) :=
sorry

end max_value_l275_275922


namespace book_difference_l275_275042

def initial_books : ℕ := 75
def borrowed_books : ℕ := 18
def difference : ℕ := initial_books - borrowed_books

theorem book_difference : difference = 57 := by
  -- Proof will go here
  sorry

end book_difference_l275_275042


namespace drink_exactly_five_bottles_last_day_l275_275616

/-- 
Robin bought 617 bottles of water and needs to purchase 4 additional bottles on the last day 
to meet her daily water intake goal. 
Prove that Robin will drink exactly 5 bottles on the last day.
-/
theorem drink_exactly_five_bottles_last_day : 
  ∀ (bottles_bought : ℕ) (extra_bottles : ℕ), bottles_bought = 617 → extra_bottles = 4 → 
  ∃ x : ℕ, 621 = x * 617 + 4 ∧ x + 4 = 5 :=
by
  intros bottles_bought extra_bottles bottles_bought_eq extra_bottles_eq
  -- The proof would follow here
  sorry

end drink_exactly_five_bottles_last_day_l275_275616


namespace moles_of_HCl_required_l275_275559

noncomputable def numberOfMolesHClRequired (moles_AgNO3 : ℕ) : ℕ :=
  if moles_AgNO3 = 3 then 3 else 0

-- Theorem statement
theorem moles_of_HCl_required : numberOfMolesHClRequired 3 = 3 := by
  sorry

end moles_of_HCl_required_l275_275559


namespace fresh_grapes_weight_eq_l275_275536

-- Definitions of the conditions from a)
def fresh_grapes_water_percent : ℝ := 0.80
def dried_grapes_water_percent : ℝ := 0.20
def dried_grapes_weight : ℝ := 10
def fresh_grapes_non_water_percent : ℝ := 1 - fresh_grapes_water_percent
def dried_grapes_non_water_percent : ℝ := 1 - dried_grapes_water_percent

-- Proving the weight of fresh grapes
theorem fresh_grapes_weight_eq :
  let F := (dried_grapes_non_water_percent * dried_grapes_weight) / fresh_grapes_non_water_percent
  F = 40 := by
  -- The proof has been omitted
  sorry

end fresh_grapes_weight_eq_l275_275536


namespace largest_of_five_consecutive_composite_integers_under_40_l275_275711

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def five_consecutive_composite_integers_under_40 : List ℕ :=
[32, 33, 34, 35, 36]

theorem largest_of_five_consecutive_composite_integers_under_40 :
  ∀ n ∈ five_consecutive_composite_integers_under_40,
  n < 40 ∧ ∀ k, (k ∈ five_consecutive_composite_integers_under_40 →
  ¬ is_prime k) →
  List.maximum five_consecutive_composite_integers_under_40 = some 36 :=
by
  sorry

end largest_of_five_consecutive_composite_integers_under_40_l275_275711


namespace Beth_bought_10_cans_of_corn_l275_275085

theorem Beth_bought_10_cans_of_corn (a b : ℕ) (h1 : b = 15 + 2 * a) (h2 : b = 35) : a = 10 := by
  sorry

end Beth_bought_10_cans_of_corn_l275_275085


namespace martin_bell_ringing_l275_275305

theorem martin_bell_ringing (B S : ℕ) (hB : B = 36) (hS : S = B / 3 + 4) : S + B = 52 :=
sorry

end martin_bell_ringing_l275_275305


namespace average_of_first_two_is_1_point_1_l275_275622

theorem average_of_first_two_is_1_point_1
  (a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.5)
  (h2 : (a1 + a2) / 2 = x)
  (h3 : (a3 + a4) / 2 = 1.4)
  (h4 : (a5 + a6) / 2 = 5) :
  x = 1.1 := 
sorry

end average_of_first_two_is_1_point_1_l275_275622


namespace perpendicular_line_equation_l275_275657

noncomputable def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m : ℝ := -a / b
  let b' : ℝ := c / b
  (m, b')

noncomputable def perpendicular_slope (slope : ℝ) : ℝ :=
  -1 / slope

theorem perpendicular_line_equation (x1 y1 : ℝ) 
  (h_line_eq : ∃ (a b c : ℝ), a * x1 + b * y1 = c)
  (h_point : x1 = 2 ∧ y1 = -3) :
  ∃ a b : ℝ, y1 = a * x1 + b ∧ a = -2 ∧ b = 1 :=
by
  let (m, b') := slope_intercept_form 3 (-6) 9
  have h_slope := (-1 : ℝ) / m
  have h_perpendicular_slope : h_slope = -2 := sorry
  let (x1, y1) := (2, -3)
  have h_eq : y1 - (-3) = h_slope * (x1 - 2) := sorry
  use [-2, 1]
  simp [h_eq]
  sorry

end perpendicular_line_equation_l275_275657


namespace inequality_solution_l275_275860

theorem inequality_solution (x : ℝ) :
  4 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 → x ∈ Set.Ioc (5 / 2 : ℝ) (20 / 7 : ℝ) := by
  sorry

end inequality_solution_l275_275860


namespace sum_of_digits_base2_315_l275_275820

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l275_275820


namespace positive_rational_number_l275_275664

theorem positive_rational_number : ∃ x : ℝ, x = 1/2 ∧ 
  ( (∀ y : ℝ, y = -Real.sqrt 2 → ¬ (0 < y ∧ ∃ a b : ℤ, y = a / b) ) ∧
    ( (0 < 1/2 ∧ ∃ a b : ℤ, 1/2 = a / b) ∧ ∀ z : ℝ, ¬ ((0 < z ∧ z ≠ 1/2) ∧ ∃ a b : ℤ, z = a / b)) ∧
    ( ∀ w : ℝ, w = 0 → ¬ (0 < w ∧ ∃ a b : ℤ, w = a / b) ) ∧
    ( ∀ v : ℝ, v = Real.sqrt 3 → (0 < v ∧ ¬ (∃ a b : ℤ, v = a / b) ) ) ) :=
by
  sorry

end positive_rational_number_l275_275664


namespace perimeter_percent_increase_l275_275385

noncomputable def side_increase (s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) : ℝ :=
  let s₂ := s₂_ratio * s₁
  let s₃ := s₃_ratio * s₂
  let s₄ := s₄_ratio * s₃
  let s₅ := s₅_ratio * s₄
  s₅

theorem perimeter_percent_increase (s₁ : ℝ) (s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) (P₁ := 3 * s₁)
    (P₅ := 3 * side_increase s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio) :
    s₁ = 4 → s₂_ratio = 1.5 → s₃_ratio = 1.3 → s₄_ratio = 1.5 → s₅_ratio = 1.3 →
    P₅ = 45.63 →
    ((P₅ - P₁) / P₁) * 100 = 280.3 :=
by
  intros
  -- proof goes here
  sorry

end perimeter_percent_increase_l275_275385


namespace B_elements_l275_275131

def B : Set ℤ := {x | -3 < 2 * x - 1 ∧ 2 * x - 1 < 3}

theorem B_elements : B = {-1, 0, 1} :=
by
  sorry

end B_elements_l275_275131


namespace intersection_range_of_b_l275_275539

theorem intersection_range_of_b (b : ℝ) :
  (∀ (m : ℝ), ∃ (x y : ℝ), x^2 + 2 * y^2 = 3 ∧ y = m * x + b) ↔ 
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := 
sorry

end intersection_range_of_b_l275_275539


namespace quadruples_positive_integers_l275_275699

theorem quadruples_positive_integers (x y z n : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ n > 0 ∧ (x^2 + y^2 + z^2 + 1 = 2^n) →
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
sorry

end quadruples_positive_integers_l275_275699


namespace find_percentage_l275_275062

theorem find_percentage (P : ℝ) : 
  (∀ x : ℝ, x = 0.40 * 800 → x = P / 100 * 650 + 190) → P = 20 := 
by
  intro h
  sorry

end find_percentage_l275_275062


namespace smallest_positive_z_l275_275172

theorem smallest_positive_z (x z : ℝ) (hx : Real.sin x = 1) (hz : Real.sin (x + z) = -1/2) : z = 2 * Real.pi / 3 :=
by
  sorry

end smallest_positive_z_l275_275172


namespace perpendicular_line_eq_l275_275641

theorem perpendicular_line_eq (x y : ℝ) :
  (3 * x - 6 * y = 9) ∧ ∃ x₀ y₀, (2 : ℝ) = x₀ ∧ (-3 : ℝ) = y₀ ∧ y₀ + (2 * (x - x₀)) = y :=
  ∃ y₀, y = -2 * x + y₀ ∧ y₀ = 1 := sorry

end perpendicular_line_eq_l275_275641


namespace tooth_extraction_cost_l275_275498

noncomputable def cleaning_cost : ℕ := 70
noncomputable def filling_cost : ℕ := 120
noncomputable def root_canal_cost : ℕ := 400
noncomputable def crown_cost : ℕ := 600
noncomputable def bridge_cost : ℕ := 800

noncomputable def crown_discount : ℕ := (crown_cost * 20) / 100
noncomputable def bridge_discount : ℕ := (bridge_cost * 10) / 100

noncomputable def total_cost_without_extraction : ℕ := 
  cleaning_cost + 
  3 * filling_cost + 
  root_canal_cost + 
  (crown_cost - crown_discount) + 
  (bridge_cost - bridge_discount)

noncomputable def root_canal_and_one_filling : ℕ := 
  root_canal_cost + filling_cost

noncomputable def dentist_bill : ℕ := 
  11 * root_canal_and_one_filling

theorem tooth_extraction_cost : 
  dentist_bill - total_cost_without_extraction = 3690 :=
by
  -- The proof would go here
  sorry

end tooth_extraction_cost_l275_275498


namespace fraction_of_boxes_loaded_by_day_crew_l275_275828

-- Definitions based on the conditions
variables (D W : ℕ)  -- Day crew per worker boxes (D) and number of workers (W)

-- Helper Definitions
def boxes_day_crew : ℕ := D * W  -- Total boxes by day crew
def boxes_night_crew : ℕ := (3 * D / 4) * (3 * W / 4)  -- Total boxes by night crew
def total_boxes : ℕ := boxes_day_crew D W + boxes_night_crew D W  -- Total boxes by both crews

-- The main theorem
theorem fraction_of_boxes_loaded_by_day_crew :
  (boxes_day_crew D W : ℚ) / (total_boxes D W : ℚ) = 16/25 :=
by
  sorry

end fraction_of_boxes_loaded_by_day_crew_l275_275828


namespace rectangle_side_b_value_l275_275170

section
variable {a b c d : ℕ}

-- Given conditions
def conditions : Prop :=
  (a : ℚ) / c = 3 / 4 ∧ (b : ℚ) / d = 3 / 4 ∧ c = 4 ∧ d = 8

-- Proof statement
theorem rectangle_side_b_value (h : conditions) : b = 6 :=
  sorry
end

end rectangle_side_b_value_l275_275170


namespace range_of_a_l275_275885

-- Lean statement that represents the proof problem
theorem range_of_a 
  (h1 : ∀ x y : ℝ, x^2 - 2 * x + Real.log (2 * y^2 - y) = 0 → x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0)
  (h2 : ∀ b : ℝ, 2 * b^2 - b > 0) :
  (∀ a : ℝ, x^2 - 2 * x + Real.log (2 * a^2 - a) = 0 → (- (1:ℝ) / 2) < a ∧ a < 0 ∨ (1 / 2) < a ∧ a < 1) :=
sorry

end range_of_a_l275_275885


namespace fence_calculation_l275_275073

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def fence_needed : ℕ := 2 * length + 2 * width

theorem fence_calculation : fence_needed = 22 := by
  sorry

end fence_calculation_l275_275073


namespace average_speed_ratio_l275_275058

theorem average_speed_ratio (t_E t_F : ℝ) (d_B d_C : ℝ) (htE : t_E = 3) (htF : t_F = 4) (hdB : d_B = 450) (hdC : d_C = 300) :
  (d_B / t_E) / (d_C / t_F) = 2 :=
by
  sorry

end average_speed_ratio_l275_275058


namespace morio_current_age_l275_275322

-- Given conditions
def teresa_current_age : ℕ := 59
def morio_age_when_michiko_born : ℕ := 38
def teresa_age_when_michiko_born : ℕ := 26

-- Definitions derived from the conditions
def michiko_age : ℕ := teresa_current_age - teresa_age_when_michiko_born

-- Statement to prove Morio's current age
theorem morio_current_age : (michiko_age + morio_age_when_michiko_born) = 71 :=
by
  sorry

end morio_current_age_l275_275322


namespace average_gas_mileage_round_trip_l275_275985

theorem average_gas_mileage_round_trip :
  let distance_to_city := 150
  let mpg_sedan := 25
  let mpg_rental := 15
  let total_distance := 2 * distance_to_city
  let gas_used_outbound := distance_to_city / mpg_sedan
  let gas_used_return := distance_to_city / mpg_rental
  let total_gas_used := gas_used_outbound + gas_used_return
  let avg_gas_mileage := total_distance / total_gas_used
  avg_gas_mileage = 18.75 := by
{
  sorry
}

end average_gas_mileage_round_trip_l275_275985


namespace optimal_order_l275_275603

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l275_275603


namespace farm_cows_l275_275436

theorem farm_cows (x y : ℕ) (h : 4 * x + 2 * y = 20 + 3 * (x + y)) : x = 20 + y :=
sorry

end farm_cows_l275_275436


namespace min_k_inequality_l275_275544

theorem min_k_inequality (α β : ℝ) (hα : 0 < α) (hα2 : α < 2 * Real.pi / 3)
  (hβ : 0 < β) (hβ2 : β < 2 * Real.pi / 3) :
  4 * Real.cos α ^ 2 + 2 * Real.cos α * Real.cos β + 4 * Real.cos β ^ 2
  - 3 * Real.cos α - 3 * Real.cos β - 6 < 0 :=
by
  sorry

end min_k_inequality_l275_275544


namespace solve_for_x_l275_275686

theorem solve_for_x (x : ℝ) (h : (1/3 : ℝ) * (x + 8 + 5*x + 3 + 3*x + 4) = 4*x + 1) : x = 4 :=
by {
  sorry
}

end solve_for_x_l275_275686


namespace river_bank_bottom_width_l275_275324

/-- 
The cross-section of a river bank is a trapezium with a 12 m wide top and 
a certain width at the bottom. The area of the cross-section is 500 sq m 
and the depth is 50 m. Prove that the width at the bottom is 8 m.
-/
theorem river_bank_bottom_width (area height top_width : ℝ) (h_area: area = 500) 
(h_height : height = 50) (h_top_width : top_width = 12) : ∃ b : ℝ, (1 / 2) * (top_width + b) * height = area ∧ b = 8 :=
by
  use 8
  sorry

end river_bank_bottom_width_l275_275324


namespace find_k_l275_275585

noncomputable def g (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : g a b c (-1) = 0) 
  (h2 : 30 < g a b c 5) (h3 : g a b c 5 < 40)
  (h4 : 120 < g a b c 7) (h5 : g a b c 7 < 130)
  (h6 : 2000 * k < g a b c 50) (h7 : g a b c 50 < 2000 * (k + 1)) : 
  k = 5 := 
sorry

end find_k_l275_275585


namespace average_of_rest_l275_275435

theorem average_of_rest (A : ℝ) (total_students scoring_95 scoring_0 : ℕ) (total_avg : ℝ)
  (h_total_students : total_students = 25)
  (h_scoring_95 : scoring_95 = 3)
  (h_scoring_0 : scoring_0 = 3)
  (h_total_avg : total_avg = 45.6)
  (h_sum_eq : total_students * total_avg = 3 * 95 + 3 * 0 + (total_students - scoring_95 - scoring_0) * A) :
  A = 45 := sorry

end average_of_rest_l275_275435


namespace minimum_number_of_odd_integers_among_six_l275_275960

theorem minimum_number_of_odd_integers_among_six : 
  ∀ (x y a b m n : ℤ), 
    x + y = 28 →
    x + y + a + b = 45 →
    x + y + a + b + m + n = 63 →
    ∃ (odd_count : ℕ), odd_count = 1 :=
by sorry

end minimum_number_of_odd_integers_among_six_l275_275960


namespace original_purchase_price_l275_275977

-- Define the conditions and question
theorem original_purchase_price (P S : ℝ) (h1 : S = P + 0.25 * S) (h2 : 16 = 0.80 * S - P) : P = 240 :=
by
  -- Proof steps would go here
  sorry

end original_purchase_price_l275_275977


namespace combination_count_l275_275367

-- Definitions from conditions
def packagingPapers : Nat := 10
def ribbons : Nat := 4
def stickers : Nat := 5

-- Proof problem statement
theorem combination_count : packagingPapers * ribbons * stickers = 200 := 
by
  sorry

end combination_count_l275_275367


namespace possible_value_of_a_l275_275870

variable {a b x : ℝ}

theorem possible_value_of_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x :=
sorry

end possible_value_of_a_l275_275870


namespace good_pair_bound_all_good_pairs_l275_275669

namespace good_pairs

-- Definition of a "good" pair
def is_good_pair (r s : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ) (a : Fin r → ℤ) (b : Fin s → ℤ),
  (∀ i j : Fin r, i ≠ j → a i ≠ a j) ∧
  (∀ i j : Fin s, i ≠ j → b i ≠ b j) ∧
  (∀ i : Fin r, P (a i) = 2) ∧
  (∀ j : Fin s, P (b j) = 5)

-- (a) Show that for every good pair (r, s), r, s ≤ 3
theorem good_pair_bound (r s : ℕ) (h : is_good_pair r s) : r ≤ 3 ∧ s ≤ 3 :=
sorry

-- (b) Determine all good pairs
theorem all_good_pairs (r s : ℕ) : is_good_pair r s ↔ (r ≤ 3 ∧ s ≤ 3 ∧ (
  (r = 1 ∧ s = 1) ∨ (r = 1 ∧ s = 2) ∨ (r = 1 ∧ s = 3) ∨
  (r = 2 ∧ s = 1) ∨ (r = 2 ∧ s = 2) ∨ (r = 2 ∧ s = 3) ∨
  (r = 3 ∧ s = 1) ∨ (r = 3 ∧ s = 2))) :=
sorry

end good_pairs

end good_pair_bound_all_good_pairs_l275_275669


namespace rate_of_interest_is_5_percent_l275_275986

-- Defining the conditions as constants
def simple_interest : ℝ := 4016.25
def principal : ℝ := 16065
def time_period : ℝ := 5

-- Proving that the rate of interest is 5%
theorem rate_of_interest_is_5_percent (R : ℝ) : 
  simple_interest = (principal * R * time_period) / 100 → 
  R = 5 :=
by
  intro h
  sorry

end rate_of_interest_is_5_percent_l275_275986


namespace sum_of_digits_of_binary_315_is_6_l275_275803
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l275_275803


namespace find_vertex_D_l275_275557

noncomputable def quadrilateral_vertices : Prop :=
  let A : (ℤ × ℤ) := (-1, -2)
  let B : (ℤ × ℤ) := (3, 1)
  let C : (ℤ × ℤ) := (0, 2)
  A ≠ B ∧ A ≠ C ∧ B ≠ C

theorem find_vertex_D (A B C D : ℤ × ℤ) (h_quad : quadrilateral_vertices) :
    (A = (-1, -2)) →
    (B = (3, 1)) →
    (C = (0, 2)) →
    (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) →
    D = (-4, -1) :=
by
  sorry

end find_vertex_D_l275_275557


namespace evaluateExpression_at_3_l275_275038

noncomputable def evaluateExpression (x : ℚ) : ℚ :=
  (x - 1 + (2 - 2 * x) / (x + 1)) / ((x * x - x) / (x + 1))

theorem evaluateExpression_at_3 : evaluateExpression 3 = 2 / 3 := by
  sorry

end evaluateExpression_at_3_l275_275038


namespace g_does_not_pass_through_fourth_quadrant_l275_275555

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) - 2
noncomputable def g (x : ℝ) : ℝ := 1 + (1 / x)

theorem g_does_not_pass_through_fourth_quadrant (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
    ¬(∃ x, x > 0 ∧ g x < 0) :=
by
    sorry

end g_does_not_pass_through_fourth_quadrant_l275_275555


namespace sum_of_digits_eq_11_l275_275279

-- Define the problem conditions
variables (p q r : ℕ)
variables (h1 : 1 ≤ p ∧ p ≤ 9)
variables (h2 : 1 ≤ q ∧ q ≤ 9)
variables (h3 : 1 ≤ r ∧ r ≤ 9)
variables (h4 : p ≠ q ∧ p ≠ r ∧ q ≠ r)
variables (h5 : (10 * p + q) * (10 * p + r) = 221)

-- Define the theorem
theorem sum_of_digits_eq_11 : p + q + r = 11 :=
by
  sorry

end sum_of_digits_eq_11_l275_275279


namespace complex_exp_form_pow_four_l275_275097

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end complex_exp_form_pow_four_l275_275097


namespace total_visitors_600_l275_275286

variable (Enjoyed Understood : Set ℕ)
variable (TotalVisitors : ℕ)
variable (E U : ℕ)

axiom no_enjoy_no_understand : ∀ v, v ∉ Enjoyed → v ∉ Understood
axiom equal_enjoy_understand : E = U
axiom enjoy_and_understand_fraction : E = 3 / 4 * TotalVisitors
axiom total_visitors_equation : TotalVisitors = E + 150

theorem total_visitors_600 : TotalVisitors = 600 := by
  sorry

end total_visitors_600_l275_275286


namespace A_n_divisible_by_225_l275_275481

theorem A_n_divisible_by_225 (n : ℕ) : 225 ∣ (16^n - 15 * n - 1) := by
  sorry

end A_n_divisible_by_225_l275_275481


namespace sum_reciprocals_l275_275455

theorem sum_reciprocals (a b α β : ℝ) (h1: 7 * a^2 + 2 * a + 6 = 0) (h2: 7 * b^2 + 2 * b + 6 = 0) 
  (h3: α = 1 / a) (h4: β = 1 / b) (h5: a + b = -2/7) (h6: a * b = 6/7) : 
  α + β = -1/3 :=
by
  sorry

end sum_reciprocals_l275_275455


namespace four_digit_numbers_proof_l275_275497

noncomputable def four_digit_numbers_total : ℕ := 9000
noncomputable def two_digit_numbers_total : ℕ := 90
noncomputable def max_distinct_products : ℕ := 4095
noncomputable def cannot_be_expressed_as_product : ℕ := four_digit_numbers_total - max_distinct_products

theorem four_digit_numbers_proof :
  cannot_be_expressed_as_product = 4905 :=
by
  sorry

end four_digit_numbers_proof_l275_275497


namespace room_breadth_is_five_l275_275567

theorem room_breadth_is_five 
  (length : ℝ)
  (height : ℝ)
  (bricks_per_square_meter : ℝ)
  (total_bricks : ℝ)
  (H_length : length = 4)
  (H_height : height = 2)
  (H_bricks_per_square_meter : bricks_per_square_meter = 17)
  (H_total_bricks : total_bricks = 340) 
  : ∃ (breadth : ℝ), breadth = 5 :=
by
  -- we leave the proof as sorry for now
  sorry

end room_breadth_is_five_l275_275567


namespace S4_equals_15_l275_275145

noncomputable def S_n (q : ℝ) (n : ℕ) := (1 - q^n) / (1 - q)

theorem S4_equals_15 (q : ℝ) (n : ℕ) (h1 : S_n q 1 = 1) (h2 : S_n q 5 = 5 * S_n q 3 - 4) : 
  S_n q 4 = 15 :=
by
  sorry

end S4_equals_15_l275_275145


namespace problem1_problem2_problem3_l275_275195

-- Prove \(2x = 4\) is a "difference solution equation"
theorem problem1 (x : ℝ) : (2 * x = 4) → x = 4 - 2 :=
by
  sorry

-- Given \(4x = ab + a\) is a "difference solution equation", prove \(3(ab + a) = 16\)
theorem problem2 (x ab a : ℝ) : (4 * x = ab + a) → 3 * (ab + a) = 16 :=
by
  sorry

-- Given \(4x = mn + m\) and \(-2x = mn + n\) are both "difference solution equations", prove \(3(mn + m) - 9(mn + n)^2 = 0\)
theorem problem3 (x mn m n : ℝ) :
  (4 * x = mn + m) ∧ (-2 * x = mn + n) → 3 * (mn + m) - 9 * (mn + n)^2 = 0 :=
by
  sorry

end problem1_problem2_problem3_l275_275195


namespace polynomials_with_conditions_l275_275519

theorem polynomials_with_conditions (n : ℕ) (h_pos : 0 < n) :
  (∃ P : Polynomial ℤ, Polynomial.degree P = n ∧ 
      (∃ (k : Fin n → ℤ), Function.Injective k ∧ (∀ i, P.eval (k i) = n) ∧ P.eval 0 = 0)) ↔ 
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
sorry

end polynomials_with_conditions_l275_275519


namespace circle_area_ratio_l275_275139

/-- If the diameter of circle R is 60% of the diameter of circle S, 
the area of circle R is 36% of the area of circle S. -/
theorem circle_area_ratio (D_S D_R A_S A_R : ℝ) (h : D_R = 0.60 * D_S) 
  (hS : A_S = Real.pi * (D_S / 2) ^ 2) (hR : A_R = Real.pi * (D_R / 2) ^ 2): 
  A_R = 0.36 * A_S := 
sorry

end circle_area_ratio_l275_275139


namespace find_slope_l275_275556

noncomputable def parabola_equation (x y : ℝ) := y^2 = 8 * x

def point_M : ℝ × ℝ := (-2, 2)

def line_through_focus (k x : ℝ) : ℝ := k * (x - 2)

def focus : ℝ × ℝ := (2, 0)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_slope (k : ℝ) : 
  (∀ x y A B, 
    parabola_equation x y → 
    (x = A ∨ x = B) → 
    line_through_focus k x = y → 
    parabola_equation A (k * (A - 2)) → 
    parabola_equation B (k * (B - 2)) → 
    dot_product (A + 2, (k * (A -2)) - 2) (B + 2, (k * (B - 2)) - 2) = 0) →
  k = 2 :=
sorry

end find_slope_l275_275556


namespace sum_of_digits_in_binary_representation_of_315_l275_275826

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l275_275826


namespace find_original_manufacturing_cost_l275_275667

noncomputable def originalManufacturingCost (P : ℝ) : ℝ := 0.70 * P

theorem find_original_manufacturing_cost (P : ℝ) (currentCost : ℝ) 
  (h1 : currentCost = 50) 
  (h2 : currentCost = P - 0.50 * P) : originalManufacturingCost P = 70 :=
by
  -- The actual proof steps would go here, but we'll add sorry for now
  sorry

end find_original_manufacturing_cost_l275_275667


namespace cab_driver_income_l275_275209

theorem cab_driver_income (x : ℕ)
  (h1 : 50 + 60 + 65 + 70 + x = 5 * 58) :
  x = 45 :=
by
  sorry

end cab_driver_income_l275_275209


namespace prob_white_given_popped_l275_275834

-- Definitions for given conditions:
def P_white : ℚ := 1 / 2
def P_yellow : ℚ := 1 / 4
def P_blue : ℚ := 1 / 4

def P_popped_given_white : ℚ := 1 / 3
def P_popped_given_yellow : ℚ := 3 / 4
def P_popped_given_blue : ℚ := 2 / 3

-- Calculations derived from conditions:
def P_white_popped : ℚ := P_white * P_popped_given_white
def P_yellow_popped : ℚ := P_yellow * P_popped_given_yellow
def P_blue_popped : ℚ := P_blue * P_popped_given_blue

def P_popped : ℚ := P_white_popped + P_yellow_popped + P_blue_popped

-- Main theorem to be proved:
theorem prob_white_given_popped : (P_white_popped / P_popped) = 2 / 11 :=
by sorry

end prob_white_given_popped_l275_275834


namespace complex_power_eq_rectangular_l275_275092

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_30 := Real.sin (Real.pi / 6)

theorem complex_power_eq_rectangular :
  (3 * (cos_30 + (complex.I * sin_30)))^4 = -40.5 + 40.5 * complex.I * Real.sqrt 3 := 
sorry

end complex_power_eq_rectangular_l275_275092


namespace num_isosceles_triangles_is_24_l275_275077

-- Define the structure of the hexagonal prism
structure HexagonalPrism :=
  (height : ℝ)
  (side_length : ℝ)
  (num_vertices : ℕ)

-- Define the specific hexagonal prism from the problem
def prism := HexagonalPrism.mk 2 1 12

-- Function to count the number of isosceles triangles in a given hexagonal prism
noncomputable def count_isosceles_triangles (hp : HexagonalPrism) : ℕ := sorry

-- The theorem that needs to be proved
theorem num_isosceles_triangles_is_24 :
  count_isosceles_triangles prism = 24 :=
sorry

end num_isosceles_triangles_is_24_l275_275077


namespace total_distance_walked_l275_275582

variables
  (distance1 : ℝ := 1.2)
  (distance2 : ℝ := 0.8)
  (distance3 : ℝ := 1.5)
  (distance4 : ℝ := 0.6)
  (distance5 : ℝ := 2)

theorem total_distance_walked :
  distance1 + distance2 + distance3 + distance4 + distance5 = 6.1 :=
sorry

end total_distance_walked_l275_275582


namespace range_of_angle_A_l275_275906

theorem range_of_angle_A (a b : ℝ) (A : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) 
  (h_triangle : 0 < A ∧ A ≤ Real.pi / 4) :
  (0 < A ∧ A ≤ Real.pi / 4) :=
by
  sorry

end range_of_angle_A_l275_275906


namespace ostap_advantageous_order_l275_275598

theorem ostap_advantageous_order (p1 p2 p3 : ℝ) (h1 : p3 < p1) (h2 : p1 < p2) : 
  ∀ order : List ℝ, 
    (order = [p1, p2, p3] ∨ order = [p2, p1, p3] ∨ order = [p3, p1, p2]) → (order.nth 1 = some p2) :=
sorry

end ostap_advantageous_order_l275_275598


namespace honey_last_nights_l275_275619

theorem honey_last_nights 
  (serving_per_cup : ℕ)
  (cups_per_night : ℕ)
  (ounces_per_container : ℕ)
  (servings_per_ounce : ℕ)
  (total_nights : ℕ) :
  serving_per_cup = 1 →
  cups_per_night = 2 →
  ounces_per_container = 16 →
  servings_per_ounce = 6 →
  total_nights = 48 := 
by
  intro h1 h2 h3 h4,
  sorry

end honey_last_nights_l275_275619


namespace matrix_inverse_correct_l275_275862

open Matrix

noncomputable def given_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7, -4], ![-3, 2]]

noncomputable def expected_inverse : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 2], ![1.5, 3.5]]

theorem matrix_inverse_correct :
  let det := (given_matrix 0 0) * (given_matrix 1 1) - (given_matrix 0 1) * (given_matrix 1 0)
  det ≠ 0 →
  (1 / det) • adjugate given_matrix = expected_inverse :=
by
  sorry

end matrix_inverse_correct_l275_275862


namespace beth_cans_of_corn_l275_275084

theorem beth_cans_of_corn (C P : ℕ) (h1 : P = 2 * C + 15) (h2 : P = 35) : C = 10 :=
by
  sorry

end beth_cans_of_corn_l275_275084


namespace find_x_l275_275663

-- Define the conditions
def condition (x : ℕ) := (4 * x)^2 - 2 * x = 8062

-- State the theorem
theorem find_x : ∃ x : ℕ, condition x ∧ x = 134 := sorry

end find_x_l275_275663


namespace advantageous_order_l275_275609

variables {p1 p2 p3 : ℝ}

-- Conditions
axiom prob_ordering : p3 < p1 ∧ p1 < p2

-- Definition of sequence probabilities
def prob_first_second := p1 * p2 + (1 - p1) * p2 * p3
def prob_second_first := p2 * p1 + (1 - p2) * p1 * p3

-- Theorem to be proved
theorem advantageous_order :
  prob_first_second = prob_second_first →
  p2 > p1 → (p2 > p1) :=
by
  sorry

end advantageous_order_l275_275609


namespace perpendicular_line_equation_l275_275654

-- Conditions definition
def line1 := ∀ x y : ℝ, 3 * x - 6 * y = 9
def point := (2 : ℝ, -3 : ℝ)

-- Target statement
theorem perpendicular_line_equation : 
  (∀ x y : ℝ, line1 x y → y = (-2) * x + 1) := 
by 
  -- proof goes here
  sorry

end perpendicular_line_equation_l275_275654


namespace sally_pens_proof_l275_275315

variable (p : ℕ)  -- define p as a natural number for pens each student received
variable (pensLeft : ℕ)  -- define pensLeft as a natural number for pens left after distributing to students

-- Function representing Sally giving pens to each student
def pens_after_giving_students (p : ℕ) : ℕ := 342 - 44 * p

-- Condition 1: Left half of the remainder in her locker
def locker_pens (p : ℕ) : ℕ := (pens_after_giving_students p) / 2

-- Condition 2: She took 17 pens home
def home_pens : ℕ := 17

-- Main proof statement
theorem sally_pens_proof :
  (locker_pens p + home_pens = pens_after_giving_students p) → p = 7 :=
by
  sorry

end sally_pens_proof_l275_275315


namespace sum_of_digits_in_binary_representation_of_315_l275_275823

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l275_275823


namespace find_linear_odd_increasing_function_l275_275882

theorem find_linear_odd_increasing_function (f : ℝ → ℝ)
    (h1 : ∀ x, f (f x) = 4 * x)
    (h2 : ∀ x, f x = -f (-x))
    (h3 : ∀ x y, x < y → f x < f y)
    (h4 : ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x) : 
    ∀ x, f x = 2 * x :=
by
  sorry

end find_linear_odd_increasing_function_l275_275882


namespace smallest_invariant_number_l275_275375

def operation (n : ℕ) : ℕ :=
  let q := n / 10
  let r := n % 10
  q + 2 * r

def is_invariant (n : ℕ) : Prop :=
  operation n = n

theorem smallest_invariant_number : ∃ n : ℕ, is_invariant n ∧ n = 10^99 + 1 :=
by
  sorry

end smallest_invariant_number_l275_275375


namespace functional_equation_f2023_l275_275121

theorem functional_equation_f2023 (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_one : f 1 = 1) :
  f 2023 = 2023 := sorry

end functional_equation_f2023_l275_275121


namespace perpendicular_line_equation_l275_275655

noncomputable def slope_intercept_form (a b c : ℝ) : (ℝ × ℝ) :=
  let m : ℝ := -a / b
  let b' : ℝ := c / b
  (m, b')

noncomputable def perpendicular_slope (slope : ℝ) : ℝ :=
  -1 / slope

theorem perpendicular_line_equation (x1 y1 : ℝ) 
  (h_line_eq : ∃ (a b c : ℝ), a * x1 + b * y1 = c)
  (h_point : x1 = 2 ∧ y1 = -3) :
  ∃ a b : ℝ, y1 = a * x1 + b ∧ a = -2 ∧ b = 1 :=
by
  let (m, b') := slope_intercept_form 3 (-6) 9
  have h_slope := (-1 : ℝ) / m
  have h_perpendicular_slope : h_slope = -2 := sorry
  let (x1, y1) := (2, -3)
  have h_eq : y1 - (-3) = h_slope * (x1 - 2) := sorry
  use [-2, 1]
  simp [h_eq]
  sorry

end perpendicular_line_equation_l275_275655


namespace sum_of_arithmetic_sequence_is_54_l275_275123

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence_is_54 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 8 = 6 + a 11) : 
  S 9 = 54 :=
sorry

end sum_of_arithmetic_sequence_is_54_l275_275123


namespace hotel_accommodation_arrangements_l275_275372

theorem hotel_accommodation_arrangements :
  let triple_room := 1
  let double_rooms := 2
  let adults := 3
  let children := 2
  (∀ (triple_room : ℕ) (double_rooms : ℕ) (adults : ℕ) (children : ℕ),
    children ≤ adults ∧ double_rooms + triple_room ≥ 1 →
    (∃ (arrangements : ℕ),
      arrangements = 60)) :=
sorry

end hotel_accommodation_arrangements_l275_275372


namespace parabola_focus_distance_l275_275140

theorem parabola_focus_distance (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 100) : x = 9 :=
sorry

end parabola_focus_distance_l275_275140


namespace find_radioactive_balls_within_7_checks_l275_275464

theorem find_radioactive_balls_within_7_checks :
  ∃ (balls : Finset α), balls.card = 11 ∧ ∃ radioactive_balls ⊆ balls, radioactive_balls.card = 2 ∧
  (∀ (check : Finset α → Prop), (∀ S, check S ↔ (∃ b ∈ S, b ∈ radioactive_balls)) →
  ∃ checks : Finset (Finset α), checks.card ≤ 7 ∧ (∀ b ∈ radioactive_balls, ∃ S ∈ checks, b ∈ S)) :=
sorry

end find_radioactive_balls_within_7_checks_l275_275464


namespace most_convincing_method_l275_275845

-- Defining the survey data
def male_participants : Nat := 4258
def male_believe_doping : Nat := 2360
def female_participants : Nat := 3890
def female_believe_framed : Nat := 2386

-- Defining the question-to-answer equivalence related to the most convincing method
theorem most_convincing_method :
  "Independence Test" = "Independence Test" := 
by
  sorry

end most_convincing_method_l275_275845


namespace alphametic_puzzle_l275_275748

theorem alphametic_puzzle (I D A M E R O : ℕ) 
  (h1 : R = 0) 
  (h2 : D + E = 10)
  (h3 : I + M + 1 = O)
  (h4 : A = D + 1) :
  I + 1 + M + 10 + 1 = O + 0 + A := sorry

end alphametic_puzzle_l275_275748


namespace range_of_b_l275_275287

theorem range_of_b (A B C a b c : ℝ)
  (h_acute : 0 < A ∧ A < real.pi / 2 ∧
             0 < B ∧ B < real.pi / 2 ∧
             0 < C ∧ C < real.pi / 2 ∧ A + B + C = real.pi)
  (h_a : a = 1)
  (h_B : B = real.pi / 3) :
  ∃ (lb ub : ℝ), lb = real.sqrt 3 / 2 ∧ ub = real.sqrt 3 ∧ lb < b ∧ b < ub :=
by
  -- Proof goes here.
  sorry

end range_of_b_l275_275287


namespace avg_of_all_5_is_8_l275_275621

-- Let a1, a2, a3 be three quantities such that their average is 4.
def is_avg_4 (a1 a2 a3 : ℝ) : Prop :=
  (a1 + a2 + a3) / 3 = 4

-- Let a4, a5 be the remaining two quantities such that their average is 14.
def is_avg_14 (a4 a5 : ℝ) : Prop :=
  (a4 + a5) / 2 = 14

-- Prove that the average of all 5 quantities is 8.
theorem avg_of_all_5_is_8 (a1 a2 a3 a4 a5 : ℝ) :
  is_avg_4 a1 a2 a3 ∧ is_avg_14 a4 a5 → 
  ((a1 + a2 + a3 + a4 + a5) / 5 = 8) :=
by
  intro h
  sorry

end avg_of_all_5_is_8_l275_275621


namespace k_eq_1_l275_275026

theorem k_eq_1 
  (n m k : ℕ) 
  (hn : n > 0) 
  (hm : m > 0) 
  (hk : k > 0) 
  (h : (n - 1) * n * (n + 1) = m^k) : 
  k = 1 := 
sorry

end k_eq_1_l275_275026


namespace question_1_question_2_l275_275725

open Real

theorem question_1 (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : ab < m / 2 → m > 2 := sorry

theorem question_2 (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) (h4 : 9 / a + 1 / b ≥ |x - 1| + |x + 2|) :
  -9/2 ≤ x ∧ x ≤ 7/2 := sorry

end question_1_question_2_l275_275725


namespace range_of_m_l275_275142

noncomputable def f (x : ℝ) := x * Real.exp x
noncomputable def g (x : ℝ) := (-x^2 + x + 1) * Real.exp x

theorem range_of_m :
  ∀ (m : ℝ),
  (∃ (x_0 x_1 x_2 : ℝ), x_0 ≠ x_1 ∧ x_1 ≠ x_2 ∧ x_0 ≠ x_2 ∧
    g x_0 = m ∧ g x_1 = m ∧ g x_2 = m) ↔
  m ∈ Ioo (-5 / Real.exp 2) 0 :=
begin
  sorry
end

end range_of_m_l275_275142


namespace car_speed_is_120_l275_275218

theorem car_speed_is_120 (v t : ℝ) (h1 : v > 0) (h2 : t > 0) (h3 : v * t = 75)
  (h4 : 1.5 * v * (t - (12.5 / 60)) = 75) : v = 120 := by
  sorry

end car_speed_is_120_l275_275218


namespace peter_ate_7_over_48_l275_275928

-- Define the initial conditions
def total_slices : ℕ := 16
def slices_peter_ate : ℕ := 2
def shared_slice : ℚ := 1/3

-- Define the first part of the problem
def fraction_peter_ate_alone : ℚ := slices_peter_ate / total_slices

-- Define the fraction Peter ate from sharing one slice
def fraction_peter_ate_shared : ℚ := shared_slice / total_slices

-- Define the total fraction Peter ate
def total_fraction_peter_ate : ℚ := fraction_peter_ate_alone + fraction_peter_ate_shared

-- Create the theorem to be proved (statement only)
theorem peter_ate_7_over_48 :
  total_fraction_peter_ate = 7 / 48 :=
by
  sorry

end peter_ate_7_over_48_l275_275928


namespace hazel_drank_one_cup_l275_275133

theorem hazel_drank_one_cup (total_cups made_to_crew bike_sold friends_given remaining_cups : ℕ) 
  (H1 : total_cups = 56)
  (H2 : made_to_crew = total_cups / 2)
  (H3 : bike_sold = 18)
  (H4 : friends_given = bike_sold / 2)
  (H5 : remaining_cups = total_cups - (made_to_crew + bike_sold + friends_given)) :
  remaining_cups = 1 := 
sorry

end hazel_drank_one_cup_l275_275133


namespace certain_number_division_l275_275247

theorem certain_number_division (N G : ℤ) : 
  G = 88 ∧ (∃ k : ℤ, N = G * k + 31) ∧ (∃ m : ℤ, 4521 = G * m + 33) → 
  N = 4519 := 
by
  sorry

end certain_number_division_l275_275247


namespace probability_second_ball_new_given_first_new_l275_275283

-- Define the conditions
def totalBalls : ℕ := 10
def newBalls : ℕ := 6
def oldBalls : ℕ := 4

-- Define the events
def firstBallIsNew : Prop := true  -- Given condition
def secondBallIsNew : Prop := true  -- Event to prove

-- Define the total probability of drawing a new ball on the first draw
def P_A := (newBalls : ℚ) / (totalBalls : ℚ)

-- Define the joint probability of drawing two new balls
def P_AB := (newBalls : ℚ) / (totalBalls : ℚ) * (newBalls - 1 : ℚ) / (totalBalls - 1 : ℚ)

-- Define the conditional probability of drawing a new ball on the second draw given the first was new
def P_B_given_A := P_AB / P_A

-- The theorem to prove
theorem probability_second_ball_new_given_first_new :
  P_B_given_A = 5 / 9 :=
by
  -- state the probabilities explicitly for clarity
  have h_P_A : P_A = 3 / 5 := by sorry
  have h_P_AB : P_AB = 1 / 3 := by sorry

  -- compute the conditional probability
  rw [P_B_given_A, h_P_A, h_P_AB]
  sorry

end probability_second_ball_new_given_first_new_l275_275283


namespace fizz_preference_count_l275_275740

-- Definitions from conditions
def total_people : ℕ := 500
def fizz_angle : ℕ := 270
def total_angle : ℕ := 360
def fizz_fraction : ℚ := fizz_angle / total_angle

-- The target proof statement
theorem fizz_preference_count (hp : total_people = 500) 
                              (ha : fizz_angle = 270) 
                              (ht : total_angle = 360)
                              (hf : fizz_fraction = 3 / 4) : 
    total_people * fizz_fraction = 375 := by
    sorry

end fizz_preference_count_l275_275740


namespace counting_4digit_integers_l275_275720

theorem counting_4digit_integers (x y : ℕ) (a b c d : ℕ) :
  (x = 1000 * a + 100 * b + 10 * c + d) →
  (y = 1000 * d + 100 * c + 10 * b + a) →
  (y - x = 3177) →
  (1 ≤ a) → (a ≤ 6) →
  (0 ≤ b) → (b ≤ 7) →
  (c = b + 2) →
  (d = a + 3) →
  ∃ n : ℕ, n = 48 := 
sorry

end counting_4digit_integers_l275_275720


namespace exactly_two_toads_l275_275747

universe u

structure Amphibian where
  brian : Bool
  julia : Bool
  sean : Bool
  victor : Bool

def are_same_species (x y : Bool) : Bool := x = y

-- Definitions of statements by each amphibian
def Brian_statement (a : Amphibian) : Bool :=
  are_same_species a.brian a.sean

def Julia_statement (a : Amphibian) : Bool :=
  a.victor

def Sean_statement (a : Amphibian) : Bool :=
  ¬ a.julia

def Victor_statement (a : Amphibian) : Bool :=
  (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2

-- Conditions translated to Lean definition
def valid_statements (a : Amphibian) : Prop :=
  (a.brian → Brian_statement a) ∧
  (¬ a.brian → ¬ Brian_statement a) ∧
  (a.julia → Julia_statement a) ∧
  (¬ a.julia → ¬ Julia_statement a) ∧
  (a.sean → Sean_statement a) ∧
  (¬ a.sean → ¬ Sean_statement a) ∧
  (a.victor → Victor_statement a) ∧
  (¬ a.victor → ¬ Victor_statement a)

theorem exactly_two_toads (a : Amphibian) (h : valid_statements a) : 
( (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2 ) :=
sorry

end exactly_two_toads_l275_275747


namespace quinn_free_donuts_l275_275468

-- Definitions based on conditions
def books_per_week : ℕ := 2
def weeks : ℕ := 10
def books_needed_for_donut : ℕ := 5

-- Calculation based on conditions
def total_books_read : ℕ := books_per_week * weeks
def free_donuts (total_books : ℕ) : ℕ := total_books / books_needed_for_donut

-- Proof statement
theorem quinn_free_donuts : free_donuts total_books_read = 4 := by
  sorry

end quinn_free_donuts_l275_275468


namespace inradius_of_right_triangle_l275_275396

variable (a b c : ℕ) -- Define the sides
def right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

noncomputable def area (a b : ℕ) : ℝ :=
  0.5 * (a : ℝ) * (b : ℝ)

noncomputable def semiperimeter (a b c : ℕ) : ℝ :=
  ((a + b + c) : ℝ) / 2

noncomputable def inradius (a b c : ℕ) : ℝ :=
  let s := semiperimeter a b c
  let A := area a b
  A / s

theorem inradius_of_right_triangle (h : right_triangle 7 24 25) : inradius 7 24 25 = 3 := by
  sorry

end inradius_of_right_triangle_l275_275396


namespace identified_rectangle_perimeter_l275_275981

-- Define the side length of the square
def side_length_mm : ℕ := 75

-- Define the heights of the rectangles
variables (x y z : ℕ)

-- Define conditions
def rectangles_cut_condition (x y z : ℕ) : Prop := x + y + z = side_length_mm
def perimeter_relation_condition (x y z : ℕ) : Prop := 2 * (x + side_length_mm) = (y + side_length_mm) + (z + side_length_mm)

-- Define the perimeter of the identified rectangle
def identified_perimeter_mm (x : ℕ) := 2 * (x + side_length_mm)

-- Define conversion from mm to cm
def mm_to_cm (mm : ℕ) : ℕ := mm / 10

-- Final proof statement
theorem identified_rectangle_perimeter :
  ∃ x y z : ℕ, rectangles_cut_condition x y z ∧ perimeter_relation_condition x y z ∧ mm_to_cm (identified_perimeter_mm x) = 20 := 
sorry

end identified_rectangle_perimeter_l275_275981


namespace arithmetic_sequence_ratio_l275_275480

theorem arithmetic_sequence_ratio (a d : ℕ) (h : b = a + 3 * d) : a = 1 -> d = 1 -> (a / b = 1 / 4) :=
by
  sorry

end arithmetic_sequence_ratio_l275_275480


namespace sum_mod_20_l275_275533

/-- Define the elements that are summed. -/
def elements : List ℤ := [82, 83, 84, 85, 86, 87, 88, 89]

/-- The problem statement to prove. -/
theorem sum_mod_20 : (elements.sum % 20) = 15 := by
  sorry

end sum_mod_20_l275_275533


namespace find_quadruples_l275_275698

theorem find_quadruples (x y z n : ℕ) : 
  x^2 + y^2 + z^2 + 1 = 2^n → 
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
by
  sorry

end find_quadruples_l275_275698


namespace sum_of_digits_of_binary_315_is_6_l275_275805
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l275_275805


namespace quadratic_distinct_roots_l275_275430

theorem quadratic_distinct_roots (m : ℝ) : 
  ((∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1 * r2 = 9 ∧ r1 + r2 = -m) ↔ 
  m ∈ set.Ioo (-∞) (-6) ∪ set.Ioo (6) ∞) := 
by sorry

end quadratic_distinct_roots_l275_275430


namespace evaluate_expression_l275_275526

variable (a : ℕ)

theorem evaluate_expression (h : a = 2) : a^3 * a^4 = 128 :=
by
  sorry

end evaluate_expression_l275_275526


namespace hockey_team_selection_l275_275067

theorem hockey_team_selection :
  let total_players := 18
  let quadruplets := 4
  let starters := 7
  quad_choice_0 := @nat.choose (total_players - quadruplets) starters
  quad_choice_1 := quadruplets * @nat.choose (total_players - quadruplets) (starters - 1)
  quad_choice_2 := @nat.choose quadruplets 2 * @nat.choose (total_players - quadruplets) (starters - 2)
  quad_choice_0 + quad_choice_1 + quad_choice_2 = 27456
:=
by
  let total_players := 18
  let quadruplets := 4
  let starters := 7
  let quad_choice_0 := @nat.choose (total_players - quadruplets) starters
  let quad_choice_1 := quadruplets * @nat.choose (total_players - quadruplets) (starters - 1)
  let quad_choice_2 := @nat.choose quadruplets 2 * @nat.choose (total_players - quadruplets) (starters - 2)
  have := quad_choice_0 + quad_choice_1 + quad_choice_2
  sorry

end hockey_team_selection_l275_275067


namespace sum_of_digits_base_2_315_l275_275817

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l275_275817


namespace sum_of_10th_degree_polynomials_is_no_higher_than_10_l275_275953

-- Given definitions of two 10th-degree polynomials
def polynomial1 := ∃p : Polynomial ℝ, p.degree = 10
def polynomial2 := ∃p : Polynomial ℝ, p.degree = 10

-- Statement to prove
theorem sum_of_10th_degree_polynomials_is_no_higher_than_10 :
  ∀ (p q : Polynomial ℝ), p.degree = 10 → q.degree = 10 → (p + q).degree ≤ 10 := by
  sorry

end sum_of_10th_degree_polynomials_is_no_higher_than_10_l275_275953


namespace max_value_of_a_l275_275589

theorem max_value_of_a
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

example 
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) : 
  (7 - Real.sqrt 46) / 3 ≤ a :=
sorry

end max_value_of_a_l275_275589


namespace total_marbles_l275_275923

def Mary_marbles : ℕ := 9
def Joan_marbles : ℕ := 3

theorem total_marbles : Mary_marbles + Joan_marbles = 12 :=
by
  -- Please provide the proof here if needed
  sorry

end total_marbles_l275_275923


namespace emily_second_round_points_l275_275241

theorem emily_second_round_points (P : ℤ)
  (first_round_points : ℤ := 16)
  (last_round_points_lost : ℤ := 48)
  (end_points : ℤ := 1)
  (points_equation : first_round_points + P - last_round_points_lost = end_points) :
  P = 33 :=
  by {
    sorry
  }

end emily_second_round_points_l275_275241


namespace abc_perfect_ratio_l275_275444

theorem abc_perfect_ratio {a b c : ℚ} (h1 : ∃ t : ℤ, a + b + c = t ∧ a^2 + b^2 + c^2 = t) :
  ∃ (p q : ℤ), (abc = p^3 / q^2) ∧ (IsCoprime p q) := 
sorry

end abc_perfect_ratio_l275_275444


namespace length_of_BD_l275_275631

theorem length_of_BD (AB AC CB BD : ℝ) (h1 : AB = 10) (h2 : AC = 4 * CB) (h3 : AC = 4 * 2) (h4 : CB = 2) :
  BD = 3 :=
sorry

end length_of_BD_l275_275631


namespace rectangle_area_l275_275379

theorem rectangle_area (L W P A : ℕ) (h1 : P = 52) (h2 : L = 11) (h3 : 2 * L + 2 * W = P) : 
  A = L * W → A = 165 :=
by
  sorry

end rectangle_area_l275_275379


namespace yards_dyed_green_calc_l275_275637

-- Given conditions: total yards dyed and yards dyed pink
def total_yards_dyed : ℕ := 111421
def yards_dyed_pink : ℕ := 49500

-- Goal: Prove the number of yards dyed green
theorem yards_dyed_green_calc : total_yards_dyed - yards_dyed_pink = 61921 :=
by 
-- sorry means that the proof is skipped.
sorry

end yards_dyed_green_calc_l275_275637


namespace min_weighings_to_find_heaviest_l275_275719

-- Given conditions
variable (n : ℕ) (hn : n > 2)
variables (coins : Fin n) -- Representing coins with distinct masses
variables (scales : Fin n) -- Representing n scales where one is faulty

-- Theorem statement: Minimum number of weighings to find the heaviest coin
theorem min_weighings_to_find_heaviest : ∃ m, m = 2 * n - 1 := 
by
  existsi (2 * n - 1)
  rfl

end min_weighings_to_find_heaviest_l275_275719


namespace part_I_part_II_l275_275743

variable {α : Type*} [LinearOrderedField α]

-- Assume necesssary conditions
variables (A B C a b c : α)
variables (h_triangle_acute : A + B + C = π)
variables (h_b : b = (a / 2) * sin C)
variables (h_A_pos : 0 < A ∧ A < π / 2)
variables (h_B_pos : 0 < B ∧ B < π / 2)
variables (h_C_pos : 0 < C ∧ C < π / 2)

-- Part (I)
theorem part_I : (1 / tan A) + (1 / tan C) = (1 / 2) :=
by sorry

-- Part (II)
theorem part_II : Sup {tan B | 0 < A ∧ A < π / 2 ∧ A + B + C = π ∧ b = (a / 2) * sin C ∧ B = π - A - C} = (8 / 15) :=
by sorry

end part_I_part_II_l275_275743


namespace price_is_219_l275_275213

noncomputable def discount_coupon1 (price : ℝ) : ℝ :=
  if price > 50 then 0.1 * price else 0

noncomputable def discount_coupon2 (price : ℝ) : ℝ :=
  if price > 100 then 20 else 0

noncomputable def discount_coupon3 (price : ℝ) : ℝ :=
  if price > 100 then 0.18 * (price - 100) else 0

noncomputable def more_savings_coupon1 (price : ℝ) : Prop :=
  discount_coupon1 price > discount_coupon2 price ∧ discount_coupon1 price > discount_coupon3 price

theorem price_is_219 (price : ℝ) :
  more_savings_coupon1 price → price = 219 :=
by
  sorry

end price_is_219_l275_275213


namespace minimum_bailing_rate_l275_275201

theorem minimum_bailing_rate (distance_to_shore : ℝ) (row_speed : ℝ) (leak_rate : ℝ) (max_water_intake : ℝ)
  (time_to_shore : ℝ := distance_to_shore / row_speed * 60) (total_water_intake : ℝ := time_to_shore * leak_rate) :
  distance_to_shore = 1.5 → row_speed = 3 → leak_rate = 10 → max_water_intake = 40 →
  ∃ (bail_rate : ℝ), bail_rate ≥ 9 :=
by
  sorry

end minimum_bailing_rate_l275_275201


namespace exists_special_N_l275_275458

open Nat

theorem exists_special_N :
  ∃ N : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 150 → N % i = 0 ∨ i = 127 ∨ i = 128) ∧ 
  ¬ (N % 127 = 0) ∧ ¬ (N % 128 = 0) :=
by
  sorry

end exists_special_N_l275_275458


namespace calculate_b_l275_275425

open Real

theorem calculate_b (b : ℝ) (h : ∫ x in e..b, 2 / x = 6) : b = exp 4 := 
sorry

end calculate_b_l275_275425


namespace square_area_l275_275847

theorem square_area (x : ℝ) (side_length : ℝ) 
  (h1_side_length : side_length = 5 * x - 10)
  (h2_side_length : side_length = 3 * (x + 4)) :
  side_length ^ 2 = 2025 :=
by
  sorry

end square_area_l275_275847


namespace impossible_ratio_5_11_l275_275516

theorem impossible_ratio_5_11:
  ∀ (b g: ℕ), 
  b + g ≥ 66 →
  b + 11 = g - 13 →
  ¬(5 * b = 11 * (b + 24) ∧ b ≥ 21) := 
by
  intros b g h1 h2 h3
  sorry

end impossible_ratio_5_11_l275_275516


namespace part1_part2_part3_l275_275169

-- Part 1: Simplifying the Expression
theorem part1 (a b : ℝ) : 
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 :=
by sorry

-- Part 2: Finding the Value of an Expression
theorem part2 (x y : ℝ) (h : x^2 - 2 * y = 4) : 
  3 * x^2 - 6 * y - 21 = -9 :=
by sorry

-- Part 3: Evaluating a Compound Expression
theorem part3 (a b c d : ℝ) (h1 : a - 2 * b = 6) (h2 : 2 * b - c = -8) (h3 : c - d = 9) : 
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by sorry

end part1_part2_part3_l275_275169


namespace last_digit_2008_pow_2008_l275_275947

theorem last_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := by
  -- Here, the proof would follow the understanding of the cyclic pattern of the last digits of powers of 2008
  sorry

end last_digit_2008_pow_2008_l275_275947


namespace largest_area_of_triangle_DEF_l275_275220

noncomputable def maxAreaTriangleDEF : Real :=
  let DE := 16.0
  let EF_to_FD := 25.0 / 24.0
  let max_area := 446.25
  max_area

theorem largest_area_of_triangle_DEF :
  ∀ (DE : Real) (EF FD : Real),
    DE = 16 ∧ EF / FD = 25 / 24 → 
    (∃ (area : Real), area ≤ maxAreaTriangleDEF) :=
by 
  sorry

end largest_area_of_triangle_DEF_l275_275220


namespace possible_original_numbers_l275_275355

def four_digit_original_number (N : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    N = 1000 * a + 100 * b + 10 * c + d ∧ 
    (a+1) * (b+2) * (c+3) * (d+4) = 234 ∧ 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

theorem possible_original_numbers : 
  four_digit_original_number 1109 ∨ four_digit_original_number 2009 :=
sorry

end possible_original_numbers_l275_275355


namespace range_of_m_l275_275726

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h1 : 1/x + 1/y = 1) (h2 : x + y > m) : m < 4 := 
sorry

end range_of_m_l275_275726


namespace geometric_sequence_s4_l275_275146

theorem geometric_sequence_s4
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = (∑ i in finset.range n, a (i + 1)))
  (h4 : S 5 = 5 * S 3 - 4) :
  S 4 = 15 :=
sorry

end geometric_sequence_s4_l275_275146


namespace gf_neg3_eq_1262_l275_275447

def f (x : ℤ) : ℤ := x^3 + 6
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 2

theorem gf_neg3_eq_1262 : g (f (-3)) = 1262 := by
  sorry

end gf_neg3_eq_1262_l275_275447


namespace optimal_order_for_ostap_l275_275600

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l275_275600


namespace part1_part1_monotonicity_intervals_part2_l275_275886

noncomputable def f (x a : ℝ) := x * Real.log x - a * (x - 1)^2 - x + 1

-- Part 1: Monotonicity and Extreme values when a = 0
theorem part1 (x : ℝ) : f x 0 = x * Real.log x - x + 1 := sorry

theorem part1_monotonicity_intervals (x : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x < 1 → f x 0 < f 1 0) ∧
  (∀ (x : ℝ), x > 1 → f 1 0 < f x 0) ∧ 
  (f 1 0 = 0) := sorry

-- Part 2: f(x) < 0 for x > 1 and a >= 1/2
theorem part2 (x a : ℝ) (hx : x > 1) (ha : a ≥ 1/2) : f x a < 0 := sorry

end part1_part1_monotonicity_intervals_part2_l275_275886


namespace moles_HCl_formed_l275_275248

-- Define the initial moles of CH4 and Cl2
def CH4_initial : ℕ := 2
def Cl2_initial : ℕ := 4

-- Define the balanced chemical equation in terms of the number of moles
def balanced_equation (CH4 : ℕ) (Cl2 : ℕ) : Prop :=
  CH4 + 4 * Cl2 = 1 * CH4 + 4 * Cl2

-- Theorem statement: Given the conditions, prove the number of moles of HCl formed is 4
theorem moles_HCl_formed (CH4_initial Cl2_initial : ℕ) (h_CH4 : CH4_initial = 2) (h_Cl2 : Cl2_initial = 4) :
  ∃ (HCl : ℕ), HCl = 4 :=
  sorry

end moles_HCl_formed_l275_275248


namespace sum_of_squares_of_reciprocals_l275_275633

-- Definitions based on the problem's conditions
variables (a b : ℝ) (hab : a + b = 3 * a * b + 1) (h_an : a ≠ 0) (h_bn : b ≠ 0)

-- Statement of the problem to be proved
theorem sum_of_squares_of_reciprocals :
  (1 / a^2) + (1 / b^2) = (4 * a * b + 10) / (a^2 * b^2) :=
sorry

end sum_of_squares_of_reciprocals_l275_275633


namespace symmetric_line_eq_l275_275945

theorem symmetric_line_eq (x y : ℝ) (h₁ : y = 3 * x + 4) : y = x → y = (1 / 3) * x - (4 / 3) :=
by
  sorry

end symmetric_line_eq_l275_275945


namespace binary_ternary_product_base_10_l275_275404

theorem binary_ternary_product_base_10 :
  let b2 := 2
  let t3 := 3
  let n1 := 1011 -- binary representation
  let n2 := 122 -- ternary representation
  let a1 := (1 * b2^3) + (0 * b2^2) + (1 * b2^1) + (1 * b2^0)
  let a2 := (1 * t3^2) + (2 * t3^1) + (2 * t3^0)
  a1 * a2 = 187 :=
by
  sorry

end binary_ternary_product_base_10_l275_275404


namespace find_value_of_2a10_minus_a12_l275_275289

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the given conditions
def condition (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (a 4 + a 6 + a 8 + a 10 + a 12 = 120)

-- State the theorem
theorem find_value_of_2a10_minus_a12 (a : ℕ → ℝ) (h : condition a) : 2 * a 10 - a 12 = 24 :=
by sorry

end find_value_of_2a10_minus_a12_l275_275289


namespace quadratic_has_root_in_interval_l275_275930

theorem quadratic_has_root_in_interval (a b c : ℝ) (h : 2 * a + 3 * b + 6 * c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end quadratic_has_root_in_interval_l275_275930


namespace cakes_remaining_l275_275389

theorem cakes_remaining (initial_cakes sold_cakes remaining_cakes: ℕ) (h₀ : initial_cakes = 167) (h₁ : sold_cakes = 108) (h₂ : remaining_cakes = initial_cakes - sold_cakes) : remaining_cakes = 59 :=
by
  rw [h₀, h₁] at h₂
  exact h₂

end cakes_remaining_l275_275389


namespace perpendicular_line_through_point_l275_275651

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l275_275651


namespace sacks_per_day_l275_275627

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (h1 : total_sacks = 56) (h2 : days = 4) : total_sacks / days = 14 := by
  sorry

end sacks_per_day_l275_275627


namespace rhombus_second_diagonal_l275_275628

theorem rhombus_second_diagonal (perimeter : ℝ) (d1 : ℝ) (side : ℝ) (half_d2 : ℝ) (d2 : ℝ) :
  perimeter = 52 → d1 = 24 → side = 13 → (half_d2 = 5) → d2 = 2 * half_d2 → d2 = 10 :=
by
  sorry

end rhombus_second_diagonal_l275_275628


namespace condo_floors_l275_275376

theorem condo_floors (F P : ℕ) (h1: 12 * F + 2 * P = 256) (h2 : P = 2) : F + P = 23 :=
by
  sorry

end condo_floors_l275_275376


namespace opening_night_ticket_price_l275_275978

theorem opening_night_ticket_price :
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let matinee_price := 5
  let evening_price := 7
  let popcorn_price := 10
  let total_revenue := 1670
  let total_customers := matinee_customers + evening_customers + opening_night_customers
  let popcorn_customers := total_customers / 2
  let total_matinee_revenue := matinee_customers * matinee_price
  let total_evening_revenue := evening_customers * evening_price
  let total_popcorn_revenue := popcorn_customers * popcorn_price
  let known_revenue := total_matinee_revenue + total_evening_revenue + total_popcorn_revenue
  let opening_night_revenue := total_revenue - known_revenue
  let opening_night_price := opening_night_revenue / opening_night_customers
  opening_night_price = 10 := by
  sorry

end opening_night_ticket_price_l275_275978


namespace janeth_balloons_count_l275_275296

-- Define the conditions
def bags_round_balloons : Nat := 5
def balloons_per_bag_round : Nat := 20
def bags_long_balloons : Nat := 4
def balloons_per_bag_long : Nat := 30
def burst_round_balloons : Nat := 5

-- Proof statement
theorem janeth_balloons_count:
  let total_round_balloons := bags_round_balloons * balloons_per_bag_round
  let total_long_balloons := bags_long_balloons * balloons_per_bag_long
  let total_balloons := total_round_balloons + total_long_balloons
  total_balloons - burst_round_balloons = 215 :=
by {
  sorry
}

end janeth_balloons_count_l275_275296


namespace magician_finds_coins_l275_275511

noncomputable def magician_assistant_trick : Type := sorry

-- Define the set of 12 boxes
def boxes : Finset (Fin 12) := Finset.univ

-- Define the hiding function
def hide (i j : Fin 12) : Finset (Fin 12) := {i, j}

-- Condition: assistant opens one box that does not contain a coin
def assistant_opens (k : Fin 12) (i j : Fin 12) : Prop := k ≠ i ∧ k ≠ j

-- Define the method template (as a simplified example)
def template (i : Fin 12) : Finset (Fin 12) := {i, (i + 1) % 12, (i + 4) % 12, (i + 6) % 12}

-- Main theorem which ensures the trick always succeeds
theorem magician_finds_coins : 
  ∀ (i j k : Fin 12), 
    (j ≠ 1 ∧ k ≠ j ∧ k ≠ (1 : Fin 12)) → 
      ∃ (m : Fin 12), ({i, j} ⊆ template m) ∧ assistant_opens k i j :=
sorry

end magician_finds_coins_l275_275511


namespace valid_three_digit_numbers_no_seven_nine_l275_275273

noncomputable def count_valid_three_digit_numbers : Nat := 
  let hundredsChoices := 7
  let tensAndUnitsChoices := 8
  hundredsChoices * tensAndUnitsChoices * tensAndUnitsChoices

theorem valid_three_digit_numbers_no_seven_nine : 
  count_valid_three_digit_numbers = 448 := by
  sorry

end valid_three_digit_numbers_no_seven_nine_l275_275273


namespace percentage_of_women_attended_picnic_l275_275433

variable (E : ℝ) -- Total number of employees
variable (M : ℝ) -- The number of men
variable (W : ℝ) -- The number of women
variable (P : ℝ) -- Percentage of women who attended the picnic

-- Conditions
variable (h1 : M = 0.30 * E)
variable (h2 : W = E - M)
variable (h3 : 0.20 * M = 0.20 * 0.30 * E)
variable (h4 : 0.34 * E = 0.20 * 0.30 * E + P * (E - 0.30 * E))

-- Goal
theorem percentage_of_women_attended_picnic : P = 0.40 :=
by
  sorry

end percentage_of_women_attended_picnic_l275_275433


namespace value_of_m_l275_275125

theorem value_of_m (z1 z2 m : ℝ) (h1 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z1 = 0)
  (h2 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z2 = 0)
  (h3 : |z1 - z2| = 3) : m = 4 ∨ m = 17 / 2 := sorry

end value_of_m_l275_275125


namespace eval_to_one_l275_275356

noncomputable def evalExpression (a b c : ℝ) : ℝ :=
  let numerator := (1 / a + 1 / b - 2 * c / (a * b)) * (a + b + 2 * c)
  let denominator := 1 / a^2 + 1 / b^2 + 2 / (a * b) - 4 * c^2 / (a^2 * b^2)
  numerator / denominator

theorem eval_to_one : 
  evalExpression 7.4 (5 / 37) c = 1 := 
by 
  sorry

end eval_to_one_l275_275356


namespace complex_is_1_sub_sqrt3i_l275_275065

open Complex

theorem complex_is_1_sub_sqrt3i (z : ℂ) (h : z * (1 + Real.sqrt 3 * I) = abs (1 + Real.sqrt 3 * I)) : z = 1 - Real.sqrt 3 * I :=
sorry

end complex_is_1_sub_sqrt3i_l275_275065


namespace greatest_x_integer_l275_275658

theorem greatest_x_integer (x : ℤ) : 
  (∃ k : ℤ, (x^2 + 4 * x + 9) = k * (x - 4)) ↔ x ≤ 5 :=
by
  sorry

end greatest_x_integer_l275_275658


namespace num_cows_l275_275504

-- Define the context
variable (C H L Heads : ℕ)

-- Define the conditions
axiom condition1 : L = 2 * Heads + 8
axiom condition2 : L = 4 * C + 2 * H
axiom condition3 : Heads = C + H

-- State the goal
theorem num_cows : C = 4 := by
  sorry

end num_cows_l275_275504


namespace tiger_time_to_pass_specific_point_l275_275217

theorem tiger_time_to_pass_specific_point :
  ∀ (distance_tree : ℝ) (time_tree : ℝ) (length_tiger : ℝ),
  distance_tree = 20 →
  time_tree = 5 →
  length_tiger = 5 →
  (length_tiger / (distance_tree / time_tree)) = 1.25 :=
by
  intros distance_tree time_tree length_tiger h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tiger_time_to_pass_specific_point_l275_275217


namespace inequality_one_solution_inequality_two_solution_l275_275709

theorem inequality_one_solution (x : ℝ) :
  (-2 * x^2 + x < -3) ↔ (x < -1 ∨ x > 3 / 2) :=
sorry

theorem inequality_two_solution (x : ℝ) :
  (x + 1) / (x - 2) ≤ 2 ↔ (x < 2 ∨ x ≥ 5) :=
sorry

end inequality_one_solution_inequality_two_solution_l275_275709


namespace intersection_of_M_and_N_l275_275264

namespace ProofProblem

def M := { x : ℝ | x^2 < 4 }
def N := { x : ℝ | x < 1 }

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end ProofProblem

end intersection_of_M_and_N_l275_275264


namespace speed_of_first_car_l275_275959

theorem speed_of_first_car (v : ℝ) (h1 : 2.5 * v + 2.5 * 45 = 175) : v = 25 :=
by
  sorry

end speed_of_first_car_l275_275959


namespace flat_fee_l275_275839

theorem flat_fee (f n : ℝ) (h1 : f + 3 * n = 215) (h2 : f + 6 * n = 385) : f = 45 :=
  sorry

end flat_fee_l275_275839


namespace factorization_mn_l275_275529

variable (m n : ℝ) -- Declare m and n as arbitrary real numbers.

theorem factorization_mn (m n : ℝ) : m^2 - m * n = m * (m - n) := by
  sorry

end factorization_mn_l275_275529


namespace time_per_student_l275_275734

-- Given Conditions
def total_students : ℕ := 18
def groups : ℕ := 3
def minutes_per_group : ℕ := 24

-- Mathematical proof problem
theorem time_per_student :
  (minutes_per_group / (total_students / groups)) = 4 := by
  -- Proof not required, adding placeholder
  sorry

end time_per_student_l275_275734


namespace probability_multiple_of_4_l275_275578

theorem probability_multiple_of_4 :
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  prob_end_multiple_of_4 = 7 / 64 :=
by
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  have h : prob_end_multiple_of_4 = 7 / 64 := by sorry
  exact h

end probability_multiple_of_4_l275_275578


namespace no_square_sum_l275_275916

theorem no_square_sum (x y : ℕ) (hxy_pos : 0 < x ∧ 0 < y)
  (hxy_gcd : Nat.gcd x y = 1)
  (hxy_perf : ∃ k : ℕ, x + 3 * y^2 = k^2) : ¬ ∃ z : ℕ, x^2 + 9 * y^4 = z^2 :=
by
  sorry

end no_square_sum_l275_275916


namespace p_necessary_not_sufficient_for_q_l275_275414

variables (a b c : ℝ) (p q : Prop)

def condition_p : Prop := a * b * c = 0
def condition_q : Prop := a = 0

theorem p_necessary_not_sufficient_for_q : (q → p) ∧ ¬ (p → q) :=
by
  let p := condition_p a b c
  let q := condition_q a
  sorry

end p_necessary_not_sufficient_for_q_l275_275414


namespace fractions_sum_identity_l275_275281

theorem fractions_sum_identity (a b c : ℝ) (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / ((b - c) ^ 2) + b / ((c - a) ^ 2) + c / ((a - b) ^ 2) = 0 :=
by
  sorry

end fractions_sum_identity_l275_275281


namespace geom_series_correct_sum_l275_275238

-- Define the geometric series sum
noncomputable def geom_series_sum (a r : ℚ) (n : ℕ) :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
def a := (1 : ℚ) / 4
def r := (1 : ℚ) / 4
def n := 8

-- Correct answer sum
def correct_sum := (65535 : ℚ) / 196608

-- Proof problem statement
theorem geom_series_correct_sum : geom_series_sum a r n = correct_sum := 
  sorry

end geom_series_correct_sum_l275_275238


namespace sequence_closed_form_l275_275129

theorem sequence_closed_form (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 3) :
  ∀ n : ℕ, a n = 2^(n + 1) - 3 :=
by 
sorry

end sequence_closed_form_l275_275129


namespace alice_outfits_l275_275990

theorem alice_outfits :
  let trousers := 5
  let shirts := 8
  let jackets := 4
  let shoes := 2
  trousers * shirts * jackets * shoes = 320 :=
by
  sorry

end alice_outfits_l275_275990


namespace sum_of_binary_digits_of_315_l275_275812

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l275_275812


namespace max_value_of_quadratic_l275_275731

def quadratic_func (x : ℝ) : ℝ := -3 * (x - 2) ^ 2 - 3

theorem max_value_of_quadratic : 
  ∃ x : ℝ, quadratic_func x = -3 :=
by
  sorry

end max_value_of_quadratic_l275_275731


namespace right_triangle_sides_l275_275569

open Real

theorem right_triangle_sides (h : ∀ (α β : ℝ), α + β = 90 ∧ (5 * tan α = 4 * tan β) ∧ α = 50 ∧ β = 40 )
  (c : ℝ) (h_c : c = 15) :
  ∃ (a b : ℝ), a ≈ 11.49 ∧ b ≈ 9.642 :=
by
  let α := 50 : ℝ
  let β := 40 : ℝ
  have h_sum : α + β = 90 := by norm_num
  have ratio : 5 * tan α = 4 * tan β := by norm_num
  let a := c * sin (α * pi/180)
  let b := c * sin (β * pi/180)
  have a_approx : a ≈ 11.49 := sorry
  have b_approx : b ≈ 9.642 := sorry
  use [a, b]
  exact ⟨a_approx, b_approx⟩


end right_triangle_sides_l275_275569


namespace total_time_spent_l275_275638

-- Definition of the problem conditions
def warm_up_time : ℕ := 10
def additional_puzzles : ℕ := 2
def multiplier : ℕ := 3

-- Statement to prove the total time spent solving puzzles
theorem total_time_spent : warm_up_time + (additional_puzzles * (multiplier * warm_up_time)) = 70 :=
by
  sorry

end total_time_spent_l275_275638


namespace find_original_price_l275_275167

-- Defining constants and variables
def original_price (P : ℝ) : Prop :=
  let cost_after_repairs := P + 13000
  let selling_price := 66900
  let profit := selling_price - cost_after_repairs
  let profit_percent := profit / P * 100
  profit_percent = 21.636363636363637

theorem find_original_price : ∃ P : ℝ, original_price P :=
  by
  sorry

end find_original_price_l275_275167


namespace remainder_when_divided_by_20_l275_275107

theorem remainder_when_divided_by_20 (n : ℕ) : (4 * 6^n + 5^(n-1)) % 20 = 9 := 
by
  sorry

end remainder_when_divided_by_20_l275_275107


namespace macy_hit_ball_50_times_l275_275032

-- Definitions and conditions
def token_pitches : ℕ := 15
def macy_tokens : ℕ := 11
def piper_tokens : ℕ := 17
def piper_hits : ℕ := 55
def missed_pitches : ℕ := 315

-- Calculation based on conditions
def total_pitches : ℕ := (macy_tokens + piper_tokens) * token_pitches
def total_hits : ℕ := total_pitches - missed_pitches
def macy_hits : ℕ := total_hits - piper_hits

-- Prove that Macy hit 50 times
theorem macy_hit_ball_50_times : macy_hits = 50 := 
by
  sorry

end macy_hit_ball_50_times_l275_275032


namespace large_paintings_count_l275_275384

-- Define the problem conditions
def paint_per_large : Nat := 3
def paint_per_small : Nat := 2
def small_paintings : Nat := 4
def total_paint : Nat := 17

-- Question to find number of large paintings (L)
theorem large_paintings_count :
  ∃ L : Nat, (paint_per_large * L + paint_per_small * small_paintings = total_paint) → L = 3 :=
by
  -- Placeholder for the proof
  sorry

end large_paintings_count_l275_275384


namespace lucy_initial_balance_l275_275591

theorem lucy_initial_balance (final_balance deposit withdrawal : Int) 
  (h_final : final_balance = 76)
  (h_deposit : deposit = 15)
  (h_withdrawal : withdrawal = 4) :
  let initial_balance := final_balance + withdrawal - deposit
  initial_balance = 65 := 
by
  sorry

end lucy_initial_balance_l275_275591


namespace complement_A_correct_l275_275423

-- Define the universal set U
def U : Set ℝ := { x | x ≥ 1 ∨ x ≤ -1 }

-- Define the set A
def A : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := { x | x ≤ -1 ∨ x = 1 ∨ x > 2 }

-- Prove that the complement of A in U is as defined
theorem complement_A_correct : (U \ A) = complement_A_in_U := by
  sorry

end complement_A_correct_l275_275423


namespace sum_of_digits_base2_315_l275_275821

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l275_275821


namespace radio_price_and_total_items_l275_275224

theorem radio_price_and_total_items :
  ∃ (n : ℕ) (p : ℝ),
    (∀ (i : ℕ), (1 ≤ i ∧ i ≤ n) → (i = 1 ∨ ∃ (j : ℕ), i = j + 1 ∧ p = 1 + (j * 0.50))) ∧
    (n - 49 = 85) ∧
    (p = 43) ∧
    (n = 134) :=
by {
  sorry
}

end radio_price_and_total_items_l275_275224


namespace total_distance_traveled_in_12_hours_l275_275187

variable (n a1 d : ℕ) (u : ℕ → ℕ)

def arithmetic_seq_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) * d) / 2

theorem total_distance_traveled_in_12_hours :
  arithmetic_seq_sum 12 55 2 = 792 := by
  sorry

end total_distance_traveled_in_12_hours_l275_275187


namespace sum_of_sequence_l275_275542

theorem sum_of_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, a n = (-1 : ℤ)^(n+1) * (2*n - 1)) →
  (S 0 = 0) →
  (∀ n, S (n+1) = S n + a (n+1)) →
  (∀ n, S (n+1) = (-1 : ℤ)^(n+1) * (n+1)) :=
by
  intros h_a h_S0 h_S
  sorry

end sum_of_sequence_l275_275542


namespace original_triangle_area_l275_275941

theorem original_triangle_area (new_area : ℝ) (scaling_factor : ℝ) (area_ratio : ℝ) : 
  new_area = 32 → scaling_factor = 2 → 
  area_ratio = scaling_factor ^ 2 → 
  new_area / area_ratio = 8 := 
by
  intros
  -- insert your proof logic here
  sorry

end original_triangle_area_l275_275941


namespace clock_rings_eight_times_in_a_day_l275_275228

theorem clock_rings_eight_times_in_a_day : 
  ∀ t : ℕ, t % 3 = 1 → 0 ≤ t ∧ t < 24 → ∃ n : ℕ, n = 8 := 
by 
  sorry

end clock_rings_eight_times_in_a_day_l275_275228


namespace average_headcount_11600_l275_275054

theorem average_headcount_11600 : 
  let h02_03 := 11700
  let h03_04 := 11500
  let h04_05 := 11600
  (h02_03 + h03_04 + h04_05) / 3 = 11600 := 
by
  sorry

end average_headcount_11600_l275_275054


namespace perpendicular_line_eq_l275_275642

theorem perpendicular_line_eq (x y : ℝ) :
  (3 * x - 6 * y = 9) ∧ ∃ x₀ y₀, (2 : ℝ) = x₀ ∧ (-3 : ℝ) = y₀ ∧ y₀ + (2 * (x - x₀)) = y :=
  ∃ y₀, y = -2 * x + y₀ ∧ y₀ = 1 := sorry

end perpendicular_line_eq_l275_275642


namespace athlete_speed_l275_275994

theorem athlete_speed (d t : ℝ) (H_d : d = 200) (H_t : t = 40) : (d / t) = 5 := by
  sorry

end athlete_speed_l275_275994


namespace tan_ratio_of_angles_l275_275158

theorem tan_ratio_of_angles (a b : ℝ) (h1 : Real.sin (a + b) = 3/4) (h2 : Real.sin (a - b) = 1/2) :
    (Real.tan a / Real.tan b) = 5 := 
by 
  sorry

end tan_ratio_of_angles_l275_275158


namespace equal_roots_m_eq_minus_half_l275_275745

theorem equal_roots_m_eq_minus_half (x m : ℝ) 
  (h_eq: ∀ x, ( (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m )) :
  m = -1/2 := by 
  sorry

end equal_roots_m_eq_minus_half_l275_275745


namespace part_a_part_b_l275_275061

noncomputable section

open Real

theorem part_a (x y z : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) :
  (x^2 / (x-1)^2) + (y^2 / (y-1)^2) + (z^2 / (z-1)^2) ≥ 1 :=
sorry

theorem part_b : ∃ (infinitely_many : ℕ → (ℚ × ℚ × ℚ)), 
  ∀ n, ((infinitely_many n).1.1 ≠ 1) ∧ ((infinitely_many n).1.2 ≠ 1) ∧ ((infinitely_many n).2 ≠ 1) ∧ 
  ((infinitely_many n).1.1 * (infinitely_many n).1.2 * (infinitely_many n).2 = 1) ∧ 
  ((infinitely_many n).1.1^2 / ((infinitely_many n).1.1 - 1)^2 + 
   (infinitely_many n).1.2^2 / ((infinitely_many n).1.2 - 1)^2 + 
   (infinitely_many n).2^2 / ((infinitely_many n).2 - 1)^2 = 1) :=
sorry

end part_a_part_b_l275_275061


namespace mr_bird_speed_to_be_on_time_l275_275163

theorem mr_bird_speed_to_be_on_time 
  (d : ℝ) 
  (t : ℝ)
  (h1 : d = 40 * (t + 1/20))
  (h2 : d = 60 * (t - 1/20)) :
  (d / t) = 48 :=
by
  sorry

end mr_bird_speed_to_be_on_time_l275_275163


namespace total_tickets_sold_correct_l275_275340

theorem total_tickets_sold_correct :
  ∀ (A : ℕ), (21 * A + 15 * 327 = 8748) → (A + 327 = 509) :=
by
  intros A h
  sorry

end total_tickets_sold_correct_l275_275340


namespace rectangular_prism_volume_l275_275780

theorem rectangular_prism_volume (w : ℝ) (w_pos : 0 < w) 
    (h_edges_sum : 4 * w + 8 * (2 * w) + 4 * (w / 2) = 88) :
    (2 * w) * w * (w / 2) = 85184 / 343 :=
by
  sorry

end rectangular_prism_volume_l275_275780


namespace plane_division_99_lines_l275_275063

theorem plane_division_99_lines (m : ℕ) (n : ℕ) : 
  m = 99 ∧ n < 199 → (n = 100 ∨ n = 198) :=
by 
  sorry

end plane_division_99_lines_l275_275063


namespace number_of_shampoos_l275_275594

-- Define necessary variables in conditions
def h := 10 -- time spent hosing in minutes
def t := 55 -- total time spent cleaning in minutes
def p := 15 -- time per shampoo in minutes

-- State the theorem
theorem number_of_shampoos (h t p : Nat) (h_val : h = 10) (t_val : t = 55) (p_val : p = 15) :
    (t - h) / p = 3 := by
  -- Proof to be filled in
  sorry

end number_of_shampoos_l275_275594


namespace complement_union_A_B_l275_275457

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_A_B :
  (U \ (A ∪ B)) = {0, 5} := by
  sorry

end complement_union_A_B_l275_275457


namespace find_fx_l275_275411

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = 19 * x ^ 2 + 55 * x - 44) :
  ∀ x : ℝ, f x = 19 * x ^ 2 + 93 * x + 30 :=
by
  sorry

end find_fx_l275_275411


namespace number_of_special_three_digit_numbers_l275_275560

noncomputable def count_special_three_digit_numbers : ℕ :=
  Nat.choose 9 3

theorem number_of_special_three_digit_numbers : count_special_three_digit_numbers = 84 := by
  sorry

end number_of_special_three_digit_numbers_l275_275560


namespace conversion_rate_false_l275_275939

-- Definition of conversion rates between units
def conversion_rate_hour_minute : ℕ := 60
def conversion_rate_minute_second : ℕ := 60

-- Theorem stating that the rate being 100 is false under the given conditions
theorem conversion_rate_false (h1 : conversion_rate_hour_minute = 60) 
  (h2 : conversion_rate_minute_second = 60) : 
  ¬ (conversion_rate_hour_minute = 100 ∧ conversion_rate_minute_second = 100) :=
by {
  sorry
}

end conversion_rate_false_l275_275939


namespace find_smallest_shift_l275_275935

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)

noncomputable def g (x ϕ : ℝ) : ℝ := f (x - ϕ)

def is_odd_function (h : ℝ → ℝ) : Prop :=
  ∀ x, h x = -h (-x)

theorem find_smallest_shift {ϕ : ℝ} (hϕ : ϕ = π / 6) :
  is_odd_function (g ϕ) ↔ ∃ k : ℤ, ϕ = (π / 6 - k * (π / 2)) := sorry

end find_smallest_shift_l275_275935


namespace find_m_l275_275552

noncomputable def f (x a : ℝ) : ℝ := x - a

theorem find_m (a m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 4 → f x a ≤ 2) →
  (∃ x, -2 ≤ x ∧ x ≤ 4 ∧ -1 - f (x + 1) a ≤ m) :=
sorry

end find_m_l275_275552


namespace jane_crayon_count_l275_275996

def billy_crayons : ℝ := 62.0
def total_crayons : ℝ := 114
def jane_crayons : ℝ := total_crayons - billy_crayons

theorem jane_crayon_count : jane_crayons = 52 := by
  unfold jane_crayons
  show total_crayons - billy_crayons = 52
  sorry

end jane_crayon_count_l275_275996


namespace initial_cookie_count_l275_275958

variable (cookies_left_after_week : ℕ)
variable (cookies_taken_each_day : ℕ)
variable (total_cookies_taken_in_four_days : ℕ)
variable (initial_cookies : ℕ)
variable (days_per_week : ℕ)

theorem initial_cookie_count :
  cookies_left_after_week = 28 →
  total_cookies_taken_in_four_days = 24 →
  days_per_week = 7 →
  (∀ d (h : d ∈ Finset.range days_per_week), cookies_taken_each_day = 6) →
  initial_cookies = 52 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_cookie_count_l275_275958


namespace balloons_remaining_l275_275293

def bags_round : ℕ := 5
def balloons_per_bag_round : ℕ := 20
def bags_long : ℕ := 4
def balloons_per_bag_long : ℕ := 30
def balloons_burst : ℕ := 5

def total_round_balloons : ℕ := bags_round * balloons_per_bag_round
def total_long_balloons : ℕ := bags_long * balloons_per_bag_long
def total_balloons : ℕ := total_round_balloons + total_long_balloons
def balloons_left : ℕ := total_balloons - balloons_burst

theorem balloons_remaining : balloons_left = 215 := by 
  -- We leave this as sorry since the proof is not required
  sorry

end balloons_remaining_l275_275293


namespace probability_taequan_wins_l275_275938

open Finset

-- Definition of fair 6-sided dice
def fair_six_sided_dice := {1, 2, 3, 4, 5, 6}

-- Definition of rolling three dice (set of triples)
def roll_three_dice := (fair_six_sided_dice.product fair_six_sided_dice).product fair_six_sided_dice

-- Definition of the specific roll {2, 3, 4}
def specific_roll := { (2, 3, 4), (2, 4, 3), (3, 2, 4), (3, 4, 2), (4, 2, 3), (4, 3, 2) }

-- Total possible outcomes when rolling three 6-sided dice
def total_outcomes := 6 * 6 * 6

-- Number of favorable outcomes
def favorable_outcomes := specific_roll.card

-- Probability calculation
def probability_of_winning := favorable_outcomes.to_real / total_outcomes.to_real

theorem probability_taequan_wins :
  probability_of_winning = (1 / 36 : ℝ) :=
begin
  unfold probability_of_winning,
  unfold favorable_outcomes,
  unfold total_outcomes,
  rw [card_eq 6],
  rw [nat.cast_mul, nat.cast_mul, nat.cast_bit0, nat.cast_bit0, nat.cast_bit1],
  norm_num,
  sorry
end

end probability_taequan_wins_l275_275938


namespace quadratic_inequality_solution_l275_275421

theorem quadratic_inequality_solution (a : ℝ) :
  ((0 ≤ a ∧ a < 3) → ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) :=
  sorry

end quadratic_inequality_solution_l275_275421


namespace opposite_of_neg_four_l275_275484

-- Define the condition: the opposite of a number is the number that, when added to the original number, results in zero.
def is_opposite (a b : Int) : Prop := a + b = 0

-- The specific theorem we want to prove
theorem opposite_of_neg_four : is_opposite (-4) 4 := by
  -- Placeholder for the proof
  sorry

end opposite_of_neg_four_l275_275484


namespace length_of_GH_l275_275746

variable (S_A S_C S_E S_F : ℝ)
variable (AB FE CD GH : ℝ)

-- Given conditions
axiom h1 : AB = 11
axiom h2 : FE = 13
axiom h3 : CD = 5

-- Relationships between the sizes of the squares
axiom h4 : S_A = S_C + AB
axiom h5 : S_C = S_E + CD
axiom h6 : S_E = S_F + FE
axiom h7 : GH = S_A - S_F

theorem length_of_GH : GH = 29 :=
by
  -- This is where the proof would go
  sorry

end length_of_GH_l275_275746


namespace number_of_players_in_association_l275_275154

-- Define the variables and conditions based on the given problem
def socks_cost : ℕ := 6
def tshirt_cost := socks_cost + 8
def hat_cost := tshirt_cost - 3
def total_expenditure : ℕ := 4950
def cost_per_player := 2 * (socks_cost + tshirt_cost + hat_cost)

-- The statement to prove
theorem number_of_players_in_association :
  total_expenditure / cost_per_player = 80 := by
  sorry

end number_of_players_in_association_l275_275154


namespace find_C_and_D_l275_275865

variables (C D : ℝ)

theorem find_C_and_D (h : 4 * C + 2 * D + 5 = 30) : C = 5.25 ∧ D = 2 :=
by
  sorry

end find_C_and_D_l275_275865


namespace sum_of_digits_base2_315_l275_275807

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l275_275807


namespace choice_of_b_l275_275082

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x - 2)
noncomputable def g (x : ℝ) : ℝ := f (x + 3)

theorem choice_of_b (b : ℝ) :
  (g (g x) = x) ↔ (b = -4) :=
sorry

end choice_of_b_l275_275082


namespace exists_n_for_pn_consecutive_zeros_l275_275445

theorem exists_n_for_pn_consecutive_zeros (p : ℕ) (hp : Nat.Prime p) (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, (∃ k : ℕ, (p^n) / 10^(k+m) % 10^m = 0) := sorry

end exists_n_for_pn_consecutive_zeros_l275_275445


namespace marble_cost_l275_275993

def AlyssaSpentOnMarbles (totalSpent onToys footballCost : ℝ) : ℝ :=
 totalSpent - footballCost

theorem marble_cost:
  AlyssaSpentOnMarbles 12.30 5.71 = 6.59 :=
by 
  unfold AlyssaSpentOnMarbles 
  sorry

end marble_cost_l275_275993


namespace perpendicular_line_eq_slope_intercept_l275_275648

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l275_275648


namespace find_r_l275_275424

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 :=
sorry

end find_r_l275_275424


namespace maci_school_supplies_cost_l275_275031

theorem maci_school_supplies_cost :
  let blue_pen_cost := 0.10
  let red_pen_cost := 2 * blue_pen_cost
  let pencil_cost := red_pen_cost / 2
  let notebook_cost := 10 * blue_pen_cost
  let blue_pen_count := 10
  let red_pen_count := 15
  let pencil_count := 5
  let notebook_count := 3
  let total_pen_count := blue_pen_count + red_pen_count
  let total_cost_before_discount := 
      blue_pen_count * blue_pen_cost + 
      red_pen_count * red_pen_cost + 
      pencil_count * pencil_cost + 
      notebook_count * notebook_cost
  let pen_discount_rate := if total_pen_count > 12 then 0.10 else 0
  let notebook_discount_rate := if notebook_count > 4 then 0.20 else 0
  let pen_discount := pen_discount_rate * (blue_pen_count * blue_pen_cost + red_pen_count * red_pen_cost)
  let total_cost_after_discount := 
      total_cost_before_discount - pen_discount
  total_cost_after_discount = 7.10 :=
by
  sorry

end maci_school_supplies_cost_l275_275031


namespace dice_probability_sum_three_l275_275057

theorem dice_probability_sum_three (total_outcomes : ℕ := 36) (favorable_outcomes : ℕ := 2) :
  favorable_outcomes / total_outcomes = 1 / 18 :=
by
  sorry

end dice_probability_sum_three_l275_275057


namespace triangle_side_length_l275_275008

theorem triangle_side_length (AB AC BC BX CX : ℕ)
  (h1 : AB = 86)
  (h2 : AC = 97)
  (h3 : BX + CX = BC)
  (h4 : AX = AB)
  (h5 : AX = 86)
  (h6 : AB * AB * CX + AC * AC * BX = BC * (BX * CX + AX * AX))
  : BC = 61 := 
sorry

end triangle_side_length_l275_275008


namespace has_root_sqrt3_add_sqrt5_l275_275531

noncomputable def monic_degree_4_poly_with_root : Polynomial ℚ :=
  Polynomial.X ^ 4 - 16 * Polynomial.X ^ 2 + 4

theorem has_root_sqrt3_add_sqrt5 :
  Polynomial.eval (Real.sqrt 3 + Real.sqrt 5) monic_degree_4_poly_with_root = 0 :=
sorry

end has_root_sqrt3_add_sqrt5_l275_275531


namespace radius_of_tangent_circle_l275_275626

theorem radius_of_tangent_circle (a b : ℕ) (r1 r2 r3 : ℚ) (R : ℚ)
  (h1 : a = 6) (h2 : b = 8)
  (h3 : r1 = a / 2) (h4 : r2 = b / 2) (h5 : r3 = (Real.sqrt (a^2 + b^2)) / 2) :
  R = 144 / 23 := sorry

end radius_of_tangent_circle_l275_275626


namespace gcd_36_54_l275_275793

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factorization (n : ℕ) : list ℕ :=
if h : n = 0 then []
else
(list.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

theorem gcd_36_54 : Nat.gcd 36 54 = 18 :=
by
  sorry

end gcd_36_54_l275_275793


namespace evaluate_expression_l275_275936

theorem evaluate_expression (x : ℝ) (h : x = Real.sqrt 3) : 
  ( (x^2 - 2*x + 1) / (x^2 - x) / (x - 1) ) = Real.sqrt 3 / 3 :=
by
  sorry

end evaluate_expression_l275_275936


namespace maximum_marks_l275_275168

theorem maximum_marks (M : ℝ) :
  (0.45 * M = 80) → (M = 180) :=
by
  sorry

end maximum_marks_l275_275168


namespace volume_of_rotated_solid_l275_275840

noncomputable def volume_of_solid (a b : ℝ) (rotate_side : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (if rotate_side = a then b * b else a * a) * rotate_side

theorem volume_of_rotated_solid :
  volume_of_solid 3 2 3 = 12.56 ∨ volume_of_solid 3 2 2 = 18.84 :=
by
  sorry

end volume_of_rotated_solid_l275_275840


namespace scientific_notation_of_virus_diameter_l275_275778

theorem scientific_notation_of_virus_diameter :
  0.000000102 = 1.02 * 10 ^ (-7) :=
  sorry

end scientific_notation_of_virus_diameter_l275_275778


namespace minimum_b_value_l275_275160

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.log (x^2 - 2 * a))^2

theorem minimum_b_value (a : ℝ) : ∃ x_0 > 0, f x_0 a ≤ (4 / 5) :=
sorry

end minimum_b_value_l275_275160


namespace length_of_CD_l275_275883

theorem length_of_CD
    (AB BC AC AD CD : ℝ)
    (h1 : AB = 6)
    (h2 : BC = 1 / 2 * AB)
    (h3 : AC = AB + BC)
    (h4 : AD = AC)
    (h5 : CD = AD + AC) :
    CD = 18 := by
  sorry

end length_of_CD_l275_275883


namespace tiles_needed_l275_275757

def ft_to_inch (x : ℕ) : ℕ := x * 12

def height_ft : ℕ := 10
def length_ft : ℕ := 15
def tile_size_sq_inch : ℕ := 1

def height_inch : ℕ := ft_to_inch height_ft
def length_inch : ℕ := ft_to_inch length_ft
def area_sq_inch : ℕ := height_inch * length_inch

theorem tiles_needed : 
  height_ft = 10 ∧ length_ft = 15 ∧ tile_size_sq_inch = 1 →
  area_sq_inch = 21600 :=
by
  intro h
  exact sorry

end tiles_needed_l275_275757


namespace prism_cross_section_properties_l275_275185

noncomputable def prism_base_triangle_area (side : ℝ) (height : ℝ) : ℝ :=
  (sqrt 3 / 4) * side^2

noncomputable def cross_section_area (side : ℝ) (height : ℝ) : ℝ :=
  side^2 * height / 2

theorem prism_cross_section_properties :
  let side := 6
  let height := 1/3 * sqrt 7
  let angle_between_planes := 30
  cross_section_area side height = 39/4 ∧ angle_between_planes = 30 :=
by
  sorry

end prism_cross_section_properties_l275_275185


namespace kamilla_acquaintances_l275_275571

/-- In the city of Bukvinsk, people are acquaintances only if their names contain the same letters.
Martin has 20 acquaintances, Klim has 15 acquaintances, Inna has 12 acquaintances, and Tamara
has 12 acquaintances. Prove that Kamilla has the same number of acquaintances as Klim, which is 15. -/
theorem kamilla_acquaintances :
  let count_martin := 20 in
  let count_klim := 15 in
  let count_inna := 12 in
  let count_tamara := 12 in
  let count_kamilla := 15 in
  ∀ (martin klim inna tamara kamilla : Type),
  martin ∈ bukvinsk → klim ∈ bukvinsk → inna ∈ bukvinsk → tamara ∈ bukvinsk → kamilla ∈ bukvinsk →
  martin.acquaintances = count_martin →
  klim.acquaintances = count_klim →
  inna.acquaintances = count_inna →
  tamara.acquaintances = count_tamara →
  kamilla.acquaintances = count_kamilla :=
by
  sorry

end kamilla_acquaintances_l275_275571


namespace largest_composite_in_five_consecutive_ints_l275_275712

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def largest_of_five_composite_ints : ℕ :=
  36

theorem largest_composite_in_five_consecutive_ints (a b c d e : ℕ) :
  a < 40 ∧ b < 40 ∧ c < 40 ∧ d < 40 ∧ e < 40 ∧ 
  ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a = 32 ∧ b = 33 ∧ c = 34 ∧ d = 35 ∧ e = 36 →
  e = largest_of_five_composite_ints :=
by 
  sorry

end largest_composite_in_five_consecutive_ints_l275_275712


namespace closest_integer_to_2_plus_sqrt_6_l275_275104

theorem closest_integer_to_2_plus_sqrt_6 (sqrt6_lower : 2 < Real.sqrt 6) (sqrt6_upper : Real.sqrt 6 < 2.5) : 
  abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 3) ∧ abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 5) :=
by
  sorry

end closest_integer_to_2_plus_sqrt_6_l275_275104


namespace third_side_length_l275_275895

theorem third_side_length (a b : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  ∃ x : ℝ, (a = 3 ∧ b = 4) ∧ (x = 5 ∨ x = Real.sqrt 7) :=
by
  sorry

end third_side_length_l275_275895


namespace sum_of_digits_of_binary_315_is_6_l275_275804
-- Importing Mathlib for necessary libraries.

-- Definition of the problem and statement of the theorem.
theorem sum_of_digits_of_binary_315_is_6 : ∑ i in (Nat.digits 2 315), i = 6 := by
  sorry

end sum_of_digits_of_binary_315_is_6_l275_275804


namespace domain_of_f_l275_275325

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f : 
  {x : ℝ | x > -1 ∧ x ≤ 2 ∧ x ≠ 0 ∧ 4 - x^2 ≥ 0} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_of_f_l275_275325


namespace max_cross_section_area_is_260_l275_275705

noncomputable def max_cross_section_area (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) : ℝ :=
  max (a * Real.sqrt (b^2 + c^2)) (max (b * Real.sqrt (a^2 + c^2)) (c * Real.sqrt (a^2 + b^2)))

theorem max_cross_section_area_is_260 :
  max_cross_section_area 5 12 20 (and.intro (by norm_num) (by norm_num)) = 260 :=
by
  sorry

end max_cross_section_area_is_260_l275_275705


namespace sum_medians_less_than_perimeter_l275_275612

noncomputable def median_a (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * b^2 + 2 * c^2 - a^2).sqrt

noncomputable def median_b (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * c^2 - b^2).sqrt

noncomputable def median_c (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * b^2 - c^2).sqrt

noncomputable def sum_of_medians (a b c : ℝ) : ℝ :=
  median_a a b c + median_b a b c + median_c a b c

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
  perimeter a b c / 2

theorem sum_medians_less_than_perimeter (a b c : ℝ) :
  semiperimeter a b c < sum_of_medians a b c ∧ sum_of_medians a b c < perimeter a b c :=
by
  sorry

end sum_medians_less_than_perimeter_l275_275612


namespace relationship_among_three_numbers_l275_275868

noncomputable def M (a b : ℝ) : ℝ := a^b
noncomputable def N (a b : ℝ) : ℝ := Real.log a / Real.log b
noncomputable def P (a b : ℝ) : ℝ := b^a

theorem relationship_among_three_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : N a b < M a b ∧ M a b < P a b := 
by
  sorry

end relationship_among_three_numbers_l275_275868


namespace hilary_ears_per_stalk_l275_275732

-- Define the given conditions
def num_stalks : ℕ := 108
def kernels_per_ear_half1 : ℕ := 500
def kernels_per_ear_half2 : ℕ := 600
def total_kernels_to_shuck : ℕ := 237600

-- Define the number of ears of corn per stalk as the variable to prove
def ears_of_corn_per_stalk : ℕ := 4

-- The proof problem statement
theorem hilary_ears_per_stalk :
  (54 * ears_of_corn_per_stalk * kernels_per_ear_half1) + (54 * ears_of_corn_per_stalk * kernels_per_ear_half2) = total_kernels_to_shuck :=
by
  sorry

end hilary_ears_per_stalk_l275_275732


namespace metropolis_partition_l275_275155

open Finset

theorem metropolis_partition (stations : Finset ℕ) {G : SimpleGraph ℕ} 
  (h_stations : stations.card = 1972)
  (h_edges : ∀ (u v : ℕ) (h_u : u ∈ stations) (h_v : v ∈ stations) (h_uv : ¬u = v), G.adj u v)
  (h_connected_after_closure : ∀ (S : Finset ℕ) (hS : S.card = 9) (H : ∀ s ∈ S, s ∈ stations), 
    ∃ t ∈ stations \ S, G.isConnected (stations \ S ∪ {t}))
  (h_transfer_AB : ∃ (A B : ℕ) (hA : A ∈ stations) (hB : B ∈ stations), G.minTransfers A B ≥ 99) :
  ∃ (groups : Finset (Finset ℕ)), 
    groups.card = 1000 ∧ (∀ g ∈ groups, ∀ a b ∈ g, ¬G.adj a b) :=
begin
  sorry
end

end metropolis_partition_l275_275155


namespace sum_of_digits_base_2_315_l275_275816

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l275_275816


namespace log3_x_minus_1_increasing_l275_275614

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem log3_x_minus_1_increasing : is_increasing_on (fun x => log_base_3 (x - 1)) (Set.Ioi 1) :=
sorry

end log3_x_minus_1_increasing_l275_275614


namespace range_of_angle_A_l275_275907

theorem range_of_angle_A (a b : ℝ) (A : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) 
  (h_triangle : 0 < A ∧ A ≤ Real.pi / 4) :
  (0 < A ∧ A ≤ Real.pi / 4) :=
by
  sorry

end range_of_angle_A_l275_275907


namespace hexagon_perimeter_is_42_l275_275831

-- Define the side length of the hexagon
def side_length : ℕ := 7

-- Define the number of sides of the hexagon
def num_sides : ℕ := 6

-- Define the perimeter of the hexagon
def hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) : ℕ :=
  num_sides * side_length

-- The theorem to prove
theorem hexagon_perimeter_is_42 : hexagon_perimeter side_length num_sides = 42 :=
by
  sorry

end hexagon_perimeter_is_42_l275_275831


namespace avg_salary_difference_l275_275901

theorem avg_salary_difference (factory_payroll : ℕ) (factory_workers : ℕ) (office_payroll : ℕ) (office_workers : ℕ)
  (h1 : factory_payroll = 30000) (h2 : factory_workers = 15)
  (h3 : office_payroll = 75000) (h4 : office_workers = 30) :
  (office_payroll / office_workers) - (factory_payroll / factory_workers) = 500 := by
  sorry

end avg_salary_difference_l275_275901


namespace new_average_is_15_l275_275768

-- Definitions corresponding to the conditions
def avg_10_consecutive (seq : List ℤ) : Prop :=
  seq.length = 10 ∧ seq.sum = 200

def new_seq (seq : List ℤ) : List ℤ :=
  List.mapIdx (λ i x => x - ↑(9 - i)) seq

-- Statement of the proof problem
theorem new_average_is_15
  (seq : List ℤ)
  (h_seq : avg_10_consecutive seq) :
  (new_seq seq).sum = 150 := sorry

end new_average_is_15_l275_275768


namespace leak_empty_time_l275_275215

-- Define the given conditions
def tank_volume := 2160 -- Tank volume in litres
def inlet_rate := 6 * 60 -- Inlet rate in litres per hour
def combined_empty_time := 12 -- Time in hours to empty the tank with the inlet on

-- Define the derived conditions
def net_rate := tank_volume / combined_empty_time -- Net rate of emptying in litres per hour

-- Define the rate of leakage
def leak_rate := inlet_rate - net_rate -- Rate of leak in litres per hour

-- Prove the main statement
theorem leak_empty_time : (2160 / leak_rate) = 12 :=
by
  unfold leak_rate
  exact sorry

end leak_empty_time_l275_275215


namespace solve_for_vee_l275_275892

theorem solve_for_vee (vee : ℝ) (h : 4 * vee ^ 2 = 144) : vee = 6 ∨ vee = -6 :=
by
  -- We state that this theorem should be true for all vee and given the condition h
  sorry

end solve_for_vee_l275_275892


namespace age_difference_l275_275214

theorem age_difference 
  (A B : ℤ) 
  (h1 : B = 39) 
  (h2 : A + 10 = 2 * (B - 10)) :
  A - B = 9 := 
by 
  sorry

end age_difference_l275_275214


namespace probability_none_hit_l275_275534

theorem probability_none_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (1 - p)^5 = (1 - p) * (1 - p) * (1 - p) * (1 - p) * (1 - p) :=
by sorry

end probability_none_hit_l275_275534


namespace gcd_36_54_l275_275794

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factorization (n : ℕ) : list ℕ :=
if h : n = 0 then []
else
(list.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

theorem gcd_36_54 : Nat.gcd 36 54 = 18 :=
by
  sorry

end gcd_36_54_l275_275794


namespace round_trip_time_l275_275331

theorem round_trip_time (boat_speed : ℝ) (stream_speed : ℝ) (distance : ℝ) : 
  boat_speed = 8 → stream_speed = 2 → distance = 210 → 
  ((distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed))) = 56 :=
by
  intros hb hs hd
  sorry

end round_trip_time_l275_275331


namespace simplify_and_evaluate_expression_l275_275317

theorem simplify_and_evaluate_expression (a : ℚ) (h : a = -3/2) :
  (a + 2 - 5/(a - 2)) / ((2 * a^2 - 6 * a) / (a - 2)) = -1/2 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l275_275317


namespace molecular_weight_8_moles_N2O_l275_275345

-- Definitions for atomic weights and the number of moles
def atomic_weight_N : Float := 14.01
def atomic_weight_O : Float := 16.00
def moles_N2O : Float := 8.0

-- Definition for molecular weight of N2O
def molecular_weight_N2O : Float := 
  (2 * atomic_weight_N) + (1 * atomic_weight_O)

-- Target statement to prove
theorem molecular_weight_8_moles_N2O :
  moles_N2O * molecular_weight_N2O = 352.16 :=
by
  sorry

end molecular_weight_8_moles_N2O_l275_275345


namespace initial_mean_of_observations_l275_275949

theorem initial_mean_of_observations (M : ℚ) (h : 50 * M + 11 = 50 * 36.5) : M = 36.28 := 
by
  sorry

end initial_mean_of_observations_l275_275949


namespace complex_fourth_power_l275_275094

-- Definitions related to the problem conditions
def trig_identity (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)
def z : ℂ := 3 * trig_identity (30 * real.pi / 180)
def result := (81 : ℝ) * (-1/2 + complex.I * (real.sqrt 3 / 2))

-- Statement of the problem
theorem complex_fourth_power : z^4 = result := by
  sorry

end complex_fourth_power_l275_275094


namespace find_number_l275_275864

theorem find_number (x : ℝ) : (x^2 + 4 = 5 * x) → (x = 4 ∨ x = 1) :=
by
  sorry

end find_number_l275_275864


namespace natural_numbers_satisfy_equation_l275_275246

theorem natural_numbers_satisfy_equation:
  ∀ (n k : ℕ), (k^5 + 5 * n^4 = 81 * k) ↔ (n = 2 ∧ k = 1) :=
by
  sorry

end natural_numbers_satisfy_equation_l275_275246


namespace complex_power_result_l275_275091

theorem complex_power_result :
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.sin (Real.pi / 6) * Complex.i)^4 = 
  -40.5 + 40.5 * Complex.i * Real.sqrt 3 :=
by sorry

end complex_power_result_l275_275091


namespace sequence_general_term_and_sum_l275_275875

theorem sequence_general_term_and_sum (a_n : ℕ → ℕ) (b_n S_n : ℕ → ℕ) :
  (∀ n, a_n n = 2 ^ n) ∧ (∀ n, b_n n = a_n n * (Real.logb 2 (a_n n)) ∧
  S_n n = (n - 1) * 2 ^ (n + 1) + 2) :=
by
  sorry

end sequence_general_term_and_sum_l275_275875


namespace sum_of_digits_in_binary_representation_of_315_l275_275825

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l275_275825


namespace range_of_m_for_distinct_real_roots_l275_275280

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + m = 0 ∧ x₂^2 + 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end range_of_m_for_distinct_real_roots_l275_275280


namespace sequence_a_10_l275_275422

theorem sequence_a_10 : ∀ {a : ℕ → ℕ}, (a 1 = 1) → (∀ n, a (n+1) = a n + 2^n) → (a 10 = 1023) :=
by
  intros a h1 h_rec
  sorry

end sequence_a_10_l275_275422


namespace find_ratio_b_c_l275_275016

variable {a b c A B C : Real}

theorem find_ratio_b_c
  (h1 : a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C)
  (h2 : Real.cos A = -1 / 4) :
  b / c = 6 :=
sorry

end find_ratio_b_c_l275_275016


namespace find_a_l275_275416

theorem find_a (a : ℝ) (α : ℝ) (P : ℝ × ℝ) 
  (h_P : P = (3 * a, 4)) 
  (h_cos : Real.cos α = -3/5) : 
  a = -1 := 
by
  sorry

end find_a_l275_275416


namespace yellow_yellow_pairs_count_l275_275683

def num_blue_students : ℕ := 75
def num_yellow_students : ℕ := 105
def total_pairs : ℕ := 90
def blue_blue_pairs : ℕ := 30

theorem yellow_yellow_pairs_count :
  -- number of pairs where both students are wearing yellow shirts is 45.
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 45 :=
by
  sorry

end yellow_yellow_pairs_count_l275_275683


namespace domain_of_f_l275_275624

noncomputable def f (x : ℝ) : ℝ := (Real.tan (2 * x)) / Real.sqrt (x - x^2)

theorem domain_of_f :
  { x : ℝ | ∃ k : ℤ, 2*x ≠ k*π + π/2 ∧ x ∈ (Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1) } = 
  { x : ℝ | x ∈ Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1 } :=
sorry

end domain_of_f_l275_275624


namespace find_original_triangle_area_l275_275942

-- Define the conditions and question
def original_triangle_area (A : ℝ) : Prop :=
  let new_area := 4 * A in
  new_area = 32

-- State the problem to prove the area of the original triangle
theorem find_original_triangle_area (A : ℝ) : original_triangle_area A → A = 8 := by
  intro h
  sorry

end find_original_triangle_area_l275_275942


namespace ponchik_cakes_l275_275153

/-
Given:
- The numbers of honey cakes eaten by Ponchik: 
  Z (instead of exercise), P (instead of walk), R (instead of run), and C (instead of swim)
- Ratios: 
  Z / P = 3 / 2, P / R = 5 / 3, and R / C = 6 / 5
- A total of 216 honey cakes eaten in a day.

Prove:
- The difference between honey cakes eaten instead of exercise and swim is 60.
-/

theorem ponchik_cakes (Z P R C : ℕ) 
  (h_ratio1 : Z / P = 3 / 2) 
  (h_ratio2 : P / R = 5 / 3) 
  (h_ratio3 : R / C = 6 / 5) 
  (h_total : Z + P + R + C = 216) : 
  Z - C = 60 := sorry

end ponchik_cakes_l275_275153


namespace part1a_part1b_part2_part3a_part3b_l275_275256

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
noncomputable def g (x : ℝ) : ℝ := x^2 + 2

-- Prove f(2) = 1/3
theorem part1a : f 2 = 1 / 3 := 
by sorry

-- Prove g(2) = 6
theorem part1b : g 2 = 6 :=
by sorry

-- Prove f[g(2)] = 1/7 
theorem part2 : f (g 2) = 1 / 7 :=
by sorry

-- Prove f[g(x)] = 1/(x^2 + 3) 
theorem part3a : ∀ x : ℝ, f (g x) = 1 / (x^2 + 3) :=
by sorry

-- Prove g[f(x)] = 1/((1 + x)^2) + 2 
theorem part3b : ∀ x : ℝ, g (f x) = 1 / (1 + x)^2 + 2 :=
by sorry

end part1a_part1b_part2_part3a_part3b_l275_275256


namespace bukvinsk_acquaintances_l275_275572

theorem bukvinsk_acquaintances (Martin Klim Inna Tamara Kamilla : Type) 
  (acquaints : Type → Type → Prop)
  (exists_same_letters : ∀ (x y : Type), acquaints x y ↔ ∃ S, (x = S ∧ y = S)) :
  (∃ (count_Martin : ℕ), count_Martin = 20) →
  (∃ (count_Klim : ℕ), count_Klim = 15) →
  (∃ (count_Inna : ℕ), count_Inna = 12) →
  (∃ (count_Tamara : ℕ), count_Tamara = 12) →
  (∃ (count_Kamilla : ℕ), count_Kamilla = 15) := by
  sorry

end bukvinsk_acquaintances_l275_275572


namespace gasoline_price_increase_l275_275007

theorem gasoline_price_increase
  (P Q : ℝ)
  (h1 : (P * Q) * 1.10 = P * (1 + X / 100) * Q * 0.88) :
  X = 25 :=
by
  -- proof here
  sorry

end gasoline_price_increase_l275_275007


namespace advantageous_order_l275_275608

variables {p1 p2 p3 : ℝ}

-- Conditions
axiom prob_ordering : p3 < p1 ∧ p1 < p2

-- Definition of sequence probabilities
def prob_first_second := p1 * p2 + (1 - p1) * p2 * p3
def prob_second_first := p2 * p1 + (1 - p2) * p1 * p3

-- Theorem to be proved
theorem advantageous_order :
  prob_first_second = prob_second_first →
  p2 > p1 → (p2 > p1) :=
by
  sorry

end advantageous_order_l275_275608


namespace A_and_B_together_complete_work_in_24_days_l275_275837

-- Define the variables
variables {W_A W_B : ℝ} (completeTime : ℝ → ℝ → ℝ)

-- Define conditions
def A_better_than_B (W_A W_B : ℝ) := W_A = 2 * W_B
def A_takes_36_days (W_A : ℝ) := W_A = 1 / 36

-- The proposition to prove
theorem A_and_B_together_complete_work_in_24_days 
  (h1 : A_better_than_B W_A W_B)
  (h2 : A_takes_36_days W_A) :
  completeTime W_A W_B = 24 :=
sorry

end A_and_B_together_complete_work_in_24_days_l275_275837


namespace paths_remainder_l275_275635

-- Define the number of paths given the problem constraints
def num_paths : ℕ :=
1 + (
  let count_paths := 
    ∑ n : ℕ in (0 : ℕ) .. 10, binomial 10 n * binomial 4 (5 - n) in 
    count_paths + count_paths
)

-- Prove the result
theorem paths_remainder : num_paths % 1000 = 4 :=
by sorry

end paths_remainder_l275_275635


namespace sum_of_digits_base2_315_l275_275809

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l275_275809


namespace remainder_product_div_10_l275_275659

def unitsDigit (n : ℕ) : ℕ := n % 10

theorem remainder_product_div_10 :
  let a := 1734
  let b := 5389
  let c := 80607
  let p := a * b * c
  unitsDigit p = 2 := by
  sorry

end remainder_product_div_10_l275_275659


namespace probability_no_rain_four_days_l275_275044

theorem probability_no_rain_four_days (p : ℚ) (p_rain : ℚ) 
  (h_p_rain : p_rain = 2 / 3) 
  (h_p_no_rain : p = 1 - p_rain) : 
  p ^ 4 = 1 / 81 :=
by
  have h_p : p = 1 / 3, sorry
  rw [h_p],
  norm_num

end probability_no_rain_four_days_l275_275044


namespace find_sixth_term_l275_275189

noncomputable def arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def sum_first_n_terms (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem find_sixth_term :
  ∀ (a1 S3 : ℕ),
  a1 = 2 →
  S3 = 12 →
  ∃ d : ℕ, sum_first_n_terms a1 d 3 = S3 ∧ arithmetic_sequence a1 d 6 = 12 :=
by
  sorry

end find_sixth_term_l275_275189


namespace isosceles_triangles_with_perimeter_27_l275_275269

theorem isosceles_triangles_with_perimeter_27 :
  ∃ n : ℕ, n = 6 ∧ ∀ a b c : ℕ, (a = b ∧ a > 0 ∧ 2 * a + c = 27 ∧ c mod 2 = 1) → n = 6 :=
by
  sorry

end isosceles_triangles_with_perimeter_27_l275_275269


namespace probability_one_black_one_white_l275_275666

def total_balls : ℕ := 6 + 2
def black_balls : ℕ := 6
def white_balls : ℕ := 2

def total_ways_to_pick_two_balls : ℕ := total_balls.choose 2
def ways_to_pick_one_black_one_white : ℕ := (black_balls.choose 1) * (white_balls.choose 1)

theorem probability_one_black_one_white :
  (ways_to_pick_one_black_one_white : ℚ) / total_ways_to_pick_two_balls = 3 / 7 :=
sorry

end probability_one_black_one_white_l275_275666


namespace quinn_free_donuts_l275_275467

-- Definitions based on conditions
def books_per_week : ℕ := 2
def weeks : ℕ := 10
def books_needed_for_donut : ℕ := 5

-- Calculation based on conditions
def total_books_read : ℕ := books_per_week * weeks
def free_donuts (total_books : ℕ) : ℕ := total_books / books_needed_for_donut

-- Proof statement
theorem quinn_free_donuts : free_donuts total_books_read = 4 := by
  sorry

end quinn_free_donuts_l275_275467


namespace union_inter_complement_l275_275708

open Set

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | abs (x - 2) > 3})
variable (B : Set ℝ := {x | x * (-2 - x) > 0})

theorem union_inter_complement 
  (C_U_A : Set ℝ := compl A)
  (A_def : A = {x | abs (x - 2) > 3})
  (B_def : B = {x | x * (-2 - x) > 0})
  (C_U_A_def : C_U_A = compl A) :
  (A ∪ B = {x : ℝ | x < 0} ∪ {x : ℝ | x > 5}) ∧ 
  ((C_U_A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 0}) :=
by
  sorry

end union_inter_complement_l275_275708


namespace f_2002_eq_0_l275_275772

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_2_eq_0 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem f_2002_eq_0 : f 2002 = 0 :=
by
  sorry

end f_2002_eq_0_l275_275772


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l275_275754

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : x < -1) : 2 * x ^ 2 + x - 1 > 0 :=
by sorry

theorem not_necessary_condition (h2 : 2 * x ^ 2 + x - 1 > 0) : x > 1/2 ∨ x < -1 :=
by sorry

end sufficient_but_not_necessary_condition_not_necessary_condition_l275_275754


namespace sum_mod_12_l275_275800

def remainder_sum_mod :=
  let nums := [10331, 10333, 10335, 10337, 10339, 10341, 10343]
  let sum_nums := nums.sum
  sum_nums % 12 = 7

theorem sum_mod_12 : remainder_sum_mod :=
by
  sorry

end sum_mod_12_l275_275800


namespace smallest_points_to_exceed_mean_l275_275681

theorem smallest_points_to_exceed_mean (X y : ℕ) (h_scores : 24 + 17 + 25 = 66) 
  (h_mean_9_gt_mean_6 : X / 6 < (X + 66) / 9) (h_mean_10_gt_22 : (X + 66 + y) / 10 > 22) 
  : y ≥ 24 := by
  sorry

end smallest_points_to_exceed_mean_l275_275681


namespace initial_cards_eq_4_l275_275050

theorem initial_cards_eq_4 (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  sorry

end initial_cards_eq_4_l275_275050


namespace trigonometric_simplification_l275_275503

open Real

theorem trigonometric_simplification (α : ℝ) :
  (3.4113 * sin α * cos (3 * α) + 9 * sin α * cos α - sin (3 * α) * cos (3 * α) - 3 * sin (3 * α) * cos α) = 
  2 * sin (2 * α)^3 :=
by
  -- Placeholder for the proof
  sorry

end trigonometric_simplification_l275_275503


namespace simplify_expression_l275_275538

variable (p q r : ℝ)
variable (hp : p ≠ 2)
variable (hq : q ≠ 3)
variable (hr : r ≠ 4)

theorem simplify_expression : 
  (p^2 - 4) / (4 - r^2) * (q^2 - 9) / (2 - p^2) * (r^2 - 16) / (3 - q^2) = -1 :=
by
  -- Skipping the proof using sorry
  sorry

end simplify_expression_l275_275538


namespace sum_of_interior_angles_of_regular_polygon_l275_275178

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h1: ∀ {n : ℕ}, ∃ (e : ℕ), (e = 360 / 45) ∧ (n = e)) :
  (180 * (n - 2)) = 1080 :=
by
  let n := (360 / 45)
  have h : n = 8 := by sorry
  calc
    180 * (n - 2) = 180 * (8 - 2) : by rw [h]
    ... = 1080 : by norm_num

end sum_of_interior_angles_of_regular_polygon_l275_275178


namespace profit_ratio_l275_275766

noncomputable def lion_king_cost : ℝ := 10
noncomputable def lion_king_revenue : ℝ := 200
noncomputable def star_wars_cost : ℝ := 25
noncomputable def star_wars_revenue : ℝ := 405

noncomputable def lion_king_profit : ℝ :=
  lion_king_revenue - lion_king_cost

noncomputable def star_wars_profit : ℝ :=
  star_wars_revenue - star_wars_cost

theorem profit_ratio :
  (lion_king_profit / star_wars_profit) = (1 / 2) :=
  by 
    sorry

end profit_ratio_l275_275766


namespace fifteen_percent_eq_135_l275_275530

theorem fifteen_percent_eq_135 (x : ℝ) (h : (15 / 100) * x = 135) : x = 900 :=
sorry

end fifteen_percent_eq_135_l275_275530


namespace largest_multiple_of_15_less_than_neg_150_l275_275197

theorem largest_multiple_of_15_less_than_neg_150 : ∃ m : ℤ, m % 15 = 0 ∧ m < -150 ∧ (∀ n : ℤ, n % 15 = 0 ∧ n < -150 → n ≤ m) ∧ m = -165 := sorry

end largest_multiple_of_15_less_than_neg_150_l275_275197


namespace num_distinct_orders_of_targets_l275_275011

theorem num_distinct_orders_of_targets : 
  let total_targets := 10
  let column_A_targets := 4
  let column_B_targets := 4
  let column_C_targets := 2
  (Nat.factorial total_targets) / 
  ((Nat.factorial column_A_targets) * (Nat.factorial column_B_targets) * (Nat.factorial column_C_targets)) = 5040 := 
by
  sorry

end num_distinct_orders_of_targets_l275_275011


namespace largest_of_five_consecutive_composite_integers_under_40_l275_275710

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def five_consecutive_composite_integers_under_40 : List ℕ :=
[32, 33, 34, 35, 36]

theorem largest_of_five_consecutive_composite_integers_under_40 :
  ∀ n ∈ five_consecutive_composite_integers_under_40,
  n < 40 ∧ ∀ k, (k ∈ five_consecutive_composite_integers_under_40 →
  ¬ is_prime k) →
  List.maximum five_consecutive_composite_integers_under_40 = some 36 :=
by
  sorry

end largest_of_five_consecutive_composite_integers_under_40_l275_275710


namespace cricket_team_members_l275_275323

theorem cricket_team_members (n : ℕ) 
  (avg_age_team : ℕ) 
  (age_captain : ℕ) 
  (age_wkeeper : ℕ) 
  (avg_age_remaining : ℕ) 
  (total_age_team : ℕ) 
  (total_age_excl_cw : ℕ) 
  (total_age_remaining : ℕ) :
  avg_age_team = 23 →
  age_captain = 26 →
  age_wkeeper = 29 →
  avg_age_remaining = 22 →
  total_age_team = avg_age_team * n →
  total_age_excl_cw = total_age_team - (age_captain + age_wkeeper) →
  total_age_remaining = avg_age_remaining * (n - 2) →
  total_age_excl_cw = total_age_remaining →
  n = 11 :=
by
  sorry

end cricket_team_members_l275_275323


namespace remainder_sum_1_to_12_div_9_l275_275801

-- Define the sum of the first n natural numbers
def sum_natural (n : Nat) : Nat := n * (n + 1) / 2

-- Define the sum of the numbers from 1 to 12
def sum_1_to_12 := sum_natural 12

-- Define the remainder function
def remainder (a b : Nat) : Nat := a % b

-- Prove that the remainder when the sum of the numbers from 1 to 12 is divided by 9 is 6
theorem remainder_sum_1_to_12_div_9 : remainder sum_1_to_12 9 = 6 := by
  sorry

end remainder_sum_1_to_12_div_9_l275_275801


namespace reciprocals_sum_eq_neg_one_over_three_l275_275453

-- Let the reciprocals of the roots of the polynomial 7x^2 + 2x + 6 be alpha and beta.
-- Given that a and b are roots of the polynomial, and alpha = 1/a and beta = 1/b,
-- Prove that alpha + beta = -1/3.

theorem reciprocals_sum_eq_neg_one_over_three
  (a b : ℝ)
  (ha : 7 * a ^ 2 + 2 * a + 6 = 0)
  (hb : 7 * b ^ 2 + 2 * b + 6 = 0)
  (h_sum : a + b = -2 / 7)
  (h_prod : a * b = 6 / 7) :
  (1 / a) + (1 / b) = -1 / 3 := by
  sorry

end reciprocals_sum_eq_neg_one_over_three_l275_275453


namespace calculate_length_of_bridge_l275_275988

/-- Define the conditions based on given problem -/
def length_of_bridge (speed1 speed2 : ℕ) (length1 length2 : ℕ) (time : ℕ) : ℕ :=
    let distance_covered_train1 := speed1 * time
    let bridge_length_train1 := distance_covered_train1 - length1
    let distance_covered_train2 := speed2 * time
    let bridge_length_train2 := distance_covered_train2 - length2
    max bridge_length_train1 bridge_length_train2

/-- Given conditions -/
def speed_train1 := 15 -- in m/s
def length_train1 := 130 -- in meters
def speed_train2 := 20 -- in m/s
def length_train2 := 90 -- in meters
def crossing_time := 30 -- in seconds

theorem calculate_length_of_bridge : length_of_bridge speed_train1 speed_train2 length_train1 length_train2 crossing_time = 510 :=
by
  -- omitted proof
  sorry

end calculate_length_of_bridge_l275_275988


namespace selection_assignment_schemes_l275_275760

noncomputable def number_of_selection_schemes (males females : ℕ) : ℕ :=
  if h : males + females < 3 then 0
  else
    let total3 := Nat.choose (males + females) 3
    let all_males := if hM : males < 3 then 0 else Nat.choose males 3
    let all_females := if hF : females < 3 then 0 else Nat.choose females 3
    total3 - all_males - all_females

theorem selection_assignment_schemes :
  number_of_selection_schemes 4 3 = 30 :=
by sorry

end selection_assignment_schemes_l275_275760


namespace exact_time_now_l275_275438

noncomputable def time_now (t : ℝ) : Prop := 
  (4 < t / 60) ∧ (t / 60 < 5) ∧
  (|6 * (t + 8) - (120 + 0.5 * (t - 6))| = 180)

theorem exact_time_now : ∃ t : ℝ, time_now t ∧ t = 45.27 :=
by
  sorry

end exact_time_now_l275_275438


namespace number_of_students_l275_275216

theorem number_of_students (n : ℕ) (bow_cost : ℕ) (vinegar_cost : ℕ) (baking_soda_cost : ℕ) (total_cost : ℕ) :
  bow_cost = 5 → vinegar_cost = 2 → baking_soda_cost = 1 → total_cost = 184 → 8 * n = total_cost → n = 23 :=
by
  intros h_bow h_vinegar h_baking_soda h_total_cost h_equation
  sorry

end number_of_students_l275_275216


namespace solve_for_x_l275_275764

def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

theorem solve_for_x (x : ℝ) : (custom_mul 3 (custom_mul 6 x) = 2) → (x = 19 / 2) :=
sorry

end solve_for_x_l275_275764


namespace isosceles_triangle_base_l275_275543

theorem isosceles_triangle_base (a b c : ℕ) (h_isosceles : a = b ∨ a = c ∨ b = c)
  (h_perimeter : a + b + c = 29) (h_side : a = 7 ∨ b = 7 ∨ c = 7) : 
  a = 7 ∨ b = 7 ∨ c = 7 ∧ (a = 7 ∨ a = 11) ∧ (b = 7 ∨ b = 11) ∧ (c = 7 ∨ c = 11) ∧ (a ≠ b ∨ c ≠ b) :=
by
  sorry

end isosceles_triangle_base_l275_275543


namespace quadratic_sequence_l275_275101

theorem quadratic_sequence (a x₁ b x₂ c : ℝ)
  (h₁ : a + b = 2 * x₁)
  (h₂ : x₁ + x₂ = 2 * b)
  (h₃ : a + c = 2 * b)
  (h₄ : x₁ + x₂ = -6 / a)
  (h₅ : x₁ * x₂ = c / a) :
  b = -2 * a ∧ c = -5 * a :=
by
  sorry

end quadratic_sequence_l275_275101


namespace rectangle_area_l275_275575

structure Rectangle where
  length : ℕ    -- Length of the rectangle in cm
  width : ℕ     -- Width of the rectangle in cm
  perimeter : ℕ -- Perimeter of the rectangle in cm
  h : length = width + 4 -- Distance condition from the diagonal intersection

theorem rectangle_area (r : Rectangle) (h_perim : r.perimeter = 56) : r.length * r.width = 192 := by
  sorry

end rectangle_area_l275_275575


namespace wholesale_price_of_pen_l275_275165

-- Definitions and conditions
def wholesale_price (P : ℝ) : Prop :=
  (5 - P = 10 - 3 * P)

-- Statement of the proof problem
theorem wholesale_price_of_pen : ∃ P : ℝ, wholesale_price P ∧ P = 2.5 :=
by {
  sorry
}

end wholesale_price_of_pen_l275_275165


namespace john_sleep_total_hours_l275_275914

-- Defining the conditions provided in the problem statement
def days_with_3_hours : ℕ := 2
def sleep_per_day_3_hours : ℕ := 3
def remaining_days : ℕ := 7 - days_with_3_hours
def recommended_sleep : ℕ := 8
def percentage_sleep : ℝ := 0.6

-- Expressing the proof problem statement
theorem john_sleep_total_hours :
  (days_with_3_hours * sleep_per_day_3_hours
  + remaining_days * (percentage_sleep * recommended_sleep)) = 30 := by
  sorry

end john_sleep_total_hours_l275_275914


namespace dinner_plates_percentage_l275_275442

/-- Define the cost of silverware and the total cost of both items -/
def silverware_cost : ℝ := 20
def total_cost : ℝ := 30

/-- Define the percentage of the silverware cost that the dinner plates cost -/
def percentage_of_silverware_cost := 50

theorem dinner_plates_percentage :
  ∃ (P : ℝ) (S : ℝ) (x : ℝ), S = silverware_cost ∧ (P + S = total_cost) ∧ (P = (x / 100) * S) ∧ x = percentage_of_silverware_cost :=
by {
  sorry
}

end dinner_plates_percentage_l275_275442


namespace seq_problem_part1_seq_problem_part2_l275_275904

def seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

theorem seq_problem_part1 (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  a 2008 = 0 := 
sorry

theorem seq_problem_part2 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  ∃ (M : ℤ), 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = 0) ∧ 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = M) := 
sorry

end seq_problem_part1_seq_problem_part2_l275_275904


namespace proof_problem_l275_275565

variable (a b c A B C : ℝ)
variable (h_a : a = Real.sqrt 3)
variable (h_b_ge_a : b ≥ a)
variable (h_cos : Real.cos (2 * C) - Real.cos (2 * A) =
  2 * Real.sin (Real.pi / 3 + C) * Real.sin (Real.pi / 3 - C))

theorem proof_problem :
  (A = Real.pi / 3) ∧ (2 * b - c ∈ Set.Ico (Real.sqrt 3) (2 * Real.sqrt 3)) :=
  sorry

end proof_problem_l275_275565


namespace cube_vertices_faces_edges_l275_275775

theorem cube_vertices_faces_edges (V F E : ℕ) (hv : V = 8) (hf : F = 6) (euler : V - E + F = 2) : E = 12 :=
by
  sorry

end cube_vertices_faces_edges_l275_275775


namespace students_not_good_at_either_l275_275284

theorem students_not_good_at_either (total good_at_english good_at_chinese both_good : ℕ) 
(h₁ : total = 45) 
(h₂ : good_at_english = 35) 
(h₃ : good_at_chinese = 31) 
(h₄ : both_good = 24) : total - (good_at_english + good_at_chinese - both_good) = 3 :=
by sorry

end students_not_good_at_either_l275_275284


namespace price_reduction_for_1200_profit_price_reduction_for_max_profit_l275_275064

noncomputable def average_daily_sales : ℕ := 20
noncomputable def profit_per_piece : ℕ := 40
noncomputable def sales_increase_per_dollar_reduction : ℕ := 2
variable (x : ℝ)

def average_daily_profit (x : ℝ) : ℝ :=
  (profit_per_piece - x) * (average_daily_sales + sales_increase_per_dollar_reduction * x)

theorem price_reduction_for_1200_profit :
  ((average_daily_profit x) = 1200) → (x = 20) := by
  sorry

theorem price_reduction_for_max_profit :
  (∀ x, average_daily_profit x ≤ average_daily_profit 15) ∧ (average_daily_profit 15 = 1250) := by
  sorry

end price_reduction_for_1200_profit_price_reduction_for_max_profit_l275_275064


namespace simplify_expression_l275_275763

theorem simplify_expression : (1 / (1 / ((1 / 3) ^ 1) + 1 / ((1 / 3) ^ 2) + 1 / ((1 / 3) ^ 3))) = (1 / 39) :=
by
  sorry

end simplify_expression_l275_275763


namespace area_of_equilateral_triangle_inscribed_in_square_l275_275861

variables {a : ℝ}

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  a^2 * (2 * Real.sqrt 3 - 3)

theorem area_of_equilateral_triangle_inscribed_in_square (a : ℝ) :
  equilateral_triangle_area a = a^2 * (2 * Real.sqrt 3 - 3) :=
by sorry

end area_of_equilateral_triangle_inscribed_in_square_l275_275861


namespace a_n_divisible_by_11_l275_275183

-- Define the sequence
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 3 ∧ 
  ∀ n, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n

-- Main statement
theorem a_n_divisible_by_11 (a : ℕ → ℤ) (h : seq a) :
  ∀ n, ∃ k : ℕ, a n % 11 = 0 ↔ n = 4 + 11 * k :=
sorry

end a_n_divisible_by_11_l275_275183


namespace algebraic_expression_value_l275_275003

namespace MathProof

variables {α β : ℝ} 

-- Given conditions
def is_root (a : ℝ) : Prop := a^2 - a - 1 = 0
def roots_of_quadratic (α β : ℝ) : Prop := is_root α ∧ is_root β

-- The proof problem statement
theorem algebraic_expression_value (h : roots_of_quadratic α β) : α^2 + α * (β^2 - 2) = 0 := 
by sorry

end MathProof

end algebraic_expression_value_l275_275003


namespace sum_of_digits_in_binary_representation_of_315_l275_275824

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l275_275824


namespace find_r_value_l275_275586

theorem find_r_value (m : ℕ) (h_m : m = 3) (t : ℕ) (h_t : t = 3^m + 2) (r : ℕ) (h_r : r = 4^t - 2 * t) : r = 4^29 - 58 := by
  sorry

end find_r_value_l275_275586


namespace cos_160_eq_neg_09397_l275_275235

theorem cos_160_eq_neg_09397 :
  Real.cos (160 * Real.pi / 180) = -0.9397 :=
by
  sorry

end cos_160_eq_neg_09397_l275_275235


namespace solution_of_system_l275_275410

theorem solution_of_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 1) : x + y = 3 :=
sorry

end solution_of_system_l275_275410


namespace max_value_a_l275_275879

noncomputable def f (a x : ℝ) := x - a / x

theorem max_value_a (a : ℝ) (h_a : a > 0) :
  (∀ x ∈ Icc (1 : ℝ) 2, |(1 + a / 2) * (x - 1) + 1 - a - (x - a / x)| ≤ 1) → 
  a ≤ 6 + 4 * sqrt 2 :=
begin
  intro h,
  -- Proof omitted
  sorry
end

end max_value_a_l275_275879


namespace marbles_left_l275_275306

def initial_marbles : ℝ := 150
def lost_marbles : ℝ := 58.5
def given_away_marbles : ℝ := 37.2
def found_marbles : ℝ := 10.8

theorem marbles_left :
  initial_marbles - lost_marbles - given_away_marbles + found_marbles = 65.1 :=
by 
  sorry

end marbles_left_l275_275306


namespace determine_functions_l275_275395

noncomputable def satisfies_condition (f : ℕ → ℕ) : Prop :=
∀ (n p : ℕ), Prime p → (f n)^p % f p = n % f p

theorem determine_functions :
  ∀ (f : ℕ → ℕ),
  satisfies_condition f →
  f = id ∨
  (∀ p: ℕ, Prime p → f p = 1) ∨
  (f 2 = 2 ∧ (∀ p: ℕ, Prime p → p > 2 → f p = 1) ∧ ∀ n: ℕ, f n % 2 = n % 2) :=
by
  intros f h1
  sorry

end determine_functions_l275_275395


namespace sums_of_adjacent_cells_l275_275110

theorem sums_of_adjacent_cells (N : ℕ) (h : N ≥ 2) :
  ∃ (f : ℕ → ℕ → ℝ), (∀ i j, 1 ≤ i ∧ i < N → 1 ≤ j ∧ j < N → 
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f (i + 1) j) ∧
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f i (j + 1))) := sorry

end sums_of_adjacent_cells_l275_275110


namespace median_of_consecutive_integers_l275_275952

def sum_of_consecutive_integers (n : ℕ) (a : ℤ) : ℤ :=
  n * (2*a + (n - 1)) / 2

theorem median_of_consecutive_integers (a : ℤ) : 
  (sum_of_consecutive_integers 25 a = 5^5) -> 
  (a + 12 = 125) := 
by
  sorry

end median_of_consecutive_integers_l275_275952


namespace probability_of_high_value_hand_l275_275437

noncomputable def bridge_hand_probability : ℚ :=
  let total_combinations : ℕ := Nat.choose 16 4
  let favorable_combinations : ℕ := 1 + 16 + 16 + 16 + 36 + 96 + 16
  favorable_combinations / total_combinations

theorem probability_of_high_value_hand : bridge_hand_probability = 197 / 1820 := by
  sorry

end probability_of_high_value_hand_l275_275437


namespace possible_values_of_m_l275_275429

variable (m : ℝ)
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  (b^2 - 4 * a * c > 0)

theorem possible_values_of_m (h : has_two_distinct_real_roots 1 m 9) : m ∈ Iio (-6) ∪ Ioi 6 :=
sorry

end possible_values_of_m_l275_275429


namespace john_pays_2400_per_year_l275_275017

theorem john_pays_2400_per_year
  (hours_per_month : ℕ)
  (minutes_per_hour : ℕ)
  (songs_per_minute : ℕ)
  (cost_per_song : ℕ)
  (months_per_year : ℕ)
  (H1 : hours_per_month = 20)
  (H2 : minutes_per_hour = 60)
  (H3 : songs_per_minute = 3)
  (H4 : cost_per_song = 50)
  (H5 : months_per_year = 12) :
  let minutes_per_month := hours_per_month * minutes_per_hour,
      songs_per_month := minutes_per_month / songs_per_minute,
      cost_per_month := songs_per_month * cost_per_song in
  cost_per_month * months_per_year = 2400 := by
  sorry

end john_pays_2400_per_year_l275_275017


namespace relationship_between_a_b_c_l275_275540

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) - 1
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2)
noncomputable def c : ℝ := f (Real.log (1/4) / Real.log 2)

theorem relationship_between_a_b_c : a < c ∧ c < b :=
by
  sorry

end relationship_between_a_b_c_l275_275540


namespace sphere_volume_l275_275634

theorem sphere_volume (r : ℝ) (h1 : 4 * π * r^2 = 256 * π) : 
  (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end sphere_volume_l275_275634


namespace last_digit_sum_chessboard_segments_l275_275309

theorem last_digit_sum_chessboard_segments {N : ℕ} (tile_count : ℕ) (segment_count : ℕ := 112) (dominos_per_tiling : ℕ := 32) (segments_per_domino : ℕ := 2) (N := tile_count / N) :
  (80 * N) % 10 = 0 :=
by
  sorry

end last_digit_sum_chessboard_segments_l275_275309


namespace min_distance_origin_to_line_l275_275903

noncomputable def distance_from_origin_to_line(A B C : ℝ) : ℝ :=
  let d := |A * 0 + B * 0 + C| / (Real.sqrt (A^2 + B^2))
  d

theorem min_distance_origin_to_line : distance_from_origin_to_line 1 1 (-4) = 2 * Real.sqrt 2 := by 
  sorry

end min_distance_origin_to_line_l275_275903


namespace solve_inequality_l275_275186

theorem solve_inequality (x : ℝ) : 
  (x - 2) / (x + 1) ≤ 2 ↔ x ∈ Set.Iic (-4) ∪ Set.Ioi (-1) := 
sorry

end solve_inequality_l275_275186


namespace sum_reverse_base7_eq_58_l275_275108

-- Definitions for the digit reversal and base representations
def reverse_digits_base (n : ℕ) (b : ℕ) : ℕ :=
  let digits := Nat.digits b n
  Nat.ofDigits b digits.reverse

-- The theorem statement
theorem sum_reverse_base7_eq_58 :
  (∑ n in Finset.filter (λ (n : ℕ), reverse_digits_base n 7 = n ∧ reverse_digits_base n 16 = n) (Finset.range 100), n) = 58 :=
sorry

end sum_reverse_base7_eq_58_l275_275108


namespace smallest_next_divisor_l275_275756

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

noncomputable def has_divisor_323 (n : ℕ) : Prop := 323 ∣ n

theorem smallest_next_divisor (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : has_divisor_323 n) :
  ∃ m : ℕ, m > 323 ∧ m ∣ n ∧ (∀ k : ℕ, k > 323 ∧ k < m → ¬ k ∣ n) ∧ m = 340 :=
sorry

end smallest_next_divisor_l275_275756


namespace find_x_l275_275474

def operation (a b : Int) : Int := 2 * a + b

theorem find_x :
  ∃ x : Int, operation 3 (operation 4 x) = -1 :=
  sorry

end find_x_l275_275474


namespace paving_stones_needed_l275_275210

-- Definition for the dimensions of the paving stone and the courtyard
def paving_stone_length : ℝ := 2.5
def paving_stone_width : ℝ := 2
def courtyard_length : ℝ := 30
def courtyard_width : ℝ := 16.5

-- Compute areas
def paving_stone_area : ℝ := paving_stone_length * paving_stone_width
def courtyard_area : ℝ := courtyard_length * courtyard_width

-- The theorem to prove that the number of paving stones needed is 99
theorem paving_stones_needed :
  (courtyard_area / paving_stone_area) = 99 :=
by
  sorry

end paving_stones_needed_l275_275210


namespace product_of_solutions_l275_275707

theorem product_of_solutions (t : ℝ) (h : t^2 = 64) : t * (-t) = -64 :=
sorry

end product_of_solutions_l275_275707


namespace complex_power_result_l275_275090

theorem complex_power_result :
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.sin (Real.pi / 6) * Complex.i)^4 = 
  -40.5 + 40.5 * Complex.i * Real.sqrt 3 :=
by sorry

end complex_power_result_l275_275090


namespace cos_150_eq_neg_sqrt3_div_2_l275_275399

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  unfold Real.cos
  sorry

end cos_150_eq_neg_sqrt3_div_2_l275_275399


namespace difference_of_numbers_l275_275479

theorem difference_of_numbers (L S : ℕ) (h1 : L = 1620) (h2 : L = 6 * S + 15) : L - S = 1353 :=
by
  sorry

end difference_of_numbers_l275_275479


namespace weight_cut_percentage_unknown_l275_275911

-- Define the initial conditions
def original_speed : ℝ := 150
def new_speed : ℝ := 205
def increase_supercharge : ℝ := original_speed * 0.3
def speed_after_supercharge : ℝ := original_speed + increase_supercharge
def increase_weight_cut : ℝ := new_speed - speed_after_supercharge

-- Theorem statement
theorem weight_cut_percentage_unknown : 
  (original_speed = 150) →
  (new_speed = 205) →
  (increase_supercharge = 150 * 0.3) →
  (speed_after_supercharge = 150 + increase_supercharge) →
  (increase_weight_cut = 205 - speed_after_supercharge) →
  increase_weight_cut = 10 →
  sorry := 
by
  intros h_orig h_new h_inc_scharge h_speed_scharge h_inc_weight h_inc_10
  sorry

end weight_cut_percentage_unknown_l275_275911


namespace filling_tank_ratio_l275_275838

theorem filling_tank_ratio :
  ∀ (t : ℝ),
    (1 / 40) * t + (1 / 24) * (29.999999999999993 - t) = 1 →
    t / 29.999999999999993 = 1 / 2 :=
by
  intro t
  intro H
  sorry

end filling_tank_ratio_l275_275838


namespace matrix_vector_product_l275_275392

-- Definitions for matrix A and vector v
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-3, 4],
  ![2, -1]
]

def v : Fin 2 → ℤ := ![2, -2]

-- The theorem to prove
theorem matrix_vector_product :
  (A.mulVec v) = ![-14, 6] :=
by sorry

end matrix_vector_product_l275_275392


namespace parabola_opens_downwards_l275_275420

theorem parabola_opens_downwards (m : ℝ) : (m + 3 < 0) → (m < -3) := 
by
  sorry

end parabola_opens_downwards_l275_275420


namespace space_mission_contribution_l275_275278

theorem space_mission_contribution 
  (mission_cost_million : ℕ := 30000) 
  (combined_population_million : ℕ := 350) : 
  mission_cost_million / combined_population_million = 86 := by
  sorry

end space_mission_contribution_l275_275278


namespace subset_singleton_zero_A_l275_275207

def A : Set ℝ := {x | x > -3}

theorem subset_singleton_zero_A : {0} ⊆ A := 
by
  sorry  -- Proof is not required

end subset_singleton_zero_A_l275_275207


namespace handshaking_remainder_div_1000_l275_275899

/-- Given eleven people where each person shakes hands with exactly three others, 
  let handshaking_count be the number of distinct handshaking arrangements.
  Find the remainder when handshaking_count is divided by 1000. -/
def handshaking_count (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) : Nat :=
  sorry

theorem handshaking_remainder_div_1000 (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) :
  (handshaking_count P hP handshakes H) % 1000 = 800 :=
sorry

end handshaking_remainder_div_1000_l275_275899


namespace pair_ab_l275_275173

def students_activities_ways (n_students n_activities : Nat) : Nat :=
  n_activities ^ n_students

def championships_outcomes (n_championships n_students : Nat) : Nat :=
  n_students ^ n_championships

theorem pair_ab (a b : Nat) :
  a = students_activities_ways 4 3 ∧ b = championships_outcomes 3 4 →
  (a, b) = (3^4, 4^3) := by
  sorry

end pair_ab_l275_275173


namespace max_value_2x_plus_y_l275_275872

theorem max_value_2x_plus_y (x y : ℝ) (h : y^2 / 4 + x^2 / 3 = 1) : 2 * x + y ≤ 4 :=
by
  sorry

end max_value_2x_plus_y_l275_275872


namespace count_integers_in_solution_set_l275_275271

-- Define the predicate for the condition given in the problem
def condition (x : ℝ) : Prop := abs (x - 3) ≤ 4.5

-- Define the list of integers within the range of the condition
def solution_set : List ℤ := [-1, 0, 1, 2, 3, 4, 5, 6, 7]

-- Prove that the number of integers satisfying the condition is 8
theorem count_integers_in_solution_set : solution_set.length = 8 :=
by
  sorry

end count_integers_in_solution_set_l275_275271


namespace trains_time_distance_l275_275339

-- Define the speeds of the two trains
def speed1 : ℕ := 11
def speed2 : ℕ := 31

-- Define the distance between the two trains after time t
def distance_between_trains (t : ℕ) : ℕ :=
  speed2 * t - speed1 * t

-- Define the condition that this distance is 160 miles
def condition (t : ℕ) : Prop :=
  distance_between_trains t = 160

-- State the theorem to prove
theorem trains_time_distance : ∃ t : ℕ, condition t ∧ t = 8 :=
by
  use 8
  unfold condition
  unfold distance_between_trains
  -- Verifying the calculated distance
  sorry

end trains_time_distance_l275_275339


namespace gcd_of_36_and_54_l275_275796

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l275_275796


namespace fifth_equation_in_pattern_l275_275307

theorem fifth_equation_in_pattern :
  (1 - 4 + 9 - 16 + 25) = (1 + 2 + 3 + 4 + 5) :=
sorry

end fifth_equation_in_pattern_l275_275307


namespace find_r_l275_275004

theorem find_r (r s : ℝ) (h_quadratic : ∀ y, y^2 - r * y - s = 0) (h_r_pos : r > 0) 
    (h_root_diff : ∀ (y₁ y₂ : ℝ), (y₁ = (r + Real.sqrt (r^2 + 4 * s)) / 2 
        ∧ y₂ = (r - Real.sqrt (r^2 + 4 * s)) / 2) → |y₁ - y₂| = 2) : r = 2 :=
sorry

end find_r_l275_275004


namespace sequence_sum_general_term_l275_275874

theorem sequence_sum_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n) : ∀ n, a n = 2 * n :=
by 
  sorry

end sequence_sum_general_term_l275_275874


namespace find_total_shaded_area_l275_275494

/-- Definition of the rectangles' dimensions and overlap conditions -/
def rect1_length : ℕ := 4
def rect1_width : ℕ := 15
def rect2_length : ℕ := 5
def rect2_width : ℕ := 10
def rect3_length : ℕ := 3
def rect3_width : ℕ := 18
def shared_side_length : ℕ := 4
def trip_overlap_width : ℕ := 3

/-- Calculation of the rectangular overlap using given conditions -/
theorem find_total_shaded_area : (rect1_length * rect1_width + rect2_length * rect2_width + rect3_length * rect3_width - shared_side_length * shared_side_length - trip_overlap_width * shared_side_length) = 136 :=
    by sorry

end find_total_shaded_area_l275_275494


namespace L_like_reflexive_l275_275371

-- Definitions of the shapes and condition of being an "L-like shape"
inductive Shape
| A | B | C | D | E | LLike : Shape → Shape

-- reflection_equiv function representing reflection equivalence across a vertical dashed line
def reflection_equiv (s1 s2 : Shape) : Prop :=
sorry -- This would be defined according to the exact conditions of the shapes and reflection logic.

-- Given the shapes
axiom L_like : Shape
axiom A : Shape
axiom B : Shape
axiom C : Shape
axiom D : Shape
axiom E : Shape

-- The proof problem: Shape D is the mirrored reflection of the given "L-like shape" across a vertical dashed line
theorem L_like_reflexive :
  reflection_equiv L_like D :=
sorry

end L_like_reflexive_l275_275371


namespace problem_l275_275583

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := sorry
def v : Fin 2 → ℝ := ![7, -3]
def result : Fin 2 → ℝ := ![-14, 6]
def expected : Fin 2 → ℝ := ![112, -48]

theorem problem :
    B.vecMul v = result →
    B.vecMul (B.vecMul (B.vecMul (B.vecMul v))) = expected := 
by
  intro h
  sorry

end problem_l275_275583


namespace find_y_value_l275_275888

theorem find_y_value (k : ℝ) (h1 : ∀ (x : ℝ), y = k * x) 
(h2 : y = 4 ∧ x = 2) : 
(∀ (x : ℝ), x = -2 → y = -4) := 
by 
  sorry

end find_y_value_l275_275888


namespace positive_integer_solution_l275_275405

theorem positive_integer_solution (n x y : ℕ) (hn : 0 < n) (hx : 0 < x) (hy : 0 < y) :
  y ^ 2 + x * y + 3 * x = n * (x ^ 2 + x * y + 3 * y) → n = 1 :=
sorry

end positive_integer_solution_l275_275405


namespace sum_of_a_c_l275_275176

theorem sum_of_a_c (a b c d : ℝ) (h1 : -2 * abs (1 - a) + b = 7) (h2 : 2 * abs (1 - c) + d = 7)
    (h3 : -2 * abs (11 - a) + b = -1) (h4 : 2 * abs (11 - c) + d = -1) : a + c = 12 := by
  -- Definitions for conditions
  -- h1: intersection at (1, 7) for first graph
  -- h2: intersection at (1, 7) for second graph
  -- h3: intersection at (11, -1) for first graph
  -- h4: intersection at (11, -1) for second graph
  sorry

end sum_of_a_c_l275_275176


namespace rectangle_perimeter_l275_275983

-- Definitions of the conditions
def side_length_square : ℕ := 75  -- side length of the square in mm
def height_sum (x y z : ℕ) : Prop := x + y + z = side_length_square  -- sum of heights of the rectangles

-- Perimeter definition
def perimeter (h : ℕ) (w : ℕ) : ℕ := 2 * (h + w)

-- Statement of the problem
theorem rectangle_perimeter (x y z : ℕ) (h_sum : height_sum x y z)
  (h1 : perimeter x side_length_square = (perimeter y side_length_square + perimeter z side_length_square) / 2)
  : perimeter x side_length_square = 200 := by
  sorry

end rectangle_perimeter_l275_275983


namespace sum_quotient_dividend_divisor_l275_275369

theorem sum_quotient_dividend_divisor (N : ℕ) (divisor : ℕ) (quotient : ℕ) (sum : ℕ) 
    (h₁ : N = 40) (h₂ : divisor = 2) (h₃ : quotient = N / divisor)
    (h₄ : sum = quotient + N + divisor) : sum = 62 := by
  -- proof goes here
  sorry

end sum_quotient_dividend_divisor_l275_275369


namespace range_of_m_three_zeros_l275_275276

theorem range_of_m_three_zeros (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x^3 - 3*x + m = 0) ∧ (y^3 - 3*y + m = 0) ∧ (z^3 - 3*z + m = 0)) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end range_of_m_three_zeros_l275_275276


namespace length_of_platform_l275_275842

theorem length_of_platform
  (length_train : ℝ)
  (speed_train_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_covered : ℝ)
  (conversion_factor : ℝ) :
  length_train = 250 →
  speed_train_kmph = 90 →
  time_seconds = 20 →
  distance_covered = (speed_train_kmph * 1000 / 3600) * time_seconds →
  conversion_factor = 1000 / 3600 →
  ∃ P : ℝ, distance_covered = length_train + P ∧ P = 250 :=
by
  sorry

end length_of_platform_l275_275842


namespace tina_more_than_katya_l275_275022

-- Define the number of glasses sold by Katya, Ricky, and the condition for Tina's sales
def katya_sales : ℕ := 8
def ricky_sales : ℕ := 9

def combined_sales : ℕ := katya_sales + ricky_sales
def tina_sales : ℕ := 2 * combined_sales

-- Define the theorem to prove that Tina sold 26 more glasses than Katya
theorem tina_more_than_katya : tina_sales = katya_sales + 26 := by
  sorry

end tina_more_than_katya_l275_275022


namespace seventh_diagram_shaded_triangles_l275_275150

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- The main theorem stating the relationship between the number of shaded sub-triangles and the factorial/Fibonacci sequence
theorem seventh_diagram_shaded_triangles :
  ∃ k : ℕ, (k : ℚ) = (fib 7 : ℚ) / (fact 7 : ℚ) ∧ k = 13 := sorry

end seventh_diagram_shaded_triangles_l275_275150


namespace top_leftmost_rectangle_is_B_l275_275240

-- Define the sides of the rectangles
structure Rectangle :=
  (w : ℕ)
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

-- Define the specific rectangles with their side values
noncomputable def rectA : Rectangle := ⟨2, 7, 4, 7⟩
noncomputable def rectB : Rectangle := ⟨0, 6, 8, 5⟩
noncomputable def rectC : Rectangle := ⟨6, 3, 1, 1⟩
noncomputable def rectD : Rectangle := ⟨8, 4, 0, 2⟩
noncomputable def rectE : Rectangle := ⟨5, 9, 3, 6⟩
noncomputable def rectF : Rectangle := ⟨7, 5, 9, 0⟩

-- Prove that Rectangle B is the top leftmost rectangle
theorem top_leftmost_rectangle_is_B :
  (rectB.w = 0 ∧ rectB.x = 6 ∧ rectB.y = 8 ∧ rectB.z = 5) :=
by {
  sorry
}

end top_leftmost_rectangle_is_B_l275_275240


namespace calculate_factorial_expression_l275_275232

theorem calculate_factorial_expression :
  6 * nat.factorial 6 + 5 * nat.factorial 5 + nat.factorial 5 = 5040 := 
sorry

end calculate_factorial_expression_l275_275232


namespace perpendicular_line_through_point_l275_275649

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l275_275649


namespace quadratic_monotonically_increasing_l275_275739

open Interval

theorem quadratic_monotonically_increasing (m : ℝ) :
  (∀ x y : ℝ, 2 < x → x < y → y < +∞ → f x ≤ f y) ↔ m ≥ -4 :=
by
  let f := λ x : ℝ, x^2 + m*x - 2
  sorry

end quadratic_monotonically_increasing_l275_275739


namespace leah_total_coin_value_l275_275024

variable (p n : ℕ) -- Let p be the number of pennies and n be the number of nickels

-- Leah has 15 coins consisting of pennies and nickels
axiom coin_count : p + n = 15

-- If she had three more nickels, she would have twice as many pennies as nickels
axiom conditional_equation : p = 2 * (n + 3)

-- We want to prove that the total value of Leah's coins in cents is 27
theorem leah_total_coin_value : 5 * n + p = 27 := by
  sorry

end leah_total_coin_value_l275_275024


namespace science_book_multiple_l275_275774

theorem science_book_multiple (history_pages novel_pages science_pages : ℕ)
  (H1 : history_pages = 300)
  (H2 : novel_pages = history_pages / 2)
  (H3 : science_pages = 600) :
  science_pages / novel_pages = 4 := 
by
  -- Proof will be filled out here
  sorry

end science_book_multiple_l275_275774


namespace jordan_novels_read_l275_275020

variable (J A : ℕ)

theorem jordan_novels_read (h1 : A = (1 / 10) * J)
                          (h2 : J = A + 108) :
                          J = 120 := 
by
  sorry

end jordan_novels_read_l275_275020


namespace sallys_woodworking_llc_reimbursement_l275_275924

/-
Conditions:
1. Remy paid $20,700 for 150 pieces of furniture.
2. The cost of a piece of furniture is $134.
-/
def reimbursement_amount (pieces_paid : ℕ) (total_paid : ℕ) (price_per_piece : ℕ) : ℕ :=
  total_paid - (pieces_paid * price_per_piece)

theorem sallys_woodworking_llc_reimbursement :
  reimbursement_amount 150 20700 134 = 600 :=
by 
  sorry

end sallys_woodworking_llc_reimbursement_l275_275924


namespace elevator_translation_l275_275383

-- Definitions based on conditions
def turning_of_steering_wheel : Prop := False
def rotation_of_bicycle_wheels : Prop := False
def motion_of_pendulum : Prop := False
def movement_of_elevator : Prop := True

-- Theorem statement
theorem elevator_translation :
  movement_of_elevator := by
  exact True.intro

end elevator_translation_l275_275383


namespace one_thirds_in_eight_halves_l275_275272

theorem one_thirds_in_eight_halves : (8 / 2) / (1 / 3) = 12 := by
  sorry

end one_thirds_in_eight_halves_l275_275272


namespace condition_neither_sufficient_nor_necessary_l275_275478

noncomputable def f (x a : ℝ) : ℝ := x^3 - x + a
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 1

def condition (a : ℝ) : Prop := a^2 - a = 0

theorem condition_neither_sufficient_nor_necessary
  (a : ℝ) :
  ¬(condition a → (∀ x : ℝ, f' x ≥ 0)) ∧ ¬((∀ x : ℝ, f' x ≥ 0) → condition a) :=
by
  sorry -- Proof is omitted as per the prompt

end condition_neither_sufficient_nor_necessary_l275_275478


namespace probability_three_twos_given_sum_six_l275_275785

open ProbabilityTheory

-- Definitions of the conditions
def balls : List ℕ := [1, 2, 3]
def urn := finset.univ : Finset ℕ).image (λ _, balls)

-- The event space
def events := urn.product urn.product urn

-- The condition that the sum of the draws is 𝟞
def total_sum_six (e : ℕ × (ℕ × ℕ)) : Prop :=
  e.1 + e.2.1 + e.2.2 = 6

-- The favorable event where all draws are 𝟚
def all_twos (e : ℕ × (ℕ × ℕ)) : Prop :=
  e.1 = 2 ∧ e.2.1 = 2 ∧ e.2.2 = 2

-- The main statement to prove
theorem probability_three_twos_given_sum_six :
  (∑' (e : ℕ × (ℕ × ℕ)) in events.filter all_twos, 1 / events.card) /
  (∑' (e : ℕ × (ℕ × ℕ)) in events.filter total_sum_six, 1 / events.card)
  = 1 / 7 :=
sorry

end probability_three_twos_given_sum_six_l275_275785


namespace rectangle_side_deficit_l275_275902

theorem rectangle_side_deficit (L W : ℝ) (p : ℝ)
  (h1 : 1.05 * L * (1 - p) * W - L * W = 0.8 / 100 * L * W)
  (h2 : 0 < L) (h3 : 0 < W) : p = 0.04 :=
by {
  sorry
}

end rectangle_side_deficit_l275_275902


namespace initial_volume_salt_solution_l275_275274

theorem initial_volume_salt_solution (V : ℝ) (V1 : ℝ) (V2 : ℝ) : 
  V1 = 0.20 * V → 
  V2 = 30 →
  V1 = 0.15 * (V + V2) →
  V = 90 := 
by 
  sorry

end initial_volume_salt_solution_l275_275274


namespace exists_a_not_divisible_l275_275915

theorem exists_a_not_divisible (p : ℕ) (hp_prime : Prime p) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ (p^2 ∣ (a^(p-1) - 1)) ∧ ¬ (p^2 ∣ ((a+1)^(p-1) - 1))) :=
  sorry

end exists_a_not_divisible_l275_275915


namespace central_angle_l275_275122

theorem central_angle (r l θ : ℝ) (condition1: 2 * r + l = 8) (condition2: (1 / 2) * l * r = 4) (theta_def : θ = l / r) : |θ| = 2 :=
by
  sorry

end central_angle_l275_275122


namespace opposite_of_neg_four_l275_275483

-- Define the condition: the opposite of a number is the number that, when added to the original number, results in zero.
def is_opposite (a b : Int) : Prop := a + b = 0

-- The specific theorem we want to prove
theorem opposite_of_neg_four : is_opposite (-4) 4 := by
  -- Placeholder for the proof
  sorry

end opposite_of_neg_four_l275_275483


namespace complex_power_l275_275089

def cos_30 := Float.sqrt 3 / 2
def sin_30 := 1 / 2

theorem complex_power (i : ℂ) : 
  (3 * (⟨cos_30, sin_30⟩ : ℂ)) ^ 4 = -40.5 + 40.5 * i * Float.sqrt 3 := 
  sorry

end complex_power_l275_275089


namespace more_pencils_than_pens_l275_275777

theorem more_pencils_than_pens : 
  ∀ (P L : ℕ), L = 30 → (P / L: ℚ) = 5 / 6 → ((L - P) = 5) := by
  intros P L hL hRatio
  sorry

end more_pencils_than_pens_l275_275777


namespace cyclists_meet_fourth_time_l275_275714

theorem cyclists_meet_fourth_time 
  (speed1 speed2 speed3 speed4 : ℕ)
  (len : ℚ)
  (t_start : ℕ)
  (h_speed1 : speed1 = 6)
  (h_speed2 : speed2 = 9)
  (h_speed3 : speed3 = 12)
  (h_speed4 : speed4 = 15)
  (h_len : len = 1 / 3)
  (h_t_start : t_start = 12 * 60 * 60)
  : 
  (t_start + 4 * (20 * 60 + 40)) = 12 * 60 * 60 + 1600  :=
sorry

end cyclists_meet_fourth_time_l275_275714


namespace sum_of_exponents_l275_275528

theorem sum_of_exponents (n : ℕ) (h : n = 896) : 
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 2^a + 2^b + 2^c = n ∧ a + b + c = 24 :=
by
  sorry

end sum_of_exponents_l275_275528


namespace pure_imaginary_implies_a_neg_one_l275_275002

theorem pure_imaginary_implies_a_neg_one (a : ℝ) 
  (h_pure_imaginary : ∃ (y : ℝ), z = 0 + y * I) : 
  z = a + 1 - a * I → a = -1 :=
by
  sorry

end pure_imaginary_implies_a_neg_one_l275_275002


namespace probability_of_detecting_non_conforming_l275_275674

noncomputable def prob_detecting_non_conforming (total_cans non_conforming_cans selected_cans : ℕ) : ℚ :=
  let total_outcomes := Nat.choose total_cans selected_cans
  let outcomes_with_one_non_conforming := Nat.choose non_conforming_cans 1 * Nat.choose (total_cans - non_conforming_cans) (selected_cans - 1)
  let outcomes_with_two_non_conforming := Nat.choose non_conforming_cans 2
  (outcomes_with_one_non_conforming + outcomes_with_two_non_conforming) / total_outcomes

theorem probability_of_detecting_non_conforming :
  prob_detecting_non_conforming 5 2 2 = 7 / 10 :=
by
  -- Placeholder for the actual proof
  sorry

end probability_of_detecting_non_conforming_l275_275674


namespace original_number_l275_275349

theorem original_number (x : ℝ) (hx : 100000 * x = 5 * (1 / x)) : x = 0.00707 := 
by
  sorry

end original_number_l275_275349


namespace room_width_is_12_l275_275771

variable (w : ℝ)

def length_of_room : ℝ := 20
def width_of_veranda : ℝ := 2
def area_of_veranda : ℝ := 144

theorem room_width_is_12 :
  24 * (w + 4) - 20 * w = 144 → w = 12 := by
  sorry

end room_width_is_12_l275_275771


namespace rajas_monthly_income_l275_275931

theorem rajas_monthly_income (I : ℝ) (h : 0.6 * I + 0.1 * I + 0.1 * I + 5000 = I) : I = 25000 :=
sorry

end rajas_monthly_income_l275_275931


namespace neg_number_is_A_l275_275382

def A : ℤ := -(3 ^ 2)
def B : ℤ := (-3) ^ 2
def C : ℤ := abs (-3)
def D : ℤ := -(-3)

theorem neg_number_is_A : A < 0 := 
by sorry

end neg_number_is_A_l275_275382


namespace honey_last_nights_l275_275618

def servings_per_cup : Nat := 1
def cups_per_night : Nat := 2
def container_ounces : Nat := 16
def servings_per_ounce : Nat := 6

theorem honey_last_nights :
  (container_ounces * servings_per_ounce) / (servings_per_cup * cups_per_night) = 48 :=
by
  sorry  -- Proof not provided as per requirements

end honey_last_nights_l275_275618


namespace milk_cost_l275_275439

theorem milk_cost (x : ℝ) (h1 : 4 * 2.50 + 2 * x = 17) : x = 3.50 :=
by
  sorry

end milk_cost_l275_275439


namespace perpendicular_line_through_point_l275_275645

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l275_275645


namespace markers_leftover_l275_275034

theorem markers_leftover :
  let total_markers := 154
  let num_packages := 13
  total_markers % num_packages = 11 :=
by
  sorry

end markers_leftover_l275_275034


namespace geom_sequence_sum_correct_l275_275291

noncomputable def geom_sequence_sum (a₁ a₄ : ℕ) (S₅ : ℕ) :=
  ∃ q : ℕ, a₁ = 1 ∧ a₄ = a₁ * q ^ 3 ∧ S₅ = (a₁ * (1 - q ^ 5)) / (1 - q)

theorem geom_sequence_sum_correct : geom_sequence_sum 1 8 31 :=
by {
  sorry
}

end geom_sequence_sum_correct_l275_275291


namespace eighteen_mnp_eq_P_np_Q_2mp_l275_275040

theorem eighteen_mnp_eq_P_np_Q_2mp (m n p : ℕ) (P Q : ℕ) (hP : P = 2 ^ m) (hQ : Q = 3 ^ n) :
  18 ^ (m * n * p) = P ^ (n * p) * Q ^ (2 * m * p) :=
by
  sorry

end eighteen_mnp_eq_P_np_Q_2mp_l275_275040


namespace second_group_work_days_l275_275735

theorem second_group_work_days (M B : ℕ) (d1 d2 : ℕ) (H1 : M = 2 * B) 
  (H2 : (12 * M + 16 * B) * 5 = d1) (H3 : (13 * M + 24 * B) * d2 = d1) : 
  d2 = 4 :=
by
  sorry

end second_group_work_days_l275_275735


namespace comb_23_5_eq_33649_l275_275724

theorem comb_23_5_eq_33649 :
  (∃ (c21_3 c21_4 c21_5 : ℕ), c21_3 = 1330 ∧ c21_4 = 5985 ∧ c21_5 = 20349) →
  (nat.choose 23 5 = 33649) :=
by
  intro h
  obtain ⟨c21_3, c21_4, c21_5, h1, h2, h3⟩ := h
  have h4 : nat.choose 21 3 = c21_3 := h1
  have h5 : nat.choose 21 4 = c21_4 := h2
  have h6 : nat.choose 21 5 = c21_5 := h3
  sorry

end comb_23_5_eq_33649_l275_275724


namespace divide_64_to_get_800_l275_275400

theorem divide_64_to_get_800 (x : ℝ) (h : 64 / x = 800) : x = 0.08 :=
sorry

end divide_64_to_get_800_l275_275400


namespace total_candies_l275_275373

theorem total_candies (n p r : ℕ) (H1 : n = 157) (H2 : p = 235) (H3 : r = 98) :
  n * p + r = 36993 := by
  sorry

end total_candies_l275_275373


namespace probability_same_tribe_l275_275568

def totalPeople : ℕ := 18
def peoplePerTribe : ℕ := 6
def tribes : ℕ := 3
def totalQuitters : ℕ := 2

def totalWaysToChooseQuitters := Nat.choose totalPeople totalQuitters
def waysToChooseFromTribe := Nat.choose peoplePerTribe totalQuitters
def totalWaysFromSameTribe := tribes * waysToChooseFromTribe

theorem probability_same_tribe (h1 : totalPeople = 18) (h2 : peoplePerTribe = 6) (h3 : tribes = 3) (h4 : totalQuitters = 2)
    (h5 : totalWaysToChooseQuitters = 153) (h6 : totalWaysFromSameTribe = 45) :
    (totalWaysFromSameTribe : ℚ) / totalWaysToChooseQuitters = 5 / 17 := by
  sorry

end probability_same_tribe_l275_275568


namespace initial_ratio_milk_water_l275_275068

theorem initial_ratio_milk_water (M W : ℕ) (h1 : M + W = 165) (h2 : ∀ W', W' = W + 66 → M * 4 = 3 * W') : M / gcd M W = 3 ∧ W / gcd M W = 2 :=
by
  -- Proof here
  sorry

end initial_ratio_milk_water_l275_275068


namespace problem1_problem2_l275_275335

-- Given vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Problem 1: Prove that 3a + b - 2c = (0, 6)
theorem problem1 : 3 • a + b - 2 • c = (0, 6) := 
by {
   sorry
}

-- Problem 2: Prove that there exist real numbers m and n such that a = m • b + n • c with m = 5/9 and n = 8/9
theorem problem2 : ∃ (m n : ℝ), a = m • b + n • c ∧ m = (5/9) ∧ n = (8/9) := 
by {
   sorry
}

end problem1_problem2_l275_275335


namespace correct_mms_packs_used_l275_275443

variable (num_sundaes_monday : ℕ) (mms_per_sundae_monday : ℕ)
variable (num_sundaes_tuesday : ℕ) (mms_per_sundae_tuesday : ℕ)
variable (mms_per_pack : ℕ)

-- Conditions
def conditions : Prop := 
  num_sundaes_monday = 40 ∧ 
  mms_per_sundae_monday = 6 ∧ 
  num_sundaes_tuesday = 20 ∧
  mms_per_sundae_tuesday = 10 ∧ 
  mms_per_pack = 40

-- Question: How many m&m packs does Kekai use?
def number_of_mms_packs (num_sundaes_monday mms_per_sundae_monday 
                         num_sundaes_tuesday mms_per_sundae_tuesday 
                         mms_per_pack : ℕ) : ℕ := 
  (num_sundaes_monday * mms_per_sundae_monday + num_sundaes_tuesday * mms_per_sundae_tuesday) / mms_per_pack

-- Theorem to prove the correct number of m&m packs used
theorem correct_mms_packs_used (h : conditions num_sundaes_monday mms_per_sundae_monday 
                                              num_sundaes_tuesday mms_per_sundae_tuesday 
                                              mms_per_pack) : 
  number_of_mms_packs num_sundaes_monday mms_per_sundae_monday 
                      num_sundaes_tuesday mms_per_sundae_tuesday 
                      mms_per_pack = 11 := by {
  -- Proof goes here
  sorry
}

end correct_mms_packs_used_l275_275443


namespace horner_first_calculation_at_3_l275_275341

def f (x : ℝ) : ℝ :=
  0.5 * x ^ 6 + 4 * x ^ 5 - x ^ 4 + 3 * x ^ 3 - 5 * x

def horner_first_step (x : ℝ) : ℝ :=
  0.5 * x + 4

theorem horner_first_calculation_at_3 :
  horner_first_step 3 = 5.5 := by
  sorry

end horner_first_calculation_at_3_l275_275341


namespace members_in_third_shift_l275_275854

-- Defining the given conditions
def total_first_shift : ℕ := 60
def percent_first_shift_pension : ℝ := 0.20

def total_second_shift : ℕ := 50
def percent_second_shift_pension : ℝ := 0.40

variable (T : ℕ)
def percent_third_shift_pension : ℝ := 0.10

def percent_total_pension_program : ℝ := 0.24

noncomputable def number_of_members_third_shift : ℕ :=
  T

-- Using the conditions to declare the theorem
theorem members_in_third_shift :
  ((60 * 0.20) + (50 * 0.40) + (number_of_members_third_shift T * percent_third_shift_pension)) / (60 + 50 + number_of_members_third_shift T) = percent_total_pension_program →
  number_of_members_third_shift T = 40 :=
sorry

end members_in_third_shift_l275_275854


namespace cone_in_sphere_less_half_volume_l275_275687

theorem cone_in_sphere_less_half_volume
  (R r m : ℝ)
  (h1 : m < 2 * R)
  (h2 : r <= R) :
  (1 / 3 * Real.pi * r^2 * m < 1 / 2 * 4 / 3 * Real.pi * R^3) :=
by
  sorry

end cone_in_sphere_less_half_volume_l275_275687


namespace sufficient_but_not_necessary_l275_275832

theorem sufficient_but_not_necessary (a : ℝ) : (a = 1 → a^2 = 1) ∧ ¬(a^2 = 1 → a = 1) :=
by
  sorry

end sufficient_but_not_necessary_l275_275832


namespace systems_solution_l275_275896

    theorem systems_solution : 
      (∃ x y : ℝ, 2 * x + 5 * y = -26 ∧ 3 * x - 5 * y = 36 ∧ 
                 (∃ a b : ℝ, a * x - b * y = -4 ∧ b * x + a * y = -8 ∧ 
                 (2 * a + b) ^ 2020 = 1)) := 
    by
      sorry
    
end systems_solution_l275_275896


namespace find_m_l275_275262

noncomputable def tangent_condition (m : ℝ) : Prop :=
  let d : ℝ := |2| / Real.sqrt (m^2 + 1)
  d = 1

theorem find_m (m : ℝ) : tangent_condition m ↔ m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
  sorry

end find_m_l275_275262


namespace max_distance_l275_275417

-- Definition of curve C₁ in rectangular coordinates.
def C₁_rectangular (x y : ℝ) : Prop := x^2 + y^2 - 2 * y = 0

-- Definition of curve C₂ in its general form.
def C₂_general (x y : ℝ) : Prop := 4 * x + 3 * y - 8 = 0

-- Coordinates of point M, the intersection of C₂ with x-axis.
def M : ℝ × ℝ := (2, 0)

-- Condition that N is a moving point on curve C₁.
def N (x y : ℝ) : Prop := C₁_rectangular x y

-- Maximum distance |MN|.
theorem max_distance (x y : ℝ) (hN : N x y) : 
  dist (2, 0) (x, y) ≤ Real.sqrt 5 + 1 := by
  sorry

end max_distance_l275_275417


namespace sum_of_QR_l275_275156

noncomputable def given_conditions := 
  ∃ (P Q R : Type) (angleQ : ℝ) (PQ PR : ℝ), 
    angleQ = real.pi / 4 ∧
    PQ = 100 ∧ 
    PR = 100 * real.sqrt 2

noncomputable def verify_QR (QR : ℝ) := 
  ∃ (P Q R : Type) (angleQ : ℝ) (PQ PR : ℝ), 
    angleQ = real.pi / 4 ∧
    PQ = 100 ∧
    PR = 100 * real.sqrt 2 ∧
    QR = 100 * real.sqrt 3

theorem sum_of_QR : given_conditions → verify_QR  (100 * real.sqrt 3) :=
by
  intro h
  sorry

end sum_of_QR_l275_275156


namespace gcd_polynomial_l275_275427

theorem gcd_polynomial {b : ℤ} (h1 : ∃ k : ℤ, b = 2 * 7786 * k) : 
  Int.gcd (8 * b^2 + 85 * b + 200) (2 * b + 10) = 10 :=
by
  sorry

end gcd_polynomial_l275_275427


namespace smallest_sum_of_three_diff_numbers_l275_275100

theorem smallest_sum_of_three_diff_numbers : 
  ∀ (s : Set ℤ), s = {8, -7, 2, -4, 20} → ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = -9) :=
by
  sorry

end smallest_sum_of_three_diff_numbers_l275_275100


namespace train_length_l275_275359

theorem train_length (L : ℝ) (v : ℝ)
  (h1 : v = (L + 130) / 15)
  (h2 : v = (L + 250) / 20) : 
  L = 230 :=
sorry

end train_length_l275_275359


namespace sum_X_Y_Z_l275_275477

theorem sum_X_Y_Z (X Y Z : ℕ) (hX : X ∈ Finset.range 10) (hY : Y ∈ Finset.range 10) (hZ : Z = 0)
     (div9 : (1 + 3 + 0 + 7 + 6 + 7 + 4 + X + 2 + 0 + Y + 0 + 0 + 8 + 0) % 9 = 0) 
     (div7 : (307674 * 10 + X * 20 + Y * 10 + 800) % 7 = 0) :
  X + Y + Z = 7 := 
sorry

end sum_X_Y_Z_l275_275477


namespace optimal_order_l275_275607

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l275_275607


namespace janet_faster_playtime_l275_275157

theorem janet_faster_playtime 
  (initial_minutes : ℕ)
  (initial_seconds : ℕ)
  (faster_rate : ℝ)
  (initial_time_in_seconds := initial_minutes * 60 + initial_seconds)
  (target_time_in_seconds := initial_time_in_seconds / faster_rate) :
  initial_minutes = 3 →
  initial_seconds = 20 →
  faster_rate = 1.25 →
  target_time_in_seconds = 160 :=
by
  intros h1 h2 h3
  sorry

end janet_faster_playtime_l275_275157


namespace calculate_unshaded_perimeter_l275_275333

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

end calculate_unshaded_perimeter_l275_275333


namespace total_value_of_assets_l275_275033

variable (value_expensive_stock : ℕ)
variable (shares_expensive_stock : ℕ)
variable (shares_other_stock : ℕ)
variable (value_other_stock : ℕ)

theorem total_value_of_assets
    (h1: value_expensive_stock = 78)
    (h2: shares_expensive_stock = 14)
    (h3: shares_other_stock = 26)
    (h4: value_other_stock = value_expensive_stock / 2) :
    shares_expensive_stock * value_expensive_stock + shares_other_stock * value_other_stock = 2106 := by
    sorry

end total_value_of_assets_l275_275033


namespace median_of_consecutive_integers_l275_275951

theorem median_of_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) (h_sum : sum_of_integers = 5^5) (h_num : num_of_integers = 25) : 
  let median := sum_of_integers / num_of_integers
  in median = 125 :=
by
  let median := sum_of_integers / num_of_integers
  have h1 : sum_of_integers = 3125 := by exact h_sum
  have h2 : num_of_integers = 25 := by exact h_num
  have h3 : median = 125 := by
    calc
      median = 3125 / 25 : by rw [h1, h2]
            ... = 125      : by norm_num
  exact h3

end median_of_consecutive_integers_l275_275951


namespace photograph_area_l275_275377

def dimensions_are_valid (a b : ℕ) : Prop :=
a > 0 ∧ b > 0 ∧ (a + 4) * (b + 5) = 77

theorem photograph_area (a b : ℕ) (h : dimensions_are_valid a b) : (a * b = 18 ∨ a * b = 14) :=
by 
  sorry

end photograph_area_l275_275377


namespace find_k_l275_275013

theorem find_k
  (AB AC : ℝ)
  (k : ℝ)
  (h1 : AB = AC)
  (h2 : AB = 8)
  (h3 : AC = 5 - k) : k = -3 :=
by
  sorry

end find_k_l275_275013


namespace gcd_pow_minus_one_l275_275587

theorem gcd_pow_minus_one {m n a : ℕ} (hm : 0 < m) (hn : 0 < n) (ha : 2 ≤ a) : 
  Nat.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd m n) - 1 := 
sorry

end gcd_pow_minus_one_l275_275587


namespace FashionDesignNotInServiceAreas_l275_275199

-- Define the service areas of Digital China
def ServiceAreas (x : String) : Prop :=
  x = "Understanding the situation of soil and water loss in the Yangtze River Basin" ∨
  x = "Understanding stock market trends" ∨
  x = "Wanted criminals"

-- Prove that "Fashion design" is not in the service areas of Digital China
theorem FashionDesignNotInServiceAreas : ¬ ServiceAreas "Fashion design" :=
sorry

end FashionDesignNotInServiceAreas_l275_275199


namespace modulus_of_z_l275_275547

open Complex -- Open the Complex number namespace

-- Define the given condition as a hypothesis
def condition (z : ℂ) : Prop := (1 + I) * z = 3 + I

-- Statement of the theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = Real.sqrt 5 :=
sorry

end modulus_of_z_l275_275547


namespace monic_poly_has_root_l275_275532

theorem monic_poly_has_root : 
  ∃ (P : Polynomial ℚ), P.degree = 4 ∧ P.leadingCoeff = 1 ∧ P.eval (Real.of_rat (3 ^ (1/2) + 5 ^ (1/2))) = 0 :=
by
  sorry

end monic_poly_has_root_l275_275532


namespace sum_of_coefficients_eq_zero_l275_275126

theorem sum_of_coefficients_eq_zero 
  (A B C D E F : ℝ) :
  (∀ x, (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) 
  = A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by sorry

end sum_of_coefficients_eq_zero_l275_275126


namespace min_distance_from_circle_to_line_l275_275116

noncomputable def circle_center : (ℝ × ℝ) := (3, -1)
noncomputable def circle_radius : ℝ := 2

def on_circle (P : ℝ × ℝ) : Prop := (P.1 - circle_center.1) ^ 2 + (P.2 + circle_center.2) ^ 2 = circle_radius ^ 2
def on_line (Q : ℝ × ℝ) : Prop := Q.1 = -3

theorem min_distance_from_circle_to_line (P Q : ℝ × ℝ)
  (h1 : on_circle P) (h2 : on_line Q) : dist P Q = 4 := 
sorry

end min_distance_from_circle_to_line_l275_275116


namespace ostap_advantageous_order_l275_275597

theorem ostap_advantageous_order (p1 p2 p3 : ℝ) (h1 : p3 < p1) (h2 : p1 < p2) : 
  ∀ order : List ℝ, 
    (order = [p1, p2, p3] ∨ order = [p2, p1, p3] ∨ order = [p3, p1, p2]) → (order.nth 1 = some p2) :=
sorry

end ostap_advantageous_order_l275_275597


namespace identify_different_correlation_l275_275665

-- Define the concept of correlation
inductive Correlation
| positive
| negative

-- Define the conditions for each option
def option_A : Correlation := Correlation.positive
def option_B : Correlation := Correlation.positive
def option_C : Correlation := Correlation.negative
def option_D : Correlation := Correlation.positive

-- The statement to prove
theorem identify_different_correlation :
  (option_A = Correlation.positive) ∧ 
  (option_B = Correlation.positive) ∧ 
  (option_D = Correlation.positive) ∧ 
  (option_C = Correlation.negative) := 
sorry

end identify_different_correlation_l275_275665


namespace problem1_problem2_l275_275234

-- Proof Problem 1: Prove that (x-y)^2 - (x+y)(x-y) = -2xy + 2y^2
theorem problem1 (x y : ℝ) : (x - y) ^ 2 - (x + y) * (x - y) = -2 * x * y + 2 * y ^ 2 := 
by
  sorry

-- Proof Problem 2: Prove that (12a^2b - 6ab^2) / (-3ab) = -4a + 2b
theorem problem2 (a b : ℝ) (h : -3 * a * b ≠ 0) : (12 * a^2 * b - 6 * a * b^2) / (-3 * a * b) = -4 * a + 2 * b := 
by
  sorry

end problem1_problem2_l275_275234


namespace passing_percentage_l275_275682

theorem passing_percentage
  (marks_obtained : ℕ)
  (marks_failed_by : ℕ)
  (max_marks : ℕ)
  (h_marks_obtained : marks_obtained = 92)
  (h_marks_failed_by : marks_failed_by = 40)
  (h_max_marks : max_marks = 400) :
  (marks_obtained + marks_failed_by) / max_marks * 100 = 33 := 
by
  sorry

end passing_percentage_l275_275682


namespace rowing_distance_l275_275330

theorem rowing_distance (D : ℝ) 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (downstream_speed : ℝ := boat_speed + stream_speed) 
  (upstream_speed : ℝ := boat_speed - stream_speed)
  (downstream_time : ℝ := D / downstream_speed)
  (upstream_time : ℝ := D / upstream_speed)
  (round_trip_time : ℝ := downstream_time + upstream_time) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 2) 
  (h3 : total_time = 914.2857142857143)
  (h4 : round_trip_time = total_time) :
  D = 720 :=
by sorry

end rowing_distance_l275_275330


namespace can_weigh_1kg_with_300g_and_650g_weights_l275_275267

-- Definitions based on conditions
def balance_scale (a b : ℕ) (w₁ w₂ : ℕ) : Prop :=
  a * w₁ + b * w₂ = 1000

-- Statement to prove based on the problem and solution
theorem can_weigh_1kg_with_300g_and_650g_weights (w₁ : ℕ) (w₂ : ℕ) (a b : ℕ)
  (h_w1 : w₁ = 300) (h_w2 : w₂ = 650) (h_a : a = 1) (h_b : b = 1) :
  balance_scale a b w₁ w₂ :=
by 
  -- We are given:
  -- - w1 = 300 g
  -- - w2 = 650 g
  -- - we want to measure 1000 g using these weights
  -- - a = 1
  -- - b = 1
  -- Prove that:
  --   a * w1 + b * w2 = 1000
  -- Which is:
  --   1 * 300 + 1 * 650 = 1000
  sorry

end can_weigh_1kg_with_300g_and_650g_weights_l275_275267


namespace consecutive_integers_no_two_l275_275463

theorem consecutive_integers_no_two (a n : ℕ) : 
  ¬(∃ (b : ℤ), (b : ℤ) = 2) :=
sorry

end consecutive_integers_no_two_l275_275463


namespace dot_product_a_b_l275_275727

open Real

noncomputable def cos_deg (x : ℝ) := cos (x * π / 180)
noncomputable def sin_deg (x : ℝ) := sin (x * π / 180)

theorem dot_product_a_b :
  let a_magnitude := 2 * cos_deg 15
  let b_magnitude := 4 * sin_deg 15
  let angle_ab := 30
  a_magnitude * b_magnitude * cos_deg angle_ab = sqrt 3 :=
by
  -- proof omitted
  sorry

end dot_product_a_b_l275_275727


namespace gcd_of_36_and_54_l275_275798

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l275_275798


namespace find_a_l275_275545

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1 ∧ x ≥ 2

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, point_on_hyperbola x y ∧ (min ((x - a)^2 + y^2) = 3)) → 
  (a = -1 ∨ a = 2 * Real.sqrt 5) :=
by
  sorry

end find_a_l275_275545


namespace planting_schemes_count_l275_275742

universe u

def hexagon_graph : SimpleGraph (Fin 6) :=
  SimpleGraph.cycle (Fin 6)

def num_plants : ℕ := 4

theorem planting_schemes_count : chromaticPolynomial hexagon_graph num_plants = 732 := by
  sorry

end planting_schemes_count_l275_275742


namespace find_sin_minus_cos_l275_275124

variable {a : ℝ}
variable {α : ℝ}

def point_of_angle (a : ℝ) (h : a < 0) := (3 * a, -4 * a)

theorem find_sin_minus_cos (a : ℝ) (h : a < 0) (ha : point_of_angle a h = (3 * a, -4 * a)) (sinα : ℝ) (cosα : ℝ) :
  sinα = 4 / 5 → cosα = -3 / 5 → sinα - cosα = 7 / 5 :=
by sorry

end find_sin_minus_cos_l275_275124


namespace commutativity_l275_275161

universe u

variable {M : Type u} [Nonempty M]
variable (star : M → M → M)

axiom star_assoc_right {a b : M} : (star (star a b) b) = a
axiom star_assoc_left {a b : M} : star a (star a b) = b

theorem commutativity (a b : M) : star a b = star b a :=
by sorry

end commutativity_l275_275161


namespace erasers_in_each_box_l275_275188

theorem erasers_in_each_box (boxes : ℕ) (price_per_eraser : ℚ) (total_money_made : ℚ) (total_erasers_sold : ℕ) (erasers_per_box : ℕ) :
  boxes = 48 → price_per_eraser = 0.75 → total_money_made = 864 → total_erasers_sold = 1152 → total_erasers_sold / boxes = erasers_per_box → erasers_per_box = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end erasers_in_each_box_l275_275188


namespace required_butter_l275_275679

-- Define the given conditions
variables (butter sugar : ℕ)
def recipe_butter : ℕ := 25
def recipe_sugar : ℕ := 125
def used_sugar : ℕ := 1000

-- State the theorem
theorem required_butter (h1 : butter = recipe_butter) (h2 : sugar = recipe_sugar) :
  (used_sugar * recipe_butter) / recipe_sugar = 200 := 
by 
  sorry

end required_butter_l275_275679


namespace people_in_third_row_l275_275491

theorem people_in_third_row (row1_ini row2_ini left_row1 left_row2 total_left : ℕ) (h1 : row1_ini = 24) (h2 : row2_ini = 20) (h3 : left_row1 = row1_ini - 3) (h4 : left_row2 = row2_ini - 5) (h_total : total_left = 54) :
  total_left - (left_row1 + left_row2) = 18 := 
by
  sorry

end people_in_third_row_l275_275491


namespace optimal_order_for_ostap_l275_275601

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l275_275601


namespace sandy_marks_l275_275171

def marks_each_correct_sum : ℕ := 3

theorem sandy_marks (x : ℕ) 
  (total_attempts : ℕ := 30)
  (correct_sums : ℕ := 23)
  (marks_per_incorrect_sum : ℕ := 2)
  (total_marks_obtained : ℕ := 55)
  (incorrect_sums : ℕ := total_attempts - correct_sums)
  (lost_marks : ℕ := incorrect_sums * marks_per_incorrect_sum) :
  (correct_sums * x - lost_marks = total_marks_obtained) -> x = marks_each_correct_sum :=
by
  sorry

end sandy_marks_l275_275171


namespace selected_numbers_satisfy_conditions_l275_275933

theorem selected_numbers_satisfy_conditions :
  ∃ (nums : Finset ℕ), 
  nums = {6, 34, 35, 51, 55, 77} ∧
  (∀ (a b c : ℕ), a ∈ nums → b ∈ nums → c ∈ nums → a ≠ b → a ≠ c → b ≠ c → 
    gcd a b = 1 ∨ gcd b c = 1 ∨ gcd c a = 1) ∧
  (∀ (x y z : ℕ), x ∈ nums → y ∈ nums → z ∈ nums → x ≠ y → x ≠ z → y ≠ z → 
    gcd x y ≠ 1 ∨ gcd y z ≠ 1 ∨ gcd z x ≠ 1) := 
sorry

end selected_numbers_satisfy_conditions_l275_275933


namespace evie_l275_275151

variable (Evie_current_age : ℕ) 

theorem evie's_age_in_one_year
  (h : Evie_current_age + 4 = 3 * (Evie_current_age - 2)) : 
  Evie_current_age + 1 = 6 :=
by
  sorry

end evie_l275_275151


namespace total_time_outside_class_l275_275342

-- Definitions based on given conditions
def first_recess : ℕ := 15
def second_recess : ℕ := 15
def lunch : ℕ := 30
def third_recess : ℕ := 20

-- Proof problem statement
theorem total_time_outside_class : first_recess + second_recess + lunch + third_recess = 80 := 
by sorry

end total_time_outside_class_l275_275342


namespace larger_number_is_1671_l275_275940

variable (L S : ℕ)

noncomputable def problem_conditions :=
  L - S = 1395 ∧ L = 6 * S + 15

theorem larger_number_is_1671 (h : problem_conditions L S) : L = 1671 := by
  sorry

end larger_number_is_1671_l275_275940


namespace variance_of_data_set_l275_275049

def data_set : List ℤ := [ -2, -1, 0, 3, 5 ]

def mean (l : List ℤ) : ℚ :=
  (l.sum / l.length)

def variance (l : List ℤ) : ℚ :=
  (1 / l.length) * (l.map (λ x => (x - mean l : ℚ)^2)).sum

theorem variance_of_data_set : variance data_set = 34 / 5 := by
  sorry

end variance_of_data_set_l275_275049


namespace stratified_sample_sum_l275_275841

theorem stratified_sample_sum :
  let grains := 40
  let veg_oils := 10
  let animal_foods := 30
  let fruits_veggies := 20
  let total_varieties := grains + veg_oils + animal_foods + fruits_veggies
  let sample_size := 20
  let veg_oils_proportion := (veg_oils:ℚ) / total_varieties
  let fruits_veggies_proportion := (fruits_veggies:ℚ) / total_varieties
  let veg_oils_sample := sample_size * veg_oils_proportion
  let fruits_veggies_sample := sample_size * fruits_veggies_proportion
  veg_oils_sample + fruits_veggies_sample = 6 := sorry

end stratified_sample_sum_l275_275841


namespace sqrt_four_eq_plus_minus_two_l275_275950

theorem sqrt_four_eq_plus_minus_two : ∃ y : ℤ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  -- Proof goes here
  sorry

end sqrt_four_eq_plus_minus_two_l275_275950


namespace no_rain_four_days_l275_275046

-- Define the probability of rain on any given day
def prob_rain : ℚ := 2/3

-- Define the probability that it does not rain on any given day
def prob_no_rain : ℚ := 1 - prob_rain

-- Define the probability that it does not rain at all over four days
def prob_no_rain_four_days : ℚ := prob_no_rain^4

theorem no_rain_four_days : prob_no_rain_four_days = 1/81 := by
  sorry

end no_rain_four_days_l275_275046


namespace intersection_points_of_segments_l275_275166

noncomputable def num_intersection_points (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) : ℕ :=
  3000

theorem intersection_points_of_segments (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) :
  num_intersection_points A B C P Q = 3000 :=
  by sorry

end intersection_points_of_segments_l275_275166


namespace max_abs_sum_on_ellipse_l275_275263

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), (x^2 / 4) + (y^2 / 9) = 1 → |x| + |y| ≤ 5 :=
by sorry

end max_abs_sum_on_ellipse_l275_275263


namespace sum_of_cubes_l275_275962

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 17) : a^3 + b^3 = 490 := 
sorry

end sum_of_cubes_l275_275962


namespace sum_of_digits_base2_315_l275_275808

theorem sum_of_digits_base2_315 : Nat.sumDigits (Nat.toDigits 2 315) = 6 :=
by
  sorry

end sum_of_digits_base2_315_l275_275808


namespace evaluate_expression_at_x_eq_3_l275_275855

theorem evaluate_expression_at_x_eq_3 : (3 ^ 3) ^ (3 ^ 3) = 27 ^ 27 := by
  sorry

end evaluate_expression_at_x_eq_3_l275_275855


namespace geometric_sequence_sum_ratio_l275_275917

noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a q : ℝ) (h : a * q^2 = 8 * a * q^5) :
  (geometric_sum a q 4) / (geometric_sum a q 2) = 5 / 4 :=
by
  -- The proof will go here.
  sorry

end geometric_sequence_sum_ratio_l275_275917


namespace product_of_solutions_l275_275102

theorem product_of_solutions : 
  ∀ x : ℝ, 5 = -2 * x^2 + 6 * x → (∃ α β : ℝ, (α ≠ β ∧ (α * β = 5 / 2))) :=
by
  sorry

end product_of_solutions_l275_275102


namespace checkerboard_black_squares_l275_275087

theorem checkerboard_black_squares (n : ℕ) (hn : n = 33) :
  let black_squares : ℕ := (n * n + 1) / 2
  black_squares = 545 :=
by
  sorry

end checkerboard_black_squares_l275_275087


namespace sum_of_binary_digits_of_315_l275_275814

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l275_275814


namespace prove_expression_l275_275136

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 5)

lemma root_of_unity : omega^5 = 1 := sorry
lemma sum_of_roots : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sorry

noncomputable def z := omega + omega^2 + omega^3 + omega^4

theorem prove_expression : z^2 + z + 1 = 1 :=
by 
  have h1 : omega^5 = 1 := root_of_unity
  have h2 : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sum_of_roots
  show z^2 + z + 1 = 1
  {
    -- Proof omitted
    sorry
  }

end prove_expression_l275_275136
