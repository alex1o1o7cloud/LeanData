import Mathlib

namespace pairs_of_participants_l1322_132243

theorem pairs_of_participants (n : Nat) (h : n = 12) : (Nat.choose n 2) = 66 := by
  sorry

end pairs_of_participants_l1322_132243


namespace solve_inequality_l1322_132232

theorem solve_inequality (x : ℝ) (h : x ≠ 1) : (x / (x - 1) ≥ 2 * x) ↔ (x ≤ 0 ∨ (1 < x ∧ x ≤ 3 / 2)) :=
by
  sorry

end solve_inequality_l1322_132232


namespace find_y_l1322_132226

theorem find_y (t : ℝ) (x y : ℝ) (h1 : x = 3 - 2 * t) (h2 : y = 5 * t + 3) (h3 : x = -7) : y = 28 :=
by {
  sorry
}

end find_y_l1322_132226


namespace average_of_P_and_R_l1322_132296

theorem average_of_P_and_R (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (Q + R) / 2 = 5250)
  (h3 : P = 3000)
  : (P + R) / 2 = 6200 := by
  sorry

end average_of_P_and_R_l1322_132296


namespace pages_copied_l1322_132280

theorem pages_copied (cost_per_page : ℕ) (amount_in_dollars : ℕ)
    (cents_per_dollar : ℕ) (total_cents : ℕ) 
    (pages : ℕ)
    (h1 : cost_per_page = 3)
    (h2 : amount_in_dollars = 25)
    (h3 : cents_per_dollar = 100)
    (h4 : total_cents = amount_in_dollars * cents_per_dollar)
    (h5 : total_cents = 2500)
    (h6 : pages = total_cents / cost_per_page) :
  pages = 833 := 
sorry

end pages_copied_l1322_132280


namespace ratio_simplified_l1322_132244

theorem ratio_simplified (total finished : ℕ) (h_total : total = 15) (h_finished : finished = 6) :
  (total - finished) / (Nat.gcd (total - finished) finished) = 3 ∧ finished / (Nat.gcd (total - finished) finished) = 2 := by
  sorry

end ratio_simplified_l1322_132244


namespace circle_equation_l1322_132276

/-- Given that point C is above the x-axis and
    the circle C with center C is tangent to the x-axis at point A(1,0) and
    intersects with circle O: x² + y² = 4 at points P and Q such that
    the length of PQ is sqrt(14)/2, the standard equation of circle C
    is (x - 1)² + (y - 1)² = 1. -/
theorem circle_equation {C : ℝ × ℝ} (hC : C.2 > 0) (tangent_at_A : C = (1, C.2))
  (intersect_with_O : ∃ P Q : ℝ × ℝ, (P ≠ Q) ∧ (P.1 ^ 2 + P.2 ^ 2 = 4) ∧ 
  (Q.1 ^ 2 + Q.2 ^ 2 = 4) ∧ ((P.1 - 1)^2 + (P.2 - C.2)^2 = C.2^2) ∧ 
  ((Q.1 - 1)^2 + (Q.2 - C.2)^2 = C.2^2) ∧ ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 14/4)) :
  (C.2 = 1) ∧ ((x - 1)^2 + (y - 1)^2 = 1) :=
by
  sorry

end circle_equation_l1322_132276


namespace largest_number_l1322_132208

-- Define the given numbers
def A : ℝ := 0.986
def B : ℝ := 0.9859
def C : ℝ := 0.98609
def D : ℝ := 0.896
def E : ℝ := 0.8979
def F : ℝ := 0.987

-- State the theorem that F is the largest number among A, B, C, D, and E
theorem largest_number : F > A ∧ F > B ∧ F > C ∧ F > D ∧ F > E := by
  sorry

end largest_number_l1322_132208


namespace proof_abc_identity_l1322_132251

variable {a b c : ℝ}

theorem proof_abc_identity
  (h_ne_a : a ≠ 1) (h_ne_na : a ≠ -1)
  (h_ne_b : b ≠ 1) (h_ne_nb : b ≠ -1)
  (h_ne_c : c ≠ 1) (h_ne_nc : c ≠ -1)
  (habc : a * b + b * c + c * a = 1) :
  a / (1 - a ^ 2) + b / (1 - b ^ 2) + c / (1 - c ^ 2) = (4 * a * b * c) / (1 - a ^ 2) / (1 - b ^ 2) / (1 - c ^ 2) :=
by 
  sorry

end proof_abc_identity_l1322_132251


namespace smallest_number_l1322_132250

theorem smallest_number:
    let a := 3.25
    let b := 3.26   -- 326% in decimal
    let c := 3.2    -- 3 1/5 in decimal
    let d := 3.75   -- 15/4 in decimal
    c < a ∧ c < b ∧ c < d :=
by
    sorry

end smallest_number_l1322_132250


namespace seven_expression_one_seven_expression_two_l1322_132289

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l1322_132289


namespace sqrt_expression_result_l1322_132263

theorem sqrt_expression_result :
  (Real.sqrt (16 - 8 * Real.sqrt 3) - Real.sqrt (16 + 8 * Real.sqrt 3)) ^ 2 = 48 := 
sorry

end sqrt_expression_result_l1322_132263


namespace arthur_walks_distance_l1322_132285

theorem arthur_walks_distance :
  ∀ (blocks_east blocks_north blocks_first blocks_other distance_first distance_other : ℕ)
  (fraction_first fraction_other : ℚ),
    blocks_east = 8 →
    blocks_north = 16 →
    blocks_first = 10 →
    blocks_other = (blocks_east + blocks_north) - blocks_first →
    fraction_first = 1 / 3 →
    fraction_other = 1 / 4 →
    distance_first = blocks_first * fraction_first →
    distance_other = blocks_other * fraction_other →
    (distance_first + distance_other) = 41 / 6 :=
by
  intros blocks_east blocks_north blocks_first blocks_other distance_first distance_other fraction_first fraction_other
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end arthur_walks_distance_l1322_132285


namespace fish_tank_ratio_l1322_132279

theorem fish_tank_ratio :
  ∀ (F1 F2 F3: ℕ),
  F1 = 15 →
  F3 = 10 →
  (F3 = (1 / 3 * F2)) →
  F2 / F1 = 2 :=
by
  intros F1 F2 F3 hF1 hF3 hF2
  sorry

end fish_tank_ratio_l1322_132279


namespace infinite_power_tower_solution_l1322_132217

theorem infinite_power_tower_solution (x : ℝ) (y : ℝ) (h1 : y = x ^ x ^ x ^ x ^ x ^ x ^ x ^ x ^ x ^ x) (h2 : y = 4) : x = Real.sqrt 2 :=
by
  sorry

end infinite_power_tower_solution_l1322_132217


namespace interest_rate_per_annum_l1322_132259

theorem interest_rate_per_annum (P A : ℝ) (T : ℝ)
  (principal_eq : P = 973.913043478261)
  (amount_eq : A = 1120)
  (time_eq : T = 3):
  (A - P) / (T * P) * 100 = 5 := 
by 
  sorry

end interest_rate_per_annum_l1322_132259


namespace vector_on_line_l1322_132245

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (p q : V)

theorem vector_on_line (k : ℝ) (hpq : p ≠ q) :
  ∃ t : ℝ, k • p + (1/2 : ℝ) • q = p + t • (q - p) → k = 1/2 :=
by
  sorry

end vector_on_line_l1322_132245


namespace profit_no_discount_l1322_132298

theorem profit_no_discount (CP SP ASP : ℝ) (discount profit : ℝ) (h1 : discount = 4 / 100) (h2 : profit = 38 / 100) (h3 : SP = CP + CP * profit) (h4 : ASP = SP - SP * discount) :
  ((SP - CP) / CP) * 100 = 38 :=
by
  sorry

end profit_no_discount_l1322_132298


namespace necessary_but_not_sufficient_condition_l1322_132230

variable {m : ℝ}

theorem necessary_but_not_sufficient_condition (h : (∃ x1 x2 : ℝ, (x1 ≠ 0 ∧ x1 = -x2) ∧ (x1^2 + x1 + m^2 - 1 = 0))): 
  0 < m ∧ m < 1 :=
by 
  sorry

end necessary_but_not_sufficient_condition_l1322_132230


namespace sale_in_fourth_month_l1322_132221

variable (sale1 sale2 sale3 sale5 sale6 sale4 : ℕ)

def average_sale (total : ℕ) (months : ℕ) : ℕ := total / months

theorem sale_in_fourth_month
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 7391)
  (avg : average_sale (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) 6 = 6900) :
  sale4 = 7230 := 
sorry

end sale_in_fourth_month_l1322_132221


namespace correct_product_l1322_132234

def reverse_digits (n: ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d2 * 10 + d1

theorem correct_product (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : b > 0) (h3 : reverse_digits a * b = 221) :
  a * b = 527 ∨ a * b = 923 :=
sorry

end correct_product_l1322_132234


namespace inequality_proof_l1322_132200

variable (a b c : ℝ)
variable (h_pos : a > 0) (h_pos2 : b > 0) (h_pos3 : c > 0)
variable (h_sum : a + b + c = 1)

theorem inequality_proof :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) > 1 / 2 := by
  sorry

end inequality_proof_l1322_132200


namespace A_investment_is_100_l1322_132278

-- Definitions directly from the conditions in a)
def A_investment (X : ℝ) := X * 12
def B_investment : ℝ := 200 * 6
def total_profit : ℝ := 100
def A_share_of_profit : ℝ := 50

-- Prove that given these conditions, A's initial investment X is 100
theorem A_investment_is_100 (X : ℝ) (h : A_share_of_profit / total_profit = A_investment X / B_investment) : X = 100 :=
by
  sorry

end A_investment_is_100_l1322_132278


namespace sum_a_b_c_d_eq_nine_l1322_132222

theorem sum_a_b_c_d_eq_nine
  (a b c d : ℤ)
  (h : (Polynomial.X ^ 2 + (Polynomial.C a) * Polynomial.X + Polynomial.C b) *
       (Polynomial.X ^ 2 + (Polynomial.C c) * Polynomial.X + Polynomial.C d) =
       Polynomial.X ^ 4 + 2 * Polynomial.X ^ 3 + Polynomial.X ^ 2 + 11 * Polynomial.X + 6) :
  a + b + c + d = 9 :=
by
  sorry

end sum_a_b_c_d_eq_nine_l1322_132222


namespace problem_statement_l1322_132227

-- Define the conditions:
def f (x : ℚ) : ℚ := sorry

axiom f_mul (a b : ℚ) : f (a * b) = f a + f b
axiom f_int (n : ℤ) : f (n : ℚ) = (n : ℚ)

-- The problem statement:
theorem problem_statement : f (8/13) < 0 :=
sorry

end problem_statement_l1322_132227


namespace binom_12_9_is_220_l1322_132267

def choose (n k : ℕ) : ℕ := n.choose k

theorem binom_12_9_is_220 :
  choose 12 9 = 220 :=
by {
  -- Proof is omitted
  sorry
}

end binom_12_9_is_220_l1322_132267


namespace probability_square_not_touching_outer_edge_l1322_132229

theorem probability_square_not_touching_outer_edge :
  let total_squares := 10 * 10
  let perimeter_squares := 10 + 10 + (10 - 2) + (10 - 2)
  let non_perimeter_squares := total_squares - perimeter_squares
  (non_perimeter_squares / total_squares) = (16 / 25) :=
by
  let total_squares := 10 * 10
  let perimeter_squares := 10 + 10 + (10 - 2) + (10 - 2)
  let non_perimeter_squares := total_squares - perimeter_squares
  have h : non_perimeter_squares / total_squares = 16 / 25 := by sorry
  exact h

end probability_square_not_touching_outer_edge_l1322_132229


namespace solve_for_x_l1322_132241

theorem solve_for_x (x : ℝ) : (0.25 * x = 0.15 * 1500 - 20) → x = 820 :=
by
  intro h
  sorry

end solve_for_x_l1322_132241


namespace find_second_term_geometric_sequence_l1322_132294

noncomputable def second_term_geometric_sequence (a r : ℝ) : ℝ :=
  a * r

theorem find_second_term_geometric_sequence:
  ∀ (a r : ℝ),
    a * r^2 = 12 →
    a * r^3 = 18 →
    second_term_geometric_sequence a r = 8 :=
by
  intros a r h1 h2
  sorry

end find_second_term_geometric_sequence_l1322_132294


namespace fiona_weekly_earnings_l1322_132271

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

end fiona_weekly_earnings_l1322_132271


namespace avg_five_probability_l1322_132213

/- Define the set of natural numbers from 1 to 9. -/
def S : Finset ℕ := Finset.range 10 \ {0}

/- Define the binomial coefficient for choosing 7 out of 9. -/
def choose_7_9 : ℕ := Nat.choose 9 7

/- Define the condition for the sum of chosen numbers to be 35. -/
def sum_is_35 (s : Finset ℕ) : Prop := s.sum id = 35

/- Number of ways to choose 3 pairs that sum to 10 and include number 5 - means sum should be 35-/
def ways_3_pairs_and_5 : ℕ := 4

/- Probability calculation. -/
def prob_sum_is_35 : ℚ := (ways_3_pairs_and_5: ℚ) / (choose_7_9: ℚ)

theorem avg_five_probability : prob_sum_is_35 = 1 / 9 := by
  sorry

end avg_five_probability_l1322_132213


namespace ramon_twice_loui_age_in_future_l1322_132224

theorem ramon_twice_loui_age_in_future : 
  ∀ (x : ℕ), 
  (∀ t : ℕ, t = 23 → 
            t * 2 = 46 → 
            ∀ r : ℕ, r = 26 → 
                      26 + x = 46 → 
                      x = 20) := 
by sorry

end ramon_twice_loui_age_in_future_l1322_132224


namespace intersection_M_N_l1322_132260

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x ≤ 2} := 
sorry

end intersection_M_N_l1322_132260


namespace largest_five_digit_number_tens_place_l1322_132210

theorem largest_five_digit_number_tens_place :
  ∀ (n : ℕ), n = 87315 → (n % 100) / 10 = 1 := 
by
  intros n h
  sorry

end largest_five_digit_number_tens_place_l1322_132210


namespace max_strips_cut_l1322_132288

-- Definitions: dimensions of the paper and the strips
def length_paper : ℕ := 14
def width_paper : ℕ := 11
def length_strip : ℕ := 4
def width_strip : ℕ := 1

-- States the main theorem: Maximum number of strips that can be cut from the rectangular piece of paper
theorem max_strips_cut (L W l w : ℕ) (H1 : L = 14) (H2 : W = 11) (H3 : l = 4) (H4 : w = 1) :
  ∃ n : ℕ, n = 33 :=
by
  sorry

end max_strips_cut_l1322_132288


namespace center_of_circle_l1322_132252

theorem center_of_circle (h k : ℝ) :
  (∀ x y : ℝ, (x - 3) ^ 2 + (y - 4) ^ 2 = 10 ↔ x ^ 2 + y ^ 2 = 6 * x + 8 * y - 15) → 
  h + k = 7 :=
sorry

end center_of_circle_l1322_132252


namespace cost_of_camel_l1322_132268

theorem cost_of_camel
  (C H O E : ℝ)
  (h1 : 10 * C = 24 * H)
  (h2 : 16 * H = 4 * O)
  (h3 : 6 * O = 4 * E)
  (h4 : 10 * E = 140000) :
  C = 5600 :=
by
  -- Skipping the proof steps
  sorry

end cost_of_camel_l1322_132268


namespace sum_of_edges_112_l1322_132239

-- Define the problem parameters
def volume (a b c : ℝ) : ℝ := a * b * c
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)
def sum_of_edges (a b c : ℝ) : ℝ := 4 * (a + b + c)

-- The main theorem 
theorem sum_of_edges_112
  (b s : ℝ) (h1 : volume (b / s) b (b * s) = 512)
  (h2 : surface_area (b / s) b (b * s) = 448)
  (h3 : 0 < b ∧ 0 < s) : 
  sum_of_edges (b / s) b (b * s) = 112 :=
sorry

end sum_of_edges_112_l1322_132239


namespace find_n_constant_term_l1322_132247

-- Given condition as a Lean term
def eq1 (n : ℕ) : ℕ := 2^(2*n) - (2^n + 992)

-- Prove that n = 5 fulfills the condition
theorem find_n : eq1 5 = 0 := by
  sorry

-- Given n = 5, find the constant term in the given expansion
def general_term (n r : ℕ) : ℤ := (-1)^r * (Nat.choose (2*n) r) * (n - 5*r/2)

-- Prove the constant term is 45 when n = 5
theorem constant_term : general_term 5 2 = 45 := by
  sorry

end find_n_constant_term_l1322_132247


namespace greatest_integer_less_than_or_equal_to_l1322_132269

theorem greatest_integer_less_than_or_equal_to (x : ℝ) (h : x = 2 + Real.sqrt 3) : 
  ⌊x^3⌋ = 51 :=
by
  have h' : x ^ 3 = (2 + Real.sqrt 3) ^ 3 := by rw [h]
  sorry

end greatest_integer_less_than_or_equal_to_l1322_132269


namespace common_chord_length_proof_l1322_132218

-- Define the first circle equation
def first_circle (x y : ℝ) : Prop := x^2 + y^2 = 50

-- Define the second circle equation
def second_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 6*y + 40 = 0

-- Define the property that the length of the common chord is equal to 2 * sqrt(5)
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 5

-- The theorem statement
theorem common_chord_length_proof :
  ∀ x y : ℝ, first_circle x y → second_circle x y → common_chord_length = 2 * Real.sqrt 5 :=
by
  intros x y h1 h2
  sorry

end common_chord_length_proof_l1322_132218


namespace part1_part2_part3_l1322_132216

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 1 - x - a * x^2

theorem part1 (x : ℝ) : f x 0 ≥ 0 :=
sorry

theorem part2 {a : ℝ} (h : ∀ x ≥ 0, f x a ≥ 0) : a ≤ 1 / 2 :=
sorry

theorem part3 (x : ℝ) (hx : x > 0) : (Real.exp x - 1) * Real.log (x + 1) > x^2 :=
sorry

end part1_part2_part3_l1322_132216


namespace initial_amount_l1322_132212

theorem initial_amount 
  (M : ℝ)
  (h1 : M * (3 / 5) * (2 / 3) * (3 / 4) * (4 / 7) = 700) : 
  M = 24500 / 6 :=
by sorry

end initial_amount_l1322_132212


namespace total_students_in_school_l1322_132265

noncomputable def small_school_students (boys girls : ℕ) (total_students : ℕ) : Prop :=
boys = 42 ∧ 
(girls : ℕ) = boys / 7 ∧
total_students = boys + girls

theorem total_students_in_school : small_school_students 42 6 48 :=
by
  sorry

end total_students_in_school_l1322_132265


namespace max_sub_min_value_l1322_132274

variable {x y : ℝ}

noncomputable def expression (x y : ℝ) : ℝ :=
  (abs (x + y))^2 / ((abs x)^2 + (abs y)^2)

theorem max_sub_min_value :
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 
  (expression x y ≤ 2 ∧ 0 ≤ expression x y) → 
  (∃ m M, m = 0 ∧ M = 2 ∧ M - m = 2) :=
by
  sorry

end max_sub_min_value_l1322_132274


namespace part1_part2_l1322_132299

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B (a : ℝ) := {x : ℝ | (x - a) * (x - a - 1) < 0}

theorem part1 (a : ℝ) : (1 ∈ set_B a) → 0 < a ∧ a < 1 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, x ∈ set_B a → x ∈ set_A) ∧ (∃ x, x ∉ set_B a ∧ x ∈ set_A) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l1322_132299


namespace original_number_is_25_l1322_132225

theorem original_number_is_25 (x : ℕ) (h : ∃ n : ℕ, (x^2 - 600)^n = x) : x = 25 :=
sorry

end original_number_is_25_l1322_132225


namespace a_must_be_negative_l1322_132237

theorem a_must_be_negative (a b : ℝ) (h1 : b > 0) (h2 : a / b < -2 / 3) : a < 0 :=
sorry

end a_must_be_negative_l1322_132237


namespace length_BD_l1322_132282

noncomputable def length_segments (CB : ℝ) : ℝ := 4 * CB

noncomputable def circle_radius_AC (CB : ℝ) : ℝ := (4 * CB) / 2

noncomputable def circle_radius_CB (CB : ℝ) : ℝ := CB / 2

noncomputable def tangent_touch_point (CB BD : ℝ) : Prop :=
  ∃ x, CB = x ∧ BD = x

theorem length_BD (CB BD : ℝ) (h : tangent_touch_point CB BD) : BD = CB :=
by
  sorry

end length_BD_l1322_132282


namespace class_size_l1322_132220

def S : ℝ := 30

theorem class_size (total percent_dogs_videogames percent_dogs_movies number_students_prefer_dogs : ℝ)
  (h1 : percent_dogs_videogames = 0.5)
  (h2 : percent_dogs_movies = 0.1)
  (h3 : number_students_prefer_dogs = 18)
  (h4 : total * (percent_dogs_videogames + percent_dogs_movies) = number_students_prefer_dogs) :
  total = S :=
by
  sorry

end class_size_l1322_132220


namespace find_minimum_abs_sum_l1322_132238

noncomputable def minimum_abs_sum (α β γ : ℝ) : ℝ :=
|α| + |β| + |γ|

theorem find_minimum_abs_sum :
  ∃ α β γ : ℝ, α + β + γ = 2 ∧ α * β * γ = 4 ∧
  minimum_abs_sum α β γ = 6 := by
  sorry

end find_minimum_abs_sum_l1322_132238


namespace max_value_xy_l1322_132284

open Real

theorem max_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 5 * y < 100) :
  ∃ (c : ℝ), c = 3703.7 ∧ ∀ (x' y' : ℝ), 0 < x' → 0 < y' → 2 * x' + 5 * y' < 100 → x' * y' * (100 - 2 * x' - 5 * y') ≤ c :=
sorry

end max_value_xy_l1322_132284


namespace sum_of_roots_l1322_132248

theorem sum_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -3) (hx1 : x1 + x2 = 2) :
  x1 + x2 = 2 :=
by {
  sorry
}

end sum_of_roots_l1322_132248


namespace roots_of_quadratic_equation_l1322_132228

theorem roots_of_quadratic_equation (a b c r s : ℝ) 
  (hr : a ≠ 0)
  (h : a * r^2 + b * r - c = 0)
  (h' : a * s^2 + b * s - c = 0)
  :
  (1 / r^2) + (1 / s^2) = (b^2 + 2 * a * c) / c^2 :=
by
  sorry

end roots_of_quadratic_equation_l1322_132228


namespace counted_integer_twice_l1322_132275

theorem counted_integer_twice (x n : ℕ) (hn : n = 100) 
  (h_sum : (n * (n + 1)) / 2 + x = 5053) : x = 3 := by
  sorry

end counted_integer_twice_l1322_132275


namespace average_donation_l1322_132256

theorem average_donation (d : ℕ) (n : ℕ) (r : ℕ) (average_donation : ℕ) 
  (h1 : d = 10)   -- $10 donated by customers
  (h2 : r = 2)    -- $2 donated by restaurant
  (h3 : n = 40)   -- number of customers
  (h4 : (r : ℕ) * n / d = 24) -- total donation by restaurant is $24
  : average_donation = 3 := 
by
  sorry

end average_donation_l1322_132256


namespace total_amount_correct_l1322_132203

noncomputable def total_amount_collected
    (single_ticket_price : ℕ)
    (couple_ticket_price : ℕ)
    (total_people : ℕ)
    (couple_tickets_sold : ℕ) : ℕ :=
  let single_tickets_sold := total_people - (couple_tickets_sold * 2)
  let amount_from_couple_tickets := couple_tickets_sold * couple_ticket_price
  let amount_from_single_tickets := single_tickets_sold * single_ticket_price
  amount_from_couple_tickets + amount_from_single_tickets

theorem total_amount_correct :
  total_amount_collected 20 35 128 16 = 2480 := by
  sorry

end total_amount_correct_l1322_132203


namespace fraction_ratio_l1322_132272

theorem fraction_ratio (x : ℚ) (h1 : 2 / 5 / (3 / 7) = x / (1 / 2)) :
  x = 7 / 15 :=
by {
  -- Proof omitted
  sorry
}

end fraction_ratio_l1322_132272


namespace celine_erasers_collected_l1322_132219

theorem celine_erasers_collected (G C J E : ℕ) 
    (hC : C = 2 * G)
    (hJ : J = 4 * G)
    (hE : E = 12 * G)
    (h_total : G + C + J + E = 151) : 
    C = 16 := 
by 
  -- Proof steps skipped, proof body not required as per instructions
  sorry

end celine_erasers_collected_l1322_132219


namespace determine_p_l1322_132207

noncomputable def roots (p : ℝ) : ℝ × ℝ :=
  let discr := p ^ 2 - 48
  ((-p + Real.sqrt discr) / 2, (-p - Real.sqrt discr) / 2)

theorem determine_p (p : ℝ) :
  let (x1, x2) := roots p
  (x1 - x2 = 1) → (p = 7 ∨ p = -7) :=
by
  intros
  sorry

end determine_p_l1322_132207


namespace point_outside_circle_l1322_132258

theorem point_outside_circle {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a * x + b * y = 1) : a^2 + b^2 > 1 :=
by sorry

end point_outside_circle_l1322_132258


namespace original_annual_pension_l1322_132204

theorem original_annual_pension (k x c d r s : ℝ) (h1 : k * (x + c) ^ (3/4) = k * x ^ (3/4) + r)
  (h2 : k * (x + d) ^ (3/4) = k * x ^ (3/4) + s) :
  k * x ^ (3/4) = (r - s) / (0.75 * (d - c)) :=
by sorry

end original_annual_pension_l1322_132204


namespace combined_rocket_height_l1322_132264

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  sorry

end combined_rocket_height_l1322_132264


namespace largest_possible_b_l1322_132205

theorem largest_possible_b 
  (V : ℕ)
  (a b c : ℤ)
  (hV : V = 360)
  (h1 : 1 < c)
  (h2 : c < b)
  (h3 : b < a)
  (h4 : a * b * c = V) 
  : b = 12 := 
  sorry

end largest_possible_b_l1322_132205


namespace minimum_xy_l1322_132292

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : xy ≥ 64 :=
sorry

end minimum_xy_l1322_132292


namespace snack_eaters_remaining_l1322_132291

noncomputable def initial_snack_eaters := 5000 * 60 / 100
noncomputable def snack_eaters_after_1_hour := initial_snack_eaters + 25
noncomputable def snack_eaters_after_70_percent_left := snack_eaters_after_1_hour * 30 / 100
noncomputable def snack_eaters_after_2_hour := snack_eaters_after_70_percent_left + 50
noncomputable def snack_eaters_after_800_left := snack_eaters_after_2_hour - 800
noncomputable def snack_eaters_after_2_thirds_left := snack_eaters_after_800_left * 1 / 3
noncomputable def final_snack_eaters := snack_eaters_after_2_thirds_left + 100

theorem snack_eaters_remaining : final_snack_eaters = 153 :=
by
  have h1 : initial_snack_eaters = 3000 := by sorry
  have h2 : snack_eaters_after_1_hour = initial_snack_eaters + 25 := by sorry
  have h3 : snack_eaters_after_70_percent_left = snack_eaters_after_1_hour * 30 / 100 := by sorry
  have h4 : snack_eaters_after_2_hour = snack_eaters_after_70_percent_left + 50 := by sorry
  have h5 : snack_eaters_after_800_left = snack_eaters_after_2_hour - 800 := by sorry
  have h6 : snack_eaters_after_2_thirds_left = snack_eaters_after_800_left * 1 / 3 := by sorry
  have h7 : final_snack_eaters = snack_eaters_after_2_thirds_left + 100 := by sorry
  -- Prove that these equal 153 overall
  sorry

end snack_eaters_remaining_l1322_132291


namespace sam_paint_cans_l1322_132277

theorem sam_paint_cans : 
  ∀ (cans_per_room : ℝ) (initial_cans remaining_cans : ℕ),
    initial_cans * cans_per_room = 40 ∧
    remaining_cans * cans_per_room = 30 ∧
    initial_cans - remaining_cans = 4 →
    remaining_cans = 12 :=
by sorry

end sam_paint_cans_l1322_132277


namespace solve_digits_l1322_132297

theorem solve_digits : ∃ A B C : ℕ, (A = 1 ∧ B = 0 ∧ (C = 9 ∨ C = 1)) ∧ 
  (∃ (X : ℕ), X ≥ 2 ∧ (C = X - 1 ∨ C = 1)) ∧ 
  (A * 1000 + B * 100 + B * 10 + C) * (C * 100 + C * 10 + A) = C * 100000 + C * 10000 + C * 1000 + C * 100 + A * 10 + C :=
by sorry

end solve_digits_l1322_132297


namespace initial_number_of_persons_l1322_132246

theorem initial_number_of_persons (n : ℕ) (h1 : ∀ n, (2.5 : ℝ) * n = 20) : n = 8 := sorry

end initial_number_of_persons_l1322_132246


namespace inequality_range_of_a_l1322_132281

theorem inequality_range_of_a (a : ℝ) :
  (∀ x y : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (1 ≤ y ∧ y ≤ 3) → 2 * x^2 - a * x * y + y^2 ≥ 0) →
  a ≤ 2 * Real.sqrt 2 :=
by
  intros h
  sorry

end inequality_range_of_a_l1322_132281


namespace inequality_proof_l1322_132286

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y + y * z + z * x = 1) :
  3 - Real.sqrt 3 + (x^2 / y) + (y^2 / z) + (z^2 / x) ≥ (x + y + z)^2 :=
by
  sorry

end inequality_proof_l1322_132286


namespace total_puzzle_pieces_l1322_132254

theorem total_puzzle_pieces : 
  ∀ (p1 p2 p3 : ℕ), 
  p1 = 1000 → 
  p2 = p1 + p1 / 2 → 
  p3 = p1 + p1 / 2 → 
  p1 + p2 + p3 = 4000 := 
by 
  intros p1 p2 p3 
  intro h1 
  intro h2 
  intro h3 
  rw [h1, h2, h3] 
  norm_num
  sorry

end total_puzzle_pieces_l1322_132254


namespace ratio_of_men_to_women_l1322_132293

-- Define constants
def total_people : ℕ := 60
def men_in_meeting : ℕ := 4
def women_in_meeting : ℕ := 6
def women_reduction_percentage : ℕ := 20

-- Statement of the problem
theorem ratio_of_men_to_women (total_people men_in_meeting women_in_meeting women_reduction_percentage: ℕ)
  (total_people_eq : total_people = 60)
  (men_in_meeting_eq : men_in_meeting = 4)
  (women_in_meeting_eq : women_in_meeting = 6)
  (women_reduction_percentage_eq : women_reduction_percentage = 20) :
  (men_in_meeting + ((total_people - men_in_meeting - women_in_meeting) * women_reduction_percentage / 100)) 
  = total_people / 2 :=
sorry

end ratio_of_men_to_women_l1322_132293


namespace sqrt_fraction_subtraction_l1322_132290

theorem sqrt_fraction_subtraction :
  (Real.sqrt (9 / 2) - Real.sqrt (2 / 9)) = (7 * Real.sqrt 2 / 6) :=
by sorry

end sqrt_fraction_subtraction_l1322_132290


namespace largest_angle_in_pentagon_l1322_132215

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
    (hA : A = 60) 
    (hB : B = 85) 
    (hCD : C = D) 
    (hE : E = 2 * C + 15) 
    (sum_angles : A + B + C + D + E = 540) : 
    E = 205 := 
by 
    sorry

end largest_angle_in_pentagon_l1322_132215


namespace max_knights_cannot_be_all_liars_l1322_132211

-- Define the conditions of the problem
structure Student :=
  (is_knight : Bool)
  (statement : String)

-- Define the function to check the truthfulness of statements
def is_truthful (s : Student) (conditions : List Student) : Bool :=
  -- Define how to check the statement based on conditions
  sorry

-- The maximum number of knights
theorem max_knights (N : ℕ) (students : List Student) (cond : ∀ s ∈ students, is_truthful s students = true ↔ s.is_knight) :
  ∃ M, M = N := by
  sorry

-- The school cannot be made up entirely of liars
theorem cannot_be_all_liars (N : ℕ) (students : List Student) (cond : ∀ s ∈ students, ¬is_truthful s students) :
  false := by
  sorry

end max_knights_cannot_be_all_liars_l1322_132211


namespace blue_parrots_count_l1322_132257

theorem blue_parrots_count (P : ℕ) (red green blue : ℕ) (h₁ : red = P / 2) (h₂ : green = P / 4) (h₃ : blue = P - red - green) (h₄ :  P + 30 = 150) : blue = 38 :=
by {
-- We will write the proof here
sorry
}

end blue_parrots_count_l1322_132257


namespace max_pens_l1322_132255

theorem max_pens (total_money notebook_cost pen_cost num_notebooks : ℝ) (notebook_qty pen_qty : ℕ):
  total_money = 18 ∧ notebook_cost = 3.6 ∧ pen_cost = 3 ∧ num_notebooks = 2 →
  (pen_qty = 1 ∨ pen_qty = 2 ∨ pen_qty = 3) ↔ (2 * notebook_cost + pen_qty * pen_cost ≤ total_money) :=
by {
  sorry
}

end max_pens_l1322_132255


namespace johns_height_l1322_132214

theorem johns_height
  (L R J : ℕ)
  (h1 : J = L + 15)
  (h2 : J = R - 6)
  (h3 : L + R = 295) :
  J = 152 :=
by sorry

end johns_height_l1322_132214


namespace small_denominator_difference_l1322_132295

theorem small_denominator_difference :
  ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧
               (5 : ℚ) / 9 < (p : ℚ) / q ∧
               (p : ℚ) / q < 4 / 7 ∧
               (∀ r, 0 < r → (5 : ℚ) / 9 < (p : ℚ) / r → (p : ℚ) / r < 4 / 7 → q ≤ r) ∧
               q - p = 7 := 
  by
  sorry

end small_denominator_difference_l1322_132295


namespace sufficient_but_not_necessary_for_circle_l1322_132287

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2)) ∧
  ¬(∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2) → (m = 0)) := sorry

end sufficient_but_not_necessary_for_circle_l1322_132287


namespace circle_symmetric_to_line_l1322_132233

theorem circle_symmetric_to_line (m : ℝ) :
  (∃ (x y : ℝ), (x^2 + y^2 - m * x + 3 * y + 3 = 0) ∧ (m * x + y - m = 0))
  → m = 3 :=
by
  sorry

end circle_symmetric_to_line_l1322_132233


namespace sin_six_theta_l1322_132249

theorem sin_six_theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (3 + Complex.I * Real.sqrt 8) / 5) : 
  Real.sin (6 * θ) = - (630 * Real.sqrt 8) / 15625 := by
  sorry

end sin_six_theta_l1322_132249


namespace express_y_in_terms_of_x_l1322_132231

variable (x y : ℝ)

theorem express_y_in_terms_of_x (h : x + y = -1) : y = -1 - x := 
by 
  sorry

end express_y_in_terms_of_x_l1322_132231


namespace solution_10_digit_divisible_by_72_l1322_132240

def attach_digits_to_divisible_72 : Prop :=
  ∃ (a d : ℕ), (a < 10) ∧ (d < 10) ∧ a * 10^9 + 20222023 * 10 + d = 3202220232 ∧ (3202220232 % 72 = 0)

theorem solution_10_digit_divisible_by_72 : attach_digits_to_divisible_72 :=
  sorry

end solution_10_digit_divisible_by_72_l1322_132240


namespace least_sugar_pounds_l1322_132242

theorem least_sugar_pounds (f s : ℕ) (hf1 : f ≥ 7 + s / 2) (hf2 : f ≤ 3 * s) : s ≥ 3 :=
by
  have h : (5 * s) / 2 ≥ 7 := sorry
  have s_ge_3 : s ≥ 3 := sorry
  exact s_ge_3

end least_sugar_pounds_l1322_132242


namespace class_2_3_tree_count_total_tree_count_l1322_132223

-- Definitions based on the given conditions
def class_2_5_trees := 142
def class_2_3_trees := class_2_5_trees - 18

-- Statements to be proved
theorem class_2_3_tree_count :
  class_2_3_trees = 124 :=
sorry

theorem total_tree_count :
  class_2_5_trees + class_2_3_trees = 266 :=
sorry

end class_2_3_tree_count_total_tree_count_l1322_132223


namespace crayons_total_l1322_132253

theorem crayons_total (Billy_crayons : ℝ) (Jane_crayons : ℝ)
  (h1 : Billy_crayons = 62.0) (h2 : Jane_crayons = 52.0) :
  Billy_crayons + Jane_crayons = 114.0 := 
by
  sorry

end crayons_total_l1322_132253


namespace max_female_students_min_people_in_group_l1322_132201

-- Problem 1: Given z = 4, the maximum number of female students is 6
theorem max_female_students (x y : ℕ) (h1 : x > y) (h2 : y > 4) (h3 : x < 8) : y <= 6 :=
sorry

-- Problem 2: The minimum number of people in the group is 12
theorem min_people_in_group (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : 2 * z > x) : 12 <= x + y + z :=
sorry

end max_female_students_min_people_in_group_l1322_132201


namespace cos_of_7pi_over_4_l1322_132202

theorem cos_of_7pi_over_4 : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 :=
by
  sorry

end cos_of_7pi_over_4_l1322_132202


namespace avg_speed_last_40_min_is_70_l1322_132273

noncomputable def avg_speed_last_interval
  (total_distance : ℝ) (total_time : ℝ)
  (speed_first_40_min : ℝ) (time_first_40_min : ℝ)
  (speed_second_40_min : ℝ) (time_second_40_min : ℝ) : ℝ :=
  let time_last_40_min := total_time - (time_first_40_min + time_second_40_min)
  let distance_first_40_min := speed_first_40_min * time_first_40_min
  let distance_second_40_min := speed_second_40_min * time_second_40_min
  let distance_last_40_min := total_distance - (distance_first_40_min + distance_second_40_min)
  distance_last_40_min / time_last_40_min

theorem avg_speed_last_40_min_is_70
  (h_total_distance : total_distance = 120)
  (h_total_time : total_time = 2)
  (h_speed_first_40_min : speed_first_40_min = 50)
  (h_time_first_40_min : time_first_40_min = 2 / 3)
  (h_speed_second_40_min : speed_second_40_min = 60)
  (h_time_second_40_min : time_second_40_min = 2 / 3) :
  avg_speed_last_interval 120 2 50 (2 / 3) 60 (2 / 3) = 70 :=
by
  sorry

end avg_speed_last_40_min_is_70_l1322_132273


namespace price_of_expensive_feed_l1322_132270

theorem price_of_expensive_feed
  (total_weight : ℝ)
  (mix_price_per_pound : ℝ)
  (cheaper_feed_weight : ℝ)
  (cheaper_feed_price_per_pound : ℝ)
  (expensive_feed_price_per_pound : ℝ) :
  total_weight = 27 →
  mix_price_per_pound = 0.26 →
  cheaper_feed_weight = 14.2105263158 →
  cheaper_feed_price_per_pound = 0.17 →
  expensive_feed_price_per_pound = 0.36 :=
by
  intros h1 h2 h3 h4
  sorry

end price_of_expensive_feed_l1322_132270


namespace max_value_of_d_l1322_132209

theorem max_value_of_d : ∀ (d e : ℕ), (∃ (n : ℕ), n = 70733 + 10^4 * d + e ∧ (∃ (k3 k11 : ℤ), n = 3 * k3 ∧ n = 11 * k11) ∧ d = e ∧ d ≤ 9) → d = 2 :=
by 
  -- Given conditions and goals:
  -- 1. The number has the form 7d7,33e which in numerical form is: n = 70733 + 10^4 * d + e
  -- 2. The number n is divisible by 3 and 11.
  -- 3. d and e are digits (0 ≤ d, e ≤ 9).
  -- 4. To maximize the value of d, ensure that the given conditions hold.
  -- Problem: Prove that the maximum value of d for which this holds is 2.
  sorry

end max_value_of_d_l1322_132209


namespace find_other_number_l1322_132262

theorem find_other_number (a b : ℕ) (h_lcm: Nat.lcm a b = 2310) (h_hcf: Nat.gcd a b = 55) (h_a: a = 210) : b = 605 := by
  sorry

end find_other_number_l1322_132262


namespace sum_of_consecutive_integers_l1322_132261

theorem sum_of_consecutive_integers (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
by
  sorry

end sum_of_consecutive_integers_l1322_132261


namespace value_of_expression_l1322_132236

theorem value_of_expression : (1 * 2 * 3 * 4 * 5 * 6 : ℚ) / (1 + 2 + 3 + 4 + 5 + 6) = 240 / 7 := 
by 
  sorry

end value_of_expression_l1322_132236


namespace range_of_a_if_slope_is_obtuse_l1322_132266

theorem range_of_a_if_slope_is_obtuse : 
  ∀ a : ℝ, (a^2 + 2 * a < 0) → -2 < a ∧ a < 0 :=
by
  intro a
  intro h
  sorry

end range_of_a_if_slope_is_obtuse_l1322_132266


namespace find_matrix_N_l1322_132283

open Matrix

variable (u : Fin 3 → ℝ)

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]

-- Define vector v as the fixed vector in the problem
def v : Fin 3 → ℝ := ![7, 3, -9]

-- Define matrix N as the matrix to be found
def N : Matrix (Fin 3) (Fin 3) ℝ := ![![0, 9, 3], ![-9, 0, -7], ![-3, 7, 0]]

-- Define the requirement condition
theorem find_matrix_N :
  ∀ (u : Fin 3 → ℝ), (N.mulVec u) = cross_product v u :=
by
  sorry

end find_matrix_N_l1322_132283


namespace triangle_area_correct_l1322_132206

def vector_a : ℝ × ℝ := (4, -3)
def vector_b : ℝ × ℝ := (-6, 5)
def vector_c : ℝ × ℝ := (2 * -6, 2 * 5)

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * |a.1 * c.2 - a.2 * c.1|

theorem triangle_area_correct :
  area_of_triangle (4, -3) (0, 0) (-12, 10) = 2 := by
  sorry

end triangle_area_correct_l1322_132206


namespace part1_part2_l1322_132235

-- Part 1: Proving the solutions for (x-1)^2 = 49
theorem part1 (x : ℝ) (h : (x - 1)^2 = 49) : x = 8 ∨ x = -6 :=
sorry

-- Part 2: Proving the time for the object to reach the ground
theorem part2 (t : ℝ) (h : 4.9 * t^2 = 10) : t = 10 / 7 :=
sorry

end part1_part2_l1322_132235
