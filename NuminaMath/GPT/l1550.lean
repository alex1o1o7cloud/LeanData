import Mathlib

namespace elena_total_pens_l1550_155086

theorem elena_total_pens 
  (cost_X : ℝ) (cost_Y : ℝ) (total_spent : ℝ) (num_brand_X : ℕ) (num_brand_Y : ℕ) (total_pens : ℕ)
  (h1 : cost_X = 4.0) 
  (h2 : cost_Y = 2.8) 
  (h3 : total_spent = 40.0) 
  (h4 : num_brand_X = 8) 
  (h5 : total_pens = num_brand_X + num_brand_Y) 
  (h6 : total_spent = num_brand_X * cost_X + num_brand_Y * cost_Y) :
  total_pens = 10 :=
sorry

end elena_total_pens_l1550_155086


namespace find_a_plus_b_l1550_155015

def star (a b : ℕ) : ℕ := a^b + a + b

theorem find_a_plus_b (a b : ℕ) (h2a : 2 ≤ a) (h2b : 2 ≤ b) (h_ab : star a b = 20) :
  a + b = 6 :=
sorry

end find_a_plus_b_l1550_155015


namespace line_parallel_to_x_axis_l1550_155063

variable (k : ℝ)

theorem line_parallel_to_x_axis :
  let point1 := (3, 2 * k + 1)
  let point2 := (8, 4 * k - 5)
  (point1.2 = point2.2) ↔ (k = 3) :=
by
  sorry

end line_parallel_to_x_axis_l1550_155063


namespace marks_lost_per_wrong_answer_l1550_155057

theorem marks_lost_per_wrong_answer 
  (marks_per_correct : ℕ)
  (total_questions : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (wrong_answers : ℕ)
  (score_from_correct : ℕ := correct_answers * marks_per_correct)
  (remaining_marks : ℕ := score_from_correct - total_marks)
  (marks_lost_per_wrong : ℕ) :
  total_questions = correct_answers + wrong_answers →
  total_marks = 130 →
  correct_answers = 38 →
  total_questions = 60 →
  marks_per_correct = 4 →
  marks_lost_per_wrong * wrong_answers = remaining_marks →
  marks_lost_per_wrong = 1 := 
sorry

end marks_lost_per_wrong_answer_l1550_155057


namespace mean_equality_and_find_y_l1550_155059

theorem mean_equality_and_find_y : 
  (8 + 9 + 18) / 3 = (15 + (25 / 3)) / 2 :=
by
  sorry

end mean_equality_and_find_y_l1550_155059


namespace percentage_of_students_owning_cats_l1550_155025

def total_students : ℕ := 500
def students_with_cats : ℕ := 75

theorem percentage_of_students_owning_cats (total_students students_with_cats : ℕ) (h_total: total_students = 500) (h_cats: students_with_cats = 75) :
  100 * (students_with_cats / total_students : ℝ) = 15 := by
  sorry

end percentage_of_students_owning_cats_l1550_155025


namespace fixed_point_always_l1550_155021

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2^x + Real.logb a (x + 1) + 3

theorem fixed_point_always (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f 0 a = 4 :=
by
  sorry

end fixed_point_always_l1550_155021


namespace find_r_l1550_155042

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 :=
sorry

end find_r_l1550_155042


namespace sum_single_digit_numbers_l1550_155036

noncomputable def are_single_digit_distinct (a b c d : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_single_digit_numbers :
  ∀ (A B C D : ℕ),
  are_single_digit_distinct A B C D →
  1000 * A + B - (5000 + 10 * C + 9) = 1000 + 100 * D + 93 →
  A + B + C + D = 18 :=
by
  sorry

end sum_single_digit_numbers_l1550_155036


namespace adult_tickets_sold_l1550_155095

open Nat

theorem adult_tickets_sold (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) :
  A = 130 :=
by
  sorry

end adult_tickets_sold_l1550_155095


namespace find_certain_number_l1550_155046

theorem find_certain_number (h1 : 2994 / 14.5 = 171) (h2 : ∃ x : ℝ, x / 1.45 = 17.1) : ∃ x : ℝ, x = 24.795 :=
by
  sorry

end find_certain_number_l1550_155046


namespace identify_INPUT_statement_l1550_155082

/-- Definition of the PRINT statement --/
def is_PRINT_statement (s : String) : Prop := s = "PRINT"

/-- Definition of the INPUT statement --/
def is_INPUT_statement (s : String) : Prop := s = "INPUT"

/-- Definition of the IF statement --/
def is_IF_statement (s : String) : Prop := s = "IF"

/-- Definition of the WHILE statement --/
def is_WHILE_statement (s : String) : Prop := s = "WHILE"

/-- Proof statement that the INPUT statement is the one for input --/
theorem identify_INPUT_statement (s : String) (h1 : is_PRINT_statement "PRINT") (h2: is_INPUT_statement "INPUT") (h3 : is_IF_statement "IF") (h4 : is_WHILE_statement "WHILE") : s = "INPUT" :=
sorry

end identify_INPUT_statement_l1550_155082


namespace machine_initial_value_l1550_155093

-- Conditions
def initial_value (P : ℝ) : Prop := P * (0.75 ^ 2) = 4000

noncomputable def initial_market_value : ℝ := 4000 / (0.75 ^ 2)

-- Proof problem statement
theorem machine_initial_value (P : ℝ) (h : initial_value P) : P = 4000 / (0.75 ^ 2) :=
by
  sorry

end machine_initial_value_l1550_155093


namespace find_number_subtracted_l1550_155053

-- Given a number x, where the ratio of the two natural numbers is 6:5,
-- and another number y is subtracted to both numbers such that the new ratio becomes 5:4,
-- and the larger number exceeds the smaller number by 5,
-- prove that y = 5.
theorem find_number_subtracted (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
by sorry

end find_number_subtracted_l1550_155053


namespace selling_price_l1550_155094

theorem selling_price (profit_percent : ℝ) (cost_price : ℝ) (h_profit : profit_percent = 5) (h_cp : cost_price = 2400) :
  let profit := (profit_percent / 100) * cost_price 
  let selling_price := cost_price + profit
  selling_price = 2520 :=
by
  sorry

end selling_price_l1550_155094


namespace common_difference_of_arithmetic_sequence_l1550_155071

noncomputable def smallest_angle : ℝ := 25
noncomputable def largest_angle : ℝ := 105
noncomputable def num_angles : ℕ := 5

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, (smallest_angle + (num_angles - 1) * d = largest_angle) ∧ d = 20 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l1550_155071


namespace friedEdgeProb_l1550_155056

-- Define a data structure for positions on the grid
inductive Pos
| A1 | A2 | A3 | A4
| B1 | B2 | B3 | B4
| C1 | C2 | C3 | C4
| D1 | D2 | D3 | D4
deriving DecidableEq, Repr

-- Define whether a position is an edge square (excluding corners)
def isEdge : Pos → Prop
| Pos.A2 | Pos.A3 | Pos.B1 | Pos.B4 | Pos.C1 | Pos.C4 | Pos.D2 | Pos.D3 => True
| _ => False

-- Define the initial state and max hops
def initialState := Pos.B2
def maxHops := 5

-- Define the recursive probability function (details omitted for brevity)
noncomputable def probabilityEdge (p : Pos) (hops : Nat) : ℚ := sorry

-- The proof problem statement
theorem friedEdgeProb :
  probabilityEdge initialState maxHops = 94 / 256 := sorry

end friedEdgeProb_l1550_155056


namespace diana_owes_amount_l1550_155002

def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount_owed : ℝ := principal + interest

theorem diana_owes_amount :
  total_amount_owed = 80.25 :=
by
  sorry

end diana_owes_amount_l1550_155002


namespace find_p_series_l1550_155070

theorem find_p_series (p : ℝ) (h : 5 + (5 + p) / 5 + (5 + 2 * p) / 5^2 + (5 + 3 * p) / 5^3 + ∑' (n : ℕ), (5 + (n + 1) * p) / 5^(n + 1) = 10) : p = 16 :=
sorry

end find_p_series_l1550_155070


namespace product_of_five_consecutive_integers_not_square_l1550_155041

theorem product_of_five_consecutive_integers_not_square (n : ℕ) :
  let P := n * (n + 1) * (n + 2) * (n + 3) * (n + 4)
  ∀ k : ℕ, P ≠ k^2 := 
sorry

end product_of_five_consecutive_integers_not_square_l1550_155041


namespace first_day_of_month_is_tuesday_l1550_155024

theorem first_day_of_month_is_tuesday (day23_is_wednesday : (23 % 7 = 3)) : (1 % 7 = 2) :=
sorry

end first_day_of_month_is_tuesday_l1550_155024


namespace N_8_12_eq_288_l1550_155039

-- Definitions for various polygonal numbers
def N3 (n : ℕ) : ℕ := n * (n + 1) / 2
def N4 (n : ℕ) : ℕ := n^2
def N5 (n : ℕ) : ℕ := 3 * n^2 / 2 - n / 2
def N6 (n : ℕ) : ℕ := 2 * n^2 - n

-- General definition conjectured
def N (n k : ℕ) : ℕ := (k - 2) * n^2 / 2 + (4 - k) * n / 2

-- The problem statement to prove N(8, 12) == 288
theorem N_8_12_eq_288 : N 8 12 = 288 := by
  -- We would need the proofs for the definitional equalities and calculation here
  sorry

end N_8_12_eq_288_l1550_155039


namespace find_abs_product_l1550_155090

noncomputable def distinct_nonzero_real (a b c : ℝ) : Prop :=
a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

theorem find_abs_product (a b c : ℝ) (h1 : distinct_nonzero_real a b c) 
(h2 : a + 1/(b^2) = b + 1/(c^2))
(h3 : b + 1/(c^2) = c + 1/(a^2)) :
  |a * b * c| = 1 :=
sorry

end find_abs_product_l1550_155090


namespace stacy_has_2_more_than_triple_steve_l1550_155005

-- Definitions based on the given conditions
def skylar_berries : ℕ := 20
def steve_berries : ℕ := skylar_berries / 2
def stacy_berries : ℕ := 32

-- Statement to be proved
theorem stacy_has_2_more_than_triple_steve :
  stacy_berries = 3 * steve_berries + 2 := by
  sorry

end stacy_has_2_more_than_triple_steve_l1550_155005


namespace phi_value_l1550_155026

open Real

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h : |φ| < π / 2) :
  (∀ x : ℝ, f (x + π / 3) φ = f (-(x + π / 3)) φ) → φ = -(π / 6) :=
by
  intro h'
  sorry

end phi_value_l1550_155026


namespace hyperbola_eccentricity_correct_l1550_155027

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_correct
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) :
  hyperbola_eccentricity a b h_a h_b h_asymptote = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_correct_l1550_155027


namespace arithmetic_and_geometric_mean_l1550_155012

theorem arithmetic_and_geometric_mean (x y : ℝ) (h1: (x + y) / 2 = 20) (h2: Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 :=
sorry

end arithmetic_and_geometric_mean_l1550_155012


namespace graph_properties_l1550_155054

theorem graph_properties (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) (positive_kb : k * b > 0) :
  (∃ (f g : ℝ → ℝ),
    (∀ x, f x = k * x + b) ∧
    (∀ x (hx : x ≠ 0), g x = k * b / x) ∧
    -- Under the given conditions, the graphs must match option (B)
    (True)) := sorry

end graph_properties_l1550_155054


namespace arithmetic_sequence_a3a6_l1550_155088

theorem arithmetic_sequence_a3a6 (a : ℕ → ℤ)
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_inc : ∀ n, a n < a (n + 1))
  (h_eq : a 3 * a 4 = 45): 
  a 2 * a 5 = 13 := 
sorry

end arithmetic_sequence_a3a6_l1550_155088


namespace total_weight_full_l1550_155018

theorem total_weight_full {x y p q : ℝ}
    (h1 : x + (3/4) * y = p)
    (h2 : x + (1/3) * y = q) :
    x + y = (8/5) * p - (3/5) * q :=
by
  sorry

end total_weight_full_l1550_155018


namespace first_term_arithmetic_sequence_l1550_155045

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end first_term_arithmetic_sequence_l1550_155045


namespace people_got_on_at_third_stop_l1550_155073

theorem people_got_on_at_third_stop :
  let people_1st_stop := 10
  let people_off_2nd_stop := 3
  let twice_people_1st_stop := 2 * people_1st_stop
  let people_off_3rd_stop := 18
  let people_after_3rd_stop := 12

  let people_after_1st_stop := people_1st_stop
  let people_after_2nd_stop := (people_after_1st_stop - people_off_2nd_stop) + twice_people_1st_stop
  let people_after_3rd_stop_but_before_new_ones := people_after_2nd_stop - people_off_3rd_stop
  let people_on_at_3rd_stop := people_after_3rd_stop - people_after_3rd_stop_but_before_new_ones

  people_on_at_3rd_stop = 3 := 
by
  sorry

end people_got_on_at_third_stop_l1550_155073


namespace problem_statement_l1550_155009

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem problem_statement : f (f (f (f (f (f 2))))) = 4 :=
by
  sorry

end problem_statement_l1550_155009


namespace parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l1550_155037

open Real

theorem parabola_tangent_perpendicular_m_eq_one (k : ℝ) (hk : k > 0) :
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + m) ∧ (y₂ = k * x₂ + m) ∧ ((x₁ / 2) * (x₂ / 2) = -1)) → m = 1 :=
sorry

theorem parabola_min_MF_NF (k : ℝ) (hk : k > 0) :
  (m = 2) → 
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + 2) ∧ (y₂ = k * x₂ + 2) ∧ |(y₁ + 1) * (y₂ + 1)| ≥ 9) :=
sorry

end parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l1550_155037


namespace base_7_to_base_10_l1550_155030

theorem base_7_to_base_10 :
  (3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 162 :=
by
  sorry

end base_7_to_base_10_l1550_155030


namespace perfect_squares_in_interval_l1550_155083

theorem perfect_squares_in_interval (s : Set Int) (h1 : ∃ a : Nat, ∀ x ∈ s, a^4 ≤ x ∧ x ≤ (a+9)^4)
                                     (h2 : ∃ b : Nat, ∀ x ∈ s, b^3 ≤ x ∧ x ≤ (b+99)^3) :
  ∃ c : Nat, c ≥ 2000 ∧ ∀ x ∈ s, x = c^2 :=
sorry

end perfect_squares_in_interval_l1550_155083


namespace vector_perpendicular_to_a_l1550_155014

theorem vector_perpendicular_to_a :
  let a := (4, 3)
  let b := (3, -4)
  a.1 * b.1 + a.2 * b.2 = 0 := by
  let a := (4, 3)
  let b := (3, -4)
  sorry

end vector_perpendicular_to_a_l1550_155014


namespace trigonometric_identity_l1550_155016

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π + α) = -1/3) : Real.sin (2 * α) / Real.cos α = 2 / 3 := by
  sorry

end trigonometric_identity_l1550_155016


namespace largest_positive_integer_n_l1550_155058

def binary_operation (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_n (x : ℤ) (h : x = -15) : 
  ∃ (n : ℤ), n > 0 ∧ binary_operation n < x ∧ ∀ m > 0, binary_operation m < x → m ≤ n :=
by
  sorry

end largest_positive_integer_n_l1550_155058


namespace chickens_after_9_years_l1550_155052

-- Definitions from the conditions
def annual_increase : ℕ := 150
def current_chickens : ℕ := 550
def years : ℕ := 9

-- Lean statement for the proof
theorem chickens_after_9_years : current_chickens + annual_increase * years = 1900 :=
by
  sorry

end chickens_after_9_years_l1550_155052


namespace total_toys_l1550_155081

theorem total_toys (m a t : ℕ) (h1 : a = m + 3 * m) (h2 : t = a + 2) (h3 : m = 6) : m + a + t = 56 := by
  sorry

end total_toys_l1550_155081


namespace cost_price_per_meter_of_cloth_l1550_155048

theorem cost_price_per_meter_of_cloth
  (meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) (total_profit : ℕ) (cost_price : ℕ)
  (meters_eq : meters = 80)
  (selling_price_eq : selling_price = 10000)
  (profit_per_meter_eq : profit_per_meter = 7)
  (total_profit_eq : total_profit = profit_per_meter * meters)
  (selling_price_calc : selling_price = cost_price + total_profit)
  (cost_price_calc : cost_price = selling_price - total_profit)
  : (selling_price - total_profit) / meters = 118 :=
by
  -- here we would provide the proof, but we skip it with sorry
  sorry

end cost_price_per_meter_of_cloth_l1550_155048


namespace distance_between_vertices_hyperbola_l1550_155078

-- Defining the hyperbola equation and necessary constants
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2) / 64 - (y^2) / 81 = 1

-- Proving the distance between the vertices is 16
theorem distance_between_vertices_hyperbola : ∀ x y : ℝ, hyperbola_eq x y → 16 = 16 :=
by
  intros x y h
  sorry

end distance_between_vertices_hyperbola_l1550_155078


namespace problem_statement_l1550_155098

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l1550_155098


namespace least_prime_factor_of_5pow6_minus_5pow4_l1550_155006

def least_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then (Nat.minFac n) else 0

theorem least_prime_factor_of_5pow6_minus_5pow4 : least_prime_factor (5^6 - 5^4) = 2 := by
  sorry

end least_prime_factor_of_5pow6_minus_5pow4_l1550_155006


namespace solution_concentration_l1550_155080

theorem solution_concentration (C : ℝ) :
  (0.16 + 0.01 * C * 2 = 0.36) ↔ (C = 10) :=
by
  sorry

end solution_concentration_l1550_155080


namespace joohyeon_snack_count_l1550_155020

theorem joohyeon_snack_count
  (c s : ℕ)
  (h1 : 300 * c + 500 * s = 3000)
  (h2 : c + s = 8) :
  s = 3 :=
sorry

end joohyeon_snack_count_l1550_155020


namespace ratio_of_inverse_l1550_155099

theorem ratio_of_inverse (a b c d : ℝ) (h : ∀ x, (3 * (a * x + b) / (c * x + d) - 2) / ((a * x + b) / (c * x + d) + 4) = x) : 
  a / c = -4 :=
sorry

end ratio_of_inverse_l1550_155099


namespace PhenotypicallyNormalDaughterProbability_l1550_155032

-- Definitions based on conditions
def HemophiliaSexLinkedRecessive := true
def PhenylketonuriaAutosomalRecessive := true
def CouplePhenotypicallyNormal := true
def SonWithBothHemophiliaPhenylketonuria := true

-- Definition of the problem
theorem PhenotypicallyNormalDaughterProbability
  (HemophiliaSexLinkedRecessive : Prop)
  (PhenylketonuriaAutosomalRecessive : Prop)
  (CouplePhenotypicallyNormal : Prop)
  (SonWithBothHemophiliaPhenylketonuria : Prop) :
  -- The correct answer from the solution
  ∃ p : ℚ, p = 3/4 :=
  sorry

end PhenotypicallyNormalDaughterProbability_l1550_155032


namespace find_a_l1550_155004

noncomputable def givenConditions (a b c R : ℝ) : Prop :=
  (a^2 / (b * c) - c / b - b / c = Real.sqrt 3) ∧ (R = 3)

theorem find_a (a b c : ℝ) (R : ℝ) (h : givenConditions a b c R) : a = 3 :=
by
  sorry

end find_a_l1550_155004


namespace find_t_l1550_155065

-- Definitions from the given conditions
def earning (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

-- The main theorem based on the translated problem
theorem find_t
  (t : ℕ)
  (h1 : earning (t - 4) (3 * t - 7) = earning (3 * t - 12) (t - 3)) :
  t = 4 := 
sorry

end find_t_l1550_155065


namespace min_value_of_y_l1550_155013

variable {x k : ℝ}

theorem min_value_of_y (h₁ : ∀ x > 0, 0 < k) 
  (h₂ : ∀ x > 0, (x^2 + k / x) ≥ 3) : k = 2 :=
sorry

end min_value_of_y_l1550_155013


namespace integer_solutions_range_l1550_155034

theorem integer_solutions_range (a : ℝ) :
  (∀ x : ℤ, x^2 - x + a - a^2 < 0 → x + 2 * a > 1) ↔ 1 < a ∧ a ≤ 2 := sorry

end integer_solutions_range_l1550_155034


namespace mock_exam_girls_count_l1550_155066

theorem mock_exam_girls_count
  (B G Bc Gc : ℕ)
  (h1: B + G = 400)
  (h2: Bc = 60 * B / 100)
  (h3: Gc = 80 * G / 100)
  (h4: Bc + Gc = 65 * 400 / 100)
  : G = 100 :=
sorry

end mock_exam_girls_count_l1550_155066


namespace binkie_gemstones_l1550_155091

variables (F B S : ℕ)

theorem binkie_gemstones :
  (B = 4 * F) →
  (S = (1 / 2 : ℝ) * F - 2) →
  (S = 1) →
  B = 24 :=
by
  sorry

end binkie_gemstones_l1550_155091


namespace mowing_time_l1550_155022

theorem mowing_time (length width: ℝ) (swath_width_overlap_rate: ℝ)
                    (walking_speed: ℝ) (ft_per_inch: ℝ)
                    (length_eq: length = 100)
                    (width_eq: width = 120)
                    (swath_eq: swath_width_overlap_rate = 24)
                    (walking_eq: walking_speed = 4500)
                    (conversion_eq: ft_per_inch = 1/12) :
                    (length / walking_speed) * (width / (swath_width_overlap_rate * ft_per_inch)) = 1.33 :=
by
    rw [length_eq, width_eq, swath_eq, walking_eq, conversion_eq]
    exact sorry

end mowing_time_l1550_155022


namespace false_statement_divisibility_l1550_155008

-- Definitions for the divisibility conditions
def divisible_by (a b : ℕ) : Prop := ∃ k, b = a * k

-- The problem statement
theorem false_statement_divisibility (N : ℕ) :
  (divisible_by 2 N ∧ divisible_by 4 N ∧ divisible_by 12 N ∧ ¬ divisible_by 24 N) →
  (¬ divisible_by 24 N) :=
by
  -- The proof will need to be filled in here
  sorry

end false_statement_divisibility_l1550_155008


namespace function_range_l1550_155092

def function_defined (x : ℝ) : Prop := x ≠ 5

theorem function_range (x : ℝ) : x ≠ 5 → function_defined x :=
by
  intro h
  exact h

end function_range_l1550_155092


namespace inv_38_mod_53_l1550_155075

theorem inv_38_mod_53 (h : 15 * 31 % 53 = 1) : ∃ x : ℤ, 38 * x % 53 = 1 ∧ (x % 53 = 22) :=
by
  sorry

end inv_38_mod_53_l1550_155075


namespace total_books_l1550_155019

-- Define the number of books Tim has
def TimBooks : ℕ := 44

-- Define the number of books Sam has
def SamBooks : ℕ := 52

-- Statement to prove that the total number of books is 96
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end total_books_l1550_155019


namespace sum_of_squares_pattern_l1550_155074

theorem sum_of_squares_pattern (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end sum_of_squares_pattern_l1550_155074


namespace students_with_B_l1550_155097

theorem students_with_B (students_jacob : ℕ) (students_B_jacob : ℕ) (students_smith : ℕ) (ratio_same : (students_B_jacob / students_jacob : ℚ) = 2 / 5) : 
  ∃ y : ℕ, (y / students_smith : ℚ) = 2 / 5 ∧ y = 12 :=
by 
  use 12
  sorry

end students_with_B_l1550_155097


namespace find_a2023_l1550_155051

theorem find_a2023
  (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2/5)
  (h3 : a 3 = 1/4)
  (h_rule : ∀ n : ℕ, 0 < n → (1 / a n + 1 / a (n + 2) = 2 / a (n + 1))) :
  a 2023 = 1 / 3034 :=
by sorry

end find_a2023_l1550_155051


namespace find_q_l1550_155023

theorem find_q (p q : ℝ) (h : ∀ x : ℝ, (x^2 + p * x + q) ≥ 1) : q = 1 + (p^2 / 4) :=
sorry

end find_q_l1550_155023


namespace geometric_seq_fourth_term_l1550_155029

-- Define the conditions
def first_term (a1 : ℝ) : Prop := a1 = 512
def sixth_term (a1 r : ℝ) : Prop := a1 * r^5 = 32

-- Define the claim
def fourth_term (a1 r a4 : ℝ) : Prop := a4 = a1 * r^3

-- State the theorem
theorem geometric_seq_fourth_term :
  ∀ a1 r a4 : ℝ, first_term a1 → sixth_term a1 r → fourth_term a1 r a4 → a4 = 64 :=
by
  intros a1 r a4 h1 h2 h3
  rw [first_term, sixth_term, fourth_term] at *
  sorry

end geometric_seq_fourth_term_l1550_155029


namespace simplify_expression_l1550_155031

variable (a b : ℚ)

theorem simplify_expression (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  (2 * a + 1) / (1 - b / (2 * b - 1)) = (2 * a + 1) * (2 * b - 1) / (b - 1) :=
by 
  sorry

end simplify_expression_l1550_155031


namespace subset_proof_l1550_155055

-- Define set M
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 1)}

-- The problem statement
theorem subset_proof : M ⊆ N ∧ ∃ y ∈ N, y ∉ M :=
by
  sorry

end subset_proof_l1550_155055


namespace arithmetic_sequence_a20_l1550_155043

theorem arithmetic_sequence_a20 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end arithmetic_sequence_a20_l1550_155043


namespace xy_equals_252_l1550_155062

-- Definitions and conditions
variables (x y : ℕ) -- positive integers
variable (h1 : x + y = 36)
variable (h2 : 4 * x * y + 12 * x = 5 * y + 390)

-- Statement of the problem
theorem xy_equals_252 (h1 : x + y = 36) (h2 : 4 * x * y + 12 * x = 5 * y + 390) : x * y = 252 := by 
  sorry

end xy_equals_252_l1550_155062


namespace union_sets_l1550_155011

def setA : Set ℝ := { x | abs (x - 1) < 3 }
def setB : Set ℝ := { x | x^2 - 4 * x < 0 }

theorem union_sets :
  setA ∪ setB = { x : ℝ | -2 < x ∧ x < 4 } :=
sorry

end union_sets_l1550_155011


namespace nabla_2_3_2_eq_4099_l1550_155068

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem nabla_2_3_2_eq_4099 : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end nabla_2_3_2_eq_4099_l1550_155068


namespace axis_of_symmetry_imp_cond_l1550_155064

-- Necessary definitions
variables {p q r s x y : ℝ}

-- Given conditions
def curve_eq (x y p q r s : ℝ) : Prop := y = (2 * p * x + q) / (r * x + 2 * s)
def axis_of_symmetry (x y : ℝ) : Prop := y = x

-- Main statement
theorem axis_of_symmetry_imp_cond (h1 : curve_eq x y p q r s) (h2 : axis_of_symmetry x y) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : p = -2 * s :=
sorry

end axis_of_symmetry_imp_cond_l1550_155064


namespace solution_exists_l1550_155076

theorem solution_exists (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (gcd_ca : Nat.gcd c a = 1) (gcd_cb : Nat.gcd c b = 1) : 
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^a + y^b = z^c :=
sorry

end solution_exists_l1550_155076


namespace function_identity_l1550_155087

theorem function_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end function_identity_l1550_155087


namespace T_simplified_l1550_155079

-- Define the polynomial expression T
def T (x : ℝ) : ℝ := (x-2)^4 - 4*(x-2)^3 + 6*(x-2)^2 - 4*(x-2) + 1

-- Prove that T simplifies to (x-3)^4
theorem T_simplified (x : ℝ) : T x = (x - 3)^4 := by
  sorry

end T_simplified_l1550_155079


namespace exists_monochromatic_rectangle_l1550_155077

theorem exists_monochromatic_rectangle 
  (coloring : ℤ × ℤ → Prop)
  (h : ∀ p : ℤ × ℤ, coloring p = red ∨ coloring p = blue)
  : ∃ (a b c d : ℤ × ℤ), (a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2) ∧ (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end exists_monochromatic_rectangle_l1550_155077


namespace smallest_prime_divides_l1550_155007

theorem smallest_prime_divides (p : ℕ) (a : ℕ) 
  (h1 : Prime p) (h2 : p > 100) (h3 : a > 1) (h4 : p ∣ (a^89 - 1) / (a - 1)) :
  p = 179 := 
sorry

end smallest_prime_divides_l1550_155007


namespace hyperbola_range_m_l1550_155028

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m + 2 > 0 ∧ m - 2 < 0) ∧ (x^2 / (m + 2) + y^2 / (m - 2) = 1)) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end hyperbola_range_m_l1550_155028


namespace PQRS_value_l1550_155035

theorem PQRS_value :
  let P := (Real.sqrt 2011 + Real.sqrt 2010)
  let Q := (-Real.sqrt 2011 - Real.sqrt 2010)
  let R := (Real.sqrt 2011 - Real.sqrt 2010)
  let S := (Real.sqrt 2010 - Real.sqrt 2011)
  P * Q * R * S = -1 :=
by
  sorry

end PQRS_value_l1550_155035


namespace sufficient_but_not_necessary_condition_l1550_155047

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 + x - 2 > 0) ∧ (∃ y, y < -2 ∧ y^2 + y - 2 > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1550_155047


namespace percent_of_value_l1550_155089

theorem percent_of_value (decimal_form : Real) (value : Nat) (expected_result : Real) : 
  decimal_form = 0.25 ∧ value = 300 ∧ expected_result = 75 → 
  decimal_form * value = expected_result := by
  sorry

end percent_of_value_l1550_155089


namespace intermediate_value_theorem_example_l1550_155044

theorem intermediate_value_theorem_example (f : ℝ → ℝ) :
  f 2007 < 0 → f 2008 < 0 → f 2009 > 0 → ∃ x, 2007 < x ∧ x < 2008 ∧ f x = 0 :=
by
  sorry

end intermediate_value_theorem_example_l1550_155044


namespace area_at_stage_7_l1550_155061

-- Define the size of one square added at each stage
def square_size : ℕ := 4

-- Define the area of one square
def area_of_one_square : ℕ := square_size * square_size

-- Define the number of stages
def number_of_stages : ℕ := 7

-- Define the total area at a given stage
def total_area (n : ℕ) : ℕ := n * area_of_one_square

-- The theorem which proves the area of the rectangle at Stage 7
theorem area_at_stage_7 : total_area number_of_stages = 112 :=
by
  -- proof goes here
  sorry

end area_at_stage_7_l1550_155061


namespace max_minus_min_all_three_languages_l1550_155060

def student_population := 1500

def english_students (e : ℕ) : Prop := 1050 ≤ e ∧ e ≤ 1125
def spanish_students (s : ℕ) : Prop := 750 ≤ s ∧ s ≤ 900
def german_students (g : ℕ) : Prop := 300 ≤ g ∧ g ≤ 450

theorem max_minus_min_all_three_languages (e s g e_s e_g s_g e_s_g : ℕ) 
    (he : english_students e)
    (hs : spanish_students s)
    (hg : german_students g)
    (pie : e + s + g - e_s - e_g - s_g + e_s_g = student_population) 
    : (M - m = 450) :=
sorry

end max_minus_min_all_three_languages_l1550_155060


namespace frank_sales_quota_l1550_155033

theorem frank_sales_quota (x : ℕ) :
  (3 * x + 12 + 23 = 50) → x = 5 :=
by sorry

end frank_sales_quota_l1550_155033


namespace find_salary_l1550_155050

theorem find_salary (x y : ℝ) (h1 : x + y = 2000) (h2 : 0.05 * x = 0.15 * y) : x = 1500 :=
sorry

end find_salary_l1550_155050


namespace not_product_of_two_integers_l1550_155000

theorem not_product_of_two_integers (n : ℕ) (hn : n > 0) :
  ∀ t k : ℕ, t * (t + k) = n^2 + n + 1 → k ≥ 2 * Nat.sqrt n :=
by
  sorry

end not_product_of_two_integers_l1550_155000


namespace negation_seated_l1550_155085

variable (Person : Type) (in_room : Person → Prop) (seated : Person → Prop)

theorem negation_seated :
  ¬ (∀ x, in_room x → seated x) ↔ ∃ x, in_room x ∧ ¬ seated x :=
by sorry

end negation_seated_l1550_155085


namespace find_a_l1550_155067

theorem find_a (a x y : ℝ) 
  (h1 : (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0) 
  (h2 : (x + 2)^2 + (y + 4)^2 = a) 
  (h3 : ∃! x y, (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0 ∧ (x + 2)^2 + (y + 4)^2 = a) :
  a = 9 ∨ a = 23 + 4 * Real.sqrt 15 :=
sorry

end find_a_l1550_155067


namespace range_of_a_for_negative_root_l1550_155084

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 7^(x + 1) - 7^x * a - a - 5 = 0) ↔ -5 < a ∧ a < 1 :=
by
  sorry

end range_of_a_for_negative_root_l1550_155084


namespace mixed_alcohol_solution_l1550_155017

theorem mixed_alcohol_solution 
    (vol_x : ℝ) (vol_y : ℝ) (conc_x : ℝ) (conc_y : ℝ) (target_conc : ℝ) (vol_y_given : vol_y = 750) 
    (conc_x_given : conc_x = 0.10) (conc_y_given : conc_y = 0.30) (target_conc_given : target_conc = 0.25) : 
    vol_x = 250 → 
    (conc_x * vol_x + conc_y * vol_y) / (vol_x + vol_y) = target_conc :=
by
  intros h_x
  rw [vol_y_given, conc_x_given, conc_y_given, target_conc_given, h_x]
  sorry

end mixed_alcohol_solution_l1550_155017


namespace intersection_points_l1550_155038

-- Define the four line equations
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2
def line4 (x y : ℝ) : Prop := 5 * x - 15 * y = 15

-- State the theorem for intersection points
theorem intersection_points : 
  (line1 (18/11) (13/11) ∧ line2 (18/11) (13/11)) ∧ 
  (line2 (21/11) (8/11) ∧ line3 (21/11) (8/11)) :=
by
  sorry

end intersection_points_l1550_155038


namespace milk_fraction_correct_l1550_155010

def fraction_of_milk_in_coffee_cup (coffee_initial : ℕ) (milk_initial : ℕ) : ℚ :=
  let coffee_transferred := coffee_initial / 3
  let milk_cup_after_transfer := milk_initial + coffee_transferred
  let coffee_left := coffee_initial - coffee_transferred
  let total_mixed := milk_cup_after_transfer
  let transfer_back := total_mixed / 2
  let coffee_back := transfer_back * (coffee_transferred / total_mixed)
  let milk_back := transfer_back * (milk_initial / total_mixed)
  let coffee_final := coffee_left + coffee_back
  let milk_final := milk_back
  milk_final / (coffee_final + milk_final)

theorem milk_fraction_correct (coffee_initial : ℕ) (milk_initial : ℕ)
  (h_coffee : coffee_initial = 6) (h_milk : milk_initial = 3) :
  fraction_of_milk_in_coffee_cup coffee_initial milk_initial = 3 / 13 :=
by
  sorry

end milk_fraction_correct_l1550_155010


namespace product_diff_squares_l1550_155069

theorem product_diff_squares (a b c d x1 y1 x2 y2 x3 y3 x4 y4 : ℕ) 
  (ha : a = x1^2 - y1^2) 
  (hb : b = x2^2 - y2^2) 
  (hc : c = x3^2 - y3^2) 
  (hd : d = x4^2 - y4^2)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  ∃ X Y : ℕ, a * b * c * d = X^2 - Y^2 :=
by
  sorry

end product_diff_squares_l1550_155069


namespace simplify_fractions_sum_l1550_155040

theorem simplify_fractions_sum :
  (48 / 72) + (30 / 45) = 4 / 3 := 
by
  sorry

end simplify_fractions_sum_l1550_155040


namespace range_of_a_l1550_155049

theorem range_of_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5})
  (hB : B = {x | 3 ≤ x ∧ x ≤ 22}) :
  A ⊆ (A ∩ B) ↔ (1 ≤ a ∧ a ≤ 9) :=
by
  sorry

end range_of_a_l1550_155049


namespace allergic_reaction_probability_is_50_percent_l1550_155003

def can_have_allergic_reaction (choice : String) : Prop :=
  choice = "peanut_butter"

def percentage_of_allergic_reaction :=
  let total_peanut_butter := 40 + 30
  let total_cookies := 40 + 50 + 30 + 20
  (total_peanut_butter : Float) / (total_cookies : Float) * 100

theorem allergic_reaction_probability_is_50_percent :
  percentage_of_allergic_reaction = 50 := sorry

end allergic_reaction_probability_is_50_percent_l1550_155003


namespace number_a_eq_223_l1550_155072

theorem number_a_eq_223 (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end number_a_eq_223_l1550_155072


namespace discount_difference_l1550_155001

theorem discount_difference (p : ℝ) (single_discount first_discount second_discount : ℝ) :
    p = 12000 →
    single_discount = 0.45 →
    first_discount = 0.35 →
    second_discount = 0.10 →
    (p * (1 - single_discount) - p * (1 - first_discount) * (1 - second_discount) = 420) := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end discount_difference_l1550_155001


namespace An_is_integer_for_all_n_l1550_155096

noncomputable def sin_theta (a b : ℕ) : ℝ :=
  if h : a^2 + b^2 ≠ 0 then (2 * a * b) / (a^2 + b^2) else 0

theorem An_is_integer_for_all_n (a b : ℕ) (n : ℕ) (h₁ : a > b) (h₂ : 0 < sin_theta a b) (h₃ : sin_theta a b < 1) :
  ∃ k : ℤ, ∀ n : ℕ, ((a^2 + b^2)^n * sin_theta a b) = k :=
sorry

end An_is_integer_for_all_n_l1550_155096
