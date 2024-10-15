import Mathlib

namespace NUMINAMATH_GPT_inequality_proof_l1651_165109

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l1651_165109


namespace NUMINAMATH_GPT_min_value_of_a_l1651_165159

theorem min_value_of_a (a : ℝ) (x : ℝ) (h1: 0 < a) (h2: a ≠ 1) (h3: 1 ≤ x → a^x ≥ a * x) : a ≥ Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_l1651_165159


namespace NUMINAMATH_GPT_intersection_of_sets_l1651_165195

def setP : Set ℝ := { x | x ≤ 3 }
def setQ : Set ℝ := { x | x > 1 }

theorem intersection_of_sets : setP ∩ setQ = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1651_165195


namespace NUMINAMATH_GPT_rides_total_l1651_165152

theorem rides_total (rides_day1 rides_day2 : ℕ) (h1 : rides_day1 = 4) (h2 : rides_day2 = 3) : rides_day1 + rides_day2 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_rides_total_l1651_165152


namespace NUMINAMATH_GPT_expenditure_representation_l1651_165161

def income_represented_pos (income : ℤ) : Prop := income > 0

def expenditure_represented_neg (expenditure : ℤ) : Prop := expenditure < 0

theorem expenditure_representation (income expenditure : ℤ) (h_income: income_represented_pos income) (exp_value: expenditure = 3) : expenditure_represented_neg expenditure := 
sorry

end NUMINAMATH_GPT_expenditure_representation_l1651_165161


namespace NUMINAMATH_GPT_neg_p_iff_a_in_0_1_l1651_165140

theorem neg_p_iff_a_in_0_1 (a : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) ∧ (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_neg_p_iff_a_in_0_1_l1651_165140


namespace NUMINAMATH_GPT_geometric_sum_2015_2016_l1651_165189

theorem geometric_sum_2015_2016 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 2)
  (h_a2_a5 : a 2 + a 5 = 0)
  (h_Sn : ∀ n, S n = (1 - (-1)^n)) :
  S 2015 + S 2016 = 2 :=
by sorry

end NUMINAMATH_GPT_geometric_sum_2015_2016_l1651_165189


namespace NUMINAMATH_GPT_boat_travel_time_downstream_l1651_165143

theorem boat_travel_time_downstream
  (v c: ℝ)
  (h1: c = 1)
  (h2: 24 / (v - c) = 6): 
  24 / (v + c) = 4 := 
by
  sorry

end NUMINAMATH_GPT_boat_travel_time_downstream_l1651_165143


namespace NUMINAMATH_GPT_number_of_regular_soda_bottles_l1651_165170

-- Define the total number of bottles and the number of diet soda bottles
def total_bottles : ℕ := 30
def diet_soda_bottles : ℕ := 2

-- Define the number of regular soda bottles
def regular_soda_bottles : ℕ := total_bottles - diet_soda_bottles

-- Statement of the main proof problem
theorem number_of_regular_soda_bottles : regular_soda_bottles = 28 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_regular_soda_bottles_l1651_165170


namespace NUMINAMATH_GPT_num_positive_integers_m_l1651_165190

theorem num_positive_integers_m (h : ∀ m : ℕ, ∃ d : ℕ, 3087 = d ∧ m^2 = d + 3) :
  ∃! m : ℕ, 0 < m ∧ (3087 % (m^2 - 3) = 0) := by
  sorry

end NUMINAMATH_GPT_num_positive_integers_m_l1651_165190


namespace NUMINAMATH_GPT_min_cells_marked_l1651_165188

/-- The minimum number of cells that need to be marked in a 50x50 grid so
each 1x6 vertical or horizontal strip has at least one marked cell is 416. -/
theorem min_cells_marked {n : ℕ} : n = 416 → 
  (∀ grid : Fin 50 × Fin 50, ∃ cells : Finset (Fin 50 × Fin 50), 
    (∀ (r c : Fin 50), (r = 6 * i + k ∨ c = 6 * i + k) →
      (∃ (cell : Fin 50 × Fin 50), cell ∈ cells)) →
    cells.card = n) := 
sorry

end NUMINAMATH_GPT_min_cells_marked_l1651_165188


namespace NUMINAMATH_GPT_problem1_problem2_l1651_165142

namespace ArithmeticSequence

-- Part (1)
theorem problem1 (a1 : ℚ) (d : ℚ) (S_n : ℚ) (n : ℕ) (a_n : ℚ) 
  (h1 : a1 = 5 / 6) 
  (h2 : d = -1 / 6) 
  (h3 : S_n = -5) 
  (h4 : S_n = n * (2 * a1 + (n - 1) * d) / 2) 
  (h5 : a_n = a1 + (n - 1) * d) : 
  (n = 15) ∧ (a_n = -3 / 2) :=
sorry

-- Part (2)
theorem problem2 (d : ℚ) (n : ℕ) (a_n : ℚ) (a1 : ℚ) (S_n : ℚ)
  (h1 : d = 2) 
  (h2 : n = 15) 
  (h3 : a_n = -10) 
  (h4 : a_n = a1 + (n - 1) * d) 
  (h5 : S_n = n * (2 * a1 + (n - 1) * d) / 2) : 
  (a1 = -38) ∧ (S_n = -360) :=
sorry

end ArithmeticSequence

end NUMINAMATH_GPT_problem1_problem2_l1651_165142


namespace NUMINAMATH_GPT_mildred_weight_l1651_165114

theorem mildred_weight (carol_weight mildred_is_heavier : ℕ) (h1 : carol_weight = 9) (h2 : mildred_is_heavier = 50) :
  carol_weight + mildred_is_heavier = 59 :=
by
  sorry

end NUMINAMATH_GPT_mildred_weight_l1651_165114


namespace NUMINAMATH_GPT_avg_score_all_matches_l1651_165183

-- Definitions from the conditions
variable (score1 score2 : ℕ → ℕ) 
variable (avg1 avg2 : ℕ)
variable (count1 count2 : ℕ)

-- Assumptions from the conditions
axiom avg_score1 : avg1 = 30
axiom avg_score2 : avg2 = 40
axiom count1_matches : count1 = 2
axiom count2_matches : count2 = 3

-- The proof statement
theorem avg_score_all_matches : 
  ((score1 0 + score1 1) + (score2 0 + score2 1 + score2 2)) / (count1 + count2) = 36 := 
  sorry

end NUMINAMATH_GPT_avg_score_all_matches_l1651_165183


namespace NUMINAMATH_GPT_initial_amount_in_cookie_jar_l1651_165141

theorem initial_amount_in_cookie_jar (doris_spent : ℕ) (martha_spent : ℕ) (amount_left : ℕ) (spent_eq_martha : martha_spent = doris_spent / 2) (amount_left_eq : amount_left = 12) (doris_spent_eq : doris_spent = 6) : (doris_spent + martha_spent + amount_left = 21) :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_in_cookie_jar_l1651_165141


namespace NUMINAMATH_GPT_negation_of_proposition_l1651_165186

theorem negation_of_proposition (p : Real → Prop) : 
  (∀ x : Real, p x) → ¬(∀ x : Real, x ≥ 1) ↔ (∃ x : Real, x < 1) := 
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l1651_165186


namespace NUMINAMATH_GPT_find_S6_l1651_165105

def arithmetic_sum (n : ℕ) : ℝ := sorry
def S_3 := 6
def S_9 := 27

theorem find_S6 : ∃ S_6 : ℝ, S_6 = 15 ∧ 
                              S_6 - S_3 = (6 + (S_9 - S_6)) / 2 :=
sorry

end NUMINAMATH_GPT_find_S6_l1651_165105


namespace NUMINAMATH_GPT_marie_keeps_lollipops_l1651_165193

def total_lollipops (raspberry mint blueberry coconut : ℕ) : ℕ :=
  raspberry + mint + blueberry + coconut

def lollipops_per_friend (total friends : ℕ) : ℕ :=
  total / friends

def lollipops_kept (total friends : ℕ) : ℕ :=
  total % friends

theorem marie_keeps_lollipops :
  lollipops_kept (total_lollipops 75 132 9 315) 13 = 11 :=
by
  sorry

end NUMINAMATH_GPT_marie_keeps_lollipops_l1651_165193


namespace NUMINAMATH_GPT_intersection_P_Q_l1651_165132

def P := {x : ℤ | x^2 - 16 < 0}
def Q := {x : ℤ | ∃ n : ℤ, x = 2 * n}

theorem intersection_P_Q :
  P ∩ Q = {-2, 0, 2} :=
sorry

end NUMINAMATH_GPT_intersection_P_Q_l1651_165132


namespace NUMINAMATH_GPT_age_of_b_l1651_165106

variable (a b c : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = 2 * c
def condition3 : Prop := a + b + c = 27

theorem age_of_b (h1 : condition1 a b)
                 (h2 : condition2 b c)
                 (h3 : condition3 a b c) : 
                 b = 10 := 
by sorry

end NUMINAMATH_GPT_age_of_b_l1651_165106


namespace NUMINAMATH_GPT_polynomial_square_solution_l1651_165180

variable (a b : ℝ)

theorem polynomial_square_solution (h : 
  ∃ g : Polynomial ℝ, g^2 = Polynomial.C (1 : ℝ) * Polynomial.X^4 -
  Polynomial.C (1 : ℝ) * Polynomial.X^3 +
  Polynomial.C (1 : ℝ) * Polynomial.X^2 +
  Polynomial.C a * Polynomial.X +
  Polynomial.C b) : b = 9 / 64 :=
by sorry

end NUMINAMATH_GPT_polynomial_square_solution_l1651_165180


namespace NUMINAMATH_GPT_sequence_length_arithmetic_sequence_l1651_165130

theorem sequence_length_arithmetic_sequence :
  ∃ n : ℕ, ∀ (a d : ℕ), a = 2 → d = 3 → a + (n - 1) * d = 2014 ∧ n = 671 :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_length_arithmetic_sequence_l1651_165130


namespace NUMINAMATH_GPT_count_integers_in_solution_set_l1651_165187

-- Define the predicate for the condition given in the problem
def condition (x : ℝ) : Prop := abs (x - 3) ≤ 4.5

-- Define the list of integers within the range of the condition
def solution_set : List ℤ := [-1, 0, 1, 2, 3, 4, 5, 6, 7]

-- Prove that the number of integers satisfying the condition is 8
theorem count_integers_in_solution_set : solution_set.length = 8 :=
by
  sorry

end NUMINAMATH_GPT_count_integers_in_solution_set_l1651_165187


namespace NUMINAMATH_GPT_votes_combined_l1651_165158

theorem votes_combined (vote_A vote_B : ℕ) (h_ratio : vote_A = 2 * vote_B) (h_A_votes : vote_A = 14) : vote_A + vote_B = 21 :=
by
  sorry

end NUMINAMATH_GPT_votes_combined_l1651_165158


namespace NUMINAMATH_GPT_transform_quadratic_equation_l1651_165172

theorem transform_quadratic_equation :
  ∀ x : ℝ, (x^2 - 8 * x - 1 = 0) → ((x - 4)^2 = 17) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_transform_quadratic_equation_l1651_165172


namespace NUMINAMATH_GPT_union_of_A_and_B_l1651_165184

open Set

def A : Set ℕ := {1, 3, 7, 8}
def B : Set ℕ := {1, 5, 8}

theorem union_of_A_and_B : A ∪ B = {1, 3, 5, 7, 8} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1651_165184


namespace NUMINAMATH_GPT_unpainted_cubes_eq_210_l1651_165191

-- Defining the structure of the 6x6x6 cube
def cube := Fin 6 × Fin 6 × Fin 6

-- Number of unit cubes in a 6x6x6 cube
def total_cubes : ℕ := 6 * 6 * 6

-- Number of unit squares painted by the plus pattern on each face
def squares_per_face := 13

-- Number of faces on the cube
def faces := 6

-- Initial total number of painted squares
def initial_painted_squares := squares_per_face * faces

-- Number of over-counted squares along edges
def edge_overcount := 12 * 2

-- Number of over-counted squares at corners
def corner_overcount := 8 * 1

-- Adjusted number of painted unit squares accounting for overcounts
noncomputable def adjusted_painted_squares := initial_painted_squares - edge_overcount - corner_overcount

-- Overlap adjustment: edge units and corner units
def edges_overlap := 24
def corners_overlap := 16

-- Final number of unique painted unit cubes
noncomputable def unique_painted_cubes := adjusted_painted_squares - edges_overlap - corners_overlap

-- Final unpainted unit cubes calculation
noncomputable def unpainted_cubes := total_cubes - unique_painted_cubes

-- Theorem to prove the number of unpainted unit cubes is 210
theorem unpainted_cubes_eq_210 : unpainted_cubes = 210 := by
  sorry

end NUMINAMATH_GPT_unpainted_cubes_eq_210_l1651_165191


namespace NUMINAMATH_GPT_find_y_if_x_l1651_165124

theorem find_y_if_x (x : ℝ) (hx : x^2 + 8 * (x / (x - 3))^2 = 53) :
  (∃ y, y = (x - 3)^3 * (x + 4) / (2 * x - 5) ∧ y = 17000 / 21) :=
  sorry

end NUMINAMATH_GPT_find_y_if_x_l1651_165124


namespace NUMINAMATH_GPT_disjunction_of_false_is_false_l1651_165174

-- Given conditions
variables (p q : Prop)

-- We are given the assumption that both p and q are false propositions
axiom h1 : ¬ p
axiom h2 : ¬ q

-- We want to prove that the disjunction p ∨ q is false
theorem disjunction_of_false_is_false (p q : Prop) (h1 : ¬ p) (h2 : ¬ q) : ¬ (p ∨ q) := 
by
  sorry

end NUMINAMATH_GPT_disjunction_of_false_is_false_l1651_165174


namespace NUMINAMATH_GPT_calculate_expression_l1651_165155

variable (x y : ℝ)

theorem calculate_expression :
  (-2 * x^2 * y)^3 = -8 * x^6 * y^3 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l1651_165155


namespace NUMINAMATH_GPT_problem_bound_l1651_165112

theorem problem_bound (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) : 
  0 ≤ y * z + z * x + x * y - 2 * (x * y * z) ∧ 
  y * z + z * x + x * y - 2 * (x * y * z) ≤ 7 / 27 :=
sorry

end NUMINAMATH_GPT_problem_bound_l1651_165112


namespace NUMINAMATH_GPT_prob_x_lt_y_is_correct_l1651_165122

open Set

noncomputable def prob_x_lt_y : ℝ :=
  let rectangle := Icc (0: ℝ) 4 ×ˢ Icc (0: ℝ) 3
  let area_rectangle := 4 * 3
  let triangle := {p : ℝ × ℝ | p.1 ∈ Icc (0: ℝ) 3 ∧ p.2 ∈ Icc (0: ℝ) 3 ∧ p.1 < p.2}
  let area_triangle := 1 / 2 * 3 * 3
  let probability := area_triangle / area_rectangle
  probability

-- To state as a theorem using Lean's notation
theorem prob_x_lt_y_is_correct : prob_x_lt_y = 3 / 8 := sorry

end NUMINAMATH_GPT_prob_x_lt_y_is_correct_l1651_165122


namespace NUMINAMATH_GPT_algebraic_expression_value_l1651_165101

theorem algebraic_expression_value 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = ab + bc + ac)
  (h2 : a = 1) : 
  (a + b - c) ^ 2004 = 1 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1651_165101


namespace NUMINAMATH_GPT_remainder_of_product_mod_10_l1651_165151

theorem remainder_of_product_mod_10 :
  (1265 * 4233 * 254 * 1729) % 10 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_product_mod_10_l1651_165151


namespace NUMINAMATH_GPT_range_of_a_l1651_165171

noncomputable def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 3}

theorem range_of_a (a : ℝ) :
  ((setA a ∩ setB) = setA a) ∧ (∃ x, x ∈ (setA a ∩ setB)) →
  (a < -3 ∨ a > 3) ∧ (a < -1 ∨ a > 1) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1651_165171


namespace NUMINAMATH_GPT_jimmy_hostel_stay_days_l1651_165113

-- Definitions based on the conditions
def nightly_hostel_charge : ℕ := 15
def nightly_cabin_charge_per_person : ℕ := 15
def total_lodging_expense : ℕ := 75
def days_in_cabin : ℕ := 2

-- The proof statement
theorem jimmy_hostel_stay_days : 
    ∃ x : ℕ, (nightly_hostel_charge * x + nightly_cabin_charge_per_person * days_in_cabin = total_lodging_expense) ∧ x = 3 := by
    sorry

end NUMINAMATH_GPT_jimmy_hostel_stay_days_l1651_165113


namespace NUMINAMATH_GPT_compound_interest_l1651_165167

theorem compound_interest (SI : ℝ) (P : ℝ) (R : ℝ) (T : ℝ) (CI : ℝ) :
  SI = 50 →
  R = 5 →
  T = 2 →
  P = (SI * 100) / (R * T) →
  CI = P * (1 + R / 100)^T - P →
  CI = 51.25 :=
by
  intros
  exact sorry -- This placeholder represents the proof that would need to be filled in 

end NUMINAMATH_GPT_compound_interest_l1651_165167


namespace NUMINAMATH_GPT_min_value_l1651_165137

-- Given points A, B, and C and their specific coordinates
def A : (ℝ × ℝ) := (1, 3)
def B (a : ℝ) : (ℝ × ℝ) := (a, 1)
def C (b : ℝ) : (ℝ × ℝ) := (-b, 0)

-- Conditions
axiom a_pos (a : ℝ) : a > 0
axiom b_pos (b : ℝ) : b > 0
axiom collinear (a b : ℝ) : 3 * a + 2 * b = 1

-- The theorem to prove
theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hcollinear : 3 * a + 2 * b = 1) : 
  ∃ z, z = 11 + 6 * Real.sqrt 2 ∧ ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 1) -> (3 / x + 1 / y) ≥ z :=
by sorry -- Proof to be provided

end NUMINAMATH_GPT_min_value_l1651_165137


namespace NUMINAMATH_GPT_choir_members_max_l1651_165102

theorem choir_members_max (m y n : ℕ) (h_square : m = y^2 + 11) (h_rect : m = n * (n + 5)) : 
  m = 300 := 
sorry

end NUMINAMATH_GPT_choir_members_max_l1651_165102


namespace NUMINAMATH_GPT_deal_saves_customer_two_dollars_l1651_165121

-- Define the conditions of the problem
def movie_ticket_price : ℕ := 8
def popcorn_price : ℕ := movie_ticket_price - 3
def drink_price : ℕ := popcorn_price + 1
def candy_price : ℕ := drink_price / 2

def normal_total_price : ℕ := movie_ticket_price + popcorn_price + drink_price + candy_price
def deal_price : ℕ := 20

-- Prove the savings
theorem deal_saves_customer_two_dollars : normal_total_price - deal_price = 2 :=
by
  -- We will fill in the proof here
  sorry

end NUMINAMATH_GPT_deal_saves_customer_two_dollars_l1651_165121


namespace NUMINAMATH_GPT_meera_fraction_4kmh_l1651_165150

noncomputable def fraction_of_time_at_4kmh (total_time : ℝ) (x : ℝ) : ℝ :=
  x / total_time

theorem meera_fraction_4kmh (total_time x : ℝ) (h1 : x = total_time / 14) :
  fraction_of_time_at_4kmh total_time x = 1 / 14 :=
by
  sorry

end NUMINAMATH_GPT_meera_fraction_4kmh_l1651_165150


namespace NUMINAMATH_GPT_isosceles_triangle_l1651_165131

theorem isosceles_triangle (a c : ℝ) (A C : ℝ) (h : a * Real.sin A = c * Real.sin C) : a = c → Isosceles :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_l1651_165131


namespace NUMINAMATH_GPT_min_distinct_values_l1651_165194

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) 
  (h_mode : mode_count = 10) (h_total : total_count = 2018) 
  (h_distinct : ∀ k, k ≠ mode_count → k < 10) : 
  n ≥ 225 :=
by
  sorry

end NUMINAMATH_GPT_min_distinct_values_l1651_165194


namespace NUMINAMATH_GPT_sales_tax_reduction_difference_l1651_165168

def sales_tax_difference (original_rate new_rate market_price : ℝ) : ℝ :=
  (market_price * original_rate) - (market_price * new_rate)

theorem sales_tax_reduction_difference :
  sales_tax_difference 0.035 0.03333 10800 = 18.36 :=
by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end NUMINAMATH_GPT_sales_tax_reduction_difference_l1651_165168


namespace NUMINAMATH_GPT_value_of_f_prime_at_2_l1651_165154

theorem value_of_f_prime_at_2 :
  ∃ (f' : ℝ → ℝ), 
  (∀ (x : ℝ), f' x = 2 * x + 3 * f' 2 + 1 / x) →
  f' 2 = - (9 / 4) := 
by 
  sorry

end NUMINAMATH_GPT_value_of_f_prime_at_2_l1651_165154


namespace NUMINAMATH_GPT_max_three_m_plus_four_n_l1651_165146

theorem max_three_m_plus_four_n (m n : ℕ) 
  (h : m * (m + 1) + n ^ 2 = 1987) : 3 * m + 4 * n ≤ 221 :=
sorry

end NUMINAMATH_GPT_max_three_m_plus_four_n_l1651_165146


namespace NUMINAMATH_GPT_relationship_of_y_coordinates_l1651_165163

theorem relationship_of_y_coordinates (b y1 y2 y3 : ℝ):
  (y1 = 3 * -2.3 + b) → (y2 = 3 * -1.3 + b) → (y3 = 3 * 2.7 + b) → (y1 < y2 ∧ y2 < y3) := 
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_relationship_of_y_coordinates_l1651_165163


namespace NUMINAMATH_GPT_point_N_coordinates_l1651_165199

/--
Given:
- point M with coordinates (5, -6)
- vector a = (1, -2)
- the vector NM equals 3 times vector a
Prove:
- the coordinates of point N are (2, 0)
-/

theorem point_N_coordinates (x y : ℝ) :
  let M := (5, -6)
  let a := (1, -2)
  let NM := (5 - x, -6 - y)
  3 * a = NM → 
  (x = 2 ∧ y = 0) :=
by 
  intros
  sorry

end NUMINAMATH_GPT_point_N_coordinates_l1651_165199


namespace NUMINAMATH_GPT_sum_f_1_to_2017_l1651_165166

noncomputable def f (x : ℝ) : ℝ :=
  if x % 6 < -1 then -(x % 6 + 2) ^ 2 else x % 6

theorem sum_f_1_to_2017 : (List.sum (List.map f (List.range' 1 2017))) = 337 :=
  sorry

end NUMINAMATH_GPT_sum_f_1_to_2017_l1651_165166


namespace NUMINAMATH_GPT_andy_diana_weight_l1651_165118

theorem andy_diana_weight :
  ∀ (a b c d : ℝ),
  a + b = 300 →
  b + c = 280 →
  c + d = 310 →
  a + d = 330 := by
  intros a b c d h₁ h₂ h₃
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_andy_diana_weight_l1651_165118


namespace NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l1651_165110

theorem ratio_of_areas_of_concentric_circles (C1 C2 : ℝ) (h1 : (60 / 360) * C1 = (45 / 360) * C2) :
  (C1 / C2) ^ 2 = (9 / 16) := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l1651_165110


namespace NUMINAMATH_GPT_Gordons_heavier_bag_weight_l1651_165178

theorem Gordons_heavier_bag_weight :
  ∀ (G : ℝ), (5 * 2 = 3 + G) → G = 7 :=
by
  intro G h
  sorry

end NUMINAMATH_GPT_Gordons_heavier_bag_weight_l1651_165178


namespace NUMINAMATH_GPT_marcus_dropped_8_pies_l1651_165179

-- Step d): Rewrite as a Lean 4 statement
-- Define all conditions from the problem
def total_pies (pies_per_batch : ℕ) (batches : ℕ) : ℕ :=
  pies_per_batch * batches

def pies_dropped (total_pies : ℕ) (remaining_pies : ℕ) : ℕ :=
  total_pies - remaining_pies

-- Prove that Marcus dropped 8 pies
theorem marcus_dropped_8_pies : 
  total_pies 5 7 - 27 = 8 := by
  sorry

end NUMINAMATH_GPT_marcus_dropped_8_pies_l1651_165179


namespace NUMINAMATH_GPT_larger_number_is_70380_l1651_165120

theorem larger_number_is_70380 (A B : ℕ) 
    (hcf : Nat.gcd A B = 20) 
    (lcm : Nat.lcm A B = 20 * 9 * 17 * 23) :
    max A B = 70380 :=
  sorry

end NUMINAMATH_GPT_larger_number_is_70380_l1651_165120


namespace NUMINAMATH_GPT_range_of_m_l1651_165196

-- Define the set A and condition
def A (m : ℝ) : Set ℝ := { x : ℝ | x^2 - 2 * x + m = 0 }

-- The theorem stating the range of m
theorem range_of_m (m : ℝ) : (A m = ∅) ↔ m > 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1651_165196


namespace NUMINAMATH_GPT_binom_floor_divisible_l1651_165123

theorem binom_floor_divisible {p n : ℕ}
  (hp : Prime p) :
  (Nat.choose n p - n / p) % p = 0 := 
by
  sorry

end NUMINAMATH_GPT_binom_floor_divisible_l1651_165123


namespace NUMINAMATH_GPT_al_original_portion_l1651_165107

theorem al_original_portion {a b c d : ℕ} 
  (h1 : a + b + c + d = 2000)
  (h2 : a - 150 + 3 * b + 3 * c + d - 50 = 2500)
  (h3 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  a = 450 :=
sorry

end NUMINAMATH_GPT_al_original_portion_l1651_165107


namespace NUMINAMATH_GPT_probability_event_A_l1651_165138

def probability_of_defective : Real := 0.3
def probability_of_all_defective : Real := 0.027
def probability_of_event_A : Real := 0.973

theorem probability_event_A :
  1 - probability_of_all_defective = probability_of_event_A :=
by
  sorry

end NUMINAMATH_GPT_probability_event_A_l1651_165138


namespace NUMINAMATH_GPT_vector_addition_subtraction_identity_l1651_165108

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (BC AB AC : V)

theorem vector_addition_subtraction_identity : BC + AB - AC = 0 := 
by sorry

end NUMINAMATH_GPT_vector_addition_subtraction_identity_l1651_165108


namespace NUMINAMATH_GPT_correct_option_l1651_165182

theorem correct_option :
  (∀ a : ℝ, a ≠ 0 → (a ^ 0 = 1)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → (a^6 / a^3 = a^2)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → ((a^2)^3 = a^5)) ∧
  ¬(∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (a / (a + b)^2 + b / (a + b)^2 = a + b)) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_option_l1651_165182


namespace NUMINAMATH_GPT_find_polar_equations_and_distance_l1651_165115

noncomputable def polar_equation_C1 (rho theta : ℝ) : Prop :=
  rho^2 * Real.cos (2 * theta) = 1

noncomputable def polar_equation_C2 (rho theta : ℝ) : Prop :=
  rho = 2 * Real.cos theta

theorem find_polar_equations_and_distance :
  (∀ rho theta, polar_equation_C1 rho theta ↔ rho^2 * Real.cos (2 * theta) = 1) ∧
  (∀ rho theta, polar_equation_C2 rho theta ↔ rho = 2 * Real.cos theta) ∧
  let theta := Real.pi / 6
  let rho_A := Real.sqrt 2
  let rho_B := Real.sqrt 3
  (|rho_A - rho_B| = |Real.sqrt 3 - Real.sqrt 2|) :=
  by sorry

end NUMINAMATH_GPT_find_polar_equations_and_distance_l1651_165115


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_m_l1651_165192

open Real

noncomputable def f (x : ℝ) := abs (x + 1) - abs (x - 2)

theorem problem1_solution_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry

theorem problem2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5 / 4 :=
sorry

end NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_m_l1651_165192


namespace NUMINAMATH_GPT_measure_of_angle_A_possibilities_l1651_165198

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end NUMINAMATH_GPT_measure_of_angle_A_possibilities_l1651_165198


namespace NUMINAMATH_GPT_minimum_value_expr_l1651_165148

noncomputable def expr (x : ℝ) : ℝ := (x^2 + 11) / Real.sqrt (x^2 + 5)

theorem minimum_value_expr : ∃ x : ℝ, expr x = 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expr_l1651_165148


namespace NUMINAMATH_GPT_solve_equation_l1651_165125

noncomputable def smallest_solution : ℝ :=
(15 - Real.sqrt 549) / 6

theorem solve_equation :
  ∃ x : ℝ, 
    (3 * x / (x - 3) + (3 * x^2 - 27) / x = 18) ∧
    x = smallest_solution :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1651_165125


namespace NUMINAMATH_GPT_possible_perimeters_l1651_165104

theorem possible_perimeters (a b c: ℝ) (h1: a = 1) (h2: b = 1) 
  (h3: c = 1) (h: ∀ x y z: ℝ, x = y ∧ y = z):
  ∃ x y: ℝ, (x = 8/3 ∧ y = 5/2) := 
  by
    sorry

end NUMINAMATH_GPT_possible_perimeters_l1651_165104


namespace NUMINAMATH_GPT_animals_in_field_l1651_165117

def dog := 1
def cats := 4
def rabbits_per_cat := 2
def hares_per_rabbit := 3

def rabbits := cats * rabbits_per_cat
def hares := rabbits * hares_per_rabbit

def total_animals := dog + cats + rabbits + hares

theorem animals_in_field : total_animals = 37 := by
  sorry

end NUMINAMATH_GPT_animals_in_field_l1651_165117


namespace NUMINAMATH_GPT_coord_of_point_M_in_third_quadrant_l1651_165111

noncomputable def point_coordinates (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0 ∧ abs y = 1 ∧ abs x = 2

theorem coord_of_point_M_in_third_quadrant : 
  ∃ (x y : ℝ), point_coordinates x y ∧ (x, y) = (-2, -1) := 
by {
  sorry
}

end NUMINAMATH_GPT_coord_of_point_M_in_third_quadrant_l1651_165111


namespace NUMINAMATH_GPT_operation_is_double_l1651_165165

theorem operation_is_double (x : ℝ) (operation : ℝ → ℝ) (h1: x^2 = 25) (h2: operation x = x / 5 + 9) : operation x = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_operation_is_double_l1651_165165


namespace NUMINAMATH_GPT_range_of_m_l1651_165129

theorem range_of_m (x y m : ℝ) 
  (h1: 3 * x + y = 1 + 3 * m) 
  (h2: x + 3 * y = 1 - m) 
  (h3: x + y > 0) : 
  m > -1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1651_165129


namespace NUMINAMATH_GPT_prove_system_of_inequalities_l1651_165169

theorem prove_system_of_inequalities : 
  { x : ℝ | x / (x - 2) ≥ 0 ∧ 2 * x + 1 ≥ 0 } = Set.Icc (-(1:ℝ)/2) 0 ∪ Set.Ioi 2 := 
by
  sorry

end NUMINAMATH_GPT_prove_system_of_inequalities_l1651_165169


namespace NUMINAMATH_GPT_max_value_product_focal_distances_l1651_165134

theorem max_value_product_focal_distances {a b c : ℝ} 
  (h1 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h2 : ∀ x : ℝ, -a ≤ x ∧ x ≤ a) 
  (e : ℝ) :
  (∀ x : ℝ, (a - e * x) * (a + e * x) ≤ a^2) :=
sorry

end NUMINAMATH_GPT_max_value_product_focal_distances_l1651_165134


namespace NUMINAMATH_GPT_dave_more_than_jerry_games_l1651_165197

variable (K D J : ℕ)  -- Declaring the variables for Ken, Dave, and Jerry respectively

-- Defining the conditions
def ken_more_games := K = D + 5
def dave_more_than_jerry := D > 7
def jerry_games := J = 7
def total_games := K + D + 7 = 32

-- Defining the proof problem
theorem dave_more_than_jerry_games (hK : ken_more_games K D) (hD : dave_more_than_jerry D) (hJ : jerry_games J) (hT : total_games K D) : D - 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_dave_more_than_jerry_games_l1651_165197


namespace NUMINAMATH_GPT_analytical_expression_of_f_l1651_165156

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b

theorem analytical_expression_of_f (a b : ℝ) (h_a : a > 0)
  (h_max : (∃ x_max : ℝ, f x_max a b = 5 ∧ (∀ x : ℝ, f x_max a b ≥ f x a b)))
  (h_min : (∃ x_min : ℝ, f x_min a b = 1 ∧ (∀ x : ℝ, f x_min a b ≤ f x a b))) :
  f x 3 1 = x^3 + 3 * x^2 + 1 := 
sorry

end NUMINAMATH_GPT_analytical_expression_of_f_l1651_165156


namespace NUMINAMATH_GPT_find_k_in_geometric_sequence_l1651_165103

theorem find_k_in_geometric_sequence (a : ℕ → ℕ) (k : ℕ)
  (h1 : ∀ n, a n = a 2 * 3^(n-2))
  (h2 : a 2 = 3)
  (h3 : a 3 = 9)
  (h4 : a k = 243) :
  k = 6 :=
sorry

end NUMINAMATH_GPT_find_k_in_geometric_sequence_l1651_165103


namespace NUMINAMATH_GPT_find_m_l1651_165128

def point (α : Type) := (α × α)

def collinear {α : Type} [LinearOrderedField α] 
  (p1 p2 p3 : point α) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

theorem find_m {m : ℚ} 
  (h : collinear (4, 10) (-3, m) (-12, 5)) : 
  m = 125 / 16 :=
by sorry

end NUMINAMATH_GPT_find_m_l1651_165128


namespace NUMINAMATH_GPT_intersection_is_14_l1651_165176

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem intersection_is_14 : A ∩ B = {1, 4} := 
by sorry

end NUMINAMATH_GPT_intersection_is_14_l1651_165176


namespace NUMINAMATH_GPT_part_1_part_2_l1651_165164

noncomputable def f (x a : ℝ) : ℝ := x^2 * |x - a|

theorem part_1 (a : ℝ) (h : a = 2) : {x : ℝ | f x a = x} = {0, 1, 1 + Real.sqrt 2} :=
by 
  sorry

theorem part_2 (a : ℝ) : 
  ∃ m : ℝ, m = 
    if a ≤ 1 then 1 - a 
    else if 1 < a ∧ a ≤ 2 then 0 
    else if 2 < a ∧ a ≤ (7 / 3 : ℝ) then 4 * (a - 2) 
    else a - 1 :=
by 
  sorry

end NUMINAMATH_GPT_part_1_part_2_l1651_165164


namespace NUMINAMATH_GPT_min_value_M_l1651_165160

theorem min_value_M (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : ∃ a b, M = 3 * a^2 - a * b^2 - 2 * b - 4 ∧ M = 2 := sorry

end NUMINAMATH_GPT_min_value_M_l1651_165160


namespace NUMINAMATH_GPT_find_principal_l1651_165173

theorem find_principal
  (P : ℝ)
  (R : ℝ := 4)
  (T : ℝ := 5)
  (SI : ℝ := (P * R * T) / 100) 
  (h : SI = P - 2400) : 
  P = 3000 := 
sorry

end NUMINAMATH_GPT_find_principal_l1651_165173


namespace NUMINAMATH_GPT_woman_traveled_by_bus_l1651_165139

noncomputable def travel_by_bus : ℕ :=
  let total_distance := 1800
  let distance_by_plane := total_distance / 4
  let distance_by_train := total_distance / 6
  let distance_by_taxi := total_distance / 8
  let remaining_distance := total_distance - (distance_by_plane + distance_by_train + distance_by_taxi)
  let distance_by_rental := remaining_distance * 2 / 3
  distance_by_rental / 2

theorem woman_traveled_by_bus :
  travel_by_bus = 275 :=
by 
  sorry

end NUMINAMATH_GPT_woman_traveled_by_bus_l1651_165139


namespace NUMINAMATH_GPT_trihedral_angle_sum_gt_180_l1651_165177

theorem trihedral_angle_sum_gt_180
    (a' b' c' α β γ : ℝ)
    (Sabc : Prop)
    (h1 : b' = π - α)
    (h2 : c' = π - β)
    (h3 : a' = π - γ)
    (triangle_inequality : a' + b' + c' < 2 * π) :
    α + β + γ > π :=
by
  sorry

end NUMINAMATH_GPT_trihedral_angle_sum_gt_180_l1651_165177


namespace NUMINAMATH_GPT_quadratic_one_real_root_l1651_165157

theorem quadratic_one_real_root (m : ℝ) : 
  (∃ x : ℝ, (x^2 - 6*m*x + 2*m = 0) ∧ 
    (∀ y : ℝ, (y^2 - 6*m*y + 2*m = 0) → y = x)) → 
  m = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_one_real_root_l1651_165157


namespace NUMINAMATH_GPT_solve_for_x_l1651_165181

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = (9 / (x / 3))^2) : x = 23.43 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1651_165181


namespace NUMINAMATH_GPT_smallest_N_l1651_165175

theorem smallest_N (p q r s t u : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (ht : 0 < t) (hu : 0 < u)
  (h_sum : p + q + r + s + t + u = 2023) :
  ∃ N : ℕ, N = max (max (max (max (p + q) (q + r)) (r + s)) (s + t)) (t + u) ∧ N = 810 :=
sorry

end NUMINAMATH_GPT_smallest_N_l1651_165175


namespace NUMINAMATH_GPT_fraction_solution_l1651_165136

theorem fraction_solution (x : ℝ) (h : 4 - 9 / x + 4 / x^2 = 0) : 3 / x = 12 ∨ 3 / x = 3 / 4 :=
by
  -- Proof to be written here
  sorry

end NUMINAMATH_GPT_fraction_solution_l1651_165136


namespace NUMINAMATH_GPT_expand_and_simplify_l1651_165153

theorem expand_and_simplify (y : ℚ) (h : y ≠ 0) :
  (3/4 * (8/y - 6*y^2 + 3*y)) = (6/y - 9*y^2/2 + 9*y/4) :=
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l1651_165153


namespace NUMINAMATH_GPT_work_done_at_4_pm_l1651_165116

noncomputable def workCompletionTime (aHours : ℝ) (bHours : ℝ) (startTime : ℝ) : ℝ :=
  let aRate := 1 / aHours
  let bRate := 1 / bHours
  let cycleWork := aRate + bRate
  let cyclesNeeded := (1 : ℝ) / cycleWork
  startTime + 2 * cyclesNeeded

theorem work_done_at_4_pm :
  workCompletionTime 8 12 6 = 16 :=  -- 16 in 24-hour format is 4 pm
by 
  sorry

end NUMINAMATH_GPT_work_done_at_4_pm_l1651_165116


namespace NUMINAMATH_GPT_waiter_earnings_l1651_165144

def num_customers : ℕ := 9
def num_no_tip : ℕ := 5
def tip_per_customer : ℕ := 8
def num_tipping_customers := num_customers - num_no_tip

theorem waiter_earnings : num_tipping_customers * tip_per_customer = 32 := by
  sorry

end NUMINAMATH_GPT_waiter_earnings_l1651_165144


namespace NUMINAMATH_GPT_cos_alpha_minus_half_beta_l1651_165147

theorem cos_alpha_minus_half_beta
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α - β / 2) = Real.sqrt 6 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_half_beta_l1651_165147


namespace NUMINAMATH_GPT_principal_amount_l1651_165119

theorem principal_amount
  (P : ℝ)
  (r : ℝ := 0.05)
  (t : ℝ := 2)
  (H : P * (1 + r)^t - P - P * r * t = 17) :
  P = 6800 :=
by sorry

end NUMINAMATH_GPT_principal_amount_l1651_165119


namespace NUMINAMATH_GPT_eq_of_div_eq_div_l1651_165185

theorem eq_of_div_eq_div {a b c : ℝ} (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end NUMINAMATH_GPT_eq_of_div_eq_div_l1651_165185


namespace NUMINAMATH_GPT_mass_percentage_Br_HBrO3_l1651_165127

theorem mass_percentage_Br_HBrO3 (molar_mass_H : ℝ) (molar_mass_Br : ℝ) (molar_mass_O : ℝ)
  (molar_mass_HBrO3 : ℝ) (mass_percentage_H : ℝ) (mass_percentage_Br : ℝ) :
  molar_mass_H = 1.01 →
  molar_mass_Br = 79.90 →
  molar_mass_O = 16.00 →
  molar_mass_HBrO3 = molar_mass_H + molar_mass_Br + 3 * molar_mass_O →
  mass_percentage_H = 0.78 →
  mass_percentage_Br = (molar_mass_Br / molar_mass_HBrO3) * 100 → 
  mass_percentage_Br = 61.98 :=
sorry

end NUMINAMATH_GPT_mass_percentage_Br_HBrO3_l1651_165127


namespace NUMINAMATH_GPT_find_number_l1651_165145

theorem find_number (x : ℕ) (h : 3 * (2 * x + 8) = 84) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1651_165145


namespace NUMINAMATH_GPT_suitable_altitude_range_l1651_165126

theorem suitable_altitude_range :
  ∀ (temperature_at_base : ℝ) (temp_decrease_per_100m : ℝ) (suitable_temp_low : ℝ) (suitable_temp_high : ℝ) (altitude_at_base : ℝ),
  (22 = temperature_at_base) →
  (0.5 = temp_decrease_per_100m) →
  (18 = suitable_temp_low) →
  (20 = suitable_temp_high) →
  (0 = altitude_at_base) →
  400 ≤ ((temperature_at_base - suitable_temp_high) / temp_decrease_per_100m * 100) ∧ ((temperature_at_base - suitable_temp_low) / temp_decrease_per_100m * 100) ≤ 800 :=
by
  intros temperature_at_base temp_decrease_per_100m suitable_temp_low suitable_temp_high altitude_at_base
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_suitable_altitude_range_l1651_165126


namespace NUMINAMATH_GPT_A_worked_days_l1651_165149

theorem A_worked_days 
  (W : ℝ)                              -- Total work in arbitrary units
  (A_work_days : ℕ)                    -- Days A can complete the work 
  (B_work_days_remaining : ℕ)          -- Days B takes to complete remaining work
  (B_work_days : ℕ)                    -- Days B can complete the work alone
  (hA : A_work_days = 15)              -- A can do the work in 15 days
  (hB : B_work_days_remaining = 12)    -- B completes the remaining work in 12 days
  (hB_alone : B_work_days = 18)        -- B alone can do the work in 18 days
  :
  ∃ (x : ℕ), x = 5                     -- A worked for 5 days before leaving the job
  := 
  sorry                                 -- Proof not provided

end NUMINAMATH_GPT_A_worked_days_l1651_165149


namespace NUMINAMATH_GPT_distinct_arrangements_l1651_165100

-- Definitions based on the conditions
def boys : ℕ := 4
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def arrangements : ℕ := Nat.factorial boys * Nat.factorial (total_people - 2) * Nat.factorial 6

-- Main statement: Verify the number of distinct arrangements
theorem distinct_arrangements : arrangements = 8640 := by
  -- We will replace this proof with our Lean steps (which is currently omitted)
  sorry

end NUMINAMATH_GPT_distinct_arrangements_l1651_165100


namespace NUMINAMATH_GPT_direct_proportional_function_point_l1651_165133

theorem direct_proportional_function_point 
    (h₁ : ∃ k : ℝ, ∀ x : ℝ, (2, -3).snd = k * (2, -3).fst)
    (h₂ : ∃ k : ℝ, ∀ x : ℝ, (4, -6).snd = k * (4, -6).fst)
    : (∃ k : ℝ, k = -(3 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_direct_proportional_function_point_l1651_165133


namespace NUMINAMATH_GPT_locus_points_eq_distance_l1651_165135

def locus_is_parabola (x y : ℝ) : Prop :=
  (y - 1) ^ 2 = 16 * (x - 2)

theorem locus_points_eq_distance (x y : ℝ) :
  locus_is_parabola x y ↔ (x, y) = (4, 1) ∨
    dist (x, y) (4, 1) = dist (x, y) (0, y) :=
by
  sorry

end NUMINAMATH_GPT_locus_points_eq_distance_l1651_165135


namespace NUMINAMATH_GPT_part1_part2_l1651_165162

theorem part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) : 
  (1 / a) + (1 / (b + 1)) ≥ 4 / 5 := 
by 
  sorry

theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 8) : 
  a + b ≥ 4 := 
by 
  sorry

end NUMINAMATH_GPT_part1_part2_l1651_165162
