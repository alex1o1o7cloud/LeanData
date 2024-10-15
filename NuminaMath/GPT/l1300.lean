import Mathlib

namespace NUMINAMATH_GPT_domain_of_sqrt_function_l1300_130026

theorem domain_of_sqrt_function (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 ≥ 0) ↔ 0 ≤ m ∧ m ≤ 4 := sorry

end NUMINAMATH_GPT_domain_of_sqrt_function_l1300_130026


namespace NUMINAMATH_GPT_part_a_part_b_l1300_130075

-- Part (a)
theorem part_a (a b c : ℚ) (z : ℚ) (h : a * z^2 + b * z + c = 0) (n : ℕ) (hn : n > 0) :
  ∃ f : ℚ → ℚ, z = f (z^n) :=
sorry

-- Part (b)
theorem part_b (x : ℚ) (h : x ≠ 0) :
  x = (x^3 + (x + 1/x)) / ((x + 1/x)^2 - 1) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1300_130075


namespace NUMINAMATH_GPT_tangent_slope_correct_l1300_130013

noncomputable def slope_of_directrix (focus: ℝ × ℝ) (p1: ℝ × ℝ) (p2: ℝ × ℝ) : ℝ :=
  let c1 := p1
  let c2 := p2
  let radius1 := Real.sqrt ((c1.1 + 1)^2 + (c1.2 + 1)^2)
  let radius2 := Real.sqrt ((c2.1 - 2)^2 + (c2.2 - 2)^2)
  let dist := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let slope := (focus.2 - p1.2) / (focus.1 - p1.1)
  let tangent_slope := (9 : ℝ) / (7 : ℝ) + (4 * Real.sqrt 2) / 7
  tangent_slope

theorem tangent_slope_correct :
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 + 4 * Real.sqrt 2) / 7) ∨
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 - 4 * Real.sqrt 2) / 7) :=
by
  -- Proof omitted here
  sorry

end NUMINAMATH_GPT_tangent_slope_correct_l1300_130013


namespace NUMINAMATH_GPT_A_pow_101_l1300_130047

def A : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 1],
  ![1, 0, 0],
  ![0, 1, 0]
]

theorem A_pow_101 :
  A ^ 101 = ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] := by
  sorry

end NUMINAMATH_GPT_A_pow_101_l1300_130047


namespace NUMINAMATH_GPT_weight_of_new_person_is_correct_l1300_130015

noncomputable def weight_new_person (increase_per_person : ℝ) (old_weight : ℝ) (group_size : ℝ) : ℝ :=
  old_weight + group_size * increase_per_person

theorem weight_of_new_person_is_correct :
  weight_new_person 7.2 65 10 = 137 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_new_person_is_correct_l1300_130015


namespace NUMINAMATH_GPT_alice_current_age_l1300_130066

theorem alice_current_age (a b : ℕ) 
  (h1 : a + 8 = 2 * (b + 8)) 
  (h2 : (a - 10) + (b - 10) = 21) : 
  a = 30 := 
by 
  sorry

end NUMINAMATH_GPT_alice_current_age_l1300_130066


namespace NUMINAMATH_GPT_boxes_in_case_number_of_boxes_in_case_l1300_130087

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

end NUMINAMATH_GPT_boxes_in_case_number_of_boxes_in_case_l1300_130087


namespace NUMINAMATH_GPT_class1_qualified_l1300_130051

variables (Tardiness : ℕ → ℕ) -- Tardiness function mapping days to number of tardy students

def classQualified (mean variance median mode : ℕ) : Prop :=
  (mean = 2 ∧ variance = 2) ∨
  (mean = 3 ∧ median = 3) ∨
  (mean = 2 ∧ variance > 0) ∨
  (median = 2 ∧ mode = 2)

def eligible (Tardiness : ℕ → ℕ) : Prop :=
  ∀ i, i < 5 → Tardiness i ≤ 5

theorem class1_qualified : 
  (∀ Tardiness, (∃ mean variance median mode,
    classQualified mean variance median mode 
    ∧ mean = 2 ∧ variance = 2 
    ∧ eligible Tardiness)) → 
  (∀ Tardiness, eligible Tardiness) :=
by
  sorry

end NUMINAMATH_GPT_class1_qualified_l1300_130051


namespace NUMINAMATH_GPT_deficit_calculation_l1300_130027

theorem deficit_calculation
    (L W : ℝ)  -- Length and Width
    (dW : ℝ)  -- Deficit in width
    (h1 : (1.08 * L) * (W - dW) = 1.026 * (L * W))  -- Condition on the calculated area
    : dW / W = 0.05 := 
by
    sorry

end NUMINAMATH_GPT_deficit_calculation_l1300_130027


namespace NUMINAMATH_GPT_chantel_bracelets_at_end_l1300_130078

-- Definitions based on conditions
def bracelets_day1 := 4
def days1 := 7
def given_away1 := 8

def bracelets_day2 := 5
def days2 := 10
def given_away2 := 12

-- Computation based on conditions
def total_bracelets := days1 * bracelets_day1 - given_away1 + days2 * bracelets_day2 - given_away2

-- The proof statement
theorem chantel_bracelets_at_end : total_bracelets = 58 := by
  sorry

end NUMINAMATH_GPT_chantel_bracelets_at_end_l1300_130078


namespace NUMINAMATH_GPT_at_least_one_not_less_than_l1300_130045

variables {A B C D a b c : ℝ}

theorem at_least_one_not_less_than :
  (a = A * C) →
  (b = A * D + B * C) →
  (c = B * D) →
  (a + b + c = (A + B) * (C + D)) →
  a ≥ (4 * (A + B) * (C + D) / 9) ∨ b ≥ (4 * (A + B) * (C + D) / 9) ∨ c ≥ (4 * (A + B) * (C + D) / 9) :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_l1300_130045


namespace NUMINAMATH_GPT_probability_greater_difficulty_probability_same_difficulty_l1300_130073

/-- A datatype representing the difficulty levels of questions. -/
inductive Difficulty
| easy : Difficulty
| medium : Difficulty
| difficult : Difficulty

/-- A datatype representing the four questions with their difficulties. -/
inductive Question
| A1 : Question
| A2 : Question
| B : Question
| C : Question

/-- The function to get the difficulty of a question. -/
def difficulty (q : Question) : Difficulty :=
  match q with
  | Question.A1 => Difficulty.easy
  | Question.A2 => Difficulty.easy
  | Question.B  => Difficulty.medium
  | Question.C  => Difficulty.difficult

/-- The set of all possible pairings of questions selected by two students A and B. -/
def all_pairs : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A1, Question.B), (Question.A1, Question.C),
    (Question.A2, Question.A1), (Question.A2, Question.A2), (Question.A2, Question.B), (Question.A2, Question.C),
    (Question.B, Question.A1), (Question.B, Question.A2), (Question.B, Question.B), (Question.B, Question.C),
    (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B), (Question.C, Question.C) ]

/-- The event that the difficulty of the question selected by student A is greater than that selected by student B. -/
def event_N : List (Question × Question) :=
  [ (Question.B, Question.A1), (Question.B, Question.A2), (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B) ]

/-- The event that the difficulties of the questions selected by both students are the same. -/
def event_M : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A2, Question.A1), (Question.A2, Question.A2), 
    (Question.B, Question.B), (Question.C, Question.C) ]

/-- The probabilities of the events. -/
noncomputable def probability_event_N : ℚ := (event_N.length : ℚ) / (all_pairs.length : ℚ)
noncomputable def probability_event_M : ℚ := (event_M.length : ℚ) / (all_pairs.length : ℚ)

/-- The theorem statements -/
theorem probability_greater_difficulty : probability_event_N = 5 / 16 := sorry
theorem probability_same_difficulty : probability_event_M = 3 / 8 := sorry

end NUMINAMATH_GPT_probability_greater_difficulty_probability_same_difficulty_l1300_130073


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1300_130034

theorem arithmetic_seq_sum {a : ℕ → ℝ} (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1300_130034


namespace NUMINAMATH_GPT_find_d_l1300_130070

noncomputable def quadratic_roots (d : ℝ) : Prop :=
∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2

theorem find_d : ∃ d : ℝ, d = 9.8 ∧ quadratic_roots d :=
sorry

end NUMINAMATH_GPT_find_d_l1300_130070


namespace NUMINAMATH_GPT_sum_even_integers_correct_l1300_130053

variable (S1 S2 : ℕ)

-- Definition: The sum of the first 50 positive even integers
def sum_first_50_even_integers : ℕ := 2550

-- Definition: The sum of even integers from 102 to 200 inclusive
def sum_even_integers_from_102_to_200 : ℕ := 7550

-- Condition: The sum of the first 50 positive even integers is 2550
axiom sum_first_50_even_integers_given : S1 = sum_first_50_even_integers

-- Problem statement: Prove that the sum of even integers from 102 to 200 inclusive is 7550
theorem sum_even_integers_correct :
  S1 = sum_first_50_even_integers →
  S2 = sum_even_integers_from_102_to_200 →
  S2 = 7550 :=
by
  intros h1 h2
  rw [h2]
  sorry

end NUMINAMATH_GPT_sum_even_integers_correct_l1300_130053


namespace NUMINAMATH_GPT_newly_grown_uneaten_potatoes_l1300_130001

variable (u : ℕ)

def initially_planted : ℕ := 8
def total_now : ℕ := 11

theorem newly_grown_uneaten_potatoes : u = total_now - initially_planted := by
  sorry

end NUMINAMATH_GPT_newly_grown_uneaten_potatoes_l1300_130001


namespace NUMINAMATH_GPT_boat_downstream_distance_l1300_130090

-- Given conditions
def speed_boat_still_water : ℕ := 25
def speed_stream : ℕ := 5
def travel_time_downstream : ℕ := 3

-- Proof statement: The distance travelled downstream is 90 km
theorem boat_downstream_distance :
  speed_boat_still_water + speed_stream * travel_time_downstream = 90 :=
by
  -- omitting the actual proof steps
  sorry

end NUMINAMATH_GPT_boat_downstream_distance_l1300_130090


namespace NUMINAMATH_GPT_range_of_z_l1300_130037

theorem range_of_z (x y : ℝ) 
  (h1 : x + 2 ≥ y) 
  (h2 : x + 2 * y ≥ 4) 
  (h3 : y ≤ 5 - 2 * x) : 
  ∃ (z_min z_max : ℝ), 
    (z_min = 1) ∧ 
    (z_max = 2) ∧ 
    (∀ z, z = (2 * x + y - 1) / (x + 1) → z_min ≤ z ∧ z ≤ z_max) :=
by
  sorry

end NUMINAMATH_GPT_range_of_z_l1300_130037


namespace NUMINAMATH_GPT_value_of_expression_l1300_130003

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 5 = 23 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_value_of_expression_l1300_130003


namespace NUMINAMATH_GPT_least_number_divisible_by_11_and_remainder_2_l1300_130020

theorem least_number_divisible_by_11_and_remainder_2 :
  ∃ n, (∀ k : ℕ, 3 ≤ k ∧ k ≤ 7 → n % k = 2) ∧ n % 11 = 0 ∧ n = 1262 :=
by
  sorry

end NUMINAMATH_GPT_least_number_divisible_by_11_and_remainder_2_l1300_130020


namespace NUMINAMATH_GPT_sum_of_ages_eq_19_l1300_130063

theorem sum_of_ages_eq_19 :
  ∃ (a b s : ℕ), (3 * a + 5 + b = s) ∧ (6 * s^2 = 2 * a^2 + 10 * b^2) ∧ (Nat.gcd a (Nat.gcd b s) = 1 ∧ a + b + s = 19) :=
sorry

end NUMINAMATH_GPT_sum_of_ages_eq_19_l1300_130063


namespace NUMINAMATH_GPT_value_of_a2_l1300_130072

theorem value_of_a2 (a : ℕ → ℤ) (h1 : ∀ n : ℕ, a (n + 1) = a n + 2)
  (h2 : ∃ r : ℤ, a 3 = r * a 1 ∧ a 4 = r * a 3) :
  a 2 = -6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a2_l1300_130072


namespace NUMINAMATH_GPT_fraction_product_l1300_130043

theorem fraction_product :
  (8 / 4) * (10 / 5) * (21 / 14) * (16 / 8) * (45 / 15) * (30 / 10) * (49 / 35) * (32 / 16) = 302.4 := by
  sorry

end NUMINAMATH_GPT_fraction_product_l1300_130043


namespace NUMINAMATH_GPT_regular_polygon_sides_l1300_130039

-- Conditions
def central_angle (θ : ℝ) := θ = 30
def sum_of_central_angles (sumθ : ℝ) := sumθ = 360

-- The proof problem
theorem regular_polygon_sides (θ sumθ : ℝ) (h₁ : central_angle θ) (h₂ : sum_of_central_angles sumθ) :
  sumθ / θ = 12 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1300_130039


namespace NUMINAMATH_GPT_jen_age_difference_l1300_130042

-- Definitions as conditions given in the problem
def son_present_age := 16
def jen_present_age := 41

-- The statement to be proved
theorem jen_age_difference :
  3 * son_present_age - jen_present_age = 7 :=
by
  sorry

end NUMINAMATH_GPT_jen_age_difference_l1300_130042


namespace NUMINAMATH_GPT_seq_50th_term_eq_327_l1300_130057

theorem seq_50th_term_eq_327 : 
  let n := 50
  let binary_representation : List Nat := [1, 1, 0, 0, 1, 0] -- 50 in binary
  let powers_of_3 := [5, 4, 1] -- Positions of 1s in the binary representation 
  let term := List.sum (powers_of_3.map (λ k => 3^k))
  term = 327 := by
  sorry

end NUMINAMATH_GPT_seq_50th_term_eq_327_l1300_130057


namespace NUMINAMATH_GPT_cube_tangent_ratio_l1300_130092

theorem cube_tangent_ratio 
  (edge_length : ℝ) 
  (midpoint K : ℝ) 
  (tangent E : ℝ) 
  (intersection F : ℝ) 
  (radius R : ℝ)
  (h1 : edge_length = 2)
  (h2 : radius = 1)
  (h3 : K = midpoint)
  (h4 : ∃ E F, tangent = E ∧ intersection = F) :
  (K - E) / (F - E) = 4 / 5 :=
sorry

end NUMINAMATH_GPT_cube_tangent_ratio_l1300_130092


namespace NUMINAMATH_GPT_find_k_for_perfect_square_l1300_130080

theorem find_k_for_perfect_square :
  ∃ k : ℤ, (k = 12 ∨ k = -12) ∧ (∀ n : ℤ, ∃ a b : ℤ, 4 * n^2 + k * n + 9 = (a * n + b)^2) :=
sorry

end NUMINAMATH_GPT_find_k_for_perfect_square_l1300_130080


namespace NUMINAMATH_GPT_age_proof_l1300_130028

theorem age_proof (A B C D k m : ℕ)
  (h1 : A + B + C + D = 76)
  (h2 : A - 3 = k)
  (h3 : B - 3 = 2*k)
  (h4 : C - 3 = 3*k)
  (h5 : A - 5 = 3*m)
  (h6 : D - 5 = 4*m)
  (h7 : B - 5 = 5*m) :
  A = 11 := 
sorry

end NUMINAMATH_GPT_age_proof_l1300_130028


namespace NUMINAMATH_GPT_triangle_inequality_l1300_130082

variables {l_a l_b l_c m_a m_b m_c h_n m_n h_h_n m_m_p : ℝ}

-- Assuming some basic properties for the variables involved (all are positive in their respective triangle context)
axiom pos_l_a : 0 < l_a
axiom pos_l_b : 0 < l_b
axiom pos_l_c : 0 < l_c
axiom pos_m_a : 0 < m_a
axiom pos_m_b : 0 < m_b
axiom pos_m_c : 0 < m_c
axiom pos_h_n : 0 < h_n
axiom pos_m_n : 0 < m_n
axiom pos_h_h_n : 0 < h_h_n
axiom pos_m_m_p : 0 < m_m_p

theorem triangle_inequality :
  (h_n / m_n) + (h_n / h_h_n) + (l_c / m_m_p) > 1 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1300_130082


namespace NUMINAMATH_GPT_sports_club_membership_l1300_130088

theorem sports_club_membership :
  (17 + 21 - 10 + 2 = 30) :=
by
  sorry

end NUMINAMATH_GPT_sports_club_membership_l1300_130088


namespace NUMINAMATH_GPT_cost_of_renting_per_month_l1300_130032

namespace RentCarProblem

def cost_new_car_per_month : ℕ := 30
def months_per_year : ℕ := 12
def yearly_difference : ℕ := 120

theorem cost_of_renting_per_month (R : ℕ) :
  (cost_new_car_per_month * months_per_year + yearly_difference) / months_per_year = R → 
  R = 40 :=
by
  sorry

end RentCarProblem

end NUMINAMATH_GPT_cost_of_renting_per_month_l1300_130032


namespace NUMINAMATH_GPT_cory_chairs_l1300_130098

theorem cory_chairs (total_cost table_cost chair_cost C : ℕ) (h1 : total_cost = 135) (h2 : table_cost = 55) (h3 : chair_cost = 20) (h4 : total_cost = table_cost + chair_cost * C) : C = 4 := 
by 
  sorry

end NUMINAMATH_GPT_cory_chairs_l1300_130098


namespace NUMINAMATH_GPT_hernandez_state_tax_l1300_130030

theorem hernandez_state_tax 
    (res_months : ℕ) (total_months : ℕ) 
    (taxable_income : ℝ) (tax_rate : ℝ) 
    (prorated_income : ℝ) (state_tax : ℝ) 
    (h1 : res_months = 9) 
    (h2 : total_months = 12) 
    (h3 : taxable_income = 42500) 
    (h4 : tax_rate = 0.04) 
    (h5 : prorated_income = taxable_income * (res_months / total_months)) 
    (h6 : state_tax = prorated_income * tax_rate) : 
    state_tax = 1275 := 
by 
  -- this is where the proof would go
  sorry

end NUMINAMATH_GPT_hernandez_state_tax_l1300_130030


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1300_130095

-- Definitions
def represents_ellipse (m n : ℝ) (x y : ℝ) : Prop := 
  (x^2 / m + y^2 / n = 1)

-- Main theorem statement
theorem necessary_but_not_sufficient_condition 
    (m n x y : ℝ) (h_mn_pos : m * n > 0) :
    (represents_ellipse m n x y) → 
    (m ≠ n ∧ m > 0 ∧ n > 0 ∧ represents_ellipse m n x y) → 
    (m * n > 0) ∧ ¬(
    ∀ m n : ℝ, (m ≠ n ∧ m > 0 ∧ n > 0) →
    represents_ellipse m n x y
    ) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1300_130095


namespace NUMINAMATH_GPT_consecutive_even_integers_sum_l1300_130067

theorem consecutive_even_integers_sum (n : ℕ) (h : n % 2 = 0) (h_pro : n * (n + 2) * (n + 4) = 3360) :
  n + (n + 2) + (n + 4) = 48 :=
by sorry

end NUMINAMATH_GPT_consecutive_even_integers_sum_l1300_130067


namespace NUMINAMATH_GPT_sequence_and_sum_l1300_130068

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

theorem sequence_and_sum
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_first_n_terms a S)
  (cond : a 2 + a 8 = 15 - a 5) :
  S 9 = 45 :=
sorry

end NUMINAMATH_GPT_sequence_and_sum_l1300_130068


namespace NUMINAMATH_GPT_largest_negative_root_l1300_130064

theorem largest_negative_root : 
  ∃ x : ℝ, (∃ k : ℤ, x = -1/2 + 2 * ↑k) ∧ 
  ∀ y : ℝ, (∃ k : ℤ, (y = -1/2 + 2 * ↑k ∨ y = 1/6 + 2 * ↑k ∨ y = 5/6 + 2 * ↑k)) → y < 0 → y ≤ x :=
sorry

end NUMINAMATH_GPT_largest_negative_root_l1300_130064


namespace NUMINAMATH_GPT_rice_grain_difference_l1300_130017

theorem rice_grain_difference :
  (3^8) - (3^1 + 3^2 + 3^3 + 3^4 + 3^5) = 6198 :=
by
  sorry

end NUMINAMATH_GPT_rice_grain_difference_l1300_130017


namespace NUMINAMATH_GPT_find_d_minus_a_l1300_130071

theorem find_d_minus_a (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b = 240)
  (h2 : (b + c) / 2 = 60)
  (h3 : (c + d) / 2 = 90) : d - a = 116 :=
sorry

end NUMINAMATH_GPT_find_d_minus_a_l1300_130071


namespace NUMINAMATH_GPT_find_triples_l1300_130059

theorem find_triples (x y z : ℕ) :
  (1 / x + 2 / y - 3 / z = 1) ↔ 
  ((x = 2 ∧ y = 1 ∧ z = 2) ∨
   (x = 2 ∧ y = 3 ∧ z = 18) ∨
   ∃ (n : ℕ), n ≥ 1 ∧ x = 1 ∧ y = 2 * n ∧ z = 3 * n ∨
   ∃ (k : ℕ), k ≥ 1 ∧ x = k ∧ y = 2 ∧ z = 3 * k) := sorry

end NUMINAMATH_GPT_find_triples_l1300_130059


namespace NUMINAMATH_GPT_compare_fractions_l1300_130014

theorem compare_fractions : (- (4 / 5) < - (2 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_compare_fractions_l1300_130014


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_l1300_130061

variable (n : ℕ)
variable (sumOdd sumEven : ℕ)
variable (terms : ℕ)

theorem arithmetic_sequence_terms
  (h1 : sumOdd = 120)
  (h2 : sumEven = 110)
  (h3 : terms = 2 * n + 1)
  (h4 : sumOdd + sumEven = 230) :
  terms = 23 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_l1300_130061


namespace NUMINAMATH_GPT_length_of_AC_l1300_130011

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (AB BC AC : ℝ)
variables (right_triangle : AB ^ 2 + BC ^ 2 = AC ^ 2)
variables (tan_A : BC / AB = 4 / 3)
variable (AB_val : AB = 4)

theorem length_of_AC :
  AC = 20 / 3 :=
sorry

end NUMINAMATH_GPT_length_of_AC_l1300_130011


namespace NUMINAMATH_GPT_product_of_two_numbers_l1300_130096

theorem product_of_two_numbers :
  ∃ x y : ℝ, x + y = 16 ∧ x^2 + y^2 = 200 ∧ x * y = 28 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1300_130096


namespace NUMINAMATH_GPT_calculate_fraction_l1300_130084

-- Define the fractions we are working with
def fraction1 : ℚ := 3 / 4
def fraction2 : ℚ := 15 / 5
def one_half : ℚ := 1 / 2

-- Define the main calculation
def main_fraction (f1 f2 one_half : ℚ) : ℚ := f1 * f2 - one_half

-- State the theorem
theorem calculate_fraction : main_fraction fraction1 fraction2 one_half = (7 / 4) := by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l1300_130084


namespace NUMINAMATH_GPT_cash_calculation_l1300_130000

theorem cash_calculation 
  (value_gold_coin : ℕ) (value_silver_coin : ℕ) 
  (num_gold_coins : ℕ) (num_silver_coins : ℕ) 
  (total_money : ℕ) : 
  value_gold_coin = 50 → 
  value_silver_coin = 25 → 
  num_gold_coins = 3 → 
  num_silver_coins = 5 → 
  total_money = 305 → 
  (total_money - (num_gold_coins * value_gold_coin + num_silver_coins * value_silver_coin) = 30) := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_cash_calculation_l1300_130000


namespace NUMINAMATH_GPT_committee_formation_l1300_130009

theorem committee_formation :
  let club_size := 15
  let num_roles := 2
  let num_members := 3
  let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
  total_ways = 60060 := by
    let club_size := 15
    let num_roles := 2
    let num_members := 3
    let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
    show total_ways = 60060
    sorry

end NUMINAMATH_GPT_committee_formation_l1300_130009


namespace NUMINAMATH_GPT_distance_between_point_and_center_l1300_130055

noncomputable def polar_to_rectangular_point (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

noncomputable def center_of_circle : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_between_point_and_center :
  distance (polar_to_rectangular_point 2 (Real.pi / 3)) center_of_circle = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_distance_between_point_and_center_l1300_130055


namespace NUMINAMATH_GPT_total_amount_owed_l1300_130033

-- Conditions
def borrowed_amount : ℝ := 500
def monthly_interest_rate : ℝ := 0.02
def months_not_paid : ℕ := 3

-- Compounded monthly formula
def amount_after_n_months (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Theorem statement
theorem total_amount_owed :
  amount_after_n_months borrowed_amount monthly_interest_rate months_not_paid = 530.604 :=
by
  -- Proof to be filled in here
  sorry

end NUMINAMATH_GPT_total_amount_owed_l1300_130033


namespace NUMINAMATH_GPT_larger_number_is_450_l1300_130065

-- Given conditions
def HCF := 30
def Factor1 := 10
def Factor2 := 15

-- Derived definitions needed for the proof
def LCM := HCF * Factor1 * Factor2

def Number1 := LCM / Factor1
def Number2 := LCM / Factor2

-- The goal is to prove the larger of the two numbers is 450
theorem larger_number_is_450 : max Number1 Number2 = 450 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_450_l1300_130065


namespace NUMINAMATH_GPT_parts_processed_per_day_l1300_130076

-- Given conditions
variable (a : ℕ)

-- Goal: Prove the daily productivity of Master Wang given the conditions
theorem parts_processed_per_day (h1 : ∀ n, n = 8) (h2 : ∃ m, m = a + 3):
  (a + 3) / 8 = (a + 3) / 8 :=
by
  sorry

end NUMINAMATH_GPT_parts_processed_per_day_l1300_130076


namespace NUMINAMATH_GPT_archibald_percentage_games_won_l1300_130040

theorem archibald_percentage_games_won
  (A B F1 F2 : ℝ) -- number of games won by Archibald, his brother, and his two friends
  (total_games : ℝ)
  (A_eq_1_1B : A = 1.1 * B)
  (F_eq_2_1B : F1 + F2 = 2.1 * B)
  (total_games_eq : A + B + F1 + F2 = total_games)
  (total_games_val : total_games = 280) :
  (A / total_games * 100) = 26.19 :=
by
  sorry

end NUMINAMATH_GPT_archibald_percentage_games_won_l1300_130040


namespace NUMINAMATH_GPT_nap_time_left_l1300_130089

def train_ride_duration : ℕ := 9
def reading_time : ℕ := 2
def eating_time : ℕ := 1
def watching_movie_time : ℕ := 3

theorem nap_time_left :
  train_ride_duration - (reading_time + eating_time + watching_movie_time) = 3 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_nap_time_left_l1300_130089


namespace NUMINAMATH_GPT_part_i_part_ii_l1300_130024

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + x - a
noncomputable def g (x a : ℝ) : ℝ := Real.sqrt (f x a)

theorem part_i (a : ℝ) :
  (∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f x a ≥ 0) ↔ (a ≤ 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

theorem part_ii (a : ℝ) :
  (∃ x0 y0 : ℝ, (x0, y0) ∈ (Set.Icc (-1) 1) ∧ y0 = Real.cos (2 * x0) ∧ g (g y0 a) a = y0) ↔ (1 ≤ a ∧ a ≤ Real.exp 1) :=
by {
  -- Suppose it is already known that theorem is true.
  sorry
}

end NUMINAMATH_GPT_part_i_part_ii_l1300_130024


namespace NUMINAMATH_GPT_radius_of_cylinder_l1300_130036

-- Define the main parameters and conditions
def diameter_cone := 8
def radius_cone := diameter_cone / 2
def altitude_cone := 10
def height_cylinder (r : ℝ) := 2 * r

-- Assume similarity of triangles
theorem radius_of_cylinder (r : ℝ) (h_c := height_cylinder r) :
  altitude_cone - h_c / r = altitude_cone / radius_cone → r = 20 / 9 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_radius_of_cylinder_l1300_130036


namespace NUMINAMATH_GPT_midpoint_sum_l1300_130054

theorem midpoint_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -4) (hy2 : y2 = -7) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 1 :=
by
  rw [hx1, hy1, hx2, hy2]
  norm_num

end NUMINAMATH_GPT_midpoint_sum_l1300_130054


namespace NUMINAMATH_GPT_sheena_sewing_hours_weekly_l1300_130031

theorem sheena_sewing_hours_weekly
  (hours_per_dress : ℕ)
  (number_of_dresses : ℕ)
  (weeks_to_complete : ℕ)
  (total_sewing_hours : ℕ)
  (hours_per_week : ℕ) :
  hours_per_dress = 12 →
  number_of_dresses = 5 →
  weeks_to_complete = 15 →
  total_sewing_hours = number_of_dresses * hours_per_dress →
  hours_per_week = total_sewing_hours / weeks_to_complete →
  hours_per_week = 4 := by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_sheena_sewing_hours_weekly_l1300_130031


namespace NUMINAMATH_GPT_pipe_filling_time_l1300_130056

theorem pipe_filling_time (T : ℝ) (h1 : T > 0) (h2 : 1/(3:ℝ) = 1/T - 1/(6:ℝ)) : T = 2 := 
by sorry

end NUMINAMATH_GPT_pipe_filling_time_l1300_130056


namespace NUMINAMATH_GPT_largest_x_solution_l1300_130044

noncomputable def solve_eq (x : ℝ) : Prop :=
  (15 * x^2 - 40 * x + 16) / (4 * x - 3) + 3 * x = 7 * x + 2

theorem largest_x_solution : 
  ∃ x : ℝ, solve_eq x ∧ x = -14 + Real.sqrt 218 := 
sorry

end NUMINAMATH_GPT_largest_x_solution_l1300_130044


namespace NUMINAMATH_GPT_alice_leaves_30_minutes_after_bob_l1300_130062

theorem alice_leaves_30_minutes_after_bob :
  ∀ (distance : ℝ) (speed_bob : ℝ) (speed_alice : ℝ) (time_diff : ℝ),
  distance = 220 ∧ speed_bob = 40 ∧ speed_alice = 44 ∧ 
  time_diff = (distance / speed_bob) - (distance / speed_alice) →
  (time_diff * 60 = 30) := by
  intro distance speed_bob speed_alice time_diff
  intro h
  have h1 : distance = 220 := h.1
  have h2 : speed_bob = 40 := h.2.1
  have h3 : speed_alice = 44 := h.2.2.1
  have h4 : time_diff = (distance / speed_bob) - (distance / speed_alice) := h.2.2.2
  sorry

end NUMINAMATH_GPT_alice_leaves_30_minutes_after_bob_l1300_130062


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1300_130012

theorem solve_quadratic_eq (b c : ℝ) :
  (∀ x : ℝ, |x - 3| = 4 ↔ x = 7 ∨ x = -1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = -1) →
  b = -6 ∧ c = -7 :=
by
  intros h_abs_val_eq h_quad_eq
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1300_130012


namespace NUMINAMATH_GPT_fg_sqrt3_eq_neg3_minus_2sqrt3_l1300_130050
noncomputable def f (x : ℝ) : ℝ := 5 - 2 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + x + 1

theorem fg_sqrt3_eq_neg3_minus_2sqrt3 : f (g (Real.sqrt 3)) = -3 - 2 * Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_fg_sqrt3_eq_neg3_minus_2sqrt3_l1300_130050


namespace NUMINAMATH_GPT_sugar_required_in_new_recipe_l1300_130085

theorem sugar_required_in_new_recipe
  (ratio_flour_water_sugar : ℕ × ℕ × ℕ)
  (double_ratio_flour_water : (ℕ → ℕ))
  (half_ratio_flour_sugar : (ℕ → ℕ))
  (new_water_cups : ℕ) :
  ratio_flour_water_sugar = (7, 2, 1) →
  double_ratio_flour_water 7 = 14 → 
  double_ratio_flour_water 2 = 4 →
  half_ratio_flour_sugar 7 = 7 →
  half_ratio_flour_sugar 1 = 2 →
  new_water_cups = 2 →
  (∃ sugar_cups : ℕ, sugar_cups = 1) :=
by
  sorry

end NUMINAMATH_GPT_sugar_required_in_new_recipe_l1300_130085


namespace NUMINAMATH_GPT_find_number_l1300_130094

theorem find_number (x : ℝ) (h : 0.60 * x - 40 = 50) : x = 150 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1300_130094


namespace NUMINAMATH_GPT_taxi_ride_cost_l1300_130018

-- Definitions based on the conditions
def fixed_cost : ℝ := 2.00
def variable_cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 7

-- Theorem statement
theorem taxi_ride_cost : fixed_cost + (variable_cost_per_mile * distance_traveled) = 4.10 :=
by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l1300_130018


namespace NUMINAMATH_GPT_find_k_l1300_130038

-- Given definition for a quadratic expression that we want to be a square of a binomial
def quadratic_expression (x k : ℝ) := x^2 - 20 * x + k

-- The binomial square matching.
def binomial_square (x b : ℝ) := (x + b)^2

-- Statement to prove that k = 100 makes the quadratic_expression to be a square of binomial
theorem find_k :
  (∃ k : ℝ, ∀ x : ℝ, quadratic_expression x k = binomial_square x (-10)) ↔ k = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1300_130038


namespace NUMINAMATH_GPT_final_height_of_tree_in_4_months_l1300_130004

-- Definitions based on the conditions
def growth_rate_cm_per_two_weeks : ℕ := 50
def current_height_meters : ℕ := 2
def weeks_per_month : ℕ := 4
def months : ℕ := 4
def cm_per_meter : ℕ := 100

-- The final height of the tree after 4 months in centimeters
theorem final_height_of_tree_in_4_months : 
  (current_height_meters * cm_per_meter) + 
  (((months * weeks_per_month) / 2) * growth_rate_cm_per_two_weeks) = 600 := 
by
  sorry

end NUMINAMATH_GPT_final_height_of_tree_in_4_months_l1300_130004


namespace NUMINAMATH_GPT_inequality_abc_l1300_130046

theorem inequality_abc (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 2) (h₃ : 0 ≤ b) (h₄ : b ≤ 2) (h₅ : 0 ≤ c) (h₆ : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end NUMINAMATH_GPT_inequality_abc_l1300_130046


namespace NUMINAMATH_GPT_percent_increase_decrease_l1300_130083

theorem percent_increase_decrease (P y : ℝ) (h : (P * (1 + y / 100) * (1 - y / 100) = 0.90 * P)) :
    y = 31.6 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_decrease_l1300_130083


namespace NUMINAMATH_GPT_women_count_l1300_130086

/-- 
Initially, the men and women in a room were in the ratio of 4:5.
Then, 2 men entered the room and 3 women left the room.
The number of women then doubled.
There are now 14 men in the room.
Prove that the number of women currently in the room is 24.
-/
theorem women_count (x : ℕ) (h1 : 4 * x + 2 = 14) (h2 : 2 * (5 * x - 3) = n) : 
  n = 24 :=
by
  sorry

end NUMINAMATH_GPT_women_count_l1300_130086


namespace NUMINAMATH_GPT_intersection_point_correct_l1300_130022

-- Points in 3D coordinate space
def P : ℝ × ℝ × ℝ := (3, -9, 6)
def Q : ℝ × ℝ × ℝ := (13, -19, 11)
def R : ℝ × ℝ × ℝ := (1, 4, -7)
def S : ℝ × ℝ × ℝ := (3, -6, 9)

-- Vectors for parameterization
def pq_vector (t : ℝ) : ℝ × ℝ × ℝ := (3 + 10 * t, -9 - 10 * t, 6 + 5 * t)
def rs_vector (s : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - 10 * s, -7 + 16 * s)

-- The proof of the intersection point equals the correct answer
theorem intersection_point_correct : 
  ∃ t s : ℝ, pq_vector t = rs_vector s ∧ 
  pq_vector t = (-19 / 3, 10 / 3, 4 / 3) := 
by
  sorry

end NUMINAMATH_GPT_intersection_point_correct_l1300_130022


namespace NUMINAMATH_GPT_max_proj_area_l1300_130006

variable {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem max_proj_area : 
  ∃ max_area : ℝ, max_area = Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) :=
by
  sorry

end NUMINAMATH_GPT_max_proj_area_l1300_130006


namespace NUMINAMATH_GPT_charge_per_block_l1300_130023

noncomputable def family_vacation_cost : ℝ := 1000
noncomputable def family_members : ℝ := 5
noncomputable def walk_start_fee : ℝ := 2
noncomputable def dogs_walked : ℝ := 20
noncomputable def total_blocks : ℝ := 128

theorem charge_per_block : 
  (family_vacation_cost / family_members) = 200 →
  (dogs_walked * walk_start_fee) = 40 →
  ((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) = 160 →
  (((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) / total_blocks) = 1.25 :=
by intros h1 h2 h3; sorry

end NUMINAMATH_GPT_charge_per_block_l1300_130023


namespace NUMINAMATH_GPT_smallest_dividend_l1300_130008

   theorem smallest_dividend (b a : ℤ) (q : ℤ := 12) (r : ℤ := 3) (h : a = b * q + r) (h' : r < b) : a = 51 :=
   by
     sorry
   
end NUMINAMATH_GPT_smallest_dividend_l1300_130008


namespace NUMINAMATH_GPT_exist_n_consecutive_not_perfect_power_l1300_130029

theorem exist_n_consecutive_not_perfect_power (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, ∀ k : ℕ, k < n → ¬ (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (m + k) = a ^ b) :=
sorry

end NUMINAMATH_GPT_exist_n_consecutive_not_perfect_power_l1300_130029


namespace NUMINAMATH_GPT_meal_combinations_l1300_130097

theorem meal_combinations (n : ℕ) (h : n = 12) : ∃ m : ℕ, m = 132 :=
by
  -- Initialize the variables for dishes chosen by Yann and Camille
  let yann_choices := n
  let camille_choices := n - 1
  
  -- Calculate the total number of combinations
  let total_combinations := yann_choices * camille_choices
  
  -- Assert the number of combinations is equal to 132
  use total_combinations
  exact sorry

end NUMINAMATH_GPT_meal_combinations_l1300_130097


namespace NUMINAMATH_GPT_min_value_of_fraction_l1300_130077

theorem min_value_of_fraction (m n : ℝ) (h1 : 2 * n + m = 4) (h2 : m > 0) (h3 : n > 0) : 
  (∀ n m, 2 * n + m = 4 ∧ m > 0 ∧ n > 0 → ∀ y, y = 2 / m + 1 / n → y ≥ 2) :=
by sorry

end NUMINAMATH_GPT_min_value_of_fraction_l1300_130077


namespace NUMINAMATH_GPT_simplify_expression_l1300_130016

theorem simplify_expression (x y z : ℝ) : (x - (2 * y + z)) - ((x + 2 * y) - 3 * z) = -4 * y + 2 * z := 
by 
sorry

end NUMINAMATH_GPT_simplify_expression_l1300_130016


namespace NUMINAMATH_GPT_jellybeans_in_jar_now_l1300_130091

def initial_jellybeans : ℕ := 90
def samantha_takes : ℕ := 24
def shelby_takes : ℕ := 12
def scarlett_takes : ℕ := 2 * shelby_takes
def scarlett_returns : ℕ := scarlett_takes / 2
def shannon_refills : ℕ := (samantha_takes + shelby_takes) / 2

theorem jellybeans_in_jar_now : 
  initial_jellybeans 
  - samantha_takes 
  - shelby_takes 
  + scarlett_returns
  + shannon_refills 
  = 84 := by
  sorry

end NUMINAMATH_GPT_jellybeans_in_jar_now_l1300_130091


namespace NUMINAMATH_GPT_geometric_sequence_sum_eq_80_243_l1300_130093

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_eq_80_243 {n : ℕ} :
  let a := (1 / 3 : ℝ)
  let r := (1 / 3 : ℝ)
  geometric_sum a r n = 80 / 243 ↔ n = 3 :=
by
  intros a r
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_eq_80_243_l1300_130093


namespace NUMINAMATH_GPT_alissa_presents_l1300_130041

theorem alissa_presents :
  let Ethan_presents := 31
  let Alissa_presents := Ethan_presents + 22
  Alissa_presents = 53 :=
by
  sorry

end NUMINAMATH_GPT_alissa_presents_l1300_130041


namespace NUMINAMATH_GPT_truncated_cone_surface_area_l1300_130081

theorem truncated_cone_surface_area (R r : ℝ) (S : ℝ)
  (h1: S = 4 * Real.pi * (R^2 + R * r + r^2)) :
  2 * Real.pi * (R^2 + R * r + r^2) = S / 2 :=
by
  sorry

end NUMINAMATH_GPT_truncated_cone_surface_area_l1300_130081


namespace NUMINAMATH_GPT_cards_problem_l1300_130021

-- Definitions of the cards and their arrangement
def cards : List ℕ := [1, 3, 4, 6, 7, 8]
def missing_numbers : List ℕ := [2, 5, 9]

-- Function to check no three consecutive numbers are in ascending or descending order
def no_three_consec (ls : List ℕ) : Prop :=
  ∀ (a b c : ℕ), a < b → b < c → b - a = 1 → c - b = 1 → False ∧
                a > b → b > c → a - b = 1 → b - c = 1 → False

-- Assume that cards A, B, and C are not visible
variables (A B C : ℕ)

-- Ensure that A, B, and C are among the missing numbers
axiom A_in_missing : A ∈ missing_numbers
axiom B_in_missing : B ∈ missing_numbers
axiom C_in_missing : C ∈ missing_numbers

-- Ensuring no three consecutive cards are in ascending or descending order
axiom no_three_consec_cards : no_three_consec (cards ++ [A, B, C])

-- The final proof problem
theorem cards_problem : A = 5 ∧ B = 2 ∧ C = 9 :=
by
  sorry

end NUMINAMATH_GPT_cards_problem_l1300_130021


namespace NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l1300_130035

theorem factorize_expr1 (x y : ℝ) : 
  3 * (x + y) * (x - y) - (x - y)^2 = 2 * (x - y) * (x + 2 * y) :=
by
  sorry

theorem factorize_expr2 (x y : ℝ) : 
  x^2 * (y^2 - 1) + 2 * x * (y^2 - 1) = x * (y + 1) * (y - 1) * (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l1300_130035


namespace NUMINAMATH_GPT_red_light_after_two_red_light_expectation_and_variance_l1300_130049

noncomputable def prob_red_light_after_two : ℚ := (2/3) * (2/3) * (1/3)
theorem red_light_after_two :
  prob_red_light_after_two = 4/27 :=
by
  -- We have defined the probability calculation directly
  sorry

noncomputable def expected_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p
noncomputable def variance_red_lights (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem red_light_expectation_and_variance :
  expected_red_lights 6 (1/3) = 2 ∧ variance_red_lights 6 (1/3) = 4/3 :=
by
  -- We have defined expectation and variance calculations directly
  sorry

end NUMINAMATH_GPT_red_light_after_two_red_light_expectation_and_variance_l1300_130049


namespace NUMINAMATH_GPT_min_sugar_l1300_130048

variable (f s : ℝ)

theorem min_sugar (h1 : f ≥ 10 + 3 * s) (h2 : f ≤ 4 * s) : s ≥ 10 := by
  sorry

end NUMINAMATH_GPT_min_sugar_l1300_130048


namespace NUMINAMATH_GPT_monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l1300_130079

/- Part 1: Monthly Average Growth Rate -/
theorem monthly_average_growth_rate (m : ℝ) (sale_april sale_june : ℝ) (h_apr_val : sale_april = 256) (h_june_val : sale_june = 400) :
  256 * (1 + m) ^ 2 = 400 → m = 0.25 :=
sorry

/- Part 2: Optimal Selling Price for Desired Profit -/
theorem optimal_selling_price_for_desired_profit (y : ℝ) (initial_price selling_price : ℝ) (sale_june : ℝ) (h_june_sale : sale_june = 400) (profit : ℝ) (h_profit : profit = 8400) :
  (y - 35) * (1560 - 20 * y) = 8400 → y = 50 :=
sorry

end NUMINAMATH_GPT_monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l1300_130079


namespace NUMINAMATH_GPT_bridge_length_correct_l1300_130060

noncomputable def length_of_bridge : ℝ :=
  let train_length := 110 -- in meters
  let train_speed_kmh := 72 -- in km/hr
  let crossing_time := 14.248860091192705 -- in seconds
  let speed_in_mps := train_speed_kmh * (1000 / 3600)
  let distance := speed_in_mps * crossing_time
  distance - train_length

theorem bridge_length_correct :
  length_of_bridge = 174.9772018238541 := by
  sorry

end NUMINAMATH_GPT_bridge_length_correct_l1300_130060


namespace NUMINAMATH_GPT_radius_of_spheres_in_cone_l1300_130058

def base_radius := 8
def cone_height := 15
def num_spheres := 3
def spheres_are_tangent := true

theorem radius_of_spheres_in_cone :
  ∃ (r : ℝ), r = (280 - 100 * Real.sqrt 3) / 121 :=
sorry

end NUMINAMATH_GPT_radius_of_spheres_in_cone_l1300_130058


namespace NUMINAMATH_GPT_unique_solution_quadratic_eq_l1300_130010

theorem unique_solution_quadratic_eq (p : ℝ) (h_nonzero : p ≠ 0) : (∀ x : ℝ, p * x^2 - 20 * x + 4 = 0) → p = 25 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_quadratic_eq_l1300_130010


namespace NUMINAMATH_GPT_value_of_expression_l1300_130099

theorem value_of_expression : 4 * (8 - 6) - 7 = 1 := by
  -- Calculation steps would go here
  sorry

end NUMINAMATH_GPT_value_of_expression_l1300_130099


namespace NUMINAMATH_GPT_number_of_passed_boys_l1300_130002

theorem number_of_passed_boys 
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 36 * 120) :
  P = 105 := 
sorry

end NUMINAMATH_GPT_number_of_passed_boys_l1300_130002


namespace NUMINAMATH_GPT_tire_swap_distance_l1300_130005

theorem tire_swap_distance : ∃ x : ℕ, 
  (1 - x / 11000) * 9000 = (1 - x / 9000) * 11000 ∧ x = 4950 := 
by
  sorry

end NUMINAMATH_GPT_tire_swap_distance_l1300_130005


namespace NUMINAMATH_GPT_part1_part2_l1300_130007

variable {α : Type*}
def A : Set ℝ := {x | 0 < x ∧ x < 9}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part (1)
theorem part1 : B 5 ∩ A = {x | 6 ≤ x ∧ x < 9} := 
sorry

-- Part (2)
theorem part2 (m : ℝ): A ∩ B m = B m ↔ m < 5 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1300_130007


namespace NUMINAMATH_GPT_delta_evaluation_l1300_130069

def delta (a b : ℕ) : ℕ := a^3 - b

theorem delta_evaluation :
  delta (2^(delta 3 8)) (5^(delta 4 9)) = 2^19 - 5^55 := 
sorry

end NUMINAMATH_GPT_delta_evaluation_l1300_130069


namespace NUMINAMATH_GPT_find_theta_l1300_130074

-- Definitions based on conditions
def angle_A : ℝ := 10
def angle_B : ℝ := 14
def angle_C : ℝ := 26
def angle_D : ℝ := 33
def sum_rect_angles : ℝ := 360
def sum_triangle_angles : ℝ := 180
def sum_right_triangle_acute_angles : ℝ := 90

-- Main theorem statement
theorem find_theta (A B C D : ℝ)
  (hA : A = angle_A)
  (hB : B = angle_B)
  (hC : C = angle_C)
  (hD : D = angle_D)
  (sum_rect : sum_rect_angles = 360)
  (sum_triangle : sum_triangle_angles = 180) :
  ∃ θ : ℝ, θ = 11 := 
sorry

end NUMINAMATH_GPT_find_theta_l1300_130074


namespace NUMINAMATH_GPT_math_problem_l1300_130025

theorem math_problem (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y * Complex.I)^2 - 46 * Complex.I = z) :
  x + y + z = 552 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1300_130025


namespace NUMINAMATH_GPT_leap_years_count_l1300_130019

theorem leap_years_count :
  let is_leap_year (y : ℕ) := (y % 900 = 150 ∨ y % 900 = 450) ∧ y % 100 = 0
  let range_start := 2100
  let range_end := 4200
  ∃ L, L = [2250, 2850, 3150, 3750, 4050] ∧ (∀ y ∈ L, is_leap_year y ∧ range_start ≤ y ∧ y ≤ range_end)
  ∧ L.length = 5 :=
by
  sorry

end NUMINAMATH_GPT_leap_years_count_l1300_130019


namespace NUMINAMATH_GPT_binomial_constant_term_l1300_130052

theorem binomial_constant_term : 
  ∃ (c : ℚ), (x : ℝ) → (x^2 + (1 / (2 * x)))^6 = c ∧ c = 15 / 16 := by
  sorry

end NUMINAMATH_GPT_binomial_constant_term_l1300_130052
