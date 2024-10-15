import Mathlib

namespace NUMINAMATH_GPT_simple_interest_rate_l2016_201679

theorem simple_interest_rate (P R T A : ℝ) (h_double: A = 2 * P) (h_si: A = P + P * R * T / 100) (h_T: T = 5) : R = 20 :=
by
  have h1: A = 2 * P := h_double
  have h2: A = P + P * R * T / 100 := h_si
  have h3: T = 5 := h_T
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l2016_201679


namespace NUMINAMATH_GPT_sam_has_75_dollars_l2016_201673

variable (S B : ℕ)

def condition1 := B = 2 * S - 25
def condition2 := S + B = 200

theorem sam_has_75_dollars (h1 : condition1 S B) (h2 : condition2 S B) : S = 75 := by
  sorry

end NUMINAMATH_GPT_sam_has_75_dollars_l2016_201673


namespace NUMINAMATH_GPT_mother_stickers_given_l2016_201638

-- Definitions based on the conditions
def initial_stickers : ℝ := 20.0
def bought_stickers : ℝ := 26.0
def birthday_stickers : ℝ := 20.0
def sister_stickers : ℝ := 6.0
def total_stickers : ℝ := 130.0

-- Statement of the problem to be proved in Lean 4.
theorem mother_stickers_given :
  initial_stickers + bought_stickers + birthday_stickers + sister_stickers + 58.0 = total_stickers :=
by
  sorry

end NUMINAMATH_GPT_mother_stickers_given_l2016_201638


namespace NUMINAMATH_GPT_hall_breadth_l2016_201624

theorem hall_breadth (l : ℝ) (w_s l_s b : ℝ) (n : ℕ)
  (hall_length : l = 36)
  (stone_width : w_s = 0.4)
  (stone_length : l_s = 0.5)
  (num_stones : n = 2700)
  (area_paving : l * b = n * (w_s * l_s)) :
  b = 15 := by
  sorry

end NUMINAMATH_GPT_hall_breadth_l2016_201624


namespace NUMINAMATH_GPT_price_of_AC_l2016_201672

theorem price_of_AC (x : ℝ) (price_car price_ac : ℝ)
  (h1 : price_car = 3 * x) 
  (h2 : price_ac = 2 * x) 
  (h3 : price_car = price_ac + 500) : 
  price_ac = 1000 := sorry

end NUMINAMATH_GPT_price_of_AC_l2016_201672


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_condition_l2016_201639

noncomputable def sum_first_n_terms (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_sum_condition (a_1 d : ℤ) :
  sum_first_n_terms a_1 d 3 = 3 →
  sum_first_n_terms a_1 d 6 = 15 →
  (a_1 + 9 * d) + (a_1 + 10 * d) + (a_1 + 11 * d) = 30 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_condition_l2016_201639


namespace NUMINAMATH_GPT_drink_exactly_five_bottles_last_day_l2016_201652

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

end NUMINAMATH_GPT_drink_exactly_five_bottles_last_day_l2016_201652


namespace NUMINAMATH_GPT_petya_time_comparison_l2016_201634

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end NUMINAMATH_GPT_petya_time_comparison_l2016_201634


namespace NUMINAMATH_GPT_find_percentage_l2016_201654

variable (dollars_1 dollars_2 dollars_total interest_total percentage_unknown : ℝ)
variable (investment_1 investment_rest interest_2 : ℝ)
variable (P : ℝ)

-- Assuming given conditions
axiom H1 : dollars_total = 12000
axiom H2 : dollars_1 = 5500
axiom H3 : interest_total = 970
axiom H4 : investment_rest = dollars_total - dollars_1
axiom H5 : interest_2 = investment_rest * 0.09
axiom H6 : interest_total = dollars_1 * P + interest_2

-- Prove that P = 0.07
theorem find_percentage : P = 0.07 :=
by
  -- Placeholder for the proof that needs to be filled in
  sorry

end NUMINAMATH_GPT_find_percentage_l2016_201654


namespace NUMINAMATH_GPT_calculate_expression_l2016_201618

theorem calculate_expression :
  let s1 := 3 + 6 + 9
  let s2 := 4 + 8 + 12
  s1 = 18 → s2 = 24 → (s1 / s2 + s2 / s1) = 25 / 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_calculate_expression_l2016_201618


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2016_201668

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  (∃ a b c : ℝ, a = k - 2 ∧ b = -2 ∧ c = 1 / 2 ∧ a ≠ 0 ∧ b ^ 2 - 4 * a * c > 0) ↔ (k < 4 ∧ k ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2016_201668


namespace NUMINAMATH_GPT_B_pow_16_eq_I_l2016_201631

noncomputable def B : Matrix (Fin 4) (Fin 4) ℝ := 
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4), 0 , 0],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4), 0 , 0],
    ![0, 0, Real.cos (Real.pi / 4), Real.sin (Real.pi / 4)],
    ![0, 0, -Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem B_pow_16_eq_I : B^16 = 1 := by
  sorry

end NUMINAMATH_GPT_B_pow_16_eq_I_l2016_201631


namespace NUMINAMATH_GPT_incorrect_proposition_l2016_201622

theorem incorrect_proposition (p q : Prop) :
  ¬(¬(p ∧ q) → ¬p ∧ ¬q) := sorry

end NUMINAMATH_GPT_incorrect_proposition_l2016_201622


namespace NUMINAMATH_GPT_largest_integer_chosen_l2016_201658

-- Define the sequence of operations and establish the resulting constraints
def transformed_value (x : ℤ) : ℤ :=
  2 * (4 * x - 30) - 10

theorem largest_integer_chosen : 
  ∃ (x : ℤ), (10 : ℤ) ≤ transformed_value x ∧ transformed_value x ≤ (99 : ℤ) ∧ x = 21 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_chosen_l2016_201658


namespace NUMINAMATH_GPT_solve_for_a_l2016_201669

theorem solve_for_a (a : ℚ) (h : 2 * a - 3 = 5 - a) : a = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2016_201669


namespace NUMINAMATH_GPT_player_reach_wingspan_l2016_201680

theorem player_reach_wingspan :
  ∀ (rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan : ℕ),
  rim_height = 120 →
  player_height = 72 →
  jump_height = 32 →
  reach_above_rim = 6 →
  reach_with_jump = player_height + jump_height →
  reach_wingspan = (rim_height + reach_above_rim) - reach_with_jump →
  reach_wingspan = 22 :=
by
  intros rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan
  intros h_rim_height h_player_height h_jump_height h_reach_above_rim h_reach_with_jump h_reach_wingspan
  rw [h_rim_height, h_player_height, h_jump_height, h_reach_above_rim] at *
  simp at *
  sorry

end NUMINAMATH_GPT_player_reach_wingspan_l2016_201680


namespace NUMINAMATH_GPT_average_of_multiples_of_6_l2016_201674

def first_n_multiples_sum (n : ℕ) : ℕ :=
  (n * (6 + 6 * n)) / 2

def first_n_multiples_avg (n : ℕ) : ℕ :=
  (first_n_multiples_sum n) / n

theorem average_of_multiples_of_6 (n : ℕ) : first_n_multiples_avg n = 66 → n = 11 := by
  sorry

end NUMINAMATH_GPT_average_of_multiples_of_6_l2016_201674


namespace NUMINAMATH_GPT_exists_2013_distinct_numbers_l2016_201651

theorem exists_2013_distinct_numbers : 
  ∃ (a : ℕ → ℕ), 
    (∀ m n, m ≠ n → m < 2013 ∧ n < 2013 → (a m + a n) % (a m - a n) = 0) ∧
    (∀ k l, k < 2013 ∧ l < 2013 → (a k) ≠ (a l)) :=
sorry

end NUMINAMATH_GPT_exists_2013_distinct_numbers_l2016_201651


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_expression_l2016_201689

theorem minimum_value_of_quadratic_expression (x y z : ℝ)
  (h : x + y + z = 2) : 
  x^2 + 2 * y^2 + z^2 ≥ 4 / 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_expression_l2016_201689


namespace NUMINAMATH_GPT_largest_multiple_of_15_who_negation_greater_than_neg_150_l2016_201688

theorem largest_multiple_of_15_who_negation_greater_than_neg_150 : 
  ∃ (x : ℤ), x % 15 = 0 ∧ -x > -150 ∧ ∀ (y : ℤ), y % 15 = 0 ∧ -y > -150 → x ≥ y :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_15_who_negation_greater_than_neg_150_l2016_201688


namespace NUMINAMATH_GPT_biker_bob_east_distance_l2016_201653

noncomputable def distance_between_towns : ℝ := 28.30194339616981
noncomputable def distance_west : ℝ := 30
noncomputable def distance_north_1 : ℝ := 6
noncomputable def distance_north_2 : ℝ := 18
noncomputable def total_distance_north : ℝ := distance_north_1 + distance_north_2
noncomputable def unknown_distance_east : ℝ := 45.0317 -- Expected distance east

theorem biker_bob_east_distance :
  ∃ (E : ℝ), (total_distance_north ^ 2 + (-distance_west + E) ^ 2 = distance_between_towns ^ 2) ∧ E = unknown_distance_east :=
by 
  sorry

end NUMINAMATH_GPT_biker_bob_east_distance_l2016_201653


namespace NUMINAMATH_GPT_dividend_is_686_l2016_201629

theorem dividend_is_686 (divisor quotient remainder : ℕ) (h1 : divisor = 36) (h2 : quotient = 19) (h3 : remainder = 2) :
  divisor * quotient + remainder = 686 :=
by
  sorry

end NUMINAMATH_GPT_dividend_is_686_l2016_201629


namespace NUMINAMATH_GPT_no_real_solutions_for_g_g_x_l2016_201613

theorem no_real_solutions_for_g_g_x (d : ℝ) :
  ¬ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 4 * x1 + d)^2 + 4 * (x1^2 + 4 * x1 + d) + d = 0 ∧
                                (x2^2 + 4 * x2 + d)^2 + 4 * (x2^2 + 4 * x2 + d) + d = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_for_g_g_x_l2016_201613


namespace NUMINAMATH_GPT_slope_of_given_line_l2016_201663

def slope_of_line (l : String) : Real :=
  -- Assuming that we have a function to parse the line equation
  -- and extract its slope. Normally, this would be a complex parsing function.
  1 -- Placeholder, as the slope calculation logic is trivial here.

theorem slope_of_given_line : slope_of_line "x - y - 1 = 0" = 1 := by
  sorry

end NUMINAMATH_GPT_slope_of_given_line_l2016_201663


namespace NUMINAMATH_GPT_find_number_l2016_201676

theorem find_number (x : ℤ) (h : x - (28 - (37 - (15 - 16))) = 55) : x = 65 :=
sorry

end NUMINAMATH_GPT_find_number_l2016_201676


namespace NUMINAMATH_GPT_count_negative_numbers_l2016_201670

theorem count_negative_numbers : 
  (List.filter (λ x => x < (0:ℚ)) [-14, 7, 0, -2/3, -5/16]).length = 3 := 
by
  sorry

end NUMINAMATH_GPT_count_negative_numbers_l2016_201670


namespace NUMINAMATH_GPT_total_valid_arrangements_l2016_201682

-- Define the students and schools
inductive Student
| G1 | G2 | B1 | B2 | B3 | BA
deriving DecidableEq

inductive School
| A | B | C
deriving DecidableEq

-- Define the condition that any two students cannot be in the same school
def is_valid_arrangement (arr : School → Student → Bool) : Bool :=
  (arr School.A Student.G1 ≠ arr School.A Student.G2) ∧ 
  (arr School.B Student.G1 ≠ arr School.B Student.G2) ∧
  (arr School.C Student.G1 ≠ arr School.C Student.G2) ∧
  ¬ arr School.C Student.G1 ∧
  ¬ arr School.C Student.G2 ∧
  ¬ arr School.A Student.BA

-- The theorem to prove the total number of different valid arrangements
theorem total_valid_arrangements : 
  ∃ n : ℕ, n = 18 ∧ ∃ arr : (School → Student → Bool), is_valid_arrangement arr := 
sorry

end NUMINAMATH_GPT_total_valid_arrangements_l2016_201682


namespace NUMINAMATH_GPT_pencils_count_l2016_201600

theorem pencils_count (initial_pencils additional_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : additional_pencils = 100) : initial_pencils + additional_pencils = 215 :=
by sorry

end NUMINAMATH_GPT_pencils_count_l2016_201600


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l2016_201665

theorem triangle_angle_contradiction (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A > 60) (h₃ : B > 60) (h₄ : C > 60) :
  false :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l2016_201665


namespace NUMINAMATH_GPT_sequence_2007th_number_l2016_201641

-- Defining the sequence according to the given rule
def a (n : ℕ) : ℕ := 2 ^ n

theorem sequence_2007th_number : a 2007 = 2 ^ 2007 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sequence_2007th_number_l2016_201641


namespace NUMINAMATH_GPT_consecutive_days_probability_l2016_201690

noncomputable def probability_of_consecutive_days : ℚ :=
  let total_days := 5
  let combinations := Nat.choose total_days 2
  let consecutive_pairs := 4
  consecutive_pairs / combinations

theorem consecutive_days_probability :
  probability_of_consecutive_days = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_days_probability_l2016_201690


namespace NUMINAMATH_GPT_imo_42_problem_l2016_201644

theorem imo_42_problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1 :=
sorry

end NUMINAMATH_GPT_imo_42_problem_l2016_201644


namespace NUMINAMATH_GPT_graph_not_in_first_quadrant_l2016_201687

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

-- Prove that the graph of f(x) does not pass through the first quadrant
theorem graph_not_in_first_quadrant : ∀ (x : ℝ), x > 0 → f x ≤ 0 := by
  intro x hx
  sorry

end NUMINAMATH_GPT_graph_not_in_first_quadrant_l2016_201687


namespace NUMINAMATH_GPT_maria_total_money_l2016_201659

theorem maria_total_money (Rene Florence Isha : ℕ) (hRene : Rene = 300)
  (hFlorence : Florence = 3 * Rene) (hIsha : Isha = Florence / 2) :
  Isha + Florence + Rene = 1650 := by
  sorry

end NUMINAMATH_GPT_maria_total_money_l2016_201659


namespace NUMINAMATH_GPT_find_base_k_l2016_201609

theorem find_base_k : ∃ k : ℕ, 6 * k^2 + 6 * k + 4 = 340 ∧ k = 7 := 
by 
  sorry

end NUMINAMATH_GPT_find_base_k_l2016_201609


namespace NUMINAMATH_GPT_largest_number_of_four_consecutive_whole_numbers_l2016_201667

theorem largest_number_of_four_consecutive_whole_numbers 
  (a : ℕ) (h1 : a + (a + 1) + (a + 2) = 184)
  (h2 : a + (a + 1) + (a + 3) = 201)
  (h3 : a + (a + 2) + (a + 3) = 212)
  (h4 : (a + 1) + (a + 2) + (a + 3) = 226) : 
  a + 3 = 70 := 
by sorry

end NUMINAMATH_GPT_largest_number_of_four_consecutive_whole_numbers_l2016_201667


namespace NUMINAMATH_GPT_mary_gave_becky_green_crayons_l2016_201698

-- Define the initial conditions
def initial_green_crayons : Nat := 5
def initial_blue_crayons : Nat := 8
def given_blue_crayons : Nat := 1
def remaining_crayons : Nat := 9

-- Define the total number of crayons initially
def total_initial_crayons : Nat := initial_green_crayons + initial_blue_crayons

-- Define the number of crayons given away
def given_crayons : Nat := total_initial_crayons - remaining_crayons

-- The crux of the problem
def given_green_crayons : Nat :=
  given_crayons - given_blue_crayons

-- Formal statement of the theorem
theorem mary_gave_becky_green_crayons
  (h_initial_green : initial_green_crayons = 5)
  (h_initial_blue : initial_blue_crayons = 8)
  (h_given_blue : given_blue_crayons = 1)
  (h_remaining : remaining_crayons = 9) :
  given_green_crayons = 3 :=
by {
  -- This should be the body of the proof, but we'll skip it for now
  sorry
}

end NUMINAMATH_GPT_mary_gave_becky_green_crayons_l2016_201698


namespace NUMINAMATH_GPT_number_of_blue_parrots_l2016_201684

-- Defining the known conditions
def total_parrots : ℕ := 120
def fraction_red : ℚ := 2 / 3
def fraction_green : ℚ := 1 / 6

-- Proving the number of blue parrots given the conditions
theorem number_of_blue_parrots : (1 - (fraction_red + fraction_green)) * total_parrots = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_blue_parrots_l2016_201684


namespace NUMINAMATH_GPT_quadratic_inequality_l2016_201614

theorem quadratic_inequality
  (a b c : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l2016_201614


namespace NUMINAMATH_GPT_max_distinct_colorings_5x5_l2016_201643

theorem max_distinct_colorings_5x5 (n : ℕ) :
  ∃ N, N ≤ (n^25 + 4 * n^15 + n^13 + 2 * n^7) / 8 :=
sorry

end NUMINAMATH_GPT_max_distinct_colorings_5x5_l2016_201643


namespace NUMINAMATH_GPT_total_dolls_l2016_201615

-- Definitions based on the given conditions
def grandmother_dolls : Nat := 50
def sister_dolls : Nat := grandmother_dolls + 2
def rene_dolls : Nat := 3 * sister_dolls

-- Statement we want to prove
theorem total_dolls : grandmother_dolls + sister_dolls + rene_dolls = 258 := by
  sorry

end NUMINAMATH_GPT_total_dolls_l2016_201615


namespace NUMINAMATH_GPT_cos_C_eq_3_5_l2016_201646

theorem cos_C_eq_3_5 (A B C : ℝ) (hABC : A^2 + B^2 = C^2) (hRight : B ^ 2 + C ^ 2 = A ^ 2) (hTan : B / C = 4 / 3) : B / A = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_C_eq_3_5_l2016_201646


namespace NUMINAMATH_GPT_linda_age_13_l2016_201678

variable (J L : ℕ)

-- Conditions: 
-- 1. Linda is 3 more than 2 times the age of Jane.
-- 2. In five years, the sum of their ages will be 28.
def conditions (J L : ℕ) : Prop :=
  L = 2 * J + 3 ∧ (J + 5) + (L + 5) = 28

-- Question/answer to prove: Linda's current age is 13.
theorem linda_age_13 (J L : ℕ) (h : conditions J L) : L = 13 :=
by
  sorry

end NUMINAMATH_GPT_linda_age_13_l2016_201678


namespace NUMINAMATH_GPT_correct_negation_statement_l2016_201664

def Person : Type := sorry

def is_adult (p : Person) : Prop := sorry
def is_teenager (p : Person) : Prop := sorry
def is_responsible (p : Person) : Prop := sorry
def is_irresponsible (p : Person) : Prop := sorry

axiom all_adults_responsible : ∀ p, is_adult p → is_responsible p
axiom some_adults_responsible : ∃ p, is_adult p ∧ is_responsible p
axiom no_teenagers_responsible : ∀ p, is_teenager p → ¬is_responsible p
axiom all_teenagers_irresponsible : ∀ p, is_teenager p → is_irresponsible p
axiom exists_irresponsible_teenager : ∃ p, is_teenager p ∧ is_irresponsible p
axiom all_teenagers_responsible : ∀ p, is_teenager p → is_responsible p

theorem correct_negation_statement
: (∃ p, is_teenager p ∧ ¬is_responsible p) ↔ 
  (∃ p, is_teenager p ∧ is_irresponsible p) :=
sorry

end NUMINAMATH_GPT_correct_negation_statement_l2016_201664


namespace NUMINAMATH_GPT_g_neg_one_add_g_one_l2016_201697

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x - y) = f x * g y - f y * g x
axiom f_one_ne_zero : f 1 ≠ 0
axiom f_one_eq_f_two : f 1 = f 2

theorem g_neg_one_add_g_one : g (-1) + g 1 = 1 := by
  sorry

end NUMINAMATH_GPT_g_neg_one_add_g_one_l2016_201697


namespace NUMINAMATH_GPT_Ann_age_is_39_l2016_201650

def current_ages (A B : ℕ) : Prop :=
  A + B = 52 ∧ (B = 2 * B - A / 3) ∧ (A = 3 * B)

theorem Ann_age_is_39 : ∃ A B : ℕ, current_ages A B ∧ A = 39 :=
by
  sorry

end NUMINAMATH_GPT_Ann_age_is_39_l2016_201650


namespace NUMINAMATH_GPT_sin_pi_minus_alpha_l2016_201602

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi - α) = -1/3) : Real.sin α = -1/3 :=
sorry

end NUMINAMATH_GPT_sin_pi_minus_alpha_l2016_201602


namespace NUMINAMATH_GPT_range_of_m_l2016_201616

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := -x^2 + x + m + 2

theorem range_of_m (m : ℝ) : 
  (∃! x : ℤ, f x m ≥ |x|) ↔ -2 ≤ m ∧ m < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2016_201616


namespace NUMINAMATH_GPT_semiperimeter_inequality_l2016_201604

theorem semiperimeter_inequality (p R r : ℝ) (hp : p ≥ 0) (hR : R ≥ 0) (hr : r ≥ 0) :
  p ≥ (3 / 2) * Real.sqrt (6 * R * r) :=
sorry

end NUMINAMATH_GPT_semiperimeter_inequality_l2016_201604


namespace NUMINAMATH_GPT_rita_bought_four_pounds_l2016_201671

def initial_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def left_amount : ℝ := 35.68

theorem rita_bought_four_pounds :
  (initial_amount - left_amount) / cost_per_pound = 4 :=
by
  sorry

end NUMINAMATH_GPT_rita_bought_four_pounds_l2016_201671


namespace NUMINAMATH_GPT_faye_age_l2016_201611
open Nat

theorem faye_age :
  ∃ (C D E F : ℕ), 
    (D = E - 3) ∧ 
    (E = C + 4) ∧ 
    (F = C + 3) ∧ 
    (D = 14) ∧ 
    (F = 16) :=
by
  sorry

end NUMINAMATH_GPT_faye_age_l2016_201611


namespace NUMINAMATH_GPT_smallest_k_l2016_201666

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_l2016_201666


namespace NUMINAMATH_GPT_prob1_prob2_l2016_201635

-- Define lines l1 and l2
def l1 (x y m : ℝ) : Prop := x + m * y + 1 = 0
def l2 (x y m : ℝ) : Prop := (m - 3) * x - 2 * y + (13 - 7 * m) = 0

-- Perpendicular condition
def perp_cond (m : ℝ) : Prop := 1 * (m - 3) - 2 * m = 0

-- Parallel condition
def parallel_cond (m : ℝ) : Prop := m * (m - 3) + 2 = 0

-- Distance between parallel lines when m = 1
def distance_between_parallel_lines (d : ℝ) : Prop := d = 2 * Real.sqrt 2

-- Problem 1: Prove that if l1 ⊥ l2, then m = -3
theorem prob1 (m : ℝ) (h : perp_cond m) : m = -3 := sorry

-- Problem 2: Prove that if l1 ∥ l2, the distance d is 2√2
theorem prob2 (m : ℝ) (h1 : parallel_cond m) (d : ℝ) (h2 : m = 1 ∨ m = -2) (h3 : m = 1) (h4 : distance_between_parallel_lines d) : d = 2 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_prob1_prob2_l2016_201635


namespace NUMINAMATH_GPT_value_of_fraction_power_series_l2016_201681

theorem value_of_fraction_power_series (x : ℕ) (h : x = 3) :
  (x^3 * x^5 * x^7 * x^9 * x^11 * x^13 * x^15 * x^17 * x^19 * x^21) /
  (x^4 * x^8 * x^12 * x^16 * x^20 * x^24) = 3^36 :=
by
  subst h
  sorry

end NUMINAMATH_GPT_value_of_fraction_power_series_l2016_201681


namespace NUMINAMATH_GPT_number_of_tables_l2016_201640

theorem number_of_tables (c t : ℕ) (h1 : c = 8 * t) (h2 : 4 * c + 3 * t = 759) : t = 22 := by
  sorry

end NUMINAMATH_GPT_number_of_tables_l2016_201640


namespace NUMINAMATH_GPT_number_of_students_in_line_l2016_201691

-- Definitions for the conditions
def yoojung_last (n : ℕ) : Prop :=
  n = 14

def eunjung_position : ℕ := 5

def students_between (n : ℕ) : Prop :=
  n = 8

noncomputable def total_students : ℕ := 14

-- The theorem to be proven
theorem number_of_students_in_line 
  (last : yoojung_last total_students) 
  (eunjung_pos : eunjung_position = 5) 
  (between : students_between 8) :
  total_students = 14 := by
  sorry

end NUMINAMATH_GPT_number_of_students_in_line_l2016_201691


namespace NUMINAMATH_GPT_parabola_min_y1_y2_squared_l2016_201630

theorem parabola_min_y1_y2_squared (x1 x2 y1 y2 : ℝ) :
  (y1^2 = 4 * x1) ∧
  (y2^2 = 4 * x2) ∧
  (x1 * x2 = 16) →
  (y1^2 + y2^2 ≥ 32) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_min_y1_y2_squared_l2016_201630


namespace NUMINAMATH_GPT_task_probabilities_l2016_201695

theorem task_probabilities (P1_on_time : ℚ) (P2_on_time : ℚ) 
  (h1 : P1_on_time = 2/3) (h2 : P2_on_time = 3/5) : 
  P1_on_time * (1 - P2_on_time) = 4/15 := 
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_task_probabilities_l2016_201695


namespace NUMINAMATH_GPT_cartesian_equation_of_parametric_l2016_201686

variable (t : ℝ) (x y : ℝ)

open Real

theorem cartesian_equation_of_parametric 
  (h1 : x = sqrt t)
  (h2 : y = 2 * sqrt (1 - t))
  (h3 : 0 ≤ t ∧ t ≤ 1) :
  (x^2 / 1) + (y^2 / 4) = 1 := by 
  sorry

end NUMINAMATH_GPT_cartesian_equation_of_parametric_l2016_201686


namespace NUMINAMATH_GPT_infinite_series_equals_l2016_201637

noncomputable def infinite_series : Real :=
  ∑' n, if h : (n : ℕ) ≥ 2 then (n^4 + 2 * n^3 + 8 * n^2 + 8 * n + 8) / (2^n * (n^4 + 4)) else 0

theorem infinite_series_equals : infinite_series = 11 / 10 :=
  sorry

end NUMINAMATH_GPT_infinite_series_equals_l2016_201637


namespace NUMINAMATH_GPT_remove_terms_sum_equals_one_l2016_201627

theorem remove_terms_sum_equals_one :
  let seq := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
  let remove := [1/12, 1/15]
  (seq.sum - remove.sum) = 1 :=
by
  sorry

end NUMINAMATH_GPT_remove_terms_sum_equals_one_l2016_201627


namespace NUMINAMATH_GPT_diagonals_in_25_sided_polygon_l2016_201694

-- Define a function to calculate the number of specific diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 5) / 2

-- Theorem stating the number of diagonals for a convex polygon with 25 sides with the given condition
theorem diagonals_in_25_sided_polygon : number_of_diagonals 25 = 250 := 
sorry

end NUMINAMATH_GPT_diagonals_in_25_sided_polygon_l2016_201694


namespace NUMINAMATH_GPT_james_profit_l2016_201636

def cattle_profit (num_cattle : ℕ) (purchase_price total_feed_increase : ℝ)
    (weight_per_cattle : ℝ) (selling_price_per_pound : ℝ) : ℝ :=
  let feed_cost := purchase_price * (1 + total_feed_increase)
  let total_cost := purchase_price + feed_cost
  let revenue_per_cattle := weight_per_cattle * selling_price_per_pound
  let total_revenue := revenue_per_cattle * num_cattle
  total_revenue - total_cost

theorem james_profit : cattle_profit 100 40000 0.20 1000 2 = 112000 := by
  sorry

end NUMINAMATH_GPT_james_profit_l2016_201636


namespace NUMINAMATH_GPT_train_length_l2016_201603

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (h1 : speed_kmh = 90) (h2 : time_s = 12) : 
  ∃ length_m : ℕ, length_m = 300 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l2016_201603


namespace NUMINAMATH_GPT_days_required_for_C_l2016_201620

noncomputable def rate_A (r_A r_B r_C : ℝ) : Prop := r_A + r_B = 1 / 3
noncomputable def rate_B (r_A r_B r_C : ℝ) : Prop := r_B + r_C = 1 / 6
noncomputable def rate_C (r_A r_B r_C : ℝ) : Prop := r_C + r_A = 1 / 4
noncomputable def days_for_C (r_C : ℝ) : ℝ := 1 / r_C

theorem days_required_for_C
  (r_A r_B r_C : ℝ)
  (h1 : rate_A r_A r_B r_C)
  (h2 : rate_B r_A r_B r_C)
  (h3 : rate_C r_A r_B r_C) :
  days_for_C r_C = 4.8 :=
sorry

end NUMINAMATH_GPT_days_required_for_C_l2016_201620


namespace NUMINAMATH_GPT_total_price_of_bananas_and_oranges_l2016_201699

variable (price_orange price_pear price_banana : ℝ)

axiom total_cost_orange_pear : price_orange + price_pear = 120
axiom cost_pear : price_pear = 90
axiom diff_orange_pear_banana : price_orange - price_pear = price_banana

theorem total_price_of_bananas_and_oranges :
  let num_bananas := 200
  let num_oranges := 2 * num_bananas
  let cost_bananas := num_bananas * price_banana
  let cost_oranges := num_oranges * price_orange
  cost_bananas + cost_oranges = 24000 :=
by
  sorry

end NUMINAMATH_GPT_total_price_of_bananas_and_oranges_l2016_201699


namespace NUMINAMATH_GPT_directrix_of_parabola_l2016_201648

theorem directrix_of_parabola :
  ∀ (x : ℝ), (∃ k : ℝ, y = (x^2 - 8 * x + 16) / 8 → k = -2) :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l2016_201648


namespace NUMINAMATH_GPT_range_of_a_l2016_201623

noncomputable def matrix_det_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem range_of_a : 
  {a : ℝ | matrix_det_2x2 (a^2) 1 3 2 < matrix_det_2x2 a 0 4 1} = {a : ℝ | -1 < a ∧ a < 3/2} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2016_201623


namespace NUMINAMATH_GPT_fair_people_ratio_l2016_201693

def next_year_ratio (this_year next_year last_year : ℕ) (total : ℕ) :=
  this_year = 600 ∧
  last_year = next_year - 200 ∧
  this_year + last_year + next_year = total → 
  next_year = 2 * this_year

theorem fair_people_ratio :
  ∀ (next_year : ℕ),
  next_year_ratio 600 next_year (next_year - 200) 2800 → next_year = 2 * 600 := by
sorry

end NUMINAMATH_GPT_fair_people_ratio_l2016_201693


namespace NUMINAMATH_GPT_find_second_number_l2016_201685

theorem find_second_number (x : ℝ) : 217 + x + 0.217 + 2.0017 = 221.2357 → x = 2.017 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l2016_201685


namespace NUMINAMATH_GPT_num_hens_in_caravan_l2016_201610

variable (H G C K : ℕ)  -- number of hens, goats, camels, keepers
variable (total_heads total_feet : ℕ)

-- Defining the conditions
def num_goats := 35
def num_camels := 6
def num_keepers := 10
def heads := H + G + C + K
def feet := 2 * H + 4 * G + 4 * C + 2 * K
def relation := feet = heads + 193

theorem num_hens_in_caravan :
  G = num_goats → C = num_camels → K = num_keepers → relation → 
  H = 60 :=
by 
  intros _ _ _ _
  sorry

end NUMINAMATH_GPT_num_hens_in_caravan_l2016_201610


namespace NUMINAMATH_GPT_condo_total_units_l2016_201601

-- Definitions from conditions
def total_floors := 23
def regular_units_per_floor := 12
def penthouse_units_per_floor := 2
def penthouse_floors := 2
def regular_floors := total_floors - penthouse_floors

-- Definition for total units
def total_units := (regular_floors * regular_units_per_floor) + (penthouse_floors * penthouse_units_per_floor)

-- Theorem statement: prove total units is 256
theorem condo_total_units : total_units = 256 :=
by
  sorry

end NUMINAMATH_GPT_condo_total_units_l2016_201601


namespace NUMINAMATH_GPT_sum_of_variables_l2016_201660

theorem sum_of_variables (x y z w : ℤ) 
(h1 : x - y + z = 7) 
(h2 : y - z + w = 8) 
(h3 : z - w + x = 4) 
(h4 : w - x + y = 3) : 
x + y + z + w = 11 := 
sorry

end NUMINAMATH_GPT_sum_of_variables_l2016_201660


namespace NUMINAMATH_GPT_intersection_is_expected_result_l2016_201657

def set_A : Set ℝ := { x | x * (x + 1) > 0 }
def set_B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 1) }
def expected_result : Set ℝ := { x | x ≥ 1 }

theorem intersection_is_expected_result : set_A ∩ set_B = expected_result := by
  sorry

end NUMINAMATH_GPT_intersection_is_expected_result_l2016_201657


namespace NUMINAMATH_GPT_custom_op_evaluation_l2016_201675

def custom_op (a b : ℤ) : ℤ := a * b - (a + b)

theorem custom_op_evaluation : custom_op 2 (-3) = -5 :=
by
sorry

end NUMINAMATH_GPT_custom_op_evaluation_l2016_201675


namespace NUMINAMATH_GPT_gcd_of_given_lcm_and_ratio_l2016_201655

theorem gcd_of_given_lcm_and_ratio (C D : ℕ) (h1 : Nat.lcm C D = 200) (h2 : C * 5 = D * 2) : Nat.gcd C D = 5 :=
sorry

end NUMINAMATH_GPT_gcd_of_given_lcm_and_ratio_l2016_201655


namespace NUMINAMATH_GPT_proposition_1_proposition_3_l2016_201683

variables {Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Condition predicates
def parallel (p q : Plane) : Prop := sorry -- parallelism of p and q
def perpendicular (p q : Plane) : Prop := sorry -- perpendicularly of p and q
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- parallelism of line and plane
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry -- perpendicularity of line and plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- line is in the plane

-- Proposition ①
theorem proposition_1 (h1 : parallel α β) (h2 : parallel α γ) : parallel β γ := sorry

-- Proposition ③
theorem proposition_3 (h1 : line_perpendicular_plane m α) (h2 : line_parallel_plane m β) : perpendicular α β := sorry

end NUMINAMATH_GPT_proposition_1_proposition_3_l2016_201683


namespace NUMINAMATH_GPT_mass_percentage_C_in_CuCO3_l2016_201625

def molar_mass_Cu := 63.546 -- g/mol
def molar_mass_C := 12.011 -- g/mol
def molar_mass_O := 15.999 -- g/mol
def molar_mass_CuCO3 := molar_mass_Cu + molar_mass_C + 3 * molar_mass_O

theorem mass_percentage_C_in_CuCO3 : 
  (molar_mass_C / molar_mass_CuCO3) * 100 = 9.72 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_C_in_CuCO3_l2016_201625


namespace NUMINAMATH_GPT_opponent_choice_is_random_l2016_201647

-- Define the possible outcomes in the game
inductive Outcome
| rock
| paper
| scissors

-- Defining the opponent's choice set
def opponent_choice := {outcome : Outcome | outcome = Outcome.rock ∨ outcome = Outcome.paper ∨ outcome = Outcome.scissors}

-- The event where the opponent chooses "scissors"
def event_opponent_chooses_scissors := Outcome.scissors ∈ opponent_choice

-- Proving that the event of opponent choosing "scissors" is a random event
theorem opponent_choice_is_random : ¬(∀outcome ∈ opponent_choice, outcome = Outcome.scissors) ∧ (∃ outcome ∈ opponent_choice, outcome = Outcome.scissors) → event_opponent_chooses_scissors := 
sorry

end NUMINAMATH_GPT_opponent_choice_is_random_l2016_201647


namespace NUMINAMATH_GPT_jordan_meets_emily_after_total_time_l2016_201628

noncomputable def meet_time
  (initial_distance : ℝ)
  (speed_ratio : ℝ)
  (decrease_rate : ℝ)
  (time_until_break : ℝ)
  (break_duration : ℝ)
  (total_meet_time : ℝ) : Prop :=
  initial_distance = 30 ∧
  speed_ratio = 2 ∧
  decrease_rate = 2 ∧
  time_until_break = 10 ∧
  break_duration = 5 ∧
  total_meet_time = 17

theorem jordan_meets_emily_after_total_time :
  meet_time 30 2 2 10 5 17 := 
by {
  -- The conditions directly state the requirements needed for the proof.
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ -- This line confirms that all inputs match the given conditions.
}

end NUMINAMATH_GPT_jordan_meets_emily_after_total_time_l2016_201628


namespace NUMINAMATH_GPT_find_sum_of_x_and_reciprocal_l2016_201626

theorem find_sum_of_x_and_reciprocal (x : ℝ) (hx_condition : x^3 + 1/x^3 = 110) : x + 1/x = 5 := 
sorry

end NUMINAMATH_GPT_find_sum_of_x_and_reciprocal_l2016_201626


namespace NUMINAMATH_GPT_geom_seq_prop_l2016_201662

variable (b : ℕ → ℝ) (r : ℝ) (s t : ℕ)
variable (h : s ≠ t)
variable (h1 : s > 0) (h2 : t > 0)
variable (h3 : b 1 = 1)
variable (h4 : ∀ n, b (n + 1) = b n * r)

theorem geom_seq_prop : s ≠ t → s > 0 → t > 0 → b 1 = 1 → (∀ n, b (n + 1) = b n * r) → (b t)^(s - 1) / (b s)^(t - 1) = 1 :=
by
  intros h h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_geom_seq_prop_l2016_201662


namespace NUMINAMATH_GPT_rooms_count_l2016_201656

theorem rooms_count (total_paintings : ℕ) (paintings_per_room : ℕ) (h1 : total_paintings = 32) (h2 : paintings_per_room = 8) : (total_paintings / paintings_per_room) = 4 := by
  sorry

end NUMINAMATH_GPT_rooms_count_l2016_201656


namespace NUMINAMATH_GPT_subset_m_values_l2016_201642

theorem subset_m_values
  {A B : Set ℝ}
  (hA : A = { x | x^2 + x - 6 = 0 })
  (hB : ∃ m, B = { x | m * x + 1 = 0 })
  (h_subset : ∀ {x}, x ∈ B → x ∈ A) :
  (∃ m, m = -1/2 ∨ m = 0 ∨ m = 1/3) :=
sorry

end NUMINAMATH_GPT_subset_m_values_l2016_201642


namespace NUMINAMATH_GPT_not_integer_fraction_l2016_201608

theorem not_integer_fraction (a b : ℤ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hrelprime : Nat.gcd a.natAbs b.natAbs = 1) : 
  ¬(∃ (k : ℤ), 2 * a * (a^2 + b^2) = k * (a^2 - b^2)) :=
  sorry

end NUMINAMATH_GPT_not_integer_fraction_l2016_201608


namespace NUMINAMATH_GPT_volume_related_to_area_l2016_201606

theorem volume_related_to_area (x y z : ℝ) 
  (bottom_area_eq : 3 * x * y = 3 * x * y)
  (front_area_eq : 2 * y * z = 2 * y * z)
  (side_area_eq : 3 * x * z = 3 * x * z) :
  (3 * x * y) * (2 * y * z) * (3 * x * z) = 18 * (x * y * z) ^ 2 := 
by sorry

end NUMINAMATH_GPT_volume_related_to_area_l2016_201606


namespace NUMINAMATH_GPT_minimum_boxes_needed_l2016_201612

theorem minimum_boxes_needed (small_box_capacity medium_box_capacity large_box_capacity : ℕ)
    (max_small_boxes max_medium_boxes max_large_boxes : ℕ)
    (total_dozens: ℕ) :
  small_box_capacity = 2 → 
  medium_box_capacity = 3 → 
  large_box_capacity = 4 → 
  max_small_boxes = 6 → 
  max_medium_boxes = 5 → 
  max_large_boxes = 4 → 
  total_dozens = 40 → 
  ∃ (small_boxes_needed medium_boxes_needed large_boxes_needed : ℕ), 
    small_boxes_needed = 5 ∧ 
    medium_boxes_needed = 5 ∧ 
    large_boxes_needed = 4 := 
by
  sorry

end NUMINAMATH_GPT_minimum_boxes_needed_l2016_201612


namespace NUMINAMATH_GPT_f_of_3_l2016_201645

theorem f_of_3 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 3 = 7 := 
sorry

end NUMINAMATH_GPT_f_of_3_l2016_201645


namespace NUMINAMATH_GPT_Annie_total_cookies_l2016_201696

theorem Annie_total_cookies :
  let monday_cookies := 5
  let tuesday_cookies := 2 * monday_cookies 
  let wednesday_cookies := 1.4 * tuesday_cookies
  monday_cookies + tuesday_cookies + wednesday_cookies = 29 :=
by
  sorry

end NUMINAMATH_GPT_Annie_total_cookies_l2016_201696


namespace NUMINAMATH_GPT_lines_positional_relationship_l2016_201605

-- Defining basic geometric entities and their properties
structure Line :=
  (a b : ℝ)
  (point_on_line : ∃ x, a * x + b = 0)

-- Defining skew lines (two lines that do not intersect and are not parallel)
def skew_lines (l1 l2 : Line) : Prop :=
  ¬(∀ x, l1.a * x + l1.b = l2.a * x + l2.b) ∧ ¬(l1.a = l2.a)

-- Defining intersecting lines
def intersect (l1 l2 : Line) : Prop :=
  ∃ x, l1.a * x + l1.b = l2.a * x + l2.b

-- Main theorem to prove
theorem lines_positional_relationship (l1 l2 k m : Line) 
  (hl1: intersect l1 k) (hl2: intersect l2 k) (hk: skew_lines l1 m) (hm: skew_lines l2 m) :
  (intersect l1 l2) ∨ (skew_lines l1 l2) :=
sorry

end NUMINAMATH_GPT_lines_positional_relationship_l2016_201605


namespace NUMINAMATH_GPT_certain_number_is_two_l2016_201607

variable (x : ℕ)  -- x is the certain number

-- Condition: Given that adding 6 incorrectly results in 8
axiom h1 : x + 6 = 8

-- The mathematically equivalent proof problem Lean statement
theorem certain_number_is_two : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_two_l2016_201607


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2016_201619

theorem arithmetic_sequence_sum : 
  ∀ (a : ℕ → ℝ) (d : ℝ), (a 1 = 2 ∨ a 1 = 8) → (a 2017 = 2 ∨ a 2017 = 8) → 
  (∀ n : ℕ, a (n + 1) = a n + d) →
  a 2 + a 1009 + a 2016 = 15 := 
by
  intro a d h1 h2017 ha
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2016_201619


namespace NUMINAMATH_GPT_minimum_n_for_factorable_polynomial_l2016_201677

theorem minimum_n_for_factorable_polynomial :
  ∃ n : ℤ, (∀ A B : ℤ, 5 * A = 48 → 5 * B + A = n) ∧
  (∀ k : ℤ, (∀ A B : ℤ, 5 * A * B = 48 → 5 * B + A = k) → n ≤ k) :=
by
  sorry

end NUMINAMATH_GPT_minimum_n_for_factorable_polynomial_l2016_201677


namespace NUMINAMATH_GPT_product_of_roots_l2016_201649

theorem product_of_roots :
  let a := 24
  let c := -216
  ∀ x : ℝ, (24 * x^2 + 36 * x - 216 = 0) → (c / a = -9) :=
by
  intros
  sorry

end NUMINAMATH_GPT_product_of_roots_l2016_201649


namespace NUMINAMATH_GPT_adams_father_total_amount_l2016_201632

noncomputable def annual_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

noncomputable def total_interest (annual_interest : ℝ) (years : ℝ) : ℝ :=
  annual_interest * years

noncomputable def total_amount (principal : ℝ) (total_interest : ℝ) : ℝ :=
  principal + total_interest

theorem adams_father_total_amount :
  let principal := 2000
  let rate := 0.08
  let years := 2.5
  let annualInterest := annual_interest principal rate
  let interest := total_interest annualInterest years
  let amount := total_amount principal interest
  amount = 2400 :=
by sorry

end NUMINAMATH_GPT_adams_father_total_amount_l2016_201632


namespace NUMINAMATH_GPT_complex_sum_l2016_201621

noncomputable def omega : ℂ := sorry
axiom omega_power_five : omega^5 = 1
axiom omega_not_one : omega ≠ 1

theorem complex_sum :
  (omega^20 + omega^25 + omega^30 + omega^35 + omega^40 + omega^45 + omega^50 + omega^55 + omega^60 + omega^65 + omega^70) = 11 :=
by
  sorry

end NUMINAMATH_GPT_complex_sum_l2016_201621


namespace NUMINAMATH_GPT_three_digit_number_l2016_201661

theorem three_digit_number (a b c : ℕ) (h1 : a * (b + c) = 33) (h2 : b * (a + c) = 40) : 
  100 * a + 10 * b + c = 347 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_l2016_201661


namespace NUMINAMATH_GPT_limonia_largest_none_providable_amount_l2016_201633

def is_achievable (n : ℕ) (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), x = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10)

theorem limonia_largest_none_providable_amount (n : ℕ) : 
  ∃ s, ¬ is_achievable n s ∧ (∀ t, t > s → is_achievable n t) ∧ s = 12 * n^2 + 14 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_limonia_largest_none_providable_amount_l2016_201633


namespace NUMINAMATH_GPT_pipe_length_l2016_201692

theorem pipe_length (L_short : ℕ) (hL_short : L_short = 59) : 
    L_short + 2 * L_short = 177 := by
  sorry

end NUMINAMATH_GPT_pipe_length_l2016_201692


namespace NUMINAMATH_GPT_regular_bike_wheels_eq_two_l2016_201617

-- Conditions
def regular_bikes : ℕ := 7
def childrens_bikes : ℕ := 11
def wheels_per_childrens_bike : ℕ := 4
def total_wheels_seen : ℕ := 58

-- Define the problem
theorem regular_bike_wheels_eq_two 
  (w : ℕ)
  (h1 : total_wheels_seen = regular_bikes * w + childrens_bikes * wheels_per_childrens_bike) :
  w = 2 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_regular_bike_wheels_eq_two_l2016_201617
