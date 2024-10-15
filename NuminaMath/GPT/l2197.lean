import Mathlib

namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2197_219732

variable (p q : Prop)

theorem necessary_but_not_sufficient (h : ¬p → q) (h1 : ¬ (q → ¬p)) : ¬q → p := 
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2197_219732


namespace NUMINAMATH_GPT_min_value_inequality_l2197_219769

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z ≥ 36) ∧ 
  ((1 / x + 4 / y + 9 / z = 36) ↔ (x = 1 / 6 ∧ y = 1 / 3 ∧ z = 1 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l2197_219769


namespace NUMINAMATH_GPT_kim_branch_marking_l2197_219712

theorem kim_branch_marking (L : ℝ) (rem_frac : ℝ) (third_piece : ℝ) (F : ℝ) :
  L = 3 ∧ rem_frac = 0.6 ∧ third_piece = 1 ∧ L * rem_frac = 1.8 → F = 1 / 15 :=
by sorry

end NUMINAMATH_GPT_kim_branch_marking_l2197_219712


namespace NUMINAMATH_GPT_find_a_l2197_219793

theorem find_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x = 0 ∧ x = 1) → a = -1 := by
  intro h
  obtain ⟨x, hx, rfl⟩ := h
  have H : 1^2 + a * 1 = 0 := hx
  linarith

end NUMINAMATH_GPT_find_a_l2197_219793


namespace NUMINAMATH_GPT_marge_funds_l2197_219718

theorem marge_funds (initial_winnings : ℕ)
    (tax_fraction : ℕ)
    (loan_fraction : ℕ)
    (savings_amount : ℕ)
    (investment_fraction : ℕ)
    (tax_paid leftover_for_loans savings_after_loans final_leftover final_leftover_after_investment : ℕ) :
    initial_winnings = 12006 →
    tax_fraction = 2 →
    leftover_for_loans = initial_winnings / tax_fraction →
    loan_fraction = 3 →
    savings_after_loans = leftover_for_loans / loan_fraction →
    savings_amount = 1000 →
    final_leftover = leftover_for_loans - savings_after_loans - savings_amount →
    investment_fraction = 5 →
    final_leftover_after_investment = final_leftover - (savings_amount / investment_fraction) →
    final_leftover_after_investment = 2802 :=
by
  intros
  sorry

end NUMINAMATH_GPT_marge_funds_l2197_219718


namespace NUMINAMATH_GPT_constant_term_zero_l2197_219792

theorem constant_term_zero (h1 : x^2 + x = 0)
                          (h2 : 2*x^2 - x - 12 = 0)
                          (h3 : 2*(x^2 - 1) = 3*(x - 1))
                          (h4 : 2*(x^2 + 1) = x + 4) :
                          (∃ (c : ℤ), c = 0 ∧ (c = 0 ∨ c = -12 ∨ c = 1 ∨ c = -2) → c = 0) :=
sorry

end NUMINAMATH_GPT_constant_term_zero_l2197_219792


namespace NUMINAMATH_GPT_parabola_focus_l2197_219791

-- Define the parabola
def parabolaEquation (x y : ℝ) : Prop := y^2 = -6 * x

-- Define the focus
def focus (x y : ℝ) : Prop := x = -3 / 2 ∧ y = 0

-- The proof problem: showing the focus of the given parabola
theorem parabola_focus : ∃ x y : ℝ, parabolaEquation x y ∧ focus x y :=
by
    sorry

end NUMINAMATH_GPT_parabola_focus_l2197_219791


namespace NUMINAMATH_GPT_great_dane_weight_l2197_219705

def weight_problem (C P G : ℝ) : Prop :=
  (P = 3 * C) ∧ (G = 3 * P + 10) ∧ (C + P + G = 439)

theorem great_dane_weight : ∃ (C P G : ℝ), weight_problem C P G ∧ G = 307 :=
by
  sorry

end NUMINAMATH_GPT_great_dane_weight_l2197_219705


namespace NUMINAMATH_GPT_solve_x_l2197_219710

theorem solve_x : ∃ x : ℝ, 2^(Real.log 5 / Real.log 2) = 3 * x + 4 ∧ x = 1 / 3 :=
by
  use 1 / 3
  sorry

end NUMINAMATH_GPT_solve_x_l2197_219710


namespace NUMINAMATH_GPT_people_in_each_playgroup_l2197_219741

theorem people_in_each_playgroup (girls boys parents playgroups : ℕ) (hg : girls = 14) (hb : boys = 11) (hp : parents = 50) (hpg : playgroups = 3) :
  (girls + boys + parents) / playgroups = 25 := by
  sorry

end NUMINAMATH_GPT_people_in_each_playgroup_l2197_219741


namespace NUMINAMATH_GPT_ratio_of_areas_is_16_l2197_219703

-- Definitions and conditions
variables (a b : ℝ)

-- Given condition: Perimeter of the larger square is 4 times the perimeter of the smaller square
def perimeter_relation (ha : a = 4 * b) : Prop := a = 4 * b

-- Theorem to prove: Ratio of the area of the larger square to the area of the smaller square is 16
theorem ratio_of_areas_is_16 (ha : a = 4 * b) : (a^2 / b^2) = 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_is_16_l2197_219703


namespace NUMINAMATH_GPT_wooden_block_length_l2197_219771

-- Define the problem conditions
def meters_to_centimeters (m : ℕ) : ℕ := m * 100
def additional_length_cm (length_cm : ℕ) (additional_cm : ℕ) : ℕ := length_cm + additional_cm

-- Formalization of the problem
theorem wooden_block_length :
  let length_in_meters := 31
  let additional_cm := 30
  additional_length_cm (meters_to_centimeters length_in_meters) additional_cm = 3130 :=
by
  sorry

end NUMINAMATH_GPT_wooden_block_length_l2197_219771


namespace NUMINAMATH_GPT_number_of_distinct_stackings_l2197_219752

-- Defining the conditions
def cubes : ℕ := 8
def edge_length : ℕ := 1
def valid_stackings (n : ℕ) : Prop := 
  n = 8 -- Stating that we are working with 8 cubes

-- The theorem stating the problem and expected solution
theorem number_of_distinct_stackings : 
  cubes = 8 ∧ edge_length = 1 ∧ valid_stackings cubes → ∃ (count : ℕ), count = 10 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_distinct_stackings_l2197_219752


namespace NUMINAMATH_GPT_inequality_of_abc_l2197_219744

variable (a b c : ℝ)

theorem inequality_of_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c ≥ a * b * c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c :=
sorry

end NUMINAMATH_GPT_inequality_of_abc_l2197_219744


namespace NUMINAMATH_GPT_f_m_minus_1_pos_l2197_219776

variable {R : Type*} [LinearOrderedField R]

def quadratic_function (x a : R) : R :=
  x^2 - x + a

theorem f_m_minus_1_pos {a m : R} (h_pos : 0 < a) (h_fm : quadratic_function m a < 0) :
  quadratic_function (m - 1 : R) a > 0 :=
sorry

end NUMINAMATH_GPT_f_m_minus_1_pos_l2197_219776


namespace NUMINAMATH_GPT_ratio_of_red_to_blue_beads_l2197_219760

theorem ratio_of_red_to_blue_beads (red_beads blue_beads : ℕ) (h1 : red_beads = 30) (h2 : blue_beads = 20) :
    (red_beads / Nat.gcd red_beads blue_beads) = 3 ∧ (blue_beads / Nat.gcd red_beads blue_beads) = 2 := 
by 
    -- Proof will go here
    sorry

end NUMINAMATH_GPT_ratio_of_red_to_blue_beads_l2197_219760


namespace NUMINAMATH_GPT_compound_proposition_l2197_219755

theorem compound_proposition (Sn P Q : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → Sn n = 2 * n^2 + 3 * n + 1) →
  (∀ n : ℕ, n > 0 → Sn n = 2 * P n + 1) →
  (¬(∀ n, n > 0 → ∃ d, (P (n + 1) - P n) = d)) ∧ (∀ n, n > 0 → P n = Q (n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_compound_proposition_l2197_219755


namespace NUMINAMATH_GPT_part1_part2_l2197_219759

namespace MathProofProblem

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part1 (x : ℝ) : f 2 * x ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 := 
by
  sorry

theorem part2 (a b : ℝ) (h₀ : a + b = 2) : f (a ^ 2) + f (b ^ 2) = 2 :=
by
  sorry

end MathProofProblem

end NUMINAMATH_GPT_part1_part2_l2197_219759


namespace NUMINAMATH_GPT_area_of_square_II_l2197_219713

theorem area_of_square_II {a b : ℝ} (h : a > b) (d : ℝ) (h1 : d = a - b)
    (A1_A : ℝ) (h2 : A1_A = (a - b)^2 / 2) (A2_A : ℝ) (h3 : A2_A = 3 * A1_A) :
  A2_A = 3 * (a - b)^2 / 2 := by
  sorry

end NUMINAMATH_GPT_area_of_square_II_l2197_219713


namespace NUMINAMATH_GPT_part_I_part_II_l2197_219721

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
x^2 / a^2 + y^2 / b^2 = 1

theorem part_I (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (eccentricity : ℝ := c / a) (h3 : eccentricity = Real.sqrt 2 / 2) (vertex : ℝ × ℝ := (0, 1)) (h4 : vertex = (0, b)) 
  : ellipse_equation (Real.sqrt 2) 1 (0:ℝ) 1 :=
sorry

theorem part_II (a b k : ℝ) (x y : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 1)
  (line_eq : ℝ → ℝ := fun x => k * x + 1) 
  (h3 : (1 + 2 * k^2) * x^2 + 4 * k * x = 0) 
  (distance_AB : ℝ := Real.sqrt 2 * 4 / 3) 
  (h4 : Real.sqrt (1 + k^2) * abs ((-4 * k) / (2 * k^2 + 1)) = distance_AB) 
  : (x, y) = (4/3, -1/3) ∨ (x, y) = (-4/3, -1/3) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l2197_219721


namespace NUMINAMATH_GPT_factorize_polynomial_l2197_219714

theorem factorize_polynomial (x : ℝ) :
  x^4 + 2 * x^3 - 9 * x^2 - 2 * x + 8 = (x + 4) * (x - 2) * (x + 1) * (x - 1) :=
sorry

end NUMINAMATH_GPT_factorize_polynomial_l2197_219714


namespace NUMINAMATH_GPT_shuai_shuai_total_words_l2197_219742

-- Conditions
def words (a : ℕ) (n : ℕ) : ℕ := a + n

-- Total words memorized in 7 days
def total_memorized (a : ℕ) : ℕ := 
  (words a 0) + (words a 1) + (words a 2) + (words a 3) + (words a 4) + (words a 5) + (words a 6)

-- Condition: Sum of words memorized in the first 4 days equals sum of words in the last 3 days
def condition (a : ℕ) : Prop := 
  (words a 0) + (words a 1) + (words a 2) + (words a 3) = (words a 4) + (words a 5) + (words a 6)

-- Theorem: If condition is satisfied, then the total number of words memorized is 84.
theorem shuai_shuai_total_words : 
  ∀ a : ℕ, condition a → total_memorized a = 84 :=
by
  intro a h
  sorry

end NUMINAMATH_GPT_shuai_shuai_total_words_l2197_219742


namespace NUMINAMATH_GPT_sheep_remain_l2197_219779

theorem sheep_remain : ∀ (total_sheep sister_share brother_share : ℕ),
  total_sheep = 400 →
  sister_share = total_sheep / 4 →
  brother_share = (total_sheep - sister_share) / 2 →
  (total_sheep - sister_share - brother_share) = 150 :=
by
  intros total_sheep sister_share brother_share h_total h_sister h_brother
  rw [h_total, h_sister, h_brother]
  sorry

end NUMINAMATH_GPT_sheep_remain_l2197_219779


namespace NUMINAMATH_GPT_reduced_price_l2197_219731

-- Definitions based on given conditions
def original_price (P : ℝ) : Prop := P > 0

def condition1 (P X : ℝ) : Prop := P * X = 700

def condition2 (P X : ℝ) : Prop := 0.7 * P * (X + 3) = 700

-- Main theorem to prove the reduced price per kg is 70
theorem reduced_price (P X : ℝ) (h1 : original_price P) (h2 : condition1 P X) (h3 : condition2 P X) : 
  0.7 * P = 70 := sorry

end NUMINAMATH_GPT_reduced_price_l2197_219731


namespace NUMINAMATH_GPT_sum_of_remainders_mod_53_l2197_219751

theorem sum_of_remainders_mod_53 (x y z : ℕ) (hx : x % 53 = 36) (hy : y % 53 = 15) (hz : z % 53 = 7) : 
  (x + y + z) % 53 = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_53_l2197_219751


namespace NUMINAMATH_GPT_periodic_sequence_not_constant_l2197_219727

theorem periodic_sequence_not_constant :
  ∃ (x : ℕ → ℤ), (∀ n : ℕ, x (n+1) = 2 * x n + 3 * x (n-1)) ∧ (∃ T > 0, ∀ n : ℕ, x (n+T) = x n) ∧ (∃ n m : ℕ, n ≠ m ∧ x n ≠ x m) :=
sorry

end NUMINAMATH_GPT_periodic_sequence_not_constant_l2197_219727


namespace NUMINAMATH_GPT_no_valid_m_l2197_219799

theorem no_valid_m
  (m : ℕ)
  (hm : m > 0)
  (h1 : ∃ k1 : ℕ, k1 > 0 ∧ 1806 = k1 * (m^2 - 2))
  (h2 : ∃ k2 : ℕ, k2 > 0 ∧ 1806 = k2 * (m^2 + 2)) :
  false :=
sorry

end NUMINAMATH_GPT_no_valid_m_l2197_219799


namespace NUMINAMATH_GPT_minimal_period_of_sum_l2197_219722

theorem minimal_period_of_sum (A B : ℝ)
  (hA : ∃ p : ℕ, p = 6 ∧ (∃ (x : ℝ) (l : ℕ), A = x / (10 ^ l * (10 ^ p - 1))))
  (hB : ∃ p : ℕ, p = 12 ∧ (∃ (y : ℝ) (m : ℕ), B = y / (10 ^ m * (10 ^ p - 1)))) :
  ∃ p : ℕ, p = 12 ∧ (∃ (z : ℝ) (n : ℕ), A + B = z / (10 ^ n * (10 ^ p - 1))) :=
sorry

end NUMINAMATH_GPT_minimal_period_of_sum_l2197_219722


namespace NUMINAMATH_GPT_distance_walked_hazel_l2197_219701

theorem distance_walked_hazel (x : ℝ) (h : x + 2 * x = 6) : x = 2 :=
sorry

end NUMINAMATH_GPT_distance_walked_hazel_l2197_219701


namespace NUMINAMATH_GPT_height_difference_of_packings_l2197_219724

theorem height_difference_of_packings :
  (let d := 12
   let n := 180
   let rowsA := n / 10
   let heightA := rowsA * d
   let height_of_hex_gap := (6 * Real.sqrt 3 : ℝ)
   let gaps := rowsA - 1
   let heightB := gaps * height_of_hex_gap + 2 * (d / 2)
   heightA - heightB) = 204 - 102 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_height_difference_of_packings_l2197_219724


namespace NUMINAMATH_GPT_calculate_expression_l2197_219739

variables {a b c : ℤ}
variable (h1 : 5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c) -- a, b, c are multiples of 5
variable (h2 : a < b ∧ b < c) -- a < b < c
variable (h3 : c = a + 10) -- c = a + 10

theorem calculate_expression :
  (a - b) * (a - c) / (b - c) = -10 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2197_219739


namespace NUMINAMATH_GPT_simplify_expression_l2197_219778

theorem simplify_expression (a b c d : ℝ) (h₁ : a + b + c + d = 0) (h₂ : a ≠ 0) (h₃ : b ≠ 0) (h₄ : c ≠ 0) (h₅ : d ≠ 0) :
  (1 / (b^2 + c^2 + d^2 - a^2) + 
   1 / (a^2 + c^2 + d^2 - b^2) + 
   1 / (a^2 + b^2 + d^2 - c^2) + 
   1 / (a^2 + b^2 + c^2 - d^2)) = 4 / d^2 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l2197_219778


namespace NUMINAMATH_GPT_rectangle_same_color_exists_l2197_219733

theorem rectangle_same_color_exists (color : ℝ × ℝ → Prop) (red blue : Prop) (h : ∀ p : ℝ × ℝ, color p = red ∨ color p = blue) :
  ∃ (a b c d : ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (color a = color b ∧ color b = color c ∧ color c = color d) :=
sorry

end NUMINAMATH_GPT_rectangle_same_color_exists_l2197_219733


namespace NUMINAMATH_GPT_total_people_counted_l2197_219702

-- Definitions based on conditions
def people_second_day : ℕ := 500
def people_first_day : ℕ := 2 * people_second_day

-- Theorem statement
theorem total_people_counted : people_first_day + people_second_day = 1500 := 
by 
  sorry

end NUMINAMATH_GPT_total_people_counted_l2197_219702


namespace NUMINAMATH_GPT_min_value_sum_reciprocal_l2197_219749

open Real

theorem min_value_sum_reciprocal (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
    (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
    1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_sum_reciprocal_l2197_219749


namespace NUMINAMATH_GPT_arithmetic_progression_first_three_terms_l2197_219735

theorem arithmetic_progression_first_three_terms 
  (S_n : ℤ) (d a_1 a_2 a_3 a_5 : ℤ)
  (h1 : S_n = 112) 
  (h2 : (a_1 + d) * d = 30)
  (h3 : (a_1 + 2 * d) + (a_1 + 4 * d) = 32) 
  (h4 : ∀ (n : ℕ), S_n = (n * (2 * a_1 + (n - 1) * d)) / 2) : 
  ((a_1 = 7 ∧ a_2 = 10 ∧ a_3 = 13) ∨ (a_1 = 1 ∧ a_2 = 6 ∧ a_3 = 11)) :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_first_three_terms_l2197_219735


namespace NUMINAMATH_GPT_cos_A_value_l2197_219770

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
-- a, b, c are the sides opposite to angles A, B, and C respectively.
-- Assumption 1: b - c = (1/4) * a
def condition1 := b - c = (1/4) * a
-- Assumption 2: 2 * sin B = 3 * sin C
def condition2 := 2 * Real.sin B = 3 * Real.sin C

-- The theorem statement: Under these conditions, prove that cos A = -1/4.
theorem cos_A_value (h1 : condition1 a b c) (h2 : condition2 B C) : 
    Real.cos A = -1/4 :=
sorry -- placeholder for the proof

end NUMINAMATH_GPT_cos_A_value_l2197_219770


namespace NUMINAMATH_GPT_negation_example_l2197_219734

theorem negation_example :
  (¬ (∀ x : ℝ, abs (x - 2) + abs (x - 4) > 3)) ↔ (∃ x : ℝ, abs (x - 2) + abs (x - 4) ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l2197_219734


namespace NUMINAMATH_GPT_focus_of_parabola_l2197_219782

theorem focus_of_parabola (x y : ℝ) (h : y^2 + 4 * x = 0) : (x, y) = (-1, 0) := sorry

end NUMINAMATH_GPT_focus_of_parabola_l2197_219782


namespace NUMINAMATH_GPT_least_possible_value_of_x_minus_y_plus_z_l2197_219743

theorem least_possible_value_of_x_minus_y_plus_z : 
  ∃ (x y z : ℕ), 3 * x = 4 * y ∧ 4 * y = 7 * z ∧ x - y + z = 19 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_value_of_x_minus_y_plus_z_l2197_219743


namespace NUMINAMATH_GPT_tangent_point_at_slope_one_l2197_219772

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem proof problem
theorem tangent_point_at_slope_one : ∃ x : ℝ, derivative x = 1 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_point_at_slope_one_l2197_219772


namespace NUMINAMATH_GPT_total_books_l2197_219764

variable (a : ℕ)

theorem total_books (h₁ : 5 = 5) (h₂ : a = a) : 5 + a = 5 + a :=
by
  sorry

end NUMINAMATH_GPT_total_books_l2197_219764


namespace NUMINAMATH_GPT_at_least_one_wins_l2197_219728

def probability_A := 1 / 2
def probability_B := 1 / 4

def probability_at_least_one (pA pB : ℚ) : ℚ := 
  1 - ((1 - pA) * (1 - pB))

theorem at_least_one_wins :
  probability_at_least_one probability_A probability_B = 5 / 8 := 
by
  sorry

end NUMINAMATH_GPT_at_least_one_wins_l2197_219728


namespace NUMINAMATH_GPT_charley_pencils_final_count_l2197_219747

def charley_initial_pencils := 50
def lost_pencils_while_moving := 8
def misplaced_fraction_first_week := 1 / 3
def lost_fraction_second_week := 1 / 4

theorem charley_pencils_final_count:
  let initial := charley_initial_pencils
  let after_moving := initial - lost_pencils_while_moving
  let misplaced_first_week := misplaced_fraction_first_week * after_moving
  let remaining_after_first_week := after_moving - misplaced_first_week
  let lost_second_week := lost_fraction_second_week * remaining_after_first_week
  let final_pencils := remaining_after_first_week - lost_second_week
  final_pencils = 21 := 
sorry

end NUMINAMATH_GPT_charley_pencils_final_count_l2197_219747


namespace NUMINAMATH_GPT_min_students_in_class_l2197_219729

-- Define the conditions
variables (b g : ℕ) -- number of boys and girls
variable (h1 : 3 * b = 4 * (2 * g)) -- Equal number of boys and girls passed the test

-- Define the desired minimum number of students
def min_students : ℕ := 17

-- The theorem which asserts that the total number of students in the class is at least 17
theorem min_students_in_class (b g : ℕ) (h1 : 3 * b = 4 * (2 * g)) : (b + g) ≥ min_students := 
sorry

end NUMINAMATH_GPT_min_students_in_class_l2197_219729


namespace NUMINAMATH_GPT_boys_count_at_table_l2197_219748

-- Definitions from conditions
def children_count : ℕ := 13
def alternates (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- The problem to be proven in Lean:
theorem boys_count_at_table : ∃ b g : ℕ, b + g = children_count ∧ alternates b ∧ alternates g ∧ b = 7 :=
by
  sorry

end NUMINAMATH_GPT_boys_count_at_table_l2197_219748


namespace NUMINAMATH_GPT_increase_by_percentage_l2197_219750

def initial_value : ℕ := 550
def percentage_increase : ℚ := 0.35
def final_value : ℚ := 742.5

theorem increase_by_percentage :
  (initial_value : ℚ) * (1 + percentage_increase) = final_value := by
  sorry

end NUMINAMATH_GPT_increase_by_percentage_l2197_219750


namespace NUMINAMATH_GPT_least_product_of_primes_gt_30_l2197_219797

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end NUMINAMATH_GPT_least_product_of_primes_gt_30_l2197_219797


namespace NUMINAMATH_GPT_smallest_n_l2197_219717

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end NUMINAMATH_GPT_smallest_n_l2197_219717


namespace NUMINAMATH_GPT_desired_depth_proof_l2197_219784

-- Definitions based on the conditions in Step a)
def initial_men : ℕ := 9
def initial_hours : ℕ := 8
def initial_depth : ℕ := 30
def extra_men : ℕ := 11
def total_men : ℕ := initial_men + extra_men
def new_hours : ℕ := 6

-- Total man-hours for initial setup
def initial_man_hours (days : ℕ) : ℕ := initial_men * initial_hours * days

-- Total man-hours for new setup to achieve desired depth
def new_man_hours (desired_depth : ℕ) (days : ℕ) : ℕ := total_men * new_hours * days

-- Proportional relationship between initial setup and desired depth
theorem desired_depth_proof (days : ℕ) (desired_depth : ℕ) :
  initial_man_hours days / initial_depth = new_man_hours desired_depth days / desired_depth → desired_depth = 18 :=
by
  sorry

end NUMINAMATH_GPT_desired_depth_proof_l2197_219784


namespace NUMINAMATH_GPT_prob_lamp_first_factory_standard_prob_lamp_standard_l2197_219715

noncomputable def P_B1 : ℝ := 0.35
noncomputable def P_B2 : ℝ := 0.50
noncomputable def P_B3 : ℝ := 0.15

noncomputable def P_B1_A : ℝ := 0.70
noncomputable def P_B2_A : ℝ := 0.80
noncomputable def P_B3_A : ℝ := 0.90

-- Question A
theorem prob_lamp_first_factory_standard : P_B1 * P_B1_A = 0.245 :=
by 
  sorry

-- Question B
theorem prob_lamp_standard : (P_B1 * P_B1_A) + (P_B2 * P_B2_A) + (P_B3 * P_B3_A) = 0.78 :=
by 
  sorry

end NUMINAMATH_GPT_prob_lamp_first_factory_standard_prob_lamp_standard_l2197_219715


namespace NUMINAMATH_GPT_total_earrings_l2197_219757

-- Definitions based on the given conditions
def bella_earrings : ℕ := 10
def monica_earrings : ℕ := 4 * bella_earrings
def rachel_earrings : ℕ := monica_earrings / 2
def olivia_earrings : ℕ := bella_earrings + monica_earrings + rachel_earrings + 5

-- The theorem to prove the total number of earrings
theorem total_earrings : bella_earrings + monica_earrings + rachel_earrings + olivia_earrings = 145 := by
  sorry

end NUMINAMATH_GPT_total_earrings_l2197_219757


namespace NUMINAMATH_GPT_Nina_second_distance_l2197_219725

theorem Nina_second_distance 
  (total_distance : ℝ) 
  (first_run : ℝ) 
  (second_same_run : ℝ)
  (run_twice : first_run = 0.08 ∧ second_same_run = 0.08)
  (total : total_distance = 0.83)
  : (total_distance - (first_run + second_same_run)) = 0.67 := by
  sorry

end NUMINAMATH_GPT_Nina_second_distance_l2197_219725


namespace NUMINAMATH_GPT_base8_to_base10_sum_l2197_219756

theorem base8_to_base10_sum (a b : ℕ) (h₁ : a = 1 * 8^3 + 4 * 8^2 + 5 * 8^1 + 3 * 8^0)
                            (h₂ : b = 5 * 8^2 + 6 * 8^1 + 7 * 8^0) :
                            ((a + b) = 2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0) →
                            (2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0 = 1124) :=
by {
  sorry
}

end NUMINAMATH_GPT_base8_to_base10_sum_l2197_219756


namespace NUMINAMATH_GPT_inequality_solution_l2197_219794

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x^2 + 1) + x) - (2 / (Real.exp x + 1))

theorem inequality_solution :
  { x : ℝ | f x + f (2 * x - 1) > -2 } = { x : ℝ | x > 1 / 3 } :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2197_219794


namespace NUMINAMATH_GPT_determine_f_zero_l2197_219783

variable (f : ℝ → ℝ)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 + y) + 4 * (f x) * y

theorem determine_f_zero (h1: functional_equation f)
    (h2 : f 2 = 4) : f 0 = 0 := 
sorry

end NUMINAMATH_GPT_determine_f_zero_l2197_219783


namespace NUMINAMATH_GPT_find_number_of_girls_l2197_219709

-- Definitions
variables (B G : ℕ)
variables (total children_holding_boys_hand children_holding_girls_hand : ℕ)
variables (children_counted_twice : ℕ)

-- Conditions
axiom cond1 : B + G = 40
axiom cond2 : children_holding_boys_hand = 22
axiom cond3 : children_holding_girls_hand = 30
axiom cond4 : total = 40

-- Goal
theorem find_number_of_girls (h : children_counted_twice = children_holding_boys_hand + children_holding_girls_hand - total) :
  G = 24 :=
sorry

end NUMINAMATH_GPT_find_number_of_girls_l2197_219709


namespace NUMINAMATH_GPT_minValueExpr_ge_9_l2197_219737

noncomputable def minValueExpr (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minValueExpr_ge_9 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  minValueExpr x y z ≥ 9 :=
by sorry

end NUMINAMATH_GPT_minValueExpr_ge_9_l2197_219737


namespace NUMINAMATH_GPT_airplane_shot_down_l2197_219766

def P_A : ℝ := 0.4
def P_B : ℝ := 0.5
def P_C : ℝ := 0.8

def P_one_hit : ℝ := 0.4
def P_two_hit : ℝ := 0.7
def P_three_hit : ℝ := 1

def P_one : ℝ := (P_A * (1 - P_B) * (1 - P_C)) + ((1 - P_A) * P_B * (1 - P_C)) + ((1 - P_A) * (1 - P_B) * P_C)
def P_two : ℝ := (P_A * P_B * (1 - P_C)) + (P_A * (1 - P_B) * P_C) + ((1 - P_A) * P_B * P_C)
def P_three : ℝ := P_A * P_B * P_C

def total_probability := (P_one * P_one_hit) + (P_two * P_two_hit) + (P_three * P_three_hit)

theorem airplane_shot_down : total_probability = 0.604 := by
  sorry

end NUMINAMATH_GPT_airplane_shot_down_l2197_219766


namespace NUMINAMATH_GPT_exists_two_same_remainder_l2197_219786

theorem exists_two_same_remainder (n : ℤ) (a : ℕ → ℤ) :
  ∃ i j : ℕ, i ≠ j ∧ 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n ∧ (a i % n = a j % n) := sorry

end NUMINAMATH_GPT_exists_two_same_remainder_l2197_219786


namespace NUMINAMATH_GPT_integer_values_of_f_l2197_219746

noncomputable def f (x : ℝ) : ℝ := (1 + x)^(1/3) + (3 - x)^(1/3)

theorem integer_values_of_f : 
  {x : ℝ | ∃ k : ℤ, f x = k} = {1 + Real.sqrt 5, 1 - Real.sqrt 5, 1 + (10/9) * Real.sqrt 3, 1 - (10/9) * Real.sqrt 3} :=
by
  sorry

end NUMINAMATH_GPT_integer_values_of_f_l2197_219746


namespace NUMINAMATH_GPT_solve_congruence_l2197_219736

theorem solve_congruence (n : ℤ) : 15 * n ≡ 9 [ZMOD 47] → n ≡ 18 [ZMOD 47] :=
by
  sorry

end NUMINAMATH_GPT_solve_congruence_l2197_219736


namespace NUMINAMATH_GPT_sum_first_32_terms_bn_l2197_219763

noncomputable def a_n (n : ℕ) : ℝ := 3 * n + 1

noncomputable def b_n (n : ℕ) : ℝ :=
  1 / ((a_n n) * Real.sqrt (a_n (n + 1)) + (a_n (n + 1)) * Real.sqrt (a_n n))

noncomputable def sum_bn (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) b_n

theorem sum_first_32_terms_bn : sum_bn 32 = 2 / 15 := 
sorry

end NUMINAMATH_GPT_sum_first_32_terms_bn_l2197_219763


namespace NUMINAMATH_GPT_compare_minus_abs_val_l2197_219719

theorem compare_minus_abs_val :
  -|(-8)| < -6 := 
sorry

end NUMINAMATH_GPT_compare_minus_abs_val_l2197_219719


namespace NUMINAMATH_GPT_blanket_cost_l2197_219706

theorem blanket_cost (x : ℝ) 
    (h₁ : 200 + 750 + 2 * x = 1350) 
    (h₂ : 2 + 5 + 2 = 9) 
    (h₃ : (200 + 750 + 2 * x) / 9 = 150) : 
    x = 200 :=
by
    have h_total : 200 + 750 + 2 * x = 1350 := h₁
    have h_avg : (200 + 750 + 2 * x) / 9 = 150 := h₃
    sorry

end NUMINAMATH_GPT_blanket_cost_l2197_219706


namespace NUMINAMATH_GPT_roots_quadratic_l2197_219768

theorem roots_quadratic (a b : ℝ) 
  (h1: a^2 + 3 * a - 2010 = 0) 
  (h2: b^2 + 3 * b - 2010 = 0)
  (h_roots: a + b = -3 ∧ a * b = -2010):
  a^2 - a - 4 * b = 2022 :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_l2197_219768


namespace NUMINAMATH_GPT_polygon_sides_l2197_219711

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end NUMINAMATH_GPT_polygon_sides_l2197_219711


namespace NUMINAMATH_GPT_total_area_of_figure_l2197_219788

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

def side_length_of_square (d : ℝ) : ℝ := d

def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def total_area (d : ℝ) : ℝ := area_of_square d + area_of_circle (radius_of_circle d)

theorem total_area_of_figure (d : ℝ) (h : d = 6) : total_area d = 36 + 9 * Real.pi :=
by
  -- skipping proof with sorry
  sorry

end NUMINAMATH_GPT_total_area_of_figure_l2197_219788


namespace NUMINAMATH_GPT_external_angle_bisector_proof_l2197_219790

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end NUMINAMATH_GPT_external_angle_bisector_proof_l2197_219790


namespace NUMINAMATH_GPT_daughter_age_l2197_219781

theorem daughter_age (m d : ℕ) (h1 : m + d = 60) (h2 : m - 10 = 7 * (d - 10)) : d = 15 :=
sorry

end NUMINAMATH_GPT_daughter_age_l2197_219781


namespace NUMINAMATH_GPT_find_integers_10_le_n_le_20_mod_7_l2197_219796

theorem find_integers_10_le_n_le_20_mod_7 :
  ∃ n, (10 ≤ n ∧ n ≤ 20 ∧ n % 7 = 4) ∧
  (n = 11 ∨ n = 18) := by
  sorry

end NUMINAMATH_GPT_find_integers_10_le_n_le_20_mod_7_l2197_219796


namespace NUMINAMATH_GPT_sara_total_quarters_l2197_219754

def initial_quarters : ℝ := 783.0
def given_quarters : ℝ := 271.0

theorem sara_total_quarters : initial_quarters + given_quarters = 1054.0 := 
by
  sorry

end NUMINAMATH_GPT_sara_total_quarters_l2197_219754


namespace NUMINAMATH_GPT_isosceles_right_triangle_leg_length_l2197_219795

theorem isosceles_right_triangle_leg_length (H : Real)
  (median_to_hypotenuse_is_half : ∀ H, (H / 2) = 12) :
  (H / Real.sqrt 2) = 12 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_leg_length_l2197_219795


namespace NUMINAMATH_GPT_least_product_of_distinct_primes_greater_than_50_l2197_219740

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  p ≠ q ∧ is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 : 
  ∃ p q, distinct_primes_greater_than_50 p q ∧ p * q = 3127 := 
sorry

end NUMINAMATH_GPT_least_product_of_distinct_primes_greater_than_50_l2197_219740


namespace NUMINAMATH_GPT_find_angle_A_l2197_219798

theorem find_angle_A (a b c A : ℝ) (h1 : b = c) (h2 : a^2 = 2 * b^2 * (1 - Real.sin A)) : 
  A = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l2197_219798


namespace NUMINAMATH_GPT_fraction_arithmetic_l2197_219762

theorem fraction_arithmetic : ( (4 / 5 - 1 / 10) / (2 / 5) ) = 7 / 4 :=
  sorry

end NUMINAMATH_GPT_fraction_arithmetic_l2197_219762


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l2197_219785

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 3 / 5) : x / y = 16 / 15 :=
sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l2197_219785


namespace NUMINAMATH_GPT_number_of_smoothies_l2197_219789

-- Definitions of the given conditions
def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def total_cost : ℕ := 17

-- Statement of the proof problem
theorem number_of_smoothies (S : ℕ) : burger_cost + sandwich_cost + S * smoothie_cost = total_cost → S = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_smoothies_l2197_219789


namespace NUMINAMATH_GPT_a10_plus_b10_l2197_219765

noncomputable def a : ℝ := sorry -- a will be a real number satisfying the conditions
noncomputable def b : ℝ := sorry -- b will be a real number satisfying the conditions

axiom ab_condition1 : a + b = 1
axiom ab_condition2 : a^2 + b^2 = 3
axiom ab_condition3 : a^3 + b^3 = 4
axiom ab_condition4 : a^4 + b^4 = 7
axiom ab_condition5 : a^5 + b^5 = 11

theorem a10_plus_b10 : a^10 + b^10 = 123 :=
by 
  sorry

end NUMINAMATH_GPT_a10_plus_b10_l2197_219765


namespace NUMINAMATH_GPT_martys_journey_length_l2197_219723

theorem martys_journey_length (x : ℝ) (h1 : x / 4 + 30 + x / 3 = x) : x = 72 :=
sorry

end NUMINAMATH_GPT_martys_journey_length_l2197_219723


namespace NUMINAMATH_GPT_simplify_expression_l2197_219704

theorem simplify_expression (w : ℝ) : 3 * w^2 + 6 * w^2 + 9 * w^2 + 12 * w^2 + 15 * w^2 + 24 = 45 * w^2 + 24 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2197_219704


namespace NUMINAMATH_GPT_marble_distribution_l2197_219761

theorem marble_distribution (x : ℚ) :
    (2 * x + 2) + (3 * x) + (x + 4) = 56 ↔ x = 25 / 3 := by
  sorry

end NUMINAMATH_GPT_marble_distribution_l2197_219761


namespace NUMINAMATH_GPT_range_of_a_l2197_219774

-- Given conditions
def p (x : ℝ) : Prop := abs (4 - x) ≤ 6
def q (x : ℝ) (a : ℝ) : Prop := (x - 1)^2 - a^2 ≥ 0

-- The statement to prove
theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : ∀ x, ¬p x → q x a) : 
  0 < a ∧ a ≤ 3 :=
by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_range_of_a_l2197_219774


namespace NUMINAMATH_GPT_celine_change_l2197_219730

theorem celine_change :
  let laptop_price := 600
  let smartphone_price := 400
  let tablet_price := 250
  let headphone_price := 100
  let laptops_purchased := 2
  let smartphones_purchased := 4
  let tablets_purchased := 3
  let headphones_purchased := 5
  let discount_rate := 0.10
  let sales_tax_rate := 0.05
  let initial_amount := 5000
  let laptop_total := laptops_purchased * laptop_price
  let smartphone_total := smartphones_purchased * smartphone_price
  let tablet_total := tablets_purchased * tablet_price
  let headphone_total := headphones_purchased * headphone_price
  let discount := discount_rate * (laptop_total + tablet_total)
  let total_before_discount := laptop_total + smartphone_total + tablet_total + headphone_total
  let total_after_discount := total_before_discount - discount
  let sales_tax := sales_tax_rate * total_after_discount
  let final_price := total_after_discount + sales_tax
  let change := initial_amount - final_price
  change = 952.25 :=
  sorry

end NUMINAMATH_GPT_celine_change_l2197_219730


namespace NUMINAMATH_GPT_prove_tirzah_handbags_l2197_219745
noncomputable def tirzah_has_24_handbags (H : ℕ) : Prop :=
  let P := 26 -- number of purses
  let fakeP := P / 2 -- half of the purses are fake
  let authP := P - fakeP -- number of authentic purses
  let fakeH := H / 4 -- one quarter of the handbags are fake
  let authH := H - fakeH -- number of authentic handbags
  authP + authH = 31 -- total number of authentic items
  → H = 24 -- prove the number of handbags is 24

theorem prove_tirzah_handbags : ∃ H : ℕ, tirzah_has_24_handbags H :=
  by
    use 24
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_prove_tirzah_handbags_l2197_219745


namespace NUMINAMATH_GPT_find_b_l2197_219767

theorem find_b (a b c : ℝ) (A B C : ℝ) (h1 : a = 10) (h2 : c = 20) (h3 : B = 120) :
  b = 10 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_find_b_l2197_219767


namespace NUMINAMATH_GPT_Fiona_Less_Than_Charles_l2197_219700

noncomputable def percentDifference (a b : ℝ) : ℝ :=
  ((a - b) / a) * 100

theorem Fiona_Less_Than_Charles : percentDifference 600 (450 * 1.1) = 17.5 :=
by
  sorry

end NUMINAMATH_GPT_Fiona_Less_Than_Charles_l2197_219700


namespace NUMINAMATH_GPT_charity_event_equation_l2197_219775

variable (x : ℕ)

theorem charity_event_equation : x + 5 * (12 - x) = 48 :=
sorry

end NUMINAMATH_GPT_charity_event_equation_l2197_219775


namespace NUMINAMATH_GPT_meridian_students_l2197_219707

theorem meridian_students
  (eighth_to_seventh_ratio : Nat → Nat → Prop)
  (seventh_to_sixth_ratio : Nat → Nat → Prop)
  (r1 : ∀ a b, eighth_to_seventh_ratio a b ↔ 7 * b = 4 * a)
  (r2 : ∀ b c, seventh_to_sixth_ratio b c ↔ 10 * c = 9 * b) :
  ∃ a b c, eighth_to_seventh_ratio a b ∧ seventh_to_sixth_ratio b c ∧ a + b + c = 73 :=
by
  sorry

end NUMINAMATH_GPT_meridian_students_l2197_219707


namespace NUMINAMATH_GPT_probability_green_then_blue_l2197_219777

theorem probability_green_then_blue :
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := green_marbles / total_marbles
  let prob_second_blue := blue_marbles / (total_marbles - 1)
  prob_first_green * prob_second_blue = 4 / 15 :=
sorry

end NUMINAMATH_GPT_probability_green_then_blue_l2197_219777


namespace NUMINAMATH_GPT_speed_of_current_l2197_219738

theorem speed_of_current (h_start: ∀ t: ℝ, t ≥ 0 → u ≥ 0) 
  (boat1_turn_2pm: ∀ t: ℝ, t >= 1 → t < 2 → boat1_turn_13_14) 
  (boat2_turn_3pm: ∀ t: ℝ, t >= 2 → t < 3 → boat2_turn_14_15) 
  (boats_meet: ∀ x: ℝ, x = 7.5) :
  v = 2.5 := 
sorry

end NUMINAMATH_GPT_speed_of_current_l2197_219738


namespace NUMINAMATH_GPT_beijing_olympics_problem_l2197_219726

theorem beijing_olympics_problem
  (M T J D: Type)
  (sports: M → Type)
  (swimming gymnastics athletics volleyball: M → Prop)
  (athlete_sits: M → M → Prop)
  (Maria Tania Juan David: M)
  (woman: M → Prop)
  (left right front next_to: M → M → Prop)
  (h1: ∀ x, swimming x → left x Maria)
  (h2: ∀ x, gymnastics x → front x Juan)
  (h3: next_to Tania David)
  (h4: ∀ x, volleyball x → ∃ y, woman y ∧ next_to y x) :
  athletics David := 
sorry

end NUMINAMATH_GPT_beijing_olympics_problem_l2197_219726


namespace NUMINAMATH_GPT_range_of_a_l2197_219708

theorem range_of_a (x y : ℝ) (a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 4 = 2 * x * y) :
  x^2 + 2 * x * y + y^2 - a * x - a * y + 1 ≥ 0 ↔ a ≤ 17 / 4 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2197_219708


namespace NUMINAMATH_GPT_elois_made_3_loaves_on_Monday_l2197_219753

theorem elois_made_3_loaves_on_Monday
    (bananas_per_loaf : ℕ)
    (twice_as_many : ℕ)
    (total_bananas : ℕ) 
    (h1 : bananas_per_loaf = 4) 
    (h2 : twice_as_many = 2) 
    (h3 : total_bananas = 36)
  : ∃ L : ℕ, (4 * L + 8 * L = 36) ∧ L = 3 :=
sorry

end NUMINAMATH_GPT_elois_made_3_loaves_on_Monday_l2197_219753


namespace NUMINAMATH_GPT_evaluate_P_l2197_219716

noncomputable def P (x : ℝ) : ℝ := x^3 - 6*x^2 - 5*x + 4

theorem evaluate_P (y : ℝ) (z : ℝ) (hz : ∀ n : ℝ, z * P y = P (y - n) + P (y + n)) : P 2 = -22 := by
  sorry

end NUMINAMATH_GPT_evaluate_P_l2197_219716


namespace NUMINAMATH_GPT_find_q_value_l2197_219773

theorem find_q_value 
  (p q r : ℕ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hr : 0 < r) 
  (h : p + 1 / (q + 1 / r : ℚ) = 25 / 19) : 
  q = 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_q_value_l2197_219773


namespace NUMINAMATH_GPT_R2_area_is_160_l2197_219758

-- Define the initial conditions.
structure Rectangle :=
(width : ℝ)
(height : ℝ)

def R1 : Rectangle := { width := 4, height := 8 }

def similar (r1 r2 : Rectangle) : Prop :=
  r2.width / r2.height = r1.width / r1.height

def R2_diagonal := 20

-- Proving that the area of R2 is 160 square inches
theorem R2_area_is_160 (R2 : Rectangle)
  (h_similar : similar R1 R2)
  (h_diagonal : R2.width^2 + R2.height^2 = R2_diagonal^2) :
  R2.width * R2.height = 160 :=
  sorry

end NUMINAMATH_GPT_R2_area_is_160_l2197_219758


namespace NUMINAMATH_GPT_sector_radius_l2197_219787

theorem sector_radius (P : ℝ) (c : ℝ → ℝ) (θ : ℝ) (r : ℝ) (π : ℝ) 
  (h1 : P = 144) 
  (h2 : θ = π)
  (h3 : P = θ * r + 2 * r) 
  (h4 : π = Real.pi)
  : r = 144 / (Real.pi + 2) := 
by
  sorry

end NUMINAMATH_GPT_sector_radius_l2197_219787


namespace NUMINAMATH_GPT_minimum_value_of_f_l2197_219720

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ -1 / Real.exp 1) ∧ (∃ x : ℝ, x > 0 ∧ f x = -1 / Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2197_219720


namespace NUMINAMATH_GPT_parabola_equation_l2197_219780

noncomputable def parabola_vertex_form (x y a : ℝ) : Prop := y = a * (x - 3)^2 + 5

noncomputable def parabola_standard_form (x y : ℝ) : Prop := y = -3 * x^2 + 18 * x - 22

theorem parabola_equation (a : ℝ) (h_vertex : parabola_vertex_form 3 5 a) (h_point : parabola_vertex_form 2 2 a) :
  ∃ x y, parabola_standard_form x y :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l2197_219780
