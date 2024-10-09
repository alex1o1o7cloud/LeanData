import Mathlib

namespace largest_possible_dividend_l1483_148335

theorem largest_possible_dividend (divisor quotient : ℕ) (remainder : ℕ) 
  (h_divisor : divisor = 18)
  (h_quotient : quotient = 32)
  (h_remainder : remainder < divisor) :
  quotient * divisor + remainder = 593 :=
by
  -- No proof here, add sorry to skip the proof
  sorry

end largest_possible_dividend_l1483_148335


namespace original_square_area_l1483_148303

theorem original_square_area {x y : ℕ} (h1 : y ≠ 1)
  (h2 : x^2 = 24 + y^2) : x^2 = 49 :=
sorry

end original_square_area_l1483_148303


namespace maximum_ab_l1483_148393

open Real

theorem maximum_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 6 * a + 5 * b = 75) :
  ab ≤ 46.875 :=
by
  -- proof goes here
  sorry

end maximum_ab_l1483_148393


namespace value_of_a7_l1483_148359

-- Let \( \{a_n\} \) be a sequence such that \( S_n \) denotes the sum of the first \( n \) terms.
-- Given \( S_{n+1}, S_{n+2}, S_{n+3} \) form an arithmetic sequence and \( a_2 = -2 \),
-- prove that \( a_7 = 64 \).

theorem value_of_a7 (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 2) + S (n + 1) = 2 * S n) →
  a 2 = -2 →
  (∀ n : ℕ, a (n + 2) = -2 * a (n + 1)) →
  a 7 = 64 :=
by
  -- skip the proof
  sorry

end value_of_a7_l1483_148359


namespace set_in_proportion_l1483_148336

theorem set_in_proportion : 
  let a1 := 3
  let a2 := 9
  let b1 := 10
  let b2 := 30
  (a1 * b2 = a2 * b1) := 
by {
  sorry
}

end set_in_proportion_l1483_148336


namespace sequence_general_term_l1483_148365

-- Given a sequence {a_n} whose sum of the first n terms S_n = 2a_n - 1,
-- prove that the general formula for the n-th term a_n is 2^(n-1).

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
    (h₁ : ∀ n : ℕ, S n = 2 * a n - 1)
    (h₂ : S 1 = 1) : ∀ n : ℕ, a (n + 1) = 2 ^ n :=
by
  sorry

end sequence_general_term_l1483_148365


namespace solve_f_log2_20_l1483_148349

noncomputable def f (x : ℝ) : ℝ :=
if -1 ≤ x ∧ x < 0 then 2^x else 0 -- Placeholder for other values

theorem solve_f_log2_20 :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 4) = f x) →
  (∀ x, -1 ≤ x ∧ x < 0 → f x = 2^x) →
  f (Real.log 20 / Real.log 2) = -4 / 5 :=
by
  sorry

end solve_f_log2_20_l1483_148349


namespace consecutive_integer_sets_l1483_148361

-- Define the problem
def sum_consecutive_integers (n a : ℕ) : ℕ :=
  (n * (2 * a + n - 1)) / 2

def is_valid_sequence (n a S : ℕ) : Prop :=
  n ≥ 2 ∧ sum_consecutive_integers n a = S

-- Lean 4 theorem statement
theorem consecutive_integer_sets (S : ℕ) (h : S = 180) :
  (∃ (n a : ℕ), is_valid_sequence n a S) →
  (∃ (n1 n2 n3 : ℕ) (a1 a2 a3 : ℕ), 
    is_valid_sequence n1 a1 S ∧ 
    is_valid_sequence n2 a2 S ∧ 
    is_valid_sequence n3 a3 S ∧
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3) :=
by
  sorry

end consecutive_integer_sets_l1483_148361


namespace money_split_l1483_148356

theorem money_split (donna_share friend_share : ℝ) (h1 : donna_share = 32.50) (h2 : friend_share = 32.50) :
  donna_share + friend_share = 65 :=
by
  sorry

end money_split_l1483_148356


namespace arithmetic_sequence_property_l1483_148357

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

def condition (S : ℕ → ℝ) : Prop :=
  (S 8 - S 5) * (S 8 - S 4) < 0

-- Theorem to prove
theorem arithmetic_sequence_property {a : ℕ → ℝ} {S : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_cond : condition S) :
  |a 5| > |a 6| := 
sorry

end arithmetic_sequence_property_l1483_148357


namespace probability_of_neither_is_correct_l1483_148375

-- Definitions of the given conditions
def total_buyers : ℕ := 100
def cake_buyers : ℕ := 50
def muffin_buyers : ℕ := 40
def both_cake_and_muffin_buyers : ℕ := 19

-- Define the probability calculation function
def probability_neither (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) : ℚ :=
  let buyers_neither := total - (cake + muffin - both)
  (buyers_neither : ℚ) / (total : ℚ)

-- State the main theorem to ensure it is equivalent to our mathematical problem
theorem probability_of_neither_is_correct :
  probability_neither total_buyers cake_buyers muffin_buyers both_cake_and_muffin_buyers = 0.29 := 
sorry

end probability_of_neither_is_correct_l1483_148375


namespace probability_same_color_white_l1483_148331

/--
Given a box with 6 white balls and 5 black balls, if 3 balls are drawn such that all drawn balls have the same color,
prove that the probability that these balls are white is 2/3.
-/
theorem probability_same_color_white :
  (∃ (n_white n_black drawn_white drawn_black total_same_color : ℕ),
    n_white = 6 ∧ n_black = 5 ∧
    drawn_white = Nat.choose n_white 3 ∧ drawn_black = Nat.choose n_black 3 ∧
    total_same_color = drawn_white + drawn_black ∧
    (drawn_white:ℚ) / total_same_color = 2 / 3) :=
sorry

end probability_same_color_white_l1483_148331


namespace polynomial_identity_l1483_148347

theorem polynomial_identity (x : ℝ) (h₁ : x^5 - 3*x + 2 = 0) (h₂ : x ≠ 1) : 
  x^4 + x^3 + x^2 + x + 1 = 3 := 
by 
  sorry

end polynomial_identity_l1483_148347


namespace seashells_after_giving_cannot_determine_starfish_l1483_148313

-- Define the given conditions
def initial_seashells : Nat := 66
def seashells_given : Nat := 52
def seashells_left : Nat := 14

-- The main theorem to prove
theorem seashells_after_giving (initial : Nat) (given : Nat) (left : Nat) :
  initial = 66 -> given = 52 -> left = 14 -> initial - given = left :=
by 
  intros 
  sorry

-- The starfish count question
def starfish (count: Option Nat) : Prop :=
  count = none

-- Prove that we cannot determine the number of starfish Benny found
theorem cannot_determine_starfish (count: Option Nat) :
  count = none :=
by 
  intros 
  sorry

end seashells_after_giving_cannot_determine_starfish_l1483_148313


namespace length_of_each_piece_l1483_148392

-- Definitions based on conditions
def total_length : ℝ := 42.5
def number_of_pieces : ℝ := 50

-- The statement that we need to prove
theorem length_of_each_piece (h1 : total_length = 42.5) (h2 : number_of_pieces = 50) : 
  total_length / number_of_pieces = 0.85 := 
by
  sorry

end length_of_each_piece_l1483_148392


namespace opposite_of_six_is_negative_six_l1483_148374

theorem opposite_of_six_is_negative_six : -6 = -6 :=
by
  sorry

end opposite_of_six_is_negative_six_l1483_148374


namespace range_of_m_l1483_148362

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x-1)^2 < m^2 → |1 - (x-1)/3| < 2) → (abs m ≤ 3) :=
by
  sorry

end range_of_m_l1483_148362


namespace market_value_calculation_l1483_148369

variables (annual_dividend_per_share face_value yield market_value : ℝ)

axiom annual_dividend_definition : annual_dividend_per_share = 0.09 * face_value
axiom face_value_definition : face_value = 100
axiom yield_definition : yield = 0.25

theorem market_value_calculation (annual_dividend_per_share face_value yield market_value : ℝ) 
  (h1: annual_dividend_per_share = 0.09 * face_value)
  (h2: face_value = 100)
  (h3: yield = 0.25):
  market_value = annual_dividend_per_share / yield :=
sorry

end market_value_calculation_l1483_148369


namespace max_value_expr_l1483_148320

theorem max_value_expr : ∃ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) = 85 :=
by sorry

end max_value_expr_l1483_148320


namespace other_root_l1483_148342

open Complex

-- Defining the conditions that are given in the problem
def quadratic_equation (x : ℂ) (m : ℝ) : Prop :=
  x^2 + (1 - 2 * I) * x + (3 * m - I) = 0

def has_real_root (x : ℂ) : Prop :=
  ∃ α : ℝ, x = α

-- The main theorem statement we need to prove
theorem other_root (m : ℝ) (α : ℝ) (α_real_root : quadratic_equation α m) :
  quadratic_equation (-1/2 + 2 * I) m :=
sorry

end other_root_l1483_148342


namespace sum_of_products_non_positive_l1483_148358

theorem sum_of_products_non_positive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end sum_of_products_non_positive_l1483_148358


namespace expected_value_of_biased_coin_l1483_148329

noncomputable def expected_value : ℚ :=
  (2 / 3) * 5 + (1 / 3) * -6

theorem expected_value_of_biased_coin :
  expected_value = 4 / 3 := by
  sorry

end expected_value_of_biased_coin_l1483_148329


namespace dmitriev_older_by_10_l1483_148312

-- Define the ages of each of the elders
variables (A B C D E F : ℕ)

-- The conditions provided in the problem
axiom hAlyosha : A > (A - 1)
axiom hBorya : B > (B - 2)
axiom hVasya : C > (C - 3)
axiom hGrisha : D > (D - 4)

-- Establishing an equation for the age differences leading to the proof
axiom age_sum_relation : A + B + C + D + E = (A - 1) + (B - 2) + (C - 3) + (D - 4) + F

-- We state that Dmitriev is older than Dima by 10 years
theorem dmitriev_older_by_10 : F = E + 10 :=
by
  -- sorry replaces the proof
  sorry

end dmitriev_older_by_10_l1483_148312


namespace compare_abc_l1483_148339

theorem compare_abc :
  let a := Real.log 17
  let b := 3
  let c := Real.exp (Real.sqrt 2)
  a < b ∧ b < c :=
by
  sorry

end compare_abc_l1483_148339


namespace problem_eq_l1483_148363

theorem problem_eq : 
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → y = x / (x + 1) → (x - y + 4 * x * y) / (x * y) = 5 :=
by
  intros x y hx hnz hyxy
  sorry

end problem_eq_l1483_148363


namespace sector_perimeter_l1483_148304

theorem sector_perimeter (r : ℝ) (c : ℝ) (angle_deg : ℝ) (angle_rad := angle_deg * Real.pi / 180) 
  (arc_length := r * angle_rad) (P := arc_length + c)
  (h1 : r = 10) (h2 : c = 10) (h3 : angle_deg = 120) :
  P = 20 * Real.pi / 3 + 10 :=
by
  sorry

end sector_perimeter_l1483_148304


namespace male_to_female_cat_weight_ratio_l1483_148311

variable (w_f w_m w_t : ℕ)

def female_cat_weight : Prop := w_f = 2
def total_weight : Prop := w_t = 6
def male_cat_heavier : Prop := w_m > w_f

theorem male_to_female_cat_weight_ratio
  (h_female_cat_weight : female_cat_weight w_f)
  (h_total_weight : total_weight w_t)
  (h_male_cat_heavier : male_cat_heavier w_m w_f) :
  w_m = 4 ∧ w_t = w_f + w_m ∧ (w_m / w_f) = 2 :=
by
  sorry

end male_to_female_cat_weight_ratio_l1483_148311


namespace solution_set_of_quadratic_inequality_l1483_148343

namespace QuadraticInequality

variables {a b : ℝ}

def hasRoots (a b : ℝ) : Prop :=
  let x1 := -1 / 2
  let x2 := 1 / 3
  (- x1 + x2 = - b / a) ∧ (-x1 * x2 = 2 / a)

theorem solution_set_of_quadratic_inequality (h : hasRoots a b) : a + b = -14 :=
sorry

end QuadraticInequality

end solution_set_of_quadratic_inequality_l1483_148343


namespace bill_experience_l1483_148338

theorem bill_experience (j b : ℕ) 
  (h₁ : j - 5 = 3 * (b - 5)) 
  (h₂ : j = 2 * b) : b = 10 :=
sorry

end bill_experience_l1483_148338


namespace rectangle_area_is_243_square_meters_l1483_148370

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l1483_148370


namespace intersection_M_N_l1483_148340

def M := { x : ℝ | x^2 - 2 * x < 0 }
def N := { x : ℝ | abs x < 1 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l1483_148340


namespace unique_triple_property_l1483_148397

theorem unique_triple_property (a b c : ℕ) (h1 : a ∣ b * c + 1) (h2 : b ∣ a * c + 1) (h3 : c ∣ a * b + 1) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a = 2 ∧ b = 3 ∧ c = 7) :=
by
  sorry

end unique_triple_property_l1483_148397


namespace find_QE_l1483_148317

noncomputable def QE (QD DE : ℝ) : ℝ :=
  QD + DE

theorem find_QE :
  ∀ (Q C R D E : Type) (QR QD DE QE : ℝ), 
  QD = 5 →
  QE = QD + DE →
  QR = DE - QD →
  QR^2 = QD * QE →
  QE = (QD + 5 + 5 * Real.sqrt 5) / 2 :=
by
  intros
  sorry

end find_QE_l1483_148317


namespace successive_discounts_eq_single_discount_l1483_148390

theorem successive_discounts_eq_single_discount :
  ∀ (x : ℝ), (1 - 0.15) * (1 - 0.25) * x = (1 - 0.3625) * x :=
by
  intro x
  sorry

end successive_discounts_eq_single_discount_l1483_148390


namespace line_equation_l1483_148330

theorem line_equation (a b : ℝ)
(h1 : a * -1 + b * 2 = 0) 
(h2 : a = b) :
((a = 1 ∧ b = -1) ∨ (a = 2 ∧ b = -1)) := 
by
  sorry

end line_equation_l1483_148330


namespace solve_for_y_l1483_148307

theorem solve_for_y (x y : ℝ) (h : 5 * x + 3 * y = 1) : y = (1 - 5 * x) / 3 :=
by
  sorry

end solve_for_y_l1483_148307


namespace problem_solution_l1483_148316

variables {a b c : ℝ}

theorem problem_solution
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) :
  (a + b) * (b + c) * (a + c) = 0 := 
sorry

end problem_solution_l1483_148316


namespace city_population_correct_l1483_148327

variable (C G : ℕ)

theorem city_population_correct :
  (C - G = 119666) ∧ (C + G = 845640) → (C = 482653) := by
  intro h
  have h1 : C - G = 119666 := h.1
  have h2 : C + G = 845640 := h.2
  sorry

end city_population_correct_l1483_148327


namespace determine_a_value_l1483_148315

theorem determine_a_value :
  ∀ (a b c d : ℕ), 
  (a = b + 3) →
  (b = c + 6) →
  (c = d + 15) →
  (d = 50) →
  a = 74 :=
by
  intros a b c d h1 h2 h3 h4
  sorry

end determine_a_value_l1483_148315


namespace find_wanderer_in_8th_bar_l1483_148354

noncomputable def wanderer_probability : ℚ := 1 / 3

theorem find_wanderer_in_8th_bar
    (total_bars : ℕ)
    (initial_prob_in_any_bar : ℚ)
    (prob_not_in_specific_bar : ℚ)
    (prob_not_in_first_seven : ℚ)
    (posterior_prob : ℚ)
    (h1 : total_bars = 8)
    (h2 : initial_prob_in_any_bar = 4 / 5)
    (h3 : prob_not_in_specific_bar = 1 - (initial_prob_in_any_bar / total_bars))
    (h4 : prob_not_in_first_seven = prob_not_in_specific_bar ^ 7)
    (h5 : posterior_prob = initial_prob_in_any_bar / prob_not_in_first_seven) :
    posterior_prob = wanderer_probability := 
sorry

end find_wanderer_in_8th_bar_l1483_148354


namespace wanda_crayons_l1483_148302

variable (Dina Jacob Wanda : ℕ)

theorem wanda_crayons : Dina = 28 ∧ Jacob = Dina - 2 ∧ Dina + Jacob + Wanda = 116 → Wanda = 62 :=
by
  intro h
  sorry

end wanda_crayons_l1483_148302


namespace problem_a_problem_b_l1483_148306

-- Problem (a)
theorem problem_a (n : Nat) : Nat.mod (7 ^ (2 * n) - 4 ^ (2 * n)) 33 = 0 := sorry

-- Problem (b)
theorem problem_b (n : Nat) : Nat.mod (3 ^ (6 * n) - 2 ^ (6 * n)) 35 = 0 := sorry

end problem_a_problem_b_l1483_148306


namespace expression_equals_5776_l1483_148376

-- Define constants used in the problem
def a : ℕ := 476
def b : ℕ := 424
def c : ℕ := 4

-- Define the expression using the constants
def expression : ℕ := (a + b) ^ 2 - c * a * b

-- The target proof statement
theorem expression_equals_5776 : expression = 5776 := by
  sorry

end expression_equals_5776_l1483_148376


namespace velvet_needed_for_box_l1483_148309

theorem velvet_needed_for_box : 
  let area_long_side := 8 * 6
  let area_short_side := 5 * 6
  let area_top_bottom := 40
  let total_area := (2 * area_long_side) + (2 * area_short_side) + (2 * area_top_bottom)
  total_area = 236 :=
by
  sorry

end velvet_needed_for_box_l1483_148309


namespace length_of_bridge_l1483_148355

theorem length_of_bridge
  (length_train : ℕ) (speed_train_kmhr : ℕ) (crossing_time : ℕ)
  (speed_conversion_factor : ℝ) (m_per_s_kmhr : ℝ) 
  (speed_train_ms : ℝ) (total_distance : ℝ) (length_bridge : ℝ)
  (h1 : length_train = 155)
  (h2 : speed_train_kmhr = 45)
  (h3 : crossing_time = 30)
  (h4 : speed_conversion_factor = 1000 / 3600)
  (h5 : m_per_s_kmhr = speed_train_kmhr * speed_conversion_factor)
  (h6 : speed_train_ms = 45 * (5 / 18))
  (h7 : total_distance = speed_train_ms * crossing_time)
  (h8 : length_bridge = total_distance - length_train):
  length_bridge = 220 :=
by
  sorry

end length_of_bridge_l1483_148355


namespace necessary_but_not_sufficient_condition_l1483_148399

variable (a : ℝ) (x : ℝ)

def inequality_holds_for_all_real_numbers (a : ℝ) : Prop :=
    ∀ x : ℝ, (a * x^2 - a * x + 1 > 0)

theorem necessary_but_not_sufficient_condition :
  (0 < a ∧ a < 4) ↔
  (inequality_holds_for_all_real_numbers a) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1483_148399


namespace greatest_C_inequality_l1483_148382

theorem greatest_C_inequality (α x y z : ℝ) (hα_pos : 0 < α) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h_xyz_sum : x * y + y * z + z * x = α) : 
  16 ≤ (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) / (x / z + z / x + 2) :=
sorry

end greatest_C_inequality_l1483_148382


namespace hours_per_day_l1483_148360

variable (m w : ℝ)
variable (h : ℕ)

-- Assume the equivalence of work done by women and men
axiom work_equiv : 3 * w = 2 * m

-- Total work done by men
def work_men := 15 * m * 21 * h
-- Total work done by women
def work_women := 21 * w * 36 * 5

-- The total work done by men and women is equal
theorem hours_per_day (h : ℕ) (w m : ℝ) (work_equiv : 3 * w = 2 * m) :
  15 * m * 21 * h = 21 * w * 36 * 5 → h = 8 :=
by
  intro H
  sorry

end hours_per_day_l1483_148360


namespace perfect_cubes_l1483_148305

theorem perfect_cubes (n : ℕ) (h : n > 0) : 
  (n = 7 ∨ n = 11 ∨ n = 12 ∨ n = 25) ↔ ∃ k : ℤ, (n^3 - 18*n^2 + 115*n - 391) = k^3 :=
by exact sorry

end perfect_cubes_l1483_148305


namespace find_reflection_line_l1483_148301

/-*
Triangle ABC has vertices with coordinates A(2,3), B(7,8), and C(-4,6).
The triangle is reflected about line L.
The image points are A'(2,-5), B'(7,-10), and C'(-4,-8).
Prove that the equation of line L is y = -1.
*-/
theorem find_reflection_line :
  ∃ (L : ℝ), (∀ (x : ℝ), (∃ (k : ℝ), L = k) ∧ (L = -1)) :=
by sorry

end find_reflection_line_l1483_148301


namespace math_problem_l1483_148300

theorem math_problem : 
  ( - (1 / 12 : ℚ) + (1 / 3 : ℚ) - (1 / 2 : ℚ) ) / ( - (1 / 18 : ℚ) ) = 4.5 := 
by
  sorry

end math_problem_l1483_148300


namespace find_x_l1483_148353

-- Define the functions δ (delta) and φ (phi)
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- State the theorem with conditions and question, and assert the answer
theorem find_x :
  (delta ∘ phi) x = 11 → x = -5/6 := by
  intros
  sorry

end find_x_l1483_148353


namespace correct_answer_is_A_l1483_148319

-- Definitions derived from problem conditions
def algorithm := Type
def has_sequential_structure (alg : algorithm) : Prop := sorry -- Actual definition should define what a sequential structure is for an algorithm

-- Given: An algorithm must contain a sequential structure.
theorem correct_answer_is_A (alg : algorithm) : has_sequential_structure alg :=
sorry

end correct_answer_is_A_l1483_148319


namespace number_of_candidates_l1483_148378

-- Definitions for the given conditions
def total_marks : ℝ := 2000
def average_marks : ℝ := 40

-- Theorem to prove the number of candidates
theorem number_of_candidates : total_marks / average_marks = 50 := by
  sorry

end number_of_candidates_l1483_148378


namespace minimum_value_of_expression_l1483_148345

noncomputable def min_value (p q r s t u : ℝ) : ℝ :=
  (1 / p) + (9 / q) + (25 / r) + (49 / s) + (81 / t) + (121 / u)

theorem minimum_value_of_expression (p q r s t u : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hs : 0 < s) (ht : 0 < t) (hu : 0 < u) (h_sum : p + q + r + s + t + u = 11) :
  min_value p q r s t u ≥ 1296 / 11 :=
by sorry

end minimum_value_of_expression_l1483_148345


namespace early_finish_hours_l1483_148387

theorem early_finish_hours 
  (h : Nat) 
  (total_customers : Nat) 
  (num_workers : Nat := 3)
  (service_rate : Nat := 7) 
  (full_hours : Nat := 8)
  (total_customers_served : total_customers = 154) 
  (two_workers_hours : Nat := 2 * full_hours * service_rate) 
  (early_worker_customers : Nat := h * service_rate)
  (total_service : total_customers = two_workers_hours + early_worker_customers) : 
  h = 6 :=
by
  sorry

end early_finish_hours_l1483_148387


namespace volume_expansion_rate_l1483_148386

theorem volume_expansion_rate (R m : ℝ) (h1 : R = 1) (h2 : (4 * π * (m^3 - 1) / 3) / (m - 1) = 28 * π / 3) : m = 2 :=
sorry

end volume_expansion_rate_l1483_148386


namespace cricket_average_l1483_148394

theorem cricket_average (x : ℝ) (h1 : 15 * x + 121 = 16 * (x + 6)) : x = 25 := by
  -- proof goes here, but we skip it with sorry
  sorry

end cricket_average_l1483_148394


namespace dice_roll_probability_bounds_l1483_148381

noncomputable def dice_roll_probability : Prop :=
  let n := 80
  let p := (1 : ℝ) / 6
  let q := 1 - p
  let epsilon := 2.58 / 24
  let lower_bound := (p - epsilon) * n
  let upper_bound := (p + epsilon) * n
  5 ≤ lower_bound ∧ upper_bound ≤ 22

theorem dice_roll_probability_bounds :
  dice_roll_probability :=
sorry

end dice_roll_probability_bounds_l1483_148381


namespace speed_ratio_l1483_148395

theorem speed_ratio (a b v1 v2 S : ℝ) (h1 : S = a * (v1 + v2)) (h2 : S = b * (v1 - v2)) (h3 : a ≠ b) : 
  v1 / v2 = (a + b) / (b - a) :=
by
  -- proof skipped
  sorry

end speed_ratio_l1483_148395


namespace square_divisibility_l1483_148371

theorem square_divisibility (n : ℤ) : n^2 % 4 = 0 ∨ n^2 % 4 = 1 := sorry

end square_divisibility_l1483_148371


namespace pyramid_lateral_surface_area_l1483_148323

noncomputable def lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) : ℝ :=
  n * S

theorem pyramid_lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) (A : ℝ) :
  A = n * S * (Real.cos α) →
  lateral_surface_area S n α = A / (Real.cos α) :=
by
  sorry

end pyramid_lateral_surface_area_l1483_148323


namespace arithmetic_sequence_a2_a8_l1483_148385

theorem arithmetic_sequence_a2_a8 (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : a 3 + a 4 + a 5 + a 6 + a 7 = 450) :
  a 2 + a 8 = 180 :=
by
  sorry

end arithmetic_sequence_a2_a8_l1483_148385


namespace value_of_x_l1483_148326

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := 
by
  sorry

end value_of_x_l1483_148326


namespace abs_of_sub_sqrt_l1483_148334

theorem abs_of_sub_sqrt (h : 2 > Real.sqrt 3) : |2 - Real.sqrt 3| = 2 - Real.sqrt 3 :=
sorry

end abs_of_sub_sqrt_l1483_148334


namespace combined_average_age_of_fifth_graders_teachers_and_parents_l1483_148348

theorem combined_average_age_of_fifth_graders_teachers_and_parents
  (num_fifth_graders : ℕ) (avg_age_fifth_graders : ℕ)
  (num_teachers : ℕ) (avg_age_teachers : ℕ)
  (num_parents : ℕ) (avg_age_parents : ℕ)
  (h1 : num_fifth_graders = 40) (h2 : avg_age_fifth_graders = 10)
  (h3 : num_teachers = 4) (h4 : avg_age_teachers = 40)
  (h5 : num_parents = 60) (h6 : avg_age_parents = 34)
  : (num_fifth_graders * avg_age_fifth_graders + num_teachers * avg_age_teachers + num_parents * avg_age_parents) /
    (num_fifth_graders + num_teachers + num_parents) = 25 :=
by sorry

end combined_average_age_of_fifth_graders_teachers_and_parents_l1483_148348


namespace artifacts_per_wing_l1483_148373

theorem artifacts_per_wing (P A w_wings p_wings a_wings : ℕ) (hp1 : w_wings = 8)
  (hp2 : A = 4 * P) (hp3 : p_wings = 3) (hp4 : (∃ L S : ℕ, L = 1 ∧ S = 12 ∧ P = 2 * S + L))
  (hp5 : a_wings = w_wings - p_wings) :
  A / a_wings = 20 :=
by
  sorry

end artifacts_per_wing_l1483_148373


namespace initially_caught_and_tagged_is_30_l1483_148328

open Real

-- Define conditions
def total_second_catch : ℕ := 50
def tagged_second_catch : ℕ := 2
def total_pond_fish : ℕ := 750

-- Define ratio condition
def ratio_condition (T : ℕ) : Prop :=
  (T : ℝ) / (total_pond_fish : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ)

-- Prove the number of fish initially caught and tagged is 30
theorem initially_caught_and_tagged_is_30 :
  ∃ T : ℕ, ratio_condition T ∧ T = 30 :=
by
  -- Skipping proof
  sorry

end initially_caught_and_tagged_is_30_l1483_148328


namespace cost_to_replace_is_800_l1483_148314

-- Definitions based on conditions
def trade_in_value (num_movies : ℕ) (trade_in_price : ℕ) : ℕ :=
  num_movies * trade_in_price

def dvd_cost (num_movies : ℕ) (dvd_price : ℕ) : ℕ :=
  num_movies * dvd_price

def replacement_cost (num_movies : ℕ) (trade_in_price : ℕ) (dvd_price : ℕ) : ℕ :=
  dvd_cost num_movies dvd_price - trade_in_value num_movies trade_in_price

-- Problem statement: it costs John $800 to replace his movies
theorem cost_to_replace_is_800 (num_movies trade_in_price dvd_price : ℕ)
  (h1 : num_movies = 100) (h2 : trade_in_price = 2) (h3 : dvd_price = 10) :
  replacement_cost num_movies trade_in_price dvd_price = 800 :=
by
  -- Proof would go here
  sorry

end cost_to_replace_is_800_l1483_148314


namespace taxi_fare_distance_l1483_148324

-- Define the fare calculation and distance function
def fare (x : ℕ) : ℝ :=
  if x ≤ 4 then 10
  else 10 + (x - 4) * 1.5

-- Proof statement
theorem taxi_fare_distance (x : ℕ) : fare x = 16 → x = 8 :=
by
  -- Proof skipped
  sorry

end taxi_fare_distance_l1483_148324


namespace min_birthdays_on_wednesday_l1483_148377

theorem min_birthdays_on_wednesday 
  (W X : ℕ) 
  (h1 : W + 6 * X = 50) 
  (h2 : W > X) : 
  W = 8 := 
sorry

end min_birthdays_on_wednesday_l1483_148377


namespace total_nominal_income_l1483_148398

theorem total_nominal_income
  (c1 : 8700 * ((1 + 0.06 / 12) ^ 6 - 1) = 264.28)
  (c2 : 8700 * ((1 + 0.06 / 12) ^ 5 - 1) = 219.69)
  (c3 : 8700 * ((1 + 0.06 / 12) ^ 4 - 1) = 175.31)
  (c4 : 8700 * ((1 + 0.06 / 12) ^ 3 - 1) = 131.15)
  (c5 : 8700 * ((1 + 0.06 / 12) ^ 2 - 1) = 87.22)
  (c6 : 8700 * (1 + 0.06 / 12 - 1) = 43.5) :
  264.28 + 219.69 + 175.31 + 131.15 + 87.22 + 43.5 = 921.15 := by
  sorry

end total_nominal_income_l1483_148398


namespace gray_percentage_correct_l1483_148322

-- Define the conditions
def total_squares := 25
def type_I_triangle_equivalent_squares := 8 * (1 / 2)
def type_II_triangle_equivalent_squares := 8 * (1 / 4)
def full_gray_squares := 4

-- Calculate the gray component
def gray_squares := type_I_triangle_equivalent_squares + type_II_triangle_equivalent_squares + full_gray_squares

-- Fraction representing the gray part of the quilt
def gray_fraction := gray_squares / total_squares

-- Translate fraction to percentage
def gray_percentage := gray_fraction * 100

theorem gray_percentage_correct : gray_percentage = 40 := by
  simp [total_squares, type_I_triangle_equivalent_squares, type_II_triangle_equivalent_squares, full_gray_squares, gray_squares, gray_fraction, gray_percentage]
  sorry -- You could expand this to a detailed proof if needed.

end gray_percentage_correct_l1483_148322


namespace usual_time_is_25_l1483_148380

-- Definitions 
variables {S T : ℝ} (h1 : S * T = 5 / 4 * S * (T - 5))

-- Theorem statement
theorem usual_time_is_25 (h : S * T = 5 / 4 * S * (T - 5)) : T = 25 :=
by 
-- Using the assumption h, we'll derive that T = 25
sorry

end usual_time_is_25_l1483_148380


namespace even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l1483_148388

-- Definitions for the conditions in Lean 4
def regular_n_gon (n : ℕ) := true -- Dummy definition; actual geometric properties not needed for statement

def connected_path_visits_each_vertex_once (n : ℕ) := true -- Dummy definition; actual path properties not needed for statement

def parallel_pair (i j p q : ℕ) (n : ℕ) : Prop := (i + j) % n = (p + q) % n

-- Statements for part (a) and (b)

theorem even_n_has_parallel_pair (n : ℕ) (h_even : n % 2 = 0) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n := 
sorry

theorem odd_n_cannot_have_exactly_one_parallel_pair (n : ℕ) (h_odd : n % 2 = 1) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ¬∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n ∧ 
  (∀ (i' j' p' q' : ℕ), (i' ≠ p' ∨ j' ≠ q') → ¬parallel_pair i' j' p' q' n) := 
sorry

end even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l1483_148388


namespace jill_food_percentage_l1483_148350

theorem jill_food_percentage (total_amount : ℝ) (tax_rate_clothing tax_rate_other_items spent_clothing_rate spent_other_rate spent_total_tax_rate : ℝ) : 
  spent_clothing_rate = 0.5 →
  spent_other_rate = 0.25 →
  tax_rate_clothing = 0.1 →
  tax_rate_other_items = 0.2 →
  spent_total_tax_rate = 0.1 →
  (spent_clothing_rate * tax_rate_clothing * total_amount) + (spent_other_rate * tax_rate_other_items * total_amount) = spent_total_tax_rate * total_amount →
  (1 - spent_clothing_rate - spent_other_rate) * total_amount / total_amount = 0.25 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end jill_food_percentage_l1483_148350


namespace gcd_lcm_product_l1483_148366

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end gcd_lcm_product_l1483_148366


namespace quadratic_rewrite_l1483_148396

theorem quadratic_rewrite :
  ∃ a d : ℤ, (∀ x : ℝ, x^2 + 500 * x + 2500 = (x + a)^2 + d) ∧ (d / a) = -240 := by
  sorry

end quadratic_rewrite_l1483_148396


namespace initial_students_count_l1483_148333

theorem initial_students_count (n W : ℝ)
    (h1 : W = n * 28)
    (h2 : W + 10 = (n + 1) * 27.4) :
    n = 29 :=
by
  sorry

end initial_students_count_l1483_148333


namespace simplify_trig_expr_l1483_148337

theorem simplify_trig_expr : 
  let θ := 60
  let tan_θ := Real.sqrt 3
  let cot_θ := (Real.sqrt 3)⁻¹
  (tan_θ^3 + cot_θ^3) / (tan_θ + cot_θ) = 7 / 3 :=
by
  sorry

end simplify_trig_expr_l1483_148337


namespace sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age_l1483_148383

-- Definitions of the conditions
def L := 2
def C := 2 * L^2  -- Chloe's age today based on Liam's age
def J := C + 3    -- Joey's age today

-- The future time when Joey's age is twice Liam's age
def future_time : ℕ := (sorry : ℕ) -- Placeholder for computation of 'n'
lemma compute_n : 2 * (L + future_time) = J + future_time := sorry

-- Joey's age at future time when it is twice Liam's age
def age_at_future_time : ℕ := J + future_time

-- Sum of the two digits of Joey's age at that future time
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Final statement: sum of the digits of Joey's age at the specified future time
theorem sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age :
  digit_sum age_at_future_time = 9 :=
by
  exact sorry

end sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age_l1483_148383


namespace athlete_speed_l1483_148332

theorem athlete_speed (d t : ℝ) (H_d : d = 200) (H_t : t = 40) : (d / t) = 5 := by
  sorry

end athlete_speed_l1483_148332


namespace sum_of_m_n_l1483_148367

-- Define the setup for the problem
def side_length_of_larger_square := 3
def side_length_of_smaller_square := 1
def side_length_of_given_rectangle_l1 := 1
def side_length_of_given_rectangle_l2 := 3
def total_area_of_larger_square := side_length_of_larger_square * side_length_of_larger_square
def area_of_smaller_square := side_length_of_smaller_square * side_length_of_smaller_square
def area_of_given_rectangle := side_length_of_given_rectangle_l1 * side_length_of_given_rectangle_l2

-- Define the variable for the area of rectangle R
def area_of_R := total_area_of_larger_square - (area_of_smaller_square + area_of_given_rectangle)

-- Given the problem statement, we need to find m and n such that the area of R is m/n.
def m := 5
def n := 1

-- We need to prove that m + n = 6 given these conditions
theorem sum_of_m_n : m + n = 6 := by
  sorry

end sum_of_m_n_l1483_148367


namespace average_price_of_initial_fruit_l1483_148308

theorem average_price_of_initial_fruit (A O : ℕ) (h1 : A + O = 10) (h2 : (40 * A + 60 * (O - 6)) / (A + O - 6) = 45) : 
  (40 * A + 60 * O) / 10 = 54 :=
by 
  sorry

end average_price_of_initial_fruit_l1483_148308


namespace no_arithmetic_progression_in_squares_l1483_148341

theorem no_arithmetic_progression_in_squares :
  ∀ (a d : ℕ), d > 0 → ¬ (∃ (f : ℕ → ℕ), 
    (∀ n, f n = a + n * d) ∧ 
    (∀ n, ∃ m, n ^ 2 = f m)) :=
by
  sorry

end no_arithmetic_progression_in_squares_l1483_148341


namespace sum_base8_to_decimal_l1483_148352

theorem sum_base8_to_decimal (a b : ℕ) (ha : a = 5) (hb : b = 0o17)
  (h_sum_base8 : a + b = 0o24) : (a + b) = 20 := by
  sorry

end sum_base8_to_decimal_l1483_148352


namespace xy_relationship_l1483_148318

theorem xy_relationship :
  let x := 123456789 * 123456786
  let y := 123456788 * 123456787
  x < y := 
by
  sorry

end xy_relationship_l1483_148318


namespace winning_strategy_l1483_148325

/-- Given a square table n x n, two players A and B are playing the following game: 
  - At the beginning, all cells of the table are empty.
  - Player A has the first move, and in each of their moves, a player will put a coin on some cell 
    that doesn't contain a coin and is not adjacent to any of the cells that already contain a coin. 
  - The player who makes the last move wins. 

  Cells are adjacent if they share an edge.

  - If n is even, player B has the winning strategy.
  - If n is odd, player A has the winning strategy.
-/
theorem winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ (B_strat : winning_strategy_for_B), True) ∧ (n % 2 = 1 → ∃ (A_strat : winning_strategy_for_A), True) :=
by {
  admit
}

end winning_strategy_l1483_148325


namespace neg_pi_lt_neg_314_l1483_148389

theorem neg_pi_lt_neg_314 (h : Real.pi > 3.14) : -Real.pi < -3.14 :=
sorry

end neg_pi_lt_neg_314_l1483_148389


namespace find_k_l1483_148321

-- Definitions for arithmetic sequence properties
noncomputable def sum_arith_seq (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n-1) / 2) * d

noncomputable def term_arith_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Given Conditions
variables (a₁ d : ℝ)
variables (k : ℕ)

axiom sum_condition : sum_arith_seq a₁ d 9 = sum_arith_seq a₁ d 4
axiom term_condition : term_arith_seq a₁ d 4 + term_arith_seq a₁ d k = 0

-- Prove k = 10
theorem find_k : k = 10 :=
by
  sorry

end find_k_l1483_148321


namespace time_to_finish_work_with_both_tractors_l1483_148344

-- Definitions of given conditions
def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 15
def time_A_worked : ℚ := 13
def remaining_work : ℚ := 1 - (work_rate_A * time_A_worked)
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Statement that needs to be proven
theorem time_to_finish_work_with_both_tractors : 
  remaining_work / combined_work_rate = 3 :=
by
  sorry

end time_to_finish_work_with_both_tractors_l1483_148344


namespace find_m_range_l1483_148310

-- Definitions for the conditions and the required proof
def condition_alpha (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m + 7
def condition_beta (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

-- Proof problem translated to Lean 4 statement
theorem find_m_range (m : ℝ) :
  (∀ x, condition_beta x → condition_alpha m x) → (-2 ≤ m ∧ m ≤ 0) :=
by sorry

end find_m_range_l1483_148310


namespace cone_height_l1483_148346

theorem cone_height
  (V1 V2 V : ℝ)
  (h1 h2 : ℝ)
  (fact1 : h1 = 10)
  (fact2 : h2 = 2)
  (h : ∀ m : ℝ, V1 = V * (10 ^ 3) / (m ^ 3) ∧ V2 = V * ((m - 2) ^ 3) / (m ^ 3))
  (equal_volumes : V1 + V2 = V) :
  (∃ m : ℝ, m = 13.897) :=
by
  sorry

end cone_height_l1483_148346


namespace postage_problem_l1483_148384

theorem postage_problem (n : ℕ) (h_positive : n > 0) (h_postage : ∀ k, k ∈ List.range 121 → ∃ a b c : ℕ, 6 * a + n * b + (n + 2) * c = k) :
  6 * n * (n + 2) - (6 + n + (n + 2)) = 120 → n = 8 := 
by
  sorry

end postage_problem_l1483_148384


namespace corrected_mean_l1483_148351

/-- The original mean of 20 observations is 36, an observation of 25 was wrongly recorded as 40.
    The correct mean is 35.25. -/
theorem corrected_mean 
  (Mean : ℝ)
  (Observations : ℕ)
  (IncorrectObservation : ℝ)
  (CorrectObservation : ℝ)
  (h1 : Mean = 36)
  (h2 : Observations = 20)
  (h3 : IncorrectObservation = 40)
  (h4 : CorrectObservation = 25) :
  (Mean * Observations - (IncorrectObservation - CorrectObservation)) / Observations = 35.25 :=
sorry

end corrected_mean_l1483_148351


namespace sufficient_conditions_for_positive_product_l1483_148368

theorem sufficient_conditions_for_positive_product (a b : ℝ) :
  (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 1 ∧ b > 1) → a * b > 0 :=
by sorry

end sufficient_conditions_for_positive_product_l1483_148368


namespace gcf_48_160_120_l1483_148372

theorem gcf_48_160_120 : Nat.gcd (Nat.gcd 48 160) 120 = 8 := by
  sorry

end gcf_48_160_120_l1483_148372


namespace min_value_expression_l1483_148364

noncomputable def expression (x y : ℝ) := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x - 6 * y

theorem min_value_expression : ∀ x y : ℝ, expression x y ≥ -14 :=
by
  sorry

end min_value_expression_l1483_148364


namespace polynomial_evaluation_l1483_148379

theorem polynomial_evaluation (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 15) - 2 = -x^4 + 3*x^3 - 5*x^2 + 15*x - 2 :=
by
  sorry

end polynomial_evaluation_l1483_148379


namespace remainder_division_x_squared_minus_one_l1483_148391

variable (f g h : ℝ → ℝ)

noncomputable def remainder_when_divided_by_x_squared_minus_one (x : ℝ) : ℝ :=
-7 * x - 9

theorem remainder_division_x_squared_minus_one (h1 : ∀ x, f x = g x * (x - 1) + 8) (h2 : ∀ x, f x = h x * (x + 1) + 1) :
  ∀ x, f x % (x^2 - 1) = -7 * x - 9 :=
sorry

end remainder_division_x_squared_minus_one_l1483_148391
