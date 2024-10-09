import Mathlib

namespace algebraic_expr_value_l2284_228464

theorem algebraic_expr_value {a b : ℝ} (h: a + b = 1) : a^2 - b^2 + 2 * b + 9 = 10 := 
by
  sorry

end algebraic_expr_value_l2284_228464


namespace range_of_x_sqrt_4_2x_l2284_228486

theorem range_of_x_sqrt_4_2x (x : ℝ) : (4 - 2 * x ≥ 0) ↔ (x ≤ 2) :=
by
  sorry

end range_of_x_sqrt_4_2x_l2284_228486


namespace range_of_m_l2284_228436

def P (x : ℝ) : Prop := |(4 - x) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) (h : m > 0) : (∀ x, ¬P x → ¬q x m) → m ≥ 9 :=
by
  intros
  sorry

end range_of_m_l2284_228436


namespace james_pre_injury_miles_600_l2284_228412

-- Define the conditions
def james_pre_injury_miles (x : ℝ) : Prop :=
  ∃ goal_increase : ℝ, ∃ days : ℝ, ∃ weekly_increase : ℝ,
  goal_increase = 1.2 * x ∧
  days = 280 ∧
  weekly_increase = 3 ∧
  (days / 7) * weekly_increase = (goal_increase - x)

-- Define the main theorem to be proved
theorem james_pre_injury_miles_600 : james_pre_injury_miles 600 :=
sorry

end james_pre_injury_miles_600_l2284_228412


namespace sequence_term_2023_l2284_228435

theorem sequence_term_2023 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h : ∀ n, 2 * S n = a n * (a n + 1)) : 
  a 2023 = 2023 :=
sorry

end sequence_term_2023_l2284_228435


namespace factorize_x_squared_minus_121_l2284_228446

theorem factorize_x_squared_minus_121 (x : ℝ) : (x^2 - 121) = (x + 11) * (x - 11) :=
by
  sorry

end factorize_x_squared_minus_121_l2284_228446


namespace average_people_per_boat_correct_l2284_228426

-- Define number of boats and number of people
def num_boats := 3.0
def num_people := 5.0

-- Definition for average people per boat
def avg_people_per_boat := num_people / num_boats

-- Theorem to prove the average number of people per boat is 1.67
theorem average_people_per_boat_correct : avg_people_per_boat = 1.67 := by
  sorry

end average_people_per_boat_correct_l2284_228426


namespace total_amount_paid_l2284_228480

def original_price_per_card : Int := 12
def discount_per_card : Int := 2
def number_of_cards : Int := 10

theorem total_amount_paid :
  original_price_per_card - discount_per_card * number_of_cards = 100 :=
by
  sorry

end total_amount_paid_l2284_228480


namespace miles_left_to_reach_E_l2284_228448

-- Given conditions as definitions
def total_journey : ℕ := 2500
def miles_driven : ℕ := 642
def miles_B_to_C : ℕ := 400
def miles_C_to_D : ℕ := 550
def detour_D_to_E : ℕ := 200

-- Proof statement
theorem miles_left_to_reach_E : 
  (miles_B_to_C + miles_C_to_D + detour_D_to_E) = 1150 :=
by
  sorry

end miles_left_to_reach_E_l2284_228448


namespace original_price_of_petrol_l2284_228410

theorem original_price_of_petrol (P : ℝ) :
  (∃ P, 
    ∀ (GA GB GC : ℝ),
    0.8 * P = 0.8 * P ∧
    GA = 200 / P ∧
    GB = 300 / P ∧
    GC = 400 / P ∧
    200 = (GA + 8) * 0.8 * P ∧
    300 = (GB + 15) * 0.8 * P ∧
    400 = (GC + 22) * 0.8 * P) → 
  P = 6.25 :=
by
  sorry

end original_price_of_petrol_l2284_228410


namespace simplify_expr1_simplify_expr2_l2284_228445

theorem simplify_expr1 (a b : ℤ) : 2 * a - (4 * a + 5 * b) + 2 * (3 * a - 4 * b) = 4 * a - 13 * b :=
by sorry

theorem simplify_expr2 (x y : ℤ) : 5 * x^2 - 2 * (3 * y^2 - 5 * x^2) + (-4 * y^2 + 7 * x * y) = 15 * x^2 - 10 * y^2 + 7 * x * y :=
by sorry

end simplify_expr1_simplify_expr2_l2284_228445


namespace find_a_and_b_l2284_228462

theorem find_a_and_b (a b : ℚ) :
  ((∃ x y : ℚ, 3 * x - y = 7 ∧ a * x + y = b) ∧
   (∃ x y : ℚ, x + b * y = a ∧ 2 * x + y = 8)) →
  a = -7/5 ∧ b = -11/5 :=
by sorry

end find_a_and_b_l2284_228462


namespace original_price_of_shoes_l2284_228458

theorem original_price_of_shoes (P : ℝ) (h1 : 2 * 0.60 * P + 0.80 * 100 = 140) : P = 50 :=
by
  sorry

end original_price_of_shoes_l2284_228458


namespace algebraic_expression_zero_iff_x_eq_2_l2284_228447

theorem algebraic_expression_zero_iff_x_eq_2 (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (1 / (x - 1) + 3 / (1 - x^2) = 0) ↔ (x = 2) :=
by
  sorry

end algebraic_expression_zero_iff_x_eq_2_l2284_228447


namespace complement_intersection_l2284_228417

def set_P : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def set_Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem complement_intersection (P Q : Set ℝ) (hP : P = set_P) (hQ : Q = set_Q) :
  (Pᶜ ∩ Q) = {x | 1 < x ∧ x < 2} :=
by
  sorry

end complement_intersection_l2284_228417


namespace sum_of_digits_is_twenty_l2284_228457

theorem sum_of_digits_is_twenty (a b c d : ℕ) (h1 : c + b = 9) (h2 : a + d = 10) 
  (H1 : a ≠ b) (H2 : a ≠ c) (H3 : a ≠ d) 
  (H4 : b ≠ c) (H5 : b ≠ d) (H6 : c ≠ d) :
  a + b + c + d = 20 := 
sorry

end sum_of_digits_is_twenty_l2284_228457


namespace solve_for_x_l2284_228474

theorem solve_for_x (x : ℝ) (h : |x - 2| = |x - 3| + 1) : x = 3 :=
by
  sorry

end solve_for_x_l2284_228474


namespace initial_percentage_of_milk_l2284_228473

theorem initial_percentage_of_milk (M : ℝ) (H1 : M / 100 * 60 = 0.58 * 86.9) : M = 83.99 :=
by
  sorry

end initial_percentage_of_milk_l2284_228473


namespace lines_intersection_l2284_228430

theorem lines_intersection :
  ∃ (x y : ℝ), 
  (3 * y = -2 * x + 6) ∧ 
  (-4 * y = 3 * x + 4) ∧ 
  (x = -36) ∧ 
  (y = 26) :=
sorry

end lines_intersection_l2284_228430


namespace solution_set_for_inequality_l2284_228418

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

end solution_set_for_inequality_l2284_228418


namespace k_value_for_inequality_l2284_228488

theorem k_value_for_inequality :
    (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d)) ∧
    (∀ k : ℝ, (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d)) → k = 3/4) :=
sorry

end k_value_for_inequality_l2284_228488


namespace comb_product_l2284_228403

theorem comb_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 :=
by
  sorry

end comb_product_l2284_228403


namespace find_a_perpendicular_lines_l2284_228478

theorem find_a_perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, ax + (a + 2) * y + 1 = 0 ∧ x + a * y + 2 = 0) → a = -3 :=
sorry

end find_a_perpendicular_lines_l2284_228478


namespace intersection_A_B_l2284_228404

-- Definitions of sets A and B
def A : Set ℕ := {2, 3, 5, 7}
def B : Set ℕ := {1, 2, 3, 5, 8}

-- Prove that the intersection of sets A and B is {2, 3, 5}
theorem intersection_A_B :
  A ∩ B = {2, 3, 5} :=
sorry

end intersection_A_B_l2284_228404


namespace scientific_notation_of_million_l2284_228460

theorem scientific_notation_of_million (x : ℝ) (h : x = 56.99) : 56.99 * 10^6 = 5.699 * 10^7 :=
by
  sorry

end scientific_notation_of_million_l2284_228460


namespace total_apples_picked_l2284_228453

def number_of_children : Nat := 33
def apples_per_child : Nat := 10
def number_of_adults : Nat := 40
def apples_per_adult : Nat := 3

theorem total_apples_picked :
  (number_of_children * apples_per_child) + (number_of_adults * apples_per_adult) = 450 := by
  -- You need to provide proof here
  sorry

end total_apples_picked_l2284_228453


namespace roots_of_equation_l2284_228437

theorem roots_of_equation (
  x y: ℝ
) (h1: x + y = 10) (h2: |x - y| = 12):
  (x = 11 ∧ y = -1) ∨ (x = -1 ∧ y = 11) ↔ ∃ (a b: ℝ), a = 11 ∧ b = -1 ∨ a = -1 ∧ b = 11 ∧ a^2 - 10*a - 22 = 0 ∧ b^2 - 10*b - 22 = 0 := 
by sorry

end roots_of_equation_l2284_228437


namespace min_fraction_value_l2284_228471

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 2

theorem min_fraction_value : ∀ x ∈ (Set.Ici (7 / 4)), (f x)^2 + 2 / (f x) ≥ 81 / 28 :=
by
  sorry

end min_fraction_value_l2284_228471


namespace crayons_total_l2284_228482

theorem crayons_total (rows : ℕ) (crayons_per_row : ℕ) (total_crayons : ℕ) :
  rows = 15 → crayons_per_row = 42 → total_crayons = rows * crayons_per_row → total_crayons = 630 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end crayons_total_l2284_228482


namespace increased_contact_area_effect_l2284_228423

-- Define the conditions as assumptions
theorem increased_contact_area_effect (k : ℝ) (A₁ A₂ : ℝ) (dTdx : ℝ) (Q₁ Q₂ : ℝ) :
  (A₂ > A₁) →
  (Q₁ = -k * A₁ * dTdx) →
  (Q₂ = -k * A₂ * dTdx) →
  (Q₂ > Q₁) →
  ∃ increased_sensation : Prop, increased_sensation :=
by 
  exfalso
  sorry

end increased_contact_area_effect_l2284_228423


namespace boxes_of_apples_with_cherries_l2284_228495

-- Define everything in the conditions
variable (A P Sp Sa : ℕ)
variable (box_cherries box_apples : ℕ)

-- Given conditions
axiom price_relation : 2 * P = 3 * A
axiom size_relation  : Sa = 12 * Sp
axiom cherries_per_box : box_cherries = 12

-- The problem statement (to be proved)
theorem boxes_of_apples_with_cherries : box_apples * A = box_cherries * P → box_apples = 18 :=
by
  sorry

end boxes_of_apples_with_cherries_l2284_228495


namespace compare_answers_l2284_228407

def num : ℕ := 384
def correct_answer : ℕ := (5 * num) / 16
def students_answer : ℕ := (5 * num) / 6
def difference : ℕ := students_answer - correct_answer

theorem compare_answers : difference = 200 := 
by
  sorry

end compare_answers_l2284_228407


namespace fruit_salad_cherries_l2284_228483

theorem fruit_salad_cherries (b r g c : ℕ) 
  (h1 : b + r + g + c = 390)
  (h2 : r = 3 * b)
  (h3 : g = 2 * c)
  (h4 : c = 5 * r) :
  c = 119 :=
by
  sorry

end fruit_salad_cherries_l2284_228483


namespace milton_books_l2284_228413

variable (z b : ℕ)

theorem milton_books (h₁ : z + b = 80) (h₂ : b = 4 * z) : z = 16 :=
by
  sorry

end milton_books_l2284_228413


namespace total_students_correct_l2284_228405

-- Given conditions
def number_of_buses : ℕ := 95
def number_of_seats_per_bus : ℕ := 118

-- Definition for the total number of students
def total_number_of_students : ℕ := number_of_buses * number_of_seats_per_bus

-- Problem statement
theorem total_students_correct :
  total_number_of_students = 11210 :=
by
  -- Proof is omitted, hence we use sorry.
  sorry

end total_students_correct_l2284_228405


namespace calculate_possible_change_l2284_228402

structure ChangeProblem where
  (change : ℕ)
  (h1 : change < 100)
  (h2 : ∃ (q : ℕ), change = 25 * q + 10 ∧ q ≤ 3)
  (h3 : ∃ (d : ℕ), change = 10 * d + 20 ∧ d ≤ 9)

theorem calculate_possible_change (p1 p2 p3 p4 : ChangeProblem) :
  p1.change + p2.change + p3.change = 180 :=
by
  sorry

end calculate_possible_change_l2284_228402


namespace probability_top_card_special_l2284_228433

-- Definition of the problem conditions
def deck_size : ℕ := 52
def special_card_count : ℕ := 16

-- The statement we need to prove
theorem probability_top_card_special : 
  (special_card_count : ℚ) / deck_size = 4 / 13 := 
  by sorry

end probability_top_card_special_l2284_228433


namespace total_spent_on_pens_l2284_228467

/-- Dorothy, Julia, and Robert go to the store to buy school supplies.
    Dorothy buys half as many pens as Julia.
    Julia buys three times as many pens as Robert.
    Robert buys 4 pens.
    The cost of one pen is $1.50.
    Prove that the total amount of money spent on pens by the three friends is $33. 
-/
theorem total_spent_on_pens :
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  total_pens * cost_per_pen = 33 := 
by
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  sorry

end total_spent_on_pens_l2284_228467


namespace mayor_vice_mayor_happy_people_l2284_228481

theorem mayor_vice_mayor_happy_people :
  (∃ (institutions_per_institution : ℕ) (num_institutions : ℕ),
    institutions_per_institution = 80 ∧
    num_institutions = 6 ∧
    num_institutions * institutions_per_institution = 480) :=
by
  sorry

end mayor_vice_mayor_happy_people_l2284_228481


namespace mixed_oil_rate_l2284_228493

/-- Given quantities and prices of three types of oils, any combination
that satisfies the volume and price conditions will achieve a final mixture rate of Rs. 65 per litre. -/
theorem mixed_oil_rate (x y z : ℝ) : 
  12.5 * 55 + 7.75 * 70 + 3.25 * 82 = 1496.5 ∧ 12.5 + 7.75 + 3.25 = 23.5 →
  x + y + z = 23.5 ∧ 55 * x + 70 * y + 82 * z = 65 * 23.5 →
  true :=
by
  intros h1 h2
  sorry

end mixed_oil_rate_l2284_228493


namespace bakery_flour_total_l2284_228477

theorem bakery_flour_total :
  (0.2 + 0.1 + 0.15 + 0.05 + 0.1 = 0.6) :=
by {
  sorry
}

end bakery_flour_total_l2284_228477


namespace dice_probability_abs_diff_2_l2284_228414

theorem dice_probability_abs_diff_2 :
  let total_outcomes := 36
  let favorable_outcomes := 8
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end dice_probability_abs_diff_2_l2284_228414


namespace calculate_expression_l2284_228421

theorem calculate_expression : |(-5 : ℤ)| + (1 / 3 : ℝ)⁻¹ - (Real.pi - 2) ^ 0 = 7 := by
  sorry

end calculate_expression_l2284_228421


namespace bounded_harmonic_is_constant_l2284_228476

noncomputable def is_harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ (x y : ℤ), f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1) = 4 * f (x, y)

theorem bounded_harmonic_is_constant (f : ℤ × ℤ → ℝ) (M : ℝ) 
  (h_bound : ∀ (x y : ℤ), |f (x, y)| ≤ M)
  (h_harmonic : is_harmonic f) :
  ∃ c : ℝ, ∀ x y : ℤ, f (x, y) = c :=
sorry

end bounded_harmonic_is_constant_l2284_228476


namespace log_defined_for_powers_of_a_if_integer_exponents_log_undefined_if_only_positive_indices_l2284_228422

variable (a : ℝ) (b : ℝ)

-- Conditions
axiom base_pos (h : a > 0) : a ≠ 1
axiom integer_exponents_only (h : ∃ n : ℤ, b = a^n) : True
axiom positive_indices_only (h : ∃ n : ℕ, b = a^n) : 0 < b ∧ b < 1 → False

-- Theorem: If we only knew integer exponents, the logarithm of any number b in base a is defined for powers of a.
theorem log_defined_for_powers_of_a_if_integer_exponents (h : ∃ n : ℤ, b = a^n) : True :=
by sorry

-- Theorem: If we only knew positive exponents, the logarithm of any number b in base a is undefined for all 0 < b < 1
theorem log_undefined_if_only_positive_indices : (∃ n : ℕ, b = a^n) → (0 < b ∧ b < 1 → False) :=
by sorry

end log_defined_for_powers_of_a_if_integer_exponents_log_undefined_if_only_positive_indices_l2284_228422


namespace possible_lengths_of_c_l2284_228485

-- Definitions of the given conditions
variables (a b c : ℝ) (S : ℝ)
variables (h₁ : a = 4)
variables (h₂ : b = 5)
variables (h₃ : S = 5 * Real.sqrt 3)

-- The main theorem stating the possible lengths of c
theorem possible_lengths_of_c : c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
  sorry

end possible_lengths_of_c_l2284_228485


namespace expected_games_is_correct_l2284_228424

def prob_A_wins : ℚ := 2 / 3
def prob_B_wins : ℚ := 1 / 3
def max_games : ℕ := 6

noncomputable def expected_games : ℚ :=
  2 * (prob_A_wins^2 + prob_B_wins^2) +
  4 * (prob_A_wins * prob_B_wins * (prob_A_wins^2 + prob_B_wins^2)) +
  6 * (prob_A_wins * prob_B_wins)^2

theorem expected_games_is_correct : expected_games = 266 / 81 := by
  sorry

end expected_games_is_correct_l2284_228424


namespace luke_piles_coins_l2284_228431

theorem luke_piles_coins (x : ℕ) (h_total_piles : 10 = 5 + 5) (h_total_coins : 10 * x = 30) :
  x = 3 :=
by
  sorry

end luke_piles_coins_l2284_228431


namespace cornbread_pieces_l2284_228461

theorem cornbread_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ)
  (hl : pan_length = 20) (hw : pan_width = 18) (hp : piece_length = 2) (hq : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 90 :=
by
  sorry

end cornbread_pieces_l2284_228461


namespace unit_cost_of_cranberry_juice_l2284_228408

theorem unit_cost_of_cranberry_juice (total_cost : ℕ) (ounces : ℕ) (h1 : total_cost = 84) (h2 : ounces = 12) :
  total_cost / ounces = 7 :=
by
  sorry

end unit_cost_of_cranberry_juice_l2284_228408


namespace find_x_minus_y_l2284_228442

theorem find_x_minus_y (x y : ℝ) (h1 : |x| + x - y = 14) (h2 : x + |y| + y = 6) : x - y = 8 :=
sorry

end find_x_minus_y_l2284_228442


namespace ylona_initial_bands_l2284_228411

variable (B J Y : ℕ)  -- Represents the initial number of rubber bands for Bailey, Justine, and Ylona respectively

-- Define the conditions
axiom h1 : J = B + 10
axiom h2 : J = Y - 2
axiom h3 : B - 4 = 8

-- Formulate the statement
theorem ylona_initial_bands : Y = 24 :=
by
  sorry

end ylona_initial_bands_l2284_228411


namespace value_of_expression_l2284_228439

variables {a b c d e f : ℝ}

theorem value_of_expression
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 / 4 :=
sorry

end value_of_expression_l2284_228439


namespace quadratic_coefficients_l2284_228469

theorem quadratic_coefficients : 
  ∀ (b k : ℝ), (∀ x : ℝ, x^2 + b * x + 5 = (x - 2)^2 + k) → b = -4 ∧ k = 1 :=
by
  intro b k h
  have h1 := h 0
  have h2 := h 1
  sorry

end quadratic_coefficients_l2284_228469


namespace num_factors_180_l2284_228487

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end num_factors_180_l2284_228487


namespace quadratic_roots_l2284_228434

theorem quadratic_roots (a b: ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0)
  (root_condition1 : a * (-1/2)^2 + b * (-1/2) + 2 = 0)
  (root_condition2 : a * (1/3)^2 + b * (1/3) + 2 = 0) 
  : a - b = -10 := 
by {
  sorry
}

end quadratic_roots_l2284_228434


namespace one_fourth_of_56_equals_75_l2284_228484

theorem one_fourth_of_56_equals_75 : (5.6 / 4) = 7 / 5 := 
by
  -- Temporarily omitting the actual proof
  sorry

end one_fourth_of_56_equals_75_l2284_228484


namespace age_problem_l2284_228498

open Classical

variable (A B C : ℕ)

theorem age_problem (h1 : A + 10 = 2 * (B - 10))
                    (h2 : C = 3 * (A - 5))
                    (h3 : A = B + 9)
                    (h4 : C = A + 4) :
  B = 39 :=
sorry

end age_problem_l2284_228498


namespace trigonometric_identity_l2284_228451

theorem trigonometric_identity :
  (Real.cos (17 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) + 
   Real.sin (163 * Real.pi / 180) * Real.sin (47 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trigonometric_identity_l2284_228451


namespace cookies_last_days_l2284_228450

variable (c1 c2 t : ℕ)

/-- Jackson's oldest son gets 4 cookies after school each day, and his youngest son gets 2 cookies. 
There are 54 cookies in the box, so the number of days the box will last is 9. -/
theorem cookies_last_days (h1 : c1 = 4) (h2 : c2 = 2) (h3 : t = 54) : 
  t / (c1 + c2) = 9 := by
  sorry

end cookies_last_days_l2284_228450


namespace correct_option_l2284_228420

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem correct_option : M ∪ (U \ N) = U :=
by
  sorry

end correct_option_l2284_228420


namespace hiker_total_distance_l2284_228432

-- Define conditions based on the problem description
def day1_distance : ℕ := 18
def day1_speed : ℕ := 3
def day2_speed : ℕ := day1_speed + 1
def day1_time : ℕ := day1_distance / day1_speed
def day2_time : ℕ := day1_time - 1
def day3_speed : ℕ := 5
def day3_time : ℕ := 3

-- Define the total distance walked based on the conditions
def total_distance : ℕ :=
  day1_distance + (day2_speed * day2_time) + (day3_speed * day3_time)

-- The theorem stating the hiker walked a total of 53 miles
theorem hiker_total_distance : total_distance = 53 := by
  sorry

end hiker_total_distance_l2284_228432


namespace solution_set_for_inequality_l2284_228440

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^2 - 2*x - 5

theorem solution_set_for_inequality :
  {x : ℝ | f x >= -2} = {x | -2 <= x ∧ x < 1 ∨ x >= 3} := sorry

end solution_set_for_inequality_l2284_228440


namespace cylinder_original_radius_l2284_228468

theorem cylinder_original_radius
    (r h: ℝ)
    (h₀: h = 4)
    (h₁: π * (r + 8)^2 * 4 = π * r^2 * 12) :
    r = 12 :=
by
  -- Insert your proof here
  sorry

end cylinder_original_radius_l2284_228468


namespace percentage_greater_than_l2284_228465

-- Definitions of the variables involved
variables (X Y Z : ℝ)

-- Lean statement to prove the formula
theorem percentage_greater_than (X Y Z : ℝ) : 
  (100 * (X - Y)) / (Y + Z) = (100 * (X - Y)) / (Y + Z) :=
by
  -- skipping the actual proof
  sorry

end percentage_greater_than_l2284_228465


namespace boat_upstream_time_is_1_5_hours_l2284_228452

noncomputable def time_to_cover_distance_upstream
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ) : ℝ :=
  distance_downstream / (speed_boat_still_water - speed_stream)

theorem boat_upstream_time_is_1_5_hours
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (downstream_distance : ℝ)
  (h1 : speed_stream = 3)
  (h2 : speed_boat_still_water = 15)
  (h3 : time_downstream = 1)
  (h4 : downstream_distance = speed_boat_still_water + speed_stream) :
  time_to_cover_distance_upstream speed_stream speed_boat_still_water time_downstream downstream_distance = 1.5 :=
by
  sorry

end boat_upstream_time_is_1_5_hours_l2284_228452


namespace terminating_decimal_count_l2284_228443

def count_terminating_decimals (n: ℕ): ℕ :=
  (n / 17)

theorem terminating_decimal_count : count_terminating_decimals 493 = 29 := by
  sorry

end terminating_decimal_count_l2284_228443


namespace exact_days_two_friends_visit_l2284_228444

-- Define the periodicities of Alice, Beatrix, and Claire
def periodicity_alice : ℕ := 1
def periodicity_beatrix : ℕ := 5
def periodicity_claire : ℕ := 7

-- Define the total days to be considered
def total_days : ℕ := 180

-- Define the number of days three friends visit together
def lcm_ab := Nat.lcm periodicity_alice periodicity_beatrix
def lcm_ac := Nat.lcm periodicity_alice periodicity_claire
def lcm_bc := Nat.lcm periodicity_beatrix periodicity_claire
def lcm_abc := Nat.lcm lcm_ab periodicity_claire

-- Define the counts of visitations
def count_ab := total_days / lcm_ab - total_days / lcm_abc
def count_ac := total_days / lcm_ac - total_days / lcm_abc
def count_bc := total_days / lcm_bc - total_days / lcm_abc

-- Finally calculate the number of days exactly two friends visit together
def days_two_friends_visit : ℕ := count_ab + count_ac + count_bc

-- The theorem to prove
theorem exact_days_two_friends_visit : days_two_friends_visit = 51 :=
by 
  -- This is where the actual proof would go
  sorry

end exact_days_two_friends_visit_l2284_228444


namespace algebraic_identity_l2284_228479

theorem algebraic_identity (a b c d : ℝ) : a - b + c - d = a + c - (b + d) :=
by
  sorry

end algebraic_identity_l2284_228479


namespace smallest_n_inverse_mod_1176_l2284_228428

theorem smallest_n_inverse_mod_1176 : ∃ n : ℕ, n > 1 ∧ Nat.Coprime n 1176 ∧ (∀ m : ℕ, m > 1 ∧ Nat.Coprime m 1176 → n ≤ m) ∧ n = 5 := by
  sorry

end smallest_n_inverse_mod_1176_l2284_228428


namespace sum_of_squares_not_perfect_square_l2284_228459

theorem sum_of_squares_not_perfect_square (n : ℤ) : ¬ (∃ k : ℤ, k^2 = (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2) :=
by
  sorry

end sum_of_squares_not_perfect_square_l2284_228459


namespace area_of_tangent_triangle_l2284_228463

noncomputable def tangentTriangleArea : ℝ :=
  let y := λ x : ℝ => x^3 + x
  let dy := λ x : ℝ => 3 * x^2 + 1
  let slope := dy 1
  let y_intercept := 2 - slope * 1
  let x_intercept := - y_intercept / slope
  let base := x_intercept
  let height := - y_intercept
  0.5 * base * height

theorem area_of_tangent_triangle :
  tangentTriangleArea = 1 / 2 :=
by
  sorry

end area_of_tangent_triangle_l2284_228463


namespace alan_carla_weight_l2284_228449

variable (a b c d : ℝ)

theorem alan_carla_weight (h1 : a + b = 280) (h2 : b + c = 230) (h3 : c + d = 250) (h4 : a + d = 300) :
  a + c = 250 := by
sorry

end alan_carla_weight_l2284_228449


namespace juice_expense_l2284_228456

theorem juice_expense (M P : ℕ) 
  (h1 : M + P = 17) 
  (h2 : 5 * M + 6 * P = 94) : 6 * P = 54 :=
by 
  sorry

end juice_expense_l2284_228456


namespace ted_candy_bars_l2284_228466

theorem ted_candy_bars (b : ℕ) (n : ℕ) (h : b = 5) (h2 : n = 3) : b * n = 15 :=
by
  sorry

end ted_candy_bars_l2284_228466


namespace total_items_18_l2284_228494

-- Define the number of dogs, biscuits per dog, and boots per set
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 5
def boots_per_set : ℕ := 4

-- Calculate the total number of items
def total_items (num_dogs biscuits_per_dog boots_per_set : ℕ) : ℕ :=
  (num_dogs * biscuits_per_dog) + (num_dogs * boots_per_set)

-- Prove that the total number of items is 18
theorem total_items_18 : total_items num_dogs biscuits_per_dog boots_per_set = 18 := by
  -- Proof is not provided
  sorry

end total_items_18_l2284_228494


namespace factorize_expression_l2284_228454

variable {R : Type} [Ring R]
variables (a b x y : R)

theorem factorize_expression :
  8 * a * x - b * y + 4 * a * y - 2 * b * x = (4 * a - b) * (2 * x + y) :=
sorry

end factorize_expression_l2284_228454


namespace preimages_of_one_under_f_l2284_228499

theorem preimages_of_one_under_f :
  {x : ℝ | (x^3 - x + 1 = 1)} = {-1, 0, 1} := by
  sorry

end preimages_of_one_under_f_l2284_228499


namespace slope_of_parallel_line_l2284_228416

/-- A line is described by the equation 3x - 6y = 12. The slope of a line 
    parallel to this line is 1/2. -/
theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1/2 := by
  sorry

end slope_of_parallel_line_l2284_228416


namespace bertha_gave_away_balls_l2284_228409

def balls_initial := 2
def balls_worn_out := 20 / 10
def balls_lost := 20 / 5
def balls_purchased := (20 / 4) * 3
def balls_after_20_games_without_giveaway := balls_initial - balls_worn_out - balls_lost + balls_purchased
def balls_after_20_games := 10

theorem bertha_gave_away_balls : balls_after_20_games_without_giveaway - balls_after_20_games = 1 := by
  sorry

end bertha_gave_away_balls_l2284_228409


namespace factorial_expression_l2284_228489

open Nat

theorem factorial_expression : ((sqrt (5! * 4!)) ^ 2 + 3!) = 2886 := by
  sorry

end factorial_expression_l2284_228489


namespace slope_of_line_between_intersections_of_circles_l2284_228497

theorem slope_of_line_between_intersections_of_circles :
  ∀ C D : ℝ × ℝ, 
    -- Conditions: equations of the circles
    (C.1^2 + C.2^2 - 6 * C.1 + 4 * C.2 - 8 = 0) ∧ (C.1^2 + C.2^2 - 8 * C.1 - 2 * C.2 + 10 = 0) →
    (D.1^2 + D.2^2 - 6 * D.1 + 4 * D.2 - 8 = 0) ∧ (D.1^2 + D.2^2 - 8 * D.1 - 2 * D.2 + 10 = 0) →
    -- Question: slope of line CD
    ((C.2 - D.2) / (C.1 - D.1) = -1 / 3) :=
by
  sorry

end slope_of_line_between_intersections_of_circles_l2284_228497


namespace nested_function_evaluation_l2284_228400

def f (x : ℕ) : ℕ := x + 3
def g (x : ℕ) : ℕ := x / 2
def f_inv (x : ℕ) : ℕ := x - 3
def g_inv (x : ℕ) : ℕ := 2 * x

theorem nested_function_evaluation : 
  f (g_inv (f_inv (g_inv (f_inv (g (f 15)))))) = 21 := 
by 
  sorry

end nested_function_evaluation_l2284_228400


namespace max_perimeter_of_triangle_l2284_228401

theorem max_perimeter_of_triangle (A B C a b c p : ℝ) 
  (h_angle_A : A = 2 * Real.pi / 3)
  (h_a : a = 3)
  (h_perimeter : p = a + b + c) 
  (h_sine_law : b = 2 * Real.sqrt 3 * Real.sin B ∧ c = 2 * Real.sqrt 3 * Real.sin C) :
  p ≤ 3 + 2 * Real.sqrt 3 :=
sorry

end max_perimeter_of_triangle_l2284_228401


namespace tangent_line_y_intercept_l2284_228475

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

theorem tangent_line_y_intercept (a : ℝ) (h : 3 * (1:ℝ)^2 - a = 1) :
  (∃ (m b : ℝ), ∀ (x : ℝ), m = 1 ∧ y = x - 2 → y = m * x + b) := 
 by
  sorry

end tangent_line_y_intercept_l2284_228475


namespace returning_players_l2284_228441

-- Definitions of conditions
def num_groups : Nat := 9
def players_per_group : Nat := 6
def new_players : Nat := 48

-- Definition of total number of players
def total_players : Nat := num_groups * players_per_group

-- Theorem: Find the number of returning players
theorem returning_players :
  total_players - new_players = 6 :=
by
  sorry

end returning_players_l2284_228441


namespace number_of_different_duty_schedules_l2284_228425

-- Define a structure for students
inductive Student
| A | B | C

-- Define days of the week excluding Sunday as all duties are from Monday to Saturday
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Define the conditions in Lean
def condition_A_does_not_take_Monday (schedules : Day → Student) : Prop :=
  schedules Day.Monday ≠ Student.A

def condition_B_does_not_take_Saturday (schedules : Day → Student) : Prop :=
  schedules Day.Saturday ≠ Student.B

-- Define the function to count valid schedules
noncomputable def count_valid_schedules : ℕ :=
  sorry  -- This would be the computation considering combinatorics

-- Theorem statement to prove the correct answer
theorem number_of_different_duty_schedules 
    (schedules : Day → Student)
    (h1 : condition_A_does_not_take_Monday schedules)
    (h2 : condition_B_does_not_take_Saturday schedules)
    : count_valid_schedules = 42 :=
sorry

end number_of_different_duty_schedules_l2284_228425


namespace Alan_ate_1_fewer_pretzel_than_John_l2284_228429

/-- Given that there are 95 pretzels in a bowl, John ate 28 pretzels, 
Marcus ate 12 more pretzels than John, and Marcus ate 40 pretzels,
prove that Alan ate 1 fewer pretzel than John. -/
theorem Alan_ate_1_fewer_pretzel_than_John 
  (h95 : 95 = 95)
  (John_ate : 28 = 28)
  (Marcus_ate_more : ∀ (x : ℕ), 40 = x + 12 → x = 28)
  (Marcus_ate : 40 = 40) :
  ∃ (Alan : ℕ), Alan = 27 ∧ 28 - Alan = 1 :=
by
  sorry

end Alan_ate_1_fewer_pretzel_than_John_l2284_228429


namespace smaller_prime_factor_l2284_228406

theorem smaller_prime_factor (a b : ℕ) (prime_a : Nat.Prime a) (prime_b : Nat.Prime b) (distinct : a ≠ b)
  (product : a * b = 316990099009901) :
  min a b = 4002001 :=
  sorry

end smaller_prime_factor_l2284_228406


namespace june_earnings_l2284_228496

theorem june_earnings 
    (total_clovers : ℕ := 300)
    (pct_3_petals : ℕ := 70)
    (pct_2_petals : ℕ := 20)
    (pct_4_petals : ℕ := 8)
    (pct_5_petals : ℕ := 2)
    (earn_3_petals : ℕ := 1)
    (earn_2_petals : ℕ := 2)
    (earn_4_petals : ℕ := 5)
    (earn_5_petals : ℕ := 10)
    (earn_total : ℕ := 510) : 
  (pct_3_petals * total_clovers) / 100 * earn_3_petals + 
  (pct_2_petals * total_clovers) / 100 * earn_2_petals + 
  (pct_4_petals * total_clovers) / 100 * earn_4_petals + 
  (pct_5_petals * total_clovers) / 100 * earn_5_petals = earn_total := 
by
  -- Proof of this theorem involves calculating each part and summing them. Skipping detailed steps with sorry.
  sorry

end june_earnings_l2284_228496


namespace arctan_sum_l2284_228490

theorem arctan_sum : 
  let x := (3 : ℝ) / 7
  let y := 7 / 3
  x * y = 1 → (Real.arctan x + Real.arctan y = Real.pi / 2) :=
by
  intros x y h
  -- Proof goes here
  sorry

end arctan_sum_l2284_228490


namespace slices_all_three_toppings_l2284_228419

def slices_with_all_toppings (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ) : ℕ := 
  (12 : ℕ)

theorem slices_all_three_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (h : total_slices = 24)
  (h1 : pepperoni_slices = 12)
  (h2 : mushroom_slices = 14)
  (h3 : olive_slices = 16)
  (hc : total_slices ≥ 0)
  (hc1 : pepperoni_slices ≥ 0)
  (hc2 : mushroom_slices ≥ 0)
  (hc3 : olive_slices ≥ 0) :
  slices_with_all_toppings total_slices pepperoni_slices mushroom_slices olive_slices = 2 :=
  sorry

end slices_all_three_toppings_l2284_228419


namespace calculation_l2284_228492

theorem calculation :
  12 - 10 + 8 / 2 * 5 + 4 - 6 * 3 + 1 = 9 :=
by
  sorry

end calculation_l2284_228492


namespace kiran_currency_notes_l2284_228470

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

end kiran_currency_notes_l2284_228470


namespace two_pow_geq_n_cubed_for_n_geq_ten_l2284_228491

theorem two_pow_geq_n_cubed_for_n_geq_ten (n : ℕ) (hn : n ≥ 10) : 2^n ≥ n^3 := 
sorry

end two_pow_geq_n_cubed_for_n_geq_ten_l2284_228491


namespace percent_increase_l2284_228438

-- Definitions based on conditions
def initial_price : ℝ := 10
def final_price : ℝ := 15

-- Goal: Prove that the percent increase in the price per share is 50%
theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 50 := 
by
  sorry  -- Proof is not required, so we skip it with sorry.

end percent_increase_l2284_228438


namespace min_value_proof_l2284_228472

noncomputable def min_value (t c : ℝ) :=
  (t^2 + c^2 - 2 * t * c + 2 * c^2) / 2

theorem min_value_proof (a b t c : ℝ) (h : a + b = t) :
  (a^2 + (b + c)^2) ≥ min_value t c :=
by
  sorry

end min_value_proof_l2284_228472


namespace car_and_bus_speeds_l2284_228415

-- Definitions of given conditions
def car_speed : ℕ := 44
def bus_speed : ℕ := 52

-- Definition of total distance after 4 hours
def total_distance (car_speed bus_speed : ℕ) := 4 * car_speed + 4 * bus_speed

-- Definition of fact that cars started from the same point and traveled in opposite directions
def cars_from_same_point (car_speed bus_speed : ℕ) := car_speed + bus_speed

theorem car_and_bus_speeds :
  total_distance car_speed (car_speed + 8) = 384 :=
by
  -- Proof constructed based on the conditions given
  sorry

end car_and_bus_speeds_l2284_228415


namespace largest_square_area_l2284_228455

theorem largest_square_area (XY YZ XZ : ℝ)
  (h1 : XZ^2 = XY^2 + YZ^2)
  (h2 : XY^2 + YZ^2 + XZ^2 = 450) :
  XZ^2 = 225 :=
by
  sorry

end largest_square_area_l2284_228455


namespace city_schools_count_l2284_228427

theorem city_schools_count (a b c : ℕ) (schools : ℕ) : 
  b = 40 → c = 51 → b < a → a < c → 
  (a > b ∧ a < c ∧ (a - 1) * 3 < (c - b + 1) * 3 + 1) → 
  schools = (c - 1) / 3 :=
by
  sorry

end city_schools_count_l2284_228427
