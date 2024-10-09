import Mathlib

namespace find_a_l2395_239588

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a (a : ℝ) (h : ∃ x, f x a = 3) : a = 1 ∨ a = 7 := 
sorry

end find_a_l2395_239588


namespace infinite_alternating_parity_l2395_239519

theorem infinite_alternating_parity (m : ℕ) : ∃ᶠ n in at_top, 
  ∀ i < m, ((5^n / 10^i) % 2) ≠ (((5^n / 10^(i+1)) % 10) % 2) :=
sorry

end infinite_alternating_parity_l2395_239519


namespace tan_diff_eqn_l2395_239554

theorem tan_diff_eqn (α : ℝ) (h : Real.tan α = 2) : Real.tan (α - 3 * Real.pi / 4) = -3 := 
by 
  sorry

end tan_diff_eqn_l2395_239554


namespace days_worked_prove_l2395_239535

/-- Work rate of A is 1/15 work per day -/
def work_rate_A : ℚ := 1/15

/-- Work rate of B is 1/20 work per day -/
def work_rate_B : ℚ := 1/20

/-- Combined work rate of A and B -/
def combined_work_rate : ℚ := work_rate_A + work_rate_B

/-- Fraction of work left after some days -/
def fraction_work_left : ℚ := 8/15

/-- Fraction of work completed after some days -/
def fraction_work_completed : ℚ := 1 - fraction_work_left

/-- Number of days A and B worked together -/
def days_worked_together : ℚ := fraction_work_completed / combined_work_rate

theorem days_worked_prove : 
    days_worked_together = 4 := 
by 
    sorry

end days_worked_prove_l2395_239535


namespace original_cost_111_l2395_239547

theorem original_cost_111 (P : ℝ) (h1 : 0.76 * P * 0.90 = 760) : P = 111 :=
by sorry

end original_cost_111_l2395_239547


namespace find_b_over_a_l2395_239512

variables {a b c : ℝ}
variables {b₃ b₇ b₁₁ : ℝ}

-- Conditions
def roots_of_quadratic (a b c b₃ b₁₁ : ℝ) : Prop :=
  ∃ p q, p + q = -b / a ∧ p * q = c / a ∧ (p = b₃ ∨ p = b₁₁) ∧ (q = b₃ ∨ q = b₁₁)

def middle_term_value (b₇ : ℝ) : Prop :=
  b₇ = 3

-- The statement to be proved
theorem find_b_over_a
  (h1 : roots_of_quadratic a b c b₃ b₁₁)
  (h2 : middle_term_value b₇)
  (h3 : b₃ + b₁₁ = 2 * b₇) :
  b / a = -6 :=
sorry

end find_b_over_a_l2395_239512


namespace passes_to_left_l2395_239536

theorem passes_to_left (total_passes right_passes center_passes left_passes : ℕ)
  (h_total : total_passes = 50)
  (h_right : right_passes = 2 * left_passes)
  (h_center : center_passes = left_passes + 2)
  (h_sum : left_passes + right_passes + center_passes = total_passes) :
  left_passes = 12 := 
by
  sorry

end passes_to_left_l2395_239536


namespace time_spent_cutting_hair_l2395_239506

theorem time_spent_cutting_hair :
  let women's_time := 50
  let men's_time := 15
  let children's_time := 25
  let women's_haircuts := 3
  let men's_haircuts := 2
  let children's_haircuts := 3
  women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255 :=
by
  -- Definitions
  let women's_time       := 50
  let men's_time         := 15
  let children's_time    := 25
  let women's_haircuts   := 3
  let men's_haircuts     := 2
  let children's_haircuts := 3
  
  show women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255
  sorry

end time_spent_cutting_hair_l2395_239506


namespace power_mod_remainder_l2395_239556

theorem power_mod_remainder 
  (h1 : 7^2 % 17 = 15)
  (h2 : 15 % 17 = -2 % 17)
  (h3 : 2^4 % 17 = -1 % 17)
  (h4 : 1011 % 2 = 1) :
  7^2023 % 17 = 12 := 
  sorry

end power_mod_remainder_l2395_239556


namespace product_of_solutions_of_x_squared_eq_49_l2395_239523

theorem product_of_solutions_of_x_squared_eq_49 : 
  (∀ x, x^2 = 49 → x = 7 ∨ x = -7) → (7 * (-7) = -49) :=
by
  intros
  sorry

end product_of_solutions_of_x_squared_eq_49_l2395_239523


namespace paint_usage_correct_l2395_239565

-- Define the parameters representing paint usage and number of paintings
def largeCanvasPaint : Nat := 3
def smallCanvasPaint : Nat := 2
def largePaintings : Nat := 3
def smallPaintings : Nat := 4

-- Define the total paint used
def totalPaintUsed : Nat := largeCanvasPaint * largePaintings + smallCanvasPaint * smallPaintings

-- Prove that total paint used is 17 ounces
theorem paint_usage_correct : totalPaintUsed = 17 :=
  by
    sorry

end paint_usage_correct_l2395_239565


namespace number_of_skew_line_pairs_in_cube_l2395_239584

theorem number_of_skew_line_pairs_in_cube : 
  let vertices := 8
  let total_lines := 28
  let sets_of_4_points := Nat.choose 8 4 - 12
  let skew_pairs_per_set := 3
  let number_of_skew_pairs := sets_of_4_points * skew_pairs_per_set
  number_of_skew_pairs = 174 := sorry

end number_of_skew_line_pairs_in_cube_l2395_239584


namespace subtract_045_from_3425_l2395_239589

theorem subtract_045_from_3425 : 34.25 - 0.45 = 33.8 :=
by sorry

end subtract_045_from_3425_l2395_239589


namespace product_of_cubes_eq_l2395_239534

theorem product_of_cubes_eq :
  ( (3^3 - 1) / (3^3 + 1) ) * 
  ( (4^3 - 1) / (4^3 + 1) ) * 
  ( (5^3 - 1) / (5^3 + 1) ) * 
  ( (6^3 - 1) / (6^3 + 1) ) * 
  ( (7^3 - 1) / (7^3 + 1) ) * 
  ( (8^3 - 1) / (8^3 + 1) ) 
  = 73 / 256 :=
by
  sorry

end product_of_cubes_eq_l2395_239534


namespace traveler_journey_possible_l2395_239577

structure Archipelago (Island : Type) :=
  (n : ℕ)
  (fare : Island → Island → ℝ)
  (unique_ferry : ∀ i j : Island, i ≠ j → fare i j ≠ fare j i)
  (distinct_fares : ∀ i j k l: Island, i ≠ j ∧ k ≠ l → fare i j ≠ fare k l)
  (connected : ∀ i j : Island, i ≠ j → fare i j = fare j i)

theorem traveler_journey_possible {Island : Type} (arch : Archipelago Island) :
  ∃ (t : Island) (seq : List (Island × Island)), -- there exists a starting island and a sequence of journeys
    seq.length = arch.n - 1 ∧                   -- length of the sequence is n-1
    (∀ i j, (i, j) ∈ seq → j ≠ i ∧ arch.fare i j < arch.fare j i) := -- fare decreases with each journey
sorry

end traveler_journey_possible_l2395_239577


namespace greatest_value_of_n_l2395_239595

theorem greatest_value_of_n : ∀ (n : ℤ), 102 * n^2 ≤ 8100 → n ≤ 8 :=
by 
  sorry

end greatest_value_of_n_l2395_239595


namespace graph_depicts_one_line_l2395_239575

theorem graph_depicts_one_line {x y : ℝ} :
  (x - 1) ^ 2 * (x + y - 2) = (y - 1) ^ 2 * (x + y - 2) →
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b :=
by
  intros h
  sorry

end graph_depicts_one_line_l2395_239575


namespace appropriate_speech_length_l2395_239543

def speech_length_min := 20
def speech_length_max := 40
def speech_rate := 120

theorem appropriate_speech_length 
  (min_words := speech_length_min * speech_rate) 
  (max_words := speech_length_max * speech_rate) : 
  ∀ n : ℕ, n >= min_words ∧ n <= max_words ↔ (n = 2500 ∨ n = 3800 ∨ n = 4600) := 
by 
  sorry

end appropriate_speech_length_l2395_239543


namespace x_intercept_of_line_is_six_l2395_239528

theorem x_intercept_of_line_is_six : ∃ x : ℝ, (∃ y : ℝ, y = 0) ∧ (2*x - 4*y = 12) ∧ x = 6 :=
by {
  sorry
}

end x_intercept_of_line_is_six_l2395_239528


namespace quotient_sum_40_5_l2395_239563

theorem quotient_sum_40_5 : (40 + 5) / 5 = 9 := by
  sorry

end quotient_sum_40_5_l2395_239563


namespace find_B_max_f_A_l2395_239591

namespace ProofProblem

-- Definitions
variables {A B C a b c : ℝ} -- Angles and sides in the triangle
noncomputable def givenCondition (A B C a b c : ℝ) : Prop :=
  2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 4

-- Problem Statements (to be proved)
theorem find_B (h : givenCondition A B C a b c) : B = Real.pi / 6 := sorry

theorem max_f_A (A : ℝ) (B : ℝ) (h1 : 0 < A) (h2 : A < 5 * Real.pi / 6) (h3 : B = Real.pi / 6) : (∃ (x : ℝ), f x = 1 / 2) := sorry

end ProofProblem

end find_B_max_f_A_l2395_239591


namespace find_distance_AB_l2395_239509

variable (vA vB : ℝ) -- speeds of Person A and Person B
variable (x : ℝ) -- distance between points A and B
variable (t1 t2 : ℝ) -- time variables

-- Conditions
def startTime := 0
def meetDistanceBC := 240
def returnPointBDistantFromA := 120
def doublingSpeedFactor := 2

-- Main questions and conditions
theorem find_distance_AB
  (h1 : vA > vB)
  (h2 : t1 = x / vB)
  (h3 : t2 = 2 * (x - meetDistanceBC) / vA) 
  (h4 : x = meetDistanceBC + returnPointBDistantFromA + (t1 * (doublingSpeedFactor * vB) - t2 * vA) / (doublingSpeedFactor - 1)) :
  x = 420 :=
sorry

end find_distance_AB_l2395_239509


namespace series_sum_eq_four_ninths_l2395_239590

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l2395_239590


namespace purity_of_alloy_l2395_239576

theorem purity_of_alloy (w1 w2 : ℝ) (p1 p2 : ℝ) (h_w1 : w1 = 180) (h_p1 : p1 = 920) (h_w2 : w2 = 100) (h_p2 : p2 = 752) : 
  let a := w1 * (p1 / 1000) + w2 * (p2 / 1000)
  let b := w1 + w2
  let p_result := (a / b) * 1000
  p_result = 860 :=
by
  sorry

end purity_of_alloy_l2395_239576


namespace range_of_function_l2395_239566

theorem range_of_function :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 2 ≤ x^2 - 2 * x + 3 ∧ x^2 - 2 * x + 3 ≤ 6) :=
by {
  sorry
}

end range_of_function_l2395_239566


namespace cos_sum_eq_neg_ratio_l2395_239581

theorem cos_sum_eq_neg_ratio (γ δ : ℝ) 
  (hγ: Complex.exp (Complex.I * γ) = 4 / 5 + 3 / 5 * Complex.I) 
  (hδ: Complex.exp (Complex.I * δ) = -5 / 13 + 12 / 13 * Complex.I) :
  Real.cos (γ + δ) = -56 / 65 :=
  sorry

end cos_sum_eq_neg_ratio_l2395_239581


namespace trigonometric_identity_l2395_239557

open Real

theorem trigonometric_identity (θ : ℝ) (h : tan θ = 2) :
  (sin θ * (1 + sin (2 * θ))) / (sqrt 2 * cos (θ - π / 4)) = 6 / 5 :=
by
  sorry

end trigonometric_identity_l2395_239557


namespace lower_limit_brother_l2395_239516

variable (W B : Real)

-- Arun's opinion
def aruns_opinion := 66 < W ∧ W < 72

-- Brother's opinion
def brothers_opinion := B < W ∧ W < 70

-- Mother's opinion
def mothers_opinion := W ≤ 69

-- Given the average probable weight of Arun which is 68 kg
def average_weight := (69 + (max 66 B)) / 2 = 68

theorem lower_limit_brother (h₁ : aruns_opinion W) (h₂ : brothers_opinion W B) (h₃ : mothers_opinion W) (h₄ : average_weight B) :
  B = 67 := sorry

end lower_limit_brother_l2395_239516


namespace cost_of_first_book_l2395_239546

-- Define the initial amount of money Shelby had.
def initial_amount : ℕ := 20

-- Define the cost of the second book.
def cost_of_second_book : ℕ := 4

-- Define the cost of one poster.
def cost_of_poster : ℕ := 4

-- Define the number of posters bought.
def num_posters : ℕ := 2

-- Define the total cost that Shelby had to spend on posters.
def total_cost_of_posters : ℕ := num_posters * cost_of_poster

-- Define the total amount spent on books and posters.
def total_spent (X : ℕ) : ℕ := X + cost_of_second_book + total_cost_of_posters

-- Prove that the cost of the first book is 8 dollars.
theorem cost_of_first_book (X : ℕ) (h : total_spent X = initial_amount) : X = 8 :=
by
  sorry

end cost_of_first_book_l2395_239546


namespace a7_of_arithmetic_seq_l2395_239515

-- Defining the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d

theorem a7_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : arithmetic_seq a d) 
  (h_a4 : a 4 = 5) 
  (h_a5_a6 : a 5 + a 6 = 11) : 
  a 7 = 6 :=
by
  sorry

end a7_of_arithmetic_seq_l2395_239515


namespace total_spend_on_four_games_l2395_239518

noncomputable def calculate_total_spend (batman_price : ℝ) (superman_price : ℝ)
                                        (batman_discount : ℝ) (superman_discount : ℝ)
                                        (tax_rate : ℝ) (game1_price : ℝ) (game2_price : ℝ) : ℝ :=
  let batman_discounted_price := batman_price - batman_discount * batman_price
  let superman_discounted_price := superman_price - superman_discount * superman_price
  let batman_price_after_tax := batman_discounted_price + tax_rate * batman_discounted_price
  let superman_price_after_tax := superman_discounted_price + tax_rate * superman_discounted_price
  batman_price_after_tax + superman_price_after_tax + game1_price + game2_price

theorem total_spend_on_four_games :
  calculate_total_spend 13.60 5.06 0.10 0.05 0.08 7.25 12.50 = 38.16 :=
by sorry

end total_spend_on_four_games_l2395_239518


namespace kids_from_lawrence_county_go_to_camp_l2395_239578

theorem kids_from_lawrence_county_go_to_camp : 
  (1201565 - 590796 = 610769) := 
by
  sorry

end kids_from_lawrence_county_go_to_camp_l2395_239578


namespace total_players_l2395_239511

theorem total_players 
  (cricket_players : ℕ) (hockey_players : ℕ)
  (football_players : ℕ) (softball_players : ℕ)
  (h_cricket : cricket_players = 12)
  (h_hockey : hockey_players = 17)
  (h_football : football_players = 11)
  (h_softball : softball_players = 10)
  : cricket_players + hockey_players + football_players + softball_players = 50 :=
by sorry

end total_players_l2395_239511


namespace taylor_class_more_girls_l2395_239587

theorem taylor_class_more_girls (b g : ℕ) (total : b + g = 42) (ratio : b / g = 3 / 4) : g - b = 6 := by
  sorry

end taylor_class_more_girls_l2395_239587


namespace match_foci_of_parabola_and_hyperbola_l2395_239537

noncomputable def focus_of_parabola (a : ℝ) : ℝ :=
a / 4

noncomputable def foci_of_hyperbola : Set ℝ :=
{2, -2}

theorem match_foci_of_parabola_and_hyperbola (a : ℝ) :
  focus_of_parabola a ∈ foci_of_hyperbola ↔ a = 8 ∨ a = -8 :=
by
  -- This is the placeholder for the proof.
  sorry

end match_foci_of_parabola_and_hyperbola_l2395_239537


namespace share_difference_3600_l2395_239571

theorem share_difference_3600 (x : ℕ) (p q r : ℕ) (h1 : p = 3 * x) (h2 : q = 7 * x) (h3 : r = 12 * x) (h4 : r - q = 4500) : q - p = 3600 := by
  sorry

end share_difference_3600_l2395_239571


namespace milk_concentration_l2395_239570

variable {V_initial V_removed V_total : ℝ}

theorem milk_concentration (h1 : V_initial = 20) (h2 : V_removed = 2) (h3 : V_total = 20) :
    (V_initial - V_removed) / V_total * 100 = 90 := 
by 
  sorry

end milk_concentration_l2395_239570


namespace apples_in_box_l2395_239580

theorem apples_in_box :
  (∀ (o p a : ℕ), 
    (o = 1 / 4 * 56) ∧ 
    (p = 1 / 2 * o) ∧ 
    (a = 5 * p) → 
    a = 35) :=
  by sorry

end apples_in_box_l2395_239580


namespace correct_integer_with_7_divisors_l2395_239551

theorem correct_integer_with_7_divisors (n : ℕ) (p : ℕ) (h_prime : Prime p) 
  (h_3_divisors : ∃ (d : ℕ), d = 3 ∧ n = p^2) : n = 4 :=
by
-- Proof omitted
sorry

end correct_integer_with_7_divisors_l2395_239551


namespace statement2_statement3_l2395_239505

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Conditions for the statements
axiom cond1 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = q ∧ f a b c q = p
axiom cond2 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = f a b c q
axiom cond3 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c (p + q) = c

-- Statement 2 correctness
theorem statement2 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c p = f a b c q) : 
  f a b c (p + q) = c :=
sorry

-- Statement 3 correctness
theorem statement3 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c (p + q) = c) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end statement2_statement3_l2395_239505


namespace range_of_a_l2395_239545

variable (a x : ℝ)

theorem range_of_a (h : x - 5 = -3 * a) (hx_neg : x < 0) : a > 5 / 3 :=
by {
  sorry
}

end range_of_a_l2395_239545


namespace largest_value_of_c_l2395_239513

noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem largest_value_of_c :
  ∃ (c : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c → |g x - 1| ≤ c) ∧ (∀ (c' : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c' → |g x - 1| ≤ c') → c' ≤ c) :=
sorry

end largest_value_of_c_l2395_239513


namespace ratio_of_sides_l2395_239540

open Real

variable (s y x : ℝ)

-- Assuming the rectangles and squares conditions
def condition1 := 4 * (x * y) + s * s = 9 * (s * s)
def condition2 := x = 2 * s
def condition3 := y = s

-- Stating the theorem
theorem ratio_of_sides (h1 : condition1 s y x) (h2 : condition2 s x) (h3 : condition3 s y) :
  x / y = 2 := by
  sorry

end ratio_of_sides_l2395_239540


namespace find_two_digit_number_l2395_239599

def digit_eq_square_of_units (n x : ℤ) : Prop :=
  10 * (x - 3) + x = n ∧ n = x * x

def units_digit_3_larger_than_tens (x : ℤ) : Prop :=
  x - 3 >= 1 ∧ x - 3 < 10 ∧ x >= 3 ∧ x < 10

theorem find_two_digit_number (n x : ℤ) (h1 : digit_eq_square_of_units n x)
  (h2 : units_digit_3_larger_than_tens x) : n = 25 ∨ n = 36 :=
by sorry

end find_two_digit_number_l2395_239599


namespace total_profit_l2395_239532

noncomputable def profit_x (P : ℕ) : ℕ := 3 * P
noncomputable def profit_y (P : ℕ) : ℕ := 2 * P

theorem total_profit
  (P_x P_y : ℕ)
  (h_ratio : P_x = 3 * (P_y / 2))
  (h_diff : P_x - P_y = 100) :
  P_x + P_y = 500 :=
by
  sorry

end total_profit_l2395_239532


namespace machinery_spent_correct_l2395_239544

def raw_materials : ℝ := 3000
def total_amount : ℝ := 5714.29
def cash (total : ℝ) : ℝ := 0.30 * total
def machinery_spent (total : ℝ) (raw : ℝ) : ℝ := total - raw - cash total

theorem machinery_spent_correct :
  machinery_spent total_amount raw_materials = 1000 := 
  by
    sorry

end machinery_spent_correct_l2395_239544


namespace scientific_notation_248000_l2395_239574

theorem scientific_notation_248000 : (248000 : Float) = 2.48 * 10^5 := 
sorry

end scientific_notation_248000_l2395_239574


namespace num_distinct_sums_of_three_distinct_elements_l2395_239531

noncomputable def arith_seq_sum_of_three_distinct : Nat :=
  let a (i : Nat) : Nat := 3 * i + 1
  let lower_bound := 21
  let upper_bound := 129
  (upper_bound - lower_bound) / 3 + 1

theorem num_distinct_sums_of_three_distinct_elements : arith_seq_sum_of_three_distinct = 37 := by
  -- We are skipping the proof by using sorry
  sorry

end num_distinct_sums_of_three_distinct_elements_l2395_239531


namespace cone_slant_height_correct_l2395_239558

noncomputable def cone_slant_height (r : ℝ) : ℝ := 4 * r

theorem cone_slant_height_correct (r : ℝ) (h₁ : π * r^2 + π * r * cone_slant_height r = 5 * π)
  (h₂ : 2 * π * r = (1/4) * 2 * π * cone_slant_height r) : cone_slant_height r = 4 :=
by
  sorry

end cone_slant_height_correct_l2395_239558


namespace coin_pile_problem_l2395_239548

theorem coin_pile_problem (x y z : ℕ) (h1 : 2 * (x - y) = 16) (h2 : 2 * y - z = 16) (h3 : 2 * z - x + y = 16) :
  x = 22 ∧ y = 14 ∧ z = 12 :=
by
  sorry

end coin_pile_problem_l2395_239548


namespace num_distinct_log_values_l2395_239517

-- Defining the set of numbers
def number_set : Set ℕ := {1, 2, 3, 4, 6, 9}

-- Define a function to count distinct logarithmic values
noncomputable def distinct_log_values (s : Set ℕ) : ℕ := 
  -- skipped, assume the implementation is done correctly
  sorry 

theorem num_distinct_log_values : distinct_log_values number_set = 17 :=
by
  sorry

end num_distinct_log_values_l2395_239517


namespace geometric_sequence_y_value_l2395_239560

theorem geometric_sequence_y_value (q : ℝ) 
  (h1 : 2 * q^4 = 18) 
  : 2 * (q^2) = 6 := by
  sorry

end geometric_sequence_y_value_l2395_239560


namespace plates_per_meal_l2395_239592

theorem plates_per_meal 
  (people : ℕ) (meals_per_day : ℕ) (total_days : ℕ) (total_plates : ℕ) 
  (h_people : people = 6) 
  (h_meals : meals_per_day = 3) 
  (h_days : total_days = 4) 
  (h_plates : total_plates = 144) 
  : (total_plates / (people * meals_per_day * total_days)) = 2 := 
  sorry

end plates_per_meal_l2395_239592


namespace marla_errand_total_time_l2395_239555

theorem marla_errand_total_time :
  let d1 := 20 -- Driving to son's school
  let b := 30  -- Taking a bus to the grocery store
  let s := 15  -- Shopping at the grocery store
  let w := 10  -- Walking to the gas station
  let g := 5   -- Filling up gas
  let r := 25  -- Riding a bicycle to the school
  let p := 70  -- Attending parent-teacher night
  let c := 30  -- Catching up with a friend at a coffee shop
  let sub := 40-- Taking the subway home
  let d2 := 20 -- Driving home
  d1 + b + s + w + g + r + p + c + sub + d2 = 265 := by
  sorry

end marla_errand_total_time_l2395_239555


namespace number_of_juniors_twice_seniors_l2395_239521

variable (j s : ℕ)

theorem number_of_juniors_twice_seniors
  (h1 : (3 / 7 : ℝ) * j = (6 / 7 : ℝ) * s) : j = 2 * s := 
sorry

end number_of_juniors_twice_seniors_l2395_239521


namespace randy_feeds_per_day_l2395_239504

theorem randy_feeds_per_day
  (pigs : ℕ) (total_feed_per_week : ℕ) (days_per_week : ℕ)
  (h1 : pigs = 2) (h2 : total_feed_per_week = 140) (h3 : days_per_week = 7) :
  total_feed_per_week / pigs / days_per_week = 10 :=
by
  sorry

end randy_feeds_per_day_l2395_239504


namespace population_net_increase_l2395_239585

def birth_rate : ℕ := 8
def birth_time : ℕ := 2
def death_rate : ℕ := 6
def death_time : ℕ := 2
def seconds_per_minute : ℕ := 60
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 24

theorem population_net_increase :
  (birth_rate / birth_time - death_rate / death_time) * (seconds_per_minute * minutes_per_hour * hours_per_day) = 86400 :=
by
  sorry

end population_net_increase_l2395_239585


namespace right_triangle_hypotenuse_length_l2395_239525

theorem right_triangle_hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12)
  (h₃ : c^2 = a^2 + b^2) : c = 13 :=
by
  -- We should provide the actual proof here, but we'll use sorry for now.
  sorry

end right_triangle_hypotenuse_length_l2395_239525


namespace average_rate_l2395_239510

variable (d_run : ℝ) (d_swim : ℝ) (r_run : ℝ) (r_swim : ℝ)
variable (t_run : ℝ := d_run / r_run) (t_swim : ℝ := d_swim / r_swim)

theorem average_rate (h_dist_run : d_run = 4) (h_dist_swim : d_swim = 4)
                      (h_run_rate : r_run = 10) (h_swim_rate : r_swim = 6) : 
                      ((d_run + d_swim) / (t_run + t_swim)) / 60 = 0.125 :=
by
  -- Properly using all the conditions given
  have := (4 + 4) / (4 / 10 + 4 / 6) / 60 = 0.125
  sorry

end average_rate_l2395_239510


namespace calculate_milk_and_oil_l2395_239550

theorem calculate_milk_and_oil (q_f div_f milk_p oil_p : ℕ) (portions q_m q_o : ℕ) :
  q_f = 1050 ∧ div_f = 350 ∧ milk_p = 70 ∧ oil_p = 30 ∧
  portions = q_f / div_f ∧
  q_m = portions * milk_p ∧
  q_o = portions * oil_p →
  q_m = 210 ∧ q_o = 90 := by
  sorry

end calculate_milk_and_oil_l2395_239550


namespace find_original_six_digit_number_l2395_239529

theorem find_original_six_digit_number (N x y : ℕ) (h1 : N = 10 * x + y) (h2 : N - x = 654321) (h3 : 0 ≤ y ∧ y ≤ 9) :
  N = 727023 :=
sorry

end find_original_six_digit_number_l2395_239529


namespace inequality_ge_9_l2395_239596

theorem inequality_ge_9 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (2 / a + 1 / b) ≥ 9 :=
sorry

end inequality_ge_9_l2395_239596


namespace inverse_proportion_function_has_m_value_l2395_239573

theorem inverse_proportion_function_has_m_value
  (k : ℝ)
  (h1 : 2 * -3 = k)
  {m : ℝ}
  (h2 : 6 = k / m) :
  m = -1 :=
by
  sorry

end inverse_proportion_function_has_m_value_l2395_239573


namespace initial_payment_mr_dubois_l2395_239538

-- Definition of the given conditions
def total_cost_of_car : ℝ := 13380
def monthly_payment : ℝ := 420
def number_of_months : ℝ := 19

-- Calculate the total amount paid in monthly installments
def total_amount_paid_in_installments : ℝ := monthly_payment * number_of_months

-- Statement of the theorem we want to prove
theorem initial_payment_mr_dubois :
  total_cost_of_car - total_amount_paid_in_installments = 5400 :=
by
  sorry

end initial_payment_mr_dubois_l2395_239538


namespace solve_for_x_l2395_239501

theorem solve_for_x :
  ∀ x : ℤ, 3 * x + 36 = 48 → x = 4 :=
by
  sorry

end solve_for_x_l2395_239501


namespace trail_mix_total_weight_l2395_239594

theorem trail_mix_total_weight :
  let peanuts := 0.16666666666666666
  let chocolate_chips := 0.16666666666666666
  let raisins := 0.08333333333333333
  let almonds := 0.14583333333333331
  let cashews := (1 / 8 : Real)
  let dried_cranberries := (3 / 32 : Real)
  (peanuts + chocolate_chips + raisins + almonds + cashews + dried_cranberries) = 0.78125 :=
by
  sorry

end trail_mix_total_weight_l2395_239594


namespace point_to_real_l2395_239567

-- Condition: Real numbers correspond one-to-one with points on the number line.
def real_numbers_correspond (x : ℝ) : Prop :=
  ∃ (p : ℝ), p = x

-- Condition: Any real number can be represented by a point on the number line.
def represent_real_by_point (x : ℝ) : Prop :=
  real_numbers_correspond x

-- Condition: Conversely, any point on the number line represents a real number.
def point_represents_real (p : ℝ) : Prop :=
  ∃ (x : ℝ), x = p

-- Condition: The number represented by any point on the number line is either a rational number or an irrational number.
def rational_or_irrational (p : ℝ) : Prop :=
  (∃ q : ℚ, (q : ℝ) = p) ∨ (¬∃ q : ℚ, (q : ℝ) = p)

theorem point_to_real (p : ℝ) : represent_real_by_point p ∧ point_represents_real p ∧ rational_or_irrational p → real_numbers_correspond p :=
by sorry

end point_to_real_l2395_239567


namespace desired_line_equation_exists_l2395_239526

theorem desired_line_equation_exists :
  ∃ (a b c : ℝ), (a * 0 + b * 1 + c = 0) ∧
  (x - 3 * y + 10 = 0) ∧
  (2 * x + y - 8 = 0) ∧
  (a * x + b * y + c = 0) :=
by
  sorry

end desired_line_equation_exists_l2395_239526


namespace oil_depth_solution_l2395_239514

theorem oil_depth_solution
  (length diameter surface_area : ℝ)
  (h : ℝ)
  (h_length : length = 12)
  (h_diameter : diameter = 4)
  (h_surface_area : surface_area = 24)
  (r : ℝ := diameter / 2)
  (c : ℝ := surface_area / length) :
  (h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3) :=
by
  sorry

end oil_depth_solution_l2395_239514


namespace part_a_part_b_l2395_239530

theorem part_a (k : ℕ) : ∃ (a : ℕ → ℕ), (∀ i, i ≤ k → a i > 0) ∧ (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → a i < a j) ∧ (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) :=
sorry

theorem part_b : ∃ C > 0, ∀ a : ℕ → ℕ, (∀ k : ℕ, (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) → a 1 > (k : ℕ) ^ (C * k : ℕ)) :=
sorry

end part_a_part_b_l2395_239530


namespace min_value_problem_l2395_239562

theorem min_value_problem (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) : 
    a^2 + b^2 + c^2 + d^2 >= 24 / 5 := 
by
  sorry

end min_value_problem_l2395_239562


namespace train_speed_l2395_239503

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) (total_distance : ℝ) 
    (speed_mps : ℝ) (speed_kmph : ℝ) 
    (h1 : train_length = 360) 
    (h2 : bridge_length = 140) 
    (h3 : time = 34.61538461538461) 
    (h4 : total_distance = train_length + bridge_length) 
    (h5 : speed_mps = total_distance / time) 
    (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 52 := 
by 
  sorry

end train_speed_l2395_239503


namespace total_soda_bottles_l2395_239586

def regular_soda : ℕ := 57
def diet_soda : ℕ := 26
def lite_soda : ℕ := 27

theorem total_soda_bottles : regular_soda + diet_soda + lite_soda = 110 := by
  sorry

end total_soda_bottles_l2395_239586


namespace find_n_l2395_239541

theorem find_n 
  (N : ℕ) 
  (hn : ¬ (N = 0))
  (parts_inv_prop : ∀ k, 1 ≤ k → k ≤ n → N / (k * (k + 1)) = x / (n * (n + 1))) 
  (smallest_part : (N : ℝ) / 400 = N / (n * (n + 1))) : 
  n = 20 :=
sorry

end find_n_l2395_239541


namespace profit_percentage_correct_l2395_239568

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 70
noncomputable def list_price : ℝ := selling_price / 0.95
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem profit_percentage_correct :
  abs (profit_percentage - 47.37) < 0.01 := sorry

end profit_percentage_correct_l2395_239568


namespace certain_number_mod_l2395_239582

theorem certain_number_mod (n : ℤ) : (73 * n) % 8 = 7 → n % 8 = 7 := 
by sorry

end certain_number_mod_l2395_239582


namespace symmetric_points_origin_a_plus_b_l2395_239542

theorem symmetric_points_origin_a_plus_b (a b : ℤ) 
  (h1 : a + 3 * b = 5)
  (h2 : a + 2 * b = -3) :
  a + b = -11 :=
by
  sorry

end symmetric_points_origin_a_plus_b_l2395_239542


namespace backyard_area_l2395_239564

theorem backyard_area {length width : ℝ} 
  (h1 : 30 * length = 1500) 
  (h2 : 12 * (2 * (length + width)) = 1500) : 
  length * width = 625 :=
by
  sorry

end backyard_area_l2395_239564


namespace max_good_diagonals_l2395_239598

def is_good_diagonal (n : ℕ) (d : ℕ) : Prop := ∀ (P : Fin n → Prop), ∃! (i j : Fin n), P i ∧ P j ∧ (d = i + j)

theorem max_good_diagonals (n : ℕ) (h : 2 ≤ n) :
  (∃ (m : ℕ), is_good_diagonal n m ∧ (m = n - 2 ↔ Even n) ∧ (m = n - 3 ↔ Odd n)) :=
by
  sorry

end max_good_diagonals_l2395_239598


namespace fifth_friend_paid_40_l2395_239520

-- Defining the conditions given in the problem
variables {a b c d e : ℝ}
variables (h1 : a = (1/3) * (b + c + d + e))
variables (h2 : b = (1/4) * (a + c + d + e))
variables (h3 : c = (1/5) * (a + b + d + e))
variables (h4 : d = (1/6) * (a + b + c + e))
variables (h5 : a + b + c + d + e = 120)

-- Proving that the amount paid by the fifth friend is $40
theorem fifth_friend_paid_40 : e = 40 :=
by
  sorry  -- Proof to be provided

end fifth_friend_paid_40_l2395_239520


namespace matrix_power_eigenvector_l2395_239533

section MatrixEigen
variable (B : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ)

theorem matrix_power_eigenvector (h : B.mulVec ![3, -1] = ![-12, 4]) :
  (B ^ 5).mulVec ![3, -1] = ![-3072, 1024] := 
  sorry
end MatrixEigen

end matrix_power_eigenvector_l2395_239533


namespace stadium_length_in_feet_l2395_239502

theorem stadium_length_in_feet (length_in_yards : ℕ) (conversion_factor : ℕ) (h1 : length_in_yards = 62) (h2 : conversion_factor = 3) : length_in_yards * conversion_factor = 186 :=
by
  sorry

end stadium_length_in_feet_l2395_239502


namespace max_log_sum_l2395_239524

open Real

theorem max_log_sum (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 4 * y = 40) :
  log x + log y ≤ 2 :=
sorry

end max_log_sum_l2395_239524


namespace find_a_b_l2395_239527

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem find_a_b (a b : ℝ) (x : ℝ) (h : 5 * (log a x) ^ 2 + 2 * (log b x) ^ 2 = (10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) + (Real.log x) ^ 2) :
  b = a ^ (2 / (5 + Real.sqrt 17)) ∨ b = a ^ (2 / (5 - Real.sqrt 17)) :=
sorry

end find_a_b_l2395_239527


namespace total_earnings_l2395_239579

-- Definitions based on conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- The main theorem to prove
theorem total_earnings : (bead_necklaces + gem_necklaces) * cost_per_necklace = 90 :=
by
  sorry

end total_earnings_l2395_239579


namespace distinct_real_roots_range_l2395_239549

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 2 * x + a = 0) ∧ (y^2 - 2 * y + a = 0))
  ↔ a < 1 := 
by
  sorry

end distinct_real_roots_range_l2395_239549


namespace intersection_is_correct_l2395_239593

noncomputable def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
noncomputable def setB : Set ℝ := {x | Real.log x / Real.log 2 ≤ 2}

theorem intersection_is_correct : setA ∩ setB = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_is_correct_l2395_239593


namespace remainder_when_2013_divided_by_85_l2395_239507

theorem remainder_when_2013_divided_by_85 : 2013 % 85 = 58 :=
by
  sorry

end remainder_when_2013_divided_by_85_l2395_239507


namespace sin_cos_product_l2395_239559

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin x * Real.cos x = 2 / 5 := by
  sorry

end sin_cos_product_l2395_239559


namespace problem_solution_l2395_239572

-- Definition of the geometric sequence and the arithmetic condition
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def arithmetic_condition (a : ℕ → ℕ) := 2 * (a 3 + 1) = a 2 + a 4

-- Definitions used in the proof
def a_n (n : ℕ) : ℕ := 2^(n-1)
def b_n (n : ℕ) := a_n n + n
def S_5 := b_n 1 + b_n 2 + b_n 3 + b_n 4 + b_n 5

-- Proof statement
theorem problem_solution : 
  (∃ a : ℕ → ℕ, geometric_sequence a 2 ∧ arithmetic_condition a ∧ a 1 = 1 ∧ (∀ n, a n = 2^(n-1))) ∧
  S_5 = 46 :=
by
  sorry

end problem_solution_l2395_239572


namespace rectangle_perimeter_greater_than_16_l2395_239508

theorem rectangle_perimeter_greater_than_16 (a b : ℝ) (h : a * b > 2 * a + 2 * b) (ha : a > 0) (hb : b > 0) : 
  2 * (a + b) > 16 :=
sorry

end rectangle_perimeter_greater_than_16_l2395_239508


namespace ratio_of_new_time_to_previous_time_l2395_239583

noncomputable def distance : ℝ := 420
noncomputable def previous_time : ℝ := 7
noncomputable def speed_increase : ℝ := 40

-- Original speed
noncomputable def original_speed : ℝ := distance / previous_time

-- New speed
noncomputable def new_speed : ℝ := original_speed + speed_increase

-- New time taken to cover the same distance at the new speed
noncomputable def new_time : ℝ := distance / new_speed

-- Ratio of new time to previous time
noncomputable def time_ratio : ℝ := new_time / previous_time

theorem ratio_of_new_time_to_previous_time :
  time_ratio = 0.6 :=
by sorry

end ratio_of_new_time_to_previous_time_l2395_239583


namespace part_a_part_b_l2395_239552

/-- Two equally skilled chess players with p = 0.5, q = 0.5. -/
def p : ℝ := 0.5
def q : ℝ := 0.5

-- Definition for binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Binomial distribution
def P (n k : ℕ) : ℝ := (binomial_coeff n k) * (p^k) * (q^(n-k))

/-- Prove that the probability of winning one out of two games is greater than the probability of winning two out of four games -/
theorem part_a : (P 2 1) > (P 4 2) := sorry

/-- Prove that the probability of winning at least two out of four games is greater than the probability of winning at least three out of five games -/
theorem part_b : (P 4 2 + P 4 3 + P 4 4) > (P 5 3 + P 5 4 + P 5 5) := sorry

end part_a_part_b_l2395_239552


namespace solve_congruence_l2395_239500

-- Define the condition and residue modulo 47
def residue_modulo (a b n : ℕ) : Prop := (a ≡ b [MOD n])

-- The main theorem to be proved
theorem solve_congruence (m : ℕ) (h : residue_modulo (13 * m) 9 47) : residue_modulo m 26 47 :=
sorry

end solve_congruence_l2395_239500


namespace smallest_positive_integer_n_l2395_239569

theorem smallest_positive_integer_n (n : ℕ) (cube : Finset (Fin 8)) :
    (∀ (coloring : Finset (Fin 8)), 
      coloring.card = n → 
      ∃ (v : Fin 8), 
        (∀ (adj : Finset (Fin 8)), adj.card = 3 → adj ⊆ cube → v ∈ adj → adj ⊆ coloring)) 
    ↔ n = 5 := 
by
  sorry

end smallest_positive_integer_n_l2395_239569


namespace total_kids_played_l2395_239522

def kids_played_week (monday tuesday wednesday thursday: ℕ): ℕ :=
  let friday := thursday + (thursday * 20 / 100)
  let saturday := friday - (friday * 30 / 100)
  let sunday := 2 * monday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem total_kids_played : 
  kids_played_week 15 18 25 30 = 180 :=
by
  sorry

end total_kids_played_l2395_239522


namespace triangle_perimeter_l2395_239561

def ellipse (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

def foci_distance (c : ℝ) := c = Real.sqrt 2

theorem triangle_perimeter {x y : ℝ} (A : ellipse x y) (F1 F2 : ℝ)
  (hF1 : F1 = -Real.sqrt 2) (hF2 : F2 = Real.sqrt 2) :
  |(x - F1)| + |(x - F2)| = 4 + 2 * Real.sqrt 2 :=
sorry

end triangle_perimeter_l2395_239561


namespace probability_of_perfect_square_sum_l2395_239597

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l2395_239597


namespace probability_both_visible_l2395_239539

noncomputable def emma_lap_time : ℕ := 100
noncomputable def ethan_lap_time : ℕ := 75
noncomputable def start_time : ℕ := 0
noncomputable def photo_start_minute : ℕ := 12 * 60 -- converted to seconds
noncomputable def photo_end_minute : ℕ := 13 * 60 -- converted to seconds
noncomputable def photo_visible_angle : ℚ := 1 / 3

theorem probability_both_visible :
  ∀ start_time photo_start_minute photo_end_minute emma_lap_time ethan_lap_time photo_visible_angle,
  start_time = 0 →
  photo_start_minute = 12 * 60 →
  photo_end_minute = 13 * 60 →
  emma_lap_time = 100 →
  ethan_lap_time = 75 →
  photo_visible_angle = 1 / 3 →
  (∃ t, photo_start_minute ≤ t ∧ t < photo_end_minute ∧
        (t % emma_lap_time ≤ (photo_visible_angle * emma_lap_time) / 2 ∨
         t % emma_lap_time ≥ emma_lap_time - (photo_visible_angle * emma_lap_time) / 2) ∧
        (t % ethan_lap_time ≤ (photo_visible_angle * ethan_lap_time) / 2 ∨
         t % ethan_lap_time ≥ ethan_lap_time - (photo_visible_angle * ethan_lap_time) / 2)) ↔
  true :=
sorry

end probability_both_visible_l2395_239539


namespace purchased_only_A_l2395_239553

-- Definitions for the conditions
def total_B (x : ℕ) := x + 500
def total_A (y : ℕ) := 2 * y

-- Question formulated in Lean 4
theorem purchased_only_A : 
  ∃ C : ℕ, (∀ x y : ℕ, 2 * x = 500 → y = total_B x → 2 * y = total_A y → C = total_A y - 500) ∧ C = 1000 :=
  sorry

end purchased_only_A_l2395_239553
