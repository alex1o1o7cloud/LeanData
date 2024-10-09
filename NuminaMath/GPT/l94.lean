import Mathlib

namespace regular_polygon_sides_l94_9489

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l94_9489


namespace variance_scaled_l94_9475

theorem variance_scaled (s1 : ℝ) (c : ℝ) (h1 : s1 = 3) (h2 : c = 3) :
  s1 * (c^2) = 27 :=
by
  rw [h1, h2]
  norm_num

end variance_scaled_l94_9475


namespace middle_part_is_28_4_over_11_l94_9411

theorem middle_part_is_28_4_over_11 (x : ℚ) :
  let part1 := x
  let part2 := (1/2) * x
  let part3 := (1/3) * x
  part1 + part2 + part3 = 104
  ∧ part2 = 28 + 4/11 := by
  sorry

end middle_part_is_28_4_over_11_l94_9411


namespace no_four_consecutive_lucky_numbers_l94_9430

def is_lucky (n : ℕ) : Prop :=
  let digits := n.digits 10
  n > 999999 ∧ n < 10000000 ∧ (∀ d ∈ digits, d ≠ 0) ∧ 
  n % (digits.foldl (λ x y => x * y) 1) = 0

theorem no_four_consecutive_lucky_numbers :
  ¬ ∃ (n : ℕ), is_lucky n ∧ is_lucky (n + 1) ∧ is_lucky (n + 2) ∧ is_lucky (n + 3) :=
sorry

end no_four_consecutive_lucky_numbers_l94_9430


namespace total_baseball_cards_l94_9439

theorem total_baseball_cards (Carlos Matias Jorge : ℕ) (h1 : Carlos = 20) (h2 : Matias = Carlos - 6) (h3 : Jorge = Matias) : Carlos + Matias + Jorge = 48 :=
by
  sorry

end total_baseball_cards_l94_9439


namespace roots_poly_sum_l94_9435

noncomputable def Q (z : ℂ) (a b c : ℝ) : ℂ := z^3 + (a:ℂ)*z^2 + (b:ℂ)*z + (c:ℂ)

theorem roots_poly_sum (a b c : ℝ) (u : ℂ)
  (h1 : u.im = 0) -- Assuming u is a real number
  (h2 : Q (u + 5 * Complex.I) a b c = 0)
  (h3 : Q (u + 15 * Complex.I) a b c = 0)
  (h4 : Q (2 * u - 6) a b c = 0) :
  a + b + c = -196 := by
  sorry

end roots_poly_sum_l94_9435


namespace no_solution_for_equation_l94_9493

theorem no_solution_for_equation :
  ¬ (∃ x : ℝ, 
    4 * x * (10 * x - (-10 - (3 * x - 8 * (x + 1)))) + 5 * (12 - (4 * (x + 1) - 3 * x)) = 
    18 * x^2 - (6 * x^2 - (7 * x + 4 * (2 * x^2 - x + 11)))) :=
by
  sorry

end no_solution_for_equation_l94_9493


namespace area_of_ADC_l94_9431

theorem area_of_ADC
  (BD DC : ℝ)
  (h_ratio : BD / DC = 2 / 3)
  (area_ABD : ℝ)
  (h_area_ABD : area_ABD = 30) :
  ∃ area_ADC, area_ADC = 45 :=
by {
  sorry
}

end area_of_ADC_l94_9431


namespace minimum_value_l94_9481

open Real

theorem minimum_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eqn : 2 * x + y = 2) :
    ∃ x y, (0 < x) ∧ (0 < y) ∧ (2 * x + y = 2) ∧ (x + sqrt (x^2 + y^2) = 8 / 5) :=
sorry

end minimum_value_l94_9481


namespace mod_equivalence_l94_9408

theorem mod_equivalence (x y m : ℤ) (h1 : x ≡ 25 [ZMOD 60]) (h2 : y ≡ 98 [ZMOD 60]) (h3 : m = 167) :
  x - y ≡ m [ZMOD 60] :=
sorry

end mod_equivalence_l94_9408


namespace consecutive_negatives_product_to_sum_l94_9417

theorem consecutive_negatives_product_to_sum :
  ∃ (n : ℤ), n * (n + 1) = 2184 ∧ n + (n + 1) = -95 :=
by {
  sorry
}

end consecutive_negatives_product_to_sum_l94_9417


namespace probability_triangle_side_decagon_l94_9444

theorem probability_triangle_side_decagon (total_vertices : ℕ) (choose_vertices : ℕ)
  (total_triangles : ℕ) (favorable_outcomes : ℕ)
  (triangle_formula : total_vertices = 10)
  (choose_vertices_formula : choose_vertices = 3)
  (total_triangle_count_formula : total_triangles = 120)
  (favorable_outcome_count_formula : favorable_outcomes = 70)
  : (favorable_outcomes : ℚ) / total_triangles = 7 / 12 := 
by 
  sorry

end probability_triangle_side_decagon_l94_9444


namespace min_value_fraction_l94_9403

theorem min_value_fraction {x : ℝ} (h : x > 8) : 
    ∃ c : ℝ, (∀ y : ℝ, y = (x^2) / ((x - 8)^2) → c ≤ y) ∧ c = 1 := 
sorry

end min_value_fraction_l94_9403


namespace no_integer_roots_l94_9425

theorem no_integer_roots : ∀ x : ℤ, x^3 - 4 * x^2 - 11 * x + 20 ≠ 0 := 
by
  sorry

end no_integer_roots_l94_9425


namespace percentage_of_students_who_own_cats_l94_9486

theorem percentage_of_students_who_own_cats (total_students cats_owned : ℕ) (h_total: total_students = 500) (h_cats: cats_owned = 75) :
  (cats_owned : ℚ) / total_students * 100 = 15 :=
by
  sorry

end percentage_of_students_who_own_cats_l94_9486


namespace positive_integer_pairs_l94_9447

theorem positive_integer_pairs (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (∃ k : ℕ, k > 0 ∧ k = a^2 / (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, 0 < l ∧ 
    ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
by
  sorry

end positive_integer_pairs_l94_9447


namespace equation_solution_l94_9424

theorem equation_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + d^2 - a * b - b * c - c * d - d + (2 / 5) = 0 ↔ 
  a = 1 / 5 ∧ b = 2 / 5 ∧ c = 3 / 5 ∧ d = 4 / 5 :=
by sorry

end equation_solution_l94_9424


namespace range_of_m_l94_9470

theorem range_of_m (m : ℝ) :
  (¬(∀ x y : ℝ, x^2 / (25 - m) + y^2 / (m - 7) = 1 → 25 - m > 0 ∧ m - 7 > 0 ∧ 25 - m > m - 7) ∨ 
   ¬(∀ x y : ℝ, y^2 / 5 - x^2 / m = 1 → 1 < (5 + m) / 5 ∧ (5 + m) / 5 < 4)) 
  → 7 < m ∧ m < 15 :=
by
  sorry

end range_of_m_l94_9470


namespace anna_cupcakes_remaining_l94_9428

theorem anna_cupcakes_remaining :
  let total_cupcakes := 60
  let cupcakes_given_away := (4 / 5 : ℝ) * total_cupcakes
  let cupcakes_after_giving := total_cupcakes - cupcakes_given_away
  let cupcakes_eaten := 3
  let cupcakes_left := cupcakes_after_giving - cupcakes_eaten
  cupcakes_left = 9 :=
by
  sorry

end anna_cupcakes_remaining_l94_9428


namespace exists_mod_inv_l94_9441

theorem exists_mod_inv (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (h : ¬ a ∣ p) : ∃ b : ℕ, a * b ≡ 1 [MOD p] :=
by
  sorry

end exists_mod_inv_l94_9441


namespace casper_candy_problem_l94_9451

theorem casper_candy_problem (o y gr : ℕ) (n : ℕ) (h1 : 10 * o = 16 * y) (h2 : 16 * y = 18 * gr) (h3 : 18 * gr = 18 * n) :
    n = 40 :=
by
  sorry

end casper_candy_problem_l94_9451


namespace min_value_expression_l94_9454

theorem min_value_expression :
  (∀ y : ℝ, abs y ≤ 1 → ∃ x : ℝ, 2 * x + y = 1 ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1 → 
    (∃ y : ℝ, 2 * x + y = 1 ∧ abs y ≤ 1 ∧ (2 * x ^ 2 + 16 * x + 3 * y ^ 2) = 3))) :=
sorry

end min_value_expression_l94_9454


namespace gcd_le_two_l94_9443

theorem gcd_le_two (a m n : ℕ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) (h4 : Odd n) :
  Nat.gcd (a^n - 1) (a^m + 1) ≤ 2 := 
sorry

end gcd_le_two_l94_9443


namespace kameron_kangaroos_l94_9461

theorem kameron_kangaroos (K : ℕ) (B_now : ℕ) (rate : ℕ) (days : ℕ)
    (h1 : B_now = 20)
    (h2 : rate = 2)
    (h3 : days = 40)
    (h4 : B_now + rate * days = K) : K = 100 := by
  sorry

end kameron_kangaroos_l94_9461


namespace men_count_eq_eight_l94_9453

theorem men_count_eq_eight (M W B : ℕ) (total_earnings : ℝ) (men_wages : ℝ)
  (H1 : M = W) (H2 : W = B) (H3 : B = 8)
  (H4 : total_earnings = 105) (H5 : men_wages = 7) :
  M = 8 := 
by 
  -- We need to show M = 8 given conditions
  sorry

end men_count_eq_eight_l94_9453


namespace line_equation_l94_9448

-- Given a point and a direction vector
def point : ℝ × ℝ := (3, 4)
def direction_vector : ℝ × ℝ := (-2, 1)

-- Equation of the line passing through the given point with the given direction vector
theorem line_equation (x y : ℝ) : 
  (x = 3 ∧ y = 4) → ∃a b c : ℝ, a = 1 ∧ b = 2 ∧ c = -11 ∧ a*x + b*y + c = 0 :=
by
  sorry

end line_equation_l94_9448


namespace percentage_increase_l94_9459

theorem percentage_increase 
  (P : ℝ)
  (bought_price : ℝ := 0.80 * P) 
  (original_profit : ℝ := 0.3600000000000001 * P) :
  ∃ X : ℝ, X = 70.00000000000002 ∧ (1.3600000000000001 * P = bought_price * (1 + X / 100)) :=
sorry

end percentage_increase_l94_9459


namespace yellow_dandelions_day_before_yesterday_l94_9463

theorem yellow_dandelions_day_before_yesterday :
  ∀ (yellow_yesterday white_yesterday yellow_today white_today : ℕ),
    yellow_yesterday = 20 →
    white_yesterday = 14 →
    yellow_today = 15 →
    white_today = 11 →
    ∃ yellow_day_before_yesterday : ℕ,
      yellow_day_before_yesterday = white_yesterday + white_today :=
by sorry

end yellow_dandelions_day_before_yesterday_l94_9463


namespace jaylene_saves_fraction_l94_9460

-- Statement of the problem
theorem jaylene_saves_fraction (r_saves : ℝ) (j_saves : ℝ) (m_saves : ℝ) 
    (r_salary_fraction : r_saves = 2 / 5) 
    (m_salary_fraction : m_saves = 1 / 2) 
    (total_savings : 4 * (r_saves * 500 + j_saves * 500 + m_saves * 500) = 3000) : 
    j_saves = 3 / 5 := 
by 
  sorry

end jaylene_saves_fraction_l94_9460


namespace repeating_decimal_divisible_by_2_or_5_l94_9446

theorem repeating_decimal_divisible_by_2_or_5 
    (m n : ℕ) 
    (x : ℝ) 
    (r s : ℕ) 
    (a b k p q u : ℕ)
    (hmn_coprime : Nat.gcd m n = 1)
    (h_rep_decimal : x = (m:ℚ) / (n:ℚ))
    (h_non_repeating_part: 0 < r) :
  n % 2 = 0 ∨ n % 5 = 0 :=
sorry

end repeating_decimal_divisible_by_2_or_5_l94_9446


namespace number_of_good_card_groups_l94_9487

noncomputable def card_value (k : ℕ) : ℕ := 2 ^ k

def is_good_card_group (cards : Finset ℕ) : Prop :=
  (cards.sum card_value = 2004)

theorem number_of_good_card_groups : 
  ∃ n : ℕ, n = 1006009 ∧ ∃ (cards : Finset ℕ), is_good_card_group cards :=
sorry

end number_of_good_card_groups_l94_9487


namespace consecutive_integers_sum_l94_9421

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 := by
  sorry

end consecutive_integers_sum_l94_9421


namespace gcf_of_lcm_9_21_and_10_22_eq_one_l94_9429

theorem gcf_of_lcm_9_21_and_10_22_eq_one :
  Nat.gcd (Nat.lcm 9 21) (Nat.lcm 10 22) = 1 :=
sorry

end gcf_of_lcm_9_21_and_10_22_eq_one_l94_9429


namespace chocolates_sold_l94_9462

theorem chocolates_sold (C S : ℝ) (n : ℕ) (h1 : 165 * C = n * S) (h2 : ((S - C) / C) * 100 = 10) : n = 150 :=
by
  sorry

end chocolates_sold_l94_9462


namespace time_difference_l94_9473

theorem time_difference (dist1 dist2 : ℕ) (speed : ℕ) (h_dist : dist1 = 600) (h_dist2 : dist2 = 550) (h_speed : speed = 40) :
  (dist1 - dist2) / speed * 60 = 75 := by
  sorry

end time_difference_l94_9473


namespace max_value_of_a_l94_9423

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := 
by 
  sorry

end max_value_of_a_l94_9423


namespace book_sale_total_amount_l94_9434

noncomputable def total_amount_received (total_books price_per_book : ℕ → ℝ) : ℝ :=
  price_per_book 80

theorem book_sale_total_amount (B : ℕ)
  (h1 : (1/3 : ℚ) * B = 40)
  (h2 : ∀ (n : ℕ), price_per_book n = 3.50) :
  total_amount_received B price_per_book = 280 := 
by
  sorry

end book_sale_total_amount_l94_9434


namespace arithmetic_sequences_ratio_l94_9483

theorem arithmetic_sequences_ratio
  (a b : ℕ → ℕ)
  (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2 - a 1))) / 2)
  (h2 : ∀ n, T n = (n * (2 * (b 1) + (n - 1) * (b 2 - b 1))) / 2)
  (h3 : ∀ n, (S n) / (T n) = (2 * n + 2) / (n + 3)) :
  (a 10) / (b 9) = 2 := sorry

end arithmetic_sequences_ratio_l94_9483


namespace ratio_Polly_to_Pulsar_l94_9476

theorem ratio_Polly_to_Pulsar (P Po Pe : ℕ) (k : ℕ) (h1 : P = 10) (h2 : Po = k * P) (h3 : Pe = Po / 6) (h4 : P + Po + Pe = 45) : Po / P = 3 :=
by 
  -- Skipping the proof, but this sets up the Lean environment
  sorry

end ratio_Polly_to_Pulsar_l94_9476


namespace intersection_is_correct_l94_9455

def A : Set ℤ := {0, 3, 4}
def B : Set ℤ := {-1, 0, 2, 3}

theorem intersection_is_correct : A ∩ B = {0, 3} := by
  sorry

end intersection_is_correct_l94_9455


namespace find_a_div_b_l94_9414

theorem find_a_div_b (a b : ℝ) (h_distinct : a ≠ b) 
  (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 0.6 :=
by
  sorry

end find_a_div_b_l94_9414


namespace question1_is_random_event_question2_probability_xiuShui_l94_9474

-- Definitions for projects
inductive Project
| A | B | C | D

-- Definition for the problem context and probability computation
def xiuShuiProjects : List Project := [Project.A, Project.B]
def allProjects : List Project := [Project.A, Project.B, Project.C, Project.D]

-- Question 1
def isRandomEvent (event : Project) : Prop :=
  event = Project.C ∧ event ∈ allProjects

theorem question1_is_random_event : isRandomEvent Project.C := by
sorry

-- Question 2: Probability both visit Xiu Shui projects is 1/4
def favorable_outcomes : List (Project × Project) :=
  [(Project.A, Project.A), (Project.A, Project.B), (Project.B, Project.A), (Project.B, Project.B)]

def total_outcomes : List (Project × Project) :=
  List.product allProjects allProjects

def probability (fav : ℕ) (total : ℕ) : ℚ := fav / total

theorem question2_probability_xiuShui : probability favorable_outcomes.length total_outcomes.length = 1 / 4 := by
sorry

end question1_is_random_event_question2_probability_xiuShui_l94_9474


namespace rainy_days_l94_9416

theorem rainy_days (n R NR : ℕ): (n * R + 3 * NR = 20) ∧ (3 * NR = n * R + 10) ∧ (R + NR = 7) → R = 2 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end rainy_days_l94_9416


namespace minimizes_G_at_7_over_12_l94_9464

def F (p q : ℝ) : ℝ :=
  -2 * p * q + 3 * p * (1 - q) + 3 * (1 - p) * q - 4 * (1 - p) * (1 - q)

noncomputable def G (p : ℝ) : ℝ :=
  max (3 * p - 4) (3 - 5 * p)

theorem minimizes_G_at_7_over_12 :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → (∀ p, G p ≥ G (7 / 12)) ↔ p = 7 / 12 :=
by
  sorry

end minimizes_G_at_7_over_12_l94_9464


namespace compute_expression_l94_9499

theorem compute_expression : 7 * (1 / 21) * 42 = 14 :=
by
  sorry

end compute_expression_l94_9499


namespace group_B_fluctuates_less_l94_9436

-- Conditions
def mean_A : ℝ := 80
def mean_B : ℝ := 90
def variance_A : ℝ := 10
def variance_B : ℝ := 5

-- Goal
theorem group_B_fluctuates_less :
  variance_B < variance_A :=
  by
    sorry

end group_B_fluctuates_less_l94_9436


namespace C_eq_D_iff_n_eq_3_l94_9488

noncomputable def C (n : ℕ) : ℝ :=
  1000 * (1 - (1 / 3^n)) / (1 - 1 / 3)

noncomputable def D (n : ℕ) : ℝ :=
  2700 * (1 - (1 / (-3)^n)) / (1 + 1 / 3)

theorem C_eq_D_iff_n_eq_3 (n : ℕ) (h : 1 ≤ n) : C n = D n ↔ n = 3 :=
by
  unfold C D
  sorry

end C_eq_D_iff_n_eq_3_l94_9488


namespace fraction_subtraction_l94_9409

theorem fraction_subtraction (a b : ℚ) (h_a: a = 5/9) (h_b: b = 1/6) : a - b = 7/18 :=
by
  sorry

end fraction_subtraction_l94_9409


namespace recurrence_sequence_a5_l94_9450

theorem recurrence_sequence_a5 :
  ∃ a : ℕ → ℚ, (a 1 = 5 ∧ (∀ n, a (n + 1) = 1 + 1 / a n) ∧ a 5 = 28 / 17) :=
  sorry

end recurrence_sequence_a5_l94_9450


namespace factorization_x6_minus_5x4_plus_8x2_minus_4_l94_9485

theorem factorization_x6_minus_5x4_plus_8x2_minus_4 (x : ℝ) :
  x^6 - 5 * x^4 + 8 * x^2 - 4 = (x - 1) * (x + 1) * (x^2 - 2)^2 :=
sorry

end factorization_x6_minus_5x4_plus_8x2_minus_4_l94_9485


namespace find_square_digit_l94_9415

-- Define the known sum of the digits 4, 7, 6, and 9
def sum_known_digits := 4 + 7 + 6 + 9

-- Define the condition that the number 47,69square must be divisible by 6
def is_multiple_of_6 (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∧ (sum_known_digits + d) % 3 = 0

-- Theorem statement that verifies both the conditions and finds possible values of square
theorem find_square_digit (d : ℕ) (h : is_multiple_of_6 d) : d = 4 ∨ d = 8 :=
by sorry

end find_square_digit_l94_9415


namespace min_y_squared_isosceles_trapezoid_l94_9426

theorem min_y_squared_isosceles_trapezoid:
  ∀ (EF GH y : ℝ) (circle_center : ℝ)
    (isosceles_trapezoid : Prop)
    (tangent_EH : Prop)
    (tangent_FG : Prop),
  isosceles_trapezoid ∧ EF = 72 ∧ GH = 45 ∧ EH = y ∧ FG = y ∧
  (∃ (circle : ℝ), circle_center = (EF / 2) ∧ tangent_EH ∧ tangent_FG)
  → y^2 = 486 :=
by sorry

end min_y_squared_isosceles_trapezoid_l94_9426


namespace max_students_exam_l94_9465

/--
An exam contains 4 multiple-choice questions, each with three options (A, B, C). Several students take the exam.
For any group of 3 students, there is at least one question where their answers are all different.
Each student answers all questions. Prove that the maximum number of students who can take the exam is 9.
-/
theorem max_students_exam (n : ℕ) (A B C : ℕ → ℕ → ℕ) (q : ℕ) :
  (∀ (s1 s2 s3 : ℕ), ∃ (q : ℕ), (1 ≤ q ∧ q ≤ 4) ∧ (A s1 q ≠ A s2 q ∧ A s1 q ≠ A s3 q ∧ A s2 q ≠ A s3 q)) →
  q = 4 ∧ (∀ s, 1 ≤ s → s ≤ n) → n ≤ 9 :=
by
  sorry

end max_students_exam_l94_9465


namespace find_30_cent_items_l94_9495

-- Define the parameters and their constraints
variables (a d b c : ℕ)

-- Define the conditions
def total_items : Prop := a + d + b + c = 50
def total_cost : Prop := 30 * a + 150 * d + 200 * b + 300 * c = 6000

-- The theorem to prove the number of 30-cent items purchased
theorem find_30_cent_items (h1 : total_items a d b c) (h2 : total_cost a d b c) : 
  ∃ a, a + d + b + c = 50 ∧ 30 * a + 150 * d + 200 * b + 300 * c = 6000 := 
sorry

end find_30_cent_items_l94_9495


namespace boxes_of_toothpicks_needed_l94_9410

def total_cards : Nat := 52
def unused_cards : Nat := 23
def cards_used : Nat := total_cards - unused_cards

def toothpicks_wall_per_card : Nat := 64
def windows_per_card : Nat := 3
def doors_per_card : Nat := 2
def toothpicks_per_window_or_door : Nat := 12
def roof_toothpicks : Nat := 1250
def box_capacity : Nat := 750

def toothpicks_for_walls : Nat := cards_used * toothpicks_wall_per_card
def toothpicks_per_card_windows_doors : Nat := (windows_per_card + doors_per_card) * toothpicks_per_window_or_door
def toothpicks_for_windows_doors : Nat := cards_used * toothpicks_per_card_windows_doors
def total_toothpicks_needed : Nat := toothpicks_for_walls + toothpicks_for_windows_doors + roof_toothpicks

def boxes_needed := Nat.ceil (total_toothpicks_needed / box_capacity)

theorem boxes_of_toothpicks_needed : boxes_needed = 7 := by
  -- Proof should be done here
  sorry

end boxes_of_toothpicks_needed_l94_9410


namespace range_of_m_l94_9402

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = -3) (h3 : x + y > 0) : m > 2 :=
by
  sorry

end range_of_m_l94_9402


namespace proof_a_in_S_l94_9404

def S : Set ℤ := {n : ℤ | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem proof_a_in_S (a : ℤ) (h1 : 3 * a ∈ S) : a ∈ S :=
sorry

end proof_a_in_S_l94_9404


namespace geoff_initial_percent_l94_9401

theorem geoff_initial_percent (votes_cast : ℕ) (win_percent : ℝ) (needed_more_votes : ℕ) (initial_votes : ℕ)
  (h1 : votes_cast = 6000)
  (h2 : win_percent = 50.5)
  (h3 : needed_more_votes = 3000)
  (h4 : initial_votes = 31) :
  (initial_votes : ℝ) / votes_cast * 100 = 0.52 :=
by
  sorry

end geoff_initial_percent_l94_9401


namespace smallest_possible_perimeter_l94_9458

open Real

theorem smallest_possible_perimeter
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a^2 + b^2 = 2016) :
  a + b + 2^3 * 3 * sqrt 14 = 48 + 2^3 * 3 * sqrt 14 :=
sorry

end smallest_possible_perimeter_l94_9458


namespace min_value_x_l94_9419

theorem min_value_x (x : ℝ) (h : ∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) : x ≥ -1 := 
sorry

end min_value_x_l94_9419


namespace find_number_69_3_l94_9442

theorem find_number_69_3 (x : ℝ) (h : (x * 0.004) / 0.03 = 9.237333333333334) : x = 69.3 :=
by
  sorry

end find_number_69_3_l94_9442


namespace train_speed_l94_9498

theorem train_speed
  (train_length : ℝ)
  (cross_time : ℝ)
  (man_speed_kmh : ℝ)
  (train_speed_kmh : ℝ) :
  (train_length = 150) →
  (cross_time = 6) →
  (man_speed_kmh = 5) →
  (man_speed_kmh * 1000 / 3600 + (train_speed_kmh * 1000 / 3600)) * cross_time = train_length →
  train_speed_kmh = 85 :=
by
  intros htl hct hmk hs
  sorry

end train_speed_l94_9498


namespace rational_range_l94_9457

theorem rational_range (a : ℚ) (h : a - |a| = 2 * a) : a ≤ 0 := 
sorry

end rational_range_l94_9457


namespace equation_of_parallel_line_l94_9477

theorem equation_of_parallel_line (A : ℝ × ℝ) (c : ℝ) : 
  A = (-1, 0) → (∀ x y, 2 * x - y + 1 = 0 → 2 * x - y + c = 0) → 
  2 * (-1) - 0 + c = 0 → c = 2 :=
by
  intros A_coord parallel_line point_on_line
  sorry

end equation_of_parallel_line_l94_9477


namespace valid_sequences_count_l94_9479

def g (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n < 3 then 0
  else g (n - 4) + 3 * g (n - 5) + 3 * g (n - 6)

theorem valid_sequences_count : g 17 = 37 :=
  sorry

end valid_sequences_count_l94_9479


namespace no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l94_9422

theorem no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49 :
  ∀ n : ℕ, ¬ (∃ k : ℤ, (n^2 + 5 * n + 1) = 49 * k) :=
by
  sorry

end no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l94_9422


namespace simplify_evaluate_l94_9482

theorem simplify_evaluate :
  ∀ (x : ℝ), x = Real.sqrt 2 - 1 →
  ((1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6))) = Real.sqrt 2 :=
by
  intros x hx
  sorry

end simplify_evaluate_l94_9482


namespace james_carrot_sticks_l94_9491

def carrots_eaten_after_dinner (total_carrots : ℕ) (carrots_before_dinner : ℕ) : ℕ :=
  total_carrots - carrots_before_dinner

theorem james_carrot_sticks : carrots_eaten_after_dinner 37 22 = 15 := by
  sorry

end james_carrot_sticks_l94_9491


namespace find_x_value_l94_9445

theorem find_x_value (x : ℝ) (h : 150 + 90 + x + 90 = 360) : x = 30 := by
  sorry

end find_x_value_l94_9445


namespace ann_older_than_susan_l94_9456

variables (A S : ℕ)

theorem ann_older_than_susan (h1 : S = 11) (h2 : A + S = 27) : A - S = 5 := by
  -- Proof is skipped
  sorry

end ann_older_than_susan_l94_9456


namespace value_added_to_075_of_number_l94_9484

theorem value_added_to_075_of_number (N V : ℝ) (h1 : 0.75 * N + V = 8) (h2 : N = 8) : V = 2 := by
  sorry

end value_added_to_075_of_number_l94_9484


namespace probability_prime_ball_l94_9427

open Finset

theorem probability_prime_ball :
  let balls := {1, 2, 3, 4, 5, 6, 8, 9}
  let total := card balls
  let primes := {2, 3, 5}
  let primes_count := card primes
  (total = 8) → (primes ⊆ balls) → 
  primes_count = 3 → 
  primes_count / total = 3 / 8 :=
by
  intros
  sorry

end probability_prime_ball_l94_9427


namespace largest_even_not_sum_of_two_composite_odds_l94_9497

-- Definitions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ k, k > 1 ∧ k < n ∧ n % k = 0

-- Theorem statement
theorem largest_even_not_sum_of_two_composite_odds :
  ∀ n : ℕ, is_even n → n > 0 → (¬ (∃ a b : ℕ, is_odd a ∧ is_odd b ∧ is_composite a ∧ is_composite b ∧ n = a + b)) ↔ n = 38 := 
by
  sorry

end largest_even_not_sum_of_two_composite_odds_l94_9497


namespace probability_of_MATHEMATICS_letter_l94_9469

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_MATHEMATICS_letter :
  let total_letters := 26
  let unique_letters_count := unique_letters_in_mathematics.card
  (unique_letters_count / total_letters : ℝ) = 8 / 26 := by
  sorry

end probability_of_MATHEMATICS_letter_l94_9469


namespace inequality_solution_sets_l94_9413

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x : ℝ, ax^2 - 5 * x + b > 0 ↔ x < -1 / 3 ∨ x > 1 / 2) →
  (∀ x : ℝ, bx^2 - 5 * x + a > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end inequality_solution_sets_l94_9413


namespace quadratic_root_range_l94_9494

noncomputable def quadratic_function (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 9 * a

theorem quadratic_root_range (a : ℝ) (h : a ≠ 0) (h_distinct_roots : ∃ x1 x2 : ℝ, quadratic_function a x1 = 0 ∧ quadratic_function a x2 = 0 ∧ x1 ≠ x2 ∧ x1 < 1 ∧ x2 > 1) :
    -(2 / 11) < a ∧ a < 0 :=
sorry

end quadratic_root_range_l94_9494


namespace sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l94_9490

theorem sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms :
    let a := 63
    let b := 25
    a + b = 88 := by
  sorry

end sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l94_9490


namespace bananas_per_truck_l94_9432

theorem bananas_per_truck (total_apples total_bananas apples_per_truck : ℝ) 
  (h_total_apples: total_apples = 132.6)
  (h_apples_per_truck: apples_per_truck = 13.26)
  (h_total_bananas: total_bananas = 6.4) :
  (total_bananas / (total_apples / apples_per_truck)) = 0.64 :=
by
  sorry

end bananas_per_truck_l94_9432


namespace ratio_of_prices_l94_9452

-- Define the problem
theorem ratio_of_prices (CP SP1 SP2 : ℝ) 
  (h1 : SP1 = CP + 0.2 * CP) 
  (h2 : SP2 = CP - 0.2 * CP) : 
  SP2 / SP1 = 2 / 3 :=
by
  -- proof
  sorry

end ratio_of_prices_l94_9452


namespace part1_part2_l94_9496

theorem part1 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) : k > 3 / 4 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) (hx1x2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 → x1 * x2 = 5) : k = 2 :=
sorry

end part1_part2_l94_9496


namespace solution_for_a_l94_9471

theorem solution_for_a (x : ℝ) (a : ℝ) (h : 2 * x - a = 0) (hx : x = 1) : a = 2 := by
  rw [hx] at h
  linarith


end solution_for_a_l94_9471


namespace diagonals_of_square_equal_proof_l94_9440

-- Let us define the conditions
def square (s : Type) : Prop := True -- Placeholder for the actual definition of square
def parallelogram (p : Type) : Prop := True -- Placeholder for the actual definition of parallelogram
def diagonals_equal (q : Type) : Prop := True -- Placeholder for the property that diagonals are equal

-- Given conditions
axiom square_is_parallelogram {s : Type} (h1 : square s) : parallelogram s
axiom diagonals_of_parallelogram_equal {p : Type} (h2 : parallelogram p) : diagonals_equal p
axiom diagonals_of_square_equal {s : Type} (h3 : square s) : diagonals_equal s

-- Proof statement
theorem diagonals_of_square_equal_proof (s : Type) (h1 : square s) : diagonals_equal s :=
by
  apply diagonals_of_square_equal h1

end diagonals_of_square_equal_proof_l94_9440


namespace find_an_from_sums_l94_9405

noncomputable def geometric_sequence (a : ℕ → ℝ) (q r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_an_from_sums (a : ℕ → ℝ) (q r : ℝ) (S3 S6 : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 :  a 1 * (1 - q^3) / (1 - q) = S3) 
  (h3 : a 1 * (1 - q^6) / (1 - q) = S6) : 
  ∃ a1 q, a n = a1 * q^(n-1) := 
by
  sorry

end find_an_from_sums_l94_9405


namespace find_common_difference_l94_9468

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * (a 1 - a 0)

noncomputable def quadratic_roots (c : ℚ) (x1 x2 : ℚ) : Prop :=
2 * x1^2 - 12 * x1 + c = 0 ∧ 2 * x2^2 - 12 * x2 + c = 0

theorem find_common_difference
  (a : ℕ → ℚ) (S : ℕ → ℚ) (c : ℚ)
  (h_arith_seq: is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_roots : quadratic_roots c (a 3) (a 7))
  (h_S13 : S 13 = c) :
  (a 1 - a 0 = -3/2) ∨ (a 1 - a 0 = -7/4) :=
sorry

end find_common_difference_l94_9468


namespace find_x_l94_9478

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (hxy : x + y + x * y = 71) : x = 8 :=
sorry

end find_x_l94_9478


namespace quadratic_solution_l94_9437

theorem quadratic_solution (a : ℝ) (h : (1 : ℝ)^2 + 1 + 2 * a = 0) : a = -1 :=
by {
  sorry
}

end quadratic_solution_l94_9437


namespace jason_daily_charge_l94_9472

theorem jason_daily_charge 
  (total_cost_eric : ℕ) (days_eric : ℕ) (daily_charge : ℕ)
  (h1 : total_cost_eric = 800) (h2 : days_eric = 20)
  (h3 : daily_charge = total_cost_eric / days_eric) :
  daily_charge = 40 := 
by
  sorry

end jason_daily_charge_l94_9472


namespace annie_bought_figurines_l94_9449

theorem annie_bought_figurines:
  let televisions := 5
  let cost_per_television := 50
  let total_spent := 260
  let cost_per_figurine := 1
  let cost_of_televisions := televisions * cost_per_television
  let remaining_money := total_spent - cost_of_televisions
  remaining_money / cost_per_figurine = 10 :=
by
  sorry

end annie_bought_figurines_l94_9449


namespace opposite_of_neg_five_l94_9418

theorem opposite_of_neg_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  sorry

end opposite_of_neg_five_l94_9418


namespace rope_cut_number_not_8_l94_9467

theorem rope_cut_number_not_8 (l : ℝ) (h1 : (1 : ℝ) % l = 0) (h2 : (2 : ℝ) % l = 0) (h3 : (3 / l) ≠ 8) : False :=
by
  sorry

end rope_cut_number_not_8_l94_9467


namespace least_number_of_people_l94_9420

-- Conditions
def first_caterer_cost (x : ℕ) : ℕ := 120 + 18 * x
def second_caterer_cost (x : ℕ) : ℕ := 250 + 15 * x

-- Proof Statement
theorem least_number_of_people (x : ℕ) (h : x ≥ 44) : first_caterer_cost x > second_caterer_cost x :=
by sorry

end least_number_of_people_l94_9420


namespace angle_B_eq_pi_over_3_range_of_area_l94_9433

-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- And given vectors m and n represented as stated and are collinear
-- Prove that angle B is π/3
theorem angle_B_eq_pi_over_3
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (a b c A B C : ℝ)
  (m := (2 * Real.sin (A + C), - Real.sqrt 3))
  (n := (Real.cos (2 * B), 2 * Real.cos (B / 2) ^ 2 - 1))
  (collinear_m_n : m.1 * n.2 = m.2 * n.1) :
  B = Real.pi / 3 :=
sorry

-- Given side b = 1, find the range of the area S of triangle ABC
theorem range_of_area
  (a c A B C : ℝ)
  (triangle_area : ℝ)
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (hB : B = Real.pi / 3)
  (hb : b = 1)
  (cosine_theorem : 1 = a^2 + c^2 - a*c)
  (area_formula : triangle_area = (1/2) * a * c * Real.sin B) :
  0 < triangle_area ∧ triangle_area ≤ (Real.sqrt 3) / 4 :=
sorry

end angle_B_eq_pi_over_3_range_of_area_l94_9433


namespace cartons_in_case_l94_9466

theorem cartons_in_case (b : ℕ) (hb : b ≥ 1) (h : 2 * c * b * 500 = 1000) : c = 1 :=
by
  -- sorry is used to indicate where the proof would go
  sorry

end cartons_in_case_l94_9466


namespace work_completion_in_days_l94_9406

noncomputable def work_days_needed : ℕ :=
  let A_rate := 1 / 9
  let B_rate := 1 / 18
  let C_rate := 1 / 12
  let D_rate := 1 / 24
  let AB_rate := A_rate + B_rate
  let CD_rate := C_rate + D_rate
  let two_day_work := AB_rate + CD_rate
  let total_cycles := 24 / 7
  let total_days := (if total_cycles % 1 = 0 then total_cycles else total_cycles + 1) * 2
  total_days

theorem work_completion_in_days :
  work_days_needed = 8 :=
by
  sorry

end work_completion_in_days_l94_9406


namespace maximize_profit_l94_9407

noncomputable def profit (x : ℝ) : ℝ :=
  let selling_price := 10 + 0.5 * x
  let sales_volume := 200 - 10 * x
  (selling_price - 8) * sales_volume

theorem maximize_profit : ∃ x : ℝ, x = 8 → profit x = profit 8 ∧ (∀ y : ℝ, profit y ≤ profit 8) := 
  sorry

end maximize_profit_l94_9407


namespace vectors_parallel_implies_fraction_l94_9480

theorem vectors_parallel_implies_fraction (α : ℝ) :
  let a := (Real.sin α, 3)
  let b := (Real.cos α, 1)
  (a.1 / b.1 = 3) → (Real.sin (2 * α) / (Real.cos α) ^ 2 = 6) :=
by
  sorry

end vectors_parallel_implies_fraction_l94_9480


namespace quadratic_solution_l94_9400

theorem quadratic_solution (m : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_roots : ∀ x, 2 * x^2 + 4 * m * x + m = 0 ↔ x = x₁ ∨ x = x₂) 
  (h_sum_squares : x₁^2 + x₂^2 = 3 / 16) :
  m = -1 / 8 :=
by
  sorry

end quadratic_solution_l94_9400


namespace f_n_f_n_eq_n_l94_9438

def f : ℕ → ℕ := sorry
axiom f_def1 : f 1 = 1
axiom f_def2 : ∀ n ≥ 2, f n = n - f (f (n - 1))

theorem f_n_f_n_eq_n (n : ℕ) (hn : 0 < n) : f (n + f n) = n :=
by sorry

end f_n_f_n_eq_n_l94_9438


namespace pow_div_mul_pow_eq_l94_9412

theorem pow_div_mul_pow_eq (a b c d : ℕ) (h_a : a = 8) (h_b : b = 5) (h_c : c = 2) (h_d : d = 6) :
  (a^b / a^c) * (4^6) = 2^21 := by
  sorry

end pow_div_mul_pow_eq_l94_9412


namespace find_a_l94_9492

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 2 then x^2 - 4 else |x - 3| + a

theorem find_a (a : ℝ) (h : f (f (Real.sqrt 6) a) a = 3) : a = 2 := by
  sorry

end find_a_l94_9492
