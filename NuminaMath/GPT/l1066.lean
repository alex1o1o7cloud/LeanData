import Mathlib

namespace max_distance_point_circle_l1066_106670

open Real

noncomputable def distance (P C : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)

theorem max_distance_point_circle :
  let C : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (3, 3)
  let r : ℝ := 2
  let max_distance : ℝ := distance P C + r
  ∃ M : ℝ × ℝ, distance P M = max_distance ∧ (M.1 - 1)^2 + (M.2 - 2)^2 = r^2 :=
by
  sorry

end max_distance_point_circle_l1066_106670


namespace abs_sum_example_l1066_106627

theorem abs_sum_example : |(-8 : ℤ)| + |(-4 : ℤ)| = 12 := by
  sorry

end abs_sum_example_l1066_106627


namespace sum_of_values_of_m_l1066_106693

-- Define the inequality conditions
def condition1 (x m : ℝ) : Prop := (x - m) / 2 ≥ 0
def condition2 (x : ℝ) : Prop := x + 3 < 3 * (x - 1)

-- Define the equation constraint for y
def fractional_equation (y m : ℝ) : Prop := (3 - y) / (2 - y) + m / (y - 2) = 3

-- Sum function for the values of m
def sum_of_m (m1 m2 m3 : ℝ) : ℝ := m1 + m2 + m3

-- Main theorem
theorem sum_of_values_of_m : sum_of_m 3 (-3) (-1) = -1 := 
by { sorry }

end sum_of_values_of_m_l1066_106693


namespace find_n_given_sum_l1066_106602

noncomputable def geometric_sequence_general_term (n : ℕ) : ℝ :=
  if n ≥ 2 then 2^(2 * n - 3) else 0

def b_n (n : ℕ) : ℝ :=
  2 * n - 3

def sum_b_n (n : ℕ) : ℝ :=
  n^2 - 2 * n

theorem find_n_given_sum : ∃ n : ℕ, sum_b_n n = 360 :=
  by { use 20, sorry }

end find_n_given_sum_l1066_106602


namespace x_varies_as_z_l1066_106619

variable {x y z : ℝ}
variable (k j : ℝ)
variable (h1 : x = k * y^3)
variable (h2 : y = j * z^(1/3))

theorem x_varies_as_z (m : ℝ) (h3 : m = k * j^3) : x = m * z := by
  sorry

end x_varies_as_z_l1066_106619


namespace min_b1_b2_sum_l1066_106605

def sequence_relation (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (3 * b n + 4073) / (2 + b (n + 1))

theorem min_b1_b2_sum (b : ℕ → ℕ) (h_seq : sequence_relation b) 
  (h_b1_pos : b 1 > 0) (h_b2_pos : b 2 > 0) :
  b 1 + b 2 = 158 :=
sorry

end min_b1_b2_sum_l1066_106605


namespace largest_possible_value_for_a_l1066_106641

theorem largest_possible_value_for_a (a b c d : ℕ) 
  (h1: a < 3 * b) 
  (h2: b < 2 * c + 1) 
  (h3: c < 5 * d - 2)
  (h4: d ≤ 50) 
  (h5: d % 5 = 0) : 
  a ≤ 1481 :=
sorry

end largest_possible_value_for_a_l1066_106641


namespace number_of_valid_pairs_l1066_106624

theorem number_of_valid_pairs : ∃ p : Finset (ℕ × ℕ), 
  (∀ (a b : ℕ), (a, b) ∈ p ↔ a ≤ 10 ∧ b ≤ 10 ∧ 3 * b < a ∧ a < 4 * b) ∧ p.card = 2 :=
by
  sorry

end number_of_valid_pairs_l1066_106624


namespace locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l1066_106685

noncomputable def locus_of_C (a x0 y0 ξ η : ℝ) : Prop :=
  (x0 - ξ) * η^2 - 2 * ξ * y0 * η + ξ^3 - 3 * x0 * ξ^2 - a^2 * ξ + 3 * a^2 * x0 = 0

noncomputable def special_case (a ξ η : ℝ) : Prop :=
  ξ = 0 ∨ ξ^2 + η^2 = a^2

theorem locus_of_C_general_case_eq_cubic (a x0 y0 ξ η : ℝ) (hs: locus_of_C a x0 y0 ξ η) : 
  locus_of_C a x0 y0 ξ η := 
  sorry

theorem locus_of_C_special_case_eq_y_axis_or_circle (a ξ η : ℝ) : 
  special_case a ξ η := 
  sorry

end locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l1066_106685


namespace triangle_perimeter_is_720_l1066_106682

-- Definitions corresponding to conditions
variables (x : ℕ)
noncomputable def shortest_side := 5 * x
noncomputable def middle_side := 6 * x
noncomputable def longest_side := 7 * x

-- Given the length of the longest side is 280 cm
axiom longest_side_eq : longest_side x = 280

-- Prove that the perimeter of the triangle is 720 cm
theorem triangle_perimeter_is_720 : 
  shortest_side x + middle_side x + longest_side x = 720 :=
by
  sorry

end triangle_perimeter_is_720_l1066_106682


namespace smallest_x_l1066_106612

theorem smallest_x {
    x : ℤ
} : (x % 11 = 9) ∧ (x % 13 = 11) ∧ (x % 15 = 13) → x = 2143 := by
sorry

end smallest_x_l1066_106612


namespace largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100_l1066_106635

def is_prime (n : ℕ) : Prop := sorry -- Use inbuilt primality function or define it

def expression (n : ℕ) : ℕ := 2^n + n^2 - 1

theorem largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100 :
  ∃ m, is_prime m ∧ (∃ n, is_prime n ∧ expression n = m ∧ m < 100) ∧
        ∀ k, is_prime k ∧ (∃ n, is_prime n ∧ expression n = k ∧ k < 100) → k <= m :=
  sorry

end largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100_l1066_106635


namespace number_of_puppies_l1066_106697

def total_portions : Nat := 105
def feeding_days : Nat := 5
def feedings_per_day : Nat := 3

theorem number_of_puppies (total_portions feeding_days feedings_per_day : Nat) : 
  (total_portions / feeding_days / feedings_per_day = 7) := 
by 
  sorry

end number_of_puppies_l1066_106697


namespace five_dice_not_all_same_number_l1066_106649
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l1066_106649


namespace polynomial_division_result_l1066_106629

-- Define the given polynomials
def f (x : ℝ) : ℝ := 4 * x ^ 4 + 12 * x ^ 3 - 9 * x ^ 2 + 2 * x + 3
def d (x : ℝ) : ℝ := x ^ 2 + 2 * x - 3

-- Define the computed quotient and remainder
def q (x : ℝ) : ℝ := 4 * x ^ 2 + 4
def r (x : ℝ) : ℝ := -12 * x + 42

theorem polynomial_division_result :
  (∀ x : ℝ, f x = q x * d x + r x) ∧ (q 1 + r (-1) = 62) :=
by
  sorry

end polynomial_division_result_l1066_106629


namespace other_solution_l1066_106626

theorem other_solution (x : ℚ) (h : 30*x^2 + 13 = 47*x - 2) (hx : x = 3/5) : x = 5/6 ∨ x = 3/5 := by
  sorry

end other_solution_l1066_106626


namespace average_hit_targets_formula_average_hit_targets_ge_half_l1066_106668

open Nat Real

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n) ^ n)

theorem average_hit_targets_formula (n : ℕ) (h : 0 < n) :
  average_hit_targets n = n * (1 - (1 - (1 : ℝ) / n) ^ n) :=
by
  sorry

theorem average_hit_targets_ge_half (n : ℕ) (h : 0 < n) :
  average_hit_targets n ≥ n / 2 :=
by
  sorry

end average_hit_targets_formula_average_hit_targets_ge_half_l1066_106668


namespace increased_cost_per_person_l1066_106614

-- Declaration of constants
def initial_cost : ℕ := 30000000000 -- 30 billion dollars in dollars
def people_sharing : ℕ := 300000000 -- 300 million people
def inflation_rate : ℝ := 0.10 -- 10% inflation rate

-- Calculation of increased cost per person
theorem increased_cost_per_person : (initial_cost * (1 + inflation_rate) / people_sharing) = 110 :=
by sorry

end increased_cost_per_person_l1066_106614


namespace calculate_A_minus_B_l1066_106615

variable (A B : ℝ)
variable (h1 : A + B + B = 814.8)
variable (h2 : 10 * B = A)

theorem calculate_A_minus_B : A - B = 611.1 :=
by
  sorry

end calculate_A_minus_B_l1066_106615


namespace problem_statement_l1066_106604

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : (a - c) ^ 3 > (b - c) ^ 3 :=
by
  sorry

end problem_statement_l1066_106604


namespace multiply_polynomials_l1066_106632

theorem multiply_polynomials (x : ℂ) : 
  (x^6 + 27 * x^3 + 729) * (x^3 - 27) = x^12 + 27 * x^9 - 19683 * x^3 - 531441 :=
by
  sorry

end multiply_polynomials_l1066_106632


namespace calculate_number_of_boys_l1066_106666

theorem calculate_number_of_boys (old_average new_average misread correct_weight : ℝ) (number_of_boys : ℕ)
  (h1 : old_average = 58.4)
  (h2 : misread = 56)
  (h3 : correct_weight = 61)
  (h4 : new_average = 58.65)
  (h5 : (number_of_boys : ℝ) * old_average + (correct_weight - misread) = (number_of_boys : ℝ) * new_average) :
  number_of_boys = 20 :=
by
  sorry

end calculate_number_of_boys_l1066_106666


namespace intersecting_lines_l1066_106637

-- Definitions for the conditions
def line1 (x y a : ℝ) : Prop := x = (1/3) * y + a
def line2 (x y b : ℝ) : Prop := y = (1/3) * x + b

-- The theorem we need to prove
theorem intersecting_lines (a b : ℝ) (h1 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line1 x y a) 
                           (h2 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line2 x y b) : 
  a + b = 10 / 3 :=
sorry

end intersecting_lines_l1066_106637


namespace trigonometric_identity_l1066_106660

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.cos (π / 6 - x) = - Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 :=
by
  sorry

end trigonometric_identity_l1066_106660


namespace pascal_row_contains_prime_47_l1066_106621

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l1066_106621


namespace correct_factorization_l1066_106696

theorem correct_factorization :
  (∀ (x y : ℝ), x^2 + y^2 ≠ (x + y)^2) ∧
  (∀ (x y : ℝ), x^2 + 2*x*y + y^2 ≠ (x - y)^2) ∧
  (∀ (x : ℝ), x^2 + x ≠ x * (x - 1)) ∧
  (∀ (x y : ℝ), x^2 - y^2 = (x + y) * (x - y)) :=
by 
  sorry

end correct_factorization_l1066_106696


namespace votes_for_winner_is_744_l1066_106608

variable (V : ℝ) -- Total number of votes cast

-- Conditions
axiom two_candidates : True
axiom winner_received_62_percent : True
axiom winner_won_by_288_votes : 0.62 * V - 0.38 * V = 288

-- Theorem to prove
theorem votes_for_winner_is_744 :
  0.62 * V = 744 :=
by
  sorry

end votes_for_winner_is_744_l1066_106608


namespace functional_equation_solution_l1066_106654

noncomputable def f (x : ℝ) (c : ℝ) : ℝ :=
  (c * x - c^2) / (1 + c)

def g (x : ℝ) (c : ℝ) : ℝ :=
  c * x - c^2

theorem functional_equation_solution (f g : ℝ → ℝ) (c : ℝ) (h : c ≠ -1) :
  (∀ x y : ℝ, f (x + g y) = x * f y - y * f x + g x) ∧
  (∀ x, f x = (c * x - c^2) / (1 + c)) ∧
  (∀ x, g x = c * x - c^2) :=
sorry

end functional_equation_solution_l1066_106654


namespace sin_45_degree_l1066_106691

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l1066_106691


namespace jason_egg_consumption_in_two_weeks_l1066_106622

def breakfast_pattern : List Nat := 
  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] -- Two weeks pattern alternating 3-egg and (2+1)-egg meals

noncomputable def count_eggs (pattern : List Nat) : Nat :=
  pattern.foldl (· + ·) 0

theorem jason_egg_consumption_in_two_weeks : 
  count_eggs breakfast_pattern = 42 :=
sorry

end jason_egg_consumption_in_two_weeks_l1066_106622


namespace edward_money_left_l1066_106655

def earnings_from_lawns (lawns_mowed : Nat) (dollar_per_lawn : Nat) : Nat :=
  lawns_mowed * dollar_per_lawn

def earnings_from_gardens (gardens_cleaned : Nat) (dollar_per_garden : Nat) : Nat :=
  gardens_cleaned * dollar_per_garden

def total_earnings (earnings_lawns : Nat) (earnings_gardens : Nat) : Nat :=
  earnings_lawns + earnings_gardens

def total_expenses (fuel_expense : Nat) (equipment_expense : Nat) : Nat :=
  fuel_expense + equipment_expense

def total_earnings_with_savings (total_earnings : Nat) (savings : Nat) : Nat :=
  total_earnings + savings

def money_left (earnings_with_savings : Nat) (expenses : Nat) : Nat :=
  earnings_with_savings - expenses

theorem edward_money_left : 
  let lawns_mowed := 5
  let dollar_per_lawn := 8
  let gardens_cleaned := 3
  let dollar_per_garden := 12
  let fuel_expense := 10
  let equipment_expense := 15
  let savings := 7
  let earnings_lawns := earnings_from_lawns lawns_mowed dollar_per_lawn
  let earnings_gardens := earnings_from_gardens gardens_cleaned dollar_per_garden
  let total_earnings_work := total_earnings earnings_lawns earnings_gardens
  let expenses := total_expenses fuel_expense equipment_expense
  let earnings_with_savings := total_earnings_with_savings total_earnings_work savings
  money_left earnings_with_savings expenses = 58
:= by sorry

end edward_money_left_l1066_106655


namespace negation_proposition_l1066_106652

theorem negation_proposition :
  (∀ x : ℝ, |x - 2| + |x - 4| > 3) = ¬(∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) :=
  by sorry

end negation_proposition_l1066_106652


namespace sasha_fractions_l1066_106600

theorem sasha_fractions (x y z t : ℕ) 
  (hx : x ≠ y) (hxy : x ≠ z) (hxz : x ≠ t)
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) :
  ∃ (q1 q2 : ℚ), (q1 ≠ q2) ∧ 
    (q1 = x / y ∨ q1 = x / z ∨ q1 = x / t ∨ q1 = y / x ∨ q1 = y / z ∨ q1 = y / t ∨ q1 = z / x ∨ q1 = z / y ∨ q1 = z / t ∨ q1 = t / x ∨ q1 = t / y ∨ q1 = t / z) ∧ 
    (q2 = x / y ∨ q2 = x / z ∨ q2 = x / t ∨ q2 = y / x ∨ q2 = y / z ∨ q2 = y / t ∨ q2 = z / x ∨ q2 = z / y ∨ q2 = z / t ∨ q2 = t / x ∨ q2 = t / y ∨ q2 = t / z) ∧ 
    |q1 - q2| ≤ 11 / 60 := by 
  sorry

end sasha_fractions_l1066_106600


namespace king_then_ten_prob_l1066_106677

def num_kings : ℕ := 4
def num_tens : ℕ := 4
def deck_size : ℕ := 52
def first_card_draw_prob := (num_kings : ℚ) / (deck_size : ℚ)
def second_card_draw_prob := (num_tens : ℚ) / (deck_size - 1 : ℚ)

theorem king_then_ten_prob : 
  first_card_draw_prob * second_card_draw_prob = 4 / 663 := by
  sorry

end king_then_ten_prob_l1066_106677


namespace remainder_of_c_plus_d_l1066_106625

theorem remainder_of_c_plus_d (c d : ℕ) (k l : ℕ) 
  (hc : c = 120 * k + 114) 
  (hd : d = 180 * l + 174) : 
  (c + d) % 60 = 48 := 
by sorry

end remainder_of_c_plus_d_l1066_106625


namespace problem_I_problem_II_l1066_106661

variable (x a m : ℝ)

theorem problem_I (h: ¬ (∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0)) : 
  a < -2 ∨ a > 3 := by
  sorry

theorem problem_II (p : ∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0) (q : m-1 ≤ a ∧ a ≤ m+3) :
  ∀ a : ℝ, -2 ≤ a ∧ a ≤ 3 → m ∈ [-1, 0] := by
  sorry

end problem_I_problem_II_l1066_106661


namespace find_triples_l1066_106667

theorem find_triples (x y z : ℕ) :
  (x + 1)^(y + 1) + 1 = (x + 2)^(z + 1) ↔ (x = 1 ∧ y = 2 ∧ z = 1) :=
sorry

end find_triples_l1066_106667


namespace describe_S_is_two_rays_l1066_106628

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ common : ℝ, 
     (common = 5 ∧ (p.1 + 3 = common ∧ p.2 - 2 ≥ common ∨ p.1 + 3 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.1 + 3 ∧ (5 = common ∧ p.2 - 2 ≥ common ∨ 5 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.2 - 2 ∧ (5 = common ∧ p.1 + 3 ≥ common ∨ 5 ≥ common ∧ p.1 + 3 = common))}

theorem describe_S_is_two_rays :
  S = {p : ℝ × ℝ | (p.1 = 2 ∧ p.2 ≥ 7) ∨ (p.2 = 7 ∧ p.1 ≥ 2)} :=
  by
    sorry

end describe_S_is_two_rays_l1066_106628


namespace min_number_of_candy_kinds_l1066_106679

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l1066_106679


namespace fixed_point_inequality_l1066_106686

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * a^((x + 1) / 2) - 4

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = -1 :=
sorry

theorem inequality (a : ℝ) (x : ℝ) (h : a > 1) :
  f a (x - 3 / 4) ≥ 3 / (a^(x^2 / 2)) - 4 :=
sorry

end fixed_point_inequality_l1066_106686


namespace number_of_rectangles_l1066_106634

theorem number_of_rectangles (horizontal_lines : Fin 6) (vertical_lines : Fin 5) 
                             (point : ℕ × ℕ) (h₁ : point = (3, 4)) : 
  ∃ ways : ℕ, ways = 24 :=
by {
  sorry
}

end number_of_rectangles_l1066_106634


namespace chord_line_equation_l1066_106684

theorem chord_line_equation (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), y1^2 = -8 * x1 ∧ y2^2 = -8 * x2 ∧ (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1 ∧ y - 1 = -4 * (x + 1)) →
  4 * x + y + 3 = 0 :=
by
  sorry

end chord_line_equation_l1066_106684


namespace landscape_breadth_l1066_106613

theorem landscape_breadth (L B : ℝ) 
  (h1 : B = 6 * L) 
  (h2 : L * B = 29400) : 
  B = 420 :=
by
  sorry

end landscape_breadth_l1066_106613


namespace geometric_sequence_problem_l1066_106653

variable {a : ℕ → ℝ}
variable (r a1 : ℝ)
variable (h_pos : ∀ n, a n > 0)
variable (h_geom : ∀ n, a (n + 1) = a 1 * r ^ n)
variable (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 2025)

theorem geometric_sequence_problem :
  a 3 + a 5 = 45 :=
by
  sorry

end geometric_sequence_problem_l1066_106653


namespace tomato_land_correct_l1066_106675

-- Define the conditions
def total_land : ℝ := 4999.999999999999
def cleared_fraction : ℝ := 0.9
def grapes_fraction : ℝ := 0.1
def potato_fraction : ℝ := 0.8

-- Define the calculated values based on conditions
def cleared_land : ℝ := cleared_fraction * total_land
def grapes_land : ℝ := grapes_fraction * cleared_land
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := cleared_land - (grapes_land + potato_land)

-- Prove the question using conditions, which should end up being 450 acres.
theorem tomato_land_correct : tomato_land = 450 :=
by sorry

end tomato_land_correct_l1066_106675


namespace find_integer_l1066_106664

def satisfies_conditions (x : ℕ) (m n : ℕ) : Prop :=
  x + 100 = m ^ 2 ∧ x + 168 = n ^ 2 ∧ m > 0 ∧ n > 0

theorem find_integer (x m n : ℕ) (h : satisfies_conditions x m n) : x = 156 :=
sorry

end find_integer_l1066_106664


namespace find_k_l1066_106633
-- Import the necessary library

-- Given conditions as definitions
def circle_eq (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8 * x + y^2 + 2 * y + k = 0

def radius_sq : ℝ := 25  -- since radius = 5, radius squared is 25

-- The statement to prove
theorem find_k (x y k : ℝ) : circle_eq x y k → radius_sq = 25 → k = -8 :=
by
  sorry

end find_k_l1066_106633


namespace probability_one_first_class_product_l1066_106643

-- Define the probabilities for the interns processing first-class products
def P_first_intern_first_class : ℚ := 2 / 3
def P_second_intern_first_class : ℚ := 3 / 4

-- Define the events 
def P_A1 : ℚ := P_first_intern_first_class * (1 - P_second_intern_first_class)
def P_A2 : ℚ := (1 - P_first_intern_first_class) * P_second_intern_first_class

-- Probability of exactly one of the two parts being first-class product
def P_one_first_class_product : ℚ := P_A1 + P_A2

-- Theorem to be proven: the probability is 5/12
theorem probability_one_first_class_product : 
    P_one_first_class_product = 5 / 12 :=
by
  -- Proof goes here
  sorry

end probability_one_first_class_product_l1066_106643


namespace four_nonzero_complex_numbers_form_square_l1066_106695

open Complex

theorem four_nonzero_complex_numbers_form_square :
  ∃ (S : Finset ℂ), S.card = 4 ∧ (∀ z ∈ S, z ≠ 0) ∧ (∀ z ∈ S, ∃ (θ : ℝ), z = exp (θ * I) ∧ (exp (4 * θ * I) - z).re = 0 ∧ (exp (4 * θ * I) - z).im = cos (π / 2)) := 
sorry

end four_nonzero_complex_numbers_form_square_l1066_106695


namespace smallest_consecutive_odd_sum_l1066_106648

theorem smallest_consecutive_odd_sum (a b c d e : ℤ)
    (h1 : b = a + 2)
    (h2 : c = a + 4)
    (h3 : d = a + 6)
    (h4 : e = a + 8)
    (h5 : a + b + c + d + e = 375) : a = 71 :=
by
  -- the proof will go here
  sorry

end smallest_consecutive_odd_sum_l1066_106648


namespace find_c_l1066_106618

open Real

-- Definition of the quadratic expression in question
def expr (x y c : ℝ) : ℝ := 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 5 * x - 5 * y + 7

-- The theorem to prove that the minimum value of this expression being 0 over all (x, y) implies c = 4
theorem find_c :
  (∀ x y : ℝ, expr x y c ≥ 0) → (∃ x y : ℝ, expr x y c = 0) → c = 4 := 
by 
  sorry

end find_c_l1066_106618


namespace rational_pair_exists_l1066_106611

theorem rational_pair_exists (a b : ℚ) (h1 : a = 3/2) (h2 : b = 3) : a ≠ b ∧ a + b = a * b :=
by {
  sorry
}

end rational_pair_exists_l1066_106611


namespace line_equation_intersects_ellipse_l1066_106631

theorem line_equation_intersects_ellipse :
  ∃ l : ℝ → ℝ → Prop,
    (∀ x y : ℝ, l x y ↔ 5 * x + 4 * y - 9 = 0) ∧
    (∃ M N : ℝ × ℝ,
      (M.1^2 / 20 + M.2^2 / 16 = 1) ∧
      (N.1^2 / 20 + N.2^2 / 16 = 1) ∧
      ((M.1 + N.1) / 2 = 1) ∧
      ((M.2 + N.2) / 2 = 1)) :=
sorry

end line_equation_intersects_ellipse_l1066_106631


namespace sqrt_D_irrational_l1066_106601

theorem sqrt_D_irrational (a b c : ℤ) (h : a + 1 = b) (h_c : c = a + b) : 
  Irrational (Real.sqrt ((a^2 : ℤ) + (b^2 : ℤ) + (c^2 : ℤ))) :=
  sorry

end sqrt_D_irrational_l1066_106601


namespace zero_point_interval_l1066_106656

noncomputable def f (x : ℝ) : ℝ := Real.pi * x + Real.log x / Real.log 2

theorem zero_point_interval : 
  f (1/4) < 0 ∧ f (1/2) > 0 → ∃ x : ℝ, 1/4 ≤ x ∧ x ≤ 1/2 ∧ f x = 0 :=
by
  sorry

end zero_point_interval_l1066_106656


namespace min_ab_eq_4_l1066_106630

theorem min_ab_eq_4 (a b : ℝ) (h : 4 / a + 1 / b = Real.sqrt (a * b)) : a * b ≥ 4 :=
sorry

end min_ab_eq_4_l1066_106630


namespace probability_of_picking_letter_from_MATHEMATICS_l1066_106698

theorem probability_of_picking_letter_from_MATHEMATICS : 
  (8 : ℤ) / 26 = (4 : ℤ) / 13 :=
by
  norm_num

end probability_of_picking_letter_from_MATHEMATICS_l1066_106698


namespace daily_sale_correct_l1066_106689

-- Define the original and additional amounts in kilograms
def original_rice := 4 * 1000 -- 4 tons converted to kilograms
def additional_rice := 4000 -- kilograms
def total_rice := original_rice + additional_rice -- total amount of rice in kilograms
def days := 4 -- days to sell all the rice

-- Statement to prove: The amount to be sold each day
def daily_sale_amount := 2000 -- kilograms per day

theorem daily_sale_correct : total_rice / days = daily_sale_amount :=
by 
  -- This is a placeholder for the proof
  sorry

end daily_sale_correct_l1066_106689


namespace probability_of_six_and_queen_l1066_106663

variable {deck : Finset (ℕ × String)}
variable (sixes : Finset (ℕ × String))
variable (queens : Finset (ℕ × String))

def standard_deck : Finset (ℕ × String) := sorry

-- Condition: the deck contains 52 cards (13 hearts, 13 clubs, 13 spades, 13 diamonds)
-- and it has 4 sixes and 4 Queens.
axiom h_deck_size : standard_deck.card = 52
axiom h_sixes : ∀ c ∈ standard_deck, c.1 = 6 → c ∈ sixes
axiom h_queens : ∀ c ∈ standard_deck, c.1 = 12 → c ∈ queens

-- Define the probability function for dealing cards
noncomputable def prob_first_six_and_second_queen : ℚ :=
  (4 / 52) * (4 / 51)

theorem probability_of_six_and_queen :
  prob_first_six_and_second_queen = 4 / 663 :=
by
  sorry

end probability_of_six_and_queen_l1066_106663


namespace find_m_of_ellipse_l1066_106640

theorem find_m_of_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (10 - m) + y^2 / (m - 2) = 1) ∧ (m - 2 > 10 - m) ∧ ((4)^2 = (m - 2) - (10 - m))) → m = 14 :=
by sorry

end find_m_of_ellipse_l1066_106640


namespace can_transform_1220_to_2012_cannot_transform_1220_to_2021_l1066_106690

def can_transform (abcd : ℕ) (wxyz : ℕ) : Prop :=
  ∀ a b c d w x y z, 
  abcd = a*1000 + b*100 + c*10 + d ∧ 
  wxyz = w*1000 + x*100 + y*10 + z →
  (∃ (k : ℕ) (m : ℕ), 
    (k = a ∧ a ≠ d  ∧ m = c  ∧ c ≠ w ∧ 
     w = b + (k - b) ∧ x = c + (m - c)) ∨
    (k = w ∧ w ≠ x  ∧ m = y  ∧ y ≠ z ∧ 
     z = a + (k - a) ∧ x = d + (m - d)))
          
theorem can_transform_1220_to_2012 : can_transform 1220 2012 :=
sorry

theorem cannot_transform_1220_to_2021 : ¬ can_transform 1220 2021 :=
sorry

end can_transform_1220_to_2012_cannot_transform_1220_to_2021_l1066_106690


namespace y_is_triangular_l1066_106659

theorem y_is_triangular (k : ℕ) (hk : k > 0) : 
  ∃ n : ℕ, y = (n * (n + 1)) / 2 :=
by
  let y := (9^k - 1) / 8
  sorry

end y_is_triangular_l1066_106659


namespace number_of_shelves_l1066_106672

theorem number_of_shelves (a d S : ℕ) (h1 : a = 3) (h2 : d = 3) (h3 : S = 225) : 
  ∃ n : ℕ, (S = n * (2 * a + (n - 1) * d) / 2) ∧ (n = 15) := 
by {
  sorry
}

end number_of_shelves_l1066_106672


namespace inequality_proof_l1066_106658

theorem inequality_proof
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * b + b * c + c * a = 1) :
  (a + b) / Real.sqrt (a * b * (1 - a * b)) + 
  (b + c) / Real.sqrt (b * c * (1 - b * c)) + 
  (c + a) / Real.sqrt (c * a * (1 - c * a)) ≤ Real.sqrt 2 / (a * b * c) :=
sorry

end inequality_proof_l1066_106658


namespace find_g2_l1066_106665

open Function

variable (g : ℝ → ℝ)

axiom g_condition : ∀ x : ℝ, g x + 2 * g (1 - x) = 5 * x ^ 2

theorem find_g2 : g 2 = -10 / 3 :=
by {
  sorry
}

end find_g2_l1066_106665


namespace quadratic_other_root_l1066_106639

theorem quadratic_other_root (m x2 : ℝ) (h₁ : 1^2 - 4*1 + m = 0) (h₂ : x2^2 - 4*x2 + m = 0) : x2 = 3 :=
sorry

end quadratic_other_root_l1066_106639


namespace Walter_age_in_2003_l1066_106609

-- Defining the conditions
def Walter_age_1998 (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  walter_age_1998 = grandmother_age_1998 / 3

def birth_years_sum (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  (1998 - walter_age_1998) + (1998 - grandmother_age_1998) = 3858

-- Defining the theorem to be proved
theorem Walter_age_in_2003 (walter_age_1998 grandmother_age_1998 : ℝ) 
  (h1 : Walter_age_1998 walter_age_1998 grandmother_age_1998) 
  (h2 : birth_years_sum walter_age_1998 grandmother_age_1998) : 
  walter_age_1998 + 5 = 39.5 :=
  sorry

end Walter_age_in_2003_l1066_106609


namespace count_of_changing_quantities_l1066_106623

-- Definitions of the problem conditions
def length_AC_unchanged : Prop := ∀ P A B C D : ℝ, true
def perimeter_square_unchanged : Prop := ∀ P A B C D : ℝ, true
def area_square_unchanged : Prop := ∀ P A B C D : ℝ, true
def area_quadrilateral_changed : Prop := ∀ P A B C D M N : ℝ, true

-- The main theorem to prove
theorem count_of_changing_quantities :
  length_AC_unchanged ∧
  perimeter_square_unchanged ∧
  area_square_unchanged ∧
  area_quadrilateral_changed →
  (1 = 1) :=
by
  sorry

end count_of_changing_quantities_l1066_106623


namespace arithmetic_sequence_15th_term_l1066_106678

theorem arithmetic_sequence_15th_term :
  ∀ (a d n : ℕ), a = 3 → d = 13 - a → n = 15 → 
  a + (n - 1) * d = 143 :=
by
  intros a d n ha hd hn
  rw [ha, hd, hn]
  sorry

end arithmetic_sequence_15th_term_l1066_106678


namespace birth_age_of_mother_l1066_106617

def harrys_age : ℕ := 50

def fathers_age (h : ℕ) : ℕ := h + 24

def mothers_age (f h : ℕ) : ℕ := f - h / 25

theorem birth_age_of_mother (h f m : ℕ) (H1 : h = harrys_age)
  (H2 : f = fathers_age h) (H3 : m = mothers_age f h) :
  m - h = 22 := sorry

end birth_age_of_mother_l1066_106617


namespace suitable_chart_for_air_composition_l1066_106687

/-- Given that air is a mixture of various gases, prove that the most suitable
    type of statistical chart to depict this data, while introducing it
    succinctly and effectively, is a pie chart. -/
theorem suitable_chart_for_air_composition :
  ∀ (air_composition : String) (suitable_for_introduction : String → Prop),
  (air_composition = "mixture of various gases") →
  (suitable_for_introduction "pie chart") →
  suitable_for_introduction "pie chart" :=
by
  intros air_composition suitable_for_introduction h_air_composition h_pie_chart
  sorry

end suitable_chart_for_air_composition_l1066_106687


namespace triangle_properties_l1066_106607

open Real

noncomputable def vec_m (a : ℝ) : ℝ × ℝ := (2 * sin (a / 2), sqrt 3)
noncomputable def vec_n (a : ℝ) : ℝ × ℝ := (cos a, 2 * cos (a / 4)^2 - 1)
noncomputable def area_triangle := 3 * sqrt 3 / 2

theorem triangle_properties (a b c : ℝ) (A : ℝ)
  (ha : a = sqrt 7)
  (hA : (1 / 2) * b * c * sin A = area_triangle)
  (hparallel : vec_m A = vec_n A) :
  A = π / 3 ∧ b + c = 5 :=
by
  sorry

end triangle_properties_l1066_106607


namespace fraction_of_subsets_l1066_106650

theorem fraction_of_subsets (S T : ℕ) (hS : S = 2^10) (hT : T = Nat.choose 10 3) :
    (T:ℚ) / (S:ℚ) = 15 / 128 :=
by sorry

end fraction_of_subsets_l1066_106650


namespace mul_powers_same_base_l1066_106669

theorem mul_powers_same_base (x : ℝ) : (x ^ 8) * (x ^ 2) = x ^ 10 :=
by
  exact sorry

end mul_powers_same_base_l1066_106669


namespace find_unit_prices_l1066_106638

theorem find_unit_prices (price_A price_B : ℕ) 
  (h1 : price_A = price_B + 5) 
  (h2 : 1000 / price_A = 750 / price_B) : 
  price_A = 20 ∧ price_B = 15 := 
by 
  sorry

end find_unit_prices_l1066_106638


namespace john_marbles_selection_l1066_106657

theorem john_marbles_selection :
  let total_marbles := 15
  let special_colors := 4
  let total_chosen := 5
  let chosen_special_colors := 2
  let remaining_colors := total_marbles - special_colors
  let chosen_normal_colors := total_chosen - chosen_special_colors
  (Nat.choose 4 2) * (Nat.choose 11 3) = 990 :=
by
  sorry

end john_marbles_selection_l1066_106657


namespace euler_distance_formula_l1066_106616

theorem euler_distance_formula 
  (d R r : ℝ) 
  (h₁ : d = distance_between_centers_of_inscribed_and_circumscribed_circles_of_triangle)
  (h₂ : R = circumradius_of_triangle)
  (h₃ : r = inradius_of_triangle) : 
  d^2 = R^2 - 2 * R * r := 
sorry

end euler_distance_formula_l1066_106616


namespace undefined_values_of_fraction_l1066_106620

theorem undefined_values_of_fraction (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end undefined_values_of_fraction_l1066_106620


namespace marketing_percentage_l1066_106674

-- Define the conditions
variable (monthly_budget : ℝ)
variable (rent : ℝ := monthly_budget / 5)
variable (remaining_after_rent : ℝ := monthly_budget - rent)
variable (food_beverages : ℝ := remaining_after_rent / 4)
variable (remaining_after_food_beverages : ℝ := remaining_after_rent - food_beverages)
variable (employee_salaries : ℝ := remaining_after_food_beverages / 3)
variable (remaining_after_employee_salaries : ℝ := remaining_after_food_beverages - employee_salaries)
variable (utilities : ℝ := remaining_after_employee_salaries / 7)
variable (remaining_after_utilities : ℝ := remaining_after_employee_salaries - utilities)
variable (marketing : ℝ := 0.15 * remaining_after_utilities)

-- Define the theorem we want to prove
theorem marketing_percentage : marketing / monthly_budget * 100 = 5.14 := by
  sorry

end marketing_percentage_l1066_106674


namespace envelope_weight_l1066_106610

theorem envelope_weight :
  (7.225 * 1000) / 850 = 8.5 :=
by
  sorry

end envelope_weight_l1066_106610


namespace diplomats_not_speaking_russian_l1066_106673

-- Definitions to formalize the problem
def total_diplomats : ℕ := 150
def speak_french : ℕ := 17
def speak_both_french_and_russian : ℕ := (10 * total_diplomats) / 100
def speak_neither_french_nor_russian : ℕ := (20 * total_diplomats) / 100

-- Theorem to prove the desired quantity
theorem diplomats_not_speaking_russian : 
  speak_neither_french_nor_russian + (speak_french - speak_both_french_and_russian) = 32 := by
  sorry

end diplomats_not_speaking_russian_l1066_106673


namespace min_orders_to_minimize_spent_l1066_106645

-- Definitions for the given conditions
def original_price (n p : ℕ) : ℕ := n * p
def discounted_price (T : ℕ) : ℕ := (3 * T) / 5  -- Equivalent to 0.6 * T, using integer math

-- Define the conditions
theorem min_orders_to_minimize_spent 
  (n p : ℕ)
  (h1 : n = 42)
  (h2 : p = 48)
  : ∃ m : ℕ, m = 3 :=
by 
  sorry

end min_orders_to_minimize_spent_l1066_106645


namespace sara_has_total_quarters_l1066_106603

-- Define the number of quarters Sara originally had
def original_quarters : ℕ := 21

-- Define the number of quarters Sara's dad gave her
def added_quarters : ℕ := 49

-- Define the total number of quarters Sara has now
def total_quarters : ℕ := original_quarters + added_quarters

-- Prove that the total number of quarters is 70
theorem sara_has_total_quarters : total_quarters = 70 := by
  -- This is where the proof would go
  sorry

end sara_has_total_quarters_l1066_106603


namespace log_expression_value_l1066_106651

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_expression_value :
  log_base 10 3 + 3 * log_base 10 2 + 2 * log_base 10 5 + 4 * log_base 10 3 + log_base 10 9 = 5.34 :=
by
  sorry

end log_expression_value_l1066_106651


namespace albert_needs_more_money_l1066_106671

def cost_of_paintbrush : ℝ := 1.50
def cost_of_paints : ℝ := 4.35
def cost_of_easel : ℝ := 12.65
def amount_already_has : ℝ := 6.50

theorem albert_needs_more_money : 
  (cost_of_paintbrush + cost_of_paints + cost_of_easel) - amount_already_has = 12.00 := 
by
  sorry

end albert_needs_more_money_l1066_106671


namespace arrangements_APPLE_is_60_l1066_106642

-- Definition of the problem statement based on the given conditions
def distinct_arrangements_APPLE : Nat :=
  let n := 5
  let n_A := 1
  let n_P := 2
  let n_L := 1
  let n_E := 1
  (n.factorial / (n_A.factorial * n_P.factorial * n_L.factorial * n_E.factorial))

-- The proof statement (without the proof itself, which is "sorry")
theorem arrangements_APPLE_is_60 : distinct_arrangements_APPLE = 60 := by
  sorry

end arrangements_APPLE_is_60_l1066_106642


namespace problem1_problem2_problem3_l1066_106676

-- Problem (1)
theorem problem1 : -36 * (5 / 4 - 5 / 6 - 11 / 12) = 18 := by
  sorry

-- Problem (2)
theorem problem2 : (-2) ^ 2 - 3 * (-1) ^ 3 + 0 * (-2) ^ 3 = 7 := by
  sorry

-- Problem (3)
theorem problem3 (x : ℚ) (y : ℚ) (h1 : x = -2) (h2 : y = 1 / 2) : 
    (3 / 2) * x^2 * y + x * y^2 = 5 / 2 := by
  sorry

end problem1_problem2_problem3_l1066_106676


namespace total_cost_jello_l1066_106688

def total_cost_james_spent : Real := 259.20

theorem total_cost_jello 
  (pounds_per_cubic_foot : ℝ := 8)
  (gallons_per_cubic_foot : ℝ := 7.5)
  (tablespoons_per_pound : ℝ := 1.5)
  (cost_red_jello : ℝ := 0.50)
  (cost_blue_jello : ℝ := 0.40)
  (cost_green_jello : ℝ := 0.60)
  (percentage_red_jello : ℝ := 0.60)
  (percentage_blue_jello : ℝ := 0.30)
  (percentage_green_jello : ℝ := 0.10)
  (volume_cubic_feet : ℝ := 6) :
  (volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_red_jello * cost_red_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_blue_jello * cost_blue_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_green_jello * cost_green_jello) = total_cost_james_spent :=
by
  sorry

end total_cost_jello_l1066_106688


namespace melinda_probability_correct_l1066_106681

def probability_two_digit_between_20_and_30 : ℚ :=
  11 / 36

theorem melinda_probability_correct :
  probability_two_digit_between_20_and_30 = 11 / 36 :=
by
  sorry

end melinda_probability_correct_l1066_106681


namespace total_jelly_beans_l1066_106606

-- Given conditions:
def vanilla_jelly_beans : ℕ := 120
def grape_jelly_beans := 5 * vanilla_jelly_beans + 50

-- The proof problem:
theorem total_jelly_beans :
  vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  -- Proof steps would go here
  sorry

end total_jelly_beans_l1066_106606


namespace calculate_value_l1066_106680

theorem calculate_value (x y d : ℕ) (hx : x = 2024) (hy : y = 1935) (hd : d = 225) : 
  (x - y)^2 / d = 35 := by
  sorry

end calculate_value_l1066_106680


namespace system_solution_l1066_106647

theorem system_solution (x y z : ℝ) 
  (h1 : 2 * x - 3 * y + z = 8) 
  (h2 : 4 * x - 6 * y + 2 * z = 16) 
  (h3 : x + y - z = 1) : 
  x = 11 / 3 ∧ y = 1 ∧ z = 11 / 3 :=
by
  sorry

end system_solution_l1066_106647


namespace sally_bread_consumption_l1066_106644

theorem sally_bread_consumption :
  (2 * 2) + (1 * 2) = 6 :=
by
  sorry

end sally_bread_consumption_l1066_106644


namespace find_g_seven_l1066_106699

noncomputable def g : ℝ → ℝ :=
  sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_six : g 6 = 7

theorem find_g_seven : g 7 = 49 / 6 :=
by
  -- Proof omitted here
  sorry

end find_g_seven_l1066_106699


namespace last_digit_3_pow_1991_plus_1991_pow_3_l1066_106683

theorem last_digit_3_pow_1991_plus_1991_pow_3 :
  (3 ^ 1991 + 1991 ^ 3) % 10 = 8 :=
  sorry

end last_digit_3_pow_1991_plus_1991_pow_3_l1066_106683


namespace divisibility_l1066_106646

theorem divisibility (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  (n^5 + 1) ∣ (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4 * k - 1) := 
sorry

end divisibility_l1066_106646


namespace no_stew_left_l1066_106692

theorem no_stew_left (company : Type) (stew : ℝ)
    (one_third_stayed : ℝ)
    (two_thirds_went : ℝ)
    (camp_consumption : ℝ)
    (range_consumption_per_portion : ℝ)
    (range_portion_multiplier : ℝ)
    (total_stew : ℝ) : 
    one_third_stayed = 1 / 3 →
    two_thirds_went = 2 / 3 →
    camp_consumption = 1 / 4 →
    range_portion_multiplier = 1.5 →
    total_stew = camp_consumption + (range_portion_multiplier * (two_thirds_went * (camp_consumption / one_third_stayed))) →
    total_stew = 1 →
    stew = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- here would be the proof steps
  sorry

end no_stew_left_l1066_106692


namespace gwen_remaining_money_l1066_106662

theorem gwen_remaining_money:
  ∀ (Gwen_received Gwen_spent Gwen_remaining: ℕ),
    Gwen_received = 5 →
    Gwen_spent = 3 →
    Gwen_remaining = Gwen_received - Gwen_spent →
    Gwen_remaining = 2 :=
by
  intros Gwen_received Gwen_spent Gwen_remaining h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end gwen_remaining_money_l1066_106662


namespace arithmetic_sequence_common_difference_l1066_106694

theorem arithmetic_sequence_common_difference (a_1 a_5 d : ℝ) 
  (h1 : a_5 = a_1 + 4 * d) 
  (h2 : a_1 + (a_1 + d) + (a_1 + 2 * d) = 6) : 
  d = 2 := 
  sorry

end arithmetic_sequence_common_difference_l1066_106694


namespace find_base_s_l1066_106636

-- Definitions based on the conditions.
def five_hundred_thirty_base (s : ℕ) : ℕ := 5 * s^2 + 3 * s
def four_hundred_fifty_base (s : ℕ) : ℕ := 4 * s^2 + 5 * s
def one_thousand_one_hundred_base (s : ℕ) : ℕ := s^3 + s^2

-- The theorem to prove.
theorem find_base_s : (∃ s : ℕ, five_hundred_thirty_base s + four_hundred_fifty_base s = one_thousand_one_hundred_base s) → s = 8 :=
by
  sorry

end find_base_s_l1066_106636
