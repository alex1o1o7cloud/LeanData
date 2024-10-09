import Mathlib

namespace possible_to_select_three_numbers_l1847_184718

theorem possible_to_select_three_numbers (n : ℕ) (a : ℕ → ℕ) (h : ∀ i j, i < j → a i < a j) (h_bound : ∀ i, a i < 2 * n) :
  ∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ a i + a j = a k := sorry

end possible_to_select_three_numbers_l1847_184718


namespace A_div_B_l1847_184798

noncomputable def A : ℝ := 
  ∑' n, if n % 2 = 0 ∧ n % 4 ≠ 0 then 1 / (n:ℝ)^2 else 0

noncomputable def B : ℝ := 
  ∑' n, if n % 4 = 0 then (-1)^(n / 4 + 1) * 1 / (n:ℝ)^2 else 0

theorem A_div_B : A / B = 17 := by
  sorry

end A_div_B_l1847_184798


namespace solve_equation_l1847_184787

theorem solve_equation (x : ℝ) : (x + 4)^2 = 5 * (x + 4) ↔ (x = -4 ∨ x = 1) :=
by sorry

end solve_equation_l1847_184787


namespace express_in_scientific_notation_l1847_184797

def scientific_notation_of_160000 : Prop :=
  160000 = 1.6 * 10^5

theorem express_in_scientific_notation : scientific_notation_of_160000 :=
  sorry

end express_in_scientific_notation_l1847_184797


namespace necessary_condition_l1847_184734

theorem necessary_condition (x : ℝ) : x = 1 → x^2 = 1 :=
by
  sorry

end necessary_condition_l1847_184734


namespace general_formula_sum_of_first_n_terms_max_sum_of_S_n_sum_of_absolute_values_l1847_184775

section ArithmeticSequence

-- Given conditions
def a1 : Int := 13
def a4 : Int := 7
def d : Int := (a4 - a1) / 3

-- General formula for a_n
def a_n (n : Int) : Int := a1 + (n - 1) * d

-- Sum of the first n terms S_n
def S_n (n : Int) : Int := n * (a1 + a_n n) / 2

-- Maximum value of S_n and corresponding term
def S_max : Int := 49
def n_max_S : Int := 7

-- Sum of the absolute values of the first n terms T_n
def T_n (n : Int) : Int :=
  if n ≤ 7 then n^2 + 12 * n
  else 98 - 12 * n - n^2

-- Statements to prove
theorem general_formula (n : Int) : a_n n = 15 - 2 * n := sorry

theorem sum_of_first_n_terms (n : Int) : S_n n = 14 * n - n^2 := sorry

theorem max_sum_of_S_n : (S_n n_max_S = S_max) := sorry

theorem sum_of_absolute_values (n : Int) : T_n n = 
  if n ≤ 7 then n^2 + 12 * n else 98 - 12 * n - n^2 := sorry

end ArithmeticSequence

end general_formula_sum_of_first_n_terms_max_sum_of_S_n_sum_of_absolute_values_l1847_184775


namespace production_value_decreased_by_10_percent_l1847_184739

variable (a : ℝ)

def production_value_in_January : ℝ := a

def production_value_in_February (a : ℝ) : ℝ := 0.9 * a

theorem production_value_decreased_by_10_percent (a : ℝ) :
  production_value_in_February a = 0.9 * production_value_in_January a := 
by
  sorry

end production_value_decreased_by_10_percent_l1847_184739


namespace complement_M_eq_interval_l1847_184748

-- Definition of the set M
def M : Set ℝ := { x | x * (x - 3) > 0 }

-- Universal set is ℝ
def U : Set ℝ := Set.univ

-- Theorem to prove the complement of M in ℝ is [0, 3]
theorem complement_M_eq_interval :
  U \ M = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end complement_M_eq_interval_l1847_184748


namespace weight_of_second_triangle_l1847_184706

theorem weight_of_second_triangle :
  let side_len1 := 4
  let density1 := 0.9
  let weight1 := 10.8
  let side_len2 := 6
  let density2 := 1.2
  let weight2 := 18.7
  let area1 := (side_len1 ^ 2 * Real.sqrt 3) / 4
  let area2 := (side_len2 ^ 2 * Real.sqrt 3) / 4
  let calc_weight1 := area1 * density1
  let calc_weight2 := area2 * density2
  calc_weight1 = weight1 → calc_weight2 = weight2 := 
by
  intros
  -- Proof logic goes here
  sorry

end weight_of_second_triangle_l1847_184706


namespace shortest_side_of_right_triangle_l1847_184741

theorem shortest_side_of_right_triangle 
  (a b : ℕ) (ha : a = 7) (hb : b = 10) (c : ℝ) (hright : a^2 + b^2 = c^2) :
  min a b = 7 :=
by
  sorry

end shortest_side_of_right_triangle_l1847_184741


namespace tricycle_wheel_count_l1847_184703

theorem tricycle_wheel_count (bicycles wheels_per_bicycle tricycles total_wheels : ℕ)
  (h1 : bicycles = 16)
  (h2 : wheels_per_bicycle = 2)
  (h3 : tricycles = 7)
  (h4 : total_wheels = 53)
  (h5 : total_wheels = (bicycles * wheels_per_bicycle) + (tricycles * (3 : ℕ))) : 
  (3 : ℕ) = 3 := by
  sorry

end tricycle_wheel_count_l1847_184703


namespace m_add_n_equals_19_l1847_184712

theorem m_add_n_equals_19 (n m : ℕ) (A_n_m : ℕ) (C_n_m : ℕ) (h1 : A_n_m = 272) (h2 : C_n_m = 136) :
  m + n = 19 :=
by
  sorry

end m_add_n_equals_19_l1847_184712


namespace factorize_expression_l1847_184746

theorem factorize_expression (a : ℝ) : 
  a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end factorize_expression_l1847_184746


namespace right_triangle_satisfies_pythagorean_l1847_184799

-- Definition of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove
theorem right_triangle_satisfies_pythagorean :
  a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_satisfies_pythagorean_l1847_184799


namespace find_quadratic_function_l1847_184783

noncomputable def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b

theorem find_quadratic_function (a b : ℝ) :
  (∀ x, (quadratic_function a b (quadratic_function a b x - x)) / (quadratic_function a b x) = x^2 + 2023 * x + 1777) →
  a = 2025 ∧ b = 249 :=
by
  intro h
  sorry

end find_quadratic_function_l1847_184783


namespace super12_teams_l1847_184730

theorem super12_teams :
  ∃ n : ℕ, (n * (n - 1) = 132) ∧ n = 12 := by
  sorry

end super12_teams_l1847_184730


namespace max_distance_circle_ellipse_l1847_184723

theorem max_distance_circle_ellipse:
  (∀ P Q : ℝ × ℝ, 
     (P.1^2 + (P.2 - 3)^2 = 1 / 4) → 
     (Q.1^2 + 4 * Q.2^2 = 4) → 
     ∃ Q_max : ℝ × ℝ, 
         Q_max = (0, -1) ∧ 
         (∀ P : ℝ × ℝ, P.1^2 + (P.2 - 3)^2 = 1 / 4 →
         |dist P Q_max| = 9 / 2)) := 
sorry

end max_distance_circle_ellipse_l1847_184723


namespace split_enthusiasts_into_100_sections_l1847_184782

theorem split_enthusiasts_into_100_sections :
  ∃ (sections : Fin 100 → Set ℕ),
    (∀ i, sections i ≠ ∅) ∧
    (∀ i j, i ≠ j → sections i ∩ sections j = ∅) ∧
    (⋃ i, sections i) = {n : ℕ | n < 5000} :=
sorry

end split_enthusiasts_into_100_sections_l1847_184782


namespace cannot_determine_red_marbles_l1847_184774

variable (Jason_blue : ℕ) (Tom_blue : ℕ) (Total_blue : ℕ)

-- Conditions
axiom Jason_has_44_blue : Jason_blue = 44
axiom Tom_has_24_blue : Tom_blue = 24
axiom Together_have_68_blue : Total_blue = 68

theorem cannot_determine_red_marbles (Jason_blue Tom_blue Total_blue : ℕ) : ¬ ∃ (Jason_red : ℕ), True := by
  sorry

end cannot_determine_red_marbles_l1847_184774


namespace factorization_m_minus_n_l1847_184793

theorem factorization_m_minus_n :
  ∃ (m n : ℤ), (6 * (x:ℝ)^2 - 5 * x - 6 = (6 * x + m) * (x + n)) ∧ (m - n = 5) :=
by {
  sorry
}

end factorization_m_minus_n_l1847_184793


namespace total_players_l1847_184788

-- Definitions based on problem conditions.
def players_kabadi : Nat := 10
def players_kho_kho_only : Nat := 20
def players_both_games : Nat := 5

-- Proof statement for the total number of players.
theorem total_players : (players_kabadi + players_kho_kho_only - players_both_games) = 25 := by
  sorry

end total_players_l1847_184788


namespace apples_count_l1847_184700

def mangoes_oranges_apples_ratio (mangoes oranges apples : Nat) : Prop :=
  mangoes / 10 = oranges / 2 ∧ mangoes / 10 = apples / 3

theorem apples_count (mangoes oranges apples : Nat) (h_ratio : mangoes_oranges_apples_ratio mangoes oranges apples) (h_mangoes : mangoes = 120) : apples = 36 :=
by
  sorry

end apples_count_l1847_184700


namespace parallel_lines_m_values_l1847_184707

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 ↔ mx + 3 * y - 2 = 0) → (m = -3 ∨ m = 2) :=
by
  sorry

end parallel_lines_m_values_l1847_184707


namespace average_rate_of_change_l1847_184784

noncomputable def f (x : ℝ) : ℝ := x^2 + x

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l1847_184784


namespace max_minus_min_depends_on_a_not_b_l1847_184786

def quadratic_function (a b x : ℝ) : ℝ := x^2 + a * x + b

theorem max_minus_min_depends_on_a_not_b (a b : ℝ) :
  let f := quadratic_function a b
  let M := max (f 0) (f 1)
  let m := min (f 0) (f 1)
  M - m == |a| :=
sorry

end max_minus_min_depends_on_a_not_b_l1847_184786


namespace log_term_evaluation_l1847_184770

theorem log_term_evaluation : (Real.log 2)^2 + (Real.log 5)^2 + 2 * (Real.log 2) * (Real.log 5) = 1 := by
  sorry

end log_term_evaluation_l1847_184770


namespace problem_statement_l1847_184733

variable { a b c x y z : ℝ }

theorem problem_statement 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) : 
  a * x + b * y + c * z ≥ 0 :=
by 
  sorry

end problem_statement_l1847_184733


namespace midpoint_square_sum_l1847_184753

theorem midpoint_square_sum (x y : ℝ) :
  (4, 1) = ((2 + x) / 2, (6 + y) / 2) → x^2 + y^2 = 52 :=
by
  sorry

end midpoint_square_sum_l1847_184753


namespace rain_in_both_areas_l1847_184729

variable (P1 P2 : ℝ)
variable (hP1 : 0 < P1 ∧ P1 < 1)
variable (hP2 : 0 < P2 ∧ P2 < 1)

theorem rain_in_both_areas :
  ∀ P1 P2, (0 < P1 ∧ P1 < 1) → (0 < P2 ∧ P2 < 1) → (1 - P1) * (1 - P2) = (1 - P1) * (1 - P2) :=
by
  intros P1 P2 hP1 hP2
  sorry

end rain_in_both_areas_l1847_184729


namespace triangular_difference_l1847_184773

/-- Definition of triangular numbers -/
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Main theorem: the difference between the 30th and 29th triangular numbers is 30 -/
theorem triangular_difference : triangular 30 - triangular 29 = 30 :=
by
  sorry

end triangular_difference_l1847_184773


namespace series_converges_l1847_184749

theorem series_converges (u : ℕ → ℝ) (h : ∀ n, u n = n / (3 : ℝ)^n) :
  ∃ l, 0 ≤ l ∧ l < 1 ∧ ∑' n, u n = l := by
  sorry

end series_converges_l1847_184749


namespace algebraic_expression_value_l1847_184732

theorem algebraic_expression_value
  (a : ℝ) 
  (h : a^2 + 2 * a - 1 = 0) : 
  -a^2 - 2 * a + 8 = 7 :=
by 
  sorry

end algebraic_expression_value_l1847_184732


namespace initial_books_l1847_184759

theorem initial_books (sold_books : ℕ) (given_books : ℕ) (remaining_books : ℕ) 
                      (h1 : sold_books = 11)
                      (h2 : given_books = 35)
                      (h3 : remaining_books = 62) :
  (sold_books + given_books + remaining_books = 108) :=
by
  -- Proof skipped
  sorry

end initial_books_l1847_184759


namespace fraction_of_is_l1847_184710

theorem fraction_of_is (a b c d e : ℚ) (h1 : a = 2) (h2 : b = 9) (h3 : c = 3) (h4 : d = 4) (h5 : e = 8/27) :
  (a / b) = e * (c / d) := 
sorry

end fraction_of_is_l1847_184710


namespace min_ab_square_is_four_l1847_184736

noncomputable def min_ab_square : Prop :=
  ∃ a b : ℝ, (a^2 + b^2 = 4 ∧ ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0)

theorem min_ab_square_is_four : min_ab_square :=
  sorry

end min_ab_square_is_four_l1847_184736


namespace length_RS_l1847_184725

open Real

-- Given definitions and conditions
def PQ : ℝ := 10
def PR : ℝ := 10
def QR : ℝ := 5
def PS : ℝ := 13

-- Prove the length of RS
theorem length_RS : ∃ (RS : ℝ), RS = 6.17362 := by
  sorry

end length_RS_l1847_184725


namespace treaty_signed_on_saturday_l1847_184744

-- Define the start day and the total days until the treaty.
def start_day_of_week : Nat := 4 -- Thursday is the 4th day (0 = Sunday, ..., 6 = Saturday)
def days_until_treaty : Nat := 919

-- Calculate the final day of the week after 919 days since start_day_of_week.
def treaty_day_of_week : Nat := (start_day_of_week + days_until_treaty) % 7

-- The goal is to prove that the treaty was signed on a Saturday.
theorem treaty_signed_on_saturday : treaty_day_of_week = 6 :=
by
  -- Implement the proof steps
  sorry

end treaty_signed_on_saturday_l1847_184744


namespace total_oil_leaked_correct_l1847_184795

-- Definitions of given conditions.
def initial_leak_A : ℕ := 6522
def leak_rate_A : ℕ := 257
def time_A : ℕ := 20

def initial_leak_B : ℕ := 3894
def leak_rate_B : ℕ := 182
def time_B : ℕ := 15

def initial_leak_C : ℕ := 1421
def leak_rate_C : ℕ := 97
def time_C : ℕ := 12

-- Total additional leaks calculation.
def additional_leak (rate time : ℕ) : ℕ := rate * time
def additional_leak_A : ℕ := additional_leak leak_rate_A time_A
def additional_leak_B : ℕ := additional_leak leak_rate_B time_B
def additional_leak_C : ℕ := additional_leak leak_rate_C time_C

-- Total leaks from each pipe.
def total_leak_A : ℕ := initial_leak_A + additional_leak_A
def total_leak_B : ℕ := initial_leak_B + additional_leak_B
def total_leak_C : ℕ := initial_leak_C + additional_leak_C

-- Total oil leaked.
def total_oil_leaked : ℕ := total_leak_A + total_leak_B + total_leak_C

-- The proof problem statement.
theorem total_oil_leaked_correct : total_oil_leaked = 20871 := by
  sorry

end total_oil_leaked_correct_l1847_184795


namespace rectangle_area_l1847_184755

theorem rectangle_area (AB AC : ℝ) (h_AB : AB = 15) (h_AC : AC = 17) : ∃ Area : ℝ, Area = 120 :=
by
  sorry

end rectangle_area_l1847_184755


namespace possible_values_of_sum_of_reciprocals_l1847_184767

theorem possible_values_of_sum_of_reciprocals {a b : ℝ} (h1 : a * b > 0) (h2 : a + b = 1) : 
  1 / a + 1 / b = 4 := 
by 
  sorry

end possible_values_of_sum_of_reciprocals_l1847_184767


namespace find_n_l1847_184771

theorem find_n (n : ℤ) (h₁ : 50 ≤ n ∧ n ≤ 120)
               (h₂ : n % 8 = 0)
               (h₃ : n % 12 = 4)
               (h₄ : n % 7 = 4) : 
  n = 88 :=
sorry

end find_n_l1847_184771


namespace profit_function_l1847_184747

def cost_per_unit : ℝ := 8

def daily_sales_quantity (x : ℝ) : ℝ := -x + 30

def profit_per_unit (x : ℝ) : ℝ := x - cost_per_unit

def total_profit (x : ℝ) : ℝ := (profit_per_unit x) * (daily_sales_quantity x)

theorem profit_function (x : ℝ) : total_profit x = -x^2 + 38*x - 240 :=
  sorry

end profit_function_l1847_184747


namespace factorization_correct_l1847_184728

noncomputable def factor_polynomial : Polynomial ℝ :=
  Polynomial.X^6 - 64

theorem factorization_correct : 
  factor_polynomial = 
  (Polynomial.X - 2) * 
  (Polynomial.X + 2) * 
  (Polynomial.X^4 + 4 * Polynomial.X^2 + 16) :=
by
  sorry

end factorization_correct_l1847_184728


namespace gemstone_necklaces_sold_correct_l1847_184701

-- Define the conditions
def bead_necklaces_sold : Nat := 4
def necklace_cost : Nat := 3
def total_earnings : Nat := 21
def bead_necklaces_earnings : Nat := bead_necklaces_sold * necklace_cost
def gemstone_necklaces_earnings : Nat := total_earnings - bead_necklaces_earnings
def gemstone_necklaces_sold : Nat := gemstone_necklaces_earnings / necklace_cost

-- Theorem to prove the number of gem stone necklaces sold
theorem gemstone_necklaces_sold_correct :
  gemstone_necklaces_sold = 3 :=
by
  -- Proof omitted
  sorry

end gemstone_necklaces_sold_correct_l1847_184701


namespace math_problem_l1847_184719

theorem math_problem (x : ℝ) (h : x = 0.18 * 4750) : 1.5 * x = 1282.5 :=
by
  sorry

end math_problem_l1847_184719


namespace intersection_eq_set_l1847_184762

def M : Set ℤ := { x | -4 < (x : Int) ∧ x < 2 }
def N : Set Int := { x | (x : ℝ) ^ 2 < 4 }
def intersection := M ∩ N

theorem intersection_eq_set : intersection = {-1, 0, 1} := 
sorry

end intersection_eq_set_l1847_184762


namespace dino_remaining_money_l1847_184754

-- Definitions of the conditions
def hours_gig_1 : ℕ := 20
def hourly_rate_gig_1 : ℕ := 10

def hours_gig_2 : ℕ := 30
def hourly_rate_gig_2 : ℕ := 20

def hours_gig_3 : ℕ := 5
def hourly_rate_gig_3 : ℕ := 40

def expenses : ℕ := 500

-- The theorem to be proved: Dino's remaining money at the end of the month
theorem dino_remaining_money : 
  (hours_gig_1 * hourly_rate_gig_1 + hours_gig_2 * hourly_rate_gig_2 + hours_gig_3 * hourly_rate_gig_3) - expenses = 500 := by
  sorry

end dino_remaining_money_l1847_184754


namespace long_jump_record_l1847_184763

theorem long_jump_record 
  (standard_distance : ℝ)
  (jump1 : ℝ)
  (jump2 : ℝ)
  (record1 : ℝ)
  (record2 : ℝ)
  (h1 : standard_distance = 4.00)
  (h2 : jump1 = 4.22)
  (h3 : jump2 = 3.85)
  (h4 : record1 = jump1 - standard_distance)
  (h5 : record2 = jump2 - standard_distance)
  : record2 = -0.15 := 
sorry

end long_jump_record_l1847_184763


namespace thomas_worked_hours_l1847_184772

theorem thomas_worked_hours (Toby Thomas Rebecca : ℕ) 
  (h_total : Thomas + Toby + Rebecca = 157) 
  (h_toby : Toby = 2 * Thomas - 10) 
  (h_rebecca_1 : Rebecca = Toby - 8) 
  (h_rebecca_2 : Rebecca = 56) : Thomas = 37 :=
by
  sorry

end thomas_worked_hours_l1847_184772


namespace sequence_term_a1000_l1847_184743

theorem sequence_term_a1000 :
  ∃ (a : ℕ → ℕ), a 1 = 1007 ∧ a 2 = 1008 ∧
  (∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n) ∧
  a 1000 = 1673 :=
by
  sorry

end sequence_term_a1000_l1847_184743


namespace no_even_threes_in_circle_l1847_184761

theorem no_even_threes_in_circle (arr : ℕ → ℕ) (h1 : ∀ i, 1 ≤ arr i ∧ arr i ≤ 2017)
  (h2 : ∀ i, (arr i + arr ((i + 1) % 2017) + arr ((i + 2) % 2017)) % 2 = 0) : false :=
sorry

end no_even_threes_in_circle_l1847_184761


namespace origin_inside_ellipse_iff_abs_k_range_l1847_184724

theorem origin_inside_ellipse_iff_abs_k_range (k : ℝ) :
  (k^2 * 0^2 + 0^2 - 4 * k * 0 + 2 * k * 0 + k^2 - 1 < 0) ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end origin_inside_ellipse_iff_abs_k_range_l1847_184724


namespace line_product_l1847_184713

theorem line_product (b m : Int) (h_b : b = -2) (h_m : m = 3) : m * b = -6 :=
by
  rw [h_b, h_m]
  norm_num

end line_product_l1847_184713


namespace rational_solutions_quadratic_l1847_184778

theorem rational_solutions_quadratic (k : ℕ) (h_pos : 0 < k) :
  (∃ (x : ℚ), k * x^2 + 24 * x + k = 0) ↔ k = 12 :=
by
  sorry

end rational_solutions_quadratic_l1847_184778


namespace ratio_Bill_to_Bob_l1847_184721

-- Define the shares
def Bill_share : ℕ := 300
def Bob_share : ℕ := 900

-- The theorem statement
theorem ratio_Bill_to_Bob : Bill_share / Bob_share = 1 / 3 := by
  sorry

end ratio_Bill_to_Bob_l1847_184721


namespace range_of_3a_minus_b_l1847_184750

theorem range_of_3a_minus_b (a b : ℝ) (ha : -5 < a) (ha' : a < 2) (hb : 1 < b) (hb' : b < 4) : 
  -19 < 3 * a - b ∧ 3 * a - b < 5 :=
by
  sorry

end range_of_3a_minus_b_l1847_184750


namespace safer_four_engine_airplane_l1847_184758

theorem safer_four_engine_airplane (P : ℝ) (hP : 0 < P ∧ P < 1):
  (∃ p : ℝ, p = 1 - P ∧ (p^4 + 4 * p^3 * (1 - p) + 6 * p^2 * (1 - p)^2 > p^2 + 2 * p * (1 - p) ↔ P > 2 / 3)) :=
sorry

end safer_four_engine_airplane_l1847_184758


namespace abs_inequality_solution_l1847_184790

theorem abs_inequality_solution (x : ℝ) : (|2 * x - 1| - |x - 2| < 0) ↔ (-1 < x ∧ x < 1) := 
sorry

end abs_inequality_solution_l1847_184790


namespace solve_inequality_l1847_184764

theorem solve_inequality (x : ℝ) : abs ((3 - x) / 4) < 1 ↔ 2 < x ∧ x < 7 :=
by {
  sorry
}

end solve_inequality_l1847_184764


namespace polynomial_divisibility_l1847_184742

theorem polynomial_divisibility (n : ℕ) : 120 ∣ (n^5 - 5*n^3 + 4*n) :=
sorry

end polynomial_divisibility_l1847_184742


namespace original_example_intended_l1847_184705

theorem original_example_intended (x : ℝ) : (3 * x - 4 = x / 3 + 4) → x = 3 :=
by
  sorry

end original_example_intended_l1847_184705


namespace BANANA_arrangements_l1847_184796

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l1847_184796


namespace value_of_f_l1847_184760

variable {x t : ℝ}

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 ∨ x = 1 then 0
  else (1 : ℝ) / x

theorem value_of_f (h1 : ∀ x, x ≠ 0 → x ≠ 1 → f (x / (x - 1)) = 1 / x)
                   (h2 : 0 ≤ t ∧ t ≤ Real.pi / 2) :
  f (Real.tan t ^ 2 + 1) = Real.sin (2 * t) ^ 2 / 4 :=
sorry

end value_of_f_l1847_184760


namespace expression_is_integer_l1847_184702

theorem expression_is_integer (n : ℤ) : (∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k) := 
sorry

end expression_is_integer_l1847_184702


namespace equal_savings_l1847_184792

theorem equal_savings (U B UE BE US BS : ℕ) (h1 : U / B = 8 / 7) 
                      (h2 : U = 16000) (h3 : UE / BE = 7 / 6) (h4 : US = BS) :
                      US = 2000 ∧ BS = 2000 :=
by
  sorry

end equal_savings_l1847_184792


namespace Duke_three_pointers_impossible_l1847_184769

theorem Duke_three_pointers_impossible (old_record : ℤ)
  (points_needed_to_tie : ℤ)
  (points_broken_record : ℤ)
  (free_throws : ℕ)
  (regular_baskets : ℕ)
  (three_pointers : ℕ)
  (normal_three_pointers_per_game : ℕ)
  (max_attempts : ℕ)
  (last_minutes : ℕ)
  (points_per_free_throw : ℤ)
  (points_per_regular_basket : ℤ)
  (points_per_three_pointer : ℤ) :
  free_throws = 5 → regular_baskets = 4 → normal_three_pointers_per_game = 2 → max_attempts = 10 → 
  points_per_free_throw = 1 → points_per_regular_basket = 2 → points_per_three_pointer = 3 →
  old_record = 257 → points_needed_to_tie = 17 → points_broken_record = 5 →
  (free_throws + regular_baskets + three_pointers ≤ max_attempts) →
  last_minutes = 6 → 
  ¬(free_throws + regular_baskets + (points_needed_to_tie + points_broken_record - 
  (free_throws * points_per_free_throw + regular_baskets * points_per_regular_basket)) / points_per_three_pointer ≤ max_attempts) := sorry

end Duke_three_pointers_impossible_l1847_184769


namespace outdoor_tables_count_l1847_184720

theorem outdoor_tables_count (num_indoor_tables : ℕ) (chairs_per_indoor_table : ℕ) (chairs_per_outdoor_table : ℕ) (total_chairs : ℕ) : ℕ :=
  let num_outdoor_tables := (total_chairs - (num_indoor_tables * chairs_per_indoor_table)) / chairs_per_outdoor_table
  num_outdoor_tables

example (h₁ : num_indoor_tables = 9)
        (h₂ : chairs_per_indoor_table = 10)
        (h₃ : chairs_per_outdoor_table = 3)
        (h₄ : total_chairs = 123) :
        outdoor_tables_count 9 10 3 123 = 11 :=
by
  -- Only the statement has to be provided; proof steps are not needed
  sorry

end outdoor_tables_count_l1847_184720


namespace average_calls_per_day_l1847_184745

/-- Conditions: Jean's calls per day -/
def calls_mon : ℕ := 35
def calls_tue : ℕ := 46
def calls_wed : ℕ := 27
def calls_thu : ℕ := 61
def calls_fri : ℕ := 31

/-- Assertion: The average number of calls Jean answers per day -/
theorem average_calls_per_day :
  (calls_mon + calls_tue + calls_wed + calls_thu + calls_fri) / 5 = 40 :=
by sorry

end average_calls_per_day_l1847_184745


namespace max_x_y_given_condition_l1847_184711

theorem max_x_y_given_condition (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 1/x + 1/y = 5) : x + y ≤ 4 :=
sorry

end max_x_y_given_condition_l1847_184711


namespace rem_fraction_of_66_l1847_184756

noncomputable def n : ℝ := 22.142857142857142
noncomputable def s : ℝ := n + 5
noncomputable def p : ℝ := s * 7
noncomputable def q : ℝ := p / 5
noncomputable def r : ℝ := q - 5

theorem rem_fraction_of_66 : r = 33 ∧ r / 66 = 1 / 2 := by 
  sorry

end rem_fraction_of_66_l1847_184756


namespace lily_calculation_l1847_184731

theorem lily_calculation (a b c : ℝ) (h1 : a - 2 * b - 3 * c = 2) (h2 : a - 2 * (b - 3 * c) = 14) :
  a - 2 * b = 6 :=
by
  sorry

end lily_calculation_l1847_184731


namespace sequence_a_n_l1847_184768

theorem sequence_a_n {n : ℕ} (S : ℕ → ℚ) (a : ℕ → ℚ)
  (hS : ∀ n, S n = (2/3 : ℚ) * n^2 - (1/3 : ℚ) * n)
  (ha : ∀ n, a n = if n = 1 then S n else S n - S (n - 1)) :
  ∀ n, a n = (4/3 : ℚ) * n - 1 := 
by
  sorry

end sequence_a_n_l1847_184768


namespace probability_A_inter_B_l1847_184708

def set_A (x : ℝ) : Prop := -1 < x ∧ x < 5
def set_B (x : ℝ) : Prop := (x-2)/(3-x) > 0

def A_inter_B (x : ℝ) : Prop := set_A x ∧ set_B x

theorem probability_A_inter_B :
  let length_A := 5 - (-1)
  let length_A_inter_B := 3 - 2 
  length_A > 0 ∧ length_A_inter_B > 0 →
  length_A_inter_B / length_A = 1 / 6 :=
by
  intro h
  sorry

end probability_A_inter_B_l1847_184708


namespace largest_circle_diameter_l1847_184794

theorem largest_circle_diameter
  (A : ℝ) (hA : A = 180)
  (w l : ℝ) (hw : l = 3 * w)
  (hA2 : w * l = A) :
  ∃ d : ℝ, d = 16 * Real.sqrt 15 / Real.pi :=
by
  sorry

end largest_circle_diameter_l1847_184794


namespace number_is_two_l1847_184751

theorem number_is_two 
  (N : ℝ)
  (h1 : N = 4 * 1 / 2)
  (h2 : (1 / 2) * N = 1) :
  N = 2 :=
sorry

end number_is_two_l1847_184751


namespace cos_540_eq_neg1_l1847_184740

theorem cos_540_eq_neg1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end cos_540_eq_neg1_l1847_184740


namespace right_triangle_area_l1847_184777

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end right_triangle_area_l1847_184777


namespace solve_for_xy_l1847_184735

theorem solve_for_xy (x y : ℝ) 
  (h1 : 0.05 * x + 0.07 * (30 + x) = 14.9)
  (h2 : 0.03 * y - 5.6 = 0.07 * x) : 
  x = 106.67 ∧ y = 435.567 := 
  by 
  sorry

end solve_for_xy_l1847_184735


namespace sum_of_cube_edges_l1847_184752

/-- A cube has 12 edges. Each edge of a cube is of equal length. Given the length of one
edge as 15 cm, the sum of the lengths of all the edges of the cube is 180 cm. -/
theorem sum_of_cube_edges (edge_length : ℝ) (num_edges : ℕ) (h1 : edge_length = 15) (h2 : num_edges = 12) :
  num_edges * edge_length = 180 :=
by
  sorry

end sum_of_cube_edges_l1847_184752


namespace quadratic_function_m_value_l1847_184737

theorem quadratic_function_m_value :
  ∃ m : ℝ, (m - 3 ≠ 0) ∧ (m^2 - 7 = 2) ∧ m = -3 :=
by
  sorry

end quadratic_function_m_value_l1847_184737


namespace evaluate_expression_l1847_184704

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-2) = 85 :=
by
  sorry

end evaluate_expression_l1847_184704


namespace many_people_sharing_car_l1847_184709

theorem many_people_sharing_car (x y : ℤ) 
  (h1 : 3 * (y - 2) = x) 
  (h2 : 2 * y + 9 = x) : 
  3 * (y - 2) = 2 * y + 9 := 
by
  -- by assumption h1 and h2, we already have the setup, refute/validate consistency
  sorry

end many_people_sharing_car_l1847_184709


namespace coordinates_of_A_in_second_quadrant_l1847_184765

noncomputable def coordinates_A (m : ℤ) : ℤ × ℤ :=
  (7 - 2 * m, 5 - m)

theorem coordinates_of_A_in_second_quadrant (m : ℤ) (h1 : 7 - 2 * m < 0) (h2 : 5 - m > 0) :
  coordinates_A m = (-1, 1) := 
sorry

end coordinates_of_A_in_second_quadrant_l1847_184765


namespace product_of_consecutive_multiples_of_4_divisible_by_768_l1847_184738

theorem product_of_consecutive_multiples_of_4_divisible_by_768 (n : ℤ) :
  (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) % 768 = 0 :=
by
  sorry

end product_of_consecutive_multiples_of_4_divisible_by_768_l1847_184738


namespace proof_x_squared_plus_y_squared_l1847_184785

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l1847_184785


namespace trains_meet_distance_from_delhi_l1847_184716

-- Define the speeds of the trains as constants
def speed_bombay_express : ℕ := 60  -- kmph
def speed_rajdhani_express : ℕ := 80  -- kmph

-- Define the time difference in hours between the departures of the two trains
def time_difference : ℕ := 2  -- hours

-- Define the distance the Bombay Express travels before the Rajdhani Express starts
def distance_head_start : ℕ := speed_bombay_express * time_difference

-- Define the relative speed between the two trains
def relative_speed : ℕ := speed_rajdhani_express - speed_bombay_express

-- Define the time taken for the Rajdhani Express to catch up with the Bombay Express
def time_to_meet : ℕ := distance_head_start / relative_speed

-- The final meeting distance from Delhi for the Rajdhani Express
def meeting_distance : ℕ := speed_rajdhani_express * time_to_meet

-- Theorem stating the solution to the problem
theorem trains_meet_distance_from_delhi : meeting_distance = 480 :=
by sorry  -- proof is omitted

end trains_meet_distance_from_delhi_l1847_184716


namespace quotient_of_division_l1847_184789

theorem quotient_of_division (dividend divisor remainder : ℕ) (h_dividend : dividend = 127) (h_divisor : divisor = 14) (h_remainder : remainder = 1) :
  (dividend - remainder) / divisor = 9 :=
by 
  -- Proof follows
  sorry

end quotient_of_division_l1847_184789


namespace june_eggs_count_l1847_184766

theorem june_eggs_count :
  (2 * 5) + 3 + 4 = 17 := 
by 
  sorry

end june_eggs_count_l1847_184766


namespace find_number_l1847_184717

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 99) : x = 4400 :=
sorry

end find_number_l1847_184717


namespace opposite_of_five_l1847_184714

theorem opposite_of_five : -5 = -5 :=
by
sorry

end opposite_of_five_l1847_184714


namespace number_of_cages_l1847_184757

-- Definitions based on the conditions
def parrots_per_cage := 2
def parakeets_per_cage := 6
def total_birds := 72

-- Goal: Prove the number of cages
theorem number_of_cages : 
  (parrots_per_cage + parakeets_per_cage) * x = total_birds → x = 9 :=
by
  sorry

end number_of_cages_l1847_184757


namespace price_of_cashew_nuts_l1847_184776

theorem price_of_cashew_nuts 
  (C : ℝ)  -- price per kilo of cashew nuts
  (P_p : ℝ := 130)  -- price per kilo of peanuts
  (cashew_kilos : ℝ := 3)  -- kilos of cashew nuts bought
  (peanut_kilos : ℝ := 2)  -- kilos of peanuts bought
  (total_kilos : ℝ := 5)  -- total kilos of nuts bought
  (total_price_per_kilo : ℝ := 178)  -- total price per kilo of all nuts
  (h_total_cost : cashew_kilos * C + peanut_kilos * P_p = total_kilos * total_price_per_kilo) :
  C = 210 :=
sorry

end price_of_cashew_nuts_l1847_184776


namespace same_terminal_angle_l1847_184722

theorem same_terminal_angle (k : ℤ) :
  ∃ α : ℝ, α = k * 360 + 40 :=
by
  sorry

end same_terminal_angle_l1847_184722


namespace graphene_scientific_notation_l1847_184780

def scientific_notation (n : ℝ) (a : ℝ) (exp : ℤ) : Prop :=
  n = a * 10 ^ exp ∧ 1 ≤ abs a ∧ abs a < 10

theorem graphene_scientific_notation :
  scientific_notation 0.00000000034 3.4 (-10) :=
by {
  sorry
}

end graphene_scientific_notation_l1847_184780


namespace integer_points_on_line_l1847_184781

/-- Given a line that passes through points C(3, 3) and D(150, 250),
prove that the number of other points with integer coordinates
that lie strictly between C and D is 48. -/
theorem integer_points_on_line {C D : ℝ × ℝ} (hC : C = (3, 3)) (hD : D = (150, 250)) :
  ∃ (n : ℕ), n = 48 ∧ 
  ∀ p : ℝ × ℝ, C.1 < p.1 ∧ p.1 < D.1 ∧ 
  C.2 < p.2 ∧ p.2 < D.2 → 
  (∃ (k : ℤ), p.1 = ↑k ∧ p.2 = (5/3) * p.1 - 2) :=
sorry

end integer_points_on_line_l1847_184781


namespace expected_value_twelve_sided_die_l1847_184715

/-- A twelve-sided die has its faces numbered from 1 to 12.
    Prove that the expected value of a roll of this die is 6.5. -/
theorem expected_value_twelve_sided_die : 
  (1 / 12 : ℚ) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_twelve_sided_die_l1847_184715


namespace complement_union_eq_l1847_184791

variable (U : Set ℝ := Set.univ)
variable (A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)})
variable (B : Set ℝ := {x | -2 ≤ x ∧ x < 4})

theorem complement_union_eq : (U \ A) ∪ B = {x | x ≥ -2} := by
  sorry

end complement_union_eq_l1847_184791


namespace fraction_percent_l1847_184727

theorem fraction_percent (x : ℝ) (h : x > 0) : ((x / 10 + x / 25) / x) * 100 = 14 :=
by
  sorry

end fraction_percent_l1847_184727


namespace intersection_equivalence_l1847_184726

open Set

noncomputable def U : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def M : Set ℤ := {-1, 0, 1}
noncomputable def N : Set ℤ := {x | x * x - x - 2 = 0}
noncomputable def complement_M_in_U : Set ℤ := U \ M

theorem intersection_equivalence : (complement_M_in_U ∩ N) = {2} := 
by
  sorry

end intersection_equivalence_l1847_184726


namespace gingerbread_to_bagels_l1847_184779

theorem gingerbread_to_bagels (gingerbread drying_rings bagels : ℕ) 
  (h1 : gingerbread = 1 → drying_rings = 6) 
  (h2 : drying_rings = 9 → bagels = 4) 
  (h3 : gingerbread = 3) : bagels = 8 :=
by
  sorry

end gingerbread_to_bagels_l1847_184779
