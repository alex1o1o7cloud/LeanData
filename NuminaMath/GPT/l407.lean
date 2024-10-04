import Mathlib

namespace mean_transformed_29_l407_407150

-- Define the problem conditions
variables {n : ℕ} (x : ℕ → ℝ)  -- x is a function from natural numbers to reals

-- Condition: the mean of the original sample data is 10
def mean_original (x : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ i in finset.range n, x i) / n = 10

-- The transformed data is given by 3x_i - 1
def transformed_data (x : ℕ → ℝ) (i : ℕ) : ℝ := 3 * x i - 1

-- Prove that the mean of the transformed sample data is 29
theorem mean_transformed_29 {n : ℕ} (x : ℕ → ℝ) (h : mean_original x n) :
  (∑ i in finset.range n, transformed_data x i) / n = 29 :=
sorry

end mean_transformed_29_l407_407150


namespace tournament_points_l407_407495

/-
In a football tournament, teams A, B, C, D, E participated.
Each team played exactly one match against each other team.
For each game: 
- The winner received 3 points,
- A draw gave each team 1 point,
- A loss gave 0 points.

Given conditions:
1. All teams scored a different number of points.
2. Team A scored the most points, even though it lost to team B.
3. Teams B and C did not lose any games.
4. Team C scored fewer points than team D.

The goal is to prove that:
Team A scored 7 points,
Team B scored 6 points,
Team C scored 4 points,
Team D scored 5 points,
Team E scored 2 points.
-/

def points (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a = 7 ∧ b = 6 ∧ c = 4 ∧ d = 5 ∧ e = 2 ∧
  a > b ∧ a > c ∧ a > d ∧ a > e ∧
  c < d ∧
  ∀ (T : ℕ), T ∈ {b, c} -> T ≠ 0

-- Lean 4 statement
theorem tournament_points :
  ∃ (a b c d e : ℕ), points a b c d e :=
by {
  use [7, 6, 4, 5, 2],
  split, repeat {split}, all_goals {exact dec_trivial},
  sorry
}

end tournament_points_l407_407495


namespace mayoral_election_1002nd_round_l407_407300

theorem mayoral_election_1002nd_round (candidates : ℕ) (half_plus_one : candidates // 2) (rounds : ℕ) (k : ℕ) :
  candidates = 2002 → rounds = 1002 → k = 2001 → 
  (∃ (ostap_wins : bool), ostap_wins = true) :=
by
  intros
  sorry

end mayoral_election_1002nd_round_l407_407300


namespace tan_alpha_calc_l407_407966

theorem tan_alpha_calc (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
by sorry

end tan_alpha_calc_l407_407966


namespace number_of_factors_of_60_l407_407207

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l407_407207


namespace bar_vs_line_trend_l407_407551

-- Definitions according to conditions
def bar_chart_shows_quantity (Q : Type) : Prop :=
  ∀ (chart : Q), chart = "bar" → shows_quantity chart

def line_chart_shows_quantity (Q : Type) : Prop :=
  ∀ (chart : Q), chart = "line" → shows_quantity chart

def line_chart_shows_trend (Q : Type) : Prop :=
  ∀ (chart : Q), chart = "line" → shows_trend chart

-- The statement to prove
theorem bar_vs_line_trend (Q : Type) (shows_quantity : Q → Prop) (shows_trend : Q → Prop):
  bar_chart_shows_quantity Q →
  line_chart_shows_quantity Q →
  line_chart_shows_trend Q →
  (∀ (chart : Q), chart = "bar" → ¬ shows_trend chart) :=
sorry

end bar_vs_line_trend_l407_407551


namespace calculation_l407_407558

-- Definition of the absolute value function
def abs (x : ℤ) : ℤ := if x < 0 then -x else x

-- Definition of the square root function as an integer (there's no native int sqrt, but using nat.sqrt for demonstration)
noncomputable def sqrt (x : ℤ) : ℤ := Int.ofNat (Nat.sqrt (Int.toNat x))

-- Definition of the square function
def square (x : ℤ) : ℤ := x * x

-- The theorem we need to prove
theorem calculation : abs (-7) + sqrt 16 - square (-3) = 2 := by
  sorry

end calculation_l407_407558


namespace solve_for_b_l407_407448

theorem solve_for_b (a b : ℤ) (h1 : 3 * a + 2 = 5) (h2 : b - 4 * a = 2) : b = 6 :=
by
  -- proof goes here
  sorry

end solve_for_b_l407_407448


namespace geli_total_pushups_first_week_l407_407672

/-- Definitions of the number of push-ups Geli does each day given the problem conditions. -/
def pushups_on_day : ℕ → ℕ
| 1 => 10
| n+1 => pushups_on_day n + 5

/-- Definition of the total number of push-ups Geli does in her first week (3 days). -/
def total_pushups_first_week := pushups_on_day 1 + pushups_on_day 2 + pushups_on_day 3

/-- The main theorem stating that the total push-ups Geli does in her first week is 45. -/
theorem geli_total_pushups_first_week : total_pushups_first_week = 45 :=
by sorry

end geli_total_pushups_first_week_l407_407672


namespace missing_score_find_missing_score_l407_407311

theorem missing_score
  (score1 score2 score3 score4 mean total : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89)
  (hTotal : total = 445) :
  score1 + score2 + score3 + score4 + x = total :=
by
  sorry

theorem find_missing_score
  (score1 score2 score3 score4 mean : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89) :
  (score1 + score2 + score3 + score4 + x) / 5 = mean
  → x = 90 :=
by
  sorry

end missing_score_find_missing_score_l407_407311


namespace num_lattice_points_on_sphere_5_l407_407771

theorem num_lattice_points_on_sphere_5 :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.to_finset.card = 54 := 
by
  sorry

end num_lattice_points_on_sphere_5_l407_407771


namespace presidency_meeting_arrangements_l407_407975

theorem presidency_meeting_arrangements : 
  let members_per_school := 5
  let schools := 4
  let total_members := members_per_school * schools
  let choose_host_school (n : ℕ) := n
  let choose_representatives (total chosen: ℕ) := choose total chosen
  (choose_host_school schools) *
  (choose_representatives members_per_school 2) *
  (choose_representatives members_per_school 1)^3 = 5000 := 
by
  have members_per_school : ℕ := 5
  have schools : ℕ := 4
  have total_members := members_per_school * schools
  have choose_host_school := schools
  have choose_representatives (total chosen: ℕ) := nat.choose total chosen
  have total_ways := choose_host_school *
    (choose_representatives members_per_school 2) *
    (choose_representatives members_per_school 1)^3
  exact total_ways = 5000
  sorry

end presidency_meeting_arrangements_l407_407975


namespace monotonically_increasing_on_interval_range_of_t_l407_407154

open Real

/-- Part 1: Prove that f(x) = x / (x^2 + 1) is monotonically increasing on (-1, 1). -/
theorem monotonically_increasing_on_interval :
  ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → (f x1 < f x2) :=
by
  let f := fun x => x / (x^2 + 1)
  sorry

/-- Part 2: Given that f(t - 1) + f(2t) < 0, find the range of t. -/
theorem range_of_t (t : ℝ) :
  (f (t - 1) + f (2 * t) < 0) → 0 < t ∧ t < 1 / 3 :=
by
  let f := fun x => x / (x^2 + 1)
  sorry

end monotonically_increasing_on_interval_range_of_t_l407_407154


namespace evaluate_expression_l407_407685

variable (a : ℝ)

def a_definition : Prop := a = Real.sqrt 11 - 1

theorem evaluate_expression (h : a_definition a) : a^2 + 2*a + 1 = 11 := by
  sorry

end evaluate_expression_l407_407685


namespace max_not_sum_S_l407_407828

def S : Set ℕ := {n | ∃ k : ℕ, n = 10^k + 1000}

theorem max_not_sum_S : ∀ x : ℕ, (∀ y ∈ S, ∃ m : ℕ, x ≠ m * y) ↔ x = 34999 := by
  sorry

end max_not_sum_S_l407_407828


namespace lattice_points_with_distance_five_from_origin_l407_407807

theorem lattice_points_with_distance_five_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 6 :=
by
  sorry

end lattice_points_with_distance_five_from_origin_l407_407807


namespace arithmetic_mean_not_prime_l407_407846

theorem arithmetic_mean_not_prime (n : ℕ) (h : n ≥ 2) : 
  ¬ Prime ((list.range (n + 1)).filter (λ k, Prime k)).sum / n :=
by
  sorry

end arithmetic_mean_not_prime_l407_407846


namespace sum_of_floors_of_square_roots_l407_407029

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l407_407029


namespace sum_of_floors_of_square_roots_l407_407028

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l407_407028


namespace saline_solution_mixture_l407_407181

theorem saline_solution_mixture 
  (x : ℝ) 
  (h₁ : 20 + 0.1 * x = 0.25 * (50 + x)) 
  : x = 50 := 
by 
  sorry

end saline_solution_mixture_l407_407181


namespace find_x_values_for_inverse_l407_407070

def f (x : ℝ) : ℝ := x^2 - 3 * x - 4

theorem find_x_values_for_inverse :
  ∃ (x : ℝ), (f x = 2 + 2 * Real.sqrt 2 ∨ f x = 2 - 2 * Real.sqrt 2) ∧ f x = x :=
sorry

end find_x_values_for_inverse_l407_407070


namespace binom_30_3_eq_4060_l407_407569

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l407_407569


namespace int_pair_satisfies_equation_l407_407644

theorem int_pair_satisfies_equation :
  ∃ m n : ℤ, 4^m - 4^n = 255 ∧ (m, n) = (4, 0) :=
by
  use [4, 0]
  split
  · sorry
  · rfl

end int_pair_satisfies_equation_l407_407644


namespace num_lattice_points_distance_five_l407_407816

open_locale classical 

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points_distance_five : 
  {n : ℕ | ∃ (x y z : ℤ), is_lattice_point x y z} = 78 :=
by sorry

end num_lattice_points_distance_five_l407_407816


namespace smallest_four_digit_congruent_one_mod_17_l407_407938

theorem smallest_four_digit_congruent_one_mod_17 :
  ∃ (n : ℕ), 1000 ≤ n ∧ n % 17 = 1 ∧ n = 1003 :=
by
sorry

end smallest_four_digit_congruent_one_mod_17_l407_407938


namespace max_distinct_prime_factors_of_c_l407_407839

-- Definitions of the conditions
variables (c d : ℕ)

def gcd_has_11_distinct_primes (c d : ℕ) : Prop :=
  nat.gcd c d |> prime_factors |> list.nodup.count = 11

def lcm_has_35_distinct_primes (c d : ℕ) : Prop :=
  nat.lcm c d |> prime_factors |> list.nodup.count = 35

-- Propositional statement of the problem
theorem max_distinct_prime_factors_of_c
  (hc : gcd_has_11_distinct_primes c d)
  (hl : lcm_has_35_distinct_primes c d)
  (hcd : nat.factors c < nat.factors d) :
  nat.factors c ≤ 23 :=
sorry

end max_distinct_prime_factors_of_c_l407_407839


namespace minimum_value_solution_l407_407831

def minimum_value_problem (alpha beta : ℝ) : Prop :=
  (3 * Real.cos alpha + 4 * Real.sin beta - 10)^2 + (3 * Real.sin alpha + 4 * Real.cos beta - 20)^2

theorem minimum_value_solution (alpha beta : ℝ) : 
  ∃ alpha beta : ℝ, minimum_value_problem alpha beta ≥ 236 := 
sorry

end minimum_value_solution_l407_407831


namespace count_equilateral_triangles_l407_407894

theorem count_equilateral_triangles :
  ∀ k : ℕ, k ∈ {0, 1, 2, 3, 4, 5} →
  let lines := [λ (x : ℝ), (k : ℝ), λ (x : ℝ), √3 * x + k, λ (x : ℝ), -√3 * x + k]
  let triangles_per_level := 4
  let levels := 6
  triangles_per_level * levels = 24 :=
by
  sorry

end count_equilateral_triangles_l407_407894


namespace number_of_factors_of_60_l407_407203

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l407_407203


namespace distinct_real_roots_l407_407666

theorem distinct_real_roots (a : ℝ) :
  let A := a - 3 in
  let B := -4 in
  let C := -1 in
  (a ≠ 3 ∧ 4 * a + 4 > 0) ↔ (a > -1 ∧ a ≠ 3) :=
by simp [A, B, C]; rink sorry

end distinct_real_roots_l407_407666


namespace expression_equals_66069_l407_407942

-- Definitions based on the conditions
def numerator : Nat := 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10
def denominator : Nat := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
def expression : Rat := numerator / denominator

-- The main theorem to be proven
theorem expression_equals_66069 : expression = 66069 := by
  sorry

end expression_equals_66069_l407_407942


namespace number_of_factors_of_60_l407_407202

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l407_407202


namespace square_root_of_9_eq_pm_3_l407_407423

theorem square_root_of_9_eq_pm_3 (x : ℝ) : x^2 = 9 → x = 3 ∨ x = -3 :=
sorry

end square_root_of_9_eq_pm_3_l407_407423


namespace Caroline_lost_4_pairs_of_socks_l407_407563

theorem Caroline_lost_4_pairs_of_socks 
  (initial_pairs : ℕ) (pairs_donated_fraction : ℚ)
  (new_pairs_purchased : ℕ) (new_pairs_gifted : ℕ)
  (final_pairs : ℕ) (L : ℕ) :
  initial_pairs = 40 →
  pairs_donated_fraction = 2/3 →
  new_pairs_purchased = 10 →
  new_pairs_gifted = 3 →
  final_pairs = 25 →
  (initial_pairs - L) * (1 - pairs_donated_fraction) + new_pairs_purchased + new_pairs_gifted = final_pairs →
  L = 4 :=
by {
  sorry
}

end Caroline_lost_4_pairs_of_socks_l407_407563


namespace stationery_store_equation_l407_407535

theorem stationery_store_equation
  (x : ℝ)
  (price_pencil : ℝ)
  (price_pen : ℝ)
  (discount_pencil : ℝ)
  (discount_pen : ℝ)
  (total_pens : ℝ)
  (total_revenue : ℝ) 
  (h_price_pencil : price_pencil = 1.2)
  (h_price_pen : price_pen = 2)
  (h_discount_pencil : discount_pencil = 0.8)
  (h_discount_pen : discount_pen = 0.9)
  (h_total_pens : total_pens = 60)
  (h_total_revenue : total_revenue = 87) :
  1.2 * 0.8 * x + 2 * 0.9 * (60 - x) = 87 :=
by {
  rw [h_price_pencil, h_price_pen, h_discount_pencil, h_discount_pen, h_total_pens, h_total_revenue],
  sorry  -- Proof is omitted
}

end stationery_store_equation_l407_407535


namespace inverse_function_correct_l407_407895

def original_function (x : ℝ) : ℝ := 3^(x + 1)

def inverse_function (x : ℝ) : ℝ := -1 + Real.logb 3 x

theorem inverse_function_correct (x : ℝ) (h1 : -1 ≤ x) (h2 : x < 0) :
  ∃ (y : ℝ), inverse_function y = x ∧ 1 ≤ y ∧ y < 3 :=
by
  sorry

end inverse_function_correct_l407_407895


namespace first_term_of_geometric_sequence_l407_407419

theorem first_term_of_geometric_sequence (a r : ℚ) 
  (h1 : a * r = 18) 
  (h2 : a * r^2 = 24) : 
  a = 27 / 2 := 
sorry

end first_term_of_geometric_sequence_l407_407419


namespace sum_f_to_2011_l407_407109

def f (n : ℕ) : ℕ := (n * (n + 1) / 2) % 10

theorem sum_f_to_2011 : (∑ i in Finset.range 2012, f i) = 7046 := by
  sorry

end sum_f_to_2011_l407_407109


namespace math_problem_l407_407683

variable (a : ℝ)
noncomputable def problem := a = Real.sqrt 11 - 1
noncomputable def target := a^2 + 2 * a + 1 = 11

theorem math_problem (h : problem a) : target a :=
  sorry

end math_problem_l407_407683


namespace num_lattice_points_l407_407776

def is_lattice_point (p : ℤ × ℤ × ℤ) : Prop := 
  let (x, y, z) := p 
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points : 
  {p : ℤ × ℤ × ℤ | is_lattice_point p}.to_finset.card = 18 :=
sorry

end num_lattice_points_l407_407776


namespace tangent_line_eq_for_a_neg_1_monotonic_intervals_for_a_gt_0_l407_407721

def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * a * x^2 - (2 * a + 1) * x + 2 * Real.log x

theorem tangent_line_eq_for_a_neg_1 :
  let a := -1 in
  let f' (x : ℝ) := -x + 1 + (2 / x) in
  ∀ x y, y = f a 1 → y = 1 / 2 → (4 * x - 2 * y - 3 = 0) :=
by sorry

theorem monotonic_intervals_for_a_gt_0 :
  ∀ a x, 
    0 < a →
    let f' (x : ℝ) := a * x - (2 * a + 1) + (2 / x) in
    (if (a > 1/2) then (0 < x ∧ x < 1/a) ∨ (2 < x) else (if a = 1/2 then 0 < x else (0 < x ∧ x < 2) ∨ (1/a < x))) →
    (∀ x, 0 < x → a * x - (2 * a + 1) + (2 / x) ≥ 0) :=
by sorry

end tangent_line_eq_for_a_neg_1_monotonic_intervals_for_a_gt_0_l407_407721


namespace inequality_for_average_daily_work_l407_407515

-- Given
def total_earthwork : ℕ := 300
def completed_earthwork_first_day : ℕ := 60
def scheduled_days : ℕ := 6
def days_ahead : ℕ := 2

-- To Prove
theorem inequality_for_average_daily_work (x : ℕ) :
  scheduled_days - days_ahead - 1 > 0 →
  (total_earthwork - completed_earthwork_first_day) ≤ x * (scheduled_days - days_ahead - 1) :=
by
  sorry

end inequality_for_average_daily_work_l407_407515


namespace participants_perfect_square_l407_407288

theorem participants_perfect_square 
  (F L : ℕ) 
  (h1 : ∀ p, p = F ∨ p = L)
  (h2 : ∀ k, ∃ m n, k = m + n ∧ n = F ∨ n = L)
  (h_half_points_from_boys : ∀ i, i ∈ (Finset.range (F + L)) → (∃ s : Set ℕ, s ⊆ (Finset.range F) ∧ (2 * (Finset.card s) = ∑ j in s, 1 + j)))
  : ∃ k : ℕ, F + L = k * k :=
sorry

end participants_perfect_square_l407_407288


namespace gcd_of_terms_l407_407945

theorem gcd_of_terms (m n : ℕ) : gcd (4 * m^3 * n) (9 * m * n^3) = m * n := 
sorry

end gcd_of_terms_l407_407945


namespace ceil_evaluation_l407_407081

theorem ceil_evaluation : 
  (Int.ceil (4 * (8 - 1 / 3 : ℚ))) = 31 :=
by
  sorry

end ceil_evaluation_l407_407081


namespace hardcover_volumes_l407_407077

theorem hardcover_volumes (h p : ℕ) (h1 : h + p = 10) (h2 : 25 * h + 15 * p = 220) : h = 7 :=
by sorry

end hardcover_volumes_l407_407077


namespace only_common_integer_is_one_l407_407420

noncomputable def a : ℕ → ℤ
| 0 => 1
| 1 => 1
| (n + 2) => a (n + 1) + 2 * a n

noncomputable def b : ℕ → ℤ
| 0 => 1
| 1 => 7
| (n + 2) => 2 * b (n + 1) + 3 * b n

theorem only_common_integer_is_one : ∀ n m, a n = b m → a n = 1 :=
begin
  sorry
end

end only_common_integer_is_one_l407_407420


namespace general_term_of_series_l407_407730

def gen_term (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a n = if n = 1 then 2 else 6 * n - 5

def series_sum (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = 3 * n ^ 2 - 2 * n + 1

theorem general_term_of_series (a S : ℕ → ℕ) (h : series_sum S) :
  gen_term a ↔ (∀ n : ℕ, a n = if n = 1 then 2 else S n - S (n - 1)) :=
by sorry

end general_term_of_series_l407_407730


namespace num_factors_of_60_l407_407238

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l407_407238


namespace laps_run_l407_407851

theorem laps_run (x : ℕ) (total_distance required_distance lap_length extra_laps : ℕ) (h1 : total_distance = 2400) (h2 : lap_length = 150) (h3 : extra_laps = 4) (h4 : total_distance = lap_length * (x + extra_laps)) : x = 12 :=
by {
  sorry
}

end laps_run_l407_407851


namespace floor_sum_sqrt_1_to_25_l407_407061

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l407_407061


namespace least_prime_of_even_sum_prime_set_l407_407490

-- Definition of conditions
def is_prime_set (q : Set ℕ) : Prop :=
  q.Countable ∧ q.finite ∧ ∀ p ∈ q, Nat.Prime p ∧ q.card = 8

def even_sum (q : Set ℕ) : Prop :=
  (q.Sum id) % 2 = 0

-- The main theorem statement
theorem least_prime_of_even_sum_prime_set (q : Set ℕ) (h1 : is_prime_set q) (h2 : even_sum q) : 
  ∃ p ∈ q, p = 2 :=
sorry

end least_prime_of_even_sum_prime_set_l407_407490


namespace num_factors_of_60_l407_407191

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l407_407191


namespace num_factors_of_60_l407_407193

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l407_407193


namespace proof_problem_l407_407349

-- Define the universal set U
def U := Set ℝ

-- Define the set A
def A : Set ℝ := {x | x^2 - 3 * x ≥ 0}

-- Define the set B
def B : Set ℕ := {x | x ≤ 3}

-- Define complement of A in U
def complement_A : Set ℝ := {x | 0 < x ∧ x < 3}

-- Define the intersection of complement_A and B
def intersection_complement_A_B : Set ℝ := complement_A ∩ (↑B : Set ℝ)

theorem proof_problem : intersection_complement_A_B = {1, 2} :=
sorry

end proof_problem_l407_407349


namespace infinitely_many_lovely_numbers_no_perfect_square_lovely_number_gt1_l407_407449

def is_lovely (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∃ d : Fin k → ℕ, (∀ i, d i > 0) ∧ n = ∏ i in Finset.finRange k, d i ∧ ∀ i, (d i)^2 ∣ n + d i

theorem infinitely_many_lovely_numbers :
  ∃ f : ℕ → ℕ, ∀ n, is_lovely(f n) :=
by sorry

theorem no_perfect_square_lovely_number_gt1 :
  ∀ m, m > 1 → ¬ is_lovely(m^2) :=
by sorry

end infinitely_many_lovely_numbers_no_perfect_square_lovely_number_gt1_l407_407449


namespace binom_30_3_eq_4060_l407_407602

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l407_407602


namespace binomial_expansion_sixth_term_constant_l407_407144

open Nat

theorem binomial_expansion_sixth_term_constant (n : ℕ) :
  (∃ (f : ℕ → ℝ), ∀ k, f k = (binom n k) * ((-1/2)^k) * (x^((n-2*k)/3)) ∧ (x^((n-2*5)/3)) = 1) -> n = 10 :=
by
  sorry

end binomial_expansion_sixth_term_constant_l407_407144


namespace max_withdrawal_l407_407372

def initial_balance : ℕ := 500
def withdraw_amount : ℕ := 300
def add_amount : ℕ := 198
def remaining_balance (x : ℕ) : Prop := 
  x % 6 = 0 ∧ x ≤ initial_balance

theorem max_withdrawal : ∃(max_withdrawal_amount : ℕ), 
  max_withdrawal_amount = initial_balance - 498 :=
sorry

end max_withdrawal_l407_407372


namespace evaluate_expression_l407_407637

-- Define the first arithmetic sequence
def seq1 := list.range' 2204 95 |> list.map (λ x => 2204 + 2 * x)

-- Define the second arithmetic sequence
def seq2 := list.range' 306 48 |> list.map (λ x => 306 + 2 * x)

-- State the problem
theorem evaluate_expression :
  (seq1.sum) - (seq2.sum) = 91056 :=
by sorry

end evaluate_expression_l407_407637


namespace trigonometric_identity_l407_407871

theorem trigonometric_identity (a : ℝ) : cos (2 * a) + 2 * sin a ^ 2 = 1 :=
by
  -- By the trigonometric identity cos 2a = 1 - 2sin^2a.
  -- And simplifying the expression: cos 2a + 2sin^2a = (1 - 2sin^2a) + 2sin^2a = 1.
  sorry

end trigonometric_identity_l407_407871


namespace minimum_value_l407_407941

variable {x : ℝ}

theorem minimum_value (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
by
  sorry

end minimum_value_l407_407941


namespace mean_score_for_exam_l407_407662

variable (M SD : ℝ)

-- Define the conditions
def condition1 : Prop := 58 = M - 2 * SD
def condition2 : Prop := 98 = M + 3 * SD

-- The problem statement
theorem mean_score_for_exam (h1 : condition1 M SD) (h2 : condition2 M SD) : M = 74 :=
sorry

end mean_score_for_exam_l407_407662


namespace floor_sum_sqrt_25_l407_407007

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l407_407007


namespace sum_floor_sqrt_l407_407049

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l407_407049


namespace remainder_of_70_div_17_l407_407462

theorem remainder_of_70_div_17 : 70 % 17 = 2 :=
by
  sorry

end remainder_of_70_div_17_l407_407462


namespace factors_of_60_l407_407234

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l407_407234


namespace two_A_plus_B_l407_407328

theorem two_A_plus_B (A B : ℕ) (h1 : A = Nat.gcd (Nat.gcd 12 18) 30) (h2 : B = Nat.lcm (Nat.lcm 12 18) 30) : 2 * A + B = 192 :=
by
  sorry

end two_A_plus_B_l407_407328


namespace arithmetic_progression_impossible_geometric_progression_possible_l407_407560

theorem arithmetic_progression_impossible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  2 * b ≠ a + c :=
by {
    sorry
}

theorem geometric_progression_possible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  ∃ r m : ℤ, (b / a)^r = (c / a)^m :=
by {
    sorry
}

end arithmetic_progression_impossible_geometric_progression_possible_l407_407560


namespace binom_30_3_eq_4060_l407_407570

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l407_407570


namespace floor_sqrt_sum_l407_407048

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l407_407048


namespace least_value_of_b_l407_407276

variable {x y b : ℝ}

noncomputable def condition_inequality (x y b : ℝ) : Prop :=
  (x^2 + y^2)^2 ≤ b * (x^4 + y^4)

theorem least_value_of_b (h : ∀ x y : ℝ, condition_inequality x y b) : b ≥ 2 := 
sorry

end least_value_of_b_l407_407276


namespace triangle_max_distance_product_l407_407130

open Real

noncomputable def max_product_of_distances
  (a b c : ℝ) (P : {p : ℝ × ℝ // True}) : ℝ :=
  let h_a := 1 -- placeholder for actual distance calculation
  let h_b := 1 -- placeholder for actual distance calculation
  let h_c := 1 -- placeholder for actual distance calculation
  h_a * h_b * h_c

theorem triangle_max_distance_product
  (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5)
  (P : {p : ℝ × ℝ // True}) :
  max_product_of_distances a b c P = (16/15 : ℝ) :=
sorry

end triangle_max_distance_product_l407_407130


namespace distinct_digits_D_possibilities_l407_407761

noncomputable def distinct_digits_values : Nat :=
  9

-- Mathematical equivalent theorem statement in Lean 4
theorem distinct_digits_D_possibilities :
  ∀ (A B C D E : ℕ), distinct {A, B, C, D, E} → 
  (ABECB + BCEDA = DEDDE) → (trailing_digit DEDDE = D) → 
  (leading_digit DEDDE = D) → 
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10) →
  D ≤ distinct_digits_values :=
by sorry

end distinct_digits_D_possibilities_l407_407761


namespace factors_of_60_l407_407231

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l407_407231


namespace sum_floor_sqrt_l407_407054

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l407_407054


namespace sum_of_floors_of_square_roots_l407_407027

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l407_407027


namespace largest_angle_at_C_l407_407826

theorem largest_angle_at_C
  (A B C A' B' : Type)
  (triangle_ABC : triangle A B C)
  (reflection_A_through_BC : reflection A BC A')
  (reflection_B_through_AC : reflection B AC B')
  (angle_equality : ∠ B A' C = ∠ B B' C) :
  largest_angle ∠A ∠B ∠C = ∠C := 
sorry

end largest_angle_at_C_l407_407826


namespace binomial_30_3_l407_407586

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l407_407586


namespace bricks_required_l407_407516

-- Courtyard dimensions in meters
def length_courtyard_m := 23
def width_courtyard_m := 15

-- Brick dimensions in centimeters
def length_brick_cm := 17
def width_brick_cm := 9

-- Conversion from meters to centimeters
def meter_to_cm (m : Int) : Int :=
  m * 100

-- Area of courtyard in square centimeters
def area_courtyard_cm2 : Int :=
  meter_to_cm length_courtyard_m * meter_to_cm width_courtyard_m

-- Area of a single brick in square centimeters
def area_brick_cm2 : Int :=
  length_brick_cm * width_brick_cm

-- Calculate the number of bricks needed, ensuring we round up to the nearest whole number
def total_bricks_needed : Int :=
  (area_courtyard_cm2 + area_brick_cm2 - 1) / area_brick_cm2

-- The theorem stating the total number of bricks needed
theorem bricks_required :
  total_bricks_needed = 22550 := by
  sorry

end bricks_required_l407_407516


namespace range_of_x_l407_407686

theorem range_of_x 
  (α : ℝ) (x : ℝ) 
  (hα1 : π / 2 < α)
  (hα2 : α < π) : 
  (sin α - sqrt 3 * cos α = real.log 2 (x^2 - x + 2)) ↔ (x ∈ set.Icc (-1 : ℝ) 0 ∪ set.Ioc 1 2) :=
sorry

end range_of_x_l407_407686


namespace general_term_product_inequality_l407_407421

noncomputable def seq_a : ℕ → ℝ
| 1     := 0
| 2     := -2 / 3
| (n+1) := (seq_a n * seq_a (n + 1) + n * seq_a n) / (n - 1)

theorem general_term (n : ℕ) (h₁ : seq_a 1 = 0) (h₂ : seq_a 2 = -2 / 3) (h₃ : ∀ k ≥ 2, (k - 1) * seq_a (k+1) - k * seq_a k = seq_a k * seq_a (k+1)) :
  seq_a n = 1 / (2 * n - 1) - 1 :=
sorry

theorem product_inequality (n : ℕ) (h₁ : seq_a 1 = 0) (h₂ : seq_a 2 = -2 / 3) (h₃ : ∀ k ≥ 2, (k - 1) * seq_a (k+1) - k * seq_a k = seq_a k * seq_a (k+1)) :
  ∏ k in finset.range n, (2 + seq_a (k + 1)) > real.sqrt (2 * n + 1) :=
sorry

end general_term_product_inequality_l407_407421


namespace number_of_factors_60_l407_407218

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l407_407218


namespace floor_sqrt_sum_l407_407042

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l407_407042


namespace blue_red_token_coincidence_l407_407393

def mid_point (lst : List Nat) : Nat :=
  (lst.length - 1) / 2

theorem blue_red_token_coincidence
  (grid : List (List Bool))  -- A 100x100 grid represented by a list of lists of Booleans
  (h_dim : grid.length = 100 ∧ ∀ row, row ∈ grid → row.length = 100)
  (black_odd : ∀ i < 100, (∑ j in Finset.range 100, if grid[i][j] then 1 else 0) % 2 = 1)
  (col_assign : ∀ j < 100, (∑ i in Finset.range 100, if grid[i][j] then 1 else 0) % 2 = 1)
  (red_positions : Finset (ℕ × ℕ))  -- The set of positions where red tokens are placed
  (h_red : ∀ i < 100, ∃! j < 100, (i, j) ∈ red_positions ∧ grid[i][j])
  (blue_positions : Finset (ℕ × ℕ))  -- The set of positions where blue tokens are placed
  (h_blue : ∀ j < 100, ∃! i < 100, (i, j) ∈ blue_positions ∧ grid[i][j]):
  ∃ cell : (ℕ × ℕ), cell ∈ red_positions ∧ cell ∈ blue_positions :=
by
  sorry

end blue_red_token_coincidence_l407_407393


namespace binomial_coefficient_30_3_l407_407594

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l407_407594


namespace price_difference_is_five_l407_407564

noncomputable def original_price : ℝ := 68 / 0.85
noncomputable def final_price : ℝ := 68 * 1.25
noncomputable def price_difference : ℝ := final_price - original_price

theorem price_difference_is_five :
  price_difference = 5 :=
by
  unfold price_difference
  unfold final_price
  unfold original_price
  calc
    68 * 1.25 - (68 / 0.85) = 85 - 80 : by sorry
                       ... = 5 : by simp

end price_difference_is_five_l407_407564


namespace lattice_points_with_distance_5_from_origin_l407_407802

theorem lattice_points_with_distance_5_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 42 :=
by
  sorry

end lattice_points_with_distance_5_from_origin_l407_407802


namespace sum_floor_sqrt_1_to_25_l407_407003

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l407_407003


namespace part1_part2_l407_407297

-- Definition of points and given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Conditions for part 1
def A1 (a : ℝ) : Point := { x := -2, y := a + 1 }
def B1 (a : ℝ) : Point := { x := a - 1, y := 4 }

-- Definition for distance calculation
def distance (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)

-- Problem 1 Statement
theorem part1 (a : ℝ) (h : a = 3) : distance (A1 a) (B1 a) = 4 :=
by 
  sorry

-- Conditions for part 2
def C2 (b : ℝ) : Point := { x := b - 2, y := b }

-- Problem 2 Statement
theorem part2 (b : ℝ) (h : abs b = 1) :
  (C2 b = { x := -1, y := 1 } ∨ C2 b = { x := -3, y := -1 }) :=
by
  sorry

end part1_part2_l407_407297


namespace number_of_factors_60_l407_407219

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l407_407219


namespace distance_from_T_to_l_l407_407565

def circles_config (A B C T : Point) (ra rb rc : ℝ) : Prop :=
  ∃ (D E F : Point) (ωa ωb ωc : Circle),
    ωa.radius = ra ∧ ωb.radius = rb ∧ ωc.radius = rc ∧
    externally_tangent (ωa, ωb, ωc) at (D, E, F) ∧
    D ∈ line B C ∧ E ∈ line C A ∧ F ∈ line A B ∧
    lines_meet (line B E) (line C F) = T

theorem distance_from_T_to_l 
  (A B C T : Point) (ra rb rc : ℝ)
  (h_circles : circles_config A B C T ra rb rc)
  (h_ra : ra = 341)
  (h_small_circle : ∃ (r : ℝ), r = 49 ∧ internally_tangent_to_all A B C ra rb rc r) 
  (ℓ : Line) 
  (h_l_tangent : tangent_to_all ℓ A B C ra rb rc) : 
  distance T ℓ = 294 := 
sorry

end distance_from_T_to_l_l407_407565


namespace characteristic_properties_l407_407744

-- Definition of characteristic function
def is_characteristic_function (ϕ : ℝ → ℂ) : Prop := ∃ ξ : Type, ∀ t : ℝ, ϕ(t) = (Expectation (exp (complex.i * t * ξ))) -- Simplified for this example

variable (ξ : Type) [IndependentlyDistributed ξ] -- Placeholder for some random variable type and its property

-- Given a characteristic function ϕ
variable (ϕ : ℝ → ℂ)
hypothesis (h : is_characteristic_function ϕ)

-- Proving the required properties are characteristic functions or not
theorem characteristic_properties :
  is_characteristic_function (λ t, complex.re (ϕ t)) ∧
  ¬ is_characteristic_function (λ t, complex.im (ϕ t)) ∧
  is_characteristic_function (λ t, complex.conj (ϕ t)) ∧
  ¬ is_characteristic_function (λ t, complex.abs (ϕ t)) ∧
  is_characteristic_function (λ t, complex.abs (ϕ t) ^ 2) :=
by
  sorry

end characteristic_properties_l407_407744


namespace slope_PQ_l407_407701

noncomputable def parabola := { p : ℝ × ℝ // p.2^2 = 2 * p.1 }

def M := (2, 2)

def line_slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

def intersection_points (M : ℝ × ℝ) (P Q : parabola) : Prop :=
  ∃ k : ℝ, 
    (line_slope M P.1 = k) ∧ 
    (line_slope M Q.1 = -k) ∧ 
    (line_slope M P.1 + line_slope M Q.1 = π)

theorem slope_PQ (M : ℝ × ℝ) (P Q : parabola) 
  (h_intersect : intersection_points M P Q) :
  line_slope P.1 Q.1 = -1/2 :=
sorry

end slope_PQ_l407_407701


namespace total_interest_received_l407_407522

theorem total_interest_received (P_B P_C : ℝ) (T_B T_C : ℝ) (R : ℝ) (SI_B SI_C : ℝ) :
  P_B = 4000 → P_C = 2000 →
  T_B = 2 → T_C = 4 →
  R = 13.75 →
  SI_B = (P_B * R * T_B) / 100 → SI_C = (P_C * R * T_C) / 100 →
  SI_B + SI_C = 2200 :=
by
  intros hP_B hP_C hT_B hT_C hR hSI_B hSI_C
  rw [hP_B, hP_C, hT_B, hT_C, hR, hSI_B, hSI_C]
  sorry

end total_interest_received_l407_407522


namespace original_loaf_had_27_slices_l407_407633

def original_slices : ℕ :=
  let slices_andy_ate := 3 * 2
  let slices_for_toast := 2 * 10
  let slices_left := 1
  slices_andy_ate + slices_for_toast + slices_left

theorem original_loaf_had_27_slices (n : ℕ) (slices_andy_ate : ℕ) (slices_for_toast : ℕ) (slices_left : ℕ) :
  slices_andy_ate = 6 → slices_for_toast = 20 → slices_left = 1 → n = slices_andy_ate + slices_for_toast + slices_left → n = 27 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

-- Verifying the statement
example : original_slices = 27 := by
  have h1 : 3 * 2 = 6 := rfl
  have h2 : 2 * 10 = 20 := rfl
  have h3 : 1 = 1 := rfl
  exact original_loaf_had_27_slices original_slices 6 20 1 h1 h2 h3 rfl

end original_loaf_had_27_slices_l407_407633


namespace family_travel_time_l407_407518

theorem family_travel_time (D : ℕ) (v1 v2 : ℕ) (d1 d2 : ℕ) (t1 t2 : ℕ) :
  D = 560 → 
  v1 = 35 → 
  v2 = 40 → 
  d1 = D / 2 →
  d2 = D / 2 →
  t1 = d1 / v1 →
  t2 = d2 / v2 → 
  t1 + t2 = 15 :=
by
  sorry

end family_travel_time_l407_407518


namespace maria_stops_at_quarter_A_l407_407369

-- Definitions based on the given conditions
def track_circumference : ℕ := 200
def total_distance : ℕ := 5000
def start_point : char := 'S'
def quarter : Type := char
def quarter_A : quarter := 'A'

-- Theorem statement: Maria stops at quarter A
theorem maria_stops_at_quarter_A :
  (total_distance / track_circumference) * track_circumference = total_distance → start_point = 'S' → 'S' = quarter_A :=
by
  intros h1 h2
  sorry

end maria_stops_at_quarter_A_l407_407369


namespace projection_of_a_onto_b_l407_407115

variables (a b : ℝ × ℝ × ℝ) (m : ℝ × ℝ × ℝ)

def vector_projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3 in
  let b_squared_norm := b.1 * b.1 + b.2 * b.2 + b.3 * b.3 in
  (dot_product / b_squared_norm * b.1, dot_product / b_squared_norm * b.2, dot_product / b_squared_norm * b.3)

theorem projection_of_a_onto_b :
  vector_projection (2, -1, 2) (2, 2, 1) = (8/9, 8/9, 4/9) :=
by
  sorry

end projection_of_a_onto_b_l407_407115


namespace sum_integers_100_to_110_l407_407470

theorem sum_integers_100_to_110 : (∑ i in finset.Icc 100 110, i) = 1155 := by
  sorry

end sum_integers_100_to_110_l407_407470


namespace max_consecutive_integers_sum_l407_407454

theorem max_consecutive_integers_sum:
  ∃ k, ∀ n: ℕ, 3 + ∑ i in (range (n - 2)), (3 + i) ≤ 500 → k = 29 := by
sorry

end max_consecutive_integers_sum_l407_407454


namespace unique_sums_l407_407642

-- Definition of the problem statement
def is_unique_sum_of_squares (n : ℕ) : Prop :=
  let representations := {r : List ℕ | r.length ≤ 5 ∧ ∑ x in r, x^2 = n}
  in representations.card = 1

-- Main theorem stating the equivalence of unique sum of squares for specific numbers
theorem unique_sums :
  {n : ℕ | is_unique_sum_of_squares n} = {1, 2, 3, 6, 7, 15} :=
by
  sorry -- Proof is not required in the statement definition

end unique_sums_l407_407642


namespace binomial_30_3_l407_407579

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l407_407579


namespace lattice_points_count_is_30_l407_407789

-- Define the function that counts the number of lattice points (x, y, z) such that x^2 + y^2 + z^2 = 25.
def count_lattice_points := {p : ℤ × ℤ × ℤ // p.1 ^ 2 + p.2.1 ^ 2 + p.2.2 ^ 2 = 25}.to_finset.card

-- The theorem to prove that the count is 30.
theorem lattice_points_count_is_30 : count_lattice_points = 30 :=
by
  sorry

end lattice_points_count_is_30_l407_407789


namespace sequence_stabilizes_l407_407340

-- Define the transformation t
def transform (A : List ℕ) (i : ℕ) : ℕ := 
  List.length (List.filter (λ x, x ≠ A[i]) (A.take i))

-- Define the main theorem to be proven
theorem sequence_stabilizes {n : ℕ} (A : List ℕ) (h₁ : n ≠ 0) 
  (h₂ : ∀ i : ℕ, i ≤ n → A.length = n + 1 → 0 ≤ A[i] ∧ A[i] ≤ i) :
  ∃ B : List ℕ, (k : ℕ) < n ∧ (B = (List.iterate transform k A)) ∧ (B = List.iterate transform k B) := 
sorry

end sequence_stabilizes_l407_407340


namespace cyclic_quadrilateral_diagonal_bisects_l407_407919

noncomputable def cyclic_quadrilateral {α : Type*} [EuclideanGeometry α] (A B C D : Point α) : Prop :=
∃ (O : Circle α), O.contains_point A ∧ O.contains_point B ∧ O.contains_point C ∧ O.contains_point D

noncomputable def tangent_point {α : Type*} [EuclideanGeometry α] (O : Circle α) (E T : Point α) : Prop :=
Line.segment E T ∧ Line.tangent_to_circle E O T

noncomputable def diagonal {α : Type*} [EuclideanGeometry α] (A B : Point α) : Line α :=
Line.join A B

theorem cyclic_quadrilateral_diagonal_bisects 
  {α : Type*} [EuclideanGeometry α]
  (A B C D E T : Point α)
  (O : Circle α)
  (h₀ : cyclic_quadrilateral A B C D)
  (h₁ : Line.intersects (Line.join A D) (Line.join B C) E)
  (h₂ : tangent_point O E T)
  (h₃ : Parallel (diagonal A C) (Line.join E T)) :
  ∃ M : Point α, Midpoint E T M ∧ Line.bisects M (diagonal B D) :=
sorry

end cyclic_quadrilateral_diagonal_bisects_l407_407919


namespace smallest_positive_period_and_range_find_angle_C_l407_407067

theorem smallest_positive_period_and_range (x : ℝ) :
  f x = (1/2) * sin x + (sqrt 3 / 2) * cos x → 
  ∀ x : ℝ, ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ 
  set_of (x : ℝ) (f x) = set_of x ∈ set.Icc (-1 : ℝ) (1 : ℝ) := sorry 

theorem find_angle_C (A B C : ℝ) (a b : ℝ) :
  0 < A → A < π → 
  0 < B → B < π →
  0 < C → C < π →
  f A = sqrt 3 / 2 →
  a = sqrt 3 / 2 * b →
  ∃ C : ℝ, C = π - A - B ∧
  C = π / 6 := sorry

end smallest_positive_period_and_range_find_angle_C_l407_407067


namespace complex_point_coordinates_l407_407269

/-- Given a complex number z, show that its coordinates in the complex plane are as expected -/
theorem complex_point_coordinates {z : ℂ} (h : z = (2 + Complex.i) / Complex.i) : z = 1 - 2 * Complex.i :=
by
  sorry

end complex_point_coordinates_l407_407269


namespace sum_of_floors_of_square_roots_l407_407013

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l407_407013


namespace binom_30_3_eq_4060_l407_407568

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l407_407568


namespace radius_of_larger_circle_l407_407418

theorem radius_of_larger_circle (r : ℝ) 
  (h_ratio : 2 * r = 2 * r ∧ 5 * r = 5 * r) 
  (h_AC_diameter : ∃ A C : Point, AC = 10 * r)
  (h_BC_tangent : ∃ B C : Point, (BC ⟂ tangent smaller_circle))
  (h_AB_eq : AB = 15) :
  5 * r = 18.75 := 
sorry

end radius_of_larger_circle_l407_407418


namespace exists_constant_sum_arrangement_l407_407373

open Finset

-- Define the set of circles and squares
def circles : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define each square's vertices as 4-element subsets of the set {1, 2, 3, 4, 5, 6, 7, 8, 9}
def squares : Finset (Finset ℕ) := { {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}, {5, 6, 7, 8}, {6, 7, 8, 9} }

-- Lean 4 statement representing the problem
theorem exists_constant_sum_arrangement : ∃ (f : Fin ℕ → ℕ), 
  (∀ x, x ∈ { 1, 2, 3, 4, 5, 6, 7, 8, 9 } → f x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
  ∀ s ∈ squares, (∑ i in s, f i) = k :=
begin
  sorry -- Proof to be filled in later
end

end exists_constant_sum_arrangement_l407_407373


namespace original_cost_prices_l407_407987

variable (COST_A COST_B COST_C : ℝ)

theorem original_cost_prices :
  (COST_A * 0.8 + 100 = COST_A * 1.05) →
  (COST_B * 1.1 - 80 = COST_B * 0.92) →
  (COST_C * 0.85 + 120 = COST_C * 1.07) →
  COST_A = 400 ∧
  COST_B = 4000 / 9 ∧
  COST_C = 6000 / 11 := by
  intro h1 h2 h3
  sorry

end original_cost_prices_l407_407987


namespace work_completion_l407_407544

theorem work_completion (amit_rate : ℝ) (ananthu_rate : ℝ) (amit_days : ℝ) (total_days : ℝ) : 
  amit_rate = 1 / 15 ∧ ananthu_rate = 1 / 30 ∧ amit_days = 3 → 
  total_days = amit_days + (1 - amit_days * amit_rate) / ananthu_rate → 
  total_days = 7.8 :=
by
  intros h1 h2
  cases h1 with h_amit_rate h_ananthu_rate
  cases h_ananthu_rate with h_ananthu_rate h_amit_days
  rw [h_amit_rate, h_ananthu_rate, h_amit_days] at *,
  exact h2

end work_completion_l407_407544


namespace necessarily_yellow_and_red_packet_l407_407934

noncomputable def problem_statement : Prop :=
  ∃ (packets : list (list string)) (yellow blue red : nat), 
    yellow + blue + red = 12 ∧
    (∀ (pkt : list string), pkt ∈ packets → pkt.length = 4 ∧
      ¬(∀ c, pkt.count c = 3) ∧
      (∃ c1 c2, pkt.count c1 = 2 ∧ pkt.count c2 = 2)) ∧
    (list.nodup packets) ∧
    (∃ pkt, pkt ∈ packets ∧ pkt.count "yellow" = 2 ∧ pkt.count "red" = 2)

theorem necessarily_yellow_and_red_packet : problem_statement :=
  sorry

end necessarily_yellow_and_red_packet_l407_407934


namespace sum_floor_sqrt_1_to_25_l407_407034

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l407_407034


namespace larry_substitution_l407_407850

theorem larry_substitution (a b c d e : ℤ)
  (h_a : a = 2)
  (h_b : b = 5)
  (h_c : c = 3)
  (h_d : d = 4)
  (h_expr1 : a + b - c - d * e = 4 - 4 * e)
  (h_expr2 : a + (b - (c - (d * e))) = 4 + 4 * e) :
  e = 0 :=
by
  sorry

end larry_substitution_l407_407850


namespace retirement_year_2020_l407_407437

-- Given conditions
def femaleRetirementAge := 55
def initialRetirementYear (birthYear : ℕ) := birthYear + femaleRetirementAge
def delayedRetirementYear (baseYear additionalYears : ℕ) := baseYear + additionalYears

def postponementStep := 3
def delayStartYear := 2018
def retirementAgeIn2045 := 65
def retirementYear (birthYear : ℕ) : ℕ :=
  let originalRetirementYear := initialRetirementYear birthYear
  let delayYears := ((originalRetirementYear - delayStartYear) / postponementStep) + 1
  delayedRetirementYear originalRetirementYear delayYears

-- Main theorem to prove
theorem retirement_year_2020 : retirementYear 1964 = 2020 := sorry

end retirement_year_2020_l407_407437


namespace binom_30_3_l407_407610

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l407_407610


namespace num_factors_60_l407_407215

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l407_407215


namespace sequence_is_decreasing_l407_407691

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

theorem sequence_is_decreasing (a : ℕ → ℝ) (h1 : a 1 < 0) (h2 : is_geometric_sequence a (1/3)) :
  ∀ n, a (n + 1) < a n :=
by
  -- Here should be the proof
  sorry

end sequence_is_decreasing_l407_407691


namespace work_completion_days_l407_407484

theorem work_completion_days (a b c : ℝ) :
  (1/a) = 1/90 → (1/b) = 1/45 → (1/a + 1/b + 1/c) = 1/5 → c = 6 :=
by
  intros ha hb habc
  sorry

end work_completion_days_l407_407484


namespace evaluate_expression_l407_407333

theorem evaluate_expression : 
  let triangle := 1 in
  let circ := -1 in
  let square := 0 in
  (triangle + square) * circ = -1 := 
by
  sorry

end evaluate_expression_l407_407333


namespace determine_d_l407_407381

theorem determine_d (u v d c : ℝ) (p q : ℝ → ℝ)
  (hp : ∀ x, p x = x^3 + c * x + d)
  (hq : ∀ x, q x = x^3 + c * x + d + 300)
  (huv : p u = 0 ∧ p v = 0)  
  (hu5_v4 : q (u + 5) = 0 ∧ q (v - 4) = 0)
  (sum_roots_p : u + v + (-u - v) = 0)
  (sum_roots_q : (u + 5) + (v - 4) + (-u - v - 1) = 0)
  : d = -4 ∨ d = 6 :=
sorry

end determine_d_l407_407381


namespace number_of_factors_60_l407_407222

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l407_407222


namespace range_of_m_l407_407687

theorem range_of_m (m : ℝ) (x : ℝ) 
  (H1 : ∀ x, x^2 - 8x - 20 ≤ 0) 
  (H2 : m > 0) 
  (H3 : ∀ x, x^2 - 2x + 1 - m^2 ≤ 0) 
  (H4 : (x > 10 ∨ x < -2) → (x < 1 - m ∨ x > 1 + m)) : 
  0 < m ∧ m ≤ 3 := 
sorry

end range_of_m_l407_407687


namespace formalized_problem_l407_407929

-- Define the conditions 
variables {P A B C D : Type*}
variable [ring P]
variables (AP : P) (CP : P) (BP : P) (DP : P)

-- Assume AP = 5 and CP = 7
def condition1 : AP = 5 := sorry
def condition2 : CP = 7 := sorry

-- Use the Power of a Point theorem
def power_of_point (AP BP CP DP : P) : AP * BP = CP * DP := sorry

-- State the proof problem
theorem formalized_problem (h1 : AP = 5) (h2 : CP = 7) (h3 : power_of_point AP BP CP DP):
  BP / DP = 7 / 5 :=
sorry

end formalized_problem_l407_407929


namespace odd_and_decreasing_function_l407_407992

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def is_decreasing (f : ℝ → ℝ) (S : set ℝ) : Prop :=
  ∀ x y ∈ S, x < y → f (x) > f (y)

def f1 (x : ℝ) : ℝ := -x^3
def f2 (x : ℝ) : ℝ := real.log (1 / 2) x
def f3 (x : ℝ) : ℝ := x
def f4 (x : ℝ) : ℝ := (1 / 2)^x

theorem odd_and_decreasing_function :
  (is_odd f1 ∧ is_decreasing f1 univ) ∧
  ¬(is_odd f2 ∧ is_decreasing f2 (set.Ioi 0)) ∧
  ¬(is_odd f3 ∧ is_decreasing f3 univ) ∧
  ¬(is_odd f4 ∧ is_decreasing f4 univ) := by
  sorry

end odd_and_decreasing_function_l407_407992


namespace triangle_area_range_l407_407818

theorem triangle_area_range (A B C : ℝ) (a b c : ℝ) 
  (h1 : a * Real.sin B = Real.sqrt 3 * b * Real.cos A)
  (h2 : a = 3) :
  0 < (1 / 2) * b * c * Real.sin A ∧ 
  (1 / 2) * b * c * Real.sin A ≤ (9 * Real.sqrt 3) / 4 := 
  sorry

end triangle_area_range_l407_407818


namespace arithmetic_sequence_sum_l407_407298

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 = 2) (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l407_407298


namespace problem_statement_l407_407364

namespace LeanProofExample

def not_divisible (n : ℕ) (p : ℕ) : Prop :=
  ¬(p ∣ n)

theorem problem_statement (x y : ℕ) 
  (hx : not_divisible x 59) 
  (hy : not_divisible y 59)
  (h : 3 * x + 28 * y ≡ 0 [MOD 59]) :
  ¬(5 * x + 16 * y ≡ 0 [MOD 59]) :=
  sorry

end LeanProofExample

end problem_statement_l407_407364


namespace odd_prime_or_9_factorial_not_divisible_l407_407643

theorem odd_prime_or_9_factorial_not_divisible (n : ℕ) (h_odd : n % 2 = 1) :
    (Prime n ∨ n = 9) → ¬ (n^2 ∣ (nat.factorial (n - 1))) :=
by
  sorry

end odd_prime_or_9_factorial_not_divisible_l407_407643


namespace equal_area_division_l407_407350

noncomputable def isosceles_right_triangle (A B C : Point) : Prop :=
  ∃ (right_angle : is_right_angle B) (hypotenuse : AC = sqrt(2) * AB), 
  Triangle A B C

noncomputable def square_on_hypotenuse (A C D E : Point) (AC : Line) : Prop :=
  Square A C D E

noncomputable def perpendicular_from_point (B : Point) (DE : Line) (K : Point) : Prop :=
  Perpendicular B DE K

noncomputable def divides_pentagon (AD CE BD BE BK : Line) (ABCDE Pentagon) : Prop :=
  ∀ regions, equal_area_divisions regions 

theorem equal_area_division (A B C D E K : Point)
  (h_triangle : isosceles_right_triangle A B C)
  (h_square : square_on_hypotenuse A C D E)
  (h_diags : intersects_at D_A D_C E)
  (h_perpendicular : perpendicular_from_point B D_E)
  : divides_pentagon D_A D_C B_D B_E B_K (pentagon A B C D E) :=
sorry

end equal_area_division_l407_407350


namespace tax_collection_amount_l407_407638

theorem tax_collection_amount (paid_tax : ℝ) (willam_percentage : ℝ) (total_collected : ℝ) (h_paid: paid_tax = 480) (h_percentage: willam_percentage = 0.3125) :
    total_collected = 1536 :=
by
  sorry

end tax_collection_amount_l407_407638


namespace evaluate_expression_l407_407636

noncomputable def log_4_8 : ℝ := Real.log 8 / Real.log 4
noncomputable def log_8_16 : ℝ := Real.log 16 / Real.log 8

theorem evaluate_expression : Real.sqrt (log_4_8 * log_8_16) = Real.sqrt 2 :=
by
  sorry

end evaluate_expression_l407_407636


namespace stack_of_crates_probability_l407_407879

theorem stack_of_crates_probability:
  let crates := 10 in
  let dimensions := [(3, 4, 6)] in
  -- Define quantities
  (prob_height_41 : (∑ a, b, c in ℤ,
    3 * a + 4 * b + 6 * c = 41 ∧ a + b + c = 10) / 3^10) ∈
  (m, n: ℕ) ∧ m.gcd n = 1 
  → m = 190 :=
sorry

end stack_of_crates_probability_l407_407879


namespace jane_hector_meet_at_corner_C_l407_407754

theorem jane_hector_meet_at_corner_C 
  (s : ℝ) -- Hector's speed
  (t : ℝ) -- Time of travel
  (a b c d : Point) -- Rectangle corners
  (x y : ℝ) -- Side lengths of the rectangle
  (hx_ratio : x / y = 3 / 2)
  (hx_perimeter : 2 * x + 2 * y = 24)
  (hx_start : Hector.start = a)
  (hy_start : Jane.start = a)
  (hx_dir : Hector.direction = "towards D")
  (hy_dir : Jane.direction = "towards B")
  (hx_speed : Jane.speed = 2 * Hector.speed)
  (h_seconds : 3 * s * t = 24)
  (hx_dist : Hector.distance_travelled = 8)
  (hy_dist : Jane.distance_travelled = 16)
  : Hector.nearest_corner_to_meeting_point = c := 
sorry

end jane_hector_meet_at_corner_C_l407_407754


namespace find_pairs_l407_407658

noncomputable theory

theorem find_pairs (a b : ℕ) (P Q : ℤ[X]) (h : ∀ (n : ℕ), π (a * n) * Q.eval n = π (b * n) * P.eval n) : a = b :=
sorry

end find_pairs_l407_407658


namespace binom_30_3_l407_407597

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l407_407597


namespace largest_k_with_three_identical_teddy_bears_l407_407536

noncomputable def max_colors_with_three_identical_bears (total_bears : Nat) (sizes : Nat) : ℕ :=
  Nat.find (λ k, ∃ (bears_per_type : Nat), sizes * k = bears_per_type ∧ 
                         total_bears > 2 * bears_per_type ∧ sizes * k = 2 * k * sizes / 2)

theorem largest_k_with_three_identical_teddy_bears (total_bears : Nat := 125) (sizes : Nat := 6) : max_colors_with_three_identical_bears total_bears sizes = 10 :=
  sorry

end largest_k_with_three_identical_teddy_bears_l407_407536


namespace num_factors_of_60_l407_407196

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l407_407196


namespace third_side_of_triangle_l407_407167

theorem third_side_of_triangle (a b v : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : v > 0) :
  ∃ c : ℝ, c = (a + b) * sqrt ((a * b - v^2) / (a * b)) :=
sorry

end third_side_of_triangle_l407_407167


namespace arithmetic_progression_impossible_geometric_progression_possible_l407_407559

theorem arithmetic_progression_impossible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  2 * b ≠ a + c :=
by {
    sorry
}

theorem geometric_progression_possible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  ∃ r m : ℤ, (b / a)^r = (c / a)^m :=
by {
    sorry
}

end arithmetic_progression_impossible_geometric_progression_possible_l407_407559


namespace num_factors_60_l407_407186

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l407_407186


namespace probability_more_than_6_grandsons_or_more_than_6_granddaughters_l407_407359

theorem probability_more_than_6_grandsons_or_more_than_6_granddaughters :
  let total_grandchildren := 12
  let total_combinations := 2^total_grandchildren
  let exactly_half_combinations := 12.choose 6
  let probability_exactly_half := exactly_half_combinations / total_combinations
  (1 - probability_exactly_half) = 793 / 1024 := by
  sorry

end probability_more_than_6_grandsons_or_more_than_6_granddaughters_l407_407359


namespace binomial_30_3_l407_407587

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l407_407587


namespace max_curvilinear_sides_l407_407639

theorem max_curvilinear_sides (n : ℕ) (h : 2 ≤ n) : 
  let F := intersection of n circles,
  let curvilinear_side (f : F) := f ∈ (boundary of F) ∧ f ∈ (boundary of one circle) ∧ (bounded by intersection points),
  let curvilinear_sides := set of curvilinear_side,
  curvilinear_sides.card = 2 * n - 2 :=
sorry

end max_curvilinear_sides_l407_407639


namespace no_maximal_radius_of_inscribed_cylinder_l407_407882

theorem no_maximal_radius_of_inscribed_cylinder
  (base_radius_cone : ℝ) (height_cone : ℝ)
  (h_base_radius : base_radius_cone = 5) (h_height : height_cone = 10) :
  ¬ ∃ r : ℝ, 0 < r ∧ r < 5 ∧
    ∀ t : ℝ, 0 < t ∧ t < 5 → 2 * Real.pi * (10 * r - r ^ 2) ≥ 2 * Real.pi * (10 * t - t ^ 2) :=
by
  sorry

end no_maximal_radius_of_inscribed_cylinder_l407_407882


namespace num_factors_of_60_l407_407194

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l407_407194


namespace fraction_power_l407_407965

variables (a b c : ℝ)

theorem fraction_power :
  ( ( -2 * a^2 * b ) / (3 * c) )^2 = ( 4 * a^4 * b^2 ) / ( 9 * c^2 ) := 
by sorry

end fraction_power_l407_407965


namespace max_expression_l407_407651

theorem max_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 25) : 
  ∃ y, y = sqrt(x + 64) + sqrt(25 - x) + 2 * sqrt(x) ∧ ∀ z, (0 ≤ z ∧ z ≤ 25) → 
  sqrt(z + 64) + sqrt(25 - z) + 2 * sqrt(z) ≤ y := by
  use sqrt(64 / 15 + 64) + sqrt(25 - 64 / 15) + 2 * sqrt(64 / 15)
  sorry

end max_expression_l407_407651


namespace total_students_correct_l407_407998

-- Define the parameters given in the problem
def halfDayPercentage : ℚ := 0.25
def fullDayStudents : ℕ := 60
def totalStudents : ℕ := 80

-- Define the proof statement
theorem total_students_correct :
  fullDayStudents = 0.75 * totalStudents → 
  totalStudents = 80 :=
by
  sorry

end total_students_correct_l407_407998


namespace sum_of_integers_100_to_110_l407_407464

theorem sum_of_integers_100_to_110 : ∑ i in Finset.range (111 - 100), (100 + i) = 1155 :=
by
  sorry

end sum_of_integers_100_to_110_l407_407464


namespace number_of_factors_60_l407_407225

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l407_407225


namespace common_chord_length_l407_407930

noncomputable def length_common_chord (r : ℝ) (h_eq_triangle : ∀ (A B C : ℝ), A = r ∧ B = r ∧ C = r ) : ℝ :=
  let altitude := (sqrt 3 / 2) * r in
  2 * altitude

theorem common_chord_length (r : ℝ) (h_eq_triangle : ∀ (A B C : ℝ), A = r ∧ B = r ∧ C = r) :
  r = 15 → length_common_chord r h_eq_triangle = 15 * sqrt 3 :=
by
  intro hr
  sorry

end common_chord_length_l407_407930


namespace chickens_cheaper_than_buying_eggs_l407_407383

theorem chickens_cheaper_than_buying_eggs :
  ∃ W, W ≥ 80 ∧ 80 + W ≤ 2 * W :=
by
  sorry

end chickens_cheaper_than_buying_eggs_l407_407383


namespace prob_div_by_4_l407_407913

theorem prob_div_by_4 (N : ℕ) (h1 : 100 ≤ N ∧N ≤ 999) (h2 : N % 10 = 4) : 
  let count_div_by_4 := (∑ y in Finset.range 10, if (10 * y + 4) % 4 = 0 then 1 else 0)
  ∑ y in Finset.range 10, if (10 * y + 4) % 4 = 0 then 1 else 0 = 5 → 
  count_div_by_4 / 10 = 1 / 2 :=
by
  sorry

end prob_div_by_4_l407_407913


namespace sum_floor_sqrt_l407_407051

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l407_407051


namespace incorrect_option_l407_407993

theorem incorrect_option (r : ℝ) (x y : ℝ) :
  (∀ (r : ℝ), abs r > 0 → abs r ≤ 1
    -- Condition A: Correlation coefficient absolute value indicates degree of linear correlation.
  ) ∧
  (∀ (x y : list ℤ), scatter_plot x y
    -- Condition B: Scatter plots in Cartesian coordinate system reveal variation patterns.
  ) ∧
  (∀ (x y : ℝ), linear_regression_equation x y 
    -- Condition C: Linear regression equation represents observed value relationship.
  ) →
  ¬ (∀ (x y : list ℤ), regression_line_significance x y)
    -- Incorrect condition D: Any observed values yield a regression line with representative significance.
:= sorry

end incorrect_option_l407_407993


namespace quadratic_distinct_roots_l407_407669

theorem quadratic_distinct_roots (a : ℝ) : 
  (a > -1 ∧ a ≠ 3) ↔ 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (a - 3) * x₁^2 - 4 * x₁ - 1 = 0 ∧ 
    (a - 3) * x₂^2 - 4 * x₂ - 1 = 0 :=
by
  sorry

end quadratic_distinct_roots_l407_407669


namespace quadrisectAD_l407_407497

open Classical

variables {A B C D M : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M]

noncomputable def quadrisect (A D : Type) : Prop :=
∃ (B C : Type), B ∈ MetricSpace.quad (A, D) ∧ C ∈ MetricSpace.quad (B, D)

noncomputable def midpoint (A B : Type) : Prop :=
∃ (M : Type), M ∈ MetricSpace.half (A, B)

variable (length : (A → B) → ℝ)

def AD_length (AD : A → D) : Prop :=
AD = (A → B) → (B → C) →(C → D)

theorem quadrisectAD (h1 : quadrisect A D) (h2 : midpoint A B) (h3 : length (A → C) = 10) : length (A → D) = 80 / 3 := 
by sorry

end quadrisectAD_l407_407497


namespace proof_eventually_constant_b_l407_407496

noncomputable def eventually_constant_b {a : ℕ → ℕ} (n : ℕ) : Prop := 
  (∀ m ≥ 2016, m^2 ∣ (∑ i in finset.range (m+1), a i)) ∧ 
  (∀ m ≥ 2016, a m ≤ (m + 2016)^2) ∧
  (∃ k, ∀ n ≥ k, a (n + 1) - a n = a (k + 1) - a k)

theorem proof_eventually_constant_b (a : ℕ → ℕ) : eventually_constant_b a :=
sorry

end proof_eventually_constant_b_l407_407496


namespace min_shots_for_destruction_l407_407857

-- Define the 10x10 game board and the 4-cell ship conditions
constant game_board : Type
constant cell : Type
constant ship : game_board → Type

-- Define the constraints on the game board and ship
axiom board_size : ∀ b : game_board, (∃ rows cols: ℕ, rows = 10 ∧ cols = 10)
axiom ship_size : ∀ s : ship game_board, (∃ length : ℕ, length = 4)

-- Define the function that counts the number of shots required 
constant min_shots_to_destroy : game_board → ℕ

-- Define the given game board instance
constant battleship_board : game_board

-- The theorem statement claiming the minimum shots required
theorem min_shots_for_destruction :
  min_shots_to_destroy battleship_board = 14 :=
sorry -- Proof is not required as per instructions

end min_shots_for_destruction_l407_407857


namespace lattice_points_with_distance_five_from_origin_l407_407808

theorem lattice_points_with_distance_five_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 6 :=
by
  sorry

end lattice_points_with_distance_five_from_origin_l407_407808


namespace triangle_circle_AE_radius_l407_407306

theorem triangle_circle_AE_radius
  (A B C D E : Type)
  (dist : A → B → ℝ)
  (h_triangle_ABC : ∃ (ABC : Triangle A B C), ∠ABC.A = 60°)
  (h_circle_BCD : Circle (B) (C) (D))
  (h_d_on_AB : D ∈ LineSegment A B)
  (h_circle_intersects_E : E ∈ Intersect (LineSegment A C, Circle B C D))
  (h_AD : dist A D = 3)
  (h_BD : dist B D = 1)
  (h_EC : dist E C = 4) :
  dist A E = 2 ∧ Radius (Circle B C D) = sqrt 7 := by
  sorry

end triangle_circle_AE_radius_l407_407306


namespace find_formula_for_a_n_l407_407331

noncomputable def sequence_a : ℕ → ℤ
| 1 := -2
| (n + 1) := 2 * sequence_a n

def sum_first_n (s : ℕ → ℤ) (n : ℕ) : ℤ :=
  (finset.range n).sum (λ k, s (k + 1))

theorem find_formula_for_a_n (n : ℕ) :
  let a := sequence_a
  let S := sum_first_n a
  a 1 = -2 → 
  (∀ n, S n = 2 * a n + 2) →
  a n = -2 ^ n :=
by
  intros a S h1 h2
  sorry

end find_formula_for_a_n_l407_407331


namespace centroid_locus_hyperbola_max_distance_E_to_MN_l407_407681

-- Define the conditions and proof statements
-- Part 1: Proof of the Locus G being a hyperbola
theorem centroid_locus_hyperbola {A B C G : ℝ × ℝ} (hB : B = (-2, 0)) (hC : C = (2, 0)) 
  (h_midline_diff : abs ((B.1 + A.1) / 2 - (C.1 + A.1) / 2) = 3 * sqrt 3) :
  G = (x, y) → (x^2 / 3) - y^2 = 1 → x ≠ ±sqrt 3 := sorry

-- Part 2: Proof of the maximum distance from E to MN
theorem max_distance_E_to_MN {E F Q M N : ℝ × ℝ} (hE : E = (-sqrt 3, 0)) (hF : F = (sqrt 3, 0))
  (hQ : Q.1 = 3 / 2) 
  (hEQ : ∀ x, (E.1 * Q.1 / (Q.1 - sqrt 3)) = M.2) 
  (hFQ : ∀ x, (F.1 * Q.1 / (Q.1 + sqrt 3)) = N.2) :
  abs ((M.2 - N.2) ^ 2 + (M.1 - N.1) ^ 2) ^ (1 / 2) = 2 + sqrt 3 := sorry

end centroid_locus_hyperbola_max_distance_E_to_MN_l407_407681


namespace sum_geom_seq_q1_sum_geom_seq_q_neg2_l407_407715

-- Define the geometric sequence and sum of first n terms 
def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1) 

def sum_geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q ^ n) / (1 - q)

-- Given conditions
def a₁ : ℝ := 1
def S₃ : ℝ := 3

-- Correct answers
def S_n_q1 (n : ℕ) : ℝ := n
def S_n_q_neg2 (n : ℕ) : ℝ := (1 - (-2 : ℝ) ^ n) / 3

theorem sum_geom_seq_q1 (n : ℕ) : sum_geom_seq a₁ 1 n = S_n_q1 n :=
sorry

theorem sum_geom_seq_q_neg2 (n : ℕ) : sum_geom_seq a₁ (-2) n = S_n_q_neg2 n :=
sorry

-- Finally, we need to prove these theorems

end sum_geom_seq_q1_sum_geom_seq_q_neg2_l407_407715


namespace find_b_and_domain_range_l407_407722

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^3 + b * x^2 + c * x

noncomputable def g (t : ℝ) : ℝ := -2 * t^3 + 6 * t^2

theorem find_b_and_domain_range
  (f : ℝ → ℝ)
  (b c t : ℝ)
  (h1 : f = λ x, x^3 + b * x^2 + c * x)
  (h2 : ∃ t : ℝ, t > 2 ∧ f (t) = g t)
  (h3 : (∀ x, 3 * x^2 + 2 * b * x + c = 0 → x = 2))
  : b = -6 ∧ (∀ t > 2, g t < 8) ∧ (∀ y, ∃ t > 2, g t = y) :=
by
  sorry

end find_b_and_domain_range_l407_407722


namespace solution_set_of_f_lt_x_l407_407688

noncomputable def f (x : ℝ) : ℝ :=
sorry

theorem solution_set_of_f_lt_x (f : ℝ → ℝ) 
  (h1 : ∀ x, deriv f x < f x)
  (h2 : ∀ x, f (x + 2) = f (-(x + 2)))
  (h3 : f 0 = 1) :
  {x : ℝ | f x < x} = set.Ioi 0 :=
sorry

end solution_set_of_f_lt_x_l407_407688


namespace lattice_points_with_distance_5_from_origin_l407_407797

theorem lattice_points_with_distance_5_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 42 :=
by
  sorry

end lattice_points_with_distance_5_from_origin_l407_407797


namespace sum_floor_sqrt_l407_407053

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l407_407053


namespace ratio_of_perimeters_l407_407957

theorem ratio_of_perimeters (s : ℝ) :
  let small_perimeter := 4 * s
      large_perimeter := 4 * (2 * s) in
  large_perimeter / small_perimeter = 2 :=
by
  let small_diagonal := s * real.sqrt 2
  let large_diagonal := 2 * small_diagonal
  have large_side : ℝ := 2 * s
  sorry

end ratio_of_perimeters_l407_407957


namespace guests_not_eating_brownies_ala_mode_l407_407928

theorem guests_not_eating_brownies_ala_mode (total_brownies : ℕ) (eaten_brownies : ℕ) (eaten_scoops : ℕ)
    (scoops_per_serving : ℕ) (scoops_per_tub : ℕ) (tubs_eaten : ℕ) : 
    total_brownies = 32 → eaten_brownies = 28 → eaten_scoops = 48 → scoops_per_serving = 2 → scoops_per_tub = 8 → tubs_eaten = 6 → (eaten_scoops - eaten_brownies * scoops_per_serving) / scoops_per_serving = 4 :=
by
  intros
  sorry

end guests_not_eating_brownies_ala_mode_l407_407928


namespace num_factors_of_60_l407_407237

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l407_407237


namespace least_period_g_of_x_over_3_least_period_H_least_period_G_least_period_F_l407_407978

-- Problem (a)
theorem least_period_g_of_x_over_3 (g : ℝ → ℝ) (h_period : ∀ x, g (x + π) = g x) :
  (∃ T, ∀ x, g (x / 3 + T) = g (x / 3)) ∧ (∀ T', (∃ x, g ((x + T') / 3) ≠ g (x / 3)) → T' ≥ 3 * π) :=
sorry

-- Problem (b)
theorem least_period_H : ∃ T, ∀ x, (sin (8 * x) + cos (4 * x)) = (sin (8 * (x + T)) + cos (4 * (x + T))) ∧
   (∀ T', (∃ x, (sin (8 * (x + T')) + cos (4 * (x + T'))) ≠ (sin (8 * x) + cos (4 * x))) → T' ≥ π / 2) :=
sorry

-- Problem (c)i
theorem least_period_G : ∃ T, ∀ x, sin (cos x) = sin (cos (x + T)) ∧
   (∀ T', (∃ x, sin (cos (x + T')) ≠ sin (cos x)) → T' ≥ 2 * π) :=
sorry

-- Problem (c)ii
theorem least_period_F : ∃ T, ∀ x, cos (sin x) = cos (sin (x + T)) ∧
   (∀ T', (∃ x, cos (sin (x + T')) ≠ cos (sin x)) → T' ≥ π) :=
sorry

end least_period_g_of_x_over_3_least_period_H_least_period_G_least_period_F_l407_407978


namespace num_lattice_points_l407_407752

noncomputable def lattice_points : ℕ :=
  ∑ (x : ℤ) in {-3, -2, -1, 0, 1, 2, 3}, ∑ (y : ℤ) in {-3, -2, -1, 0, 1, 2, 3}, 
    if ((|x| - 1)^2 + (|y| - 1)^2 < 2) then 1 else 0

theorem num_lattice_points :
  lattice_points = 16 :=
sorry

end num_lattice_points_l407_407752


namespace probability_mean_greater_median_l407_407312
open Nat

theorem probability_mean_greater_median : 
  let fair_six_sided_dice := {d : Nat | d ∈ Finset.range 1 7}
  let outcomes : Finset (Finset (Finset fair_six_sided_dice)) := Finset.powerset (Finset.product fair_six_sided_dice fair_six_sided_dice) 
  ∀ (a b c : fair_six_sided_dice), 
    let mean := (a + b + c) / 3
    let sorted := List.sort [a, b, c]
    let median := sorted.head _ (* considered sorted list of [a, b, c] case analysis on length *)
    mean > median → 
    (∃ m n, m = 29 ∧ n = 72 ∧ gcd m n = 1 ∧ m + n = 101) :=
sorry

end probability_mean_greater_median_l407_407312


namespace smallest_abc_sum_l407_407641

theorem smallest_abc_sum : 
  ∃ (a b c : ℕ), (a * c + 2 * b * c + a + 2 * b = c^2 + c + 6) ∧ (∀ (a' b' c' : ℕ), (a' * c' + 2 * b' * c' + a' + 2 * b' = c'^2 + c' + 6) → (a' + b' + c' ≥ a + b + c)) → (a, b, c) = (2, 1, 1) := 
by
  sorry

end smallest_abc_sum_l407_407641


namespace floor_sqrt_sum_l407_407045

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l407_407045


namespace not_perfect_cube_of_cond_l407_407327

open Int

theorem not_perfect_cube_of_cond (n : ℤ) (h₁ : 0 < n) (k : ℤ) 
  (h₂ : n^5 + n^3 + 2 * n^2 + 2 * n + 2 = k ^ 3) : 
  ¬ ∃ m : ℤ, 2 * n^2 + n + 2 = m ^ 3 :=
sorry

end not_perfect_cube_of_cond_l407_407327


namespace product_of_solutions_eq_zero_l407_407904

theorem product_of_solutions_eq_zero : 
  (∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4)) → 
  ∃ (x1 x2 : ℝ), (x1 = 0 ∨ x1 = 5) ∧ (x2 = 0 ∨ x2 = 5) ∧ x1 * x2 = 0 :=
by
  sorry

end product_of_solutions_eq_zero_l407_407904


namespace probability_at_least_one_black_eq_seven_tenth_l407_407917

noncomputable def probability_drawing_at_least_one_black_ball : ℚ :=
  let total_ways := Nat.choose 5 2
  let ways_no_black := Nat.choose 3 2
  1 - (ways_no_black / total_ways)

theorem probability_at_least_one_black_eq_seven_tenth :
  probability_drawing_at_least_one_black_ball = 7 / 10 :=
by
  sorry

end probability_at_least_one_black_eq_seven_tenth_l407_407917


namespace binom_30_3_l407_407612

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l407_407612


namespace evaluate_expr_l407_407080

theorem evaluate_expr :
  (3 * Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) = 
  -1 / 6 * (Real.sqrt 21 + Real.sqrt 35 - Real.sqrt 77) - 7 / 3 := by
  sorry

end evaluate_expr_l407_407080


namespace ellipse_properties_l407_407693

noncomputable def ellipse_equation (a b : ℝ) (h : a > b > 0) : Prop :=
  ∃ e, e = 1/2 ∧ ∃ dist, dist = 2 ∧ ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1

def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

theorem ellipse_properties (a b : ℝ) (h : a > b > 0) (k : ℝ):
  (∃ c, (c / a = 1/2) ∧ c = 1 ∧ b = real.sqrt 3 ∧ 
  ((x y : ℝ), (x^2 / 4) + (y^2 / 3) = 1)) ∧
  (∀ P M N, ∃ x0 y0 x1 y1, (P = (x0, y0)) ∧ (M = (x1, y1)) ∧ (N = (-x1, -y1)) 
            ∧ (y0 + y1) / (x0 + x1) * (y0 - y1) / (x0 - x1) = -3/4) ∧
  (∀ P M N, ∃ max_area, max_area = 2 * real.sqrt 3)
:=
begin
  sorry
end

end ellipse_properties_l407_407693


namespace total_vessels_l407_407510

open Nat

theorem total_vessels (x y z w : ℕ) (hx : x > 0) (hy : y > x) (hz : z > y) (hw : w > z) :
  ∃ total : ℕ, total = x * (2 * y + 1) + z * (1 + 1 / w) := sorry

end total_vessels_l407_407510


namespace plane_tetrahedra_l407_407286

theorem plane_tetrahedra (n : ℕ) (h1 : n ≥ 5) 
  (h2 : ∀ (p1 p2 p3 : Plane), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → 
                             (∃! x : Point, x ∈ p1 ∧ x ∈ p2 ∧ x ∈ p3 ))
  (h3 : ∀ (P : Finset Plane), P.card = 5 → 
                             ¬(∃ x : Point, ∀ p ∈ P, x ∈ p)) :
  ∃ k, k ≥ (2 * n - 3) / 4 := sorry

end plane_tetrahedra_l407_407286


namespace evaluate_h_j_l407_407840

def h (x : ℝ) : ℝ := 3 * x - 4
def j (x : ℝ) : ℝ := x - 2

theorem evaluate_h_j : h (2 + j 3) = 5 := by
  sorry

end evaluate_h_j_l407_407840


namespace sum_series_proof_l407_407376

open Nat

theorem sum_series_proof (n : ℕ) (h : 0 < n): 
  ∑ i in range n, 1 / ((2 * i + 1) * (2 * i + 3)) = n / (2 * n + 1) :=
sorry

end sum_series_proof_l407_407376


namespace cl_mass_percentage_in_ccl4_l407_407093

noncomputable def mass_percentage_of_cl_in_ccl4 : ℝ :=
  let mass_C : ℝ := 12.01
  let mass_Cl : ℝ := 35.45
  let num_Cl : ℝ := 4
  let total_mass_Cl : ℝ := num_Cl * mass_Cl
  let total_mass_CCl4 : ℝ := mass_C + total_mass_Cl
  (total_mass_Cl / total_mass_CCl4) * 100

theorem cl_mass_percentage_in_ccl4 :
  abs (mass_percentage_of_cl_in_ccl4 - 92.19) < 0.01 := 
sorry

end cl_mass_percentage_in_ccl4_l407_407093


namespace trapezoid_tangent_parallel_lines_l407_407959

open EuclideanGeometry

theorem trapezoid_tangent_parallel_lines 
  {A B C D : Point} 
  (h_trapezoid : trapezoid A B C D)
  (h_parallel : parallel A B C D)
  (w1 w2 : Circle)
  (h_tangent_w1 : tangent_circle_to_lines w1 D A B)
  (h_tangent_w2 : tangent_circle_to_lines w2 B C D)
  (l1 : Line)
  (h_tangent_l1 : tangent_line_through_point w2 A l1 ∧ l1 ≠ line_through D A)
  (l2 : Line)
  (h_tangent_l2 : tangent_line_through_point w1 C l2 ∧ l2 ≠ line_through C B) :
  parallel l1 l2 :=
sorry

end trapezoid_tangent_parallel_lines_l407_407959


namespace half_area_cutting_line_l407_407926

def P : ℝ × ℝ := (0, 10)
def Q : ℝ × ℝ := (3, 0)
def R : ℝ × ℝ := (9, 0)

def M : ℝ × ℝ := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)
def y_intercept (m x1 y1 : ℝ) : ℝ := y1 - m * x1

def line_eq (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem half_area_cutting_line :
  let m := slope Q M,
      b := y_intercept m Q.1 Q.2 in
  m + b = -20 / 3 :=
by
  sorry

end half_area_cutting_line_l407_407926


namespace factors_of_60_l407_407227

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l407_407227


namespace count_valid_48_tuples_l407_407179

open BigOperators

theorem count_valid_48_tuples : 
  ∃ n : ℕ, n = 54 ^ 48 ∧ 
  ( ∃ a : Fin 48 → ℕ, 
    (∀ i : Fin 48, 0 ≤ a i ∧ a i ≤ 100) ∧ 
    (∀ (i j : Fin 48), i < j → a i ≠ a j ∧ a i ≠ a j + 1) 
  ) :=
by
  sorry

end count_valid_48_tuples_l407_407179


namespace number_of_factors_of_60_l407_407200

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l407_407200


namespace num_lattice_points_l407_407780

def is_lattice_point (p : ℤ × ℤ × ℤ) : Prop := 
  let (x, y, z) := p 
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points : 
  {p : ℤ × ℤ × ℤ | is_lattice_point p}.to_finset.card = 18 :=
sorry

end num_lattice_points_l407_407780


namespace llesis_more_rice_l407_407358

theorem llesis_more_rice :
  let total_rice := 50
  let llesis_fraction := 7 / 10
  let llesis_rice := total_rice * llesis_fraction
  let everest_rice := total_rice - llesis_rice
  llesis_rice - everest_rice = 20 := by
    sorry

end llesis_more_rice_l407_407358


namespace number_of_factors_of_60_l407_407204

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l407_407204


namespace symmetric_line_passes_through_fixed_point_l407_407272

theorem symmetric_line_passes_through_fixed_point
  (k : ℝ) :
  let l1 := λ x, k * (x - 4)
  let point_of_symmetry := (2, 1) in
  -- l2 is the symmetric line through the point (2, 1)
  let l2 := λ x, 2 - (l1 (2 - x - 2) + 1) in
  l2 0 = 2 :=
sorry  -- proof omitted

end symmetric_line_passes_through_fixed_point_l407_407272


namespace num_factors_of_60_l407_407197

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l407_407197


namespace interest_rate_and_years_l407_407532

theorem interest_rate_and_years
    (P : ℝ)
    (n : ℕ)
    (e : ℝ)
    (h1 : P * (e ^ n) * e = P * (e ^ (n + 1)) + 4156.02)
    (h2 : P * (e ^ (n - 1)) = P * (e ^ n) - 3996.12) :
    (e = 1.04) ∧ (P = 60000) ∧ (E = 4/100) ∧ (n = 14) := by
  sorry

end interest_rate_and_years_l407_407532


namespace sum_floor_sqrt_1_to_25_l407_407039

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l407_407039


namespace equivalent_functions_l407_407480

theorem equivalent_functions :
  ∀ (x t : ℝ), (x^2 - 2*x - 1 = t^2 - 2*t + 1) := 
by
  intros x t
  sorry

end equivalent_functions_l407_407480


namespace num_factors_60_l407_407216

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l407_407216


namespace common_divisors_and_lcm_60_84_l407_407263

theorem common_divisors_and_lcm_60_84 :
  (∃ (d : ℕ) (hd : d ∣ 60 ∧ d ∣ 84), true) ∧
  (∀ d : ℕ, d ∣ 60 ∧ d ∣ 84 → 
     d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12) ∧
  (nat.gcd 60 84 = 12) :=
by
  sorry

end common_divisors_and_lcm_60_84_l407_407263


namespace find_annual_interest_rate_l407_407089

noncomputable def compound_interest_rate (P : ℝ) (t : ℕ) (CI : ℝ) (n : ℕ) : ℝ :=
  let A := P + CI
  (A / P) ^ (1 / (n * t).toReal) - 1

theorem find_annual_interest_rate :
  let P := 500
  let t := 5
  let CI := 138.14
  let n := 1
  (compound_interest_rate P t CI n) ≈ 0.04985 :=
by
  sorry

end find_annual_interest_rate_l407_407089


namespace domain_of_f_l407_407618

def f (x : ℝ) := 1 / (x + 9) + 1 / (x^2 - 9) + 1 / (x^4 + 9)

theorem domain_of_f : 
  ∀ x : ℝ, (x ∈ (-∞ : Set ℝ, -9) ∪ (-9, -3) ∪ (-3, 3) ∪ (3, ∞)) ↔ 
  (f x = f x) := 
sorry

end domain_of_f_l407_407618


namespace number_of_positive_factors_of_60_l407_407251

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l407_407251


namespace bridge_length_is_correct_l407_407956

-- Train length in meters
def train_length : ℕ := 130

-- Train speed in km/hr
def train_speed_kmh : ℕ := 45

-- Time to cross bridge in seconds
def time_to_cross_bridge : ℕ := 30

-- Conversion factor from km/hr to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh * 1000) / 3600

-- Train speed in m/s
def train_speed_mps := kmh_to_mps train_speed_kmh

-- Total distance covered by the train in 30 seconds
def total_distance := train_speed_mps * time_to_cross_bridge

-- Length of the bridge
def bridge_length := total_distance - train_length

theorem bridge_length_is_correct : bridge_length = 245 := by
  sorry

end bridge_length_is_correct_l407_407956


namespace number_of_factors_of_60_l407_407201

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l407_407201


namespace squared_distance_focus_product_tangents_l407_407671

variable {a b : ℝ}
variable {x0 y0 : ℝ}
variable {P Q R F : ℝ × ℝ}

-- Conditions
def is_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def outside_ellipse (x0 y0 : ℝ) (a b : ℝ) : Prop :=
  (x0^2 / a^2) + (y0^2 / b^2) > 1

-- Question (statement we need to prove)
theorem squared_distance_focus_product_tangents
  (h_ellipse : is_ellipse Q.1 Q.2 a b)
  (h_ellipse' : is_ellipse R.1 R.2 a b)
  (h_outside : outside_ellipse x0 y0 a b)
  (h_a_greater_b : a > b) :
  ‖P - F‖^2 > ‖Q - F‖ * ‖R - F‖ := sorry

end squared_distance_focus_product_tangents_l407_407671


namespace george_purchased_two_large_pizzas_l407_407984

noncomputable def small_slices := 4
noncomputable def large_slices := 8
noncomputable def small_pizzas_purchased := 3
noncomputable def george_slices := 3
noncomputable def bob_slices := george_slices + 1
noncomputable def susie_slices := bob_slices / 2
noncomputable def bill_slices := 3
noncomputable def fred_slices := 3
noncomputable def mark_slices := 3
noncomputable def leftover_slices := 10

noncomputable def total_slices_consumed := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

noncomputable def total_slices_before_eating := total_slices_consumed + leftover_slices

noncomputable def small_pizza_total_slices := small_pizzas_purchased * small_slices

noncomputable def large_pizza_total_slices := total_slices_before_eating - small_pizza_total_slices

noncomputable def large_pizzas_purchased := large_pizza_total_slices / large_slices

theorem george_purchased_two_large_pizzas : large_pizzas_purchased = 2 :=
sorry

end george_purchased_two_large_pizzas_l407_407984


namespace num_factors_of_60_l407_407243

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l407_407243


namespace expected_value_of_c_is_343_over_8_l407_407861

-- Variables representing number of resulting rolls
variables (a b c : ℝ)

-- Definitions of expected value for each step in the problem
def expected_a : ℝ := (1 + 2 + 3 + 4 + 5 + 6) / 6
def expected_b : ℝ := expected_a * (7 / 2)
def expected_c : ℝ := expected_b * (7 / 2)

-- Expected value of c
def expected_value_c : ℝ := (7 / 2) ^ 3

-- Theorem stating the expected value of c is 343/8
theorem expected_value_of_c_is_343_over_8 : expected_value_c = 343 / 8 :=
by
  -- place the proof here
  sorry

end expected_value_of_c_is_343_over_8_l407_407861


namespace binomial_coefficient_30_3_l407_407589

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l407_407589


namespace meeting_time_l407_407971

noncomputable def start_time : ℕ := 13 -- 1 pm in 24-hour format
noncomputable def speed_A : ℕ := 5 -- in kmph
noncomputable def speed_B : ℕ := 7 -- in kmph
noncomputable def initial_distance : ℕ := 24 -- in km

theorem meeting_time : start_time + (initial_distance / (speed_A + speed_B)) = 15 :=
by
  sorry

end meeting_time_l407_407971


namespace floor_sum_sqrt_25_l407_407004

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l407_407004


namespace problem_equiv_proof_l407_407750

noncomputable def prob_alex : ℚ := 3 / 5
noncomputable def prob_chelsea : ℚ := 2 / 15
noncomputable def prob_mel : ℚ := 4 / 15

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def multi_permute (n k l : ℕ) : ℚ :=
  factorial n / (factorial k * factorial l * factorial (n - k - l))

def probability :=
  (prob_alex ^ 4) * (prob_mel ^ 2) * (prob_chelsea) * multi_permute 7 4 2

theorem problem_equiv_proof : 
  ∀ (p_A p_M p_C : ℚ), p_A = 3 / 5 → 
                        p_M = 4 / 15 → 
                        p_C = 2 / 15 → 
                        p_A + p_M + p_C = 1 → 
                        probability = 18144 / 1125 := 
by
  intros p_A p_M p_C h1 h2 h3 h4
  have : probability = 18144 / 1125 := 
    sorry
  exact this

end problem_equiv_proof_l407_407750


namespace chickens_cheaper_than_buying_eggs_after_81_weeks_l407_407385

-- Definitions based on conditions
def cost_chickens (num_chickens : ℕ) (cost_per_chicken : ℕ) : ℕ := num_chickens * cost_per_chicken
def egg_production (num_chickens : ℕ) (eggs_per_chicken_per_week : ℕ) : ℕ := num_chickens * eggs_per_chicken_per_week
def weekly_savings (cost_per_dozen : ℕ) (weekly_feed_cost : ℕ) : ℕ := cost_per_dozen - weekly_feed_cost
def break_even_weeks (total_cost : ℕ) (weekly_savings : ℕ) : ℕ := total_cost / weekly_savings
def cheaper_than_after_weeks (break_even_weeks : ℕ) : ℕ := break_even_weeks + 1

-- Theorem to prove
theorem chickens_cheaper_than_buying_eggs_after_81_weeks :
  ∀ (cost_per_chicken weekly_feed_cost eggs_per_chicken_per_week cost_per_dozen num_chickens : ℕ),
  cost_per_chicken = 20 →
  weekly_feed_cost = 1 →
  eggs_per_chicken_per_week = 3 →
  cost_per_dozen = 2 →
  num_chickens = 4 →
  let total_cost := cost_chickens num_chickens cost_per_chicken,
      weekly_savings_amt := weekly_savings cost_per_dozen weekly_feed_cost,
      break_even := break_even_weeks total_cost weekly_savings_amt,
      weeks_needed := cheaper_than_after_weeks break_even
  in weeks_needed = 81 :=
begin
  intros,
  sorry
end

end chickens_cheaper_than_buying_eggs_after_81_weeks_l407_407385


namespace cost_of_pen_l407_407862

theorem cost_of_pen 
  (total_amount_spent : ℕ)
  (total_items : ℕ)
  (number_of_pencils : ℕ)
  (cost_of_pencil : ℕ)
  (cost_of_pen : ℕ)
  (h1 : total_amount_spent = 2000)
  (h2 : total_items = 36)
  (h3 : number_of_pencils = 16)
  (h4 : cost_of_pencil = 25)
  (remaining_amount_spent : ℕ)
  (number_of_pens : ℕ)
  (h5 : remaining_amount_spent = total_amount_spent - (number_of_pencils * cost_of_pencil))
  (h6 : number_of_pens = total_items - number_of_pencils)
  (total_cost_of_pens : ℕ)
  (h7 : total_cost_of_pens = remaining_amount_spent)
  (h8 : total_cost_of_pens = number_of_pens * cost_of_pen)
  : cost_of_pen = 80 := by
  sorry

end cost_of_pen_l407_407862


namespace N_is_square_of_a_defined_value_l407_407829

theorem N_is_square_of_a_defined_value : 
  ∃ x : ℤ, x = 24^2 + 3 * 24 + 1 ∧ 24 * 25 * 26 * 27 + 1 = x^2 :=
by
  existsi (24^2 + 3 * 24 + 1)
  split
  sorry

end N_is_square_of_a_defined_value_l407_407829


namespace num_lattice_points_l407_407779

def is_lattice_point (p : ℤ × ℤ × ℤ) : Prop := 
  let (x, y, z) := p 
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points : 
  {p : ℤ × ℤ × ℤ | is_lattice_point p}.to_finset.card = 18 :=
sorry

end num_lattice_points_l407_407779


namespace ratio_of_triangle_areas_l407_407923

noncomputable def area (a b c : ℕ) : ℚ := if a * a + b * b = c * c then (a * b : ℚ) / 2 else 0

theorem ratio_of_triangle_areas :
  let PQR := (7, 24, 25)
  let STU := (9, 40, 41)
  area PQR.1 PQR.2 PQR.3 / area STU.1 STU.2 STU.3 = (7 / 15 : ℚ) :=
by
  sorry

end ratio_of_triangle_areas_l407_407923


namespace greatest_int_satisfying_inequality_l407_407450

theorem greatest_int_satisfying_inequality : 
  ∃ m : ℤ, (∀ x : ℤ, x - 5 > 4 * x - 1 → x ≤ -2) ∧ (∀ k : ℤ, k < -2 → k - 5 > 4 * k - 1) :=
by
  sorry

end greatest_int_satisfying_inequality_l407_407450


namespace max_value_of_x_sq_plus_y_sq_l407_407717

theorem max_value_of_x_sq_plus_y_sq (x y : ℝ) 
  (h : x^2 + y^2 + 4 * x - 2 * y - 4 = 0) : 
  ∃ M, M = 14 + 6 * Real.sqrt 5 ∧ ∀ (x y : ℝ), x^2 + y^2 + 4 * x - 2 * y - 4 = 0 → x^2 + y^2 ≤ M :=
sorry

end max_value_of_x_sq_plus_y_sq_l407_407717


namespace bug_closest_point_after_fourth_move_l407_407509

-- Define the initial position and the moves
def initial_position : ℝ × ℝ := (0, 0)
def first_move : ℝ × ℝ := (2, 0)
def second_move : ℝ × ℝ := (2, 1)
def third_move : ℝ × ℝ := (1.5, 1)
def fourth_move : ℝ × ℝ := (1.5, 0.75)

-- Define the closest point to be proven after the fourth move
def closest_point : ℝ × ℝ := (7 / 4, 7 / 8)

-- Prove that the fourth move results in the closest point being (7/4, 7/8)
theorem bug_closest_point_after_fourth_move :
  distance fourth_move closest_point ≤ distance fourth_move (3/2, 5/4) ∧
  distance fourth_move closest_point ≤ distance fourth_move (2, 1) ∧
  distance fourth_move closest_point ≤ distance fourth_move (5/2, 3/2) :=
sorry

end bug_closest_point_after_fourth_move_l407_407509


namespace john_sleep_hours_l407_407318

theorem john_sleep_hours (s1 p1 p2 : ℝ) (h : p1 * s1 = p2 * (480 / 90)) : 
    p1 = 80 → s1 = 6 → p2 = 90 → s2 = 480 / 90 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at h
  exact h

end john_sleep_hours_l407_407318


namespace Geli_pushups_total_l407_407676

-- Define conditions
def first_day_pushups : ℕ := 10
def additional_pushups : ℕ := 5
def total_days : ℕ := 3

-- Define pushups for each day
def pushups_day (n : ℕ) : ℕ :=
  first_day_pushups + (n - 1) * additional_pushups

-- Define the sum of pushups over the week
def total_pushups_week : ℕ := 
  (List.range total_days).sum (λ d, pushups_day (d + 1))

-- State the proof problem
theorem Geli_pushups_total : total_pushups_week = 45 := 
sorry

end Geli_pushups_total_l407_407676


namespace lattice_points_count_l407_407784

theorem lattice_points_count :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2^2 + p.3^2 = 25}.to_finset.card = 30 :=
sorry

end lattice_points_count_l407_407784


namespace find_a_l407_407414

-- Conditions
variables {y₀ : ℝ} {a : ℝ}
def P := (3/2 : ℝ, y₀)
def parabola_eq (a : ℝ) := ∀ (y₀ : ℝ), y₀^2 = a * (3/2)

-- Focus of the parabola y^2 = ax
def focus (a : ℝ) := (a / 4, 0)

-- Distance from point P to the focus F is 2
def distance_P_F_is_2 (a : ℝ) : Prop :=
  let F : ℝ × ℝ := focus a in
  real.dist (fst P) (fst F) + abs (3/2 + (a / 4)) = 2

-- Proof problem
theorem find_a (a_pos : 0 < a) (h₀ : parabola_eq a y₀) (h₁ : distance_P_F_is_2 a) : a = 2 :=
sorry

end find_a_l407_407414


namespace cannot_be_one_last_l407_407367

theorem cannot_be_one_last (L : List ℕ) (h_init : L = List.range' 1 100)
    (h_op : ∀ a b c : ℕ, c = |a - b| → List.erase (List.erase L a) b = List.cons c (List.erase (List.erase L a) b))
    (h_steps : ∀ (i : ℕ), i < 99 → ∃ a b : ℕ, a ∈ L ∧ b ∈ L)
    : ¬((L.length = 1) ∧ (L.head = 1)) :=
sorry

end cannot_be_one_last_l407_407367


namespace total_tickets_sold_l407_407526

def price_adult_ticket : ℕ := 7
def price_child_ticket : ℕ := 4
def total_revenue : ℕ := 5100
def adult_tickets_sold : ℕ := 500

theorem total_tickets_sold : 
  ∃ (child_tickets_sold : ℕ), 
    price_adult_ticket * adult_tickets_sold + price_child_ticket * child_tickets_sold = total_revenue ∧
    adult_tickets_sold + child_tickets_sold = 900 :=
by
  sorry

end total_tickets_sold_l407_407526


namespace probability_S_together_l407_407499

theorem probability_S_together (n : ℕ) (h1 : n = 11) (hS : 4 ≤ n) (hI : 2 ≤ n) (p : ℕ → ℕ := λ m, m.factorial) :
  (∃ total favorable, total = (p n) / (p 4 * p 2) ∧ favorable = (p 8) / (p 2) ∧ 
     (favorable : ℚ) / (total : ℚ) = 4 / 165) :=
by
  let discus := "DISCUSSIONS"
  have h_length : discus.length = 11 := sorry
  have h_count_S : discus.count 'S' = 4 := sorry
  have h_count_I : discus.count 'I' = 2 := sorry
  let total := (p discus.length) / (p 4 * p 2)
  let favorable := (p 8) / (p 2)
  use total, favorable
  split
  · sorry
  · split
    · sorry
    · exact (favorable : ℚ) / (total : ℚ) = 4 / 165 한 sorry

end probability_S_together_l407_407499


namespace binomial_30_3_l407_407581

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l407_407581


namespace no_good_subset_with_405_elements_l407_407322

-- Define the set A
def A : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2017 }

-- Define a *good* subset S
def good (S : Set ℕ) : Prop := ∀ x ∈ S, (∑ y in S \ {x}, y) % 10 = x % 10

-- State the main theorem
theorem no_good_subset_with_405_elements (S : Set ℕ) (hA : S ⊆ A) (hS : S.card = 405) (hgood : good S) : False := 
sorry

end no_good_subset_with_405_elements_l407_407322


namespace fourth_graders_pizza_l407_407512

theorem fourth_graders_pizza :
  ∃ n : ℕ, (n * 20 * 6 = 1200) ∧ n = 10 :=
begin
  sorry
end

end fourth_graders_pizza_l407_407512


namespace sum_of_kowai_numbers_l407_407531

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_kowai_number (q : ℕ) : Prop :=
  ∃ p : ℕ, is_prime q ∧ is_prime p ∧ is_prime (p^2 - 2) ∧ is_prime (p^2 - 8) ∧ is_prime (p^3 + 6) ∧ q = p^2 + 10

theorem sum_of_kowai_numbers : ∑ q in {q : ℕ | is_kowai_number q}, q = 59 :=
by sorry

end sum_of_kowai_numbers_l407_407531


namespace product_of_intersection_coordinates_l407_407460

-- Define circle equations
def circle1_eq (x y : ℝ) : Prop := (x - 2) * (x - 2) + (y - 3) * (y - 3) = 1
def circle2_eq (x y : ℝ) : Prop := (x - 4) * (x - 4) + (y - 6) * (y - 6) = 4

-- Define the proof statement
theorem product_of_intersection_coordinates : 
  (circle1_eq x y ∧ circle2_eq x y → False) → True ∧ 0 = 0 :=
by 
  intro h
  triv
  trivial
  sorry

end product_of_intersection_coordinates_l407_407460


namespace binomial_30_3_l407_407585

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l407_407585


namespace turtle_hatchlings_l407_407626

theorem turtle_hatchlings (num_turtles : ℕ) (eggs_per_turtle : ℕ) (hatch_rate : ℝ) :
  (num_turtles = 6 ∧ eggs_per_turtle = 20 ∧ hatch_rate = 0.4) →
  (num_turtles * (eggs_per_turtle * hatch_rate).toNat = 48) :=
by
  intros h;
  cases h with h1 h2;
  cases h2 with h3 h4;
  rw [h1, h3];
  have hatchlings_per_turtle : ℕ := (eggs_per_turtle * hatch_rate).toNat;
  rw [← h3];
  exact sorry

end turtle_hatchlings_l407_407626


namespace number_of_positive_factors_of_60_l407_407252

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l407_407252


namespace equivalent_functions_l407_407479

theorem equivalent_functions :
  ∀ (x t : ℝ), (x^2 - 2*x - 1 = t^2 - 2*t + 1) := 
by
  intros x t
  sorry

end equivalent_functions_l407_407479


namespace tangent_eqn_at_1_intersection_points_eqn_l407_407151

noncomputable def f (x : ℝ) : ℝ := -x^2 + 8 * x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 6 * real.log x + m

theorem tangent_eqn_at_1 : 
  ∃ (m : ℝ), ∃ c, ∀ x, f(x) = 6 * x + c := 
sorry

theorem intersection_points_eqn : 
  ∃ m, 7 < m ∧ m < 15 - 6 * real.log 3 ∧ 
    ∃ x1 x2 x3, (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ∧ 
                g(x1) m = f(x1) ∧ 
                g(x2) m = f(x2) ∧ 
                g(x3) m = f(x3) := 
sorry

end tangent_eqn_at_1_intersection_points_eqn_l407_407151


namespace binomial_30_3_l407_407576

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l407_407576


namespace ratio_of_sums_l407_407131

open Nat

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 8 - 2 * a 3) / 7)

def arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem ratio_of_sums
    (a : ℕ → ℝ)
    (S : ℕ → ℝ)
    (a_arith : arithmetic_sequence_property a 1)
    (s_def : ∀ n, S n = sum_of_first_n_terms a n)
    (a8_eq_2a3 : a 8 = 2 * a 3) :
  S 15 / S 5 = 6 :=
sorry

end ratio_of_sums_l407_407131


namespace mean_score_74_l407_407660

theorem mean_score_74 
  (M SD : ℝ)
  (h1 : 58 = M - 2 * SD)
  (h2 : 98 = M + 3 * SD) : 
  M = 74 :=
by
  sorry

end mean_score_74_l407_407660


namespace floor_sqrt_sum_l407_407041

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l407_407041


namespace inconsistent_intercepts_l407_407716

-- Define the ellipse equation
def ellipse (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m + y^2 / 4 = 1

-- Define the line equations
def line1 (x k : ℝ) : ℝ := k * x + 1
def line2 (x : ℝ) (k : ℝ) : ℝ := - k * x - 2

-- Disc calculation for line1
def disc1 (m k : ℝ) : ℝ :=
  let a := 4 + m * k^2
  let b := 2 * m * k
  let c := -3 * m
  b^2 - 4 * a * c

-- Disc calculation for line2
def disc2 (m k : ℝ) : ℝ :=
  let bb := 4 * m * k
  bb^2

-- Statement of the problem
theorem inconsistent_intercepts (m k : ℝ) (hm_pos : 0 < m) :
  disc1 m k ≠ disc2 m k :=
by
  sorry

end inconsistent_intercepts_l407_407716


namespace initial_percentage_increase_l407_407868

variable (S : ℝ) (P : ℝ)

theorem initial_percentage_increase :
  (S + (P / 100) * S) - 0.10 * (S + (P / 100) * S) = S + 0.15 * S →
  P = 16.67 :=
by
  sorry

end initial_percentage_increase_l407_407868


namespace ivan_reaches_koschei_kingdom_b_l407_407309

noncomputable def total_distance (d: ℝ) :=
  let day_one := d / 3
  let day_two := d / 6
  let day_four := 100
  day_one + day_two + day_four

theorem ivan_reaches_koschei_kingdom_b (d: ℝ) (travel_per_day: ℝ):
  (total_distance d = d) ∧ (travel_per_day = 100) →
  (∀ n, n <= 1 → n * travel_per_day = 100) →
  d = 200 → travel_per_day = 100 → Ivan_Reaches := true
Proof
  sorry

end ivan_reaches_koschei_kingdom_b_l407_407309


namespace least_positive_period_f_find_lambda_l407_407720

-- Definitions for the given conditions
def f (x : ℝ) : ℝ := sin (5 * Real.pi / 6 - 2 * x) - 2 * sin (x - Real.pi / 4) * cos (x + 3 * Real.pi / 4)
def F (x : ℝ) (λ : ℝ) : ℝ := -4 * λ * f x - cos (4 * x - Real.pi / 3)

-- Prove the least positive period of f(x) is π
theorem least_positive_period_f : ∀ x : ℝ, f (x + Real.pi) = f x := sorry

-- Given F(x) and conditions, prove the value of λ
theorem find_lambda (x : ℝ) (h₁ : x ∈ Set.Icc (Real.pi / 12) (Real.pi / 3))
  (h₂ : ∀ x, ∃ x_min ∈ Set.Icc (Real.pi / 12) (Real.pi / 3), F x_min λ = -3 / 2) :
  λ = 1 / 4 := sorry

end least_positive_period_f_find_lambda_l407_407720


namespace train_speed_in_kmph_l407_407538

-- Define the conditions and the result
def train_length : ℝ := 165
def bridge_length : ℝ := 660
def total_time : ℝ := 54.995600351971845
def total_distance : ℝ := train_length + bridge_length
def speed_mps := total_distance / total_time
def speed_kmph := speed_mps * 3.6

-- The theorem to prove
theorem train_speed_in_kmph : speed_kmph ≈ 54.0036 := by
  sorry

end train_speed_in_kmph_l407_407538


namespace sum_of_first_n_terms_l407_407845

variable {α : Type*} [linear_ordered_field α]

def sum_of_arithmetic_sequence (a : ℕ → α) (n : ℕ) : α :=
  n * (a 1 + a n)/ 2

def is_geometric (a : α) (b : α) (c : α) : Prop :=
  b^2 = a * c

noncomputable def Sn (n : ℕ) := 
  if n = 3 then 12 else sorry

theorem sum_of_first_n_terms (a : ℕ → α) (n : ℕ) (h1 : Sn 3 = 12)
  (h2 : is_geometric (2 * a 1) (a 2) (a 3 + 1)) :
  sum_of_arithmetic_sequence a n = (1/2) * n * (3 * n - 1) 
  ∨ sum_of_arithmetic_sequence a n = 2 * n * (5 - n) :=
sorry

end sum_of_first_n_terms_l407_407845


namespace lattice_points_count_l407_407783

theorem lattice_points_count :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2^2 + p.3^2 = 25}.to_finset.card = 30 :=
sorry

end lattice_points_count_l407_407783


namespace floor_sum_sqrt_1_to_25_l407_407062

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l407_407062


namespace total_stars_l407_407988

theorem total_stars (g s : ℕ) (hg : g = 10^11) (hs : s = 10^11) : g * s = 10^22 :=
by
  rw [hg, hs]
  sorry

end total_stars_l407_407988


namespace outfits_count_l407_407949

def num_red_shirts : ℕ := 5
def num_green_shirts : ℕ := 5
def num_blue_shirts : ℕ := 5
def num_pants : ℕ := 6
def num_green_hats : ℕ := 8
def num_red_hats : ℕ := 8

theorem outfits_count :
  let case1 := num_red_shirts * num_pants * num_green_hats,
      case2 := num_green_shirts * num_pants * num_red_hats,
      case3 := num_blue_shirts * num_pants * (num_green_hats + num_red_hats)
  in case1 + case2 + case3 = 960 :=
by
  sorry

end outfits_count_l407_407949


namespace fourth_term_geometric_sequence_l407_407407

theorem fourth_term_geometric_sequence :
  let a1 := real.sqrt 3
  let a2 := real.exp (real.log 3 / 4)
  let a3 := real.exp (real.log 3 / 12)
  exists a4 : ℝ, a4 = real.exp (real.log 3 * (-1 / 6)) :=
by
  let a1 := real.sqrt 3
  let a2 := real.exp (real.log 3 / 4)
  let a3 := real.exp (real.log 3 / 12)
  use real.exp (real.log 3 * (-1 / 6))
  sorry

end fourth_term_geometric_sequence_l407_407407


namespace floor_sum_sqrt_1_to_25_l407_407064

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l407_407064


namespace constant_function_of_conditions_l407_407336

open Set

theorem constant_function_of_conditions {f : ℝ → ℝ}
  (h1 : ∀ x, x ∈ (0:ℝ, 1) → f x > 0)
  (h2 : ∀ x y, x ∈ (0:ℝ, 1) → y ∈ (0:ℝ, 1) → (f x / f y) + (f (1 - x) / f (1 - y)) ≤ 2) :
  ∃ c : ℝ, ∀ x, x ∈ (0:ℝ, 1) → f x = c :=
sorry

end constant_function_of_conditions_l407_407336


namespace num_factors_of_60_l407_407244

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l407_407244


namespace problem_inequality_l407_407834

noncomputable def f : ℕ+ → ℕ+ := sorry

theorem problem_inequality (k : ℕ+) (h1 : ∀ n: ℕ+, f (f n) = k * n) (h2 : ∀ n m : ℕ+, n < m → f n < f m) :
  ∀ n : ℕ+, (2 * k / (k + 1) * n : ℚ) ≤ f n ∧ f n ≤ (k + 1) / 2 * n :=
begin
  sorry
end

end problem_inequality_l407_407834


namespace parallel_implies_not_intersect_2D_not_intersect_implies_parallel_2D_parallel_implies_not_intersect_3D_not_intersect_does_not_imply_parallel_3D_l407_407294

-- Definitions for lines in 2D and 3D spaces
structure Line2D (α : Type*) :=
(point : α)
(direction : α)

structure Line3D (α : Type*) :=
(point : α)
(direction : α)

-- Parallel and intersection definitions in 2D
def parallel_2D {α : Type*} [field α] (L1 L2 : Line2D α) := -- Lines are parallel in 2D
    L1.direction = L2.direction

def intersect_2D {α : Type*} [field α] (L1 L2 : Line2D α) := -- Lines intersect in 2D
    ∃ p : α, L1.point = p ∧ L2.point = p

-- Parallel and intersection definitions in 3D
def parallel_3D {α : Type*} [field α] (L1 L2 : Line3D α) := -- Lines are parallel in 3D
    L1.direction = L2.direction

def intersect_3D {α : Type*} [field α] (L1 L2 : Line3D α) := -- Lines intersect in 3D
    ∃ p : α, L1.point = p ∧ L2.point = p

-- Skew definition in 3D
def skew_3D {α : Type*} [field α] (L1 L2 : Line3D α) := -- Lines are skew in 3D
    ¬ parallel_3D L1 L2 ∧ ¬ intersect_3D L1 L2

-- Proof statements for 2D
theorem parallel_implies_not_intersect_2D {α : Type*} [field α] (L1 L2 : Line2D α) :
    parallel_2D L1 L2 → ¬ intersect_2D L1 L2 := sorry

theorem not_intersect_implies_parallel_2D {α : Type*} [field α] (L1 L2 : Line2D α) :
    ¬ intersect_2D L1 L2 → parallel_2D L1 L2 := sorry

-- Proof statements for 3D
theorem parallel_implies_not_intersect_3D {α : Type*} [field α] (L1 L2 : Line3D α) :
    parallel_3D L1 L2 → ¬ intersect_3D L1 L2 := sorry

theorem not_intersect_does_not_imply_parallel_3D {α : Type*} [field α] (L1 L2 : Line3D α) :
    ¬ intersect_3D L1 L2 → (parallel_3D L1 L2 ∨ skew_3D L1 L2) := sorry

end parallel_implies_not_intersect_2D_not_intersect_implies_parallel_2D_parallel_implies_not_intersect_3D_not_intersect_does_not_imply_parallel_3D_l407_407294


namespace find_smallest_n_l407_407433

noncomputable def Q (n : ℕ) : ℝ :=
  let term k := (↑k^2) / (3 + k^2 : ℝ)
  ( ∏ i in Finset.range (n - 1), term (i + 1) ) * (3 / (3 + n^2 : ℝ))

theorem find_smallest_n : ∃ n, Q n < 1 / 5000 ∧ ∀ m < n, Q m ≥ 1 / 5000 := by
  sorry

end find_smallest_n_l407_407433


namespace volume_ratio_of_truncated_pyramid_l407_407293

-- Define the conditions of the truncated triangular pyramid
structure TruncatedTriangularPyramid where
  side_ratio : ℕ → ℕ → Prop
  volume_division : ℕ → ℕ

-- State the theorem based on the given problem
theorem volume_ratio_of_truncated_pyramid (P : TruncatedTriangularPyramid) 
  (h1 : P.side_ratio 1 2) 
  (h2 : True) : P.volume_division 3 4 :=
sorry

end volume_ratio_of_truncated_pyramid_l407_407293


namespace trajectory_and_angle_property_l407_407168

noncomputable def point := ℝ × ℝ

def is_tangent (line parabola: ℝ → ℝ) : Prop :=
  ∀ x, line x = parabola x

def parabola (x: ℝ) := x^2 / 4
def line (m x: ℝ) := x + m

def ellipse (x y: ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1 ∧ y ≠ 0

def line_eq (t x: ℝ) := (1 / 2) * x + t

def intersects_ellipse (t: ℝ) (x y: ℝ) : Prop :=
  line_eq t x = y ∧ ellipse x y

def perpendicular (p1 p2 p: point) : Prop :=
  let k1 := (p.snd - p1.snd) / (p.fst - p1.fst) in
  let k2 := (p2.snd - p1.snd) / (p2.fst - p1.fst) in
  k1 * k2 = -1

def angle_eq (a p n b: point) : Prop :=
  ∃ k1 k2, k1 = (a.snd - p.snd) / (a.fst - p.fst) ∧ k2 = (b.snd - p.snd) / (b.fst - p.fst) ∧ k1 + k2 = 0

theorem trajectory_and_angle_property:
  ∀ (m: ℝ) (M N P: point) (C: point → Prop),
  is_tangent (λ x, line m x) parabola →
  M = (1, 0) →
  N = (-1, 0) →
  |M.fst - P.fst| + |M.snd - P.snd| + |N.fst - P.fst| + |N.snd - P.snd| = 6 →
  (C = ellipse) →
  ∀ (t: ℝ), t ≠ ±1 →
  ∀ (A B: point),
  intersects_ellipse t A.fst A.snd →
  intersects_ellipse t B.fst B.snd →
  perpendicular N M (1, 1.5) →
  angle_eq A P N B :=
by sorry

end trajectory_and_angle_property_l407_407168


namespace hyperbola_eccentricity_is_sqrt2_l407_407979

noncomputable def hyperbola_eccentricity : ℝ :=
  let a := 1 in -- without loss of generality, we can assume a = 1
  let b := a in
  let c := real.sqrt (a^2 + b^2) in
  c / a

theorem hyperbola_eccentricity_is_sqrt2 :
  hyperbola_eccentricity = real.sqrt 2 :=
by
  -- define the values of a, b, and c as per the conditions.
  let a := 1
  let b := a
  let c := real.sqrt (a^2 + b^2)
  -- show that the hyperbola's eccentricity is sqrt 2
  show c / a = real.sqrt 2
  -- replace c with sqrt(2 * a ^ 2) = sqrt(2)
  sorry

end hyperbola_eccentricity_is_sqrt2_l407_407979


namespace find_coordinates_of_M_l407_407374

def point_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.2) = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs (P.1) = d

theorem find_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_second_quadrant M ∧ distance_to_x_axis M 5 ∧ distance_to_y_axis M 3 ∧ M = (-3, 5) :=
by
  sorry

end find_coordinates_of_M_l407_407374


namespace quadratic_eq_with_roots_l407_407140

theorem quadratic_eq_with_roots (x y : ℝ) (h : (x^2 - 6 * x + 9) = -|y - 1|) : 
  ∃ a : ℝ, (a^2 - 4 * a + 3 = 0) :=
by 
  sorry

end quadratic_eq_with_roots_l407_407140


namespace binom_30_3_eq_4060_l407_407604

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l407_407604


namespace original_loaf_had_27_slices_l407_407635

def original_slices : ℕ :=
  let slices_andy_ate := 3 * 2
  let slices_for_toast := 2 * 10
  let slices_left := 1
  slices_andy_ate + slices_for_toast + slices_left

theorem original_loaf_had_27_slices (n : ℕ) (slices_andy_ate : ℕ) (slices_for_toast : ℕ) (slices_left : ℕ) :
  slices_andy_ate = 6 → slices_for_toast = 20 → slices_left = 1 → n = slices_andy_ate + slices_for_toast + slices_left → n = 27 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

-- Verifying the statement
example : original_slices = 27 := by
  have h1 : 3 * 2 = 6 := rfl
  have h2 : 2 * 10 = 20 := rfl
  have h3 : 1 = 1 := rfl
  exact original_loaf_had_27_slices original_slices 6 20 1 h1 h2 h3 rfl

end original_loaf_had_27_slices_l407_407635


namespace necessary_but_not_sufficient_condition_l407_407964

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2) → (∃ x : ℂ, x^2 + (a : ℂ) * x + 1 = 0 ∧ x.im ≠ 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l407_407964


namespace not_divisible_59_l407_407361

theorem not_divisible_59 (x y : ℕ) (hx : ¬ (59 ∣ x)) (hy : ¬ (59 ∣ y)) 
  (h : (3 * x + 28 * y) % 59 = 0) : (5 * x + 16 * y) % 59 ≠ 0 :=
by
  sorry

end not_divisible_59_l407_407361


namespace pairs_divisible_by_three_l407_407283

theorem pairs_divisible_by_three (P T : ℕ) (h : 5 * P = 3 * T) : ∃ k : ℕ, P = 3 * k := 
sorry

end pairs_divisible_by_three_l407_407283


namespace jill_and_bob_payment_l407_407316

-- Definitions of the conditions
def price_of_first_house (X : ℝ) := X
def price_of_second_house (Y X : ℝ) := 2 * X

theorem jill_and_bob_payment :
  ∃ X, ∃ Y, Y = 2 * X ∧ X + Y = 600000 ∧ X = 200000 :=
by
  sorry

end jill_and_bob_payment_l407_407316


namespace range_upper_limit_of_D_l407_407870

-- Definitions based on conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def D : Set ℕ := { p | 10 < p ∧ is_prime p }

-- Condition 1: D is a set of prime numbers greater than 10
-- Condition 2: The range of D is 12
axiom range_D : Set ℕ → ℕ
axiom range_D_eq_12 : range_D D = 12

-- Proof Problem: Prove the upper limit of D is 23.
theorem range_upper_limit_of_D : (upper_limit : ℕ) → upper_limit ∈ D → ∃ l : ℕ, l = 23 :=
begin
  sorry
end

end range_upper_limit_of_D_l407_407870


namespace transformation_matrix_non_singular_l407_407128

def transformation_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![ (-1:ℝ), 0, 0],
    ![ 0, 1, 0],
    ![ 0, 0, 1] ]

theorem transformation_matrix_non_singular : det transformation_matrix ≠ 0 := 
by
  sorry

end transformation_matrix_non_singular_l407_407128


namespace rectangle_perimeter_l407_407950

theorem rectangle_perimeter (x y m n : ℕ) (h1 : ∀ (A B C D E F A' : Point), is_rectangle A B C D → 
  points_on_line_segment E A B → points_on_line_segment F C D → fold_creates_point A A' D → 
  (AE = 3 ∧ EB = 15 ∧ CF = 8) → (AB = 18 ∧ CD = 23 ∧ Perimeter = 82)) :
  m = 82 → n = 1 → m + n = 83 := 
sorry

end rectangle_perimeter_l407_407950


namespace complex_number_solution_l407_407147

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i^2 = -1) 
  (h : -i * z = (3 + 2 * i) * (1 - i)) : z = 1 + 5 * i :=
by
  sorry

end complex_number_solution_l407_407147


namespace proof_cos_2x_cos_2y_l407_407703

variable {θ x y : ℝ}

-- Conditions
def is_arith_seq (a b c : ℝ) := b = (a + c) / 2
def is_geom_seq (a b c : ℝ) := b^2 = a * c

-- Proving the given statement with the provided conditions
theorem proof_cos_2x_cos_2y (h_arith : is_arith_seq (Real.sin θ) (Real.sin x) (Real.cos θ))
                            (h_geom : is_geom_seq (Real.sin θ) (Real.sin y) (Real.cos θ)) :
  2 * Real.cos (2 * x) = Real.cos (2 * y) :=
sorry

end proof_cos_2x_cos_2y_l407_407703


namespace range_of_a_l407_407719

def f (x : ℝ) (a : ℝ) := log x - a * (x - 1) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (x + 1) * log x + 2 * a < (x - 1) * log x * (x + 1)) →
  a ∈ set.Iic 2 :=
by
  intro h
  -- Proof of the theorem goes here
  sorry

end range_of_a_l407_407719


namespace apple_weight_probability_l407_407946

theorem apple_weight_probability (P_lt_200 P_gt_300 : ℝ) (h1 : P_lt_200 = 0.10) (h2 : P_gt_300 = 0.12) :
  ∃ P_200_300 : ℝ, P_200_300 = 1 - P_lt_200 - P_gt_300 ∧ P_200_300 = 0.78 :=
by
  use 1 - P_lt_200 - P_gt_300
  split
  · rfl
  · rw [h1, h2]
    norm_num

end apple_weight_probability_l407_407946


namespace cistern_leak_empty_time_l407_407951

noncomputable def leak_time_to_empty (R L : ℝ) (H1 : R = 1 / 6) (H2 : R - L = 1 / 7) : ℝ :=
  1 / L

theorem cistern_leak_empty_time :
  ∀ (R L : ℝ), 
  (R = 1 / 6) → 
  (R - L = 1 / 7) → 
  (leak_time_to_empty R L (by assumption) (by assumption) = 42) :=
by intros R L H1 H2
   sorry

end cistern_leak_empty_time_l407_407951


namespace suitable_squares_count_l407_407749

-- Define the grid
def grid : Type := array (Fin 5 × Fin 5) bool

-- Define what it means for a square to have an equal number of black and white cells
def equal_black_white_cells (g : grid) (x y : Fin 5) (size : Nat) : Prop :=
  size % 2 = 0 ∧
  let num_black := ∑ i in (range size), ∑ j in (range size), if g ⟨x + i, y + j⟩ then 1 else 0 in
  let num_white := size * size - num_black in
  num_black = num_white

-- Define how to count all suitable squares in the grid
def count_suitable_squares (g : grid) : Nat :=
  let f := λ x y size, if equal_black_white_cells g x y size then 1 else 0 in
  ∑ x in (range 4), ∑ y in (range 4), f x y 2 + ∑ x in (range 2), ∑ y in (range 2), f x y 4

-- The theorem to be proven
theorem suitable_squares_count (g : grid) : count_suitable_squares g = 16 :=
sorry

end suitable_squares_count_l407_407749


namespace reciprocal_of_mixed_number_l407_407907

def mixed_number := -1 - (4 / 5)

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_mixed_number : reciprocal mixed_number = -5 / 9 := 
by
  sorry

end reciprocal_of_mixed_number_l407_407907


namespace a8_value_l407_407533

def sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧ ∀ n, (∑ k in finset.range n, a k + a (k+1)) = 2 ^ (n+1) - 2

theorem a8_value (a : ℕ → ℕ) (h : sequence a) : a 8 = 85 :=
sorry

end a8_value_l407_407533


namespace find_a_b_l407_407640

noncomputable def abs (x : ℝ) := if x >= 0 then x else -x

theorem find_a_b (a b : ℝ) :
    (abs a + abs b ≥ 2 / Real.sqrt 3) ∧ (∀ x, abs (a * Real.sin x + b * Real.sin (2 * x)) ≤ 1) →
    (a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
    (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
    (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
    (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) :=
sorry

end find_a_b_l407_407640


namespace right_triangle_bisector_inequality_l407_407378

theorem right_triangle_bisector_inequality (A B C D A' B' : Point)
  (h_right_angle : ∠ A C B = 90)
  (h_cd_bisector : CD_is_bisector C D (triangle A B C))
  (h_projections : A' = projection A C D ∧ B' = projection B C D)
  (h_inequality : AC ≥ BC) :
  CD ≤ (1 / 2) * (distance A A' + distance B B') :=
sorry

end right_triangle_bisector_inequality_l407_407378


namespace number_of_factors_60_l407_407220

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l407_407220


namespace sum_of_floors_of_square_roots_l407_407021

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l407_407021


namespace max_consecutive_integers_sum_500_l407_407453

theorem max_consecutive_integers_sum_500 : ∀ S k max_n,
  (∀ (n : ℕ), S n = n * k + n * (n - 1) / 2) →
  (k = 3) →
  (∀ (n : ℕ), 2 * S n ≤ 1000) →
  max_n = 29 :=
by
  intros S k max_n S_def hk hineq
  sorry

end max_consecutive_integers_sum_500_l407_407453


namespace binom_30_3_eq_4060_l407_407572

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l407_407572


namespace sum_of_floors_of_square_roots_l407_407025

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l407_407025


namespace davids_biology_marks_l407_407069

theorem davids_biology_marks (english mathematics physics chemistry : ℕ)
    (average : ℤ) (total_subjects : ℕ)
    (h_english : english = 96)
    (h_mathematics : mathematics = 95)
    (h_physics : physics = 82)
    (h_chemistry : chemistry = 97)
    (h_average : average = 93)
    (h_total_subjects : total_subjects = 5) :
    ∃ biology, biology = 95 := 
by
  -- Define the total marks obtained in the known subjects
  let total_marks_known := english + mathematics + physics + chemistry
  -- Define the total marks for all subjects given the average
  let total_marks_all := average * total_subjects
  -- Obtain the biology marks by subtracting known marks from total marks
  let biology := total_marks_all - total_marks_known
  -- Assertion of the solution
  have : total_marks_known = 96 + 95 + 82 + 97 := by sorry
  have : total_marks_all = 93 * 5 := by sorry
  have : biology = (93 * 5) - (96 + 95 + 82 + 97) := by sorry
  use 95
  exact this

end davids_biology_marks_l407_407069


namespace binom_30_3_l407_407611

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l407_407611


namespace max_consecutive_integers_sum_500_l407_407451

theorem max_consecutive_integers_sum_500 : ∀ S k max_n,
  (∀ (n : ℕ), S n = n * k + n * (n - 1) / 2) →
  (k = 3) →
  (∀ (n : ℕ), 2 * S n ≤ 1000) →
  max_n = 29 :=
by
  intros S k max_n S_def hk hineq
  sorry

end max_consecutive_integers_sum_500_l407_407451


namespace polynomial_factor_l407_407900

def factorization_condition (p q : ℤ) : Prop :=
  ∃ r s : ℤ, 
    p = 4 * r ∧ 
    q = -3 * r + 4 * s ∧ 
    40 = 2 * r - 3 * s + 16 ∧ 
    -20 = s - 12

theorem polynomial_factor (p q : ℤ) (hpq : factorization_condition p q) : (p, q) = (0, -32) :=
by sorry

end polynomial_factor_l407_407900


namespace binom_30_3_eq_4060_l407_407605

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l407_407605


namespace count_paths_from_0_0_to_3_3_l407_407528

def is_valid_path : ℕ × ℕ → list (ℕ × ℕ) → Prop
| (0, 0), [] => true
| (x, y), (dx, dy) :: rest => 
  (x + dx, y + dy) = (0, 0) ∨ 
  (x + dx ≥ 0) ∧ 
  (y + dy ≥ 0) ∧ 
  is_valid_path (x + dx, y + dy) rest

def valid_moves :=
[(1, 0), (-1, 0), (0, 1)]

def number_of_paths (start : ℕ × ℕ) (end : ℕ × ℕ) (n : ℕ) : ℕ := 
  (list.replicate n (list.filter (λ move, move ∈ valid_moves) valid_moves)).sum

theorem count_paths_from_0_0_to_3_3 :
  number_of_paths (0, 0) (3, 3) 6 = 80 := sorry

end count_paths_from_0_0_to_3_3_l407_407528


namespace sum_term_addition_l407_407444

theorem sum_term_addition (k : ℕ) (hk : k ≥ 2) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end sum_term_addition_l407_407444


namespace hydrochloric_acid_moles_l407_407087

theorem hydrochloric_acid_moles (mass_water : ℝ) (molar_mass_water : ℝ) (moles_naHCO₃ : ℝ) :
  (mass_water = 36) → (molar_mass_water = 18.015) → (moles_naHCO₃ = 2) → 
  (moles_naHCO₃ = mass_water / molar_mass_water) →
  (mass_water / molar_mass_water) = 2 :=
by {
  intros h_mass h_molar_mass h_moles_NaHCO₃ h_react,
  have h_water_eq : (36 : ℝ) / (18.015 : ℝ) = 2, by norm_num,
  exact h_water_eq,
}

end hydrochloric_acid_moles_l407_407087


namespace train_passing_time_l407_407537

-- Define the conditions
def train_length : ℝ := 140 -- length of the train in meters
def train_speed : ℝ := 50 * (1000 / 3600) -- speed of the train in m/s
def man_speed : ℝ := 4 * (1000 / 3600) -- speed of the man in m/s

-- Calculate the relative speed
def relative_speed : ℝ := train_speed + man_speed

-- Time required for the train to pass the man
def passing_time : ℝ := train_length / relative_speed

-- Prove that the passing time is 9.33 seconds
theorem train_passing_time : passing_time = 9.33 := by
  -- Proof
  sorry

end train_passing_time_l407_407537


namespace pipeB_fill_time_is_32_l407_407440

-- Definitions for conditions
def pipeA_fill_time := 24
def pipes_open_time := 8
def total_fill_time := 18

-- The amount of tank filled by pipe A in one minute
def pipeA_fill_rate := 1 / pipeA_fill_time.toRat

-- The amount of tank filled by pipe B in t minutes
def pipeB_fill_rate (t : ℚ) := 1 / t

-- The equation that needs to be satisfied
def total_fill_equation (t : ℚ) := 
  8 * (pipeA_fill_rate + pipeB_fill_rate t) + 10 * pipeA_fill_rate = 1

-- Main proposition to prove that t = 32 satisfies the given conditions
theorem pipeB_fill_time_is_32 : total_fill_equation 32 := by 
  -- Leave proof as sorry, focusing on the statement for now
  sorry

end pipeB_fill_time_is_32_l407_407440


namespace num_lattice_points_on_sphere_5_l407_407770

theorem num_lattice_points_on_sphere_5 :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.to_finset.card = 54 := 
by
  sorry

end num_lattice_points_on_sphere_5_l407_407770


namespace lattice_points_count_is_30_l407_407794

-- Define the function that counts the number of lattice points (x, y, z) such that x^2 + y^2 + z^2 = 25.
def count_lattice_points := {p : ℤ × ℤ × ℤ // p.1 ^ 2 + p.2.1 ^ 2 + p.2.2 ^ 2 = 25}.to_finset.card

-- The theorem to prove that the count is 30.
theorem lattice_points_count_is_30 : count_lattice_points = 30 :=
by
  sorry

end lattice_points_count_is_30_l407_407794


namespace number_of_positive_factors_of_60_l407_407249

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l407_407249


namespace sum_floor_sqrt_1_to_25_l407_407031

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l407_407031


namespace ratio_of_triangle_areas_l407_407922

noncomputable def area (a b c : ℕ) : ℚ := if a * a + b * b = c * c then (a * b : ℚ) / 2 else 0

theorem ratio_of_triangle_areas :
  let PQR := (7, 24, 25)
  let STU := (9, 40, 41)
  area PQR.1 PQR.2 PQR.3 / area STU.1 STU.2 STU.3 = (7 / 15 : ℚ) :=
by
  sorry

end ratio_of_triangle_areas_l407_407922


namespace domain_of_h_l407_407650

noncomputable def h (x : ℝ) : ℝ := (x^3 - 2*x^2 - x + 1) / (x^2 - 9)

theorem domain_of_h : 
  {x : ℝ | x ≠ -3 ∧ x ≠ 3} = set.univ \ {x | x = -3 ∨ x = 3} := 
begin
  sorry
end

end domain_of_h_l407_407650


namespace different_arrangements_l407_407908

-- Definitions from conditions
def subjects : Set String := {"Chinese", "Mathematics", "English", "Science Comprehensive"}
def classes : Finset ℕ := {1, 2, 3}

-- Conditions translated to Lean definitions
def distinct_subjects (s1 s2 : String) : Prop := s1 ≠ s2
def no_math_science_together (c : Finset (Set String)) : Prop :=
  ∀ s ∈ c, "Mathematics" ∈ s → "Science Comprehensive" ∉ s

-- Lean statement for the math proof problem
theorem different_arrangements :
  ∃ arrangements : Finset (Finset (Set String)),
    arrangements.card = 30 ∧
    ∀ c ∈ arrangements, no_math_science_together c ∧
    ∀ s ∈ c, s ⊆ subjects ∧
    (∀ s1 s2 ∈ subjects, distinct_subjects s1 s2) :=
sorry

end different_arrangements_l407_407908


namespace largest_value_of_n_in_base_10_l407_407878

theorem largest_value_of_n_in_base_10 :
  ∃ (n : ℕ) (A B C : ℕ), 0 ≤ A ∧ A < 7 ∧ 0 ≤ B ∧ B < 7 ∧ 0 ≤ C ∧ C < 7 ∧
  n = 49 * A + 7 * B + C ∧ n = 121 * C + 11 * B + A ∧
  ∀ (m : ℕ) (X Y Z : ℕ), 0 ≤ X ∧ X < 7 ∧ 0 ≤ Y ∧ Y < 7 ∧ 0 ≤ Z ∧ Z < 7 ∧
  m = 49 * X + 7 * Y + Z ∧ m = 121 * Z + 11 * Y + X → m ≤ n :=
begin
  sorry
end

end largest_value_of_n_in_base_10_l407_407878


namespace find_matrix_N_l407_407097

theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℚ) :
  N * !![!![2, -5], !![4, -3]] = !![!![-20, -8], !![9, 3]] →
  N = !![!![(46/7 : ℚ), -(58/7 : ℚ)], !![-(39/14 : ℚ), (51/14 : ℚ)]] :=
by 
  intro h
  sorry

end find_matrix_N_l407_407097


namespace curve_l_intersect_distances_l407_407767

variable (a : ℝ) (t : ℝ)
variable (A B P : ℝ × ℝ)

-- Assuming conditions as hypotheses
def curve_c (x y : ℝ) := y^2 = 2 * a * x
def line_l (x y : ℝ) := x - y - 2 = 0
def point_P : (ℝ × ℝ) := (-2, -4)
def parametric_line_x (t : ℝ) := -2 + (Real.sqrt 2 / 2) * t
def parametric_line_y (t : ℝ) := -4 + (Real.sqrt 2 / 2) * t
def line_l_intersection_curve_c (t : ℝ) := curve_c (parametric_line_x t) (parametric_line_y t) 
    ∧ line_l (parametric_line_x t) (parametric_line_y t)

-- Given conditions
axiom hyp_curve_c : curve_c
axiom hyp_line_l : line_l
axiom hyp_point_P : point_P
axiom hyp_parametric_line_x : parametric_line_x
axiom hyp_parametric_line_y : parametric_line_y
axiom hyp_line_l_intersection_curve_c : line_l_intersection_curve_c 

-- Prove the condition regarding distances
theorem curve_l_intersect_distances : 
  (|P - A| * |P - B| = |A - B|^2) → a = 1 :=
sorry

end curve_l_intersect_distances_l407_407767


namespace cross_section_quadrilateral_is_cylinder_l407_407944

-- Definition of the solids
inductive Solid
| cone
| cylinder
| sphere

-- Predicate for the cross-section being a quadrilateral
def is_quadrilateral_cross_section (solid : Solid) : Prop :=
  match solid with
  | Solid.cylinder => true
  | Solid.cone     => false
  | Solid.sphere   => false

-- Main theorem statement
theorem cross_section_quadrilateral_is_cylinder (s : Solid) :
  is_quadrilateral_cross_section s → s = Solid.cylinder :=
by
  cases s
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]

end cross_section_quadrilateral_is_cylinder_l407_407944


namespace olivia_earning_l407_407076

theorem olivia_earning
  (cost_per_bar : ℝ)
  (total_bars : ℕ)
  (unsold_bars : ℕ)
  (sold_bars : ℕ := total_bars - unsold_bars)
  (earnings : ℝ := sold_bars * cost_per_bar) :
  cost_per_bar = 3 → total_bars = 7 → unsold_bars = 4 → earnings = 9 :=
by
  sorry

end olivia_earning_l407_407076


namespace solve_inequality_1_find_range_of_a_l407_407498

-- Question 1: Solve the inequality |1/log_{1/2} x + 2| ≥ 3/2

theorem solve_inequality_1 (x : ℝ) (hx1 : x > 0) (hx2 : x ≠ 1) :
  (| (1 / Real.log x / Real.log (1 / 2)) + 2| ≥ 3 / 2) ↔
  x ∈ Set.union (Set.union (Set.Ioo 0 1) (Set.Ioc 1 2)) (Set.Ici 4) :=
  sorry

-- Question 2: Find the range of values for a such that the integer solutions of 
-- the inequality 0 ≤ ax + 5 ≤ 4 are 1, 2, 3, 4

theorem find_range_of_a (a : ℝ) :
  (∀ x : ℤ, x ∈ {1, 2, 3, 4} → (0 ≤ a * (x : ℝ) + 5 ∧ a * (x : ℝ) + 5 ≤ 4)) ↔
  a ∈ Set.Icc (-5 / 4) (-1) :=
  sorry

end solve_inequality_1_find_range_of_a_l407_407498


namespace num_factors_of_60_l407_407236

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l407_407236


namespace average_age_combined_l407_407401

theorem average_age_combined (n5th_graders : ℕ) (avg5th_graders : ℕ) (nparents : ℕ) (avgparents : ℕ) 
  (nteachers : ℕ) (avgteachers : ℕ) :
  n5th_graders = 40 →
  avg5th_graders = 10 →
  nparents = 60 →
  avgparents = 35 →
  nteachers = 5 →
  avgteachers = 45 →
  (40 * 10 + 60 * 35 + 5 * 45) / (40 + 60 + 5) = 26 :=
by {
  intros,
  sorry
}

end average_age_combined_l407_407401


namespace find_k_l407_407104

noncomputable def k_val : ℝ := 19.2

theorem find_k (k : ℝ) :
  (4 + ∑' n : ℕ, (4 + n * k) / (5^(n + 1))) = 10 ↔ k = k_val :=
  sorry

end find_k_l407_407104


namespace negative_integers_l407_407547

theorem negative_integers (s : set ℝ) (h : s = {0, 3, -5, -3.6}) : s ∩ set_of (λ x, x < 0 ∧ x ∈ (set_of (λ y, ∃ n : ℤ, y = n))) = {-5} :=
by {
  sorry
}

end negative_integers_l407_407547


namespace solve_inequality_l407_407874

theorem solve_inequality (x : ℝ) : (1 / x < 1 / 2) ↔ (x ∈ Set.Ioo (-∞ : ℝ) 0 ∪ Set.Ioo 2 ∞) :=
sorry

end solve_inequality_l407_407874


namespace question1_question2_l407_407158

theorem question1 (k a : ℝ) (a_pos : a > 0) (a_ne_one : a ≠ 1)
  (h1 : k * a^0 = 1)
  (h2 : k * a^(-3) = 8) :
  k = 1 ∧ a = 1 / 2 :=
by {
  sorry
}

theorem question2 {f : ℝ → ℝ} (g : ℝ → ℝ) :
  (∀ x, f x = 1 * (1 / 2)^(-x)) →
  (∀ x, g x = (f x - 1) / (f x + 1)) →
  (∀ x, g (-x) = -g x) :=
by {
  sorry
}

end question1_question2_l407_407158


namespace problem_k_star_k_star_k_l407_407071

def star (x y : ℝ) : ℝ := 2 * x^2 - y

theorem problem_k_star_k_star_k (k : ℝ) : star k (star k k) = k :=
by
  sorry

end problem_k_star_k_star_k_l407_407071


namespace factors_of_60_l407_407230

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l407_407230


namespace functions_are_identical_l407_407478

def f1 (x : ℝ) : ℝ := 1
def f2 (x : ℝ) : ℝ := x^0

theorem functions_are_identical : ∀ (x : ℝ), f1 x = f2 x :=
by
  intro x
  simp [f1, f2]
  sorry

end functions_are_identical_l407_407478


namespace prism_volume_l407_407983

noncomputable def volume_of_prism (l w h : ℝ) : ℝ :=
l * w * h

theorem prism_volume (l w h : ℝ) (h1 : l = 2 * w) (h2 : l * w = 10) (h3 : w * h = 18) (h4 : l * h = 36) :
  volume_of_prism l w h = 36 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end prism_volume_l407_407983


namespace sum_floor_sqrt_1_to_25_l407_407000

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l407_407000


namespace boys_caps_vs_boots_l407_407390

theorem boys_caps_vs_boots (total_boys : ℕ) (without_boots : ℕ) (without_caps : ℕ) 
  (h_total : total_boys = 16) (h_without_boots : without_boots = 10) 
  (h_without_caps : without_caps = 2) (h_boots_caps : ∀ b, b = 6) :
  let boys_with_boots := total_boys - without_boots,
      boys_with_caps := total_boys - without_caps,
      boys_with_both := boys_with_boots,
      boys_with_caps_no_boots := boys_with_caps - boys_with_both in
  boys_with_cap_no_boots = 8 ∧ boys_with_boots = 6 ∧ boys_with_caps_no_boots - boys_with_boots = 2 := 
by
  have boys_with_boots_eq : total_boys - without_boots = 6 := by sorry
  have boys_with_caps_eq : total_boys - without_caps = 14 := by sorry
  have boys_with_both_eq : total_boys - without_boots = 6 := by sorry
  have boys_with_cap_no_boots_eq : (total_boys - without_caps) - (total_boys - without_boots) = 8 := by sorry
  show boys_with_cap_no_boots = 8 ∧ boys_with_boots = 6 ∧ boys_with_caps_no_boots - boys_with_boots = 2 from
  ⟨by rw boys_with_cap_no_boots_eq, by rw boys_with_boots_eq, by simp [boys_with_cap_no_boots_eq, boys_with_boots_eq]⟩


end boys_caps_vs_boots_l407_407390


namespace log_sum_geometric_seq_l407_407837

open Real

theorem log_sum_geometric_seq
  (a : ℕ → ℝ)
  (h_geom : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h_decreasing : ∀ n : ℕ, a (n + 1) ≤ a n)
  (h_sum : a 1 + a 2 = 11)
  (h_prod : a 1 * a 2 = 10) :
  ∑ i in Finset.range 10, log (a (i + 1)) = -35 := 
by
  sorry

end log_sum_geometric_seq_l407_407837


namespace weight_of_dry_grapes_l407_407487

theorem weight_of_dry_grapes (w_fresh : ℝ) (perc_water_fresh perc_water_dried : ℝ) (w_non_water : ℝ) (w_dry : ℝ) :
  w_fresh = 5 →
  perc_water_fresh = 0.90 →
  perc_water_dried = 0.20 →
  w_non_water = w_fresh * (1 - perc_water_fresh) →
  w_non_water = w_dry * (1 - perc_water_dried) →
  w_dry = 0.625 :=
by sorry

end weight_of_dry_grapes_l407_407487


namespace point_on_y_axis_coordinates_l407_407145

theorem point_on_y_axis_coordinates (a : ℝ) (h : a - 1 = 0) : (a-1, a^2-9) = (0, -8) :=
by
  have ha : a = 1 := by linarith
  rw [ha, sq]
  linarith

end point_on_y_axis_coordinates_l407_407145


namespace parapola_eq_and_ratio_l407_407996

-- Definitions as conditions
def is_parabola (C : ℝ → ℝ → Prop) (p : ℝ) : Prop :=
  ∀ x y, C x y ↔ x^2 = 2 * p * y

def is_circle (C : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C x y ↔ x^2 + y^2 = 1

def line_eq (P Q : ℝ × ℝ) (m : ℝ) (b : ℝ) : Prop :=
  ∀ x y, (y - snd P = m * (x - fst P)) ∧ (y = m * x - b)

-- The problem statement
theorem parapola_eq_and_ratio (O F P Q : ℝ × ℝ) (p : ℝ) (C1 C2 : ℝ → ℝ → Prop)
  (H_O : O = (0, 0))
  (H_F : F = (0, p/2))
  (H_C1 : is_parabola C1 p)
  (H_C2 : is_circle C2)
  (H_tangent : ∀ x y, C1 x y → ∀ x' y', C2 x' y' → (line_eq (x, y) (x', y') 1 (-sqrt 2)))
  (H_eq_line : line_eq P Q 1 (-sqrt 2)) :
  (∀ x y, C1 x y ↔ x^2 = 4 * sqrt 2 * y) ∧
  (∃ S1 S2 : ℝ, let ratio := S1 / S2 in ∀ p > 0, ratio ≥ 2 * sqrt 2 + 3) :=
sorry

end parapola_eq_and_ratio_l407_407996


namespace gas_fee_calculation_l407_407281

theorem gas_fee_calculation (x : ℕ) (h1 : x > 60) 
  (h2 : 0.88 * x = 0.8 * 60 + 1.2 * (x - 60)) :
  0.88 * x = 66 :=
by
  sorry

end gas_fee_calculation_l407_407281


namespace find_possible_values_of_n_l407_407501

/--
Given:
1. 5610 people went to the bookstore to buy books.
2. Each person buys 3 different kinds of books.
3. Any two people share at least one book in common.
4. Each book is bought by at most 5 people.
5. 10 people bought n different kinds of books.

Prove: The possible values of n are 6 and 7.
-/
theorem find_possible_values_of_n :
  (∃ n, ∀ (people : Fin 5610 → Fin 5 → Fin n), 
     (∀ (p1 p2 : Fin 5610), p1 ≠ p2 → ∃ i, people p1 i = people p2 i) ∧
     (∀ (b : Fin n), (card {p1 | ∃ i, people p1 i = b} ≤ 5)) ∧
     (∃ ten_people : Fin 10 → Fin 5610, ∃ books : Fin n, ∀ i, ∃ j k l, 
       people (ten_people i) j = books k ∧ 
       people (ten_people i) k = books l ∧ 
       people (ten_people i) l = books (j + k + l - books k - books l - books j))) ↔
  (n = 6 ∨ n = 7) :=
by
  sorry

end find_possible_values_of_n_l407_407501


namespace real_condition_complex_condition_purely_imaginary_condition_l407_407473

open Complex

-- Definition of z based on m
def z (m : ℝ) : ℂ := (m - 1) + (m + 1) * Complex.i

-- Theorem statements
theorem real_condition (m : ℝ) : (m = -1) → (Complex.imag (z m) = 0) :=
sorry

theorem complex_condition (m : ℝ) : (m ≠ -1) → (Complex.imag (z m) ≠ 0) :=
sorry

theorem purely_imaginary_condition (m : ℝ) : (m = 1) → (Complex.real (z m) = 0 ∧ Complex.imag (z m) ≠ 0) :=
sorry

end real_condition_complex_condition_purely_imaginary_condition_l407_407473


namespace max_real_part_z6_l407_407365

noncomputable def greatest_real_part (z : ℂ) : ℂ :=
if z = -3 then z
else if z = -2 + complex.i then z
else if z = -real.sqrt 3 - real.sqrt 3 * complex.i then z
else if z = 1 - real.sqrt 2 * complex.i then z
else if z = -complex.i then z
else 0

theorem max_real_part_z6 :
  greatest_real_part (-3) = -3 :=
by {
  sorry
}

end max_real_part_z6_l407_407365


namespace words_per_page_is_106_l407_407507

noncomputable def book_pages := 154
noncomputable def max_words_per_page := 120
noncomputable def total_words_mod := 221
noncomputable def mod_val := 217

def number_of_words_per_page (p : ℕ) : Prop :=
  (book_pages * p ≡ total_words_mod [MOD mod_val]) ∧ (p ≤ max_words_per_page)

theorem words_per_page_is_106 : number_of_words_per_page 106 :=
by
  sorry

end words_per_page_is_106_l407_407507


namespace lattice_points_with_distance_5_from_origin_l407_407799

theorem lattice_points_with_distance_5_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 42 :=
by
  sorry

end lattice_points_with_distance_5_from_origin_l407_407799


namespace baker_weekend_hours_l407_407505

noncomputable def loaves_per_hour : ℕ := 5
noncomputable def ovens : ℕ := 4
noncomputable def weekday_hours : ℕ := 5
noncomputable def total_loaves : ℕ := 1740
noncomputable def weeks : ℕ := 3
noncomputable def weekday_days : ℕ := 5
noncomputable def weekend_days : ℕ := 2

theorem baker_weekend_hours :
  ((total_loaves - (weeks * weekday_days * weekday_hours * (loaves_per_hour * ovens))) / (weeks * (loaves_per_hour * ovens))) / weekend_days = 4 := by
  sorry

end baker_weekend_hours_l407_407505


namespace distance_from_origin_to_line_l407_407175

theorem distance_from_origin_to_line {a b θ : ℝ} (hne: a ≠ b) 
  (h1: a^2 * sin θ + a * cos θ - 1 = 0) 
  (h2: b^2 * sin θ + b * cos θ - 1 = 0) :
  let line_eq : ℝ × ℝ → ℝ := λ p, p.1 * cos θ + p.2 * sin θ - 1 in
  let distance (p : ℝ × ℝ) (line : ℝ × ℝ → ℝ) : ℝ := 
    abs (line p) / sqrt (cos θ ^ 2 + sin θ ^ 2) in
  distance (0, 0) line_eq = 1 :=
sorry

end distance_from_origin_to_line_l407_407175


namespace collinear_X_Y_Z_l407_407307

-- Definitions for the given elements and the conditions.
structure Triangle :=
(A B C : Point)

structure PointOnCircle (Γ : Circle) :=
(P : Point)
(on_circle : Γ.contains P)

-- Definitions of points involved
def circumcenter (Δ : Triangle) : Point := sorry -- Assuming we have a function to find circumcenter
def altitude (Δ : Triangle) (C : Point) : Line := sorry -- Function to find altitude from C
def midpoint (A C : Point) : Point := sorry -- Function to find the midpoint of AC
def intersects_circle (l : Line) (Γ : Circle) : set Point := sorry -- Points where line intersects circle

-- Main theorem statement
theorem collinear_X_Y_Z
  (Δ : Triangle)
  (O : Point := circumcenter Δ)
  (D : Point := sorry) -- D is on line altitude Δ, through C
  (CD : Line := altitude Δ Δ.C) 
  (M : Point := midpoint Δ.A Δ.C)
  (Γ : Circle) -- Circle with diameter AD
  (Y : PointOnCircle Γ)
  (X : PointOnCircle (circumcircle Δ))
  (Z_position : Point := intersection (Line_through DO) (Line_of_side_AC))
  : Collinear X.Y X.Z X.Y := sorry

end collinear_X_Y_Z_l407_407307


namespace monthly_compounding_greater_than_yearly_l407_407958

open Nat Real

theorem monthly_compounding_greater_than_yearly : 
  1 + 3 / 100 < (1 + 3 / (12 * 100)) ^ 12 :=
by
  -- This is the proof we need to write.
  sorry

end monthly_compounding_greater_than_yearly_l407_407958


namespace intersection_of_M_and_N_l407_407174

theorem intersection_of_M_and_N :
  let M := { x : ℝ | -6 ≤ x ∧ x < 4 }
  let N := { x : ℝ | -2 < x ∧ x ≤ 8 }
  M ∩ N = { x | -2 < x ∧ x < 4 } :=
by
  sorry -- Proof is omitted

end intersection_of_M_and_N_l407_407174


namespace floor_sqrt_sum_l407_407046

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l407_407046


namespace multiplication_expansion_l407_407360

theorem multiplication_expansion (y : ℤ) :
  (y^4 + 9 * y^2 + 81) * (y^2 - 9) = y^6 - 729 :=
by
  sorry

end multiplication_expansion_l407_407360


namespace no_such_19gon_exists_l407_407625

theorem no_such_19gon_exists :
  ¬(∃ (α : Fin 19 → ℕ), (∀ i j : Fin 19, i ≠ j → α i ≠ α j) ∧ (∀ i : Fin 19, α i < 180) ∧ (Finset.univ.sum (λ i, α i) = 360)) :=
by
  -- proof goes here
  sorry

end no_such_19gon_exists_l407_407625


namespace polynomial_even_odd_coefficients_l407_407320

theorem polynomial_even_odd_coefficients
  (p q : ℤ[X])
  (h1 : ∀ c in (p * q).coeffs, even c)
  (h2 : ∃ c in (p * q).coeffs, ¬ (4 ∣ c)) :
  (∃ (a : ℤ[X]), (∀ i, even (a.coeff i)) ∧ (∃ j, ¬ even ((p * q).coeff j))) ∨ (∃ (a : ℤ[X]), (∀ i, even (a.coeff i)) ∧ (∃ j, ¬ even ((p * q).coeff j))) :=
by sorry

end polynomial_even_odd_coefficients_l407_407320


namespace cone_volume_l407_407713

theorem cone_volume (l : ℝ) (θ : ℝ) (V : ℝ) (R : ℝ) (h : ℝ) 
  (h1 : l = 4)
  (h2 : θ = π / 2)
  (h3 : 1/2 * θ * l = 2 * π * R)
  (h4 : R = 1)
  (h5 : h = sqrt (l^2 - R^2))
  (h6 : h = sqrt 15)
  (h7 : V = 1/3 * π * R^2 * h) :
  V = (sqrt 15) / 3 * π := by
  sorry

end cone_volume_l407_407713


namespace minimum_value_l407_407940

variable {x : ℝ}

theorem minimum_value (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
by
  sorry

end minimum_value_l407_407940


namespace base9_addition_correct_l407_407540

-- Definition of base 9 addition problem.
def add_base9 (a b c : ℕ) : ℕ :=
  let sum := a + b + c -- Sum in base 10
  let d0 := sum % 9 -- Least significant digit in base 9
  let carry1 := sum / 9
  (carry1 + carry1 / 9 * 9 + carry1 % 9) + d0 -- Sum in base 9 considering carry

-- The specific values converted to base 9 integers
def n1 := 3 * 9^2 + 4 * 9 + 6
def n2 := 8 * 9^2 + 0 * 9 + 2
def n3 := 1 * 9^2 + 5 * 9 + 7

-- The expected result converted to base 9 integer
def expected_sum := 1 * 9^3 + 4 * 9^2 + 1 * 9 + 6

theorem base9_addition_correct : add_base9 n1 n2 n3 = expected_sum := by
  -- Proof will be provided here
  sorry

end base9_addition_correct_l407_407540


namespace hyperbola_eccentricity_l407_407126

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) :
  let e := (real.sqrt (1 + (b^2) / (a^2))) in
  (\overrightarrow{AF} = (5 / 2) * \overrightarrow{FB}) →
  e = (2 * real.sqrt 14) / 7 :=
by
  sorry

end hyperbola_eccentricity_l407_407126


namespace binom_30_3_l407_407600

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l407_407600


namespace num_factors_of_60_l407_407195

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l407_407195


namespace sally_purchased_20_fifty_cent_items_l407_407869

noncomputable def num_fifty_cent_items (x y z : ℕ) (h1 : x + y + z = 30) (h2 : 50 * x + 500 * y + 1000 * z = 10000) : ℕ :=
x

theorem sally_purchased_20_fifty_cent_items
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 50 * x + 500 * y + 1000 * z = 10000)
  : num_fifty_cent_items x y z h1 h2 = 20 :=
sorry

end sally_purchased_20_fifty_cent_items_l407_407869


namespace num_factors_60_l407_407185

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l407_407185


namespace paths_to_form_2005_l407_407395

-- Define the conditions as described in the problem
def number_of_paths : ℕ := 
  let initial_moves := 6 in
  let moves_to_second_zero := 2 in
  let moves_to_five := 3 in
  initial_moves * moves_to_second_zero * moves_to_five

-- Statement to prove the calculated paths
theorem paths_to_form_2005 : number_of_paths = 36 :=
by {
  -- Calculate explicitly
  have h1 : number_of_paths = 6 * 2 * 3 := rfl,
  -- Check the final calculation
  have h2 : 6 * 2 * 3 = 36 := by norm_num,
  exact h2,
}

end paths_to_form_2005_l407_407395


namespace binomial_coefficient_30_3_l407_407588

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l407_407588


namespace gunny_bag_can_hold_packets_l407_407520

theorem gunny_bag_can_hold_packets :
  let ton_to_kg := 1000
  let max_capacity_tons := 13
  let pound_to_kg := 0.453592
  let ounce_to_g := 28.3495
  let kilo_to_g := 1000
  let wheat_packet_pounds := 16
  let wheat_packet_ounces := 4
  let max_capacity_kg := max_capacity_tons * ton_to_kg
  let wheat_packet_kg := wheat_packet_pounds * pound_to_kg + (wheat_packet_ounces * ounce_to_g) / kilo_to_g
  max_capacity_kg / wheat_packet_kg >= 1763 := 
by
  sorry

end gunny_bag_can_hold_packets_l407_407520


namespace picnic_problem_l407_407529

variable (M W A C : ℕ)

theorem picnic_problem (h1 : M = 90)
  (h2 : M = W + 40)
  (h3 : M + W + C = 240) :
  A = M + W ∧ A - C = 40 := by
  sorry

end picnic_problem_l407_407529


namespace cannot_be_right_angle_triangle_l407_407546

-- Definition of the converse of the Pythagorean theorem
def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

-- Definition to check if a given set of sides cannot form a right-angled triangle
def cannot_form_right_angle_triangle (a b c : ℕ) : Prop :=
  ¬ is_right_angle_triangle a b c

-- Given sides of the triangle option D
theorem cannot_be_right_angle_triangle : cannot_form_right_angle_triangle 3 4 6 :=
  by sorry

end cannot_be_right_angle_triangle_l407_407546


namespace smallest_perimeter_of_triangle_with_consecutive_odd_primes_l407_407539

theorem smallest_perimeter_of_triangle_with_consecutive_odd_primes : 
  ∃ (p : ℕ), prime p ∧ prime (p + 2) ∧ prime (p + 4) ∧ prime (3 * p + 6) ∧ 3 * p + 6 = 41 :=
sorry

end smallest_perimeter_of_triangle_with_consecutive_odd_primes_l407_407539


namespace find_matrix_N_l407_407095

theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℚ) :
  N * ![![2, -5], ![4, -3]] = ![![ -20, -8], ![9, 3]] ->
  N = ![![43 / 7, -54 / 7], ![-33 / 14, 24 / 7]] := 
by
  intros h
  sorry

end find_matrix_N_l407_407095


namespace find_points_C_l407_407290

noncomputable def point : Type := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def area_triangle (A B C : point) : ℝ :=
  (1 / 2) * Real.abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

def perimeter_triangle (A B C : point) : ℝ :=
  distance A B + distance B C + distance C A

theorem find_points_C :
  let A : point := (0, 0)
  let B : point := (12, 0)
  (distance A B = 12) →
  (∃ C : point, (perimeter_triangle A B C = 60) ∧ (area_triangle A B C = 120)) →
  ∃ C : point, number_of_C_points = 4 := sorry

end find_points_C_l407_407290


namespace binomial_n_value_l407_407706

-- Defining binomial distribution conditions
def binomial (n : ℕ) (p : ℝ) : Prop :=
  ∀ X : ℝ, ∃ X : ℝ, (E X = n * p ∧ D X = n * p * (1 - p))

-- Given conditions
axiom hE : ∃ (X : ℝ), E X = 4
axiom hD : ∃ (X : ℝ), D X = 2

-- Proof statement
theorem binomial_n_value (n : ℕ) (p : ℝ) (X : ℝ) (h : binomial n p) (hE : E X = 4) (hD : D X = 2) : n = 8 :=
  sorry

end binomial_n_value_l407_407706


namespace floor_sum_sqrt_25_l407_407011

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l407_407011


namespace probability_no_adjacent_red_marbles_l407_407823

theorem probability_no_adjacent_red_marbles (total_marbles red_marbles blue_marbles : ℕ)
  (h_total : total_marbles = 12) 
  (h_red : red_marbles = 4)
  (h_blue : blue_marbles = 8) : 
  (total_marbles = red_marbles + blue_marbles) → 
  (∃ p : ℚ, p = 7 / 33) :=
by
  intro h_sum
  -- Place the total number of arrangements of 4 red and 8 blue marbles in a line
  have total_arrangements : ℕ := nat.choose total_marbles red_marbles
  -- Convert the problem to a linear arrangement of 4 red and 7 blue marbles (fixing one blue marble)
  have linear_total : ℕ := nat.choose (total_marbles - 1) red_marbles
  -- Calculate the number of valid arrangements where no two red marbles are adjacent
  have valid_arrangements : ℕ := nat.choose (blue_marbles + 1) red_marbles
  -- Calculate the probability
  have calculated_probability : ℚ := valid_arrangements / linear_total
  use calculated_probability
  -- Skip actual proof
  sorry

end probability_no_adjacent_red_marbles_l407_407823


namespace equal_ratios_of_segments_l407_407995

theorem equal_ratios_of_segments 
  (A B C I D H P Q O L N : Point)
  (h1 : is_incenter I A B C)
  (h2 : on_line_vertical I D B C)
  (h3 : perpendicular A H B C)
  (h4 : intersects_line_at P A H B I)
  (h5 : intersects_line_at Q A H C I)
  (h6 : is_circumcenter O I P Q)
  (h7 : intersects_extension_at L A O B C)
  (h8 : N_on_circumcircle_of A I L B C) :
  AB > AC →
  (BD/CD) = (BN/CN) :=
sorry

end equal_ratios_of_segments_l407_407995


namespace number_of_positive_factors_of_60_l407_407253

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l407_407253


namespace sum_floor_sqrt_l407_407052

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l407_407052


namespace raisins_cost_fraction_l407_407486

theorem raisins_cost_fraction (R : ℝ) : 
  let cost_of_raisins := 3 * R,
      cost_of_nuts := 4 * (2 * R),
      total_cost := cost_of_raisins + cost_of_nuts
  in (cost_of_raisins / total_cost) = 3 / 11 :=
by
  sorry

end raisins_cost_fraction_l407_407486


namespace prove_A_conditions_l407_407856

variables (Judy : Type) (received_A : Judy → Prop) (score : Judy → ℝ) (submitted_essays : Judy → Prop)

theorem prove_A_conditions (h : ∀ j : Judy, (score j ≥ 90.0) ∧ submitted_essays j → received_A j) :
    ∀ j : Judy, received_A j → (score j ≥ 90.0) ∧ submitted_essays j :=
by
  intro j
  intro ha
  apply classical.by_contradiction
  intro hnot
  cases em (score j ≥ 90.0 ∧ submitted_essays j) with hsub hnsub
  . exact False.elim (hnot hsub)
  . exact False.elim (h ha hsub).elim
  sorry

end prove_A_conditions_l407_407856


namespace product_of_magnitudes_l407_407339

noncomputable def complex_numbers_with_unit_modulus (a b c : ℂ) : Prop :=
  complex.abs a = 1 ∧ complex.abs b = 1 ∧ complex.abs c = 1

noncomputable def given_equation (a b c : ℂ) : Prop :=
  a^2 / (b * c) + b^2 / (c * a) + c^2 / (a * b) = 1

theorem product_of_magnitudes (a b c : ℂ) 
  (h1 : complex_numbers_with_unit_modulus a b c) 
  (h2 : given_equation a b c) : 
  {z : ℂ | ∃ a b c, complex.abs a = 1 ∧ complex.abs b = 1 ∧ complex.abs c = 1 ∧ 
  a^2 / (b * c) + b^2 / (c * a) + c^2 / (a * b) = 1 ∧ z = complex.abs (a + b + c)}.prod (λ z, z) = 2 := 
sorry

end product_of_magnitudes_l407_407339


namespace min_value_at_1_l407_407155

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2 * a * x + 8 else x + 4 / x + 2 * a

theorem min_value_at_1 (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ (a = 5/4 ∨ a = 2 ∨ a = 4) :=
by
  sorry

end min_value_at_1_l407_407155


namespace carriage_and_people_l407_407762

variable {x y : ℕ}

theorem carriage_and_people :
  (3 * (x - 2) = y) ∧ (2 * x + 9 = y) :=
sorry

end carriage_and_people_l407_407762


namespace lattice_points_count_is_30_l407_407795

-- Define the function that counts the number of lattice points (x, y, z) such that x^2 + y^2 + z^2 = 25.
def count_lattice_points := {p : ℤ × ℤ × ℤ // p.1 ^ 2 + p.2.1 ^ 2 + p.2.2 ^ 2 = 25}.to_finset.card

-- The theorem to prove that the count is 30.
theorem lattice_points_count_is_30 : count_lattice_points = 30 :=
by
  sorry

end lattice_points_count_is_30_l407_407795


namespace tangency_ratio_l407_407375

-- Define the points and the cube structure using a structure or type 
structure Point (α : Type) := (x : α) (y : α) (z : α)
structure Cube (α : Type) := 
  (A : Point α)
  (B : Point α)
  (C : Point α)
  (D : Point α)
  (A1 : Point α)
  (B1 : Point α)
  (C1 : Point α)
  (D1 : Point α)

-- Define a function to check midpoint
def is_midpoint {α : Type} [LinearOrder α] (K A A1 : Point α) : Prop :=
  K.x = (A.x + A1.x) / 2 ∧ K.y = (A.y + A1.y) / 2 ∧ K.z = (A.z + A1.z) / 2

-- Define a function to state the condition for the point on edge BC
def on_edge {α : Type} [LinearOrder α] (L B C : Point α) : Prop :=
  L.x = B.x ∧ L.z = B.z ∧ B.y ≤ L.y ∧ L.y ≤ C.y

-- Define variables as the given problem conditions
variable {α : Type} [LinearOrder α] (cube : Cube α)
variable (K L P : Point α)

-- The main theorem which contains the assertion of the problem
theorem tangency_ratio 
  (midpoint_K : is_midpoint K cube.A cube.A1)
  (point_on_edge_L : on_edge L cube.B cube.C)
  (tangency_condition : ∃ S : Point α, ∀ P K L,
  /* Your tangency condition, e.g., distance computations can come here */ sorry):
  (KP : dist K P) = 4 / 5 * (dist P L)  :=
sorry

end tangency_ratio_l407_407375


namespace num_lattice_points_on_sphere_5_l407_407773

theorem num_lattice_points_on_sphere_5 :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.to_finset.card = 54 := 
by
  sorry

end num_lattice_points_on_sphere_5_l407_407773


namespace average_weight_section_A_l407_407918

variables {n_A n_B : ℕ} {W_B W_total W_A : ℝ}

-- Defining the conditions
def condition_1 : n_A = 40 := by sorry
def condition_2 : n_B = 30 := by sorry
def condition_3 : W_B = 60 := by sorry
def condition_4 : W_total = 54.285714285714285 := by sorry

-- Total number of students
def total_students : ℕ := n_A + n_B

-- Define total weight
def total_weight : ℝ := n_A * W_A + n_B * W_B

-- Define the problem statement, showing average weight of section A is 50 kg
theorem average_weight_section_A : 
    (total_weight / total_students) = W_total → W_A = 50 := by sorry

end average_weight_section_A_l407_407918


namespace ratio_of_areas_of_triangles_l407_407920

theorem ratio_of_areas_of_triangles 
  (a b c d e f : ℕ)
  (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (h4 : d = 9) (h5 : e = 40) (h6 : f = 41) : 
  (84 : ℚ) / (180 : ℚ) = 7 / 15 := by
  have hPQR : a^2 + b^2 = c^2 := by
    rw [h1, h2, h3]
    norm_num
  have hSTU : d^2 + e^2 = f^2 := by
    rw [h4, h5, h6]
    norm_num
  have areaPQR : (1/2 : ℚ) * a * b = 84 := by
    rw [h1, h2]
    norm_num
  have areaSTU : (1/2 : ℚ) * d * e = 180 := by
    rw [h4, h5]
    norm_num
  sorry

end ratio_of_areas_of_triangles_l407_407920


namespace koala_fiber_intake_l407_407824

theorem koala_fiber_intake (absorption_percentage : ℝ) (absorbed_fiber : ℝ) (total_fiber : ℝ) :
  absorption_percentage = 0.30 → absorbed_fiber = 12 → absorbed_fiber = absorption_percentage * total_fiber → total_fiber = 40 :=
by
  intros h1 h2 h3
  sorry

end koala_fiber_intake_l407_407824


namespace b_in_terms_of_a_max_value_f_geq_g_l407_407164

namespace ProofProblem

open Real

-- Definitions of f and g
def f (x a : ℝ) : ℝ := (1 / 2) * x^2 + 2 * a * x
def g (x a b : ℝ) : ℝ := 3 * a^2 * log x + b

-- Proving b in terms of a and its maximum value
theorem b_in_terms_of_a_max_value (a x0 : ℝ) (h : 0 < a) (h_f_tangent : f x0 a = g x0 a ( (1 / 2) * a^2 + 2 * a^2 - 3 * a^2 * log a )) 
        (h_g_tangent : (x0 + 2 * a) = (3 * a^2) / x0) : 
    (∀ t > 0, (5 / 2) * t^2 - 3 * t^2 * log t ≤ (3 / 2) * exp (2 / 3)) :=
sorry

-- Proving f(x) ≥ g(x) for all x > 0
theorem f_geq_g (a : ℝ) (h : 0 < a) (b : ℝ)
    (h_b : b = (5 / 2) * a^2 - 3 * a^2 * log a) :
    ∀ x > 0, f x a ≥ g x a b :=
sorry

end ProofProblem

end b_in_terms_of_a_max_value_f_geq_g_l407_407164


namespace graph_of_inverse_function_does_not_intersect_axes_l407_407166

theorem graph_of_inverse_function_does_not_intersect_axes :
  ∀ (x : ℝ), (x ≠ 0) → (x * (-5 / x) ≠ 0) ∧ (-5 / x ≠ 0) :=
by
  -- assume x is a real number and x ≠ 0
  intros x hx_ne_0,
  split,
  -- show that x * (-5 / x) ≠ 0
  {
    exact (mul_ne_zero hx_ne_0 (div_ne_zero (neg_ne_zero.mpr (ne_of_lt (by norm_num : (-5 : ℝ) < 0))) hx_ne_0)),
  },
  -- show that -5 / x ≠ 0
  {
    exact (div_ne_zero (neg_ne_zero.mpr (ne_of_lt (by norm_num : (-5 : ℝ) < 0))) hx_ne_0),
  }

end graph_of_inverse_function_does_not_intersect_axes_l407_407166


namespace penguins_fish_l407_407969

theorem penguins_fish:
  ∀ (a : Fin 10 → ℕ), (∑ i, a i = 50) →  (∀ i, a i ≥ 1) → ∃ (i j : Fin 10), i ≠ j ∧ a i = a j :=
by
  sorry

end penguins_fish_l407_407969


namespace arithmetic_geometric_sequences_l407_407833

noncomputable def a_n (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
noncomputable def b_n (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

theorem arithmetic_geometric_sequences 
  (a₁ : ℝ) (d : ℝ) (b₁ q : ℝ)
  (h_b1: b₁ = a₁^2)
  (h_b2: b_n b₁ q 2 = a_n a₁ d 2 ^ 2)
  (h_b3: b_n b₁ q 3 = a_n a₁ d 3 ^ 2)
  (h_lt: a₁ < a_n a₁ d 2)
  (h_lim: ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, | (finset.sum (finset.range n) (λ i, b_n b₁ q (i + 1))) - (√2 + 1)| < ε) :
  a₁ = -√2 ∧ d = 2*√2 - 2 := 
by 
  sorry

end arithmetic_geometric_sequences_l407_407833


namespace scientific_notation_of_thirteen_point_seven_million_l407_407319

-- Define the conditions for scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (num : ℝ) : Prop :=
  (1 ≤ |a| ∧ |a| < 10) ∧ num = a * (10 ^ n)

-- Define the given number
def thirteen_point_seven_million := 13.7 * 10^6

-- Statement to prove
theorem scientific_notation_of_thirteen_point_seven_million : 
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n thirteen_point_seven_million ∧ a = 1.37 ∧ n = 7 := sorry

end scientific_notation_of_thirteen_point_seven_million_l407_407319


namespace circle_x_intersect_is_6_l407_407513

noncomputable def circle_x_intersect : ℝ :=
  let center := (2, 2) in
  let radius := 2 * Real.sqrt 5 in
  let circle_eq := λ x y, (x - 2) ^ 2 + (y - 2) ^ 2 = 20 in
  let x_intercepts := 
    { x : ℝ // circle_eq x 0 } in
  { x in x_intercepts }.val

theorem circle_x_intersect_is_6 :
  circle_x_intersect = 6 :=
sorry

end circle_x_intersect_is_6_l407_407513


namespace binomial_30_3_l407_407584

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l407_407584


namespace measure_of_angle_A_smallest_side_length_l407_407279

-- Definitions for conditions of the triangle ABC
variables (a b c : ℝ) -- sides opposite to angles A, B, and C respectively
variables (A B C : ℝ) -- angles of the triangle
variables (sin cos : ℝ → ℝ) -- trigonometric functions

-- Question 1: Prove that A = 2π / 3 given the condtion
theorem measure_of_angle_A (h : sqrt 3 * a * sin C - c * (2 + cos A) = 0) : A = 2 * π / 3 :=
sorry

-- Question 2: Prove the length of the smallest side is 1 given conditions
theorem smallest_side_length
  (hA : A = 2 * π / 3)
  (h_longest : sqrt 13 = max a (max b c)) 
  (h_sin : sin C = 3 * sin B) : min a (min b c) = 1 :=
sorry

end measure_of_angle_A_smallest_side_length_l407_407279


namespace find_b_l407_407646

open Real

theorem find_b (b : ℝ) (h : b + ⌈b⌉ = 21.5) : b = 10.5 :=
sorry

end find_b_l407_407646


namespace functions_are_identical_l407_407477

def f1 (x : ℝ) : ℝ := 1
def f2 (x : ℝ) : ℝ := x^0

theorem functions_are_identical : ∀ (x : ℝ), f1 x = f2 x :=
by
  intro x
  simp [f1, f2]
  sorry

end functions_are_identical_l407_407477


namespace problem_l407_407530

-- Definitions of the function g and its values at specific points
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

-- Conditions given in the problem
theorem problem (d e f : ℝ)
  (h0 : g d e f 0 = 8)
  (h1 : g d e f 1 = 5) :
  d + e + 2 * f = 13 :=
by
  sorry

end problem_l407_407530


namespace floor_sum_sqrt_25_l407_407005

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l407_407005


namespace sum_of_floors_of_square_roots_l407_407020

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l407_407020


namespace sequence_sum_l407_407424

theorem sequence_sum (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n ≥ 2, S n - S (n - 1) = 2 * n - 1) (h2 : S 2 = 3) : 
  a 1 + a 3 = 5 := 
by 
  -- Define \(S_1\) from the given \( S_2 \)
  have hS1 : S 1 = 0, 
  {
    rw ← nat.sub_eq_zero_iff_le at h1,
    specialize h1 2 (by norm_num),
    rw h2 at h1,
    linarith
  },
  -- Define \(a_1\) and \(a_3\)
  have ha1 : a 1 = S 1 := by rw [hS1],
  have hS3 : S 3 = S 2 + 5 := 
  {
    specialize h1 3 (by norm_num),
    rw ← nat.succ_eq_add_one 2 S 3 S 2 at h1,
    linarith
  },
  have ha3 : a 3 = S 3 - S 2 := by rw [hS3, h2],
  -- Calculate \(a_1 + a_3\)
  rw [ha1, ha3],
  norm_num,
  sorry   -- Skip proof details

end sequence_sum_l407_407424


namespace Geli_pushups_total_l407_407675

-- Define conditions
def first_day_pushups : ℕ := 10
def additional_pushups : ℕ := 5
def total_days : ℕ := 3

-- Define pushups for each day
def pushups_day (n : ℕ) : ℕ :=
  first_day_pushups + (n - 1) * additional_pushups

-- Define the sum of pushups over the week
def total_pushups_week : ℕ := 
  (List.range total_days).sum (λ d, pushups_day (d + 1))

-- State the proof problem
theorem Geli_pushups_total : total_pushups_week = 45 := 
sorry

end Geli_pushups_total_l407_407675


namespace total_pears_picked_l407_407990

theorem total_pears_picked (alyssa : ℕ) (nancy : ℕ) (h_alyssa : alyssa = 42) (h_nancy : nancy = 17) : alyssa + nancy = 59 :=
by
  rw [h_alyssa, h_nancy]
  exact rfl

end total_pears_picked_l407_407990


namespace count_solutions_correct_l407_407337

def g (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

theorem count_solutions_correct :
  ∃ (count : ℕ), count = 6 ∧ ∀ x, -1 ≤ x ∧ x ≤ 1 → g (g (g x)) = g x → count > 0 :=
sorry

end count_solutions_correct_l407_407337


namespace pencils_per_student_l407_407432

theorem pencils_per_student (num_students total_pencils : ℕ)
  (h1 : num_students = 4) (h2 : total_pencils = 8) : total_pencils / num_students = 2 :=
by
  -- Proof omitted
  sorry

end pencils_per_student_l407_407432


namespace angle_BPC_lt_120_l407_407341

variables {A B C X Y P : Type}
variables (α β γ : ℝ)

-- Let ABC be a triangle with angle BAC < 60 degrees
def triangle (A B C : Type) := 
  ∃ (α β γ: ℝ), α < 60 ∧ 
    ∃ X Y, (segment CA + AX = segment CB + BX) ∧ 
           (segment BA + AY = segment BC + CY) ∧
           P.perpendicular_to PX AB ∧
           P.perpendicular_to PY AC

-- Define segments for conditions
def segment (a b : Type) := ℝ

-- Define perpendicular condition
def is_perpendicular (l1 l2 : Type) := ℝ

-- Proposition to prove
theorem angle_BPC_lt_120 (hABC: triangle A B C) 
  (hCAX : segment CA + segment AX = segment CB + segment BX)
  (hCAY : segment BA + segment AY = segment BC + segment CY)
  (hPX_perp_AB : is_perpendicular PX AB)
  (hPY_perp_AC : is_perpendicular PY AC) :
  ∠ B P C < 120 :=
sorry

end angle_BPC_lt_120_l407_407341


namespace sum_floor_sqrt_1_to_25_l407_407037

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l407_407037


namespace isosceles_not_all_congruent_l407_407482

theorem isosceles_not_all_congruent :
  (∀ Δ₁ Δ₂ : Triangle, isosceles Δ₁ ∧ isosceles Δ₂ → congruent Δ₁ Δ₂) = False :=
sorry

end isosceles_not_all_congruent_l407_407482


namespace lattice_points_with_distance_5_from_origin_l407_407796

theorem lattice_points_with_distance_5_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 42 :=
by
  sorry

end lattice_points_with_distance_5_from_origin_l407_407796


namespace sum_floor_sqrt_l407_407050

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l407_407050


namespace polar_to_cartesian_max_and_min_x_plus_y_l407_407727

-- Define the given polar equation and convert it to Cartesian equations
def polar_equation (rho θ : ℝ) : Prop :=
  rho^2 - 4 * (Real.sqrt 2) * rho * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Cartesian equation derived from the polar equation
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 4 * y + 6 = 0

-- Prove equivalence of the given polar equation and its equivalent Cartesian form for all ρ and \theta
theorem polar_to_cartesian (rho θ : ℝ) : 
  (∃ (x y : ℝ), polar_equation rho θ ∧ x = rho * Real.cos θ ∧ y = rho * Real.sin θ ∧ cartesian_equation x y) :=
by
  sorry

-- Property of points (x, y) on the circle defined by the Cartesian equation
def lies_on_circle (x y : ℝ) : Prop :=
  cartesian_equation x y

-- Given a point (x, y) on the circle defined by cartesian_equation, show bounds for x + y
theorem max_and_min_x_plus_y (x y : ℝ) (h : lies_on_circle x y) : 
  2 ≤ x + y ∧ x + y ≤ 6 :=
by
  sorry

end polar_to_cartesian_max_and_min_x_plus_y_l407_407727


namespace number_of_divisors_l407_407180

theorem number_of_divisors :
  let n := 2^4 * 3^2 * 5 in
  ∃ a ∈ {a : ℕ | a ∣ n}, card {a : ℕ | a ∣ n} = 30 := 
by
  let n := 2 ^ 4 * 3 ^ 2 * 5
  let divisors := {a : ℕ | a ∣ n}
  have h_divisors: ∀ x ∈ divisors, ∃ x y z, a = 2 ^ x * 3 ^ y * 5 ^ z ∧ 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 2 ∧ 0 ≤ z ∧ z ≤ 1 
  -- The proof would go here, but we skip it
  sorry 

end number_of_divisors_l407_407180


namespace sum_squares_of_solutions_is_3036_l407_407616

noncomputable def sum_squares_of_solutions : ℝ :=
  let pairs := { (x, y) : ℝ × ℝ | x = y^2 - 20 ∧ y = x^2 + x - 21 } in
  ∑ (p : ℝ × ℝ) in pairs, (p.1^2 + p.2^2)

theorem sum_squares_of_solutions_is_3036 : sum_squares_of_solutions = 3036 :=
  sorry

end sum_squares_of_solutions_is_3036_l407_407616


namespace number_of_factors_60_l407_407224

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l407_407224


namespace find_x_l407_407742

variable (x : ℝ)
def vector_a : ℝ × ℝ := (x, 2)
def vector_b : ℝ × ℝ := (x - 1, 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (h1 : dot_product (vector_a x + vector_b x) (vector_a x - vector_b x) = 0) : x = -1 := by 
  sorry

end find_x_l407_407742


namespace find_c_l407_407113

theorem find_c (x y c : ℝ) (h1 : x + y - 2 = 0) (h2 : x - 2y + 4 = 0) 
               (h3 : ∀ x y, (x + y - 2 = 0 ∧ x - 2y + 4 = 0) → (y = 3x + c)) : 
  c = 2 :=
by
  sorry

end find_c_l407_407113


namespace sum_of_floors_of_square_roots_l407_407023

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l407_407023


namespace probability_B_C_not_adjacent_A_l407_407284

-- Definitions for participants and the probability computation
def participants := {A, B, C, D, E} -- Set of participants

theorem probability_B_C_not_adjacent_A :
  let arrangements := {ls : List (participants) // ls.length = 5 ∧ ls.nodup} in
  let condition (ls : List participants) : Prop :=
    ¬(ls.adjacent A B ∨ ls.adjacent A C) in
  (∀ ls ∈ arrangements, condition ls) →
  (∃ p : ℚ, p = 3 / 10) :=
sorry

end probability_B_C_not_adjacent_A_l407_407284


namespace sum_of_floors_of_square_roots_l407_407014

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l407_407014


namespace complex_number_in_third_quadrant_l407_407402

theorem complex_number_in_third_quadrant :
  let z := (1 + complex.I) / ((1 - complex.I) * (1 - complex.I))
  -1/2 < z.re ∧ z.re <= 0 ∧ 0 < z.im ∧ z.im <= 1/2 ∧
  (z.re < 0 ∧ z.im < 0) :=
by
  let z : ℂ := (1 + complex.I) / ((1 - complex.I) * (1 - complex.I))
  have h1 : z = -1 / 2 + 1 / 2 * complex.I,
  {
    sorry,
  }
  let z_re := z.re
  let z_im := z.im
  have h2 : z_re = -1 / 2 ∧ z_im = 1 / 2,
  {
    sorry,
  }
  exact ⟨by linarith, by linarith, by linarith, by linarith⟩

end complex_number_in_third_quadrant_l407_407402


namespace cannot_be_arithmetic_progression_can_be_geometric_progression_l407_407562

theorem cannot_be_arithmetic_progression (a b c : ℝ) (ha : a = 2) (hb : b = real.sqrt 6) (hc : c = 4.5) : 
  ¬ ∃ d : ℝ, b = a + d ∧ c = a + 2 * d := by
  sorry

theorem can_be_geometric_progression (a b c : ℝ) (ha : a = 2) (hb : b = real.sqrt 6) (hc : c = 4.5) : 
  ∃ q : ℝ, q = real.sqrt (3 / 2) ∧ b = a * q ∧ c = a * q^2 := by
  sorry

end cannot_be_arithmetic_progression_can_be_geometric_progression_l407_407562


namespace geometric_series_mod_500_l407_407556

theorem geometric_series_mod_500 :
  let S := ∑ i in finset.range 1000, 9^i
  in S % 500 = 0 :=
by
  sorry

end geometric_series_mod_500_l407_407556


namespace rohan_food_expense_percentage_l407_407386

theorem rohan_food_expense_percentage (
  (savings_in_ruppees : ℕ) (salary_in_ruppees : ℕ)
  (house_rent_percentage : ℕ) (entertainment_percentage : ℕ) 
  (conveyance_percentage : ℕ) (savings_percentage : ℕ) :
  savings_in_ruppees = 2500 → 
  salary_in_ruppees = 12500 →
  house_rent_percentage = 20 →
  entertainment_percentage = 10 → 
  conveyance_percentage = 10 → 
  savings_percentage = 20 →
  savings_percentage = (2500 * 100) / 12500 →
  ∃ (food_percentage : ℕ), food_percentage = 40 :=
by 
  sorry

end rohan_food_expense_percentage_l407_407386


namespace lattice_points_count_l407_407782

theorem lattice_points_count :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2^2 + p.3^2 = 25}.to_finset.card = 30 :=
sorry

end lattice_points_count_l407_407782


namespace binom_30_3_l407_407609

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l407_407609


namespace range_of_g_l407_407086

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

-- State the theorem about the range of g(x)
theorem range_of_g : 
  set.range g = {-Real.pi / 3, Real.pi / 3} :=
sorry

end range_of_g_l407_407086


namespace num_ways_select_three_nonadjacent_l407_407277

-- Definitions based on the conditions
def select_from_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def non_adjacent (x y : ℕ) : Prop := abs (x - y) > 1

-- Lean statement to prove the equivalent problem
theorem num_ways_select_three_nonadjacent :
  ∃ s : set (set ℕ), s.card = 56 ∧ ∀ t ∈ s, t.card = 3 ∧ (∀ x y ∈ t, x ≠ y → non_adjacent x y) :=
sorry

end num_ways_select_three_nonadjacent_l407_407277


namespace lattice_points_with_distance_five_from_origin_l407_407803

theorem lattice_points_with_distance_five_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 6 :=
by
  sorry

end lattice_points_with_distance_five_from_origin_l407_407803


namespace problem1_collinear_problem2_k_l407_407838

variables {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Definitions of vectors a and b
variables (a b : α)
-- Non-degenerate conditions
variable (h_non_collinear : ¬Collinear ({0, a, b} : Set α))

-- Problem 1: Collinearity of A, B, C
variables (OA OB OC : α)
hypothesis h_OA : OA = 2 • a - b
hypothesis h_OB : OB = 3 • a + b
hypothesis h_OC : OC = a - 3 • b

-- Problem 2: Collinearity of A, C, D and finding k
variables (AB BC CD AC : α)
hypothesis h_AB : AB = a + b
hypothesis h_BC : BC = 2 • a - 3 • b
hypothesis h_CD : ∃ k : ℝ, CD = 2 • a - k • b

-- Collinearity of points A, C, and D
variables (AC : α)
hypothesis h_AC : AC = AB + BC
hypothesis h_collinear_ACD : Collinear ({0, AC, CD} : Set α)

-- Proof goals
theorem problem1_collinear : Collinear ({OA, OB, OC} : Set α) :=
sorry

theorem problem2_k : ∃ k : ℝ, (∀ (k = 4 / 3), h_collinear_ACD) :=
sorry

end problem1_collinear_problem2_k_l407_407838


namespace equation_of_line_length_of_AB_l407_407127

variable {x y : ℝ}

-- Definitions of the ellipse and midpoint conditions
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1
def midpoint_condition (A B : (ℝ × ℝ)) : Prop := (A.fst + B.fst) / 2 = 2 ∧ (A.snd + B.snd) / 2 = 1

-- First proof problem
theorem equation_of_line (A B : (ℝ × ℝ)) : 
  (ellipse A.fst A.snd) → (ellipse B.fst B.snd) → (midpoint_condition A B) →
  ∃ l : ℝ × ℝ → Prop, (∀ P : ℝ × ℝ, l P ↔ P.snd = -P.fst + 3) :=
by sorry

-- Second proof problem
theorem length_of_AB (A B : (ℝ × ℝ)) : 
  (ellipse A.fst A.snd) → (ellipse B.fst B.snd) → (midpoint_condition A B) →
  ∥A - B∥ = (4 / 3) * real.sqrt 3 :=
by sorry

end equation_of_line_length_of_AB_l407_407127


namespace num_factors_of_60_l407_407192

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l407_407192


namespace find_angle_B_l407_407748

variable (A B C a b c : ℝ)
variable (h1 : ∠A + ∠B + ∠C = π)
variable (h2 : ∃ (a b c : ℝ), a^2 + c^2 - b^2 = sqrt 3 * a * c)
variable (h3 : sin A^2 + sin C^2 - sin B^2 = sqrt 3 * sin A * sin C)

theorem find_angle_B (h : sin A^2 + sin C^2 - sin B^2 = sqrt 3 * sin A * sin C) : ∠B = π / 6 :=
sorry

end find_angle_B_l407_407748


namespace num_factors_of_60_l407_407258

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l407_407258


namespace sufficient_not_necessary_condition_l407_407963

noncomputable def has_negative_root (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x < 0

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∃ x : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ x < 0) ↔ (a < 0) :=
sorry

end sufficient_not_necessary_condition_l407_407963


namespace num_factors_60_l407_407210

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l407_407210


namespace distance_between_centers_of_externally_tangent_circles_l407_407416

noncomputable def external_tangent_distance (R r : ℝ) (hR : R = 2) (hr : r = 3) (tangent : R > 0 ∧ r > 0) : ℝ :=
  R + r

theorem distance_between_centers_of_externally_tangent_circles :
  external_tangent_distance 2 3 (by rfl) (by rfl) (by norm_num) = 5 :=
sorry

end distance_between_centers_of_externally_tangent_circles_l407_407416


namespace num_factors_of_60_l407_407259

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l407_407259


namespace angle_between_skew_line_and_plane_l407_407135

theorem angle_between_skew_line_and_plane
  (x : ℝ)
  (h₁ : x > 0 ∧ x < 2 * real.sqrt 5)
  (h₂ : P : ℝ)
  (h₃ : PN = x)
  (h₄ : ∃ N, N.position = min_position)
  : (∃ angle : ℝ, angle = real.arctan (real.sqrt 2 / 4)) :=
sorry

end angle_between_skew_line_and_plane_l407_407135


namespace num_factors_of_60_l407_407242

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l407_407242


namespace min_students_solved_both_l407_407426

/-- A simple mathematical proof problem to find the minimum number of students who solved both problems correctly --/
theorem min_students_solved_both (total_students first_problem second_problem : ℕ)
  (h₀ : total_students = 30)
  (h₁ : first_problem = 21)
  (h₂ : second_problem = 18) :
  ∃ (both_solved : ℕ), both_solved = 9 :=
by
  sorry

end min_students_solved_both_l407_407426


namespace unique_rectangles_l407_407133

theorem unique_rectangles (a b x y : ℝ) (h_dim : a < b) 
    (h_perimeter : 2 * (x + y) = a + b)
    (h_area : x * y = (a * b) / 2) : 
    (∃ x y : ℝ, (2 * (x + y) = a + b) ∧ (x * y = (a * b) / 2) ∧ (x < a) ∧ (y < b)) → 
    (∃! z w : ℝ, (2 * (z + w) = a + b) ∧ (z * y = (a * b) / 2) ∧ (z < a) ∧ (w < b)) :=
sorry

end unique_rectangles_l407_407133


namespace find_base_of_denominator_l407_407270

theorem find_base_of_denominator 
  (some_base : ℕ)
  (h1 : (1/2)^16 * (1/81)^8 = 1 / some_base^16) : 
  some_base = 18 :=
sorry

end find_base_of_denominator_l407_407270


namespace find_number_l407_407527

def number : ℝ := 3550
def quotient : ℝ := 1002 / 20.04

theorem find_number : number - quotient = 3500 := by
  sorry

end find_number_l407_407527


namespace problem1_part1_problem1_part2_problem2_l407_407177

-- Given definitions
def e1 : ℝ × ℝ := (1, 0)
def e2 : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (3 * e1.1 - 2 * e2.1, 3 * e1.2 - 2 * e2.2)
def b : ℝ × ℝ := (4 * e1.1 + e2.1, 4 * e1.2 + e2.2)

-- Dot product definition
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Magnitude definition
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Cosine of the angle definition
def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  (dot_product v1 v2) / ((magnitude v1) * (magnitude v2))

-- The statements to prove
theorem problem1_part1 : dot_product a b = 10 := by sorry

theorem problem1_part2 : magnitude (a.1 + b.1, a.2 + b.2) = 5 * Real.sqrt 2 := by sorry

theorem problem2 : cos_theta a b = 10 / Real.sqrt 221 := by sorry

end problem1_part1_problem1_part2_problem2_l407_407177


namespace lattice_points_with_distance_five_from_origin_l407_407809

theorem lattice_points_with_distance_five_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 6 :=
by
  sorry

end lattice_points_with_distance_five_from_origin_l407_407809


namespace binom_30_3_eq_4060_l407_407567

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l407_407567


namespace factors_of_60_l407_407235

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l407_407235


namespace find_y_l407_407083

noncomputable def y : ℝ := 12

theorem find_y (y : ℝ) (h1 : ceil y * y = 156) : y = 12 := by
  sorry

end find_y_l407_407083


namespace harmonic_series_numerator_divisible_by_prime_l407_407377

theorem harmonic_series_numerator_divisible_by_prime (p : ℕ) (hp : Nat.Prime p) (hp_gt2 : p > 2) :
  p ∣ (Nat.num (∑ k in Finset.range (p - 1), (1 : ℚ) / (k + 1))) :=
sorry

end harmonic_series_numerator_divisible_by_prime_l407_407377


namespace number_of_factors_of_60_l407_407206

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l407_407206


namespace geometric_seq_arith_mean_l407_407712

theorem geometric_seq_arith_mean 
  (b : ℕ → ℝ) 
  (r : ℝ) 
  (b_geom : ∀ n, b (n + 1) = r * b n)
  (h_arith_mean : b 9 = (3 + 5) / 2) :
  b 1 * b 17 = 16 :=
by
  sorry

end geometric_seq_arith_mean_l407_407712


namespace total_pushups_first_week_l407_407678

-- Definitions based on conditions
def first_day_pushups : ℕ := 10
def increment_per_day : ℕ := 5
def days_workout : ℕ := 3

-- Function to calculate the number of push-ups on a specific day
def pushups_on_day (day : ℕ) : ℕ :=
  first_day_pushups + (day - 1) * increment_per_day

-- Function to calculate total push-ups in the first week
def total_pushups_week : ℕ :=
  ∑ i in Finset.range days_workout, pushups_on_day (i + 1)

-- Theorem to prove the total number of push-ups in the first week
theorem total_pushups_first_week : total_pushups_week = 45 := by
  -- Proof will be provided here
  sorry

end total_pushups_first_week_l407_407678


namespace largest_value_x_l407_407092

-- Definition of the conditions
def equation (x : ℚ) : Prop :=
  (16 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 8 * x - 2

-- Statement of the proof 
theorem largest_value_x : ∀ x : ℚ, equation x → x ≤ 9 / 4 := sorry

end largest_value_x_l407_407092


namespace unique_function_existence_l407_407865

noncomputable def unique_function_f (f: ℝ+ → ℝ+) : Prop :=
  ∀ (x: ℝ+) , f (f x) = 6 * x - f x

theorem unique_function_existence (f: ℝ+ → ℝ+): (unique_function_f f) → (∀ x: ℝ+, f x = 2 * x) := 
by
  sorry

end unique_function_existence_l407_407865


namespace angle_C_is_135_l407_407325

variable {a b c S : ℝ}
variable (h1 : S = 1 / 4 * (c^2 - a^2 - b^2))
variable (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) -- Triangle inequality and positive side lengths

theorem angle_C_is_135 (h1 : S = 1 / 4 * (c^2 - a^2 - b^2)) : ∃ C: ℝ, C = 135 ∧ C = angle_of_triangle_with_side_lengths a b c := sorry

end angle_C_is_135_l407_407325


namespace min_students_proof_l407_407751

noncomputable def min_students :=
  let score_min (n : ℕ) := 70 * (n - 7),
      score_top := 7 * 120,
      total_score (n : ℕ) := 85 * n in
  ∀ (n : ℕ), (score_top + score_min n = total_score n) → n = 24

theorem min_students_proof :
  min_students :=
by
  sorry

end min_students_proof_l407_407751


namespace series_sum_ln2_minus_half_l407_407962

-- Definition of the general term of the series
def a_n (n : Nat) : ℝ := 1 / ((2 * n - 1) * (2 * n) * (2 * n + 1))

-- Statement of the theorem
theorem series_sum_ln2_minus_half : 
  ∑' n, a_n n = Real.log 2 - 1 / 2 :=
sorry

end series_sum_ln2_minus_half_l407_407962


namespace average_speed_round_trip_l407_407853

-- Define the conditions explicitly
def time_to_park := 45 -- minutes
def distance_to_park := 1.5 -- km
def time_park_to_school := 15 -- minutes
def distance_park_to_school := 0.5 -- km
def time_return_trip := 20 -- minutes

-- Define total time and distance
def total_time_minutes := time_to_park + time_park_to_school + time_return_trip -- 80 minutes
def total_time_hours := total_time_minutes / 60 -- in hours
def total_distance := (distance_to_park + distance_park_to_school) * 2 -- 4 km

-- Define the goal: the average speed for the round trip
theorem average_speed_round_trip : 
  total_distance / total_time_hours = 3 := 
by
  sorry

end average_speed_round_trip_l407_407853


namespace find_profits_range_of_x_relationship_y_x_maximize_profit_l407_407973

-- Declare the variables and conditions
variables (a b : ℝ) (x : ℕ) (y : ℝ)

-- Conditions on profits from selling masks
axiom cond1 : 80 * a + 45 * b = 21
axiom cond2 : 40 * a + 60 * b = 18

-- Establishing the first two parts as equations to prove
theorem find_profits :
  a = 0.15 ∧ b = 0.2 :=
by {
  -- Proof requires solving two linear equations
  sorry
}

-- Define the total profit function and the range of x
def total_profit (x : ℕ) := -0.05 * x + 400

-- Constraints on x
axiom x_limit1 : 2000 - x ≥ x
axiom x_limit2 : 2000 - x ≤ 3 * x

-- Range of x from constraints
theorem range_of_x : 500 ≤ x ∧ x ≤ 1000 :=
by {
  sorry
}

-- Proving the functional relationship between y and x
theorem relationship_y_x : ∀ x, y = total_profit x :=
by {
  -- From the profit function and definitions
  sorry
}

-- Proving maximum profit at a specific x
theorem maximize_profit : x = 500 ∧ 2000 - x = 1500 :=
by {
  -- Optimal values derived from conditions and maximizing function
  sorry
}

end find_profits_range_of_x_relationship_y_x_maximize_profit_l407_407973


namespace abs_inequality_solution_l407_407647

theorem abs_inequality_solution (x : ℝ) : 
  (|5 - 2*x| >= 3) ↔ (x ≤ 1 ∨ x ≥ 4) := sorry

end abs_inequality_solution_l407_407647


namespace angle_BPC_lt_120_l407_407342

variables {A B C X Y P : Type}
variables (α β γ : ℝ)

-- Let ABC be a triangle with angle BAC < 60 degrees
def triangle (A B C : Type) := 
  ∃ (α β γ: ℝ), α < 60 ∧ 
    ∃ X Y, (segment CA + AX = segment CB + BX) ∧ 
           (segment BA + AY = segment BC + CY) ∧
           P.perpendicular_to PX AB ∧
           P.perpendicular_to PY AC

-- Define segments for conditions
def segment (a b : Type) := ℝ

-- Define perpendicular condition
def is_perpendicular (l1 l2 : Type) := ℝ

-- Proposition to prove
theorem angle_BPC_lt_120 (hABC: triangle A B C) 
  (hCAX : segment CA + segment AX = segment CB + segment BX)
  (hCAY : segment BA + segment AY = segment BC + segment CY)
  (hPX_perp_AB : is_perpendicular PX AB)
  (hPY_perp_AC : is_perpendicular PY AC) :
  ∠ B P C < 120 :=
sorry

end angle_BPC_lt_120_l407_407342


namespace tucker_remaining_tissues_l407_407431

def BrandA_tissues_per_box := 160
def BrandB_tissues_per_box := 180
def BrandC_tissues_per_box := 200

def bought_A_boxes := 2
def bought_B_boxes := 3
def bought_C_boxes := 1

def used_A_tissues := 210
def used_B_tissues := 340
def used_C_tissues := 100
def given_away_B_boxes := 1

theorem tucker_remaining_tissues :
  let initial_A_tissues := bought_A_boxes * BrandA_tissues_per_box in
  let initial_B_tissues := bought_B_boxes * BrandB_tissues_per_box in
  let initial_C_tissues := bought_C_boxes * BrandC_tissues_per_box in
  let remaining_A_tissues := initial_A_tissues - used_A_tissues in
  let remaining_B_tissues := initial_B_tissues - used_B_tissues - given_away_B_boxes * BrandB_tissues_per_box in
  let remaining_C_tissues := initial_C_tissues - used_C_tissues in
  remaining_A_tissues + remaining_B_tissues + remaining_C_tissues = 230 := 
by
  sorry

end tucker_remaining_tissues_l407_407431


namespace Alice_minimum_speed_l407_407492

-- Definition of the problem in Lean 4
theorem Alice_minimum_speed (dist ab : ℝ) (bob_speed : ℝ) (alice_delay : ℝ) (alice_time : ℝ) :
  dist = 120 → bob_speed = 40 → alice_delay = 0.5 → alice_time = 3 - alice_delay → (dist / alice_time) = 48 → alice_speed > 48 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, ← h3] at h4
  -- Alice's speed must be greater than 48 mph
  sorry

end Alice_minimum_speed_l407_407492


namespace number_of_factors_60_l407_407226

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l407_407226


namespace estimate_white_balls_l407_407758

theorem estimate_white_balls
  (total_balls : ℕ)
  (trials : ℕ)
  (white_draws : ℕ)
  (proportion_white : ℚ)
  (hw : total_balls = 10)
  (ht : trials = 400)
  (hd : white_draws = 240)
  (hprop : proportion_white = 0.6) :
  ∃ x : ℕ, x = 6 :=
by
  sorry

end estimate_white_balls_l407_407758


namespace sum_floor_sqrt_1_to_25_l407_407036

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l407_407036


namespace not_divisible_59_l407_407362

theorem not_divisible_59 (x y : ℕ) (hx : ¬ (59 ∣ x)) (hy : ¬ (59 ∣ y)) 
  (h : (3 * x + 28 * y) % 59 = 0) : (5 * x + 16 * y) % 59 ≠ 0 :=
by
  sorry

end not_divisible_59_l407_407362


namespace area_of_PQRS_is_correct_l407_407863

noncomputable def areaOf_PQRS (EFGH : ℝ) (P Q R S : EFGH × EFGH): ℝ :=
  let side_length := sqrt EFGH
  let height := side_length * (sqrt 3 / 2)
  let diagonal := side_length + 2 * height
  1 / 2 * (diagonal ^ 2)

theorem area_of_PQRS_is_correct (EFGH : ℝ)
  (cond1 : ∀ (P Q R S : EFGH × EFGH), True)
  (cond2 : ∀ (T : ℝ), T = EFGH → T = 25)
  (cond3 : ∀ (P Q R S : EFGH × EFGH), True):
  areaOf_PQRS EFGH _ P Q R S = 50 + 25 * sqrt 3 :=
by
  have EFGH_area := cond2 reflectValue by sorry
  have _: EFGH_area = 25 := sorry
  sorry

end area_of_PQRS_is_correct_l407_407863


namespace num_factors_60_l407_407214

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l407_407214


namespace maxwell_meets_brad_time_l407_407854

theorem maxwell_meets_brad_time :
  ∀ (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) (start_diff : ℝ),
    distance = 14 ∧ maxwell_speed = 4 ∧ brad_speed = 6 ∧ start_diff = 1 →
      ∃ t : ℝ, (maxwell_speed * (t + start_diff) + brad_speed * t = distance) ∧ (t + start_diff = 2) :=
begin
  intros distance maxwell_speed brad_speed start_diff h,
  rcases h with ⟨hdistance, hmaxwell_speed, hbrad_speed, hstart_diff⟩,
  use 1,
  split,
  { have h1 : maxwell_speed * (1 + start_diff) + brad_speed * 1 = 14,
    { rw [hmaxwell_speed, hbrad_speed, hstart_diff],
      norm_num },
    exact h1 },
  { rw [hstart_diff],
    norm_num }
end

end maxwell_meets_brad_time_l407_407854


namespace binomial_30_3_l407_407577

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l407_407577


namespace shopkeeper_loss_percent_l407_407952

theorem shopkeeper_loss_percent (I : ℝ) (h1 : I > 0) : 
  (0.1 * (I - 0.4 * I)) = 0.4 * (1.1 * I) :=
by
  -- proof goes here
  sorry

end shopkeeper_loss_percent_l407_407952


namespace spherical_to_rect_coords_l407_407621

open Real

noncomputable def spherical_to_rect (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * sin phi * cos theta, rho * sin phi * sin theta, rho * cos phi)

theorem spherical_to_rect_coords :
  spherical_to_rect 3 (π / 4) (π / 6) = (3 * sqrt 2 / 4, 3 * sqrt 2 / 4, 3 * sqrt 3 / 2) :=
by
  sorry

end spherical_to_rect_coords_l407_407621


namespace num_factors_60_l407_407213

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l407_407213


namespace highest_cardinality_subset_l407_407108

def v2 (a : ℤ) : ℕ :=
  (Nat.find_greatest (λ k, 2^k ∣ a) a.nat_abs).get_or_else 0

theorem highest_cardinality_subset (n : ℕ) :
  ∃ (A : Set ℕ), (∀ x y ∈ A, x ≠ y → v2 (x - y) % 2 = 0) ∧
  A.card ≤ 2^(Nat.floor ((n + 1) / 2)) :=
sorry

end highest_cardinality_subset_l407_407108


namespace max_consecutive_integers_sum_l407_407455

theorem max_consecutive_integers_sum:
  ∃ k, ∀ n: ℕ, 3 + ∑ i in (range (n - 2)), (3 + i) ≤ 500 → k = 29 := by
sorry

end max_consecutive_integers_sum_l407_407455


namespace num_factors_60_l407_407189

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l407_407189


namespace cost_of_first_house_l407_407313

theorem cost_of_first_house (C : ℝ) (h₀ : 2 * C + C = 600000) : C = 200000 := by
  -- proof placeholder
  sorry

end cost_of_first_house_l407_407313


namespace Geli_pushups_total_l407_407677

-- Define conditions
def first_day_pushups : ℕ := 10
def additional_pushups : ℕ := 5
def total_days : ℕ := 3

-- Define pushups for each day
def pushups_day (n : ℕ) : ℕ :=
  first_day_pushups + (n - 1) * additional_pushups

-- Define the sum of pushups over the week
def total_pushups_week : ℕ := 
  (List.range total_days).sum (λ d, pushups_day (d + 1))

-- State the proof problem
theorem Geli_pushups_total : total_pushups_week = 45 := 
sorry

end Geli_pushups_total_l407_407677


namespace floor_sum_sqrt_1_to_25_l407_407066

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l407_407066


namespace necessary_but_not_sufficient_l407_407889

-- Define the conditions given in the problem
def condition_5_minus_m (m : ℝ) : Prop := 5 - m > 0
def condition_m_plus_3 (m : ℝ) : Prop := m + 3 > 0
def condition_inequality (m : ℝ) : Prop := 5 - m ≠ m + 3

-- Define the overall condition for m to represent the ellipse
def represents_ellipse (m : ℝ) : Prop :=
  condition_5_minus_m m ∧ condition_m_plus_3 m ∧ condition_inequality m

-- The necessary but not sufficient condition to prove
def necessary_condition (m : ℝ) : Prop := -3 < m ∧ m < 5

-- Theorem statement: 
theorem necessary_but_not_sufficient (m : ℝ) :
  necessary_condition m → represents_ellipse m → False :=
begin
  sorry
end

end necessary_but_not_sufficient_l407_407889


namespace largest_inscribed_square_side_length_l407_407308

theorem largest_inscribed_square_side_length (s : ℝ) (y : ℝ) :
  ( ∃ s y,
    let large_square_side := 12
    ∧ let triangles_side := s
    ∧ let inscribed_square_side := y
    ∧ let diagonal_length := large_square_side * Real.sqrt 2
    in
    2 * s = diagonal_length
    ∧ y = diagonal_length / 2 - s / Real.sqrt 2
    ∧ y = 6 * Real.sqrt 2 - 6 ) :=
by
  sorry

end largest_inscribed_square_side_length_l407_407308


namespace peter_savings_l407_407902

noncomputable def calc_discounted_price (original_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    original_price * (1 - discount_percentage / 100)

noncomputable def calc_savings (original_price : ℝ) (external_price : ℝ) : ℝ :=
    original_price - external_price

noncomputable def total_savings : ℝ :=
    let math_original := 45.0
    let math_discount := 20.0
    let science_original := 60.0
    let science_discount := 25.0
    let literature_original := 35.0
    let literature_discount := 15.0
    let math_external := calc_discounted_price math_original math_discount
    let science_external := calc_discounted_price science_original science_discount
    let literature_external := calc_discounted_price literature_original literature_discount
    let math_savings := calc_savings math_original math_external
    let science_savings := calc_savings science_original science_external
    let literature_savings := calc_savings literature_original literature_external
    math_savings + science_savings + literature_savings

theorem peter_savings :
  total_savings = 29.25 :=
by
    sorry

end peter_savings_l407_407902


namespace original_loaf_slices_l407_407629

-- Define the given conditions
def andy_slices_1 := 3
def andy_slices_2 := 3
def toast_slices_per_piece := 2
def pieces_of_toast := 10
def slices_left_over := 1

-- Define the variables
def total_andy_slices := andy_slices_1 + andy_slices_2
def total_toast_slices := toast_slices_per_piece * pieces_of_toast

-- State the theorem
theorem original_loaf_slices : 
  ∃ S : ℕ, S = total_andy_slices + total_toast_slices + slices_left_over := 
by {
  sorry
}

end original_loaf_slices_l407_407629


namespace faster_train_length_is_1320_l407_407443

noncomputable def length_of_faster_train (speed_faster_train_kmph speed_slower_train_kmph seconds_to_overtake) : ℝ :=
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := (relative_speed_kmph * 5) / 18
  relative_speed_mps * seconds_to_overtake

theorem faster_train_length_is_1320 :
  length_of_faster_train 162 18 33 = 1320 :=
by
  sorry

end faster_train_length_is_1320_l407_407443


namespace angle_APB_is_60_degrees_l407_407332

noncomputable def height_of_equilateral_triangle (s : ℝ) : ℝ :=
  (√3 / 2) * s

noncomputable def point_P_height (s : ℝ) : ℝ :=
  (1 / 3) * (height_of_equilateral_triangle s)

noncomputable def angle_APB (s : ℝ) : ℝ :=
  60 -- degrees

theorem angle_APB_is_60_degrees (s : ℝ) :
  ∀ (XYZ : Type) (P A B : XYZ),
  (s > 0) →
  (is_equilateral_triangle XYZ s) →
  (point P inside_triangle XYZ) →
  (height_point P XYZ = point_P_height s) →
  (distance P A = s) →
  (distance P B = s) →
  (angle P A B = angle_APB s) :=
by
  intro XYZ P A B h_pos is_eq_tri P_in_tri P_height PA_lens PB_lens
  sorry

end angle_APB_is_60_degrees_l407_407332


namespace brenda_total_erasers_l407_407553

theorem brenda_total_erasers (number_of_groups : ℕ) (erasers_per_group : ℕ) (h1 : number_of_groups = 3) (h2 : erasers_per_group = 90) : number_of_groups * erasers_per_group = 270 := 
by
  sorry

end brenda_total_erasers_l407_407553


namespace tan_half_angle_l407_407117

open Real

noncomputable def cos_alpha := sqrt 3 / 3
noncomputable def alpha_quadrant := (∀ a, α = 2 * π - a ∧ 0 < a ∧ a < π / 2)

theorem tan_half_angle (h1 : cos α = cos_alpha) (h2 : alpha_quadrant α) :
  tan (α / 2) = (sqrt 2 - sqrt 6) / 2 := by
  sorry

end tan_half_angle_l407_407117


namespace max_runs_one_day_match_l407_407285

theorem max_runs_one_day_match
  (overs : ℕ)
  (balls_per_over : ℕ)
  (max_runs_per_ball : ℕ)
  (no_wide : Bool)
  (no_extras : Bool)
  (no_overthrows : Bool)
  (no_wides : Bool)
  (overs_eq : overs = 50)
  (balls_per_over_eq : balls_per_over = 6)
  (max_runs_per_ball_eq : max_runs_per_ball = 6)
  : 50 * 6 * 6 = 1800 :=
by
  rw [overs_eq, balls_per_over_eq, max_runs_per_ball_eq]
  norm_num
  sorry

end max_runs_one_day_match_l407_407285


namespace num_factors_60_l407_407187

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l407_407187


namespace binom_30_3_l407_407595

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l407_407595


namespace binomial_30_3_l407_407578

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l407_407578


namespace unique_solution_l407_407084

noncomputable def func_prop (f : ℝ → ℝ) : Prop :=
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) ∧
  (∀ x ≥ 1, f (x + 1) = (f x)^2 / x - 1 / x)

theorem unique_solution (f : ℝ → ℝ) :
  func_prop f → ∀ x ≥ 1, f x = x + 1 :=
by
  sorry

end unique_solution_l407_407084


namespace meetings_percentage_correct_l407_407394

/-- Define the conditions of the problem -/
def work_day_length_hours : ℕ := 10
def first_meeting_minutes : ℕ := 60
def second_meeting_multiplier : ℚ := 1.5

/-- Define the total length of the work day in minutes -/
def work_day_length_minutes : ℕ := work_day_length_hours * 60

/-- Calculate the duration of the second meeting -/
def second_meeting_minutes : ℕ := (first_meeting_minutes * second_meeting_multiplier).toNat

/-- Calculate the total time spent in meetings -/
def total_meeting_time_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Calculate the fraction of the work day spent in meetings -/
def meeting_time_fraction : ℚ := total_meeting_time_minutes / work_day_length_minutes

/-- Calculate the percentage of the work day spent in meetings -/
def meeting_time_percentage : ℚ := meeting_time_fraction * 100

/-- The theorem to be proved: The percentage of the work day spent in meetings is 25% -/
theorem meetings_percentage_correct : meeting_time_percentage = 25 := sorry

end meetings_percentage_correct_l407_407394


namespace correct_statements_about_h_l407_407708

noncomputable def g : ℝ → ℝ := λ x, 2^x
noncomputable def f : ℝ → ℝ := λ x, Real.log (x) / Real.log 2 -- this is equivalent to log_2(x)
noncomputable def h : ℝ → ℝ := λ x, f (1 - |x|)

def symmetric_about_y_axis (h : ℝ → ℝ) := ∀ x : ℝ, h(x) = h(-x)
def monotonic_increasing_on (h : ℝ → ℝ) (a b : ℝ) := ∀ x y : ℝ, a < x → x < y → y < b → h(x) ≤ h(y)

theorem correct_statements_about_h :
  symmetric_about_y_axis h ∧ monotonic_increasing_on h (-1) 0 :=
by
  sorry

end correct_statements_about_h_l407_407708


namespace binom_30_3_eq_4060_l407_407606

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l407_407606


namespace prism_height_l407_407867

theorem prism_height
  (PQ PR : ℝ)
  (PQ_eq_PR : PQ = PR)
  (PQ_value : PQ = Real.sqrt 5)
  (volume : ℝ)
  (volume_value : volume = 25.000000000000004) :
  ∃ h : ℝ, h = 10 :=
by
  let area := (1 / 2) * PQ * PR
  have area_value : area = 2.5 := by
    rw [PQ_eq_PR, PQ_value, Real.sqrt_mul_self (by norm_num : (0 : ℝ) ≤ 5)]
    norm_num
  use volume / area
  rw [volume_value, area_value]
  norm_num
  sorry

end prism_height_l407_407867


namespace find_k_l407_407835

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (1, 1)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (1 + k, 2 + k)

theorem find_k (k : ℝ) (h : vector_b.1 * vector_c k.1 + vector_b.2 * vector_c k.2 = 0) : k = -3/2 :=
sorry

end find_k_l407_407835


namespace unique_belligerent_max_product_roots_q_at_3_l407_407981

def quadratic (q : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, q = λ x, 2 * (x - a) * (x - b)

def belligerent (q : ℝ → ℝ) : Prop :=
  quadratic q ∧ ∃ r1 r2 r3 r4 : ℝ, (q (q r1) = 0) ∧ (q (q r2) = 0) ∧ (q (q r3) = 0) ∧ (q (q r4) = 0)

noncomputable def unique_max_product_roots (q : ℝ → ℝ) : Prop := 
  belligerent q ∧ ∀ q' : ℝ → ℝ, belligerent q' → (∏ r in (root_set q), r) ≤ (∏ r in (root_set q'), r)

def root_set (p : ℝ → ℝ) : Finset ℝ :=
  { r | p r = 0 }

theorem unique_belligerent_max_product_roots_q_at_3 (q : ℝ → ℝ) (hq : unique_max_product_roots q) : 
  q 3 = 8 :=
sorry

end unique_belligerent_max_product_roots_q_at_3_l407_407981


namespace impossible_to_move_checkers_to_bottom_right_corner_l407_407368

noncomputable def symmetric_move (pos1 pos2 : ℕ × ℕ) (board_size : ℕ) : ℕ × ℕ :=
  ((2 * pos2.1 - pos1.1), (2 * pos2.2 - pos1.2))

def is_white_cell (pos : ℕ × ℕ) : Prop :=
  (pos.1 + pos.2) % 2 = 0

theorem impossible_to_move_checkers_to_bottom_right_corner (initial_positions : list (ℕ × ℕ)) :
  length initial_positions = 9 →
  (∀ (pos : ℕ × ℕ), pos ∈ initial_positions → pos.1 < 8 ∧ pos.2 < 8) →
  let final_positions := [(7, 7), (7, 6), (6, 7), (6, 6), (5, 5), (5, 6), (6, 5), (7, 5), (5, 7)] in
  let white_cells_initial := initial_positions.countp is_white_cell in
  let white_cells_final := final_positions.countp is_white_cell in
  white_cells_initial ≠ white_cells_final →
  false :=
by
  intros len_positions valid_positions white_cells_initial white_cells_final
  sorry

end impossible_to_move_checkers_to_bottom_right_corner_l407_407368


namespace tickets_spent_l407_407999

-- defining the initial ticket count and tickets left
def initial_tickets : ℕ := 98
def tickets_left : ℕ := 55

-- the problem statement: prove that the tickets spent is 43
theorem tickets_spent : initial_tickets - tickets_left = 43 :=
by
  exact eq.refl 43

end tickets_spent_l407_407999


namespace chickens_cheaper_than_buying_eggs_l407_407382

theorem chickens_cheaper_than_buying_eggs :
  ∃ W, W ≥ 80 ∧ 80 + W ≤ 2 * W :=
by
  sorry

end chickens_cheaper_than_buying_eggs_l407_407382


namespace union_of_A_and_B_l407_407345

section
variable {A B : Set ℝ}
variable (a b : ℝ)

def setA := {x : ℝ | x^2 - 3 * x + a = 0}
def setB := {x : ℝ | x^2 + b = 0}

theorem union_of_A_and_B:
  setA a ∩ setB b = {2} →
  setA a ∪ setB b = ({-2, 1, 2} : Set ℝ) := by
  sorry
end

end union_of_A_and_B_l407_407345


namespace vector_subtraction_l407_407726

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def vec_smul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_a : ℝ × ℝ := (3, 5)
def vec_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction : vec_sub vec_a (vec_smul 2 vec_b) = (7, 3) :=
by
  sorry

end vector_subtraction_l407_407726


namespace velocity_current_1_l407_407524

noncomputable def velocity_of_current (distance : ℝ) (speed_man : ℝ) (total_time : ℝ) : ℝ :=
let v := ((speed_man : ℝ) - (speed_man^2 - (distance / total_time) * (speed_man * total_time))^0.5) in
v

theorem velocity_current_1 :
  velocity_of_current 2.4 5 1 = 1 := 
by 
simp [velocity_of_current]
sorry

end velocity_current_1_l407_407524


namespace rice_difference_l407_407356

theorem rice_difference:
  ∀ (total : ℕ) (kept_fraction : ℚ) (kept : ℕ) (given : ℕ),
  total = 50 →
  kept_fraction = (7 : ℚ) / 10 →
  kept = total * kept_fraction →
  given = total - kept →
  (kept - given = 20) :=
by
  intros total kept_fraction kept given
  intros h_total h_fraction h_kept h_given
  sorry

end rice_difference_l407_407356


namespace smallest_N_for_99N_threes_l407_407102

theorem smallest_N_for_99N_threes (N : ℕ) (h : ∀ k : ℕ, 99 * N = 3 * 10^k + 3 * 10^(k-1) + ... + 3 * 10^0) :
  N = 3367 := 
sorry

end smallest_N_for_99N_threes_l407_407102


namespace R_squared_is_one_l407_407745

variables {α β : Type*} [AddCommGroup β]

def observations (n : ℕ) := fin n → (α × β)

noncomputable def R_squared {n : ℕ} (obs : observations n) (b : β) (a : β) (e : fin n → β) (yi_eq : ∀ i, snd (obs i) = b * fst (obs i) + a + e i) (ei_zero : ∀ i, e i = 0): Prop :=
  R_squared obs = 1

theorem R_squared_is_one {n : ℕ} (obs : observations n) (b : β) (a : β) (e : fin n → β) 
  (yi_eq : ∀ i, snd (obs i) = b * fst (obs i) + a + e i) 
  (ei_zero : ∀ i, e i = 0) : R_squared obs b a e yi_eq ei_zero :=
sorry

end R_squared_is_one_l407_407745


namespace sum_of_floors_of_square_roots_l407_407019

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l407_407019


namespace f_at_2_l407_407472

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4

-- State the theorem that we need to prove
theorem f_at_2 : f 2 = 2 := by
  -- the proof will go here
  sorry

end f_at_2_l407_407472


namespace binomial_30_3_l407_407580

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l407_407580


namespace sum_floor_sqrt_l407_407055

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l407_407055


namespace interpretation_equivalence_l407_407898

variable {α : Type} [LinearOrderedField α]

theorem interpretation_equivalence (a : α) (h : 0.8 * a = a * 0.8) :
  ∀ a, (0.8 * a = ∑ i in (fin a), (0.8 : α)) :=
by
  sorry

end interpretation_equivalence_l407_407898


namespace max_n_arithmetic_seq_l407_407881

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def abs_sum_eq (a : ℕ → ℤ) (n : ℕ) : Prop :=
  (∑ i in range(n+1), |a i|) = 2019 ∧
  (∑ i in range(n+1), |(a i) + 1|) = 2019 ∧
  (∑ i in range(n+1), |(a i) - 2|) = 2019

theorem max_n_arithmetic_seq (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_abs_sum : abs_sum_eq a 2019) :
  ∃ n, n = 50 :=
sorry

end max_n_arithmetic_seq_l407_407881


namespace g_value_l407_407149

variable {ℝ : Type*} [linear_ordered_field ℝ] 

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom f_defined (x : ℝ) : f x ≠ 0
axiom symmetry_f (x : ℝ) : f x = 2 - f (-x)
axiom inverse_g (x : ℝ) : g (f x) = x

-- Proof Problem Statement
theorem g_value (a : ℝ) : g(a) + g(2 - a) = 0 :=
sorry

end g_value_l407_407149


namespace num_factors_60_l407_407211

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l407_407211


namespace ellipse_solution_length_AB_l407_407694

noncomputable def ellipse_equation (a b : ℝ) (e : ℝ) (minor_axis : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ e = 3 / 4 ∧ 2 * b = minor_axis ∧ minor_axis = 2 * Real.sqrt 7

theorem ellipse_solution (a b : ℝ) (e : ℝ) (minor_axis : ℝ) :
  ellipse_equation a b e minor_axis →
  (a^2 = 16 ∧ b^2 = 7 ∧ (1 / a^2) = 1 / 16 ∧ (1 / b^2) = 1 / 7) :=
by 
  intros h
  sorry

noncomputable def area_ratio (S1 S2 : ℝ) : Prop :=
  S1 / S2 = 9 / 13

theorem length_AB (S1 S2 : ℝ) :
  area_ratio S1 S2 →
  |S1 / S2| = |(9 * Real.sqrt 105) / 26| :=
by
  intros h
  sorry

end ellipse_solution_length_AB_l407_407694


namespace cevas_theorem_l407_407961

theorem cevas_theorem (A B C P X Y Z : Point) (hP : is_in_triangle A B C P) 
  (hX : is_intersect AP BC X) (hY: is_intersect BP AC Y) (hZ: is_intersect CP AB Z) :
  ( BX / CX ) * ( CY / AY ) * ( AZ / BZ ) = 1 := 
sorry

end cevas_theorem_l407_407961


namespace distance_between_joe_and_pete_l407_407317

-- Define the conditions
def joe_speed := 0.133333333333 -- Joe's speed in km/min
def time := 40 -- time in minutes

-- Pete's speed is half of Joe's speed
def pete_speed := joe_speed / 2

-- Calculate distances
def joe_distance := joe_speed * time
def pete_distance := pete_speed * time

-- Total distance between Joe and Pete
def total_distance := joe_distance + pete_distance

-- The main statement to be proved
theorem distance_between_joe_and_pete : total_distance = 8 := by
  sorry -- Proof to be completed

end distance_between_joe_and_pete_l407_407317


namespace num_lattice_points_l407_407775

def is_lattice_point (p : ℤ × ℤ × ℤ) : Prop := 
  let (x, y, z) := p 
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points : 
  {p : ℤ × ℤ × ℤ | is_lattice_point p}.to_finset.card = 18 :=
sorry

end num_lattice_points_l407_407775


namespace range_of_k_l407_407143

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x^2 - 11 * x + (30 + k) = 0 → x > 5) → 0 < k ∧ k ≤ 1/4 :=
begin
  sorry
end

end range_of_k_l407_407143


namespace length_PF_equals_8_l407_407170

-- Define the parameters and conditions of the problem
def parabola : set (ℝ × ℝ) := {p | p.2^2 = 8 * p.1}
def focus := (2, 0)
def directrix (x : ℝ) := x = -2
def perpendicularToDirectrix (p : ℝ × ℝ) := p.1 = -2
def slopeOfAF := -√3

-- Define a point P and point A on the parabola that satisfy the given conditions
def point_A := (-2, 4 * √3)
def point_P := (6, 4 * √3)

-- Proof statement: Given the conditions, |PF| = 8
theorem length_PF_equals_8 (p_A : ℝ × ℝ) (p_P : ℝ × ℝ) (F : ℝ × ℝ) : 
  p_A = point_A → 
  p_P = point_P → 
  F = focus → 
  |(p_P.1 - F.1)| = 8 := 
by 
  sorry

end length_PF_equals_8_l407_407170


namespace lattice_points_with_distance_five_from_origin_l407_407804

theorem lattice_points_with_distance_five_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 6 :=
by
  sorry

end lattice_points_with_distance_five_from_origin_l407_407804


namespace binom_30_3_l407_407601

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l407_407601


namespace right_triangle_hypotenuse_l407_407911

-- Define the right triangle conditions and hypotenuse calculation
theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : b = a + 3) (h2 : 1 / 2 * a * b = 120) :
  c^2 = 425 :=
by
  sorry

end right_triangle_hypotenuse_l407_407911


namespace stool_height_l407_407543

/-- Prove that given the conditions about the ceiling height, Alice's height, her reach, and the position
    of the light bulb, the height of the stool required for Alice to reach the light bulb is 30 centimeters. -/
theorem stool_height (ceiling_height : ℕ) (light_bulb_below_ceiling : ℕ)
    (alice_height : ℕ) (alice_reach : ℕ) (stool_height : ℕ) :
  ceiling_height = 250 → light_bulb_below_ceiling = 15 →
  alice_height = 155 → alice_reach = 50 →
  alice_height + alice_reach + stool_height = ceiling_height - light_bulb_below_ceiling →
  stool_height = 30 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end stool_height_l407_407543


namespace system_solution_a_l407_407746

theorem system_solution_a (x y a : ℝ) (h1 : 3 * x + y = a) (h2 : 2 * x + 5 * y = 2 * a) (hx : x = 3) : a = 13 :=
by
  sorry

end system_solution_a_l407_407746


namespace count_possible_values_of_k_l407_407552

open Nat

def is_prime (n : ℕ) : Prop := Prime n

theorem count_possible_values_of_k :
  ∀ k : ℕ, 
    (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 107 ∧ p * q = k) → false :=
by
  intros k h
  obtain ⟨p, q, hp, hq, h_sum, h_prod⟩ := h
  sorry

end count_possible_values_of_k_l407_407552


namespace differential_y_l407_407649

-- Define the function y as given in the conditions
def y (x : ℝ) : ℝ := ln (abs (cos (sqrt x))) + sqrt x * tan (sqrt x)

-- State the theorem to find the differential dy
theorem differential_y (x : ℝ) : 
  (λ x, @differential _ _ _ _ y x) x = (1 / (2 * (cos (sqrt x))^2)) * differential x :=
by sorry

end differential_y_l407_407649


namespace no_pair_equal_area_perimeter_l407_407435

open Real EuclideanGeometry

def Circle (O : Point) (r : ℝ) : Prop := ∀ P : Point, dist O P = r

def inside (O : Point) (P Q : Point) : Prop := dist O P < dist O Q

noncomputable def k := Circle Center 2
noncomputable def k_a := Circle (Center_A) (radius_a)
noncomputable def k_b := Circle (Center_B) (radius_b)
noncomputable def k_c := Circle (Center_C) (radius_c)

def tangent_inside (k k_a k_b k_c O : Point) : Prop :=
  (inside O k_a ∧ inside O k_b ∧ inside O k_c) ∧
  (k_a.center = k.center ∧ k_b.center = k.center ∧ k_c.center = k.center)

def angles_ABC (A B C : Point) : Prop :=
  ∠ABC = 30 ∧ ∠BCA = 60 ∧ ∠CAB = 90

theorem no_pair_equal_area_perimeter :
  ∀ k k_a k_b k_c (O A B C : Point),
    k = Circle O 2 →
    k_a = Circle A 1 →
    k_b = Circle B 1 →
    k_c = Circle C 1 →
    tangent_inside O k k_a k_b k_c →
    angles_ABC A B C →
    ¬∃ (R1 R2 : Region), area R1 = area R2 ∧ perimeter R1 = perimeter R2 := 
sorry

end no_pair_equal_area_perimeter_l407_407435


namespace find_x_given_conditions_l407_407740

variable (x y z : ℝ)

theorem find_x_given_conditions
  (h1: x * y / (x + y) = 4)
  (h2: x * z / (x + z) = 9)
  (h3: y * z / (y + z) = 16)
  (h_pos: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_distinct: x ≠ y ∧ x ≠ z ∧ y ≠ z) :
  x = 384/21 :=
sorry

end find_x_given_conditions_l407_407740


namespace number_of_positive_factors_of_60_l407_407248

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l407_407248


namespace angle_relationship_l407_407304

variable {α : Type*} [LinearOrderedField α]

-- We introduce points A, B, C, D, E, F on the affine plane.
variables (A B C D E F : Point α)

-- Define the angles involved
variables (a b c : RealAngle)

-- Hypotheses abstracted from the problem
variable (AB_equal_AC : dist A B = dist A C)
variable (AB_AC_not_col : ∼Collinear A B C)

variable (DEF_isosceles_right : ∠D + ∠E + ∠F = 180° ∧ ∠D = ∠F)
variable (D_on_AB : LiesOn D (Line A B))
variable (E_on_BC : LiesOn E (Line B C))
variable (F_on_CA : LiesOn F (Line C A))

-- Denote the angles as per variables
variable (angle_BFD : ∠ B F D = a)
variable (angle_ADE : ∠ A D E = b)
variable (angle_FEC : ∠ F E C = c)

-- Proof goal
theorem angle_relationship (a b c : RealAngle) 
  (AB_equal_AC : dist A B = dist A C)
  (DEF_isosceles_right : ∠D + ∠E + ∠F = 180° ∧ ∠D = ∠F)
  (D_on_AB : LiesOn D (Line A B))
  (E_on_BC : LiesOn E (Line B C))
  (F_on_CA : LiesOn F (Line C A))
  (angle_BFD : ∠ B F D = a)
  (angle_ADE : ∠ A D E = b)
  (angle_FEC : ∠ F E C = c) : 
  b = a + 90° := 
sorry

end angle_relationship_l407_407304


namespace solve_fractional_equation_for_c_l407_407391

theorem solve_fractional_equation_for_c :
  (∃ c : ℝ, (c - 37) / 3 = (3 * c + 7) / 8) → c = -317 := by
sorry

end solve_fractional_equation_for_c_l407_407391


namespace binom_30_3_eq_4060_l407_407603

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l407_407603


namespace num_lattice_points_distance_five_l407_407811

open_locale classical 

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points_distance_five : 
  {n : ℕ | ∃ (x y z : ℤ), is_lattice_point x y z} = 78 :=
by sorry

end num_lattice_points_distance_five_l407_407811


namespace number_of_nonincreasing_abs_val_sequences_l407_407737

theorem number_of_nonincreasing_abs_val_sequences : 
  ∃ n : ℕ, n = 2^7 ∧ 
  ∀ (l : list ℤ), (l.perm ([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7] : list ℤ)) →
  (∀ (a b : ℤ), (|a| ≤ |b| → ∀ idx_a idx_b, list.index_of (Option.get idx_b) l ≤ list.index_of (Option.get idx_a) l)) → 
  n = 128 :=
sorry

end number_of_nonincreasing_abs_val_sequences_l407_407737


namespace arithmetic_mean_of_set_l407_407704

theorem arithmetic_mean_of_set {x : ℝ} (mean_eq_12 : (8 + 16 + 20 + x + 12) / 5 = 12) : x = 4 :=
by
  sorry

end arithmetic_mean_of_set_l407_407704


namespace find_two_digit_number_l407_407271

theorem find_two_digit_number (x y : ℕ) (h1 : 10 * x + y = 4 * (x + y) + 3) (h2 : 10 * x + y = 3 * x * y + 5) : 10 * x + y = 23 :=
by {
  sorry
}

end find_two_digit_number_l407_407271


namespace mean_score_for_exam_l407_407661

variable (M SD : ℝ)

-- Define the conditions
def condition1 : Prop := 58 = M - 2 * SD
def condition2 : Prop := 98 = M + 3 * SD

-- The problem statement
theorem mean_score_for_exam (h1 : condition1 M SD) (h2 : condition2 M SD) : M = 74 :=
sorry

end mean_score_for_exam_l407_407661


namespace lattice_points_with_distance_5_from_origin_l407_407800

theorem lattice_points_with_distance_5_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 42 :=
by
  sorry

end lattice_points_with_distance_5_from_origin_l407_407800


namespace sum_of_integers_abs_greater_than_1_and_less_than_3_5_l407_407912

theorem sum_of_integers_abs_greater_than_1_and_less_than_3_5 :
  (∑ x in ({y : ℤ | abs y > 1 ∧ abs y < 3.5}.to_finset), x) = 0 :=
by
  -- Proof goes here
  sorry

end sum_of_integers_abs_greater_than_1_and_less_than_3_5_l407_407912


namespace ellipse_definition_l407_407910

-- Define the fixed points A and B and their distance d
variables {A B P : ℝ × ℝ}
def d : ℝ := dist A B

-- Define the condition PA + PB = 2 * d
def ellipse_condition (P : ℝ × ℝ) : Prop :=
  dist P A + dist P B = 2 * d

-- The theorem stating P satisfies the ellipse condition implies P lies on the ellipse with foci A and B
theorem ellipse_definition : (∀ P, ellipse_condition P ↔ P ∈ set_of_points_on_ellipse A B d) :=
sorry

end ellipse_definition_l407_407910


namespace garden_arrangements_l407_407514

theorem garden_arrangements:
  (∃ n : Nat, ∃ r c : Nat, 4 * r + 3 * c = 60 ∧ n = (finset.Icc 0 15).filter (λ r, (60 - 4 * r) % 3 = 0).card ∧ n = 6) := sorry

end garden_arrangements_l407_407514


namespace quadratic_range_l407_407665

noncomputable def quadratic_condition (a m : ℝ) : Prop :=
  (a > 0) ∧ (a ≠ 1) ∧ (- (1 + 1 / m) > 0) ∧
  (3 * m^2 - 2 * m - 1 ≤ 0)

theorem quadratic_range (a m : ℝ) :
  quadratic_condition a m → - (1 / 3) ≤ m ∧ m < 0 :=
by sorry

end quadratic_range_l407_407665


namespace ratio_EG_GF_l407_407970

-- Define a midpoint function
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define conditions
variables (A B C : ℝ × ℝ) (E F G M : ℝ × ℝ)
variable (ratio_AE_AF : ℝ)
variable (ratio_EG_GF : ℝ)

-- Hypotheses
-- A at origin, B on x-axis, C on y-axis
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (12, 0)
def C : ℝ × ℝ := (0, 16)

-- M is the midpoint of BC
def M := midpoint B C

-- E and F are points on AC and AB respectively where AE = 2AF
def E : ℝ × ℝ := (0, 2 * F.1)

-- Line EF intersects AM at G
def collinear (P Q R : ℝ × ℝ) :=
  (Q.1 - P.1) * (R.2 - P.2) = (R.1 - P.1) * (Q.2 - P.2)

-- Given conditions for the problem
axiom AE_2AF : E.1 = 0 ∧ E.2 = 2 * F.2
axiom G_on_AM : collinear A M G
axiom G_on_EF : collinear E F G

-- Desired ratio
theorem ratio_EG_GF : ratio_EG_GF = 3 / 2 := 
  sorry

end ratio_EG_GF_l407_407970


namespace trajectory_of_point_Q_l407_407411

theorem trajectory_of_point_Q (m n : ℝ) (h : m^2 + n^2 = 1) : 
  let x := m - n in
  let y := 2 * m * n in
  x^2 + y = 1 :=
begin
  -- Proof omitted
  sorry
end

end trajectory_of_point_Q_l407_407411


namespace interest_for_20000_l407_407739

-- Definition of simple interest
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

variables (P1 P2 I1 I2 r : ℝ)
-- Given conditions
def h1 := (P1 = 5000)
def h2 := (I1 = 250)
def h3 := (r = I1 / P1)
-- Question condition
def h4 := (P2 = 20000)
def t := 1

theorem interest_for_20000 :
  P1 = 5000 →
  I1 = 250 →
  P2 = 20000 →
  r = I1 / P1 →
  simple_interest P2 r t = 1000 :=
by
  intros
  -- Proof goes here
  sorry

end interest_for_20000_l407_407739


namespace max_consecutive_integers_sum_le_500_l407_407457

def consecutive_sum (n : ℕ) : ℕ :=
  -- Formula for sum starting from 3
  (n * (n + 1)) / 2 - 3

theorem max_consecutive_integers_sum_le_500 : ∃ n : ℕ, consecutive_sum n ≤ 500 ∧ ∀ m : ℕ, m > n → consecutive_sum m > 500 :=
by
  sorry

end max_consecutive_integers_sum_le_500_l407_407457


namespace function_neither_odd_nor_even_l407_407159

def f (x : ℝ) : ℝ := x^2 + 6 * x

theorem function_neither_odd_nor_even : 
  ¬ (∀ x, f (-x) = f x) ∧ ¬ (∀ x, f (-x) = -f x) :=
by
  sorry

end function_neither_odd_nor_even_l407_407159


namespace probability_odd_divisor_23_factorial_l407_407899

theorem probability_odd_divisor_23_factorial :
  let n := 23!
  let num_divisors := (∀ p ∈ [2, 3, 5, 7, 11, 13, 17, 19, 23],
                       ∑ k in (1..∞), ⌊n / p^k⌋)
  let num_odd_divisors := (∀ p ∈ [3, 5, 7, 11, 13, 17, 19, 23],
                           ∑ k in (1..∞), ⌊n / p^k⌋)
  (num_odd_divisors / num_divisors) = 1 / 20 :=
begin
  sorry
end

end probability_odd_divisor_23_factorial_l407_407899


namespace volume_solid_correct_l407_407892

noncomputable def volume_of_solid : ℝ := 
  let area_rhombus := 1250 -- Area of the rhombus calculated from the bounded region
  let height := 10 -- Given height of the solid
  area_rhombus * height -- Volume of the solid

theorem volume_solid_correct (height: ℝ := 10) :
  volume_of_solid = 12500 := by
  sorry

end volume_solid_correct_l407_407892


namespace lattice_points_count_l407_407785

theorem lattice_points_count :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2^2 + p.3^2 = 25}.to_finset.card = 30 :=
sorry

end lattice_points_count_l407_407785


namespace promotional_pricing_plan_l407_407974

theorem promotional_pricing_plan (n : ℕ) : 
  (8 * 100 = 800) ∧ 
  (∀ n > 100, 6 * n < 640) :=
by
  sorry

end promotional_pricing_plan_l407_407974


namespace intersection_of_sets_l407_407732

noncomputable def setM : Set ℝ := { x | x + 1 > 0 }
noncomputable def setN : Set ℝ := { x | 2 * x - 1 < 0 }

theorem intersection_of_sets : setM ∩ setN = { x : ℝ | -1 < x ∧ x < 1 / 2 } := by
  sorry

end intersection_of_sets_l407_407732


namespace binomial_30_3_l407_407583

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l407_407583


namespace ratio_expenditure_l407_407525

variable (I : ℝ) -- Assume the income in the first year is I.

-- Conditions
def savings_first_year := 0.25 * I
def expenditure_first_year := 0.75 * I
def income_second_year := 1.25 * I
def savings_second_year := 2 * savings_first_year
def expenditure_second_year := income_second_year - savings_second_year
def total_expenditure_two_years := expenditure_first_year + expenditure_second_year

-- Statement to be proved
theorem ratio_expenditure 
  (savings_first_year : ℝ := 0.25 * I)
  (expenditure_first_year : ℝ := 0.75 * I)
  (income_second_year : ℝ := 1.25 * I)
  (savings_second_year : ℝ := 2 * savings_first_year)
  (expenditure_second_year : ℝ := income_second_year - savings_second_year)
  (total_expenditure_two_years : ℝ := expenditure_first_year + expenditure_second_year) :
  (total_expenditure_two_years / expenditure_first_year) = 2 := by
    sorry

end ratio_expenditure_l407_407525


namespace number_of_matching_parity_sequences_l407_407736

-- Definition of digits and parity property

def digits : Finset ℕ := Finset.range 10  -- {0, 1, ..., 9}

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
    
def matching_parity_sequence (n : ℕ) : Finset (Fin n → ℕ) :=
  {s | ∀ i, i < n - 1 → (is_even (s i) ↔ is_even (s (i + 1))) ∨ (is_odd (s i) ↔ is_odd (s (i + 1)))}
    
-- The Problem Statement
theorem number_of_matching_parity_sequences : 
  (matching_parity_sequence 7).card = 156250 :=
sorry

end number_of_matching_parity_sequences_l407_407736


namespace area_ABC_of_isosceles_right_triangle_DEF_l407_407549

def Triangle (α : Type) := 
    {p : α // true}  -- abstract point collection representation

def isIsoscelesRightTriangle {α : Type} (T : Triangle α) : Prop :=
    sorry  -- skipped geometry property definition

noncomputable def areaOfTriangle {α : Type} (T : Triangle α) : ℝ :=
    sorry  -- function to calculate the area of a triangle

theorem area_ABC_of_isosceles_right_triangle_DEF :
  ∀ (α : Type) (A B C D E F G : α)
  (DEF ABC : Triangle α),
  isIsoscelesRightTriangle DEF →
  areaOfTriangle DEF = 1 →
  (/* DEF hypotenuse lies on ABC hypotenuse */ sorry) →
  (/* segment ratio conditions and parallelism by geometry properties */ sorry) →
  areaOfTriangle ABC = 36 :=
by
  intros,
  sorry

end area_ABC_of_isosceles_right_triangle_DEF_l407_407549


namespace num_lattice_points_on_sphere_5_l407_407772

theorem num_lattice_points_on_sphere_5 :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.to_finset.card = 54 := 
by
  sorry

end num_lattice_points_on_sphere_5_l407_407772


namespace climb_stairs_time_l407_407566

theorem climb_stairs_time (n a d : ℕ) (h0 : n = 7) (h1 : a = 25) (h2 : d = 10) : 
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  in S_n = 385 :=
by
  sorry

end climb_stairs_time_l407_407566


namespace f_odd_and_minimum_period_pi_l407_407891

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x

theorem f_odd_and_minimum_period_pi :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x) :=
  sorry

end f_odd_and_minimum_period_pi_l407_407891


namespace triangle_area_l407_407760

variable (A B C : ℝ)
variable (AB : ℝ) (sinA_plus_B sinA_minus_B : ℝ)

def sinA_plus_B := 3 / 5
def sinA_minus_B := 1 / 5
def AB := 3

theorem triangle_area :
  ∃ h : ℝ,
    AB * h / 2 = 3 * (Real.sqrt 6 + 2) / 2 :=
  sorry

end triangle_area_l407_407760


namespace math_problem_l407_407682

variable (a : ℝ)
noncomputable def problem := a = Real.sqrt 11 - 1
noncomputable def target := a^2 + 2 * a + 1 = 11

theorem math_problem (h : problem a) : target a :=
  sorry

end math_problem_l407_407682


namespace real_number_implies_a_eq_one_l407_407885

variable (a : ℂ)

def Z := (1 + complex.i) * (1 - a * complex.i)

theorem real_number_implies_a_eq_one (h : Z a ∈ set.real) : a = 1 := 
by 
  sorry

end real_number_implies_a_eq_one_l407_407885


namespace alexis_sew_skirt_time_l407_407542

theorem alexis_sew_skirt_time : 
  ∀ (S : ℝ), 
  (∀ (C : ℝ), C = 7) → 
  (6 * S + 4 * 7 = 40) → 
  S = 2 := 
by
  intros S _ h
  sorry

end alexis_sew_skirt_time_l407_407542


namespace f_comp_f_neg4_l407_407125

noncomputable theory

def f (x : ℝ) : ℝ :=
if x < 1 then x^2 else x - 1

theorem f_comp_f_neg4 : f (f (-4)) = 15 := by
  sorry

end f_comp_f_neg4_l407_407125


namespace smallest_N_for_99N_threes_l407_407101

theorem smallest_N_for_99N_threes (N : ℕ) (h : ∀ k : ℕ, 99 * N = 3 * 10^k + 3 * 10^(k-1) + ... + 3 * 10^0) :
  N = 3367 := 
sorry

end smallest_N_for_99N_threes_l407_407101


namespace original_loaf_had_27_slices_l407_407634

def original_slices : ℕ :=
  let slices_andy_ate := 3 * 2
  let slices_for_toast := 2 * 10
  let slices_left := 1
  slices_andy_ate + slices_for_toast + slices_left

theorem original_loaf_had_27_slices (n : ℕ) (slices_andy_ate : ℕ) (slices_for_toast : ℕ) (slices_left : ℕ) :
  slices_andy_ate = 6 → slices_for_toast = 20 → slices_left = 1 → n = slices_andy_ate + slices_for_toast + slices_left → n = 27 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

-- Verifying the statement
example : original_slices = 27 := by
  have h1 : 3 * 2 = 6 := rfl
  have h2 : 2 * 10 = 20 := rfl
  have h3 : 1 = 1 := rfl
  exact original_loaf_had_27_slices original_slices 6 20 1 h1 h2 h3 rfl

end original_loaf_had_27_slices_l407_407634


namespace vertex_of_parabola_l407_407623

theorem vertex_of_parabola (x y : ℝ) :
  (∀ x : ℝ, y = -x^2 + 2 * x + 3) → (∃ h k : ℝ, y = -(x - h)^2 + k ∧ h = 1 ∧ k = 4) :=
begin
  sorry
end

end vertex_of_parabola_l407_407623


namespace sum_arithmetic_sequence_100_to_110_l407_407467

theorem sum_arithmetic_sequence_100_to_110 :
  let a := 100
  let l := 110
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 1155 := by
  sorry

end sum_arithmetic_sequence_100_to_110_l407_407467


namespace num_lattice_points_on_sphere_5_l407_407768

theorem num_lattice_points_on_sphere_5 :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.to_finset.card = 54 := 
by
  sorry

end num_lattice_points_on_sphere_5_l407_407768


namespace cyclist_distance_l407_407517

theorem cyclist_distance
  (v t d : ℝ)
  (h1 : d = v * t)
  (h2 : d = (v + 1) * (t - 0.5))
  (h3 : d = (v - 1) * (t + 1)) :
  d = 6 :=
by
  sorry

end cyclist_distance_l407_407517


namespace sum_of_floors_of_square_roots_l407_407018

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l407_407018


namespace sum_floor_sqrt_1_to_25_l407_407001

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l407_407001


namespace chord_intersection_length_l407_407884

noncomputable def chord_length (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) : ℝ :=
  let center := (1, 3)
  let radius := Real.sqrt 10
  let dist := (|1 - 9 + 3| / Real.sqrt 10)
  2 * Real.sqrt (radius^2 - dist^2)

theorem chord_intersection_length
  (line : ℝ → ℝ → Prop := λ x y, x - 3 * y + 3 = 0)
  (circle : ℝ → ℝ → Prop := λ x y, (x - 1)^2 + (y - 3)^2 = 10) :
  chord_length line circle = Real.sqrt 30 := by
  sorry

end chord_intersection_length_l407_407884


namespace floor_sum_sqrt_1_to_25_l407_407065

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l407_407065


namespace ratio_ac_l407_407905

variable {a b c d : ℝ}

-- Given the conditions
axiom ratio_ab : a / b = 5 / 4
axiom ratio_cd : c / d = 4 / 3
axiom ratio_db : d / b = 1 / 5

-- The statement to prove
theorem ratio_ac : a / c = 75 / 16 :=
  by sorry

end ratio_ac_l407_407905


namespace geo_seq_sum_neg_six_l407_407765

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (a₁ q : ℝ), q ≠ 0 ∧ ∀ n, a n = a₁ * q^n

theorem geo_seq_sum_neg_six
  (a : ℕ → ℝ)
  (hgeom : geometric_sequence a)
  (ha_neg : a 1 < 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = -6 :=
  sorry

end geo_seq_sum_neg_six_l407_407765


namespace lattice_points_count_l407_407786

theorem lattice_points_count :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2^2 + p.3^2 = 25}.to_finset.card = 30 :=
sorry

end lattice_points_count_l407_407786


namespace magnitude_of_2a_minus_b_l407_407146

variables (a b : ℝ → ℝ → ℝ)
-- Assuming plane vectors are functions from ℝ^2

def dot_product (u v : ℝ → ℝ → ℝ) : ℝ :=
  (u 1 0) * (v 1 0) + (u 0 1) * (v 0 1) -- dot product in 2D

def magnitude (v : ℝ → ℝ → ℝ) : ℝ :=
  real.sqrt ((v 1 0) ^ 2 + (v 0 1) ^ 2) -- magnitude in 2D

variables (h₁ : dot_product a b = 0)
          (h₂ : magnitude a = 1)
          (h₃ : magnitude b = real.sqrt 3)

theorem magnitude_of_2a_minus_b : magnitude (λ x y, 2 * a x y - b x y) = real.sqrt 7 :=
sorry

end magnitude_of_2a_minus_b_l407_407146


namespace range_of_a_l407_407119

variable {x a : ℝ}

def p (a : ℝ) : set ℝ := {x : ℝ | abs (x - a) < 4}
def q : set ℝ := {x : ℝ | (x - 2) * (3 - x) > 0}

theorem range_of_a (a : ℝ) (h : (compl (p a)) ⊆ compl q) : -1 ≤ a ∧ a ≤ 6 :=
sorry

end range_of_a_l407_407119


namespace walnut_trees_planted_today_l407_407429

-- Define the number of walnut trees before planting
def walnut_trees_before_planting : ℕ := 22

-- Define the number of walnut trees after planting
def walnut_trees_after_planting : ℕ := 55

-- Define a theorem to prove the number of walnut trees planted
theorem walnut_trees_planted_today : 
  walnut_trees_after_planting - walnut_trees_before_planting = 33 :=
by
  -- The proof will be inserted here.
  sorry

end walnut_trees_planted_today_l407_407429


namespace Only_Prop2_Correct_l407_407334

variables (a b c : Line) (α β γ : Plane)

-- Propositions as separate variables
def Prop1 := (c ∥ α ∧ c ∥ β) → (α ∥ β)
def Prop2 := (α ∥ γ ∧ β ∥ γ) → (α ∥ β)
def Prop3 := (c ∥ α ∧ a ∥ c) → (a ∥ α)
def Prop4 := (a ∥ γ ∧ α ∥ γ) → (a ∥ α)

-- The main theorem stating that only Prop2 is correct
theorem Only_Prop2_Correct : Prop2 ∧ ¬Prop1 ∧ ¬Prop3 ∧ ¬Prop4 := by
  sorry

end Only_Prop2_Correct_l407_407334


namespace infinitary_image_empty_l407_407324

def S := { q : ℚ // q ≠ -1 ∧ q ≠ 0 ∧ q ≠ 1 }

def f (x : S) : S := ⟨x.val - 1 / x.val, by 
  have h := x.property
  cases h with h₀ h₁
  cases h₁ with h₀ h₁
  split
  { intro h_neg1, 
    have : 1 / x.val = x.val + 1, 
    { rw [← sub_eq_iff_eq_add, eq_comm] at h_neg1,
      exact h_neg1 },
    linarith },
  { split,
    { intro h_0, 
      linarith },
    { intro h_1, 
      have : 1 / x.val = x.val - 1, 
      { rw [← sub_eq_iff_eq_add'] at h_1,
        exact h_1 },
      linarith }
  }
sorry⟩

noncomputable def f_iter (n : ℕ) (x : S) : S :=
nat.iterate f n x

theorem infinitary_image_empty : (⋂n, Set.image (f_iter n) S) = ∅ :=
sorry

end infinitary_image_empty_l407_407324


namespace functions_are_identical_l407_407476

def f1 (x : ℝ) : ℝ := 1
def f2 (x : ℝ) : ℝ := x^0

theorem functions_are_identical : ∀ (x : ℝ), f1 x = f2 x :=
by
  intro x
  simp [f1, f2]
  sorry

end functions_are_identical_l407_407476


namespace num_lattice_points_on_sphere_5_l407_407774

theorem num_lattice_points_on_sphere_5 :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.to_finset.card = 54 := 
by
  sorry

end num_lattice_points_on_sphere_5_l407_407774


namespace arithmetic_sequence_general_formula_sum_of_special_sequence_terms_l407_407132

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a i

theorem arithmetic_sequence_general_formula :
  ∀ (a : ℕ → ℤ),
  is_arithmetic_sequence a →
  a 1 + 3 = 8 →
  sum_of_first_n_terms a 10 = 185 →
  ∀ n : ℕ, a n = 3 * n + 2 :=
by
  sorry

theorem sum_of_special_sequence_terms :
  ∀ (a b : ℕ → ℤ),
  is_arithmetic_sequence a →
  a 1 + 3 = 8 →
  sum_of_first_n_terms a 10 = 185 →
  b = λ n, a (2^n) →
  ∀ n : ℕ, (sum_of_first_n_terms b (n+1) = 6 * (2 ^ (n+1) - 1) + 2 * (n+1)) :=
by
  sorry

end arithmetic_sequence_general_formula_sum_of_special_sequence_terms_l407_407132


namespace find_y_coordinate_l407_407301

noncomputable def circle_radius_max := 2

def point1 := (2 : ℝ, y : ℝ)
def point2 := (-2 : ℝ, 0 : ℝ)

theorem find_y_coordinate (y : ℝ) (h1 : (dist point1 point2) = 2 * circle_radius_max) : y = 0 :=
by
  sorry

end find_y_coordinate_l407_407301


namespace num_factors_60_l407_407190

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l407_407190


namespace sum_floor_sqrt_1_to_25_l407_407032

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l407_407032


namespace contradiction_proof_l407_407933

theorem contradiction_proof (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) : ¬ (¬ (a > 0) ∨ ¬ (b > 0) ∨ ¬ (c > 0)) → false :=
by sorry

end contradiction_proof_l407_407933


namespace probability_xi_leq_0_l407_407711

-- Defining the random variable ξ and its distribution
variable {ξ : ℝ}
axiom normal_distribution : ∀ (μ σ : ℝ), ProbDist ξ (normal μ σ)

-- Given conditions
axiom xi_normal : ProbDist ξ (normal 2 σ)
axiom xi_leq_4 : P(ξ ≤ 4) = 0.66

-- The proof problem
theorem probability_xi_leq_0 : P (ξ ≤ 0) = 0.32 := sorry

end probability_xi_leq_0_l407_407711


namespace max_m_value_l407_407123

theorem max_m_value (a b m : ℝ) (ha : a > 0) (hb : b > 0) (H : (3/a + 1/b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
sorry

end max_m_value_l407_407123


namespace g_2002_eq_1_l407_407162

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ := λ x => f x + 1 - x)

axiom f_one : f 1 = 1
axiom f_inequality_1 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

theorem g_2002_eq_1 : g 2002 = 1 := by
  sorry

end g_2002_eq_1_l407_407162


namespace sally_last_10_shots_made_l407_407541

def sally_initial_shots : ℕ := 30
def sally_initial_success_rate : ℝ := 0.60
def sally_additional_shots : ℕ := 10
def sally_final_success_rate : ℝ := 0.65

theorem sally_last_10_shots_made (x : ℕ) 
  (h1 : sally_initial_success_rate * sally_initial_shots = 18)
  (h2 : sally_final_success_rate * (sally_initial_shots + sally_additional_shots) = 26) :
  x = 8 :=
by
  sorry

end sally_last_10_shots_made_l407_407541


namespace circle_center_l407_407088

theorem circle_center (x y : ℝ) : x^2 + 4 * x + y^2 - 6 * y - 12 = 0 → (∃ h k : ℝ, (x + 2) ^ 2 + (y - 3) ^ 2 = 25 ∧ h = -2 ∧ k = 3) :=
by
  intro h
  use [-2, 3]
  constructor
  {
    exact h
  }
  constructor
  {
    exact -2
  }
  {
    exact 3
  }

end circle_center_l407_407088


namespace part_I_part_IIa_part_IIb_part_IIc_l407_407346

noncomputable def f (x a : ℝ) : ℝ := exp x * (x^2 - (1 + a) * x + 1)

theorem part_I (a : ℝ) : (deriv (λ x, f x a)) 0 = 1 → a = -1 := 
by 
  sorry

theorem part_IIa (a : ℝ) : a = -1 → ∀ x : ℝ, (deriv (λ x, f x a)) x ≥ 0 := 
by 
  sorry

theorem part_IIb (a : ℝ) : a < -1 → 
  (∀ x, x < a ∨ x > -1 → (deriv (λ x, f x a)) x > 0) ∧ 
  (∀ x, a < x ∧ x < -1 → (deriv (λ x, f x a)) x < 0) :=
by 
  sorry

theorem part_IIc (a : ℝ) : a > -1 → 
  (∀ x, x < -1 ∨ x > a → (deriv (λ x, f x a)) x > 0) ∧ 
  (∀ x, -1 < x ∧ x < a → (deriv (λ x, f x a)) x < 0) :=
by 
  sorry

end part_I_part_IIa_part_IIb_part_IIc_l407_407346


namespace part1_part2_part3_l407_407173

-- Define the sequence {a_n}
def a_seq : ℕ → ℤ
| 1 := 1
| 2 := 2
| n + 1 := if n + 1 >= 3 then (if (n + 1) % 2 = 0 then 2 * a_seq n else a_seq n + a_seq (n - 1)) else 0

-- Part 1
theorem part1 (m : ℕ) (h1 : m ≥ 4) (h2 : a_seq m % 2 = 1) (h3 : a_seq m % 3 ≠ 0) : 
  a_seq m = 2 * a_seq (m - 2) + a_seq (m - 3) :=
sorry

-- Part 2
theorem part2 : 
  ∑ n in Finset.range 2013, (a_seq n / (a_seq (n + 1) * a_seq (n + 2))) < 1 / 2 :=
sorry

-- Part 3
theorem part3 : ∃ max_t, (∀ t, t ≤ max_t) ∧ max_t = 1343 :=
sorry

end part1_part2_part3_l407_407173


namespace number_of_factors_60_l407_407223

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l407_407223


namespace average_squared_sums_100_l407_407989

def sequence := list (ℤ)

def A (n : ℕ) : set (sequence) :=
  { seq | seq.length = n ∧ ∀ x ∈ seq, x = -1 ∨ x = 1 }

def sum_seq (s : sequence) : ℤ :=
  list.sum s

def M (n : ℕ) : ℝ :=
  (∑ s in A n, (sum_seq s)^2 : ℝ) / (2^n : ℝ)

theorem average_squared_sums_100 : M 100 = 100 :=
sorry

end average_squared_sums_100_l407_407989


namespace intersection_A_B_l407_407699

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | 0 < x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l407_407699


namespace intersection_eq_l407_407731

def setA : Set ℝ := { x | abs (x - 3) < 2 }
def setB : Set ℝ := { x | (x - 4) / x ≥ 0 }

theorem intersection_eq : setA ∩ setB = { x | 4 ≤ x ∧ x < 5 } :=
by 
  sorry

end intersection_eq_l407_407731


namespace pentagon_area_proof_l407_407935

noncomputable def pentagonArea : ℝ :=
  let vertices := [(1, 1), (4, 4), (7, 2), (6, -3), (2, -2)]
  let term1 := List.sum (List.map (λ i, (fst (vertices.get! i)) * (snd (vertices.get! (i + 1) % 5))) (List.range 5))
  let term2 := List.sum (List.map (λ i, (snd (vertices.get! i)) * (fst (vertices.get! (i + 1) % 5))) (List.range 5))
  0.5 * abs (term1 - term2)

theorem pentagon_area_proof : pentagonArea = 14 :=
  sorry

end pentagon_area_proof_l407_407935


namespace ratio_milk_water_l407_407289

/--
In a mixture of 45 litres, the ratio of milk to water is 4:1.
Additional 3 litres of water is added to the mixture.
Prove the ratio of milk to water in the resulting mixture is 3:1.
-/
theorem ratio_milk_water (total_volume : ℕ) (initial_ratio_milk : ℕ) (initial_ratio_water : ℕ) (added_water : ℕ)
    (H_total_volume : total_volume = 45)
    (H_initial_ratio : initial_ratio_milk = 4 ∧ initial_ratio_water = 1) 
    (H_added_water : added_water = 3) :
    let initial_milk := (initial_ratio_milk * total_volume) / (initial_ratio_milk + initial_ratio_water)
    let initial_water := (initial_ratio_water * total_volume) / (initial_ratio_milk + initial_ratio_water)
    let new_water := initial_water + added_water
    let new_ratio := initial_milk : new_water
    new_ratio = 3 : 1 :=
by {
  sorry
}

end ratio_milk_water_l407_407289


namespace num_factors_of_60_l407_407240

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l407_407240


namespace closest_point_on_line_l407_407654

theorem closest_point_on_line (x y: ℚ) (h1: y = -4 * x + 3) (h2: ∀ p q: ℚ, y = -4 * p + 3 ∧ y = q * (-4 * p) - q * (-4 * 1 + 0)): (x, y) = (-1 / 17, 55 / 17) :=
sorry

end closest_point_on_line_l407_407654


namespace num_lattice_points_distance_five_l407_407813

open_locale classical 

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points_distance_five : 
  {n : ℕ | ∃ (x y z : ℤ), is_lattice_point x y z} = 78 :=
by sorry

end num_lattice_points_distance_five_l407_407813


namespace axis_and_centrally_symmetric_shapes_count_l407_407545

-- Definitions for shapes being axisymmetric and centrally symmetric.
def is_axisymmetric : Type -> Prop := sorry
def is_centrally_symmetric : Type -> Prop := sorry

-- Shapes considered in the problem
inductive Shape
| equilateral_triangle
| parallelogram
| rectangle
| rhombus
| square
| regular_pentagon

open Shape

-- Properties of each shape
def properties : Shape -> (Bool × Bool)
| equilateral_triangle := (true, false) -- (axisymmetric, centrally symmetric)
| parallelogram := (false, true)
| rectangle := (true, true)
| rhombus := (true, true)
| square := (true, true)
| regular_pentagon := (true, false)

-- Theorem stating the required proof problem
theorem axis_and_centrally_symmetric_shapes_count :
  (list.filter (λ s, (properties s).1 = true ∧ (properties s).2 = true) [equilateral_triangle, parallelogram, rectangle, rhombus, square, regular_pentagon]).length = 3 :=
by {
  sorry
}

end axis_and_centrally_symmetric_shapes_count_l407_407545


namespace binomial_coefficient_30_3_l407_407592

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l407_407592


namespace ELMO_value_l407_407827

def digits := {n : ℕ // n < 10}

variables (L E T M O : digits)

-- Conditions
axiom h1 : L.val ≠ 0
axiom h2 : O.val = 0
axiom h3 : (1000 * L.val + 100 * E.val + 10 * E.val + T.val) + (100 * L.val + 10 * M.val + T.val) = 1000 * T.val + L.val

-- Conclusion
theorem ELMO_value : E.val * 1000 + L.val * 100 + M.val * 10 + O.val = 1880 :=
sorry

end ELMO_value_l407_407827


namespace m_perp_beta_l407_407832

variable {Point Line Plane : Type}
variable {belongs : Point → Line → Prop}
variable {perp : Line → Plane → Prop}
variable {intersect : Plane → Plane → Line}

variable (α β γ : Plane)
variable (m n l : Line)

-- Conditions for the problem
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- Proof goal: proving m is perpendicular to β
theorem m_perp_beta : perp m β :=
by
  sorry

end m_perp_beta_l407_407832


namespace lattice_points_count_is_30_l407_407790

-- Define the function that counts the number of lattice points (x, y, z) such that x^2 + y^2 + z^2 = 25.
def count_lattice_points := {p : ℤ × ℤ × ℤ // p.1 ^ 2 + p.2.1 ^ 2 + p.2.2 ^ 2 = 25}.to_finset.card

-- The theorem to prove that the count is 30.
theorem lattice_points_count_is_30 : count_lattice_points = 30 :=
by
  sorry

end lattice_points_count_is_30_l407_407790


namespace mean_score_74_l407_407659

theorem mean_score_74 
  (M SD : ℝ)
  (h1 : 58 = M - 2 * SD)
  (h2 : 98 = M + 3 * SD) : 
  M = 74 :=
by
  sorry

end mean_score_74_l407_407659


namespace average_age_of_cricket_team_l407_407955

theorem average_age_of_cricket_team 
  (num_members : ℕ)
  (avg_age : ℕ)
  (wicket_keeper_age : ℕ)
  (remaining_avg : ℕ)
  (cond1 : num_members = 11)
  (cond2 : avg_age = 29)
  (cond3 : wicket_keeper_age = avg_age + 3)
  (cond4 : remaining_avg = avg_age - 1) : 
  avg_age = 29 := 
by 
  have h1 : num_members = 11 := cond1
  have h2 : avg_age = 29 := cond2
  have h3 : wicket_keeper_age = avg_age + 3 := cond3
  have h4 : remaining_avg = avg_age - 1 := cond4
  -- proof steps will go here
  sorry

end average_age_of_cricket_team_l407_407955


namespace minimize_y_l407_407842

noncomputable def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

theorem minimize_y (a b c : ℝ) : ∃ x : ℝ, (∀ x0 : ℝ, y x a b c ≤ y x0 a b c) ∧ x = (a + b + c) / 3 :=
by
  sorry

end minimize_y_l407_407842


namespace number_of_factors_60_l407_407221

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l407_407221


namespace barley_germination_prob_l407_407506

noncomputable def prob_germination (tests : List (ℕ × ℕ)) :=
  (tests.map (λ x, (x.snd.toRat / x.fst.toRat))).foldr (λ x y, x + y) 0 / tests.length

theorem barley_germination_prob :
  let tests := [(50, 47), (100, 96), (300, 284), (400, 380), (600, 571), (1000, 948)]
  prob_germination tests ≈ 0.95 :=
sorry

end barley_germination_prob_l407_407506


namespace num_subsets_of_set_l407_407909

theorem num_subsets_of_set : 
  ∀ (A : set Int), A = {-1, 0, 1} → (A.powerset.card = 8) :=
by
  intros A hA
  have : A.powerset.card = 8 := sorry
  exact this

end num_subsets_of_set_l407_407909


namespace index_difference_correct_l407_407663

noncomputable def index_diff (n k_f x_f n_m x_m : ℕ) : ℚ :=
  let index_females := ((n - k_f) / n.to_rat) * ((n - x_f) / n.to_rat)
  let index_males := ((n - n_m) / n.to_rat) * ((n - x_m) / n.to_rat)
  index_females - index_males

theorem index_difference_correct (h1 : 8 < 25) (h2 : 6 < 8)
  (n := 25) (k_f := 8) (x_f := 6) (n_m := 17) (x_m := 10) :
  index_diff n k_f x_f n_m x_m = (203 : ℚ) / 625 := by
    sorry

end index_difference_correct_l407_407663


namespace abs_p_minus_1_ge_2_l407_407321

theorem abs_p_minus_1_ge_2 (p : ℝ) (a : ℕ → ℝ) 
  (h₀ : a 0 = 1)
  (h₁ : a 1 = p)
  (h₂ : a 2 = p * (p - 1))
  (h₃ : ∀ n : ℕ, a (n + 3) = p * a (n + 2) - p * a (n + 1) + a n)
  (h₄ : ∀ n : ℕ, a n > 0)
  (h₅ : ∀ m n : ℕ, m ≥ n → a m * a n > a (m + 1) * a (n - 1)) :
  |p - 1| ≥ 2 :=
sorry

end abs_p_minus_1_ge_2_l407_407321


namespace tan_domain_exclusion_l407_407072

noncomputable def domain_of_tan : Set ℝ :=
  {x | ∀ k : ℤ, x ≠ -π/8 - π*k/2}

theorem tan_domain_exclusion :
  \{x ∈ ℝ | ∀ k : ℤ, x ≠ -π/8 - π*k/2\}\ :
  {x \in ℝ | (\frac{π}{4} - 2x) \not= (2k + 1)\frac{π}{2}, ∀ k: ℤ} :=
begin
  sorry
end

end tan_domain_exclusion_l407_407072


namespace original_loaf_slices_l407_407627

-- Define the given conditions
def andy_slices_1 := 3
def andy_slices_2 := 3
def toast_slices_per_piece := 2
def pieces_of_toast := 10
def slices_left_over := 1

-- Define the variables
def total_andy_slices := andy_slices_1 + andy_slices_2
def total_toast_slices := toast_slices_per_piece * pieces_of_toast

-- State the theorem
theorem original_loaf_slices : 
  ∃ S : ℕ, S = total_andy_slices + total_toast_slices + slices_left_over := 
by {
  sorry
}

end original_loaf_slices_l407_407627


namespace binom_expansion_sum_coeffs_l407_407738

theorem binom_expansion_sum_coeffs (a : ℝ) (coeffs : Fin 9 → ℝ) (h1 : (a - 1)^8 = ∑ i, coeffs i * (1:ℝ)^i)
    (h2 : coeffs 5 = 56) : ∑ i, coeffs i = 256 :=
by
  sorry

end binom_expansion_sum_coeffs_l407_407738


namespace repeating_decimal_sum_num_denom_l407_407474

noncomputable def repeating_decimal_to_fraction (n d : ℕ) (rep : ℚ) : ℚ :=
(rep * (10^d) - rep) / ((10^d) - 1)

theorem repeating_decimal_sum_num_denom
  (x : ℚ)
  (h1 : x = repeating_decimal_to_fraction 45 2 0.45)
  (h2 : repeating_decimal_to_fraction 45 2 0.45 = 5/11) : 
  (5 + 11) = 16 :=
by 
  sorry

end repeating_decimal_sum_num_denom_l407_407474


namespace range_of_f_l407_407718

noncomputable def f (x : Real) : Real :=
  if x ≤ 1 then 2 * x + 1 else Real.log x + 1

theorem range_of_f (x : Real) : f x + f (x + 1) > 1 ↔ (x > -(3 / 4)) :=
  sorry

end range_of_f_l407_407718


namespace range_of_m_l407_407273

theorem range_of_m {m : ℝ} (h : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ x^2 + 2 * x - m = 0) : 8 < m ∧ m < 15 :=
sorry

end range_of_m_l407_407273


namespace original_slices_proof_l407_407632

def original_slices (andy_consumption toast_slices leftover_slice: ℕ) : ℕ :=
  andy_consumption + toast_slices + leftover_slice

theorem original_slices_proof :
  original_slices (3 * 2) (10 * 2) 1 = 27 :=
by
  sorry

end original_slices_proof_l407_407632


namespace lattice_points_count_l407_407787

theorem lattice_points_count :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2^2 + p.3^2 = 25}.to_finset.card = 30 :=
sorry

end lattice_points_count_l407_407787


namespace necessary_and_sufficient_condition_l407_407692

variables {A B C P D E F : Point}
variables {L : ℝ}
variables {triangle ABC : Triangle}
variables [acute_triangle : is_acute_triangle ABC]
variables {perimeter : L = (side_length A B + side_length B C + side_length C A)}
variables {projections : P.proj_line BC = D ∧ P.proj_line CA = E ∧ P.proj_line AB = F}

theorem necessary_and_sufficient_condition (
  h1 : ∀ (P1 : Point), is_interior P1 ABC →
                       projections (P1) →
                       2 * (distance P1 D + distance P1 E + distance P1 F) = L) :
  (is_incenter I ABC ∧ is_circumcenter O ABC ∧ lies_on_line I O P) :=
begin
  sorry
end

end necessary_and_sufficient_condition_l407_407692


namespace sum_of_geometric_terms_l407_407142

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

theorem sum_of_geometric_terms {a : ℕ → ℝ} 
  (hseq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_sum135 : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end sum_of_geometric_terms_l407_407142


namespace num_factors_60_l407_407184

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l407_407184


namespace cosA_value_l407_407278

theorem cosA_value (A B C : ℝ) (a b c k : ℝ)
    (h1 : sin A / sin B = 4 / 3)
    (h2 : sin A / sin C = 4 / 2)
    (h3 : a = 4 * k)
    (h4 : b = 3 * k)
    (h5 : c = 2 * k) :
  cos A = -1 / 4 := 
begin
  sorry
end

end cosA_value_l407_407278


namespace longest_tape_l407_407354

theorem longest_tape (Minji Seungyeon Hyesu: ℝ) (h1: Minji = 0.74) (h2: Seungyeon = 13/20) (h3: Hyesu = 4/5):
  Hyesu > Minji ∧ Hyesu > Seungyeon := 
by
  have h4 : Seungyeon = 0.65 := by norm_num [h2]
  have h5 : Hyesu = 0.8 := by norm_num [h3]
  rw [h5, h1, h4]
  norm_num
  sorry

end longest_tape_l407_407354


namespace part_a_part_b_l407_407968

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def valid_permutation (P : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (P i + P (i + 1))) ∧
  ∀ i, P i ∈ (Finset.range 16).image (λ x => x + 1)

def valid_cyclic_permutation (C : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (C i + C (i + 1))) ∧
  is_perfect_square (C 15 + C 0) ∧
  ∀ i, C i ∈ (Finset.range 16).image (λ x => x + 1)

theorem part_a :
  ∃ P : Fin 16 → ℕ, valid_permutation P := sorry

theorem part_b :
  ¬ ∃ C : Fin 16 → ℕ, valid_cyclic_permutation C := sorry

end part_a_part_b_l407_407968


namespace number_of_positive_factors_of_60_l407_407245

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l407_407245


namespace train_crossing_time_l407_407503

variable (length_train : ℝ) (time_pole : ℝ) (length_platform : ℝ) (time_platform : ℝ)

-- Given conditions
def train_conditions := 
  length_train = 300 ∧
  time_pole = 14 ∧
  length_platform = 535.7142857142857

-- Theorem statement
theorem train_crossing_time (h : train_conditions length_train time_pole length_platform) :
  time_platform = 39 := sorry

end train_crossing_time_l407_407503


namespace geli_total_pushups_first_week_l407_407674

/-- Definitions of the number of push-ups Geli does each day given the problem conditions. -/
def pushups_on_day : ℕ → ℕ
| 1 => 10
| n+1 => pushups_on_day n + 5

/-- Definition of the total number of push-ups Geli does in her first week (3 days). -/
def total_pushups_first_week := pushups_on_day 1 + pushups_on_day 2 + pushups_on_day 3

/-- The main theorem stating that the total push-ups Geli does in her first week is 45. -/
theorem geli_total_pushups_first_week : total_pushups_first_week = 45 :=
by sorry

end geli_total_pushups_first_week_l407_407674


namespace quadratic_solution_value_l407_407624

theorem quadratic_solution_value (a b c : ℝ) (h : 5 * x^2 + 4 * x + 20 = 0) (c_val : ℝ) (d_val : ℝ) :
  c_val = -2 / 5 ∧ d_val = 4 * real.sqrt 6 / 5 → (c_val + d_val^2 = 86 / 25) :=
by
  intro H
  sorry

end quadratic_solution_value_l407_407624


namespace solve_equation_l407_407873

theorem solve_equation :
  ∀ x : ℝ, 4 * x * (6 * x - 1) = 1 - 6 * x ↔ (x = 1/6 ∨ x = -1/4) := 
by
  sorry

end solve_equation_l407_407873


namespace sweets_invariant_l407_407755

theorem sweets_invariant {n : ℕ} (S : ℕ := 1000) (boys girls : ℕ → Prop):
    (∀ i, boys i ∨ girls i) →
    ∀ perm : (Fin n) → (Fin n),
    (∀ i j, i < j → boys i ∧ girls j → sweets_taken (perm i) (perm j) = sweets_taken i j) → 
    total_sweets_taken boys (range n) = total_sweets_taken boys (perm '' (range n)) := 
by sorry

end sweets_invariant_l407_407755


namespace num_factors_60_l407_407217

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l407_407217


namespace geli_total_pushups_first_week_l407_407673

/-- Definitions of the number of push-ups Geli does each day given the problem conditions. -/
def pushups_on_day : ℕ → ℕ
| 1 => 10
| n+1 => pushups_on_day n + 5

/-- Definition of the total number of push-ups Geli does in her first week (3 days). -/
def total_pushups_first_week := pushups_on_day 1 + pushups_on_day 2 + pushups_on_day 3

/-- The main theorem stating that the total push-ups Geli does in her first week is 45. -/
theorem geli_total_pushups_first_week : total_pushups_first_week = 45 :=
by sorry

end geli_total_pushups_first_week_l407_407673


namespace shift_right_pi_over_12_l407_407759

-- Definitions of the functions involved
def sin3x (x : ℝ) : ℝ := Real.sin (3 * x)
def sin3x_minus_pi_over_4 (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

-- Theorem stating that shifting the graph to the right by π/12 units yields the desired function
theorem shift_right_pi_over_12 (x : ℝ) : sin3x (x - Real.pi / 12) = sin3x_minus_pi_over_4 x := 
sorry

end shift_right_pi_over_12_l407_407759


namespace max_consecutive_integers_sum_l407_407456

theorem max_consecutive_integers_sum:
  ∃ k, ∀ n: ℕ, 3 + ∑ i in (range (n - 2)), (3 + i) ≤ 500 → k = 29 := by
sorry

end max_consecutive_integers_sum_l407_407456


namespace sum_arithmetic_sequence_100_to_110_l407_407466

theorem sum_arithmetic_sequence_100_to_110 :
  let a := 100
  let l := 110
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 1155 := by
  sorry

end sum_arithmetic_sequence_100_to_110_l407_407466


namespace find_points_X_l407_407695

theorem find_points_X (circle : Set Point) (A B C D : Point) (a : ℝ) : 
  (Chord circle A B) ∧ (Chord circle C D) ∧ (length a) → ∃! X : Point, X ∈ circle ∧ (Chord circle A X) ∧ (Chord circle B X) ∧ (|EF| = a) :=
  sorry

end find_points_X_l407_407695


namespace solve_for_x_l407_407105

theorem solve_for_x (x : ℝ) : 
  2.5 * ((3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002 → 
  x = 2.5 :=
by 
  sorry

end solve_for_x_l407_407105


namespace isosceles_triangles_l407_407406

variables {A B C D M Q : Type} [EuclideanGeometry A B C D M Q]

-- conditions
axiom bisector_AD : ∀ (A B C D M : Point), is_bisector AD (triangle A B C)
axiom extension_AD_M : ∀ (A B C D M : Point), extension_circumcircle AD (triangle A B C) M
axiom incenter_Q : ∀ (A B C Q : Point), is_incenter Q (triangle A B C)

-- theorem we want to prove
theorem isosceles_triangles 
  (hyp1 : is_bisector AD (triangle A B C))
  (hyp2 : extension_circumcircle AD (triangle A B C) M)
  (hyp3 : is_incenter Q (triangle A B C)) :
  isosceles (triangle M B Q) ∧ isosceles (triangle M C Q) :=
sorry

end isosceles_triangles_l407_407406


namespace smallest_n_for_rotation_matrix_l407_407656

theorem smallest_n_for_rotation_matrix :
  let R := λ (θ : ℝ), matrix.of ![![real.cos θ, -real.sin θ], [real.sin θ, real.cos θ]]
  let θ := real.pi * 170 / 180
  let I := (1 : ℝ) • matrix.square_id ℝ (fin 2)
  ∃ n : ℕ, (0 < n) ∧ (R θ ^ n = I) ∧ (∀ m : ℕ, (0 < m) ∧ (R θ ^ m = I) → n ≤ m) :=
begin
  let n := 36,
  use n,
  split,
  { -- Proof of 0 < n
    exact nat.succ_pos 35 },
  split,
  { -- Proof of R θ ^ n = I
    -- This proof would involve showing that a 36*rotation by 170° equals a full rotation.
    sorry },
  { -- Proof that n is the smallest such number
    -- This would involve showing that any smaller number does not satisfy R θ ^ m = I
    sorry }
end

end smallest_n_for_rotation_matrix_l407_407656


namespace sum_integers_100_to_110_l407_407469

theorem sum_integers_100_to_110 : (∑ i in finset.Icc 100 110, i) = 1155 := by
  sorry

end sum_integers_100_to_110_l407_407469


namespace floor_sum_sqrt_25_l407_407012

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l407_407012


namespace area_parallelogram_new_vectors_l407_407351

variables {V : Type*} [AddCommGroup V] [Module ℝ V] [InnerProductSpace ℝ V]

-- Given vectors a and b
variables (a b : V)

-- Given condition
axiom area_parallelogram_ab : ∥a × b∥ = 15

-- Definition and theorem to be proved
theorem area_parallelogram_new_vectors :
  ∥((3 : ℝ) • a - b) × ((4 : ℝ) • a + (2 : ℝ) • b)∥ = 150 :=
by
  sorry

end area_parallelogram_new_vectors_l407_407351


namespace floor_sum_sqrt_1_to_25_l407_407063

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l407_407063


namespace num_factors_of_60_l407_407256

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l407_407256


namespace sum_of_slope_and_y_intercept_of_bisecting_line_l407_407925

-- Define the vertices of the triangle
def P := (0, 10)
def Q := (3, 0)
def R := (9, 0)

-- Define the midpoint S of line PR
def S := ((0 + 9) / 2, (10 + 0) / 2)

-- Define the slope of the line passing through Q and S
def slope := (5 - 0) / ((9 / 2) - 3)

-- Define the equation of the line passing through Q using the point-slope form
def line_y (x : ℝ) := slope * (x - 3)

-- Find the y-intercept of the line
def y_intercept := line_y 0

-- Calculate the sum of the slope and y-intercept
def sum_slope_y_intercept := slope + y_intercept

theorem sum_of_slope_and_y_intercept_of_bisecting_line :
  sum_slope_y_intercept = -(20 / 3) :=
by
  sorry

end sum_of_slope_and_y_intercept_of_bisecting_line_l407_407925


namespace slope_OM_l407_407171

noncomputable def ellipse_param_x (t : ℝ) : ℝ := 2 * Real.cos t
noncomputable def ellipse_param_y (t : ℝ) : ℝ := 4 * Real.sin t

def point_M : ℝ × ℝ := (ellipse_param_x (Real.pi / 3), ellipse_param_y (Real.pi / 3))
def point_O : ℝ × ℝ := (0, 0)

def slope (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

theorem slope_OM :
  slope point_O point_M = 2 * Real.sqrt 3 :=
by
  sorry

end slope_OM_l407_407171


namespace area_ratio_of_quads_l407_407883

theorem area_ratio_of_quads {A B C D K L M N : Point}
  (parallelogram_ABCD : is_parallelogram A B C D)
  (ratio_diagonals : ℝ)
  (h_ratio : ratio_diagonals = dist B D / dist A C)
  (non_neg_ratio : 0 ≤ ratio_diagonals ∧ ratio_diagonals ≤ 1)
  (bisectors_intersect : angle_bisectors_intersect K L M N A B C D) :
  area (quadrilateral K L M N) / area (parallelogram A B C D) = (2 * ratio_diagonals) / (1 + ratio_diagonals) ^ 2 := sorry

end area_ratio_of_quads_l407_407883


namespace lattice_points_with_distance_five_from_origin_l407_407806

theorem lattice_points_with_distance_five_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 6 :=
by
  sorry

end lattice_points_with_distance_five_from_origin_l407_407806


namespace length_of_AC_l407_407817

-- Given triangle ABC and medians AM and BN perpendicularly intersecting at centroid G,
-- with lengths AM = 15 and BN = 20, we want to prove the length of side AC is 100/3.

theorem length_of_AC (A B C M N G : Point)
  (hA : A ≠ G)
  (hB : B ≠ G)
  (hC : C ≠ G)
  (hM : M = midpoint B C)
  (hN : N = midpoint A C)
  (hAM : Segment A M)
  (hBN : Segment B N)
  (hCentroid : centroid A B C = G)
  (hPerp : perpendicular (Segment A M) (Segment B N))
  (hAM_len : length (Segment A M) = 15)
  (hBN_len : length (Segment B N) = 20) :
  length (Segment A C) = 100 / 3 := 
sorry

end length_of_AC_l407_407817


namespace graph_passes_through_point_l407_407408

theorem graph_passes_through_point (a : ℝ) (ha : 0 < a) (ha_ne_one : a ≠ 1) :
  let f := fun x : ℝ => a^(x - 3) + 2
  f 3 = 3 := by
  sorry

end graph_passes_through_point_l407_407408


namespace limit_sin_cos_l407_407555

open Real Topology

theorem limit_sin_cos (f : ℝ → ℝ) :
    (∀ x : ℝ, f x = (1 - (sin x)^3) / (cos x)^2) →
    tends_to f (𝓝 (π / 2)) (𝓝 (3 / 2)) :=
begin
  intros h,
  have h_lim : ∀ y : ℝ, f (y + π / 2) = (1 - (cos y)^3) / (sin y)^2,
  { intro y,
    calc
      f (y + π / 2) = (1 - (sin (y + π / 2))^3) / (cos (y + π / 2))^2 : by rw h
      ... = (1 - (cos y)^3) / (sin y)^2 : by
      { rw [sin_add_pi_div_two, cos_add_pi_div_two],
        ring }, },
  have lim_y_to_zero : filter.tendsto (λ y, (1 - (cos y)^3) / (sin y)^2) (𝓝 0) (𝓝 (3 / 2)),
  { rw [tendsto_iff_norm_tendsto_zero, eventually_eq],
    intros ε hε,
    simp only [sub_eq_zero],
    sorry },
  convert lim_y_to_zero.comp (tendsto_sub_self_const nt_0 (π / 2)),
  ext, norm_num,
end

end limit_sin_cos_l407_407555


namespace factors_of_60_l407_407232

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l407_407232


namespace circle_condition_k_l407_407403

theorem circle_condition_k (k : ℝ) : 
  (∃ (h : ℝ), (x^2 + y^2 - 2*x + 6*y + k = 0)) → k < 10 :=
by
  sorry

end circle_condition_k_l407_407403


namespace sum_of_floors_of_square_roots_l407_407015

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l407_407015


namespace percent_increase_perimeter_is_700_l407_407985

def side_length (n : ℕ) : ℝ :=
  if n = 1 then
    3
  else
    side_length (n - 1) * 2

noncomputable def perimeter (n : ℕ) : ℝ :=
  4 * side_length n

theorem percent_increase_perimeter_is_700 :
  let initial_perimeter := perimeter 1
  let fourth_perimeter := perimeter 4
  let percent_increase := ((fourth_perimeter - initial_perimeter) / initial_perimeter) * 100
  percent_increase = 700 :=
by
  sorry

end percent_increase_perimeter_is_700_l407_407985


namespace factors_of_60_l407_407233

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l407_407233


namespace num_lattice_points_l407_407777

def is_lattice_point (p : ℤ × ℤ × ℤ) : Prop := 
  let (x, y, z) := p 
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points : 
  {p : ℤ × ℤ × ℤ | is_lattice_point p}.to_finset.card = 18 :=
sorry

end num_lattice_points_l407_407777


namespace laser_beam_distance_l407_407521

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def reflection_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflection_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem laser_beam_distance :
  let A : ℝ × ℝ := (4, 6)
  let D' : ℝ × ℝ := (-8, 6)
  in distance A D' = 12 :=
by
  let A : ℝ × ℝ := (4, 6)
  let D' : ℝ × ℝ := (-8, 6)
  have dist : real.sqrt ((A.1 - D'.1)^2 + (A.2 - D'.2)^2) = 12
  {
    -- the actual calculations are omitted here
    sorry
  }
  exact dist

end laser_beam_distance_l407_407521


namespace twentieth_term_is_78_l407_407617

-- Define the arithmetic sequence parameters
def first_term : ℤ := 2
def common_difference : ℤ := 4

-- Define the function to compute the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- Formulate the theorem to prove
theorem twentieth_term_is_78 : nth_term 20 = 78 :=
by
  sorry

end twentieth_term_is_78_l407_407617


namespace quadratic_distinct_roots_l407_407668

theorem quadratic_distinct_roots (a : ℝ) : 
  (a > -1 ∧ a ≠ 3) ↔ 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (a - 3) * x₁^2 - 4 * x₁ - 1 = 0 ∧ 
    (a - 3) * x₂^2 - 4 * x₂ - 1 = 0 :=
by
  sorry

end quadratic_distinct_roots_l407_407668


namespace intersection_points_count_l407_407619

-- Definition of the two equations as conditions
def eq1 (x y : ℝ) : Prop := y = 3 * x^2
def eq2 (x y : ℝ) : Prop := y^2 - 6 * y + 8 = x^2

-- The theorem stating that the number of intersection points of the two graphs is exactly 4
theorem intersection_points_count : 
  ∃ (points : Finset (ℝ × ℝ)), (∀ p : ℝ × ℝ, p ∈ points ↔ eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 4 :=
by
  sorry

end intersection_points_count_l407_407619


namespace sum_floor_sqrt_l407_407057

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l407_407057


namespace ratio_xy_half_l407_407436

noncomputable def common_ratio_k (x y z : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) : ℝ := sorry

theorem ratio_xy_half (x y z k : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) :
  ∃ k, (x + 4) = 2 * k ∧ (y + 9) = k * (z - 3) ∧ (x + 5) = k * (z - 5) → (x / y) = 1 / 2 :=
sorry

end ratio_xy_half_l407_407436


namespace ratio_of_areas_of_triangles_l407_407921

theorem ratio_of_areas_of_triangles 
  (a b c d e f : ℕ)
  (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (h4 : d = 9) (h5 : e = 40) (h6 : f = 41) : 
  (84 : ℚ) / (180 : ℚ) = 7 / 15 := by
  have hPQR : a^2 + b^2 = c^2 := by
    rw [h1, h2, h3]
    norm_num
  have hSTU : d^2 + e^2 = f^2 := by
    rw [h4, h5, h6]
    norm_num
  have areaPQR : (1/2 : ℚ) * a * b = 84 := by
    rw [h1, h2]
    norm_num
  have areaSTU : (1/2 : ℚ) * d * e = 180 := by
    rw [h4, h5]
    norm_num
  sorry

end ratio_of_areas_of_triangles_l407_407921


namespace lattice_points_with_distance_5_from_origin_l407_407798

theorem lattice_points_with_distance_5_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 42 :=
by
  sorry

end lattice_points_with_distance_5_from_origin_l407_407798


namespace stratified_prob_equals_systematic_prob_l407_407664

-- Define the population capacity with a constraint
def population_capacity (m : ℕ) : Prop := m ≥ 2017

-- Define the probability of selection using systematic sampling
def systematic_prob := (3 : ℝ) / 2017

-- Define the probability of selection using stratified sampling
def stratified_prob := (3 : ℝ) / 2017

-- The theorem to prove the equality of probabilities
theorem stratified_prob_equals_systematic_prob (m : ℕ) (h_m : population_capacity m) :
  stratified_prob = systematic_prob := 
sorry

end stratified_prob_equals_systematic_prob_l407_407664


namespace proof_problem_l407_407404

/--
  Let M be a point such that its distance to the line x = -1 is equal to its distance from the point F(1, 0).
  (1) Prove that the trajectory C of point M is a parabola.
  (2) Let a line l₁ passing through point F with a slope of k intersect the parabola C at two points A and B such that |AB| = 6.
      Prove that the equation of l₁ is √2x - y - √2 = 0 and √2x + y - √2 = 0.
-/
theorem proof_problem 
  (M F : ℝ × ℝ) 
  (l : ℝ) 
  (x y k A B : ℝ) 
  (h₁ : distance M (1, 0) = distance M (1, 0) ) 
  (h₂ : M = (x, y)) 
  (h₃ : F = (1, 0)) 
  (h₄ : line x = -1) 
  (h₅ : line l₁ passes F with slope k) 
  (h₆ : l₁ intersects parabola C at A B) 
  (h₇ : |AB| = 6) : 
  (trajectory M = parabola ∧ 
  line l₁ = √2x - y - √2 = 0 ∨ √2x + y - √2 = 0) := 
sorry

end proof_problem_l407_407404


namespace calculate_power_sum_l407_407557

theorem calculate_power_sum (pi : ℝ) (hpi : pi ≠ 0) : (-pi)^0 + 2^(-2) = 5 / 4 :=
by sorry

end calculate_power_sum_l407_407557


namespace num_lattice_points_distance_five_l407_407814

open_locale classical 

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points_distance_five : 
  {n : ℕ | ∃ (x y z : ℤ), is_lattice_point x y z} = 78 :=
by sorry

end num_lattice_points_distance_five_l407_407814


namespace four_brothers_money_l407_407670

theorem four_brothers_money:
  ∃ (J W C T : ℕ), 
    J + 2 = W - 2 ∧ 
    W - 2 = 2 * C ∧ 
    2 * C = T / 2 ∧ 
    J + W + C + T = 45 := by
  exists 8 12 5 20
  simp
  sorry -- Proof goes here 

end four_brothers_money_l407_407670


namespace evaluate_infinite_series_l407_407082

theorem evaluate_infinite_series :
  (∑ k in (Set.Icc 1 (⊤ : ℕ)), (k^2 : ℝ) / 2^k) = 6 := sorry

end evaluate_infinite_series_l407_407082


namespace smallest_positive_period_and_monotonic_intervals_range_of_m_l407_407348

noncomputable def f (x : ℝ) : ℝ :=
  (sin (x + real.pi))^2 - (cos (x - real.pi / 3))^2

theorem smallest_positive_period_and_monotonic_intervals :
  (∀ x : ℝ, f(x) = f(x + real.pi)) ∧
  (∀ k : ℤ, ∀ x : ℝ, k * real.pi + real.pi / 6 ≤ x ∧ x ≤ k * real.pi + 2 * real.pi / 3 → 
              f ' (x) > 0) :=
sorry

theorem range_of_m :
  (∀ m : ℝ, (∀ x : ℝ, x ∈ set.Icc (- real.pi / 6) (real.pi / 4) → abs (f(x) - m) ≤ 2) ↔
             - 7 / 4 ≤ m ∧ m ≤ 3 / 2) :=
sorry

end smallest_positive_period_and_monotonic_intervals_range_of_m_l407_407348


namespace part1_part2_l407_407864

theorem part1 (a b h3 : ℝ) (C : ℝ) (h : 1 / h3 = 1 / a + 1 / b) : C ≤ 120 :=
sorry

theorem part2 (a b m3 : ℝ) (C : ℝ) (h : 1 / m3 = 1 / a + 1 / b) : C ≥ 120 :=
sorry

end part1_part2_l407_407864


namespace tsvetochny_street_naming_l407_407299

def exists_initial_street_naming (n m : ℕ) (h : m ≥ n + 1) 
  (streets : list (ℕ × ℕ)) (initial_naming : (ℕ × ℕ) → bool) 
  (renaming_action : ℕ → ((ℕ × ℕ) → bool) → ((ℕ × ℕ) → bool)) : Prop :=
  ∃ naming : ((ℕ × ℕ) → bool), 
    ¬ ∀ seq, 
      let final_naming := seq.foldl renaming_action naming in
      (∀ street, final_naming street = tt) ∨ (∀ street, final_naming street = ff)

-- Hypothetical streets and renaming action for formality of the Lean statement
-- These do not come from the solution and should be used only for demonstrating the structure

variable (n m : ℕ) (h : m ≥ n + 1)
variable (streets : list (ℕ × ℕ)) 
variable (initial_naming : (ℕ × ℕ) → bool)

-- a simple renaming action, placeholder: renames all emanating streets from a given square
def renaming_action (square : ℕ) (naming : (ℕ × ℕ) → bool) (street: (ℕ × ℕ)) : bool :=
  if street.1 = square ∨ street.2 = square then !naming street else naming street

theorem tsvetochny_street_naming : exists_initial_street_naming n m h streets initial_naming renaming_action := sorry

end tsvetochny_street_naming_l407_407299


namespace find_theta_l407_407916

noncomputable def vector_length (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def vector_dot (u v : ℝ × ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def cos_theta (u v : ℝ × ℝ × ℝ) : ℝ :=
vector_dot u v / (vector_length u * vector_length v)

theorem find_theta :
  ∀ (a b c d : ℝ × ℝ × ℝ),
    vector_length a = 2 →
    vector_length b = 2 →
    vector_length c = 3 →
    vector_length d = 1 →
    a.cross (b.cross c) + d = (0, 0, 0) →
    cos_theta a c = 1/3 ∨ cos_theta a c = -1/3 :=
begin
  intros a b c d ha hb hc hd h,
  sorry,
end

end find_theta_l407_407916


namespace minimize_distance_l407_407733

noncomputable def find_minimizing_point (a b c : ℝ) : ℝ := 
  (a + 2 * b + 3 * c) / 6

theorem minimize_distance (a b c : ℝ) :
  ∃ P : ℝ, P = find_minimizing_point a b c := 
begin
  use find_minimizing_point a b c,
  sorry,
end

end minimize_distance_l407_407733


namespace binom_30_3_eq_4060_l407_407571

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l407_407571


namespace max_money_collected_l407_407858

-- Definition of the grid and conditions

def grid_6x6 : List (List ℕ) := [
  [4, 5, 6, 1, 3, 2], 
  [5, 9, 8, 2, 7, 1],
  [4, 6, 3, 5, 2, 6],
  [3, 7, 4, 8, 5, 9],
  [5, 1, 6, 7, 3, 8],
  [5, 6, 4, 2, 1, 7]
]

-- Statement to maximize the collection of money
theorem max_money_collected : 
  ∃ (path : List (ℕ × ℕ)), (∀ (i j : ℕ × ℕ), i ≠ j → i.fst = j.fst ∨ i.snd = j.snd ∨ abs (i.fst - j.fst) = abs (i.snd - j.snd) → false ∧ 
  ∑ (p : ℕ × ℕ) in path, grid_6x6[p.fst][p.snd] = 47 :=
  sorry

end max_money_collected_l407_407858


namespace certain_number_divided_by_10_l407_407735
-- Broad import to bring in necessary libraries

-- Define the constants and hypotheses
variable (x : ℝ)
axiom condition : 5 * x = 100

-- Theorem to prove the required equality
theorem certain_number_divided_by_10 : (x / 10) = 2 :=
by
  -- The proof is skipped by sorry
  sorry

end certain_number_divided_by_10_l407_407735


namespace prob_A_or_B_selected_l407_407972

def participants : Finset ℕ := {0, 1, 2, 3, 4} -- Representing participants A, B, C, D and E by 0, 1, 2, 3, and 4 respectively

def choose_two (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ x, x.card = 2)

def prob_A_or_B : ℚ :=
  let total_outcomes := (choose_two participants).card
  let opposite_event := (choose_two {2, 3, 4}).card
  1 - (opposite_event : ℚ) / (total_outcomes : ℚ)

theorem prob_A_or_B_selected : prob_A_or_B = 7 / 10 := 
by 
  sorry

end prob_A_or_B_selected_l407_407972


namespace num_factors_of_60_l407_407239

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l407_407239


namespace excluded_number_is_35_l407_407954

theorem excluded_number_is_35 {a1 a2 a3 a4 a5 : ℝ}
  (h1 : (a1 + a2 + a3 + a4 + a5) / 5 = 27)
  (h2 : (a1 + a2 + a3 + a4) / 4 = 25 ∨
        (a1 + a2 + a3 + a5) / 4 = 25 ∨
        (a1 + a2 + a4 + a5) / 4 = 25 ∨
        (a1 + a3 + a4 + a5) / 4 = 25 ∨
        (a2 + a3 + a4 + a5) / 4 = 25) :
  ∃ x ∈ {a1, a2, a3, a4, a5}, x = 35 := by
  sorry

end excluded_number_is_35_l407_407954


namespace angle_BPC_lt_120_l407_407344

-- Definitions of given conditions
variables {A B C P X Y : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited X] [Inhabited Y]

variables {a b c : ℝ} (hA : a < 60) -- angle at A is less than 60 degrees
          (h1 : c + a = b + a)      -- CA + AX = CB + BX
          (h2 : b + a = c + a)      -- BA + AY = BC + CY

-- Define the point P such that lines PX and PY are perpendicular to AB and AC respectively
variables (hP_perp1 : line P X ⊥ line A B)
          (hP_perp2 : line P Y ⊥ line A C)

-- Angle BPC < 120 degrees
theorem angle_BPC_lt_120 (hA : a < 60) (h1 : c + a = b + a) (h2 : b + a = c + a)
  (hP_perp1 : line P X ⊥ line A B) (hP_perp2 : line P Y ⊥ line A C) :
  angle B P C < 120 :=
sorry

end angle_BPC_lt_120_l407_407344


namespace largest_n_satisfying_expression_l407_407936

theorem largest_n_satisfying_expression :
  ∃ n < 100000, (n - 3)^5 - n^2 + 10 * n - 30 ≡ 0 [MOD 3] ∧ 
  (∀ m, m < 100000 → (m - 3)^5 - m^2 + 10 * m - 30 ≡ 0 [MOD 3] → m ≤ 99998) := sorry

end largest_n_satisfying_expression_l407_407936


namespace num_factors_of_60_l407_407260

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l407_407260


namespace widescreen_tv_horizontal_length_l407_407852

theorem widescreen_tv_horizontal_length :
  ∀ (h l : ℝ),
    h * 16 = l * 9 →
    sqrt (h^2 + l^2) = 40 →
    l = 640 * sqrt 337 / 337 :=
by
  intros h l ratio_diag len_diag
  sorry

end widescreen_tv_horizontal_length_l407_407852


namespace find_matrix_N_l407_407094

theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℚ) :
  N * ![![2, -5], ![4, -3]] = ![![ -20, -8], ![9, 3]] ->
  N = ![![43 / 7, -54 / 7], ![-33 / 14, 24 / 7]] := 
by
  intros h
  sorry

end find_matrix_N_l407_407094


namespace curve_transformation_l407_407169

variable (x y x0 y0 : ℝ)

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -2], ![0, 1]]

def C (x0 y0 : ℝ) : Prop := (x0 - y0)^2 + y0^2 = 1

def transform (x0 y0 : ℝ) : ℝ × ℝ :=
  let x := 2 * x0 - 2 * y0
  let y := y0
  (x, y)

def C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

theorem curve_transformation :
  ∀ x0 y0, C x0 y0 → C' (2 * x0 - 2 * y0) y0 := sorry

end curve_transformation_l407_407169


namespace maximum_value_f_max_f_value_l407_407652

def f (x : ℝ) : ℝ := |x| - |x - 3|

theorem maximum_value_f : ∀ x : ℝ, f x ≤ 3 :=
by
  intros x
  cases lt_or_ge x 0 with x_neg x_nonneg
  case inl =>
    have f_x : f x = -3 := by 
      simp [f, abs_of_neg x_neg, abs_of_pos (sub_pos.2 x_neg)]
    exact le_trans (le_of_eq f_x) (by norm_num)
  case inr =>
    cases le_total x 3 with x_le_3 x_gt_3
    case inl =>
      have f_x : ∃ y, y = x → f x = 2 * y - 3 := by
        simp [f, abs_of_nonneg x_le_3, abs_of_nonneg (sub_nonneg.2 x_nonneg)]
      exact calc
        f x = 2 * x - 3 := by assumption
        ... ≤ 3 := by linarith
    case inr =>
      have f_x : f x = 3 := by 
        simp [f, abs_of_pos x_nonneg, abs_of_nonneg (sub_nonneg.2 x_nonneg)]
      exact le_of_eq f_x

theorem max_f_value : ∃ x : ℝ, f x = 3 :=
exists.intro 3 (by simp [f, abs_of_nonneg (by norm_num), abs_of_nonneg (by norm_num)])

end maximum_value_f_max_f_value_l407_407652


namespace total_interest_l407_407953

theorem total_interest (n : ℕ) :
  (1000 * (3 / 100.0) * n) + (1200 * (5 / 100.0) * n) = 350 →
  n = 4 :=
by
  sorry

end total_interest_l407_407953


namespace emily_walks_more_l407_407438

def distance_walked (base_distance: ℕ) (detours: List ℕ) : ℕ := 
  2 * (base_distance * 5 + detours.sum)

def troy_base_distance : ℕ := 75
def emily_base_distance : ℕ := 98

def troy_detours : List ℕ := [15, 20, 10, 10, 5]
def emily_detours : List ℕ := [10, 25, 10, 25, 10]

def additional_detour_thursday : ℕ := 10 

def troy_total_dist : ℕ := distance_walked troy_base_distance (troy_detours.map (λ x, x + additional_detour_thursday))
def emily_total_dist : ℕ := distance_walked emily_base_distance (emily_detours.map (λ x, x + additional_detour_thursday))

theorem emily_walks_more : (emily_total_dist - troy_total_dist) = 270 := by
  sorry

end emily_walks_more_l407_407438


namespace meet_time_l407_407821

/-- Jonathan leaves Town A at 7:45 AM traveling towards Town B at 15 miles per hour.
    Elizabeth leaves Town B at 8:15 AM traveling towards Town A at 18 miles per hour, 
    but stops for 15 minutes to rest. The distance between Town A and Town B is 90 miles. 
    Prove they meet at 10:53 AM. --/
theorem meet_time (x : ℝ) (time_meet : ℝ := 3.136) :
  7.75 + x = time_meet →
  (15 * x) + (18 * (x - 0.75)) = 90 →
  time_meet ≈ 10.8833 :=
by
  sorry

end meet_time_l407_407821


namespace efficiency_ratio_l407_407866

theorem efficiency_ratio (K : ℝ) (hK : K > 0):
  ∃ R : ℝ, (R = (1/2) * K) ∧ ((30 * R) = (10 * (R + K))) ∧ (R / K = 1 / 2) :=
by
  -- definitions from the conditions
  let R := (1/2) * K
  have h1 : R = (1/2) * K := rfl
  have h2 : 30 * R = 10 * (R + K) := 
    calc
      30 * R = 30 * ((1/2) * K)       : by rw h1
          ... = 15 * K                 : by rw mul_assoc; norm_num
          ... = 10 * (K/2 + K)         : by norm_num
          ... = 10 * (R + K)           : by rw h1
  have h3 : R / K = 1 / 2 := by rw h1; field_simp
  use R
  -- combining conditions to assert the theorem
  exact ⟨h1, h2, h3⟩

end efficiency_ratio_l407_407866


namespace no_real_coeff_quadratic_with_roots_sum_and_product_l407_407098

theorem no_real_coeff_quadratic_with_roots_sum_and_product (a b c : ℝ) (h : a ≠ 0) :
  ¬ ∃ (α β : ℝ), (α = a + b + c) ∧ (β = a * b * c) ∧ (α + β = -b / a) ∧ (α * β = c / a) :=
by
  sorry

end no_real_coeff_quadratic_with_roots_sum_and_product_l407_407098


namespace angle_BPC_lt_120_l407_407343

-- Definitions of given conditions
variables {A B C P X Y : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited X] [Inhabited Y]

variables {a b c : ℝ} (hA : a < 60) -- angle at A is less than 60 degrees
          (h1 : c + a = b + a)      -- CA + AX = CB + BX
          (h2 : b + a = c + a)      -- BA + AY = BC + CY

-- Define the point P such that lines PX and PY are perpendicular to AB and AC respectively
variables (hP_perp1 : line P X ⊥ line A B)
          (hP_perp2 : line P Y ⊥ line A C)

-- Angle BPC < 120 degrees
theorem angle_BPC_lt_120 (hA : a < 60) (h1 : c + a = b + a) (h2 : b + a = c + a)
  (hP_perp1 : line P X ⊥ line A B) (hP_perp2 : line P Y ⊥ line A C) :
  angle B P C < 120 :=
sorry

end angle_BPC_lt_120_l407_407343


namespace sum_floor_sqrt_1_to_25_l407_407002

theorem sum_floor_sqrt_1_to_25 :
  (∑ n in finset.range 25, ⌊real.sqrt (n + 1)⌋) = 75 :=
sorry

end sum_floor_sqrt_1_to_25_l407_407002


namespace num_factors_of_60_l407_407255

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l407_407255


namespace rectangle_area_l407_407982

theorem rectangle_area (x : ℝ) (h : (x - 3) * (2 * x + 3) = 4 * x - 9) : x = 7 / 2 :=
sorry

end rectangle_area_l407_407982


namespace polynomial_divisibility_l407_407267

def poly1 (x : ℝ) (k : ℝ) : ℝ := 3*x^3 - 9*x^2 + k*x - 12

theorem polynomial_divisibility (k : ℝ) :
  (∀ (x : ℝ), poly1 x k = (x - 3) * (3*x^2 + 4)) → (poly1 3 k = 0) := sorry

end polynomial_divisibility_l407_407267


namespace binomial_coefficient_x3_l407_407648

theorem binomial_coefficient_x3 (n : ℕ) : 
  (∃ c : ℕ, (x + 1 : ℂ)^n = c * x^3 + ∑ k in range n, k ≠ 3 → binom n k * x^k * (1 : ℂ)^(n - k) ) → 
  c = n.choose 3 := 
sorry

end binomial_coefficient_x3_l407_407648


namespace find_a_l407_407723

theorem find_a
  {a b c x s t : ℝ} 
  (h1 : a < 0)
  (h2 : ∀ (x : ℝ), a*x^2 + b*x + c ≥ 0)
  (h3 : s ≠ t)
  (h4 : (s, sqrt(a*s^2 + b * s + c)) = (t + sqrt(1), sqrt(a*(t + sqrt(1)) * (t + sqrt(1)) + b * (t + sqrt(1)) + c)))
  : a = -4 :=
sorry

end find_a_l407_407723


namespace simplify_expression_l407_407388

theorem simplify_expression (y : ℝ) : 3 * y + 5 * y + 6 * y + 10 = 14 * y + 10 :=
by
  sorry

end simplify_expression_l407_407388


namespace num_factors_of_60_l407_407262

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l407_407262


namespace find_ratio_l407_407303

variable (A B C D : Type)

variable [InnerProductSpace ℝ A]
variable [InnerProductSpace ℝ B]
variable [InnerProductSpace ℝ C]
variable [InnerProductSpace ℝ D]

variable (AD AB AC : ℝ)
variable (angleBAC : ℝ)
variable (S1 S2 S3 S4 : ℝ)

axiom tetrahedron_properties :
  AD > AB ∧
  AD ⟂ AB ∧
  AD ⟂ AC ∧
  angleBAC = Real.pi / 3

axiom areas :
  S1 + S2 = S3 + S4

theorem find_ratio :
  AD > AB →
  AD ⟂ AB →
  AD ⟂ AC →
  angleBAC = Real.pi / 3 →
  S1 + S2 = S3 + S4 →
  (S3 / S1) + (S3 / S2) = 3 / 2 :=
by sorry

end find_ratio_l407_407303


namespace num_factors_60_l407_407212

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l407_407212


namespace problem1_problem2_l407_407967

theorem problem1 (x : ℝ) (hx : x ∈ Set.Icc (-1 : ℝ) (1 : ℝ)) :
  |4 * x ^ 3 - 3 * x| ≤ 1 :=
sorry

theorem problem2 (a b c M : ℝ) (hM : M = Real.Sup (Set.image (λ x, |4 * x ^ 3 + a * x ^ 2 + b * x + c|) (Set.Icc (-1 : ℝ) (1 : ℝ)))) :
  M ≥ 1 ∧ a = 0 ∧ b = -3 ∧ c = 0 :=
sorry

end problem1_problem2_l407_407967


namespace std_deviation_calc_l407_407400

theorem std_deviation_calc 
  (μ : ℝ) (σ : ℝ) (V : ℝ) (k : ℝ)
  (hμ : μ = 14.0)
  (hσ : σ = 1.5)
  (hV : V = 11)
  (hk : k = (μ - V) / σ) :
  k = 2 := by
  sorry

end std_deviation_calc_l407_407400


namespace not_on_hyperbola_l407_407296

noncomputable def opposite_number (x y : ℝ) : Prop := x = -y

theorem not_on_hyperbola (x y : ℝ) (h : opposite_number x y) : y ≠ 1 / x :=
by {
  assume h1: y = 1 / x,
  have h2: x * y = 1, from (eq_div_iff (ne_of_eq_of_ne (eq.symm h) zero_ne_one)).mp h1,
  have h3: x * (-x) = 1, from h ▸ h2,
  have h4: -x * x = -1, by ring,
  contradiction, }

end not_on_hyperbola_l407_407296


namespace range_of_m_l407_407943

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

theorem range_of_m :
  (∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → 2^x - Real.log x / Real.log (1/2) + m ≤ 0) →
  m ≤ -5 :=
sorry

end range_of_m_l407_407943


namespace four_sides_equal_is_rhombus_l407_407896

theorem four_sides_equal_is_rhombus (Q : Type) [Quadrilateral Q] :
  (∀ (AB BC CD DA : Q), (EQ AB BC) → (EQ BC CD) → (EQ CD DA) → (EQ DA AB) → Rhombus Q) :=
sorry

end four_sides_equal_is_rhombus_l407_407896


namespace sum_floor_sqrt_1_to_25_l407_407033

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l407_407033


namespace cannot_be_arithmetic_progression_can_be_geometric_progression_l407_407561

theorem cannot_be_arithmetic_progression (a b c : ℝ) (ha : a = 2) (hb : b = real.sqrt 6) (hc : c = 4.5) : 
  ¬ ∃ d : ℝ, b = a + d ∧ c = a + 2 * d := by
  sorry

theorem can_be_geometric_progression (a b c : ℝ) (ha : a = 2) (hb : b = real.sqrt 6) (hc : c = 4.5) : 
  ∃ q : ℝ, q = real.sqrt (3 / 2) ∧ b = a * q ∧ c = a * q^2 := by
  sorry

end cannot_be_arithmetic_progression_can_be_geometric_progression_l407_407561


namespace find_matrix_N_l407_407096

theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℚ) :
  N * !![!![2, -5], !![4, -3]] = !![!![-20, -8], !![9, 3]] →
  N = !![!![(46/7 : ℚ), -(58/7 : ℚ)], !![-(39/14 : ℚ), (51/14 : ℚ)]] :=
by 
  intro h
  sorry

end find_matrix_N_l407_407096


namespace geometric_sequence_third_term_l407_407165

theorem geometric_sequence_third_term (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h1 : a 1 * a 5 = 16) :
  a 3 = 4 ∨ a 3 = -4 := 
sorry

end geometric_sequence_third_term_l407_407165


namespace quadratic_roots_l407_407138

-- Definitions based on the conditions provided.
def condition1 (x y : ℝ) : Prop := x^2 - 6 * x + 9 = -(abs (y - 1))

-- The main theorem we want to prove.
theorem quadratic_roots (x y : ℝ) (h : condition1 x y) : (a : ℝ) → (a - 3) * (a - 1) = a^2 - 4 * a + 3 :=
  by sorry

end quadratic_roots_l407_407138


namespace evaluate_expression_l407_407684

variable (a : ℝ)

def a_definition : Prop := a = Real.sqrt 11 - 1

theorem evaluate_expression (h : a_definition a) : a^2 + 2*a + 1 = 11 := by
  sorry

end evaluate_expression_l407_407684


namespace parabola_tangent_rational_slope_l407_407329

variables {x y a b c : ℚ}

/-- Define parabola P₁ -/
def P1 (x : ℚ) : ℚ := x^2 + 203/200

/-- Define parabola P₂ -/
def P2 (y : ℚ) : ℚ := y^2 + 27/2

/-- Define line L -/
def L (a b c : ℚ) (x y : ℚ) : Prop := a * x + b * y = c

theorem parabola_tangent_rational_slope
  (h1 : ∃ x, L a b c x (P1 x))
  (h2 : ∃ y, L a b c (P2 y) y)
  (gcd_abc : Int.gcd (Int.ofRat a) (Int.gcd (Int.ofRat b) (Int.ofRat c)) = 1)
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  a + b + c = 13 :=
sorry

end parabola_tangent_rational_slope_l407_407329


namespace pies_sold_l407_407997

-- Definitions based on conditions
def cost_per_slice : ℕ := 7
def slices_per_pie : ℕ := 6
def total_revenue : ℕ := 294
def cost_per_pie := cost_per_slice * slices_per_pie

-- Theorem to prove the number of pies sold
theorem pies_sold : total_revenue / cost_per_pie = 7 :=
begin
  -- Proof goes here
  sorry
end

end pies_sold_l407_407997


namespace sum_floor_sqrt_l407_407056

theorem sum_floor_sqrt :
  (∑ n in Finset.range 25, Int.floor (Real.sqrt (n + 1))) = 71 := by
  sorry

end sum_floor_sqrt_l407_407056


namespace sum_of_integers_100_to_110_l407_407465

theorem sum_of_integers_100_to_110 : ∑ i in Finset.range (111 - 100), (100 + i) = 1155 :=
by
  sorry

end sum_of_integers_100_to_110_l407_407465


namespace binom_30_3_l407_407614

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l407_407614


namespace binomial_coefficient_30_3_l407_407590

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l407_407590


namespace quadratic_function_even_l407_407729

theorem quadratic_function_even (a b : ℝ) (h1 : ∀ x : ℝ, x^2 + (a-1)*x + a + b = x^2 - (a-1)*x + a + b) (h2 : 4 + (a-1)*2 + a + b = 0) : a + b = -4 := 
sorry

end quadratic_function_even_l407_407729


namespace find_b_age_l407_407488

-- Define the conditions as hypotheses
variables (a b : ℕ)
hypotheses (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 8)

-- Prove that b = 38
theorem find_b_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 8) : b = 38 := by
  sorry

end find_b_age_l407_407488


namespace anne_cleaning_time_l407_407485

theorem anne_cleaning_time :
  (∃ (B A : ℝ), B + A = 1 / 4 ∧ B + 2 * A = 1 / 3) →
  (1 / A = 12) :=
by
  intros h
  have hB := h.1
  have hA := h.2
  sorry  -- Proof steps are skipped as per instructions

end anne_cleaning_time_l407_407485


namespace original_slices_proof_l407_407631

def original_slices (andy_consumption toast_slices leftover_slice: ℕ) : ℕ :=
  andy_consumption + toast_slices + leftover_slice

theorem original_slices_proof :
  original_slices (3 * 2) (10 * 2) 1 = 27 :=
by
  sorry

end original_slices_proof_l407_407631


namespace floor_sum_sqrt_1_to_25_l407_407059

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l407_407059


namespace nonnegative_integer_solutions_l407_407085

theorem nonnegative_integer_solutions :
  {ab : ℕ × ℕ | 3 * 2^ab.1 + 1 = ab.2^2} = {(0, 2), (3, 5), (4, 7)} :=
by
  sorry

end nonnegative_integer_solutions_l407_407085


namespace fraction_savings_spent_on_furniture_l407_407849

theorem fraction_savings_spent_on_furniture (savings : ℝ) (tv_cost : ℝ) (F : ℝ) 
  (h1 : savings = 840) (h2 : tv_cost = 210) 
  (h3 : F * savings + tv_cost = savings) : F = 3 / 4 :=
sorry

end fraction_savings_spent_on_furniture_l407_407849


namespace original_slices_proof_l407_407630

def original_slices (andy_consumption toast_slices leftover_slice: ℕ) : ℕ :=
  andy_consumption + toast_slices + leftover_slice

theorem original_slices_proof :
  original_slices (3 * 2) (10 * 2) 1 = 27 :=
by
  sorry

end original_slices_proof_l407_407630


namespace sum_floor_sqrt_1_to_25_l407_407035

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l407_407035


namespace problem1_problem2_problem3_l407_407689

noncomputable def f : ℝ → ℝ := sorry

open_locale classical

variables {x y : ℝ}
-- Given conditions
axiom cond1 (hx : x > 0) : f (x) < 0 ↔ x > 1
axiom cond2 (hx : x > 0) (hy : y > 0) : f (x / y) = f (x) - f (y)
axiom cond3 : f(3) = -1

-- Prove f(1) = 0
theorem problem1 : f(1) = 0 := sorry

-- Prove f(x) is strictly decreasing on (0, ∞)
theorem problem2 : ∀ {x1 x2 : ℝ}, 0 < x1 → 0 < x2 → x1 > x2 → f(x1) < f(x2) := sorry

-- Solve the inequality f(|x|) < -2
theorem problem3 : ∀ {x : ℝ}, f(abs(x)) < -2 → x < -9 ∨ x > 9 := sorry

end problem1_problem2_problem3_l407_407689


namespace lily_profit_is_correct_l407_407848

-- Define the conditions
def first_ticket_price : ℕ := 1
def price_increment : ℕ := 1
def number_of_tickets : ℕ := 5
def prize_amount : ℕ := 11

-- Define the sum of arithmetic series formula
def total_amount_collected (n : ℕ) (a : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Calculate the total amount collected
def total : ℕ := total_amount_collected number_of_tickets first_ticket_price price_increment

-- Define the profit calculation
def profit : ℕ := total - prize_amount

-- The statement we need to prove
theorem lily_profit_is_correct : profit = 4 := by
  sorry

end lily_profit_is_correct_l407_407848


namespace num_lattice_points_distance_five_l407_407815

open_locale classical 

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points_distance_five : 
  {n : ℕ | ∃ (x y z : ℤ), is_lattice_point x y z} = 78 :=
by sorry

end num_lattice_points_distance_five_l407_407815


namespace sum_of_floors_of_square_roots_l407_407022

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l407_407022


namespace quadratic_eq_with_roots_l407_407141

theorem quadratic_eq_with_roots (x y : ℝ) (h : (x^2 - 6 * x + 9) = -|y - 1|) : 
  ∃ a : ℝ, (a^2 - 4 * a + 3 = 0) :=
by 
  sorry

end quadratic_eq_with_roots_l407_407141


namespace number_of_positive_factors_of_60_l407_407247

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l407_407247


namespace line_through_intersection_and_perpendicular_l407_407090

theorem line_through_intersection_and_perpendicular :
  ∃ (A B C : ℝ), (A = 16) ∧ (B = -12) ∧ (C = 29) ∧
  (∀ x y : ℝ, (2 * x + 3 * y + 1 = 0) ∧ (x - 3 * y + 4 = 0) →
    16 * x - 12 * y + 29 = 0) ∧
  (∀ x y : ℝ, (3 * x + 4 * y - 7 = 0) →
    16 * x - 12 * y + 29 = 0) :=
begin
  sorry
end

end line_through_intersection_and_perpendicular_l407_407090


namespace lattice_points_with_distance_five_from_origin_l407_407805

theorem lattice_points_with_distance_five_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 6 :=
by
  sorry

end lattice_points_with_distance_five_from_origin_l407_407805


namespace trajectory_eq_sum_reciprocal_distances_l407_407843

variable (x y : ℝ)

def ellipse := { p : ℝ×ℝ // 9 * p.1^2 + 4 * p.2^2 = 36 }
def transform (M : ellipse) : ℝ × ℝ := (M.val.1, M.val.2 * (2 : ℝ) * (√2 : ℝ))

theorem trajectory_eq :
  ∀ p : ellipse, (∃ x y, p.1 = x ∧ p.2 = y ∧ 9 * x^2 + 8 * y^2 = 72) :=
by sorry

theorem sum_reciprocal_distances (F : ℝ × ℝ) (A B C D : ℝ × ℝ)
  (hFA : F.1 = 1 ∧ F.2 = 0)
  (hP : 9 * A.1^2 + 8 * A.2^2 = 72 ∧ 9 * B.1^2 + 8 * B.2^2 = 72)
  (hQ : 9 * C.1^2 + 8 * C.2^2 = 72 ∧ 9 * D.1^2 + 8 * D.2^2 = 72)
  (hl1 : ∀ x, y = k * (x - 1)) (hl2 : ∀ x, y = -(1 / k) * (x - 1)) :
  1 / real.dist A B + 1 / real.dist C D = 17 / 48 :=
by sorry

end trajectory_eq_sum_reciprocal_distances_l407_407843


namespace sum_congruence_example_l407_407475

theorem sum_congruence_example (a b c : ℤ) (h1 : a % 15 = 7) (h2 : b % 15 = 3) (h3 : c % 15 = 9) : 
  (a + b + c) % 15 = 4 :=
by 
  sorry

end sum_congruence_example_l407_407475


namespace marble_arrangement_remainder_l407_407079

theorem marble_arrangement_remainder :
  ∃ (m N : ℕ), m = 17 ∧ N = Nat.choose 23 6 ∧ (N % 1000 = 947) := by
  have m : ℕ := 17
  have N : ℕ := Nat.choose 23 6
  use m, N
  split
  · rfl
  split
  · rfl
  · sorry

end marble_arrangement_remainder_l407_407079


namespace sum_of_integers_100_to_110_l407_407463

theorem sum_of_integers_100_to_110 : ∑ i in Finset.range (111 - 100), (100 + i) = 1155 :=
by
  sorry

end sum_of_integers_100_to_110_l407_407463


namespace lattice_points_count_is_30_l407_407791

-- Define the function that counts the number of lattice points (x, y, z) such that x^2 + y^2 + z^2 = 25.
def count_lattice_points := {p : ℤ × ℤ × ℤ // p.1 ^ 2 + p.2.1 ^ 2 + p.2.2 ^ 2 = 25}.to_finset.card

-- The theorem to prove that the count is 30.
theorem lattice_points_count_is_30 : count_lattice_points = 30 :=
by
  sorry

end lattice_points_count_is_30_l407_407791


namespace num_factors_60_l407_407183

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l407_407183


namespace shortest_path_on_cube_l407_407994

theorem shortest_path_on_cube (a : ℝ) (ha : a = 2) : 
  shortest_path_length (a) = 3 :=
begin
  sorry
end

end shortest_path_on_cube_l407_407994


namespace num_factors_of_60_l407_407261

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l407_407261


namespace sum_of_floors_of_square_roots_l407_407024

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l407_407024


namespace evaluate_f_at_5pi_over_6_l407_407156

noncomputable def f (x : ℝ) : ℝ := Real.sin (π / 2 + x) * Real.sin (π + x)

theorem evaluate_f_at_5pi_over_6 : f (5 * π / 6) = sqrt 3 / 4 :=
by
  sorry

end evaluate_f_at_5pi_over_6_l407_407156


namespace fisherman_bass_count_l407_407519

theorem fisherman_bass_count (B T G : ℕ) (h1 : T = B / 4) (h2 : G = 2 * B) (h3 : B + T + G = 104) : B = 32 :=
by
  sorry

end fisherman_bass_count_l407_407519


namespace initial_price_of_gasoline_l407_407415

theorem initial_price_of_gasoline 
  (P0 : ℝ) 
  (P1 : ℝ := 1.30 * P0)
  (P2 : ℝ := 0.75 * P1)
  (P3 : ℝ := 1.10 * P2)
  (P4 : ℝ := 0.85 * P3)
  (P5 : ℝ := 0.80 * P4)
  (h : P5 = 102.60) : 
  P0 = 140.67 :=
by sorry

end initial_price_of_gasoline_l407_407415


namespace james_delivers_bags_l407_407820

theorem james_delivers_bags :
  let bags_per_trip := 10
  let trips_per_day := 20
  let days := 5
  bags_per_trip * trips_per_day * days = 1000 :=
by
  let bags_per_trip := 10
  let trips_per_day := 20
  let days := 5
  calc
    10 * 20 * 5 = 200 * 5 : by ring
             ... = 1000 : by ring

end james_delivers_bags_l407_407820


namespace binomial_coefficient_30_3_l407_407591

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l407_407591


namespace product_lcm_gcd_l407_407461

-- Definitions based on the conditions given in a)
def a : ℕ := 24
def b : ℕ := 54

-- The statement we want to prove.
theorem product_lcm_gcd (a b : ℕ) : (Nat.gcd a b) * (Nat.lcm a b) = 1296 :=
by
  have h1 : a = 24 := rfl
  have h2 : b = 54 := rfl
  sorry

end product_lcm_gcd_l407_407461


namespace probability_divisible_by_5_l407_407330

-- Define the set S with the given condition
def S : Set ℤ := { n | ∃ j k : ℕ, j ≠ k ∧ 0 ≤ j ∧ j < 40 ∧ 0 ≤ k ∧ k < 40 ∧ n = 2^j + 2^k }

-- Define the count of elements in S (this must be proven as a separate lemma/theorem)
axiom S_cardinality : S.card = 780 

-- Define the sets of pairs that sum to multiples of 5 (mod 5)
def is_divisible_by_5 (n : ℤ) : Prop := n % 5 = 0

-- Prove that the probability of picking a number from S that is divisible by 5 is 3/26
theorem probability_divisible_by_5 : ∃ p q, p + q = 29 ∧ ∀ n, (n ∈ S → is_divisible_by_5 n → (p, q).gcd = 1) :=
by
  let p := 3
  let q := 26
  have gcd_pq : Int.gcd p q = 1 := by
    -- Proof obligation of gcd(p, q) = 1, you can assume it as axiom or prove it
    sorry
  -- Prove that p + q = 29
  have pq_sum : p + q = 29 := by
    -- Proof obligation of sum of p and q = 29
    -- you can prove it straightforwardly
    exact rfl
  use [p, q]
  apply And.intro pq_sum gcd_pq
  -- skip the comprehensive proof steps
  sorry

end probability_divisible_by_5_l407_407330


namespace abelian_group_of_order_p4_l407_407825

open Finite
open QuotientGroup

theorem abelian_group_of_order_p4
  (p : ℕ)
  (h_prime : Prime p)
  (A : Type*)
  [fintype A]
  [add_comm_group A]
  (hA : card A = p^4)
  (N : add_subgroup A)
  (hN : card N = p) 
  (hquot : Quotient A N ≃* (zpowers (p^3 : ℕ))) :
  A ≃+ (add_zmod (p^4 : ℕ)) ∨ A ≃+ (add_zmod (p^3 : ℕ) × add_zmod (p : ℕ)) :=
sorry

end abelian_group_of_order_p4_l407_407825


namespace part1_part2_l407_407124

-- Definition of the complex number Z1 and conditions
def Z1 (a : ℝ) := 2 + a * complex.i
def Z (a : ℝ) := (2 + a * complex.i) / (1 - complex.i)

-- Assuming Z1^2 is pure imaginary and finding the value of a
theorem part1 (a : ℝ) (ha1 : a > 0) (ha2 : (Z1 a) ^ 2).re = 0 : a = 2 := by
  sorry

-- Given Z1 and a = 2, find the modulus of Z
theorem part2 (a : ℝ) (ha : a = 2) : complex.abs (Z a) = 2 := by
  sorry

end part1_part2_l407_407124


namespace probability_blue_odd_yellow_even_l407_407439

noncomputable def blue_ball_odd_probability : ℚ :=
  ∑' (k : ℕ) in { k | odd k }, 3⁻¹ ^ k

noncomputable def yellow_ball_even_probability : ℚ :=
  ∑' (k : ℕ) in { k | even k }, (3 / 2)⁻¹ ^ k

theorem probability_blue_odd_yellow_even :
  blue_ball_odd_probability * yellow_ball_even_probability = 1 / 5 := by
  sorry

end probability_blue_odd_yellow_even_l407_407439


namespace max_consecutive_integers_sum_le_500_l407_407458

def consecutive_sum (n : ℕ) : ℕ :=
  -- Formula for sum starting from 3
  (n * (n + 1)) / 2 - 3

theorem max_consecutive_integers_sum_le_500 : ∃ n : ℕ, consecutive_sum n ≤ 500 ∧ ∀ m : ℕ, m > n → consecutive_sum m > 500 :=
by
  sorry

end max_consecutive_integers_sum_le_500_l407_407458


namespace sum_of_floors_of_square_roots_l407_407026

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l407_407026


namespace equation_rearrangement_l407_407396

variables {α β z y a b c d : ℝ}

def AD_minus BC := (a + d) - real.sqrt ((a + d)^2 + 4*(b*c - a*d))
def AD_plus BC := (a + d) + real.sqrt ((a + d)^2 + 4*(b*c - a*d))
noncomputable def k := AD_minus BC / AD_plus BC

theorem equation_rearrangement :
  x = (a * x + b) / (c * x + d) → α = (a * α + b) / (c * α + d) → β = (a * β + b) / (c * β + d) →
  y = (a * z + b) / (c * z + d) →
  (y - α) / (y - β) = k * (z - α) / (z - β) :=
by {
  sorry
}

end equation_rearrangement_l407_407396


namespace problem_l407_407690

-- Definitions and conditions
variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of first n terms

-- Condition: a_n ≠ 0 for all n ∈ ℕ^*
axiom h1 : ∀ n : ℕ, n > 0 → a n ≠ 0

-- Condition: a_n * a_{n+1} = S_n
axiom h2 : ∀ n : ℕ, n > 0 → a n * a (n + 1) = S n

-- Given: S_1 = a_1
axiom h3 : S 1 = a 1

-- Given: S_2 = a_1 + a_2
axiom h4 : S 2 = a 1 + a 2

-- Prove: a_3 - a_1 = 1
theorem problem : a 3 - a 1 = 1 := by
  sorry

end problem_l407_407690


namespace num_lattice_points_on_sphere_5_l407_407769

theorem num_lattice_points_on_sphere_5 :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.to_finset.card = 54 := 
by
  sorry

end num_lattice_points_on_sphere_5_l407_407769


namespace intersection_M_N_l407_407700

def M : Set ℝ := { x | Real.exp (x - 1) > 1 }
def N : Set ℝ := { x | x^2 - 2*x - 3 < 0 }

theorem intersection_M_N :
  (M ∩ N : Set ℝ) = { x | 1 < x ∧ x < 3 } := 
by
  sorry

end intersection_M_N_l407_407700


namespace number_of_positive_factors_of_60_l407_407250

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l407_407250


namespace sum_of_altitudes_of_triangle_l407_407068

theorem sum_of_altitudes_of_triangle : 
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  altitude1 + altitude2 + altitude3 = (22 * Real.sqrt 73 + 48) / Real.sqrt 73 :=
by
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  sorry

end sum_of_altitudes_of_triangle_l407_407068


namespace find_k_l407_407707

variable {f : ℝ → ℝ}
variable {k : ℝ}

-- Condition: f is a decreasing function on (-∞, 1]
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ y ≤ b ∧ x ≤ y → f x ≥ f y

def problem_conditions (f : ℝ → ℝ) (k : ℝ) : Prop :=
  is_decreasing_on f (-∞) 1 ∧ ∀ x, f (k - real.sin x) ≥ f (k^2 - (real.sin x)^2)

-- The theorem we prove
theorem find_k (h : problem_conditions f k) : k = -1 :=
sorry

end find_k_l407_407707


namespace total_pushups_first_week_l407_407680

-- Definitions based on conditions
def first_day_pushups : ℕ := 10
def increment_per_day : ℕ := 5
def days_workout : ℕ := 3

-- Function to calculate the number of push-ups on a specific day
def pushups_on_day (day : ℕ) : ℕ :=
  first_day_pushups + (day - 1) * increment_per_day

-- Function to calculate total push-ups in the first week
def total_pushups_week : ℕ :=
  ∑ i in Finset.range days_workout, pushups_on_day (i + 1)

-- Theorem to prove the total number of push-ups in the first week
theorem total_pushups_first_week : total_pushups_week = 45 := by
  -- Proof will be provided here
  sorry

end total_pushups_first_week_l407_407680


namespace option_A_option_B_option_C_option_D_l407_407766

namespace Inequalities

theorem option_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  a + (1/a) > b + (1/b) :=
sorry

theorem option_B (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  (m + 1) / (n + 1) < m / n :=
sorry

theorem option_C (c a b : ℝ) (hc : c > 0) (ha : a > 0) (hb : b > 0) (hca : c > a) (hab : a > b) :
  a / (c - a) > b / (c - b) :=
sorry

theorem option_D (a b : ℝ) (ha : a > -1) (hb : b > -1) (hab : a ≥ b) :
  a / (a + 1) ≥ b / (b + 1) :=
sorry

end Inequalities

end option_A_option_B_option_C_option_D_l407_407766


namespace lattice_points_count_is_30_l407_407792

-- Define the function that counts the number of lattice points (x, y, z) such that x^2 + y^2 + z^2 = 25.
def count_lattice_points := {p : ℤ × ℤ × ℤ // p.1 ^ 2 + p.2.1 ^ 2 + p.2.2 ^ 2 = 25}.to_finset.card

-- The theorem to prove that the count is 30.
theorem lattice_points_count_is_30 : count_lattice_points = 30 :=
by
  sorry

end lattice_points_count_is_30_l407_407792


namespace rhombus_diagonal_l407_407887

def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_diagonal (d1 : ℝ) (area : ℝ) (h1 : d1 = 60) (h2 : area = 1950) : ∃ d2 : ℝ, d2 = 65 :=
by
  use 65
  simp [area_of_rhombus, h1, h2]
  sorry

end rhombus_diagonal_l407_407887


namespace problem1_l407_407445

theorem problem1 (x : ℝ) (h1 : y = sqrt x) (h2 : y' = 1/(2*sqrt x)) : 2 * y * y' = 1 :=
by
  sorry

end problem1_l407_407445


namespace max_value_of_a_l407_407622

variable {R : Type*} [LinearOrderedField R]

def det (a b c d : R) : R := a * d - b * c

theorem max_value_of_a (a : R) :
  (∀ x : R, det (x - 1) (a - 2) (a + 1) x ≥ 1) → a ≤ (3 / 2 : R) :=
by
  sorry

end max_value_of_a_l407_407622


namespace tangent_line_ln_slope_l407_407710

theorem tangent_line_ln_slope :
  (∃ k : ℝ, ∀ x t : ℝ, y = ln(x) → t = k*x → t = x → y' = 1/x) → k = 1 / Real.exp 1 :=
  sorry

end tangent_line_ln_slope_l407_407710


namespace line_passes_through_quadrants_l407_407697

theorem line_passes_through_quadrants (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c > 0) :
  (ax + by - c = 0) → (passes_through_quadrants (hline := (ax + by - c = 0)) {1, 3, 4}) :=
by sorry

end line_passes_through_quadrants_l407_407697


namespace range_of_m_l407_407137

theorem range_of_m (m x : ℝ) :
  (m-1 < x ∧ x < m+1) → (2 < x ∧ x < 6) → (3 ≤ m ∧ m ≤ 5) :=
by
  intros hp hq
  sorry

end range_of_m_l407_407137


namespace line_intersects_circle_l407_407148

noncomputable def diameter : ℝ := 8
noncomputable def radius : ℝ := diameter / 2
noncomputable def center_to_line_distance : ℝ := 3

theorem line_intersects_circle :
  center_to_line_distance < radius → True :=
by {
  /- The proof would go here, but for now, we use sorry. -/
  sorry
}

end line_intersects_circle_l407_407148


namespace distance_between_cars_l407_407888

theorem distance_between_cars 
  (initial_distance : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ) :
  initial_distance = 200 ∧ speed_car1 = 60 ∧ speed_car2 = 80 →
  (distance_between_cars_in_one_hour initial_distance speed_car1 speed_car2 = 60 ∨
  distance_between_cars_in_one_hour initial_distance speed_car1 speed_car2 = 340 ∨
  distance_between_cars_in_one_hour initial_distance speed_car1 speed_car2 = 180 ∨
  distance_between_cars_in_one_hour initial_distance speed_car1 speed_car2 = 220) :=
sorry

def distance_between_cars_in_one_hour (initial_distance speed_car1 speed_car2 : ℝ) : ℝ :=
  if speed_car1 + speed_car2 = speed_car1 + speed_car2 then 
  initial_distance - (speed_car1 + speed_car2) * 1
  else if speed_car1 + speed_car2 = speed_car1 + speed_car2 then 
  initial_distance + (speed_car1 + speed_car2) * 1
  else if speed_car2 - speed_car1 = speed_car2 - speed_car1 then 
  initial_distance - (speed_car2 - speed_car1) * 1
  else 
  initial_distance + (speed_car2 - speed_car1) * 1

end distance_between_cars_l407_407888


namespace approval_count_l407_407399

theorem approval_count (total_sample : ℕ) (oppose_pct indifferent_pct: ℝ) :
  total_sample = 600 →
  oppose_pct = 0.06 →
  indifferent_pct = 0.14 →
  (total_sample * (1 - oppose_pct - indifferent_pct) = 480) := 
begin
  intros h1 h2 h3,
  have terms_sum : 1 - oppose_pct - indifferent_pct = 0.80,
  { rw [h2, h3], norm_num },

  rw h1,
  have total_approvers := 600 * 0.80,
  norm_num at total_approvers,

  rw terms_sum,
  exact total_approvers,
end

end approval_count_l407_407399


namespace ratio_of_intercepts_l407_407931

theorem ratio_of_intercepts
  (u v : ℚ)
  (h1 : 2 = 5 * u)
  (h2 : 3 = -7 * v) :
  u / v = -14 / 15 :=
by
  sorry

end ratio_of_intercepts_l407_407931


namespace stripeTailedPermutationsCount_l407_407447

def isStripeTailed (a_1 a_2 a_3 a_4 a_5 : ℕ) : Prop :=
  a_1 + a_2 < a_3 + 2 * a_4 + a_5

def permute (l : List ℕ) : List (List ℕ) := 
  l.permutations

theorem stripeTailedPermutationsCount : 
  ∃ n : ℕ, n = 56 ∧ 
  (List.filter 
    (λ (l : List ℕ), isStripeTailed l.head l.tail.head l.tail.tail.head l.tail.tail.tail.head l.tail.tail.tail.tail.head)
    (permute [1, 2, 3, 4, 5])).length = n :=
by
  sorry

end stripeTailedPermutationsCount_l407_407447


namespace wall_height_proof_l407_407511

-- The dimensions of the brick in meters
def brick_length : ℝ := 0.30
def brick_width : ℝ := 0.12
def brick_height : ℝ := 0.10

-- The dimensions of the wall in meters
def wall_length : ℝ := 6
def wall_width : ℝ := 4

-- The number of bricks needed
def number_of_bricks : ℝ := 1366.6666666666667

-- The height of the wall in meters
def wall_height : ℝ := 0.205

-- The volume of one brick
def volume_of_one_brick : ℝ := brick_length * brick_width * brick_height

-- The total volume of all bricks needed
def total_volume_of_bricks : ℝ := number_of_bricks * volume_of_one_brick

-- The volume of the wall
def volume_of_wall : ℝ := wall_length * wall_width * wall_height

-- Proof that the height of the wall is 0.205 meters
theorem wall_height_proof : volume_of_wall = total_volume_of_bricks :=
by
  -- use definitions to evaluate the equality
  sorry

end wall_height_proof_l407_407511


namespace max_value_2019m_2020n_l407_407120

theorem max_value_2019m_2020n (m n : ℤ) (h1 : 0 ≤ m - n) (h2 : m - n ≤ 1) (h3 : 2 ≤ m + n) (h4 : m + n ≤ 4) :
  (∀ (m' n' : ℤ), (0 ≤ m' - n') → (m' - n' ≤ 1) → (2 ≤ m' + n') → (m' + n' ≤ 4) → (m - 2 * n ≥ m' - 2 * n')) →
  2019 * m + 2020 * n = 2019 :=
by
  sorry

end max_value_2019m_2020n_l407_407120


namespace min_value_of_3x_plus_2y_l407_407274

theorem min_value_of_3x_plus_2y 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : 2 * x + 3 * y = 5 * x * y) 
  : 3 * x + 2 * y ≥ 5 := 
sorry

example (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 3 * y = 5 * x * y) 
    : ∃ (a : ℝ), a = 3 * x + 2 * y ∧ a ≥ 5 :=
begin
  use 3 * x + 2 * y,
  split,
  { refl, },
  { exact min_value_of_3x_plus_2y x y hx hy h, }
end

end min_value_of_3x_plus_2y_l407_407274


namespace volunteers_allocation_scheme_count_l407_407075

theorem volunteers_allocation_scheme_count :
  let volunteers := 6
  let groups_of_two := 2
  let groups_of_one := 2
  let pavilions := 4
  let calculate_combinations (n k : ℕ) := Nat.choose n k
  calculate_combinations volunteers 2 * calculate_combinations (volunteers - 2) 2 * 
  calculate_combinations pavilions 2 * Nat.factorial pavilions = 1080 := by
sorry

end volunteers_allocation_scheme_count_l407_407075


namespace MNPQ_is_parallelogram_l407_407370

theorem MNPQ_is_parallelogram
  (A B C D M N P Q : Type)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup M] [AddCommGroup N] [AddCommGroup P] [AddCommGroup Q]
  (ABCD_is_parallelogram : ∃ AB CD : line_segment A D, parallelogram (AB, CD))
  (equilateral_triangles :
    (∃ AM AB : line_segment A B, 
      triangle (AM, AB, BM) ∧ equilateral (AM, AB, BM)) ∧
    (∃ BN BC : line_segment B C,
      triangle (BN, BC, CN) ∧ equilateral (BN, BC, CN)) ∧
    (∃ CP CD : line_segment C D,
      triangle (CP, CD, DP) ∧ equilateral (CP, CD, DP)) ∧
    (∃ DQ DA : line_segment D A,
      triangle (DQ, DA, AQ) ∧ equilateral (DQ, DA, AQ))) :
  parallelogram ( M, N, P, Q) :=
sorry

end MNPQ_is_parallelogram_l407_407370


namespace least_total_cost_is_172_l407_407352

noncomputable def least_total_cost : ℕ :=
  let lcm := Nat.lcm (Nat.lcm 6 5) 8
  let strawberry_packs := lcm / 6
  let blueberry_packs := lcm / 5
  let cherry_packs := lcm / 8
  let strawberry_cost := strawberry_packs * 2
  let blueberry_cost := blueberry_packs * 3
  let cherry_cost := cherry_packs * 4
  strawberry_cost + blueberry_cost + cherry_cost

theorem least_total_cost_is_172 : least_total_cost = 172 := 
by
  sorry

end least_total_cost_is_172_l407_407352


namespace sin_ratio_two_area_two_sqrt_three_l407_407280

variables {A B C a b c : ℝ}

-- Condition:
def triangle_condition (C cosB : ℝ) (c b : ℝ) : Prop :=
  cos C + (c / b) * cosB = 2

-- Proof tasks:
noncomputable def sin_ratio (A B : ℝ) : ℝ :=
  sin A / sin B

noncomputable def area_of_triangle (a b C : ℝ) : ℝ :=
  1 / 2 * a * b * sin C

-- Questions translated to Lean statements:
theorem sin_ratio_two (C cosB : ℝ) (c b : ℝ) 
  (h1: triangle_condition C cosB c b) : 
    ∃ A B : ℝ, sin_ratio A B = 2 :=
sorry

theorem area_two_sqrt_three (b a : ℝ) : 
  C = π / 3 → c = 2 * sqrt 3 → 
  area_of_triangle a b C = 2 * sqrt 3 :=
sorry

end sin_ratio_two_area_two_sqrt_three_l407_407280


namespace solution_set_part1_solution_set_part2_l407_407160

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 1

theorem solution_set_part1 :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem solution_set_part2 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end solution_set_part1_solution_set_part2_l407_407160


namespace product_of_numbers_l407_407106

theorem product_of_numbers (a b : ℝ) 
  (h1 : a + b = 5 * (a - b))
  (h2 : a * b = 18 * (a - b)) : 
  a * b = 54 :=
by
  sorry

end product_of_numbers_l407_407106


namespace original_total_price_l407_407107

-- Definitions of the original prices
def original_price_candy_box : ℕ := 10
def original_price_soda : ℕ := 6
def original_price_chips : ℕ := 4
def original_price_chocolate_bar : ℕ := 2

-- Mathematical problem statement
theorem original_total_price :
  original_price_candy_box + original_price_soda + original_price_chips + original_price_chocolate_bar = 22 :=
by
  sorry

end original_total_price_l407_407107


namespace train_time_to_pass_platform_l407_407502

theorem train_time_to_pass_platform (train_length platform_length : ℕ) 
    (time_to_pass_tree : ℕ) 
    (time_to_pass_platform : ℝ)
    (h1 : train_length = 2400)
    (h2 : platform_length = 1800)
    (h3 : time_to_pass_tree = 90)
    (h4 : time_to_pass_platform ≈ 157.5) : Prop :=
  let speed := train_length / time_to_pass_tree in
  let total_distance := train_length + platform_length in
  time_to_pass_platform = total_distance / speed

-- This establishes the relationship between the distance, time, and speed, and proves that the calculated time to pass the platform is correct.

end train_time_to_pass_platform_l407_407502


namespace fibonacci_units_digit_l407_407897

def fibonacci (n : ℕ) : ℕ :=
match n with
| 0     => 4
| 1     => 3
| (n+2) => fibonacci (n+1) + fibonacci n

def units_digit (n : ℕ) : ℕ :=
n % 10

theorem fibonacci_units_digit : units_digit (fibonacci (fibonacci 10)) = 3 := by
  sorry

end fibonacci_units_digit_l407_407897


namespace number_of_positive_factors_of_60_l407_407246

theorem number_of_positive_factors_of_60 (n : ℕ) (h : n = 60) 
  (h_prime_fact : (60 = 2^2 * 3 * 5)) : 
  nat.totient_factors 60 = 12 :=
sorry

end number_of_positive_factors_of_60_l407_407246


namespace exists_point_lt_l407_407932

-- Definitions of polynomials P and Q
def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d
def Q (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

-- Main theorem statement
theorem exists_point_lt (a b c d p q : ℝ) (I : Set ℝ) (Hlength : I.length > 2)
  (HnegI_P : ∀ x ∈ I, P x a b c d < 0)
  (HnegI_Q : ∀ x ∈ I, Q x p q < 0)
  (Hpos_outside_P : ∀ x ∉ I, P x a b c d ≥ 0)
  (Hpos_outside_Q : ∀ x ∉ I, Q x p q ≥ 0) : 
  ∃ x₀, P x₀ a b c d < Q x₀ p q := 
by
  sorry

end exists_point_lt_l407_407932


namespace complex_number_quadrant_l407_407764

theorem complex_number_quadrant : 
  (let z : ℂ := 2 - 1 * Complex.I in z.im < 0 ∧ z.re > 0) -> "Fourth Quadrant" := sorry

end complex_number_quadrant_l407_407764


namespace correct_proposition_l407_407728

variable (p q : Prop)
variable (h_p : ¬ (∀ x : ℝ, 2^x < 3^x))
variable (h_q : ∃ x : ℝ, x^3 = 1 - x^2)

theorem correct_proposition : (¬ p ∧ q) :=
by
  have h_p_f : p ↔ false := sorry  -- Proof or known result that p is false
  have h_q_t : q ↔ true := sorry  -- Proof or known result that q is true
  exact and.intro (h_p_f.mp (not_not.mp h_p)) (h_q_t.mpr trivial)

end correct_proposition_l407_407728


namespace chickens_cheaper_than_buying_eggs_after_81_weeks_l407_407384

-- Definitions based on conditions
def cost_chickens (num_chickens : ℕ) (cost_per_chicken : ℕ) : ℕ := num_chickens * cost_per_chicken
def egg_production (num_chickens : ℕ) (eggs_per_chicken_per_week : ℕ) : ℕ := num_chickens * eggs_per_chicken_per_week
def weekly_savings (cost_per_dozen : ℕ) (weekly_feed_cost : ℕ) : ℕ := cost_per_dozen - weekly_feed_cost
def break_even_weeks (total_cost : ℕ) (weekly_savings : ℕ) : ℕ := total_cost / weekly_savings
def cheaper_than_after_weeks (break_even_weeks : ℕ) : ℕ := break_even_weeks + 1

-- Theorem to prove
theorem chickens_cheaper_than_buying_eggs_after_81_weeks :
  ∀ (cost_per_chicken weekly_feed_cost eggs_per_chicken_per_week cost_per_dozen num_chickens : ℕ),
  cost_per_chicken = 20 →
  weekly_feed_cost = 1 →
  eggs_per_chicken_per_week = 3 →
  cost_per_dozen = 2 →
  num_chickens = 4 →
  let total_cost := cost_chickens num_chickens cost_per_chicken,
      weekly_savings_amt := weekly_savings cost_per_dozen weekly_feed_cost,
      break_even := break_even_weeks total_cost weekly_savings_amt,
      weeks_needed := cheaper_than_after_weeks break_even
  in weeks_needed = 81 :=
begin
  intros,
  sorry
end

end chickens_cheaper_than_buying_eggs_after_81_weeks_l407_407384


namespace lattice_points_with_distance_5_from_origin_l407_407801

theorem lattice_points_with_distance_5_from_origin :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 42 :=
by
  sorry

end lattice_points_with_distance_5_from_origin_l407_407801


namespace max_consecutive_integers_sum_le_500_l407_407459

def consecutive_sum (n : ℕ) : ℕ :=
  -- Formula for sum starting from 3
  (n * (n + 1)) / 2 - 3

theorem max_consecutive_integers_sum_le_500 : ∃ n : ℕ, consecutive_sum n ≤ 500 ∧ ∀ m : ℕ, m > n → consecutive_sum m > 500 :=
by
  sorry

end max_consecutive_integers_sum_le_500_l407_407459


namespace find_gx_l407_407163

noncomputable def g (x : ℝ) : ℝ :=
2^(x+1) * x^2

noncomputable def f (x : ℝ) : ℝ :=
2 * x^2 - 8 * x - 4

theorem find_gx (x : ℝ) : g(x) = 2^(x+1) * (x^2 - 4 * x - 2) :=
by
  sorry

end find_gx_l407_407163


namespace shortest_distance_is_correct_l407_407655

noncomputable def shortest_distance_to_hyperbola (a : ℝ) (h : a ≠ 0) : ℝ :=
  let point_on_hyperbola := (a, 1 / a)
  real.sqrt ((-5 - 5) ^ 2 + ((-1 / 5) - 15) ^ 2)

theorem shortest_distance_is_correct (a : ℝ) (h : a ≠ 0) : shortest_distance_to_hyperbola a h = 53 / 5 := 
  sorry

end shortest_distance_is_correct_l407_407655


namespace walk_to_bus_stop_time_l407_407446

/-- Walking with 4/5 of my usual speed, I arrive at the bus stop 7 minutes later than normal.
    How many minutes does it take to walk to the bus stop at my usual speed? -/
theorem walk_to_bus_stop_time (S T : ℝ) (h : T > 0) 
  (d_usual : S * T = (4/5) * S * (T + 7)) : 
  T = 28 :=
by
  sorry

end walk_to_bus_stop_time_l407_407446


namespace isosceles_triangles_l407_407847

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangles (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c > 0)
    (H : ∀ n : ℕ, a ^ n + b ^ n > c ^ n ∧ b ^ n + c ^ n > a ^ n ∧ c ^ n + a ^ n > b ^ n) :
    is_isosceles_triangle a b c :=
  sorry

end isosceles_triangles_l407_407847


namespace not_enough_money_l407_407948

-- Define the prices of the books
def price_animal_world : Real := 21.8
def price_fairy_tale_stories : Real := 19.5

-- Define the total amount of money Xiao Ming has
def xiao_ming_money : Real := 40.0

-- Define the statement we want to prove
theorem not_enough_money : (price_animal_world + price_fairy_tale_stories) > xiao_ming_money := by
  sorry

end not_enough_money_l407_407948


namespace problem1_solution_problem2_solution_l407_407161

-- Define the function f(x) when a = 2
def f1 (x : ℝ) : ℝ := |x + 4| + |x - 4|

-- Define the inequality f(x) ≤ 13
def inequality_f1 (x : ℝ) : Prop := f1 x ≤ 13

-- Statement (1): Prove the solution set of the inequality when a = 2
theorem problem1_solution (x : ℝ) : inequality_f1 x ↔ -13/2 ≤ x ∧ x ≤ 13/2 :=
by
  sorry

-- Define the general function f(x) for any a
def f2 (x a: ℝ) : ℝ := |x + 4| + |x - 2 * a|

-- Define the inequality f(x) ≥ a^2 + 5a
def inequality_f2 (x a: ℝ) : Prop := f2 x a ≥ a^2 + 5 * a

-- Statement (2): Prove the range of the real number a such that the inequality always holds
theorem problem2_solution (a: ℝ) : (∀ x, inequality_f2 x a) → a ∈ set.Icc ((-7 - real.sqrt 33) / 2) 1 :=
by
  sorry

end problem1_solution_problem2_solution_l407_407161


namespace angles_of_triangle_PQK_l407_407860

theorem angles_of_triangle_PQK {A B C P Q K : Type} [Inhabited A] [Inhabited B] [Inhabited C]
  (β φ : ℝ) (AC_midpoint : midpoint A C K) (AB_side : side A B) (BC_side : side B C)
  (APB_right_triangle : right_triangle A P B) (BQC_right_triangle : right_triangle B Q C)
  (∠APB = φ) (∠BQC = φ) :
  ∠PKQ = 180 - 2 * φ ∧ ∠QPK = φ ∧ ∠PQK = φ := by
sorry

end angles_of_triangle_PQK_l407_407860


namespace proof_original_problem_l407_407118

-- Definition of f(x)
def f (x : ℝ) : ℝ := 1 / x * Real.cos x

-- Formal problem statement proving the original question's answer
theorem proof_original_problem :
  f π + (deriv f) (π / 2) = -3 / π :=
by
  sorry

end proof_original_problem_l407_407118


namespace tamil_speakers_l407_407757

variables (P E B H T : ℕ)
variables (Prob_Hindi : ℝ)

-- Total number of people in the village
def population := P = 1024

-- Number of English speakers
def english_speakers := E = 562

-- Number of people who speak both English and Tamil
def both_speakers := B = 346

-- Probability of a person speaking Hindi
def hindi_probability := Prob_Hindi = 0.0859375

-- Number of Hindi speakers
def hindi_speakers := H = (P : ℝ) * Prob_Hindi

-- The objective to prove
theorem tamil_speakers : 
  population P →
  english_speakers E →
  both_speakers B →
  hindi_probability Prob_Hindi →
  hindi_speakers P Prob_Hindi H →
  T = 720 :=
by
  sorry

end tamil_speakers_l407_407757


namespace problem_statement_l407_407363

namespace LeanProofExample

def not_divisible (n : ℕ) (p : ℕ) : Prop :=
  ¬(p ∣ n)

theorem problem_statement (x y : ℕ) 
  (hx : not_divisible x 59) 
  (hy : not_divisible y 59)
  (h : 3 * x + 28 * y ≡ 0 [MOD 59]) :
  ¬(5 * x + 16 * y ≡ 0 [MOD 59]) :=
  sorry

end LeanProofExample

end problem_statement_l407_407363


namespace find_x_that_satisfies_f_l407_407893

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem find_x_that_satisfies_f (α : ℝ) (x : ℝ) (h : power_function α (-2) = -1/8) : 
  power_function α x = 27 → x = 1/3 :=
  by
  sorry

end find_x_that_satisfies_f_l407_407893


namespace sequence_periodicity_a_2014_eq_1_l407_407116

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then -1 / (1 + sequence 0)
  else -1 / (1 + sequence (n - 1))

theorem sequence_periodicity :
  ∀ n, sequence (n + 3) = sequence n :=
λ n, by sorry

theorem a_2014_eq_1 : sequence 2014 = 1 :=
by {
  have h1 : 2014 % 3 = 1 := by norm_num,
  have h2 : ∀ k, sequence (3 * k + 1) = sequence 1 := by {
    intro k,
    induction k with k ih,
    { refl },
    {
      rw [Nat.mul_succ, ← Nat.add_assoc, sequence_periodicity],
      exact ih,
    },
  },
  rw [Nat.div_add_mod, h1, add_comm, Nat.div_mul_cancel (by norm_num : 2014 / 3)],
  exact h2 (2014 / 3),
}

end sequence_periodicity_a_2014_eq_1_l407_407116


namespace three_digit_number_property_l407_407645

theorem three_digit_number_property (A : ℕ) (a b c : ℕ) :
  A = 100 * a + 10 * b + c →
  A ∈ {n | n = 111 ∨ n = 222 ∨ n = 333 ∨ n = 444 ∨ n = 555 ∨ n = 666 ∨
           n = 777 ∨ n = 888 ∨ n = 999 ∨ n = 407 ∨ n = 518 ∨ n = 629 ∨
           n = 370 ∨ n = 481 ∨ n = 592} ↔
  222 * (a + b + c) = 600 * a + 60 * b + 6 * c :=
begin
  sorry
end

end three_digit_number_property_l407_407645


namespace total_pushups_first_week_l407_407679

-- Definitions based on conditions
def first_day_pushups : ℕ := 10
def increment_per_day : ℕ := 5
def days_workout : ℕ := 3

-- Function to calculate the number of push-ups on a specific day
def pushups_on_day (day : ℕ) : ℕ :=
  first_day_pushups + (day - 1) * increment_per_day

-- Function to calculate total push-ups in the first week
def total_pushups_week : ℕ :=
  ∑ i in Finset.range days_workout, pushups_on_day (i + 1)

-- Theorem to prove the total number of push-ups in the first week
theorem total_pushups_first_week : total_pushups_week = 45 := by
  -- Proof will be provided here
  sorry

end total_pushups_first_week_l407_407679


namespace circle_radius_l407_407937

theorem circle_radius : 
  ∃ (r : ℝ), (r = 2.5 ∧ ∃ y : ℝ, (0 - 0)^2 + (5 - y)^2 = r^2 ∧ (2 - 0)^2 + (1 - y)^2 = r^2 ) :=
begin
  sorry
end

end circle_radius_l407_407937


namespace reflect_across_x_axis_l407_407886

theorem reflect_across_x_axis (x y : ℝ) (P : ℝ × ℝ) (hP : P = (-2, 1)) : 
    let Q := (x, -y) in
    Q = (-2, -1) := 
by
  -- Proof omitted
  sorry

end reflect_across_x_axis_l407_407886


namespace proof_mn_eq_9_l407_407275

theorem proof_mn_eq_9 (m n : ℕ) (h1 : 2 * m + n = 8) (h2 : m - n = 1) : m^n = 9 :=
by {
  sorry 
}

end proof_mn_eq_9_l407_407275


namespace equivalent_functions_l407_407481

theorem equivalent_functions :
  ∀ (x t : ℝ), (x^2 - 2*x - 1 = t^2 - 2*t + 1) := 
by
  intros x t
  sorry

end equivalent_functions_l407_407481


namespace binom_30_3_eq_4060_l407_407573

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l407_407573


namespace binomial_coefficient_30_3_l407_407593

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l407_407593


namespace correct_average_l407_407491

theorem correct_average (n : ℕ) (wrong_avg : ℕ) (wrong_num correct_num : ℕ) (correct_avg : ℕ)
  (h1 : n = 10) 
  (h2 : wrong_avg = 21)
  (h3 : wrong_num = 26)
  (h4 : correct_num = 36)
  (h5 : correct_avg = 22) :
  (wrong_avg * n + (correct_num - wrong_num)) / n = correct_avg :=
by
  sorry

end correct_average_l407_407491


namespace factors_of_60_l407_407228

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l407_407228


namespace simplify_complex_l407_407387

theorem simplify_complex : (4 - 3 * complex.i)^3 = -44 - 117 * complex.i := 
by 
  have h : complex.i^2 = -1 := by sorry
  sorry

end simplify_complex_l407_407387


namespace original_loaf_slices_l407_407628

-- Define the given conditions
def andy_slices_1 := 3
def andy_slices_2 := 3
def toast_slices_per_piece := 2
def pieces_of_toast := 10
def slices_left_over := 1

-- Define the variables
def total_andy_slices := andy_slices_1 + andy_slices_2
def total_toast_slices := toast_slices_per_piece * pieces_of_toast

-- State the theorem
theorem original_loaf_slices : 
  ∃ S : ℕ, S = total_andy_slices + total_toast_slices + slices_left_over := 
by {
  sorry
}

end original_loaf_slices_l407_407628


namespace Yeonseo_skirts_l407_407483

theorem Yeonseo_skirts
  (P : ℕ)
  (more_than_two_skirts : ∀ S : ℕ, S > 2)
  (more_than_two_pants : P > 2)
  (ways_to_choose : P + 3 = 7) :
  ∃ S : ℕ, S = 3 := by
  sorry

end Yeonseo_skirts_l407_407483


namespace jacob_writing_speed_ratio_l407_407819

theorem jacob_writing_speed_ratio (N : ℕ) (J : ℕ) (hN : N = 25) (h1 : J + N = 75) : J / N = 2 :=
by {
  sorry
}

end jacob_writing_speed_ratio_l407_407819


namespace num_factors_60_l407_407182

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l407_407182


namespace magnitude_of_T_l407_407830

noncomputable theory

open Complex

def T := (1 + sqrt 3 * I)^13 - (1 - sqrt 3 * I)^13

theorem magnitude_of_T : abs T = 8192 * sqrt 3 := by
  -- Proof will go here
  sorry

end magnitude_of_T_l407_407830


namespace sin_cos_105_l407_407914

theorem sin_cos_105 (h1 : ∀ x : ℝ, Real.sin x * Real.cos x = 1 / 2 * Real.sin (2 * x))
                    (h2 : ∀ x : ℝ, Real.sin (180 * Real.pi / 180 + x) = - Real.sin x)
                    (h3 : Real.sin (30 * Real.pi / 180) = 1 / 2) :
  Real.sin (105 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) = - 1 / 4 :=
by
  sorry

end sin_cos_105_l407_407914


namespace blocks_for_house_l407_407380

-- Definitions based on given conditions
variable (RandyBlocks: ℕ)
variable (BlocksForHouse: ℕ)
variable (BlocksForTower: ℕ)

-- Given conditions
axiom h1 : RandyBlocks = 90
axiom h2 : BlocksForTower = 63
axiom h3 : BlocksForHouse = BlocksForTower + 26

-- Proving that Randy used 89 blocks to build the house
theorem blocks_for_house : BlocksForHouse = 89 := by
  rw [h2, h3]
  simp
  sorry

end blocks_for_house_l407_407380


namespace find_vector_c_l407_407178

variable (a b c : ℝ × ℝ)
variable (x y : ℝ)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_vector_c :
  let a := (1, 2)
  let b := (2, -3)
  let c := (-7 / 9, -7 / 3) in
  parallel (c.1 + a.1, c.2 + a.2) b ∧
  perpendicular c (a.1 + b.1, a.2 + b.2) :=
by
  let a := (1, 2)
  let b := (2, -3)
  let c := (-7 / 9, -7 / 3)
  unfold parallel perpendicular
  sorry

end find_vector_c_l407_407178


namespace num_factors_of_60_l407_407199

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l407_407199


namespace simplify_correct_l407_407389

def simplify_polynomial (x : Real) : Real :=
  (12 * x^10 + 6 * x^9 + 3 * x^8) + (2 * x^11 + x^10 + 4 * x^9 + x^7 + 4 * x^4 + 7 * x + 9)

theorem simplify_correct (x : Real) :
  simplify_polynomial x = 2 * x^11 + 13 * x^10 + 10 * x^9 + 3 * x^8 + x^7 + 4 * x^4 + 7 * x + 9 :=
by
  sorry

end simplify_correct_l407_407389


namespace floor_sum_sqrt_25_l407_407010

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l407_407010


namespace distinct_real_roots_l407_407667

theorem distinct_real_roots (a : ℝ) :
  let A := a - 3 in
  let B := -4 in
  let C := -1 in
  (a ≠ 3 ∧ 4 * a + 4 > 0) ↔ (a > -1 ∧ a ≠ 3) :=
by simp [A, B, C]; rink sorry

end distinct_real_roots_l407_407667


namespace trig_identity_eq_zero_l407_407915

theorem trig_identity_eq_zero (α : ℝ) : 
  sin^2 (π + α) + cos (2 * π + α) * cos (-α) - 1 = 0 := 
by
  sorry

end trig_identity_eq_zero_l407_407915


namespace cost_of_first_house_l407_407314

theorem cost_of_first_house (C : ℝ) (h₀ : 2 * C + C = 600000) : C = 200000 := by
  -- proof placeholder
  sorry

end cost_of_first_house_l407_407314


namespace sum_of_squares_l407_407392

noncomputable def equation (x : ℝ) : Prop :=
  (x^2 - 2*x + 4)^(x^2 - 2*x + 3) = 625

theorem sum_of_squares (x : ℝ) :
  (equation x) → 
  ((1 + Real.sqrt 2)^2 + (1 - Real.sqrt 2)^2 = 6) :=
begin
  sorry
end

end sum_of_squares_l407_407392


namespace sum_of_floors_of_square_roots_l407_407016

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l407_407016


namespace sin_eq_sin_sin_no_solution_l407_407264

open Real

theorem sin_eq_sin_sin_no_solution (h : ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 → sin θ > θ) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ asin 1000 → ¬ (sin x = sin (sin x)) :=
by
  intro x hx
  sorry -- proof goes here

end sin_eq_sin_sin_no_solution_l407_407264


namespace binom_30_3_l407_407599

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l407_407599


namespace trig_identity_l407_407266

theorem trig_identity (x : ℝ) (h1 : 0 < x) (h2 : x < π/2) (h3 : sin x / (1 + cos x) = 1 / 3) : 
  sin (2 * x) / (1 + cos (2 * x)) = 3 / 4 :=
by sorry

end trig_identity_l407_407266


namespace binom_30_3_l407_407615

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l407_407615


namespace binom_30_3_l407_407596

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l407_407596


namespace rise_ratio_of_water_levels_l407_407442

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * π * r^3

theorem rise_ratio_of_water_levels :
  ∀ h₁ h₂ : ℝ,
  ∀ r₁ r₂ : ℝ,
  ∀ s₁ s₂ : ℝ,
  let h₁ := 4 * h₂,
  let r₁ := 4,
  let r₂ := 8,
  sphere_volume s₁ = sphere_volume 2 ∧ sphere_volume s₂ = sphere_volume 2 →
  h₁ + (sphere_volume 2 * 3 / (π * r₁^2)) = 4 * h₂ + (sphere_volume 2 * 3 / (π * r₂^2)) ∧
  h₂ + (sphere_volume 2 * 3 / (π * r₂^2)) = h₂ + 1/6 →
  h₁ + (sphere_volume 2 * 3 / (π * r₁^2)) - h₁ = 4 * (h₂ + 1/6 - h₂) :=
  sorry

end rise_ratio_of_water_levels_l407_407442


namespace shaded_area_of_square_with_circles_l407_407371

theorem shaded_area_of_square_with_circles (side_length : ℝ)
  (h₁ : side_length = 50)
  (radius := side_length / (2 * 3))
  (area_square := side_length ^ 2)
  (area_circle := Real.pi * radius ^ 2)
  (n_circles := 3)
  (total_area_circles := n_circles * area_circle) :
  let shaded_area := area_square - total_area_circles in
  shaded_area = 2500 - 1875 * Real.pi / 9 := by
  sorry

end shaded_area_of_square_with_circles_l407_407371


namespace binom_30_3_l407_407598

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l407_407598


namespace total_amount_paid_is_correct_l407_407548

def rate_per_kg_grapes := 98
def quantity_grapes := 15
def rate_per_kg_mangoes := 120
def quantity_mangoes := 8
def rate_per_kg_pineapples := 75
def quantity_pineapples := 5
def rate_per_kg_oranges := 60
def quantity_oranges := 10

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes
def cost_pineapples := rate_per_kg_pineapples * quantity_pineapples
def cost_oranges := rate_per_kg_oranges * quantity_oranges

def total_amount_paid := cost_grapes + cost_mangoes + cost_pineapples + cost_oranges

theorem total_amount_paid_is_correct : total_amount_paid = 3405 := by
  sorry

end total_amount_paid_is_correct_l407_407548


namespace explicit_formula_l407_407121

noncomputable def a : ℕ → ℤ
| 1       := 1
| (n + 1) := 2 * a n + (n^2)

theorem explicit_formula (n : ℕ) : a n = 7 * 2^(n - 1) - n^2 - 2 * n - 3 := sorry

end explicit_formula_l407_407121


namespace sum_floor_sqrt_1_to_25_l407_407038

theorem sum_floor_sqrt_1_to_25 : (Finset.sum (Finset.range 25) (λ n, n.sqrt.floor)) + 1 = 71 :=
by 
  have h1 : (Finset.range 25).sum (λ n, if 1 <= n +1 ∧ n +1 < 4 then 1 else 0) = 3, by sorry,
  have h2 : (Finset.range 25).sum (λ n, if 4 <= n +1 ∧ n +1 < 9 then 2 else 0) = 10, by sorry,
  have h3 : (Finset.range 25).sum (λ n, if 9 <= n +1 ∧ n +1 < 16 then 3 else 0) = 21, by sorry,
  have h4 : (Finset.range 25).sum (λ n, if 16 <= n +1 ∧ n +1 < 25 then 4 else 0) = 32, by sorry,
  have h5 : (Finset.range 25).sum (λ n, if n +1 = 25 then 5 else 0) = 5, by sorry,
  exact h1 + h2 + h3 + h4 + h5

end sum_floor_sqrt_1_to_25_l407_407038


namespace number_of_factors_of_60_l407_407205

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l407_407205


namespace determine_angle_A_l407_407747

theorem determine_angle_A 
  (a b c : ℝ)
  (A B C : ℝ) 
  (h_triangle : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_relation : 2 * a * sin A = (2 * b + c) * sin B + (2 * c + b) * sin C) :
  A = 2 * π / 3 :=
sorry

end determine_angle_A_l407_407747


namespace max_consecutive_integers_sum_500_l407_407452

theorem max_consecutive_integers_sum_500 : ∀ S k max_n,
  (∀ (n : ℕ), S n = n * k + n * (n - 1) / 2) →
  (k = 3) →
  (∀ (n : ℕ), 2 * S n ≤ 1000) →
  max_n = 29 :=
by
  intros S k max_n S_def hk hineq
  sorry

end max_consecutive_integers_sum_500_l407_407452


namespace true_propositions_l407_407696

-- Definitions according to conditions:
def p (x y : ℝ) : Prop := x > y → -x < -y
def q (x y : ℝ) : Prop := x > y → x^2 > y^2

-- Given that p is true and q is false.
axiom p_true {x y : ℝ} : p x y
axiom q_false {x y : ℝ} : ¬ q x y

-- Proving the actual propositions that are true:
theorem true_propositions (x y : ℝ) : 
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  have h1 : p x y := p_true
  have h2 : ¬ q x y := q_false
  constructor
  · left; exact h1
  · constructor; assumption; assumption

end true_propositions_l407_407696


namespace value_of_six_in_sequence_l407_407291

theorem value_of_six_in_sequence :
  (∀ n, n ∈ {1, 2, 3, 4, 5} → sequence_value n = n * 6) →
  sequence_value 6 = 36 :=
by
  intro h
  have : sequence_value 6 = 6 * 6 := by sorry
  rw this
  exact rfl

noncomputable def sequence_value (n : ℕ) : ℕ := n * 6

end value_of_six_in_sequence_l407_407291


namespace initial_yards_lost_l407_407977

theorem initial_yards_lost (x : ℤ) (h : -x + 7 = 2) : x = 5 := by
  sorry

end initial_yards_lost_l407_407977


namespace purely_imaginary_z_quadrant_3_or_4_z_l407_407841

noncomputable def z (m : ℝ) : ℂ :=
  let i : ℂ := complex.I
  m^2 * (1 / (m + 8) + i) + (6 * m - 16) * i - (m + 2) / (m + 8)

theorem purely_imaginary_z (m : ℝ) :
  (∃ (b : ℂ), z m = b * complex.I) → m = -1 :=
sorry

theorem quadrant_3_or_4_z (m : ℝ) :
  (Re (z m) ≠ 0 ∧ Im (z m) < 0) → m ∈ set.Ioo (-8 : ℝ) 2 ∪ {-1} :=
sorry

end purely_imaginary_z_quadrant_3_or_4_z_l407_407841


namespace biknis_fraction_l407_407550

-- Definitions of the fractions.
variable (T : ℝ) [fact (T = 0.25)] 
variable (B : ℝ) 
variable (E : ℝ) [fact (E = 0.63)]

-- The theorem stating the condition and what we want to prove.
theorem biknis_fraction : E = B + T → B = 0.38 :=
by
  intros h
  rw [fact.out (E = 0.63)] at h
  rw [fact.out (T = 0.25)] at h
  exact sorry

end biknis_fraction_l407_407550


namespace integer_values_a_count_l407_407698

theorem integer_values_a_count (a b : ℝ) (hb : 1 ≤ b ∧ b ≤ 4) (h : a + 8 * b - 2 * b^2 = 7) : 
  ∃ (S : set ℤ), S = {x : ℤ | -1 ≤ x ∧ x ≤ 7} ∧ S.card = 9 := 
by 
  sorry

end integer_values_a_count_l407_407698


namespace num_factors_of_60_l407_407257

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l407_407257


namespace monotone_on_interval_and_extreme_values_l407_407379

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem monotone_on_interval_and_extreme_values :
  (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2) → f x1 > f x2) ∧ (f 1 = 5 ∧ f 2 = 4) := 
by
  sorry

end monotone_on_interval_and_extreme_values_l407_407379


namespace binomial_30_3_l407_407575

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l407_407575


namespace price_of_first_variety_l407_407398

theorem price_of_first_variety
  (P : ℝ)
  (H1 : 1 * P + 1 * 135 + 2 * 175.5 = 4 * 153) :
  P = 126 :=
by
  sorry

end price_of_first_variety_l407_407398


namespace binom_30_3_eq_4060_l407_407607

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l407_407607


namespace P_inter_Q_subset_P_l407_407397
open Set

def P : Set ℕ := {1, 2, 3, 4, 5, 6}
def Q : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}

theorem P_inter_Q_subset_P : (P ∩ Q) ⊆ P := by sorry

end P_inter_Q_subset_P_l407_407397


namespace sin_x_values_l407_407122

theorem sin_x_values (x : ℝ) (hx1 : 0 < x) (hx2 : x < real.pi)
    (h : real.sin (x + real.arccos (4 / 5)) = real.sqrt 3 / 2) :
    real.sin x = (1 / 10) * (4 * real.sqrt 3 - 3) ∨ real.sin x = (1 / 10) * (4 * real.sqrt 3 + 3) :=
by
  sorry

end sin_x_values_l407_407122


namespace students_not_receive_gift_l407_407500

-- Define the main properties given in the problem:
def total_students := 2016

-- Define the problem statement in Lean, which includes checking the count of students
-- who do not receive a New Year's gift.
theorem students_not_receive_gift : 
  ∀ τ : ℕ, τ = total_students → 
  ∀ n : ℕ, 
  (2 ≤ n ∧ τ % n = 0) →
  ∃ k : ℕ, ∀ m : ℕ, m = φ(τ) → m = 576 :=
by
  -- Here φ(τ) refers to Euler's Totient function, which we assume to be defined.
  sorry

end students_not_receive_gift_l407_407500


namespace sum_arithmetic_sequence_100_to_110_l407_407468

theorem sum_arithmetic_sequence_100_to_110 :
  let a := 100
  let l := 110
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 1155 := by
  sorry

end sum_arithmetic_sequence_100_to_110_l407_407468


namespace num_lattice_points_distance_five_l407_407810

open_locale classical 

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points_distance_five : 
  {n : ℕ | ∃ (x y z : ℤ), is_lattice_point x y z} = 78 :=
by sorry

end num_lattice_points_distance_five_l407_407810


namespace sqrt_defined_iff_nonneg_l407_407302

theorem sqrt_defined_iff_nonneg (x : ℝ) : (∃ y : ℝ, y * y = x) ↔ x ≥ 0 := 
by sorry

example : ¬ ∃ y : ℝ, y * y = -1 := 
by {
  have : (-1 : ℝ) < 0 := by norm_num,
  rw sqrt_defined_iff_nonneg at this,
  exact this
}

end sqrt_defined_iff_nonneg_l407_407302


namespace floor_sqrt_sum_l407_407043

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l407_407043


namespace correct_conclusions_l407_407134

def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r, ∀ n, seq (n + 1) = r * seq n

def is_arithmetic_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) = seq n + d

def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 2 * n + 1

noncomputable def x_n (n : ℕ) : Finset ℕ :=
  (Finset.range (2^(n + 1) + 1)).filter (λ m, b n ≤ m ∧ m ≤ a (n + 1))

theorem correct_conclusions : 
  (is_geometric_sequence (λ n, a (b n))) ∧
  (¬ is_arithmetic_sequence (λ n, b (a n))) ∧
  (x_n.card = 2^(n+1) - 2*n) ∧
  ((Finset.sort (a '' (Finset.range 2023) ∪ b '' (Finset.range 2023)) ! 2022) = 4025)
:= 
by
  sorry

end correct_conclusions_l407_407134


namespace sufficient_condition_l407_407136

-- Define the sequence and condition that each term is greater than zero.
variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, a n > 0)

-- Define what it means for the sequence to be geometric with common ratio q > 0.
def is_geometric (q : ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

-- Define the condition given in the problem
def condition (a1 a8 a4 a5 : ℝ) : Prop :=
  a1 + a8 < a4 + a5

-- State the theorem to prove
theorem sufficient_condition (h : ∀ n, a n > 0) : 
  (condition (a 1) (a 8) (a 4) (a 5)) → ¬ is_geometric a :=
by 
  sorry

end sufficient_condition_l407_407136


namespace students_present_entire_duration_l407_407430

def total_students := 28
def fraction_absent := 2 / 7
def total_class_hours := 4
def min_hours_present := 3 * total_class_hours / 4
def students_left_early := 3
def students_late := 2

def number_of_students_present_for_entire_duration : Nat :=
  total_students - (fraction_absent * total_students).toNat - students_left_early + students_late

theorem students_present_entire_duration : number_of_students_present_for_entire_duration = 17 := by
  sorry

end students_present_entire_duration_l407_407430


namespace num_solutions_ffx_eq_3_l407_407876

def f (x : ℝ) : ℝ :=
  if h : -3 ≤ x ∧ x ≤ 0 then x^2 + 3 * x + 2
  else if h : 0 < x ∧ x ≤ 3 then -x + 3
  else if h : 3 < x ∧ x ≤ 5 then 2 * x - 8
  else 0 -- although this last case falls outside of our specified domain, Lean requires a possible default return type

theorem num_solutions_ffx_eq_3 : 
  let solutions := {x : ℝ | -3 ≤ x ∧ x ≤ 5 ∧ f (f x) = 3} in 
  solutions.to_finset.card = 2 :=
by
  sorry

end num_solutions_ffx_eq_3_l407_407876


namespace fraction_sum_simplest_terms_l407_407412

theorem fraction_sum_simplest_terms (a b : ℕ) (h₁ : 0.4375 = a / b) :
  a = 7 ∧ b = 16 → a + b = 23 :=
by
  -- Place holder for the actual proof
  sorry

end fraction_sum_simplest_terms_l407_407412


namespace floor_sum_sqrt_25_l407_407006

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l407_407006


namespace rice_difference_l407_407355

theorem rice_difference:
  ∀ (total : ℕ) (kept_fraction : ℚ) (kept : ℕ) (given : ℕ),
  total = 50 →
  kept_fraction = (7 : ℚ) / 10 →
  kept = total * kept_fraction →
  given = total - kept →
  (kept - given = 20) :=
by
  intros total kept_fraction kept given
  intros h_total h_fraction h_kept h_given
  sorry

end rice_difference_l407_407355


namespace plane_perpendicular_l407_407335

-- Define types for lines and planes
axiom Line : Type
axiom Plane : Type

-- Define the relationships between lines and planes
axiom Parallel (l : Line) (p : Plane) : Prop
axiom Perpendicular (l : Line) (p : Plane) : Prop
axiom PlanePerpendicular (p1 p2 : Plane) : Prop

-- The setting conditions
variables (c : Line) (α β : Plane)

-- The given conditions
axiom c_perpendicular_β : Perpendicular c β
axiom c_parallel_α : Parallel c α

-- The proof goal (without the proof body)
theorem plane_perpendicular : PlanePerpendicular α β :=
by
  sorry

end plane_perpendicular_l407_407335


namespace sum_of_roots_eq_two_l407_407103

theorem sum_of_roots_eq_two (k : ℝ) :
    let a := -48
    let b := 96
    polynomial.aeval 0 (polynomial.C k + polynomial.X * polynomial.C b + (polynomial.X ^ 2) * polynomial.C a) = 0 →
    (polynomial.sum_roots (polynomial.C k + polynomial.X * polynomial.C b + (polynomial.X ^ 2) * polynomial.C a) = 2) := 
sorry

end sum_of_roots_eq_two_l407_407103


namespace find_y_value_l407_407523

theorem find_y_value (y : ℝ) (h1 : y > 0) (h2 : (sqrt ((7 - 2)^2 + (y - 4)^2) = 6)) : y = 4 + sqrt 11 :=
  sorry

end find_y_value_l407_407523


namespace line_circle_intersect_l407_407901

theorem line_circle_intersect (θ : ℝ) :
  (let x := 2 * Real.cos θ,
       y := 2 * Real.sin θ,
       line := 3 * x - 4 * y - 9) in
  (∃ (r : ℝ), x^2 + y^2 = 4 ∧ r = 2 ∧ |line| / (Real.sqrt (3^2 + (-4)^2)) < r ∧ ¬(line = 0)) :=
sorry

end line_circle_intersect_l407_407901


namespace lattice_points_count_l407_407788

theorem lattice_points_count :
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2^2 + p.3^2 = 25}.to_finset.card = 30 :=
sorry

end lattice_points_count_l407_407788


namespace free_ends_equal_1001_l407_407172

theorem free_ends_equal_1001 :
  ∃ k : ℕ, 1 + 4 * k = 1001 :=
begin
  use 250,
  norm_num,
end

end free_ends_equal_1001_l407_407172


namespace binom_30_3_l407_407613

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l407_407613


namespace earl_distance_to_top_l407_407078

def initial_floor : ℕ := 1
def floors_moved_up_1 : ℕ := 5
def floors_moved_down : ℕ := 2
def floors_moved_up_2 : ℕ := 7
def total_floors : ℕ := 20

theorem earl_distance_to_top :
  let final_floor := initial_floor + floors_moved_up_1 - floors_moved_down + floors_moved_up_2 in
  total_floors - final_floor = 9 :=
by sorry

end earl_distance_to_top_l407_407078


namespace solve_system_and_compute_l407_407114

-- Given system of equations
variables {x y : ℝ}
variables (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5)

-- Statement to prove
theorem solve_system_and_compute :
  (x - y = -1) ∧ (x + y = 3) ∧ ((1/3 * (x^2 - y^2)) * (x^2 - 2*x*y + y^2) = -1) :=
by
  sorry

end solve_system_and_compute_l407_407114


namespace smallest_N_99N_all_threes_l407_407100

theorem smallest_N_99N_all_threes : ∃ N : ℕ, (∀ d ∈ (digits 10 (99 * N)), d = 3) ∧ 
  (∀ M : ℕ, (∀ d ∈ (digits 10 (99 * M)), d = 3) → M ≥ N) :=
sorry

end smallest_N_99N_all_threes_l407_407100


namespace bookseller_sales_l407_407508

-- Definitions for the conditions
variables {b1 b2 b3 b4 : ℝ}

-- Condition 1: Total books sold in the first three months
def condition1 : Prop := b1 + b2 + b3 = 45

-- Condition 2: Books sold in April
def condition2 : Prop := b4 = 0.75 * (b1 + b2)

-- Condition 3: Average number of books sold in the first four months is 18
def condition3 : Prop := b1 + b2 + b3 + b4 = 72

-- The proof statement
theorem bookseller_sales
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) :
  b3 = 9 ∧ b1 + b2 = 36 :=
by
  sorry

end bookseller_sales_l407_407508


namespace sqrt_expression_simplification_l407_407554

theorem sqrt_expression_simplification : (sqrt ((sqrt 5)^4))^6 = 15625 := sorry

end sqrt_expression_simplification_l407_407554


namespace floor_sqrt_sum_l407_407047

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l407_407047


namespace hyperbola_distance_range_l407_407724

noncomputable def distance (A B : ℝ × ℝ) : ℝ := 
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

def hyperbola (x y : ℝ) : Prop := 
  (x^2 / 4) - y^2 = 1

def onLine (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop := 
  P.2 = l P.1

def rightFocus_eq (x : ℝ) : Prop := 
  x = real.sqrt 5

def absIntersections (A B : ℝ × ℝ) (a : ℝ) : Prop := 
  distance A B = a

theorem hyperbola_distance_range (a : ℝ) :
  (∀ l : ℝ → ℝ, ∃ A B : ℝ × ℝ, rightFocus_eq A.1 ∧ rightFocus_eq B.1 ∧ hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ onLine l A ∧ onLine l B ∧ absIntersections A B a) → a = 4 := 
sorry

end hyperbola_distance_range_l407_407724


namespace constant_subsequence_exists_l407_407326

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem constant_subsequence_exists (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (f : ℕ → ℕ) (c : ℕ), (∀ n m, n < m → f n < f m) ∧ (∀ n, sum_of_digits (⌊a * ↑(f n) + b⌋₊) = c) :=
sorry

end constant_subsequence_exists_l407_407326


namespace floor_sum_sqrt_25_l407_407008

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l407_407008


namespace find_perimeter_l407_407413

-- Define the parameters
def inradius (r : ℝ) := r = 5.0
def area (A : ℝ) := A = 105.0
def perimeter (p : ℝ) := A = r * (p / 2)

theorem find_perimeter (r A p : ℝ) (hr : inradius r) (hA : area A) :
   perimeter p → p = 42 :=
by
  sorry

end find_perimeter_l407_407413


namespace num_factors_of_60_l407_407241

theorem num_factors_of_60 : (∃ p₁ p₂ p₃ : ℕ, 60 = (2^2) * (3^1) * (5^1) ∧ p₁ = 2 ∧ p₂ = 1 ∧ p₃ = 1) → 
  nat.num_divisors 60 = 12 :=
begin
  intro h,
  cases h with p₁ hp₁,
  cases hp₁ with p₂ hp₂,
  cases hp₂ with p₃ hp₃,
  cases hp₃ with hp₋60 hpₚ₁,
  cases hpₚ₁ with hpₚ₂ hpₚ₃,
  have h_factors := by {
     rw [← hp₋60],
     exact by use (2: ℕ) , (1: ℕ), 1, rfl
     },
  sorry
end

end num_factors_of_60_l407_407241


namespace floor_sum_sqrt_1_to_25_l407_407060

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l407_407060


namespace evaluate_f_at_5pi_over_6_l407_407157

noncomputable def f (x : ℝ) : ℝ := Real.sin (π / 2 + x) * Real.sin (π + x)

theorem evaluate_f_at_5pi_over_6 : f (5 * π / 6) = sqrt 3 / 4 :=
by
  sorry

end evaluate_f_at_5pi_over_6_l407_407157


namespace individual_5th_is_16_l407_407986

/-- We have 50 individuals numbered from 01 to 50.
    We use a random number table to select 6 individuals.
    We start from the 9th and 10th columns of the 6th row
    and select two numbers at a time, moving from left to right.
    We need to prove that the number of the 5th individual selected is 16.
-/
theorem individual_5th_is_16 : 
  let row6 := [2748, 6198, 7164, 4148, 7086, 2888, 8519, 1620] in
  let selected_numbers := [41, 48, 28, 19, 16, 20] in
  selected_numbers[4] = 16 := by
  sorry

end individual_5th_is_16_l407_407986


namespace max_value_of_function_l407_407653

theorem max_value_of_function :
  ∀ x ∈ set.Icc (0 : ℝ) real.pi, 
  let y := 3 * real.sin x + 2 * real.sqrt (2 + 2 * real.cos (2 * x))
  in y ≤ 5 :=
by
  intro x hx
  let y := 3 * real.sin x + 2 * real.sqrt (2 + 2 * real.cos (2 * x))
  sorry

end max_value_of_function_l407_407653


namespace floor_sum_sqrt_25_l407_407009

theorem floor_sum_sqrt_25 : (∑ i in Finset.range 25, (Real.floor (Real.sqrt (i + 1)))) = 75 :=
sorry

end floor_sum_sqrt_25_l407_407009


namespace manolo_face_mask_time_l407_407110
variable (x : ℕ)
def time_to_make_mask_first_hour := x
def face_masks_made_first_hour := 60 / x
def face_masks_made_next_three_hours := 180 / 6
def total_face_masks_in_four_hours := face_masks_made_first_hour + face_masks_made_next_three_hours

theorem manolo_face_mask_time : 
  total_face_masks_in_four_hours x = 45 ↔ x = 4 := sorry

end manolo_face_mask_time_l407_407110


namespace binomial_30_3_l407_407574

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l407_407574


namespace initial_shells_l407_407822

theorem initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end initial_shells_l407_407822


namespace square_root_of_b_minus_a_l407_407714

theorem square_root_of_b_minus_a 
  (a b : ℝ)
  (h1 : 2 - 3 * a = a + 2 ∨ 2 - 3 * a = -(a + 2))
  (h2 : real.cbrt (5 * a + 3 * b - 1) = 3) 
  (h3 : a = 2) 
  (h4 : b = 6) :
  real.sqrt (b - a) = 2 :=
by
  sorry

end square_root_of_b_minus_a_l407_407714


namespace infinite_P_if_finite_Q_l407_407338

noncomputable theory

open Set

-- Let S be an infinite set of positive integers
variable (S : Set ℕ) (h_inf : Infinite S) (h_S_pos : ∀ x ∈ S, x > 0)

-- Let T be the set of all sums of two distinct elements in S
def T : Set ℕ := {x | ∃ a b ∈ S, a ≠ b ∧ x = a + b}

-- Let Q be the set of primes that are 1 mod 4 and divide some element of T
def Q : Set ℕ := {p | Prime p ∧ p % 4 = 1 ∧ ∃ x ∈ T, p ∣ x}

-- Let P be the set of primes that divide some element of S
def P : Set ℕ := {p | Prime p ∧ ∃ x ∈ S, p ∣ x}

-- The problem to prove
theorem infinite_P_if_finite_Q (hQ_finite : Finite Q) : Infinite P :=
sorry

end infinite_P_if_finite_Q_l407_407338


namespace function_positive_for_x_gt_neg1_l407_407409

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (3*x^2 + 6*x + 9)

theorem function_positive_for_x_gt_neg1 : ∀ (x : ℝ), x > -1 → f x > 0.5 :=
by
  sorry

end function_positive_for_x_gt_neg1_l407_407409


namespace difference_B_C_share_l407_407980

-- Total profit
def total_profit : ℝ := 20000

-- Proportions for A, B, C, D
def proportions : List ℝ := [2, 3, 5, 4]

-- Value of each part (total_profit / sum of proportions)
def part_value (total_profit : ℝ) (proportions : List ℝ) : ℝ :=
  total_profit / (proportions.foldr (+) 0)

-- Shares of B and C
def B_share (total_profit : ℝ) (proportion : ℝ) (part_value : ℝ) : ℝ := 
  proportion * part_value

def C_share (total_profit : ℝ) (proportion : ℝ) (part_value : ℝ) : ℝ := 
  proportion * part_value

-- The main theorem
theorem difference_B_C_share 
  (total_profit : ℝ) (proportions : List ℝ)
  (part_value : ℝ) (B_share : ℝ) (C_share : ℝ) :
  B_share = (3 * part_value) →
  C_share = (5 * part_value) →
  (C_share - B_share) = 2857.14 :=
by
  sorry

end difference_B_C_share_l407_407980


namespace sum_of_A_and_B_l407_407657

-- Define the conditions as Lean 4 statements
variables A B : ℕ
axiom condition1 : 3 / 7 = 6 / A
axiom condition2 : 6 / A = B / 21

-- Problem statement: Prove that A + B = 23 given these conditions
theorem sum_of_A_and_B :
  3 / 7 = 6 / A →
  6 / A = B / 21 →
  A + B = 23 :=
by
  intro h1 h2
  sorry

end sum_of_A_and_B_l407_407657


namespace ladybug_total_l407_407875

theorem ladybug_total (ladybugs_with_spots ladybugs_without_spots : ℕ) 
  (h1 : ladybugs_with_spots = 12170) 
  (h2 : ladybugs_without_spots = 54912) : 
  ladybugs_with_spots + ladybugs_without_spots = 67082 := 
by
  rw [h1, h2]
  sorry

end ladybug_total_l407_407875


namespace red_car_percentage_is_60_l407_407427

-- Definitions of the given conditions
def total_cars : ℕ := 900
def honda_cars : ℕ := 500
def red_percentage_honda : ℤ := 90
def red_percentage_non_honda : ℤ := 22.5

-- The final theorem to prove
theorem red_car_percentage_is_60 :
  let non_honda_cars := total_cars - honda_cars
  let red_honda_cars := (red_percentage_honda / 100) * honda_cars
  let red_non_honda_cars := (red_percentage_non_honda / 100) * non_honda_cars
  let total_red_cars := red_honda_cars + red_non_honda_cars
  (total_red_cars / total_cars) * 100 = 60 := 
by
  sorry

end red_car_percentage_is_60_l407_407427


namespace gray_area_l407_407441

-- Given conditions
def rect1_length : ℕ := 8
def rect1_width : ℕ := 10
def rect2_length : ℕ := 12
def rect2_width : ℕ := 9
def black_area : ℕ := 37

-- Define areas based on conditions
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width
def white_area : ℕ := area_rect1 - black_area

-- Theorem to prove the area of the gray part
theorem gray_area : area_rect2 - white_area = 65 :=
by
  sorry

end gray_area_l407_407441


namespace number_of_boys_in_school_l407_407756

theorem number_of_boys_in_school (B : ℝ) (h1 : 542.0 = B + 155) : B = 387 :=
by
  sorry

end number_of_boys_in_school_l407_407756


namespace breadth_of_rectangular_plot_is_18_l407_407410

/-- Problem statement:
The length of a rectangular plot is thrice its breadth. 
If the area of the rectangular plot is 972 sq m, 
this theorem proves that the breadth of the rectangular plot is 18 meters.
-/
theorem breadth_of_rectangular_plot_is_18 (b l : ℝ) (h_length : l = 3 * b) (h_area : l * b = 972) : b = 18 :=
by
  sorry

end breadth_of_rectangular_plot_is_18_l407_407410


namespace smallest_number_of_elements_l407_407323

theorem smallest_number_of_elements (n : ℕ) (hn : n ≥ 2) (S : Finset ℝ) 
  (hS : S.card = n) (h_distinct : ∀ x ∈ S, ∀ y ∈ S, x ≠ y → ∃ z, z ∈ S \ x ∧ ⇑z = (x + y) / 2) : 
  ∃ A_S : Finset ℝ, A_S.card = 2 * n - 3 :=
by
  sorry

end smallest_number_of_elements_l407_407323


namespace f_minus_ten_l407_407153

noncomputable def f : ℝ → ℝ := sorry

theorem f_minus_ten :
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) →
  (f 1 = 2) →
  f (-10) = 90 :=
by
  intros h1 h2
  sorry

end f_minus_ten_l407_407153


namespace binomial_30_3_l407_407582

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l407_407582


namespace problem_solution_l407_407741

variable (a b : ℝ)
variable (lg : ℝ → ℝ)
variable (two three ten eighteen : ℝ)

-- Assumptions based on conditions
axiom lg_two : lg two = a
axiom lg_three : lg three = b
axiom lg_ten : lg ten = 1
axiom eq_two_times_three_sq_div_ten_sq : eighteen / (ten * ten) = two * (three * three)

noncomputable def lg_eighteen_div_hundred : Prop := lg (eighteen / (ten * ten)) = a + 2 * b - 2

theorem problem_solution : lg_eighteen_div_hundred :=
by
  split
  sorry

end problem_solution_l407_407741


namespace sum_of_floors_of_square_roots_l407_407017

theorem sum_of_floors_of_square_roots :
  ∑ n in Finset.range 26, ⌊Real.sqrt n⌋ = 75 :=
by
  sorry

end sum_of_floors_of_square_roots_l407_407017


namespace sum_of_slope_and_y_intercept_of_bisecting_line_l407_407924

-- Define the vertices of the triangle
def P := (0, 10)
def Q := (3, 0)
def R := (9, 0)

-- Define the midpoint S of line PR
def S := ((0 + 9) / 2, (10 + 0) / 2)

-- Define the slope of the line passing through Q and S
def slope := (5 - 0) / ((9 / 2) - 3)

-- Define the equation of the line passing through Q using the point-slope form
def line_y (x : ℝ) := slope * (x - 3)

-- Find the y-intercept of the line
def y_intercept := line_y 0

-- Calculate the sum of the slope and y-intercept
def sum_slope_y_intercept := slope + y_intercept

theorem sum_of_slope_and_y_intercept_of_bisecting_line :
  sum_slope_y_intercept = -(20 / 3) :=
by
  sorry

end sum_of_slope_and_y_intercept_of_bisecting_line_l407_407924


namespace num_lattice_points_distance_five_l407_407812

open_locale classical 

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points_distance_five : 
  {n : ℕ | ∃ (x y z : ℤ), is_lattice_point x y z} = 78 :=
by sorry

end num_lattice_points_distance_five_l407_407812


namespace quadratic_roots_l407_407139

-- Definitions based on the conditions provided.
def condition1 (x y : ℝ) : Prop := x^2 - 6 * x + 9 = -(abs (y - 1))

-- The main theorem we want to prove.
theorem quadratic_roots (x y : ℝ) (h : condition1 x y) : (a : ℝ) → (a - 3) * (a - 1) = a^2 - 4 * a + 3 :=
  by sorry

end quadratic_roots_l407_407139


namespace can_transport_machines_l407_407434

theorem can_transport_machines :
  ∀ (machine1_mass machine2_mass machine3_mass truck_capacity : ℕ),
  machine1_mass = 800 →
  machine2_mass = 500 →
  machine3_mass = 600 →
  truck_capacity = 2000 →
  (machine1_mass + machine2_mass + machine3_mass ≤ truck_capacity) :=
by
  intros machine1_mass machine2_mass machine3_mass truck_capacity
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  have : 800 + 500 + 600 = 1900 := by norm_num
  rw this
  exact le_of_lt (by norm_num : 1900 < 2000)

end can_transport_machines_l407_407434


namespace extremum_at_zero_only_for_B_l407_407991

-- Define the functions
def f_A (x : ℝ) : ℝ := -x^3
def f_B (x : ℝ) : ℝ := -cos x
def f_C (x : ℝ) : ℝ := sin x - x
def f_D (x : ℝ) : ℝ := 1 / x

-- Define the extremum condition at x = 0 for a given function
def has_extremum_at_zero (f : ℝ → ℝ) : Prop :=
  deriv f 0 = 0 ∧
  (∀ ε > 0, ∀ δ > 0, (∀ x, 0 < x ∧ x < δ → abs (f x - f 0) < ε)
  ∨ (∀ x, -δ < x ∧ x < 0 → abs (f x - f 0) < ε))

-- The final theorem statement:
theorem extremum_at_zero_only_for_B :
  (has_extremum_at_zero f_B) ∧ 
  ¬(has_extremum_at_zero f_A) ∧ 
  ¬(has_extremum_at_zero f_C) ∧ 
  ¬(has_extremum_at_zero f_D) :=
by
  sorry

end extremum_at_zero_only_for_B_l407_407991


namespace no_value_of_c_for_one_vertical_asymptote_l407_407112

theorem no_value_of_c_for_one_vertical_asymptote :
  ¬ ∃ (c : ℝ), (∀ (x : ℝ), x ≠ 5 → x ≠ -2 → x ≠ (-(3/2) + (sqrt(25-4*c))/2) ∧ x ≠ (-(3/2) - (sqrt(25-4*c))/2))) :=
by
  sorry

end no_value_of_c_for_one_vertical_asymptote_l407_407112


namespace sqrt_19_between_4_and_5_l407_407428

theorem sqrt_19_between_4_and_5 : 4 < Real.sqrt 19 ∧ Real.sqrt 19 < 5 := by
  -- Given conditions
  have h1 : 16 < 19 := by norm_num
  have h2 : 19 < 25 := by norm_num
  -- Known values of square roots of perfect squares
  have h3 : Real.sqrt 16 = 4 := by norm_num
  have h4 : Real.sqrt 25 = 5 := by norm_num
  -- Proof to show 4 < Real.sqrt 19 < 5 is skipped
  sorry

end sqrt_19_between_4_and_5_l407_407428


namespace find_range_of_ratio_l407_407705

variables (P Q M : Type) [metric_space P] [metric_space Q] [metric_space M]

theorem find_range_of_ratio (x₀ y₀ : ℝ) (k : ℝ) 
  (h1 : P ∈ {p : ℝ × ℝ | p.1 + 2 * p.2 = 1}) 
  (h2 : Q ∈ {q : ℝ × ℝ | q.1 + 2 * q.2 = -3})
  (hM : M = (x₀, y₀)) 
  (hyx : y₀ = k * x₀) 
  (h_ineq : y₀ > x₀ + 2) : 
  - (1 / 2) < k ∧ k < - (1 / 5) :=
begin
  sorry,
end

end find_range_of_ratio_l407_407705


namespace average_age_of_adults_is_26_2_l407_407753

noncomputable def average_age_of_adults
  (num_members : ℕ)
  (avg_age_members : ℝ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_adults : ℕ)
  (avg_age_girls : ℝ)
  (avg_age_boys : ℝ)
  : ℝ :=
let total_age_members := num_members * avg_age_members in
let total_age_girls := num_girls * avg_age_girls in
let total_age_boys := num_boys * avg_age_boys in
(total_age_members - total_age_girls - total_age_boys) / num_adults

theorem average_age_of_adults_is_26_2
  (h1 : num_members = 50)
  (h2 : avg_age_members = 20)
  (h3 : num_girls = 22)
  (h4 : num_boys = 18)
  (h5 : num_adults = 10)
  (h6 : avg_age_girls = 18)
  (h7 : avg_age_boys = 19)
  : average_age_of_adults 50 20 22 18 10 18 19 = 26.2 :=
by
  let total_age_members := 50 * 20
  let total_age_girls := 22 * 18
  let total_age_boys := 18 * 19
  let total_age_adults := total_age_members - total_age_girls - total_age_boys
  have h_total_age_members : total_age_members = 1000 := rfl
  have h_total_age_girls : total_age_girls = 396 := rfl
  have h_total_age_boys : total_age_boys = 342 := rfl
  have h_total_age_adults : total_age_adults = 1000 - 396 - 342 := rfl
  have h_age_adults := (1000 - 396 - 342) / 10
  have h_average_age_adults : h_age_adults = 26.2 := rfl
  sorry

end average_age_of_adults_is_26_2_l407_407753


namespace fifth_derivative_l407_407494

variable (x : ℝ)
def y : ℝ := (4 * x^3 + 5) * exp (2 * x + 1)

theorem fifth_derivative :
  deriv^[5] (λ x : ℝ, (4 * x^3 + 5) * exp (2 * x + 1)) x = 32 * (4 * x^3 + 30 * x^2 + 60 * x + 35) * exp (2 * x + 1) :=
sorry

end fifth_derivative_l407_407494


namespace plane_determined_by_three_lines_l407_407295

noncomputable def determine_plane (L1 L2 L3 : Set Point) : Prop :=
  (∀ (p1 ∈ L1) (p2 ∈ L2), p1 ≠ p2) ∧  -- L1 and L2 intersect
  (∀ (p1 ∈ L2) (p2 ∈ L3), p1 ≠ p2) ∧  -- L2 and L3 intersect
  (∀ (p1 ∈ L1) (p2 ∈ L3), p1 ≠ p2) ∧  -- L1 and L3 intersect
  ¬(∃ p, p ∈ L1 ∧ p ∈ L2 ∧ p ∈ L3)    -- No common intersection point

theorem plane_determined_by_three_lines (L1 L2 L3 : Set Point) :
  determine_plane L1 L2 L3 → ∃ plane : Set Point, Subset L1 plane ∧ Subset L2 plane ∧ Subset L3 plane :=
by
  sorry

end plane_determined_by_three_lines_l407_407295


namespace quadratic_real_roots_l407_407422

theorem quadratic_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ (x^2 - m * x + (m - 1) = 0) ∧ (y^2 - m * y + (m - 1) = 0) 
  ∨ ∃ z : ℝ, (z^2 - m * z + (m - 1) = 0) := 
sorry

end quadratic_real_roots_l407_407422


namespace lattice_points_count_is_30_l407_407793

-- Define the function that counts the number of lattice points (x, y, z) such that x^2 + y^2 + z^2 = 25.
def count_lattice_points := {p : ℤ × ℤ × ℤ // p.1 ^ 2 + p.2.1 ^ 2 + p.2.2 ^ 2 = 25}.to_finset.card

-- The theorem to prove that the count is 30.
theorem lattice_points_count_is_30 : count_lattice_points = 30 :=
by
  sorry

end lattice_points_count_is_30_l407_407793


namespace sum_integers_100_to_110_l407_407471

theorem sum_integers_100_to_110 : (∑ i in finset.Icc 100 110, i) = 1155 := by
  sorry

end sum_integers_100_to_110_l407_407471


namespace smallest_N_99N_all_threes_l407_407099

theorem smallest_N_99N_all_threes : ∃ N : ℕ, (∀ d ∈ (digits 10 (99 * N)), d = 3) ∧ 
  (∀ M : ℕ, (∀ d ∈ (digits 10 (99 * M)), d = 3) → M ≥ N) :=
sorry

end smallest_N_99N_all_threes_l407_407099


namespace num_perfect_cube_factors_4410_l407_407903

theorem num_perfect_cube_factors_4410 : 
  let a := 1
  let b := 2
  let c := 1
  let d := 2
  let factors : List (ℕ × ℕ) := [(2, a), (3, b), (5, c), (7, d)]
  let is_perfect_cube (n : ℕ × ℕ) : Prop := n.snd % 3 = 0
  (List.countp is_perfect_cube factors) = 1 := 
sorry

end num_perfect_cube_factors_4410_l407_407903


namespace number_of_factors_of_60_l407_407208

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l407_407208


namespace zero_of_f_in_interval_l407_407890

noncomputable def f (x m : ℝ) : ℝ :=
  -2 * Real.tan x + m

theorem zero_of_f_in_interval (x m : ℝ) :
  (x ∈ Icc (-π/4) (π/3)) → (∃ x, f x m = 0) → m ∈ Icc (-2) (2 * Real.sqrt 3) :=
by
  intros hx hzero
  sorry -- complete the proof here

end zero_of_f_in_interval_l407_407890


namespace half_area_cutting_line_l407_407927

def P : ℝ × ℝ := (0, 10)
def Q : ℝ × ℝ := (3, 0)
def R : ℝ × ℝ := (9, 0)

def M : ℝ × ℝ := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)
def y_intercept (m x1 y1 : ℝ) : ℝ := y1 - m * x1

def line_eq (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem half_area_cutting_line :
  let m := slope Q M,
      b := y_intercept m Q.1 Q.2 in
  m + b = -20 / 3 :=
by
  sorry

end half_area_cutting_line_l407_407927


namespace jill_and_bob_payment_l407_407315

-- Definitions of the conditions
def price_of_first_house (X : ℝ) := X
def price_of_second_house (Y X : ℝ) := 2 * X

theorem jill_and_bob_payment :
  ∃ X, ∃ Y, Y = 2 * X ∧ X + Y = 600000 ∧ X = 200000 :=
by
  sorry

end jill_and_bob_payment_l407_407315


namespace num_factors_of_60_l407_407198

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end num_factors_of_60_l407_407198


namespace remainder_of_171_divided_by_21_l407_407859

theorem remainder_of_171_divided_by_21 : 
  ∃ r, 171 = (21 * 8) + r ∧ r = 3 := 
by
  sorry

end remainder_of_171_divided_by_21_l407_407859


namespace num_factors_of_60_l407_407254

theorem num_factors_of_60 : 
  ∃ n, (prime_factors 60 = {2^2, 3, 5}) ∧ (number_of_factors 60 = n) ∧ (n = 12) :=
sorry

end num_factors_of_60_l407_407254


namespace unique_real_solution_equation_l407_407111

theorem unique_real_solution_equation (m : ℝ) :
  (∀ (x : ℝ), (x + 3) * (x + 2) = m + 3 * x ↔ x^2 + 2 * x + (6 - m) = 0) →
  (∃! x, (x + 3) * (x + 2) = m + 3 * x) ↔ m = 5 :=
begin
  sorry
end

end unique_real_solution_equation_l407_407111


namespace triangle_side_b_value_l407_407836

theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) (h1 : a = Real.sqrt 3) (h2 : A = 60) (h3 : C = 75) : b = Real.sqrt 2 :=
sorry

end triangle_side_b_value_l407_407836


namespace part_a_not_isosceles_part_b_isosceles_l407_407960

-- Definitions for part (a)
variables {A B C M N O : Type}
variables [Triangle A B C] [Point M] [Point N] [Point O]
variables (hM : PointOn M AB) (hN : PointOn N BC)
variables (h_intersect : IntersectionPoint (Line A N) (Line C M) O)
variables (h_eq : SegmentsEqual (Segment A O) (Segment C O))
variables (h_eq_CN_AM : SegmentsEqual (Segment A M) (Segment C N))

-- Definitions for part (b)
variables (h_eq_BM_BN : SegmentsEqual (Segment B M) (Segment B N))

-- Part (a): Triangle is not necessarily isosceles
theorem part_a_not_isosceles : ¬IsIsoscelesTriangle A B C := by
  sorry

-- Part (b): Triangle is necessarily isosceles
theorem part_b_isosceles : IsIsoscelesTriangle A B C := by
  sorry

end part_a_not_isosceles_part_b_isosceles_l407_407960


namespace llesis_more_rice_l407_407357

theorem llesis_more_rice :
  let total_rice := 50
  let llesis_fraction := 7 / 10
  let llesis_rice := total_rice * llesis_fraction
  let everest_rice := total_rice - llesis_rice
  llesis_rice - everest_rice = 20 := by
    sorry

end llesis_more_rice_l407_407357


namespace hyperbola_properties_l407_407405

theorem hyperbola_properties :
  ∃ (C : Set (ℝ × ℝ)), -- C represents the set of points on the hyperbola
  (asymptote_eq : ∀ (x y : ℝ), (x, y) ∈ C → x - 2 * y = 0) ∧
  (C_passing : (2 * sqrt 2, 1) ∈ C) ∧
  (hyperbola_eq : ∀ (x y : ℝ), (x, y) ∈ C → x^2 / 4 - y^2 = 1) ∧
  (∃ (A1 A2 : (ℝ × ℝ)),
     A1 = (-2, 0) ∧ A2 = (2, 0) ∧
     (∀ (P : (ℝ × ℝ)), P ∈ C →
      let M := (1, 3 * ((P.2 - 0) / (P.1 + 2)))) 
      (N := (1, -(P.2 / (P.1 - 2)))) 
      (MN : ℝ) := |M.2 - N.2|
      MN = sqrt 3)) :=
begin
  sorry
end

end hyperbola_properties_l407_407405


namespace hyperbola_eccentricity_l407_407725

-- Given conditions
variables {a b c : ℝ}
variables (h_a_pos : a > 0) (h_b_pos : b > 0)
def hyperbola_eq := x/a^2 - y/b^2 = 1
def focus_distance := c = 2 * a / sqrt 3
def slope_asymptotes := b/a = 1/sqrt 3

-- Prove the eccentricity
theorem hyperbola_eccentricity (h1 : hyperbola_eq x y) (h2 : slope_asymptotes) (h3 : focus_distance):
∃ e : ℝ, e = sqrt 21 / 3 :=
begin
  sorry
end

end hyperbola_eccentricity_l407_407725


namespace jean_average_mark_l407_407310

/-
  Jean writes five tests and achieves the following marks: 80, 70, 60, 90, and 80.
  Prove that her average mark on these five tests is 76.
-/
theorem jean_average_mark : 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  average_mark = 76 :=
by 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  sorry

end jean_average_mark_l407_407310


namespace num_lattice_points_l407_407781

def is_lattice_point (p : ℤ × ℤ × ℤ) : Prop := 
  let (x, y, z) := p 
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points : 
  {p : ℤ × ℤ × ℤ | is_lattice_point p}.to_finset.card = 18 :=
sorry

end num_lattice_points_l407_407781


namespace f_strictly_decreasing_on_negative_interval_l407_407073

noncomputable def f (x : ℝ) : ℝ := log (x^2 - 1) / log 2

theorem f_strictly_decreasing_on_negative_interval :
  ∀ x y : ℝ, x < -1 → y < -1 → x < y → f y < f x :=
by
  sorry

end f_strictly_decreasing_on_negative_interval_l407_407073


namespace factors_of_60_l407_407229

/-- The prime factorization of 60 is 2^2 * 3 * 5 -/
def prime_factors_60 : ℕ → ℕ
| 2 := 2
| 3 := 1
| 5 := 1
| _ := 0

/-- The number of positive factors of 60 is 12 -/
theorem factors_of_60 : (1 + prime_factors_60 2) * (1 + prime_factors_60 3) * (1 + prime_factors_60 5) = 12 := by
  sorry

end factors_of_60_l407_407229


namespace balls_per_pack_l407_407855

theorem balls_per_pack (total_packs total_cost cost_per_ball total_balls balls_per_pack : ℕ)
  (h1 : total_packs = 4)
  (h2 : total_cost = 24)
  (h3 : cost_per_ball = 2)
  (h4 : total_balls = total_cost / cost_per_ball)
  (h5 : total_balls = 12)
  (h6 : balls_per_pack = total_balls / total_packs) :
  balls_per_pack = 3 := by 
  sorry

end balls_per_pack_l407_407855


namespace sum_of_floors_of_square_roots_l407_407030

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l407_407030


namespace arrange_books_l407_407265

theorem arrange_books : 
  let total_books := 7 
  let identical_math_books := 3 
  let identical_science_books := 2 
  (nat.factorial total_books) / (nat.factorial identical_math_books * nat.factorial identical_science_books) = 420 :=
by
  sorry

end arrange_books_l407_407265


namespace inequality_holds_for_all_real_numbers_l407_407074

theorem inequality_holds_for_all_real_numbers (x : ℝ) : 3 * x - 5 ≤ 12 - 2 * x + x^2 :=
by sorry

end inequality_holds_for_all_real_numbers_l407_407074


namespace different_selections_count_l407_407493

-- Definition of the attractions
inductive Attraction
| A | B | C

open Attraction

-- Definition of the selections (pairs of attractions)
def selections : List (Attraction × Attraction) :=
  [(A, B), (A, C), (B, C)]

-- Statement of the problem
theorem different_selections_count :
  ∃ (count : ℕ), count = 6 ∧
  ∀ (s1 s2 : Attraction × Attraction), s1 ∈ selections → s2 ∈ selections → s1 ≠ s2 → true :=
begin
  use 6,
  split,
  { 
    -- Proof of count = 6
    sorry
  },
  {
    -- Proof that s1 ≠ s2 implies s1 and s2 exist in selections and are different
    intros s1 s2 hs1 hs2 hneq,
    exact trivial,
  }
end

end different_selections_count_l407_407493


namespace triangle_height_l407_407880

theorem triangle_height (A : ℝ) (b : ℝ) (h : ℝ) 
  (hA : A = 615) 
  (hb : b = 123)
  (h_area : A = 0.5 * b * h) : 
  h = 10 :=
by 
  -- Placeholder for the proof
  sorry

end triangle_height_l407_407880


namespace deepak_present_age_l407_407417

theorem deepak_present_age (R D : ℕ) (h1 : R =  4 * D / 3) (h2 : R + 10 = 26) : D = 12 :=
by
  sorry

end deepak_present_age_l407_407417


namespace sum_of_odd_terms_is_2151_l407_407534

-- Define the sequence x as a sequence of 2010 terms where each term is 1 more than the previous term
def seq (n : ℕ) (x : ℕ) : ℕ := x + n

-- Define the sum of the first 2010 terms is 5307
def sum_all (x : ℕ) : Prop := (Finset.sum (Finset.range 2010) (seq x)) = 5307

-- Define the sum of the odd-numbered terms
def sum_odd (x : ℕ) : ℕ := Finset.sum (Finset.filter (λ n, odd n) (Finset.range 2010)) (seq x)

-- Define the theorem stating that the sum of the odd-numbered terms is 2151
theorem sum_of_odd_terms_is_2151 (x : ℕ) (h : sum_all x) : sum_odd x = 2151 := sorry

end sum_of_odd_terms_is_2151_l407_407534


namespace num_factors_60_l407_407209

theorem num_factors_60 : 
  (let n := 60 in 
   let prime_factors := [(2, 2), (3, 1), (5, 1)] in 
   (prime_factors.foldl (λ acc p, acc * (p.snd + 1)) 1) = 12) := 
begin
  sorry
end

end num_factors_60_l407_407209


namespace parallel_lines_slope_l407_407176

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 2 = 0 → ax * x - 8 * y - 3 = 0 → a = -6) :=
by
  sorry

end parallel_lines_slope_l407_407176


namespace find_angle_C_find_area_of_triangle_l407_407305

variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides of the triangle

-- Proof 1: Prove \(C = \frac{\pi}{3}\) given \(a \cos B \cos C + b \cos A \cos C = \frac{c}{2}\).

theorem find_angle_C 
  (h : a * Real.cos B * Real.cos C + b * Real.cos A * Real.cos C = c / 2) : C = π / 3 :=
sorry

-- Proof 2: Prove the area of triangle \(ABC = \frac{3\sqrt{3}}{2}\) given \(c = \sqrt{7}\), \(a + b = 5\), and \(C = \frac{\pi}{3}\).

theorem find_area_of_triangle 
  (h1 : c = Real.sqrt 7) (h2 : a + b = 5) (h3 : C = π / 3) : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
sorry

end find_angle_C_find_area_of_triangle_l407_407305


namespace find_m_l407_407734

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + (m + 2) * y - 1 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x + m * y = 0

-- Define the condition of perpendicularity
def perpendicular (m : ℝ) : Prop :=
  let slope1 := if m ≠ 0 then -(m/(m + 2)) else 0,
      slope2 := if m ≠ m then -((m - 1)/m) else 0 in
  slope1 * slope2 = -1

-- State the problem in Lean
theorem find_m (m : ℝ) :
  (line1 m 0 0 ∧ line2 m 0 0) → (perpendicular m) → (m = 0 ∨ m = -1/2) :=
by
  intro h_lines h_perp
  sorry

end find_m_l407_407734


namespace tetrahedron_inequality_l407_407292

theorem tetrahedron_inequality {A1 A2 A3 A4 P : Type} 
  (l : ℕ → ℝ) (d : ℕ → ℝ) (h : ℕ → ℝ) (k : ℝ) (hk : k ≥ 3/4) : 
  (∑ i in [1, 2, 3, 4], (l i / (2 * h i + d i)) ^ k) ≥ (4 / 3 ^ k) :=
sorry

end tetrahedron_inequality_l407_407292


namespace smallest_perimeter_l407_407939

noncomputable def smallest_possible_perimeter : ℕ :=
  let n := 3
  n + (n + 1) + (n + 2)

theorem smallest_perimeter (n : ℕ) (h : n > 2) (ineq1 : n + (n + 1) > (n + 2)) 
  (ineq2 : n + (n + 2) > (n + 1)) (ineq3 : (n + 1) + (n + 2) > n) : 
  smallest_possible_perimeter = 12 :=
by
  sorry

end smallest_perimeter_l407_407939


namespace flippy_numbers_div_by_40_l407_407504

-- Definition of a 6-digit flippy number alternating between two distinct digits.
def isFlippyNumber (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ n = 100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b

-- Definition of divisibility by 40.
def isDivisibleBy40 (n : ℕ) : Prop :=
  n % 40 = 0

-- Main theorem stating there are exactly 2 valid 6-digit flippy numbers divisible by 40.
theorem flippy_numbers_div_by_40 : 
  { n : ℕ | n >= 100000 ∧ n < 1000000 ∧ isFlippyNumber n ∧ isDivisibleBy40 n }.to_finset.card = 2 :=
by
  sorry

end flippy_numbers_div_by_40_l407_407504


namespace num_factors_60_l407_407188

theorem num_factors_60 : (nat.factors 60).length + 1 = 12 := by
  -- Here goes the proof
  sorry

end num_factors_60_l407_407188


namespace snail_noodles_pricing_and_optimization_l407_407872

variable (x y a : ℝ)

-- Conditions
def conditionA := (x = 100)
def conditionB := (y = 1.5 * x)
def conditionC := (800 / x) - (900 / y) = 2
def conditionD := a ≤ 15
def conditionE := ∃ a : ℝ, a + (30 - a) = 30

-- Prove
theorem snail_noodles_pricing_and_optimization :
  conditionA ∧ conditionB ∧ conditionC ∧ conditionD ∧ conditionE →
  x = 100 ∧ y = 150 ∧ -45 * 15 + 4050 = 3375 :=
by {
  assume h,
  sorry
}

end snail_noodles_pricing_and_optimization_l407_407872


namespace midpoint_is_grid_point_l407_407287

theorem midpoint_is_grid_point
  (P : Fin 5 → (ℤ × ℤ)) :
  ∃ (i j : Fin 5), i ≠ j ∧ (P i).fst % 2 = (P j).fst % 2 ∧ (P i).snd % 2 = (P j).snd % 2 :=
by
  sorry

end midpoint_is_grid_point_l407_407287


namespace vector_subtraction_example_l407_407702

theorem vector_subtraction_example (OA OB AB : ℝ × ℝ) 
  (h1 : OA = (2, 8)) 
  (h2 : OB = (-7, 2)) :
  AB = (fst OB - fst OA, snd OB - snd OA) → AB = (-9, -6) :=
by
  intro h
  simp [h1, h2] at h
  assumption

end vector_subtraction_example_l407_407702


namespace gcd_of_three_numbers_l407_407091

theorem gcd_of_three_numbers : 
  let a := 4560
  let b := 6080
  let c := 16560
  gcd (gcd a b) c = 80 := 
by {
  -- placeholder for the proof
  sorry
}

end gcd_of_three_numbers_l407_407091


namespace find_p_value_l407_407763

theorem find_p_value (p : ℝ) (hp : p > 0) :
  (∃ (a : ℝ) b, (a^2 = 3) ∧ (b^2 = 1) ∧ 
  (asymptote x y a b (-3/2)) ∧ 
  parabolas_asymptote y p x (-3/2)) → p = 3 :=
sorry

def parabolas_asymptote (y : ℝ) (p : ℝ) (x : ℝ) (asymptote : ℝ) : Prop :=
(x = -p/2)

def asymptote (x : ℝ) (y : ℝ) (a : ℝ) (b : ℝ) : Prop :=
(x = -(a^2)/2)

end find_p_value_l407_407763


namespace sum_y_coordinates_intersection_l407_407425

-- Definitions of the functions f and g
def f (x : ℝ) : ℝ := (x + 1) / x
def g (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x) + 1

-- The interval and the sum of y-coordinates
def interval := Set.Icc (-5 : ℝ) 5
axiom intersection_points : Set ℝ := {x | x ∈ interval ∧ f x = g x}
axiom intersection_y_sum : intersection_points.sum (λ x, f x) = 10

-- The theorem statement
theorem sum_y_coordinates_intersection : intersection_y_sum :=
sorry

end sum_y_coordinates_intersection_l407_407425


namespace no_more_than_five_planes_land_in_one_city_l407_407282

-- Definitions for the problem
def city := Type
def distance (A B : city) : ℝ := sorry
def distinct_distances (cities : set city) : Prop :=
  ∀ (A B C D : city), A ≠ B → C ≠ D → distance A B ≠ distance C D
def nearest_neighbor (A : city) (cities : set city) : city :=
  sorry -- Definition to find the nearest neighboring city of A in the set of cities

-- Main theorem statement
theorem no_more_than_five_planes_land_in_one_city
  (cities: set city)
  (h_distinct: distinct_distances cities)
  (h_planes: ∀ A ∈ cities, ∃ B ∈ cities, nearest_neighbor A cities = B):
  ∀ city P ∈ cities, ¬ (∃ A1 A2 A3 A4 A5 A6 ∈ cities,
    nearest_neighbor A1 cities = P ∧
    nearest_neighbor A2 cities = P ∧
    nearest_neighbor A3 cities = P ∧
    nearest_neighbor A4 cities = P ∧
    nearest_neighbor A5 cities = P ∧
    nearest_neighbor A6 cities = P) :=
by
  sorry

end no_more_than_five_planes_land_in_one_city_l407_407282


namespace surface_area_of_sphere_l407_407129

noncomputable def sphere_surface_area (A B C D : Point) (BD : ℝ) (angle_BAD angle_BCD : RealAngle) 
  (plane_ABD plane_BCD : Plane) (hBD : BD = 2 * Real.sqrt 3) 
  (hangle_BAD : angle_BAD = 60) 
  (hangle_BCD : angle_BCD = 90) 
  (hplane_perp : is_perpendicular plane_ABD plane_BCD) : ℝ :=
  let R := 2 in
  4 * Real.pi * R^2

theorem surface_area_of_sphere 
  (A B C D : Point) (BD : ℝ) (angle_BAD angle_BCD : RealAngle) 
  (plane_ABD plane_BCD : Plane) 
  (hBD : BD = 2 * Real.sqrt 3) 
  (hangle_BAD : angle_BAD = 60) 
  (hangle_BCD : angle_BCD = 90) 
  (hplane_perp : is_perpendicular plane_ABD plane_BCD) : 
  sphere_surface_area A B C D BD angle_BAD angle_BCD plane_ABD plane_BCD hBD hangle_BAD hangle_BCD hplane_perp = 16 * Real.pi :=
sorry

end surface_area_of_sphere_l407_407129


namespace total_boys_in_camp_l407_407489

-- Define the given conditions as variables
variable (T : ℕ) -- Total number of boys in the camp
variable (h1 : 0.70 * (0.20 * T) = 35)

-- Define the theorem to prove the total number of boys is 250 under the given conditions
theorem total_boys_in_camp : T = 250 :=
by
  -- Using the conditions to structure the proof
  sorry

end total_boys_in_camp_l407_407489


namespace proof_problem_l407_407877

-- Define the problem conditions
variables (x y : Real)

-- Condition 1: 0.25x = 0.455y
def condition1 : Prop := 0.25 * x = 0.455 * y

-- Condition 2: x - y = 155
def condition2 : Prop := x - y = 155

-- Define the statement to be proved: x ≈ 344 and y ≈ 189
def target1 : Prop := x ≈ 344
def target2 : Prop := y ≈ 189

theorem proof_problem (h1 : condition1) (h2 : condition2) : target1 ∧ target2 :=
sorry

end proof_problem_l407_407877


namespace Seryozha_cannot_prevent_Vanya_l407_407366

-- Define the dimensions of the board
def board_height := 4
def board_width := 6

-- Define the initial state and movements
structure BoardState :=
  (black_pieces : list (ℕ × ℕ))  -- list of positions of black pieces (row, column)
  (white_pieces : list (ℕ × ℕ))  -- list of positions of white pieces (row, column)

-- Define the capture condition
def is_captured (black_pos : (ℕ × ℕ)) (white_pos1 white_pos2 : (ℕ × ℕ)) : Prop :=
  (black_pos.1 = white_pos1.1 ∧ black_pos.1 = white_pos2.1 ∧ white_pos1.2 < black_pos.2 ∧ black_pos.2 < white_pos2.2)
  ∨ (black_pos.1 ≠ white_pos1.1 ∧ abs (black_pos.1 - white_pos1.1) = abs (black_pos.2 - white_pos1.2) 
  ∧ abs (black_pos.1 - white_pos2.1) = abs (black_pos.2 - white_pos2.2))

-- Define the movement condition
def move (piece : ℕ × ℕ) : ℕ × ℕ :=
  (piece.1 + 1, piece.2)

-- Define the goal state for Vanya
def goal_state (black_pieces : list (ℕ × ℕ)) : Prop :=
  ∀ piece ∈ black_pieces, piece.1 = board_height

-- Main theorem asserting that Sergey cannot prevent Vanya
theorem Seryozha_cannot_prevent_Vanya (initial_state : BoardState) :
  ∃ final_state : BoardState, goal_state final_state.black_pieces :=
sorry

end Seryozha_cannot_prevent_Vanya_l407_407366


namespace real_solution_exists_l407_407906

noncomputable def realSolution (x : ℝ) : Prop :=
  9 / (3^x - 1) + 1 = 3^x

theorem real_solution_exists : ∃ x : ℝ, realSolution x ∧ x = log 3 4 := 
by
  have h : realSolution (log 3 4) := sorry
  use log 3 4
  split
  { exact h }
  { refl }

end real_solution_exists_l407_407906


namespace binom_30_3_eq_4060_l407_407608

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l407_407608


namespace not_equiv_5div30_times_10neg7_l407_407947

theorem not_equiv_5div30_times_10neg7 : 
  let a := 0.000000165
  let b := 1.65 * 10^(-7)
  let option_d := (5 / 30) * 10^(-7)
  a = b → option_d ≠ b :=
by
  let a := 0.000000165
  let b := 1.65 * 10^(-7)
  let option_d := (5 / 30) * 10^(-7)
  intro h
  sorry

end not_equiv_5div30_times_10neg7_l407_407947


namespace Mahesh_marked_percentage_l407_407353

noncomputable def marked_percentage_above_cp 
  (CP SP : ℝ) (discount_percentage : ℝ) : ℝ :=
let MP := SP / (1 - discount_percentage / 100) in
((MP - CP) / CP) * 100

theorem Mahesh_marked_percentage
  (CP SP : ℝ) (discount_percentage : ℝ) 
  (h_CP : CP = 540) 
  (h_SP : SP = 459) 
  (h_discount : discount_percentage = 26.08695652173913) :
  marked_percentage_above_cp CP SP discount_percentage = 15 :=
by
  simp [marked_percentage_above_cp, h_CP, h_SP, h_discount]
  norm_num -- norm_num helps to simplify numerical calculations

end Mahesh_marked_percentage_l407_407353


namespace min_f_when_a_neg3_range_of_a_l407_407347

open Real

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- First statement: Minimum value of f(x) when a = -3
theorem min_f_when_a_neg3 : (∀ x : ℝ, f x (-3) ≥ 4) ∧ (∃ x : ℝ,  f x (-3) = 4) := by
  sorry

-- Second statement: Range of a given the condition
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≤ 2 * a + 2 * abs (x - 1)) ↔ a ≥ 1/3 := by
  sorry

end min_f_when_a_neg3_range_of_a_l407_407347


namespace num_lattice_points_l407_407778

def is_lattice_point (p : ℤ × ℤ × ℤ) : Prop := 
  let (x, y, z) := p 
  x^2 + y^2 + z^2 = 25

theorem num_lattice_points : 
  {p : ℤ × ℤ × ℤ | is_lattice_point p}.to_finset.card = 18 :=
sorry

end num_lattice_points_l407_407778


namespace floor_sum_sqrt_1_to_25_l407_407058

theorem floor_sum_sqrt_1_to_25 : 
  ∑ n in Finset.range 25.succ, (⌊real.sqrt n⌋₊ : ℝ) = 75 := sorry

end floor_sum_sqrt_1_to_25_l407_407058


namespace problem_ACD_equivalency_l407_407152

noncomputable def f (x a : ℝ) : ℝ := real.exp x - a * x

theorem problem_ACD_equivalency :
  (∀ a, a ≤ 0 → (∀ x, 0 < deriv (λ x, f x a) x)) ∧
  (deriv (λ x, f x 1) 0 = 0 ∧ (∀ x < 0, deriv (λ x, f x 1) x < 0) ∧ (∀ x > 0, deriv (λ x, f x 1) x > 0)) ∧
  (∃ a > 0, ∃ x, (∀ x', (f x' a ≥ f x a)) ∧ (x ≠ 0)) :=
by
  sorry

end problem_ACD_equivalency_l407_407152


namespace floor_sqrt_sum_l407_407040

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l407_407040


namespace convert_90_deg_to_radians_l407_407620

theorem convert_90_deg_to_radians (pi_eq_180deg : 180 = real.pi) : 90 = real.pi / 2 := 
sorry

end convert_90_deg_to_radians_l407_407620


namespace min_distance_l407_407844

-- Define the equations of the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x + 2 * y + 1 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 4 * y + 6 = 0

-- Define the proposition to prove
theorem min_distance : 
  ∀ (P Q : ℝ × ℝ), (C1 P.1 P.2 ∧ C2 Q.1 Q.2) → 
  dist P Q ≥ 3 - real.sqrt 2 :=
sorry

end min_distance_l407_407844


namespace floor_sqrt_sum_l407_407044

theorem floor_sqrt_sum : 
  (∑ n in Finset.range 26, ⌊Real.sqrt n⌋₊) = 75 := by
  sorry

end floor_sqrt_sum_l407_407044


namespace volume_of_sphere_l407_407709

theorem volume_of_sphere (l r R : ℝ) (h : ℝ) (V : ℝ)
  (hl : l = 2 * √3)
  (hr : r = l / 2)
  (hh : h = √(l^2 - r^2))
  (hR : 2 * R = l^2 / h)
  (V_def : V = (4 / 3) * π * R^3) :
  V = (32 / 3) * π :=
  sorry

end volume_of_sphere_l407_407709


namespace expected_value_of_painted_faces_of_random_small_cube_l407_407976

noncomputable def cubeExpectedValue : ℚ :=
  let P : ℕ → ℚ := λ n, match n with
    | 0 => 64 / 125
    | 1 => 48 / 125
    | 2 => 12 / 125
    | 3 => 1 / 125
    | _ => 0
  in (0 : ℚ) * P 0 + (1 : ℚ) * P 1 + (2 : ℚ) * P 2 + (3 : ℚ) * P 3

theorem expected_value_of_painted_faces_of_random_small_cube :
  cubeExpectedValue = 3 / 5 := by
sorry

end expected_value_of_painted_faces_of_random_small_cube_l407_407976


namespace solve_for_k_l407_407268

theorem solve_for_k (k : ℝ) (h₀ : k > 0) (h₁ : (5 - k) = (k + 1) * (2 - k)) : k = (1 + real.sqrt 29) / 2 :=
by
  sorry

end solve_for_k_l407_407268


namespace sqrt_defined_range_l407_407743

theorem sqrt_defined_range (x : ℝ) : (∃ r : ℝ, r = sqrt (x - 5)) ↔ x ≥ 5 :=
by
  sorry

end sqrt_defined_range_l407_407743
