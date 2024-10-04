import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.GCDMonoid
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binom
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.Init.Data.Int.Order
import Mathlib.Logic.Basic
import Mathlib.Order.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.Real
import probability_theory

namespace prob_correct_l105_105682

-- Define the individual probabilities.
def prob_first_ring := 1 / 10
def prob_second_ring := 3 / 10
def prob_third_ring := 2 / 5
def prob_fourth_ring := 1 / 10

-- Define the total probability of answering within the first four rings.
def prob_answer_within_four_rings := 
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring

-- State the theorem.
theorem prob_correct : prob_answer_within_four_rings = 9 / 10 :=
by
  -- We insert a placeholder for the proof.
  sorry

end prob_correct_l105_105682


namespace candy_game_solution_l105_105472

open Nat

theorem candy_game_solution 
  (total_candies : ℕ) 
  (nick_candies : ℕ) 
  (tim_candies : ℕ)
  (tim_wins : ℕ)
  (m n : ℕ)
  (htotal : total_candies = 55) 
  (hnick : nick_candies = 30) 
  (htim : tim_candies = 25)
  (htim_wins : tim_wins = 2)
  (hrounds_total : total_candies = nick_candies + tim_candies)
  (hwinner_condition1 : m > n) 
  (hwinner_condition2 : n > 0) 
  (hwinner_candies_total : total_candies = tim_wins * m + (total_candies / (m + n) - tim_wins) * n)
: m = 8 := 
sorry

end candy_game_solution_l105_105472


namespace range_of_m_l105_105695

theorem range_of_m (m : ℝ)
  (p : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), x^2 - 2 * x - 4 * m^2 + 8 * m - 2 ≥ 0)
  (q : ∃ x ∈ set.Icc (1 : ℝ) (2 : ℝ), real.logb (1 / 2) (x^2 - m * x + 1) < -1)
  (H : (p ∨ q) ∧ ¬(p ∧ q)) :
  m < 1 / 2 ∨ m = 3 / 2 :=
by
  sorry

end range_of_m_l105_105695


namespace code_word_MEET_l105_105920

def translate_GREAT_TIME : String → ℕ 
| "G" => 0
| "R" => 1
| "E" => 2
| "A" => 3
| "T" => 4
| "I" => 5
| "M" => 6
| _   => 0 -- Default case for simplicity, not strictly necessary

theorem code_word_MEET : translate_GREAT_TIME "M" = 6 ∧ translate_GREAT_TIME "E" = 2 ∧ translate_GREAT_TIME "T" = 4 →
  let MEET : ℕ := (translate_GREAT_TIME "M" * 1000) + 
                  (translate_GREAT_TIME "E" * 100) + 
                  (translate_GREAT_TIME "E" * 10) + 
                  (translate_GREAT_TIME "T")
  MEET = 6224 :=
sorry

end code_word_MEET_l105_105920


namespace range_of_a_l105_105267

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x > a
def q (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem range_of_a 
  (h_sufficient : ∀ x, p x a → q x)
  (h_not_necessary : ∃ x, q x ∧ ¬ p x a) :
  a ≥ 1 :=
sorry

end range_of_a_l105_105267


namespace parity_of_expression_l105_105758

theorem parity_of_expression (a b c : ℕ) (ha : a % 2 = 1) (hb : b % 2 = 0) :
  (3 ^ a + (b - 1) ^ 2 * (c + 1)) % 2 = if c % 2 = 0 then 1 else 0 :=
by
  sorry

end parity_of_expression_l105_105758


namespace find_small_circle_circumference_l105_105515

def two_pi : ℝ := 2 * real.pi

noncomputable def radius_of_large_circle (C_L : ℝ) : ℝ := C_L / two_pi

noncomputable def radius_of_small_circle (C_L : ℝ) (ΔA : ℝ) := 
  let R := radius_of_large_circle C_L
  real.sqrt ((R^2 - ΔA / real.pi))

noncomputable def circumference_of_small_circle (C_s : ℝ) (C_L : ℝ) (ΔA : ℝ) : ℝ := 
  2 * real.pi * (radius_of_small_circle C_L ΔA)

theorem find_small_circle_circumference (C_L : ℝ) (ΔA : ℝ) (C_s : ℝ) :
  C_L = 380 ∧ ΔA = 5775.414574918697 → 
  abs (circumference_of_small_circle C_s C_L ΔA - 267.9) < 0.1 :=
begin
  sorry
end

end find_small_circle_circumference_l105_105515


namespace problem_statement_l105_105852

noncomputable def verify_monotonic_increasing : Prop :=
  ∀ (k : ℤ), monotonically_increasing (λ x : ℝ, sqrt 3 * sin x - cos x) 
  (set.Icc (2 * k * π - π / 3) (2 * k * π + 2 * π / 3))

theorem problem_statement : verify_monotonic_increasing :=
sorry

end problem_statement_l105_105852


namespace find_investment_l105_105576

noncomputable theory

def investments (a b c : ℕ) (profit total : ℕ) : Prop :=
  (400 : ℝ) / (1000 : ℝ) = (1200 : ℝ) / (800 + b + 1200 : ℝ)

theorem find_investment (b : ℕ) (a c : ℕ) (profit total : ℕ) (h1 : a = 800) 
                        (h2 : c = 1200) (h3 : profit = 400) (h4 : total = 1000) :
  investments a b c profit total → b = 1000 :=
by {
  intros h,
  sorry
}

end find_investment_l105_105576


namespace invalid_votes_percentage_l105_105790

theorem invalid_votes_percentage (A_valid_percentage : ℝ) (TotalVotes A_valid_votes : ℕ) 
  (h1 : A_valid_percentage = 80 / 100) 
  (h2 : TotalVotes = 560000) 
  (h3 : A_valid_votes = 380800) : 
  (InvalidPercentage : ℝ) :=
have h4 : V = A_valid_votes / A_valid_percentage, from sorry,
have h5 : 476000 = V, from sorry,
have h6 : 476000 = (100 - x) / 100 * TotalVotes, from sorry,
have h7 : x = 15, from sorry,
15

end invalid_votes_percentage_l105_105790


namespace a_is_one_l105_105459

noncomputable def odd (n : ℕ) : Prop := ¬ (∃ k : ℕ, n = 2 * k)

theorem a_is_one (a b c d : ℕ) (k m : ℕ) 
  (H1 : odd a) (H2 : odd b) (H3 : odd c) (H4 : odd d) 
  (H5 : 0 < a) (H6 : a < b) (H7 : b < c) (H8 : c < d) 
  (H9 : a * d = b * c) (H10 : a + d = 2^k) (H11 : b + c = 2^m) :
  a = 1 :=
begin
  sorry
end

end a_is_one_l105_105459


namespace younger_brother_fraction_silver_l105_105514

noncomputable def fraction_of_silver_received_young_brother (z s : ℝ) (h1 : z + s = 600) (h2 : z / 5 + s / 7 = 100) : ℝ :=
  let younger_gold := z / 7 in
  let younger_silver := 100 - younger_gold in
  let fraction := younger_silver / s in
  fraction

theorem younger_brother_fraction_silver (z s : ℝ) (h1 : z + s = 600) (h2 : z / 5 + s / 7 = 100) :
  fraction_of_silver_received_young_brother z s h1 h2 = 9 / 49 :=
by
  sorry

end younger_brother_fraction_silver_l105_105514


namespace evaluate_expression_l105_105218

noncomputable def x : ℝ := (Real.sqrt ((9: ℝ)^2)) - 2

theorem evaluate_expression : (Real.log 2 (3*x^3 - 7) + Real.sin (Real.pi * x) = Real.log 2 1022) :=
by
  -- Intermediate steps skipped
  sorry

end evaluate_expression_l105_105218


namespace inequality_solution_l105_105535

noncomputable def solution_set_inequality : Set ℝ := {x | -2 < x ∧ x < 1 / 3}

theorem inequality_solution :
  {x : ℝ | (2 * x - 1) / (3 * x + 1) > 1} = solution_set_inequality :=
by
  sorry

end inequality_solution_l105_105535


namespace largest_multiple_of_12_neg_gt_neg_150_l105_105965

theorem largest_multiple_of_12_neg_gt_neg_150 : ∃ m : ℤ, (m % 12 = 0) ∧ (-m > -150) ∧ ∀ n : ℤ, (n % 12 = 0) ∧ (-n > -150) → n ≤ m := sorry

end largest_multiple_of_12_neg_gt_neg_150_l105_105965


namespace central_cell_value_l105_105377

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l105_105377


namespace triangle_YZ_side_length_l105_105436

theorem triangle_YZ_side_length :
  ∀ (X Y Z : ℝ) (XY YZ XZ : ℝ),
  ∃ (X Y Z : ℝ),
  (cos (2 * X - Y) + sin (X + Y) = 2) ∧
  (XY = 6) →
  (YZ = 3 * real.sqrt 3) :=
begin
  sorry
end

end triangle_YZ_side_length_l105_105436


namespace exists_point_with_distances_greater_than_perimeter_l105_105873

theorem exists_point_with_distances_greater_than_perimeter
  (n : ℕ) (h1 : n ≥ 7) (P : Fin n → ℝ) :
  ∃ p : ℝ × ℝ, ∑ i in Finset.univ, dist p (P i) > (∑ i in Finset.univ, dist (P i) (P (i + 1) % n)) :=
sorry

end exists_point_with_distances_greater_than_perimeter_l105_105873


namespace geometric_sequence_problem_l105_105256

noncomputable def geometric_sequence_sum_condition 
  (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 1 + a 2 + a 3 + a 4 + a 5 = 6) ∧ 
  (a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 18) ∧ 
  (∀ n, a n = a 1 * q ^ (n - 1)) ∧ 
  (q ≠ 1)

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) (q : ℝ) 
  (h : geometric_sequence_sum_condition a q) : 
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := 
by 
  sorry

end geometric_sequence_problem_l105_105256


namespace number_of_integer_solutions_l105_105655

theorem number_of_integer_solutions : 
  {x : ℤ | (x^2 - 3*x + 1)^(2*x - 3) = 1}.card = 2 :=
by
  sorry

end number_of_integer_solutions_l105_105655


namespace correct_sunset_time_proof_l105_105016

def Time := ℕ × ℕ  -- hours and minutes

def sunrise_time : Time := (7, 12)  -- 7:12 AM
def incorrect_daylight_duration : Time := (11, 15)  -- 11 hours 15 minutes as per newspaper

def add_time (t1 t2 : Time) : Time :=
  let (h1, m1) := t1
  let (h2, m2) := t2
  let minutes := m1 + m2
  let hours := h1 + h2 + minutes / 60
  (hours % 24, minutes % 60)

def correct_sunset_time : Time := (18, 27)  -- 18:27 in 24-hour format equivalent to 6:27 PM in 12-hour format

theorem correct_sunset_time_proof :
  add_time sunrise_time incorrect_daylight_duration = correct_sunset_time :=
by
  -- skipping the detailed proof for now
  sorry

end correct_sunset_time_proof_l105_105016


namespace rhombus_constructibility_l105_105650

theorem rhombus_constructibility (k u : ℝ) : 
  (u > 0) → (k > 0) → (sqrt 2 * u ≤ k) ∧ (k < 2 * u) :=
by 
  intros
  sorry

end rhombus_constructibility_l105_105650


namespace interval_increasing_inequality_solution_l105_105273

noncomputable def lambda : ℝ := 2 * Real.sqrt 3

def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

def vec_b (x : ℝ) : ℝ × ℝ := 
  (lambda * Real.sin x - Real.cos x, Real.cos (π / 2 - x))

def f (x : ℝ) : ℝ := 
  (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

axiom condition : f (-π / 3) = f 0

theorem interval_increasing :
  ∃ (k : ℤ), ∀ x, (kπ - π / 6 ≤ x ∧ x ≤ kπ + π / 3) ∧
    (kπ < kπ ∧ kπ + π / 3 < kπ + π / 3) :=
sorry

theorem inequality_solution :
  ∃ (k : ℤ), ∀ x, (kπ < x ∧ x < kπ + π / 6) :=
sorry

end interval_increasing_inequality_solution_l105_105273


namespace general_equation_line_C1_center_of_circle_C2_polar_length_of_chord_AB_l105_105800

-- Definitions based on the conditions
def LineC1 (t : ℝ) : ℝ × ℝ := (1 + t, 2 + t)

def CircleC2_polar (θ : ℝ) : ℝ :=
  -2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

def CircleC2_center_rect : ℝ × ℝ := (-1, Real.sqrt 3)

def CircleC2_center_polar : ℝ × ℝ := (2, 2 * Real.pi / 3)

-- Proof statements
theorem general_equation_line_C1 (t : ℝ) : 
  let (x, y) := LineC1 t in x - y + 1 = 0 := 
sorry

theorem center_of_circle_C2_polar :
  CircleC2_center_polar = (2, 2 * Real.pi / 3) :=
sorry

theorem length_of_chord_AB :
  let center := CircleC2_center_rect in
  let line_eq := ∀ t, let (x, y) := LineC1 t in x - y + 1 = 0 in
  let d := |center.1 + center.2 - 1| / Real.sqrt 2 in
  d = Real.sqrt 6 / 2 →
  let AB := 2 * Real.sqrt (4 - (d^2) / 4) in
  AB = Real.sqrt 10 :=
sorry

end general_equation_line_C1_center_of_circle_C2_polar_length_of_chord_AB_l105_105800


namespace domain_intersection_l105_105047

noncomputable def domain_f : Set ℝ := {x : ℝ | 1 - x > 0}
noncomputable def domain_g : Set ℝ := {x : ℝ | 1 + x > 0}

theorem domain_intersection : domain_f ∩ domain_g = set.Ioo (-1 : ℝ) 1 :=
by
  sorry

end domain_intersection_l105_105047


namespace sqrt_sum_lt_four_l105_105062

theorem sqrt_sum_lt_four (x y z t : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1)
  (hz : 0 < z ∧ z < 1) (ht : 0 < t ∧ t < 1) :
  sqrt(x^2 + (1 - t)^2) + sqrt(y^2 + (1 - x)^2) + sqrt(z^2 + (1 - y)^2) + sqrt(t^2 + (1 - z)^2) < 4 :=
by
  sorry

end sqrt_sum_lt_four_l105_105062


namespace compare_fx_l105_105730

noncomputable def f (a x : ℝ) := a * x ^ 2 + 2 * a * x + 4

theorem compare_fx (a x1 x2 : ℝ) (h₁ : -3 < a) (h₂ : a < 0) (h₃ : x1 < x2) (h₄ : x1 + x2 ≠ 1 + a) :
  f a x1 > f a x2 :=
sorry

end compare_fx_l105_105730


namespace friends_configuration_count_l105_105206

theorem friends_configuration_count : 
  let friends := ["Alice", "Bob", "Charlie"] in
  let configurations : set (set (set String)) := {
    {{"Alice", "Bob"}, {"Bob", "Charlie"}, {"Alice", "Charlie"}}, -- Triangle
    {{"Alice", "Bob"}, {"Alice", "Charlie"}}, -- Alice in the middle
    {{"Bob", "Alice"}, {"Bob", "Charlie"}}, -- Bob in the middle
    {{"Charlie", "Alice"}, {"Charlie", "Bob"}} -- Charlie in the middle
  } in
  configurations.card = 4 :=
by
  sorry

end friends_configuration_count_l105_105206


namespace find_value_added_l105_105667

theorem find_value_added (x y : ℕ) (hx : x = 3) (h : x + y = 60 * (1/x)) : y = 17 :=
by
  subst hx
  rw Nat.div_eq_of_lt (show 3 > 0 from Nat.succ_pos 2) at h
  exact h
  sorry

end find_value_added_l105_105667


namespace rationalize_denominator_correct_l105_105880

noncomputable def rationalize_denominator : Prop :=
  (1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2)

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l105_105880


namespace find_central_cell_l105_105413

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l105_105413


namespace sufficient_and_necessary_condition_l105_105460

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (x + Real.sqrt (x^2 + 1)) / Real.log 2

theorem sufficient_and_necessary_condition {a b : ℝ} (h : a + b ≥ 0) : f a + f b ≥ 0 :=
sorry

end sufficient_and_necessary_condition_l105_105460


namespace minimum_additional_marbles_l105_105853

noncomputable def LisaMarbles : Nat := 60
noncomputable def Friends : Nat := 12
noncomputable def PrimeMarblesLeft : Nat := 2
noncomputable def NeededMarbles : Nat := 78 + PrimeMarblesLeft - LisaMarbles

theorem minimum_additional_marbles : NeededMarbles = 20 := by
  have total_marbles_needed : Nat := (Friends * (Friends + 1)) / 2
  have initial_marbles : Nat := LisaMarbles
  have additional_marbles_needed : Nat := total_marbles_needed + PrimeMarblesLeft - initial_marbles
  have prime_marble_condition : additional_marbles_needed = 20
  rw [←prime_marble_condition]
  sorry

end minimum_additional_marbles_l105_105853


namespace find_P8_l105_105453

theorem find_P8 :
  (∃ P : Polynomial ℝ, P.monic ∧ degree P = 7 ∧
      P.eval 1 = 1 ∧
      P.eval 2 = 2 ∧
      P.eval 3 = 3 ∧
      P.eval 4 = 4 ∧
      P.eval 5 = 5 ∧
      P.eval 6 = 6 ∧
      P.eval 7 = 7) →
  ∃ P : Polynomial ℝ, P.eval 8 = 5048 := 
by
  sorry

end find_P8_l105_105453


namespace sufficient_but_not_necessary_log_condition_l105_105280

theorem sufficient_but_not_necessary_log_condition :
  ∀ (x a : ℝ), (0 < x ∧ x < 1/2) →
  (a < 0 → log (1/2) x > x + a) ∧
  ((∃ a : ℝ, a < 0 ∧ log (1/2) x > x) ∧ (∃ a : ℝ, log (1/2) x > x + a)) :=
by
  sorry

end sufficient_but_not_necessary_log_condition_l105_105280


namespace verify_BG_BF_verify_FG_EG_find_x_l105_105798

noncomputable def verify_angles (CBG GBE EBF BCF FCE : ℝ) :=
  CBG = 20 ∧ GBE = 40 ∧ EBF = 20 ∧ BCF = 50 ∧ FCE = 30

theorem verify_BG_BF (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → BG = BF :=
by
  sorry

theorem verify_FG_EG (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → FG = EG :=
by
  sorry

theorem find_x (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → x = 30 :=
by
  sorry

end verify_BG_BF_verify_FG_EG_find_x_l105_105798


namespace min_value_of_reciprocal_squares_of_coefficients_l105_105300

theorem min_value_of_reciprocal_squares_of_coefficients 
  (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (cond : ∃ (C_1 C_2 : ℝ × ℝ → ℝ), C_1 = λ p, p.1 ^ 2 + p.2 ^ 2 + 4 * a * p.1 + 4 * a ^ 2 - 4 ∧ C_2 = λ p, p.1 ^ 2 + p.2 ^ 2 - 2 * b * p.2 + b ^ 2 - 1)
  (common_tangent : ∀ (C_1 C_2 : ℝ × ℝ → ℝ), C_1 = λ p, p.1 ^ 2 + p.2 ^ 2 + 4 * a * p.1 + 4 * a ^ 2 - 4 →  C_2 = λ p, p.1 ^ 2 + p.2 ^ 2 - 2 * b * p.2 + b ^ 2 - 1 → ∃ l : ℝ × ℝ → ℝ, ∀ (p : ℝ × ℝ), l p = 0 → (C_1 p = 0 ∨  C_2 p = 0)) :
  ∃ (x : ℝ), x = (4 * (a^2) + b^2) ∧ ∀ y : ℝ, y = 1 / (a ^ 2) + 1 / (b ^ 2) → y = 9 :=
by sorry

end min_value_of_reciprocal_squares_of_coefficients_l105_105300


namespace ce_ad_ratio_l105_105775

theorem ce_ad_ratio (CD DB AE EB CP PE : ℝ) (h₁ : CD / DB = 4 / 1) (h₂ : AE / EB = 2 / 3) :
  CP / PE = 10 := 
sorry

end ce_ad_ratio_l105_105775


namespace inletRate_is_3_l105_105178

def volumeTank (v_cubic_feet : ℕ) : ℕ :=
  1728 * v_cubic_feet

def outletRate1 : ℕ := 9 -- rate of first outlet in cubic inches/min
def outletRate2 : ℕ := 6 -- rate of second outlet in cubic inches/min
def tankVolume : ℕ := volumeTank 30 -- tank volume in cubic inches
def minutesToEmpty : ℕ := 4320 -- time to empty the tank in minutes

def effectiveRate (inletRate : ℕ) : ℕ :=
  outletRate1 + outletRate2 - inletRate

theorem inletRate_is_3 : (15 - 3) * minutesToEmpty = tankVolume :=
  by simp [outletRate1, outletRate2, tankVolume, minutesToEmpty]; sorry

end inletRate_is_3_l105_105178


namespace vector_dot_product_l105_105741

-- Define the vectors a and b as unit vectors.
variables (a b : ℝ^3)
axiom a_unit : ‖a‖ = 1
axiom b_unit : ‖b‖ = 1
axiom angle_120 : real.angle a b = real.pi * 2 / 3

-- Define the problem statement
theorem vector_dot_product (a b : ℝ^3) (h_a_unit : ‖a‖ = 1) (h_b_unit : ‖b‖ = 1) (h_angle_120 : real.angle a b = real.pi * 2/3) :
  (a + b) • b = 1 / 2 := sorry

end vector_dot_product_l105_105741


namespace sqrt_sum_ineq_l105_105060

theorem sqrt_sum_ineq (x y z t : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1)
                      (hz : 0 < z ∧ z < 1) (ht : 0 < t ∧ t < 1) :
    sqrt (x^2 + (1 - t)^2) + sqrt (y^2 + (1 - x)^2) +
    sqrt (z^2 + (1 - y)^2) + sqrt (t^2 + (1 - z)^2) < 4 :=
by 
  sorry

end sqrt_sum_ineq_l105_105060


namespace point_A_coordinates_l105_105726

-- Definition of the ellipse
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- The focus of the ellipse
def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Line passing through the right focus with non-zero slope
def line_through_focus (k x : ℝ) : ℝ := k * (x - (Real.sqrt 3))

-- Condition AO^2 = AB
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2)^2 + (y1 - y2)^2

def AO_squared_eq_AB (xA yA xB yB : ℝ) : Prop :=
  distance_squared xA yA 0 0 = distance_squared xA yA xB yB

-- The main theorem to be proven
theorem point_A_coordinates :
  ∀ (xA yA xB yB k : ℝ),
  ellipse xA yA →
  ellipse xB yB →
  yA = line_through_focus k xA →
  yB = line_through_focus k xB →
  AO_squared_eq_AB xA yA xB yB →
  (xA = Real.sqrt 3 / 2 ∧ (yA = Real.sqrt 13 / 4 ∨ yA = - Real.sqrt 13 / 4)) :=
by
  intros xA yA xB yB k h_ellipseA h_ellipseB h_lineA h_lineB h_AOeqAB
  sorry

end point_A_coordinates_l105_105726


namespace ellipse_problem_l105_105702

-- Define the conditions and the theorem
theorem ellipse_problem :
  ∃ (N : ℝ × ℝ), N.1 = 11 / 8 ∧ N.2 = 0 ∧
    (∀ (a b : ℝ), a > b ∧ b > 0 ∧ ((a^2 - b^2)/(a^2)) = 1/2 ∧
    (∃ (F b : ℝ), 1 + (a*b^2) - 3/4 * (b^2 + (a^2 - b^2)) = sqrt(3)/2 * sqrt(b^2 + a^2 - b^2) ∧ 
    a^2 = b^2 + (sqrt(3)/2 * sqrt(b^2)) ∧ 
    (∀ (x1 x2 y1 y2 m k : ℝ), (3*x1^2 + 4*y1^2 = 12) ∧ y1 = k*(x1-1) ∧ 
    (3*x2^2 + 4*y2^2 = 12) ∧ y2 = k*(x2-1) ∧ 
    ((x1 - m)*(x2 - m) + y1 * y2 = m^2 - m*(8*k^2/(4*k^2+3)) + k^2*(x1 - 1)*(x2 - 1)) →
    ((1 + k^2)*(4*k^2 - 12)/(4*k^2 + 3) - (m + k^2)*8*k^2/(4*k^2 + 3) + k^2 + m^2 = 4/3) →
    m = 11/8 ∧ k*(x1 - 1) = -135/64)) :=
begin
  sorry
end

end ellipse_problem_l105_105702


namespace bisected_chord_exists_l105_105698

noncomputable def exists_bisected_chord (K : Set Point) (A : Point) : Prop :=
  convex K →
  A ∈ K →
  ∃ C D : Point, segment C D ∈ K ∧ A ∈ segment C D ∧ dist C A = dist A D

theorem bisected_chord_exists (K : Set Point) (A : Point) :
  exists_bisected_chord K A :=
by
  sorry

end bisected_chord_exists_l105_105698


namespace standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l105_105542

theorem standing_in_a_row (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 → 
  ∃ (ways : ℕ), ways = 120 :=
by
  sorry

theorem standing_in_a_row_AB_adj_CD_not_adj (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 24 :=
by
  sorry

theorem assign_to_classes (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 150 :=
by
  sorry

end standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l105_105542


namespace correct_relationship_l105_105733

-- Define the function
def f (x : ℝ) : ℝ := x / Real.log x

-- State the problem as a theorem to prove the correct option
theorem correct_relationship (x : ℝ) (h : 0 < x ∧ x < 1) : 
  f x < f (x^2) ∧ f (x^2) < f (f x) := 
sorry

end correct_relationship_l105_105733


namespace diagonals_150_sides_not_half_of_diagonals_150_sides_l105_105679

theorem diagonals_150_sides :
  let n := 150
  in (n * (n - 3)) / 2 = 11025 := 
by
  sorry

theorem not_half_of_diagonals_150_sides :
  9900 ≠ 11025 / 2 := 
by
  sorry

end diagonals_150_sides_not_half_of_diagonals_150_sides_l105_105679


namespace jana_total_distance_l105_105812

def jana_walk_rate : ℝ := 1 / 30  -- rate in miles per minute
def jana_cycle_rate : ℝ := 2 / 15  -- rate in miles per minute
def jana_walk_time : ℝ := 45  -- time in minutes
def jana_cycle_time : ℝ := 30  -- time in minutes

theorem jana_total_distance :
  jana_walk_rate * jana_walk_time + jana_cycle_rate * jana_cycle_time = 5.5 :=
by
  sorry

end jana_total_distance_l105_105812


namespace solve_triangle_l105_105439

variable {A B C : ℝ}
variable {a b c : ℝ}

noncomputable def sin_B_plus_pi_four (a b c : ℝ) : ℝ :=
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  let sin_B := Real.sqrt (1 - cos_B^2)
  sin_B * Real.sqrt 2 / 2 + cos_B * Real.sqrt 2 / 2

theorem solve_triangle 
  (a b c : ℝ)
  (h1 : b = 2 * Real.sqrt 5)
  (h2 : c = 3)
  (h3 : 3 * a * (a^2 + b^2 - c^2) / (2 * a * b) = 2 * c * (b^2 + c^2 - a^2) / (2 * b * c)) :
  a = Real.sqrt 5 ∧ 
  sin_B_plus_pi_four a b c = Real.sqrt 10 / 10 :=
by 
  sorry

end solve_triangle_l105_105439


namespace find_d_l105_105052

theorem find_d (d : ℝ) : (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  { sorry }

end find_d_l105_105052


namespace find_central_cell_l105_105417

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l105_105417


namespace asymptote_and_slope_of_parabola_l105_105712

/-- Definitions and Conditions --/

-- Definition for hyperbola C
def hyperbola_C (x y : ℝ) (a : ℝ) := 
  a > 0 ∧ (x^2 / a^2 - y^2 / 9 = 1)

-- Point P on the hyperbola and |PF1| - |PF2| = 2
def point_P_on_hyperbola_P_property (P F1 F2 : ℝ × ℝ) (a : ℝ) := 
  ∃ (x y : ℝ), hyperbola_C x y a ∧ P = (x, y) ∧ abs (dist P F1 - dist P F2) = 2

-- Right vertex and parabola L
def right_vertex_focus_parabola_L (C L: ℝ → ℝ → Prop) (a : ℝ) :=
  ∃ (x y : ℝ), C x y ∧ L y = 4 * x

-- Main proof problem (questions + correct answers)
theorem asymptote_and_slope_of_parabola 
  (a : ℝ)
  (C L : ℝ → ℝ → Prop) 
  (P F1 F2 : ℝ × ℝ) :
  -- Given conditions
  (hyperbola_C x y a)  ∧ 
  (point_P_on_hyperbola_P_property P F1 F2 a) ∧ 
  (right_vertex_focus_parabola_L C L a) →

  -- Questions and correct answers
  -- (Ⅰ) Asymptotes and standard equation of parabola L
  (∀ x y : ℝ, hyperbola_C x y a → C x y = (x ^ 2 / 1 ^ 2) - y ^ 2 / 9 = 1 ) ∧
  (asymptotes C = (y = 3 * x) ∨ (y = -3 * x)) ∧
  ( ∀ x y : ℝ, right_focus_parabola_L = L y ^ 2 = 4 * x ) ∧
  -- (Ⅱ) Slope of the line
  (∀ k : ℝ, slope_of_line.circle_and_focus k = k = ± (sqrt 2 / 2)) :=
sorry

end asymptote_and_slope_of_parabola_l105_105712


namespace distance_between_centers_of_circles_l105_105517

theorem distance_between_centers_of_circles :
  ∀ (rect_width rect_height circle_radius distance_between_centers : ℝ),
  rect_width = 11 
  ∧ rect_height = 7 
  ∧ circle_radius = rect_height / 2 
  ∧ distance_between_centers = rect_width - 2 * circle_radius 
  → distance_between_centers = 4 := by
  intros rect_width rect_height circle_radius distance_between_centers
  sorry

end distance_between_centers_of_circles_l105_105517


namespace largest_multiple_of_12_neg_gt_neg_150_l105_105964

theorem largest_multiple_of_12_neg_gt_neg_150 : ∃ m : ℤ, (m % 12 = 0) ∧ (-m > -150) ∧ ∀ n : ℤ, (n % 12 = 0) ∧ (-n > -150) → n ≤ m := sorry

end largest_multiple_of_12_neg_gt_neg_150_l105_105964


namespace original_fraction_is_one_third_l105_105042

theorem original_fraction_is_one_third
  (a b : ℕ) (h₁ : Nat.gcd a b = 1)
  (h₂ : (a + 2) * b = 3 * a * b^2) : 
  a = 1 ∧ b = 3 := by
  sorry

end original_fraction_is_one_third_l105_105042


namespace magician_method_N_2k_magician_method_values_l105_105166

-- (a) Prove that if there is a method for N = k, then there is a method for N = 2k.
theorem magician_method_N_2k (k : ℕ) (method_k : Prop) : 
  (∃ method_N_k : Prop, method_k → method_N_k) → 
  (∃ method_N_2k : Prop, method_k → method_N_2k) :=
sorry

-- (b) Find all values of N for which the magician and the assistant have a method.
theorem magician_method_values (N : ℕ) : 
  (∃ method : Prop, method) ↔ (∃ m : ℕ, N = 2^m) :=
sorry

end magician_method_N_2k_magician_method_values_l105_105166


namespace lines_parallel_if_perpendicular_to_same_plane_l105_105176

-- Definitions for lines and planes in a geometric context.
variable (P : Type) [Plane P] 
variable (l1 l2 : Line)
variable (parallel_to_plane : l1 ∥ P ∧ l2 ∥ P)
variable (equal_angles : ∀ θ, measure_angle (l1, P) = measure_angle (l2, P))
variable (l1_parallel_to_l2_plane : l1 ∥ plane_containing l2)
variable (perpendicular_to_same_plane : ∀ P, l1 ⊥ P ∧ l2 ⊥ P)

-- Statement of the theorem that a sufficient condition for l1 and l2 to be parallel is:
theorem lines_parallel_if_perpendicular_to_same_plane :
  (∀ P, l1 ⊥ P ∧ l2 ⊥ P) → (l1 ∥ l2) :=
begin
  sorry
end

end lines_parallel_if_perpendicular_to_same_plane_l105_105176


namespace range_of_a_l105_105792

open Real

theorem range_of_a (a x y : ℝ)
  (h1 : (x - a) ^ 2 + (y - (a + 2)) ^ 2 = 1)
  (h2 : ∃ M : ℝ × ℝ, (M.1 - a) ^ 2 + (M.2 - (a + 2)) ^ 2 = 1
                       ∧ dist M (0, 3) = 2 * dist M (0, 0)) :
  -3 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l105_105792


namespace quadratic_has_two_distinct_real_roots_l105_105083

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l105_105083


namespace min_points_triangle_with_integer_area_l105_105452

theorem min_points_triangle_with_integer_area (n : ℕ) 
  (points : Fin n → Fin 100 × Fin 100)
  (h : n = 5) :
  ∃ (i j k : Fin n), 
    i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
    let (x1, y1) := points i,
        (x2, y2) := points j,
        (x3, y3) := points k in
    (1/2 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))).denominator = 1 :=
by
  sorry

end min_points_triangle_with_integer_area_l105_105452


namespace perimeter_of_triangle_AEC_l105_105605

-- Define the vertices and side length
def A := (0, 2)
def B := (0, 0)
def C := (2, 0)
def D := (2, 2)
def side_length := 2

-- Define the coordinates after the fold
def C' := (2, 4/3)
def E := (3/2, 3/2)

-- Calculate distances with auxiliary functions:
def distance (p1 p2 : (ℚ × ℚ)) : ℚ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distances AE, AC', EC'
def AE := distance A E
def AC' := distance A C'
def EC' := distance E C'

-- Define the perimeter of triangle △AEC'
def perimeter_AEC' := AE + EC' + AC'

-- The main statement we need to prove
theorem perimeter_of_triangle_AEC' : 
  perimeter_AEC' = (17 * real.sqrt 10 + real.sqrt 37) / 12 := sorry

end perimeter_of_triangle_AEC_l105_105605


namespace triangle_problem_l105_105803

theorem triangle_problem (n : ℕ) (h : 1 < n ∧ n < 4) : n = 2 ∨ n = 3 :=
by
  -- Valid realizability proof omitted
  sorry

end triangle_problem_l105_105803


namespace log_base_change_l105_105871

theorem log_base_change {a k P : ℝ} (h₁ : 0 < a) (h₂ : a ≠ 1) : log (a^k) P = (log a P) / k := 
by
  sorry

end log_base_change_l105_105871


namespace f_zeros_f_inequality_solution_l105_105731

noncomputable def f : ℝ → ℝ := λ x, if x ≤ 0 then (1 / 2) ^ x - 2 else log (x + 1) / log 2 - 1

theorem f_zeros : f (-1) = 0 ∧ f 1 = 0 :=
by {
  -- Provide reasoning or proof here (omitted)
  sorry
}

theorem f_inequality_solution : ∀ x, f x > 0 ↔ x < -1 ∨ x > 1 :=
by {
  -- Provide reasoning or proof here (omitted)
  sorry
}

end f_zeros_f_inequality_solution_l105_105731


namespace median_moon_count_l105_105558

def moon_counts : List ℕ := [0, 0, 0, 1, 1, 2, 5, 14, 27, 67, 82]

theorem median_moon_count : median moon_counts = 2 :=
by
  sorry

end median_moon_count_l105_105558


namespace philosophy_implies_①②③_l105_105442

-- Defining the conditions as Lean 4 definitions
def iodine_essential : Prop := true
def one_size_fits_all_can_lead_to_excessive_intake : Prop := true
def shanghai_increased_sale_non_iodized_salt : Prop := true
def residents_adjust_iodine_intake_according_to_situation : Prop := true

-- Defining the propositions ①, ②, ③
def grasp_measure_of_things : Prop := true
def strive_for_comprehensive_view : Prop := true
def analyze_specific_issues_specifically : Prop := true

-- The reflection of the philosophy
def philosophy_reflected := iodine_essential ∧ one_size_fits_all_can_lead_to_excessive_intake ∧ 
                             shanghai_increased_sale_non_iodized_salt ∧ 
                             residents_adjust_iodine_intake_according_to_situation

theorem philosophy_implies_①②③ :
  philosophy_reflected → grasp_measure_of_things ∧ strive_for_comprehensive_view ∧ analyze_specific_issues_specifically :=
by {
  intros,
  sorry
}

end philosophy_implies_①②③_l105_105442


namespace solve_equation_l105_105930

theorem solve_equation : ∃ x : ℝ, 2 * x - 1 = 3 * x + 2 ∧ x = -3 :=
by
  use -3
  split
  · exact (eq_of_sub_eq_zero (calc
      2 * (-3) - 1 = -6 - 1 : by norm_num
      ... = -7          : by norm_num))
      ... =(3 * -,3 + 2  : calc
      3 * (-3) + 2 = -9 + 2 : by norm_num
      ... = -7 : by norm_num))

  sorry

end solve_equation_l105_105930


namespace tyler_bird_pairs_l105_105121

theorem tyler_bird_pairs (n_species : ℕ) (pairs_per_species : ℕ) (total_pairs : ℕ)
  (h1 : n_species = 29)
  (h2 : pairs_per_species = 7)
  (h3 : total_pairs = n_species * pairs_per_species) : total_pairs = 203 :=
by
  sorry

end tyler_bird_pairs_l105_105121


namespace least_multiple_of_29_gt_500_l105_105127

theorem least_multiple_of_29_gt_500 : ∃ n : ℕ, n > 0 ∧ 29 * n > 500 ∧ 29 * n = 522 :=
by
  use 18
  sorry

end least_multiple_of_29_gt_500_l105_105127


namespace borya_wins_with_optimal_play_l105_105944

def points := 123

structure game_state :=
  (points: ℕ)
  (colors: list (option bool)) -- None for uncolored points, some bool where true is blue, false is red

noncomputable def player_wins (player1_first: bool) (current_state: game_state): bool :=
  sorry -- This definition will hold the logic of the game to check if a player has won

theorem borya_wins_with_optimal_play : 
  ∀(initial_state: game_state), 
  initial_state.points = points →
  (∀i < initial_state.colors.length, initial_state.colors.nth i ∈ [none]) →
  player_wins false initial_state :=
by
  sorry

end borya_wins_with_optimal_play_l105_105944


namespace three_digit_multiples_of_7_l105_105315

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l105_105315


namespace remainder_of_division_l105_105130

-- Define the dividend and divisor
def dividend : ℕ := 3^303 + 303
def divisor : ℕ := 3^101 + 3^51 + 1

-- State the theorem to be proven
theorem remainder_of_division:
  (dividend % divisor) = 303 := by
  sorry

end remainder_of_division_l105_105130


namespace increase_by_percentage_l105_105151

theorem increase_by_percentage (x : ℝ) (y : ℝ): x = 90 → y = 0.50 → x + x * y = 135 := 
by
  intro h1 h2
  sorry

end increase_by_percentage_l105_105151


namespace average_speed_calculation_l105_105590

def distance_km := 136.4
def time_hours := 6
def time_minutes := 30

def distance_meters := distance_km * 1000
def time_seconds := (time_hours * 60 + time_minutes) * 60

theorem average_speed_calculation :
  (distance_meters / time_seconds) ≈ 5.83 := by
  sorry

end average_speed_calculation_l105_105590


namespace absolute_value_difference_l105_105132

theorem absolute_value_difference :
  |(14 - 5)| - |(8 - 12)| = 5 := by
  sorry

end absolute_value_difference_l105_105132


namespace range_f_l105_105734

noncomputable def f (x : ℝ) : ℝ := 3 * sin(2 * x - π / 6)

theorem range_f : 
  set.Icc (-3 / 2 : ℝ) 3 = set.image f (set.Icc (0 : ℝ) (π / 2)) :=
sorry

end range_f_l105_105734


namespace jessica_family_travel_time_l105_105813

def distance_to_new_york : ℝ := 525
def travel_speed : ℝ := 45
def rest_duration_minutes : ℝ := 45
def time_between_rests_hours : ℝ := 1.5
def total_travel_time : ℝ := 16.92

theorem jessica_family_travel_time :
  let total_driving_time := distance_to_new_york / travel_speed,
      number_of_rest_stops := (total_driving_time / time_between_rests_hours).toInt,
      total_rest_time_minutes := number_of_rest_stops * rest_duration_minutes,
      total_rest_time_hours := total_rest_time_minutes / 60,
      total_travel_time_computed := total_driving_time + total_rest_time_hours
  in total_travel_time_computed = total_travel_time := by
  -- proof or sorry goes here
  sorry

end jessica_family_travel_time_l105_105813


namespace modulus_z_l105_105716

-- Define the imaginary unit
def I : ℂ := complex.I

-- Define the complex number z
def z : ℂ := 2 * I - 5 / (2 - I)

-- State the theorem which asserts that the modulus of z is √5
theorem modulus_z : ∥z∥ = real.sqrt 5 := by
  sorry

end modulus_z_l105_105716


namespace find_m_l105_105213

def sequence (x : ℕ → ℝ) : Prop :=
  x 0 = 7 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 6 * x n - 2) / (x n + 8)

theorem find_m : ∃ m : ℕ, sequence x ∧ x m ≤ 3 + 1 / 2^10 ∧ 35 ≤ m ∧ m ≤ 60 :=
sorry

end find_m_l105_105213


namespace patricia_initial_candies_l105_105869

def candies_after_taken_away (initial_candies taken_away : ℕ) : ℕ :=
  initial_candies - taken_away

theorem patricia_initial_candies :
  ∃ initial_candies : ℕ, candies_after_taken_away initial_candies 5 = 71 :=
begin
  use 76,
  unfold candies_after_taken_away,
  simp,
  sorry
end

end patricia_initial_candies_l105_105869


namespace sqrt_fourth_power_l105_105978

theorem sqrt_fourth_power (x : ℝ) (h : (√x)^4 = 256) : x = 16 := 
sorry

end sqrt_fourth_power_l105_105978


namespace fraction_addition_l105_105246

theorem fraction_addition (a b : ℕ) (hb : b ≠ 0) (h : a / (b : ℚ) = 3 / 5) : (a + b) / (b : ℚ) = 8 / 5 := 
by
sorry

end fraction_addition_l105_105246


namespace even_function_among_options_l105_105832

-- Define the function f(x) = 1 / (x^3 + 1)
def f (x : ℝ) : ℝ := 1 / (x^3 + 1)

-- Define the given functions
def A (x : ℝ) : ℝ := f (x + 1)
def B (x : ℝ) : ℝ := f (2 * x)
def C (x : ℝ) : ℝ := f (x - 1)
def D (x : ℝ) : ℝ := f (x^2)

-- Define the statement that D is the even function among A, B, C, D
theorem even_function_among_options : 
  ∀ x : ℝ, D x = D (-x) ∧ (A x ≠ A (-x) ∨ B x ≠ B (-x) ∨ C x ≠ C (-x)) := sorry

end even_function_among_options_l105_105832


namespace proposition_p_proposition_q_math_proof_problem_l105_105478

theorem proposition_p : ∀ {A B C : ℝ} (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = 180), (∠C > ∠B ↔ Math.sin ∠C > Math.sin ∠B) :=
by
  sorry

theorem proposition_q : ∀ {a b c : ℝ}, (a > b → a^2 * c > b^2 * c) ∧ ¬ (a^2 * c > b^2 * c → a > b) :=
by
  sorry

theorem math_proof_problem : 
  (∀ {A B C : ℝ} (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = 180), (∠C > ∠B ↔ Math.sin ∠C > Math.sin ∠B)) ∧ 
  ¬ (∀ {a b c : ℝ}, (a > b → a^2 * c > b^2 * c) ∧ ¬ (a^2 * c > b^2 * c → a > b)) :=
by
  exact proposition_p ∧ not proposition_q 

end proposition_p_proposition_q_math_proof_problem_l105_105478


namespace factors_of_expr_l105_105950

-- Given that a, b, and c each have exactly four natural-number factors
axiom h_a : ∃ p₁ : ℕ, prime p₁ ∧ a = p₁^3
axiom h_b : ∃ p₂ : ℕ, prime p₂ ∧ b = p₂^3
axiom h_c : ∃ p₃ : ℕ, prime p₃ ∧ c = p₃^3

-- Define a, b, c
variables {a b c : ℕ}

-- Define a statement that the number of factors of a^3 * b^2 * c^4 is 910
theorem factors_of_expr : 
  ∀ (a b c : ℕ), (∃ p₁ : ℕ, prime p₁ ∧ a = p₁^3) → (∃ p₂ : ℕ, prime p₂ ∧ b = p₂^3) → 
  (∃ p₃ : ℕ, prime p₃ ∧ c = p₃^3) → 
  nat.factors (a^3 * b^2 * c^4) = 910 :=
by 
  sorry

end factors_of_expr_l105_105950


namespace coeff_x2_expansion_l105_105900

theorem coeff_x2_expansion :
  let polynomial := ((Polynomial.X + Polynomial.C 1) ^ 5 * (Polynomial.X - Polynomial.C 2))
  Polynomial.coeff polynomial 2 = -15 :=
by
  sorry

end coeff_x2_expansion_l105_105900


namespace quadratic_has_two_distinct_real_roots_l105_105084

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l105_105084


namespace painted_by_all_three_l105_105110

/-
Statement: Given that 75% of the floor is painted red, 70% painted green, and 65% painted blue,
prove that at least 10% of the floor is painted with all three colors.
-/

def painted_by_red (floor : ℝ) : ℝ := 0.75 * floor
def painted_by_green (floor : ℝ) : ℝ := 0.70 * floor
def painted_by_blue (floor : ℝ) : ℝ := 0.65 * floor

theorem painted_by_all_three (floor : ℝ) :
  ∃ (x : ℝ), x = 0.10 * floor ∧
  (painted_by_red floor) + (painted_by_green floor) + (painted_by_blue floor) ≥ 2 * floor :=
sorry

end painted_by_all_three_l105_105110


namespace tangent_perpendicular_at_point_l105_105361

theorem tangent_perpendicular_at_point (a : ℝ) :
  (∀ x, y = Real.log x + x^2 + 1 → ∀ y', y' = (1 / x + 2 * x) → 
  (∀ A B, (x = A ∧ y = B) → ∀ x A B, (x = 1 ∧ y = 2)) → by slope_perpendicular (tangent_slope, line_slope, derivative, point)) → a = 3 := sorry

end tangent_perpendicular_at_point_l105_105361


namespace sum_of_x_and_y_l105_105533

theorem sum_of_x_and_y :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧
    (let n := 540
     in n * x = k * k ∧
        n * y = m * m * m) ∧
    (∀ z : ℕ, z > 0 → (540 * z = k * k → z ≥ x) ∧ (540 * z = m * m * m → z ≥ y)) ∧
    x + y = 65) :=
begin
  sorry
end

end sum_of_x_and_y_l105_105533


namespace number_of_shares_is_25_l105_105168

def wife_weekly_savings := 100
def husband_monthly_savings := 225
def duration_months := 4
def cost_per_share := 50

def total_savings : ℕ :=
  (wife_weekly_savings * 4 * duration_months) + (husband_monthly_savings * duration_months)

def amount_invested := total_savings / 2

def number_of_shares := amount_invested / cost_per_share

theorem number_of_shares_is_25 : number_of_shares = 25 := by
  sorry

end number_of_shares_is_25_l105_105168


namespace percent_decrease_is_36_l105_105198

-- Definitions based on the conditions
def last_week_price_per_box : ℝ := 5 / 4 -- dollars per box last week
def this_week_price_per_box : ℝ := 4 / 5 -- dollars per box this week

-- The mathematically equivalent proof problem
theorem percent_decrease_is_36 :
  ((last_week_price_per_box - this_week_price_per_box) / last_week_price_per_box) * 100 = 36 := by
  sorry

end percent_decrease_is_36_l105_105198


namespace sqrt_fourth_power_l105_105979

theorem sqrt_fourth_power (x : ℝ) (h : (√x)^4 = 256) : x = 16 := 
sorry

end sqrt_fourth_power_l105_105979


namespace trip_first_part_length_l105_105158

theorem trip_first_part_length
  (total_distance : ℝ := 50)
  (first_speed : ℝ := 66)
  (second_speed : ℝ := 33)
  (average_speed : ℝ := 44) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ total_distance) ∧ 44 = total_distance / (x / first_speed + (total_distance - x) / second_speed) ∧ x = 25 :=
by
  sorry

end trip_first_part_length_l105_105158


namespace Carly_injured_week_miles_l105_105641

def week1_miles : ℕ := 2
def week2_miles : ℕ := week1_miles * 2 + 3
def week3_miles : ℕ := week2_miles * 9 / 7
def week4_miles : ℕ := week3_miles - 5

theorem Carly_injured_week_miles : week4_miles = 4 :=
  by
    sorry

end Carly_injured_week_miles_l105_105641


namespace distinct_solution_count_l105_105306

theorem distinct_solution_count : 
  ∃! x : ℝ, |x - 3| = |x + 5| :=
begin
  sorry
end

end distinct_solution_count_l105_105306


namespace max_moves_vasya_l105_105587

-- Defining the 129-sided polygon
constant n : ℕ := 129

-- Defining the conditions for Petya and Vasya's game
structure GameState :=
  (petya_turn : bool)
  (unmarked_vertices : set ℕ)
  (last_petya_move : option ℕ)

-- Initial state of the game
def initial_state : GameState :=
{ petya_turn := true,
  unmarked_vertices := {0, 1, 2, ..., n - 1},
  last_petya_move := none }

-- Legal move function for Vasya
def vasya_legal_move (state : GameState) (move : ℕ) : Prop :=
  (∃ k, state.last_petya_move = some (k) ∧ move = (k + 2) % n) ∧ 
  move ∈ state.unmarked_vertices

-- Game ending condition
def game_ended (state : GameState) : Prop :=
  ¬(∃ move, vasya_legal_move state move)

-- Theorem to state the maximum moves Vasya can make
theorem max_moves_vasya : ∃ moves : ℕ, Vasya_max_moves 129 moves := sorry

end max_moves_vasya_l105_105587


namespace apartment_living_room_size_l105_105855

theorem apartment_living_room_size :
  (∀ (a_total r_total r_living : ℝ), a_total = 160 → r_total = 6 → (∃ r_other, r_total = 5 + 1 ∧ r_living = 3 * r_other) → r_living = 60) :=
by
  intros a_total r_total r_living a_total_eq r_total_eq h
  cases h with r_other h'
  cases h' with r_total_eq' r_living_eq
  have r_other_eq : r_other = 20 :=
    by
      exact (by linarith : r_other = 20)
  rw [r_other_eq] at r_living_eq
  exact r_living_eq

end apartment_living_room_size_l105_105855


namespace quadratic_roots_distinct_l105_105096

theorem quadratic_roots_distinct (m : ℝ) :
  let Δ := m^2 + 32 in
  Δ > 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 - 8 = 0 ∧ x2^2 + m * x2 - 8 = 0) :=
begin
  sorry
end

end quadratic_roots_distinct_l105_105096


namespace initially_calculated_average_is_correct_l105_105513

theorem initially_calculated_average_is_correct :
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  initially_avg = 22 :=
by
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  show initially_avg = 22
  sorry

end initially_calculated_average_is_correct_l105_105513


namespace find_starting_number_l105_105035

theorem find_starting_number (x : ℕ) (h1 : (50 + 250) / 2 = 150)
  (h2 : (x + 400) / 2 = 150 + 100) : x = 100 := by
  sorry

end find_starting_number_l105_105035


namespace three_digit_multiples_of_7_l105_105319

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l105_105319


namespace f4_is_even_l105_105834

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

def f (x : ℝ) : ℝ := 1 / (x^3 + 1)

def f1 (x : ℝ) : ℝ := f (x + 1)

def f2 (x : ℝ) : ℝ := f (2 * x)

def f3 (x : ℝ) : ℝ := f (x - 1)

def f4 (x : ℝ) : ℝ := f (x^2)

theorem f4_is_even : is_even_function f4 :=
sorry

end f4_is_even_l105_105834


namespace distinct_solutions_abs_eq_l105_105309

theorem distinct_solutions_abs_eq (x : ℝ) : 
  (|x - 3| = |x + 5|) → x = -1 :=
by
  sorry

end distinct_solutions_abs_eq_l105_105309


namespace central_cell_value_l105_105380

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l105_105380


namespace calculate_area_of_stripe_l105_105152

-- Define the given conditions as constants
def diameter : ℝ := 40
def height : ℝ := 100
def stripe_width : ℝ := 4
def revolutions : ℝ := 3
def slope_angle : ℝ := 30
def cos_slope_angle : ℝ := Real.cos (slope_angle * Real.pi / 180) -- Conversion to radians

-- Define the target area to prove
def expected_area : ℝ := 462

-- State the theorem with the given conditions and the expected result
theorem calculate_area_of_stripe :
  let circumference := Real.pi * diameter,
      stripe_length := revolutions * circumference,
      effective_height := height / cos_slope_angle,
      area := stripe_width * effective_height in
  Real.round area = expected_area := sorry

end calculate_area_of_stripe_l105_105152


namespace f_is_decreasing_on_interval_l105_105521

noncomputable def f (x : ℝ) : ℝ := -3 * x ^ 2 - 2

theorem f_is_decreasing_on_interval :
  ∀ x y : ℝ, (1 ≤ x ∧ x < y ∧ y ≤ 2) → f y < f x :=
by
  sorry

end f_is_decreasing_on_interval_l105_105521


namespace three_digit_multiples_of_7_l105_105329

theorem three_digit_multiples_of_7 : 
  ∃! n : ℕ, (n = 128) ∧ (∀ k, (100 ≤ 7 * k ∧ 7 * k ≤ 999) ↔ (15 ≤ k ∧ k ≤ 142)) :=
begin
  sorry
end

end three_digit_multiples_of_7_l105_105329


namespace area_of_triangle_l105_105438

noncomputable def triangle_area (A B C : Point) (AD BE : Segment) (angle_AC_AD : ℝ) : ℝ :=
  have h1 : AD.is_median A B C := sorry,
  have h2 : BE.is_median A B C := sorry,
  have h3 : AD.perpendicular_to(BE) := sorry,
  have h4 : AD.length = 18 := sorry,
  have h5 : BE.length = 24 := sorry,
  have h6 : angle_AC_AD = 30 := sorry,
  216

theorem area_of_triangle (A B C : Point) (AD BE : Segment) (angle_AC_AD : ℝ) :
  let area := triangle_area A B C AD BE angle_AC_AD in
  AD.is_median A B C ∧ BE.is_median A B C ∧ AD.perpendicular_to(BE) ∧ AD.length = 18 ∧ BE.length = 24 ∧ angle_AC_AD = 30 → area = 216 :=
by
  intros,
  simp [triangle_area],
  sorry

end area_of_triangle_l105_105438


namespace determine_f_2048_l105_105925

theorem determine_f_2048 (f : ℕ → ℝ)
  (A1 : ∀ a b n : ℕ, a > 0 → b > 0 → a * b = 2^n → f a + f b = n^2)
  : f 2048 = 121 := by
  sorry

end determine_f_2048_l105_105925


namespace shapiro_inequality_6_l105_105955

theorem shapiro_inequality_6 {a : Fin 6 → ℝ} 
  (h_pos : ∀ i, 0 < a i)
  (h_prod : (∏ i, a i) = 1) :
  (1 / (a 0 * (a 1 + 1)) + 
   1 / (a 1 * (a 2 + 1)) + 
   1 / (a 2 * (a 3 + 1)) + 
   1 / (a 3 * (a 4 + 1)) + 
   1 / (a 4 * (a 5 + 1)) + 
   1 / (a 5 * (a 0 + 1))) ≥ 3 := by
  sorry

end shapiro_inequality_6_l105_105955


namespace roy_needs_new_box_on_monday_l105_105026

-- Define the constants and conditions
def rate_morning := 1 / 6
def rate_evening := 1 / 3
def starting_cans := 8
def daily_consumption := rate_morning + rate_evening

-- Define the main theorem statement based on the conditions and the correct answer
theorem roy_needs_new_box_on_monday :
  ∑ i in range 8, (rate_morning + rate_evening) > starting_cans - daily_consumption := by
  sorry

end roy_needs_new_box_on_monday_l105_105026


namespace A_plus_B_of_multiple_of_72_l105_105614

theorem A_plus_B_of_multiple_of_72 (A B : ℕ) (h1: A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h2: B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h3: (100000 * A + 10000 * 4 + 1000 * 4 + 100 * 6 + 10 * 1 + B) % 72 = 0) : 
  A + B = 12 := 
sorry

end A_plus_B_of_multiple_of_72_l105_105614


namespace ratio_of_coconut_flavored_red_jelly_beans_l105_105947

theorem ratio_of_coconut_flavored_red_jelly_beans :
  ∀ (total_jelly_beans jelly_beans_coconut_flavored : ℕ)
    (three_fourths_red : total_jelly_beans > 0 ∧ (3/4 : ℝ) * total_jelly_beans = 3 * (total_jelly_beans / 4))
    (h1 : jelly_beans_coconut_flavored = 750)
    (h2 : total_jelly_beans = 4000),
  (250 : ℝ)/(3000 : ℝ) = 1/4 :=
by
  intros total_jelly_beans jelly_beans_coconut_flavored three_fourths_red h1 h2
  sorry

end ratio_of_coconut_flavored_red_jelly_beans_l105_105947


namespace distinct_solutions_abs_eq_l105_105308

theorem distinct_solutions_abs_eq (x : ℝ) : 
  (|x - 3| = |x + 5|) → x = -1 :=
by
  sorry

end distinct_solutions_abs_eq_l105_105308


namespace line_equation_l105_105675

def point := ℝ × ℝ

def line (a b c : ℝ) : set point := {p | a * p.1 + b * p.2 + c = 0}

def passes_through (l : set point) (P : point) : Prop :=
  P ∈ l

def perpendicular (l1 l2 : set point) : Prop :=
  ∃ (a1 b1 c1 a2 b2 c2 : ℝ), 
  (l1 = line a1 b1 c1 ∧ l2 = line a2 b2 c2) ∧ 
  (a1 * a2 + b1 * b2 = 0)

def equal_intercepts (l : set point) : Prop :=
  ∃ (a b c : ℝ), l = line a b c ∧ a = b

theorem line_equation
  (P : point) (l : set point)
  (h_pass : passes_through l P)
  (h_perp : perpendicular l (line 1 (-1) 1))
  (h_intercept : equal_intercepts l) :
  l = line 1 1 (-3) ∨ l = line 1 (-2) 0 :=
sorry

end line_equation_l105_105675


namespace greatest_b_non_in_range_of_poly_l105_105126

theorem greatest_b_non_in_range_of_poly (b : ℤ):
  (-6 ∉ range (λ x : ℝ, x^2 + (b : ℝ)*x + 15)) → b ≤ 9 :=
by
  -- The conditions and definitions for the proof
  let discriminant := b^2 - 84
  have : discriminant < 0 := sorry -- This step needs to be proved.
  exact sorry

end greatest_b_non_in_range_of_poly_l105_105126


namespace combined_weight_of_candies_l105_105236

variable Frank_weight : ℕ
variable Gwen_weight : ℕ
variable combined_weight : ℕ

theorem combined_weight_of_candies (h1 : Frank_weight = 10) (h2 : Gwen_weight = 7):
  combined_weight = Frank_weight + Gwen_weight :=
sorry

example : combined_weight_of_candies (by rfl) (by rfl) = 17 := by sorry

end combined_weight_of_candies_l105_105236


namespace multiples_of_7_are_128_l105_105320

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l105_105320


namespace min_value_l105_105274

-- Definitions
variables (a b : ℝ) (O A B C : ℝ × ℝ)
variables (λ : ℝ)
variables (hλ : λ = 1 / 2)

-- Conditions
def OA := (-2 : ℝ, 4 : ℝ)
def OB := (-a, 2 : ℝ)
def OC := (b, 0 : ℝ)

-- Collinearity condition
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, (B.1 - A.1, B.2 - A.2) = λ • (C.1 - A.1, C.2 - A.2)

axiom collinearity : collinear OA OB OC
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Objective
theorem min_value : a > 0 → b > 0 → collinear OA OB OC → (1 / a + 1 / b) = (3 + 2 * real.sqrt 2) / 2 :=
by
  sorry

end min_value_l105_105274


namespace intersection_A_B_l105_105768

set A := {(x, y) : ℝ × ℝ | x + y = 5}
set B := {(x, y) : ℝ × ℝ | x - y = 1}

theorem intersection_A_B : A ∩ B = {(3, 2)} :=
by sorry

end intersection_A_B_l105_105768


namespace find_C_and_D_l105_105520

theorem find_C_and_D (C D : ℚ) :
  (∀ x : ℚ, ((6 * x - 8) / (2 * x^2 + 5 * x - 3) = (C / (x - 1)) + (D / (2 * x + 3)))) →
  (2*x^2 + 5*x - 3 = (2*x - 1)*(x + 3)) →
  (∀ x : ℚ, ((C*(2*x + 3) + D*(x - 1)) / ((2*x - 1)*(x + 3))) = ((6*x - 8) / ((2*x - 1)*(x + 3)))) →
  (∀ x : ℚ, C*(2*x + 3) + D*(x - 1) = 6*x - 8) →
  C = -2/5 ∧ D = 34/5 := 
by 
  sorry

end find_C_and_D_l105_105520


namespace circle_area_l105_105021

open Real

-- Given points C and D
def C : Point := (-2, 3)
def D : Point := (6, 9)

-- Distance function for two points in a plane
def distance (p1 p2 : Point) : ℝ := 
  sqrt((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Diameter of the circle with endpoints C and D
def diameter : ℝ := distance C D

-- Radius is half the diameter
def radius : ℝ := diameter / 2

-- Area of the circle
def area_of_circle : ℝ := π * radius ^ 2

theorem circle_area : area_of_circle = 25 * π := by
  sorry

end circle_area_l105_105021


namespace area_condition_l105_105019

-- Definitions:
-- convex_hexagon: A type to represent convex hexagons with vertices A, B, C, D, E, F 
-- area: A function that computes the area of a given geometric shape
-- parallel: A predicate indicating that two sides are parallel

structure convex_hexagon :=
(A B C D E F : Point)

def area (h : convex_hexagon) : ℝ := sorry

def parallel (p q : Line) : Prop := sorry

noncomputable def area_ineq (h : convex_hexagon) : Prop :=
  let s_hex := area h
  let s_ace := area (convex_hexagon.mk h.A h.C h.E h.A h.C h.E)
  s_ace ≥ s_hex / 2

noncomputable def area_eq (h : convex_hexagon) : Prop :=
  let s_hex := area h
  let s_ace := area (convex_hexagon.mk h.A h.C h.E h.A h.C h.E)
  s_ace = s_hex / 2

theorem area_condition (h : convex_hexagon) (h_parallel : parallel h.AB h.DE ∧ parallel h.BC h.EF ∧ parallel h.CD h.FA) :
  area_ineq h ∧ (area_eq h ↔ (∀ a b c d e f ∈ h, a = d ∧ b = e ∧ c = f)) := sorry

end area_condition_l105_105019


namespace five_digit_palindromes_l105_105755

theorem five_digit_palindromes : 
  let first_digit := 9 in
  let second_digit := 10 in
  let third_digit := 10 in
  first_digit * second_digit * third_digit = 900 :=
by
  let first_digit := 9
  let second_digit := 10
  let third_digit := 10
  trivial

end five_digit_palindromes_l105_105755


namespace find_triplets_l105_105680

noncomputable def triplets_solution (x y z : ℝ) : Prop := 
  (x^2 + y^2 = -x + 3*y + z) ∧ 
  (y^2 + z^2 = x + 3*y - z) ∧ 
  (x^2 + z^2 = 2*x + 2*y - z) ∧ 
  (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)

theorem find_triplets : 
  { (x, y, z) : ℝ × ℝ × ℝ | triplets_solution x y z } = 
  { (0, 1, -2), (-3/2, 5/2, -1/2) } :=
sorry

end find_triplets_l105_105680


namespace three_digit_multiples_of_7_l105_105318

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l105_105318


namespace max_angle_between_tangents_l105_105709

open Real

noncomputable def asymptote (x : ℝ) : ℝ := 2 * x

noncomputable def circle_center : ℝ × ℝ := (5, 0)
noncomputable def circle_radius : ℝ := sqrt 5

noncomputable def distance_from_center_to_line : ℝ :=
  abs (2 * fst circle_center) / sqrt (1^2 + 2^2)

noncomputable def theta : ℝ :=
  asin (circle_radius / distance_from_center_to_line)

theorem max_angle_between_tangents :
  let max_angle := 2 * θ in
  θ = 30 ∧ max_angle = 60 := sorry

end max_angle_between_tangents_l105_105709


namespace stickers_total_l105_105569

theorem stickers_total (w e : ℝ) (H1 : w = 278.5) (H2 : e = 43.8) : w + e = 322.3 :=
by
  rw [H1, H2]
  norm_num
  sorry

end stickers_total_l105_105569


namespace integers_solution_l105_105064

theorem integers_solution (a b : ℤ) (S D : ℤ) 
  (h1 : S = a + b) (h2 : D = a - b) (h3 : S / D = 3) (h4 : S * D = 300) : 
  ((a = 20 ∧ b = 10) ∨ (a = -20 ∧ b = -10)) :=
by
  sorry

end integers_solution_l105_105064


namespace AM_GM_Ineq_l105_105877

theorem AM_GM_Ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by
  sorry

end AM_GM_Ineq_l105_105877


namespace properly_colored_squares_even_l105_105216

theorem properly_colored_squares_even
  (m n : ℕ)
  (vertex_color : ℕ × ℕ → ℕ)  -- represents the color of vertex at position (i, j) where 1, 2, 3 denote blue, green, red respectively
  (boundary_red : ∀ (i j : ℕ), (i = 0 ∨ i = m ∨ j = 0 ∨ j = n) → vertex_color (i, j) = 3)  -- all boundary vertices are red
  (properly_colored_square : (ℕ × ℕ) → Prop)
  (properly_colored_square_def : ∀ i j, properly_colored_square (i, j) ↔
    ∃ (c₁ c₂ c₃ c₄ : ℕ),
      vertex_color (i, j) = c₁ ∧
      vertex_color (i + 1, j) = c₂ ∧
      vertex_color (i, j + 1) = c₃ ∧
      vertex_color (i + 1, j + 1) = c₄ ∧
      (c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₁ ≠ c₄ ∧ c₂ ≠ c₃ ∧ c₂ ≠ c₄ ∧ c₃ ≠ c₄) ∧
      (vertex_color (i, j) = vertex_color (i + 1, j) ∨
       vertex_color (i, j) = vertex_color (i, j + 1) ∨
       vertex_color (i + 1, j + 1) = vertex_color (i + 1, j) ∨
       vertex_color (i + 1, j + 1) = vertex_color (i, j + 1))
    )
  : ∃ N, (finset.sum (finset.range m) (λ i, finset.sum (finset.range n) (λ j, if properly_colored_square (i, j) then 1 else 0))) = 2 * N :=
by
  sorry

end properly_colored_squares_even_l105_105216


namespace negation_of_exists_eq_sin_l105_105057

theorem negation_of_exists_eq_sin : ¬ (∃ x : ℝ, x = Real.sin x) ↔ ∀ x : ℝ, x ≠ Real.sin x :=
by
  sorry

end negation_of_exists_eq_sin_l105_105057


namespace sum_of_intersection_l105_105699

theorem sum_of_intersection (f : ℝ → ℝ) (m : ℕ)
  (hf : ∀ x : ℝ, f (-x) = 2 - f x)
  (h_intersections : fin m → ℝ × ℝ)
  (h_property : ∀ i : fin m, (h_intersections i).2 = f ((h_intersections i).1) ∧ 
                             ((h_intersections i).2 = (1 + 1 / (h_intersections i).1))) : 
  ∑ i in finset.fin_range m, (h_intersections i).1 + (h_intersections i).2 = m := 
sorry

end sum_of_intersection_l105_105699


namespace central_cell_value_l105_105370

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l105_105370


namespace central_cell_value_l105_105368

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l105_105368


namespace quadratic_same_roots_abs_l105_105444

theorem quadratic_same_roots_abs (d e : ℤ) : 
  (∀ x : ℤ, |x - 8| = 3 ↔ x = 11 ∨ x = 5) →
  (∀ x : ℤ, x^2 + d * x + e = 0 ↔ x = 11 ∨ x = 5) →
  (d, e) = (-16, 55) :=
by
  intro h₁ h₂
  have h₃ : ∀ x : ℤ, x^2 - 16 * x + 55 = 0 ↔ x = 11 ∨ x = 5 := sorry
  sorry

end quadratic_same_roots_abs_l105_105444


namespace number_of_polynomials_satisfying_P_neg1_eq_neg12_l105_105648

noncomputable def count_polynomials_satisfying_condition : ℕ := 
  sorry

theorem number_of_polynomials_satisfying_P_neg1_eq_neg12 :
  count_polynomials_satisfying_condition = 455 := 
  sorry

end number_of_polynomials_satisfying_P_neg1_eq_neg12_l105_105648


namespace sum_of_four_triangles_l105_105617

theorem sum_of_four_triangles (x y : ℝ) (h1 : 3 * x + 2 * y = 27) (h2 : 2 * x + 3 * y = 23) : 4 * y = 12 :=
sorry

end sum_of_four_triangles_l105_105617


namespace pieces_to_cut_l105_105313

-- Define the conditions
def rodLength : ℝ := 42.5  -- Length of the rod
def pieceLength : ℝ := 0.85  -- Length of each piece

-- Define the theorem that needs to be proven
theorem pieces_to_cut (h1 : rodLength = 42.5) (h2 : pieceLength = 0.85) : 
  (rodLength / pieceLength) = 50 := 
  by sorry

end pieces_to_cut_l105_105313


namespace gender_related_interest_expectation_value_l105_105034

/-
Given the contingency table:
Male Interested: 240, Male Less Interested: 160, Female Interested: 150, Female Less Interested: 50,
Total Interested: 390, Total Less Interested: 210, Total: 600.
-/

def contingency_table := {
  male_interested : ℕ := 240,
  male_less_interested : ℕ := 160,
  female_interested : ℕ := 150,
  female_less_interested : ℕ := 50,
  total_interested : ℕ := 390,
  total_less_interested : ℕ := 210,
  total : ℕ := 600
}

 /-
 Calculate the K^2 statistic.
 -/

def K_squared (n a b c d : ℕ) :=
  (n * (a * d - b * c)^2) / (a + b) / (c + d) / (a + c) / (b + d)

noncomputable def k_value :=
  K_squared 
    contingency_table.total 
    contingency_table.male_interested 
    contingency_table.male_less_interested 
    contingency_table.female_interested 
    contingency_table.female_less_interested

/-
 Prove that gender is related to interest in new energy vehicles.
 -/
theorem gender_related_interest : k_value > 6.635 := sorry

/-
 Given male-to-female ratio is 2:1, from 6 selected individuals (2 females, 4 males), let X be the number of females among 3 individuals selected.
 Find the distribution and expectation of X.
 -/
def male_female_ratio (total females males : ℕ) := 
  females = 1/3 * total ∧ 
  males = 2/3 * total

variable (total_individuals: ℕ := 6)
variable (selected_individuals: ℕ := 3)

def probs (females males selected : ℕ) :=
  (P (X = 0) := 1/5) ∧ 
  (P (X = 1) := 3/5) ∧ 
  (P (X = 2) := 1/5)

def expectation_X (p0 p1 p2 : ℕ) :=
  (0 * p0 + 1 * p1 + 2 * p2 : real)

noncomputable def E_X :=
  expectation_X 1/5 3/5 1/5

theorem expectation_value : E_X = 1 := sorry

end gender_related_interest_expectation_value_l105_105034


namespace friends_pets_ratio_l105_105508

theorem friends_pets_ratio (pets_total : ℕ) (pets_taylor : ℕ) (pets_friend4 : ℕ) (pets_friend5 : ℕ)
  (pets_first3_total : ℕ) : pets_total = 32 → pets_taylor = 4 → pets_friend4 = 2 → pets_friend5 = 2 →
  pets_first3_total = pets_total - pets_taylor - pets_friend4 - pets_friend5 →
  (pets_first3_total : ℚ) / pets_taylor = 6 :=
by
  sorry

end friends_pets_ratio_l105_105508


namespace distance_between_two_cars_l105_105991

theorem distance_between_two_cars 
    (initial_distance : ℝ) 
    (first_car_distance1 : ℝ) 
    (first_car_distance2 : ℝ)
    (second_car_distance : ℝ) 
    (final_distance : ℝ) :
    initial_distance = 150 →
    first_car_distance1 = 25 →
    first_car_distance2 = 25 →
    second_car_distance = 35 →
    final_distance = initial_distance - (first_car_distance1 + first_car_distance2 + second_car_distance) →
    final_distance = 65 :=
by
  intros h_initial h_first1 h_first2 h_second h_final
  sorry

end distance_between_two_cars_l105_105991


namespace cos_sin_values_l105_105353

theorem cos_sin_values (θ : ℝ) (h : ∑' n : ℕ, (cos θ)^(2 * n) = 9) :
  cos (2 * θ) = 7 / 9 ∧ sin θ^2 = 1 / 9 :=
by
  sorry

end cos_sin_values_l105_105353


namespace at_least_one_divisible_by_5_l105_105015

theorem at_least_one_divisible_by_5 (k m n : ℕ) (hk : ¬ (5 ∣ k)) (hm : ¬ (5 ∣ m)) (hn : ¬ (5 ∣ n)) : 
  (5 ∣ (k^2 - m^2)) ∨ (5 ∣ (m^2 - n^2)) ∨ (5 ∣ (n^2 - k^2)) :=
by {
    sorry
}

end at_least_one_divisible_by_5_l105_105015


namespace tofu_largest_square_side_length_l105_105570

open Nat

theorem tofu_largest_square_side_length (L W n : ℕ) (hL : L = 54) (hW : W = 20) (hn : n = 3) :
  (L / n ≤ W) → n * (L / n) = L :=
by
  subst hL
  subst hW
  subst hn
  intro h
  simp [Nat.le_div_iff_mul_le (by norm_num : 0 < 3)]
  sorry

end tofu_largest_square_side_length_l105_105570


namespace probability_of_transformed_z_in_S_l105_105173

-- Define the region S
def region_S (z : ℂ) : Prop := -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1

-- Define the transformation
def transformation (z : ℂ) : ℂ := (3/4 + 3/4*I) * z

-- The theorem we want to prove
theorem probability_of_transformed_z_in_S :
  let S_area := 4
  let T_area := (2 * real.sqrt 2 / 3)^2
  T_area / S_area = 7 / 9 :=
by
  let S_area := 4
  let T_area := (2 * real.sqrt 2 / 3)^2
  calc
  T_area / S_area = (8 / 9) / 4 : by sorry
                ... = 7 / 9 : by sorry

end probability_of_transformed_z_in_S_l105_105173


namespace central_cell_value_l105_105385

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l105_105385


namespace max_area_triangle_AOB_exists_l105_105288

open Real

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop := (x^2 / (a^2)) + (y^2 / (b^2)) = 1

def companion_circle_equation (a b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2 + b^2

def is_perpendicular (l : ℝ → ℝ) (axis : ℝ → ℝ) : Prop := ∃ m : ℝ, l m = axis m

theorem max_area_triangle_AOB_exists 
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (minor_axis_eq : 2 * b = 2)
  (eccentricity_eq : (sqrt 6 / 3) = (sqrt (1 - (b^2 / a^2))))
  (companion_eq : ∀ (x y : ℝ), companion_circle_equation a b x y)
  (line_l : ℝ → ℝ)
  (points_A_B_C_D_on_curves : 
    ∀ (x y : ℝ), 
      (ellipse_equation a b x y ∧ ∀ yy : ℝ, line_l x = yy) ∧ 
      (companion_circle_equation a b x y ∧ ∀ yy : ℝ, line_l x = yy))
  (CD_eq_len_sqrt13 : ∀ (x y : ℝ), companion_circle_equation a b x y → ∃ (C D : ℝ × ℝ), |C.1 - D.1| = sqrt 13)
  : ∃ A B : ℝ × ℝ, 
    ellipse_equation a b A.1 A.2 ∧ 
    ellipse_equation a b B.1 B.2 ∧
    (let O := (0, 0 : ℝ × ℝ) in 
      max_area_of_triangle O A B = sqrt 3) :=
sorry

end max_area_triangle_AOB_exists_l105_105288


namespace sphere_surface_area_l105_105284

theorem sphere_surface_area (C : ℝ) (hC : C = 2 * Real.pi) : ∃ A, A = 4 * Real.pi := 
by 
  -- Since C = 2π, we find the radius R
  have R : ℝ := 1 by sorry
  -- Then we calculate the surface area A = 4πR²
  have A : ℝ := 4 * Real.pi * (R ^ 2) by sorry
  -- Prove the calculated A is 4π
  use A
  exact sorry

end sphere_surface_area_l105_105284


namespace problem1_problem2_problem3_l105_105721

-- Define the problem conditions
structure Square (V : Type) [EuclideanSpace V] :=
(ABCD: Quadrilateral V)
(base_square: IsSquare ABCD)
(PA_perp_base: ∀u : V, PA ⊥ u)
(PA_eq_AD: PA = (AD : ℝ))
(points_on_edges: ∃M N : V, M ∈ segment (PD) ∧ N ∈ segment (PC))
(PC_perp_AMN: plane_perp PC (@hyperplane AMN))

-- Problem 1: Proving AM ⊥ PD
theorem problem1 (V : Type) [EuclideanSpace V] (s : Square V) : AM ⊥ PD :=
sorry

-- Problem 2: Finding the dihedral angle
theorem problem2 (V : Type) [EuclideanSpace V] (s : Square V) : 
  dihedralAngle (PAM) (NAM) = real.arccos (Real.sqrt 3 / 3) :=
sorry

-- Problem 3: Finding the angle between line CD and plane AMN
theorem problem3 (V : Type) [EuclideanSpace V] (s : Square V) : 
  ∠(CD, plane AMN) = real.arcsin (Real.sqrt 3 / 3) :=
sorry

end problem1_problem2_problem3_l105_105721


namespace triangle_median_length_l105_105437

theorem triangle_median_length (XY XZ XM YZ : ℝ) 
  (hXY : XY = 5) 
  (hXZ : XZ = 8) 
  (hXM : XM = 4) 
  (hMedian : XM = 1/2 * sqrt (2*XY^2 + 2*XZ^2 - YZ^2)) :
  YZ = sqrt 114 := 
by 
  sorry

end triangle_median_length_l105_105437


namespace log_range_implies_a_gt_1_l105_105289

theorem log_range_implies_a_gt_1 (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, log a (x^2 + 2 * a * x + 1) = y) ↔ (1 < a) :=
by
  sorry

end log_range_implies_a_gt_1_l105_105289


namespace residue_class_equivalence_l105_105022

variable {a m : ℤ}
variable {b : ℤ}

def residue_class (a m b : ℤ) : Prop := ∃ t : ℤ, b = m * t + a

theorem residue_class_equivalence (m a b : ℤ) :
  (∃ t : ℤ, b = m * t + a) ↔ b % m = a % m :=
by sorry

end residue_class_equivalence_l105_105022


namespace quadratic_roots_distinct_l105_105095

theorem quadratic_roots_distinct (m : ℝ) :
  let Δ := m^2 + 32 in
  Δ > 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 - 8 = 0 ∧ x2^2 + m * x2 - 8 = 0) :=
begin
  sorry
end

end quadratic_roots_distinct_l105_105095


namespace angle_between_asymptotes_of_hyperbola_l105_105294

theorem angle_between_asymptotes_of_hyperbola (m : ℝ) :
  (∃ (a b : ℝ), a = sqrt (m + 4) ∧ b = sqrt m ∧ (sqrt (1 + b^2 / a^2)) = (2 * sqrt 3 / 3)) →
  ∃ θ, θ = π / 3 :=
by
  sorry

end angle_between_asymptotes_of_hyperbola_l105_105294


namespace tiling_rectangles_single_cut_l105_105807

theorem tiling_rectangles_single_cut : ∃ (tiling : set (set (ℝ × ℝ))) (cut : ℝ × ℝ → Prop),
  (∀ (rect : set (ℝ × ℝ)), rect ∈ tiling → (∃ (a b c : ℝ), ∀ (x y : ℝ), (x, y) ∈ rect ↔ (a ≤ x ∧ x ≤ b ∧ c ≤ y ∧ y ≤ d))) ∧
  (∀ (rect : set (ℝ × ℝ)), rect ∈ tiling → (∀ (x : ℝ × ℝ), cut x → x ∈ rect)) :=
sorry

end tiling_rectangles_single_cut_l105_105807


namespace ravon_has_card_10_l105_105785

structure Player :=
(name : String)
(cards : Finset ℕ)
(score : ℕ)

def players : List Player :=
[
  {name := "Ravon", cards := {x | 1 ≤ x ∧ x ≤ 15}, score := 19},
  {name := "Oscar", cards := {x | 1 ≤ x ∧ x ≤ 15}, score := 9},
  {name := "Aditi", cards := {x | 1 ≤ x ∧ x ≤ 15}, score := 13},
  {name := "Tyrone", cards := {x | 1 ≤ x ∧ x ≤ 15}, score := 21},
  {name := "Kim", cards := {x | 1 ≤ x ∧ x ≤ 15}, score := 14},
  {name := "Leah", cards := {x | 1 ≤ x ∧ x ≤ 15}, score := 22}
]

def card_for_player (player_name : String) (card : ℕ) : Prop :=
  ∃ p ∈ players, p.name = player_name ∧ card ∈ p.cards

theorem ravon_has_card_10 : card_for_player "Ravon" 10 :=
sorry

end ravon_has_card_10_l105_105785


namespace knights_probability_sum_l105_105109

theorem knights_probability_sum (P : ℚ) (num den : ℕ) 
  (hP : P = 53 / 85) 
  (h_frac : P = num / den) 
  (h_gcd : Nat.gcd num den = 1) : 
  num + den = 138 := 
by {
  -- The proof steps would determine the conditions were met but we use sorry here
  sorry
}

end knights_probability_sum_l105_105109


namespace central_cell_value_l105_105374

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l105_105374


namespace tim_has_156_golf_balls_l105_105112

theorem tim_has_156_golf_balls (dozen: ℕ) (balls_per_dozen: ℕ) (tim_dozen: ℕ) (h1: dozen = 12) (h2: tim_dozen = 13) : 
  tim_dozen * dozen = 156 :=
by
  rw [h1, h2]
  exact Nat.mul_comm 156 1 -- given format to complete the multiplicative proof.

-- added sorry as a placeholder for simplicity.

end tim_has_156_golf_balls_l105_105112


namespace total_edge_length_of_polyhedron_l105_105866

theorem total_edge_length_of_polyhedron (pyramid_edges tetrahedron_edges edges_lost_due_to_gluing edges_lost_due_to_merging edge_length : ℕ) :
  pyramid_edges = 8 → tetrahedron_edges = 6 → edges_lost_due_to_gluing = 3 → edges_lost_due_to_merging = 2 → edge_length = 2 →
  (pyramid_edges + tetrahedron_edges - edges_lost_due_to_gluing - edges_lost_due_to_merging) * edge_length = 18 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end total_edge_length_of_polyhedron_l105_105866


namespace number_of_jars_good_for_sale_l105_105003

def numberOfGoodJars (initialCartons : Nat) (cartonsNotDelivered : Nat) (jarsPerCarton : Nat)
  (damagedJarsPerCarton : Nat) (numberOfDamagedCartons : Nat) (oneTotallyDamagedCarton : Nat) : Nat := 
  let deliveredCartons := initialCartons - cartonsNotDelivered
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := (damagedJarsPerCarton * numberOfDamagedCartons) + oneTotallyDamagedCarton
  totalJars - damagedJars

theorem number_of_jars_good_for_sale : 
  numberOfGoodJars 50 20 20 3 5 20 = 565 :=
by
  sorry

end number_of_jars_good_for_sale_l105_105003


namespace count_comic_book_orders_l105_105860

theorem count_comic_book_orders
  (E Spiderman Archie Garfield Batman Superman: Type)
  (spiderman_comics : [6]E)
  (archie_comics : [5]E)
  (garfield_comics : [4]E)
  (batman_comics : [3]E)
  (superman_comics : [2]E) 
  : 6.factorial * 5.factorial * 4.factorial * 3.factorial * 2.factorial * 5.factorial = 1_492_992_000 := by
  sorry

end count_comic_book_orders_l105_105860


namespace find_x_l105_105996

-- Define the condition from the problem statement
def condition1 (x : ℝ) : Prop := 70 = 0.60 * x + 22

-- Translate the question to the Lean statement form
theorem find_x (x : ℝ) (h : condition1 x) : x = 80 :=
by {
  sorry
}

end find_x_l105_105996


namespace constant_term_binomial_expansion_l105_105672

theorem constant_term_binomial_expansion :
  (constant_term (\(x^2 + 1/(2*x))^6) = 15/16 : ℚ) :=
sorry

end constant_term_binomial_expansion_l105_105672


namespace sum_of_numerator_denominator_l105_105219

noncomputable def repeating_decimal_to_fraction : ℚ := 115 / 111

theorem sum_of_numerator_denominator : 
  let f := repeating_decimal_to_fraction in 
  let num := f.num in 
  let denom := f.denom in 
  num + denom = 226 :=
by
  let f : ℚ := 115 / 111 
  let num := f.num 
  let denom := f.denom 
  have h₁ : num = 115 := by sorry
  have h₂ : denom = 111 := by sorry
  rw [h₁, h₂]
  exact rfl

end sum_of_numerator_denominator_l105_105219


namespace ratio_of_tangency_segments_l105_105589

theorem ratio_of_tangency_segments (A B C : Type) [LinearOrderedField A]
  (T : Triangle A)
  (hT₁ : T.side1 = 9)
  (hT₂ : T.side2 = 15)
  (hT₃ : T.side3 = 18)
  (r s : A)
  (r_lt_s : r < s)
  (tangency_condition : T.point_of_tangency_seg = r + s = 9) :
  r / s = 1 / 2 := sorry

end ratio_of_tangency_segments_l105_105589


namespace cuboid_ratio_l105_105051

theorem cuboid_ratio (length breadth height: ℕ) (h_length: length = 90) (h_breadth: breadth = 75) (h_height: height = 60) : 
(length / Nat.gcd length (Nat.gcd breadth height) = 6) ∧ 
(breadth / Nat.gcd length (Nat.gcd breadth height) = 5) ∧ 
(height / Nat.gcd length (Nat.gcd breadth height) = 4) := by 
  -- intentionally skipped proof 
  sorry

end cuboid_ratio_l105_105051


namespace central_cell_value_l105_105373

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l105_105373


namespace three_digit_multiples_of_7_l105_105339

theorem three_digit_multiples_of_7 :
  ∃ n : ℕ, (n = ∑ k in finset.range (143 - 15), ∀ k ∈ finset.range (143 - 15), 100 ≤ 7 * (15 + k) ∧ 7 * (15 + k) ≤ 999) :=
sorry

end three_digit_multiples_of_7_l105_105339


namespace central_cell_value_l105_105399

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l105_105399


namespace solve_system_l105_105560

theorem solve_system :
  ∃ (x y : ℤ), 15 * x + 25 * y = 35 ∧ 25 * x + 15 * y = 85 ∧ x = 4 ∧ y = -1 :=
by
  use 4, -1
  split
  {
    exact sorry
  }
  split
  {
    exact sorry
  }
  split
  {
    refl
  }
  {
    refl
  }

end solve_system_l105_105560


namespace central_cell_value_l105_105379

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l105_105379


namespace multiples_of_7_are_128_l105_105321

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l105_105321


namespace deanna_wins_l105_105616

theorem deanna_wins (A B C D : ℕ) (total_games : ℕ) (total_wins : ℕ) (A_wins : A = 5) (B_wins : B = 2)
  (C_wins : C = 1) (total_games_def : total_games = 6) (total_wins_def : total_wins = 12)
  (total_wins_eq : A + B + C + D = total_wins) : D = 4 :=
by
  sorry

end deanna_wins_l105_105616


namespace fg_of_2_eq_0_l105_105352

def f (x : ℝ) : ℝ := 4 - x^2
def g (x : ℝ) : ℝ := 3 * x - x^3

theorem fg_of_2_eq_0 : f (g 2) = 0 := by
  sorry

end fg_of_2_eq_0_l105_105352


namespace range_of_b_l105_105715

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 1 < x ∧ x < e → f' x ≥ 0) → b ≤ 1 := 
by
  let f (x : ℝ) := x + b / x
  let f' (x : ℝ) := 1 - b / x^2
  intro h
  have h1 : ∀ x : ℝ, 1 < x ∧ x < e → 1 - b / x^2 ≥ 0 := by {
    assume x hx,
    exact h x hx
  }
  have h2 : ∀ x : ℝ, 1 < x ∧ x < e → b / x^2 ≤ 1 := by {
    assume x hx,
    linarith
  }
  have h3 : ∀ x : ℝ, 1 < x ∧ x < e → b ≤ x^2 := by {
    assume x hx,
    apply le_of_mul_le_mul_right (h2 x hx) (by linarith)
  }
  have h4 : ∀ x : ℝ, 1 < x ∧ x < e → b ≤ 1 := by {
    assume x hx,
    have hx2 : 1 < x^2 ∧ x^2 < e^2 := by {
      split,
      exact sq_pos_of_pos hx.left,
      exact sq_lt_sq hx.right
    }
    exact (h3 x hx).trans_le (by linarith)
  }
  apply le_of_forall (λ x, h4 x (by linarith))
  exact sorry

end range_of_b_l105_105715


namespace sqrt_sum_lt_four_l105_105061

theorem sqrt_sum_lt_four (x y z t : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1)
  (hz : 0 < z ∧ z < 1) (ht : 0 < t ∧ t < 1) :
  sqrt(x^2 + (1 - t)^2) + sqrt(y^2 + (1 - x)^2) + sqrt(z^2 + (1 - y)^2) + sqrt(t^2 + (1 - z)^2) < 4 :=
by
  sorry

end sqrt_sum_lt_four_l105_105061


namespace fourteen_root_of_unity_l105_105660

theorem fourteen_root_of_unity (n : ℕ) (hn : n < 14) :
  (∃ k : ℤ, (tan (π / 7) + complex.I) / (tan (π / 7) - complex.I) =
            complex.exp (complex.I * ↑(2 * k * π / 14)) ∧
            (0 ≤ n ∧ n ≤ 13)) :=
by
  use 4
  sorry

end fourteen_root_of_unity_l105_105660


namespace integer_part_sqrt_sum_l105_105253

theorem integer_part_sqrt_sum (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_eq : x * y + y * z + z * x = 1) :
  (⌊sqrt (3 * x * y + 1) + sqrt (3 * y * z + 1) + sqrt (3 * z * x + 1)⌋ = 4) := sorry

end integer_part_sqrt_sum_l105_105253


namespace concurrency_if_collinear_l105_105622

theorem concurrency_if_collinear 
  (A B C D E F F₀ D₀ E₀ : Type*)
  [geometric_structure A B C D E F D₀ E₀ F₀] :
  points_on_line D B C → points_on_line E C A → points_on_line F A B → 
  inter (line D E) (line A B) = F₀ →
  inter (line E F) (line B C) = D₀ →
  inter (line F D) (line C A) = E₀ →
  collinear D₀ E₀ F₀ → 
  concurrent (line A D) (line B E) (line C F) :=
by
  sorry

end concurrency_if_collinear_l105_105622


namespace remove_red_balls_l105_105997

theorem remove_red_balls (total_balls : ℕ) 
  (initial_red_percent : ℚ) (desired_red_percent : ℚ) (red_balls_removed : ℕ) :
  total_balls = 800 ∧ 
  initial_red_percent = 0.7 ∧
  desired_red_percent = 0.6 ∧
  red_balls_removed = 200 →
  let initial_red_balls := initial_red_percent * total_balls,
      remaining_balls := total_balls - red_balls_removed,
      remaining_red_balls := initial_red_balls - red_balls_removed,
      actual_red_percent := remaining_red_balls / remaining_balls in
  actual_red_percent = desired_red_percent :=
by
  intros h
  sorry

end remove_red_balls_l105_105997


namespace red_toys_removed_l105_105945

theorem red_toys_removed (R W : ℕ) (h1 : R + W = 134) (h2 : 2 * W = 88) (h3 : R - 2 * W / 2 = 88) : R - 88 = 2 :=
by {
  sorry
}

end red_toys_removed_l105_105945


namespace reflect_A_across_y_axis_l105_105516

structure Point where
  x : ℝ
  y : ℝ

def reflect_across_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem reflect_A_across_y_axis :
  reflect_across_y_axis ⟨2, -1⟩ = ⟨-2, -1⟩ :=
by
  simp [reflect_across_y_axis]
  sorry

end reflect_A_across_y_axis_l105_105516


namespace simplify_fraction_l105_105492

theorem simplify_fraction : (12.factorial : ℚ) / (10.factorial + 3 * 9.factorial) = 1320 / 13 := by
  sorry

end simplify_fraction_l105_105492


namespace parabola_focus_distance_l105_105723

-- Given condition: vertex of the parabola is at the center of the ellipse
def ellipse_center : Prop := (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) → (0, 0) = (0, 0))

-- Given condition: the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4 + y^2 / 3 = 1)

-- Given condition: the parabola intersects the ellipse at point P in the first quadrant
def parabola_eq (x y : ℝ) : Prop := (y^2 = 4 * x)

-- Point P in the first quadrant
def point_P (x y : ℝ) : Prop := x = 2 / 3 ∧ y = 2 * real.sqrt 6 / 3

-- Definition of the right focus F2 of the ellipse
def right_focus (x : ℝ) : Prop := x = 1

-- Function for calculating the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Proof problem to be stated in Lean
theorem parabola_focus_distance :
  (∀ x y : ℝ, ellipse_eq x y → parabola_eq x y → point_P x y) →
  (∀ x y : ℝ, point_P x y → right_focus 1 → distance x y 1 0 = 5 / 3) :=
by { intros, sorry }

end parabola_focus_distance_l105_105723


namespace pizza_eating_group_l105_105595

theorem pizza_eating_group (x y : ℕ) (h1 : 6 * x + 2 * y ≥ 49) (h2 : 7 * x + 3 * y ≤ 59) : x = 8 ∧ y = 2 := by
  sorry

end pizza_eating_group_l105_105595


namespace BP_bisects_AH_l105_105805

open EuclideanGeometry

section TriangleBisect

variables {A B C M H P : Point}

-- Assume A, B, C are points forming a right triangle with the right angle at A
axiom right_triangle (A B C : Point) : angle B A C = 90

-- M is the midpoint of BC
axiom midpoint_BC (B C : Point) (M : Point) : midpoint B C M

-- AH is the altitude from A to BC
axiom altitude_AH (A B C H : Point) : line_perp (line_through A H) (line_through B C) ∧ collinear [A, H, B, C]

-- Given a line through M perpendicular to AC intersects the circumcircle of triangle AMC at P for the second time
axiom line_perpendicular (M A C P : Point) (circumcircle : Circle) :
  line_perp (line_through M P) (line_through A C) ∧ on_circle M circumcircle ∧ on_circle A circumcircle ∧ on_circle C circumcircle ∧ on_circle P circumcircle

-- Prove that BP bisects segment AH
theorem BP_bisects_AH : bisects (line_through B P) (segment A H) :=
sorry

end TriangleBisect

end BP_bisects_AH_l105_105805


namespace zoo_camels_l105_105939

theorem zoo_camels (x y : ℕ) (h1 : x - y = 10) (h2 : x + 2 * y = 55) : x + y = 40 :=
by sorry

end zoo_camels_l105_105939


namespace probability_not_red_light_l105_105624

theorem probability_not_red_light :
  ∀ (red_light yellow_light green_light : ℕ),
    red_light = 30 →
    yellow_light = 5 →
    green_light = 40 →
    (yellow_light + green_light) / (red_light + yellow_light + green_light) = (3 : ℚ) / 5 :=
by intros red_light yellow_light green_light h_red h_yellow h_green
   sorry

end probability_not_red_light_l105_105624


namespace area_of_given_rectangle_l105_105577

/-- The coordinates of the vertices of the rectangle -/
def vertices := [(-9, 1), (1, 1), (1, -8), (-9, -8)]

/-- The function to calculate the distance between two points in 1D space -/
def distance (a b : Int) : Int := Int.abs (a - b)

/-- The function to calculate the area of the rectangle given the vertices -/
def rectangle_area (vs : List (Int × Int)) : Int :=
  match vs with 
  | [(x1, y1), (x2, y2), _, _] => 
      let length := distance x1 x2
      let width := distance y1 (snd (vs.get! 2))
      length * width
  | _ => 0

/-- Theorem: The area of the rectangle defined by the given vertices is 90 square units -/
theorem area_of_given_rectangle : rectangle_area vertices = 90 := by
  sorry

end area_of_given_rectangle_l105_105577


namespace quadratic_has_two_distinct_real_roots_l105_105091

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end quadratic_has_two_distinct_real_roots_l105_105091


namespace divisible_by_12_l105_105479

theorem divisible_by_12 (a b c d : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (hpos_d : 0 < d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) := 
by
  sorry

end divisible_by_12_l105_105479


namespace quadrilateral_circle_area_ratio_l105_105025

theorem quadrilateral_circle_area_ratio (p q r : ℕ) (s : ℝ)
  (h_radius : s > 0)
  (h_angle_FEG : ∠FEG = 40)
  (h_angle_EFG : ∠EFG = 50)
  (h_quadrilateral_inscribed : EFGH : quadrilateral)
  (h_EG_diameter : is_diameter E G)
  (h_angles_right : ∠F = 90 ∧ ∠H = 90)
  (h_special_triangle : ∀ t : triangle, (t = EFG ∨ t = EGH) → (is_50_40_90 t)) :
  (A_EFGH / A_circle) = (2 / π) → (p = 2) ∧ (q = 0) ∧ (r = 1) → p + q + r = 3 := by
  sorry

end quadrilateral_circle_area_ratio_l105_105025


namespace total_earnings_correct_l105_105486

section earnings

variable (S : ℝ) (Sa : ℝ) (M : ℝ) (P : ℝ)

-- Given conditions
def salvadore_earnings := S = 1956
def santo_earnings := Sa = S / 2
def maria_earnings := M = 3 * Sa
def pedro_earnings := P = Sa + M

def salvadore_pay_after_taxes := S * 0.8
def santo_pay_after_taxes := Sa * 0.85
def maria_pay_after_taxes := M * 0.9
def pedro_pay_after_taxes := P * 0.75

-- Total earnings after taxes
def total_earnings_after_taxes := salvadore_pay_after_taxes + santo_pay_after_taxes + maria_pay_after_taxes + pedro_pay_after_taxes

-- Problem Statement in Lean
theorem total_earnings_correct : 
  salvadore_earnings S → 
  santo_earnings S Sa → 
  maria_earnings Sa M → 
  pedro_earnings Sa M P → 
  total_earnings_after_taxes S Sa M P = 7971.70 := by
  sorry

end earnings

end total_earnings_correct_l105_105486


namespace three_digit_multiples_of_7_l105_105337

theorem three_digit_multiples_of_7 :
  ∃ n : ℕ, (n = ∑ k in finset.range (143 - 15), ∀ k ∈ finset.range (143 - 15), 100 ≤ 7 * (15 + k) ∧ 7 * (15 + k) ≤ 999) :=
sorry

end three_digit_multiples_of_7_l105_105337


namespace min_value_f_l105_105635

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem min_value_f (h : ∀ x > 10, f x ≥ 40) : ∀ x > 10, f x = 40 → x = 20 :=
by
  sorry

end min_value_f_l105_105635


namespace decagon_area_l105_105591

theorem decagon_area 
    (perimeter_square : ℝ) 
    (side_division : ℕ) 
    (side_length : ℝ) 
    (triangle_area : ℝ) 
    (total_triangle_area : ℝ) 
    (square_area : ℝ)
    (decagon_area : ℝ) :
    perimeter_square = 150 →
    side_division = 5 →
    side_length = perimeter_square / 4 →
    triangle_area = 1 / 2 * (side_length / side_division) * (side_length / side_division) →
    total_triangle_area = 8 * triangle_area →
    square_area = side_length * side_length →
    decagon_area = square_area - total_triangle_area →
    decagon_area = 1181.25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end decagon_area_l105_105591


namespace no_carrying_pairs_count_l105_105685

theorem no_carrying_pairs_count :
  ∃ N : ℕ, N = {pair | let (x, y) := pair in 1000 ≤ x ∧ x < y ∧ y ≤ 3000 ∧ no_carrying(x, y) ∧ leading_two_digits_sum_leq_9(x, y)}.card := 
begin
  -- Define the helper functions and properties used in the condition
  def no_carrying (x y : ℕ) : Prop :=
    ∀ d in [0..(digit_count x)], get_digit x d + get_digit y d < 10

  def leading_two_digits_sum_leq_9 (x y : ℕ) : Prop :=
    let (x1, x2) := (get_digit x (digit_count x - 1), get_digit x (digit_count x - 2)) in
    let (y1, y2) := (get_digit y (digit_count y - 1), get_digit y (digit_count y - 2)) in
    (x1 + x2 ≤ 9) ∧ (y1 + y2 ≤ 9)

  sorry -- Placeholder for the proof
end

end no_carrying_pairs_count_l105_105685


namespace reflection_circumcircle_of_isosceles_triangle_l105_105425

theorem reflection_circumcircle_of_isosceles_triangle (A B C P Q R P' : Point) 
(h_iso : isosceles A B C) 
(hP_on_BC : lies_on_line P B C)
(hR_parallel_AB : parallel_through_point R P A B)
(hQ_parallel_AC : parallel_through_point Q P A C)
(hP'_reflection : reflection_over_line P' P R Q) :
lies_on_circumcircle P' A B C :=
sorry

end reflection_circumcircle_of_isosceles_triangle_l105_105425


namespace speed_difference_l105_105858

theorem speed_difference
  (distance_park : ℝ)
  (maya_jog_distance : ℝ)
  (maya_jog_speed : ℝ)
  (maya_walk_speed : ℝ)
  (naomi_total_distance : ℝ)
  (naomi_travel_time_minutes : ℝ)
  (correct_answer : ℝ)
  (maya_walk_distance : maya_walk_distance = distance_park - maya_jog_distance)
  (naomi_travel_time_hrs : naomi_travel_time_hrs = naomi_travel_time_minutes / 60)
  (naomi_speed : naomi_speed = naomi_total_distance / naomi_travel_time_hrs)
  (maya_jog_time : maya_jog_time = maya_jog_distance / maya_jog_speed)
  (maya_walk_time : maya_walk_time = maya_walk_distance / maya_walk_speed)
  (maya_total_time : maya_total_time = maya_jog_time + maya_walk_time)
  (maya_avg_speed : maya_avg_speed = distance_park / maya_total_time)
  (speed_diff : speed_diff = naomi_speed - maya_avg_speed)
  (proof : speed_diff ≈ correct_answer) : Prop :=
correct_answer = 28

end speed_difference_l105_105858


namespace problem_statement_l105_105645

open Real

theorem problem_statement (n : ℕ) (h1 : n > 2) :
  log10 (↑((n - 2).fact)) + log10 (↑((n - 1).fact)) + 1 = 2 * log10 (↑(n.fact)) → n = 5 :=
by
  sorry

end problem_statement_l105_105645


namespace central_cell_value_l105_105382

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l105_105382


namespace number_of_4_digit_numbers_l105_105305

/-- Prove the number of 4-digit numbers where:
- the first digit is a prime number below 50,
- the second digit is a perfect square,
- the third digit is a Fibonacci number,
- the fourth digit is a multiple of 7,
- the digit 2 can only be used once,
is equal to 216.
-/

def primes_below_50 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
def single_digit_primes : List ℕ := [2, 3, 5, 7]
def perfect_squares : List ℕ := [0, 1, 4, 9]
def fibonacci_numbers : List ℕ := [0, 1, 1, 2, 3, 5, 8]
def multiples_of_7 : List ℕ := [0, 7]

theorem number_of_4_digit_numbers : 
    (∃ first_digit ∈ single_digit_primes, 
     ∃ second_digit ∈ perfect_squares, 
     ∃ third_digit ∈ fibonacci_numbers, 
     ∃ fourth_digit ∈ multiples_of_7, 
     first_digit ≠ 2 ∨ third_digit ≠ 2) ∧ 
    (first_digit = 2 ∨ first_digit ≠ 2) → 
    216 :=
by
  sorry

end number_of_4_digit_numbers_l105_105305


namespace cannot_be_the_lengths_l105_105983

theorem cannot_be_the_lengths (x y z : ℝ) (h1 : x^2 + y^2 = 16) (h2 : x^2 + z^2 = 25) (h3 : y^2 + z^2 = 49) : false :=
by
  sorry

end cannot_be_the_lengths_l105_105983


namespace salt_fraction_l105_105931

variables {a x : ℝ}

-- First condition: the shortfall in salt the first time
def shortfall_first (a x : ℝ) : ℝ := a - x

-- Second condition: the shortfall in salt the second time
def shortfall_second (a x : ℝ) : ℝ := a - 2 * x

-- Third condition: relationship given by the problem
axiom condition : shortfall_first a x = 2 * shortfall_second a x

-- Prove fraction of necessary salt added the first time is 1/3
theorem salt_fraction (a x : ℝ) (h : shortfall_first a x = 2 * shortfall_second a x) : x = a / 3 :=
by
  sorry

end salt_fraction_l105_105931


namespace find_counterfeit_10_l105_105141

theorem find_counterfeit_10 (coins : Fin 10 → ℕ) (h_counterfeit : ∃ k, ∀ i, i ≠ k → coins i < coins k) : 
  ∃ w : ℕ → ℕ → Prop, (∀ g1 g2, g1 ≠ g2 → w g1 g2 ∨ w g2 g1) → 
  ∃ k, ∀ i, i ≠ k → coins i < coins k :=
sorry

end find_counterfeit_10_l105_105141


namespace max_value_relationship_l105_105566

theorem max_value_relationship (x y : ℝ) :
  (2005 - (x + y)^2 = 2005) → (x = -y) :=
by
  intro h
  sorry

end max_value_relationship_l105_105566


namespace fourth_vertex_area_l105_105544

/-- Define square and coordinates of points P, Q, R on the edges -/
structure Square :=
  (A B C D: Point)
  (AB BC CD DA : Line)
  (E F : Point)
  (AB_EQ_AD : dist A B = dist A D)
  (E_midpoint : is_midpoint E B C)
  (F_midpoint : is_midpoint F A D)

/-- Define rhombus vertices as points on the square -/
structure Rhombus (sq: Square) :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (P_on_AB : lies_on P sq.AB)
  (Q_on_BC : lies_on Q sq.BC)
  (R_on_CD : lies_on R sq.CD)
  (K_mid_PR : is_midpoint K P R)
  (S_reflect_Q : reflects Q K S)

/-- Prove the area of the possible region for fourth vertices S -/
theorem fourth_vertex_area (sq : Square) (rh: Rhombus sq) : 
  region_of_possible_S.area = sq.ABCD.area := 
sorry

end fourth_vertex_area_l105_105544


namespace largest_multiple_of_12_negation_l105_105958

theorem largest_multiple_of_12_negation (k : ℤ) (h1 : 12 * k = 144) (h2 : -12 * k > -150) : 12 * k = 144 :=
by
  unfold has_mul.mul
  unfold has_neg.neg
  sorry

end largest_multiple_of_12_negation_l105_105958


namespace central_cell_value_l105_105408

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l105_105408


namespace thomas_total_bill_is_102_l105_105188

noncomputable def item_prices := {
  shirts := 12 * 3,
  socks := 5,
  shorts := 15 * 2,
  swim_trunks := 14
}

noncomputable def total_purchase_price : ℝ :=
  item_prices.shirts + item_prices.socks + item_prices.shorts + item_prices.swim_trunks

noncomputable def shipping_cost (total_purchase_price : ℝ) : ℝ :=
  if total_purchase_price < 50 then 5
  else 0.20 * total_purchase_price

noncomputable def total_bill : ℝ :=
  total_purchase_price + shipping_cost total_purchase_price

theorem thomas_total_bill_is_102 :
  total_bill = 102 :=
by
  -- To be filled in with detailed steps if needed
  sorry

end thomas_total_bill_is_102_l105_105188


namespace quadratic_has_two_distinct_real_roots_l105_105089

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end quadratic_has_two_distinct_real_roots_l105_105089


namespace sqrt_sum_ineq_l105_105059

theorem sqrt_sum_ineq (x y z t : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1)
                      (hz : 0 < z ∧ z < 1) (ht : 0 < t ∧ t < 1) :
    sqrt (x^2 + (1 - t)^2) + sqrt (y^2 + (1 - x)^2) +
    sqrt (z^2 + (1 - y)^2) + sqrt (t^2 + (1 - z)^2) < 4 :=
by 
  sorry

end sqrt_sum_ineq_l105_105059


namespace integral_part_odd_l105_105878

theorem integral_part_odd (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, (⌊(3 + Real.sqrt 5)^n⌋ = 2 * m + 1) := 
by
  -- Sorry used since the proof steps are not required in the task
  sorry

end integral_part_odd_l105_105878


namespace distance_AB_l105_105433

theorem distance_AB : 
  let l := (λ (x y : ℝ), y = 3 * x)
  let C := (λ (x y : ℝ), x^2 - y^2 = -4)
  ∀ (x1 y1 x2 y2 : ℝ),
    l x1 y1 → C x1 y1 → 
    l x2 y2 → C x2 y2 → 
    (x1, y1) ≠ (x2, y2) →
    dist (x1, y1) (x2, y2) = 2 * sqrt 5 :=
by
  -- declare variables
  intros l C x1 y1 x2 y2 _ _ _ _ _
  sorry

end distance_AB_l105_105433


namespace simplify_fraction_l105_105490

theorem simplify_fraction : (12.factorial : ℚ) / (10.factorial + 3 * 9.factorial) = 1320 / 13 := by
  sorry

end simplify_fraction_l105_105490


namespace binom_coeff_eq_l105_105658

theorem binom_coeff_eq (n : ℕ) :
  (nat.choose 12 n = nat.choose 12 (2 * n - 3)) ↔ (n = 3 ∨ n = 5) :=
by
  sorry

end binom_coeff_eq_l105_105658


namespace central_cell_value_l105_105391

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l105_105391


namespace unique_x1_exists_l105_105240

theorem unique_x1_exists (x : ℕ → ℝ) :
  (∀ n : ℕ+, x (n+1) = x n * (x n + 1 / n)) →
  ∃! (x1 : ℝ), (∀ n : ℕ+, 0 < x n ∧ x n < x (n+1) ∧ x (n+1) < 1) :=
sorry

end unique_x1_exists_l105_105240


namespace a11_plus_b11_l105_105473

-- Definitions of the given conditions
variable {a b : ℝ}

def a_plus_b := a + b = 1
def a2_plus_b2 := a^2 + b^2 = 3
def a3_plus_b3 := a^3 + b^3 = 4
def a4_plus_b4 := a^4 + b^4 = 7
def a5_plus_b5 := a^5 + b^5 = 11

-- Recursive definition for n >= 3
def recurrence_relation (n : ℕ) (h : n ≥ 3) : Prop :=
  a^n + b^n = a^(n-2) + b^(n-2) + a^(n-1) + b^(n-1)

-- Theorem to prove the solution
theorem a11_plus_b11 : a_plus_b → a2_plus_b2 → a3_plus_b3 → a4_plus_b4 → a5_plus_b5 → 
  (∀ n, n ≥ 3 → recurrence_relation n (by linarith)) → a^11 + b^11 = 199 :=
by
  intros
  sorry

end a11_plus_b11_l105_105473


namespace cube_coloring_possible_l105_105147

namespace CubeColoring

-- Definitions
structure Cube :=
  (size : ℕ)
  (E_edge : ℤ × ℤ × ℤ)
  (F_face : ℤ × ℤ × ℤ)

-- Conditions
theorem cube_coloring_possible (Ω : Finset (ℤ × ℤ × ℤ)) (hΩ : Ω.card = 2020) :
  ∃ (blue_faces : Finset (ℤ × ℤ × ℤ)),
    (∀ e ∈ Ω, odd (blue_faces.count e)) ∧
    (∀ e, e ∉ Ω → even (blue_faces.count e)) := by
  -- Proof goes here
  sorry

end CubeColoring

end cube_coloring_possible_l105_105147


namespace original_fraction_is_one_third_l105_105041

theorem original_fraction_is_one_third (a b : ℕ) 
  (coprime_ab : Nat.gcd a b = 1) 
  (h : (a + 2) * b = 3 * a * b^2) : 
  (a = 1 ∧ b = 3) := 
by 
  sorry

end original_fraction_is_one_third_l105_105041


namespace school_growth_difference_l105_105528

theorem school_growth_difference (X Y : ℕ) (H₁ : Y = 2400)
  (H₂ : X + Y = 4000) : (X + 7 * X / 100 - X) - (Y + 3 * Y / 100 - Y) = 40 :=
by
  sorry

end school_growth_difference_l105_105528


namespace change_in_y_when_x_increases_l105_105646

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- State the theorem
theorem change_in_y_when_x_increases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -5 :=
by
  sorry

end change_in_y_when_x_increases_l105_105646


namespace age_difference_between_brother_and_cousin_l105_105001

-- Define the ages used in the problem 
def Lexie_age : ℕ := 8
def Grandma_age : ℕ := 68
def Brother_age : ℕ := Lexie_age - 6
def Sister_age : ℕ := 2 * Lexie_age
def Uncle_age : ℕ := Grandma_age - 12
def Cousin_age : ℕ := Brother_age + 5

-- The proof problem statement in Lean 4
theorem age_difference_between_brother_and_cousin : 
  Brother_age < Cousin_age ∧ Cousin_age - Brother_age = 5 :=
by
  -- Definitions and imports are done above. The statement below should prove the age difference.
  sorry

end age_difference_between_brother_and_cousin_l105_105001


namespace rope_total_length_is_54m_l105_105120

noncomputable def totalRopeLength : ℝ :=
  let horizontalDistance : ℝ := 16
  let heightAB : ℝ := 18
  let heightCD : ℝ := 30
  let ropeBC := Real.sqrt (horizontalDistance^2 + (heightCD - heightAB)^2)
  let ropeAC := Real.sqrt (horizontalDistance^2 + heightCD^2)
  ropeBC + ropeAC

theorem rope_total_length_is_54m : totalRopeLength = 54 := sorry

end rope_total_length_is_54m_l105_105120


namespace part_a_l105_105575

theorem part_a (x y : ℝ) : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

end part_a_l105_105575


namespace quadratic_has_two_distinct_real_roots_l105_105100

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  let a := 1
      c := -8
      b := m
      Δ := b^2 - 4 * a * c 
  in (Δ > 0) :=
by
  let a := 1
  let c := -8
  let b := m
  let Δ := b^2 - 4 * a * c
  show Δ > 0
  sorry

end quadratic_has_two_distinct_real_roots_l105_105100


namespace living_room_size_is_96_l105_105857

-- Define the total area of the apartment
def total_area : ℕ := 16 * 10

-- Define the number of units
def units : ℕ := 5

-- Define the size of one unit
def size_of_one_unit : ℕ := total_area / units

-- Define the size of the living room
def living_room_size : ℕ := size_of_one_unit * 3

-- Proving that the living room size is indeed 96 square feet
theorem living_room_size_is_96 : living_room_size = 96 := 
by
  -- not providing proof, thus using sorry
  sorry

end living_room_size_is_96_l105_105857


namespace central_cell_value_l105_105367

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l105_105367


namespace highest_elevation_reached_l105_105598

noncomputable def max_elevation (s : ℝ → ℝ) : ℝ :=
  let t := -400 / (-40) in
  s t

theorem highest_elevation_reached :
  ∀ (t : ℝ), s = (λ t, 200 * t - 20 * t^2) → max_elevation s = 500 :=
by
  intros t s_hyp
  let s := λ t, 200 * t - 20 * t^2
  unfold max_elevation
  simp
  sorry

end highest_elevation_reached_l105_105598


namespace central_cell_value_l105_105397

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l105_105397


namespace square_and_rectangle_equal_area_l105_105604

theorem square_and_rectangle_equal_area (a b c : ℝ) (h : a^2 = b * c) : a = sqrt (b * c) :=
sorry

end square_and_rectangle_equal_area_l105_105604


namespace original_work_days_l105_105157

-- Definitions based on conditions
noncomputable def L : ℕ := 7  -- Number of laborers originally employed
noncomputable def A : ℕ := 3  -- Number of absent laborers
noncomputable def t : ℕ := 14 -- Number of days it took the remaining laborers to finish the work

-- Theorem statement to prove
theorem original_work_days : (L - A) * t = L * 8 := by
  sorry

end original_work_days_l105_105157


namespace project_payment_correct_l105_105145

noncomputable def total_project_payment (q p h : ℕ) : ℕ :=
  27 * h

theorem project_payment_correct (q : ℕ) (h : ℕ)
  (hq_pos : q = 18)
  (p_eq : 1.5 * q = q + 9)
  (hours_relation : 27 * h = 18 * (h + 10)) :
  total_project_payment q p h = 540 := 
by
  rw [p_eq, hq_pos]
  have h_correct : h = 20 := by
    sorry
  rw [h_correct]
  show 27 * 20 = 540
  norm_num
  sorry

end project_payment_correct_l105_105145


namespace quadratic_has_two_distinct_real_roots_l105_105082

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l105_105082


namespace sum_of_squares_interval_l105_105461

variable {n : ℕ} {k : ℝ} (x : Fin n → ℝ)
variable (hx_pos : ∀ i, 0 < x i) (hx_sum : ∑ i, x i = k) (hk_pos : 0 < k)

/-- The possible values for the sum of the squares of the sequence x_i of positive real numbers 
whose sum is equal to k lie in the interval (0, k^2). -/
theorem sum_of_squares_interval : 0 < ∑ i, (x i) ^ 2 ∧ ∑ i, (x i) ^ 2 < k ^ 2 :=
  sorry

end sum_of_squares_interval_l105_105461


namespace functional_equation_solution_l105_105458

def odd_integers (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem functional_equation_solution (f : ℤ → ℤ)
  (h_odd : ∀ x : ℤ, odd_integers (f x))
  (h_eq : ∀ x y : ℤ, 
    f (x + f x + y) + f (x - f x - y) = f (x + y) + f (x - y)) :
  ∃ (d k : ℤ) (ell : ℕ → ℤ), 
    (∀ i : ℕ, i < d → odd_integers (ell i)) ∧
    ∀ (m : ℤ) (i : ℕ), i < d → 
      f (m * d + i) = 2 * k * m * d + ell i :=
sorry

end functional_equation_solution_l105_105458


namespace chocolates_problem_l105_105354

theorem chocolates_problem (C S : ℝ) (n : ℕ) 
  (h1 : 24 * C = n * S)
  (h2 : (S - C) / C = 0.5) : 
  n = 16 :=
by 
  sorry

end chocolates_problem_l105_105354


namespace max_value_sqrt_sum_l105_105237

theorem max_value_sqrt_sum (x : ℝ) (h : -16 ≤ x ∧ x ≤ 16) : 
  sqrt (16 + x) + sqrt (16 - x) ≤ 8 :=
begin
  sorry
end

end max_value_sqrt_sum_l105_105237


namespace number_of_marbles_pat_keeps_l105_105868

theorem number_of_marbles_pat_keeps 
  (x : ℕ) 
  (h1 : x / 6 = 9) 
  : x / 3 = 18 :=
by
  sorry

end number_of_marbles_pat_keeps_l105_105868


namespace angle_between_vectors_l105_105346

open RealInnerProductSpace
open Complex

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_vectors
  {a b : V} (hna : a ≠ 0) (hnb : b ≠ 0) 
  (h1 : ⟪a - (2:ℝ) • b, a⟫ = 0)
  (h2 : ⟪b - (2:ℝ) • a, b⟫ = 0) :
  angle a b = real.pi / 3 :=
by {
  sorry
}

end angle_between_vectors_l105_105346


namespace altitude_of_equilateral_triangle_on_rectangle_diagonal_l105_105262

variables (l w : ℝ)

theorem altitude_of_equilateral_triangle_on_rectangle_diagonal (h : ℝ) :
  (h * h = (3 / 4) * (l^2 + w^2)) ↔ 
  (h = (sqrt 3 / 2) * sqrt (l^2 + w^2)) := 
sorry

end altitude_of_equilateral_triangle_on_rectangle_diagonal_l105_105262


namespace solve_system_l105_105031

theorem solve_system : ∃ x y : ℝ, x + y = 4 ∧ x - 2 * y = 1 :=
by
  use 3
  use 1
  split
  · exact rfl
  · exact rfl

end solve_system_l105_105031


namespace Borya_wins_optimal_play_l105_105941

/--
There are 123 points equally spaced along the circumference of a circle.
Anya and Borya take turns coloring one of these points either blue or red.
The player who causes two adjacent points to be the same color loses.
Anya goes first.
--/
theorem Borya_wins_optimal_play :
  ∀ (points : Fin 123 → Prop) (colors : Fin 123 → Prop) (turn : ℕ → Prop),
  (∀ i, (i ∈ points) ∧ (‎colors i = blue ∨ colors i = red)) →
  (∀ i, ((colors i = colors (i + 1)) → colors (i + 1) = default)) →
  (0 ∈ turn → Anya wins) → (1 ∈ turn → Borya wins) →
  (∃ n, Borya wins).
proof :=
by {
  sorry
}

end Borya_wins_optimal_play_l105_105941


namespace find_value_of_a_l105_105102

theorem find_value_of_a (a : ℝ) (h : ( (-2 - (2 * a - 1)) / (3 - (-2)) = -1 )) : a = 2 :=
sorry

end find_value_of_a_l105_105102


namespace dance_total_people_l105_105066

theorem dance_total_people
  (ratio_b_g : 3 / 4)
  (num_girls : ℕ := 60)
  (teacher_percentage : ℕ := 20) :
  let num_parts := num_girls / 4;
      num_boys := num_parts * 3;
      num_teachers := num_boys * 20 / 100;
      total_people := num_girls + num_boys + num_teachers
  in
    total_people = 114 :=
by
  let num_parts := 60 / 4;
  let num_boys := num_parts * 3;
  let num_teachers := num_boys * 20 / 100;
  let total_people := 60 + num_boys + num_teachers;
  have : total_people = 114 := sorry,
  assumption

end dance_total_people_l105_105066


namespace total_pizza_order_cost_l105_105886

def pizza_cost_per_pizza := 10
def topping_cost_per_topping := 1
def tip_amount := 5
def number_of_pizzas := 3
def number_of_toppings := 4

theorem total_pizza_order_cost : 
  (pizza_cost_per_pizza * number_of_pizzas + topping_cost_per_topping * number_of_toppings + tip_amount) = 39 := by
  sorry

end total_pizza_order_cost_l105_105886


namespace max_area_of_backyard_l105_105483

theorem max_area_of_backyard (fence_length : ℕ) (h1 : fence_length = 500) 
  (l w : ℕ) (h2 : l = 2 * w) (h3 : l + 2 * w = fence_length) : 
  l * w = 31250 := 
by
  sorry

end max_area_of_backyard_l105_105483


namespace Wendy_Ran_Farther_l105_105124

theorem Wendy_Ran_Farther (
  distance_ran : ℝ,
  distance_walked : ℝ
)
  (h1 : distance_ran = 19.833333333333332)
  (h2 : distance_walked = 9.166666666666666)
  : distance_ran - distance_walked = 10.666666666666666 := 
by
  rw [h1, h2]
  norm_num
  sorry

end Wendy_Ran_Farther_l105_105124


namespace perp_of_foot_altitudes_l105_105822

theorem perp_of_foot_altitudes (A B C O H_B H_C : Type)
  [linear_ordered_field A]
  (triangle_ABC : is_triangle A B C)
  (circumcenter : is_circumcenter A B C O)
  (foot_HB : is_foot_from B C H_B)
  (foot_HC : is_foot_from C B H_C) :
  perp (line_segment H_B H_C) (line_segment A O) := 
sorry

end perp_of_foot_altitudes_l105_105822


namespace dance_total_people_l105_105065

theorem dance_total_people
  (ratio_b_g : 3 / 4)
  (num_girls : ℕ := 60)
  (teacher_percentage : ℕ := 20) :
  let num_parts := num_girls / 4;
      num_boys := num_parts * 3;
      num_teachers := num_boys * 20 / 100;
      total_people := num_girls + num_boys + num_teachers
  in
    total_people = 114 :=
by
  let num_parts := 60 / 4;
  let num_boys := num_parts * 3;
  let num_teachers := num_boys * 20 / 100;
  let total_people := 60 + num_boys + num_teachers;
  have : total_people = 114 := sorry,
  assumption

end dance_total_people_l105_105065


namespace simplify_trig_expression_l105_105502

variable (A : ℝ)

-- Define trigonometric functions in terms of sine and cosine
def cot (A : ℝ) := cos A / sin A
def csc (A : ℝ) := 1 / sin A
def tan (A : ℝ) := sin A / cos A
def sec (A : ℝ) := 1 / cos A

-- Theorem stating the simplification of the given expression
theorem simplify_trig_expression (A : ℝ) :
  (1 - cot A + csc A) * (1 - tan A - sec A) = 
  2 * ((cos A - sin A) / (sin A * cos A)) :=
by
  sorry

end simplify_trig_expression_l105_105502


namespace tan_of_fourth_quadrant_l105_105275

theorem tan_of_fourth_quadrant (α : ℝ) (h₁ : Real.sin α = -5 / 13) (h₂ : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) : Real.tan α = -5 / 12 :=
sorry

end tan_of_fourth_quadrant_l105_105275


namespace hotpot_total_cost_l105_105162

def table_cost : ℝ := 280
def table_limit : ℕ := 8
def extra_person_cost : ℝ := 29.9
def total_people : ℕ := 12

theorem hotpot_total_cost : 
  total_people > table_limit →
  table_cost + (total_people - table_limit) * extra_person_cost = 369.7 := 
by 
  sorry

end hotpot_total_cost_l105_105162


namespace five_digit_palindromes_l105_105753

theorem five_digit_palindromes : 
  let first_digit := 9 in
  let second_digit := 10 in
  let third_digit := 10 in
  first_digit * second_digit * third_digit = 900 :=
by
  let first_digit := 9
  let second_digit := 10
  let third_digit := 10
  trivial

end five_digit_palindromes_l105_105753


namespace tangent_line_at_one_l105_105292

noncomputable def f (x : ℝ) : ℝ :=
  2 * f (2 - x) - x ^ 2 + 8 * x - 8

-- We need to show the equation of the tangent line at the point (1, f 1).
theorem tangent_line_at_one (f : ℝ → ℝ)
  (h : ∀ x, f x = 2 * f (2 - x) - x ^ 2 + 8 * x - 8) :
  ∃ (m b : ℝ), (∀ x, f 1 = 1 ^ 2) ∧ (∀ x, differentiable_at ℝ f x) ∧ (f 1 = 1 ^ 2 ∧ ∀ x, m = deriv f 1 ∧ b = f 1 - deriv f 1 * 1) ∧ (∀ x, y = m*x + b) :=
sorry

end tangent_line_at_one_l105_105292


namespace unique_function_satisfies_condition_l105_105839

noncomputable def S : Set ℝ := { x : ℝ | x > -1 }

def strictly_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ I → y ∈ I → x < y → f x < f y

theorem unique_function_satisfies_condition :
  ∃! f : ℝ → ℝ, (∀ x y : ℝ, x ∈ S → y ∈ S → f (x + f y + x * f y) = y + f x + y * f x) ∧
    strictly_increasing_on (λ x, f x / x) { x : ℝ | -1 < x ∧ x < 0 } ∧
    strictly_increasing_on (λ x, f x / x) { x : ℝ | x > 0 } ∧
    (∀ x : ℝ, x ∈ S → f x = - x / (x + 1)) :=
begin
  sorry
end

end unique_function_satisfies_condition_l105_105839


namespace max_value_t_min_value_y_l105_105360

open Real

-- Maximum value of t for ∀ x ∈ ℝ, |3x + 2| + |3x - 1| ≥ t
theorem max_value_t :
  ∃ t, (∀ x : ℝ, |3 * x + 2| + |3 * x - 1| ≥ t) ∧ t = 3 :=
by
  sorry

-- Minimum value of y for 4m + 5n = 3
theorem min_value_y (m n: ℝ) (hm : m > 0) (hn: n > 0) (h: 4 * m + 5 * n = 3) :
  ∃ y, (y = (1 / (m + 2 * n)) + (4 / (3 * m + 3 * n))) ∧ y = 3 :=
by
  sorry

end max_value_t_min_value_y_l105_105360


namespace range_of_m_satisfies_inequality_l105_105770

theorem range_of_m_satisfies_inequality (m : ℝ) :
  ((∀ x : ℝ, (1 - m^2) * x^2 - (1 + m) * x - 1 < 0) ↔ (m ≤ -1 ∨ m > 5/3)) :=
sorry

end range_of_m_satisfies_inequality_l105_105770


namespace exists_point_with_distances_greater_than_perimeter_l105_105872

theorem exists_point_with_distances_greater_than_perimeter
  (n : ℕ) (h1 : n ≥ 7) (P : Fin n → ℝ) :
  ∃ p : ℝ × ℝ, ∑ i in Finset.univ, dist p (P i) > (∑ i in Finset.univ, dist (P i) (P (i + 1) % n)) :=
sorry

end exists_point_with_distances_greater_than_perimeter_l105_105872


namespace total_balls_without_holes_l105_105012

theorem total_balls_without_holes 
  (soccer_balls : ℕ) (soccer_balls_with_hole : ℕ)
  (basketballs : ℕ) (basketballs_with_hole : ℕ)
  (h1 : soccer_balls = 40)
  (h2 : soccer_balls_with_hole = 30)
  (h3 : basketballs = 15)
  (h4 : basketballs_with_hole = 7) :
  soccer_balls - soccer_balls_with_hole + (basketballs - basketballs_with_hole) = 18 := by
  sorry

end total_balls_without_holes_l105_105012


namespace min_degree_polynomial_l105_105270

theorem min_degree_polynomial (n : ℕ) (h : n ≥ 3) 
  (points : Fin n → ℝ × ℝ) (h_no_collinear : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → ¬collinear (points i) (points j) (points k))
  (c : Fin n → ℝ) :
  ∃ P : ℝ × ℝ → ℝ, (∀ i, P (points i) = c i) ∧ 
  (∃ k, degree P ≤ k) ∧ k = Nat.floor (n / 2) :=
begin
  sorry
end

end min_degree_polynomial_l105_105270


namespace central_cell_value_l105_105395

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l105_105395


namespace trapezoid_parallel_intersect_on_DA_l105_105846

variable (α : Type*) [LinearOrderedField α] 
variables (A B C D P : EuclideanGeometry.Point α)
variables (AB CD : EuclideanGeometry.Line α)
variables (BC DA : EuclideanGeometry.Line α)
variables (AP PD : EuclideanGeometry.Line α)

-- Given conditions
variable (AB_parallel_CD : AB.parallel CD)
variable (P_on_BC : BC.contains P)

-- To Prove
theorem trapezoid_parallel_intersect_on_DA (A B C D P : EuclideanGeometry.Point α)
    (AB CD BC DA AP PD : EuclideanGeometry.Line α)
    (AB_parallel_CD : AB.parallel CD)
    (P_on_BC : BC.contains P) :
    let X := DA.intersect (AP.parallel_through C),
        Y := DA.intersect (PD.parallel_through B)
    in X = Y := 
sorry

end trapezoid_parallel_intersect_on_DA_l105_105846


namespace solution_set_inequality_l105_105534

open Real

theorem solution_set_inequality (k : ℤ) (x : ℝ) :
  (x ∈ Set.Ioo (-π/4 + k * π) (k * π)) ↔ cos (4 * x) - 2 * sin (2 * x) - sin (4 * x) - 1 > 0 :=
by
  sorry

end solution_set_inequality_l105_105534


namespace anita_gave_apples_l105_105509

theorem anita_gave_apples (initial_apples needed_for_pie apples_left_after_pie : ℝ)
  (h_initial : initial_apples = 10.0)
  (h_needed : needed_for_pie = 4.0)
  (h_left : apples_left_after_pie = 11.0) :
  ∃ (anita_apples : ℝ), anita_apples = 5 :=
by
  sorry

end anita_gave_apples_l105_105509


namespace simplified_expression_simplified_expression_neg_undefined_case_l105_105890

noncomputable def simplify_expression (u v : ℝ) : ℝ :=
  sqrt (u^2 - 2 * u * v + 3 * v^2 + 2 * v * sqrt (3 * u * (u - 2 * v)))

theorem simplified_expression (u v : ℝ) (h1 : 3 * u * (u - 2 * v) ≥ 0) (h2 : u * (u - 2 * v) ≥ 0) :
    (0 ≤ v ∧ v ≤ u / 2) ∨ (u ≤ 0 ∧ 0 ≤ v) →
    simplify_expression u v = sqrt (u * (u - 2 * v)) + sqrt 3 * v :=
sorry

theorem simplified_expression_neg (u v : ℝ) (h1 : 3 * u * (u - 2 * v) ≥ 0) (h2 : u * (u - 2 * v) ≥ 0) :
    v < 0 ∧ 3 * v < u ∧ u ≤ 2 * v < 0 ∨ (0 ≤ u ∧ u < -v) →
    simplify_expression u v = sqrt 3 * |v| - sqrt (u * (u - 2 * |v|)) :=
sorry

theorem undefined_case (u v : ℝ) (h1 : 3 * u * (u - 2 * v) ≥ 0) (h2 : u * (u - 2 * v) ≥ 0) :
    0 < u ∧ u < 2 * v ∨ (2 * v < u ∧ u < 0) →
    ¬ (∃ x, simplify_expression u v = x) :=
sorry

end simplified_expression_simplified_expression_neg_undefined_case_l105_105890


namespace problem_number_of_subsets_of_union_l105_105850

noncomputable def A : Set ℝ := {x | x^2 - 2 * x = 0}
def B : Set ℝ := {0, 1}
def union_set : Set ℝ := A ∪ B
def num_subsets (s : Set ℝ) : ℕ := 2 ^ s.to_finset.card

theorem problem_number_of_subsets_of_union :
  num_subsets union_set = 8 := by
  sorry

end problem_number_of_subsets_of_union_l105_105850


namespace seashells_total_l105_105028

theorem seashells_total {sally tom jessica : ℕ} (h₁ : sally = 9) (h₂ : tom = 7) (h₃ : jessica = 5) : sally + tom + jessica = 21 := by
  sorry

end seashells_total_l105_105028


namespace robbie_weekly_fat_intake_l105_105882

theorem robbie_weekly_fat_intake
  (morning_cups : ℕ) (afternoon_cups : ℕ) (evening_cups : ℕ)
  (fat_per_cup : ℕ) (days_per_week : ℕ) :
  morning_cups = 3 →
  afternoon_cups = 2 →
  evening_cups = 5 →
  fat_per_cup = 10 →
  days_per_week = 7 →
  (morning_cups * fat_per_cup + afternoon_cups * fat_per_cup + evening_cups * fat_per_cup) * days_per_week = 700 :=
by
  intros
  sorry

end robbie_weekly_fat_intake_l105_105882


namespace find_three_digit_number_l105_105234

theorem find_three_digit_number :
  ∃ (x y z : ℕ), 
  (100 * x + 10 * y + z = 139) ∧
  (4 * x + y - (7 / 9) * z = 0) ∧
  (47 * x - 13 * y = 8) ∧
  (33 * x + 2 * y - 5 * z = -6) :=
by
  sorry

end find_three_digit_number_l105_105234


namespace lines_equal_plane_angle_l105_105772

-- Definitions of the conditions
variables {Line : Type} [Inhabited Line]
variables (l1 l2 : Line) (plane : Line → ℝ) -- Simplified to represent angle with the plane
variable (distinct : l1 ≠ l2)
variable (equalAngle : plane l1 = plane l2)

-- Proof statement
theorem lines_equal_plane_angle :
    (l1 ≠ l2 → plane l1 = plane l2 → (∃ parallel, (parallel = (l1 ∥ l2)) ∨ ∃ intersecting, intersecting = (l1 ∩ l2) ∨ ∃ skew, skew = (skewLine l1 l2)))
:= sorry

end lines_equal_plane_angle_l105_105772


namespace min_value_of_func_l105_105678

def min_func (x : ℝ) : ℝ := (x^2 + 11) / Real.sqrt (x^2 + 5)

theorem min_value_of_func : ∃ x : ℝ, min_func x = 2 * Real.sqrt 6 :=
by
  sorry

end min_value_of_func_l105_105678


namespace quadratic_roots_distinct_l105_105079

-- Define the quadratic equation condition
def quadratic_eq : (ℝ → ℝ) :=
  λ x m => x^2 + m * x - 8

-- State the problem
theorem quadratic_roots_distinct (m : ℝ) : 
  let Δ := m^2 + 32 in Δ > 0 :=
  by
   -- We need this theorem to state that the equation always has distinct real roots
  let Δ := m^2 + 32
  sorry

end quadratic_roots_distinct_l105_105079


namespace exists_point_distance_sum_greater_than_perimeter_l105_105875

theorem exists_point_distance_sum_greater_than_perimeter (n : ℕ) (h_n : n ≥ 7) 
  (P : ℝ) (convex_n_gon : Type) [ConvexPolygon n convex_n_gon] :
  ∃ X : convex_n_gon, (∑ v in vertices convex_n_gon, dist X v) > P := sorry

end exists_point_distance_sum_greater_than_perimeter_l105_105875


namespace find_a₁₈_l105_105263

open Classical

-- Define the arithmetic sequence and conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ)

-- Define a₁ (the first term of the arithmetic sequence)
def a₁ := (2 : ℝ)

-- Define the arithmetic sequence formula
def a (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def Sn (n : ℕ) : ℝ := (n / 2) * (2 * a₁ + (n - 1) * d)

-- Given conditions
axiom Sn_equal : Sn 8 = Sn 10

-- Theorem statement to find a₁₈
theorem find_a₁₈ : a 18 = -2 :=
by
  -- Conditions derived from the problem (used as axioms here)
  have h1 : a₁ = 2 := rfl
  have h2 : S 8 = S 10 := Sn_equal
  -- Placeholder for proof, which is omitted using "sorry"
  sorry

end find_a₁₈_l105_105263


namespace maximum_marks_for_test_l105_105175

noncomputable def max_marks_set_for_test : ℝ :=
  let M := 300
  in M

theorem maximum_marks_for_test (M : ℝ) (h1 : 0.60 * M = 80 + 100) : M = 300 :=
by
  exact sorry

end maximum_marks_for_test_l105_105175


namespace increasing_over_domain_C_l105_105343

def f_A (x : ℝ) : ℝ := x^2
def f_B (x : ℝ) : ℝ := Real.exp (-x)
def f_C (x : ℝ) : ℝ := x - Real.sin x
def f_D (x : ℝ) : ℝ := -Real.sqrt x

theorem increasing_over_domain_C :
  ∀ x y : ℝ, x ≤ y → f_C x ≤ f_C y ∧ 
             (∀ x : ℝ, 
                 (x < 0 → f_A' x < 0) ∧ (0 ≤ x → f_A' x ≥ 0)) ∧ 
             (∀ x : ℝ, f_B' x < 0) ∧ 
             (∀ x : ℝ, 0 ≤ x → f_D' x ≤ 0) := 
by
  intros x y hxy
  sorry

end increasing_over_domain_C_l105_105343


namespace similar_triangles_x_value_l105_105795

theorem similar_triangles_x_value
  (x : ℝ)
  (h_similar : ∀ (AB BC DE EF : ℝ), AB / BC = DE / EF)
  (h_AB : AB = x)
  (h_BC : BC = 33)
  (h_DE : DE = 96)
  (h_EF : EF = 24) :
  x = 132 :=
by
  -- Proof steps will be here
  sorry

end similar_triangles_x_value_l105_105795


namespace central_cell_value_l105_105387

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l105_105387


namespace infinite_divisibles_l105_105475

def color (n : ℤ) : Prop

theorem infinite_divisibles (A B : set ℤ) (hA : ∀ n, (color n ↔ n ∈ A)) (hB : ∀ n, (¬ color n ↔ n ∈ B)) :
  ∃ (color_has_property : ∀ k : ℕ, ∃∞ n : ℤ, n ∣ k ∧ color n ↔ k) :=
sorry

end infinite_divisibles_l105_105475


namespace tank_A_height_l105_105897

theorem tank_A_height :
  ∀ (h_B : ℝ) (C_A C_B : ℝ) (V_ratio : ℝ),
    C_A = 9 →
    C_B = 10 →
    h_B = 9 →
    V_ratio = 0.9000000000000001 →
    let r_A := C_A / (2 * π),
        r_B := C_B / (2 * π),
        V_A := π * r_A^2 * h_A,
        V_B := π * r_B^2 * h_B
    in V_A = V_ratio * V_B →
       h_A = 8.1 :=
by
  intros h_B C_A C_B V_ratio hCAB hCBB hBB hVR
  simp [hCAB, hCBB, hBB, hVR]
  sorry

end tank_A_height_l105_105897


namespace find_added_value_l105_105665

theorem find_added_value (x y : ℕ) (hx : x = 3) (h : x + y = 60 * (1 / x)) : y = 17 := by
  rw [hx] at h
  have h1 : 60 * (1 / 3) = 20 := by norm_num
  rw [h1] at h
  linarith

example : find_added_value 3 17 3 rfl := by sorry

end find_added_value_l105_105665


namespace andrew_spent_total_amount_l105_105621

def cost_fruit_A : ℝ := 14 * 54
def cost_fruit_B_local : ℝ := 10 * 62
def cost_pineapple_C : ℝ := 8 * 40
def cost_kiwi_C : ℝ := 5 * 30
def cost_fruit_C_local : ℝ := cost_pineapple_C + cost_kiwi_C

def exchange_rate_EUR_to_USD : ℝ := 1 / 0.85
def exchange_rate_GBP_to_USD : ℝ := 1 / 0.74

def cost_fruit_B_USD : ℝ := cost_fruit_B_local * exchange_rate_EUR_to_USD
def cost_fruit_C_USD : ℝ := cost_fruit_C_local * exchange_rate_GBP_to_USD

def total_cost_USD : ℝ := cost_fruit_A + cost_fruit_B_USD + cost_fruit_C_USD

theorem andrew_spent_total_amount :
  total_cost_USD = 2120.55 :=
by 
  simp [cost_fruit_A, cost_fruit_B_local, cost_kiwi_C, cost_pineapple_C, cost_fruit_C_local, 
        exchange_rate_EUR_to_USD, exchange_rate_GBP_to_USD, cost_fruit_B_USD, cost_fruit_C_USD, 
        total_cost_USD]
  sorry

end andrew_spent_total_amount_l105_105621


namespace projectors_cast_shadow_l105_105423

noncomputable def shadow_casting_projectors (shadow_length : ℝ) (projector_count : ℕ) : Prop :=
  ∃ (positions : Fin projector_count → ℝ × ℝ), 
    ∀ (i : Fin projector_count), 
      ∃ (j : Fin projector_count), 
        i ≠ j ∧ 
        within_illuminated_quadrant (positions i) (positions j) ∧ 
        between (positions i) (positions j) shadow_length

def within_illuminated_quadrant (p1 p2 : ℝ × ℝ) : Prop :=
  (p2.1 > p1.1 ∧ p2.2 >= p1.2) ∨ (p2.1 >= p1.1 ∧ p2.2 > p1.2) ∨
  (p2.1 < p1.1 ∧ p2.2 <= p1.2) ∨ (p2.1 <= p1.1 ∧ p2.2 < p1.2)

def between (p1 p2 : ℝ × ℝ) (dist : ℝ) : Prop :=
  euclidean_distance p1 p2 = dist

def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem projectors_cast_shadow : shadow_casting_projectors 7 7 :=
sorry

end projectors_cast_shadow_l105_105423


namespace simplify_factorial_l105_105494

theorem simplify_factorial :
  ∀ (f12 f10 f9 : ℕ), f12 = 12! ∧ f10 = 10! ∧ f9 = 9! →
  (f12 / (f10 + 3 * f9) = 4 / 3) :=
by
  sorry

end simplify_factorial_l105_105494


namespace businesswoman_loss_percentage_l105_105999

def SP (CP : ℝ) (percentage: ℝ) (loss: Bool) : ℝ :=
  if loss then CP * (1 - percentage / 100) else CP * (1 + percentage / 100)

def overall_profit_or_loss_percentage 
  (CPs : List ℝ) 
  (percentages : List ℝ) 
  (losses : List Bool) : ℝ :=
  let total_CP := CPs.sum
  let total_SP := (List.zipWith3 SP CPs percentages losses).sum
  ((total_SP - total_CP) / total_CP) * 100

theorem businesswoman_loss_percentage :
  overall_profit_or_loss_percentage 
    [240000, 560000, 850000, 1500000, 3440000] 
    [20, 25, 7, 15, 30] 
    [true, false, true, false, true] = -11.75 :=
by
  sorry

end businesswoman_loss_percentage_l105_105999


namespace increasing_interval_l105_105913

def quadratic_function (x : ℝ) : ℝ := -x^2 + 2 * x - 5

theorem increasing_interval :
  {x : ℝ | ∀ (y : ℝ), quadratic_function y > quadratic_function x → y < x} = Iio 1 := 
sorry

end increasing_interval_l105_105913


namespace intersecting_lines_l105_105917

theorem intersecting_lines (c d : ℝ) :
  (∀ x y, (x = (1/3) * y + c) ∧ (y = (1/3) * x + d) → x = 3 ∧ y = 6) →
  c + d = 6 :=
by
  sorry

end intersecting_lines_l105_105917


namespace total_opponents_score_is_72_l105_105588

theorem total_opponents_score_is_72 : 
  ∃ (team_scores : Fin 12 → ℕ) (opponents_scores : Fin 12 → ℕ),
    (team_scores = fun i ↦ i + 1) ∧ 
    (∃ (lost_games won_games : Finset (Fin 12)),
      lost_games.card = 6 ∧ won_games.card = 6 ∧
      (∀ i ∈ lost_games, opponents_scores i = team_scores i + 2) ∧
      (∀ i ∈ won_games, opponents_scores i = team_scores i - 3)) ∧ 
    (Finset.sum Finset.univ opponents_scores = 72) := 
sorry

end total_opponents_score_is_72_l105_105588


namespace ellipse_conditions_inscribed_circle_l105_105720

-- Definitions coming from conditions
def ellipse_eq (a b x y : ℝ) : Prop :=
  ( a > b ∧ b > 0 ∧ 
    (x^2 / a^2) + (y^2 / b^2) = 1 )

def eccentricity_eq (a b : ℝ) : Prop :=
  a / b = 2 ∧ (a > b)

def scalar_condition (a x₁ x₂ b : ℝ) : Prop :=
  (x₁ + x₂ = -8 * real.sqrt(3) * b / 5 ∧ 2 * a * (x₁ + x₂) = -32 * real.sqrt(3) / 5)

def inscribed_circle_eq (m n t : ℝ) : Prop :=
  ∃ x y, (x - t)^2 + y^2 = (m - t)^2 + n^2 ∧ y = x + real.sqrt(3) ∧
  ( - sqrt(3) - t )^2 = (m - t)^2 + n^2 ∧ n^2 = 1 - (m^2 / 4)

-- Lean 4 statements for the corresponding math proof problem
theorem ellipse_conditions (a b x y : ℝ) (h1 : ellipse_eq a b x y) (h2 : eccentricity_eq a b) (x₁ x₂ : ℝ) :
  scalar_condition a x₁ x₂ b := sorry

theorem inscribed_circle (m n t x y : ℝ) (h3 : inscribed_circle_eq m n t) :
  ∃ e_f : ℝ × ℝ, e_f = (- (real.sqrt 3) / 2, 0) := sorry

end ellipse_conditions_inscribed_circle_l105_105720


namespace hyperbola_length_AB_l105_105257

open Real

-- Definition of the hyperbola condition
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - y^2 = 1

-- Definition of the property of the line passing through the right focus of the hyperbola
def line_passing_through_right_focus (x y : ℝ) : Prop := 
  ∃ l : ℝ, ∃ c : Real, y = l * x + c ∧ hyperbola x y ∧ (c = -l * 2 * Real.sqrt(3))

-- The length |AB|
def length_AB (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Prove that the range of the value a is exactly 4
theorem hyperbola_length_AB (a : ℝ) : 
  (∃ (l₁ l₂ l₃ : ℝ), 
    length_AB A B = a ∧
    line_passing_through_right_focus A.1 A.2 ∧ 
    line_passing_through_right_focus B.1 B.2) → 
  a = 4 := 
sorry

end hyperbola_length_AB_l105_105257


namespace pf1_pf2_range_l105_105272

noncomputable def ellipse_point (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 = 1

noncomputable def dot_product (x y : ℝ) : ℝ :=
  (x ^ 2 + y ^ 2 - 3)

theorem pf1_pf2_range (x y : ℝ) (h : ellipse_point x y) :
  -2 ≤ dot_product x y ∧ dot_product x y ≤ 1 :=
by
  sorry

end pf1_pf2_range_l105_105272


namespace cyclic_polygon_product_l105_105705

theorem cyclic_polygon_product (n : ℕ) (h : n ≥ 2) 
  (P : Point) (A : fin (2*n) → Point)
  (h_convex_cyclic : isConvexCyclicPolygon A) 
  (h_angles : ∀ i : fin (2*n), ∠PA (A i) (A ((i + 1) % (2*n))) = ∠PA (A ((i + 1) % (2*n))) (A ((i + 2) % (2*n)))) : 
  (∏ i in range n, dist (A (2 * i)) (A (2 * i + 1))) = 
  (∏ i in range n, dist (A (2 * i + 1)) (A ((2 * i + 2) % (2 * n)))) :=
sorry

end cyclic_polygon_product_l105_105705


namespace max_n_exist_real_numbers_l105_105230

theorem max_n_exist_real_numbers (n : ℕ) : 
  (∀ (x : Fin n → ℝ), (∀ i j : Fin n, (i < j) → 
    (1 + x i * x j) ^ 2 ≤ 0.99 * (1 + (x i) ^ 2) * (1 + (x j) ^ 2)) ↔ n ≤ 31) := 
begin 
  -- proof omitted 
  sorry 
end

end max_n_exist_real_numbers_l105_105230


namespace central_cell_value_l105_105366

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l105_105366


namespace infinitely_many_sum_form_l105_105848

theorem infinitely_many_sum_form {a : ℕ → ℕ} (h : ∀ n, a n < a (n + 1)) :
  ∀ i, ∃ᶠ n in at_top, ∃ r s j, r > 0 ∧ s > 0 ∧ i < j ∧ a n = r * a i + s * a j := 
by
  sorry

end infinitely_many_sum_form_l105_105848


namespace clock_hands_straight_twenty_four_hours_l105_105783

noncomputable def hands_straight_per_day : ℕ :=
  2 * 22

theorem clock_hands_straight_twenty_four_hours :
  hands_straight_per_day = 44 :=
by
  sorry

end clock_hands_straight_twenty_four_hours_l105_105783


namespace common_solution_y_l105_105215

theorem common_solution_y (x y : ℝ) (h₁ : x^2 + y^2 = 9) (h₂ : x^2 = 4y - 8) :
  y = -2 + Real.sqrt 21 ∨ y = -2 - Real.sqrt 21 :=
by {
  -- Proof would go here
  sorry
}

end common_solution_y_l105_105215


namespace exists_duplicate_in_grid_of_differences_bounded_l105_105777

theorem exists_duplicate_in_grid_of_differences_bounded :
  ∀ (f : ℕ × ℕ → ℤ), 
  (∀ i j, i < 10 → j < 10 → (i + 1 < 10 → (abs (f (i, j) - f (i + 1, j)) ≤ 5)) 
                             ∧ (j + 1 < 10 → (abs (f (i, j) - f (i, j + 1)) ≤ 5))) → 
  ∃ x y : ℕ × ℕ, x ≠ y ∧ f x = f y :=
by
  intros
  sorry -- Proof goes here

end exists_duplicate_in_grid_of_differences_bounded_l105_105777


namespace sqrt_expression_is_869_l105_105644

theorem sqrt_expression_is_869 :
  (31 * 30 * 29 * 28 + 1) = 869 := 
sorry

end sqrt_expression_is_869_l105_105644


namespace pin_code_combinations_l105_105211

theorem pin_code_combinations : 
  let digits := [5, 3, 3, 7] in
  multiset.card (multiset.of_list digits) = 12 :=
by
  let n := 4
  let n1 := 1
  let n2 := 2
  let n3 := 1
  let factorial := Nat.factorial
  have perm := factorial n / (factorial n1 * factorial n2 * factorial n3)
  exact perm = 12
  sorry

end pin_code_combinations_l105_105211


namespace smallest_n_to_lose_90_percent_l105_105896

theorem smallest_n_to_lose_90_percent
  (log10_3 : Real.log 10 3 = 0.477) :
  ∃ n : ℕ, ∀ k : ℕ, (k < n) → (0.9 : ℝ)^k > 0.1 ∧ (0.9 : ℝ)^n ≤ 0.1 :=
begin
  -- The proof will go here
  sorry
end

end smallest_n_to_lose_90_percent_l105_105896


namespace central_cell_value_l105_105402

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l105_105402


namespace intersection_A_B_l105_105739

def A : Set (ℚ × ℚ) := {p | p.1 + 3 * p.2 = 7}
def B : Set (ℚ × ℚ) := {p | p.1 - p.2 = -1}

theorem intersection_A_B :
  A ∩ B = { (1 / 2 : ℚ, 3 / 2 : ℚ) } :=
by
  sorry

end intersection_A_B_l105_105739


namespace max_area_of_triangle_l105_105840

/--
Let A = (0, 0), B = (3, 4), and C = (p, -p^2 + 5p) be three points on the parabola 
y = -x^2 + 5x, where 0 ≤ p ≤ 3. Prove that the largest possible area of 
triangle ABC is 121/24.
-/
theorem max_area_of_triangle 
  (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 3) : 
  let A := (0, 0),
      B := (3, 4),
      C := (p, -p^2 + 5p) in
  ∃ q : ℝ, (q = 121/24) ∧ 
  ∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 3 → 
  (1 / 2) * abs (-3 * p'^2 + 11 * p') ≤ q :=
by
  sorry

end max_area_of_triangle_l105_105840


namespace reciprocal_of_F_in_third_quadrant_outside_unit_circle_l105_105794

theorem reciprocal_of_F_in_third_quadrant_outside_unit_circle
  (F : ℂ) 
  (hF_third_quadrant : F.re < 0 ∧ F.im < 0) 
  (hF_outside_unit_circle : F.norm_sq > 1) :
  ∃ A : ℂ, (A.re < 0 ∧ A.im > 0 ∧ A.norm_sq < 1) :=
by
  sorry

end reciprocal_of_F_in_third_quadrant_outside_unit_circle_l105_105794


namespace central_cell_value_l105_105372

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l105_105372


namespace quadratic_has_two_distinct_real_roots_l105_105087

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end quadratic_has_two_distinct_real_roots_l105_105087


namespace order_of_abc_l105_105278

noncomputable def a := (2 / 3) ^ 0.2
noncomputable def b := log 2 3
noncomputable def c := Real.ln (1 / 2)

theorem order_of_abc : c < a ∧ a < b := by
  sorry

end order_of_abc_l105_105278


namespace arithmetic_seq_problem_l105_105842

theorem arithmetic_seq_problem (S : ℕ → ℤ) (n : ℕ) (h1 : S 6 = 36) 
                               (h2 : S n = 324) (h3 : S (n - 6) = 144) (hn : n > 6) : 
  n = 18 := 
sorry

end arithmetic_seq_problem_l105_105842


namespace division_remainder_l105_105129

theorem division_remainder :
  (1225 * 1227 * 1229) % 12 = 3 :=
by sorry

end division_remainder_l105_105129


namespace count_five_digit_numbers_condition_l105_105766

theorem count_five_digit_numbers_condition (nums : Finset ℕ) (h1 : nums = {1, 2, 3}) :
  let total := (3:ℕ)^5
  let same_digits := 3
  let two_distinct := 3 * ((2:ℕ)^5 - 2)
  total - same_digits - two_distinct = 150 :=
by
  let total := (3:ℕ)^5
  let same_digits := 3
  let two_distinct := 3 * ((2:ℕ)^5 - 2)
  have h_total : total = 243 := by norm_num
  have h_same_digits : same_digits = 3 := by norm_num
  have h_two_distinct : two_distinct = 90 := by norm_num
  calc
    243 - 3 - 90 = 243 - (3 + 90) := by rw [sub_sub, add_comm 3 90]
    ... = 243 - 93 := by norm_num
    ... = 150 := by norm_num

end count_five_digit_numbers_condition_l105_105766


namespace intersection_and_union_range_for_a_l105_105838
-- Given sets A and B as defined:
def SetA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 4}
def SetB : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Proof Problem 1: With a = 0, find A ∩ B and A ∪ (complement of B in ℝ)
-- Expected answer: 
--   A ∩ B = {x | 0 ≤ x ∧ x ≤ 3}
--   A ∪ (complement of B in ℝ) = {x | x < -2 ∨ x ≥ 0}

-- Proof Problem 2: If A ∪ B = B, find the range for a
-- Expected answer: -2 ≤ a ≤ -1


def SetA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 4}
def SetB : Set ℝ := {x | x^2 - x - 6 ≤ 0}

theorem intersection_and_union (a : ℝ) (h : a = 0) :
  (SetA a) ∩ SetB = {x | 0 ≤ x ∧ x ≤ 3} ∧
  (SetA a) ∪ (compl SetB) = {x | x < -2 ∨ x ≥ 0} := 
by
  sorry

theorem range_for_a (a : ℝ) :
  (SetA a) ∪ SetB = SetB → -2 ≤ a ∧ a ≤ -1 := 
by
  sorry

end intersection_and_union_range_for_a_l105_105838


namespace gcd_smallest_value_l105_105348

theorem gcd_smallest_value {m n : ℕ} (h1 : 0 < m) (h2 : 0 < n) (h3 : Nat.gcd m n = 12) : Nat.gcd (8 * m) (18 * n) = 24 :=
by
  sorry

end gcd_smallest_value_l105_105348


namespace parabola_focus_distance_l105_105907

theorem parabola_focus_distance (p m : ℝ) (h1 : p > 0) (h2 : (2 - (-p/2)) = 4) : p = 4 := 
by
  sorry

end parabola_focus_distance_l105_105907


namespace good_jars_l105_105006

def original_cartons : Nat := 50
def jars_per_carton : Nat := 20
def less_cartons_received : Nat := 20
def damaged_jars_per_5_cartons : Nat := 3
def total_damaged_cartons : Nat := 1
def total_good_jars : Nat := 565

theorem good_jars (original_cartons jars_per_carton less_cartons_received damaged_jars_per_5_cartons total_damaged_cartons : Nat) :
  (original_cartons - less_cartons_received) * jars_per_carton 
  - (5 * damaged_jars_per_5_cartons + total_damaged_cartons * jars_per_carton) = total_good_jars := 
by 
  sorry

end good_jars_l105_105006


namespace quadratic_has_two_distinct_real_roots_l105_105086

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l105_105086


namespace central_cell_value_l105_105390

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l105_105390


namespace largest_band_members_l105_105602

def band_formation (m r x : ℕ) : Prop :=
  m < 100 ∧ m = r * x + 2 ∧ (r - 2) * (x + 1) = m ∧ r - 2 * x = 4

theorem largest_band_members : ∃ (r x m : ℕ), band_formation m r x ∧ m = 98 := 
  sorry

end largest_band_members_l105_105602


namespace max_value_real_roots_l105_105282

theorem max_value_real_roots (k x1 x2 : ℝ) :
  (∀ k, k^2 + 3 * k + 5 ≥ 0) →
  (x1 + x2 = k - 2) →
  (x1 * x2 = k^2 + 3 * k + 5) →
  (x1^2 + x2^2 ≤ 18) :=
by
  intro h1 h2 h3
  sorry

end max_value_real_roots_l105_105282


namespace boxes_in_cube_l105_105420

theorem boxes_in_cube (w l h : ℕ) (cube_side : ℕ) (n_boxes : ℕ)
  (h_w : w = 8) (h_l : l = 12) (h_h : h = 30)
  (h_cube_side : cube_side = Nat.lcm w (Nat.lcm l h))
  (volume_cube : cube_side^3 = 1728000)
  (volume_box : w * l * h = 2880)
  (h_n_boxes : n_boxes = volume_cube / volume_box) : n_boxes = 600 :=
by
  sorry

end boxes_in_cube_l105_105420


namespace max_volume_l105_105611

-- Conditions and Definitions
variables (r h: ℝ) (π : ℝ := Real.pi)
variable (r_pos : r > 0)
variable (h_pos : h > 0)
variable (cost_equation : 200 * π * r * h + 160 * π * r^2 = 12000 * π)

-- Derived Definitions
noncomputable def height_function (r : ℝ) : ℝ := 1 / (5 * r) * (300 - 4 * r^2)
noncomputable def volume_function (r : ℝ) : ℝ := (π / 5) * (300 * r - 4 * r^3)
noncomputable def volume_derivative (r : ℝ) : ℝ := (π / 5) * (300 - 12 * r^2)

# reduce ∀ r, 0 < r ∧ r < 5 * Real.sqrt 3 → volume_function r = (π / 5) * (300 * r - 4 * r^3)

-- Main Theorem Statement
theorem max_volume :
  volume_function 5 = 100 * π ∧ height_function 5 = 8 :=
by
  sorry

end max_volume_l105_105611


namespace polynomial_division_remainder_l105_105561

def dividend : Polynomial ℚ := 3 * X^5 + 7 * X^4 - 15 * X^3 - 35 * X^2 + 22 * X + 24
def divisor : Polynomial ℚ := X^3 + 5 * X^2 - 4 * X + 2
def remainder : Polynomial ℚ := -258 * X^2 + 186 * X - 50

theorem polynomial_division_remainder :
  (dividend % divisor) = remainder :=
  sorry

end polynomial_division_remainder_l105_105561


namespace problem_range_b_l105_105728

open Real

theorem problem_range_b : 
  (∀ b : ℝ, 
    (∀ x : ℝ, f x = -x^3 + b * x → (x^2 ≤ 4 → -2 ≤ x ∧ x ≤ 2)) →
    (∀ x : ℝ, (0 < x ∧ x < 1) → (f' b x = -3 * x^2 + b → (f' b x ≥ 0))) → (3 ≤ b ∧ b ≤ 4)) :=
by
  sorry

end problem_range_b_l105_105728


namespace central_cell_value_l105_105384

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l105_105384


namespace number_of_tickets_bought_l105_105123

theorem number_of_tickets_bought
    (cost_per_ticket : ℕ)
    (total_spent : ℕ)
    (h_cost : cost_per_ticket = 44)
    (h_total : total_spent = 308) :
    total_spent / cost_per_ticket = 7 :=
by {
    rw [h_cost, h_total],
    norm_num,
    sorry
}

end number_of_tickets_bought_l105_105123


namespace borya_wins_with_optimal_play_l105_105943

def points := 123

structure game_state :=
  (points: ℕ)
  (colors: list (option bool)) -- None for uncolored points, some bool where true is blue, false is red

noncomputable def player_wins (player1_first: bool) (current_state: game_state): bool :=
  sorry -- This definition will hold the logic of the game to check if a player has won

theorem borya_wins_with_optimal_play : 
  ∀(initial_state: game_state), 
  initial_state.points = points →
  (∀i < initial_state.colors.length, initial_state.colors.nth i ∈ [none]) →
  player_wins false initial_state :=
by
  sorry

end borya_wins_with_optimal_play_l105_105943


namespace investment_months_l105_105613

theorem investment_months (i_a i_b i_c a_gain total_gain : ℝ) (m : ℝ) :
  i_a = 1 ∧ i_b = 2 * i_a ∧ i_c = 3 * i_a ∧ a_gain = 6100 ∧ total_gain = 18300 ∧ m * i_b * (12 - m) + i_c * 3 * 4 = 12200 →
  a_gain / total_gain = i_a * 12 / (i_a * 12 + i_b * (12 - m) + i_c * 4) → m = 6 :=
by
  intros h1 h2
  obtain ⟨ha, hb, hc, hag, htg, h⟩ := h1
  -- proof omitted
  sorry

end investment_months_l105_105613


namespace middle_number_is_9_l105_105933

-- Define the problem conditions
variable (x y z : ℕ)

-- Lean proof statement
theorem middle_number_is_9 
  (h1 : x + y = 16)
  (h2 : x + z = 21)
  (h3 : y + z = 23)
  (h4 : x < y)
  (h5 : y < z) : y = 9 :=
by
  sorry

end middle_number_is_9_l105_105933


namespace percentage_error_in_area_l105_105984

theorem percentage_error_in_area (E : ℝ) (h : E > 0) :
  let E' := E * 1.02 in
  let A := E ^ 2 in
  let A' := (E') ^ 2 in
  let error := A' - A in
  (error / A) * 100 = 4.04 := 
by
  sorry

end percentage_error_in_area_l105_105984


namespace number_of_shares_is_25_l105_105167

def wife_weekly_savings := 100
def husband_monthly_savings := 225
def duration_months := 4
def cost_per_share := 50

def total_savings : ℕ :=
  (wife_weekly_savings * 4 * duration_months) + (husband_monthly_savings * duration_months)

def amount_invested := total_savings / 2

def number_of_shares := amount_invested / cost_per_share

theorem number_of_shares_is_25 : number_of_shares = 25 := by
  sorry

end number_of_shares_is_25_l105_105167


namespace central_cell_value_l105_105364

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l105_105364


namespace number_of_rectangles_in_4x4_grid_l105_105756

theorem number_of_rectangles_in_4x4_grid : 
  let n := 4 in
  let num_points_per_side := n + 1 in
  let vertical_lines := (num_points_per_side.choose 2) in
  let horizontal_lines := (num_points_per_side.choose 2) in
  vertical_lines * horizontal_lines = 100 :=
by
  let n := 4
  let num_points_per_side := n + 1
  have vertical_lines : num_points_per_side.choose 2 = 10 := 
    by rw Nat.choose_eq_factorial_div_factorial_sub_factorial; norm_num
  have horizontal_lines : num_points_per_side.choose 2 = 10 := 
    by rw Nat.choose_eq_factorial_div_factorial_sub_factorial; norm_num
  show vertical_lines * horizontal_lines = 100
  rw vertical_lines
  rw horizontal_lines
  norm_num

end number_of_rectangles_in_4x4_grid_l105_105756


namespace polynomial_coef_sum_l105_105357

theorem polynomial_coef_sum :
  ∃ (a b c d : ℝ), (∀ x : ℝ, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 14) :=
by
  sorry

end polynomial_coef_sum_l105_105357


namespace linear_function_increase_l105_105431

theorem linear_function_increase (k : ℝ) (h : (k - 2) > 0) : k > 2 :=
by
  exact h

end linear_function_increase_l105_105431


namespace problem_l105_105809

-- Define the conditions and values
theorem problem (
    h1 : ∀ x, f (π / 4) = f (5 * π / 4),
    h2 : 0 < ω,
    h3 : -π / 2 < φ ∧ φ < π / 2,
    h4 : α ∈ -3 * π / 4 ..-π / 4,
    h5 : f α = -4 / 5
) : 
    -- Proving the values
       ω = 1 
    ∧ φ = π / 4 
    ∧ sin α = -7 * sqrt 2 / 10 :=
sorry

end problem_l105_105809


namespace odds_against_horse_C_winning_l105_105545

theorem odds_against_horse_C_winning (odds_A : ℚ) (odds_B : ℚ) (odds_C : ℚ) 
  (cond1 : odds_A = 5 / 2) 
  (cond2 : odds_B = 3 / 1) 
  (race_condition : odds_C = 1 - ((2 / (5 + 2)) + (1 / (3 + 1))))
  : odds_C / (1 - odds_C) = 15 / 13 := 
sorry

end odds_against_horse_C_winning_l105_105545


namespace minimum_value_of_function_l105_105231

theorem minimum_value_of_function :
  ∃ x y : ℝ, 2 * x ^ 2 + 3 * x * y + 4 * y ^ 2 - 8 * x + y = 3.7391 := by
  sorry

end minimum_value_of_function_l105_105231


namespace logan_gas_expense_l105_105002

-- Definitions based on conditions:
def annual_salary := 65000
def rent_expense := 20000
def grocery_expense := 5000
def desired_savings := 42000
def new_income_target := annual_salary + 10000

-- The property to be proved:
theorem logan_gas_expense : 
  ∀ (gas_expense : ℕ), 
  new_income_target - desired_savings = rent_expense + grocery_expense + gas_expense → 
  gas_expense = 8000 := 
by 
  sorry

end logan_gas_expense_l105_105002


namespace first_tied_month_l105_105179

structure CumulativeHomeRuns where
  month : String
  mcgwire : Nat
  sosa : Nat
  griffey : Nat
  deriving Repr

def monthlyHomeRuns : List CumulativeHomeRuns := [
  { month := "March", mcgwire := 2, sosa := 1, griffey := 2 },
  { month := "April", mcgwire := 11, sosa := 6, griffey := 10 },
  { month := "May", mcgwire := 26, sosa := 14, griffey := 25 },
  { month := "June", mcgwire := 38, sosa := 36, griffey := 37 },
  { month := "July", mcgwire := 45, sosa := 42, griffey := 44 },
  { month := "August", mcgwire := 57, sosa := 55, griffey := 54 },
  { month := "September", mcgwire := 65, sosa := 66, griffey := 66 }
]

theorem first_tied_month (homeRuns : List CumulativeHomeRuns) :
  ∃ m, (homeRuns.getLast?).map (λ x => x.mcname == x.sosa ∧ x.sosa == x.griffey) = some true :=
by
  sorry

end first_tied_month_l105_105179


namespace correct_statements_for_function_l105_105686

theorem correct_statements_for_function :
  let f : ℝ → ℝ := λ x, x^3 - 3 * x^2,
      f' := λ x, 3 * x^2 - 6 * x in
  let increasing_intervals := (λ x, x < 0 ∨ x > 2),
      decreasing_interval := (λ x, 0 < x ∧ x < 2),
      is_maximum := (f 0 = 0),
      is_minimum := (f 2 = -4) in
  (¬ (∀ x, f' x > 0) ∧ ¬ (∀ x, f' x < 0) ∧
   (∀ x, increasing_intervals x ↔ (f' x > 0) ∧
    decreasing_interval x ↔ (f' x < 0)) ∧
   is_maximum ∧ is_minimum) → true := 
sorry

end correct_statements_for_function_l105_105686


namespace domain_of_function_l105_105518

theorem domain_of_function :
  {x : ℝ | ∃ (k : ℤ), (↑k * 2 * Real.pi + Real.pi / 6) ≤ x ∧ x ≤ (↑k * 2 * Real.pi + 5 * Real.pi / 6)} =
  {x : ℝ | (sin x - 1/2) ≥ 0} :=
sorry

end domain_of_function_l105_105518


namespace sum_first_2022_terms_l105_105700

/- Define the sequence and conditions -/
def a : ℕ → ℝ
| 0     := 0  -- not used, we start from n = 1
| 1     := 1
| (n+2) := let S_n := (finset.range (n+2)).sum a in
           let S_n_minus_1 := (finset.range (n+1)).sum a in
           (real.sqrt S_n) + (real.sqrt S_n_minus_1)

/- Define the sum of the sequence required -/
def S_n (n : ℕ) : ℝ :=
(finset.range (n+1)).sum a

def sequence_sum (n : ℕ) : ℝ :=
(finset.range n).sum (λ k, 1 / (a (k+1) * a (k+2)))

/- The theorem that sums up the first 2022 terms of the sequence -/
theorem sum_first_2022_terms :
  sequence_sum 2022 = 2022 / 4045 :=
sorry

end sum_first_2022_terms_l105_105700


namespace Carly_injured_week_miles_l105_105640

def week1_miles : ℕ := 2
def week2_miles : ℕ := week1_miles * 2 + 3
def week3_miles : ℕ := week2_miles * 9 / 7
def week4_miles : ℕ := week3_miles - 5

theorem Carly_injured_week_miles : week4_miles = 4 :=
  by
    sorry

end Carly_injured_week_miles_l105_105640


namespace polynomial_identity_factorization_l105_105279

theorem polynomial_identity_factorization :
  (∃ (p q r s t u : ℤ),
  ∀ x : ℤ,
  512 * x^3 + 125 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u) ∧
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 6410) :=
by
  use [8, 5, 0, 64, -40, 25]
  intro x
  split
  { calc
    512 * x^3 + 125 = (8 * x + 5) * (64 * x^2 - 40 * x + 25) : by ring
  }
  { sorry }

end polynomial_identity_factorization_l105_105279


namespace modular_inverse_of_15_mod_17_l105_105585

theorem modular_inverse_of_15_mod_17 : ∃ a : ℤ, 15 * a % 17 = 1 := by
  use 9
  have h : 0 ≤ 9 := by norm_num
  have h_mod : 15 * 9 % 17 = 1 := by
    calc
      15 * 9 % 17 = 135 % 17 : by norm_num
      _ = 1 : by norm_num
  exact h_mod
  sorry

end modular_inverse_of_15_mod_17_l105_105585


namespace Zach_more_points_l105_105994

theorem Zach_more_points (Zach_points : ℕ) (Ben_points : ℕ) (h₁ : Zach_points = 42) (h₂ : Ben_points = 21) : Zach_points - Ben_points = 21 :=
by
  rw [h₁, h₂]
  exact rfl

end Zach_more_points_l105_105994


namespace find_principal_l105_105989

noncomputable def principal (CI_diff SI_diff : ℝ) (R T : ℝ) := CI_diff / (R ^ T - R * T / 100)

theorem find_principal :
  let CI_diff := 0.21   -- CI - P (compound interest after removing principal)
  let SI_diff := 0.2    -- SI  - P (simple interest after removing principal)
  let R := 10 / 100     -- rate of interest (as decimal)
  let T := 2            -- time in years
  let P := 1300 in      -- the principal sum of money
  CI_diff - SI_diff = 0.01 →   -- given difference between CI and SI
  P = principal 13 R T :=
by
  intros
  have CI_diff := 13 / 1300
  have SI_diff := 0.2
  have R := 10.0 / 100.0
  have T := 2.0
  have diff := CI_diff * 1300 - SI_diff * 1300
  have h : 13 = diff
  sorry

end find_principal_l105_105989


namespace no_largest_integer_exists_l105_105967

/--
  Define a predicate to check whether an integer is a non-square.
-/
def is_non_square (n : ℕ) : Prop :=
  ¬ ∃ m : ℕ, m * m = n

/--
  Define the main theorem which states that there is no largest positive integer
  that cannot be expressed as the sum of a positive integral multiple of 36
  and a positive non-square integer less than 36.
-/
theorem no_largest_integer_exists : ¬ ∃ (n : ℕ), 
  ∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b < 36 ∧ is_non_square b →
  n ≠ 36 * a + b :=
sorry

end no_largest_integer_exists_l105_105967


namespace number_of_pawn_placements_l105_105427

theorem number_of_pawn_placements : let chessboard := List (List Bool) in
  (∀ row : chessboard, count(row, true) = 2) ∧ 
  (∀ col : List (List Bool), count(List.transpose(chessboard), true) = 2) → 
  count_valid_pawn_placements(chessboard) = 90 :=
sorry

end number_of_pawn_placements_l105_105427


namespace relational_proof_l105_105697

noncomputable def a : ℝ := Math.cos (Real.pi * 50 / 180) * Math.cos (Real.pi * 127 / 180) + 
                           Math.cos (Real.pi * 40 / 180) * Math.cos (Real.pi * 37 / 180)

noncomputable def b : ℝ := (Real.sqrt 2 / 2) * (Math.sin (Real.pi * 56 / 180) - 
                            Math.cos (Real.pi * 56 / 180))

noncomputable def c : ℝ := (1 - Math.tan (Real.pi * 39 / 180)^2) / 
                            (1 + Math.tan (Real.pi * 39 / 180)^2)

noncomputable def d : ℝ := (1 / 2) * (Math.cos (Real.pi * 80 / 180) - 
                            2 * Math.cos (Real.pi * 50 / 180)^2 + 1)

theorem relational_proof : a > c ∧ c > b ∧ b > d :=
by
  sorry

end relational_proof_l105_105697


namespace distinct_natural_sum_22_l105_105140

theorem distinct_natural_sum_22 (a b c d e f : ℕ) (h_distinct: list.nodup [a, b, c, d, e, f]) (h_sum: a + b + c + d + e + f = 22) : 
  (a, b, c, d, e, f) = (1, 2, 3, 4, 5, 7) ∨ 
  (a, b, c, d, e, f) = (1, 2, 3, 4, 7, 5) ∨ 
  (a, b, c, d, e, f) = (1, 2, 3, 5, 4, 7) ∨
  -- Continue for all permutations of (1, 2, 3, 4, 5, 7)
  sorry

end distinct_natural_sum_22_l105_105140


namespace central_cell_value_l105_105365

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l105_105365


namespace taxi_is_quicker_l105_105610

-- Define the speeds of the truck and the taxi
def truck_distance : ℝ := 2.1
def truck_time : ℝ := 1
def taxi_distance : ℝ := 10.5
def taxi_time : ℝ := 4

def truck_speed : ℝ := truck_distance / truck_time
def taxi_speed : ℝ := taxi_distance / taxi_time

theorem taxi_is_quicker :
  (taxi_speed / truck_speed) = 1.25 :=
by
  -- calculation skipped, leaves a placeholder for proof
  sorry

end taxi_is_quicker_l105_105610


namespace num_five_digit_palindromes_l105_105748

theorem num_five_digit_palindromes : 
  (∃ A B C : ℕ, 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) → 
  (nat.count (λ n, ∃ A B C : ℕ, n = A * 10001 + B * 1010 + C * 100 + B * 10 + A ∧ 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) (1, 99999)) = 900 := 
by 
  sorry

end num_five_digit_palindromes_l105_105748


namespace value_of_y_around_2_l105_105801

def regression_eq (x : ℝ) : ℝ := Real.log (4 * x - 20)

theorem value_of_y_around_2 (x : ℝ) (h : x = 30) : abs (regression_eq x - 2) < 1 := by
  sorry

end value_of_y_around_2_l105_105801


namespace min_cos_value_l105_105054

noncomputable def cos_min_value (x : ℝ) (hx : x ∈ set.Icc (Real.pi / 6) (2 * Real.pi / 3)) : ℝ :=
  Real.cos (x - Real.pi / 8)

theorem min_cos_value : 
  Infimum (set.image cos_min_value (set.Icc (Real.pi / 6) (2 * Real.pi / 3))) = Real.cos (13 * Real.pi / 24) := 
by
  sorry

end min_cos_value_l105_105054


namespace max_number_of_gangsters_l105_105946

/-- Given there are 36 criminal gangs with specific constraints on gangster memberships, 
we need to prove the maximum number of gangsters in Chicago. --/
def max_gangsters (n : ℕ) : Prop :=
  ∃ n, n ≤ 3 ^ 12 ∧ 
  (∀ g₁ g₂ : ℕ, sets_of_gangs g₁ ≠ sets_of_gangs g₂) ∧
  (∀ g : ℕ, no_hostile_gangs g) ∧ 
  (∀ g : ℕ, no_gang_no_member g)

-- Definitions for sets_of_gangs, no_hostile_gangs and no_gang_no_member should be made
-- according to the problem constraints
def sets_of_gangs : ℕ → set (set ℕ) := sorry
def no_hostile_gangs : ℕ → Prop := sorry
def no_gang_no_member : ℕ → Prop := sorry

theorem max_number_of_gangsters : max_gangsters 531441 :=
sorry

end max_number_of_gangsters_l105_105946


namespace treShaun_marker_ink_left_l105_105549

noncomputable def ink_left_percentage (marker_area : ℕ) (total_colored_area : ℕ) : ℕ :=
if total_colored_area >= marker_area then 0 else ((marker_area - total_colored_area) * 100) / marker_area

theorem treShaun_marker_ink_left :
  let marker_area := 3 * (4 * 4)
  let colored_area := (2 * (6 * 2) + 8 * 4)
  ink_left_percentage marker_area colored_area = 0 :=
by
  sorry

end treShaun_marker_ink_left_l105_105549


namespace compute_a_l105_105719

theorem compute_a (a : ℝ) (h : 2.68 * 0.74 = a) : a = 1.9832 :=
by
  -- Here skip the proof steps
  sorry

end compute_a_l105_105719


namespace y_alternates_l105_105466

def y : ℕ → ℤ 
def x : ℕ → ℤ 

axiom y_initial_terms : y 1 = 3 ∧ y 2 = 17 ∧ y 3 = 99 ∧ y 4 = 577

axiom y_even (s : ℕ) : y (2 * s) = y s ^ 2 + 2 * x s ^ 2
axiom y_odd (s : ℕ) : y (2 * s + 1) = 4 * x (2 * s) + 3 * y (2 * s)
axiom y_square_related (s : ℕ) : y s ^ 2 = 2 * x s ^ 2 + 1

theorem y_alternates (k : ℕ) (s : ℕ) :
    (y k = (2 * x s) ^ 2 + 1 ∧ k % 2 = 0) ∨ 
    (y k = [2 * (x s + y s)] ^ 2 - 1 ∧ k % 2 = 1) :=
sorry

end y_alternates_l105_105466


namespace sum_series_computation_l105_105204

theorem sum_series_computation :
  (∑ n in finset.Ico 2 (finset.Ico_infinity_start_idx), 
    (5 * (n:ℝ)^3 - 2 * n^2 - 2 * n + 2) / 
    (n * (n - 1) * (n^4 - n^3 + n^2 - n + 2)) 
  ) = 2 := 
sorry

end sum_series_computation_l105_105204


namespace length_of_PR_l105_105429

-- Define the entities and conditions
variables (x y : ℝ)
variables (xy_area : ℝ := 125)
variables (PR_length : ℝ := 10 * Real.sqrt 5)

-- State the problem in Lean
theorem length_of_PR (x y : ℝ) (hxy : x * y = 125) :
  x^2 + (125 / x)^2 = (10 * Real.sqrt 5)^2 :=
sorry

end length_of_PR_l105_105429


namespace problem_proof_l105_105911

-- Definitions based on the conditions
def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x^2 else x^2 - x - 3

-- The goal we need to prove
theorem problem_proof : f (1 / f 3) = 8 / 9 := by
  -- Proof goes here
  sorry

end problem_proof_l105_105911


namespace suraj_average_after_13th_innings_l105_105579

theorem suraj_average_after_13th_innings
  (A : ℝ)
  (h : (12 * A + 96) / 13 = A + 5) :
  (12 * A + 96) / 13 = 36 :=
by
  sorry

end suraj_average_after_13th_innings_l105_105579


namespace sum_of_ages_in_two_years_l105_105009

-- Definitions taken directly from conditions
def Fem_current_age : ℕ := 11
def Matt_current_age : ℕ := 4 * Fem_current_age

-- The problem to prove
theorem sum_of_ages_in_two_years (Fem_current_age = 11) (Matt_current_age = 4 * Fem_current_age) : 
  Fem_current_age + 2 + (Matt_current_age + 2) = 59 := 
by sorry

end sum_of_ages_in_two_years_l105_105009


namespace problem_1_problem_2_l105_105254

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := real.log (x + 1) / real.log a
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := real.log (1 / (1 - x)) / real.log a
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := 2 * f a x + g a x

theorem problem_1 (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : 
  ∀ x, F a x = 0 → (-1 < x ∧ x < 1) → x = 0 := sorry

theorem problem_2 (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (m : ℝ) :
  (∃! x, 0 ≤ x ∧ x < 1 ∧ F a x = m) →
  if a > 1 then m ≥ 0 else m ≤ 0 := sorry

end problem_1_problem_2_l105_105254


namespace quadratic_roots_distinct_l105_105080

-- Define the quadratic equation condition
def quadratic_eq : (ℝ → ℝ) :=
  λ x m => x^2 + m * x - 8

-- State the problem
theorem quadratic_roots_distinct (m : ℝ) : 
  let Δ := m^2 + 32 in Δ > 0 :=
  by
   -- We need this theorem to state that the equation always has distinct real roots
  let Δ := m^2 + 32
  sorry

end quadratic_roots_distinct_l105_105080


namespace total_food_per_day_l105_105661

def num_dogs : ℝ := 2
def food_per_dog_per_day : ℝ := 0.12

theorem total_food_per_day : (num_dogs * food_per_dog_per_day) = 0.24 :=
by sorry

end total_food_per_day_l105_105661


namespace ordering_of_log_values_l105_105825

def P := log 3 / log 2  -- log base 2 of 3
def Q := log 2 / log 3  -- log base 3 of 2
def R := log (log 2 / log 3) / log 2  -- log base 2 of (log base 3 of 2)

theorem ordering_of_log_values : R < Q ∧ Q < P :=
by
  -- Proof would go here
  sorry

end ordering_of_log_values_l105_105825


namespace correct_operation_l105_105982

theorem correct_operation (a b : ℝ) :
  (¬ (2 * a + 3 * b = 5 * a * b)) ∧
  (a^2 * a^4 = a^6) ∧
  (¬ ((2 * a^3)^3 = 8 * a^6)) ∧
  (¬ ((a - b)^2 = a^2 - 2 * a * b - b^2)) :=
by
  sorry

end correct_operation_l105_105982


namespace height_of_shorter_can_l105_105951

theorem height_of_shorter_can (h : ℝ) (r : ℝ) (taller_can_height : ℝ) (taller_can_radius : ℝ) (vol_shorter_can vol_taller_can : ℝ)
  (h_nonzero : h ≠ 0) 
  (height_relation : taller_can_height = 4 * h)
  (radius_relation : taller_can_radius = 5)
  (vol_relation1 : vol_shorter_can = π * r^2 * h)
  (vol_relation2 : vol_taller_can = π * taller_can_radius^2 * taller_can_height)
  (volumes_equal : vol_shorter_can = vol_taller_can) :
  r = 10 :=
by {
  sorry,
}

end height_of_shorter_can_l105_105951


namespace minimum_length_segment_AB_l105_105677

theorem minimum_length_segment_AB :
  let A_set := {p : ℝ × ℝ | p.snd^2 - 3 * p.fst^2 - 2 * p.snd * p.fst - 9 - 12 * p.fst = 0}
  let B_set := {p : ℝ × ℝ | p.fst^2 - 8 * p.snd + 23 + 6 * p.fst + p.snd^2 = 0}
  ∃ A B : ℝ × ℝ, A ∈ A_set ∧ B ∈ B_set ∧
  segment_length A B = (7 * Real.sqrt 10 / 10) - Real.sqrt 2 := sorry

end minimum_length_segment_AB_l105_105677


namespace range_of_m_l105_105251

theorem range_of_m (m : ℝ) (h : m = (- (sqrt 3) / 3) * (- 2 * sqrt 21)) : 5 < m ∧ m < 6 :=
by
  sorry

end range_of_m_l105_105251


namespace printer_ratio_l105_105538

-- Define the given conditions
def total_price_basic_computer_printer := 2500
def enhanced_computer_extra := 500
def basic_computer_price := 1500

-- The lean statement to prove the ratio of the price of the printer to the total price of the enhanced computer and printer is 1/3
theorem printer_ratio : ∀ (C_basic P C_enhanced Total_enhanced : ℕ), 
  C_basic + P = total_price_basic_computer_printer →
  C_enhanced = C_basic + enhanced_computer_extra →
  C_basic = basic_computer_price →
  C_enhanced + P = Total_enhanced →
  P / Total_enhanced = 1 / 3 := 
by
  intros C_basic P C_enhanced Total_enhanced h1 h2 h3 h4
  sorry

end printer_ratio_l105_105538


namespace exists_polynomial_l105_105836
-- Import the necessary Mathlib library

-- Define the naturals and integers
variable (f : ℕ → ℤ)

-- Define the conditions on f
def divides_condition (n m : ℕ) : Prop := n > m → (n - m) ∣ (f n - f m)

def polynomial_bound (P : ℕ → ℤ) : Prop := ∀ n : ℕ, |(f n)| < P n

-- The main theorem statement
theorem exists_polynomial (f: ℕ → ℤ) (P : ℕ → ℤ) 
  (h1 : ∀ n m : ℕ, n > m → (n - m) ∣ (f n - f m))
  (h2 : ∃ P : ℕ → ℤ, ∀ n : ℕ, |(f n)| < P n) :
  ∃ (Q : ℕ → ℤ), ∀ (n : ℕ), f n = Q n :=
sorry

end exists_polynomial_l105_105836


namespace white_trees_count_l105_105171

noncomputable def calculate_white_trees (total_trees pink_percent red_trees : ℕ) : ℕ :=
  total_trees - (total_trees * pink_percent / 100 + red_trees)

theorem white_trees_count 
  (h1 : total_trees = 42)
  (h2 : pink_percent = 100 / 3)
  (h3 : red_trees = 2) :
  calculate_white_trees total_trees pink_percent red_trees = 26 :=
by
  -- proof will go here
  sorry

end white_trees_count_l105_105171


namespace octahedron_labeling_ways_l105_105341

def is_relatively_prime (a b : ℕ) : Prop 
  := Nat.gcd a b = 1

def faces_share_edge (octahedron : List (List ℕ)) (i j : ℕ) : Prop 
  := ∃ (f1 f2 : List ℕ), List.nth octahedron i = some f1 ∧ List.nth octahedron j = some f2 ∧ (f1 ∩ f2).length = 2

def labeling_is_valid (labeling : List ℕ) (octahedron : List (List ℕ)) : Prop :=
  ∀ i j, i ≠ j → faces_share_edge octahedron i j → is_relatively_prime (labeling.nth_le i sorry) (labeling.nth_le j sorry)

def total_labelings (labelings : List (List ℕ)) : ℕ :=
  List.length (labelings.filter (λ labeling, labeling_is_valid labeling octahedron_faces))

def distinct_labelings (labelings : List (List ℕ)) : ℕ :=
  -- This hypothetically includes logic to account for rotational symmetries
  -- Here we place a placeholder for actual symmetry reduction logic
  total_labelings labelings / 24 -- Replace 24 with actual computation for symmetries

def octahedron_faces : List (List ℕ) := 
  -- Presume we have a list of faces (edges in a combinatorial structure) for the regular octahedron
  sorry

theorem octahedron_labeling_ways : distinct_labelings all_possible_labelings = 12 :=
  sorry

end octahedron_labeling_ways_l105_105341


namespace cos_seven_pi_over_six_l105_105633

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l105_105633


namespace determine_x_l105_105659

theorem determine_x (x : ℚ) : 
  x + 5 / 8 = 2 + 3 / 16 - 2 / 3 → 
  x = 43 / 48 := 
by
  intro h
  sorry

end determine_x_l105_105659


namespace total_number_of_admission_methods_l105_105184

/-
Problem: Admit 4 students into 3 universities, with each university admitting at least one student. The total number of different admission methods is 36.
-/

def C (n k : ℕ) : ℕ := (nat.choose n k)

noncomputable def total_admission_methods : ℕ :=
  let scenario1 := 3 * (C 4 2) * 2 in
  scenario1

theorem total_number_of_admission_methods :
  total_admission_methods = 36 :=
by
  sorry

end total_number_of_admission_methods_l105_105184


namespace correct_number_of_ways_to_draw_balls_l105_105242

def number_of_ways_to_draw_balls (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n - Nat.choose (2 * n) (n + 1)

theorem correct_number_of_ways_to_draw_balls (n : ℕ) :
  (∃ f : ℕ → ℕ, (∀ k < n, f k = 0) →
   (∀ k < n, f (2 * n - k) - f k ≤ n) →
   ∑ k in range (2 * n), f k) =
  number_of_ways_to_draw_balls n :=
sorry

end correct_number_of_ways_to_draw_balls_l105_105242


namespace min_value_log_am_gm_l105_105252

theorem min_value_log_am_gm (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : log 10 (2 ^ x) + log 10 (8 ^ y) = log 10 2) :
  ∃ (min_val : ℝ), min_val = 4 ∧ ∀ z, z = (1 / x) + (1 / (3 * y)) → z ≥ min_val :=
begin
  sorry
end

end min_value_log_am_gm_l105_105252


namespace cos_seven_pi_over_six_l105_105632

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l105_105632


namespace b3_value_sum_bn_l105_105260

variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q 

def initial_conditions : Prop :=
  a 1 = 2 ∧ a 4 = 16

def bn_def : Prop :=
  ∀ n, b n = n * a n

-- Part (1) Prove that b₃ = 24
theorem b3_value
  (q : ℝ)
  (h_geo : geometric_sequence a q)
  (h_init : initial_conditions)
  (h_bn : bn_def) :
  b 3 = 24 := by
sorry

-- Part (2) Prove the sum of the first n terms of bₙ
def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)

theorem sum_bn
  (q : ℝ)
  (h_geo : geometric_sequence a q)
  (h_init : initial_conditions)
  (h_bn : bn_def) :
  ∀ n, T b n = (n - 1) * 2^(n + 1) + 2 := by
sorry

end b3_value_sum_bn_l105_105260


namespace corveus_lack_of_sleep_l105_105209

def daily_sleep_actual : ℕ := 4
def daily_sleep_recommended : ℕ := 6
def days_in_week : ℕ := 7

theorem corveus_lack_of_sleep : (daily_sleep_recommended - daily_sleep_actual) * days_in_week = 14 := 
by 
  sorry

end corveus_lack_of_sleep_l105_105209


namespace contest_questions_l105_105782

noncomputable def num_questions (N : ℕ) (Petya_correct : ℕ) (Vasya_correct : ℕ) (a b c : ℕ) : Prop :=
  Petya_correct = 10 ∧
  Vasya_correct = 13 ∧
  Petya_correct < a ∧ Petya_correct < b ∧ Petya_correct < c ∧
  Vasya_correct > a ∧ Vasya_correct > b ∧ Vasya_correct > c ∧
  11 ≤ a ∧ a ≤ 12 ∧
  11 ≤ b ∧ b ≤ 12 ∧
  11 ≤ c ∧ c ≤ 12 ∧
  5 * N = 3 * (a + b + c) + Petya_correct + Vasya_correct ∧
  4 * N = Petya_correct + Vasya_correct + a + b + c

theorem contest_questions : ∃ N, num_questions N 10 13 11 11 11 ∧ N = 14 := by
  let N : ℕ := 14
  let Petya_correct : ℕ := 10
  let Vasya_correct : ℕ := 13
  let a : ℕ := 11
  let b : ℕ := 11
  let c : ℕ := 11
  have : num_questions N Petya_correct Vasya_correct a b c := by
    unfold num_questions
    repeat { split }
    <|> exact rfl
    <|> exact nat.succ_pos' _
    <|> dec_trivial
    done
  use N
  split
  <|> assumption
  <|> exact rfl
  done

end contest_questions_l105_105782


namespace proportion_no_MATH_l105_105525

theorem proportion_no_MATH (A M T E I C S : Type) [Fintype A] [Fintype M] [Fintype T] [Fintype E] [Fintype I] [Fintype C] [Fintype S]
  (h1 : A = M = T = E = I = C = S)
  (h2 : M =! A ∧ A =! T)
  (h3 : multiplicity M "MATHEMATICS" = 2 ∧ multiplicity A "MATHEMATICS" = 2 ∧ multiplicity T "MATHEMATICS" = 2) :
  (494 : ℚ) / 495 = 
  1 - (((fintype.card 8) ! / (fintype.card 2)! ^ 2) / ((fintype.card 11) ! / (fintype.card 2)! ^ 3)) :=
sorry

end proportion_no_MATH_l105_105525


namespace central_cell_value_l105_105383

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l105_105383


namespace pizza_order_cost_l105_105884

def base_cost_per_pizza : ℕ := 10
def cost_per_topping : ℕ := 1
def topping_count_pepperoni : ℕ := 1
def topping_count_sausage : ℕ := 1
def topping_count_black_olive_and_mushroom : ℕ := 2
def tip : ℕ := 5

theorem pizza_order_cost :
  3 * base_cost_per_pizza + (topping_count_pepperoni * cost_per_topping) + (topping_count_sausage * cost_per_topping) + (topping_count_black_olive_and_mushroom * cost_per_topping) + tip = 39 := by
  sorry

end pizza_order_cost_l105_105884


namespace total_fencing_needed_l105_105543

def width1 : ℕ := 4
def length1 : ℕ := 2 * width1 - 1

def length2 : ℕ := length1 + 3
def width2 : ℕ := width1 - 2

def width3 : ℕ := (width1 + width2) / 2
def length3 : ℚ := (length1 + length2) / 2

def perimeter (w l : ℚ) : ℚ := 2 * (w + l)

def P1 : ℚ := perimeter width1 length1
def P2 : ℚ := perimeter width2 length2
def P3 : ℚ := perimeter width3 length3

def total_fence : ℚ := P1 + P2 + P3

theorem total_fencing_needed : total_fence = 69 := 
  sorry

end total_fencing_needed_l105_105543


namespace tangent_line_equation_no_extreme_values_range_of_k_l105_105269

section problem

variable (a : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ) (phi : ℝ → ℝ) 

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (x + a)
noncomputable def phi (x : ℝ) : ℝ := f x * g x

-- (1) Proving the equation of the tangent line at the point (1, f(1)) is y - e*x = 0.
theorem tangent_line_equation : ∃ (y : ℝ), y - f 1 * x = 0 := 
by 
  have f1 := Real.exp 1
  have f1' := Real.exp 1
  exact sorry

-- (2) Proving that φ(x) has no extreme values.
theorem no_extreme_values : ∀ x, ∃ m, (phi x) > m := 
by 
  have phi' (x : ℝ) : ℝ := f x * (1 / (x + a) + (Real.log (x + a))) 
  exact sorry

-- (3) Proving the range of values for k when a = 0.
theorem range_of_k (k : ℝ) : ∃ k, (k ∈ Set.Ici (-Real.exp 1)) := 
by 
  assume s t (h : s > t)
  have h' := Real.log s - Real.log t > k * (1 / (Real.exp s) - 1 / (Real.exp t))
  exact sorry

end problem

end tangent_line_equation_no_extreme_values_range_of_k_l105_105269


namespace magnitude_subtract_vectors_l105_105303

noncomputable def vector_a : ℝ × ℝ := (2, 3)
noncomputable def vector_b (t : ℝ) : ℝ × ℝ := (t, -1)

theorem magnitude_subtract_vectors :
  ∀ t : ℝ, (vector_a.fst * t + vector_a.snd * (-1) = 0) →
  real.sqrt ((vector_a.fst - 2 * (vector_b t).fst)^2 + (vector_a.snd - 2 * (vector_b t).snd)^2) = real.sqrt 26 :=
by
  intro t h
  sorry

end magnitude_subtract_vectors_l105_105303


namespace balance_four_heartsuits_with_five_circles_l105_105998

variables (x y z : ℝ)

-- Given conditions
axiom condition1 : 4 * x + 3 * y = 12 * z
axiom condition2 : 2 * x = y + 3 * z

-- Statement to prove
theorem balance_four_heartsuits_with_five_circles : 4 * y = 5 * z :=
by sorry

end balance_four_heartsuits_with_five_circles_l105_105998


namespace students_taking_german_l105_105780

def total_students : ℕ := 94
def students_french : ℕ := 41
def students_both : ℕ := 9
def students_neither : ℕ := 40

theorem students_taking_german : ℕ :=
  let students_at_least_one := total_students - students_neither
  let students_only_french := students_french - students_both
  let students_only_german := students_at_least_one - students_only_french - students_both
  students_only_german + students_both = 22 := by sorry

end students_taking_german_l105_105780


namespace z_neq_5_for_every_k_l105_105653

theorem z_neq_5_for_every_k (z : ℕ) (h₁ : z = 5) :
  ¬ (∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ (∃ m, n ^ 9 % 10 ^ k = z * (10 ^ m))) :=
by
  intro h
  sorry

end z_neq_5_for_every_k_l105_105653


namespace central_cell_value_l105_105376

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l105_105376


namespace no_real_roots_ffx_l105_105261

theorem no_real_roots_ffx 
  {a b c : ℝ} 
  (h : (b - 1)^2 - 4 * a * c < 0) 
  (f : ℝ → ℝ := λ x, a * x^2 + b * x + c) :
  ¬ ∃ x : ℝ, f(f x) = x :=
sorry

end no_real_roots_ffx_l105_105261


namespace smallest_number_is_109871748_l105_105563

noncomputable def lcm_conditions : ℕ := Nat.lcm 29 (Nat.lcm 53 (Nat.lcm 37 (Nat.lcm 41 47)))

def meets_conditions (x : ℕ) : Prop :=
  (x + 11) % 29 = 0 ∧ 
  (x + 11) % 53 = 0 ∧ 
  (x + 11) % 37 = 0 ∧ 
  (x + 11) % 41 = 0 ∧ 
  (x + 11) % 47 = 0

theorem smallest_number_is_109871748 : ∃ x, meets_conditions(x) ∧ x = 109871748 :=
by
  let x := 109871748
  have h : meets_conditions x := by
    unfold meets_conditions
    simp [lcm_conditions]
    repeat { apply Nat.mod_eq_zero_of_dvd; apply Nat.dvd_of_mod_eq_zero; sorry }
  use x
  split
  exact h
  refl

end smallest_number_is_109871748_l105_105563


namespace math_expectation_xi_eq_27_div_10_l105_105154

/-- 
  Define the probability of selecting first-class parts from production lines A and B respectively.
  Given the total parts extracted and statistical table data.
-/
def probability_first_class_parts_A : ℚ := 75 / 100
def probability_first_class_parts_B : ℚ := 48 / 80

/-- Define the probability distribution table for the number of first-class parts among 4 selected parts. -/
def prob_distribution_xi : list (ℚ × ℚ) :=
  [ (0, 4 / 400),
    (1, 36 / 400),
    (2, 117 / 400),
    (3, 162 / 400),
    (4, 81 / 400) ]

/-- Calculate the mathematical expectation E(ξ) for the number of first-class parts -/
def expectation_xi : ℚ :=
  (0 * (4 / 400)) + (1 * (36 / 400)) + (2 * (117 / 400)) + (3 * (162 / 400)) + (4 * (81 / 400))

theorem math_expectation_xi_eq_27_div_10 : expectation_xi = 27 / 10 :=
by
  -- This skipps the actual computational proof, which should be handled separately.
  sorry

end math_expectation_xi_eq_27_div_10_l105_105154


namespace total_people_at_dance_l105_105070

variable (B G T : ℕ)
variable (R : ℚ := 3/4)
variable (G_count : G = 60)
variable (T_percent : ℚ := 0.20)

theorem total_people_at_dance :
  (G = 60) →
  (B = (3 * 60 / 4)) →
  (T = (0.20 * B).toNat) →
  (B + G + T = 114) :=
by
  intros hG hB hT
  rw [hG, hB, hT]
  sorry

end total_people_at_dance_l105_105070


namespace max_area_of_cone_section_l105_105286

noncomputable def max_cone_section_area (h : ℝ) (V : ℝ) : ℝ :=
  if h = 1 ∧ V = Mathlib.pi then 2 else 0

theorem max_area_of_cone_section :
  max_cone_section_area 1 Mathlib.pi = 2 :=
by
  sorry

end max_area_of_cone_section_l105_105286


namespace focus_of_parabola_l105_105227

theorem focus_of_parabola (f : ℝ) : 
  (∀ (x: ℝ), x^2 + ((- 1 / 16) * x^2 - f)^2 = ((- 1 / 16) * x^2 - (f + 8))^2) 
  → f = -4 :=
by
  intro h
  sorry

end focus_of_parabola_l105_105227


namespace count_five_digit_palindromes_l105_105750

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 = d5 ∧ d2 = d4

theorem count_five_digit_palindromes : 
  (finset.filter is_five_digit_palindrome (finset.Icc 10000 99999)).card = 900 :=
by
  sorry

end count_five_digit_palindromes_l105_105750


namespace find_a_l105_105724

theorem find_a (a : ℝ) (y : ℝ → ℝ) (y' : ℝ → ℝ) 
    (h_curve : ∀ x, y x = x^4 + a * x^2 + 1)
    (h_derivative : ∀ x, y' x = (4 * x^3 + 2 * a * x))
    (h_tangent_slope : y' (-1) = 8) :
    a = -6 :=
by
  -- To be proven
  sorry

end find_a_l105_105724


namespace find_FC_l105_105245

theorem find_FC (DC : ℝ) (CB : ℝ) (AB AD ED FC : ℝ) 
  (h1 : DC = 9) 
  (h2 : CB = 10) 
  (h3 : AB = (1/3) * AD) 
  (h4 : ED = (3/4) * AD) 
  (h5 : FC = 14.625) : FC = 14.625 :=
by sorry

end find_FC_l105_105245


namespace last_year_price_l105_105922

theorem last_year_price (a : ℝ) : 
  let last_year_price := a / 0.8 
  (price drop : ℝ) : last_year_price * (1 - 0.2) = a := 
by
  sorry

end last_year_price_l105_105922


namespace central_cell_value_l105_105389

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l105_105389


namespace value_of_a_max_value_of_k_l105_105290

noncomputable def f (x a : ℝ) : ℝ := x * (a + Real.log x)

-- Condition: The function f(x) has a minimum value of -e^{-2}
axiom min_value_f (a : ℝ) : ∃ x : ℝ, f x a = -Real.exp (-2)

-- Question 1: Prove that the real number a = 1
theorem value_of_a : ∃ (a : ℝ), (∀ (x : ℝ), f x a = -Real.exp (-2)) → a = 1 :=
by sorry

-- Condition: k is an integer
axiom k_is_integer (k : ℤ): Prop

-- Condition: k < f(x) / (x - 1) for any x > 1
axiom k_condition (k : ℤ): ∀ x > 1, k < f x 1 / (x - 1)

-- Question 2: Prove that the maximum value of k is 3
theorem max_value_of_k : ∀ (k : ℤ), (k_is_integer k) → (∀ x > 1, k < f x 1 / (x - 1)) → k ≤ 3 :=
by sorry

end value_of_a_max_value_of_k_l105_105290


namespace max_diff_set_l105_105557

def set := {-20, -5, 0, 3, 7, 15}
def max_difference (s : set ℤ) : ℤ :=
  if hs : s.nonempty then
    let a := s.max' hs
    let b := s.min' hs
    a - b
  else 0

theorem max_diff_set : max_difference set = 35 := 
by {
  have h1: set.nonempty := by simp [set],
  have h2: set.max' h1 = 15 := by simp [set.max'],
  have h3: set.min' h1 = -20 := by simp [set.min'],
  simp [max_difference, h2, h3],
  sorry
}

end max_diff_set_l105_105557


namespace simplify_expression_l105_105501

variable (p : ℤ)

-- Defining the given expression
def initial_expression : ℤ := ((5 * p + 1) - 2 * p * 4) * 3 + (4 - 1 / 3) * (6 * p - 9)

-- Statement asserting the simplification
theorem simplify_expression : initial_expression p = 13 * p - 30 := 
sorry

end simplify_expression_l105_105501


namespace geometric_sequence_sum_l105_105932

theorem geometric_sequence_sum (n : ℕ) (a r : ℚ) (h_a : a = 1) (h_r : r = 1 / 2) (h_sum : ∑ i in Finset.range n, a * r ^ i = 31 / 16) : n = 5 :=
by
  sorry

end geometric_sequence_sum_l105_105932


namespace parabola_intersections_l105_105647

theorem parabola_intersections :
  let a_set := {-3, -2, -1, 0, 1, 2, 3}
  let b_set := {-4, -3, -2, -1, 1, 2, 3, 4}
  let n_parabolas := Set.card a_set * Set.card b_set
  let n_pairs := Nat.choose n_parabolas 2
  let non_intersecting_pairs := Set.card a_set * (Nat.choose 4 2 + Nat.choose 4 2)
  let intersecting_pairs := n_pairs - non_intersecting_pairs
  let intersection_points := 2 * intersecting_pairs
  intersection_points = 2912 :=
by 
  let a_set := {-3, -2, -1, 0, 1, 2, 3}
  let b_set := {-4, -3, -2, -1, 1, 2, 3, 4}
  let n_parabolas := Set.card a_set * Set.card b_set
  let n_pairs := Nat.choose n_parabolas 2
  let non_intersecting_pairs := Set.card a_set * (Nat.choose 4 2 + Nat.choose 4 2)
  let intersecting_pairs := n_pairs - non_intersecting_pairs
  let intersection_points := 2 * intersecting_pairs
  have h_card_a : Set.card a_set = 7 := by sorry
  have h_card_b : Set.card b_set = 8 := by sorry
  have h_n_parabolas : n_parabolas = 56 := by simp only [Set.card, h_card_a, h_card_b]
  have h_n_pairs : n_pairs = 1540 := by simp only [Nat.choose, h_n_parabolas]
  have h_non_intersecting_pairs : non_intersecting_pairs = 84 := by simp only [Set.card, Nat.choose, h_card_a]
  have h_intersecting_pairs : intersecting_pairs = 1456 := by simp only [Nat.sub, h_n_pairs, h_non_intersecting_pairs]
  have h_intersection_points : intersection_points = 2912 := by simp only [Nat.mul, h_intersecting_pairs]
  exact h_intersection_points

end parabola_intersections_l105_105647


namespace team_selection_correct_l105_105862

-- Define the basic sets and experienced members
def boys := finset.range 7  -- 7 boys
def girls := finset.range 10 -- 10 girls
def experienced_boy := 0
def experienced_girl := 0

-- Define the conditions for the selection
def n_select_boy := boys.erase experienced_boy
def n_select_girl := girls.erase experienced_girl

-- Binomial coefficient
def binomial (n k : ℕ) : ℕ := nat.choose n k

-- Definition of the number of ways to select 3 boys and 3 girls including the experienced members
def ways_to_select_team : ℕ :=
  1 * (binomial 6 2) * 1 * (binomial 9 2)

theorem team_selection_correct : 
  ways_to_select_team = 540 :=
by
  sorry

end team_selection_correct_l105_105862


namespace number_of_three_digit_multiples_of_7_l105_105330

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l105_105330


namespace remainder_of_3_pow_99_plus_5_mod_9_l105_105131

theorem remainder_of_3_pow_99_plus_5_mod_9 : (3 ^ 99 + 5) % 9 = 5 := by
  -- Here we state the main goal
  sorry -- Proof to be filled in

end remainder_of_3_pow_99_plus_5_mod_9_l105_105131


namespace triangle_area_l105_105223

-- Define the points A, B, and C
def A : ℝ × ℝ × ℝ := (0, 7, 10)
def B : ℝ × ℝ × ℝ := (-1, 6, 6)
def C : ℝ × ℝ × ℝ := (-4, 9, 6)

-- Helper function to calculate the distance between two points in R^3
def dist (P Q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 + (Q.3 - P.3)^2)

-- Define the side lengths of the triangle
def AB : ℝ := dist A B
def AC : ℝ := dist A C
def BC : ℝ := dist B C

-- Prove the area of the triangle is 9
theorem triangle_area : 0.5 * AB * BC = 9 := 
by {
  sorry
}

end triangle_area_l105_105223


namespace sum_of_dihedral_angles_l105_105523

theorem sum_of_dihedral_angles {n : ℕ} 
  (halves : Fin n → Plane) (l : Line)
  (share_edge : ∀ i : Fin n, LineContainsPlane l (halves i))
  (spans_space : ∑ i in Finset.range n, PlaneAngle (halves i) (halves ((i + 1) % n)) = 2 * Real.pi) :
  ∑ i in Finset.range n, PlaneAngle (halves i) (halves ((i + 1) % n)) = 2 * Real.pi :=
begin
  sorry
end

end sum_of_dihedral_angles_l105_105523


namespace cricket_team_players_l105_105107

-- Define conditions 
def non_throwers (T P : ℕ) : ℕ := P - T
def left_handers (N : ℕ) : ℕ := N / 3
def right_handers_non_thrower (N : ℕ) : ℕ := 2 * N / 3
def total_right_handers (T R : ℕ) : Prop := R = T + right_handers_non_thrower (non_throwers T R)

-- Assume conditions are given
variables (P N R T : ℕ)
axiom hT : T = 37
axiom hR : R = 49
axiom hNonThrower : N = non_throwers T P
axiom hRightHanders : right_handers_non_thrower N = R - T

-- Prove the total number of players is 55
theorem cricket_team_players : P = 55 :=
by
  sorry

end cricket_team_players_l105_105107


namespace distance_between_axes_of_symmetry_l105_105906

theorem distance_between_axes_of_symmetry :
  ∀ x : ℝ, y = 3 * sin (2 * x + π / 4) → distance_between_axes_of_symmetry y = π / 2 :=
sorry

end distance_between_axes_of_symmetry_l105_105906


namespace sum_f_equals_2016_l105_105159

def f (x : ℕ) : ℝ := sorry -- Placeholder for the function definition

noncomputable theory

-- Conditions from the problem
axiom f_add (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) : f (m + n) = f m * f n
axiom f_one : f 1 = 2

-- Statement to be proven
theorem sum_f_equals_2016 :
  (finset.range 1008).sum (λ i, f (2 * (i + 1)) / f (2 * i + 1)) = 2016 :=
sorry

end sum_f_equals_2016_l105_105159


namespace three_digit_multiples_of_7_l105_105327

theorem three_digit_multiples_of_7 : 
  ∃! n : ℕ, (n = 128) ∧ (∀ k, (100 ≤ 7 * k ∧ 7 * k ≤ 999) ↔ (15 ≤ k ∧ k ≤ 142)) :=
begin
  sorry
end

end three_digit_multiples_of_7_l105_105327


namespace three_digit_multiples_of_7_l105_105316

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l105_105316


namespace min_n_of_inequality_l105_105714

theorem min_n_of_inequality : 
  ∀ (n : ℕ), (1 ≤ n) → (1 / n - 1 / (n + 1) < 1 / 10) → (n = 3 ∨ ∃ (k : ℕ), k ≥ 3 ∧ n = k) :=
by
  sorry

end min_n_of_inequality_l105_105714


namespace total_amount_spent_l105_105194

variable (boxes packs_per_box tissues_per_pack : ℕ)
variable (cost_per_tissue : ℕ)

-- Define the conditions:
def num_boxes := 10
def packs_per_box := 20
def tissues_per_pack := 100
def cost_per_tissue := 5 -- cents

-- Define the proof problem:
theorem total_amount_spent :
  let total_packs := num_boxes * packs_per_box in
  let total_tissues := total_packs * tissues_per_pack in
  let total_cost := total_tissues * cost_per_tissue in
  total_cost = 1000 * 100 := -- converting $10.00 to cents
by
  sorry

end total_amount_spent_l105_105194


namespace installation_cost_l105_105879

noncomputable def labelledPrice := 17500 / 0.80
noncomputable def transportCost := 125
noncomputable def TCP (IC : ℝ) := labelledPrice + transportCost + IC
noncomputable def SP (TCP : ℝ) := 1.10 * TCP

theorem installation_cost :
  ∃ IC : ℝ, SP (TCP IC) = 24475 ∧ (IC = 250) :=
by
  sorry

end installation_cost_l105_105879


namespace remainder_of_division_l105_105904

theorem remainder_of_division :
  ∀ (L S R : ℕ), 
  L = 1575 → 
  L - S = 1365 → 
  S * 7 + R = L → 
  R = 105 :=
by
  intros L S R h1 h2 h3
  sorry

end remainder_of_division_l105_105904


namespace side_length_is_fraction_l105_105484

noncomputable def side_length_of_square (DE EF DF t h : ℝ) : Prop :=
  (triangle_area: 0.5 * DE * EF = 30) ∧
  (hypotenuse: DE^2 + EF^2 = DF^2) ∧
  (altitude: 0.5 * DF * h = 30) ∧
  (similar_triangles: t / DE = t / EF ∧ (h - t) / DF = t / EF) ∧
  (h := 60 / DF) ∧
  (solving_for_t: t = 60 / (13 + 60/13)) 

theorem side_length_is_fraction (DE EF DF : ℝ) (h t : ℝ) (h_TRUE : h = 60 / DF) (t_TRUE : t = 780 / 229):
  side_length_of_square DE EF DF t h → t = 780 / 229 :=
by
  intros hyp
  rw [t_TRUE]
  rw [h_TRUE]
  have t_eq_freq := solving_for_t
  sorry

end side_length_is_fraction_l105_105484


namespace min_value_expression_l105_105128

theorem min_value_expression : ∀ x : ℝ, (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 ≥ 3625 :=
by
  sorry

end min_value_expression_l105_105128


namespace seq_inequality_l105_105075

def a : ℕ → ℝ
| 0     := 2
| (n+1) := a n ^ 2 - a n + 1

theorem seq_inequality : 
  1 - 1 / 2003 ^ 2003 < (∑ i in Finset.range 2003, 1 / a (i + 1)) ∧ 
  (∑ i in Finset.range 2003, 1 / a (i + 1)) < 1 := 
sorry

end seq_inequality_l105_105075


namespace required_moles_of_NaHCO₃_l105_105311

-- Definition of the balanced chemical equation
def balanced_chemical_equation : Prop :=
  ∀ (NaHCO₃ HCl NaCl H₂O CO₂ : ℕ), 
  NaHCO₃ + HCl = 1 ∧
  NaCl + H₂O + CO₂ = 1

-- Proposition to prove the required moles of NaHCO₃
theorem required_moles_of_NaHCO₃ 
  (NaHCO₃ HCl NaCl H₂O CO₂ : ℕ) 
  (h : balanced_chemical_equation NaHCO₃ HCl NaCl H₂O CO₂) :
  (HCl = 2) → (NaHCO₃ = 2) ∧ (NaCl = 2) ∧ (H₂O = 2) ∧ (CO₂ = 2) :=
by 
  sorry

end required_moles_of_NaHCO₃_l105_105311


namespace probability_odd_sum_probability_even_product_l105_105952
open Classical

noncomputable def number_of_possible_outcomes : ℕ := 36
noncomputable def number_of_odd_sum_outcomes : ℕ := 18
noncomputable def number_of_even_product_outcomes : ℕ := 27

theorem probability_odd_sum (n : ℕ) (m_1 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_1 = number_of_odd_sum_outcomes) : (m_1 : ℝ) / n = 1 / 2 :=
by
  sorry

theorem probability_even_product (n : ℕ) (m_2 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_2 = number_of_even_product_outcomes) : (m_2 : ℝ) / n = 3 / 4 :=
by
  sorry

end probability_odd_sum_probability_even_product_l105_105952


namespace largest_multiple_of_12_negation_greater_than_150_l105_105960

theorem largest_multiple_of_12_negation_greater_than_150 : 
  ∃ (k : ℤ), (k * 12 = 144) ∧ (-k * 12 > -150) :=
by
  -- Definitions and conditions
  let multiple_12 (k : ℤ) := k * 12
  have condition : -multiple_12 (-12) > -150 := by sorry
  existsi -12
  exact ⟨rfl, condition⟩

end largest_multiple_of_12_negation_greater_than_150_l105_105960


namespace parabola_vertex_l105_105902

theorem parabola_vertex :
  ∀ (x : ℝ), ∃ (y : ℝ), y = 3 * (x + 1)^2 + 4 → (x = -1 ∧ y = 4) :=
begin
  intro x,
  use 4,
  intro h,
  split,
  { sorry },
  { exact h }
end

end parabola_vertex_l105_105902


namespace infinite_relatively_prime_set_l105_105024

theorem infinite_relatively_prime_set : 
  ∃ S : Set ℕ, (∀ n ∈ S, ∃ k : ℕ, n = 2^k - 3) ∧ 
               (∀ a b ∈ S, a ≠ b → Nat.gcd a b = 1) ∧ 
               S.Infinite := 
by 
  sorry

end infinite_relatively_prime_set_l105_105024


namespace sufficient_not_necessary_of_and_false_or_true_l105_105271

variables (p q : Prop)

theorem sufficient_not_necessary_of_and_false_or_true :
  (¬(p ∧ q) → (p ∨ q)) ∧ ((p ∨ q) → ¬(¬(p ∧ q))) :=
sorry

end sufficient_not_necessary_of_and_false_or_true_l105_105271


namespace simplified_expression_eq_l105_105030

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  let y := 2 * x
  let sin_y := 2 * Real.sin x * Real.cos x
  let cos_y := 2 * Real.cos x * Real.cos x - 1
  (1 - sin_y + cos_y) / (1 + sin_y - cos_y)

theorem simplified_expression_eq (x : ℝ) :
  simplify_expression x = Real.csc x - 1 :=
by
  sorry

end simplified_expression_eq_l105_105030


namespace ratio_square_to_triangle_l105_105546

variables (a n : ℝ) (ADF FEC : Triangle)

-- Definitions based on the conditions
def is_right_triangle (T : Triangle) : Prop := sorry -- Definition of a right triangle
def divides_into_square_and_triangles (T : Triangle) (P : Point) : Prop := sorry -- Point P divides T into a square and two right triangles
def area (T : Triangle) : ℝ := sorry -- Definition of area of triangle
def parallel_lines (P : Point) (line1 line2 : Line) : Prop := sorry -- Definition of parallel lines through point P

-- Conditions
axiom (T : Triangle) (P : Point) (line1 line2 : Line) 
  (cond1 : is_right_triangle T)
  (cond2 : divides_into_square_and_triangles T P)
  (cond3 : parallel_lines P line1 line2)
  (cond4 : area ADF = a)
  (cond5 : area FEC = n * a)

-- Goal: Prove the ratio of the area of the square to the area of the larger small triangle is 1/n
theorem ratio_square_to_triangle (h : n > 0) : 
  (area square / area FEC) = 1 / n := 
sorry

end ratio_square_to_triangle_l105_105546


namespace even_function_among_options_l105_105833

-- Define the function f(x) = 1 / (x^3 + 1)
def f (x : ℝ) : ℝ := 1 / (x^3 + 1)

-- Define the given functions
def A (x : ℝ) : ℝ := f (x + 1)
def B (x : ℝ) : ℝ := f (2 * x)
def C (x : ℝ) : ℝ := f (x - 1)
def D (x : ℝ) : ℝ := f (x^2)

-- Define the statement that D is the even function among A, B, C, D
theorem even_function_among_options : 
  ∀ x : ℝ, D x = D (-x) ∧ (A x ≠ A (-x) ∨ B x ≠ B (-x) ∨ C x ≠ C (-x)) := sorry

end even_function_among_options_l105_105833


namespace angle_QOR_126_l105_105706

-- Definitions for the conditions
variables {P Q R O : Type} [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited O]

-- Conditions of the problem
def isosceles_triangle (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R] : Prop :=
  ∃ (PQ QR : ℝ), PQ = QR ∧ angle P Q R = 108

def point_inside_triangle (O P Q R : Type) [Inhabited O] [Inhabited P] [Inhabited Q] [Inhabited R] : Prop :=
  angle O R P = 30 ∧ angle O P R = 24

-- The statement of the theorem
theorem angle_QOR_126 (P Q R O : Type) [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited O]
  (h1 : isosceles_triangle P Q R) (h2 : point_inside_triangle O P Q R) : 
  angle Q O R = 126 :=
sorry

end angle_QOR_126_l105_105706


namespace cos_seven_pi_over_six_l105_105630

open Real

theorem cos_seven_pi_over_six : cos (7 * π / 6) = - (sqrt 3 / 2) := 
by
  sorry

end cos_seven_pi_over_six_l105_105630


namespace find_number_of_even_numbers_l105_105898

-- Define the average of the first n even numbers
def average_of_first_n_even (n : ℕ) : ℕ :=
  (n * (1 + n)) / n

-- The given condition: The average is 21
def average_is_21 (n : ℕ) : Prop :=
  average_of_first_n_even n = 21

-- The theorem to prove: If the average is 21, then n = 20
theorem find_number_of_even_numbers (n : ℕ) (h : average_is_21 n) : n = 20 :=
  sorry

end find_number_of_even_numbers_l105_105898


namespace total_people_at_dance_l105_105068

variable (B G T : ℕ)
variable (R : ℚ := 3/4)
variable (G_count : G = 60)
variable (T_percent : ℚ := 0.20)

theorem total_people_at_dance :
  (G = 60) →
  (B = (3 * 60 / 4)) →
  (T = (0.20 * B).toNat) →
  (B + G + T = 114) :=
by
  intros hG hB hT
  rw [hG, hB, hT]
  sorry

end total_people_at_dance_l105_105068


namespace all_points_lie_on_parabola_l105_105241

noncomputable def parabola_curve (u : ℝ) : ℝ × ℝ :=
  let x := 3^u - 4
  let y := 9^u - 7 * 3^u - 2
  (x, y)

theorem all_points_lie_on_parabola (u : ℝ) :
  let (x, y) := parabola_curve u
  y = x^2 + x - 6 := sorry

end all_points_lie_on_parabola_l105_105241


namespace find_father_age_l105_105137

variable (M F : ℕ)

noncomputable def age_relation_1 : Prop := M = (2 / 5) * F
noncomputable def age_relation_2 : Prop := M + 5 = (1 / 2) * (F + 5)

theorem find_father_age (h1 : age_relation_1 M F) (h2 : age_relation_2 M F) : F = 25 := by
  sorry

end find_father_age_l105_105137


namespace distance_point_line_eq_l105_105674

noncomputable def distance_from_point_to_line : ℝ :=
  let a := (2, 0, -1)
  let b1 := (1, 3, 2)
  let b2 := (3, 0, 5)
  let d := Math.sqrt (5337) / 11
  d

theorem distance_point_line_eq :
  let a := (2, 0, -1)
  let b1 := (1, 3, 2)
  let b2 := (3, 0, 5)
  distance_from_point_to_line = (Math.sqrt 5337) / 11 :=
by
  sorry

end distance_point_line_eq_l105_105674


namespace minimize_sum_l105_105217

-- Define a structure for a triangle with points in a 2D plane
structure Triangle (α : Type _) [EuclideanGeometry α] :=
(A B C : α)

-- Define positive real numbers
variables {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)

-- Define the Apollonius point property
def is_apollonius_point {α : Type _} [EuclideanGeometry α] 
  (P : α) (T : Triangle α) := 
  ∃ (X Y Z : α), (T.A = X ∧ T.B = Y ∧ T.C = Z) ∧
                   dist P X / dist P Y = a / b ∧
                   dist P Y / dist P Z = b / c

-- Define the minimum expression property
def minimizes_expression {α : Type _} [EuclideanGeometry α] 
  (T : Triangle α) (D E F : α) (P : α) := 
  is_apollonius_point P T →
  (∀ (D' E' F' : α), 
    a * dist E' F' + b * dist F' D' + c * dist D' E' ≥ 
    a * dist E F + b * dist F D + c * dist D E)

-- Lean theorem statement
theorem minimize_sum {α : Type _} [EuclideanGeometry α] 
    (T : Triangle α) (D E F P : α) :
  minimizes_expression T D E F P :=
sorry

end minimize_sum_l105_105217


namespace number_of_guinea_pigs_l105_105924

variable (rabbits guinea_pigs : ℕ)

def ratio (rabbits guinea_pigs : ℕ) : Prop :=
  rabbits * 4 = guinea_pigs * 5

def number_of_rabbits : ℕ := 25

theorem number_of_guinea_pigs
  (h1 : ratio 25 guinea_pigs)
  : guinea_pigs = 20 :=
begin
  rw [ratio] at h1,
  sorry
end

end number_of_guinea_pigs_l105_105924


namespace count_five_digit_palindromes_l105_105752

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 = d5 ∧ d2 = d4

theorem count_five_digit_palindromes : 
  (finset.filter is_five_digit_palindrome (finset.Icc 10000 99999)).card = 900 :=
by
  sorry

end count_five_digit_palindromes_l105_105752


namespace monotonic_increasing_interval_l105_105055

noncomputable def f : ℝ → ℝ := λ x, real.log (1/2) (x^2 - 1)

theorem monotonic_increasing_interval :
  (∀ x, x ∈ (-∞ : ℝ, -1) → is_increasing (f x)) :=
sorry

end monotonic_increasing_interval_l105_105055


namespace unique_function_satisfying_condition_l105_105462

theorem unique_function_satisfying_condition (k : ℕ) (hk : 0 < k) :
  ∀ f : ℕ → ℕ, (∀ m n : ℕ, 0 < m → 0 < n → f m + f n ∣ (m + n) ^ k) →
  ∃ c : ℕ, ∀ n : ℕ, f n = n + c :=
by
  sorry

end unique_function_satisfying_condition_l105_105462


namespace necessary_not_sufficient_l105_105707

variable (x : ℝ)

def p := x^2 - x < 1
def q := ∀ x, log 2 (x^2 - x) < 0

theorem necessary_not_sufficient (x : ℝ) : (p x → q x) ∧ (q x → p x) := sorry

end necessary_not_sufficient_l105_105707


namespace number_of_valid_n_l105_105684

theorem number_of_valid_n 
  : ∃ (n_values : Finset ℤ), 
      (∀ n ∈ n_values, ∃ k l : ℤ, 3200 = (2^6 * 5^2) ∧ 3200 * (5^n) * (2^(-3 * n)) ∈ ℤ) ∧
      n_values.card = 3 := 
sorry

end number_of_valid_n_l105_105684


namespace athlete_with_most_stable_performance_l105_105627

def variance_A : ℝ := 0.78
def variance_B : ℝ := 0.2
def variance_C : ℝ := 1.28

theorem athlete_with_most_stable_performance : variance_B < variance_A ∧ variance_B < variance_C :=
by {
  -- Variance comparisons:
  -- 0.2 < 0.78
  -- 0.2 < 1.28
  sorry
}

end athlete_with_most_stable_performance_l105_105627


namespace pencils_purchased_l105_105199

variable (P : ℕ)

theorem pencils_purchased (misplaced broke found bought left : ℕ) (h1 : misplaced = 7) (h2 : broke = 3) (h3 : found = 4) (h4 : bought = 2) (h5 : left = 16) :
  P - misplaced - broke + found + bought = left → P = 22 :=
by
  intros h
  have h_eq : P - 7 - 3 + 4 + 2 = 16 := by
    rw [h1, h2, h3, h4, h5] at h; exact h
  sorry

end pencils_purchased_l105_105199


namespace sin_cos_identity_l105_105581

theorem sin_cos_identity : (Real.sin (65 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) 
  - Real.cos (65 * Real.pi / 180) * Real.sin (35 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end sin_cos_identity_l105_105581


namespace combined_rate_is_29_l105_105481

def combined_rate_of_mpg (miles_ray : ℕ) (mpg_ray : ℕ) (miles_tom : ℕ) (mpg_tom : ℕ) (miles_jerry : ℕ) (mpg_jerry : ℕ) : ℕ :=
  let gallons_ray := miles_ray / mpg_ray
  let gallons_tom := miles_tom / mpg_tom
  let gallons_jerry := miles_jerry / mpg_jerry
  let total_gallons := gallons_ray + gallons_tom + gallons_jerry
  let total_miles := miles_ray + miles_tom + miles_jerry
  total_miles / total_gallons

theorem combined_rate_is_29 :
  combined_rate_of_mpg 60 50 60 20 60 30 = 29 :=
by
  sorry

end combined_rate_is_29_l105_105481


namespace range_of_k_l105_105519

noncomputable def circle_equation (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem range_of_k (k : ℝ) :
  circle_equation k →
  k ∈ (Set.Iio (-1) ∪ Set.Ioi 4) :=
sorry

end range_of_k_l105_105519


namespace bisector_plane_divides_ratio_l105_105511

noncomputable def tetrahedron (A B C D : Type*) := (P Q : ℚ) → Prop

theorem bisector_plane_divides_ratio (A B C D M : Type*) (P Q : ℚ)
  (Face_ABC : area P) (Face_ADC : area Q) (Bisector_plane_intersection : M = bisector_plane_intersection AC BD) :
  BM / DM = P / Q := 
sorry

end bisector_plane_divides_ratio_l105_105511


namespace cheapest_book_price_l105_105687

theorem cheapest_book_price
  (n : ℕ) (c : ℕ) (d : ℕ)
  (h1 : n = 40)
  (h2 : d = 3)
  (h3 : c + d * 19 = 75) :
  c = 18 :=
sorry

end cheapest_book_price_l105_105687


namespace total_pizza_order_cost_l105_105885

def pizza_cost_per_pizza := 10
def topping_cost_per_topping := 1
def tip_amount := 5
def number_of_pizzas := 3
def number_of_toppings := 4

theorem total_pizza_order_cost : 
  (pizza_cost_per_pizza * number_of_pizzas + topping_cost_per_topping * number_of_toppings + tip_amount) = 39 := by
  sorry

end total_pizza_order_cost_l105_105885


namespace three_digit_multiples_of_7_l105_105328

theorem three_digit_multiples_of_7 : 
  ∃! n : ℕ, (n = 128) ∧ (∀ k, (100 ≤ 7 * k ∧ 7 * k ≤ 999) ↔ (15 ≤ k ∧ k ≤ 142)) :=
begin
  sorry
end

end three_digit_multiples_of_7_l105_105328


namespace find_d_l105_105053

theorem find_d (d : ℝ) : (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  { sorry }

end find_d_l105_105053


namespace avg_first_six_results_l105_105036

theorem avg_first_six_results (A : ℝ) :
  (∀ (results : Fin 12 → ℝ), 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5 + 
     results 6 + results 7 + results 8 + results 9 + results 10 + results 11) / 11 = 60 → 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5) / 6 = A → 
    (results 5 + results 6 + results 7 + results 8 + results 9 + results 10) / 6 = 63 → 
    results 5 = 66) → 
  A = 58 :=
by
  sorry

end avg_first_six_results_l105_105036


namespace initial_group_machines_l105_105485

-- Define the number of bags produced by n machines in one minute and 150 machines in one minute
def bags_produced (machines : ℕ) (bags_per_minute : ℕ) : Prop :=
  machines * bags_per_minute = 45

def bags_produced_150 (bags_produced_in_8_mins : ℕ) : Prop :=
  150 * (bags_produced_in_8_mins / 8) = 450

-- Given the conditions, prove that the number of machines in the initial group is 15
theorem initial_group_machines (n : ℕ) (bags_produced_in_8_mins : ℕ) :
  bags_produced n 45 → bags_produced_150 bags_produced_in_8_mins → n = 15 :=
by
  intro h1 h2
  -- use the conditions to derive the result
  sorry

end initial_group_machines_l105_105485


namespace min_value_a_4b_l105_105248

theorem min_value_a_4b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = a + b) :
  a + 4 * b = 9 :=
sorry

end min_value_a_4b_l105_105248


namespace largest_multiple_of_12_neg_gt_neg_150_l105_105963

theorem largest_multiple_of_12_neg_gt_neg_150 : ∃ m : ℤ, (m % 12 = 0) ∧ (-m > -150) ∧ ∀ n : ℤ, (n % 12 = 0) ∧ (-n > -150) → n ≤ m := sorry

end largest_multiple_of_12_neg_gt_neg_150_l105_105963


namespace distinct_heptagons_l105_105106

theorem distinct_heptagons (points : Finset Point) (h_card : points.card = 9) : 
  ∃ h1 h2 : Finset Point, h1.card = 7 ∧ h2.card = 7 ∧ h1 ≠ h2 ∧ ¬ (h1 ~ h2) :=
by
  sorry

end distinct_heptagons_l105_105106


namespace derivative_y_l105_105673

-- Define the function y
def y (x : ℝ) : ℝ := (1 - x) / ((1 + x^2) * cos x)

-- Define the derivative of y
noncomputable def y' (x : ℝ) : ℝ :=
  ((x^2 - 2 * x - 1) * cos x + (1 - x) * (1 + x^2) * sin x) / ((1 + x^2)^2 * (cos x)^2)

-- The theorem to be proven
theorem derivative_y (x : ℝ) : deriv y x = y' x := by
  sorry

end derivative_y_l105_105673


namespace total_weight_of_fish_is_correct_l105_105810

noncomputable def totalWeightInFirstTank := 15 * 0.08 + 12 * 0.05

noncomputable def totalWeightInSecondTank := 2 * 15 * 0.08 + 3 * 12 * 0.05

noncomputable def totalWeightInThirdTank := 3 * 15 * 0.08 + 2 * 12 * 0.05 + 5 * 0.14

noncomputable def totalWeightAllTanks := totalWeightInFirstTank + totalWeightInSecondTank + totalWeightInThirdTank

theorem total_weight_of_fish_is_correct : 
  totalWeightAllTanks = 11.5 :=
by         
  sorry

end total_weight_of_fish_is_correct_l105_105810


namespace total_people_at_dance_l105_105072

theorem total_people_at_dance (n_girls n_boys n_teachers total_people : ℕ) 
  (h1 : ratio_girls_boys = 4/3) 
  (h2 : n_girls = 60) 
  (h3 : n_teachers = 0.2 * n_boys) 
  (h4 : n_boys = 3/4 * n_girls) : 
  total_people = n_girls + n_boys + n_teachers :=
  begin
    -- Calculate number of boys
    have h_boys : n_boys = 45, from sorry,
    -- Calculate number of teachers
    have h_teachers : n_teachers = 9, from sorry,
    -- Total number of people calculation
    show total_people = 60 + 45 + 9, from sorry
  end

end total_people_at_dance_l105_105072


namespace simplify_factorial_l105_105493

theorem simplify_factorial :
  ∀ (f12 f10 f9 : ℕ), f12 = 12! ∧ f10 = 10! ∧ f9 = 9! →
  (f12 / (f10 + 3 * f9) = 4 / 3) :=
by
  sorry

end simplify_factorial_l105_105493


namespace diagonal_of_cylinder_rectangular_plate_l105_105586

theorem diagonal_of_cylinder_rectangular_plate (h : ℝ) (p : ℝ) (d : ℝ) : 
  h = 8 → p = 6 → d = 10 :=
by
  assume h_eq : h = 8,
  assume p_eq : p = 6,
  have length_eq : Float := p,
  have width_eq : Float := h,
  have diag_squared := length_eq^2 + width_eq^2,
  have diag := Real.sqrt diag_squared,
  have d_eq := d,
  have : diag = d_eq,
  sorry

end diagonal_of_cylinder_rectangular_plate_l105_105586


namespace flour_from_300kg_wheat_wheat_for_170kg_flour_l105_105048

def flour_yield : ℝ := 0.85

def flour_from_wheat (wheat : ℝ) : ℝ :=
  wheat * flour_yield

def wheat_for_flour (flour : ℝ) : ℝ :=
  flour / flour_yield

theorem flour_from_300kg_wheat :
  flour_from_wheat 300 = 255 := by
  sorry

theorem wheat_for_170kg_flour :
  wheat_for_flour 170 = 200 := by
  sorry

end flour_from_300kg_wheat_wheat_for_170kg_flour_l105_105048


namespace probability_x_plus_y_even_l105_105143

theorem probability_x_plus_y_even :
  let x_range := {1, 2, 3, 4, 5}
  let y_range := {7, 8, 9, 10}
  choose x from x_range
  choose y from y_range
  let even_prob := ∑ x in x_range, ∑ y in y_range, ((x + y) % 2 = 0)
  in (even_prob / (x_range.card * y_range.card) = 1/2) :=
sorry

end probability_x_plus_y_even_l105_105143


namespace count_multiples_5_and_6_not_8_or_9_l105_105310

theorem count_multiples_5_and_6_not_8_or_9 :
  ∃ n : ℕ, n = 4 ∧ (∀ m : ℕ, (1 ≤ m ∧ m ≤ 300) → (m % 30 = 0) →
    ¬(m % 8 = 0 ∨ m % 9 = 0) → m ∈ ({30, 60, 150, 300} : set ℕ)) := 
sorry

end count_multiples_5_and_6_not_8_or_9_l105_105310


namespace shortest_tangent_length_l105_105823

-- Define the circles C3 and C4
def C3 (x y : ℝ) : Prop := (x - 8)^2 + y^2 = 25
def C4 (x y : ℝ) : Prop := (x + 10)^2 + y^2 = 49

-- Proven statement: the length of the shortest tangent line segment RS given the circles C3 and C4
theorem shortest_tangent_length : 
  (∀ R S : ℝ × ℝ, C3 R.1 R.2 → C4 S.1 S.2 → 
  tangent_to_circle R C3 → tangent_to_circle S C4 → 
  length_tangent_segment R S) = 15 :=
sorry

end shortest_tangent_length_l105_105823


namespace lucas_change_l105_105468

-- Define the given conditions as constants in Lean
def num_bananas : ℕ := 5
def cost_per_banana : ℝ := 0.70
def num_oranges : ℕ := 2
def cost_per_orange : ℝ := 0.80
def amount_paid : ℝ := 10.00

-- Define a noncomputable constant to represent the change received
noncomputable def change_received : ℝ := 
  amount_paid - (num_bananas * cost_per_banana + num_oranges * cost_per_orange)

-- State the theorem to be proved
theorem lucas_change : change_received = 4.90 := 
by 
  -- Dummy proof since the actual proof is not required
  sorry

end lucas_change_l105_105468


namespace sqrt_three_irrational_sqrt_three_decimal_expansion_l105_105656

theorem sqrt_three_irrational : 
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (nat.coprime p q) ∧ (3 = (p * p) / (q * q)) := by sorry

theorem sqrt_three_decimal_expansion :
  irrational (abs.sqrt 3) :=
begin
  assume h : rational (abs.sqrt 3),
  cases h with p q,
  cases sqrt_three_irrational with p q,
  sorry,
end

end sqrt_three_irrational_sqrt_three_decimal_expansion_l105_105656


namespace locus_of_P_l105_105696

-- Given conditions
variables {a b : ℝ} (ha : 0 < a) (hb : a < b)
variables (A : ℝ × ℝ := (a, 0)) (B : ℝ × ℝ := (b, 0))

-- Definitions of lines l and m
def line_l (s : ℝ) (y : ℝ) := s * y + a
def line_m (t : ℝ) (y : ℝ) := t * y + b

-- Intersection of lines with the parabola
def parabola (y : ℝ) := y^2

-- Problem statement to prove
theorem locus_of_P (s t : ℝ) (hst : s + t = 0) :
  ∀ y : ℝ, line_l s y = parabola y → line_m t y = parabola y →
  ∃ x, x = (a + b) / 2 :=
sorry

end locus_of_P_l105_105696


namespace total_people_at_dance_l105_105073

theorem total_people_at_dance (n_girls n_boys n_teachers total_people : ℕ) 
  (h1 : ratio_girls_boys = 4/3) 
  (h2 : n_girls = 60) 
  (h3 : n_teachers = 0.2 * n_boys) 
  (h4 : n_boys = 3/4 * n_girls) : 
  total_people = n_girls + n_boys + n_teachers :=
  begin
    -- Calculate number of boys
    have h_boys : n_boys = 45, from sorry,
    -- Calculate number of teachers
    have h_teachers : n_teachers = 9, from sorry,
    -- Total number of people calculation
    show total_people = 60 + 45 + 9, from sorry
  end

end total_people_at_dance_l105_105073


namespace problem_statement_l105_105764

variables (x y : ℚ)

theorem problem_statement 
  (h1 : x + y = 8 / 15) 
  (h2 : x - y = 1 / 105) : 
  x^2 - y^2 = 8 / 1575 :=
sorry

end problem_statement_l105_105764


namespace option_d_correct_l105_105277

variable (a b c d : ℝ)

theorem option_d_correct (h1: a > b) (h2: b > 0) (h3: c > d) (h4: d > 0): 
  (sqrt (a / d) > sqrt (b / c)) :=
by {
  sorry
}

end option_d_correct_l105_105277


namespace grasshopper_opposite_corner_moves_l105_105949

noncomputable def grasshopper_jump_count : ℕ :=
  Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9)

theorem grasshopper_opposite_corner_moves :
  grasshopper_jump_count = Nat.factorial 27 / (Nat.factorial 9 * Nat.factorial 9 * Nat.factorial 9) :=
by
  -- The detailed proof would go here.
  sorry

end grasshopper_opposite_corner_moves_l105_105949


namespace length_of_DE_correct_sum_of_num_and_denom_l105_105550

noncomputable def length_of_DE (AB AC BC : ℝ) : ℚ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  let r := K / s
  let DE := (BC * (AB + AC - Real.sqrt ((AB - AC)^2 + 4 * r^2)) / (AB + AC))
  (DE.num / DE.denom : ℚ)

-- Assuming the known lengths of the sides of the triangle:
def AB := 28
def AC := 29
def BC := 26

-- Prove the length of DE is 806/57
theorem length_of_DE_correct : length_of_DE AB AC BC = 806 / 57 :=
by 
  sorry

-- Prove the sum of the numerator and denominator is 863
theorem sum_of_num_and_denom : (806 + 57) = 863 :=
by 
  sorry

end length_of_DE_correct_sum_of_num_and_denom_l105_105550


namespace cube_number_sum_l105_105063

/-
Proof Problem:
Given:
1. The numbers on the faces of the cube are consecutive whole numbers between 15 and 20.
2. The sum of the numbers on each pair of opposite faces is equal.
3. The middle number (17.5 in the range 15-20) is the largest number on one of the faces.

Prove that the sum of all the numbers on this cube is 105.
-/

theorem cube_number_sum :
  ∃ (numbers : Finset ℕ),
    (∀ n, n ∈ numbers ↔ 15 ≤ n ∧ n ≤ 20) ∧
    (numbers.sum id = 105) ∧
    (∃ middle : ℕ, middle = 18 ∧ middle ∈ numbers) ∧
    (∃ opposite_pairs : list (ℕ × ℕ), (∀ p ∈ opposite_pairs, p.1 + p.2 = 35)) :=
by
  sorry

end cube_number_sum_l105_105063


namespace log_comparison_l105_105693

theorem log_comparison 
  (a : ℝ := 1 / 6 * Real.log 8)
  (b : ℝ := 1 / 2 * Real.log 5)
  (c : ℝ := Real.log (Real.sqrt 6) - Real.log (Real.sqrt 2)) :
  a < c ∧ c < b := 
by
  sorry

end log_comparison_l105_105693


namespace pen_cost_l105_105007

theorem pen_cost (x : ℝ) (h1 : 5 * x + x = 24) : x = 4 :=
by
  sorry

end pen_cost_l105_105007


namespace solve_system_l105_105892

-- Define the conditions of the system of equations
def condition1 (x y : ℤ) := 4 * x - 3 * y = -13
def condition2 (x y : ℤ) := 5 * x + 3 * y = -14

-- Define the proof goal using the conditions
theorem solve_system : ∃ (x y : ℤ), condition1 x y ∧ condition2 x y ∧ x = -3 ∧ y = 1 / 3 :=
by
  sorry

end solve_system_l105_105892


namespace second_discount_percentage_l105_105928

-- Define the conditions
def original_price : ℝ := 175
def first_discount_rate : ℝ := 0.20
def final_sale_price : ℝ := 133
def first_discount_price : ℝ := original_price * (1 - first_discount_rate)

-- Define what we need to prove
theorem second_discount_percentage :
  first_discount_price * (1 - d2) = final_sale_price →
  d2 * 100 = 5 :=
by
  simp [first_discount_price, original_price, first_discount_rate, final_sale_price, d2]
  sorry

end second_discount_percentage_l105_105928


namespace probability_A_miss_at_least_once_probability_A_2_hits_B_3_hits_l105_105584

variable {p q : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1)

theorem probability_A_miss_at_least_once :
  1 - p^4 = (1 - p^4) := by
sorry

theorem probability_A_2_hits_B_3_hits :
  24 * p^2 * q^3 * (1 - p)^2 * (1 - q) = 24 * p^2 * q^3 * (1 - p)^2 * (1 - q) := by
sorry

end probability_A_miss_at_least_once_probability_A_2_hits_B_3_hits_l105_105584


namespace no_polynomial_with_conditions_l105_105876

theorem no_polynomial_with_conditions :
  ¬ ∃ (f : ℤ[X]), (f.eval 7 = 11) ∧ (f.eval 11 = 13) :=
by
  -- Proof omitted
  sorry

end no_polynomial_with_conditions_l105_105876


namespace integer_solutions_2x2_2xy_9x_y_eq_2_l105_105503

theorem integer_solutions_2x2_2xy_9x_y_eq_2 : ∀ (x y : ℤ), 2 * x^2 - 2 * x * y + 9 * x + y = 2 → (x, y) = (1, 9) ∨ (x, y) = (2, 8) ∨ (x, y) = (0, 2) ∨ (x, y) = (-1, 3) := 
by 
  intros x y h
  sorry

end integer_solutions_2x2_2xy_9x_y_eq_2_l105_105503


namespace number_of_jars_good_for_sale_l105_105004

def numberOfGoodJars (initialCartons : Nat) (cartonsNotDelivered : Nat) (jarsPerCarton : Nat)
  (damagedJarsPerCarton : Nat) (numberOfDamagedCartons : Nat) (oneTotallyDamagedCarton : Nat) : Nat := 
  let deliveredCartons := initialCartons - cartonsNotDelivered
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := (damagedJarsPerCarton * numberOfDamagedCartons) + oneTotallyDamagedCarton
  totalJars - damagedJars

theorem number_of_jars_good_for_sale : 
  numberOfGoodJars 50 20 20 3 5 20 = 565 :=
by
  sorry

end number_of_jars_good_for_sale_l105_105004


namespace shares_of_stocks_they_can_buy_l105_105169

def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def months_of_savings : ℕ := 4
def cost_per_share : ℕ := 50

theorem shares_of_stocks_they_can_buy :
  (((weekly_savings_wife * 4) + monthly_savings_husband) * months_of_savings / 2) / cost_per_share = 25 :=
by
  -- sorry for the implementation
  sorry

end shares_of_stocks_they_can_buy_l105_105169


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l105_105117

theorem frequencies_of_first_class_products (total_A total_B first_class_A first_class_B : ℕ) 
  (h1 : total_A = 200)
  (h2 : first_class_A = 150)
  (h3 : total_B = 200)
  (h4 : first_class_B = 120) :
  (first_class_A / total_A = 3 / 4) ∧ (first_class_B / total_B = 3 / 5) :=
by {
  sorry
}

theorem confidence_in_difference_of_quality (total_A total_B first_class_A second_class_A first_class_B second_class_B : ℕ)
  (h1 : total_A = 200)
  (h2 : first_class_A = 150)
  (h3 : second_class_A = 50)
  (h4 : total_B = 200)
  (h5 : first_class_B = 120)
  (h6 : second_class_B = 80)
  (total : ℕ)
  (h7 : total = total_A + total_B)
  (k_value : ℝ)
  (h8 : k_value = 6.635):
  let K_squared := (total * (first_class_A * second_class_B - second_class_A * first_class_B) ^ 2) / 
                   ((first_class_A + second_class_A) * (first_class_B + second_class_B) * (first_class_A + first_class_B) * (second_class_A + second_class_B)) 
  in K_squared > k_value :=
by {
  sorry
}

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l105_105117


namespace possible_values_of_k_l105_105455

noncomputable def unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop := ∥v∥ = 1

theorem possible_values_of_k
    (a b c : EuclideanSpace ℝ (Fin 3))
    (ha : unit_vector a)
    (hb : unit_vector b)
    (hc : unit_vector c)
    (ha_b : a ⬝ b = 0)
    (ha_c : a ⬝ c = 0)
    (angle_bc : real.angle b c = real.pi / 3) :
    ∃ k : ℝ, a = k • (b × c) ∧ (k = 2 * real.sqrt 3 / 3 ∨ k = - (2 * real.sqrt 3 / 3)) :=
sorry

end possible_values_of_k_l105_105455


namespace infinitely_many_Silesian_integers_infinitely_many_non_Silesian_integers_l105_105652

def is_Silesian (n : ℤ) : Prop :=
  ∃ (a b c : ℕ), n = (a^2 + b^2 + c^2) / (ab + bc + ca)

theorem infinitely_many_Silesian_integers :
  ∀ k : ℕ, ∃ n : ℤ, is_Silesian n := 
  sorry

theorem infinitely_many_non_Silesian_integers :
  ∃ k : ℕ, ∀ n : ℤ, ¬ is_Silesian (3 * k) :=
  sorry

end infinitely_many_Silesian_integers_infinitely_many_non_Silesian_integers_l105_105652


namespace multiples_of_7_are_128_l105_105322

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l105_105322


namespace mean_value_of_triangle_angles_l105_105971

theorem mean_value_of_triangle_angles : ∀ {α β γ : ℝ}, α + β + γ = 180 → 
    (α + β + γ) / 3 = 60 :=
by
  intros α β γ h α β γ 
  calc 
    (α + β + γ) / 3 = 180 / 3 : by rw h
                  ... = 60   : by norm_num

end mean_value_of_triangle_angles_l105_105971


namespace value_of_x_l105_105977

theorem value_of_x (x : ℝ) (h : (sqrt x)^4 = 256) : x = 16 :=
sorry

end value_of_x_l105_105977


namespace original_fraction_is_one_third_l105_105043

theorem original_fraction_is_one_third
  (a b : ℕ) (h₁ : Nat.gcd a b = 1)
  (h₂ : (a + 2) * b = 3 * a * b^2) : 
  a = 1 ∧ b = 3 := by
  sorry

end original_fraction_is_one_third_l105_105043


namespace tangent_line_at_e_range_a_if_exists_max_k_satisfies_f_l105_105735

open Real

noncomputable theory

/-- The equation of the tangent line to the function f(x) at x = e is 2x - y - e = 0 -/
def tangent_line (f : ℝ → ℝ) (x : ℝ) : Prop :=
∀ y : ℝ, f(e) = e → f'(e) = 2 → y - e = 2 * (x - e)

/-- If there exists at least one x₀ in [1, e] such that f(x₀) < g(x₀) holds, then the range of values for a is (0, +∞). -/
def range_of_a (a : ℝ) (f g : ℝ → ℝ) : Prop :=
(∃ x₀ ∈ Icc (1 : ℝ) (real.exp 1), f(x₀) < g(x₀)) ↔ a ∈ Ioi 0

/-- The maximum value of the integer k such that f(x) > (k-3)x-k+2 always holds for x > 1 is 5. -/
def max_value_of_k (f : ℝ → ℝ) : Prop :=
∀ k : ℤ, (∀ x : ℝ, 1 < x → f(x) > (k-3)*x - k + 2) ↔ k ≤ 5

-- Define the main properties based on the given functions
def f (x : ℝ) : ℝ := x * log x
def g (a : ℝ) (x : ℝ) : ℝ := (a * x^2) / 2

-- Main theorem statements
theorem tangent_line_at_e : tangent_line f (real.exp 1) := 
sorry

theorem range_a_if_exists (a : ℝ) : range_of_a a f (g a) := 
sorry

theorem max_k_satisfies_f : max_value_of_k f := 
sorry

end tangent_line_at_e_range_a_if_exists_max_k_satisfies_f_l105_105735


namespace distinct_solutions_to_abs_eq_l105_105232

theorem distinct_solutions_to_abs_eq (x : ℝ) :
  (∃! x, (abs (x - abs (3 * x - 2)) = 4)) :=
sorry

end distinct_solutions_to_abs_eq_l105_105232


namespace central_cell_value_l105_105409

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l105_105409


namespace remainder_of_p_div_10_is_6_l105_105136

-- Define the problem
def a : ℕ := sorry -- a is a positive integer and a multiple of 2

-- Define p based on a
def p : ℕ := 4^a

-- The main goal is to prove the remainder when p is divided by 10 is 6
theorem remainder_of_p_div_10_is_6 (ha : a > 0 ∧ a % 2 = 0) : p % 10 = 6 := by
  sorry

end remainder_of_p_div_10_is_6_l105_105136


namespace diagram_representation_l105_105185

theorem diagram_representation:
  ∀ (m n : ℝ), m ≠ 0 ∧ n ≠ 0 →
  (∃ D, D = "A" ∧
    (∀ x y : ℝ, m * x + n * y^2 = 0) ∧
    (∀ x y : ℝ, m * x^2 + n * y^2 = 1)) :=
by
  intros m n h
  use "A"
  split
  . rfl
  . sorry
  . sorry

end diagram_representation_l105_105185


namespace boys_from_school_A_non_science_l105_105778

theorem boys_from_school_A_non_science (total_boys : ℕ) (percent_A : ℝ) (percent_science_A : ℝ) (boys_A_non_science : ℕ) : 
  total_boys = 550 →
  percent_A = 0.20 →
  percent_science_A = 0.30 →
  boys_A_non_science = (percent_A * total_boys).to_nat - (percent_science_A * (percent_A * total_boys)).to_nat →
  boys_A_non_science = 77 :=
by
  sorry

end boys_from_school_A_non_science_l105_105778


namespace fraction_identity_l105_105349

variable {a b x : ℝ}

-- Conditions
axiom h1 : x = a / b
axiom h2 : a ≠ b
axiom h3 : b ≠ 0

-- Question to prove
theorem fraction_identity :
  (a + b) / (a - b) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_identity_l105_105349


namespace num_five_digit_palindromes_l105_105749

theorem num_five_digit_palindromes : 
  (∃ A B C : ℕ, 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) → 
  (nat.count (λ n, ∃ A B C : ℕ, n = A * 10001 + B * 1010 + C * 100 + B * 10 + A ∧ 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) (1, 99999)) = 900 := 
by 
  sorry

end num_five_digit_palindromes_l105_105749


namespace remaining_area_l105_105164

-- Definitions based on conditions
def large_rectangle_length (x : ℝ) : ℝ := 2 * x + 8
def large_rectangle_width (x : ℝ) : ℝ := x + 6
def hole_length (x : ℝ) : ℝ := 3 * x - 4
def hole_width (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem remaining_area (x : ℝ) : (large_rectangle_length x) * (large_rectangle_width x) - (hole_length x) * (hole_width x) = -x^2 + 21 * x + 52 :=
by
  -- Proof is skipped
  sorry

end remaining_area_l105_105164


namespace binomial_expansion_l105_105038

theorem binomial_expansion (x : ℝ) :
  let a : ℝ := 2 * x ^ 2,
      b : ℝ := - 1 / (5 * x),
      n : ℕ := 5 in
  ((a + b) ^ n).coeff (monomial_coefficient 1) = -8/25 :=
sorry

end binomial_expansion_l105_105038


namespace leak_drain_time_l105_105573

/-- Statement: Given the rates at which a pump fills a tank and a leak drains the tank, 
prove that the leak can drain all the water in the tank in 14 hours. -/
theorem leak_drain_time :
  (∀ P L: ℝ, P = 1/2 → (P - L) = 3/7 → L = 1/14 → (1 / L) = 14) := 
by
  intros P L hP hPL hL
  -- Proof is omitted (to be provided)
  sorry

end leak_drain_time_l105_105573


namespace central_cell_value_l105_105407

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l105_105407


namespace trig_expression_value_l105_105214

theorem trig_expression_value :
  (real.sin 410 * real.sin 550 - real.sin 680 * real.cos 370) = 1 / 2 :=
by 
  have h₁ : real.sin 410 = real.sin 50 := by sorry,
  have h₂ : real.sin 550 = -real.sin 10 := by sorry,
  have h₃ : real.sin 680 = -real.sin 40 := by sorry,
  have h₄ : real.cos 370 = real.cos 10 := by sorry,
  calc
    real.sin 410 * real.sin 550 - real.sin 680 * real.cos 370
      = (real.sin 50) * (-real.sin 10) - (-real.sin 40) * (real.cos 10) : by rw [h₁, h₂, h₃, h₄]
  ... = sorry : by sorry 
  ... = 1 / 2 : by sorry

end trig_expression_value_l105_105214


namespace calum_disco_ball_budget_l105_105203

theorem calum_disco_ball_budget :
  ∀ (budget : ℝ) (n_disco_balls : ℕ) (n_food : ℕ) (cost_food : ℝ) (n_decor : ℕ) (cost_decor : ℝ) (disco_ball_percentage : ℝ),
  budget = 600 ∧ n_disco_balls = 4 ∧ n_food = 10 ∧ cost_food = 25 ∧ n_decor = 20 ∧ cost_decor = 10 ∧ disco_ball_percentage = 0.30 →
  (budget * disco_ball_percentage) / n_disco_balls = 45 :=
by 
  intros budget n_disco_balls n_food cost_food n_decor cost_decor disco_ball_percentage h,
  cases h with hb1 h,
  cases h with hb2 h,
  cases h with nf h,
  cases h with cf h,
  cases h with nd h,
  cases h with cd hb,
  rw [hb1, hb2, cf, nf, cd, nd, hb],
  norm_num,
  sorry

end calum_disco_ball_budget_l105_105203


namespace decreasing_interval_3_pow_x_squared_minus_4x_l105_105903

theorem decreasing_interval_3_pow_x_squared_minus_4x :
  ∀ x1 x2 : ℝ, x1 < 2 → x2 < 2 → x1 < x2 → 3^(x1^2 - 4 * x1) > 3^(x2^2 - 4 * x2) :=
by
  intros x1 x2 hx1_lt2 hx2_lt2 hx1_lt_x2
  sorry

end decreasing_interval_3_pow_x_squared_minus_4x_l105_105903


namespace find_some_number_l105_105992

theorem find_some_number :
  ∃ n : ℤ, (481 + 426) * n - 4 * 481 * 426 = 3025 → n = 906 :=
begin
  sorry
end

end find_some_number_l105_105992


namespace maximize_revenue_revenue_increase_l105_105238

variables (p n x z : ℝ)
variable (hx : 0 < x ∧ x ≤ 10)

-- The conditions are defined as hypotheses
def conditions :=
  (z = (1 + x / 100) * (1 - x / 100))

-- Question (1) proof statement
theorem maximize_revenue (h : conditions p n x z) : x = 5 :=
  sorry

-- Question (2) proof statement
theorem revenue_increase (h : conditions p n x z) : 0 < x ∧ x < 5 ↔ z > 1 :=
  sorry

end maximize_revenue_revenue_increase_l105_105238


namespace clock_hands_overlap_l105_105155

theorem clock_hands_overlap:
  ∃ x y: ℚ,
  -- Conditions
  (60 * 10 + x = 60 * 11 * 54 + 6 / 11) ∧
  (y - (5 / 60) * y = 60) ∧
  (65 * 5 / 11 = y) := sorry

end clock_hands_overlap_l105_105155


namespace train_length_l105_105163

-- Define the conditions
def jogger_speed_kmh : ℝ := 9
def train_speed_kmh : ℝ := 45
def initial_distance_m : ℝ := 240
def time_to_pass_s : ℝ := 34

-- Calculate the speed in m/s
def kmh_to_ms (v : ℝ) : ℝ := (v * 1000) / 3600
def jogger_speed_ms : ℝ := kmh_to_ms jogger_speed_kmh
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

-- Relative speed in m/s
def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms

-- Prove the length of the train
theorem train_length : ∃ (L : ℝ), L = 100 :=
by
  sorry

end train_length_l105_105163


namespace quadratic_roots_distinct_l105_105094

theorem quadratic_roots_distinct (m : ℝ) :
  let Δ := m^2 + 32 in
  Δ > 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 - 8 = 0 ∧ x2^2 + m * x2 - 8 = 0) :=
begin
  sorry
end

end quadratic_roots_distinct_l105_105094


namespace charity_donation_correct_l105_105507

-- Define each donation series for Suzanne, Maria, and James
def suzanne_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 10
  | (n+1)  => 2 * suzanne_donation_per_km n

def maria_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 15
  | (n+1)  => 1.5 * maria_donation_per_km n

def james_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 20
  | (n+1)  => 2 * james_donation_per_km n

-- Total donations after 5 kilometers
def total_donation_suzanne : ℝ := (List.range 5).map suzanne_donation_per_km |>.sum
def total_donation_maria : ℝ := (List.range 5).map maria_donation_per_km |>.sum
def total_donation_james : ℝ := (List.range 5).map james_donation_per_km |>.sum

def total_donation_charity : ℝ :=
  total_donation_suzanne + total_donation_maria + total_donation_james

-- Statement to be proven
theorem charity_donation_correct : total_donation_charity = 1127.81 := by
  sorry

end charity_donation_correct_l105_105507


namespace cotangent_identity_l105_105480

variables (a b c : ℝ) (α β γ : ℝ)
  
-- Assume 2b^2 = a^2 + c^2 and we need to prove 2 * cot β = cot α + cot γ
theorem cotangent_identity (h : 2 * b^2 = a^2 + c^2) : 
  2 * (Real.cot β) = (Real.cot α) + (Real.cot γ) :=
sorry

end cotangent_identity_l105_105480


namespace quadratic_has_two_distinct_real_roots_l105_105085

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l105_105085


namespace concavity_convexity_inflection_point_l105_105441

open Real

noncomputable def y (x : ℝ) : ℝ := x^3 - 6 * x^2 + 4

def first_derivative (f : ℝ → ℝ) : ℝ → ℝ := λ x, deriv f x

def second_derivative (f : ℝ → ℝ) : ℝ → ℝ := λ x, deriv (first_derivative f) x

theorem concavity_convexity_inflection_point :
  (∀ x : ℝ, x < 2 → second_derivative y x < 0) ∧
  (∀ x : ℝ, x > 2 → second_derivative y x > 0) ∧
  (∃ x : ℝ, x = 2 ∧ y x = -12) :=
by
  sorry

end concavity_convexity_inflection_point_l105_105441


namespace fish_per_fishbowl_l105_105541

theorem fish_per_fishbowl {fishbowls fish : ℕ} (h1 : fishbowls = 261) (h2 : fish = 6003) :
  fish / fishbowls = 23 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end fish_per_fishbowl_l105_105541


namespace trigonometric_solution_exists_l105_105993

theorem trigonometric_solution_exists (a b c : ℝ) : 
  (∃ x : ℝ, a * sin x + b * cos x + c = 0) ∨ (∃ x : ℝ, 2 * a * tan x + b * tan x + 2 * c = 0) :=
sorry

end trigonometric_solution_exists_l105_105993


namespace angle_between_vectors_is_pi_div_2_l105_105718

open Real

noncomputable def unit_vec (v: ℝ × ℝ): Prop :=
  v.1^2 + v.2^2 = 1

theorem angle_between_vectors_is_pi_div_2
  (s t: ℝ)
  (hs : s ≠ 0)
  (ht : t ≠ 0)
  (i j: ℝ × ℝ)
  (hi: unit_vec i)
  (hj: unit_vec j)
  (h: ∥ (s * i.1, s * i.2) + (t * j.1, t * j.2) ∥ = ∥ (t * i.1, t * i.2) - (s * j.1, s * j.2) ∥)
  : ∃ θ, θ = π / 2 :=
begin
  sorry
end

end angle_between_vectors_is_pi_div_2_l105_105718


namespace proof_sum_divisible_by_3_proof_quadratic_solution_proof_positive_solutions_system_l105_105547

noncomputable def probability_sum_divisible_by_3 : ℚ :=
  let favorable_cases := [(1,2), (2,1), (1,5), (2,4), (3,3), (4,2), (5,1), 
                          (3,6), (4,5), (5,4), (6,3), (6,6)] in
  favorable_cases.length / 36

theorem proof_sum_divisible_by_3 : probability_sum_divisible_by_3 = 1/3 := sorry

noncomputable def probability_quadratic_solution : ℚ :=
  let favorable_cases := [(2,1), (4,4), (3,1), (3,2), (4,1), (4,2), (4,3), 
                          (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), 
                          (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)] in
  favorable_cases.length / 36

theorem proof_quadratic_solution : probability_quadratic_solution = 19/36 := sorry

noncomputable def probability_positive_solutions_system : ℚ :=
  let favorable_cases := [(2,1), (2,2), (3,1), (3,2), (4,1), (4,2), 
                          (5,1), (5,2), (6,1), (6,2), (1,4), (1,5), (1,6)] in
  favorable_cases.length / 36

theorem proof_positive_solutions_system : probability_positive_solutions_system = 13/36 := sorry

end proof_sum_divisible_by_3_proof_quadratic_solution_proof_positive_solutions_system_l105_105547


namespace cost_of_computer_chair_l105_105119

noncomputable def cost_of_computer_table : ℝ := 140
noncomputable def cost_of_joystick : ℝ := 20
noncomputable def fraction_frank_pays_of_joystick := 1/4
noncomputable def extra_frank_pays := 30

theorem cost_of_computer_chair (C : ℝ) :
  let F := cost_of_computer_table + fraction_frank_pays_of_joystick * cost_of_joystick in
  let E := C + (1 - fraction_frank_pays_of_joystick) * cost_of_joystick in
  F = E + extra_frank_pays → C = 100 :=
begin
  intros,
  sorry
end

end cost_of_computer_chair_l105_105119


namespace find_z_l105_105974

noncomputable def z : ℂ := -6 + 17i

theorem find_z :
  ∃ z : ℂ, z = -6 + 17i ∧ 2 * (5 - 3i) + z = 4 + 11i :=
begin
  use z,
  split,
  { refl },
  { 
    calc
      2 * (5 - 3i) + z = 10 - 6i + z : by ring
                    ... = 10 - 6i + (-6 + 17i) : by rw z
                    ... = 4 + 11i : by ring } 
end

end find_z_l105_105974


namespace fourth_worker_operated_three_looms_l105_105789

-- Defining the number of skilled workers and looms
axiom m : ℕ+ -- m is a positive natural number
axiom n : ℕ+ -- n is a positive natural number

-- Defining the notation a_{ij}
def a (i j : ℕ+) : ℕ := if (1 <= i ∧ i <= m) ∧ (1 <= j ∧ j <= n) then 1 else 0

-- Stating the question and providing the relevant proof goal
theorem fourth_worker_operated_three_looms (h : a 4 1 + a 4 2 + a 4 3 + ... + a 4 n = 3) : 
  ∑ j in finset.range n, a 4 j = 3 := sorry

end fourth_worker_operated_three_looms_l105_105789


namespace jose_wander_time_l105_105446

theorem jose_wander_time (distance : ℝ) (speed : ℝ) (h_distance : distance = 15) (h_speed : speed = 1.5) : 
  distance / speed = 10 :=
by
  rw [h_distance, h_speed]
  norm_num
  sorry

end jose_wander_time_l105_105446


namespace is_area_18_l105_105050

-- Define the given values
def hypotenuse : ℝ := 6 * Real.sqrt 2

-- Define the property of an isosceles right triangle
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * Real.sqrt 2

-- Define the conditions
def conditions (a : ℝ) : Prop :=
  is_isosceles_right_triangle a a hypotenuse

-- Define the area calculation
def triangle_area (a : ℝ) : ℝ :=
  0.5 * a * a

-- The theorem we want to prove
theorem is_area_18 (a : ℝ) (h : conditions a) : triangle_area a = 18 := by
  sorry

end is_area_18_l105_105050


namespace inequality_lemma_l105_105843

open Nat

variable {n : Nat}
variable {a : Fin n → ℝ}

-- Condition: all a_i are positive
axiom pos_a (i : Fin n) : 0 < a i

-- Condition: a_{k+1} - a_{k} ≥ 1 for all k
axiom diff_a (k : Fin (n - 1)) : 1 ≤ a (Fin.succ k) - a k

-- Theorem to prove
theorem inequality_lemma (n : Nat) (a : Fin n → ℝ) (pos_a : ∀ i, 0 < a i) (diff_a : ∀ k : Fin (n - 1), 1 ≤ a (Fin.succ k) - a k) :
    1 + (∏ i in Finset.range n, 1 / (a i - a 0)) ≤ ∏ i in Finset.range n, 1 / a i :=
  sorry

end inequality_lemma_l105_105843


namespace part_a_valid_pairs_part_b_infinite_k_l105_105791

/-- Part (a): There are 500 pawns in an m x n grid with pawns on each lattice point and center of unit squares. -/
theorem part_a_valid_pairs (m n : ℕ) (h : 2 * m * n + m + n + 1 = 500) :
  (m, n) = (1, 166) ∨ (m, n) = (4, 55) ∨ (m, n) = (13, 18) :=
begin
  sorry
end

/-- Part (b): There are infinitely many positive integers k such that no grid contains exactly k pawns. -/
theorem part_b_infinite_k : ∀ k : ℕ, ∃ p : ℕ, nat.prime p ∧ k = (p + 1) / 2 :=
begin
  sorry
end

end part_a_valid_pairs_part_b_infinite_k_l105_105791


namespace diff_largest_second_largest_l105_105363

def nums : set ℕ := {10, 11, 12}

theorem diff_largest_second_largest :
  let largest := 12 in
  let second_largest := 11 in
  largest - second_largest = 1 :=
by
  let largest := 12
  let second_largest := 11
  show largest - second_largest = 1
  sorry

end diff_largest_second_largest_l105_105363


namespace intersection_eq_l105_105298

open Set

def A := {-1, 0, 1, 2}
def B := {-2, 0, 2, 4}

theorem intersection_eq : A ∩ B = {0, 2} :=
by
  sorry

end intersection_eq_l105_105298


namespace negation_of_every_square_positive_l105_105056

theorem negation_of_every_square_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, n^2 ≤ 0 := sorry

end negation_of_every_square_positive_l105_105056


namespace area_ratio_l105_105435

noncomputable theory
open real

-- Define the geometric properties and midpoints
variable {A B C D E F G H : ℝ × ℝ}
variable (h1 : ∠ABC = 90)
variable (h2 : dist A B = 6)
variable (h3 : dist B C = 8)
variable (D : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ)
variable (hD : D = midpoint A C)
variable (hE : E = midpoint A B)
variable (hF : F = midpoint B C)
variable (G : ℝ × ℝ) (H : ℝ × ℝ)
variable (hG : G = midpoint D F)
variable (hH : H = midpoint E F)
variable (hAABC : dist A C = sqrt (A.1^2 + B.1^2)) -- where AC is the hypotenuse

-- Area of triangle ABC
def area_ABC := (1/2) * dist A B * dist B C

-- Area of quadrilateral ADHE using the shoelace formula
def area_ADHE := (1/2) * abs ((A.1 * (D.2 - H.2) + D.1 * (E.2 - A.2) + H.1 * (A.2 - E.2) + E.1 * (D.2 - H.2)) - 
                               (A.2 * (D.1 - H.1) + D.2 * (E.1 - A.1) + H.2 * (A.1 - E.1) + E.2 * (D.1 - H.1)))

-- The final theorem to prove the ratio of area_ADHE to area_ABC is 1/8
theorem area_ratio : area_ADHE / area_ABC = 1 / 8 := sorry

end area_ratio_l105_105435


namespace central_cell_value_l105_105392

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l105_105392


namespace proof_problem_l105_105264

-- Define the ellipse and conditions
def ellipse (a b : ℝ) (a_gt_b : a > b) := ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

def focus_points (c : ℝ) (F1 F2 : ℝ × ℝ) := 
  F1 = (-c, 0) ∧ F2 = (c, 0)

def point_on_ellipse (A : ℝ × ℝ) (a b : ℝ) :=
  A = (1, real.sqrt 2 / 2) ∧ ellipse a b (by sorry)

-- Condition for line with slope 2 intersecting ellipse at two distinct points
def line_intersect (a b : ℝ) : Prop :=
  ¬ ∃ (t : ℝ) (M N : ℝ × ℝ), 
  (2 > t ∧ t > -3) ∧ 
  (∃ P : ℝ × ℝ, P.2 = 5 / 3) ∧
  (2 * M.1 + t = M.2 ∧ 2 * N.1 + t = N.2) ∧ 
  ∃ Q : ℝ × ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ vector.eq (P.1 - M.1, P.2 - M.2) (N.1 - Q.1, N.2 - Q.2)

-- Proof problem statement
theorem proof_problem (a b : ℝ) (a_gt_b : a > b) (c : ℝ) (F1 F2 A : ℝ × ℝ): 
  ellipse a b a_gt_b → focus_points c F1 F2 → point_on_ellipse A a b →
  a = sqrt 2 ∧ b = 1 ∧ ellipse a^2 = 2 ∧ y^2 = 1  →
  line_intersect a b :=
begin
  intros h_ellipse h_focus_points h_point_on_ellipse h_standard_eq,
  -- Proof is left as an exercise
  sorry,
end

end proof_problem_l105_105264


namespace lark_combination_count_l105_105000

def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def prime_numbers : List ℕ := List.filter is_prime (List.range' 1 30)
def even_numbers : List ℕ := List.filter is_even (List.range' 1 30)
def multiples_of_4 : List ℕ := List.filter is_multiple_of_4 (List.range' 1 30)

theorem lark_combination_count :
  prime_numbers.length * even_numbers.length * multiples_of_4.length = 1050 := by
  sorry

end lark_combination_count_l105_105000


namespace maximum_distance_on_circle_l105_105255

open Complex Real

-- Define the relevant constants
noncomputable def z0 : ℂ := 1 - 2 * complex.I

-- Define the main statement
theorem maximum_distance_on_circle {z : ℂ} (hz : |z| = 3) : |z - z0| ≤ 3 + sqrt 5 :=
begin
  sorry
end

end maximum_distance_on_circle_l105_105255


namespace parabola_problem_l105_105457

open Real

noncomputable def parabola : Set (ℝ × ℝ) := {p | ∃ y, p = (y^2 / 12, y)}
def focus : ℝ × ℝ := (3, 0)
def origin : ℝ × ℝ := (0, 0)

def point_on_parabola (M : ℝ × ℝ) := M ∈ parabola
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
def area_of_triangle (A B C : ℝ × ℝ) : ℝ := 0.5 * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

theorem parabola_problem (M : ℝ × ℝ) :
  point_on_parabola M ∧ distance M focus = 5 → M.1 = 2 ∧ area_of_triangle origin M focus = 3 * sqrt 6 := 
by
  sorry

end parabola_problem_l105_105457


namespace simplify_factorial_l105_105496

theorem simplify_factorial :
  ∀ (f12 f10 f9 : ℕ), f12 = 12! ∧ f10 = 10! ∧ f9 = 9! →
  (f12 / (f10 + 3 * f9) = 4 / 3) :=
by
  sorry

end simplify_factorial_l105_105496


namespace central_cell_value_l105_105381

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l105_105381


namespace max_min_of_f_l105_105918

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem max_min_of_f : 
  ∃ a b : ℝ, (∀ x ∈ set.Icc (1 : ℝ) 4, f x ≤ a) ∧ (∀ x ∈ set.Icc (1 : ℝ) 4, b ≤ f x) ∧
  (∀ x ∈ set.Icc (1 : ℝ) 4, f 1 = b) ∧ (∀ y ∈ set.Icc (1 : ℝ) 4, f 4 = a) ∧ a = 9/5 ∧ b = 3/2 :=
by
  -- Proof omitted
  sorry

end max_min_of_f_l105_105918


namespace current_time_l105_105808

theorem current_time (t : ℝ) 
  (h1 : 6 * (t + 10) - (90 + 0.5 * (t - 5)) = 90 ∨ 6 * (t + 10) - (90 + 0.5 * (t - 5)) = -90) :
  t = 3 + 11 / 60 := sorry

end current_time_l105_105808


namespace problem_equivalent_l105_105135

theorem problem_equivalent :
  (¬ ∃ ε : ℝ, ε > 0 ∧ ε < 1 / ε) ∧
  (∀ (x y : ℝ), (y = x^2 - 1 → (x, y) ≠ y)) ∧
  ({1, 3 / 2, 6 / 4, abs (-1 / 2), 0.5}.card = 3) ∧
  (∀ (x y : ℝ), (xy ≤ 0 ↔ (x = 0 ∨ y = 0 ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)))) →
  0 = 4 :=
by 
  sorry

end problem_equivalent_l105_105135


namespace inequality_proof_l105_105144

variables {x y z a b c : ℝ}
variables (hx : 1 / 2 ≤ x ∧ x ≤ 2) (hy : 1 / 2 ≤ y ∧ y ≤ 2) (hz : 1 / 2 ≤ z ∧ z ≤ 2)
variables (h_perm : {a, b, c} = {x, y, z})

theorem inequality_proof :
  (60 * a^2 - 1) / (4 * x * y + 5 * z) +
  (60 * b^2 - 1) / (4 * y * z + 5 * x) +
  (60 * c^2 - 1) / (4 * z * x + 5 * y) ≥ 12 :=
by
  sorry

end inequality_proof_l105_105144


namespace games_before_championship_l105_105122

theorem games_before_championship 
    (total_points : ℕ) (avg_champ_score_diff : ℕ → ℕ) (opponent_score : ℕ) (lost_by : ℕ) (n : ℕ) :
    total_points = 720 →
    avg_champ_score_diff n = (720 / 2 / n) - 2 →
    opponent_score = 11 →
    lost_by = 2 →
    (let uf_champ_score := opponent_score + lost_by in 
    avg_champ_score_diff n = uf_champ_score →
    n = 24) :=
begin
  sorry
end

end games_before_championship_l105_105122


namespace num_valid_combinations_abc_d_l105_105819

theorem num_valid_combinations_abc_d
  {a b c d : ℕ}
  (h_digits : d > c ∧ c > b ∧ b > a ∧ a ≥ 0)
  (h_sum : a + b + c + d = 16) :
  ∃! combos, combos.card = 19 :=
sorry

end num_valid_combinations_abc_d_l105_105819


namespace problem_expression_value_l105_105717

variable (m n p q : ℝ)
variable (h1 : m + n = 0) (h2 : m / n = -1)
variable (h3 : p * q = 1) (h4 : m ≠ n)

theorem problem_expression_value : 
  (m + n) / m + 2 * p * q - m / n = 3 :=
by sorry

end problem_expression_value_l105_105717


namespace sum_of_binary_numbers_l105_105233
open Finset

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem sum_of_binary_numbers (n : ℕ) :
  (∑ k in (fintype.piFinset (λ i : fin (2 * n), fin 2)).filter (λ x, (x.filter (λ x, x = 1)).card = n), nat.of_digits 2 x) = 2 :=
sorry

end sum_of_binary_numbers_l105_105233


namespace ap_80th_term_l105_105771

/--
If the sum of the first 20 terms of an arithmetic progression is 200,
and the sum of the first 60 terms is 180, then the 80th term is -573/40.
-/
theorem ap_80th_term (S : ℤ → ℚ) (a d : ℚ)
  (h1 : S 20 = 200)
  (h2 : S 60 = 180)
  (hS : ∀ n, S n = n / 2 * (2 * a + (n - 1) * d)) :
  a + 79 * d = -573 / 40 :=
by {
  sorry
}

end ap_80th_term_l105_105771


namespace factorial_simplification_l105_105498

theorem factorial_simplification :
  (12.factorial : ℚ) / ((10.factorial : ℚ) + 3 * (9.factorial : ℚ)) = 1320 / 13 := by
sorry

end factorial_simplification_l105_105498


namespace three_digit_multiples_of_7_l105_105317

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l105_105317


namespace kite_area_correct_l105_105596

variable (scale_ratio : ℝ) (d1_map_inch d2_map_inch : ℝ)
variable (actual_area : ℝ)

def map_to_act (inches : ℝ) (scale : ℝ) : ℝ := inches * scale 

noncomputable def kite_area (diag1 diag2 : ℝ) : ℝ := 
  0.5 * diag1 * diag2

theorem kite_area_correct :
  scale_ratio = 100 ∧ d1_map_inch = 2 ∧ d2_map_inch = 10 ∧ 
  actual_area = kite_area (map_to_act d1_map_inch scale_ratio) (map_to_act d2_map_inch scale_ratio) →
  actual_area = 100000 := by
  intro h
  cases h with hsrl ht
  cases ht with d1 ht
  cases ht with d2 harea
  rw [map_to_act, map_to_act, kite_area, hsrl, d1, d2] at harea
  sorry

end kite_area_correct_l105_105596


namespace sum_of_first_20_terms_arithmetic_sequence_l105_105504

noncomputable def arithmetic_sequence (a_2 : ℝ) (d : ℝ) : ℕ → ℝ
| 0       => a_2 - d
| (n + 1) => (arithmetic_sequence a_2 d n) + d

def geometric_sequence_property (a_2 d : ℝ) : Prop :=
(2 * (2 + 3 * d) = (2 + d) ^ 2)

def sum_to_n (a₁ d : ℝ) (n : ℕ) : ℝ :=
n / 2 * (2 * a₁ + (n - 1) * d)

theorem sum_of_first_20_terms_arithmetic_sequence :
  ∃ d, d ≠ 0 ∧ geometric_sequence_property 2 d ∧
  let a₁ := 2 - d in sum_to_n a₁ d 20 = 380 :=
by
  sorry

end sum_of_first_20_terms_arithmetic_sequence_l105_105504


namespace regular_tetrahedron_properties_l105_105201

-- Definitions
def equilateral (T : Type) : Prop := sorry -- equilateral triangle property
def equal_sides (T : Type) : Prop := sorry -- all sides equal property
def equal_angles (T : Type) : Prop := sorry -- all angles equal property

def regular (H : Type) : Prop := sorry -- regular tetrahedron property
def equal_edges (H : Type) : Prop := sorry -- all edges are equal
def equal_edge_angles (H : Type) : Prop := sorry -- angles between two edges at the same vertex are equal
def congruent_equilateral_faces (H : Type) : Prop := sorry -- faces are congruent equilateral triangles
def equal_dihedral_angles (H : Type) : Prop := sorry -- dihedral angles between adjacent faces are equal

-- Theorem statement
theorem regular_tetrahedron_properties :
  ∀ (T H : Type), 
    (equilateral T → equal_sides T ∧ equal_angles T) →
    (regular H → 
      (equal_edges H ∧ equal_edge_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_dihedral_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_edge_angles H)) :=
by
  intros T H hT hH
  sorry

end regular_tetrahedron_properties_l105_105201


namespace circle_sum_of_center_and_radius_l105_105847

theorem circle_sum_of_center_and_radius :
  let C := { p : ℝ × ℝ | (p.1 ^ 2 + 8 * p.1 - 5 * p.2 = -p.2 ^ 2 + 2 * p.1) }
  let centerX : ℝ := -3
  let centerY : ℝ := 5 / 2
  let radius : ℝ := (61 / 4).sqrt
  (centerX + centerY + radius = (61.sqrt - 1) / 2) :=
by
  sorry

end circle_sum_of_center_and_radius_l105_105847


namespace diving_degree_of_difficulty_l105_105779

theorem diving_degree_of_difficulty
(score1 score2 score3 score4 score5 : ℝ)
(H : set.univ = {score1, score2, score3, score4, score5})
(H_scale : ∀ s ∈ {score1, score2, score3, score4, score5}, 1 ≤ s ∧ s ≤ 10)
(max_score = max score1 (max score2 (max score3 (max score4 score5))))
(min_score = min score1 (min score2 (min score3 (min score4 score5))))
(point_val : ℝ)
(H_point_val : point_val = 76.16) :
  (score1 + score2 + score3 + score4 + score5 - max_score - min_score) * (76.16 / (score1 + score2 + score3 + score4 + score5 - max_score - min_score)) = point_val → 
  (76.16 / (score1 + score2 + score3 + score4 + score5 - max_score - min_score)) = 3.2 := by
sorry

end diving_degree_of_difficulty_l105_105779


namespace number_of_three_digit_multiples_of_7_l105_105334

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l105_105334


namespace proof_equivalent_problem_l105_105725

variable {α : Type} [LinearOrderedField α]
variable (x : list α)
variable (n : ℕ)
variable (x_n1 : α)

-- Conditions
def initial_data (n : ℕ) (x : list α) : Prop :=
  x.length = n ∧ n ≥ 3 ∧ n ∈ {n | n > 0}

def median (x : list α) : α := sorry -- Assuming the definition of median
def mean (x : list α) : α := sorry -- Assuming the definition of mean
def variance (x : list α) : α := sorry -- Assuming the definition of variance

variable (y : α) (z : α)

axiom median_def : median x = y
axiom mean_def : mean x = y
axiom variance_def : variance x = z

def new_median (x : list α) : α := sorry -- Assuming the definition of new median
def new_mean (x : list α) : α := sorry -- Assuming the definition of new mean
def new_variance (x : list α) : α := sorry -- Assuming the definition of new variance

-- Conclusion
def avg_income_increases (x : list α) (x_n1 : α) : Prop :=
  new_mean (x ++ [x_n1]) > mean x

def median_remain_change (x : list α) (x_n1 : α) : Prop :=
  new_median (x ++ [x_n1]) = median x ∨ new_median (x ++ [x_n1]) > median x

def variance_increases (x : list α) (x_n1 : α) : Prop :=
  new_variance (x ++ [x_n1]) > variance x

theorem proof_equivalent_problem :
  ∀ (x : list α) (n : ℕ) (x_n1 : α), initial_data n x → 
  avg_income_increases x x_n1 ∧ 
  median_remain_change x x_n1 ∧ 
  variance_increases x x_n1 :=
by
  intros,
  sorry

end proof_equivalent_problem_l105_105725


namespace total_balls_without_holes_correct_l105_105011

variable (soccerBalls basketballs soccerBallsWithHoles basketballsWithHoles : ℕ)

def totalBallsWithoutHoles (soccerBalls basketballs soccerBallsWithHoles basketballsWithHoles : ℕ) : ℕ :=
  (soccerBalls - soccerBallsWithHoles) + (basketballs - basketballsWithHoles)

theorem total_balls_without_holes_correct
  (h1 : soccerBalls = 40)
  (h2 : basketballs = 15)
  (h3 : soccerBallsWithHoles = 30)
  (h4 : basketballsWithHoles = 7) :
  totalBallsWithoutHoles 40 15 30 7 = 18 :=
by
  unfold totalBallsWithoutHoles
  rw [h1, h2, h3, h4]
  norm_num
  -- final result should yield 18
  sorry

end total_balls_without_holes_correct_l105_105011


namespace average_percent_increase_per_year_l105_105532

-- Definitions and conditions
def initialPopulation : ℕ := 175000
def finalPopulation : ℕ := 297500
def numberOfYears : ℕ := 10

-- Statement to prove
theorem average_percent_increase_per_year : 
  ((finalPopulation - initialPopulation) / numberOfYears : ℚ) / initialPopulation * 100 = 7 := by
  sorry

end average_percent_increase_per_year_l105_105532


namespace three_digit_multiples_of_7_l105_105336

theorem three_digit_multiples_of_7 :
  ∃ n : ℕ, (n = ∑ k in finset.range (143 - 15), ∀ k ∈ finset.range (143 - 15), 100 ≤ 7 * (15 + k) ∧ 7 * (15 + k) ≤ 999) :=
sorry

end three_digit_multiples_of_7_l105_105336


namespace parabola_focus_l105_105226

theorem parabola_focus (a : ℝ) (h k x : ℝ) (hx : h = 0) (kx : k = 0) (a_eq : a = -1/16) :
  focus (y = -a * x^2) = (0, -4) :=
by
  sorry

end parabola_focus_l105_105226


namespace paula_paint_cans_l105_105870

-- Define the initial conditions
def initially_paint_capacity (A : ℕ) : Prop :=
  A = 42

def lost_cans (B : ℕ) : Prop :=
  B = 4

def remaining_paint_capacity (C : ℕ) : Prop :=
  C = 34

-- Define the relationship between the cans lost and the paint capacity
def paint_coverage_per_can (n : ℕ) : Prop :=
  (42 - (34 + 4) = 8) ∧ (8 / 4 = 2)

-- Define the objective: the number of cans needed to paint 34 rooms
def number_of_cans_used (N : ℕ) : Prop :=
  34 / 2 = N

theorem paula_paint_cans : ∃ N, 
  initially_paint_capacity 42 ∧ lost_cans 4 ∧ remaining_paint_capacity 34 ∧
  paint_coverage_per_can 2 ∧ number_of_cans_used N ∧ N = 17 :=
by {
  exist 17,
  repeat { split },
  sorry -- Proof
}

end paula_paint_cans_l105_105870


namespace pieces_of_chocolate_left_l105_105150

theorem pieces_of_chocolate_left (initial_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ) 
    (h1 : initial_boxes = 14) (h2 : given_away_boxes = 8) (h3 : pieces_per_box = 3) : 
    (initial_boxes - given_away_boxes) * pieces_per_box = 18 := 
by 
  -- The proof will be here
  sorry

end pieces_of_chocolate_left_l105_105150


namespace river_lengths_l105_105914

theorem river_lengths (x : ℝ) (dnieper don : ℝ)
  (h1 : dnieper = (5 / (19 / 3)) * x)
  (h2 : don = (6.5 / 9.5) * x)
  (h3 : dnieper - don = 300) :
  x = 2850 ∧ dnieper = 2250 ∧ don = 1950 :=
by
  sorry

end river_lengths_l105_105914


namespace ball_travel_distance_fourth_hit_l105_105607

theorem ball_travel_distance_fourth_hit :
  let initial_height := 150
  let rebound_ratio := 1 / 3
  let distances := [initial_height, 
                    initial_height * rebound_ratio, 
                    initial_height * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio]
  distances.sum = 294 + 1 / 3 := by
  sorry

end ball_travel_distance_fourth_hit_l105_105607


namespace central_cell_value_l105_105371

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l105_105371


namespace find_central_cell_l105_105418

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l105_105418


namespace central_cell_value_l105_105400

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l105_105400


namespace denomination_of_bill_l105_105008

def cost_berries : ℝ := 7.19
def cost_peaches : ℝ := 6.83
def change_received : ℝ := 5.98

theorem denomination_of_bill :
  (cost_berries + cost_peaches) + change_received = 20.0 := 
by 
  sorry

end denomination_of_bill_l105_105008


namespace BP_squared_max_l105_105828

-- Define the objects in the problem
variables (ω : Circle) (A B C T P : Point)
variables (AB : LineSegment) (CT : Line)
variables [Diameter AB ω] [OnCircle T ω] [Tangent CT ω] [PerpendicularFrom A CT P]
variables (AB_length : ℝ) (AB_length_eq : AB_length = 16)

-- Define the segment BP
noncomputable def BP : LineSegment := LineSegment.mk B P

-- Define the squared length of BP
def BP_squared : ℝ := (length BP) ^ 2

-- Proof statement
theorem BP_squared_max : BP_squared = 320 :=
sorry

end BP_squared_max_l105_105828


namespace diet_soda_ratio_l105_105594

def total_bottles : ℕ := 60
def diet_soda_bottles : ℕ := 14

theorem diet_soda_ratio : (diet_soda_bottles * 30) = (total_bottles * 7) :=
by {
  -- We're given that total_bottles = 60 and diet_soda_bottles = 14
  -- So to prove the ratio 14/60 is equivalent to 7/30:
  -- Multiplying both sides by 30 and 60 simplifies the arithmetic.
  sorry
}

end diet_soda_ratio_l105_105594


namespace part1_l105_105683

open Nat

theorem part1 (a b c : ℕ) (h_cop_coprime_ab : Nat.coprime a b)
  (h_cop_coprime_bc : Nat.coprime b c)
  (h_cop_coprime_ca : Nat.coprime c a)
  (h_a_lt_b : 1 < a ∧ a < b)
  (h_b_lt_c : b < c) :
  (a⁻¹ % b + b⁻¹ % c + c⁻¹ % a > Nat.sqrt a) := sorry

end part1_l105_105683


namespace water_height_in_cylinder_l105_105187

-- Definitions of given values
def cone_base_radius : ℝ := 15
def cone_height : ℝ := 15
def cylinder_base_radius : ℝ := 30

-- Volume of cone formula
def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Volume of cylinder formula
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

-- Given the volumes are equal and solving for the height of the water level in the cylinder
theorem water_height_in_cylinder : 
  (cylinder_volume cylinder_base_radius 1.25) = (cone_volume cone_base_radius cone_height) := 
by 
  sorry -- proof would go here, but is not required for this task

end water_height_in_cylinder_l105_105187


namespace farm_total_animals_equal_to_8_l105_105784

def number_of_animals (total_legs ducks legs_per_duck legs_per_dog : ℕ) : ℕ :=
  let total_legs_duck := ducks * legs_per_duck
  let remaining_legs := total_legs - total_legs_duck
  let dogs := remaining_legs / legs_per_dog
  ducks + dogs

theorem farm_total_animals_equal_to_8 (h_ducks : 4 = 4)
  (h_total_legs : 24 = 24)
  (h_legs_per_duck : 2 = 2)
  (h_legs_per_dog : 4 = 4) : number_of_animals 24 4 2 4 = 8 :=
by
  simp [number_of_animals]
  sorry

end farm_total_animals_equal_to_8_l105_105784


namespace polyhedron_vertices_l105_105601

theorem polyhedron_vertices (P : Type) [polyhedron P] 
  (h_faces : num_faces P = 6) 
  (h_edges : num_edges P = 12) : 
  num_vertices P = 8 := sorry

end polyhedron_vertices_l105_105601


namespace quadratic_has_two_distinct_real_roots_l105_105088

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end quadratic_has_two_distinct_real_roots_l105_105088


namespace area_of_right_triangle_l105_105802

theorem area_of_right_triangle (A B C: Type) [right_triangle A B C]
  (h_hypotenuse : dist A B = 60)
  (h_median_CA : ∀ x, is_median x (A, C) (y, x + 3))
  (h_median_CB : ∀ x, is_median x (B, C) (y, 2x + 4)) :
  area_triangle A B C = 400 :=
sorry

end area_of_right_triangle_l105_105802


namespace points_C_E_D_F_are_vertices_of_rhombus_l105_105156

-- Given conditions for the problem
variables {C D E F B : Point}
variable {circle1 : Circle}
variable {circle2 : Circle}
variable {secant : Line}
variable {perpendicular : Line}

-- Definitions based on conditions
def common_secant_intersects_circles_at_C_and_D (circle1 circle2 : Circle) (C D : Point) (secant : Line) : Prop :=
  secant ∈ (circle1.intersection_circles (circle2)) ∧ C, D ∈ secant

def line_through_B_perpendicular_to_secant_intersects_at_E_and_F (B E F: Point) (secant perpendicular : Line) : Prop :=
  perpendicular ⊥ secant ∧ perpendicular ∈ circle1 ∩ circle2 ∧ E, F ∈ perpendicular

-- The goal which requires proof
theorem points_C_E_D_F_are_vertices_of_rhombus (h1 : common_secant_intersects_circles_at_C_and_D circle1 circle2 C D secant)
                                              (h2 : line_through_B_perpendicular_to_secant_intersects_at_E_and_F B E F secant perpendicular) : 
  is_rhombus C E D F :=
sorry

end points_C_E_D_F_are_vertices_of_rhombus_l105_105156


namespace cos_identity_l105_105565

theorem cos_identity : cos(2018 * Real.pi - Real.pi / 6) = sqrt 3 / 2 := 
  sorry

end cos_identity_l105_105565


namespace focus_of_parabola_l105_105228

theorem focus_of_parabola (f : ℝ) : 
  (∀ (x: ℝ), x^2 + ((- 1 / 16) * x^2 - f)^2 = ((- 1 / 16) * x^2 - (f + 8))^2) 
  → f = -4 :=
by
  intro h
  sorry

end focus_of_parabola_l105_105228


namespace area_of_triangle_BCD_l105_105430

/-- Given the area of triangle ABC and the coordinates of points A, C, D, and B,
    prove that the area of triangle BCD is 144 square units. -/
theorem area_of_triangle_BCD : 
  ∀ (A C D B : ℝ × ℝ),
  let area_ABC := 36 in
  let h := 9 in
  A = (0, 0) →
  C = (8, 0) →
  D = (40, 0) →
  B = (12, 20) →
  1 / 2 * 32 * h = 144 :=
by
  sorry

end area_of_triangle_BCD_l105_105430


namespace g_not_even_or_odd_l105_105443

def g (x : ℝ) : ℝ := 5 ^ (x ^ 2 - 5) - |x - 1|

theorem g_not_even_or_odd :
  (∀ x : ℝ, g (-x) ≠ g x) ∧ (∀ x : ℝ, g (-x) ≠ -g x) :=
by
  sorry

end g_not_even_or_odd_l105_105443


namespace polynomial_symmetric_factor_l105_105845

variable {R : Type*} [CommRing R]

def is_factor (p q : R[X][Y]) : Prop := ∃ r : R[X][Y], q = p * r

theorem polynomial_symmetric_factor
  (p : R[X][Y])
  (h_sym : ∀ x y : R, p.eval x y = p.eval y x)
  (h_factor : ∀ x y : R, (x - y) | p.eval x y) :
  ∀ x y : R, (x - y)^2 | p.eval x y :=
sorry

end polynomial_symmetric_factor_l105_105845


namespace price_difference_l105_105935

theorem price_difference (total_cost shirt_price : ℝ) (h1 : total_cost = 80.34) (h2 : shirt_price = 36.46) :
  (total_cost - shirt_price) - shirt_price = 7.42 :=
by
  sorry

end price_difference_l105_105935


namespace frequency_machineA_frequency_machineB_K_squared_approx_quality_difference_confidence_l105_105116

-- Define the problem conditions
def MachineA_first_class := 150
def MachineA_second_class := 50
def MachineA_total := 200

def MachineB_first_class := 120
def MachineB_second_class := 80
def MachineB_total := 200

def total_products := 400

-- Frequencies
theorem frequency_machineA : (MachineA_first_class : ℚ) / MachineA_total = 3 / 4 := by
  sorry

theorem frequency_machineB : (MachineB_first_class : ℚ) / MachineB_total = 3 / 5 := by
  sorry

-- Calculate K^2
def a := MachineA_first_class
def b := MachineA_second_class
def c := MachineB_first_class
def d := MachineB_second_class
def n := total_products

def K_squared : ℚ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Prove K^2 approximation
theorem K_squared_approx : K_squared ≈ 10.256 := by
  sorry

-- Prove confidence level comparison
theorem quality_difference_confidence : K_squared > 6.635 := by
  sorry

end frequency_machineA_frequency_machineB_K_squared_approx_quality_difference_confidence_l105_105116


namespace exists_point_distance_sum_greater_than_perimeter_l105_105874

theorem exists_point_distance_sum_greater_than_perimeter (n : ℕ) (h_n : n ≥ 7) 
  (P : ℝ) (convex_n_gon : Type) [ConvexPolygon n convex_n_gon] :
  ∃ X : convex_n_gon, (∑ v in vertices convex_n_gon, dist X v) > P := sorry

end exists_point_distance_sum_greater_than_perimeter_l105_105874


namespace clubs_same_member_count_l105_105788

theorem clubs_same_member_count (village : Type) [inhabitants : Nonempty village]
  (clubs : Type) (membership : village → set clubs) (k : ℕ)
  (member_of_at_least_k_clubs : ∀ v, k ≤ (membership v).card)
  (at_most_one_common : ∀ c1 c2 : clubs, c1 ≠ c2 → 
    (∃! v : village, v ∈ membership c1 ∧ v ∈ membership c2 ∨ 
    ∀ v : village, v ∉ membership c1 ∨ v ∉ membership c2)) :
  ∃ (same_size_clubs : set (set village)), k ≤ same_size_clubs.card ∧ 
    ∃ (n : ℕ), ∀ (s ∈ same_size_clubs), s.card = n :=
  sorry

end clubs_same_member_count_l105_105788


namespace Gwen_shelves_of_picture_books_l105_105745

theorem Gwen_shelves_of_picture_books (total_books shelves_mystery_books books_per_shelf : ℕ) 
  (h_total_books : total_books = 72) 
  (h_shelves_mystery_books : shelves_mystery_books = 3)
  (h_books_per_shelf : books_per_shelf = 9) :
  ∃ shelves_picture_books : ℕ, shelves_picture_books = 5 :=
by
  let total_books := 72
  let shelves_mystery_books := 3
  let books_per_shelf := 9
  have h_mystery_books : shelves_mystery_books * books_per_shelf = 27 := rfl
  have h_picture_books : total_books - 27 = 45 := rfl
  have h_shelves_picture_books : 45 / books_per_shelf = 5 := rfl
  use 5
  exact rfl

end Gwen_shelves_of_picture_books_l105_105745


namespace complex_number_quadrant_l105_105531

theorem complex_number_quadrant :
  let z := (3 - complex.i) / (1 - complex.i) in
  let conj := (1 + complex.i) in
  let numerator := (3 - complex.i) * conj in
  let denominator := (1 - complex.i) * conj in
  let simplified := numerator / denominator in
  simplified.re > 0 ∧ simplified.im > 0 :=
by
  sorry

end complex_number_quadrant_l105_105531


namespace train_passing_post_time_l105_105578

theorem train_passing_post_time (train_length : ℝ) (train_speed_kmph : ℝ)
  (h_train_length : train_length = 90)
  (h_train_speed : train_speed_kmph = 44) : 
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  in train_length / train_speed_mps ≈ 7.36 := by
  sorry

end train_passing_post_time_l105_105578


namespace min_pos_period_max_value_l105_105919

def f (x : Real) : Real :=
  3 * sin (x / 4 + π / 6) - 1

theorem min_pos_period_max_value :
  (∃ T > 0, ∀ x : Real, f (x + T) = f x) ∧ (∀ x : Real, f x ≤ 2) :=
  sorry

end min_pos_period_max_value_l105_105919


namespace arithmetic_expression_l105_105637
-- We will prove that 2^0 - | -3 | + (-1/2) = - 5/2

theorem arithmetic_expression : 2^0 - | -3 | + ( - 1 / 2 : ℝ ) = - ( 5 / 2 : ℝ ) := 
by
  sorry

end arithmetic_expression_l105_105637


namespace range_of_f_greater_than_two_l105_105820

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 - 2*x - 1 else -2*x + 6

theorem range_of_f_greater_than_two:
  { t : ℝ | f t > 2 } = { t | t < 0 } ∪ { t | t > 3 } :=
sorry

end range_of_f_greater_than_two_l105_105820


namespace shares_of_stocks_they_can_buy_l105_105170

def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def months_of_savings : ℕ := 4
def cost_per_share : ℕ := 50

theorem shares_of_stocks_they_can_buy :
  (((weekly_savings_wife * 4) + monthly_savings_husband) * months_of_savings / 2) / cost_per_share = 25 :=
by
  -- sorry for the implementation
  sorry

end shares_of_stocks_they_can_buy_l105_105170


namespace oranges_taken_by_susan_l105_105104

-- Defining the conditions
def original_number_of_oranges_in_box : ℕ := 55
def oranges_left_in_box_after_susan_takes : ℕ := 20

-- Statement to prove:
theorem oranges_taken_by_susan :
  original_number_of_oranges_in_box - oranges_left_in_box_after_susan_takes = 35 :=
by
  sorry

end oranges_taken_by_susan_l105_105104


namespace five_mondays_in_november_l105_105894

theorem five_mondays_in_november (N : ℕ)
  (october_has_31_days : ∀ (d : ℕ), 1 ≤ d ∧ d ≤ 31) 
  (november_has_30_days : ∀ (d : ℕ), 1 ≤ d ∧ d ≤ 30)
  (five_fridays_in_october : ∃ F : ℕ → ℕ, (∀ i, 1 ≤ F i ∧ F i ≤ 31) ∧ (#(range F) = 5)) :
  ∃ M : ℕ → ℕ, (∀ i, 1 ≤ M i ∧ M i ≤ 30) ∧ (#(range M) = 5) :=
sorry

end five_mondays_in_november_l105_105894


namespace dance_total_people_l105_105067

theorem dance_total_people
  (ratio_b_g : 3 / 4)
  (num_girls : ℕ := 60)
  (teacher_percentage : ℕ := 20) :
  let num_parts := num_girls / 4;
      num_boys := num_parts * 3;
      num_teachers := num_boys * 20 / 100;
      total_people := num_girls + num_boys + num_teachers
  in
    total_people = 114 :=
by
  let num_parts := 60 / 4;
  let num_boys := num_parts * 3;
  let num_teachers := num_boys * 20 / 100;
  let total_people := 60 + num_boys + num_teachers;
  have : total_people = 114 := sorry,
  assumption

end dance_total_people_l105_105067


namespace trig_identity_l105_105692

theorem trig_identity (α : ℝ) (h : Real.tan (Real.pi - α) = -2) : 
  1 / (Real.cos (2 * α) + Real.cos(α) ^ 2) = -5 / 2 := 
by
  sorry

end trig_identity_l105_105692


namespace range_of_a_l105_105285

open Real

noncomputable def A (x : ℝ) : Prop := (x + 1) / (x - 2) ≥ 0
noncomputable def B (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≥ 0

theorem range_of_a :
  (∀ x, A x → B x a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l105_105285


namespace dot_product_parallel_condition_l105_105743

variables {a b : ℝ} [inner_product_space ℝ V]

-- Necessary and sufficient condition for parallel vectors
theorem dot_product_parallel_condition (a b : V) : 
  abs (inner (a, b)) = norm a * norm b ↔ is_parallel a b :=
sorry

end dot_product_parallel_condition_l105_105743


namespace sequence_form_l105_105649

theorem sequence_form {a : ℕ → ℚ} (h_eq : ∀ n : ℕ, a n * x ^ 2 - a (n + 1) * x + 1 = 0) 
  (h_roots : ∀ α β : ℚ, 6 * α - 2 * α * β + 6 * β = 3 ) (h_a1 : a 1 = 7 / 6) :
  ∀ n : ℕ, a n = (1 / 2) ^ n + 2 / 3 :=
by
  sorry

end sequence_form_l105_105649


namespace find_angle_and_area_l105_105434

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {m n : ℝ × ℝ}

-- Conditions
def side_opposite_angles (B A C : ℝ) (b a c : ℝ) : Prop := True -- Represents a triangular relationship
def vector_m (a c b : ℝ) : ℝ × ℝ := (2 * a - c, -b)
def vector_n (B C : ℝ) : ℝ × ℝ := (Real.cos B, Real.cos C)
def perpendicular (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0

-- Proof
theorem find_angle_and_area (h1 : side_opposite_angles B A C b a c)
                            (h2 : vector_m a c b = (2 * a - c, -b))
                            (h3 : vector_n B C = (Real.cos B, Real.cos C))
                            (h4 : perpendicular (2 * a - c, -b) (Real.cos B, Real.cos C))
                            (h5 : a = 3)
                            (h6 : b = Real.sqrt 19):
  (B = Real.pi / 3) ∧ (1/2 * a * 5 * Real.sin (Real.pi / 3) = 15 * Real.sqrt 3 / 4) :=
by
  sorry

end find_angle_and_area_l105_105434


namespace cube_root_59319_cube_root_195112_l105_105643

theorem cube_root_59319 :
  ∃ n : ℕ, n^3 = 59319 ∧ n = 39 :=
begin
  use 39,
  split,
  { norm_num, },
  { refl, },
end

theorem cube_root_195112 :
  ∃ n : ℕ, n^3 = 195112 ∧ n = 58 :=
begin
  use 58,
  split,
  { norm_num, },
  { refl, },
end

end cube_root_59319_cube_root_195112_l105_105643


namespace find_a_l105_105359

theorem find_a (a b : ℝ) 
(h1 : ∀ x < 1, deriv (λ x, 3*x^2 + 2*(a-1)*x + b) x < 0)
(h2 : ∀ x ≥ 1, deriv (λ x, 3*x^2 + 2*(a-1)*x + b) x > 0) : 
a = -2 := 
by
  sorry

end find_a_l105_105359


namespace height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l105_105220

-- Definitions
def Height (x : ℕ) : Prop := x = 140
def Weight (x : ℕ) : Prop := x = 23
def BookLength (x : ℕ) : Prop := x = 20
def BookThickness (x : ℕ) : Prop := x = 7
def CargoCapacity (x : ℕ) : Prop := x = 4
def SleepTime (x : ℕ) : Prop := x = 9
def TreeHeight (x : ℕ) : Prop := x = 12

-- Propositions
def XiaohongHeightUnit := "centimeters"
def XiaohongWeightUnit := "kilograms"
def MathBookLengthUnit := "centimeters"
def MathBookThicknessUnit := "millimeters"
def TruckCargoCapacityUnit := "tons"
def ChildrenSleepTimeUnit := "hours"
def BigTreeHeightUnit := "meters"

theorem height_is_centimeters (x : ℕ) (h : Height x) : XiaohongHeightUnit = "centimeters" := sorry
theorem weight_is_kilograms (x : ℕ) (w : Weight x) : XiaohongWeightUnit = "kilograms" := sorry
theorem book_length_is_centimeters (x : ℕ) (l : BookLength x) : MathBookLengthUnit = "centimeters" := sorry
theorem book_thickness_is_millimeters (x : ℕ) (t : BookThickness x) : MathBookThicknessUnit = "millimeters" := sorry
theorem cargo_capacity_is_tons (x : ℕ) (c : CargoCapacity x) : TruckCargoCapacityUnit = "tons" := sorry
theorem sleep_time_is_hours (x : ℕ) (s : SleepTime x) : ChildrenSleepTimeUnit = "hours" := sorry
theorem tree_height_is_meters (x : ℕ) (th : TreeHeight x) : BigTreeHeightUnit = "meters" := sorry

end height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l105_105220


namespace power_greater_than_any_l105_105125

theorem power_greater_than_any {p M : ℝ} (hp : p > 0) (hM : M > 0) : ∃ n : ℕ, (1 + p)^n > M :=
by
  sorry

end power_greater_than_any_l105_105125


namespace find_central_cell_l105_105414

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l105_105414


namespace largest_multiple_of_12_negation_l105_105959

theorem largest_multiple_of_12_negation (k : ℤ) (h1 : 12 * k = 144) (h2 : -12 * k > -150) : 12 * k = 144 :=
by
  unfold has_mul.mul
  unfold has_neg.neg
  sorry

end largest_multiple_of_12_negation_l105_105959


namespace picnic_problem_l105_105138

theorem picnic_problem
  (M W C A : ℕ)
  (h1 : M + W + C = 240)
  (h2 : M = W + 80)
  (h3 : A = C + 80)
  (h4 : A = M + W) :
  M = 120 :=
by
  sorry

end picnic_problem_l105_105138


namespace price_per_sheep_l105_105017

-- Definitions of the initial conditions
def ratio_goats_sheep : ℚ := 5 / 7
def total_animals : ℕ := 360
def price_per_goat : ℚ := 40
def total_money_made : ℚ := 7200
def goats_sold_fraction : ℚ := 1 / 2
def sheep_sold_fraction : ℚ := 2 / 3

-- Proof statement to prove that each sheep is sold for $30
theorem price_per_sheep 
  (G S : ℕ)
  (h1 : G / (G:ℚ + S:ℚ) = ratio_goats_sheep)
  (h2 : G + S = total_animals)
  (h3 : (G:ℚ * goats_sold_fraction) * price_per_goat + (S:ℚ * sheep_sold_fraction) * ?m_1 = total_money_made) : 
  ?m_1 = 30 := 
sorry

end price_per_sheep_l105_105017


namespace sum_reciprocal_and_sum_squares_l105_105536

variables (x y : ℝ)

theorem sum_reciprocal_and_sum_squares 
  (h₁ : x + y = 15) 
  (h₂ : x * y = 36) :
  (1 / x + 1 / y = 5 / 12) ∧ (x^2 + y^2 = 153) :=
by
  rw [←add_div, add_comm (1/x) (1/y)], -- Rewriting the sum of reciprocals
  simp only [*, div_eq_mul_inv, mul_inv_rev, mul_comm y x, inv_mul_eq_div, add_div], -- Apply identities and given conditions
  have: x + y = 15 := h₁,   
  have: x * y = 36 := h₂,
  sorry -- Proof will be inserted here

end sum_reciprocal_and_sum_squares_l105_105536


namespace g_value_at_50_l105_105465

noncomputable def g (x : ℝ) : ℝ := (1 - x) / 2

theorem g_value_at_50 :
  (∀ x y : ℝ, 0 < x → 0 < y → 
  (x * g y - y * g x = g (x / y) + x - y)) →
  g 50 = -24.5 :=
by
  intro h
  have h_g : ∀ x : ℝ, 0 < x → g x = (1 - x) / 2 := 
    fun x x_pos => sorry -- g(x) derivation proof goes here
  exact sorry -- Final answer proof goes here

end g_value_at_50_l105_105465


namespace num_four_digit_even_without_repetition_l105_105527

theorem num_four_digit_even_without_repetition : 
  let digits := [1, 2, 3, 4, 5]
  ∃ n : ℕ, n = 48 ∧
  num_valid_four_digit_even_numbers digits = n := 
  sorry

def num_valid_four_digit_even_numbers (digits : List ℕ) : ℕ :=
  -- Define the logic to calculate the number of valid four-digit even numbers
  sorry

end num_four_digit_even_without_repetition_l105_105527


namespace binary_representation_of_21_l105_105208

theorem binary_representation_of_21 : nat.binary_repr 21 = "10101" :=
sorry

end binary_representation_of_21_l105_105208


namespace steinburg_marching_band_l105_105938

theorem steinburg_marching_band :
  ∃ n : ℤ, n > 0 ∧ 30 * n < 1200 ∧ 30 * n % 34 = 6 ∧ 30 * n = 720 := by
  sorry

end steinburg_marching_band_l105_105938


namespace james_height_in_inches_l105_105811

def tree_height : ℝ := 60
def tree_shadow : ℝ := 20
def james_shadow_in_feet : ℝ := 25 / 12 -- 25 inches converted to feet

def shadow_ratio : ℝ := tree_height / tree_shadow

theorem james_height_in_inches :
  ∃ height_in_inches : ℝ, height_in_inches = (shadow_ratio * james_shadow_in_feet * 12) := -- Convert back to inches
sorry

end james_height_in_inches_l105_105811


namespace probability_at_least_one_passes_l105_105111

open Probability

theorem probability_at_least_one_passes (A B C : Event) (P : Prob) :
  P(A) = 1/3 ∧ P(B) = 1/3 ∧ P(C) = 1/3 ∧ indep_indep A B ∧ indep_indep B C ∧ indep_indep A C ->
  P(A ∪ B ∪ C) = 19/27 := by
  intros h
  sorry

end probability_at_least_one_passes_l105_105111


namespace translation_right_4_units_l105_105793

structure Point :=
  (x : Int)
  (y : Int)

def translate_right (p : Point) (d : Int) : Point :=
  { x := p.x + d, ..p }

theorem translation_right_4_units (P : Point) (h : P.x = -2 ∧ P.y = 3) :
  translate_right P 4 = { x := 2, y := 3 } :=
by
  sorry

end translation_right_4_units_l105_105793


namespace number_of_three_digit_multiples_of_7_l105_105333

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l105_105333


namespace area_weighted_sum_zero_l105_105774

open Real EuclideanGeometry

variables (A B C O : Point)
variables (S_A S_B S_C : ℝ)
variables (OA OB OC : vec ℝ 3)

def area_ΔABC (O : Point) : ℝ :=
  sorry -- Define the specific area calculation here

theorem area_weighted_sum_zero (A B C O : Point)
  (S_A S_B S_C : ℝ)
  (h_SA : S_A = area_ΔABC B C O)
  (h_SB : S_B = area_ΔABC C A O)
  (h_SC : S_C = area_ΔABC A B O)
  (OA OB OC : vec ℝ 3)
  (h_OA : OA = O - A)
  (h_OB : OB = O - B)
  (h_OC : OC = O - C) :
  S_A • OA + S_B • OB + S_C • OC = (0 : vec ℝ 3) := 
sorry

end area_weighted_sum_zero_l105_105774


namespace total_cost_janet_paid_l105_105612

-- Conditions definitions
def adult_ticket_price : ℕ := 30
def child_ticket_price := adult_ticket_price / 2
def discount := 0.20
def soda_price : ℕ := 5
def total_people : ℕ := 10
def children : ℕ := 4
def adults : ℕ := total_people - children

-- Prove the total cost
theorem total_cost_janet_paid : 
  adults * adult_ticket_price + children * child_ticket_price - discount * (adults * adult_ticket_price + children * child_ticket_price) + soda_price = 197 :=
by
  unfold adult_ticket_price child_ticket_price discount soda_price total_people children adults
  calc
  (6 * 30 + 4 * 15 : ℕ) - (0.20 * (6 * 30 + 4 * 15) : ℕ) + 5
  = (180 + 60) - (0.2 * (180 + 60)) + 5 : by sorry
  = 240 - 48 + 5 : by sorry
  = 197 : by sorry

end total_cost_janet_paid_l105_105612


namespace simplify_and_evaluate_l105_105889

-- Defining the conditions
def a : Int := -3
def b : Int := -2

-- Defining the expression
def expr (a b : Int) : Int := (3 * a^2 * b + 2 * a * b^2) - (2 * (a^2 * b - 1) + 3 * a * b^2 + 2)

-- Stating the theorem/proof problem
theorem simplify_and_evaluate : expr a b = -6 := by
  sorry

end simplify_and_evaluate_l105_105889


namespace new_average_weight_l105_105037

theorem new_average_weight (n : ℕ) (avg_weight : ℝ) (new_student_weight : ℝ) (new_n : ℕ) (new_avg_weight : ℝ) 
  (h₁ : n = 19) 
  (h₂ : avg_weight = 15) 
  (h₃ : new_student_weight = 13) 
  (h₄ : new_n = 20) 
  (h₅ : new_avg_weight = 14.9) 
  (h_total_weight : n * avg_weight + new_student_weight = 298)
  : new_avg_weight = (n * avg_weight + new_student_weight) / new_n := 
by
  rw [h₁, h₂, h₃, h₄, h₅] at h_total_weight
  have : 19 * 15 + 13 = 298 := by norm_num
  exact h_total_weight

end new_average_weight_l105_105037


namespace negation_proof_l105_105526

open Classical

variable {x : ℝ}

theorem negation_proof :
  (∀ x : ℝ, (x + 1) ≥ 0 ∧ (x^2 - x) ≤ 0) ↔ ¬ (∃ x_0 : ℝ, (x_0 + 1) < 0 ∨ (x_0^2 - x_0) > 0) := 
by
  sorry

end negation_proof_l105_105526


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l105_105118

theorem frequencies_of_first_class_products (total_A total_B first_class_A first_class_B : ℕ) 
  (h1 : total_A = 200)
  (h2 : first_class_A = 150)
  (h3 : total_B = 200)
  (h4 : first_class_B = 120) :
  (first_class_A / total_A = 3 / 4) ∧ (first_class_B / total_B = 3 / 5) :=
by {
  sorry
}

theorem confidence_in_difference_of_quality (total_A total_B first_class_A second_class_A first_class_B second_class_B : ℕ)
  (h1 : total_A = 200)
  (h2 : first_class_A = 150)
  (h3 : second_class_A = 50)
  (h4 : total_B = 200)
  (h5 : first_class_B = 120)
  (h6 : second_class_B = 80)
  (total : ℕ)
  (h7 : total = total_A + total_B)
  (k_value : ℝ)
  (h8 : k_value = 6.635):
  let K_squared := (total * (first_class_A * second_class_B - second_class_A * first_class_B) ^ 2) / 
                   ((first_class_A + second_class_A) * (first_class_B + second_class_B) * (first_class_A + first_class_B) * (second_class_A + second_class_B)) 
  in K_squared > k_value :=
by {
  sorry
}

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l105_105118


namespace relationship_between_abc_l105_105694

variable (a b c : ℝ)

theorem relationship_between_abc (h₁ : a = 2 ^ (1 / 2)) 
                                (h₂ : b = (Real.log 2) ^ (-1 / 2)) 
                                (h₃ : c = Real.log 2) : 
  c < b ∧ b < a :=
by
  sorry

end relationship_between_abc_l105_105694


namespace company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l105_105995

-- Define the conditions about the fishing company's boat purchase and expenses
def initial_purchase_cost : ℕ := 980000
def first_year_expenses : ℕ := 120000
def expense_increment : ℕ := 40000
def annual_income : ℕ := 500000

-- Prove that the company starts to make a profit in the third year
theorem company_starts_to_make_profit_in_third_year : 
  ∃ (year : ℕ), year = 3 ∧ 
  annual_income * year > initial_purchase_cost + first_year_expenses + (expense_increment * (year - 1) * year / 2) :=
sorry

-- Prove that the first option is more cost-effective
theorem first_option_more_cost_effective : 
  (annual_income * 3 - (initial_purchase_cost + first_year_expenses + expense_increment * (3 - 1) * 3 / 2) + 260000) > 
  (annual_income * 5 - (initial_purchase_cost + first_year_expenses + expense_increment * (5 - 1) * 5 / 2) + 80000) :=
sorry

end company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l105_105995


namespace tangent_line_equation_at_1_2_l105_105293

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem tangent_line_equation_at_1_2 :
  let x₀ := 1
  let y₀ := 2
  let slope := -2
  ∀ (x y : ℝ),
    y - y₀ = slope * (x - x₀) →
    2 * x + y - 4 = 0 :=
by
  sorry

end tangent_line_equation_at_1_2_l105_105293


namespace find_k_l105_105763

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 2)) (h2 : k ≠ 0) :
  k = 2 :=
sorry

end find_k_l105_105763


namespace smallest_num_is_1113805958_l105_105972

def smallest_num (n : ℕ) : Prop :=
  (n + 5) % 19 = 0 ∧ (n + 5) % 73 = 0 ∧ (n + 5) % 101 = 0 ∧ (n + 5) % 89 = 0

theorem smallest_num_is_1113805958 : ∃ n, smallest_num n ∧ n = 1113805958 :=
by
  use 1113805958
  unfold smallest_num
  simp
  sorry

end smallest_num_is_1113805958_l105_105972


namespace parallel_vectors_l105_105302

noncomputable def vector_a (m : ℝ) : ℝ × ℝ × ℝ := (m, 5, -1)
noncomputable def vector_b (r : ℝ) : ℝ × ℝ × ℝ := (3, 1, r)

theorem parallel_vectors (m r : ℝ) :
  (∃ λ : ℝ, vector_a m = (λ * fst (vector_b r), λ * snd (vector_b r), λ * trd (vector_b r))) →
  m * r = -3 :=
by
  sorry

end parallel_vectors_l105_105302


namespace Q_div_P_eq_10_over_3_l105_105895

noncomputable def solve_Q_over_P (P Q : ℤ) :=
  (Q / P = 10 / 3)

theorem Q_div_P_eq_10_over_3 (P Q : ℤ) (x : ℝ) :
  (∀ x, x ≠ 3 → x ≠ 4 → (P / (x + 3) + Q / (x^2 - 10 * x + 16) = (x^2 - 6 * x + 18) / (x^3 - 7 * x^2 + 14 * x - 48))) →
  solve_Q_over_P P Q :=
sorry

end Q_div_P_eq_10_over_3_l105_105895


namespace inequality_solution_l105_105222

theorem inequality_solution :
  {x : ℝ | (3 * x - 9) * (x - 4) / (x - 1) ≥ 0} = {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 3} ∪ {x : ℝ | x ≥ 4} :=
by
  sorry

end inequality_solution_l105_105222


namespace slopes_product_of_tangents_at_intersection_l105_105704

theorem slopes_product_of_tangents_at_intersection 
  (a1 b1 a2 b2 : ℝ) (P : ℝ × ℝ)
  (h1 : a1 > b1) (h2 : b1 > 0) (h3 : a2 > 0) (h4 : b2 > 0)
  (hE : P.1^2 / a1^2 + P.2^2 / b1^2 = 1)
  (hH : P.1^2 / a2^2 - P.2^2 / b2^2 = 1)
  (same_foci : (a1^2 - b1^2 = a2^2 + b2^2)) :
  let k1 := - (b1^2 * P.1) / (a1^2 * P.2),
      k2 := (b2^2 * P.1) / (a2^2 * P.2)
  in k1 * k2 = -1 :=
by
  sorry

end slopes_product_of_tangents_at_intersection_l105_105704


namespace count_multiples_l105_105314

theorem count_multiples (n : ℕ) : 
  n = 1 ↔ ∃ k : ℕ, k < 500 ∧ k > 0 ∧ k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0 ∧ k % 7 = 0 :=
by
  sorry

end count_multiples_l105_105314


namespace Cevas_Theorem_l105_105554

-- Assumptions
variable {A B C A1 B1 C1 : Point}
variable (m1 m2 m3 : ℝ)

-- Define the center of mass O
def center_of_mass (A B C : Point) (m1 m2 m3 : ℝ) : Point := 
(C.mul m3 + A.mul m1 + B.mul m2) / (m1 + m2 + m3)

-- Condition (12)
def condition_12 (A B C A1 B1 C1 : Point) : Prop :=
(* some mathematical condition here *)

-- Equations (13) to (15)
def condition_13_15 (A B C A1 B1 C1 : Point) : Prop :=
(* some mathematical equations here *)

-- Ceva's Theorem
theorem Cevas_Theorem (A B C A1 B1 C1 : Point) (h12 : condition_12 A B C A1 B1 C1)
 (h13_15 : condition_13_15 A B C A1 B1 C1):
  let O := center_of_mass A B C m1 m2 m3 in
  collinear (C, O, C1) ∧ collinear (A, O, A1) ∧ collinear (B, O, B1) :=
sorry

end Cevas_Theorem_l105_105554


namespace max_value_expression_eq_l105_105239

noncomputable def max_value_expression : ℝ :=
  real.Sup {v | ∃ x : ℝ, v = x^6 / (x^8 + 2*x^7 - 4*x^6 + 8*x^5 + 16*x^4)}

theorem max_value_expression_eq :
  max_value_expression = 1 / 12 :=
sorry

end max_value_expression_eq_l105_105239


namespace eq_fraction_l105_105759

def f(x : ℤ) : ℤ := 3 * x + 4
def g(x : ℤ) : ℤ := 2 * x - 1

theorem eq_fraction : (f (g (f 3))) / (g (f (g 3))) = 79 / 37 := by
  sorry

end eq_fraction_l105_105759


namespace apartment_living_room_size_l105_105854

theorem apartment_living_room_size :
  (∀ (a_total r_total r_living : ℝ), a_total = 160 → r_total = 6 → (∃ r_other, r_total = 5 + 1 ∧ r_living = 3 * r_other) → r_living = 60) :=
by
  intros a_total r_total r_living a_total_eq r_total_eq h
  cases h with r_other h'
  cases h' with r_total_eq' r_living_eq
  have r_other_eq : r_other = 20 :=
    by
      exact (by linarith : r_other = 20)
  rw [r_other_eq] at r_living_eq
  exact r_living_eq

end apartment_living_room_size_l105_105854


namespace central_cell_value_l105_105401

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l105_105401


namespace central_cell_value_l105_105394

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l105_105394


namespace central_cell_value_l105_105388

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l105_105388


namespace equation_of_ellipse_range_of_slope_l105_105703

def ellipse_eqn {a b : ℝ} (h_cond : a > b ∧ b > 0) : 
  eccentricity : ℝ := 
    (sqrt (2) / 2) 
  (focus_1 focus_2 : ℝ) :
  P (x y : ℝ) 
  F (focus_1 focus_2 : ℝ) : 
  Perpendicular_bisector (segment PF1) := by
  sorry

theorem equation_of_ellipse (a b : ℝ) (h_cond : a > b ∧ b > 0)
  (e : ℝ) (h_e : e = (sqrt (2) / 2))
  (c : ℝ) (h_c : c = sqrt (a^2 - b^2))
  (focus_1 focus_2 : ℝ)
  (h_focus : focus_1 = (-c, 0) ∧ focus_2 = (c, 0))
  (P : ℝ × ℝ) (h_P : P = (2, sqrt 3))
  (h_bisector : focus_2 lies on the perpendicular bisector of (segment P focus_1)) :
  ∃ (a b : ℝ), (b^2 = 1 ∧ a^2 = 2) ∧ (eq : ∀ x y, (x^2) / (2) + y^2 = 1) := by
  sorry

theorem range_of_slope (k : ℝ) (h_k : k ≠ 0) 
  {x1 x2 y1 y2 : ℝ} (h_intersection : (y = k * (x - 2)) ∧ (x = (2 x) / (1 + 2 (k^2))))
  (h_discriminant : 8 * (1 - 2 * k^2) > 0)
  (h_focus_angle : ∠ NF_2 F_1 = ∠ MF_2 A) :
  (x1 + x2 = (8 * k^2) / (1 + 2 * k^2) ∧ x1 * x2 = (8 * k^2 - 2) / (1 + 2 * k^2)) ∧ 
  ∃ k ∈ (ℝ), ((- sqrt 2 / 2) < k ∧ k < 0) ∨ (0 < k ∧ k < (sqrt 2 / 2)) := by
  sorry

end equation_of_ellipse_range_of_slope_l105_105703


namespace second_rate_is_correct_l105_105469

-- Define the constants and conditions
noncomputable def total_investment : ℝ := 6000
noncomputable def total_interest : ℝ := 624
noncomputable def first_investment : ℝ := 1800
noncomputable def first_rate : ℝ := 0.09
noncomputable def second_investment := total_investment - first_investment
noncomputable def first_interest := first_investment * first_rate

-- Define the rate of the second investment
def second_rate : ℝ := (total_interest - first_interest) / second_investment

-- The proof problem statement
theorem second_rate_is_correct : second_rate = 0.11 := 
by {
    calc
    second_rate = (total_interest - first_interest) / second_investment : by rw second_rate
               ... = (624 - 162) / 4200 : by {
                                let _ := total_investment,
                                let _ := total_interest,
                                let _ := first_investment,
                                let _ := first_rate,
                                let _ := second_investment,
                                let _ := first_interest,
                                rfl
                            }
               ... = 462 / 4200 : by {simp}
               ... = 0.11 : by sorry -- Simplifying 462 / 4200 gives 0.11 as a result
}

end second_rate_is_correct_l105_105469


namespace largest_n_property_l105_105676

open Nat

def is_odd_prime (p : ℕ) : Prop :=
  prime p ∧ p % 2 = 1

def condition_satisfied (n : ℕ) : Prop :=
  ∀ p, is_odd_prime p → p < n → prime (n - p)

theorem largest_n_property : 
  ∃ (n : ℕ), (condition_satisfied n) ∧ (∀ m > n, ¬ condition_satisfied m) :=
begin
  existsi 10,
  split,
  { intros p hp hpn,
    cases hp with prime_p odd_p,
    cases p,
    {exfalso, linarith},
    cases p,
    {exfalso, linarith},
    {cases p,
     {exfalso, linarith},
     {cases p,
      {norm_num,
       have h : 10 - ↑3 = 7, {norm_num}, rw h, exact prime_7},
      {cases p, {norm_num}, {exfalso, linarith}}}}},
  sorry -- Needs specific proofs for greater constraints.
end

end largest_n_property_l105_105676


namespace centroid_locus_l105_105954

noncomputable def locus_of_centroids : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), ∃ (m : ℝ), -1 ≤ m ∧ m ≤ 1 ∧ p.1 = m ∧ p.2 = (m^2 - 3) / 2

theorem centroid_locus :
  ∀ (p : ℝ × ℝ), locus_of_centroids p → p.2 = (p.1^2 / 2) - (3 / 2) ∧ -1 ≤ p.1 ∧ p.1 ≤ 1 :=
by sorry

end centroid_locus_l105_105954


namespace number_of_free_ranging_chickens_l105_105103

-- Define the conditions as constants
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def barn_chickens : ℕ := coop_chickens / 2
def total_chickens_in_coop_and_run : ℕ := coop_chickens + run_chickens    
def free_ranging_chickens_condition : ℕ := 2 * run_chickens - 4
def ratio_condition : Prop := total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + free_ranging_chickens_condition)
def target_free_ranging_chickens : ℕ := 105

-- The proof statement
theorem number_of_free_ranging_chickens : 
  total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + target_free_ranging_chickens) →
  free_ranging_chickens_condition = target_free_ranging_chickens :=
by {
  sorry
}

end number_of_free_ranging_chickens_l105_105103


namespace quadratic_roots_distinct_l105_105092

theorem quadratic_roots_distinct (m : ℝ) :
  let Δ := m^2 + 32 in
  Δ > 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 - 8 = 0 ∧ x2^2 + m * x2 - 8 = 0) :=
begin
  sorry
end

end quadratic_roots_distinct_l105_105092


namespace part1_part2_l105_105830

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 2 :=
by
  sorry

theorem part2 (x : ℝ) : (∀ b : ℝ, b ≠ 0 → f x ≥ (|2 * b + 1| - |1 - b|) / |b|) → (x ≤ -1.5 ∨ x ≥ 1.5) :=
by
  sorry

end part1_part2_l105_105830


namespace impossible_to_divide_into_seven_equal_regions_l105_105888

-- Definition of the circle and points inside it
variables {k : Circle} (A B C : Point) 

-- Hypotheses: Points are chosen inside the circle and are non-collinear
variables (hAin : inside_circle k A)
          (hBin : inside_circle k B)
          (hCin : inside_circle k C)
          (hNonCollinear : ¬collinear A B C)

-- The statement to be proven
theorem impossible_to_divide_into_seven_equal_regions : 
  ¬ (∃ regions : list (set Circle), regions.length = 7 ∧ (∀ r ∈ regions, area r = (π * radius k ^ 2) / 7)) :=
  by sorry

end impossible_to_divide_into_seven_equal_regions_l105_105888


namespace inequality_am_gm_l105_105281

variable {a b c : ℝ}

theorem inequality_am_gm (habc_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc_eq_1 : a * b * c = 1) : 
  a^3 + b^3 + c^3 + (a * b / (a^2 + b^2) + b * c / (b^2 + c^2) + c * a / (c^2 + a^2)) ≥ 9 / 2 := 
by
  sorry

end inequality_am_gm_l105_105281


namespace length_of_path_l105_105074

theorem length_of_path (L_total : ℝ) (hL : L_total = 73) : 3 * L_total = 219 :=
by
  rw [hL]
  norm_num
  sorry

end length_of_path_l105_105074


namespace daniel_practices_total_minutes_in_week_l105_105651

theorem daniel_practices_total_minutes_in_week :
  let school_minutes_per_day := 15
  let school_days := 5
  let weekend_minutes_per_day := 2 * school_minutes_per_day
  let weekend_days := 2
  let total_school_week_minutes := school_minutes_per_day * school_days
  let total_weekend_minutes := weekend_minutes_per_day * weekend_days
  total_school_week_minutes + total_weekend_minutes = 135 :=
by
  sorry

end daniel_practices_total_minutes_in_week_l105_105651


namespace central_cell_value_l105_105405

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l105_105405


namespace solve_diophantine_l105_105668

theorem solve_diophantine : ∀ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ (x^3 - y^3 = x * y + 61) → (x, y) = (6, 5) :=
by
  intros x y h
  sorry

end solve_diophantine_l105_105668


namespace prove_value_of_expressions_l105_105244

theorem prove_value_of_expressions (a b : ℕ) 
  (h₁ : 2^a = 8^b) 
  (h₂ : a + 2 * b = 5) : 
  2^a + 8^b = 16 := 
by 
  -- proof steps go here
  sorry

end prove_value_of_expressions_l105_105244


namespace find_m_l105_105827

open Complex

-- Define the set T based on the given conditions
def T : Set ℂ := {z | ∃ (x y : ℝ), z = x + y * Complex.I ∧ (Real.sqrt 3 / 2 ≤ x ∧ x ≤ 2 / Real.sqrt 3)}

-- The main theorem we want to prove
theorem find_m :
  ∃ (m : ℕ), (m = 12) ∧ ∀ (n : ℕ), (n ≥ m) → (∃ (z : ℂ), z ∈ T ∧ z^n = 1) :=
by
  let m := 12
  use m
  split
  · rfl
  · intro n hn
    sorry

end find_m_l105_105827


namespace original_price_of_trouser_l105_105814

-- Define conditions
def sale_price : ℝ := 20
def discount : ℝ := 0.80

-- Define what the proof aims to show
theorem original_price_of_trouser (P : ℝ) (h : sale_price = P * (1 - discount)) : P = 100 :=
sorry

end original_price_of_trouser_l105_105814


namespace largest_multiple_of_12_negation_greater_than_150_l105_105962

theorem largest_multiple_of_12_negation_greater_than_150 : 
  ∃ (k : ℤ), (k * 12 = 144) ∧ (-k * 12 > -150) :=
by
  -- Definitions and conditions
  let multiple_12 (k : ℤ) := k * 12
  have condition : -multiple_12 (-12) > -150 := by sorry
  existsi -12
  exact ⟨rfl, condition⟩

end largest_multiple_of_12_negation_greater_than_150_l105_105962


namespace max_material_utilization_rate_l105_105861

noncomputable def sphere_radius : ℝ := 2
noncomputable def cylinder_height : ℝ := 4

def volume_sphere (r : ℝ) : ℝ := (4/3) * π * r^3
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem max_material_utilization_rate : 
  ∃ r : ℝ, r = real.sqrt 2 ∧ 
  let V_sphere := volume_sphere sphere_radius in
  let V_cylinder := volume_cylinder r cylinder_height in
  V_cylinder / V_sphere = real.sqrt 3 / 3 :=
by 
  sorry

end max_material_utilization_rate_l105_105861


namespace john_glasses_of_water_l105_105474

theorem john_glasses_of_water (t_in_hours : ℝ) (g : ℝ) (drinking_rate : ℝ) :
  t_in_hours = 5 → drinking_rate = 20 → g = (t_in_hours * 60) / drinking_rate → g = 15 :=
begin
  intros h1 h2 h3,
  rw [h1, h2],
  norm_num at h3,
  exact h3,
end

end john_glasses_of_water_l105_105474


namespace corveus_lack_of_sleep_l105_105210

def daily_sleep_actual : ℕ := 4
def daily_sleep_recommended : ℕ := 6
def days_in_week : ℕ := 7

theorem corveus_lack_of_sleep : (daily_sleep_recommended - daily_sleep_actual) * days_in_week = 14 := 
by 
  sorry

end corveus_lack_of_sleep_l105_105210


namespace max_number_after_removal_l105_105295

-- Definition of the given number as a list of digits for analysis
def given_number : list ℕ := ["1", "2", "3", ..., "1", "0", "0"].map (λ c, c.to_nat)

-- Placeholder for the resultant number after removing 100 digits
def largest_possible_number : list ℕ := ["9", "9", "9", "9", "9", "7", "8", "5", "9", "6", "0", "6", "1", ..., "9", "9", "1", "0", "0"].map (λ c, c.to_nat)

-- The Lean statement, including the proof placeholder
theorem max_number_after_removal : ∃ l, l.length = given_number.length - 100 ∧ l = largest_possible_number := by
  sorry

end max_number_after_removal_l105_105295


namespace mean_median_difference_l105_105422

-- Define the percentages and scores as given in the conditions
def percent_60 := 0.20
def percent_75 := 0.25
def percent_85 := 0.40
def percent_95 := 1 - (percent_60 + percent_75 + percent_85)
def score_60 := 60
def score_75 := 75
def score_85 := 85
def score_95 := 95

-- Define the median score based on the given percentages
def median_score : Real := score_85

-- Calculate the mean score using the given percentages and scores
def mean_score : Real := (percent_60 * score_60) + (percent_75 * score_75) + (percent_85 * score_85) + (percent_95 * score_95)

-- State the theorem that the difference between the mean and median score is 6
theorem mean_median_difference : (median_score - mean_score) = 6 := by
  sorry

end mean_median_difference_l105_105422


namespace central_cell_value_l105_105398

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l105_105398


namespace magnitude_of_vector_a_when_perpendicular_l105_105299

variables (n : ℝ)
def vec_a : EuclideanSpace ℝ (Fin 3) := ![1, n, 2]
def vec_b : EuclideanSpace ℝ (Fin 3) := ![-2, 1, 2]
def vec_c : EuclideanSpace ℝ (Fin 3) := 2 • vec_a - vec_b

theorem magnitude_of_vector_a_when_perpendicular :
  (vec_c ⬝ vec_b = 0) → ∥vec_a∥ = Real.sqrt(21) / 2 :=
by
  sorry

end magnitude_of_vector_a_when_perpendicular_l105_105299


namespace min_value_g_max_value_M_l105_105291

noncomputable def f (x b : ℝ) : ℝ := x^2 - 2*b*x + b/4

noncomputable def g (b : ℝ) : ℝ :=
if h : 1 ≤ b ∧ b ≤ 4 then -b^2 + b/4
else if 4 < b then 16 - 31/4*b
else 0 -- This won't happen as b >= 1

noncomputable def M : ℝ := -3/4

theorem min_value_g (b : ℝ) (hb : 1 ≤ b) : 
  (if h : 1 ≤ b ∧ b ≤ 4 then -b^2 + b/4
  else if 4 < b then 16 - 31/4*b
  else 0) = g b :=
begin
  unfold g,
  split_ifs; 
  simp only [h, h_1]; 
  refl,
end

theorem max_value_M (b : ℝ) (hb : 1 ≤ b) : 
  (∀ b, g b ≤ -3/4) :=
begin
  sorry
end

end min_value_g_max_value_M_l105_105291


namespace proof_fraction_l105_105212

noncomputable def A' : ℝ :=
  ∑' n in {n : ℕ | n % 2 = 1 ∧ n % 3 ≠ 0}, (-1)^(n / 2) / n^3

noncomputable def B' : ℝ :=
  ∑' n in {n : ℕ | n % 2 = 1 ∧ n % 3 = 0}, (-1)^((n - 3) / 6) / n^3

theorem proof_fraction :
  A' / B' = 28 := by
  sorry

end proof_fraction_l105_105212


namespace poles_not_moved_l105_105905

-- Define the initial conditions
def original_distance : ℕ := 45 * (53 - 1)
def initial_interval : ℕ := 45
def total_poles : ℕ := 53
def new_interval : ℕ := 60

-- Theorem to prove the number of poles in between that do not need to be moved
theorem poles_not_moved : 
  ∃ (n : ℕ), n = (original_distance / nat.lcm initial_interval new_interval) - 1 ∧ n = 12 :=
by
  sorry

end poles_not_moved_l105_105905


namespace largest_number_is_minus_2a_l105_105762

-- Define the condition a = -3
def a : ℤ := -3

-- Define the set of expressions
noncomputable def expression_set : set ℤ := { -2 * a, 3 * a, 18 / a, a ^ 3, 2 }

-- Goal: Prove that the largest number in the set is -2a when a = -3.
theorem largest_number_is_minus_2a (h : a = -3) : 
  ∀ x ∈ expression_set, x ≤ -2 * a :=
sorry

end largest_number_is_minus_2a_l105_105762


namespace june_eggs_count_l105_105816

theorem june_eggs_count :
  (2 * 5) + 3 + 4 = 17 := 
by 
  sorry

end june_eggs_count_l105_105816


namespace count_ordered_triples_l105_105224

open Nat

theorem count_ordered_triples :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), 
  (∀ t ∈ s, let ⟨x, y, z⟩ := t in 
    lcm x y = 180 ∧ lcm x z = 840 ∧ lcm y z = 1260 ∧ gcd (gcd x y) z = 6) ∧
  s.card = 2 :=
sorry

end count_ordered_triples_l105_105224


namespace expression_equals_39_l105_105634

def expression : ℤ := (-2)^4 + (-2)^3 + (-2)^2 + (-2)^1 + 3 + 2^1 + 2^2 + 2^3 + 2^4

theorem expression_equals_39 : expression = 39 := by 
  sorry

end expression_equals_39_l105_105634


namespace number_of_three_digit_multiples_of_7_l105_105332

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l105_105332


namespace sum_of_properly_paintable_numbers_l105_105207

-- Definitions based on conditions
def properly_paintable (a b c : ℕ) : Prop :=
  ∀ n : ℕ, (n % a = 0 ∧ n % b ≠ 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b = 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b ≠ 1 ∧ n % c = 3) → n < 100

-- Main theorem to prove
theorem sum_of_properly_paintable_numbers : 
  (properly_paintable 3 3 6) ∧ (properly_paintable 4 2 8) → 
  100 * 3 + 10 * 3 + 6 + 100 * 4 + 10 * 2 + 8 = 764 :=
by
  sorry  -- The proof goes here, but it's not required

-- Note: The actual condition checks in the definition of properly_paintable 
-- might need more detailed splits into depending on specific post visits and a 
-- more rigorous formalization to comply with the exact checking as done above. 
-- This definition is a simplified logical structure to represent the condition.


end sum_of_properly_paintable_numbers_l105_105207


namespace value_of_x_l105_105976

theorem value_of_x (x : ℝ) (h : (sqrt x)^4 = 256) : x = 16 :=
sorry

end value_of_x_l105_105976


namespace largest_n_l105_105966

theorem largest_n : ∃ (n : ℕ), n < 1000 ∧ (∃ (m : ℕ), lcm m n = 3 * m * gcd m n) ∧ (∀ k, k < 1000 ∧ (∃ (m' : ℕ), lcm m' k = 3 * m' * gcd m' k) → k ≤ 972) := sorry

end largest_n_l105_105966


namespace quadratic_roots_distinct_l105_105081

-- Define the quadratic equation condition
def quadratic_eq : (ℝ → ℝ) :=
  λ x m => x^2 + m * x - 8

-- State the problem
theorem quadratic_roots_distinct (m : ℝ) : 
  let Δ := m^2 + 32 in Δ > 0 :=
  by
   -- We need this theorem to state that the equation always has distinct real roots
  let Δ := m^2 + 32
  sorry

end quadratic_roots_distinct_l105_105081


namespace find_central_cell_l105_105419

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l105_105419


namespace eq_of_plane_l105_105049

-- Definitions for the conditions
def foot_perpendicular_to_plane : ℝ × ℝ × ℝ := (10, -2, 5)

def plane_eq (A B C D : ℤ) (x y z : ℝ) := (A * x + B * y + C * z + D = 0)

-- The main theorem to prove
theorem eq_of_plane : 
  ∃ (A B C D : ℤ), A = 10 ∧ B = -2 ∧ C = 5 ∧ D = -129 ∧ 
  plane_eq A B C D (10 : ℝ) (-2 : ℝ) (5 : ℝ) ∧ 
  A > 0 ∧ Int.gcd (Int.natAbs A) (Int.natAbs B) (Int.natAbs C) (Int.natAbs D) = 1 :=
sorry

end eq_of_plane_l105_105049


namespace scientific_notation_of_viewers_l105_105863

def million : ℝ := 10^6
def viewers : ℝ := 70.62 * million

theorem scientific_notation_of_viewers : viewers = 7.062 * 10^7 := by
  sorry

end scientific_notation_of_viewers_l105_105863


namespace player_matches_l105_105599

theorem player_matches (n : ℕ) :
  (34 * n + 78 = 38 * (n + 1)) → n = 10 :=
by
  intro h
  have h1 : 34 * n + 78 = 38 * n + 38 := by sorry
  have h2 : 78 = 4 * n + 38 := by sorry
  have h3 : 40 = 4 * n := by sorry
  have h4 : n = 10 := by sorry
  exact h4

end player_matches_l105_105599


namespace sasha_wins_with_optimal_play_l105_105948

theorem sasha_wins_with_optimal_play :
  ∃ a b, a = 1703 ∧ b = 2022 ∧ (∀ turns left, winning_strategy a b turns left ) :=
begin
  -- Define the initial conditions
  let a := 1703,
  let b := 2022,
  
  -- Define the game condition for winning strategy
  def winning_strategy (a : ℕ) (b : ℕ) (turns : ℕ) : Prop := sorry, -- Describe the recursive strategy

  -- Show that the winning strategy exists for given initial values
  use [a, b],
  split, { refl }, split, { refl },
  intro turns_left,
  exact winning_strategy a b turns_left 
end

end sasha_wins_with_optimal_play_l105_105948


namespace fraction_of_income_from_tips_l105_105180

variable (S T : ℝ)
variable (salary_condition : T = 5 / 3 * S)

theorem fraction_of_income_from_tips (S_pos : S > 0) : T / (S + T) = 5 / 8 :=
by
  have h₁ : T = 5 / 3 * S := salary_condition
  have h₂ : S + T = (1 + 5 / 3) * S := by 
    rw [h₁]
    ring -- Use the ring to simplify the expression
  have h₃ : S + T = 8 / 3 * S := by 
    rw [←add_div, add_comm]
    norm_num
  have h₄ : T / (S + T) = (5 / 3 * S) / (8 / 3 * S) := by 
    rw [h₁, h₃]
  have h₅ : (5 / 3 * S) / (8 / 3 * S) = 5 / 8 := by 
    field_simp
    norm_num
  exact h₅

end fraction_of_income_from_tips_l105_105180


namespace expression_varies_l105_105909

noncomputable def expr (x : ℝ) : ℝ := (3 * x^2 - 2 * x - 5) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3))

theorem expression_varies (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) : ∃ y : ℝ, expr x = y ∧ ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → expr x₁ ≠ expr x₂ :=
by
  sorry

end expression_varies_l105_105909


namespace cos_seven_pi_over_six_l105_105631

open Real

theorem cos_seven_pi_over_six : cos (7 * π / 6) = - (sqrt 3 / 2) := 
by
  sorry

end cos_seven_pi_over_six_l105_105631


namespace kira_trip_duration_l105_105817

def city1_to_city2 := 3 -- hours by train
def break_at_city2 := 0.75 -- hours (45 minutes)
def city2_to_city3 := 1.5 -- hours by car rental (preferred since faster option)

def city3_to_city4 := 4 -- hours by train
def break_at_city4 := 0.5 -- hours (30 minutes)
def city4_to_city5 := 1.5 -- hours by bus (car rental already used)

def city5_to_city6 := 2 -- hours by train
def break_at_city6 := 0.25 -- hours (15 minutes)
def city6_to_city7 := 3 -- hours by bus

def total_time := 
    city1_to_city2 + break_at_city2 + 
    city2_to_city3 + city3_to_city4 + 
    break_at_city4 + city4_to_city5 + 
    city5_to_city6 + break_at_city6 + 
    city6_to_city7

theorem kira_trip_duration : total_time = 16.5 := by
  sorry

end kira_trip_duration_l105_105817


namespace Borya_wins_optimal_play_l105_105942

/--
There are 123 points equally spaced along the circumference of a circle.
Anya and Borya take turns coloring one of these points either blue or red.
The player who causes two adjacent points to be the same color loses.
Anya goes first.
--/
theorem Borya_wins_optimal_play :
  ∀ (points : Fin 123 → Prop) (colors : Fin 123 → Prop) (turn : ℕ → Prop),
  (∀ i, (i ∈ points) ∧ (‎colors i = blue ∨ colors i = red)) →
  (∀ i, ((colors i = colors (i + 1)) → colors (i + 1) = default)) →
  (0 ∈ turn → Anya wins) → (1 ∈ turn → Borya wins) →
  (∃ n, Borya wins).
proof :=
by {
  sorry
}

end Borya_wins_optimal_play_l105_105942


namespace find_pinions_l105_105937

-- Definitions for the given conditions
def production_rate (workers hours gearloks pinions : ℕ) : Prop :=
  (workers * hours = 120) ∧ (gearlocks = workers * hours * 3) ∧ (pinions = workers  * hours * 2)

theorem find_pinions (n : ℕ) :
  (production_rate 120 1 360 240) →
  (production_rate 80 3 480 600) →
  (∃ n, production_rate 40 4 160 n) →
  n = 80 :=
by
  sorry

end find_pinions_l105_105937


namespace range_of_a_l105_105722

noncomputable def f (x : ℝ) : ℝ := sorry -- The actual definition of the function f is not given
def g (a x : ℝ) : ℝ := a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc (-2 : ℝ) 2 → ∃ x₀ : ℝ, x₀ ∈ Set.Icc (-2 : ℝ) 2 ∧ g a x₀ = f x₁) ↔
  a ≤ -1/2 ∨ 5/2 ≤ a :=
by 
  sorry

end range_of_a_l105_105722


namespace simplify_fraction_l105_105489

theorem simplify_fraction : (12.factorial : ℚ) / (10.factorial + 3 * 9.factorial) = 1320 / 13 := by
  sorry

end simplify_fraction_l105_105489


namespace square_side_length_l105_105044

noncomputable def diagonal_in_inches : ℝ := 2 * Real.sqrt 2
noncomputable def inches_to_feet : ℝ := 1 / 12
noncomputable def diagonal_in_feet := diagonal_in_inches * inches_to_feet
noncomputable def factor_sqrt_2 : ℝ := 1 / Real.sqrt 2

theorem square_side_length :
  let diagonal_feet := diagonal_in_feet 
  let side_length_feet := diagonal_feet * factor_sqrt_2
  side_length_feet = 1 / 6 :=
sorry

end square_side_length_l105_105044


namespace find_second_sum_l105_105606

-- Define the total sum and rates
def total_sum : ℝ := 2743
def rate1 : ℝ := 0.03
def time1 : ℝ := 8
def rate2 : ℝ := 0.05
def time2 : ℝ := 3

-- Define the interest calculations
def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

-- Prove that the second part is Rs. 1688 given the conditions
theorem find_second_sum (x : ℝ) (h : interest x rate1 time1 = interest (total_sum - x) rate2 time2) :
  total_sum - x = 1688 :=
by
  -- Interest equality condition
  have interest_eq : interest x rate1 time1 = interest (total_sum - x) rate2 time2 := h
  
  -- Substitute the interest function
  rw [interest] at interest_eq
  
  -- Simplify the equation
  sorry

end find_second_sum_l105_105606


namespace repeating_decimal_subtraction_l105_105636

theorem repeating_decimal_subtraction :
  (let x := (567.to_rat / 999) in
  let y := (234.to_rat / 999) in
  let z := (345.to_rat / 999) in
  x - y - z = -4 / 333) :=
by sorry

end repeating_decimal_subtraction_l105_105636


namespace length_BD_l105_105796

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def B : point := (3, 9)
def C : point := (3, -3)
def D : point := (-2, -3)

theorem length_BD :
  distance B D = 13 :=
by
  -- Proof can be done using the given conditions and the pythagorean theorem
  sorry

end length_BD_l105_105796


namespace coprime_and_multiple_l105_105826

theorem coprime_and_multiple (n : ℕ) (S : Finset ℕ) (h₁ : ∀ x ∈ S, x ∈ Finset.range (2 * n + 1)) (h₂ : S.card = n + 1) :
  ∃ a b ∈ S, Nat.coprime a b ∧ ∃ c d ∈ S, c ≠ d ∧ (c % d = 0 ∨ d % c = 0) :=
  sorry

end coprime_and_multiple_l105_105826


namespace simplify_trig_identity_l105_105029

theorem simplify_trig_identity (a : ℝ) (h : a = 160 * (π / 180)) :
  1 / (sqrt (1 + tan a ^ 2)) = -cos a := 
by
  rw [h]
  -- Here we would proceed with simplifying in the proof steps, which are skipped.
  sorry

end simplify_trig_identity_l105_105029


namespace eccentricity_conic_proof_l105_105769

noncomputable def eccentricity_conic (a : ℝ) : ℝ :=
  if a > 0 then (↑ (sqrt (a - 1))) / (↑ (sqrt a))
  else if a < 0 then (↑ (sqrt (1 - a))) / 1
  else 0

theorem eccentricity_conic_proof (a : ℝ) (h : 1, 81 are geometric seq) :
  eccentricity_conic 9 = (2 * sqrt 2) / 3 ∨ 
  eccentricity_conic (-9) = sqrt 10 :=
sorry

end eccentricity_conic_proof_l105_105769


namespace initial_salmons_l105_105200

theorem initial_salmons (x : ℕ) (hx : 10 * x = 5500) : x = 550 := 
by
  sorry

end initial_salmons_l105_105200


namespace geometric_sequence_condition_l105_105829

variable (a b c : ℝ)

-- Condition: For a, b, c to form a geometric sequence.
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b ≠ 0) ∧ (b^2 = a * c)

-- Given that a, b, c are real numbers
-- Prove that ac = b^2 is a necessary but not sufficient condition for a, b, c to form a geometric sequence.
theorem geometric_sequence_condition (a b c : ℝ) (h : a * c = b^2) :
  ¬ (∃ b : ℝ, b^2 = a * c → (is_geometric_sequence a b c)) :=
sorry

end geometric_sequence_condition_l105_105829


namespace three_digit_multiples_of_7_l105_105325

theorem three_digit_multiples_of_7 : 
  ∃! n : ℕ, (n = 128) ∧ (∀ k, (100 ≤ 7 * k ∧ 7 * k ≤ 999) ↔ (15 ≤ k ∧ k ≤ 142)) :=
begin
  sorry
end

end three_digit_multiples_of_7_l105_105325


namespace part1_cyclic_harmonic_part2_concyclic_l105_105623

open BigOperators

-- Part 1: Proving MP * MT = NP * NT
theorem part1_cyclic_harmonic {ABC : Triangle} (M N P T : Point) (O : Circle) (CP_parallel_MN : Parallel CP MN)
  (midpoint_M : IsMidpoint M (Arc BC O)) (midpoint_N : IsMidpoint N (Arc AC O)) 
  (P_on_O : OnCircle P O) (T_on_O : OnCircle T O) (I_incenter : Incenter I ABC) (PI_extended : ExtendLine PI T O) :
  MP * MT = NP * NT := by 
  sorry

-- Part 2: Proving Q, T, Y, and X are concyclic
theorem part2_concyclic {ABC : Triangle} (M N P T Q X Y : Point) (O : Circle)
  (CP_parallel_MN : Parallel CP MN) 
  (midpoint_M : IsMidpoint M (Arc BC O)) (midpoint_N : IsMidpoint N (Arc AC O)) 
  (P_on_O : OnCircle P O) (T_on_O : OnCircle T O) (I_incenter : Incenter I ABC)
  (PI_extended : ExtendLine PI T O) (Q_on_arc_AB : OnArc Q (Arc AB O)) (Q_ne_ABC_T : Q ≠ A ∧ Q ≠ B ∧ Q ≠ T) 
  (X_incenter_AQC : Incenter X (Triangle A Q C)) (Y_incenter_BQC : Incenter Y (Triangle B Q C)) :
  Concyclic Q T Y X := by
  sorry

end part1_cyclic_harmonic_part2_concyclic_l105_105623


namespace part_I_part_II_l105_105713

def S_n (n : ℕ) : ℕ := 2^(n + 1) - 2

def a_n (n : ℕ) : ℕ := 2^n

def b_n (n : ℕ) : ℕ := n * 2^n

def T_n (n : ℕ) : ℕ :=
  (n - 1) * 2^(n + 1) + 2

theorem part_I (n : ℕ) (hn : n > 0) : a_n n = 2^n :=
sorry

theorem part_II (n : ℕ) (hn : n > 0) : ∑ i in Finset.range n, b_n (i + 1) = T_n n :=
sorry

end part_I_part_II_l105_105713


namespace word_appearance_through_glass_l105_105539

theorem word_appearance_through_glass (word : String) (viewed_from_inside : String) : 
  (word = "BANK") → (viewed_from_inside = "ИАИВ") :=
  by
    intros h,
    rw [h],
    -- The actual proof steps would go here, but we'll use 'sorry' as a placeholder.
    sorry

end word_appearance_through_glass_l105_105539


namespace degree_of_sum_is_3_l105_105032

noncomputable def degree_sum_polynomials (f g : ℤ[X]) : ℤ :=
(f + g).natDegree

theorem degree_of_sum_is_3
  (a0 a1 a2 a3 b0 b1 : ℤ)
  (h3 : a3 ≠ 0)
  (f := a3 * X^3 + a2 * X^2 + a1 * X + a0)
  (g := 0 * X^2 + b1 * X + b0) :
  degree_sum_polynomials f g = 3 := by
  sorry

end degree_of_sum_is_3_l105_105032


namespace complex_quadrant_l105_105249

open Complex

theorem complex_quadrant (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let z : ℂ := ⟨a, a - 1⟩ in
  z.re > 0 ∧ z.im < 0 := 
by {
  sorry
}

end complex_quadrant_l105_105249


namespace f4_is_even_l105_105835

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

def f (x : ℝ) : ℝ := 1 / (x^3 + 1)

def f1 (x : ℝ) : ℝ := f (x + 1)

def f2 (x : ℝ) : ℝ := f (2 * x)

def f3 (x : ℝ) : ℝ := f (x - 1)

def f4 (x : ℝ) : ℝ := f (x^2)

theorem f4_is_even : is_even_function f4 :=
sorry

end f4_is_even_l105_105835


namespace eval_f_f_neg2_l105_105732

noncomputable def f (x : ℝ) : ℝ :=
if x > 1 then log 2 x else (1 / 2) ^ x

theorem eval_f_f_neg2 : f (f (-2)) = 2 := by
  sorry

end eval_f_f_neg2_l105_105732


namespace find_value_added_l105_105666

theorem find_value_added (x y : ℕ) (hx : x = 3) (h : x + y = 60 * (1/x)) : y = 17 :=
by
  subst hx
  rw Nat.div_eq_of_lt (show 3 > 0 from Nat.succ_pos 2) at h
  exact h
  sorry

end find_value_added_l105_105666


namespace average_percentage_25_students_l105_105990

theorem average_percentage_25_students (s1 s2 : ℕ) (p1 p2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : p1 = 75) (h3 : s2 = 10) (h4 : p2 = 95) (h5 : n = 25) :
  ((s1 * p1 + s2 * p2) / n) = 83 := 
by
  sorry

end average_percentage_25_students_l105_105990


namespace problem_solution_l105_105202

noncomputable def problem_statement : Prop :=
  8 * (Real.cos (25 * Real.pi / 180)) ^ 2 - Real.tan (40 * Real.pi / 180) - 4 = Real.sqrt 3

theorem problem_solution : problem_statement :=
by
sorry

end problem_solution_l105_105202


namespace five_digit_palindromes_l105_105754

theorem five_digit_palindromes : 
  let first_digit := 9 in
  let second_digit := 10 in
  let third_digit := 10 in
  first_digit * second_digit * third_digit = 900 :=
by
  let first_digit := 9
  let second_digit := 10
  let third_digit := 10
  trivial

end five_digit_palindromes_l105_105754


namespace unit_vectors_have_same_magnitude_l105_105757

noncomputable def unit_vector (v : ℝ^3) : Prop :=
  ∥v∥ = 1

theorem unit_vectors_have_same_magnitude
  (a b : ℝ^3)
  (ha : unit_vector a) 
  (hb : unit_vector b) :
  ∥a∥ = ∥b∥ :=
by
  unfold unit_vector at ha hb
  rw [ha, hb]
  sorry

end unit_vectors_have_same_magnitude_l105_105757


namespace seating_arrangement_unique_l105_105626

theorem seating_arrangement_unique :
  ∃! (arrangement : ℕ → char), 
    (∀ i, 1 ≤ i ∧ i ≤ 10 → 
      (if i = 1 then arrangement i = 'M'
      else if i = 10 then arrangement i = 'E'
      else arrangement (i + 1) ≠ arrangement i)) ∧
    (∀ i, 1 < i → arrangement i = 
      (if arrangement (i - 1) = 'M' then 'V'
       else if arrangement (i - 1) = 'V' then 'E'
       else 'M')) :=
begin
  -- Proof skipped
  sorry
end

end seating_arrangement_unique_l105_105626


namespace sum_of_f_values_l105_105250

def f (x : ℝ) : ℝ := (2 / (2 ^ x + 1)) + Real.sin x

theorem sum_of_f_values : f (-2) + f (-1) + f 0 + f 1 + f 2 = 5 := by
  sorry

end sum_of_f_values_l105_105250


namespace central_cell_value_l105_105369

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l105_105369


namespace num_tables_is_multiple_of_239_l105_105708

-- Define the grid structure
structure Grid (rows cols : ℕ) :=
(numbers : Fin rows → Fin cols → ℕ)
(cond : ∀ c : Fin cols, numbers 0 c < numbers 1 c)
(row_inc1 : ∀ j : Fin (cols - 1), numbers 0 j ≤ numbers 0 (j + 1))
(row_inc2 : ∀ j : Fin (cols - 1), numbers 1 j ≤ numbers 1 (j + 1))

-- The main theorem statement
theorem num_tables_is_multiple_of_239
  (valid_tables_count : ℕ) :
  (∃ grid120 : Grid 2 120, valid_tables_count = number of such grids) → valid_tables_count % 239 = 0 :=
sorry

end num_tables_is_multiple_of_239_l105_105708


namespace isosceles_trapezoid_angle_45_l105_105266

theorem isosceles_trapezoid_angle_45 (A B C E D F G : Type)
  (h1 : is_isosceles_trapezoid A B C E)
  (h2 : base_of_trapezoid BC 3 4)
  (h3 : smaller_lateral_side AB 3)
  (h4 : on_segment D AE (ratios 3 1))
  (h5 : on_segment F AD (ratios 2 1))
  (h6 : on_segment G BD (ratios 1 2)) :
  measure_angle CFG = 45 := 
sorry

end isosceles_trapezoid_angle_45_l105_105266


namespace bus_stops_per_hour_l105_105142

noncomputable def time_stopped_per_hour (speed_no_stops : ℝ) (speed_with_stops : ℝ) : ℝ :=
  let distance_reduction := speed_no_stops - speed_with_stops
  let speed_per_minute := speed_no_stops / 60
  distance_reduction / speed_per_minute

theorem bus_stops_per_hour :
  time_stopped_per_hour 52 45 = 8.08 := by
  sorry

end bus_stops_per_hour_l105_105142


namespace Ada_initial_seat_l105_105235

-- Define initial positions
variables {A B C D E : ℕ}

-- Define conditions
theorem Ada_initial_seat :
  ∃ (A B C D E : ℕ),
    A ∈ {1, 2, 3, 4, 5} ∧
    B ∈ {1, 2, 3, 4, 5} ∧
    C ∈ {1, 2, 3, 4, 5} ∧
    D ∈ {1, 2, 3, 4, 5} ∧
    E ∈ {1, 2, 3, 4, 5} ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧ D ≠ E ∧
    Bea_new = B + 2 ∧
    Cee_new = D - 1 ∧
    (D, E) = (swap D E) ∧
    A = 1 →
  A = 2 :=
begin
  sorry
end

end Ada_initial_seat_l105_105235


namespace remaining_integer_after_steps_l105_105555

theorem remaining_integer_after_steps (list : List ℕ) (h : list = List.range (2013 + 1).tail) :
  ∃ n, (∀ (a b : ℕ), a ∈ list → b ∈ list → list.length = 1 → list.erase a (list.erase b list) = |a - b|) → n = 0 :=
by
  sorry

end remaining_integer_after_steps_l105_105555


namespace sufficient_not_necessary_condition_of_sin_l105_105849

open Real

theorem sufficient_not_necessary_condition_of_sin (θ : ℝ) :
  (abs (θ - π / 12) < π / 12) → (sin θ < 1 / 2) :=
sorry

end sufficient_not_necessary_condition_of_sin_l105_105849


namespace film_finishes_earlier_on_first_channel_l105_105552

theorem film_finishes_earlier_on_first_channel (n m : ℕ) (hn : 20 * n = 10 * m) :
  22 * n - 2 < 11 * m - 1 :=
by
  rw [hn, mul_comm 10 n, mul_assoc, div_eq_iff_eq_mul]; sorry

end film_finishes_earlier_on_first_channel_l105_105552


namespace trajectory_equation_maximum_area_AMBN_l105_105727

theorem trajectory_equation (x y: ℝ) (hC: x^2 + y^2 = 4) 
  (hP: ∃ (P: ℝ × ℝ), P ∈ {q : ℝ × ℝ | q.1^2 + q.2^2 = 4})
  (hH: ∃ (H: ℝ × ℝ), (H = (P.1, 0))) 
  (hQ: Q = (0.5 * (P.1 + H.1), 0.5 * (P.2 + H.2))) 
  (P: ℝ × ℝ) (Q: ℝ × ℝ) (H: ℝ × ℝ) : 
  (0.25 * x^2 + y^2 = 1) := sorry

theorem maximum_area_AMBN : ∃ (S: ℝ), S = 2 * Real.sqrt 2 := sorry

end trajectory_equation_maximum_area_AMBN_l105_105727


namespace meaningful_range_l105_105923

theorem meaningful_range (x : ℝ) : (x < 4) ↔ (4 - x > 0) := 
by sorry

end meaningful_range_l105_105923


namespace ziggy_song_requests_l105_105571

theorem ziggy_song_requests :
  ∃ T : ℕ, 
    (T = (1/2) * T + (1/6) * T + 5 + 2 + 1 + 2) →
    T = 30 :=
by 
  sorry

end ziggy_song_requests_l105_105571


namespace pizza_order_cost_l105_105883

def base_cost_per_pizza : ℕ := 10
def cost_per_topping : ℕ := 1
def topping_count_pepperoni : ℕ := 1
def topping_count_sausage : ℕ := 1
def topping_count_black_olive_and_mushroom : ℕ := 2
def tip : ℕ := 5

theorem pizza_order_cost :
  3 * base_cost_per_pizza + (topping_count_pepperoni * cost_per_topping) + (topping_count_sausage * cost_per_topping) + (topping_count_black_olive_and_mushroom * cost_per_topping) + tip = 39 := by
  sorry

end pizza_order_cost_l105_105883


namespace inequality_proof_l105_105710

noncomputable def given_condition_1 (a b c u : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ x, (a * x^2 - b * x + c = 0)) ∧
  a * u^2 - b * u + c ≤ 0

noncomputable def given_condition_2 (A B C v : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ (∃ x, (A * x^2 - B * x + C = 0)) ∧
  A * v^2 - B * v + C ≤ 0

theorem inequality_proof (a b c A B C u v : ℝ) (h1 : given_condition_1 a b c u) (h2 : given_condition_2 A B C v) :
  (a * u + A * v) * (c / u + C / v) ≤ (b + B) ^ 2 / 4 :=
by
    sorry

end inequality_proof_l105_105710


namespace overall_average_speed_round_trip_l105_105153

theorem overall_average_speed_round_trip :
  ∀ (d1 d2 d3 s1 s2 s3 : ℝ),
  d1 = 120 ∧ s1 = 60 ∧ 
  d2 = 80 ∧ s2 = 40 ∧ 
  d3 = 100 ∧ s3 = 50 →
  (let t1 := d1 / s1 in 
   let t2 := d2 / s2 in 
   let t3 := d3 / s3 in 
   let total_distance := d1 + d2 + d3 in 
   let total_time := t1 + t2 + t3 in 
   let avg_speed_to_sf := total_distance / total_time in 
   let r_t1 := 2 * t1 in 
   let r_t2 := 2 * t2 in 
   let r_t3 := 2 * t3 in 
   let total_return_time := r_t1 + r_t2 + r_t3 in 
   let round_trip_distance := 2 * total_distance in 
   let round_trip_time := total_time + total_return_time in 
   round_trip_distance / round_trip_time = 33.33) := true :=
by sorry

end overall_average_speed_round_trip_l105_105153


namespace central_cell_value_l105_105404

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l105_105404


namespace function_D_has_extreme_point_l105_105981

-- Definitions of the given functions
def f_A (x : ℝ) : ℝ := 2 / x
def f_B (x : ℝ) : ℝ := Real.exp x
def f_C (x : ℝ) : ℝ := Real.ln x
def f_D (x : ℝ) : ℝ := x^2 - 2 * x

-- The Lean statement for the proof problem
theorem function_D_has_extreme_point :
  ∃ x : ℝ, ∀ (y : ℝ) (f_A y = f_A y)
  ∧ ∀ (y : ℝ) (f_B y = f_B y)
  ∧ ∀ (y : ℝ) (f_C y = f_C y)
  ∧ (deriv f_D x = 0) ∧ (deriv f_D x ≠ 0) := sorry

end function_D_has_extreme_point_l105_105981


namespace find_x_l105_105345

theorem find_x (x : ℝ) (h1 : sin (π / 2 - x) = - (√3) / 2) (h2 : π < x ∧ x < 2 * π) : x = 7 * π / 6 := by
sorry

end find_x_l105_105345


namespace two_digit_number_pairs_l105_105574

theorem two_digit_number_pairs :
  ∃ (pairs : Finset (ℕ × ℕ)), 
  (∀ pair ∈ pairs, 
    let (a, b) := pair in 
    10 ≤ a ∧ a < 100 ∧ 
    10 ≤ b ∧ b < 100 ∧ 
    10 ≤ a + 20 ∧ a + 20 < 100 ∧ 
    10 ≤ b - 15 ∧ b - 15 < 100 ∧ 
    (a + 20) * (b - 15) = a * b) ∧ 
  card pairs = 16 :=
by sorry

end two_digit_number_pairs_l105_105574


namespace friends_total_l105_105470

-- Define the conditions as constants
def can_go : Nat := 8
def can't_go : Nat := 7

-- Define the total number of friends and the correct answer
def total_friends : Nat := can_go + can't_go
def correct_answer : Nat := 15

-- Prove that the total number of friends is 15
theorem friends_total : total_friends = correct_answer := by
  -- We use the definitions and the conditions directly here
  sorry

end friends_total_l105_105470


namespace largest_multiple_of_12_negation_greater_than_150_l105_105961

theorem largest_multiple_of_12_negation_greater_than_150 : 
  ∃ (k : ℤ), (k * 12 = 144) ∧ (-k * 12 > -150) :=
by
  -- Definitions and conditions
  let multiple_12 (k : ℤ) := k * 12
  have condition : -multiple_12 (-12) > -150 := by sorry
  existsi -12
  exact ⟨rfl, condition⟩

end largest_multiple_of_12_negation_greater_than_150_l105_105961


namespace watch_cost_price_l105_105987

theorem watch_cost_price (CP : ℝ) (h1 : (0.90 * CP) + 280 = 1.04 * CP) : CP = 2000 := 
by 
  sorry

end watch_cost_price_l105_105987


namespace distinct_solution_count_l105_105307

theorem distinct_solution_count : 
  ∃! x : ℝ, |x - 3| = |x + 5| :=
begin
  sorry
end

end distinct_solution_count_l105_105307


namespace permutation_digit_remainders_l105_105654

theorem permutation_digit_remainders {a b c : ℕ} :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∀ (x y z : ℕ), ( x = a ∧ y = b ∧ z = c ∨ 
                    x = a ∧ y = c ∧ z = b ∨ 
                    x = b ∧ y = a ∧ z = c ∨ 
                    x = b ∧ y = c ∧ z = a ∨ 
                    x = c ∧ y = a ∧ z = b ∨ 
                    x = c ∧ y = b ∧ z = a) →
   (100 * x + 10 * y + z) % 4 = (100 * a + 10 * b + c) % 4) →
  (a % 4 = b % 4 ∧ b % 4 = c % 4) ∨ 
  (∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
                   {x, y, z} = {1, 5, 9}) :=
begin
  intros h1 h2,
  sorry  -- This is where the proof would go
end

end permutation_digit_remainders_l105_105654


namespace like_terms_l105_105033

-- definitions based on conditions
def contains_same_letters (term1 term2 : String) : Prop := ∀ c : Char, c ∈ term1 ↔ c ∈ term2
def same_exponents (term1 term2 : List (Char × Nat)) : Prop := ∀ c : Char, (term1.filter (λ t => t.1 = c)).length = (term2.filter (λ t => t.1 = c)).length

-- the proof statement for the problem
theorem like_terms (term1 term2 : List (Char × Nat)) (s1 s2 : String)
    (h1 : contains_same_letters s1 s2)
    (h2 : same_exponents term1 term2) : 
    contains_same_letters s1 s2 ∧ same_exponents term1 term2 ↔ "like terms" := sorry

end like_terms_l105_105033


namespace least_positive_integer_mod_cond_l105_105969

theorem least_positive_integer_mod_cond (N : ℕ) :
  (N % 6 = 5) ∧ 
  (N % 7 = 6) ∧ 
  (N % 8 = 7) ∧ 
  (N % 9 = 8) ∧ 
  (N % 10 = 9) ∧ 
  (N % 11 = 10) →
  N = 27719 :=
by
  sorry

end least_positive_integer_mod_cond_l105_105969


namespace percentage_error_in_side_measurement_l105_105620

theorem percentage_error_in_side_measurement :
  (forall (S S' : ℝ) (A A' : ℝ), 
    A = S^2 ∧ A' = S'^2 ∧ (A' - A) / A * 100 = 25.44 -> 
    (S' - S) / S * 100 = 12.72) :=
by
  intros S S' A A' h
  sorry

end percentage_error_in_side_measurement_l105_105620


namespace solution_set_of_inequality_l105_105929

theorem solution_set_of_inequality :
  ∀ x : ℝ, |2 * x^2 - 1| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end solution_set_of_inequality_l105_105929


namespace binom_12_2_l105_105205

-- Definition of binomial coefficient using factorials.
def binom : ℕ → ℕ → ℕ
| n k := (n.factorial) / (k.factorial * (n - k).factorial)

-- Statement of the problem.
theorem binom_12_2 : binom 12 2 = 66 :=
by {
  -- Proof would go here.
  sorry
}

end binom_12_2_l105_105205


namespace acute_angle_root_range_l105_105615

theorem acute_angle_root_range (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  (∀ x : ℝ, 3 * x^2 * real.sin α - 4 * x * real.cos α + 2 = 0 → x ∈ ℝ) ↔ (0 < α ∧ α ≤ π / 6) :=
by
  sorry

end acute_angle_root_range_l105_105615


namespace quadratic_has_two_distinct_real_roots_l105_105098

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  let a := 1
      c := -8
      b := m
      Δ := b^2 - 4 * a * c 
  in (Δ > 0) :=
by
  let a := 1
  let c := -8
  let b := m
  let Δ := b^2 - 4 * a * c
  show Δ > 0
  sorry

end quadratic_has_two_distinct_real_roots_l105_105098


namespace count_five_digit_palindromes_l105_105751

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 = d5 ∧ d2 = d4

theorem count_five_digit_palindromes : 
  (finset.filter is_five_digit_palindrome (finset.Icc 10000 99999)).card = 900 :=
by
  sorry

end count_five_digit_palindromes_l105_105751


namespace final_box_balls_l105_105424

theorem final_box_balls (r b : ℕ → ℕ) 
  (h1_boxes : ∀ i, 1 ≤ i ∧ i ≤ 10) 
  (h2_no_empty : ∀ i, (r i ≠ 0) ∨ (b i ≠ 0)) 
  (h3_non_decreasing : ∀ i j, (1 ≤ i ∧ i < j ∧ j ≤ 10) → (r i + b i ≤ r j + b j)) 
  (h4_unique_comb : ∀ i j, (1 ≤ i ∧ i < j ∧ j ≤ 10) → (r i ≠ r j ∨ b i ≠ b j)) 
  (sum_red : (∑ i in finset.range 10, r (i+1)) = 14)
  (sum_blue : (∑ i in finset.range 10, b (i+1)) = 10) :
  (r 10 = 4 ∧ b 10 = 0) :=
sorry

end final_box_balls_l105_105424


namespace coefficient_x4y2_in_expansion_l105_105039

theorem coefficient_x4y2_in_expansion :
  (coefficient_of_term (x^4 * y^2) (expand (x - 1/x + y)^8) = -168) :=
sorry

end coefficient_x4y2_in_expansion_l105_105039


namespace shaded_area_of_pattern_in_foot_length_l105_105487

theorem shaded_area_of_pattern_in_foot_length
  (d : ℝ) (length : ℝ)
  (h1 : d = 2) (h2 : length = 12) :
  let r := d / 2
  in (length / d) * π * r^2 = 6 * π :=
by
  sorry

end shaded_area_of_pattern_in_foot_length_l105_105487


namespace number_of_friends_gave_money_l105_105681

-- Definition of given data in conditions
def amount_per_friend : ℕ := 6
def total_amount : ℕ := 30

-- Theorem to be proved
theorem number_of_friends_gave_money : total_amount / amount_per_friend = 5 :=
by
  sorry

end number_of_friends_gave_money_l105_105681


namespace solve_inequality_l105_105893

theorem solve_inequality (x : ℝ) : 
  (3 * x - 9) / (x - 3) ^ 2 < 0 ↔ x ∈ Iio 3 :=
  sorry

end solve_inequality_l105_105893


namespace min_decimal_digits_l105_105559

theorem min_decimal_digits (numerator : ℕ) (p q r : ℕ) 
  (h_num : numerator = 987654321)
  (h_p : p = 30)
  (h_q : q = 2)
  (h_r : r = 3) :
  ∃ (d : ℕ), d = 30 :=
by {
  use 30,
  sorry
}

end min_decimal_digits_l105_105559


namespace increasing_f_solve_inequality_l105_105583

-- Definitions and conditions from the problem
def f : ℝ → ℝ
def cond1 (x y : ℝ) : Prop := f x + f y = f (x + y) + 2
def cond2 (x : ℝ) : Prop := x > 0 → f x > 2

-- Statement 1: Prove that f(x) is an increasing function on ℝ
theorem increasing_f (cond1 : ∀ x y : ℝ, cond1 x y) (cond2 : ∀ x : ℝ, cond2 x) :
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := sorry

-- Definitions and conditions for Part 2
def cond3 : Prop := f 3 = 5

-- Statement 2: Prove the solution set for the inequality
theorem solve_inequality (cond1 : ∀ x y : ℝ, cond1 x y) (cond2 : ∀ x : ℝ, cond2 x) (cond3 : cond3) :
  {a : ℝ | f (a^2 - 2 * a - 2) < 3} = {a : ℝ | -1 < a ∧ a < 3} := sorry

end increasing_f_solve_inequality_l105_105583


namespace problem_l105_105356

variable (a : Int)
variable (h : -a = 1)

theorem problem : 3 * a - 2 = -5 :=
by
  -- Proof will go here
  sorry

end problem_l105_105356


namespace num_words_is_16_l105_105304

-- Define the set of letters
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the problem as the number of 3-letter words with at least two 'A's
def num_words_with_at_least_two_A : ℕ :=
  let all_words := (Finset.pi Finset.univ (fun _ => letters)).to_finset in
  let words_with_at_least_two_A := all_words.filter
    (λ s, (s.count 'A' ≥ 2)) in
  words_with_at_least_two_A.card

-- Translate the proof problem
theorem num_words_is_16 : num_words_with_at_least_two_A = 16 := by
  -- Proof omitted
  sorry

end num_words_is_16_l105_105304


namespace total_balls_without_holes_correct_l105_105010

variable (soccerBalls basketballs soccerBallsWithHoles basketballsWithHoles : ℕ)

def totalBallsWithoutHoles (soccerBalls basketballs soccerBallsWithHoles basketballsWithHoles : ℕ) : ℕ :=
  (soccerBalls - soccerBallsWithHoles) + (basketballs - basketballsWithHoles)

theorem total_balls_without_holes_correct
  (h1 : soccerBalls = 40)
  (h2 : basketballs = 15)
  (h3 : soccerBallsWithHoles = 30)
  (h4 : basketballsWithHoles = 7) :
  totalBallsWithoutHoles 40 15 30 7 = 18 :=
by
  unfold totalBallsWithoutHoles
  rw [h1, h2, h3, h4]
  norm_num
  -- final result should yield 18
  sorry

end total_balls_without_holes_correct_l105_105010


namespace total_boys_in_class_l105_105986

theorem total_boys_in_class (x y : ℕ) (h1 : x = 19) (h2 : y = 19) : x + y - 1 = 37 := by
  rw [h1, h2]
  norm_num
  sorry

end total_boys_in_class_l105_105986


namespace sum_of_solutions_l105_105133

theorem sum_of_solutions (x : ℝ) :
  (∃ x, 2 * (x + 5).abs = 18) → (finset.univ.filter (λ x, 2 * (x + 5).abs = 18)).sum id = -10 := by
sorry

end sum_of_solutions_l105_105133


namespace prime_power_divides_binomial_l105_105463

theorem prime_power_divides_binomial {p n k α : ℕ} (hp : Nat.Prime p) 
  (h : p^α ∣ Nat.choose n k) : p^α ≤ n := 
sorry

end prime_power_divides_binomial_l105_105463


namespace find_central_cell_l105_105415

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l105_105415


namespace prob_A_is_half_prob_B_is_nine_sixteenth_possible_values_Y_range_most_likely_Y_is_34_l105_105510

open_locale big_operators

noncomputable def prob_A : ℚ :=
  (nat.choose 59 29) / (nat.choose 60 30)

theorem prob_A_is_half : prob_A = 1 / 2 := sorry

noncomputable def prob_B : ℚ :=
  ((nat.choose 59 44)^2) / ((nat.choose 60 45)^2)

theorem prob_B_is_nine_sixteenth : prob_B = 9 / 16 := sorry

def possible_values_Y := finset.range 46 \ finset.range 30

theorem possible_values_Y_range : ∀ y, y ∈ possible_values_Y → 30 ≤ y ∧ y ≤ 45 :=
begin
  intro y,
  simp [possible_values_Y],
  intro hy,
  exact ⟨nat.succ_le_iff.mp (finset.mem_range.mp hy), finset.mem_range.mp hy⟩,
end

noncomputable def f (n : ℕ) := (nat.choose 60 n) * (nat.choose (60 - n) (45 - n)) * (nat.choose 15 (45 - n))

theorem most_likely_Y_is_34 : ∃ y ∈ possible_values_Y, y = 34 ∧ ∀ n ∈ possible_values_Y, f y ≥ f n :=
sorry

end prob_A_is_half_prob_B_is_nine_sixteenth_possible_values_Y_range_most_likely_Y_is_34_l105_105510


namespace not_special_n_sq_min_2n_find_special_numbers_l105_105844

section SpecialNumbers

variables (n S : ℕ)
variables (n1 n2 : ℕ)
variables {k : ℕ}

/-- A number S is defined as special if for every positive integer k and for every decomposition of n into a sum of k strictly positive integers
    n = n₁ + n₂ + ⋯ + nₖ with n₁ ≤ n₂ ≤ ⋯ ≤ nₖ, there exist integers 0 ≤ a₁ < a₂ < ⋯ < aₖ such that a₁n₁ + a₂n₂ + ⋯ + aₖnₖ = S. -/
def special (S n : ℕ) : Prop :=
∀ (k : ℕ), k > 0 → ∀ (ns : Fin k → ℕ), (Σ i, ns i = n) →
  (Sorted (ns ∘ (Fin.val : Fin k → ℕ))) →
  ∃ (as : Fin k → ℕ), (∀ i, 0 ≤ as i) ∧ (∀ i j, i < j → as i < as j) ∧
  (Σ i, (as i) * (ns i) = S)

/-- Show that n² - 2n is not special when n ≥ 2.-/
theorem not_special_n_sq_min_2n {n : ℕ} (hn : n ≥ 2) :
  ¬ special (n^2 - 2 * n) n :=
sorry

/-- Find all the special numbers when n ≥ 2. -/
theorem find_special_numbers {n : ℕ} (hn : n ≥ 2) :
  ∀ S, special S n ↔ ∃ k, k ≥ (n^2 - n) ∧ ∃ m, S = k * n :=
sorry

end SpecialNumbers

end not_special_n_sq_min_2n_find_special_numbers_l105_105844


namespace circle_points_iff_perpendicular_l105_105688

noncomputable theory

/-- Given four circles, none of which lies inside another, passing through the point P.
    Two circles touch the line L at P, and the other two touch the line M at P.
    The other points of intersection of the circles are A, B, C, and D. 
    Show that A, B, C, and D lie on a circle if and only if L and M are perpendicular. -/
theorem circle_points_iff_perpendicular
  (P : Point)
  (L M : Line)
  (circles : Finset Circle)
  (A B C D : Point) :
  (∀ c ∈ circles, P ∈ c) ∧
  (∃ c1 c2 ∈ circles, c1 ≠ c2 ∧ touches_at c1 L P ∧ touches_at c2 L P) ∧
  (∃ c3 c4 ∈ circles, c3 ≠ c4 ∧ touches_at c3 M P ∧ touches_at c4 M P) ∧
  (∃ A B C D ∈ circles, other_intersections {A, B, C, D} = {A, B, C, D})
  ↔ is_perpendicular L M :=
sorry  -- Proof needed

end circle_points_iff_perpendicular_l105_105688


namespace angle_equality_trapezoid_l105_105449

theorem angle_equality_trapezoid
  {A B C D P Q : Type}
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  [Inhabited P] [Inhabited Q]
  (is_trapezoid : ∃ (AB CD : Type), true) -- placeholder for the trapezoid condition
  (P_on_AC : ∃ (AC : Type), true) -- placeholder for P being on AC
  (Q_on_BD : ∃ (BD : Type), true) -- placeholder for Q being on BD
  (angle_APD_eq_angle_BQC : ∀ (α β : ℝ), α = β) -- placeholder angle equality
  : ∀ (γ δ : ℝ), γ = δ := -- to prove: angle AQD = angle BPC
begin
  sorry
end

end angle_equality_trapezoid_l105_105449


namespace julia_birth_year_is_1979_l105_105776

-- Definitions based on conditions
def wayne_age_in_2021 : ℕ := 37
def wayne_birth_year : ℕ := 2021 - wayne_age_in_2021
def peter_birth_year : ℕ := wayne_birth_year - 3
def julia_birth_year : ℕ := peter_birth_year - 2

-- Theorem to prove
theorem julia_birth_year_is_1979 : julia_birth_year = 1979 := by
  sorry

end julia_birth_year_is_1979_l105_105776


namespace combined_weight_of_Leo_and_Kendra_l105_105351

theorem combined_weight_of_Leo_and_Kendra :
  ∃ (K : ℝ), (92 + K = 160) ∧ (102 = 1.5 * K) :=
by
  sorry

end combined_weight_of_Leo_and_Kendra_l105_105351


namespace Carly_running_distance_l105_105638

theorem Carly_running_distance :
  let week1 := 2 in
  let week2 := 2 * week1 + 3 in
  let week3 := (9 / 7) * week2 in
  let week4 := week3 - 5 in
  week4 = 4 :=
by
  let week1 := 2
  let week2 := 2 * week1 + 3
  let week3 := (9 / 7) * week2
  let week4 := week3 - 5
  show week4 = 4
  sorry

end Carly_running_distance_l105_105638


namespace find_central_cell_l105_105412

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l105_105412


namespace percentage_problem_l105_105350

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by
  sorry

end percentage_problem_l105_105350


namespace sum_reality_conjugate_necessity_sum_reality_conjugate_not_sufficiency_l105_105837

open Complex

theorem sum_reality_conjugate_necessity (z1 z2 : ℂ) :
  (z1 + z2).im = 0 → (z2 = conj z1) :=
by 
  sorry

theorem sum_reality_conjugate_not_sufficiency (z1 z2 : ℂ) :
  (z1 + z2).im = 0 ∧ ¬ (z2 = conj z1) → True :=
by
  sorry

end sum_reality_conjugate_necessity_sum_reality_conjugate_not_sufficiency_l105_105837


namespace total_number_of_teachers_l105_105488

theorem total_number_of_teachers (departments : ℕ) (teachers_per_department : ℕ) (h₁ : departments = 15) (h₂ : teachers_per_department = 35) : 
  departments * teachers_per_department = 525 := 
by
  have h₃ : 15 * 35 = 525 := by norm_num
  rw [h₁, h₂]
  exact h₃

end total_number_of_teachers_l105_105488


namespace center_radius_sum_l105_105824

theorem center_radius_sum (a b r : ℝ) (h : ∀ x y : ℝ, (x^2 - 8*x - 4*y = -y^2 + 2*y + 13) ↔ (x - 4)^2 + (y - 3)^2 = 38) :
  a = 4 ∧ b = 3 ∧ r = Real.sqrt 38 → a + b + r = 7 + Real.sqrt 38 :=
by
  sorry

end center_radius_sum_l105_105824


namespace line_intersects_ellipse_if_and_only_if_l105_105916

theorem line_intersects_ellipse_if_and_only_if (k : ℝ) (m : ℝ) :
  (∀ x, ∃ y, y = k * x + 1 ∧ (x^2 / 5 + y^2 / m = 1)) ↔ (m ≥ 1 ∧ m ≠ 5) := 
sorry

end line_intersects_ellipse_if_and_only_if_l105_105916


namespace fourth_term_geometric_progression_l105_105910

theorem fourth_term_geometric_progression : 
  let a1 := 2^(1/4)
  let a2 := 2^(1/8)
  let a3 := 2^(1/16)
  ∃ a4 : ℝ, a4 = 2^(-1/16) :=
begin
  let r := a2 / a1,
  let a4 := a3 * r,
  exact ⟨a4, sorry⟩,
sorry

end fourth_term_geometric_progression_l105_105910


namespace cosine_of_rotation_l105_105553

/-- Given points A and A1 are transformed by rotation around the origin,
prove the cosine of the angle of rotation is 24/25 -/
theorem cosine_of_rotation :
  let A := (6 : ℝ, 8 : ℝ)
  let A1 := (8 : ℝ, 6 : ℝ)
  let OA := (A.1, A.2)
  let OA1 := (A1.1, A1.2)
  (OA.1 * OA1.1 + OA.2 * OA1.2) / (Real.sqrt (OA.1 ^ 2 + OA.2 ^ 2) * Real.sqrt (OA1.1 ^ 2 + OA1.2 ^ 2)) = 24 / 25 :=
by {
  let A := (6 : ℝ, 8 : ℝ),
  let A1 := (8 : ℝ, 6 : ℝ),
  let OA := (A.1, A.2),
  let OA1 := (A1.1, A1.2),
  let dot_product := OA.1 * OA1.1 + OA.2 * OA1.2,
  let magnitude_OA := Real.sqrt (OA.1 ^ 2 + OA.2 ^ 2),
  let magnitude_OA1 := Real.sqrt (OA1.1 ^ 2 + OA1.2 ^ 2),
  have h_cosine_formula := (dot_product / (magnitude_OA * magnitude_OA1) = 24 / 25),
  exact h_cosine_formula
  sorry
}

end cosine_of_rotation_l105_105553


namespace factorial_simplification_l105_105499

theorem factorial_simplification :
  (12.factorial : ℚ) / ((10.factorial : ℚ) + 3 * (9.factorial : ℚ)) = 1320 / 13 := by
sorry

end factorial_simplification_l105_105499


namespace find_range_of_f_sin_alpha_sub_beta_l105_105729

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - 2 * cos x ^ 2 + 1

theorem find_range_of_f :
  set.range (λ x, f x) ∩ set.Icc (-π/12) (π/2) = set.Icc (-sqrt 3) 2 :=
sorry

variables {α β : ℝ}
  (hα : 0 < α ∧ α < π/2)
  (hβ : 0 < β ∧ β < π/2)
  (h1 : f ((1/2) * α + π/12) = 10/13)
  (h2 : f ((1/2) * β + π/3) = 6/5)

theorem sin_alpha_sub_beta :
  sin(α - β) = -33/65 :=
sorry

end find_range_of_f_sin_alpha_sub_beta_l105_105729


namespace central_cell_value_l105_105406

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l105_105406


namespace frequency_machineA_frequency_machineB_K_squared_approx_quality_difference_confidence_l105_105115

-- Define the problem conditions
def MachineA_first_class := 150
def MachineA_second_class := 50
def MachineA_total := 200

def MachineB_first_class := 120
def MachineB_second_class := 80
def MachineB_total := 200

def total_products := 400

-- Frequencies
theorem frequency_machineA : (MachineA_first_class : ℚ) / MachineA_total = 3 / 4 := by
  sorry

theorem frequency_machineB : (MachineB_first_class : ℚ) / MachineB_total = 3 / 5 := by
  sorry

-- Calculate K^2
def a := MachineA_first_class
def b := MachineA_second_class
def c := MachineB_first_class
def d := MachineB_second_class
def n := total_products

def K_squared : ℚ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Prove K^2 approximation
theorem K_squared_approx : K_squared ≈ 10.256 := by
  sorry

-- Prove confidence level comparison
theorem quality_difference_confidence : K_squared > 6.635 := by
  sorry

end frequency_machineA_frequency_machineB_K_squared_approx_quality_difference_confidence_l105_105115


namespace projection_vectors_equal_q_l105_105742

def vector1 := (λ (i : Fin 2), ite (i = 0) (-3) (ite (i = 1) 5 0))
def vector2 := (λ (i : Fin 2), ite (i = 0) 4 (ite (i = 1) 1 0))
def direction := (λ (i : Fin 2), ite (i = 0) 7 (ite (i = 1) (-4) 0))
def t := (1 : ℝ) / 33
def q := (λ (i : Fin 2), ite (i = 0) (-92 / 33) (ite (i = 1) (161 / 33) 0))

theorem projection_vectors_equal_q :
  let line_eq := λ t : ℝ, (λ (i : Fin 2), vector1 i + t * (direction i))
  (line_eq t 0) = q 0 ∧ (line_eq t 1) = q 1 :=
by
  sorry

end projection_vectors_equal_q_l105_105742


namespace eq_solutions_l105_105670

theorem eq_solutions (x : ℂ) : (x^4 + 81 = 0) ↔ 
  (x = (3 * real.sqrt 2 / 2) + (3 * real.sqrt 2 / 2) * complex.I ∨ 
   x = -(3 * real.sqrt 2 / 2) - (3 * real.sqrt 2 / 2) * complex.I ∨ 
   x = (3 * real.sqrt 2 / 2) * complex.I - (3 * real.sqrt 2 / 2) ∨ 
   x = -(3 * real.sqrt 2 / 2) * complex.I + (3 * real.sqrt 2 / 2)) := 
sorry

end eq_solutions_l105_105670


namespace number_of_three_digit_multiples_of_7_l105_105331

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l105_105331


namespace sum_of_longest_altitudes_5_12_13_l105_105340

theorem sum_of_longest_altitudes_5_12_13 :
  let triangle := (5, 12, 13) in
  is_right_triangle triangle ->
  let a := triangle.1 in
  let b := triangle.2 in
  let c := triangle.3 in
  a + b = 17 :=
by
  -- define right triangle property
  def is_right_triangle (t: ℕ × ℕ × ℕ) : Prop := (t.1)^2 + (t.2)^2 = (t.3)^2
  
  -- given the conditions
  let triangle := (5, 12, 13)
  have h : is_right_triangle triangle := by
    unfold is_right_triangle
    calc
      5^2 + 12^2 = 25 + 144 := by norm_num
            ... = 169 := by norm_num
            ... = 13^2 := by norm_num

  -- using the defined conditions
  let a := triangle.1
  let b := triangle.2
  let c := triangle.3
  
  -- showing the result
  calc
    a + b = 5 + 12 := by rfl
        ... = 17 := by norm_num

-- Proof skipped
sorry

end sum_of_longest_altitudes_5_12_13_l105_105340


namespace num_five_digit_palindromes_l105_105747

theorem num_five_digit_palindromes : 
  (∃ A B C : ℕ, 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) → 
  (nat.count (λ n, ∃ A B C : ℕ, n = A * 10001 + B * 1010 + C * 100 + B * 10 + A ∧ 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) (1, 99999)) = 900 := 
by 
  sorry

end num_five_digit_palindromes_l105_105747


namespace multiplicative_inverse_modulo_2799_l105_105921

theorem multiplicative_inverse_modulo_2799 :
  ∃ n : ℤ, 0 ≤ n ∧ n < 2799 ∧ (225 * n) % 2799 = 1 :=
by {
  -- conditions are expressed directly in the theorem assumption
  sorry
}

end multiplicative_inverse_modulo_2799_l105_105921


namespace sum_of_zeros_of_transformed_parabola_l105_105522

noncomputable def transformedParabola := -(x - 6)^2 + 7

theorem sum_of_zeros_of_transformed_parabola: 
  let p := 6 + Real.sqrt 7,
      q := 6 - Real.sqrt 7
  in p + q = 12 :=
by
  sorry

end sum_of_zeros_of_transformed_parabola_l105_105522


namespace one_point_one_seven_three_billion_in_scientific_notation_l105_105182

theorem one_point_one_seven_three_billion_in_scientific_notation :
  (1.173 * 10^9 = 1.173 * 1000000000) :=
by
  sorry

end one_point_one_seven_three_billion_in_scientific_notation_l105_105182


namespace central_cell_value_l105_105411

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l105_105411


namespace days_taken_to_complete_work_l105_105572

-- Conditions
def work_rate_B : ℚ := 1 / 33
def work_rate_A : ℚ := 2 * work_rate_B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Proof statement
theorem days_taken_to_complete_work : combined_work_rate ≠ 0 → 1 / combined_work_rate = 11 :=
by
  sorry

end days_taken_to_complete_work_l105_105572


namespace nina_money_l105_105014

theorem nina_money (W : ℝ) (h1 : W > 0) (h2 : 10 * W = 14 * (W - 1)) : 10 * W = 35 := by
  sorry

end nina_money_l105_105014


namespace range_of_expression_l105_105243

variable (a b c : ℝ)

theorem range_of_expression (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 :=
sorry

end range_of_expression_l105_105243


namespace tanker_filling_rate_l105_105608

theorem tanker_filling_rate :
  let barrels_per_minute := 5
  let liters_per_barrel := 159
  let minutes_per_hour := 60
  let liters_per_cubic_meter := 1000
  (barrels_per_minute * liters_per_barrel * minutes_per_hour) / 
  liters_per_cubic_meter = 47.7 :=
by
  sorry

end tanker_filling_rate_l105_105608


namespace central_cell_value_l105_105410

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l105_105410


namespace geometric_sequence_property_l105_105799

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ)
  (H_geo : ∀ n, a (n + 1) = a n * q)
  (H_cond1 : a 5 * a 7 = 2)
  (H_cond2 : a 2 + a 10 = 3) :
  (a 12 / a 4 = 2) ∨ (a 12 / a 4 = 1/2) :=
sorry

end geometric_sequence_property_l105_105799


namespace exists_zero_in_interval_l105_105524

def f (x : ℝ) : ℝ := -|x - 2| + Real.exp x

theorem exists_zero_in_interval : ∃ c ∈ Ioo (0 : ℝ) 1, f c = 0 :=
by
  let f := λ x, -|x - 2| + Real.exp x
  have h0 : f 0 < 0 := by
    calc f 0 = -|0 - 2| + Real.exp 0 : by rfl
        ... = -2 + 1 : by norm_num
        ... = -1 : by norm_num
    
  have h1 : f 1 > 0 := by
    calc f 1 = -|1 - 2| + Real.exp 1 : by rfl
        ... = -1 + Real.exp 1 : by norm_num
        ... = Real.exp 1 - 1 : by ring
        ... > 0 : by linarith [Real.exp_pos 1]
  
  exact IntermediateValueTheorem h0 h1

end exists_zero_in_interval_l105_105524


namespace caffeine_over_l105_105467

section caffeine_problem

-- Definitions of the given conditions
def cups_of_coffee : Nat := 3
def cans_of_soda : Nat := 1
def cups_of_tea : Nat := 2

def caffeine_per_cup_coffee : Nat := 80
def caffeine_per_can_soda : Nat := 40
def caffeine_per_cup_tea : Nat := 50

def caffeine_goal : Nat := 200

-- Calculate the total caffeine consumption
def caffeine_from_coffee : Nat := cups_of_coffee * caffeine_per_cup_coffee
def caffeine_from_soda : Nat := cans_of_soda * caffeine_per_can_soda
def caffeine_from_tea : Nat := cups_of_tea * caffeine_per_cup_tea

def total_caffeine : Nat := caffeine_from_coffee + caffeine_from_soda + caffeine_from_tea

-- Calculate the caffeine amount over the goal
def caffeine_over_goal : Nat := total_caffeine - caffeine_goal

-- Theorem statement
theorem caffeine_over {total_caffeine caffeine_goal : Nat} (h : total_caffeine = 380) (g : caffeine_goal = 200) :
  caffeine_over_goal = 180 := by
  -- The proof goes here.
  sorry

end caffeine_problem

end caffeine_over_l105_105467


namespace quadratic_has_two_distinct_real_roots_l105_105090

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end quadratic_has_two_distinct_real_roots_l105_105090


namespace rate_of_interest_l105_105177

noncomputable def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ (n : ℝ)

theorem rate_of_interest (P : ℝ) (r : ℝ) (A : ℕ → ℝ) :
  A 2 = compound_interest P r 2 →
  A 3 = compound_interest P r 3 →
  A 2 = 2420 →
  A 3 = 2662 →
  r = 10 :=
by
  sorry

end rate_of_interest_l105_105177


namespace central_cell_value_l105_105403

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l105_105403


namespace Carly_running_distance_l105_105639

theorem Carly_running_distance :
  let week1 := 2 in
  let week2 := 2 * week1 + 3 in
  let week3 := (9 / 7) * week2 in
  let week4 := week3 - 5 in
  week4 = 4 :=
by
  let week1 := 2
  let week2 := 2 * week1 + 3
  let week3 := (9 / 7) * week2
  let week4 := week3 - 5
  show week4 = 4
  sorry

end Carly_running_distance_l105_105639


namespace pages_read_per_day_l105_105642

-- Define the total number of pages in the book
def total_pages := 96

-- Define the number of days it took to finish the book
def number_of_days := 12

-- Define pages read per day for Charles
def pages_per_day := total_pages / number_of_days

-- Prove that the number of pages read per day is equal to 8
theorem pages_read_per_day : pages_per_day = 8 :=
by
  sorry

end pages_read_per_day_l105_105642


namespace quadratic_roots_range_l105_105355

theorem quadratic_roots_range (m : ℝ) :
  (∃ x : ℝ, x^2 - (2 * m + 1) * x + m^2 = 0 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 - (2 * m + 1) * y + m^2 = 0)) ↔ m > -1 / 4 :=
by sorry

end quadratic_roots_range_l105_105355


namespace number_of_factors_l105_105312

-- Define the sum S as S = 2^2024 + 2^2023 + 2^2022
def S : ℕ := 2^2024 + 2^2023 + 2^2022

-- Define a set of numbers to be checked
def numbers_to_check : Set ℕ := {6, 7, 8, 9, 10}

-- Problem statement: Prove that exactly 2 numbers in the set divide S
theorem number_of_factors : (Set.filter (λ x => x ∣ S) numbers_to_check).card = 2 := by
  sorry

end number_of_factors_l105_105312


namespace perpendicular_op_gh_l105_105797

noncomputable theory
open_locale classical

variables {A B C D P E F G H O : Point}
variables [InscribedQuadrilateral A B C D O]
variables (P_intersection : Intersection (Diagonals A C B D P))
variables (E_symm : SymmetricLine (B C) (B D) (A D) E)
variables (F_symm : SymmetricLine (A D) (B D) (B C) F)
variables (G_parallel : ParallelLine (E G) (C D) (A B) G)
variables (H_parallel : ParallelLine (F H) (A B) (C D) H)

theorem perpendicular_op_gh :
  Perpendicular OP GH :=
sorry

end perpendicular_op_gh_l105_105797


namespace modeling_clay_blocks_needed_l105_105174

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

noncomputable def volume_block : ℝ := 4 * 3 * 1

noncomputable def blocks_needed (cylinder_volume block_volume : ℝ) : ℕ :=
  let needed := cylinder_volume / block_volume
  needed.ceil

theorem modeling_clay_blocks_needed : 
  let r := 2.5
  let h := 10
  let cylinder_vol := volume_cylinder r h
  let block_vol := volume_block
  blocks_needed cylinder_vol block_vol = 17 := by
  sorry

end modeling_clay_blocks_needed_l105_105174


namespace lattice_hexagon_division_l105_105865

def hexagon_area : ℝ := 18
def grid_point_triangle_area : ℝ := 1 / 2

theorem lattice_hexagon_division :
  (hexagon_area / grid_point_triangle_area) = 36 := by
  sorry

end lattice_hexagon_division_l105_105865


namespace inequality_solution_l105_105671

theorem inequality_solution (y : ℚ) :
  (7/36 : ℚ) + |y - (13/72 : ℚ)| < (11/24 : ℚ) ↔ y ∈ Ioo (-1/12 : ℚ) (4/9 : ℚ) := 
by
  sorry

end inequality_solution_l105_105671


namespace general_formula_sequence_no_arith_geo_seq_l105_105265

-- Definitions
def geometric_seq (a : ℕ → ℤ) := ∃ r : ℤ, ∀ n, a (n+1) = r * a n
def arith_seq (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n, a (n+1) = a n + d

-- Problem Part (1): Find general formula
theorem general_formula_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) (hn_pos : n > 0) (h : ∀ n : ℕ, a (n+1) = 2 * S n + 2) (h2 : ∀ n : ℕ, S n = ∑ i in range n, a i) :
  (n ≥ 1) → a n = 2 * 3^(n-1) :=
sorry

-- Problem Part (2): Investigate d_n sequence within the conditions
theorem no_arith_geo_seq (d : ℕ → ℤ) (m k p : ℕ) (h_arith : m + p = 2 * k) (h_geo : d k^2 = d m * d p) :
  ¬(arith_seq d ∧ geometric_seq d ∧ distinct m k p) :=
sorry

end general_formula_sequence_no_arith_geo_seq_l105_105265


namespace probability_x_plus_y_le_5_l105_105600

theorem probability_x_plus_y_le_5 :
  (∑ in (set_of (λ (x y : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5 ∧ x + y ≤ 5)), 1)
  / (∑ in (set_of (λ (x y : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5)), 1) = 3/5 := by
sorry

end probability_x_plus_y_le_5_l105_105600


namespace unique_alphabets_count_l105_105628

theorem unique_alphabets_count
  (total_alphabets : ℕ)
  (each_written_times : ℕ)
  (total_written : total_alphabets * each_written_times = 10) :
  total_alphabets = 5 := by
  -- The proof would be filled in here.
  sorry

end unique_alphabets_count_l105_105628


namespace T_lies_on_line_l_l105_105737

noncomputable def PQ := (d : ℝ)

structure Point :=
(x : ℝ)
(y : ℝ)

def circle (r : ℝ) (p : Point) : Prop :=
  p.x^2 + p.y^2 = r^2

def P := Point.mk 0 (-r)
def Q (d : ℝ) := Point.mk d (-r)
def A (r α : ℝ) := Point.mk (r * real.cos α) (r * real.sin α)
def B (r α β : ℝ) := Point.mk (r * real.cos (α - β)) (r * real.sin (α - β))

def midpoint (p1 p2 : Point) : Point :=
  Point.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2)

def perp_bisector_slope (p1 p2 : Point) : ℝ :=
  -((p2.x - p1.x) / (p2.y - p1.y))

def perp_bisector (p1 p2 : Point) : ℝ -> ℝ :=
  λ x, ((midpoint p1 p2).y + perp_bisector_slope p1 p2 * (x - (midpoint p1 p2).x))

def T (r d α β : ℝ) : Point :=
  let t1 := perp_bisector (A r α) P in
  let t2 := perp_bisector (B r α β) (Q d) in
  let intersect_x := (t1 1 + t2 1) / 2 in
  Point.mk intersect_x (t1 intersect_x)

def line_l (r d : ℝ) : ℝ -> ℝ :=
  λ x, ((2 * r - real.sqrt (4 * r^2 - d^2)) / d) * x - r

theorem T_lies_on_line_l
  (r d α : ℝ) :
  ∀ (β : ℝ), circle r (A r α) ∧ circle r (B r α β) ∧ (abs ((A r α).x - (B r α β).x) = d) → 
  ∃ t : Point, t = T r d α β ∧ (t.y = line_l r d t.x) :=
by sorry  -- Proof is omitted, just setting up the theorem statement

end T_lies_on_line_l_l105_105737


namespace probability_only_one_selected_l105_105597

theorem probability_only_one_selected :
  let P_H := 2 / 5
      P_W := 3 / 7
      P_H_and_W := 1 / 3
      P_H' := 1 - P_H
      P_W' := 1 - P_W
  in P_H * P_W' + P_H' * P_W = 17 / 35 :=
by
  sorry

end probability_only_one_selected_l105_105597


namespace thomas_total_bill_is_102_l105_105189

noncomputable def item_prices := {
  shirts := 12 * 3,
  socks := 5,
  shorts := 15 * 2,
  swim_trunks := 14
}

noncomputable def total_purchase_price : ℝ :=
  item_prices.shirts + item_prices.socks + item_prices.shorts + item_prices.swim_trunks

noncomputable def shipping_cost (total_purchase_price : ℝ) : ℝ :=
  if total_purchase_price < 50 then 5
  else 0.20 * total_purchase_price

noncomputable def total_bill : ℝ :=
  total_purchase_price + shipping_cost total_purchase_price

theorem thomas_total_bill_is_102 :
  total_bill = 102 :=
by
  -- To be filled in with detailed steps if needed
  sorry

end thomas_total_bill_is_102_l105_105189


namespace positive_value_of_A_l105_105454

theorem positive_value_of_A (A : ℝ) (H₁ : ∀ A B : ℝ, A \# B = A^2 + B^2) (H₂ : A \# 7 = 290) :
  A = Real.sqrt 241 := 
by
  -- sorry enables skipping the proof
  sorry

end positive_value_of_A_l105_105454


namespace max_sundays_in_51_days_l105_105956

theorem max_sundays_in_51_days (days_in_week: ℕ) (total_days: ℕ) 
  (start_on_first: Bool) (first_day_sunday: Prop) 
  (is_sunday: ℕ → Bool) :
  days_in_week = 7 ∧ total_days = 51 ∧ start_on_first = tt ∧ first_day_sunday → 
  (∃ n, ∀ i < total_days, is_sunday i → n ≤ 8) ∧ 
  (∀ j, j ≤ total_days → is_sunday j → j ≤ 8) := by
  sorry

end max_sundays_in_51_days_l105_105956


namespace harmonic_sequence_min_value_l105_105767

/-- 
  Define the sequence {a_n} as a harmonic sequence where 
  (1/a_{n+1} - 1/a_n = d).
  We are given that {1/x_n} is a harmonic sequence and the sum 
  of the first 20 terms of {x_n} is 200.
  We need to prove that the minimum value of (1/x_3 + 1/x_18) is (1/5).
-/
theorem harmonic_sequence_min_value (d : ℝ) (x : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → (1 / x (n + 1) - 1 / x n = d))
  (h2 : ∑ i in finset.range 20, x (i + 1) = 200) :
  ∃ t : ℝ, t = 1/5 ∧ ( ∀ x_3 x_18 : ℝ, (∑ i in finset.range 20, x (i + 1) = 200) → 
  1 / x_3 + 1 / x_18 ≥ t) := 
begin
  sorry
end

end harmonic_sequence_min_value_l105_105767


namespace multiples_of_7_are_128_l105_105324

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l105_105324


namespace true_propositions_count_l105_105619

-- Define the propositions based on the conditions.
def prop1 (p q : Prop) : Prop := ¬(p ∧ q) → ¬p ∧ ¬q
def prop2 : Prop := (¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0))
def prop3 : Prop := (¬(x ≤ 1) → (1 / x < 1)) ∧ (¬(1 / x < 1) → (x ≤ 1))  -- Sufficient but not necessary condition
def prop4 (X : ℝ → ℝ) (C : ℝ) : Prop := 
  (X follows_normal_distribution 3 7) ∧ (P(X > C + 1) = P(X < C - 1)) → (C = 3)

-- Define the main theorem stating the number of true propositions.
theorem true_propositions_count (p q : Prop) (X : ℝ → ℝ) (C : ℝ) : 
  ¬ prop1 p q ∧ prop2 ∧ prop3 ∧ prop4 X C → true_prop_count = 3 :=
by
  sorry

end true_propositions_count_l105_105619


namespace total_people_hike_l105_105529

theorem total_people_hike :
  let cars := 5
  let people_per_car := 4
  let taxis := 8
  let people_per_taxi := 6
  let vans := 3
  let people_per_van := 5
  let buses := 2
  let people_per_bus := 20
  (cars * people_per_car + taxis * people_per_taxi + vans * people_per_van + buses * people_per_bus) = 123 := 
by
  let cars := 5
  let people_per_car := 4
  let taxis := 8
  let people_per_taxi := 6
  let vans := 3
  let people_per_van := 5
  let buses := 2
  let people_per_bus := 20
  have h_cars : cars * people_per_car = 20 := by sorry
  have h_taxis : taxis * people_per_taxi = 48 := by sorry
  have h_vans : vans * people_per_van = 15 := by sorry
  have h_buses : buses * people_per_bus = 40 := by sorry
  calc
    cars * people_per_car + taxis * people_per_taxi + vans * people_per_van + buses * people_per_bus
        = 20 + 48 + 15 + 40 : by sorry
    ... = 123 : by sorry

end total_people_hike_l105_105529


namespace max_checkerboard_squares_covered_l105_105864

-- Define the side lengths for the checkerboard squares and the square card
def checkerboard_square_side : ℝ := 1.25
def card_side : ℝ := 1.75

-- Problem statement: Prove that the card covers at most 9 checkerboard squares
theorem max_checkerboard_squares_covered (checkerboard_square_side card_side : ℝ)
  (h_checkerboard : checkerboard_square_side = 1.25)
  (h_card : card_side = 1.75) :
  ∃ n, n = 9 :=
by 
  use 9
  sorry

end max_checkerboard_squares_covered_l105_105864


namespace average_of_last_four_numbers_l105_105899

theorem average_of_last_four_numbers
  (seven_numbers : list ℝ)
  (h_length : seven_numbers.length = 7)
  (h_avg_seven : (seven_numbers.sum / 7) = 70)
  (h_first_three : (seven_numbers.take 3).sum / 3 = 65)
  : (seven_numbers.drop 3).sum / 4 = 73.75 := by
  sorry

end average_of_last_four_numbers_l105_105899


namespace parallelogram_within_triangle_l105_105841

-- Definition of the main theorem.
theorem parallelogram_within_triangle
  {A B C P O D E F G : Point}
  (hO: is_circumcenter O A B C)
  (hP: is_inside_triangle P A O B)
  (hD: is_projection_of P D B C)
  (hE: is_projection_of P E C A)
  (hF: is_projection_of P F A B)
  (hPar: is_parallelogram D F G E) :
  is_inside_triangle D F G E A B C :=
sorry

end parallelogram_within_triangle_l105_105841


namespace total_bill_l105_105191

-- Define the cost of individual items
def cost_shirt : ℕ := 12
def num_shirts : ℕ := 3
def cost_socks : ℕ := 5
def num_socks : ℕ := 1
def cost_shorts : ℕ := 15
def num_shorts : ℕ := 2
def cost_swim_trunks : ℕ := 14
def num_swim_trunks : ℕ := 1

-- Define the shipping cost based on the total purchase amount
def shipping_cost (total : ℕ) :=
  if total < 50 then 5 else total / 5

-- Calculate the total cost of the items
def total_cost_before_shipping := 
  (num_shirts * cost_shirt) + 
  (num_socks * cost_socks) + 
  (num_shorts * cost_shorts) + 
  (num_swim_trunks * cost_swim_trunks)

-- Statement to prove the final total bill
theorem total_bill :
  let total := total_cost_before_shipping
  let shipping := shipping_cost total
  total + shipping = 102 := by
{
  let total := total_cost_before_shipping in
  let shipping := shipping_cost total in
  rw [total_cost_before_shipping, shipping_cost],
  refl,
}

-- Using 'sorry' to omit proof details and ensure the Lean code builds successfully
sorry

end total_bill_l105_105191


namespace hardly_arrangements_l105_105746

theorem hardly_arrangements : 
  ∀ (letters : List Char), letters = ['h', 'a', 'r', 'd', 'l', 'y'] → 
  count_permutations letters (4, 'd') = 120 :=
by 
  intros letters h
  sorry

end hardly_arrangements_l105_105746


namespace Paige_stickers_l105_105867

theorem Paige_stickers (space_stickers : ℕ) (cat_stickers : ℕ) (friends : ℕ) (space_remainder : ℕ) (cat_remainder : ℕ) :
  space_stickers = 100 →
  cat_stickers = 50 →
  friends = 3 →
  space_stickers % friends = space_remainder →
  cat_stickers % friends = cat_remainder →
  space_remainder + cat_remainder = 3 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4 h5
  rw [←h4, ←h5]
  apply congr_arg2
  { have space_rem : 100 % 3 = 1 := rfl
    exact space_rem }
  { have cat_rem : 50 % 3 = 2 := rfl
    exact cat_rem }
  sorry

end Paige_stickers_l105_105867


namespace drawing_at_least_one_good_l105_105690

-- Defining the problem
variable (products : ℕ) (good : ℕ) (defective : ℕ)

-- Main theorem statement
theorem drawing_at_least_one_good (products = 12) (good = 10) (defective = 2) :
  ∀ (drawn_products : Finset ℕ), drawn_products.card = 3 → ∃ x ∈ drawn_products, x ≤ good :=
by sorry

end drawing_at_least_one_good_l105_105690


namespace social_gathering_married_men_fraction_l105_105197

theorem social_gathering_married_men_fraction {W : ℝ} {MW : ℝ} {MM : ℝ} 
  (hW_pos : 0 < W)
  (hMW_def : MW = W * (3/7))
  (hMM_def : MM = W - MW)
  (h_total_people : 2 * MM + MW = 11) :
  (MM / 11) = 4/11 :=
by {
  sorry
}

end social_gathering_married_men_fraction_l105_105197


namespace mike_total_games_l105_105859

-- Define the number of games Mike went to this year
def games_this_year : ℕ := 15

-- Define the number of games Mike went to last year
def games_last_year : ℕ := 39

-- Prove the total number of games Mike went to
theorem mike_total_games : games_this_year + games_last_year = 54 :=
by
  sorry

end mike_total_games_l105_105859


namespace profit_at_end_of_first_year_l105_105027

theorem profit_at_end_of_first_year :
  let total_amount := 50000
  let part1 := 30000
  let interest_rate1 := 0.10
  let part2 := total_amount - part1
  let interest_rate2 := 0.20
  let time_period := 1
  let interest1 := part1 * interest_rate1 * time_period
  let interest2 := part2 * interest_rate2 * time_period
  let total_profit := interest1 + interest2
  total_profit = 7000 := 
by 
  sorry

end profit_at_end_of_first_year_l105_105027


namespace area_of_triangle_ABC_l105_105020

-- Definitions for points, lines, and segments.
variable (A B C P D E F : Type)
variable [linear_ordered_semiring A] [linear_ordered_semiring B] [linear_ordered_semiring C]
variable [linear_ordered_semiring P] [linear_ordered_semiring D] [linear_ordered_semiring E] [linear_ordered_semiring F]

-- Given conditions
variable (AP BP PD PE CF : ℝ)
variable (h_AP : AP = 6)
variable (h_BP : BP = 9)
variable (h_PD : PD = 6)
variable (h_PE : PE = 3)
variable (h_CF : CF = 20)

-- Main theorem statement
theorem area_of_triangle_ABC (h_AP : AP = 6) (h_BP : BP = 9) (h_PD : PD = 6) (h_PE : PE = 3) (h_CF : CF = 20) :
  ∃ (area : ℝ), area = 108 := by sorry

end area_of_triangle_ABC_l105_105020


namespace factorial_simplification_l105_105497

theorem factorial_simplification :
  (12.factorial : ℚ) / ((10.factorial : ℚ) + 3 * (9.factorial : ℚ)) = 1320 / 13 := by
sorry

end factorial_simplification_l105_105497


namespace fib_fact_last_two_sum_is_five_l105_105973

def fib_fact_last_two_sum (s : List (Fin 100)) : Fin 100 :=
  s.sum

theorem fib_fact_last_two_sum_is_five :
  fib_fact_last_two_sum [1, 1, 2, 6, 20, 20, 0] = 5 :=
by 
  sorry

end fib_fact_last_two_sum_is_five_l105_105973


namespace set_problems_l105_105738

-- Define set A
def A : Set ℝ := {x | 2^x > 8}

-- Define set B
def B : Set ℝ := {x | x^2 - 3x - 4 < 0}

-- Define universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define the complement of A in U
def complement_A : Set ℝ := {x | ¬ (x ∈ A)}

-- Prove the required statement
theorem set_problems :
    A = {x | 3 < x} ∧
    B = {x | -1 < x ∧ x < 4} ∧
    (complement_A ∩ B) = {x | -1 < x ∧ x ≤ 3} :=
by
  sorry

end set_problems_l105_105738


namespace only_one_student_remains_l105_105195

theorem only_one_student_remains (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2002) :
  (∃! k, k = n ∧ n % 1331 = 0) ↔ n = 1331 :=
by
  sorry

end only_one_student_remains_l105_105195


namespace linear_function_increase_l105_105432

theorem linear_function_increase (k : ℝ) (h : (k - 2) > 0) : k > 2 :=
by
  exact h

end linear_function_increase_l105_105432


namespace correct_operation_l105_105568

theorem correct_operation :
  (3 * m^2 + 4 * m^2 ≠ 7 * m^4) ∧
  (4 * m^3 * 5 * m^3 ≠ 20 * m^3) ∧
  ((-2 * m)^3 ≠ -6 * m^3) ∧
  (m^10 / m^5 = m^5) :=
by
  sorry

end correct_operation_l105_105568


namespace problem_statement_l105_105582

-- Definitions from the problem conditions
variable (r : ℝ) (A B C : ℝ)

-- Problem condition that A, B are endpoints of the diameter of the circle
-- Defining the length AB being the diameter -> length AB = 2r
def AB := 2 * r

-- Condition that ABC is inscribed in a circle and AB is the diameter implies the angle ACB = 90°
-- Using Thales' theorem we know that A, B, C satisfy certain geometric properties in a right triangle
-- AC and BC are the other two sides with H right angle at C.

-- Proving the target equation
theorem problem_statement (h : C ≠ A ∧ C ≠ B) : (AC + BC)^2 ≤ 8 * r^2 := 
sorry


end problem_statement_l105_105582


namespace total_people_at_dance_l105_105069

variable (B G T : ℕ)
variable (R : ℚ := 3/4)
variable (G_count : G = 60)
variable (T_percent : ℚ := 0.20)

theorem total_people_at_dance :
  (G = 60) →
  (B = (3 * 60 / 4)) →
  (T = (0.20 * B).toNat) →
  (B + G + T = 114) :=
by
  intros hG hB hT
  rw [hG, hB, hT]
  sorry

end total_people_at_dance_l105_105069


namespace incorrect_statement_C_l105_105761

theorem incorrect_statement_C (a : ℝ) (x : ℝ) (h : a > 1) :
  ¬(y = 1 ∧ x = 0) → y = log a (x^2 + 1) :=
by
  intro h1
  sorry

end incorrect_statement_C_l105_105761


namespace length_of_square_side_is_correct_l105_105881

noncomputable def length_of_square_side : ℚ :=
  let PQ : ℚ := 7
  let QR : ℚ := 24
  let hypotenuse := (PQ^2 + QR^2).sqrt
  (25 * 175) / (24 * 32)

theorem length_of_square_side_is_correct :
  length_of_square_side = 4375 / 768 := 
by 
  sorry

end length_of_square_side_is_correct_l105_105881


namespace three_digit_multiples_of_7_l105_105338

theorem three_digit_multiples_of_7 :
  ∃ n : ℕ, (n = ∑ k in finset.range (143 - 15), ∀ k ∈ finset.range (143 - 15), 100 ≤ 7 * (15 + k) ∧ 7 * (15 + k) ≤ 999) :=
sorry

end three_digit_multiples_of_7_l105_105338


namespace measured_weight_loss_l105_105985

-- Definitions for initial weights and weight loss percentages
def initial_weight_you : ℝ := 95
def initial_weight_friendA : ℝ := 85
def initial_weight_friendB : ℝ := 100

def weight_loss_percentage_you : ℝ := 0.14
def weight_loss_percentage_friendA : ℝ := 0.11
def weight_loss_percentage_friendB : ℝ := 0.18

def clothes_weight_percentage_you : ℝ := 0.02
def clothes_weight_percentage_friendA : ℝ := 0.03
def clothes_weight_percentage_friendB : ℝ := 0.01

-- Calculations of final measured weights
def measure_weight_loss (initial_weight loss_percentage clothes_percentage) : ℝ :=
  let weight_after_loss := initial_weight * (1 - loss_percentage)
  let final_measured_weight := weight_after_loss * (1 + clothes_percentage)
  (initial_weight - final_measured_weight) / initial_weight * 100

-- Theorem statement to prove
theorem measured_weight_loss :
  measure_weight_loss initial_weight_you weight_loss_percentage_you clothes_weight_percentage_you ≈ 12.28 ∧
  measure_weight_loss initial_weight_friendA weight_loss_percentage_friendA clothes_weight_percentage_friendA ≈ 8.33 ∧
  measure_weight_loss initial_weight_friendB weight_loss_percentage_friendB clothes_weight_percentage_friendB ≈ 17.18 :=
by
  sorry

end measured_weight_loss_l105_105985


namespace trapezoid_area_sum_l105_105448

-- Definitions of geometric objects and constants
variables {A B C D E F : Type}
variables (square_side : ℝ) (EF_distance : ℝ)

-- The main theorem
theorem trapezoid_area_sum {ABCD : square_side = 4} {E_on_BC : E ∈ BC} {F_on_DA : F ∈ DA}
  (h_square : is_square ABCD)
  (h_EF : EF = 5) : 
  (sum_min_max_areas_trapezoid_BEDF ABCD E F) = 16 :=
begin
  sorry
end

end trapezoid_area_sum_l105_105448


namespace odd_integer_calculation_l105_105556

theorem odd_integer_calculation : 
  let nth_odd (n : ℕ) := 2 * n - 1 in
  let nth_odd_100 := nth_odd 100 in
  nth_odd_100 = 199 ∧ nth_odd_100 + 10 = 209 :=
by
  sorry

end odd_integer_calculation_l105_105556


namespace Joy_finds_fourth_rod_count_l105_105815

theorem Joy_finds_fourth_rod_count :
  let rod_lengths := (finset.range 41).erase 0,
      placed_rods := {5, 10, 22},
      remaining_rods := rod_lengths \ placed_rods,
      valid_rod_set := remaining_rods.filter (λ x, 8 ≤ x ∧ x < 37) in
  valid_rod_set.card = 27 := 
by
  sorry

end Joy_finds_fourth_rod_count_l105_105815


namespace at_least_one_obtuse_angle_l105_105268

variables {V : Type*} [inner_product_space ℝ V]

-- Define the non-collinearity of points
def non_collinear (O A B C : V) : Prop :=
  ¬∃ k1 k2 k3 : ℝ, k1 • (A - O) + k2 • (B - O) + k3 • (C - O) = (0 : V)

-- Define the condition and main theorem
theorem at_least_one_obtuse_angle 
  {O A B C : V} (h_non_collinear : non_collinear O A B C)
  (h_linear_combination : ∃ λ1 λ2 λ3 : ℝ, λ1 • (A - O) + λ2 • (B - O) + λ3 • (C - O) = (0 : V)) :
  ∃ θ₁ θ₂ θ₃ : ℝ, 
    (θ₁ = real.angle O A B ∨ θ₁ = real.angle O B A ∨ θ₁ = real.angle B O A) ∧
    (θ₂ = real.angle B O C ∨ θ₂ = real.angle O C B ∨ θ₂ = real.angle C B O) ∧
    (θ₃ = real.angle C O A ∨ θ₃ = real.angle O A C ∨ θ₃ = real.angle A C O) ∧
    (θ₁ > real.pi/2 ∨ θ₂ > real.pi/2 ∨ θ₃ > real.pi/2) :=
sorry

end at_least_one_obtuse_angle_l105_105268


namespace trigonometric_identity_l105_105247

theorem trigonometric_identity (x : ℝ) (h : Real.tan (x + Real.pi / 2) = 5) : 
  1 / (Real.sin x * Real.cos x) = -26 / 5 :=
by
  sorry

end trigonometric_identity_l105_105247


namespace central_cell_value_l105_105375

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l105_105375


namespace least_multiple_of_15_greater_than_500_l105_105970

theorem least_multiple_of_15_greater_than_500 : 
  ∃ (n : ℕ), n > 500 ∧ (∃ (k : ℕ), n = 15 * k) ∧ (n = 510) :=
by
  sorry

end least_multiple_of_15_greater_than_500_l105_105970


namespace value_at_12_l105_105629

-- Define the quadratic function and the symmetry point
def p (x : ℝ) : ℝ := d * x^2 + e * x + f

-- Assume conditions
variables (d e f : ℝ)
axiom symmetry_of_quadratic : ∀ x : ℝ, p (7 - x) = p x
axiom point_on_graph : p (-5) = -5

-- Required proof
theorem value_at_12 : p 12 = -5 :=
by
  -- symmetric property p(12) = p(-5) combined with given point
  have h : p 12 = p (-5),
    from symmetry_of_quadratic 12
  rw point_on_graph at h
  exact h

end value_at_12_l105_105629


namespace AM_BM_CM_product_l105_105806

theorem AM_BM_CM_product 
  (AM BM CM : ℝ) 
  (h₁: MA1 = MB1 = MC1 = 3) 
  (h₂: AM + BM + CM = 43) : 
  AM * BM * CM = 441 := 
sorry

end AM_BM_CM_product_l105_105806


namespace three_digit_multiples_of_7_l105_105335

theorem three_digit_multiples_of_7 :
  ∃ n : ℕ, (n = ∑ k in finset.range (143 - 15), ∀ k ∈ finset.range (143 - 15), 100 ≤ 7 * (15 + k) ∧ 7 * (15 + k) ≤ 999) :=
sorry

end three_digit_multiples_of_7_l105_105335


namespace zero_in_interval_l105_105540

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x / 2) - 2 / x

theorem zero_in_interval :
  (Real.log (3 / 2) - 2 < 0) ∧ (Real.log 3 - 2 / 3 > 0) →
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- conditions from the problem statement
  intros h
  -- proving the result
  sorry

end zero_in_interval_l105_105540


namespace find_number_l105_105512

theorem find_number (x : ℝ) :
  let avg1 := (10 + 20 + 60) / 3 in
  let avg2 := (10 + x + 25) / 3 in
  avg2 = avg1 - 5 →
  x = 40 :=
by
  intro h
  simp at h
  linarith

end find_number_l105_105512


namespace square_side_length_l105_105915

theorem square_side_length (x S : ℕ) (h1 : S > 0) (h2 : x = 4) (h3 : 4 * S = 6 * x) : S = 6 := by
  subst h2
  sorry

end square_side_length_l105_105915


namespace find_added_value_l105_105664

theorem find_added_value (x y : ℕ) (hx : x = 3) (h : x + y = 60 * (1 / x)) : y = 17 := by
  rw [hx] at h
  have h1 : 60 * (1 / 3) = 20 := by norm_num
  rw [h1] at h
  linarith

example : find_added_value 3 17 3 rfl := by sorry

end find_added_value_l105_105664


namespace total_earning_correct_l105_105988

-- Definitions based on the conditions
def worked_days_a : ℕ := 6
def worked_days_b : ℕ := 9
def worked_days_c : ℕ := 4

def daily_wage_c : ℕ := 95

noncomputable def daily_wages_ratio : ℕ × ℕ × ℕ := (3, 4, 5)

noncomputable def total_earning : ℕ :=
  let part := daily_wage_c / daily_wages_ratio.2.2 in
  let daily_wage_a := daily_wages_ratio.1 * part in
  let daily_wage_b := daily_wages_ratio.1.2 * part in
  (daily_wage_a * worked_days_a) + (daily_wage_b * worked_days_b) + (daily_wage_c * worked_days_c)

-- The theorem to prove:
theorem total_earning_correct : total_earning = 1406 := by
  sorry

end total_earning_correct_l105_105988


namespace seating_six_people_l105_105781

structure Company (n : Nat) :=
(acquainted : Fin n → Fin n → Prop)
(acquainted_symm : ∀ x y, acquainted x y → acquainted y x)

def can_seat_round_table (n : Nat) (company : Company n) (p : Set (Fin n)) : Prop :=
∃ (perm : Fintype.perm (Fin n)), ∀ i, company.acquainted (perm i) (perm ((i + 1) % n))

theorem seating_six_people (company: Company 6)
  (H : ∀ (s : Finset (Fin 6)), s.card = 5 → can_seat_round_table 5 (Company.mk (λ x y, company.acquainted x y) (λ x y h, company.acquainted_symm x y h)) s) :
  can_seat_round_table 6 company (Finset.univ.to_set) :=
sorry

end seating_six_people_l105_105781


namespace circle_center_coordinates_l105_105901

theorem circle_center_coordinates (x y : ℝ) :
  (x^2 + y^2 - 2*x + 4*y + 3 = 0) → (x = 1 ∧ y = -2) :=
by
  sorry

end circle_center_coordinates_l105_105901


namespace not_odd_not_even_min_value_3_l105_105851

def f (x : ℝ) : ℝ := x^2 + abs (x - 2) - 1

-- Statement 1: Prove that the function is neither odd nor even.
theorem not_odd_not_even : 
  ¬(∀ x, f (-x) = -f x) ∧ ¬(∀ x, f (-x) = f x) :=
sorry

-- Statement 2: Prove that the minimum value of the function is 3.
theorem min_value_3 : ∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≥ 3 :=
sorry

end not_odd_not_even_min_value_3_l105_105851


namespace value_of_x_l105_105362

theorem value_of_x (x : ℝ) (h : x = 80 + 0.2 * 80) : x = 96 :=
sorry

end value_of_x_l105_105362


namespace total_people_at_dance_l105_105071

theorem total_people_at_dance (n_girls n_boys n_teachers total_people : ℕ) 
  (h1 : ratio_girls_boys = 4/3) 
  (h2 : n_girls = 60) 
  (h3 : n_teachers = 0.2 * n_boys) 
  (h4 : n_boys = 3/4 * n_girls) : 
  total_people = n_girls + n_boys + n_teachers :=
  begin
    -- Calculate number of boys
    have h_boys : n_boys = 45, from sorry,
    -- Calculate number of teachers
    have h_teachers : n_teachers = 9, from sorry,
    -- Total number of people calculation
    show total_people = 60 + 45 + 9, from sorry
  end

end total_people_at_dance_l105_105071


namespace find_principal_amount_l105_105046

-- Define constants for the conditions
def rate_per_annum : ℝ := 0.10
def time_period_years : ℝ := 1
def compounding_frequency : ℕ := 2 -- Compounded half-yearly
def si_ci_difference : ℝ := 3

-- Define the formulas for SI and CI
def simple_interest (P : ℝ) : ℝ := P * rate_per_annum * time_period_years
def compound_interest (P : ℝ) : ℝ := P * (1 + rate_per_annum / compounding_frequency)^(compounding_frequency * time_period_years) - P

-- Define the principal amount
def principal_amount : ℝ := 1200

-- State the proof problem as a theorem in Lean 4
theorem find_principal_amount :
  ∃ (P : ℝ), (compound_interest P - simple_interest P = si_ci_difference) ∧ P = principal_amount :=
by
  sorry

end find_principal_amount_l105_105046


namespace janet_lives_l105_105580

variable (initial_lives lost_lives gained_lives final_lives : ℕ)

theorem janet_lives (h1 : initial_lives = 47) (h2 : lost_lives = 23) (h3 : gained_lives = 46) 
  (h4 : final_lives = initial_lives - lost_lives + gained_lives) : 
  final_lives = 70 := by
  rw [h1, h2, h3]
  sorry

end janet_lives_l105_105580


namespace exists_n_le_2500_perfect_square_l105_105229

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_squares_segment (n : ℕ) : ℚ :=
  ((26 * n^3 + 12 * n^2 + n) / 3)

theorem exists_n_le_2500_perfect_square :
  ∃ (n : ℕ), n ≤ 2500 ∧ ∃ (k : ℚ), k^2 = (sum_of_squares n) * (sum_of_squares_segment n) :=
sorry

end exists_n_le_2500_perfect_square_l105_105229


namespace min_distinct_solutions_l105_105506
-- We use a broader import to bring in the entirety of the necessary library

-- Definitions of degrees and constants for the polynomials
def P (z : ℂ) : ℂ := z^3 + 3
def Q (z : ℂ) : ℂ := z^4 + 4
def R (z : ℂ) : ℂ := (z + 1)^7 + 7

-- The main theorem statement
theorem min_distinct_solutions :
  ∃ P Q R : ℂ → ℂ, 
    (deg P = 3) ∧ (coeff_of_z0 P = 3) ∧ 
    (deg Q = 4) ∧ (coeff_of_z0 Q = 4) ∧ 
    (deg R = 7) ∧ (coeff_of_z0 R = 7) ∧ 
    (R (z : ℂ) = P(z) * Q(z))  ∧ (∀ z : ℂ, 
      (P(z) * Q(z) = R(z)) -> N_min = 1) :=
sorry

end min_distinct_solutions_l105_105506


namespace total_balls_without_holes_l105_105013

theorem total_balls_without_holes 
  (soccer_balls : ℕ) (soccer_balls_with_hole : ℕ)
  (basketballs : ℕ) (basketballs_with_hole : ℕ)
  (h1 : soccer_balls = 40)
  (h2 : soccer_balls_with_hole = 30)
  (h3 : basketballs = 15)
  (h4 : basketballs_with_hole = 7) :
  soccer_balls - soccer_balls_with_hole + (basketballs - basketballs_with_hole) = 18 := by
  sorry

end total_balls_without_holes_l105_105013


namespace total_limes_picked_l105_105689

def Fred_limes : ℕ := 36
def Alyssa_limes : ℕ := 32
def Nancy_limes : ℕ := 35
def David_limes : ℕ := 42
def Eileen_limes : ℕ := 50

theorem total_limes_picked :
  Fred_limes + Alyssa_limes + Nancy_limes + David_limes + Eileen_limes = 195 :=
by
  sorry

end total_limes_picked_l105_105689


namespace num_of_cabinets_indeterminate_l105_105183

def shelves_figure_capacity (shelves : ℕ) (figures_per_shelf : ℕ) : ℕ := shelves * figures_per_shelf

theorem num_of_cabinets_indeterminate (figures_per_shelf : ℕ) (total_shelves : ℕ) (total_figures : ℕ) :
  figures_per_shelf = 11 → total_shelves = 4 → total_figures = 44 →
  (∃ (cabinets : ℕ), true) := by
  intros h1 h2 h3
  use 0
  apply true.intro

#check shelves_figure_capacity -- To ensure that the Lean code can be built successfully

end num_of_cabinets_indeterminate_l105_105183


namespace tori_needs_more_correct_answers_l105_105786

theorem tori_needs_more_correct_answers :
  let total_questions := 80
  let arithmetic_questions := 20
  let algebra_questions := 25
  let geometry_questions := 35
  let arithmetic_correct := 0.60 * arithmetic_questions
  let algebra_correct := Float.round (0.50 * algebra_questions)
  let geometry_correct := Float.round (0.70 * geometry_questions)
  let correct_answers := arithmetic_correct + algebra_correct + geometry_correct
  let passing_percentage := 0.65
  let required_correct := passing_percentage * total_questions
-- assertion
  required_correct - correct_answers = 2 := 
by 
  sorry

end tori_needs_more_correct_answers_l105_105786


namespace proof_problem_l105_105023

noncomputable def f : ℤ[X] :=
  polynomial.monic_iff_nat_degree_eq_degree.mpr 
    ⟨(x^2 - 2) * (x^2 - 7) * (x^2 - 14), by {
      rw [polynomial.degree_mul, polynomial.degree_mul, 
          polynomial.degree_sub_eq_of_not_is_root, 
          polynomial.degree_sub_eq_of_not_is_root,
          polynomial.degree_sub_eq_of_not_is_root],
      exact dec_trivial
    }⟩

theorem proof_problem :
  (∀ m : ℤ, f.eval m ≠ 0) ∧ 
  (∀ n : ℕ, n % 2 = 1 → ∃ k : ℕ, k > 0 ∧ n ∣ f.eval k) := by
  sorry

end proof_problem_l105_105023


namespace number_of_saturday_sales_l105_105744

def caricatures_sold_on_saturday (total_earnings weekend_earnings price_per_drawing sunday_sales : ℕ) : ℕ :=
  (total_earnings - (sunday_sales * price_per_drawing)) / price_per_drawing

theorem number_of_saturday_sales : caricatures_sold_on_saturday 800 800 20 16 = 24 := 
by 
  sorry

end number_of_saturday_sales_l105_105744


namespace apple_tree_baskets_l105_105186

theorem apple_tree_baskets
  (apples_per_tree : ℕ)
  (baskets_per_tree : ℕ)
  (apples_per_basket : ℕ)
  : (from_trees : ℕ) (total_apples : ℕ)
  (h1 : apples_per_basket = 15)
  (h2 : from_trees = 10)
  (h3 : total_apples = 3000)
  (h4 : apples_per_tree = total_apples / from_trees)
  (h5 : baskets_per_tree = apples_per_tree / apples_per_basket) :
  baskets_per_tree = 20 := by
  sorry

end apple_tree_baskets_l105_105186


namespace coin_stack_height_l105_105421

theorem coin_stack_height :
  ∃ x y z w : ℕ, 1.25 * x + 1.65 * y + 1.05 * z + 1.85 * w = 16.5 ∧ x + y + z + w = 10 :=
sorry

end coin_stack_height_l105_105421


namespace tank_capacity_is_400_l105_105193

-- Definitions and problem setup
def capacity_of_tank := Real

-- Conditions
def outlet_empty_rate (C : capacity_of_tank) := C / 5
def inlet_fill_rate := 4 * 60 -- in litres per hour
def outlet_with_inlet_rate (C : capacity_of_tank) := C / 8

-- Problem statement
theorem tank_capacity_is_400 :
  ∀ (C : capacity_of_tank),
    outlet_empty_rate C - inlet_fill_rate = outlet_with_inlet_rate C →
    C = 400 :=
begin
  sorry
end

end tank_capacity_is_400_l105_105193


namespace mary_carrots_correct_l105_105887

def sandy_carrots := 8
def total_carrots := 14

def mary_carrots := total_carrots - sandy_carrots

theorem mary_carrots_correct : mary_carrots = 6 := by
  unfold mary_carrots
  unfold total_carrots
  unfold sandy_carrots
  sorry

end mary_carrots_correct_l105_105887


namespace triangle_pqr_pq2_plus_pr2_difference_l105_105804

theorem triangle_pqr_pq2_plus_pr2_difference 
  (QR : ℝ) (PS : ℝ) (S_midpoint : ∀ (P Q R S : Point) (QR_midpoint : Point),
  QR_midpoint = midpoint Q R → QR = dist Q R → PS = dist P QR_midpoint →
  (maximal PQ^2 + PR^2 : ℝ) - (minimal PQ^2 + PR^2 : ℝ) = 100 :=
sorry

end triangle_pqr_pq2_plus_pr2_difference_l105_105804


namespace multiples_of_7_are_128_l105_105323

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l105_105323


namespace value_of_n_l105_105161

variable (n : ℕ)
variable (S : Fin 8 → Fin 4 → ℕ)

-- Condition 1: Each person solved exactly 11 of the 12 problems
def each_friend_solved_11_problems (h : ℕ) : Prop :=
  h = 11

-- Condition 2: The sum of the 32 entries in the table (S) equals 256
def sum_of_table_entries (S : Fin 8 → Fin 4 → ℕ) : Prop :=
  ∑ i, ∑ j, S i j = 256

-- Define the main theorem to prove the value of n
theorem value_of_n (H_sum : sum_of_table_entries S) : n = 32 :=
  sorry

end value_of_n_l105_105161


namespace Aiyanna_has_more_cookies_l105_105618

theorem Aiyanna_has_more_cookies (cookies_Alyssa : ℕ) (cookies_Aiyanna : ℕ) (h1 : cookies_Alyssa = 129) (h2 : cookies_Aiyanna = cookies_Alyssa + 11) : cookies_Aiyanna = 140 := by
  sorry

end Aiyanna_has_more_cookies_l105_105618


namespace fabric_overlap_area_l105_105953

theorem fabric_overlap_area
  (w1 w2 : ℝ) (θ : ℝ)
  (h_w1 : w1 = 3)
  (h_w2 : w2 = 5)
  (h_θ : θ = π / 4) :
  (area : ℝ) ∃ (a : ℝ), a = 9 * Real.sqrt 2 ∧ area = a :=
by
  sorry

end fabric_overlap_area_l105_105953


namespace mutually_exclusive_necessary_not_sufficient_complementary_l105_105477

variables {Ω : Type} {A1 A2 : Set Ω}

/-- Definition of mutually exclusive events -/
def mutually_exclusive (A1 A2 : Set Ω) : Prop :=
  A1 ∩ A2 = ∅

/-- Definition of complementary events -/
def complementary (A1 A2 : Set Ω) : Prop :=
  A1 ∪ A2 = Set.univ ∧ mutually_exclusive A1 A2

/-- The proposition that mutually exclusive events are necessary but not sufficient for being complementary -/
theorem mutually_exclusive_necessary_not_sufficient_complementary :
  (mutually_exclusive A1 A2 → complementary A1 A2) = false 
  ∧ (complementary A1 A2 → mutually_exclusive A1 A2) = true :=
sorry

end mutually_exclusive_necessary_not_sufficient_complementary_l105_105477


namespace good_jars_l105_105005

def original_cartons : Nat := 50
def jars_per_carton : Nat := 20
def less_cartons_received : Nat := 20
def damaged_jars_per_5_cartons : Nat := 3
def total_damaged_cartons : Nat := 1
def total_good_jars : Nat := 565

theorem good_jars (original_cartons jars_per_carton less_cartons_received damaged_jars_per_5_cartons total_damaged_cartons : Nat) :
  (original_cartons - less_cartons_received) * jars_per_carton 
  - (5 * damaged_jars_per_5_cartons + total_damaged_cartons * jars_per_carton) = total_good_jars := 
by 
  sorry

end good_jars_l105_105005


namespace ana_vs_bob_l105_105196

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25

def ana_total (original : ℝ) (tax_rate : ℝ) (discount : ℝ) : ℝ :=
  (original * (1 + tax_rate)) * (1 - discount)

def bob_total (original : ℝ) (tax_rate : ℝ) (discount : ℝ) : ℝ :=
  (original * (1 - discount)) * (1 + tax_rate)

theorem ana_vs_bob (original : ℝ) (tax_rate : ℝ) (discount : ℝ) :
  ana_total original tax_rate discount - bob_total original tax_rate discount = 0 :=
by
  rw [ana_total, bob_total]
  calc
    (original * (1 + tax_rate)) * (1 - discount)
      = (original * (1 - discount)) * (1 + tax_rate) : by ring
    ... = bob_total original tax_rate discount : rfl
  sorry

end ana_vs_bob_l105_105196


namespace John_rejection_rate_l105_105818

variable (P : ℝ) -- Total number of products
variable (J : ℝ) -- The percentage (as a decimal) of products rejected by John

-- Given conditions
def Jane_products_inspected := 0.625 * P
def Jane_reject_rate := 0.009
def Total_reject_rate := 0.0075

-- Jane's rejected products
def Jane_rejected := Jane_reject_rate * Jane_products_inspected

-- John's inspected products
def John_products_inspected := 0.375 * P

-- John's rejected products
def John_rejected := J * John_products_inspected

-- Total rejected products
def Total_rejected := 0.0075 * P

-- The equation derived from conditions
def equation := Jane_rejected + John_rejected = Total_rejected

-- Proposition to prove: solve for J
theorem John_rejection_rate
  (h1 : Jane_rejected = 0.005625 * P)
  (h2 : Total_rejected = 0.0075 * P) :
  J = 0.005 :=
by
  -- here proof steps would go
  sorry

end John_rejection_rate_l105_105818


namespace math_problem_l105_105287

-- Define the given points O and A
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (3, 0)

-- Define the moving point P
variables {x y : ℝ}
def P : ℝ × ℝ := (x, y)

-- Condition: |PA / PO| = 2
def PA := Real.sqrt ((x - 3)^2 + y^2)
def PO := Real.sqrt (x^2 + y^2)
def condition_PA_PO := abs (PA / PO) = 2

-- Equation of the curve Γ
def Γ := (x + 1)^2 + y^2 = 4

-- Points B and C on Γ such that |BC| = 2√3, and N is the midpoint of BC
variables {B C : ℝ × ℝ}
def BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
def midpoint_N := (B.1 + C.1) / 2 = x ∧ (B.2 + C.2) / 2 = y
def condition_BC := BC = 2 * Real.sqrt 3

-- The line l passing through A tangent to Γ at M, and maximum area of triangle AMN
def area_AMN :=
  let AM := Real.sqrt ((3 - x)^2 + y^2) in
  let DN := 1 in (* Distance from (-1, 0) to N *)
  let height := AM + DN in
  1 / 2 * AM * height

-- The final theorem to prove the problem statement
theorem math_problem (condition_PA_PO : abs (PA / PO) = 2)
  (condition_BC : BC = 2 * Real.sqrt 3)
  (midpoint_N : (B.1 + C.1) / 2 = x ∧ (B.2 + C.2) / 2 = y) :
  ((x + 1)^2 + y^2 = 4) ∧
  (area_AMN = 3 * Real.sqrt 3) := sorry

end math_problem_l105_105287


namespace isosceles_triangle_angle_ABC_36_l105_105936

noncomputable theory
open Triangle

variables {A B C D F : Point} {a b : Real}

/-- The triangle ABC is isosceles with AB = BC,
    D is the foot of the altitude from B to AC,
    F is the foot of the internal bisector from A, and
    AF = 2BD. Prove the angle ABC is 36 degrees. -/
theorem isosceles_triangle_angle_ABC_36 (h1 : is_isosceles ∠ABC B A B)
(h2 : is_altitude B D AC)
(h3 : is_internal_bisector A F BC)
(h4 : AF = 2 * BD) :
  ∠ABC = 36 := sorry

end isosceles_triangle_angle_ABC_36_l105_105936


namespace central_cell_value_l105_105396

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l105_105396


namespace quadratic_has_two_distinct_real_roots_l105_105101

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  let a := 1
      c := -8
      b := m
      Δ := b^2 - 4 * a * c 
  in (Δ > 0) :=
by
  let a := 1
  let c := -8
  let b := m
  let Δ := b^2 - 4 * a * c
  show Δ > 0
  sorry

end quadratic_has_two_distinct_real_roots_l105_105101


namespace largest_multiple_of_12_negation_l105_105957

theorem largest_multiple_of_12_negation (k : ℤ) (h1 : 12 * k = 144) (h2 : -12 * k > -150) : 12 * k = 144 :=
by
  unfold has_mul.mul
  unfold has_neg.neg
  sorry

end largest_multiple_of_12_negation_l105_105957


namespace three_digit_multiples_of_7_l105_105326

theorem three_digit_multiples_of_7 : 
  ∃! n : ℕ, (n = 128) ∧ (∀ k, (100 ≤ 7 * k ∧ 7 * k ≤ 999) ↔ (15 ≤ k ∧ k ≤ 142)) :=
begin
  sorry
end

end three_digit_multiples_of_7_l105_105326


namespace find_x_l105_105765

def op (a b : ℕ) : ℕ := a * b - b + b ^ 2

theorem find_x (x : ℕ) : (∃ x : ℕ, op x 8 = 80) :=
  sorry

end find_x_l105_105765


namespace arrangements_of_45520_l105_105426

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (k : Nat) : Nat :=
  factorial n / factorial k

theorem arrangements_of_45520 : 
  let n0_pos := 4
  let remaining_digits := 4 * arrangements 4 2
  n0_pos * remaining_digits = 48 :=
by
  -- Definitions and lemmas can be introduced here
  sorry

end arrangements_of_45520_l105_105426


namespace quadratic_roots_distinct_l105_105077

-- Define the quadratic equation condition
def quadratic_eq : (ℝ → ℝ) :=
  λ x m => x^2 + m * x - 8

-- State the problem
theorem quadratic_roots_distinct (m : ℝ) : 
  let Δ := m^2 + 32 in Δ > 0 :=
  by
   -- We need this theorem to state that the equation always has distinct real roots
  let Δ := m^2 + 32
  sorry

end quadratic_roots_distinct_l105_105077


namespace compute_v_l105_105505

variable (a b c : ℝ)

theorem compute_v (H1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -8)
                  (H2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 12)
                  (H3 : a * b * c = 1) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = -8.5 :=
sorry

end compute_v_l105_105505


namespace locus_of_pole_l105_105149

noncomputable def parabola_p1 := { p : ℝ // p > 0 } -- Define the parabola P1 with p > 0
noncomputable def parabola_p2 := { p : ℝ // p > 0 }

theorem locus_of_pole (p : parabola_p1) :
  (∀ x y : ℝ, (∃ λ : ℝ, 2 * λ * x + y - 2 * p * λ^2 = 0 ∧
    (y + 2 * p * λ^2 = 0 ∧ λ * x - y = 0)) → x^2 + 2 * p * y = 0) :=
begin
  sorry
end

end locus_of_pole_l105_105149


namespace k_domain_and_range_correct_l105_105593

noncomputable def h : ℝ → ℝ := sorry -- Assume h is a placeholder function

-- h is defined on the interval [1, 4], and its range is [2, 5]
def h_domain : Set ℝ := Icc (1 : ℝ) (4 : ℝ)
def h_range : Set ℝ := Icc (2 : ℝ) (5 : ℝ)

def k (x : ℝ) : ℝ := 3 - h (x - 1)

-- Define the domain and range of k
def k_domain : Set ℝ := Icc (2 : ℝ) (5 : ℝ)
def k_range : Set ℝ := Icc (-2 : ℝ) (1 : ℝ)

theorem k_domain_and_range_correct :
  (k_domain = Icc (2 : ℝ) (5 : ℝ)) ∧ (k_range = Icc (-2 : ℝ) (1 : ℝ)) :=
by
  sorry

end k_domain_and_range_correct_l105_105593


namespace vector_ratio_l105_105283

variables {a b : ℝ^2}

-- Conditions and Given
def angle_between_vectors := (real.angle a b) = π / 3
def inequality_holds := ∀ (λ : ℝ), ∥a - λ • b∥ ≥ ∥a - b∥

-- The theorem that needs to be proved
theorem vector_ratio (h1 : angle_between_vectors) (h2 : inequality_holds) : ∥a∥ = 2 * ∥b∥ :=
sorry

end vector_ratio_l105_105283


namespace smallest_two_digit_number_unused_card_l105_105567

theorem smallest_two_digit_number_unused_card :
  ∀ (a b c : ℕ), a = 3 → b = 7 → c = 2 → 
  (∀ (x y : ℕ), {x, y} = {a, c} → 10 * x + y = 23 → 
  ∃ z : ℕ, z ∈ {a, b, c} ∧ z ∉ {x, y})
:= 
by {
  intros a b c ha hb hc x y hset hvalue,
  use b,
  split,
  {
    rw [ha, hb, hc],
    simp,
  },
  {
    intro hxy,
    cases hxy,
    { rw [←hset] at hxy, 
      by_contradiction,
      apply hvalue,
      linarith,
    },
    { rw [←hset] at hxy, 
      by_contradiction,
      apply hvalue,
      linarith,
    },
  },
}

end smallest_two_digit_number_unused_card_l105_105567


namespace proof_problem_l105_105297

noncomputable def p : ℝ := -5 / 3
noncomputable def q : ℝ := -1

def A (p : ℝ) : Set ℝ := {x | 2 * x^2 + 3 * p * x + 2 = 0}
def B (q : ℝ) : Set ℝ := {x | 2 * x^2 + x + q = 0}

theorem proof_problem (h : (A p ∩ B q) = {1 / 2}) :
    p = -5 / 3 ∧ q = -1 ∧ (A p ∪ B q) = {-1, 1 / 2, 2} := by
  sorry

end proof_problem_l105_105297


namespace reduced_price_proof_l105_105139

noncomputable def reduced_price (P: ℝ) := 0.88 * P

theorem reduced_price_proof :
  ∃ R P : ℝ, R = reduced_price P ∧ 1200 / R = 1200 / P + 6 ∧ R = 24 :=
by
  sorry

end reduced_price_proof_l105_105139


namespace number_of_distinct_pairs_l105_105058

theorem number_of_distinct_pairs : 
  {n // n = 3 ∧ ∃ (x y : ℤ), 0 < x ∧ x < y ∧ √2009 = √x + √y} :=
begin
  sorry
end

end number_of_distinct_pairs_l105_105058


namespace find_g50_l105_105912

variables (p q r s : ℝ)
variables [non_zero : p ≠ 0] [non_zero : q ≠ 0] [non_zero : r ≠ 0] [non_zero : s ≠ 0]

-- Definition of the function g(x)
def g (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

-- Given conditions
axiom g_23 : g 23 = 23
axiom g_101 : g 101 = 101
axiom g_inv : ∀ x ≠ -s / r, g (g x) = x

-- Proof problem statement
theorem find_g50 : g 50 = -61 := by
  sorry

end find_g50_l105_105912


namespace part1_position_part1_time_part2_position_part3_coincidence_l105_105018

-- Define the initial properties of the point Q and its movement pattern
def speed := 4
def move_pattern (n : ℕ) : ℤ :=
  if n % 2 = 1 then n else -n

def position_after_moves (n : ℕ) : ℤ :=
  (Finset.range n).sum move_pattern

-- 1. Proving position and time after the 8th move
def position_after_8_moves : Prop := position_after_moves 8 = -4

def total_time_after_8_moves : Prop :=
  let distance := (Finset.range 8).sum (λ n, (n + 1)) in
  distance / speed = 9

-- 2. Proving position after 7 seconds
def position_after_7_seconds : Prop :=
  let distance := 7 * speed in
  let moves := nat.find (λ n, (Finset.range n).sum (λ n, n + 1) ≥ distance) in
  position_after_moves moves = 4

-- 3. Proving coincidence with point A after 294 seconds
def coincidence_with_point_A : Prop :=
  let A := -24
  let distance := 294 * speed in
  let moves := nat.find (λ n, (Finset.range n).sum (λ n, n + 1) ≥ distance) in
  position_after_moves moves = A

theorem part1_position : position_after_8_moves := by sorry
theorem part1_time : total_time_after_8_moves := by sorry
theorem part2_position : position_after_7_seconds := by sorry
theorem part3_coincidence : coincidence_with_point_A := by sorry

end part1_position_part1_time_part2_position_part3_coincidence_l105_105018


namespace beetle_paths_independent_l105_105181

/-- The number of paths from (1, 1) to (1000, n) on a 1000 x n board -/
theorem beetle_paths_independent (n : ℕ) (h_odd : odd n) (h_gt : n > 2020) : 
  ∃ paths : ℕ, (∀ (n' : ℕ), n' ≠ n → odd n' → n' > 2020 → paths = beetle_paths (1000, n')) := 
sorry

end beetle_paths_independent_l105_105181


namespace locus_of_midpoints_of_segments_l105_105301

-- Define lines a and b
def skew_lines (a b : ℝ → ℝ × ℝ × ℝ) : Prop := 
  ∃ l1 l2 : ℝ → ℝ × ℝ × ℝ, ∀ (t1 t2 : ℝ),
  l1 t1 ≠ l2 t2 ∧ ∀ (u : ℝ), l1 t1 + u * (l2 t2 - l1 t1) ≠ l2 t2

-- Define point on a line
def point_on_line (line : ℝ → ℝ × ℝ × ℝ) (x y z : ℝ) : Prop :=
  ∃ t : ℝ, line t = (x, y, z)

-- Define the midpoint
def midpoint (A B M : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x1 y1 z1 x2 y2 z2 : ℝ), 
  A = (x1, y1, z1) ∧ B = (x2, y2, z2) ∧ 
  M = ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)

-- Define the locus as a plane
def locus (a b : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∀ (A B M : ℝ × ℝ × ℝ), 
  point_on_line a A.1 A.2 A.3 → point_on_line b B.1 B.2 B.3 → 
  midpoint A B M → 
  exists k : ℝ, (M.1, M.2, M.3 + k) = B ∨ (M.1, M.2 + k, M.3) = B ∨ (M.1 + k, M.2, M.3) = B

theorem locus_of_midpoints_of_segments (a b : ℝ → ℝ × ℝ × ℝ) :
  skew_lines a b → locus a b := 
by 
  intro h_skew
  sorry

end locus_of_midpoints_of_segments_l105_105301


namespace minimum_socks_needed_l105_105342

theorem minimum_socks_needed 
  (blue_socks : ℕ) (red_socks : ℕ) (total_socks : ℕ) :
  blue_socks = 6 → red_socks = 6 → total_socks = 13 →
  ∃ n, n ≤ total_socks ∧ (∀ picked, picked ≥ n → ∃ x y, x ≠ y ∧ x ∈ picked ∧ y ∈ picked ∧ (x = "Blue" ∨ x = "Red") ∧ (y = "Blue" ∨ y = "Red")) :=
by
  intros h1 h2 h3
  use 7
  split
  { sorry }, -- Here we check 7 ≤ 13, which is true.
  { sorry } -- Here we verify that any selection of 7 socks will include both Blue and Red.

end minimum_socks_needed_l105_105342


namespace inclination_angle_of_line_l105_105258

theorem inclination_angle_of_line :
  ∀ (A B : ℝ × ℝ), A = (-2, 0) → B = (-5, 3) → 
  let θ := real.atan (real.to_real (snd A - snd B) / (fst A - fst B)) 
  in θ = real.pi * (3 / 4) :=
begin
  sorry
end

end inclination_angle_of_line_l105_105258


namespace line_segment_endpoint_l105_105165

theorem line_segment_endpoint (x : ℝ) (h1 : (x - 3)^2 + 36 = 289) (h2 : x < 0) : x = 3 - Real.sqrt 253 :=
sorry

end line_segment_endpoint_l105_105165


namespace system_solution_unique_l105_105669

theorem system_solution_unique : 
  ∀ (x y z : ℝ),
  (4 * x^2) / (1 + 4 * x^2) = y ∧
  (4 * y^2) / (1 + 4 * y^2) = z ∧
  (4 * z^2) / (1 + 4 * z^2) = x 
  → (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solution_unique_l105_105669


namespace not_perfect_square_l105_105926

open Nat

theorem not_perfect_square (n : ℕ) (h1 : n.digits 10 = list.repeat 1 123) (h0 : all_zeroes n) : ¬ is_square n := by
  sorry

-- Helper definition to prove there are 123 ones in the representation
def all_zeroes (n : ℕ) : Prop :=
  ∀ k, k ∈ (list.drop 123 (n.digits 10)) → k = 0

-- Helper definition to identify if a number is a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ m, m * m = n

end not_perfect_square_l105_105926


namespace general_term_a_n_sum_first_n_terms_l105_105428

-- Definitions for the arithmetic sequence \{a_n\}
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Conditions provided in the problem for first term calculations
def a_2 : ℕ := 5
def a_2_eq : a_n 2 = 5 := by sorry
def a_1_a_3_a_4_sum : ℕ := 19
def a_1_a_3_a_4_eq : a_n 1 + a_n 3 + a_n 4 = a_1_a_3_a_4_sum := by sorry

-- General term formula for the arithmetic sequence {a_n}
theorem general_term_a_n (n : ℕ) : a_n n = 2 * n + 1 := by sorry

-- Definitions and conditions associated with the sequence \{b_n\} and \{c_n\}
variable (λ : ℝ) -- Lambda, the constant in the problem
def S_n (n : ℕ) : ℝ  := λ - n * (1 / 2) ^ (n - 1)
def a_n_real (n : ℕ) : ℝ := 2 * n + 1
def S_n_condition (n : ℕ) : S_n n + (a_n_real n - 1) / 2 ^ n = λ := by sorry

-- Sequence {c_n} and the sum T_n of its first n terms
def b_n (n : ℕ) : ℝ := S_n n - S_n (n - 1)
def c_n (n : ℕ) : ℝ := b_n (n + 1)
def T_n (n : ℕ) : ℝ := (n - 1).sum (λ k => k * (1 / 2) ^ (k + 1))

-- The final theorem to be proven
theorem sum_first_n_terms (n : ℕ) : T_n n = 1 - (n + 1) * (1 / 2) ^ n := by sorry

end general_term_a_n_sum_first_n_terms_l105_105428


namespace lcm_one_to_twelve_l105_105562

theorem lcm_one_to_twelve : 
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 
  (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 (Nat.lcm 11 12)))))))))) = 27720 := 
by sorry

end lcm_one_to_twelve_l105_105562


namespace marble_difference_l105_105551

theorem marble_difference (A B : ℕ) (rA yA rB yB total_yellow : ℕ) 
  (hA_ratio : 7 * yA = 3 * rA) 
  (hB_ratio : 4 * yB = rB) 
  (h_total_yellow : yA + yB = total_yellow)
  (h_equal_marbles : rA + yA = rB + yB)
  (h_total_yellow_120 : total_yellow = 120) :
  rA - rB = -24 :=
by
  sorry

end marble_difference_l105_105551


namespace no_rearrangement_of_polynomial_F_l105_105296

def polynomial_F (x : ℕ) : ℕ := 
  ∑ i in (Finset.range 100), (i+1) * x^i

def polynomial_rearranged (F : ℕ → ℕ) :=
  ∃ G : ℕ → ℕ, (∀ (k : ℕ), k ≥ 2 → (F k - G k) % 100 ≠ 0)

theorem no_rearrangement_of_polynomial_F (F G : ℕ → ℕ) :
  polynomial_F = F → ¬ polynomial_rearranged F :=
by 
  intros hF hG
  sorry

end no_rearrangement_of_polynomial_F_l105_105296


namespace problem_solution_exists_l105_105276

theorem problem_solution_exists (p q : ℕ) (h_sum : (Finset.sum (Finset.range 18) (λ k, Real.sin (10 * (k + 1) : ℝ))) = Real.tan ((p : ℝ) / (q : ℝ))) (angles_in_degrees : True) (relatively_prime : Nat.coprime p q) (fraction_less_than_90 : (p : ℝ) / (q : ℝ) < 90) : p + q = 86 :=
sorry

end problem_solution_exists_l105_105276


namespace segment_ratios_equal_l105_105450

-- Definitions for circles and tangents
variables {O1 O2 : Type*} {P Q A B C D : Type*}
  [IsCircle O1] [IsCircle O2]
  (h_int : intersect O1 O2 P Q)
  (h_tan1 : tangent O1 A B)
  (h_tan2 : tangent O2 B A)
  {Γ : Type*} [IsCircle Γ]
  (h_Gamma : passes_through Γ A B)
  (h_ΓO1 : intersect Γ O1 D)
  (h_ΓO2 : intersect Γ O2 C)

-- The proof statement
theorem segment_ratios_equal (h_cond : h_int ∧ h_tan1 ∧ h_tan2 ∧ h_Gamma ∧ h_ΓO1 ∧ h_ΓO2) :
  (CP / CQ) = (DP / DQ) :=
begin
  sorry
end

end segment_ratios_equal_l105_105450


namespace total_bill_l105_105190

-- Define the cost of individual items
def cost_shirt : ℕ := 12
def num_shirts : ℕ := 3
def cost_socks : ℕ := 5
def num_socks : ℕ := 1
def cost_shorts : ℕ := 15
def num_shorts : ℕ := 2
def cost_swim_trunks : ℕ := 14
def num_swim_trunks : ℕ := 1

-- Define the shipping cost based on the total purchase amount
def shipping_cost (total : ℕ) :=
  if total < 50 then 5 else total / 5

-- Calculate the total cost of the items
def total_cost_before_shipping := 
  (num_shirts * cost_shirt) + 
  (num_socks * cost_socks) + 
  (num_shorts * cost_shorts) + 
  (num_swim_trunks * cost_swim_trunks)

-- Statement to prove the final total bill
theorem total_bill :
  let total := total_cost_before_shipping
  let shipping := shipping_cost total
  total + shipping = 102 := by
{
  let total := total_cost_before_shipping in
  let shipping := shipping_cost total in
  rw [total_cost_before_shipping, shipping_cost],
  refl,
}

-- Using 'sorry' to omit proof details and ensure the Lean code builds successfully
sorry

end total_bill_l105_105190


namespace crossing_BBEE_bbEE_proportion_F2_generation_fertile_offspring_proportion_homozygous_fertile_genotypes_l105_105076

-- Define the alleles and corresponding flower types
inductive Allele
| B
| b
| E
| e

-- Define the genotype
structure Genotype where
  B : Allele
  E : Allele

-- Define the flower type
inductive FlowerType
| wild
| doublePistils
| sterile

-- Define the rules for flower types based on genotypes
def flowerType (g : Genotype) : FlowerType :=
  match g with
  | ⟨Allele.B, Allele.E⟩ => FlowerType.wild
  | ⟨Allele.b, Allele.E⟩ => FlowerType.doublePistils
  | _ => FlowerType.sterile

-- Given conditions
def BBEE := Genotype.mk Allele.B Allele.E
def bbEE := Genotype.mk Allele.b Allele.E
def BbEe := ⟨Allele.B, Allele.E⟩

-- Proof statements
theorem crossing_BBEE_bbEE :
  (F1_genotype : Genotype) → (flowerType F1_genotype = FlowerType.wild) :=
sorry

theorem proportion_F2_generation :
  (F2_proportion : ℝ) → (F2_proportion = (3/4)) :=
sorry

theorem fertile_offspring_proportion :
  (fertile_offspring : ℝ) → (fertile_offspring = (3/4)) :=
sorry

theorem homozygous_fertile_genotypes :
  (homozygous_genotype: Genotype) → (flowerType homozygous_genotype = FlowerType.wild) ∨ (flowerType homozygous_genotype = FlowerType.doublePistils)=
sorry

end crossing_BBEE_bbEE_proportion_F2_generation_fertile_offspring_proportion_homozygous_fertile_genotypes_l105_105076


namespace sum_of_all_n_l105_105564

def binomial (n k : ℕ) : ℕ := Nat.choose n k

lemma pascal_identity (n k : ℕ) :
  binomial (n - 1) (k - 1) + binomial (n - 1) k = binomial n k :=
  Nat.choose_succ_succ n k

lemma binomial_symmetry (n k : ℕ) : binomial n k = binomial n (n - k) :=
  Nat.choose_symm n k

theorem sum_of_all_n :
  (∀ n : ℕ, binomial 25 n + binomial 25 12 = binomial 26 13 → n = 13) ∧ 13 = 13 :=
by
  sorry

end sum_of_all_n_l105_105564


namespace signs_used_fully_l105_105537

theorem signs_used_fully (total_signs : ℕ) (unused_signs : ℕ) (additional_area_codes : ℕ) (area_code_length : ℕ) :
  total_signs = 124 → unused_signs = 2 → additional_area_codes = 488 → area_code_length = 2 →
  let used_signs := total_signs - unused_signs in
  used_signs = 120 :=
by
  intros h1 h2 h3 h4
  simp only [h1, h2, h3, h4]
  let used_signs := 124 - 2
  have : (124 - 2)^2 + 488 = 124^2 := by sorry
  have : (124 - 2)^2 + 488 = used_signs^2 + 488 := by sorry
  have eq1 : 124^2 = used_signs^2 + 488 := by sorry
  have eq2 : used_signs^2 = 14888 := by linarith
  have h5 : used_signs = 120 := by linarith
  exact h5

end signs_used_fully_l105_105537


namespace unbroken_seashells_l105_105113

theorem unbroken_seashells (total_seashells : ℕ) (broken_seashells : ℕ) (h1 : total_seashells = 23) (h2 : broken_seashells = 11) : total_seashells - broken_seashells = 12 := by
  sorry

end unbroken_seashells_l105_105113


namespace simplify_fraction_l105_105491

theorem simplify_fraction : (12.factorial : ℚ) / (10.factorial + 3 * 9.factorial) = 1320 / 13 := by
  sorry

end simplify_fraction_l105_105491


namespace matrix_square_eq_eigenvalues_of_A_l105_105736

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 1], ![2, 2]]

theorem matrix_square_eq :
  A*A = ![![11, 5], ![10, 6]] := sorry

theorem eigenvalues_of_A :
  let p := (Polynomial.ringEquivMatrix (Fin 2) ℤ).toFun A 
  let charPoly := Polynomial.charPoly p
  charPoly = (Polynomial.C 1 * X^2 - Polynomial.C 5 * X + Polynomial.C 4) ∧
  (Polynomial.root (Polynomial.C 1 * X^2 - Polynomial.C 5 * X + Polynomial.C 4) 1) ∧
  (Polynomial.root (Polynomial.C 1 * X^2 - Polynomial.C 5 * X + Polynomial.C 4) 4) := sorry

end matrix_square_eq_eigenvalues_of_A_l105_105736


namespace num_surjections_l105_105456

open Finset

theorem num_surjections (A B : Finset ℕ) (hA : A.card = 4) (hB : B.card = 3) :
  (∃ f : A → B, Function.Surjective f) →
  {f : A → B | Function.Surjective f}.to_finset.card = 36 :=
by
  sorry

end num_surjections_l105_105456


namespace central_cell_value_l105_105386

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l105_105386


namespace correct_statements_count_l105_105657

def prism (faces : Set (Set Point)) : Prop := sorry
def right_prism (faces : Set (Set Point)) : Prop := sorry
def parallelepiped (faces : Set (Set Point)) : Prop := sorry
def rectangular_parallelepiped (faces : Set (Set Point)) : Prop := sorry
def regular_quadrilateral_prism (faces : Set (Set Point)) : Prop := sorry

def statement1 (faces : Set (Set Point)) : Prop :=
  prism faces → (∀ f ∈ faces, is_square f) → right_prism faces

def statement2 (faces : Set (Set Point)) : Prop :=
  parallelepiped faces → (∀ f1 f2 ∈ faces, opposite_faces f1 f2 → congruent f1 f2) → rectangular_parallelepiped faces

def statement3 (faces : Set (Set Point)) : Prop :=
  rectangular_parallelepiped faces → regular_quadrilateral_prism faces

def statement4 (faces : Set (Set Point)) : Prop :=
  prism faces → (∃ f1 f2 ∈ faces, adjacent_lateral_faces f1 f2 ∧ is_rectangle f1 ∧ is_rectangle f2) → right_prism faces

theorem correct_statements_count : ∀ faces : Set (Set Point), 
  (statement2 faces) ∧ (statement4 faces) → 
  ¬(statement1 faces) ∧ ¬(statement3 faces) → 
  count_correct_statements faces = 2 := 
sorry

end correct_statements_count_l105_105657


namespace train_crosses_pole_l105_105609

theorem train_crosses_pole
  (speed_kmph : ℝ)
  (train_length_meters : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (time_seconds : ℝ)
  (h1 : speed_kmph = 270)
  (h2 : train_length_meters = 375.03)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_mps = speed_kmph * conversion_factor)
  (h5 : time_seconds = train_length_meters / speed_mps)
  : time_seconds = 5.0004 :=
by
  sorry

end train_crosses_pole_l105_105609


namespace keiko_speed_l105_105447

-- Define the variables and constants used in the problem
variable (a b : ℝ) -- The length of the straight sections and radius of the inner semicircle
variable (s : ℝ) -- Keiko's speed in meters per second
constant width : ℝ := 6 -- The width of the track in meters
constant time_difference : ℝ := 36 -- The time difference in seconds

-- Define the lengths of the inner and outer tracks
def L_inner : ℝ := 2 * a + 2 * Real.pi * b
def L_outer : ℝ := 2 * a + 2 * Real.pi * (b + width)

-- Define the equation based on the time difference
def track_equation : Prop := (L_outer / s = L_inner / s + time_difference)

-- The theorem to prove Keiko's speed
theorem keiko_speed : track_equation a b s →
  s = Real.pi / 3 := by
  sorry

end keiko_speed_l105_105447


namespace central_cell_value_l105_105378

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l105_105378


namespace permutation_inequality_l105_105451

theorem permutation_inequality (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) ∧ 
  2 * (x * y + z * w)^2 > (x^2 + y^2) * (z^2 + w^2) := 
sorry

end permutation_inequality_l105_105451


namespace distance_between_intersections_l105_105259

noncomputable def f (x : ℝ) : ℝ := a * x + b

theorem distance_between_intersections :
  (let a := (8 : ℝ);
       b := (-3/2 : ℝ) in
    √((a^2 + 1) * (a^2 + 4 * b))) = 3 * √2 :=
by
  -- conditions
  have h₁ : (let a := 8; b := -3/2 in (a^2 + 1) * (a^2 + 4 * (b + 2))) = 90 := sorry,
  have h₂ : (let a := 8; b := -3/2 in (a^2 + 1) * (a^2 + 4 * (b + 3))) = 126 := sorry,
  -- goal
  sorry

end distance_between_intersections_l105_105259


namespace youngest_person_age_l105_105105

theorem youngest_person_age :
  ∀ (Y : ℝ), 
    (let total_age_now := 7 * 30 in
     let total_age_when_youngest_born := 6 * 22 in
     total_age_now = total_age_when_youngest_born + 7 * Y) →
    Y = 11.14 :=
by
  intro Y,
  assume h,
  have h1 : 210 = 132 + 7 * Y := h,
  have h2 : 210 - 132 = 7 * Y := by linarith,
  have h3 : 78 = 7 * Y := h2,
  have h4 : Y = 78 / 7 := by linarith,
  show Y = 11.14, from h4

end youngest_person_age_l105_105105


namespace quadratic_has_two_distinct_real_roots_l105_105097

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  let a := 1
      c := -8
      b := m
      Δ := b^2 - 4 * a * c 
  in (Δ > 0) :=
by
  let a := 1
  let c := -8
  let b := m
  let Δ := b^2 - 4 * a * c
  show Δ > 0
  sorry

end quadratic_has_two_distinct_real_roots_l105_105097


namespace find_y_l105_105221

theorem find_y :
  (∃ y : ℝ, (4 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4) ∧ y = 1251) :=
by
  sorry

end find_y_l105_105221


namespace x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq_l105_105760

theorem x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq (x y : ℝ) :
  ¬((x > y) → (x^2 > y^2)) ∧ ¬((x^2 > y^2) → (x > y)) :=
by
  sorry

end x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq_l105_105760


namespace quadratic_no_positive_real_root_l105_105192

theorem quadratic_no_positive_real_root :
  let S := { b : ℤ | abs b ≤ 3 },
      T := { c : ℤ | abs c ≤ 3 },
      pairs := { (b, c) : ℤ × ℤ | b ∈ S ∧ c ∈ T } in
  ∃ (p : ℕ),
  p = ( ∑ (b, c) in pairs, if (b^2 - 4*c < 0) ∨ (sqrt (b^2 - 4*c) ≤ b) then 1 else 0 ) 
  ∧ p = 34 ∧ card pairs = 49 :=
by sorry

end quadratic_no_positive_real_root_l105_105192


namespace total_birds_on_fence_l105_105108

variable (initial_birds : ℕ := 1)
variable (added_birds : ℕ := 4)

theorem total_birds_on_fence : initial_birds + added_birds = 5 := by
  sorry

end total_birds_on_fence_l105_105108


namespace problem_proof_l105_105711

variable {a1 a2 b1 b2 b3 : ℝ}

theorem problem_proof 
  (h1 : ∃ d, -7 + d = a1 ∧ a1 + d = a2 ∧ a2 + d = -1)
  (h2 : ∃ r, -4 * r = b1 ∧ b1 * r = b2 ∧ b2 * r = b3 ∧ b3 * r = -1)
  (ha : a2 - a1 = 2)
  (hb : b2 = -2) :
  (a2 - a1) / b2 = -1 :=
by
  sorry

end problem_proof_l105_105711


namespace original_fraction_is_one_third_l105_105040

theorem original_fraction_is_one_third (a b : ℕ) 
  (coprime_ab : Nat.gcd a b = 1) 
  (h : (a + 2) * b = 3 * a * b^2) : 
  (a = 1 ∧ b = 3) := 
by 
  sorry

end original_fraction_is_one_third_l105_105040


namespace parabola_focus_l105_105225

theorem parabola_focus (a : ℝ) (h k x : ℝ) (hx : h = 0) (kx : k = 0) (a_eq : a = -1/16) :
  focus (y = -a * x^2) = (0, -4) :=
by
  sorry

end parabola_focus_l105_105225


namespace not_parallel_to_a_l105_105464

noncomputable def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem not_parallel_to_a : ∀ k : ℝ, ¬ is_parallel (k^2 + 1, k^2 + 1) (1, -2) :=
sorry

end not_parallel_to_a_l105_105464


namespace age_of_youngest_boy_l105_105146

theorem age_of_youngest_boy (average_age : ℕ) (age_proportion : ℕ → ℕ) 
  (h1 : average_age = 120) 
  (h2 : ∀ x, age_proportion x = 2 * x ∨ age_proportion x = 6 * x ∨ age_proportion x = 8 * x)
  (total_age : ℕ) 
  (h3 : total_age = 3 * average_age) :
  ∃ x, age_proportion x = 2 * x ∧ 2 * x * (3 * average_age / total_age) = 45 :=
by {
  sorry
}

end age_of_youngest_boy_l105_105146


namespace shaded_area_l105_105114

-- Define the radii of the two smaller circles
def radius1 : ℝ := 4
def radius2 : ℝ := 5

-- Define the radius of the circumscribing circle
def large_radius : ℝ := 9

-- Prove that the area of the shaded region is 40π
theorem shaded_area :
  let area_large := real.pi * large_radius^2 in
  let area_small1 := real.pi * radius1^2 in
  let area_small2 := real.pi * radius2^2 in
  area_large - (area_small1 + area_small2) = 40 * real.pi :=
by
  sorry

end shaded_area_l105_105114


namespace range_of_a_l105_105358

theorem range_of_a (h : ¬ ∃ x : ℝ, x < 2023 ∧ x > a) : a ≥ 2023 := 
sorry

end range_of_a_l105_105358


namespace arrange_students_l105_105476

theorem arrange_students (boys girls : ℕ) (total_students : ℕ) : 
  boys + girls = total_students ∧ boys = 3 ∧ girls = 2 ∧ total_students = 5 →
  let female_pair_arrangement := factorial 4 * factorial 2 in
  female_pair_arrangement = 48 :=
by
  intros
  suffices : factorial 4 * factorial 2 = 48, from this
  sorry

end arrange_students_l105_105476


namespace mountain_hike_rate_l105_105160

theorem mountain_hike_rate :
  ∃ R : ℝ, (∀ (rate_descent rate_ascent : ℝ) (time_ascent : ℝ), 
    rate_descent = 1.5 * rate_ascent ∧ time_ascent = 2 ∧ (2 * rate_descent * time_ascent) = 21 → 
    rate_ascent = 7) :=
begin
  sorry
end

end mountain_hike_rate_l105_105160


namespace living_room_size_is_96_l105_105856

-- Define the total area of the apartment
def total_area : ℕ := 16 * 10

-- Define the number of units
def units : ℕ := 5

-- Define the size of one unit
def size_of_one_unit : ℕ := total_area / units

-- Define the size of the living room
def living_room_size : ℕ := size_of_one_unit * 3

-- Proving that the living room size is indeed 96 square feet
theorem living_room_size_is_96 : living_room_size = 96 := 
by
  -- not providing proof, thus using sorry
  sorry

end living_room_size_is_96_l105_105856


namespace largest_coefficient_term_l105_105934

theorem largest_coefficient_term:
  (∃ r: ℕ, r = 4 ∧ ∀ k: ℕ, 0 ≤ k ∧ k ≤ 7 → nat.choose 7 k ≤ nat.choose 7 4) :=
sorry

end largest_coefficient_term_l105_105934


namespace f_neg_def_l105_105831

variable (f : ℝ → ℝ)
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_def_pos : ∀ x : ℝ, 0 < x → f x = x * (1 + x)

theorem f_neg_def (x : ℝ) (hx : x < 0) : f x = x * (1 - x) := by
  sorry

end f_neg_def_l105_105831


namespace log_base_conversion_l105_105344

theorem log_base_conversion (x : ℝ) 
  (h : log 16 (x - 3) = 1 / 2) : log 81 x = (log 3 7) / 4 :=
by
  sorry

end log_base_conversion_l105_105344


namespace lemonade_percentage_l105_105603

theorem lemonade_percentage (V : ℝ) (L : ℝ) :
  (0.80 * 0.40 * V + (100 - L) / 100 * 0.60 * V = 0.65 * V) →
  L = 99.45 :=
by
  intro h
  -- The proof would go here
  sorry

end lemonade_percentage_l105_105603


namespace handshakes_at_event_l105_105625

theorem handshakes_at_event (total_people group1_people group2_people : ℕ) 
  (h_total : total_people = 40)
  (h_group1 : group1_people = 25)
  (h_group2 : group2_people = 15)
  (h_interaction1 : ∀ x y, x ∈ group1_people → y ∈ group1_people → x ≠ y → ∀ interaction, interaction = "hug")
  (h_interaction2 : ∀ x y, x ∈ group2_people → y ∈ group1_people → x ≠ y → ∀ interaction, interaction = "shake")
  (h_interaction3 : ∀ x y, x ∈ group2_people → y ∈ group2_people → x ≠ y → ∀ interaction, interaction = "shake"):
  handshakes = 292 :=
by
  sorry

end handshakes_at_event_l105_105625


namespace percent_parents_without_fulltime_jobs_l105_105787

theorem percent_parents_without_fulltime_jobs
  (total_parents : ℕ)
  (percent_women : ℚ)
  (percent_mothers_with_jobs : ℚ)
  (percent_fathers_with_jobs : ℚ)
  (h_percent_women : percent_women = 4 / 10)
  (h_percent_mothers_with_jobs : percent_mothers_with_jobs = 9 / 10)
  (h_percent_fathers_with_jobs : percent_fathers_with_jobs = 3 / 4) :
  let num_mothers := percent_women * total_parents,
      num_fathers := (1 - percent_women) * total_parents,
      mothers_with_jobs := percent_mothers_with_jobs * num_mothers,
      fathers_with_jobs := percent_fathers_with_jobs * num_fathers,
      total_with_jobs := mothers_with_jobs + fathers_with_jobs,
      total_without_jobs := total_parents - total_with_jobs,
      percent_without_jobs := total_without_jobs / total_parents * 100 in
  percent_without_jobs = 19 := by
  sorry

end percent_parents_without_fulltime_jobs_l105_105787


namespace quadratic_roots_distinct_l105_105093

theorem quadratic_roots_distinct (m : ℝ) :
  let Δ := m^2 + 32 in
  Δ > 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 - 8 = 0 ∧ x2^2 + m * x2 - 8 = 0) :=
begin
  sorry
end

end quadratic_roots_distinct_l105_105093


namespace return_journey_equation_l105_105662

noncomputable def distance_uphill  := 3.3 - x - y
noncomputable def speed_uphill   := 3
noncomputable def speed_flat     := 4
noncomputable def speed_downhill := 5

theorem return_journey_equation (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_total_distance : x + y + distance_uphill = 3.3)
  (h_first_eq : x / 3 + y / 4 + distance_uphill / 5 = 51 / 60) :
  x / speed_downhill + y / speed_flat + distance_uphill / speed_uphill = 53 / 60 :=
sorry

end return_journey_equation_l105_105662


namespace central_cell_value_l105_105393

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l105_105393


namespace find_central_cell_l105_105416

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l105_105416


namespace total_items_ike_and_mike_l105_105773

/-
  Given the conditions:
  - Total money is 50 dollars
  - Cost of a sandwich is 6 dollars
  - Cost of a pastry is 1.5 dollars
  
  Prove that the total number of items bought is 9.
-/

theorem total_items_ike_and_mike (total_money sandwich_cost pastry_cost : ℝ) 
  (h1 : total_money = 50) 
  (h2 : sandwich_cost = 6) 
  (h3 : pastry_cost = 1.5) : 
  let sandwiches := ⌊ total_money / sandwich_cost ⌋ in
  let money_left := total_money - sandwiches * sandwich_cost in
  let pastries := ⌊ money_left / pastry_cost ⌋ in
  sandwiches + pastries = 9 :=
by
  sorry

end total_items_ike_and_mike_l105_105773


namespace has_no_real_solution_l105_105530

noncomputable def quadratic_function := 
  ∃ (a b c : ℝ), 
    (a ≠ 0) ∧ 
    (c = 2) ∧ 
    (a + b + c = 1) ∧ 
    (4a + 2b + c = 2) ∧ 
    (9a + 3b + c = 5) ∧ 
    (16a + 4b + c = 10) ∧ 
    (∀ x : ℝ, a * x^2 + b * x + c = 0 → false)

theorem has_no_real_solution : quadratic_function :=
sorry

end has_no_real_solution_l105_105530


namespace reversed_digit_multiple_of_sum_l105_105148

variable (u v k : ℕ)

theorem reversed_digit_multiple_of_sum (h1 : 10 * u + v = k * (u + v)) :
  10 * v + u = (11 - k) * (u + v) :=
sorry

end reversed_digit_multiple_of_sum_l105_105148


namespace proof_problem_l105_105740

def f (x : ℝ) : ℝ := (2017 ^ x - 1) / (2017 ^ x + 1)

def g (x : ℝ) : ℝ := 1 - x^2

def p : Prop := ∀ x : ℝ, f(-x) = -f(x)

def q : Prop := ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → 0 < x2 → g(x1) < g(x2)

theorem proof_problem : p ∨ q :=
by
  have h1 : p := 
    by sorry
  have h2 : ¬q := 
    by sorry
  exact Or.inl h1

end proof_problem_l105_105740


namespace concyclic_P_Q_P1_Q1_l105_105440

-- Definitions of the points and geometric relations
variables {α : Type} [EuclideanSpace α]
variables (A B C A1 B1 P Q P1 Q1 : α)

-- Assume points A1 and B1 are on segments BC and AC respectively
variables (hA1_on_BC : A1 ∈ line B C) (hB1_on_AC : B1 ∈ line A C)
-- Assume P and Q are on segments AA1 and BB1, respectively
variables (hP_on_AA1 : P ∈ line A A1) (hQ_on_BB1 : Q ∈ line B B1)
-- PQ is parallel to AB
variables (hPQ_parallel_AB : line P Q ∥ line A B)
-- Angle conditions for P1 and Q1
variables (hAngle_PP1C_eq_ABC : ∠ P P1 C = ∠ A B C) 
variables (hAngle_CQ1Q_eq_CBA : ∠ C Q1 Q = ∠ C B A)

-- The proof problem: show that P, Q, P1, and Q1 are concyclic
theorem concyclic_P_Q_P1_Q1 (hA1_on_BC : A1 ∈ line B C) (hB1_on_AC : B1 ∈ line A C) (hP_on_AA1 : P ∈ line A A1) 
  (hQ_on_BB1 : Q ∈ line B B1) (hPQ_parallel_AB : line P Q ∥ line A B) 
  (hAngle_PP1C_eq_ABC : ∠ P P1 C = ∠ A B C) (hAngle_CQ1Q_eq_CBA : ∠ C Q1 Q = ∠ C B A) : 
  ∃ (ω : circle α), P ∈ ω ∧ Q ∈ ω ∧ P1 ∈ ω ∧ Q1 ∈ ω := 
sorry

end concyclic_P_Q_P1_Q1_l105_105440


namespace factorize_quadratic_example_l105_105663

theorem factorize_quadratic_example (x : ℝ) :
  4 * x^2 - 8 * x + 4 = 4 * (x - 1)^2 :=
by
  sorry

end factorize_quadratic_example_l105_105663


namespace simplify_factorial_l105_105495

theorem simplify_factorial :
  ∀ (f12 f10 f9 : ℕ), f12 = 12! ∧ f10 = 10! ∧ f9 = 9! →
  (f12 / (f10 + 3 * f9) = 4 / 3) :=
by
  sorry

end simplify_factorial_l105_105495


namespace solve_quadratic_l105_105891

theorem solve_quadratic : ∀ (x : ℝ), x^2 - 5 * x + 1 = 0 →
  (x = (5 + Real.sqrt 21) / 2) ∨ (x = (5 - Real.sqrt 21) / 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_l105_105891


namespace octal_742_to_decimal_l105_105592

theorem octal_742_to_decimal : (7 * 8^2 + 4 * 8^1 + 2 * 8^0 = 482) :=
by
  sorry

end octal_742_to_decimal_l105_105592


namespace correct_answer_l105_105980

def expression1 := -(-2)
def expression2 := -| -2 |
def expression3 := -(2^2)
def expression4 := | -2 |^2

theorem correct_answer : ((expression2 < 0) ∧ (expression3 < 0)) ∧ ((expression1 >= 0) ∧ (expression4 >= 0)) :=
by
  have h1 : expression1 = 2 := by simp [expression1]
  have h2 : expression2 = -2 := by simp [expression2]
  have h3 : expression3 = -4 := by simp [expression3]
  have h4 : expression4 = 4 := by simp [expression4]

  exact ⟨⟨by linarith [h2], by linarith [h3]⟩, ⟨by linarith [h1], by linarith [h4]⟩⟩


end correct_answer_l105_105980


namespace game_ends_after_37_rounds_l105_105927

def initial_tokens : ℕ × ℕ × ℕ := (15, 14, 13)

def distribute_tokens (t : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  if t.1 > t.2 ∧ t.1 > t.3 then
    (t.1 - 2, t.2 + 1, t.3 + 1)
  else if t.2 > t.1 ∧ t.2 > t.3 then
    (t.1 + 1, t.2 - 2, t.3 + 1)
  else
    (t.1 + 1, t.2 + 1, t.3 - 2)

def play_game (t : ℕ × ℕ × ℕ) (n : ℕ) : ℕ × ℕ × ℕ :=
  match n with
  | 0   => t
  | n+1 => play_game (distribute_tokens t) n

theorem game_ends_after_37_rounds :
  ∃ n, n = 37 ∧ play_game initial_tokens n = (0, 3, 2) :=
by
  exists 37
  simp [play_game, initial_tokens, distribute_tokens]
  sorry

end game_ends_after_37_rounds_l105_105927


namespace least_possible_k_l105_105908

theorem least_possible_k :
  ∃ k, k = 2016 ∧
  ∀ (f g : ℝ → ℝ) (Hf : f = (λ x, (x - 1) * (x - 2) * ... * (x - 2016))) 
    (Hg : g = (λ x, (x - 1) * (x - 2) * ... * (x - 2016))),
    (∀ k_factors k : ℕ, k_factors = 2016 ∧ k ≤ k_factors ∧
    (∃ f' g' : ℝ → ℝ, (f' = erase_factors f k) ∧ (g' = erase_factors g k) ∧
    (∃ x : ℝ, f' x = g' x) → False)) :=
sorry

-- Helper function to represent the process of erasing k factors.
noncomputable def erase_factors (h : ℝ → ℝ) (k : ℕ) : ℝ → ℝ := 
  -- This is just a placeholder. Actual implementation is beyond this example.
  h 


end least_possible_k_l105_105908


namespace position_of_2014_in_sequence_l105_105482

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m * m * m = n

def Nat_set_A := {n : ℕ | 1 ≤ n ∧ n ≤ 10000}

def filtered_set := {n ∈ Nat_set_A | ¬ is_perfect_square n ∧ ¬ is_perfect_cube n}

def sequence := (filtered_set.to_finset.sort (≤) : list ℕ)

theorem position_of_2014_in_sequence :
  list.index_of 2014 sequence = 1961 :=
sorry

end position_of_2014_in_sequence_l105_105482


namespace ellipse_focus_perpendicular_area_9_l105_105691

theorem ellipse_focus_perpendicular_area_9 (a b c : ℝ) (F1 F2 P : EuclideanSpace ℝ (Fin 2)) 
(h1 : a > b ∧ b > 0) 
(h2 : dist P F1 + dist P F2 = 2 * a) 
(h3 : inner (P -ᵥ F1) (P -ᵥ F2) = 0)
(area_triangle : 1 / 2 * dist P F1 * dist P F2 = 9) 
(hb : b = Real.sqrt(a^2 - c^2))
(hfocidist : 4 * c^2 = dist F1 F2 ^ 2) : 
b = 3 :=
sorry

end ellipse_focus_perpendicular_area_9_l105_105691


namespace find_smallest_A_divisible_by_51_l105_105975

theorem find_smallest_A_divisible_by_51 :
  ∃ (x y : ℕ), (A = 1100 * x + 11 * y) ∧ 
    (0 ≤ x) ∧ (x ≤ 9) ∧ 
    (0 ≤ y) ∧ (y ≤ 9) ∧ 
    (A % 51 = 0) ∧ 
    (A = 1122) :=
sorry

end find_smallest_A_divisible_by_51_l105_105975


namespace white_trees_count_l105_105172

noncomputable def calculate_white_trees (total_trees pink_percent red_trees : ℕ) : ℕ :=
  total_trees - (total_trees * pink_percent / 100 + red_trees)

theorem white_trees_count 
  (h1 : total_trees = 42)
  (h2 : pink_percent = 100 / 3)
  (h3 : red_trees = 2) :
  calculate_white_trees total_trees pink_percent red_trees = 26 :=
by
  -- proof will go here
  sorry

end white_trees_count_l105_105172


namespace sum_abc_eq_ten_l105_105347

theorem sum_abc_eq_ten (a b c : ℝ) (h : (a - 5)^2 + (b - 3)^2 + (c - 2)^2 = 0) : a + b + c = 10 :=
by
  sorry

end sum_abc_eq_ten_l105_105347


namespace transformation_correct_l105_105548

theorem transformation_correct:
  ∀ (x : ℝ), (∃ c, y = c * sin(2 * x + π / 5)) ↔ 
             (∃ d, y = d * sin (x + 7 * π / 10)) :=
begin
  sorry
end

end transformation_correct_l105_105548


namespace factorial_simplification_l105_105500

theorem factorial_simplification :
  (12.factorial : ℚ) / ((10.factorial : ℚ) + 3 * (9.factorial : ℚ)) = 1320 / 13 := by
sorry

end factorial_simplification_l105_105500


namespace least_positive_integer_mod_cond_l105_105968

theorem least_positive_integer_mod_cond (N : ℕ) :
  (N % 6 = 5) ∧ 
  (N % 7 = 6) ∧ 
  (N % 8 = 7) ∧ 
  (N % 9 = 8) ∧ 
  (N % 10 = 9) ∧ 
  (N % 11 = 10) →
  N = 27719 :=
by
  sorry

end least_positive_integer_mod_cond_l105_105968


namespace quadratic_has_two_distinct_real_roots_l105_105099

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  let a := 1
      c := -8
      b := m
      Δ := b^2 - 4 * a * c 
  in (Δ > 0) :=
by
  let a := 1
  let c := -8
  let b := m
  let Δ := b^2 - 4 * a * c
  show Δ > 0
  sorry

end quadratic_has_two_distinct_real_roots_l105_105099


namespace good_coloring_count_l105_105940

noncomputable def c_n (n : ℕ) : ℤ :=
  1 / 2 * (3^(n + 1) + (-1)^(n + 1))

theorem good_coloring_count (n : ℕ) : 
  ∃ c : ℕ → ℤ, c n = c_n n := sorry

end good_coloring_count_l105_105940


namespace maximum_T3_l105_105701

-- Define the sequence {a_n} under given conditions
variables (a : ℕ → ℤ)
axiom a1 : a 1 = 10
axiom a2_is_integer : ∃ k : ℤ, a 2 = k
axiom a3_range : 3 ≤ a 3 ∧ a 3 ≤ 5

-- Prove the general term formula
def general_term_formula (n : ℕ) : ℤ := 13 - 3 * n

-- The sequence {b_n}
def b (n : ℕ) : ℚ := 1 / ((a n) * (a (n + 1)))

-- Define the sum of the first n terms of {b_n}
def T (n : ℕ) : ℚ := ∑ i in finset.range n, b i

-- Prove that the maximum value of T_n when n = 3 is 3/10
theorem maximum_T3 : T 3 = 3 / 10 :=
by
  -- Mathematical proof goes here but skipping with sorry
  sorry

end maximum_T3_l105_105701


namespace total_matches_played_l105_105471

def isKnockoutTournament (teams : Nat) :=
  ∀ (matches: Nat), matches = teams - 1

theorem total_matches_played (teams : Nat) (matches: Nat) :
  isKnockoutTournament teams → teams = 20 → matches = 19 :=
by
  intros h_tournament h_teams
  have h_matches : matches = teams - 1 := h_tournament matches
  rw h_teams at h_matches
  exact h_matches

end total_matches_played_l105_105471


namespace Jasmine_maximum_cards_l105_105445

theorem Jasmine_maximum_cards (n : ℕ) (h_budget : 9.00) (h_cost : 0.95) :
  ∃ (n : ℕ), 0.95 * n ≤ 9.00 ∧ n = 9 :=
by
  sorry

end Jasmine_maximum_cards_l105_105445


namespace units_digit_7_pow_103_l105_105134

theorem units_digit_7_pow_103 : Nat.mod (7 ^ 103) 10 = 3 := sorry

end units_digit_7_pow_103_l105_105134


namespace ratio_of_perimeter_of_one_sector_to_circumference_l105_105045
open Real

-- Conditions:
def radius (r : ℝ) : Prop := r > 0
def sector_count : ℕ := 3

-- Circumference of the circle
def circumference (r : ℝ) := 2 * π * r

-- Length of the perimeter of one sector
def perimeter_of_one_sector (r : ℝ) := (2 * π * r) / 3 + 2 * r

-- Required Ratio
def required_ratio (r : ℝ) := (perimeter_of_one_sector r) / (circumference r)

-- Mathematical Problem:
theorem ratio_of_perimeter_of_one_sector_to_circumference (r : ℝ) (hr : radius r) :
  required_ratio r = (π + 3) / (3 * π) :=
by
  sorry

end ratio_of_perimeter_of_one_sector_to_circumference_l105_105045


namespace radius_distance_relation_l105_105821

variables {A B C : Point} (Γ₁ Γ₂ ω₀ : Circle)
variables (ω : ℕ → Circle)
variables (r d : ℕ → ℝ)

def diam_circle (P Q : Point) : Circle := sorry  -- This is to define a circle with diameter PQ
def tangent (κ κ' κ'' : Circle) : Prop := sorry  -- This is to define that three circles are mutually tangent

-- Defining the properties as given in the conditions
axiom Γ₁_def : Γ₁ = diam_circle A B
axiom Γ₂_def : Γ₂ = diam_circle A C
axiom ω₀_def : ω₀ = diam_circle B C
axiom ω_def : ∀ n : ℕ, tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₁ (ω n) ∧ tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₂ (ω n) -- ωₙ is tangent to previous circle, Γ₁ and Γ₂

-- The main proof statement
theorem radius_distance_relation (n : ℕ) : r n = 2 * n * d n :=
sorry

end radius_distance_relation_l105_105821


namespace quadratic_roots_distinct_l105_105078

-- Define the quadratic equation condition
def quadratic_eq : (ℝ → ℝ) :=
  λ x m => x^2 + m * x - 8

-- State the problem
theorem quadratic_roots_distinct (m : ℝ) : 
  let Δ := m^2 + 32 in Δ > 0 :=
  by
   -- We need this theorem to state that the equation always has distinct real roots
  let Δ := m^2 + 32
  sorry

end quadratic_roots_distinct_l105_105078
