import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Modular
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Powers
import Mathlib.Algebra.Prime
import Mathlib.Algebra.Ring
import Mathlib.Analysis.Calculus.Limits
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.LinearAlgebra.VecSpace
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.NumberTheory.Basic
import Mathlib.ProbTheory.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Local

namespace triangle_cosine_l827_827993

theorem triangle_cosine {A B C : Type} [EuclideanGeometry A B C] (h : ∠ABC = 90°) (AB : ℝ) (AC : ℝ) (hAB : AB = 12) (hAC : AC = 20) : cos ∠BAC = 4 / 5 :=
by
  sorry

end triangle_cosine_l827_827993


namespace time_to_cut_one_piece_l827_827787

theorem time_to_cut_one_piece (total_time : ℕ) (number_of_pieces : ℕ) (time_per_piece : ℝ) 
  (h1 : total_time = 580) (h2 : number_of_pieces = 146) (h3 : time_per_piece = total_time / number_of_pieces) :
  time_per_piece ≈ 4 :=
by sorry

end time_to_cut_one_piece_l827_827787


namespace problem_2_power_condition_l827_827452

def divisible (a b : ℕ) : Prop := ∃ (c : ℕ), b = a * c

theorem problem_2_power_condition :
  ∀ k : ℕ,
    (∀ n : ℕ, n > 0 → ¬ divisible (2 ^ ((k - 1) * n + 1)) ((kn)! / n!)) ↔
    ∃ (a : ℕ), k = 2 ^ a :=
by
  sorry

end problem_2_power_condition_l827_827452


namespace problem1_decreasing_interval_problem2_range_problem3_lambda_value_l827_827930

theorem problem1_decreasing_interval :
  ∀ (f : ℝ → ℝ) (ω : ℝ) (ϕ : ℝ), 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧ (∀ x : ℝ, f x = sqrt 3 * sin (ω * x + ϕ) + 2 * sin (ω * x + ϕ / 2)^2 - 1)
  → ∀ x ∈ Icc (-π/2) (π/4), f (2 * x) < f ((π/4) - (π/6)) :=
sorry

theorem problem2_range :
  ∀ (f : ℝ → ℝ) (g : ℝ → ℝ) (ω : ℝ) (ϕ : ℝ), 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧ (∀ x : ℝ, f x = sqrt 3 * sin (ω * x + ϕ) + 2 * sin (ω * x + ϕ / 2)^2 - 1)
  → ∀ x ∈ Icc (-π/12) (π/6), g x = 2 * sin (4 * x - π/3) 
g x ∈ Icc (-2) (sqrt 3) :=
sorry

theorem problem3_lambda_value :
  ∀ (f h : ℝ → ℝ) (ω : ℝ) (ϕ λ : ℝ), 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧ (∀ x : ℝ, f x = sqrt 3 * sin (ω * x + ϕ) + 2 * sin (ω * x + ϕ / 2)^2 - 1)
  ∧ (∀ x : ℝ, h x = f x + λ * cos (2 * x)) 
  ∧ (∀ x : ℝ, x = π/6) → 
  (λ : ℝ) = 2 * sqrt 3 / 3 :=
sorry

end problem1_decreasing_interval_problem2_range_problem3_lambda_value_l827_827930


namespace closest_distance_l827_827317

theorem closest_distance (x y z : ℕ)
  (h1 : x + y = 10)
  (h2 : y + z = 13)
  (h3 : z + x = 11) :
  min x (min y z) = 4 :=
by
  -- Here you would provide the proof steps in Lean, but for the statement itself, we leave it as sorry.
  sorry

end closest_distance_l827_827317


namespace movie_ticket_ratio_l827_827623

-- Definitions based on the conditions
def monday_cost : ℕ := 5
def wednesday_cost : ℕ := 2 * monday_cost

theorem movie_ticket_ratio (S : ℕ) (h1 : wednesday_cost + S = 35) :
  S / monday_cost = 5 :=
by
  -- Placeholder for proof
  sorry

end movie_ticket_ratio_l827_827623


namespace DC_dot_AP_range_l827_827161

def vector (α : Type) := α × α × α

def D : vector ℝ := (0, 0, 0)
def C : vector ℝ := (0, 1, 0)
def B : vector ℝ := (1, 1, 0)
def D1 : vector ℝ := (0, 0, 1)

def DC : vector ℝ := (0, 1, 0)
def BD1 : vector ℝ := (-1, -1, 1)

noncomputable def BP (λ : ℝ) : vector ℝ := (-λ, -λ, λ)
def AB : vector ℝ := (0, 0, 0)
noncomputable def AP (λ : ℝ) : vector ℝ := (1 - λ, -λ, λ)

noncomputable def dot_product : vector ℝ → vector ℝ → ℝ
| (x1, y1, z1), (x2, y2, z2) => x1 * x2 + y1 * y2 + z1 * z2

theorem DC_dot_AP_range : ∀ λ ∈ set.Icc (0:ℝ) 1, dot_product DC (AP λ) ∈ set.Icc (0:ℝ) 1 :=
by
  intro λ hλ
  have hAP : AP λ = (1 - λ, -λ, λ) := rfl
  rw [hAP]
  have hDot : dot_product DC (1 - λ, -λ, λ) = 1 - λ := by
    simp [dot_product, DC]
  exact set.mem_Icc.mpr ⟨by sorry, by sorry⟩

end DC_dot_AP_range_l827_827161


namespace problem1_correct_problem2_correct_l827_827819

noncomputable def problem1 : Real :=
  2 * Real.sqrt (2 / 3) - 3 * Real.sqrt (3 / 2) + Real.sqrt 24

theorem problem1_correct : problem1 = (7 * Real.sqrt 6) / 6 := by
  sorry

noncomputable def problem2 : Real :=
  Real.sqrt (25 / 2) + Real.sqrt 32 - Real.sqrt 18 - (Real.sqrt 2 - 1)^2

theorem problem2_correct : problem2 = (11 * Real.sqrt 2) / 2 - 3 := by
  sorry

end problem1_correct_problem2_correct_l827_827819


namespace probability_even_units_digit_l827_827352

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem probability_even_units_digit :
  (finset.card (finset.filter is_even (finset.range 10)) : ℚ) / finset.card (finset.range 10) = 1 / 2 :=
by
  sorry

end probability_even_units_digit_l827_827352


namespace min_sum_log_geq_four_l827_827958

theorem min_sum_log_geq_four (m n : ℝ) (hm : 0 < m) (hn : 0 < n) 
  (hlog : Real.log m / Real.log 3 + Real.log n / Real.log 3 ≥ 4) : 
  m + n ≥ 18 :=
sorry

end min_sum_log_geq_four_l827_827958


namespace ratio_of_red_to_black_l827_827984

theorem ratio_of_red_to_black (r b : ℕ) (h_r : r = 26) (h_b : b = 70) :
  r / Nat.gcd r b = 13 ∧ b / Nat.gcd r b = 35 :=
by
  sorry

end ratio_of_red_to_black_l827_827984


namespace sin_210_eq_neg_half_l827_827438

theorem sin_210_eq_neg_half :
  ∃ θ : ℝ, θ = 210 ∧ real.sin (θ * real.pi / 180) = -1/2 := sorry

end sin_210_eq_neg_half_l827_827438


namespace order_fractions_l827_827831

theorem order_fractions :
  let S := { (3 : ℚ) / 7, (3 : ℚ) / 2, (6 : ℚ) / 7, (3 : ℚ) / 5 }
  ∃ ordered_S, ordered_S = [ (3 : ℚ) / 7, (3 : ℚ) / 5, (6 : ℚ) / 7, (3 : ℚ) / 2 ] ∧
    list.sorted (<) ordered_S := by
{
  sorry
}

end order_fractions_l827_827831


namespace Dimitri_calories_l827_827440

theorem Dimitri_calories (burgers_per_day : ℕ) (calories_per_burger : ℕ) (days : ℕ) :
  (burgers_per_day = 3) → (calories_per_burger = 20) → (days = 2) →
  (burgers_per_day * calories_per_burger * days = 120) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Dimitri_calories_l827_827440


namespace set_of_x_for_which_f_x_is_greater_than_1_l827_827553

theorem set_of_x_for_which_f_x_is_greater_than_1 :
  ∀ x : ℝ, (2^x > 1) ↔ (x > 0) :=
by sorry

end set_of_x_for_which_f_x_is_greater_than_1_l827_827553


namespace sum_primes_between_10_and_20_is_60_l827_827230

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827230


namespace unique_function_l827_827855

open Function

noncomputable def is_perfect_square (f : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ g : ℕ, f n = g * g

noncomputable def satisfies_functional_equation (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (m + n) = f m + f n + 2 * m * n

theorem unique_function (f : ℕ → ℕ) (h1 : ∀ n : ℕ, is_perfect_square f n) 
  (h2 : satisfies_functional_equation f) : 
  f = (λ n, n * n) :=
sorry

end unique_function_l827_827855


namespace gcd_lcm_problem_l827_827667

theorem gcd_lcm_problem (b : ℤ) (x : ℕ) (hx_pos : 0 < x) (hx : x = 12) :
  gcd 30 b = x + 3 ∧ lcm 30 b = x * (x + 3) → b = 90 := 
by
  sorry

end gcd_lcm_problem_l827_827667


namespace find_intersection_l827_827457

noncomputable def intersection_of_lines : Prop :=
  ∃ (x y : ℚ), (5 * x - 3 * y = 15) ∧ (6 * x + 2 * y = 14) ∧ (x = 11 / 4) ∧ (y = -5 / 4)

theorem find_intersection : intersection_of_lines :=
  sorry

end find_intersection_l827_827457


namespace victor_percentage_l827_827712

-- Define Victor's obtained marks
def marks_obtained := 460

-- Define the maximum marks
def maximum_marks := 500

-- Compute the percentage of marks
def percentage_of_marks := (marks_obtained / maximum_marks.toFloat) * 100

-- Prove that Victor's percentage of marks is 92
theorem victor_percentage :
  percentage_of_marks = 92 := 
by
  sorry

end victor_percentage_l827_827712


namespace largest_prime_factor_12769_l827_827848

theorem largest_prime_factor_12769 : ∃ p, prime p ∧ p ∣ 12769 ∧ ∀ q, prime q ∧ q ∣ 12769 → q ≤ p :=
by
  use 251
  split
  { exact prime_251 }
  { split
    { exact dvd_of_mul_right_eq (by norm_num) }
    { intros q h_prime_q h_dvd_q
      -- proof that 251 is greater than or equal to any other prime factor
      sorry } }

end largest_prime_factor_12769_l827_827848


namespace period_of_f_l827_827493

noncomputable def f (x : ℝ) : ℝ := sorry

theorem period_of_f (a : ℝ) (h : a ≠ 0) (H : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : 
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = 4 * |a| :=
by
  sorry

end period_of_f_l827_827493


namespace moles_of_NaOH_l827_827957

-- Statement of the problem conditions and desired conclusion
theorem moles_of_NaOH (moles_H2SO4 moles_NaHSO4 : ℕ) (h : moles_H2SO4 = 3) (h_eq : moles_H2SO4 = moles_NaHSO4) : moles_NaHSO4 = 3 := by
  sorry

end moles_of_NaOH_l827_827957


namespace find_length_of_lateral_edge_l827_827791

noncomputable def length_of_lateral_edge (AB b : ℝ) : ℝ := b

theorem find_length_of_lateral_edge
  (S A B C : Point)
  (pyramid_SABC : regular_triangular_pyramid S A B C)
  (AB_eq : AB = 1)
  (medians_not_intersect : ¬ intersects (median_point A B) (median_point B C))
  (medians_on_cube_edges : ∃cube_edges, ∀edge ∈ cube_edges, edge.contains (median_point A B) ∧ edge.contains (median_point B C)) :
  length_of_lateral_edge AB (b := sqrt(3/2)) = (sqrt(6) / 2) := 
sorry

end find_length_of_lateral_edge_l827_827791


namespace even_sum_probability_l827_827702

-- Definitions of the conditions
def balls : list ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Function to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Function to calculate the sum of two numbers
def sum (a b : ℕ) : ℕ := a + b

-- Function to calculate the probability
noncomputable def probability_even_sum : ℚ := 
  let total_outcomes := 12 * 11 in
  let favorable_outcomes := 30 + 30 in
  favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem even_sum_probability :
  probability_even_sum = 5 / 11 :=
  by
  sorry

end even_sum_probability_l827_827702


namespace location_determined_l827_827313

def determine_location(p : String) : Prop :=
  p = "Longitude 118°E, Latitude 40°N"

axiom row_2_in_cinema : ¬determine_location "Row 2 in a cinema"
axiom daqiao_south_road_nanjing : ¬determine_location "Daqiao South Road in Nanjing"
axiom thirty_degrees_northeast : ¬determine_location "30° northeast"
axiom longitude_latitude : determine_location "Longitude 118°E, Latitude 40°N"

theorem location_determined : determine_location "Longitude 118°E, Latitude 40°N" :=
longitude_latitude

end location_determined_l827_827313


namespace min_distance_exists_l827_827908

open Real

-- Define the distance formula function
noncomputable def distance (x : ℝ) : ℝ :=
sqrt ((x - 1) ^ 2 + (3 - 2 * x) ^ 2 + (3 * x - 3) ^ 2)

theorem min_distance_exists :
  ∃ (x : ℝ), distance x = sqrt (14 * x^2 - 32 * x + 19) ∧
               ∀ y, distance y ≥ (sqrt 35) / 7 :=
sorry

end min_distance_exists_l827_827908


namespace max_quadratics_no_roots_l827_827382

def quadratic_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

theorem max_quadratics_no_roots :
  let a := 1000
  let b := 1001
  let c := 1002
  let perms := [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]
  (∀ p ∈ perms, quadratic_no_real_roots p.1 p.2 p.3) →
  perms.length = 6 :=
by
  sorry

end max_quadratics_no_roots_l827_827382


namespace chlorine_treats_120_cubic_feet_l827_827872

noncomputable def volume_of_pool (length width depth : ℕ) : ℕ :=
  length * width * depth

noncomputable def quarts_of_chlorine (total_spent cost_per_quart : ℕ) : ℕ :=
  total_spent / cost_per_quart

noncomputable def cubic_feet_per_quart (volume quarts : ℕ) : ℕ :=
  volume / quarts

theorem chlorine_treats_120_cubic_feet (length width depth cost_per_quart total_spent : ℕ)
  (h_length: length = 10) (h_width: width = 8) (h_depth: depth = 6)
  (h_cost: cost_per_quart = 3) (h_spent: total_spent = 12) :
  cubic_feet_per_quart (volume_of_pool length width depth) (quarts_of_chlorine total_spent cost_per_quart) = 120 :=
by
  rw [h_length, h_width, h_depth, h_cost, h_spent]
  dsimp only [volume_of_pool, quarts_of_chlorine, cubic_feet_per_quart]
  norm_num
  sorry

end chlorine_treats_120_cubic_feet_l827_827872


namespace ratio_y_x_l827_827175

theorem ratio_y_x (x y : ℝ) 
  (h1 : sqrt (3 * x) * (1 + 1 / (x + y)) = 2)
  (h2 : sqrt (7 * y) * (1 - 1 / (x + y)) = 4 * sqrt 2) : 
  y / x = 6 :=
sorry

end ratio_y_x_l827_827175


namespace sum_primes_between_10_and_20_l827_827288

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827288


namespace last_locker_opened_is_509_l827_827337

def num_lockers : ℕ := 512
def initial_opening (n : ℕ) : ℕ := if n = 1 then 1 else 0

def next_openings (state : list ℕ) (skip : ℕ) : list ℕ :=
  state.map_with_index (λ i l, if (i + skip + 1) % (skip + 3) = 0 then 1 else l)

def final_state : list ℕ :=
  let initial_state := (list.repeat 0 (num_lockers - 1)).cons 1
  in (fin.range num_lockers).foldl (λ st skip, next_openings st skip) initial_state

def last_opened_locker : ℕ :=
  match final_state.enum.find? (λ ⟨_, opened⟩, opened = 1) with
  | some (idx, _) => idx + 1 -- Adjust for 0-based index offset
  | none => 0 -- This should be impossible by the definition of the problem

theorem last_locker_opened_is_509 : last_opened_locker = 509 := by
  /- The proof goes here -/
  sorry

end last_locker_opened_is_509_l827_827337


namespace sum_of_fractions_l827_827817

theorem sum_of_fractions :
  (7:ℚ) / 12 + (11:ℚ) / 15 = 79 / 60 :=
by
  sorry

end sum_of_fractions_l827_827817


namespace five_digit_numbers_with_2_and_3_not_adjacent_six_digit_numbers_with_123_descending_l827_827471

-- Define the problem settings
def digits : finset ℕ := {0, 1, 2, 3, 4, 5}

-- Question 1: Prove the number of five-digit numbers containing 2 and 3, but not adjacent is 252
theorem five_digit_numbers_with_2_and_3_not_adjacent : 
  (number of five-digit numbers containing 2 and 3 but not adjacent) = 252 := 
sorry

-- Question 2: Prove the number of six-digit numbers where digits 1, 2, 3 are in descending order is 100
theorem six_digit_numbers_with_123_descending : 
  (number of six-digit numbers where digits 1, 2, 3 are in descending order) = 100 := 
sorry

end five_digit_numbers_with_2_and_3_not_adjacent_six_digit_numbers_with_123_descending_l827_827471


namespace number_of_possible_lists_l827_827332

/-- 
Define the basic conditions: 
- 18 balls, numbered 1 through 18
- Selection process is repeated 4 times 
- Each selection is independent
- After each selection, the ball is replaced 
- We need to prove the total number of possible lists of four numbers 
--/
def number_of_balls : ℕ := 18
def selections : ℕ := 4

theorem number_of_possible_lists : (number_of_balls ^ selections) = 104976 := by
  sorry

end number_of_possible_lists_l827_827332


namespace probability_relatively_prime_l827_827191

open Nat

def relatively_prime_to (n m : ℕ) := gcd n m = 1

theorem probability_relatively_prime (N : ℕ) (hN : N = 42) : 
  (∃ (count : ℕ), count = (Nat.totient 42)) → (rat.ofInt count / (rat.ofInt 42) = 2 / 7) := 
by
  intro h_totient
  sorry

end probability_relatively_prime_l827_827191


namespace Binkie_gemstones_l827_827837

-- Define the number of gemstones each cat has
variables (F S B : ℕ)

-- Conditions based on the problem statement
axiom Spaatz_has_one : S = 1
axiom Spaatz_equation : S = F / 2 - 2
axiom Binkie_equation : B = 4 * F

-- Theorem statement
theorem Binkie_gemstones : B = 24 :=
by
  -- Proof will be inserted here
  sorry

end Binkie_gemstones_l827_827837


namespace cos_periodicity_l827_827328

theorem cos_periodicity (a : ℝ) (k : ℤ) : cos (a + k * π) = (-1)^k * cos a := 
sorry

end cos_periodicity_l827_827328


namespace john_works_30_hours_per_week_l827_827982

/-- Conditions --/
def hours_per_week_fiona : ℕ := 40
def hours_per_week_jeremy : ℕ := 25
def hourly_wage : ℕ := 20
def monthly_total_payment : ℕ := 7600
def weeks_in_month : ℕ := 4

/-- Derived Definitions --/
def monthly_hours_fiona_jeremy : ℕ :=
  (hours_per_week_fiona + hours_per_week_jeremy) * weeks_in_month

def monthly_payment_fiona_jeremy : ℕ :=
  hourly_wage * monthly_hours_fiona_jeremy

def monthly_payment_john : ℕ :=
  monthly_total_payment - monthly_payment_fiona_jeremy

def hours_per_month_john : ℕ :=
  monthly_payment_john / hourly_wage

def hours_per_week_john : ℕ :=
  hours_per_month_john / weeks_in_month

/-- Theorem stating that John works 30 hours per week --/
theorem john_works_30_hours_per_week :
  hours_per_week_john = 30 := by
  sorry

end john_works_30_hours_per_week_l827_827982


namespace money_distribution_l827_827376

theorem money_distribution (x y z : ℝ) :
    y = 18 →
    x + y + z = 78 →
    z = 0.50 * x →
    y = 0.45 * x :=
by
  intros hy htotal hzx
  nlinarith

end money_distribution_l827_827376


namespace correct_distance_function_l827_827784

noncomputable def f (x : ℝ) : ℝ :=
if h0 : 0 <= x ∧ x <= 1 then x else
if h1 : 1 < x ∧ x <= 2 then real.sqrt (1 + (x - 1)^2) else
if h2 : 2 < x ∧ x <= 3 then real.sqrt (1 + (3 - x)^2) else
if h3 : 3 < x ∧ x <= 4 then 4 - x else 0

theorem correct_distance_function :
  ∀ x, 0 <= x ∧ x <= 4 → f x =
    (if 0 <= x ∧ x <= 1 then x else
    if 1 < x ∧ x <= 2 then real.sqrt (1 + (x - 1)^2) else
    if 2 < x ∧ x <= 3 then real.sqrt (1 + (3 - x)^2) else
    if 3 < x ∧ x <= 4 then 4 - x else 0) :=
begin
  intros x hx,
  unfold f,
  split_ifs; simp; try { apply or.inl hx.right };
  exact rfl,
end

end correct_distance_function_l827_827784


namespace train_average_speed_l827_827378

variable (x : ℝ) (h_pos : x > 0)

def average_speed (d1 d2 t1 t2 : ℝ) : ℝ := 
  (d1 + d2) / (t1 + t2)

theorem train_average_speed 
  (d1 d2 : ℝ) 
  (s1 s2 : ℝ) 
  (t1 t2 : ℝ)
  (h_d1 : d1 = x) 
  (h_d2 : d2 = 2 * x) 
  (h_s1 : s1 = 40) 
  (h_s2 : s2 = 20) 
  (h_t1 : t1 = x / s1) 
  (h_t2 : t2 = d2 / s2) : 
  average_speed d1 d2 t1 t2 = 32 :=
by
  sorry

end train_average_speed_l827_827378


namespace c_left_days_before_completion_l827_827731

-- Definitions for the given conditions
def work_done_by_a_in_one_day := 1 / 30
def work_done_by_b_in_one_day := 1 / 30
def work_done_by_c_in_one_day := 1 / 40
def total_days := 12

-- Proof problem statement (to prove that c left 8 days before the completion)
theorem c_left_days_before_completion :
  ∃ x : ℝ, 
  (12 - x) * (7 / 60) + x * (1 / 15) = 1 → 
  x = 8 := sorry

end c_left_days_before_completion_l827_827731


namespace average_rainfall_feb2020_mawsynram_l827_827555

theorem average_rainfall_feb2020_mawsynram :
  let days := 29
  let hours_per_day := 24
  let total_rainfall := 280
  let total_hours := days * hours_per_day
  (total_rainfall / total_hours) = (280 / 696) :=
by
  let days := 29
  let hours_per_day := 24
  let total_rainfall := 280
  let total_hours := days * hours_per_day
  have h : total_hours = 696, by norm_num
  rw [h]
  norm_num
  sorry

end average_rainfall_feb2020_mawsynram_l827_827555


namespace proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l827_827852

noncomputable def problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : Prop :=
  y ≤ 4.5

noncomputable def problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : Prop :=
  y ≥ -8

noncomputable def problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : Prop :=
  -1 ≤ y ∧ y ≤ 1

noncomputable def problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : Prop :=
  y ≤ 1/3

-- Proving that the properties hold:
theorem proof_of_problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : problem1 x y h :=
  sorry

theorem proof_of_problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : problem2 x y h :=
  sorry

theorem proof_of_problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : problem3 x y h :=
  sorry

theorem proof_of_problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : problem4 x y h :=
  sorry

end proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l827_827852


namespace complex_sum_eq_l827_827543

-- Define the complex numbers
def B := 7 - (2 : ℂ) * complex.I
def N := -3 + (4 : ℂ) * complex.I
def T := -complex.I
def Q := (5 : ℂ)
def R := 1 + (3 : ℂ) * complex.I

-- State the theorem to be proven
theorem complex_sum_eq :
  B - N + T - Q + R = 6 + (4 : ℂ) * complex.I :=
by
  -- This is a placeholder for the proof.
  sorry

end complex_sum_eq_l827_827543


namespace planes_parallel_or_intersect_l827_827509

-- Define what it means for a line to be parallel to a plane
def line_parallel_to_plane (L : set Point) (P : set Point) : Prop :=
  ∀ p1 p2 ∈ L, ∃ q, q ∈ P ∧ p1 ≠ q ∧ p2 ≠ q

-- Define what it means for two planes to be parallel
def planes_parallel (P Q : set Point) : Prop :=
  ∃ v, ∀ p ∈ P, ∀ q ∈ Q, (p - q) = v ∨ (p - q) = -v

-- Define what it means for two planes to intersect
def planes_intersect (P Q : set Point) : Prop :=
  ∃ p, p ∈ P ∧ p ∈ Q

-- The main theorem
theorem planes_parallel_or_intersect (P Q : set Point) 
  (h : ∀ l : set Point, (∀ p ∈ l, ∃ q ∈ P, p ≠ q) → line_parallel_to_plane l Q):
  planes_parallel P Q ∨ planes_intersect P Q :=
sorry

end planes_parallel_or_intersect_l827_827509


namespace graph_c2_l827_827165

variable {α : Type*} [Add α] [One α] [Sub α] [Neg α] [HasRefl α] [HasShift α]
variable (f : α → α) (x : α)

def reflection (f : α → α) : α → α := λ x, f (2 - x)
def shift_one_left (f : α → α) : α → α := λ x, f (x + 1)

theorem graph_c2 {f : α → α} : shift_one_left (reflection f) x = f (1 - x) :=
by
  unfold reflection shift_one_left
  sorry

end graph_c2_l827_827165


namespace triangle_obtuse_l827_827489

variable {a b c : ℝ}

theorem triangle_obtuse (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  ∃ C : ℝ, 0 ≤ C ∧ C ≤ π ∧ Real.cos C = -1/4 ∧ C > Real.pi / 2 :=
by
  sorry

end triangle_obtuse_l827_827489


namespace monotonicity_of_f_when_a_is_three_fourths_f_difference_ge_ln2_add_three_fourths_l827_827936

-- Definition of the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * ln x + 1 / (x - 1)

-- Part (I)
theorem monotonicity_of_f_when_a_is_three_fourths :
  ∀ x : ℝ, x > 0 → x < 1 → (f (3/4) (x) < f (3/4) (x + ε) ↔ 0 < x ∧ x < 1/3 ∨ x > 3) ∧
                           (f (3/4) (x) > f (3/4) (x + ε) ↔ 1/3 < x ∧ x < 1 ∨ 1 < x ∧ x < 3) :=
begin
  sorry
end

-- Part (II)
theorem f_difference_ge_ln2_add_three_fourths {a x1 x2 : ℝ} (ha : a ∈ Icc (1 / 2 : ℝ) 2) :
  x1 ∈ Ioo 0 (1 / 2) → x2 ∈ Ioi 2 → f a x2 - f a x1 ≥ ln 2 + 3 / 4 :=
begin
  sorry
end

end monotonicity_of_f_when_a_is_three_fourths_f_difference_ge_ln2_add_three_fourths_l827_827936


namespace find_fd_closest_to_3_l827_827603

open Real

def parallelogram (A B C D : Point) :=
  (A, B, C) ≠ (C, D, A)

def angle_eq (angle1 angle2 : ℝ) : Prop :=
  angle1 = angle2

def extend (C D E : Point) (distance : ℝ) : Prop :=
  dist C D + distance = dist C E

def similar_triangles (A B C D E F : Point) : Prop :=
  let angle1 := angle A B C
  let angle2 := angle D E F
  let angle3 := angle B A C
  let angle4 := angle E D F
  angle_eq angle1 angle2 ∧ angle_eq angle3 angle4

def find_closest_to_3 (x : ℝ) : ℝ :=
  if abs (x - 3) < abs (x - 1) ∧ abs (x - 3) < abs (x - 2) ∧
     abs (x - 3) < abs (x - 4) ∧ abs (x - 3) < abs (x - 5) then 3 else
  if abs (x - 4) < abs (x - 5) ∧ abs (x - 4) < abs (x - 3) ∧
     abs (x - 4) < abs (x - 2) ∧ abs (x - 4) < abs (x - 1) then 4 else
  if abs (x - 5) < abs (x - 4) ∧ abs (x - 5) < abs (x - 3) ∧
     abs (x - 5) < abs (x - 2) ∧ abs (x - 5) < abs (x - 1) then 5 else
  if abs (x - 2) < abs (x - 4) ∧ abs (x - 2) < abs (x - 3) ∧
     abs (x - 2) < abs (x - 5) ∧ abs (x - 2) < abs (x - 1) then 2 else
  1

theorem find_fd_closest_to_3 (A B C D E F : Point)
  (parallelogram ABCD : parallelogram A B C D)
  (angle_ABC : angle_eq (angle A B C) (120 * π / 180))
  (AB : dist A B = 16)
  (BC : dist B C = 10)
  (extend_CD_E : extend C D E 4)
  (intersection_F : intersect (line_through B E) (line_through A D) = F) :
  find_closest_to_3 (dist F D) = 3 :=
sorry

end find_fd_closest_to_3_l827_827603


namespace find_AB_distance_l827_827994

variables (α θ : ℝ) (x y x' y' ρ : ℝ)
variables (C1_eqn : x = 2 + 2 * Real.cos α ∧ y = 2 * Real.sin α)
variables (C2_eqn : (x ^ 2) / 4 + y ^ 2 = 1)
variables (transformation : x' = (1 / 2) * x ∧ y' = y)
variables (OA OB AB : ℝ)

noncomputable def distance_AB : Prop :=
  let A := 4 * Real.cos (π / 3)
  let B := 1
  (C1_eqn : Prop) ∧
  (C2_eqn : Prop) ∧
  (transformation : Prop) → (abs (A - B) = 1)

theorem find_AB_distance : distance_AB := by
  sorry

end find_AB_distance_l827_827994


namespace arc_length_l827_827577

theorem arc_length (O Q I S : Point) (angle_QIS : ℝ) (OQ : ℝ) :
  angle_QIS = 45 ∧ OQ = 15 → arc_length O Q S = 7.5 * real.pi :=
by
  sorry

end arc_length_l827_827577


namespace sum_primes_10_to_20_l827_827298

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827298


namespace parsnip_box_fullness_l827_827397

theorem parsnip_box_fullness (capacity : ℕ) (fraction_full : ℚ) (avg_boxes : ℕ) (avg_parsnips : ℕ) :
  capacity = 20 →
  fraction_full = 3 / 4 →
  avg_boxes = 20 →
  avg_parsnips = 350 →
  ∃ (full_boxes : ℕ) (non_full_boxes : ℕ) (parsnips_in_full_boxes : ℕ) (parsnips_in_non_full_boxes : ℕ)
    (avg_fullness_non_full_boxes : ℕ),
    full_boxes = fraction_full * avg_boxes ∧
    non_full_boxes = avg_boxes - full_boxes ∧
    parsnips_in_full_boxes = full_boxes * capacity ∧
    parsnips_in_non_full_boxes = avg_parsnips - parsnips_in_full_boxes ∧
    avg_fullness_non_full_boxes = parsnips_in_non_full_boxes / non_full_boxes ∧
    avg_fullness_non_full_boxes = 10 :=
by
  sorry

end parsnip_box_fullness_l827_827397


namespace shortest_path_bridge_minimal_distance_bridge_l827_827372

noncomputable def optimal_bridge_point
  (y_axis : ℝ) -- river flows along y-axis, assume y_axis = 0 for simplicity
  (h_A h_B h : ℝ) : ℝ × ℝ :=
let θ := real.arctan (h / (h_A + h_B)) in
  (h / real.cos θ, (h_A + h_B) / real.sin θ)

theorem shortest_path_bridge
  (y_axis : ℝ := 0) -- river flows along y-axis
  (h_A h_B h : ℝ) : ℝ :=
let C := optimal_bridge_point y_axis h_A h_B h in
C.2  -- vertical coordinate of the intersection point (C_y)

-- state that the bridge at point C results in the shortest path
theorem minimal_distance_bridge 
  (y_axis : ℝ := 0) -- river flows along y-axis
  (h_A h_B h : ℝ) 
  (C : ℝ × ℝ) -- C is the point where the line intersects the river bank
  (θ := real.arctan (h / (h_A + h_B))) -- angle for optimal path
  [bridge_at_C : C = optimal_bridge_point y_axis h_A h_B h] : 
  ∀ (P: ℝ × ℝ), P = C :=
sorry

end shortest_path_bridge_minimal_distance_bridge_l827_827372


namespace sum_of_primes_between_10_and_20_l827_827257

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827257


namespace contrapositive_l827_827310

variables (p q : Prop)

theorem contrapositive (hpq : p → q) : ¬ q → ¬ p :=
by sorry

end contrapositive_l827_827310


namespace right_triangle_area_l827_827563

theorem right_triangle_area (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : (1 / 2 : ℚ) * (a * b) = 864 := 
by 
  sorry

end right_triangle_area_l827_827563


namespace area_of_B_l827_827423

noncomputable def regionB : set ℂ :=
  {z | let x := z.re, y := z.im in 
       (0 ≤ x ∧ x ≤ 30) ∧ (0 ≤ y ∧ y ≤ 30) ∧
       (0 ≤ 30 * x / (x^2 + y^2) ∧ 30 * x / (x^2 + y^2) ≤ 1) ∧
       (0 ≤ 30 * y / (x^2 + y^2) ∧ 30 * y / (x^2 + y^2) ≤ 1) }
       
theorem area_of_B : sorry

end area_of_B_l827_827423


namespace inequality_geometric_mean_sum_comparison_l827_827745

-- Part 1
theorem inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : b / a < (b + m) / (a + m) :=
sorry

-- Part 2.i
theorem geometric_mean (a : ℕ → ℝ) (h1 : ∀ n, (∏ i in finset.range n, a i) ^ (1 / n) = 3 ^ ((n - 1) / 2))
    (h2 : a 0 = 1) : ∀ n, a n = 3 ^ (n - 1) :=
sorry

-- Part 2.ii
theorem sum_comparison (a : ℕ → ℝ) (b : ℕ → ℝ)
    (h1 : ∀ n, (∏ i in finset.range n, a i) ^ (1 / n) = 3 ^ ((n - 1) / 2))
    (h2 : a 0 = 1)
    (h3 : ∀ n, b n = (n + 1) * a n - 1) :
    ∀ n, (∑ i in finset.range n, i / b i) < 3 / 2 :=
sorry

end inequality_geometric_mean_sum_comparison_l827_827745


namespace mans_speed_against_current_l827_827783

variable (V_downstream V_current : ℝ)
variable (V_downstream_eq : V_downstream = 15)
variable (V_current_eq : V_current = 2.5)

theorem mans_speed_against_current : V_downstream - 2 * V_current = 10 :=
by
  rw [V_downstream_eq, V_current_eq]
  exact (15 - 2 * 2.5)

end mans_speed_against_current_l827_827783


namespace vector_perpendicular_lambda_l827_827531

theorem vector_perpendicular_lambda (λ : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-1, 3)
  (a.1 * (a.1 + λ * b.1) + a.2 * (a.2 - λ * b.2) = 0) ↔ λ = 5 := 
by 
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-1, 3)
  have h : a.1 * (a.1 + λ * b.1) + a.2 * (a.2 - λ * b.2) = 0 := 
    by sorry
  sorry

end vector_perpendicular_lambda_l827_827531


namespace rock_paper_scissors_probability_l827_827806

theorem rock_paper_scissors_probability :
  let outcomes := {("Alice", "Bob"), ("Alice", "Conway"), ("Bob", "Alice"), ("Bob", "Conway"), ("Conway", "Alice"), ("Conway", "Bob") : (String × String)}
  n : ℕ := outcomes.size
  f : Finset (Finset (String × String)) := {s ∈ outcomes.powerset | s.size = 3 ∧ ("Alice", "Bob") ∈ s ∧ ("Bob", "Conway") ∈ s ∧ ("Conway", "Alice") ∈ s}
  total_outcomes : ℕ := 8
  favorable : ℕ := f.size
  probability : ℚ := favorable / total_outcomes
  probability = 1 / 4 :=
by
  sorry

end rock_paper_scissors_probability_l827_827806


namespace relationship_between_abc_l827_827109

noncomputable def a := (4 / 5) ^ (1 / 2)
noncomputable def b := (5 / 4) ^ (1 / 5)
noncomputable def c := (3 / 4) ^ (3 / 4)

theorem relationship_between_abc : c < a ∧ a < b := by
  sorry

end relationship_between_abc_l827_827109


namespace choose_marbles_l827_827596

theorem choose_marbles : 
  let marbles := 15
  let specific_colors := 4
  let remaining_marbles := marbles - specific_colors
  let ways_to_choose_specific := specific_colors
  let ways_to_choose_remaining := Nat.choose remaining_marbles 4
  in ways_to_choose_specific * ways_to_choose_remaining = 1320 :=
by
  sorry

end choose_marbles_l827_827596


namespace find_m_for_one_real_solution_l827_827466

variables {m x : ℝ}

-- Given condition
def equation := (x + 4) * (x + 1) = m + 2 * x

-- The statement to prove
theorem find_m_for_one_real_solution : (∃ m : ℝ, m = 7 / 4 ∧ ∀ (x : ℝ), (x + 4) * (x + 1) = m + 2 * x) :=
by
  -- The proof starts here, which we will skip with sorry
  sorry

end find_m_for_one_real_solution_l827_827466


namespace calculate_T1_T2_l827_827653

def triangle (a b c : ℤ) : ℤ := a + b - 2 * c

def T1 := triangle 3 4 5
def T2 := triangle 6 8 2

theorem calculate_T1_T2 : 2 * T1 + 3 * T2 = 24 :=
  by
    sorry

end calculate_T1_T2_l827_827653


namespace boat_speed_in_still_water_l827_827569

variable (B S : ℝ)

theorem boat_speed_in_still_water :
  (B + S = 38) ∧ (B - S = 16) → B = 27 :=
by
  sorry

end boat_speed_in_still_water_l827_827569


namespace triangle_AD_relation_l827_827990

theorem triangle_AD_relation (A B C D: Point)
  (h1 : ∠BAC = 90) 
  (h2 : D ∈ LineSegment B C)
  (h3 : ∠BDA = 2 * ∠BAD) :
  1 / (dist A D) = 1 / 2 * (1 / (dist B D) + 1 / (dist C D)) := 
by
  sorry

end triangle_AD_relation_l827_827990


namespace uneaten_chips_l827_827431

theorem uneaten_chips :
  ∀ (chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips : ℕ),
    (chips_per_cookie = 7) →
    (cookies_total = 12 * 4) →
    (half_cookies = cookies_total / 2) →
    (uneaten_cookies = cookies_total - half_cookies) →
    (uneaten_chips = uneaten_cookies * chips_per_cookie) →
    uneaten_chips = 168 :=
by
  intros chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips
  intros chips_per_cookie_eq cookies_total_eq half_cookies_eq uneaten_cookies_eq uneaten_chips_eq
  rw [chips_per_cookie_eq, cookies_total_eq, half_cookies_eq, uneaten_cookies_eq, uneaten_chips_eq]
  norm_num
  sorry

end uneaten_chips_l827_827431


namespace sum_primes_10_to_20_l827_827295

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827295


namespace election_valid_votes_other_candidate_l827_827734

theorem election_valid_votes_other_candidate 
  (total_votes : ℕ) 
  (invalid_percentage : ℝ) 
  (candidate1_percentage : ℝ)
  (total_votes = 7000) 
  (invalid_percentage = 0.20)
  (candidate1_percentage = 0.55) : 
  ∃ (valid_votes : ℕ), valid_votes = 2520 := 
by
  sorry

end election_valid_votes_other_candidate_l827_827734


namespace relationship_l827_827893

-- Given conditions
def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

-- The theorem to be proven
theorem relationship : c > b ∧ b > a := by
  sorry

end relationship_l827_827893


namespace top_square_is_9_l827_827334

def initial_grid : List (List ℕ) := 
  [[1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]]

def fold_step_1 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col3 := grid.map (fun row => row.get! 2)
  let col2 := grid.map (fun row => row.get! 1)
  [[col1.get! 0, col3.get! 0, col2.get! 0],
   [col1.get! 1, col3.get! 1, col2.get! 1],
   [col1.get! 2, col3.get! 2, col2.get! 2]]

def fold_step_2 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col2 := grid.map (fun row => row.get! 1)
  let col3 := grid.map (fun row => row.get! 2)
  [[col2.get! 0, col1.get! 0, col3.get! 0],
   [col2.get! 1, col1.get! 1, col3.get! 1],
   [col2.get! 2, col1.get! 2, col3.get! 2]]

def fold_step_3 (grid : List (List ℕ)) : List (List ℕ) :=
  let row1 := grid.get! 0
  let row2 := grid.get! 1
  let row3 := grid.get! 2
  [row3, row2, row1]

def folded_grid : List (List ℕ) :=
  fold_step_3 (fold_step_2 (fold_step_1 initial_grid))

theorem top_square_is_9 : folded_grid.get! 0 = [9, 7, 8] :=
  sorry

end top_square_is_9_l827_827334


namespace negation_of_p_l827_827124

-- Define the proposition p
def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 > 0

-- State the theorem: the negation of proposition p
theorem negation_of_p : ¬ proposition_p ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by 
  sorry

end negation_of_p_l827_827124


namespace inequality_proof_l827_827888

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  sorry

end inequality_proof_l827_827888


namespace julia_monday_kids_l827_827599

theorem julia_monday_kids :
  ∃ M : ℕ, (15 + 2 + M = 34) ∧ (M = 17) :=
by
  use 17
  split
  · norm_num
  · rfl

end julia_monday_kids_l827_827599


namespace find_A_l827_827496

variable {A B C : ℚ}

theorem find_A (h1 : A = 1/2 * B) (h2 : B = 3/4 * C) (h3 : A + C = 55) : A = 15 :=
by
  sorry

end find_A_l827_827496


namespace smallest_sum_ratios_l827_827796

theorem smallest_sum_ratios
  (A : ℝ)
  (red_rectangles blue_rectangles : list (ℝ × ℝ))
  (h_area_eq : (∑ r in red_rectangles, r.1 * r.2) = (∑ b in blue_rectangles, b.1 * b.2))
  (h_area_sum: (∑ r in red_rectangles, r.1 * r.2) = A^2 / 2)
  : (∑ b in blue_rectangles, b.2 / b.1) + (∑ r in red_rectangles, r.1 / r.2) ≥ 5 / 2 :=
sorry

end smallest_sum_ratios_l827_827796


namespace circle_center_coordinates_l827_827660

theorem circle_center_coordinates :
  ∀ (x y : ℝ), x^2 + y^2 - 10 * x + 6 * y + 25 = 0 → (5, -3) = ((-(-10) / 2), (-6 / 2)) :=
by
  intros x y h
  have H : (5, -3) = ((-(-10) / 2), (-6 / 2)) := sorry
  exact H

end circle_center_coordinates_l827_827660


namespace calculate_expr_l827_827403

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l827_827403


namespace artist_paints_29_square_meters_l827_827387

theorem artist_paints_29_square_meters :
  ∀ (cubes : ℕ) (edge : ℝ) (layer1 : ℕ) (layer2 : ℕ) (layer3 : ℕ) (layer4 : ℕ),
    cubes = 14 ∧ edge = 1 ∧ layer1 = 1 ∧ layer2 = 3 ∧ layer3 = 4 ∧ layer4 = 6 →
    let top_faces := layer1 + layer2 + layer3 + layer4 in
    let side_faces := 5 + 6 + 4 + 0 in
    let total_painted := top_faces + side_faces in
    total_painted = 29 :=
begin
  intros cubes edge layer1 layer2 layer3 layer4 h,
  sorry
end

end artist_paints_29_square_meters_l827_827387


namespace gymnast_score_difference_l827_827461

theorem gymnast_score_difference 
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x2 + x3 + x4 + x5 = 36)
  (h2 : x1 + x2 + x3 + x4 = 36.8) :
  x1 - x5 = 0.8 :=
by sorry

end gymnast_score_difference_l827_827461


namespace area_of_rectangle_l827_827999

-- Definitions of the conditions
def isRectangle (A B C D : Point) : Prop := sorry
def onSegment (E C D : Point) : Prop := sorry
def length (P Q : Point) : ℝ := sorry
def area (P Q R : Point) : ℝ := sorry
def eq_ratio (CE DE : ℝ) : Prop := CE = 2 * DE

-- Variables to represent points
variables (A B C D E : Point)

-- Given conditions
variables (hRectangle : isRectangle A B C D)
          (hEonCD : onSegment E C D)
          (hRatio : eq_ratio (length C E) (length D E))
          (hAreaBCE : area B C E = 10)

-- Proof goal
theorem area_of_rectangle : area A B C D = 30 :=
sorry

end area_of_rectangle_l827_827999


namespace company_a_percentage_l827_827824

theorem company_a_percentage (total_profits: ℝ) (p_b: ℝ) (profit_b: ℝ) (profit_a: ℝ) :
  p_b = 0.40 →
  profit_b = 60000 →
  profit_a = 90000 →
  total_profits = profit_b / p_b →
  (profit_a / total_profits) * 100 = 60 :=
by
  intros h_pb h_profit_b h_profit_a h_total_profits
  sorry

end company_a_percentage_l827_827824


namespace trajectory_is_line_segment_l827_827022

open Real

-- Define the fixed points
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Condition: Moving point P satisfies |PF1| + |PF2| = 8
def condition (P : ℝ × ℝ) : Prop :=
  dist P F1 + dist P F2 = 8

-- The trajectory of P is the line segment between F1 and F2
theorem trajectory_is_line_segment (P : ℝ × ℝ) 
  (h : condition P) : 
  P.1 ^ 2 + P.2 ^ 2 <= 16∧
  abs(P.1) <= 4 :=
sorry

end trajectory_is_line_segment_l827_827022


namespace ant_maximum_journey_l827_827386

noncomputable theory

def edge_length : ℝ := 1
def face_diagonal_length : ℝ := real.sqrt 2
def max_travel_distance : ℝ := 6 * real.sqrt 2 + 2

theorem ant_maximum_journey :
  ∃ path : list ℝ, path.sum = max_travel_distance ∧
  ∀ i, (path.nth i = some edge_length ∨ path.nth i = some face_diagonal_length) ∧
  (∀ j k, j ≠ k → path.nth j ≠ path.nth k) :=
sorry

end ant_maximum_journey_l827_827386


namespace area_of_rectangle_l827_827997

-- Define the necessary conditions and the area of the rectangle
def rectangleABCD (length width : ℝ) : Prop :=
  2 * (length + width) = 160 ∧ length = 4 * width

-- Define the actual theorem statement
theorem area_of_rectangle : ∃ (length width : ℝ), rectangleABCD length width ∧ (4 * width^2 = 1024) :=
by
  -- Provide the existence statement directly.
  use 64, 16  -- Here, length = 64 and width = 16 as derived from the solution.
  unfold rectangleABCD
  -- Show that these values satisfy the conditions provided.
  split
  -- Proof of the perimeter condition
  simp
  -- Proof of the area condition
  simp
  sorry

end area_of_rectangle_l827_827997


namespace Binkie_gemstones_l827_827839

-- Define the number of gemstones each cat has
variables (F S B : ℕ)

-- Conditions based on the problem statement
axiom Spaatz_has_one : S = 1
axiom Spaatz_equation : S = F / 2 - 2
axiom Binkie_equation : B = 4 * F

-- Theorem statement
theorem Binkie_gemstones : B = 24 :=
by
  -- Proof will be inserted here
  sorry

end Binkie_gemstones_l827_827839


namespace angle_CFB_is_60_degrees_l827_827366

-- Definitions of the points and quadrilateral
def is_rectangle (ABCD : Type) := 
  ∃ A B C D : ℝ × ℝ,
    (A.1 = B.1) ∧ (C.1 = D.1) ∧ (B.2 = C.2) ∧ (A.2 = D.2) ∧
    (B.1 - A.1 = C.1 - D.1) ∧ (B.2 - A.2 = C.2 - D.2) ∧
    (B.1 - A.1 = D.1 - C.1) ∧ (B.2 - A.2 = D.2 - C.2) ∧
    (C.2 = D.2 + B.1 - C.1)

-- definition of quadrilateral ABCD
def quadrilateral_ABCD (A B C D : ℝ × ℝ) (ABCD : Type) : Prop := 
  ∃ (A B C D : ℝ × ℝ), 
    AB = CD ∧ 
    A.1 = B.1 ∧ 
    A.2 = B.2 ∧ 
    B = C ∧
    B.2 = C.1

-- definition of equilateral triangle BCF
def equilateral_triangle (B C F : ℝ × ℝ) : Prop := 
  (dist B C = dist C F) ∧ (dist C F = dist F B)

-- question to prove
theorem angle_CFB_is_60_degrees (A B C D F : ℝ × ℝ) : 
  quadrilateral_ABCD A B C D ABCD →
  equilateral_triangle B C F →
  (mangle CFB = 60) :=
by
  sorry

end angle_CFB_is_60_degrees_l827_827366


namespace arithmetic_sequence_e_value_l827_827574

/-- Definition of arithmetic sequence terms --/
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + n * d

/-- The problem statement --/
theorem arithmetic_sequence_e_value :
  ∃ a d : ℝ, a = 11 ∧ 
  (∃ z : ℝ, z = 39 ∧ 
  (∃ n1 n2 n3 n4 : ℕ, 
  arithmetic_seq a d 1 = 11 ∧ 
  arithmetic_seq a d 5 = 39 ∧ 
  let e := arithmetic_seq a d 2 in 
  e = 22.2)) :=
begin
  use 11, -- initial term
  use (39 - 11) / 5, -- common difference
  split,
  { refl },
  use 39, -- final term
  split,
  { refl },
  existsi 1, -- dummy variables to represent the term positions
  existsi 2,
  existsi 3,
  existsi 4,
  simp [arithmetic_seq],
  exact rfl,
  simp [arithmetic_seq, e],
  sorry, -- Proof goes here
end

end arithmetic_sequence_e_value_l827_827574


namespace smallest_b_1111_is_square_l827_827459

theorem smallest_b_1111_is_square : 
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, (b^3 + b^2 + b + 1 = n^2 → b = 7)) :=
by
  sorry

end smallest_b_1111_is_square_l827_827459


namespace sum_primes_between_10_and_20_l827_827246

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827246


namespace min_distance_hyperbola_circle_l827_827014

theorem min_distance_hyperbola_circle (a b c d : ℝ) (hab : a * b = 1) (hcd : c^2 + d^2 = 1) :
  ∃ (x : ℝ), x = 3 - 2 * real.sqrt 2 ∧ (a - c)^2 + (b - d)^2 = x :=
sorry

end min_distance_hyperbola_circle_l827_827014


namespace solve_for_x_l827_827649

theorem solve_for_x (x : ℤ) (h : 3^(x - 2) = 9^(x + 1)) : x = -4 :=
by
  sorry

end solve_for_x_l827_827649


namespace vertex_C_moves_uniformly_l827_827637

-- Condition: Points A and B move uniformly with equal angular velocities
def uniform_angular_motion (O1 O2 : ℂ) (r1 r2 : ℝ) (t1 t2 : ℝ) (A B : ℂ) := 
  ∃ ω : ℝ, 
    (∀ t : ℝ, A = O1 + r1 * exp (complex.i * (t1 + ω * t))) ∧ 
    (∀ t : ℝ, B = O2 + r2 * exp (complex.i (t2 + ω * t)))

-- Theorem to prove
theorem vertex_C_moves_uniformly {O1 O2 A B : ℂ} {r1 r2 t1 t2 : ℝ} 
  (h : uniform_angular_motion O1 O2 r1 r2 t1 t2 A B) : 
  ∃ (μ η : ℂ), 
    (∀ t : ℝ, 
      let C := some_point (A, B, t) in -- some_point form the equilateral triangle ABC
      C = μ + η * exp (complex.i * ω * t)) :=
sorry

end vertex_C_moves_uniformly_l827_827637


namespace binkie_gemstones_l827_827836

variable (Binkie Frankie Spaatz : ℕ)

-- Define the given conditions
def condition1 : Binkie = 4 * Frankie := by sorry
def condition2 : Spaatz = (1 / 2) * Frankie - 2 := by sorry
def condition3 : Spaatz = 1 := by sorry

-- State the theorem to be proved
theorem binkie_gemstones : Binkie = 24 := by
  have h_Frankie : Frankie = 6 := by
    sorry
  rw [←condition3, ←condition2] at h_Frankie
  have h_Binkie : Binkie = 4 * 6 := by
    rw [condition1]
    sorry
  rw [h_Binkie]
  exact
    show 4 * 6 = 24 from rfl

end binkie_gemstones_l827_827836


namespace new_basis_A_new_basis_C_l827_827544

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (a b c : V)

def form_basis (s : set V) : Prop :=
linear_independent ℝ (λ x, x : s → V) ∧ submodule.span ℝ s = ⊤

axiom ha_basis : form_basis {a, b, c}

theorem new_basis_A :
  form_basis {a + b, a + c, a} :=
sorry

theorem new_basis_C :
  form_basis {a - b + c, a - b, a + c} :=
sorry

end new_basis_A_new_basis_C_l827_827544


namespace quotient_is_12_l827_827684

theorem quotient_is_12 (a b q : ℕ) (h1: q = a / b) (h2: q = a / 2) (h3: q = 6 * b) : q = 12 :=
by 
  sorry

end quotient_is_12_l827_827684


namespace students_playing_both_l827_827394

theorem students_playing_both
    (total_students baseball_team hockey_team : ℕ)
    (h1 : total_students = 36)
    (h2 : baseball_team = 25)
    (h3 : hockey_team = 19)
    (h4 : total_students = baseball_team + hockey_team - students_both) :
    students_both = 8 := by
  sorry

end students_playing_both_l827_827394


namespace intersection_complement_l827_827015

open Set

variable {α : Type*}
noncomputable def A : Set ℝ := {x | x^2 ≥ 1}
noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_complement :
  A ∩ (compl B) = (Iic (-1)) ∪ (Ioi 2) := by
sorry

end intersection_complement_l827_827015


namespace A_more_likely_than_B_l827_827803

-- Define the conditions
variables (n : ℕ) (k : ℕ)
-- n is the total number of programs, k is the chosen number of programs
def total_programs : ℕ := 10
def selected_programs : ℕ := 3
-- Probability of person B correctly completing each program
def probability_B_correct : ℚ := 3/5
-- Person A can correctly complete 6 out of 10 programs
def person_A_correct : ℕ := 6

-- The probability of person B successfully completing the challenge
def probability_B_success : ℚ := (3 * (9/25) * (2/5)) + (27/125)

-- Define binomial coefficient function for easier combination calculations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probabilities for the number of correct programs for person A
def P_X_0 : ℚ := (choose 4 3 : ℕ) / (choose 10 3 : ℕ)
def P_X_1 : ℚ := (choose 6 1 * choose 4 2 : ℕ) / (choose 10 3 : ℕ)
def P_X_2 : ℚ := (choose 6 2 * choose 4 1 : ℕ) / (choose 10 3 : ℕ)
def P_X_3 : ℚ := (choose 6 3 : ℕ) / (choose 10 3 : ℕ)

-- The distribution and expectation of X for person A
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

-- The probability of person A successfully completing the challenge
def P_A_success : ℚ := P_X_2 + P_X_3

-- Final comparisons to determine who is more likely to succeed
def compare_success : Prop := P_A_success > probability_B_success

-- Lean statement
theorem A_more_likely_than_B : compare_success := by
  sorry

end A_more_likely_than_B_l827_827803


namespace angle_bisectors_intersect_at_one_point_l827_827991

theorem angle_bisectors_intersect_at_one_point
  (A B C D E : Type)
  (vertices_ABCDE : Set (A))
  (incenter_ABCDE : ∃ (I : Type), 
    ∀ (angle : Type), 
    (angle = ∠ DAC ∨ angle = ∠ EBD ∨ angle = ∠ ACE ∨ angle = ∠ BDA ∨ angle = ∠ CEB) → 
    bisects angle I)
  (cyclic_ABCDE : cyclic_polygon A B C D E)
  (T P Q R S : Type)
  (vertices_TPQRS : Set (T))
  (cyclic_TPQRS : cyclic_polygon T P Q R S) :
  ∃ (J : Type), 
    (bisects (∠ TPQ) J ∧ 
    bisects (∠ PQR) J ∧ 
    bisects (∠ QRS) J ∧ 
    bisects (∠ RST) J ∧ 
    bisects (∠ STP) J) :=
begin
  sorry
end

end angle_bisectors_intersect_at_one_point_l827_827991


namespace chromium_percentage_alloy_l827_827992

theorem chromium_percentage_alloy 
  (w1 w2 w3 w4 : ℝ)
  (p1 p2 p3 p4 : ℝ)
  (h_w1 : w1 = 15)
  (h_w2 : w2 = 30)
  (h_w3 : w3 = 10)
  (h_w4 : w4 = 5)
  (h_p1 : p1 = 0.12)
  (h_p2 : p2 = 0.08)
  (h_p3 : p3 = 0.15)
  (h_p4 : p4 = 0.20) :
  (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4) / (w1 + w2 + w3 + w4) * 100 = 11.17 := 
  sorry

end chromium_percentage_alloy_l827_827992


namespace derek_saves_money_l827_827843

theorem derek_saves_money : 
  let a : ℕ := 2
  let r : ℕ := 2
  let n : ℕ := 12
  S_n a r n = 8190
  where S_n (a : ℕ) (r : ℕ) (n : ℕ) : ℕ := a * (1 - r^n) / (1 - r) :=
  sorry

end derek_saves_money_l827_827843


namespace solve_for_x_l827_827550

theorem solve_for_x (x : ℝ) (hp : 0 < x) (h : 4 * x^2 = 1024) : x = 16 :=
sorry

end solve_for_x_l827_827550


namespace value_of_expression_l827_827309

theorem value_of_expression : ((25 + 8)^2 - (8^2 + 25^2) = 400) :=
by 
  sorry

end value_of_expression_l827_827309


namespace area_of_triangle_PF1F2_l827_827012

noncomputable theory

open Real

theorem area_of_triangle_PF1F2 :
  ∀ (x y : ℝ) (F1 F2 : ℝ × ℝ),
  (x^2 - y^2 / 8 = 1) →
  (P : F1 = (-3, 0)) →
  (Q : F2 = (3, 0)) →
  (P_y: ∀ y, y = 3) →
  let |P_F1 := dist (x,y) F1
  let |P_F2 := dist (x,y) F2
  (h_r: |P_F1| / |P_F2| = 3 / 4) →
  ∃ (a : ℝ), a = 8 * sqrt 5 :=
sorry

end area_of_triangle_PF1F2_l827_827012


namespace odd_indexed_convergents_same_for_3_and_4_l827_827832

-- Definitions used in Lean 4 Statement
noncomputable def continued_fraction (x: ℝ) : Stream ℚ := sorry

def convergent (cf: Stream ℚ) (n: ℕ) : ℚ := sorry

def sqrt_continued_fraction (c: ℕ) : Stream ℚ :=
  continued_fraction (Real.sqrt (c^2 + c))

-- Assertions based on given conditions
axiom nat_c : c ∈ ℕ

-- Lean 4 statement of the mathematically equivalent problem
theorem odd_indexed_convergents_same_for_3_and_4 :
  (∀ n, n % 2 = 1 → convergent (sqrt_continued_fraction 3) n = convergent (sqrt_continued_fraction 4) n) :=
sorry

end odd_indexed_convergents_same_for_3_and_4_l827_827832


namespace sum_of_digits_l827_827499

theorem sum_of_digits (a b c d : ℕ) (h_diff : ∀ x y : ℕ, (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) (h1 : a + c = 10) (h2 : b + c = 8) (h3 : a + d = 11) : 
  a + b + c + d = 18 :=
by
  sorry

end sum_of_digits_l827_827499


namespace area_of_shape_l827_827708

-- Define the radius of the circle and the central angles of the sectors
def radius : ℝ := 12
def angle1 : ℝ := 90
def angle2 : ℝ := 120

-- Calculate the area of the full circle
def circle_area (r : ℝ) : ℝ := π * r^2

-- Calculate the fraction of the full circle that each sector represents
def sector_fraction (angle : ℝ) : ℝ := angle / 360

-- Define the areas of each sector
def sector_area (r : ℝ) (angle : ℝ) : ℝ := (sector_fraction angle) * (circle_area r)

-- Define the total area of the shape formed by these two sectors
def total_area (r : ℝ) (angle1 angle2 : ℝ) : ℝ := (sector_area r angle1) + (sector_area r angle2)

-- Theorem to prove the area of the shape formed by the two sectors is 84π
theorem area_of_shape : total_area radius angle1 angle2 = 84 * π :=
by
  -- Provide initial steps if necessary or directly conclude with sorry if proof isn't required
  sorry

end area_of_shape_l827_827708


namespace cos_alpha_minus_pi_six_l827_827477

theorem cos_alpha_minus_pi_six (α : ℝ) (h : Real.sin (α + Real.pi / 3) = 4 / 5) : 
  Real.cos (α - Real.pi / 6) = 4 / 5 :=
sorry

end cos_alpha_minus_pi_six_l827_827477


namespace problem_solution_l827_827338

noncomputable def number_of_balls_in_the_box : ℕ :=
let x := 124 in x

theorem problem_solution : ∃ x: ℕ, (x - 92 = 156 - x) ∧ x = 124 :=
begin
  use number_of_balls_in_the_box,
  split,
  { -- Prove condition
    calc
      number_of_balls_in_the_box - 92 = 124 - 92 : by refl
      ... = 32 : by norm_num
      ... = 156 - 124 : by norm_num
      ... = 156 - number_of_balls_in_the_box : by refl },
  { -- Prove that x = 124
    refl }
end

end problem_solution_l827_827338


namespace distance_between_parallel_lines_l827_827907

theorem distance_between_parallel_lines
  (l1 : ∀ (x y : ℝ), 2*x + y + 1 = 0)
  (l2 : ∀ (x y : ℝ), 4*x + 2*y - 1 = 0) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 5 / 10 := by
  sorry

end distance_between_parallel_lines_l827_827907


namespace exists_special_2020_digit_number_l827_827444

theorem exists_special_2020_digit_number :
  ∃ (N : ℕ), (N.digits 10).length = 2020 ∧
             (∀ d ∈ N.digits 10, d ≠ 0 ∧ d ≠ 1) ∧
             N % ((N.digits 10).sum) = 0 := 
sorry

end exists_special_2020_digit_number_l827_827444


namespace shares_difference_l827_827385

theorem shares_difference (x : ℝ) (h_ratio : 2.5 * x + 3.5 * x + 7.5 * x + 9.8 * x = (23.3 * x))
  (h_difference : 7.5 * x - 3.5 * x = 4500) : 9.8 * x - 2.5 * x = 8212.5 :=
by
  sorry

end shares_difference_l827_827385


namespace find_sum_of_x_and_y_l827_827172

theorem find_sum_of_x_and_y : 
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (∃ k : ℕ, 450 * x = k^2) ∧ (∃ m : ℕ, 450 * y = m^3) ∧ x + y = 62 :=
by
  -- conditions
  have h1 : 450 = 2 * 3^2 * 5^2 := by norm_num,
  
  -- solving for x
  have x1 := 2,

  -- solving for y
  have y1 := 60,

  -- sum of x and y
  have sum := x1 + y1,
  exact ⟨x1, y1, by norm_num, by norm_num, ⟨30, by norm_num⟩, ⟨30, by norm_num⟩, by norm_num⟩,

  -- result
  sorry

end find_sum_of_x_and_y_l827_827172


namespace sum_of_primes_between_10_and_20_l827_827263

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827263


namespace sum_of_x_y_l827_827961

theorem sum_of_x_y (m x y : ℝ) (h₁ : x + m = 4) (h₂ : y - 3 = m) : x + y = 7 :=
sorry

end sum_of_x_y_l827_827961


namespace function_properties_l827_827006

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 3 + (a - 1) * x ^ 2 + 27 * (a - 2) * x + b

-- Statement of the proof problem
theorem function_properties (a b : ℝ):
  (∀ x : ℝ, f a b (-x) = -f a b x) → -- symmetric about origin implies odd function
  a = 1 ∧ b = 0 ∧
  (∀ x ∈ Icc (-3 : ℝ) 3, deriv (f 1 0) x < 0) ∧
  ((∀ x ∈ Icc (-4 : ℝ) (-3), deriv (f 1 0) x > 0) ∧ (∀ x ∈ Icc (3 : ℝ) 5, deriv (f 1 0) x > 0)) ∧
  (∃ max_x ∈ Icc (-4 : ℝ) 5, f 1 0 max_x = 54) ∧
  (∃ min_x ∈ Icc (-4 : ℝ) 5, f 1 0 min_x = -54) :=
by
  sorry

end function_properties_l827_827006


namespace sum_primes_between_10_and_20_l827_827242

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827242


namespace sum_of_primes_between_10_and_20_l827_827254

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827254


namespace sum_primes_between_10_and_20_l827_827243

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827243


namespace sum_of_consecutive_integers_l827_827044

theorem sum_of_consecutive_integers (x y : ℤ) (h1 : x < ↑(Real.sqrt 5) + 1) (h2 : ↑(Real.sqrt 5) + 1 < y) (h3 : y = x + 1) : x + y = 7 := by
  sorry -- Proof can be filled in later

end sum_of_consecutive_integers_l827_827044


namespace solution_for_correct_statement_l827_827726

theorem solution_for_correct_statement (x : ℝ) (h : sqrt x = x) : x = 0 ∨ x = 1 := by
  sorry

end solution_for_correct_statement_l827_827726


namespace problem_statement_l827_827538

-- Definitions of conditions
def is_multiple_of_6 (n : ℕ) := n % 6 = 0
def ends_in_4 (n : ℕ) := n % 10 = 4
def less_than_600 (n : ℕ) := n < 600

-- Theorems/Statements to prove
theorem problem_statement : 
  {n : ℕ | is_multiple_of_6 n ∧ ends_in_4 n ∧ less_than_600 n}.to_finset.card = 10 :=
by {
  sorry
}

end problem_statement_l827_827538


namespace triangle_properties_l827_827500

theorem triangle_properties (b c : ℝ) (C : ℝ)
  (hb : b = 10)
  (hc : c = 5 * Real.sqrt 6)
  (hC : C = Real.pi / 3) :
  let R := c / (2 * Real.sin C)
  let B := Real.arcsin (b * Real.sin C / c)
  R = 5 * Real.sqrt 2 ∧ B = Real.pi / 4 :=
by
  sorry

end triangle_properties_l827_827500


namespace relationship_l827_827889

-- Given conditions
def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

-- The theorem to be proven
theorem relationship : c > b ∧ b > a := by
  sorry

end relationship_l827_827889


namespace inflection_point_on_3x_l827_827004

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4 * Real.sin x - Real.cos x
noncomputable def f' (x : ℝ) : ℝ := 3 + 4 * Real.cos x + Real.sin x
noncomputable def f'' (x : ℝ) : ℝ := -4 * Real.sin x + Real.cos x

theorem inflection_point_on_3x {x0 : ℝ} (h : f'' x0 = 0) : (f x0) = 3 * x0 := by
  sorry

end inflection_point_on_3x_l827_827004


namespace max_voters_is_five_l827_827713

noncomputable def max_voters_after_T (x : ℕ) : ℕ :=
if h : 0 ≤ (x - 11) then x - 11 else 0

theorem max_voters_is_five (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 10) :
  max_voters_after_T x = 5 :=
by
  sorry

end max_voters_is_five_l827_827713


namespace cookout_ratio_l827_827048

theorem cookout_ratio (K_2004 K_2005 : ℕ) (h1 : K_2004 = 60) (h2 : (2 / 3) * K_2005 = 20) :
  K_2005 / K_2004 = 1 / 2 :=
by sorry

end cookout_ratio_l827_827048


namespace maximum_n_for_positive_sum_l827_827995

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :=
  S n > 0

-- Definition of the arithmetic sequence properties
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d
  
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)

-- Given conditions
variable (h₁ : a 1 > 0)
variable (h₅ : a 2016 + a 2017 > 0)
variable (h₆ : a 2016 * a 2017 < 0)

-- Add the definition of the sum of the first n terms of the arithmetic sequence
noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * (a 0 + a n) / 2

-- Prove the final statement
theorem maximum_n_for_positive_sum : max_n_for_positive_sum a S 4032 :=
by
  -- conditions to use in the proof
  have h₁ : a 1 > 0 := sorry
  have h₅ : a 2016 + a 2017 > 0 := sorry
  have h₆ : a 2016 * a 2017 < 0 := sorry
  -- positively bounded sum
  let Sn := sum_of_first_n_terms a
  -- proof to utilize Lean's capabilities, replace with actual proof later
  sorry

end maximum_n_for_positive_sum_l827_827995


namespace dice_sum_22_probability_l827_827040

/-- If four standard six-faced dice are rolled, what is the probability that the sum of the 
face-up integers is 22? -/
theorem dice_sum_22_probability :
  (∃ dice : Fin 6 → ℕ, (∀ i, dice i ∈ Finset.range 1 7) ∧ dice 0 + dice 1 + dice 2 + dice 3 = 22) → 
  (1 / 6 ^ 4 * 4 + 1 / 6 ^ 4 = 5 / 1296) :=
sorry

end dice_sum_22_probability_l827_827040


namespace m_greater_than_3_range_of_a_l827_827523

-- Define the function f(x)
def f (a x : ℝ) : ℝ := log a ((x - 3) / (x + 3))

-- Define the conditions and the hypotheses
variables (f m n : ℝ) (a : ℝ)
variable (h_a : 0 < a ∧ a < 1)
variable (h_domain_1 : m ≤ n)
variable (h_domain_2 : is_domain (λ x, log a ((x - 3) / (x + 3)) ≥ log a [a (n - 1)] ∧ log a ((x - 3) / (x + 3)) ≤ log a [a (m - 1)]))

-- Prove that m > 3
theorem m_greater_than_3 : m > 3 :=
sorry

-- Establish the range of the positive number a
theorem range_of_a : 0 < a ∧ a < (2 - real.sqrt 3) / 4 :=
sorry

end m_greater_than_3_range_of_a_l827_827523


namespace Y_4_3_l827_827546

def Y (x y : ℝ) : ℝ := x^2 - 3 * x * y + y^2

theorem Y_4_3 : Y 4 3 = -11 :=
by
  -- This line is added to skip the proof and focus on the statement.
  sorry

end Y_4_3_l827_827546


namespace tax_computation_l827_827049

def income : ℕ := 56000
def first_portion_income : ℕ := 40000
def first_portion_rate : ℝ := 0.12
def remaining_income : ℕ := income - first_portion_income
def remaining_rate : ℝ := 0.20
def expected_tax : ℝ := 8000

theorem tax_computation :
  (first_portion_rate * first_portion_income) +
  (remaining_rate * remaining_income) = expected_tax := by
  sorry

end tax_computation_l827_827049


namespace george_change_l827_827472

theorem george_change (x : ℕ) (hx1 : x < 100)
  (hquarter : ∃ q : ℕ, x = 25 * q + 7)
  (hdime : ∃ d : ℕ, x = 10 * d + 4) :
  ∑ k in {k | k < 100 ∧ (∃ q : ℕ, k = 25 * q + 7) ∧ (∃ d : ℕ, k = 10 * d + 4)}, k = 139 :=
by
  sorry

end george_change_l827_827472


namespace number_of_sequences_l827_827625

theorem number_of_sequences
  (students_per_class : ℕ)
  (classes : ℕ)
  (meetings_per_week : ℕ)
  (students_per_class = 8) 
  (classes = 2) 
  (meetings_per_week = 3) : 
  students_per_class ^ (classes * meetings_per_week) = 262144 := 
sorry

end number_of_sequences_l827_827625


namespace exists_set_M_l827_827445

-- Define what it means for a set of natural numbers to satisfy the given conditions.
def set_condition (M : set ℕ) : Prop :=
  M.size = 1992 ∧
  (∀ x ∈ M, ∃ m k : ℕ, x = m^k ∧ k ≥ 2) ∧
  (∀ S ⊆ M, ∃ m k : ℕ, (S.sum id) = m^k ∧ k ≥ 2)
  
theorem exists_set_M : ∃ M : set ℕ, set_condition M := 
sorry

end exists_set_M_l827_827445


namespace dot_product_zero_l827_827170

noncomputable def parabola_focus := (4, 0 : ℝ)

noncomputable def point_A (y : ℝ) : (ℝ × ℝ) :=
if abs y = 4 then (0, y) else (0, 0)

noncomputable def point_B := (-4, 0 : ℝ)

noncomputable def vec_FA (y : ℝ) : ℝ × ℝ :=
prod.mk (-4) y

noncomputable def vec_AB (y : ℝ) : ℝ × ℝ :=
if y = 4 then (-4, -4 : ℝ) else if y = -4 then (-4, 4 : ℝ) else (-4, 0)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_zero (y : ℝ) (hy : abs y = 4) :
  dot_product (vec_FA y) (vec_AB y) = 0 :=
begin
  sorry
end

end dot_product_zero_l827_827170


namespace compare_logs_l827_827110

noncomputable def e := Real.exp 1
noncomputable def log_base_10 (x : Real) := Real.log x / Real.log 10

theorem compare_logs (x : Real) (hx : e < x ∧ x < 10) :
  let a := Real.log (Real.log x)
  let b := log_base_10 (log_base_10 x)
  let c := Real.log (log_base_10 x)
  let d := log_base_10 (Real.log x)
  c < b ∧ b < d ∧ d < a := 
sorry

end compare_logs_l827_827110


namespace range_of_a_minus_b_l827_827541

theorem range_of_a_minus_b (a b : ℝ) (h1 : 1 < a ∧ a < 4) (h2 : -2 < b ∧ b < 4) : -3 < a - b ∧ a - b < 6 :=
sorry

end range_of_a_minus_b_l827_827541


namespace sum_of_primes_between_10_and_20_is_60_l827_827271

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827271


namespace midpoint_y_coordinate_Sn_formula_lambda_value_l827_827911

def A (x1 y1 : ℝ) : Prop :=
  y1 = 1/2 + log 2 (x1 / (1 - x1))

def B (x2 y2 : ℝ) : Prop :=
  y2 = 1/2 + log 2 (x2 / (1 - x2))

def M (x1 x2 y1 y2 : ℝ) : Prop :=
  x1 + x2 = 1 ∧ y1 + y2 = 1

def f (x : ℝ) : ℝ :=
  1/2 + log 2 (x / (1 - x))

theorem midpoint_y_coordinate (x1 y1 x2 y2 : ℝ) (h1: A x1 y1) (h2: B x2 y2) (h3: M x1 x2 y1 y2) : 
  (y1 + y2) / 2 = 1/2 := by
  sorry

noncomputable def Sn (n : ℕ) :=
  (List.range (n-1)).sum (λ i, f (i+1/n))

theorem Sn_formula (n : ℕ) (hn : n ≥ 2) : 
  Sn n = (n - 1) / 2 := by
  sorry

noncomputable def an (n : ℕ) : ℝ :=
  if n = 1 then 2 / 3 else 1 / ((Sn n) + 1) * (Sn (n+1) + 1)

noncomputable def Tn (n : ℕ) :=
  (List.range n).sum (λ i, an (i+1))

theorem lambda_value (n : ℕ) (hn : n ≥ 1) : 
  ∃ λ : ℝ, λ = 1 ∧ Tn n ≤ λ * (Sn (n+1) + 1) := by
  sorry

end midpoint_y_coordinate_Sn_formula_lambda_value_l827_827911


namespace point_in_fourth_quadrant_l827_827149

theorem point_in_fourth_quadrant (θ : ℝ) (h : -1 < Real.cos θ ∧ Real.cos θ < 0) :
    ∃ (x y : ℝ), x = Real.sin (Real.cos θ) ∧ y = Real.cos (Real.cos θ) ∧ x < 0 ∧ y > 0 :=
by
  sorry

end point_in_fourth_quadrant_l827_827149


namespace average_book_width_l827_827597

def book_widths : List ℝ := [3, 0.5, 1.5, 4, 2, 5, 8]

def number_of_books : ℕ := 7

theorem average_book_width :
  (book_widths.sum / number_of_books.toReal) = 3.43 :=
by
  sorry

end average_book_width_l827_827597


namespace new_alcohol_percentage_l827_827771

theorem new_alcohol_percentage (initial_percentage : ℝ) (replacement_percentage : ℝ) (replaced_quantity : ℝ) (total_volume : ℝ)
(initial_percentage_eq : initial_percentage = 0.40)
(replacement_percentage_eq : replacement_percentage = 0.19)
(replaced_quantity_eq : replaced_quantity = 0.6666666666666666)
(total_volume_eq : total_volume = 1) :
  let remaining_whisky := total_volume - replaced_quantity,
      remaining_alcohol := remaining_whisky * initial_percentage,
      replacement_alcohol := replaced_quantity * replacement_percentage,
      total_alcohol := remaining_alcohol + replacement_alcohol,
      new_percentage := total_alcohol / total_volume * 100 in
  new_percentage = 26 :=
by
  sorry

end new_alcohol_percentage_l827_827771


namespace probability_relatively_prime_to_42_l827_827186

/-- Two integers are relatively prime if they have no common factors other than 1 or -1. -/
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set of positive integers less than or equal to 42. -/
def in_range (n : ℕ) : Prop := n ≤ 42

/-- The set of integers that are relatively prime to 42 and less than or equal to 42. -/
def relatively_prime_to_42 (n : ℕ) : Prop := relatively_prime n 42 ∧ in_range n

/-- The probability that a positive integer less than or equal to 42 is relatively prime to 42. 
Expressed as a common fraction. -/
theorem probability_relatively_prime_to_42 : 
  (Finset.filter relatively_prime_to_42 (Finset.range 43)).card * 7 = 12 * 42 :=
sorry

end probability_relatively_prime_to_42_l827_827186


namespace tangents_and_EF_concur_l827_827601

open EuclideanGeometry Finset

-- Define points and circle
variables {ω : Set Point}
variables {A B C D E F : Point}

-- Assume given conditions
axiom h1 : CyclicQuadrilateral A B C D ω
axiom h2 : IsTangent A ω
axiom h3 : LineParallel (Line.passThrough D) (TangentTo ω A)
axiom h4 : LineMeetsCircle (Line.parallelThrough D (TangentTo ω A)) ω E
axiom h5 : OnDifferentSide F E CD
axiom h6 : OnSameSide A E CD
axiom h7 : AE * AD * CF = BE * BC * DF
axiom h8 : ∠CFD = 2 * ∠AFB

-- Theorem to prove the tangents and line EF concur
theorem tangents_and_EF_concur : Concurrent (TangentTo ω A) (TangentTo ω B) (Line.passThrough E F) :=
sorry

end tangents_and_EF_concur_l827_827601


namespace prime_eq_solution_l827_827612

theorem prime_eq_solution (a b : ℕ) (h1 : Nat.Prime a) (h2 : b > 0)
  (h3 : 9 * (2 * a + b) ^ 2 = 509 * (4 * a + 511 * b)) : 
  (a = 251 ∧ b = 7) :=
sorry

end prime_eq_solution_l827_827612


namespace union_complement_eq_l827_827018

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}

theorem union_complement_eq : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} := by
  sorry

end union_complement_eq_l827_827018


namespace calculate_ladder_cost_l827_827592

theorem calculate_ladder_cost (ladders1 ladders2 rungs1 rungs2 rung_cost : ℕ) : 
  (ladders1 = 10) → 
  (rungs1 = 50) → 
  (ladders2 = 20) → 
  (rungs2 = 60) → 
  (rung_cost = 2) → 
  (ladders1 * rungs1 + ladders2 * rungs2) * rung_cost = 3400 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5] 
  norm_num 
  sorry

end calculate_ladder_cost_l827_827592


namespace min_AB_value_l827_827901

theorem min_AB_value :
  ∃ (a : ℝ), (∀ x1 x2, a = real.exp (2 * x1 + 1) ∧ a = real.sqrt (2 * x2 - 1) →
            abs (x2 - x1) = (1 / 2) * abs (a^2 - real.log a + 2)) ∧
  abs ((1 / 2) * abs ((real.sqrt 2 / 2)^2 - real.log (real.sqrt 2 / 2) + 2)) = (5 + real.log 2) / 4 :=
by
  sorry

end min_AB_value_l827_827901


namespace jackson_vs_brandon_meagan_l827_827588

def jackson_initial_usd : ℝ := 500
def brandon_initial_cad : ℝ := 600
def meagan_initial_jpy : ℝ := 400000

def exchange_rate_usd_to_cad : ℝ := 1.2
def exchange_rate_usd_to_jpy : ℝ := 110

def jackson_final_usd : ℝ := jackson_initial_usd * 4
def brandon_final_cad : ℝ := brandon_initial_cad * 0.2
def meagan_final_jpy : ℝ := meagan_initial_jpy * 1.5

def brandon_final_usd : ℝ := brandon_final_cad / exchange_rate_usd_to_cad
def meagan_final_usd : ℝ := meagan_final_jpy / exchange_rate_usd_to_jpy

def total_brandon_meagan_final_usd : ℝ := brandon_final_usd + meagan_final_usd

theorem jackson_vs_brandon_meagan :
  jackson_final_usd - total_brandon_meagan_final_usd = -3554.55 := sorry

end jackson_vs_brandon_meagan_l827_827588


namespace sum_primes_10_20_l827_827220

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827220


namespace clock_opposite_directions_period_l827_827050

theorem clock_opposite_directions_period (H : clock_shows_opposite_directions (22 : ℕ)) : period = 24 := sorry

end clock_opposite_directions_period_l827_827050


namespace can_have_property_l827_827820

/-- A function that checks if two vectors are parallel -/
def parallel (v w : ℝ^3) : Prop :=
∃ k : ℝ, k * v = w

/-- A predicate describing the conditions of the finite set S -/
def finite_set_property (S : set ℝ^3) : Prop :=
  ∃ (S_finite : finite S) (not_coplanar : ¬ coplanar S), 
  ∀ A B ∈ S, ∃ C D ∈ S, C ≠ D ∧ (parallel (B - A) (D - C)) ∧ ¬ collinear {A, B, C, D}

/-- The proof problem statement -/
theorem can_have_property :
  ∃ S : set ℝ^3, finite_set_property S :=
sorry

end can_have_property_l827_827820


namespace angle_2013_in_third_quadrant_l827_827335

theorem angle_2013_in_third_quadrant :
  2013 % 360 = 213 ∧ (213 ∈ third_quadrant) → (2013 ∈ third_quadrant) := by
  sorry

end angle_2013_in_third_quadrant_l827_827335


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_l827_827207

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4 : 
  Nat.minFac (11^5 - 11^4) = 2 := 
by sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_l827_827207


namespace bernie_postcards_l827_827396

theorem bernie_postcards :
  let initial_postcards := 18
  let price_sell := 15
  let price_buy := 5
  let sold_postcards := initial_postcards / 2
  let earned_money := sold_postcards * price_sell
  let bought_postcards := earned_money / price_buy
  let remaining_postcards := initial_postcards - sold_postcards
  let final_postcards := remaining_postcards + bought_postcards
  final_postcards = 36 := by sorry

end bernie_postcards_l827_827396


namespace sum_of_primes_between_10_and_20_l827_827250

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827250


namespace inequality_proof_l827_827884

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  sorry

end inequality_proof_l827_827884


namespace solve_for_q_l827_827648

theorem solve_for_q
  (n m q : ℚ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m - n) / 66)
  (h3 : 5 / 6 = (q - m) / 150) :
  q = 230 :=
by
  sorry

end solve_for_q_l827_827648


namespace natural_numbers_coprime_sum_l827_827857

theorem natural_numbers_coprime_sum :
  ∀ (k : ℕ), k ≠ 1 ∧ k ≠ 2 ∧ k ≠ 3 ∧ k ≠ 4 ∧ k ≠ 6 →
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ Nat.coprime a b ∧ k = a + b :=
by
  intro k h
  sorry

end natural_numbers_coprime_sum_l827_827857


namespace ladder_cost_l827_827589

theorem ladder_cost (ladders1 ladders2 rung_count1 rung_count2 cost_per_rung : ℕ)
  (h1 : ladders1 = 10) (h2 : ladders2 = 20) (h3 : rung_count1 = 50) (h4 : rung_count2 = 60) (h5 : cost_per_rung = 2) :
  (ladders1 * rung_count1 + ladders2 * rung_count2) * cost_per_rung = 3400 :=
by 
  sorry

end ladder_cost_l827_827589


namespace figure_area_is_68_l827_827998

-- Define the segments as variables
def segment_length_bottom : ℕ := 5
def segment_length_left : ℕ := 7
def segment_length_middle : ℕ := 3
def segment_length_upper : ℕ := 4
def segment_length_right : ℕ := 5
def segment_length_overlap : ℕ := 1

-- Define the areas of individual rectangles
def area_bottom_rectangle : ℕ := segment_length_bottom * segment_length_left
def area_middle_rectangle : ℕ := segment_length_middle * segment_length_middle
def area_overlap_rectangle : ℕ := segment_length_upper * segment_length_overlap
def area_rightmost_rectangle : ℕ := segment_length_right * segment_length_upper

-- Define the total area
def total_area : ℕ := area_bottom_rectangle + area_middle_rectangle + area_overlap_rectangle + area_rightmost_rectangle

-- State the theorem that the total area of the figure is 68 square units.
theorem figure_area_is_68 : total_area = 68 :=
by
  simp [total_area, area_bottom_rectangle, area_middle_rectangle, area_overlap_rectangle, area_rightmost_rectangle,
        segment_length_bottom, segment_length_left, segment_length_middle, segment_length_upper, segment_length_right, segment_length_overlap]
  sorry

end figure_area_is_68_l827_827998


namespace parabola_vertex_in_third_quadrant_l827_827693

-- Defining the vertex of the given parabola
def vertex (a b c : ℝ) : ℝ × ℝ :=
  let h := -b/(2*a)
  let k := a*h^2 + b*h + c
  (h, k)

def parabola_eq_vertex_quadrant : Prop :=
  vertex (-2) 6 (-21) = (-3, -21) ∧ (-3 < 0 ∧ -21 < 0) -- Verifying the vertex (x, y) = (-3, -21) lies in the third quadrant

theorem parabola_vertex_in_third_quadrant :
  parabola_eq_vertex_quadrant :=
by 
  sorry

end parabola_vertex_in_third_quadrant_l827_827693


namespace first_number_in_set_l827_827154

theorem first_number_in_set (x : ℝ)
  (h : (x + 40 + 60) / 3 = (10 + 80 + 15) / 3 + 5) :
  x = 20 := by
  sorry

end first_number_in_set_l827_827154


namespace tanya_time_proof_l827_827735

noncomputable def time_sakshi : ℝ := 10
noncomputable def efficiency_increase : ℝ := 1.25
noncomputable def time_tanya (time_sakshi : ℝ) (efficiency_increase : ℝ) : ℝ := time_sakshi / efficiency_increase

theorem tanya_time_proof : time_tanya time_sakshi efficiency_increase = 8 := 
by 
  sorry

end tanya_time_proof_l827_827735


namespace total_adults_across_all_three_buses_l827_827695

def total_passengers : Nat := 450
def bus_A_passengers : Nat := 120
def bus_B_passengers : Nat := 210
def bus_C_passengers : Nat := 120
def children_ratio_A : ℚ := 1/3
def children_ratio_B : ℚ := 2/5
def children_ratio_C : ℚ := 3/8

theorem total_adults_across_all_three_buses :
  let children_A := bus_A_passengers * children_ratio_A
  let children_B := bus_B_passengers * children_ratio_B
  let children_C := bus_C_passengers * children_ratio_C
  let adults_A := bus_A_passengers - children_A
  let adults_B := bus_B_passengers - children_B
  let adults_C := bus_C_passengers - children_C
  (adults_A + adults_B + adults_C) = 281 := by {
    -- The proof steps will go here
    sorry
}

end total_adults_across_all_three_buses_l827_827695


namespace middle_part_of_sum_is_120_l827_827542

theorem middle_part_of_sum_is_120 (x : ℚ) (h : 2 * x + x + (1 / 2) * x = 120) : 
  x = 240 / 7 := sorry

end middle_part_of_sum_is_120_l827_827542


namespace hyperbola_eccentricity_l827_827897

theorem hyperbola_eccentricity
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (hyp : ∀ k : ℝ, ∃ x y : ℝ, (x - 1)^2 + y^2 = 3/4 ∧ y = k * x ∧ (x^2 / a^2 - y^2 / b^2 = 1))
  : 2 < sqrt (1 + (b/a)^2) :=
by
  sorry

end hyperbola_eccentricity_l827_827897


namespace exists_six_numbers_with_properties_l827_827323

theorem exists_six_numbers_with_properties :
  ∃ (a : Fin 6 → ℕ), 
  (∀ i j : Fin 6, i ≠ j → ¬ (a i ∣ a j)) ∧ 
  (∀ i : Fin 6, (a i)^2 ∣ (∏ (j : Fin 6) in Finset.univ.filter (λ k, k ≠ i), a j)) :=
by
  sorry

end exists_six_numbers_with_properties_l827_827323


namespace binkie_gemstones_l827_827834

variable (Binkie Frankie Spaatz : ℕ)

-- Define the given conditions
def condition1 : Binkie = 4 * Frankie := by sorry
def condition2 : Spaatz = (1 / 2) * Frankie - 2 := by sorry
def condition3 : Spaatz = 1 := by sorry

-- State the theorem to be proved
theorem binkie_gemstones : Binkie = 24 := by
  have h_Frankie : Frankie = 6 := by
    sorry
  rw [←condition3, ←condition2] at h_Frankie
  have h_Binkie : Binkie = 4 * 6 := by
    rw [condition1]
    sorry
  rw [h_Binkie]
  exact
    show 4 * 6 = 24 from rfl

end binkie_gemstones_l827_827834


namespace add_same_sign_abs_l827_827316

theorem add_same_sign_abs (a b : ℤ) : 
  (∀ a b : ℤ, (a ≥ 0 ∧ b ≥ 0) → (|a + b| = |a| + |b| ∧ a + b ≥ 0)) ∧ 
  (∀ a b : ℤ, (a < 0 ∧ b < 0) → (|a + b| = |a| + |b| ∧ a + b < 0)) :=
by
  intro a b
  sorry

end add_same_sign_abs_l827_827316


namespace total_supervisors_correct_l827_827676

-- Define the number of supervisors on each bus
def bus_supervisors : List ℕ := [4, 5, 3, 6, 7]

-- Define the total number of supervisors
def total_supervisors := bus_supervisors.sum

-- State the theorem to prove that the total number of supervisors is 25
theorem total_supervisors_correct : total_supervisors = 25 :=
by
  sorry -- Proof is to be completed

end total_supervisors_correct_l827_827676


namespace cost_prices_three_watches_l827_827802

theorem cost_prices_three_watches :
  ∃ (C1 C2 C3 : ℝ), 
    (0.9 * C1 + 210 = 1.04 * C1) ∧ 
    (0.85 * C2 + 180 = 1.03 * C2) ∧ 
    (0.95 * C3 + 250 = 1.06 * C3) ∧ 
    C1 = 1500 ∧ 
    C2 = 1000 ∧ 
    C3 = (25000 / 11) :=
by 
  sorry

end cost_prices_three_watches_l827_827802


namespace odd_function_identity_l827_827920

def f (x : ℝ) : ℝ := if x > 0 then 2^x - 3 else if x < 0 then -(2^(-x) - 3) else 0

theorem odd_function_identity : 
  f(-2) + f(0) = -1 :=
by
  sorry

end odd_function_identity_l827_827920


namespace quadratic_factoring_method_l827_827688

-- Define the condition for solving a quadratic equation by factoring
def turns_to_zero (equation : polynomial ℝ) : Prop :=
  ∃ (a b c : ℝ), equation = (a * X^2 + b * X + c) ∧ (a * X^2 + b * X + c = 0)

-- Define the method of factoring by common factor or formula method
def factoring_method (equation : polynomial ℝ) : Prop :=
  ∃ (p q : polynomial ℝ), equation = p * q

-- The final output method we are proving
def method_to_transform : Prop := "multiplication (or multiplying)"

theorem quadratic_factoring_method (equation : polynomial ℝ)
  (h1 : turns_to_zero equation)
  (h2 : factoring_method equation) :
  method_to_transform = "multiplication (or multiplying)" :=
sorry

end quadratic_factoring_method_l827_827688


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2_l827_827208

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2 :
  nat.min_fac (11^5 - 11^4) = 2 :=
by
  sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2_l827_827208


namespace uneaten_chips_correct_l827_827428

def cookies_per_dozen : Nat := 12
def dozens : Nat := 4
def chips_per_cookie : Nat := 7

def total_cookies : Nat := dozens * cookies_per_dozen
def total_chips : Nat := total_cookies * chips_per_cookie
def eaten_cookies : Nat := total_cookies / 2
def uneaten_cookies : Nat := total_cookies - eaten_cookies

def uneaten_chips : Nat := uneaten_cookies * chips_per_cookie

theorem uneaten_chips_correct : uneaten_chips = 168 :=
by
  -- Placeholder for the proof
  sorry

end uneaten_chips_correct_l827_827428


namespace relationship_between_abc_l827_827968

theorem relationship_between_abc (a b c k : ℝ) 
  (hA : -3 = - (k^2 + 1) / a)
  (hB : -2 = - (k^2 + 1) / b)
  (hC : 1 = - (k^2 + 1) / c)
  (hk : 0 < k^2 + 1) : c < a ∧ a < b :=
by
  sorry

end relationship_between_abc_l827_827968


namespace ellipse_equation_l827_827905

theorem ellipse_equation (center_origin : ∀ P, P ∈ E → (P = (0, 0)))
                         (eccentricity_half : ∀ (E : Ellipse), E.eccentricity = 1/2)
                         (focus_center_circle : ∀ P, P.foci ∈ E → P ∈ {p : Point | p.x^2 + p.y^2 - 4 * p.x + 2 = 0}) :
  ∃ (a b : ℝ), a = 4 ∧ b^2 = 12 ∧ ∀ (x y : ℝ), (x, y) ∈ E ↔ (x^2 / 16 + y^2 / 12 = 1) :=
sorry

end ellipse_equation_l827_827905


namespace calories_after_two_days_l827_827441

theorem calories_after_two_days 
    (burgers_per_day : ℕ) (calories_per_burger : ℕ) (days : ℕ) :
    burgers_per_day = 3 → calories_per_burger = 20 → days = 2 →
    burgers_per_day * calories_per_burger * days = 120 :=
by
  intros h_burgers h_calories h_days
  rw [h_burgers, h_calories, h_days]
  norm_num

end calories_after_two_days_l827_827441


namespace number_of_sides_of_polygon_l827_827561

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (h1 : exterior_angle = 30) (h2 : sum_exterior_angles = 360) :
  sum_exterior_angles / exterior_angle = 12 := 
by
  sorry

end number_of_sides_of_polygon_l827_827561


namespace turtles_remaining_on_log_l827_827765
-- Importing necessary modules

-- Defining the problem
def initial_turtles : ℕ := 9
def turtles_climbed : ℕ := (initial_turtles * 3) - 2
def total_turtles : ℕ := initial_turtles + turtles_climbed
def remaining_turtles : ℕ := total_turtles / 2

-- Stating the proof problem
theorem turtles_remaining_on_log : remaining_turtles = 17 := 
  sorry

end turtles_remaining_on_log_l827_827765


namespace problem_l827_827114

noncomputable def nums : Type := { p q r s : ℝ // p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s }

theorem problem (n : nums) :
  let p := n.1
      q := n.2.1
      r := n.2.2.1
      s := n.2.2.2.1
  in (r + s = 12 * p) → (r * s = -13 * q) → (p + q = 12 * r) → (p * q = -13 * s) → p + q + r + s = 2028 :=
by
  intros
  sorry

end problem_l827_827114


namespace find_rate_of_interest_l827_827358

-- Define the problem conditions
def principal_B : ℝ := 4000
def principal_C : ℝ := 2000
def time_B : ℝ := 2
def time_C : ℝ := 4
def total_interest : ℝ := 2200

-- Define the unknown rate of interest per annum
noncomputable def rate_of_interest (R : ℝ) : Prop :=
  let interest_B := (principal_B * R * time_B) / 100
  let interest_C := (principal_C * R * time_C) / 100
  interest_B + interest_C = total_interest

-- Statement to prove that the rate of interest is 13.75%
theorem find_rate_of_interest : rate_of_interest 13.75 := by
  sorry

end find_rate_of_interest_l827_827358


namespace ratio_of_black_to_white_area_l827_827866

theorem ratio_of_black_to_white_area :
  let R₁ := 1
  let R₂ := 4
  let R₃ := 6
  let R₄ := 8
  let R₅ := 10
  let area r := π * r^2
  let A₁ := area R₁
  let A₂ := area R₂
  let A₃ := area R₃
  let A₄ := area R₄
  let A₅ := area R₅
  let black_area := A₁ + (A₃ - A₂) + (A₅ - A₄)
  let white_area := (A₂ - A₁) + (A₄ - A₃)
  black_area / white_area = 57 / 43 :=
by
  let R₁ := 1
  let R₂ := 4
  let R₃ := 6
  let R₄ := 8
  let R₅ := 10
  let area r := π * r^2
  let A₁ := area R₁
  let A₂ := area R₂
  let A₃ := area R₃
  let A₄ := area R₄
  let A₅ := area R₅
  let black_area := A₁ + (A₃ - A₂) + (A₅ - A₄)
  let white_area := (A₂ - A₁) + (A₄ - A₃)
  have black_area_eq : black_area = 57 * π := sorry
  have white_area_eq : white_area = 43 * π := sorry
  exact div_eq_div_of_eq (black_area_eq, white_area_eq, π_ne_zero; by sorry)


end ratio_of_black_to_white_area_l827_827866


namespace total_payment_l827_827754

-- Define the discount rules
def discount (total_price: Real) : Real :=
  if total_price ≤ 30 then 
    total_price
  else if total_price ≤ 50 then 
    total_price * 0.9
  else 
    50 * 0.9 + (total_price - 50) * 0.8

-- Given the combined original price
def original_price_combo : Real :=
  23 + 36 * 10 / 9

-- Definition of how much Li Hua should pay if he buys the same books in one trip
theorem total_payment (a : Real) (h : a = discount original_price_combo) : a = 55.4 :=
by
  rewrite h
  sorry

end total_payment_l827_827754


namespace distinct_even_three_digit_numbers_count_l827_827533

theorem distinct_even_three_digit_numbers_count :
  let digits := {1, 2, 3, 4, 5}
  ∧ (∀ x ∈ digits, x ∈ {1, 2, 3, 4, 5})
  ∧ (∀ a b c : Nat, a ∈ digits → b ∈ digits → c ∈ digits → (a ≠ b ∧ b ≠ c ∧ a ≠ c))
  ∧ (∃ u ∈ {2, 4}, ∃ h t : Nat, h ∈ digits \ {u} ∧ t ∈ digits \ {u, h}) →
  2 * 3 * 3 = 18 := sorry

end distinct_even_three_digit_numbers_count_l827_827533


namespace quarts_per_bottle_l827_827587

-- Conditions
def hot_tub_volume_gallons : ℝ := 40
def quarts_per_gallon : ℝ := 4
def bottle_cost_dollars : ℝ := 50
def discount_rate : ℝ := 0.20
def total_spent_dollars : ℝ := 6400

-- Conclusion we want to prove
theorem quarts_per_bottle (hot_tub_volume_gallons > 0) 
                         (quarts_per_gallon > 0) 
                         (bottle_cost_dollars > 0) 
                         (discount_rate >= 0) 
                         (discount_rate < 1)
                         (total_spent_dollars > 0) : 
                         ∃ (quarts_per_bottle : ℝ), quarts_per_bottle = 1 := 
by
  sorry

end quarts_per_bottle_l827_827587


namespace proposition_B_l827_827137

-- Definitions of the conditions
def line (α : Type) := α
def plane (α : Type) := α
def is_within {α : Type} (a : line α) (p : plane α) : Prop := sorry
def is_perpendicular {α : Type} (a : line α) (p : plane α) : Prop := sorry
def planes_are_perpendicular {α : Type} (p₁ p₂ : plane α) : Prop := sorry
def is_prism (poly : Type) : Prop := sorry

-- Propositions
def p {α : Type} (a : line α) (α₁ α₂ : plane α) : Prop :=
  is_within a α₁ ∧ is_perpendicular a α₂ → planes_are_perpendicular α₁ α₂

def q (poly : Type) : Prop := 
  (∃ (face1 face2 : poly), face1 ≠ face2 ∧ sorry) ∧ sorry

-- Proposition B
theorem proposition_B {α : Type} (a : line α) (α₁ α₂ : plane α) (poly : Type) :
  (p a α₁ α₂) ∧ ¬(q poly) :=
by {
  -- Skipping proof
  sorry
}

end proposition_B_l827_827137


namespace num_subsets_num_proper_subsets_l827_827850

-- Defining the set
def my_set : set ℕ := {1, 2, 3, 4, 5}

-- Number of elements in the set
def n : ℕ := 5

-- Prove the number of subsets is 32
theorem num_subsets : fintype.card (set (fin n)) = 2^n := by
sorry

-- Prove the number of proper subsets is 31
theorem num_proper_subsets : fintype.card (set (fin n)) - 1 = 2^n - 1 := by
sorry

end num_subsets_num_proper_subsets_l827_827850


namespace mans_speed_against_current_l827_827781

variable (V_downstream V_current : ℝ)
variable (V_downstream_eq : V_downstream = 15)
variable (V_current_eq : V_current = 2.5)

theorem mans_speed_against_current : V_downstream - 2 * V_current = 10 :=
by
  rw [V_downstream_eq, V_current_eq]
  exact (15 - 2 * 2.5)

end mans_speed_against_current_l827_827781


namespace binkie_gemstones_l827_827835

variable (Binkie Frankie Spaatz : ℕ)

-- Define the given conditions
def condition1 : Binkie = 4 * Frankie := by sorry
def condition2 : Spaatz = (1 / 2) * Frankie - 2 := by sorry
def condition3 : Spaatz = 1 := by sorry

-- State the theorem to be proved
theorem binkie_gemstones : Binkie = 24 := by
  have h_Frankie : Frankie = 6 := by
    sorry
  rw [←condition3, ←condition2] at h_Frankie
  have h_Binkie : Binkie = 4 * 6 := by
    rw [condition1]
    sorry
  rw [h_Binkie]
  exact
    show 4 * 6 = 24 from rfl

end binkie_gemstones_l827_827835


namespace cylinder_height_relationship_l827_827706

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
π * r^2 * h

theorem cylinder_height_relationship (V : ℝ) (r1 r2 h1 h2 : ℝ)
  (hV1 : cylinder_volume r1 h1 = V)
  (hV2 : cylinder_volume r2 h2 = V)
  (hRadii : r2 = 1.2 * r1) : h1 = 1.44 * h2 :=
by
  unfold cylinder_volume at *
  sorry

end cylinder_height_relationship_l827_827706


namespace vector_parallel_unit_vector_coordinates_l827_827950

-- Step (1): Prove k = -1 given the vector conditions
theorem vector_parallel (k : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (-3, 2)) :
  (k * a + 2 * b = ((k - 6), 2 * k + 4)) ∧ ((2 * a - 4 * b) = (14, -4)) →
  (k = -1) := 
by sorry

-- Step (2): Prove the coordinates of vector c
theorem unit_vector_coordinates (c b : ℝ × ℝ) (hb : b = (-3, 2)) :
  (|c| = 1) ∧ (|c - b| = 2 * √5) →
  (c = (1, 0)) ∨ (c = (5 / 13, -12 / 13)) :=
by sorry

-- Adding noncomputable attribute if necessary
noncomputable def main_theorem (k : ℝ) (a b c : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (-3, 2)) :
  ((k * a + 2 * b = ((k - 6), 2 * k + 4)) ∧ ((2 * a - 4 * b) = (14, -4)) → (k = -1)) ∧
  ((|c| = 1) ∧ (|c - b| = 2 * √5) → (c = (1, 0)) ∨ (c = (5 / 13, -12 / 13))) :=
by
  split
  · exact vector_parallel k a b ha hb
  · exact unit_vector_coordinates c b hb

end vector_parallel_unit_vector_coordinates_l827_827950


namespace domain_of_f_monotonic_intervals_find_a_l827_827933

noncomputable def f (x a : ℝ) : ℝ := Real.log x + Real.log (2 - x) + a * x

theorem domain_of_f (a : ℝ) (ha : 0 < a) :
  {x : ℝ | f x a = Real.log x + Real.log (2 - x) + a * x} ⊆ (0, 2) :=
sorry

theorem monotonic_intervals (x : ℝ) :
  (∀ x ∈ (0, Real.sqrt 2), f x 1 = Real.log x + Real.log (2 - x) + 1 * x) ∧
  (∀ x ∈ (Real.sqrt 2, 2), f x 1 = Real.log x + Real.log (2 - x) + 1 * x) :=
sorry

theorem find_a (a : ℝ) (h_max : ∀ x ∈ (0, 1], f x a ≤ 1/2) :
  a = 1/2 :=
sorry

end domain_of_f_monotonic_intervals_find_a_l827_827933


namespace relationship_l827_827892

-- Given conditions
def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

-- The theorem to be proven
theorem relationship : c > b ∧ b > a := by
  sorry

end relationship_l827_827892


namespace smallest_possible_value_other_integer_l827_827670

theorem smallest_possible_value_other_integer (x : ℕ) (n : ℕ) (h_pos : x > 0)
  (h_gcd : ∃ m, Nat.gcd m n = x + 3 ∧ m = 30) 
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) :
  n = 162 := 
by sorry

end smallest_possible_value_other_integer_l827_827670


namespace vertical_asymptote_l827_827464

def function (x : ℚ) : ℚ := (x^2 + 2 * x + 3) / (5 * x - 9)

theorem vertical_asymptote : 
  let x := (9 / 5 : ℚ) in 
  (5 * x - 9 = 0) ∧ ¬ (x^2 + 2 * x + 3 = 0) :=
by
  sorry

end vertical_asymptote_l827_827464


namespace distance_point_to_vertical_line_l827_827160

/-- The distance from a point to a vertical line equals the absolute difference in the x-coordinates. -/
theorem distance_point_to_vertical_line (x1 y1 x2 : ℝ) (h_line : x2 = -2) (h_point : (x1, y1) = (1, 2)) :
  abs (x1 - x2) = 3 :=
by
  -- Place proof here
  sorry

end distance_point_to_vertical_line_l827_827160


namespace speed_of_current_l827_827360

-- Define the context and variables
variables (m c : ℝ)
-- State the conditions
variables (h1 : m + c = 12) (h2 : m - c = 8)

-- State the goal which is to prove the speed of the current
theorem speed_of_current : c = 2 :=
by
  sorry

end speed_of_current_l827_827360


namespace sum_of_primes_between_10_and_20_l827_827262

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827262


namespace sum_primes_between_10_and_20_l827_827240

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827240


namespace similar_not_congruent_l827_827727

/-- A figure is similar if it has the same shape, but not necessarily the same size. -/
def isSimilar (fig1 fig2 : Type) : Prop :=
  ∃ (f : fig1 → fig2), (∀ x y, (∃ r : ℝ, r ≠ 1 ∧ (f x = r * x) ∧ (f y = r * y)))

/-- A figure is congruent if it is identical in shape and size. -/
def isCongruent (fig1 fig2 : Type) : Prop :=
  ∃ (f : fig1 → fig2), (∀ x y, f x = x ∧ f y = y)

/-- A theorem stating that similar figures are not necessarily congruent. -/
theorem similar_not_congruent (A B : Type) (h_sim : isSimilar A B) :
  ¬ isCongruent A B := sorry

end similar_not_congruent_l827_827727


namespace elective_schemes_count_l827_827793

theorem elective_schemes_count :
  let total_courses := 10
  let choices_per_student := 3
  let conflicting_courses := 3
  let remaining_courses := total_courses - conflicting_courses
  combinatorial (choices_per_student := choices_per_student)
    (total_courses := total_courses)
    (choose conflicting_courses 1 * choose remaining_courses 2 + 
     choose remaining_courses 3) = 98
:=
begin
  sorry
end

end elective_schemes_count_l827_827793


namespace prob_all_zeroes_is_correct_prob_product_four_is_correct_l827_827697

noncomputable def prob_all_zeroes : ℚ := 
  (1 : ℚ) / 21

theorem prob_all_zeroes_is_correct :
  let bag_A := [0, 1, 1, 2, 2, 2]
  let bag_B := [0, 0, 0, 0, 1, 2, 2]
  let draw_from_A := ∀ x, x ∈ bag_A
  let draw_two_from_B := ∀ x y, x ∈ bag_B ∧ y ∈ bag_B ∧ x ≠ y
  let probability : ℚ := by
    have total_A := 6
    have total_B := (7.choose 2)
    have favorable_A := 1
    have favorable_B := (4.choose 2)
    exact (favorable_A * favorable_B : ℚ) / (total_A * total_B)
  in probability = prob_all_zeroes 
:= by sorry

noncomputable def prob_product_four : ℚ := 
  (4 : ℚ) / 63

theorem prob_product_four_is_correct :
  let bag_A := [0, 1, 1, 2, 2, 2]
  let bag_B := [0, 0, 0, 0, 1, 2, 2]
  let draw_from_A := ∀ x, x ∈ bag_A
  let draw_two_from_B := ∀ x y, x ∈ bag_B ∧ y ∈ bag_B ∧ x ≠ y
  let probability : ℚ := by
    have total_A := 6
    have total_B := (7.choose 2)
    have favorable_A := (3.choose 2 * 2.choose 1)
    have favorable_B := 2
    exact (favorable_A * favorable_B : ℚ) / (total_A * total_B)
  in probability = prob_product_four 
:= by sorry

end prob_all_zeroes_is_correct_prob_product_four_is_correct_l827_827697


namespace arithmetic_square_root_16_l827_827741

theorem arithmetic_square_root_16 : ∃ x : ℝ, x * x = 16 ∧ x ≥ 0 ∧ x = 4 :=
by
  use 4
  split
  · exact rfl
  · split
    · linarith
    · exact rfl

end arithmetic_square_root_16_l827_827741


namespace point_P_distance_to_y_axis_l827_827503

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, 3)

-- The distance from point P to the y-axis
def distance_to_y_axis (pt : ℝ × ℝ) : ℝ :=
  abs pt.1

-- Statement to prove
theorem point_P_distance_to_y_axis :
  distance_to_y_axis point_P = 2 :=
by
  sorry

end point_P_distance_to_y_axis_l827_827503


namespace sum_primes_between_10_and_20_is_60_l827_827235

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827235


namespace inequality_holds_l827_827873

theorem inequality_holds (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : ab > ac :=
by sorry

end inequality_holds_l827_827873


namespace root_condition_l827_827007

noncomputable def f (a : ℝ) (x : ℝ) := a * x^3 - 3 * x^2 + 1

theorem root_condition (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 0 ∧ ∀ x ≠ x₀, f a x ≠ 0 ∧ x₀ < 0) → a > 2 :=
sorry

end root_condition_l827_827007


namespace root_expr_value_eq_175_div_11_l827_827106

noncomputable def root_expr_value (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + bc + ca = 25) (h3 : abc = 10) : ℝ :=
  (a / (1 / a + b * c)) + (b / (1 / b + c * a)) + (c / (1 / c + a * b))

theorem root_expr_value_eq_175_div_11 (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : ab + bc + ca = 25) 
  (h3 : abc = 10) : 
  root_expr_value a b c h1 h2 h3 = 175 / 11 := 
sorry

end root_expr_value_eq_175_div_11_l827_827106


namespace hyperbola_asymptote_l827_827501

theorem hyperbola_asymptote (b : ℝ) (hb : 0 < b) (h_asymptote : ∃ (x y : ℝ), y = 3 * x ∧ (y^2 = b^2 * x^2)) : b = 3 :=
by
  have h1 : ∀ (x : ℝ), y = 3 * x → y^2 = (3*x)^2 := sorry
  have h2 : ∀ (x : ℝ), y^2 = (3*x)^2 → y^2 = b^2 * x^2 := sorry
  sorry

end hyperbola_asymptote_l827_827501


namespace simplify_expression_l827_827647

theorem simplify_expression (a : ℝ) : a^2 * (-a)^4 = a^6 := by
  sorry

end simplify_expression_l827_827647


namespace conjugate_of_z_l827_827898

-- Define the given complex number z
def z : ℂ := (3 + complex.i) / (1 - complex.i)

-- State that the conjugate of z is 1 - 2i
theorem conjugate_of_z : complex.conj z = 1 - 2i := 
  sorry

end conjugate_of_z_l827_827898


namespace first_offset_length_l827_827860

-- Definition of variables based on the given conditions
def diagonal : ℝ := 28
def second_offset : ℝ := 2
def area : ℝ := 140

-- The proof goal
theorem first_offset_length : (first_offset : ℝ) (diagonal * (first_offset + second_offset) / 2 = area) → first_offset = 8 :=
by
  sorry

end first_offset_length_l827_827860


namespace simplest_fraction_l827_827314

theorem simplest_fraction (x y : ℝ) (h1 : 2 * x ≠ 0) (h2 : x + y ≠ 0) :
  let A := (2 * x) / (4 * x^2)
  let B := (x^2 + y^2) / (x + y)
  let C := (x^2 + 2 * x + 1) / (x + 1)
  let D := (x^2 - 4) / (x + 2)
  B = (x^2 + y^2) / (x + y) ∧
  A ≠ (2 * x) / (4 * x^2) ∧
  C ≠ (x^2 + 2 * x + 1) / (x + 1) ∧
  D ≠ (x^2 - 4) / (x + 2) := sorry

end simplest_fraction_l827_827314


namespace problem_statement_l827_827879

noncomputable def a : ℝ := 31 / 32
noncomputable def b : ℝ := Real.cos (1 / 4)
noncomputable def c : ℝ := 4 * Real.sin (1 / 4)

theorem problem_statement : c > b ∧ b > a := 
by {
  sorry
}

end problem_statement_l827_827879


namespace max_table_sum_l827_827565

-- Define the conditions
def grid (x y : ℕ) := (x < 4) ∧ (y < 8)
def is_corner (x y : ℕ) := (x = 0 ∧ y = 0) ∨ (x = 0 ∧ y = 7) ∨ (x = 3 ∧ y = 0) ∨ (x = 3 ∧ y = 7)
def not_corner (x y : ℕ) := grid x y ∧ ¬is_corner x y
def cross_cells (x y : ℕ) := [(x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1)]

-- Formalize the theorem
theorem max_table_sum :
  ∀ (f : ℕ → ℕ → ℝ), (∀ x y, not_corner x y → ∑ c in (cross_cells x y), f (prod.fst c) (prod.snd c) ≤ 8) →
  ∑ x y, if not_corner x y then f x y else 0 = 96 :=
begin
  sorry
end

end max_table_sum_l827_827565


namespace sum_primes_10_to_20_l827_827289

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827289


namespace option_b_option_c_option_d_l827_827723

theorem option_b (M N U : Set) (h1 : M ⊆ U) (h2 : N ⊆ U) : M ∪ N ⊆ U :=
by
  sorry

theorem option_c (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hm : 0 < m) (hxy : x < y) : 
  (x + m) / (y + m) > x / y :=
by
  sorry

theorem option_d (x y m n : ℝ) (hx : 0 < x) (hy : 0 < y) (hm : 0 < m) (hn : 0 < n) 
  (hxy : x > y) (hmn : m > n) : (y + m) / (x + m) > (y + n) / (x + n) :=
by
  sorry

end option_b_option_c_option_d_l827_827723


namespace taxi_fare_l827_827691

theorem taxi_fare (x : ℝ) (h : 3.00 + 0.25 * ((x - 0.75) / 0.1) = 12) : x = 4.35 :=
  sorry

end taxi_fare_l827_827691


namespace find_length_of_polaroid_l827_827157

theorem find_length_of_polaroid 
  (C : ℝ) (W : ℝ) (L : ℝ)
  (hC : C = 40) (hW : W = 8) 
  (hFormula : C = 2 * (L + W)) : 
  L = 12 :=
by
  sorry

end find_length_of_polaroid_l827_827157


namespace sum_primes_between_10_and_20_l827_827284

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827284


namespace circumcircle_eqn_l827_827510

variables {α : Type*}
variables {x y ω a b c : α}
variables {A B M N C : α → α}

def antiparallel (angle : α) (m n c : α) := m * n = c ^ 2

theorem circumcircle_eqn (xOy: α) (A B : α → α) (C : α → α) (a b c : α) :
  (m n : α) →
  antiparallel xOy m (c) →
  antiparallel xOy n (c) →
  let M := (m, 0),
      N := (n, 0) in
  (a b : α) (y : α) →
  (m + c ^ 2 / m = a + b) →
  (y = c) →
  x^2 + 2 * x * y * cos ω + y^2 - (a + b) * x - 2 * c * y + c^2 = 0 :=
sorry

end circumcircle_eqn_l827_827510


namespace problem_statement_l827_827409

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l827_827409


namespace root_expr_value_eq_175_div_11_l827_827108

noncomputable def root_expr_value (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + bc + ca = 25) (h3 : abc = 10) : ℝ :=
  (a / (1 / a + b * c)) + (b / (1 / b + c * a)) + (c / (1 / c + a * b))

theorem root_expr_value_eq_175_div_11 (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : ab + bc + ca = 25) 
  (h3 : abc = 10) : 
  root_expr_value a b c h1 h2 h3 = 175 / 11 := 
sorry

end root_expr_value_eq_175_div_11_l827_827108


namespace simplify_expression_l827_827959

theorem simplify_expression (a : ℝ) (h : a < 1 / 2) : Real.sqrt (1 - 2 * a) = Real.sqrt (Real.abs (2 * a - 1)) :=
by
  sorry 

end simplify_expression_l827_827959


namespace transform_P_to_Q_l827_827896

theorem transform_P_to_Q (a b c : ℝ) (ha : a ≠ 0) :
  ∃ f : ℝ → ℝ, ∀ x, f (P x) = Q x :=
by
  let P := λ x : ℝ, a * x^2 + b * x + c
  let Q := λ x : ℝ, x^2
  sorry

end transform_P_to_Q_l827_827896


namespace distinct_real_numbers_sum_l827_827116

theorem distinct_real_numbers_sum:
  ∀ (p q r s : ℝ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    (r + s = 12 * p) →
    (r * s = -13 * q) →
    (p + q = 12 * r) →
    (p * q = -13 * s) →
    p + q + r + s = 2028 :=
by
  intros p q r s h_distinct h1 h2 h3 h4
  sorry

end distinct_real_numbers_sum_l827_827116


namespace problem1_problem2_l827_827614

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^2 * x^2

theorem problem1 (a : ℝ) (h : a > 0) : 
  (∀ x, (λ x, a^2 * (x - 1)^2) x = f(x - 1)) ∧
  (∀ y, y ≥ 0 → ∃ x, (λ x, a^2 * (x - 1)^2) x = y) :=
by
  sorry

theorem problem2 (a : ℝ) (h : a > 0) :
  (∀ n : ℤ, 3 ≤ (set.card {x : ℤ | (x - 1)^2 > f x a})) ↔
  (4 / 3 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end problem1_problem2_l827_827614


namespace max_planes_l827_827987

-- Definition: Any three out of five points are not collinear.
def not_collinear (points : Finset (Fin 5 → ℝ)) : Prop := 
  ∀ (a b c : Fin 5 → ℝ), {a, b, c} ⊆ points → 
  ¬(∃ l : ℝ → ℝ → ℝ, 
  ∀ (p : Fin 5 → ℝ), p ∈ {a, b, c} → l p.1 p.2 = 0)

-- Definition: Only four triangular pyramids can be constructed.
def four_triangular_pyramids (points: Finset (Fin 5 → ℝ)) : Prop :=
  (points.card = 5) ∧ 
  (∀ (p : Finset (Fin 4 → ℝ)), p ⊆ points → p.card = 4 → 
  ∃ c : Finset (Fin 5 → ℝ), ¬ (p ⊆ c) ∧ ¬ collinear p)

-- Lean statement to prove the problem.
theorem max_planes (points : Finset (Fin 5 → ℝ)) 
  (h1 : not_collinear points) 
  (h2 : four_triangular_pyramids points):
  ∃ (n : ℕ), n = 7 := 
sorry

end max_planes_l827_827987


namespace determine_k_l827_827845

noncomputable def k_value (k : ℤ) : Prop :=
  let m := (-2 - 2) / (3 - 1)
  let b := 2 - m * 1
  let y := m * 4 + b
  let point := (4, k / 3)
  point.2 = y

theorem determine_k :
  ∃ k : ℤ, k_value k ∧ k = -12 :=
by
  use -12
  sorry

end determine_k_l827_827845


namespace calculate_expr_l827_827402

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l827_827402


namespace lino_shells_total_l827_827129

def picked_up_shells : Float := 324.0
def put_back_shells : Float := 292.0

theorem lino_shells_total : picked_up_shells - put_back_shells = 32.0 :=
by
  sorry

end lino_shells_total_l827_827129


namespace smallest_integer_l827_827674

theorem smallest_integer (x : ℕ) (n : ℕ) (h_pos : 0 < x)
  (h_gcd : Nat.gcd 30 n = x + 3)
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) : n = 70 :=
begin
  sorry
end

end smallest_integer_l827_827674


namespace shenille_points_l827_827562

def shenille_total_points (x y : ℕ) : ℝ :=
  0.6 * x + 0.6 * y

theorem shenille_points (x y : ℕ) (h : x + y = 30) : 
  shenille_total_points x y = 18 := by
  sorry

end shenille_points_l827_827562


namespace line_position_l827_827965

variables (L1 L2 L3 : Type*) [line L1] [line L2] [line L3]

/-- Definition: Skew lines -/
def skew (L1 L2 : Type*) [line L1] [line L2] : Prop :=
  ¬∃ p : Type*, p ∈ L1 ∧ p ∈ L2 ∧ ¬∃ f g : Type*, f ≠ g ∧ f ∈ L1 ∧ g ∈ L1

/-- Definition: Parallel lines -/
def parallel (L1 L2 : Type*) [line L1] [line L2] : Prop :=
  ∃ p q : Type*, p ≠ q ∧ p ∈ L1 ∧ q ∈ L2

/-- Given that L1 and L2 are skew lines and L3 is parallel to L1, prove that L3 is either intersecting or skew to L2 -/
theorem line_position (h1 : skew L1 L2) (h2 : parallel L3 L1) : intersecting L3 L2 ∨ skew L3 L2 :=
sorry

end line_position_l827_827965


namespace sum_of_primes_between_10_and_20_l827_827249

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827249


namespace proof_problem_l827_827120

noncomputable def S : Set ℝ := {x : ℝ | x ≠ 0}

variables (g : ℝ → ℝ)

-- Given conditions
axiom g_condition1 : ∀ x ∈ S, g (1 / x) = x^2 * g x
axiom g_condition2 : ∀ x y ∈ S, x + y ∈ S → g (1 / x) + g (1 / y) = 4 + g (1 / (x + y))

-- Define the problem statement
theorem proof_problem : 
  let n := 1 in            -- Since we found only one possible value for g(1)
  let s := 8 in            -- The sum of the possible values for g(1)
  n * s = 8 :=
by
  sorry

end proof_problem_l827_827120


namespace max_abs_sum_l827_827606

-- Lean 4 Statement
theorem max_abs_sum (a b c d : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : 1 ≤ a) (h5 : d ≤ 9) :
    |a - b| + |b - c| + |c - d| + |d - a| ≤ 16 :=
by
  -- skipping proof with sorry
  sorry

end max_abs_sum_l827_827606


namespace S_sum_l827_827121

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2)
  else (n + 1) / 2

theorem S_sum :
  S 19 + S 37 + S 52 = 3 :=
by
  sorry

end S_sum_l827_827121


namespace maximum_M_value_l827_827017

theorem maximum_M_value (x y z u M : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < u)
  (h5 : x - 2 * y = z - 2 * u) (h6 : 2 * y * z = u * x) (h7 : z ≥ y) 
  : ∃ M, M ≤ z / y ∧ M ≤ 6 + 4 * Real.sqrt 2 :=
sorry

end maximum_M_value_l827_827017


namespace binary_sequence_debruijn_l827_827604

theorem binary_sequence_debruijn (n N : ℕ) (T : ℕ) (h1 : n < N) (h2 : N = 2^n + (n - 1))
  (h3 : ∀ x, binary_number x → x < 2^n) :
  ∃ T : ℕ, ∀ m, (m < N - n + 1) → (m + n - 1 < N) →
  (∃ B : ℕ, (B = (T / (10^(N - (m + n - 1)))) % (2^n)) ∧ 
  (B ∈ (set_of_substrings n N T))) :=
sorry

-- Definitions to make the theorem clearer

def binary_number (x : ℕ) : Prop := ∀ i, i < n → (x % (2^(i+1))) / 2^i < 2

noncomputable def set_of_substrings (n N T : ℕ) : set ℕ :=
{ x | ∃ m, m < N - n + 1 ∧ x = (T / (10^(N - (m + n - 1)))) % (2^n) }

end binary_sequence_debruijn_l827_827604


namespace extreme_values_in_interval_l827_827008

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem extreme_values_in_interval :
  (∀ x ∈ set.univ, differentiable ℝ (f x)) →
  (∀ x ∈ set.Icc (-3:ℝ) (4:ℝ), (f x) ≤ (f (-2)) ∧ (f x) ≥ (f (2))) :=
by
  intro h_diff
  intros x h_mem_interval
  have : f' : ℝ → ℝ := λ x, x^2 - 4
  sorry

end extreme_values_in_interval_l827_827008


namespace function_min_no_max_l827_827163

-- Given conditions
def f (x : ℝ) : ℝ
noncomputable def f' (x : ℝ) : ℝ := sorry
noncomputable def f'' (x : ℝ) : ℝ := sorry

axiom condition1 : ∀ x : ℝ, x * f'' x + f x = Real.exp x
axiom condition2 : f 1 = Real.exp 1

-- The theorem we need to prove
theorem function_min_no_max : 
  ∃ c : ℝ, (c = 1) ∧ (∀ x : ℝ, x ≠ 1 → f x > f 1 ∨ f x < f 1) :=
sorry

end function_min_no_max_l827_827163


namespace roots_imply_value_l827_827100

noncomputable def value_of_expression (a b c : ℝ) : ℝ :=
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)

theorem roots_imply_value {a b c : ℝ} 
  (h1 : a + b + c = 15) 
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 10) 
  : value_of_expression a b c = 175 / 11 :=
sorry

end roots_imply_value_l827_827100


namespace area_of_N_l827_827582

theorem area_of_N :
  let N := {p : ℝ × ℝ | p.1 < p.2 ∧ |p.1| < 3 ∧ |p.2| < 3 ∧ (5*p.1 - 3*p.2)*(p.1 + p.2) ≤ 0} in
  let area_N := 81 / 5 in
  measure N = area_N :=
sorry

end area_of_N_l827_827582


namespace calc_angle_CAB_l827_827159

theorem calc_angle_CAB (α β γ ε : ℝ) (hα : α = 79) (hβ : β = 63) (hγ : γ = 131) (hε : ε = 123.5) : 
  ∃ φ : ℝ, φ = 24 + 52 / 60 :=
by
  sorry

end calc_angle_CAB_l827_827159


namespace derek_savings_l827_827842

theorem derek_savings : 
  let a : ℕ := 2 in
  let savings := λ (n : ℕ) => a * 2^n in
  savings 11 = 4096 := 
by
  -- Define initial savings and the savings function
  let a : ℕ := 2
  let savings := λ (n : ℕ) => a * 2^n
  -- Prove that Derek's savings in December (n = 11) is 4096
  have : savings 11 = 2 * 2^11 := by rfl
  have : 2 * 2^11 = 4096 := by norm_num
  show savings 11 = 4096 by
    rw [this]
    sorry

end derek_savings_l827_827842


namespace units_digit_of_a_l827_827977

theorem units_digit_of_a (a : ℕ) (h : ∃ d ∈ {1, 3, 5, 7, 9}, (a^2 / 10) % 10 = d) : (a % 10 = 4) ∨ (a % 10 = 6) :=
by
  sorry

end units_digit_of_a_l827_827977


namespace sum_primes_between_10_and_20_l827_827245

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827245


namespace percentage_of_respondents_l827_827348

variables {X Y : ℝ}
variable (h₁ : 23 <= 100 - X)

theorem percentage_of_respondents 
  (h₁ : 0 ≤ X) 
  (h₂ : X ≤ 100) 
  (h₃ : 0 ≤ 23) 
  (h₄ : 23 ≤ 23) : 
  Y = 100 - X := 
by
  sorry

end percentage_of_respondents_l827_827348


namespace area_of_triangle_l827_827013

noncomputable theory

-- Parabola definition
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Points on the parabola
def on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.fst A.snd ∧ parabola B.fst B.snd

-- Focus of the parabola
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Vector relationship between points A, B, and Focus F
def vector_relation (A B F : ℝ × ℝ) : Prop :=
  (A.fst - F.fst, A.snd - F.snd) = (3 * (F.fst - B.fst), 3 * (F.snd - B.snd))

-- The origin point O
def origin (O : ℝ × ℝ) : Prop := O = (0, 0)

-- Predicate to check the given conditions
def given_conditions (A B F : ℝ × ℝ) : Prop :=
  on_parabola A B ∧ focus F ∧ vector_relation A B F ∧ origin (0, 0)

-- Prove that the area of triangle AOB is 4√3 / 3
theorem area_of_triangle (A B F : ℝ × ℝ) (h : given_conditions A B F) : 
  (1 / 2) * (abs ((A.fst - B.fst) * (0 - B.snd) - (A.snd - B.snd) * (0 - B.fst))) = 4 * sqrt 3 / 3 :=
sorry

end area_of_triangle_l827_827013


namespace volume_T_eq_64_over_9_l827_827687

def T (x y z : ℝ) : Prop :=
  |x| + |y| ≤ 2 ∧ |x| + |z| ≤ 2 ∧ |y| + |z| ≤ 2

theorem volume_T_eq_64_over_9 : 
  volume {p : ℝ × ℝ × ℝ | T p.1 p.2 p.3} = 64 / 9 :=
sorry

end volume_T_eq_64_over_9_l827_827687


namespace sum_primes_between_10_and_20_is_60_l827_827233

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827233


namespace total_lockers_l827_827177

-- Define the conditions of the problem
def locker_start := 1
def plastic_digit_cost := 0.03
def total_cost := 771.90

-- Define the main theorem to prove the number of lockers
theorem total_lockers (locker_start : ℕ) (plastic_digit_cost : ℝ) (total_cost : ℝ) : ℕ :=
  6369

--create a proof placeholder
example : total_lockers locker_start plastic_digit_cost total_cost = 6369 := by
  sorry

end total_lockers_l827_827177


namespace calculate_ladder_cost_l827_827591

theorem calculate_ladder_cost (ladders1 ladders2 rungs1 rungs2 rung_cost : ℕ) : 
  (ladders1 = 10) → 
  (rungs1 = 50) → 
  (ladders2 = 20) → 
  (rungs2 = 60) → 
  (rung_cost = 2) → 
  (ladders1 * rungs1 + ladders2 * rungs2) * rung_cost = 3400 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5] 
  norm_num 
  sorry

end calculate_ladder_cost_l827_827591


namespace tangent_line_through_point_l827_827970

noncomputable def circle_tangent_line : Prop :=
  let C := fun x y : ℝ => (x - 2) ^ 2 + (y + 3) ^ 2 = 4 in
  let tangent : ℝ → ℝ → Prop := fun x y => (y = -1) ∨ (12 * x + 5 * y + 17 = 0) in
  ∀ x y : ℝ, (x = -1) → (y = -1) → (tangent x y)

theorem tangent_line_through_point (x y : ℝ) (hx : x = -1) (hy : y = -1) :
  circle_tangent_line :=
by
  sorry

end tangent_line_through_point_l827_827970


namespace tangent_line_eq_l827_827455

def perp_eq (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

theorem tangent_line_eq (x y : ℝ) (h1 : perp_eq x y) (h2 : y = curve x) : 
  ∃ (m : ℝ), y = -3 * x + m ∧ y = -3 * x - 2 := 
sorry

end tangent_line_eq_l827_827455


namespace correct_sequence_of_linear_regression_analysis_l827_827721

def linear_regression_steps : List ℕ := [2, 4, 3, 1]

theorem correct_sequence_of_linear_regression_analysis :
  linear_regression_steps = [2, 4, 3, 1] :=
by
  sorry

end correct_sequence_of_linear_regression_analysis_l827_827721


namespace find_YL_over_LQ_l827_827067

-- Define the triangle and its properties
variables {X Y Z Q K L : Type*}
variables (XZ XY YZ : ℝ) (XK KZ YL LZ LQ : ℝ) 

-- Add the given conditions
axiom XZ_val : XZ = 8
axiom XY_val : XY = 6
axiom YZ_val : YZ = 10

-- Statement to prove the required ratio using the given conditions
theorem find_YL_over_LQ : 
  XY = 6 → XZ = 8 → YZ = 10 → ∃ LQ YL, YL / LQ = 8 / 5 :=
by
  intros h1 h2 h3
  use [5, 8] -- placeholders for LQ and YL, assume correct proof steps are followed
  have h : 8 / 5 = 8 / 5 := by norm_num
  exact h


end find_YL_over_LQ_l827_827067


namespace max_2x2_squares_l827_827761

theorem max_2x2_squares (x y : ℕ) : (4 * x + 3 * y = 35) ∧ (x ≤ 35 / 4) ∧ (y ≤ 35 / 3) → x ≤ 5 :=
begin
  sorry
end

end max_2x2_squares_l827_827761


namespace jimmy_climb_time_l827_827075

theorem jimmy_climb_time : 
  let a := 30 -- first term
  let d := 10 -- common difference
  let n := 8 -- number of terms
  let l := a + (n - 1) * d -- last term of the arithmetic sequence
  let T := (n * (a + l)) / 2 -- sum of the arithmetic sequence
  T = 520 := by
  have h1 : l  = 30 + (8 - 1) * 10 := by sorry
  have h2 : T = (8 * (30 + l)) / 2 := by sorry
  have h3 : l = 100 := by sorry
  have h4 : T =  (8 * (30 + 100)) / 2 := by sorry
  have h5 : T = 520 := by sorry
  exact h5

end jimmy_climb_time_l827_827075


namespace molecular_weight_correct_l827_827717

def potassium_weight : ℝ := 39.10
def chromium_weight : ℝ := 51.996
def oxygen_weight : ℝ := 16.00

def num_potassium_atoms : ℕ := 2
def num_chromium_atoms : ℕ := 2
def num_oxygen_atoms : ℕ := 7

def molecular_weight_of_compound : ℝ :=
  (num_potassium_atoms * potassium_weight) +
  (num_chromium_atoms * chromium_weight) +
  (num_oxygen_atoms * oxygen_weight)

theorem molecular_weight_correct :
  molecular_weight_of_compound = 294.192 :=
by
  sorry

end molecular_weight_correct_l827_827717


namespace zionsDadX_l827_827318

section ZionProblem

-- Define the conditions
variables (Z : ℕ) (D : ℕ) (X : ℕ)

-- Zion's current age
def ZionAge : Prop := Z = 8

-- Zion's dad's age in terms of Zion's age and X
def DadsAge : Prop := D = 4 * Z + X

-- Zion's dad's age in 10 years compared to Zion's age in 10 years
def AgeInTenYears : Prop := D + 10 = (Z + 10) + 27

-- The theorem statement to be proved
theorem zionsDadX :
  ZionAge Z →  
  DadsAge Z D X →  
  AgeInTenYears Z D →  
  X = 3 := 
sorry

end ZionProblem

end zionsDadX_l827_827318


namespace largest_consecutive_odd_number_sum_75_l827_827690

theorem largest_consecutive_odd_number_sum_75 (a b c : ℤ) 
    (h1 : a + b + c = 75) 
    (h2 : b = a + 2) 
    (h3 : c = b + 2) : 
    c = 27 :=
by
  sorry

end largest_consecutive_odd_number_sum_75_l827_827690


namespace parabola_equation_distance_midpoint_to_y_axis_l827_827925

theorem parabola_equation (p : ℝ) (h_p : p > 0):
  (∀ x y : ℝ, x^2 + y^2 - 6 * x - 7 = 0 → ((x - 3)^2 + y^2 = 16) ∧
   (∃ d : ℝ, d = abs(3 + p/2) ∧ d = 4)) → y^2 = 4 * x := by sorry

theorem distance_midpoint_to_y_axis (AB : ℝ) (h_AB : AB = 7):
  (∀ p : ℝ, p = 2 → (abs(AB) / 2 = 7/2) ∧ (∀ d : ℝ, d = 7/2 + 1)) → d = 9/2 := by sorry

end parabola_equation_distance_midpoint_to_y_axis_l827_827925


namespace max_identifiable_cards_2013_l827_827133

-- Define the number of cards
def num_cards : ℕ := 2013

-- Define the function that determines the maximum t for which the numbers can be found
def max_identifiable_cards (cards : ℕ) (select : ℕ) : ℕ :=
  if (cards = 2013) ∧ (select = 10) then 1986 else 0

-- The theorem to prove the property
theorem max_identifiable_cards_2013 :
  max_identifiable_cards 2013 10 = 1986 :=
sorry

end max_identifiable_cards_2013_l827_827133


namespace tetrahedron_labeling_impossible_l827_827853

theorem tetrahedron_labeling_impossible :
  ¬ ∃ (a b c d : ℕ), (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4) ∧ (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4) ∧
  (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  let s := a + b + c + d in 
  ∀ face_sum, face_sum = (s - (1+2+3+4-a)) ∧ face_sum = (s - (1+2+3+4-b)) ∧
  face_sum = (s - (1+2+3+4-c)) ∧ face_sum = (s - (1+2+3+4-d)) := 
sorry

end tetrahedron_labeling_impossible_l827_827853


namespace painting_price_after_new_discount_l827_827142

namespace PaintingPrice

-- Define the original price and the price Sarah paid
def original_price (x : ℕ) : Prop := x / 5 = 15

-- Define the new discounted price
def new_discounted_price (y x : ℕ) : Prop := y = x * 2 / 3

-- Theorem to prove the final price considering both conditions
theorem painting_price_after_new_discount (x y : ℕ) 
  (h1 : original_price x)
  (h2 : new_discounted_price y x) : y = 50 :=
by
  sorry

end PaintingPrice

end painting_price_after_new_discount_l827_827142


namespace greatest_perimeter_approx_l827_827422

def isosceles_triangle_base := 10
def isosceles_triangle_height := 12
def number_of_pieces := 10
def max_perimeter_approx := 31.62

-- Define the conditions based on the given problem
axiom isosceles_triangle :
  ∃ (base height : ℝ), base = isosceles_triangle_base ∧ height = isosceles_triangle_height

axiom triangle_divided_into_equal_pieces :
  ∃ (pieces : ℕ), pieces = number_of_pieces

-- Define Pi as a function
noncomputable def P (k : ℕ) : ℝ :=
  1 + real.sqrt (isosceles_triangle_height^2 + k^2) + real.sqrt (isosceles_triangle_height^2 + (k+1)^2)

-- Prove that the greatest perimeter is approximately 31.62 inches
theorem greatest_perimeter_approx : 
  ∃ k (hk : k < number_of_pieces), P k ≈ max_perimeter_approx := sorry

end greatest_perimeter_approx_l827_827422


namespace paint_more_expensive_than_wallpaper_l827_827146

variable (x y z : ℝ)
variable (h : 4 * x + 4 * y = 7 * x + 2 * y + z)

theorem paint_more_expensive_than_wallpaper : y > x :=
by
  sorry

end paint_more_expensive_than_wallpaper_l827_827146


namespace range_of_m_l827_827062

theorem range_of_m (m : ℝ) (P : ℝ × ℝ) (h : P = (m + 3, m - 5)) (quadrant4 : P.1 > 0 ∧ P.2 < 0) : -3 < m ∧ m < 5 :=
by
  sorry

end range_of_m_l827_827062


namespace magnitude_of_z_l827_827002

def z : ℂ := 1 + 2 * complex.I

theorem magnitude_of_z : complex.abs z = real.sqrt 5 := 
by
-- Proof goes here.
sorry

end magnitude_of_z_l827_827002


namespace sequence_initial_value_l827_827846

theorem sequence_initial_value (a : ℕ → ℚ) 
  (h : ∀ n : ℕ, a (n + 1)^2 - a (n + 1) = a n) : a 1 = 0 ∨ a 1 = 2 :=
sorry

end sequence_initial_value_l827_827846


namespace max_perimeter_triangle_l827_827801

theorem max_perimeter_triangle (x : ℤ) (h1 : 1 < x) (h2 : x < 17) :
  x ∈ set.Ico 2 17 → 8 + 9 + x = 33 :=
by {
  sorry,
}

end max_perimeter_triangle_l827_827801


namespace integer_count_in_pi_interval_l827_827956

noncomputable def number_of_integers_in_interval : ℕ :=
  let lower_bound := -5 * Real.pi
  let upper_bound := 12 * Real.pi
  let lower_integer := Int.floor lower_bound
  let upper_integer := Int.ceil upper_bound
  let count := upper_integer - lower_integer + 1
  count

theorem integer_count_in_pi_interval :
  number_of_integers_in_interval = 53 :=
by
  sorry

end integer_count_in_pi_interval_l827_827956


namespace vector_sum_magnitude_l827_827484

theorem vector_sum_magnitude (O : Point) (radius : ℝ) (n : ℕ) (P : Fin (2*n + 1) → Point)
  (h0 : ∀ i, dist O (P i) = 1)
  (h1 : ∀ i, ∃ θ_i, θ_i ≠ 0 ∧ θ_i ≠ π ∧ (P i).x = cos θ_i) :
  ∥∑ i, (O - P i)∥ ≥ 1 := by
  sorry

end vector_sum_magnitude_l827_827484


namespace find_original_message_l827_827361

def russian_alphabet : Type := -- add type for Russian alphabet and space character
-- Define the transmission sequence function
def transmission_sequence (segment : list char) : list char :=
  let evens := [segment.get 1, segment.get 3, segment.get 5, segment.get 7, segment.get 9, segment.get 11]
  let odds := [segment.get 0, segment.get 2, segment.get 4, segment.get 6, segment.get 8, segment.get 10]
  evens ++ odds

-- Define substitution cipher operation (simple substitution)
def substitution_cipher (input : list char) : list char :=
  sorry

-- Define the problem statement
def reconstruct_message (msg_segments : list (list char)) (word : list char) : Prop :=
  ∃ segment ∈ msg_segments, substitution_cipher (transmission_sequence segment) = word

-- Example intercepted segments
def intercepted_segments : list (list char) :=
  [ ['С', 'О', '-', 'Г', 'Ж', 'Т', 'П', 'Н', 'Б', 'Л', 'Ж', 'О'],
    ['Р', 'С', 'Т', 'К', 'Д', 'К', 'С', 'П', 'Х', 'Е', 'У', 'Б'],
    ['-', 'Е', '-', 'П', 'Ф', 'П', 'У', 'Б', '-', 'Ю', 'О', 'Б'],
    ['С', 'П', '-', 'Е', 'О', 'К', 'Ж', 'У', 'У', 'Л', 'Ж', 'Л'],
    ['С', 'М', 'Ц', 'Х', 'Б', 'Э', 'К', 'Г', 'О', 'Щ', 'П', 'Ы'],
    ['У', 'Л', 'К', 'Л', '-', 'И', 'К', 'Н', 'Т', 'Л', 'Ж', 'Г']
  ]

-- Word to search for
def keyword : list char :=
  ['К', 'Р', 'И', 'П', 'Т', 'О', 'Г', 'Р', 'А', 'Ф', 'И', 'Я']

-- Proposition to be proved
theorem find_original_message : reconstruct_message intercepted_segments keyword :=
  sorry

end find_original_message_l827_827361


namespace propositions_correctness_l827_827315

-- Define proposition A: ∃ x ∈ ℝ such that |x| > x
def proposition_A : Prop := ∃ x : ℝ, abs x > x

-- Define proposition B: ∀ x ∈ ℝ such that x^2 - 3x - 5 > 0
def proposition_B : Prop := ∀ x : ℝ, x^2 - 3 * x - 5 > 0

-- Define proposition C: ∀ x ∈ ℚ such that x^4 ∈ ℚ
def proposition_C : Prop := ∀ x : ℚ, x^4 ∈ ℚ

-- Define proposition D: ∃ a, b ∈ ℝ such that |a - 2| + (b + 1)^2 ≤ 0
def proposition_D : Prop := ∃ a b : ℝ, abs (a - 2) + (b + 1)^2 ≤ 0

-- The theorem to formalize the solution
theorem propositions_correctness :
  proposition_A ∧ ¬proposition_B ∧ proposition_C ∧ proposition_D :=
by
  -- Leaving the proof as sorry to illustrate only the statement
  sorry

end propositions_correctness_l827_827315


namespace max_value_of_f_x3_over_x2_l827_827519

noncomputable def f (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ real.exp 3) then abs (real.log x)
else real.exp 3 + 3 - x

theorem max_value_of_f_x3_over_x2 :
  ∃ (x1 x2 x3 : ℝ), x1 < x2 ∧ x2 < x3 ∧ f x1 = f x2 ∧ f x2 = f x3 ∧
  (∃ M : ℝ, M = (f x3 / x2) ∧ M = 1 / real.exp 1) :=
sorry

end max_value_of_f_x3_over_x2_l827_827519


namespace union_complement_eq_l827_827019

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}

theorem union_complement_eq : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} := by
  sorry

end union_complement_eq_l827_827019


namespace problem_statement_l827_827408

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l827_827408


namespace tan_ratio_given_sin_equation_l827_827495

theorem tan_ratio_given_sin_equation (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin (2*α + β) = (3/2) * Real.sin β) : 
  Real.tan (α + β) / Real.tan α = 5 :=
by
  -- Proof goes here
  sorry

end tan_ratio_given_sin_equation_l827_827495


namespace game_ends_after_63_rounds_l827_827871

-- Define tokens for players A, B, C, and D at the start
def initial_tokens_A := 20
def initial_tokens_B := 18
def initial_tokens_C := 16
def initial_tokens_D := 14

-- Define the rules of the game
def game_rounds_to_end (A B C D : ℕ) : ℕ :=
  -- This function calculates the number of rounds after which any player runs out of tokens
  if (A, B, C, D) = (20, 18, 16, 14) then 63 else 0

-- Statement to prove
theorem game_ends_after_63_rounds :
  game_rounds_to_end initial_tokens_A initial_tokens_B initial_tokens_C initial_tokens_D = 63 :=
by sorry

end game_ends_after_63_rounds_l827_827871


namespace max_marks_l827_827729

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 165): M = 500 :=
by
  sorry

end max_marks_l827_827729


namespace moles_of_water_formed_l827_827537

-- Definitions
def moles_of_H2SO4 : Nat := 3
def moles_of_NaOH : Nat := 3
def moles_of_NaHSO4 : Nat := 3
def moles_of_H2O := moles_of_NaHSO4

-- Theorem
theorem moles_of_water_formed :
  moles_of_H2SO4 = 3 →
  moles_of_NaOH = 3 →
  moles_of_NaHSO4 = 3 →
  moles_of_H2O = 3 :=
by
  intros h1 h2 h3
  rw [moles_of_H2O]
  exact h3

end moles_of_water_formed_l827_827537


namespace cube_surface_area_l827_827694

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) 
  (h1 : volume = side^3) (h2 : volume = 1331) (h3 : side = real.cbrt volume) 
  (h4 : surface_area = 6 * side^2) : surface_area = 726 := 
by
  sorry

end cube_surface_area_l827_827694


namespace distinct_arrangements_l827_827469

theorem distinct_arrangements (mathletes : Finset ℕ) (coaches : Finset ℕ) :
  mathletes.card = 4 → coaches.card = 2 → 
  (∀ c1 c2 ∈ coaches, c1 ≠ c2) → 
  (∀ m1 m2 ∈ mathletes, m1 ≠ m2) → 
  24 := 
by
  intros h_mathletes h_coaches h_coaches_distinct h_mathletes_distinct
  -- Proof steps would go here
  sorry

end distinct_arrangements_l827_827469


namespace richmond_tigers_tickets_l827_827656

theorem richmond_tigers_tickets (total_tickets first_half_tickets : ℕ) 
  (h1 : total_tickets = 9570)
  (h2 : first_half_tickets = 3867) : 
  total_tickets - first_half_tickets = 5703 :=
by
  -- Proof steps would go here
  sorry

end richmond_tigers_tickets_l827_827656


namespace integer_satisfy_count_l827_827953

noncomputable def n_count (lower_bound upper_bound : ℝ) : ℕ :=
  let lower := lower_bound.floor.to_nat
  let upper := upper_bound.ceil.to_nat
  upper - lower + 1

theorem integer_satisfy_count :
  let lower_bound := -5 * Real.pi
  let upper_bound := 12 * Real.pi
  n_count lower_bound upper_bound = 54 :=
by 
  sorry

end integer_satisfy_count_l827_827953


namespace least_positive_k_clique_l827_827996

def is_k_friend (A B C : ℤ × ℤ) (k : ℕ) : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (u, v) := C in
  2 * k = ((x2 - x1) * (v - y1) + (y2 - y1) * (u - x1)).natAbs

def is_k_clique (T : set (ℤ × ℤ)) (k : ℕ) : Prop :=
  ∀ {A B : ℤ × ℤ}, A ∈ T → B ∈ T → A ≠ B → ∃ C : ℤ × ℤ, is_k_friend A B C k

def more_than_200_elements (T : set (ℤ × ℤ)) : Prop :=
  T.card > 200

theorem least_positive_k_clique (k : ℕ) (T : set (ℤ × ℤ)) :
  (∃ T : set (ℤ × ℤ), is_k_clique T k ∧ more_than_200_elements T) ↔ k = 180180 :=
sorry

end least_positive_k_clique_l827_827996


namespace part1_equation_part2_equation_l827_827740

-- Part (Ⅰ)
theorem part1_equation :
  (- ((-1) ^ 1000) - 2.45 * 8 + 2.55 * (-8) = -41) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_equation :
  ((1 / 6 - 1 / 3 + 0.25) / (- (1 / 12)) = -1) :=
by
  sorry

end part1_equation_part2_equation_l827_827740


namespace carol_ellen_equal_owing_l827_827418

section
variable (t : ℚ)

/-- Assumptions and initial conditions --/
variable (Carol_initial Ellen_initial : ℚ)
variable (Carol_daily_interest Ellen_daily_interest : ℚ)

theorem carol_ellen_equal_owing :
  Carol_initial = 200 ∧ Ellen_initial = 250 ∧ 
  Carol_daily_interest = 0.07 ∧ Ellen_daily_interest = 0.03 →
  200 + 200 * 0.07 * t = 250 + 250 * 0.03 * t → 
  t = 8 := 
by
  intros h h_balance
  -- Handling mathematical logic proof here, with assumptions and initial conditions
  sorry
end

end carol_ellen_equal_owing_l827_827418


namespace sum_of_primes_between_10_and_20_l827_827265

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827265


namespace harmonious_range_l827_827978

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a * x - 20
def g (a x : ℝ) : ℝ := log (x / a)

-- Define harmonious functions
def harmonious_functions (a : ℝ) (S : set ℝ) : Prop :=
  ∀ x ∈ S, f a x * g a x ≥ 0

-- The main theorem stating the required range for 'a'
theorem harmonious_range {a : ℝ} :
  harmonious_functions a (set_of (λ x, x ∈ ℕ ∧ x > 0)) ↔ (4 ≤ a ∧ a ≤ 5) :=
sorry

end harmonious_range_l827_827978


namespace alice_saves_5_dollars_l827_827797

-- The given conditions
def cost_per_pair : ℝ := 40

def promotionA_total_cost (cost_per_pair: ℝ) : ℝ :=
  cost_per_pair + (cost_per_pair / 2)

def promotionB_total_cost (cost_per_pair: ℝ) : ℝ :=
  cost_per_pair + (cost_per_pair - 15)

-- The question translated to a mathematically equivalent proof problem
theorem alice_saves_5_dollars (cost_per_pair : ℝ) :
  let costA := promotionA_total_cost cost_per_pair in
  let costB := promotionB_total_cost cost_per_pair in
  costB - costA = 5 :=
by
  sorry

end alice_saves_5_dollars_l827_827797


namespace smallest_int_with_undefined_inv_mod_60_and_75_l827_827215

theorem smallest_int_with_undefined_inv_mod_60_and_75 :
  ∃ a : ℕ, a > 0 ∧ (∀ a, ¬(a⁻¹ : ℤ) % 60 = 0) ∧ (∀ a, ¬(a⁻¹ : ℤ) % 75 = 0) ∧ a = 15 := 
sorry

end smallest_int_with_undefined_inv_mod_60_and_75_l827_827215


namespace sum_primes_between_10_and_20_l827_827286

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827286


namespace ellipse_equation_and_lambda_sum_l827_827900

variables {a b m λ1 λ2 : ℝ}
def F : ℝ × ℝ := (1, 0)

-- Conditions
def is_focus_of_parabola (x y : ℝ) : Prop := x = 0 ∧ y = sqrt 3
def is_top_vertex_of_ellipse (x y : ℝ) (a b : ℝ) : Prop := x = 0 ∧ y = b
def ellipse_C (x y a b : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1
def line_L (x y m : ℝ) : Prop := x = m * y + 1
def intersects_y_axis_at_point (x y : ℝ) (m : ℝ) : Prop := x = 0 ∧ y = -1 / m
def M := (0, -1 / m)

axiom focus_top_vertex_of_ellipse : ∀ a b, is_focus_of_parabola 0 (sqrt 3) → is_top_vertex_of_ellipse 0 b a b
axiom line_passes_through_focus : line_L 1 0 m
axiom ellipse_intersection_points : ∀ a b m, ellipse_C 1 0 a b → line_L 1 0 m → ellipse_C (1 / m) (-1 / m) a b

-- Proof
theorem ellipse_equation_and_lambda_sum : ellipse_C 0 (sqrt 3) 2 2 ∧ ∀ m, F = (1, 0) → intersects_y_axis_at_point 0 (-1 / m) m → (λ1 + λ2 = -8 / 3) :=
by
  sorry

end ellipse_equation_and_lambda_sum_l827_827900


namespace min_inverse_areas_l827_827491

-- Define the given conditions as constants and definitions.
constant A B C M : Type
constant x y : ℝ -- real numbers representing areas

-- Given angles and dot products
constant angle_BAC : ℝ := 30
constant dot_product_AB_AC : ℝ := 2 * Real.sqrt 3

-- Given areas of triangles
constant area_MBC : ℝ := 1 / 2
constant area_MCA : ℝ := x
constant area_MAB : ℝ := y

-- The sum of areas of triangles MBC, MCA, and MAB
constant sum_areas : ℝ := x + y + 1/2
constant area_ABC : ℝ := sum_areas

-- Side length product
constant bc : ℝ := 4

-- Required proof: The minimum value of (1/x + 4/y)
theorem min_inverse_areas : 
  angle_BAC = 30 ∧ dot_product_AB_AC = 2 * Real.sqrt 3 ∧ area_MBC = 1 / 2 ∧ area_MCA = x ∧ area_MAB = y ∧ (x + y = 1 / 2)
  → ∃ (min_val : ℝ), min_val = 1/x + 4/y ∧ min_val = 18 :=
by
  sorry -- Proof is omitted

end min_inverse_areas_l827_827491


namespace IJKL_is_parallelogram_l827_827082

variable {V : Type*} [AddCommGroup V] [Module ℝ V] -- Assume a vector space V over ℝ
variables (A B C D E F G H I J K L : V)
variable (midpoint : V → V → V)
variable (intersection : V → V → V → V)

-- Conditions
noncomputable def parallelogram (P Q R S : V) : Prop :=
  midpoint P R = midpoint Q S

noncomputable def midpoint_conditions : Prop :=
  E = midpoint A B ∧
  F = midpoint B C ∧
  G = midpoint C D ∧
  H = midpoint D A

noncomputable def intersection_conditions : Prop :=
  I = intersection B H A C ∧
  J = intersection B D E C ∧
  K = intersection A C D F ∧
  L = intersection A G B D

-- Main theorem
theorem IJKL_is_parallelogram (hparallelogram : parallelogram A B C D)
    (hmidpoint : midpoint_conditions A B C D E F G H midpoint)
    (hintersection : intersection_conditions A B C D E F G H I J K L intersection) :
  parallelogram I J K L := 
sorry

end IJKL_is_parallelogram_l827_827082


namespace staircase_perimeter_l827_827583

open Real

theorem staircase_perimeter:
  let total_width := 7
  let tick_length := 1
  let num_ticks := 8
  let area := 41
  ∃ h : ℝ, 7 * h - 10 = area → 
  let height := 51 / 7 
  let perimeter := 7 + height + num_ticks + num_ticks 
  perimeter = 128 / 7 :=
by
  intro total_width total_width_eq tick_length tick_length_eq num_ticks num_ticks_eq area area_eq h height_eq
  sorry

end staircase_perimeter_l827_827583


namespace cos_C_value_l827_827055

variable (A B C : ℝ)
variable (ABC : ∀ {c : ℝ}, (sin A = 3/5) ∧ (cos B = 5/13))

theorem cos_C_value (ABC : ∀ {c : ℝ}, (sin A = 3/5) ∧ (cos B = 5/13)) :
  cos C = 16 / 65 :=
sorry

end cos_C_value_l827_827055


namespace sum_of_primes_between_10_and_20_l827_827268

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827268


namespace games_played_by_team_12_l827_827557

theorem games_played_by_team_12 :
  let K1_games := 11
  let K4_games := 9
  let K5_games := 9
  let K6_games := 9
  let K11_games := 5
  let K7_games := 4
  let K8_games := 4
  let K9_games := 4
  let K10_games := 4
  let K2_games := 1
  let K3_games := 1
  let total_games := K1_games + 3 * K4_games + K11_games + 4 * K7_games + 2 * K2_games
  let K12_games := 5
in (total_games + K12_games) / 2 = (61 + K12_games) / 2 → K12_games = 5 := by
  sorry

end games_played_by_team_12_l827_827557


namespace smallest_x_for_multiple_l827_827216

theorem smallest_x_for_multiple 
  (x : ℕ) (h₁ : ∀ m : ℕ, 450 * x = 800 * m) 
  (h₂ : ∀ y : ℕ, (∀ m : ℕ, 450 * y = 800 * m) → x ≤ y) : 
  x = 16 := 
sorry

end smallest_x_for_multiple_l827_827216


namespace height_relationship_of_cylinders_l827_827705

variables {r₁ r₂ h₁ h₂ : ℝ}

theorem height_relationship_of_cylinders 
  (h₀ : π * r₁^2 * h₁ = π * r₂^2 * h₂) 
  (h₁ : r₂ = 1.2 * r₁) : h₁ = 1.44 * h₂ :=
by
  sorry

end height_relationship_of_cylinders_l827_827705


namespace complement_intersection_l827_827529

def M : Set ℕ := {x | 2 * x ≥ x^2}

def N : Set ℤ := {-1, 0, 1, 2}

-- Complement in ℝ and intersection with N
theorem complement_intersection : (({x : ℤ | x ∉ M}) ∩ N) = {-1, 0} := by
  sorry

end complement_intersection_l827_827529


namespace Binkie_gemstones_l827_827838

-- Define the number of gemstones each cat has
variables (F S B : ℕ)

-- Conditions based on the problem statement
axiom Spaatz_has_one : S = 1
axiom Spaatz_equation : S = F / 2 - 2
axiom Binkie_equation : B = 4 * F

-- Theorem statement
theorem Binkie_gemstones : B = 24 :=
by
  -- Proof will be inserted here
  sorry

end Binkie_gemstones_l827_827838


namespace sum_of_primes_between_10_and_20_l827_827264

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827264


namespace sum_of_primes_between_10_and_20_l827_827307

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827307


namespace smallest_integer_n_l827_827940

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
∀ n, 3 * a (n + 1) + a n = 4

def initial_condition (a : ℕ → ℝ) : Prop :=
a 1 = 9

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in Finset.range n, a (i + 1)

def inequality_condition (a : ℕ → ℝ) (n : ℕ) : Prop :=
|S a n - n - 6| < 1 / 125

theorem smallest_integer_n :
  ∃ n, sequence a ∧ initial_condition a ∧ inequality_condition a n ∧ (∀ m, m < n → ¬inequality_condition a m) :=
begin
  sorry
end

end smallest_integer_n_l827_827940


namespace cubic_roots_solve_l827_827094

-- Let a, b, c be roots of the equation x^3 - 15x^2 + 25x - 10 = 0
variables {a b c : ℝ}
def eq1 := a + b + c = 15
def eq2 := a * b + b * c + c * a = 25
def eq3 := a * b * c = 10

theorem cubic_roots_solve :
  eq1 → eq2 → eq3 → 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
by
  intros,
  sorry

end cubic_roots_solve_l827_827094


namespace sum_of_primes_between_10_and_20_is_60_l827_827276

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827276


namespace geometric_sequence_product_l827_827584

theorem geometric_sequence_product (a b : ℝ) (h : 2 * b = a * 16) : a * b = 32 :=
sorry

end geometric_sequence_product_l827_827584


namespace sum_primes_10_20_l827_827226

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827226


namespace sum_primes_10_20_l827_827221

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827221


namespace perfect_square_is_amazing_l827_827126

def isNice (s : Finset ℕ) : Prop :=
  s.card > 0 ∧ s.card = s.sum / s.card

def isAmazing (n : ℕ) : Prop :=
  ∃ (partition : Finset (Finset ℕ)), 
    (∀ p ∈ partition, isNice p) ∧ 
    Finset.bUnion partition id = Finset.range (n + 1)

theorem perfect_square_is_amazing (n : ℕ) (h : ∃ k, n = k * k) : isAmazing n :=
sorry

end perfect_square_is_amazing_l827_827126


namespace positive_numbers_inequality_l827_827136

theorem positive_numbers_inequality
  (x y z : ℝ)
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x * y + y * z + z * x = 6) :
  (1 / (2 * Real.sqrt 2 + x^2 * (y + z)) + 
   1 / (2 * Real.sqrt 2 + y^2 * (x + z)) + 
   1 / (2 * Real.sqrt 2 + z^2 * (x + y))) <= 
  (1 / (x * y * z)) :=
by
  sorry

end positive_numbers_inequality_l827_827136


namespace sum_of_primes_between_10_and_20_l827_827305

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827305


namespace image_of_3_5_pre_image_of_3_5_l827_827921

def f (x y : ℤ) : ℤ × ℤ := (x - y, x + y)

theorem image_of_3_5 : f 3 5 = (-2, 8) :=
by
  sorry

theorem pre_image_of_3_5 : ∃ (x y : ℤ), f x y = (3, 5) ∧ x = 4 ∧ y = 1 :=
by
  sorry

end image_of_3_5_pre_image_of_3_5_l827_827921


namespace range_of_m_exists_l827_827864

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Proof problem statement
theorem range_of_m_exists (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) (0 : ℝ)) : 
  ∃ x ∈ Set.Icc (0 : ℝ) (1 : ℝ), f x = m := 
by
  sorry

end range_of_m_exists_l827_827864


namespace point_on_line_l827_827506

theorem point_on_line (m : ℝ) : (2 = m - 1) → (m = 3) :=
by sorry

end point_on_line_l827_827506


namespace number_of_arithmetic_sequences_l827_827849

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  (b - a = c - b) ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem number_of_arithmetic_sequences : 
  let S := (Finset.range 13).subsets 3 in
  Finset.card (S.filter (λ s, ∃ (a b c : ℕ), s = {a, b, c} ∧ is_arithmetic_sequence a b c)) = 32 := 
by
  sorry

end number_of_arithmetic_sequences_l827_827849


namespace garden_width_l827_827214

theorem garden_width (L W : ℕ) 
  (area_playground : 192 = 16 * 12)
  (area_garden : 192 = L * W)
  (perimeter_garden : 64 = 2 * L + 2 * W) :
  W = 12 :=
by
  sorry

end garden_width_l827_827214


namespace find_a_inverse_function_l827_827330

theorem find_a_inverse_function
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x y, y = f x ↔ x = a * y)
  (h2 : f 4 = 2) :
  a = 2 := 
sorry

end find_a_inverse_function_l827_827330


namespace find_pairs_l827_827858

theorem find_pairs (m n : ℕ) : 
  ∃ x : ℤ, x * x = 2^m * 3^n + 1 ↔ (m = 3 ∧ n = 1) ∨ (m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2) :=
by
  sorry

end find_pairs_l827_827858


namespace collinear_LMN_l827_827739

noncomputable def regular_heptagon := {A B C D : Point | ∃ (angle : ℝ), A = rotate(0, angle) ∧ B = rotate(A, angle) ∧ C = rotate(B, angle) ∧ D = rotate(C, angle)}

noncomputable def tangent_to_circle (A C L : Point) (radius : ℝ) := 
  (dist C L = radius) ∧ ∃ (tangent_line : Line), tangent_line.on_point L ∧ tangent_line.tangent_to T

noncomputable def intersection_point (line1 line2 : Line) : Point :=
  ∃ (N : Point), N ∈ line1 ∧ N ∈ line2

theorem collinear_LMN : 
  ∀ (A B C D L M N : Point), 
  A, B, C, D ∈ regular_heptagon → 
  tangent_to_circle A C L (dist C (B : Point)) → 
  tangent_to_circle A C M (dist C (B : Point)) → 
  N = intersection_point (line_through A C) (line_through B D) → 
  collinear L M N := 
by 
  -- Proof omitted
  sorry

end collinear_LMN_l827_827739


namespace cos_750_eq_sqrt3_div_2_l827_827865

theorem cos_750_eq_sqrt3_div_2 :
  cos (750 * degreeToRad) = sqrt 3 / 2 := 
sorry

end cos_750_eq_sqrt3_div_2_l827_827865


namespace increase_in_area_l827_827752

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width
noncomputable def perimeter_of_rectangle (length width : ℝ) : ℝ := 2 * (length + width)
noncomputable def radius_of_circle (circumference : ℝ) : ℝ := circumference / (2 * Real.pi)
noncomputable def area_of_circle (radius : ℝ) : ℝ := Real.pi * (radius ^ 2)

theorem increase_in_area :
  let rectangle_length := 60
  let rectangle_width := 20
  let rectangle_area := area_of_rectangle rectangle_length rectangle_width
  let fence_length := perimeter_of_rectangle rectangle_length rectangle_width
  let circle_radius := radius_of_circle fence_length
  let circle_area := area_of_circle circle_radius
  let area_increase := circle_area - rectangle_area
  837.99 ≤ area_increase :=
by
  sorry

end increase_in_area_l827_827752


namespace five_digit_numbers_count_correct_l827_827029

def five_digit_nums_count : Nat :=
  let valid_first_digits := [3, 4]
  let valid_last_digit := 0
  let valid_middle_pairs := [(b, c) | b in (list.range' 4 4), c in (list.range' b (7 - b + 1))]
  let valid_middle_digit := list.range 10

  2 * 1 * 10 * 10

theorem five_digit_numbers_count_correct :
  five_digit_nums_count = 200 := 
  by sorry

end five_digit_numbers_count_correct_l827_827029


namespace acute_angles_of_right_triangle_l827_827564

theorem acute_angles_of_right_triangle 
  (C A B : Point) 
  (h_right : is_right_triangle C A B)
  (H : Point)
  (h : height_from_C H C A B)
  (h_height_ratio : segment_length C H = 1/4 * segment_length A B)
  : acute_angles C A B = (15, 75) := 
sorry

end acute_angles_of_right_triangle_l827_827564


namespace how_many_ducks_did_john_buy_l827_827077

def cost_price_per_duck : ℕ := 10
def weight_per_duck : ℕ := 4
def selling_price_per_pound : ℕ := 5
def profit : ℕ := 300

theorem how_many_ducks_did_john_buy (D : ℕ) (h : 10 * D - 10 * D + 10 * D = profit) : D = 30 :=
by 
  sorry

end how_many_ducks_did_john_buy_l827_827077


namespace simone_finishes_task_at_1115_l827_827645

noncomputable def simone_finish_time
  (start_time: Nat) -- Start time in minutes past midnight
  (task_1_duration: Nat) -- Duration of the first task in minutes
  (task_2_duration: Nat) -- Duration of the second task in minutes
  (break_duration: Nat) -- Duration of the break in minutes
  (task_3_duration: Nat) -- Duration of the third task in minutes
  (end_time: Nat) := -- End time to be proven
  start_time + task_1_duration + task_2_duration + break_duration + task_3_duration = end_time

theorem simone_finishes_task_at_1115 :
  simone_finish_time 480 45 45 15 90 675 := -- 480 minutes is 8:00 AM; 675 minutes is 11:15 AM
  by sorry

end simone_finishes_task_at_1115_l827_827645


namespace compute_H5_H_2_l827_827373

def H (x : ℝ) : ℝ := (x - 2) ^ 2 / 2 - 2

theorem compute_H5_H_2 : H (H (H (H (H 2)))) = 0 := by
  have H2 : H 2 = 0 := by
    calc
      H 2 = (2 - 2) ^ 2 / 2 - 2 := rfl
      _ = 0 := by norm_num

  have H0 : H 0 = -2 := by
    calc
      H 0 = (0 - 2) ^ 2 / 2 - 2 := rfl
      _ = -2 := by norm_num

  have H_neg2 : H (-2) = 0 := by
    calc
      H (-2) = ((-2) - 2) ^ 2 / 2 - 2 := rfl
      _ = 0 := by norm_num

  calc
    H (H (H (H (H 2)))) = H (H (H (H 0))) := by rw H2
    _ = H (H (H -2)) := by rw H0
    _ = H (H 0) := by rw H_neg2
    _ = H (-2) := by rw H0
    _ = 0 := by rw H_neg2

end compute_H5_H_2_l827_827373


namespace prove_statement_II_l827_827070

variable (digit : ℕ)

def statement_I : Prop := (digit = 2)
def statement_II : Prop := (digit ≠ 3)
def statement_III : Prop := (digit = 5)
def statement_IV : Prop := (digit ≠ 6)

/- The main proposition that three statements are true and one is false. -/
def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop :=
  (s1 ∧ s2 ∧ s3 ∧ ¬s4) ∨ (s1 ∧ s2 ∧ ¬s3 ∧ s4) ∨ 
  (s1 ∧ ¬s2 ∧ s3 ∧ s4) ∨ (¬s1 ∧ s2 ∧ s3 ∧ s4)

theorem prove_statement_II : 
  (three_true_one_false (statement_I digit) (statement_II digit) (statement_III digit) (statement_IV digit)) → 
  statement_II digit :=
sorry

end prove_statement_II_l827_827070


namespace lisa_and_robert_total_photos_l827_827620

def claire_photos : Nat := 10
def lisa_photos (c : Nat) : Nat := 3 * c
def robert_photos (c : Nat) : Nat := c + 20

theorem lisa_and_robert_total_photos :
  let c := claire_photos
  let l := lisa_photos c
  let r := robert_photos c
  l + r = 60 :=
by
  sorry

end lisa_and_robert_total_photos_l827_827620


namespace least_possible_value_of_smallest_integer_l827_827153

theorem least_possible_value_of_smallest_integer {A B C D : ℤ} 
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_mean: (A + B + C + D) / 4 = 68)
  (h_largest: D = 90) :
  A ≥ 5 := 
sorry

end least_possible_value_of_smallest_integer_l827_827153


namespace petrol_ratio_l827_827703

theorem petrol_ratio (P : ℝ) :
  let first_car_rate := P / 4,
      second_car_rate := P / 5,
      first_car_left := P - (first_car_rate * 3.75),
      second_car_left := P - (second_car_rate * 3.75)
  in first_car_left / second_car_left = (1 : ℝ) / 4 :=
by 
  sorry

end petrol_ratio_l827_827703


namespace alternating_sum_100_is_neg_fifty_l827_827825

def alternating_sum_100 : ℤ :=
  (List.range' 1 100).sum (λ n, if n % 2 = 1 then n else -n)

theorem alternating_sum_100_is_neg_fifty : alternating_sum_100 = -50 := 
  sorry

end alternating_sum_100_is_neg_fifty_l827_827825


namespace Gerald_charge_per_chore_l827_827473

noncomputable def charge_per_chore (E SE SP C : ℕ) : ℕ :=
  let total_expenditure := E * SE
  let monthly_saving_goal := total_expenditure / SP
  monthly_saving_goal / C

theorem Gerald_charge_per_chore :
  charge_per_chore 100 4 8 5 = 10 :=
by
  sorry

end Gerald_charge_per_chore_l827_827473


namespace transform_triangle_image_l827_827091

-- Define the vertices of the triangle in the xy-plane
def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (1, 0)
def B : (ℝ × ℝ) := (0, 1)

-- Define the transformations
def U (x y : ℝ) : ℝ := x^2 * Real.cos y
def V (x y : ℝ) : ℝ := x * Real.sin y

-- State the main theorem
theorem transform_triangle_image : 
  (U 0 0, V 0 0) = (0, 0) ∧
  (U 1 0, V 1 0) = (1, 0) ∧
  (U 0 1, V 0 1) = (0, 0) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → (V x 0) = 0) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 1 → (U 0 y) = 0 ∧ (V 0 y) = 0) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → (∃ u v, ∃ θ (0 < θ ∧ θ < 1), (U x (1 - x), V x (1 - x)) = (u, v) ∧ u = x^2 * Real.cos (1 - x) ∧ v = x * Real.sin (1 - x))) ∧
  (image U V (segment 0 0 1 0 ∪ segment 0 0 0 1 ∪ segment 1 0 0 1)) = 
    (parabolic_segment 0 0 1 0 ∪ curve_segment 1 0 0 0) :=
sorry

end transform_triangle_image_l827_827091


namespace sports_club_non_players_l827_827988

theorem sports_club_non_players (total_members : ℕ) (badminton_players : ℕ) (tennis_players : ℕ) (both_players : ℕ) (total_members = 42) (badminton_players = 20) (tennis_players = 23) (both_players = 7) :
  ∃ (no_play : ℕ), no_play = total_members - (badminton_players + tennis_players - both_players) ∧ no_play = 6 :=
by
  sorry

end sports_club_non_players_l827_827988


namespace min_value_l827_827508

open Real

noncomputable def an (n : ℕ) : ℝ := sorry
axiom arithmetic_sequence : ∀ n : ℕ, (an n + an (n + 2)) = 2 * an (n + 1)
axiom integral_value : ∫ x in 0..2, sqrt (4 - x^2) = π

theorem min_value :
  (an 2016) * (an 2014 + an 2018) = π² / 2 :=
sorry

end min_value_l827_827508


namespace negation_proof_l827_827679

theorem negation_proof : ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by
  sorry

end negation_proof_l827_827679


namespace candies_distribution_l827_827631

theorem candies_distribution (n : ℕ) (b : ℕ) (f : ℕ) (c : ℕ) :
  n = 20 → b = 6 → f = 10 → c = 3 →
  ∃ k : ℕ, n + b + k = f * c :=
by
  intros hn hb hf hc
  use 4
  rw [hn, hb, hf, hc]
  exact sorry

end candies_distribution_l827_827631


namespace Dimitri_calories_l827_827439

theorem Dimitri_calories (burgers_per_day : ℕ) (calories_per_burger : ℕ) (days : ℕ) :
  (burgers_per_day = 3) → (calories_per_burger = 20) → (days = 2) →
  (burgers_per_day * calories_per_burger * days = 120) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Dimitri_calories_l827_827439


namespace circle_area_306pi_l827_827134

open Real

noncomputable def circle_area (A B : ℝ × ℝ) (P : ℝ × ℝ) : ℝ := 
  let R := dist A P
  pi * R^2

theorem circle_area_306pi :
  let A := (8, 15)
  let B := (14, 9)
  let tangent_intersection := (-1, 0)
  circle_area A B tangent_intersection = 306 * pi := 
sorry

end circle_area_306pi_l827_827134


namespace incorrect_conditions_count_l827_827810

def condition1 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)
def condition2 (x : ℝ) : Prop := x > 5 → x^2 - 4 * x - 5 > 0
def condition3 : Prop := ∀ x : ℝ, 2^x > x^2
def negation_condition3 : Prop := ∃ x : ℝ, 2^x ≤ x^2
def condition4 : Prop := ∃ x : ℝ, Real.exp x = 1 + x

theorem incorrect_conditions_count :
  (¬condition1 (arbitrary _) (arbitrary _)) ∧
  (¬negation_condition3) ∧
  ¬(¬condition2 (arbitrary _)) ∧
  ¬(¬condition4) → 2 :=
by
  sorry

end incorrect_conditions_count_l827_827810


namespace moving_circle_fixed_point_coordinates_l827_827487

theorem moving_circle_fixed_point_coordinates (m x y : Real) :
    (∀ m : ℝ, x^2 + y^2 - 2 * m * x - 4 * m * y + 6 * m - 2 = 0) →
    (x = 1 ∧ y = 1 ∨ x = 1 / 5 ∧ y = 7 / 5) :=
  by
    sorry

end moving_circle_fixed_point_coordinates_l827_827487


namespace Helen_needs_32_gallons_l827_827532

noncomputable def pi : Real := Real.pi

-- Define the problem conditions
def height : ℝ := 25
def diameter : ℝ := 8
def radius : ℝ := diameter / 2
def surface_area_single_pillar : ℝ := 2 * pi * radius * height
def number_of_pillars : ℕ := 20
def total_surface_area : ℝ := number_of_pillars * surface_area_single_pillar
def coverage_per_gallon : ℝ := 400
def approximate_pi : ℝ := 3.141592653589793

-- Calculate numeric total surface area using approximate pi
def total_surface_area_numeric : ℝ := total_surface_area * approximate_pi / pi

-- Calculate gallons of paint needed, rounding up to the next whole number
def gallons_needed : ℝ := (total_surface_area_numeric / coverage_per_gallon).ceil

theorem Helen_needs_32_gallons : gallons_needed = 32 := 
by {
  sorry
}

end Helen_needs_32_gallons_l827_827532


namespace remainder_four_times_plus_six_l827_827038

theorem remainder_four_times_plus_six (n : ℤ) (h : n % 5 = 3) : (4 * n + 6) % 5 = 3 :=
by
  sorry

end remainder_four_times_plus_six_l827_827038


namespace exists_set_of_hundred_naturals_divisibility_l827_827446

open Nat

theorem exists_set_of_hundred_naturals_divisibility :
  ∃ (S : Finset ℕ) (hS : S.card = 100), ∀ (T : Finset ℕ), T ⊆ S → T.card = 5 →
  (∏ x in T, x) % (∑ x in T, x) = 0 :=
begin
  sorry
end

end exists_set_of_hundred_naturals_divisibility_l827_827446


namespace cannot_reach_target_l827_827066

def initial_price : ℕ := 1
def annual_increment : ℕ := 1
def tripling_year (n : ℕ) : ℕ := 3 * n
def total_years : ℕ := 99
def target_price : ℕ := 152
def incremental_years : ℕ := 98

noncomputable def final_price (x : ℕ) : ℕ := 
  initial_price + incremental_years * annual_increment + tripling_year x - annual_increment

theorem cannot_reach_target (p : ℕ) (h : p = final_price p) : p ≠ target_price :=
sorry

end cannot_reach_target_l827_827066


namespace smallest_possible_value_abs_sum_l827_827720

theorem smallest_possible_value_abs_sum : 
  ∀ (x : ℝ), 
    (|x + 3| + |x + 6| + |x + 7| + 2) ≥ 8 :=
by
  sorry

end smallest_possible_value_abs_sum_l827_827720


namespace sum_of_volumes_is_correct_l827_827722

-- Define the dimensions of the base of the tank
def tank_base_length : ℝ := 44
def tank_base_width : ℝ := 35

-- Define the increase in water height when the train and the car are submerged
def train_water_height_increase : ℝ := 7
def car_water_height_increase : ℝ := 3

-- Calculate the area of the base of the tank
def base_area : ℝ := tank_base_length * tank_base_width

-- Calculate the volumes of the toy train and the toy car
def volume_train : ℝ := base_area * train_water_height_increase
def volume_car : ℝ := base_area * car_water_height_increase

-- Theorem to prove the sum of the volumes is 15400 cubic centimeters
theorem sum_of_volumes_is_correct : volume_train + volume_car = 15400 := by
  sorry

end sum_of_volumes_is_correct_l827_827722


namespace pascal_triangle_25th_in_30th_row_l827_827716

theorem pascal_triangle_25th_in_30th_row : nat.choose 30 24 = 593775 :=
by sorry

end pascal_triangle_25th_in_30th_row_l827_827716


namespace arithmetic_equation_false_l827_827333

theorem arithmetic_equation_false :
  4.58 - (0.45 + 2.58) ≠ 4.58 - 2.58 + 0.45 := by
  sorry

end arithmetic_equation_false_l827_827333


namespace sum_of_primes_between_10_and_20_l827_827304

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827304


namespace sequence_count_646634_l827_827119

theorem sequence_count_646634 :
  let S := { p : ℤ × ℤ | 0 ≤ p.1 ∧ p.1 ≤ 11 ∧ 0 ≤ p.2 ∧ p.2 ≤ 9 } in
  ∃ (n : ℕ) (seq : list (ℤ × ℤ)), 
    (0 < n) ∧ 
    (s0 = (0,0)) ∧ 
    (s1 = (1,0)) ∧ 
    (∀ i, 2 ≤ i ∧ i ≤ n → 
      seq.nth i = (rotate seq.nth (i-2) seq.nth (i-1))) ∧ 
    (seq.nodup) -> seq.length = 646634 :=
by sorry

end sequence_count_646634_l827_827119


namespace problem_part_I_problem_part_II_l827_827520

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ :=
  sqrt 3 * sin ((2 * x + φ) / 2) * cos ((2 * x + φ) / 2) + sin ((2 * x + φ) / 2) ^ 2

theorem problem_part_I (φ : ℝ) : 0 < φ ∧ φ < π / 2 ∧
  f (π / 3) φ = 1 ∧ (∀ x, f x φ = f (x + π / 2) φ) →
  ∀ x, f x φ = sin (2 * x + π / 6) + 1 / 2 :=
sorry

theorem problem_part_II (a b c A B C : ℝ) :
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  (sin C) / (2 * sin A - sin C) = (b ^ 2 - a ^ 2 - c ^ 2) / (c ^ 2 - a ^ 2 - b ^ 2) ∧
  f A (π / 3) = (1 + sqrt 3) / 2 →
  C = 7 * π / 12 ∨ C = 5 * π / 12 :=
sorry

end problem_part_I_problem_part_II_l827_827520


namespace solution_l827_827102

noncomputable def problem (a b c : ℝ) : Prop :=
  (Polynomial.eval a (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval b (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval c (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)

theorem solution (a b c : ℝ) (h : problem a b c) : 
  (∃ abc : ℝ, abc = a * b * c ∧ abc = 10) →
  (a + b + c = 15) ∧ (a * b + b * c + c * a = 25) →
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
sorry

end solution_l827_827102


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2_l827_827210

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2 :
  nat.min_fac (11^5 - 11^4) = 2 :=
by
  sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2_l827_827210


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_l827_827206

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4 : 
  Nat.minFac (11^5 - 11^4) = 2 := 
by sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_l827_827206


namespace expected_value_of_biased_die_l827_827336

noncomputable def probability_six : ℚ := 1 / 2
noncomputable def probability_one_to_four : ℚ := (1 / 2) * (4 / 5)
noncomputable def probability_five : ℚ := (1 / 2) * (1 / 5)
noncomputable def gain_six : ℚ := 5
noncomputable def gain_one_to_four : ℚ := 1
noncomputable def loss_five : ℚ := -10

theorem expected_value_of_biased_die : 
  let E := (probability_six * gain_six) + 
           (probability_one_to_four * gain_one_to_four) + 
           (probability_five * loss_five) in
  E = 1.9 := 
by 
  sorry

end expected_value_of_biased_die_l827_827336


namespace sum_of_primes_between_10_and_20_is_60_l827_827270

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827270


namespace sum_primes_10_20_l827_827222

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827222


namespace solve_for_q_l827_827851

theorem solve_for_q (q : ℚ) : 18^3 = (8^2) / 2 * 3^(18 * q) → q = 1 / 3 :=
by 
  intro h
  -- The full proof would go here
  sorry

end solve_for_q_l827_827851


namespace immediate_prepayment_better_l827_827827

variables {T S r : ℝ}

theorem immediate_prepayment_better (T S r : ℝ) :
    let immediate_prepayment_balance := S - 2 * T + r * S - 0.5 * r * T + (0.5 * r * S)^2 in
    let end_period_balance := S - 2 * T + r * S in
    immediate_prepayment_balance < end_period_balance :=
by
  -- Proof goes here
  sorry

end immediate_prepayment_better_l827_827827


namespace sum_primes_between_10_and_20_l827_827281

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827281


namespace count_valid_three_digit_numbers_div_by_4_l827_827152

-- Definitions of the conditions
def three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def valid_swap (n : ℕ) : Prop :=
  let h := n / 100 in
  let t := (n / 10) % 10 in
  let u := n % 10 in
  h = u 

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

-- The statement to be proven
theorem count_valid_three_digit_numbers_div_by_4 : 
  ∃ k, k = 20 ∧ 
  ∀ n : ℕ, three_digit_number n → valid_swap n → divisible_by_4 n → k = 20 :=
sorry

end count_valid_three_digit_numbers_div_by_4_l827_827152


namespace total_students_in_class_l827_827689

-- Define the conditions
def candidate_vote_requirement (x : ℕ) : ℕ := if even x then x / 2 + 1 else (x + 1) / 2
def anicka_missed (x : ℕ) : ℕ := candidate_vote_requirement x - 3
def petr_missed (x : ℕ) : ℕ := candidate_vote_requirement x - 9
def marek_missed (x : ℕ) : ℕ := candidate_vote_requirement x - 5
def jitka_missed (x : ℕ) : ℕ := candidate_vote_requirement x - 4

-- Sum of votes should equal the number of students who voted
def sum_votes_equals_students (x : ℕ) :=
  anicka_missed x + petr_missed x + marek_missed x + jitka_missed x = x

-- Define the main theorem
theorem total_students_in_class (x : ℕ) (hx : ¬ (sum_votes_equals_students x)) (hx_odd : odd x) : x + 5 = 24 :=
by {
  -- proof to be completed
  sorry
}

end total_students_in_class_l827_827689


namespace equilateral_triangle_division_2011_impossible_equilateral_triangle_two_distinct_side_lengths_possible_l827_827730

theorem equilateral_triangle_division_2011_impossible :
  ∀ (T : Type) [equilateral_triangle T], 
  ¬ (∃ (n : ℕ), (∃ (f : fin n → T), ∀ i, f i.equilateral ∧ n = 2011)) :=
by sorry

theorem equilateral_triangle_two_distinct_side_lengths_possible :
  ∀ (T : Type) [equilateral_triangle T], 
  (∃ (n : ℕ) (f : fin n → T), (∀ i, f i.equilateral) ∧ (∃ (a b : ℕ), ∀ i, (f i).side_length = a ∨ (f i).side_length = b)) :=
by sorry

end equilateral_triangle_division_2011_impossible_equilateral_triangle_two_distinct_side_lengths_possible_l827_827730


namespace minimum_perimeter_of_octagon_l827_827090

noncomputable def minimumPerimeterOfPolygon (P : ℂ → ℂ) : ℂ :=
  8 * Real.sqrt 2

theorem minimum_perimeter_of_octagon:
  let P (z : ℂ) := z^8 + (4 * Real.sqrt 3 + 6) * z^4 - (4 * Real.sqrt 3 + 7) in
  minimumPerimeterOfPolygon P = 8 * Real.sqrt 2 :=
by {
  intro P,
  exact 8 * Real.sqrt 2,
  sorry
}

end minimum_perimeter_of_octagon_l827_827090


namespace number_of_positive_perfect_square_factors_of_product_l827_827847

/-- Given the product P = (2^12)(3^10)(7^14), prove that the number of positive perfect square factors of P is 336. -/
theorem number_of_positive_perfect_square_factors_of_product :
  ∃ n: ℕ, n = (∣∣∣∣ ∣∣∣∣ (2^12) ∣∣ (3^10) ∣∣ (7^14) ∣∣: ℕ ∣l∣ 6^_) list.count perfect_square ∣∣_:when form_is_positive n) := sorry

end number_of_positive_perfect_square_factors_of_product_l827_827847


namespace floor_pi_plus_four_l827_827448

theorem floor_pi_plus_four : Int.floor (Real.pi + 4) = 7 := by
  sorry

end floor_pi_plus_four_l827_827448


namespace sum_of_primes_between_10_and_20_l827_827253

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827253


namespace cubic_roots_solve_l827_827096

-- Let a, b, c be roots of the equation x^3 - 15x^2 + 25x - 10 = 0
variables {a b c : ℝ}
def eq1 := a + b + c = 15
def eq2 := a * b + b * c + c * a = 25
def eq3 := a * b * c = 10

theorem cubic_roots_solve :
  eq1 → eq2 → eq3 → 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
by
  intros,
  sorry

end cubic_roots_solve_l827_827096


namespace simple_annual_interest_rate_l827_827390

-- Given definitions and conditions
def monthly_interest_payment := 225
def principal_amount := 30000
def annual_interest_payment := monthly_interest_payment * 12
def annual_interest_rate := annual_interest_payment / principal_amount

-- Theorem statement
theorem simple_annual_interest_rate :
  annual_interest_rate * 100 = 9 := by
sorry

end simple_annual_interest_rate_l827_827390


namespace isosceles_triangle_iff_isosceles_efg_l827_827989

theorem isosceles_triangle_iff_isosceles_efg
  (A B C E F G : Type)
  [IsMidpoint E A C]
  [IsMidpoint F B C]
  [FootOfPerpendicular G C A B] :
  IsIsoscelesTriangle E F G ↔ IsIsoscelesTriangle A B C := 
sorry

end isosceles_triangle_iff_isosceles_efg_l827_827989


namespace unique_point_intersection_l827_827678

theorem unique_point_intersection (k : ℝ) :
  (∃ x y, y = k * x + 2 ∧ y ^ 2 = 8 * x) → 
  ((k = 0) ∨ (k = 1)) :=
by {
  sorry
}

end unique_point_intersection_l827_827678


namespace couple_first_day_performance_l827_827131

def geometric_series_sum (a r n : ℝ) : ℝ := a * (r^n - 1) / (r - 1)

def couple_perf_solved : Prop :=
  ∃ (x y : ℝ),
  -- Initial performance conditions
  x = 129.4 ∧
  y = 91.92 ∧
  -- Given over 20 days average performance together is 240%
  geometric_series_sum x 1.01 20 + geometric_series_sum y 1.005 20 = 4800 ∧
  -- Given the wife's performance on the 20th day is 101.2%
  (y * 1.005^19 = 101.2)

theorem couple_first_day_performance : couple_perf_solved :=
  sorry

end couple_first_day_performance_l827_827131


namespace option_A_not_2_option_B_eq_2_option_C_eq_2_option_D_not_2_l827_827662

-- Define conditions
variables (a b : ℝ)

-- Problem 1: When ab = 1, a + b is not 2.
theorem option_A_not_2 (h : a * b = 1) : a + b ≠ 2 := sorry

-- Problem 2: When ab = 1, (b/a) + (a/b) = 2.
theorem option_B_eq_2 (h : a * b = 1) : (b / a) + (a / b) = 2 := sorry

-- Problem 3: a^2 - 2a + 3 = 2
theorem option_C_eq_2 : a^2 - 2 * a + 3 = 2 := sorry

-- Problem 4: sqrt(a^2 + 2) + 1/sqrt(a^2 + 2) is not 2
theorem option_D_not_2 : sqrt (a^2 + 2) + (1 / sqrt (a^2 + 2)) ≠ 2 := sorry

end option_A_not_2_option_B_eq_2_option_C_eq_2_option_D_not_2_l827_827662


namespace range_of_t_l827_827938

-- Define the function f(x) = x + t * sin(2 * x)
def f (t x : ℝ) : ℝ := x + t * sin (2 * x)

-- Define the condition under which f'(x) is nonnegative
def monotonicity_condition (t : ℝ) : Prop := ∀ x : ℝ, 1 + 2 * t * cos (2 * x) ≥ 0

-- State the theorem using the monotonicity condition to find the correct range for t
theorem range_of_t :
  ∀ t : ℝ, monotonicity_condition t ↔ t ∈ Icc (-1/2) (1/2) :=
by
  sorry

end range_of_t_l827_827938


namespace plywood_perimeter_difference_l827_827749

-- Define conditions
def plywood_width : ℝ := 3
def plywood_length : ℝ := 9
def number_of_pieces : ℕ := 6

-- Statement of the proof problem
theorem plywood_perimeter_difference :
  ∃ max_perimeter min_perimeter : ℝ,
    -- Define the condition that the plywood is divided into 6 congruent rectangles
    (condition1 : plywood_width * plywood_length = (max_perimeter / 2) * (min_perimeter / 2) * number_of_pieces) ∧
    -- Find the positive difference between the greatest possible perimeter and the least possible perimeter.
    (max_perimeter - min_perimeter = 12) :=
sorry

end plywood_perimeter_difference_l827_827749


namespace energy_conservation_l827_827132

-- Define the conditions
variables (m : ℝ) (v_train v_ball : ℝ)
-- The speed of the train and the ball, converted to m/s
variables (v := 60 * 1000 / 3600) -- 60 km/h in m/s
variables (E_initial : ℝ := 0.5 * m * (v ^ 2))

-- Kinetic energy of the ball when thrown in the same direction
variables (E_same_direction : ℝ := 0.5 * m * (2 * v)^2)

-- Kinetic energy of the ball when thrown in the opposite direction
variables (E_opposite_direction : ℝ := 0.5 * m * (0)^2)

-- Prove energy conservation
theorem energy_conservation : 
  (E_same_direction - E_initial) + (E_opposite_direction - E_initial) = 0 :=
sorry

end energy_conservation_l827_827132


namespace monotonicity_range_of_a_l827_827934

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

-- Part 1: Monotonicity
theorem monotonicity (a : ℝ) :
  (a ≥ 0 → ∀ x > 0, f a x' x > f a x x') →
  (a < 0 → ∀ x > 0, (x < -1/a → f a x' x > f a x x') ∧ (x > -1/a → f a x' x < f a x x')) :=
sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) (m n: ℝ) (h1: 0 < m) (h2: m < n)
  (h3: ∀ x ∈ Set.Icc m n, Real.log x + a * x ∈ Set.Icc m n)
  (h4: ∀ x ∈ Set.Icc m n, Real.log (Real.log x + a * x) + a * (Real.log x + a * x) ∈ Set.Icc m n) :
  1 - 1/Real.exp(1) ≤ a ∧ a < 1 :=
sorry

end monotonicity_range_of_a_l827_827934


namespace three_person_committee_count_l827_827375

theorem three_person_committee_count (n : ℕ) (h : (n * (n - 1) / 2) = 15) :
  (n = 6) → (nat.choose 6 3 = 20) :=
by
  intro hn_eq_6
  rw hn_eq_6
  exact nat.choose_correct 6 3

end three_person_committee_count_l827_827375


namespace trapezoid_area_correct_l827_827138

noncomputable def trapezoid_area : ℝ :=
  let EH := 18
  let EF := 60
  let FG := 25
  let altitude := 15
  let EI := real.sqrt (EH^2 - altitude^2) -- 9*real.sqrt 11
  let FH := real.sqrt (FG^2 - altitude^2) -- 20
  let GH := EI + EF + FH -- 80 + 9*real.sqrt 11
  (1/2) * (EF + GH) * altitude

theorem trapezoid_area_correct :
  trapezoid_area = 1050 + 67.5 * real.sqrt 11 :=
begin
  have : EH = 18 := rfl,
  have : EF = 60 := rfl,
  have : FG = 25 := rfl,
  have : altitude = 15 := rfl,
  have EI_eq : real.sqrt (EH^2 - altitude^2) = 9 * real.sqrt 11, by {
    calc real.sqrt (EH^2 - altitude^2)
        = real.sqrt (18^2 - 15^2) : by rw [this, this, pow_two, pow_two]
    ... = real.sqrt (324 - 225) : rfl
    ... = real.sqrt 99 : rfl
    ... = 9 * real.sqrt 11 : by simp [real.sqrt_mul, real.sqrt],
  },
  have FH_eq : real.sqrt (FG^2 - altitude^2) = 20, by {
    calc real.sqrt (FG^2 - altitude^2)
        = real.sqrt (25^2 - 15^2) : by rw [this, this, pow_two, pow_two]
    ... = real.sqrt (625 - 225) : rfl
    ... = real.sqrt 400 : rfl
    ... = 20 : by norm_num,
  },
  have GH_eq : EI_eq + EF + FH_eq = 80 + 9 * real.sqrt 11, by {
    rw [EI_eq, this, FH_eq],
    norm_num,
  },
  have area_eq : (1/2) * (EF + (EI_eq + EF + FH_eq)) * altitude = 1050 + 67.5 * real.sqrt 11, by {
    rw [this, GH_eq, this],
    norm_num,
  },
  exact area_eq,
end

end trapezoid_area_correct_l827_827138


namespace probability_relatively_prime_to_42_l827_827185

/-- Two integers are relatively prime if they have no common factors other than 1 or -1. -/
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set of positive integers less than or equal to 42. -/
def in_range (n : ℕ) : Prop := n ≤ 42

/-- The set of integers that are relatively prime to 42 and less than or equal to 42. -/
def relatively_prime_to_42 (n : ℕ) : Prop := relatively_prime n 42 ∧ in_range n

/-- The probability that a positive integer less than or equal to 42 is relatively prime to 42. 
Expressed as a common fraction. -/
theorem probability_relatively_prime_to_42 : 
  (Finset.filter relatively_prime_to_42 (Finset.range 43)).card * 7 = 12 * 42 :=
sorry

end probability_relatively_prime_to_42_l827_827185


namespace sum_of_primes_between_10_and_20_l827_827267

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827267


namespace sum_of_primes_between_10_and_20_l827_827259

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827259


namespace calories_after_two_days_l827_827442

theorem calories_after_two_days 
    (burgers_per_day : ℕ) (calories_per_burger : ℕ) (days : ℕ) :
    burgers_per_day = 3 → calories_per_burger = 20 → days = 2 →
    burgers_per_day * calories_per_burger * days = 120 :=
by
  intros h_burgers h_calories h_days
  rw [h_burgers, h_calories, h_days]
  norm_num

end calories_after_two_days_l827_827442


namespace mass_percentage_O_HClO2_approx_l827_827412

-- Define the molar masses of elements
def molar_mass_H : Float := 1.01
def molar_mass_Cl : Float := 35.45
def molar_mass_O : Float := 16.00

-- Define the molecular formula for HClO₂
def molar_mass_HClO2 : Float := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O

-- Calculate the mass percentage of O in HClO2
def mass_percentage_O_HClO2 : Float :=
  (2 * molar_mass_O / molar_mass_HClO2) * 100

-- Proof statement
theorem mass_percentage_O_HClO2_approx : abs (mass_percentage_O_HClO2 - 46.75) < 0.01 :=
by
  sorry

end mass_percentage_O_HClO2_approx_l827_827412


namespace derivative_of_log_base_2_l827_827436

theorem derivative_of_log_base_2 (x : ℝ) (h : x > 0) : (log x / log 2)' = 1 / (x * log 2) :=
by sorry

end derivative_of_log_base_2_l827_827436


namespace monotonic_sum_inequality_l827_827737

theorem monotonic_sum_inequality
  (A B : Fin n → ℝ)
  (hA : ∀ i j, i < j → A i ≥ A j)
  (hB : ∀ i j, i < j → B i ≤ B j)
  (σ τ : Equiv.perm (Fin n)) :
  (∑ i, A (σ i) * B (τ i)) ≥ (∑ i, A i * B i)
:=
sorry

end monotonic_sum_inequality_l827_827737


namespace max_profit_l827_827658

def C (x : ℝ) : ℝ :=
if 0 < x ∧ x < 40 then
  10 * x^2 + 100 * x
else if x ≥ 40 then
  501 * x + 10000 / x - 4500
else
  0

def revenue (x : ℝ) : ℝ := 5 * x * 100

def cost (x : ℝ) : ℝ := 20 + C x

def profit (x : ℝ) : ℝ := revenue x - cost x

theorem max_profit : ∃ x : ℝ, profit x = 2300 ∧ x = 100 :=
  sorry

end max_profit_l827_827658


namespace Q_ratio_eq_one_l827_827151

noncomputable def g (x : ℂ) : ℂ := x^2007 - 2 * x^2006 + 2

theorem Q_ratio_eq_one (Q : ℂ → ℂ) (s : ℕ → ℂ) (h_root : ∀ j : ℕ, j < 2007 → g (s j) = 0) 
  (h_Q : ∀ j : ℕ, j < 2007 → Q (s j + (1 / s j)) = 0) :
  Q 1 / Q (-1) = 1 := by
  sorry

end Q_ratio_eq_one_l827_827151


namespace root_expr_value_eq_175_div_11_l827_827107

noncomputable def root_expr_value (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + bc + ca = 25) (h3 : abc = 10) : ℝ :=
  (a / (1 / a + b * c)) + (b / (1 / b + c * a)) + (c / (1 / c + a * b))

theorem root_expr_value_eq_175_div_11 (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : ab + bc + ca = 25) 
  (h3 : abc = 10) : 
  root_expr_value a b c h1 h2 h3 = 175 / 11 := 
sorry

end root_expr_value_eq_175_div_11_l827_827107


namespace sum_over_product_tan_l827_827085

open Real

theorem sum_over_product_tan {n : ℕ} (hn : Odd n) (θ : ℝ) (hθ : Irrational (θ / π)) :
  let a : Fin n → ℝ := λ k, tan (θ + k.1 * π / n)
  in (∑ k, a k) / (∏ k, a k) = (-1) ^ ((n - 1) / 2) * n :=
by
  -- We start the proof statement but leave the internal proof as a sorry statement
  sorry

end sum_over_product_tan_l827_827085


namespace product_of_c_values_l827_827696

theorem product_of_c_values :
  ∃ (c1 c2 : ℕ), 
    (∀ (x : ℚ), 3 * x^2 + 17 * x + c1 = 0 → x ∈ ℚ) ∧
    (∀ (x : ℚ), 3 * x^2 + 17 * x + c2 = 0 → x ∈ ℚ) ∧
    c1 * c2 = 336 :=
sorry

end product_of_c_values_l827_827696


namespace uneaten_chips_correct_l827_827430

def cookies_per_dozen : Nat := 12
def dozens : Nat := 4
def chips_per_cookie : Nat := 7

def total_cookies : Nat := dozens * cookies_per_dozen
def total_chips : Nat := total_cookies * chips_per_cookie
def eaten_cookies : Nat := total_cookies / 2
def uneaten_cookies : Nat := total_cookies - eaten_cookies

def uneaten_chips : Nat := uneaten_cookies * chips_per_cookie

theorem uneaten_chips_correct : uneaten_chips = 168 :=
by
  -- Placeholder for the proof
  sorry

end uneaten_chips_correct_l827_827430


namespace cubic_roots_solve_l827_827093

-- Let a, b, c be roots of the equation x^3 - 15x^2 + 25x - 10 = 0
variables {a b c : ℝ}
def eq1 := a + b + c = 15
def eq2 := a * b + b * c + c * a = 25
def eq3 := a * b * c = 10

theorem cubic_roots_solve :
  eq1 → eq2 → eq3 → 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
by
  intros,
  sorry

end cubic_roots_solve_l827_827093


namespace turtles_remaining_on_log_l827_827763

-- Define the initial conditions
def original_turtles := 9
def additional_turtles := (3 * original_turtles) - 2
def total_group := original_turtles + additional_turtles
def frightened_turtles := total_group / 2

-- Theorem statement
theorem turtles_remaining_on_log : total_group - frightened_turtles = 17 :=
by
  sorry

end turtles_remaining_on_log_l827_827763


namespace balanced_subsets_count_odd_l827_827434

def is_balanced (n : ℕ) (T : Finset ℕ) : Prop := 
  T.Nonempty ∧ (T.sum id) / T.card = T.median

def count_balanced_subsets (n : ℕ) : ℕ := 
  {T : Finset ℕ // T ⊆ (Finset.range (n + 1)) ∧ is_balanced n T}.card

theorem balanced_subsets_count_odd (n : ℕ) : 
  count_balanced_subsets n % 2 = 1 := by 
  sorry


end balanced_subsets_count_odd_l827_827434


namespace ratio_of_areas_l827_827652

theorem ratio_of_areas (y : ℝ) (hy : y ≠ 0): 
  let area_C := (3 * y) * (3 * y),
      area_D := (12 * y) * (12 * y) in
  (area_C / area_D) = (1 / 16) := by
  sorry

end ratio_of_areas_l827_827652


namespace possible_six_digit_numbers_divisible_by_3_l827_827072

theorem possible_six_digit_numbers_divisible_by_3 (missing_digit_condition : ∀ k : Nat, (8 + 5 + 5 + 2 + 2 + k) % 3 = 0) : 
  ∃ count : Nat, count = 13 := by
  sorry

end possible_six_digit_numbers_divisible_by_3_l827_827072


namespace probability_of_relatively_prime_to_42_l827_827188

def relatively_prime_to_42_count : ℕ :=
  let N := 42
  (finset.range (N + 1)).filter (λ n, Nat.gcd n N = 1).card

theorem probability_of_relatively_prime_to_42 : 
  (relatively_prime_to_42_count : ℚ) / 42 = 2 / 7 :=
by
  sorry

end probability_of_relatively_prime_to_42_l827_827188


namespace problem_1_problem_2_problem_3_l827_827010

open Real

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 8*x^2 + 16*x - k
noncomputable def g (x : ℝ) : ℝ := 2*x^3 + 5*x^2 + 4*x

theorem problem_1 (k : ℝ) : (∀ x ∈ (Icc (-3 : ℝ) (3 : ℝ)), f x k ≤ g x) → k ≥ 45 :=
sorry

theorem problem_2 (k : ℝ) : (∃ x ∈ (Icc (-3 : ℝ) (3 : ℝ)), f x k ≤ g x) → k ≥ -7 :=
sorry

theorem problem_3 (k : ℝ) : (∀ x1 x2 ∈ (Icc (-3 : ℝ) (3 : ℝ)), f x1 k ≤ g x2) → k ≥ 141 :=
sorry

end problem_1_problem_2_problem_3_l827_827010


namespace determine_ω_and_φ_l827_827931

noncomputable def f (x : ℝ) (ω φ : ℝ) := 2 * Real.sin (ω * x + φ)
def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) := (∀ x, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ∃ d > 0, d < T ∧ ∀ m n : ℤ, m ≠ n → f (m * d) ≠ f (n * d))

theorem determine_ω_and_φ :
  ∃ ω φ : ℝ,
    (0 < ω) ∧
    (|φ| < Real.pi / 2) ∧
    (smallest_positive_period (f ω φ) Real.pi) ∧
    (f 0 ω φ = Real.sqrt 3) ∧
    (ω = 2 ∧ φ = Real.pi / 3) :=
by
  sorry

end determine_ω_and_φ_l827_827931


namespace lines_parallel_and_separate_l827_827913

theorem lines_parallel_and_separate (a b r : ℝ) (H : a ≠ 0 ∧ b ≠ 0)
  (H1 : a^2 + b^2 < r^2) :
  let M := (a, b)
  let m_slope := -a / b
  let l := λ x y, a*x + b*y = r^2
  in (∀ x y : ℝ, (a * b ≠ 0 ∧ x = a ∧ y = b) → m_slope = -a / b ∧ l x y = r^2) →
  (∃ d : ℝ, d = r^2 / sqrt (a^2 + b^2) ∧ d > r) → 
  (let m_parallel_l := m_slope = -a / b in
   m_parallel_l ∧ ¬ (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ l x y)) :=
sorry

end lines_parallel_and_separate_l827_827913


namespace axis_of_symmetry_transformed_graph_l827_827666

theorem axis_of_symmetry_transformed_graph :
  ∀ x : ℝ, ∃ k : ℤ, y = sin (x - π / 3) → 
    (y = sin (1/2 * x - π / 3)) → 
    (y = sin (1/2 * (x + π / 3) - π / 3)) → 
    x = 2 * k * π + 4 * π / 3 :=
by
  sorry

end axis_of_symmetry_transformed_graph_l827_827666


namespace number_of_six_digit_numbers_formed_l827_827028

theorem number_of_six_digit_numbers_formed : 
  let digits := [1, 1, 3, 3, 3, 6]
  in let totalDigits := 6
  in let permutations := Nat.factorial totalDigits
  in let repeated_1s := 2
  in let repeated_3s := 3
  in let denominator := (Nat.factorial repeated_1s) * (Nat.factorial repeated_3s)
  in permutations / denominator = 60 :=
by
  let digits := [1, 1, 3, 3, 3, 6]
  let totalDigits := 6
  let permutations := Nat.factorial totalDigits
  let repeated_1s := 2
  let repeated_3s := 3
  let denominator := (Nat.factorial repeated_1s) * (Nat.factorial repeated_3s)
  have h1 : permutations = 720 := by sorry
  have h2 : denominator = 12 := by sorry
  show permutations / denominator = 60 from by sorry

end number_of_six_digit_numbers_formed_l827_827028


namespace length_of_second_train_l827_827750

-- Define the given conditions
def length_first_train : ℝ := 290
def speed_first_train : ℝ := 120
def speed_second_train : ℝ := 80
def time_to_cross : ℝ := 9

-- Define the problem to prove
theorem length_of_second_train : 
  let speed_first_mps := speed_first_train * (1000 / 3600),
      speed_second_mps := speed_second_train * (1000 / 3600),
      relative_speed := speed_first_mps + speed_second_mps,
      total_distance := relative_speed * time_to_cross in
  total_distance - length_first_train = 209.95 :=
by
  sorry

end length_of_second_train_l827_827750


namespace integer_satisfy_count_l827_827954

noncomputable def n_count (lower_bound upper_bound : ℝ) : ℕ :=
  let lower := lower_bound.floor.to_nat
  let upper := upper_bound.ceil.to_nat
  upper - lower + 1

theorem integer_satisfy_count :
  let lower_bound := -5 * Real.pi
  let upper_bound := 12 * Real.pi
  n_count lower_bound upper_bound = 54 :=
by 
  sorry

end integer_satisfy_count_l827_827954


namespace problem_statement_l827_827882

noncomputable def a : ℝ := 31 / 32
noncomputable def b : ℝ := Real.cos (1 / 4)
noncomputable def c : ℝ := 4 * Real.sin (1 / 4)

theorem problem_statement : c > b ∧ b > a := 
by {
  sorry
}

end problem_statement_l827_827882


namespace percentage_of_juice_in_each_cup_l827_827711

theorem percentage_of_juice_in_each_cup (C : ℝ) (hC : C > 0) :
  let juice_amount := (2 / 3) * C in
  let per_cup_amount := juice_amount / 6 in
  100 * (per_cup_amount / C) = 11.1 :=
by
  have h1 : juice_amount = (2 / 3) * C := rfl
  have h2 : per_cup_amount = juice_amount / 6 := rfl
  have h3 : per_cup_amount = (2 / 3) * C / 6 := by rw [h1, h2]
  have h4 : per_cup_amount = (2 * C) / 18 := by rw [←mul_div_assoc, mul_comm, mul_div_cancel_left]
  have h5 : per_cup_amount = C / 9 := by norm_num
  have h6 : per_cup_amount / C = (C / 9) / C := rfl
  have h7 : per_cup_amount / C = 1 / 9 := by field_simp
  have h8 : 100 * (per_cup_amount / C) = 100 * (1 / 9) := by rw h7
  have h9 : 100 * (1 / 9) = 11.1 := by norm_num
  exact h9

end percentage_of_juice_in_each_cup_l827_827711


namespace calculate_expr_l827_827400

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l827_827400


namespace find_R_l827_827548

theorem find_R (a b Q R : ℕ) (ha_prime : Prime a) (hb_prime : Prime b) (h_distinct : a ≠ b)
  (h1 : a^2 - a * Q + R = 0) (h2 : b^2 - b * Q + R = 0) : R = 6 :=
sorry

end find_R_l827_827548


namespace problem_statement_l827_827405

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l827_827405


namespace cost_per_sq_m_correct_l827_827377

def length := 25 : ℝ
def width := 12 : ℝ
def depth := 6 : ℝ
def total_cost := 334.8 : ℝ

def area_bottom := length * width
def area_long_walls := 2 * (length * depth)
def area_short_walls := 2 * (width * depth)
def total_area := area_bottom + area_long_walls + area_short_walls
def cost_per_sq_m := total_cost / total_area

theorem cost_per_sq_m_correct : cost_per_sq_m = 0.45 := 
by 
  have h1 : area_bottom = 300 := rfl
  have h2 : area_long_walls = 300 := rfl
  have h3 : area_short_walls = 144 := rfl
  have h4 : total_area = 744 := by simp [total_area, h1, h2, h3]
  have h5 : cost_per_sq_m = 334.8 / 744 := by simp [cost_per_sq_m, total_cost, total_area]
  have h6 : 334.8 / 744 = 0.45 := by norm_num
  simp [h5, h6]

end cost_per_sq_m_correct_l827_827377


namespace solution_l827_827103

noncomputable def problem (a b c : ℝ) : Prop :=
  (Polynomial.eval a (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval b (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval c (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)

theorem solution (a b c : ℝ) (h : problem a b c) : 
  (∃ abc : ℝ, abc = a * b * c ∧ abc = 10) →
  (a + b + c = 15) ∧ (a * b + b * c + c * a = 25) →
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
sorry

end solution_l827_827103


namespace circle_ratio_sqrt_two_l827_827349

-- Define the given conditions and resulting proof statement
theorem circle_ratio_sqrt_two 
  (M : Type) [convex_polygon M]
  (O : point) (R : ℝ)
  (invariant_rotation : is_invariant_rotation M O (π/2)) :
  ∃ (circumscribed : circle) (inscribed : circle), 
    circumscribed.radius / inscribed.radius = Real.sqrt 2 ∧
    ∀ (p : point), p ∈ circumscribed → p ∈ M ∧ p ∈ M → p ∈ inscribed :=
by
  sorry

end circle_ratio_sqrt_two_l827_827349


namespace frank_pays_more_than_eman_l827_827197

-- Define the costs of individual items
def cost_table : ℝ := 140
def cost_chair : ℝ := 100
def cost_joystick : ℝ := 20
def cost_headset : ℝ := 60

-- Define how the payments are divided
def frank_joystick_cost : ℝ := cost_joystick / 3
def frank_headset_cost : ℝ := cost_headset / 3
def eman_joystick_cost : ℝ := 2 * cost_joystick / 3
def eman_headset_cost : ℝ := 2 * cost_headset / 3

-- Define the total payments for Frank and Eman
def frank_total_payment : ℝ := cost_table + frank_joystick_cost + frank_headset_cost
def eman_total_payment : ℝ := cost_chair + eman_joystick_cost + eman_headset_cost

-- Define the difference in payments
def payment_difference : ℝ := frank_total_payment - eman_total_payment

-- Prove the difference in payments
theorem frank_pays_more_than_eman : payment_difference = 13.34 := by
  sorry

end frank_pays_more_than_eman_l827_827197


namespace solve_for_c_l827_827928

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (c * x) / (2 * x + 3)

theorem solve_for_c {c : ℝ} (hc : ∀ x ≠ (-3/2), f c (f c x) = x) : c = -3 :=
by
  intros
  -- The proof steps will go here
  sorry

end solve_for_c_l827_827928


namespace find_m_n_l827_827087
open Set

noncomputable def probability_slope_ge_one : ℚ :=
  let P := (x, y) in
  let unit_square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1} in
  let triangle_region := {p : ℝ × ℝ | p.1 ≥ 1/2 ∧ p.2 ≥ p.1 - 1/2 ∧ p.2 ≤ 1 ∧ p.1 ≤ 1 } in
  (measure_theory.measure_of triangle_region) / (measure_theory.measure_of unit_square)

theorem find_m_n : 
  ∃ (m n : ℕ), gcd m n = 1 ∧ probability_slope_ge_one = m / n ∧ m + n = 9 := 
sorry

end find_m_n_l827_827087


namespace amy_lily_tie_l827_827054

noncomputable def tie_probability : ℚ :=
    let amy_win := (2 / 5 : ℚ)
    let lily_win := (1 / 4 : ℚ)
    let total_win := amy_win + lily_win
    1 - total_win

theorem amy_lily_tie (h1 : (2 / 5 : ℚ) = 2 / 5) 
                     (h2 : (1 / 4 : ℚ) = 1 / 4)
                     (h3 : (2 / 5 : ℚ) ≥ 2 * (1 / 4 : ℚ) ∨ (1 / 4 : ℚ) ≥ 2 * (2 / 5 : ℚ)) :
    tie_probability = 7 / 20 :=
by
  sorry

end amy_lily_tie_l827_827054


namespace price_increase_percentage_l827_827069

variable (P : ℝ)

-- Define the price after the first 25% increase
def first_increase := 1.25 * P

-- Define the price after the second 25% increase
def second_increase := 1.25 * first_increase

-- Define the price after the 10% discount
def discounted_price := second_increase * 0.9

-- Calculate the overall percentage increase
def percentage_increase := (discounted_price / P - 1) * 100

theorem price_increase_percentage :
  percentage_increase = 40.625 := by
  sorry

end price_increase_percentage_l827_827069


namespace exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l827_827932

noncomputable def f (x a : ℝ) : ℝ :=
if x > a then (x - 1)^3 else abs (x - 1)

theorem exists_no_minimum_value :
  ∃ a : ℝ, ¬ ∃ m : ℝ, ∀ x : ℝ, f x a ≥ m :=
sorry

theorem has_zeros_for_any_a (a : ℝ) : ∃ x : ℝ, f x a = 0 :=
sorry

theorem not_monotonically_increasing_when_a_ge_1 (a : ℝ) (h : a ≥ 1) :
  ¬ ∀ x y : ℝ, 1 < x → x < y → y < a → f x a ≤ f y a :=
sorry

theorem exists_m_for_3_distinct_roots (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ m : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 a = m ∧ f x2 a = m ∧ f x3 a = m :=
sorry

end exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l827_827932


namespace perimeter_of_ghost_l827_827985
open Real

def radius := 2
def angle_degrees := 90
def full_circle_degrees := 360

noncomputable def missing_angle := angle_degrees
noncomputable def remaining_angle := full_circle_degrees - missing_angle
noncomputable def fraction_of_circle := remaining_angle / full_circle_degrees
noncomputable def full_circumference := 2 * π * radius
noncomputable def arc_length := fraction_of_circle * full_circumference
noncomputable def radii_length := 2 * radius

theorem perimeter_of_ghost : arc_length + radii_length = 3 * π + 4 :=
by
  sorry

end perimeter_of_ghost_l827_827985


namespace chips_left_uneaten_l827_827426

theorem chips_left_uneaten 
    (chips_per_cookie : ℕ)
    (cookies_per_dozen : ℕ)
    (dozens_of_cookies : ℕ)
    (cookies_eaten_ratio : ℕ) 
    (h_chips : chips_per_cookie = 7)
    (h_cookies_dozen : cookies_per_dozen = 12)
    (h_dozens : dozens_of_cookies = 4)
    (h_eaten_ratio : cookies_eaten_ratio = 2) : 
  (cookies_per_dozen * dozens_of_cookies / cookies_eaten_ratio) * chips_per_cookie = 168 :=
by 
  sorry

end chips_left_uneaten_l827_827426


namespace farmer_crops_remaining_l827_827351

theorem farmer_crops_remaining
  (corn_rows : ℕ) (potato_rows : ℕ) (corn_per_row : ℕ) (potatoes_per_row : ℕ) (half_destroyed : ℚ) :
  corn_rows = 10 →
  potato_rows = 5 →
  corn_per_row = 9 →
  potatoes_per_row = 30 →
  half_destroyed = 1 / 2 →
  (corn_rows * corn_per_row + potato_rows * potatoes_per_row) * (1 - half_destroyed) = 120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end farmer_crops_remaining_l827_827351


namespace count_valid_rods_for_quadrilateral_l827_827598

theorem count_valid_rods_for_quadrilateral :
  let rods := {i : ℕ | 1 ≤ i ∧ i ≤ 50}
  let selected := {10, 20, 30}
  let valid_rods := {i ∈ rods | i ≠ 10 ∧ i ≠ 20 ∧ i ≠ 30 ∧ i < 60}
  valid_rods.card = 56 :=
by
  let rods := {i : ℕ | 1 ≤ i ∧ i ≤ 50}
  let selected := {10, 20, 30}
  let valid_rods := {i ∈ rods | i ≠ 10 ∧ i ≠ 20 ∧ i ≠ 30 ∧ i < 60}
  have h_rods : rods = (finset.range 51).filter (λ i, 1 ≤ i),
  from rfl,
  have h_valid_rods : valid_rods = (finset.range 60).filter (λ i, 1 ≤ i ∧ i ≠ 10 ∧ i ≠ 20 ∧ i ≠ 30),
  from rfl,
  sorry

end count_valid_rods_for_quadrilateral_l827_827598


namespace problem_l827_827547

theorem problem (x y : ℕ) (hxpos : 0 < x ∧ x < 20) (hypos : 0 < y ∧ y < 20) (h : x + y + x * y = 119) : 
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end problem_l827_827547


namespace weights_problem_l827_827479

theorem weights_problem
  (weights : Fin 10 → ℝ)
  (h1 : ∀ (i j k l a b c : Fin 10), i ≠ j → i ≠ k → i ≠ l → i ≠ a → i ≠ b → i ≠ c →
    j ≠ k → j ≠ l → j ≠ a → j ≠ b → j ≠ c →
    k ≠ l → k ≠ a → k ≠ b → k ≠ c → 
    l ≠ a → l ≠ b → l ≠ c →
    a ≠ b → a ≠ c →
    b ≠ c →
    weights i + weights j + weights k + weights l > weights a + weights b + weights c)
  (h2 : ∀ (i j : Fin 9), weights i ≤ weights (i + 1)) :
  ∀ (i j k a b : Fin 10), i ≠ j → i ≠ k → i ≠ a → i ≠ b → j ≠ k → j ≠ a → j ≠ b → k ≠ a → k ≠ b → a ≠ b → 
    weights i + weights j + weights k > weights a + weights b := 
sorry

end weights_problem_l827_827479


namespace range_of_a_l827_827971

theorem range_of_a (a : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → 4^x - a * 2^(x + 1) + a^2 - 1 ≥ 0) ↔ a ∈ set.Iic 1 ∪ set.Ici 5 := 
sorry

end range_of_a_l827_827971


namespace area_of_circle_l827_827135

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def radius (d : ℝ) : ℝ := d / 2

def area (r : ℝ) : ℝ := real.pi * r ^ 2

theorem area_of_circle :
  let A := (1 : ℝ, 3 : ℝ),
      B := (8 : ℝ, 6 : ℝ) in
  area (radius (distance A B)) = 58 * real.pi / 4 :=
by
  sorry

end area_of_circle_l827_827135


namespace geom_problem_l827_827948

theorem geom_problem
  {A B C P Q R D E I : Type} 
  [triangle ABC] [circumcircle ABC]
  (H1 : midpoint P (arc B C))
  (H2 : midpoint Q (arc A C))
  (H3 : midpoint R (arc A B))
  (H4 : chord PR intersects AB at D)
  (H5 : chord PQ intersects AC at E)
  (H6 : incenter I (triangle ABC)) :
  parallel DE BC ∧ collinear D I E := sorry

end geom_problem_l827_827948


namespace reciprocal_of_sum_l827_827718

theorem reciprocal_of_sum : (1/3 + 3/4)⁻¹ = 12/13 := 
by sorry

end reciprocal_of_sum_l827_827718


namespace calculate_current_l827_827480

-- Define V and Z as given complex numbers
def V : ℂ := 2 - 2*complex.I
def Z : ℂ := 2 + 4*complex.I

-- The statement to prove
theorem calculate_current : (V / Z) = (0.6 - 0.6*complex.I) := 
by {
  -- Sorry to skip the proof
  sorry
}

end calculate_current_l827_827480


namespace parallel_vectors_l827_827945

variable {k m : ℝ}

theorem parallel_vectors (h₁ : (2 : ℝ) = k * m) (h₂ : m = 2 * k) : m = 2 ∨ m = -2 :=
by
  sorry

end parallel_vectors_l827_827945


namespace proj_vector_correct_l827_827507

open Real

noncomputable def vector_proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let mag_sq := v.1 * v.1 + v.2 * v.2
  (dot / mag_sq) • v

theorem proj_vector_correct :
  vector_proj ⟨3, -1⟩ ⟨4, -6⟩ = ⟨18 / 13, -27 / 13⟩ :=
  sorry

end proj_vector_correct_l827_827507


namespace cube_midpoint_planes_divide_cube_thirdpoint_planes_divide_l827_827203

/-- Given a cube, 4 planes perpendicular to each body diagonal at their midpoints divide the surface into 14 parts. -/
theorem cube_midpoint_planes_divide (cube : ℝ) : 
  let planes := 4 in 
  planes = 4 → 
  ∃ parts, parts = 14 := 
by sorry

/-- Given a cube, 8 planes perpendicular to each body diagonal at both third-points divide the surface into 24 parts. -/
theorem cube_thirdpoint_planes_divide (cube : ℝ) : 
  let planes := 8 in 
  planes = 8 → 
  ∃ parts, parts = 24 := 
by sorry

end cube_midpoint_planes_divide_cube_thirdpoint_planes_divide_l827_827203


namespace probability_of_relatively_prime_to_42_l827_827187

def relatively_prime_to_42_count : ℕ :=
  let N := 42
  (finset.range (N + 1)).filter (λ n, Nat.gcd n N = 1).card

theorem probability_of_relatively_prime_to_42 : 
  (relatively_prime_to_42_count : ℚ) / 42 = 2 / 7 :=
by
  sorry

end probability_of_relatively_prime_to_42_l827_827187


namespace max_value_of_y_diffs_l827_827324

noncomputable def y (x : ℕ → ℝ) (k : ℕ) : ℝ := 
  if k = 0 then 0 else (∑ i in Finset.range k, x (i + 1)) / k

theorem max_value_of_y_diffs (x : ℕ → ℝ)
  (h : ∑ i in Finset.range 1992, |x i - x (i + 1)| = 1993) :
  ∑ i in Finset.range 1992, |y x (i + 1) - y x (i + 2)| ≤ 1992 :=
sorry

end max_value_of_y_diffs_l827_827324


namespace selling_price_ratio_l827_827359

theorem selling_price_ratio (C : ℝ) (hC : C > 0) :
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  S₂ / S₁ = 21 / 8 :=
by
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  sorry

end selling_price_ratio_l827_827359


namespace crossing_time_proof_l827_827709

noncomputable def length_first_train : ℝ := 120
noncomputable def length_second_train : ℝ := 150
noncomputable def speed_first_train : ℝ := 80 * 1000 / 3600
noncomputable def speed_second_train : ℝ := 100 * 1000 / 3600
noncomputable def time_same_direction : ℝ := 60

def crossing_time_opposite_directions 
  (l1 l2 v1 v2 t_same : ℝ) : ℝ := 
  let total_length := l1 + l2 in
  let relative_speed := v1 + v2 in
  total_length / relative_speed

theorem crossing_time_proof :
  crossing_time_opposite_directions length_first_train length_second_train speed_first_train speed_second_train time_same_direction = 5.4 :=
by
  sorry

end crossing_time_proof_l827_827709


namespace binomial_expansion_constant_term_l827_827460

theorem binomial_expansion_constant_term (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → ∑ r in (finset.range 7), (nat.choose 6 r * (- (real.sqrt a) / x^2) ^ r * x ^ (6 - 3 * r) = 60))
: a = 4 :=
sorry

end binomial_expansion_constant_term_l827_827460


namespace find_f1_l827_827504

noncomputable def f : ℝ → ℝ := sorry

theorem find_f1
  (h1 : ∀ x, 0 < x → ∀ y, 0 < y → x ≤ y → f(x) ≤ f(y)) -- f is monotonous
  (h2 : ∀ x, 0 < x → f(x) * f(f(x) + 2 / x) = 2) -- Given functional equation
  (h3 : 0 < 1) -- Given domain of f(x) is (0, +∞)
  : f(1) = 1 + real.sqrt 5 ∨ f(1) = 1 - real.sqrt 5 := sorry

end find_f1_l827_827504


namespace not_a_function_relationship_l827_827725

noncomputable def is_function_relationship (f : ℝ → ℝ) : Prop :=
∀ x x', f x = f x' → x = x'

noncomputable def relationships : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ) → Prop :=
  λ relA relB relC,
    ¬ is_function_relationship relD

theorem not_a_function_relationship :
  relationships
    (λ a : ℝ, a^3) -- Edge length and volume of a cube
    (λ α : ℝ, Real.sin α) -- Degree of an angle and its sine value
    (λ x : ℝ, a * x) -- Total grain yield and land area when the unit yield is constant
    (λ sunlight yield : ℝ, sorry) -- The relationship between the amount of sunlight and the yield per acre of rice
    :=
sorry

end not_a_function_relationship_l827_827725


namespace beth_win_strategy_l827_827986
-- Conditions definitions
def takes_one_brick_or_one_pair (n : ℕ) (m : ℕ) (k : ℕ) : Prop := 
  (n > 0 ∧ m ≥ 0 ∧ k ≥ 0) ∨ 
  (n ≥ 0 ∧ m > 0 ∧ k ≥ 0) ∨ 
  (n ≥ 0 ∧ m ≥ 0 ∧ k > 0) ∨ 
  (n ≥ 2 ∧ m ≥ 0 ∧ k ≥ 0) ∨ 
  (n ≥ 0 ∧ m ≥ 2 ∧ k ≥ 0) ∨ 
  (n ≥ 0 ∧ m ≥ 0 ∧ k ≥ 2)
  
def not_two_bricks_from_separate_walls (n1 n2 n3 : ℕ) (case : ℕ → Prop) : Prop := 
  (case n1 ∧ case n2 ∧ n3 = 0) ∨
  (case n1 ∧ n2 = 0 ∧ case n3) ∨
  (n1 = 0 ∧ case n2 ∧ case n3)

def nim_value (n : ℕ) := n

-- Main theorem to be proved in Lean 4 statement
theorem beth_win_strategy : 
  takes_one_brick_or_one_pair 7 3 2 → 
  not_two_bricks_from_separate_walls 7 3 2 nim_value → 
  nim_value 7 ⊕ nim_value 3 ⊕ nim_value 2 = 0 :=
by 
  -- Using the defines and theorems from the Lean library to prove by the conditions. 
  intros,
  sorry -- Proof to be filled in subsequently.

end beth_win_strategy_l827_827986


namespace roots_imply_value_l827_827098

noncomputable def value_of_expression (a b c : ℝ) : ℝ :=
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)

theorem roots_imply_value {a b c : ℝ} 
  (h1 : a + b + c = 15) 
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 10) 
  : value_of_expression a b c = 175 / 11 :=
sorry

end roots_imply_value_l827_827098


namespace area_of_triangles_sum_l827_827979

theorem area_of_triangles_sum {A B C D E : Type} [HasDist A B C] [HasMeasAngle A B C D E] 
(h_midpoint_E : E.midpoint (B, C)) (h_midpoint_D : 2 * AD = DB) (h_length_AB : AB = 2)
(h_angle_BAC : ∡BAC = 80°) (h_angle_ABC : ∡ABC = 70°) (h_angle_ACB : ∡ACB = 30°)
(h_angle_DEC : ∡DEC = 40°) : 
  (area A B C) + (area C D E) = (5 / 4) * sin 80 :=
begin
  sorry
end

end area_of_triangles_sum_l827_827979


namespace triangle_SineRule_l827_827554

-- Declare the conditions as variables
variables 
  (b : ℝ) 
  (angleB : ℝ) 
  (sinA : ℝ)

-- Define the main theorem
theorem triangle_SineRule 
  (hb : b = 5) 
  (hangleB : angleB = Real.pi / 4) 
  (hsinA : sinA = 1 / 3) 
  : ∃ a : ℝ, a = 5 * Real.sqrt 2 / 3 := 
by
  use 5 * Real.sqrt 2 / 3
  sorry

end triangle_SineRule_l827_827554


namespace good_triangulations_diff_exactly_two_l827_827379

-- Definition: A "good" triangulation T of a convex polygon Π 
-- is a division of Π using diagonals such that all resulting triangles have equal area.
structure GoodTriangulation (Π : Type) [ConvexPolygon Π] where
  area_equal : ∀ t1 t2 ∈ T, area t1 = area t2

-- Main Theorem: For any two different good triangulations T1 and T2 of convex polygon Π, exactly two triangles differ.
theorem good_triangulations_diff_exactly_two {Π : Type} [ConvexPolygon Π] 
  (T1 T2 : GoodTriangulation Π) (hT1T2 : T1 ≠ T2) :
  ∃ (Δ1 Δ2 ∈ T1) (Δ'1 Δ'2 ∈ T2), (Δ1 ≠ Δ'1 ∨ Δ2 ≠ Δ'2) :=
sorry

end good_triangulations_diff_exactly_two_l827_827379


namespace mans_speed_upstream_l827_827775

-- Define the conditions
def V_downstream : ℝ := 15  -- Speed with the current (downstream)
def V_current : ℝ := 2.5    -- Speed of the current

-- Calculate the man's speed against the current (upstream)
theorem mans_speed_upstream : V_downstream - 2 * V_current = 10 :=
by
  sorry

end mans_speed_upstream_l827_827775


namespace sum_of_primes_between_10_and_20_l827_827302

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827302


namespace kevin_height_l827_827751

theorem kevin_height (tree_height_ft : ℝ) (tree_shadow_ft : ℝ) (kevin_shadow_in : ℝ) 
  (ht_tree : tree_height_ft = 50) 
  (sh_tree : tree_shadow_ft = 25) 
  (sh_kevin : kevin_shadow_in = 20) : 
  (Kevin_height : ℝ) := 
begin
  have ratio := tree_height_ft / tree_shadow_ft,
  have kevin_height_in := ratio * kevin_shadow_in,
  have kevin_height_ft := kevin_height_in / 12,
  exact kevin_height_ft = 10/3,
end

end kevin_height_l827_827751


namespace coordinates_of_point_l827_827636

theorem coordinates_of_point (a : ℝ) (h : a - 3 = 0) : (a + 2, a - 3) = (5, 0) :=
by
  sorry

end coordinates_of_point_l827_827636


namespace non_prime_in_sequence_l827_827145

theorem non_prime_in_sequence : ∃ n : ℕ, ¬Prime (41 + n * (n - 1)) :=
by {
  use 41,
  sorry
}

end non_prime_in_sequence_l827_827145


namespace union_complement_eq_l827_827021

namespace SetTheory

def U : set ℕ := {0, 1, 2, 4, 6, 8}
def M : set ℕ := {0, 4, 6}
def N : set ℕ := {0, 1, 6}

def complement_U (s : set ℕ) : set ℕ := U \ s

theorem union_complement_eq :
  M ∪ complement_U N = {0, 2, 4, 6, 8} :=
by {
  sorry
}

end SetTheory

end union_complement_eq_l827_827021


namespace pipe_leak_time_l827_827347

noncomputable def pipe_fill (a_hours : ℝ) (b_hours : ℝ) : ℝ :=
  1 / a_hours - 1 / b_hours

theorem pipe_leak_time :
  (a : ℝ) (b : ℝ) (net_time : ℝ),
  a = 8 → 
  net_time = 23.999999999999996 →
  pipe_fill a b = 1 / net_time → 
  b = 12 :=
by
  intros a b net_time ha hnet hfill
  sorry

end pipe_leak_time_l827_827347


namespace union_complement_eq_l827_827020

namespace SetTheory

def U : set ℕ := {0, 1, 2, 4, 6, 8}
def M : set ℕ := {0, 4, 6}
def N : set ℕ := {0, 1, 6}

def complement_U (s : set ℕ) : set ℕ := U \ s

theorem union_complement_eq :
  M ∪ complement_U N = {0, 2, 4, 6, 8} :=
by {
  sorry
}

end SetTheory

end union_complement_eq_l827_827020


namespace Keiko_text_messages_l827_827600

theorem Keiko_text_messages :
  ∀ (last_week this_week total : ℕ),
    last_week = 111 →
    this_week = 2 * last_week - 50 →
    total = last_week + this_week →
    total = 283 :=
by
  intros last_week this_week total h_last h_this h_total
  rw [h_last, h_this, h_total]
  sorry

end Keiko_text_messages_l827_827600


namespace general_formula_geometric_seq_sum_of_b_n_l827_827902

noncomputable def a_n (n : ℕ) : ℕ := 2^n

def b_n (n : ℕ) : ℤ := -n * 2^(n+1)

def S_n (n : ℕ) : ℤ := (Finset.range n).sum (λ k, b_n (k + 1))

theorem general_formula_geometric_seq :
  ∀ n, a_n n = 2^n :=
by sorry

theorem sum_of_b_n :
  ∀ n, S_n n = 2^(n+2) - n * 2^(n+2) - 4 :=
by sorry

end general_formula_geometric_seq_sum_of_b_n_l827_827902


namespace inequality_check_l827_827497

variables {a b : ℝ} {n : ℕ} (h : a ≠ b) (ha : a > 0) (hb : b > 0)
variables (x : ℕ → ℝ) (y : ℕ → ℝ)
noncomputable theory

def arithmetic_sequence (x : ℕ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k ∈ finset.range n → x k = a + (k + 1 : ℝ) * ((b - a) / (n + 1))
def geometric_sequence (y : ℕ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k ∈ finset.range n → y k = a * ( (b / a) ^ ((k + 1 : ℝ) / (n + 1)))

theorem inequality_check
  (h_arith : arithmetic_sequence x a b n)
  (h_geom : geometric_sequence y a b n) :
    (1 / n * (finset.range n).sum x > sqrt (a * b) + ((sqrt a - sqrt b) / 2) ^ 2) ∧
    (√[n]{(finset.range n).prod y} < sqrt (a * b) ∧ 
    (√[n]{(finset.range n).prod y} < (a + b) / 2 - ((sqrt a - sqrt b) / 2)^2)) :=
  sorry

end inequality_check_l827_827497


namespace bethany_age_l827_827181

theorem bethany_age : ∀ (B S R : ℕ),
  (B - 3 = 2 * (S - 3)) →
  (B - 3 = R - 3 + 4) →
  (S + 5 = 16) →
  (R + 5 = 21) →
  B = 19 :=
by
  intros B S R h1 h2 h3 h4
  sorry

end bethany_age_l827_827181


namespace expected_value_of_winnings_after_one_flip_l827_827756

-- Definitions based on conditions from part a)
def prob_heads : ℚ := 1 / 3
def prob_tails : ℚ := 2 / 3
def win_heads : ℚ := 3
def lose_tails : ℚ := -2

-- The statement to prove:
theorem expected_value_of_winnings_after_one_flip :
  prob_heads * win_heads + prob_tails * lose_tails = -1 / 3 :=
by
  sorry

end expected_value_of_winnings_after_one_flip_l827_827756


namespace sin_sum_of_triangle_l827_827016

theorem sin_sum_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : (2 * b)^2 = 4 * (c + a) * (c - a))
  (h2 : 5 * a - 3 * c = 0) 
  (hA : sin A = a / c)
  (hB : sin B = b / c)
  (hC : C = π / 2)
  : sin A + sin B + sin C = 12 / 5 := 
by
  sorry

end sin_sum_of_triangle_l827_827016


namespace length_of_arc_QS_l827_827576

theorem length_of_arc_QS (O Q I S : Point) (r : ℝ) (angle_QIS : ℝ) (arc_length : ℝ) 
  (hOQ : OQ = 15) (h_angle_QIS : angle_QIS = 45) 
  (h_center_O : center O) (h_radius_OQ : radius OQ) : 
  arc_length = 7.5 * π :=
by
  sorry

end length_of_arc_QS_l827_827576


namespace length_AB_is_4_trajectory_of_C_l827_827904

-- Definitions based on the given conditions
def ellipse_equation : Prop := ∀ (x y : ℝ), x^2 + 5 * y^2 = 5
def sin_relation : Prop := ∀ (A B C : ℝ), sin B - sin A = sin C

-- Length of segment AB given the ellipse equation condition
theorem length_AB_is_4 (A B : ℝ × ℝ) (hA : A = (-2, 0)) (hB : B = (2, 0)) (h_ellipse : ellipse_equation) : 
  (dist A B) = 4 := 
sorry

-- Equation of the trajectory of vertex C given the sin relation condition
theorem trajectory_of_C (C : ℝ × ℝ) (h_sin : sin_relation) :
  (C.1)^2 - (C.2)^2 / 3 = 1 ∧ C.1 > 1 :=
sorry

end length_AB_is_4_trajectory_of_C_l827_827904


namespace scrap_cookie_radius_l827_827356

/-- Prove that the radius of the cookie made from the leftover scrap dough is sqrt(17) inches,
given the following conditions:
1. A large circular piece of cookie dough has a radius of 5 inches.
2. Eight cookies, each with a radius of 1 inch, are cut from this dough.
3. Neighboring cookies are tangent to each other, and all except the center one are tangent to the edge of the dough.
4. The leftover scrap is reformed into another cookie of the same thickness.
-/
theorem scrap_cookie_radius :
  let r_large := 5
  let r_small := 1
  let A_large := Real.pi * r_large^2
  let A_small := Real.pi * r_small^2
  let A_total_small := 8 * A_small
  let A_scrap := A_large - A_total_small
  ∃ r_scrap, A_scrap = Real.pi * r_scrap^2 ∧ r_scrap = Real.sqrt 17 :=
by {
  let r_large := 5
  let r_small := 1
  let A_large := Real.pi * r_large^2
  let A_small := Real.pi * r_small^2
  let A_total_small := 8 * A_small
  let A_scrap := A_large - A_total_small
  use Real.sqrt 17,
  split,
  {
    calc 
      A_scrap = 25 * Real.pi - 8 * Real.pi : by rw [A_large, A_total_small]
            ... = 17 * Real.pi           : by ring,
    rw [A_scrap, mul_comm Real.pi],
  },
  {
    rfl,
  },
  sorry
}

end scrap_cookie_radius_l827_827356


namespace man_speed_against_current_proof_l827_827779

def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

theorem man_speed_against_current_proof 
  (V_m : ℝ) 
  (h_with_current : V_m + speed_of_current = man_speed_with_current) :
  V_m - speed_of_current = man_speed_against_current := 
by 
  sorry

end man_speed_against_current_proof_l827_827779


namespace imaginary_part_of_complex_number_l827_827166

theorem imaginary_part_of_complex_number : complex.im (complex.pow (complex.sub (complex.inv complex.I) complex.I) 3) = 8 := by
  sorry

end imaginary_part_of_complex_number_l827_827166


namespace total_payroll_calc_l827_827388

theorem total_payroll_calc
  (h : ℕ := 129)          -- pay per day for heavy operators
  (l : ℕ := 82)           -- pay per day for general laborers
  (n : ℕ := 31)           -- total number of people hired
  (g : ℕ := 1)            -- number of general laborers employed
  : (h * (n - g) + l * g) = 3952 := 
by
  sorry

end total_payroll_calc_l827_827388


namespace normal_distribution_probability_l827_827617

variable {σ : ℝ}
variable (ξ : ℝ) (hξ : ∀ x, P((ξ > x) ↔ (x > 4) = 0.1))

theorem normal_distribution_probability (h_norm : ∀ ξ, ξ ~ Normal 2 σ) (h_prob : P(ξ > 4) = 0.1) : P(ξ < 0) = 0.1 :=
sorry

end normal_distribution_probability_l827_827617


namespace green_block_weight_l827_827078

theorem green_block_weight (y g : ℝ) (h1 : y = 0.6) (h2 : y = g + 0.2) : g = 0.4 :=
by
  sorry

end green_block_weight_l827_827078


namespace length_of_arc_QS_l827_827575

theorem length_of_arc_QS (O Q I S : Point) (r : ℝ) (angle_QIS : ℝ) (arc_length : ℝ) 
  (hOQ : OQ = 15) (h_angle_QIS : angle_QIS = 45) 
  (h_center_O : center O) (h_radius_OQ : radius OQ) : 
  arc_length = 7.5 * π :=
by
  sorry

end length_of_arc_QS_l827_827575


namespace complement_union_l827_827530

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem complement_union (U : Set ℝ) (A B : Set ℝ) (hU : U = univ) (hA : A = { x | x < 0 }) (hB : B = { x | x ≥ 2 }) :
  C_U U (A ∪ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end complement_union_l827_827530


namespace wheel_stop_probability_l827_827380

theorem wheel_stop_probability 
  (pD pE pG pF : ℚ) 
  (h1 : pD = 1 / 4) 
  (h2 : pE = 1 / 3) 
  (h3 : pG = 1 / 6) 
  (h4 : pD + pE + pG + pF = 1) : 
  pF = 1 / 4 := 
by 
  sorry

end wheel_stop_probability_l827_827380


namespace parabola_intersections_l827_827421

open Set Function

noncomputable def parabola : Type := {a : ℤ // -3 ≤ a ∧ a ≤ 3} × {b : ℤ // -4 ≤ b ∧ b ≤ 4}

theorem parabola_intersections :
  let parabolas : Set parabola := {p | True}  -- All valid parabolas in the set
  ∃ I : ℕ, I = 2898 ∧
  ∀ (p1 p2 p3 : parabola),
      p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
      (¬ ∃ pt : ℝ × ℝ, pt ∈ (λ p, {pt | ∃ x y, pt = (x, y) ∧ parabola_dir_eq p x y})) →
         ∃ pt : ℝ × ℝ,
         pt ∈ (λ p, {pt | ∃ x y, pt = (x, y) ∧ parabola_dir_eq p x y}) :=
by
  sorry

-- Auxiliary definition to handle parabola's directrix lines for correctness
def parabola_dir_eq (p : parabola) (x : ℝ) (y : ℝ) : Prop :=
  (y = p.1.1 * x + p.2.1)

end parabola_intersections_l827_827421


namespace indistinguishable_balls_into_distinguishable_boxes_l827_827031

theorem indistinguishable_balls_into_distinguishable_boxes :
  (∃ (n : ℕ), n = 6) →
  (∃ (m : ℕ), m = 4) →
  (∑ (k : ℕ) in {1, 12, 12, 6, 12, 12, 6, 4}, k = 65) :=
begin
  sorry
end

end indistinguishable_balls_into_distinguishable_boxes_l827_827031


namespace area_of_ABCD_l827_827468

theorem area_of_ABCD :
  ∀ (w h : ℕ), 
  (w = 7) → (h = 2 * w) → 
  ((AB_width = 2 * w) ∧ (AB_length = 2 * h)) → 
  (area : ℕ) → 
  (area = AB_width * AB_length) → 
  area = 392 :=
by
  intros w h hw hh hrect area ha
  rw [hw, hh] at hrect
  cases hrect with hab_width hab_length
  rw [hab_width, hab_length] at ha
  sorry

end area_of_ABCD_l827_827468


namespace ae_ec_ratio_l827_827981

open Segment

theorem ae_ec_ratio (A B C D E T : Type) [aff S : affine_space Point Segment]
  (h1 : D ∈ between B C)
  (h2 : E ∈ between A C)
  (h3 : AD ∩ BE = T)
  (h4 : ratio AT DT = 2)
  (h5 : ratio BT ET = 5) :
  ratio AE EC = 3 :=
sorry

end ae_ec_ratio_l827_827981


namespace red_higher_than_blue_probability_l827_827789

/-- Definition of the probability function for a ball landing in bin k --/
def P (k : ℕ) : ℝ := if k % 2 = 1 then (1 / (3 : ℝ)^k) else (1 / 2) * (1 / (3 : ℝ)^k)

/-- The theorem stating the probability problem --/
theorem red_higher_than_blue_probability :
  (∑' (i : ℕ), ∑' (j : ℕ), if i > j then P i * P j else 0) = 1 / 4 :=
sorry

end red_higher_than_blue_probability_l827_827789


namespace distinct_painted_cubes_eq_two_l827_827759

-- Given conditions
def cube : Type := unit
def color := {c // c = "blue" ∨ c = "red" ∨ c = "yellow"}

-- Number of distinct painted cubes considering rotational symmetry
noncomputable def distinct_painted_cubes :=
  {c : cube // (c.1, c.2) = (2, 2, 2)}

-- The statement to prove
theorem distinct_painted_cubes_eq_two : fintype.card distinct_painted_cubes = 2 := 
sorry

end distinct_painted_cubes_eq_two_l827_827759


namespace class_B_has_more_stable_grades_l827_827823

-- Definitions based on conditions
def avg_score_class_A : ℝ := 85
def avg_score_class_B : ℝ := 85
def var_score_class_A : ℝ := 120
def var_score_class_B : ℝ := 90

-- Proving which class has more stable grades (lower variance indicates more stability)
theorem class_B_has_more_stable_grades :
  var_score_class_B < var_score_class_A :=
by
  -- The proof will need to show the given condition and establish the inequality
  sorry

end class_B_has_more_stable_grades_l827_827823


namespace sum_of_first_19_terms_l827_827000

-- Definitions for the arithmetic sequence and sum of n terms
def arithmetic_sum (a_1 d : ℕ → ℕ) (n : ℕ) : ℕ := n * (2 * a_1 + (n - 1) * d) / 2

-- Given conditions
axiom a6_a14_sum (a_1 d : ℕ → ℕ) : a_1 6 + a_1 14 = 20

-- Prove that the sum of the first 19 terms S_19 = 190
theorem sum_of_first_19_terms (a_1 d : ℕ → ℕ) :
  arithmetic_sum a_1 d 19 = 190 := 
by 
  sorry

end sum_of_first_19_terms_l827_827000


namespace find_a_l827_827972

theorem find_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 4 * y = 0 → (3 * x + y + a = 0))) → a = 1 :=
sorry

end find_a_l827_827972


namespace sum_primes_between_10_and_20_l827_827282

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827282


namespace probability_same_color_l827_827540

theorem probability_same_color (r g : ℕ) (select : ℕ) (total_combinations : ℕ) (red_combinations : ℕ) (green_combinations : ℕ) (same_color_probability : ℚ) :
  r = 6 → g = 5  → select = 3 → 
  total_combinations = Nat.choose 11 3 → 
  red_combinations = Nat.choose 6 3 → 
  green_combinations = Nat.choose 5 3 → 
  same_color_probability = (red_combinations + green_combinations) / total_combinations → 
  same_color_probability = 2 / 11 :=
by 
  intros h_r h_g h_s h_tc h_rc h_gc h_p
  rw [h_r, h_g, h_s] at *
  rw [h_tc, h_rc, h_gc, h_p]
  exact sorry

end probability_same_color_l827_827540


namespace find_number_l827_827785

theorem find_number 
    (x : ℝ)
    (h1 : 3 < x) 
    (h2 : x < 8) 
    (h3 : 6 < x) 
    (h4 : x < 10) : 
    x = 7 :=
sorry

end find_number_l827_827785


namespace hexagon_AF_length_l827_827059

theorem hexagon_AF_length (BC CD DE AB EF AF : ℝ) (mB mC mD mF : ℝ) (a b : ℕ) :
  BC = 2 ∧ CD = 2 ∧ DE = 2 ∧ AB = 3 ∧ EF = 3 ∧ mF = 90 ∧ mB = 135 ∧ mC = 135 ∧ mD = 135 ∧ AF = 3 + 2 * real.sqrt 2 →
  (a + 2 * real.sqrt b = 3 + 2 * real.sqrt 2) → a + b = 5 :=
by
  intros h_conditions h_AF
  sorry

end hexagon_AF_length_l827_827059


namespace sum_of_primes_between_10_and_20_l827_827260

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827260


namespace simplify_trigonometric_expression_l827_827646

noncomputable def trigonometric_simplification (x y : ℝ) : ℝ :=
  sin x ^ 2 + cos (x + y) ^ 2 + 2 * sin x * sin y * cos (x + y)

theorem simplify_trigonometric_expression (x y : ℝ) :
  trigonometric_simplification x y = 1 + cos y ^ 2 := 
by
  -- Proof goes here
  sorry

end simplify_trigonometric_expression_l827_827646


namespace problem_theorem_l827_827757

noncomputable theory

open_locale euclidean_geometry

variables {A B C D E P Q X Y : Point} {ω: Circle}

-- Conditions of the problem
def conditions (ω : Circle) (A B C P Q D E X Y : Point) : Prop :=
  ω.is_circumscribed_triangle A B C ∧
  Line.parallel (line_through A C) (line_through D E) ∧
  Line.parallel (line_through D P) (line_through E Q) ∧
  ω.arc_contains A C P ∧
  ω.arc_contains A C Q ∧
  line_intersection (line_through Q A) (line_through D E) = X ∧
  line_intersection (line_through P C) (line_through D E) = Y

-- Theorem statement
theorem problem_theorem (ω : Circle) (A B C P Q D E X Y : Point) 
  (h : conditions ω A B C P Q D E X Y) :
  ∠X B Y + ∠P B Q = 180 :=
sorry

end problem_theorem_l827_827757


namespace circle_radius_d_l827_827465

theorem circle_radius_d (d : ℝ) : ∀ (x y : ℝ), (x^2 + 8 * x + y^2 + 2 * y + d = 0) → (∃ r : ℝ, r = 5) → d = -8 :=
by
  sorry

end circle_radius_d_l827_827465


namespace rad_times_trivia_eq_10000_l827_827148

theorem rad_times_trivia_eq_10000 
  (h a r v d m i t : ℝ)
  (H1 : h * a * r * v * a * r * d = 100)
  (H2 : m * i * t = 100)
  (H3 : h * m * m * t = 100) :
  (r * a * d) * (t * r * i * v * i * a) = 10000 := 
  sorry

end rad_times_trivia_eq_10000_l827_827148


namespace constructible_triangle_degenerate_on_circumcircle_l827_827064

open Triangle

theorem constructible_triangle (A B C P : Point) (hABC : ¬collinear A B C)
  (hP : P ≠ circumcenter A B C) :
  let sA := dist P A * sin (angle B C A)
  let sB := dist P B * sin (angle A C B)
  let sC := dist P C * sin (angle A B C)
  in is_triangle sA sB sC :=
by
  sorry

theorem degenerate_on_circumcircle (A B C P : Point) (hABC : ¬collinear A B C)
  (hP : P = circumcenter A B C) :
  let sA := dist P A * sin (angle B C A)
  let sB := dist P B * sin (angle A C B)
  let sC := dist P C * sin (angle A B C)
  in degenerate_triangle sA sB sC :=
by
  sorry

end constructible_triangle_degenerate_on_circumcircle_l827_827064


namespace rational_terms_not_adjacent_l827_827579

-- Defining the conditions for the problem.
def binomial_expansion {x n : ℕ} : ℕ := 
  let a := sqrt x
  let b := (1 : ℝ) / (2 * (x^(1/4)))
  a + b ^ n

-- The coefficients of the first three terms must form an arithmetic sequence.
def arith_sequence_condition (n : ℕ) : Prop :=
  ∃ Cn0 Cn1 Cn2 : ℕ,
    let coeffs := [Cn0, (1/2) * Cn1, (1/4) * Cn2]
    coeffs.nth 1 = some (Cn0 + (1/4) * Cn2)

-- The probability that rational terms are not adjacent is 5/12.
theorem rational_terms_not_adjacent (x : ℕ) (n : ℕ) 
  (h1 : binomial_expansion = n) 
  (h2 : arith_sequence_condition n) : 
  probability_rational_terms_not_adjacent(binomial_expansion) = 5/12 :=
sorry

end rational_terms_not_adjacent_l827_827579


namespace uneaten_chips_l827_827432

theorem uneaten_chips :
  ∀ (chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips : ℕ),
    (chips_per_cookie = 7) →
    (cookies_total = 12 * 4) →
    (half_cookies = cookies_total / 2) →
    (uneaten_cookies = cookies_total - half_cookies) →
    (uneaten_chips = uneaten_cookies * chips_per_cookie) →
    uneaten_chips = 168 :=
by
  intros chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips
  intros chips_per_cookie_eq cookies_total_eq half_cookies_eq uneaten_cookies_eq uneaten_chips_eq
  rw [chips_per_cookie_eq, cookies_total_eq, half_cookies_eq, uneaten_cookies_eq, uneaten_chips_eq]
  norm_num
  sorry

end uneaten_chips_l827_827432


namespace sum_of_primes_between_10_and_20_is_60_l827_827278

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827278


namespace sum_of_primes_between_10_and_20_is_60_l827_827269

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827269


namespace hyperbola_eccentricity_l827_827195

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  let e := (1 + (b^2) / (a^2)).sqrt
  e

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a + b = 5)
  (h2 : a * b = 6)
  (h3 : a > b) :
  eccentricity a b = Real.sqrt 13 / 3 :=
sorry

end hyperbola_eccentricity_l827_827195


namespace last_digit_a_128_1_mod_10_l827_827840

def seq (i j : ℕ) : ℕ
| 1 n       := n ^ n
| (i + 1) j := seq i j + seq i (j + 1)

theorem last_digit_a_128_1_mod_10 :
  seq 128 1 % 10 = 4 := 
sorry

end last_digit_a_128_1_mod_10_l827_827840


namespace least_integer_excluding_19_and_20_l827_827354

theorem least_integer_excluding_19_and_20 :
  ∃ N : ℕ, (∀ m ∈ (finset.range 30).erase 18 ∪ (finset.range 30).erase 19, m ≠ 0 → m ∣ N) ∧ (¬ 19 ∣ N) ∧ (¬ 20 ∣ N) ∧ (N = 2329089562800) :=
sorry

end least_integer_excluding_19_and_20_l827_827354


namespace question_d_l827_827478

variable {x a : ℝ}

theorem question_d (h1 : x < a) (h2 : a < 0) : x^3 > a * x ∧ a * x < 0 :=
  sorry

end question_d_l827_827478


namespace four_letter_words_with_E_count_l827_827026

open Finset

/-- Number of 4-letter words from alphabet {A, B, C, D, E} with at least one E --/
theorem four_letter_words_with_E_count :
  let alphabet := {A, B, C, D, E}
      total_words := (Finset.card alphabet) ^ 4,
      words_without_E := (Finset.card (alphabet \ {'E'})) ^ 4,
      words_with_at_least_one_E := total_words - words_without_E in
  words_with_at_least_one_E = 369 :=
by
  let alphabet := {A, B, C, D, E}
  let total_words := (Finset.card alphabet) ^ 4
  let words_without_E := (Finset.card (alphabet \ {'E'})) ^ 4
  let words_with_at_least_one_E := total_words - words_without_E
  have h_total_words : total_words = 625 := by sorry
  have h_words_without_E : words_without_E = 256 := by sorry
  have h : words_with_at_least_one_E = 369 := by sorry
  exact h

end four_letter_words_with_E_count_l827_827026


namespace smallest_possible_value_l827_827910

theorem smallest_possible_value (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) : 
  ∃ (m : ℝ), m = -1/12 ∧ (∀ x y : ℝ, (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → (x + y) / (x^2) ≥ m) :=
sorry

end smallest_possible_value_l827_827910


namespace sin_tan_eq_neg10_f_2pi_alpha_l827_827476

def f (x : ℝ) : ℝ := Math.sin x + Math.tan (x / 2) + 1

-- assumption 1: f(-α) = 11
axiom f_neg_alpha (α : ℝ) : f (-α) = 11

-- hypothesis: simplified result from the conditions
theorem sin_tan_eq_neg10 (α : ℝ) : Math.sin α + Math.tan (α / 2) = -10 :=
by
  -- proof omitted for illustrative purposes
  sorry

-- main theorem
theorem f_2pi_alpha (α : ℝ) : f (2 * Math.pi + α) = -9 :=
by
  have h : f (-α) = 11 := f_neg_alpha α
  have h2 : Math.sin α + Math.tan (α / 2) = -10 := sin_tan_eq_neg10 α
  -- proof omitted for illustrative purposes
  sorry

end sin_tan_eq_neg10_f_2pi_alpha_l827_827476


namespace inequality_proof_l827_827878

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  -- Proof omitted
  sorry

end inequality_proof_l827_827878


namespace range_of_h_l827_827939

def f (x : ℝ) (h : ℝ) : ℝ := x - Real.log x + h

theorem range_of_h {a b c h : ℝ} 
  (h_interval : ∀ x, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 2))
  (h_triangle_ineq : ∀ a b c, a + b > c ∧ a + c > b ∧ b + c > a) : 
  (a, b, c ∈ Set.Icc (1 / Real.exp 1) (Real.exp 2) → 
  (h > Real.exp 2 - 4) → 
  ∃ (x y z : ℝ), x = f a h ∧ y = f b h ∧ z = f c h ∧ h_triangle_ineq x y z) :=
sorry

end range_of_h_l827_827939


namespace distinct_sum_of_three_l827_827952

def set := {2, 5, 8, 11, 14, 17, 20}

theorem distinct_sum_of_three (S : Set ℕ) (hS : S = set) : 
  ∃! (n : ℕ), 
    (∀ (s1 s2 s3 : ℕ), s1 ≠ s2 → s2 ≠ s3 → s1 ≠ s3 → 
      s1 ∈ S ∧ s2 ∈ S ∧ s3 ∈ S → n = s1 + s2 + s3) ∧ n = 13 :=
by
  -- The proof goes here
  sorry

end distinct_sum_of_three_l827_827952


namespace circle_equation_l827_827498

/-- Let (6, 1) be a point on a circle that is tangent to the y-axis, and its center lies on the line x - 3y = 0. 
Prove that the equation of the circle is (x - 3)^2 + (y - 1)^2 = 9. -/
theorem circle_equation :
  ∃ (h k r : ℝ), (h - 3)^2 + (k - 1)^2 = r^2 ∧ r = 3 ∧ (6 - h)^2 + (1 - k)^2 = r^2 ∧ k = 1/3 * h ∧ (x - h)^2 + (y - k)^2 = r^2 :=
begin
  use [3, 1, 3],
  split,
  { exact (by norm_num : (3 - 3)^2 + (1 - 1)^2 = 3 ^ 2) },
  split,
  { refl },
  split,
  { exact (by norm_num : (6 - 3)^2 + (1 - 1)^2 = 3 ^ 2) },
  split,
  { exact (by norm_num : 1 = 1/3 * 3) },
  { refl }
end

end circle_equation_l827_827498


namespace ellis_more_cards_than_orion_l827_827681

theorem ellis_more_cards_than_orion : 
  ∀ (N : ℕ) (r1 r2 : ℕ), N = 500 → r1 = 11 → r2 = 9 → 
  (r1 * (N / (r1 + r2)) - r2 * (N / (r1 + r2)) = 50) := 
by
  intros N r1 r2 hN hr1 hr2
  rw [hN, hr1, hr2]
  sorry

end ellis_more_cards_than_orion_l827_827681


namespace probability_of_paramecium_l827_827045

theorem probability_of_paramecium 
  (total_volume : ℝ) 
  (observation_volume : ℝ) 
  (paramecium_present : Prop) 
  (h1 : total_volume = 500)
  (h2 : observation_volume = 2)
  (h3 : paramecium_present) :
  let probability : ℝ := observation_volume / total_volume in
  probability = 0.004 :=
by
  -- Because we want only the theorem statement without proof
  sorry

end probability_of_paramecium_l827_827045


namespace problem_l827_827113

noncomputable def nums : Type := { p q r s : ℝ // p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s }

theorem problem (n : nums) :
  let p := n.1
      q := n.2.1
      r := n.2.2.1
      s := n.2.2.2.1
  in (r + s = 12 * p) → (r * s = -13 * q) → (p + q = 12 * r) → (p * q = -13 * s) → p + q + r + s = 2028 :=
by
  intros
  sorry

end problem_l827_827113


namespace distance_OP_is_correct_l827_827758

-- Given conditions
def radius_O : ℝ := 12
def radius_P : ℝ := 4
def distance_OQ : ℝ := radius_O
def distance_PQ : ℝ := radius_P

-- Defining the function to calculate the distance OP
noncomputable def distance_OP : ℝ := 
  let OP_dist := sqrt (radius_O^2 + radius_P^2 - 2*radius_O*radius_P*sqrt(1 - (radius_P / radius_O)^2))
  OP_dist

-- The theorem we need to prove
theorem distance_OP_is_correct : distance_OP = 4 * sqrt 10 :=
by sorry

end distance_OP_is_correct_l827_827758


namespace bound_c_n_l827_827686

theorem bound_c_n (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, a (n + 1) = a n * (a n - 1)) →
  (∀ n, 2^b n = a n) →
  (∀ n, 2^(n - c n) = b n) →
  ∃ (m M : ℝ), (m = 0) ∧ (M = 1) ∧ ∀ n > 0, m ≤ c n ∧ c n ≤ M :=
by
  intro h1 h2 h3 h4
  use 0
  use 1
  sorry

end bound_c_n_l827_827686


namespace log_sufficient_not_necessary_l827_827960

theorem log_sufficient_not_necessary (m : ℝ) (h_log : Real.log_base 6 m = -1) : 
  (∃ m' : ℝ, (m' = 1/6 ∨ m' = 0) ∧ ∀ x y : ℝ, x + 2 * m' * y - 1 = 0 → (3 * m' - 1) * x - m' * y - 1 = 0) :=
sorry

end log_sufficient_not_necessary_l827_827960


namespace count_even_three_digit_numbers_l827_827535

theorem count_even_three_digit_numbers : 
  let digits := {1, 2, 3, 4, 5}
  in ∑ a in digits, ∑ b in digits \ {a}, ∑ c in {2, 4} if b != c,
     (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) = 24 := 
sorry

end count_even_three_digit_numbers_l827_827535


namespace angle_in_third_quadrant_l827_827692

theorem angle_in_third_quadrant (theta : ℝ) (h_theta : theta = -11 * π / 4) : 
  π < theta % (2 * π) + 2 * π < 3 * π / 2 :=
sorry

end angle_in_third_quadrant_l827_827692


namespace sqrt_floor_probability_zero_l827_827609

theorem sqrt_floor_probability_zero (x : ℝ) (h1 : 400 ≤ x ∧ x ≤ 600) (h2 : ⌊real.sqrt x⌋ = 23) : 
  probability (⌊real.sqrt (100 * x)⌋ = 480) = 0 := by
  sorry

end sqrt_floor_probability_zero_l827_827609


namespace soap_total_weight_l827_827593

theorem soap_total_weight 
  (initial_weight : ℝ)
  (return_weight : ℝ)
  (perfume_bottles : ℕ)
  (perfume_weight_oz_per_bottle : ℝ)
  (chocolate_weight_lb : ℝ)
  (jam_jars : ℕ)
  (jam_weight_oz_per_jar : ℝ)
  (oz_to_lb : ℝ) :
  let total_perfume_weight_lb := (perfume_bottles * perfume_weight_oz_per_bottle) / oz_to_lb,
      total_jam_weight_lb := (jam_jars * jam_weight_oz_per_jar) / oz_to_lb,
      total_known_weight := chocolate_weight_lb + total_perfume_weight_lb + total_jam_weight_lb,
      additional_weight := return_weight - initial_weight
  in additional_weight - total_known_weight = 0.625 :=
sorry

end soap_total_weight_l827_827593


namespace grades_with_fewer_students_l827_827178

-- Definitions of the involved quantities
variables (G1 G2 G5 G1_2 : ℕ)
variables (Set_X : ℕ)

-- Conditions given in the problem
theorem grades_with_fewer_students (h1: G1_2 = Set_X + 30) (h2: G5 = G1 - 30) :
  exists Set_X, G1_2 - Set_X = 30 :=
by 
  sorry

end grades_with_fewer_students_l827_827178


namespace area_ratio_triangle_l827_827088

open EuclideanGeometry

noncomputable def PA {A B C P : Point} : Vector := (vec A) - (vec P)
noncomputable def PB {A B C P : Point} : Vector := (vec B) - (vec P)
noncomputable def PC {A B C P : Point} : Vector := (vec C) - (vec P)

theorem area_ratio_triangle (A B C P : Point)
  (h : PA A B C P + 3 * PB A B C P + 2 * PC A B C P = 0) :
  (area (triangle A B C)) / (area (triangle A P B)) = 3 :=
sorry

end area_ratio_triangle_l827_827088


namespace turtles_remaining_on_log_l827_827764

-- Define the initial conditions
def original_turtles := 9
def additional_turtles := (3 * original_turtles) - 2
def total_group := original_turtles + additional_turtles
def frightened_turtles := total_group / 2

-- Theorem statement
theorem turtles_remaining_on_log : total_group - frightened_turtles = 17 :=
by
  sorry

end turtles_remaining_on_log_l827_827764


namespace sum_primes_10_20_l827_827228

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827228


namespace natural_integer_solutions_l827_827856

theorem natural_integer_solutions (x y z : ℕ) :
  x^2 + y^2 = 9 + z^2 - 2 * x * y ↔ 
  (x, y, z) ∈ {(0, 5, 4), (1, 4, 4), (2, 3, 4), (3, 2, 4), (4, 1, 4), (5, 0, 4), 
               (0, 3, 0), (1, 2, 0), (2, 1, 0), (3, 0, 0)} :=
by sorry

end natural_integer_solutions_l827_827856


namespace sum_of_primes_between_10_and_20_is_60_l827_827273

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827273


namespace trapezoid_perimeter_l827_827183

theorem trapezoid_perimeter (AB CD BC DA : ℝ) (BCD_angle : ℝ)
  (h1 : AB = 60) (h2 : CD = 40) (h3 : BC = DA) (h4 : BCD_angle = 120) :
  AB + BC + CD + DA = 220 := 
sorry

end trapezoid_perimeter_l827_827183


namespace smallest_integer_y_solution_l827_827719

theorem smallest_integer_y_solution :
  ∃ y : ℤ, (∀ z : ℤ, (z / 4 + 3 / 7 > 9 / 4) → (z ≥ y)) ∧ (y = 8) := 
by
  sorry

end smallest_integer_y_solution_l827_827719


namespace cube_partition_valid_l827_827895

-- Defining the set of cube side lengths
def cubes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

-- Defining group 1 and group 2 based on the provided solution
def group1 : List ℕ := [1, 4, 6, 7, 10, 11, 13, 16]
def group2 : List ℕ := [2, 3, 5, 8, 9, 12, 14, 15]

-- Prove that the partition provides groups with equal volumes, lateral surface areas, edge lengths, and counts of cubes
theorem cube_partition_valid :
  (group1.sum (λ x => x ^ 3) = group2.sum (λ x => x ^ 3)) ∧
  (group1.sum (λ x => 6 * x ^ 2) = group2.sum (λ x => 6 * x ^ 2)) ∧
  (group1.sum (λ x => 12 * x) = group2.sum (λ x => 12 * x)) ∧
  (group1.length = group2.length) :=
by
  -- This skips the proof as instructed
  sorry

end cube_partition_valid_l827_827895


namespace sum_of_primes_between_10_and_20_l827_827300

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827300


namespace sequence_a8_value_l827_827941

theorem sequence_a8_value :
  ∃ a : ℕ → ℚ, a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) / a n = n / (n + 1)) ∧ a 8 = 1 / 8 :=
by
  -- To be proved
  sorry

end sequence_a8_value_l827_827941


namespace coeff_x2_in_expansion_of_2x_minus_3_to_5_l827_827158

theorem coeff_x2_in_expansion_of_2x_minus_3_to_5 :
  let general_term (r : ℕ) := (−3 : ℤ)^r * 2^(5 - r) * Nat.choose 5 r * (x : ℤ)^(5 - r) in
  let r := 3 in
  let coeff_of_x2 := (−3 : ℤ)^r * 2^(5 - r) * Nat.choose 5 r in
  coeff_of_x2 = -1080 :=
by
  sorry

end coeff_x2_in_expansion_of_2x_minus_3_to_5_l827_827158


namespace meet_time_l827_827821

-- Definitions from Problem Conditions
def Cassie_departure_time := 8.5 -- 8:30 AM in hours
def Cassie_speed := 12 -- miles per hour
def Brian_departure_time := 9.0 -- 9:00 AM in hours
def Brian_speed := 16 -- miles per hour
def route_distance := 62 -- miles

-- Problem Statement encoded in Lean
theorem meet_time :
  let x := (route_distance + (Brian_speed / 2)) / (Cassie_speed + Brian_speed)
  Cassie_departure_time + x = 11 := by sorry

end meet_time_l827_827821


namespace sum_primes_10_to_20_l827_827291

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827291


namespace sara_received_quarters_correct_l827_827141

-- Define the initial number of quarters Sara had
def sara_initial_quarters : ℕ := 21

-- Define the total number of quarters Sara has now
def sara_total_quarters : ℕ := 70

-- Define the number of quarters Sara received from her dad
def sara_received_quarters : ℕ := 49

-- State that the number of quarters Sara received can be deduced by the difference
theorem sara_received_quarters_correct :
  sara_total_quarters = sara_initial_quarters + sara_received_quarters :=
by simp [sara_initial_quarters, sara_total_quarters, sara_received_quarters]

end sara_received_quarters_correct_l827_827141


namespace f_2016_eq_2_l827_827518

noncomputable def f : ℝ → ℝ
| x := if x < 0 then x^5 - 1 else
       if -1 ≤ x ∧ x ≤ 1 then if x < 0 then -f (-x) else if x = 0 then 0 else f x
       else f (x - 1)

theorem f_2016_eq_2 :
  (∀ x : ℝ, x < 0 → f x = x^5 - 1) →
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f x) →
  (∀ x : ℝ, x > 0 → f (x + 1) = f x) →
  f 2016 = 2 :=
by
  intros h1 h2 h3
  sorry

end f_2016_eq_2_l827_827518


namespace green_preference_percentage_l827_827156

theorem green_preference_percentage :
  let total_responses := 70 + 80 + 50 + 50 + 30 + 20 in
  let green_responses := 50 in
  (green_responses / total_responses.toFloat) * 100 = 16.67
:= by
  let total_responses := 70 + 80 + 50 + 50 + 30 + 20
  let green_responses := 50
  have total_responses_eq : total_responses = 300 := by 
    unfold total_responses
    norm_num
  have green_percentage : (green_responses.toFloat / total_responses.toFloat) * 100 = 16.67 := by
    rw [total_responses_eq]
    calc 
      (50 : Float) / (300 : Float) * 100
      _ = (5 / 30) * 100 : by norm_num
      _ = 16.67 : by norm_num
  exact green_percentage

end green_preference_percentage_l827_827156


namespace average_weight_increase_l827_827155

theorem average_weight_increase (A : ℝ) (X : ℝ) (h : (8 * A - 65 + 93) / 8 = A + X) :
  X = 3.5 :=
sorry

end average_weight_increase_l827_827155


namespace min_value_on_neg1_0_l827_827494

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 2^x

theorem min_value_on_neg1_0 (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b)
  (h₂ : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f a b x ≤ 4) :
  (f a b (-1) = -3 / 2) :=
by
  -- Given: a and b are positive real numbers
  have hab : a + b = 2 :=
    sorry -- We know the maximum value of f(x) on [0,1] is 4 at x=1, hence a + b = 2
  
  -- Show: minimum value of f(x) on [-1,0] is -3/2
  have hmin : f a b (-1) = -(a + b) + 2^(-1) :=
    sorry -- Calculation for f(-1)
  
  rw hab at hmin
  exact hmin

end min_value_on_neg1_0_l827_827494


namespace calculate_expr_l827_827399

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l827_827399


namespace find_m_and_cartesian_equation_l827_827572

open Real

theorem find_m_and_cartesian_equation (m : ℝ) :
  (∀ t : ℝ, (∃ t : ℝ, (1/2) * t = x) ∧ (m + (sqrt 3/2) * t = y))
  ∧ (∀ θ : ℝ, 4*cos(θ - π/6) = p) →
  ((x^2 + y^2 - 2 * sqrt 3 * x - 2 * y = 0)
  ∧ (∀ P Q : ℝ × ℝ, 
       minimum_distance (sqrt 3 * fst P - snd P + m = 0)
       ((fst Q - sqrt 3)^2 + (snd Q - 1)^2 = 4) 
       = 1 →
    m = 4 ∨ m = -8)) := sorry

end find_m_and_cartesian_equation_l827_827572


namespace combined_rocket_height_l827_827076

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  sorry

end combined_rocket_height_l827_827076


namespace s_minus_q_l827_827812

theorem s_minus_q (p q r s : ℕ) (h1 : p^3 = q^2) (h2 : r^5 = s^4) (h3 : r - p = 31) : s - q = -2351 :=
by
  sorry

end s_minus_q_l827_827812


namespace slope_l3_is_5_over_6_l827_827128

noncomputable theory

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, -3)
def B : ℝ × ℝ := (2, 2)
def C (x : ℝ) : ℝ × ℝ := (x, 2)

-- Define the lines l₁, l₂, and the constraint for l₃ passing through A and C
def line_l1 (p : ℝ × ℝ) : Prop := 4 * p.1 - 3 * p.2 = 2
def line_l2 (p : ℝ × ℝ) : Prop := p.2 = 2

-- Define the area of triangle ABC
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Define the slope function of a line given two points
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- The proof problem statement
theorem slope_l3_is_5_over_6 : ∃ x : ℝ, 
  (line_l1 B) ∧ (line_l1 A) ∧ (line_l2 B) ∧ (line_l2 (C x)) ∧
  (area_of_triangle A B (C x) = 5) ∧ (slope A (C x) = 5 / 6) :=
sorry

end slope_l3_is_5_over_6_l827_827128


namespace sum_of_primes_between_10_and_20_l827_827251

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827251


namespace immigration_per_year_l827_827171

-- Definitions based on the initial conditions
def initial_population : ℕ := 100000
def birth_rate : ℕ := 60 -- this represents 60%
def duration_years : ℕ := 10
def emigration_per_year : ℕ := 2000
def final_population : ℕ := 165000

-- Theorem statement: The number of people that immigrated per year
theorem immigration_per_year (immigration_per_year : ℕ) :
  immigration_per_year = 2500 :=
  sorry

end immigration_per_year_l827_827171


namespace turtles_remaining_on_log_l827_827767
-- Importing necessary modules

-- Defining the problem
def initial_turtles : ℕ := 9
def turtles_climbed : ℕ := (initial_turtles * 3) - 2
def total_turtles : ℕ := initial_turtles + turtles_climbed
def remaining_turtles : ℕ := total_turtles / 2

-- Stating the proof problem
theorem turtles_remaining_on_log : remaining_turtles = 17 := 
  sorry

end turtles_remaining_on_log_l827_827767


namespace division_error_1990_l827_827651

theorem division_error_1990 (A : ℕ) (d : ℕ) (h₁ : 0 ≤ d ∧ d ≤ 9) 
  (h₂ : ∃ n : ℕ, (A : ℚ) / 1990 = ∑ i in finset.range n, d*(10:ℚ)^(-i))
  : false :=
sorry

end division_error_1990_l827_827651


namespace trees_planted_l827_827558

/-
In a garden, some trees are planted at equal distances along a yard 700 meters long,
one tree being at each end of the yard. The distance between two consecutive trees
is 28 meters. Prove that the number of trees planted in the garden is 26.
-/

theorem trees_planted (yard_length : ℕ) (dist_between_trees : ℕ) (h_yard : yard_length = 700) (h_dist : dist_between_trees = 28) :
  ∃ n : ℕ, n = 26 ∧ yard_length / dist_between_trees + 1 = n :=
by
  use 26
  sorry

end trees_planted_l827_827558


namespace parabola_line_equation_l827_827912

-- Define the focus of the parabola y^2 = 8x.
def F : ℝ × ℝ := (2, 0)

-- Definition of the parabola
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1

-- Define the collinearity condition that FM = 4 FP
def collinear_points (P M : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k = 4 ∧
  (M.1 - F.1, M.2 - F.2) = k * (P.1 - F.1, P.2 - F.2)

-- Proof statement we need to show
theorem parabola_line_equation (P M : ℝ × ℝ)
  (hP : parabola P)
  (hM : collinear_points P M) :
  ∃ k : ℝ, k = 2 * √2 ∧ (P.2 = k * (P.1 - 2) ∨ P.2 = -k * (P.1 - 2)) :=
sorry

end parabola_line_equation_l827_827912


namespace tangent_line_through_point_area_between_curve_and_line_l827_827005

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 1

theorem tangent_line_through_point :
  ∃ (m b : ℝ), (∀ x, f x = m * x + b) ∧ (m = 0 ∧ b = 1 ∨ m = -2 ∧ b = 3) :=
by
  sorry

theorem area_between_curve_and_line :
  (∫ x in -2..2, abs (f x - (-1))) = 16/3 :=
by
  sorry

end tangent_line_through_point_area_between_curve_and_line_l827_827005


namespace inequality_proof_l827_827886

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  sorry

end inequality_proof_l827_827886


namespace integer_sequence_2000_digits_l827_827168

theorem integer_sequence_2000_digits (a b : ℕ) (h : ∀ (n m : ℕ), (n < 2000) → (m = n + 1) → (17 ∣ 10 * digit X n + digit X m ∨ 23 ∣ 10 * digit X n + digit X m)) :
  a = 2 ∧ b = 5 → a + b = 7 :=
by
  sorry

end integer_sequence_2000_digits_l827_827168


namespace sum_primes_between_10_and_20_l827_827285

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827285


namespace candy_given_l827_827868

theorem candy_given (A R G : ℕ) (h1 : A = 15) (h2 : R = 9) : G = 6 :=
by
  sorry

end candy_given_l827_827868


namespace no_solution_iff_discriminant_l827_827975

theorem no_solution_iff_discriminant (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) ↔ -2 ≤ k ∧ k ≤ 2 := by
  sorry

end no_solution_iff_discriminant_l827_827975


namespace lynne_final_cost_l827_827621

theorem lynne_final_cost : 
  let cat_books := 7 * 7
  let dog_books := 5 * 6
  let sports_books := 4 * 9
  let solar_books := 2 * 11
  let gardening_magazines := 3 * 4
  let cooking_magazines := 4 * 5
  let fashion_magazines := 2 * 6
  let cat_dog_discount := (cat_books + dog_books) * 0.10
  let solar_discount := solar_books * 0.15
  let sports_discount := 9
  let total_discount := cat_dog_discount + solar_discount + sports_discount
  let total_cost_before_tax := (cat_books + dog_books + solar_books + sports_books + gardening_magazines + cooking_magazines + fashion_magazines) - total_discount
  let sales_tax := total_cost_before_tax * 0.06
  let final_cost := total_cost_before_tax + sales_tax
  final_cost.roundNearestCent = 170.45 := 
by
  let cat_books := 7 * 7
  let dog_books := 5 * 6
  let sports_books := 4 * 9
  let solar_books := 2 * 11
  let gardening_magazines := 3 * 4
  let cooking_magazines := 4 * 5
  let fashion_magazines := 2 * 6
  let cat_dog_discount := (cat_books + dog_books) * 0.10
  let solar_discount := solar_books * 0.15
  let sports_discount := 9
  let total_discount := cat_dog_discount + solar_discount + sports_discount
  let total_cost_before_tax := (cat_books + dog_books + solar_books + sports_books + gardening_magazines + cooking_magazines + fashion_magazines) - total_discount
  let sales_tax := total_cost_before_tax * 0.06
  let final_cost := total_cost_before_tax + sales_tax
  have h : total_discount = (49 + 30 + 22 + 36) * 0.10 + 3.30 + 9 := by sorry
  have h1 : total_cost_before_tax - total_discount = 160.80 := by sorry
  have h2 : final_cost.roundNearestCent = 170.45 := by sorry
  exact h2

end lynne_final_cost_l827_827621


namespace range_of_a_l827_827123

def p (x : ℝ) : Prop := (4 * x - 3)^2 ≤ 1
def q (x : ℝ) (a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

def A : set ℝ := { x | 1 / 2 ≤ x ∧ x ≤ 1 }
def B (a : ℝ) : set ℝ := { x | a ≤ x ∧ x ≤ a + 1 }

theorem range_of_a (a : ℝ) : (¬∀ x, p x → q x a) ∧ ¬(¬∀ x, q x a → p x) → (0 ≤ a ∧ a ≤ 1 / 2) :=
by
  sorry

end range_of_a_l827_827123


namespace reflect_point_is_B_l827_827368

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def plane (p : Point3D) : Prop := p.x + p.y + p.z = 10

def pointA : Point3D := {x := -4, y := 10, z := 10}
def pointC : Point3D := {x := 4, y := 6, z := 8}
def pointB : Point3D := {x := 118 / 43, y := 244 / 43, z := 202 / 43}

theorem reflect_point_is_B : 
  ∃ B : Point3D, 
    (plane B) ∧ 
    (B.x = 118 / 43) ∧ 
    (B.y = 244 / 43) ∧ 
    (B.z = 202 / 43) :=
by
  use pointB
  split
  · exact rfl -- plane B holds
  · exact rfl -- B.x holds
  · exact rfl -- B.y holds
  · exact rfl -- B.z holds

end reflect_point_is_B_l827_827368


namespace hemisphere_radius_and_surface_area_l827_827001

theorem hemisphere_radius_and_surface_area
  (V : ℝ) (radius : ℝ) (surface_area : ℝ) 
  (hV : V = 19404)
  (hVolume : V = 2 / 3 * Real.pi * radius^3)
  (hSurfaceArea : surface_area = 3 * Real.pi * radius^2) :
  radius ≈ 21.08 ∧ surface_area ≈ 4186.98 :=
by
  sorry

end hemisphere_radius_and_surface_area_l827_827001


namespace isabel_money_left_l827_827585

theorem isabel_money_left (initial_amount : ℕ) (half_toy_expense half_book_expense money_left : ℕ) :
  initial_amount = 204 →
  half_toy_expense = initial_amount / 2 →
  half_book_expense = (initial_amount - half_toy_expense) / 2 →
  money_left = initial_amount - half_toy_expense - half_book_expense →
  money_left = 51 :=
by
  intros h1 h2 h3 h4
  sorry

end isabel_money_left_l827_827585


namespace students_arrangement_are_960_l827_827144

def students_permutation (students : Fin 7) (adjAB : students ![(0,1)] ∨ students ![(1,0)]) (not_adjCD : ¬(students ![(2,3)] ∨ students ![(3,2)])) : Nat :=
  960

theorem students_arrangement_are_960 : ∀ (students : Fin 7)
  (adjAB : students ![(0,1)] ∨ students ![(1,0)])
  (not_adjCD : ¬(students ![(2,3)] ∨ students ![(3,2)])),
  students_permutation students adjAB not_adjCD = 960 :=
by sorry

end students_arrangement_are_960_l827_827144


namespace geometric_sequence_a6_l827_827580

theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 4) (h2 : a 4 = 2) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 6 = 4 :=
sorry

end geometric_sequence_a6_l827_827580


namespace surface_area_calculation_l827_827748

noncomputable def surface_area_of_modified_cube : ℕ :=
let initial_cube_size := 12
let size := 6
let smaller_cube_size := 2
let num_smaller_cubes := 7
let exposed_area_per_face := smaller_cube_size * smaller_cube_size * 4  -- 4 faces per removed smaller cube per face
let initial_surface_area_per_smaller_cube := size * size * 6
let added_surface_area_per_smaller_cube := exposed_area_per_face * 6 -- each face of smaller cubes
let modified_surface_area_per_smaller_cube := initial_surface_area_per_smaller_cube + added_surface_area_per_smaller_cube
let total_initial_surface_area := num_smaller_cubes * modified_surface_area_per_smaller_cube
let shared_internal_faces := 12 * (smaller_cube_size * smaller_cube_size)
let final_surface_area := total_initial_surface_area - shared_internal_faces
in final_surface_area

theorem surface_area_calculation : surface_area_of_modified_cube = 1752 :=
by
  sorry

end surface_area_calculation_l827_827748


namespace first_alloy_amount_15_l827_827568

noncomputable def amount_of_first_alloy_used (x : ℝ) : Prop :=
  let chromium_first := 0.10 * x
  let chromium_second := 0.06 * 35
  let total_weight_third := x + 35
  let chromium_third := 0.072 * total_weight_third
  chromium_first + chromium_second = chromium_third

theorem first_alloy_amount_15 :
  amount_of_first_alloy_used 15 :=
by
  unfold amount_of_first_alloy_used
  -- directly assert the solution
  rw [←eq.symm (by norm_num : 0.10 * 15 = 1.5)]
  rw [←eq.symm (by norm_num : 0.06 * 35 = 2.1)]
  rw [←eq.symm (by norm_num : 15 + 35 = 50)]
  rw [←eq.symm (by norm_num : 0.072 * 50 = 3.6)]
  norm_num

end first_alloy_amount_15_l827_827568


namespace sum_of_first_five_terms_is_31_l827_827486

variable (a : ℕ → ℝ) (q : ℝ)

-- The geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Condition 1: a_2 * a_3 = 2 * a_1
def condition1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 3 = 2 * a 1

-- Condition 2: The arithmetic mean of a_4 and 2 * a_7 is 5/4
def condition2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 2 * a 7) / 2 = 5 / 4

-- Sum of the first 5 terms of the geometric sequence
def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

-- The theorem to prove
theorem sum_of_first_five_terms_is_31 (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) 
  (hc1 : condition1 a q) 
  (hc2 : condition2 a q) : 
  S_5 a = 31 := by
  sorry

end sum_of_first_five_terms_is_31_l827_827486


namespace find_missing_digit_divisible_by_9_l827_827162

theorem find_missing_digit_divisible_by_9 :
  ∃ (x : ℕ), 2 + 4 + 6 + 8 + x ≡ 0 [MOD 9] ∧ x = 7 :=
begin
  use 7,
  split,
  { norm_num },
  { refl }
end

end find_missing_digit_divisible_by_9_l827_827162


namespace arc_length_correct_l827_827411

noncomputable def arcLengthOfCurve : ℝ :=
  ∫ φ in (0 : ℝ)..(5 * Real.pi / 12), (2 : ℝ) * (Real.sqrt (φ ^ 2 + 1))

theorem arc_length_correct :
  arcLengthOfCurve = (65 / 144) + Real.log (3 / 2) := by
  sorry

end arc_length_correct_l827_827411


namespace bracelet_arrangements_l827_827060

theorem bracelet_arrangements (n : ℕ) (h : n = 8) : 
  let factorial := Nat.factorial n in
  let rotations := n in
  let reflections := 2 in
  factorial / (rotations * reflections) = 2520 :=
by
  have h_factorial : Nat.factorial 8 = 40320 := rfl
  rw [h, h_factorial]
  have rotations_reflections : 8 * 2 = 16 := rfl
  rw [rotations_reflections]
  norm_num
  sorry

end bracelet_arrangements_l827_827060


namespace initial_population_l827_827747

theorem initial_population (P : ℝ) (h1 : 0.76 * P = 3553) : P = 4678 :=
by
  sorry

end initial_population_l827_827747


namespace Uncle_Fyodor_age_l827_827630

variable (age : ℕ)

-- Conditions from the problem
def Sharik_statement : Prop := age > 11
def Matroskin_statement : Prop := age > 10

-- The theorem stating the problem to be proved
theorem Uncle_Fyodor_age
  (H : (Sharik_statement age ∧ ¬Matroskin_statement age) ∨ (¬Sharik_statement age ∧ Matroskin_statement age)) :
  age = 11 :=
by
  sorry

end Uncle_Fyodor_age_l827_827630


namespace unused_card_is_one_l827_827311

theorem unused_card_is_one (cards : List ℕ) (H : cards = [4, 3, 1]) :
  ∃ n: ℕ, n ∈ cards ∧ (∀ a b: ℕ, a ∈ cards → b ∈ cards → a ≠ b → 10 * a + b ≤ 43 → n = 1) := 
by
  exists 1
  split
  trivial
  intro a b _ _ _ h
  sorry

end unused_card_is_one_l827_827311


namespace better_partial_repayment_now_l827_827830

variable (T : ℝ) (S : ℝ) (r : ℝ)

theorem better_partial_repayment_now : 
  let immediate_balance := S - (T - 0.5 * r * S) - T + 0.5 * r * (S - T + 0.5 * r * S) in
  let end_balance := S - 2 * T + r * S in
  immediate_balance < end_balance :=
by {
  sorry
}

end better_partial_repayment_now_l827_827830


namespace value_of_ON_l827_827364

-- Definitions from the conditions
def ellipse (M : ℝ × ℝ) : Prop := M.1^2 / 25 + M.2^2 / 9 = 1

def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_midpoint (N A B : ℝ × ℝ) := (N.1 = (A.1 + B.1) / 2) ∧ (N.2 = (A.2 + B.2) / 2)

def left_focus : ℝ × ℝ := (-4, 0) -- Assuming this based on typical properties of ellipse.
def right_focus : ℝ × ℝ := (4, 0)  -- Based on typical horizontal ellipse properties.

theorem value_of_ON 
  (M : ℝ × ℝ) (N : ℝ × ℝ) (O : ℝ × ℝ) :
  ellipse M →
  distance M left_focus = 2 →
  is_midpoint N M left_focus →
  is_midpoint O left_focus right_focus →
  distance (0, 0) N = 4 :=
by
  sorry

end value_of_ON_l827_827364


namespace immediate_prepayment_better_l827_827828

variables {T S r : ℝ}

theorem immediate_prepayment_better (T S r : ℝ) :
    let immediate_prepayment_balance := S - 2 * T + r * S - 0.5 * r * T + (0.5 * r * S)^2 in
    let end_period_balance := S - 2 * T + r * S in
    immediate_prepayment_balance < end_period_balance :=
by
  -- Proof goes here
  sorry

end immediate_prepayment_better_l827_827828


namespace fraction_difference_l827_827962

variable {x y : ℝ} (h : xy = x - y ∧ xy ≠ 0) 

theorem fraction_difference (h : xy = x - y ∧ xy ≠ 0) : 
  (\frac{1}{y} - \frac{1}{x}) = 1 := 
sorry

end fraction_difference_l827_827962


namespace chips_left_uneaten_l827_827427

theorem chips_left_uneaten 
    (chips_per_cookie : ℕ)
    (cookies_per_dozen : ℕ)
    (dozens_of_cookies : ℕ)
    (cookies_eaten_ratio : ℕ) 
    (h_chips : chips_per_cookie = 7)
    (h_cookies_dozen : cookies_per_dozen = 12)
    (h_dozens : dozens_of_cookies = 4)
    (h_eaten_ratio : cookies_eaten_ratio = 2) : 
  (cookies_per_dozen * dozens_of_cookies / cookies_eaten_ratio) * chips_per_cookie = 168 :=
by 
  sorry

end chips_left_uneaten_l827_827427


namespace proof_of_sum_f_values_l827_827517

theorem proof_of_sum_f_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a + a⁻¹ = 3) :
  let f (x : ℝ) := a^x + a^(-x) in
  f 0 + f 1 + f 2 = 12 :=
by
  sorry

end proof_of_sum_f_values_l827_827517


namespace cube_volume_proof_l827_827663

noncomputable def cube_volume (AG : ℝ) : ℝ :=
  let s := AG / Real.sqrt 3 in s^3

theorem cube_volume_proof (AG : ℝ) (h : AG = 15) : cube_volume AG = 375 * Real.sqrt 3 :=
by
  have s_eq : s = 5 * Real.sqrt 3 := by sorry
  have volume_eq : cube_volume AG = (5 * Real.sqrt 3)^3 := by sorry
  rw volume_eq
  -- Continue steps to show volume equals 375 * Real.sqrt 3
  sorry

end cube_volume_proof_l827_827663


namespace integer_count_in_pi_interval_l827_827955

noncomputable def number_of_integers_in_interval : ℕ :=
  let lower_bound := -5 * Real.pi
  let upper_bound := 12 * Real.pi
  let lower_integer := Int.floor lower_bound
  let upper_integer := Int.ceil upper_bound
  let count := upper_integer - lower_integer + 1
  count

theorem integer_count_in_pi_interval :
  number_of_integers_in_interval = 53 :=
by
  sorry

end integer_count_in_pi_interval_l827_827955


namespace no_perfect_squares_xy_zt_l827_827909

theorem no_perfect_squares_xy_zt
    (x y z t : ℕ) 
    (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < t)
    (h_eq1 : x + y = z + t) 
    (h_eq2 : xy - zt = x + y) : ¬(∃ a b : ℕ, xy = a^2 ∧ zt = b^2) :=
by
  sorry

end no_perfect_squares_xy_zt_l827_827909


namespace initial_money_l827_827450

theorem initial_money (x : ℝ) (cupcake_cost total_cookie_cost total_cost money_left : ℝ) 
  (h1 : cupcake_cost = 10 * 1.5) 
  (h2 : total_cookie_cost = 5 * 3)
  (h3 : total_cost = cupcake_cost + total_cookie_cost)
  (h4 : money_left = 30)
  (h5 : 3 * x = total_cost + money_left) 
  : x = 20 := 
sorry

end initial_money_l827_827450


namespace x_intercept_of_line_l827_827772

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

def line_equation (p1 : ℝ × ℝ) (m : ℝ) : ℝ → ℝ :=
  λ x, m * (x - p1.1) + p1.2

theorem x_intercept_of_line :
  ∃ x, (line_equation (2, -2) (-2)) x = 0 :=
begin
  use 1,
  simp [line_equation],
  norm_num,
end

end x_intercept_of_line_l827_827772


namespace proof_m_div_x_plus_y_l827_827618

variables (a b c x y m : ℝ)

-- 1. The ratio of 'a' to 'b' is 4 to 5
axiom h1 : a / b = 4 / 5

-- 2. 'c' is half of 'a'.
axiom h2 : c = a / 2

-- 3. 'x' equals 'a' increased by 27 percent of 'a'.
axiom h3 : x = 1.27 * a

-- 4. 'y' equals 'b' decreased by 16 percent of 'b'.
axiom h4 : y = 0.84 * b

-- 5. 'm' equals 'c' increased by 14 percent of 'c'.
axiom h5 : m = 1.14 * c

theorem proof_m_div_x_plus_y : m / (x + y) = 0.2457 :=
by
  -- Proof goes here
  sorry

end proof_m_div_x_plus_y_l827_827618


namespace charge_per_square_foot_l827_827080

noncomputable def area_one_lawn : ℕ := 20 * 15
noncomputable def num_lawns : ℕ := 3
noncomputable def additional_area : ℕ := 600
noncomputable def book_cost : ℝ := 150

theorem charge_per_square_foot :
  let total_area := num_lawns * area_one_lawn + additional_area in
  (book_cost / total_area : ℝ) = 0.10 :=
by sorry

end charge_per_square_foot_l827_827080


namespace cylinder_height_relationship_l827_827707

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
π * r^2 * h

theorem cylinder_height_relationship (V : ℝ) (r1 r2 h1 h2 : ℝ)
  (hV1 : cylinder_volume r1 h1 = V)
  (hV2 : cylinder_volume r2 h2 = V)
  (hRadii : r2 = 1.2 * r1) : h1 = 1.44 * h2 :=
by
  unfold cylinder_volume at *
  sorry

end cylinder_height_relationship_l827_827707


namespace turtle_finishes_in_10_minutes_l827_827367

def skunk_time : ℕ := 6
def rabbit_speed_ratio : ℕ := 3
def turtle_speed_ratio : ℕ := 5
def rabbit_time := skunk_time / rabbit_speed_ratio
def turtle_time := turtle_speed_ratio * rabbit_time

theorem turtle_finishes_in_10_minutes : turtle_time = 10 := by
  sorry

end turtle_finishes_in_10_minutes_l827_827367


namespace eccentricity_range_of_ellipse_l827_827914

theorem eccentricity_range_of_ellipse
  (F₁ F₂ : EuclideanSpace ℝ (Fin 2))
  (M : EuclideanSpace ℝ (Fin 2))
  (a b c : ℝ)
  (eccentricity : ℝ)
  (h1 : dot_product (M - F₁) (M - F₂) = 0)
  (h2 : (M - F₁).norm < a ∧ (M - F₂).norm < a)
  (h3 : c < b)
  (h4 : b^2 = a^2 - c^2) :
  0 < eccentricity ∧ eccentricity < real.sqrt 2 / 2 :=
sorry

end eccentricity_range_of_ellipse_l827_827914


namespace smallest_integer_l827_827673

theorem smallest_integer (x : ℕ) (n : ℕ) (h_pos : 0 < x)
  (h_gcd : Nat.gcd 30 n = x + 3)
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) : n = 70 :=
begin
  sorry
end

end smallest_integer_l827_827673


namespace sum_primes_between_10_and_20_l827_827244

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827244


namespace find_b_l827_827514

theorem find_b
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) -> True)
  (h4 : ∃ e, e = (Real.sqrt 5) / 3)
  (h5 : 2 * a = 12) :
  b = 4 :=
by
  sorry

end find_b_l827_827514


namespace find_x_l827_827698

theorem find_x (x y : ℝ)
  (h1 : 2 * x + (x - 30) = 360)
  (h2 : y = x - 30)
  (h3 : 2 * x = 4 * y) :
  x = 130 := 
sorry

end find_x_l827_827698


namespace area_triangle_DEF_l827_827068

noncomputable def triangleDEF (DE EF DF : ℝ) (angleDEF : ℝ) : ℝ :=
  if angleDEF = 60 ∧ DF = 3 ∧ EF = 6 / Real.sqrt 3 then
    1 / 2 * DE * EF * Real.sin (Real.pi / 3)
  else
    0

theorem area_triangle_DEF :
  triangleDEF (Real.sqrt 3) (6 / Real.sqrt 3) 3 60 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end area_triangle_DEF_l827_827068


namespace identical_mantissas_iff_perfect_squares_l827_827071

theorem identical_mantissas_iff_perfect_squares (m n : ℕ) (h_diff : m ≠ n) :
  (∀ (k : ℤ), (∃ (√m : ℝ), √m^2 = m) ∧ (∃ (√n : ℝ), √n^2 = n) ∧ (∃ (mantissa : ℝ), √m - int.natAbs mantissa = √n * (mantissa / mantissa) - int.natAbs mantissa)) ↔ 
  (∃ (k1 k2 : ℤ), k1 * k1 = m ∧ k2 * k2 = n) :=
by
  sorry

end identical_mantissas_iff_perfect_squares_l827_827071


namespace area_of_rhombus_l827_827616

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 4) (h2 : d2 = 4) :
    (d1 * d2) / 2 = 8 := by
  sorry

end area_of_rhombus_l827_827616


namespace total_prairie_area_l827_827365

theorem total_prairie_area (A B C : ℕ) (Z1 Z2 Z3 : ℚ) (unaffected : ℕ) (total_area : ℕ) : 
  A = 55000 →
  B = 35000 →
  C = 45000 →
  Z1 = 0.80 →
  Z2 = 0.60 →
  Z3 = 0.95 →
  unaffected = 1500 →
  total_area = Z1 * A + Z2 * B + Z3 * C + unaffected →
  total_area = 109250 := sorry

end total_prairie_area_l827_827365


namespace part1_part2_part3_l827_827944

open Set

-- Definitions of universal set R, set A, and set B
def R := ℝ
def A : Set ℝ := {x | x^2 - 5 * x + 6 ≥ 0}
def B : Set ℝ := {x | -3 < x + 1 ∧ x + 1 < 3}

-- Lean statements for the equivalent proof problems
theorem part1 : A ∩ B = {x | -4 < x ∧ x < 2} :=
sorry

theorem part2 : A ∪ B = {x | x ≤ 2 ∨ x ≥ 3} :=
sorry

theorem part3 : (R \ A) ∩ B = ∅ :=
sorry

end part1_part2_part3_l827_827944


namespace strange_die_expected_winnings_l827_827798

noncomputable def probabilities : List ℚ := [1/4, 1/4, 1/6, 1/6, 1/6, 1/12]
noncomputable def winnings : List ℚ := [2, 2, 4, 4, -6, -12]

def expected_value (p : List ℚ) (w : List ℚ) : ℚ :=
  List.sum (List.zipWith (λ pi wi => pi * wi) p w)

theorem strange_die_expected_winnings :
  expected_value probabilities winnings = 0.17 :=
by
  sorry

end strange_die_expected_winnings_l827_827798


namespace volume_of_box_l827_827362

-- Definitions of initial conditions
def length_initial : ℝ := 100
def width_initial : ℝ := 50
def side_square : ℝ := 10

-- Definitions of derived dimensions
def length_final : ℝ := length_initial - 2 * side_square
def width_final : ℝ := width_initial - 2 * side_square
def height : ℝ := side_square

-- Theorem statement to be proved
theorem volume_of_box : length_final * width_final * height = 24000 := by
  -- Placeholder for proof
  sorry

end volume_of_box_l827_827362


namespace inequality_proof_l827_827875

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  -- Proof omitted
  sorry

end inequality_proof_l827_827875


namespace binomial_expansion_decimal_example_l827_827655

theorem binomial_expansion_decimal_example
  (x y r : ℝ)
  (h_cond : |x| > |y|)
  (expansion : ∀ n : ℕ, n ≤ 2 → (x + y) ^ r = 
              x^r + r * x^(r-1) * y + (r * (r-1) / 2) * x^(r-2) * y^2 + sum (i in range n, (r * (r - 1) * ... * (r - (i - 1))) / (i! * x^(r - i) * y^i))) : 
  let answer := 428 in 
  (∀ ans : ℕ, ans = answer) := 
sorry

end binomial_expansion_decimal_example_l827_827655


namespace mans_speed_upstream_l827_827776

-- Define the conditions
def V_downstream : ℝ := 15  -- Speed with the current (downstream)
def V_current : ℝ := 2.5    -- Speed of the current

-- Calculate the man's speed against the current (upstream)
theorem mans_speed_upstream : V_downstream - 2 * V_current = 10 :=
by
  sorry

end mans_speed_upstream_l827_827776


namespace parallel_lines_distance_range_l827_827194

-- Define points P and Q
def P : ℝ × ℝ := (-1, 2)
def Q : ℝ × ℝ := (2, -3)

-- Define the function to calculate the distance between two points
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

-- Define the range of distances
def distance_range (P Q : ℝ × ℝ) : set ℝ :=
  {d | 0 < d ∧ d ≤ distance P Q}

-- Proof statement
theorem parallel_lines_distance_range :
  distance_range P Q = {d | 0 < d ∧ d ≤ real.sqrt 34} :=
sorry

end parallel_lines_distance_range_l827_827194


namespace painted_cubes_l827_827581

theorem painted_cubes :
  ∀ (total_cubes no_paint_cubes painted_cubes: ℕ),
    total_cubes = 112 →
    no_paint_cubes = 8 →
    painted_cubes = total_cubes - no_paint_cubes →
    painted_cubes = 104 := by
  intros total_cubes no_paint_cubes painted_cubes ht hn hp
  rw [ht, hn, hp]
  sorry

end painted_cubes_l827_827581


namespace sum_primes_between_10_and_20_l827_827283

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827283


namespace actual_cost_of_article_l827_827804

theorem actual_cost_of_article {x : ℝ} (h : 0.76 * x = 760) : x = 1000 :=
by
  sorry

end actual_cost_of_article_l827_827804


namespace sum_of_positive_integer_solutions_to_congruence_l827_827218

theorem sum_of_positive_integer_solutions_to_congruence :
  (∑ x in Finset.filter (λ x : ℕ, 0 < x ∧ x ≤ 20 ∧ 13 * (3 * x - 2) % 9 = 35 % 9) (Finset.range 21), x) = 15 :=
by
  sorry

end sum_of_positive_integer_solutions_to_congruence_l827_827218


namespace sum_of_primes_between_10_and_20_is_60_l827_827277

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827277


namespace students_receiving_B_lee_l827_827556

def num_students_receiving_B (students_kipling: ℕ) (B_kipling: ℕ) (students_lee: ℕ) : ℕ :=
  let ratio := (B_kipling * students_lee) / students_kipling
  ratio

theorem students_receiving_B_lee (students_kipling B_kipling students_lee : ℕ) 
  (h : B_kipling = 8 ∧ students_kipling = 12 ∧ students_lee = 30) :
  num_students_receiving_B students_kipling B_kipling students_lee = 20 :=
by
  sorry

end students_receiving_B_lee_l827_827556


namespace eventually_stable_votes_l827_827326

noncomputable def voting_system_stabilizes (n : ℕ) (r : ℕ → fin 25 → bool) : Prop :=
  ∀ i : fin 25,
    (r (n + 1) i = r n i) ↔
      ((r n i = r n i.pred 1 ∨ r n i = r n i.succ 1) ∨
       (r n i ≠ r n i.pred 1 ∧ r n i ≠ r n i.succ 1))

theorem eventually_stable_votes : 
  ∀ (r : ℕ → fin 25 → bool), 
    ∃ n, ∀ m ≥ n, ∀ i : fin 25, r (m + 1) i = r m i :=
begin
  sorry
end

end eventually_stable_votes_l827_827326


namespace highest_power_of_3_l827_827654

-- Define the integer M formed by concatenating the 3-digit numbers from 100 to 250
def M : ℕ := sorry  -- We should define it in a way that represents the concatenation

-- Define a proof that the highest power of 3 that divides M is 3^1
theorem highest_power_of_3 (n : ℕ) (h : M = n) : ∃ m : ℕ, 3^m ∣ n ∧ ¬ (3^(m + 1) ∣ n) ∧ m = 1 :=
by sorry  -- We will not provide proofs; we're only writing the statement

end highest_power_of_3_l827_827654


namespace least_prime_factor_of_expression_l827_827213

theorem least_prime_factor_of_expression : 
  ∀ (p : ℕ), p.prime → (p ∣ (11 ^ 5 - 11 ^ 4)) → (p = 2) :=
sorry

end least_prime_factor_of_expression_l827_827213


namespace sum_primes_between_10_and_20_l827_827280

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827280


namespace turtles_remaining_on_log_l827_827770

-- Definition of the problem parameters
def initial_turtles : ℕ := 9
def additional_turtles : ℕ := (3 * initial_turtles) - 2
def total_turtles : ℕ := initial_turtles + additional_turtles
def frightened_turtles : ℕ := total_turtles / 2
def remaining_turtles : ℕ := total_turtles - frightened_turtles

-- The final theorem stating the number of turtles remaining
theorem turtles_remaining_on_log : remaining_turtles = 17 := by
  -- Proof is omitted
  sorry

end turtles_remaining_on_log_l827_827770


namespace sum_of_primes_between_10_and_20_l827_827255

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827255


namespace inequality_proof_l827_827876

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  -- Proof omitted
  sorry

end inequality_proof_l827_827876


namespace min_pos_period_and_range_find_A_and_b_l827_827894

-- Definitions based on the conditions given in the problem
noncomputable def f (x : ℝ) : ℝ := sin(x)^2 + sqrt(3) * sin(x) * cos(x) - 1/2

variable (x : ℝ)
variable (A : ℝ)
variable (a b c : ℝ)

-- Statement for minimum positive period and range of f(x)
theorem min_pos_period_and_range :
  (∀ x, f(x) = sin(2 * x - π/6)) → 
  (∀ T, T > 0 → ∃ k, ∀ x, f(x + k * T) = f(x)) → 
  ∀ x, f(x) ∈ set.Icc (-1) (1) := 
sorry

-- Statement for finding angle A and side b
theorem find_A_and_b (A_acute : 0 < A ∧ A < π / 2) 
  (a_val : a = 2 * sqrt 3)
  (c_val : c = 4)
  (f_A_1 : f(A) = 1) :
  A = π / 3 ∧ b = 2 :=
sorry

-- Note: The proof is provided in the above statements using 'sorry' as a placeholder.

end min_pos_period_and_range_find_A_and_b_l827_827894


namespace percentage_increase_first_year_l827_827470

-- Definitions directly from conditions
def price_at_year_end (initial_price : ℝ) (rate : ℝ) : ℝ := initial_price * (1 + rate/100)
def price_after_drop (initial_price : ℝ) (rate : ℝ) : ℝ := initial_price * (1 - rate/100)
def price_after_rise (initial_price : ℝ) (rate : ℝ) : ℝ := initial_price * (1 + rate/100)

-- The specified condition in the problem
def condition (P X : ℝ) : Prop := price_after_rise (price_after_drop (price_at_year_end P X) 25) 25 = P * 1.125

-- Lean statement to prove
theorem percentage_increase_first_year (P : ℝ) : ∃ X : ℝ, condition P X ∧ X = 20 := sorry

end percentage_increase_first_year_l827_827470


namespace find_b_when_func_max_is_7_l827_827524

-- Definition of the function
def func (b θ : ℝ) : ℝ := 4 * b^2 - 3 * b^2 * (sin (2 * θ)) - 3 * b * (sin θ) + 9 / 4

-- The statement of the problem
theorem find_b_when_func_max_is_7 (b θ : ℝ) (h: ∀ θ : ℝ, func b θ ≤ 7) : b = 1 ∨ b = -1 :=
by {
  sorry
}

end find_b_when_func_max_is_7_l827_827524


namespace infinite_series_k3_over_3k_l827_827449

theorem infinite_series_k3_over_3k :
  ∑' k : ℕ, (k^3 : ℝ) / 3^k = 165 / 16 := 
sorry

end infinite_series_k3_over_3k_l827_827449


namespace construct_quadrilateral_l827_827424

-- Definitions of the angles and side lengths
variables {α β γ δ : ℝ} {a b : ℝ}

-- The main theorem statement
theorem construct_quadrilateral (α β γ δ : ℝ) (a b : ℝ) :
  ∃ (A B C D : Type) [points_quadrilateral A B C D], 
    angle A B C = α ∧ angle B C D = β ∧ angle C D A = γ ∧ angle D A B = δ ∧ 
    dist A B = a ∧ dist C D = b := 
  sorry

end construct_quadrilateral_l827_827424


namespace roots_imply_value_l827_827097

noncomputable def value_of_expression (a b c : ℝ) : ℝ :=
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)

theorem roots_imply_value {a b c : ℝ} 
  (h1 : a + b + c = 15) 
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 10) 
  : value_of_expression a b c = 175 / 11 :=
sorry

end roots_imply_value_l827_827097


namespace geometric_sequence_problem_l827_827919

theorem geometric_sequence_problem
  (q : ℝ) (h_q : |q| ≠ 1) (m : ℕ)
  (a : ℕ → ℝ)
  (h_a1 : a 1 = -1)
  (h_am : a m = a 1 * a 2 * a 3 * a 4 * a 5) 
  (h_gseq : ∀ n, a (n + 1) = a n * q) :
  m = 11 :=
by
  sorry

end geometric_sequence_problem_l827_827919


namespace f_comp_f_neg1_l827_827111

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x + 2 else 1

theorem f_comp_f_neg1 : f (f (-1)) = 3 :=
by
  unfold f
  split_ifs
  -- Proving steps here
  sorry

end f_comp_f_neg1_l827_827111


namespace train_speed_approx_l827_827799

-- Define the condition statements
def train_length : ℝ := 90
def bridge_length : ℝ := 200
def time_to_cross : ℝ := 36

-- Define the speed calculation
def speed_of_train : ℝ := (train_length + bridge_length) / time_to_cross

-- State the theorem to prove the speed of the train is approximately 8.0556 m/s
theorem train_speed_approx : |speed_of_train - 8.0556| < 1e-4 :=
by
  -- Skip the proof
  sorry

end train_speed_approx_l827_827799


namespace sum_of_primes_between_10_and_20_l827_827252

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827252


namespace verify_compound_interest_rate_l827_827814

noncomputable def compound_interest_rate
  (P A : ℝ) (t n : ℕ) : ℝ :=
  let r := (A / P) ^ (1 / (n * t)) - 1
  n * r

theorem verify_compound_interest_rate :
  let P := 5000
  let A := 6800
  let t := 4
  let n := 1
  compound_interest_rate P A t n = 8.02 / 100 :=
by
  sorry

end verify_compound_interest_rate_l827_827814


namespace problem1_problem2_l827_827929

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * x^2

def m (x : ℝ) (a : ℝ) : ℝ := (deriv (λ x : ℝ, f x a)) x

def g (x : ℝ) (a : ℝ) : ℝ := f x a - a * x^2 + a * x

theorem problem1 (a : ℝ) (h : (deriv (λ x : ℝ, m x a)) 1 = 3) : a = 2 := sorry

theorem problem2 (a : ℝ) (h : ∀ x > 0, deriv (λ x : ℝ, g x a) x ≥ 0) : a ≥ 0 := sorry

end problem1_problem2_l827_827929


namespace probability_relatively_prime_l827_827190

open Nat

def relatively_prime_to (n m : ℕ) := gcd n m = 1

theorem probability_relatively_prime (N : ℕ) (hN : N = 42) : 
  (∃ (count : ℕ), count = (Nat.totient 42)) → (rat.ofInt count / (rat.ofInt 42) = 2 / 7) := 
by
  intro h_totient
  sorry

end probability_relatively_prime_l827_827190


namespace tan_alpha_plus_beta_alpha_plus_2beta_l827_827573

noncomputable def x_coord_A := - (Real.sqrt 2) / 10
noncomputable def x_coord_B := - (2 * Real.sqrt 5) / 5
noncomputable def y_coord_A := Real.sqrt (1 - x_coord_A ^ 2)
noncomputable def y_coord_B := Real.sqrt (1 - x_coord_B ^ 2)

variable (α β : ℝ)
-- These initial angles should be such that their tangent values match given coordinates

axiom α_obtuse : π / 2 < α ∧ α < π
axiom β_obtuse : π / 2 < β ∧ β < π

axiom tan_α : Real.tan α = y_coord_A / x_coord_A
axiom tan_β : Real.tan β = y_coord_B / x_coord_B

theorem tan_alpha_plus_beta : Real.tan (α + β) = -5 / 3 := by
  sorry

theorem alpha_plus_2beta : α + 2 * β = 9 * π / 4 := by
  sorry

end tan_alpha_plus_beta_alpha_plus_2beta_l827_827573


namespace range_of_k_for_domain_real_l827_827515

theorem range_of_k_for_domain_real (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 6 * k * x + (k + 8) ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end range_of_k_for_domain_real_l827_827515


namespace graphs_intersect_exactly_one_point_l827_827467

theorem graphs_intersect_exactly_one_point (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 5 * x + 4 = 2 * x - 6 → x = (7 / (2 * k))) ↔ k = (49 / 40) := 
by
  sorry

end graphs_intersect_exactly_one_point_l827_827467


namespace area_of_quadrilateral_ABEC_l827_827701

-- Definitions of points and lengths
variables {A B E C : Type*}
variables (AB BE BC : ℝ)
variables (hAB : AB = 15) (hBE : BE = 20) (hBC : BC = 25)

-- Right triangles conditions (assumed precalculated using Pythagorean theorem)
def isRightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Proof statement
theorem area_of_quadrilateral_ABEC
  (hABE : isRightTriangle 15 20 25)
  (hBEC : isRightTriangle 20 15 25) :
  let area_ABE : ℝ := 0.5 * 15 * 20 in
  let area_BEC : ℝ := 0.5 * 20 * 15 in
  area_ABE + area_BEC = 300 :=
by sorry

end area_of_quadrilateral_ABEC_l827_827701


namespace find_second_dimension_l827_827371

noncomputable def rectangular_tank_second_dimension (cost: ℝ) (cost_per_sqft: ℝ) (length: ℕ) (height: ℕ) : ℝ :=
  let total_surface_area := cost / cost_per_sqft
  let second_dimension := (total_surface_area - 2 * length * height - 2 * length * 2 - 2 * height * 2) / (2 * length + 2 * 2)
  second_dimension

theorem find_second_dimension :
  let length := 3
  let height := 2
  let cost := 1440
  let cost_per_sqft := 20 in
  rectangular_tank_second_dimension cost cost_per_sqft length height = 6 :=
by
  let length := 3 in
  let height := 2 in
  let cost := 1440 in
  let cost_per_sqft := 20 in
  sorry

end find_second_dimension_l827_827371


namespace car_dealer_sales_l827_827340

variable (x a b : ℕ)

theorem car_dealer_sales
  (h1 : (7 * x) / x = 7)
  (h2 : (7 * x - a) / (x - 1) = 8)
  (h3 : (7 * x - b) / (x - 1) = 5)
  (h4 : (7 * x - a - b) / (x - 2) = 5.75) :
  7 * x = 42 := by
  sorry

end car_dealer_sales_l827_827340


namespace uneaten_chips_correct_l827_827429

def cookies_per_dozen : Nat := 12
def dozens : Nat := 4
def chips_per_cookie : Nat := 7

def total_cookies : Nat := dozens * cookies_per_dozen
def total_chips : Nat := total_cookies * chips_per_cookie
def eaten_cookies : Nat := total_cookies / 2
def uneaten_cookies : Nat := total_cookies - eaten_cookies

def uneaten_chips : Nat := uneaten_cookies * chips_per_cookie

theorem uneaten_chips_correct : uneaten_chips = 168 :=
by
  -- Placeholder for the proof
  sorry

end uneaten_chips_correct_l827_827429


namespace expression_positive_l827_827037

theorem expression_positive (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : 5 * a ^ 2 - 6 * a * b + 5 * b ^ 2 > 0 :=
by
  sorry

end expression_positive_l827_827037


namespace find_x_l827_827063

-- Define points and lines
variables (A B C D E F : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]
variables (line_BAD : Line A D) (line_BCE : Line B E) (line_ACF : Line A F) (line_DEF : Line D E)

-- Define lengths and equalities
variables (BA BC AD AF EB ED : ℝ)
hypothesis h1 : BA = BC
hypothesis h2 : AD = AF
hypothesis h3 : EB = ED

-- Define angle BED
variables (x : ℝ) -- x = ∠BED

-- Statement to be proven
theorem find_x (BAD BCE ACF DEF straight_lines) : x = 108 :=
by
  sorry

end find_x_l827_827063


namespace angle_BDF_eq_angle_CDE_l827_827488

variable {α : Type} [EuclideanGeometry α]
variables {A B C P D E F : α}

-- Given conditions
axiom circumcircle_of_triangle : P ∈ circumcircle (triangle A B C)
axiom projection_D : projection P (line B C) = D
axiom projection_E : projection P (line C A) = E
axiom projection_F : projection P (line A B) = F
axiom C_between_AE : between C A E

-- Theorem statement
theorem angle_BDF_eq_angle_CDE :
  ∠B D F = ∠C D E := by
  sorry

end angle_BDF_eq_angle_CDE_l827_827488


namespace probability_relatively_prime_l827_827192

open Nat

def relatively_prime_to (n m : ℕ) := gcd n m = 1

theorem probability_relatively_prime (N : ℕ) (hN : N = 42) : 
  (∃ (count : ℕ), count = (Nat.totient 42)) → (rat.ofInt count / (rat.ofInt 42) = 2 / 7) := 
by
  intro h_totient
  sorry

end probability_relatively_prime_l827_827192


namespace sum_primes_between_10_and_20_l827_827239

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827239


namespace nice_set_l827_827084

def nice (P : Set (ℤ × ℤ)) : Prop :=
  ∀ (a b c d : ℤ), (a, b) ∈ P ∧ (c, d) ∈ P → (b, a) ∈ P ∧ (a + c, b - d) ∈ P

def is_solution (p q : ℤ) : Prop :=
  Int.gcd p q = 1 ∧ p % 2 ≠ q % 2

theorem nice_set (p q : ℤ) (P : Set (ℤ × ℤ)) :
  nice P → (p, q) ∈ P → is_solution p q → P = Set.univ := 
  sorry

end nice_set_l827_827084


namespace find_point_P_l827_827003

-- Define the function
def f (x : ℝ) := x^4 - 2 * x

-- Define the derivative of the function
def f' (x : ℝ) := 4 * x^3 - 2

theorem find_point_P :
  ∃ (P : ℝ × ℝ), (f' P.1 = 2) ∧ (f P.1 = P.2) ∧ (P = (1, -1)) :=
by
  -- here would go the actual proof
  sorry

end find_point_P_l827_827003


namespace smallest_possible_value_other_integer_l827_827671

theorem smallest_possible_value_other_integer (x : ℕ) (n : ℕ) (h_pos : x > 0)
  (h_gcd : ∃ m, Nat.gcd m n = x + 3 ∧ m = 30) 
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) :
  n = 162 := 
by sorry

end smallest_possible_value_other_integer_l827_827671


namespace area_enclosed_by_three_circles_l827_827180

theorem area_enclosed_by_three_circles (R : ℝ) :
  let r := R * (Real.sqrt 3 - 2) in
  let S_tri := r^2 * Real.sqrt 3 in
  let S_sec := 3 * (1/6 * π * r^2) in
  S_tri - S_sec = 1/2 * R^2 * (7 - 4 * Real.sqrt 3) * (6 * Real.sqrt 3 - 3 * π) :=
by
  sorry

end area_enclosed_by_three_circles_l827_827180


namespace find_w_l827_827947

variables {x y z w : ℝ}

theorem find_w (h : (1 / x) + (1 / y) + (1 / z) = 1 / w) :
  w = (x * y * z) / (y * z + x * z + x * y) := by
  sorry

end find_w_l827_827947


namespace sum_of_primes_between_10_and_20_l827_827301

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827301


namespace exist_subset_S_l827_827605

def is_lattice_point (P : ℤ × ℤ) : Prop := true

def adjacent (P Q : ℤ × ℤ) : Prop :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2) = 1

def S : set (ℤ × ℤ) := {P | ∃ k : ℤ, P.1 + 2 * P.2 = 5 * k }

theorem exist_subset_S :
  ∃ S : set (ℤ × ℤ), 
  (∀ P : ℤ × ℤ, is_lattice_point P → (S P ∧ ∀ Q : ℤ × ℤ, adjacent P Q → ¬ S Q) ∨
                                     (¬ S P ∧ ∃ Q : ℤ × ℤ, adjacent P Q ∧ S Q)) :=
sorry

end exist_subset_S_l827_827605


namespace smallest_side_for_table_rotation_l827_827811

theorem smallest_side_for_table_rotation (S : ℕ) : (S ≥ Int.ofNat (Nat.sqrt (8^2 + 12^2) + 1)) → S = 15 := 
by
  sorry

end smallest_side_for_table_rotation_l827_827811


namespace least_prime_factor_of_expression_l827_827212

theorem least_prime_factor_of_expression : 
  ∀ (p : ℕ), p.prime → (p ∣ (11 ^ 5 - 11 ^ 4)) → (p = 2) :=
sorry

end least_prime_factor_of_expression_l827_827212


namespace total_strength_college_l827_827733

-- Defining the conditions
def C : ℕ := 500
def B : ℕ := 600
def Both : ℕ := 220

-- Declaring the theorem
theorem total_strength_college : (C + B - Both) = 880 :=
by
  -- The proof is not required, put sorry
  sorry

end total_strength_college_l827_827733


namespace tax_free_value_is_500_l827_827058

-- Definitions of the given conditions
def total_value : ℝ := 730
def paid_tax : ℝ := 18.40
def tax_rate : ℝ := 0.08

-- Definition of the excess value
def excess_value (E : ℝ) := tax_rate * E = paid_tax

-- Definition of the tax-free threshold value
def tax_free_limit (V : ℝ) := total_value - (paid_tax / tax_rate) = V

-- The theorem to be proven
theorem tax_free_value_is_500 : 
  ∃ V : ℝ, (total_value - (paid_tax / tax_rate) = V) ∧ V = 500 :=
  by
    sorry -- Proof to be completed

end tax_free_value_is_500_l827_827058


namespace cubic_roots_solve_l827_827095

-- Let a, b, c be roots of the equation x^3 - 15x^2 + 25x - 10 = 0
variables {a b c : ℝ}
def eq1 := a + b + c = 15
def eq2 := a * b + b * c + c * a = 25
def eq3 := a * b * c = 10

theorem cubic_roots_solve :
  eq1 → eq2 → eq3 → 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
by
  intros,
  sorry

end cubic_roots_solve_l827_827095


namespace initial_lives_l827_827073

theorem initial_lives (x : ℕ) (h1 : x - 23 + 46 = 70) : x = 47 := 
by 
  sorry

end initial_lives_l827_827073


namespace am_gm_example_l827_827481

open Real

theorem am_gm_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1)^3 / b + (b + 1)^3 / c + (c + 1)^3 / a ≥ 81 / 4 := 
by 
  sorry

end am_gm_example_l827_827481


namespace shift_graph_left_by_3_l827_827525

theorem shift_graph_left_by_3 (f : ℝ → ℝ) : (∀ x, f(x) = 2*x - 3) → (∀ x, f(2*x + 3) = 2*(x - 3)) :=
by
  assume h : ∀ x, f(x) = 2*x - 3
  sorry

end shift_graph_left_by_3_l827_827525


namespace sum_of_primes_between_10_and_20_l827_827256

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827256


namespace sum_distances_to_faces_of_tetrahedron_constant_l827_827560

theorem sum_distances_to_faces_of_tetrahedron_constant
    (T : Tetrahedron)
    (P : Point)
    (h₁ : IsRegularTetrahedron T)
    (h₂ : PointInsideTetrahedron P T) :
  (∑ f in T.faces, distance_from_point_to_face P f) = constant_value :=
by sorry

end sum_distances_to_faces_of_tetrahedron_constant_l827_827560


namespace count_multiples_of_12_l827_827030

theorem count_multiples_of_12 (a b : ℤ) (h1 : a = 5) (h2 : b = 145) :
  ∃ n : ℕ, (12 * n + 12 ≤ b) ∧ (12 * n + 12 > a) ∧ n = 12 :=
by
  sorry

end count_multiples_of_12_l827_827030


namespace product_of_x_and_z_l827_827061

theorem product_of_x_and_z
  (EF GH HE FG : ℝ)
  (x z : ℝ)
  (hEF : EF = 52)
  (hGH : GH = 5 * x + 6)
  (hHE : HE = 16)
  (hFG : FG = 4 * z^2 + 4)
  (parallelogram : EF = GH ∧ FG = HE) :
  x * z = (46 * Real.sqrt 3) / 5 :=
by
  -- given conditions
  have h1 : 52 = 5 * x + 6 := parallelogram.1
  have h2 : 4 * z^2 + 4 = 16 := parallelogram.2

  -- solve for x and z
  have x_val : x = 46 / 5 := sorry
  have z_val : z = Real.sqrt 3 := sorry
  
  -- calculate x * z
  have prod : x * z = (46 / 5) * Real.sqrt 3 := sorry
  
  -- simplify the product
  show x * z = (46 * Real.sqrt 3) / 5, from sorry

end product_of_x_and_z_l827_827061


namespace optimal_lineup_probability_l827_827393

theorem optimal_lineup_probability :
  let p := 1 - (2 ^ 15 / 3 ^ 15 : ℝ) in
  p = 1 - (2 ^ 15 / 3 ^ 15 : ℝ) := 
by
  sorry

end optimal_lineup_probability_l827_827393


namespace sum_of_primes_between_10_and_20_l827_827303

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827303


namespace polyhedron_edge_coloring_l827_827808

theorem polyhedron_edge_coloring :
  ∀ (P : Type) [Polyhedron P]
    (h_faces : ∀ f, Face f P → Triangle f),
  ∃ (red_edges blue_edges : set (Edge P)),
    (∀ e, Edge e P → e ∈ red_edges ∨ e ∈ blue_edges) ∧
    (∀ (v₁ v₂ : Vertex P),
      (Connected P red_edges v₁ v₂) ∧ 
      (Connected P blue_edges v₁ v₂)) :=
sorry

end polyhedron_edge_coloring_l827_827808


namespace limit_n_b_n_l827_827435

noncomputable def M (x : ℝ) : ℝ := x - x^3 / 3

noncomputable def b_n (n : ℕ) : ℝ := (nat.iterate M n (20 / n.toReal))

theorem limit_n_b_n (L : ℝ) : 
  (∀ n : ℕ, n > 0 → b_n n = nat.iterate M n (20 / n.toReal)) →
  tendsto (λ n : ℕ, (n.toReal) * (b_n n)) atTop (𝓝 (60 / 17)) :=
by
  sorry

end limit_n_b_n_l827_827435


namespace inscribed_circle_implies_rhombus_l827_827641

theorem inscribed_circle_implies_rhombus (AB : ℝ) (AD : ℝ)
  (h_parallelogram : AB = CD ∧ AD = BC) 
  (h_inscribed : AB + CD = AD + BC) : 
  AB = AD := by
  sorry

end inscribed_circle_implies_rhombus_l827_827641


namespace arc_length_l827_827578

theorem arc_length (O Q I S : Point) (angle_QIS : ℝ) (OQ : ℝ) :
  angle_QIS = 45 ∧ OQ = 15 → arc_length O Q S = 7.5 * real.pi :=
by
  sorry

end arc_length_l827_827578


namespace find_z_l827_827586

theorem find_z (x y z : ℝ) (h1 : y = 3 * x - 5) (h2 : z = 3 * x + 3) (h3 : y = 1) : z = 9 := 
by
  sorry

end find_z_l827_827586


namespace solution_l827_827104

noncomputable def problem (a b c : ℝ) : Prop :=
  (Polynomial.eval a (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval b (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval c (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)

theorem solution (a b c : ℝ) (h : problem a b c) : 
  (∃ abc : ℝ, abc = a * b * c ∧ abc = 10) →
  (a + b + c = 15) ∧ (a * b + b * c + c * a = 25) →
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
sorry

end solution_l827_827104


namespace binomial_expansion_problem_l827_827492

theorem binomial_expansion_problem 
  (a0 a1 : ℝ) 
  (a : ℕ → ℝ) 
  (h₁ : (∀ x : ℝ, (2 * x - 1) ^ 2015 = a0 + ∑ i in finset.range 2016, a i * x ^ i))
  (h₂ : a0 = -1)
  (h₃ : 1 / 2 * a1 + ∑ i in finset.range 2015, (1 / 2^(i + 2)) * a (i + 2) = 1) : 
  1 / 2 + ∑ i in finset.range 2014, (a (i + 2) / 2^(i + 2) / a1) = 1 / 4030 := 
  sorry

end binomial_expansion_problem_l827_827492


namespace max_problems_to_miss_to_pass_l827_827813

theorem max_problems_to_miss_to_pass (total_problems : ℕ) (pass_percentage : ℝ) :
  total_problems = 50 → pass_percentage = 0.85 → 7 = ↑total_problems * (1 - pass_percentage) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end max_problems_to_miss_to_pass_l827_827813


namespace area_triangle_bounds_l827_827319

variables {a b c r S : ℝ} (p : ℝ) (cond1 : S = sqrt (p * (p - a) * (p - b) * (p - c)))
  (cond2 : p = (a + b + c) / 2) (cond3 : S >= 3 * sqrt 3 * r^2)
  (cond4 : S = p * r) (cond5 : (a + b + c)^2 <= 3 * (a^2 + b^2 + c^2))

theorem area_triangle_bounds (p : ℝ) (S : ℝ) (r : ℝ) (a b c : ℝ)
    (cond1 : S = sqrt (p * (p - a) * (p - b) * (p - c)))
    (cond2 : p = (a + b + c) / 2)
    (cond3 : S >= 3 * √3 * r^2)
    (cond4 : S = p * r)
    (cond5 : (a + b + c)^2 ≤ 3 * (a^2 + b^2 + c^2)) :
  3 * √3 * r^2 ≤ S ∧ S ≤ (p^2) / (3 * √3) ∧ S ≤ (a^2 + b^2 + c^2) / (4 * √3) :=
by sorry

end area_triangle_bounds_l827_827319


namespace probability_of_relatively_prime_to_42_l827_827189

def relatively_prime_to_42_count : ℕ :=
  let N := 42
  (finset.range (N + 1)).filter (λ n, Nat.gcd n N = 1).card

theorem probability_of_relatively_prime_to_42 : 
  (relatively_prime_to_42_count : ℚ) / 42 = 2 / 7 :=
by
  sorry

end probability_of_relatively_prime_to_42_l827_827189


namespace sum_primes_between_10_and_20_l827_827279

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827279


namespace equal_roots_if_discriminant_is_zero_l827_827969

theorem equal_roots_if_discriminant_is_zero (k : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ discriminant 1 (-2) k = 0) → k = 1 := 
sorry

end equal_roots_if_discriminant_is_zero_l827_827969


namespace segment_EC_equals_radius_l827_827202

-- Define the entities: points and circle
variables {Point : Type} [MetricSpace Point]
variables (A B C D E : Point) (S : set Point) (R : ℝ)

-- Define the conditions in an equilateral triangle and geometric problem
def is_equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def on_circle (S : set Point) (P : Point) : Prop :=
  P ∈ S

def radius_of_circle (S : set Point) (O X : Point) : ℝ :=
  dist O X

def geometric_conditions (A B C D E : Point) (S : set Point) (R : ℝ) : Prop :=
  is_equilateral_triangle A B C ∧
  on_circle S A ∧
  on_circle S B ∧
  ¬ on_circle S C ∧
  on_circle S D ∧
  on_circle S E ∧
  dist B D = dist A B ∧
  ∃ P : Point, P ∈ S ∧ on_circle (set_of (λ P, dist C P = R)) E ∧ (set_of (λ P, dist C P = R) ∩ line_through C D) = {E}

-- Define the main statement
theorem segment_EC_equals_radius
  {Point : Type} [MetricSpace Point]
  (A B C D E : Point) (S : set Point) (R : ℝ) :
  (geometric_conditions A B C D E S R) → (dist E C = R) :=
sorry

end segment_EC_equals_radius_l827_827202


namespace sum_primes_between_10_and_20_l827_827287

theorem sum_primes_between_10_and_20 : ∑ (p : ℕ) in {11, 13, 17, 19}, p = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l827_827287


namespace sqrt_divisors_l827_827732

theorem sqrt_divisors (n : ℕ) (h1 : n = p ^ 4) (hp : Prime p) : Nat.divisors (Nat.sqrt n) = {1, p, p^2} := by
  sorry

end sqrt_divisors_l827_827732


namespace sum_primes_10_to_20_l827_827297

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827297


namespace count_integer_values_l827_827454

theorem count_integer_values : 
  { x : ℤ // 289 < x ∧ x ≤ 324 }.card = 35 := 
sorry

end count_integer_values_l827_827454


namespace sum_primes_between_10_and_20_is_60_l827_827229

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827229


namespace reciprocal_inequality_pos_reciprocal_inequality_neg_l827_827036

theorem reciprocal_inequality_pos {a b : ℝ} (h : a < b) (ha : 0 < a) : (1 / a) > (1 / b) :=
sorry

theorem reciprocal_inequality_neg {a b : ℝ} (h : a < b) (hb : b < 0) : (1 / a) < (1 / b) :=
sorry

end reciprocal_inequality_pos_reciprocal_inequality_neg_l827_827036


namespace max_area_of_cone_l827_827918

noncomputable def max_cross_sectional_area (l θ : ℝ) : ℝ := (1/2) * l^2 * Real.sin θ

theorem max_area_of_cone :
  (∀ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) → max_cross_sectional_area 3 θ ≤ (9 / 2))
  ∧ (∃ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) ∧ max_cross_sectional_area 3 θ = (9 / 2)) := 
by
  sorry

end max_area_of_cone_l827_827918


namespace radius_of_cone_base_l827_827794

-- Define the sector radius and the central angle
def sector_radius : Real := 15
def central_angle : Real := 120

-- Define the formula for the length of the arc
def arc_length (theta : Real) (R : Real) : Real :=
  (theta / 360) * 2 * Real.pi * R

-- Given the sector's parameters
def given_arc_length : Real := arc_length central_angle sector_radius

-- Define the formula for the circumference of the base of the cone
def cone_base_circumference (r : Real) : Real := 2 * Real.pi * r

-- Prove that the radius of the base of the cone is 5 cm
theorem radius_of_cone_base : ∃ (r : Real), cone_base_circumference(r) = given_arc_length ∧ r = 5 :=
by 
  use 5
  sorry

end radius_of_cone_base_l827_827794


namespace integer_solutions_count_l827_827863

theorem integer_solutions_count :
  (finset.card (finset.filter (λ x, 
    real.sqrt (1 - real.sin (real.pi * x / 4) - 3 * real.cos (real.pi * x / 2)) - 
    real.sqrt 6 * real.sin (real.pi * x / 4) ≥ 0)
    (finset.range (2014 - 1991)).map (λ n, n + 1991))) = 8 :=
sorry

end integer_solutions_count_l827_827863


namespace largest_circle_area_from_rectangle_string_l827_827370

theorem largest_circle_area_from_rectangle_string
  (length : ℝ) (width : ℝ) (h_length : length = 16) (h_width : width = 8) :
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  let area := Real.pi * (radius ^ 2)
  Real.floor (area + 0.5) = 183 :=
by {
  sorry
}

end largest_circle_area_from_rectangle_string_l827_827370


namespace asymptote_of_hyperbola_l827_827513

open Real

noncomputable def common_focus_equation (m n : ℝ) : Prop :=
  3 * m^2 - 5 * n^2 = 2 * m^2 + 3 * n^2

noncomputable def asymptote_equation {m n x y : ℝ} (h : common_focus_equation m n) : Prop :=
  y = (sqrt 3 / 4) * x ∨ y = -(sqrt 3 / 4) * x

theorem asymptote_of_hyperbola (m n : ℝ) (h_focus : common_focus_equation m n) :
  asymptote_equation h_focus :=
sorry

end asymptote_of_hyperbola_l827_827513


namespace turtles_remaining_on_log_l827_827769

-- Definition of the problem parameters
def initial_turtles : ℕ := 9
def additional_turtles : ℕ := (3 * initial_turtles) - 2
def total_turtles : ℕ := initial_turtles + additional_turtles
def frightened_turtles : ℕ := total_turtles / 2
def remaining_turtles : ℕ := total_turtles - frightened_turtles

-- The final theorem stating the number of turtles remaining
theorem turtles_remaining_on_log : remaining_turtles = 17 := by
  -- Proof is omitted
  sorry

end turtles_remaining_on_log_l827_827769


namespace count_even_three_digit_numbers_l827_827536

theorem count_even_three_digit_numbers : 
  let digits := {1, 2, 3, 4, 5}
  in ∑ a in digits, ∑ b in digits \ {a}, ∑ c in {2, 4} if b != c,
     (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) = 24 := 
sorry

end count_even_three_digit_numbers_l827_827536


namespace sum_primes_10_20_l827_827224

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827224


namespace isosceles_triangles_with_60_degree_internal_angle_are_similar_l827_827383

-- Define the condition: Both triangles are isosceles and have an internal angle of 60 degrees.
def is_isosceles_and_has_60_degree_internal_angle (T : Triangle) : Prop :=
  is_isosceles T ∧ (∃ (angle : ℝ), internal_angle T angle ∧ angle = 60)

-- The statement to prove: Two isosceles triangles with a 60 degree internal angle are similar.
theorem isosceles_triangles_with_60_degree_internal_angle_are_similar
  (T1 T2 : Triangle)
  (h1 : is_isosceles_and_has_60_degree_internal_angle T1)
  (h2 : is_isosceles_and_has_60_degree_internal_angle T2) :
  similar T1 T2 :=
by
  sorry

end isosceles_triangles_with_60_degree_internal_angle_are_similar_l827_827383


namespace top_and_bottom_edges_same_color_l827_827420

-- Define the vertices for top and bottom pentagonal faces
inductive Vertex
| A1 | A2 | A3 | A4 | A5
| B1 | B2 | B3 | B4 | B5

-- Define the edges
inductive Edge : Type
| TopEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) : Edge
| BottomEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge
| SideEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge

-- Define colors
inductive Color
| Red | Blue

-- Define a function that assigns a color to each edge
def edgeColor : Edge → Color := sorry

-- Define a function that checks if a triangle is monochromatic
def isMonochromatic (e1 e2 e3 : Edge) : Prop :=
  edgeColor e1 = edgeColor e2 ∧ edgeColor e2 = edgeColor e3

-- Define our main theorem statement
theorem top_and_bottom_edges_same_color (h : ∀ v1 v2 v3 : Vertex, ¬ isMonochromatic (Edge.TopEdge v1 v2 sorry sorry) (Edge.SideEdge v1 v3 sorry sorry) (Edge.BottomEdge v2 v3 sorry sorry)) : 
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → edgeColor (Edge.TopEdge v1 v2 sorry sorry) = edgeColor (Edge.TopEdge Vertex.A1 Vertex.A2 sorry sorry)) ∧
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → edgeColor (Edge.BottomEdge v1 v2 sorry sorry) = edgeColor (Edge.BottomEdge Vertex.B1 Vertex.B2 sorry sorry)) :=
sorry

end top_and_bottom_edges_same_color_l827_827420


namespace billy_music_book_songs_l827_827815

theorem billy_music_book_songs (can_play : ℕ) (needs_to_learn : ℕ) (total_songs : ℕ) 
  (h1 : can_play = 24) (h2 : needs_to_learn = 28) : 
  total_songs = can_play + needs_to_learn ↔ total_songs = 52 :=
by
  sorry

end billy_music_book_songs_l827_827815


namespace parallel_lines_l827_827041

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + a + 3 = 0) ∧ (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) 
  → a = -2 :=
sorry

end parallel_lines_l827_827041


namespace arithmetic_sequence_sum_l827_827413

theorem arithmetic_sequence_sum :
  let a1 := 7
  let d := 4
  let an := 147
  let n := (an - a1) / d + 1
  S := (n * (a1 + an)) / 2
  S = 2772 :=
by
  sorry

end arithmetic_sequence_sum_l827_827413


namespace tony_water_per_day_l827_827182

theorem tony_water_per_day (bottle_size : ℕ) (fill_times_per_week : ℕ) (days_per_week : ℕ) 
    (h1 : bottle_size = 84) (h2 : fill_times_per_week = 6) (h3 : days_per_week = 7) : 
    (bottle_size * fill_times_per_week) / days_per_week = 72 :=
by
  rw [h1, h2, h3]
  norm_num

end tony_water_per_day_l827_827182


namespace inequality_proof_l827_827086

noncomputable def a : ℝ := real.exp (real.log 2 / 3)
noncomputable def b : ℝ := real.log 3 / real.log 4
noncomputable def c : ℝ := real.log 5 / real.log 8

theorem inequality_proof : a > b ∧ b > c := sorry

end inequality_proof_l827_827086


namespace concurrent_rest_days_14_times_l827_827381

-- Define Al's cycle
def Al_cycle : List String := ["W", "W", "W", "W", "W", "R", "R"]

-- Define Barb's cycle
def Barb_cycle : List String := ["W", "W", "W", "W", "R"]

-- Prove that they coincide rest days exactly 14 times within 500 days
theorem concurrent_rest_days_14_times :
  (∃ l: List String, l = Al_cycle ∧
    ∃ b: List String, b = Barb_cycle ∧
      let days := 500
      in let lcm := 35
      in (days / lcm) * 1 = 14) :=
sorry

end concurrent_rest_days_14_times_l827_827381


namespace find_m_and_line_equation_l827_827906

noncomputable def circle_C (x y : ℝ) : Prop := x^2 - 4*x + y^2 + 3 = 0

noncomputable def is_tangent (m : ℝ) : Prop :=
  let c := (2, 0)
  let d := abs (2*m + 1) / real.sqrt (1 + m^2)
  d = 1

noncomputable def is_tangent_m : set ℝ :=
  {m | is_tangent m}

noncomputable def is_tangent_to_circle_C (m : ℝ) : Prop :=
  m ∈ is_tangent_m ∧ m = 0 ∨ m = -4/3

noncomputable def line_l (k : ℝ) (x : ℝ) : ℝ := k * x - 2

noncomputable def intersection_points (k : ℝ) (sorry : ℕ) :=
  let k1 := 1
  let k2 := 5/3 
  line_l k = line_l k1 ∨ line_l k = line_l k2

theorem find_m_and_line_equation :
  (∀ m : ℝ, is_tangent_to_circle_C m) ∧
  (∀ k : ℝ, intersection_points k  
    (∀ x1 x2 : ℝ, ∃ k1 k2, line_l k _ 
      ∧ (x1 + x2 = 4 * (k + 1) / (1 + k^2)
      ∧ x1 * x2 = 7 / (1 + k^2)
      ∧ k1 * k2 = -1 / 7))
    ∧ k ∈ {1, 5/3}) :=
sorry

end find_m_and_line_equation_l827_827906


namespace number_of_camels_l827_827983

theorem number_of_camels (hens goats keepers camel_feet heads total_feet : ℕ)
  (h_hens : hens = 50) (h_goats : goats = 45) (h_keepers : keepers = 15)
  (h_feet_diff : total_feet = heads + 224)
  (h_heads : heads = hens + goats + keepers)
  (h_hens_feet : hens * 2 = 100)
  (h_goats_feet : goats * 4 = 180)
  (h_keepers_feet : keepers * 2 = 30)
  (h_camels_feet : camel_feet = 24)
  (h_total_feet : total_feet = 334)
  (h_feet_without_camels : 100 + 180 + 30 = 310) :
  camel_feet / 4 = 6 := sorry

end number_of_camels_l827_827983


namespace good_price_before_discount_l827_827805

noncomputable def original_price (P : ℝ) : ℝ := P

theorem good_price_before_discount (P : ℝ) (h : 0.684 * P = 6400) :
  P ≈ 9356.725 :=
begin
  sorry
end

end good_price_before_discount_l827_827805


namespace fraction_spent_on_sweater_l827_827081

theorem fraction_spent_on_sweater {T : ℝ} (h1 : 40 + 100 + 20 = T) (h2 : 40 + 60 = 100) : 
  40 / T = 1 / 4 :=
by {
  have T_eq : T = 40 + 100 + 20, from h1,
  have jewelry_eq : 100 = 40 + 60, from h2,
  rw [T_eq, jewelry_eq] at *,
  sorry
}

end fraction_spent_on_sweater_l827_827081


namespace max_ab_l827_827482

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 8) : 
  ab ≤ 8 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 8 ∧ ab = 8 :=
by
  sorry

end max_ab_l827_827482


namespace mono_decreasing_y_sin_2x_l827_827167

open Real

theorem mono_decreasing_y_sin_2x 
  (k : ℤ) : 
  (∀ x, (2 * k * π + π / 2 ≤ x) ∧ (x ≤ 2 * k * π + 3 * π / 2) -> mon_decreasing (sin x)) →
  (∀ x, (π * k + π / 4 ≤ x) ∧ (x ≤ π * k + 3 * π / 4) -> mon_decreasing (sin (2 * x))) :=
by
  sorry

end mono_decreasing_y_sin_2x_l827_827167


namespace find_curve_and_length_segment_l827_827916

noncomputable def point := ℝ × ℝ

def parabola (F : point) (d : ℝ) (M : point) : Prop :=
  (M.1 - F.1) ^ 2 + M.2 ^ 2 = ((M.1 + d) ^ 2)

def distance (A B : point) : ℝ :=
  euclideanDist A B

theorem find_curve_and_length_segment :
  (∀ (M : point), distance M (1, 0) + 1 = distance M (-2, M.2)) →
  (∀ A B : point, A ∈ {M : point | parabola (1, 0) (-1) M} ∧ B ∈ {M : point | parabola (1, 0) (-1) M} ∧ A.2 = A.1 - 1 ∧ B.2 = B.1 - 1 →
  distance A B = 8) ∧
  {M : point | parabola (1, 0) (-1) M} = {M : point | M.2 ^ 2 = 4 * M.1} :=
begin
  sorry
end

end find_curve_and_length_segment_l827_827916


namespace man_speed_against_current_proof_l827_827778

def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

theorem man_speed_against_current_proof 
  (V_m : ℝ) 
  (h_with_current : V_m + speed_of_current = man_speed_with_current) :
  V_m - speed_of_current = man_speed_against_current := 
by 
  sorry

end man_speed_against_current_proof_l827_827778


namespace sum_of_primes_between_10_and_20_l827_827299

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827299


namespace line_through_points_has_sum_m_b_3_l827_827661

-- Define the structure that two points are given
structure LineThroughPoints (P1 P2 : ℝ × ℝ) : Prop :=
  (slope_intercept_form : ∃ m b, (P1.snd = m * P1.fst + b) ∧ (P2.snd = b)) 

-- Define the particular points
def point1 : ℝ × ℝ := (-2, 0)
def point2 : ℝ × ℝ := (0, 2)

-- The theorem statement
theorem line_through_points_has_sum_m_b_3 
  (h : LineThroughPoints point1 point2) : 
  ∃ m b, (point1.snd = m * point1.fst + b) ∧ (point2.snd = b) ∧ (m + b = 3) :=
by
  sorry

end line_through_points_has_sum_m_b_3_l827_827661


namespace turtles_remaining_on_log_l827_827762

-- Define the initial conditions
def original_turtles := 9
def additional_turtles := (3 * original_turtles) - 2
def total_group := original_turtles + additional_turtles
def frightened_turtles := total_group / 2

-- Theorem statement
theorem turtles_remaining_on_log : total_group - frightened_turtles = 17 :=
by
  sorry

end turtles_remaining_on_log_l827_827762


namespace pow_totient_lcm_eq_one_mod_l827_827742

open Nat

def λ (m : ℕ) : ℕ := 
  let prime_factors := m.factors.toFinset
  List.lcm (prime_factors.toList.map (fun p => eulerTotient (p ^ (m.factors.count p))))

theorem pow_totient_lcm_eq_one_mod (m a : ℕ) (h_coprime : gcd a m = 1) :
  a^ λ m ≡ 1 [MOD m] := by
sorry

end pow_totient_lcm_eq_one_mod_l827_827742


namespace ellis_more_cards_than_orion_l827_827680

theorem ellis_more_cards_than_orion : 
  ∀ (N : ℕ) (r1 r2 : ℕ), N = 500 → r1 = 11 → r2 = 9 → 
  (r1 * (N / (r1 + r2)) - r2 * (N / (r1 + r2)) = 50) := 
by
  intros N r1 r2 hN hr1 hr2
  rw [hN, hr1, hr2]
  sorry

end ellis_more_cards_than_orion_l827_827680


namespace derek_savings_l827_827841

theorem derek_savings : 
  let a : ℕ := 2 in
  let savings := λ (n : ℕ) => a * 2^n in
  savings 11 = 4096 := 
by
  -- Define initial savings and the savings function
  let a : ℕ := 2
  let savings := λ (n : ℕ) => a * 2^n
  -- Prove that Derek's savings in December (n = 11) is 4096
  have : savings 11 = 2 * 2^11 := by rfl
  have : 2 * 2^11 = 4096 := by norm_num
  show savings 11 = 4096 by
    rw [this]
    sorry

end derek_savings_l827_827841


namespace determine_session_duration_l827_827339

open Nat

noncomputable def sessionDuration : ℝ := 1.833

theorem determine_session_duration
  (start1 : ℝ := 12) (start2 : ℝ := 13) (start7 : ℝ := 23) (start8 : ℝ := 24)
  (sessions_equal : ∀ i j : Nat, 1 ≤ i ∧ i ≤ 8 → 1 ≤ j ∧ j ≤ 8 →  (i ≠ j → |start1 + (i - 1) * sessionDuration - (start1 + (j - 1) * sessionDuration)|)
  ) :
  ∀ i j : Nat, 1 ≤ i ∧ i ≤ 8 → 1 ≤ j ∧ j ≤ 8 → (i ≠ j → |start1 + (i - 1) * sessionDuration - (start1 + (j - 1) * sessionDuration)|) =
  ( j - i) * sessionDuration :=
sorry

end determine_session_duration_l827_827339


namespace petrol_price_increase_l827_827973

def price_increase (P P_new C C_new : ℝ) : ℝ := 
  if (P * C = P_new * C_new) ∧ (C_new = 0.8 * C) then 
    let x := (P_new / P - 1) * 100 in x
  else 
    0

theorem petrol_price_increase (P P_new C : ℝ) (h_expenditure : P_new * (0.8 * C) = P * C) :
  price_increase P P_new C (0.8 * C) = 25 :=
by
  sorry

end petrol_price_increase_l827_827973


namespace find_t_l827_827035

theorem find_t (t : ℝ) :
  (sqrt (3 * sqrt (t - 3)) = real.root 4 (10 - t)) →
  t = 37 / 10 :=
by
  sorry

end find_t_l827_827035


namespace ratio_of_e_and_d_l827_827683

theorem ratio_of_e_and_d :
  (∀ {d e : ℤ}, (∀ {x : ℝ}, x^2 + 900 * x + 1800 = (x + d)^2 + e) → (e = -200700) → (d = 450) → (e / d = -446)) :=
  by
  intros d e hd_eq he_val hd_val
  rw [hd_val, he_val]
  norm_num
  sorry

end ratio_of_e_and_d_l827_827683


namespace tan_alpha_of_cos_alpha_l827_827474

theorem tan_alpha_of_cos_alpha (α : ℝ) (hα : 0 < α ∧ α < Real.pi) (h_cos : Real.cos α = -3/5) :
  Real.tan α = -4/3 :=
sorry

end tan_alpha_of_cos_alpha_l827_827474


namespace slope_l3_is_5_over_6_l827_827127

noncomputable theory

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, -3)
def B : ℝ × ℝ := (2, 2)
def C (x : ℝ) : ℝ × ℝ := (x, 2)

-- Define the lines l₁, l₂, and the constraint for l₃ passing through A and C
def line_l1 (p : ℝ × ℝ) : Prop := 4 * p.1 - 3 * p.2 = 2
def line_l2 (p : ℝ × ℝ) : Prop := p.2 = 2

-- Define the area of triangle ABC
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Define the slope function of a line given two points
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- The proof problem statement
theorem slope_l3_is_5_over_6 : ∃ x : ℝ, 
  (line_l1 B) ∧ (line_l1 A) ∧ (line_l2 B) ∧ (line_l2 (C x)) ∧
  (area_of_triangle A B (C x) = 5) ∧ (slope A (C x) = 5 / 6) :=
sorry

end slope_l3_is_5_over_6_l827_827127


namespace sales_tax_percentage_l827_827074

theorem sales_tax_percentage (total_worth : ℝ) (tax_rate : ℝ) (tax_free_items_cost : ℝ) :
  total_worth = 45 ∧ tax_rate = 0.06 ∧ tax_free_items_cost = 39.7 →
  (tax_rate * (total_worth - tax_free_items_cost) / total_worth) * 100 ≈ 0.67 :=
by
  sorry

end sales_tax_percentage_l827_827074


namespace sum_primes_between_10_and_20_l827_827247

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827247


namespace find_values_of_x_y_l827_827786

theorem find_values_of_x_y : ∃ (x y : ℚ), 3 * (2 * x + 9 * y) = 75 ∧ x + y = 10 ∧ x = 65 / 7 ∧ y = 5 / 7 := 
by {
  use [65 / 7, 5 / 7],
  split,
  { calc 3 * (2 * (65 / 7) + 9 * (5 / 7))
       = 3 * (130 / 7 + 45 / 7) : by rw [mul_div_assoc, mul_div_assoc]
   ... = 3 * (175 / 7) : by rw [add_div]
   ... = 3 * 25 : by norm_num
   ... = 75 : by norm_num },
  split,
  { norm_num },
  split,
  { refl },
  { refl }
}

end find_values_of_x_y_l827_827786


namespace train_length_l827_827198

-- Define the condition that two trains have equal length
def equal_length (l1 l2 : ℝ) : Prop :=
  l1 = l2

-- Define the speeds in km/h
def speed1_kmh : ℝ := 90
def speed2_kmh : ℝ := 85

-- Convert speeds to m/s
def kmh_to_ms (v : ℝ) : ℝ :=
  v * (1000 / 3600)

-- Relative speed in m/s
def relative_speed_ms : ℝ :=
  kmh_to_ms speed1_kmh + kmh_to_ms speed2_kmh

-- Time in seconds
def time_seconds : ℝ := 8.64

-- Calculate the combined distance traveled when passing each other
def combined_distance : ℝ :=
  relative_speed_ms * time_seconds

-- Prove the length of each train given equal length and conditions
theorem train_length (l : ℝ) (h_equal : equal_length l l) : l = 209.96 :=
  by
  sorry

end train_length_l827_827198


namespace number_of_fence_poles_l827_827788

theorem number_of_fence_poles (l w distance_between_poles : ℕ) (h_l : l = 90) (h_w : w = 60) (h_d : distance_between_poles = 5) :
  (2 * (l + w)) / distance_between_poles = 60 :=
by
  rw [h_l, h_w, h_d]
  norm_num
  sorry

end number_of_fence_poles_l827_827788


namespace maintain_proportion_and_amounts_l827_827622

/-- Define the initial condition of the recipe and the resulting amounts -/
def initial_flour : ℝ := 7.0
def initial_sugar : ℝ := 3.0
def initial_vegetable_oil : ℝ := 2.0

def original_ratio : ℝ := initial_flour / initial_sugar ∧ initial_sugar / initial_vegetable_oil = 7 / 3 * 3 / 2

def extra_flour : ℝ := 2.0

def new_flour_amount : ℝ := initial_flour + extra_flour

noncomputable def ratio_factor : ℝ := new_flour_amount / initial_flour

noncomputable def new_sugar_amount : ℝ := initial_sugar * ratio_factor

noncomputable def new_vegetable_oil_amount : ℝ := initial_vegetable_oil * ratio_factor

theorem maintain_proportion_and_amounts :
  new_flour_amount = 9.0 ∧
  new_sugar_amount ≈ 3.86 ∧
  new_vegetable_oil_amount ≈ 2.57 := by
  sorry

end maintain_proportion_and_amounts_l827_827622


namespace value_of_b_l827_827204

theorem value_of_b (a b : ℝ) (h1 : 2 * a + 1 = 1) (h2 : b - a = 1) : b = 1 := 
by 
  sorry

end value_of_b_l827_827204


namespace probability_relatively_prime_to_42_l827_827184

/-- Two integers are relatively prime if they have no common factors other than 1 or -1. -/
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set of positive integers less than or equal to 42. -/
def in_range (n : ℕ) : Prop := n ≤ 42

/-- The set of integers that are relatively prime to 42 and less than or equal to 42. -/
def relatively_prime_to_42 (n : ℕ) : Prop := relatively_prime n 42 ∧ in_range n

/-- The probability that a positive integer less than or equal to 42 is relatively prime to 42. 
Expressed as a common fraction. -/
theorem probability_relatively_prime_to_42 : 
  (Finset.filter relatively_prime_to_42 (Finset.range 43)).card * 7 = 12 * 42 :=
sorry

end probability_relatively_prime_to_42_l827_827184


namespace sum_primes_between_10_and_20_is_60_l827_827234

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827234


namespace conjugate_of_complex_expression_l827_827485

theorem conjugate_of_complex_expression :
  let z : ℂ := (1 - 3 * complex.i) / complex.i + 2 * complex.i
  in complex.conj z = -3 - complex.i :=
by
  sorry

end conjugate_of_complex_expression_l827_827485


namespace no_extreme_value_l827_827665

-- Define the function f(x) for x > 0
def f (x : ℝ) : ℝ := x

-- Define the domain condition
def domain_condition (x : ℝ) : Prop := x > 0

-- State the problem in Lean 4 statement form
theorem no_extreme_value : ∀ x : ℝ, domain_condition x → (¬ (∀ T : ℝ, ∃ a : ℝ, a = f(x) ∧ domain_condition a ∧ f(a) = T ∨ f(a) ≤ T ∧ f(x) ≤ f(a))) :=
by
  sorry

end no_extreme_value_l827_827665


namespace wire_gap_height_l827_827174

def radius_of_earth_at_equator : ℝ := 6378 * 1000  -- Conversion to meters

def initial_circumference (R : ℝ) : ℝ := 2 * Real.pi * R

def new_circumference (R : ℝ) : ℝ := initial_circumference R + 1  -- Increased by 1 meter

def height_gap (R : ℝ) : ℝ := 1 / (2 * Real.pi)  -- Calculation for h

theorem wire_gap_height (R : ℝ) (h : ℝ) (H : R = radius_of_earth_at_equator) :
  2 * Real.pi * (R + h) = new_circumference R → h = height_gap R :=
by
  intro H1
  rw [new_circumference, initial_circumference, height_gap, H]
  sorry

end wire_gap_height_l827_827174


namespace uneaten_chips_l827_827433

theorem uneaten_chips :
  ∀ (chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips : ℕ),
    (chips_per_cookie = 7) →
    (cookies_total = 12 * 4) →
    (half_cookies = cookies_total / 2) →
    (uneaten_cookies = cookies_total - half_cookies) →
    (uneaten_chips = uneaten_cookies * chips_per_cookie) →
    uneaten_chips = 168 :=
by
  intros chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips
  intros chips_per_cookie_eq cookies_total_eq half_cookies_eq uneaten_cookies_eq uneaten_chips_eq
  rw [chips_per_cookie_eq, cookies_total_eq, half_cookies_eq, uneaten_cookies_eq, uneaten_chips_eq]
  norm_num
  sorry

end uneaten_chips_l827_827433


namespace AX_perp_HM_l827_827602

-- Definitions of given conditions
variable {α : Type*} [EuclideanGeometry α]
variables (A B C H M D E X : α)

-- Given conditions
variable (ABC_triangle : triangle α A B C)
variable (H_is_orthocenter : orthocenter α H A B C)
variable (M_is_midpoint : midpoint α M B C)
variable (D_on_AB : on_segment α D A B)
variable (E_on_AC : on_segment α E A C)
variable (AD_eq_AE : distance α A D = distance α A E)
variable (DHE_collinear : collinear α D H E)
variable (X_on_circumcircles : ∀ (P : α), on_circumcircle α A B C X ∧ on_circumcircle α A D E X)

-- Goal to prove
theorem AX_perp_HM : ∀ {A B C H M D E X : α},
  triangle α A B C →
  orthocenter α H A B C →
  midpoint α M B C →
  on_segment α D A B →
  on_segment α E A C →
  distance α A D = distance α A E →
  collinear α D H E →
  (∀ (P : α), on_circumcircle α A B C X ∧ on_circumcircle α A D E X) →
  perpendicular α (line_through α A X) (line_through α H M) :=
 sorry

end AX_perp_HM_l827_827602


namespace geometric_sequence_value_l827_827011

theorem geometric_sequence_value 
  (a : ℕ → ℝ) 
  (h1 : ∃ r, ∀ n, a (n + 1) = a n * r)
  (h2 : a 6 + a 8 = ∫ x in 0..4, real.sqrt (16 - x^2)) :
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 * real.pi ^ 2 :=
by
  have h3 : ∫ x in 0..4, real.sqrt (16 - x ^ 2) = 4 * real.pi, from sorry
  rw [h3] at h2
  sorry

end geometric_sequence_value_l827_827011


namespace inlet_pipe_filling_rate_l827_827357

def leak_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def net_emptying_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def inlet_rate_per_hour (net_rate : ℕ) (leak_rate : ℕ) : ℕ :=
  leak_rate - net_rate

def convert_to_minutes (rate_per_hour : ℕ) : ℕ :=
  rate_per_hour / 60

theorem inlet_pipe_filling_rate :
  let volume := 4320
  let time_to_empty_with_leak := 6
  let net_time_to_empty := 12
  let leak_rate := leak_rate volume time_to_empty_with_leak
  let net_rate := net_emptying_rate volume net_time_to_empty
  let fill_rate_per_hour := inlet_rate_per_hour net_rate leak_rate
  convert_to_minutes fill_rate_per_hour = 6 := by
    -- Proof ends with a placeholder 'sorry'
    sorry

end inlet_pipe_filling_rate_l827_827357


namespace monotonicity_of_f_range_of_b_l827_827937

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - a)/2 * x^2 - x

-- Statement for the first part
theorem monotonicity_of_f (a x : ℝ) (h : a < 1) :
  (if 0 < a then
    (if a < 1 / 2 then
      ∀ x, (0 < x ∧ x < a / (1 - a) ∨ 1 < x) → f a x > f a (x + ε) ∧ (a / (1 - a) < x ∧ x < 1) → f a x < f a (x + ε)
    else if a = 1 / 2 then
      ∀ x, 0 < x → f a x > f a (x + ε)
    else
      ∀ x, (0 < x ∧ x < 1 ∨ a / (1 - a) < x) → f a x > f a (x + ε) ∧ (1 < x ∧ x < a / (1 - a)) → f a x < f a (x + ε)
  ) else
    ∀ x, (0 < x ∧ x < 1 → f a x < f a (x + ε)) ∧ (x > 1 → f a x > f a (x + ε))
  ) :=
sorry

-- Statement for the second part
theorem range_of_b (x b : ℝ) (h : ∀ x > 0, b * x + 1 ≥ f 1 x) :
  b ∈ set.Ici ((1 / Real.exp 2) - 1) :=
sorry

end monotonicity_of_f_range_of_b_l827_827937


namespace gcd_lcm_problem_l827_827669

theorem gcd_lcm_problem (b : ℤ) (x : ℕ) (hx_pos : 0 < x) (hx : x = 12) :
  gcd 30 b = x + 3 ∧ lcm 30 b = x * (x + 3) → b = 90 := 
by
  sorry

end gcd_lcm_problem_l827_827669


namespace problem_solution_l827_827451

theorem problem_solution :
  0.45 * 0.65 + 0.1 * 0.2 = 0.3125 :=
by
  sorry

end problem_solution_l827_827451


namespace quadrilateral_not_necessarily_parallelogram_l827_827809

theorem quadrilateral_not_necessarily_parallelogram (Q : Type) [quadrilateral Q]
  (pair_parallel : ∃ (a b : Q), parallel a b)
  (pair_equal : ∃ (c d : Q), sides_equal c d) :
  ¬ parallelogram Q := 
sorry

end quadrilateral_not_necessarily_parallelogram_l827_827809


namespace minimizes_sum_at_point_M_l827_827967

/-- The point A with coordinates (3, 2) -/
def pointA : ℝ × ℝ := (3, 2)

/-- The focus of the parabola y^2 = 2x -/
def focusF : ℝ × ℝ := (1/2, 0)

/-- The parabola definition given as y^2 = 2x -/
def isOnParabola (M : ℝ × ℝ) : Prop := (M.2)^2 = 2 * (M.1)

/-- The minimum sum condition |MF| + |MA| is minimized at (2, 2) -/
def minimizesSum (M : ℝ × ℝ) : Prop := 
  ∀ N : ℝ × ℝ, isOnParabola N → 
  (dist N focusF + dist N pointA) ≥ (dist M focusF + dist M pointA)

/-- The point M (2, 2) minimizes the sum |MF| + |MA| -/
theorem minimizes_sum_at_point_M : isOnParabola (2, 2) ∧ minimizesSum (2, 2) :=
by
  sorry

end minimizes_sum_at_point_M_l827_827967


namespace rectangle_short_side_l827_827345

theorem rectangle_short_side
  (r : ℝ) (a_circle : ℝ) (a_rect : ℝ) (d : ℝ) (other_side : ℝ) :
  r = 6 →
  a_circle = Real.pi * r^2 →
  a_rect = 3 * a_circle →
  d = 2 * r →
  a_rect = d * other_side →
  other_side = 9 * Real.pi :=
by
  sorry

end rectangle_short_side_l827_827345


namespace square_construction_possible_l827_827903

noncomputable def construct_square (O : Point) (γ : Circle) (L : Line) : Prop :=
  ∃ A B C D : Point,
    circle.contains γ A ∧ circle.contains γ B ∧
    line.contains L C ∧ line.contains L D ∧
    is_square A B C D ∧
    adjacent A B ∧ adjacent C D

theorem square_construction_possible (O : Point) (γ : Circle) (L : Line) (h1: ∃ P : Point, circle.center γ = O ∧ ¬ line.contains L P) :
  construct_square O γ L :=
sorry

end square_construction_possible_l827_827903


namespace monkey_reaches_tree_top_in_hours_l827_827728

-- Definitions based on conditions
def height_of_tree : ℕ := 22
def hop_per_hour : ℕ := 3
def slip_per_hour : ℕ := 2
def effective_climb_per_hour : ℕ := hop_per_hour - slip_per_hour

-- The theorem we want to prove
theorem monkey_reaches_tree_top_in_hours
  (height_of_tree hop_per_hour slip_per_hour : ℕ)
  (h1 : height_of_tree = 22)
  (h2 : hop_per_hour = 3)
  (h3 : slip_per_hour = 2) :
  ∃ t : ℕ, t = 22 ∧ effective_climb_per_hour * (t - 1) + hop_per_hour = height_of_tree := by
  sorry

end monkey_reaches_tree_top_in_hours_l827_827728


namespace positive_integer_satisfies_condition_l827_827331

theorem positive_integer_satisfies_condition : 
  ∃ n : ℕ, (12 * n = n^2 + 36) ∧ n = 6 :=
by
  sorry

end positive_integer_satisfies_condition_l827_827331


namespace integer_solutions_for_abs_x_lt_3pi_l827_827169

theorem integer_solutions_for_abs_x_lt_3pi : 
  ∃ (s : Set ℤ), (∀ x ∈ s, |(x : ℝ)| < 3 * Real.pi) ∧ (s.card = 19) :=
by
  sorry

end integer_solutions_for_abs_x_lt_3pi_l827_827169


namespace pos_relationship_l827_827032

variables {a b : Line} {α : Plane}

def skew_lines (a b : Line) : Prop :=
  ¬ (a ∥ b) ∧ (∀ p ∈ a, ∀ q ∈ b, p ≠ q)

def parallel_to_plane (a : Line) (α : Plane) : Prop :=
  ∀ p ∈ a, ∃ l ∈ α, p ∈ l ∧ ∀ q ∈ a, ∃ r ∈ α, q ∈ r ∧ l ∥ r

theorem pos_relationship 
  (hab_skew : skew_lines a b)
  (ha_parallel_alpha : parallel_to_plane a α) : 
  (b ∥ α) ∨ (∃ p, p ∈ b ∧ p ∈ α) ∨ (b ⊆ α) :=
sorry

end pos_relationship_l827_827032


namespace point_on_line_segment_l827_827089

theorem point_on_line_segment (A B P : ℝ^3) (h : ∃ t u : ℚ, t + u = 1 ∧ P = t • A + u • B) (ratio: 3 / 8 = t ∧ 5 / 8 = u) :
    ∃ t u : ℚ, P = t • A + u • B ∧ t = 5 / 8 ∧ u = 3 / 8 := by
  sorry

end point_on_line_segment_l827_827089


namespace range_of_m_l827_827922

theorem range_of_m (m : ℝ) (θ : ℝ) : 
  (m^2 + (cos θ ^ 2 - 5) * m + 4 * sin θ ^ 2 ≥ 0) →
  (m ≥ 4 ∨ m ≤ 1) :=
sorry

end range_of_m_l827_827922


namespace units_digit_G_1000_l827_827627

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G_1000 : (G 1000) % 10 = 4 :=
  sorry

end units_digit_G_1000_l827_827627


namespace inequality_proof_l827_827877

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  -- Proof omitted
  sorry

end inequality_proof_l827_827877


namespace root_expr_value_eq_175_div_11_l827_827105

noncomputable def root_expr_value (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + bc + ca = 25) (h3 : abc = 10) : ℝ :=
  (a / (1 / a + b * c)) + (b / (1 / b + c * a)) + (c / (1 / c + a * b))

theorem root_expr_value_eq_175_div_11 (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : ab + bc + ca = 25) 
  (h3 : abc = 10) : 
  root_expr_value a b c h1 h2 h3 = 175 / 11 := 
sorry

end root_expr_value_eq_175_div_11_l827_827105


namespace circle_line_distance_eq_radius_half_l827_827511

theorem circle_line_distance_eq_radius_half (a : ℝ) :
  let C := {p : ℝ × ℝ | p.1^2 + p.2^2 + 6 * p.2 - a = 0},
      center := (0, -3 : ℝ),
      radius := real.sqrt (a + 9),
      line := {p : ℝ × ℝ | p.1 - p.2 - 1 = 0},
      distance := abs (-3 - 1) / real.sqrt (1^2 + (-1)^2)
  in distance = radius / 2 → a = -1 := 
by {
  sorry
}

end circle_line_distance_eq_radius_half_l827_827511


namespace probability_queen_in_center_after_2004_moves_l827_827052

def initial_probability (n : ℕ) : ℚ :=
if n = 0 then 1
else if n = 1 then 0
else if n % 2 = 0 then (1 : ℚ) / 2^(n / 2)
else (1 - (1 : ℚ) / 2^((n - 1) / 2)) / 2

theorem probability_queen_in_center_after_2004_moves :
  initial_probability 2004 = 1 / 3 + 1 / (3 * 2^2003) :=
sorry

end probability_queen_in_center_after_2004_moves_l827_827052


namespace number_of_players_l827_827807

theorem number_of_players (S : ℕ) (h1 : S = 22) (h2 : ∀ (n : ℕ), S = n * 2) : ∃ n, n = 11 :=
by
  sorry

end number_of_players_l827_827807


namespace students_in_math_class_l827_827395

theorem students_in_math_class (a b : ℕ) : 
  (a + b + 6 = 52) ∧ (2 * (a + 6) = b + 6) → b + 6 = 38 :=
by
  intros h,
  sorry

end students_in_math_class_l827_827395


namespace turtles_remaining_on_log_l827_827766
-- Importing necessary modules

-- Defining the problem
def initial_turtles : ℕ := 9
def turtles_climbed : ℕ := (initial_turtles * 3) - 2
def total_turtles : ℕ := initial_turtles + turtles_climbed
def remaining_turtles : ℕ := total_turtles / 2

-- Stating the proof problem
theorem turtles_remaining_on_log : remaining_turtles = 17 := 
  sorry

end turtles_remaining_on_log_l827_827766


namespace sum_primes_between_10_and_20_is_60_l827_827238

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827238


namespace coefficient_of_monomial_l827_827657

theorem coefficient_of_monomial (a b c : ℝ) : 
  coefficient (-2 * Real.pi * a^2 * b * c) = -2 * Real.pi :=
by
  sorry

end coefficient_of_monomial_l827_827657


namespace height_relationship_of_cylinders_l827_827704

variables {r₁ r₂ h₁ h₂ : ℝ}

theorem height_relationship_of_cylinders 
  (h₀ : π * r₁^2 * h₁ = π * r₂^2 * h₂) 
  (h₁ : r₂ = 1.2 * r₁) : h₁ = 1.44 * h₂ :=
by
  sorry

end height_relationship_of_cylinders_l827_827704


namespace derivative_at_point_is_not_constant_l827_827200

theorem derivative_at_point_is_not_constant :
  ∀ (f : ℝ → ℝ) (x : ℝ), ¬ (∃ c : ℝ, ∀ h : ℝ, h ≠ 0 → (f (x + h) - f x) / h = c) :=
by 
  intros f x,
  rw not_exists,
  intro c,
  rw not_forall,
  use 1, -- We can choose any non-zero h, here h = 1
  intro h_ne_zero,
  exfalso,
  sorry

end derivative_at_point_is_not_constant_l827_827200


namespace coeff_x4_in_f_f_x_expansion_l827_827009

def f (x : ℝ) : ℝ :=
  if x > 1 then (x + 1) ^ 5 else x ^ 2 + 2

theorem coeff_x4_in_f_f_x_expansion (h : 0 < x ∧ x < 1) :
  (f (f x).nat_coeff 4 = 270 :=
sorry

end coeff_x4_in_f_f_x_expansion_l827_827009


namespace max_visible_sum_of_cubes_l827_827854

theorem max_visible_sum_of_cubes :
  ∃ (cubes : List (List ℕ)),
  cubes.length = 4 ∧ 
  (∀ cube ∈ cubes, cube ⊆ [1, 2, 3, 4, 5, 6] ∧ cube.length = 6) ∧
  (∀ cube ∈ cubes, let hidden_faces := [(1, 3), (2, 4), (5, 6)];
                   (∀ x ∈ hidden_faces, x.1 ∈ cube ∧ x.2 ∈ cube)) ∧
  (∀ cube ∈ cubes, let visible_faces := cube.filter (λ n, n≠1 ∧ n≠3 ∧ n≠2 ∧ n≠4 ∧ n≠5 ∧ n≠6);
                   list.sum visible_faces = 17) ∧
  list.sum(cubes.map (λ cube, list.sum (cube.filter (λ n, n≠1 ∧ n≠3 ∧ n≠2 ∧ n≠4 ∧ n≠5 ∧ n≠6)))) = 68 :=
by
  sorry

end max_visible_sum_of_cubes_l827_827854


namespace total_distance_covered_l827_827353

theorem total_distance_covered :
  let speed_fox := 50       -- km/h
  let speed_rabbit := 60    -- km/h
  let speed_deer := 80      -- km/h
  let time_hours := 2       -- hours
  let distance_fox := speed_fox * time_hours
  let distance_rabbit := speed_rabbit * time_hours
  let distance_deer := speed_deer * time_hours
  distance_fox + distance_rabbit + distance_deer = 380 := by
sorry

end total_distance_covered_l827_827353


namespace point_equidistant_from_vertices_is_circumcenter_l827_827682

theorem point_equidistant_from_vertices_is_circumcenter (A B C : Type) [EuclideanSpace A] :
  ∃ (O : A), (∀ (P : A), (dist A P = dist B P ∧ dist A P = dist C P) ↔ P = O) :=
sorry

end point_equidistant_from_vertices_is_circumcenter_l827_827682


namespace equivalent_form_zero_l827_827112

theorem equivalent_form_zero (x : ℝ) : (sqrt (sin x ^ 4 + 3 * cos x ^ 2) - sqrt (cos x ^ 4 + 3 * sin x ^ 2)) = 0 :=
by
  sorry

end equivalent_form_zero_l827_827112


namespace max_rotation_angle_l827_827140

def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 4*x) - Real.sqrt 3

theorem max_rotation_angle
  (h1 : ∀ (x : ℝ), x ∈ Set.Icc (1 : ℝ) 3 → f x) 
  (h2 : ∀ (θ : ℝ), θ < π / 2)
  : ∃ θ : ℝ, θ = π / 3 ∧ (∀ x : ℝ, x ∈ Set.Icc (1 : ℝ) 3 → f x) := 
  sorry

end max_rotation_angle_l827_827140


namespace option_a_is_correct_l827_827724

theorem option_a_is_correct (a b : ℝ) :
  (a - b) * (-a - b) = b^2 - a^2 :=
sorry

end option_a_is_correct_l827_827724


namespace temperature_difference_for_rods_l827_827391

noncomputable def initial_temperature := 80
noncomputable def initial_length := 2 -- in meters
noncomputable def alpha_Fe := 0.0000118 -- coefficient of linear expansion for iron
noncomputable def alpha_Zn := 0.000031 -- coefficient of linear expansion for zinc
noncomputable def length_difference := 0.0015 -- in meters

theorem temperature_difference_for_rods :
  ∃ x₁ x₂ : ℝ, 
    (2 * (1 + alpha_Fe * (x₁ - initial_temperature)) - 2 * (1 + alpha_Zn * (x₁ - initial_temperature)) = length_difference) ∧
    (2 * (1 + alpha_Fe * (x₂ - initial_temperature)) - 2 * (1 + alpha_Zn * (x₂ - initial_temperature)) = length_difference) ∧
    (x₁ = 41) ∧ (x₂ = 119) :=
sorry

end temperature_difference_for_rods_l827_827391


namespace minimum_value_of_ratio_minimum_value_example_l827_827867

def f (n : ℕ) : ℕ :=
  (List.range (n + 1)).sum (λ k, (k+1)^(n-k))

theorem minimum_value_of_ratio (n : ℕ) (h : 1 ≤ n) : 
  (f(n+1) / f(n)) ≥ 3 :=
by
  sorry

-- Minimum value occurs at n = 2
theorem minimum_value_example : (f 3 / f 2) = 8 / 3 :=
by
  sorry

end minimum_value_of_ratio_minimum_value_example_l827_827867


namespace turtles_remaining_on_log_l827_827768

-- Definition of the problem parameters
def initial_turtles : ℕ := 9
def additional_turtles : ℕ := (3 * initial_turtles) - 2
def total_turtles : ℕ := initial_turtles + additional_turtles
def frightened_turtles : ℕ := total_turtles / 2
def remaining_turtles : ℕ := total_turtles - frightened_turtles

-- The final theorem stating the number of turtles remaining
theorem turtles_remaining_on_log : remaining_turtles = 17 := by
  -- Proof is omitted
  sorry

end turtles_remaining_on_log_l827_827768


namespace convex_quadrilateral_parallelogram_l827_827638

theorem convex_quadrilateral_parallelogram 
  (A B C D M : Point)
  (hconvex : Convex (Quadrilateral A B C D))
  (hdiag1 : divides_into_equal_areas A C (Quadrilateral A B C D))
  (hdiag2 : divides_into_equal_areas B D (Quadrilateral A B C D))
  (hinter : intersection_diagonals A C B D = M)
  : is_parallelogram (Quadrilateral A B C D) := 
sorry

end convex_quadrilateral_parallelogram_l827_827638


namespace peter_pizza_fraction_l827_827633

def pizza_slices : ℕ := 16
def peter_slices_alone : ℕ := 2
def shared_slice : ℚ := 1 / 2

theorem peter_pizza_fraction :
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  total_fraction = 5 / 32 :=
by
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  sorry

end peter_pizza_fraction_l827_827633


namespace sum_primes_10_to_20_l827_827296

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827296


namespace nell_more_ace_cards_than_baseball_cards_l827_827628

theorem nell_more_ace_cards_than_baseball_cards :
  let initial_baseball_cards := 239 in
  let initial_ace_cards := 38 in
  let current_ace_cards := 376 in
  let current_baseball_cards := 111 in
  current_ace_cards - current_baseball_cards = 265 :=
by
  let initial_baseball_cards := 239
  let initial_ace_cards := 38
  let current_ace_cards := 376
  let current_baseball_cards := 111
  show current_ace_cards - current_baseball_cards = 265
  calc
    current_ace_cards - current_baseball_cards
        = 376 - 111 : by rfl
    ... = 265 : by norm_num
  sorry

end nell_more_ace_cards_than_baseball_cards_l827_827628


namespace angle_skew_lines_BA₁_AC₁_is_60_l827_827065

open EuclideanGeometry

-- Definitions related to the problem
variables (A B C A₁ B₁ C₁ D : Point)
variable  (RightTriangularPrism : Prop)
variable  (AngleBAC : Angle = 90)
variable  (EqualityAB_AC_AA₁ : length A B = length A C ∧ length A B = length A A₁)

-- The theorem to prove
theorem angle_skew_lines_BA₁_AC₁_is_60 :
  RightTriangularPrism → AngleBAC → EqualityAB_AC_AA₁ → ∠(B, A₁, C₁) = 60 :=
by
  sorry

end angle_skew_lines_BA₁_AC₁_is_60_l827_827065


namespace a_1998_value_l827_827122

/-- An increasing sequence of nonnegative integers where every nonnegative integer can 
    be uniquely expressed as \(a_i + 2a_j + 4a_k\). --/
def sequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ n, ∃ (i j k : ℕ), n = a i + 2 * a j + 4 * a k)

/-- Determine the value of \(a_{1998}\) in the given sequence. --/
theorem a_1998_value (a : ℕ → ℕ) (h : sequence a) : 
  a 1998 = 1227096648 :=
sorry

end a_1998_value_l827_827122


namespace point_P_position_l827_827462

variable {a b c d : ℝ}
variable (h1: a ≠ b) (h2: a ≠ c) (h3: a ≠ d) (h4: b ≠ c) (h5: b ≠ d) (h6: c ≠ d)

theorem point_P_position (P : ℝ) (hP: b < P ∧ P < c) (hRatio: (|a - P| / |P - d|) = (|b - P| / |P - c|)) : 
  P = (a * c - b * d) / (a - b + c - d) := 
by
  sorry

end point_P_position_l827_827462


namespace unique_x_ffx_eq_5_l827_827164

noncomputable def f (x : ℝ) : ℝ :=
  if h : (-4 : ℝ) ≤ x ∧ x ≤ -1 then -x + 6
  else if h : (-1 : ℝ) < x ∧ x ≤ 3 then x - 1
  else if h : (3 : ℝ) < x ∧ x ≤ 5 then -2 * x + 10
  else 0   -- Default case for f defined outside the required range, for completeness

theorem unique_x_ffx_eq_5 :
  ∃! x : ℝ, f(f(x)) = 5 := by
  sorry

end unique_x_ffx_eq_5_l827_827164


namespace sum_first_n_terms_l827_827942

theorem sum_first_n_terms (a b : ℕ → ℕ) (n : ℕ)
    (h1 : ∀ k : ℕ, k ≥ 2 → a k = 2 * a (k - 1) + 1)
    (h2 : a 1 = 1)
    (h3 : ∀ k : ℕ, b k = a k + 1) :
    let T (n : ℕ) := ∑ i in finset.range (n + 1), i * b i in
    T n = 2 + (n - 1) * 2 ^ (n + 1) :=
by
  sorry

end sum_first_n_terms_l827_827942


namespace range_of_b_div_a_l827_827369

theorem range_of_b_div_a (a b : ℝ) (x1 x2 : ℝ) :
  (a ≠ 0) ∧
  (0 < x1 ∧ x1 < 1) ∧ (x2 > 1) ∧
  (x1^2 + (a + 1) * x1 + a + b + 1 = 0) ∧
  (x2^2 + (a + 1) * x2 + a + b + 1 = 0) →
  (-3 < 2a + b) ∧ (a + b > -1) →
  -2 < b / a ∧ b / a < -1 :=
begin
  sorry
end

end range_of_b_div_a_l827_827369


namespace count_sets_without_good_elements_of_size_3_l827_827943

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def is_good_element (S : Set ℕ) (x : ℕ) : Prop :=
  (x + 1 ∉ S) ∧ (x - 1 ∉ S)

def all_subsets_of_size (S : Set ℕ) (n : ℕ) : Set (Set ℕ) :=
  { T | T ⊆ S ∧ T.card = n }

def has_good_element (S T : Set ℕ) : Prop :=
  ∃ x ∈ T, is_good_element S x

def subsets_without_good_elements (S : Set ℕ) (size : ℕ) : Set (Set ℕ) :=
  { T ∈ all_subsets_of_size S size | ¬has_good_element S T }

theorem count_sets_without_good_elements_of_size_3 : 
  (subsets_without_good_elements S 3).card = 6 := 
sorry

end count_sets_without_good_elements_of_size_3_l827_827943


namespace chef_needs_200_fries_l827_827343

-- Define initial conditions
def fries_per_potato : ℕ := 25
def initial_potatoes : ℕ := 15
def leftover_potatoes : ℕ := 7

-- Define the number of fries the chef needs to make
def needed_fries : ℕ :=
  let used_potatoes := initial_potatoes - leftover_potatoes
    used_potatoes * fries_per_potato

-- State the theorem
theorem chef_needs_200_fries : needed_fries = 200 := by
  sorry

end chef_needs_200_fries_l827_827343


namespace unit_digit_3_pow_2012_sub_1_l827_827415

theorem unit_digit_3_pow_2012_sub_1 :
  (3 ^ 2012 - 1) % 10 = 0 :=
sorry

end unit_digit_3_pow_2012_sub_1_l827_827415


namespace length_of_train_l827_827800

-- Definitions based on the conditions in the problem
def time_to_cross_signal_pole : ℝ := 18
def time_to_cross_platform : ℝ := 54
def length_of_platform : ℝ := 600.0000000000001

-- Prove that the length of the train is 300.00000000000005 meters
theorem length_of_train
    (L V : ℝ)
    (h1 : L = V * time_to_cross_signal_pole)
    (h2 : L + length_of_platform = V * time_to_cross_platform) :
    L = 300.00000000000005 :=
by
  sorry

end length_of_train_l827_827800


namespace solve_equation_l827_827650

theorem solve_equation : ∀ (x : ℝ), 2 * (x - 1) = 2 - (5 * x - 2) → x = 6 / 7 :=
by
  sorry

end solve_equation_l827_827650


namespace sum_of_first_20_terms_l827_827322

variable (a d : ℝ) 

def nth_term (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

def sum_first_n_terms (a d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)

theorem sum_of_first_20_terms 
  (a d : ℝ) 
  (h : nth_term a d 4 + nth_term a d 12 = 20) : 
  sum_first_n_terms a d 20 = 200 + 50 * d := by
  sorry

end sum_of_first_20_terms_l827_827322


namespace S9_value_l827_827743

variable {x : ℝ}

def S (m : ℕ) : ℝ :=
  x^m + (1 / x)^m

theorem S9_value (h : x + 1/x = 4) : S 9 = 140248 :=
  sorry

end S9_value_l827_827743


namespace result_when_decreased_by_5_and_divided_by_7_l827_827043

theorem result_when_decreased_by_5_and_divided_by_7 (x y : ℤ)
  (h1 : (x - 5) / 7 = y)
  (h2 : (x - 6) / 8 = 6) :
  y = 7 :=
by
  sorry

end result_when_decreased_by_5_and_divided_by_7_l827_827043


namespace jessica_coloring_l827_827595

noncomputable def color_grid_ways : ℕ :=
  3

theorem jessica_coloring (colors : ℕ) (adj_diff : ∀ i j : ℕ, (adjacent i j) → color i ≠ color j) :
  colors = 3 → (grid_ways colors adj_diff) = 3 :=
by
  sorry

def adjacent (i j : ℕ) : Prop := -- Definition to determine if i, j are adjacent (sharing a side)
  (i / 3 = j / 3 ∧ (i % 3 = j % 3 + 1 ∨ i % 3 = j % 3 - 1)) ∨
  (i % 3 = j % 3 ∧ (i / 3 = j / 3 + 1 ∨ i / 3 = j / 3 - 1))

def color (i : ℕ) : ℕ := -- Definition to assign a color to each grid cell
  sorry -- This would typically depend on a specific coloring strategy, omitted for simplicity

def grid_ways (colors : ℕ) (adj_diff : ∀ i j : ℕ, (adjacent i j) → color i ≠ color j) : ℕ :=
  -- Here would be the logic to count the valid colorings based on the conditions
  sorry

end jessica_coloring_l827_827595


namespace better_partial_repayment_now_l827_827829

variable (T : ℝ) (S : ℝ) (r : ℝ)

theorem better_partial_repayment_now : 
  let immediate_balance := S - (T - 0.5 * r * S) - T + 0.5 * r * (S - T + 0.5 * r * S) in
  let end_balance := S - 2 * T + r * S in
  immediate_balance < end_balance :=
by {
  sorry
}

end better_partial_repayment_now_l827_827829


namespace sum_primes_between_10_and_20_is_60_l827_827231

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827231


namespace distance_between_stations_l827_827710

theorem distance_between_stations
  (v₁ v₂ : ℝ)
  (D₁ D₂ : ℝ)
  (T : ℝ)
  (h₁ : v₁ = 20)
  (h₂ : v₂ = 25)
  (h₃ : D₂ = D₁ + 70)
  (h₄ : D₁ = v₁ * T)
  (h₅ : D₂ = v₂ * T) : 
  D₁ + D₂ = 630 := 
by
  sorry

end distance_between_stations_l827_827710


namespace area_of_triangle_DMN_l827_827083

variable (a b c : ℝ)

theorem area_of_triangle_DMN (x y : ℝ) :
  ∃ z, x * y = z ∧ (2 * b) * z = z^2 - 2 * (a + c) * z + 4 * a * c →
  ∃ area_DMN, area_DMN = sqrt((a + b + c)^2 - 4 * a * c) :=
sorry

end area_of_triangle_DMN_l827_827083


namespace comedies_in_terms_of_a_l827_827320

variable (T a : ℝ)
variables (Comedies Dramas Action : ℝ)
axiom Condition1 : Comedies = 0.64 * T
axiom Condition2 : Dramas = 5 * a
axiom Condition3 : Action = a
axiom Condition4 : Comedies + Dramas + Action = T

theorem comedies_in_terms_of_a : Comedies = 10.67 * a :=
by sorry

end comedies_in_terms_of_a_l827_827320


namespace find_b_c_l827_827980

noncomputable def triangle_AB (a b c : ℝ) : Prop :=
  a = 3 ∧
  ∠C = 2 * Real.pi / 3 ∧
  (1 / 2) * a * b * Real.sin (2 * Real.pi / 3) = 3 * Real.sqrt 3 / 4

theorem find_b_c : ∃ b c : ℝ, 
  triangle_AB 3 b c ∧ b = 1 ∧ c = Real.sqrt 13 :=
begin
  sorry
end

end find_b_c_l827_827980


namespace inequality_proof_l827_827874

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  -- Proof omitted
  sorry

end inequality_proof_l827_827874


namespace smallest_floor_sum_l827_827033

theorem smallest_floor_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (⌊(a + b + d) / c⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + a + d) / b⌋) = 9 :=
sorry

end smallest_floor_sum_l827_827033


namespace kevin_cards_on_last_page_l827_827079

theorem kevin_cards_on_last_page :
  let total_albums := 10
  let pages_per_album := 50
  let cards_per_page_old := 8
  let cards_per_page_new := 12
  let fully_filled_albums := 5
  let pages_filled_in_sixth_album := 40
  let total_old_cards := total_albums * pages_per_album * cards_per_page_old
  let cards_in_fully_filled_albums := fully_filled_albums * pages_per_album * cards_per_page_new
  let cards_in_sixth_album := pages_filled_in_sixth_album * cards_per_page_new
  let total_new_cards_filled := cards_in_fully_filled_albums + cards_in_sixth_album
  let cards_remaining := total_old_cards - total_new_cards_filled
  let last_page_cards := cards_remaining - cards_in_sixth_album
  in last_page_cards = 40 := by
  let total_albums := 10
  let pages_per_album := 50
  let cards_per_page_old := 8
  let cards_per_page_new := 12
  let fully_filled_albums := 5
  let pages_filled_in_sixth_album := 40
  let total_old_cards := total_albums * pages_per_album * cards_per_page_old
  let cards_in_fully_filled_albums := fully_filled_albums * pages_per_album * cards_per_page_new
  let cards_in_sixth_album := pages_filled_in_sixth_album * cards_per_page_new
  let total_new_cards_filled := cards_in_fully_filled_albums + cards_in_sixth_album
  let cards_remaining := total_old_cards - total_new_cards_filled
  let last_page_cards := cards_remaining
  show last_page_cards = 40 from sorry

end kevin_cards_on_last_page_l827_827079


namespace area_of_triangle_ABC_l827_827344

noncomputable def triangle_area (r₁ r₂ : ℝ) (tangent_triangle : ℝ → Set (ℝ × ℝ)) : ℝ :=
-- assume the function that calculates the area, passed conditions as parameters.
sorry

theorem area_of_triangle_ABC (r1 r2 : ℝ) (tangent_triangle : ℝ → Set (ℝ × ℝ))
  (H1 : r1 = 2) (H2 : r2 = 3)
  (H3 : ∃ A B C : ℝ × ℝ, tangent_triangle r1 = {A, B, C} ∧ tangent_triangle r2 = {A, B, C} 
        ∧ (dist A B = dist A C)) : triangle_area r1 r2 tangent_triangle = 26 * real.sqrt 6 :=
sorry

end area_of_triangle_ABC_l827_827344


namespace max_people_with_apples_l827_827443

theorem max_people_with_apples (n : ℕ) (h : ∑ i in finset.range (n + 1), i ≤ 100) : n ≤ 13 := sorry

end max_people_with_apples_l827_827443


namespace inequality_proof_l827_827885

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  sorry

end inequality_proof_l827_827885


namespace exists_center_of_gravity_at_O_l827_827118

-- Define the physical scenario of the triangle with a point inside
structure Triangle := 
(A B C O : Point)
(h_in_triangle : Inside O (Triangle.mk A B C))

-- Mass assignments to the vertices of the triangle
structure MassAssignment (A B C : Point) := 
(m1 m2 m3 : ℝ)
(hm_positive : m1 > 0 ∧ m2 > 0 ∧ m3 > 0)

-- Define the centroid function given the mass assignments at vertices
def centroid (A B C : Point) (m : MassAssignment A B C) : Point :=
((m.m1 * A.x + m.m2 * B.x + m.m3 * C.x) / (m.m1 + m.m2 + m.m3), 
 (m.m1 * A.y + m.m2 * B.y + m.m3 * C.y) / (m.m1 + m.m2 + m.m3))

-- The statement of the problem, i.e., there exists such a MassAssignment
theorem exists_center_of_gravity_at_O 
{A B C O : Point} 
(h : Triangle A B C O) : 
∃ m : MassAssignment A B C, centroid A B C m = O :=
sorry

end exists_center_of_gravity_at_O_l827_827118


namespace sum_primes_10_20_l827_827227

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827227


namespace sequence_term_l827_827505

theorem sequence_term (a : ℕ → ℤ) (n : ℕ) (h : ∀ n, a n = (-1)^n * n) : a 4 = 4 :=
by
  have h₄ : a 4 = (-1)^4 * 4 := h 4
  have h₄_simp : (-1)^4 * 4 = 1 * 4 := by norm_num
  rw h₄_simp at h₄
  rw mul_one at h₄
  exact h₄

end sequence_term_l827_827505


namespace problem_1_problem_2_problem_3_l827_827946

def vec_a : ℝ × ℝ := (1, real.sqrt 3)
def vec_b : ℝ × ℝ := (-2, 0)

def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

def vec_norm (u : ℝ × ℝ) : ℝ :=
  real.sqrt (u.1 * u.1 + u.2 * u.2)

def vec_dot (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def calc_angle (u v : ℝ × ℝ) : ℝ :=
  real.arccos (vec_dot u v / (vec_norm u * vec_norm v))

theorem problem_1 : vec_norm (vec_sub vec_a vec_b) = 2 * real.sqrt 3 := 
by sorry

theorem problem_2 : calc_angle (vec_sub vec_a vec_b) vec_a = real.pi / 6 := 
by sorry

theorem problem_3 {t : ℝ} : 
  ∃ l, ∀ t, vec_norm (vec_sub vec_a (t • vec_b)) ≥ l ∧ l = real.sqrt 3 := 
by sorry

end problem_1_problem_2_problem_3_l827_827946


namespace mixed_operation_with_rationals_l827_827410

theorem mixed_operation_with_rationals :
  (- (2 / 21)) / (1 / 6 - 3 / 14 + 2 / 3 - 9 / 7) = 1 / 7 := 
by 
  sorry

end mixed_operation_with_rationals_l827_827410


namespace shortest_path_to_vertex_midpoint_edge_l827_827389

noncomputable def shortest_path_on_octahedron (a : ℝ) : ℝ :=
  let h := (Real.sqrt 3) / 2 in h

theorem shortest_path_to_vertex_midpoint_edge : shortest_path_on_octahedron 1 = Real.sqrt 3 / 2 :=
  sorry

end shortest_path_to_vertex_midpoint_edge_l827_827389


namespace number_of_valid_pairs_l827_827437

noncomputable def numberOfPairs : ℕ :=
  let log3 := Real.log 3
  let log4 := Real.log 4
  let lower_bound_m (n : ℕ) := Real.ceil ((log3 / log4) * n)
  let upper_bound_m (n : ℕ) := Real.floor ((log3 / log4) * (n + 1) - 3)
  List.length (List.filter (λ (p : ℕ × ℕ), 
    let m := p.1
    let n := p.2
    1 ≤ m ∧ m ≤ 3019 ∧ lower_bound_m n ≤ m ∧ m ≤ upper_bound_m n
  ) (List.product (List.range 3019) (List.range 1025)))

theorem number_of_valid_pairs : numberOfPairs = 1024 :=
sorry

end number_of_valid_pairs_l827_827437


namespace count_4_letter_words_with_E_l827_827023

theorem count_4_letter_words_with_E :
  let letters := {'A', 'B', 'C', 'D', 'E'}
  let total_4_letter_words := (letters.card) ^ 4
  let words_without_E := (letters.erase 'E').card ^ 4
  total_4_letter_words - words_without_E = 369 := by
  sorry

end count_4_letter_words_with_E_l827_827023


namespace pure_imaginary_a_zero_l827_827552

theorem pure_imaginary_a_zero (a : ℝ) (i : ℂ) (hi : i^2 = -1) :
  (z = (1 - (a:ℝ)^2 * i) / i) ∧ (∀ (z : ℂ), z.re = 0 → z = (0 : ℂ)) → a = 0 :=
by
  sorry

end pure_imaginary_a_zero_l827_827552


namespace decimal_to_binary_19_l827_827833

theorem decimal_to_binary_19 : nat_to_binary 19 = 10011 := by
  sorry

end decimal_to_binary_19_l827_827833


namespace correct_calculation_of_given_options_l827_827312

theorem correct_calculation_of_given_options (a : ℝ) : 
  (a^3)^3 = a^(3*3) ∧ 
  ¬((2*a)^3 = 6 * a^3) ∧ 
  ¬((-a^3)^2 = -a^6) ∧ 
  ¬((-a^2)^3 = a^6) :=
by {
  split,
  { calc (a^3)^3 = a^(3*3) : by rw [pow_mul] },
  split,
  { intro h, 
    have h1 : (2*a)^3 = 2^3 * a^3 := by rw [mul_pow], 
    contradiction },
  split,
  { intro h, 
    have h2 : (-a^3)^2 = a^6 := by rw [neg_pow_of_even], 
    contradiction },
  { intro h, 
    have h3 : (-a^2)^3 = -(a^6) := by rw [neg_pow_of_odd], 
    contradiction }
}

end correct_calculation_of_given_options_l827_827312


namespace find_S_9_l827_827619

-- Conditions
def aₙ (n : ℕ) : ℕ := sorry  -- arithmetic sequence

def Sₙ (n : ℕ) : ℕ := sorry  -- sum of the first n terms of the sequence

axiom condition_1 : 2 * aₙ 8 = 6 + aₙ 11

-- Proof goal
theorem find_S_9 : Sₙ 9 = 54 :=
sorry

end find_S_9_l827_827619


namespace sports_club_play_both_l827_827321

theorem sports_club_play_both 
  (N B T Neither : ℕ)
  (hN : N = 30)
  (hB : B = 17)
  (hT : T = 19)
  (hNeither : Neither = 3) :
  (B + T - N + Neither = 9) :=
by
  rw [hN, hB, hT, hNeither]
  norm_num
  sorry

end sports_club_play_both_l827_827321


namespace problem_statement_l827_827883

noncomputable def a : ℝ := 31 / 32
noncomputable def b : ℝ := Real.cos (1 / 4)
noncomputable def c : ℝ := 4 * Real.sin (1 / 4)

theorem problem_statement : c > b ∧ b > a := 
by {
  sorry
}

end problem_statement_l827_827883


namespace sum_of_primes_between_10_and_20_is_60_l827_827275

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827275


namespace seq_b_100_l827_827795

def seq_b (n : ℕ) : ℚ :=
  if n = 1 then 2
  else if n = 2 then 2 / 3
  else (2 - seq_b (n - 1)) / (3 * seq_b (n - 2))

theorem seq_b_100 : seq_b 100 = 33 / 100 :=
by sorry

end seq_b_100_l827_827795


namespace smallest_positive_period_is_pi_range_of_f_on_interval_l827_827521

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.sin (x + Real.pi / 2)

-- Statement 1: Prove the smallest positive period is π
theorem smallest_positive_period_is_pi :
  ∃ T > 0, ∀ x, f x = f (x + T) ∧ (∀ T', (T' > 0 ∧ ∀ x, f x = f (x + T')) → T ≤ T') :=
sorry

-- Statement 2: Prove the range of f(x) on [0, 2π/3] is [0, 3]
theorem range_of_f_on_interval :
  set.range (λ x, if 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 then f x else 0) = set.Icc 0 3 :=
sorry

end smallest_positive_period_is_pi_range_of_f_on_interval_l827_827521


namespace part1_part2_l827_827615

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x + 1) - 2

-- Conditions
variables (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
variables (point_proof : f a 1 = 7)

-- Proof goal 1: Prove that if the function passes through (1, 7), then a = 3 and find the zero 
theorem part1 : a = 3 ∧ (∃ x, f a x = 0 ∧ x = Real.log 2 / Real.log 3 - 1) :=
by sorry

-- Define the specific function g where a = 3
def g (x : ℝ) : ℝ := f 3 x

-- Proof goal 2: Prove the solution set of the inequality g(x) >= -5/3 is x >= -2
theorem part2 : ∀ x, g x ≥ - (5 / 3) ↔ x ≥ -2 :=
by sorry

end part1_part2_l827_827615


namespace phillip_english_percentage_l827_827635

-- Define the context and conditions
variable (math_questions : Nat) (math_percentage : ℝ) (english_questions : Nat) (total_correct : Nat)
variable (math_correct : ℝ) (english_correct : ℝ) (english_percentage : ℝ)

-- Assign the given conditions
axiom h1 : math_questions = 40
axiom h2 : math_percentage = 0.75
axiom h3 : english_questions = 50
axiom h4 : total_correct = 79

-- Define the number of correctly answered questions in math and English
def math_correct := math_percentage * math_questions
def english_correct := total_correct - math_correct

-- Define the percentage of correctly answered English questions
def english_percentage := (english_correct / english_questions) * 100

-- The statement to prove
theorem phillip_english_percentage : english_percentage = 98 := by
  -- Insert proof steps here
  sorry

end phillip_english_percentage_l827_827635


namespace problem_statement_l827_827407

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l827_827407


namespace chips_left_uneaten_l827_827425

theorem chips_left_uneaten 
    (chips_per_cookie : ℕ)
    (cookies_per_dozen : ℕ)
    (dozens_of_cookies : ℕ)
    (cookies_eaten_ratio : ℕ) 
    (h_chips : chips_per_cookie = 7)
    (h_cookies_dozen : cookies_per_dozen = 12)
    (h_dozens : dozens_of_cookies = 4)
    (h_eaten_ratio : cookies_eaten_ratio = 2) : 
  (cookies_per_dozen * dozens_of_cookies / cookies_eaten_ratio) * chips_per_cookie = 168 :=
by 
  sorry

end chips_left_uneaten_l827_827425


namespace prevent_white_cube_n2_prevent_white_cube_n3_l827_827634

def min_faces_to_paint (n : ℕ) : ℕ :=
  if n = 2 then 2 else if n = 3 then 12 else sorry

theorem prevent_white_cube_n2 : min_faces_to_paint 2 = 2 := by
  sorry

theorem prevent_white_cube_n3 : min_faces_to_paint 3 = 12 := by
  sorry

end prevent_white_cube_n2_prevent_white_cube_n3_l827_827634


namespace find_n_values_l827_827915

theorem find_n_values (n : ℕ)
  (h : ∃ k : ℕ, 37.5^n + 26.5^n = k) : 
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 :=
sorry

end find_n_values_l827_827915


namespace sum_of_primes_between_10_and_20_is_60_l827_827272

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827272


namespace solution_l827_827101

noncomputable def problem (a b c : ℝ) : Prop :=
  (Polynomial.eval a (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval b (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (Polynomial.eval c (Polynomial.mk [0, 0, -10, 15, -25, 1]) = 0) ∧
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)

theorem solution (a b c : ℝ) (h : problem a b c) : 
  (∃ abc : ℝ, abc = a * b * c ∧ abc = 10) →
  (a + b + c = 15) ∧ (a * b + b * c + c * a = 25) →
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 175 / 11) :=
sorry

end solution_l827_827101


namespace probability_x_leq_1_l827_827363

noncomputable def probability_in_interval (a b : ℝ) (p : ℝ) : Prop :=
  if h : a < b then (p = (min b (1 : ℝ) - a) / (b - a)) else false

theorem probability_x_leq_1 (a b : ℝ) (hab : a < b) (ha : a = 0) (hb : b = 3) :
  probability_in_interval a b (1 / 3) :=
by
  unfold probability_in_interval
  rw [ha, hb]
  split_ifs
  . exact rfl
  . exfalso
    linarith

end probability_x_leq_1_l827_827363


namespace seq_formula_l827_827117

def S (n : ℕ) (a : ℕ → ℤ) : ℤ := 2 * a n + 1

theorem seq_formula (a : ℕ → ℤ) (S_n : ℕ → ℤ)
  (hS : ∀ n, S_n n = S n a) :
  a = fun n => -2^(n-1) := by
  sorry

end seq_formula_l827_827117


namespace line_passes_through_fixed_point_l827_827571

theorem line_passes_through_fixed_point :
  ∃ (l : ℝ → ℝ) (P Q : ℝ × ℝ), 
    (∀ x y, l x = y → (x^2 / 3 + y^2 = 1)) ∧
    (P.2 = l P.1) ∧ 
    (Q.2 = l Q.1) ∧ 
    (P ≠ (0, 1)) ∧ 
    (Q ≠ (0, 1)) ∧ 
    ((P.1, P.2 - 1) ⬝ (Q.1, Q.2 - 1) = 0) → 
    ∃! (fp : ℝ × ℝ), fp = (0, -1 / 2) :=
sorry

end line_passes_through_fixed_point_l827_827571


namespace calculate_expr_l827_827398

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l827_827398


namespace percent_less_than_m_add_d_l827_827341

variable (population : Type) [measurable_space population] (measure : measure population)
variable (m d : ℝ)
variable (distribution : population → ℝ)

-- Assume distribution is symmetric about mean m
def symmetric_about_mean (m : ℝ) (dist : population → ℝ) : Prop :=
  ∀ x, dist (2 * m - x) = dist x

-- Assume 60% of the distribution lies within one standard deviation d of mean
def within_one_std_dev (d : ℝ) (dist : population → ℝ) : Prop :=
  measure ({x | dist x ∈ set.Icc (m - d) (m + d)}) = 0.6

theorem percent_less_than_m_add_d :
  symmetric_about_mean m distribution →
  within_one_std_dev d distribution →
  measure ({x | distribution x < m + d}) = 0.8 :=
by
  sorry

end percent_less_than_m_add_d_l827_827341


namespace negation_of_p_is_not_p_l827_827528

-- Define the proposition p
def p (x : ℝ) : Prop := ∀ x ∈ set.Ioi 0, 3 * x + 1 < 0

-- State the negation of p
def not_p : Prop := ∃ x ∈ set.Ioi 0, 3 * x + 1 ≥ 0

-- The Lean statement that shows the proof problem
theorem negation_of_p_is_not_p : ¬p = not_p := by
  sorry

end negation_of_p_is_not_p_l827_827528


namespace inequality_proof_l827_827887

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  sorry

end inequality_proof_l827_827887


namespace sum_of_corners_l827_827325

noncomputable def rectangle := { n m : ℕ // n * m = 221 }

theorem sum_of_corners (s : rectangle) (a : ℕ → ℕ → ℝ)
  (sum_elements : (∑ i in finset.range s.1, ∑ j in finset.range s.2, a i j) = 110721)
  (rows_arith_prog : ∀ i, is_arith_prog (finset.range s.2).map (λ j, a i j))
  (first_col_arith_prog : is_arith_prog (finset.range s.1).map (λ i, a i 0))
  (last_col_arith_prog : is_arith_prog (finset.range s.1).map (λ i, a i (s.2 - 1))) :
  a 0 0 + a 0 (s.2 - 1) + a (s.1 - 1) 0 + a (s.1 - 1) (s.2 - 1) = 2004 := 
sorry

end sum_of_corners_l827_827325


namespace eval_op_l827_827966

namespace ProofProblem

-- Define the new operation ⊕
def op (x y : ℝ) : ℝ := Real.sqrt (x * y + 4)

-- State the problem: prove that (4 ⊕ 8) ⊕ 2 = 4
theorem eval_op : op (op 4 8) 2 = 4 := 
by 
  sorry

end ProofProblem

end eval_op_l827_827966


namespace anna_prob_at_least_two_correct_l827_827392

open scoped BigOperators

noncomputable def binomial_probability (k n : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

def prob_at_least_two_correct : ℝ :=
  1 - (binomial_probability 0 5 (1/4) + binomial_probability 1 5 (1/4))

theorem anna_prob_at_least_two_correct : 
  prob_at_least_two_correct = 47 / 128 :=
by
  sorry

end anna_prob_at_least_two_correct_l827_827392


namespace proof_problem_l827_827453

noncomputable def valid_x (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 / 3 ∧ x ≤ 2

theorem proof_problem (x : ℝ) (h : valid_x x) :
  (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ≤ 2 :=
sorry

end proof_problem_l827_827453


namespace intersection_complement_eq_l827_827125

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}
def U : Set ℝ := Set.univ  -- Universal set U is the set of all real numbers

theorem intersection_complement_eq : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_eq_l827_827125


namespace find_k_range_for_two_roots_l827_827522

noncomputable def f (k x : ℝ) : ℝ := (Real.log x / x) - k * x

theorem find_k_range_for_two_roots :
  ∃ k_min k_max : ℝ, k_min = (2 / (Real.exp 4)) ∧ k_max = (1 / (2 * Real.exp 1)) ∧
  ∀ k : ℝ, (k_min ≤ k ∧ k < k_max) ↔
    ∃ x1 x2 : ℝ, 
    (1 / Real.exp 1) ≤ x1 ∧ x1 ≤ Real.exp 2 ∧ 
    (1 / Real.exp 1) ≤ x2 ∧ x2 ≤ Real.exp 2 ∧ 
    f k x1 = 0 ∧ f k x2 = 0 ∧ 
    x1 ≠ x2 :=
sorry

end find_k_range_for_two_roots_l827_827522


namespace distinct_even_three_digit_numbers_count_l827_827534

theorem distinct_even_three_digit_numbers_count :
  let digits := {1, 2, 3, 4, 5}
  ∧ (∀ x ∈ digits, x ∈ {1, 2, 3, 4, 5})
  ∧ (∀ a b c : Nat, a ∈ digits → b ∈ digits → c ∈ digits → (a ≠ b ∧ b ≠ c ∧ a ≠ c))
  ∧ (∃ u ∈ {2, 4}, ∃ h t : Nat, h ∈ digits \ {u} ∧ t ∈ digits \ {u, h}) →
  2 * 3 * 3 = 18 := sorry

end distinct_even_three_digit_numbers_count_l827_827534


namespace relationship_l827_827891

-- Given conditions
def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

-- The theorem to be proven
theorem relationship : c > b ∧ b > a := by
  sorry

end relationship_l827_827891


namespace min_value_range_l827_827899

noncomputable def f (a x : ℝ) := x^2 + a * x

theorem min_value_range (a : ℝ) :
  (∃x : ℝ, ∀y : ℝ, f a (f a x) ≥ f a (f a y)) ∧ (∀x : ℝ, f a x ≥ f a (-a / 2)) →
  a ≤ 0 ∨ a ≥ 2 := sorry

end min_value_range_l827_827899


namespace measure_of_angle_C_area_of_triangle_l827_827047

variables {A B C : ℝ}
variables {a b c : ℝ} -- sides opposite to angles A, B, and C respectively

-- Definition of given condition
def condition_eq (a c A B C : ℝ) : Prop :=
  (a - c) * (Real.sin A + Real.sin C) = (a - b) * Real.sin B

-- The measure of angle C is 60 degrees
theorem measure_of_angle_C (h : condition_eq a c A B C) : C = Real.pi / 3 :=
sorry

-- Given a = 5 and c = 7, the area of the triangle is 10 sqrt 3
theorem area_of_triangle (h : condition_eq 5 7 A B C) (hC : C = Real.pi / 3) : 
  let a := 5 in let c := 7 in 
  let b := 8 in -- Derived from the quadratic equation solution in the problem
  (1 / 2) * a * b * Real.sin C = 10 * Real.sqrt 3 :=
sorry

end measure_of_angle_C_area_of_triangle_l827_827047


namespace sum_of_primes_between_10_and_20_l827_827258

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range (b - a + 1)).map (λ x, x + a).filter is_prime).sum

theorem sum_of_primes_between_10_and_20 : sum_of_primes_between 10 20 = 60 := 
  by 
    -- Definitions used in conditions (e.g., identifying prime numbers, summing them)
    sorry

end sum_of_primes_between_10_and_20_l827_827258


namespace root_at_neg_x0_l827_827034

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom x0_root : ∃ x0, f x0 = Real.exp x0

-- Theorem
theorem root_at_neg_x0 : 
  (∃ x0, (f (-x0) * Real.exp (-x0) + 1 = 0))
  → (∃ x0, (f x0 * Real.exp x0 + 1 = 0)) := 
sorry

end root_at_neg_x0_l827_827034


namespace distinct_real_numbers_sum_l827_827115

theorem distinct_real_numbers_sum:
  ∀ (p q r s : ℝ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    (r + s = 12 * p) →
    (r * s = -13 * q) →
    (p + q = 12 * r) →
    (p * q = -13 * s) →
    p + q + r + s = 2028 :=
by
  intros p q r s h_distinct h1 h2 h3 h4
  sorry

end distinct_real_numbers_sum_l827_827115


namespace future_years_l827_827774

theorem future_years (P A F : ℝ) (Y : ℝ) 
  (h1 : P = 50)
  (h2 : P = 1.25 * A)
  (h3 : P = 5 / 6 * F)
  (h4 : A + 10 + Y = F) : 
  Y = 10 := sorry

end future_years_l827_827774


namespace ladder_cost_l827_827590

theorem ladder_cost (ladders1 ladders2 rung_count1 rung_count2 cost_per_rung : ℕ)
  (h1 : ladders1 = 10) (h2 : ladders2 = 20) (h3 : rung_count1 = 50) (h4 : rung_count2 = 60) (h5 : cost_per_rung = 2) :
  (ladders1 * rung_count1 + ladders2 * rung_count2) * cost_per_rung = 3400 :=
by 
  sorry

end ladder_cost_l827_827590


namespace find_a_value_l827_827042

noncomputable def curve (x : ℝ) : ℝ := x^(-1 / 2)

noncomputable def tangent_slope (a : ℝ) : ℝ := - (1 / 2) * a^(-3 / 2)

noncomputable def y_intercept (a : ℝ) : ℝ := (3 / 2) * a^(- 1 / 2)

noncomputable def x_intercept (a : ℝ) : ℝ := 3 * a

-- Area of the triangle formed by the tangent line and the coordinate axes
noncomputable def triangle_area (a : ℝ) : ℝ := (1 / 2) * (3 * a) * ((3 / 2) * a^(-1 / 2))

theorem find_a_value (a : ℝ) (h : triangle_area a = 18) : a = 64 := by
  sorry

end find_a_value_l827_827042


namespace farmer_crops_remaining_l827_827350

theorem farmer_crops_remaining
  (corn_rows : ℕ) (potato_rows : ℕ) (corn_per_row : ℕ) (potatoes_per_row : ℕ) (half_destroyed : ℚ) :
  corn_rows = 10 →
  potato_rows = 5 →
  corn_per_row = 9 →
  potatoes_per_row = 30 →
  half_destroyed = 1 / 2 →
  (corn_rows * corn_per_row + potato_rows * potatoes_per_row) * (1 - half_destroyed) = 120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end farmer_crops_remaining_l827_827350


namespace originally_anticipated_profit_margin_l827_827342

theorem originally_anticipated_profit_margin (decrease_percent increase_percent : ℝ) (original_price current_price : ℝ) (selling_price : ℝ) :
  decrease_percent = 6.4 → 
  increase_percent = 8 → 
  original_price = 1 → 
  current_price = original_price - original_price * decrease_percent / 100 → 
  selling_price = original_price * (1 + x / 100) → 
  selling_price = current_price * (1 + (x + increase_percent) / 100) →
  x = 117 :=
by
  intros h_dec_perc h_inc_perc h_org_price h_cur_price h_selling_price_orig h_selling_price_cur
  sorry

end originally_anticipated_profit_margin_l827_827342


namespace least_prime_factor_of_expression_l827_827211

theorem least_prime_factor_of_expression : 
  ∀ (p : ℕ), p.prime → (p ∣ (11 ^ 5 - 11 ^ 4)) → (p = 2) :=
sorry

end least_prime_factor_of_expression_l827_827211


namespace value_of_ff1_l827_827545

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 5

theorem value_of_ff1 : f (f 1) = 7 :=
by
  unfold f
  norm_num
  sorry

end value_of_ff1_l827_827545


namespace gauss_polynomial_reciprocal_l827_827639

def gauss_polynomial (k l : ℤ) (x : ℝ) : ℝ := sorry -- Placeholder for actual polynomial definition

theorem gauss_polynomial_reciprocal (k l : ℤ) (x : ℝ) : 
  x^(k * l) * gauss_polynomial k l (1 / x) = gauss_polynomial k l x :=
sorry

end gauss_polynomial_reciprocal_l827_827639


namespace quadratic_other_root_l827_827974

theorem quadratic_other_root (a : ℝ) (h1 : ∃ (x : ℝ), x^2 - 2 * x + a = 0 ∧ x = -1) :
  ∃ (x2 : ℝ), x2^2 - 2 * x2 + a = 0 ∧ x2 = 3 :=
sorry

end quadratic_other_root_l827_827974


namespace probability_of_drawing_red_ball_l827_827057

/-- Define the colors of the balls in the bag -/
def yellow_balls : ℕ := 2
def red_balls : ℕ := 3
def white_balls : ℕ := 5

/-- Define the total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + red_balls + white_balls

/-- Define the probability of drawing exactly one red ball -/
def probability_of_red_ball : ℚ := red_balls / total_balls

/-- The main theorem to prove the given problem -/
theorem probability_of_drawing_red_ball :
  probability_of_red_ball = 3 / 10 :=
by
  -- Calculation steps would go here, but are omitted
  sorry

end probability_of_drawing_red_ball_l827_827057


namespace count_line_segments_l827_827790

theorem count_line_segments (n : ℕ) (h : n ≥ 3) : -- condition specifying a polygon with at least 3 sides
  (∑ i in range n, ∑ j in range n, (i ≠ j ∧ (dist i j ≥ 2))) := -- counting segments, avoiding adjacency
  ∑ i in range n, ∑ j in range n, (i ≠ j ∧ (dist i j ≥ 2)) = (n * (n - 3)) / 2 + n + n * (n - 3) - 2 * n := sorry

end count_line_segments_l827_827790


namespace log_add_property_l827_827416

theorem log_add_property (log : ℝ → ℝ) (h1 : ∀ a b : ℝ, 0 < a → 0 < b → log a + log b = log (a * b)) (h2 : log 10 = 1) :
  log 5 + log 2 = 1 :=
by
  sorry

end log_add_property_l827_827416


namespace area_of_triangle_ACF_l827_827196

-- Define the conditions
variables (A B C D E F : Type)
variables [point : EuclideanSpace ℝ 2]
variables (a b c d e f : point)
variables (AB AC BD : ℝ)
variables (right_triangle : point → point → point → Prop)
variables (rectangle : point → point → point → point → Prop)
variables (units : ℝ)
variables (length_AB : (a.distance b = AB))
variables (length_AC : (a.distance c = AC))
variables (length_BD : (b.distance d = BD))
variables (BC_parallel_CF : parallel (lineBetween b c) (lineBetween c f))
variables (E_on_BD : on_line e (lineBetween b d))
variables (F_above_E : point_above f e)
variables (common_side_AB : on_line a (lineBetween b d))
variables (triangle_ABC : right_triangle a b c)
variables (triangle_ABD : right_triangle a b d)
variables (rectangle_BCEF : rectangle b c e f)

-- Define the goal
theorem area_of_triangle_ACF : 
  (1/2) * (1/2) * (AB * AC) = 24 :=
by sorry

end area_of_triangle_ACF_l827_827196


namespace florist_fertilizer_total_l827_827760

theorem florist_fertilizer_total (f: ℕ) (d: ℕ) (extra: ℕ) (days: ℕ):
  (f = 2) →
  (d = 9) →
  (extra = 4) →
  (days = 10) →
  let total_fertilizer := d * f + (f + extra)
  in total_fertilizer = 24 :=
by
  intros f_eq d_eq extra_eq days_eq
  let total_fertilizer := d * f + (f + extra)
  sorry

end florist_fertilizer_total_l827_827760


namespace find_f_condition_l827_827610

theorem find_f_condition {f : ℂ → ℂ} (h : ∀ z : ℂ, f z + z * f (1 - z) = 1 + z) :
  ∀ z : ℂ, f z = 1 :=
by
  sorry

end find_f_condition_l827_827610


namespace remainder_of_12111_div_3_l827_827738

theorem remainder_of_12111_div_3 : 12111 % 3 = 0 := by
  sorry

end remainder_of_12111_div_3_l827_827738


namespace average_charge_proof_l827_827714

noncomputable def averageChargePerPerson
  (chargeFirstDay : ℝ)
  (chargeSecondDay : ℝ)
  (chargeThirdDay : ℝ)
  (chargeFourthDay : ℝ)
  (ratioFirstDay : ℝ)
  (ratioSecondDay : ℝ)
  (ratioThirdDay : ℝ)
  (ratioFourthDay : ℝ)
  : ℝ :=
  let totalRevenue := ratioFirstDay * chargeFirstDay + ratioSecondDay * chargeSecondDay + ratioThirdDay * chargeThirdDay + ratioFourthDay * chargeFourthDay
  let totalVisitors := ratioFirstDay + ratioSecondDay + ratioThirdDay + ratioFourthDay
  totalRevenue / totalVisitors

theorem average_charge_proof :
  averageChargePerPerson 25 15 7.5 2.5 3 7 11 19 = 7.75 := by
  simp [averageChargePerPerson]
  sorry

end average_charge_proof_l827_827714


namespace domain_of_sqrt_log_function_l827_827659

open Real

theorem domain_of_sqrt_log_function :
  {x : ℝ | ∃ y, y = sqrt (log (0.5) (4 * x^2 - 3 * x))} =
  Icc (-1 / 4 : ℝ) 0 ∪ Icc (3 / 4 : ℝ) 1 :=
by
  -- Assuming we have omitted the proof for now.
  sorry

end domain_of_sqrt_log_function_l827_827659


namespace sum_primes_between_10_and_20_l827_827248

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827248


namespace gcd_lcm_problem_l827_827668

theorem gcd_lcm_problem (b : ℤ) (x : ℕ) (hx_pos : 0 < x) (hx : x = 12) :
  gcd 30 b = x + 3 ∧ lcm 30 b = x * (x + 3) → b = 90 := 
by
  sorry

end gcd_lcm_problem_l827_827668


namespace total_time_in_cocoons_l827_827179

theorem total_time_in_cocoons (CA CB CC: ℝ) 
    (h1: 4 * CA = 90)
    (h2: 4 * CB = 120)
    (h3: 4 * CC = 150) 
    : CA + CB + CC = 90 := 
by
  -- To be proved
  sorry

end total_time_in_cocoons_l827_827179


namespace pedro_furniture_area_l827_827632

theorem pedro_furniture_area :
  let width : ℝ := 2
  let length : ℝ := 2.5
  let door_arc_area := (1 / 4) * Real.pi * (0.5 ^ 2)
  let window_arc_area := 2 * (1 / 2) * Real.pi * (0.5 ^ 2)
  let room_area := width * length
  room_area - door_arc_area - window_arc_area = (80 - 9 * Real.pi) / 16 := 
by
  sorry

end pedro_furniture_area_l827_827632


namespace library_visitor_ratio_l827_827594

theorem library_visitor_ratio (T : ℕ) (h1 : 50 + T + 20 * 4 = 250) : T / 50 = 2 :=
by
  sorry

end library_visitor_ratio_l827_827594


namespace sarah_trips_to_fill_tank_l827_827143

noncomputable def volume_hemisphere (r_b : ℝ) : ℝ :=
  (2 / 3) * π * r_b^3

noncomputable def volume_cylinder (r_t h_t : ℝ) : ℝ :=
  π * r_t^2 * h_t

noncomputable def number_of_trips (volume_tank volume_bucket : ℝ) : ℝ :=
  (volume_tank / volume_bucket).ceil

theorem sarah_trips_to_fill_tank : 
  let r_t := 8
  let h_t := 24
  let r_b := 5
  number_of_trips (volume_cylinder r_t h_t) (volume_hemisphere r_b) = 19 := by
  sorry

end sarah_trips_to_fill_tank_l827_827143


namespace sum_primes_between_10_and_20_is_60_l827_827232

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827232


namespace sum_of_primes_between_10_and_20_l827_827306

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827306


namespace loss_percentage_is_correct_l827_827039

def purchase_price := 490
def selling_price := 465.5
def loss_amount := purchase_price - selling_price
def loss_percentage := (loss_amount / purchase_price) * 100

theorem loss_percentage_is_correct : loss_percentage = 5 := 
by 
  -- {yuould typically provide the arithmetic steps here in a complete proof, but we'll use sorry} 
  sorry

end loss_percentage_is_correct_l827_827039


namespace sphere_section_area_l827_827512

theorem sphere_section_area :
  let edge_length := 4
  let midpoint_to_center := 2 * Real.sqrt 2
  let sphere_radius := 2 * Real.sqrt 3
  let cross_section_radius := sqrt ((2 * Real.sqrt 3) ^ 2 - (2 * Real.sqrt 2) ^ 2)
  cross_section_radius = 2 → 
  (π * cross_section_radius^2) = 4 * π := 
by
  intros edge_length midpoint_to_center sphere_radius cross_section_radius h_c;
  sorry

end sphere_section_area_l827_827512


namespace problem_statement_l827_827881

noncomputable def a : ℝ := 31 / 32
noncomputable def b : ℝ := Real.cos (1 / 4)
noncomputable def c : ℝ := 4 * Real.sin (1 / 4)

theorem problem_statement : c > b ∧ b > a := 
by {
  sorry
}

end problem_statement_l827_827881


namespace part_one_part_two_l827_827502

noncomputable theory

open Real

def M : Point := ⟨1, 1⟩

def ellipse (x y : ℝ) (b : ℝ) : Prop := (x^2) / 4 + (y^2) / (b^2) = 1

def line (x y t : ℝ) : Prop := x + t * y - 1 = 0

def intersects (f g : ℝ → ℝ → ℝ) (x y : ℝ) : Prop := f x y = 0 ∧ g x y = 0

theorem part_one (b : ℝ) (hb : 0 < b ∧ b < 2) (hM : ellipse 1 1 b)
  (hx : ∀ x y, intersects (λ x y, (x^2) / 4 + (y^2) / (b^2) - 1) (λ x y, x + y - 1) x y)
  : Area (Triangle A B M) = sqrt 13 / 4 := sorry

theorem part_two (b : ℝ) (hb : 0 < b ∧ b < 2) (hM : ellipse 1 1 b)
  (ht : S_triangle PQM = 5 * S_triangle ABM)
  : t = 3 * sqrt 2 / 2 ∨ t = - (3 * sqrt 2 / 2) := sorry

end part_one_part_two_l827_827502


namespace driving_distance_l827_827626

def miles_per_gallon : ℕ := 20
def gallons_of_gas : ℕ := 5

theorem driving_distance :
  miles_per_gallon * gallons_of_gas = 100 :=
  sorry

end driving_distance_l827_827626


namespace sum_primes_between_10_and_20_l827_827241

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers between 10 and 20
def primes_between_10_and_20 : list ℕ := [11, 13, 17, 19]

-- Prove the sum of these prime numbers is 60
theorem sum_primes_between_10_and_20 : primes_between_10_and_20.sum = 60 := by
  sorry

end sum_primes_between_10_and_20_l827_827241


namespace smallest_x_for_multiple_of_625_l827_827217

theorem smallest_x_for_multiple_of_625 (x : ℕ) (hx_pos : 0 < x) : (500 * x) % 625 = 0 → x = 5 :=
by
  sorry

end smallest_x_for_multiple_of_625_l827_827217


namespace part_I_part_II_part_III_l827_827516

noncomputable def f (x : ℝ) : ℝ := ln x - (x - 1)^2 / 2

theorem part_I (h : x ≠ 0) : 0 < x ∧ x < (1 + Real.sqrt 5) / 2 → f x > f (x - ε) :=
sorry

theorem part_II (x : ℝ) (h : 1 < x) : f x < x - 1 :=
sorry

theorem part_III (k : ℝ) : k < 1 →
  (∃ x0 > 1, ∀ x ∈ set.Ioo 1 x0, f x > k * (x - 1)) :=
sorry

end part_I_part_II_part_III_l827_827516


namespace sum_primes_10_to_20_l827_827293

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827293


namespace problem1_problem2_l827_827744

-- Problem 1
-- Simplify S_n given a != 0 and n ∈ ℕ*
theorem problem1 {a : ℝ} {n : ℕ} (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : n > 0) :
  (∑ i in finset.range n, (i + 1) * a^i) = (1 - a^n) / (1 - a)^2 - (n * a^n) / (1 - a) :=
sorry

-- Problem 2
-- Find the sum of the first n terms of the sequence given the provided conditions
theorem problem2 {n : ℕ} (h1 : n > 0) :
  (∑ i in finset.range n, 1 / ((i + 1) * (i + 2))) = n / (n + 1) :=
sorry

end problem1_problem2_l827_827744


namespace sum_convergent_l827_827419

theorem sum_convergent :
  ∃ S : ℝ, ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (∑ j in finset.range n, ∑ k in finset.range (j + 1), 3^(- 2 * k - j - (k + j)^2) - S) < ε :=
sorry

end sum_convergent_l827_827419


namespace tom_age_ratio_l827_827700

theorem tom_age_ratio (T : ℕ) (h1 : T = 3 * (3 : ℕ)) (h2 : T - 5 = 3 * ((T / 3) - 10)) : T / 5 = 9 := 
by
  sorry

end tom_age_ratio_l827_827700


namespace amy_final_money_l827_827384

theorem amy_final_money :
  let initial_money := 2
  let chore_payment := 5 * 13
  let birthday_gift := 3
  let toy_cost := 12
  let remaining_money := initial_money + chore_payment + birthday_gift - toy_cost
  let grandparents_reward := 2 * remaining_money
  remaining_money + grandparents_reward = 174 := 
by
  sorry

end amy_final_money_l827_827384


namespace sum_primes_10_20_l827_827219

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827219


namespace sum_of_arithmetic_progression_l827_827976

theorem sum_of_arithmetic_progression 
  (a d : ℚ) 
  (S : ℕ → ℚ)
  (h_sum_15 : S 15 = 150)
  (h_sum_75 : S 75 = 30)
  (h_arith_sum : ∀ n, S n = (n / 2) * (2 * a + (n - 1) * d)) :
  S 90 = -180 :=
by
  sorry

end sum_of_arithmetic_progression_l827_827976


namespace problem_statement_l827_827880

noncomputable def a : ℝ := 31 / 32
noncomputable def b : ℝ := Real.cos (1 / 4)
noncomputable def c : ℝ := 4 * Real.sin (1 / 4)

theorem problem_statement : c > b ∧ b > a := 
by {
  sorry
}

end problem_statement_l827_827880


namespace sum_primes_10_to_20_l827_827290

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827290


namespace rubber_band_problem_l827_827792

noncomputable def a : ℤ := 4
noncomputable def b : ℤ := 12
noncomputable def c : ℤ := 3
noncomputable def band_length := a * Real.pi + b * Real.sqrt c

theorem rubber_band_problem (r1 r2 d : ℝ) (h1 : r1 = 3) (h2 : r2 = 9) (h3 : d = 12) :
  let a := 4
  let b := 12
  let c := 3
  let band_length := a * Real.pi + b * Real.sqrt c
  a + b + c = 19 :=
by
  sorry

end rubber_band_problem_l827_827792


namespace find_n_times_s_l827_827611

-- Definition of the set of positive real numbers
def S := { x : ℝ | 0 < x }

-- Function g mapping from S to ℝ
variable (g : S → ℝ)

-- Condition given in the problem
axiom g_condition (x y : S) : g x * g y = g (x * y) + 2010 * (1 / x + 1 / y + 2009)

-- Lean statement to prove
theorem find_n_times_s : let g_2 := (g (2 : S)) in
  let n := 1 in
  let s := (1 / 2 + 2010 : ℝ) in
  n * s = 4021 / 2 :=
by
  sorry

end find_n_times_s_l827_827611


namespace sum_primes_10_20_l827_827225

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827225


namespace man_speed_against_current_proof_l827_827780

def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

theorem man_speed_against_current_proof 
  (V_m : ℝ) 
  (h_with_current : V_m + speed_of_current = man_speed_with_current) :
  V_m - speed_of_current = man_speed_against_current := 
by 
  sorry

end man_speed_against_current_proof_l827_827780


namespace find_sum_placed_on_simple_interest_l827_827736

open BigOperators

def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * ((1 + r)^t - 1)

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  (P * r * t) / 100

theorem find_sum_placed_on_simple_interest : 
  ∃ (P : ℝ), 
  let C := compound_interest 4000 0.10 2 in
  let S := simple_interest P 4 2 in
  S = C / 2 ∧ P = 5250 :=
by
  sorry

end find_sum_placed_on_simple_interest_l827_827736


namespace max_third_altitude_l827_827193

theorem max_third_altitude (h1 h2 : ℕ) (h1_eq : h1 = 6) (h2_eq : h2 = 18) (triangle_scalene : true)
: (exists h3 : ℕ, (∀ h3_alt > h3, h3_alt > 8)) := 
sorry

end max_third_altitude_l827_827193


namespace sum_first_n_abs_terms_arithmetic_seq_l827_827490

noncomputable def sum_abs_arithmetic_sequence (n : ℕ) (h : n ≥ 3) : ℚ :=
  if n = 1 ∨ n = 2 then (n * (4 + 7 - 3 * n)) / 2
  else (3 * n^2 - 11 * n + 20) / 2

theorem sum_first_n_abs_terms_arithmetic_seq (n : ℕ) (h : n ≥ 3) :
  sum_abs_arithmetic_sequence n h = (3 * n^2) / 2 - (11 * n) / 2 + 10 :=
sorry

end sum_first_n_abs_terms_arithmetic_seq_l827_827490


namespace satisfies_inequality_l827_827859

theorem satisfies_inequality (x : ℝ) :
  (x ≠ 0 → (x^2 - 2*x^3 + 3*x^4) / (x - 2*x^2 + 3*x^3) ≥ -1) ↔ (x ∈ set.Ico (-1 : ℝ) 0 ∨ x ∈ set.Ioi 0) :=
by
  sorry

end satisfies_inequality_l827_827859


namespace distance_rowed_downstream_l827_827773

-- Define the conditions
def speed_in_still_water (b s: ℝ) := b - s = 60 / 4
def speed_of_stream (s: ℝ) := s = 3
def time_downstream (t: ℝ) := t = 4

-- Define the function that computes the downstream speed
def downstream_speed (b s t: ℝ) := (b + s) * t

-- The theorem we want to prove
theorem distance_rowed_downstream (b s t : ℝ) 
    (h1 : speed_in_still_water b s)
    (h2 : speed_of_stream s)
    (h3 : time_downstream t) : 
    downstream_speed b s t = 84 := by
    sorry

end distance_rowed_downstream_l827_827773


namespace circle_center_l827_827861

theorem circle_center :
  ∃ c : ℝ × ℝ, c = (-1, 3) ∧ ∀ (x y : ℝ), (4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 96 = 0 ↔ (x + 1)^2 + (y - 3)^2 = 14) :=
by
  sorry

end circle_center_l827_827861


namespace mans_speed_against_current_l827_827782

variable (V_downstream V_current : ℝ)
variable (V_downstream_eq : V_downstream = 15)
variable (V_current_eq : V_current = 2.5)

theorem mans_speed_against_current : V_downstream - 2 * V_current = 10 :=
by
  rw [V_downstream_eq, V_current_eq]
  exact (15 - 2 * 2.5)

end mans_speed_against_current_l827_827782


namespace roots_of_equation_l827_827685

theorem roots_of_equation (x : ℝ) (h : 2 * real.sqrt x + 2 * x^(-1 / 2) = 5) : 
  4 * x^2 - 17 * x + 4 = 0 :=
sorry

end roots_of_equation_l827_827685


namespace reciprocals_directly_proportional_l827_827199

variables {x y k : ℝ}

theorem reciprocals_directly_proportional (h : y = k * x) : ∃ c : ℝ, c ≠ 0 ∧ (1 / y) = c * (1 / x) :=
by 
  use (1 / k)
  split
  { 
    intro h1,
    have : k ≠ 0 := by sorry, -- This follows from the definition of direct proportionality
    contradiction
  }
  { 
    calc 
      1 / y = 1 / (k * x) : by rw h
          ... = (1 / k) * (1 / x) : by sorry
  }

end reciprocals_directly_proportional_l827_827199


namespace probability_of_valid_triangle_l827_827447

open ProbabilityTheory

noncomputable def stick_lengths : List ℕ := [3, 4, 5, 8, 10, 12, 15, 18]

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ (a ≥ 10 ∨ b ≥ 10 ∨ c ≥ 10)

def count_valid_sets : ℕ :=
  stick_lengths.combinations 3 |>.filter (λ l, match l.sorted with
    | [a, b, c] => valid_triangle a b c
    | _ => false
  end) |>.length

def total_combinations : ℕ := (Nat.choose 8 3)

def probability_valid_triangle : ℚ :=
  count_valid_sets / total_combinations

theorem probability_of_valid_triangle : probability_valid_triangle = 11 / 56 := by
  sorry

end probability_of_valid_triangle_l827_827447


namespace river_speed_max_l827_827816

noncomputable def max_river_speed (x y : ℝ) (hx : x < y) : ℝ :=
  let v := 26 in
  v

theorem river_speed_max (x y : ℝ) (hx : x < y) :
  ∀ (v : ℝ), 
  (v ≥ 6 → ((y / 6) < (x / 11 + (x + y) / v)) → (v ≤ 26)) :=
begin
  intros v hv hineq,
  sorry, -- Proof goes here
end

example : max_river_speed x y hx = 26 := by
  rfl

end river_speed_max_l827_827816


namespace star_eq_122_l827_827963

noncomputable def solveForStar (star : ℕ) : Prop :=
  45 - (28 - (37 - (15 - star))) = 56

theorem star_eq_122 : solveForStar 122 :=
by
  -- proof
  sorry

end star_eq_122_l827_827963


namespace isosceles_triangle_max_area_l827_827629

theorem isosceles_triangle_max_area {a b c : ℝ} (ha: a > 0) (hb: b > 0) (hc: c > 0) :
  let A := (a + b + c) / 2 in 
  ∀ (α β γ: ℝ), α + β + γ = a + b + c → b = c → 
  (1/2) * b * sqrt (A * (A - b) * (A - α) * (A - γ)) >
  (1/2) * α * sqrt (A * (A - α) * (A - b) * (A - β)) :=
sorry

end isosceles_triangle_max_area_l827_827629


namespace sqrt_neg4_squared_l827_827417

theorem sqrt_neg4_squared : Real.sqrt ((-4 : ℝ) ^ 2) = 4 := 
by 
-- add proof here
sorry

end sqrt_neg4_squared_l827_827417


namespace central_projection_intersect_l827_827822

def central_projection (lines : Set (Set Point)) : Prop :=
  ∃ point : Point, ∀ line ∈ lines, line (point)

theorem central_projection_intersect :
  ∀ lines : Set (Set Point), central_projection lines → ∃ point : Point, ∀ line ∈ lines, line (point) :=
by
  sorry

end central_projection_intersect_l827_827822


namespace investment_present_value_approx_l827_827539

theorem investment_present_value_approx :
  let F : ℝ := 600000
  let r : ℝ := 0.04 / 2
  let n : ℕ := 12 * 2
  let P : ℝ := F / (1 + r) ^ n
  abs (P - 374811.16) < 1.00 :=
by
  -- Given condition
  have h_F : F = 600000 := rfl
  have h_r : r = 0.02 := by norm_num
  have h_n : n = 24 := by norm_num
  -- Calculation for P
  have h_P : P = F / (1 + r) ^ n :=
    by simp [h_F, h_r, h_n, Real.div_eq_mul_inv, Real.mul_inv_rev]
  sorry -- To be completed with an approximation proof

end investment_present_value_approx_l827_827539


namespace rectangular_eq_of_curve_polar_coord_intersection_l827_827526

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  let x := 1 + (1 / 2) * t in
  let y := Real.sqrt 3 + Real.sqrt 3 * t in
  (x, y)

def polar_curve (θ ρ : ℝ) : Prop :=
  Real.sin θ - Real.sqrt 3 * ρ * Real.cos θ^2 = 0

theorem rectangular_eq_of_curve :
  ∀ (x y : ℝ), (∃ ρ θ, ρ * Real.sin θ = y ∧ ρ * Real.cos θ = x ∧ polar_curve θ ρ) ↔ y = Real.sqrt 3 * x^2 := sorry

theorem polar_coord_intersection :
  ∃ t, parametric_line t = (1, Real.sqrt 3) ∧
       ρ = 2 ∧ θ = Real.pi / 3 := sorry
  where ρ = 2 and θ = Real.pi / 3 


end rectangular_eq_of_curve_polar_coord_intersection_l827_827526


namespace count_4_letter_words_with_E_l827_827024

theorem count_4_letter_words_with_E :
  let letters := {'A', 'B', 'C', 'D', 'E'}
  let total_4_letter_words := (letters.card) ^ 4
  let words_without_E := (letters.erase 'E').card ^ 4
  total_4_letter_words - words_without_E = 369 := by
  sorry

end count_4_letter_words_with_E_l827_827024


namespace h1_even_h2_neither_l827_827483

noncomputable def f (x : ℝ) : ℝ := x / (2 ^ x - 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
noncomputable def h1 (x : ℝ) : ℝ := f(x) + g(x)
noncomputable def h2 (x : ℝ) : ℝ := f(x) * g(x)

theorem h1_even : ∀ x : ℝ, h1(-x) = h1(x) :=
by
  intro x
  sorry

theorem h2_neither : ¬ (∀ x : ℝ, h2(-x) = h2(x)) ∧ ¬ (∀ x : ℝ, h2(-x) = -h2(x)) :=
by
  sorry

end h1_even_h2_neither_l827_827483


namespace symmetric_point_sum_l827_827923

theorem symmetric_point_sum :
  ∀ (a b : ℝ), (2 = b) ∧ (a = 3) → a + b = 5 :=
by
  intros a b h,
  cases h with hb ha,
  rw [hb, ha],
  exact rfl

end symmetric_point_sum_l827_827923


namespace most_probable_1s_in_500rolls_l827_827964

noncomputable theory

theorem most_probable_1s_in_500rolls (P : ℕ → ℝ)
  (hP : ∀ r, P r = Nat.choose 500 r * ((1/6 : ℝ)^r) * ((5/6 : ℝ)^(500 - r))) :
  ∃ r, r = 83 ∧ ∀ k ≠ 83, P 83 > P k :=
sorry

end most_probable_1s_in_500rolls_l827_827964


namespace points_concyclic_l827_827613

variables {A B C I K E F D : Type*}
variables [EuclideanGeometry A B C I K E F D]

-- Definitions reflecting conditions from the problem
def incenter (I : Type*) (triangle : Triangle A B C) : Prop := -- incenter definition
sorry

def meet (line1 : Line BI) (line2 : Line EF) (K : Type*) : Prop := -- two lines meet at K
sorry

def concyclic (points : Set Type*) : Prop := -- concyclicity definition
sorry

-- Problem statement
theorem points_concyclic {A B C I K E F D : Type*}
  (h1 : incenter I (Triangle A B C))
  (h2 : meet (Line.segment A B I) (Line.segment E F) K) :
  concyclic {I, K, E, C, D} :=
sorry

end points_concyclic_l827_827613


namespace area_of_quadrilateral_l827_827570

theorem area_of_quadrilateral (J K L M P Q R S W X Y Z : Type)
  [is_square : square JK LM 4]
  [trisect_JK : trisect JK P Q]
  [trisect_LM : trisect LM R S]
  [midpoints : midpoints (JP QR LS SM) W X Y Z] : 
  area WXYZ = 4 / 9 :=
sorry

end area_of_quadrilateral_l827_827570


namespace smallest_positive_omega_l827_827935

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * (x + Real.pi / 4) - Real.pi / 6)

theorem smallest_positive_omega (ω : ℝ) :
  (∀ x : ℝ, g (ω) x = g (ω) (-x)) → (ω = 4 / 3) := sorry

end smallest_positive_omega_l827_827935


namespace eccentricity_range_l827_827677

variables {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (c_def : c = Real.sqrt (a^2 - b^2))

def ellipse_equation (x y: ℝ) : Prop := (x^2 / a^2 + y^2 / b^2) = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

theorem eccentricity_range (e : ℝ) :
  (∀ x y : ℝ, ellipse_equation a b x y →
    let F1 := (-c, 0)
    let F2 := (c, 0)
    let P := (x, y)
    let PF1 := (F1.1 - x, F1.2 - y)
    let PF2 := (F2.1 - x, F2.2 - y)
    let dot_product := PF1.1 * PF2.1 + PF1.2 * PF2.2
    dot_product ≤ 3 * c^2 ∧ 
    dot_product ≥ c^2) →
  (e = eccentricity a b) →
  e ∈ Set.Icc (1 / 2) (Real.sqrt 2 / 2) :=
by
  intros
  unfold ellipse_equation
  unfold eccentricity
  sorry

end eccentricity_range_l827_827677


namespace amount_left_correct_l827_827130

-- Define all the conditions
def initial_amount := 200.50
def spent_on_sweets := 35.25
def spent_on_stickers := 10.75
def amount_per_friend := 25.20
def number_of_friends := 4
def donated_to_charity := 15.30
def amount_sent_abroad := 20.00
def exchange_rate_fee := 0.10 * amount_sent_abroad

-- Calculate the total expenses
def total_given_to_friends := number_of_friends * amount_per_friend
def total_transaction_abroad := amount_sent_abroad + exchange_rate_fee
def total_expenses := spent_on_sweets + spent_on_stickers + total_given_to_friends + donated_to_charity + total_transaction_abroad

-- Calculate the amount left
def amount_left := initial_amount - total_expenses

theorem amount_left_correct : 
  amount_left = 16.40 := 
  by
    have total_given_to_friends := number_of_friends * amount_per_friend
    have exchange_rate_fee_val := 0.10 * amount_sent_abroad
    have total_transaction_abroad := amount_sent_abroad + exchange_rate_fee_val
    have total_expenses_calculated := spent_on_sweets + spent_on_stickers + total_given_to_friends + donated_to_charity + total_transaction_abroad
    have initial_amount_val := initial_amount
    have amount_left_calculated := initial_amount_val - total_expenses_calculated
    show amount_left = 16.40 from sorry

end amount_left_correct_l827_827130


namespace roots_imply_value_l827_827099

noncomputable def value_of_expression (a b c : ℝ) : ℝ :=
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)

theorem roots_imply_value {a b c : ℝ} 
  (h1 : a + b + c = 15) 
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 10) 
  : value_of_expression a b c = 175 / 11 :=
sorry

end roots_imply_value_l827_827099


namespace least_positive_integer_n_l827_827458

noncomputable def sin_inverse_sum : ℝ :=
  (∑ k in (finset.range 118).map (λ x, 30 + x),
    1 / (Real.sin (k * (Real.pi / 180)) * Real.sin ((k + 1) * (Real.pi / 180))))

theorem least_positive_integer_n : ∃ n : ℕ, 
  (1 ≤ n) ∧ (sin_inverse_sum = 1 / (Real.sin (n * (Real.pi / 180)))) ∧ (∀ m : ℕ, (1 ≤ m) → (sin_inverse_sum = 1 / (Real.sin (m * (Real.pi / 180))) → m = 1)) :=
by 
  sorry

end least_positive_integer_n_l827_827458


namespace max_red_tiles_l827_827374

theorem max_red_tiles (n : ℕ) (color : ℕ → ℕ → color) :
    (∀ i j, color i j ≠ color (i + 1) j ∧ color i j ≠ color i (j + 1) ∧ color i j ≠ color (i + 1) (j + 1) 
           ∧ color i j ≠ color (i - 1) j ∧ color i j ≠ color i (j - 1) ∧ color i j ≠ color (i - 1) (j - 1)) 
    → ∃ m ≤ 2500, ∀ i j, (color i j = red ↔ i * n + j < m) :=
sorry

end max_red_tiles_l827_827374


namespace relationship_l827_827890

-- Given conditions
def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

-- The theorem to be proven
theorem relationship : c > b ∧ b > a := by
  sorry

end relationship_l827_827890


namespace problem_statement_l827_827404

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l827_827404


namespace length_of_BC_l827_827566

/-- Given a triangle ABC with AB = 3, AC = 4, and ∠A = 60°, prove that BC = √13 -/
theorem length_of_BC (A B C : Type) [inner_product_space ℝ A] 
  (AB AC : A → ℝ) (cos_A : real.cos (real.pi / 3) = 1 / 2) :
  AB (vector.sub B A) = 3 → AC (vector.sub C A) = 4 → 
  BC (vector.sub B C) = real.sqrt 13 :=
by sorry

end length_of_BC_l827_827566


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_l827_827205

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4 : 
  Nat.minFac (11^5 - 11^4) = 2 := 
by sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_l827_827205


namespace find_n_l827_827715

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n < 59) (h₂ : 58 * n ≡ 20 [ZMOD 59]) : n = 39 :=
  sorry

end find_n_l827_827715


namespace derek_saves_money_l827_827844

theorem derek_saves_money : 
  let a : ℕ := 2
  let r : ℕ := 2
  let n : ℕ := 12
  S_n a r n = 8190
  where S_n (a : ℕ) (r : ℕ) (n : ℕ) : ℕ := a * (1 - r^n) / (1 - r) :=
  sorry

end derek_saves_money_l827_827844


namespace distance_from_O_to_AB_constant_l827_827949

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line via two points
structure Line where
  p1 : Point
  p2 : Point

-- Define the conditions and state the theorem
theorem distance_from_O_to_AB_constant (O A B : Point) (a b : Line) :
  (non_intersecting a b) → (center_of_symmetry O) → (rotating_right_angle O A B) → 
  (distance_from_point_to_line O (line_through_points A B)) = constant := 
by
  sorry

-- The actual definitions of the helper functions used above would be needed in Lean code.
-- For example:
-- non_intersecting: checks if two lines do not intersect
-- center_of_symmetry: checks if a point is the center of symmetry for two lines
-- rotating_right_angle: checks if the points form a right angle rotating around a vertex
-- distance_from_point_to_line: calculates the distance of a point to a line
-- constant: represents that the distance remains the same

end distance_from_O_to_AB_constant_l827_827949


namespace star_intersections_l827_827139

theorem star_intersections (n k : ℕ) (h_coprime : Nat.gcd n k = 1) (h_n_ge_5 : 5 ≤ n) (h_k_lt_n_div_2 : k < n / 2) :
    k = 25 → n = 2018 → n * (k - 1) = 48432 := by
  intros
  sorry

end star_intersections_l827_827139


namespace handball_tournament_theorem_l827_827559

-- Define the problem and conditions
noncomputable def handball_tournament (teams : ℕ) (matches : ℕ) : Prop :=
  -- Number of teams
  teams = 20 ∧
  -- Each team plays every other team twice
  matches = Nat.choose 20 2 * 2 ∧
  -- Initial conditions described in the problem
  ∀ pts1 pts2 : ℕ → ℕ, 
    -- First round distinct points for all teams
    (∀ i j, i ≠ j → pts1 i ≠ pts1 j) ∧
    -- Second round equal points for all teams
    (∀ i, pts2 i = 38) →
    -- Prove that there exists a pair of teams (i, j) such that:
    -- each team won exactly once against the other
    ∃ i j, i ≠ j ∧ 
    (win1 i j ∧ win1 j i)

-- Use hypotheses from previous rounds to describe win conditions
def win1 (i j : ℕ) : Prop := sorry -- win condition for the first round

-- The main theorem translates our equivalent problem to Lean 4
theorem handball_tournament_theorem : handball_tournament 20 380 :=
by
  sorry

end handball_tournament_theorem_l827_827559


namespace variance_of_given_data_set_is_0_l827_827927

def data_set : List ℝ := [4.7, 4.8, 5.1, 5.4, 5.5]

def mean (l : List ℝ) : ℝ :=
  (l.sum / l.length)

def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / l.length

theorem variance_of_given_data_set_is_0.1 :
  variance data_set = 0.1 :=
by
  sorry

end variance_of_given_data_set_is_0_l827_827927


namespace arithmetic_sequence_common_difference_l827_827053

theorem arithmetic_sequence_common_difference :
  let a := 5
  let a_n := 50
  let S_n := 330
  exists (d n : ℤ), (a + (n - 1) * d = a_n) ∧ (n * (a + a_n) / 2 = S_n) ∧ (d = 45 / 11) :=
by
  let a := 5
  let a_n := 50
  let S_n := 330
  use 45 / 11, 12
  sorry

end arithmetic_sequence_common_difference_l827_827053


namespace complete_the_square_l827_827201

theorem complete_the_square (x : ℝ) (h : x^2 + 7 * x - 5 = 0) : (x + 7 / 2) ^ 2 = 69 / 4 :=
sorry

end complete_the_square_l827_827201


namespace eccentricity_of_ellipse_l827_827046

-- Define the variables and constants used in the problem
variable (A B C : Point) -- A, B, and C are points in the plane
variable (angle_A : ℝ) -- ∠A in degrees
variable (AB AC BC : ℝ) -- Side lengths in triangle ABC
variable (area_ABC : ℝ) -- Area of triangle ABC
variable (e : ℝ) -- Eccentricity of the ellipse

-- Declare the conditions given in the problem
axiom angle_A_30 : angle_A = 30
axiom AB_2 : AB = 2
axiom area_ABC_sqrt3 : area_ABC = Real.sqrt 3

-- Define the focal property of the ellipse and distance relations
axiom ellipse_foci_conditions : 
  ∃ AC BC : ℝ, 
    area_ABC = 0.5 * AB * AC * Real.sin (angle_A * Real.pi / 180) ∧
    BC^2 = AB^2 + AC^2 - 2 * AB * AC * Real.cos (angle_A * Real.pi / 180) ∧
    2 * e = (2 / (2 * Real.sqrt(3) + 2))

-- The desired proof statement that the eccentricity is equal to the given value
theorem eccentricity_of_ellipse : 
  e = (Real.sqrt 3 - 1) / 2 :=
by 
  -- Begin the proof here
  sorry

end eccentricity_of_ellipse_l827_827046


namespace relationship_among_abc_l827_827475

noncomputable def a : ℝ := 0.7 ^ 2.1
noncomputable def b : ℝ := 0.7 ^ 2.5
noncomputable def c : ℝ := 2.1 ^ 0.7

theorem relationship_among_abc : b < a ∧ a < c :=
by
  sorry

end relationship_among_abc_l827_827475


namespace integral_exp_neg_l827_827456

theorem integral_exp_neg : ∫ x in (Set.Ioi 0), Real.exp (-x) = 1 := sorry

end integral_exp_neg_l827_827456


namespace sequence_count_even_odd_l827_827818

/-- The number of 8-digit sequences such that no two adjacent digits have the same parity
    and the sequence starts with an even number. -/
theorem sequence_count_even_odd : 
  let choices_for_even := 5
  let choices_for_odd := 5
  let total_positions := 8
  (choices_for_even * (choices_for_odd * choices_for_even) ^ (total_positions / 2 - 1)) = 390625 :=
by
  sorry

end sequence_count_even_odd_l827_827818


namespace area_of_triangle_l827_827567

theorem area_of_triangle
  (a b c : ℝ) -- side lengths
  (A B C : ℝ) -- angles opposite to a, b, c respectively
  (h1 : ¬(A = π / 2 ∨ B = π / 2 ∨ C = π / 2)) -- not a right triangle
  (h2 : sin A * sin A + sin B * sin B - sin C * sin C = a * b * sin A * sin B * sin (2 * C)) -- given condition
  (h3 : sin A = a / c)
  (h4 : sin B = b / c)
  (h5 : sin C = c / c)
  (h6 : (sin (2 * C)) = 2 * sin C * cos C) :
  1/2 = 1/2 * a * b * sin C := 
by 
  sorry -- proof to be filled in later

end area_of_triangle_l827_827567


namespace max_abs_sum_sin_double_l827_827463

theorem max_abs_sum_sin_double 
(n : ℕ) (a : ℝ) (x : ℕ → ℝ) 
(h1 : 0 ≤ a) (h2 : a ≤ n) 
(h3 : ∑ i in finset.range n, real.sin (x i) ^ 2 = a) 
: ∃ b : ℝ, (b = 2 * real.sqrt (a * (n - a))) ∧ (| ∑ i in finset.range n, real.sin (2 * x i) | ≤ b) :=
by sorry

end max_abs_sum_sin_double_l827_827463


namespace parallel_lines_perpendicular_to_same_plane_l827_827924

variables {Plane Line : Type} [HasPerp Line Plane] [HasParallel Line Line]

-- Given conditions
variables (α β γ : Plane) (a b : Line)

-- Assume the perpendicular conditions
axiom perp_a_α : a ⟂ α
axiom perp_b_α : b ⟂ α

-- Proof goal
theorem parallel_lines_perpendicular_to_same_plane :
  a ⟂ α → b ⟂ α → (a ∥ b) :=
by
  intro perp_a perp_b
  sorry

end parallel_lines_perpendicular_to_same_plane_l827_827924


namespace sum_primes_between_10_and_20_is_60_l827_827236

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827236


namespace arithmetic_progression_contains_geometric_progression_l827_827664

theorem arithmetic_progression_contains_geometric_progression :
  ∃ a d: ℕ, d > 0 ∧ ∃ (a_0 ... a_2014: ℕ), 
    (∀ k, 0 ≤ k ≤ 2014 → a_0 + k * d = a ^ k) :=
begin
  sorry
end

end arithmetic_progression_contains_geometric_progression_l827_827664


namespace quotient_of_N_div_3_l827_827173

-- Define the number N
def N : ℕ := 7 * 12 + 4

-- Statement we need to prove
theorem quotient_of_N_div_3 : N / 3 = 29 :=
by
  sorry

end quotient_of_N_div_3_l827_827173


namespace problem_statement_l827_827406

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l827_827406


namespace line_equation_curve_equation_distance_AB_l827_827527

-- Define parametric line equations and polar curve
def line_x (t : ℝ) : ℝ := -1 + 3 * t
def line_y (t : ℝ) : ℝ := 2 - 4 * t

def polar_rho (θ : ℝ) : ℝ := 2 * sqrt 2 * cos (θ - π / 4)

-- General equation of line l (parametric to cartesian)
theorem line_equation : ∀ t : ℝ, 4 * (line_x t) + 3 * (line_y t) - 2 = 0 := 
by
  intro t
  sorry

-- Rectangular coordinate equation of curve C
theorem curve_equation : ∀ θ : ℝ, 
  ∃ ρ : ℝ, ρ = polar_rho θ → ρ^2 = 2 * ρ * cos θ + 2 * ρ * sin θ → (sqrt (ρ^2))^2 + (sqrt (2 * ρ * cos θ))^2 + (sqrt (2 * ρ * sin θ))^2 - 2 * sqrt (2 * ρ * cos θ) - 2 * sqrt (2 * ρ * sin θ) = 0 := 
by 
  intro θ
  sorry

-- Distance |AB| on the intersection points
theorem distance_AB : let t1 := -3 in let t2 := -1 in abs(t1 - t2) = 2 := 
by
  intros t1 t2
  sorry

end line_equation_curve_equation_distance_AB_l827_827527


namespace p_is_composite_p_multiple_of_2015_pow_33_l827_827644

noncomputable theory

open Set

def S : Set ℕ := {n | 2 ≤ n ∧ n ≤ 2015}

def partition (S1 S2 : Set ℕ) := S1 ∪ S2 = S ∧ S1 ∩ S2 = ∅ ∧ S.card = 2014 ∧ S1.card = S2.card

def product (S : Set ℕ) := S.prod id

def characteristic_number (S1 S2 : Set ℕ) := product S1 + product S2

theorem p_is_composite (S1 S2 : Set ℕ) (h : partition S1 S2) : Composite (characteristic_number S1 S2) := sorry

theorem p_multiple_of_2015_pow_33 (S1 S2 : Set ℕ) (h : partition S1 S2) (h2 : ¬(2 ∣ characteristic_number S1 S2)) :
  2015^33 ∣ characteristic_number S1 S2 := sorry

end p_is_composite_p_multiple_of_2015_pow_33_l827_827644


namespace pure_imaginary_solution_l827_827551

theorem pure_imaginary_solution (b : ℝ) (z : ℂ) 
  (H : z = (b + Complex.I) / (2 + Complex.I))
  (H_imaginary : z.im = z ∧ z.re = 0) :
  b = -1 / 2 := 
by 
  sorry

end pure_imaginary_solution_l827_827551


namespace overall_financial_outcome_l827_827355

theorem overall_financial_outcome
  (house_selling_price : ℝ) (house_loss_percentage : ℝ) 
  (store_selling_price : ℝ) (store_gain_percentage : ℝ) 
  (apartment_selling_price : ℝ) (apartment_gain_percentage : ℝ) 
  (total_selling_price total_cost_price : ℝ) :
  house_selling_price = 15000 →
  house_loss_percentage = 0.15 →
  store_selling_price = 18000 →
  store_gain_percentage = 0.20 →
  apartment_selling_price = 10000 →
  apartment_gain_percentage = 0.10 →
  total_selling_price = house_selling_price + store_selling_price + apartment_selling_price →
  let h := house_selling_price / (1 - house_loss_percentage) in
  let s := store_selling_price / (1 + store_gain_percentage) in
  let a := apartment_selling_price / (1 + apartment_gain_percentage) in
  total_cost_price = h + s + a →
  total_cost_price < total_selling_price →
  total_selling_price - total_cost_price = 1262.03 :=
sorry

end overall_financial_outcome_l827_827355


namespace stack_logs_sum_l827_827414

theorem stack_logs_sum :
  let bottom_row := 15
  let rows := List.range bottom_row
  let logs_per_row := λ n, bottom_row - n
  rows.sum logs_per_row = 120 := by
sorry

end stack_logs_sum_l827_827414


namespace sum_primes_10_to_20_l827_827294

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827294


namespace sum_primes_between_10_and_20_is_60_l827_827237

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem sum_primes_between_10_and_20_is_60 :
  (∑ p in { n | n > 10 ∧ n < 20 ∧ is_prime n }.to_finset, p) = 60 := by
  sorry

end sum_primes_between_10_and_20_is_60_l827_827237


namespace appetizer_cost_l827_827869

def cost_per_person : ℝ :=
  (3 * 1 + 5 + 73) / 3

theorem appetizer_cost : cost_per_person = 27 := by
  unfold cost_per_person
  norm_num
  exact eq.refl 27

end appetizer_cost_l827_827869


namespace total_players_in_tournament_players_count_l827_827051

noncomputable def total_points (n : ℕ) : ℕ := 2 * (n * (n - 1)) / 2 + 56

noncomputable def total_games (n : ℕ) : ℕ := ((n + 8) * (n + 7)) / 2

theorem total_players_in_tournament
  (n : ℕ)
  (h : 2 * (n * (n - 1)) / 2 + 56 = ((n + 8) * (n + 7)) / 2) :
  n = 14 ∨ n = 2 := by
  have h₁ : (2 * n * (n - 1) / 2 + 56) = ((n + 8) * (n + 7) / 2) := h
  sorry
  
theorem players_count
  : ∃ n : ℕ, let total := n + 8 in total = 22 :=
    exists.intro 14 (rfl)  -- since 14 + 8 = 22

end total_players_in_tournament_players_count_l827_827051


namespace disk_division_max_areas_l827_827346

theorem disk_division_max_areas (n : ℕ) : 
  ∀ (radii secants : ℕ), radii = 3 * n → secants = 2 → 
  (max_areas_in_disk radii secants) = 4 + 6 * n :=
  begin
    intros radii secants h_radii h_secants,
    sorry
  end

end disk_division_max_areas_l827_827346


namespace smallest_possible_value_other_integer_l827_827672

theorem smallest_possible_value_other_integer (x : ℕ) (n : ℕ) (h_pos : x > 0)
  (h_gcd : ∃ m, Nat.gcd m n = x + 3 ∧ m = 30) 
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) :
  n = 162 := 
by sorry

end smallest_possible_value_other_integer_l827_827672


namespace sum_of_primes_between_10_and_20_is_60_l827_827274

/-- Define prime numbers between 10 and 20 -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- List the prime numbers between 10 and 20 -/
def primes_between_10_and_20 : List ℕ :=
  List.filter is_prime [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

/-- Compute the sum of a list of natural numbers -/
def sum_primes_between_10_and_20 :=
  List.sum primes_between_10_and_20

/-- Theorem stating that the sum of all prime numbers between 10 and 20 is 60 -/
theorem sum_of_primes_between_10_and_20_is_60 : sum_primes_between_10_and_20 = 60 :=
  sorry

end sum_of_primes_between_10_and_20_is_60_l827_827274


namespace sum_of_primes_between_10_and_20_l827_827261

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827261


namespace max_coins_on_board_l827_827027

def maxCoins (d : Fin 10 → ℕ) : ℕ :=
  Fin.sum (Fin 10) d

theorem max_coins_on_board :
  ∃ d : Fin 10 → ℕ, 
    (∀ i, d i ≤ 10) ∧
    (Fin.sum (Fin 10) (λ i, d i * (d i - 1))) / 2 ≤ 45 ∧
    maxCoins d = 34 :=
begin
  sorry

end max_coins_on_board_l827_827027


namespace parallelogram_height_l827_827862

theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 336) 
  (h_base : base = 14) 
  (h_formula : area = base * height) : 
  height = 24 := 
by 
  sorry

end parallelogram_height_l827_827862


namespace sum_primes_10_20_l827_827223

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_prime_in_range (a b : ℕ) : ℕ :=
  (list.filter is_prime (list.range (b + 1))).filter (λ x, a < x ∧ x < b).sum

theorem sum_primes_10_20 :
  sum_prime_in_range 10 20 = 60 :=
by
  sorry

end sum_primes_10_20_l827_827223


namespace calculate_stored_bales_l827_827699

theorem calculate_stored_bales : 
  ∀ (initial_bales current_bales stored_bales : ℕ), 
    initial_bales = 73 →
    current_bales = 96 →
    stored_bales = current_bales - initial_bales →
    stored_bales = 23 :=
by
  intros initial_bales current_bales stored_bales h_initial h_current h_calc
  rw [h_calc, h_initial, h_current]
  norm_num
  sorry

end calculate_stored_bales_l827_827699


namespace students_at_end_of_year_l827_827327

axiom initialStudents : Nat := 4
axiom studentsLeft : Nat := 3
axiom newStudents : Nat := 42

theorem students_at_end_of_year : initialStudents - studentsLeft + newStudents = 43 := 
by
  sorry

end students_at_end_of_year_l827_827327


namespace log_expression_eval_l827_827329

theorem log_expression_eval :
  log 2 6 - log 2 3 - 3 ^ (log 3 (1 / 2)) + (1 / 4) ^ (-1 / 2) = 5 / 2 :=
by
  sorry

end log_expression_eval_l827_827329


namespace infinitely_many_positive_integers_in_sequence_l827_827176

open Function

theorem infinitely_many_positive_integers_in_sequence
  (a : ℕ → ℕ) 
  (ha1 : a 1 = 0) 
  (hrec : ∀ n : ℕ, n ≥ 1 → (n + 1) ^ 3 * a (n + 1) = 2 * n ^ 2 * (2 * n + 1) * a n + 2 * (3 * n + 1)) 
  (hbinom : ∀ p : ℕ, Nat.Prime p → p ^ 2 ∣ Nat.choose (2 * p) p - 2) :
  ∃ infinitely_many n, a n ∈ ℕ ∧ a n > 0 := by
  sorry

end infinitely_many_positive_integers_in_sequence_l827_827176


namespace determinant_roots_l827_827092

theorem determinant_roots
  (a b c p q : ℝ)
  (h1 : a + b + c = 2)
  (h2 : ab + bc + ca = p)
  (h3 : abc = -q) :
  let D := Matrix.det (Matrix.of ![![a, b, c], ![b, c, a], ![c, a, b]]) 
  in D = -p - 8 := by
  sorry

end determinant_roots_l827_827092


namespace multiple_within_interval_l827_827640

theorem multiple_within_interval (k : ℕ) (h : k > 0) : 
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ k^4 ∧ m % k = 0 ∧ (m.digits 10).nodup ∧ (m.digits 10).length ≤ 4 :=
by
  sorry

end multiple_within_interval_l827_827640


namespace sarah_trip_total_distance_l827_827643

/-- Sarah's trip conditions and total distance -/
theorem sarah_trip_total_distance :
  ∀ (D : ℝ), 
    let t₁ := 15 in 
    let t₂ := 1 in 
    let v₁ := 15 in 
    let v₂ := 60 in 
    let arrival_time_diff := 1.5 in
    let driven_at_v₁ := v₁ * t₂ in 
    let total_driven_distance := driven_at_v₁ + D in 
    15 = D / ((D / v₂) + arrival_time_diff) → D = 30 → total_driven_distance = 45 :=
by
  intro D
  intros _ t₁ t₂ v₁ v₂ arrival_time_diff driven_at_v₁ total_driven_distance condition_D_is_valid 
    explicit_distance_is_30
  sorry

end sarah_trip_total_distance_l827_827643


namespace probability_of_qualified_shirt_number_of_defective_shirts_l827_827753

-- Definitions based on conditions
def number_of_shirts_inspected := [50, 100, 200, 500, 800, 1000]
def frequency_of_qualified_shirts := [47, 95, 188, 480, 763, 949]
def frequency_rate_of_qualified_shirts := [0.94, 0.95, 0.94, 0.96, 0.95, 0.95]

-- Given total shirts sold
def total_shirts_sold := 2000

-- Proof problem statements
theorem probability_of_qualified_shirt : 
  (List.sum frequency_rate_of_qualified_shirts / frequency_rate_of_qualified_shirts.length) = 0.95 :=
sorry

theorem number_of_defective_shirts : 
  total_shirts_sold * (1 - 0.95) = 100 :=
sorry

end probability_of_qualified_shirt_number_of_defective_shirts_l827_827753


namespace number_when_added_by_5_is_30_l827_827147

theorem number_when_added_by_5_is_30 (x: ℕ) (h: x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end number_when_added_by_5_is_30_l827_827147


namespace range_eccentricity_l827_827917

-- Definitions to capture the given conditions
variables (a b m n : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : 0 < m) (h4 : 0 < n)
def e_1 := sqrt (1 - (b^2 / a^2))
def e_2 := sqrt (1 + (n^2 / m^2))

-- The proof statement
theorem range_eccentricity (h_ell : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) → True) 
                           (h_hyp : ∀ x y : ℝ, (x^2 / m^2 - y^2 / n^2 = 1) → True)
                           (P F1 F2 : ℝ → ℝ → Prop) 
                           (h_PF1F2 : P (x_1) (y_1) → F1 (x_1) (y_1) → F2 (x_1) (y_1) → 
                                      ((F2 (x_2) + F2 (x_1)) * F1 (x_2) = 0)) :
  ∃ x y : ℝ, ∀ (P (x) (y) (h_p1 : x > 0 > y) (h_p2 : P (x) = true), 
  (4 + e_1 * e_2) / (2 * e_1) ≥ 6 := by
sorry

end range_eccentricity_l827_827917


namespace least_value_expression_l827_827549

theorem least_value_expression (x : ℝ) (h : x < -2) :
  2 * x < x ∧ 2 * x < x + 2 ∧ 2 * x < (1 / 2) * x ∧ 2 * x < x - 2 :=
by
  sorry

end least_value_expression_l827_827549


namespace part1_part2_l827_827607

def f (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x + 5)

theorem part1 : ∀ x, f x < 10 ↔ (x > -19 / 3 ∧ x ≤ -5) ∨ (-5 < x ∧ x < -1) :=
  sorry

theorem part2 (a b x : ℝ) (ha : abs a < 3) (hb : abs b < 3) :
  abs (a + b) + abs (a - b) < f x :=
  sorry

end part1_part2_l827_827607


namespace possible_lengths_of_CD_l827_827826

/-- Theorem stating the possible lengths of CD in the tetrahedron inscribed in a cylinder. -/
theorem possible_lengths_of_CD (A B C D : Point)
  (hAB : distance A B = 4) 
  (hAC : distance A C = 5)
  (hCB : distance C B = 5)
  (hAD : distance A D = 7)
  (hBD : distance B D = 7)
  (inscribed_in_cylinder : inscribed_in_cylinder ABCD)
  (CD_parallel_axis : parallel CD (axis cylinder)) :
  distance C D = sqrt 41 + sqrt 17 ∨ distance C D = sqrt 41 - sqrt 17 :=
sorry

end possible_lengths_of_CD_l827_827826


namespace sum_of_primes_between_10_and_20_l827_827308

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primesInRange (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

theorem sum_of_primes_between_10_and_20 :
  (primesInRange 10 20).sum = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827308


namespace sum_primes_10_to_20_l827_827292

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_10_to_20 : 
  (11 + 13 + 17 + 19 = 60) :=
by
  have h11 : is_prime 11 := sorry
  have h13 : is_prime 13 := sorry
  have h17 : is_prime 17 := sorry
  have h19 : is_prime 19 := sorry
  have h12 : ¬ is_prime 12 := sorry
  have h14 : ¬ is_prime 14 := sorry
  have h15 : ¬ is_prime 15 := sorry
  have h16 : ¬ is_prime 16 := sorry
  have h18 : ¬ is_prime 18 := sorry
  have h20 : ¬ is_prime 20 := sorry
  show 11 + 13 + 17 + 19 = 60, from sorry

end sum_primes_10_to_20_l827_827292


namespace largest_power_of_2_l827_827608

noncomputable def p : ℝ := ∑ k in finset.range 8, (k + 1) * Real.log (k + 1)

theorem largest_power_of_2 (h : p = ∑ k in finset.range 8, (k + 1) * Real.log (k + 1)) :
  (2 : ℝ) ^ 40 ∣ Real.exp p :=
sorry

end largest_power_of_2_l827_827608


namespace mans_speed_upstream_l827_827777

-- Define the conditions
def V_downstream : ℝ := 15  -- Speed with the current (downstream)
def V_current : ℝ := 2.5    -- Speed of the current

-- Calculate the man's speed against the current (upstream)
theorem mans_speed_upstream : V_downstream - 2 * V_current = 10 :=
by
  sorry

end mans_speed_upstream_l827_827777


namespace find_m_real_l827_827926

noncomputable def complex_number (m : ℝ) : ℂ :=
  ((1 : ℂ) + complex.I) / ((1 : ℂ) - complex.I) +
  m * (((1 : ℂ) - complex.I) / ((1 : ℂ) + complex.I))

theorem find_m_real (m : ℝ) (h : ∀ z : ℂ, z = complex_number m → z.im = 0) :
  m = 1 :=
sorry

end find_m_real_l827_827926


namespace axis_of_symmetry_is_1_l827_827870

-- Define the quadratic function and its components
def quadratic_fun : ℝ → ℝ := λ x, -2 * (x - 1)^2 + 3

-- Prove that the axis of symmetry is x = 1
theorem axis_of_symmetry_is_1 : ∀ x : ℝ, (∃ a b c : ℝ, quadratic_fun x = a * (x - b)^2 + c ∧ b = 1) :=
by
  intros x a b c
  use [-2, 1, 3]
  split
  unfold quadratic_fun
  ring
  sorry

end axis_of_symmetry_is_1_l827_827870


namespace chocolates_cost_l827_827755

-- Define the conditions given in the problem.
def boxes_needed (candies_total : ℕ) (candies_per_box : ℕ) : ℕ := 
    candies_total / candies_per_box

def total_cost_without_discount (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := 
    num_boxes * cost_per_box

def discount (total_cost : ℕ) : ℕ := 
    total_cost * 10 / 100

def final_cost (total_cost : ℕ) (discount : ℕ) : ℕ :=
    total_cost - discount

-- Theorem stating the total cost of buying 660 chocolate after discount is $138.60
theorem chocolates_cost (candies_total : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : 
     candies_total = 660 ∧ candies_per_box = 30 ∧ cost_per_box = 7 → 
     final_cost (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box) 
          (discount (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box)) = 13860 := 
by 
    intros h
    let ⟨h1, h2, h3⟩ := h 
    sorry 

end chocolates_cost_l827_827755


namespace calculate_expr_l827_827401

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l827_827401


namespace time_to_eat_24_slices_l827_827624

def MrFast_rate : ℝ := 1 / 5
def MrSlow_rate : ℝ := 1 / 10
def MrSteady_rate : ℝ := 1 / 8

def combined_rate : ℝ := MrFast_rate + MrSlow_rate + MrSteady_rate

theorem time_to_eat_24_slices : 
  let T := 24 / combined_rate in 
  round T = 56 := 
by sorry

end time_to_eat_24_slices_l827_827624


namespace smallest_integer_l827_827675

theorem smallest_integer (x : ℕ) (n : ℕ) (h_pos : 0 < x)
  (h_gcd : Nat.gcd 30 n = x + 3)
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) : n = 70 :=
begin
  sorry
end

end smallest_integer_l827_827675


namespace percentage_of_males_l827_827056

noncomputable def total_employees := 5200
noncomputable def males_below_50 := 1170

theorem percentage_of_males (P : ℕ) :
  let males := P * total_employees / 100 in
  let males_below_50_percent := 50 / 100 in
  males_below_50 = males_below_50_percent * males →
  P = 45 := 
sorry

end percentage_of_males_l827_827056


namespace four_letter_words_with_E_count_l827_827025

open Finset

/-- Number of 4-letter words from alphabet {A, B, C, D, E} with at least one E --/
theorem four_letter_words_with_E_count :
  let alphabet := {A, B, C, D, E}
      total_words := (Finset.card alphabet) ^ 4,
      words_without_E := (Finset.card (alphabet \ {'E'})) ^ 4,
      words_with_at_least_one_E := total_words - words_without_E in
  words_with_at_least_one_E = 369 :=
by
  let alphabet := {A, B, C, D, E}
  let total_words := (Finset.card alphabet) ^ 4
  let words_without_E := (Finset.card (alphabet \ {'E'})) ^ 4
  let words_with_at_least_one_E := total_words - words_without_E
  have h_total_words : total_words = 625 := by sorry
  have h_words_without_E : words_without_E = 256 := by sorry
  have h : words_with_at_least_one_E = 369 := by sorry
  exact h

end four_letter_words_with_E_count_l827_827025


namespace magnitude_of_vector_sum_l827_827951

theorem magnitude_of_vector_sum (m : ℝ) (h : (4 : ℝ) * (1 : ℝ) + m * (-2) = 0) :
  let a : ℝ × ℝ := (4, m),
      b : ℝ × ℝ := (1, -2),
      c : ℝ × ℝ := (a.1 + 2 * b.1, a.2 + 2 * b.2) in
  |c| = 2 * real.sqrt 10 :=
by
  sorry

end magnitude_of_vector_sum_l827_827951


namespace equality_sum_fibonacci_l827_827642

-- Definitions to match the conditions
def C (n k : ℕ) : ℕ := Nat.binomial n k
def S (n : ℕ) : ℕ := ∑ k in Finset.range (n+1), C(n - k) k
def F : ℕ → ℕ 
| 0 => 0
| 1 => 1
| n + 2 => F (n + 1) + F n

-- The theorem statement matching the question
theorem equality_sum_fibonacci (n : ℕ) : S n = F (n + 1) :=
sorry

end equality_sum_fibonacci_l827_827642


namespace find_number_l827_827746

-- Define the conditions: 0.80 * x - 20 = 60
variables (x : ℝ)
axiom condition : 0.80 * x - 20 = 60

-- State the theorem that x = 100 given the condition
theorem find_number : x = 100 :=
by
  sorry

end find_number_l827_827746


namespace sum_of_primes_between_10_and_20_l827_827266

theorem sum_of_primes_between_10_and_20 : 
  (∑ p in {n ∈ Finset.range 21 | 10 < n ∧ n.Prime}, p) = 60 := by
  sorry

end sum_of_primes_between_10_and_20_l827_827266


namespace least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2_l827_827209

theorem least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2 :
  nat.min_fac (11^5 - 11^4) = 2 :=
by
  sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_4_is_2_l827_827209


namespace five_pq_is_odd_l827_827150

theorem five_pq_is_odd (p q : ℤ) (hp : odd p) (hq : odd q) : odd (5 * p * q) :=
sorry

end five_pq_is_odd_l827_827150
